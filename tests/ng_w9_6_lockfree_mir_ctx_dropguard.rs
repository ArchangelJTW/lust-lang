#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! NG-W9-6: Lock-free single-threaded `MirContext` (`LocalCell`/`LocalLock`, `dirty_allocations` /
//! `dirty_allocations_count`) and transactional `core::mem::DropGuard` OOM rollback + one-shot guards.

use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use lust::bytecode::{FlatArrayBuffer, FlatArrayElemTag, LustArray, Value};
use lust::mir::cranelift_backend::{
    safepoint_reset, take_current_sticky_error, take_deopt_payload,
};
use lust::mir::ir::MirType;
use lust::mir::runtime_helpers::{
    LocalCell, LocalLock, MIR_STATUS_ERR_OOM, MIR_STATUS_OK, MirContext, UNBOUNDED_GAS_CHUNK,
    enter_mir_context, lust_mir_alloc_array_typed, lust_mir_array_get, lust_mir_array_set,
    set_unbounded_gas_chunk_override, unbounded_gas_chunk, with_current_mir_context,
    MIR_ARRAY_ELEM_TAG_INT,
};
use lust::stdlib::coord::semaphore::Semaphore as CoordSemaphore;
use lust::stdlib::sync::Semaphore as SyncSemaphore;
use lust::vfs::VirtualConsole;
use lust::vm::cranelift_dispatch::build_or_get_cached_flat_array;
use lust::vm::{VM, reset_effect_handler_state};
use lust::{EmbeddedModule, LustConfig, compile_program_with_config, load_program_from_embedded};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn reset_all_state() {
    safepoint_reset();
    reset_effect_handler_state();
    take_deopt_payload();
    take_current_sticky_error();
}

fn compile_embedded_vm(source: &str, use_cranelift: bool) -> VM {
    let entries = &[EmbeddedModule {
        module: "main",
        parent: None,
        source: Some(source),
    }];
    let program = load_program_from_embedded(entries, "main").expect("load embedded program");
    let mut cfg = LustConfig::default().with_cranelift_jit(use_cranelift);
    cfg.set_use_mir(false);
    let mut vm = compile_program_with_config(program, &cfg).expect("compile embedded program");
    vm.console = VirtualConsole::captured();
    if use_cranelift {
        vm.compile_cranelift().expect("compile cranelift");
    }
    vm
}

fn find_fn_name(vm: &VM, suffix: &str) -> String {
    for candidate in [suffix.to_string(), format!("main.{suffix}")] {
        if vm.cranelift_compiled.contains_key(&candidate) {
            return candidate;
        }
    }
    vm.cranelift_compiled
        .keys()
        .find(|k| k.ends_with(suffix))
        .cloned()
        .unwrap_or_else(|| panic!("Function '{suffix}' not compiled in Cranelift"))
}

// Compile-time `!Send` and `!Sync` check on `MirContext` and `Arc<MirContext>`.
#[allow(dead_code)]
fn assert_mir_context_not_send_or_sync() {
    trait AmbiguousIfImpl<A> {
        fn check() {}
    }
    impl<T: ?Sized> AmbiguousIfImpl<()> for T {}
    impl<T: ?Sized + Send> AmbiguousIfImpl<u8> for T {}
    impl<T: ?Sized + Sync> AmbiguousIfImpl<u16> for T {}

    let _ = <MirContext as AmbiguousIfImpl<_>>::check;
    let _ = <Arc<MirContext> as AmbiguousIfImpl<_>>::check;
}

// ============================================================================
// 1. Scalar JIT call leaves dirty_allocations == false & count == 0
// ============================================================================

#[test]
fn test_ng_w9_6_scalar_jit_leaves_dirty_allocations_zero() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
function add_loop(n: int): int
    local i: int = 0
    local acc: int = 0
    while i < n do
        acc = acc + i
        i = i + 1
    end
    return acc
end

function make_sum(n: int): int
    local arr: Array<int> = [1, 2, 3]
    array.push(arr, n)
    local s: int = 0
    local i: int = 0
    while i < array.len(arr) do
        if arr[i] is Ok(v) then
            s = s + v
        end
        i = i + 1
    end
    return s
end
"#;

    let mut vm = compile_embedded_vm(src, true);
    let add_name = find_fn_name(&vm, "add_loop");
    let make_name = find_fn_name(&vm, "make_sum");

    assert!(!vm.mir_context.dirty_allocations());
    assert_eq!(vm.mir_context.dirty_allocations_count(), 0);

    // Run scalar JIT function repeatedly
    for _ in 0..10 {
        let res = vm.call(&add_name, vec![Value::Int(100)]).expect("scalar JIT call");
        assert_eq!(res, Value::Int(4950));
        assert!(
            !vm.mir_context.dirty_allocations(),
            "Scalar JIT call must leave dirty_allocations == false"
        );
        assert_eq!(
            vm.mir_context.dirty_allocations_count(),
            0,
            "Scalar JIT call must leave dirty_allocations_count == 0"
        );
    }

    // Now run an allocating JIT function: dirty_allocations_count must increase,
    // and post-call clear_to_baselines(0, 0) must reset dirty_allocations back to false.
    let res = vm.call(&make_name, vec![Value::Int(4)]).expect("allocating JIT call");
    assert_eq!(res, Value::Int(10));
    let count_after_alloc = vm.mir_context.dirty_allocations_count();
    assert!(
        count_after_alloc > 0,
        "Allocating JIT call must increment dirty_allocations_count"
    );
    assert!(
        !vm.mir_context.dirty_allocations(),
        "Top-level clear_to_baselines(0, 0) must reset dirty_allocations to false"
    );

    // Subsequent scalar JIT call must again avoid dirtying MirContext
    let res2 = vm.call(&add_name, vec![Value::Int(50)]).expect("post-alloc scalar JIT call");
    assert_eq!(res2, Value::Int(1225));
    assert!(!vm.mir_context.dirty_allocations());
    assert_eq!(
        vm.mir_context.dirty_allocations_count(),
        count_after_alloc,
        "Scalar JIT call after cleanup must not add dirty allocations"
    );
}

// ============================================================================
// 2. OOM rollback via core::mem::DropGuard in build_or_get_cached_flat_array
// ============================================================================

#[test]
fn test_ng_w9_6_flat_array_oom_and_mismatch_dropguard_rollback() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    // Case A: OOM rollback from memory_used == 0 -> rolls back to exact 0
    {
        let ctx = MirContext::new();
        ctx.set_max_memory_limit(64);
        ctx.set_memory_used(0);
        let arr = Rc::new(LustArray::new((0..64).map(Value::Int).collect()));
        let elems = arr.borrow();
        let err = build_or_get_cached_flat_array(&ctx, &arr, &elems, FlatArrayElemTag::Int)
            .expect_err("Oversized flat cache build must return OOM error");
        assert!(err.to_string().contains("Out of memory"), "got: {err}");
        assert_eq!(ctx.last_status(), MIR_STATUS_ERR_OOM);
        assert_eq!(
            ctx.memory_used(),
            0,
            "DropGuard must roll back memory_used to 0 on OOM"
        );
        assert!(!arr.has_cached_flat());
    }

    // Case B: OOM rollback from non-zero baseline -> rolls back to exact baseline
    {
        let ctx = MirContext::new();
        ctx.set_max_memory_limit(200);
        ctx.set_memory_used(128);
        let arr = Rc::new(LustArray::new((0..32).map(Value::Int).collect()));
        let elems = arr.borrow();
        let _ = build_or_get_cached_flat_array(&ctx, &arr, &elems, FlatArrayElemTag::Int)
            .expect_err("Exceeding remaining budget must return OOM error");
        assert_eq!(ctx.last_status(), MIR_STATUS_ERR_OOM);
        assert_eq!(
            ctx.memory_used(),
            128,
            "DropGuard must roll back memory_used to exact baseline on OOM"
        );
        assert!(!arr.has_cached_flat());
    }

    // Case C: Element type mismatch -> DropGuard refunds reserved bytes, returns Ok(None)
    {
        let ctx = MirContext::new();
        ctx.set_max_memory_limit(4096);
        ctx.set_memory_used(24);
        let mixed = Rc::new(LustArray::new(vec![Value::Int(1), Value::Bool(true)]));
        let elems = mixed.borrow();
        let res = build_or_get_cached_flat_array(&ctx, &mixed, &elems, FlatArrayElemTag::Int)
            .expect("Type mismatch returns Ok(None)");
        assert!(res.is_none());
        assert_eq!(ctx.memory_used(), 24);
        assert!(!mixed.has_cached_flat());
    }

    // Case D: Happy path -> DropGuard dismissed, memory remains charged & cache populated
    {
        let ctx = MirContext::new();
        ctx.set_max_memory_limit(4096);
        ctx.set_memory_used(0);
        let arr = Rc::new(LustArray::new(vec![Value::Int(10), Value::Int(20), Value::Int(30)]));
        let elems = arr.borrow();
        let (buf, hit) = build_or_get_cached_flat_array(&ctx, &arr, &elems, FlatArrayElemTag::Int)
            .expect("build must succeed")
            .expect("homogeneous int array must produce buffer");
        assert!(!hit);
        assert_eq!(buf.word_len(), 5);
        assert_eq!(ctx.memory_used(), 5 * 8);
        assert!(arr.has_cached_flat());
        assert_eq!(ctx.last_status(), MIR_STATUS_OK);

        // Cache hit charges 0 additional bytes
        let (_, hit2) = build_or_get_cached_flat_array(&ctx, &arr, &elems, FlatArrayElemTag::Int)
            .expect("hit must succeed")
            .expect("cached buffer");
        assert!(hit2);
        assert_eq!(ctx.memory_used(), 5 * 8);
    }
}

// ============================================================================
// 3. Reentrant enter_mir_context does NOT cause BorrowMutError
// ============================================================================

#[test]
fn test_ng_w9_6_reentrant_enter_mir_context_no_borrow_mut_error() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let ctx = MirContext::new();
    assert!(!ctx.dirty_allocations());
    assert_eq!(ctx.dirty_allocations_count(), 0);

    // Nested enter_mir_context on the SAME MirContext + reentrancy inside with_current_mir_context
    {
        let _g1 = enter_mir_context(&ctx);
        let h1 = ctx.register_handle(Value::string("outer"));
        let outer_arr = lust_mir_alloc_array_typed(2, MIR_ARRAY_ELEM_TAG_INT);
        assert!(!outer_arr.is_null());
        lust_mir_array_set(outer_arr, 0, 111);
        assert!(ctx.dirty_allocations());

        let inner_val = with_current_mir_context(|active| {
            // Reentrant enter_mir_context while inside with_current_mir_context!
            let _g2 = enter_mir_context(active);
            let b_inner = active.capture_baseline();

            // Register borrowed readonly array inside nested scope
            let flat = Rc::new(FlatArrayBuffer::new(vec![2, 2, 77, 88]));
            active
                .register_borrowed_readonly_array(flat.as_ptr(), flat.word_len(), MirType::Int)
                .expect("borrowed readonly registration must succeed");
            assert_eq!(lust_mir_array_get(flat.as_ptr(), 1), 88);

            // Nested inner allocation + another level of reentrancy
            {
                let _g3 = enter_mir_context(active);
                let _h3 = active.register_handle(Value::string("deep"));
                let deep_arr = lust_mir_alloc_array_typed(1, MIR_ARRAY_ELEM_TAG_INT);
                assert!(!deep_arr.is_null());
                lust_mir_array_set(deep_arr, 0, 999);
                assert_eq!(lust_mir_array_get(deep_arr, 0), 999);
            }

            active.clear_to_baselines(b_inner.handles, b_inner.buffers);
            active.get_handle(h1)
        });
        assert_eq!(inner_val, Some(Value::string("outer")));
        assert_eq!(lust_mir_array_get(outer_arr, 0), 111);
    }

    // Full cleanup after all guards dropped
    ctx.clear_to_baselines(0, 0);
    assert_eq!(ctx.handle_count(), 0);
    assert_eq!(ctx.buffer_count(), 0);
    assert_eq!(ctx.allocation_records.borrow().len(), 0);
    assert_eq!(ctx.borrowed_readonly_stack.borrow().len(), 0);
    assert!(!ctx.dirty_allocations());
    assert!(ctx.dirty_allocations_count() > 0);
}

// ============================================================================
// 4. LocalCell (i64 & usize) and LocalLock adapter semantics
// ============================================================================

#[test]
fn test_ng_w9_6_local_cell_and_local_lock_adapters() {
    let c_i64 = LocalCell::<i64>::new(10);
    assert_eq!(c_i64.get(), 10);
    assert_eq!(c_i64.load(Ordering::SeqCst), 10);
    assert_eq!(c_i64.fetch_add(5, Ordering::SeqCst), 10);
    assert_eq!(c_i64.get(), 15);
    assert_eq!(c_i64.fetch_sub(7, Ordering::Relaxed), 15);
    assert_eq!(c_i64.load(Ordering::Acquire), 8);
    c_i64.store(i64::MAX, Ordering::Release);
    assert_eq!(c_i64.fetch_add(1, Ordering::SeqCst), i64::MAX);
    assert_eq!(c_i64.get(), i64::MIN);

    let c_usize = LocalCell::<usize>::new(1);
    assert_eq!(c_usize.fetch_add(3, Ordering::SeqCst), 1);
    assert_eq!(c_usize.get(), 4);
    assert_eq!(c_usize.fetch_sub(2, Ordering::SeqCst), 4);
    assert_eq!(c_usize.load(Ordering::SeqCst), 2);
    c_usize.set(0);
    assert_eq!(c_usize.fetch_sub(1, Ordering::SeqCst), 0);
    assert_eq!(c_usize.get(), usize::MAX);

    let lock = LocalLock::new(vec![1usize, 2]);
    assert_eq!(lock.borrow().len(), 2);
    lock.borrow_mut().push(3);
    lock.lock().unwrap().push(4);
    assert_eq!(*lock.borrow(), vec![1, 2, 3, 4]);
}

// ============================================================================
// 5. One-shot DropGuard replacements (UnboundedGasChunkGuard, CycleGuards, PermitGuards)
// ============================================================================

#[test]
fn test_ng_w9_6_oneshot_dropguards_normal_and_unwind() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    // 5a. UnboundedGasChunkGuard restores on normal drop and panic unwind
    assert_eq!(unbounded_gas_chunk(), UNBOUNDED_GAS_CHUNK);
    {
        let _g1 = set_unbounded_gas_chunk_override(Some(1234));
        assert_eq!(unbounded_gas_chunk(), 1234);
        let unwind_res = catch_unwind(|| {
            let _g2 = set_unbounded_gas_chunk_override(Some(56));
            assert_eq!(unbounded_gas_chunk(), 56);
            panic!("intentional unwind");
        });
        assert!(unwind_res.is_err());
        assert_eq!(
            unbounded_gas_chunk(),
            1234,
            "Inner DropGuard must restore outer override on unwind"
        );
    }
    assert_eq!(unbounded_gas_chunk(), UNBOUNDED_GAS_CHUNK);

    // 5b. Cyclic Value Debug / Display / PartialEq DropGuards clean up visited sets
    let arr_a = Rc::new(LustArray::new(vec![]));
    arr_a.borrow_mut().push(Value::Array(Rc::clone(&arr_a)));
    let arr_b = Rc::new(LustArray::new(vec![]));
    arr_b.borrow_mut().push(Value::Array(Rc::clone(&arr_b)));

    let va = Value::Array(Rc::clone(&arr_a));
    let vb = Value::Array(Rc::clone(&arr_b));

    for _ in 0..3 {
        let dbg = format!("{va:?}");
        assert!(dbg.contains("[...]"), "cyclic debug should terminate: {dbg}");
        let disp = format!("{va}");
        assert!(disp.contains("[...]"), "cyclic display should terminate: {disp}");
        assert_eq!(va, vb, "structurally identical cyclic arrays must compare equal");
    }
    // Break cycles before drop so Rc leaks do not occur in test
    arr_a.borrow_mut().clear();
    arr_b.borrow_mut().clear();

    // 5c. stdlib::coord::semaphore::Semaphore::with_permits DropGuard
    let coord_sem = RefCell::new(CoordSemaphore::new(4));
    let ok_val = coord_sem
        .borrow_mut()
        .with_permits(3, || 42usize)
        .expect("permits available");
    assert_eq!(ok_val, 42);
    assert_eq!(coord_sem.borrow().available_permits(), 4);

    let panic_res = catch_unwind(AssertUnwindSafe(|| {
        let _ = coord_sem.borrow_mut().with_permits(3, || {
            panic!("unwind inside coord semaphore");
        });
    }));
    assert!(panic_res.is_err());
    assert_eq!(
        coord_sem.borrow().available_permits(),
        4,
        "CoordSemaphore DropGuard must return permits on unwind"
    );

    // 5d. stdlib::sync::Semaphore::with_permit DropGuard
    let sync_sem = RefCell::new(SyncSemaphore::new(5));
    let ok_sync = sync_sem
        .borrow_mut()
        .with_permit(2, |s| s.available_permits())
        .expect("permit available");
    assert_eq!(ok_sync, 3);
    assert_eq!(sync_sem.borrow().available_permits(), 5);

    let panic_sync = catch_unwind(AssertUnwindSafe(|| {
        let _ = sync_sem.borrow_mut().with_permit(4, |s| {
            assert_eq!(s.available_permits(), 1);
            panic!("unwind inside sync semaphore");
        });
    }));
    assert!(panic_sync.is_err());
    assert_eq!(
        sync_sem.borrow().available_permits(),
        5,
        "SyncSemaphore DropGuard must release permits on unwind"
    );
}

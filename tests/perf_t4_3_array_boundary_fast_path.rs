#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! PERF-T4-3: Compiler-Enforced `LustArray` Encapsulation & Borrowed Read-Only
//! Flat-View Boundary Fast-Path.

use std::rc::Rc;
use std::sync::Mutex;

use lust::bytecode::{FlatArrayBuffer, FlatArrayElemTag, LustArray, Value};
use lust::effect_abi::LUST_STATUS_ERR_GENERIC;
use lust::mir::cranelift_backend::{
    safepoint_reset, take_current_sticky_error, take_deopt_payload,
};
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::safepoint_request_deopt_on_poll;
use lust::mir::ir::MirType;
use lust::mir::runtime_helpers::{
    MIR_STATUS_ERR_OOM, MirContext, enter_mir_context, lust_mir_array_get, lust_mir_array_len,
    lust_mir_array_push, lust_mir_array_set,
};
use lust::vfs::VirtualConsole;
use lust::vm::cranelift_dispatch::{
    build_or_get_cached_flat_array, cranelift_writeback_commit_count,
    reset_cranelift_writeback_commit_count,
};
use lust::vm::{VM, reset_effect_handler_state};
use lust::{EmbeddedModule, LustConfig, compile_program_with_config, load_program_from_embedded};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn reset_all_state() {
    safepoint_reset();
    reset_effect_handler_state();
    take_deopt_payload();
    take_current_sticky_error();
    reset_cranelift_writeback_commit_count();
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
    let candidates = [suffix.to_string(), format!("main.{suffix}")];
    for c in candidates {
        if vm.cranelift_compiled.contains_key(&c) {
            return c;
        }
    }
    vm.cranelift_compiled
        .keys()
        .find(|k| k.ends_with(suffix))
        .cloned()
        .unwrap_or_else(|| panic!("Function '{suffix}' not compiled in Cranelift"))
}

// ============================================================================
// 1. CORE GREEN TEST: 100 Calls -> 1 Build + 99 Hits + Invalidation Parity
// ============================================================================

#[test]
fn test_perf_t4_3_core_int_array_100_calls_one_build_99_hits_and_invalidation() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
function count_even(arr: Array<int>): int
    local n: int = array.len(arr)
    local i: int = 0
    local c: int = 0
    while i < n do
        if arr[i] is Ok(v) then
            if v % 2 == 0 then
                c = c + 1
            end
        end
        i = i + 1
    end
    return c
end

function sum_arr(arr: Array<int>): int
    local n: int = array.len(arr)
    local i: int = 0
    local s: int = 0
    while i < n do
        if arr[i] is Ok(v) then
            s = s + v
        end
        i = i + 1
    end
    return s
end

function set_elem(arr: Array<int>, idx: int, v: int): int
    arr[idx] = v
    if arr[idx] is Ok(cur) then
        return cur
    end
    return -1
end

function push_elem(arr: Array<int>, v: int): int
    array.push(arr, v)
    return array.len(arr)
end

function identity_arr(arr: Array<int>): Array<int>
    return arr
end
"#;

    let mut vm = compile_embedded_vm(src, true);
    let count_even_name = find_fn_name(&vm, "count_even");
    let sum_arr_name = find_fn_name(&vm, "sum_arr");
    let set_elem_name = find_fn_name(&vm, "set_elem");
    let push_elem_name = find_fn_name(&vm, "push_elem");
    let identity_name = find_fn_name(&vm, "identity_arr");

    assert_eq!(vm.cranelift_readonly_params(&count_even_name), Some(vec![true]));
    assert_eq!(vm.cranelift_has_external_or_host_calls(&count_even_name), Some(false));
    assert_eq!(vm.cranelift_readonly_params(&sum_arr_name), Some(vec![true]));
    assert_eq!(vm.cranelift_has_external_or_host_calls(&sum_arr_name), Some(false));

    // 1000-element Int array: 0, 1, ..., 999 -> 500 even elements, sum = 499500
    let arr_rc = Rc::new(LustArray::new((0..1000).map(Value::Int).collect()));
    assert!(!arr_rc.has_cached_flat());

    let entry_start = vm.cranelift_entry_count();
    let builds_start = vm.cranelift_array_cache_builds();
    let hits_start = vm.cranelift_array_cache_hits();

    for _ in 0..100 {
        let res = vm
            .call(&count_even_name, vec![Value::Array(Rc::clone(&arr_rc))])
            .expect("count_even native call");
        assert_eq!(res, Value::Int(500));
    }

    assert_eq!(
        vm.cranelift_entry_count() - entry_start,
        100,
        "All 100 calls must execute natively via Cranelift"
    );
    assert_eq!(
        vm.cranelift_array_cache_builds() - builds_start,
        1,
        "Unchanged 1000-element Int array must build flat cache exactly ONCE"
    );
    assert_eq!(
        vm.cranelift_array_cache_hits() - hits_start,
        99,
        "Remaining 99 calls must hit the cached borrowed flat buffer in O(1)"
    );
    assert!(arr_rc.has_cached_flat());
    assert_eq!(
        cranelift_writeback_commit_count(),
        0,
        "Read-only borrowed calls must commit zero writebacks"
    );

    // Returning a borrowed read-only array preserves exact Rc identity and hits cache
    let id_res = vm
        .call(&identity_name, vec![Value::Array(Rc::clone(&arr_rc))])
        .expect("identity_arr");
    match id_res {
        Value::Array(ret_rc) => assert!(Rc::ptr_eq(&arr_rc, &ret_rc)),
        other => panic!("expected Value::Array, got {other:?}"),
    }
    assert_eq!(vm.cranelift_array_cache_builds() - builds_start, 1);
    assert_eq!(vm.cranelift_array_cache_hits() - hits_start, 100);

    // 1. Same-length overwrite via VM native call invalidates flat_cache immediately
    let set_res = vm
        .call(
            &set_elem_name,
            vec![Value::Array(Rc::clone(&arr_rc)), Value::Int(0), Value::Int(1)],
        )
        .expect("set_elem");
    assert_eq!(set_res, Value::Int(1));
    assert!(
        !arr_rc.has_cached_flat(),
        "Writeback commit via borrow_mut() must immediately invalidate flat_cache"
    );

    let after_overwrite = vm
        .call(&count_even_name, vec![Value::Array(Rc::clone(&arr_rc))])
        .expect("count_even after overwrite");
    assert_eq!(after_overwrite, Value::Int(499));
    assert_eq!(
        vm.cranelift_array_cache_builds() - builds_start,
        2,
        "First read-only call after overwrite must rebuild flat cache"
    );

    // 2. Growth via array.push invalidates flat_cache immediately
    let new_len = vm
        .call(
            &push_elem_name,
            vec![Value::Array(Rc::clone(&arr_rc)), Value::Int(2000)],
        )
        .expect("push_elem");
    assert_eq!(new_len, Value::Int(1001));
    assert!(!arr_rc.has_cached_flat());

    let after_push_sum = vm
        .call(&sum_arr_name, vec![Value::Array(Rc::clone(&arr_rc))])
        .expect("sum_arr after push");
    // Original sum 499500 + (1 - 0) + 2000 = 501501
    assert_eq!(after_push_sum, Value::Int(501_501));
    assert_eq!(vm.cranelift_array_cache_builds() - builds_start, 3);

    // 3. Mutation through a second Rc alias invalidates flat_cache immediately
    let alias_rc = Rc::clone(&arr_rc);
    alias_rc.borrow_mut()[1] = Value::Int(101); // +100 delta
    assert!(!arr_rc.has_cached_flat());
    let after_alias = vm
        .call(&sum_arr_name, vec![Value::Array(Rc::clone(&arr_rc))])
        .expect("sum_arr after alias mutation");
    assert_eq!(after_alias, Value::Int(501_601));
    assert_eq!(vm.cranelift_array_cache_builds() - builds_start, 4);
}

#[test]
fn test_perf_t4_3_float_bool_empty_and_readonly_duplicate_alias() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
function sum_float(arr: Array<float>): float
    local n: int = array.len(arr)
    local i: int = 0
    local s: float = 0.0
    while i < n do
        if arr[i] is Ok(v) then
            s = s + v
        end
        i = i + 1
    end
    return s
end

function count_true(arr: Array<bool>): int
    local n: int = array.len(arr)
    local i: int = 0
    local c: int = 0
    while i < n do
        if arr[i] is Ok(v) then
            if v then
                c = c + 1
            end
        end
        i = i + 1
    end
    return c
end

function sum_two_ro(a: Array<int>, b: Array<int>): int
    local n: int = array.len(a)
    local i: int = 0
    local s: int = 0
    while i < n do
        if a[i] is Ok(va) then
            if b[i] is Ok(vb) then
                s = s + va + vb
            end
        end
        i = i + 1
    end
    return s
end
"#;

    let mut vm = compile_embedded_vm(src, true);
    let sum_float_name = find_fn_name(&vm, "sum_float");
    let count_true_name = find_fn_name(&vm, "count_true");
    let sum_two_ro_name = find_fn_name(&vm, "sum_two_ro");

    // Float homogeneous array: 1 build + 19 hits
    let f_rc = Rc::new(LustArray::new(vec![
        Value::Float(1.5),
        Value::Float(2.25),
        Value::Float(3.75),
    ]));
    let b0 = vm.cranelift_array_cache_builds();
    let h0 = vm.cranelift_array_cache_hits();
    for _ in 0..20 {
        assert_eq!(
            vm.call(&sum_float_name, vec![Value::Array(Rc::clone(&f_rc))])
                .unwrap(),
            Value::Float(7.5)
        );
    }
    assert_eq!(vm.cranelift_array_cache_builds() - b0, 1);
    assert_eq!(vm.cranelift_array_cache_hits() - h0, 19);
    f_rc.borrow_mut()[0] = Value::Float(10.5);
    assert_eq!(
        vm.call(&sum_float_name, vec![Value::Array(Rc::clone(&f_rc))])
            .unwrap(),
        Value::Float(16.5)
    );
    assert_eq!(vm.cranelift_array_cache_builds() - b0, 2);

    // Bool homogeneous array: 1 build + 19 hits
    let bool_rc = Rc::new(LustArray::new(vec![
        Value::Bool(true),
        Value::Bool(false),
        Value::Bool(true),
        Value::Bool(true),
    ]));
    let b1 = vm.cranelift_array_cache_builds();
    let h1 = vm.cranelift_array_cache_hits();
    for _ in 0..20 {
        assert_eq!(
            vm.call(&count_true_name, vec![Value::Array(Rc::clone(&bool_rc))])
                .unwrap(),
            Value::Int(3)
        );
    }
    assert_eq!(vm.cranelift_array_cache_builds() - b1, 1);
    assert_eq!(vm.cranelift_array_cache_hits() - h1, 19);
    bool_rc.borrow_mut()[1] = Value::Bool(true);
    assert_eq!(
        vm.call(&count_true_name, vec![Value::Array(Rc::clone(&bool_rc))])
            .unwrap(),
        Value::Int(4)
    );
    assert_eq!(vm.cranelift_array_cache_builds() - b1, 2);

    // Empty array + duplicate read-only alias (a, a)
    let empty_rc = Rc::new(LustArray::new(Vec::new()));
    let b2 = vm.cranelift_array_cache_builds();
    let h2 = vm.cranelift_array_cache_hits();
    for _ in 0..10 {
        assert_eq!(
            vm.call(
                &sum_two_ro_name,
                vec![
                    Value::Array(Rc::clone(&empty_rc)),
                    Value::Array(Rc::clone(&empty_rc)),
                ],
            )
            .unwrap(),
            Value::Int(0)
        );
    }
    assert_eq!(vm.cranelift_array_cache_builds() - b2, 1);
    assert_eq!(vm.cranelift_array_cache_hits() - h2, 9);

    let pair_rc = Rc::new(LustArray::new(vec![Value::Int(10), Value::Int(20)]));
    let b3 = vm.cranelift_array_cache_builds();
    let h3 = vm.cranelift_array_cache_hits();
    for _ in 0..5 {
        assert_eq!(
            vm.call(
                &sum_two_ro_name,
                vec![
                    Value::Array(Rc::clone(&pair_rc)),
                    Value::Array(Rc::clone(&pair_rc)),
                ],
            )
            .unwrap(),
            Value::Int(60)
        );
    }
    assert_eq!(vm.cranelift_array_cache_builds() - b3, 1);
    assert_eq!(vm.cranelift_array_cache_hits() - h3, 4);
}

// ============================================================================
// 2. ADVERSARIAL NEGATIVE TESTS (ALL 4 MANDATORY!)
// ============================================================================

#[test]
fn test_perf_t4_3_adv1_aliased_readonly_and_mutable_params_disables_borrowed_path() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
function ro_rw_alias(ro: Array<int>, rw: Array<int>): int
    rw[0] = 42
    if ro[0] is Ok(v) then
        return v
    end
    return -1
end

function sum_ro(arr: Array<int>): int
    local s: int = 0
    if arr[0] is Ok(v0) then
        s = s + v0
    end
    if arr[1] is Ok(v1) then
        s = s + v1
    end
    return s
end
"#;

    let mut vm = compile_embedded_vm(src, true);
    let ro_rw_name = find_fn_name(&vm, "ro_rw_alias");
    let sum_ro_name = find_fn_name(&vm, "sum_ro");

    assert_eq!(
        vm.cranelift_readonly_params(&ro_rw_name),
        Some(vec![true, false]),
        "Param 0 must be readonly=true and param 1 readonly=false"
    );

    let a_rc = Rc::new(LustArray::new(vec![Value::Int(5), Value::Int(10)]));

    // Warm flat_cache on `a_rc` first via read-only call
    assert_eq!(
        vm.call(&sum_ro_name, vec![Value::Array(Rc::clone(&a_rc))])
            .unwrap(),
        Value::Int(15)
    );
    assert!(a_rc.has_cached_flat());
    assert_eq!(vm.cranelift_array_cache_builds(), 1);
    assert_eq!(vm.cranelift_array_cache_hits(), 0);

    // Now pass the SAME Rc<LustArray> as (a, a) to (ro, rw):
    // MUST NOT take borrowed read-only path, MUST observe 42 via alias, MUST commit 42 & invalidate cache!
    let builds_before = vm.cranelift_array_cache_builds();
    let hits_before = vm.cranelift_array_cache_hits();
    let res = vm
        .call(
            &ro_rw_name,
            vec![Value::Array(Rc::clone(&a_rc)), Value::Array(Rc::clone(&a_rc))],
        )
        .expect("ro_rw_alias(a, a)");
    assert_eq!(res, Value::Int(42), "In-call alias write rw[0]=42 must be visible via ro[0]");
    assert_eq!(a_rc.borrow()[0], Value::Int(42), "Writeback must commit 42");
    assert_eq!(
        vm.cranelift_array_cache_builds(),
        builds_before,
        "Aliased (ro, rw) call must NOT build borrowed cache"
    );
    assert_eq!(
        vm.cranelift_array_cache_hits(),
        hits_before,
        "Aliased (ro, rw) call must NOT hit borrowed cache"
    );
    assert!(
        !a_rc.has_cached_flat(),
        "Writeback commit must invalidate stale flat_cache"
    );

    // Subsequent read-only call must rebuild cache and observe 42 + 10 = 52
    assert_eq!(
        vm.call(&sum_ro_name, vec![Value::Array(Rc::clone(&a_rc))])
            .unwrap(),
        Value::Int(52)
    );
    assert_eq!(vm.cranelift_array_cache_builds(), builds_before + 1);
}

#[test]
fn test_perf_t4_3_adv2_external_rust_borrow_mut_same_length_overwrite_invalidates_immediately() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
function sum_arr(arr: Array<int>): int
    local n: int = array.len(arr)
    local i: int = 0
    local s: int = 0
    while i < n do
        if arr[i] is Ok(v) then
            s = s + v
        end
        i = i + 1
    end
    return s
end
"#;

    let mut vm = compile_embedded_vm(src, true);
    let sum_arr_name = find_fn_name(&vm, "sum_arr");

    let arr_rc = Rc::new(LustArray::new(vec![
        Value::Int(1),
        Value::Int(2),
        Value::Int(3),
    ]));

    assert_eq!(
        vm.call(&sum_arr_name, vec![Value::Array(Rc::clone(&arr_rc))])
            .unwrap(),
        Value::Int(6)
    );
    assert!(arr_rc.has_cached_flat());
    let gen_before = arr_rc.generation();

    // Failed try_borrow_mut while shared borrow is active MUST NOT bump generation or clear cache
    {
        let _shared = arr_rc.borrow();
        assert!(arr_rc.try_borrow_mut().is_err());
        assert_eq!(arr_rc.generation(), gen_before);
        assert!(arr_rc.has_cached_flat());
    }

    // External Rust same-length overwrite via borrow_mut() OUTSIDE any VM opcode!
    {
        let mut guard = arr_rc.borrow_mut();
        assert!(
            !arr_rc.has_cached_flat(),
            "flat_cache must be cleared BEFORE caller even writes through RefMut"
        );
        assert!(arr_rc.generation() > gen_before);
        guard[0] = Value::Int(999);
    }

    // Next native call MUST rebuild cache and observe 999 + 2 + 3 = 1004
    let res = vm
        .call(&sum_arr_name, vec![Value::Array(Rc::clone(&arr_rc))])
        .unwrap();
    assert_eq!(res, Value::Int(1004));
    assert_eq!(vm.cranelift_array_cache_builds(), 2);

    // Same via try_borrow_mut()
    let gen_mid = arr_rc.generation();
    {
        let mut guard = arr_rc.try_borrow_mut().expect("try_borrow_mut ok");
        assert!(!arr_rc.has_cached_flat());
        assert!(arr_rc.generation() > gen_mid);
        guard[2] = Value::Int(100);
    }
    let res2 = vm
        .call(&sum_arr_name, vec![Value::Array(Rc::clone(&arr_rc))])
        .unwrap();
    assert_eq!(res2, Value::Int(1101));
    assert_eq!(vm.cranelift_array_cache_builds(), 3);
}

#[test]
fn test_perf_t4_3_adv3_active_mutable_borrow_held_across_native_call_fails_closed() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
function sum_arr(arr: Array<int>): int
    local n: int = array.len(arr)
    local i: int = 0
    local s: int = 0
    while i < n do
        if arr[i] is Ok(v) then
            s = s + v
        end
        i = i + 1
    end
    return s
end
"#;

    let mut vm = compile_embedded_vm(src, true);
    let sum_arr_name = find_fn_name(&vm, "sum_arr");

    let arr_rc = Rc::new(LustArray::new(vec![Value::Int(10), Value::Int(20)]));
    // Warm cache first
    assert_eq!(
        vm.call(&sum_arr_name, vec![Value::Array(Rc::clone(&arr_rc))])
            .unwrap(),
        Value::Int(30)
    );

    // Hold active mutable borrow across native call -> MUST fail closed via try_borrow() without panic/UB
    {
        let _active_mut = arr_rc.borrow_mut();
        let err = vm
            .call(&sum_arr_name, vec![Value::Array(Rc::clone(&arr_rc))])
            .expect_err("Active mutable borrow across native call must fail closed");
        assert!(
            err.to_string().contains("borrow"),
            "Expected borrow error, got: {err}"
        );
        assert_eq!(
            vm.call_stack_depth(),
            0,
            "Call stack must unwind cleanly on pre-entry borrow rejection"
        );
    }

    // After releasing mutable borrow, VM recovers cleanly and runs natively
    assert_eq!(
        vm.call(&sum_arr_name, vec![Value::Array(Rc::clone(&arr_rc))])
            .unwrap(),
        Value::Int(30)
    );

    // Also directly exercise `try_run_cranelift_top_frame_scoped`'s pre-entry `_borrowed_array_guards`
    // `try_borrow()` check using a MIR-loaded function (no bytecode signature pre-check).
    {
        use lust::intern::Interner;
        use lust::lexer::Lexer;
        use lust::mir::ir::MirModule;
        use lust::mir::lowering::AstLowering;
        use lust::parser::Parser;

        let raw_src = r#"
function mir_sum_arr(arr: Array<int>): int
    local n: int = array.len(arr)
    local i: int = 0
    local s: int = 0
    while i < n do
        s = s + arr[i]
        i = i + 1
    end
    return s
end
"#;
        let mut interner = Interner::new();
        let tokens = Lexer::new(raw_src, &mut interner).tokenize().unwrap();
        let items = Parser::new(tokens).parse().unwrap();
        let mut module = MirModule::new("adv3_direct".into());
        for item in &items {
            if let lust::ast::ItemKind::Function(f) = &item.kind {
                module.add_function(AstLowering::lower_function(f).unwrap());
            }
        }
        let mut mir_vm = VM::new();
        mir_vm.set_backend(lust::vm::VmBackend::Cranelift);
        mir_vm.load_mir_module(&module).unwrap();
        mir_vm.compile_cranelift().unwrap();

        let arr2 = Rc::new(LustArray::new(vec![Value::Int(4), Value::Int(6)]));
        assert_eq!(
            mir_vm
                .call("mir_sum_arr", vec![Value::Array(Rc::clone(&arr2))])
                .unwrap(),
            Value::Int(10)
        );
        assert_eq!(mir_vm.cranelift_array_cache_builds(), 1);

        {
            let _active_mut = arr2.borrow_mut();
            let err = mir_vm
                .call("mir_sum_arr", vec![Value::Array(Rc::clone(&arr2))])
                .expect_err("Cranelift dispatch pre-entry try_borrow() must fail closed");
            assert!(
                err.to_string().contains("mutable borrow active"),
                "Expected Cranelift dispatch borrow error, got: {err}"
            );
            assert_eq!(mir_vm.call_stack_depth(), 0);
        }
        assert_eq!(
            mir_vm
                .call("mir_sum_arr", vec![Value::Array(Rc::clone(&arr2))])
                .unwrap(),
            Value::Int(10)
        );
    }
}

#[test]
fn test_perf_t4_3_adv4_mixed_type_oom_rollback_borrowed_mutation_fail_closed_and_host_gate() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    // 4a. Mixed-type array `[1, "two"]`: direct helper refunds reserved bytes and leaves cache None
    {
        let ctx = MirContext::new();
        ctx.set_max_memory_limit(1024);
        ctx.set_memory_used(16);
        let mixed = Rc::new(LustArray::new(vec![Value::Int(1), Value::string("two")]));
        let slice = mixed.borrow();
        let res = build_or_get_cached_flat_array(&ctx, &mixed, &slice, FlatArrayElemTag::Int)
            .expect("Mixed type must return Ok(None) for transparent fallback");
        assert!(res.is_none(), "Mixed-type array must not build flat cache");
        assert!(
            !mixed.has_cached_flat(),
            "flat_cache must remain None after element type mismatch"
        );
        assert_eq!(
            ctx.memory_used(),
            16,
            "Reserved bytes must be fully refunded on element type mismatch"
        );
    }

    // 4b. Tight max_memory_limit OOM during cache build: direct helper + VM level
    {
        let ctx = MirContext::new();
        // 100 elements -> (100 + 2) * 8 = 816 bytes; limit = 64 bytes -> OOM
        ctx.set_max_memory_limit(64);
        ctx.set_memory_used(0);
        let arr = Rc::new(LustArray::new((0..100).map(Value::Int).collect()));
        let slice = arr.borrow();
        let err = build_or_get_cached_flat_array(&ctx, &arr, &slice, FlatArrayElemTag::Int)
            .expect_err("Cache build exceeding max_memory_limit must fail with OOM");
        assert!(
            err.to_string().contains("Out of memory"),
            "Expected OOM error, got: {err}"
        );
        assert_eq!(ctx.last_status(), MIR_STATUS_ERR_OOM);
        assert_eq!(ctx.memory_used(), 0, "OOM must roll back memory_used to 0");
        assert!(
            !arr.has_cached_flat(),
            "OOM rollback must leave flat_cache completely untouched (None)"
        );
    }

    {
        let src = r#"
function sum_arr(arr: Array<int>): int
    local n: int = array.len(arr)
    local i: int = 0
    local s: int = 0
    while i < n do
        if arr[i] is Ok(v) then
            s = s + v
        end
        i = i + 1
    end
    return s
end

function print_first(arr: Array<int>): int
    if arr[0] is Ok(v) then
        println(v)
        return v
    end
    return -1
end
"#;
        let mut vm = compile_embedded_vm(src, true);
        let sum_arr_name = find_fn_name(&vm, "sum_arr");
        let print_first_name = find_fn_name(&vm, "print_first");

        // 4d. External/console call gating: `print_first` has `has_external_or_host_calls == true`
        assert_eq!(vm.cranelift_has_external_or_host_calls(&sum_arr_name), Some(false));
        assert_eq!(vm.cranelift_has_external_or_host_calls(&print_first_name), Some(true));

        let p_arr = Rc::new(LustArray::new(vec![Value::Int(77)]));
        assert_eq!(
            vm.call(&print_first_name, vec![Value::Array(Rc::clone(&p_arr))])
                .unwrap(),
            Value::Int(77)
        );
        assert_eq!(
            vm.cranelift_array_cache_builds(),
            0,
            "Functions with external/host/console calls must stay on standard copy-in marshalling"
        );
        assert!(!p_arr.has_cached_flat());

        // VM-level OOM on cache build for large array
        vm.set_memory_budget_bytes(32).expect("set tight budget");
        let big_arr = Rc::new(LustArray::new((0..200).map(Value::Int).collect()));
        let err = vm
            .call(&sum_arr_name, vec![Value::Array(Rc::clone(&big_arr))])
            .expect_err("Tight VM memory budget must reject flat-cache build with OOM");
        assert!(
            err.to_string().contains("Out of memory"),
            "Expected OOM error, got: {err}"
        );
        assert!(
            !big_arr.has_cached_flat(),
            "Failed OOM VM call must leave flat_cache None"
        );
        assert_eq!(vm.cranelift_array_cache_builds(), 0);
        assert_eq!(vm.call_stack_depth(), 0);

        // Mixed-type array via VM call falls back transparently without corrupting flat_cache
        vm.clear_memory_budget().expect("clear budget");
        let mixed_vm = Rc::new(LustArray::new(vec![Value::Int(1), Value::string("two")]));
        let _ = vm.call(&sum_arr_name, vec![Value::Array(Rc::clone(&mixed_vm))]);
        assert!(!mixed_vm.has_cached_flat());
        assert_eq!(vm.cranelift_array_cache_builds(), 0);
    }

    // 4c. Borrowed read-only provenance fail-closed rejection on lust_mir_array_push & lust_mir_array_set
    {
        let ctx = MirContext::new();
        let buf = Rc::new(FlatArrayBuffer::new(vec![3, 3, 10, 20, 30]));
        {
            let _guard = enter_mir_context(&ctx);
            ctx.register_borrowed_readonly_array(buf.as_ptr(), buf.word_len(), MirType::Int)
                .expect("register borrowed readonly array");

            // Reads work seamlessly
            assert_eq!(lust_mir_array_len(buf.as_ptr()), 3);
            assert_eq!(lust_mir_array_get(buf.as_ptr(), 0), 10);
            assert_eq!(lust_mir_array_get(buf.as_ptr(), 2), 30);

            // Attempted push MUST fail closed without mutating or redirecting buffer
            take_current_sticky_error();
            let push_rc = lust_mir_array_push(buf.as_ptr() as *mut i64, 999);
            assert_eq!(push_rc, LUST_STATUS_ERR_GENERIC as i64);
            assert_eq!(ctx.last_status(), LUST_STATUS_ERR_GENERIC as i64);
            let sticky = take_current_sticky_error().expect("sticky error set on borrowed push");
            assert!(sticky.contains("borrowed read-only"));

            // Attempted set MUST fail closed without mutating buffer
            ctx.set_status(0);
            lust_mir_array_set(buf.as_ptr() as *mut i64, 0, 999);
            assert_eq!(ctx.last_status(), LUST_STATUS_ERR_GENERIC as i64);
            let sticky2 = take_current_sticky_error().expect("sticky error set on borrowed set");
            assert!(sticky2.contains("borrowed read-only"));

            // Buffer header and elements remain 100% untouched
            unsafe {
                assert_eq!(*buf.as_ptr(), 3);
                assert_eq!(*buf.as_ptr().add(1), 3);
                assert_eq!(*buf.as_ptr().add(2), 10);
                assert_eq!(*buf.as_ptr().add(3), 20);
                assert_eq!(*buf.as_ptr().add(4), 30);
            }

            // clear_to_baselines inside scope must never free borrowed buffer
            assert_eq!(ctx.buffer_count(), 0);
        }

        // After MirContextGuard drops: provenance entry is removed and Rc<FlatArrayBuffer> is still alive & intact!
        assert!(ctx.get_allocation_record(buf.as_ptr() as usize).is_none());
        assert_eq!(Rc::strong_count(&buf), 1);
        unsafe {
            assert_eq!(*buf.as_ptr().add(2), 10);
        }
    }
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_perf_t4_3_adv4_mid_loop_deopt_on_borrowed_readonly_array_resumes_cleanly() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
function sum_arr(arr: Array<int>): int
    local n: int = array.len(arr)
    local i: int = 0
    local s: int = 0
    while i < n do
        if arr[i] is Ok(v) then
            s = s + v
        end
        i = i + 1
    end
    return s
end
"#;

    let mut vm = compile_embedded_vm(src, true);
    let sum_arr_name = find_fn_name(&vm, "sum_arr");

    let arr_rc = Rc::new(LustArray::new((1..=20).map(Value::Int).collect()));
    // Warm flat cache first
    assert_eq!(
        vm.call(&sum_arr_name, vec![Value::Array(Rc::clone(&arr_rc))])
            .unwrap(),
        Value::Int(210)
    );
    assert_eq!(vm.cranelift_array_cache_builds(), 1);
    assert_eq!(vm.cranelift_array_cache_hits(), 0);

    // Force mid-loop deopt at poll 5 while running on the borrowed read-only cache hit!
    safepoint_reset();
    safepoint_request_deopt_on_poll(None, 5);
    let res = vm
        .call(&sum_arr_name, vec![Value::Array(Rc::clone(&arr_rc))])
        .expect("Mid-loop deopt on borrowed read-only array must resume cleanly");
    assert_eq!(res, Value::Int(210));
    assert_eq!(
        vm.cranelift_array_cache_hits(),
        1,
        "Deopted call must have entered via the borrowed read-only cache hit"
    );
    assert_eq!(vm.call_stack_depth(), 0);

    // Borrow guard MUST have been released cleanly after deopt resume, allowing immediate mutation
    {
        let mut m = arr_rc
            .try_borrow_mut()
            .expect("Borrowed Ref guard must be released after deopt return");
        m[0] = Value::Int(101); // +100 delta
    }
    assert!(!arr_rc.has_cached_flat());

    safepoint_reset();
    assert_eq!(
        vm.call(&sum_arr_name, vec![Value::Array(Rc::clone(&arr_rc))])
            .unwrap(),
        Value::Int(310)
    );
    assert_eq!(vm.cranelift_array_cache_builds(), 2);
}

#[test]
fn test_perf_t4_3_adv5_mismatched_call_arg_type_marks_has_external_or_host_calls() {
    use lust::ast::Span;
    use lust::mir::ir::{
        FunctionSignature, MirConstant, MirFunction, MirModule, Operand, Place, Terminator,
    };
    use lust::vm::cranelift_dispatch::compute_has_external_or_host_calls;

    // Callee expects (Int) -> Int
    let mut callee = MirFunction::new(
        FunctionSignature {
            name: "pure_callee".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: Default::default(),
        },
        Span::dummy(),
    );
    let p0 = callee.params[0];
    callee.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(p0))),
        Span::dummy(),
    ));

    // Caller A calls pure_callee with matching Int constant -> safe (false)
    let mut caller_ok = MirFunction::new(
        FunctionSignature {
            name: "caller_ok".into(),
            param_types: vec![],
            return_type: MirType::Int,
            is_public: true,
            effects: Default::default(),
        },
        Span::dummy(),
    );
    let ret_ok = caller_ok.add_local(Some("r".into()), MirType::Int, false, Span::dummy());
    let bb1 = caller_ok.new_block();
    caller_ok.blocks[0].terminator = Some(Terminator::Call {
        callee: "pure_callee".into(),
        args: vec![Operand::Constant(MirConstant::Int(7))],
        destination: Place::Local(ret_ok),
        success: bb1,
        error: None,
        span: Span::dummy(),
    });
    caller_ok.blocks[bb1.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(ret_ok))),
        Span::dummy(),
    ));

    // Caller B calls pure_callee with mismatched Float constant -> fail-closed (true)
    let mut caller_mismatch = MirFunction::new(
        FunctionSignature {
            name: "caller_mismatch".into(),
            param_types: vec![],
            return_type: MirType::Int,
            is_public: true,
            effects: Default::default(),
        },
        Span::dummy(),
    );
    let ret_mm = caller_mismatch.add_local(Some("r".into()), MirType::Int, false, Span::dummy());
    let bb1_mm = caller_mismatch.new_block();
    caller_mismatch.blocks[0].terminator = Some(Terminator::Call {
        callee: "pure_callee".into(),
        args: vec![Operand::Constant(MirConstant::Float(7.5))],
        destination: Place::Local(ret_mm),
        success: bb1_mm,
        error: None,
        span: Span::dummy(),
    });
    caller_mismatch.blocks[bb1_mm.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(ret_mm))),
        Span::dummy(),
    ));

    // Caller C calls pure_callee with Unknown local -> fail-closed (true)
    let mut caller_unknown = MirFunction::new(
        FunctionSignature {
            name: "caller_unknown".into(),
            param_types: vec![MirType::Unknown],
            return_type: MirType::Int,
            is_public: true,
            effects: Default::default(),
        },
        Span::dummy(),
    );
    let unk = caller_unknown.params[0];
    let ret_u = caller_unknown.add_local(Some("r".into()), MirType::Int, false, Span::dummy());
    let bb1_u = caller_unknown.new_block();
    caller_unknown.blocks[0].terminator = Some(Terminator::Call {
        callee: "pure_callee".into(),
        args: vec![Operand::Copy(Place::Local(unk))],
        destination: Place::Local(ret_u),
        success: bb1_u,
        error: None,
        span: Span::dummy(),
    });
    caller_unknown.blocks[bb1_u.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(ret_u))),
        Span::dummy(),
    ));

    let mut module = MirModule::new("type_compat_gate".into());
    module.add_function(callee);
    module.add_function(caller_ok);
    module.add_function(caller_mismatch);
    module.add_function(caller_unknown);

    let flags = compute_has_external_or_host_calls(&module);
    assert_eq!(flags.get("pure_callee"), Some(&false));
    assert_eq!(flags.get("caller_ok"), Some(&false));
    assert_eq!(
        flags.get("caller_mismatch"),
        Some(&true),
        "Mismatched argument/parameter type must mark caller fail-closed"
    );
    assert_eq!(
        flags.get("caller_unknown"),
        Some(&true),
        "Unknown argument type must mark caller fail-closed"
    );
}

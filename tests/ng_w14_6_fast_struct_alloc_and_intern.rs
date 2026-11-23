#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! NG-W14-6: Fast struct allocation & interned struct type names (`tree`).
//! Verifies:
//! 1. Interned `Rc<str>` pointer sharing across `lust_mir_alloc_struct_named`, `lust_mir_alloc_struct`,
//!    and `MirContext::alloc_struct_with_fields` (MRU hit, HashSet hit, distinct source byte buffers,
//!    invalid UTF-8 / null / negative length fail-closed, and bounded intern table cap).
//! 2. Exact `SEC-02` memory budget accounting and transactional `DropGuard` OOM rollback with zero drift.
//! 3. Recursive `tree`-style `Node` struct allocation/traversal JIT == VM parity (including >8-field
//!    structs exercising heap fallback), tight memory-budget OOM trap, and `--features test-deopt` parity.

use std::rc::Rc;
use std::sync::Mutex;

use lust::bytecode::Value;
use lust::mir::cranelift_backend::{
    safepoint_reset, take_current_sticky_error, take_deopt_payload,
};
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::{get_current_sticky_error, safepoint_request_deopt_on_poll};
use lust::mir::runtime_helpers::{
    MIR_STATUS_ERR_OOM, MIR_STATUS_OK, MirAllocationKind, MirContext, lust_mir_alloc_struct,
    lust_mir_alloc_struct_named,
};
use lust::vfs::VirtualConsole;
#[cfg(feature = "test-deopt")]
use lust::vm::DeoptPayloadSource;
use lust::vm::{VM, reset_effect_handler_state};
use lust::{EmbeddedModule, LustConfig, compile_program_with_config, load_program_from_embedded};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn reset_all_state() {
    safepoint_reset();
    reset_effect_handler_state();
    take_deopt_payload();
    take_current_sticky_error();
}

fn compile_embedded_vm(source: &str, cranelift: bool) -> VM {
    let entries = &[EmbeddedModule {
        module: "main",
        parent: None,
        source: Some(source),
    }];
    let program = load_program_from_embedded(entries, "main").expect("load embedded program");
    let mut cfg = LustConfig::default().with_cranelift_jit(cranelift);
    cfg.set_use_mir(false);
    let mut vm = compile_program_with_config(program, &cfg).expect("compile embedded program");
    vm.console = VirtualConsole::captured();
    vm
}

fn find_fn_name(vm: &VM, suffix: &str) -> String {
    for candidate in [suffix.to_string(), format!("main.{suffix}")] {
        if vm.mir_functions.contains_key(&candidate) {
            return candidate;
        }
    }
    vm.mir_functions
        .keys()
        .find(|k| k.ends_with(suffix))
        .cloned()
        .unwrap_or_else(|| panic!("Function '{suffix}' not found in mir_functions"))
}

#[test]
fn test_ng_w14_6_struct_type_name_interning_ptr_eq_and_cap() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let ctx = MirContext::new();
    let _guard = ctx.enter();

    // Two distinct heap buffers holding the same UTF-8 bytes "Node"
    let buf_a = String::from("Node");
    let buf_b = String::from("Node");
    assert_ne!(buf_a.as_ptr(), buf_b.as_ptr());

    let p1 = lust_mir_alloc_struct_named(3, buf_a.as_ptr(), buf_a.len() as i64);
    let p2 = lust_mir_alloc_struct_named(3, buf_b.as_ptr(), buf_b.len() as i64);
    let p_leaf = lust_mir_alloc_struct_named(2, b"Leaf".as_ptr(), 4);
    // Re-allocate "Node" after MRU eviction by "Leaf" -> must still hit HashSet and share Rc
    let p3 = lust_mir_alloc_struct_named(3, buf_b.as_ptr(), buf_b.len() as i64);
    let p4 = ctx.alloc_struct_with_fields("Node", &[10, 20, 30]);

    assert!(!p1.is_null() && !p2.is_null() && !p_leaf.is_null() && !p3.is_null() && !p4.is_null());
    assert_eq!(ctx.interned_struct_type_name_count(), 2);

    let name_of = |ptr: *mut i64| -> Rc<str> {
        let rec = ctx
            .get_active_allocation_record(ptr as usize)
            .expect("allocation record present");
        match rec.kind {
            MirAllocationKind::Struct(rc) => rc,
            other => panic!("expected MirAllocationKind::Struct, got {other:?}"),
        }
    };

    let n1 = name_of(p1);
    let n2 = name_of(p2);
    let n3 = name_of(p3);
    let n4 = name_of(p4);
    let n_leaf = name_of(p_leaf);

    assert_eq!(n1.as_ref(), "Node");
    assert_eq!(n_leaf.as_ref(), "Leaf");
    assert!(Rc::ptr_eq(&n1, &n2), "MRU hit must share Rc<str>");
    assert!(Rc::ptr_eq(&n1, &n3), "HashSet hit after MRU eviction must share Rc<str>");
    assert!(Rc::ptr_eq(&n1, &n4), "alloc_struct_with_fields must share interned Rc<str>");

    // Verify fields populated by alloc_struct_with_fields
    let f_words = unsafe { std::slice::from_raw_parts(p4, 5) };
    assert_eq!(f_words, &[3, 3, 10, 20, 30]);

    // Anonymous structs share empty_struct_type_name Rc<str>
    let anon1 = lust_mir_alloc_struct(1);
    let anon2 = lust_mir_alloc_struct(2);
    assert!(!anon1.is_null() && !anon2.is_null());
    let a1 = name_of(anon1);
    let a2 = name_of(anon2);
    assert_eq!(a1.as_ref(), "");
    assert!(Rc::ptr_eq(&a1, &a2), "anonymous struct allocations must share empty Rc<str>");

    // Invalid inputs fail closed without mutating memory_used
    let mem_before = ctx.memory_used();
    let bad_utf8: [u8; 2] = [0xFF, 0xFE];
    assert!(lust_mir_alloc_struct_named(1, bad_utf8.as_ptr(), 2).is_null());
    assert_eq!(
        ctx.last_status(),
        lust::effect_abi::LUST_STATUS_ERR_GENERIC as i64
    );
    assert!(lust_mir_alloc_struct_named(-1, b"Node".as_ptr(), 4).is_null());
    assert!(lust_mir_alloc_struct_named(1, std::ptr::null(), 4).is_null());
    assert!(lust_mir_alloc_struct_named(1, b"Node".as_ptr(), 0).is_null());
    assert!(ctx.alloc_struct_with_fields("", &[1]).is_null());
    assert_eq!(ctx.memory_used(), mem_before);

    // Intern set stays bounded even under adversarial unique names
    for i in 0..4200usize {
        let dyn_name = format!("DynType_{i}");
        let rc = ctx.intern_struct_type_name(&dyn_name);
        assert_eq!(rc.as_ref(), dyn_name.as_str());
    }
    assert_eq!(
        ctx.interned_struct_type_name_count(),
        4096,
        "interned struct type table must stay bounded at 4096 entries"
    );
}

#[test]
fn test_ng_w14_6_oom_budget_exact_and_rollback() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let ctx = MirContext::new();
    let _guard = ctx.enter();

    // 3 fields => (3 + 2) * 8 = 40 bytes per struct
    ctx.set_max_memory_limit(80);
    ctx.set_memory_used(0);
    ctx.set_status(MIR_STATUS_OK);

    let p1 = lust_mir_alloc_struct_named(3, b"Node".as_ptr(), 4);
    assert!(!p1.is_null());
    assert_eq!(ctx.memory_used(), 40);
    assert_eq!(ctx.last_status(), MIR_STATUS_OK);

    let p2 = ctx.alloc_struct_with_fields("Node", &[1, 2, 3]);
    assert!(!p2.is_null());
    assert_eq!(ctx.memory_used(), 80);
    assert_eq!(ctx.last_status(), MIR_STATUS_OK);

    // Next allocation via lust_mir_alloc_struct_named must OOM and roll back memory_used to 80
    let p3 = lust_mir_alloc_struct_named(3, b"Node".as_ptr(), 4);
    assert!(p3.is_null());
    assert_eq!(ctx.last_status(), MIR_STATUS_ERR_OOM);
    assert_eq!(ctx.memory_used(), 80);

    // Next allocation via alloc_struct_with_fields must also OOM and roll back memory_used to 80
    ctx.set_status(MIR_STATUS_OK);
    let p4 = ctx.alloc_struct_with_fields("Node", &[4, 5, 6]);
    assert!(p4.is_null());
    assert_eq!(ctx.last_status(), MIR_STATUS_ERR_OOM);
    assert_eq!(ctx.memory_used(), 80);

    // Overflowing length must OOM cleanly without panicking or changing memory_used
    ctx.set_max_memory_limit(i64::MAX);
    ctx.set_status(MIR_STATUS_OK);
    let p5 = lust_mir_alloc_struct_named(i64::MAX, b"Node".as_ptr(), 4);
    assert!(p5.is_null());
    assert_eq!(ctx.last_status(), MIR_STATUS_ERR_OOM);
    assert_eq!(ctx.memory_used(), 80);
}

#[test]
fn test_ng_w14_6_tree_alloc_oom_and_deopt_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let tree_src = r#"
struct Node
    value: int
    left: Option<Node>
    right: Option<Node>
end

struct BigNode
    f0: int
    f1: int
    f2: int
    f3: int
    f4: int
    f5: int
    f6: int
    f7: int
    f8: int
    next: Option<BigNode>
end

function build(depth: int, value: int): Node
    if depth == 0 then
        return Node { value = value, left = Option.None, right = Option.None }
    end
    local l: Node = build(depth - 1, value * 2)
    local r: Node = build(depth - 1, value * 2 + 1)
    return Node { value = value, left = Option.Some(l), right = Option.Some(r) }
end

function sum(node: Node): int
    local s: int = node.value
    if node.left is Some(l) then
        s = s + sum(l)
    end
    if node.right is Some(r) then
        s = s + sum(r)
    end
    return s
end

function build_and_sum(depth: int): int
    local t: Node = build(depth, 1)
    local total: int = 0
    local i: int = 0
    while i < 4 do
        total = total + sum(t)
        i = i + 1
    end
    return total
end

function build_big(depth: int): BigNode
    if depth == 0 then
        return BigNode {
            f0 = 1, f1 = 2, f2 = 3, f3 = 4, f4 = 5,
            f5 = 6, f6 = 7, f7 = 8, f8 = 9, next = Option.None
        }
    end
    local child: BigNode = build_big(depth - 1)
    return BigNode {
        f0 = depth, f1 = 2, f2 = 3, f3 = 4, f4 = 5,
        f5 = 6, f6 = 7, f7 = 8, f8 = 9, next = Option.Some(child)
    }
end

function sum_big(node: BigNode): int
    local s: int = node.f0 + node.f1 + node.f2 + node.f3 + node.f4 + node.f5 + node.f6 + node.f7 + node.f8
    if node.next is Some(c) then
        s = s + sum_big(c)
    end
    return s
end
"#;

    // 1. VM oracle vs Cranelift JIT parity for recursive `Node` tree and >8-field `BigNode`
    let mut vm_oracle = compile_embedded_vm(tree_src, false);
    let mut vm_clif = compile_embedded_vm(tree_src, true);
    vm_clif.compile_cranelift().expect("compile_cranelift");

    let build_fn = find_fn_name(&vm_clif, "build");
    let sum_fn = find_fn_name(&vm_clif, "sum");
    let build_and_sum_fn = find_fn_name(&vm_clif, "build_and_sum");
    let build_big_fn = find_fn_name(&vm_clif, "build_big");
    let sum_big_fn = find_fn_name(&vm_clif, "sum_big");

    assert!(vm_clif.cranelift_compiled.contains_key(&build_fn));
    assert!(vm_clif.cranelift_compiled.contains_key(&sum_fn));
    assert!(vm_clif.cranelift_compiled.contains_key(&build_and_sum_fn));
    assert!(vm_clif.cranelift_compiled.contains_key(&build_big_fn));
    assert!(vm_clif.cranelift_compiled.contains_key(&sum_big_fn));

    // Pre-interned struct names populated during sync_mir_context_struct_metadata
    assert!(
        vm_clif.mir_context.interned_struct_type_name_count() >= 2,
        "Expected Node and BigNode to be pre-interned in MirContext"
    );

    let expected_sum = vm_oracle
        .call(&build_and_sum_fn, vec![Value::Int(8)])
        .expect("oracle build_and_sum");
    let actual_sum = vm_clif
        .call(&build_and_sum_fn, vec![Value::Int(8)])
        .expect("cranelift build_and_sum");
    assert_eq!(actual_sum, expected_sum, "JIT/VM parity mismatch on build_and_sum(8)");
    assert!(vm_clif.cranelift_entry_count >= 1);

    // Boundary marshal + decode roundtrip (build returns Node to VM, then sum marshals Node back to JIT)
    let tree_val = vm_clif
        .call(&build_fn, vec![Value::Int(7), Value::Int(1)])
        .expect("cranelift build(7, 1)");
    let expected_tree_sum = vm_oracle
        .call(&sum_fn, vec![tree_val.clone()])
        .expect("oracle sum(tree)");
    for _ in 0..3 {
        let jit_tree_sum = vm_clif
            .call(&sum_fn, vec![tree_val.clone()])
            .expect("cranelift sum(tree)");
        assert_eq!(jit_tree_sum, expected_tree_sum, "sum(tree) boundary marshal parity mismatch");
    }

    // >8-field struct (BigNode) marshal + decode parity (exercises RawFieldBuf::Heap path)
    let big_val = vm_clif
        .call(&build_big_fn, vec![Value::Int(5)])
        .expect("cranelift build_big(5)");
    let expected_big_sum = vm_oracle
        .call(&sum_big_fn, vec![big_val.clone()])
        .expect("oracle sum_big");
    let jit_big_sum = vm_clif
        .call(&sum_big_fn, vec![big_val])
        .expect("cranelift sum_big");
    assert_eq!(jit_big_sum, expected_big_sum, "BigNode (>8 fields) JIT/VM parity mismatch");

    // 2. Tight SEC-02 memory budget during recursive tree allocation must trap OOM cleanly
    //    without panicking, and leave MirContext buffers cleaned up after call exit.
    {
        let mut vm_oom = compile_embedded_vm(tree_src, true);
        vm_oom.compile_cranelift().expect("compile_cranelift for OOM test");
        vm_oom
            .set_memory_budget_bytes(1024)
            .expect("set memory budget");
        let err = vm_oom
            .call(&build_fn, vec![Value::Int(8), Value::Int(1)])
            .expect_err("depth-8 tree (511 nodes * 40B = 20440B) must OOM under 1024B budget");
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("memory") || msg.contains("oom"),
            "expected memory/OOM error, got: {err}"
        );
        assert_eq!(
            vm_oom.mir_context.buffer_count(),
            0,
            "MirContext buffers must be fully cleaned up after OOM trap"
        );
        assert_eq!(
            vm_oom.mir_context.allocation_records.borrow().len(),
            0,
            "MirContext allocation_records must be empty after OOM trap"
        );
    }

    // 3. Under `--features test-deopt`, force native safepoint deopt inside build_and_sum
    //    and verify exact deopt materialization of live `Node` tree + exact oracle parity.
    #[cfg(feature = "test-deopt")]
    {
        reset_all_state();
        let mut vm_deopt = compile_embedded_vm(tree_src, true);
        vm_deopt.compile_cranelift().expect("compile_cranelift for deopt test");
        let fn_name = find_fn_name(&vm_deopt, "build_and_sum");

        safepoint_request_deopt_on_poll(Some(fn_name.clone()), 1);
        let entries_before = vm_deopt.cranelift_entry_count;
        let deopt_res = vm_deopt
            .call(&fn_name, vec![Value::Int(6)])
            .expect("poll-1 deopt in build_and_sum(6) must materialize Node tree and resume cleanly");
        safepoint_reset();

        let expected_6 = vm_oracle
            .call(&build_and_sum_fn, vec![Value::Int(6)])
            .expect("oracle build_and_sum(6)");
        assert!(
            vm_deopt.cranelift_entry_count >= entries_before + 1,
            "build_and_sum must enter Cranelift natively (and subsequent sum(t) calls may re-enter JIT)"
        );
        assert_eq!(
            vm_deopt.last_deopt_source,
            Some(DeoptPayloadSource::NativeExecution),
            "deopt must come from native execution"
        );
        assert_eq!(deopt_res, expected_6, "post-deopt tree result must match VM oracle");
        assert!(take_deopt_payload().is_none());
        assert!(get_current_sticky_error().is_none());
    }
}

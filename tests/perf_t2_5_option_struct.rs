#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::path::PathBuf;
use std::process::Command;
use std::rc::Rc;
use std::sync::Mutex;

use lust::ast::{Span, Type, TypeKind};
use lust::bytecode::{FieldStorage, StructLayout, Value};
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::safepoint_request_deopt_on_poll;
use lust::mir::cranelift_backend::{
    MIR_ARRAY_ELEM_TAG_INT, lust_mir_alloc_array_typed, lust_mir_alloc_struct_named,
    safepoint_reset, take_current_sticky_error, take_deopt_payload,
};
use lust::mir::runtime_helpers::MirContext;
use lust::vfs::VirtualConsole;
use lust::vm::cranelift_dispatch::{MAX_STRUCT_DECODE_DEPTH, validate_and_decode_struct};
use lust::vm::{VM, reset_effect_handler_state};
use lust::{EmbeddedModule, LustConfig, compile_program_with_config, load_program_from_embedded};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn lust_bin() -> &'static str {
    env!("CARGO_BIN_EXE_lust")
}

fn suite_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks")
        .join("suite")
        .join(name)
}

fn reset_all_state() {
    safepoint_reset();
    reset_effect_handler_state();
    take_deopt_payload();
    take_current_sticky_error();
}

fn parse_cranelift_stats(stderr: &str) -> (u64, u64) {
    for line in stderr.lines() {
        if let Some(rest) = line.strip_prefix("Cranelift stats:") {
            let mut entries = None;
            let mut compilations = None;
            for tok in rest.split_whitespace() {
                if let Some(v) = tok.strip_prefix("entries=") {
                    entries = v.parse::<u64>().ok();
                } else if let Some(v) = tok.strip_prefix("compilations=") {
                    compilations = v.parse::<u64>().ok();
                }
            }
            if let (Some(e), Some(c)) = (entries, compilations) {
                return (e, c);
            }
        }
    }
    panic!("Missing `Cranelift stats: entries=... compilations=...` in stderr:\n{stderr}");
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

fn make_node_layout() -> Rc<StructLayout> {
    let span = Span::dummy();
    let opt_node = Type::new(
        TypeKind::Option(Box::new(Type::new(TypeKind::Named("Node".to_string()), span))),
        span,
    );
    Rc::new(StructLayout::new(
        "Node".to_string(),
        vec![
            Rc::new("value".to_string()),
            Rc::new("left".to_string()),
            Rc::new("right".to_string()),
        ],
        vec![FieldStorage::Strong, FieldStorage::Strong, FieldStorage::Strong],
        vec![Type::new(TypeKind::Int, span), opt_node.clone(), opt_node],
        vec![None, None, None],
    ))
}

#[test]
fn test_perf_t2_5_recursive_option_struct_tree() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let tree_path = suite_path("tree.lust");

    let vm_out = Command::new(lust_bin())
        .args(["--jit=off"])
        .arg(&tree_path)
        .output()
        .expect("run lust --jit=off tree.lust");
    assert!(
        vm_out.status.success(),
        "lust --jit=off tree.lust failed: {}",
        String::from_utf8_lossy(&vm_out.stderr)
    );
    let vm_stdout = String::from_utf8_lossy(&vm_out.stdout);
    assert_eq!(vm_stdout, "10738090560\n", "VM oracle stdout mismatch for tree.lust");

    let clif_out = Command::new(lust_bin())
        .args(["--jit=cranelift"])
        .arg(&tree_path)
        .env("LUST_JIT_STATS", "1")
        .output()
        .expect("run lust --jit=cranelift tree.lust");
    assert!(
        clif_out.status.success(),
        "lust --jit=cranelift tree.lust failed: {}",
        String::from_utf8_lossy(&clif_out.stderr)
    );
    let clif_stdout = String::from_utf8_lossy(&clif_out.stdout);
    let clif_stderr = String::from_utf8_lossy(&clif_out.stderr);
    assert_eq!(
        clif_stdout, vm_stdout,
        "tree.lust Cranelift stdout mismatch vs VM oracle"
    );
    let (entries, compilations) = parse_cranelift_stats(&clif_stderr);
    assert!(
        compilations >= 4 && entries >= 1,
        "Expected tree.lust to compile build, sum, count_even, and __script in Cranelift (compilations >= 4, entries >= 1), got entries={entries} compilations={compilations}\nstderr:\n{clif_stderr}"
    );
}

#[test]
fn test_perf_t2_5_sieve_compiles_native_1_1() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let sieve_path = suite_path("sieve.lust");
    let clif_out = Command::new(lust_bin())
        .args(["--jit=cranelift"])
        .arg(&sieve_path)
        .env("LUST_JIT_STATS", "1")
        .output()
        .expect("run lust --jit=cranelift sieve.lust");
    assert!(
        clif_out.status.success(),
        "lust --jit=cranelift sieve.lust failed: {}",
        String::from_utf8_lossy(&clif_out.stderr)
    );
    let clif_stdout = String::from_utf8_lossy(&clif_out.stdout);
    let clif_stderr = String::from_utf8_lossy(&clif_out.stderr);
    assert_eq!(clif_stdout, "664579\n", "sieve.lust stdout mismatch");
    let (entries, compilations) = parse_cranelift_stats(&clif_stderr);
    assert_eq!(
        (entries, compilations),
        (1, 1),
        "Expected sieve.lust to compile natively with entries=1 compilations=1, got entries={entries} compilations={compilations}\nstderr:\n{clif_stderr}"
    );
}

#[test]
fn test_perf_t2_5_option_struct_marshal_roundtrip_and_writeback_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
        struct Node
            value: int
            left: Option<Node>
            right: Option<Node>
        end

        function make_none(): Option<Node>
            return Option.None
        end

        function make_some(v: int): Option<Node>
            local leaf: Node = Node { value = v + 1, left = Option.None, right = Option.None }
            local root: Node = Node { value = v, left = Option.Some(leaf), right = Option.None }
            return Option.Some(root)
        end

        function make_diamond(v: int): Node
            local shared: Node = Node { value = v, left = Option.None, right = Option.None }
            return Node { value = v * 10, left = Option.Some(shared), right = Option.Some(shared) }
        end

        function inspect_opt(opt: Option<Node>): int
            if opt is None then
                return -1
            end
            if opt is Some(n) then
                local total: int = n.value
                if n.left is Some(l) then
                    total = total + l.value
                end
                if n.right is Some(r) then
                    total = total + r.value
                end
                return total
            end
            return -999
        end

        function attach_left(node: Node, v: int): int
            local child: Node = Node { value = v, left = Option.None, right = Option.None }
            node.left = Option.Some(child)
            return node.value + v
        end

        function clear_left(node: Node): int
            node.left = Option.None
            return node.value
        end
    "#;

    let mut vm_clif = compile_embedded_vm(src, true);
    let mut vm_bc = compile_embedded_vm(src, false);

    // 1. Return Option.None -> exact Value::Enum("Option", "None", None)
    let e0 = vm_clif.cranelift_entry_count;
    let none_clif = vm_clif.call("make_none", vec![]).expect("clif make_none");
    let none_bc = vm_bc.call("make_none", vec![]).expect("bc make_none");
    assert!(vm_clif.cranelift_entry_count > e0, "make_none must execute natively in Cranelift");
    assert_eq!(none_clif, Value::none());
    assert_eq!(none_clif, none_bc);
    assert_eq!(none_clif.as_enum(), Some(("Option", "None", None)));

    // 2. Return Option.Some(Node { ... }) -> exact Value::Enum("Option", "Some", Some([Value::Struct(_)]))
    let e1 = vm_clif.cranelift_entry_count;
    let some_clif = vm_clif.call("make_some", vec![Value::Int(41)]).expect("clif make_some");
    let some_bc = vm_bc.call("make_some", vec![Value::Int(41)]).expect("bc make_some");
    assert!(vm_clif.cranelift_entry_count > e1, "make_some must execute natively in Cranelift");
    assert_eq!(some_clif, some_bc);
    match some_clif.as_enum() {
        Some(("Option", "Some", Some([Value::Struct(root_rc)]))) => {
            let fields = root_rc.fields.borrow();
            assert_eq!(fields[0], Value::Int(41));
            assert!(matches!(fields[1].as_enum(), Some(("Option", "Some", Some([Value::Struct(_)])))));
            assert_eq!(fields[2], Value::none());
        }
        other => panic!("Expected Option.Some(Struct(Node)), got {other:?}"),
    }

    // 3. Marshal Option.None & Option.Some(Node) back into Cranelift as parameters
    let e2 = vm_clif.cranelift_entry_count;
    let insp_none_clif = vm_clif.call("inspect_opt", vec![none_clif.clone()]).expect("clif inspect none");
    let insp_none_bc = vm_bc.call("inspect_opt", vec![none_bc.clone()]).expect("bc inspect none");
    assert!(vm_clif.cranelift_entry_count > e2, "inspect_opt(None) must execute natively");
    assert_eq!(insp_none_clif, Value::Int(-1));
    assert_eq!(insp_none_clif, insp_none_bc);

    let e3 = vm_clif.cranelift_entry_count;
    let insp_some_clif = vm_clif.call("inspect_opt", vec![some_clif.clone()]).expect("clif inspect some");
    let insp_some_bc = vm_bc.call("inspect_opt", vec![some_bc.clone()]).expect("bc inspect some");
    assert!(vm_clif.cranelift_entry_count > e3, "inspect_opt(Some) must execute natively");
    assert_eq!(insp_some_clif, Value::Int(83));
    assert_eq!(insp_some_clif, insp_some_bc);

    // 4. Diamond DAG sharing (non-cyclic) must succeed and preserve parity
    let e4 = vm_clif.cranelift_entry_count;
    let diam_clif = vm_clif.call("make_diamond", vec![Value::Int(7)]).expect("clif make_diamond");
    let diam_bc = vm_bc.call("make_diamond", vec![Value::Int(7)]).expect("bc make_diamond");
    assert!(vm_clif.cranelift_entry_count > e4, "make_diamond must execute natively");
    assert_eq!(diam_clif, diam_bc);
    let insp_diam_clif = vm_clif
        .call("inspect_opt", vec![Value::some(diam_clif)])
        .expect("clif inspect diamond");
    let insp_diam_bc = vm_bc
        .call("inspect_opt", vec![Value::some(diam_bc)])
        .expect("bc inspect diamond");
    assert_eq!(insp_diam_clif, Value::Int(84));
    assert_eq!(insp_diam_clif, insp_diam_bc);

    // 5. Mutation writeback: node.left = Option.Some(child) and node.left = Option.None
    let leaf_clif = vm_clif
        .instantiate_struct_ordered("main.Node", vec![Value::Int(10), Value::none(), Value::none()])
        .expect("instantiate Node (clif)");
    let leaf_bc = vm_bc
        .instantiate_struct_ordered("main.Node", vec![Value::Int(10), Value::none(), Value::none()])
        .expect("instantiate Node (bc)");

    let e5 = vm_clif.cranelift_entry_count;
    let r_att_clif = vm_clif
        .call("attach_left", vec![leaf_clif.clone(), Value::Int(25)])
        .expect("clif attach_left");
    let r_att_bc = vm_bc
        .call("attach_left", vec![leaf_bc.clone(), Value::Int(25)])
        .expect("bc attach_left");
    assert!(vm_clif.cranelift_entry_count > e5, "attach_left must execute natively");
    assert_eq!(r_att_clif, Value::Int(35));
    assert_eq!(r_att_clif, r_att_bc);
    assert_eq!(leaf_clif, leaf_bc, "Option.Some writeback must match VM oracle");

    let e6 = vm_clif.cranelift_entry_count;
    let r_clr_clif = vm_clif
        .call("clear_left", vec![leaf_clif.clone()])
        .expect("clif clear_left");
    let r_clr_bc = vm_bc
        .call("clear_left", vec![leaf_bc.clone()])
        .expect("bc clear_left");
    assert!(vm_clif.cranelift_entry_count > e6, "clear_left must execute natively");
    assert_eq!(r_clr_clif, Value::Int(10));
    assert_eq!(r_clr_clif, r_clr_bc);
    assert_eq!(leaf_clif, leaf_bc, "Option.None writeback must match VM oracle");
}

#[test]
fn test_perf_t2_5_adversarial_cyclic_pointer_graph_rejected_fail_closed() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let node_layout = make_node_layout();
    let ctx = MirContext::new();
    let _guard = ctx.enter();

    // 1. Direct self-referential cycle: node.left = node
    let self_ptr = lust_mir_alloc_struct_named(3, b"Node".as_ptr(), 4);
    assert!(!self_ptr.is_null());
    unsafe {
        *self_ptr.add(2) = 1;
        *self_ptr.add(3) = self_ptr as i64; // self-cycle
        *self_ptr.add(4) = 0;
    }
    let err_self = validate_and_decode_struct(&ctx, &[], self_ptr as i64, "Node", Some(&node_layout))
        .expect_err("Self-referential struct pointer cycle must be rejected fail-closed");
    let msg_self = format!("{err_self}");
    assert!(
        msg_self.contains("Cyclic"),
        "Expected cyclic pointer graph rejection, got: {msg_self}"
    );

    // 2. Two-node mutual cycle: a.left = b; b.left = a
    let ptr_a = lust_mir_alloc_struct_named(3, b"Node".as_ptr(), 4);
    let ptr_b = lust_mir_alloc_struct_named(3, b"Node".as_ptr(), 4);
    assert!(!ptr_a.is_null() && !ptr_b.is_null());
    unsafe {
        *ptr_a.add(2) = 10;
        *ptr_a.add(3) = ptr_b as i64;
        *ptr_a.add(4) = 0;
        *ptr_b.add(2) = 20;
        *ptr_b.add(3) = ptr_a as i64;
        *ptr_b.add(4) = 0;
    }
    let err_two = validate_and_decode_struct(&ctx, &[], ptr_a as i64, "Node", Some(&node_layout))
        .expect_err("2-node cyclic struct pointer graph must be rejected fail-closed");
    let msg_two = format!("{err_two}");
    assert!(
        msg_two.contains("Cyclic"),
        "Expected 2-node cyclic rejection, got: {msg_two}"
    );

    // 3. Non-cyclic chain deeper than MAX_STRUCT_DECODE_DEPTH (256) must fail closed on depth guard
    let mut prev: i64 = 0;
    for i in 0..=(MAX_STRUCT_DECODE_DEPTH + 2) {
        let p = lust_mir_alloc_struct_named(3, b"Node".as_ptr(), 4);
        assert!(!p.is_null());
        unsafe {
            *p.add(2) = i as i64;
            *p.add(3) = prev;
            *p.add(4) = 0;
        }
        prev = p as i64;
    }
    let err_depth = validate_and_decode_struct(&ctx, &[], prev, "Node", Some(&node_layout))
        .expect_err("Struct chain exceeding MAX_STRUCT_DECODE_DEPTH must be rejected fail-closed");
    let msg_depth = format!("{err_depth}");
    assert!(
        msg_depth.contains("depth limit"),
        "Expected depth limit rejection, got: {msg_depth}"
    );
    drop(_guard);

    // 4. End-to-end Cranelift guest creating a self-cycle on fresh struct AND on input writeback
    let cycle_src = r#"
        struct Node
            value: int
            left: Option<Node>
            right: Option<Node>
        end

        function make_cycle(): Node
            local n: Node = Node { value = 1, left = Option.None, right = Option.None }
            n.left = Option.Some(n)
            return n
        end

        function mutate_into_cycle(n: Node): int
            n.left = Option.Some(n)
            return n.value
        end

        function read_val(n: Node): int
            return n.value
        end
    "#;
    let mut vm_clif = compile_embedded_vm(cycle_src, true);

    let err_fresh = vm_clif
        .call("make_cycle", vec![])
        .expect_err("Cranelift returning self-cyclic Node must fail closed before stack overflow");
    assert!(
        format!("{err_fresh}").contains("Cyclic"),
        "Expected Cyclic error on fresh cycle return, got: {err_fresh}"
    );

    let input_node = vm_clif
        .instantiate_struct_ordered("main.Node", vec![Value::Int(5), Value::none(), Value::none()])
        .expect("instantiate Node");
    let err_wb = vm_clif
        .call("mutate_into_cycle", vec![input_node.clone()])
        .expect_err("Cranelift mutating input Node into cycle must fail closed during writeback");
    assert!(
        format!("{err_wb}").contains("Cyclic"),
        "Expected Cyclic error on input cycle writeback, got: {err_wb}"
    );

    // 5. Host-constructed cyclic VM Value::Struct passed as argument to Cranelift must fail closed
    if let Value::Struct(ref rc) = input_node {
        rc.fields.borrow_mut()[1] = Value::some(input_node.clone());
    }
    let err_in = vm_clif
        .call("read_val", vec![input_node.clone()])
        .expect_err("Marshaling cyclic VM Struct into Cranelift must fail closed");
    assert!(
        format!("{err_in}").contains("Cyclic"),
        "Expected Cyclic error on cyclic input marshaling, got: {err_in}"
    );
    // Break cycle before drop to avoid Rc leak in test process
    if let Value::Struct(ref rc) = input_node {
        rc.fields.borrow_mut()[1] = Value::none();
    }
}

#[test]
fn test_perf_t2_5_adversarial_forged_pointer_in_option_field_rejected_fail_closed() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let node_layout = make_node_layout();
    let ctx = MirContext::new();
    let _guard = ctx.enter();

    // 1. Forged unowned pointer inside Node.left
    let ptr_forged = lust_mir_alloc_struct_named(3, b"Node".as_ptr(), 4);
    assert!(!ptr_forged.is_null());
    unsafe {
        *ptr_forged.add(2) = 42;
        *ptr_forged.add(3) = 0xdead_beef_i64; // forged non-owned pointer
        *ptr_forged.add(4) = 0;
    }
    let err_forged =
        validate_and_decode_struct(&ctx, &[], ptr_forged as i64, "Node", Some(&node_layout))
            .expect_err("Forged non-owned pointer in Option<Node> field must fail closed");
    assert!(
        format!("{err_forged}").contains("not an owned allocation base"),
        "Expected provenance error for forged pointer, got: {err_forged}"
    );

    // 2. Wrong-kind owned allocation (Array<int> pointer) inside Node.left
    let arr_ptr = lust_mir_alloc_array_typed(2, MIR_ARRAY_ELEM_TAG_INT);
    assert!(!arr_ptr.is_null());
    let ptr_wrong_kind = lust_mir_alloc_struct_named(3, b"Node".as_ptr(), 4);
    assert!(!ptr_wrong_kind.is_null());
    unsafe {
        *ptr_wrong_kind.add(2) = 7;
        *ptr_wrong_kind.add(3) = arr_ptr as i64;
        *ptr_wrong_kind.add(4) = 0;
    }
    let err_kind =
        validate_and_decode_struct(&ctx, &[], ptr_wrong_kind as i64, "Node", Some(&node_layout))
            .expect_err("Owned Array pointer inside Option<Node> field must fail closed");
    assert!(
        format!("{err_kind}").contains("Type/kind mismatch"),
        "Expected kind mismatch error, got: {err_kind}"
    );

    // 3. Wrong-struct owned allocation (`Other` struct pointer) inside Node.left
    let other_ptr = lust_mir_alloc_struct_named(3, b"Other".as_ptr(), 5);
    assert!(!other_ptr.is_null());
    let ptr_wrong_struct = lust_mir_alloc_struct_named(3, b"Node".as_ptr(), 4);
    assert!(!ptr_wrong_struct.is_null());
    unsafe {
        *ptr_wrong_struct.add(2) = 9;
        *ptr_wrong_struct.add(3) = other_ptr as i64;
        *ptr_wrong_struct.add(4) = 0;
    }
    let err_struct =
        validate_and_decode_struct(&ctx, &[], ptr_wrong_struct as i64, "Node", Some(&node_layout))
            .expect_err("Owned wrong-struct pointer inside Option<Node> field must fail closed");
    assert!(
        format!("{err_struct}").contains("Struct type mismatch"),
        "Expected struct type mismatch error, got: {err_struct}"
    );
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_perf_t2_5_option_struct_live_deopt_resumes_cleanly() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
        struct Node
            value: int
            left: Option<Node>
            right: Option<Node>
        end

        function walk_sum(n: int): int
            local cur: Option<Node> = Option.None
            local i: int = 1
            while i <= n do
                local next_node: Node = Node { value = i, left = cur, right = Option.None }
                cur = Option.Some(next_node)
                i = i + 1
            end
            local acc: int = 0
            local step: int = 0
            while step < n do
                if cur is Some(node) then
                    acc = acc + node.value
                    cur = node.left
                end
                step = step + 1
            end
            return acc
        end
    "#;

    let mut vm_clif = compile_embedded_vm(src, true);
    let mut vm_bc = compile_embedded_vm(src, false);

    // Deopt mid-loop while live `cur: Option<Node>` holds a linked chain of `Node`s!
    safepoint_request_deopt_on_poll(Some("walk_sum".to_string()), 4);
    let clif_res = vm_clif
        .call("walk_sum", vec![Value::Int(10)])
        .expect("Cranelift walk_sum deopt must resume cleanly in continuation bytecode");
    let bc_res = vm_bc
        .call("walk_sum", vec![Value::Int(10)])
        .expect("Bytecode walk_sum oracle");

    assert_eq!(clif_res, Value::Int(55));
    assert_eq!(clif_res, bc_res);
    assert!(
        vm_clif.cranelift_entry_count >= 1,
        "walk_sum must enter Cranelift natively before deopting"
    );
}

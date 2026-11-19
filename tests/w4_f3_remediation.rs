//! Independent 1:1 RED/GREEN remediation suite for TASK-W4-F3-REMEDIATE (F3-A..F3-F).

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use lust::ast::Span;
use lust::bytecode::compiler::Compiler;
use lust::bytecode::Value;
use lust::config::LustConfig;
use lust::data::layout::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1, TargetAbi};
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{
    reset_native_instrumentation_counters, CraneliftJitCompiler, MIR_STATUS_ERR_OUT_OF_BOUNDS,
};
use lust::mir::ir::*;
use lust::mir::lowering::AstLowering;
use lust::mir::runtime_helpers::{
    enter_mir_context, lust_mir_alloc_array_typed, lust_mir_alloc_struct, lust_mir_alloc_tuple,
    lust_mir_array_get, lust_mir_array_len, lust_mir_array_set, lust_mir_get_status,
    lust_mir_set_status, MirAllocationKind, MirContext, MIR_ARRAY_ELEM_TAG_INT,
};
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{VmBackend, VM};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn setup_vms(source: &str, module_name: &str) -> (VM, VM) {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("Tokenization must succeed");
    let items = Parser::new(tokens).parse().expect("Parsing must succeed");

    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let struct_defs = checker.struct_definitions();

    let struct_def_refs: Vec<&lust::ast::StructDef> = struct_defs.values().collect();
    let mut module = MirModule::new(module_name.into());
    for item in &items {
        if let lust::ast::ItemKind::Function(func_def) = &item.kind {
            let mir_fn = AstLowering::lower_function_with_structs(func_def, &struct_def_refs)
                .expect("MIR lowering must succeed")
                .0;
            module.add_function(mir_fn);
        }
    }

    let mut vm_clif = VM::new();
    vm_clif.set_backend(VmBackend::Cranelift);
    vm_clif.load_mir_module(&module).expect("Load MIR module");
    vm_clif.register_structs(&struct_defs);

    let mut vm_bc = VM::new();
    vm_bc.register_structs(&struct_defs);
    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    if let Ok(bc_functions) = compiler.compile_module(&items) {
        vm_bc.load_functions(bc_functions);
    } else {
        vm_bc.load_mir_module(&module).expect("Load MIR module for bc");
    }
    vm_bc.set_backend(VmBackend::Bytecode);

    (vm_clif, vm_bc)
}

fn setup_clif_vm_only(source: &str, module_name: &str) -> VM {
    setup_vms(source, module_name).0
}

fn run_isolated_child(test_name: &str, env_var: &str) {
    let exe = std::env::current_exe().expect("current_exe");
    let mut child = std::process::Command::new(exe)
        .arg("--exact")
        .arg(test_name)
        .arg("--nocapture")
        .env(env_var, "1")
        .spawn()
        .expect("Spawn isolated child process");

    let start = Instant::now();
    let timeout = Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            assert!(
                status.success(),
                "Isolated child '{test_name}' failed/crashed with status: {status:?}"
            );
            break;
        }
        if start.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            panic!("Isolated child '{test_name}' timed out after {:?}", timeout);
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

// =========================================================================
// F3-A: Dynamic first element, nested aggregates, heterogeneous & empty []
// =========================================================================

#[test]
fn test_f3_a1_dynamic_float_add_first_elem_array_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    let src = r#"
        function f3_a1_expr(a: float, b: float): Array<float>
            return [a + b, 3.0]
        end

        function f3_a1_local(a: float, b: float): Array<float>
            local s = a + b
            return [s, 3.0]
        end
    "#;
    let (mut vm_clif, mut vm_bc) = setup_vms(src, "f3_a1");

    for fn_name in ["f3_a1_expr", "f3_a1_local"] {
        let e0 = vm_clif.cranelift_entry_count;
        let res_clif = vm_clif
            .call(fn_name, vec![Value::Float(1.5), Value::Float(2.75)])
            .expect("Cranelift call must succeed");
        let res_bc = vm_bc
            .call(fn_name, vec![Value::Float(1.5), Value::Float(2.75)])
            .expect("Bytecode call must succeed");

        assert!(
            vm_clif.cranelift_entry_count > e0,
            "{fn_name} must execute natively via Cranelift"
        );
        assert_eq!(
            res_clif,
            Value::Array(Rc::new(RefCell::new(vec![
                Value::Float(4.25),
                Value::Float(3.0)
            ])))
        );
        assert_eq!(res_clif, res_bc, "Oracle mismatch for {fn_name}");
    }
}

#[test]
fn test_f3_a2_dynamic_string_concat_first_elem_array() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    let src = r#"
        function f3_a2(a: string, b: string): Array<string>
            return [a + b, "tail"]
        end
    "#;
    let (mut vm_clif, mut vm_bc) = setup_vms(src, "f3_a2");

    let e0 = vm_clif.cranelift_entry_count;
    let res_clif = vm_clif
        .call(
            "f3_a2",
            vec![Value::string("hello_"), Value::string("world")],
        )
        .expect("Cranelift string concat array call must succeed");
    let res_bc = vm_bc
        .call(
            "f3_a2",
            vec![Value::string("hello_"), Value::string("world")],
        )
        .expect("Bytecode string concat array call must succeed");

    assert!(
        vm_clif.cranelift_entry_count > e0,
        "f3_a2 must execute natively via Cranelift"
    );
    assert_eq!(
        res_clif,
        Value::Array(Rc::new(RefCell::new(vec![
            Value::string("hello_world"),
            Value::string("tail")
        ])))
    );
    assert_eq!(res_clif, res_bc);
}

#[test]
fn test_f3_a3_dynamic_float_neg_and_call_first_elem_array() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    let src = r#"
        function double_f(x: float): float
            return x * 2.0
        end

        function f3_a3_neg(x: float): Array<float>
            return [-x, 1.0]
        end

        function f3_a3_call(x: float): Array<float>
            return [double_f(x), 1.0]
        end
    "#;
    let (mut vm_clif, mut vm_bc) = setup_vms(src, "f3_a3");

    let e0 = vm_clif.cranelift_entry_count;
    let neg_clif = vm_clif
        .call("f3_a3_neg", vec![Value::Float(2.5)])
        .expect("f3_a3_neg clif");
    let neg_bc = vm_bc
        .call("f3_a3_neg", vec![Value::Float(2.5)])
        .expect("f3_a3_neg bc");
    assert!(vm_clif.cranelift_entry_count > e0);
    assert_eq!(
        neg_clif,
        Value::Array(Rc::new(RefCell::new(vec![
            Value::Float(-2.5),
            Value::Float(1.0)
        ])))
    );
    assert_eq!(neg_clif, neg_bc);

    let e1 = vm_clif.cranelift_entry_count;
    let call_clif = vm_clif
        .call("f3_a3_call", vec![Value::Float(2.5)])
        .expect("f3_a3_call clif");
    let call_bc = vm_bc
        .call("f3_a3_call", vec![Value::Float(2.5)])
        .expect("f3_a3_call bc");
    assert!(vm_clif.cranelift_entry_count > e1);
    assert_eq!(
        call_clif,
        Value::Array(Rc::new(RefCell::new(vec![
            Value::Float(5.0),
            Value::Float(1.0)
        ])))
    );
    assert_eq!(call_clif, call_bc);
}

#[test]
fn test_f3_a4a_nested_array_literal_element_no_ptr_leak() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    let src = r#"
        function f3_a4a_ret(): Array<Array<int>>
            return [[10, 20], [30, 40]]
        end

        function f3_a4a_local(): int
            local a = [[10, 20], [30, 40]]
            return len(a)
        end
    "#;
    let (mut vm_clif, mut vm_bc) = setup_vms(src, "f3_a4a");

    let e0 = vm_clif.cranelift_entry_count;
    let res_clif = vm_clif.call("f3_a4a_ret", vec![]).expect("f3_a4a_ret clif");
    let res_bc = vm_bc.call("f3_a4a_ret", vec![]).expect("f3_a4a_ret bc");

    assert_eq!(
        vm_clif.cranelift_entry_count, e0,
        "Nested Array<Array<int>> must fall back pre-entry instead of leaking raw pointers"
    );
    assert_eq!(res_clif, res_bc);
    let Value::Array(outer) = res_clif else {
        panic!("Expected Value::Array");
    };
    assert!(
        matches!(&outer.borrow()[0], Value::Array(inner) if *inner.borrow() == vec![Value::Int(10), Value::Int(20)]),
        "Outer element must remain Value::Array, not leaked Value::Int pointer"
    );

    let e1 = vm_clif.cranelift_entry_count;
    let len_clif = vm_clif.call("f3_a4a_local", vec![]).expect("f3_a4a_local clif");
    let len_bc = vm_bc.call("f3_a4a_local", vec![]).expect("f3_a4a_local bc");
    assert_eq!(
        vm_clif.cranelift_entry_count, e1,
        "Function with local nested array literal [[10, 20], [30, 40]] must fall back pre-entry"
    );
    assert_eq!(len_clif, Value::Int(2));
    assert_eq!(len_clif, len_bc);
}

#[test]
fn test_f3_a4b_nested_tuple_literal_element_no_ptr_leak() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    let src = r#"
        function f3_a4b_ret(): Array<(int, int)>
            return [(1, 2), (3, 4)]
        end

        function f3_a4b_local(): int
            local a = [(1, 2), (3, 4)]
            return len(a)
        end
    "#;
    let (mut vm_clif, mut vm_bc) = setup_vms(src, "f3_a4b");

    let e0 = vm_clif.cranelift_entry_count;
    let res_clif = vm_clif.call("f3_a4b_ret", vec![]).expect("f3_a4b_ret clif");
    let res_bc = vm_bc.call("f3_a4b_ret", vec![]).expect("f3_a4b_ret bc");

    assert_eq!(
        vm_clif.cranelift_entry_count, e0,
        "Nested Array<(int, int)> must fall back pre-entry instead of leaking raw pointers"
    );
    assert_eq!(res_clif, res_bc);
    let Value::Array(outer) = res_clif else {
        panic!("Expected Value::Array");
    };
    assert!(
        matches!(&outer.borrow()[0], Value::Tuple(t) if **t == vec![Value::Int(1), Value::Int(2)]),
        "Outer element must remain Value::Tuple, not leaked Value::Int pointer"
    );

    let e1 = vm_clif.cranelift_entry_count;
    let len_clif = vm_clif.call("f3_a4b_local", vec![]).expect("f3_a4b_local clif");
    let len_bc = vm_bc.call("f3_a4b_local", vec![]).expect("f3_a4b_local bc");
    assert_eq!(
        vm_clif.cranelift_entry_count, e1,
        "Function with local nested tuple array literal must fall back pre-entry"
    );
    assert_eq!(len_clif, Value::Int(2));
    assert_eq!(len_clif, len_bc);
}

#[test]
fn test_f3_a4c_nested_struct_literal_element_no_ptr_leak() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    let src_ret = r#"
        struct BoxInt
            v: int
        end

        function f3_a4c_ret(): Array<BoxInt>
            return [BoxInt { v = 7 }, BoxInt { v = 9 }]
        end
    "#;
    let (mut vm_clif, mut vm_bc) = setup_vms(src_ret, "f3_a4c_ret");

    let e0 = vm_clif.cranelift_entry_count;
    let res_clif = vm_clif.call("f3_a4c_ret", vec![]).expect("f3_a4c_ret clif");
    let res_bc = vm_bc.call("f3_a4c_ret", vec![]).expect("f3_a4c_ret bc");

    assert_eq!(
        vm_clif.cranelift_entry_count, e0,
        "Nested Array<BoxInt> must fall back pre-entry instead of leaking raw pointers"
    );
    let Value::Array(outer_bc) = res_bc else {
        panic!("Expected Value::Array from bc");
    };
    assert!(
        matches!(&outer_bc.borrow()[0], Value::Struct(s) if *s.fields.borrow() == vec![Value::Int(7)]),
        "Bytecode oracle must produce Value::Struct"
    );
    assert!(
        matches!(res_clif, Value::Array(_)),
        "Fallback must produce Value::Array without native entry"
    );

    let src_local = r#"
        struct BoxInt
            v: int
        end

        function f3_a4c_local(): int
            local a = [BoxInt { v = 7 }, BoxInt { v = 9 }]
            return len(a)
        end
    "#;
    let mut vm_local = setup_clif_vm_only(src_local, "f3_a4c_local");
    let e1 = vm_local.cranelift_entry_count;
    let len_clif = vm_local
        .call("f3_a4c_local", vec![])
        .expect("f3_a4c_local clif");
    assert_eq!(
        vm_local.cranelift_entry_count, e1,
        "Function with local nested struct array literal must fall back pre-entry"
    );
    assert_eq!(len_clif, Value::Int(2));
}

#[test]
fn test_f3_a5_heterogeneous_array_float_int_fallback_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    let src = r#"
        function f3_a5(): float
            local a = [1.5, 2]
            return a[1]
        end
    "#;
    let mut vm_clif = setup_clif_vm_only(src, "f3_a5");

    let e0 = vm_clif.cranelift_entry_count;
    let res_clif = vm_clif.call("f3_a5", vec![]).expect("f3_a5 fallback");

    assert_eq!(
        vm_clif.cranelift_entry_count, e0,
        "Heterogeneous [1.5, 2] must fall back before Cranelift native entry"
    );
    assert_eq!(res_clif, Value::Int(2));
}

#[test]
fn test_f3_a6_heterogeneous_array_int_string_fallback_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    let src = r#"
        function f3_a6(): Array<int>
            return [10, "hello"]
        end
    "#;
    let mut vm_clif = setup_clif_vm_only(src, "f3_a6");

    let e0 = vm_clif.cranelift_entry_count;
    let res_clif = vm_clif.call("f3_a6", vec![]).expect("f3_a6 fallback");

    assert_eq!(
        vm_clif.cranelift_entry_count, e0,
        "Heterogeneous [10, \"hello\"] must fall back before Cranelift native entry"
    );
    assert_eq!(
        res_clif,
        Value::Array(Rc::new(RefCell::new(vec![
            Value::Int(10),
            Value::string("hello")
        ])))
    );
}

#[test]
fn test_f3_a7_empty_array_float_and_string_annotation_and_return() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    let src = r#"
        function f3_a7_float_annot(): Array<float>
            local a: Array<float> = []
            return a
        end

        function f3_a7_float_ret(): Array<float>
            return []
        end

        function f3_a7_str_annot(): Array<string>
            local s: Array<string> = []
            return s
        end

        function f3_a7_str_ret(): Array<string>
            return []
        end
    "#;
    let (mut vm_clif, mut vm_bc) = setup_vms(src, "f3_a7");

    for fn_name in [
        "f3_a7_float_annot",
        "f3_a7_float_ret",
        "f3_a7_str_annot",
        "f3_a7_str_ret",
    ] {
        let e0 = vm_clif.cranelift_entry_count;
        let res_clif = vm_clif.call(fn_name, vec![]).expect(fn_name);
        let res_bc = vm_bc.call(fn_name, vec![]).expect(fn_name);
        assert!(
            vm_clif.cranelift_entry_count > e0,
            "{fn_name} must execute natively via Cranelift"
        );
        assert_eq!(res_clif, Value::Array(Rc::new(RefCell::new(vec![]))));
        assert_eq!(res_clif, res_bc);
    }
}

// =========================================================================
// F3-B: Tuple/Struct metadata validation, param packing, nested fields, mixed tuples
// =========================================================================

#[test]
fn test_f3_b1_tuple_wildcard_metadata_rejected_on_decode() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    let ptr = lust_mir_alloc_tuple(2);
    assert!(!ptr.is_null());
    unsafe {
        *ptr.add(2) = 10;
        *ptr.add(3) = 1.5f64.to_bits() as i64;
    }

    let err = lust::vm::cranelift_dispatch::validate_and_decode_tuple(
        &ctx,
        ptr as i64,
        &[MirType::Int, MirType::Float],
    )
    .expect_err("Wildcard empty Tuple(vec![]) metadata must be rejected on decode");
    assert!(
        err.to_string().contains("mismatch") || err.to_string().contains("Tuple"),
        "Unexpected error: {err}"
    );
    ctx.clear_to_baselines(0, 0);
}

#[test]
fn test_f3_b2_struct_wildcard_metadata_rejected_on_decode() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    let ptr = lust_mir_alloc_struct(1);
    assert!(!ptr.is_null());
    unsafe {
        *ptr.add(2) = 42;
    }

    let layout = Rc::new(lust::bytecode::StructLayout::new(
        "BoxedInt".to_string(),
        vec![Rc::new("x".to_string())],
        vec![lust::bytecode::FieldStorage::Strong],
        vec![lust::ast::Type::new(
            lust::ast::TypeKind::Int,
            Span::dummy(),
        )],
        vec![None],
    ));

    let err = lust::vm::cranelift_dispatch::validate_and_decode_struct(
        &ctx,
        &[],
        ptr as i64,
        "BoxedInt",
        Some(&layout),
    )
    .expect_err("Wildcard empty Struct(\"\") metadata must be rejected on decode");
    assert!(
        err.to_string().contains("mismatch") || err.to_string().contains("Struct"),
        "Unexpected error: {err}"
    );
    ctx.clear_to_baselines(0, 0);
}

#[test]
fn test_f3_b3_struct_param_packing_mismatched_name_and_arity_no_oob() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    let src = r#"
        struct ExpectedPair
            a: int,
            b: int
        end

        struct WrongSingle
            x: int
        end

        struct WrongPair
            x: int,
            y: int
        end

        function read_expected_b(p: ExpectedPair): int
            return p.b
        end
    "#;
    let mut vm = setup_clif_vm_only(src, "f3_b3");

    let wrong_single = vm
        .instantiate_struct_ordered("WrongSingle", vec![Value::Int(11)])
        .expect("Instantiate WrongSingle");
    let wrong_pair = vm
        .instantiate_struct_ordered("WrongPair", vec![Value::Int(11), Value::Int(22)])
        .expect("Instantiate WrongPair");

    let ok_pair = vm
        .instantiate_struct_ordered("ExpectedPair", vec![Value::Int(1), Value::Int(2)])
        .unwrap();
    let e_ok = vm.cranelift_entry_count;
    let ok_res = vm
        .call("read_expected_b", vec![ok_pair.clone()])
        .expect("Valid ExpectedPair must succeed");
    assert!(
        vm.cranelift_entry_count > e_ok,
        "Valid ExpectedPair must execute natively via Cranelift"
    );
    assert_eq!(ok_res, Value::Int(2));

    // 1. Wrong name + smaller arity must fall back pre-entry (no native entry, no OOB read)
    let e0 = vm.cranelift_entry_count;
    let _ = vm.call("read_expected_b", vec![wrong_single]);
    assert_eq!(
        vm.cranelift_entry_count, e0,
        "Mismatched struct name/arity must fall back before Cranelift native entry"
    );
    assert_eq!(vm.mir_context.buffer_count(), 0);

    // 2. Wrong name + matching arity must also fall back pre-entry
    let e1 = vm.cranelift_entry_count;
    let _ = vm.call("read_expected_b", vec![wrong_pair]);
    assert_eq!(
        vm.cranelift_entry_count, e1,
        "Mismatched struct name must fall back before Cranelift native entry"
    );
    assert_eq!(vm.mir_context.buffer_count(), 0);

    // 3. Unknown-typed MIR param receiving a Struct must fall back pre-entry
    let sig_unk = FunctionSignature {
        name: "unk_struct_param".into(),
        param_types: vec![MirType::Unknown],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut fn_unk = MirFunction::new(sig_unk, Span::dummy());
    fn_unk.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Constant(MirConstant::Int(99))),
        Span::dummy(),
    ));
    vm.load_mir_function(fn_unk).unwrap();
    let e2 = vm.cranelift_entry_count;
    let res = vm.call("unk_struct_param", vec![ok_pair]).unwrap();
    assert_eq!(vm.cranelift_entry_count, e2);
    assert_eq!(res, Value::Int(99));
}

#[test]
fn test_f3_b4_struct_with_array_field_rejected_pre_entry_no_ptr_leak() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    let src = r#"
        struct Holder
            arr: Array<int>
        end

        function make_holder(): Holder
            return Holder { arr = [1, 2, 3] }
        end

        function id_holder(h: Holder): Holder
            return h
        end
    "#;
    let (mut vm_clif, mut vm_bc) = setup_vms(src, "f3_b4");

    let e0 = vm_clif.cranelift_entry_count;
    let _made_clif = vm_clif.call("make_holder", vec![]).expect("make_holder");
    let made = vm_bc.call("make_holder", vec![]).expect("make_holder bc");
    assert_eq!(
        vm_clif.cranelift_entry_count, e0,
        "Struct with Array field must be rejected pre-entry in Cranelift"
    );
    let Value::Struct(ref s) = made else {
        panic!("Expected Value::Struct");
    };
    assert!(
        matches!(&s.fields.borrow()[0], Value::Array(arr) if *arr.borrow() == vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
        "Holder.arr must be Value::Array, not leaked raw pointer integer"
    );

    let e1 = vm_clif.cranelift_entry_count;
    let echoed = vm_clif.call("id_holder", vec![made.clone()]).expect("id_holder");
    assert_eq!(
        vm_clif.cranelift_entry_count, e1,
        "id_holder taking Struct with Array field must be rejected pre-entry in Cranelift"
    );
    assert_eq!(echoed, made);
}

#[test]
fn test_f3_b5_mixed_tuple_literal_lowering_and_execution_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    let src = r#"
        function f3_b5(a: int, b: float, c: bool, d: string): (int, float, bool, string)
            return (a, b, c, d)
        end
    "#;
    let (mut vm_clif, mut vm_bc) = setup_vms(src, "f3_b5");

    let args = vec![
        Value::Int(42),
        Value::Float(3.5),
        Value::Bool(true),
        Value::string("ok"),
    ];
    let e0 = vm_clif.cranelift_entry_count;
    let res_clif = vm_clif.call("f3_b5", args.clone()).expect("f3_b5 clif");
    let res_bc = vm_bc.call("f3_b5", args).expect("f3_b5 bc");

    assert!(
        vm_clif.cranelift_entry_count > e0,
        "Mixed scalar tuple must execute natively via Cranelift"
    );
    assert_eq!(
        res_clif,
        Value::Tuple(Rc::new(vec![
            Value::Int(42),
            Value::Float(3.5),
            Value::Bool(true),
            Value::string("ok"),
        ]))
    );
    assert_eq!(res_clif, res_bc);
}

// =========================================================================
// F3-C: C-flat struct allocation without [len, cap] header overwrite
// =========================================================================

#[test]
fn test_f3_c1_cflat_struct_aggregate_preserves_fields_and_no_header_overwrite() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);

    let logical_pair = LogicalLayoutV1::new_with_abi(
        9992,
        "CFlatPair",
        vec![
            LogicalFieldDesc::new(0, "a", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(1, "b", CanonicalTypeTag::Int64),
        ],
        TargetAbi::aarch64(),
    );
    let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift");
    compiler.register_struct_layouts(&[MirStructLayout::from(&logical_pair)]);

    let sig = FunctionSignature {
        name: "make_cflat_pair".into(),
        param_types: vec![],
        return_type: MirType::Struct("CFlatPair".into()),
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let s_loc = func.add_local(
        Some("p".into()),
        MirType::Struct("CFlatPair".into()),
        false,
        Span::dummy(),
    );
    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(s_loc),
        Rvalue::Aggregate(
            MirType::Struct("CFlatPair".into()),
            vec![
                Operand::Constant(MirConstant::Int(123)),
                Operand::Constant(MirConstant::Int(456)),
            ],
        ),
        Span::dummy(),
    ));
    func.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(s_loc))),
        Span::dummy(),
    ));

    let code = compiler.compile_function(&func).expect("Compile CFlatPair");
    let f: extern "C" fn() -> i64 = unsafe { std::mem::transmute(code) };
    let ptr = f() as *mut i64;
    assert!(!ptr.is_null());

    // Exact C-flat memory footprint: 16 bytes (2 words), NOT 18 words (144 bytes)
    let rec = ctx
        .get_active_allocation_record(ptr as usize)
        .expect("CFlatStruct allocation must be tracked in MirContext");
    assert_eq!(rec.word_capacity, 2, "CFlatPair is 16 bytes = 2 words");
    assert_eq!(ctx.memory_used(), 16, "CFlatPair must charge 16 bytes");
    unsafe {
        assert_eq!(*ptr, 123, "Field 0 at byte_offset 0");
        assert_eq!(*ptr.add(1), 456, "Field 1 at byte_offset 8");
    }

    // Array helpers & VM headered struct decoder must reject CFlatStruct pointers
    lust_mir_set_status(0);
    assert_eq!(lust_mir_array_len(ptr), 0);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);

    let layout = Rc::new(lust::bytecode::StructLayout::new(
        "CFlatPair".to_string(),
        vec![Rc::new("a".to_string()), Rc::new("b".to_string())],
        vec![
            lust::bytecode::FieldStorage::Strong,
            lust::bytecode::FieldStorage::Strong,
        ],
        vec![
            lust::ast::Type::new(lust::ast::TypeKind::Int, Span::dummy()),
            lust::ast::Type::new(lust::ast::TypeKind::Int, Span::dummy()),
        ],
        vec![None, None],
    ));
    assert!(
        lust::vm::cranelift_dispatch::validate_and_decode_struct(
            &ctx,
            &[],
            ptr as i64,
            "CFlatPair",
            Some(&layout),
        )
        .is_err(),
        "CFlatStruct must be rejected by headered VM struct decoder"
    );

    ctx.clear_to_baselines(0, 0);
}

// =========================================================================
// F3-D: Pointer validation in array runtime helpers & MirUnaryOp::Len / array.len
// =========================================================================

#[test]
fn test_f3_d1_invalid_pointer_in_array_helpers_isolated_child_watchdog() {
    if std::env::var("LUST_F3_D1_CHILD").ok().as_deref() == Some("1") {
        let ctx = MirContext::new();
        let _guard = enter_mir_context(&ctx);

        let bogus = 0x1000usize as *mut i64;
        lust_mir_set_status(0);
        assert_eq!(lust_mir_array_len(bogus), 0);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);

        lust_mir_set_status(0);
        assert_eq!(lust_mir_array_get(bogus, 0), 0);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);

        lust_mir_set_status(0);
        lust_mir_array_set(bogus, 0, 99);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);

        // Null pointer
        lust_mir_set_status(0);
        assert_eq!(lust_mir_array_len(core::ptr::null()), 0);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);

        // Wrong kind (Tuple / Struct)
        let tup_ptr = lust_mir_alloc_tuple(2);
        assert!(!tup_ptr.is_null());
        lust_mir_set_status(0);
        assert_eq!(lust_mir_array_len(tup_ptr), 0);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);

        // Valid array OOB index sets MIR_STATUS_ERR_OUT_OF_BOUNDS
        let arr_ptr = lust_mir_alloc_array_typed(2, MIR_ARRAY_ELEM_TAG_INT);
        assert!(!arr_ptr.is_null());
        lust_mir_set_status(0);
        assert_eq!(lust_mir_array_get(arr_ptr, 5), 0);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);

        ctx.clear_to_baselines(0, 0);
        return;
    }

    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    run_isolated_child(
        "test_f3_d1_invalid_pointer_in_array_helpers_isolated_child_watchdog",
        "LUST_F3_D1_CHILD",
    );
}

#[test]
fn test_f3_d2_array_len_builtin_and_method_with_int_arg_isolated_child_watchdog() {
    if std::env::var("LUST_F3_D2_CHILD").ok().as_deref() == Some("1") {
        reset_native_instrumentation_counters();

        // 1. Direct CraneliftJitCompiler must reject MirUnaryOp::Len on Int
        let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift");
        let sig_un = FunctionSignature {
            name: "bad_unary_len".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut fn_un = MirFunction::new(sig_un, Span::dummy());
        let p = fn_un.params[0];
        let tmp = fn_un.add_local(None, MirType::Int, true, Span::dummy());
        fn_un.blocks[0].statements.push(Statement::Assign(
            Place::Local(tmp),
            Rvalue::UnaryOp(MirUnaryOp::Len, Operand::Copy(Place::Local(p))),
            Span::dummy(),
        ));
        fn_un.blocks[0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(tmp))),
            Span::dummy(),
        ));
        assert!(
            compiler.compile_function(&fn_un).is_err(),
            "MirUnaryOp::Len on Int must be rejected at compile time"
        );

        // 2. Direct CraneliftJitCompiler must reject array.len(Int)
        let sig_call = FunctionSignature {
            name: "bad_array_len_call".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut fn_call = MirFunction::new(sig_call, Span::dummy());
        let p_c = fn_call.params[0];
        let dst_c = fn_call.add_local(None, MirType::Int, true, Span::dummy());
        let succ_bb = fn_call.new_block();
        fn_call.blocks[0].terminator = Some(Terminator::Call {
            callee: "array.len".into(),
            args: vec![Operand::Copy(Place::Local(p_c))],
            destination: Place::Local(dst_c),
            success: succ_bb,
            error: None,
            span: Span::dummy(),
        });
        fn_call.blocks[succ_bb.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(dst_c))),
            Span::dummy(),
        ));
        assert!(
            compiler.compile_function(&fn_call).is_err(),
            "array.len(Int) must be rejected at compile time"
        );

        // 3. VM::call must reject/fallback pre-entry on len(0x1000) and x.len() with int arg (no SIGSEGV)
        let src = r#"
            function bad_builtin_len(x: int): int
                return len(x)
            end

            function bad_method_len(x: int): int
                return x.len()
            end
        "#;
        let mut vm = setup_clif_vm_only(src, "f3_d2");
        let e0 = vm.cranelift_entry_count;
        let _ = vm.call("bad_builtin_len", vec![Value::Int(0x1000)]);
        assert_eq!(vm.cranelift_entry_count, e0);

        let e1 = vm.cranelift_entry_count;
        let _ = vm.call("bad_method_len", vec![Value::Int(0x1000)]);
        assert_eq!(vm.cranelift_entry_count, e1);
        return;
    }

    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    run_isolated_child(
        "test_f3_d2_array_len_builtin_and_method_with_int_arg_isolated_child_watchdog",
        "LUST_F3_D2_CHILD",
    );
}

// =========================================================================
// F3-E: MirContext scope & baseline cleanup isolation
// =========================================================================

#[test]
fn test_f3_e1_clear_to_baseline_with_positive_handles_and_no_new_handle() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);

    let outer_buf = lust_mir_alloc_array_typed(2, MIR_ARRAY_ELEM_TAG_INT);
    assert!(!outer_buf.is_null());

    let _outer_h = ctx.register_handle(Value::string("outer"));
    let b_h = ctx.handle_count();
    assert_eq!(b_h, 1);

    let inner_buf = lust_mir_alloc_array_typed(2, MIR_ARRAY_ELEM_TAG_INT);
    assert!(!inner_buf.is_null());

    ctx.clear_to_baseline(b_h);

    assert!(
        ctx.get_allocation_record(outer_buf as usize).is_some(),
        "outer_buf must survive clear_to_baseline(b_h)"
    );
    assert!(
        ctx.get_allocation_record(inner_buf as usize).is_none(),
        "inner_buf allocated at handle_epoch == b_h without new handle must be freed"
    );
    assert_eq!(ctx.buffer_count(), 1);
    ctx.clear_to_baselines(0, 0);
}

#[test]
fn test_f3_e2_register_allocation_no_scope_collision_or_double_free() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let ext_ptr = Box::into_raw(Box::new([1i64, 1i64, 42i64])) as *mut i64;
    {
        let ctx = MirContext::new();
        ctx.register_allocation(ext_ptr, 3, MirAllocationKind::Array(MirType::Int));
        assert_eq!(
            ctx.buffer_count(),
            0,
            "register_allocation must NOT push caller-owned Box::into_raw into allocated_buffers"
        );
        assert!(ctx.get_active_allocation_record(ext_ptr as usize).is_some());

        {
            let _inner = ctx.enter();
            let inner_owned = lust_mir_alloc_array_typed(2, MIR_ARRAY_ELEM_TAG_INT);
            assert!(!inner_owned.is_null());
            assert!(
                ctx.get_active_allocation_record(ext_ptr as usize).is_none(),
                "Outer caller-owned allocation must be hidden inside inner ctx.enter()"
            );
            assert!(
                ctx.get_active_allocation_record(inner_owned as usize)
                    .is_some(),
                "Inner owned allocation must be active inside inner ctx.enter()"
            );
        }

        assert!(
            ctx.get_active_allocation_record(ext_ptr as usize).is_some(),
            "Outer caller-owned allocation must be active again after inner guard drops"
        );
        // Dropping ctx here must NOT free ext_ptr (zero double-free when manually freed below)
    }
    unsafe {
        drop(Box::from_raw(ext_ptr as *mut [i64; 3]));
    }
}

#[test]
fn test_f3_e3_shared_handle_epoch_nested_enter_preserves_outer_and_frees_inner() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let ctx = MirContext::new();
    let _outer_guard = enter_mir_context(&ctx);

    let _h = ctx.register_handle(Value::string("shared"));
    let b_h = ctx.handle_count();
    assert_eq!(b_h, 1);

    let outer_buf = lust_mir_alloc_array_typed(2, MIR_ARRAY_ELEM_TAG_INT);
    assert!(!outer_buf.is_null());

    {
        let _inner = ctx.enter();
        let inner_kept = lust_mir_alloc_array_typed(2, MIR_ARRAY_ELEM_TAG_INT);
        assert!(!inner_kept.is_null());

        let _h2 = ctx.register_handle(Value::string("inner_base"));
        let b_h2 = ctx.handle_count();
        assert_eq!(b_h2, 2);

        let inner_buf = lust_mir_alloc_array_typed(2, MIR_ARRAY_ELEM_TAG_INT);
        assert!(!inner_buf.is_null());

        ctx.clear_to_baseline(b_h2);

        assert!(
            ctx.get_allocation_record(inner_buf as usize).is_none(),
            "inner_buf sharing handle_epoch == b_h2 must be freed"
        );
        assert!(
            ctx.get_allocation_record(inner_kept as usize).is_some(),
            "inner_kept strictly predating b_h2 must survive inner clear_to_baseline(b_h2)"
        );

        ctx.clear_to_baseline(b_h);
        assert!(
            ctx.get_allocation_record(inner_kept as usize).is_none(),
            "inner_kept sharing handle_epoch == b_h must be freed by clear_to_baseline(b_h)"
        );
        assert!(
            ctx.get_allocation_record(outer_buf as usize).is_some(),
            "outer_buf sharing handle_epoch == b_h must survive inner clear_to_baseline(b_h)"
        );
    }

    assert!(
        ctx.get_active_allocation_record(outer_buf as usize)
            .is_some(),
        "outer_buf must be active after inner scope exits"
    );
    ctx.clear_to_baselines(0, 0);
}

#[test]
fn test_f3_e4_caller_owned_outer_survives_partial_and_nested_inner_cleanup() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let ext_ptr = Box::into_raw(Box::new([1i64, 1i64, 7i64, 0i64])) as *mut i64;
    {
        let ctx = MirContext::new();
        let _outer_guard = enter_mir_context(&ctx);

        let owned_a = lust_mir_alloc_array_typed(2, MIR_ARRAY_ELEM_TAG_INT);
        let owned_b = lust_mir_alloc_array_typed(2, MIR_ARRAY_ELEM_TAG_INT);
        assert!(!owned_a.is_null() && !owned_b.is_null());

        // Caller-owned buffer registered *after* owned_b
        ctx.register_allocation(ext_ptr, 4, MirAllocationKind::Array(MirType::Int));

        // Case A: Top-level partial cleanup draining owned_b and owned_c
        let owned_c = lust_mir_alloc_array_typed(2, MIR_ARRAY_ELEM_TAG_INT);
        assert!(!owned_c.is_null());
        ctx.clear_to_baselines(0, 1);

        assert!(
            ctx.get_allocation_record(owned_a as usize).is_some(),
            "owned_a must survive partial baseline (0, 1)"
        );
        assert!(
            ctx.get_allocation_record(owned_b as usize).is_none(),
            "owned_b must be freed by partial baseline (0, 1)"
        );
        assert!(
            ctx.get_allocation_record(owned_c as usize).is_none(),
            "owned_c must be freed by partial baseline (0, 1)"
        );
        assert!(
            ctx.get_active_allocation_record(ext_ptr as usize).is_some(),
            "Caller-owned ext_outer registered after owned_b must survive partial cleanup"
        );

        // Case B: Nested ctx.enter() cleanup
        {
            let _inner = ctx.enter();
            let owned_d = lust_mir_alloc_array_typed(2, MIR_ARRAY_ELEM_TAG_INT);
            assert!(!owned_d.is_null());
            assert!(
                ctx.get_active_allocation_record(ext_ptr as usize).is_none(),
                "ext_outer must be hidden inside _inner"
            );

            ctx.clear_to_baselines(0, 0);

            assert!(
                ctx.get_allocation_record(owned_d as usize).is_none(),
                "owned_d must be freed by inner clear_to_baselines(0, 0)"
            );
            assert!(
                ctx.get_allocation_record(owned_a as usize).is_some(),
                "outer owned_a must survive inner clear_to_baselines(0, 0)"
            );
            assert!(
                ctx.get_active_allocation_record(ext_ptr as usize).is_none(),
                "ext_outer must remain hidden inside _inner"
            );
        }

        assert!(
            ctx.get_active_allocation_record(ext_ptr as usize).is_some(),
            "ext_outer must be active again after _inner drops"
        );
    }

    // Manual free after drop(ctx) must succeed with zero double-free
    unsafe {
        drop(Box::from_raw(ext_ptr as *mut [i64; 4]));
    }
}

#[test]
fn test_f3_d3_simd_chk_block_pattern_invalid_ptr_rejected_isolated_child() {
    if std::env::var("LUST_F3_D3_CHILD").ok().as_deref() == Some("1") {
        let ctx = MirContext::new();
        let _guard = enter_mir_context(&ctx);

        let sig = FunctionSignature {
            name: "simd_chk_sum".into(),
            param_types: vec![MirType::Array(Box::new(MirType::Int)), MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut func = MirFunction::new(sig, Span::dummy());
        let a_loc = func.params[0];
        let n_loc = func.params[1];
        let acc_loc = func.add_local(Some("acc".into()), MirType::Int, false, Span::dummy());
        let i_loc = func.add_local(Some("i".into()), MirType::Int, false, Span::dummy());
        let cond_loc = func.add_local(Some("cond".into()), MirType::Bool, true, Span::dummy());
        let elem_loc = func.add_local(Some("elem".into()), MirType::Int, true, Span::dummy());

        let bb_hdr = func.new_block();
        let bb_body = func.new_block();
        let bb_exit = func.new_block();

        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(acc_loc),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));
        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(i_loc),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));
        func.blocks[0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

        func.blocks[bb_hdr.0].statements.push(Statement::Assign(
            Place::Local(cond_loc),
            Rvalue::BinaryOp(
                MirBinaryOp::Lt,
                Operand::Copy(Place::Local(i_loc)),
                Operand::Copy(Place::Local(n_loc)),
            ),
            Span::dummy(),
        ));
        func.blocks[bb_hdr.0].terminator = Some(Terminator::Cond {
            cond: Operand::Copy(Place::Local(cond_loc)),
            then_block: bb_body,
            else_block: bb_exit,
            span: Span::dummy(),
        });

        func.blocks[bb_body.0].statements.push(Statement::Assign(
            Place::Local(elem_loc),
            Rvalue::Use(Operand::Copy(Place::Index(
                Box::new(Place::Local(a_loc)),
                i_loc,
            ))),
            Span::dummy(),
        ));
        func.blocks[bb_body.0].statements.push(Statement::Assign(
            Place::Local(acc_loc),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(acc_loc)),
                Operand::Copy(Place::Local(elem_loc)),
            ),
            Span::dummy(),
        ));
        func.blocks[bb_body.0].statements.push(Statement::Assign(
            Place::Local(i_loc),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(i_loc)),
                Operand::Constant(MirConstant::Int(1)),
            ),
            Span::dummy(),
        ));
        func.blocks[bb_body.0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

        func.blocks[bb_exit.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(acc_loc))),
            Span::dummy(),
        ));

        let changed = lust::mir::opt::simd::SimdOptimizer::run_on_function(&mut func);
        assert!(changed, "Loop must vectorize to produce bb_vec_chk_* blocks");
        assert!(
            !lust::mir::opt::simd::recognize_vectorized_loop_blocks(&func).is_empty(),
            "Vectorized loop blocks must be recognized"
        );

        let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift");
        let code = compiler
            .compile_function(&func)
            .expect("Compile vectorized sum");
        let native_fn: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(code) };

        lust_mir_set_status(0);
        let _ = native_fn(0x1000, 4);
        assert_eq!(
            lust_mir_get_status(),
            MIR_STATUS_ERR_OUT_OF_BOUNDS,
            "bb_vec_chk_* must validate array pointer via lust_mir_array_len without SIGSEGV"
        );
        return;
    }

    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    run_isolated_child(
        "test_f3_d3_simd_chk_block_pattern_invalid_ptr_rejected_isolated_child",
        "LUST_F3_D3_CHILD",
    );
}

#[test]
fn test_f3_b6_missing_struct_metadata_rejected_pre_entry_and_decode() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    // 1. VM dispatch: missing struct_metadata prevents native entry (None layout unreachable in VM)
    let sig = FunctionSignature {
        name: "mk_missing_meta".into(),
        param_types: vec![],
        return_type: MirType::Struct("MissingMeta".into()),
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let s_loc = func.add_local(
        Some("s".into()),
        MirType::Struct("MissingMeta".into()),
        false,
        Span::dummy(),
    );
    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(s_loc),
        Rvalue::Aggregate(
            MirType::Struct("MissingMeta".into()),
            vec![Operand::Constant(MirConstant::Int(42))],
        ),
        Span::dummy(),
    ));
    func.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(s_loc))),
        Span::dummy(),
    ));
    let mut module = MirModule::new("f3_b6".into());
    module.add_function(func);
    let mut vm = VM::new();
    vm.set_backend(VmBackend::Cranelift);
    vm.load_mir_module(&module).expect("Load MIR module");
    // Intentionally omit vm.register_structs(&...) so MissingMeta has no struct_metadata
    let entries_before = vm.cranelift_entry_count;
    let _ = vm.call("mk_missing_meta", vec![]);
    assert_eq!(
        vm.cranelift_entry_count, entries_before,
        "VM must refuse native entry when struct_metadata lacks the struct"
    );

    // 2. Direct validate_and_decode_struct(..., None) must reject active allocation pointers in fields
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    let inner_arr = lust_mir_alloc_array_typed(1, MIR_ARRAY_ELEM_TAG_INT);
    assert!(!inner_arr.is_null());

    let s_ptr = lust_mir_alloc_struct(1);
    assert!(!s_ptr.is_null());
    assert!(ctx.set_allocation_kind(s_ptr as usize, MirAllocationKind::Struct("BoxedPtr".into())));
    unsafe {
        *s_ptr.add(2) = inner_arr as i64;
    }

    let err = lust::vm::cranelift_dispatch::validate_and_decode_struct(
        &ctx,
        &[],
        s_ptr as i64,
        "BoxedPtr",
        None,
    );
    assert!(
        err.is_err(),
        "validate_and_decode_struct(..., None) must reject raw active allocation pointers, got {err:?}"
    );
    ctx.clear_to_baselines(0, 0);
}

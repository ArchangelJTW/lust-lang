//! Regression and adversarial provenance suite for TASK-W4-ARRAY-TYPED-PROVENANCE-F3.

use std::rc::Rc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use lust::ast::Span;
use lust::bytecode::Value;
use lust::config::LustConfig;
use lust::data::layout::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1, TargetAbi};
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::*;
use lust::mir::ir::*;
use lust::mir::lowering::AstLowering;
use lust::mir::runtime_helpers::{enter_mir_context, lust_mir_alloc_array, MirAllocationKind, MirContext};
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{VmBackend, VM};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn setup_vm_from_source(source: &str, module_name: &str) -> VM {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("Tokenization must succeed");
    let items = Parser::new(tokens).parse().expect("Parsing must succeed");

    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let struct_defs = checker.struct_definitions();

    let mut module = MirModule::new(module_name.into());
    for item in &items {
        if let lust::ast::ItemKind::Function(func_def) = &item.kind {
            let mir_fn = AstLowering::lower_function(func_def).expect("MIR lowering must succeed");
            module.add_function(mir_fn);
        }
    }

    let mut vm = VM::new();
    vm.set_backend(VmBackend::Cranelift);
    vm.load_mir_module(&module).expect("Load MIR module");
    vm.register_structs(&struct_defs);
    vm
}

fn compile_and_call_zero_arg_ptr(
    compiler: &mut CraneliftJitCompiler,
    func: MirFunction,
) -> *mut i64 {
    let ptr = compiler.compile_function(&func).expect("Compile MIR fn");
    let f: extern "C" fn() -> i64 = unsafe { std::mem::transmute(ptr) };
    f() as *mut i64
}

fn make_zero_arg_aggregate_fn(name: &str, ret_ty: MirType, agg_ty: MirType, elems: Vec<Operand>) -> MirFunction {
    let sig = FunctionSignature {
        name: name.into(),
        param_types: vec![],
        return_type: ret_ty.clone(),
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let tmp = func.add_local(Some("tmp".into()), agg_ty.clone(), false, Span::dummy());
    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(tmp),
        Rvalue::Aggregate(agg_ty, elems),
        Span::dummy(),
    ));
    func.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(tmp))),
        Span::dummy(),
    ));
    func
}

fn make_zero_arg_const_return_fn(name: &str, ret_ty: MirType, val: i64) -> MirFunction {
    let sig = FunctionSignature {
        name: name.into(),
        param_types: vec![],
        return_type: ret_ty.clone(),
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let tmp = func.add_local(Some("raw_ptr".into()), ret_ty.clone(), false, Span::dummy());
    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(tmp),
        Rvalue::Cast(Operand::Constant(MirConstant::Int(val)), ret_ty),
        Span::dummy(),
    ));
    func.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(tmp))),
        Span::dummy(),
    ));
    func
}

#[test]
fn test_w4_f3_null_and_rawwords_rejected_before_deref() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    // 1. Direct validate_and_decode_array rejects null pointer (0x0)
    {
        let ctx = MirContext::new();
        let err = lust::vm::cranelift_dispatch::validate_and_decode_array(
            &ctx,
            &[],
            0,
            &MirType::Int,
        )
        .expect_err("Null pointer (0x0) must be rejected by validate_and_decode_array");
        assert!(
            err.to_string().contains("null") || err.to_string().contains("0x0"),
            "Expected null provenance error, got: {err}"
        );
    }

    // 2. Direct validate_and_decode_array rejects RawWords from lust_mir_alloc_array
    {
        let ctx = MirContext::new();
        let _guard = enter_mir_context(&ctx);
        let raw_ptr = lust_mir_alloc_array(2);
        assert!(!raw_ptr.is_null());
        let err = lust::vm::cranelift_dispatch::validate_and_decode_array(
            &ctx,
            &[],
            raw_ptr as i64,
            &MirType::Int,
        )
        .expect_err("RawWords allocation must be rejected by validate_and_decode_array");
        assert!(
            err.to_string().contains("Type/kind mismatch") || err.to_string().contains("RawWords"),
            "Expected kind mismatch error, got: {err}"
        );
        let rec = ctx
            .get_allocation_record(raw_ptr as usize)
            .expect("RawWords record must remain intact");
        assert_eq!(
            rec.kind,
            MirAllocationKind::RawWords,
            "Rejected RawWords record must NOT be mutated to Array"
        );
        ctx.clear_to_baselines(0, 0);
    }

    // 3. End-to-end compiled VM::call returning null (0) for Array, Tuple, Struct must fail closed
    {
        let mut vm = setup_vm_from_source(
            r#"
            struct BoxedInt
                x: int
            end
            "#,
            "f3_null_return",
        );

        let fn_arr = make_zero_arg_const_return_fn(
            "ret_null_arr",
            MirType::Array(Box::new(MirType::Int)),
            0,
        );
        let fn_tup = make_zero_arg_const_return_fn(
            "ret_null_tup",
            MirType::Tuple(vec![MirType::Int, MirType::Int]),
            0,
        );
        let fn_str = make_zero_arg_const_return_fn(
            "ret_null_struct",
            MirType::Struct("BoxedInt".into()),
            0,
        );

        vm.load_mir_function(fn_arr).unwrap();
        vm.load_mir_function(fn_tup).unwrap();
        vm.load_mir_function(fn_str).unwrap();

        let entry_before = vm.cranelift_entry_count;
        let err_arr = vm
            .call("ret_null_arr", vec![])
            .expect_err("Null Array return must fail closed");
        assert_eq!(vm.cranelift_entry_count, entry_before + 1);
        assert!(
            err_arr.to_string().contains("null") || err_arr.to_string().contains("0x0"),
            "Unexpected null Array error: {err_arr}"
        );

        let entry_before = vm.cranelift_entry_count;
        let err_tup = vm
            .call("ret_null_tup", vec![])
            .expect_err("Null Tuple return must fail closed");
        assert_eq!(vm.cranelift_entry_count, entry_before + 1);
        assert!(
            err_tup.to_string().contains("null") || err_tup.to_string().contains("0x0"),
            "Unexpected null Tuple error: {err_tup}"
        );

        let entry_before = vm.cranelift_entry_count;
        let err_str = vm
            .call("ret_null_struct", vec![])
            .expect_err("Null Struct return must fail closed");
        assert_eq!(vm.cranelift_entry_count, entry_before + 1);
        assert!(
            err_str.to_string().contains("null") || err_str.to_string().contains("0x0"),
            "Unexpected null Struct error: {err_str}"
        );
        assert_eq!(vm.call_stack_depth(), 0);
    }
}

#[test]
fn test_w4_f3_compiled_aggregate_kinds_and_cross_kind_rejection() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    // 1. Verify actual compiled Cranelift aggregates register exact typed MirAllocationKind
    {
        let ctx = MirContext::new();
        let _guard = enter_mir_context(&ctx);
        let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift");

        let fn_arr_int = make_zero_arg_aggregate_fn(
            "alloc_int_arr",
            MirType::Array(Box::new(MirType::Int)),
            MirType::Array(Box::new(MirType::Int)),
            vec![
                Operand::Constant(MirConstant::Int(10)),
                Operand::Constant(MirConstant::Int(20)),
            ],
        );
        let fn_tup = make_zero_arg_aggregate_fn(
            "alloc_tup",
            MirType::Tuple(vec![MirType::Int, MirType::Int]),
            MirType::Tuple(vec![MirType::Int, MirType::Int]),
            vec![
                Operand::Constant(MirConstant::Int(30)),
                Operand::Constant(MirConstant::Int(40)),
            ],
        );
        let fn_struct = make_zero_arg_aggregate_fn(
            "alloc_struct",
            MirType::Struct("BoxedInt".into()),
            MirType::Struct("BoxedInt".into()),
            vec![Operand::Constant(MirConstant::Int(42))],
        );

        let arr_ptr = compile_and_call_zero_arg_ptr(&mut compiler, fn_arr_int);
        let tup_ptr = compile_and_call_zero_arg_ptr(&mut compiler, fn_tup);
        let struct_ptr = compile_and_call_zero_arg_ptr(&mut compiler, fn_struct);

        let arr_rec = ctx
            .get_active_allocation_record(arr_ptr as usize)
            .expect("Compiled array must be registered");
        assert_eq!(arr_rec.kind, MirAllocationKind::Array(MirType::Int));

        let tup_rec = ctx
            .get_active_allocation_record(tup_ptr as usize)
            .expect("Compiled tuple must be registered");
        assert!(
            matches!(tup_rec.kind, MirAllocationKind::Tuple(_)),
            "Compiled tuple must have MirAllocationKind::Tuple, got {:?}",
            tup_rec.kind
        );

        let struct_rec = ctx
            .get_active_allocation_record(struct_ptr as usize)
            .expect("Compiled struct must be registered");
        assert!(
            matches!(struct_rec.kind, MirAllocationKind::Struct(_)),
            "Compiled struct must have MirAllocationKind::Struct, got {:?}",
            struct_rec.kind
        );

        // Correct array elem type succeeds
        let ok_arr = lust::vm::cranelift_dispatch::validate_and_decode_array(
            &ctx,
            &[],
            arr_ptr as i64,
            &MirType::Int,
        )
        .expect("Typed Array(Int) must decode as Int");
        assert_eq!(ok_arr.elements, vec![Value::Int(10), Value::Int(20)]);

        // Wrong array elem type fails
        let err_elem = lust::vm::cranelift_dispatch::validate_and_decode_array(
            &ctx,
            &[],
            arr_ptr as i64,
            &MirType::Float,
        )
        .expect_err("Array(Int) decoded as Float must fail");
        assert!(
            err_elem.to_string().contains("element type mismatch"),
            "Unexpected error: {err_elem}"
        );

        // Compiled Tuple and Struct rejected by validate_and_decode_array
        let err_tup_as_arr = lust::vm::cranelift_dispatch::validate_and_decode_array(
            &ctx,
            &[],
            tup_ptr as i64,
            &MirType::Int,
        )
        .expect_err("Compiled Tuple must be rejected as Array");
        assert!(
            err_tup_as_arr.to_string().contains("Type/kind mismatch"),
            "Unexpected error: {err_tup_as_arr}"
        );

        let err_struct_as_arr = lust::vm::cranelift_dispatch::validate_and_decode_array(
            &ctx,
            &[],
            struct_ptr as i64,
            &MirType::Int,
        )
        .expect_err("Compiled Struct must be rejected as Array");
        assert!(
            err_struct_as_arr.to_string().contains("Type/kind mismatch"),
            "Unexpected error: {err_struct_as_arr}"
        );

        ctx.clear_to_baselines(0, 0);
    }

    // 2. End-to-end VM::call cross-kind return rejection (Array<->Tuple<->Struct)
    {
        let mut vm = setup_vm_from_source(
            r#"
            struct BoxedInt
                x: int
            end
            "#,
            "f3_cross_kind",
        );

        let arr_as_tup = make_zero_arg_aggregate_fn(
            "arr_as_tup",
            MirType::Tuple(vec![MirType::Int, MirType::Int]),
            MirType::Array(Box::new(MirType::Int)),
            vec![
                Operand::Constant(MirConstant::Int(1)),
                Operand::Constant(MirConstant::Int(2)),
            ],
        );
        let arr_as_struct = make_zero_arg_aggregate_fn(
            "arr_as_struct",
            MirType::Struct("BoxedInt".into()),
            MirType::Array(Box::new(MirType::Int)),
            vec![Operand::Constant(MirConstant::Int(99))],
        );
        let tup_as_arr = make_zero_arg_aggregate_fn(
            "tup_as_arr",
            MirType::Array(Box::new(MirType::Int)),
            MirType::Tuple(vec![MirType::Int, MirType::Int]),
            vec![
                Operand::Constant(MirConstant::Int(3)),
                Operand::Constant(MirConstant::Int(4)),
            ],
        );

        vm.load_mir_function(arr_as_tup).unwrap();
        vm.load_mir_function(arr_as_struct).unwrap();
        vm.load_mir_function(tup_as_arr).unwrap();

        let entry_before = vm.cranelift_entry_count;
        let e1 = vm
            .call("arr_as_tup", vec![])
            .expect_err("Array returned as Tuple must fail");
        assert_eq!(vm.cranelift_entry_count, entry_before + 1);
        assert!(e1.to_string().contains("Type/kind mismatch"), "Got: {e1}");

        let entry_before = vm.cranelift_entry_count;
        let e2 = vm
            .call("arr_as_struct", vec![])
            .expect_err("Array returned as Struct must fail");
        assert_eq!(vm.cranelift_entry_count, entry_before + 1);
        assert!(e2.to_string().contains("Type/kind mismatch"), "Got: {e2}");

        let entry_before = vm.cranelift_entry_count;
        let e3 = vm
            .call("tup_as_arr", vec![])
            .expect_err("Tuple returned as Array must fail");
        assert_eq!(vm.cranelift_entry_count, entry_before + 1);
        assert!(e3.to_string().contains("Type/kind mismatch"), "Got: {e3}");

        assert_eq!(vm.mir_context.buffer_count(), 0);
        assert_eq!(vm.call_stack_depth(), 0);
    }
}

#[test]
fn test_w4_f3_hostile_unowned_tuple_struct_pointer_isolated_watchdog() {
    if std::env::var("LUST_F3_HOSTILE_CHILD").ok().as_deref() == Some("1") {
        let mut vm = setup_vm_from_source(
            r#"
            struct BoxedInt
                x: int
            end
            "#,
            "f3_hostile_child",
        );

        let bad_tup = make_zero_arg_const_return_fn(
            "bad_tup_ptr",
            MirType::Tuple(vec![MirType::Int, MirType::Int]),
            0x1000,
        );
        let bad_struct = make_zero_arg_const_return_fn(
            "bad_struct_ptr",
            MirType::Struct("BoxedInt".into()),
            0x1000,
        );
        vm.load_mir_function(bad_tup).unwrap();
        vm.load_mir_function(bad_struct).unwrap();

        let entry_before = vm.cranelift_entry_count;
        let err1 = vm
            .call("bad_tup_ptr", vec![])
            .expect_err("Unowned Tuple pointer 0x1000 must fail before dereference");
        assert_eq!(vm.cranelift_entry_count, entry_before + 1);
        assert!(
            err1.to_string().contains("not an owned allocation base"),
            "Unexpected Tuple provenance error: {err1}"
        );

        let entry_before = vm.cranelift_entry_count;
        let err2 = vm
            .call("bad_struct_ptr", vec![])
            .expect_err("Unowned Struct pointer 0x1000 must fail before dereference");
        assert_eq!(vm.cranelift_entry_count, entry_before + 1);
        assert!(
            err2.to_string().contains("not an owned allocation base"),
            "Unexpected Struct provenance error: {err2}"
        );
        return;
    }

    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let exe = std::env::current_exe().expect("current_exe");
    let mut child = std::process::Command::new(exe)
        .arg("--exact")
        .arg("test_w4_f3_hostile_unowned_tuple_struct_pointer_isolated_watchdog")
        .arg("--nocapture")
        .env("LUST_F3_HOSTILE_CHILD", "1")
        .spawn()
        .expect("Spawn isolated child process");

    let start = Instant::now();
    let timeout = Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            assert!(
                status.success(),
                "Isolated hostile unowned-pointer child failed/crashed with status: {status:?}"
            );
            break;
        }
        if start.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            panic!("Isolated hostile child timed out after {:?}", timeout);
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn test_w4_f3_short_extent_and_nested_lifetime_isolation() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    // 1. Short word_capacity < 2 rejected before dereferencing header word 1
    {
        let ctx = MirContext::new();
        let short_raw = Box::into_raw(Box::new([1i64])) as *mut i64;
        ctx.register_allocation(short_raw, 1, MirAllocationKind::RawWords);
        let err_raw = lust::vm::cranelift_dispatch::validate_and_decode_array(
            &ctx,
            &[],
            short_raw as i64,
            &MirType::Int,
        )
        .expect_err("1-word RawWords must fail");
        assert!(err_raw.to_string().contains("Type/kind mismatch"));
        assert_eq!(
            ctx.get_allocation_record(short_raw as usize).unwrap().kind,
            MirAllocationKind::RawWords
        );

        ctx.register_allocation(short_raw, 1, MirAllocationKind::Array(MirType::Int));
        let err_short = lust::vm::cranelift_dispatch::validate_and_decode_array(
            &ctx,
            &[],
            short_raw as i64,
            &MirType::Int,
        )
        .expect_err("1-word Array must fail word_capacity < 2");
        assert!(err_short.to_string().contains("word_capacity"));

        ctx.register_allocation(short_raw, 1, MirAllocationKind::Tuple(vec![]));
        let err_short_tup = lust::vm::cranelift_dispatch::validate_and_decode_tuple(
            &ctx,
            short_raw as i64,
            &[MirType::Int],
        )
        .expect_err("1-word Tuple must fail word_capacity < 2");
        assert!(err_short_tup.to_string().contains("word_capacity"));

        ctx.register_allocation(short_raw, 1, MirAllocationKind::Struct("BoxedInt".into()));
        let err_short_struct = lust::vm::cranelift_dispatch::validate_and_decode_struct(
            &ctx,
            &[],
            short_raw as i64,
            "BoxedInt",
            None,
        )
        .expect_err("1-word Struct must fail word_capacity < 2");
        assert!(err_short_struct.to_string().contains("word_capacity"));
        unsafe {
            drop(Box::from_raw(short_raw as *mut [i64; 1]));
        }
    }

    // 2. Short extent Tuple / Struct via VM::call
    {
        let mut vm = setup_vm_from_source(
            r#"
            struct TwoFields
                a: int,
                b: int
            end
            "#,
            "f3_short_extent",
        );

        let short_tup = make_zero_arg_aggregate_fn(
            "short_tup",
            MirType::Tuple(vec![MirType::Int, MirType::Int]),
            MirType::Tuple(vec![MirType::Int]),
            vec![Operand::Constant(MirConstant::Int(1))],
        );
        let short_struct = make_zero_arg_aggregate_fn(
            "short_struct",
            MirType::Struct("TwoFields".into()),
            MirType::Struct("TwoFields".into()),
            vec![Operand::Constant(MirConstant::Int(1))],
        );
        vm.load_mir_function(short_tup).unwrap();
        vm.load_mir_function(short_struct).unwrap();

        assert!(
            vm.call("short_tup", vec![]).is_err(),
            "Short 1-elem Tuple returned as 2-elem Tuple must fail"
        );
        assert!(
            vm.call("short_struct", vec![]).is_err(),
            "Short 1-field Struct returned as 2-field TwoFields must fail"
        );
        assert_eq!(vm.mir_context.buffer_count(), 0);
    }

    // 3. Nested MirContext lifetime isolation with actual compiled aggregate paths
    {
        let ctx = MirContext::new();
        let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift");

        let fn_arr = make_zero_arg_aggregate_fn(
            "nested_arr",
            MirType::Array(Box::new(MirType::Int)),
            MirType::Array(Box::new(MirType::Int)),
            vec![
                Operand::Constant(MirConstant::Int(11)),
                Operand::Constant(MirConstant::Int(22)),
            ],
        );
        let fn_tup = make_zero_arg_aggregate_fn(
            "nested_tup",
            MirType::Tuple(vec![MirType::Int, MirType::Int]),
            MirType::Tuple(vec![MirType::Int, MirType::Int]),
            vec![
                Operand::Constant(MirConstant::Int(33)),
                Operand::Constant(MirConstant::Int(44)),
            ],
        );
        let fn_struct = make_zero_arg_aggregate_fn(
            "nested_struct",
            MirType::Struct("BoxedInt".into()),
            MirType::Struct("BoxedInt".into()),
            vec![Operand::Constant(MirConstant::Int(55))],
        );

        let arr_code = compiler.compile_function(&fn_arr).expect("Compile nested_arr");
        let tup_code = compiler.compile_function(&fn_tup).expect("Compile nested_tup");
        let struct_code = compiler.compile_function(&fn_struct).expect("Compile nested_struct");
        let call_arr: extern "C" fn() -> i64 = unsafe { std::mem::transmute(arr_code) };
        let call_tup: extern "C" fn() -> i64 = unsafe { std::mem::transmute(tup_code) };
        let call_struct: extern "C" fn() -> i64 = unsafe { std::mem::transmute(struct_code) };

        let _outer_guard = ctx.enter();
        let outer_arr = call_arr() as *mut i64;
        let outer_tup = call_tup() as *mut i64;
        let outer_struct = call_struct() as *mut i64;

        {
            let _inner_guard = ctx.enter();
            let inner_baseline = ctx.capture_baseline();
            let inner_arr = call_arr() as *mut i64;
            let inner_tup = call_tup() as *mut i64;
            let inner_struct = call_struct() as *mut i64;

            assert!(
                lust::vm::cranelift_dispatch::validate_and_decode_array(
                    &ctx,
                    &[],
                    outer_arr as i64,
                    &MirType::Int,
                )
                .is_err(),
                "Inner scope must reject outer Array"
            );
            assert!(
                lust::vm::cranelift_dispatch::validate_and_decode_tuple(
                    &ctx,
                    outer_tup as i64,
                    &[MirType::Int, MirType::Int],
                )
                .is_err(),
                "Inner scope must reject outer Tuple"
            );
            assert!(
                lust::vm::cranelift_dispatch::validate_and_decode_struct(
                    &ctx,
                    &[],
                    outer_struct as i64,
                    "BoxedInt",
                    None,
                )
                .is_err(),
                "Inner scope must reject outer Struct"
            );

            let inner_ok = lust::vm::cranelift_dispatch::validate_and_decode_array(
                &ctx,
                &[],
                inner_arr as i64,
                &MirType::Int,
            )
            .expect("Inner Array must decode in inner scope");
            assert_eq!(inner_ok.elements, vec![Value::Int(11), Value::Int(22)]);

            let inner_tup_ok = lust::vm::cranelift_dispatch::validate_and_decode_tuple(
                &ctx,
                inner_tup as i64,
                &[MirType::Int, MirType::Int],
            )
            .expect("Inner Tuple must decode in inner scope");
            assert_eq!(inner_tup_ok, vec![Value::Int(33), Value::Int(44)]);

            let inner_struct_ok = lust::vm::cranelift_dispatch::validate_and_decode_struct(
                &ctx,
                &[],
                inner_struct as i64,
                "BoxedInt",
                None,
            )
            .expect("Inner Struct must decode in inner scope");
            assert!(matches!(inner_struct_ok, Value::Struct(_)));

            ctx.clear_to_baselines(inner_baseline.handles, inner_baseline.buffers);
            assert!(ctx.get_allocation_record(inner_arr as usize).is_none());
            assert!(ctx.get_allocation_record(inner_tup as usize).is_none());
            assert!(ctx.get_allocation_record(inner_struct as usize).is_none());
        }

        // Outer scope still owns outer_arr, outer_tup, outer_struct
        let outer_ok = lust::vm::cranelift_dispatch::validate_and_decode_array(
            &ctx,
            &[],
            outer_arr as i64,
            &MirType::Int,
        )
        .expect("Outer Array must survive inner cleanup");
        assert_eq!(outer_ok.elements, vec![Value::Int(11), Value::Int(22)]);
        assert_eq!(
            lust::vm::cranelift_dispatch::validate_and_decode_tuple(
                &ctx,
                outer_tup as i64,
                &[MirType::Int, MirType::Int],
            )
            .expect("Outer Tuple must survive inner cleanup"),
            vec![Value::Int(33), Value::Int(44)]
        );
        assert!(lust::vm::cranelift_dispatch::validate_and_decode_struct(
            &ctx,
            &[],
            outer_struct as i64,
            "BoxedInt",
            None,
        )
        .is_ok());

        ctx.clear_to_baselines(0, 0);
        assert_eq!(ctx.buffer_count(), 0);
    }
}

#[test]
fn test_w4_f3_legitimate_native_aggregates_and_raw_abi_preserved() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    // 1. Legitimate VM::call native Array (Int/Bool/Float/String), Tuple, and Struct allocation/return
    {
        let mut vm = setup_vm_from_source(
            r#"
            struct Point
                x: int,
                y: int
            end

            function make_int_arr(): Array<int>
                return [10, 20, 30]
            end

            function make_tuple(): (int, int)
                return (7, 11)
            end

            function make_point(a: int, b: int): Point
                return Point { x = a, y = b }
            end

            function id_point(p: Point): Point
                return p
            end
            "#,
            "f3_legit_aggregates",
        );

        let r_arr = vm.call("make_int_arr", vec![]).unwrap();
        assert_eq!(
            r_arr,
            Value::array(vec![
                Value::Int(10),
                Value::Int(20),
                Value::Int(30)
            ])
        );

        let r_tup = vm.call("make_tuple", vec![]).unwrap();
        assert_eq!(
            r_tup,
            Value::Tuple(Rc::new(vec![Value::Int(7), Value::Int(11)]))
        );

        let r_pt = vm
            .call("make_point", vec![Value::Int(3), Value::Int(4)])
            .unwrap();
        let Value::Struct(ref s) = r_pt else {
            panic!("Expected Value::Struct");
        };
        assert_eq!(*s.fields.borrow(), vec![Value::Int(3), Value::Int(4)]);

        let id_pt = vm.call("id_point", vec![r_pt.clone()]).unwrap();
        assert_eq!(id_pt, r_pt);

        // Also test Float / Bool / String native array aggregates via MIR
        let fn_float = make_zero_arg_aggregate_fn(
            "make_float_arr",
            MirType::Array(Box::new(MirType::Float)),
            MirType::Array(Box::new(MirType::Float)),
            vec![
                Operand::Constant(MirConstant::Float(1.5)),
                Operand::Constant(MirConstant::Float(-0.0)),
            ],
        );
        let fn_bool = make_zero_arg_aggregate_fn(
            "make_bool_arr",
            MirType::Array(Box::new(MirType::Bool)),
            MirType::Array(Box::new(MirType::Bool)),
            vec![
                Operand::Constant(MirConstant::Bool(true)),
                Operand::Constant(MirConstant::Bool(false)),
            ],
        );
        let fn_str = make_zero_arg_aggregate_fn(
            "make_str_arr",
            MirType::Array(Box::new(MirType::String)),
            MirType::Array(Box::new(MirType::String)),
            vec![
                Operand::Constant(MirConstant::String("hello".into())),
                Operand::Constant(MirConstant::String("world".into())),
            ],
        );
        vm.load_mir_function(fn_float).unwrap();
        vm.load_mir_function(fn_bool).unwrap();
        vm.load_mir_function(fn_str).unwrap();

        let rf = vm.call("make_float_arr", vec![]).unwrap();
        let Value::Array(rf_rc) = rf else {
            panic!("Expected Float array");
        };
        let rf_elems = rf_rc.borrow();
        let Value::Float(f0) = rf_elems[0] else { panic!() };
        let Value::Float(f1) = rf_elems[1] else { panic!() };
        assert_eq!(f0, 1.5);
        assert_eq!(f1.to_bits(), (-0.0f64).to_bits());

        assert_eq!(
            vm.call("make_bool_arr", vec![]).unwrap(),
            Value::array(vec![
                Value::Bool(true),
                Value::Bool(false)
            ])
        );
        assert_eq!(
            vm.call("make_str_arr", vec![]).unwrap(),
            Value::array(vec![
                Value::string("hello"),
                Value::string("world")
            ])
        );
        assert_eq!(vm.mir_context.buffer_count(), 0);
        assert_eq!(vm.call_stack_depth(), 0);
    }

    // 2. Raw ABI compatibility: C-flat struct layout + direct lust_mir_alloc_array
    {
        let ctx = MirContext::new();
        let _guard = enter_mir_context(&ctx);

        let raw_buf = lust_mir_alloc_array(3);
        assert!(!raw_buf.is_null());
        unsafe {
            assert_eq!(*raw_buf, 3);
            assert_eq!(*raw_buf.add(1), 3);
            *raw_buf.add(2) = 100;
        }
        assert_eq!(lust_mir_array_len(raw_buf), 3);
        assert_eq!(lust_mir_array_get(raw_buf, 0), 100);

        let logical_pair = LogicalLayoutV1::new_with_abi(
            9991,
            "PairAbi",
            vec![
                LogicalFieldDesc::new(0, "a", CanonicalTypeTag::Int64),
                LogicalFieldDesc::new(1, "b", CanonicalTypeTag::Int64),
            ],
            TargetAbi::aarch64(),
        );
        let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift");
        compiler.register_struct_layouts(&[MirStructLayout::from(&logical_pair)]);

        let sig = FunctionSignature {
            name: "pair_abi_sum".into(),
            param_types: vec![],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut func = MirFunction::new(sig, Span::dummy());
        let s_loc = func.add_local(
            Some("p".into()),
            MirType::Struct("PairAbi".into()),
            false,
            Span::dummy(),
        );
        let f0 = func.add_local(Some("f0".into()), MirType::Int, false, Span::dummy());
        let f1 = func.add_local(Some("f1".into()), MirType::Int, false, Span::dummy());
        let sum = func.add_local(Some("sum".into()), MirType::Int, false, Span::dummy());
        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(s_loc),
            Rvalue::Aggregate(
                MirType::Struct("PairAbi".into()),
                vec![
                    Operand::Constant(MirConstant::Int(12)),
                    Operand::Constant(MirConstant::Int(30)),
                ],
            ),
            Span::dummy(),
        ));
        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(f0),
            Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(s_loc)), 0))),
            Span::dummy(),
        ));
        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(f1),
            Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(s_loc)), 1))),
            Span::dummy(),
        ));
        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(sum),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(f0)),
                Operand::Copy(Place::Local(f1)),
            ),
            Span::dummy(),
        ));
        func.blocks[0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(sum))),
            Span::dummy(),
        ));

        let ptr = compiler.compile_function(&func).expect("Compile PairAbi");
        let f: extern "C" fn() -> i64 = unsafe { std::mem::transmute(ptr) };
        assert_eq!(f(), 42);

        ctx.clear_to_baselines(0, 0);
    }
}

#[test]
fn test_w4_f3_lust_source_typed_arrays_preserved() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    let source = r#"
        function src_float_arr(): Array<float>
            return [1.5, 2.5]
        end

        function src_bool_arr(): Array<bool>
            return [true, false]
        end

        function src_str_arr(): Array<string>
            return ["alpha", "beta"]
        end
    "#;
    let mut vm = setup_vm_from_source(source, "f3_src_typed_arrays");

    let e0 = vm.cranelift_entry_count;
    let floats = vm.call("src_float_arr", vec![]).unwrap();
    assert_eq!(vm.cranelift_entry_count, e0 + 1);
    assert_eq!(
        floats,
        Value::array(vec![Value::Float(1.5), Value::Float(2.5)])
    );

    let e1 = vm.cranelift_entry_count;
    let bools = vm.call("src_bool_arr", vec![]).unwrap();
    assert_eq!(vm.cranelift_entry_count, e1 + 1);
    assert_eq!(
        bools,
        Value::array(vec![Value::Bool(true), Value::Bool(false)])
    );

    let e2 = vm.cranelift_entry_count;
    let strs = vm.call("src_str_arr", vec![]).unwrap();
    assert_eq!(vm.cranelift_entry_count, e2 + 1);
    assert_eq!(
        strs,
        Value::array(vec![
            Value::string("alpha"),
            Value::string("beta")
        ])
    );
    assert_eq!(vm.mir_context.buffer_count(), 0);
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_w4_f3_deopt_provenance_rejects_bad_and_preserves_legitimate_array() {
    use lust::vm::DeoptPayloadSource;

    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();
    reset_native_instrumentation_counters();
    take_deopt_payload();

    let fn_name = "deopt_typed_array_ok";
    let sig = FunctionSignature {
        name: fn_name.into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let n = func.params[0];
    let a = func.add_local(
        Some("a".into()),
        MirType::Array(Box::new(MirType::Int)),
        false,
        Span::dummy(),
    );
    let i = func.add_local(Some("i".into()), MirType::Int, false, Span::dummy());
    let zero = func.add_local(Some("zero".into()), MirType::Int, false, Span::dummy());
    let cond = func.add_local(Some("cond".into()), MirType::Bool, false, Span::dummy());
    let res = func.add_local(Some("res".into()), MirType::Int, false, Span::dummy());

    let bb0 = BlockId(0);
    let bb_hdr = func.new_block();
    let bb_body = func.new_block();
    let bb_exit = func.new_block();

    func.blocks[bb0.0].statements.push(Statement::Assign(
        Place::Local(a),
        Rvalue::Aggregate(
            MirType::Array(Box::new(MirType::Int)),
            vec![Operand::Constant(MirConstant::Int(77))],
        ),
        Span::dummy(),
    ));
    func.blocks[bb0.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    func.blocks[bb0.0].statements.push(Statement::Assign(
        Place::Local(zero),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    func.blocks[bb0.0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    func.blocks[bb_hdr.0].statements.push(Statement::Assign(
        Place::Local(cond),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(i)),
            Operand::Copy(Place::Local(n)),
        ),
        Span::dummy(),
    ));
    func.blocks[bb_hdr.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond)),
        then_block: bb_body,
        else_block: bb_exit,
        span: Span::dummy(),
    });

    func.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    func.blocks[bb_body.0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    func.blocks[bb_exit.0].statements.push(Statement::Assign(
        Place::Local(res),
        Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(a)), zero))),
        Span::dummy(),
    ));
    func.blocks[bb_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(res))),
        Span::dummy(),
    ));

    let mut vm = VM::new();
    vm.set_jit_backend(VmBackend::Cranelift);
    vm.load_mir_function(func).unwrap();
    vm.test_deopt_spike_hook = Some(lust::vm::TestDeoptSpikeHook::new(fn_name, bb_hdr.0));

    safepoint_request_deopt_on_poll(Some(fn_name.into()), 1);
    let e0 = vm.cranelift_entry_count;
    let out = vm.call(fn_name, vec![Value::Int(1)]).expect("Deopt must resume and succeed");
    assert_eq!(vm.cranelift_entry_count, e0 + 1);
    assert_eq!(vm.last_deopt_source, Some(DeoptPayloadSource::NativeExecution));
    assert_eq!(out, Value::Int(77));
    assert_eq!(vm.mir_context.buffer_count(), 0);

    // Hostile BlockEntry deopt where native Array local holds null (0x0) must fail closed before deref
    let bad_fn_name = "deopt_null_array_rejected";
    let bad_sig = FunctionSignature {
        name: bad_fn_name.into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut bad_func = MirFunction::new(bad_sig, Span::dummy());
    let bn = bad_func.params[0];
    let ba = bad_func.add_local(
        Some("a".into()),
        MirType::Array(Box::new(MirType::Int)),
        false,
        Span::dummy(),
    );
    let bi = bad_func.add_local(Some("i".into()), MirType::Int, false, Span::dummy());
    let bzero = bad_func.add_local(Some("zero".into()), MirType::Int, false, Span::dummy());
    let bcond = bad_func.add_local(Some("cond".into()), MirType::Bool, false, Span::dummy());
    let bres = bad_func.add_local(Some("res".into()), MirType::Int, false, Span::dummy());

    let b0 = BlockId(0);
    let b_hdr = bad_func.new_block();
    let b_body = bad_func.new_block();
    let b_exit = bad_func.new_block();

    bad_func.blocks[b0.0].statements.push(Statement::Assign(
        Place::Local(ba),
        Rvalue::Cast(
            Operand::Constant(MirConstant::Int(0)),
            MirType::Array(Box::new(MirType::Int)),
        ),
        Span::dummy(),
    ));
    bad_func.blocks[b0.0].statements.push(Statement::Assign(
        Place::Local(bi),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    bad_func.blocks[b0.0].statements.push(Statement::Assign(
        Place::Local(bzero),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    bad_func.blocks[b0.0].terminator = Some(Terminator::Goto(b_hdr, Span::dummy()));

    bad_func.blocks[b_hdr.0].statements.push(Statement::Assign(
        Place::Local(bcond),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(bi)),
            Operand::Copy(Place::Local(bn)),
        ),
        Span::dummy(),
    ));
    bad_func.blocks[b_hdr.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(bcond)),
        then_block: b_body,
        else_block: b_exit,
        span: Span::dummy(),
    });
    bad_func.blocks[b_body.0].statements.push(Statement::Assign(
        Place::Local(bi),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(bi)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    bad_func.blocks[b_body.0].terminator = Some(Terminator::Goto(b_hdr, Span::dummy()));
    bad_func.blocks[b_exit.0].statements.push(Statement::Assign(
        Place::Local(bres),
        Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(ba)), bzero))),
        Span::dummy(),
    ));
    bad_func.blocks[b_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(bres))),
        Span::dummy(),
    ));

    let mut bad_vm = VM::new();
    bad_vm.set_jit_backend(VmBackend::Cranelift);
    bad_vm.load_mir_function(bad_func).unwrap();
    bad_vm.test_deopt_spike_hook = Some(lust::vm::TestDeoptSpikeHook::new(bad_fn_name, b_hdr.0));
    safepoint_request_deopt_on_poll(Some(bad_fn_name.into()), 1);

    let e1 = bad_vm.cranelift_entry_count;
    let bad_err = bad_vm
        .call(bad_fn_name, vec![Value::Int(1)])
        .expect_err("Deopt with null Array local must fail closed");
    assert_eq!(bad_vm.cranelift_entry_count, e1 + 1);
    assert!(
        bad_err.to_string().contains("null") || bad_err.to_string().contains("0x0"),
        "Expected null provenance error on deopt, got: {bad_err}"
    );
    assert_eq!(bad_vm.call_stack_depth(), 0);
    assert_eq!(bad_vm.mir_context.buffer_count(), 0);
}

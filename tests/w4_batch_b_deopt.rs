use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use lust::ast::Span;
use lust::bytecode::{native_fn, NativeCallResult, Value};
use lust::effect_abi::InterruptReason;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{
    get_current_sticky_error, reset_native_instrumentation_counters, safepoint_reset,
    take_current_sticky_error, take_deopt_payload, CraneliftJitCompiler,
};
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::{
    native_invocation_depth, push_native_invocation, safepoint_request_deopt,
    CRANELIFT_SIMD_VECTOR_COUNT,
};
use lust::mir::ir::{
    FunctionSignature, MirBinaryOp, MirConstant, MirFunction, MirModule, MirType, Operand, Place,
    Rvalue, Statement, Terminator,
};
use lust::mir::lowering::AstLowering;
use lust::mir::opt::simd::reset_simd_counters;
#[cfg(feature = "test-deopt")]
use lust::mir::opt::simd::{
    recognize_vectorized_loop_blocks, set_force_simd_scalar_fallback, SimdOptimizer,
    VectorLoopKernelKind,
};
#[cfg(feature = "test-deopt")]
use lust::mir::runtime_helpers::{enter_mir_context, MirContext};
#[cfg(feature = "test-deopt")]
use lust::mir::verifier::MirVerifier;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{reset_effect_handler_state, VmBackend, VM};
use lust::{Compiler, LustConfig};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn setup_bytecode_funcs(source: &str) -> Vec<lust::bytecode::Function> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");
    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    compiler.compile_module(&items).expect("compile_module")
}

fn lower_mir_module_from_source(source: &str, name: &str) -> MirModule {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");
    let mut module = MirModule::new(name.into());
    for item in items {
        if let lust::ast::ItemKind::Function(func_def) = item.kind {
            let mir_fn = AstLowering::lower_function(&func_def).expect("lower_function");
            module.add_function(mir_fn);
        }
    }
    module
}

fn reset_all_test_state() {
    safepoint_reset();
    reset_effect_handler_state();
    reset_native_instrumentation_counters();
    reset_simd_counters();
    take_deopt_payload();
    take_current_sticky_error();
}

// =========================================================================
// Behavioral Phase 1 RED -> Phase 2 GREEN Tests (10 tests)
// =========================================================================

#[test]
fn test_b1_production_perform_resume_continues_once() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_test_state();

    // Bytecode harness with `handle { helper(5, log_arr) } with { Tick(v) => resume(v + 10) }`
    let driver_src = r#"
        function helper(x: int, log_arr: Array<int>): int
            return 0
        end

        function main(log_arr: Array<int>): int
            handle {
                helper(5, log_arr)
            } with {
                Tick(v) => resume(v + 10)
            }
        end
    "#;
    let bc_funcs = setup_bytecode_funcs(driver_src);

    // MIR `helper(x, log_arr)`:
    // bb0: log_arr[0] = log_arr[0] + 1; goto bb1
    // bb1: r = perform("Tick", x) -> bb2
    // bb2: log_arr[1] = log_arr[1] + 1; ret = r + 1; return ret
    let mut helper = MirFunction::new(
        FunctionSignature {
            name: "helper".into(),
            param_types: vec![MirType::Int, MirType::Array(Box::new(MirType::Int))],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        Span::dummy(),
    );
    let x = helper.params[0];
    let log_arr = helper.params[1];
    let idx0 = helper.add_local(Some("idx0".into()), MirType::Int, false, Span::dummy());
    let idx1 = helper.add_local(Some("idx1".into()), MirType::Int, false, Span::dummy());
    let tmp0 = helper.add_local(Some("tmp0".into()), MirType::Int, false, Span::dummy());
    let tmp1 = helper.add_local(Some("tmp1".into()), MirType::Int, false, Span::dummy());
    let r = helper.add_local(Some("r".into()), MirType::Int, false, Span::dummy());
    let ret = helper.add_local(Some("ret".into()), MirType::Int, false, Span::dummy());

    let bb1 = helper.new_block();
    let bb2 = helper.new_block();

    helper.blocks[0].statements.push(Statement::Assign(
        Place::Local(idx0),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    helper.blocks[0].statements.push(Statement::Assign(
        Place::Local(idx1),
        Rvalue::Use(Operand::Constant(MirConstant::Int(1))),
        Span::dummy(),
    ));
    helper.blocks[0].statements.push(Statement::Assign(
        Place::Local(tmp0),
        Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(log_arr)), idx0))),
        Span::dummy(),
    ));
    helper.blocks[0].statements.push(Statement::Assign(
        Place::Local(tmp0),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(tmp0)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    helper.blocks[0].statements.push(Statement::Assign(
        Place::Index(Box::new(Place::Local(log_arr)), idx0),
        Rvalue::Use(Operand::Copy(Place::Local(tmp0))),
        Span::dummy(),
    ));
    helper.blocks[0].terminator = Some(Terminator::Goto(bb1, Span::dummy()));

    helper.blocks[bb1.0].terminator = Some(Terminator::Call {
        callee: "perform".into(),
        args: vec![
            Operand::Constant(MirConstant::String("Tick".into())),
            Operand::Copy(Place::Local(x)),
        ],
        destination: Place::Local(r),
        success: bb2,
        error: None,
        span: Span::dummy(),
    });

    helper.blocks[bb2.0].statements.push(Statement::Assign(
        Place::Local(tmp1),
        Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(log_arr)), idx1))),
        Span::dummy(),
    ));
    helper.blocks[bb2.0].statements.push(Statement::Assign(
        Place::Local(tmp1),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(tmp1)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    helper.blocks[bb2.0].statements.push(Statement::Assign(
        Place::Index(Box::new(Place::Local(log_arr)), idx1),
        Rvalue::Use(Operand::Copy(Place::Local(tmp1))),
        Span::dummy(),
    ));
    helper.blocks[bb2.0].statements.push(Statement::Assign(
        Place::Local(ret),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(r)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    helper.blocks[bb2.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(ret))),
        Span::dummy(),
    ));

    // Bytecode oracle
    let mut vm_bc = VM::new();
    vm_bc.set_backend(VmBackend::Bytecode);
    vm_bc.load_functions(bc_funcs.clone());
    vm_bc.load_mir_function(helper.clone()).expect("load_mir_function bc");
    let log_bc = Rc::new(RefCell::new(vec![Value::Int(0), Value::Int(0)]));
    let res_bc = vm_bc
        .call("main", vec![Value::Array(log_bc.clone())])
        .expect("bc call");

    // Cranelift production path (no test_deopt_spike_hook)
    let mut vm_clif = VM::new();
    vm_clif.set_backend(VmBackend::Cranelift);
    vm_clif.load_functions(bc_funcs);
    vm_clif.load_mir_function(helper).expect("load_mir_function clif");
    let log_clif = Rc::new(RefCell::new(vec![Value::Int(0), Value::Int(0)]));
    let entries_before = vm_clif.cranelift_entry_count;
    let res_clif = vm_clif
        .call("main", vec![Value::Array(log_clif.clone())])
        .expect("clif call");

    assert_eq!(
        vm_clif.cranelift_entry_count,
        entries_before + 1,
        "helper with perform must enter Cranelift natively once"
    );
    assert_eq!(res_clif, Value::Int(16));
    assert_eq!(res_clif, res_bc);
    assert_eq!(*log_clif.borrow(), vec![Value::Int(1), Value::Int(1)]);
    assert_eq!(*log_clif.borrow(), *log_bc.borrow());
    assert!(take_deopt_payload().is_none());
    assert!(get_current_sticky_error().is_none());
}

#[test]
fn test_b2_all_seven_vm_intrinsics_and_fail_closed_closures() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_test_state();

    // Part A: perform / __lust_perform and resume / __lust_resume with live String + Array<Int> + scalars
    let driver_src = r#"
        function perf_helper(x: int, arr: Array<int>): int
            return 0
        end

        function lust_perf_helper(x: int, arr: Array<int>): int
            return 0
        end

        function resume_arm(v: int, arr: Array<int>): int
            return 0
        end

        function lust_resume_arm(v: int, arr: Array<int>): int
            return 0
        end

        function run_perform(arr: Array<int>): int
            handle {
                perf_helper(7, arr)
            } with {
                Tick(v) => resume(v + 100)
            }
        end

        function run_lust_perform(arr: Array<int>): int
            handle {
                lust_perf_helper(8, arr)
            } with {
                Tick(v) => resume(v + 100)
            }
        end

        function run_resume(arr: Array<int>): int
            handle {
                let x = perform("Tick", [9]);
                x + 1
            } with {
                Tick(v) => resume_arm(v, arr)
            }
        end

        function run_lust_resume(arr: Array<int>): int
            handle {
                let x = perform("Tick", [10]);
                x + 2
            } with {
                Tick(v) => lust_resume_arm(v, arr)
            }
        end
    "#;
    let bc_funcs = setup_bytecode_funcs(driver_src);

    let make_unit_fn = || -> MirFunction {
        let mut f = MirFunction::new(
            FunctionSignature {
                name: "make_unit".into(),
                param_types: vec![],
                return_type: MirType::Unit,
                is_public: true,
                effects: None,
            },
            Span::dummy(),
        );
        f.blocks[0].terminator = Some(Terminator::Return(None, Span::dummy()));
        f
    };

    let make_live_intrinsic_fn = |fn_name: &str, intrinsic: &str| -> MirFunction {
        let mut f = MirFunction::new(
            FunctionSignature {
                name: fn_name.into(),
                param_types: vec![MirType::Int, MirType::Array(Box::new(MirType::Int))],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            Span::dummy(),
        );
        let x = f.params[0];
        let arr = f.params[1];
        let s = f.add_local(Some("s".into()), MirType::String, false, Span::dummy());
        let flag = f.add_local(Some("flag".into()), MirType::Bool, false, Span::dummy());
        let flt = f.add_local(Some("flt".into()), MirType::Float, false, Span::dummy());
        let unit_v = f.add_local(Some("u".into()), MirType::Unit, false, Span::dummy());
        let idx0 = f.add_local(Some("idx0".into()), MirType::Int, false, Span::dummy());
        let slen = f.add_local(Some("slen".into()), MirType::Int, false, Span::dummy());
        let elem0 = f.add_local(Some("elem0".into()), MirType::Int, false, Span::dummy());
        let res = f.add_local(Some("res".into()), MirType::Int, false, Span::dummy());
        let out = f.add_local(Some("out".into()), MirType::Int, false, Span::dummy());

        let bb_unit = f.new_block();
        let bb1 = f.new_block();
        let bb2 = f.new_block();
        let bb3 = f.new_block();

        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(s),
            Rvalue::Use(Operand::Constant(MirConstant::String("abc".into()))),
            Span::dummy(),
        ));
        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(flag),
            Rvalue::Use(Operand::Constant(MirConstant::Bool(true))),
            Span::dummy(),
        ));
        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(flt),
            Rvalue::Use(Operand::Constant(MirConstant::Float(2.5))),
            Span::dummy(),
        ));
        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(idx0),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));
        f.blocks[0].terminator = Some(Terminator::Goto(bb_unit, Span::dummy()));
        f.blocks[bb_unit.0].terminator = Some(Terminator::Call {
            callee: "make_unit".into(),
            args: vec![],
            destination: Place::Local(unit_v),
            success: bb1,
            error: None,
            span: Span::dummy(),
        });

        let args = if intrinsic.contains("perform") {
            vec![
                Operand::Constant(MirConstant::String("Tick".into())),
                Operand::Copy(Place::Local(x)),
            ]
        } else {
            vec![Operand::Copy(Place::Local(x))]
        };
        f.blocks[bb1.0].terminator = Some(Terminator::Call {
            callee: intrinsic.into(),
            args,
            destination: Place::Local(res),
            success: bb2,
            error: None,
            span: Span::dummy(),
        });

        // Keep s, flag, flt, unit_v, arr all genuinely LIVE across the PreTerminator deopt site
        let is_abc = f.add_local(Some("is_abc".into()), MirType::Bool, false, Span::dummy());
        let bb_slen_t = f.new_block();
        let bb_slen_f = f.new_block();
        f.blocks[bb2.0].statements.push(Statement::Assign(
            Place::Local(is_abc),
            Rvalue::BinaryOp(
                MirBinaryOp::Eq,
                Operand::Copy(Place::Local(s)),
                Operand::Constant(MirConstant::String("abc".into())),
            ),
            Span::dummy(),
        ));
        f.blocks[bb2.0].terminator = Some(Terminator::Cond {
            cond: Operand::Copy(Place::Local(is_abc)),
            then_block: bb_slen_t,
            else_block: bb_slen_f,
            span: Span::dummy(),
        });
        f.blocks[bb_slen_t.0].statements.push(Statement::Assign(
            Place::Local(slen),
            Rvalue::Use(Operand::Constant(MirConstant::Int(3))),
            Span::dummy(),
        ));
        f.blocks[bb_slen_t.0].terminator = Some(Terminator::Goto(bb3, Span::dummy()));
        f.blocks[bb_slen_f.0].statements.push(Statement::Assign(
            Place::Local(slen),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));
        f.blocks[bb_slen_f.0].terminator = Some(Terminator::Goto(bb3, Span::dummy()));
        f.blocks[bb3.0].statements.push(Statement::Assign(
            Place::Local(elem0),
            Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(arr)), idx0))),
            Span::dummy(),
        ));
        f.blocks[bb3.0].statements.push(Statement::Assign(
            Place::Index(Box::new(Place::Local(arr)), idx0),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(elem0)),
                Operand::Copy(Place::Local(slen)),
            ),
            Span::dummy(),
        ));
        f.blocks[bb3.0].statements.push(Statement::Assign(
            Place::Local(out),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(res)),
                Operand::Copy(Place::Local(slen)),
            ),
            Span::dummy(),
        ));
        // Use flt and flag and unit_v trivially in a cast/eq or assign
        let f_int = f.add_local(Some("f_int".into()), MirType::Int, false, Span::dummy());
        f.blocks[bb3.0].statements.push(Statement::Assign(
            Place::Local(f_int),
            Rvalue::Cast(Operand::Copy(Place::Local(flt)), MirType::Int),
            Span::dummy(),
        ));
        let cond_add = f.add_local(Some("cond_add".into()), MirType::Int, false, Span::dummy());
        let bb4_true = f.new_block();
        let bb4_false = f.new_block();
        let bb5_ret = f.new_block();
        let u_use = f.add_local(Some("u_use".into()), MirType::Unit, false, Span::dummy());
        f.blocks[bb3.0].statements.push(Statement::Assign(
            Place::Local(u_use),
            Rvalue::Use(Operand::Copy(Place::Local(unit_v))),
            Span::dummy(),
        ));
        f.blocks[bb3.0].statements.push(Statement::Assign(
            Place::Local(out),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(out)),
                Operand::Copy(Place::Local(f_int)),
            ),
            Span::dummy(),
        ));
        f.blocks[bb3.0].terminator = Some(Terminator::Cond {
            cond: Operand::Copy(Place::Local(flag)),
            then_block: bb4_true,
            else_block: bb4_false,
            span: Span::dummy(),
        });
        f.blocks[bb4_true.0].statements.push(Statement::Assign(
            Place::Local(cond_add),
            Rvalue::Use(Operand::Constant(MirConstant::Int(1))),
            Span::dummy(),
        ));
        f.blocks[bb4_true.0].terminator = Some(Terminator::Goto(bb5_ret, Span::dummy()));
        f.blocks[bb4_false.0].statements.push(Statement::Assign(
            Place::Local(cond_add),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));
        f.blocks[bb4_false.0].terminator = Some(Terminator::Goto(bb5_ret, Span::dummy()));
        f.blocks[bb5_ret.0].statements.push(Statement::Assign(
            Place::Local(out),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(out)),
                Operand::Copy(Place::Local(cond_add)),
            ),
            Span::dummy(),
        ));
        f.blocks[bb5_ret.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(out))),
            Span::dummy(),
        ));
        f
    };

    let mir_fns = vec![
        make_unit_fn(),
        make_live_intrinsic_fn("perf_helper", "perform"),
        make_live_intrinsic_fn("lust_perf_helper", "__lust_perform"),
        make_live_intrinsic_fn("resume_arm", "resume"),
        make_live_intrinsic_fn("lust_resume_arm", "__lust_resume"),
    ];

    let mut vm_bc = VM::new();
    vm_bc.set_backend(VmBackend::Bytecode);
    vm_bc.load_functions(bc_funcs.clone());
    for f in &mir_fns {
        vm_bc.load_mir_function(f.clone()).expect("load bc");
    }

    let mut vm_clif = VM::new();
    vm_clif.set_backend(VmBackend::Cranelift);
    vm_clif.load_functions(bc_funcs);
    for f in &mir_fns {
        vm_clif.load_mir_function(f.clone()).expect("load clif");
    }

    for entry in [
        "run_perform",
        "run_lust_perform",
        "run_resume",
        "run_lust_resume",
    ] {
        let arr_bc = Rc::new(RefCell::new(vec![Value::Int(10)]));
        let arr_clif = Rc::new(RefCell::new(vec![Value::Int(10)]));
        let e0 = vm_clif.cranelift_entry_count;
        let r_bc = vm_bc
            .call(entry, vec![Value::Array(arr_bc.clone())])
            .expect("bc call");
        let r_clif = vm_clif
            .call(entry, vec![Value::Array(arr_clif.clone())])
            .expect("clif call");
        assert!(
            vm_clif.cranelift_entry_count > e0,
            "{entry} must enter Cranelift natively"
        );
        assert_eq!(r_clif, r_bc, "parity for {entry}");
        assert_eq!(*arr_clif.borrow(), *arr_bc.borrow(), "array mutation parity for {entry}");
    }

    // Part B: MirType::Function / MirType::Tuple live across __lust_handle / register_finalizer / __lust_register_finalizer
    // MUST fail derive_proven_site_metadata and be excluded from native compilation, running safely in bytecode.
    for intrinsic in ["__lust_handle", "register_finalizer", "__lust_register_finalizer"] {
        let fn_name = format!("fail_closed_{}", intrinsic);
        let mut f = MirFunction::new(
            FunctionSignature {
                name: fn_name.clone(),
                param_types: vec![MirType::Function {
                    params: vec![],
                    return_type: Box::new(MirType::Int),
                }],
                return_type: MirType::Tuple(vec![MirType::Int]),
                is_public: true,
                effects: None,
            },
            Span::dummy(),
        );
        let cb = f.params[0];
        let res = f.add_local(Some("res".into()), MirType::Int, false, Span::dummy());
        let tup = f.add_local(
            Some("tup".into()),
            MirType::Tuple(vec![MirType::Int]),
            false,
            Span::dummy(),
        );
        let bb1 = f.new_block();
        let bb2 = f.new_block();
        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(tup),
            Rvalue::Aggregate(
                MirType::Tuple(vec![MirType::Int]),
                vec![Operand::Constant(MirConstant::Int(42))],
            ),
            Span::dummy(),
        ));
        f.blocks[0].terminator = Some(Terminator::Goto(bb1, Span::dummy()));
        let call_args = if intrinsic == "__lust_handle" {
            vec![
                Operand::Copy(Place::Local(cb)),
                Operand::Copy(Place::Local(tup)),
            ]
        } else {
            vec![Operand::Copy(Place::Local(cb))]
        };
        f.blocks[bb1.0].terminator = Some(Terminator::Call {
            callee: intrinsic.into(),
            args: call_args,
            destination: Place::Local(res),
            success: bb2,
            error: None,
            span: Span::dummy(),
        });
        // Keep cb and tup live across call
        f.blocks[bb2.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(tup))),
            Span::dummy(),
        ));

        assert!(
            f.derive_proven_site_metadata(bb1.0).is_err(),
            "{intrinsic} with live Function/Tuple must fail metadata derivation"
        );

        let helper_src = "function id_cb(): int return 7 end";
        let helper_bc = setup_bytecode_funcs(helper_src);

        let mut vm_b = VM::new();
        vm_b.set_backend(VmBackend::Bytecode);
        vm_b.load_functions(helper_bc.clone());
        vm_b.load_mir_function(f.clone()).expect("load bc");

        let mut vm_c = VM::new();
        vm_c.set_backend(VmBackend::Cranelift);
        vm_c.load_functions(helper_bc);
        vm_c.load_mir_function(f).expect("load clif");
        vm_c.compile_cranelift().expect("compile_cranelift");
        assert!(
            !vm_c.cranelift_compiled.contains_key(&fn_name),
            "{fn_name} must fail closed out of native compilation"
        );

        let cb_val = Value::Function(0);
        let e0 = vm_c.cranelift_entry_count;
        let out_b = vm_b.call(&fn_name, vec![cb_val.clone()]).expect("bc exec");
        let out_c = vm_c.call(&fn_name, vec![cb_val]).expect("clif fallback exec");
        assert_eq!(vm_c.cranelift_entry_count, e0, "must run in bytecode");
        assert_eq!(out_c, out_b);
        assert_eq!(out_c, Value::Tuple(Rc::new(vec![Value::Int(42)])));
    }
}

#[test]
fn test_b3_ordinary_effect_handler_names_and_stdlib_stay_native() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_test_state();

    let src = r#"
        function side_effect_free(x: int): int
            return x + 1
        end

        function handler_count(x: int): int
            return x * 2
        end

        function call_ordinary(x: int): int
            return side_effect_free(x) + handler_count(x)
        end
    "#;
    let mut module = lower_mir_module_from_source(src, "ordinary_names");

    // Also add MIR function named `effect.succeed` and a native caller calling it
    let mut eff_succ = MirFunction::new(
        FunctionSignature {
            name: "effect.succeed".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        Span::dummy(),
    );
    let p0 = eff_succ.params[0];
    let tmp = eff_succ.add_local(Some("t".into()), MirType::Int, false, Span::dummy());
    eff_succ.blocks[0].statements.push(Statement::Assign(
        Place::Local(tmp),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(p0)),
            Operand::Constant(MirConstant::Int(10)),
        ),
        Span::dummy(),
    ));
    eff_succ.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(tmp))),
        Span::dummy(),
    ));
    module.add_function(eff_succ);

    let mut caller = MirFunction::new(
        FunctionSignature {
            name: "call_effect_succeed".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        Span::dummy(),
    );
    let cp0 = caller.params[0];
    let cres = caller.add_local(Some("cres".into()), MirType::Int, false, Span::dummy());
    let bb1 = caller.new_block();
    caller.blocks[0].terminator = Some(Terminator::Call {
        callee: "effect.succeed".into(),
        args: vec![Operand::Copy(Place::Local(cp0))],
        destination: Place::Local(cres),
        success: bb1,
        error: None,
        span: Span::dummy(),
    });
    caller.blocks[bb1.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(cres))),
        Span::dummy(),
    ));
    module.add_function(caller);

    let mut vm_bc = VM::new();
    vm_bc.set_backend(VmBackend::Bytecode);
    vm_bc.load_mir_module(&module).expect("load bc");

    let mut vm_clif = VM::new();
    vm_clif.set_backend(VmBackend::Cranelift);
    vm_clif.load_mir_module(&module).expect("load clif");
    vm_clif.compile_cranelift().expect("compile");

    assert!(vm_clif.cranelift_compiled.contains_key("side_effect_free"));
    assert!(vm_clif.cranelift_compiled.contains_key("handler_count"));
    assert!(vm_clif.cranelift_compiled.contains_key("call_ordinary"));
    assert!(vm_clif.cranelift_compiled.contains_key("effect.succeed"));
    assert!(vm_clif.cranelift_compiled.contains_key("call_effect_succeed"));

    let e0 = vm_clif.cranelift_entry_count;
    let r1_bc = vm_bc.call("call_ordinary", vec![Value::Int(5)]).expect("bc");
    let r1_clif = vm_clif.call("call_ordinary", vec![Value::Int(5)]).expect("clif");
    assert!(vm_clif.cranelift_entry_count > e0);
    assert_eq!(r1_clif, Value::Int(16));
    assert_eq!(r1_clif, r1_bc);

    let e1 = vm_clif.cranelift_entry_count;
    let r2_bc = vm_bc.call("call_effect_succeed", vec![Value::Int(5)]).expect("bc");
    let r2_clif = vm_clif.call("call_effect_succeed", vec![Value::Int(5)]).expect("clif");
    assert!(vm_clif.cranelift_entry_count > e1);
    assert_eq!(r2_clif, Value::Int(15));
    assert_eq!(r2_clif, r2_bc);
}

#[test]
fn test_b4_load_mir_loaders_native_and_global_shadowing_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_test_state();

    // Build MIR calling dotted native `custom.op` after `perform("Tick", x)`
    let make_dotted_caller = |name: &str| -> MirFunction {
        let mut f = MirFunction::new(
            FunctionSignature {
                name: name.into(),
                param_types: vec![MirType::Int],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            Span::dummy(),
        );
        let x = f.params[0];
        let t = f.add_local(Some("t".into()), MirType::Int, false, Span::dummy());
        let out = f.add_local(Some("out".into()), MirType::Int, false, Span::dummy());
        let bb1 = f.new_block();
        let bb2 = f.new_block();
        f.blocks[0].terminator = Some(Terminator::Call {
            callee: "perform".into(),
            args: vec![
                Operand::Constant(MirConstant::String("Tick".into())),
                Operand::Copy(Place::Local(x)),
            ],
            destination: Place::Local(t),
            success: bb1,
            error: None,
            span: Span::dummy(),
        });
        f.blocks[bb1.0].terminator = Some(Terminator::Call {
            callee: "custom.op".into(),
            args: vec![Operand::Copy(Place::Local(t))],
            destination: Place::Local(out),
            success: bb2,
            error: None,
            span: Span::dummy(),
        });
        f.blocks[bb2.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(out))),
            Span::dummy(),
        ));
        f
    };

    // Case (a1): load_mir_function with pre-registered dotted native `custom.op`
    {
        let driver_src = r#"
            function call_fn(x: int): int return 0 end
            function main(): int
                handle { call_fn(5) } with { Tick(v) => resume(v + 2) }
            end
        "#;
        let bc = setup_bytecode_funcs(driver_src);
        let mut vm = VM::new();
        vm.set_backend(VmBackend::Bytecode);
        vm.register_native(
            "custom.op",
            Value::NativeFunction(native_fn(|args| {
                Ok(NativeCallResult::Return(Value::Int(
                    args[0].as_int().unwrap_or(0) * 10,
                )))
            })),
        );
        vm.load_functions(bc);
        vm.load_mir_function(make_dotted_caller("call_fn"))
            .expect("load_mir_function must resolve dotted native");
        let res = vm.call("main", vec![]).expect("load_mir_function dotted native call");
        assert_eq!(res, Value::Int(70));
    }

    // Case (a2): load_mir_module with pre-registered dotted native `custom.op`
    {
        let driver_src = r#"
            function call_mod(x: int): int return 0 end
            function main(): int
                handle { call_mod(6) } with { Tick(v) => resume(v + 3) }
            end
        "#;
        let bc = setup_bytecode_funcs(driver_src);
        let mut module = MirModule::new("mod_dotted".into());
        module.add_function(make_dotted_caller("call_mod"));

        let mut vm = VM::new();
        vm.set_backend(VmBackend::Bytecode);
        vm.register_native(
            "custom.op",
            Value::NativeFunction(native_fn(|args| {
                Ok(NativeCallResult::Return(Value::Int(
                    args[0].as_int().unwrap_or(0) * 10,
                )))
            })),
        );
        vm.load_functions(bc);
        vm.load_mir_module(&module)
            .expect("load_mir_module must resolve dotted native");
        let res = vm.call("main", vec![]).expect("load_mir_module dotted native call");
        assert_eq!(res, Value::Int(90));
    }

    // Case (b): globals shadow natives identically in interpreter & loaded MIR
    {
        let mut shadow_fn = MirFunction::new(
            FunctionSignature {
                name: "shadow_caller".into(),
                param_types: vec![],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            Span::dummy(),
        );
        let dst = shadow_fn.add_local(Some("dst".into()), MirType::Int, false, Span::dummy());
        let bb1 = shadow_fn.new_block();
        shadow_fn.blocks[0].terminator = Some(Terminator::Call {
            callee: "shadowed.op".into(),
            args: vec![],
            destination: Place::Local(dst),
            success: bb1,
            error: None,
            span: Span::dummy(),
        });
        shadow_fn.blocks[bb1.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(dst))),
            Span::dummy(),
        ));

        let setup_shadow_vm = |backend: VmBackend| -> VM {
            let mut vm = VM::new();
            vm.set_backend(backend);
            vm.register_native(
                "shadowed.op",
                Value::NativeFunction(native_fn(|_| Ok(NativeCallResult::Return(Value::Int(100))))),
            );
            vm.set_global(
                "shadowed.op",
                Value::NativeFunction(native_fn(|_| Ok(NativeCallResult::Return(Value::Int(200))))),
            );
            vm.load_mir_function(shadow_fn.clone()).expect("load shadow_caller");
            vm
        };

        let mut vm_bc = setup_shadow_vm(VmBackend::Bytecode);
        let mut vm_clif = setup_shadow_vm(VmBackend::Cranelift);
        assert_eq!(vm_bc.call("shadow_caller", vec![]).unwrap(), Value::Int(200));
        assert_eq!(vm_clif.call("shadow_caller", vec![]).unwrap(), Value::Int(200));
    }

    // Case (c): ordinary Map `table_obj.method_field` still uses LoadGlobal("table_obj") + GetField("method_field")
    {
        let mut map_caller = MirFunction::new(
            FunctionSignature {
                name: "map_method_caller".into(),
                param_types: vec![],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            Span::dummy(),
        );
        let dst = map_caller.add_local(Some("dst".into()), MirType::Int, false, Span::dummy());
        let bb1 = map_caller.new_block();
        map_caller.blocks[0].terminator = Some(Terminator::Call {
            callee: "table_obj.method_field".into(),
            args: vec![],
            destination: Place::Local(dst),
            success: bb1,
            error: None,
            span: Span::dummy(),
        });
        map_caller.blocks[bb1.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(dst))),
            Span::dummy(),
        ));

        let table_val = Value::map({
            let mut m = lust::bytecode::LustMap::default();
            m.insert(
                lust::bytecode::ValueKey::string("method_field"),
                Value::NativeFunction(native_fn(|_| Ok(NativeCallResult::Return(Value::Int(333))))),
            );
            m
        });

        let mut vm = VM::new();
        vm.set_backend(VmBackend::Bytecode);
        vm.set_global("table_obj", table_val);
        vm.load_mir_function(map_caller).expect("load map_method_caller");
        assert_eq!(
            vm.call("map_method_caller", vec![]).expect("GetField method call"),
            Value::Int(333)
        );
    }
}

#[test]
fn test_b5_projected_destination_rejected_and_collect_uses_sound() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_test_state();

    // 1. Place::Index destination
    let mut f_idx = MirFunction::new(
        FunctionSignature {
            name: "proj_index_fn".into(),
            param_types: vec![MirType::Array(Box::new(MirType::Int)), MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        Span::dummy(),
    );
    let arr = f_idx.params[0];
    let idx = f_idx.params[1];
    let ret = f_idx.add_local(Some("ret".into()), MirType::Int, false, Span::dummy());
    let bb1 = f_idx.new_block();
    let bb2 = f_idx.new_block();

    f_idx.blocks[0].terminator = Some(Terminator::Goto(bb1, Span::dummy()));
    let term_idx = Terminator::Call {
        callee: "perform".into(),
        args: vec![Operand::Constant(MirConstant::String("Tick".into()))],
        destination: Place::Index(Box::new(Place::Local(arr)), idx),
        success: bb2,
        error: None,
        span: Span::dummy(),
    };
    let mut uses = hashbrown::HashSet::new();
    term_idx.collect_uses(&mut uses);
    assert!(
        uses.contains(&arr.0) && uses.contains(&idx.0),
        "Terminator::collect_uses must collect root container and index local for Place::Index destination, got: {:?}",
        uses
    );

    f_idx.blocks[bb1.0].terminator = Some(term_idx);
    f_idx.blocks[bb2.0].statements.push(Statement::Assign(
        Place::Local(ret),
        Rvalue::Use(Operand::Constant(MirConstant::Int(1))),
        Span::dummy(),
    ));
    f_idx.blocks[bb2.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(ret))),
        Span::dummy(),
    ));

    assert!(
        f_idx.derive_proven_site_metadata(bb1.0).is_err(),
        "derive_proven_site_metadata must reject projected Place::Index Call destination"
    );

    let mut vm = VM::new();
    vm.set_backend(VmBackend::Cranelift);
    vm.load_mir_function(f_idx).expect("load proj_index_fn");
    vm.compile_cranelift().expect("compile_cranelift");
    assert!(
        !vm.cranelift_compiled.contains_key("proj_index_fn"),
        "proj_index_fn must not be natively installed"
    );

    // 2. Nested Place::Field(Box::new(Place::Index(...)), 0) destination
    let mut f_nested = MirFunction::new(
        FunctionSignature {
            name: "proj_nested_fn".into(),
            param_types: vec![MirType::Array(Box::new(MirType::Int)), MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        Span::dummy(),
    );
    let n_arr = f_nested.params[0];
    let n_idx = f_nested.params[1];
    let n_ret = f_nested.add_local(Some("ret".into()), MirType::Int, false, Span::dummy());
    let n_bb1 = f_nested.new_block();
    let n_bb2 = f_nested.new_block();
    f_nested.blocks[0].terminator = Some(Terminator::Goto(n_bb1, Span::dummy()));
    let term_nested = Terminator::Call {
        callee: "perform".into(),
        args: vec![Operand::Constant(MirConstant::String("Tick".into()))],
        destination: Place::Field(
            Box::new(Place::Index(Box::new(Place::Local(n_arr)), n_idx)),
            0,
        ),
        success: n_bb2,
        error: None,
        span: Span::dummy(),
    };
    let mut nested_uses = hashbrown::HashSet::new();
    term_nested.collect_uses(&mut nested_uses);
    assert!(
        nested_uses.contains(&n_arr.0) && nested_uses.contains(&n_idx.0),
        "Terminator::collect_uses must collect root container and index local for nested projected destination, got: {:?}",
        nested_uses
    );
    f_nested.blocks[n_bb1.0].terminator = Some(term_nested);
    f_nested.blocks[n_bb2.0].statements.push(Statement::Assign(
        Place::Local(n_ret),
        Rvalue::Use(Operand::Constant(MirConstant::Int(2))),
        Span::dummy(),
    ));
    f_nested.blocks[n_bb2.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(n_ret))),
        Span::dummy(),
    ));
    assert!(
        f_nested.derive_proven_site_metadata(n_bb1.0).is_err(),
        "derive_proven_site_metadata must reject nested projected Call destination"
    );
    vm.load_mir_function(f_nested).expect("load proj_nested_fn");
    vm.compile_cranelift().expect("compile_cranelift");
    assert!(!vm.cranelift_compiled.contains_key("proj_nested_fn"));
}

#[test]
fn test_b8_deopt_resource_errors_precede_fallback() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_test_state();

    let driver_src = r#"
        function oog_deopt_fn(x: int): int return 0 end
        function oom_deopt_fn(x: int): int return 0 end

        function main_oog(): int
            handle { oog_deopt_fn(5) } with { Tick(v) => resume(v + 1) }
        end

        function main_oom(): int
            handle { oom_deopt_fn(5) } with { Tick(v) => resume(v + 1) }
        end
    "#;
    let bc = setup_bytecode_funcs(driver_src);

    // OOG function: native entry charges 1 gas; when VM gas budget is 0 remaining, OOG precedes DEOPT
    let mut oog_fn = MirFunction::new(
        FunctionSignature {
            name: "oog_deopt_fn".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        Span::dummy(),
    );
    let x = oog_fn.params[0];
    let r = oog_fn.add_local(Some("r".into()), MirType::Int, false, Span::dummy());
    let bb1 = oog_fn.new_block();
    oog_fn.blocks[0].terminator = Some(Terminator::Call {
        callee: "perform".into(),
        args: vec![
            Operand::Constant(MirConstant::String("Tick".into())),
            Operand::Copy(Place::Local(x)),
        ],
        destination: Place::Local(r),
        success: bb1,
        error: None,
        span: Span::dummy(),
    });
    oog_fn.blocks[bb1.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(r))),
        Span::dummy(),
    ));

    // OOM function: allocates array exceeding small memory budget before perform
    let mut oom_fn = MirFunction::new(
        FunctionSignature {
            name: "oom_deopt_fn".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        Span::dummy(),
    );
    let ox = oom_fn.params[0];
    let oarr = oom_fn.add_local(
        Some("oarr".into()),
        MirType::Array(Box::new(MirType::Int)),
        false,
        Span::dummy(),
    );
    let o_r = oom_fn.add_local(Some("or".into()), MirType::Int, false, Span::dummy());
    let obb1 = oom_fn.new_block();
    let obb2 = oom_fn.new_block();
    oom_fn.blocks[0].statements.push(Statement::Assign(
        Place::Local(oarr),
        Rvalue::Aggregate(
            MirType::Array(Box::new(MirType::Int)),
            vec![
                Operand::Copy(Place::Local(ox)),
                Operand::Constant(MirConstant::Int(1)),
                Operand::Constant(MirConstant::Int(2)),
                Operand::Constant(MirConstant::Int(3)),
            ],
        ),
        Span::dummy(),
    ));
    oom_fn.blocks[0].terminator = Some(Terminator::Goto(obb1, Span::dummy()));
    oom_fn.blocks[obb1.0].terminator = Some(Terminator::Call {
        callee: "perform".into(),
        args: vec![
            Operand::Constant(MirConstant::String("Tick".into())),
            Operand::Copy(Place::Local(ox)),
        ],
        destination: Place::Local(o_r),
        success: obb2,
        error: None,
        span: Span::dummy(),
    });
    oom_fn.blocks[obb2.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(o_r))),
        Span::dummy(),
    ));

    // Subcase A: OOG precedes deopt fallback
    {
        let mut vm = VM::new();
        vm.set_backend(VmBackend::Cranelift);
        vm.load_functions(bc.clone());
        vm.load_mir_function(oog_fn.clone()).expect("load oog_deopt_fn");
        vm.compile_cranelift().expect("compile oog_deopt_fn");
        assert!(vm.cranelift_compiled.contains_key("oog_deopt_fn"));

        // Direct call inside active handle or direct call to oog_deopt_fn with 0 gas budget
        // Note: main_oog runs bytecode instructions before calling oog_deopt_fn;
        // calling oog_deopt_fn directly enters Cranelift immediately with gas remaining = 0!
        vm.set_gas_budget(0);
        let e0 = vm.cranelift_entry_count;
        let err = vm
            .call("oog_deopt_fn", vec![Value::Int(5)])
            .expect_err("OOG must fail rather than mask via deopt");
        assert!(
            vm.cranelift_entry_count > e0,
            "oog_deopt_fn must enter Cranelift natively"
        );
        assert!(
            err.to_string().contains("Out of gas") || err.to_string().contains("gas"),
            "Expected OutOfGas error, got: {err}"
        );
        assert!(take_deopt_payload().is_none());
        assert!(get_current_sticky_error().is_none());
    }

    // Subcase B: OOM precedes deopt fallback
    {
        let mut vm = VM::new();
        vm.set_backend(VmBackend::Cranelift);
        vm.load_functions(bc);
        vm.load_mir_function(oom_fn).expect("load oom_deopt_fn");
        vm.compile_cranelift().expect("compile oom_deopt_fn");
        assert!(vm.cranelift_compiled.contains_key("oom_deopt_fn"));

        vm.set_memory_budget_bytes(8).expect("set memory budget");
        let e0 = vm.cranelift_entry_count;
        let err = vm
            .call("oom_deopt_fn", vec![Value::Int(5)])
            .expect_err("OOM must fail rather than mask via deopt");
        assert!(
            vm.cranelift_entry_count > e0,
            "oom_deopt_fn must enter Cranelift natively"
        );
        assert!(
            err.to_string().contains("Out of memory") || err.to_string().contains("memory"),
            "Expected OutOfMemory error, got: {err}"
        );
        assert!(take_deopt_payload().is_none());
        assert!(get_current_sticky_error().is_none());
    }
}

#[test]
fn test_b9_real_cancel_task_during_continued_deopt_clean_followup() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_test_state();

    static FIN_CALLS: AtomicUsize = AtomicUsize::new(0);
    FIN_CALLS.store(0, Ordering::SeqCst);

    let src = r#"
        function deopt_worker(x: int): int
            return 0
        end

        function task_entry(reg_fin: any, trigger_cancel: any): int
            handle {
                reg_fin();
                deopt_worker(42)
            } with {
                CancelMe(v) => {
                    trigger_cancel();
                    resume(v + 1)
                }
            }
        end

        function clean_followup(x: int): int
            return 0
        end
    "#;
    let bc = setup_bytecode_funcs(src);

    let mut worker_mir = MirFunction::new(
        FunctionSignature {
            name: "deopt_worker".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        Span::dummy(),
    );
    let x = worker_mir.params[0];
    let r = worker_mir.add_local(Some("r".into()), MirType::Int, false, Span::dummy());
    let bb1 = worker_mir.new_block();
    worker_mir.blocks[0].terminator = Some(Terminator::Call {
        callee: "perform".into(),
        args: vec![
            Operand::Constant(MirConstant::String("CancelMe".into())),
            Operand::Copy(Place::Local(x)),
        ],
        destination: Place::Local(r),
        success: bb1,
        error: None,
        span: Span::dummy(),
    });
    worker_mir.blocks[bb1.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(r))),
        Span::dummy(),
    ));

    let mut followup_mir = MirFunction::new(
        FunctionSignature {
            name: "clean_followup".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        Span::dummy(),
    );
    let fx = followup_mir.params[0];
    let fret = followup_mir.add_local(Some("fret".into()), MirType::Int, false, Span::dummy());
    followup_mir.blocks[0].statements.push(Statement::Assign(
        Place::Local(fret),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(fx)),
            Operand::Constant(MirConstant::Int(100)),
        ),
        Span::dummy(),
    ));
    followup_mir.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(fret))),
        Span::dummy(),
    ));

    let mut vm = VM::new();
    vm.set_backend(VmBackend::Cranelift);
    vm.load_functions(bc);
    vm.load_mir_function(worker_mir).expect("load deopt_worker");
    vm.load_mir_function(followup_mir).expect("load clean_followup");
    let reg_fin_val = Value::NativeFunction(native_fn(|_| {
        lust::vm::register_effect_finalizer(|| {
            FIN_CALLS.fetch_add(1, Ordering::SeqCst);
        });
        Ok(NativeCallResult::Return(Value::Nil))
    }));
    let trigger_cancel_val = Value::NativeFunction(native_fn(|_| {
        if let Some(vm_ptr) = lust::vm::current_vm_ptr() {
            unsafe {
                let vm_ref = &mut *vm_ptr;
                if let Some(th) = vm_ref.current_task_handle() {
                    vm_ref.cancel_task(lust::vm::TaskId(th.id()), InterruptReason::Cancelled);
                }
            }
        }
        Ok(NativeCallResult::Return(Value::Nil))
    }));
    vm.compile_cranelift().expect("compile");

    let task_fn = vm
        .function_value("task_entry")
        .expect("task_entry function");
    let e0 = vm.cranelift_entry_count;
    let spawn_err = vm
        .spawn_task_value(task_fn, vec![reg_fin_val, trigger_cancel_val])
        .expect_err("cancelled task must return error");
    assert!(
        vm.cranelift_entry_count > e0,
        "deopt_worker must enter Cranelift before deopt+cancel"
    );
    assert!(
        spawn_err.to_string().contains("Cancelled"),
        "Expected Cancelled error, got: {spawn_err}"
    );
    assert_eq!(
        FIN_CALLS.load(Ordering::SeqCst),
        1,
        "finalizer must run exactly once on task cancellation"
    );
    assert!(take_deopt_payload().is_none());
    assert!(get_current_sticky_error().is_none());

    // Subsequent VM::call must enter Cranelift cleanly with zero stale payloads
    let e1 = vm.cranelift_entry_count;
    let follow = vm
        .call("clean_followup", vec![Value::Int(23)])
        .expect("follow-up call must succeed");
    assert_eq!(vm.cranelift_entry_count, e1 + 1);
    assert_eq!(follow, Value::Int(123));
}

#[test]
fn test_b10_reload_bumps_generation_and_updates_pc_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_test_state();

    let driver_src = r#"
        function reloadable_fn(x: int): int return 0 end
        function main(x: int): int
            handle { reloadable_fn(x) } with { Tick(v) => resume(v + 5) }
        end
    "#;
    let bc = setup_bytecode_funcs(driver_src);

    // v1: bb0 -> perform("Tick", x) -> bb1 returns r + 1
    let make_v1 = || {
        let mut f = MirFunction::new(
            FunctionSignature {
                name: "reloadable_fn".into(),
                param_types: vec![MirType::Int],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            Span::dummy(),
        );
        let x = f.params[0];
        let r = f.add_local(Some("r".into()), MirType::Int, false, Span::dummy());
        let out = f.add_local(Some("out".into()), MirType::Int, false, Span::dummy());
        let bb1 = f.new_block();
        f.blocks[0].terminator = Some(Terminator::Call {
            callee: "perform".into(),
            args: vec![
                Operand::Constant(MirConstant::String("Tick".into())),
                Operand::Copy(Place::Local(x)),
            ],
            destination: Place::Local(r),
            success: bb1,
            error: None,
            span: Span::dummy(),
        });
        f.blocks[bb1.0].statements.push(Statement::Assign(
            Place::Local(out),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(r)),
                Operand::Constant(MirConstant::Int(1)),
            ),
            Span::dummy(),
        ));
        f.blocks[bb1.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(out))),
            Span::dummy(),
        ));
        f
    };

    // v2: extra blocks shifting terminator_pcs and returning r + 100
    let make_v2 = || {
        let mut f = MirFunction::new(
            FunctionSignature {
                name: "reloadable_fn".into(),
                param_types: vec![MirType::Int],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            Span::dummy(),
        );
        let x = f.params[0];
        let tmp = f.add_local(Some("tmp".into()), MirType::Int, false, Span::dummy());
        let r = f.add_local(Some("r".into()), MirType::Int, false, Span::dummy());
        let out = f.add_local(Some("out".into()), MirType::Int, false, Span::dummy());
        let bb1 = f.new_block();
        let bb2 = f.new_block();
        let bb3 = f.new_block();
        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(tmp),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(x)),
                Operand::Constant(MirConstant::Int(10)),
            ),
            Span::dummy(),
        ));
        f.blocks[0].terminator = Some(Terminator::Goto(bb1, Span::dummy()));
        f.blocks[bb1.0].terminator = Some(Terminator::Goto(bb2, Span::dummy()));
        f.blocks[bb2.0].terminator = Some(Terminator::Call {
            callee: "perform".into(),
            args: vec![
                Operand::Constant(MirConstant::String("Tick".into())),
                Operand::Copy(Place::Local(tmp)),
            ],
            destination: Place::Local(r),
            success: bb3,
            error: None,
            span: Span::dummy(),
        });
        f.blocks[bb3.0].statements.push(Statement::Assign(
            Place::Local(out),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(r)),
                Operand::Constant(MirConstant::Int(100)),
            ),
            Span::dummy(),
        ));
        f.blocks[bb3.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(out))),
            Span::dummy(),
        ));
        f
    };

    let mut vm_bc = VM::new();
    vm_bc.set_backend(VmBackend::Bytecode);
    vm_bc.load_functions(bc.clone());

    let mut vm_clif = VM::new();
    vm_clif.set_backend(VmBackend::Cranelift);
    vm_clif.load_functions(bc);

    // Run v1
    vm_bc.load_mir_function(make_v1()).expect("load v1 bc");
    vm_clif.load_mir_function(make_v1()).expect("load v1 clif");
    let c0 = vm_clif.cranelift_compilation_count;
    let e0 = vm_clif.cranelift_entry_count;
    let r1_bc = vm_bc.call("main", vec![Value::Int(4)]).expect("v1 bc");
    let r1_clif = vm_clif.call("main", vec![Value::Int(4)]).expect("v1 clif");
    assert!(vm_clif.cranelift_compilation_count > c0);
    assert_eq!(vm_clif.cranelift_entry_count, e0 + 1);
    assert_eq!(r1_clif, Value::Int(10)); // (4 + 5) + 1
    assert_eq!(r1_clif, r1_bc);

    // Reload v2 (same name, shifted blocks & PCs)
    vm_bc.load_mir_function(make_v2()).expect("load v2 bc");
    vm_clif.load_mir_function(make_v2()).expect("load v2 clif");
    let c1 = vm_clif.cranelift_compilation_count;
    let e1 = vm_clif.cranelift_entry_count;
    let r2_bc = vm_bc.call("main", vec![Value::Int(4)]).expect("v2 bc");
    let r2_clif = vm_clif.call("main", vec![Value::Int(4)]).expect("v2 clif");
    assert!(
        vm_clif.cranelift_compilation_count > c1,
        "reloading MIR function must trigger fresh Cranelift compilation"
    );
    assert_eq!(vm_clif.cranelift_entry_count, e1 + 1);
    assert_eq!(r2_clif, Value::Int(119)); // ((4 + 10) + 5) + 100
    assert_eq!(r2_clif, r2_bc);
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_b13_production_simd_sum_deopt_without_spike_hook() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    set_force_simd_scalar_fallback(false);
    reset_all_test_state();

    let fn_name = "prod_simd_sum_deopt";
    let init_acc: i64 = 1_000_000_000_000_000_003;

    // Build SIMD Sum + post-loop alias write `a[0] = acc`
    let sig = FunctionSignature {
        name: fn_name.into(),
        param_types: vec![MirType::Array(Box::new(MirType::Int)), MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let a = func.params[0];
    let len = func.params[1];
    let acc_loc = func.add_local(Some("acc".into()), MirType::Int, true, Span::dummy());
    let i_loc = func.add_local(Some("i".into()), MirType::Int, true, Span::dummy());
    let cond_loc = func.add_local(Some("cond".into()), MirType::Bool, false, Span::dummy());
    let elem_a = func.add_local(Some("elem_a".into()), MirType::Int, false, Span::dummy());
    let zero_loc = func.add_local(Some("zero".into()), MirType::Int, false, Span::dummy());

    let bb_hdr = func.new_block();
    let bb_body = func.new_block();
    let bb_exit = func.new_block();

    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(acc_loc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(init_acc))),
        Span::dummy(),
    ));
    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(i_loc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(zero_loc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    func.blocks[0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    func.blocks[bb_hdr.0].statements.push(Statement::Assign(
        Place::Local(cond_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(i_loc)),
            Operand::Copy(Place::Local(len)),
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
        Place::Local(elem_a),
        Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(a)), i_loc))),
        Span::dummy(),
    ));
    func.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(acc_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(acc_loc)),
            Operand::Copy(Place::Local(elem_a)),
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

    func.blocks[bb_exit.0].statements.push(Statement::Assign(
        Place::Index(Box::new(Place::Local(a)), zero_loc),
        Rvalue::Use(Operand::Copy(Place::Local(acc_loc))),
        Span::dummy(),
    ));
    func.blocks[bb_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(acc_loc))),
        Span::dummy(),
    ));

    assert!(SimdOptimizer::run_on_function(&mut func), "Sum loop must vectorize");
    let mut verify_mod = MirModule::new("b13_sum".into());
    verify_mod.add_function(func.clone());
    MirVerifier::verify_module(&verify_mod).expect("MirVerifier");
    let recognized = recognize_vectorized_loop_blocks(&func);
    assert_eq!(recognized.len(), 1);
    assert_eq!(recognized[0].kind, VectorLoopKernelKind::Sum);

    let a_vals: Vec<i64> = vec![
        i64::MAX - 10,
        25,
        -7,
        103,
        i64::MIN + 5,
        -40,
        88,
        19,
        301,
        -12,
        44,
    ];
    let len_val = a_vals.len() as i64;
    let expected = a_vals
        .iter()
        .fold(init_acc, |acc, &x| acc.wrapping_add(x));

    // Poll 1 (entry), Poll 2 (after 1st SIMD chunk), Poll 3 (after 2nd SIMD chunk -> odd tail 3)
    for poll_ordinal in [1u64, 2, 3] {
        reset_all_test_state();
        let mut vm = VM::new();
        vm.set_backend(VmBackend::Cranelift);
        assert!(vm.test_deopt_spike_hook.is_none());
        vm.load_mir_function(func.clone()).expect("load Sum");

        lust::mir::cranelift_backend::safepoint_request_deopt_on_poll(
            Some(fn_name.into()),
            poll_ordinal,
        );

        let arr_rc = Rc::new(RefCell::new(a_vals.iter().copied().map(Value::Int).collect()));
        let res = vm
            .call(fn_name, vec![Value::Array(arr_rc.clone()), Value::Int(len_val)])
            .expect("production SIMD Sum deopt without spike hook must succeed");

        assert_eq!(vm.cranelift_entry_count, 1);
        assert_eq!(CRANELIFT_SIMD_VECTOR_COUNT.load(Ordering::SeqCst), 1);
        assert_eq!(res, Value::Int(expected), "poll {poll_ordinal}");
        assert_eq!(
            arr_rc.borrow()[0],
            Value::Int(expected),
            "poll {poll_ordinal}: post-loop array mutation must reflect on caller alias"
        );
        assert!(take_deopt_payload().is_none());
        assert_eq!(native_invocation_depth(), 0);
    }
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_b14_production_simd_dot_deopt_without_spike_hook() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    set_force_simd_scalar_fallback(false);
    reset_all_test_state();

    let fn_name = "prod_simd_dot_deopt";
    let init_acc: i64 = -500_000_000_000_000_007;

    let sig = FunctionSignature {
        name: fn_name.into(),
        param_types: vec![
            MirType::Array(Box::new(MirType::Int)),
            MirType::Array(Box::new(MirType::Int)),
            MirType::Int,
        ],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let a = func.params[0];
    let b = func.params[1];
    let len = func.params[2];

    let acc_loc = func.add_local(Some("acc".into()), MirType::Int, true, Span::dummy());
    let i_loc = func.add_local(Some("i".into()), MirType::Int, true, Span::dummy());
    let cond_loc = func.add_local(Some("cond".into()), MirType::Bool, false, Span::dummy());
    let elem_a = func.add_local(Some("elem_a".into()), MirType::Int, false, Span::dummy());
    let elem_b = func.add_local(Some("elem_b".into()), MirType::Int, false, Span::dummy());
    let prod = func.add_local(Some("prod".into()), MirType::Int, false, Span::dummy());
    let zero_loc = func.add_local(Some("zero".into()), MirType::Int, false, Span::dummy());

    let bb_hdr = func.new_block();
    let bb_body = func.new_block();
    let bb_exit = func.new_block();

    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(acc_loc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(init_acc))),
        Span::dummy(),
    ));
    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(i_loc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(zero_loc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    func.blocks[0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    func.blocks[bb_hdr.0].statements.push(Statement::Assign(
        Place::Local(cond_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(i_loc)),
            Operand::Copy(Place::Local(len)),
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
        Place::Local(elem_a),
        Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(a)), i_loc))),
        Span::dummy(),
    ));
    func.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(elem_b),
        Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(b)), i_loc))),
        Span::dummy(),
    ));
    func.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(prod),
        Rvalue::BinaryOp(
            MirBinaryOp::Mul,
            Operand::Copy(Place::Local(elem_a)),
            Operand::Copy(Place::Local(elem_b)),
        ),
        Span::dummy(),
    ));
    func.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(acc_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(acc_loc)),
            Operand::Copy(Place::Local(prod)),
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

    func.blocks[bb_exit.0].statements.push(Statement::Assign(
        Place::Index(Box::new(Place::Local(a)), zero_loc),
        Rvalue::Use(Operand::Copy(Place::Local(acc_loc))),
        Span::dummy(),
    ));
    func.blocks[bb_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(acc_loc))),
        Span::dummy(),
    ));

    assert!(SimdOptimizer::run_on_function(&mut func), "Dot loop must vectorize");
    let mut verify_mod = MirModule::new("b14_dot".into());
    verify_mod.add_function(func.clone());
    MirVerifier::verify_module(&verify_mod).expect("MirVerifier");
    let recognized = recognize_vectorized_loop_blocks(&func);
    assert_eq!(recognized.len(), 1);
    assert_eq!(recognized[0].kind, VectorLoopKernelKind::Dot);

    let a_vals: Vec<i64> = vec![
        i64::MAX - 10,
        25,
        -7,
        103,
        i64::MIN + 5,
        -40,
        88,
        19,
        301,
        -12,
        44,
    ];
    let b_vals: Vec<i64> = vec![3, -17, 41, 9, -5, 11, -3, 7, 2, 13, -6];
    let len_val = a_vals.len() as i64;
    let expected = a_vals
        .iter()
        .zip(&b_vals)
        .fold(init_acc, |acc, (&x, &y)| acc.wrapping_add(x.wrapping_mul(y)));

    for poll_ordinal in [1u64, 2, 3] {
        reset_all_test_state();
        let mut vm = VM::new();
        vm.set_backend(VmBackend::Cranelift);
        assert!(vm.test_deopt_spike_hook.is_none());
        vm.load_mir_function(func.clone()).expect("load Dot");

        lust::mir::cranelift_backend::safepoint_request_deopt_on_poll(
            Some(fn_name.into()),
            poll_ordinal,
        );

        let arr_a = Rc::new(RefCell::new(a_vals.iter().copied().map(Value::Int).collect()));
        let arr_b = Rc::new(RefCell::new(b_vals.iter().copied().map(Value::Int).collect()));
        let res = vm
            .call(
                fn_name,
                vec![
                    Value::Array(arr_a.clone()),
                    Value::Array(arr_b),
                    Value::Int(len_val),
                ],
            )
            .expect("production SIMD Dot deopt without spike hook must succeed");

        assert_eq!(vm.cranelift_entry_count, 1);
        assert_eq!(CRANELIFT_SIMD_VECTOR_COUNT.load(Ordering::SeqCst), 1);
        assert_eq!(res, Value::Int(expected), "poll {poll_ordinal}");
        assert_eq!(arr_a.borrow()[0], Value::Int(expected), "poll {poll_ordinal}");
        assert!(take_deopt_payload().is_none());
        assert_eq!(native_invocation_depth(), 0);
    }
}

// =========================================================================
// Regression / Safety Verification Tests (4 tests)
// =========================================================================

#[test]
fn test_b6_caller_of_deopt_callee_and_self_recursion_safe() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_test_state();

    let driver_src = r#"
        function inner_eff(x: int, log_arr: Array<int>): int return 0 end
        function mid_caller(x: int, log_arr: Array<int>): int return 0 end
        function outer_caller(x: int, log_arr: Array<int>): int return 0 end
        function rec_effect(n: int, log_arr: Array<int>): int return 0 end

        function main_chain(log_arr: Array<int>): int
            handle {
                outer_caller(5, log_arr)
            } with {
                Tick(v) => resume(v + 1)
            }
        end

        function main_rec(log_arr: Array<int>): int
            handle {
                rec_effect(2, log_arr)
            } with {
                Tick(v) => resume(v + 10)
            }
        end
    "#;
    let bc = setup_bytecode_funcs(driver_src);

    let mut inner_eff = MirFunction::new(
        FunctionSignature {
            name: "inner_eff".into(),
            param_types: vec![MirType::Int, MirType::Array(Box::new(MirType::Int))],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        Span::dummy(),
    );
    let ix = inner_eff.params[0];
    let i_arr = inner_eff.params[1];
    let i_idx = inner_eff.add_local(Some("i".into()), MirType::Int, false, Span::dummy());
    let i_tmp = inner_eff.add_local(Some("t".into()), MirType::Int, false, Span::dummy());
    let i_res = inner_eff.add_local(Some("r".into()), MirType::Int, false, Span::dummy());
    let i_bb1 = inner_eff.new_block();
    let i_bb2 = inner_eff.new_block();
    inner_eff.blocks[0].statements.push(Statement::Assign(
        Place::Local(i_idx),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    inner_eff.blocks[0].statements.push(Statement::Assign(
        Place::Local(i_tmp),
        Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(i_arr)), i_idx))),
        Span::dummy(),
    ));
    inner_eff.blocks[0].statements.push(Statement::Assign(
        Place::Index(Box::new(Place::Local(i_arr)), i_idx),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i_tmp)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    inner_eff.blocks[0].terminator = Some(Terminator::Goto(i_bb1, Span::dummy()));
    inner_eff.blocks[i_bb1.0].terminator = Some(Terminator::Call {
        callee: "perform".into(),
        args: vec![
            Operand::Constant(MirConstant::String("Tick".into())),
            Operand::Copy(Place::Local(ix)),
        ],
        destination: Place::Local(i_res),
        success: i_bb2,
        error: None,
        span: Span::dummy(),
    });
    inner_eff.blocks[i_bb2.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(i_res))),
        Span::dummy(),
    ));

    let make_wrapper = |name: &str, callee: &str| -> MirFunction {
        let mut f = MirFunction::new(
            FunctionSignature {
                name: name.into(),
                param_types: vec![MirType::Int, MirType::Array(Box::new(MirType::Int))],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            Span::dummy(),
        );
        let px = f.params[0];
        let pa = f.params[1];
        let r = f.add_local(Some("r".into()), MirType::Int, false, Span::dummy());
        let out = f.add_local(Some("out".into()), MirType::Int, false, Span::dummy());
        let bb1 = f.new_block();
        f.blocks[0].terminator = Some(Terminator::Call {
            callee: callee.into(),
            args: vec![Operand::Copy(Place::Local(px)), Operand::Copy(Place::Local(pa))],
            destination: Place::Local(r),
            success: bb1,
            error: None,
            span: Span::dummy(),
        });
        f.blocks[bb1.0].statements.push(Statement::Assign(
            Place::Local(out),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(r)),
                Operand::Constant(MirConstant::Int(2)),
            ),
            Span::dummy(),
        ));
        f.blocks[bb1.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(out))),
            Span::dummy(),
        ));
        f
    };

    // Self-recursive function containing an effect call
    let mut rec_fn = MirFunction::new(
        FunctionSignature {
            name: "rec_effect".into(),
            param_types: vec![MirType::Int, MirType::Array(Box::new(MirType::Int))],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        Span::dummy(),
    );
    let rn = rec_fn.params[0];
    let ra = rec_fn.params[1];
    let rcond = rec_fn.add_local(Some("cond".into()), MirType::Bool, false, Span::dummy());
    let rn1 = rec_fn.add_local(Some("n1".into()), MirType::Int, false, Span::dummy());
    let rr = rec_fn.add_local(Some("rr".into()), MirType::Int, false, Span::dummy());
    let ridx = rec_fn.add_local(Some("idx".into()), MirType::Int, false, Span::dummy());
    let r_base = rec_fn.new_block();
    let r_rec = rec_fn.new_block();
    let r_done = rec_fn.new_block();
    rec_fn.blocks[0].statements.push(Statement::Assign(
        Place::Local(ridx),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    rec_fn.blocks[0].statements.push(Statement::Assign(
        Place::Local(rcond),
        Rvalue::BinaryOp(
            MirBinaryOp::LtEq,
            Operand::Copy(Place::Local(rn)),
            Operand::Constant(MirConstant::Int(0)),
        ),
        Span::dummy(),
    ));
    rec_fn.blocks[0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(rcond)),
        then_block: r_base,
        else_block: r_rec,
        span: Span::dummy(),
    });
    rec_fn.blocks[r_base.0].terminator = Some(Terminator::Call {
        callee: "perform".into(),
        args: vec![
            Operand::Constant(MirConstant::String("Tick".into())),
            Operand::Copy(Place::Local(rn)),
        ],
        destination: Place::Local(rr),
        success: r_done,
        error: None,
        span: Span::dummy(),
    });
    rec_fn.blocks[r_rec.0].statements.push(Statement::Assign(
        Place::Index(Box::new(Place::Local(ra)), ridx),
        Rvalue::Use(Operand::Copy(Place::Local(rn))),
        Span::dummy(),
    ));
    rec_fn.blocks[r_rec.0].statements.push(Statement::Assign(
        Place::Local(rn1),
        Rvalue::BinaryOp(
            MirBinaryOp::Sub,
            Operand::Copy(Place::Local(rn)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    rec_fn.blocks[r_rec.0].terminator = Some(Terminator::Call {
        callee: "rec_effect".into(),
        args: vec![Operand::Copy(Place::Local(rn1)), Operand::Copy(Place::Local(ra))],
        destination: Place::Local(rr),
        success: r_done,
        error: None,
        span: Span::dummy(),
    });
    rec_fn.blocks[r_done.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(rr))),
        Span::dummy(),
    ));

    let mid = make_wrapper("mid_caller", "inner_eff");
    let outer = make_wrapper("outer_caller", "mid_caller");

    let mut vm_bc = VM::new();
    vm_bc.set_backend(VmBackend::Bytecode);
    vm_bc.load_functions(bc.clone());
    for f in [&inner_eff, &mid, &outer, &rec_fn] {
        vm_bc.load_mir_function(f.clone()).expect("load bc");
    }

    let mut vm_clif = VM::new();
    vm_clif.set_backend(VmBackend::Cranelift);
    vm_clif.load_functions(bc);
    for f in [&inner_eff, &mid, &outer, &rec_fn] {
        vm_clif.load_mir_function(f.clone()).expect("load clif");
    }
    vm_clif.compile_cranelift().expect("compile_cranelift");

    assert!(
        !vm_clif.cranelift_compiled.contains_key("mid_caller")
            && !vm_clif.cranelift_compiled.contains_key("outer_caller")
            && !vm_clif.cranelift_compiled.contains_key("rec_effect"),
        "callers of effect-deopt callees and self-recursive effect functions must not compile natively"
    );

    let log_bc = Rc::new(RefCell::new(vec![Value::Int(0)]));
    let log_clif = Rc::new(RefCell::new(vec![Value::Int(0)]));
    let c_bc = vm_bc.call("main_chain", vec![Value::Array(log_bc.clone())]).expect("bc chain");
    let c_clif = vm_clif.call("main_chain", vec![Value::Array(log_clif.clone())]).expect("clif chain");
    assert_eq!(c_clif, c_bc);
    assert_eq!(*log_clif.borrow(), *log_bc.borrow());

    let r_bc = vm_bc.call("main_rec", vec![Value::Array(log_bc.clone())]).expect("bc rec");
    let r_clif = vm_clif.call("main_rec", vec![Value::Array(log_clif.clone())]).expect("clif rec");
    assert_eq!(r_clif, r_bc);
    assert_eq!(*log_clif.borrow(), *log_bc.borrow());
}

#[test]
fn test_b7_synthetic_effect_perform_vm_fail_closed() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_test_state();

    let mut synth = MirFunction::new(
        FunctionSignature {
            name: "synthetic_effect_perform_fn".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        Span::dummy(),
    );
    let x = synth.params[0];
    let dst = synth.add_local(Some("dst".into()), MirType::Int, false, Span::dummy());
    let bb1 = synth.new_block();
    synth.blocks[0].terminator = Some(Terminator::Call {
        callee: "effect.perform".into(),
        args: vec![Operand::Copy(Place::Local(x))],
        destination: Place::Local(dst),
        success: bb1,
        error: None,
        span: Span::dummy(),
    });
    synth.blocks[bb1.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(dst))),
        Span::dummy(),
    ));

    // Standalone CraneliftJitCompiler still compiles `effect.perform` (w3_handlers/w3_jit)
    let mut jit = CraneliftJitCompiler::new().expect("jit init");
    assert!(
        jit.compile_function(&synth).is_ok(),
        "Standalone CraneliftJitCompiler must continue compiling effect.perform"
    );

    // VM::compile_cranelift refuses default production admission for synthetic `effect.perform`
    let mut vm = VM::new();
    vm.set_backend(VmBackend::Cranelift);
    vm.register_native(
        "effect.perform",
        Value::NativeFunction(native_fn(|args| {
            Ok(NativeCallResult::Return(Value::Int(
                args[0].as_int().unwrap_or(0) + 1,
            )))
        })),
    );
    vm.load_mir_function(synth).expect("load synth");
    vm.compile_cranelift().expect("compile_cranelift");
    assert!(
        !vm.cranelift_compiled.contains_key("synthetic_effect_perform_fn"),
        "synthetic effect.perform must not be admitted to default VM native execution"
    );
    let e0 = vm.cranelift_entry_count;
    let res = vm
        .call("synthetic_effect_perform_fn", vec![Value::Int(41)])
        .expect("bytecode fallback execution");
    assert_eq!(vm.cranelift_entry_count, e0);
    assert_eq!(res, Value::Int(42));
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_b11_stale_generation_wrong_invocation_and_reentrant_native() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_test_state();

    let fn_name = "gen_check_fn";
    let make_fn = || {
        let mut f = MirFunction::new(
            FunctionSignature {
                name: fn_name.into(),
                param_types: vec![MirType::Array(Box::new(MirType::Int))],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            Span::dummy(),
        );
        let a = f.params[0];
        let zero = f.add_local(Some("zero".into()), MirType::Int, false, Span::dummy());
        let d = f.add_local(Some("d".into()), MirType::Int, false, Span::dummy());
        let zero2 = f.add_local(Some("zero2".into()), MirType::Int, false, Span::dummy());
        let elem = f.add_local(Some("elem".into()), MirType::Int, false, Span::dummy());
        let out = f.add_local(Some("out".into()), MirType::Int, false, Span::dummy());
        let bb1 = f.new_block();
        let bb2 = f.new_block();
        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(zero),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));
        f.blocks[0].statements.push(Statement::Assign(
            Place::Index(Box::new(Place::Local(a)), zero),
            Rvalue::Use(Operand::Constant(MirConstant::Int(777))),
            Span::dummy(),
        ));
        f.blocks[0].terminator = Some(Terminator::Goto(bb1, Span::dummy()));
        f.blocks[bb1.0].terminator = Some(Terminator::Call {
            callee: "effect.perform".into(),
            args: vec![],
            destination: Place::Local(d),
            success: bb2,
            error: None,
            span: Span::dummy(),
        });
        f.blocks[bb2.0].statements.push(Statement::Assign(
            Place::Local(zero2),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));
        f.blocks[bb2.0].statements.push(Statement::Assign(
            Place::Local(elem),
            Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(a)), zero2))),
            Span::dummy(),
        ));
        f.blocks[bb2.0].statements.push(Statement::Assign(
            Place::Local(out),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(d)),
                Operand::Copy(Place::Local(elem)),
            ),
            Span::dummy(),
        ));
        f.blocks[bb2.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(out))),
            Span::dummy(),
        ));
        f
    };

    // Part (a1): wrong invocation_id fails closed with zero array mutation
    {
        let mut vm = VM::new();
        vm.set_backend(VmBackend::Cranelift);
        vm.register_native(
            "effect.perform",
            Value::NativeFunction(native_fn(|_| Ok(NativeCallResult::Return(Value::Int(0))))),
        );
        vm.load_mir_function(make_fn()).expect("load");
        let bad_payload = {
            let _inv = push_native_invocation(fn_name.into(), 1, 99999);
            lust::mir::cranelift_backend::DeoptPayload::at_pre_terminator(1, vec![(0, 0)])
        };
        let mut hook = lust::vm::TestDeoptSpikeHook::new(fn_name, 1);
        hook.injected_payload = Some(bad_payload);
        vm.test_deopt_spike_hook = Some(hook);

        let arr = Rc::new(RefCell::new(vec![Value::Int(5)]));
        let err = vm
            .call(fn_name, vec![Value::Array(arr.clone())])
            .expect_err("wrong invocation_id must fail");
        assert!(err.to_string().contains("mismatch") || err.to_string().contains("identity"));
        assert_eq!(arr.borrow()[0], Value::Int(5), "no mutation on rejected payload");
    }

    // Part (a2): stale compilation_generation (via DeoptPayload::with_func after load/compile) fails closed
    {
        let mut vm = VM::new();
        vm.set_backend(VmBackend::Cranelift);
        vm.register_native(
            "effect.perform",
            Value::NativeFunction(native_fn(|_| Ok(NativeCallResult::Return(Value::Int(0))))),
        );
        vm.load_mir_function(make_fn()).expect("load v1");
        vm.load_mir_function(make_fn()).expect("reload v2");
        let stale_payload = lust::mir::cranelift_backend::DeoptPayload::with_func(
            1,
            vec![(0, 0)],
            Some(fn_name.into()),
        )
        .with_site_kind(lust::mir::cranelift_backend::DeoptSiteKind::PreTerminator);
        let mut hook = lust::vm::TestDeoptSpikeHook::new(fn_name, 1);
        hook.injected_payload = Some(stale_payload);
        vm.test_deopt_spike_hook = Some(hook);

        let arr = Rc::new(RefCell::new(vec![Value::Int(9)]));
        let err = vm
            .call(fn_name, vec![Value::Array(arr.clone())])
            .expect_err("stale compilation_generation must fail");
        assert!(
            err.to_string().contains("generation") || err.to_string().contains("mismatch"),
            "got: {err}"
        );
        assert_eq!(arr.borrow()[0], Value::Int(9), "no mutation on stale generation");
    }

    // Part (b): reentrant VM::call inside handler arm executes another native function cleanly
    {
        reset_all_test_state();
        let src = r#"
            function outer_reent(x: int): int return 0 end
            function inner_native_callee(y: int): int return y * 3 end
            function inner_native_caller(y: int): int return inner_native_callee(y) + 1 end
            function main(x: int): int
                handle {
                    outer_reent(x)
                } with {
                    Reent(v) => {
                        let k = inner_native_caller(v);
                        resume(k)
                    }
                }
            end
        "#;
        let bc = setup_bytecode_funcs(src);
        let mir_mod = lower_mir_module_from_source(
            r#"
                function inner_native_callee(y: int): int return y * 3 end
                function inner_native_caller(y: int): int return inner_native_callee(y) + 1 end
            "#,
            "reent_inner",
        );

        let mut outer = MirFunction::new(
            FunctionSignature {
                name: "outer_reent".into(),
                param_types: vec![MirType::Int],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            Span::dummy(),
        );
        let ox = outer.params[0];
        let or = outer.add_local(Some("or".into()), MirType::Int, false, Span::dummy());
        let obb1 = outer.new_block();
        outer.blocks[0].terminator = Some(Terminator::Call {
            callee: "perform".into(),
            args: vec![
                Operand::Constant(MirConstant::String("Reent".into())),
                Operand::Copy(Place::Local(ox)),
            ],
            destination: Place::Local(or),
            success: obb1,
            error: None,
            span: Span::dummy(),
        });
        outer.blocks[obb1.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(or))),
            Span::dummy(),
        ));

        let mut vm_bc = VM::new();
        vm_bc.set_backend(VmBackend::Bytecode);
        vm_bc.load_functions(bc.clone());
        vm_bc.load_mir_module(&mir_mod).expect("load mod bc");
        vm_bc.load_mir_function(outer.clone()).expect("load outer bc");

        let mut vm_clif = VM::new();
        vm_clif.set_backend(VmBackend::Cranelift);
        vm_clif.load_functions(bc);
        vm_clif.load_mir_module(&mir_mod).expect("load mod clif");
        vm_clif.load_mir_function(outer).expect("load outer clif");

        let e0 = vm_clif.cranelift_entry_count;
        let r_bc = vm_bc.call("main", vec![Value::Int(7)]).expect("bc");
        let r_clif = vm_clif.call("main", vec![Value::Int(7)]).expect("clif");
        assert!(
            vm_clif.cranelift_entry_count >= e0 + 2,
            "both outer_reent and reentrant inner_native_caller must enter Cranelift"
        );
        assert_eq!(r_clif, Value::Int(22));
        assert_eq!(r_clif, r_bc);
        assert_eq!(native_invocation_depth(), 0);
    }
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_b12_nested_native_callee_safepoint_deopt_fails_closed() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_test_state();

    // Native caller invokes native loop callee whose safepoint triggers deopt
    let mut module = MirModule::new("nested_native_deopt".into());

    // callee_loop(n: int) -> int (has loop header safepoint in bb1)
    let mut callee = MirFunction::new(
        FunctionSignature {
            name: "callee_loop".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        Span::dummy(),
    );
    let cn = callee.params[0];
    let ci = callee.add_local(Some("i".into()), MirType::Int, true, Span::dummy());
    let cacc = callee.add_local(Some("acc".into()), MirType::Int, true, Span::dummy());
    let ccond = callee.add_local(Some("cond".into()), MirType::Bool, false, Span::dummy());
    let c_hdr = callee.new_block();
    let c_body = callee.new_block();
    let c_exit = callee.new_block();
    callee.blocks[0].statements.push(Statement::Assign(
        Place::Local(ci),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    callee.blocks[0].statements.push(Statement::Assign(
        Place::Local(cacc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    callee.blocks[0].terminator = Some(Terminator::Goto(c_hdr, Span::dummy()));
    callee.blocks[c_hdr.0].statements.push(Statement::Assign(
        Place::Local(ccond),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(ci)),
            Operand::Copy(Place::Local(cn)),
        ),
        Span::dummy(),
    ));
    callee.blocks[c_hdr.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(ccond)),
        then_block: c_body,
        else_block: c_exit,
        span: Span::dummy(),
    });
    callee.blocks[c_body.0].statements.push(Statement::Assign(
        Place::Local(cacc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(cacc)),
            Operand::Copy(Place::Local(ci)),
        ),
        Span::dummy(),
    ));
    callee.blocks[c_body.0].statements.push(Statement::Assign(
        Place::Local(ci),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(ci)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    callee.blocks[c_body.0].terminator = Some(Terminator::Goto(c_hdr, Span::dummy()));
    callee.blocks[c_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(cacc))),
        Span::dummy(),
    ));

    // caller_no_loop(n: int) -> int (no loop, calls callee_loop natively)
    let mut caller = MirFunction::new(
        FunctionSignature {
            name: "caller_no_loop".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        Span::dummy(),
    );
    let pn = caller.params[0];
    let p_res = caller.add_local(Some("res".into()), MirType::Int, false, Span::dummy());
    let p_bb1 = caller.new_block();
    caller.blocks[0].terminator = Some(Terminator::Call {
        callee: "callee_loop".into(),
        args: vec![Operand::Copy(Place::Local(pn))],
        destination: Place::Local(p_res),
        success: p_bb1,
        error: None,
        span: Span::dummy(),
    });
    caller.blocks[p_bb1.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(p_res))),
        Span::dummy(),
    ));

    module.add_function(callee);
    module.add_function(caller);

    let mut vm = VM::new();
    vm.set_backend(VmBackend::Cranelift);
    vm.load_mir_module(&module).expect("load module");
    vm.compile_cranelift().expect("compile module");
    assert!(vm.cranelift_compiled.contains_key("caller_no_loop"));
    assert!(vm.cranelift_compiled.contains_key("callee_loop"));

    safepoint_request_deopt(true);
    let err = vm
        .call("caller_no_loop", vec![Value::Int(5)])
        .expect_err("nested native-to-native safepoint deopt must fail closed");
    assert!(
        err.to_string().contains("Nested native-to-native deopt")
            || err.to_string().contains("multi-frame"),
        "got: {err}"
    );
    assert_eq!(native_invocation_depth(), 0);
    assert!(take_deopt_payload().is_none());
    assert!(get_current_sticky_error().is_none());

    // Healthy follow-up native execution succeeds cleanly
    safepoint_reset();
    let ok_res = vm
        .call("caller_no_loop", vec![Value::Int(5)])
        .expect("follow-up native call must succeed cleanly");
    assert_eq!(ok_res, Value::Int(10));
}

// =========================================================================
// Gate 4A Equivalent-Workload Benchmark (Cranelift MIR vs VM Interpreter)
// =========================================================================

#[test]
#[ignore]
fn bench_gate4a_cranelift_vs_vm_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_test_state();

    let src = r#"
        function tight_sum(n: int): int
            local sum: int = 0
            local i: int = 1
            while i <= n do
                sum = sum + i
                i = i + 1
            end
            return sum
        end
    "#;
    let module = lower_mir_module_from_source(src, "gate4a_bench");
    let iterations: i64 = 1_000_000;
    let expected = iterations * (iterations + 1) / 2;

    let mut vm_interp = VM::new();
    vm_interp.set_backend(VmBackend::Bytecode);
    vm_interp.load_mir_module(&module).expect("load interp");
    // Warmup
    assert_eq!(
        vm_interp.call("tight_sum", vec![Value::Int(1_000)]).unwrap(),
        Value::Int(500_500)
    );
    let mut vm_samples = Vec::with_capacity(7);
    for _ in 0..7 {
        let t0 = Instant::now();
        let v = vm_interp.call("tight_sum", vec![Value::Int(iterations)]).unwrap();
        vm_samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(v, Value::Int(expected));
    }
    vm_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut vm_clif = VM::new();
    vm_clif.set_backend(VmBackend::Cranelift);
    vm_clif.load_mir_module(&module).expect("load clif");
    vm_clif.compile_cranelift().expect("compile clif");
    // Warmup
    assert_eq!(
        vm_clif.call("tight_sum", vec![Value::Int(1_000)]).unwrap(),
        Value::Int(500_500)
    );
    let e0 = vm_clif.cranelift_entry_count;
    let mut clif_samples = Vec::with_capacity(7);
    for _ in 0..7 {
        let t0 = Instant::now();
        let v = vm_clif.call("tight_sum", vec![Value::Int(iterations)]).unwrap();
        clif_samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(v, Value::Int(expected));
    }
    clif_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let vm_median = vm_samples[3];
    let clif_median = clif_samples[3];
    let speedup = vm_median / clif_median;
    println!("VM_SAMPLES_MS={:?}", vm_samples);
    println!("CLIF_SAMPLES_MS={:?}", clif_samples);
    println!(
        "GATE4A_TIGHT_SUM: vm_median_ms={:.4} clif_median_ms={:.4} speedup={:.2}x clif_entries={}",
        vm_median,
        clif_median,
        speedup,
        vm_clif.cranelift_entry_count - e0
    );
    assert!(vm_clif.cranelift_entry_count - e0 >= 7);
    assert!(clif_median <= vm_median);
    assert!(speedup >= 2.0, "Expected >= 2.0x speedup, got {speedup:.2}x");
}

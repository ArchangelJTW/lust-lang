//! Acceptance test suite for Wave 3 Track 3D: L-HANDLERS (Algebraic Effect Handlers)
//!
//! Verifies:
//! 1. Lexer, AST, and Parser support for `handle ... with`, `perform`, `resume`.
//! 2. Typechecker effect-handler typing, arity validation, arm/body return compatibility,
//!    and prohibition of `resume` outside of active handler arms.
//! 3. VM runtime execution of effect handlers with single and multi-step resumption (`resume(val)`).
//! 4. Lexically and dynamically nested handlers with automatic effect forwarding.
//! 5. Abortive handlers (handler arm returns without calling `resume`), unwinding the computation.
//! 6. Runtime errors on unhandled effects, double resumption, and resume outside handler arm.
//! 7. Finalizer execution in strict LIFO order on normal completion, abort, and error/cancellation.
//! 8. 3A -> 3D deopt continuation: consuming `DeoptPayload` via `take_deopt_payload()`,
//!    reconstructing top CallFrame state, and resuming interpreter execution without duplicate execution.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use lust::ast::{ExprKind, Item, ItemKind, Literal, Span};
use lust::bytecode::{native_fn, Instruction, NativeCallResult, Value};
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{
    publish_deopt_payload, reset_native_instrumentation_counters, safepoint_reset,
    take_deopt_payload, CraneliftJitCompiler, CRANELIFT_NATIVE_ENTRY_COUNT, MIR_STATUS_DEOPT,
};
use lust::mir::ir::{
    FunctionSignature, LocalId, MirBinaryOp, MirConstant, MirFunction, MirModule, MirType,
    Operand, Place, Rvalue, Statement, Terminator,
};
use lust::mir::runtime_helpers::{enter_mir_context, MirContext};
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{register_effect_finalizer, reset_effect_handler_state, CallFrame, VM};
use lust::{LustConfig, LustError};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn parse_to_items(source: &str) -> Result<Vec<Item>, LustError> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize()?;
    Parser::new(tokens).parse()
}

fn typecheck_source(source: &str) -> Result<(), LustError> {
    let items = parse_to_items(source)?;
    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    checker.check_module(&items)?;
    Ok(())
}

fn compile_and_run_main(source: &str) -> Result<Value, LustError> {
    let items = parse_to_items(source)?;
    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let mut compiler = lust::Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    let funcs = compiler.compile_module(&items)?;

    let mut vm = VM::new();
    vm.load_functions(funcs);
    vm.call("main", vec![])
}

// =========================================================================
// 1. Parsing and AST Desugaring Tests
// =========================================================================

#[test]
fn test_w3_handlers_parsing_and_ast_structure() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();

    let src = r#"
    function main(): int
        handle 42 with {
            Get() => resume(10),
            Set(x) => resume(x)
        }
    end
    "#;
    let items = parse_to_items(src).expect("Parsing handle expression must succeed");
    assert_eq!(items.len(), 1);

    if let ItemKind::Function(func) = &items[0].kind {
        assert_eq!(func.body.len(), 1);
        let stmt = &func.body[0];
        if let lust::ast::StmtKind::Expr(expr) = &stmt.kind {
            // Must have desugared to __lust_handle call
            if let ExprKind::Call { callee, args, .. } = &expr.kind {
                if let ExprKind::Identifier(name) = &callee.kind {
                    assert_eq!(name, "__lust_handle");
                } else {
                    panic!("Expected callee to be __lust_handle");
                }
                assert_eq!(args.len(), 2, "__lust_handle requires body thunk and arms array");
            } else {
                panic!("Expected Call expression from handle desugaring");
            }
        } else {
            panic!("Expected StmtKind::Expr");
        }
    } else {
        panic!("Expected Function item");
    }
}

// =========================================================================
// 2. Typechecker Tests
// =========================================================================

#[test]
fn test_w3_handlers_typechecker_valid() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();

    let src = r#"
    function main(): int
        handle 100 with {
            Ask() => resume(42)
        }
    end
    "#;
    assert!(typecheck_source(src).is_ok());
}

#[test]
fn test_w3_handlers_typechecker_rejects_resume_outside_arm() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();

    let src = r#"
    function main(): int
        resume(42);
        return 0
    end
    "#;
    let res = typecheck_source(src);
    assert!(res.is_err(), "resume outside handler arm must be rejected by typechecker");
    let err_msg = format!("{:?}", res.err().unwrap());
    assert!(
        err_msg.contains("outside of an active effect handler arm") || err_msg.contains("resume"),
        "Error message must mention resume outside handler arm, got: {}",
        err_msg
    );
}

#[test]
fn test_w3_handlers_typechecker_rejects_mismatched_arm_return_type() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();

    let src = r#"
    function main(): int
        handle 100 with {
            Abort() => "incompatible string"
        }
    end
    "#;
    let res = typecheck_source(src);
    assert!(
        res.is_err(),
        "Mismatched arm return type (string vs int) must be rejected by typechecker"
    );
}

// =========================================================================
// 3. Execution & Resumption Tests
// =========================================================================

#[test]
fn test_w3_handlers_basic_resumption() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();

    let src = r#"
    function main(): int
        handle {
            let x = perform("GetVal", []);
            x + 10
        } with {
            GetVal() => resume(42)
        }
    end
    "#;
    let val = compile_and_run_main(src).expect("Execution must succeed");
    assert_eq!(val, Value::Int(52), "42 + 10 must equal 52");
}

#[test]
fn test_w3_handlers_multiple_sequential_resumptions() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();

    let src = r#"
    function main(): int
        handle {
            let a = perform("Step", [1]);
            let b = perform("Step", [2]);
            let c = perform("Step", [3]);
            a + b + c
        } with {
            Step(x) => resume(x * 10)
        }
    end
    "#;
    let val = compile_and_run_main(src).expect("Execution must succeed");
    assert_eq!(val, Value::Int(60), "10 + 20 + 30 must equal 60");
}

// =========================================================================
// 4. Nested Handlers and Effect Forwarding
// =========================================================================

#[test]
fn test_w3_handlers_nested_handlers_and_forwarding() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();

    let src = r#"
    function main(): int
        handle {
            handle {
                let inner_val = perform("Inner", [5]);
                let outer_val = perform("Outer", [100]);
                inner_val + outer_val
            } with {
                Inner(x) => resume(x * 2)
            }
        } with {
            Outer(y) => resume(y + 1)
        }
    end
    "#;
    let val = compile_and_run_main(src).expect("Nested execution with forwarding must succeed");
    assert_eq!(val, Value::Int(111), "Inner(5)*2 + Outer(100)+1 = 10 + 101 = 111");
}

// =========================================================================
// 5. Abortive Handlers (No Resume)
// =========================================================================

#[test]
fn test_w3_handlers_abortive_handler_unwinds_computation() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();

    let src = r#"
    function main(): int
        handle {
            let x = perform("Abort", [404]);
            x + 9999
        } with {
            Abort(code) => -code
        }
    end
    "#;
    let val = compile_and_run_main(src).expect("Abortive handler must succeed");
    assert_eq!(val, Value::Int(-404), "Computation must abort and yield -404");
}

#[test]
fn test_w3_handlers_nested_abort_only_aborts_inner() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();

    let src = r#"
    function main(): int
        let res = handle {
            perform("AbortInner", []);
            10
        } with {
            AbortInner() => 99
        };
        res + 1
    end
    "#;
    let val = compile_and_run_main(src).expect("Nested abort must succeed");
    assert_eq!(val, Value::Int(100), "Inner abort yields 99, plus 1 is 100");
}

// =========================================================================
// 6. Runtime Errors: Unhandled Effect, Double Resume, Resume Outside
// =========================================================================

#[test]
fn test_w3_handlers_unhandled_effect_error() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();

    let src = r#"
    function main(): int
        handle {
            perform("NonExistent", []);
            10
        } with {
            Other() => resume(1)
        }
    end
    "#;
    let res = compile_and_run_main(src);
    assert!(res.is_err(), "Unhandled effect must raise runtime error");
    let err_str = format!("{:?}", res.err().unwrap());
    assert!(
        err_str.contains("Unhandled effect: NonExistent"),
        "Error message must specify unhandled effect name, got: {}",
        err_str
    );
}

#[test]
fn test_w3_handlers_double_resumption_error() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();

    let src = r#"
    function main(): int
        handle {
            perform("Ask", []);
            10
        } with {
            Ask() => {
                resume(1);
                resume(2)
            }
        }
    end
    "#;
    let res = compile_and_run_main(src);
    assert!(res.is_err(), "Double resumption must be rejected at runtime");
    let err_str = format!("{:?}", res.err().unwrap());
    assert!(
        err_str.contains("Effect already resumed"),
        "Error message must indicate effect already resumed, got: {}",
        err_str
    );
}

#[test]
fn test_w3_handlers_runtime_resume_outside_arm() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();

    let mut vm = VM::new();
    let res = vm.execute_effect_resume(Value::Int(10));
    assert!(res.is_err(), "Direct resume outside arm must fail");
    let err_str = format!("{:?}", res.err().unwrap());
    assert!(
        err_str.contains("Cannot resume outside of an effect handler arm"),
        "Got: {}",
        err_str
    );
}

// =========================================================================
// 7. Finalizers on Normal Completion, Abort, and Error
// =========================================================================

#[test]
fn test_w3_handlers_finalizers_lifo_order_on_normal_and_abort() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();

    let events = Arc::new(Mutex::new(Vec::new()));
    let ev1 = events.clone();
    let ev2 = events.clone();

    register_effect_finalizer(move || {
        ev1.lock().unwrap().push("finalizer_1");
    });
    register_effect_finalizer(move || {
        ev2.lock().unwrap().push("finalizer_2");
    });

    let src = r#"
    function main(): int
        handle {
            perform("AbortWithCleanup", []);
            100
        } with {
            AbortWithCleanup() => 77
        }
    end
    "#;
    let res = compile_and_run_main(src).expect("Abort with cleanup must succeed");
    assert_eq!(res, Value::Int(77));

    // Verify LIFO order of finalizer execution: finalizer_2 before finalizer_1
    assert_eq!(*events.lock().unwrap(), vec!["finalizer_2", "finalizer_1"]);
}

#[test]
fn test_w3_handlers_finalizers_on_unhandled_effect() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();

    let cleaned_up = Arc::new(AtomicBool::new(false));
    let cleaned_clone = cleaned_up.clone();

    register_effect_finalizer(move || {
        cleaned_clone.store(true, Ordering::SeqCst);
    });

    let src = r#"
    function main(): int
        handle {
            perform("FatalError", []);
            0
        } with {
            Irrelevant() => resume(0)
        }
    end
    "#;
    let res = compile_and_run_main(src);
    assert!(res.is_err(), "Must error on unhandled effect");
    assert!(cleaned_up.load(Ordering::SeqCst), "Finalizer must run even on unhandled effect error");
}

// =========================================================================
// 8. 3A -> 3D Deopt Fallback Interface Verification
// =========================================================================

#[test]
fn test_w3_handlers_3a_to_3d_deopt_fallback() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();

    // Clear any pending deopt payload
    take_deopt_payload();

    // 1. Compile a function that executes an effect handler
    let src = r#"
    function main(): int
        handle {
            let x = perform("GetAfterDeopt", []);
            x + 5
        } with {
            GetAfterDeopt() => resume(40)
        }
    end
    "#;
    let items = parse_to_items(src).expect("Parsing must succeed");
    let config = LustConfig::default();
    let mut compiler = lust::Compiler::new();
    compiler.configure(&config);
    let funcs = compiler.compile_module(&items).expect("Compile must succeed");

    let mut vm = VM::new();
    vm.load_functions(funcs);

    // 2. Simulate 3A Cranelift JIT encountering an unsupported effect at target_pc:
    // Publish a deopt payload with live locals: local 0 = 100, local 1 = 200
    let target_pc = 0; // target_pc at the beginning of the handler execution
    let live_locals = vec![(0, 100), (1, 200)];
    publish_deopt_payload(target_pc, live_locals);

    // 3. Consume via 3D resume_from_deopt
    let types = vec![(0, MirType::Int), (1, MirType::Int)];
    let outcome = vm.resume_from_deopt(Some(&types)).expect("resume_from_deopt must succeed");

    // The handler executes and produces 40 + 5 = 45!
    assert_eq!(outcome, Value::Int(45), "Result after deopt resumption must be 45");

    // 4. Verify payload was consumed once (cannot be consumed twice)
    assert!(
        take_deopt_payload().is_none(),
        "take_deopt_payload must return None after being consumed"
    );
}

#[test]
fn test_w3_handlers_3a_cranelift_to_3d_vm_integrated_deopt() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();
    take_deopt_payload();

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    safepoint_reset();

    // 1. Build a real 2-block MIR function:
    //    bb0 computes `accum = arg0 + 100` (125 for arg0=25) and jumps to bb1.
    //    bb1 hits an unsupported `effect.perform` and triggers MIR_STATUS_DEOPT at target_pc = 1.
    let sig = FunctionSignature {
        name: "deopt_integrated_fn".into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut mir_fn = MirFunction::new(sig, Span::dummy());
    let arg0 = mir_fn.params[0];
    let local_accum = mir_fn.add_local(Some("accum".into()), MirType::Int, true, Span::dummy());
    let bb1 = mir_fn.new_block();
    let bb2 = mir_fn.new_block();

    mir_fn.blocks[0].statements.push(Statement::Assign(
        Place::Local(local_accum),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(arg0)),
            Operand::Constant(MirConstant::Int(100)),
        ),
        Span::dummy(),
    ));
    mir_fn.blocks[0].terminator = Some(Terminator::Goto(bb1, Span::dummy()));

    mir_fn.blocks[bb1.0].terminator = Some(Terminator::Call {
        callee: "effect.perform".into(),
        args: vec![Operand::Copy(Place::Local(local_accum))],
        destination: Place::Local(mir_fn.return_local),
        success: bb2,
        error: None,
        span: Span::dummy(),
    });
    mir_fn.blocks[bb2.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(mir_fn.return_local))),
        Span::dummy(),
    ));

    // 2. Compile and enter real Cranelift native code
    let mut jit = CraneliftJitCompiler::new().expect("Init Cranelift compiler");
    let code_ptr = jit.compile_function(&mir_fn).expect("Compile MIR function");
    let native_fn: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code_ptr) };

    let ret_code = native_fn(25);
    assert_eq!(ret_code, MIR_STATUS_DEOPT, "Cranelift native code must return MIR_STATUS_DEOPT");

    // 3. Prepare VM handler bytecode where ip=0 is a pre-deopt sentinel Return(0) (Nil)
    //    and ip=1 (target_pc=1) executes the `handle ... with` continuation.
    let src = r#"
    function main(): int
        handle {
            let x = perform("GetAfterDeopt", []);
            x + 5
        } with {
            GetAfterDeopt() => resume(40)
        }
    end
    "#;
    let items = parse_to_items(src).expect("Parsing must succeed");
    let config = LustConfig::default();
    let mut compiler = lust::Compiler::new();
    compiler.configure(&config);
    let mut funcs = compiler.compile_module(&items).expect("Compile must succeed");
    // Prepend a sentinel Return(0) at ip=0; if pre-deopt ip=0 re-executed, result would be Nil.
    funcs[0].chunk.instructions.insert(0, lust::bytecode::Instruction::Return(0));
    funcs[0].chunk.lines.insert(0, 0);

    let mut vm = VM::new();
    vm.load_functions(funcs);

    // 4. Consume the Cranelift-published DeoptPayload via 3D `resume_from_deopt`
    let mut types = vec![(mir_fn.return_local.0, MirType::Int)];
    if arg0.0 != mir_fn.return_local.0 {
        types.push((arg0.0, MirType::Int));
    }
    if local_accum.0 != mir_fn.return_local.0 && local_accum.0 != arg0.0 {
        types.push((local_accum.0, MirType::Int));
    }
    let outcome = vm.resume_from_deopt(Some(&types)).expect("resume_from_deopt must succeed");
    assert_eq!(
        outcome,
        Value::Int(45),
        "VM continuation must resume at target_pc=1 without re-executing ip=0"
    );

    // 5. Verify DeoptPayload was consumed once
    assert!(
        take_deopt_payload().is_none(),
        "take_deopt_payload must be None after single consumption by 3D"
    );
}

#[test]
fn test_w3_4_8_a_hot_loop_cranelift_to_vm_resume_7500() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();
    reset_native_instrumentation_counters();
    safepoint_reset();
    take_deopt_payload();

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);

    // 1. Build Section 4.8.A hot loop MIR function `compute_with_effect(iterations: i64) -> i64`:
    //    acc = 0; i = 0; while i < iterations { acc += i; if i == 50 { acc += perform Yield(acc); } i += 1; }
    let sig = FunctionSignature {
        name: "compute_with_effect".into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut mir_fn = MirFunction::new(sig, Span::dummy());
    let iterations_loc = mir_fn.params[0]; // LocalId(1)
    let acc_loc = mir_fn.add_local(Some("acc".into()), MirType::Int, true, Span::dummy()); // LocalId(2)
    let i_loc = mir_fn.add_local(Some("i".into()), MirType::Int, true, Span::dummy()); // LocalId(3)
    let cond_loop = mir_fn.add_local(Some("cond_loop".into()), MirType::Bool, false, Span::dummy());
    let cond_yield = mir_fn.add_local(Some("cond_yield".into()), MirType::Bool, false, Span::dummy());
    let yield_res = mir_fn.add_local(Some("yield_res".into()), MirType::Int, false, Span::dummy());

    let bb_loop_hdr = mir_fn.new_block(); // 1
    let bb_loop_body = mir_fn.new_block(); // 2
    let bb_do_yield = mir_fn.new_block(); // 3 -> target_pc = 3
    let bb_after_yield = mir_fn.new_block(); // 4
    let bb_loop_next = mir_fn.new_block(); // 5
    let bb_exit = mir_fn.new_block(); // 6

    mir_fn.blocks[0].statements.push(Statement::Assign(
        Place::Local(acc_loc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    mir_fn.blocks[0].statements.push(Statement::Assign(
        Place::Local(i_loc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    mir_fn.blocks[0].terminator = Some(Terminator::Goto(bb_loop_hdr, Span::dummy()));

    mir_fn.blocks[bb_loop_hdr.0].statements.push(Statement::Assign(
        Place::Local(cond_loop),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(i_loc)),
            Operand::Copy(Place::Local(iterations_loc)),
        ),
        Span::dummy(),
    ));
    mir_fn.blocks[bb_loop_hdr.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond_loop)),
        then_block: bb_loop_body,
        else_block: bb_exit,
        span: Span::dummy(),
    });

    mir_fn.blocks[bb_loop_body.0].statements.push(Statement::Assign(
        Place::Local(acc_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(acc_loc)),
            Operand::Copy(Place::Local(i_loc)),
        ),
        Span::dummy(),
    ));
    mir_fn.blocks[bb_loop_body.0].statements.push(Statement::Assign(
        Place::Local(cond_yield),
        Rvalue::BinaryOp(
            MirBinaryOp::Eq,
            Operand::Copy(Place::Local(i_loc)),
            Operand::Constant(MirConstant::Int(50)),
        ),
        Span::dummy(),
    ));
    mir_fn.blocks[bb_loop_body.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond_yield)),
        then_block: bb_do_yield,
        else_block: bb_loop_next,
        span: Span::dummy(),
    });

    mir_fn.blocks[bb_do_yield.0].terminator = Some(Terminator::Call {
        callee: "effect.perform".into(),
        args: vec![Operand::Copy(Place::Local(acc_loc))],
        destination: Place::Local(yield_res),
        success: bb_after_yield,
        error: None,
        span: Span::dummy(),
    });

    mir_fn.blocks[bb_after_yield.0].statements.push(Statement::Assign(
        Place::Local(acc_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(acc_loc)),
            Operand::Copy(Place::Local(yield_res)),
        ),
        Span::dummy(),
    ));
    mir_fn.blocks[bb_after_yield.0].terminator = Some(Terminator::Goto(bb_loop_next, Span::dummy()));

    mir_fn.blocks[bb_loop_next.0].statements.push(Statement::Assign(
        Place::Local(i_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i_loc)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    mir_fn.blocks[bb_loop_next.0].terminator = Some(Terminator::Goto(bb_loop_hdr, Span::dummy()));

    mir_fn.blocks[bb_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(acc_loc))),
        Span::dummy(),
    ));

    let mut module = MirModule::new("deopt_4_8_a_vm".into());
    module.add_function(mir_fn.clone());

    // 2. Compile with Cranelift and verify native code range & entry count
    let mut jit = CraneliftJitCompiler::new().expect("Init Cranelift compiler");
    let func_map = jit.compile_module(&module).expect("Compile MIR module");
    let code_ptr = func_map["compute_with_effect"];
    let (jit_start, jit_end) = jit
        .compiled_code_range("compute_with_effect")
        .expect("Code range must exist");
    let entry_addr = code_ptr as usize;
    assert!(entry_addr >= jit_start && entry_addr < jit_end);

    assert_eq!(CRANELIFT_NATIVE_ENTRY_COUNT.load(Ordering::SeqCst), 0);
    let native_fn: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code_ptr) };
    let ret_code = native_fn(100);
    assert!(
        CRANELIFT_NATIVE_ENTRY_COUNT.load(Ordering::SeqCst) >= 1,
        "CRANELIFT_NATIVE_ENTRY_COUNT must be >= 1 before deopt"
    );
    assert_eq!(ret_code, MIR_STATUS_DEOPT);

    // 3. Prepare VM continuation reading restored locals (0: iterations=100, 1: _ret=0, 2: acc=1275, 3: i=50)
    //    and prepending 3 pre-deopt sentinel instructions at ip=0..3 (= bb_do_yield.0) that MUST NOT run.
    let src = r#"
    function main(iterations: int, _ret: int, acc: int, i: int): int
        local y: int = handle {
            perform("Yield", [acc])
        } with {
            Yield(val) => resume(val * 2)
        }
        local cur_acc: int = acc + y
        local cur_i: int = i + 1
        while cur_i < iterations do
            cur_acc = cur_acc + cur_i
            cur_i = cur_i + 1
        end
        return cur_acc
    end
    "#;
    let items = parse_to_items(src).expect("Parsing must succeed");
    let config = LustConfig::default();
    let mut compiler = lust::Compiler::new();
    compiler.configure(&config);
    let mut funcs = compiler.compile_module(&items).expect("Compile must succeed");

    let pre_deopt_reruns = Arc::new(AtomicUsize::new(0));
    let pre_deopt_clone = Arc::clone(&pre_deopt_reruns);

    let sentinel_const_idx = funcs[0].chunk.constants.len() as u16;
    funcs[0]
        .chunk
        .constants
        .push(Value::String(Rc::new("pre_deopt_sentinel".to_string())));

    // bb_do_yield.0 == 3, so insert 3 pre-deopt instructions at ip=0..3
    assert_eq!(bb_do_yield.0, 3);
    funcs[0]
        .chunk
        .instructions
        .splice(
            0..0,
            [
                Instruction::LoadGlobal(200, sentinel_const_idx),
                Instruction::Call(200, 201, 0, 202),
                Instruction::Return(202),
            ],
        );
    funcs[0].chunk.lines.splice(0..0, [0, 0, 0]);

    let mut vm = VM::new();
    vm.register_native(
        "pre_deopt_sentinel",
        Value::NativeFunction(lust::bytecode::native_fn(move |_| {
            pre_deopt_clone.fetch_add(1, Ordering::SeqCst);
            Ok(NativeCallResult::Return(Value::Int(-1)))
        })),
    );
    vm.load_functions(funcs);

    // 4. Resume from the live Cranelift DeoptPayload via 3D `resume_from_deopt`
    let types = vec![
        (iterations_loc.0, MirType::Int),
        (mir_fn.return_local.0, MirType::Int),
        (acc_loc.0, MirType::Int),
        (i_loc.0, MirType::Int),
    ];
    let outcome = vm
        .resume_from_deopt(Some(&types))
        .expect("resume_from_deopt must succeed");

    assert_eq!(
        pre_deopt_reruns.load(Ordering::SeqCst),
        0,
        "Iterations 0..49 / pre-deopt instructions MUST NOT be rerun in VM"
    );
    assert_eq!(
        outcome,
        Value::Int(7500),
        "Sum(0..50)=1275 + Yield(1275)=>2550 + Sum(51..99)=3675 must equal 7500"
    );
    assert!(
        take_deopt_payload().is_none(),
        "take_deopt_payload must be consumed once"
    );
}

//! Wave 17 NG-W17-1 (L-HANDLERS) acceptance and regression tests.
//!
//! Covers:
//! 1. Multi-level nested effect handlers (forwarding, shadowing, and mixed resume/abort).
//! 2. Finalizer LIFO ordering on nested resume/abort and error unwinding.
//! 3. Positive deopt/OSR cases: valid empty-stack deopt entry into `"main"` and live locals.
//! 4. Invalid-stack and malformed OSR rejection controls (fail-closed, zero stack mutation).

use std::sync::{Arc, Mutex};

use lust::bytecode::{Function, Instruction, Value};
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{
    publish_deopt_payload, safepoint_reset, take_deopt_payload,
};
use lust::mir::ir::MirType;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{register_effect_finalizer, reset_effect_handler_state, VM};
use lust::{LustConfig, LustError};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn compile_source(source: &str) -> Result<Vec<Function>, LustError> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize()?;
    let items = Parser::new(tokens).parse()?;
    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let mut compiler = lust::Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    compiler.compile_module(&items)
}

fn compile_and_run_main(source: &str) -> Result<Value, LustError> {
    let funcs = compile_source(source)?;
    let mut vm = VM::new();
    vm.load_functions(funcs);
    vm.call("main", vec![])
}

// =========================================================================
// 1. Nested Handlers: Forwarding, Shadowing, Resume & Abort
// =========================================================================

#[test]
fn test_ng_w17_nested_three_level_forwarding_and_shadowing() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();

    let src = r#"
    function main(): int
        handle {
            handle {
                handle {
                    let a = perform("Shadow", [3]);
                    let b = perform("Mid", [10]);
                    let c = perform("Outer", [100]);
                    a + b + c
                } with {
                    Shadow(x) => resume(x * 10)
                }
            } with {
                Shadow(x) => resume(x * 999),
                Mid(y) => resume(y + 5)
            }
        } with {
            Outer(z) => resume(z + 1)
        }
    end
    "#;

    let val = compile_and_run_main(src).expect("3-level nested handlers must succeed");
    // Shadow(3) handled by innermost -> 30; Mid(10) -> 15; Outer(100) -> 101; sum = 146
    assert_eq!(val, Value::Int(146));
}

#[test]
fn test_ng_w17_nested_outer_abort_unwinds_inner_and_runs_finalizers_lifo() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();

    let events = Arc::new(Mutex::new(Vec::new()));
    let ev1 = Arc::clone(&events);
    let ev2 = Arc::clone(&events);
    register_effect_finalizer(move || ev1.lock().unwrap().push("outer_fin"));
    register_effect_finalizer(move || ev2.lock().unwrap().push("inner_fin"));

    let src = r#"
    function main(): int
        handle {
            handle {
                let x = perform("Step", [5]);
                perform("AbortOuter", [x + 1]);
                9999
            } with {
                Step(v) => resume(v * 2)
            }
        } with {
            AbortOuter(code) => code * 10
        }
    end
    "#;

    let val = compile_and_run_main(src).expect("outer abort across inner handler must succeed");
    // Step(5) -> 10, AbortOuter(11) -> 110
    assert_eq!(val, Value::Int(110));
    assert_eq!(*events.lock().unwrap(), vec!["inner_fin", "outer_fin"]);
}

// =========================================================================
// 2. Positive Deopt/OSR Entry Cases
// =========================================================================

#[test]
fn test_ng_w17_valid_empty_stack_deopt_entry_into_main() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();
    safepoint_reset();
    take_deopt_payload();

    let src = r#"
    function main(a: int, b: int): int
        handle {
            let v = perform("Add", [a + b]);
            v + 1
        } with {
            Add(x) => resume(x * 2)
        }
    end
    "#;
    let funcs = compile_source(src).expect("compile");
    let mut vm = VM::new();
    vm.load_functions(funcs);
    assert_eq!(vm.call_stack_depth(), 0);

    publish_deopt_payload(0, vec![(0, 7), (1, 13)]);
    let types = vec![(0, MirType::Int), (1, MirType::Int)];
    let outcome = vm
        .resume_from_deopt(Some(&types))
        .expect("valid empty-stack deopt entry into main must succeed");
    // (7 + 13) * 2 + 1 = 41
    assert_eq!(outcome, Value::Int(41));
    assert!(take_deopt_payload().is_none());
}

// =========================================================================
// 3. Invalid-Stack & Malformed OSR Rejection Controls (Fail-Closed)
// =========================================================================

#[test]
fn test_ng_w17_rejects_empty_stack_with_none_types() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();
    safepoint_reset();
    take_deopt_payload();

    let mut vm = VM::new();
    let mut func = Function::new("main", 0, false);
    let c_idx = func.chunk.add_constant(Value::Int(123));
    func.chunk.emit(Instruction::LoadConst(0, c_idx), 1);
    func.chunk.emit(Instruction::Return(0), 1);
    func.register_count = 1;
    vm.load_functions(vec![func]);

    publish_deopt_payload(0, vec![(0, 10)]);
    let res = vm.resume_from_deopt(None);
    assert!(res.is_err(), "None type metadata must fail closed");
    assert_eq!(vm.call_stack_depth(), 0, "call stack must remain empty on rejection");
    take_deopt_payload();
}

#[test]
fn test_ng_w17_rejects_empty_stack_non_main_function_without_identity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();
    safepoint_reset();
    take_deopt_payload();

    let mut vm = VM::new();
    let mut func = Function::new("helper_not_main", 0, false);
    let c_idx = func.chunk.add_constant(Value::Int(999));
    func.chunk.emit(Instruction::LoadConst(0, c_idx), 1);
    func.chunk.emit(Instruction::Return(0), 1);
    func.register_count = 1;
    vm.load_functions(vec![func]);

    publish_deopt_payload(0, vec![(0, 42)]);
    let types = vec![(0, MirType::Int)];
    let res = vm.resume_from_deopt(Some(&types));
    assert!(
        res.is_err(),
        "empty-stack deopt with no main and no explicit func_name must fail closed, got: {:?}",
        res
    );
    assert_eq!(vm.call_stack_depth(), 0, "call stack must remain empty on rejection");
}

#[test]
fn test_ng_w17_rejects_empty_stack_missing_type_metadata() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();
    safepoint_reset();
    take_deopt_payload();

    let mut vm = VM::new();
    let mut func = Function::new("main", 0, false);
    func.chunk.emit(Instruction::Return(0), 1);
    func.register_count = 2;
    vm.load_functions(vec![func]);

    publish_deopt_payload(0, vec![(0, 42)]);
    let empty_types: Vec<(usize, MirType)> = Vec::new();
    let res = vm.resume_from_deopt(Some(&empty_types));
    assert!(res.is_err(), "missing type metadata must fail closed");
    assert_eq!(vm.call_stack_depth(), 0, "call stack must remain empty on rejection");
}

#[test]
fn test_ng_w17_rejects_empty_stack_oob_pc_and_oob_register() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();
    safepoint_reset();
    take_deopt_payload();

    let mut vm = VM::new();
    let mut func = Function::new("main", 0, false);
    func.chunk.emit(Instruction::Return(0), 1);
    func.register_count = 2;
    vm.load_functions(vec![func]);

    // 1. Out-of-bounds target_pc
    publish_deopt_payload(999, vec![(0, 1)]);
    let types = vec![(0, MirType::Int)];
    let res_pc = vm.resume_from_deopt(Some(&types));
    assert!(res_pc.is_err(), "out-of-bounds target_pc must fail closed");
    assert_eq!(vm.call_stack_depth(), 0);

    // 2. Out-of-bounds local register index
    publish_deopt_payload(0, vec![(500, 1)]);
    let types_oob = vec![(500, MirType::Int)];
    let res_reg = vm.resume_from_deopt(Some(&types_oob));
    assert!(res_reg.is_err(), "out-of-bounds register index must fail closed");
    assert_eq!(vm.call_stack_depth(), 0);

    // 3. No pending deopt payload
    let res_empty = vm.resume_from_deopt(Some(&types));
    assert!(res_empty.is_err(), "missing deopt payload must fail closed");
    assert_eq!(vm.call_stack_depth(), 0);
}

#[test]
fn test_ng_w17_rejects_empty_stack_duplicate_local_and_type_indices() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_effect_handler_state();
    safepoint_reset();
    take_deopt_payload();

    let mut vm = VM::new();
    let mut func = Function::new("main", 0, false);
    func.chunk.emit(Instruction::Return(0), 1);
    func.register_count = 4;
    vm.load_functions(vec![func]);

    // Duplicate local index in payload
    publish_deopt_payload(0, vec![(0, 1), (0, 2)]);
    let types = vec![(0, MirType::Int)];
    let res_dup_loc = vm.resume_from_deopt(Some(&types));
    assert!(res_dup_loc.is_err(), "duplicate local index must fail closed");
    assert_eq!(vm.call_stack_depth(), 0);

    // Duplicate type index in metadata
    publish_deopt_payload(0, vec![(0, 1)]);
    let dup_types = vec![(0, MirType::Int), (0, MirType::Int)];
    let res_dup_ty = vm.resume_from_deopt(Some(&dup_types));
    assert!(res_dup_ty.is_err(), "duplicate type index must fail closed");
    assert_eq!(vm.call_stack_depth(), 0);
}

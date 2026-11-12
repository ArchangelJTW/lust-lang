//! Gate 1: MinimalSimScheduler & Mutant M1 Micro-Oracle Acceptance Test (D4, Gate 1)
//!
//! Validates:
//! 1. MinimalSimScheduler executes single-fiber steps with cooperative task support.
//! 2. Yielded tasks (TaskState::Yielded) correctly pause and can be resumed with values.
//! 3. High-teeth invariant INV-BUDGET-01 Gas Progress Oracle:
//!    - Safety: gas_used strictly monotonically increases on execution step.
//!    - Liveness: failure to consume gas when progress is made triggers immediate fail-stop.
//! 4. 100% Mutation Acceptance Gate for Mutant M1 (Gas charge omission in instruction dispatch):
//!    - Baseline GREEN: normal gas charging verified.
//!    - Mutation injected (MUTATION_M1_SKIP_GAS = true).
//!    - Execution Witness confirmed (MUTATION_M1_EXECUTION_COUNT > 0).
//!    - Target Oracle RED: INV-BUDGET-01 oracle detects stalled gas and fails.
//! 5. Simulation step bound watchdog prevents infinite loops (EXHAUSTION_LIMIT_REACHED).

use core::sync::atomic::Ordering;
use lust::bytecode::compiler::Compiler;
use lust::bytecode::{Function, Instruction, NativeCallResult, Value, native_fn};
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::parser::Parser;
use lust::sim::{GasProgressOracle, MinimalSimScheduler, StepOutcome};
use lust::typechecker::TypeChecker;
use lust::vm::{
    MUTATION_M1_EXECUTION_COUNT, MUTATION_M1_SKIP_GAS, MutationM1Guard, TaskState, VM,
};

fn compile_source(source: &str) -> lust::Result<Vec<Function>> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize()?;
    let items = Parser::new(tokens).parse()?;
    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    compiler.compile_module(&items)
}

#[test]
fn test_minimal_sim_scheduler_cooperative_yield_and_resume() {
    let mut vm = VM::with_seed(100);

    let yield_fn = native_fn(|args| {
        let val = args.first().cloned().unwrap_or(Value::Int(42));
        Ok(NativeCallResult::Yield(val))
    });

    let mut func = Function::new("worker".to_string(), 0, false);
    func.register_count = 8;
    let c_fn = func.chunk.add_constant(Value::NativeFunction(yield_fn));
    let c_val = func.chunk.add_constant(Value::Int(77));
    func.chunk.emit(Instruction::LoadConst(0, c_fn), 1);
    func.chunk.emit(Instruction::LoadConst(1, c_val), 1);
    // Call yield_fn(77) -> yields 77, on resume dest reg 2 receives the resume value
    func.chunk.emit(Instruction::Call(0, 1, 1, 2), 1);
    func.chunk.emit(Instruction::Return(2), 1);

    vm.load_functions(vec![func]);

    let mut scheduler = MinimalSimScheduler::new(vm);
    let handle = scheduler
        .spawn_fiber(Value::Function(0), vec![])
        .expect("spawn must succeed");

    // First step: worker calls yield_now(77) and transitions to Yielded
    let outcome1 = scheduler.step_fiber(handle, None).expect("step 1 must succeed");
    assert_eq!(outcome1, StepOutcome::Yielded(Some(Value::Int(77))));

    // Verify task state in VM
    let task = scheduler.vm.get_task_instance(handle).expect("task must exist");
    assert_eq!(task.state, TaskState::Yielded(lust::vm::YieldReason::Explicit));

    // Second step: resume fiber with value 123 -> returns 123
    let outcome2 = scheduler
        .step_fiber(handle, Some(Value::Int(123)))
        .expect("step 2 must succeed");
    assert_eq!(outcome2, StepOutcome::Completed(Some(Value::Int(123))));

    let task = scheduler.vm.get_task_instance(handle).expect("task must exist");
    assert_eq!(task.state, TaskState::Completed);
}

static MUTATION_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn test_inv_budget_01_baseline_green() {
    let _lock = MUTATION_TEST_MUTEX.lock().unwrap();
    // Reset mutation flags to ensure clean baseline
    MUTATION_M1_SKIP_GAS.store(false, Ordering::SeqCst);
    MUTATION_M1_EXECUTION_COUNT.store(0, Ordering::SeqCst);

    let mut vm = VM::with_seed(200);
    let src = r#"
        function compute(): int
            local sum = 0
            for i = 1, 10 do
                sum = sum + i
            end
            return sum
        end
    "#;
    let functions = compile_source(src).expect("compile must succeed");
    vm.load_functions(functions);

    let mut scheduler = MinimalSimScheduler::new(vm);
    let initial_gas = scheduler.vm.gas_used();
    let mut oracle = GasProgressOracle::new(initial_gas);

    let handle = scheduler
        .spawn_fiber(Value::Function(0), vec![])
        .expect("spawn must succeed");

    let count_before = MUTATION_M1_EXECUTION_COUNT.load(Ordering::SeqCst);
    let outcome = scheduler.step_fiber(handle, None).expect("step must succeed");
    let count_after = MUTATION_M1_EXECUTION_COUNT.load(Ordering::SeqCst);
    assert_eq!(outcome, StepOutcome::Completed(Some(Value::Int(55))));

    let gas_after = scheduler.vm.gas_used();
    assert!(
        gas_after > initial_gas,
        "Baseline execution must consume gas: gas_after ({}) > initial_gas ({})",
        gas_after,
        initial_gas
    );

    // Oracle must verify progress successfully (GREEN)
    let oracle_res = oracle.verify_step_progress(gas_after, 1);
    assert!(oracle_res.is_ok(), "Oracle must PASS on baseline execution");
    assert_eq!(
        count_after,
        count_before,
        "No mutant branches should execute in baseline"
    );
}

#[test]
fn test_mutant_m1_killed_by_micro_oracle() {
    let _lock = MUTATION_TEST_MUTEX.lock().unwrap();
    // Setup clean state
    MUTATION_M1_SKIP_GAS.store(false, Ordering::SeqCst);

    let mut vm = VM::with_seed(300);
    let src = r#"
        function loop_work(): int
            local acc = 0
            for i = 1, 5 do
                acc = acc + 1
            end
            return acc
        end
    "#;
    let functions = compile_source(src).expect("compile must succeed");
    vm.load_functions(functions);

    let mut scheduler = MinimalSimScheduler::new(vm);

    let handle = scheduler
        .spawn_fiber(Value::Function(0), vec![])
        .expect("spawn must succeed");

    // =========================================================================
    // INJECT MUTANT M1: Omit gas consumption in instruction dispatch
    // =========================================================================
    let count_before = MUTATION_M1_EXECUTION_COUNT.load(Ordering::SeqCst);
    let gas_before = scheduler.vm.gas_used();
    let outcome = {
        let _m1_guard = MutationM1Guard::enable();
        scheduler.step_fiber(handle, None).expect("step must execute")
    };
    let gas_after = scheduler.vm.gas_used();

    let count_after = MUTATION_M1_EXECUTION_COUNT.load(Ordering::SeqCst);
    let witness_count = count_after.saturating_sub(count_before);

    // 1. Confirm mutant execution witness
    assert!(
        witness_count > 0,
        "Execution Witness requirement failed: mutant branches executed must be > 0 (was {})",
        witness_count
    );

    // 2. Confirm defective behavior (progress made, but gas did not increase)
    assert_eq!(outcome, StepOutcome::Completed(Some(Value::Int(5))));
    assert_eq!(
        gas_after, gas_before,
        "Mutant M1 defect confirmed: gas remained unchanged despite progress"
    );

    // 3. Confirm Oracle FAIL (RED) -> Mutant M1 is killed!
    let oracle_verdict = scheduler.gas_oracle.verify_step_progress(gas_after, 1);
    assert!(
        oracle_verdict.is_err(),
        "Oracle must catch mutant M1 and return an error"
    );
    assert_eq!(
        oracle_verdict.unwrap_err(),
        "INV-BUDGET-01-FAIL: Progress made without gas consumption",
        "Must fail with exact INV-BUDGET-01 violation code"
    );
}

#[test]
fn test_scheduler_step_exhaustion_limit() {
    let vm = VM::with_seed(400);
    let mut scheduler = MinimalSimScheduler::new(vm).with_step_limit(3);

    let handle = scheduler.vm.create_native_future_task();

    // 3 steps succeed
    for _ in 0..3 {
        let _ = scheduler.step_fiber(handle, None);
    }

    // 4th step must fail-stop with exhaustion limit
    let err = scheduler.step_fiber(handle, None);
    assert!(err.is_err());
    let err_msg = err.unwrap_err().to_string();
    assert!(
        err_msg.contains("EXHAUSTION_LIMIT_REACHED"),
        "Scheduler must prevent infinite simulation loops: {}",
        err_msg
    );
}

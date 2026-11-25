//! Gate 5: 100% Mutation Acceptance Gate (DST Phase 5, Section 3).
//!
//! Validates the strict mutation testing chain:
//! Baseline GREEN -> Mutant Injection (M1, M2, M3) -> Witness > 0 -> Specific Oracle RED.
//!
//! - Mutant M1 (Gas charge skipped):
//!   * Injection: `MUTATION_M1_SKIP_GAS` in `src/vm/execution.rs`.
//!   * Witness: `MUTATION_M1_EXECUTION_COUNT > 0`.
//!   * Oracle: `INV-BUDGET-01` (progress made without gas consumption).
//!
//! - Mutant M2 (Skip finalizer on panic/failed task):
//!   * Injection: `MUTATION_M2_SKIP_PANIC_FINALIZER` in `src/vm/tasks.rs`.
//!   * Witness: `MUTATION_M2_EXECUTION_COUNT > 0`.
//!   * Oracle: `INV-FIBER-01-EXACTLY-ONCE` (registered finalizers count != executed count).
//!
//! - Mutant M3 (Race in Rx Admission):
//!   * Injection: `MUTATION_M3_SKIP_CLOSED_CHECK` in `src/rx/scheduler.rs`.
//!   * Witness: `MUTATION_M3_EXECUTION_COUNT > 0`.
//!   * Oracle: `INV-RX-01` (admission accepted on closed scope / write after join).

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use lust::bytecode::compiler::Compiler;
use lust::bytecode::{Function, TaskHandle, Value};
use lust::config::LustConfig;
use lust::effect_abi::Scope;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::parser::Parser;
use lust::rx::scheduler::{
    MutationM3Guard, ScopeJoinCoordinator, MUTATION_M3_EXECUTION_COUNT,
    MUTATION_M3_SKIP_CLOSED_CHECK,
};
use lust::sim::invariants::{
    InvBudget01Oracle, InvFiber01Oracle, InvRx01Oracle, InvariantViolation,
};
use lust::sim::MinimalSimScheduler;
use lust::typechecker::TypeChecker;
use lust::vm::{
    MutationM1Guard, MutationM2Guard, TaskState, VM, MUTATION_M1_EXECUTION_COUNT,
    MUTATION_M1_SKIP_GAS, MUTATION_M2_EXECUTION_COUNT, MUTATION_M2_SKIP_PANIC_FINALIZER,
};

/// External watchdog ensuring no mutation test hangs.
fn with_test_watchdog<F: FnOnce() + Send + 'static>(name: &'static str, f: F) {
    let done = Arc::new(AtomicBool::new(false));
    let done_clone = done.clone();

    let handle = std::thread::spawn(move || {
        f();
        done_clone.store(true, Ordering::SeqCst);
    });

    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(5);

    while !done.load(Ordering::SeqCst) {
        if start.elapsed() > timeout {
            panic!("FAIL: WATCHDOG_TIMEOUT: Test '{}' exceeded 5s timeout", name);
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    handle.join().expect("Test thread panicked");
}

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

fn create_test_scheduler(seed: u64) -> (MinimalSimScheduler, TaskHandle) {
    let mut vm = VM::with_seed(seed);
    let src = r#"
        function loop_work(): int
            local acc = 0
            for i = 1, 3 do
                acc = acc + 1
            end
            return acc
        end
    "#;
    let functions = compile_source(src).expect("compile must succeed");
    vm.load_functions(functions);
    let mut sched = MinimalSimScheduler::new(vm);
    let fiber = sched
        .spawn_fiber(Value::Function(0), vec![])
        .expect("spawn must succeed");
    (sched, fiber)
}

// ============================================================================
// MUTANT M1: Gas Charge Skipping
// ============================================================================

#[test]
fn test_mutation_gate_m1_gas_charge_chain() {
    with_test_watchdog("mutation_gate_m1", || {
        // Step 1: Baseline GREEN
        MUTATION_M1_SKIP_GAS.store(false, Ordering::SeqCst);
        MUTATION_M1_EXECUTION_COUNT.store(0, Ordering::SeqCst);

        let (mut sched, fiber) = create_test_scheduler(101);
        let gas_before = sched.vm.gas_used();
        let outcome = sched.step_fiber(fiber, None).unwrap();
        assert_eq!(outcome, lust::sim::StepOutcome::Completed(Some(Value::Int(3))));
        let gas_after = sched.vm.gas_used();
        assert!(gas_after > gas_before, "Baseline must consume gas");

        let mut oracle = InvBudget01Oracle::new(gas_before);
        assert!(
            oracle.check_step_progress(gas_after, 1).is_ok(),
            "Baseline must pass oracle INV-BUDGET-01"
        );
        assert_eq!(
            MUTATION_M1_EXECUTION_COUNT.load(Ordering::SeqCst),
            0,
            "Baseline witness must be 0"
        );

        // Step 2: Inject Mutant M1
        let witness_count = {
            let _guard = MutationM1Guard::enable();
            let count_before = MUTATION_M1_EXECUTION_COUNT.load(Ordering::SeqCst);

            let (mut sched_mutant, fiber_mutant) = create_test_scheduler(102);
            let gas_before_mutant = sched_mutant.vm.gas_used();
            // Step execution under Mutant M1 (gas charges skipped!)
            let _ = sched_mutant.step_fiber(fiber_mutant, None).unwrap();
            let gas_after_mutant = sched_mutant.vm.gas_used();

            // Step 3: Witness verification
            let count_after = MUTATION_M1_EXECUTION_COUNT.load(Ordering::SeqCst);
            assert!(
                count_after > count_before,
                "Execution witness MUTATION_M1_EXECUTION_COUNT must increment (> 0)"
            );
            assert_eq!(
                gas_after_mutant, gas_before_mutant,
                "Gas must not change when Mutant M1 is active"
            );

            // Step 4: Specific Oracle RED (INV-BUDGET-01)
            let mut mutant_oracle = InvBudget01Oracle::new(gas_before_mutant);
            let res = mutant_oracle.check_step_progress(gas_after_mutant, 1);
            match res {
                Err(InvariantViolation::Budget01GasNotMonotonic {
                    gas_before,
                    gas_after,
                    ..
                }) => {
                    assert_eq!(gas_before, gas_before_mutant);
                    assert_eq!(gas_after, gas_after_mutant);
                    assert!(res.unwrap_err().to_string().contains("INV-BUDGET-01-FAIL"));
                }
                other => panic!("Expected InvariantViolation::Budget01GasNotMonotonic, got {:?}", other),
            }
            count_after
        };

        // Step 5: Guard dropped -> Baseline GREEN restored
        assert!(witness_count > 0);
        let (mut sched_fresh, fresh_fiber) = create_test_scheduler(103);
        let fresh_gas_before = sched_fresh.vm.gas_used();
        let _ = sched_fresh.step_fiber(fresh_fiber, None).unwrap();
        let fresh_gas_after = sched_fresh.vm.gas_used();

        let mut fresh_oracle = InvBudget01Oracle::new(fresh_gas_before);
        assert!(fresh_oracle.check_step_progress(fresh_gas_after, 1).is_ok());
    });
}

// ============================================================================
// MUTANT M2: Finalizer Skipping on Task Panic/Failure
// ============================================================================

#[test]
fn test_mutation_gate_m2_panic_finalizer_chain() {
    with_test_watchdog("mutation_gate_m2", || {
        // Step 1: Baseline GREEN
        MUTATION_M2_SKIP_PANIC_FINALIZER.store(false, Ordering::SeqCst);
        MUTATION_M2_EXECUTION_COUNT.store(0, Ordering::SeqCst);

        let mut vm = VM::new();
        let handle = vm.create_native_future_task();
        let mut scope = Scope::new(201, None);
        vm.register_scope_task(&mut scope, handle).unwrap();

        let events = Rc::new(RefCell::new(Vec::new()));
        let mut oracle = InvFiber01Oracle::new();

        // Register 2 finalizers in LIFO order
        for id in [1u64, 2u64] {
            oracle.record_register(id);
            let ev = events.clone();
            scope
                .register_finalizer(move || {
                    ev.borrow_mut().push(id);
                })
                .unwrap();
        }

        // Child task fails (panics)
        vm.complete_native_future_task(handle, Err("Simulated task panic".to_string()))
            .unwrap();
        assert_eq!(vm.get_task_instance(handle).unwrap().state, TaskState::Failed);

        // Scope closes
        assert!(vm.close_scope(&mut scope));
        assert_eq!(
            MUTATION_M2_EXECUTION_COUNT.load(Ordering::SeqCst),
            0,
            "Baseline witness must be 0"
        );

        // Record executions in oracle
        for &executed_id in events.borrow().iter() {
            oracle.record_execution(executed_id).unwrap();
        }
        assert_eq!(*events.borrow(), vec![2, 1], "Must finalize in strict LIFO order");
        assert!(
            oracle.verify_completion().is_ok(),
            "Baseline must satisfy completion guarantee"
        );

        // Step 2: Inject Mutant M2
        let mutant_witness = {
            let _guard = MutationM2Guard::enable();
            let count_before = MUTATION_M2_EXECUTION_COUNT.load(Ordering::SeqCst);

            let mut mutant_vm = VM::new();
            let mutant_handle = mutant_vm.create_native_future_task();
            let mut mutant_scope = Scope::new(202, None);
            mutant_vm.register_scope_task(&mut mutant_scope, mutant_handle).unwrap();

            let mutant_events = Rc::new(RefCell::new(Vec::new()));
            let mut mutant_oracle = InvFiber01Oracle::new();

            for id in [1u64, 2u64] {
                mutant_oracle.record_register(id);
                let ev = mutant_events.clone();
                mutant_scope
                    .register_finalizer(move || {
                        ev.borrow_mut().push(id);
                    })
                    .unwrap();
            }

            // Task fails
            mutant_vm
                .complete_native_future_task(mutant_handle, Err("Panic in task".to_string()))
                .unwrap();
            assert_eq!(
                mutant_vm.get_task_instance(mutant_handle).unwrap().state,
                TaskState::Failed
            );

            // Close scope: Mutant M2 intercepts and SKIPS finalizer execution!
            assert!(mutant_vm.close_scope(&mut mutant_scope));

            // Step 3: Witness verification
            let count_after = MUTATION_M2_EXECUTION_COUNT.load(Ordering::SeqCst);
            assert!(
                count_after > count_before,
                "Execution witness MUTATION_M2_EXECUTION_COUNT must increment (> 0)"
            );

            // Finalizers were skipped!
            assert!(
                mutant_events.borrow().is_empty(),
                "Finalizers must have been skipped by Mutant M2"
            );

            // Step 4: Specific Oracle RED (INV-FIBER-01-EXACTLY-ONCE)
            let err = mutant_oracle.verify_completion().unwrap_err();
            match err {
                InvariantViolation::Fiber01Incomplete {
                    registered,
                    executed,
                } => {
                    assert_eq!(registered, 2);
                    assert_eq!(executed, 0);
                    assert!(err.to_string().contains("INV-FIBER-01-EXACTLY-ONCE-FAIL"));
                }
                other => panic!("Expected InvariantViolation::Fiber01Incomplete, got {:?}", other),
            }

            count_after
        };

        // Step 5: Guard dropped -> Baseline GREEN restored
        assert!(mutant_witness > 0);
    });
}

// ============================================================================
// MUTANT M3: Race in Rx Admission (Closed Check Skipped)
// ============================================================================

#[test]
fn test_mutation_gate_m3_rx_admission_chain() {
    with_test_watchdog("mutation_gate_m3", || {
        // Step 1: Baseline GREEN
        MUTATION_M3_SKIP_CLOSED_CHECK.store(false, Ordering::SeqCst);
        MUTATION_M3_EXECUTION_COUNT.store(0, Ordering::SeqCst);

        let coordinator = ScopeJoinCoordinator::new();
        let scope_id = 301;
        coordinator.close_admission(scope_id);

        let oracle = InvRx01Oracle::new();
        // Admission rejected on closed scope
        assert!(
            oracle.verify_active_rejection(&coordinator, scope_id).is_ok(),
            "Baseline must reject admission on closed scope"
        );
        assert_eq!(
            MUTATION_M3_EXECUTION_COUNT.load(Ordering::SeqCst),
            0,
            "Baseline witness must be 0"
        );

        // Step 2: Inject Mutant M3
        let mutant_witness = {
            let _guard = MutationM3Guard::enable();
            let count_before = MUTATION_M3_EXECUTION_COUNT.load(Ordering::SeqCst);

            let mutant_coordinator = ScopeJoinCoordinator::new();
            let closed_scope_id = 302;
            mutant_coordinator.close_admission(closed_scope_id);

            let mutant_oracle = InvRx01Oracle::new();

            // Step 3 & 4: Active Rejection fails because closed check was skipped!
            let err = mutant_oracle
                .verify_active_rejection(&mutant_coordinator, closed_scope_id)
                .unwrap_err();

            let count_after = MUTATION_M3_EXECUTION_COUNT.load(Ordering::SeqCst);
            assert!(
                count_after > count_before,
                "Execution witness MUTATION_M3_EXECUTION_COUNT must increment (> 0)"
            );

            match err {
                InvariantViolation::Rx01ActiveRejectionFailed { scope_id } => {
                    assert_eq!(scope_id, closed_scope_id);
                    assert!(err.to_string().contains("INV-RX-01-FAIL"));
                }
                other => panic!("Expected InvariantViolation::Rx01ActiveRejectionFailed, got {:?}", other),
            }

            count_after
        };

        // Step 5: Guard dropped -> Baseline GREEN restored
        assert!(mutant_witness > 0);
        let fresh_coordinator = ScopeJoinCoordinator::new();
        let fresh_scope_id = 303;
        fresh_coordinator.close_admission(fresh_scope_id);
        let fresh_oracle = InvRx01Oracle::new();
        assert!(fresh_oracle.verify_active_rejection(&fresh_coordinator, fresh_scope_id).is_ok());
    });
}

//! Gate 5: Formal Invariants with Teeth (DST Phase 5, Section 2).
//!
//! Validates:
//! 1. INV-FIBER-01 (LIFO Finalization):
//!    - Safety: Exactly-Once (strictly 1 execution, exact reverse registration order).
//!    - Liveness: Completion Guarantee (executed count == registered count > 0).
//!    - Negative Teeth: Out-of-order, duplicate, and incomplete finalizations detected.
//! 2. INV-FIBER-02 (Masking Safe Point):
//!    - Safety: Preserved Delivery (interrupt preserved inside nested masks).
//!    - Liveness: Bound Safe Point (delivery at exactly N=1 instruction after unmasking).
//!    - Negative Teeth: Leaked delivery while masked, bound exceeded (>1), lost interrupt.
//! 3. INV-FIBER-03 (Teardown Quiescence):
//!    - Safety: Full Quiescence (all children must be Completed, Failed, or Stopped).
//!    - Liveness: Termination Bound (max_execution_steps).
//!    - Negative Teeth: Non-terminal child (Yielded/Ready/Running) rejected.
//! 4. INV-RX-01 (Rx Quiescence):
//!    - Safety: Zero-Leak (no writes to sink after join).
//!    - Liveness: Active Rejection (register_participant rejected on closed scope).
//!    - Negative Teeth: Post-join writes and failed active rejection detected.
//! 5. INV-BUDGET-01 (Gas Monotonicity):
//!    - Safety: Progress Accounting (gas strictly increases on progress).
//!    - Liveness: Non-Stall (total gas grows with instruction progress).
//!    - Negative Teeth: Zero gas consumption on progress and stall detected.
//! 6. StepCounterWatchdog & Process Timeout Watchdog.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use lust::effect_abi::{InterruptReason, Scope, ScopeTeardown};
use lust::rx::scheduler::ScopeJoinCoordinator;
use lust::sim::invariants::{
    InvBudget01Oracle, InvFiber01Oracle, InvFiber02Oracle, InvFiber03Oracle, InvRx01Oracle,
    InvariantViolation, StepCounterWatchdog,
};
use lust::vm::{TaskState, VM};

/// Helper: run a test with an external 5-second watchdog.
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

// ============================================================================
// 1. INV-FIBER-01: LIFO Finalization
// ============================================================================

#[test]
fn test_inv_fiber_01_baseline_green() {
    with_test_watchdog("inv_fiber_01_green", || {
        let mut oracle = InvFiber01Oracle::new();
        // Register 3 finalizers: 10, 20, 30
        oracle.record_register(10);
        oracle.record_register(20);
        oracle.record_register(30);

        // Execute in exact reverse (LIFO) order: 30, 20, 10
        assert!(oracle.record_execution(30).is_ok());
        assert!(oracle.record_execution(20).is_ok());
        assert!(oracle.record_execution(10).is_ok());

        // Liveness: completion guarantee
        assert!(oracle.verify_completion().is_ok());
    });
}

#[test]
fn test_inv_fiber_01_teeth_out_of_order() {
    with_test_watchdog("inv_fiber_01_teeth_order", || {
        let mut oracle = InvFiber01Oracle::new();
        oracle.record_register(1);
        oracle.record_register(2);

        // Wrong order: 1 executed instead of 2
        let err = oracle.record_execution(1).unwrap_err();
        match err {
            InvariantViolation::Fiber01LifoOrder {
                expected_id,
                actual_id,
            } => {
                assert_eq!(expected_id, 2);
                assert_eq!(actual_id, 1);
            }
            other => panic!("Expected Fiber01LifoOrder, got {:?}", other),
        }
    });
}

#[test]
fn test_inv_fiber_01_teeth_duplicate() {
    with_test_watchdog("inv_fiber_01_teeth_duplicate", || {
        let mut oracle = InvFiber01Oracle::new();
        oracle.record_register(1);
        oracle.record_register(2);

        assert!(oracle.record_execution(2).is_ok());
        // Duplicate execution of 2
        let err = oracle.record_execution(2).unwrap_err();
        match err {
            InvariantViolation::Fiber01DuplicateExecution { id } => {
                assert_eq!(id, 2);
            }
            other => panic!("Expected Fiber01DuplicateExecution, got {:?}", other),
        }
    });
}

#[test]
fn test_inv_fiber_01_teeth_incomplete_liveness() {
    with_test_watchdog("inv_fiber_01_teeth_incomplete", || {
        let mut oracle = InvFiber01Oracle::new();
        oracle.record_register(1);
        oracle.record_register(2);
        // Only 1 executed out of 2
        assert!(oracle.record_execution(2).is_ok());

        let err = oracle.verify_completion().unwrap_err();
        match err {
            InvariantViolation::Fiber01Incomplete {
                registered,
                executed,
            } => {
                assert_eq!(registered, 2);
                assert_eq!(executed, 1);
            }
            other => panic!("Expected Fiber01Incomplete, got {:?}", other),
        }
    });
}

// ============================================================================
// 2. INV-FIBER-02: Masking Safe Point
// ============================================================================

#[test]
fn test_inv_fiber_02_baseline_green() {
    with_test_watchdog("inv_fiber_02_green", || {
        let mut oracle = InvFiber02Oracle::new();

        // 1. Enter nested masks
        oracle.on_enter_mask(); // depth 1
        oracle.on_enter_mask(); // depth 2

        // 2. Request interrupt while masked
        oracle.on_request_interrupt(InterruptReason::Cancelled);

        // 3. Instruction steps while masked: preserved delivery, no error
        assert!(oracle.on_instruction_step().is_ok());
        assert!(oracle.on_instruction_step().is_ok());

        // 4. Exit inner mask
        oracle.on_exit_mask(); // depth 1
        assert!(oracle.on_instruction_step().is_ok());

        // 5. Exit outer mask
        oracle.on_exit_mask(); // depth 0

        // 6. Exactly N=1 instruction step outside mask
        assert!(oracle.on_instruction_step().is_ok());

        // 7. Deliver interrupt on step 1
        assert!(oracle.on_deliver_interrupt().is_ok());
        assert!(oracle.verify_delivery().is_ok());
    });
}

#[test]
fn test_inv_fiber_02_teeth_premature_delivery_while_masked() {
    with_test_watchdog("inv_fiber_02_teeth_premature", || {
        let mut oracle = InvFiber02Oracle::new();
        oracle.on_enter_mask();
        oracle.on_request_interrupt(InterruptReason::Cancelled);

        // Attempt delivery while mask_depth > 0 -> must fail
        let err = oracle.on_deliver_interrupt().unwrap_err();
        match err {
            InvariantViolation::Fiber02MaskLeakedDelivery { mask_depth } => {
                assert_eq!(mask_depth, 1);
            }
            other => panic!("Expected Fiber02MaskLeakedDelivery, got {:?}", other),
        }
    });
}

#[test]
fn test_inv_fiber_02_teeth_safe_point_bound_exceeded() {
    with_test_watchdog("inv_fiber_02_teeth_bound_exceeded", || {
        let mut oracle = InvFiber02Oracle::new();
        oracle.on_enter_mask();
        oracle.on_request_interrupt(InterruptReason::Cancelled);
        oracle.on_exit_mask(); // depth 0

        // Step 1: OK
        assert!(oracle.on_instruction_step().is_ok());

        // Step 2 without delivery: Bound exceeded (N=1 bound violated!)
        let err = oracle.on_instruction_step().unwrap_err();
        match err {
            InvariantViolation::Fiber02UnmaskBoundExceeded {
                instructions_after_unmask,
            } => {
                assert_eq!(instructions_after_unmask, 2);
            }
            other => panic!("Expected Fiber02UnmaskBoundExceeded, got {:?}", other),
        }
    });
}

#[test]
fn test_inv_fiber_02_teeth_lost_interrupt() {
    with_test_watchdog("inv_fiber_02_teeth_lost", || {
        let mut oracle = InvFiber02Oracle::new();
        oracle.on_enter_mask();
        oracle.on_request_interrupt(InterruptReason::Cancelled);
        oracle.on_exit_mask();

        // Scope/task exits without ever delivering interrupt
        let err = oracle.verify_delivery().unwrap_err();
        assert_eq!(err, InvariantViolation::Fiber02LostInterrupt);
    });
}

// ============================================================================
// 3. INV-FIBER-03: Teardown Quiescence
// ============================================================================

#[test]
fn test_inv_fiber_03_baseline_green() {
    with_test_watchdog("inv_fiber_03_green", || {
        let oracle = InvFiber03Oracle::new(100);
        let mut vm = VM::new();
        let handle1 = vm.create_native_future_task();
        let handle2 = vm.create_native_future_task();

        let mut scope = Scope::new(1, None);
        vm.register_scope_task(&mut scope, handle1).unwrap();
        vm.register_scope_task(&mut scope, handle2).unwrap();

        // Complete children
        vm.get_task_instance_mut(handle1).unwrap().state = TaskState::Completed;
        vm.get_task_instance_mut(handle2).unwrap().state = TaskState::Stopped;

        assert!(vm.close_scope(&mut scope));
        assert_eq!(scope.teardown, ScopeTeardown::Done);

        // Verification passes: all children are terminal
        assert!(oracle.verify_scope_quiescence(&vm, &scope).is_ok());
    });
}

#[test]
fn test_inv_fiber_03_teeth_yielded_child_quiescence_violation() {
    with_test_watchdog("inv_fiber_03_teeth_yielded", || {
        let oracle = InvFiber03Oracle::new(100);
        let mut vm = VM::new();
        let handle = vm.create_native_future_task();

        let mut scope = Scope::new(2, None);
        vm.register_scope_task(&mut scope, handle).unwrap();

        // Child remains Yielded
        vm.get_task_instance_mut(handle).unwrap().state =
            TaskState::Yielded(lust::vm::YieldReason::Explicit);

        // If scope were artificially marked Done while child is Yielded:
        scope.teardown = ScopeTeardown::Done;
        let err = oracle.verify_scope_quiescence(&vm, &scope).unwrap_err();
        match err {
            InvariantViolation::Fiber03NonTerminalChild {
                scope_id,
                child_id,
                state,
            } => {
                assert_eq!(scope_id, 2);
                assert_eq!(child_id, handle.id());
                assert_eq!(state, "yielded");
            }
            other => panic!("Expected Fiber03NonTerminalChild, got {:?}", other),
        }
    });
}

#[test]
fn test_inv_fiber_03_teeth_step_exhaustion_limit() {
    with_test_watchdog("inv_fiber_03_teeth_exhaustion", || {
        let mut oracle = InvFiber03Oracle::new(10);
        assert!(oracle.record_step(5).is_ok());
        let err = oracle.record_step(5).unwrap_err();
        match err {
            InvariantViolation::Fiber03ExhaustionLimitReached { steps, max_steps } => {
                assert_eq!(steps, 10);
                assert_eq!(max_steps, 10);
            }
            other => panic!("Expected Fiber03ExhaustionLimitReached, got {:?}", other),
        }
    });
}

// ============================================================================
// 4. INV-RX-01: Rx Quiescence
// ============================================================================

#[test]
fn test_inv_rx_01_baseline_green() {
    with_test_watchdog("inv_rx_01_green", || {
        let coordinator = ScopeJoinCoordinator::new();
        let scope_id = 42;
        let p = coordinator.register_participant(scope_id).unwrap();
        coordinator.acknowledge_exit(scope_id, p);

        // Close and wait quiescent
        assert!(coordinator.close_and_wait_quiescent(scope_id, Some(Duration::from_millis(50))));

        let mut oracle = InvRx01Oracle::new();
        oracle.mark_joined(scope_id);

        // Liveness: Active rejection on closed scope
        assert!(oracle.verify_active_rejection(&coordinator, scope_id).is_ok());
    });
}

#[test]
fn test_inv_rx_01_teeth_post_join_sink_write() {
    with_test_watchdog("inv_rx_01_teeth_sink_write", || {
        let mut oracle = InvRx01Oracle::new();
        let scope_id = 42;
        oracle.mark_joined(scope_id);

        // Attempt write after join completed -> must fail
        let err = oracle.record_sink_write(scope_id).unwrap_err();
        match err {
            InvariantViolation::Rx01WriteAfterJoin {
                scope_id: err_scope,
            } => {
                assert_eq!(err_scope, 42);
            }
            other => panic!("Expected Rx01WriteAfterJoin, got {:?}", other),
        }
    });
}

#[test]
fn test_inv_rx_01_teeth_active_rejection_violation() {
    with_test_watchdog("inv_rx_01_teeth_rejection", || {
        let coordinator = ScopeJoinCoordinator::new();
        let scope_id = 99; // scope NOT closed in coordinator
        let oracle = InvRx01Oracle::new();

        // Active rejection fails because coordinator permits registration on non-closed scope
        let err = oracle.verify_active_rejection(&coordinator, scope_id).unwrap_err();
        match err {
            InvariantViolation::Rx01ActiveRejectionFailed { scope_id: sid } => {
                assert_eq!(sid, 99);
            }
            other => panic!("Expected Rx01ActiveRejectionFailed, got {:?}", other),
        }
    });
}

// ============================================================================
// 5. INV-BUDGET-01: Gas Monotonicity
// ============================================================================

#[test]
fn test_inv_budget_01_baseline_green() {
    with_test_watchdog("inv_budget_01_green", || {
        let mut oracle = InvBudget01Oracle::new(100);
        assert!(oracle.verify_step_progress(105, 1).is_ok());
        assert!(oracle.verify_step_progress(110, 2).is_ok());
        assert!(oracle.verify_non_stall().is_ok());
    });
}

#[test]
fn test_inv_budget_01_teeth_gas_not_monotonic() {
    with_test_watchdog("inv_budget_01_teeth_zero_gas", || {
        let mut oracle = InvBudget01Oracle::new(100);
        // Instruction stepped with 0 gas consumed (gas remains 100)
        let err = oracle.check_step_progress(100, 1).unwrap_err();
        match err {
            InvariantViolation::Budget01GasNotMonotonic {
                step,
                gas_before,
                gas_after,
            } => {
                assert_eq!(step, 1);
                assert_eq!(gas_before, 100);
                assert_eq!(gas_after, 100);
            }
            other => panic!("Expected Budget01GasNotMonotonic, got {:?}", other),
        }
    });
}

#[test]
fn test_inv_budget_01_teeth_stall() {
    with_test_watchdog("inv_budget_01_teeth_stall", || {
        let mut oracle = InvBudget01Oracle::new(100);
        oracle.total_instructions = 10;
        oracle.total_gas_consumed = 0;

        let err = oracle.verify_non_stall().unwrap_err();
        match err {
            InvariantViolation::Budget01Stall {
                total_instructions,
                total_gas,
            } => {
                assert_eq!(total_instructions, 10);
                assert_eq!(total_gas, 0);
            }
            other => panic!("Expected Budget01Stall, got {:?}", other),
        }
    });
}

// ============================================================================
// 6. StepCounterWatchdog (Hang Protection)
// ============================================================================

#[test]
fn test_step_counter_watchdog() {
    with_test_watchdog("step_counter_watchdog", || {
        let mut watchdog = StepCounterWatchdog::new(5);
        for _ in 0..4 {
            assert!(watchdog.tick().is_ok());
        }
        let err = watchdog.tick().unwrap_err();
        match err {
            InvariantViolation::StepLimitExhausted {
                step_counter,
                limit,
            } => {
                assert_eq!(step_counter, 5);
                assert_eq!(limit, 5);
                assert!(err.to_string().contains("EXHAUSTION_LIMIT_REACHED"));
            }
            other => panic!("Expected StepLimitExhausted, got {:?}", other),
        }
    });
}

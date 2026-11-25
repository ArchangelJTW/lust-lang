//! High-Teeth Formal Invariants (DST Phase 5, Section 2).
//!
//! Provides fail-stop monitoring of safety and liveness invariants:
//! - INV-FIBER-01 (LIFO Finalization): Safety Exactly-Once + Liveness Completion Guarantee
//! - INV-FIBER-02 (Masking Safe Point): Safety Preserved Delivery + Liveness Bound Safe Point (N=1)
//! - INV-FIBER-03 (Teardown Quiescence): Safety Full Quiescence + Liveness Termination Bound (max_execution_steps)
//! - INV-RX-01 (Rx Quiescence): Safety Zero-Leak + Liveness Active Rejection
//! - INV-BUDGET-01 (Gas Monotonicity): Safety Progress Accounting + Liveness Non-Stall
//! - StepCounterWatchdog: Independent physical scheduler step counter (EXHAUSTION_LIMIT_REACHED)

use crate::effect_abi::{InterruptReason, Scope, ScopeTeardown};
use crate::rx::scheduler::ScopeJoinCoordinator;
use crate::rx::RxError;
use crate::vm::{TaskState, VM};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

/// Fail-stop invariant violation types with deterministic diagnostic labels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvariantViolation {
    /// INV-FIBER-01: Finalizer was executed in an order that violates LIFO.
    Fiber01LifoOrder {
        expected_id: u64,
        actual_id: u64,
    },
    /// INV-FIBER-01: Finalizer was executed more than once (violates Exactly-Once).
    Fiber01DuplicateExecution {
        id: u64,
    },
    /// INV-FIBER-01: Scope closed without completing all registered finalizers (Liveness).
    Fiber01Incomplete {
        registered: usize,
        executed: usize,
    },
    /// INV-FIBER-02: Interrupt was prematurely delivered while interrupt mask was active.
    Fiber02MaskLeakedDelivery {
        mask_depth: usize,
    },
    /// INV-FIBER-02: Safe point bound violated: interrupt was not delivered at N=1 instruction after unmasking.
    Fiber02UnmaskBoundExceeded {
        instructions_after_unmask: usize,
    },
    /// INV-FIBER-02: Pending interrupt was lost or dropped without delivery.
    Fiber02LostInterrupt,
    /// INV-FIBER-03: Scope marked Done while child task is in non-terminal state (e.g. Yielded/Ready/Running).
    Fiber03NonTerminalChild {
        scope_id: u64,
        child_id: u64,
        state: String,
    },
    /// INV-FIBER-03: Teardown did not reach quiescence within step bound limit.
    Fiber03ExhaustionLimitReached {
        steps: u64,
        max_steps: u64,
    },
    /// INV-RX-01: Zero-Leak violated: write attempted after join() completed.
    Rx01WriteAfterJoin {
        scope_id: u64,
    },
    /// INV-RX-01: Active Rejection violated: admission accepted on closed/retired scope.
    Rx01ActiveRejectionFailed {
        scope_id: u64,
    },
    /// INV-BUDGET-01: Progress made without monotonic gas consumption.
    Budget01GasNotMonotonic {
        step: u64,
        gas_before: u64,
        gas_after: u64,
    },
    /// INV-BUDGET-01: Non-stall invariant violated: instructions executed without consuming gas.
    Budget01Stall {
        total_instructions: u64,
        total_gas: u64,
    },
    /// Independent scheduler step limit reached.
    StepLimitExhausted {
        step_counter: u64,
        limit: u64,
    },
}

impl fmt::Display for InvariantViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fiber01LifoOrder {
                expected_id,
                actual_id,
            } => write!(
                f,
                "INV-FIBER-01-FAIL: Finalizer executed out of LIFO order (expected #{}, got #{})",
                expected_id, actual_id
            ),
            Self::Fiber01DuplicateExecution { id } => write!(
                f,
                "INV-FIBER-01-FAIL: Finalizer #{} executed more than once (violates Exactly-Once)",
                id
            ),
            Self::Fiber01Incomplete {
                registered,
                executed,
            } => write!(
                f,
                "INV-FIBER-01-EXACTLY-ONCE-FAIL: Completion guarantee violated (registered {}, executed {})",
                registered, executed
            ),
            Self::Fiber02MaskLeakedDelivery { mask_depth } => write!(
                f,
                "INV-FIBER-02-FAIL: Interrupt delivered prematurely while mask depth is {}",
                mask_depth
            ),
            Self::Fiber02UnmaskBoundExceeded {
                instructions_after_unmask,
            } => write!(
                f,
                "INV-FIBER-02-FAIL: Safe point bound exceeded: interrupt was not delivered at N=1 instruction after unmasking (instructions={})",
                instructions_after_unmask
            ),
            Self::Fiber02LostInterrupt => write!(
                f,
                "INV-FIBER-02-FAIL: Pending interrupt was lost or dropped without delivery"
            ),
            Self::Fiber03NonTerminalChild {
                scope_id,
                child_id,
                state,
            } => write!(
                f,
                "INV-FIBER-03-FAIL: Scope {} closed but child {} is in non-terminal state '{}'",
                scope_id, child_id, state
            ),
            Self::Fiber03ExhaustionLimitReached { steps, max_steps } => write!(
                f,
                "EXHAUSTION_LIMIT_REACHED: Child tasks in scope did not reach quiescence within limit (steps {} >= max {})",
                steps, max_steps
            ),
            Self::Rx01WriteAfterJoin { scope_id } => write!(
                f,
                "INV-RX-01-FAIL: Zero-Leak violated: write attempted after join() on closed scope {}",
                scope_id
            ),
            Self::Rx01ActiveRejectionFailed { scope_id } => write!(
                f,
                "INV-RX-01-FAIL: Active Rejection violated: admission accepted on closed/retired scope {}",
                scope_id
            ),
            Self::Budget01GasNotMonotonic {
                step,
                gas_before,
                gas_after,
            } => write!(
                f,
                "INV-BUDGET-01-FAIL: Progress made without gas consumption at step {} (gas_before={}, gas_after={})",
                step, gas_before, gas_after
            ),
            Self::Budget01Stall {
                total_instructions,
                total_gas,
            } => write!(
                f,
                "INV-BUDGET-01-FAIL: Non-stall invariant violated (instructions={}, gas={})",
                total_instructions, total_gas
            ),
            Self::StepLimitExhausted {
                step_counter,
                limit,
            } => write!(
                f,
                "EXHAUSTION_LIMIT_REACHED: Maximum simulation execution steps exceeded ({} >= {})",
                step_counter, limit
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for InvariantViolation {}

// ============================================================================
// INV-FIBER-01: LIFO Finalization Oracle
// ============================================================================

/// Invariant INV-FIBER-01 (LIFO Finalization):
/// - Safety: Exactly-Once — each registered defer/finalizer executes strictly 1 time in exact reverse registration order (LIFO).
/// - Liveness: Completion Guarantee — by the time the scope closes, the count of executed finalizers equals the count of registered finalizers (> 0).
#[derive(Debug, Clone, Default)]
pub struct InvFiber01Oracle {
    pub registered: Vec<u64>,
    pub executed: Vec<u64>,
}

impl InvFiber01Oracle {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record registration of a finalizer with given unique identifier.
    pub fn record_register(&mut self, id: u64) {
        self.registered.push(id);
    }

    /// Record execution of a finalizer with given identifier.
    /// Checks:
    /// 1. Exactly-Once: id has not been executed previously.
    /// 2. LIFO order: id matches the current expected top of stack.
    pub fn record_execution(&mut self, id: u64) -> Result<(), InvariantViolation> {
        if self.executed.contains(&id) {
            return Err(InvariantViolation::Fiber01DuplicateExecution { id });
        }
        let exec_idx = self.executed.len();
        if exec_idx >= self.registered.len() {
            return Err(InvariantViolation::Fiber01LifoOrder {
                expected_id: 0,
                actual_id: id,
            });
        }
        let expected_id = self.registered[self.registered.len() - 1 - exec_idx];
        if id != expected_id {
            return Err(InvariantViolation::Fiber01LifoOrder {
                expected_id,
                actual_id: id,
            });
        }
        self.executed.push(id);
        Ok(())
    }

    /// Liveness: Completion Guarantee.
    /// Asserts that all registered finalizers have executed and registered count > 0.
    pub fn verify_completion(&self) -> Result<(), InvariantViolation> {
        if self.registered.is_empty() || self.executed.len() != self.registered.len() {
            return Err(InvariantViolation::Fiber01Incomplete {
                registered: self.registered.len(),
                executed: self.executed.len(),
            });
        }
        Ok(())
    }

    /// Fail-stop assertion helper.
    pub fn assert_valid(&self) {
        if let Err(e) = self.verify_completion() {
            panic!("{}", e);
        }
    }
}

// ============================================================================
// INV-FIBER-02: Masking Safe Point Oracle
// ============================================================================

/// Invariant INV-FIBER-02 (Masking Safe Point):
/// - Safety: Preserved Delivery — deferred cancellation/interrupt is preserved inside masks of any nesting depth.
/// - Liveness: Bound Safe Point — delivery of cancellation occurs at exactly N=1 instruction after exiting the outer interrupt mask.
#[derive(Debug, Clone, Default)]
pub struct InvFiber02Oracle {
    pub mask_depth: usize,
    pub pending_interrupt: Option<InterruptReason>,
    pub unmasked_instruction_steps: usize,
    pub interrupt_delivered: bool,
}

impl InvFiber02Oracle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn on_enter_mask(&mut self) {
        self.mask_depth += 1;
    }

    pub fn on_exit_mask(&mut self) {
        self.mask_depth = self.mask_depth.saturating_sub(1);
        if self.mask_depth == 0 {
            self.unmasked_instruction_steps = 0;
        }
    }

    pub fn on_request_interrupt(&mut self, reason: InterruptReason) {
        if self.pending_interrupt.is_none() {
            self.pending_interrupt = Some(reason);
        }
    }

    /// Record execution of an instruction step.
    /// If outside any mask and an interrupt is pending, counts steps towards delivery bound.
    pub fn on_instruction_step(&mut self) -> Result<(), InvariantViolation> {
        if self.mask_depth == 0 && self.pending_interrupt.is_some() && !self.interrupt_delivered {
            self.unmasked_instruction_steps += 1;
            // Bound Safe Point: delivery must happen at N=1 instruction after unmasking
            if self.unmasked_instruction_steps > 1 {
                return Err(InvariantViolation::Fiber02UnmaskBoundExceeded {
                    instructions_after_unmask: self.unmasked_instruction_steps,
                });
            }
        }
        Ok(())
    }

    /// Record actual delivery of the interrupt.
    pub fn on_deliver_interrupt(&mut self) -> Result<(), InvariantViolation> {
        // Safety: cannot deliver while masked
        if self.mask_depth > 0 {
            return Err(InvariantViolation::Fiber02MaskLeakedDelivery {
                mask_depth: self.mask_depth,
            });
        }
        if self.pending_interrupt.is_none() {
            return Err(InvariantViolation::Fiber02LostInterrupt);
        }
        self.interrupt_delivered = true;
        Ok(())
    }

    /// Liveness: asserts pending interrupt was delivered.
    pub fn verify_delivery(&self) -> Result<(), InvariantViolation> {
        if self.pending_interrupt.is_some() && !self.interrupt_delivered {
            return Err(InvariantViolation::Fiber02LostInterrupt);
        }
        Ok(())
    }
}

// ============================================================================
// INV-FIBER-03: Teardown Quiescence Oracle
// ============================================================================

/// Invariant INV-FIBER-03 (Teardown Quiescence):
/// - Safety: Full Quiescence — Scope cannot transition to Done until child tasks transition to Completed, Failed, or Stopped (including guarding against Yielded).
/// - Liveness: Termination Bound — all child tasks must reach terminal within a bounded quantum of steps max_execution_steps (fail-stop).
#[derive(Debug, Clone)]
pub struct InvFiber03Oracle {
    pub max_execution_steps: u64,
    pub steps_taken: u64,
}

impl InvFiber03Oracle {
    pub fn new(max_execution_steps: u64) -> Self {
        Self {
            max_execution_steps,
            steps_taken: 0,
        }
    }

    pub fn record_step(&mut self, count: u64) -> Result<(), InvariantViolation> {
        self.steps_taken += count;
        if self.steps_taken >= self.max_execution_steps {
            return Err(InvariantViolation::Fiber03ExhaustionLimitReached {
                steps: self.steps_taken,
                max_steps: self.max_execution_steps,
            });
        }
        Ok(())
    }

    /// Safety: asserts all child tasks of a Done scope have reached terminal state.
    pub fn verify_scope_quiescence(&self, vm: &VM, scope: &Scope) -> Result<(), InvariantViolation> {
        if scope.teardown == ScopeTeardown::Done {
            for child_id in &scope.children {
                if let Some(task) = vm.task_manager.get(crate::vm::TaskId(*child_id)) {
                    if !matches!(
                        task.state,
                        TaskState::Completed | TaskState::Failed | TaskState::Stopped
                    ) {
                        return Err(InvariantViolation::Fiber03NonTerminalChild {
                            scope_id: scope.id,
                            child_id: *child_id,
                            state: task.state.as_str().to_string(),
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// INV-RX-01: Rx Quiescence Oracle
// ============================================================================

/// Invariant INV-RX-01 (Rx Quiescence):
/// - Safety: Zero-Leak — after join() completes, no callback executes writes to sink.
/// - Liveness: Active Rejection — scenario attempts to write after join, verifying rejection/drop.
#[derive(Debug, Clone, Default)]
pub struct InvRx01Oracle {
    pub joined_scopes: hashbrown::HashSet<u64>,
    pub writes_after_join: u64,
}

impl InvRx01Oracle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mark_joined(&mut self, scope_id: u64) {
        self.joined_scopes.insert(scope_id);
    }

    pub fn is_joined(&self, scope_id: u64) -> bool {
        self.joined_scopes.contains(&scope_id)
    }

    /// Safety: Zero-Leak. Verifies that sink writes are forbidden on a joined scope.
    pub fn record_sink_write(&mut self, scope_id: u64) -> Result<(), InvariantViolation> {
        if self.is_joined(scope_id) {
            self.writes_after_join += 1;
            return Err(InvariantViolation::Rx01WriteAfterJoin { scope_id });
        }
        Ok(())
    }

    /// Liveness: Active Rejection. Attempts participant registration on joined scope and asserts rejection.
    pub fn verify_active_rejection(
        &self,
        coordinator: &ScopeJoinCoordinator,
        scope_id: u64,
    ) -> Result<(), InvariantViolation> {
        match coordinator.register_participant(scope_id) {
            Err(RxError::Cancelled) => Ok(()),
            _ => Err(InvariantViolation::Rx01ActiveRejectionFailed { scope_id }),
        }
    }
}

// ============================================================================
// INV-BUDGET-01: Gas Monotonicity Oracle
// ============================================================================

/// Invariant INV-BUDGET-01 (Gas Monotonicity):
/// - Safety: Progress Accounting — on every computation progress step, gas_used strictly monotonically increases by opcode weight.
/// - Liveness: Non-Stall — total gas grows proportionally to the number of executed instructions.
#[derive(Debug, Clone)]
pub struct InvBudget01Oracle {
    pub last_gas: u64,
    pub total_gas_consumed: u64,
    pub total_instructions: u64,
    pub step: u64,
}

impl InvBudget01Oracle {
    pub fn new(initial_gas: u64) -> Self {
        Self {
            last_gas: initial_gas,
            total_gas_consumed: 0,
            total_instructions: 0,
            step: 0,
        }
    }

    /// Verifies step progress against Safety Progress Accounting.
    /// If instructions_stepped > 0, gas must strictly increase.
    pub fn verify_step_progress(
        &mut self,
        current_gas: u64,
        instructions_stepped: u64,
    ) -> Result<(), &'static str> {
        self.step += 1;
        if instructions_stepped > 0 {
            if current_gas <= self.last_gas {
                return Err("INV-BUDGET-01-FAIL: Progress made without gas consumption");
            }
            self.total_gas_consumed += current_gas.saturating_sub(self.last_gas);
            self.total_instructions += instructions_stepped;
        }
        self.last_gas = current_gas;
        Ok(())
    }

    /// Rich error variant of verify_step_progress.
    pub fn check_step_progress(
        &mut self,
        current_gas: u64,
        instructions_stepped: u64,
    ) -> Result<(), InvariantViolation> {
        self.step += 1;
        if instructions_stepped > 0 {
            if current_gas <= self.last_gas {
                return Err(InvariantViolation::Budget01GasNotMonotonic {
                    step: self.step,
                    gas_before: self.last_gas,
                    gas_after: current_gas,
                });
            }
            self.total_gas_consumed += current_gas.saturating_sub(self.last_gas);
            self.total_instructions += instructions_stepped;
        }
        self.last_gas = current_gas;
        Ok(())
    }

    /// Liveness: Non-Stall. Asserts that progress in instructions yielded proportional gas consumption.
    pub fn verify_non_stall(&self) -> Result<(), InvariantViolation> {
        if self.total_instructions > 0 && self.total_gas_consumed == 0 {
            return Err(InvariantViolation::Budget01Stall {
                total_instructions: self.total_instructions,
                total_gas: self.total_gas_consumed,
            });
        }
        Ok(())
    }
}

pub type GasProgressOracle = InvBudget01Oracle;
pub type GasMonotonicityOracle = InvBudget01Oracle;

// ============================================================================
// Hang Protection: StepCounterWatchdog
// ============================================================================

/// Independent physical scheduler step counter preventing infinite simulation loops.
/// Aborts with EXHAUSTION_LIMIT_REACHED when step limit is reached.
#[derive(Debug, Clone)]
pub struct StepCounterWatchdog {
    pub step_counter: u64,
    pub max_execution_steps: u64,
}

impl StepCounterWatchdog {
    pub fn new(max_execution_steps: u64) -> Self {
        Self {
            step_counter: 0,
            max_execution_steps,
        }
    }

    pub fn tick(&mut self) -> Result<u64, InvariantViolation> {
        self.step_counter += 1;
        if self.step_counter >= self.max_execution_steps {
            return Err(InvariantViolation::StepLimitExhausted {
                step_counter: self.step_counter,
                limit: self.max_execution_steps,
            });
        }
        Ok(self.step_counter)
    }

    pub fn reset(&mut self) {
        self.step_counter = 0;
    }
}

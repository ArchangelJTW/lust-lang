//! Deterministic Simulation Testing (DST) framework for Lust runtime.
//! Phase 1: MinimalSimScheduler and state isolation micro-harness.
//! Phase 2: Clock & Timers separation, deterministic virtual time, nonblocking sleep.

pub mod clock;
pub mod fault;
pub mod host_call;
pub mod invariants;
pub mod replay;
pub mod scheduler;
pub mod tier2;

pub use clock::{
    CivilDateTime, LustClock, MonotonicClock, SimClock, SystemMonotonicClock, SystemWallClock,
    TimerControl, TimerId, TimerQueue, WallClock,
};
pub use fault::{
    FaultConfiguration, FaultInjectedVfs, FaultKind, InterruptSchedule, MemoryFaultPolicy,
    VfsFaultRule, VfsOp,
};
pub use host_call::{
    BlockingHostCallContract, assert_non_blocking_host_call, fail_stop_blocking_host_call,
};
pub use invariants::{
    GasMonotonicityOracle, GasProgressOracle, InvariantViolation, InvBudget01Oracle,
    InvFiber01Oracle, InvFiber02Oracle, InvFiber03Oracle, InvRx01Oracle, StepCounterWatchdog,
};
pub use replay::{
    EventLog, ReplayBundle, ReplayEngine, ReplayError, ReplayEvent, ReplayEventKind,
    ReplayManifest, ReplayResult, Shrinker, ShrunkReplay,
};
pub use scheduler::PreemptiveExplorationScheduler;
pub use tier2::{
    Tier2InterruptSchedule, Tier2ReplayEngine, Tier2SafepointController, Tier2SimAction,
    Tier2SimSession,
};

use crate::bytecode::{TaskHandle, Value};
use crate::error::{LustError, Result};
use crate::vm::TaskState;
use crate::vm::VM;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::result::Result as CoreResult;
use core::time::Duration;

#[derive(Debug, Clone, PartialEq)]
pub enum StepOutcome {
    Progress(TaskState),
    Completed(Option<Value>),
    Yielded(Option<Value>),
    YieldedTimer(TimerId),
    Stopped,
    Failed,
}

/// MinimalSimScheduler: minimal cooperative loop with support for TaskState::Yielded,
/// virtual time advancement, single-fiber stepping, and micro-oracle integration.
pub struct MinimalSimScheduler {
    pub vm: VM,
    pub step_counter: u64,
    pub max_execution_steps: u64,
    pub gas_oracle: GasProgressOracle,
}

impl MinimalSimScheduler {
    pub fn new(vm: VM) -> Self {
        let initial_gas = vm.gas_used();
        Self {
            vm,
            step_counter: 0,
            max_execution_steps: 10_000,
            gas_oracle: GasProgressOracle::new(initial_gas),
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        let mut vm = VM::with_seed(seed);
        // MinimalSimScheduler defaults to simulated clock, MemFs, isolated env, and captured console
        vm.clock = alloc::sync::Arc::new(LustClock::simulated(0, 0));
        #[cfg(feature = "std")]
        {
            vm.vfs = alloc::sync::Arc::new(crate::vfs::MemFs::new());
            vm.env = crate::vfs::VirtualEnv::isolated();
            vm.console = crate::vfs::VirtualConsole::captured();
        }
        Self::new(vm)
    }

    pub fn with_sim_clock(seed: u64, epoch_nanos: u64, monotonic_nanos: u64) -> Self {
        let mut vm = VM::with_sim_clock(seed, epoch_nanos, monotonic_nanos);
        #[cfg(feature = "std")]
        {
            vm.vfs = alloc::sync::Arc::new(crate::vfs::MemFs::new());
            vm.env = crate::vfs::VirtualEnv::isolated();
            vm.console = crate::vfs::VirtualConsole::captured();
        }
        Self::new(vm)
    }

    pub fn with_step_limit(mut self, limit: u64) -> Self {
        self.max_execution_steps = limit;
        self
    }

    /// Spawns a fiber in Ready state without running it immediately.
    pub fn spawn_fiber(&mut self, func: Value, args: Vec<Value>) -> Result<TaskHandle> {
        self.vm.create_task_value(func, args)
    }

    /// Execute a single cooperative step on a specific fiber.
    /// Supports resuming yielded fibers and transitions to TaskState::Yielded, Completed, Stopped, Failed.
    pub fn step_fiber(
        &mut self,
        handle: TaskHandle,
        resume_val: Option<Value>,
    ) -> Result<StepOutcome> {
        if self.step_counter >= self.max_execution_steps {
            return Err(LustError::RuntimeError {
                message: "EXHAUSTION_LIMIT_REACHED: Maximum simulation execution steps exceeded".to_string(),
            });
        }
        self.step_counter += 1;

        let _vm_guard = crate::vm::CurrentVmGuard::from_vm(&mut self.vm);

        let res = self.vm.resume_task_handle(handle, resume_val);

        let (state, last_yield, last_result, timer_id) = {
            let task = self.vm.get_task_instance(handle)?;
            (
                task.state.clone(),
                task.last_yield.clone(),
                task.last_result.clone(),
                task.timer_id,
            )
        };

        match res {
            Ok(()) => match state {
                TaskState::Yielded(reason) => match reason {
                    crate::vm::YieldReason::Timer(tid) => Ok(StepOutcome::YieldedTimer(tid)),
                    crate::vm::YieldReason::QuantumExhausted => Ok(StepOutcome::Yielded(None)),
                    crate::vm::YieldReason::Parked(_) => Ok(StepOutcome::Yielded(None)),
                    crate::vm::YieldReason::Explicit => {
                        if let Some(tid) = timer_id {
                            Ok(StepOutcome::YieldedTimer(tid))
                        } else {
                            Ok(StepOutcome::Yielded(last_yield))
                        }
                    }
                },
                TaskState::Completed => Ok(StepOutcome::Completed(last_result)),
                TaskState::Stopped => Ok(StepOutcome::Stopped),
                TaskState::Failed => Ok(StepOutcome::Failed),
                other => Ok(StepOutcome::Progress(other)),
            },
            Err(e) => {
                if state == TaskState::Failed {
                    Ok(StepOutcome::Failed)
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Advance simulated virtual time by duration, triggering any expired timers.
    pub fn advance_time(&mut self, duration: Duration) {
        self.vm.clock.advance(duration);
        let now = self.vm.clock.now_nanos();
        let actions = self.vm.clock.poll_expired_timers(now);
        for action in actions {
            action();
        }
    }

    /// Return earliest pending timer deadline in nanoseconds.
    pub fn next_deadline(&self) -> Option<u64> {
        self.vm.clock.next_deadline()
    }

    /// Advance virtual time to the next pending timer deadline and fire expired timers.
    pub fn advance_to_next_deadline(&mut self) -> bool {
        if let Some(next) = self.next_deadline() {
            let now = self.vm.clock.now_nanos();
            if next > now {
                self.advance_time(Duration::from_nanos(next - now));
            } else {
                let actions = self.vm.clock.poll_expired_timers(now);
                for action in actions {
                    action();
                }
            }
            return true;
        }
        false
    }

    /// Runs a fiber with virtual time advancement until it completes or reaches max_steps.
    pub fn run_task_with_virtual_time(
        &mut self,
        handle: TaskHandle,
        max_steps: u64,
    ) -> Result<StepOutcome> {
        let mut steps = 0;
        while steps < max_steps {
            if self.step_counter >= self.max_execution_steps {
                return Err(LustError::RuntimeError {
                    message: "EXHAUSTION_LIMIT_REACHED: Maximum simulation execution steps exceeded".to_string(),
                });
            }
            let outcome = self.step_fiber(handle, None)?;
            steps += 1;
            match outcome {
                StepOutcome::YieldedTimer(_tid) => {
                    if self.advance_to_next_deadline() {
                        if let Ok(task) = self.vm.get_task_instance_mut(handle) {
                            task.timer_id = None;
                        }
                        continue;
                    }
                    return Ok(outcome);
                }
                StepOutcome::Yielded(_) => {
                    return Ok(outcome);
                }
                terminal => return Ok(terminal),
            }
        }
        Err(LustError::RuntimeError {
            message: "EXHAUSTION_LIMIT_REACHED: Task did not reach terminal within step limit".to_string(),
        })
    }

    /// Verifies mutant M1: checks step progress against the gas progress oracle
    pub fn verify_m1_step(
        &mut self,
        handle: TaskHandle,
        resume_val: Option<Value>,
    ) -> CoreResult<(), &'static str> {
        let gas_before = self.vm.gas_used();
        let _ = self.step_fiber(handle, resume_val);
        let gas_after = self.vm.gas_used();

        if gas_after <= gas_before {
            return Err("INV-BUDGET-01-FAIL: Progress made without gas consumption");
        }
        self.gas_oracle.verify_step_progress(gas_after, 1)
    }
}

#[cfg(feature = "std")]
std::thread_local! {
    static SIMULATED_MODE_FLAG: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

#[cfg(not(feature = "std"))]
static SIMULATED_MODE_FLAG: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Check if global simulation mode is active.
pub fn is_simulated_mode() -> bool {
    #[cfg(feature = "std")]
    {
        SIMULATED_MODE_FLAG.with(|flag| flag.get())
    }
    #[cfg(not(feature = "std"))]
    {
        SIMULATED_MODE_FLAG.load(core::sync::atomic::Ordering::SeqCst)
    }
}

/// Explicitly set global simulation mode.
pub fn set_simulated_mode(active: bool) {
    #[cfg(feature = "std")]
    {
        SIMULATED_MODE_FLAG.with(|flag| flag.set(active));
    }
    #[cfg(not(feature = "std"))]
    {
        SIMULATED_MODE_FLAG.store(active, core::sync::atomic::Ordering::SeqCst);
    }
}

/// RAII Guard activating simulation mode for the scope of the guard.
pub struct SimModeGuard {
    prev: bool,
}

impl SimModeGuard {
    pub fn enter() -> Self {
        let prev = is_simulated_mode();
        set_simulated_mode(true);
        Self { prev }
    }
}

impl Drop for SimModeGuard {
    fn drop(&mut self) {
        set_simulated_mode(self.prev);
    }
}


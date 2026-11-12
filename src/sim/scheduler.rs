//! Preemptive Exploration Scheduler (DES Preemption & Interleaving, D4).
//!
//! Provides deterministic instruction-level preemption and ready queue interleaving
//! driven by pseudo-random entropy (StdRng with explicit seed).

use crate::bytecode::{TaskHandle, Value};
use crate::error::{LustError, Result};
use crate::sim::{GasProgressOracle, LustClock, SimModeGuard};
use crate::vm::{FiberContinuation, TaskState, VM, YieldReason};
use alloc::string::ToString;
use alloc::vec::Vec;
use core::time::Duration;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

pub struct PreemptiveExplorationScheduler {
    pub vm: VM,
    pub ready_queue: Vec<TaskHandle>,
    pub rng: StdRng,
    pub seed: u64,
    pub max_quantum: u64,
    pub step_counter: usize,
    pub max_execution_steps: usize,
    pub gas_oracle: GasProgressOracle,
    pub execution_log: Vec<(TaskHandle, u64, TaskState)>,
    pub event_log: Option<crate::sim::replay::EventLog>,
}

impl PreemptiveExplorationScheduler {
    pub fn new(vm: VM, seed: u64) -> Self {
        let initial_gas = vm.gas_used();
        Self {
            vm,
            ready_queue: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
            seed,
            max_quantum: 5,
            step_counter: 0,
            max_execution_steps: 100_000,
            gas_oracle: GasProgressOracle::new(initial_gas),
            execution_log: Vec::new(),
            event_log: None,
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        let mut vm = VM::with_seed(seed);
        vm.clock = alloc::sync::Arc::new(LustClock::simulated(0, 0));
        #[cfg(feature = "std")]
        {
            vm.vfs = alloc::sync::Arc::new(crate::vfs::MemFs::new());
            vm.env = crate::vfs::VirtualEnv::isolated();
            vm.console = crate::vfs::VirtualConsole::captured();
        }
        Self::new(vm, seed)
    }

    pub fn with_max_quantum(mut self, max_quantum: u64) -> Self {
        self.max_quantum = max_quantum.max(1);
        self
    }

    pub fn with_step_limit(mut self, limit: usize) -> Self {
        self.max_execution_steps = limit;
        self
    }

    /// Enables event logging for this scheduler.
    pub fn with_event_log(mut self) -> Self {
        self.event_log = Some(crate::sim::replay::EventLog::new());
        self
    }

    /// Enables event logging in-place.
    pub fn enable_event_log(&mut self) {
        if self.event_log.is_none() {
            self.event_log = Some(crate::sim::replay::EventLog::new());
        }
    }

    /// Takes the recorded event log, replacing it with None.
    pub fn take_event_log(&mut self) -> Option<crate::sim::replay::EventLog> {
        self.event_log.take()
    }

    /// Spawns a fiber and places it in the ready queue.
    pub fn spawn_fiber(&mut self, func: Value, args: Vec<Value>) -> Result<TaskHandle> {
        let handle = self.vm.create_task_value(func, args)?;
        self.ready_queue.push(handle);
        Ok(handle)
    }

    /// Adds an existing fiber handle to the ready queue if not already present.
    pub fn add_ready_fiber(&mut self, handle: TaskHandle) {
        if !self.ready_queue.contains(&handle) {
            self.ready_queue.push(handle);
        }
    }

    /// Returns whether the scheduler is quiescent (no ready fibers and no pending timers).
    pub fn is_quiescent(&self) -> bool {
        self.ready_queue.is_empty() && self.vm.clock.next_deadline().is_none()
    }

    /// Advance virtual time to the next pending timer deadline, triggering expired timers
    /// and moving unblocked fibers into the ready queue.
    pub fn advance_to_next_deadline(&mut self) -> bool {
        if let Some(next) = self.vm.clock.next_deadline() {
            let now = self.vm.clock.now_nanos();
            if next > now {
                self.advance_time(Duration::from_nanos(next - now));
            } else {
                let actions = self.vm.clock.poll_expired_timers(now);
                for action in actions {
                    action();
                }
                self.wake_expired_timers();
            }
            return true;
        }
        false
    }

    /// Advance virtual time by a specific duration and wake expired timers.
    pub fn advance_time(&mut self, duration: Duration) {
        self.vm.clock.advance(duration);
        let now = self.vm.clock.now_nanos();
        if let Some(ref mut log) = self.event_log {
            log.record(
                self.step_counter as u64,
                now,
                crate::sim::replay::ReplayEventKind::TimeAdvanced {
                    delta_nanos: duration.as_nanos() as u64,
                    now_nanos: now,
                },
            );
        }
        let actions = self.vm.clock.poll_expired_timers(now);
        for action in actions {
            action();
        }
        self.wake_expired_timers();
    }

    fn wake_expired_timers(&mut self) {
        let mut woken = Vec::new();
        for task in self.vm.task_manager.iter() {
            if let TaskState::Yielded(YieldReason::Timer(tid)) = task.state {
                if !self.vm.clock.is_timer_pending(tid) {
                    woken.push(task.handle());
                }
            }
        }
        woken.sort_by_key(|h| h.0);
        for handle in woken {
            if let Ok(task) = self.vm.get_task_instance_mut(handle) {
                task.timer_id = None;
                task.state = TaskState::Ready;
                if !self.ready_queue.contains(&handle) {
                    self.ready_queue.push(handle);
                }
            }
        }
    }

    /// Executes one quantum step on a deterministically chosen fiber from the ready queue.
    pub fn step_one(&mut self) -> Result<Option<(TaskHandle, TaskState)>> {
        if self.step_counter >= self.max_execution_steps {
            return Err(LustError::RuntimeError {
                message: "EXHAUSTION_LIMIT_REACHED: Maximum simulation execution steps exceeded".to_string(),
            });
        }

        // If ready queue is empty, attempt to advance virtual time to next timer
        if self.ready_queue.is_empty() {
            if self.advance_to_next_deadline() {
                // Timers advanced; check if tasks became ready
            }
            if self.ready_queue.is_empty() {
                return Ok(None);
            }
        }

        // Shuffle ready queue deterministically based on PRNG seed
        self.ready_queue.shuffle(&mut self.rng);
        let handle = self.ready_queue.remove(0);

        // Deterministically pick quantum between 1 and max_quantum
        let quantum = if self.max_quantum <= 1 {
            1
        } else {
            self.rng.random_range(1..=self.max_quantum)
        };

        self.step_counter += 1;
        let gas_before = self.vm.gas_used();

        let _guard = SimModeGuard::enter();
        let state = self.vm.resume_task_handle_with_quantum(handle, quantum, None)?;

        let gas_after = self.vm.gas_used();
        if gas_after > gas_before {
            let _ = self.gas_oracle.verify_step_progress(gas_after, 1);
        }

        if let Some(ref mut log) = self.event_log {
            let now = self.vm.clock.now_nanos();
            log.record(
                self.step_counter as u64,
                now,
                crate::sim::replay::ReplayEventKind::TaskScheduled {
                    task_id: handle.0,
                    quantum,
                },
            );
            if gas_after > gas_before {
                log.record(
                    self.step_counter as u64,
                    now,
                    crate::sim::replay::ReplayEventKind::GasBurned {
                        task_id: handle.0,
                        gas_delta: gas_after - gas_before,
                        total_gas: gas_after,
                    },
                );
            }
            log.record(
                self.step_counter as u64,
                now,
                crate::sim::replay::ReplayEventKind::StateChanged {
                    task_id: handle.0,
                    from_state: "Ready".to_string(),
                    to_state: alloc::format!("{:?}", state),
                },
            );
        }

        self.execution_log.push((handle, quantum, state.clone()));

        match &state {
            TaskState::Yielded(YieldReason::QuantumExhausted) => {
                // Preempted: fiber remains ready for next scheduling round
                self.ready_queue.push(handle);
            }
            TaskState::Yielded(YieldReason::Explicit) => {
                // Voluntarily yielded: fiber is ready for next scheduling round
                self.ready_queue.push(handle);
            }
            TaskState::Yielded(YieldReason::Timer(_)) => {
                // Fiber is waiting on a timer; will be added to ready_queue when timer fires
            }
            TaskState::Ready => {
                self.ready_queue.push(handle);
            }
            TaskState::Completed | TaskState::Stopped | TaskState::Failed => {
                // Terminal state; not re-enqueued
            }
            TaskState::Running => unreachable!(),
        }

        Ok(Some((handle, state)))
    }

    /// Drives the simulation until all ready fibers and timers complete (quiescence),
    /// or until the given max_steps limit is reached.
    pub fn run_until_quiescent_or_limit(&mut self, max_steps: usize) -> Result<usize> {
        let mut steps = 0;
        while steps < max_steps {
            match self.step_one()? {
                Some(_) => {
                    steps += 1;
                }
                None => {
                    // No ready fibers and no pending timers -> Quiescence achieved!
                    return Ok(steps);
                }
            }
        }
        if self.is_quiescent() {
            Ok(steps)
        } else {
            Err(LustError::RuntimeError {
                message: "EXHAUSTION_LIMIT_REACHED: Simulation did not reach quiescence within step limit".to_string(),
            })
        }
    }

    /// Inspect current state of a task.
    pub fn get_task_state(&self, handle: TaskHandle) -> Result<TaskState> {
        let task = self.vm.get_task_instance(handle)?;
        Ok(task.state.clone())
    }

    /// Inspect continuation of a yielded task.
    pub fn get_task_continuation(&self, handle: TaskHandle) -> Result<Option<FiberContinuation>> {
        let task = self.vm.get_task_instance(handle)?;
        Ok(task.continuation.clone())
    }

    /// Inspect final result of a completed task.
    pub fn get_task_result(&self, handle: TaskHandle) -> Result<Option<Value>> {
        let task = self.vm.get_task_instance(handle)?;
        Ok(task.last_result.clone())
    }
}

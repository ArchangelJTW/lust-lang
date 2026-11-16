//! Coordination Deferred: single-assignment promise / latch for asynchronous fiber coordination.
//!
//! Complies with `RFC-STDLIB-ASYNC-COORDINATION` (§3.4):
//! - Single-assignment promise backed by a stable VM `WaitKey`.
//! - Idempotent completion (`complete`, `fail`, `complete_exit`): only the first call sets the outcome.
//! - Multiple parked awaiters wake simultaneously upon resolution.
//! - Race-safe await: if completed concurrently during parking, wakes immediately without deadlock.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use crate::effect_abi::{Cause, Exit, FiberExecutionState, InterruptReason};
use crate::vm::{TaskId, TaskManager, TaskState, WaitKey, YieldReason, VM};

static NEXT_DEFERRED_WAIT_KEY: AtomicU64 = AtomicU64::new(70_000);

/// Single-assignment promise representing a value of type `T` or error `E`.
#[derive(Debug, Clone)]
pub struct Deferred<T, E = String> {
    result: Option<Exit<T, E>>,
    key: WaitKey,
    waiters: Vec<TaskId>,
}

impl<T, E> Deferred<T, E> {
    /// Creates an empty `Deferred` with a newly allocated unique `WaitKey`.
    pub fn new() -> Self {
        let key = WaitKey(NEXT_DEFERRED_WAIT_KEY.fetch_add(1, Ordering::SeqCst));
        Self {
            result: None,
            key,
            waiters: Vec::new(),
        }
    }

    /// Alias for `new`.
    pub fn make() -> Self {
        Self::new()
    }

    /// Creates a `Deferred` bound to a specific `WaitKey`.
    pub fn with_key(key: WaitKey) -> Self {
        Self {
            result: None,
            key,
            waiters: Vec::new(),
        }
    }

    /// Returns the associated `WaitKey`.
    pub fn key(&self) -> WaitKey {
        self.key
    }

    /// Non-blocking check for completion.
    pub fn poll(&self) -> Option<&Exit<T, E>> {
        self.result.as_ref()
    }

    /// Returns `true` if this `Deferred` has already been resolved.
    pub fn is_completed(&self) -> bool {
        self.result.is_some()
    }

    /// Returns the count of currently registered awaiters.
    pub fn waiters_count(&self) -> usize {
        self.waiters.len()
    }

    /// Resolves this `Deferred` with an explicit `Exit` status.
    /// Returns `true` on the first call, `false` if already completed.
    pub fn complete_exit(&mut self, exit: Exit<T, E>) -> bool {
        if self.result.is_some() {
            false
        } else {
            self.result = Some(exit);
            true
        }
    }

    /// Resolves with a successful value.
    /// Returns `true` on the first call, `false` if already completed.
    pub fn complete(&mut self, val: T) -> bool {
        self.complete_exit(Exit::Success(val))
    }

    /// Resolves with a failure error.
    /// Returns `true` on the first call, `false` if already completed.
    pub fn fail(&mut self, err: E) -> bool {
        self.complete_exit(Exit::Failure(Cause::Fail(err)))
    }

    /// Resolves with an interruption reason.
    /// Returns `true` on the first call, `false` if already completed.
    pub fn interrupt(&mut self, reason: InterruptReason) -> bool {
        self.complete_exit(Exit::Failure(Cause::Interrupted(reason)))
    }

    /// Resolves the `Deferred` and wakes all waiting fibers in the `TaskManager`.
    /// Returns `(true, woken_tasks)` on success, or `(false, [])` if already resolved.
    pub fn complete_and_wake_tm(
        &mut self,
        exit: Exit<T, E>,
        tm: &mut TaskManager,
    ) -> (bool, Vec<TaskId>) {
        if self.complete_exit(exit) {
            self.waiters.clear();
            let woken = tm.wake(self.key, true);
            (true, woken)
        } else {
            (false, Vec::new())
        }
    }

    /// Resolves the `Deferred` and wakes all waiting fibers in the `VM`.
    pub fn complete_and_wake(
        &mut self,
        exit: Exit<T, E>,
        vm: &mut VM,
    ) -> (bool, Vec<TaskId>) {
        self.complete_and_wake_tm(exit, &mut vm.task_manager)
    }

    /// Awaits the value or parks the calling task in the `TaskManager`.
    ///
    /// If completed, returns `Ok(Some(&exit))`.
    /// If pending, parks `task_id` on this `WaitKey` and returns `Err(WaitKey)`.
    pub fn await_or_park_tm<'a>(
        &'a mut self,
        task_id: TaskId,
        tm: &mut TaskManager,
    ) -> Result<&'a Exit<T, E>, WaitKey> {
        if let Some(res) = self.result.as_ref() {
            return Ok(res);
        }
        if !self.waiters.contains(&task_id) {
            self.waiters.push(task_id);
        }
        if tm.park(task_id, self.key) {
            // Race-safe re-check after parking: if resolved during park, wake immediately
            if let Some(res) = self.result.as_ref() {
                tm.wake(self.key, false);
                return Ok(res);
            }
            Err(self.key)
        } else if let Some(res) = self.result.as_ref() {
            Ok(res)
        } else {
            Err(self.key)
        }
    }

    /// Awaits the value or parks the calling task in the `VM`.
    pub fn await_or_park_vm<'a>(
        &'a mut self,
        task_id: TaskId,
        vm: &mut VM,
    ) -> Result<&'a Exit<T, E>, WaitKey> {
        self.await_or_park_tm(task_id, &mut vm.task_manager)
    }

    /// Cancels a registered waiter task from this `Deferred`.
    pub fn cancel_waiter(&mut self, task_id: TaskId) {
        self.waiters.retain(|id| *id != task_id);
    }

    /// Cancels a waiter and unregisters it from the `TaskManager`.
    pub fn cancel_waiter_tm(&mut self, task_id: TaskId, tm: &mut TaskManager) {
        self.cancel_waiter(task_id);
        tm.unregister_task_from_waiters(task_id);
    }

    /// Cancels a waiter and unregisters it from the `VM`.
    pub fn cancel_waiter_vm(&mut self, task_id: TaskId, vm: &mut VM) {
        self.cancel_waiter_tm(task_id, &mut vm.task_manager);
    }

    /// Cleans up any dead or cancelled waiters from the waiter list.
    pub fn cleanup_dead_waiters_tm(&mut self, tm: &TaskManager) {
        let key = self.key;
        self.waiters.retain(|&id| {
            tm.get(id).is_some_and(|t| {
                matches!(t.state, TaskState::Yielded(YieldReason::Parked(k)) if k == key)
                    && t.fiber_context.state != FiberExecutionState::Interrupted
            })
        });
    }
}

impl<T, E> Default for Deferred<T, E> {
    fn default() -> Self {
        Self::new()
    }
}

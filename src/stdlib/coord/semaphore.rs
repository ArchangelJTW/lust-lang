//! Coordination Semaphore: counting semaphore for cooperative fiber concurrency control.
//!
//! Complies with `RFC-STDLIB-ASYNC-COORDINATION` (§3.4):
//! - Counting semaphore (`make(permits)`, `with_permits(n, eff)`, `acquire`/`release`).
//! - Fair FIFO waiter queue with head-of-line blocking (preventing queue jumping).
//! - Automatic pruning of cancelled or interrupted waiters WITHOUT permit leakage.
//! - RAII guard pattern in `with_permits` guaranteeing 100% permit restoration on exit, error, or cancellation.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use crate::effect_abi::FiberExecutionState;
use crate::vm::{TaskId, TaskManager, TaskState, WaitKey, YieldReason, VM};

static NEXT_SEMAPHORE_WAIT_KEY: AtomicU64 = AtomicU64::new(80_000);

/// Core error types for semaphore operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemaphoreError {
    /// Insufficient permits available for non-blocking acquisition.
    Unavailable,
    /// Requested zero permits or invalid permit count.
    InvalidPermits(usize),
}

impl core::fmt::Display for SemaphoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unavailable => write!(f, "Semaphore permits unavailable"),
            Self::InvalidPermits(n) => write!(f, "Invalid permit count: {}", n),
        }
    }
}

impl std::error::Error for SemaphoreError {}

/// Counting semaphore managing shared permits among cooperative fibers.
#[derive(Debug)]
pub struct Semaphore {
    permits: usize,
    key: WaitKey,
    waiters: Vec<(TaskId, usize)>,
}

impl Semaphore {
    /// Creates a new `Semaphore` with the specified initial number of permits.
    pub fn new(permits: usize) -> Self {
        let key = WaitKey(NEXT_SEMAPHORE_WAIT_KEY.fetch_add(1, Ordering::SeqCst));
        Self {
            permits,
            key,
            waiters: Vec::new(),
        }
    }

    /// Alias for `new`.
    pub fn make(permits: usize) -> Self {
        Self::new(permits)
    }

    /// Creates a `Semaphore` bound to a specific `WaitKey`.
    pub fn with_key(permits: usize, key: WaitKey) -> Self {
        Self {
            permits,
            key,
            waiters: Vec::new(),
        }
    }

    /// Returns the associated `WaitKey`.
    pub fn key(&self) -> WaitKey {
        self.key
    }

    /// Number of currently available permits.
    pub fn available_permits(&self) -> usize {
        self.permits
    }

    /// Add permits directly without waking waiters.
    pub fn add_permits(&mut self, n: usize) {
        self.permits = self.permits.saturating_add(n);
    }

    /// Number of currently queued waiters.
    pub fn waiters_count(&self) -> usize {
        self.waiters.len()
    }

    /// Attempts to immediately acquire `n` permits without parking.
    /// Returns `true` if acquired, `false` if insufficient permits are available.
    pub fn try_acquire(&mut self, n: usize) -> bool {
        if n == 0 {
            return true;
        }
        if n <= self.permits && self.waiters.is_empty() {
            self.permits -= n;
            true
        } else {
            false
        }
    }

    /// Acquires `n` permits immediately or parks the calling task in the `TaskManager`.
    ///
    /// Preserves strict FIFO order: if earlier waiters are queued, cannot jump ahead.
    pub fn acquire_or_park_tm(&mut self, id: TaskId, n: usize, tm: &mut TaskManager) -> bool {
        self.cleanup_dead_waiters_tm(tm);

        // Can acquire only if no preceding waiters exist and permits suffice
        if self.waiters.is_empty() && n <= self.permits {
            self.permits -= n;
            return true;
        }

        self.waiters.push((id, n));
        tm.park(id, self.key);

        // Race-safe re-check after parking:
        if self.permits >= n && self.waiters.first().is_some_and(|(front_id, _)| *front_id == id) {
            self.waiters.remove(0);
            self.permits -= n;
            tm.wake(self.key, false);
            return true;
        }

        false
    }

    /// Acquires `n` permits immediately or parks the calling task in the `VM`.
    pub fn acquire_or_park_vm(&mut self, id: TaskId, n: usize, vm: &mut VM) -> bool {
        self.acquire_or_park_tm(id, n, &mut vm.task_manager)
    }

    /// Releases `n` permits back to the semaphore and wakes queued waiters in FIFO order.
    ///
    /// Skips any cancelled or stopped waiters WITHOUT consuming permits.
    /// Returns the list of `TaskId`s that were successfully woken.
    pub fn release_tm(&mut self, n: usize, tm: &mut TaskManager) -> Vec<TaskId> {
        self.permits = self.permits.saturating_add(n);
        let mut woken = Vec::new();

        while let Some(&(id, need)) = self.waiters.first() {
            let is_alive = tm.get(id).is_some_and(|t| {
                matches!(t.state, TaskState::Yielded(YieldReason::Parked(k)) if k == self.key)
                    && t.fiber_context.state != FiberExecutionState::Interrupted
            });

            if !is_alive {
                // Cancelled or dead waiter: remove without deducting permits!
                self.waiters.remove(0);
                tm.unregister_task_from_waiters(id);
                continue;
            }

            if need <= self.permits {
                self.waiters.remove(0);
                self.permits -= need;
                if tm.wake(self.key, false).contains(&id) {
                    woken.push(id);
                }
            } else {
                // Head-of-line: next waiter needs more permits than available. Stop.
                break;
            }
        }

        woken
    }

    /// Releases `n` permits back to the semaphore in the `VM`.
    pub fn release_vm(&mut self, n: usize, vm: &mut VM) -> Vec<TaskId> {
        self.release_tm(n, &mut vm.task_manager)
    }

    /// Cancels a specific waiter task from the semaphore.
    pub fn cancel_waiter(&mut self, id: TaskId) {
        self.waiters.retain(|(x, _)| *x != id);
    }

    /// Cancels a waiter and unregisters it from the `TaskManager`.
    pub fn cancel_waiter_tm(&mut self, id: TaskId, tm: &mut TaskManager) {
        self.cancel_waiter(id);
        tm.unregister_task_from_waiters(id);
    }

    /// Cancels a waiter and unregisters it from the `VM`.
    pub fn cancel_waiter_vm(&mut self, id: TaskId, vm: &mut VM) {
        self.cancel_waiter_tm(id, &mut vm.task_manager);
    }

    /// Scoped RAII execution with `n` permits.
    ///
    /// Acquires `n` permits immediately. If unavailable, returns `Err(SemaphoreError::Unavailable)`.
    /// On success, executes `f()` and ensures permits are 100% returned upon drop, even on error or panic.
    pub fn with_permits<R, F: FnOnce() -> R>(&mut self, n: usize, f: F) -> Result<R, SemaphoreError> {
        if n == 0 {
            return Ok(f());
        }
        if !self.try_acquire(n) {
            return Err(SemaphoreError::Unavailable);
        }

        let guard = core::mem::DropGuard::new(self, move |sem| {
            // Direct permit return without TM (since no TM is passed in pure Rust guard)
            sem.permits = sem.permits.saturating_add(n);
        });
        let res = f();
        drop(guard);
        Ok(res)
    }

    /// Cleans up any dead, interrupted, or cancelled waiters from the queue.
    fn cleanup_dead_waiters_tm(&mut self, tm: &TaskManager) {
        let key = self.key;
        self.waiters.retain(|&(id, _)| {
            tm.get(id).is_some_and(|t| {
                matches!(t.state, TaskState::Yielded(YieldReason::Parked(k)) if k == key)
                    && t.fiber_context.state != FiberExecutionState::Interrupted
            })
        });
    }
}

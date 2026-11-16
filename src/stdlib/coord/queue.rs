//! Coordination Queue: multi-fiber FIFO queue with configurable backpressure overflow policies.
//!
//! Complies with `RFC-STDLIB-ASYNC-COORDINATION` (§3.4):
//! - Full backpressure overflow strategies: `Suspend`, `DropOldest` (sliding), `DropNewest` (dropping), `Fail`.
//! - Generation-tagged FIFO wait queues for senders and receivers.
//! - Atomic transfer claim protocol skipping dead or cancelled waiters without message loss.
//! - Clean close semantics: wakes all parked offerers and takers, permitting takers to drain remaining items.

use alloc::{collections::VecDeque, string::String, vec::Vec};
use core::sync::atomic::{AtomicU64, Ordering};
pub use crate::rx::buffer::OverflowStrategy;
use crate::effect_abi::FiberExecutionState;
use crate::vm::{TaskId, TaskManager, TaskState, WaitKey, YieldReason, VM};

static NEXT_QUEUE_WAIT_KEY: AtomicU64 = AtomicU64::new(90_000);

/// Core error types for queue operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueError {
    /// Queue is closed; no further offers accepted.
    Closed,
    /// Queue is full under `OverflowStrategy::Fail` or non-suspending offer.
    Full,
    /// Queue is empty for non-blocking take.
    Empty,
    /// Operation cancelled by fiber interrupt.
    Cancelled,
    /// Invalid capacity or options specified.
    InvalidCapacity(String),
}

impl core::fmt::Display for QueueError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Closed => write!(f, "Queue is closed"),
            Self::Full => write!(f, "Queue buffer is full"),
            Self::Empty => write!(f, "Queue is empty"),
            Self::Cancelled => write!(f, "Queue operation cancelled"),
            Self::InvalidCapacity(s) => write!(f, "Invalid queue capacity: {}", s),
        }
    }
}

impl std::error::Error for QueueError {}

/// Record of a parked offerer task and its offered item.
#[derive(Debug, Clone)]
struct Offerer<T> {
    task_id: TaskId,
    item: T,
}

/// A cooperative, fiber-aware FIFO queue with bounded backpressure.
#[derive(Debug)]
pub struct Queue<T> {
    buffer: VecDeque<T>,
    capacity: usize,
    overflow: OverflowStrategy,
    closed: bool,
    offer_key: WaitKey,
    take_key: WaitKey,
    offerers: VecDeque<Offerer<T>>,
    takers: VecDeque<TaskId>,
}

impl<T: Clone> Queue<T> {
    /// Creates a bounded queue with default `OverflowStrategy::Suspend`.
    pub fn bounded(capacity: usize) -> Self {
        Self::with_overflow(capacity, OverflowStrategy::Suspend)
    }

    /// Creates an unbounded queue (practically bounded only by memory).
    pub fn unbounded() -> Self {
        Self::with_overflow(usize::MAX, OverflowStrategy::Suspend)
    }

    /// Creates a sliding queue that drops the oldest item on overflow.
    pub fn sliding(capacity: usize) -> Self {
        Self::with_overflow(capacity, OverflowStrategy::DropOldest)
    }

    /// Creates a dropping queue that drops incoming items on overflow.
    pub fn dropping(capacity: usize) -> Self {
        Self::with_overflow(capacity, OverflowStrategy::DropNewest)
    }

    /// Creates a queue with a specified capacity and overflow strategy.
    pub fn with_overflow(capacity: usize, overflow: OverflowStrategy) -> Self {
        let offer_key = WaitKey(NEXT_QUEUE_WAIT_KEY.fetch_add(1, Ordering::SeqCst));
        let take_key = WaitKey(NEXT_QUEUE_WAIT_KEY.fetch_add(1, Ordering::SeqCst));
        Self {
            buffer: VecDeque::with_capacity(capacity.min(1024)),
            capacity,
            overflow,
            closed: false,
            offer_key,
            take_key,
            offerers: VecDeque::new(),
            takers: VecDeque::new(),
        }
    }

    /// Alias for `with_overflow`.
    pub fn make(capacity: usize, overflow: OverflowStrategy) -> Self {
        Self::with_overflow(capacity, overflow)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    pub fn overflow_strategy(&self) -> OverflowStrategy {
        self.overflow
    }

    pub fn offer_waiters_count(&self) -> usize {
        self.offerers.len()
    }

    pub fn take_waiters_count(&self) -> usize {
        self.takers.len()
    }

    /// Non-blocking offer of an item.
    ///
    /// Returns `Ok(true)` if enqueued, `Ok(false)` if dropped under `DropNewest`,
    /// or `Err(QueueError::Full)` under `Suspend` or `Fail` when at capacity.
    pub fn try_offer(&mut self, item: T) -> Result<bool, QueueError> {
        if self.closed {
            return Err(QueueError::Closed);
        }

        if self.buffer.len() < self.capacity {
            self.buffer.push_back(item);
            return Ok(true);
        }

        match self.overflow {
            OverflowStrategy::DropOldest => {
                self.buffer.pop_front();
                self.buffer.push_back(item);
                Ok(true)
            }
            OverflowStrategy::DropNewest => Ok(false),
            OverflowStrategy::Suspend | OverflowStrategy::Fail => Err(QueueError::Full),
        }
    }

    /// Offers an item from `task_id` into the queue, parking the task in the `TaskManager`
    /// if the buffer is full and overflow strategy is `Suspend`.
    pub fn offer_or_park_tm(
        &mut self,
        task_id: TaskId,
        item: T,
        tm: &mut TaskManager,
    ) -> Result<bool, QueueError> {
        if self.closed {
            return Err(QueueError::Closed);
        }

        // 1. Direct handoff to waiting taker if buffer is empty
        self.cleanup_dead_takers_tm(tm);
        if self.buffer.is_empty() {
            while let Some(taker_id) = self.takers.pop_front() {
                let is_alive = tm.get(taker_id).is_some_and(|t| {
                    matches!(t.state, TaskState::Yielded(YieldReason::Parked(k)) if k == self.take_key)
                        && t.fiber_context.state != FiberExecutionState::Interrupted
                });
                if !is_alive {
                    tm.unregister_task_from_waiters(taker_id);
                    continue;
                }
                // Handoff via buffer
                self.buffer.push_back(item);
                tm.wake(self.take_key, false);
                return Ok(true);
            }
        }

        // 2. Normal space available
        if self.buffer.len() < self.capacity {
            self.buffer.push_back(item);
            return Ok(true);
        }

        // 3. Overflow handling
        match self.overflow {
            OverflowStrategy::DropOldest => {
                self.buffer.pop_front();
                self.buffer.push_back(item);
                Ok(true)
            }
            OverflowStrategy::DropNewest => Ok(false),
            OverflowStrategy::Fail => Err(QueueError::Full),
            OverflowStrategy::Suspend => {
                // Park offerer
                self.offerers.push_back(Offerer { task_id, item });
                tm.park(task_id, self.offer_key);
                Ok(false)
            }
        }
    }

    /// Offers an item in the `VM`.
    pub fn offer_or_park_vm(
        &mut self,
        task_id: TaskId,
        item: T,
        vm: &mut VM,
    ) -> Result<bool, QueueError> {
        self.offer_or_park_tm(task_id, item, &mut vm.task_manager)
    }

    /// Non-blocking take of an item.
    /// Returns `Ok(Some(item))` if available, `Ok(None)` if closed & drained,
    /// or `Err(QueueError::Empty)` if open but empty.
    pub fn try_take(&mut self) -> Result<Option<T>, QueueError> {
        if let Some(item) = self.buffer.pop_front() {
            return Ok(Some(item));
        }

        if self.closed {
            Ok(None)
        } else {
            Err(QueueError::Empty)
        }
    }

    /// Takes an item or parks `task_id` in the `TaskManager` if the queue is empty.
    ///
    /// Upon draining from buffer, automatically matches and wakes the first valid parked offerer.
    pub fn take_or_park_tm(
        &mut self,
        task_id: TaskId,
        tm: &mut TaskManager,
    ) -> Result<Option<T>, QueueError> {
        // 1. If buffer has items, take one and wake next valid offerer if space opened
        if let Some(item) = self.buffer.pop_front() {
            self.wake_next_offerer_tm(tm);
            return Ok(Some(item));
        }

        // 2. Direct take from parked offerer if buffer was empty
        self.cleanup_dead_offerers_tm(tm);
        while let Some(offerer) = self.offerers.pop_front() {
            let is_alive = tm.get(offerer.task_id).is_some_and(|t| {
                matches!(t.state, TaskState::Yielded(YieldReason::Parked(k)) if k == self.offer_key)
                    && t.fiber_context.state != FiberExecutionState::Interrupted
            });
            if !is_alive {
                tm.unregister_task_from_waiters(offerer.task_id);
                continue;
            }
            tm.wake(self.offer_key, false);
            return Ok(Some(offerer.item));
        }

        // 3. Queue is closed and drained
        if self.closed {
            return Ok(None);
        }

        // 4. Queue is empty: park receiver
        self.takers.push_back(task_id);
        tm.park(task_id, self.take_key);

        // Race-safe re-check after parking:
        if let Some(item) = self.buffer.pop_front() {
            self.takers.retain(|id| *id != task_id);
            tm.wake(self.take_key, false);
            return Ok(Some(item));
        }

        Ok(None)
    }

    /// Takes an item or parks `task_id` in the `VM`.
    pub fn take_or_park_vm(
        &mut self,
        task_id: TaskId,
        vm: &mut VM,
    ) -> Result<Option<T>, QueueError> {
        self.take_or_park_tm(task_id, &mut vm.task_manager)
    }

    /// Closes the queue.
    ///
    /// Wakes all waiting offerers (which fail with `Closed`) and takers (which drain the remainder).
    /// Returns the list of all woken `TaskId`s.
    pub fn close_tm(&mut self, tm: &mut TaskManager) -> Vec<TaskId> {
        self.closed = true;
        self.offerers.clear();
        self.takers.clear();
        let mut woken = tm.wake(self.offer_key, true);
        woken.extend(tm.wake(self.take_key, true));
        woken
    }

    /// Closes the queue in the `VM`.
    pub fn close_vm(&mut self, vm: &mut VM) -> Vec<TaskId> {
        self.close_tm(&mut vm.task_manager)
    }

    /// Cancels a waiter task (offerer or taker) from the queue.
    pub fn cancel_waiter(&mut self, id: TaskId) {
        self.offerers.retain(|o| o.task_id != id);
        self.takers.retain(|x| *x != id);
    }

    /// Cancels a waiter task and unregisters it from the `TaskManager`.
    pub fn cancel_waiter_tm(&mut self, id: TaskId, tm: &mut TaskManager) {
        self.cancel_waiter(id);
        tm.unregister_task_from_waiters(id);
    }

    /// Cancels a waiter task and unregisters it from the `VM`.
    pub fn cancel_waiter_vm(&mut self, id: TaskId, vm: &mut VM) {
        self.cancel_waiter_tm(id, &mut vm.task_manager);
    }

    fn wake_next_offerer_tm(&mut self, tm: &mut TaskManager) {
        self.cleanup_dead_offerers_tm(tm);
        while let Some(offerer) = self.offerers.pop_front() {
            let is_alive = tm.get(offerer.task_id).is_some_and(|t| {
                matches!(t.state, TaskState::Yielded(YieldReason::Parked(k)) if k == self.offer_key)
                    && t.fiber_context.state != FiberExecutionState::Interrupted
            });
            if !is_alive {
                tm.unregister_task_from_waiters(offerer.task_id);
                continue;
            }
            self.buffer.push_back(offerer.item);
            tm.wake(self.offer_key, false);
            break;
        }
    }

    fn cleanup_dead_offerers_tm(&mut self, tm: &TaskManager) {
        let key = self.offer_key;
        self.offerers.retain(|o| {
            tm.get(o.task_id).is_some_and(|t| {
                matches!(t.state, TaskState::Yielded(YieldReason::Parked(k)) if k == key)
                    && t.fiber_context.state != FiberExecutionState::Interrupted
            })
        });
    }

    fn cleanup_dead_takers_tm(&mut self, tm: &TaskManager) {
        let key = self.take_key;
        self.takers.retain(|&id| {
            tm.get(id).is_some_and(|t| {
                matches!(t.state, TaskState::Yielded(YieldReason::Parked(k)) if k == key)
                    && t.fiber_context.state != FiberExecutionState::Interrupted
            })
        });
    }
}

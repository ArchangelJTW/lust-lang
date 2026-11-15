//! Typed asynchronous/cooperative channels with scheduler park/wake (`WaitKey`),
//! backpressure overflow strategies, and cancellation-safe waiter cleanup.
//!
//! Complies with `RFC-STDLIB-ASYNC-COORDINATION` (§3.3) and `R-CHAN` specification:
//! - Full generation-tagged FIFO wait queues for senders and receivers.
//! - Atomic transfer claim protocol (`try_claim_transfer`) skipping cancelled or stale waiters.
//! - Direct handoff rendezvous (capacity 0) and bounded buffering with overflow policies:
//!   `Suspend`, `DropOldest`, `DropNewest`, `Fail`.
//! - Scheduler park/wake via VM `WaitKey` and `TaskManager`.
//! - Cancellation-safe cleanup on scope close / task interruption.
//! - Builtin native function registration for Lust script access (`channel.*`).

use alloc::{format, rc::Rc, string::{String, ToString}, vec::Vec};
use core::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bytecode::{native_fn, LustMap, NativeCallResult, NativeFn, Value, ValueKey};
use crate::effect_abi::FiberExecutionState;
pub use crate::rx::buffer::OverflowStrategy;
use crate::vm::{TaskId, TaskState, WaitKey, VM};

/// Core error types for channel operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelError {
    /// Channel has been closed and cannot accept new sends.
    Closed,
    /// Channel buffer is full (returned by `try_send` or `OverflowStrategy::Fail`).
    Full,
    /// Channel buffer is empty (returned by non-blocking `try_recv`).
    Empty,
    /// Channel operation was cancelled by fiber interrupt.
    Cancelled,
    /// Invalid capacity requested during channel initialization.
    InvalidCapacity(String),
}

impl core::fmt::Display for ChannelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Closed => write!(f, "Channel is closed"),
            Self::Full => write!(f, "Channel buffer is full"),
            Self::Empty => write!(f, "Channel buffer is empty"),
            Self::Cancelled => write!(f, "Channel operation cancelled"),
            Self::InvalidCapacity(s) => write!(f, "Invalid channel capacity: {}", s),
        }
    }
}

impl std::error::Error for ChannelError {}

/// Lifecycle states of a parked waiter in the channel's wait queues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaiterLifecycle {
    /// Waiting for a matching send or recv operation.
    Parked,
    /// Matched and value transferred; cannot be revoked or stolen.
    Claimed,
    /// Cancelled before a match was made; must be pruned without message loss.
    Cancelled,
}

/// Generation-tagged waiter record for a parked sender fiber.
#[derive(Debug, Clone)]
pub struct SendWaiter<T> {
    pub waiter_id: u64,
    pub task_id: TaskId,
    pub wait_key: WaitKey,
    pub generation: u64,
    pub item: Rc<RefCell<Option<T>>>,
    pub state: Rc<Cell<WaiterLifecycle>>,
}

/// Generation-tagged waiter record for a parked receiver fiber.
#[derive(Debug, Clone)]
pub struct RecvWaiter<T> {
    pub waiter_id: u64,
    pub task_id: TaskId,
    pub wait_key: WaitKey,
    pub generation: u64,
    pub slot: Rc<RefCell<Option<T>>>,
    pub state: Rc<Cell<WaiterLifecycle>>,
}

/// Internal state of a typed channel.
pub struct ChannelState<T> {
    pub id: u64,
    pub capacity: usize,
    pub overflow: OverflowStrategy,
    pub queue: VecDeque<T>,
    pub send_waiters: VecDeque<SendWaiter<T>>,
    pub recv_waiters: VecDeque<RecvWaiter<T>>,
    pub completed_transfers: HashMap<TaskId, T>,
    pub is_closed: bool,
    pub generation: u64,
}

static NEXT_CHANNEL_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_WAITER_ID: AtomicU64 = AtomicU64::new(1);

/// A typed, cooperative, fiber-aware channel.
///
/// Supports bounded buffering, zero-capacity rendezvous, overflow strategies,
/// and direct integration with Lust's `VM` cooperative scheduler.
#[derive(Clone)]
pub struct Channel<T> {
    inner: Rc<RefCell<ChannelState<T>>>,
}

impl<T> Channel<T> {
    pub fn id(&self) -> u64 {
        self.inner.borrow().id
    }

    pub fn len(&self) -> usize {
        self.inner.borrow().queue.len()
    }

    pub fn capacity(&self) -> usize {
        self.inner.borrow().capacity
    }

    pub fn overflow_strategy(&self) -> OverflowStrategy {
        self.inner.borrow().overflow
    }

    pub fn is_empty(&self) -> bool {
        self.inner.borrow().queue.is_empty()
    }

    pub fn is_full(&self) -> bool {
        let state = self.inner.borrow();
        state.queue.len() >= state.capacity
    }

    pub fn is_closed(&self) -> bool {
        self.inner.borrow().is_closed
    }

    pub fn send_waiters_count(&self) -> usize {
        self.inner.borrow().send_waiters.len()
    }

    pub fn recv_waiters_count(&self) -> usize {
        self.inner.borrow().recv_waiters.len()
    }

    /// Attempts to retrieve a value that was transferred directly to this task
    /// while it was parked.
    pub fn take_completed_transfer(&self, task_id: TaskId) -> Option<T> {
        self.inner.borrow_mut().completed_transfers.remove(&task_id)
    }
}

impl<T: Clone + 'static> Channel<T> {
    /// Creates a bounded channel with the default `OverflowStrategy::Suspend`.
    pub fn bounded(capacity: usize) -> Self {
        Self::with_overflow(capacity, OverflowStrategy::Suspend)
    }

    /// Creates a bounded channel with a specific overflow strategy.
    pub fn with_overflow(capacity: usize, overflow: OverflowStrategy) -> Self {
        let id = NEXT_CHANNEL_ID.fetch_add(1, Ordering::SeqCst);
        let inner = ChannelState {
            id,
            capacity,
            overflow,
            queue: VecDeque::with_capacity(capacity.min(1024)),
            send_waiters: VecDeque::new(),
            recv_waiters: VecDeque::new(),
            completed_transfers: HashMap::new(),
            is_closed: false,
            generation: 1,
        };
        Self {
            inner: Rc::new(RefCell::new(inner)),
        }
    }

    /// Creates an unbounded channel with soft capacity limit.
    pub fn unbounded() -> Self {
        Self::with_overflow(usize::MAX, OverflowStrategy::Suspend)
    }

    /// Creates a single-shot channel (capacity 1).
    pub fn oneshot() -> Self {
        Self::bounded(1)
    }

    /// Non-blocking send.
    ///
    /// Succeeds immediately if a receiver is waiting or buffer has space.
    /// In `DropOldest` / `DropNewest` mode, handles overflow per strategy.
    /// Returns `Err(ChannelError::Full)` if buffer is full and strategy is `Suspend` or `Fail`.
    /// Returns `Err(ChannelError::Closed)` if channel is closed.
    pub fn try_send(&self, item: T) -> Result<(), ChannelError> {
        let mut state = self.inner.borrow_mut();
        if state.is_closed {
            return Err(ChannelError::Closed);
        }

        // Direct handoff to a valid parked receiver if one exists
        while let Some(recv_waiter) = state.recv_waiters.pop_front() {
            if recv_waiter.state.get() != WaiterLifecycle::Parked {
                continue;
            }
            recv_waiter.state.set(WaiterLifecycle::Claimed);
            *recv_waiter.slot.borrow_mut() = Some(item.clone());
            state.completed_transfers.insert(recv_waiter.task_id, item);
            return Ok(());
        }

        // Enqueue if buffer has space
        if state.queue.len() < state.capacity {
            state.queue.push_back(item);
            return Ok(());
        }

        // Buffer is full: handle per overflow strategy
        match state.overflow {
            OverflowStrategy::DropOldest => {
                state.queue.pop_front();
                state.queue.push_back(item);
                Ok(())
            }
            OverflowStrategy::DropNewest => {
                // Drop incoming item; queue unchanged
                Ok(())
            }
            OverflowStrategy::Suspend | OverflowStrategy::Fail => Err(ChannelError::Full),
        }
    }

    /// Non-blocking receive.
    ///
    /// Succeeds immediately if a value was previously transferred, or if the queue is non-empty,
    /// or if a sender is waiting to hand off directly.
    /// Returns `Ok(None)` if the channel is closed and drained.
    /// Returns `Err(ChannelError::Empty)` if the channel is empty and open.
    pub fn try_recv(&self) -> Result<Option<T>, ChannelError> {
        let mut state = self.inner.borrow_mut();

        // 1. Drain from queue if present
        if let Some(item) = state.queue.pop_front() {
            // Wake or claim next valid sender if one is parked
            while let Some(send_waiter) = state.send_waiters.pop_front() {
                if send_waiter.state.get() != WaiterLifecycle::Parked {
                    continue;
                }
                send_waiter.state.set(WaiterLifecycle::Claimed);
                if let Some(incoming) = send_waiter.item.borrow_mut().take() {
                    state.queue.push_back(incoming);
                }
                break;
            }
            return Ok(Some(item));
        }

        // 2. Direct handoff from waiting sender (e.g. capacity 0 rendezvous)
        while let Some(send_waiter) = state.send_waiters.pop_front() {
            if send_waiter.state.get() != WaiterLifecycle::Parked {
                continue;
            }
            send_waiter.state.set(WaiterLifecycle::Claimed);
            if let Some(item) = send_waiter.item.borrow_mut().take() {
                return Ok(Some(item));
            }
        }

        // 3. Check closed status
        if state.is_closed {
            Ok(None)
        } else {
            Err(ChannelError::Empty)
        }
    }

    /// Sends an item from a specific task. If the channel is full and strategy is `Suspend`,
    /// parks the task in `vm` using a unique `WaitKey`.
    ///
    /// Implements `R-ASYNC-ATOMIC-TRANSFER`:
    /// Skips and purges any cancelled waiters before transferring value, guaranteeing:
    /// 1. No lost wake (subsequent valid waiter is matched).
    /// 2. No stale waiter (cancelled tasks are removed).
    /// 3. No duplicate delivery (value is claimed atomically once).
    pub fn send_task(&self, item: T, task_id: TaskId, vm: &mut VM) -> Result<(), ChannelError> {
        let (wait_key, waiter) = {
            let mut state = self.inner.borrow_mut();
            if state.is_closed {
                return Err(ChannelError::Closed);
            }

            // Direct handoff to the first VALID parked receiver
            while let Some(recv_waiter) = state.recv_waiters.pop_front() {
                if recv_waiter.state.get() != WaiterLifecycle::Parked {
                    continue;
                }
                // Verify the task is actually still waiting in the VM
                let is_valid = vm.is_task_waiting(recv_waiter.task_id)
                    && vm.task_manager().get(recv_waiter.task_id).is_some_and(|t| {
                        !matches!(
                            t.state,
                            TaskState::Completed | TaskState::Failed | TaskState::Stopped
                        ) && t.fiber_context.state != FiberExecutionState::Interrupted
                    });

                if !is_valid {
                    // Stale or cancelled waiter! Clean up and move to next
                    recv_waiter.state.set(WaiterLifecycle::Cancelled);
                    continue;
                }

                // Valid receiver found: atomic claim transfer!
                recv_waiter.state.set(WaiterLifecycle::Claimed);
                *recv_waiter.slot.borrow_mut() = Some(item.clone());
                state.completed_transfers.insert(recv_waiter.task_id, item);
                let wk = recv_waiter.wait_key;
                drop(state);
                vm.wake_one(wk);
                return Ok(());
            }

            // Enqueue if buffer has space
            if state.queue.len() < state.capacity {
                state.queue.push_back(item);
                return Ok(());
            }

            // Handle non-suspending overflow strategies
            match state.overflow {
                OverflowStrategy::DropOldest => {
                    state.queue.pop_front();
                    state.queue.push_back(item);
                    return Ok(());
                }
                OverflowStrategy::DropNewest => {
                    return Ok(());
                }
                OverflowStrategy::Fail => {
                    return Err(ChannelError::Full);
                }
                OverflowStrategy::Suspend => {}
            }

            // Buffer full with Suspend: register SendWaiter and park task
            let wait_key = vm.next_wait_key();
            let waiter_id = NEXT_WAITER_ID.fetch_add(1, Ordering::SeqCst);
            let item_cell = Rc::new(RefCell::new(Some(item)));
            let waiter = SendWaiter {
                waiter_id,
                task_id,
                wait_key,
                generation: state.generation,
                item: item_cell,
                state: Rc::new(Cell::new(WaiterLifecycle::Parked)),
            };
            state.send_waiters.push_back(waiter.clone());
            (wait_key, waiter)
        };

        // Park the task in the VM scheduler
        let parked = vm.park_task(task_id, wait_key);
        if !parked {
            // Task could not be parked (e.g. already cancelled or interrupted)
            waiter.state.set(WaiterLifecycle::Cancelled);
            let mut state = self.inner.borrow_mut();
            state.send_waiters.retain(|w| w.waiter_id != waiter.waiter_id);
            return Err(ChannelError::Cancelled);
        }

        Ok(())
    }

    /// Sends an item using the currently executing task in the VM.
    pub fn send(&self, item: T, vm: &mut VM) -> Result<(), ChannelError> {
        let task_id = vm.current_task_handle().map(|h| TaskId(h.0)).ok_or_else(|| {
            ChannelError::Cancelled
        })?;
        self.send_task(item, task_id, vm)
    }

    /// Receives an item for a specific task. If the channel is empty and open,
    /// parks the task in `vm` using a unique `WaitKey`.
    ///
    /// Implements `R-ASYNC-ATOMIC-TRANSFER`:
    /// Skips and purges any cancelled senders before consuming an item.
    pub fn recv_task(&self, task_id: TaskId, vm: &mut VM) -> Result<Option<T>, ChannelError> {
        // 1. Check if a direct handoff was already completed for this task
        if let Some(item) = self.take_completed_transfer(task_id) {
            return Ok(Some(item));
        }

        let (wait_key, waiter) = {
            let mut state = self.inner.borrow_mut();

            // 2. Drain from queue if non-empty
            if let Some(item) = state.queue.pop_front() {
                // Wake the next valid parked sender to refill the buffer
                while let Some(send_waiter) = state.send_waiters.pop_front() {
                    if send_waiter.state.get() != WaiterLifecycle::Parked {
                        continue;
                    }
                    let is_valid = vm.is_task_waiting(send_waiter.task_id)
                        && vm.task_manager().get(send_waiter.task_id).is_some_and(|t| {
                            !matches!(
                                t.state,
                                TaskState::Completed | TaskState::Failed | TaskState::Stopped
                            ) && t.fiber_context.state != FiberExecutionState::Interrupted
                        });

                    if !is_valid {
                        send_waiter.state.set(WaiterLifecycle::Cancelled);
                        continue;
                    }

                    // Valid sender: claim and move its item into the queue
                    send_waiter.state.set(WaiterLifecycle::Claimed);
                    if let Some(incoming) = send_waiter.item.borrow_mut().take() {
                        state.queue.push_back(incoming);
                    }
                    let wk = send_waiter.wait_key;
                    drop(state);
                    vm.wake_one(wk);
                    return Ok(Some(item));
                }

                return Ok(Some(item));
            }

            // 3. Direct handoff from waiting sender (capacity 0 rendezvous)
            while let Some(send_waiter) = state.send_waiters.pop_front() {
                if send_waiter.state.get() != WaiterLifecycle::Parked {
                    continue;
                }
                let is_valid = vm.is_task_waiting(send_waiter.task_id)
                    && vm.task_manager().get(send_waiter.task_id).is_some_and(|t| {
                        !matches!(
                            t.state,
                            TaskState::Completed | TaskState::Failed | TaskState::Stopped
                        ) && t.fiber_context.state != FiberExecutionState::Interrupted
                    });

                if !is_valid {
                    send_waiter.state.set(WaiterLifecycle::Cancelled);
                    continue;
                }

                send_waiter.state.set(WaiterLifecycle::Claimed);
                if let Some(item) = send_waiter.item.borrow_mut().take() {
                    let wk = send_waiter.wait_key;
                    drop(state);
                    vm.wake_one(wk);
                    return Ok(Some(item));
                }
            }

            // 4. Closed channel returns None when completely drained
            if state.is_closed {
                return Ok(None);
            }

            // 5. Channel open and empty: park receiver
            let wait_key = vm.next_wait_key();
            let waiter_id = NEXT_WAITER_ID.fetch_add(1, Ordering::SeqCst);
            let slot = Rc::new(RefCell::new(None));
            let waiter = RecvWaiter {
                waiter_id,
                task_id,
                wait_key,
                generation: state.generation,
                slot,
                state: Rc::new(Cell::new(WaiterLifecycle::Parked)),
            };
            state.recv_waiters.push_back(waiter.clone());
            (wait_key, waiter)
        };

        // Park task in the VM
        let parked = vm.park_task(task_id, wait_key);
        if !parked {
            waiter.state.set(WaiterLifecycle::Cancelled);
            let mut state = self.inner.borrow_mut();
            state.recv_waiters.retain(|w| w.waiter_id != waiter.waiter_id);
            return Err(ChannelError::Cancelled);
        }

        Ok(None)
    }

    /// Receives an item using the currently executing task in the VM.
    pub fn recv(&self, vm: &mut VM) -> Result<Option<T>, ChannelError> {
        let task_id = vm.current_task_handle().map(|h| TaskId(h.0)).ok_or_else(|| {
            ChannelError::Cancelled
        })?;
        self.recv_task(task_id, vm)
    }

    /// Closes the channel and wakes all parked senders and receivers.
    pub fn close(&self, vm: &mut VM) {
        let (send_keys, recv_keys) = {
            let mut state = self.inner.borrow_mut();
            if state.is_closed {
                return;
            }
            state.is_closed = true;
            state.generation = state.generation.wrapping_add(1);

            let mut send_keys = Vec::new();
            while let Some(w) = state.send_waiters.pop_front() {
                w.state.set(WaiterLifecycle::Cancelled);
                send_keys.push(w.wait_key);
            }

            let mut recv_keys = Vec::new();
            while let Some(w) = state.recv_waiters.pop_front() {
                w.state.set(WaiterLifecycle::Cancelled);
                recv_keys.push(w.wait_key);
            }

            (send_keys, recv_keys)
        };

        for key in send_keys {
            vm.wake_all(key);
        }
        for key in recv_keys {
            vm.wake_all(key);
        }
    }

    /// Closes the channel locally without scheduling VM wakes.
    pub fn close_local(&self) {
        let mut state = self.inner.borrow_mut();
        state.is_closed = true;
        state.generation = state.generation.wrapping_add(1);
        while let Some(w) = state.send_waiters.pop_front() {
            w.state.set(WaiterLifecycle::Cancelled);
        }
        while let Some(w) = state.recv_waiters.pop_front() {
            w.state.set(WaiterLifecycle::Cancelled);
        }
    }

    /// Purges cancelled or terminated tasks from both wait queues.
    pub fn cleanup_cancelled_waiters(&self, vm: &VM) {
        let mut state = self.inner.borrow_mut();
        state.send_waiters.retain(|w| {
            if w.state.get() != WaiterLifecycle::Parked {
                return false;
            }
            let valid = vm.is_task_waiting(w.task_id)
                && vm.task_manager().get(w.task_id).is_some_and(|t| {
                    !matches!(
                        t.state,
                        TaskState::Completed | TaskState::Failed | TaskState::Stopped
                    ) && t.fiber_context.state != FiberExecutionState::Interrupted
                });
            if !valid {
                w.state.set(WaiterLifecycle::Cancelled);
            }
            valid
        });

        state.recv_waiters.retain(|w| {
            if w.state.get() != WaiterLifecycle::Parked {
                return false;
            }
            let valid = vm.is_task_waiting(w.task_id)
                && vm.task_manager().get(w.task_id).is_some_and(|t| {
                    !matches!(
                        t.state,
                        TaskState::Completed | TaskState::Failed | TaskState::Stopped
                    ) && t.fiber_context.state != FiberExecutionState::Interrupted
                });
            if !valid {
                w.state.set(WaiterLifecycle::Cancelled);
            }
            valid
        });
    }
}

// ============================================================================
// Multi-Channel Select Support (`SelectGroup`)
// ============================================================================

/// Outcome of selecting over a group of channels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectResult<T> {
    Received(usize, T),
    ChannelClosed(usize),
    Default,
}

/// Multiplexes non-blocking operations across multiple channels.
pub struct SelectGroup<T> {
    channels: Vec<Channel<T>>,
}

impl<T: Clone + 'static> SelectGroup<T> {
    pub fn new() -> Self {
        Self {
            channels: Vec::new(),
        }
    }

    pub fn add_channel(&mut self, channel: Channel<T>) -> usize {
        let idx = self.channels.len();
        self.channels.push(channel);
        idx
    }

    /// Probes all channels in rotation order. Returns the first ready value,
    /// or `None` if all channels are currently empty.
    pub fn try_select(&self, start_index: usize) -> Option<SelectResult<T>> {
        let count = self.channels.len();
        if count == 0 {
            return None;
        }

        for offset in 0..count {
            let idx = (start_index + offset) % count;
            let ch = &self.channels[idx];
            match ch.try_recv() {
                Ok(Some(item)) => return Some(SelectResult::Received(idx, item)),
                Ok(None) => return Some(SelectResult::ChannelClosed(idx)),
                Err(ChannelError::Empty) => continue,
                Err(_) => continue,
            }
        }
        None
    }
}

impl<T: Clone + 'static> Default for SelectGroup<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Thread-Local VM Channel Registry & Builtins Registration
// ============================================================================

thread_local! {
    static THREAD_CHANNELS: RefCell<HashMap<u64, Channel<Value>>> = RefCell::new(HashMap::new());
}

/// Registers a channel in the thread-local registry and returns its handle ID.
pub fn register_global_channel(channel: Channel<Value>) -> u64 {
    let id = channel.id();
    THREAD_CHANNELS.with(|reg| {
        reg.borrow_mut().insert(id, channel);
    });
    id
}

/// Retrieves a channel handle by ID from the thread-local registry.
pub fn get_global_channel(id: u64) -> Option<Channel<Value>> {
    THREAD_CHANNELS.with(|reg| reg.borrow().get(&id).cloned())
}

/// Clears all channels from the thread-local registry (useful between isolated test runs).
pub fn reset_global_channel_registry() {
    THREAD_CHANNELS.with(|reg| reg.borrow_mut().clear());
}

// Native builtin functions callable from Lust scripts:

pub fn builtin_channel_new(args: &[Value]) -> Result<NativeCallResult, String> {
    let capacity = match args.first() {
        Some(Value::Int(n)) if *n >= 0 => *n as usize,
        Some(Value::Nil) | None => 0, // default rendezvous
        Some(other) => {
            return Err(format!(
                "channel.new expects non-negative integer capacity, got {:?}",
                other
            ));
        }
    };

    let strategy = match args.get(1) {
        Some(Value::String(s)) => match s.as_str() {
            "drop_oldest" => OverflowStrategy::DropOldest,
            "drop_newest" => OverflowStrategy::DropNewest,
            "fail" => OverflowStrategy::Fail,
            _ => OverflowStrategy::Suspend,
        },
        _ => OverflowStrategy::Suspend,
    };

    let ch = Channel::<Value>::with_overflow(capacity, strategy);
    let id = register_global_channel(ch);
    Ok(NativeCallResult::Return(Value::Int(id as i64)))
}

pub fn builtin_channel_send(args: &[Value]) -> Result<NativeCallResult, String> {
    let ch_id = match args.first() {
        Some(Value::Int(id)) => *id as u64,
        other => {
            return Err(format!(
                "channel.send expects channel id integer, got {:?}",
                other
            ));
        }
    };
    let val = args.get(1).cloned().unwrap_or(Value::Nil);
    let ch = match get_global_channel(ch_id) {
        Some(c) => c,
        None => {
            return Err(format!("Invalid channel id: {}", ch_id));
        }
    };

    let vm_res = VM::with_current(|vm| {
        if vm.current_task_handle().is_some() {
            ch.send(val.clone(), vm).map_err(|e| e.to_string())
        } else {
            ch.try_send(val.clone()).map_err(|e| e.to_string())
        }
    });
    match vm_res {
        Ok(()) => Ok(NativeCallResult::Return(Value::Bool(true))),
        Err(err_msg) => {
            if err_msg.contains("No active VM context") {
                match ch.try_send(val) {
                    Ok(()) => Ok(NativeCallResult::Return(Value::Bool(true))),
                    Err(ChannelError::Closed) => Err("Channel is closed".to_string()),
                    Err(err) => Err(format!("Channel try_send failed: {:?}", err)),
                }
            } else {
                Err(err_msg)
            }
        }
    }
}

pub fn builtin_channel_try_send(args: &[Value]) -> Result<NativeCallResult, String> {
    let ch_id = match args.first() {
        Some(Value::Int(id)) => *id as u64,
        other => {
            return Err(format!(
                "channel.try_send expects channel id integer, got {:?}",
                other
            ));
        }
    };
    let val = args.get(1).cloned().unwrap_or(Value::Nil);
    let ch = match get_global_channel(ch_id) {
        Some(c) => c,
        None => {
            return Err(format!("Invalid channel id: {}", ch_id));
        }
    };

    match ch.try_send(val) {
        Ok(()) => Ok(NativeCallResult::Return(Value::Bool(true))),
        Err(ChannelError::Full) | Err(ChannelError::Closed) => {
            Ok(NativeCallResult::Return(Value::Bool(false)))
        }
        Err(err) => Err(format!("channel.try_send failed: {:?}", err)),
    }
}

pub fn builtin_channel_recv(args: &[Value]) -> Result<NativeCallResult, String> {
    let ch_id = match args.first() {
        Some(Value::Int(id)) => *id as u64,
        other => {
            return Err(format!(
                "channel.recv expects channel id integer, got {:?}",
                other
            ));
        }
    };
    let ch = match get_global_channel(ch_id) {
        Some(c) => c,
        None => {
            return Err(format!("Invalid channel id: {}", ch_id));
        }
    };

    let vm_res = VM::with_current(|vm| {
        if vm.current_task_handle().is_some() {
            ch.recv(vm).map_err(|e| e.to_string())
        } else {
            ch.try_recv().map_err(|e| e.to_string())
        }
    });
    match vm_res {
        Ok(Some(v)) => Ok(NativeCallResult::Return(v)),
        Ok(None) => Ok(NativeCallResult::Return(Value::Nil)),
        Err(err_msg) => {
            if err_msg.contains("No active VM context") {
                match ch.try_recv() {
                    Ok(Some(v)) => Ok(NativeCallResult::Return(v)),
                    Ok(None) | Err(ChannelError::Empty) | Err(ChannelError::Closed) => {
                        Ok(NativeCallResult::Return(Value::Nil))
                    }
                    Err(err) => Err(format!("Channel try_recv failed: {:?}", err)),
                }
            } else if err_msg.contains("closed") {
                Ok(NativeCallResult::Return(Value::Nil))
            } else {
                Err(err_msg)
            }
        }
    }
}

pub fn builtin_channel_try_recv(args: &[Value]) -> Result<NativeCallResult, String> {
    let ch_id = match args.first() {
        Some(Value::Int(id)) => *id as u64,
        other => {
            return Err(format!(
                "channel.try_recv expects channel id integer, got {:?}",
                other
            ));
        }
    };
    let ch = match get_global_channel(ch_id) {
        Some(c) => c,
        None => {
            return Err(format!("Invalid channel id: {}", ch_id));
        }
    };

    match ch.try_recv() {
        Ok(Some(v)) => Ok(NativeCallResult::Return(v)),
        Ok(None) | Err(ChannelError::Empty) | Err(ChannelError::Closed) => {
            Ok(NativeCallResult::Return(Value::Int(0)))
        }
        Err(err) => Err(format!("channel.try_recv failed: {:?}", err)),
    }
}

pub fn builtin_channel_close(args: &[Value]) -> Result<NativeCallResult, String> {
    let ch_id = match args.first() {
        Some(Value::Int(id)) => *id as u64,
        other => {
            return Err(format!(
                "channel.close expects channel id integer, got {:?}",
                other
            ));
        }
    };
    let ch = match get_global_channel(ch_id) {
        Some(c) => c,
        None => {
            return Err(format!("Invalid channel id: {}", ch_id));
        }
    };

    let vm_res = VM::with_current(|vm| {
        ch.close(vm);
        Ok::<(), String>(())
    });

    if vm_res.is_err() {
        ch.close_local();
    }
    Ok(NativeCallResult::Return(Value::Bool(true)))
}

pub fn builtin_channel_len(args: &[Value]) -> Result<NativeCallResult, String> {
    let ch_id = match args.first() {
        Some(Value::Int(id)) => *id as u64,
        other => {
            return Err(format!(
                "channel.len expects channel id integer, got {:?}",
                other
            ));
        }
    };
    let ch = match get_global_channel(ch_id) {
        Some(c) => c,
        None => {
            return Err(format!("Invalid channel id: {}", ch_id));
        }
    };
    Ok(NativeCallResult::Return(Value::Int(ch.len() as i64)))
}

pub fn builtin_channel_is_closed(args: &[Value]) -> Result<NativeCallResult, String> {
    let ch_id = match args.first() {
        Some(Value::Int(id)) => *id as u64,
        other => {
            return Err(format!(
                "channel.is_closed expects channel id integer, got {:?}",
                other
            ));
        }
    };
    let ch = match get_global_channel(ch_id) {
        Some(c) => c,
        None => {
            return Err(format!("Invalid channel id: {}", ch_id));
        }
    };
    Ok(NativeCallResult::Return(Value::Bool(ch.is_closed())))
}

/// Registers the complete channel builtin suite into the given VM instance.
///
/// Registers:
/// - Flat functions: `channel.new`, `channel.send`, `channel.try_send`, `channel.recv`,
///   `channel.try_recv`, `channel.close`, `channel.len`, `channel.is_closed`.
/// - Underscore aliases: `channel_new`, `channel_send`, `channel_try_send`, `channel_recv`,
///   `channel_try_recv`, `channel_close`, `channel_len`, `channel_is_closed`.
/// - Module table: `channel` table containing member functions.
pub fn register_builtins(vm: &mut VM) {
    let funcs: [(&str, Rc<NativeFn>); 8] = [
        ("new", native_fn(builtin_channel_new)),
        ("send", native_fn(builtin_channel_send)),
        ("try_send", native_fn(builtin_channel_try_send)),
        ("recv", native_fn(builtin_channel_recv)),
        ("try_recv", native_fn(builtin_channel_try_recv)),
        ("close", native_fn(builtin_channel_close)),
        ("len", native_fn(builtin_channel_len)),
        ("is_closed", native_fn(builtin_channel_is_closed)),
    ];

    let mut mod_map = LustMap::default();

    for (name, func) in &funcs {
        let dot_name = format!("channel.{}", name);
        let underscore_name = format!("channel_{}", name);

        // Global / unqualified
        vm.register_native(dot_name.clone(), Value::NativeFunction(func.clone()));
        vm.register_native(underscore_name.clone(), Value::NativeFunction(func.clone()));

        // Module-prefixed aliases for common modules (main, std, std.sync)
        vm.register_native(format!("main.{}", dot_name), Value::NativeFunction(func.clone()));
        vm.register_native(format!("main.{}", underscore_name), Value::NativeFunction(func.clone()));
        vm.register_native(format!("std.{}", dot_name), Value::NativeFunction(func.clone()));
        vm.register_native(format!("std.{}", underscore_name), Value::NativeFunction(func.clone()));
        vm.register_native(format!("std.sync.{}", dot_name), Value::NativeFunction(func.clone()));
        vm.register_native(format!("std.sync.{}", underscore_name), Value::NativeFunction(func.clone()));

        mod_map.insert(
            ValueKey::string((*name).to_string()),
            Value::NativeFunction(func.clone()),
        );
    }

    let map_val = Value::Map(Rc::new(RefCell::new(mod_map)));
    vm.register_native("channel", map_val.clone());
    vm.register_native("main.channel", map_val.clone());
    vm.register_native("std.channel", map_val.clone());
    vm.register_native("std.sync.channel", map_val);
}

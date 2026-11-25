//! Reactive Lifecycle State Machine, Subscription, and Observer implementations.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use super::buffer::{BoundedQueue, BufferOptions, MeasurableItem};
use super::scheduler::ScopeJoinCoordinator;
use super::RxError;

/// Re-entrant admission gate that serializes callback start and terminal transitions
/// while permitting re-entrant calls (such as self-unsubscribe or self-completion)
/// from within an active callback on the same thread.
#[derive(Default)]
pub struct AdmissionGate {
    state: Mutex<(Option<u64>, usize)>,
    cond: Condvar,
}

pub struct AdmissionGuard<'a> {
    gate: &'a AdmissionGate,
}

impl AdmissionGate {
    pub fn new() -> Self {
        Self {
            state: Mutex::new((None, 0)),
            cond: Condvar::new(),
        }
    }

    pub fn lock(&self) -> AdmissionGuard<'_> {
        let tid = thread_id_to_u64();
        let mut st = self.state.lock().unwrap();
        loop {
            match *st {
                (Some(holder), ref mut depth) if holder == tid => {
                    *depth += 1;
                    return AdmissionGuard { gate: self };
                }
                (None, _) => {
                    *st = (Some(tid), 1);
                    return AdmissionGuard { gate: self };
                }
                _ => {
                    st = self.cond.wait(st).unwrap();
                }
            }
        }
    }

    fn unlock(&self) {
        let tid = thread_id_to_u64();
        let mut st = self.state.lock().unwrap();
        if let (Some(holder), ref mut depth) = *st {
            if holder == tid {
                *depth -= 1;
                if *depth == 0 {
                    *st = (None, 0);
                    self.cond.notify_one();
                }
            }
        }
    }
}

impl<'a> Drop for AdmissionGuard<'a> {
    fn drop(&mut self) {
        self.gate.unlock();
    }
}

/// Lifecycle states of a subscription as defined in RFC §16.3 and Astra's brief.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubscriptionState {
    Active,
    Stopping,
    Joining,
    Releasing,
    Closed,
}

/// Final completion exit status of a subscription.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExitStatus<E> {
    Completed,
    Failed(RxError<E>),
    Cancelled,
}

/// Final outcome package returned when joining a subscription.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completion<E> {
    pub exit: ExitStatus<E>,
    pub items_delivered: usize,
}

/// Trait or structure representing a consumer of stream events.
pub struct Observer<T, E = String> {
    pub on_next: Box<dyn FnMut(T) -> Result<(), RxError<E>> + Send + 'static>,
    pub on_complete: Box<dyn FnOnce() + Send + 'static>,
    pub on_failure: Box<dyn FnOnce(RxError<E>) + Send + 'static>,
}

impl<T, E: 'static> Observer<T, E> {
    pub fn new<N, C, F>(on_next: N, on_complete: C, on_failure: F) -> Self
    where
        N: FnMut(T) -> Result<(), RxError<E>> + Send + 'static,
        C: FnOnce() + Send + 'static,
        F: FnOnce(RxError<E>) + Send + 'static,
    {
        Self {
            on_next: Box::new(on_next),
            on_complete: Box::new(on_complete),
            on_failure: Box::new(on_failure),
        }
    }

    /// Construct a simple next-only observer.
    pub fn from_next<N>(on_next: N) -> Self
    where
        N: FnMut(T) -> Result<(), RxError<E>> + Send + 'static,
    {
        Self {
            on_next: Box::new(on_next),
            on_complete: Box::new(|| {}),
            on_failure: Box::new(|_| {}),
        }
    }
}

struct TerminalHandlers<E> {
    on_complete: Box<dyn FnOnce() + Send + 'static>,
    on_failure: Box<dyn FnOnce(RxError<E>) + Send + 'static>,
}

/// Internal shared control block for a subscription.
pub struct SubscriptionControl<T: MeasurableItem + Send + 'static, E: Clone + Send + 'static> {
    pub scope_id: u64,
    pub generation: AtomicU32,
    pub state: Mutex<SubscriptionState>,
    pub exit_status: Mutex<Option<ExitStatus<E>>>,
    pub queue: Mutex<BoundedQueue<T>>,
    next_fn: Mutex<Option<Box<dyn FnMut(T) -> Result<(), RxError<E>> + Send + 'static>>>,
    terminal_fn: Mutex<Option<TerminalHandlers<E>>>,
    pub admission_gate: AdmissionGate,
    pub coordinator: Arc<ScopeJoinCoordinator>,
    pub items_delivered: AtomicUsize,
    pub current_executing_thread: AtomicU64,
    participant_id: Mutex<Option<super::scheduler::ParticipantId>>,
}

impl<T: MeasurableItem + Send + 'static, E: Clone + Send + 'static> SubscriptionControl<T, E> {
    pub fn new(
        scope_id: u64,
        options: BufferOptions,
        observer: Observer<T, E>,
        coordinator: Arc<ScopeJoinCoordinator>,
    ) -> Self {
        let terminal = TerminalHandlers {
            on_complete: observer.on_complete,
            on_failure: observer.on_failure,
        };
        Self {
            scope_id,
            generation: AtomicU32::new(1),
            state: Mutex::new(SubscriptionState::Active),
            exit_status: Mutex::new(None),
            queue: Mutex::new(BoundedQueue::new(options)),
            next_fn: Mutex::new(Some(observer.on_next)),
            terminal_fn: Mutex::new(Some(terminal)),
            admission_gate: AdmissionGate::new(),
            coordinator,
            items_delivered: AtomicUsize::new(0),
            current_executing_thread: AtomicU64::new(0),
            participant_id: Mutex::new(None),
        }
    }

    /// Attach a registered participant identifier to this subscription control.
    pub fn attach_participant(&self, pid: super::scheduler::ParticipantId) {
        *self.participant_id.lock().unwrap() = Some(pid);
    }

    /// Acknowledge participant exit with the coordinator once.
    pub fn acknowledge_participant_exit(&self) {
        if let Some(pid) = self.participant_id.lock().unwrap().take() {
            self.coordinator.acknowledge_exit(self.scope_id, pid);
        }
    }

    /// Linearize cancellation:
    /// - Acquire admission_gate to serialize with any pending callback admission
    /// - Transition Active -> Stopping
    /// - Increment generation so late timer envelopes / arrivals are discarded
    /// - Close admission gate: no new callbacks will start
    /// - Record ExitStatus::Cancelled if no terminal outcome committed yet
    pub fn cancel(&self) {
        let _gate = self.admission_gate.lock();
        let mut state = self.state.lock().unwrap();
        if *state != SubscriptionState::Active {
            return;
        }
        *state = SubscriptionState::Stopping;
        self.generation.fetch_add(1, Ordering::SeqCst);

        let mut exit = self.exit_status.lock().unwrap();
        if exit.is_none() {
            *exit = Some(ExitStatus::Cancelled);
        }
        self.coordinator.close_admission(self.scope_id);

        // Discard terminal handlers on cancellation per RFC §16.3
        drop(self.terminal_fn.lock().unwrap().take());
        self.acknowledge_participant_exit();
    }

    /// Check if subscription is actively accepting new callbacks.
    pub fn is_active(&self) -> bool {
        let state = self.state.lock().unwrap();
        *state == SubscriptionState::Active
    }

    /// Deliver next item to observer.
    /// Inviolable invariant: No callback starts after cancel() has committed!
    pub fn deliver_next(&self, item: T) -> Result<(), RxError<E>> {
        // Hold admission gate across the entire callback.
        // Serialized with cancel(), complete(), and fail().
        // Re-entrant calls from same thread (e.g. self-cancel) succeed without deadlock.
        let _gate = self.admission_gate.lock();

        {
            let state = self.state.lock().unwrap();
            if *state != SubscriptionState::Active {
                return Err(RxError::Cancelled);
            }
        }

        let current_tid = thread_id_to_u64();
        self.current_executing_thread.store(current_tid, Ordering::SeqCst);

        let mut next_guard = self.next_fn.lock().unwrap();
        let res = if let Some(ref mut next_cb) = *next_guard {
            let next_res = next_cb(item);
            if next_res.is_ok() {
                self.items_delivered.fetch_add(1, Ordering::SeqCst);
            }
            next_res
        } else {
            Err(RxError::Cancelled)
        };
        drop(next_guard);

        self.current_executing_thread.store(0, Ordering::SeqCst);

        // If observer returned an error, fail the subscription
        if let Err(ref e) = res {
            self.fail(e.clone());
        }

        res
    }

    /// Commit normal stream completion.
    /// Terminal callback is executed OUTSIDE of all mutexes to eliminate reentrancy deadlocks.
    pub fn complete(&self) {
        let action = {
            let _gate = self.admission_gate.lock();
            let mut state = self.state.lock().unwrap();
            if *state == SubscriptionState::Active {
                *state = SubscriptionState::Stopping;
                let mut exit = self.exit_status.lock().unwrap();
                if exit.is_none() {
                    *exit = Some(ExitStatus::Completed);
                }
                self.coordinator.close_admission(self.scope_id);
                self.terminal_fn.lock().unwrap().take()
            } else {
                None
            }
        };

        // Invoke outside of all mutexes!
        if let Some(terminal) = action {
            self.acknowledge_participant_exit();
            (terminal.on_complete)();
        } else {
            self.acknowledge_participant_exit();
        }
    }

    /// Commit stream failure.
    /// Terminal callback is executed OUTSIDE of all mutexes to eliminate reentrancy deadlocks.
    pub fn fail(&self, err: RxError<E>) {
        let action = {
            let _gate = self.admission_gate.lock();
            let mut state = self.state.lock().unwrap();
            if *state == SubscriptionState::Active {
                *state = SubscriptionState::Stopping;
                let mut exit = self.exit_status.lock().unwrap();
                if exit.is_none() {
                    *exit = Some(ExitStatus::Failed(err.clone()));
                }
                self.coordinator.close_admission(self.scope_id);
                self.terminal_fn.lock().unwrap().take()
            } else {
                None
            }
        };

        // Invoke outside of all mutexes!
        if let Some(terminal) = action {
            self.acknowledge_participant_exit();
            (terminal.on_failure)(err);
        } else {
            self.acknowledge_participant_exit();
        }
    }

    /// Join and wait for all participants to exit.
    /// Returns Completion report.
    pub fn join(&self, timeout: Option<Duration>) -> Result<Completion<E>, RxError<E>> {
        // Detect self-join from within the observer's callback thread
        let current_tid = thread_id_to_u64();
        if self.current_executing_thread.load(Ordering::SeqCst) == current_tid && current_tid != 0 {
            return Err(RxError::SelfJoin);
        }

        {
            let mut state = self.state.lock().unwrap();
            if *state != SubscriptionState::Closed {
                *state = SubscriptionState::Joining;
            }
        }

        // Acknowledge participant exit if not already acknowledged
        self.acknowledge_participant_exit();

        // Wait for all registered producers, timers, and callbacks to acknowledge exit
        let quiescent = self.coordinator.close_and_wait_quiescent(self.scope_id, timeout);
        if !quiescent {
            return Err(RxError::Timeout);
        }

        // Transition Joining -> Releasing -> Closed
        {
            let mut state = self.state.lock().unwrap();
            *state = SubscriptionState::Releasing;
            // Clean up queue and observer captures
            let mut q = self.queue.lock().unwrap();
            q.clear();
            *self.next_fn.lock().unwrap() = None;
            *self.terminal_fn.lock().unwrap() = None;
            *state = SubscriptionState::Closed;
        }

        // Retire scope from coordinator
        self.coordinator.retire_scope(self.scope_id);

        let exit = self
            .exit_status
            .lock()
            .unwrap()
            .clone()
            .unwrap_or(ExitStatus::Completed);
        let delivered = self.items_delivered.load(Ordering::SeqCst);

        Ok(Completion {
            exit,
            items_delivered: delivered,
        })
    }
}

/// Opaque subscription handle held by caller.
#[derive(Clone)]
pub struct Subscription<T: MeasurableItem + Send + 'static, E: Clone + Send + 'static> {
    pub control: Arc<SubscriptionControl<T, E>>,
}

impl<T: MeasurableItem + Send + 'static, E: Clone + Send + 'static> Subscription<T, E> {
    pub fn new(control: Arc<SubscriptionControl<T, E>>) -> Self {
        Self { control }
    }

    /// Cancel the subscription.
    pub fn cancel(&self) {
        self.control.cancel();
    }

    /// Join subscription, waiting for all participants to exit and resources to be released.
    pub fn join(&self, timeout: Option<Duration>) -> Result<Completion<E>, RxError<E>> {
        self.control.join(timeout)
    }

    /// Combined cancel and join.
    pub fn close(&self, timeout: Option<Duration>) -> Result<Completion<E>, RxError<E>> {
        self.cancel();
        self.join(timeout)
    }

    /// Current subscription state.
    pub fn state(&self) -> SubscriptionState {
        *self.control.state.lock().unwrap()
    }
}

fn thread_id_to_u64() -> u64 {
    let tid = std::thread::current().id();
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    tid.hash(&mut hasher);
    hasher.finish()
}

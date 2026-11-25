//! Reactive Streams (Rx) Foundation for Lust.
//! Implements scoped cancellation, backpressure, virtual scheduling, and deterministic timers.

pub mod buffer;
pub mod clock;
pub mod lifecycle;
pub mod observable;
pub mod scheduler;
pub mod subject;
pub mod timer;

pub use buffer::{BoundedQueue, BufferOptions, EnqueueResult, MeasurableItem, OverflowStrategy};
pub use clock::{PhysicalClock, RxClock, SharedClock, VirtualClock};
pub use lifecycle::{
    Completion, ExitStatus, Observer, Subscription, SubscriptionControl, SubscriptionState,
};
pub use observable::{interval, Observable, ProducerFn};
pub use scheduler::{
    ParticipantId, ReadyTask, RxScheduler, ScopeJoinCoordinator,
    MUTATION_M3_EXECUTION_COUNT, MUTATION_M3_SKIP_CLOSED_CHECK, MutationM3Guard,
};
pub use subject::{Subject, SubjectOptions};
pub use timer::{RxTimerService, TimerId};

/// Core error types for the Reactive Streams subsystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RxError<E = String> {
    /// Upstream source failed with a user error.
    Source(E),
    /// Queue capacity reached and overflow strategy is Fail.
    BackpressureOverflow,
    /// Single item exceeds total byte capacity of buffer.
    ItemTooLarge,
    /// Invalid configuration options (e.g., non-positive capacity or inverted watermarks).
    InvalidOptions(String),
    /// Subscription handle was invalidated, stale, or from an unauthorized domain.
    InvalidHandle,
    /// Attempted self-join from within an active callback.
    SelfJoin,
    /// Operation timed out.
    Timeout,
    /// Subscription cancelled by scope or downstream observer.
    Cancelled,
}

impl<E: std::fmt::Display> std::fmt::Display for RxError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Source(e) => write!(f, "Rx source error: {}", e),
            Self::BackpressureOverflow => write!(f, "Rx backpressure overflow"),
            Self::ItemTooLarge => write!(f, "Rx item exceeds buffer byte capacity"),
            Self::InvalidOptions(msg) => write!(f, "Invalid Rx buffer options: {}", msg),
            Self::InvalidHandle => write!(f, "Invalid or stale Rx subscription handle"),
            Self::SelfJoin => write!(f, "Deadlock prevented: cannot join subscription from its own callback"),
            Self::Timeout => write!(f, "Rx operation timed out"),
            Self::Cancelled => write!(f, "Rx stream subscription was cancelled"),
        }
    }
}

impl<E: std::fmt::Debug + std::fmt::Display> std::error::Error for RxError<E> {}

use std::sync::Arc;
use std::time::Duration;

/// Unified Reactive Streams environment managing execution, scheduling, timers, and scopes.
#[derive(Clone)]
pub struct RxEnvironment {
    pub scheduler: Arc<RxScheduler>,
}

impl Default for RxEnvironment {
    fn default() -> Self {
        Self::new_virtual()
    }
}

impl RxEnvironment {
    /// Create a new Rx environment with a virtual clock starting at 0 nanoseconds.
    pub fn new_virtual() -> Self {
        Self {
            scheduler: Arc::new(RxScheduler::new_virtual()),
        }
    }

    /// Create a new Rx environment with a virtual clock initialized to specific nanoseconds.
    pub fn new_virtual_with_initial_time(initial_nanos: u64, max_ready_tasks: usize) -> Self {
        let clock: SharedClock = Arc::new(VirtualClock::new(initial_nanos));
        Self {
            scheduler: Arc::new(RxScheduler::new_with_clock(clock, max_ready_tasks)),
        }
    }

    /// Create a new Rx environment with a real physical clock.
    pub fn new_physical() -> Self {
        let clock: SharedClock = Arc::new(PhysicalClock::new());
        Self {
            scheduler: Arc::new(RxScheduler::new_with_clock(clock, 1024)),
        }
    }

    /// Create a new Rx environment with a custom clock and capacity.
    pub fn with_clock(clock: SharedClock, capacity: usize) -> Self {
        Self {
            scheduler: Arc::new(RxScheduler::new_with_clock(clock, capacity)),
        }
    }

    /// Access the underlying scheduler.
    pub fn scheduler(&self) -> &Arc<RxScheduler> {
        &self.scheduler
    }

    /// Access the shared clock.
    pub fn clock(&self) -> &SharedClock {
        &self.scheduler.clock
    }

    /// Access the timer service.
    pub fn timers(&self) -> &Arc<RxTimerService> {
        &self.scheduler.timers
    }

    /// Access the scope coordinator.
    pub fn coordinator(&self) -> &Arc<ScopeJoinCoordinator> {
        &self.scheduler.coordinator
    }

    /// Current timestamp in nanoseconds according to the environment clock.
    pub fn now_nanos(&self) -> u64 {
        self.scheduler.clock.now_nanos()
    }

    /// Advance virtual time by a given duration (no-op on physical clock).
    pub fn advance_time(&self, duration: Duration) {
        self.scheduler.clock.advance(duration);
    }

    /// Schedule a task for cooperative execution in the scheduler.
    pub fn schedule_task<F>(&self, priority: u8, task: F) -> Result<(), RxError>
    where
        F: FnOnce() + Send + 'static,
    {
        self.scheduler.schedule_task(priority, task)
    }

    /// Schedule a one-shot timer to execute an action after the specified delay.
    pub fn schedule_timer<F>(&self, delay: Duration, action: F) -> TimerId
    where
        F: FnOnce() + Send + 'static,
    {
        let deadline = self
            .now_nanos()
            .saturating_add(delay.as_nanos().min(u64::MAX as u128) as u64);
        self.scheduler.timers.schedule(deadline, 1, action)
    }

    /// Cancel an active timer by its identifier.
    pub fn cancel_timer(&self, timer_id: TimerId) -> bool {
        self.scheduler.timers.cancel(timer_id, 1)
    }

    /// Subscribe an observer to an observable within this environment.
    pub fn subscribe<T, E>(
        &self,
        stream: Observable<T, E>,
        observer: Observer<T, E>,
        scope_id: u64,
        options: BufferOptions,
    ) -> Result<Subscription<T, E>, RxError<E>>
    where
        T: MeasurableItem + Send + Sync + 'static,
        E: Clone + Send + 'static,
    {
        stream.subscribe(observer, scope_id, options, self.scheduler.clone())
    }

    /// Cancel admission for a scope, preventing new participants.
    pub fn cancel_scope(&self, scope_id: u64) {
        self.scheduler.coordinator.close_admission(scope_id);
    }

    /// Wait for all active participants of a scope to complete/quiesce.
    pub fn wait_scope_quiescent(&self, scope_id: u64, timeout: Option<Duration>) -> bool {
        self.scheduler.coordinator.wait_for_quiescence(scope_id, timeout)
    }

    /// Advance time by the given duration and drain any expired timers and ready tasks.
    pub fn advance_and_drain(&self, duration: Duration) -> usize {
        self.scheduler.advance_and_drain(duration)
    }

    /// Run the scheduler until all ready tasks and virtual timer deadlines are settled up to max_steps.
    pub fn run_until_quiescent(&self, max_steps: usize) -> usize {
        self.scheduler.run_until_quiescent(max_steps)
    }

    /// Create a hot multicast Subject in this environment.
    pub fn subject<T, E>(&self) -> Subject<T, E>
    where
        T: MeasurableItem + Clone + Send + Sync + 'static,
        E: Clone + Send + 'static,
    {
        Subject::new(self.scheduler.clone())
    }

    /// Create a hot multicast Subject with custom options in this environment.
    pub fn subject_with_options<T, E>(&self, options: SubjectOptions) -> Subject<T, E>
    where
        T: MeasurableItem + Clone + Send + Sync + 'static,
        E: Clone + Send + 'static,
    {
        Subject::with_options(self.scheduler.clone(), options)
    }

    /// Shut down the scheduler.
    pub fn shutdown(&self) {
        self.scheduler.shutdown();
    }
}

/// Standard library symbol resolver for `@std/rx` / `std.rx`.
pub fn resolve_rx_function(item_name: &str) -> Option<&'static str> {
    match item_name {
        "Subject" => Some("rx.Subject"),
        "Observable" => Some("rx.Observable"),
        "interval" => Some("rx.interval"),
        _ => None,
    }
}

/// Virtual standard module functions exposed by `@std/rx` / `std.rx`.
pub fn virtual_rx_functions() -> Vec<(&'static str, &'static str)> {
    vec![
        ("Subject", "rx.Subject"),
        ("Observable", "rx.Observable"),
        ("interval", "rx.interval"),
    ]
}

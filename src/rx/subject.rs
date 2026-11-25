//! Hot Multicast Subject for Reactive Streams in Lust.
//!
//! Normative contract per RFC §16.3 and Gate-12 review:
//! 1. Hot Multicast: Values emitted before subscription are discarded (no replay).
//! 2. Single Scheduler Domain: All subscribers and the Subject live in the same scheduler domain.
//! 3. Bounded Backpressure & Slow-Subscriber Isolation:
//!    - Each subscriber has its own independent `BoundedQueue`.
//!    - When a slow subscriber's queue overflows, by default that subscriber is terminated
//!      with a typed `Fail` (`RxError::BackpressureOverflow`).
//!    - Fast subscribers continue uninterrupted and receive all items.
//! 4. Subscriber Cap:
//!    - Maximum active subscribers is strictly capped (`max_subscribers`).
//!    - Subscribing beyond capacity fails with `RxError::BackpressureOverflow`.
//! 5. Terminal Lifecycle:
//!    - `complete()` transitions Subject to completed; all active subscribers receive `complete()`.
//!    - `fail(err)` transitions Subject to failed; all active subscribers receive `fail(err)`.
//!    - Terminal transitions are idempotent.
//!    - Subscribing to an already-completed/failed Subject immediately delivers the terminal signal.
//! 6. Zero Lingering Callbacks Post-Join / Unsubscribe:
//!    - Cancellation cleanly removes the subscriber from the Subject's active registry.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::buffer::{BufferOptions, EnqueueResult, MeasurableItem};
use super::lifecycle::{ExitStatus, Observer, Subscription, SubscriptionControl};
use super::observable::Observable;
use super::scheduler::RxScheduler;
use super::RxError;

/// Configuration options for a Subject.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubjectOptions {
    /// Maximum number of concurrent active subscribers permitted.
    pub max_subscribers: usize,
    /// Default buffer options applied to new subscriptions if not overridden.
    pub default_buffer_options: BufferOptions,
}

impl Default for SubjectOptions {
    fn default() -> Self {
        Self {
            max_subscribers: 64,
            default_buffer_options: BufferOptions::default_bounded(),
        }
    }
}

impl SubjectOptions {
    pub fn new(max_subscribers: usize, default_buffer_options: BufferOptions) -> Result<Self, &'static str> {
        if max_subscribers == 0 {
            return Err("max_subscribers must be greater than zero");
        }
        Ok(Self {
            max_subscribers,
            default_buffer_options,
        })
    }
}

struct SubscriberEntry<T: MeasurableItem + Send + 'static, E: Clone + Send + 'static> {
    _id: u64,
    control: Arc<SubscriptionControl<T, E>>,
}

struct SubjectInner<T: MeasurableItem + Send + 'static, E: Clone + Send + 'static> {
    subscribers: Vec<SubscriberEntry<T, E>>,
    terminal: Option<ExitStatus<E>>,
    next_sub_id: u64,
}

/// A hot multicast reactive stream primitive broadcasting items to zero or more observers.
#[derive(Clone)]
pub struct Subject<T: MeasurableItem + Send + Sync + 'static, E: Clone + Send + 'static = String> {
    scheduler: Arc<RxScheduler>,
    options: SubjectOptions,
    inner: Arc<Mutex<SubjectInner<T, E>>>,
    total_emitted: Arc<AtomicU64>,
}

impl<T: MeasurableItem + Clone + Send + Sync + 'static, E: Clone + Send + 'static> Subject<T, E> {
    /// Create a new Subject with default options within the given scheduler domain.
    pub fn new(scheduler: Arc<RxScheduler>) -> Self {
        Self::with_options(scheduler, SubjectOptions::default())
    }

    /// Create a new Subject with custom options within the given scheduler domain.
    pub fn with_options(scheduler: Arc<RxScheduler>, options: SubjectOptions) -> Self {
        Self {
            scheduler,
            options,
            inner: Arc::new(Mutex::new(SubjectInner {
                subscribers: Vec::new(),
                terminal: None,
                next_sub_id: 1,
            })),
            total_emitted: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Access the underlying scheduler domain.
    pub fn scheduler(&self) -> &Arc<RxScheduler> {
        &self.scheduler
    }

    /// Access configured Subject options.
    pub fn options(&self) -> &SubjectOptions {
        &self.options
    }

    /// Total count of items submitted via `next()` while active.
    pub fn total_emitted(&self) -> u64 {
        self.total_emitted.load(Ordering::SeqCst)
    }

    /// Check if Subject is active and accepting subscriptions/emissions.
    pub fn is_active(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.terminal.is_none()
    }

    /// Check if Subject reached terminal completion.
    pub fn is_completed(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        matches!(inner.terminal, Some(ExitStatus::Completed))
    }

    /// Check if Subject reached terminal failure.
    pub fn is_failed(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        matches!(inner.terminal, Some(ExitStatus::Failed(_)))
    }

    /// Current count of active subscribers (pruning any that have been cancelled or closed).
    pub fn subscriber_count(&self) -> usize {
        let mut inner = self.inner.lock().unwrap();
        inner.subscribers.retain(|entry| entry.control.is_active());
        inner.subscribers.len()
    }

    /// Subscribe an existing SubscriptionControl directly to this hot Subject.
    pub fn subscribe_control(
        &self,
        control: Arc<SubscriptionControl<T, E>>,
    ) -> Result<Subscription<T, E>, RxError<E>> {
        let mut inner = self.inner.lock().unwrap();

        // 1. Check terminal state first
        if let Some(ref term) = inner.terminal {
            match term {
                ExitStatus::Completed => {
                    control.complete();
                }
                ExitStatus::Failed(err) => {
                    control.fail(err.clone());
                }
                ExitStatus::Cancelled => {
                    control.cancel();
                }
            }
            return Ok(Subscription::new(control));
        }

        // 2. Prune inactive subscribers before checking cap
        inner.subscribers.retain(|entry| entry.control.is_active());

        // 3. Enforce subscriber cap
        if inner.subscribers.len() >= self.options.max_subscribers {
            return Err(RxError::BackpressureOverflow);
        }

        let sub_id = inner.next_sub_id;
        inner.next_sub_id = inner.next_sub_id.wrapping_add(1);

        inner.subscribers.push(SubscriberEntry {
            _id: sub_id,
            control: control.clone(),
        });

        Ok(Subscription::new(control))
    }

    /// Subscribe an observer to this hot multicast Subject.
    ///
    /// Contract:
    /// - If the Subject is already completed or failed, the observer is notified immediately
    ///   and a terminal subscription is returned without registering in the active list.
    /// - If the active subscriber cap (`max_subscribers`) has been reached, returns
    ///   `Err(RxError::BackpressureOverflow)`.
    /// - Items emitted prior to subscription are NOT delivered (no replay).
    pub fn subscribe(
        &self,
        observer: Observer<T, E>,
        scope_id: u64,
        options: BufferOptions,
    ) -> Result<Subscription<T, E>, RxError<E>> {
        let participant_id = self
            .scheduler
            .coordinator
            .register_participant(scope_id)
            .map_err(|_| RxError::Cancelled)?;

        let control = Arc::new(SubscriptionControl::new(
            scope_id,
            options,
            observer,
            self.scheduler.coordinator.clone(),
        ));
        control.attach_participant(participant_id);

        self.subscribe_control(control)
    }

    /// Emit an item to all current active subscribers.
    ///
    /// Contract:
    /// - Hot multicast: broadcasted to every active subscriber.
    /// - Slow subscriber isolation: if a subscriber's queue overflows with `Fail` strategy,
    ///   only that subscriber is failed with `RxError::BackpressureOverflow` and pruned.
    ///   Other subscribers continue receiving items unimpeded.
    /// - Drops locks before invoking delivery callbacks to guarantee re-entrant safety
    ///   and prevent deadlock.
    /// - Returns the count of subscribers successfully delivered to.
    pub fn next(&self, item: T) -> Result<usize, RxError<E>> {
        // Snapshot active subscribers without holding lock during delivery
        let targets: Vec<Arc<SubscriptionControl<T, E>>> = {
            let mut inner = self.inner.lock().unwrap();
            if inner.terminal.is_some() {
                return Err(RxError::Cancelled);
            }
            inner.subscribers.retain(|entry| entry.control.is_active());
            inner.subscribers.iter().map(|e| e.control.clone()).collect()
        };

        self.total_emitted.fetch_add(1, Ordering::SeqCst);

        let mut delivered_count = 0;
        let mut failed_subscribers = Vec::new();

        for control in targets {
            if !control.is_active() {
                failed_subscribers.push(control);
                continue;
            }

            // Check bounded queue capacity and overflow strategy for this subscriber
            let enqueue_res = {
                let mut q = control.queue.lock().unwrap();
                q.try_enqueue(item.clone())
            };

            match enqueue_res {
                EnqueueResult::Enqueued => {
                    // Deliver to observer callback outside queue lock
                    let deliv_res = control.deliver_next(item.clone());
                    // Remove item from subscriber queue after delivery
                    {
                        let mut q = control.queue.lock().unwrap();
                        let _ = q.dequeue();
                    }
                    if deliv_res.is_ok() {
                        delivered_count += 1;
                    } else {
                        // deliver_next already invoked control.fail on error
                        failed_subscribers.push(control);
                    }
                }
                EnqueueResult::ReplacedOldest(_) => {
                    let deliv_res = control.deliver_next(item.clone());
                    {
                        let mut q = control.queue.lock().unwrap();
                        let _ = q.dequeue();
                    }
                    if deliv_res.is_ok() {
                        delivered_count += 1;
                    } else {
                        failed_subscribers.push(control);
                    }
                }
                EnqueueResult::DroppedNewest => {
                    // Item was dropped for this subscriber per DropNewest strategy.
                    // Subscriber remains active, but item not delivered to it.
                }
                EnqueueResult::OverflowError
                | EnqueueResult::ItemTooLarge
                | EnqueueResult::MustSuspend => {
                    // Slow-subscriber overflow: isolate failure to THIS subscriber only!
                    // Terminate with BackpressureOverflow per normative RFC §16.3.
                    control.fail(RxError::BackpressureOverflow);
                    failed_subscribers.push(control);
                }
            }
        }

        // Clean up failed/inactive subscribers from inner registry
        if !failed_subscribers.is_empty() {
            let mut inner = self.inner.lock().unwrap();
            inner.subscribers.retain(|entry| entry.control.is_active());
        }

        Ok(delivered_count)
    }

    /// Commit terminal completion on the Subject.
    /// All active subscribers receive `on_complete()` and are cleared.
    /// Subsequent emissions or subscriptions fail / complete immediately.
    pub fn complete(&self) {
        let subscribers = {
            let mut inner = self.inner.lock().unwrap();
            if inner.terminal.is_some() {
                return;
            }
            inner.terminal = Some(ExitStatus::Completed);
            std::mem::take(&mut inner.subscribers)
        };

        // Notify subscribers outside lock
        for entry in subscribers {
            entry.control.complete();
        }
    }

    /// Commit terminal failure on the Subject.
    /// All active subscribers receive `on_failure(err)` and are cleared.
    pub fn fail(&self, err: RxError<E>) {
        let subscribers = {
            let mut inner = self.inner.lock().unwrap();
            if inner.terminal.is_some() {
                return;
            }
            inner.terminal = Some(ExitStatus::Failed(err.clone()));
            std::mem::take(&mut inner.subscribers)
        };

        // Notify subscribers outside lock
        for entry in subscribers {
            entry.control.fail(err.clone());
        }
    }

    /// Convert this Subject into an `Observable<T, E>`.
    /// When subscribed, the resulting Observable forwards subscriptions to this hot Subject,
    /// enabling composition with `map`, `filter`, `take`, etc.
    pub fn as_observable(&self) -> Observable<T, E> {
        let subject = self.clone();
        Observable::new(move |control| {
            subject.subscribe_control(control)?;
            Ok(())
        })
    }

    /// Create an `Observer<T, E>` that pushes items into this Subject.
    /// Useful for subscribing this Subject to an upstream Observable.
    pub fn as_observer(&self) -> Observer<T, E> {
        let subject = self.clone();
        Observer::new(
            move |item| {
                subject.next(item).map(|_| ())
            },
            {
                let subject = self.clone();
                move || subject.complete()
            },
            {
                let subject = self.clone();
                move |err| subject.fail(err)
            },
        )
    }
}

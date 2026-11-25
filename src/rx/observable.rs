//! Observable blueprint and stream creation factories.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use super::buffer::{BufferOptions, MeasurableItem};
use super::lifecycle::{Observer, Subscription, SubscriptionControl};
use super::scheduler::RxScheduler;
use super::RxError;

/// Type alias for stream producer closure.
pub type ProducerFn<T, E> =
    Arc<dyn Fn(Arc<SubscriptionControl<T, E>>) -> Result<(), RxError<E>> + Send + Sync + 'static>;

/// A cold, lazy blueprint for a reactive stream.
/// Executed only upon call to `subscribe`.
#[derive(Clone)]
pub struct Observable<T: MeasurableItem + Send + Sync + 'static, E: Clone + Send + 'static = String> {
    producer: ProducerFn<T, E>,
}

impl<T: MeasurableItem + Send + Sync + 'static, E: Clone + Send + 'static> Observable<T, E> {
    pub fn new<F>(producer: F) -> Self
    where
        F: Fn(Arc<SubscriptionControl<T, E>>) -> Result<(), RxError<E>> + Send + Sync + 'static,
    {
        Self {
            producer: Arc::new(producer),
        }
    }

    /// Create an Observable from an in-memory iterable collection.
    /// Pulls items lazily on demand until collection ends or subscription cancels.
    pub fn from_iter<I>(iter: I) -> Self
    where
        T: Clone,
        I: IntoIterator<Item = T>,
    {
        let items: Vec<T> = iter.into_iter().collect();
        Self::new(move |control| {
            for item in items.iter() {
                if !control.is_active() {
                    return Err(RxError::Cancelled);
                }
                if let Err(e) = control.deliver_next(item.clone()) {
                    return Err(e);
                }
            }
            control.complete();
            Ok(())
        })
    }

    /// Create an Observable driven by an arbitrary cooperative producer closure.
    pub fn from_producer<F>(producer_fn: F) -> Self
    where
        F: Fn(Arc<SubscriptionControl<T, E>>) -> Result<(), RxError<E>> + Send + Sync + 'static,
    {
        Self::new(producer_fn)
    }

    /// Transform stream values with a projection function.
    pub fn map<R, F>(self, f: F) -> Observable<R, E>
    where
        R: MeasurableItem + Send + Sync + 'static,
        F: Fn(T) -> R + Send + Sync + 'static,
    {
        let upstream_producer = self.producer.clone();
        let f = Arc::new(f);
        Observable::new(move |downstream_ctrl| {
            let f_clone = f.clone();
            let ds_ctrl = downstream_ctrl.clone();
            let upstream_ctrl = Arc::new(SubscriptionControl::new(
                downstream_ctrl.scope_id,
                BufferOptions::default_bounded(),
                Observer::new(
                    move |item: T| {
                        let mapped = f_clone(item);
                        ds_ctrl.deliver_next(mapped)
                    },
                    {
                        let ds_ctrl = downstream_ctrl.clone();
                        move || ds_ctrl.complete()
                    },
                    {
                        let ds_ctrl = downstream_ctrl.clone();
                        move |err| ds_ctrl.fail(err)
                    },
                ),
                downstream_ctrl.coordinator.clone(),
            ));
            let res = upstream_producer(upstream_ctrl);
            if let Err(RxError::Cancelled) = res {
                Ok(())
            } else {
                res
            }
        })
    }

    /// Filter stream values by a predicate.
    pub fn filter<P>(self, predicate: P) -> Observable<T, E>
    where
        P: Fn(&T) -> bool + Send + Sync + 'static,
    {
        let upstream_producer = self.producer.clone();
        let predicate = Arc::new(predicate);
        Observable::new(move |downstream_ctrl| {
            let pred_clone = predicate.clone();
            let ds_ctrl = downstream_ctrl.clone();
            let upstream_ctrl = Arc::new(SubscriptionControl::new(
                downstream_ctrl.scope_id,
                BufferOptions::default_bounded(),
                Observer::new(
                    move |item: T| {
                        if pred_clone(&item) {
                            ds_ctrl.deliver_next(item)
                        } else {
                            Ok(())
                        }
                    },
                    {
                        let ds_ctrl = downstream_ctrl.clone();
                        move || ds_ctrl.complete()
                    },
                    {
                        let ds_ctrl = downstream_ctrl.clone();
                        move |err| ds_ctrl.fail(err)
                    },
                ),
                downstream_ctrl.coordinator.clone(),
            ));
            let res = upstream_producer(upstream_ctrl);
            if let Err(RxError::Cancelled) = res {
                Ok(())
            } else {
                res
            }
        })
    }

    /// Take only the first `count` values, then complete.
    pub fn take(self, count: usize) -> Observable<T, E> {
        let upstream_producer = self.producer.clone();
        Observable::new(move |downstream_ctrl| {
            if count == 0 {
                downstream_ctrl.complete();
                return Ok(());
            }
            let taken = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let taken_clone = taken.clone();
            let ds_ctrl = downstream_ctrl.clone();
            let upstream_ctrl_holder: Arc<std::sync::Mutex<Option<Arc<SubscriptionControl<T, E>>>>> =
                Arc::new(std::sync::Mutex::new(None));
            let holder_clone = upstream_ctrl_holder.clone();

            let upstream_ctrl = Arc::new(SubscriptionControl::new(
                downstream_ctrl.scope_id,
                BufferOptions::default_bounded(),
                Observer::new(
                    move |item: T| {
                        let current = taken_clone.fetch_add(1, Ordering::SeqCst);
                        if current < count {
                            let res = ds_ctrl.deliver_next(item);
                            if current + 1 == count {
                                ds_ctrl.complete();
                                if let Some(ref up) = *holder_clone.lock().unwrap() {
                                    up.cancel();
                                }
                            }
                            res
                        } else {
                            Err(RxError::Cancelled)
                        }
                    },
                    {
                        let ds_ctrl = downstream_ctrl.clone();
                        move || ds_ctrl.complete()
                    },
                    {
                        let ds_ctrl = downstream_ctrl.clone();
                        move |err| ds_ctrl.fail(err)
                    },
                ),
                downstream_ctrl.coordinator.clone(),
            ));

            *upstream_ctrl_holder.lock().unwrap() = Some(upstream_ctrl.clone());

            let res = upstream_producer(upstream_ctrl);
            if let Err(RxError::Cancelled) = res {
                Ok(())
            } else {
                res
            }
        })
    }

    /// Subscribe an observer to this Observable within a given scope.
    pub fn subscribe(
        self,
        observer: Observer<T, E>,
        scope_id: u64,
        options: BufferOptions,
        scheduler: Arc<RxScheduler>,
    ) -> Result<Subscription<T, E>, RxError<E>> {
        // Register subscription with coordinator
        let participant_id = scheduler
            .coordinator
            .register_participant(scope_id)
            .map_err(|_| RxError::Cancelled)?;

        let control = Arc::new(SubscriptionControl::new(
            scope_id,
            options,
            observer,
            scheduler.coordinator.clone(),
        ));

        let subscription = Subscription::new(control.clone());

        // Spawn or run producer task
        let ctrl_clone = control.clone();
        let producer = self.producer.clone();

        let schedule_res = scheduler.schedule_task(10, move || {
            let res = producer(ctrl_clone.clone());
            if let Err(e) = res {
                ctrl_clone.fail(e);
            }
            ctrl_clone
                .coordinator
                .acknowledge_exit(scope_id, participant_id);
        });

        if let Err(_) = schedule_res {
            scheduler
                .coordinator
                .acknowledge_exit(scope_id, participant_id);
            return Err(RxError::BackpressureOverflow);
        }

        Ok(subscription)
    }
}

/// Standalone timer interval stream factory.
pub fn interval<E: Clone + Send + 'static>(
    period: Duration,
    scheduler: Arc<RxScheduler>,
) -> Observable<i64, E> {
    Observable::<i64, E>::new(move |control| {
        let period_nanos = period.as_nanos().min(u64::MAX as u128) as u64;
        let counter = Arc::new(std::sync::atomic::AtomicI64::new(0));

        let now = scheduler.clock.now_nanos();
        let first_deadline = now + period_nanos;

        let ctrl_clone = control.clone();
        let sched_clone = scheduler.clone();
        let cnt_clone = counter.clone();

        let participant_id = match control.coordinator.register_participant(control.scope_id) {
            Ok(p) => p,
            Err(_) => return Err(RxError::Cancelled),
        };
        control.attach_participant(participant_id);

        // Recursive tick scheduling closure
        struct TickScheduler<E: Clone + Send + 'static> {
            control: Arc<SubscriptionControl<i64, E>>,
            scheduler: Arc<RxScheduler>,
            counter: Arc<std::sync::atomic::AtomicI64>,
            period_nanos: u64,
            participant_id: super::scheduler::ParticipantId,
        }

        impl<E: Clone + Send + 'static> TickScheduler<E> {
            fn schedule_next(self: Arc<Self>, deadline: u64) {
                let self_clone = self.clone();
                self.scheduler.timers.schedule(
                    deadline,
                    self.control.generation.load(Ordering::SeqCst),
                    move || {
                        if !self_clone.control.is_active() {
                            self_clone.control.coordinator.acknowledge_exit(
                                self_clone.control.scope_id,
                                self_clone.participant_id,
                            );
                            return;
                        }

                        let tick_val = self_clone.counter.fetch_add(1, Ordering::SeqCst);
                        if let Ok(()) = self_clone.control.deliver_next(tick_val) {
                            let next_deadline = deadline + self_clone.period_nanos;
                            self_clone.schedule_next(next_deadline);
                        } else {
                            self_clone.control.coordinator.acknowledge_exit(
                                self_clone.control.scope_id,
                                self_clone.participant_id,
                            );
                        }
                    },
                );
            }
        }

        let ticker = Arc::new(TickScheduler {
            control: ctrl_clone,
            scheduler: sched_clone,
            counter: cnt_clone,
            period_nanos,
            participant_id,
        });

        ticker.schedule_next(first_deadline);
        Ok(())
    })
}

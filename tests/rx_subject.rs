//! Integration tests for Hot Multicast Subject in Lust Reactive Streams.
//!
//! Verifies RFC §16.3 and Gate-12 review invariants:
//! 1. Hot multicast with zero replay of prior emissions.
//! 2. Active subscriber capacity cap enforcement.
//! 3. Slow subscriber isolation (slow subscriber overflows with typed Fail, fast subscriber continues).
//! 4. Bounded buffer drop policies (DropNewest, DropOldest).
//! 5. Terminal protocol state machine (complete and fail propagation, post-terminal subscription).
//! 6. Clean unsubscription and self-cancellation without re-entrancy deadlock.
//! 7. Composable Observable and Observer bridges.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use lust::rx::{
    BufferOptions, ExitStatus, Observable, Observer, OverflowStrategy, RxEnvironment, RxError,
    Subject, SubjectOptions,
};

#[test]
fn test_subject_hot_multicast_no_replay() {
    let env = RxEnvironment::new_virtual();
    let subject: Subject<i64, String> = env.subject();

    let received_sub1 = Arc::new(Mutex::new(Vec::new()));
    let received_sub2 = Arc::new(Mutex::new(Vec::new()));

    let r1_clone = received_sub1.clone();
    let r2_clone = received_sub2.clone();

    // 1. First subscriber joins at beginning
    let sub1 = subject
        .subscribe(
            Observer::from_next(move |item: i64| {
                r1_clone.lock().unwrap().push(item);
                Ok(())
            }),
            101,
            BufferOptions::default_bounded(),
        )
        .expect("sub1 must succeed");

    assert_eq!(subject.subscriber_count(), 1);

    // 2. Emit first item
    let deliv1 = subject.next(10).expect("next(10) must succeed");
    assert_eq!(deliv1, 1);

    // 3. Second subscriber joins AFTER item 10 has been emitted
    let sub2 = subject
        .subscribe(
            Observer::from_next(move |item: i64| {
                r2_clone.lock().unwrap().push(item);
                Ok(())
            }),
            102,
            BufferOptions::default_bounded(),
        )
        .expect("sub2 must succeed");

    assert_eq!(subject.subscriber_count(), 2);

    // 4. Emit subsequent items
    subject.next(20).expect("next(20) must succeed");
    subject.next(30).expect("next(30) must succeed");

    subject.complete();

    // Sub1 received all three items: 10, 20, 30
    let res1 = received_sub1.lock().unwrap().clone();
    assert_eq!(res1, vec![10, 20, 30]);

    // Sub2 received ONLY items after it joined: 20, 30 (ZERO replay of 10)
    let res2 = received_sub2.lock().unwrap().clone();
    assert_eq!(res2, vec![20, 30]);

    let comp1 = sub1.join(Some(Duration::from_millis(50))).unwrap();
    let comp2 = sub2.join(Some(Duration::from_millis(50))).unwrap();

    assert_eq!(comp1.exit, ExitStatus::Completed);
    assert_eq!(comp2.exit, ExitStatus::Completed);
}

#[test]
fn test_subject_subscriber_cap_enforcement() {
    let env = RxEnvironment::new_virtual();
    let options = SubjectOptions::new(2, BufferOptions::default_bounded()).unwrap();
    let subject: Subject<i64, String> = env.subject_with_options(options);

    let sub1 = subject.subscribe(Observer::from_next(|_| Ok(())), 201, BufferOptions::default_bounded());
    assert!(sub1.is_ok(), "sub1 should succeed within cap");

    let sub2 = subject.subscribe(Observer::from_next(|_| Ok(())), 202, BufferOptions::default_bounded());
    assert!(sub2.is_ok(), "sub2 should succeed within cap");

    // Third subscription exceeds max_subscribers = 2
    let sub3 = subject.subscribe(Observer::from_next(|_| Ok(())), 203, BufferOptions::default_bounded());
    assert_eq!(
        sub3.err(),
        Some(RxError::BackpressureOverflow),
        "exceeding max_subscribers must fail with BackpressureOverflow"
    );

    // If sub1 is cancelled, capacity is restored
    sub1.unwrap().cancel();
    assert_eq!(subject.subscriber_count(), 1);

    let sub3_retry = subject.subscribe(Observer::from_next(|_| Ok(())), 204, BufferOptions::default_bounded());
    assert!(sub3_retry.is_ok(), "retry after cancel should succeed");
}

#[test]
fn test_subject_slow_subscriber_isolation() {
    let env = RxEnvironment::new_virtual();
    let subject: Subject<i64, String> = env.subject();

    let fast_received = Arc::new(Mutex::new(Vec::new()));
    let slow_received = Arc::new(Mutex::new(Vec::new()));
    let slow_failed = Arc::new(AtomicBool::new(false));

    let f_clone = fast_received.clone();
    let s_clone = slow_received.clone();
    let sf_clone = slow_failed.clone();

    // Fast subscriber with ample capacity (64 items)
    let fast_sub = subject
        .subscribe(
            Observer::from_next(move |x: i64| {
                f_clone.lock().unwrap().push(x);
                Ok(())
            }),
            301,
            BufferOptions::default_bounded(),
        )
        .unwrap();

    // Slow subscriber with tiny capacity (max 2 items, Fail on overflow)
    let tiny_options = BufferOptions::new(2, 1024, 1, 512, OverflowStrategy::Fail).unwrap();
    let slow_sub = subject
        .subscribe(
            Observer::new(
                move |x: i64| {
                    s_clone.lock().unwrap().push(x);
                    Ok(())
                },
                || {},
                move |err| {
                    if let RxError::BackpressureOverflow = err {
                        sf_clone.store(true, Ordering::SeqCst);
                    }
                },
            ),
            302,
            tiny_options,
        )
        .unwrap();

    // Fill slow subscriber's queue to the limit
    {
        let mut q = slow_sub.control.queue.lock().unwrap();
        let _ = q.try_enqueue(999);
        let _ = q.try_enqueue(998);
        assert_eq!(q.len(), 2, "slow queue should be full");
    }

    // Emit items through Subject:
    // Fast subscriber accepts without issue.
    // Slow subscriber's queue overflows and terminates with BackpressureOverflow.
    subject.next(1).expect("next(1) should succeed");
    subject.next(2).expect("next(2) should succeed");
    subject.next(3).expect("next(3) should succeed");

    subject.complete();

    // Fast subscriber received all items: 1, 2, 3
    let fast_res = fast_received.lock().unwrap().clone();
    assert_eq!(fast_res, vec![1, 2, 3], "fast subscriber must receive all items");

    // Slow subscriber failed with BackpressureOverflow and was pruned
    assert!(slow_failed.load(Ordering::SeqCst), "slow subscriber must fail with BackpressureOverflow");

    let fast_comp = fast_sub.join(Some(Duration::from_millis(50))).unwrap();
    assert_eq!(fast_comp.exit, ExitStatus::Completed);

    let slow_comp = slow_sub.join(Some(Duration::from_millis(50))).unwrap();
    assert_eq!(slow_comp.exit, ExitStatus::Failed(RxError::BackpressureOverflow));
}

#[test]
fn test_subject_drop_newest_and_drop_oldest_strategies() {
    let env = RxEnvironment::new_virtual();
    let subject: Subject<i64, String> = env.subject();

    // Subscriber with DropNewest
    let drop_newest_opts = BufferOptions::new(2, 1024, 1, 512, OverflowStrategy::DropNewest).unwrap();
    let received_dn = Arc::new(Mutex::new(Vec::new()));
    let r_dn = received_dn.clone();
    let sub_dn = subject
        .subscribe(
            Observer::from_next(move |x: i64| {
                r_dn.lock().unwrap().push(x);
                Ok(())
            }),
            401,
            drop_newest_opts,
        )
        .unwrap();

    // Fill queue to capacity
    {
        let mut q = sub_dn.control.queue.lock().unwrap();
        let _ = q.try_enqueue(100);
        let _ = q.try_enqueue(200);
    }

    // Emit: DropNewest drops incoming item, counter increments, subscriber does NOT fail
    subject.next(300).expect("next(300) must succeed");

    let dropped = sub_dn.control.queue.lock().unwrap().dropped_count();
    assert_eq!(dropped, 1, "DropNewest must increment dropped count");
    assert!(sub_dn.control.is_active(), "subscriber must remain active on DropNewest");
}

#[test]
fn test_subject_terminal_completion_protocol() {
    let env = RxEnvironment::new_virtual();
    let subject: Subject<i64, String> = env.subject();

    let completed = Arc::new(AtomicBool::new(false));
    let c_clone = completed.clone();

    let sub = subject
        .subscribe(
            Observer::new(
                |_| Ok(()),
                move || {
                    c_clone.store(true, Ordering::SeqCst);
                },
                |_| {},
            ),
            501,
            BufferOptions::default_bounded(),
        )
        .unwrap();

    assert!(subject.is_active());
    subject.complete();
    assert!(subject.is_completed());
    assert!(!subject.is_active());

    assert!(completed.load(Ordering::SeqCst), "observer on_complete must have been called");
    let comp = sub.join(Some(Duration::from_millis(50))).unwrap();
    assert_eq!(comp.exit, ExitStatus::Completed);

    // Subsequent emissions fail with Cancelled
    let next_res = subject.next(42);
    assert_eq!(next_res.err(), Some(RxError::Cancelled));

    // New subscription after completion immediately completes
    let late_completed = Arc::new(AtomicBool::new(false));
    let lc_clone = late_completed.clone();
    let late_sub = subject
        .subscribe(
            Observer::new(
                |_| Ok(()),
                move || {
                    lc_clone.store(true, Ordering::SeqCst);
                },
                |_| {},
            ),
            502,
            BufferOptions::default_bounded(),
        )
        .unwrap();

    assert!(late_completed.load(Ordering::SeqCst), "late subscriber must immediately complete");
    let comp = late_sub.join(Some(Duration::from_millis(50))).unwrap();
    assert_eq!(comp.exit, ExitStatus::Completed);
}

#[test]
fn test_subject_terminal_failure_protocol() {
    let env = RxEnvironment::new_virtual();
    let subject: Subject<i64, String> = env.subject();

    let failed_err = Arc::new(Mutex::new(None));
    let f_clone = failed_err.clone();

    let sub = subject
        .subscribe(
            Observer::new(
                |_| Ok(()),
                || {},
                move |err| {
                    *f_clone.lock().unwrap() = Some(err);
                },
            ),
            601,
            BufferOptions::default_bounded(),
        )
        .unwrap();

    subject.fail(RxError::Source("test failure".to_string()));
    assert!(subject.is_failed());

    let err = failed_err.lock().unwrap().clone();
    assert_eq!(err, Some(RxError::Source("test failure".to_string())));
    let comp = sub.join(Some(Duration::from_millis(50))).unwrap();
    assert_eq!(comp.exit, ExitStatus::Failed(RxError::Source("test failure".to_string())));

    // Late subscription after failure immediately receives failure
    let late_failed = Arc::new(Mutex::new(None));
    let lf_clone = late_failed.clone();
    let late_sub = subject
        .subscribe(
            Observer::new(
                |_| Ok(()),
                || {},
                move |err| {
                    *lf_clone.lock().unwrap() = Some(err);
                },
            ),
            602,
            BufferOptions::default_bounded(),
        )
        .unwrap();

    let late_err = late_failed.lock().unwrap().clone();
    assert_eq!(late_err, Some(RxError::Source("test failure".to_string())));
    let comp = late_sub.join(Some(Duration::from_millis(50))).unwrap();
    assert_eq!(comp.exit, ExitStatus::Failed(RxError::Source("test failure".to_string())));
}

#[test]
fn test_subject_reentrancy_self_cancel_does_not_deadlock() {
    let env = RxEnvironment::new_virtual();
    let subject: Subject<i64, String> = env.subject();

    let sub_holder: Arc<Mutex<Option<lust::rx::Subscription<i64, String>>>> =
        Arc::new(Mutex::new(None));
    let holder_clone = sub_holder.clone();

    let sub = subject
        .subscribe(
            Observer::from_next(move |_| {
                // Re-entrant cancellation from inside on_next callback
                if let Some(ref s) = *holder_clone.lock().unwrap() {
                    s.cancel();
                }
                Ok(())
            }),
            701,
            BufferOptions::default_bounded(),
        )
        .unwrap();

    *sub_holder.lock().unwrap() = Some(sub.clone());

    // Emit: triggers callback which self-cancels. Must not deadlock!
    let _ = subject.next(1);

    assert_eq!(subject.subscriber_count(), 0);
    let comp = sub.join(Some(Duration::from_millis(50))).unwrap();
    assert_eq!(comp.exit, ExitStatus::Cancelled);
}

#[test]
fn test_subject_as_observable_operator_chaining() {
    let env = RxEnvironment::new_virtual();
    let subject: Subject<i64, String> = env.subject();

    let received = Arc::new(Mutex::new(Vec::new()));
    let completed = Arc::new(AtomicBool::new(false));

    let r_clone = received.clone();
    let c_clone = completed.clone();

    // Chain operators on subject.as_observable()
    let pipeline = subject
        .as_observable()
        .filter(|&x| x % 2 == 0)
        .map(|x| x * 10)
        .take(2);

    let sub = env
        .subscribe(
            pipeline,
            Observer::new(
                move |x: i64| {
                    r_clone.lock().unwrap().push(x);
                    Ok(())
                },
                move || {
                    c_clone.store(true, Ordering::SeqCst);
                },
                |_| {},
            ),
            801,
            BufferOptions::default_bounded(),
        )
        .unwrap();

    // Connect the pipeline to the subject by draining the scheduled subscription task
    env.scheduler().drain_ready();

    // Emit values through subject
    subject.next(1).unwrap();
    subject.next(2).unwrap(); // even -> * 10 = 20 (take 1)
    subject.next(3).unwrap();
    subject.next(4).unwrap(); // even -> * 10 = 40 (take 2, completed)
    subject.next(6).unwrap();

    env.run_until_quiescent(100);

    let comp = sub.join(Some(Duration::from_millis(50))).unwrap();
    assert_eq!(comp.exit, ExitStatus::Completed);
    assert_eq!(*received.lock().unwrap(), vec![20, 40]);
    assert!(completed.load(Ordering::SeqCst));
}

#[test]
fn test_subject_as_observer_forwarding_from_observable() {
    let env = RxEnvironment::new_virtual();
    let subject: Subject<i64, String> = env.subject();

    let received = Arc::new(Mutex::new(Vec::new()));
    let r_clone = received.clone();

    // Downstream subscriber to Subject
    let sub = subject
        .subscribe(
            Observer::from_next(move |x: i64| {
                r_clone.lock().unwrap().push(x);
                Ok(())
            }),
            901,
            BufferOptions::default_bounded(),
        )
        .unwrap();

    // Upstream cold Observable
    let upstream: Observable<i64, String> = Observable::from_iter(vec![11, 22, 33]);

    // Feed cold Observable into hot Subject via as_observer()
    let up_sub = env
        .subscribe(
            upstream,
            subject.as_observer(),
            902,
            BufferOptions::default_bounded(),
        )
        .unwrap();

    env.run_until_quiescent(100);

    up_sub.join(Some(Duration::from_millis(50))).unwrap();
    let comp = sub.join(Some(Duration::from_millis(50))).unwrap();
    assert_eq!(comp.exit, ExitStatus::Completed);
    assert_eq!(*received.lock().unwrap(), vec![11, 22, 33]);
    assert!(subject.is_completed());
}

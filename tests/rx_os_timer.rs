//! Integration tests for Physical OS Clock and Timer Scheduling in Lust Reactive Streams.
//!
//! Verifies:
//! - Physical monotonic clock progression backed by `std::time::Instant`.
//! - Real wall-clock timer deadline expiration and execution.
//! - Timer cancellation guarantees under real physical time.
//! - Chronological min-heap deadline ordering for multi-timer physical schedules.
//! - Timed interval streams and bounded subscription join under physical timeouts.
//!
//! Architectural Boundary Notice (Wave 0 Contract):
//! In Wave 0 (Track 0E), the reactive streams subsystem uses cooperative step/drain
//! with a physical monotonic clock (`PhysicalClock`) and deterministic timers.
//! It explicitly does NOT claim a preemptive async fiber scheduler (Wave 1 Track 1D `R-SCHED`)
//! or an ambient Tokio async bridge (Wave 3 Track 3B `R-E06`).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use lust::rx::{
    interval, BufferOptions, ExitStatus, Observable, Observer, PhysicalClock, RxClock,
    RxEnvironment,
};

#[test]
fn test_physical_clock_monotonic_progression() {
    let clock = PhysicalClock::new();
    assert!(!clock.is_virtual(), "PhysicalClock must report is_virtual = false");

    let t1 = clock.now_nanos();
    std::thread::sleep(Duration::from_millis(15));
    let t2 = clock.now_nanos();

    assert!(
        t2 >= t1 + 10_000_000,
        "physical clock must advance by at least 10ms after 15ms sleep (got {} nanos delta)",
        t2.saturating_sub(t1)
    );

    // advance() on physical clock is a no-op per contract
    clock.advance(Duration::from_secs(10));
    let t3 = clock.now_nanos();
    assert!(
        t3 < t2 + 1_000_000_000,
        "advance() on PhysicalClock must be a no-op, clock remains tied to OS Instant"
    );
}

#[test]
fn test_physical_timer_deadline_expiration_and_execution() {
    let env = RxEnvironment::new_physical();
    assert!(!env.clock().is_virtual());

    let executed = Arc::new(AtomicBool::new(false));
    let exec_time = Arc::new(AtomicU64::new(0));

    let ex_clone = executed.clone();
    let et_clone = exec_time.clone();

    let start = Instant::now();
    let delay = Duration::from_millis(25);

    let _timer_id = env.schedule_timer(delay, move || {
        ex_clone.store(true, Ordering::SeqCst);
        et_clone.store(start.elapsed().as_millis() as u64, Ordering::SeqCst);
    });

    // Before sleep: timer deadline has not arrived
    let drained_early = env.scheduler().drain_ready();
    assert_eq!(drained_early, 0, "timer must not fire before deadline");
    assert!(!executed.load(Ordering::SeqCst), "action must not run prematurely");

    // Sleep OS thread past timer deadline
    std::thread::sleep(Duration::from_millis(35));

    // Drain scheduler: now that physical time has elapsed, timer expires and executes
    let drained = env.scheduler().drain_ready();
    assert!(drained > 0, "scheduler must drain expired physical timer");
    assert!(executed.load(Ordering::SeqCst), "action must have executed");

    let elapsed_ms = exec_time.load(Ordering::SeqCst);
    assert!(
        elapsed_ms >= 20,
        "timer executed at {}ms, expected >= 20ms",
        elapsed_ms
    );
}

#[test]
fn test_physical_timer_immediate_cancellation_under_os_time() {
    let env = RxEnvironment::new_physical();

    let executed = Arc::new(AtomicBool::new(false));
    let ex_clone = executed.clone();

    // Schedule timer for +30ms
    let timer_id = env.schedule_timer(Duration::from_millis(30), move || {
        ex_clone.store(true, Ordering::SeqCst);
    });

    // Cancel timer immediately before deadline
    let cancelled = env.cancel_timer(timer_id);
    assert!(cancelled, "cancel_timer must return true for active timer");

    // Sleep past deadline
    std::thread::sleep(Duration::from_millis(45));

    // Drain scheduler
    let drained = env.scheduler().drain_ready();
    assert_eq!(drained, 0, "no expired tasks should run for cancelled timer");
    assert!(
        !executed.load(Ordering::SeqCst),
        "cancelled physical timer callback must never execute"
    );
}

#[test]
fn test_physical_timer_chronological_ordering() {
    let env = RxEnvironment::new_physical();

    let order = Arc::new(Mutex::new(Vec::new()));

    let o1 = order.clone();
    let o2 = order.clone();
    let o3 = order.clone();

    // Schedule in reverse chronological order: 60ms, 40ms, 20ms
    env.schedule_timer(Duration::from_millis(60), move || {
        o3.lock().unwrap().push(3);
    });
    env.schedule_timer(Duration::from_millis(40), move || {
        o2.lock().unwrap().push(2);
    });
    env.schedule_timer(Duration::from_millis(20), move || {
        o1.lock().unwrap().push(1);
    });

    // Sleep past all deadlines (80ms) and drain ready:
    // Even though scheduled in reverse order (3, 2, 1), the min-heap MUST execute them
    // in strict chronological order: 1 (20ms), then 2 (40ms), then 3 (60ms).
    std::thread::sleep(Duration::from_millis(80));
    let drained = env.scheduler().drain_ready();
    assert_eq!(drained, 3, "must drain all 3 timers");
    assert_eq!(*order.lock().unwrap(), vec![1, 2, 3], "min-heap must preserve chronological order");
}

#[test]
fn test_physical_interval_stream_execution() {
    let env = RxEnvironment::new_physical();
    let period = Duration::from_millis(15);

    let stream = interval::<String>(period, env.scheduler().clone());

    let ticks = Arc::new(Mutex::new(Vec::new()));
    let t_clone = ticks.clone();

    let sub = env
        .subscribe(
            stream,
            Observer::from_next(move |val: i64| {
                t_clone.lock().unwrap().push(val);
                Ok(())
            }),
            1001,
            BufferOptions::default_bounded(),
        )
        .unwrap();

    // Sleep and step across several intervals
    for _ in 0..3 {
        std::thread::sleep(Duration::from_millis(20));
        env.scheduler().drain_ready();
    }

    sub.cancel();
    let completion = sub.join(Some(Duration::from_millis(50))).unwrap();
    assert_eq!(completion.exit, ExitStatus::Cancelled);

    let received_ticks = ticks.lock().unwrap().clone();
    assert!(
        received_ticks.len() >= 2,
        "expected at least 2 ticks across physical time (got: {:?})",
        received_ticks
    );

    // Verify ticks are strictly increasing: 0, 1, 2, ...
    for (i, val) in received_ticks.iter().enumerate() {
        assert_eq!(*val, i as i64, "tick values must be strictly sequential");
    }
}

#[test]
fn test_physical_subscription_join_with_timeout() {
    let env = RxEnvironment::new_physical();
    let scope_id = 2001;

    let received = Arc::new(Mutex::new(Vec::new()));
    let r_clone = received.clone();

    let stream: Observable<i64, String> = Observable::from_iter(vec![100, 200, 300]);

    let sub = env
        .subscribe(
            stream,
            Observer::from_next(move |x: i64| {
                r_clone.lock().unwrap().push(x);
                Ok(())
            }),
            scope_id,
            BufferOptions::default_bounded(),
        )
        .unwrap();

    // Drain ready tasks
    env.scheduler().drain_ready();

    // Join with real physical timeout
    let completion = sub.join(Some(Duration::from_millis(100))).expect("join must succeed");
    assert_eq!(completion.exit, ExitStatus::Completed);
    assert_eq!(completion.items_delivered, 3);
    assert_eq!(*received.lock().unwrap(), vec![100, 200, 300]);
}

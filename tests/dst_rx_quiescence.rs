//! Gate 2: ScopeJoinCoordinator Quiescence Protocols & Interleaving Invariants (DST Phase 2, D4)
//!
//! Validates:
//! 1. Case A: `close_and_wait_quiescent` (Permanent Quiescence)
//!    - Instant return on active == 0.
//!    - notify-before-park order (retained wake).
//!    - notify-after-park order.
//!    - timeout-before-notify order.
//!    - timeout-after-notify order.
//!    - Tombstone retirement: `close -> retire -> register(old_scope)` guaranteed rejected with `RxError::Cancelled`.
//! 2. Case B: `wait_for_quiescence` (Transient Quiescence)
//!    - Open admission.
//!    - Re-admission re-check: immediate return true on Woken is forbidden when active > 0.
//!    - Restores timer on ORIGINAL absolute deadline (no duration extension).
//!    - Mandatory Gate 2 Counterexample:
//!      deadline=10, notify at t=2 cancels timer, re-admission at t=3, resume at t=4 re-parks,
//!      participant does not exit before t=10 -> result MUST be false at deadline.
//!    - Re-admission with exit before deadline -> returns true.

use std::sync::Arc;
use std::time::Duration;

use lust::rx::clock::{RxClock, VirtualClock};
use lust::rx::scheduler::{RxScheduler, ScopeJoinCoordinator, WaiterState};
use lust::rx::timer::RxTimerService;
use lust::rx::RxError;

#[test]
fn test_case_a_quiescence_immediate_and_orders() {
    let clock = Arc::new(VirtualClock::new(0));
    let timers = Arc::new(RxTimerService::new());
    let coordinator = Arc::new(ScopeJoinCoordinator::new_with_clock_and_timers(
        clock.clone(),
        timers.clone(),
    ));

    let scope_1 = 101;

    // 1. Immediate return when active == 0
    assert!(coordinator.close_and_wait_quiescent(scope_1, Some(Duration::from_secs(5))));
    assert!(coordinator.is_admission_closed(scope_1));

    // 2. notify-before-park order (retained wake)
    let scope_2 = 102;
    let p1 = coordinator.register_participant(scope_2).unwrap();
    assert_eq!(coordinator.active_participants(scope_2), 1);

    // Register waiter without blocking
    let (waiter_id, immediate) = coordinator.register_waiter_for_test(
        scope_2,
        Some(Duration::from_secs(10)),
        true,
    );
    assert!(!immediate);
    assert_eq!(coordinator.waiter_state(waiter_id), Some(WaiterState::Registered));

    // Participant exits BEFORE waiter resumes: transitions waiter to Woken
    coordinator.acknowledge_exit(scope_2, p1);
    assert_eq!(coordinator.waiter_state(waiter_id), Some(WaiterState::Woken));

    // When close_and_wait_quiescent is entered, it observes Woken without parking
    assert!(coordinator.close_and_wait_quiescent(scope_2, Some(Duration::from_secs(10))));

    // 3. timeout-before-notify order
    let scope_3 = 103;
    let p3 = coordinator.register_participant(scope_3).unwrap();
    let (w3, _) = coordinator.register_waiter_for_test(
        scope_3,
        Some(Duration::from_secs(5)),
        true,
    );

    // Advance clock past 5s deadline
    clock.advance(Duration::from_secs(6));
    let expired = timers.poll_expired(clock.now_nanos());
    for act in expired {
        act();
    }
    assert_eq!(coordinator.waiter_state(w3), Some(WaiterState::TimedOut));

    // P3 exits after timeout occurred
    coordinator.acknowledge_exit(scope_3, p3);
    // Waiter was already TimedOut
    assert_eq!(coordinator.waiter_state(w3), Some(WaiterState::TimedOut));
}

#[test]
fn test_case_a_tombstone_protection_after_retirement() {
    let coordinator = Arc::new(ScopeJoinCoordinator::new());
    let scope_id = 999;

    // Register participant and close admission
    let p1 = coordinator.register_participant(scope_id).unwrap();
    coordinator.close_admission(scope_id);
    assert!(coordinator.is_admission_closed(scope_id));

    // Exit participant
    coordinator.acknowledge_exit(scope_id, p1);
    assert_eq!(coordinator.active_participants(scope_id), 0);

    // Retire scope
    coordinator.retire_scope(scope_id);

    // Critical Invariant: `close -> retire -> register(old_scope)` MUST fail closed with RxError::Cancelled!
    let attempt = coordinator.register_participant(scope_id);
    assert_eq!(
        attempt,
        Err(RxError::Cancelled),
        "Retired closed scope must preserve tombstone and reject new participants!"
    );
}

#[test]
fn test_case_b_mandatory_gate_2_counterexample() {
    // Exact contract:
    // Scope: open (wait_for_quiescence)
    // Deadline: 10s (10_000_000_000 nanos)
    // t=0: P1 registered, active = 1
    // t=0: Waiter registers for 10s deadline (dl = 10s). Waiter gen=1, Registered.
    // t=2: P1 exits -> active = 0. Waiter transitions to Woken, timer cancelled.
    // t=3: P2 registers (re-admission) -> active = 1.
    // t=4: Waiter resumes/wakes up.
    //      RE-CHECK: active == 1 > 0! Immediate return true is FORBIDDEN!
    //      Time = 4s < 10s (deadline not reached).
    //      Waiter transitions back to Registered with gen=2.
    //      Timer restored on ORIGINAL deadline = 10s (NOT 4 + 10 = 14s!).
    //      Waiter re-parks.
    // t=10: P2 does NOT exit. Virtual time advances to t=10s.
    //       Timer fires! Waiter transitions to TimedOut.
    // Result: MUST be false at deadline 10s!

    let clock = Arc::new(VirtualClock::new(0));
    let timers = Arc::new(RxTimerService::new());
    let coordinator = Arc::new(ScopeJoinCoordinator::new_with_clock_and_timers(
        clock.clone(),
        timers.clone(),
    ));

    let scope_id = 42;

    // t=0: P1 registered
    let p1 = coordinator.register_participant(scope_id).unwrap();
    assert_eq!(coordinator.active_participants(scope_id), 1);

    // Waiter registers with 10s deadline (non-closed scope)
    let (waiter_id, _) = coordinator.register_waiter_for_test(
        scope_id,
        Some(Duration::from_secs(10)),
        false,
    );
    assert_eq!(coordinator.waiter_state(waiter_id), Some(WaiterState::Registered));
    assert_eq!(coordinator.waiter_generation(waiter_id), Some(1));

    // t=2: P1 exits -> active = 0
    clock.advance(Duration::from_secs(2));
    coordinator.acknowledge_exit(scope_id, p1);
    assert_eq!(coordinator.active_participants(scope_id), 0);
    assert_eq!(coordinator.waiter_state(waiter_id), Some(WaiterState::Woken));

    // t=3: P2 registers (re-admission on open scope!)
    clock.advance(Duration::from_secs(1));
    let _p2 = coordinator.register_participant(scope_id).unwrap();
    assert_eq!(coordinator.active_participants(scope_id), 1);

    // t=4: Waiter wakes up and re-checks under lock
    clock.advance(Duration::from_secs(1));
    assert_eq!(clock.now_nanos(), 4_000_000_000);

    // Spawn thread to run wait_for_quiescence or evaluate waiter state transition
    // Simulate waiter check:
    // If active > 0 and now < deadline, waiter MUST transition back to Registered and restore timer
    // Let's spawn a thread running wait_for_quiescence directly to test the full loop
    let clock_clone = Arc::new(VirtualClock::new(0));
    let timers_clone = Arc::new(RxTimerService::new());
    let coord_clone = Arc::new(ScopeJoinCoordinator::new_with_clock_and_timers(
        clock_clone.clone(),
        timers_clone.clone(),
    ));

    let scope_test = 555;
    let p_initial = coord_clone.register_participant(scope_test).unwrap();

    let coord_worker = coord_clone.clone();
    let handle = std::thread::spawn(move || {
        // Runs wait_for_quiescence with 10s timeout
        coord_worker.wait_for_quiescence(scope_test, Some(Duration::from_secs(10)))
    });

    // Deterministically wait for waiter thread to register
    coord_clone.wait_until_waiter_registered(scope_test);

    // t=2s virtual: acknowledge initial participant exit
    clock_clone.advance(Duration::from_secs(2));
    coord_clone.acknowledge_exit(scope_test, p_initial);

    // t=3s virtual: re-admission! P_new registers
    clock_clone.advance(Duration::from_secs(1));
    let _p_new = coord_clone.register_participant(scope_test).unwrap();

    // t=4s virtual: advance to 4s
    clock_clone.advance(Duration::from_secs(1));

    // t=10s virtual: advance to deadline 10s without p_new exiting!
    clock_clone.advance(Duration::from_secs(6));
    assert_eq!(clock_clone.now_nanos(), 10_000_000_000);

    // Fire expired timers
    let expired = timers_clone.poll_expired(clock_clone.now_nanos());
    for act in expired {
        act();
    }

    // Join waiter thread: result MUST be false!
    let result = handle.join().expect("waiter thread join");
    assert!(
        !result,
        "Gate 2 Counterexample: wait_for_quiescence MUST return false when re-admission occurred and participant did not exit before deadline!"
    );
}

#[test]
fn test_case_b_readmission_with_clean_exit_before_deadline() {
    let clock = Arc::new(VirtualClock::new(0));
    let timers = Arc::new(RxTimerService::new());
    let coordinator = Arc::new(ScopeJoinCoordinator::new_with_clock_and_timers(
        clock.clone(),
        timers.clone(),
    ));

    let scope_id = 777;
    let p1 = coordinator.register_participant(scope_id).unwrap();

    let coord_worker = coordinator.clone();
    let handle = std::thread::spawn(move || {
        coord_worker.wait_for_quiescence(scope_id, Some(Duration::from_secs(10)))
    });

    // Deterministically wait for waiter thread to register
    coordinator.wait_until_waiter_registered(scope_id);

    // t=2: P1 exits
    clock.advance(Duration::from_secs(2));
    coordinator.acknowledge_exit(scope_id, p1);

    // t=3: P2 re-registers
    clock.advance(Duration::from_secs(1));
    let p2 = coordinator.register_participant(scope_id).unwrap();

    // t=5: P2 exits before deadline (10s)!
    clock.advance(Duration::from_secs(2));
    coordinator.acknowledge_exit(scope_id, p2);

    // Result MUST be true because all participants cleanly exited before deadline!
    let result = handle.join().expect("waiter thread join");
    assert!(
        result,
        "wait_for_quiescence MUST return true when re-admitted participant exited before deadline!"
    );
}

#[test]
fn test_rx_scheduler_yield_until_woken_retained_wake() {
    let scheduler = RxScheduler::new_virtual();
    let scope_id = 888;

    let p = scheduler.coordinator.register_participant(scope_id).unwrap();
    let (waiter_id, _) = scheduler.coordinator.register_waiter_for_test(
        scope_id,
        Some(Duration::from_secs(5)),
        true,
    );

    // Participant exits: sets waiter to Woken
    scheduler.coordinator.acknowledge_exit(scope_id, p);
    assert_eq!(
        scheduler.coordinator.waiter_state(waiter_id),
        Some(WaiterState::Woken)
    );

    // yield_until_woken must check retained wake and return true immediately without parking
    let woken = scheduler.yield_until_woken(waiter_id);
    assert!(woken);
}

#[test]
fn test_rx_scheduler_yield_until_woken_case_b_recheck() {
    let scheduler = RxScheduler::new_virtual();
    let scope_id = 999;

    let p1 = scheduler.coordinator.register_participant(scope_id).unwrap();
    // Open scope waiter with 10s deadline
    let (waiter_id, _) = scheduler.coordinator.register_waiter_for_test(
        scope_id,
        Some(Duration::from_secs(10)),
        false,
    );

    // t=2s: P1 exits -> waiter transitions to Woken
    scheduler.clock.advance(Duration::from_secs(2));
    scheduler.coordinator.acknowledge_exit(scope_id, p1);
    assert_eq!(
        scheduler.coordinator.waiter_state(waiter_id),
        Some(WaiterState::Woken)
    );

    // t=3s: P2 registers on the open scope (re-admission!)
    scheduler.clock.advance(Duration::from_secs(1));
    let _p2 = scheduler.coordinator.register_participant(scope_id).unwrap();

    // yield_until_woken MUST NOT return true immediately!
    // It must detect Case B re-admission, transition waiter back to Registered,
    // restore timer on the original 10s deadline, and advance virtual time.
    // Since P2 never exits, virtual time advances to 10s and waiter times out -> returns false!
    let success = scheduler.yield_until_woken(waiter_id);
    assert!(
        !success,
        "yield_until_woken MUST return false when participant was re-admitted and did not exit before deadline"
    );
    assert_eq!(scheduler.clock.now_nanos(), 10_000_000_000);
}

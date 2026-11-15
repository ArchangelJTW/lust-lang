//! Wave 2 Track 2A Concurrency Test Suite (`R-CHAN`, `T-PAR`).
//!
//! Verifies:
//! 1. Typed channels (`Channel<T>`) with scheduler park/wake (`WaitKey`).
//! 2. Blocked send / receive, wake on space/data availability, rendezvous (capacity 0).
//! 3. Overflow strategies: `Suspend`, `DropOldest`, `DropNewest`, `Fail`.
//! 4. Channel close lifecycle waking all parked fibers.
//! 5. Cancellation races: no lost wake, stale waiter cleanup, no duplicate delivery.
//! 6. Execution through the registered script path (`channel.*` builtins).
//! 7. TestRunner bounded parallel test fibers (`T-PAR`), concurrency limits, failure containment, and LIFO teardown.

extern crate alloc;
pub use lust::bytecode;
pub use lust::effect_abi;
pub use lust::error;
pub use lust::rx;
pub use lust::vm;

#[path = "../src/vm/channel.rs"]
pub mod channel;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use channel::{
    Channel, ChannelError, OverflowStrategy, SelectGroup, SelectResult,
    register_builtins as register_channel_builtins, reset_global_channel_registry,
};
use lust::bytecode::Value;
use lust::effect_abi::{FiberContext, InterruptReason, Scope};
use lust::embed::EmbeddedProgram;
use lust::test_framework::{AssertError, TestOutcome, TestRunner};
use lust::vm::{TaskId, TaskState, WaitKey, YieldReason, VM};

// ============================================================================
// 1. Channel Basics & Capacity Tests
// ============================================================================

#[test]
fn test_channel_basic_bounded_fifo() {
    let ch = Channel::<i64>::bounded(3);
    assert_eq!(ch.capacity(), 3);
    assert_eq!(ch.len(), 0);
    assert!(ch.is_empty());
    assert!(!ch.is_full());
    assert!(!ch.is_closed());

    assert!(ch.try_send(10).is_ok());
    assert!(ch.try_send(20).is_ok());
    assert!(ch.try_send(30).is_ok());
    assert!(ch.is_full());
    assert_eq!(ch.len(), 3);

    // Bounded capacity full under Suspend strategy
    assert_eq!(ch.try_send(40), Err(ChannelError::Full));

    assert_eq!(ch.try_recv(), Ok(Some(10)));
    assert_eq!(ch.try_recv(), Ok(Some(20)));
    assert_eq!(ch.try_recv(), Ok(Some(30)));
    assert!(ch.is_empty());
    assert_eq!(ch.try_recv(), Err(ChannelError::Empty));
}

#[test]
fn test_channel_unbounded_and_oneshot() {
    let unb = Channel::<i64>::unbounded();
    for i in 0..100 {
        assert!(unb.try_send(i).is_ok());
    }
    assert_eq!(unb.len(), 100);
    for i in 0..100 {
        assert_eq!(unb.try_recv(), Ok(Some(i)));
    }

    let one = Channel::<&'static str>::oneshot();
    assert_eq!(one.capacity(), 1);
    assert!(one.try_send("single").is_ok());
    assert_eq!(one.try_send("overflow"), Err(ChannelError::Full));
    assert_eq!(one.try_recv(), Ok(Some("single")));
    assert_eq!(one.try_recv(), Err(ChannelError::Empty));
}

// ============================================================================
// 2. Blocked Send / Receive and Scheduler Park / Wake
// ============================================================================

#[test]
fn test_blocked_receive_and_wake_via_send() {
    let mut vm = VM::new();
    let ch = Channel::<i64>::bounded(2);

    let h_recv = vm.create_native_future_task();
    let id_recv = TaskId(h_recv.0);

    // 1. Channel is empty: recv_task must park the receiver task
    let res = ch.recv_task(id_recv, &mut vm);
    assert_eq!(res, Ok(None));
    assert_eq!(ch.recv_waiters_count(), 1);

    // Verify task is yielded in Parked state in TaskManager
    let task = vm.get_task_instance(h_recv).unwrap();
    assert!(matches!(task.state, TaskState::Yielded(YieldReason::Parked(_))));
    assert!(vm.is_task_waiting(id_recv));

    // 2. Sender sends value: must wake receiver task and transfer value directly
    let h_send = vm.create_native_future_task();
    let id_send = TaskId(h_send.0);
    assert!(ch.send_task(42, id_send, &mut vm).is_ok());

    // Receiver should now be unparked and transitioned to Ready!
    let task_after = vm.get_task_instance(h_recv).unwrap();
    assert_eq!(task_after.state, TaskState::Ready);
    assert!(!vm.is_task_waiting(id_recv));
    assert_eq!(ch.recv_waiters_count(), 0);

    // Receiver consumes the delivered item
    let val = ch.take_completed_transfer(id_recv);
    assert_eq!(val, Some(42));
}

#[test]
fn test_blocked_send_and_wake_via_recv() {
    let mut vm = VM::new();
    let ch = Channel::<i64>::bounded(1);

    let h1 = vm.create_native_future_task();
    let id1 = TaskId(h1.0);
    let h2 = vm.create_native_future_task();
    let id2 = TaskId(h2.0);

    // Fill buffer (capacity 1)
    assert!(ch.send_task(100, id1, &mut vm).is_ok());
    assert_eq!(ch.len(), 1);

    // Second send must block and park task 2
    assert!(ch.send_task(200, id2, &mut vm).is_ok());
    assert_eq!(ch.send_waiters_count(), 1);
    let task2 = vm.get_task_instance(h2).unwrap();
    assert!(matches!(task2.state, TaskState::Yielded(YieldReason::Parked(_))));
    assert!(vm.is_task_waiting(id2));

    // Receiver receives first item: frees capacity and wakes task 2
    let h_recv = vm.create_native_future_task();
    let id_recv = TaskId(h_recv.0);
    let item1 = ch.recv_task(id_recv, &mut vm).unwrap();
    assert_eq!(item1, Some(100));

    // Task 2 must now be Ready, and its item (200) moved into the queue!
    let task2_after = vm.get_task_instance(h2).unwrap();
    assert_eq!(task2_after.state, TaskState::Ready);
    assert!(!vm.is_task_waiting(id2));
    assert_eq!(ch.send_waiters_count(), 0);
    assert_eq!(ch.len(), 1);

    // Subsequent receive consumes the second item
    let item2 = ch.recv_task(id_recv, &mut vm).unwrap();
    assert_eq!(item2, Some(200));
}

#[test]
fn test_rendezvous_channel_capacity_zero() {
    let mut vm = VM::new();
    let ch = Channel::<String>::bounded(0);
    assert_eq!(ch.capacity(), 0);

    let h_send = vm.create_native_future_task();
    let id_send = TaskId(h_send.0);
    let h_recv = vm.create_native_future_task();
    let id_recv = TaskId(h_recv.0);

    // Sender blocks on empty rendezvous
    assert!(ch.send_task("rendezvous_data".to_string(), id_send, &mut vm).is_ok());
    assert_eq!(ch.send_waiters_count(), 1);
    assert!(matches!(
        vm.get_task_instance(h_send).unwrap().state,
        TaskState::Yielded(YieldReason::Parked(_))
    ));

    // Receiver arrives: direct handoff without intermediate buffer!
    let received = ch.recv_task(id_recv, &mut vm).unwrap();
    assert_eq!(received, Some("rendezvous_data".to_string()));
    assert_eq!(ch.send_waiters_count(), 0);
    assert_eq!(ch.len(), 0);
    assert_eq!(vm.get_task_instance(h_send).unwrap().state, TaskState::Ready);
}

// ============================================================================
// 3. Overflow Strategies: DropOldest, DropNewest, Fail
// ============================================================================

#[test]
fn test_overflow_strategy_drop_oldest() {
    let ch = Channel::<i32>::with_overflow(2, OverflowStrategy::DropOldest);
    assert!(ch.try_send(1).is_ok());
    assert!(ch.try_send(2).is_ok());
    // Queue is [1, 2]. Adding 3 drops oldest (1) -> [2, 3]
    assert!(ch.try_send(3).is_ok());
    assert_eq!(ch.len(), 2);

    assert_eq!(ch.try_recv(), Ok(Some(2)));
    assert_eq!(ch.try_recv(), Ok(Some(3)));
}

#[test]
fn test_overflow_strategy_drop_newest() {
    let ch = Channel::<i32>::with_overflow(2, OverflowStrategy::DropNewest);
    assert!(ch.try_send(10).is_ok());
    assert!(ch.try_send(20).is_ok());
    // Queue is [10, 20]. Adding 30 is dropped -> [10, 20]
    assert!(ch.try_send(30).is_ok());
    assert_eq!(ch.len(), 2);

    assert_eq!(ch.try_recv(), Ok(Some(10)));
    assert_eq!(ch.try_recv(), Ok(Some(20)));
    assert_eq!(ch.try_recv(), Err(ChannelError::Empty));
}

#[test]
fn test_overflow_strategy_fail() {
    let ch = Channel::<i32>::with_overflow(1, OverflowStrategy::Fail);
    assert!(ch.try_send(100).is_ok());
    assert_eq!(ch.try_send(200), Err(ChannelError::Full));
    assert_eq!(ch.len(), 1);
}

// ============================================================================
// 4. Channel Close Lifecycle
// ============================================================================

#[test]
fn test_channel_close_wakes_all_waiters_and_drains() {
    let mut vm = VM::new();
    let ch = Channel::<i64>::bounded(1);

    // Enqueue 1 item
    assert!(ch.try_send(999).is_ok());

    // Park 2 senders
    let s1 = vm.create_native_future_task();
    let s2 = vm.create_native_future_task();
    assert!(ch.send_task(1000, TaskId(s1.0), &mut vm).is_ok());
    assert!(ch.send_task(2000, TaskId(s2.0), &mut vm).is_ok());
    assert_eq!(ch.send_waiters_count(), 2);

    // Close channel
    ch.close(&mut vm);
    assert!(ch.is_closed());

    // Both senders must be woken
    assert_eq!(vm.get_task_instance(s1).unwrap().state, TaskState::Ready);
    assert_eq!(vm.get_task_instance(s2).unwrap().state, TaskState::Ready);
    assert_eq!(ch.send_waiters_count(), 0);

    // Buffered item is still drained
    assert_eq!(ch.try_recv(), Ok(Some(999)));
    // Once drained, recv returns None
    assert_eq!(ch.try_recv(), Ok(None));

    // New sends fail immediately
    assert_eq!(ch.try_send(3000), Err(ChannelError::Closed));
}

// ============================================================================
// 5. Cancellation Races (No Lost Wake, Stale Waiter, No Duplicate Delivery)
// ============================================================================

#[test]
fn test_cancellation_race_no_lost_wake_on_send() {
    let mut vm = VM::new();
    let ch = Channel::<i64>::bounded(2);

    // Park three receivers R1, R2, R3 in FIFO order
    let r1 = vm.create_native_future_task();
    let r2 = vm.create_native_future_task();
    let r3 = vm.create_native_future_task();
    let id1 = TaskId(r1.0);
    let id2 = TaskId(r2.0);
    let id3 = TaskId(r3.0);

    assert_eq!(ch.recv_task(id1, &mut vm), Ok(None));
    assert_eq!(ch.recv_task(id2, &mut vm), Ok(None));
    assert_eq!(ch.recv_task(id3, &mut vm), Ok(None));
    assert_eq!(ch.recv_waiters_count(), 3);

    // CANCEL R1: marks task Stopped and unregisters it in TaskManager
    vm.cancel_task(id1, InterruptReason::Cancelled);
    assert_eq!(vm.get_task_instance(r1).unwrap().state, TaskState::Stopped);
    assert!(!vm.is_task_waiting(id1));

    // Now Send 777:
    // Atomic transfer must detect R1 is cancelled, clean it up, and wake R2!
    // Must NOT lose the wakeup or deliver to dead R1!
    let s = vm.create_native_future_task();
    assert!(ch.send_task(777, TaskId(s.0), &mut vm).is_ok());

    // R2 must be woken!
    assert_eq!(vm.get_task_instance(r2).unwrap().state, TaskState::Ready);
    assert_eq!(ch.take_completed_transfer(id2), Some(777));

    // R1 was cleaned up and never received value
    assert_eq!(ch.take_completed_transfer(id1), None);

    // R3 is still waiting
    assert!(vm.is_task_waiting(id3));
}

#[test]
fn test_cancellation_race_no_lost_wake_on_recv() {
    let mut vm = VM::new();
    let ch = Channel::<i64>::bounded(1);

    // Fill buffer
    assert!(ch.try_send(10).is_ok());

    // Park S1 and S2
    let s1 = vm.create_native_future_task();
    let s2 = vm.create_native_future_task();
    let id_s1 = TaskId(s1.0);
    let id_s2 = TaskId(s2.0);

    assert!(ch.send_task(20, id_s1, &mut vm).is_ok());
    assert!(ch.send_task(30, id_s2, &mut vm).is_ok());
    assert_eq!(ch.send_waiters_count(), 2);

    // Cancel S1
    vm.cancel_task(id_s1, InterruptReason::Cancelled);
    assert_eq!(vm.get_task_instance(s1).unwrap().state, TaskState::Stopped);

    // Receiver reads 10: frees space.
    // Must skip cancelled S1, wake S2, and enqueue S2's value (30)!
    let r = vm.create_native_future_task();
    let item = ch.recv_task(TaskId(r.0), &mut vm).unwrap();
    assert_eq!(item, Some(10));

    // S2 must be woken and ready!
    assert_eq!(vm.get_task_instance(s2).unwrap().state, TaskState::Ready);
    assert_eq!(ch.len(), 1);

    // Next recv gets 30
    let item2 = ch.recv_task(TaskId(r.0), &mut vm).unwrap();
    assert_eq!(item2, Some(30));
}

#[test]
fn test_cancellation_race_no_duplicate_delivery() {
    let mut vm = VM::new();
    let ch = Channel::<i64>::bounded(0); // rendezvous

    let r1 = vm.create_native_future_task();
    let r2 = vm.create_native_future_task();
    let id1 = TaskId(r1.0);
    let id2 = TaskId(r2.0);

    assert_eq!(ch.recv_task(id1, &mut vm), Ok(None));
    assert_eq!(ch.recv_task(id2, &mut vm), Ok(None));

    // Send single value 555
    let s = vm.create_native_future_task();
    assert!(ch.send_task(555, TaskId(s.0), &mut vm).is_ok());

    // Exactly one receiver receives 555
    let got1 = ch.take_completed_transfer(id1);
    let got2 = ch.take_completed_transfer(id2);

    assert_eq!((got1, got2), (Some(555), None));
    assert_eq!(vm.get_task_instance(r1).unwrap().state, TaskState::Ready);
    assert_eq!(
        vm.get_task_instance(r2).unwrap().state,
        TaskState::Yielded(YieldReason::Parked(WaitKey(2)))
    );
}

#[test]
fn test_cleanup_cancelled_waiters_explicit() {
    let mut vm = VM::new();
    let ch = Channel::<i64>::bounded(2);

    let r1 = vm.create_native_future_task();
    let r2 = vm.create_native_future_task();
    let id1 = TaskId(r1.0);
    let id2 = TaskId(r2.0);

    assert_eq!(ch.recv_task(id1, &mut vm), Ok(None));
    assert_eq!(ch.recv_task(id2, &mut vm), Ok(None));
    assert_eq!(ch.recv_waiters_count(), 2);

    vm.cancel_task(id1, InterruptReason::Cancelled);
    ch.cleanup_cancelled_waiters(&vm);

    // Stale waiter R1 is pruned; R2 remains
    assert_eq!(ch.recv_waiters_count(), 1);
}

// ============================================================================
// 6. Registered Script Path Execution (`channel.*` builtins)
// ============================================================================

#[test]
fn test_script_path_channel_builtins() {
    reset_global_channel_registry();

    let script = r#"
        extern
            function channel_new(int): int
            function channel_send(int, int): bool
            function channel_recv(int): int
            function channel_close(int): bool
            function channel_len(int): int
            function channel_is_closed(int): bool
        end

        function run_channel_pipeline(): int
            -- 1. Create channel with capacity 2
            local ch: int = channel_new(2)

            -- 2. Send values
            channel_send(ch, 100)
            channel_send(ch, 200)

            -- 3. Verify length
            local len1: int = channel_len(ch)

            -- 4. Receive values
            local a: int = channel_recv(ch)
            local b: int = channel_recv(ch)

            -- 5. Close channel
            channel_close(ch)
            local closed: bool = channel_is_closed(ch)

            if closed and len1 == 2 then
                return a + b
            else
                return -1
            end
        end
    "#;

    let mut prog = EmbeddedProgram::builder()
        .module("main", script)
        .entry_module("main")
        .compile()
        .expect("compile channel script");

    // Register channel builtins on the program's VM
    register_channel_builtins(prog.vm_mut());

    let res = prog
        .vm_mut()
        .call("main.run_channel_pipeline", vec![])
        .expect("execute script");

    assert_eq!(res, Value::Int(300));
}

#[test]
fn test_script_path_try_send_try_recv_and_overflow() {
    reset_global_channel_registry();

    let script = r#"
        extern
            function channel_new(int): int
            function channel_try_send(int, int): bool
            function channel_try_recv(int): int
        end

        function test_nonblocking(): int
            local ch: int = channel_new(1)
            local ok1: bool = channel_try_send(ch, 42)
            local ok2: bool = channel_try_send(ch, 99) -- must fail (full)

            local val: int = channel_try_recv(ch)
            local empty_val: int = channel_try_recv(ch) -- nil / 0

            if ok1 and not ok2 and val == 42 and empty_val == 0 then
                return 1
            else
                return 0
            end
        end
    "#;

    let mut prog = EmbeddedProgram::builder()
        .module("main", script)
        .entry_module("main")
        .compile()
        .expect("compile nonblocking test");

    register_channel_builtins(prog.vm_mut());

    let res = prog
        .vm_mut()
        .call("main.test_nonblocking", vec![])
        .expect("execute script");

    assert_eq!(res, Value::Int(1));
}

// ============================================================================
// 7. Multi-Channel SelectGroup
// ============================================================================

#[test]
fn test_select_group_rotational_probe() {
    let ch1 = Channel::<&'static str>::bounded(2);
    let ch2 = Channel::<&'static str>::bounded(2);

    let mut select = SelectGroup::<&'static str>::new();
    let idx1 = select.add_channel(ch1.clone());
    let idx2 = select.add_channel(ch2.clone());

    assert_eq!(select.try_select(0), None);

    // Send item to ch2
    assert!(ch2.try_send("data_from_ch2").is_ok());

    let sel = select.try_select(0);
    assert_eq!(sel, Some(SelectResult::Received(idx2, "data_from_ch2")));

    // When ch1 is closed
    ch1.close_local();
    let sel_closed = select.try_select(0);
    assert_eq!(sel_closed, Some(SelectResult::ChannelClosed(idx1)));
}

// ============================================================================
// 8. TestRunner Bounded Parallelism (`T-PAR`), Concurrency Limits, Failures & Teardown
// ============================================================================

#[test]
fn test_runner_concurrency_limit_strictly_enforced() {
    let concurrency_limit = 3;
    let runner = TestRunner::new().with_concurrency(concurrency_limit);
    runner.reset_peak_concurrency();

    let active_counter = Arc::new(AtomicUsize::new(0));
    let max_observed = Arc::new(AtomicUsize::new(0));

    let mut test_jobs = Vec::new();
    for i in 0..12 {
        let active = Arc::clone(&active_counter);
        let max_obs = Arc::clone(&max_observed);
        test_jobs.push((
            format!("parallel_test_{}", i),
            move |_fib: &mut FiberContext, _scope: &mut Scope| -> Result<(), AssertError> {
                let curr = active.fetch_add(1, Ordering::SeqCst) + 1;
                let mut prev = max_obs.load(Ordering::SeqCst);
                while curr > prev {
                    match max_obs.compare_exchange_weak(
                        prev,
                        curr,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    ) {
                        Ok(_) => break,
                        Err(actual) => prev = actual,
                    }
                }

                // Simulate brief execution work
                std::thread::sleep(Duration::from_millis(15));

                active.fetch_sub(1, Ordering::SeqCst);
                Ok(())
            },
        ));
    }

    let results = runner.run_tests_parallel(test_jobs);
    assert_eq!(results.len(), 12);
    for res in &results {
        assert_eq!(res.outcome, TestOutcome::Passed);
    }

    // Invariant check: active fibers never exceeded the concurrency limit!
    let observed_peak = max_observed.load(Ordering::SeqCst);
    assert!(
        observed_peak <= concurrency_limit,
        "Observed peak concurrency {} exceeded limit {}",
        observed_peak,
        concurrency_limit
    );
    assert!(
        runner.peak_concurrency() <= concurrency_limit,
        "Runner reported peak concurrency {} exceeded limit {}",
        runner.peak_concurrency(),
        concurrency_limit
    );
}

#[test]
fn test_runner_failure_containment_and_isolation() {
    let runner = TestRunner::new().with_concurrency(4);

    let jobs = vec![
        (
            "pass_job",
            Box::new(|_fib: &mut FiberContext, _scope: &mut Scope| Ok(()))
                as Box<
                    dyn FnOnce(&mut FiberContext, &mut Scope) -> Result<(), AssertError>
                        + Send
                        + std::panic::UnwindSafe,
                >,
        ),
        (
            "assert_fail_job",
            Box::new(|_fib: &mut FiberContext, _scope: &mut Scope| {
                Err(AssertError {
                    message: "explicit assert failure".to_string(),
                    location: None,
                    diff: None,
                    fiber_id: None,
                })
            }),
        ),
        (
            "panic_job",
            Box::new(|_fib: &mut FiberContext, _scope: &mut Scope| {
                panic!("fatal panic in child fiber must be contained!");
            }),
        ),
        (
            "interrupt_oog_job",
            Box::new(|fib: &mut FiberContext, _scope: &mut Scope| {
                fib.pending_interrupt = Some(InterruptReason::OutOfGas);
                Ok(())
            }),
        ),
    ];

    let results = runner.run_tests_parallel(jobs);
    assert_eq!(results.len(), 4);

    assert_eq!(results[0].outcome, TestOutcome::Passed);

    match &results[1].outcome {
        TestOutcome::Failed(err) => assert!(err.message.contains("explicit assert failure")),
        other => panic!("expected assert failure, got {:?}", other),
    }

    match &results[2].outcome {
        TestOutcome::Failed(err) => assert!(err.message.contains("fatal panic in child fiber")),
        other => panic!("expected panic failure, got {:?}", other),
    }

    match &results[3].outcome {
        TestOutcome::Failed(err) => assert!(err.message.contains("OutOfGas")),
        other => panic!("expected OutOfGas failure, got {:?}", other),
    }
}

#[test]
fn test_runner_lifo_teardown_guarantee_on_all_outcomes() {
    let runner = TestRunner::new().with_concurrency(2);

    let teardown_log = Arc::new(Mutex::new(Vec::new()));

    let log1 = Arc::clone(&teardown_log);
    let log2 = Arc::clone(&teardown_log);

    let jobs = vec![
        (
            "passing_with_finalizers",
            Box::new(move |_fib: &mut FiberContext, scope: &mut Scope| {
                let l_a = Arc::clone(&log1);
                scope
                    .register_finalizer(move || {
                        l_a.lock().unwrap().push("pass_fin_1");
                    })
                    .unwrap();

                let l_b = Arc::clone(&log1);
                scope
                    .register_finalizer(move || {
                        l_b.lock().unwrap().push("pass_fin_2");
                    })
                    .unwrap();
                Ok(())
            })
                as Box<
                    dyn FnOnce(&mut FiberContext, &mut Scope) -> Result<(), AssertError>
                        + Send
                        + std::panic::UnwindSafe,
                >,
        ),
        (
            "panicking_with_finalizers",
            Box::new(move |_fib: &mut FiberContext, scope: &mut Scope| {
                let l_c = Arc::clone(&log2);
                scope
                    .register_finalizer(move || {
                        l_c.lock().unwrap().push("panic_fin_1");
                    })
                    .unwrap();

                let l_d = Arc::clone(&log2);
                scope
                    .register_finalizer(move || {
                        l_d.lock().unwrap().push("panic_fin_2");
                    })
                    .unwrap();

                panic!("panic during test execution");
            }),
        ),
    ];

    let results = runner.run_tests_parallel(jobs);
    assert_eq!(results.len(), 2);

    let log = teardown_log.lock().unwrap();
    // Scope guarantees LIFO order of finalizer execution:
    // For passing test: fin_2 ran before fin_1
    let pos_p1 = log.iter().position(|s| *s == "pass_fin_1").unwrap();
    let pos_p2 = log.iter().position(|s| *s == "pass_fin_2").unwrap();
    assert!(pos_p2 < pos_p1, "LIFO finalizer order violated for pass test");

    // For panicking test: fin_2 ran before fin_1 despite the panic!
    let pos_pan1 = log.iter().position(|s| *s == "panic_fin_1").unwrap();
    let pos_pan2 = log.iter().position(|s| *s == "panic_fin_2").unwrap();
    assert!(
        pos_pan2 < pos_pan1,
        "LIFO finalizer order violated for panicking test"
    );
}

#[test]
fn test_runner_bail_halts_pending_tests() {
    let runner = TestRunner::new().with_bail(true).with_concurrency(1);

    let executed = Arc::new(AtomicUsize::new(0));
    let exec1 = Arc::clone(&executed);
    let exec2 = Arc::clone(&executed);
    let exec3 = Arc::clone(&executed);

    let jobs = vec![
        (
            "job_1_pass",
            Box::new(move |_fib: &mut FiberContext, _scope: &mut Scope| {
                exec1.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
                as Box<
                    dyn FnOnce(&mut FiberContext, &mut Scope) -> Result<(), AssertError>
                        + Send
                        + std::panic::UnwindSafe,
                >,
        ),
        (
            "job_2_fail",
            Box::new(move |_fib: &mut FiberContext, _scope: &mut Scope| {
                exec2.fetch_add(1, Ordering::SeqCst);
                Err(AssertError {
                    message: "bail trigger".to_string(),
                    location: None,
                    diff: None,
                    fiber_id: None,
                })
            }),
        ),
        (
            "job_3_unreachable",
            Box::new(move |_fib: &mut FiberContext, _scope: &mut Scope| {
                exec3.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }),
        ),
    ];

    let results = runner.run_tests_parallel(jobs);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].outcome, TestOutcome::Passed);
    assert!(matches!(results[1].outcome, TestOutcome::Failed(_)));
    assert_eq!(executed.load(Ordering::SeqCst), 2);
}

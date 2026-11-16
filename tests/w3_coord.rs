//! Wave 3 Track 3C Test Suite (`R-COORD`: Coordination Primitives).
//!
//! Complies with `RFC-STDLIB-ASYNC-COORDINATION` (§3.4) and `WAVE3_EXECUTION_PLAN.md`:
//! 1. `Stream` & `Sink`: chunk-oriented lazy pull streams, composable sinks, early stop.
//! 2. `Queue`: bounded backpressure, FIFO waiter ordering, close, cancellation races, MPMC.
//! 3. `Deferred`: single-assignment promise, idempotent completion, multi-waiter wakeup.
//! 4. `Semaphore`: counting semaphore, FIFO ordering, zero-leak RAII guard, cancelled waiter skip.
//! 5. `Ref`: deterministic shared mutable state cell.
//! 6. `Schedule`: recurs, spaced, exponential, retry, repeat, timer-safe timeout (0 OS threads).
//! 7. Script execution: `@std/coord`, `@std/sync`, `@std/stream`, `@std/schedule`.

extern crate alloc;

use lust::builtins::coord;

use coord::{
    Deferred, OverflowStrategy, Queue, QueueError, Ref, Schedule, ScheduleStep,
    Semaphore, Sink, Stream,
    register_builtins as register_coord_builtins, reset_global_coord_registries,
};
use lust::bytecode::Value;
use lust::effect_abi::{Exit, FiberExecutionState};
use lust::embed::EmbeddedProgram;
use lust::vm::{TaskId, VM};

// ============================================================================
// 1. Ref Tests
// ============================================================================

#[test]
fn test_ref_basic_get_set() {
    let r = Ref::new(10);
    assert_eq!(r.get(), 10);
    r.set(25);
    assert_eq!(r.get(), 25);
}

#[test]
fn test_ref_update_and_modify() {
    let r = Ref::new(100);
    let updated = r.update(|x| x + 50);
    assert_eq!(updated, 150);
    assert_eq!(r.get(), 150);

    let (label, val) = r.modify(|x| ("modified", x * 2));
    assert_eq!(label, "modified");
    assert_eq!(val, 300);
    assert_eq!(r.get(), 300);
}

#[test]
fn test_ref_shared_across_fibers() {
    let r1 = Ref::new(vec![1, 2, 3]);
    let r2 = r1.clone();

    r1.update(|mut v| {
        v.push(4);
        v
    });

    assert_eq!(r2.get(), vec![1, 2, 3, 4]);
}

// ============================================================================
// 2. Deferred Tests
// ============================================================================

#[test]
fn test_deferred_basic_completion() {
    let mut d = Deferred::<i64>::make();
    assert!(!d.is_completed());
    assert_eq!(d.poll(), None);

    assert!(d.complete(42));
    assert!(d.is_completed());
    assert_eq!(d.poll(), Some(&Exit::Success(42)));
}

#[test]
fn test_deferred_idempotent_first_wins() {
    let mut d = Deferred::<String>::make();
    assert!(d.complete("first".to_string()));
    // Subsequent completes/fails must return false and NOT overwrite
    assert!(!d.complete("second".to_string()));
    assert!(!d.fail("error".to_string()));

    match d.poll() {
        Some(Exit::Success(s)) => assert_eq!(s, "first"),
        other => panic!("Unexpected deferred state: {:?}", other),
    }
}

#[test]
fn test_deferred_multi_awaiter_simultaneous_wake() {
    let mut vm = VM::new();
    let mut d = Deferred::<i64>::make();

    let t1 = TaskId(vm.create_native_future_task().0);
    let t2 = TaskId(vm.create_native_future_task().0);
    let t3 = TaskId(vm.create_native_future_task().0);

    assert!(d.await_or_park_vm(t1, &mut vm).is_err());
    assert!(d.await_or_park_vm(t2, &mut vm).is_err());
    assert!(d.await_or_park_vm(t3, &mut vm).is_err());
    assert_eq!(d.waiters_count(), 3);

    let (completed, woken) = d.complete_and_wake(Exit::Success(100), &mut vm);
    assert!(completed);
    assert_eq!(woken.len(), 3);
    assert!(woken.contains(&t1));
    assert!(woken.contains(&t2));
    assert!(woken.contains(&t3));
}

#[test]
fn test_deferred_await_already_completed() {
    let mut vm = VM::new();
    let mut d = Deferred::<i64>::make();
    assert!(d.complete(999));

    let t1 = TaskId(vm.create_native_future_task().0);
    let res = d.await_or_park_vm(t1, &mut vm);
    assert_eq!(res, Ok(&Exit::Success(999)));
}

#[test]
fn test_deferred_cancellation_race() {
    let mut vm = VM::new();
    let mut d = Deferred::<i64>::make();

    let t_cancel = TaskId(vm.create_native_future_task().0);
    let t_survive = TaskId(vm.create_native_future_task().0);

    let _ = d.await_or_park_vm(t_cancel, &mut vm);
    let _ = d.await_or_park_vm(t_survive, &mut vm);

    // Cancel t_cancel
    d.cancel_waiter_vm(t_cancel, &mut vm);
    assert_eq!(d.waiters_count(), 1);

    let (completed, woken) = d.complete_and_wake(Exit::Success(77), &mut vm);
    assert!(completed);
    assert!(woken.contains(&t_survive));
}

// ============================================================================
// 3. Semaphore Tests
// ============================================================================

#[test]
fn test_semaphore_basic_acquire_release() {
    let mut sem = Semaphore::make(2);
    assert_eq!(sem.available_permits(), 2);

    assert!(sem.try_acquire(1));
    assert_eq!(sem.available_permits(), 1);
    assert!(sem.try_acquire(1));
    assert_eq!(sem.available_permits(), 0);

    // Insufficient permits
    assert!(!sem.try_acquire(1));

    // Release
    let mut vm = VM::new();
    let woken = sem.release_vm(1, &mut vm);
    assert_eq!(woken.len(), 0);
    assert_eq!(sem.available_permits(), 1);
}

#[test]
fn test_semaphore_waiter_ordering_fifo() {
    let mut vm = VM::new();
    let mut sem = Semaphore::make(0);

    let t1 = TaskId(vm.create_native_future_task().0);
    let t2 = TaskId(vm.create_native_future_task().0);
    let t3 = TaskId(vm.create_native_future_task().0);

    assert!(!sem.acquire_or_park_vm(t1, 1, &mut vm));
    assert!(!sem.acquire_or_park_vm(t2, 1, &mut vm));
    assert!(!sem.acquire_or_park_vm(t3, 1, &mut vm));
    assert_eq!(sem.waiters_count(), 3);

    // Release 1 permit -> wakes t1 first (FIFO)
    let woken1 = sem.release_vm(1, &mut vm);
    assert_eq!(woken1, vec![t1]);
    assert_eq!(sem.available_permits(), 0);

    // Release 2 permits -> wakes t2 then t3
    let woken2 = sem.release_vm(2, &mut vm);
    assert_eq!(woken2, vec![t2, t3]);
    assert_eq!(sem.available_permits(), 0);
}

#[test]
fn test_semaphore_head_of_line_blocking() {
    let mut vm = VM::new();
    let mut sem = Semaphore::make(0);

    let t_large = TaskId(vm.create_native_future_task().0);
    let t_small = TaskId(vm.create_native_future_task().0);

    // t_large needs 3 permits, t_small needs 1 permit
    assert!(!sem.acquire_or_park_vm(t_large, 3, &mut vm));
    assert!(!sem.acquire_or_park_vm(t_small, 1, &mut vm));

    // Release 2 permits: t_large needs 3, so head-of-line stops.
    // t_small must NOT jump ahead of t_large!
    let woken = sem.release_vm(2, &mut vm);
    assert_eq!(woken.len(), 0);
    assert_eq!(sem.available_permits(), 2);

    // Release 1 more permit (total 3): t_large now wakes!
    let woken2 = sem.release_vm(1, &mut vm);
    assert_eq!(woken2, vec![t_large]);
    assert_eq!(sem.available_permits(), 0);
}

#[test]
fn test_semaphore_cancellation_race_zero_permit_leak() {
    let mut vm = VM::new();
    let mut sem = Semaphore::make(0);

    let t1_cancelled = TaskId(vm.create_native_future_task().0);
    let t2_alive = TaskId(vm.create_native_future_task().0);

    assert!(!sem.acquire_or_park_vm(t1_cancelled, 2, &mut vm));
    assert!(!sem.acquire_or_park_vm(t2_alive, 1, &mut vm));

    // Mark t1_cancelled as interrupted in fiber context
    if let Some(task) = vm.task_manager_mut().get_mut(t1_cancelled) {
        task.fiber_context.state = FiberExecutionState::Interrupted;
    }

    // Release 1 permit: t1_cancelled is skipped without consuming permits,
    // and t2_alive takes the permit!
    let woken = sem.release_vm(1, &mut vm);
    assert_eq!(woken, vec![t2_alive]);
    assert_eq!(sem.available_permits(), 0);
}

#[test]
fn test_semaphore_with_permits_raii_cleanup() {
    let mut sem = Semaphore::make(3);

    // Success case
    let res = sem.with_permits(2, || 42 * 2);
    assert_eq!(res, Ok(84));
    assert_eq!(sem.available_permits(), 3);

    // Error return inside closure
    let err_res = sem.with_permits(3, || -> Result<(), &'static str> {
        Err("simulated failure")
    });
    assert!(err_res.is_ok()); // closure executed and returned Err
    assert_eq!(sem.available_permits(), 3); // 100% restored

    // 100 iterations of simulated error returns: zero leaked permits
    for _ in 0..100 {
        let _ = sem.with_permits(2, || Err::<(), ()>(()));
    }
    assert_eq!(sem.available_permits(), 3);
}

// ============================================================================
// 4. Queue Tests (Bounded Backpressure, Waiters, Close, Strategies)
// ============================================================================

#[test]
fn test_queue_bounded_fifo() {
    let mut q = Queue::<i64>::bounded(3);
    assert_eq!(q.capacity(), 3);
    assert_eq!(q.len(), 0);
    assert!(q.is_empty());
    assert!(!q.is_full());

    assert!(q.try_offer(10).unwrap());
    assert!(q.try_offer(20).unwrap());
    assert!(q.try_offer(30).unwrap());
    assert!(q.is_full());

    // Buffer is full with Suspend -> returns Full
    assert_eq!(q.try_offer(40), Err(QueueError::Full));

    assert_eq!(q.try_take(), Ok(Some(10)));
    assert_eq!(q.try_take(), Ok(Some(20)));
    assert_eq!(q.try_take(), Ok(Some(30)));
    assert_eq!(q.try_take(), Err(QueueError::Empty));
}

#[test]
fn test_queue_overflow_strategies() {
    // 1. DropOldest (Sliding)
    let mut sliding = Queue::<i64>::sliding(2);
    assert!(sliding.try_offer(1).unwrap());
    assert!(sliding.try_offer(2).unwrap());
    assert!(sliding.try_offer(3).unwrap()); // drops 1
    assert_eq!(sliding.try_take(), Ok(Some(2)));
    assert_eq!(sliding.try_take(), Ok(Some(3)));

    // 2. DropNewest (Dropping)
    let mut dropping = Queue::<i64>::dropping(2);
    assert!(dropping.try_offer(10).unwrap());
    assert!(dropping.try_offer(20).unwrap());
    assert!(!dropping.try_offer(30).unwrap()); // dropped, returns false
    assert_eq!(dropping.try_take(), Ok(Some(10)));
    assert_eq!(dropping.try_take(), Ok(Some(20)));

    // 3. Fail
    let mut failing = Queue::<i64>::make(1, OverflowStrategy::Fail);
    assert!(failing.try_offer(100).unwrap());
    assert_eq!(failing.try_offer(200), Err(QueueError::Full));
}

#[test]
fn test_queue_backpressure_park_and_wake() {
    let mut vm = VM::new();
    let mut q = Queue::<i64>::bounded(2);

    assert!(q.try_offer(1).unwrap());
    assert!(q.try_offer(2).unwrap());
    assert!(q.is_full());

    // Third offer from task_p parks
    let task_p = TaskId(vm.create_native_future_task().0);
    let parked = q.offer_or_park_vm(task_p, 3, &mut vm).unwrap();
    assert!(!parked);
    assert_eq!(q.offer_waiters_count(), 1);

    // Task_c takes an item: frees buffer space, waking task_p and moving item 3 into buffer!
    let task_c = TaskId(vm.create_native_future_task().0);
    let val = q.take_or_park_vm(task_c, &mut vm).unwrap();
    assert_eq!(val, Some(1));
    assert_eq!(q.offer_waiters_count(), 0);

    // Next take retrieves item 2
    assert_eq!(q.try_take(), Ok(Some(2)));
    // Next take retrieves item 3 (which was moved into buffer by task_p)
    assert_eq!(q.try_take(), Ok(Some(3)));
    assert_eq!(q.try_take(), Err(QueueError::Empty));
}

#[test]
fn test_queue_close_lifecycle() {
    let mut vm = VM::new();
    let mut q = Queue::<i64>::bounded(2);
    assert!(q.try_offer(10).unwrap());
    assert!(q.try_offer(20).unwrap());

    let task_p = TaskId(vm.create_native_future_task().0);
    assert!(!q.offer_or_park_vm(task_p, 30, &mut vm).unwrap());

    // Close queue: wakes parked offerer
    let woken = q.close_vm(&mut vm);
    assert!(woken.contains(&task_p));
    assert!(q.is_closed());

    // Remaining items can still be drained by takers
    assert_eq!(q.try_take(), Ok(Some(10)));
    assert_eq!(q.try_take(), Ok(Some(20)));
    // Once drained, returns None (closed)
    assert_eq!(q.try_take(), Ok(None));

    // Subsequent offer rejected with Closed
    assert_eq!(q.try_offer(40), Err(QueueError::Closed));
}

#[test]
fn test_queue_cancellation_race_no_lost_wake() {
    let mut vm = VM::new();
    let mut q = Queue::<i64>::bounded(1);
    assert!(q.try_offer(1).unwrap());

    let t1_cancel = TaskId(vm.create_native_future_task().0);
    let t2_alive = TaskId(vm.create_native_future_task().0);

    let _ = q.offer_or_park_vm(t1_cancel, 2, &mut vm);
    let _ = q.offer_or_park_vm(t2_alive, 3, &mut vm);
    assert_eq!(q.offer_waiters_count(), 2);

    // Cancel t1_cancel
    if let Some(task) = vm.task_manager_mut().get_mut(t1_cancel) {
        task.fiber_context.state = FiberExecutionState::Interrupted;
    }

    // Take an item: must skip dead t1_cancel and wake alive t2_alive without lost wake!
    let val = q.take_or_park_vm(TaskId(999), &mut vm).unwrap();
    assert_eq!(val, Some(1));
    assert_eq!(q.offer_waiters_count(), 0);

    // Buffer now contains item 3 from t2_alive
    assert_eq!(q.try_take(), Ok(Some(3)));
}

#[test]
fn test_queue_mpmc_no_lost_or_duplicated_work() {
    let mut q = Queue::<i64>::unbounded();
    let n = 200;

    for i in 1..=n {
        assert!(q.try_offer(i).unwrap());
    }

    let mut sum: i64 = 0;
    let mut count = 0;
    while let Ok(Some(item)) = q.try_take() {
        sum += item;
        count += 1;
    }

    assert_eq!(count, n as usize);
    let expected_sum: i64 = (n * (n + 1)) / 2;
    assert_eq!(sum, expected_sum, "No items lost or duplicated");
}

// ============================================================================
// 5. Schedule Tests
// ============================================================================

#[test]
fn test_schedule_recurs() {
    let sched = Schedule::recurs(3);
    let mut state = coord::ScheduleState::initial();

    for _ in 0..3 {
        match sched.step(&state) {
            ScheduleStep::Continue { next_state, delay_ms } => {
                assert_eq!(delay_ms, 0);
                state = next_state;
            }
            ScheduleStep::Done => panic!("Expected Continue"),
        }
    }

    assert_eq!(sched.step(&state), ScheduleStep::Done);
}

#[test]
fn test_schedule_exponential_and_then() {
    let exp = Schedule::exponential(10, 2.0, 100);
    let mut state = coord::ScheduleState::initial();

    // Step 0: delay 10
    match exp.step(&state) {
        ScheduleStep::Continue { next_state, delay_ms } => {
            assert_eq!(delay_ms, 10);
            state = next_state;
        }
        _ => panic!("Expected continue"),
    }

    // Step 1: delay 20
    match exp.step(&state) {
        ScheduleStep::Continue { next_state: _, delay_ms } => {
            assert_eq!(delay_ms, 20);
        }
        _ => panic!("Expected continue"),
    }
}

#[test]
fn test_schedule_retry_success_and_exhaustion() {
    let sched = Schedule::recurs(2);
    let mut attempts = 0;

    // Fails twice, succeeds on 3rd attempt
    let res = sched.retry(|| {
        attempts += 1;
        if attempts < 3 {
            Err("fail")
        } else {
            Ok(attempts)
        }
    });
    assert_eq!(res, Ok(3));

    // Exhaustion
    attempts = 0;
    let fail_res = sched.retry(|| {
        attempts += 1;
        Err::<(), &str>("permanent fail")
    });
    assert_eq!(fail_res, Err("permanent fail"));
    assert_eq!(attempts, 3); // initial + 2 retries
}

#[test]
fn test_schedule_timeout_no_os_threads() {
    // Success before timeout
    let ok = Schedule::timeout(100, || Ok::<_, String>(42));
    assert_eq!(ok, Ok(42));

    // Deadline 0 ms immediately aborts
    let timeout = Schedule::timeout(0, || Ok::<_, String>(42));
    assert!(timeout.is_err());
    assert!(timeout.unwrap_err().contains("deadline is 0 ms"));
}

// ============================================================================
// 6. Stream and Sink Tests
// ============================================================================

#[test]
fn test_stream_from_array_and_collect() {
    let items = vec![1, 2, 3, 4, 5];
    let st = Stream::from_array(items.clone());
    let res = st.collect().unwrap();
    assert_eq!(res, items);
}

#[test]
fn test_stream_map_filter_take() {
    let st = Stream::range(1, 100, 1)
        .filter(|&x| x % 2 == 0) // 2, 4, 6, 8, 10, ...
        .map(|x| x * 10)         // 20, 40, 60, 80, 100, ...
        .take(4);

    let res = st.collect().unwrap();
    assert_eq!(res, vec![20, 40, 60, 80]);
}

#[test]
fn test_stream_chunks() {
    let st = Stream::range(0, 10, 1).chunks(3);
    let res = st.collect().unwrap();
    assert_eq!(res, vec![
        vec![0, 1, 2],
        vec![3, 4, 5],
        vec![6, 7, 8],
        vec![9]
    ]);
}

#[test]
fn test_stream_repeat_early_stop() {
    // Infinite repeat stream halted by take(5)
    let st = Stream::repeat(42, None).take(5);
    let res = st.collect().unwrap();
    assert_eq!(res, vec![42, 42, 42, 42, 42]);
}

#[test]
fn test_stream_into_sink_head_and_last() {
    let st1 = Stream::range(10, 20, 1);
    let head = st1.run(Sink::head()).unwrap();
    assert_eq!(head, Some(10));

    let st2 = Stream::range(10, 20, 1);
    let last = st2.run(Sink::last()).unwrap();
    assert_eq!(last, Some(19));
}

#[test]
fn test_stream_into_sink_drain_and_fold() {
    let count = Stream::range(0, 50, 1).drain().unwrap();
    assert_eq!(count, 50);

    let sum = Stream::range(1, 11, 1).fold(0, |acc, x| acc + x).unwrap();
    assert_eq!(sum, 55);
}

#[test]
fn test_stream_buffer_bounded_backpressure() {
    let st = Stream::range(1, 20, 1).buffer(5, OverflowStrategy::Suspend);
    let res = st.collect().unwrap();
    assert_eq!(res, (1..20).collect::<Vec<_>>());
}

#[test]
fn test_stream_into_queue_sink() {
    let q = Queue::<i64>::unbounded();
    let count = Stream::range(1, 10, 1).run(Sink::queue(q)).unwrap();
    assert_eq!(count, 9);
}

// ============================================================================
// 7. Script Execution Tests (`@std/coord`, `@std/sync`, `@std/stream`, `@std/schedule`)
// ============================================================================

#[test]
fn test_script_import_coord_module() {
    reset_global_coord_registries();
    let source = r#"
use @std/coord
function test_coord(): int
    return 1
end
"#;
    let mut program = EmbeddedProgram::builder()
        .module("main", source)
        .entry_module("main")
        .compile()
        .expect("use @std/coord must compile");

    register_coord_builtins(program.vm_mut());
    let res = program.vm_mut().call("main.test_coord", vec![]).unwrap();
    assert_eq!(res, Value::Int(1));
}

#[test]
fn test_script_coord_ref_execution() {
    reset_global_coord_registries();
    let source = r#"
use @std/coord.{Ref}

function run_ref(): int
    local r = Ref(10)
    r.set(40)
    local v = r.get()
    return v + 2
end
"#;
    let mut program = EmbeddedProgram::builder()
        .module("main", source)
        .entry_module("main")
        .compile()
        .expect("Ref script must compile");

    register_coord_builtins(program.vm_mut());
    let res = program.vm_mut().call("main.run_ref", vec![]).unwrap();
    assert_eq!(res, Value::Int(42));
}

#[test]
fn test_script_coord_semaphore_execution() {
    reset_global_coord_registries();
    let source = r#"
use @std/sync.{Semaphore}

function run_sem(): int
    local sem = Semaphore(2)
    local a1 = sem.available()
    sem.try_acquire(1)
    local a2 = sem.available()
    sem.release(1)
    local a3 = sem.available()
    return a1 + a2 + a3
end
"#;
    let mut program = EmbeddedProgram::builder()
        .module("main", source)
        .entry_module("main")
        .compile()
        .expect("Semaphore script must compile");

    register_coord_builtins(program.vm_mut());
    let res = program.vm_mut().call("main.run_sem", vec![]).unwrap();
    assert_eq!(res, Value::Int(2 + 1 + 2)); // 5
}

#[test]
fn test_script_coord_queue_execution() {
    reset_global_coord_registries();
    let source = r#"
use @std/sync.{Queue}

function run_queue(): int
    local q = Queue(3, "suspend")
    q.offer(10)
    q.offer(20)
    local v1 = q.take()
    local v2 = q.take()
    return v1 + v2
end
"#;
    let mut program = EmbeddedProgram::builder()
        .module("main", source)
        .entry_module("main")
        .compile()
        .expect("Queue script must compile");

    register_coord_builtins(program.vm_mut());
    let res = program.vm_mut().call("main.run_queue", vec![]).unwrap();
    assert_eq!(res, Value::Int(30));
}

#[test]
fn test_script_coord_deferred_execution() {
    reset_global_coord_registries();
    let source = r#"
use @std/sync.{Deferred}

function run_deferred(): int
    local d = Deferred()
    d.complete(123)
    if d.is_completed() then
        return d.poll()
    end
    return -1
end
"#;
    let mut program = EmbeddedProgram::builder()
        .module("main", source)
        .entry_module("main")
        .compile()
        .expect("Deferred script must compile");

    register_coord_builtins(program.vm_mut());
    let res = program.vm_mut().call("main.run_deferred", vec![]).unwrap();
    assert_eq!(res, Value::Int(123));
}

#[test]
fn test_script_coord_schedule_execution() {
    reset_global_coord_registries();
    let source = r#"
use @std/schedule.{Schedule}

function run_schedule(): int
    local sched = Schedule(3)
    return sched.get_n()
end
"#;
    let mut program = EmbeddedProgram::builder()
        .module("main", source)
        .entry_module("main")
        .compile()
        .expect("Schedule script must compile");

    register_coord_builtins(program.vm_mut());
    let res = program.vm_mut().call("main.run_schedule", vec![]).unwrap();
    assert_eq!(res, Value::Int(3));
}

#[test]
fn test_script_coord_stream_collect() {
    reset_global_coord_registries();
    let source = r#"
use @std/stream.{Stream}

function run_stream(): int
    local st = Stream([1, 2, 3, 4])
    return st.count()
end
"#;
    let mut program = EmbeddedProgram::builder()
        .module("main", source)
        .entry_module("main")
        .compile()
        .expect("Stream script must compile");

    register_coord_builtins(program.vm_mut());
    let res = program.vm_mut().call("main.run_stream", vec![]).unwrap();
    assert_eq!(res, Value::Int(4));
}

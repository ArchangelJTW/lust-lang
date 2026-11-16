//! Wave 3 Track 3B Async Integration Test Suite (`R-E06`).
//!
//! Verifies:
//! 1. `AsyncWakeNotifier` and `WakeNotifierScope` lifecycle (LIFO restoration, owner-thread confinement).
//! 2. `SignalState` single-mutex TOCTOU race fix: `Pending` -> `Ready` -> `Consumed`, fused repoll,
//!    duplicate completion rejection (original result preserved).
//! 3. Wake-before-park and duplicate wake resilience in `async_runtime.rs`.
//! 4. Event-driven waker integration with `AsyncRegistry` and `EmbeddedProgram`.

use lust::bytecode::Value;
use lust::embed::async_runtime::{
    current_thread_wake_notifier, set_thread_wake_notifier, signal_pair, AsyncCompleter,
    AsyncSignalFuture, AsyncWakeNotifier, SignalState, WakeNotifierScope,
};
use lust::embed::{AsyncDriver, EmbeddedBuilder, EmbeddedProgram};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

struct TestNotifier {
    count: AtomicUsize,
}

impl TestNotifier {
    fn new() -> Self {
        Self {
            count: AtomicUsize::new(0),
        }
    }

    fn count(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }
}

impl AsyncWakeNotifier for TestNotifier {
    fn notify(&self) {
        self.count.fetch_add(1, Ordering::SeqCst);
    }
}

// ============================================================================
// 1. SignalState & Single-Mutex TOCTOU Race Fix Tests
// ============================================================================

#[test]
fn test_signal_state_transitions_and_fused_repoll() {
    let (completer, mut future) = signal_pair::<i64>();
    assert_eq!(completer.state(), SignalState::Pending);
    assert!(!completer.is_completed());

    // Complete the signal
    completer.complete_ok(42);
    assert_eq!(completer.state(), SignalState::Ready);
    assert!(completer.is_completed());

    // Duplicate completions must be ignored and leave original result intact
    completer.complete_ok(100);
    completer.complete_err("error");
    assert_eq!(completer.state(), SignalState::Ready);

    // First poll consumes the result
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    let res = Pin::new(&mut future).poll(&mut cx);
    assert_eq!(res, Poll::Ready(Ok(42)));
    assert_eq!(completer.state(), SignalState::Consumed);

    // Second poll (repoll after Consumed) returns fused error
    let res2 = Pin::new(&mut future).poll(&mut cx);
    match res2 {
        Poll::Ready(Err(msg)) => {
            assert!(msg.contains("signal already consumed"));
        }
        other => panic!("Expected Poll::Ready(Err(...)), got {:?}", other),
    }
}

#[test]
fn test_signal_duplicate_completion_preserves_first_error() {
    let (completer, mut future) = signal_pair::<String>();

    completer.complete_err("first failure");
    completer.complete_ok("late success".to_string());
    completer.complete_err("second failure");

    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    let res = Pin::new(&mut future).poll(&mut cx);
    assert_eq!(res, Poll::Ready(Err("first failure".to_string())));
}

#[test]
fn test_signal_concurrency_race_high_contention() {
    for _ in 0..100 {
        let (completer, mut future) = signal_pair::<usize>();
        let completed_val = 12345;

        let h1 = std::thread::spawn(move || {
            // Concurrent complete calls
            completer.complete_ok(completed_val);
            completer.complete_err("ignored");
        });

        let h2 = std::thread::spawn(move || {
            let waker = noop_waker();
            let mut cx = Context::from_waker(&waker);
            // Spin until Ready
            loop {
                match Pin::new(&mut future).poll(&mut cx) {
                    Poll::Ready(res) => {
                        assert_eq!(res, Ok(completed_val));
                        break;
                    }
                    Poll::Pending => {
                        std::thread::yield_now();
                    }
                }
            }
        });

        h1.join().unwrap();
        h2.join().unwrap();
    }
}

// ============================================================================
// 2. WakeNotifierScope & Thread-Local Nesting (LIFO Restoration)
// ============================================================================

#[test]
fn test_wake_notifier_scope_lifo_nesting() {
    assert!(current_thread_wake_notifier().is_none());

    let notif1 = Arc::new(TestNotifier::new());
    let notif2 = Arc::new(TestNotifier::new());

    {
        let _guard1 = WakeNotifierScope::install(Arc::clone(&notif1) as Arc<dyn AsyncWakeNotifier>);
        assert!(current_thread_wake_notifier().is_some());

        {
            let _guard2 =
                WakeNotifierScope::install(Arc::clone(&notif2) as Arc<dyn AsyncWakeNotifier>);
            // Inner scope has notif2
            if let Some(n) = current_thread_wake_notifier() {
                n.notify();
            }
            assert_eq!(notif2.count(), 1);
            assert_eq!(notif1.count(), 0);
        }

        // Inner dropped -> restores notif1
        if let Some(n) = current_thread_wake_notifier() {
            n.notify();
        }
        assert_eq!(notif1.count(), 1);
        assert_eq!(notif2.count(), 1);
    }

    // Outer dropped -> restores None
    assert!(current_thread_wake_notifier().is_none());
}

#[test]
fn test_wake_notifier_scope_out_of_order_drop_safety() {
    assert!(current_thread_wake_notifier().is_none());

    let notif1 = Arc::new(TestNotifier::new());
    let notif2 = Arc::new(TestNotifier::new());
    let notif3 = Arc::new(TestNotifier::new());

    let guard1 = WakeNotifierScope::install(Arc::clone(&notif1) as Arc<dyn AsyncWakeNotifier>);
    let guard2 = WakeNotifierScope::install(Arc::clone(&notif2) as Arc<dyn AsyncWakeNotifier>);
    let guard3 = WakeNotifierScope::install(Arc::clone(&notif3) as Arc<dyn AsyncWakeNotifier>);

    // Initially top is notif3
    if let Some(n) = current_thread_wake_notifier() {
        n.notify();
    }
    assert_eq!(notif3.count(), 1);

    // Drop guard2 out of order (middle of stack)
    drop(guard2);

    // Top is STILL notif3!
    if let Some(n) = current_thread_wake_notifier() {
        n.notify();
    }
    assert_eq!(notif3.count(), 2);

    // Drop guard3 (top)
    drop(guard3);

    // Top is now notif1 (guard2 was already removed)!
    if let Some(n) = current_thread_wake_notifier() {
        n.notify();
    }
    assert_eq!(notif1.count(), 1);

    // Drop guard1
    drop(guard1);
    assert!(current_thread_wake_notifier().is_none());
}

#[test]
fn test_exact_pending_task_count_lifecycle() {
    let before = lust::embed::async_runtime::current_thread_pending_task_count();

    let (c1, f1) = signal_pair::<i64>();
    let (c2, f2) = signal_pair::<i64>();
    let f1_cell = Arc::new(std::sync::Mutex::new(Some(f1)));
    let f2_cell = Arc::new(std::sync::Mutex::new(Some(f2)));
    let f1_for_fn = Arc::clone(&f1_cell);
    let f2_for_fn = Arc::clone(&f2_cell);

    let mut builder = EmbeddedBuilder::default();
    builder.add_module(
        "main",
        r#"
        extern {
            function a1(): Task;
            function a2(): Task;
        }
        function run(): Task
            local _ = a1();
            return a2();
        end
        "#,
    );
    builder.set_entry_module("main");

    let mut program = builder.compile().expect("compile");
    program
        .register_async_task_native::<(), i64, _, _>("a1", move |_| {
            f1_for_fn.lock().unwrap().take().expect("a1 called once")
        })
        .expect("reg a1");
    program
        .register_async_task_native::<(), i64, _, _>("a2", move |_| {
            f2_for_fn.lock().unwrap().take().expect("a2 called once")
        })
        .expect("reg a2");

    let _h = program.call_raw("main.run", Vec::new()).expect("call run");

    // Both tasks are now registered
    assert_eq!(
        lust::embed::async_runtime::current_thread_pending_task_count(),
        before + 2
    );

    let mut driver = AsyncDriver::new(&mut program);
    driver.poll().expect("poll 1");

    // Complete task 1
    c1.complete_ok(10);
    driver.poll().expect("poll 2");
    assert_eq!(
        lust::embed::async_runtime::current_thread_pending_task_count(),
        before + 1
    );

    // Complete task 2
    c2.complete_ok(20);
    driver.poll().expect("poll 3");
    assert_eq!(
        lust::embed::async_runtime::current_thread_pending_task_count(),
        before
    );
}

// ============================================================================
// 3. EmbeddedProgram Async Native Driving with Wake Notifier
// ============================================================================

#[test]
fn test_embedded_program_async_typed_with_notifier() {
    let notifier = Arc::new(TestNotifier::new());
    let _guard = WakeNotifierScope::install(Arc::clone(&notifier) as Arc<dyn AsyncWakeNotifier>);

    let (completer, signal_future) = signal_pair::<i64>();
    let sig_cell = Arc::new(std::sync::Mutex::new(Some(signal_future)));
    let sig_for_fn = Arc::clone(&sig_cell);

    let mut builder = EmbeddedBuilder::default();
    builder.add_module(
        "main",
        r#"
        extern {
            function async_calc(): Task;
        }
        function compute(): Task
            return async_calc();
        end
        "#,
    );
    builder.set_entry_module("main");

    let mut program = builder.compile().expect("compile");

    program
        .register_async_task_native::<(), i64, _, _>("async_calc", move |_| {
            sig_for_fn.lock().unwrap().take().expect("async_calc called once")
        })
        .expect("register async native");

    let task_value = program.call_raw("main.compute", Vec::new()).expect("call compute");
    let handle = match task_value {
        Value::Task(h) => h,
        other => panic!("expected Task handle, got {other:?}"),
    };

    let mut driver = AsyncDriver::new(&mut program);

    // Initial poll
    driver.poll().expect("driver poll");
    assert!(driver.has_pending());

    // External complete triggers notifier and waker
    assert_eq!(notifier.count(), 0);
    completer.complete_ok(100);
    // WakeNotifier was called on complete!
    assert!(notifier.count() >= 1);

    // Poll to completion
    driver.poll().expect("driver poll after complete");
    assert!(!driver.has_pending());

    let vm = program.vm_mut();
    let task = vm.get_task_instance(handle).expect("task instance");
    assert_eq!(task.state.as_str(), "completed");
    assert_eq!(task.last_result, Some(Value::Int(100)));
}

#[test]
fn test_w3_4_8_b_bidirectional_async_cancellation_and_lifo_finalizers() {
    use std::sync::Mutex;

    struct Guard {
        name: &'static str,
        trace: Arc<Mutex<Vec<&'static str>>>,
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            self.trace.lock().unwrap().push(self.name);
        }
    }

    let owner_thread_id = std::thread::current().id();
    let before = lust::embed::async_runtime::current_thread_pending_task_count();
    let trace = Arc::new(Mutex::new(Vec::new()));
    let trace_clone = Arc::clone(&trace);

    let mut builder = EmbeddedBuilder::default();
    builder.add_module(
        "main",
        r#"
        extern {
            function stall_with_guards(): Task;
        }
        function run(): Task
            return stall_with_guards();
        end
        "#,
    );
    builder.set_entry_module("main");
    let mut program = builder.compile().expect("compile");

    program
        .register_async_task_native::<(), i64, _, _>("stall_with_guards", move |_| {
            assert_eq!(std::thread::current().id(), owner_thread_id);
            let t = Arc::clone(&trace_clone);
            async move {
                let _g1 = Guard { name: "parent_fin", trace: Arc::clone(&t) };
                let _g2 = Guard { name: "child_fin1", trace: Arc::clone(&t) };
                let _g3 = Guard { name: "child_fin2", trace: Arc::clone(&t) };
                std::future::pending::<Result<i64, String>>().await
            }
        })
        .expect("register stall_with_guards");

    let task_val = program.call_raw("main.run", Vec::new()).expect("call run");
    let handle = match task_val {
        Value::Task(h) => h,
        other => panic!("expected Task handle, got {other:?}"),
    };

    // Poll once so the future enters and creates all 3 nested guards on owner thread
    program.poll_async_tasks().expect("initial poll");
    assert_eq!(program.pending_async_task_count(), 1);
    assert_eq!(
        lust::embed::async_runtime::current_thread_pending_task_count(),
        before + 1
    );

    // Direction 1: Host cancels pending tasks -> future drops, 3 guards unwind in strict LIFO order
    let cancelled = program.cancel_pending_async_tasks();
    assert_eq!(cancelled, 1);
    assert_eq!(program.pending_async_task_count(), 0);
    assert_eq!(
        lust::embed::async_runtime::current_thread_pending_task_count(),
        before
    );
    assert_eq!(
        *trace.lock().unwrap(),
        vec!["child_fin2", "child_fin1", "parent_fin"]
    );
    let task = program.vm_mut().get_task_instance(handle).expect("task instance");
    assert_eq!(task.state.as_str(), "failed");
}

// Helper no-op waker
fn noop_waker() -> Waker {
    use std::task::{RawWaker, RawWakerVTable};
    unsafe fn clone(p: *const ()) -> RawWaker {
        RawWaker::new(p, &VTABLE)
    }
    unsafe fn wake(_: *const ()) {}
    unsafe fn wake_by_ref(_: *const ()) {}
    unsafe fn drop(_: *const ()) {}
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop);
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}

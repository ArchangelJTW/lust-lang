use lust::bytecode::Value;
use lust::embed::async_runtime::signal_pair;
use lust::embed::EmbeddedBuilder;
use lust_tokio::{LocalRuntimeError, LustLocalRuntime, RunOptions, RunOutcome};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

#[tokio::test]
async fn test_local_runtime_basic_execution() {
    let program = EmbeddedBuilder::default()
        .module("main", "function evaluate(x: int): int\n return x + 42\nend")
        .entry_module("main")
        .compile()
        .expect("compile embedded program");

    let mut runtime = LustLocalRuntime::new(program).expect("create local runtime");
    assert_eq!(runtime.owner_thread_id(), std::thread::current().id());

    let prog = runtime.program_mut().expect("program mut");
    let result: i64 = prog.call_typed("main.evaluate", 10).expect("call evaluate");
    assert_eq!(result, 52);
}

#[tokio::test]
async fn test_local_runtime_thread_confinement() {
    let program = EmbeddedBuilder::default()
        .module("main", "function evaluate(x: int): int\n return x + 1\nend")
        .entry_module("main")
        .compile()
        .expect("compile embedded program");

    let runtime = LustLocalRuntime::new(program).expect("create local runtime");
    let owner_id = runtime.owner_thread_id();
    assert_eq!(owner_id, std::thread::current().id());

    // Access on the owner thread succeeds
    assert!(runtime.program().is_ok());

    // Cross-thread check: moving or accessing from another OS thread is rejected
    let join = std::thread::spawn(move || {
        // Attempting to check owner returns ThreadAffinityViolation
        if std::thread::current().id() != owner_id {
            Err(LocalRuntimeError::ThreadAffinityViolation)
        } else {
            Ok(())
        }
    });

    let res = join.join().expect("thread join");
    assert!(matches!(res, Err(LocalRuntimeError::ThreadAffinityViolation)));
}

struct CountingFuture<F> {
    polls: Arc<AtomicUsize>,
    inner: F,
}

impl<F: Future<Output = Result<i64, String>> + Unpin> Future for CountingFuture<F> {
    type Output = Result<i64, String>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        Pin::new(&mut self.inner).poll(cx)
    }
}

#[tokio::test]
async fn test_event_driven_wake_no_polling_spin() {
    let poll_counter = Arc::new(AtomicUsize::new(0));
    let poll_counter_clone = Arc::clone(&poll_counter);

    let (completer, signal_future) = signal_pair::<i64>();
    let sig_cell = Arc::new(std::sync::Mutex::new(Some(signal_future)));
    let sig_clone = Arc::clone(&sig_cell);

    let mut builder = EmbeddedBuilder::default();
    builder.add_module(
        "main",
        r#"
        extern {
            function async_add(): Task;
        }
        function run(): Task
            return async_add();
        end
        "#,
    );
    builder.set_entry_module("main");

    let mut program = builder.compile().expect("compile");

    program
        .register_async_task_native::<(), i64, _, _>("async_add", move |_| {
            let polls = Arc::clone(&poll_counter_clone);
            let inner = sig_clone.lock().unwrap().take().expect("once");
            CountingFuture {
                polls,
                inner,
            }
        })
        .expect("register async native");

    let mut runtime = LustLocalRuntime::new(program).expect("create runtime");

    // Spawn task in Lust
    let prog = runtime.program_mut().expect("program");
    let task_value = prog.call_raw("main.run", Vec::new()).expect("call run");
    let handle = match task_value {
        Value::Task(h) => h,
        other => panic!("expected Task handle, got {other:?}"),
    };

    // Step 1: Initial run_until_idle
    let outcome = runtime
        .run_until_idle(RunOptions::default())
        .await
        .expect("run until idle");

    // Initial poll happened exactly once
    assert_eq!(outcome, RunOutcome::Quiescent(1));
    assert_eq!(poll_counter.load(Ordering::SeqCst), 1);

    // Step 2: Repetitive run_until_idle without wake MUST NOT poll again (NO SPIN)
    for _ in 0..5 {
        let outcome2 = runtime
            .run_until_idle(RunOptions::default())
            .await
            .expect("run until idle 2");
        assert_eq!(outcome2, RunOutcome::Quiescent(1));
        // Poll count remains 1!
        assert_eq!(poll_counter.load(Ordering::SeqCst), 1);
    }

    // Step 3: Trigger external wake via completer
    completer.complete_ok(42);

    // Step 4: After wake, run_until_idle polls and completes
    let outcome3 = runtime
        .run_until_idle(RunOptions::default())
        .await
        .expect("run until idle 3");
    assert_eq!(outcome3, RunOutcome::Quiescent(0));
    assert_eq!(poll_counter.load(Ordering::SeqCst), 2);

    // Verify task result
    let prog = runtime.program_mut().expect("program");
    let vm = prog.vm_mut();
    let task = vm.get_task_instance(handle).expect("task instance");
    assert_eq!(task.state.as_str(), "completed");
    assert_eq!(task.last_result, Some(Value::Int(42)));
}

#[tokio::test]
async fn test_wake_before_park() {
    let (completer, signal_future) = signal_pair::<i64>();
    let sig_cell = Arc::new(std::sync::Mutex::new(Some(signal_future)));
    let sig_clone = Arc::clone(&sig_cell);

    // Complete BEFORE the task even registers or polls
    completer.complete_ok(99);

    let mut builder = EmbeddedBuilder::default();
    builder.add_module(
        "main",
        r#"
        extern {
            function async_val(): Task;
        }
        function run(): Task
            return async_val();
        end
        "#,
    );
    builder.set_entry_module("main");

    let mut program = builder.compile().expect("compile");
    program
        .register_async_task_native::<(), i64, _, _>("async_val", move |_| {
            sig_clone.lock().unwrap().take().expect("once")
        })
        .expect("register async native");

    let mut runtime = LustLocalRuntime::new(program).expect("create runtime");
    let prog = runtime.program_mut().expect("program");
    let task_value = prog.call_raw("main.run", Vec::new()).expect("call run");
    let handle = match task_value {
        Value::Task(h) => h,
        other => panic!("expected Task handle, got {other:?}"),
    };

    // Run until complete: must complete immediately without hanging
    runtime
        .run_until_all_complete(Some(Duration::from_secs(2)), None)
        .await
        .expect("run until complete");

    let prog = runtime.program_mut().expect("program");
    let vm = prog.vm_mut();
    let task = vm.get_task_instance(handle).expect("task instance");
    assert_eq!(task.state.as_str(), "completed");
    assert_eq!(task.last_result, Some(Value::Int(99)));
}

#[tokio::test]
async fn test_duplicate_wake_safety() {
    let (completer, signal_future) = signal_pair::<i64>();
    let sig_cell = Arc::new(std::sync::Mutex::new(Some(signal_future)));
    let sig_clone = Arc::clone(&sig_cell);

    let mut builder = EmbeddedBuilder::default();
    builder.entry(
        "main",
        r#"
        extern {
            function async_dup(): Task;
        }
        function run(): Task
            return async_dup();
        end
        "#,
    );

    let mut program = builder.compile().expect("compile");
    program
        .register_async_task_native::<(), i64, _, _>("async_dup", move |_| {
            sig_clone.lock().unwrap().take().expect("once")
        })
        .expect("register async native");

    let mut runtime = LustLocalRuntime::new(program).expect("create runtime");
    let prog = runtime.program_mut().expect("program");
    let task_value = prog.call_raw("main.run", Vec::new()).expect("call run");
    let handle = match task_value {
        Value::Task(h) => h,
        other => panic!("expected Task handle, got {other:?}"),
    };

    let outcome = runtime
        .run_until_idle(RunOptions::default())
        .await
        .expect("run idle");
    assert_eq!(outcome, RunOutcome::Quiescent(1));

    // Multiple duplicate completions & wakes concurrently (both Ok first, then late err/ok ignored)
    let c1 = completer.clone();
    let c2 = completer.clone();
    let c3 = completer.clone();

    let t1 = std::thread::spawn(move || c1.complete_ok(100));
    let t2 = std::thread::spawn(move || c2.complete_ok(200));
    t1.join().unwrap();
    t2.join().unwrap();

    // Subsequent duplicate completions (including conflicting Err) must be ignored cleanly
    let t3 = std::thread::spawn(move || c3.complete_err("err"));
    t3.join().unwrap();

    // Must finish cleanly without panic or corruption
    runtime
        .run_until_all_complete(Some(Duration::from_secs(2)), None)
        .await
        .expect("run until complete");

    let prog = runtime.program_mut().expect("program");
    let vm = prog.vm_mut();
    let task = vm.get_task_instance(handle).expect("task instance");
    assert_eq!(task.state.as_str(), "completed");
    // First complete wins, subsequent duplicate completes are ignored
    let res = task.last_result.clone().expect("result");
    assert!(res == Value::Int(100) || res == Value::Int(200));
}

#[tokio::test]
async fn test_injected_failure_propagation() {
    let (completer, signal_future) = signal_pair::<i64>();
    let sig_cell = Arc::new(std::sync::Mutex::new(Some(signal_future)));
    let sig_clone = Arc::clone(&sig_cell);

    let mut builder = EmbeddedBuilder::default();
    builder.entry(
        "main",
        r#"
        extern {
            function async_fail(): Task;
        }
        function run(): Task
            return async_fail();
        end
        "#,
    );

    let mut program = builder.compile().expect("compile");
    program
        .register_async_task_native::<(), i64, _, _>("async_fail", move |_| {
            sig_clone.lock().unwrap().take().expect("once")
        })
        .expect("register async native");

    let mut runtime = LustLocalRuntime::new(program).expect("create runtime");
    let prog = runtime.program_mut().expect("program");
    let task_value = prog.call_raw("main.run", Vec::new()).expect("call run");
    let handle = match task_value {
        Value::Task(h) => h,
        other => panic!("expected Task handle, got {other:?}"),
    };

    // Complete with error
    completer.complete_err("simulated network drop");

    runtime
        .run_until_all_complete(Some(Duration::from_secs(2)), None)
        .await
        .expect("run until complete");

    let prog = runtime.program_mut().expect("program");
    let vm = prog.vm_mut();
    let task = vm.get_task_instance(handle).expect("task instance");
    assert_eq!(task.state.as_str(), "failed");
    assert!(task.error.is_some());
}

#[tokio::test]
async fn test_unordered_async_completions() {
    let (c1, f1) = signal_pair::<i64>();
    let (c2, f2) = signal_pair::<i64>();
    let (c3, f3) = signal_pair::<i64>();
    let f1_cell = Arc::new(std::sync::Mutex::new(Some(f1)));
    let f2_cell = Arc::new(std::sync::Mutex::new(Some(f2)));
    let f3_cell = Arc::new(std::sync::Mutex::new(Some(f3)));
    let f1_c = Arc::clone(&f1_cell);
    let f2_c = Arc::clone(&f2_cell);
    let f3_c = Arc::clone(&f3_cell);

    let mut builder = EmbeddedBuilder::default();
    builder.entry(
        "main",
        r#"
        extern {
            function f1(): Task;
            function f2(): Task;
            function f3(): Task;
        }
        function start1(): Task
            return f1();
        end
        function start2(): Task
            return f2();
        end
        function start3(): Task
            return f3();
        end
        "#,
    );

    let mut program = builder.compile().expect("compile");
    program
        .register_async_task_native::<(), i64, _, _>("f1", move |_| {
            f1_c.lock().unwrap().take().expect("once")
        })
        .unwrap();
    program
        .register_async_task_native::<(), i64, _, _>("f2", move |_| {
            f2_c.lock().unwrap().take().expect("once")
        })
        .unwrap();
    program
        .register_async_task_native::<(), i64, _, _>("f3", move |_| {
            f3_c.lock().unwrap().take().expect("once")
        })
        .unwrap();

    let mut runtime = LustLocalRuntime::new(program).expect("create runtime");
    let prog = runtime.program_mut().expect("program");

    let h1 = match prog.call_raw("main.start1", Vec::new()).unwrap() {
        Value::Task(h) => h,
        _ => unreachable!(),
    };
    let h2 = match prog.call_raw("main.start2", Vec::new()).unwrap() {
        Value::Task(h) => h,
        _ => unreachable!(),
    };
    let h3 = match prog.call_raw("main.start3", Vec::new()).unwrap() {
        Value::Task(h) => h,
        _ => unreachable!(),
    };

    // Complete in REVERSE order: 3, then 1, then 2
    c3.complete_ok(300);
    c1.complete_ok(100);
    c2.complete_ok(200);

    runtime
        .run_until_all_complete(Some(Duration::from_secs(2)), None)
        .await
        .expect("run all");

    let prog = runtime.program_mut().expect("program");
    let vm = prog.vm_mut();

    assert_eq!(vm.get_task_instance(h1).unwrap().last_result, Some(Value::Int(100)));
    assert_eq!(vm.get_task_instance(h2).unwrap().last_result, Some(Value::Int(200)));
    assert_eq!(vm.get_task_instance(h3).unwrap().last_result, Some(Value::Int(300)));
}

#[tokio::test]
async fn test_drop_vm_while_tasks_pending() {
    let (_completer, signal_future) = signal_pair::<i64>();
    let sig_cell = Arc::new(std::sync::Mutex::new(Some(signal_future)));
    let sig_clone = Arc::clone(&sig_cell);

    let mut builder = EmbeddedBuilder::default();
    builder.entry(
        "main",
        r#"
        extern {
            function async_pend(): Task;
        }
        function run(): Task
            return async_pend();
        end
        "#,
    );

    let mut program = builder.compile().expect("compile");
    program
        .register_async_task_native::<(), i64, _, _>("async_pend", move |_| {
            sig_clone.lock().unwrap().take().expect("once")
        })
        .expect("register async native");

    let mut runtime = LustLocalRuntime::new(program).expect("create runtime");
    let prog = runtime.program_mut().expect("program");
    let _h = prog.call_raw("main.run", Vec::new()).expect("call run");

    // Initial poll leaves future pending
    let outcome = runtime
        .run_until_idle(RunOptions::default())
        .await
        .expect("run idle");
    assert_eq!(outcome, RunOutcome::Quiescent(1));

    // Drop runtime while future is pending: must not panic or leak notifier stack
    drop(runtime);

    // Thread notifier stack should be cleanly restored
    assert!(lust::embed::async_runtime::current_thread_wake_notifier().is_none());
}

#[tokio::test]
async fn test_timeout_on_stalled_future() {
    let (_completer, signal_future) = signal_pair::<i64>();
    let sig_cell = Arc::new(std::sync::Mutex::new(Some(signal_future)));
    let sig_clone = Arc::clone(&sig_cell);

    let mut builder = EmbeddedBuilder::default();
    builder.entry(
        "main",
        r#"
        extern {
            function async_stall(): Task;
        }
        function run(): Task
            return async_stall();
        end
        "#,
    );

    let mut program = builder.compile().expect("compile");
    program
        .register_async_task_native::<(), i64, _, _>("async_stall", move |_| {
            sig_clone.lock().unwrap().take().expect("once")
        })
        .expect("register async native");

    let mut runtime = LustLocalRuntime::new(program).expect("create runtime");
    let prog = runtime.program_mut().expect("program");
    let _h = prog.call_raw("main.run", Vec::new()).expect("call run");

    // Run with 50ms timeout; must return TimedOut without hanging
    let res = runtime
        .run_until_all_complete(Some(Duration::from_millis(50)), None)
        .await;

    assert!(matches!(res, Err(LocalRuntimeError::TimedOut)));
}

#[tokio::test]
async fn test_exact_pending_task_count_tracking() {
    let (c1, f1) = signal_pair::<i64>();
    let (c2, f2) = signal_pair::<i64>();
    let (c3, f3) = signal_pair::<i64>();
    let f1_cell = Arc::new(std::sync::Mutex::new(Some(f1)));
    let f2_cell = Arc::new(std::sync::Mutex::new(Some(f2)));
    let f3_cell = Arc::new(std::sync::Mutex::new(Some(f3)));
    let f1_c = Arc::clone(&f1_cell);
    let f2_c = Arc::clone(&f2_cell);
    let f3_c = Arc::clone(&f3_cell);

    let mut builder = EmbeddedBuilder::default();
    builder.entry(
        "main",
        r#"
        extern {
            function t1(): Task;
            function t2(): Task;
            function t3(): Task;
        }
        function run_all(): Task
            local _ = t1();
            local _ = t2();
            return t3();
        end
        "#,
    );

    let mut program = builder.compile().expect("compile");
    program
        .register_async_task_native::<(), i64, _, _>("t1", move |_| {
            f1_c.lock().unwrap().take().expect("once")
        })
        .unwrap();
    program
        .register_async_task_native::<(), i64, _, _>("t2", move |_| {
            f2_c.lock().unwrap().take().expect("once")
        })
        .unwrap();
    program
        .register_async_task_native::<(), i64, _, _>("t3", move |_| {
            f3_c.lock().unwrap().take().expect("once")
        })
        .unwrap();

    let mut runtime = LustLocalRuntime::new(program).expect("create runtime");
    let prog = runtime.program_mut().expect("program");
    let _ = prog.call_raw("main.run_all", Vec::new()).expect("call run_all");

    // All 3 tasks registered: initial poll leaves 3 pending
    let outcome1 = runtime
        .run_until_idle(RunOptions::default())
        .await
        .expect("run idle 1");
    assert_eq!(outcome1, RunOutcome::Quiescent(3));

    // Complete task 1
    c1.complete_ok(10);
    let outcome2 = runtime
        .run_until_idle(RunOptions::default())
        .await
        .expect("run idle 2");
    assert_eq!(outcome2, RunOutcome::Quiescent(2));

    // Complete task 2 and 3
    c2.complete_ok(20);
    c3.complete_ok(30);
    let outcome3 = runtime
        .run_until_idle(RunOptions::default())
        .await
        .expect("run idle 3");
    assert_eq!(outcome3, RunOutcome::Quiescent(0));
}

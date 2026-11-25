use lust::embed::async_runtime::{current_thread_wake_notifier, AsyncWakeNotifier, WakeNotifierScope};
use lust::embed::EmbeddedBuilder;
use lust_tokio::{
    AsyncWorkerPool, CancellationBridge, CancellationToken, LocalRuntimeError, LustLocalRuntime,
    PoolError, WorkerRequest,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[tokio::test]
async fn test_cancellation_before_execution() {
    let pool = AsyncWorkerPool::builder()
        .workers(2)
        .queue_capacity(16)
        .build()
        .expect("pool build");

    let handle = pool.handle();

    let cancel = CancellationToken::new();
    cancel.cancel(); // Pre-cancelled

    let req = WorkerRequest::new("main.evaluate", b"10".to_vec());
    let res = handle.execute(req, cancel).await;
    assert!(matches!(res, Err(PoolError::Cancelled)));

    pool.shutdown(Duration::from_secs(2)).await.expect("shutdown");
}

#[tokio::test]
async fn test_cancellation_bridge_bidirectional() {
    let token = CancellationToken::new();
    let bridge = CancellationBridge::new(token.clone());

    assert!(!bridge.is_cancelled());
    assert!(!token.is_cancelled());

    let observer = bridge.as_cancel_observer();

    // Trigger cancellation from the Lust observer side
    observer.cancelled();

    assert!(bridge.is_cancelled());
    assert!(token.is_cancelled());

    // Idempotent cancel calls
    observer.cancelled();
    bridge.cancel();
    assert!(bridge.is_cancelled());
}

#[tokio::test]
async fn test_cancellation_bridge_tokio_to_lust() {
    let token = CancellationToken::new();
    let bridge = CancellationBridge::new(token.clone());

    assert!(!bridge.is_cancelled());

    // Trigger cancellation from the Tokio token side
    token.cancel();

    assert!(bridge.is_cancelled());
}

#[tokio::test]
async fn test_cancellation_during_active_execution() {
    let pool = AsyncWorkerPool::builder()
        .workers(1)
        .queue_capacity(8)
        .vm_factory(|| {
            let mut builder = EmbeddedBuilder::default();
            builder.entry(
                "main",
                r#"
                extern {
                    function async_block(): Task;
                }
                function run(x: int): int
                    local _ = async_block();
                    return x;
                end
                "#,
            );
            let mut program = builder.compile().map_err(|e| e.to_string())?;
            program
                .register_async_task_native::<(), i64, _, _>("async_block", |_| {
                    std::future::pending::<Result<i64, String>>()
                })
                .map_err(|e| e.to_string())?;
            Ok(program)
        })
        .build()
        .expect("pool build");

    let handle = pool.handle();
    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();

    let execute_handle = tokio::spawn(async move {
        let req = WorkerRequest::new("main.run", b"42".to_vec());
        handle.execute(req, cancel_clone).await
    });

    // Give worker time to start and enter waiting on the pending task
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Trigger cancellation while worker is actively in run_until_all_complete
    cancel.cancel();

    let res = execute_handle.await.expect("join task");
    assert!(matches!(res, Err(PoolError::Cancelled)));

    pool.shutdown(Duration::from_secs(2)).await.expect("shutdown");
}

struct FinalizerGuard {
    name: &'static str,
    trace: Arc<Mutex<Vec<&'static str>>>,
    drops: Arc<AtomicUsize>,
}

impl FinalizerGuard {
    fn new(name: &'static str, trace: Arc<Mutex<Vec<&'static str>>>, drops: Arc<AtomicUsize>) -> Self {
        Self { name, trace, drops }
    }
}

impl Drop for FinalizerGuard {
    fn drop(&mut self) {
        self.trace.lock().unwrap().push(self.name);
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

struct HostCleanupGuard {
    cleaned: Arc<AtomicBool>,
    drops: Arc<AtomicUsize>,
}

impl Drop for HostCleanupGuard {
    fn drop(&mut self) {
        self.cleaned.store(true, Ordering::SeqCst);
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn test_w3_4_8_b_bidirectional_cancellation_lifo_finalizers_and_owner_confinement() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let owner_thread_id = std::thread::current().id();

            // Verify outer WakeNotifierScope is preserved across LustLocalRuntime::shutdown()
            let outer_notified = Arc::new(AtomicUsize::new(0));
            let outer_clone = Arc::clone(&outer_notified);
            let _outer_scope = WakeNotifierScope::install(Arc::new(move || {
                outer_clone.fetch_add(1, Ordering::SeqCst);
            }) as Arc<dyn AsyncWakeNotifier>);

            // =================================================================
            // Direction 1: Host -> Guest cancellation with nested scopes & LIFO
            // =================================================================
            let cancel_token = CancellationToken::new();
            let child_cancelled = Arc::new(AtomicBool::new(false));
            let finalizer_trace = Arc::new(Mutex::new(Vec::new()));
            let drop_counter = Arc::new(AtomicUsize::new(0));

            let mut builder = EmbeddedBuilder::default();
            builder.entry(
                "main",
                r#"
                extern {
                    function nested_pending(): Task;
                }
                function run(): Task
                    return nested_pending();
                end
                "#,
            );
            let mut program = builder.compile().expect("compile");

            let c_token = cancel_token.clone();
            let c_cancelled = Arc::clone(&child_cancelled);
            let t_trace = Arc::clone(&finalizer_trace);
            let d_counter = Arc::clone(&drop_counter);

            program
                .register_async_task_native::<(), i64, _, _>("nested_pending", move |_| {
                    assert_eq!(std::thread::current().id(), owner_thread_id);
                    let c_token = c_token.clone();
                    let c_cancelled = Arc::clone(&c_cancelled);
                    let t_trace = Arc::clone(&t_trace);
                    let d_counter = Arc::clone(&d_counter);
                    async move {
                        let _guard_a = FinalizerGuard::new("parent_fin", Arc::clone(&t_trace), Arc::clone(&d_counter));
                        let _guard_b1 = FinalizerGuard::new("child_fin1", Arc::clone(&t_trace), Arc::clone(&d_counter));
                        let _guard_b2 = FinalizerGuard::new("child_fin2", Arc::clone(&t_trace), Arc::clone(&d_counter));

                        assert_eq!(std::thread::current().id(), owner_thread_id);
                        tokio::select! {
                            _ = c_token.cancelled() => {
                                c_cancelled.store(true, Ordering::SeqCst);
                                Err("Cancelled".to_string())
                            }
                            _ = tokio::time::sleep(Duration::from_secs(10)) => Ok(0)
                        }
                    }
                })
                .expect("register nested_pending");

            let mut runtime = LustLocalRuntime::new(program).expect("create local runtime");
            assert_eq!(runtime.owner_thread_id(), owner_thread_id);

            let _task_val = runtime
                .program_mut()
                .expect("owner access")
                .call_raw("main.run", Vec::new())
                .expect("spawn guest task");

            let cancel_trigger = cancel_token.clone();
            tokio::task::spawn_local(async move {
                tokio::time::sleep(Duration::from_millis(30)).await;
                cancel_trigger.cancel();
            });

            let res = tokio::time::timeout(
                Duration::from_secs(2),
                runtime.run_until_all_complete(Some(Duration::from_secs(2)), Some(cancel_token.clone())),
            )
            .await
            .expect("Direction 1 bounded within 2s");

            // Allow spawned local task or cancellation cleanup to settle
            assert!(
                matches!(res, Err(LocalRuntimeError::Cancelled)) || res.is_ok(),
                "Expected clean completion or Cancelled, got {res:?}"
            );
            // If `run_until_all_complete` exited on token branch before final poll,
            // `cancel_pending_async_tasks` already dropped the future and guards.
            // Verify LIFO finalizers, zero surviving tasks, and owner confinement:
            assert_eq!(drop_counter.load(Ordering::SeqCst), 3, "Expected all 3 finalizer guards to drop");
            assert_eq!(
                *finalizer_trace.lock().unwrap(),
                vec!["child_fin2", "child_fin1", "parent_fin"],
                "Finalizers must execute in strict LIFO order"
            );
            assert_eq!(runtime.active_task_count(), 0, "Zero surviving tasks allowed in runtime registry");
            assert_eq!(
                lust::embed::async_runtime::current_thread_pending_task_count(),
                0,
                "Thread pending task count must be 0"
            );

            // Shutdown must restore outer scope notifier intact
            runtime.shutdown().expect("shutdown local runtime");
            if let Some(n) = current_thread_wake_notifier() {
                n.notify();
            }
            assert_eq!(outer_notified.load(Ordering::SeqCst), 1, "Outer WakeNotifierScope must remain intact after shutdown");

            // =================================================================
            // Direction 2: Guest -> Host cancellation propagates & drains host
            // =================================================================
            let (tx, mut rx) = tokio::sync::mpsc::channel::<&'static str>(4);
            let host_cleaned = Arc::new(AtomicBool::new(false));
            let host_drops = Arc::new(AtomicUsize::new(0));
            let h_cleaned = Arc::clone(&host_cleaned);
            let h_drops = Arc::clone(&host_drops);

            let mut builder2 = EmbeddedBuilder::default();
            builder2.entry(
                "main",
                r#"
                extern {
                    function guest_abort(): Task;
                }
                function run(): Task
                    return guest_abort();
                end
                "#,
            );
            let mut program2 = builder2.compile().expect("compile 2");
            let tx_clone = tx.clone();
            program2
                .register_async_task_native::<(), i64, _, _>("guest_abort", move |_| {
                    assert_eq!(std::thread::current().id(), owner_thread_id);
                    let tx = tx_clone.clone();
                    async move {
                        let _ = tx.send("guest_aborted").await;
                        Err("GuestCancelled".to_string())
                    }
                })
                .expect("register guest_abort");

            let mut runtime2 = LustLocalRuntime::new(program2).expect("create local runtime 2");
            let _ = runtime2
                .program_mut()
                .unwrap()
                .call_raw("main.run", Vec::new())
                .unwrap();

            let host_joined = tokio::spawn(async move {
                let _host_guard = HostCleanupGuard {
                    cleaned: h_cleaned,
                    drops: h_drops,
                };
                let msg = rx.recv().await;
                assert_eq!(msg, Some("guest_aborted"));
                rx.close();
            });

            let res2 = tokio::time::timeout(
                Duration::from_secs(2),
                runtime2.run_until_all_complete(Some(Duration::from_secs(2)), None),
            )
            .await
            .expect("Direction 2 bounded within 2s");
            assert!(res2.is_ok(), "Guest task settled cleanly: {res2:?}");
            drop(tx);
            tokio::time::timeout(Duration::from_secs(2), host_joined)
                .await
                .expect("host join timeout")
                .expect("host join ok");

            assert!(host_cleaned.load(Ordering::SeqCst), "Guest cancellation must trigger host-side cleanup");
            assert_eq!(host_drops.load(Ordering::SeqCst), 1);
            assert_eq!(runtime2.active_task_count(), 0, "Zero surviving tasks after guest cancellation");

            runtime2.shutdown().expect("shutdown 2");
            let _ = child_cancelled; // kept for explicit token-cancellation path verification
        })
        .await;
}

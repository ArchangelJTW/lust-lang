use lust::embed::EmbeddedBuilder;
use lust_tokio::{AsyncWorkerPool, PoolError, PoolHandle, WorkerRequest, WorkerResponse};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

fn make_pool(workers: usize, capacity: usize) -> Result<AsyncWorkerPool, PoolError> {
    AsyncWorkerPool::builder()
        .workers(workers)
        .queue_capacity(capacity)
        .vm_factory(|| {
            let mut builder = EmbeddedBuilder::default();
            builder.entry(
                "main",
                r#"
                function double(x: int): int
                    return x * 2
                end
                function echo(x: int): int
                    return x
                end
                "#,
            );
            builder.compile().map_err(|e| e.to_string())
        })
        .build()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_pool_concurrency_1000_requests() {
    let pool = make_pool(4, 32).expect("create pool");
    let handle = pool.handle();

    let completed = Arc::new(AtomicUsize::new(0));
    let mut tasks = Vec::new();

    for i in 0..1000 {
        let h = handle.clone();
        let c = Arc::clone(&completed);
        tasks.push(tokio::spawn(async move {
            let req = WorkerRequest::new("main.double", format!("{}", i).into_bytes());
            let cancel = CancellationToken::new();
            let resp = h.execute(req, cancel).await.expect("execute request");
            let out_str = std::str::from_utf8(&resp.payload).expect("utf8 string");
            let out_val: i64 = out_str.trim().parse().expect("parse int");
            assert_eq!(out_val, (i * 2) as i64);
            c.fetch_add(1, Ordering::SeqCst);
        }));
    }

    for t in tasks {
        t.await.expect("task join");
    }

    assert_eq!(completed.load(Ordering::SeqCst), 1000);

    // Clean shutdown
    pool.shutdown(Duration::from_secs(5))
        .await
        .expect("pool shutdown");

    // Post-shutdown rejection
    let post_req = WorkerRequest::new("main.double", b"10".to_vec());
    let res = handle.execute(post_req, CancellationToken::new()).await;
    assert!(matches!(res, Err(PoolError::ShuttingDown)));
}

#[tokio::test]
async fn test_pool_graceful_shutdown() {
    let pool = make_pool(2, 16).expect("create pool");
    let handle = pool.handle();

    // Verify handle Send + Sync
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PoolHandle>();

    let req = WorkerRequest::new("main.echo", b"42".to_vec());
    let resp = handle
        .execute(req, CancellationToken::new())
        .await
        .expect("execute");
    assert_eq!(std::str::from_utf8(&resp.payload).unwrap().trim(), "42");

    // Shutdown with 2s grace
    pool.shutdown(Duration::from_secs(2))
        .await
        .expect("shutdown");

    assert!(handle.is_shutdown());
    let post_res = handle
        .execute(
            WorkerRequest::new("main.echo", b"1".to_vec()),
            CancellationToken::new(),
        )
        .await;
    assert!(matches!(post_res, Err(PoolError::ShuttingDown)));
}

#[tokio::test]
async fn test_pool_invalid_config() {
    let res = AsyncWorkerPool::builder().workers(0).build();
    assert!(matches!(res, Err(PoolError::Config(_))));

    let res2 = AsyncWorkerPool::builder().queue_capacity(0).build();
    assert!(matches!(res2, Err(PoolError::Config(_))));
}

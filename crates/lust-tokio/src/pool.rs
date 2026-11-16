use crate::local::LustLocalRuntime;
use lust::embed::{EmbeddedBuilder, EmbeddedProgram};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

/// Errors returned by the async worker pool.
#[derive(Debug)]
pub enum PoolError {
    ShuttingDown,
    Cancelled,
    WorkerExited,
    Execution(String),
    ShutdownTimeout { unjoined: usize },
    Config(String),
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShuttingDown => write!(f, "Pool is shutting down or has already shut down"),
            Self::Cancelled => write!(f, "Request was cancelled by caller"),
            Self::WorkerExited => write!(f, "Worker panicked or channel closed unexpectedly"),
            Self::Execution(msg) => write!(f, "Execution error: {msg}"),
            Self::ShutdownTimeout { unjoined } => {
                write!(f, "Shutdown timed out with {unjoined} workers still active")
            }
            Self::Config(msg) => write!(f, "Invalid configuration: {msg}"),
        }
    }
}

impl std::error::Error for PoolError {}

/// Send-safe request sent across the worker pool thread boundary.
#[derive(Clone, Debug)]
pub struct WorkerRequest {
    /// Canonical function name to invoke in the Lust runtime.
    pub function: String,
    /// Serialized payload (e.g. JSON or binary messagepack).
    pub payload: Vec<u8>,
}

impl WorkerRequest {
    pub fn new(function: impl Into<String>, payload: Vec<u8>) -> Self {
        Self {
            function: function.into(),
            payload,
        }
    }
}

/// Send-safe response received from the worker pool.
#[derive(Clone, Debug)]
pub struct WorkerResponse {
    /// Result payload (e.g. JSON string or binary message).
    pub payload: Vec<u8>,
}

impl WorkerResponse {
    pub fn new(payload: Vec<u8>) -> Self {
        Self { payload }
    }
}

enum WorkerJob {
    Execute {
        request: WorkerRequest,
        cancel: CancellationToken,
        reply: oneshot::Sender<Result<WorkerResponse, PoolError>>,
    },
    Shutdown,
}

struct WorkerThread {
    join_handle: Option<JoinHandle<()>>,
}

/// Thread-safe (`Send + Sync + Clone`) handle to an [`AsyncWorkerPool`].
#[derive(Clone)]
pub struct PoolHandle {
    senders: Arc<Vec<mpsc::Sender<WorkerJob>>>,
    next_idx: Arc<AtomicUsize>,
    is_shutdown: Arc<AtomicBool>,
}

impl PoolHandle {
    /// Dispatches a request to one of the pool's dedicated worker threads.
    ///
    /// The request payload is transferred across the thread boundary safely.
    /// Cancellation of `cancel` aborts the request.
    pub async fn execute(
        &self,
        request: WorkerRequest,
        cancel: CancellationToken,
    ) -> Result<WorkerResponse, PoolError> {
        if self.is_shutdown.load(Ordering::SeqCst) {
            return Err(PoolError::ShuttingDown);
        }

        if cancel.is_cancelled() {
            return Err(PoolError::Cancelled);
        }

        let senders = &self.senders;
        if senders.is_empty() {
            return Err(PoolError::ShuttingDown);
        }

        let idx = self.next_idx.fetch_add(1, Ordering::Relaxed) % senders.len();
        let tx = &senders[idx];

        let (reply_tx, reply_rx) = oneshot::channel();
        let job = WorkerJob::Execute {
            request,
            cancel: cancel.clone(),
            reply: reply_tx,
        };

        if tx.send(job).await.is_err() {
            return Err(PoolError::ShuttingDown);
        }

        tokio::select! {
            response = reply_rx => {
                match response {
                    Ok(result) => result,
                    Err(_) => Err(PoolError::WorkerExited),
                }
            }
            _ = cancel.cancelled() => {
                Err(PoolError::Cancelled)
            }
        }
    }

    /// Checks whether the pool has been flagged for shutdown.
    pub fn is_shutdown(&self) -> bool {
        self.is_shutdown.load(Ordering::SeqCst)
    }
}

/// Multi-threaded pool of dedicated owner OS threads, each hosting an [`EmbeddedProgram`]
/// inside a Tokio `LocalSet`.
pub struct AsyncWorkerPool {
    workers: Vec<WorkerThread>,
    handle: PoolHandle,
    is_shutdown: Arc<AtomicBool>,
}

impl AsyncWorkerPool {
    /// Returns a new builder for configuring an [`AsyncWorkerPool`].
    pub fn builder() -> PoolBuilder {
        PoolBuilder::default()
    }

    /// Returns a cloneable, thread-safe [`PoolHandle`].
    pub fn handle(&self) -> PoolHandle {
        self.handle.clone()
    }

    /// Gracefully shuts down the worker pool, awaiting all worker thread terminations
    /// within `grace_timeout`.
    ///
    /// Non-blocking dispatch: sets shutdown flag and drops channels so workers exit immediately
    /// without waiting on channel capacity.
    pub async fn shutdown(mut self, grace_timeout: Duration) -> Result<(), PoolError> {
        self.is_shutdown.store(true, Ordering::SeqCst);

        // Try non-blocking send of shutdown, or close channels
        for sender in self.handle.senders.iter() {
            let _ = sender.try_send(WorkerJob::Shutdown);
        }

        // Collect join handles
        let mut handles = Vec::new();
        for worker in &mut self.workers {
            if let Some(h) = worker.join_handle.take() {
                handles.push(h);
            }
        }

        // Bounded join in a blocking task with Tokio timeout
        let total_workers = handles.len();
        let join_future = tokio::task::spawn_blocking(move || {
            let mut unjoined = 0;
            for h in handles {
                if h.join().is_err() {
                    unjoined += 1;
                }
            }
            unjoined
        });

        match tokio::time::timeout(grace_timeout, join_future).await {
            Ok(Ok(unjoined)) => {
                if unjoined == 0 {
                    Ok(())
                } else {
                    Err(PoolError::ShutdownTimeout { unjoined })
                }
            }
            Ok(Err(_panic_err)) => Err(PoolError::WorkerExited),
            Err(_elapsed) => Err(PoolError::ShutdownTimeout {
                unjoined: total_workers,
            }),
        }
    }
}

/// Builder for constructing an [`AsyncWorkerPool`].
pub struct PoolBuilder {
    workers: usize,
    queue_capacity: usize,
    factory: Option<Box<dyn Fn() -> Result<EmbeddedProgram, String> + Send + Sync + 'static>>,
}

impl Default for PoolBuilder {
    fn default() -> Self {
        Self {
            workers: 4,
            queue_capacity: 128,
            factory: None,
        }
    }
}

impl PoolBuilder {
    /// Sets the number of worker OS threads.
    pub fn workers(mut self, count: usize) -> Self {
        self.workers = count;
        self
    }

    /// Sets the queue capacity per worker.
    pub fn queue_capacity(mut self, capacity: usize) -> Self {
        self.queue_capacity = capacity;
        self
    }

    /// Sets the VM factory producing an [`EmbeddedProgram`] on each worker thread.
    pub fn vm_factory<F>(mut self, factory: F) -> Self
    where
        F: Fn() -> Result<EmbeddedProgram, String> + Send + Sync + 'static,
    {
        self.factory = Some(Box::new(factory));
        self
    }

    /// Builds and starts the worker pool.
    pub fn build(self) -> Result<AsyncWorkerPool, PoolError> {
        if self.workers == 0 {
            return Err(PoolError::Config("workers must be greater than 0".into()));
        }
        if self.queue_capacity == 0 {
            return Err(PoolError::Config(
                "queue_capacity must be greater than 0".into(),
            ));
        }

        let factory = Arc::new(self.factory.unwrap_or_else(|| {
            Box::new(|| {
                EmbeddedBuilder::default()
                    .module("main", "function evaluate(x: int): int\n return x * 2\nend")
                    .entry_module("main")
                    .compile()
                    .map_err(|e| e.to_string())
            })
        }));

        let mut senders = Vec::with_capacity(self.workers);
        let mut workers = Vec::with_capacity(self.workers);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();

        for i in 0..self.workers {
            let (tx, mut rx) = mpsc::channel::<WorkerJob>(self.queue_capacity);
            senders.push(tx);

            let factory_clone = Arc::clone(&factory);
            let thread_ready = ready_tx.clone();

            let join_handle = std::thread::Builder::new()
                .name(format!("lust-worker-{}", i))
                .spawn(move || {
                    let rt = match tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        Ok(rt) => rt,
                        Err(err) => {
                            let _ = thread_ready.send(Err(format!("Tokio rt build failed: {err}")));
                            return;
                        }
                    };

                    let local = tokio::task::LocalSet::new();

                    local.block_on(&rt, async move {
                        let program = match factory_clone() {
                            Ok(p) => p,
                            Err(err) => {
                                let _ = thread_ready.send(Err(format!("VM factory failed: {err}")));
                                return;
                            }
                        };

                        let mut runtime = match LustLocalRuntime::new(program) {
                            Ok(r) => r,
                            Err(err) => {
                                let _ = thread_ready.send(Err(format!("LocalRuntime::new failed: {err}")));
                                return;
                            }
                        };

                        // Signal ready
                        let _ = thread_ready.send(Ok(()));

                        while let Some(job) = rx.recv().await {
                            match job {
                                WorkerJob::Execute {
                                    request,
                                    cancel,
                                    reply,
                                } => {
                                    if reply.is_closed() {
                                        continue;
                                    }

                                    if cancel.is_cancelled() {
                                        let _ = reply.send(Err(PoolError::Cancelled));
                                        continue;
                                    }

                                    // Reset gas budget before execution
                                    if let Ok(prog) = runtime.program_mut() {
                                        prog.reset_gas_counter();
                                        prog.set_gas_budget(100_000);
                                    }

                                    // Check cancellation again right before invocation
                                    if cancel.is_cancelled() {
                                        let _ = reply.send(Err(PoolError::Cancelled));
                                        continue;
                                    }

                                    // Invoke function via call_json or execute
                                    let mut result = match runtime.program_mut() {
                                        Ok(prog) => {
                                            // Format payload as arguments if valid UTF-8
                                            let arg_str = std::str::from_utf8(&request.payload)
                                                .unwrap_or("");
                                            let args = if arg_str.is_empty() {
                                                Vec::new()
                                            } else if let Ok(n) = arg_str.trim().parse::<i64>() {
                                                vec![lust::bytecode::Value::Int(n)]
                                            } else {
                                                vec![lust::bytecode::Value::string(arg_str)]
                                            };
                                            prog.call_raw(&request.function, args)
                                                .map(|val| match val {
                                                    lust::bytecode::Value::Int(n) => {
                                                        WorkerResponse::new(format!("{n}").into_bytes())
                                                    }
                                                    lust::bytecode::Value::String(s) => {
                                                        WorkerResponse::new(s.as_bytes().to_vec())
                                                    }
                                                    other => {
                                                        WorkerResponse::new(format!("{other:?}").into_bytes())
                                                    }
                                                })
                                                .map_err(|e| PoolError::Execution(e.to_string()))
                                        }
                                        Err(err) => Err(PoolError::Execution(err.to_string())),
                                    };

                                    // If synchronous call succeeded and tasks were registered, drive them until complete or cancelled
                                    if result.is_ok() {
                                        let has_pending = runtime.program().map(|p| p.has_pending_async_tasks()).unwrap_or(false);
                                        if has_pending {
                                            match runtime.run_until_all_complete(None, Some(cancel.clone())).await {
                                                Ok(()) => {}
                                                Err(crate::LocalRuntimeError::Cancelled) => {
                                                    result = Err(PoolError::Cancelled);
                                                }
                                                Err(err) => {
                                                    result = Err(PoolError::Execution(err.to_string()));
                                                }
                                            }
                                        }
                                    }

                                    let _ = reply.send(result);
                                }
                                WorkerJob::Shutdown => {
                                    rx.close();
                                    break;
                                }
                            }
                        }

                        let _ = runtime.shutdown();
                    });
                })
                .map_err(|e| PoolError::Config(format!("Thread spawn failed: {e}")))?;

            workers.push(WorkerThread {
                join_handle: Some(join_handle),
            });
        }

        drop(ready_tx);

        // Await all workers ready
        for _ in 0..self.workers {
            match ready_rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => return Err(PoolError::Config(err)),
                Err(_) => return Err(PoolError::Config("Worker startup channel closed".into())),
            }
        }

        let is_shutdown = Arc::new(AtomicBool::new(false));
        let handle = PoolHandle {
            senders: Arc::new(senders),
            next_idx: Arc::new(AtomicUsize::new(0)),
            is_shutdown: Arc::clone(&is_shutdown),
        };

        Ok(AsyncWorkerPool {
            workers,
            handle,
            is_shutdown,
        })
    }
}

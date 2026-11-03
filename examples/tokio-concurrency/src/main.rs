use lust::{EmbeddedBuilder, EmbeddedProgram};
use std::thread::JoinHandle;
use tokio::sync::{mpsc, oneshot};

type Reply = Result<i64, String>;

enum Job {
    Evaluate {
        input: i64,
        reply: oneshot::Sender<Reply>,
    },
    Close,
}

#[derive(Clone)]
struct Client {
    tx: mpsc::Sender<Job>,
}

struct Worker {
    tx: mpsc::Sender<Job>,
    join: JoinHandle<()>,
}

async fn join_worker(join: JoinHandle<()>) -> Result<(), String> {
    tokio::task::spawn_blocking(move || join.join())
        .await
        .map_err(|error| format!("join task failed: {error}"))?
        .map_err(|_| "worker panicked".to_owned())
}

impl Worker {
    async fn start(builder: EmbeddedBuilder, capacity: usize) -> Result<(Self, Client), String> {
        if capacity == 0 {
            return Err("capacity must be positive".into());
        }

        let (tx, mut rx) = mpsc::channel(capacity);
        let (ready_tx, ready_rx) = oneshot::channel();
        let join = std::thread::Builder::new()
            .name("lust-owner".into())
            .spawn(move || {
                // The compiled program never leaves this thread.
                let mut program = match builder.compile() {
                    Ok(program) => {
                        if ready_tx.send(Ok(())).is_err() {
                            return;
                        }
                        program
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(error.to_string()));
                        return;
                    }
                };

                while let Some(job) = rx.blocking_recv() {
                    match job {
                        Job::Evaluate { input, reply } => {
                            if reply.is_closed() {
                                continue;
                            }
                            program.reset_gas_counter();
                            program.set_gas_budget(10_000);
                            let result = program
                                .call_typed("main.evaluate", input)
                                .map_err(|error| error.to_string());
                            let _ = reply.send(result);
                        }
                        Job::Close => rx.close(),
                    }
                }
            })
            .map_err(|error| format!("thread creation failed: {error}"))?;

        match ready_rx.await {
            Ok(Ok(())) => Ok((
                Self {
                    tx: tx.clone(),
                    join,
                },
                Client { tx },
            )),
            startup => {
                drop(tx);
                let joined = join_worker(join).await;
                let reason = match startup {
                    Ok(Err(error)) => format!("initialization failed: {error}"),
                    Err(_) => "worker exited before ready".to_owned(),
                    Ok(Ok(())) => unreachable!(),
                };
                Err(match joined {
                    Ok(()) => reason,
                    Err(error) => format!("{reason}; {error}"),
                })
            }
        }
    }

    async fn shutdown(self) -> Result<(), String> {
        let Self { tx, join } = self;
        let closed = tx.send(Job::Close).await.is_err();
        drop(tx);
        join_worker(join).await?;
        if closed {
            Err("worker closed before shutdown request".into())
        } else {
            Ok(())
        }
    }
}

impl Client {
    async fn evaluate(&self, input: i64) -> Reply {
        let (reply, result) = oneshot::channel();
        self.tx
            .send(Job::Evaluate { input, reply })
            .await
            .map_err(|_| "worker closed".to_owned())?;
        result
            .await
            .map_err(|_| "worker exited without reply".to_owned())?
    }
}

fn builder() -> EmbeddedBuilder {
    EmbeddedProgram::builder()
        .module("main", "function evaluate(x: int): int return x + 1 end")
        .entry_module("main")
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), String> {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<EmbeddedBuilder>();
    assert_send_sync::<Client>();

    if Worker::start(EmbeddedProgram::builder(), 8).await.is_ok() {
        return Err("an empty builder unexpectedly compiled".into());
    }
    if Worker::start(builder(), 0).await.is_ok() {
        return Err("zero queue capacity unexpectedly succeeded".into());
    }

    let (worker, client) = Worker::start(builder(), 8).await?;
    let mut tasks = Vec::new();
    for input in 0..1_000_i64 {
        let client = client.clone();
        tasks.push(tokio::spawn(async move {
            let result = client.evaluate(input).await?;
            if result != input + 1 {
                return Err(format!(
                    "input {input}: expected {}, got {result}",
                    input + 1
                ));
            }
            Ok::<(), String>(())
        }));
    }
    for task in tasks {
        task.await.map_err(|error| error.to_string())??;
    }

    worker.shutdown().await?;
    if client.evaluate(0).await.is_ok() {
        return Err("a request unexpectedly succeeded after shutdown".into());
    }
    println!("PASS: 1000 requests, owner-thread VM, startup errors, and shutdown");
    Ok(())
}

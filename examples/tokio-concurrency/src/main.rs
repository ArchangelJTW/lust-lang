use lust::{EmbeddedBuilder, EmbeddedProgram};
use std::thread::JoinHandle;
use tokio::sync::{mpsc, oneshot};

type Reply = Result<i64, String>;

enum Job {
    Evaluate {
        input: i64,
        reply: oneshot::Sender<Reply>,
    },
    #[cfg(test)]
    GatedEvaluate {
        input: i64,
        reply: oneshot::Sender<Reply>,
        gate_tx: std::sync::mpsc::Sender<()>,
        release_rx: std::sync::mpsc::Receiver<()>,
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
                        #[cfg(test)]
                        Job::GatedEvaluate {
                            input,
                            reply,
                            gate_tx,
                            release_rx,
                        } => {
                            if reply.is_closed() {
                                continue;
                            }
                            let _ = gate_tx.send(());
                            let _ = release_rx.recv();
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

    #[cfg(test)]
    async fn evaluate_gated(
        &self,
        input: i64,
        gate_tx: std::sync::mpsc::Sender<()>,
        release_rx: std::sync::mpsc::Receiver<()>,
    ) -> Reply {
        let (reply, result) = oneshot::channel();
        self.tx
            .send(Job::GatedEvaluate {
                input,
                reply,
                gate_tx,
                release_rx,
            })
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

#[cfg(feature = "fail-move")]
fn invalid_spawn(program: EmbeddedProgram) {
    std::thread::spawn(move || drop(program));
}

#[cfg(feature = "fail-sync")]
fn invalid_sync() {
    fn check<T: Sync>() {}
    check::<EmbeddedProgram>();
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
    println!(
        "PASS: builder Send+Sync, builder move, startup error, zero capacity, 1000 callers, shutdown, closed send"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_moves_program_stays() -> Result<(), String> {
        let moved_builder = builder();
        let value = std::thread::spawn(move || -> Reply {
            let mut program = moved_builder.compile().map_err(|e| e.to_string())?;
            program
                .call_typed("main.evaluate", 41_i64)
                .map_err(|e| e.to_string())
        })
        .join()
        .map_err(|_| "worker panicked".to_owned())??;
        assert_eq!(value, 42);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drain_and_abandoned_reply() -> Result<(), String> {
        let (worker, client) = Worker::start(builder(), 4).await?;
        let (abandoned_tx, abandoned_rx) = oneshot::channel();
        drop(abandoned_rx);
        client
            .tx
            .send(Job::Evaluate {
                input: 10,
                reply: abandoned_tx,
            })
            .await
            .map_err(|_| "closed".to_owned())?;
        let (reply, result) = oneshot::channel();
        client
            .tx
            .send(Job::Evaluate { input: 41, reply })
            .await
            .map_err(|_| "closed".to_owned())?;
        worker.shutdown().await?;
        assert_eq!(result.await.map_err(|e| e.to_string())??, 42);
        assert!(client.evaluate(0).await.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn evaluation_error() -> Result<(), String> {
        let wrong = EmbeddedProgram::builder()
            .module("main", "function other(): int return 1 end")
            .entry_module("main");
        let (worker, client) = Worker::start(wrong, 1).await?;
        assert!(
            client
                .evaluate(0)
                .await
                .unwrap_err()
                .contains("No type information")
        );
        worker.shutdown().await
    }

    #[tokio::test]
    async fn lost_reply_and_panic_join() {
        let (tx, mut rx) = mpsc::channel(1);
        let client = Client { tx };
        let call = tokio::spawn(async move { client.evaluate(0).await });
        match rx.recv().await.unwrap() {
            Job::Evaluate { reply, .. } => drop(reply),
            _ => panic!("unexpected job"),
        }
        assert_eq!(
            call.await.unwrap().unwrap_err(),
            "worker exited without reply"
        );
        assert_eq!(
            join_worker(std::thread::spawn(|| panic!("test panic")))
                .await
                .unwrap_err(),
            "worker panicked"
        );
    }

    #[tokio::test]
    async fn startup_zero_capacity_and_compile_error() {
        match Worker::start(builder(), 0).await {
            Err(e) => assert_eq!(e, "capacity must be positive"),
            Ok(_) => panic!("expected zero capacity startup to fail"),
        }
        match Worker::start(EmbeddedProgram::builder(), 1).await {
            Err(e) => assert!(e.contains("initialization failed")),
            Ok(_) => panic!("expected empty builder startup to fail"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deterministic_capacity_one_backpressure_and_cancellation() -> Result<(), String> {
        let (worker, client) = Worker::start(builder(), 1).await?;

        // 1. Gate owner call using test-only host synchronization
        let (gate_tx, gate_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let client_gated = client.clone();
        let call1 =
            tokio::spawn(async move { client_gated.evaluate_gated(10, gate_tx, release_rx).await });

        // Deterministically wait until the owner thread has entered call 1
        gate_rx.recv().map_err(|e| e.to_string())?;

        // 2. Fill the single queue slot (capacity = 1)
        let (reply2_tx, reply2_rx) = oneshot::channel();
        client
            .tx
            .try_send(Job::Evaluate {
                input: 20,
                reply: reply2_tx,
            })
            .map_err(|e| format!("failed to fill queue slot: {e}"))?;

        // Queue is now at capacity 1 while call 1 is executing on owner thread.
        // 3. Prove a further send is pending without timing-only sleeps.
        let (reply3_tx, reply3_rx) = oneshot::channel();
        let mut send3 = std::pin::pin!(client.tx.send(Job::Evaluate {
            input: 30,
            reply: reply3_tx,
        }));

        struct Noop;
        impl std::task::Wake for Noop {
            fn wake(self: std::sync::Arc<Self>) {}
        }
        let waker = std::task::Waker::from(std::sync::Arc::new(Noop));
        let mut cx = std::task::Context::from_waker(&waker);

        // Polling send3 must return Pending because the queue slot is filled.
        assert!(
            send3.as_mut().poll(&mut cx).is_pending(),
            "send into full channel must be pending"
        );

        // 4. Cancel another pending send before acceptance
        {
            let (reply_cancel_tx, mut reply_cancel_rx) = oneshot::channel();
            let mut send_cancel = std::pin::pin!(client.tx.send(Job::Evaluate {
                input: 99,
                reply: reply_cancel_tx,
            }));
            assert!(
                send_cancel.as_mut().poll(&mut cx).is_pending(),
                "send into full channel must be pending before cancellation"
            );
            // Dropping send_cancel cancels the send before acceptance into the queue.
            drop(send_cancel);
            // Verify reply channel was dropped and never completed.
            assert!(
                reply_cancel_rx.try_recv().is_err(),
                "cancelled send must not have delivered a reply"
            );
        }

        // 5. Release gate and assert admission and results
        release_tx.send(()).map_err(|e| e.to_string())?;

        // Call 1 completes
        assert_eq!(call1.await.map_err(|e| e.to_string())??, 11);

        // Call 2 completes
        assert_eq!(reply2_rx.await.map_err(|e| e.to_string())??, 21);

        // Call 3 (previously pending) can now be awaited and admitted
        send3.await.map_err(|_| "worker closed".to_owned())?;
        assert_eq!(reply3_rx.await.map_err(|e| e.to_string())??, 31);

        // Clean shutdown
        worker.shutdown().await?;
        assert_eq!(client.evaluate(0).await.unwrap_err(), "worker closed");

        Ok(())
    }
}

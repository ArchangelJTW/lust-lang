# Embedding Lust in a multi-threaded Tokio host

`EmbeddedProgram` is neither `Send` nor `Sync`. Construct, compile, use, and
drop each program on one owning OS thread. A `Mutex` cannot make the compiled
program transferable. This example moves an `EmbeddedBuilder` into a dedicated
worker thread and compiles the program there; Tokio tasks exchange only owned
integer requests and results through a bounded channel.

## Lifecycle and startup behavior

The example demonstrates deterministic actor lifecycle and error handling:

- **Startup validation:** `Worker::start` validates parameters and rejects zero
  capacity (`capacity == 0`) before channel creation.
- **Pre-publication compilation:** The dedicated OS worker thread compiles the
  `EmbeddedBuilder` before announcing readiness or publishing any `Client`.
- **Startup failure cleanup:** The demonstrated startup failure is a builder
  compilation failure; the worker reports it through the startup channel and
  attempts to join the spawned OS thread before returning `Err`. Cleanup is
  established only if that join completes successfully; join-task failure or
  cancellation does not establish that the worker thread joined.
- **Graceful shutdown and drain:** A `shutdown` request enqueues a close sentinel,
  closes the receiver against new sends, drains accepted jobs, and joins the
  worker thread.

## Concurrency, backpressure, and cancellation

- **Backpressure:** The channel capacity bounds buffered requests waiting for the
  worker. When the buffer is full, further caller sends remain pending until the
  worker admits the next job.
- **Cancellation before acceptance:** Dropping or cancelling a pending send before
  channel acceptance cancels the request cleanly without executing work on the
  worker.
- **Limitations:** Dropping a request or reply receiver *after* acceptance or while
  executing does not interrupt a call already running in the VM; a caller retry
  may repeat side effects. The actor handles synchronous script evaluation and
  does not drive Lust's cooperative async task registry. For hard timeouts,
  resource containment, or forced termination, use a supervised worker process.

## Commands and verification

Run the positive checks from the repository root:

```sh
cargo run --offline --locked --manifest-path examples/tokio-concurrency/Cargo.toml
cargo test --offline --locked --manifest-path examples/tokio-concurrency/Cargo.toml
```

Negative auto-trait probes verify that Rust's type system prevents moving or
sharing `EmbeddedProgram` across threads. Each probe deliberately fails with
compiler error `E0277` at the program boundary:

```sh
# Proves moving EmbeddedProgram into std::thread::spawn fails E0277
cargo check --offline --locked --manifest-path examples/tokio-concurrency/Cargo.toml --features fail-move

# Proves EmbeddedProgram does not implement Sync fails E0277
cargo check --offline --locked --manifest-path examples/tokio-concurrency/Cargo.toml --features fail-sync
```

Because these negative features deliberately fail compilation, do not run
`--all-features` as a positive check.

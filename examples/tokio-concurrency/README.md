# Embedding Lust in a multi-threaded Tokio host

`EmbeddedProgram` is neither `Send` nor `Sync`. Construct, compile, use, and
drop each program on one owning OS thread. A `Mutex` cannot make the compiled
program transferable. This example moves an `EmbeddedBuilder` into a dedicated
worker thread and compiles the program there; Tokio tasks exchange only owned
integer requests and results through a bounded channel.

Run the example from the repository root:

```sh
cargo run --locked --manifest-path examples/tokio-concurrency/Cargo.toml
```

The example has its own Cargo lockfile. Tokio is used only by this example and
is not added to the Lust runtime dependencies.

The queue bounds buffered jobs, not the number of tasks waiting to enqueue, VM
memory, host callbacks, or total service memory. Synchronous script execution
blocks the dedicated worker. Dropping a request/reply does not interrupt a call
already running in the VM; a caller retry may repeat side effects. The example
does not implement Lust's async task driver and does not provide a hard timeout
or process isolation. Use a supervised worker process when script execution
must be forcibly terminated.

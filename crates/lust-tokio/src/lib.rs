//! `lust-tokio`: Tokio async runtime bridge and worker pool for the Lust scripting language.
//!
//! Provides:
//! - [`LustLocalRuntime`]: Single-threaded `tokio::task::LocalSet` bridge (`E06`) with event-driven
//!   wakeups (zero polling spin) and strict owner-thread confinement for `!Send + !Sync` programs.
//! - [`AsyncWorkerPool`]: Multi-threaded pool of dedicated owner OS threads, exposing a thread-safe
//!   (`Send + Sync + Clone`) [`PoolHandle`] with bounded queues and clean shutdown/join.
//! - [`TokioWakeNotifier`]: Bridge between Lust's [`lust::embed::async_runtime::AsyncWakeNotifier`]
//!   and Tokio's [`tokio::sync::Notify`].
//! - [`CancellationBridge`]: Bidirectional cancellation coordinator linking Tokio [`CancellationToken`]
//!   and Lust fiber / scope lifecycles.

pub mod cancellation;
pub mod local;
pub mod notifier;
pub mod pool;

pub use cancellation::CancellationBridge;
pub use local::{LocalRuntimeError, LustLocalRuntime, RunOptions, RunOutcome};
pub use notifier::TokioWakeNotifier;
pub use pool::{AsyncWorkerPool, PoolBuilder, PoolError, PoolHandle, WorkerRequest, WorkerResponse};
pub use tokio_util::sync::CancellationToken;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assert_send_and_sync_bounds() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<PoolHandle>();
        assert_sync::<PoolHandle>();
        assert_send::<WorkerRequest>();
        assert_send::<WorkerResponse>();
        assert_send::<CancellationBridge>();
        assert_sync::<CancellationBridge>();
    }
}

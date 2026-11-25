use lust::embed::async_runtime::AsyncCancelObserver;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Bi-directional cancellation coordinator linking a Tokio [`CancellationToken`]
/// with Lust task / scope cancellation.
#[derive(Clone)]
pub struct CancellationBridge {
    token: CancellationToken,
    cancelled: Arc<AtomicBool>,
}

impl CancellationBridge {
    /// Creates a new bridge around the given Tokio cancellation token.
    pub fn new(token: CancellationToken) -> Self {
        Self {
            token,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Checks if cancellation has been requested from either Tokio or Lust.
    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled() || self.cancelled.load(Ordering::SeqCst)
    }

    /// Triggers cancellation, notifying both the Tokio token and setting the internal flag.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        self.token.cancel();
    }

    /// Returns the underlying Tokio [`CancellationToken`].
    pub fn token(&self) -> &CancellationToken {
        &self.token
    }

    /// Creates an [`AsyncCancelObserver`] that triggers this bridge when Lust requests cancellation.
    pub fn as_cancel_observer(&self) -> Arc<dyn AsyncCancelObserver> {
        let bridge = self.clone();
        Arc::new(move || {
            bridge.cancel();
        })
    }
}

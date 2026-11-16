use lust::embed::async_runtime::AsyncWakeNotifier;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::Notify;

/// An implementation of Lust's [`AsyncWakeNotifier`] that signals a Tokio [`Notify`].
///
/// When an async native task finishes or makes progress in Lust, the waker calls `notify()`,
/// which in turn calls `notify.notify_one()`, waking any task waiting on `notify.notified()`.
#[derive(Clone)]
pub struct TokioWakeNotifier {
    notify: Arc<Notify>,
    woken: Arc<AtomicBool>,
}

impl TokioWakeNotifier {
    /// Creates a new notifier wrapping the given [`Notify`].
    pub fn new(notify: Arc<Notify>) -> Self {
        Self {
            notify,
            woken: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn with_woken(notify: Arc<Notify>, woken: Arc<AtomicBool>) -> Self {
        Self { notify, woken }
    }

    /// Returns a reference to the inner [`Notify`].
    pub fn notify_handle(&self) -> &Arc<Notify> {
        &self.notify
    }

    pub fn woken_handle(&self) -> &Arc<AtomicBool> {
        &self.woken
    }
}

impl AsyncWakeNotifier for TokioWakeNotifier {
    fn notify(&self) {
        self.woken.store(true, Ordering::SeqCst);
        self.notify.notify_one();
    }
}

use crate::notifier::TokioWakeNotifier;
use lust::embed::async_runtime::{AsyncWakeNotifier, WakeNotifierScope};
use lust::embed::EmbeddedProgram;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use std::thread::ThreadId;
use std::time::Duration;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

/// Outcome of a `run_until_idle` execution step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunOutcome {
    /// Registry is quiescent with the specified number of pending external tasks remaining.
    Quiescent(usize),
    /// Execution was cancelled via [`CancellationToken`].
    Cancelled,
    /// Execution exceeded deadline/timeout.
    TimedOut,
}

/// Options configuring a `run_until_idle` execution.
#[derive(Default, Clone)]
pub struct RunOptions {
    pub cancellation: Option<CancellationToken>,
    pub timeout: Option<Duration>,
}

/// Errors occurring during local runtime execution.
#[derive(Debug)]
pub enum LocalRuntimeError {
    ThreadAffinityViolation,
    Cancelled,
    TimedOut,
    Runtime(String),
    ShutDown,
}

impl std::fmt::Display for LocalRuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ThreadAffinityViolation => {
                write!(f, "Thread affinity violation: LustLocalRuntime accessed from non-owner thread")
            }
            Self::Cancelled => write!(f, "Operation was cancelled"),
            Self::TimedOut => write!(f, "Operation timed out"),
            Self::Runtime(msg) => write!(f, "Runtime error: {msg}"),
            Self::ShutDown => write!(f, "Runtime has already shut down"),
        }
    }
}

impl std::error::Error for LocalRuntimeError {}

/// Single-threaded Tokio `LocalSet` bridge for [`EmbeddedProgram`] (`!Send + !Sync`).
///
/// Ensures strict owner-thread confinement, zero extra OS threads, event-driven wakeups
/// without polling spin, and clean scope teardown.
pub struct LustLocalRuntime {
    program: EmbeddedProgram,
    notify: Arc<Notify>,
    woken: Arc<std::sync::atomic::AtomicBool>,
    owner_thread: ThreadId,
    scope_guard: Option<WakeNotifierScope>,
    is_shutdown: bool,
    _not_send: PhantomData<Rc<()>>,
}

impl LustLocalRuntime {
    /// Creates a new local runtime around an [`EmbeddedProgram`].
    ///
    /// Must be created on the current thread where it will be driven.
    pub fn new(mut program: EmbeddedProgram) -> Result<Self, LocalRuntimeError> {
        let current_thread = std::thread::current().id();
        let notify = Arc::new(Notify::new());
        let woken = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let notifier = Arc::new(TokioWakeNotifier::with_woken(Arc::clone(&notify), Arc::clone(&woken)));
        let scope_guard = WakeNotifierScope::install(Arc::clone(&notifier) as Arc<dyn AsyncWakeNotifier>);
        program.set_wake_notifier(Some(Arc::clone(&notifier) as Arc<dyn AsyncWakeNotifier>));

        Ok(Self {
            program,
            notify,
            woken,
            owner_thread: current_thread,
            scope_guard: Some(scope_guard),
            is_shutdown: false,
            _not_send: PhantomData,
        })
    }

    /// Returns the number of currently active pending async tasks in this runtime.
    pub fn active_task_count(&self) -> usize {
        self.program.pending_async_task_count()
    }

    fn check_owner(&self) -> Result<(), LocalRuntimeError> {
        if std::thread::current().id() != self.owner_thread {
            return Err(LocalRuntimeError::ThreadAffinityViolation);
        }
        if self.is_shutdown {
            return Err(LocalRuntimeError::ShutDown);
        }
        Ok(())
    }

    /// Returns the owner thread ID of this runtime.
    pub fn owner_thread_id(&self) -> ThreadId {
        self.owner_thread
    }

    /// Returns a reference to the inner [`EmbeddedProgram`].
    pub fn program(&self) -> Result<&EmbeddedProgram, LocalRuntimeError> {
        self.check_owner()?;
        Ok(&self.program)
    }

    /// Returns a mutable reference to the inner [`EmbeddedProgram`].
    pub fn program_mut(&mut self) -> Result<&mut EmbeddedProgram, LocalRuntimeError> {
        self.check_owner()?;
        self.woken.store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(&mut self.program)
    }

    /// Drives currently runnable tasks until quiescence without polling spin.
    ///
    /// If there are pending external tasks waiting on futures, this returns
    /// `RunOutcome::Quiescent(pending_count)` without busy-spinning.
    pub async fn run_until_idle(&mut self, options: RunOptions) -> Result<RunOutcome, LocalRuntimeError> {
        self.check_owner()?;

        if let Some(cancel) = &options.cancellation {
            if cancel.is_cancelled() {
                self.program.cancel_pending_async_tasks();
                return Ok(RunOutcome::Cancelled);
            }
        }

        // Only poll runnable tasks if there has been a wake notification or on initial dispatch
        if self.woken.swap(false, std::sync::atomic::Ordering::SeqCst) {
            self.program
                .poll_async_tasks()
                .map_err(|e| LocalRuntimeError::Runtime(e.to_string()))?;
        }

        // Check if cancellation occurred during poll
        if let Some(cancel) = &options.cancellation {
            if cancel.is_cancelled() {
                self.program.cancel_pending_async_tasks();
                return Ok(RunOutcome::Cancelled);
            }
        }

        let pending = lust::embed::async_runtime::current_thread_pending_task_count();
        if !self.program.has_pending_async_tasks() || pending == 0 {
            Ok(RunOutcome::Quiescent(0))
        } else {
            Ok(RunOutcome::Quiescent(pending))
        }
    }

    /// Drives all tasks until all pending native futures complete, or timeout/cancellation occurs.
    pub async fn run_until_all_complete(
        &mut self,
        timeout: Option<Duration>,
        cancel: Option<CancellationToken>,
    ) -> Result<(), LocalRuntimeError> {
        self.check_owner()?;

        let start = std::time::Instant::now();

        loop {
            // Check cancellation first
            if let Some(c) = &cancel {
                if c.is_cancelled() {
                    self.program.cancel_pending_async_tasks();
                    return Err(LocalRuntimeError::Cancelled);
                }
            }

            // Check timeout
            if let Some(t) = timeout {
                if start.elapsed() >= t {
                    self.program.cancel_pending_async_tasks();
                    return Err(LocalRuntimeError::TimedOut);
                }
            }

            // Arm the notification future BEFORE polling to prevent lost wakeups
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            // Poll runnable tasks
            self.program
                .poll_async_tasks()
                .map_err(|e| LocalRuntimeError::Runtime(e.to_string()))?;

            if !self.program.has_pending_async_tasks() {
                return Ok(());
            }

            // Compute remaining timeout duration if specified
            let remaining_timeout = timeout.map(|t| {
                let elapsed = start.elapsed();
                if elapsed >= t {
                    Duration::from_millis(0)
                } else {
                    t - elapsed
                }
            });

            // Wait for external wake, cancellation, or deadline
            tokio::select! {
                _ = &mut notified => {
                    // Woken up by task completion or progress
                }
                _ = async {
                    if let Some(c) = &cancel {
                        c.cancelled().await
                    } else {
                        std::future::pending().await
                    }
                } => {
                    self.program.cancel_pending_async_tasks();
                    return Err(LocalRuntimeError::Cancelled);
                }
                _ = async {
                    if let Some(rem) = remaining_timeout {
                        tokio::time::sleep(rem).await
                    } else {
                        std::future::pending().await
                    }
                } => {
                    self.program.cancel_pending_async_tasks();
                    return Err(LocalRuntimeError::TimedOut);
                }
            }
        }
    }

    /// Gracefully shuts down the local runtime, cancelling pending tasks and popping its own notifier scope.
    pub fn shutdown(&mut self) -> Result<(), LocalRuntimeError> {
        self.check_owner()?;
        self.is_shutdown = true;
        self.program.cancel_pending_async_tasks();
        self.program.set_wake_notifier(None);
        self.scope_guard.take();
        Ok(())
    }
}

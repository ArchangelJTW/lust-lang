//! Isolated child process launcher and monotonic watchdog supervisor.

use crate::sandbox::darwin::{RlimitReport, configure_posix_pre_exec, kill_process_group};
use crate::sandbox::{SandboxError, SandboxPolicy};
use std::io::{Read, Write};
use std::os::unix::process::ExitStatusExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

const OUTPUT_DRAIN_GRACE: Duration = Duration::from_millis(200);

/// Outcome of a supervised sandboxed execution.
#[derive(Debug, Clone)]
pub struct SandboxOutcome {
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
    pub timed_out: bool,
    pub cancelled: bool,
    pub elapsed: Duration,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    pub output_drain_timed_out: bool,
    pub rlimits: RlimitReport,
    pub temp_dir_path: PathBuf,
    pub temp_dir_cleaned: bool,
    pub platform_mode: &'static str,
}

#[derive(Default, Clone)]
struct BoundedCapture {
    bytes: Vec<u8>,
    truncated: bool,
}

struct BoundedReaderHandle {
    capture: Arc<Mutex<BoundedCapture>>,
    done_rx: mpsc::Receiver<()>,
    join_handle: thread::JoinHandle<()>,
}

fn spawn_bounded_reader<R: Read + Send + 'static>(
    mut reader: R,
    max_bytes: usize,
) -> BoundedReaderHandle {
    let capture = Arc::new(Mutex::new(BoundedCapture::default()));
    let capture_writer = Arc::clone(&capture);
    let (done_tx, done_rx) = mpsc::channel();

    let join_handle = thread::spawn(move || {
        let mut buf = [0u8; 4096];
        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    let mut guard = capture_writer
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if guard.bytes.len() < max_bytes {
                        let remaining = max_bytes - guard.bytes.len();
                        let take = n.min(remaining);
                        guard.bytes.extend_from_slice(&buf[..take]);
                        if n > remaining {
                            guard.truncated = true;
                        }
                    } else {
                        guard.truncated = true;
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
        let _ = done_tx.send(());
    });

    BoundedReaderHandle {
        capture,
        done_rx,
        join_handle,
    }
}

fn collect_bounded_reader(
    handle: Option<BoundedReaderHandle>,
    deadline: Instant,
) -> (BoundedCapture, bool) {
    let Some(BoundedReaderHandle {
        capture,
        done_rx,
        join_handle,
    }) = handle
    else {
        return (BoundedCapture::default(), false);
    };

    let remaining = deadline.saturating_duration_since(Instant::now());
    match done_rx.recv_timeout(remaining) {
        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => {
            let _ = join_handle.join();
            let guard = capture
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (guard.clone(), false)
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            // Detach reader thread so a setsid()-escaped descendant holding
            // stdout/stderr open cannot hang the supervisor indefinitely.
            drop(join_handle);
            let guard = capture
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (guard.clone(), true)
        }
    }
}

fn platform_mode_label() -> &'static str {
    if cfg!(target_os = "macos") {
        "darwin-posix-baseline"
    } else {
        "unix-posix-baseline"
    }
}

/// Executes a child process under the supplied [`SandboxPolicy`].
#[cfg(unix)]
pub fn run_sandboxed(policy: &SandboxPolicy) -> Result<SandboxOutcome, SandboxError> {
    policy.validate()?;

    let temp_dir = tempfile::Builder::new()
        .prefix("lust_sandbox_")
        .tempdir()
        .map_err(SandboxError::TempDir)?;
    let temp_dir_path = temp_dir.path().to_path_buf();

    let mut cmd = Command::new(&policy.executable);
    cmd.args(&policy.args)
        .env_clear()
        .envs(&policy.allowed_env)
        .current_dir(&temp_dir_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if policy.stdin.is_some() {
        cmd.stdin(Stdio::piped());
    } else {
        cmd.stdin(Stdio::null());
    }

    let mut prepared = configure_posix_pre_exec(&mut cmd, &policy.limits)
        .map_err(SandboxError::Spawn)?;

    let total_start = Instant::now();
    let spawn_res = cmd.spawn();
    prepared.finish_parent_spawn();

    let mut child = spawn_res.map_err(SandboxError::Spawn)?;
    let rlimits = prepared.read_rlimit_report();

    let child_pid = child.id() as i32;

    if let Some(stdin_bytes) = policy.stdin.clone()
        && let Some(mut stdin_pipe) = child.stdin.take()
    {
        thread::spawn(move || {
            let _ = stdin_pipe.write_all(&stdin_bytes);
            let _ = stdin_pipe.flush();
        });
    }

    let stdout_handle = child
        .stdout
        .take()
        .map(|out| spawn_bounded_reader(out, policy.limits.max_stdout_bytes));
    let stderr_handle = child
        .stderr
        .take()
        .map(|err| spawn_bounded_reader(err, policy.limits.max_stderr_bytes));

    let watchdog_start = Instant::now();
    let mut timed_out = false;
    let mut cancelled = false;

    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                let is_cancelled = policy
                    .cancel_flag
                    .as_ref()
                    .is_some_and(|f| f.load(Ordering::Acquire));
                let is_timed_out = watchdog_start.elapsed() >= policy.limits.wall_timeout;

                if is_cancelled || is_timed_out {
                    timed_out = is_timed_out && !is_cancelled;
                    cancelled = is_cancelled;
                    // Ignore ESRCH if the child raced to exit right before killpg.
                    let _ = kill_process_group(child_pid);
                    break child
                        .wait()
                        .map_err(|e| SandboxError::Supervisor(format!("wait failed: {e}")))?;
                }
                thread::sleep(Duration::from_millis(5));
            }
            Err(e) => {
                let _ = kill_process_group(child_pid);
                let _ = child.wait();
                return Err(SandboxError::Supervisor(format!("try_wait failed: {e}")));
            }
        }
    };

    let drain_deadline = Instant::now() + OUTPUT_DRAIN_GRACE;
    let (stdout_cap, stdout_drain_timed_out) =
        collect_bounded_reader(stdout_handle, drain_deadline);
    let (stderr_cap, stderr_drain_timed_out) =
        collect_bounded_reader(stderr_handle, drain_deadline);
    let output_drain_timed_out = stdout_drain_timed_out || stderr_drain_timed_out;

    let elapsed = total_start.elapsed();

    let close_ok = temp_dir.close().is_ok();
    let temp_dir_cleaned = close_ok && !temp_dir_path.exists();

    Ok(SandboxOutcome {
        exit_code: status.code(),
        signal: status.signal(),
        timed_out,
        cancelled,
        elapsed,
        stdout: stdout_cap.bytes,
        stderr: stderr_cap.bytes,
        stdout_truncated: stdout_cap.truncated,
        stderr_truncated: stderr_cap.truncated,
        output_drain_timed_out,
        rlimits,
        temp_dir_path,
        temp_dir_cleaned,
        platform_mode: platform_mode_label(),
    })
}

/// Non-Unix stub failing closed.
#[cfg(not(unix))]
pub fn run_sandboxed(policy: &SandboxPolicy) -> Result<SandboxOutcome, SandboxError> {
    policy.validate()?;
    Err(SandboxError::UnsupportedIsolation(
        "POSIX process-group sandbox requires Unix".into(),
    ))
}

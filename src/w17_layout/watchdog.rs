//! Isolated child-process runner and external POSIX watchdog (`TASK_ALLOWS_UNSAFE=0`).
//!
//! Every C fixture, callback probe, and unbounded worker runs in its own process group
//! (`CommandExt::process_group(0)`). An external safety watchdog enforces hard wall-clock
//! deadlines and kills/reaps the entire child process group via `/bin/kill` without any
//! authored `unsafe` code.
//!
//! Gate-SCHEMA-13 Rule:
//! - Expected worker containment is verified by explicit `terminate_and_reap_worker`
//!   acknowledgement + reap before the outer safety watchdog triggers.
//! - Termination by the external safety watchdog is ALWAYS classified as `WatchdogKilledFail`.

#![forbid(unsafe_code)]

use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

/// Classification of an isolated child execution under an external watchdog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WatchdogOutcome {
    /// Child terminated cleanly on its own (or via cooperative cancellation) before deadline.
    Completed {
        exit_code: Option<i32>,
        signal: Option<i32>,
    },
    /// Child exceeded the external safety watchdog timeout and was forcibly killed & reaped.
    /// Per Gate-SCHEMA-09 / Gate-SCHEMA-13, an outer watchdog kill is ALWAYS a FAIL.
    WatchdogKilledFail {
        pid: u32,
        elapsed_ms: u128,
        reaped: bool,
    },
}

impl WatchdogOutcome {
    pub fn is_watchdog_kill(&self) -> bool {
        matches!(self, WatchdogOutcome::WatchdogKilledFail { .. })
    }

    pub fn is_clean_success(&self) -> bool {
        matches!(
            self,
            WatchdogOutcome::Completed {
                exit_code: Some(0),
                ..
            }
        )
    }
}

/// Full captured output and evidence for a single isolated child execution.
#[derive(Debug, Clone)]
pub struct WatchdogRunResult {
    pub command_line: String,
    pub outcome: WatchdogOutcome,
    pub stdout: String,
    pub stderr: String,
    pub elapsed_ms: u128,
}

fn send_posix_signal(target: &str, signal: &str) {
    let _ = Command::new("/bin/kill")
        .arg(format!("-{signal}"))
        .arg("--")
        .arg(target)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

#[cfg(unix)]
fn status_signal(status: &ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    status.signal()
}

#[cfg(not(unix))]
fn status_signal(_status: &ExitStatus) -> Option<i32> {
    None
}

/// Spawns `program` with `args` and `envs` inside an isolated process group, monitored by an
/// external safety watchdog for `timeout`.
pub fn run_child_with_watchdog(
    program: &Path,
    args: &[&str],
    envs: &[(&str, &str)],
    timeout: Duration,
) -> Result<WatchdogRunResult, String> {
    let tmp = tempfile::tempdir().map_err(|e| format!("watchdog tempdir: {e}"))?;
    let stdout_path = tmp.path().join("stdout.log");
    let stderr_path = tmp.path().join("stderr.log");

    let stdout_file = File::create(&stdout_path).map_err(|e| format!("create stdout: {e}"))?;
    let stderr_file = File::create(&stderr_path).map_err(|e| format!("create stderr: {e}"))?;

    let mut cmd = Command::new(program);
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file));
    for (k, v) in envs {
        cmd.env(k, v);
    }
    #[cfg(unix)]
    {
        cmd.process_group(0);
    }

    let command_line = format!("{} {}", program.display(), args.join(" "));
    let start = Instant::now();
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("spawn `{command_line}`: {e}"))?;
    let pid = child.id();

    let mut finished: Option<ExitStatus> = None;
    while start.elapsed() < timeout {
        match child.try_wait() {
            Ok(Some(st)) => {
                finished = Some(st);
                break;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("try_wait failed for PID {pid}: {e}"));
            }
        }
    }

    let elapsed_ms = start.elapsed().as_millis();
    let outcome = if let Some(st) = finished {
        WatchdogOutcome::Completed {
            exit_code: st.code(),
            signal: status_signal(&st),
        }
    } else {
        // Watchdog timeout expired: forcibly terminate entire child process group and reap.
        send_posix_signal(&format!("-{pid}"), "KILL");
        let _ = child.kill();
        let reaped = child.wait().is_ok();
        WatchdogOutcome::WatchdogKilledFail {
            pid,
            elapsed_ms,
            reaped,
        }
    };

    let stdout = fs::read_to_string(&stdout_path).unwrap_or_default();
    let stderr = fs::read_to_string(&stderr_path).unwrap_or_default();

    Ok(WatchdogRunResult {
        command_line,
        outcome,
        stdout,
        stderr,
        elapsed_ms,
    })
}

/// Result of explicit worker containment (`terminate + reap`) under an outer safety watchdog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerTerminationResult {
    pub pid: u32,
    pub readiness_line: String,
    pub terminated_by_manager: bool,
    pub acknowledged: bool,
    pub reaped: bool,
    pub killed_by_outer_watchdog: bool,
    pub termination_signal: Option<i32>,
    pub elapsed_ms: u128,
}

/// Spawns an isolated worker child process, waits for its readiness line, enforces the
/// operation deadline by sending an explicit `SIGTERM` (and `SIGKILL` fallback if needed)
/// to the worker's process group, and waits for `waitpid` reap — all guarded by a separate
/// outer safety watchdog (`outer_watchdog_timeout`).
///
/// If the worker is not reaped before `outer_watchdog_timeout`, `killed_by_outer_watchdog`
/// becomes `true` (which is a Gate-SCHEMA-13 FAIL).
pub fn terminate_and_reap_worker(
    program: &Path,
    args: &[&str],
    envs: &[(&str, &str)],
    operation_deadline: Duration,
    reap_grace: Duration,
    outer_watchdog_timeout: Duration,
) -> Result<WorkerTerminationResult, String> {
    let mut cmd = Command::new(program);
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (k, v) in envs {
        cmd.env(k, v);
    }
    #[cfg(unix)]
    {
        cmd.process_group(0);
    }

    let start = Instant::now();
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("spawn worker `{}`: {e}", program.display()))?;
    let pid = child.id();

    // Wait briefly for readiness line so we prove the worker actually started executing.
    let mut readiness_line = String::new();
    if let Some(stdout) = child.stdout.take() {
        let mut reader = BufReader::new(stdout);
        let _ = reader.read_line(&mut readiness_line);
    }

    // Let worker execute up to its declared operation deadline.
    let deadline_start = Instant::now();
    while deadline_start.elapsed() < operation_deadline {
        if let Ok(Some(st)) = child.try_wait() {
            return Ok(WorkerTerminationResult {
                pid,
                readiness_line: readiness_line.trim().to_string(),
                terminated_by_manager: false,
                acknowledged: true,
                reaped: true,
                killed_by_outer_watchdog: false,
                termination_signal: status_signal(&st),
                elapsed_ms: start.elapsed().as_millis(),
            });
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    // Operation deadline reached: worker manager initiates explicit terminate + reap.
    send_posix_signal(&format!("-{pid}"), "TERM");
    let term_start = Instant::now();
    let mut status: Option<ExitStatus> = None;
    while term_start.elapsed() < reap_grace && start.elapsed() < outer_watchdog_timeout {
        if let Ok(Some(st)) = child.try_wait() {
            status = Some(st);
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let mut killed_by_outer_watchdog = false;
    if status.is_none() {
        if start.elapsed() >= outer_watchdog_timeout {
            killed_by_outer_watchdog = true;
        }
        send_posix_signal(&format!("-{pid}"), "KILL");
        let _ = child.kill();
        if let Ok(st) = child.wait() {
            status = Some(st);
        }
    }

    let reaped = status.is_some();
    let sig = status.as_ref().and_then(status_signal);
    Ok(WorkerTerminationResult {
        pid,
        readiness_line: readiness_line.trim().to_string(),
        terminated_by_manager: !killed_by_outer_watchdog,
        acknowledged: reaped && !killed_by_outer_watchdog,
        reaped,
        killed_by_outer_watchdog,
        termination_signal: sig,
        elapsed_ms: start.elapsed().as_millis(),
    })
}

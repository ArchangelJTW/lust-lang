//! Autonomous Failure-to-DST Pipeline, ShadowVfsLog and Reproduction Task generation.
//! Implements Track 2H (DST-CAP) under RFC-DST-CAPTURE-MASKING-01 and WAVE2_EXECUTION_PLAN.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use serde::{Deserialize, Serialize};

use crate::sim::replay::{EventLog, ReplayBundle, ReplayError, ReplayManifest, UstarArchive};

/// Thread-safe and panic-safe shadow VFS log.
/// Mirrors VFS writes so that in the event of a panic or runtime crash,
/// the virtual filesystem state can be retrieved even if `RefCell` or `Mutex`
/// locks are poisoned or inaccessible.
#[derive(Debug, Default)]
pub struct ShadowVfsLog {
    entries: Mutex<Vec<(PathBuf, Vec<u8>)>>,
}

impl ShadowVfsLog {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
        }
    }

    /// Records or updates a file write in the shadow log.
    pub fn record_write(&self, path: impl Into<PathBuf>, content: Vec<u8>) {
        let path = path.into();
        match self.entries.lock() {
            Ok(mut lock) => {
                if let Some(entry) = lock.iter_mut().find(|(p, _)| *p == path) {
                    entry.1 = content;
                } else {
                    lock.push((path, content));
                }
            }
            Err(poisoned) => {
                let mut lock = poisoned.into_inner();
                if let Some(entry) = lock.iter_mut().find(|(p, _)| *p == path) {
                    entry.1 = content;
                } else {
                    lock.push((path, content));
                }
            }
        }
    }

    /// Safely snapshots all recorded VFS files without panicking on mutex poisoning.
    pub fn snapshot(&self) -> Vec<(PathBuf, Vec<u8>)> {
        match self.entries.lock() {
            Ok(lock) => lock.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    /// Clears all recorded entries.
    pub fn clear(&self) {
        match self.entries.lock() {
            Ok(mut lock) => lock.clear(),
            Err(poisoned) => poisoned.into_inner().clear(),
        }
    }
}

static GLOBAL_SHADOW_VFS: std::sync::OnceLock<ShadowVfsLog> = std::sync::OnceLock::new();

/// Access the global singleton ShadowVfsLog instance.
pub fn global_shadow_vfs() -> &'static ShadowVfsLog {
    GLOBAL_SHADOW_VFS.get_or_init(ShadowVfsLog::new)
}

/// Category and detail of an execution failure captured for DST reproduction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureKind {
    RuntimeError(String),
    Panic(String),
    InvariantViolation { oracle: String, details: String },
    CompilationError(String),
    WatchdogTimeout { step_limit: u64, elapsed_ms: u64 },
    HostError(String),
    Unknown(String),
}

impl fmt::Display for FailureKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
            Self::Panic(msg) => write!(f, "Panic: {}", msg),
            Self::InvariantViolation { oracle, details } => {
                write!(f, "Invariant violation ({}): {}", oracle, details)
            }
            Self::CompilationError(msg) => write!(f, "Compilation error: {}", msg),
            Self::WatchdogTimeout {
                step_limit,
                elapsed_ms,
            } => write!(
                f,
                "Watchdog timeout: step_limit={} elapsed_ms={}",
                step_limit, elapsed_ms
            ),
            Self::HostError(msg) => write!(f, "Host error: {}", msg),
            Self::Unknown(msg) => write!(f, "Unknown failure: {}", msg),
        }
    }
}

/// Configuration options for capturing a failure into a ReplayBundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureConfig {
    pub bundle_output_path: PathBuf,
    pub script_path: Option<PathBuf>,
    pub script_source: Option<String>,
    pub args: Vec<String>,
    pub env_vars: Vec<(String, String)>,
    pub seed: u64,
    pub step_count: u64,
    pub target_name: String,
    pub oracle: Option<String>,
    pub include_source_tree: bool,
    pub include_vendor: bool,
}

impl CaptureConfig {
    pub fn new(output_path: impl Into<PathBuf>) -> Self {
        Self {
            bundle_output_path: output_path.into(),
            script_path: None,
            script_source: None,
            args: Vec::new(),
            env_vars: Vec::new(),
            seed: 0,
            step_count: 0,
            target_name: "failure_capture".to_string(),
            oracle: None,
            include_source_tree: false,
            include_vendor: false,
        }
    }

    pub fn with_script(mut self, path: impl Into<PathBuf>, source: impl Into<String>) -> Self {
        self.script_path = Some(path.into());
        self.script_source = Some(source.into());
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_step_count(mut self, steps: u64) -> Self {
        self.step_count = steps;
        self
    }

    pub fn with_target_name(mut self, name: impl Into<String>) -> Self {
        self.target_name = name.into();
        self
    }

    pub fn with_oracle(mut self, oracle: impl Into<String>) -> Self {
        self.oracle = Some(oracle.into());
        self
    }

    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.args = args;
        self
    }

    pub fn with_env_vars(mut self, env_vars: Vec<(String, String)>) -> Self {
        self.env_vars = env_vars;
        self
    }
}

/// Errors occurring during the failure capture process.
/// Distinguishes capture-level failures (e.g. unwritable output destination)
/// from the original underlying execution failure.
#[derive(Debug)]
pub enum CaptureError {
    Io(std::io::Error),
    Replay(ReplayError),
    UnwritableOutput(String),
    Serialization(String),
}

impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "Capture I/O error: {}", err),
            Self::Replay(err) => write!(f, "Capture Replay error: {}", err),
            Self::UnwritableOutput(msg) => write!(f, "Capture unwritable output error: {}", msg),
            Self::Serialization(msg) => write!(f, "Capture serialization error: {}", msg),
        }
    }
}

impl std::error::Error for CaptureError {}

impl From<std::io::Error> for CaptureError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<ReplayError> for CaptureError {
    fn from(e: ReplayError) -> Self {
        Self::Replay(e)
    }
}

/// Construct a complete ReplayBundle from failure context and simulation telemetry.
pub fn capture_failure_to_bundle(
    config: &CaptureConfig,
    failure: &FailureKind,
    events: EventLog,
    vfs_snapshot: Vec<(PathBuf, Vec<u8>)>,
) -> Result<ReplayBundle, CaptureError> {
    let now_timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let failure_desc = serde_json::to_string(failure)
        .unwrap_or_else(|_| format!("{:?}", failure));

    let mut manifest = ReplayManifest::new(
        config.seed,
        &config.target_name,
        config.step_count,
    )
    .with_timestamp(now_timestamp)
    .with_fault_config(failure_desc);

    if let Some(oracle) = &config.oracle {
        manifest = manifest.with_oracle(oracle);
    } else if let FailureKind::InvariantViolation { oracle, .. } = failure {
        manifest = manifest.with_oracle(oracle);
    }

    let mut bundle = ReplayBundle::new(manifest, events, vfs_snapshot.clone());

    // 1. Script source
    if let Some(src) = &config.script_source {
        bundle = bundle.with_script(src.clone());
    } else if let Some(path) = &config.script_path {
        if let Ok(src) = fs::read_to_string(path) {
            bundle = bundle.with_script(src);
        }
    }

    // 2. CLI Arguments
    if !config.args.is_empty() {
        let args_json = serde_json::to_string(&config.args)
            .map_err(|e| CaptureError::Serialization(e.to_string()))?;
        bundle = bundle.with_args(args_json);
    }

    // 3. Environment Variables
    if !config.env_vars.is_empty() {
        let env_json = serde_json::to_string(&config.env_vars)
            .map_err(|e| CaptureError::Serialization(e.to_string()))?;
        bundle = bundle.with_env(env_json);
    }

    // 4. Initial VFS tar archive
    if !vfs_snapshot.is_empty() {
        let mut vfs_tar = UstarArchive::new();
        for (path, content) in &vfs_snapshot {
            let relative = path
                .strip_prefix("/")
                .unwrap_or(path)
                .to_string_lossy();
            vfs_tar.add_file(&relative, content)?;
        }
        bundle = bundle.with_initial_vfs(vfs_tar.finish());
    }

    // 5. Source tree tar archive (if requested)
    if config.include_source_tree {
        let mut src_tar = UstarArchive::new();
        if let Some(script_path) = &config.script_path {
            if let Some(src) = &config.script_source {
                let name = script_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("script.lust");
                src_tar.add_file(name, src.as_bytes())?;
            }
        }
        bundle = bundle.with_source_tree(src_tar.finish());
    }

    Ok(bundle)
}

/// Atomically writes out a ReplayBundle archive and verifies output writeability.
/// If writing fails (e.g. destination directory is unwritable), returns `CaptureError::UnwritableOutput`
/// and guarantees that any temporary files created during the attempt are cleaned up.
pub fn write_bundle_and_reproduction_task(
    bundle: &ReplayBundle,
    output_path: &Path,
) -> Result<PathBuf, CaptureError> {
    let parent = output_path.parent().unwrap_or_else(|| Path::new("."));
    if !parent.exists() {
        if let Err(e) = fs::create_dir_all(parent) {
            return Err(CaptureError::UnwritableOutput(format!(
                "Failed to create bundle parent directory '{}': {}",
                parent.display(),
                e
            )));
        }
    }

    // Verify parent is writable by testing file creation
    let pid = std::process::id();
    let now_nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp_filename = format!(".tmp_bundle_{}_{}.tar", pid, now_nanos);
    let tmp_path = parent.join(tmp_filename);

    let tar_bytes = bundle.export_tar_bytes().map_err(CaptureError::Replay)?;

    let mut tmp_file = match File::create(&tmp_path) {
        Ok(f) => f,
        Err(e) => {
            return Err(CaptureError::UnwritableOutput(format!(
                "Cannot write bundle to '{}': {}",
                output_path.display(),
                e
            )));
        }
    };

    if let Err(e) = tmp_file.write_all(&tar_bytes) {
        let _ = fs::remove_file(&tmp_path);
        return Err(CaptureError::UnwritableOutput(format!(
            "Failed to write TAR payload to '{}': {}",
            tmp_path.display(),
            e
        )));
    }

    if let Err(e) = tmp_file.sync_all() {
        let _ = fs::remove_file(&tmp_path);
        return Err(CaptureError::UnwritableOutput(format!(
            "Failed to sync TAR payload to '{}': {}",
            tmp_path.display(),
            e
        )));
    }
    drop(tmp_file);

    // Atomically rename temporary file to destination path
    if let Err(e) = fs::rename(&tmp_path, output_path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(CaptureError::UnwritableOutput(format!(
            "Failed to move bundle into place at '{}': {}",
            output_path.display(),
            e
        )));
    }

    Ok(output_path.to_path_buf())
}

/// Global hook state for panic capture
struct PanicHookContext {
    config: CaptureConfig,
    events: Mutex<EventLog>,
}

static PANIC_HOOK_CTX: std::sync::OnceLock<PanicHookContext> = std::sync::OnceLock::new();

/// Installs a panic hook that automatically records panic details, extracts the current
/// VFS state from `ShadowVfsLog`, and writes a ReplayBundle to the specified path.
pub fn install_failure_panic_hook(config: CaptureConfig) {
    let output_path = config.bundle_output_path.clone();
    let ctx = PANIC_HOOK_CTX.get_or_init(|| PanicHookContext {
        config: config.clone(),
        events: Mutex::new(EventLog::new()),
    });

    let prev_hook = std::panic::take_hook();

    std::panic::set_hook(Box::new(move |panic_info| {
        let msg = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "Box<dyn Any>".to_string()
        };

        let location = panic_info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "unknown location".to_string());

        let failure = FailureKind::Panic(format!("{} at {}", msg, location));
        let events = match ctx.events.lock() {
            Ok(lock) => lock.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        };

        let vfs_snapshot = global_shadow_vfs().snapshot();

        match capture_failure_to_bundle(&ctx.config, &failure, events, vfs_snapshot) {
            Ok(bundle) => {
                match write_bundle_and_reproduction_task(&bundle, &output_path) {
                    Ok(path) => {
                        eprintln!(
                            "[DST-CAP] Failure captured to replay bundle: {}",
                            path.display()
                        );
                        eprintln!(
                            "[DST-CAP] Reproduce with: lust sim replay {}",
                            path.display()
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "[DST-CAP] Warning: Failed to write replay bundle: {}",
                            e
                        );
                    }
                }
            }
            Err(e) => {
                eprintln!("[DST-CAP] Warning: Failed to capture replay bundle: {}", e);
            }
        }

        prev_hook(panic_info);
    }));
}

//! Autonomous Replay Bundle & Shrinking Engine (DST Phase 6, Gate 6).
//!
//! Provides:
//! 1. `ReplayManifest`: metadata describing seed, version, target, faults, violated invariant, steps, commit SHA.
//! 2. `EventLog`: deterministic simulation event recording stream in `events.jsonl` format.
//! 3. `ReplayBundle`: export and import of self-contained archives (POSIX tar byte stream / directory) containing
//!    `manifest.json`, `events.jsonl`, and snapshot of the virtual filesystem state.
//! 4. `ReplayEngine`: bitwise identical trace execution verification on a clean simulation environment.
//! 5. `Shrinker`: deterministic search for the minimal execution prefix reproducing invariant violations.

use crate::sim::invariants::InvariantViolation;
use crate::vfs::{MemFs, VirtualFileSystem};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[path = "capture.rs"]
pub mod capture;
pub use capture::*;

/// Error types occurring during Replay Bundle operations.
#[derive(Debug)]
pub enum ReplayError {
    Serialization(String),
    Deserialization(String),
    Io(std::io::Error),
    TarFormat(String),
    MissingEntry(String),
    TraceDivergence {
        step: u64,
        expected: String,
        actual: String,
    },
    ShrinkFailed(String),
    ExecutionFailed(String),
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Serialization(msg) => write!(f, "Replay serialization error: {}", msg),
            Self::Deserialization(msg) => write!(f, "Replay deserialization error: {}", msg),
            Self::Io(err) => write!(f, "Replay I/O error: {}", err),
            Self::TarFormat(msg) => write!(f, "Replay TAR format error: {}", msg),
            Self::MissingEntry(entry) => write!(f, "Replay bundle missing entry: {}", entry),
            Self::TraceDivergence {
                step,
                expected,
                actual,
            } => write!(
                f,
                "Replay trace divergence at step {}: expected '{}', got '{}'",
                step, expected, actual
            ),
            Self::ShrinkFailed(msg) => write!(f, "Replay shrink failed: {}", msg),
            Self::ExecutionFailed(msg) => write!(f, "Replay execution failed: {}", msg),
        }
    }
}

impl std::error::Error for ReplayError {}

impl From<std::io::Error> for ReplayError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<serde_json::Error> for ReplayError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization(err.to_string())
    }
}

/// Metadata identifying a simulation run for deterministic replay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayManifest {
    pub seed: u64,
    pub version: String,
    pub target_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fault_config: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invariant_oracle_name: Option<String>,
    pub step_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit_sha: Option<String>,
    pub timestamp: u64,
}

impl ReplayManifest {
    pub fn new(seed: u64, target_name: impl Into<String>, step_count: u64) -> Self {
        Self {
            seed,
            version: env!("CARGO_PKG_VERSION").to_string(),
            target_name: target_name.into(),
            fault_config: None,
            invariant_oracle_name: None,
            step_count,
            commit_sha: option_env!("GIT_HASH").map(|s| s.to_string()),
            timestamp: 0,
        }
    }

    pub fn with_fault_config(mut self, fault: impl Into<String>) -> Self {
        self.fault_config = Some(fault.into());
        self
    }

    pub fn with_oracle(mut self, oracle: impl Into<String>) -> Self {
        self.invariant_oracle_name = Some(oracle.into());
        self
    }

    pub fn with_timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = timestamp;
        self
    }

    pub fn with_commit_sha(mut self, sha: impl Into<String>) -> Self {
        self.commit_sha = Some(sha.into());
        self
    }

    pub fn to_json(&self) -> Result<String, ReplayError> {
        serde_json::to_string_pretty(self).map_err(|e| ReplayError::Serialization(e.to_string()))
    }

    pub fn from_json(json: &str) -> Result<Self, ReplayError> {
        serde_json::from_str(json).map_err(|e| ReplayError::Deserialization(e.to_string()))
    }
}

/// Individual simulation event kinds for fine-grained execution trace logging.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "detail")]
pub enum ReplayEventKind {
    TaskScheduled {
        task_id: u64,
        quantum: u64,
    },
    StateChanged {
        task_id: u64,
        from_state: String,
        to_state: String,
    },
    VfsOperation {
        op: String,
        path: String,
        success: bool,
        bytes: usize,
    },
    QuantumAssigned {
        task_id: u64,
        quantum: u64,
    },
    GasBurned {
        task_id: u64,
        gas_delta: u64,
        total_gas: u64,
    },
    TimeAdvanced {
        delta_nanos: u64,
        now_nanos: u64,
    },
    TimerFired {
        timer_id: u64,
    },
    InvariantChecked {
        oracle: String,
        passed: bool,
    },
    InvariantViolated {
        oracle: String,
        details: String,
    },
    Custom {
        tag: String,
        payload: String,
    },
    SafepointChecked {
        safepoint_id: u64,
        mask_depth: usize,
    },
    InterruptInjected {
        reason: String,
        deferred: bool,
    },
    InterruptDelivered {
        reason: String,
        safepoint_id: u64,
    },
    MaskEntered {
        new_depth: usize,
    },
    MaskExited {
        new_depth: usize,
        unmasked_interrupt: Option<String>,
    },
    FinalizerRegistered {
        finalizer_id: u64,
    },
    FinalizerExecuted {
        finalizer_id: u64,
    },
    Tier2CompiledEntry {
        func_name: String,
        target_block: usize,
    },
}

/// A timestamped, step-indexed simulation event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayEvent {
    pub step: u64,
    pub timestamp_nanos: u64,
    pub event: ReplayEventKind,
}

/// In-memory stream of deterministic simulation events (`events.jsonl` format).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventLog {
    pub events: Vec<ReplayEvent>,
}

impl EventLog {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    pub fn record(&mut self, step: u64, timestamp_nanos: u64, event: ReplayEventKind) {
        self.events.push(ReplayEvent {
            step,
            timestamp_nanos,
            event,
        });
    }

    pub fn push(&mut self, event: ReplayEvent) {
        self.events.push(event);
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Serializes event log as a newline-delimited JSON Lines (`events.jsonl`) string.
    pub fn to_jsonl(&self) -> Result<String, ReplayError> {
        let mut out = String::new();
        for ev in &self.events {
            let line =
                serde_json::to_string(ev).map_err(|e| ReplayError::Serialization(e.to_string()))?;
            out.push_str(&line);
            out.push('\n');
        }
        Ok(out)
    }

    /// Deserializes event log from a newline-delimited JSON Lines (`events.jsonl`) string.
    pub fn from_jsonl(jsonl: &str) -> Result<Self, ReplayError> {
        let mut events = Vec::new();
        for (line_idx, line) in jsonl.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let ev: ReplayEvent = serde_json::from_str(trimmed).map_err(|e| {
                ReplayError::Deserialization(format!("Line {}: {}", line_idx + 1, e))
            })?;
            events.push(ev);
        }
        Ok(Self { events })
    }
}

/// A self-contained simulation archive bundle.
/// Contains `manifest.json`, `events.jsonl`, virtual filesystem snapshot,
/// and optional execution snapshot lock artifacts (`script.lust`, `args.json`, `env.json`,
/// `source_tree.tar`, `vendor.tar`, `initial_vfs.tar`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayBundle {
    pub manifest: ReplayManifest,
    pub events: EventLog,
    pub vfs_files: Vec<(PathBuf, Vec<u8>)>,
    pub source_tree_tar: Option<Vec<u8>>,
    pub vendor_tar: Option<Vec<u8>>,
    pub script_lust: Option<String>,
    pub args_json: Option<String>,
    pub env_json: Option<String>,
    pub initial_vfs_tar: Option<Vec<u8>>,
}

impl ReplayBundle {
    pub fn new(
        manifest: ReplayManifest,
        events: EventLog,
        vfs_files: Vec<(PathBuf, Vec<u8>)>,
    ) -> Self {
        Self {
            manifest,
            events,
            vfs_files,
            source_tree_tar: None,
            vendor_tar: None,
            script_lust: None,
            args_json: None,
            env_json: None,
            initial_vfs_tar: None,
        }
    }

    pub fn with_source_tree(mut self, tar_bytes: Vec<u8>) -> Self {
        self.source_tree_tar = Some(tar_bytes);
        self
    }

    pub fn with_vendor(mut self, tar_bytes: Vec<u8>) -> Self {
        self.vendor_tar = Some(tar_bytes);
        self
    }

    pub fn with_script(mut self, script: impl Into<String>) -> Self {
        self.script_lust = Some(script.into());
        self
    }

    pub fn with_args(mut self, args: impl Into<String>) -> Self {
        self.args_json = Some(args.into());
        self
    }

    pub fn with_env(mut self, env: impl Into<String>) -> Self {
        self.env_json = Some(env.into());
        self
    }

    pub fn with_initial_vfs(mut self, tar_bytes: Vec<u8>) -> Self {
        self.initial_vfs_tar = Some(tar_bytes);
        self
    }

    /// Creates a ReplayBundle from current MemFs and event log.
    pub fn from_simulation(manifest: ReplayManifest, events: EventLog, mem_fs: &MemFs) -> Self {
        Self::new(manifest, events, mem_fs.snapshot())
    }

    /// Restores the virtual filesystem into a target MemFs instance.
    pub fn restore_vfs_into(&self, mem_fs: &MemFs) {
        mem_fs.restore_snapshot(&self.vfs_files);
    }

    /// Serializes the bundle into a standard POSIX USTAR byte archive (`bundle.tar`).
    pub fn export_tar_bytes(&self) -> Result<Vec<u8>, ReplayError> {
        let mut archive = UstarArchive::new();

        // 1. manifest.json
        let manifest_bytes = self.manifest.to_json()?.into_bytes();
        archive.add_file("manifest.json", &manifest_bytes)?;

        // 2. events.jsonl
        let events_bytes = self.events.to_jsonl()?.into_bytes();
        archive.add_file("events.jsonl", &events_bytes)?;

        // 3. script.lust
        if let Some(script) = &self.script_lust {
            archive.add_file("script.lust", script.as_bytes())?;
        }

        // 4. args.json
        if let Some(args) = &self.args_json {
            archive.add_file("args.json", args.as_bytes())?;
        }

        // 5. env.json
        if let Some(env) = &self.env_json {
            archive.add_file("env.json", env.as_bytes())?;
        }

        // 6. source_tree.tar
        if let Some(src_tar) = &self.source_tree_tar {
            archive.add_file("source_tree.tar", src_tar)?;
        }

        // 7. vendor.tar
        if let Some(vendor_tar) = &self.vendor_tar {
            archive.add_file("vendor.tar", vendor_tar)?;
        }

        // 8. initial_vfs.tar
        if let Some(initial_vfs) = &self.initial_vfs_tar {
            archive.add_file("initial_vfs.tar", initial_vfs)?;
        } else if !self.vfs_files.is_empty() {
            let mut vfs_tar = UstarArchive::new();
            for (path, content) in &self.vfs_files {
                let relative = path.strip_prefix("/").unwrap_or(path).to_string_lossy();
                vfs_tar.add_file(&relative, content)?;
            }
            let vfs_tar_bytes = vfs_tar.finish();
            archive.add_file("initial_vfs.tar", &vfs_tar_bytes)?;
        }

        // 9. vfs files
        for (path, content) in &self.vfs_files {
            let relative_path = path
                .strip_prefix("/")
                .unwrap_or(path)
                .to_string_lossy()
                .into_owned();
            let entry_name = format!("vfs/{}", relative_path);
            archive.add_file(&entry_name, content)?;
        }

        Ok(archive.finish())
    }

    /// Deserializes a ReplayBundle from a standard POSIX USTAR byte archive.
    pub fn import_tar_bytes(bytes: &[u8]) -> Result<Self, ReplayError> {
        let entries = UstarArchive::parse(bytes)?;

        let mut manifest_opt = None;
        let mut events_opt = None;
        let mut script_lust = None;
        let mut args_json = None;
        let mut env_json = None;
        let mut source_tree_tar = None;
        let mut vendor_tar = None;
        let mut initial_vfs_tar = None;
        let mut vfs_files = Vec::new();

        for (name, data) in entries {
            if name == "manifest.json" {
                let json = std::str::from_utf8(&data)
                    .map_err(|e| ReplayError::Deserialization(e.to_string()))?;
                manifest_opt = Some(ReplayManifest::from_json(json)?);
            } else if name == "events.jsonl" {
                let jsonl = std::str::from_utf8(&data)
                    .map_err(|e| ReplayError::Deserialization(e.to_string()))?;
                events_opt = Some(EventLog::from_jsonl(jsonl)?);
            } else if name == "script.lust" {
                let s = std::str::from_utf8(&data)
                    .map_err(|e| ReplayError::Deserialization(e.to_string()))?;
                script_lust = Some(s.to_string());
            } else if name == "args.json" {
                let s = std::str::from_utf8(&data)
                    .map_err(|e| ReplayError::Deserialization(e.to_string()))?;
                args_json = Some(s.to_string());
            } else if name == "env.json" {
                let s = std::str::from_utf8(&data)
                    .map_err(|e| ReplayError::Deserialization(e.to_string()))?;
                env_json = Some(s.to_string());
            } else if name == "source_tree.tar" {
                source_tree_tar = Some(data);
            } else if name == "vendor.tar" {
                vendor_tar = Some(data);
            } else if name == "initial_vfs.tar" {
                initial_vfs_tar = Some(data);
            } else if let Some(vfs_path) = name.strip_prefix("vfs/") {
                let full_path = if vfs_path.starts_with('/') {
                    PathBuf::from(vfs_path)
                } else {
                    PathBuf::from(format!("/{}", vfs_path))
                };
                vfs_files.push((full_path, data));
            }
        }

        let manifest =
            manifest_opt.ok_or_else(|| ReplayError::MissingEntry("manifest.json".to_string()))?;
        let events =
            events_opt.ok_or_else(|| ReplayError::MissingEntry("events.jsonl".to_string()))?;

        if vfs_files.is_empty() {
            if let Some(vfs_tar_bytes) = &initial_vfs_tar {
                if let Ok(vfs_entries) = UstarArchive::parse(vfs_tar_bytes) {
                    for (entry_path, entry_data) in vfs_entries {
                        let full_path = if entry_path.starts_with('/') {
                            PathBuf::from(entry_path)
                        } else {
                            PathBuf::from(format!("/{}", entry_path))
                        };
                        vfs_files.push((full_path, entry_data));
                    }
                }
            }
        }

        vfs_files.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(Self {
            manifest,
            events,
            vfs_files,
            source_tree_tar,
            vendor_tar,
            script_lust,
            args_json,
            env_json,
            initial_vfs_tar,
        })
    }

    /// Exports bundle as a directory on the specified VirtualFileSystem.
    pub fn export_to_directory(
        &self,
        dir: &Path,
        vfs: &dyn VirtualFileSystem,
    ) -> Result<(), ReplayError> {
        vfs.create_dir_all(dir)?;

        // 1. manifest.json
        let manifest_path = dir.join("manifest.json");
        vfs.write(&manifest_path, self.manifest.to_json()?.as_bytes())?;

        // 2. events.jsonl
        let events_path = dir.join("events.jsonl");
        vfs.write(&events_path, self.events.to_jsonl()?.as_bytes())?;

        // 3. Optional metadata files
        if let Some(script) = &self.script_lust {
            vfs.write(&dir.join("script.lust"), script.as_bytes())?;
        }
        if let Some(args) = &self.args_json {
            vfs.write(&dir.join("args.json"), args.as_bytes())?;
        }
        if let Some(env) = &self.env_json {
            vfs.write(&dir.join("env.json"), env.as_bytes())?;
        }
        if let Some(src_tar) = &self.source_tree_tar {
            vfs.write(&dir.join("source_tree.tar"), src_tar)?;
        }
        if let Some(vendor_tar) = &self.vendor_tar {
            vfs.write(&dir.join("vendor.tar"), vendor_tar)?;
        }
        if let Some(initial_vfs) = &self.initial_vfs_tar {
            vfs.write(&dir.join("initial_vfs.tar"), initial_vfs)?;
        }

        // 4. vfs files
        let vfs_root = dir.join("vfs");
        vfs.create_dir_all(&vfs_root)?;

        for (path, content) in &self.vfs_files {
            let relative = path.strip_prefix("/").unwrap_or(path);
            let target_path = vfs_root.join(relative);
            if let Some(parent) = target_path.parent() {
                vfs.create_dir_all(parent)?;
            }
            vfs.write(&target_path, content)?;
        }

        Ok(())
    }

    /// Imports bundle from a directory on the specified VirtualFileSystem.
    pub fn import_from_directory(
        dir: &Path,
        vfs: &dyn VirtualFileSystem,
    ) -> Result<Self, ReplayError> {
        // 1. manifest.json
        let manifest_path = dir.join("manifest.json");
        let manifest_str = vfs.read_to_string(&manifest_path)?;
        let manifest = ReplayManifest::from_json(&manifest_str)?;

        // 2. events.jsonl
        let events_path = dir.join("events.jsonl");
        let events_str = vfs.read_to_string(&events_path)?;
        let events = EventLog::from_jsonl(&events_str)?;

        // 3. Optional metadata
        let script_path = dir.join("script.lust");
        let script_lust = if vfs.exists(&script_path) {
            Some(vfs.read_to_string(&script_path)?)
        } else {
            None
        };

        let args_path = dir.join("args.json");
        let args_json = if vfs.exists(&args_path) {
            Some(vfs.read_to_string(&args_path)?)
        } else {
            None
        };

        let env_path = dir.join("env.json");
        let env_json = if vfs.exists(&env_path) {
            Some(vfs.read_to_string(&env_path)?)
        } else {
            None
        };

        let src_path = dir.join("source_tree.tar");
        let source_tree_tar = if vfs.exists(&src_path) {
            Some(vfs.read(&src_path)?)
        } else {
            None
        };

        let vendor_path = dir.join("vendor.tar");
        let vendor_tar = if vfs.exists(&vendor_path) {
            Some(vfs.read(&vendor_path)?)
        } else {
            None
        };

        let vfs_tar_path = dir.join("initial_vfs.tar");
        let initial_vfs_tar = if vfs.exists(&vfs_tar_path) {
            Some(vfs.read(&vfs_tar_path)?)
        } else {
            None
        };

        // 4. vfs files
        let mut vfs_files = Vec::new();
        let vfs_root = dir.join("vfs");
        if vfs.exists(&vfs_root) {
            collect_vfs_files_recursive(&vfs_root, &vfs_root, vfs, &mut vfs_files)?;
        }
        vfs_files.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(Self {
            manifest,
            events,
            vfs_files,
            source_tree_tar,
            vendor_tar,
            script_lust,
            args_json,
            env_json,
            initial_vfs_tar,
        })
    }

    /// Exports bundle to a TAR file on the specified VirtualFileSystem.
    pub fn export_to_tar(
        &self,
        tar_path: &Path,
        vfs: &dyn VirtualFileSystem,
    ) -> Result<(), ReplayError> {
        if let Some(parent) = tar_path.parent() {
            vfs.create_dir_all(parent)?;
        }
        let tar_bytes = self.export_tar_bytes()?;
        vfs.write(tar_path, &tar_bytes)?;
        Ok(())
    }

    /// Imports bundle from a TAR file on the specified VirtualFileSystem.
    pub fn import_from_tar(
        tar_path: &Path,
        vfs: &dyn VirtualFileSystem,
    ) -> Result<Self, ReplayError> {
        let bytes = vfs.read(tar_path)?;
        Self::import_tar_bytes(&bytes)
    }

    /// Exports bundle to a TAR file on the host filesystem (via OsFs).
    pub fn export_to_host_tar(&self, tar_path: &Path) -> Result<(), ReplayError> {
        let os_fs = crate::vfs::OsFs::new();
        self.export_to_tar(tar_path, &os_fs)
    }

    /// Imports bundle from a TAR file on the host filesystem (via OsFs).
    pub fn import_from_host_tar(tar_path: &Path) -> Result<Self, ReplayError> {
        let os_fs = crate::vfs::OsFs::new();
        Self::import_from_tar(tar_path, &os_fs)
    }
}

fn collect_vfs_files_recursive(
    current: &Path,
    root: &Path,
    vfs: &dyn VirtualFileSystem,
    out: &mut Vec<(PathBuf, Vec<u8>)>,
) -> Result<(), ReplayError> {
    let entries = vfs.read_dir(current)?;
    for entry in entries {
        let meta = vfs.metadata(&entry)?;
        if meta.is_dir() {
            collect_vfs_files_recursive(&entry, root, vfs, out)?;
        } else if meta.is_file() {
            let content = vfs.read(&entry)?;
            let relative = entry.strip_prefix(root).unwrap_or(&entry);
            let virtual_path = PathBuf::from(format!("/{}", relative.to_string_lossy()));
            out.push((virtual_path, content));
        }
    }
    Ok(())
}

/// Verification result of bitwise trace replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayResult {
    pub matched: bool,
    pub total_events_checked: usize,
    pub divergence_step: Option<u64>,
    pub divergence_detail: Option<String>,
}

/// Verification engine for deterministic execution traces.
pub struct ReplayEngine;

impl ReplayEngine {
    /// Compares two event logs step-by-step for bit-for-bit equivalence.
    pub fn verify_trace(expected: &EventLog, actual: &EventLog) -> ReplayResult {
        let min_len = expected.len().min(actual.len());
        for i in 0..min_len {
            let exp = &expected.events[i];
            let act = &actual.events[i];

            if exp != act {
                return ReplayResult {
                    matched: false,
                    total_events_checked: i,
                    divergence_step: Some(act.step),
                    divergence_detail: Some(format!(
                        "Event mismatch at index {}: expected '{:?}', got '{:?}'",
                        i, exp, act
                    )),
                };
            }
        }

        if expected.len() != actual.len() {
            let extra_step = if actual.len() > expected.len() {
                actual.events[min_len].step
            } else {
                expected.events[min_len].step
            };
            return ReplayResult {
                matched: false,
                total_events_checked: min_len,
                divergence_step: Some(extra_step),
                divergence_detail: Some(format!(
                    "Event count mismatch: expected {} events, actual {} events",
                    expected.len(),
                    actual.len()
                )),
            };
        }

        ReplayResult {
            matched: true,
            total_events_checked: expected.len(),
            divergence_step: None,
            divergence_detail: None,
        }
    }

    /// Replays a bundle using a provided execution runner function.
    pub fn replay_with<F>(bundle: &ReplayBundle, runner: F) -> Result<ReplayResult, ReplayError>
    where
        F: FnOnce(&ReplayManifest, &[(PathBuf, Vec<u8>)]) -> Result<EventLog, String>,
    {
        let actual_log =
            runner(&bundle.manifest, &bundle.vfs_files).map_err(ReplayError::ExecutionFailed)?;
        Ok(Self::verify_trace(&bundle.events, &actual_log))
    }
}

/// Result of deterministic execution trace shrinking.
#[derive(Debug, Clone)]
pub struct ShrunkReplay {
    pub original_steps: u64,
    pub shrunk_steps: u64,
    pub invariant_violated: String,
    pub minimal_bundle: ReplayBundle,
}

/// Deterministic execution prefix shrinker for invariant violations.
pub struct Shrinker {
    pub max_iterations: usize,
}

impl Default for Shrinker {
    fn default() -> Self {
        Self::new()
    }
}

impl Shrinker {
    pub fn new() -> Self {
        Self { max_iterations: 64 }
    }

    pub fn with_max_iterations(mut self, iters: usize) -> Self {
        self.max_iterations = iters;
        self
    }

    /// Binary search for minimal step count reproducing the invariant violation.
    pub fn shrink_steps<F>(
        &self,
        bundle: &ReplayBundle,
        mut runner: F,
    ) -> Result<ShrunkReplay, ReplayError>
    where
        F: FnMut(u64) -> Result<(), InvariantViolation>,
    {
        let target_oracle = bundle
            .manifest
            .invariant_oracle_name
            .clone()
            .unwrap_or_else(|| "UNKNOWN_ORACLE".to_string());

        let initial_steps = bundle.manifest.step_count;
        if initial_steps <= 1 {
            return Ok(ShrunkReplay {
                original_steps: initial_steps,
                shrunk_steps: initial_steps,
                invariant_violated: target_oracle,
                minimal_bundle: bundle.clone(),
            });
        }

        // Verify initial failure
        if runner(initial_steps).is_ok() {
            return Err(ReplayError::ShrinkFailed(
                "Initial execution did not violate invariant".to_string(),
            ));
        }

        let mut low: u64 = 1;
        let mut high: u64 = initial_steps;
        let mut min_reproducing = initial_steps;
        let mut iters = 0;

        while low <= high && iters < self.max_iterations {
            iters += 1;
            let mid = low + (high - low) / 2;
            match runner(mid) {
                Err(err) => {
                    let err_str = err.to_string();
                    if err_str.contains(&target_oracle) || target_oracle == "UNKNOWN_ORACLE" {
                        min_reproducing = mid;
                        if mid == 1 {
                            break;
                        }
                        high = mid - 1;
                    } else {
                        low = mid + 1;
                    }
                }
                Ok(()) => {
                    low = mid + 1;
                }
            }
        }

        let mut shrunk_bundle = bundle.clone();
        shrunk_bundle.manifest.step_count = min_reproducing;
        shrunk_bundle
            .events
            .events
            .retain(|e| e.step <= min_reproducing);

        Ok(ShrunkReplay {
            original_steps: initial_steps,
            shrunk_steps: min_reproducing,
            invariant_violated: target_oracle,
            minimal_bundle: shrunk_bundle,
        })
    }
}

// =============================================================================
// Standalone POSIX USTAR Implementation with Prefix (up to 256 bytes) Support
// =============================================================================

pub struct UstarArchive {
    buffer: Vec<u8>,
}

impl Default for UstarArchive {
    fn default() -> Self {
        Self::new()
    }
}

impl UstarArchive {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    pub fn add_file(&mut self, name: &str, content: &[u8]) -> Result<(), ReplayError> {
        let mut header = [0u8; 512];

        // 1. File name and prefix (POSIX USTAR header)
        if name.len() <= 100 {
            let name_bytes = name.as_bytes();
            header[..name_bytes.len()].copy_from_slice(name_bytes);
        } else if name.len() <= 256 {
            let name_bytes = name.as_bytes();
            let mut split_pos = None;
            let max_prefix = 155.min(name.len());
            for pos in (1..=max_prefix).rev() {
                if name_bytes[pos - 1] == b'/' {
                    let suffix_len = name.len() - pos;
                    if suffix_len <= 100 && suffix_len > 0 {
                        split_pos = Some(pos - 1);
                        break;
                    }
                }
            }

            match split_pos {
                Some(pos) => {
                    let prefix = &name_bytes[..pos];
                    let suffix = &name_bytes[pos + 1..];
                    header[345..345 + prefix.len()].copy_from_slice(prefix);
                    header[..suffix.len()].copy_from_slice(suffix);
                }
                None => {
                    return Err(ReplayError::TarFormat(format!(
                        "Filename cannot be split into USTAR prefix (<=155) and name (<=100): {}",
                        name
                    )));
                }
            }
        } else {
            return Err(ReplayError::TarFormat(format!(
                "Filename too long for USTAR header (max 256 bytes): {}",
                name
            )));
        }

        // 2. Mode: 100..108 (0o644)
        write_octal(&mut header[100..108], 0o644);

        // 3. UID: 108..116 (0)
        write_octal(&mut header[108..116], 0);

        // 4. GID: 116..124 (0)
        write_octal(&mut header[116..124], 0);

        // 5. Size: 124..136 (file size in octal)
        write_octal(&mut header[124..136], content.len() as u64);

        // 6. Mtime: 136..148 (0)
        write_octal(&mut header[136..148], 0);

        // 7. Checksum placeholder: 148..156 (spaces)
        for b in &mut header[148..156] {
            *b = b' ';
        }

        // 8. Typeflag: 156 (b'0' = normal file)
        header[156] = b'0';

        // 9. Magic: 257..263 ("ustar\0")
        header[257..263].copy_from_slice(b"ustar\0");

        // 10. Version: 263..265 ("00")
        header[263..265].copy_from_slice(b"00");

        // Compute checksum
        let mut chksum: u32 = 0;
        for b in &header {
            chksum += *b as u32;
        }
        write_chksum(&mut header[148..156], chksum);

        // Append header block
        self.buffer.extend_from_slice(&header);

        // Append content
        self.buffer.extend_from_slice(content);

        // Append padding to 512 boundary
        let remainder = content.len() % 512;
        if remainder != 0 {
            let padding = 512 - remainder;
            self.buffer.extend(core::iter::repeat_n(0u8, padding));
        }

        Ok(())
    }

    pub fn finish(mut self) -> Vec<u8> {
        // Two 512-byte zero blocks mark end of TAR archive
        self.buffer.extend(core::iter::repeat_n(0u8, 1024));
        self.buffer
    }

    pub fn parse(bytes: &[u8]) -> Result<Vec<(String, Vec<u8>)>, ReplayError> {
        let mut entries = Vec::new();
        let mut offset = 0;

        while offset + 512 <= bytes.len() {
            let block = &bytes[offset..offset + 512];
            if block.iter().all(|&b| b == 0) {
                // End of archive
                break;
            }

            // Verify USTAR magic
            if &block[257..262] != b"ustar" {
                return Err(ReplayError::TarFormat("Missing USTAR magic".to_string()));
            }

            // Extract file name suffix (0..100)
            let name_end = block[0..100]
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(100);
            let suffix = std::str::from_utf8(&block[0..name_end])
                .map_err(|e| ReplayError::TarFormat(e.to_string()))?;

            // Extract prefix (345..500)
            let prefix_end = block[345..500]
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(155);
            let prefix = std::str::from_utf8(&block[345..345 + prefix_end])
                .map_err(|e| ReplayError::TarFormat(e.to_string()))?;

            let name = if prefix.is_empty() {
                suffix.to_string()
            } else {
                format!("{}/{}", prefix, suffix)
            };

            // Extract size
            let size = parse_octal(&block[124..136])?;
            offset += 512;

            if offset + (size as usize) > bytes.len() {
                return Err(ReplayError::TarFormat(
                    "Unexpected EOF in TAR payload".to_string(),
                ));
            }

            let data = bytes[offset..offset + (size as usize)].to_vec();
            entries.push((name, data));

            offset += size as usize;
            let rem = size as usize % 512;
            if rem != 0 {
                offset += 512 - rem;
            }
        }

        Ok(entries)
    }
}

fn write_octal(slice: &mut [u8], value: u64) {
    let s = format!("{:01$o}\0", value, slice.len() - 1);
    let bytes = s.as_bytes();
    let copy_len = bytes.len().min(slice.len());
    slice[..copy_len].copy_from_slice(&bytes[..copy_len]);
}

fn write_chksum(slice: &mut [u8], value: u32) {
    let s = format!("{:06o}\0 ", value);
    let bytes = s.as_bytes();
    let copy_len = bytes.len().min(slice.len());
    slice[..copy_len].copy_from_slice(&bytes[..copy_len]);
}

fn parse_octal(slice: &[u8]) -> Result<u64, ReplayError> {
    let s = std::str::from_utf8(slice).map_err(|e| ReplayError::TarFormat(e.to_string()))?;
    let trimmed = s.trim_matches(['\0', ' ']);
    if trimmed.is_empty() {
        return Ok(0);
    }
    u64::from_str_radix(trimmed, 8)
        .map_err(|e| ReplayError::TarFormat(format!("Invalid octal '{}': {}", trimmed, e)))
}

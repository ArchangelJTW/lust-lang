//! Deterministic Fault Injection Engine (DST Phase 5, D8).
//!
//! Provides deterministic fault injection across:
//! 1. VFS I/O errors (read, write, create_dir_all, delete, metadata, canonicalize, exists).
//! 2. Spontaneous fiber interrupts / cancellations (InterruptReason::Cancelled).
//! 3. Memory exhaustion / OOM budget injection.

use crate::effect_abi::InterruptReason;
use crate::error::Result as LustResult;
use crate::vfs::{VfsMetadata, VirtualFileSystem};
use crate::vm::{TaskId, VM};
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::io::{Error as IoError, ErrorKind as IoErrorKind};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Supported VFS operation types for fault injection rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VfsOp {
    Read,
    Write,
    Append,
    CreateDir,
    Delete,
    Metadata,
    Canonicalize,
    Exists,
}

/// A rule specifying when and how to inject a VFS I/O fault.
#[derive(Debug, Clone)]
pub struct VfsFaultRule {
    pub op: Option<VfsOp>,
    pub path_pattern: Option<String>,
    pub error_kind: IoErrorKind,
    pub error_message: Option<String>,
    pub probability: f64,
    pub max_injections: Option<usize>,
    pub current_injections: usize,
}

impl VfsFaultRule {
    pub fn matches(&self, op: VfsOp, path: &Path) -> bool {
        if let Some(target_op) = self.op {
            if target_op != op {
                return false;
            }
        }
        if let Some(ref pattern) = self.path_pattern {
            let path_str = path.to_string_lossy();
            if !path_str.contains(pattern) {
                return false;
            }
        }
        true
    }
}

/// Policy for simulating memory allocation failures and budget exhaustion.
#[derive(Debug, Clone)]
pub enum MemoryFaultPolicy {
    None,
    MaxAllocBytes(usize),
    StepOom(u64),
    Probability { rate: f64 },
}

/// Scheduled spontaneous fiber interrupt.
#[derive(Debug, Clone)]
pub struct InterruptSchedule {
    pub fiber_id: Option<u64>,
    pub step: u64,
    pub reason: InterruptReason,
}

/// Generic classification of fault kinds for reporting and assertions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FaultKind {
    Vfs(IoErrorKind),
    MemoryOom,
    FiberInterrupt(InterruptReason),
}

/// Configurable, seed-driven fault configuration engine.
pub struct FaultConfiguration {
    pub seed: u64,
    pub rng: StdRng,
    pub vfs_rules: Vec<VfsFaultRule>,
    pub memory_policy: MemoryFaultPolicy,
    pub interrupt_schedules: Vec<InterruptSchedule>,
    pub vfs_fault_count: usize,
    pub memory_fault_count: usize,
    pub interrupt_fault_count: usize,
}

impl FaultConfiguration {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            rng: StdRng::seed_from_u64(seed),
            vfs_rules: Vec::new(),
            memory_policy: MemoryFaultPolicy::None,
            interrupt_schedules: Vec::new(),
            vfs_fault_count: 0,
            memory_fault_count: 0,
            interrupt_fault_count: 0,
        }
    }

    /// Add a deterministic rule injecting an I/O error on reads matching `path_pattern`.
    pub fn with_vfs_read_error(
        mut self,
        path_pattern: impl Into<String>,
        error_kind: IoErrorKind,
    ) -> Self {
        self.vfs_rules.push(VfsFaultRule {
            op: Some(VfsOp::Read),
            path_pattern: Some(path_pattern.into()),
            error_kind,
            error_message: None,
            probability: 1.0,
            max_injections: None,
            current_injections: 0,
        });
        self
    }

    /// Add a deterministic rule injecting an I/O error on writes matching `path_pattern`.
    pub fn with_vfs_write_error(
        mut self,
        path_pattern: impl Into<String>,
        error_kind: IoErrorKind,
    ) -> Self {
        self.vfs_rules.push(VfsFaultRule {
            op: Some(VfsOp::Write),
            path_pattern: Some(path_pattern.into()),
            error_kind,
            error_message: None,
            probability: 1.0,
            max_injections: None,
            current_injections: 0,
        });
        self
    }

    /// Add a VFS fault rule with probability and optional operation filter.
    pub fn with_vfs_error_rate(
        mut self,
        op: Option<VfsOp>,
        probability: f64,
        error_kind: IoErrorKind,
    ) -> Self {
        self.vfs_rules.push(VfsFaultRule {
            op,
            path_pattern: None,
            error_kind,
            error_message: None,
            probability: probability.clamp(0.0, 1.0),
            max_injections: None,
            current_injections: 0,
        });
        self
    }

    /// Configure simulated memory failure (OOM) at a specific scheduler step.
    pub fn with_oom_at_step(mut self, step: u64) -> Self {
        self.memory_policy = MemoryFaultPolicy::StepOom(step);
        self
    }

    /// Configure simulated memory failure when total memory allocated exceeds bytes.
    pub fn with_oom_at_bytes(mut self, bytes: usize) -> Self {
        self.memory_policy = MemoryFaultPolicy::MaxAllocBytes(bytes);
        self
    }

    /// Schedule a spontaneous fiber interrupt at a specific step.
    pub fn with_spontaneous_interrupt(
        mut self,
        fiber_id: Option<u64>,
        step: u64,
        reason: InterruptReason,
    ) -> Self {
        self.interrupt_schedules.push(InterruptSchedule {
            fiber_id,
            step,
            reason,
        });
        self
    }

    /// Evaluates VFS fault rules against the requested operation and path.
    pub fn should_fault_vfs(&mut self, op: VfsOp, path: &Path) -> Option<IoError> {
        for rule in &mut self.vfs_rules {
            if rule.matches(op, path) {
                if let Some(max) = rule.max_injections {
                    if rule.current_injections >= max {
                        continue;
                    }
                }
                let should_inject = if rule.probability >= 1.0 {
                    true
                } else {
                    self.rng.random_bool(rule.probability)
                };

                if should_inject {
                    rule.current_injections += 1;
                    self.vfs_fault_count += 1;
                    let msg = rule.error_message.clone().unwrap_or_else(|| {
                        format!("FaultInjectedVfs: Simulated error on {:?} for {}", op, path.display())
                    });
                    return Some(IoError::new(rule.error_kind, msg));
                }
            }
        }
        None
    }

    /// Evaluates memory fault policy.
    pub fn should_fault_memory(&mut self, step: u64, current_bytes: usize) -> bool {
        match self.memory_policy {
            MemoryFaultPolicy::None => false,
            MemoryFaultPolicy::StepOom(target_step) => {
                if step == target_step {
                    self.memory_fault_count += 1;
                    true
                } else {
                    false
                }
            }
            MemoryFaultPolicy::MaxAllocBytes(max_bytes) => {
                if current_bytes >= max_bytes {
                    self.memory_fault_count += 1;
                    true
                } else {
                    false
                }
            }
            MemoryFaultPolicy::Probability { rate } => {
                if self.rng.random_bool(rate.clamp(0.0, 1.0)) {
                    self.memory_fault_count += 1;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Evaluates scheduled spontaneous fiber interrupts for a given step.
    pub fn should_interrupt_fiber(&mut self, fiber_id: u64, step: u64) -> Option<InterruptReason> {
        for s in &self.interrupt_schedules {
            if s.step == step && s.fiber_id.map_or(true, |fid| fid == fiber_id) {
                self.interrupt_fault_count += 1;
                return Some(s.reason);
            }
        }
        None
    }

    /// Apply scheduled step faults directly to a VM instance.
    pub fn apply_step_faults(&mut self, vm: &mut VM, step: u64) -> LustResult<()> {
        // 1. Spontaneous interrupts
        for s in &self.interrupt_schedules {
            if s.step == step {
                if let Some(fid) = s.fiber_id {
                    if let Some(task) = vm.task_manager.get_mut(TaskId(fid)) {
                        task.request_interrupt(s.reason);
                        self.interrupt_fault_count += 1;
                    }
                } else if let Some(current_id) = vm.current_task {
                    if let Some(task) = vm.task_manager.get_mut(current_id) {
                        task.request_interrupt(s.reason);
                        self.interrupt_fault_count += 1;
                    }
                }
            }
        }

        // 2. Memory faults
        let mem_used = vm.memory_used_bytes();
        if self.should_fault_memory(step, mem_used) {
            // Trigger memory budget error by setting 0 bytes budget or trapping
            let _ = vm.set_memory_budget_bytes(0);
        }

        Ok(())
    }
}

/// FaultInjectedVfs: VirtualFileSystem decorator that wraps any underlying VFS
/// and injects deterministic errors per FaultConfiguration.
pub struct FaultInjectedVfs {
    pub inner: Arc<dyn VirtualFileSystem>,
    pub config: Arc<Mutex<FaultConfiguration>>,
}

impl FaultInjectedVfs {
    pub fn new(inner: Arc<dyn VirtualFileSystem>, config: Arc<Mutex<FaultConfiguration>>) -> Self {
        Self { inner, config }
    }
}

impl VirtualFileSystem for FaultInjectedVfs {
    fn read(&self, path: &Path) -> Result<Vec<u8>, IoError> {
        if let Some(err) = self.config.lock().unwrap().should_fault_vfs(VfsOp::Read, path) {
            return Err(err);
        }
        self.inner.read(path)
    }

    fn read_to_string(&self, path: &Path) -> Result<String, IoError> {
        if let Some(err) = self.config.lock().unwrap().should_fault_vfs(VfsOp::Read, path) {
            return Err(err);
        }
        self.inner.read_to_string(path)
    }

    fn write(&self, path: &Path, contents: &[u8]) -> Result<(), IoError> {
        if let Some(err) = self.config.lock().unwrap().should_fault_vfs(VfsOp::Write, path) {
            return Err(err);
        }
        self.inner.write(path, contents)
    }

    fn append(&self, path: &Path, contents: &[u8]) -> Result<(), IoError> {
        if let Some(err) = self.config.lock().unwrap().should_fault_vfs(VfsOp::Append, path) {
            return Err(err);
        }
        self.inner.append(path, contents)
    }

    fn create_dir_all(&self, path: &Path) -> Result<(), IoError> {
        if let Some(err) = self.config.lock().unwrap().should_fault_vfs(VfsOp::CreateDir, path) {
            return Err(err);
        }
        self.inner.create_dir_all(path)
    }

    fn remove_file(&self, path: &Path) -> Result<(), IoError> {
        if let Some(err) = self.config.lock().unwrap().should_fault_vfs(VfsOp::Delete, path) {
            return Err(err);
        }
        self.inner.remove_file(path)
    }

    fn remove_dir_all(&self, path: &Path) -> Result<(), IoError> {
        if let Some(err) = self.config.lock().unwrap().should_fault_vfs(VfsOp::Delete, path) {
            return Err(err);
        }
        self.inner.remove_dir_all(path)
    }

    fn rename(&self, from: &Path, to: &Path) -> Result<(), IoError> {
        if let Some(err) = self.config.lock().unwrap().should_fault_vfs(VfsOp::Write, from) {
            return Err(err);
        }
        self.inner.rename(from, to)
    }

    fn read_dir(&self, path: &Path) -> Result<Vec<PathBuf>, IoError> {
        if let Some(err) = self.config.lock().unwrap().should_fault_vfs(VfsOp::Read, path) {
            return Err(err);
        }
        self.inner.read_dir(path)
    }

    fn metadata(&self, path: &Path) -> Result<VfsMetadata, IoError> {
        if let Some(err) = self.config.lock().unwrap().should_fault_vfs(VfsOp::Metadata, path) {
            return Err(err);
        }
        self.inner.metadata(path)
    }

    fn canonicalize(&self, path: &Path) -> Result<PathBuf, IoError> {
        if let Some(err) = self.config.lock().unwrap().should_fault_vfs(VfsOp::Canonicalize, path) {
            return Err(err);
        }
        self.inner.canonicalize(path)
    }

    fn exists(&self, path: &Path) -> bool {
        if self.config.lock().unwrap().should_fault_vfs(VfsOp::Exists, path).is_some() {
            return false;
        }
        self.inner.exists(path)
    }
}

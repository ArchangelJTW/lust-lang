//! Fail-closed sandbox policy types and validation.

use crate::sandbox::SandboxError;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

/// Explicit OS isolation requirements that must never be silently downgraded.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IsolationRequirements {
    pub require_linux_cgroups_v2: bool,
    pub require_seccomp: bool,
    pub require_net_namespace: bool,
}

/// Resource limits and output bounds enforced by the sandbox supervisor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxLimits {
    pub wall_timeout: Duration,
    pub max_stdout_bytes: usize,
    pub max_stderr_bytes: usize,
    pub cpu_seconds: Option<u64>,
    pub nofile_limit: Option<u64>,
    pub fsize_bytes: Option<u64>,
    pub address_space_bytes: Option<u64>,
    pub data_bytes: Option<u64>,
}

impl Default for SandboxLimits {
    fn default() -> Self {
        Self {
            wall_timeout: Duration::from_secs(5),
            max_stdout_bytes: 64 * 1024,
            max_stderr_bytes: 64 * 1024,
            cpu_seconds: None,
            nofile_limit: None,
            fsize_bytes: None,
            address_space_bytes: None,
            data_bytes: None,
        }
    }
}

/// Full configuration for running a child process inside the sandbox.
#[derive(Debug, Clone)]
pub struct SandboxPolicy {
    pub executable: PathBuf,
    pub args: Vec<String>,
    pub allowed_env: BTreeMap<String, String>,
    pub limits: SandboxLimits,
    pub isolation: IsolationRequirements,
    pub stdin: Option<Vec<u8>>,
    pub cancel_flag: Option<Arc<AtomicBool>>,
}

impl SandboxPolicy {
    /// Creates a restrictive default policy for `executable` with an empty allowlisted environment.
    pub fn new(executable: impl Into<PathBuf>) -> Self {
        Self {
            executable: executable.into(),
            args: Vec::new(),
            allowed_env: BTreeMap::new(),
            limits: SandboxLimits::default(),
            isolation: IsolationRequirements::default(),
            stdin: None,
            cancel_flag: None,
        }
    }

    /// Inserts an environment variable into the explicit allowlist.
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.allowed_env.insert(key.into(), value.into());
        self
    }

    /// Validates the policy and fails closed on invalid limits or unsupported isolation.
    pub fn validate(&self) -> Result<(), SandboxError> {
        if self.limits.wall_timeout.is_zero() {
            return Err(SandboxError::InvalidPolicy(
                "wall_timeout must be greater than zero".into(),
            ));
        }
        if self.limits.max_stdout_bytes == 0 {
            return Err(SandboxError::InvalidPolicy(
                "max_stdout_bytes must be greater than zero".into(),
            ));
        }
        if self.limits.max_stderr_bytes == 0 {
            return Err(SandboxError::InvalidPolicy(
                "max_stderr_bytes must be greater than zero".into(),
            ));
        }
        if self.executable.as_os_str().is_empty() || !self.executable.is_file() {
            return Err(SandboxError::InvalidPolicy(format!(
                "executable does not exist or is not a file: {}",
                self.executable.display()
            )));
        }

        if self.isolation.require_linux_cgroups_v2 {
            return Err(SandboxError::UnsupportedIsolation(
                "Linux cgroups v2 isolation is unavailable on this baseline".into(),
            ));
        }
        if self.isolation.require_seccomp {
            return Err(SandboxError::UnsupportedIsolation(
                "Linux seccomp-bpf isolation is unavailable on this baseline".into(),
            ));
        }
        if self.isolation.require_net_namespace {
            return Err(SandboxError::UnsupportedIsolation(
                "Linux network namespace isolation is unavailable on this baseline".into(),
            ));
        }

        Ok(())
    }
}

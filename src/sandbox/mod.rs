//! OS-level process sandbox and watchdog supervisor (`std` only).
//!
//! Provides a Darwin POSIX baseline (`setpgid(0, 0)`, `setrlimit`, descriptor
//! sanitization, sanitized environment, private temporary working directory,
//! byte-capped output draining, and `killpg(SIGKILL)` monotonic watchdog).
//! Linux-specific isolation features (`cgroups v2`, `seccomp-bpf`, network
//! namespaces) fail closed with [`SandboxError::UnsupportedIsolation`] on
//! non-Linux targets rather than silently downgrading.

pub mod darwin;
pub mod policy;
pub mod supervisor;

pub use darwin::{RlimitEntry, RlimitReport};
pub use policy::{IsolationRequirements, SandboxLimits, SandboxPolicy};
pub use supervisor::{SandboxOutcome, run_sandboxed};

use std::fmt;
use std::io;

/// Errors returned by sandbox policy validation, setup, or supervision.
#[derive(Debug)]
pub enum SandboxError {
    /// Policy configuration is invalid (e.g. zero timeout, zero output cap, missing executable).
    InvalidPolicy(String),
    /// Requested isolation primitive is unsupported on this host OS (fail-closed).
    UnsupportedIsolation(String),
    /// Failed to create or manage the private temporary working directory.
    TempDir(io::Error),
    /// Failed during POSIX pre-exec / spawn / rlimit setup.
    Spawn(io::Error),
    /// Failed while supervising or terminating the child process group.
    Supervisor(String),
}

impl fmt::Display for SandboxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPolicy(msg) => write!(f, "invalid sandbox policy: {msg}"),
            Self::UnsupportedIsolation(msg) => write!(f, "unsupported isolation: {msg}"),
            Self::TempDir(err) => write!(f, "sandbox tempdir error: {err}"),
            Self::Spawn(err) => write!(f, "sandbox spawn error: {err}"),
            Self::Supervisor(msg) => write!(f, "sandbox supervisor error: {msg}"),
        }
    }
}

impl std::error::Error for SandboxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TempDir(err) | Self::Spawn(err) => Some(err),
            _ => None,
        }
    }
}

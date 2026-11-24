//! Gate-SCHEMA-13 Declarative Semantic FFI Boundary & Isolation Runtime (`TASK_ALLOWS_UNSAFE=0`).
//!
//! Implements RFC Section 16.4 & Section 10 semantic boundary verification:
//! - Fail-closed manifest validation (overlapping mappings, incomplete ownership, invalid errno,
//!   unsupported varargs/unregistered callbacks, missing isolation for untrusted workers).
//! - Capability & library constructor gating via `crate::config::CapabilityMask` (`denied == 0`,
//!   `permitted > 0`).
//! - Owner/tenant/generation/ABI descriptor validation via `crate::data::owner::OwnerRegistry`.
//! - Exact outcome mapping into `crate::effect_abi::{Exit, Cause, InterruptReason}`:
//!   - Declared success + checked payload (including negative numbers & `i64::MIN`) -> `Exit::Success`
//!   - Declared recoverable errors / immediate errno / uncancelled EINTR -> `Exit::Failure(Cause::Fail)`
//!   - Acknowledged cancellation with cancel token -> `Exit::Failure(Cause::Interrupted(Cancelled))`
//!   - Invalid-success (null out), unknown status, unrequested cancel -> `Exit::Failure(Cause::Die)`
//! - Isolated child execution with external safety watchdog (`WatchdogKilledFail == FAIL`).
//! - Explicit mutations (`SkipCapabilityCheck`, `MisclassifyEintrAsInterrupt`, `LeakOwnerOnFailure`)
//!   for Gate-SCHEMA-13 negative mutant verification.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::config::CapabilityMask;
use crate::data::owner::{LeaseError, OwnerId, OwnerRegistry};
use crate::effect_abi::{Cause, Exit, InterruptReason};
use crate::w17_layout::watchdog::{
    WatchdogOutcome, run_child_with_watchdog, terminate_and_reap_worker,
};

/// Canonical ABI version constant matching `LUST_ABI_VERSION_V1`.
pub const FFI_ABI_VERSION_V1: u32 = 1;
/// Canonical descriptor byte size matching `sizeof(LustStructBindingInfoV1)`.
pub const FFI_DESCRIPTOR_SIZE_V1: u32 = 64;

/// Ownership contract for a native FFI parameter or output slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiOwnership {
    BorrowedUntilReturn,
    OwnedOnSuccessWithFree,
    IncompleteUnspecified,
}

/// Execution containment mode declared in an FFI wrapper manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiExecutionMode {
    NonblockingTrustedInProcess,
    IsolatedWorker,
    UnsupportedDirectCallback,
}

/// Semantic category for a declared native integer status code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiStatusKind {
    Success { requires_non_null_out: bool },
    RecoverableNotFound,
    RecoverableIoWithErrno,
    RecoverableEintr,
    RecoverablePartialInitFailed,
    CancelledRequiresToken,
}

/// Single status mapping rule in an `FfiManifest`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FfiStatusRule {
    pub raw_status: i32,
    pub kind: FfiStatusKind,
    pub uses_errno: bool,
}

/// Declarative FFI wrapper manifest (RFC Section 16.4).
#[derive(Debug, Clone)]
pub struct FfiManifest {
    pub library_id: String,
    pub symbol: String,
    pub abi: String,
    pub required_capability: String,
    pub input_ownership: FfiOwnership,
    pub output_ownership: FfiOwnership,
    pub execution: FfiExecutionMode,
    pub allows_varargs: bool,
    pub untrusted_or_blocking: bool,
    pub rules: Vec<FfiStatusRule>,
}

/// Fail-closed manifest validation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FfiManifestError {
    OverlappingStatusCode(i32),
    IncompleteOwnership(&'static str),
    UnsupportedAbi(String),
    VarargsRejected,
    UnregisteredCallbackRejected,
    ErrnoDisallowedForStatus(i32),
    MissingIsolationForUntrustedWorker,
    MissingSuccessRule,
}

impl fmt::Display for FfiManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for FfiManifestError {}

impl FfiManifest {
    /// Canonical `pinned_store_v1::store_read` manifest matching RFC Section 16.4.
    pub fn store_read_v1() -> Self {
        Self {
            library_id: "pinned_store_v1".to_string(),
            symbol: "store_read".to_string(),
            abi: "cdecl_for_target".to_string(),
            required_capability: "Store.read".to_string(),
            input_ownership: FfiOwnership::BorrowedUntilReturn,
            output_ownership: FfiOwnership::OwnedOnSuccessWithFree,
            execution: FfiExecutionMode::IsolatedWorker,
            allows_varargs: false,
            untrusted_or_blocking: true,
            rules: vec![
                FfiStatusRule {
                    raw_status: 0,
                    kind: FfiStatusKind::Success {
                        requires_non_null_out: true,
                    },
                    uses_errno: false,
                },
                FfiStatusRule {
                    raw_status: 10,
                    kind: FfiStatusKind::RecoverableNotFound,
                    uses_errno: false,
                },
                FfiStatusRule {
                    raw_status: 11,
                    kind: FfiStatusKind::RecoverableIoWithErrno,
                    uses_errno: true,
                },
                FfiStatusRule {
                    raw_status: 12,
                    kind: FfiStatusKind::RecoverableEintr,
                    uses_errno: true,
                },
                FfiStatusRule {
                    raw_status: 13,
                    kind: FfiStatusKind::CancelledRequiresToken,
                    uses_errno: false,
                },
                FfiStatusRule {
                    raw_status: 14,
                    kind: FfiStatusKind::RecoverablePartialInitFailed,
                    uses_errno: false,
                },
            ],
        }
    }

    /// Validates all manifest invariants fail-closed before loading any library or dispatching.
    pub fn validate(&self) -> Result<(), FfiManifestError> {
        if self.abi != "cdecl_for_target" && self.abi != "C" {
            return Err(FfiManifestError::UnsupportedAbi(self.abi.clone()));
        }
        if self.allows_varargs {
            return Err(FfiManifestError::VarargsRejected);
        }
        if self.execution == FfiExecutionMode::UnsupportedDirectCallback {
            return Err(FfiManifestError::UnregisteredCallbackRejected);
        }
        if self.untrusted_or_blocking && self.execution != FfiExecutionMode::IsolatedWorker {
            return Err(FfiManifestError::MissingIsolationForUntrustedWorker);
        }
        if self.input_ownership == FfiOwnership::IncompleteUnspecified {
            return Err(FfiManifestError::IncompleteOwnership("input"));
        }
        if self.output_ownership == FfiOwnership::IncompleteUnspecified {
            return Err(FfiManifestError::IncompleteOwnership("output"));
        }

        let mut seen_statuses = BTreeSet::new();
        let mut has_success = false;
        for rule in &self.rules {
            if !seen_statuses.insert(rule.raw_status) {
                return Err(FfiManifestError::OverlappingStatusCode(rule.raw_status));
            }
            match rule.kind {
                FfiStatusKind::Success { .. } => {
                    has_success = true;
                    if rule.uses_errno {
                        return Err(FfiManifestError::ErrnoDisallowedForStatus(rule.raw_status));
                    }
                }
                FfiStatusKind::RecoverableNotFound
                | FfiStatusKind::RecoverablePartialInitFailed
                | FfiStatusKind::CancelledRequiresToken => {
                    if rule.uses_errno {
                        return Err(FfiManifestError::ErrnoDisallowedForStatus(rule.raw_status));
                    }
                }
                FfiStatusKind::RecoverableIoWithErrno | FfiStatusKind::RecoverableEintr => {}
            }
        }
        if !has_success {
            return Err(FfiManifestError::MissingSuccessRule);
        }
        Ok(())
    }
}

/// Versioned ABI descriptor validated before every native dispatch (RFC Section 10 & 10.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FfiAbiDescriptor {
    pub abi_version: u32,
    pub struct_size: u32,
    pub flags: u64,
    pub tenant_id: u32,
    pub owner: OwnerId,
}

impl FfiAbiDescriptor {
    pub fn valid(tenant_id: u32, owner: OwnerId) -> Self {
        Self {
            abi_version: FFI_ABI_VERSION_V1,
            struct_size: FFI_DESCRIPTOR_SIZE_V1,
            flags: 0x0F,
            tenant_id,
            owner,
        }
    }
}

/// Typed recoverable domain errors mapped from native FFI status codes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeCallError {
    CapabilityDenied(String),
    InvalidAbiVersion { expected: u32, actual: u32 },
    ShortDescriptorSize { required: u32, actual: u32 },
    InvalidOwnerLease(LeaseError),
    NotFound,
    NativeIo { errno: i32 },
    InterruptedSysCall { errno: i32 },
    PartialInitFailed { cleaned_owners: u64 },
}

/// Explicit mutant selector for Gate-SCHEMA-13 negative mutation tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiMutationMode {
    None,
    /// Mutant 1: skips capability enforcement (violates denied constructor counter == 0).
    SkipCapabilityCheck,
    /// Mutant 2: misclassifies uncancelled EINTR as `Cause::Interrupted` instead of `Cause::Fail`.
    MisclassifyEintrAsInterrupt,
    /// Mutant 3: skips owner release/cleanup on failure (violates allocs == frees / leak-free).
    LeakOwnerOnFailure,
}

/// Complete evidence captured from an isolated FFI boundary invocation.
#[derive(Debug, Clone)]
pub struct FfiCallEvidence {
    pub mode: String,
    pub exit: Exit<i64, NativeCallError>,
    pub raw_status: Option<i32>,
    pub captured_errno: i32,
    pub constructor_runs_observed: u64,
    pub native_entries_observed: u64,
    pub owner_allocs: u64,
    pub owner_frees: u64,
    pub cleanup_order_checksum: u64,
    pub callbacks_ran: u64,
    pub post_unreg_callbacks: u64,
    pub watchdog_killed: bool,
    pub stdout: String,
    pub stderr: String,
}

impl FfiCallEvidence {
    pub fn owner_leaks(&self) -> u64 {
        self.owner_allocs.saturating_sub(self.owner_frees)
    }
}

/// Evidence from testing isolated worker containment (`terminate + reap`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerContainmentReport {
    pub pid: u32,
    pub readiness_line: String,
    pub terminated_by_manager: bool,
    pub acknowledged: bool,
    pub reaped: bool,
    pub killed_by_outer_watchdog: bool,
    pub termination_signal: Option<i32>,
    pub elapsed_ms: u128,
}

/// Gate-SCHEMA-13 FFI boundary coordinator enforcing capabilities, owner leases, ABI version,
/// and isolated worker dispatch.
pub struct FfiBoundaryRuntime {
    pub runtime_id: u32,
    pub manifest: FfiManifest,
    pub owners: OwnerRegistry,
    pub fixture_binary: PathBuf,
    pub total_native_entries: u64,
    pub total_constructor_runs: u64,
    pub active_native_owners: u64,
}

impl FfiBoundaryRuntime {
    /// RFC Section 16.4 & Gate-SCHEMA-13 honesty invariant:
    /// Trusted in-process FFI is NEVER claimed to be sandboxed.
    pub const fn in_process_ffi_is_sandboxed() -> bool {
        false
    }

    pub fn new(
        runtime_id: u32,
        manifest: FfiManifest,
        fixture_binary: PathBuf,
    ) -> Result<Self, FfiManifestError> {
        manifest.validate()?;
        Ok(Self {
            runtime_id,
            manifest,
            owners: OwnerRegistry::new(runtime_id),
            fixture_binary,
            total_native_entries: 0,
            total_constructor_runs: 0,
            active_native_owners: 0,
        })
    }

    pub fn allocate_owner(&mut self, tenant_id: u32) -> OwnerId {
        self.owners.allocate(tenant_id)
    }

    pub fn revoke_owner(&mut self, owner: OwnerId) -> Result<(), LeaseError> {
        self.owners.cancel(owner)?;
        self.owners.acknowledge_join(owner)?;
        self.owners.release(owner)
    }

    fn check_capabilities(&self, caps: &CapabilityMask) -> Result<(), NativeCallError> {
        if !caps.allow_ffi {
            return Err(NativeCallError::CapabilityDenied(
                "allow_ffi=false".to_string(),
            ));
        }
        if !caps
            .custom_capabilities
            .contains(&self.manifest.required_capability)
        {
            return Err(NativeCallError::CapabilityDenied(format!(
                "missing custom capability `{}`",
                self.manifest.required_capability
            )));
        }
        Ok(())
    }

    fn check_descriptor(
        &self,
        desc: &FfiAbiDescriptor,
        current_tenant: u32,
    ) -> Result<(), NativeCallError> {
        if desc.abi_version != FFI_ABI_VERSION_V1 {
            return Err(NativeCallError::InvalidAbiVersion {
                expected: FFI_ABI_VERSION_V1,
                actual: desc.abi_version,
            });
        }
        if desc.struct_size < FFI_DESCRIPTOR_SIZE_V1 {
            return Err(NativeCallError::ShortDescriptorSize {
                required: FFI_DESCRIPTOR_SIZE_V1,
                actual: desc.struct_size,
            });
        }
        if desc.tenant_id != current_tenant {
            return Err(NativeCallError::InvalidOwnerLease(
                LeaseError::TenantMismatch {
                    expected: current_tenant,
                    actual: desc.tenant_id,
                },
            ));
        }
        self.owners
            .validate_access(desc.owner, current_tenant)
            .map_err(NativeCallError::InvalidOwnerLease)
    }

    /// Invokes the isolated C fixture for `mode` after validating capability, ABI descriptor,
    /// and owner lease. Records constructor and native-entry counters via an external log file
    /// written only by the C `__attribute__((constructor))`.
    pub fn invoke_isolated(
        &mut self,
        mode: &str,
        request_cancelled: bool,
        caps: &CapabilityMask,
        desc: &FfiAbiDescriptor,
        current_tenant: u32,
        watchdog_timeout: Duration,
        mutation: FfiMutationMode,
    ) -> Result<FfiCallEvidence, String> {
        let tmp = tempfile::tempdir().map_err(|e| format!("ctor log tempdir: {e}"))?;
        let ctor_log_path = tmp.path().join("ctor.log");

        // 1. Capability check (before spawning/loading C binary).
        if mutation != FfiMutationMode::SkipCapabilityCheck {
            if let Err(err) = self.check_capabilities(caps) {
                let ctor_runs = count_ctor_log_lines(&ctor_log_path);
                return Ok(FfiCallEvidence {
                    mode: mode.to_string(),
                    exit: Exit::Failure(Cause::Fail(err)),
                    raw_status: None,
                    captured_errno: 0,
                    constructor_runs_observed: ctor_runs,
                    native_entries_observed: 0,
                    owner_allocs: 0,
                    owner_frees: 0,
                    cleanup_order_checksum: 0,
                    callbacks_ran: 0,
                    post_unreg_callbacks: 0,
                    watchdog_killed: false,
                    stdout: String::new(),
                    stderr: String::new(),
                });
            }
        }

        // 2. ABI version, struct size, tenant, and generation-checked owner lease check.
        if let Err(err) = self.check_descriptor(desc, current_tenant) {
            let ctor_runs = count_ctor_log_lines(&ctor_log_path);
            return Ok(FfiCallEvidence {
                mode: mode.to_string(),
                exit: Exit::Failure(Cause::Fail(err)),
                raw_status: None,
                captured_errno: 0,
                constructor_runs_observed: ctor_runs,
                native_entries_observed: 0,
                owner_allocs: 0,
                owner_frees: 0,
                cleanup_order_checksum: 0,
                callbacks_ran: 0,
                post_unreg_callbacks: 0,
                watchdog_killed: false,
                stdout: String::new(),
                stderr: String::new(),
            });
        }

        // 3. Dispatch into isolated child process under external safety watchdog.
        let effective_mode = if mutation == FfiMutationMode::LeakOwnerOnFailure {
            "mutant_leaked_owner"
        } else {
            mode
        };
        let cancel_arg = if request_cancelled { "1" } else { "0" };
        let ctor_env = ctor_log_path.display().to_string();
        let run = run_child_with_watchdog(
            &self.fixture_binary,
            &[effective_mode, cancel_arg],
            &[("LUST_W17_CTOR_LOG", &ctor_env)],
            watchdog_timeout,
        )?;

        let ctor_runs_from_file = count_ctor_log_lines(&ctor_log_path);
        self.total_constructor_runs += ctor_runs_from_file;

        if let WatchdogOutcome::WatchdogKilledFail { pid, elapsed_ms, .. } = run.outcome {
            return Ok(FfiCallEvidence {
                mode: mode.to_string(),
                exit: Exit::Failure(Cause::Die(format!(
                    "WATCHDOG_KILLED_FAIL: child PID {pid} exceeded {elapsed_ms}ms"
                ))),
                raw_status: None,
                captured_errno: 0,
                constructor_runs_observed: ctor_runs_from_file,
                native_entries_observed: 0,
                owner_allocs: 0,
                owner_frees: 0,
                cleanup_order_checksum: 0,
                callbacks_ran: 0,
                post_unreg_callbacks: 0,
                watchdog_killed: true,
                stdout: run.stdout,
                stderr: run.stderr,
            });
        }

        if !run.outcome.is_clean_success() {
            return Ok(FfiCallEvidence {
                mode: mode.to_string(),
                exit: Exit::Failure(Cause::Die(format!(
                    "isolated C fixture exited abnormally: {:?}, stderr={}",
                    run.outcome, run.stderr
                ))),
                raw_status: None,
                captured_errno: 0,
                constructor_runs_observed: ctor_runs_from_file,
                native_entries_observed: 0,
                owner_allocs: 0,
                owner_frees: 0,
                cleanup_order_checksum: 0,
                callbacks_ran: 0,
                post_unreg_callbacks: 0,
                watchdog_killed: false,
                stdout: run.stdout,
                stderr: run.stderr,
            });
        }

        let val: serde_json::Value = serde_json::from_str(run.stdout.trim())
            .map_err(|e| format!("parse fixture json `{}`: {e}", run.stdout))?;

        let raw_status = val["raw_status"].as_i64().unwrap_or(777) as i32;
        let has_payload = val["has_payload"].as_bool().unwrap_or(false);
        let payload = val["payload"].as_i64().unwrap_or(0);
        let out_ptr_non_null = val["out_ptr_non_null"].as_bool().unwrap_or(false);
        let raw_errno = val["captured_errno"].as_i64().unwrap_or(0) as i32;
        let native_entries = val["native_entries"].as_u64().unwrap_or(0);
        let owner_allocs = val["owner_allocs"].as_u64().unwrap_or(0);
        let owner_frees = val["owner_frees"].as_u64().unwrap_or(0);
        let cleanup_order_checksum = val["cleanup_order_checksum"].as_u64().unwrap_or(0);
        let callbacks_ran = val["callbacks_ran"].as_u64().unwrap_or(0);
        let post_unreg_callbacks = val["post_unreg_callbacks"].as_u64().unwrap_or(0);

        self.total_native_entries += native_entries;
        self.active_native_owners = self
            .active_native_owners
            .saturating_add(owner_allocs.saturating_sub(owner_frees));

        // 4. Semantic status & errno mapping per manifest rules (RFC Section 16.4).
        let rule = self
            .manifest
            .rules
            .iter()
            .find(|r| r.raw_status == raw_status)
            .copied();

        let (exit, effective_errno) = match rule {
            None => (
                Exit::Failure(Cause::Die(format!("UnexpectedNativeStatus({raw_status})"))),
                0,
            ),
            Some(r) => {
                let err_val = if r.uses_errno { raw_errno } else { 0 };
                let mapped = match r.kind {
                    FfiStatusKind::Success {
                        requires_non_null_out,
                    } => {
                        if requires_non_null_out && (!out_ptr_non_null || !has_payload) {
                            Exit::Failure(Cause::Die(
                                "InvalidSuccessPayload: status==0 with NULL output pointer"
                                    .to_string(),
                            ))
                        } else {
                            Exit::Success(payload)
                        }
                    }
                    FfiStatusKind::RecoverableNotFound => {
                        Exit::Failure(Cause::Fail(NativeCallError::NotFound))
                    }
                    FfiStatusKind::RecoverableIoWithErrno => {
                        Exit::Failure(Cause::Fail(NativeCallError::NativeIo { errno: err_val }))
                    }
                    FfiStatusKind::RecoverableEintr => {
                        if mutation == FfiMutationMode::MisclassifyEintrAsInterrupt {
                            // Intentional Gate-13 Mutant 2: misclassify uncancelled EINTR as Interrupt.
                            Exit::Failure(Cause::Interrupted(InterruptReason::Cancelled))
                        } else {
                            Exit::Failure(Cause::Fail(NativeCallError::InterruptedSysCall {
                                errno: err_val,
                            }))
                        }
                    }
                    FfiStatusKind::RecoverablePartialInitFailed => {
                        Exit::Failure(Cause::Fail(NativeCallError::PartialInitFailed {
                            cleaned_owners: owner_frees,
                        }))
                    }
                    FfiStatusKind::CancelledRequiresToken => {
                        if request_cancelled {
                            Exit::Failure(Cause::Interrupted(InterruptReason::Cancelled))
                        } else {
                            Exit::Failure(Cause::Die(
                                "ProtocolDefect: STORE_CANCELLED without guest cancellation token"
                                    .to_string(),
                            ))
                        }
                    }
                };
                (mapped, err_val)
            }
        };

        Ok(FfiCallEvidence {
            mode: mode.to_string(),
            exit,
            raw_status: Some(raw_status),
            captured_errno: effective_errno,
            constructor_runs_observed: ctor_runs_from_file,
            native_entries_observed: native_entries,
            owner_allocs,
            owner_frees,
            cleanup_order_checksum,
            callbacks_ran,
            post_unreg_callbacks,
            watchdog_killed: false,
            stdout: run.stdout,
            stderr: run.stderr,
        })
    }

    /// Runs an unbounded blocking C worker inside an isolated child process and enforces
    /// explicit `terminate + reap` containment before `outer_watchdog_timeout`.
    pub fn run_unbounded_worker_containment(
        &self,
        caps: &CapabilityMask,
        desc: &FfiAbiDescriptor,
        current_tenant: u32,
        operation_deadline: Duration,
        reap_grace: Duration,
        outer_watchdog_timeout: Duration,
    ) -> Result<WorkerContainmentReport, String> {
        self.check_capabilities(caps)
            .map_err(|e| format!("capability check failed: {e:?}"))?;
        self.check_descriptor(desc, current_tenant)
            .map_err(|e| format!("descriptor check failed: {e:?}"))?;

        let term = terminate_and_reap_worker(
            &self.fixture_binary,
            &["unbounded_worker"],
            &[],
            operation_deadline,
            reap_grace,
            outer_watchdog_timeout,
        )?;

        Ok(WorkerContainmentReport {
            pid: term.pid,
            readiness_line: term.readiness_line,
            terminated_by_manager: term.terminated_by_manager,
            acknowledged: term.acknowledged,
            reaped: term.reaped,
            killed_by_outer_watchdog: term.killed_by_outer_watchdog,
            termination_signal: term.termination_signal,
            elapsed_ms: term.elapsed_ms,
        })
    }
}

fn count_ctor_log_lines(path: &Path) -> u64 {
    fs::read_to_string(path)
        .map(|s| s.lines().filter(|l| l.contains("CONSTRUCTOR_RUN=")).count() as u64)
        .unwrap_or(0)
}

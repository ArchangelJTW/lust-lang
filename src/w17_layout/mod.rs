//! Wave 17 Track NG-W17-3: Layout / C FFI Verification & Semantic Boundary Runtime
//! (D-LAYOUT Gate-SCHEMA-09 & Gate-SCHEMA-13).
//!
//! Strictly safe Rust (`TASK_ALLOWS_UNSAFE=0`). All C ABI probes, sanitizer checks,
//! C callbacks, and unbounded worker containment execute in isolated child processes
//! supervised by an external watchdog.
//!
//! 17C Registration Proposal (`NG-W17-9`):
//! - module: `src/w17_layout/mod.rs`
//! - parent: `src/lib.rs`
//! - registration: `#[cfg(feature = "std")] pub mod w17_layout;`

#![forbid(unsafe_code)]

pub mod c_abi;
pub mod ffi_boundary;
pub mod model_contract;
pub mod watchdog;

pub use c_abi::{
    CBuildEvidence, CLayoutReport, CStructLayoutEntry, DisassemblyAudit, ZigLayoutEvidence,
    audit_c_accessor_disassembly, build_c_fixture_with_sanitizers, compare_layout_with_c_entry,
    compile_c_negative_layout_mutant, verify_zig_layout_probe,
};
pub use ffi_boundary::{
    FfiAbiDescriptor, FfiBoundaryRuntime, FfiCallEvidence, FfiExecutionMode, FfiManifest,
    FfiManifestError, FfiMutationMode, FfiOwnership, FfiStatusKind, FfiStatusRule,
    NativeCallError, WorkerContainmentReport,
};
pub use model_contract::{
    DbOperation, DbWritePlan, GeneratedIdPolicy, InsertInput, ModelContract, ModelContractError,
    ModelLifecycleHooks, PatchField, PatchInput, RecordedDbWrite, RecordingDbAdapter,
    StructBindingPlan, WireAliasMap, decode_json_with_plan,
};
pub use watchdog::{WatchdogOutcome, WatchdogRunResult, run_child_with_watchdog, terminate_and_reap_worker};

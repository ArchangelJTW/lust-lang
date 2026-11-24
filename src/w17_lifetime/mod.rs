//! Wave 17 Track NG-W17-2 (`D-LIFE`): Static Borrow / Lifetime Checker,
//! Runtime Weak-VM / Generation Invalidation, Destruction & Cross-Thread Transfer Rules,
//! and ADR-003 Isolated Allocation Oracle.
//!
//! Strictly safe Rust (`#![forbid(unsafe_code)]`, `TASK_ALLOWS_UNSAFE=0`).
//!
//! 17C Registration Proposal (`NG-W17-9`):
//! module: src/w17_lifetime/mod.rs
//! parent: src/lib.rs
//! registration: #[cfg(feature = "std")] pub mod w17_lifetime;

#![forbid(unsafe_code)]

pub mod alloc_oracle;
pub mod generation;
pub mod script_bridge;
pub mod transfer;

pub use alloc_oracle::{
    AllocCounterReport, AllocOracleEvidence, build_dyld_alloc_interposer,
    emit_alloc_interval_begin, emit_alloc_interval_end, run_isolated_alloc_child,
};
pub use generation::{
    GenArrayHandle, GenFunctionHandle, GenMapHandle, GenStructHandle, LifecycleError, LifetimeVm,
    PreparedScalarHandle, ScopedViewGuard,
};
pub use script_bridge::{declare_view_externs, deep_clone_value, install_view_runtime};
pub use transfer::{
    CompileFailResult, OwnedTransferPayload, TransferTaskHandle, deep_materialize_to_data_value,
    materialize_array_for_transfer, materialize_map_for_transfer, materialize_struct_for_transfer,
    verify_compile_fail_not_send,
};

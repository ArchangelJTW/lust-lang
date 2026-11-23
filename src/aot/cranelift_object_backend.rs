//! Conditional / offline-blocked Cranelift AOT object-file adapter (`NG-W13-2`, `F-11`).
//!
//! Because `cranelift-object` is not present in the offline Cargo cache and `object = "0.40.0"`
//! is compiled only with `features = ["std", "read_core", "read"]` (without `write`),
//! emitting raw Mach-O / ELF object files is explicitly `BLOCKED` offline.
//!
//! Per `F-11`, this module MUST NOT hand-roll a binary Mach-O or ELF encoder. Instead it:
//! 1. Validates that 100% of MIR functions in the module compile cleanly to CLIF via
//!    [`crate::mir::cranelift_backend::CraneliftJitCompiler`].
//! 2. Returns a deterministic diagnostic error directing callers to `--backend rust` until
//!    `cranelift-object` or `object/write` is added to the offline cache.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::error::{LustError, Result};
use crate::mir::cranelift_backend::CraneliftJitCompiler;
use crate::mir::ir::{MirModule, Terminator};
use crate::mir::opt::MirOptimizer;

/// Canonical diagnostic message returned when `--backend cranelift-aot` is requested offline.
pub const CRANELIFT_OBJECT_BLOCKED_REASON: &str = "cranelift-aot object emission is BLOCKED offline: cranelift-object / object::write not in offline cargo cache; use --backend rust";

/// Canonical status label recorded in multiarch / AOT capability reports.
pub const CRANELIFT_OBJECT_BACKEND_STATUS: &str =
    "BLOCKED (missing cranelift-object / object::write in offline cache)";

/// Result of validating a [`MirModule`] through [`CraneliftJitCompiler`] prior to object emission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CraneliftClifValidationReport {
    pub module_name: String,
    pub validated_functions: Vec<String>,
    pub target_triple: Option<String>,
    pub object_emission_status: &'static str,
}

fn canonicalize_module_callees(module: &mut MirModule) {
    let exact_names: HashSet<String> = module
        .functions
        .iter()
        .map(|f| f.signature.name.clone())
        .collect();
    let mut suffix_map: HashMap<String, Option<String>> = HashMap::new();
    for name in &exact_names {
        if let Some((_, short)) = name.rsplit_once('.') {
            suffix_map
                .entry(short.to_string())
                .and_modify(|slot| *slot = None)
                .or_insert_with(|| Some(name.clone()));
        }
    }
    for func in &mut module.functions {
        for block in &mut func.blocks {
            if let Some(Terminator::Call { callee, .. }) = &mut block.terminator
                && !exact_names.contains(callee.as_str())
                && let Some(Some(qualified)) = suffix_map.get(callee.as_str())
            {
                *callee = qualified.clone();
            }
        }
    }
}

/// Validates that every function in `module` lowers and compiles cleanly through
/// [`CraneliftJitCompiler`] (100% CLIF compilability check, zero object emission).
pub fn validate_mir_module_clif(
    module: &MirModule,
    target_triple: Option<&str>,
) -> Result<CraneliftClifValidationReport> {
    let mut opt_module = module.clone();
    canonicalize_module_callees(&mut opt_module);
    MirOptimizer::optimize_module(&mut opt_module);

    let mut compiler = CraneliftJitCompiler::new().map_err(|err| {
        LustError::CompileError(format!(
            "Failed to initialize Cranelift CLIF validator: {err}"
        ))
    })?;

    let compiled = compiler.compile_module(&opt_module).map_err(|err| {
        LustError::CompileError(format!("Cranelift CLIF validation failed: {err}"))
    })?;

    let mut validated_functions = Vec::with_capacity(opt_module.functions.len());
    for func in &opt_module.functions {
        let fn_name = &func.signature.name;
        match compiled.get(fn_name) {
            Some(ptr) if !ptr.is_null() => validated_functions.push(fn_name.clone()),
            _ => {
                return Err(LustError::CompileError(format!(
                    "Cranelift CLIF validation missed function '{fn_name}'"
                )));
            }
        }
    }

    Ok(CraneliftClifValidationReport {
        module_name: opt_module.name,
        validated_functions,
        target_triple: target_triple.map(ToString::to_string),
        object_emission_status: CRANELIFT_OBJECT_BACKEND_STATUS,
    })
}

/// Conditional `--backend cranelift-aot` entrypoint (`F-11`).
///
/// Validates 100% CLIF compilability of `module`, then fails closed with
/// [`CRANELIFT_OBJECT_BLOCKED_REASON`] because `cranelift-object` / `object::write` is not
/// available in the offline Cargo cache. Never emits or fabricates an object file.
pub fn compile_mir_to_object_file(
    module: &MirModule,
    _output_obj: &Path,
    target_triple: Option<&str>,
) -> Result<PathBuf> {
    let _report = validate_mir_module_clif(module, target_triple)?;
    Err(LustError::CompileError(
        CRANELIFT_OBJECT_BLOCKED_REASON.to_string(),
    ))
}

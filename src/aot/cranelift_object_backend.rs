//! Cranelift AOT object-file backend (`NG-W13-2`, `F-11`, `NG-W17-4` / `W16-CRANELIFT-OBJ`).
//!
//! - Without `--features cranelift-aot`: preserves the exact offline-blocked `F-11` contract,
//!   validating 100% CLIF compilability through [`CraneliftJitCompiler`] and failing closed with
//!   [`CRANELIFT_OBJECT_BLOCKED_REASON`] without creating any file on disk.
//! - With `--features cranelift-aot`: lowers MIR through the module-generic
//!   [`crate::mir::cranelift_backend::CraneliftObjectCompiler`] backed by
//!   `cranelift_object::ObjectBuilder` / `cranelift_object::ObjectModule` and emits a real
//!   Mach-O or ELF `.o` object file via the upstream `object` writer.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::error::{LustError, Result};
#[cfg(all(feature = "cranelift", not(feature = "cranelift-aot")))]
use crate::mir::cranelift_backend::CraneliftJitCompiler;
#[cfg(feature = "cranelift-aot")]
use crate::mir::cranelift_backend::CraneliftObjectCompiler;
use crate::mir::ir::{MirModule, MirStructLayout, Terminator};
use crate::mir::opt::{MirOptimizer, ScalarReplacementOfAggregates};
use crate::mir::verifier::MirVerifier;

/// Canonical diagnostic message returned when `--backend cranelift-aot` is requested without
/// the `cranelift-aot` feature enabled.
pub const CRANELIFT_OBJECT_BLOCKED_REASON: &str = "cranelift-aot object emission is BLOCKED offline: cranelift-object / object::write not in offline cargo cache; use --backend rust";

/// Canonical status label recorded in multiarch / AOT capability reports.
#[cfg(not(feature = "cranelift-aot"))]
pub const CRANELIFT_OBJECT_BACKEND_STATUS: &str =
    "BLOCKED (missing cranelift-object / object::write in offline cache)";

/// Canonical status label recorded when `--features cranelift-aot` is enabled.
#[cfg(feature = "cranelift-aot")]
pub const CRANELIFT_OBJECT_BACKEND_STATUS: &str =
    "ENABLED (cranelift-object Mach-O / ELF emission)";

/// Result of validating a [`MirModule`] prior to object emission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CraneliftClifValidationReport {
    pub module_name: String,
    pub validated_functions: Vec<String>,
    pub target_triple: Option<String>,
    pub object_emission_status: &'static str,
}

/// Detailed report produced by real `cranelift-object` emission under `--features cranelift-aot`.
#[cfg(feature = "cranelift-aot")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CraneliftObjectEmissionReport {
    pub output_path: PathBuf,
    pub module_name: String,
    pub target_triple: String,
    pub binary_format: String,
    pub architecture: String,
    pub exported_functions: Vec<String>,
    pub imported_runtime_symbols: Vec<String>,
    pub object_size_bytes: usize,
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

fn optimize_module_for_cranelift_aot(module: &mut MirModule) {
    MirOptimizer::optimize_module(module);

    let mut struct_layouts: hashbrown::HashMap<String, MirStructLayout> =
        hashbrown::HashMap::new();
    let mut conflicting: hashbrown::HashSet<String> = hashbrown::HashSet::new();
    let mut record_layout = |layout: &MirStructLayout| {
        if conflicting.contains(&layout.type_name) {
            return;
        }
        match struct_layouts.get(&layout.type_name) {
            Some(existing) if existing != layout => {
                struct_layouts.remove(&layout.type_name);
                conflicting.insert(layout.type_name.clone());
            }
            Some(_) => {}
            None => {
                struct_layouts.insert(layout.type_name.clone(), layout.clone());
            }
        }
    };
    for layout in &module.struct_layouts {
        record_layout(layout);
    }
    for func in &module.functions {
        for layout in &func.struct_layouts {
            record_layout(layout);
        }
    }

    if !struct_layouts.is_empty() {
        for func in &mut module.functions {
            if !func.cleanup_blocks.is_empty() {
                continue;
            }
            let pre_sroa = func.clone();
            if ScalarReplacementOfAggregates::run_on_function(func, &struct_layouts) {
                MirOptimizer::optimize_function(func);
                if MirVerifier::verify_fn(func).is_err() {
                    *func = pre_sroa;
                }
            }
        }
    }
}

#[cfg(feature = "cranelift-aot")]
fn prevalidate_object_target_triple(trimmed: &str) -> Result<()> {
    if trimmed.is_empty()
        || trimmed
            .bytes()
            .any(|b| !b.is_ascii_alphanumeric() && b != b'-' && b != b'_' && b != b'.')
    {
        return Err(LustError::CompileError(format!(
            "Unsupported or invalid target triple '{trimmed}' for cranelift-aot object emission"
        )));
    }

    let parts: Vec<&str> = trimmed.split('-').collect();
    if parts.len() < 2 || parts.len() > 4 || parts.iter().any(|p| p.is_empty()) {
        return Err(LustError::CompileError(format!(
            "Unsupported or invalid target triple '{trimmed}' for cranelift-aot object emission"
        )));
    }

    let arch = parts[0];
    let supported_64bit_arch = matches!(
        arch,
        "aarch64" | "arm64" | "x86_64" | "riscv64" | "riscv64gc" | "s390x"
    );
    if !supported_64bit_arch {
        return Err(LustError::CompileError(format!(
            "Unsupported target architecture '{arch}' in triple '{trimmed}' for cranelift-aot object emission (supported 64-bit object architectures: aarch64, x86_64, riscv64, s390x)"
        )));
    }

    let (os, env_opt) = match parts.len() {
        2 => (parts[1], None),
        3 => (parts[2], None),
        4 => (parts[2], Some(parts[3])),
        _ => unreachable!(),
    };

    let os_supported = matches!(
        os,
        "linux"
            | "freebsd"
            | "netbsd"
            | "openbsd"
            | "dragonfly"
            | "illumos"
            | "solaris"
            | "windows"
            | "none"
            | "android"
            | "fuchsia"
            | "redox"
            | "haiku"
            | "aix"
            | "hurd"
            | "cygwin"
    ) || os.starts_with("darwin")
        || os.starts_with("macos")
        || os.starts_with("ios");

    if !os_supported {
        return Err(LustError::CompileError(format!(
            "Unsupported target operating system '{os}' in triple '{trimmed}' for cranelift-aot object emission"
        )));
    }

    if let Some(env) = env_opt {
        let env_supported = matches!(
            env,
            "gnu"
                | "gnueabi"
                | "gnueabihf"
                | "gnuabi64"
                | "musl"
                | "msvc"
                | "android"
                | "androideabi"
                | "eabi"
                | "eabihf"
                | "elf"
                | "macho"
                | "sgx"
                | "sim"
                | "macabi"
                | "ohos"
                | "newlib"
                | "none"
                | "uclibc"
        );
        if !env_supported {
            return Err(LustError::CompileError(format!(
                "Unsupported target environment '{env}' in triple '{trimmed}' for cranelift-aot object emission"
            )));
        }
    }

    Ok(())
}

#[cfg(feature = "cranelift-aot")]
fn build_object_target_isa(
    target_triple: Option<&str>,
) -> Result<(cranelift_codegen::isa::OwnedTargetIsa, String)> {
    use cranelift_codegen::isa::{self, LookupError};
    use cranelift_codegen::settings::{self, Configurable};

    let mut flag_builder = settings::builder();
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| LustError::CompileError(format!("Failed to set Cranelift opt_level: {e}")))?;
    flag_builder
        .set("is_pic", "true")
        .map_err(|e| LustError::CompileError(format!("Failed to set Cranelift is_pic: {e}")))?;
    flag_builder
        .set("use_colocated_libcalls", "false")
        .map_err(|e| {
            LustError::CompileError(format!(
                "Failed to set Cranelift use_colocated_libcalls: {e}"
            ))
        })?;
    if !cfg!(debug_assertions) {
        flag_builder
            .set("enable_verifier", "false")
            .map_err(|e| {
                LustError::CompileError(format!("Failed to set Cranelift enable_verifier: {e}"))
            })?;
    }
    let flags = settings::Flags::new(flag_builder);

    let isa_builder = match target_triple {
        Some(raw_triple) => {
            let trimmed = raw_triple.trim();
            if trimmed.is_empty() {
                return Err(LustError::CompileError(
                    "Invalid empty target triple for cranelift-aot object emission".to_string(),
                ));
            }
            // Pre-validate triple syntax and supported components BEFORE calling
            // `isa::lookup_by_name`, which panics via `target_lexicon::triple!` on
            // malformed strings (fatal under `panic = "abort"` release profiles).
            prevalidate_object_target_triple(trimmed)?;
            let lookup_res = std::panic::catch_unwind(|| isa::lookup_by_name(trimmed)).map_err(
                |_| {
                    LustError::CompileError(format!(
                        "Unsupported or invalid target triple '{trimmed}' for cranelift-aot object emission"
                    ))
                },
            )?;
            lookup_res.map_err(|err| match err {
                LookupError::SupportDisabled => LustError::CompileError(format!(
                    "Target triple '{trimmed}' ISA backend is disabled in this cranelift-codegen build \
                     (enable cranelift-codegen/all-native-arch or the target architecture feature)"
                )),
                LookupError::Unsupported => LustError::CompileError(format!(
                    "Unsupported target triple '{trimmed}' for cranelift-aot object emission"
                )),
            })?
        }
        None => cranelift_native::builder().map_err(|msg| {
            LustError::CompileError(format!(
                "Host target not supported by Cranelift object backend: {msg}"
            ))
        })?,
    };

    let isa = isa_builder.finish(flags).map_err(|e| {
        LustError::CompileError(format!(
            "Failed to finalize Cranelift TargetIsa for {:?}: {e}",
            target_triple
        ))
    })?;

    let resolved_triple = isa.triple().to_string();
    Ok((isa, resolved_triple))
}

/// Returns `true` iff `cranelift-codegen` in this build has the ISA backend compiled in for
/// `target_triple` AND `cranelift-object` supports its binary format and architecture.
#[cfg(feature = "cranelift-aot")]
pub fn is_target_isa_compiled(target_triple: &str) -> bool {
    let Ok((isa, _)) = build_object_target_isa(Some(target_triple)) else {
        return false;
    };
    cranelift_object::ObjectBuilder::new(
        isa,
        b"probe".to_vec(),
        cranelift_module::default_libcall_names(),
    )
    .is_ok()
}

/// Compiles `module` into in-memory Mach-O / ELF object bytes plus structured metadata.
///
/// Fails closed on empty modules, unsupported/disabled triples, unresolved or non-C-ABI
/// runtime imports, and unimplemented AOT semantics (e.g. VM-only effect intrinsics or
/// compiler-process string handle literals).
#[cfg(feature = "cranelift-aot")]
pub fn compile_mir_to_object_bytes(
    module: &MirModule,
    target_triple: Option<&str>,
) -> Result<(Vec<u8>, CraneliftObjectEmissionReport)> {
    use cranelift_object::{ObjectBuilder, ObjectModule};

    if module.functions.is_empty() {
        return Err(LustError::CompileError(
            "Cranelift AOT object emission failed: MIR module contains no functions".to_string(),
        ));
    }

    let mut opt_module = module.clone();
    canonicalize_module_callees(&mut opt_module);
    optimize_module_for_cranelift_aot(&mut opt_module);

    let (isa, resolved_triple) = build_object_target_isa(target_triple)?;
    let binary_format = format!("{:?}", isa.triple().binary_format);
    let architecture = isa.triple().architecture.to_string();

    let obj_name = if opt_module.name.is_empty() {
        "lust_module"
    } else {
        opt_module.name.as_str()
    };
    let obj_builder = ObjectBuilder::new(
        isa,
        obj_name.as_bytes().to_vec(),
        cranelift_module::default_libcall_names(),
    )
    .map_err(|e| {
        LustError::CompileError(format!(
            "Unsupported target format/architecture for triple '{resolved_triple}': {e}"
        ))
    })?;

    let obj_module = ObjectModule::new(obj_builder);
    let mut compiler = CraneliftObjectCompiler::from_module_aot(obj_module);
    compiler.set_gas_budget_active(true);

    let lowered = compiler.lower_module(&opt_module).map_err(|err| {
        LustError::CompileError(format!(
            "Cranelift AOT object lowering failed for '{resolved_triple}': {err}"
        ))
    })?;

    if lowered.exported_functions.is_empty() {
        return Err(LustError::CompileError(format!(
            "Cranelift AOT object emission produced zero exported functions for module '{}'",
            opt_module.name
        )));
    }

    let product = compiler.into_module().finish();
    let bytes = product.emit().map_err(|err| {
        LustError::CompileError(format!(
            "Cranelift object emission failed for '{resolved_triple}': {err}"
        ))
    })?;

    if bytes.is_empty() {
        return Err(LustError::CompileError(format!(
            "Cranelift object emission produced an empty object file for '{resolved_triple}'"
        )));
    }

    let report = CraneliftObjectEmissionReport {
        output_path: PathBuf::new(),
        module_name: opt_module.name,
        target_triple: resolved_triple,
        binary_format,
        architecture,
        exported_functions: lowered.exported_functions,
        imported_runtime_symbols: lowered.imported_runtime_symbols,
        object_size_bytes: bytes.len(),
    };
    Ok((bytes, report))
}

/// Validates that every function in `module` lowers and compiles cleanly through Cranelift.
///
/// - Without `cranelift-aot`: validates through [`CraneliftJitCompiler`] with zero object emission.
/// - With `cranelift-aot`: validates target ISA + full [`CraneliftObjectCompiler`] object emission
///   in memory without touching disk.
pub fn validate_mir_module_clif(
    module: &MirModule,
    target_triple: Option<&str>,
) -> Result<CraneliftClifValidationReport> {
    #[cfg(feature = "cranelift-aot")]
    {
        let (_bytes, report) = compile_mir_to_object_bytes(module, target_triple)?;
        Ok(CraneliftClifValidationReport {
            module_name: report.module_name,
            validated_functions: report.exported_functions,
            target_triple: Some(report.target_triple),
            object_emission_status: CRANELIFT_OBJECT_BACKEND_STATUS,
        })
    }

    #[cfg(all(feature = "cranelift", not(feature = "cranelift-aot")))]
    {
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

    #[cfg(not(feature = "cranelift"))]
    {
        let _ = (module, target_triple);
        Err(LustError::CompileError(
            "Cranelift CLIF validation requires the `cranelift` feature".to_string(),
        ))
    }
}

/// Emits a real Mach-O or ELF `.o` file to `output_obj` and returns a full emission report.
///
/// Fails closed before creating or modifying `output_obj` on any validation, target ISA,
/// lowering, or object-writing error.
#[cfg(feature = "cranelift-aot")]
pub fn compile_mir_to_object_file_with_report(
    module: &MirModule,
    output_obj: &Path,
    target_triple: Option<&str>,
) -> Result<CraneliftObjectEmissionReport> {
    let (bytes, mut report) = compile_mir_to_object_bytes(module, target_triple)?;
    if let Some(parent) = output_obj.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| {
            LustError::CompileError(format!(
                "Failed to create output directory '{}': {e}",
                parent.display()
            ))
        })?;
    }
    std::fs::write(output_obj, &bytes).map_err(|e| {
        LustError::CompileError(format!(
            "Failed to write Cranelift object file '{}': {e}",
            output_obj.display()
        ))
    })?;
    report.output_path = output_obj.to_path_buf();
    Ok(report)
}

/// `--backend cranelift-aot` entrypoint (`F-11`, `W16-CRANELIFT-OBJ`).
///
/// - With `--features cranelift-aot`: compiles `module` through `cranelift_object::ObjectModule`
///   for `target_triple` (or host ISA when `None`), writes the emitted Mach-O / ELF `.o` bytes to
///   `output_obj`, and returns `Ok(output_obj.to_path_buf())`.
/// - Without `--features cranelift-aot`: validates 100% CLIF compilability, then fails closed with
///   [`CRANELIFT_OBJECT_BLOCKED_REASON`] without creating `output_obj`.
pub fn compile_mir_to_object_file(
    module: &MirModule,
    output_obj: &Path,
    target_triple: Option<&str>,
) -> Result<PathBuf> {
    #[cfg(feature = "cranelift-aot")]
    {
        let report = compile_mir_to_object_file_with_report(module, output_obj, target_triple)?;
        Ok(report.output_path)
    }

    #[cfg(not(feature = "cranelift-aot"))]
    {
        let _ = output_obj;
        let _report = validate_mir_module_clif(module, target_triple)?;
        Err(LustError::CompileError(
            CRANELIFT_OBJECT_BLOCKED_REASON.to_string(),
        ))
    }
}

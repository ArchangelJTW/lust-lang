use crate::ast::{FunctionDef, Item, ItemKind, Span, Type, TypeKind};
use crate::error::{LustError, Result};
use crate::intern::Interner;
use crate::lexer::Lexer;
use crate::mir::ir::MirModule;
use crate::mir::lowering::AstLowering;
use crate::mir::opt::MirOptimizer;
use crate::parser::Parser;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::c_emitter::{CEmitterOptions, emit_c_from_mir};
use super::rust_emitter::{
    GasMetering, RustEmitterOptions, RustInteropOptions, emit_rust_from_mir,
    emit_rust_from_mir_with_ast, emit_rust_from_mir_with_interop,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Options configuring the Next-Gen Rust AOT compilation pipeline (`NG-W12-4A`).
#[derive(Debug, Clone)]
pub struct RustCompileOptions {
    pub aot: AotCompileOptions,
    pub gas_metering: GasMetering,
    pub gas_limit: Option<u64>,
    pub pure_core: bool,
    pub no_std: bool,
    pub cache_dir: Option<PathBuf>,
}

impl Default for RustCompileOptions {
    fn default() -> Self {
        Self {
            aot: AotCompileOptions::default(),
            gas_metering: GasMetering::None,
            gas_limit: None,
            pure_core: false,
            no_std: false,
            cache_dir: None,
        }
    }
}

impl RustCompileOptions {
    pub fn to_emitter_options(&self) -> RustEmitterOptions {
        RustEmitterOptions {
            is_library: self.aot.target_kind != AotTargetKind::Executable,
            max_memory_bytes: self.aot.max_memory_bytes,
            gas_metering: self.gas_metering,
            gas_limit: self.gas_limit,
            pure_core: self.pure_core,
            no_std: self.no_std,
        }
    }
}

/// Build target kind for AOT compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AotTargetKind {
    Executable,
    SharedLibrary,
    StaticLibrary,
}

/// Options configuring the autonomous AOT compilation pipeline.
#[derive(Debug, Clone)]
pub struct AotCompileOptions {
    pub target_kind: AotTargetKind,
    pub release_mode: String,
    pub strip: bool,
    pub max_memory_bytes: Option<usize>,
    pub target_triple: Option<String>,
}

impl Default for AotCompileOptions {
    fn default() -> Self {
        Self {
            target_kind: AotTargetKind::Executable,
            release_mode: "ReleaseFast".to_string(),
            strip: false,
            max_memory_bytes: None,
            target_triple: None,
        }
    }
}

/// Link options for external libraries and capabilities in AOT compilation.
#[derive(Debug, Clone)]
pub struct AotLinkOptions {
    pub link_libraries: Vec<PathBuf>,
    pub link_dirs: Vec<PathBuf>,
    pub allow_ffi: bool,
}

impl Default for AotLinkOptions {
    fn default() -> Self {
        Self {
            link_libraries: Vec::new(),
            link_dirs: Vec::new(),
            allow_ffi: true,
        }
    }
}

/// Direct Zero-FFI external Rust crate `.rlib` binding (`--backend rust --link-crate`, `NG-W13-1C`).
#[derive(Debug, Clone)]
pub struct RustLinkedCrate {
    /// Sanitized Rust crate identifier (e.g. `my_ext`), emitted as `extern crate my_ext;`.
    pub crate_name: String,
    /// Path to the compiled `.rlib` artifact passed via `--extern <crate_name>=<rlib_path>`.
    pub rlib_path: PathBuf,
    /// Optional transitive dependency search directories passed via `-L dependency=<dir>`.
    pub dependency_dirs: Vec<PathBuf>,
}

/// Extended Zero-FFI linking, ThinLTO & PGO options for Rust AOT compilation (`NG-W13-1C`, `NG-W16-2`).
#[derive(Debug, Clone, Default)]
pub struct RustLinkOptions {
    pub aot_links: AotLinkOptions,
    /// External Rust crates linked directly as `.rlib` without `LustExtVal` C-ABI marshaling.
    pub linked_crates: Vec<RustLinkedCrate>,
    /// Enable `-C lto=thin` in `rustc` (automatically enabled whenever `linked_crates` is non-empty).
    pub lto_thin: bool,
    /// Optional `-C profile-generate=<dir>` output directory (`AOT-PGO-CLI`).
    pub pgo_generate: Option<PathBuf>,
    /// Optional `-C profile-use=<profdata>` path (`AOT-PGO-CLI`).
    pub pgo_use: Option<PathBuf>,
    /// Optional `-C profile-sample-use=<prof>` path (`AOT-PGO-CLI`).
    pub pgo_sample_use: Option<PathBuf>,
    /// Optional `liblust_aot_rt.rlib` override path (or `LUST_AOT_RT_RLIB` env var).
    pub aot_rt_rlib_override: Option<PathBuf>,
}

/// Local Cargo path dependency added to a generated `--emit=rust-crate` package (`NG-W13-1C`).
#[derive(Debug, Clone)]
pub struct RustCrateDependency {
    /// Cargo package name as declared in `Cargo.toml`.
    pub package_name: String,
    /// Sanitized Rust crate identifier used by `extern crate <crate_name>;`.
    pub crate_name: String,
    /// Path to the external Cargo crate directory.
    pub path: PathBuf,
}

/// Artifacts produced by `lust build --emit=rust-crate -o <dir>` (`NG-W13-1C`).
#[derive(Debug, Clone)]
pub struct RustCrateEmission {
    pub crate_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub lib_rs_path: PathBuf,
    pub package_name: String,
    pub lib_name: String,
}

/// Deterministic build outcome for Rust AOT cross-target compilation (`NG-W13-2`, `F-11`).
///
/// Honestly separates installed offline targets (compiled directly by `rustc`) from
/// uninstalled offline targets (`libcore-*.rlib` missing from `rustc --print target-libdir`),
/// for which a complete `#![no_std]` `.rs` file is generated instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RustAotBuildOutcome {
    /// Target sysroot is installed offline and `rustc` produced the binary/library artifact.
    Compiled {
        target: String,
        artifact_path: PathBuf,
    },
    /// Target `libcore-*.rlib` is not installed offline; emitted `#![no_std]` Rust source.
    TargetStdlibNotInstalledOffline {
        target: String,
        generated_rs_path: PathBuf,
    },
}

/// Returns `true` when `target_triple` names a bare-metal embedded MCU target
/// (e.g. `thumbv7em-none-eabihf`, `riscv32imac-unknown-none-elf`).
pub fn is_embedded_mcu_target(target_triple: &str) -> bool {
    let t = target_triple.to_ascii_lowercase();
    t.starts_with("thumbv")
        || t.starts_with("riscv32")
        || t.contains("-none-")
        || t.ends_with("-none-elf")
        || t.ends_with("-none-eabi")
        || t.ends_with("-none-eabihf")
}

/// Returns `true` when `target_triple` defaults to `#![no_std]` emission in Rust AOT
/// (`wasm32-unknown-unknown` and bare-metal MCU targets such as `thumbv7em`/`riscv32`).
pub fn is_nostd_target(target_triple: &str) -> bool {
    target_triple.eq_ignore_ascii_case("wasm32-unknown-unknown")
        || is_embedded_mcu_target(target_triple)
}

/// Checks whether the active `rustc` installation has `libcore-*.rlib` installed offline
/// for `target_triple` via `rustc --print target-libdir --target <triple>` (`NG-W13-2`, `F-11`).
pub fn check_target_core_installed(target_triple: &str) -> Result<bool> {
    let rustc = find_rustc();
    let output = Command::new(&rustc)
        .args(["--print", "target-libdir", "--target", target_triple])
        .output()
        .map_err(|e| {
            LustError::CompileError(format!(
                "Failed to execute {rustc} --print target-libdir --target {target_triple}: {e}"
            ))
        })?;
    if !output.status.success() {
        return Ok(false);
    }
    let libdir_raw = String::from_utf8_lossy(&output.stdout);
    let libdir = PathBuf::from(libdir_raw.trim());
    if !libdir.is_dir() {
        return Ok(false);
    }
    let entries = match fs::read_dir(&libdir) {
        Ok(e) => e,
        Err(_) => return Ok(false),
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with("libcore-") && name_str.ends_with(".rlib") {
            return Ok(true);
        }
    }
    Ok(false)
}

fn detect_host_target_triple() -> String {
    let rustc = find_rustc();
    if let Ok(output) = Command::new(&rustc).arg("-vV").output()
        && output.status.success()
    {
        let text = String::from_utf8_lossy(&output.stdout);
        for line in text.lines() {
            if let Some(host) = line.strip_prefix("host:") {
                let trimmed = host.trim();
                if !trimmed.is_empty() {
                    return trimmed.to_string();
                }
            }
        }
    }
    "host".to_string()
}

fn effective_cross_target_options(opts: &RustCompileOptions) -> RustCompileOptions {
    let mut effective = opts.clone();
    if let Some(triple) = effective.aot.target_triple.as_deref() {
        if is_nostd_target(triple) {
            effective.no_std = true;
        }
        if triple.eq_ignore_ascii_case("wasm32-unknown-unknown")
            && effective.aot.target_kind == AotTargetKind::Executable
        {
            effective.aot.target_kind = AotTargetKind::SharedLibrary;
        }
    }
    effective
}

fn resolve_offline_fallback_rs_path(output_bin: &Path) -> PathBuf {
    if output_bin.extension().and_then(|e| e.to_str()) == Some("rs") {
        output_bin.to_path_buf()
    } else {
        output_bin.with_extension("rs")
    }
}

/// Autonomous AOT Compiler: Produces native release binaries without requiring system `zig`.
pub struct AotCompiler;

impl AotCompiler {
    /// Compiles a MIR module into a native machine-code binary or library.
    pub fn compile_module(
        module: &MirModule,
        output_bin: &Path,
        opts: &AotCompileOptions,
    ) -> Result<PathBuf> {
        Self::compile_module_with_links(module, output_bin, opts, &AotLinkOptions::default())
    }

    /// Compiles a MIR module with external link options.
    pub fn compile_module_with_links(
        module: &MirModule,
        output_bin: &Path,
        opts: &AotCompileOptions,
        links: &AotLinkOptions,
    ) -> Result<PathBuf> {
        let mut opt_module = module.clone();

        // 1. Run MIR optimizations (Inlining, ConstantFolding, DeadCodeElimination)
        MirOptimizer::optimize_module(&mut opt_module);

        // 2. Emit C99 code
        let emitter_opts = CEmitterOptions {
            is_library: opts.target_kind != AotTargetKind::Executable,
            max_memory_bytes: opts.max_memory_bytes,
        };
        let c_source = emit_c_from_mir(&opt_module, &emitter_opts);

        // 3. Compile C code with host C compiler
        compile_c_source(&c_source, output_bin, opts, links)
    }

    /// Compiles Lust source code string into a native machine-code binary.
    pub fn compile_source(
        source: &str,
        output_bin: &Path,
        opts: &AotCompileOptions,
    ) -> Result<PathBuf> {
        Self::compile_source_with_links(source, output_bin, opts, &AotLinkOptions::default())
    }

    /// Compiles Lust source code string with external link options.
    pub fn compile_source_with_links(
        source: &str,
        output_bin: &Path,
        opts: &AotCompileOptions,
        links: &AotLinkOptions,
    ) -> Result<PathBuf> {
        let mut interner = Interner::new();
        let tokens = Lexer::new(source, &mut interner)
            .tokenize()
            .map_err(|e| LustError::Unknown(format!("Lexer error: {e}")))?;
        let items = Parser::new(tokens)
            .parse()
            .map_err(|e| LustError::Unknown(format!("Parser error: {e}")))?;

        let prepared_items = prepare_items_for_mir(&items);
        let mut module = AstLowering::lower_module_items("main_module", &prepared_items)
            .map_err(|e| LustError::Unknown(format!("MIR lowering error: {e}")))?;
        register_ast_struct_layouts(&mut module, &items);

        Self::compile_module_with_links(&module, output_bin, opts, links)
    }

    /// Compiles a `.lust` source file on disk into a native binary.
    pub fn compile_file(
        source_path: &Path,
        output_bin: &Path,
        opts: &AotCompileOptions,
    ) -> Result<PathBuf> {
        Self::compile_file_with_links(source_path, output_bin, opts, &AotLinkOptions::default())
    }

    /// Compiles a `.lust` source file on disk with external link options.
    pub fn compile_file_with_links(
        source_path: &Path,
        output_bin: &Path,
        opts: &AotCompileOptions,
        links: &AotLinkOptions,
    ) -> Result<PathBuf> {
        let source = fs::read_to_string(source_path)
            .map_err(|e| LustError::Unknown(format!("Failed to read {}: {e}", source_path.display())))?;
        Self::compile_source_with_links(&source, output_bin, opts, links)
    }

    /// Emits Rust AOT source from an already-lowered `MirModule` (`NG-W12-4A`).
    pub fn emit_rust_module(module: &MirModule, opts: &RustCompileOptions) -> Result<String> {
        let emitter_opts = opts.to_emitter_options();
        emit_rust_from_mir(module, &emitter_opts)
    }

    /// Emits Rust AOT source from a `MirModule` together with direct AST items (`NG-W13-1A`, `F-19`),
    /// forwarding `ServiceDef`, `TraitDef`, `ImplBlock`, `FunctionDef.effects`, and desugared
    /// `__lust_handle`/`__lust_perform`/`__lust_resume` metadata without modifying `src/mir/ir.rs`.
    pub fn emit_rust_module_with_ast(
        module: &MirModule,
        ast_items: &[Item],
        opts: &RustCompileOptions,
    ) -> Result<String> {
        let emitter_opts = opts.to_emitter_options();
        emit_rust_from_mir_with_ast(module, ast_items, &emitter_opts)
    }

    /// Runs the full frontend (`Lexer -> Parser -> TypeChecker -> prepare_items_for_mir ->
    /// AstLowering -> MirOptimizer + SROA + refine + EscapeAnalysis -> cfg_structurer -> Rust`)
    /// on a Lust source string and returns the generated Rust code with direct AST forwarding.
    pub fn emit_rust_source(source: &str, opts: &RustCompileOptions) -> Result<String> {
        let (module, items) = lower_and_typecheck_source(source)?;
        Self::emit_rust_module_with_ast(&module, &items, opts)
    }

    /// Compiles a `MirModule` via the Rust AOT pipeline (`rustc` + cached `liblust_aot_rt.rlib`).
    pub fn compile_module_rust(
        module: &MirModule,
        output_bin: &Path,
        opts: &RustCompileOptions,
    ) -> Result<PathBuf> {
        Self::compile_module_rust_with_links(module, output_bin, opts, &AotLinkOptions::default())
    }

    /// Compiles a `MirModule` via the Rust AOT pipeline with external link options.
    pub fn compile_module_rust_with_links(
        module: &MirModule,
        output_bin: &Path,
        opts: &RustCompileOptions,
        links: &AotLinkOptions,
    ) -> Result<PathBuf> {
        let rust_source = Self::emit_rust_module(module, opts)?;
        compile_rust_source(&rust_source, output_bin, opts, links)
    }

    /// Compiles a `MirModule` + direct AST items via the Rust AOT pipeline (`NG-W13-1A`, `F-19`).
    pub fn compile_module_rust_with_ast(
        module: &MirModule,
        ast_items: &[Item],
        output_bin: &Path,
        opts: &RustCompileOptions,
    ) -> Result<PathBuf> {
        Self::compile_module_rust_with_ast_and_links(
            module,
            ast_items,
            output_bin,
            opts,
            &AotLinkOptions::default(),
        )
    }

    /// Compiles a `MirModule` + direct AST items via the Rust AOT pipeline with external link options.
    pub fn compile_module_rust_with_ast_and_links(
        module: &MirModule,
        ast_items: &[Item],
        output_bin: &Path,
        opts: &RustCompileOptions,
        links: &AotLinkOptions,
    ) -> Result<PathBuf> {
        let rust_source = Self::emit_rust_module_with_ast(module, ast_items, opts)?;
        compile_rust_source(&rust_source, output_bin, opts, links)
    }

    /// Compiles a Lust source string via the Rust AOT pipeline.
    pub fn compile_source_rust(
        source: &str,
        output_bin: &Path,
        opts: &RustCompileOptions,
    ) -> Result<PathBuf> {
        Self::compile_source_rust_with_links(source, output_bin, opts, &AotLinkOptions::default())
    }

    /// Compiles a Lust source string via the Rust AOT pipeline with external link options.
    pub fn compile_source_rust_with_links(
        source: &str,
        output_bin: &Path,
        opts: &RustCompileOptions,
        links: &AotLinkOptions,
    ) -> Result<PathBuf> {
        let (module, items) = lower_and_typecheck_source(source)?;
        Self::compile_module_rust_with_ast_and_links(&module, &items, output_bin, opts, links)
    }

    /// Compiles a `.lust` file on disk via the Rust AOT pipeline.
    pub fn compile_file_rust(
        source_path: &Path,
        output_bin: &Path,
        opts: &RustCompileOptions,
    ) -> Result<PathBuf> {
        let source = fs::read_to_string(source_path).map_err(|e| {
            LustError::Unknown(format!("Failed to read {}: {e}", source_path.display()))
        })?;
        Self::compile_source_rust(&source, output_bin, opts)
    }

    /// Emits Rust AOT source with explicit Cargo/Zero-FFI interop options (`NG-W13-1C`).
    pub fn emit_rust_module_with_interop(
        module: &MirModule,
        ast_items: &[Item],
        opts: &RustCompileOptions,
        interop: &RustInteropOptions,
    ) -> Result<String> {
        let emitter_opts = opts.to_emitter_options();
        emit_rust_from_mir_with_interop(module, ast_items, &emitter_opts, interop)
    }

    /// Emits an `include!()`-compatible Rust AOT module (no crate-level `#![...]` attributes,
    /// no `fn main()`, with public typed Rust wrapper functions) for `RustAotBuilder` (`NG-W13-1C`).
    pub fn emit_rust_include_module_from_items(
        module: &MirModule,
        ast_items: &[Item],
        opts: &RustCompileOptions,
        linked_crates: &[String],
    ) -> Result<String> {
        let mut emitter_opts = opts.to_emitter_options();
        emitter_opts.is_library = true;
        let interop = RustInteropOptions {
            linked_crates: linked_crates.to_vec(),
            emit_as_include_module: true,
            export_public_api: true,
        };
        emit_rust_from_mir_with_interop(module, ast_items, &emitter_opts, &interop)
    }

    /// Generates a self-contained Cargo library crate (`Cargo.toml` + `src/lib.rs`) depending on
    /// `lust-aot-rt` (and any optional linked local Cargo crates) for `lust build --emit=rust-crate`
    /// (`NG-W13-1C`).
    pub fn emit_rust_crate(
        module: &MirModule,
        ast_items: &[Item],
        output_dir: &Path,
        package_name: &str,
        opts: &RustCompileOptions,
        crate_deps: &[RustCrateDependency],
    ) -> Result<RustCrateEmission> {
        let sanitized_pkg = sanitize_cargo_package_name(package_name);
        let lib_name = sanitized_pkg.replace('-', "_");
        let rt_dir = find_aot_rt_crate_dir()?;
        let rt_dir_canon = fs::canonicalize(&rt_dir).unwrap_or(rt_dir);

        let mut emitter_opts = opts.to_emitter_options();
        emitter_opts.is_library = true;
        let interop = RustInteropOptions {
            linked_crates: crate_deps.iter().map(|d| d.crate_name.clone()).collect(),
            emit_as_include_module: false,
            export_public_api: true,
        };
        let lib_rs_source =
            emit_rust_from_mir_with_interop(module, ast_items, &emitter_opts, &interop)?;

        let src_dir = output_dir.join("src");
        fs::create_dir_all(&src_dir).map_err(|e| {
            LustError::Unknown(format!(
                "Failed to create crate directory {}: {e}",
                src_dir.display()
            ))
        })?;

        let mut manifest = String::new();
        manifest.push_str("[package]\n");
        manifest.push_str(&format!(
            "name = \"{}\"\n",
            escape_toml_string(&sanitized_pkg)
        ));
        manifest.push_str("version = \"0.1.0\"\n");
        manifest.push_str("edition = \"2024\"\n\n");
        manifest.push_str("[lib]\n");
        manifest.push_str(&format!("name = \"{}\"\n", escape_toml_string(&lib_name)));
        manifest.push_str("path = \"src/lib.rs\"\n\n");
        manifest.push_str("[dependencies]\n");
        if opts.no_std {
            manifest.push_str(&format!(
                "lust-aot-rt = {{ path = \"{}\", default-features = false }}\n",
                escape_toml_string(&rt_dir_canon.to_string_lossy())
            ));
        } else {
            manifest.push_str(&format!(
                "lust-aot-rt = {{ path = \"{}\" }}\n",
                escape_toml_string(&rt_dir_canon.to_string_lossy())
            ));
        }
        for dep in crate_deps {
            let dep_canon = fs::canonicalize(&dep.path).unwrap_or_else(|_| dep.path.clone());
            manifest.push_str(&format!(
                "{} = {{ path = \"{}\" }}\n",
                dep.package_name,
                escape_toml_string(&dep_canon.to_string_lossy())
            ));
        }
        manifest.push_str("\n[workspace]\n");

        let manifest_path = output_dir.join("Cargo.toml");
        let lib_rs_path = src_dir.join("lib.rs");
        fs::write(&manifest_path, manifest).map_err(|e| {
            LustError::Unknown(format!("Failed to write {}: {e}", manifest_path.display()))
        })?;
        fs::write(&lib_rs_path, lib_rs_source).map_err(|e| {
            LustError::Unknown(format!("Failed to write {}: {e}", lib_rs_path.display()))
        })?;

        Ok(RustCrateEmission {
            crate_dir: output_dir.to_path_buf(),
            manifest_path,
            lib_rs_path,
            package_name: sanitized_pkg,
            lib_name,
        })
    }

    /// Compiles a Lust source string directly into a Cargo library crate (`Cargo.toml` + `src/lib.rs`).
    pub fn emit_rust_crate_from_source(
        source: &str,
        output_dir: &Path,
        package_name: &str,
        opts: &RustCompileOptions,
    ) -> Result<RustCrateEmission> {
        let (module, items) = lower_and_typecheck_source(source)?;
        Self::emit_rust_crate(&module, &items, output_dir, package_name, opts, &[])
    }

    /// Compiles a `MirModule` + AST items via Rust AOT with explicit offline cross-target
    /// detection (`NG-W13-2`, `F-11`).
    ///
    /// - Installed targets (`aarch64-apple-darwin`, `wasm32-wasip1`, `wasm32-unknown-unknown`):
    ///   compiles directly with `rustc` and returns [`RustAotBuildOutcome::Compiled`].
    /// - Uninstalled offline targets (`wasm32-wasip2`, `thumbv7em-none-eabihf`,
    ///   `riscv32imac-unknown-none-elf`): generates a complete `#![no_std]` `.rs` file
    ///   (with `StaticSlabAllocator<65536>` preamble) and returns
    ///   [`RustAotBuildOutcome::TargetStdlibNotInstalledOffline`].
    pub fn compile_module_rust_cross(
        module: &MirModule,
        ast_items: &[Item],
        output_bin: &Path,
        opts: &RustCompileOptions,
        links: &RustLinkOptions,
    ) -> Result<RustAotBuildOutcome> {
        let effective_opts = effective_cross_target_options(opts);
        if let Some(triple) = effective_opts.aot.target_triple.as_deref()
            && !check_target_core_installed(triple)?
        {
            let mut nostd_opts = effective_opts.clone();
            nostd_opts.no_std = true;
            let interop = RustInteropOptions {
                linked_crates: links
                    .linked_crates
                    .iter()
                    .map(|c| c.crate_name.clone())
                    .collect(),
                emit_as_include_module: false,
                export_public_api: true,
            };
            let rust_source =
                Self::emit_rust_module_with_interop(module, ast_items, &nostd_opts, &interop)?;
            let generated_rs_path = resolve_offline_fallback_rs_path(output_bin);
            if let Some(parent) = generated_rs_path.parent()
                && !parent.as_os_str().is_empty()
            {
                let _ = fs::create_dir_all(parent);
            }
            fs::write(&generated_rs_path, &rust_source).map_err(|e| {
                LustError::CompileError(format!(
                    "Failed to write offline #![no_std] Rust source {}: {e}",
                    generated_rs_path.display()
                ))
            })?;
            return Ok(RustAotBuildOutcome::TargetStdlibNotInstalledOffline {
                target: triple.to_string(),
                generated_rs_path,
            });
        }

        let is_wasm_unknown = effective_opts
            .aot
            .target_triple
            .as_deref()
            .is_some_and(|t| t.eq_ignore_ascii_case("wasm32-unknown-unknown"));
        let interop = RustInteropOptions {
            linked_crates: links
                .linked_crates
                .iter()
                .map(|c| c.crate_name.clone())
                .collect(),
            emit_as_include_module: false,
            export_public_api: effective_opts.aot.target_kind != AotTargetKind::Executable
                || is_wasm_unknown,
        };
        let rust_source =
            Self::emit_rust_module_with_interop(module, ast_items, &effective_opts, &interop)?;
        let artifact_path =
            compile_rust_source_with_rlibs(&rust_source, output_bin, &effective_opts, links)?;
        let resolved_target = effective_opts
            .aot
            .target_triple
            .unwrap_or_else(detect_host_target_triple);
        Ok(RustAotBuildOutcome::Compiled {
            target: resolved_target,
            artifact_path,
        })
    }

    /// Compiles a Lust source string via Rust AOT with explicit offline cross-target detection
    /// (`NG-W13-2`, `F-11`).
    pub fn compile_source_rust_cross(
        source: &str,
        output_bin: &Path,
        opts: &RustCompileOptions,
        links: &RustLinkOptions,
    ) -> Result<RustAotBuildOutcome> {
        let (module, items) = lower_and_typecheck_source(source)?;
        Self::compile_module_rust_cross(&module, &items, output_bin, opts, links)
    }

    /// Compiles a `MirModule` + AST items via Rust AOT with direct Zero-FFI `.rlib` crate
    /// linking (`--extern <ext>=<path>.rlib`) and ThinLTO (`-C lto=thin`) (`NG-W13-1C`).
    pub fn compile_module_rust_with_rlibs(
        module: &MirModule,
        ast_items: &[Item],
        output_bin: &Path,
        opts: &RustCompileOptions,
        links: &RustLinkOptions,
    ) -> Result<PathBuf> {
        match Self::compile_module_rust_cross(module, ast_items, output_bin, opts, links)? {
            RustAotBuildOutcome::Compiled { artifact_path, .. } => Ok(artifact_path),
            RustAotBuildOutcome::TargetStdlibNotInstalledOffline {
                target,
                generated_rs_path,
            } => Err(LustError::CompileError(format!(
                "TargetStdlibNotInstalledOffline {{ target: \"{target}\", generated_rs_path: \"{}\" }}",
                generated_rs_path.display()
            ))),
        }
    }

    /// Compiles a Lust source string via Rust AOT with direct Zero-FFI `.rlib` crate linking
    /// and ThinLTO (`-C lto=thin`) (`NG-W13-1C`).
    pub fn compile_source_rust_with_rlibs(
        source: &str,
        output_bin: &Path,
        opts: &RustCompileOptions,
        links: &RustLinkOptions,
    ) -> Result<PathBuf> {
        let (module, items) = lower_and_typecheck_source(source)?;
        Self::compile_module_rust_with_rlibs(&module, &items, output_bin, opts, links)
    }

    /// Fallback invoked when system `zig` is unavailable or disabled.
    pub fn compile_zig_fallback(
        zig_path: &Path,
        output_bin: &Path,
        opts: &crate::codegen_zig::CompileOptions,
        link_libc: bool,
    ) -> Result<PathBuf> {
        Self::compile_zig_fallback_with_links(
            zig_path,
            output_bin,
            opts,
            link_libc,
            &AotLinkOptions::default(),
        )
    }

    /// Fallback invoked when system `zig` is unavailable or disabled, passing link options.
    pub fn compile_zig_fallback_with_links(
        zig_path: &Path,
        output_bin: &Path,
        opts: &crate::codegen_zig::CompileOptions,
        _link_libc: bool,
        links: &AotLinkOptions,
    ) -> Result<PathBuf> {
        let aot_opts = AotCompileOptions {
            target_kind: match opts.target_kind {
                crate::codegen_zig::BuildTargetKind::Executable => AotTargetKind::Executable,
                crate::codegen_zig::BuildTargetKind::SharedLibrary => AotTargetKind::SharedLibrary,
                crate::codegen_zig::BuildTargetKind::StaticLibrary => AotTargetKind::StaticLibrary,
            },
            release_mode: opts.release_mode.clone(),
            strip: opts.strip,
            max_memory_bytes: opts.max_memory_bytes,
            target_triple: opts.target_triple.clone(),
        };

        // Try to find original .lust file beside the .zig file
        let lust_path = zig_path.with_extension("lust");
        if lust_path.exists() {
            return Self::compile_file_with_links(&lust_path, output_bin, &aot_opts, links);
        }

        // Try to extract embedded source from the .zig file
        if zig_path.exists() {
            if let Ok(zig_content) = fs::read_to_string(zig_path) {
                if let Some(source) = extract_embedded_lust_source(&zig_content) {
                    return Self::compile_source_with_links(&source, output_bin, &aot_opts, links);
                }
            }
        }

        Err(LustError::Unknown(format!(
            "AOT compiler fallback could not find original Lust source for {}",
            zig_path.display()
        )))
    }
}

/// If script has top-level script statements, wrap them into `main` function for MIR lowering.
pub fn prepare_items_for_mir(items: &[Item]) -> Vec<Item> {
    let mut prepared = Vec::new();
    let mut script_stmts = Vec::new();
    let mut has_main = false;

    for item in items {
        match &item.kind {
            ItemKind::Function(f) => {
                if f.name == "main" {
                    has_main = true;
                }
                prepared.push(item.clone());
            }
            ItemKind::Script(stmts) => {
                script_stmts.extend(stmts.clone());
            }
            _ => {
                prepared.push(item.clone());
            }
        }
    }

    if !has_main && !script_stmts.is_empty() {
        let dummy_span = Span::dummy();
        let main_fn = FunctionDef {
            name: "main".into(),
            type_params: Vec::new(),
            trait_bounds: Vec::new(),
            params: Vec::new(),
            return_type: Some(Type {
                kind: TypeKind::Int,
                span: dummy_span,
            }),
            body: script_stmts,
            is_method: false,
            visibility: crate::ast::Visibility::Public,
            doc: None,
            effects: None,
        };
        prepared.push(Item {
            kind: ItemKind::Function(main_fn),
            span: dummy_span,
        });
    }

    prepared
}

/// Extracts embedded Lust source if stored in a header comment inside the Zig file.
pub fn extract_embedded_lust_source(zig_src: &str) -> Option<String> {
    const PREFIX: &str = "// [LUST_ORIGINAL_SOURCE: ";
    for line in zig_src.lines() {
        if let Some(rest) = line.strip_prefix(PREFIX) {
            if let Some(encoded) = rest.strip_suffix(']') {
                // Decodes hex-encoded source string
                return decode_hex(encoded);
            }
        }
    }
    None
}

/// Encodes source string into a single-line hex comment header for zero-loss recovery.
pub fn encode_embedded_lust_source(source: &str) -> String {
    let mut hex = String::with_capacity(source.len() * 2);
    for b in source.as_bytes() {
        use core::fmt::Write;
        let _ = write!(&mut hex, "{:02x}", b);
    }
    format!("// [LUST_ORIGINAL_SOURCE: {hex}]\n")
}

fn decode_hex(hex: &str) -> Option<String> {
    if hex.len() % 2 != 0 {
        return None;
    }
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    for i in (0..hex.len()).step_by(2) {
        let byte = u8::from_str_radix(&hex[i..i + 2], 16).ok()?;
        bytes.push(byte);
    }
    String::from_utf8(bytes).ok()
}

/// Compiles C source code to output binary using the host C compiler (`cc`, `clang`, or `gcc`).
fn compile_c_source(
    c_source: &str,
    output_bin: &Path,
    opts: &AotCompileOptions,
    links: &AotLinkOptions,
) -> Result<PathBuf> {
    let temp_dir = tempfile::tempdir()
        .map_err(|e| LustError::Unknown(format!("Failed to create tempdir: {e}")))?;
    let c_file = temp_dir.path().join("aot_payload.c");
    fs::write(&c_file, c_source)
        .map_err(|e| LustError::Unknown(format!("Failed to write C payload: {e}")))?;

    if let Some(parent) = output_bin.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let cc = find_c_compiler().ok_or_else(|| {
        LustError::Unknown("No host C compiler found (cc, clang, gcc)".into())
    })?;

    if !links.allow_ffi && (!links.link_libraries.is_empty() || !links.link_dirs.is_empty()) {
        return Err(LustError::Unknown(
            "CapabilityDenied: external library linking requires 'ffi' capability".to_string(),
        ));
    }
    for lib in &links.link_libraries {
        if !lib.exists() {
            return Err(LustError::Unknown(format!(
                "missing external library at '{}'",
                lib.display()
            )));
        }
    }

    let mut cmd = Command::new(&cc);

    for dir in &links.link_dirs {
        cmd.arg(format!("-L{}", dir.display()));
    }
    for lib in &links.link_libraries {
        cmd.arg(lib);
    }

    // Release mode optimization flags
    if opts.release_mode.starts_with("Release") || opts.release_mode == "release" {
        cmd.arg("-O3");
    } else {
        cmd.arg("-O0").arg("-g");
    }

    match opts.target_kind {
        AotTargetKind::Executable => {
            cmd.arg(&c_file);
            cmd.arg("-o").arg(output_bin);
            cmd.arg("-lm");
        }
        AotTargetKind::SharedLibrary => {
            cmd.arg("-shared").arg("-fPIC");
            cmd.arg(&c_file);
            cmd.arg("-o").arg(output_bin);
            cmd.arg("-lm");
        }
        AotTargetKind::StaticLibrary => {
            let o_file = temp_dir.path().join("payload.o");
            cmd.arg("-c").arg(&c_file).arg("-o").arg(&o_file);

            let status = cmd
                .status()
                .map_err(|e| LustError::Unknown(format!("Failed to run C compiler {cc}: {e}")))?;
            if !status.success() {
                return Err(LustError::Unknown("C compilation to object file failed".into()));
            }

            let mut ar_cmd = Command::new("ar");
            ar_cmd.arg("rcs").arg(output_bin).arg(&o_file);
            let ar_status = ar_cmd
                .status()
                .map_err(|e| LustError::Unknown(format!("Failed to run ar: {e}")))?;
            if !ar_status.success() {
                return Err(LustError::Unknown("Failed to create static library with ar".into()));
            }

            return Ok(output_bin.to_path_buf());
        }
    }

    if opts.strip {
        cmd.arg("-s");
    }

    let output = cmd
        .output()
        .map_err(|e| LustError::Unknown(format!("Failed to execute {cc}: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(LustError::Unknown(format!(
            "C compilation failed with status {}:\n{stderr}\nC Source:\n{c_source}",
            output.status
        )));
    }

    Ok(output_bin.to_path_buf())
}

fn find_c_compiler() -> Option<String> {
    if let Ok(cc) = std::env::var("CC") {
        if !cc.trim().is_empty() {
            return Some(cc);
        }
    }
    for candidate in &["cc", "clang", "gcc"] {
        if Command::new(candidate).arg("--version").output().is_ok() {
            return Some((*candidate).to_string());
        }
    }
    None
}

fn lower_and_typecheck_source(source: &str) -> Result<(MirModule, Vec<Item>)> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .map_err(|e| LustError::Unknown(format!("Lexer error: {e}")))?;
    let items = Parser::new(tokens)
        .parse()
        .map_err(|e| LustError::Unknown(format!("Parser error: {e}")))?;

    let config = crate::config::LustConfig::default();
    let mut checker = crate::typechecker::TypeChecker::with_config(&config);
    if let Err(e) = checker.check_module(&items) {
        let msg = e.to_string();
        // `TypeChecker::check_lust_perform` emits `TypeKind::Infer` (`'_'`) and
        // `TypeChecker::check_lust_resume` hardcodes `TypeKind::Unit` (`'unit'` / `'()'`)
        // instead of `BodyRet` (cf. `src/typechecker/expr_checker/operations.rs:548,571`).
        // Narrow suppression strictly to diagnostics whose span originates inside a
        // `handle..with` block (or desugared `__lust_perform`/`__lust_resume` call site)
        // so real user type errors outside `handle` are NEVER masked (`AOT-BUDGET-SOUND`).
        let is_genuine_effect_error = msg.contains("outside of an active effect handler arm")
            || msg.contains("incompatible with handle body return type")
            || msg.contains("Unsatisfied requirement");
        let effect_spans = collect_effect_handle_line_ranges(&items);
        let err_in_effect_span = match &e {
            LustError::TypeErrorWithSpan { line, .. } => effect_spans
                .iter()
                .any(|(start_line, end_line)| *line >= *start_line && *line <= *end_line),
            _ => false,
        };
        let is_effect_infer_gap = !is_genuine_effect_error
            && err_in_effect_span
            && ((items_contain_effect_handle(&items) && msg.contains("'_'"))
                || (items_contain_nontail_effect_handle(&items)
                    && (msg.contains("'unit'") || msg.contains("'()'") || msg.contains("'_'"))));
        if !is_effect_infer_gap {
            return Err(LustError::Unknown(format!("TypeChecker error: {e}")));
        }
    }

    let prepared_items = prepare_items_for_mir(&items);
    let rust_aot_items = desugar_effects_for_rust_aot(&prepared_items);
    let mut module = AstLowering::lower_module_items("main_module", &rust_aot_items)
        .map_err(|e| LustError::Unknown(format!("MIR lowering error: {e}")))?;
    register_ast_struct_layouts(&mut module, &items);
    Ok((module, items))
}

/// Prepares already-parsed/typechecked AST items (including effect/scope desugaring) and lowers
/// them into a `MirModule` ready for Rust AOT emission (`NG-W13-1C`).
pub fn prepare_and_lower_items_for_rust_aot(items: &[Item]) -> Result<MirModule> {
    let prepared_items = prepare_items_for_mir(items);
    let rust_aot_items = desugar_effects_for_rust_aot(&prepared_items);
    let mut module = AstLowering::lower_module_items("main_module", &rust_aot_items)
        .map_err(|e| LustError::Unknown(format!("MIR lowering error: {e}")))?;
    register_ast_struct_layouts(&mut module, items);
    Ok(module)
}

/// Ensures every AST `struct` declaration (including non-`@repr` structs and structs nested inside
/// `ItemKind::Module`) has its `MirStructLayout` registered in `module.struct_layouts` so
/// `Struct.decode(raw)` works uniformly across all AOT backends (`NG-W15-1`).
fn register_ast_struct_layouts(module: &mut MirModule, items: &[Item]) {
    for item in items {
        match &item.kind {
            ItemKind::Struct(def) => {
                if !module
                    .struct_layouts
                    .iter()
                    .any(|l| l.type_name == def.name)
                {
                    let layout = crate::data::materialize::struct_def_to_layout(def);
                    module.register_struct_layout(crate::mir::ir::MirStructLayout::from(&layout));
                }
            }
            ItemKind::Module { items: sub, .. } => {
                register_ast_struct_layouts(module, sub);
            }
            _ => {}
        }
    }
}

/// Collects inclusive `(start_line, end_line)` ranges for every `handle..with` (`__lust_handle`)
/// expression in `items` so `lower_and_typecheck_source` only permits the known effect-inference
/// typechecker gap inside actual `handle..with` spans (`AOT-BUDGET-SOUND`).
fn collect_effect_handle_line_ranges(items: &[Item]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    for item in items {
        match &item.kind {
            ItemKind::Function(f) => collect_effect_handle_ranges_in_stmts(&f.body, &mut ranges),
            ItemKind::Impl(imp) => {
                for m in &imp.methods {
                    collect_effect_handle_ranges_in_stmts(&m.body, &mut ranges);
                }
            }
            ItemKind::Script(stmts) => collect_effect_handle_ranges_in_stmts(stmts, &mut ranges),
            ItemKind::Module { items: sub, .. } => {
                ranges.extend(collect_effect_handle_line_ranges(sub));
            }
            _ => {}
        }
    }
    ranges
}

fn expand_span_line_bounds(span: Span, min_line: &mut usize, max_line: &mut usize) {
    if span.start_line > 0 {
        *min_line = (*min_line).min(span.start_line);
        *max_line = (*max_line).max(span.end_line.max(span.start_line));
    }
}

fn collect_expr_line_bounds(expr: &crate::ast::Expr, min_line: &mut usize, max_line: &mut usize) {
    expand_span_line_bounds(expr.span, min_line, max_line);
    match &expr.kind {
        crate::ast::ExprKind::Call { callee, args, .. } => {
            collect_expr_line_bounds(callee, min_line, max_line);
            for a in args {
                collect_expr_line_bounds(a, min_line, max_line);
            }
        }
        crate::ast::ExprKind::MethodCall { receiver, args, .. } => {
            collect_expr_line_bounds(receiver, min_line, max_line);
            for a in args {
                collect_expr_line_bounds(a, min_line, max_line);
            }
        }
        crate::ast::ExprKind::Binary { left, right, .. }
        | crate::ast::ExprKind::Range {
            start: left,
            end: right,
            ..
        } => {
            collect_expr_line_bounds(left, min_line, max_line);
            collect_expr_line_bounds(right, min_line, max_line);
        }
        crate::ast::ExprKind::Unary { operand: inner, .. }
        | crate::ast::ExprKind::FieldAccess { object: inner, .. }
        | crate::ast::ExprKind::Paren(inner)
        | crate::ast::ExprKind::Cast { expr: inner, .. }
        | crate::ast::ExprKind::TypeCheck { expr: inner, .. }
        | crate::ast::ExprKind::IsPattern { expr: inner, .. }
        | crate::ast::ExprKind::Lambda { body: inner, .. } => {
            collect_expr_line_bounds(inner, min_line, max_line);
        }
        crate::ast::ExprKind::Index { object, index } => {
            collect_expr_line_bounds(object, min_line, max_line);
            collect_expr_line_bounds(index, min_line, max_line);
        }
        crate::ast::ExprKind::Array(es)
        | crate::ast::ExprKind::Tuple(es)
        | crate::ast::ExprKind::EnumConstructor { args: es, .. }
        | crate::ast::ExprKind::Return(es) => {
            for e in es {
                collect_expr_line_bounds(e, min_line, max_line);
            }
        }
        crate::ast::ExprKind::Map(pairs) => {
            for (k, v) in pairs {
                collect_expr_line_bounds(k, min_line, max_line);
                collect_expr_line_bounds(v, min_line, max_line);
            }
        }
        crate::ast::ExprKind::StructLiteral { fields, .. } => {
            for f in fields {
                collect_expr_line_bounds(&f.value, min_line, max_line);
            }
        }
        crate::ast::ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_expr_line_bounds(condition, min_line, max_line);
            collect_expr_line_bounds(then_branch, min_line, max_line);
            if let Some(eb) = else_branch {
                collect_expr_line_bounds(eb, min_line, max_line);
            }
        }
        crate::ast::ExprKind::Block(b) | crate::ast::ExprKind::EffectBlock(b) => {
            for s in b {
                collect_stmt_line_bounds(s, min_line, max_line);
            }
        }
        _ => {}
    }
}

fn collect_stmt_line_bounds(stmt: &crate::ast::Stmt, min_line: &mut usize, max_line: &mut usize) {
    expand_span_line_bounds(stmt.span, min_line, max_line);
    match &stmt.kind {
        crate::ast::StmtKind::Local {
            initializer: Some(es),
            ..
        }
        | crate::ast::StmtKind::Return(es) => {
            for e in es {
                collect_expr_line_bounds(e, min_line, max_line);
            }
        }
        crate::ast::StmtKind::Assign { targets, values } => {
            for e in targets.iter().chain(values.iter()) {
                collect_expr_line_bounds(e, min_line, max_line);
            }
        }
        crate::ast::StmtKind::CompoundAssign { target, value, .. } => {
            collect_expr_line_bounds(target, min_line, max_line);
            collect_expr_line_bounds(value, min_line, max_line);
        }
        crate::ast::StmtKind::Expr(e) => collect_expr_line_bounds(e, min_line, max_line),
        crate::ast::StmtKind::If {
            condition,
            then_block,
            elseif_branches,
            else_block,
        } => {
            collect_expr_line_bounds(condition, min_line, max_line);
            for s in then_block {
                collect_stmt_line_bounds(s, min_line, max_line);
            }
            for (c, b) in elseif_branches {
                collect_expr_line_bounds(c, min_line, max_line);
                for s in b {
                    collect_stmt_line_bounds(s, min_line, max_line);
                }
            }
            if let Some(b) = else_block {
                for s in b {
                    collect_stmt_line_bounds(s, min_line, max_line);
                }
            }
        }
        crate::ast::StmtKind::While { condition, body } => {
            collect_expr_line_bounds(condition, min_line, max_line);
            for s in body {
                collect_stmt_line_bounds(s, min_line, max_line);
            }
        }
        crate::ast::StmtKind::ForNumeric {
            start,
            end,
            step,
            body,
            ..
        } => {
            collect_expr_line_bounds(start, min_line, max_line);
            collect_expr_line_bounds(end, min_line, max_line);
            if let Some(st) = step {
                collect_expr_line_bounds(st, min_line, max_line);
            }
            for s in body {
                collect_stmt_line_bounds(s, min_line, max_line);
            }
        }
        crate::ast::StmtKind::ForIn { iterator, body, .. } => {
            collect_expr_line_bounds(iterator, min_line, max_line);
            for s in body {
                collect_stmt_line_bounds(s, min_line, max_line);
            }
        }
        crate::ast::StmtKind::Block(b) | crate::ast::StmtKind::Scoped(b) => {
            for s in b {
                collect_stmt_line_bounds(s, min_line, max_line);
            }
        }
        crate::ast::StmtKind::Defer(s) | crate::ast::StmtKind::ErrDefer(s) => {
            collect_stmt_line_bounds(s, min_line, max_line);
        }
        _ => {}
    }
}

fn collect_effect_handle_ranges_in_stmts(
    stmts: &[crate::ast::Stmt],
    out: &mut Vec<(usize, usize)>,
) {
    for stmt in stmts {
        match &stmt.kind {
            crate::ast::StmtKind::Local {
                initializer: Some(es),
                ..
            }
            | crate::ast::StmtKind::Return(es) => {
                for e in es {
                    collect_effect_handle_ranges_in_expr(e, out);
                }
            }
            crate::ast::StmtKind::Assign { targets, values } => {
                for e in targets.iter().chain(values.iter()) {
                    collect_effect_handle_ranges_in_expr(e, out);
                }
            }
            crate::ast::StmtKind::CompoundAssign { target, value, .. } => {
                collect_effect_handle_ranges_in_expr(target, out);
                collect_effect_handle_ranges_in_expr(value, out);
            }
            crate::ast::StmtKind::Expr(e) => collect_effect_handle_ranges_in_expr(e, out),
            crate::ast::StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                collect_effect_handle_ranges_in_expr(condition, out);
                collect_effect_handle_ranges_in_stmts(then_block, out);
                for (c, b) in elseif_branches {
                    collect_effect_handle_ranges_in_expr(c, out);
                    collect_effect_handle_ranges_in_stmts(b, out);
                }
                if let Some(b) = else_block {
                    collect_effect_handle_ranges_in_stmts(b, out);
                }
            }
            crate::ast::StmtKind::While { condition, body } => {
                collect_effect_handle_ranges_in_expr(condition, out);
                collect_effect_handle_ranges_in_stmts(body, out);
            }
            crate::ast::StmtKind::ForNumeric {
                start,
                end,
                step,
                body,
                ..
            } => {
                collect_effect_handle_ranges_in_expr(start, out);
                collect_effect_handle_ranges_in_expr(end, out);
                if let Some(st) = step {
                    collect_effect_handle_ranges_in_expr(st, out);
                }
                collect_effect_handle_ranges_in_stmts(body, out);
            }
            crate::ast::StmtKind::ForIn { iterator, body, .. } => {
                collect_effect_handle_ranges_in_expr(iterator, out);
                collect_effect_handle_ranges_in_stmts(body, out);
            }
            crate::ast::StmtKind::Block(b) | crate::ast::StmtKind::Scoped(b) => {
                collect_effect_handle_ranges_in_stmts(b, out);
            }
            crate::ast::StmtKind::Defer(s) | crate::ast::StmtKind::ErrDefer(s) => {
                collect_effect_handle_ranges_in_stmts(std::slice::from_ref(s.as_ref()), out);
            }
            _ => {}
        }
    }
}

fn collect_effect_handle_ranges_in_expr(expr: &crate::ast::Expr, out: &mut Vec<(usize, usize)>) {
    if is_handle_call_expr(expr) {
        let mut min_line = usize::MAX;
        let mut max_line = 0usize;
        collect_expr_line_bounds(expr, &mut min_line, &mut max_line);
        if min_line <= max_line {
            out.push((min_line, max_line));
        }
        return;
    }
    match &expr.kind {
        crate::ast::ExprKind::Call { callee, args, .. } => {
            if matches!(&callee.kind, crate::ast::ExprKind::Identifier(n) if n == "__lust_perform" || n == "perform" || n == "__lust_resume" || n == "resume")
                && expr.span.start_line > 0
            {
                out.push((
                    expr.span.start_line,
                    expr.span.end_line.max(expr.span.start_line),
                ));
            }
            collect_effect_handle_ranges_in_expr(callee, out);
            for a in args {
                collect_effect_handle_ranges_in_expr(a, out);
            }
        }
        crate::ast::ExprKind::MethodCall { receiver, args, .. } => {
            collect_effect_handle_ranges_in_expr(receiver, out);
            for a in args {
                collect_effect_handle_ranges_in_expr(a, out);
            }
        }
        crate::ast::ExprKind::Binary { left, right, .. }
        | crate::ast::ExprKind::Range {
            start: left,
            end: right,
            ..
        } => {
            collect_effect_handle_ranges_in_expr(left, out);
            collect_effect_handle_ranges_in_expr(right, out);
        }
        crate::ast::ExprKind::Unary { operand: inner, .. }
        | crate::ast::ExprKind::FieldAccess { object: inner, .. }
        | crate::ast::ExprKind::Paren(inner)
        | crate::ast::ExprKind::Cast { expr: inner, .. }
        | crate::ast::ExprKind::TypeCheck { expr: inner, .. }
        | crate::ast::ExprKind::IsPattern { expr: inner, .. }
        | crate::ast::ExprKind::Lambda { body: inner, .. } => {
            collect_effect_handle_ranges_in_expr(inner, out);
        }
        crate::ast::ExprKind::Index { object, index } => {
            collect_effect_handle_ranges_in_expr(object, out);
            collect_effect_handle_ranges_in_expr(index, out);
        }
        crate::ast::ExprKind::Array(es)
        | crate::ast::ExprKind::Tuple(es)
        | crate::ast::ExprKind::EnumConstructor { args: es, .. }
        | crate::ast::ExprKind::Return(es) => {
            for e in es {
                collect_effect_handle_ranges_in_expr(e, out);
            }
        }
        crate::ast::ExprKind::Map(pairs) => {
            for (k, v) in pairs {
                collect_effect_handle_ranges_in_expr(k, out);
                collect_effect_handle_ranges_in_expr(v, out);
            }
        }
        crate::ast::ExprKind::StructLiteral { fields, .. } => {
            for f in fields {
                collect_effect_handle_ranges_in_expr(&f.value, out);
            }
        }
        crate::ast::ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_effect_handle_ranges_in_expr(condition, out);
            collect_effect_handle_ranges_in_expr(then_branch, out);
            if let Some(eb) = else_branch {
                collect_effect_handle_ranges_in_expr(eb, out);
            }
        }
        crate::ast::ExprKind::Block(b) | crate::ast::ExprKind::EffectBlock(b) => {
            collect_effect_handle_ranges_in_stmts(b, out);
        }
        _ => {}
    }
}

#[derive(Default)]
struct CpsModuleCtx {
    helpers: std::collections::HashMap<String, FunctionDef>,
    generated_fns: Vec<FunctionDef>,
    loop_counter: usize,
    current_fn_ret_ty: Option<Type>,
    current_fn_param_tys: std::collections::HashMap<String, Type>,
}

thread_local! {
    static CPS_MODULE_CTX: std::cell::RefCell<CpsModuleCtx> =
        std::cell::RefCell::new(CpsModuleCtx::default());
}

fn collect_module_helpers(items: &[Item], out: &mut std::collections::HashMap<String, FunctionDef>) {
    for item in items {
        match &item.kind {
            ItemKind::Function(f) => {
                out.insert(f.name.clone(), f.clone());
            }
            ItemKind::Module { items: sub, .. } => {
                collect_module_helpers(sub, out);
            }
            _ => {}
        }
    }
}

/// Pre-lowers Lust `EffectBlock` / `:provide(..)` / `.provide(..)` / `:run_sync()` / `.run_sync()`
/// and implicit tail `handle..with` expressions into MIR-lowerable AST forms for Rust AOT
/// (`NG-W13-1A`, `F-19`, `NG-W16-2`) without modifying `src/mir/ir.rs` or `src/mir/lowering/mod.rs`.
fn desugar_effects_for_rust_aot(items: &[Item]) -> Vec<Item> {
    let mut helpers = std::collections::HashMap::new();
    collect_module_helpers(items, &mut helpers);
    CPS_MODULE_CTX.with(|ctx| {
        let mut c = ctx.borrow_mut();
        c.helpers = helpers;
        c.generated_fns.clear();
        c.loop_counter = 0;
        c.current_fn_ret_ty = None;
        c.current_fn_param_tys.clear();
    });

    let mut out = desugar_items_effects_inner(items);

    // Drain any module-level `__cps_loop_step_*` functions generated by non-tail loop CPS lowering,
    // desugaring each one in turn until a fixed point is reached.
    loop {
        let pending: Vec<FunctionDef> = CPS_MODULE_CTX.with(|ctx| {
            let mut c = ctx.borrow_mut();
            std::mem::take(&mut c.generated_fns)
        });
        if pending.is_empty() {
            break;
        }
        for gen_fn in pending {
            let desugared = desugar_function_effects(&gen_fn);
            out.push(Item {
                kind: ItemKind::Function(desugared),
                span: Span::dummy(),
            });
        }
    }

    out
}

fn desugar_items_effects_inner(items: &[Item]) -> Vec<Item> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let new_kind = match &item.kind {
            ItemKind::Function(f) => ItemKind::Function(desugar_function_effects(f)),
            ItemKind::Impl(imp) => {
                let mut new_imp = imp.clone();
                new_imp.methods = imp.methods.iter().map(desugar_function_effects).collect();
                ItemKind::Impl(new_imp)
            }
            ItemKind::Module { name, items: sub } => ItemKind::Module {
                name: name.clone(),
                items: desugar_items_effects_inner(sub),
            },
            other => other.clone(),
        };
        out.push(Item {
            kind: new_kind,
            span: item.span,
        });
    }
    out
}

fn desugar_function_effects(func: &FunctionDef) -> FunctionDef {
    use std::collections::HashSet;

    let mut param_tys = std::collections::HashMap::new();
    for p in &func.params {
        param_tys.insert(p.name.clone(), p.ty.clone());
    }
    let prev_ctx = CPS_MODULE_CTX.with(|ctx| {
        let mut c = ctx.borrow_mut();
        let prev = (c.current_fn_ret_ty.clone(), std::mem::replace(&mut c.current_fn_param_tys, param_tys));
        c.current_fn_ret_ty = func.return_type.clone();
        prev
    });

    let mut effect_vars: HashSet<String> = HashSet::new();
    for p in &func.params {
        if matches!(p.ty.kind, TypeKind::Effect { .. }) {
            effect_vars.insert(p.name.clone());
        }
    }

    // Identify local `effect do ... end` bindings that are used with `.provide`/`:provide` or escape
    let mut provided_or_escaped_locals: HashSet<String> = HashSet::new();
    collect_effect_vars_needing_closure(&func.body, &mut provided_or_escaped_locals);

    let mut new_body = desugar_stmts_effects(
        &func.body,
        &mut effect_vars,
        &provided_or_escaped_locals,
        false,
        &mut 0usize,
    );

    CPS_MODULE_CTX.with(|ctx| {
        let mut c = ctx.borrow_mut();
        c.current_fn_ret_ty = prev_ctx.0;
        c.current_fn_param_tys = prev_ctx.1;
    });

    // If a non-unit function ends with a bare `__lust_handle(...)` or `__lust_nontail_handle(...)`
    // expression statement (as in `function main(): int handle { ... } with { ... } end`),
    // turn it into an explicit return.
    let returns_value = func
        .return_type
        .as_ref()
        .is_some_and(|rt| !matches!(rt.kind, TypeKind::Unit));
    if returns_value
        && let Some(last) = new_body.last_mut()
        && let crate::ast::StmtKind::Expr(expr) = &last.kind
        && is_handle_call_expr(expr)
    {
        let span = last.span;
        let expr_owned = expr.clone();
        last.kind = crate::ast::StmtKind::Return(vec![expr_owned]);
        last.span = span;
    }

    let mut new_fn = func.clone();
    new_fn.body = new_body;
    new_fn
}

fn is_handle_call_expr(expr: &crate::ast::Expr) -> bool {
    match &expr.kind {
        crate::ast::ExprKind::Paren(inner) => is_handle_call_expr(inner),
        crate::ast::ExprKind::Call { callee, .. } => {
            matches!(&callee.kind, crate::ast::ExprKind::Identifier(n) if n == "__lust_handle" || n == "__lust_nontail_handle")
        }
        _ => false,
    }
}

fn items_contain_nontail_effect_handle(items: &[Item]) -> bool {
    for item in items {
        match &item.kind {
            ItemKind::Function(f) if stmts_contain_nontail_effect_handle(&f.body) => return true,
            ItemKind::Impl(imp)
                if imp
                    .methods
                    .iter()
                    .any(|m| stmts_contain_nontail_effect_handle(&m.body)) =>
            {
                return true;
            }
            ItemKind::Script(stmts) if stmts_contain_nontail_effect_handle(stmts) => return true,
            ItemKind::Module { items: sub, .. } if items_contain_nontail_effect_handle(sub) => {
                return true;
            }
            _ => {}
        }
    }
    false
}

fn stmts_contain_nontail_effect_handle(stmts: &[crate::ast::Stmt]) -> bool {
    for stmt in stmts {
        let found = match &stmt.kind {
            crate::ast::StmtKind::Local {
                initializer: Some(es),
                ..
            }
            | crate::ast::StmtKind::Return(es) => es.iter().any(expr_contains_nontail_effect_handle),
            crate::ast::StmtKind::Assign { targets, values } => {
                targets.iter().any(expr_contains_nontail_effect_handle)
                    || values.iter().any(expr_contains_nontail_effect_handle)
            }
            crate::ast::StmtKind::CompoundAssign { target, value, .. } => {
                expr_contains_nontail_effect_handle(target)
                    || expr_contains_nontail_effect_handle(value)
            }
            crate::ast::StmtKind::Expr(e) => expr_contains_nontail_effect_handle(e),
            crate::ast::StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                expr_contains_nontail_effect_handle(condition)
                    || stmts_contain_nontail_effect_handle(then_block)
                    || elseif_branches.iter().any(|(c, b)| {
                        expr_contains_nontail_effect_handle(c)
                            || stmts_contain_nontail_effect_handle(b)
                    })
                    || else_block
                        .as_ref()
                        .is_some_and(|b| stmts_contain_nontail_effect_handle(b))
            }
            crate::ast::StmtKind::While { condition, body } => {
                expr_contains_nontail_effect_handle(condition)
                    || stmts_contain_nontail_effect_handle(body)
            }
            crate::ast::StmtKind::ForNumeric {
                start,
                end,
                step,
                body,
                ..
            } => {
                expr_contains_nontail_effect_handle(start)
                    || expr_contains_nontail_effect_handle(end)
                    || step
                        .as_ref()
                        .is_some_and(expr_contains_nontail_effect_handle)
                    || stmts_contain_nontail_effect_handle(body)
            }
            crate::ast::StmtKind::ForIn { iterator, body, .. } => {
                expr_contains_nontail_effect_handle(iterator)
                    || stmts_contain_nontail_effect_handle(body)
            }
            crate::ast::StmtKind::Block(b) | crate::ast::StmtKind::Scoped(b) => {
                stmts_contain_nontail_effect_handle(b)
            }
            crate::ast::StmtKind::Defer(s) | crate::ast::StmtKind::ErrDefer(s) => {
                stmts_contain_nontail_effect_handle(std::slice::from_ref(s.as_ref()))
            }
            _ => false,
        };
        if found {
            return true;
        }
    }
    false
}

fn handle_arms_have_non_tail(arms_expr: &crate::ast::Expr) -> bool {
    let crate::ast::ExprKind::Array(tuples) = &arms_expr.kind else {
        return false;
    };
    for tup in tuples {
        if let crate::ast::ExprKind::Tuple(elems) = &tup.kind
            && elems.len() == 2
            && let crate::ast::ExprKind::Lambda { body, .. } = &elems[1].kind
            && super::rust_emitter::classify_handler_arm_expr(body)
                == super::rust_emitter::HandlerArmTailKind::NonTailOrMultiShot
        {
            return true;
        }
    }
    false
}

fn expr_contains_nontail_effect_handle(expr: &crate::ast::Expr) -> bool {
    if let crate::ast::ExprKind::Call { callee, args, .. } = &expr.kind
        && matches!(&callee.kind, crate::ast::ExprKind::Identifier(n) if n == "__lust_handle")
        && args.len() == 2
        && handle_arms_have_non_tail(&args[1])
    {
        return true;
    }
    match &expr.kind {
        crate::ast::ExprKind::Call { callee, args, .. } => {
            expr_contains_nontail_effect_handle(callee)
                || args.iter().any(expr_contains_nontail_effect_handle)
        }
        crate::ast::ExprKind::MethodCall { receiver, args, .. } => {
            expr_contains_nontail_effect_handle(receiver)
                || args.iter().any(expr_contains_nontail_effect_handle)
        }
        crate::ast::ExprKind::Binary { left, right, .. }
        | crate::ast::ExprKind::Range {
            start: left,
            end: right,
            ..
        } => {
            expr_contains_nontail_effect_handle(left) || expr_contains_nontail_effect_handle(right)
        }
        crate::ast::ExprKind::Unary { operand: inner, .. }
        | crate::ast::ExprKind::FieldAccess { object: inner, .. }
        | crate::ast::ExprKind::Paren(inner)
        | crate::ast::ExprKind::Cast { expr: inner, .. }
        | crate::ast::ExprKind::TypeCheck { expr: inner, .. }
        | crate::ast::ExprKind::IsPattern { expr: inner, .. }
        | crate::ast::ExprKind::Lambda { body: inner, .. } => {
            expr_contains_nontail_effect_handle(inner)
        }
        crate::ast::ExprKind::Index { object, index } => {
            expr_contains_nontail_effect_handle(object)
                || expr_contains_nontail_effect_handle(index)
        }
        crate::ast::ExprKind::Array(es)
        | crate::ast::ExprKind::Tuple(es)
        | crate::ast::ExprKind::EnumConstructor { args: es, .. }
        | crate::ast::ExprKind::Return(es) => es.iter().any(expr_contains_nontail_effect_handle),
        crate::ast::ExprKind::Map(pairs) => pairs.iter().any(|(k, v)| {
            expr_contains_nontail_effect_handle(k) || expr_contains_nontail_effect_handle(v)
        }),
        crate::ast::ExprKind::StructLiteral { fields, .. } => fields
            .iter()
            .any(|f| expr_contains_nontail_effect_handle(&f.value)),
        crate::ast::ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            expr_contains_nontail_effect_handle(condition)
                || expr_contains_nontail_effect_handle(then_branch)
                || else_branch
                    .as_deref()
                    .is_some_and(expr_contains_nontail_effect_handle)
        }
        crate::ast::ExprKind::Block(b) | crate::ast::ExprKind::EffectBlock(b) => {
            stmts_contain_nontail_effect_handle(b)
        }
        _ => false,
    }
}

fn items_contain_effect_handle(items: &[Item]) -> bool {
    for item in items {
        match &item.kind {
            ItemKind::Function(f) if stmts_contain_effect_handle(&f.body) => return true,
            ItemKind::Impl(imp)
                if imp
                    .methods
                    .iter()
                    .any(|m| stmts_contain_effect_handle(&m.body)) =>
            {
                return true;
            }
            ItemKind::Script(stmts) if stmts_contain_effect_handle(stmts) => return true,
            ItemKind::Module { items: sub, .. } if items_contain_effect_handle(sub) => return true,
            _ => {}
        }
    }
    false
}

fn stmts_contain_effect_handle(stmts: &[crate::ast::Stmt]) -> bool {
    for stmt in stmts {
        let found = match &stmt.kind {
            crate::ast::StmtKind::Local {
                initializer: Some(es),
                ..
            }
            | crate::ast::StmtKind::Return(es) => es.iter().any(expr_contains_effect_handle),
            crate::ast::StmtKind::Assign { targets, values } => {
                targets.iter().any(expr_contains_effect_handle)
                    || values.iter().any(expr_contains_effect_handle)
            }
            crate::ast::StmtKind::CompoundAssign { target, value, .. } => {
                expr_contains_effect_handle(target) || expr_contains_effect_handle(value)
            }
            crate::ast::StmtKind::Expr(e) => expr_contains_effect_handle(e),
            crate::ast::StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                expr_contains_effect_handle(condition)
                    || stmts_contain_effect_handle(then_block)
                    || elseif_branches.iter().any(|(c, b)| {
                        expr_contains_effect_handle(c) || stmts_contain_effect_handle(b)
                    })
                    || else_block
                        .as_ref()
                        .is_some_and(|b| stmts_contain_effect_handle(b))
            }
            crate::ast::StmtKind::While { condition, body } => {
                expr_contains_effect_handle(condition) || stmts_contain_effect_handle(body)
            }
            crate::ast::StmtKind::ForNumeric {
                start,
                end,
                step,
                body,
                ..
            } => {
                expr_contains_effect_handle(start)
                    || expr_contains_effect_handle(end)
                    || step.as_ref().is_some_and(expr_contains_effect_handle)
                    || stmts_contain_effect_handle(body)
            }
            crate::ast::StmtKind::ForIn { iterator, body, .. } => {
                expr_contains_effect_handle(iterator) || stmts_contain_effect_handle(body)
            }
            crate::ast::StmtKind::Block(b) | crate::ast::StmtKind::Scoped(b) => {
                stmts_contain_effect_handle(b)
            }
            crate::ast::StmtKind::Defer(s) | crate::ast::StmtKind::ErrDefer(s) => {
                stmts_contain_effect_handle(std::slice::from_ref(s.as_ref()))
            }
            _ => false,
        };
        if found {
            return true;
        }
    }
    false
}

fn expr_contains_effect_handle(expr: &crate::ast::Expr) -> bool {
    if is_handle_call_expr(expr) {
        return true;
    }
    match &expr.kind {
        crate::ast::ExprKind::Call { callee, args, .. } => {
            expr_contains_effect_handle(callee) || args.iter().any(expr_contains_effect_handle)
        }
        crate::ast::ExprKind::MethodCall { receiver, args, .. } => {
            expr_contains_effect_handle(receiver) || args.iter().any(expr_contains_effect_handle)
        }
        crate::ast::ExprKind::Binary { left, right, .. }
        | crate::ast::ExprKind::Range {
            start: left,
            end: right,
            ..
        } => expr_contains_effect_handle(left) || expr_contains_effect_handle(right),
        crate::ast::ExprKind::Unary { operand: inner, .. }
        | crate::ast::ExprKind::FieldAccess { object: inner, .. }
        | crate::ast::ExprKind::Paren(inner)
        | crate::ast::ExprKind::Cast { expr: inner, .. }
        | crate::ast::ExprKind::TypeCheck { expr: inner, .. }
        | crate::ast::ExprKind::IsPattern { expr: inner, .. }
        | crate::ast::ExprKind::Lambda { body: inner, .. } => expr_contains_effect_handle(inner),
        crate::ast::ExprKind::Index { object, index } => {
            expr_contains_effect_handle(object) || expr_contains_effect_handle(index)
        }
        crate::ast::ExprKind::Array(es)
        | crate::ast::ExprKind::Tuple(es)
        | crate::ast::ExprKind::EnumConstructor { args: es, .. }
        | crate::ast::ExprKind::Return(es) => es.iter().any(expr_contains_effect_handle),
        crate::ast::ExprKind::Map(pairs) => pairs
            .iter()
            .any(|(k, v)| expr_contains_effect_handle(k) || expr_contains_effect_handle(v)),
        crate::ast::ExprKind::StructLiteral { fields, .. } => {
            fields.iter().any(|f| expr_contains_effect_handle(&f.value))
        }
        crate::ast::ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            expr_contains_effect_handle(condition)
                || expr_contains_effect_handle(then_branch)
                || else_branch
                    .as_deref()
                    .is_some_and(expr_contains_effect_handle)
        }
        crate::ast::ExprKind::Block(b) | crate::ast::ExprKind::EffectBlock(b) => {
            stmts_contain_effect_handle(b)
        }
        _ => false,
    }
}

fn collect_effect_vars_needing_closure(
    stmts: &[crate::ast::Stmt],
    out: &mut std::collections::HashSet<String>,
) {
    for stmt in stmts {
        match &stmt.kind {
            crate::ast::StmtKind::Local {
                initializer: Some(es),
                ..
            } => {
                for e in es {
                    collect_effect_vars_in_expr(e, false, out);
                }
            }
            crate::ast::StmtKind::Return(es) => {
                for e in es {
                    collect_effect_vars_in_expr(e, true, out);
                }
            }
            crate::ast::StmtKind::Assign { targets, values } => {
                for e in targets.iter().chain(values.iter()) {
                    collect_effect_vars_in_expr(e, false, out);
                }
            }
            crate::ast::StmtKind::CompoundAssign { target, value, .. } => {
                collect_effect_vars_in_expr(target, false, out);
                collect_effect_vars_in_expr(value, false, out);
            }
            crate::ast::StmtKind::Expr(e) => collect_effect_vars_in_expr(e, false, out),
            crate::ast::StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                collect_effect_vars_in_expr(condition, false, out);
                collect_effect_vars_needing_closure(then_block, out);
                for (ec, eb) in elseif_branches {
                    collect_effect_vars_in_expr(ec, false, out);
                    collect_effect_vars_needing_closure(eb, out);
                }
                if let Some(eb) = else_block {
                    collect_effect_vars_needing_closure(eb, out);
                }
            }
            crate::ast::StmtKind::While { condition, body } => {
                collect_effect_vars_in_expr(condition, false, out);
                collect_effect_vars_needing_closure(body, out);
            }
            crate::ast::StmtKind::ForNumeric {
                start,
                end,
                step,
                body,
                ..
            } => {
                collect_effect_vars_in_expr(start, false, out);
                collect_effect_vars_in_expr(end, false, out);
                if let Some(st) = step {
                    collect_effect_vars_in_expr(st, false, out);
                }
                collect_effect_vars_needing_closure(body, out);
            }
            crate::ast::StmtKind::ForIn { iterator, body, .. } => {
                collect_effect_vars_in_expr(iterator, false, out);
                collect_effect_vars_needing_closure(body, out);
            }
            crate::ast::StmtKind::Block(b) | crate::ast::StmtKind::Scoped(b) => {
                collect_effect_vars_needing_closure(b, out);
            }
            crate::ast::StmtKind::Defer(inner) | crate::ast::StmtKind::ErrDefer(inner) => {
                collect_effect_vars_needing_closure(std::slice::from_ref(inner.as_ref()), out);
            }
            _ => {}
        }
    }
}

fn collect_effect_vars_in_expr(
    expr: &crate::ast::Expr,
    escaping: bool,
    out: &mut std::collections::HashSet<String>,
) {
    match &expr.kind {
        crate::ast::ExprKind::Identifier(name) if escaping => {
            out.insert(name.clone());
        }
        crate::ast::ExprKind::MethodCall {
            receiver,
            method,
            args,
            ..
        } => {
            if method == "provide"
                && let crate::ast::ExprKind::Identifier(name) = &receiver.kind
            {
                out.insert(name.clone());
            }
            collect_effect_vars_in_expr(receiver, false, out);
            for a in args {
                collect_effect_vars_in_expr(a, true, out);
            }
        }
        crate::ast::ExprKind::Call { callee, args, .. } => {
            if let crate::ast::ExprKind::FieldAccess { object, field } = &callee.kind
                && matches!(field.as_str(), "provide" | "run_sync")
                && let crate::ast::ExprKind::Identifier(name) = &object.kind
            {
                out.insert(name.clone());
            }
            collect_effect_vars_in_expr(callee, false, out);
            for a in args {
                collect_effect_vars_in_expr(a, true, out);
            }
        }
        crate::ast::ExprKind::Binary { left, right, .. }
        | crate::ast::ExprKind::Range {
            start: left,
            end: right,
            ..
        } => {
            collect_effect_vars_in_expr(left, false, out);
            collect_effect_vars_in_expr(right, false, out);
        }
        crate::ast::ExprKind::Unary { operand: inner, .. }
        | crate::ast::ExprKind::FieldAccess { object: inner, .. }
        | crate::ast::ExprKind::Paren(inner)
        | crate::ast::ExprKind::Cast { expr: inner, .. }
        | crate::ast::ExprKind::TypeCheck { expr: inner, .. }
        | crate::ast::ExprKind::IsPattern { expr: inner, .. }
        | crate::ast::ExprKind::Lambda { body: inner, .. } => {
            collect_effect_vars_in_expr(inner, escaping, out);
        }
        crate::ast::ExprKind::Index { object, index } => {
            collect_effect_vars_in_expr(object, false, out);
            collect_effect_vars_in_expr(index, false, out);
        }
        crate::ast::ExprKind::Array(es)
        | crate::ast::ExprKind::Tuple(es)
        | crate::ast::ExprKind::EnumConstructor { args: es, .. }
        | crate::ast::ExprKind::Return(es) => {
            for e in es {
                collect_effect_vars_in_expr(e, false, out);
            }
        }
        crate::ast::ExprKind::Map(pairs) => {
            for (k, v) in pairs {
                collect_effect_vars_in_expr(k, false, out);
                collect_effect_vars_in_expr(v, false, out);
            }
        }
        crate::ast::ExprKind::StructLiteral { fields, .. } => {
            for f in fields {
                collect_effect_vars_in_expr(&f.value, false, out);
            }
        }
        crate::ast::ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_effect_vars_in_expr(condition, false, out);
            collect_effect_vars_in_expr(then_branch, escaping, out);
            if let Some(eb) = else_branch {
                collect_effect_vars_in_expr(eb, escaping, out);
            }
        }
        crate::ast::ExprKind::Block(b) | crate::ast::ExprKind::EffectBlock(b) => {
            collect_effect_vars_needing_closure(b, out);
        }
        _ => {}
    }
}

fn desugar_stmts_effects(
    stmts: &[crate::ast::Stmt],
    effect_vars: &mut std::collections::HashSet<String>,
    needs_closure: &std::collections::HashSet<String>,
    in_scoped_block: bool,
    nt_counter: &mut usize,
) -> Vec<crate::ast::Stmt> {
    let mut out = Vec::with_capacity(stmts.len());
    for stmt in stmts {
        let new_kind = match &stmt.kind {
            crate::ast::StmtKind::Local {
                bindings,
                initializer,
            } => {
                // Keep plain `local e = effect do ... end` for direct `e:run_sync()` unless `e`
                // uses `.provide`/`:provide`/`.run_sync()` or escapes.
                if bindings.len() == 1
                    && let Some(inits) = initializer
                    && inits.len() == 1
                    && let crate::ast::ExprKind::EffectBlock(eb_stmts) = &inits[0].kind
                    && !needs_closure.contains(&bindings[0].name)
                {
                    let new_eb =
                        desugar_stmts_effects(eb_stmts, effect_vars, needs_closure, false, nt_counter);
                    crate::ast::StmtKind::Local {
                        bindings: bindings.clone(),
                        initializer: Some(vec![crate::ast::Expr::new(
                            crate::ast::ExprKind::EffectBlock(new_eb),
                            inits[0].span,
                        )]),
                    }
                } else {
                    let new_inits = initializer.as_ref().map(|es| {
                        es.iter()
                            .map(|e| {
                                desugar_expr_effects(e, effect_vars, needs_closure, nt_counter)
                            })
                            .collect::<Vec<_>>()
                    });
                    if bindings.len() == 1
                        && let Some(inits) = initializer
                        && inits.len() == 1
                        && (matches!(inits[0].kind, crate::ast::ExprKind::EffectBlock(_))
                            || expr_is_effect_value(&inits[0], effect_vars))
                    {
                        effect_vars.insert(bindings[0].name.clone());
                    }
                    crate::ast::StmtKind::Local {
                        bindings: bindings.clone(),
                        initializer: new_inits,
                    }
                }
            }
            crate::ast::StmtKind::Assign { targets, values } => crate::ast::StmtKind::Assign {
                targets: targets
                    .iter()
                    .map(|e| desugar_expr_effects(e, effect_vars, needs_closure, nt_counter))
                    .collect(),
                values: values
                    .iter()
                    .map(|e| desugar_expr_effects(e, effect_vars, needs_closure, nt_counter))
                    .collect(),
            },
            crate::ast::StmtKind::CompoundAssign { target, op, value } => {
                crate::ast::StmtKind::CompoundAssign {
                    target: desugar_expr_effects(target, effect_vars, needs_closure, nt_counter),
                    op: *op,
                    value: desugar_expr_effects(value, effect_vars, needs_closure, nt_counter),
                }
            }
            crate::ast::StmtKind::Expr(e) => crate::ast::StmtKind::Expr(desugar_expr_effects(
                e,
                effect_vars,
                needs_closure,
                nt_counter,
            )),
            crate::ast::StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => crate::ast::StmtKind::If {
                condition: desugar_expr_effects(condition, effect_vars, needs_closure, nt_counter),
                then_block: desugar_stmts_effects(
                    then_block,
                    effect_vars,
                    needs_closure,
                    in_scoped_block,
                    nt_counter,
                ),
                elseif_branches: elseif_branches
                    .iter()
                    .map(|(ec, eb)| {
                        (
                            desugar_expr_effects(ec, effect_vars, needs_closure, nt_counter),
                            desugar_stmts_effects(
                                eb,
                                effect_vars,
                                needs_closure,
                                in_scoped_block,
                                nt_counter,
                            ),
                        )
                    })
                    .collect(),
                else_block: else_block.as_ref().map(|eb| {
                    desugar_stmts_effects(
                        eb,
                        effect_vars,
                        needs_closure,
                        in_scoped_block,
                        nt_counter,
                    )
                }),
            },
            crate::ast::StmtKind::While { condition, body } => crate::ast::StmtKind::While {
                condition: desugar_expr_effects(condition, effect_vars, needs_closure, nt_counter),
                body: desugar_stmts_effects(
                    body,
                    effect_vars,
                    needs_closure,
                    in_scoped_block,
                    nt_counter,
                ),
            },
            crate::ast::StmtKind::ForNumeric {
                variable,
                start,
                end,
                step,
                body,
            } => crate::ast::StmtKind::ForNumeric {
                variable: variable.clone(),
                start: desugar_expr_effects(start, effect_vars, needs_closure, nt_counter),
                end: desugar_expr_effects(end, effect_vars, needs_closure, nt_counter),
                step: step
                    .as_ref()
                    .map(|s| desugar_expr_effects(s, effect_vars, needs_closure, nt_counter)),
                body: desugar_stmts_effects(
                    body,
                    effect_vars,
                    needs_closure,
                    in_scoped_block,
                    nt_counter,
                ),
            },
            crate::ast::StmtKind::ForIn {
                variables,
                iterator,
                body,
            } => crate::ast::StmtKind::ForIn {
                variables: variables.clone(),
                iterator: desugar_expr_effects(iterator, effect_vars, needs_closure, nt_counter),
                body: desugar_stmts_effects(
                    body,
                    effect_vars,
                    needs_closure,
                    in_scoped_block,
                    nt_counter,
                ),
            },
            crate::ast::StmtKind::Return(es) => crate::ast::StmtKind::Return(
                es.iter()
                    .map(|e| desugar_expr_effects(e, effect_vars, needs_closure, nt_counter))
                    .collect(),
            ),
            crate::ast::StmtKind::Block(b) => crate::ast::StmtKind::Block(desugar_stmts_effects(
                b,
                effect_vars,
                needs_closure,
                in_scoped_block,
                nt_counter,
            )),
            crate::ast::StmtKind::Scoped(b) => {
                // Lower `StmtKind::Scoped` into 3-phase LIFO `AotScope` (`__lust_scoped`)
                // whenever it does not contain outer-targeting `return`/`break`/`continue` (`NG-W13-1B`).
                if !stmts_contain_enclosing_control_flow(b) {
                    let inner_stmts =
                        desugar_stmts_effects(b, effect_vars, needs_closure, true, nt_counter);
                    let thunk = crate::ast::Expr::new(
                        crate::ast::ExprKind::Lambda {
                            params: Vec::new(),
                            return_type: Some(Type {
                                kind: TypeKind::Unit,
                                span: stmt.span,
                            }),
                            body: Box::new(crate::ast::Expr::new(
                                crate::ast::ExprKind::Block(inner_stmts),
                                stmt.span,
                            )),
                        },
                        stmt.span,
                    );
                    crate::ast::StmtKind::Expr(crate::ast::Expr::new(
                        crate::ast::ExprKind::Call {
                            callee: Box::new(crate::ast::Expr::new(
                                crate::ast::ExprKind::Identifier("__lust_scoped".to_string()),
                                stmt.span,
                            )),
                            type_args: None,
                            args: vec![thunk],
                        },
                        stmt.span,
                    ))
                } else {
                    crate::ast::StmtKind::Scoped(desugar_stmts_effects(
                        b,
                        effect_vars,
                        needs_closure,
                        false,
                        nt_counter,
                    ))
                }
            }
            crate::ast::StmtKind::Defer(inner) => {
                let mut v = desugar_stmts_effects(
                    std::slice::from_ref(inner.as_ref()),
                    effect_vars,
                    needs_closure,
                    false,
                    nt_counter,
                );
                if in_scoped_block {
                    // Register `defer` inside `scoped do ... end` directly onto the active `AotScope`
                    // so Phase 3 (`Finalizing`) runs it in strict LIFO order AFTER `Cancelling -> Joining`!
                    let fin_thunk = crate::ast::Expr::new(
                        crate::ast::ExprKind::Lambda {
                            params: Vec::new(),
                            return_type: Some(Type {
                                kind: TypeKind::Unit,
                                span: stmt.span,
                            }),
                            body: Box::new(crate::ast::Expr::new(
                                crate::ast::ExprKind::Block(v),
                                stmt.span,
                            )),
                        },
                        stmt.span,
                    );
                    crate::ast::StmtKind::Expr(crate::ast::Expr::new(
                        crate::ast::ExprKind::Call {
                            callee: Box::new(crate::ast::Expr::new(
                                crate::ast::ExprKind::Identifier(
                                    "__lust_scope_finalizer".to_string(),
                                ),
                                stmt.span,
                            )),
                            type_args: None,
                            args: vec![fin_thunk],
                        },
                        stmt.span,
                    ))
                } else {
                    crate::ast::StmtKind::Defer(Box::new(v.remove(0)))
                }
            }
            crate::ast::StmtKind::ErrDefer(inner) => {
                let mut v = desugar_stmts_effects(
                    std::slice::from_ref(inner.as_ref()),
                    effect_vars,
                    needs_closure,
                    false,
                    nt_counter,
                );
                crate::ast::StmtKind::ErrDefer(Box::new(v.remove(0)))
            }
            other => other.clone(),
        };
        out.push(crate::ast::Stmt::new(new_kind, stmt.span));
    }
    out
}

fn stmts_contain_enclosing_control_flow(stmts: &[crate::ast::Stmt]) -> bool {
    for stmt in stmts {
        match &stmt.kind {
            crate::ast::StmtKind::Return(_)
            | crate::ast::StmtKind::Break
            | crate::ast::StmtKind::Continue => return true,
            crate::ast::StmtKind::If {
                then_block,
                elseif_branches,
                else_block,
                ..
            } => {
                if stmts_contain_enclosing_control_flow(then_block)
                    || elseif_branches
                        .iter()
                        .any(|(_, b)| stmts_contain_enclosing_control_flow(b))
                    || else_block
                        .as_ref()
                        .is_some_and(|b| stmts_contain_enclosing_control_flow(b))
                {
                    return true;
                }
            }
            crate::ast::StmtKind::While { body, .. }
            | crate::ast::StmtKind::ForNumeric { body, .. }
            | crate::ast::StmtKind::ForIn { body, .. } => {
                // `break`/`continue` inside an inner loop target that inner loop, only `return` escapes.
                if stmts_contain_return(body) {
                    return true;
                }
            }
            crate::ast::StmtKind::Block(b) | crate::ast::StmtKind::Scoped(b) => {
                if stmts_contain_enclosing_control_flow(b) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn stmts_contain_return(stmts: &[crate::ast::Stmt]) -> bool {
    for stmt in stmts {
        match &stmt.kind {
            crate::ast::StmtKind::Return(_) => return true,
            crate::ast::StmtKind::If {
                then_block,
                elseif_branches,
                else_block,
                ..
            } => {
                if stmts_contain_return(then_block)
                    || elseif_branches.iter().any(|(_, b)| stmts_contain_return(b))
                    || else_block.as_ref().is_some_and(|b| stmts_contain_return(b))
                {
                    return true;
                }
            }
            crate::ast::StmtKind::While { body, .. }
            | crate::ast::StmtKind::ForNumeric { body, .. }
            | crate::ast::StmtKind::ForIn { body, .. }
            | crate::ast::StmtKind::Block(body)
            | crate::ast::StmtKind::Scoped(body) => {
                if stmts_contain_return(body) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn expr_is_effect_value(
    expr: &crate::ast::Expr,
    effect_vars: &std::collections::HashSet<String>,
) -> bool {
    match &expr.kind {
        crate::ast::ExprKind::Paren(inner) => expr_is_effect_value(inner, effect_vars),
        crate::ast::ExprKind::EffectBlock(_) => true,
        crate::ast::ExprKind::Identifier(name) => effect_vars.contains(name),
        crate::ast::ExprKind::MethodCall { method, .. } if method == "provide" => true,
        crate::ast::ExprKind::Call { callee, .. } => match &callee.kind {
            crate::ast::ExprKind::Identifier(n) if n == "__lust_provide" => true,
            crate::ast::ExprKind::FieldAccess { field, .. } if field == "provide" => true,
            _ => false,
        },
        _ => false,
    }
}

fn desugar_expr_effects(
    expr: &crate::ast::Expr,
    effect_vars: &mut std::collections::HashSet<String>,
    needs_closure: &std::collections::HashSet<String>,
    nt_counter: &mut usize,
) -> crate::ast::Expr {
    let span = expr.span;
    match &expr.kind {
        crate::ast::ExprKind::EffectBlock(stmts) => {
            let new_stmts =
                desugar_stmts_effects(stmts, effect_vars, needs_closure, false, nt_counter);
            crate::ast::Expr::new(
                crate::ast::ExprKind::Lambda {
                    params: Vec::new(),
                    return_type: None,
                    body: Box::new(crate::ast::Expr::new(
                        crate::ast::ExprKind::Block(new_stmts),
                        span,
                    )),
                },
                span,
            )
        }
        crate::ast::ExprKind::MethodCall {
            receiver,
            method,
            type_args,
            args,
        } => {
            let recv_is_eff = expr_is_effect_value(receiver, effect_vars);
            let new_recv = desugar_expr_effects(receiver, effect_vars, needs_closure, nt_counter);
            let new_args: Vec<_> = args
                .iter()
                .map(|a| desugar_expr_effects(a, effect_vars, needs_closure, nt_counter))
                .collect();
            if method == "provide" && new_args.len() == 1 && recv_is_eff {
                return crate::ast::Expr::new(
                    crate::ast::ExprKind::Call {
                        callee: Box::new(crate::ast::Expr::new(
                            crate::ast::ExprKind::Identifier("__lust_provide".to_string()),
                            span,
                        )),
                        type_args: None,
                        args: vec![new_recv, new_args.into_iter().next().unwrap()],
                    },
                    span,
                );
            }
            if method == "run_sync" && new_args.is_empty() && recv_is_eff {
                return crate::ast::Expr::new(
                    crate::ast::ExprKind::Call {
                        callee: Box::new(crate::ast::Expr::new(
                            crate::ast::ExprKind::Identifier("__lust_run_sync".to_string()),
                            span,
                        )),
                        type_args: None,
                        args: vec![new_recv],
                    },
                    span,
                );
            }
            crate::ast::Expr::new(
                crate::ast::ExprKind::MethodCall {
                    receiver: Box::new(new_recv),
                    method: method.clone(),
                    type_args: type_args.clone(),
                    args: new_args,
                },
                span,
            )
        }
        crate::ast::ExprKind::Call {
            callee,
            type_args,
            args,
        } => {
            // Non-tail one-shot `handle..with` -> state-machine split (`NG-W13-1B`, `F-12`, `F-19`)
            if matches!(&callee.kind, crate::ast::ExprKind::Identifier(n) if n == "__lust_handle")
                && args.len() == 2
                && handle_arms_have_non_tail(&args[1])
                && let Some(transformed) = desugar_nontail_handle(
                    &args[0],
                    &args[1],
                    span,
                    effect_vars,
                    needs_closure,
                    nt_counter,
                )
            {
                return transformed;
            }
            if let crate::ast::ExprKind::FieldAccess { object, field } = &callee.kind
                && expr_is_effect_value(object, effect_vars)
            {
                let new_obj = desugar_expr_effects(object, effect_vars, needs_closure, nt_counter);
                let new_args: Vec<_> = args
                    .iter()
                    .map(|a| desugar_expr_effects(a, effect_vars, needs_closure, nt_counter))
                    .collect();
                if field == "provide" && new_args.len() == 1 {
                    return crate::ast::Expr::new(
                        crate::ast::ExprKind::Call {
                            callee: Box::new(crate::ast::Expr::new(
                                crate::ast::ExprKind::Identifier("__lust_provide".to_string()),
                                span,
                            )),
                            type_args: None,
                            args: vec![new_obj, new_args.into_iter().next().unwrap()],
                        },
                        span,
                    );
                }
                if field == "run_sync" && new_args.is_empty() {
                    return crate::ast::Expr::new(
                        crate::ast::ExprKind::Call {
                            callee: Box::new(crate::ast::Expr::new(
                                crate::ast::ExprKind::Identifier("__lust_run_sync".to_string()),
                                span,
                            )),
                            type_args: None,
                            args: vec![new_obj],
                        },
                        span,
                    );
                }
            }
            crate::ast::Expr::new(
                crate::ast::ExprKind::Call {
                    callee: Box::new(desugar_expr_effects(
                        callee,
                        effect_vars,
                        needs_closure,
                        nt_counter,
                    )),
                    type_args: type_args.clone(),
                    args: args
                        .iter()
                        .map(|a| desugar_expr_effects(a, effect_vars, needs_closure, nt_counter))
                        .collect(),
                },
                span,
            )
        }
        crate::ast::ExprKind::Binary { left, op, right } => crate::ast::Expr::new(
            crate::ast::ExprKind::Binary {
                left: Box::new(desugar_expr_effects(
                    left,
                    effect_vars,
                    needs_closure,
                    nt_counter,
                )),
                op: *op,
                right: Box::new(desugar_expr_effects(
                    right,
                    effect_vars,
                    needs_closure,
                    nt_counter,
                )),
            },
            span,
        ),
        crate::ast::ExprKind::Unary { op, operand } => crate::ast::Expr::new(
            crate::ast::ExprKind::Unary {
                op: *op,
                operand: Box::new(desugar_expr_effects(
                    operand,
                    effect_vars,
                    needs_closure,
                    nt_counter,
                )),
            },
            span,
        ),
        crate::ast::ExprKind::FieldAccess { object, field } => crate::ast::Expr::new(
            crate::ast::ExprKind::FieldAccess {
                object: Box::new(desugar_expr_effects(
                    object,
                    effect_vars,
                    needs_closure,
                    nt_counter,
                )),
                field: field.clone(),
            },
            span,
        ),
        crate::ast::ExprKind::Index { object, index } => crate::ast::Expr::new(
            crate::ast::ExprKind::Index {
                object: Box::new(desugar_expr_effects(
                    object,
                    effect_vars,
                    needs_closure,
                    nt_counter,
                )),
                index: Box::new(desugar_expr_effects(
                    index,
                    effect_vars,
                    needs_closure,
                    nt_counter,
                )),
            },
            span,
        ),
        crate::ast::ExprKind::Array(es) => crate::ast::Expr::new(
            crate::ast::ExprKind::Array(
                es.iter()
                    .map(|e| desugar_expr_effects(e, effect_vars, needs_closure, nt_counter))
                    .collect(),
            ),
            span,
        ),
        crate::ast::ExprKind::Map(pairs) => crate::ast::Expr::new(
            crate::ast::ExprKind::Map(
                pairs
                    .iter()
                    .map(|(k, v)| {
                        (
                            desugar_expr_effects(k, effect_vars, needs_closure, nt_counter),
                            desugar_expr_effects(v, effect_vars, needs_closure, nt_counter),
                        )
                    })
                    .collect(),
            ),
            span,
        ),
        crate::ast::ExprKind::Tuple(es) => crate::ast::Expr::new(
            crate::ast::ExprKind::Tuple(
                es.iter()
                    .map(|e| desugar_expr_effects(e, effect_vars, needs_closure, nt_counter))
                    .collect(),
            ),
            span,
        ),
        crate::ast::ExprKind::StructLiteral { name, fields } => crate::ast::Expr::new(
            crate::ast::ExprKind::StructLiteral {
                name: name.clone(),
                fields: fields
                    .iter()
                    .map(|f| crate::ast::StructLiteralField {
                        name: f.name.clone(),
                        value: desugar_expr_effects(
                            &f.value,
                            effect_vars,
                            needs_closure,
                            nt_counter,
                        ),
                        span: f.span,
                    })
                    .collect(),
            },
            span,
        ),
        crate::ast::ExprKind::EnumConstructor {
            enum_name,
            variant,
            args,
        } => crate::ast::Expr::new(
            crate::ast::ExprKind::EnumConstructor {
                enum_name: enum_name.clone(),
                variant: variant.clone(),
                args: args
                    .iter()
                    .map(|a| desugar_expr_effects(a, effect_vars, needs_closure, nt_counter))
                    .collect(),
            },
            span,
        ),
        crate::ast::ExprKind::Lambda {
            params,
            return_type,
            body,
        } => crate::ast::Expr::new(
            crate::ast::ExprKind::Lambda {
                params: params.clone(),
                return_type: return_type.clone(),
                body: Box::new(desugar_expr_effects(
                    body,
                    effect_vars,
                    needs_closure,
                    nt_counter,
                )),
            },
            span,
        ),
        crate::ast::ExprKind::Paren(inner) => crate::ast::Expr::new(
            crate::ast::ExprKind::Paren(Box::new(desugar_expr_effects(
                inner,
                effect_vars,
                needs_closure,
                nt_counter,
            ))),
            span,
        ),
        crate::ast::ExprKind::Cast {
            expr: inner,
            target_type,
        } => crate::ast::Expr::new(
            crate::ast::ExprKind::Cast {
                expr: Box::new(desugar_expr_effects(
                    inner,
                    effect_vars,
                    needs_closure,
                    nt_counter,
                )),
                target_type: target_type.clone(),
            },
            span,
        ),
        crate::ast::ExprKind::TypeCheck {
            expr: inner,
            check_type,
        } => crate::ast::Expr::new(
            crate::ast::ExprKind::TypeCheck {
                expr: Box::new(desugar_expr_effects(
                    inner,
                    effect_vars,
                    needs_closure,
                    nt_counter,
                )),
                check_type: check_type.clone(),
            },
            span,
        ),
        crate::ast::ExprKind::IsPattern {
            expr: inner,
            pattern,
        } => crate::ast::Expr::new(
            crate::ast::ExprKind::IsPattern {
                expr: Box::new(desugar_expr_effects(
                    inner,
                    effect_vars,
                    needs_closure,
                    nt_counter,
                )),
                pattern: pattern.clone(),
            },
            span,
        ),
        crate::ast::ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => crate::ast::Expr::new(
            crate::ast::ExprKind::If {
                condition: Box::new(desugar_expr_effects(
                    condition,
                    effect_vars,
                    needs_closure,
                    nt_counter,
                )),
                then_branch: Box::new(desugar_expr_effects(
                    then_branch,
                    effect_vars,
                    needs_closure,
                    nt_counter,
                )),
                else_branch: else_branch.as_ref().map(|eb| {
                    Box::new(desugar_expr_effects(
                        eb,
                        effect_vars,
                        needs_closure,
                        nt_counter,
                    ))
                }),
            },
            span,
        ),
        crate::ast::ExprKind::Block(b) => crate::ast::Expr::new(
            crate::ast::ExprKind::Block(desugar_stmts_effects(
                b,
                effect_vars,
                needs_closure,
                false,
                nt_counter,
            )),
            span,
        ),
        crate::ast::ExprKind::Return(es) => crate::ast::Expr::new(
            crate::ast::ExprKind::Return(
                es.iter()
                    .map(|e| desugar_expr_effects(e, effect_vars, needs_closure, nt_counter))
                    .collect(),
            ),
            span,
        ),
        crate::ast::ExprKind::Range {
            start,
            end,
            inclusive,
        } => crate::ast::Expr::new(
            crate::ast::ExprKind::Range {
                start: Box::new(desugar_expr_effects(
                    start,
                    effect_vars,
                    needs_closure,
                    nt_counter,
                )),
                end: Box::new(desugar_expr_effects(
                    end,
                    effect_vars,
                    needs_closure,
                    nt_counter,
                )),
                inclusive: *inclusive,
            },
            span,
        ),
        _ => expr.clone(),
    }
}

// ---------------------------------------------------------------------------
// NG-W13-1B: AST CPS / State-Machine Transformation for Non-Tail `handle..with`
// ---------------------------------------------------------------------------

fn desugar_nontail_handle(
    body_thunk: &crate::ast::Expr,
    arms_array: &crate::ast::Expr,
    span: Span,
    effect_vars: &mut std::collections::HashSet<String>,
    needs_closure: &std::collections::HashSet<String>,
    nt_counter: &mut usize,
) -> Option<crate::ast::Expr> {
    let crate::ast::ExprKind::Lambda {
        body: raw_body_expr,
        ..
    } = &body_thunk.kind
    else {
        return None;
    };
    let crate::ast::ExprKind::Array(arm_tuples) = &arms_array.kind else {
        return None;
    };

    // Collect handled operation names
    let mut handled_ops: std::collections::HashSet<String> = std::collections::HashSet::new();
    for tup in arm_tuples {
        if let crate::ast::ExprKind::Tuple(elems) = &tup.kind
            && elems.len() == 2
            && let crate::ast::ExprKind::Literal(crate::ast::Literal::String(op_name)) =
                &elems[0].kind
        {
            handled_ops.insert(op_name.clone());
        }
    }
    if handled_ops.is_empty() {
        return None;
    }

    // Collect simple AST type hints for `perform(op, [arg0, ...])` arguments so untyped handler
    // arm parameters (`Ask(v) => ...`) get accurate MIR types (`f64`, `bool`, `String`, `()`, `i64`).
    let mut op_arg_hints: std::collections::HashMap<String, Vec<Option<Type>>> =
        std::collections::HashMap::new();
    let mut local_ty_env: std::collections::HashMap<String, Type> =
        std::collections::HashMap::new();
    collect_perform_arg_hints_in_expr(raw_body_expr, &handled_ops, &mut local_ty_env, &mut op_arg_hints);

    // Infer `ResumeVal` type hint from `__lust_resume(val)` calls inside the arms
    let mut resume_hint: Option<Type> = None;
    let mut resume_hint_conflict = false;
    let mut rewritten_tuples = Vec::with_capacity(arm_tuples.len());
    for tup in arm_tuples {
        let crate::ast::ExprKind::Tuple(elems) = &tup.kind else {
            return None;
        };
        if elems.len() != 2 {
            return None;
        }
        let op_name = match &elems[0].kind {
            crate::ast::ExprKind::Literal(crate::ast::Literal::String(s)) => s.clone(),
            _ => return None,
        };
        let crate::ast::ExprKind::Lambda {
            params,
            return_type,
            body,
        } = &elems[1].kind
        else {
            return None;
        };

        let mut typed_params = Vec::with_capacity(params.len());
        let mut arm_local_env = std::collections::HashMap::new();
        let hints = op_arg_hints.get(&op_name);
        for (pi, (pname, pty)) in params.iter().enumerate() {
            let final_ty = pty
                .clone()
                .or_else(|| hints.and_then(|v| v.get(pi).cloned().flatten()));
            if let Some(t) = &final_ty {
                arm_local_env.insert(pname.clone(), t.clone());
            }
            typed_params.push((pname.clone(), final_ty));
        }

        collect_resume_arg_hint_in_expr(
            body,
            &mut arm_local_env,
            &mut resume_hint,
            &mut resume_hint_conflict,
        );

        let desugared_arm_body =
            desugar_expr_effects(body, effect_vars, needs_closure, nt_counter);
        let rewritten_arm_body = rewrite_resume_to_nontail_expr(&desugared_arm_body);
        let new_arm_lambda = crate::ast::Expr::new(
            crate::ast::ExprKind::Lambda {
                params: typed_params,
                return_type: return_type.clone(),
                body: Box::new(rewritten_arm_body),
            },
            elems[1].span,
        );
        rewritten_tuples.push(crate::ast::Expr::new(
            crate::ast::ExprKind::Tuple(vec![elems[0].clone(), new_arm_lambda]),
            tup.span,
        ));
    }
    if resume_hint_conflict {
        resume_hint = Some(Type {
            kind: TypeKind::Unknown,
            span,
        });
    }

    let body_stmts = match &raw_body_expr.kind {
        crate::ast::ExprKind::Block(stmts) => stmts.clone(),
        _ => vec![crate::ast::Stmt::new(
            crate::ast::StmtKind::Expr((**raw_body_expr).clone()),
            raw_body_expr.span,
        )],
    };

    // NG-W16-2 (`AOT-CPS-LOOPS`):
    // 1. Inline module-local helper calls that transitively perform a handled operation.
    let body_stmts = inline_handled_helper_calls_in_stmts(&body_stmts, &handled_ops, nt_counter);
    // 2. Lower `Defer` / `Scoped` blocks into explicit LIFO cleanup at block exit.
    let body_stmts = lower_defers_and_scoped_in_stmts(&body_stmts, nt_counter);
    // 3. Desugar `ForNumeric` / `ForIn` loops containing a handled perform into canonical `While`.
    let body_stmts = desugar_for_loops_with_perform(&body_stmts, &handled_ops, nt_counter);
    // 4. Lower `While` loops containing a handled perform into recursive `__cps_loop_step_*` calls.
    let body_stmts = lower_while_loops_with_perform(
        &body_stmts,
        &handled_ops,
        arms_array,
        &local_ty_env,
        nt_counter,
        span,
    );

    let split_stmts = split_body_stmts_for_effect_yield(
        &body_stmts,
        &handled_ops,
        arms_array,
        resume_hint.as_ref(),
        effect_vars,
        needs_closure,
        nt_counter,
        span,
    );

    let split_body_thunk = crate::ast::Expr::new(
        crate::ast::ExprKind::Lambda {
            params: Vec::new(),
            return_type: None,
            body: Box::new(crate::ast::Expr::new(
                crate::ast::ExprKind::Block(split_stmts),
                raw_body_expr.span,
            )),
        },
        body_thunk.span,
    );
    let rewritten_arms = crate::ast::Expr::new(
        crate::ast::ExprKind::Array(rewritten_tuples),
        arms_array.span,
    );

    Some(crate::ast::Expr::new(
        crate::ast::ExprKind::Call {
            callee: Box::new(crate::ast::Expr::new(
                crate::ast::ExprKind::Identifier("__lust_nontail_handle".to_string()),
                span,
            )),
            type_args: None,
            args: vec![split_body_thunk, rewritten_arms],
        },
        span,
    ))
}

fn inline_handled_helper_calls_in_stmts(
    stmts: &[crate::ast::Stmt],
    handled_ops: &std::collections::HashSet<String>,
    nt_counter: &mut usize,
) -> Vec<crate::ast::Stmt> {
    let mut out = Vec::with_capacity(stmts.len());
    for stmt in stmts {
        // Case A: `local x [: T] = helper(args...)`
        if let crate::ast::StmtKind::Local {
            bindings,
            initializer: Some(inits),
        } = &stmt.kind
            && bindings.len() == 1
            && inits.len() == 1
            && let crate::ast::ExprKind::Call { callee, args, .. } = &inits[0].kind
            && let crate::ast::ExprKind::Identifier(fn_name) = &callee.kind
            && let Some(helper) = lookup_handled_helper(fn_name, handled_ops)
        {
            let inlined = build_inlined_helper_stmts(
                &helper,
                args,
                Some(bindings.clone()),
                stmt.span,
                nt_counter,
            );
            let recursed = inline_handled_helper_calls_in_stmts(&inlined, handled_ops, nt_counter);
            out.extend(recursed);
            continue;
        }
        // Case B: bare `helper(args...)`
        if let crate::ast::StmtKind::Expr(e) = &stmt.kind
            && let crate::ast::ExprKind::Call { callee, args, .. } = &e.kind
            && let crate::ast::ExprKind::Identifier(fn_name) = &callee.kind
            && let Some(helper) = lookup_handled_helper(fn_name, handled_ops)
        {
            let inlined = build_inlined_helper_stmts(&helper, args, None, stmt.span, nt_counter);
            let recursed = inline_handled_helper_calls_in_stmts(&inlined, handled_ops, nt_counter);
            out.extend(recursed);
            continue;
        }
        // Recurse into compound statements
        let new_stmt = match &stmt.kind {
            crate::ast::StmtKind::While { condition, body } => crate::ast::Stmt::new(
                crate::ast::StmtKind::While {
                    condition: condition.clone(),
                    body: inline_handled_helper_calls_in_stmts(body, handled_ops, nt_counter),
                },
                stmt.span,
            ),
            crate::ast::StmtKind::ForNumeric {
                variable,
                start,
                end,
                step,
                body,
            } => crate::ast::Stmt::new(
                crate::ast::StmtKind::ForNumeric {
                    variable: variable.clone(),
                    start: start.clone(),
                    end: end.clone(),
                    step: step.clone(),
                    body: inline_handled_helper_calls_in_stmts(body, handled_ops, nt_counter),
                },
                stmt.span,
            ),
            crate::ast::StmtKind::ForIn {
                variables,
                iterator,
                body,
            } => crate::ast::Stmt::new(
                crate::ast::StmtKind::ForIn {
                    variables: variables.clone(),
                    iterator: iterator.clone(),
                    body: inline_handled_helper_calls_in_stmts(body, handled_ops, nt_counter),
                },
                stmt.span,
            ),
            crate::ast::StmtKind::Scoped(b) => crate::ast::Stmt::new(
                crate::ast::StmtKind::Scoped(inline_handled_helper_calls_in_stmts(
                    b,
                    handled_ops,
                    nt_counter,
                )),
                stmt.span,
            ),
            crate::ast::StmtKind::Block(b) => crate::ast::Stmt::new(
                crate::ast::StmtKind::Block(inline_handled_helper_calls_in_stmts(
                    b,
                    handled_ops,
                    nt_counter,
                )),
                stmt.span,
            ),
            crate::ast::StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => crate::ast::Stmt::new(
                crate::ast::StmtKind::If {
                    condition: condition.clone(),
                    then_block: inline_handled_helper_calls_in_stmts(
                        then_block,
                        handled_ops,
                        nt_counter,
                    ),
                    elseif_branches: elseif_branches
                        .iter()
                        .map(|(c, b)| {
                            (
                                c.clone(),
                                inline_handled_helper_calls_in_stmts(b, handled_ops, nt_counter),
                            )
                        })
                        .collect(),
                    else_block: else_block
                        .as_ref()
                        .map(|b| inline_handled_helper_calls_in_stmts(b, handled_ops, nt_counter)),
                },
                stmt.span,
            ),
            _ => stmt.clone(),
        };
        out.push(new_stmt);
    }
    out
}

fn lookup_handled_helper(
    fn_name: &str,
    handled_ops: &std::collections::HashSet<String>,
) -> Option<FunctionDef> {
    CPS_MODULE_CTX.with(|ctx| {
        let c = ctx.borrow();
        let helper = c.helpers.get(fn_name)?;
        if helper
            .body
            .iter()
            .any(|s| stmt_contains_handled_perform(s, handled_ops))
        {
            Some(helper.clone())
        } else {
            None
        }
    })
}

fn build_inlined_helper_stmts(
    helper: &FunctionDef,
    args: &[crate::ast::Expr],
    target_bindings: Option<Vec<crate::ast::LocalBinding>>,
    span: Span,
    nt_counter: &mut usize,
) -> Vec<crate::ast::Stmt> {
    let id = *nt_counter;
    *nt_counter += 1;
    let mut rename_map: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    let mut out = Vec::new();

    for (idx, param) in helper.params.iter().enumerate() {
        let fresh = format!("__inl_{id}_{}", param.name);
        rename_map.insert(param.name.clone(), fresh.clone());
        let arg_expr = args
            .get(idx)
            .cloned()
            .unwrap_or_else(|| make_unit_expr(span));
        out.push(crate::ast::Stmt::new(
            crate::ast::StmtKind::Local {
                bindings: vec![crate::ast::LocalBinding {
                    name: fresh,
                    type_annotation: Some(param.ty.clone()),
                    span,
                }],
                initializer: Some(vec![arg_expr]),
            },
            span,
        ));
    }

    // Alpha-rename top-level locals inside helper body to avoid shadowing collisions
    for s in &helper.body {
        if let crate::ast::StmtKind::Local { bindings, .. } = &s.kind {
            for b in bindings {
                let fresh = format!("__inl_{id}_{}", b.name);
                rename_map.insert(b.name.clone(), fresh);
            }
        }
    }

    let body_len = helper.body.len();
    for (idx, s) in helper.body.iter().enumerate() {
        let renamed = rename_idents_in_stmt(s, &rename_map);
        let is_last = idx + 1 == body_len;
        if is_last && let crate::ast::StmtKind::Return(vals) = &renamed.kind {
            if let Some(bindings) = &target_bindings {
                out.push(crate::ast::Stmt::new(
                    crate::ast::StmtKind::Local {
                        bindings: bindings.clone(),
                        initializer: Some(vals.clone()),
                    },
                    span,
                ));
            } else if let Some(first) = vals.first() {
                out.push(crate::ast::Stmt::new(
                    crate::ast::StmtKind::Expr(first.clone()),
                    span,
                ));
            }
            continue;
        }
        out.push(renamed);
    }
    out
}

fn rename_idents_in_stmt(
    stmt: &crate::ast::Stmt,
    map: &std::collections::HashMap<String, String>,
) -> crate::ast::Stmt {
    let kind = match &stmt.kind {
        crate::ast::StmtKind::Local {
            bindings,
            initializer,
        } => crate::ast::StmtKind::Local {
            bindings: bindings
                .iter()
                .map(|b| crate::ast::LocalBinding {
                    name: map.get(&b.name).cloned().unwrap_or_else(|| b.name.clone()),
                    type_annotation: b.type_annotation.clone(),
                    span: b.span,
                })
                .collect(),
            initializer: initializer
                .as_ref()
                .map(|es| es.iter().map(|e| rename_idents_in_expr(e, map)).collect()),
        },
        crate::ast::StmtKind::Assign { targets, values } => crate::ast::StmtKind::Assign {
            targets: targets.iter().map(|e| rename_idents_in_expr(e, map)).collect(),
            values: values.iter().map(|e| rename_idents_in_expr(e, map)).collect(),
        },
        crate::ast::StmtKind::CompoundAssign { target, op, value } => {
            crate::ast::StmtKind::CompoundAssign {
                target: rename_idents_in_expr(target, map),
                op: *op,
                value: rename_idents_in_expr(value, map),
            }
        }
        crate::ast::StmtKind::Expr(e) => {
            crate::ast::StmtKind::Expr(rename_idents_in_expr(e, map))
        }
        crate::ast::StmtKind::Return(es) => {
            crate::ast::StmtKind::Return(es.iter().map(|e| rename_idents_in_expr(e, map)).collect())
        }
        crate::ast::StmtKind::If {
            condition,
            then_block,
            elseif_branches,
            else_block,
        } => crate::ast::StmtKind::If {
            condition: rename_idents_in_expr(condition, map),
            then_block: then_block
                .iter()
                .map(|s| rename_idents_in_stmt(s, map))
                .collect(),
            elseif_branches: elseif_branches
                .iter()
                .map(|(c, b)| {
                    (
                        rename_idents_in_expr(c, map),
                        b.iter().map(|s| rename_idents_in_stmt(s, map)).collect(),
                    )
                })
                .collect(),
            else_block: else_block
                .as_ref()
                .map(|b| b.iter().map(|s| rename_idents_in_stmt(s, map)).collect()),
        },
        crate::ast::StmtKind::While { condition, body } => crate::ast::StmtKind::While {
            condition: rename_idents_in_expr(condition, map),
            body: body.iter().map(|s| rename_idents_in_stmt(s, map)).collect(),
        },
        crate::ast::StmtKind::Block(b) => {
            crate::ast::StmtKind::Block(b.iter().map(|s| rename_idents_in_stmt(s, map)).collect())
        }
        crate::ast::StmtKind::Scoped(b) => {
            crate::ast::StmtKind::Scoped(b.iter().map(|s| rename_idents_in_stmt(s, map)).collect())
        }
        crate::ast::StmtKind::Defer(s) => {
            crate::ast::StmtKind::Defer(Box::new(rename_idents_in_stmt(s, map)))
        }
        crate::ast::StmtKind::ErrDefer(s) => {
            crate::ast::StmtKind::ErrDefer(Box::new(rename_idents_in_stmt(s, map)))
        }
        other => other.clone(),
    };
    crate::ast::Stmt::new(kind, stmt.span)
}

fn rename_idents_in_expr(
    expr: &crate::ast::Expr,
    map: &std::collections::HashMap<String, String>,
) -> crate::ast::Expr {
    let kind = match &expr.kind {
        crate::ast::ExprKind::Identifier(name) => crate::ast::ExprKind::Identifier(
            map.get(name).cloned().unwrap_or_else(|| name.clone()),
        ),
        crate::ast::ExprKind::Call {
            callee,
            type_args,
            args,
        } => crate::ast::ExprKind::Call {
            callee: Box::new(rename_idents_in_expr(callee, map)),
            type_args: type_args.clone(),
            args: args.iter().map(|a| rename_idents_in_expr(a, map)).collect(),
        },
        crate::ast::ExprKind::MethodCall {
            receiver,
            method,
            type_args,
            args,
        } => crate::ast::ExprKind::MethodCall {
            receiver: Box::new(rename_idents_in_expr(receiver, map)),
            method: method.clone(),
            type_args: type_args.clone(),
            args: args.iter().map(|a| rename_idents_in_expr(a, map)).collect(),
        },
        crate::ast::ExprKind::Binary { left, op, right } => crate::ast::ExprKind::Binary {
            left: Box::new(rename_idents_in_expr(left, map)),
            op: *op,
            right: Box::new(rename_idents_in_expr(right, map)),
        },
        crate::ast::ExprKind::Unary { op, operand } => crate::ast::ExprKind::Unary {
            op: *op,
            operand: Box::new(rename_idents_in_expr(operand, map)),
        },
        crate::ast::ExprKind::Paren(inner) => {
            crate::ast::ExprKind::Paren(Box::new(rename_idents_in_expr(inner, map)))
        }
        crate::ast::ExprKind::Array(es) => {
            crate::ast::ExprKind::Array(es.iter().map(|e| rename_idents_in_expr(e, map)).collect())
        }
        crate::ast::ExprKind::Tuple(es) => {
            crate::ast::ExprKind::Tuple(es.iter().map(|e| rename_idents_in_expr(e, map)).collect())
        }
        crate::ast::ExprKind::Index { object, index } => crate::ast::ExprKind::Index {
            object: Box::new(rename_idents_in_expr(object, map)),
            index: Box::new(rename_idents_in_expr(index, map)),
        },
        crate::ast::ExprKind::FieldAccess { object, field } => crate::ast::ExprKind::FieldAccess {
            object: Box::new(rename_idents_in_expr(object, map)),
            field: field.clone(),
        },
        crate::ast::ExprKind::Cast {
            expr: inner,
            target_type,
        } => crate::ast::ExprKind::Cast {
            expr: Box::new(rename_idents_in_expr(inner, map)),
            target_type: target_type.clone(),
        },
        other => other.clone(),
    };
    crate::ast::Expr::new(kind, expr.span)
}

fn lower_defers_and_scoped_in_stmts(
    stmts: &[crate::ast::Stmt],
    nt_counter: &mut usize,
) -> Vec<crate::ast::Stmt> {
    let mut main_stmts: Vec<crate::ast::Stmt> = Vec::with_capacity(stmts.len());
    let mut deferred: Vec<crate::ast::Stmt> = Vec::new();

    for stmt in stmts {
        match &stmt.kind {
            crate::ast::StmtKind::Defer(inner) => {
                deferred.push((**inner).clone());
            }
            crate::ast::StmtKind::Scoped(inner) => {
                let lowered = lower_defers_and_scoped_in_stmts(inner, nt_counter);
                main_stmts.extend(lowered);
            }
            _ => {
                main_stmts.push(stmt.clone());
            }
        }
    }

    if deferred.is_empty() {
        return main_stmts;
    }

    // If the block ends with an expression or return value, evaluate and bind it before
    // running deferred cleanup statements in LIFO order.
    let tail_is_return = matches!(
        main_stmts.last().map(|s| &s.kind),
        Some(crate::ast::StmtKind::Return(_))
    );
    let tail_val: Option<(crate::ast::Expr, Span)> = match main_stmts.last().map(|s| &s.kind) {
        Some(crate::ast::StmtKind::Expr(e)) => {
            let span = main_stmts.last().map(|s| s.span).unwrap_or_else(Span::dummy);
            let expr = e.clone();
            main_stmts.pop();
            Some((expr, span))
        }
        Some(crate::ast::StmtKind::Return(vals)) if vals.len() == 1 => {
            let span = main_stmts.last().map(|s| s.span).unwrap_or_else(Span::dummy);
            let expr = vals[0].clone();
            main_stmts.pop();
            Some((expr, span))
        }
        _ => None,
    };

    if let Some((expr, span)) = tail_val {
        let id = *nt_counter;
        *nt_counter += 1;
        let ret_var = format!("__nt_defer_ret_{id}");
        main_stmts.push(crate::ast::Stmt::new(
            crate::ast::StmtKind::Local {
                bindings: vec![crate::ast::LocalBinding {
                    name: ret_var.clone(),
                    type_annotation: None,
                    span,
                }],
                initializer: Some(vec![expr]),
            },
            span,
        ));
        for d in deferred.into_iter().rev() {
            main_stmts.push(d);
        }
        let ret_ident = crate::ast::Expr::new(crate::ast::ExprKind::Identifier(ret_var), span);
        if tail_is_return {
            main_stmts.push(crate::ast::Stmt::new(
                crate::ast::StmtKind::Return(vec![ret_ident]),
                span,
            ));
        } else {
            main_stmts.push(crate::ast::Stmt::new(
                crate::ast::StmtKind::Expr(ret_ident),
                span,
            ));
        }
    } else {
        for d in deferred.into_iter().rev() {
            main_stmts.push(d);
        }
    }

    main_stmts
}

fn desugar_for_loops_with_perform(
    stmts: &[crate::ast::Stmt],
    handled_ops: &std::collections::HashSet<String>,
    nt_counter: &mut usize,
) -> Vec<crate::ast::Stmt> {
    let mut out = Vec::with_capacity(stmts.len());
    for stmt in stmts {
        if let crate::ast::StmtKind::ForNumeric {
            variable,
            start,
            end,
            step,
            body,
        } = &stmt.kind
            && stmt_contains_handled_perform(stmt, handled_ops)
        {
            let span = stmt.span;
            let id = *nt_counter;
            *nt_counter += 1;
            let idx_name = format!("__nt_for_i_{id}");
            let end_name = format!("__nt_for_end_{id}");
            let step_name = format!("__nt_for_step_{id}");
            let int_ty = Type::new(TypeKind::Int, span);

            let step_expr = step.clone().unwrap_or_else(|| {
                crate::ast::Expr::new(
                    crate::ast::ExprKind::Literal(crate::ast::Literal::Integer(1)),
                    span,
                )
            });
            let is_negative_step = matches!(
                &step_expr.kind,
                crate::ast::ExprKind::Unary {
                    op: crate::ast::UnaryOp::Neg,
                    ..
                }
            );
            let cmp_op = if is_negative_step {
                crate::ast::BinaryOp::Ge
            } else {
                crate::ast::BinaryOp::Le
            };

            out.push(crate::ast::Stmt::new(
                crate::ast::StmtKind::Local {
                    bindings: vec![crate::ast::LocalBinding {
                        name: idx_name.clone(),
                        type_annotation: Some(int_ty.clone()),
                        span,
                    }],
                    initializer: Some(vec![start.clone()]),
                },
                span,
            ));
            out.push(crate::ast::Stmt::new(
                crate::ast::StmtKind::Local {
                    bindings: vec![crate::ast::LocalBinding {
                        name: end_name.clone(),
                        type_annotation: Some(int_ty.clone()),
                        span,
                    }],
                    initializer: Some(vec![end.clone()]),
                },
                span,
            ));
            out.push(crate::ast::Stmt::new(
                crate::ast::StmtKind::Local {
                    bindings: vec![crate::ast::LocalBinding {
                        name: step_name.clone(),
                        type_annotation: Some(int_ty.clone()),
                        span,
                    }],
                    initializer: Some(vec![step_expr]),
                },
                span,
            ));

            let mut while_body = Vec::with_capacity(body.len() + 2);
            while_body.push(crate::ast::Stmt::new(
                crate::ast::StmtKind::Local {
                    bindings: vec![crate::ast::LocalBinding {
                        name: variable.clone(),
                        type_annotation: Some(int_ty),
                        span,
                    }],
                    initializer: Some(vec![crate::ast::Expr::new(
                        crate::ast::ExprKind::Identifier(idx_name.clone()),
                        span,
                    )]),
                },
                span,
            ));
            while_body.extend(desugar_for_loops_with_perform(body, handled_ops, nt_counter));
            while_body.push(crate::ast::Stmt::new(
                crate::ast::StmtKind::Assign {
                    targets: vec![crate::ast::Expr::new(
                        crate::ast::ExprKind::Identifier(idx_name.clone()),
                        span,
                    )],
                    values: vec![crate::ast::Expr::new(
                        crate::ast::ExprKind::Binary {
                            left: Box::new(crate::ast::Expr::new(
                                crate::ast::ExprKind::Identifier(idx_name.clone()),
                                span,
                            )),
                            op: crate::ast::BinaryOp::Add,
                            right: Box::new(crate::ast::Expr::new(
                                crate::ast::ExprKind::Identifier(step_name),
                                span,
                            )),
                        },
                        span,
                    )],
                },
                span,
            ));

            let cond = crate::ast::Expr::new(
                crate::ast::ExprKind::Binary {
                    left: Box::new(crate::ast::Expr::new(
                        crate::ast::ExprKind::Identifier(idx_name),
                        span,
                    )),
                    op: cmp_op,
                    right: Box::new(crate::ast::Expr::new(
                        crate::ast::ExprKind::Identifier(end_name),
                        span,
                    )),
                },
                span,
            );
            out.push(crate::ast::Stmt::new(
                crate::ast::StmtKind::While {
                    condition: cond,
                    body: while_body,
                },
                span,
            ));
            continue;
        }
        out.push(stmt.clone());
    }
    out
}

fn lower_while_loops_with_perform(
    stmts: &[crate::ast::Stmt],
    handled_ops: &std::collections::HashSet<String>,
    orig_arms_array: &crate::ast::Expr,
    local_ty_env: &std::collections::HashMap<String, Type>,
    _nt_counter: &mut usize,
    fallback_span: Span,
) -> Vec<crate::ast::Stmt> {
    let Some(while_idx) = stmts.iter().position(|s| {
        matches!(&s.kind, crate::ast::StmtKind::While { .. })
            && stmt_contains_handled_perform(s, handled_ops)
    }) else {
        return stmts.to_vec();
    };

    let prefix = &stmts[..while_idx];
    let target = &stmts[while_idx];
    let rest = &stmts[while_idx + 1..];

    let crate::ast::StmtKind::While { condition, body } = &target.kind else {
        return stmts.to_vec();
    };

    // Collect known variable types from enclosing function params, local_ty_env, and prefix locals
    let mut known_types: std::collections::HashMap<String, Type> =
        CPS_MODULE_CTX.with(|ctx| ctx.borrow().current_fn_param_tys.clone());
    for (k, v) in local_ty_env {
        known_types.entry(k.clone()).or_insert_with(|| v.clone());
    }
    for s in prefix {
        if let crate::ast::StmtKind::Local {
            bindings,
            initializer,
        } = &s.kind
        {
            for (idx, b) in bindings.iter().enumerate() {
                if let Some(ty) = &b.type_annotation {
                    known_types.insert(b.name.clone(), ty.clone());
                } else if let Some(inits) = initializer
                    && let Some(init_e) = inits.get(idx)
                {
                    let inferred = infer_simple_ast_expr_type(init_e, &known_types)
                        .unwrap_or_else(|| Type::new(TypeKind::Int, b.span));
                    known_types.insert(b.name.clone(), inferred);
                } else {
                    known_types.insert(b.name.clone(), Type::new(TypeKind::Int, b.span));
                }
            }
        }
    }

    // Collect all identifiers referenced in condition, body, and rest that are in known_types
    let mut referenced = std::collections::BTreeSet::new();
    collect_referenced_idents_in_expr(condition, &mut referenced);
    for s in body {
        collect_referenced_idents_in_stmt(s, &mut referenced);
    }
    for s in rest {
        collect_referenced_idents_in_stmt(s, &mut referenced);
    }
    let live_vars: Vec<(String, Type)> = referenced
        .into_iter()
        .filter_map(|name| known_types.get(&name).cloned().map(|ty| (name, ty)))
        .collect();

    let (step_fn_name, ret_ty) = CPS_MODULE_CTX.with(|ctx| {
        let mut c = ctx.borrow_mut();
        let lid = c.loop_counter;
        c.loop_counter += 1;
        let rty = c
            .current_fn_ret_ty
            .clone()
            .unwrap_or_else(|| Type::new(TypeKind::Int, fallback_span));
        (format!("__cps_loop_step_{lid}"), rty)
    });

    let span = target.span;
    let live_arg_exprs: Vec<crate::ast::Expr> = live_vars
        .iter()
        .map(|(n, _)| crate::ast::Expr::new(crate::ast::ExprKind::Identifier(n.clone()), span))
        .collect();

    let recursive_call = crate::ast::Expr::new(
        crate::ast::ExprKind::Call {
            callee: Box::new(crate::ast::Expr::new(
                crate::ast::ExprKind::Identifier(step_fn_name.clone()),
                span,
            )),
            type_args: None,
            args: live_arg_exprs.clone(),
        },
        span,
    );

    let mut iter_body = body.clone();
    iter_body.push(crate::ast::Stmt::new(
        crate::ast::StmtKind::Return(vec![recursive_call.clone()]),
        span,
    ));

    let iter_thunk = crate::ast::Expr::new(
        crate::ast::ExprKind::Lambda {
            params: Vec::new(),
            return_type: None,
            body: Box::new(crate::ast::Expr::new(
                crate::ast::ExprKind::Block(iter_body),
                span,
            )),
        },
        span,
    );
    let iter_handle = crate::ast::Expr::new(
        crate::ast::ExprKind::Call {
            callee: Box::new(crate::ast::Expr::new(
                crate::ast::ExprKind::Identifier("__lust_handle".to_string()),
                span,
            )),
            type_args: None,
            args: vec![iter_thunk, orig_arms_array.clone()],
        },
        span,
    );

    let else_block = if rest
        .iter()
        .any(|s| stmt_contains_handled_perform(s, handled_ops))
    {
        let rest_thunk = crate::ast::Expr::new(
            crate::ast::ExprKind::Lambda {
                params: Vec::new(),
                return_type: None,
                body: Box::new(crate::ast::Expr::new(
                    crate::ast::ExprKind::Block(rest.to_vec()),
                    span,
                )),
            },
            span,
        );
        let rest_handle = crate::ast::Expr::new(
            crate::ast::ExprKind::Call {
                callee: Box::new(crate::ast::Expr::new(
                    crate::ast::ExprKind::Identifier("__lust_handle".to_string()),
                    span,
                )),
                type_args: None,
                args: vec![rest_thunk, orig_arms_array.clone()],
            },
            span,
        );
        vec![crate::ast::Stmt::new(
            crate::ast::StmtKind::Return(vec![rest_handle]),
            span,
        )]
    } else {
        let mut eb = rest.to_vec();
        if let Some(last) = eb.last_mut()
            && let crate::ast::StmtKind::Expr(e) = &last.kind
        {
            let e_owned = e.clone();
            last.kind = crate::ast::StmtKind::Return(vec![e_owned]);
        }
        eb
    };

    let step_fn_body = vec![crate::ast::Stmt::new(
        crate::ast::StmtKind::If {
            condition: condition.clone(),
            then_block: vec![crate::ast::Stmt::new(
                crate::ast::StmtKind::Return(vec![iter_handle]),
                span,
            )],
            elseif_branches: Vec::new(),
            else_block: Some(else_block),
        },
        span,
    )];

    let step_fn = FunctionDef {
        name: step_fn_name,
        type_params: Vec::new(),
        trait_bounds: Vec::new(),
        params: live_vars
            .iter()
            .map(|(n, ty)| crate::ast::FunctionParam {
                name: n.clone(),
                ty: ty.clone(),
                is_self: false,
            })
            .collect(),
        return_type: Some(ret_ty),
        body: step_fn_body,
        is_method: false,
        visibility: crate::ast::Visibility::Private,
        doc: None,
        effects: None,
    };

    CPS_MODULE_CTX.with(|ctx| {
        ctx.borrow_mut().generated_fns.push(step_fn);
    });

    let mut out = prefix.to_vec();
    out.push(crate::ast::Stmt::new(
        crate::ast::StmtKind::Return(vec![recursive_call]),
        span,
    ));
    out
}

fn collect_referenced_idents_in_stmt(
    stmt: &crate::ast::Stmt,
    out: &mut std::collections::BTreeSet<String>,
) {
    match &stmt.kind {
        crate::ast::StmtKind::Local {
            initializer: Some(es),
            ..
        }
        | crate::ast::StmtKind::Return(es) => {
            for e in es {
                collect_referenced_idents_in_expr(e, out);
            }
        }
        crate::ast::StmtKind::Assign { targets, values } => {
            for e in targets.iter().chain(values.iter()) {
                collect_referenced_idents_in_expr(e, out);
            }
        }
        crate::ast::StmtKind::CompoundAssign { target, value, .. } => {
            collect_referenced_idents_in_expr(target, out);
            collect_referenced_idents_in_expr(value, out);
        }
        crate::ast::StmtKind::Expr(e) => collect_referenced_idents_in_expr(e, out),
        crate::ast::StmtKind::If {
            condition,
            then_block,
            elseif_branches,
            else_block,
        } => {
            collect_referenced_idents_in_expr(condition, out);
            for s in then_block {
                collect_referenced_idents_in_stmt(s, out);
            }
            for (c, b) in elseif_branches {
                collect_referenced_idents_in_expr(c, out);
                for s in b {
                    collect_referenced_idents_in_stmt(s, out);
                }
            }
            if let Some(b) = else_block {
                for s in b {
                    collect_referenced_idents_in_stmt(s, out);
                }
            }
        }
        crate::ast::StmtKind::While { condition, body } => {
            collect_referenced_idents_in_expr(condition, out);
            for s in body {
                collect_referenced_idents_in_stmt(s, out);
            }
        }
        crate::ast::StmtKind::Block(b) | crate::ast::StmtKind::Scoped(b) => {
            for s in b {
                collect_referenced_idents_in_stmt(s, out);
            }
        }
        crate::ast::StmtKind::Defer(s) | crate::ast::StmtKind::ErrDefer(s) => {
            collect_referenced_idents_in_stmt(s, out);
        }
        _ => {}
    }
}

fn collect_referenced_idents_in_expr(
    expr: &crate::ast::Expr,
    out: &mut std::collections::BTreeSet<String>,
) {
    match &expr.kind {
        crate::ast::ExprKind::Identifier(name) => {
            out.insert(name.clone());
        }
        crate::ast::ExprKind::Call { callee, args, .. } => {
            collect_referenced_idents_in_expr(callee, out);
            for a in args {
                collect_referenced_idents_in_expr(a, out);
            }
        }
        crate::ast::ExprKind::MethodCall { receiver, args, .. } => {
            collect_referenced_idents_in_expr(receiver, out);
            for a in args {
                collect_referenced_idents_in_expr(a, out);
            }
        }
        crate::ast::ExprKind::Binary { left, right, .. } => {
            collect_referenced_idents_in_expr(left, out);
            collect_referenced_idents_in_expr(right, out);
        }
        crate::ast::ExprKind::Unary { operand: inner, .. }
        | crate::ast::ExprKind::Paren(inner)
        | crate::ast::ExprKind::FieldAccess { object: inner, .. }
        | crate::ast::ExprKind::Cast { expr: inner, .. } => {
            collect_referenced_idents_in_expr(inner, out);
        }
        crate::ast::ExprKind::Array(es) | crate::ast::ExprKind::Tuple(es) => {
            for e in es {
                collect_referenced_idents_in_expr(e, out);
            }
        }
        crate::ast::ExprKind::Index { object, index } => {
            collect_referenced_idents_in_expr(object, out);
            collect_referenced_idents_in_expr(index, out);
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn split_body_stmts_for_effect_yield(
    stmts: &[crate::ast::Stmt],
    handled_ops: &std::collections::HashSet<String>,
    orig_arms_array: &crate::ast::Expr,
    resume_hint: Option<&Type>,
    effect_vars: &mut std::collections::HashSet<String>,
    needs_closure: &std::collections::HashSet<String>,
    nt_counter: &mut usize,
    fallback_span: Span,
) -> Vec<crate::ast::Stmt> {
    // Locate the first statement containing a handled `perform(op, args)`
    let first_perf_idx = stmts
        .iter()
        .position(|s| stmt_contains_handled_perform(s, handled_ops));

    let Some(split_idx) = first_perf_idx else {
        // No handled perform in this segment -> complete via `__lust_yield_completed(val)`
        let desugared =
            desugar_stmts_effects(stmts, effect_vars, needs_closure, false, nt_counter);
        return wrap_stmts_as_yield_completed(&desugared, fallback_span);
    };

    let prefix = desugar_stmts_effects(
        &stmts[..split_idx],
        effect_vars,
        needs_closure,
        false,
        nt_counter,
    );
    let target = &stmts[split_idx];
    let rest = &stmts[split_idx + 1..];

    // Subcase 1: `local x [: T] = __lust_perform("Op", [args])`
    if let crate::ast::StmtKind::Local {
        bindings,
        initializer: Some(inits),
    } = &target.kind
        && bindings.len() == 1
        && inits.len() == 1
        && let Some((op_lit, args_arr)) = match_direct_handled_perform(&inits[0], handled_ops)
    {
        let id = *nt_counter;
        *nt_counter += 1;
        let resumed_name = format!("__nt_resumed_{id}");
        let resumed_ty = bindings[0]
            .type_annotation
            .clone()
            .or_else(|| resume_hint.cloned());
        let bind_stmt = crate::ast::Stmt::new(
            crate::ast::StmtKind::Local {
                bindings: bindings.clone(),
                initializer: Some(vec![crate::ast::Expr::new(
                    crate::ast::ExprKind::Identifier(resumed_name.clone()),
                    target.span,
                )]),
            },
            target.span,
        );
        let mut remainder = Vec::with_capacity(1 + rest.len());
        remainder.push(bind_stmt);
        remainder.extend_from_slice(rest);

        let cont_body = build_nontail_continuation_body(
            &remainder,
            handled_ops,
            orig_arms_array,
            effect_vars,
            needs_closure,
            nt_counter,
            target.span,
        );
        let cont_lambda = crate::ast::Expr::new(
            crate::ast::ExprKind::Lambda {
                params: vec![(resumed_name, resumed_ty)],
                return_type: None,
                body: Box::new(crate::ast::Expr::new(
                    crate::ast::ExprKind::Block(cont_body),
                    target.span,
                )),
            },
            target.span,
        );
        let desugared_args =
            desugar_expr_effects(&args_arr, effect_vars, needs_closure, nt_counter);
        let mut out = prefix;
        out.push(make_yield_op_return(
            op_lit,
            desugared_args,
            cont_lambda,
            target.span,
        ));
        return out;
    }

    // Subcase 2: bare `__lust_perform("Op", [args])` statement
    if let crate::ast::StmtKind::Expr(e) = &target.kind
        && let Some((op_lit, args_arr)) = match_direct_handled_perform(e, handled_ops)
    {
        let id = *nt_counter;
        *nt_counter += 1;
        let resumed_name = format!("__nt_resumed_{id}");
        let resumed_ty = resume_hint.cloned();
        let cont_body = if rest.is_empty() {
            vec![crate::ast::Stmt::new(
                crate::ast::StmtKind::Return(vec![crate::ast::Expr::new(
                    crate::ast::ExprKind::Identifier(resumed_name.clone()),
                    target.span,
                )]),
                target.span,
            )]
        } else {
            build_nontail_continuation_body(
                rest,
                handled_ops,
                orig_arms_array,
                effect_vars,
                needs_closure,
                nt_counter,
                target.span,
            )
        };
        let cont_lambda = crate::ast::Expr::new(
            crate::ast::ExprKind::Lambda {
                params: vec![(resumed_name, resumed_ty)],
                return_type: None,
                body: Box::new(crate::ast::Expr::new(
                    crate::ast::ExprKind::Block(cont_body),
                    target.span,
                )),
            },
            target.span,
        );
        let desugared_args =
            desugar_expr_effects(&args_arr, effect_vars, needs_closure, nt_counter);
        let mut out = prefix;
        out.push(make_yield_op_return(
            op_lit,
            desugared_args,
            cont_lambda,
            target.span,
        ));
        return out;
    }

    // Subcase 3a: `StmtKind::If` whose condition has no handled `perform`
    if let crate::ast::StmtKind::If {
        condition,
        then_block,
        elseif_branches,
        else_block,
    } = &target.kind
        && !expr_contains_handled_perform(condition, handled_ops)
    {
        let mut then_full = then_block.clone();
        then_full.extend_from_slice(rest);
        let new_then = split_body_stmts_for_effect_yield(
            &then_full,
            handled_ops,
            orig_arms_array,
            resume_hint,
            effect_vars,
            needs_closure,
            nt_counter,
            target.span,
        );
        let mut new_elseifs = Vec::with_capacity(elseif_branches.len());
        for (ec, eb) in elseif_branches {
            let mut eb_full = eb.clone();
            eb_full.extend_from_slice(rest);
            new_elseifs.push((
                desugar_expr_effects(ec, effect_vars, needs_closure, nt_counter),
                split_body_stmts_for_effect_yield(
                    &eb_full,
                    handled_ops,
                    orig_arms_array,
                    resume_hint,
                    effect_vars,
                    needs_closure,
                    nt_counter,
                    target.span,
                ),
            ));
        }
        let mut else_full = else_block.clone().unwrap_or_default();
        else_full.extend_from_slice(rest);
        let new_else = split_body_stmts_for_effect_yield(
            &else_full,
            handled_ops,
            orig_arms_array,
            resume_hint,
            effect_vars,
            needs_closure,
            nt_counter,
            target.span,
        );
        let mut out = prefix;
        out.push(crate::ast::Stmt::new(
            crate::ast::StmtKind::If {
                condition: desugar_expr_effects(condition, effect_vars, needs_closure, nt_counter),
                then_block: new_then,
                elseif_branches: new_elseifs,
                else_block: Some(new_else),
            },
            target.span,
        ));
        return out;
    }

    // Subcase 3b: `StmtKind::Expr(ExprKind::If { .. })` (unwrapping parens) whose condition has no handled `perform`
    if let crate::ast::StmtKind::Expr(e) = &target.kind {
        let mut unwrapped = e;
        while let crate::ast::ExprKind::Paren(inner) = &unwrapped.kind {
            unwrapped = inner;
        }
        if let crate::ast::ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } = &unwrapped.kind
            && !expr_contains_handled_perform(condition, handled_ops)
        {
            let mut then_full = expr_to_stmts(then_branch);
            then_full.extend_from_slice(rest);
            let new_then = split_body_stmts_for_effect_yield(
                &then_full,
                handled_ops,
                orig_arms_array,
                resume_hint,
                effect_vars,
                needs_closure,
                nt_counter,
                target.span,
            );
            let mut else_full = else_branch
                .as_deref()
                .map(expr_to_stmts)
                .unwrap_or_default();
            else_full.extend_from_slice(rest);
            let new_else = split_body_stmts_for_effect_yield(
                &else_full,
                handled_ops,
                orig_arms_array,
                resume_hint,
                effect_vars,
                needs_closure,
                nt_counter,
                target.span,
            );
            let mut out = prefix;
            out.push(crate::ast::Stmt::new(
                crate::ast::StmtKind::If {
                    condition: desugar_expr_effects(
                        condition,
                        effect_vars,
                        needs_closure,
                        nt_counter,
                    ),
                    then_block: new_then,
                    elseif_branches: Vec::new(),
                    else_block: Some(new_else),
                },
                target.span,
            ));
            return out;
        }
    }

    // Subcase 4: Extract leftmost handled `perform` from an arbitrary expression inside `target`
    let id = *nt_counter;
    *nt_counter += 1;
    let tmp_name = format!("__nt_p_{id}");
    if let Some((rewritten_target, op_lit, args_arr)) =
        extract_first_handled_perform_in_stmt(target, handled_ops, &tmp_name)
    {
        let perf_call = crate::ast::Expr::new(
            crate::ast::ExprKind::Call {
                callee: Box::new(crate::ast::Expr::new(
                    crate::ast::ExprKind::Identifier("__lust_perform".to_string()),
                    target.span,
                )),
                type_args: None,
                args: vec![op_lit, args_arr],
            },
            target.span,
        );
        let bind_tmp = crate::ast::Stmt::new(
            crate::ast::StmtKind::Local {
                bindings: vec![crate::ast::LocalBinding {
                    name: tmp_name,
                    type_annotation: resume_hint.cloned(),
                    span: target.span,
                }],
                initializer: Some(vec![perf_call]),
            },
            target.span,
        );
        let mut rewritten = Vec::with_capacity(2 + rest.len());
        rewritten.push(bind_tmp);
        rewritten.push(rewritten_target);
        rewritten.extend_from_slice(rest);
        let mut out = prefix;
        out.extend(split_body_stmts_for_effect_yield(
            &rewritten,
            handled_ops,
            orig_arms_array,
            resume_hint,
            effect_vars,
            needs_closure,
            nt_counter,
            target.span,
        ));
        return out;
    }

    // Fallback (unreachable unless perform is inside an unsupported construct)
    let desugared = desugar_stmts_effects(stmts, effect_vars, needs_closure, false, nt_counter);
    wrap_stmts_as_yield_completed(&desugared, fallback_span)
}

fn build_nontail_continuation_body(
    remainder: &[crate::ast::Stmt],
    handled_ops: &std::collections::HashSet<String>,
    orig_arms_array: &crate::ast::Expr,
    effect_vars: &mut std::collections::HashSet<String>,
    needs_closure: &std::collections::HashSet<String>,
    nt_counter: &mut usize,
    span: Span,
) -> Vec<crate::ast::Stmt> {
    let has_more = remainder
        .iter()
        .any(|s| stmt_contains_handled_perform(s, handled_ops));
    if has_more {
        let nested_thunk = crate::ast::Expr::new(
            crate::ast::ExprKind::Lambda {
                params: Vec::new(),
                return_type: None,
                body: Box::new(crate::ast::Expr::new(
                    crate::ast::ExprKind::Block(remainder.to_vec()),
                    span,
                )),
            },
            span,
        );
        let nested_handle = crate::ast::Expr::new(
            crate::ast::ExprKind::Call {
                callee: Box::new(crate::ast::Expr::new(
                    crate::ast::ExprKind::Identifier("__lust_handle".to_string()),
                    span,
                )),
                type_args: None,
                args: vec![nested_thunk, orig_arms_array.clone()],
            },
            span,
        );
        let desugared = desugar_expr_effects(&nested_handle, effect_vars, needs_closure, nt_counter);
        vec![crate::ast::Stmt::new(
            crate::ast::StmtKind::Return(vec![desugared]),
            span,
        )]
    } else {
        let mut desugared =
            desugar_stmts_effects(remainder, effect_vars, needs_closure, false, nt_counter);
        if let Some(last) = desugared.last_mut()
            && let crate::ast::StmtKind::Expr(e) = &last.kind
        {
            let e_owned = e.clone();
            last.kind = crate::ast::StmtKind::Return(vec![e_owned]);
        }
        desugared
    }
}

fn make_yield_op_return(
    op_lit: crate::ast::Expr,
    args_arr: crate::ast::Expr,
    cont_lambda: crate::ast::Expr,
    span: Span,
) -> crate::ast::Stmt {
    crate::ast::Stmt::new(
        crate::ast::StmtKind::Return(vec![crate::ast::Expr::new(
            crate::ast::ExprKind::Call {
                callee: Box::new(crate::ast::Expr::new(
                    crate::ast::ExprKind::Identifier("__lust_yield_op".to_string()),
                    span,
                )),
                type_args: None,
                args: vec![op_lit, args_arr, cont_lambda],
            },
            span,
        )]),
        span,
    )
}

fn make_yield_completed_expr(val: crate::ast::Expr, span: Span) -> crate::ast::Expr {
    crate::ast::Expr::new(
        crate::ast::ExprKind::Call {
            callee: Box::new(crate::ast::Expr::new(
                crate::ast::ExprKind::Identifier("__lust_yield_completed".to_string()),
                span,
            )),
            type_args: None,
            args: vec![val],
        },
        span,
    )
}

fn make_unit_expr(span: Span) -> crate::ast::Expr {
    crate::ast::Expr::new(crate::ast::ExprKind::Tuple(Vec::new()), span)
}

fn wrap_stmts_as_yield_completed(
    stmts: &[crate::ast::Stmt],
    fallback_span: Span,
) -> Vec<crate::ast::Stmt> {
    if stmts.is_empty() {
        let unit_lit = make_unit_expr(fallback_span);
        return vec![crate::ast::Stmt::new(
            crate::ast::StmtKind::Return(vec![make_yield_completed_expr(unit_lit, fallback_span)]),
            fallback_span,
        )];
    }
    let mut out = Vec::with_capacity(stmts.len());
    let last_idx = stmts.len() - 1;
    for (idx, stmt) in stmts.iter().enumerate() {
        let is_last = idx == last_idx;
        let new_kind = match &stmt.kind {
            crate::ast::StmtKind::Return(vals) => {
                let v = vals
                    .first()
                    .cloned()
                    .unwrap_or_else(|| make_unit_expr(stmt.span));
                crate::ast::StmtKind::Return(vec![make_yield_completed_expr(v, stmt.span)])
            }
            crate::ast::StmtKind::Expr(e) if is_last => {
                crate::ast::StmtKind::Return(vec![make_yield_completed_expr(e.clone(), stmt.span)])
            }
            crate::ast::StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => crate::ast::StmtKind::If {
                condition: condition.clone(),
                then_block: wrap_branch_returns_as_yield_completed(then_block, is_last, stmt.span),
                elseif_branches: elseif_branches
                    .iter()
                    .map(|(ec, eb)| {
                        (
                            ec.clone(),
                            wrap_branch_returns_as_yield_completed(eb, is_last, stmt.span),
                        )
                    })
                    .collect(),
                else_block: else_block.as_ref().map(|eb| {
                    wrap_branch_returns_as_yield_completed(eb, is_last, stmt.span)
                }),
            },
            other => other.clone(),
        };
        out.push(crate::ast::Stmt::new(new_kind, stmt.span));
    }
    if !matches!(
        out.last().map(|s| &s.kind),
        Some(crate::ast::StmtKind::Return(_) | crate::ast::StmtKind::If { else_block: Some(_), .. })
    ) {
        let unit_lit = make_unit_expr(fallback_span);
        out.push(crate::ast::Stmt::new(
            crate::ast::StmtKind::Return(vec![make_yield_completed_expr(unit_lit, fallback_span)]),
            fallback_span,
        ));
    }
    out
}

fn wrap_branch_returns_as_yield_completed(
    stmts: &[crate::ast::Stmt],
    wrap_tail: bool,
    span: Span,
) -> Vec<crate::ast::Stmt> {
    if wrap_tail {
        return wrap_stmts_as_yield_completed(stmts, span);
    }
    stmts
        .iter()
        .map(|stmt| match &stmt.kind {
            crate::ast::StmtKind::Return(vals) => {
                let v = vals
                    .first()
                    .cloned()
                    .unwrap_or_else(|| make_unit_expr(stmt.span));
                crate::ast::Stmt::new(
                    crate::ast::StmtKind::Return(vec![make_yield_completed_expr(v, stmt.span)]),
                    stmt.span,
                )
            }
            _ => stmt.clone(),
        })
        .collect()
}

fn expr_to_stmts(expr: &crate::ast::Expr) -> Vec<crate::ast::Stmt> {
    match &expr.kind {
        crate::ast::ExprKind::Block(s) => s.clone(),
        _ => vec![crate::ast::Stmt::new(
            crate::ast::StmtKind::Expr(expr.clone()),
            expr.span,
        )],
    }
}

fn match_direct_handled_perform(
    expr: &crate::ast::Expr,
    handled_ops: &std::collections::HashSet<String>,
) -> Option<(crate::ast::Expr, crate::ast::Expr)> {
    match &expr.kind {
        crate::ast::ExprKind::Paren(inner) => match_direct_handled_perform(inner, handled_ops),
        crate::ast::ExprKind::Call { callee, args, .. }
            if matches!(&callee.kind, crate::ast::ExprKind::Identifier(n) if n == "__lust_perform" || n == "perform")
                && args.len() >= 2 =>
        {
            if let crate::ast::ExprKind::Literal(crate::ast::Literal::String(op_name)) =
                &args[0].kind
                && handled_ops.contains(op_name)
            {
                return Some((args[0].clone(), args[1].clone()));
            }
            None
        }
        _ => None,
    }
}

fn stmt_contains_handled_perform(
    stmt: &crate::ast::Stmt,
    handled_ops: &std::collections::HashSet<String>,
) -> bool {
    match &stmt.kind {
        crate::ast::StmtKind::Local {
            initializer: Some(es),
            ..
        }
        | crate::ast::StmtKind::Return(es) => {
            es.iter().any(|e| expr_contains_handled_perform(e, handled_ops))
        }
        crate::ast::StmtKind::Assign { targets, values } => {
            targets
                .iter()
                .any(|e| expr_contains_handled_perform(e, handled_ops))
                || values
                    .iter()
                    .any(|e| expr_contains_handled_perform(e, handled_ops))
        }
        crate::ast::StmtKind::CompoundAssign { target, value, .. } => {
            expr_contains_handled_perform(target, handled_ops)
                || expr_contains_handled_perform(value, handled_ops)
        }
        crate::ast::StmtKind::Expr(e) => expr_contains_handled_perform(e, handled_ops),
        crate::ast::StmtKind::If {
            condition,
            then_block,
            elseif_branches,
            else_block,
        } => {
            expr_contains_handled_perform(condition, handled_ops)
                || then_block
                    .iter()
                    .any(|s| stmt_contains_handled_perform(s, handled_ops))
                || elseif_branches.iter().any(|(c, b)| {
                    expr_contains_handled_perform(c, handled_ops)
                        || b.iter().any(|s| stmt_contains_handled_perform(s, handled_ops))
                })
                || else_block.as_ref().is_some_and(|b| {
                    b.iter().any(|s| stmt_contains_handled_perform(s, handled_ops))
                })
        }
        crate::ast::StmtKind::While { condition, body } => {
            expr_contains_handled_perform(condition, handled_ops)
                || body
                    .iter()
                    .any(|s| stmt_contains_handled_perform(s, handled_ops))
        }
        crate::ast::StmtKind::ForNumeric {
            start,
            end,
            step,
            body,
            ..
        } => {
            expr_contains_handled_perform(start, handled_ops)
                || expr_contains_handled_perform(end, handled_ops)
                || step
                    .as_ref()
                    .is_some_and(|st| expr_contains_handled_perform(st, handled_ops))
                || body
                    .iter()
                    .any(|s| stmt_contains_handled_perform(s, handled_ops))
        }
        crate::ast::StmtKind::ForIn { iterator, body, .. } => {
            expr_contains_handled_perform(iterator, handled_ops)
                || body
                    .iter()
                    .any(|s| stmt_contains_handled_perform(s, handled_ops))
        }
        crate::ast::StmtKind::Block(b) | crate::ast::StmtKind::Scoped(b) => {
            b.iter().any(|s| stmt_contains_handled_perform(s, handled_ops))
        }
        crate::ast::StmtKind::Defer(s) | crate::ast::StmtKind::ErrDefer(s) => {
            stmt_contains_handled_perform(s, handled_ops)
        }
        _ => false,
    }
}

fn expr_contains_handled_perform(
    expr: &crate::ast::Expr,
    handled_ops: &std::collections::HashSet<String>,
) -> bool {
    if match_direct_handled_perform(expr, handled_ops).is_some() {
        return true;
    }
    match &expr.kind {
        // Do not cross into nested closures, nested handlers, or effect blocks
        crate::ast::ExprKind::Lambda { .. } | crate::ast::ExprKind::EffectBlock(_) => false,
        crate::ast::ExprKind::Call { callee, args, .. } => {
            if matches!(&callee.kind, crate::ast::ExprKind::Identifier(n) if n == "__lust_handle" || n == "__lust_nontail_handle")
            {
                return false;
            }
            expr_contains_handled_perform(callee, handled_ops)
                || args
                    .iter()
                    .any(|a| expr_contains_handled_perform(a, handled_ops))
        }
        crate::ast::ExprKind::MethodCall { receiver, args, .. } => {
            expr_contains_handled_perform(receiver, handled_ops)
                || args
                    .iter()
                    .any(|a| expr_contains_handled_perform(a, handled_ops))
        }
        crate::ast::ExprKind::Binary { left, right, .. }
        | crate::ast::ExprKind::Range {
            start: left,
            end: right,
            ..
        } => {
            expr_contains_handled_perform(left, handled_ops)
                || expr_contains_handled_perform(right, handled_ops)
        }
        crate::ast::ExprKind::Unary { operand: inner, .. }
        | crate::ast::ExprKind::FieldAccess { object: inner, .. }
        | crate::ast::ExprKind::Paren(inner)
        | crate::ast::ExprKind::Cast { expr: inner, .. }
        | crate::ast::ExprKind::TypeCheck { expr: inner, .. }
        | crate::ast::ExprKind::IsPattern { expr: inner, .. } => {
            expr_contains_handled_perform(inner, handled_ops)
        }
        crate::ast::ExprKind::Index { object, index } => {
            expr_contains_handled_perform(object, handled_ops)
                || expr_contains_handled_perform(index, handled_ops)
        }
        crate::ast::ExprKind::Array(es)
        | crate::ast::ExprKind::Tuple(es)
        | crate::ast::ExprKind::EnumConstructor { args: es, .. }
        | crate::ast::ExprKind::Return(es) => es
            .iter()
            .any(|e| expr_contains_handled_perform(e, handled_ops)),
        crate::ast::ExprKind::Map(pairs) => pairs.iter().any(|(k, v)| {
            expr_contains_handled_perform(k, handled_ops)
                || expr_contains_handled_perform(v, handled_ops)
        }),
        crate::ast::ExprKind::StructLiteral { fields, .. } => fields
            .iter()
            .any(|f| expr_contains_handled_perform(&f.value, handled_ops)),
        crate::ast::ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            expr_contains_handled_perform(condition, handled_ops)
                || expr_contains_handled_perform(then_branch, handled_ops)
                || else_branch
                    .as_deref()
                    .is_some_and(|eb| expr_contains_handled_perform(eb, handled_ops))
        }
        crate::ast::ExprKind::Block(b) => {
            b.iter().any(|s| stmt_contains_handled_perform(s, handled_ops))
        }
        _ => false,
    }
}

fn extract_first_handled_perform_in_stmt(
    stmt: &crate::ast::Stmt,
    handled_ops: &std::collections::HashSet<String>,
    tmp_name: &str,
) -> Option<(crate::ast::Stmt, crate::ast::Expr, crate::ast::Expr)> {
    match &stmt.kind {
        crate::ast::StmtKind::Local {
            bindings,
            initializer: Some(inits),
        } => {
            let mut new_inits = inits.clone();
            for (idx, e) in inits.iter().enumerate() {
                if let Some((ne, op, args)) =
                    extract_first_handled_perform_in_expr(e, handled_ops, tmp_name)
                {
                    new_inits[idx] = ne;
                    return Some((
                        crate::ast::Stmt::new(
                            crate::ast::StmtKind::Local {
                                bindings: bindings.clone(),
                                initializer: Some(new_inits),
                            },
                            stmt.span,
                        ),
                        op,
                        args,
                    ));
                }
            }
            None
        }
        crate::ast::StmtKind::Return(es) => {
            let mut new_es = es.clone();
            for (idx, e) in es.iter().enumerate() {
                if let Some((ne, op, args)) =
                    extract_first_handled_perform_in_expr(e, handled_ops, tmp_name)
                {
                    new_es[idx] = ne;
                    return Some((
                        crate::ast::Stmt::new(crate::ast::StmtKind::Return(new_es), stmt.span),
                        op,
                        args,
                    ));
                }
            }
            None
        }
        crate::ast::StmtKind::Assign { targets, values } => {
            let mut new_vals = values.clone();
            for (idx, e) in values.iter().enumerate() {
                if let Some((ne, op, args)) =
                    extract_first_handled_perform_in_expr(e, handled_ops, tmp_name)
                {
                    new_vals[idx] = ne;
                    return Some((
                        crate::ast::Stmt::new(
                            crate::ast::StmtKind::Assign {
                                targets: targets.clone(),
                                values: new_vals,
                            },
                            stmt.span,
                        ),
                        op,
                        args,
                    ));
                }
            }
            None
        }
        crate::ast::StmtKind::CompoundAssign { target, op, value } => {
            if let Some((nv, perf_op, args)) =
                extract_first_handled_perform_in_expr(value, handled_ops, tmp_name)
            {
                return Some((
                    crate::ast::Stmt::new(
                        crate::ast::StmtKind::CompoundAssign {
                            target: target.clone(),
                            op: *op,
                            value: nv,
                        },
                        stmt.span,
                    ),
                    perf_op,
                    args,
                ));
            }
            None
        }
        crate::ast::StmtKind::Expr(e) => {
            let (ne, op, args) = extract_first_handled_perform_in_expr(e, handled_ops, tmp_name)?;
            Some((
                crate::ast::Stmt::new(crate::ast::StmtKind::Expr(ne), stmt.span),
                op,
                args,
            ))
        }
        crate::ast::StmtKind::If {
            condition,
            then_block,
            elseif_branches,
            else_block,
        } => {
            let (nc, op, args) =
                extract_first_handled_perform_in_expr(condition, handled_ops, tmp_name)?;
            Some((
                crate::ast::Stmt::new(
                    crate::ast::StmtKind::If {
                        condition: nc,
                        then_block: then_block.clone(),
                        elseif_branches: elseif_branches.clone(),
                        else_block: else_block.clone(),
                    },
                    stmt.span,
                ),
                op,
                args,
            ))
        }
        _ => None,
    }
}

fn extract_first_handled_perform_in_expr(
    expr: &crate::ast::Expr,
    handled_ops: &std::collections::HashSet<String>,
    tmp_name: &str,
) -> Option<(crate::ast::Expr, crate::ast::Expr, crate::ast::Expr)> {
    if let Some((op, args)) = match_direct_handled_perform(expr, handled_ops) {
        return Some((
            crate::ast::Expr::new(
                crate::ast::ExprKind::Identifier(tmp_name.to_string()),
                expr.span,
            ),
            op,
            args,
        ));
    }
    let span = expr.span;
    match &expr.kind {
        crate::ast::ExprKind::Lambda { .. } | crate::ast::ExprKind::EffectBlock(_) => None,
        crate::ast::ExprKind::Binary { left, op, right } => {
            if let Some((nl, pop, pargs)) =
                extract_first_handled_perform_in_expr(left, handled_ops, tmp_name)
            {
                return Some((
                    crate::ast::Expr::new(
                        crate::ast::ExprKind::Binary {
                            left: Box::new(nl),
                            op: *op,
                            right: right.clone(),
                        },
                        span,
                    ),
                    pop,
                    pargs,
                ));
            }
            if let Some((nr, pop, pargs)) =
                extract_first_handled_perform_in_expr(right, handled_ops, tmp_name)
            {
                return Some((
                    crate::ast::Expr::new(
                        crate::ast::ExprKind::Binary {
                            left: left.clone(),
                            op: *op,
                            right: Box::new(nr),
                        },
                        span,
                    ),
                    pop,
                    pargs,
                ));
            }
            None
        }
        crate::ast::ExprKind::Unary { op, operand } => {
            let (no, pop, pargs) =
                extract_first_handled_perform_in_expr(operand, handled_ops, tmp_name)?;
            Some((
                crate::ast::Expr::new(
                    crate::ast::ExprKind::Unary {
                        op: *op,
                        operand: Box::new(no),
                    },
                    span,
                ),
                pop,
                pargs,
            ))
        }
        crate::ast::ExprKind::Paren(inner) => {
            let (ni, pop, pargs) =
                extract_first_handled_perform_in_expr(inner, handled_ops, tmp_name)?;
            Some((
                crate::ast::Expr::new(crate::ast::ExprKind::Paren(Box::new(ni)), span),
                pop,
                pargs,
            ))
        }
        crate::ast::ExprKind::Call {
            callee,
            type_args,
            args,
        } => {
            if matches!(&callee.kind, crate::ast::ExprKind::Identifier(n) if n == "__lust_handle" || n == "__lust_nontail_handle")
            {
                return None;
            }
            let mut new_args = args.clone();
            for (idx, a) in args.iter().enumerate() {
                if let Some((na, pop, pargs)) =
                    extract_first_handled_perform_in_expr(a, handled_ops, tmp_name)
                {
                    new_args[idx] = na;
                    return Some((
                        crate::ast::Expr::new(
                            crate::ast::ExprKind::Call {
                                callee: callee.clone(),
                                type_args: type_args.clone(),
                                args: new_args,
                            },
                            span,
                        ),
                        pop,
                        pargs,
                    ));
                }
            }
            None
        }
        crate::ast::ExprKind::Array(es) => {
            let mut new_es = es.clone();
            for (idx, e) in es.iter().enumerate() {
                if let Some((ne, pop, pargs)) =
                    extract_first_handled_perform_in_expr(e, handled_ops, tmp_name)
                {
                    new_es[idx] = ne;
                    return Some((
                        crate::ast::Expr::new(crate::ast::ExprKind::Array(new_es), span),
                        pop,
                        pargs,
                    ));
                }
            }
            None
        }
        crate::ast::ExprKind::Cast {
            expr: inner,
            target_type,
        } => {
            let (ni, pop, pargs) =
                extract_first_handled_perform_in_expr(inner, handled_ops, tmp_name)?;
            Some((
                crate::ast::Expr::new(
                    crate::ast::ExprKind::Cast {
                        expr: Box::new(ni),
                        target_type: target_type.clone(),
                    },
                    span,
                ),
                pop,
                pargs,
            ))
        }
        _ => None,
    }
}

fn rewrite_resume_to_nontail_expr(expr: &crate::ast::Expr) -> crate::ast::Expr {
    let span = expr.span;
    match &expr.kind {
        crate::ast::ExprKind::Call {
            callee,
            type_args,
            args,
        } => {
            // Do NOT rewrite resumes belonging to a nested handle's arms
            if matches!(&callee.kind, crate::ast::ExprKind::Identifier(n) if n == "__lust_handle" || n == "__lust_nontail_handle")
                && args.len() == 2
            {
                return crate::ast::Expr::new(
                    crate::ast::ExprKind::Call {
                        callee: callee.clone(),
                        type_args: type_args.clone(),
                        args: vec![rewrite_resume_to_nontail_expr(&args[0]), args[1].clone()],
                    },
                    span,
                );
            }
            let new_callee = if matches!(&callee.kind, crate::ast::ExprKind::Identifier(n) if n == "__lust_resume" || n == "resume")
            {
                crate::ast::Expr::new(
                    crate::ast::ExprKind::Identifier("__lust_nontail_resume".to_string()),
                    callee.span,
                )
            } else {
                rewrite_resume_to_nontail_expr(callee)
            };
            crate::ast::Expr::new(
                crate::ast::ExprKind::Call {
                    callee: Box::new(new_callee),
                    type_args: type_args.clone(),
                    args: args.iter().map(rewrite_resume_to_nontail_expr).collect(),
                },
                span,
            )
        }
        crate::ast::ExprKind::MethodCall {
            receiver,
            method,
            type_args,
            args,
        } => crate::ast::Expr::new(
            crate::ast::ExprKind::MethodCall {
                receiver: Box::new(rewrite_resume_to_nontail_expr(receiver)),
                method: method.clone(),
                type_args: type_args.clone(),
                args: args.iter().map(rewrite_resume_to_nontail_expr).collect(),
            },
            span,
        ),
        crate::ast::ExprKind::Binary { left, op, right } => crate::ast::Expr::new(
            crate::ast::ExprKind::Binary {
                left: Box::new(rewrite_resume_to_nontail_expr(left)),
                op: *op,
                right: Box::new(rewrite_resume_to_nontail_expr(right)),
            },
            span,
        ),
        crate::ast::ExprKind::Unary { op, operand } => crate::ast::Expr::new(
            crate::ast::ExprKind::Unary {
                op: *op,
                operand: Box::new(rewrite_resume_to_nontail_expr(operand)),
            },
            span,
        ),
        crate::ast::ExprKind::Paren(inner) => crate::ast::Expr::new(
            crate::ast::ExprKind::Paren(Box::new(rewrite_resume_to_nontail_expr(inner))),
            span,
        ),
        crate::ast::ExprKind::Cast {
            expr: inner,
            target_type,
        } => crate::ast::Expr::new(
            crate::ast::ExprKind::Cast {
                expr: Box::new(rewrite_resume_to_nontail_expr(inner)),
                target_type: target_type.clone(),
            },
            span,
        ),
        crate::ast::ExprKind::FieldAccess { object, field } => crate::ast::Expr::new(
            crate::ast::ExprKind::FieldAccess {
                object: Box::new(rewrite_resume_to_nontail_expr(object)),
                field: field.clone(),
            },
            span,
        ),
        crate::ast::ExprKind::Index { object, index } => crate::ast::Expr::new(
            crate::ast::ExprKind::Index {
                object: Box::new(rewrite_resume_to_nontail_expr(object)),
                index: Box::new(rewrite_resume_to_nontail_expr(index)),
            },
            span,
        ),
        crate::ast::ExprKind::Array(es) => crate::ast::Expr::new(
            crate::ast::ExprKind::Array(es.iter().map(rewrite_resume_to_nontail_expr).collect()),
            span,
        ),
        crate::ast::ExprKind::Tuple(es) => crate::ast::Expr::new(
            crate::ast::ExprKind::Tuple(es.iter().map(rewrite_resume_to_nontail_expr).collect()),
            span,
        ),
        crate::ast::ExprKind::Return(es) => crate::ast::Expr::new(
            crate::ast::ExprKind::Return(es.iter().map(rewrite_resume_to_nontail_expr).collect()),
            span,
        ),
        crate::ast::ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => crate::ast::Expr::new(
            crate::ast::ExprKind::If {
                condition: Box::new(rewrite_resume_to_nontail_expr(condition)),
                then_branch: Box::new(rewrite_resume_to_nontail_expr(then_branch)),
                else_branch: else_branch
                    .as_ref()
                    .map(|eb| Box::new(rewrite_resume_to_nontail_expr(eb))),
            },
            span,
        ),
        crate::ast::ExprKind::Block(stmts) => crate::ast::Expr::new(
            crate::ast::ExprKind::Block(rewrite_resume_to_nontail_stmts(stmts)),
            span,
        ),
        _ => expr.clone(),
    }
}

fn rewrite_resume_to_nontail_stmts(stmts: &[crate::ast::Stmt]) -> Vec<crate::ast::Stmt> {
    stmts
        .iter()
        .map(|stmt| {
            let nk = match &stmt.kind {
                crate::ast::StmtKind::Local {
                    bindings,
                    initializer,
                } => crate::ast::StmtKind::Local {
                    bindings: bindings.clone(),
                    initializer: initializer
                        .as_ref()
                        .map(|es| es.iter().map(rewrite_resume_to_nontail_expr).collect()),
                },
                crate::ast::StmtKind::Assign { targets, values } => crate::ast::StmtKind::Assign {
                    targets: targets.iter().map(rewrite_resume_to_nontail_expr).collect(),
                    values: values.iter().map(rewrite_resume_to_nontail_expr).collect(),
                },
                crate::ast::StmtKind::CompoundAssign { target, op, value } => {
                    crate::ast::StmtKind::CompoundAssign {
                        target: rewrite_resume_to_nontail_expr(target),
                        op: *op,
                        value: rewrite_resume_to_nontail_expr(value),
                    }
                }
                crate::ast::StmtKind::Expr(e) => {
                    crate::ast::StmtKind::Expr(rewrite_resume_to_nontail_expr(e))
                }
                crate::ast::StmtKind::Return(es) => {
                    crate::ast::StmtKind::Return(es.iter().map(rewrite_resume_to_nontail_expr).collect())
                }
                crate::ast::StmtKind::If {
                    condition,
                    then_block,
                    elseif_branches,
                    else_block,
                } => crate::ast::StmtKind::If {
                    condition: rewrite_resume_to_nontail_expr(condition),
                    then_block: rewrite_resume_to_nontail_stmts(then_block),
                    elseif_branches: elseif_branches
                        .iter()
                        .map(|(ec, eb)| {
                            (
                                rewrite_resume_to_nontail_expr(ec),
                                rewrite_resume_to_nontail_stmts(eb),
                            )
                        })
                        .collect(),
                    else_block: else_block
                        .as_ref()
                        .map(|eb| rewrite_resume_to_nontail_stmts(eb)),
                },
                crate::ast::StmtKind::While { condition, body } => crate::ast::StmtKind::While {
                    condition: rewrite_resume_to_nontail_expr(condition),
                    body: rewrite_resume_to_nontail_stmts(body),
                },
                crate::ast::StmtKind::Block(b) => {
                    crate::ast::StmtKind::Block(rewrite_resume_to_nontail_stmts(b))
                }
                crate::ast::StmtKind::Scoped(b) => {
                    crate::ast::StmtKind::Scoped(rewrite_resume_to_nontail_stmts(b))
                }
                other => other.clone(),
            };
            crate::ast::Stmt::new(nk, stmt.span)
        })
        .collect()
}

fn infer_simple_ast_expr_type(
    expr: &crate::ast::Expr,
    env: &std::collections::HashMap<String, Type>,
) -> Option<Type> {
    let span = expr.span;
    match &expr.kind {
        crate::ast::ExprKind::Paren(inner) => infer_simple_ast_expr_type(inner, env),
        crate::ast::ExprKind::Literal(lit) => Some(Type {
            kind: match lit {
                crate::ast::Literal::Integer(_) => TypeKind::Int,
                crate::ast::Literal::Float(_) => TypeKind::Float,
                crate::ast::Literal::Bool(_) => TypeKind::Bool,
                crate::ast::Literal::String(_) => TypeKind::String,
            },
            span,
        }),
        crate::ast::ExprKind::Tuple(elems) if elems.is_empty() => Some(Type {
            kind: TypeKind::Unit,
            span,
        }),
        crate::ast::ExprKind::Identifier(name) => {
            if name == "nil" {
                Some(Type {
                    kind: TypeKind::Unit,
                    span,
                })
            } else {
                env.get(name).cloned()
            }
        }
        crate::ast::ExprKind::Unary { op, operand } => match op {
            crate::ast::UnaryOp::Not => Some(Type {
                kind: TypeKind::Bool,
                span,
            }),
            _ => infer_simple_ast_expr_type(operand, env),
        },
        crate::ast::ExprKind::Binary { left, op, right } => match op {
            crate::ast::BinaryOp::Eq
            | crate::ast::BinaryOp::Ne
            | crate::ast::BinaryOp::Lt
            | crate::ast::BinaryOp::Le
            | crate::ast::BinaryOp::Gt
            | crate::ast::BinaryOp::Ge
            | crate::ast::BinaryOp::And
            | crate::ast::BinaryOp::Or => Some(Type {
                kind: TypeKind::Bool,
                span,
            }),
            crate::ast::BinaryOp::Concat => Some(Type {
                kind: TypeKind::String,
                span,
            }),
            _ => {
                let lt = infer_simple_ast_expr_type(left, env);
                let rt = infer_simple_ast_expr_type(right, env);
                if matches!(lt.as_ref().map(|t| &t.kind), Some(TypeKind::Float))
                    || matches!(rt.as_ref().map(|t| &t.kind), Some(TypeKind::Float))
                {
                    Some(Type {
                        kind: TypeKind::Float,
                        span,
                    })
                } else {
                    lt.or(rt)
                }
            }
        },
        crate::ast::ExprKind::Cast { target_type, .. } => Some(target_type.clone()),
        _ => None,
    }
}

fn collect_perform_arg_hints_in_stmts(
    stmts: &[crate::ast::Stmt],
    handled_ops: &std::collections::HashSet<String>,
    env: &mut std::collections::HashMap<String, Type>,
    out: &mut std::collections::HashMap<String, Vec<Option<Type>>>,
) {
    for stmt in stmts {
        match &stmt.kind {
            crate::ast::StmtKind::Local {
                bindings,
                initializer,
            } => {
                if let Some(inits) = initializer {
                    for (idx, e) in inits.iter().enumerate() {
                        collect_perform_arg_hints_in_expr(e, handled_ops, env, out);
                        if let Some(b) = bindings.get(idx) {
                            let ty = b
                                .type_annotation
                                .clone()
                                .or_else(|| infer_simple_ast_expr_type(e, env));
                            if let Some(t) = ty {
                                env.insert(b.name.clone(), t);
                            }
                        }
                    }
                }
            }
            crate::ast::StmtKind::Return(es) => {
                for e in es {
                    collect_perform_arg_hints_in_expr(e, handled_ops, env, out);
                }
            }
            crate::ast::StmtKind::Assign { targets, values } => {
                for e in targets.iter().chain(values.iter()) {
                    collect_perform_arg_hints_in_expr(e, handled_ops, env, out);
                }
            }
            crate::ast::StmtKind::CompoundAssign { target, value, .. } => {
                collect_perform_arg_hints_in_expr(target, handled_ops, env, out);
                collect_perform_arg_hints_in_expr(value, handled_ops, env, out);
            }
            crate::ast::StmtKind::Expr(e) => {
                collect_perform_arg_hints_in_expr(e, handled_ops, env, out);
            }
            crate::ast::StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                collect_perform_arg_hints_in_expr(condition, handled_ops, env, out);
                collect_perform_arg_hints_in_stmts(then_block, handled_ops, env, out);
                for (ec, eb) in elseif_branches {
                    collect_perform_arg_hints_in_expr(ec, handled_ops, env, out);
                    collect_perform_arg_hints_in_stmts(eb, handled_ops, env, out);
                }
                if let Some(eb) = else_block {
                    collect_perform_arg_hints_in_stmts(eb, handled_ops, env, out);
                }
            }
            crate::ast::StmtKind::Block(b) | crate::ast::StmtKind::Scoped(b) => {
                collect_perform_arg_hints_in_stmts(b, handled_ops, env, out);
            }
            _ => {}
        }
    }
}

fn collect_perform_arg_hints_in_expr(
    expr: &crate::ast::Expr,
    handled_ops: &std::collections::HashSet<String>,
    env: &mut std::collections::HashMap<String, Type>,
    out: &mut std::collections::HashMap<String, Vec<Option<Type>>>,
) {
    if let Some((op_lit, args_arr)) = match_direct_handled_perform(expr, handled_ops)
        && let crate::ast::ExprKind::Literal(crate::ast::Literal::String(op_name)) = &op_lit.kind
        && let crate::ast::ExprKind::Array(elems) = &args_arr.kind
    {
        let inferred: Vec<Option<Type>> = elems
            .iter()
            .map(|e| infer_simple_ast_expr_type(e, env))
            .collect();
        out.entry(op_name.clone())
            .and_modify(|prev| {
                for (pi, cand) in inferred.iter().enumerate() {
                    if let Some(slot) = prev.get_mut(pi)
                        && slot.is_none()
                    {
                        *slot = cand.clone();
                    }
                }
            })
            .or_insert(inferred);
        return;
    }
    match &expr.kind {
        crate::ast::ExprKind::Block(stmts) => {
            collect_perform_arg_hints_in_stmts(stmts, handled_ops, env, out);
        }
        crate::ast::ExprKind::Binary { left, right, .. } => {
            collect_perform_arg_hints_in_expr(left, handled_ops, env, out);
            collect_perform_arg_hints_in_expr(right, handled_ops, env, out);
        }
        crate::ast::ExprKind::Unary { operand: inner, .. }
        | crate::ast::ExprKind::Paren(inner)
        | crate::ast::ExprKind::Cast { expr: inner, .. } => {
            collect_perform_arg_hints_in_expr(inner, handled_ops, env, out);
        }
        crate::ast::ExprKind::Call { callee, args, .. } => {
            if matches!(&callee.kind, crate::ast::ExprKind::Identifier(n) if n == "__lust_handle" || n == "__lust_nontail_handle")
            {
                return;
            }
            for a in args {
                collect_perform_arg_hints_in_expr(a, handled_ops, env, out);
            }
        }
        crate::ast::ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_perform_arg_hints_in_expr(condition, handled_ops, env, out);
            collect_perform_arg_hints_in_expr(then_branch, handled_ops, env, out);
            if let Some(eb) = else_branch {
                collect_perform_arg_hints_in_expr(eb, handled_ops, env, out);
            }
        }
        _ => {}
    }
}

fn collect_resume_arg_hint_in_stmts(
    stmts: &[crate::ast::Stmt],
    env: &mut std::collections::HashMap<String, Type>,
    hint: &mut Option<Type>,
    conflict: &mut bool,
) {
    for stmt in stmts {
        match &stmt.kind {
            crate::ast::StmtKind::Local {
                bindings,
                initializer: Some(inits),
            } => {
                for (idx, e) in inits.iter().enumerate() {
                    collect_resume_arg_hint_in_expr(e, env, hint, conflict);
                    if let Some(b) = bindings.get(idx) {
                        let ty = b
                            .type_annotation
                            .clone()
                            .or_else(|| infer_simple_ast_expr_type(e, env));
                        if let Some(t) = ty {
                            env.insert(b.name.clone(), t);
                        }
                    }
                }
            }
            crate::ast::StmtKind::Return(es) => {
                for e in es {
                    collect_resume_arg_hint_in_expr(e, env, hint, conflict);
                }
            }
            crate::ast::StmtKind::Assign { values, .. } => {
                for e in values {
                    collect_resume_arg_hint_in_expr(e, env, hint, conflict);
                }
            }
            crate::ast::StmtKind::CompoundAssign { value, .. } | crate::ast::StmtKind::Expr(value) => {
                collect_resume_arg_hint_in_expr(value, env, hint, conflict);
            }
            crate::ast::StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                collect_resume_arg_hint_in_expr(condition, env, hint, conflict);
                collect_resume_arg_hint_in_stmts(then_block, env, hint, conflict);
                for (ec, eb) in elseif_branches {
                    collect_resume_arg_hint_in_expr(ec, env, hint, conflict);
                    collect_resume_arg_hint_in_stmts(eb, env, hint, conflict);
                }
                if let Some(eb) = else_block {
                    collect_resume_arg_hint_in_stmts(eb, env, hint, conflict);
                }
            }
            crate::ast::StmtKind::Block(b) | crate::ast::StmtKind::Scoped(b) => {
                collect_resume_arg_hint_in_stmts(b, env, hint, conflict);
            }
            _ => {}
        }
    }
}

fn collect_resume_arg_hint_in_expr(
    expr: &crate::ast::Expr,
    env: &mut std::collections::HashMap<String, Type>,
    hint: &mut Option<Type>,
    conflict: &mut bool,
) {
    match &expr.kind {
        crate::ast::ExprKind::Call { callee, args, .. } => {
            if matches!(&callee.kind, crate::ast::ExprKind::Identifier(n) if n == "__lust_resume" || n == "resume")
                && let Some(arg0) = args.first()
                && let Some(ty) = infer_simple_ast_expr_type(arg0, env)
            {
                match hint {
                    None => *hint = Some(ty),
                    Some(prev) if prev.kind == ty.kind => {}
                    Some(_) => *conflict = true,
                }
                return;
            }
            if matches!(&callee.kind, crate::ast::ExprKind::Identifier(n) if n == "__lust_handle" || n == "__lust_nontail_handle")
            {
                return;
            }
            for a in args {
                collect_resume_arg_hint_in_expr(a, env, hint, conflict);
            }
        }
        crate::ast::ExprKind::Block(stmts) => {
            collect_resume_arg_hint_in_stmts(stmts, env, hint, conflict);
        }
        crate::ast::ExprKind::Binary { left, right, .. } => {
            collect_resume_arg_hint_in_expr(left, env, hint, conflict);
            collect_resume_arg_hint_in_expr(right, env, hint, conflict);
        }
        crate::ast::ExprKind::Unary { operand: inner, .. }
        | crate::ast::ExprKind::Paren(inner)
        | crate::ast::ExprKind::Cast { expr: inner, .. } => {
            collect_resume_arg_hint_in_expr(inner, env, hint, conflict);
        }
        crate::ast::ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_resume_arg_hint_in_expr(condition, env, hint, conflict);
            collect_resume_arg_hint_in_expr(then_branch, env, hint, conflict);
            if let Some(eb) = else_branch {
                collect_resume_arg_hint_in_expr(eb, env, hint, conflict);
            }
        }
        _ => {}
    }
}

fn find_rustc() -> String {
    if let Ok(rustc) = std::env::var("RUSTC") {
        if !rustc.trim().is_empty() {
            return rustc;
        }
    }
    "rustc".to_string()
}

fn find_aot_rt_lib_rs() -> Result<PathBuf> {
    if let Ok(override_path) = std::env::var("LUST_AOT_RT_SRC") {
        let p = PathBuf::from(override_path);
        if p.exists() {
            return Ok(p);
        }
    }

    let manifest_candidate = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("crates")
        .join("lust-aot-rt")
        .join("src")
        .join("lib.rs");
    if manifest_candidate.exists() {
        return Ok(manifest_candidate);
    }

    if let Ok(cwd) = std::env::current_dir() {
        let cwd_candidate = cwd
            .join("crates")
            .join("lust-aot-rt")
            .join("src")
            .join("lib.rs");
        if cwd_candidate.exists() {
            return Ok(cwd_candidate);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.ancestors() {
            let cand = ancestor
                .join("crates")
                .join("lust-aot-rt")
                .join("src")
                .join("lib.rs");
            if cand.exists() {
                return Ok(cand);
            }
        }
    }

    Err(LustError::Unknown(
        "Could not locate crates/lust-aot-rt/src/lib.rs for Rust AOT runtime compilation"
            .to_string(),
    ))
}

/// Locates the `crates/lust-aot-rt` Cargo crate directory for `--emit=rust-crate` (`NG-W13-1C`).
pub fn find_aot_rt_crate_dir() -> Result<PathBuf> {
    let lib_rs = find_aot_rt_lib_rs()?;
    let crate_dir = lib_rs
        .parent()
        .and_then(|src| src.parent())
        .ok_or_else(|| {
            LustError::Unknown("Invalid crates/lust-aot-rt/src/lib.rs path".to_string())
        })?;
    Ok(crate_dir.to_path_buf())
}

fn sanitize_cargo_package_name(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches(|c| c == '-' || c == '_');
    if trimmed.is_empty() {
        "lust_aot_crate".to_string()
    } else if trimmed
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_digit())
    {
        format!("lust_{trimmed}")
    } else {
        trimmed.to_string()
    }
}

fn escape_toml_string(raw: &str) -> String {
    raw.replace('\\', "\\\\").replace('"', "\\\"")
}

fn resolve_aot_cache_dir(explicit: Option<&Path>) -> PathBuf {
    if let Some(dir) = explicit {
        return dir.to_path_buf();
    }
    if let Ok(env_dir) = std::env::var("LUST_AOT_CACHE_DIR") {
        if !env_dir.trim().is_empty() {
            return PathBuf::from(env_dir);
        }
    }
    std::env::temp_dir().join("lust-aot-rlib-cache")
}

fn release_mode_to_opt_level(release_mode: &str) -> &'static str {
    match release_mode {
        "ReleaseFast" | "release" | "ReleaseSafe" => "3",
        "ReleaseSmall" => "z",
        "Debug" | "debug" => "0",
        other if other.starts_with("Release") => "3",
        _ => "0",
    }
}

/// Ensures `liblust_aot_rt.rlib` is compiled and cached for the active `rustc`,
/// source tree, optimization level, `no_std` mode, and target triple (`NG-W12-4A`).
///
/// Subsequent invocations hit the cached `.rlib` directly in `< 1 ms`.
pub fn ensure_cached_aot_rt_rlib(opts: &RustCompileOptions) -> Result<PathBuf> {
    let rustc = find_rustc();
    let rt_lib_rs = find_aot_rt_lib_rs()?;
    let rt_src_dir = rt_lib_rs
        .parent()
        .ok_or_else(|| LustError::Unknown("Invalid lust-aot-rt lib.rs path".to_string()))?;

    let opt_level = release_mode_to_opt_level(&opts.aot.release_mode);
    let target_tag = opts.aot.target_triple.as_deref().unwrap_or("host");
    let std_tag = if opts.no_std { "nostd" } else { "std" };

    // Compute deterministic cache key from rustc version + all lust-aot-rt sources + flags
    let mut hasher = DefaultHasher::new();
    "lust-aot-rt-v1".hash(&mut hasher);
    opt_level.hash(&mut hasher);
    target_tag.hash(&mut hasher);
    std_tag.hash(&mut hasher);

    if let Ok(ver_out) = Command::new(&rustc).arg("-vV").output() {
        ver_out.stdout.hash(&mut hasher);
    }

    let mut src_files: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = fs::read_dir(rt_src_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) == Some("rs") {
                src_files.push(p);
            }
        }
    }
    src_files.sort();
    for src_file in &src_files {
        if let Some(fname) = src_file.file_name().and_then(|s| s.to_str()) {
            fname.hash(&mut hasher);
        }
        let bytes = fs::read(src_file).map_err(|e| {
            LustError::Unknown(format!("Failed to read {}: {e}", src_file.display()))
        })?;
        bytes.hash(&mut hasher);
    }

    let key = format!("{:016x}", hasher.finish());
    let cache_root = resolve_aot_cache_dir(opts.cache_dir.as_deref());
    let bucket_dir = cache_root.join(format!("rt-{std_tag}-O{opt_level}-{key}"));
    let rlib_path = bucket_dir.join("liblust_aot_rt.rlib");

    if rlib_path.exists()
        && fs::metadata(&rlib_path)
            .map(|m| m.len() > 0)
            .unwrap_or(false)
    {
        return Ok(rlib_path);
    }

    fs::create_dir_all(&bucket_dir).map_err(|e| {
        LustError::Unknown(format!(
            "Failed to create AOT cache directory {}: {e}",
            bucket_dir.display()
        ))
    })?;

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let tmp_rlib = bucket_dir.join(format!(
        "liblust_aot_rt.{}.{}.tmp.rlib",
        std::process::id(),
        nanos
    ));

    let mut cmd = Command::new(&rustc);
    cmd.arg("--edition=2024")
        .arg("--crate-name")
        .arg("lust_aot_rt")
        .arg("--crate-type=rlib")
        .arg("-C")
        .arg(format!("opt-level={opt_level}"))
        .arg("-C")
        .arg("panic=abort")
        .arg("-C")
        .arg("embed-bitcode=yes")
        .arg("-C")
        .arg("codegen-units=1");

    if !opts.no_std {
        cmd.arg("--cfg").arg("feature=\"std\"");
    }
    if let Some(triple) = &opts.aot.target_triple {
        cmd.arg("--target").arg(triple);
    }
    cmd.arg(&rt_lib_rs).arg("-o").arg(&tmp_rlib);

    let output = cmd.output().map_err(|e| {
        LustError::Unknown(format!(
            "Failed to execute {rustc} for lust-aot-rt rlib build: {e}"
        ))
    })?;

    if !output.status.success() {
        let _ = fs::remove_file(&tmp_rlib);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(LustError::Unknown(format!(
            "Failed to build cached liblust_aot_rt.rlib (status {}):\n{stderr}",
            output.status
        )));
    }

    // Atomic publish into cache (parallel-test safe)
    if fs::rename(&tmp_rlib, &rlib_path).is_err() {
        let _ = fs::remove_file(&tmp_rlib);
        if !rlib_path.exists() {
            return Err(LustError::Unknown(format!(
                "Failed to publish cached {}",
                rlib_path.display()
            )));
        }
    }

    Ok(rlib_path)
}

/// Compiles generated Rust AOT source directly via `rustc` linked against cached `liblust_aot_rt.rlib`.
fn compile_rust_source(
    rust_source: &str,
    output_bin: &Path,
    opts: &RustCompileOptions,
    links: &AotLinkOptions,
) -> Result<PathBuf> {
    let rust_links = RustLinkOptions {
        aot_links: links.clone(),
        ..RustLinkOptions::default()
    };
    compile_rust_source_with_rlibs(rust_source, output_bin, opts, &rust_links)
}

/// Compiles generated Rust AOT source directly via `rustc` linked against cached `liblust_aot_rt.rlib`
/// plus optional direct Zero-FFI `.rlib` crates (`--extern <ext>=<path>.rlib`) with `-C lto=thin`
/// (`NG-W13-1C`).
fn compile_rust_source_with_rlibs(
    rust_source: &str,
    output_bin: &Path,
    opts: &RustCompileOptions,
    links: &RustLinkOptions,
) -> Result<PathBuf> {
    let has_external_links = !links.aot_links.link_libraries.is_empty()
        || !links.aot_links.link_dirs.is_empty()
        || !links.linked_crates.is_empty();
    if (!links.aot_links.allow_ffi || opts.pure_core) && has_external_links {
        return Err(LustError::Unknown(
            "CapabilityDenied: external library linking requires 'ffi' capability".to_string(),
        ));
    }
    for lib in &links.aot_links.link_libraries {
        if !lib.exists() {
            return Err(LustError::Unknown(format!(
                "missing external library at '{}'",
                lib.display()
            )));
        }
    }
    for lc in &links.linked_crates {
        if !lc.rlib_path.exists() {
            return Err(LustError::Unknown(format!(
                "missing external library at '{}'",
                lc.rlib_path.display()
            )));
        }
    }

    let rlib_path = if let Some(override_path) = links
        .aot_rt_rlib_override
        .clone()
        .or_else(|| {
            std::env::var_os("LUST_AOT_RT_RLIB")
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
        }) {
        if !override_path.exists() {
            return Err(LustError::Unknown(format!(
                "LUST_AOT_RT_RLIB override not found at '{}'",
                override_path.display()
            )));
        }
        override_path
    } else {
        ensure_cached_aot_rt_rlib(opts)?
    };
    let rustc = find_rustc();
    let opt_level = release_mode_to_opt_level(&opts.aot.release_mode);

    let temp_dir = tempfile::tempdir()
        .map_err(|e| LustError::Unknown(format!("Failed to create tempdir: {e}")))?;
    let rs_file = temp_dir.path().join("aot_payload.rs");
    fs::write(&rs_file, rust_source)
        .map_err(|e| LustError::Unknown(format!("Failed to write Rust payload: {e}")))?;

    if let Some(parent) = output_bin.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let mut cmd = Command::new(&rustc);
    cmd.arg("--edition=2024")
        .arg("-C")
        .arg(format!("opt-level={opt_level}"))
        .arg("-C")
        .arg("panic=abort")
        .arg("-C")
        .arg("codegen-units=1");

    if links.lto_thin || !links.linked_crates.is_empty() {
        cmd.arg("-C").arg("lto=thin");
    }
    if let Some(pgo_gen_dir) = &links.pgo_generate {
        let _ = fs::create_dir_all(pgo_gen_dir);
        cmd.arg(format!("-Cprofile-generate={}", pgo_gen_dir.display()));
    }
    if let Some(pgo_use_path) = &links.pgo_use {
        cmd.arg(format!("-Cprofile-use={}", pgo_use_path.display()));
        cmd.arg("-Cllvm-args=-pgo-warn-missing-function");
    }
    if let Some(pgo_sample_path) = &links.pgo_sample_use {
        cmd.arg(format!("-Cprofile-sample-use={}", pgo_sample_path.display()));
    }
    if opt_level == "0" {
        cmd.arg("-C").arg("debuginfo=2");
    }
    if opts.aot.strip {
        cmd.arg("-C").arg("strip=symbols");
    }
    if let Some(triple) = &opts.aot.target_triple {
        cmd.arg("--target").arg(triple);
    }
    match opts.aot.target_kind {
        AotTargetKind::Executable => {}
        AotTargetKind::SharedLibrary => {
            cmd.arg("--crate-type=cdylib");
        }
        AotTargetKind::StaticLibrary => {
            cmd.arg("--crate-type=staticlib");
        }
    }

    cmd.arg("--extern")
        .arg(format!("lust_aot_rt={}", rlib_path.display()));

    for lc in &links.linked_crates {
        cmd.arg("--extern")
            .arg(format!("{}={}", lc.crate_name, lc.rlib_path.display()));
        for dep_dir in &lc.dependency_dirs {
            cmd.arg("-L")
                .arg(format!("dependency={}", dep_dir.display()));
        }
    }

    for dir in &links.aot_links.link_dirs {
        cmd.arg("-L").arg(format!("native={}", dir.display()));
    }
    for lib in &links.aot_links.link_libraries {
        cmd.arg("-C").arg(format!("link-arg={}", lib.display()));
    }

    cmd.arg(&rs_file).arg("-o").arg(output_bin);

    let output = cmd
        .output()
        .map_err(|e| LustError::Unknown(format!("Failed to execute {rustc}: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(LustError::Unknown(format!(
            "Rust AOT compilation failed with status {}:\n{stderr}\nGenerated Rust Source:\n{rust_source}",
            output.status
        )));
    }

    Ok(output_bin.to_path_buf())
}

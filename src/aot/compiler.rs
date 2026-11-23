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
use super::rust_emitter::{GasMetering, RustEmitterOptions, emit_rust_from_mir};
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
        let module = AstLowering::lower_module_items("main_module", &prepared_items)
            .map_err(|e| LustError::Unknown(format!("MIR lowering error: {e}")))?;

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

    /// Runs the full frontend (`Lexer -> Parser -> TypeChecker -> prepare_items_for_mir ->
    /// AstLowering -> MirOptimizer + SROA + refine + EscapeAnalysis -> cfg_structurer -> Rust`)
    /// on a Lust source string and returns the generated Rust code.
    pub fn emit_rust_source(source: &str, opts: &RustCompileOptions) -> Result<String> {
        let module = lower_and_typecheck_source(source)?;
        Self::emit_rust_module(&module, opts)
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
        let module = lower_and_typecheck_source(source)?;
        Self::compile_module_rust_with_links(&module, output_bin, opts, links)
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

fn lower_and_typecheck_source(source: &str) -> Result<MirModule> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .map_err(|e| LustError::Unknown(format!("Lexer error: {e}")))?;
    let items = Parser::new(tokens)
        .parse()
        .map_err(|e| LustError::Unknown(format!("Parser error: {e}")))?;

    let config = crate::config::LustConfig::default();
    let mut checker = crate::typechecker::TypeChecker::with_config(&config);
    checker
        .check_module(&items)
        .map_err(|e| LustError::Unknown(format!("TypeChecker error: {e}")))?;

    let prepared_items = prepare_items_for_mir(&items);
    AstLowering::lower_module_items("main_module", &prepared_items)
        .map_err(|e| LustError::Unknown(format!("MIR lowering error: {e}")))
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

    let rlib_path = ensure_cached_aot_rt_rlib(opts)?;
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

    for dir in &links.link_dirs {
        cmd.arg("-L").arg(format!("native={}", dir.display()));
    }
    for lib in &links.link_libraries {
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

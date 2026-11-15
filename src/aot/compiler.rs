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

/// Autonomous AOT Compiler: Produces native release binaries without requiring system `zig`.
pub struct AotCompiler;

impl AotCompiler {
    /// Compiles a MIR module into a native machine-code binary or library.
    pub fn compile_module(
        module: &MirModule,
        output_bin: &Path,
        opts: &AotCompileOptions,
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
        compile_c_source(&c_source, output_bin, opts)
    }

    /// Compiles Lust source code string into a native machine-code binary.
    pub fn compile_source(
        source: &str,
        output_bin: &Path,
        opts: &AotCompileOptions,
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

        Self::compile_module(&module, output_bin, opts)
    }

    /// Compiles a `.lust` source file on disk into a native binary.
    pub fn compile_file(
        source_path: &Path,
        output_bin: &Path,
        opts: &AotCompileOptions,
    ) -> Result<PathBuf> {
        let source = fs::read_to_string(source_path)
            .map_err(|e| LustError::Unknown(format!("Failed to read {}: {e}", source_path.display())))?;
        Self::compile_source(&source, output_bin, opts)
    }

    /// Fallback invoked when system `zig` is unavailable or disabled.
    pub fn compile_zig_fallback(
        zig_path: &Path,
        output_bin: &Path,
        opts: &crate::codegen_zig::CompileOptions,
        _link_libc: bool,
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
            return Self::compile_file(&lust_path, output_bin, &aot_opts);
        }

        // Try to extract embedded source from the .zig file
        if zig_path.exists() {
            if let Ok(zig_content) = fs::read_to_string(zig_path) {
                if let Some(source) = extract_embedded_lust_source(&zig_content) {
                    return Self::compile_source(&source, output_bin, &aot_opts);
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
fn prepare_items_for_mir(items: &[Item]) -> Vec<Item> {
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

    let mut cmd = Command::new(&cc);

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

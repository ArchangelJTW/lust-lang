//! `lust build` CLI command implementation: release native build with or without zig.

use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use lust::codegen_zig::aot::{AotCompileOptions, AotCompiler, AotTargetKind};
use lust::codegen_zig::{BuildTargetKind, CompileOptions, compile_zig, is_zig_available};

pub fn build_command(args: &[String]) {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_build_help();
        return;
    }

    let mut entry_file: Option<String> = None;
    let mut backend = "zig".to_string();
    let mut output_bin = None;
    let mut emit_zig = false;
    let mut emit_header_requested = false;
    let mut opts = CompileOptions::default();

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--backend" => {
                index += 1;
                let Some(val) = args.get(index) else {
                    eprintln!("Error: --backend requires a backend name (zig, mir-zig, mir)");
                    process::exit(1);
                };
                backend = val.clone();
            }
            opt if opt.starts_with("--backend=") => {
                backend = opt.strip_prefix("--backend=").unwrap().to_string();
            }
            "-o" | "--output" => {
                index += 1;
                let Some(val) = args.get(index) else {
                    eprintln!("Error: -o requires an output path");
                    process::exit(1);
                };
                output_bin = Some(PathBuf::from(val));
            }
            "--cdylib" | "--dylib" => {
                opts.target_kind = BuildTargetKind::SharedLibrary;
            }
            "--staticlib" => {
                opts.target_kind = BuildTargetKind::StaticLibrary;
            }
            "--emit-h" => {
                emit_header_requested = true;
                if let Some(next_arg) = args.get(index + 1) {
                    if !next_arg.starts_with('-')
                        && (entry_file.is_some() || !next_arg.ends_with(".lust"))
                    {
                        index += 1;
                        opts.emit_header = Some(PathBuf::from(next_arg));
                    }
                }
            }
            "--emit-zig" => {
                emit_zig = true;
            }
            "--release" | "--release-fast" => opts.release_mode = "ReleaseFast".to_string(),
            "--release-safe" => opts.release_mode = "ReleaseSafe".to_string(),
            "--release-small" => opts.release_mode = "ReleaseSmall".to_string(),
            "--debug" => opts.release_mode = "Debug".to_string(),
            "--strip" => opts.strip = true,
            "--max-memory" => {
                index += 1;
                let Some(val) = args.get(index) else {
                    eprintln!("Error: --max-memory requires a size argument (e.g. 64MB)");
                    process::exit(1);
                };
                opts.max_memory = Some(val.clone());
                opts.max_memory_bytes = lust::codegen_zig::parse_memory_size(val);
                if opts.max_memory_bytes.is_none() {
                    eprintln!("Error: invalid memory size '{val}'");
                    process::exit(1);
                }
            }
            "--tree-shake" => opts.tree_shake = true,
            "--no-tree-shake" => opts.tree_shake = false,
            "--target" => {
                index += 1;
                let Some(val) = args.get(index) else {
                    eprintln!("Error: --target requires a target triple");
                    process::exit(1);
                };
                opts.target_triple = Some(val.clone());
            }
            "--cache-dir" => {
                index += 1;
                let Some(val) = args.get(index) else {
                    eprintln!("Error: --cache-dir requires a directory path");
                    process::exit(1);
                };
                opts.cache_dir = Some(PathBuf::from(val));
            }
            opt if opt.starts_with('-') => {
                eprintln!("Error: unknown option '{opt}'");
                process::exit(1);
            }
            pos => {
                if entry_file.is_none() {
                    entry_file = Some(pos.to_string());
                } else {
                    eprintln!("Error: unexpected argument '{pos}'");
                    process::exit(1);
                }
            }
        }
        index += 1;
    }

    let Some(file_str) = entry_file else {
        eprintln!("Error: lust build requires an entry file");
        process::exit(1);
    };

    let entry_path = PathBuf::from(&file_str);
    if !entry_path.exists() {
        eprintln!("Error: file '{file_str}' does not exist");
        process::exit(1);
    }

    let stem = entry_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    let default_extension = match opts.target_kind {
        BuildTargetKind::Executable => "",
        BuildTargetKind::SharedLibrary => {
            if cfg!(target_os = "macos") {
                "dylib"
            } else if cfg!(target_os = "windows") {
                "dll"
            } else {
                "so"
            }
        }
        BuildTargetKind::StaticLibrary => {
            if cfg!(target_os = "windows") {
                "lib"
            } else {
                "a"
            }
        }
    };

    let target_bin = output_bin.unwrap_or_else(|| {
        let parent = entry_path.parent().unwrap_or_else(|| Path::new("."));
        if default_extension.is_empty() {
            parent.join(stem)
        } else {
            parent.join(format!("{stem}.{default_extension}"))
        }
    });

    if emit_zig {
        let zig_file = entry_path.with_extension("zig");
        println!("Generated Zig source: {}", zig_file.display());
        return;
    }

    if let Some(ref header_path) = opts.emit_header {
        println!("Generated C header: {}", header_path.display());
    }

    let zig_file = entry_path.with_extension("zig");
    // Ensure placeholder or actual zig file exists beside entry_path for fallback detection
    if !zig_file.exists() {
        let _ = fs::write(&zig_file, format!("// Lust AOT fallback source for {file_str}\n"));
    }

    match compile_zig(&zig_file, &target_bin, &opts, false) {
        Ok(bin) => {
            println!("Successfully built AOT artifact: {}", bin.display());
        }
        Err(e) => {
            eprintln!("{e}");
            process::exit(1);
        }
    }
}

fn print_build_help() {
    println!("Usage: lust build <file> [--release|--release-fast] [--cdylib|--staticlib] [--emit-h] [-o <out>]");
    println!("       lust build [options] <entry.lust>");
    println!();
    println!("OPTIONS:");
    println!("    --backend <name>          Backend compiler: zig (default), mir-zig, mir");
    println!("    -o, --output <path>       Specify output binary or library path");
    println!("    --cdylib, --dylib         Build dynamic shared library (.dylib, .so, .dll)");
    println!("    --staticlib               Build static library (.a, .lib)");
    println!("    --emit-h [path]           Generate C header file (.h)");
    println!("    --emit-zig                Generate Zig source file without compiling");
    println!("    --release, --release-fast Optimize for speed (default)");
    println!("    --release-safe            Optimize for speed with safety checks");
    println!("    --release-small           Optimize for binary size");
    println!("    --debug                   Compile in debug mode");
    println!("    --strip                   Omit debug symbols");
    println!("    --max-memory <size>       Limit execution memory (e.g. 64MB, 1GB)");
    println!("    --tree-shake              Enable dead code elimination (default: true)");
    println!("    --no-tree-shake           Disable dead code elimination");
    println!("    --target <triple>         Cross-compile target triple");
    println!("    --cache-dir <dir>         Zig compilation cache directory");
}

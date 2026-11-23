#![cfg(feature = "std")]
// Host CLI boundary: lust CLI binary runs on host OS and orchestrates process execution.
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]
#[cfg(all(feature = "cli-mimalloc", not(target_arch = "wasm32")))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
use lust::LoadedRustModule;
#[cfg(feature = "lua_transpile")]
use lust::lua_compat::transpile::transpile_lua_stub;
#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
use lust::lua_compat::{LuaModuleSpec, LuaValue, lua_to_lust, render_table_stub, trace_luaopen};
#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
use lust::packages::{
    DEFAULT_BASE_URL, DependencyResolution, DownloadedArchive, PackageDetails, PackageManager,
    PackageManifest, PreparedRustDependency, RegistryClient, build_package_archive,
    clear_credentials, collect_rust_dependency_artifacts, credentials_file, load_credentials,
    load_prepared_rust_dependencies, prepare_rust_dependencies, resolve_dependencies,
    save_credentials, write_stub_files,
};
use lust::error::IntoDiagnostic;
use lust::{Compiler, Item, LustConfig, ModuleLoader, Span, TypeChecker, VM};
use std::{
    env, fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::{self, Command},
};
#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
use toml::{self, Value, map::Map};
const VERSION: &str = env!("CARGO_PKG_VERSION");

mod cli {
    #[path = "diag.rs"]
    pub mod diag;
    #[path = "ext.rs"]
    pub mod ext;
    #[path = "test.rs"]
    pub mod test;
    #[path = "sim.rs"]
    pub mod sim;
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage(&args[0]);
        process::exit(1);
    }

    #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
    if let Some(pos) = args.iter().position(|a| a == "--test-load-extension" || a.starts_with("--test-load-extension=")) {
        let dylib_path = if args[pos] == "--test-load-extension" {
            if pos + 1 < args.len() {
                PathBuf::from(&args[pos + 1])
            } else {
                eprintln!("Error: IncompatibleAbiVersion: --test-load-extension requires a path argument");
                process::exit(1);
            }
        } else {
            PathBuf::from(args[pos].strip_prefix("--test-load-extension=").unwrap())
        };

        let canary_path = if let Some(cpos) = args.iter().position(|a| a == "--host-canary-path" || a.starts_with("--host-canary-path=")) {
            if args[cpos] == "--host-canary-path" {
                if cpos + 1 < args.len() {
                    Some(PathBuf::from(&args[cpos + 1]))
                } else {
                    eprintln!("Error: IncompatibleAbiVersion: --host-canary-path requires a path argument");
                    process::exit(1);
                }
            } else {
                Some(PathBuf::from(args[cpos].strip_prefix("--host-canary-path=").unwrap()))
            }
        } else {
            None
        };

        match lust::packages::test_load_extension_cli(&dylib_path, canary_path.as_deref()) {
            Ok(()) => process::exit(0),
            Err(err) => {
                eprintln!("Error: {err}");
                process::exit(1);
            }
        }
    }

    if args.iter().any(|a| a == "--test-hang-probe") {
        std::thread::sleep(std::time::Duration::from_secs(30));
        process::exit(0);
    }

    // Collect all paths from --transpile / --tr flags before dispatching.
    // Each flag consumes subsequent non-flag arguments as paths, so both
    //   lust --transpile .
    //   lust --tr a.lua b.lua   (shell-expanded glob)
    //   lust --tr a.lua --tr b.lua
    // all work as expected.
    let mut transpile_paths: Vec<String> = Vec::new();
    {
        let mut i = 1usize;
        while i < args.len() {
            if args[i] == "--transpile" || args[i] == "--tr" {
                i += 1;
                while i < args.len() && !args[i].starts_with('-') {
                    transpile_paths.push(args[i].clone());
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
    }
    if !transpile_paths.is_empty() {
        #[cfg(feature = "lua_transpile")]
        {
            transpile_lua_files(&transpile_paths);
            return;
        }
        #[cfg(not(feature = "lua_transpile"))]
        {
            eprintln!(
                "Warning: --transpile requires the lua_transpile feature; recompile with --features lua_transpile to enable."
            );
            process::exit(1);
        }
    }

    match args[1].as_str() {
        "--help" | "-h" => {
            print_help(&args[0]);
            return;
        }

        "--version" | "-v" => {
            print_version();
            return;
        }

        "--dump-externs" => {
            if args.len() < 3 {
                eprintln!("Error: --dump-externs requires a file argument");
                process::exit(1);
            }

            dump_externs(&args[2]);
            return;
        }

        "bindgen" => {
            bindgen_command(&args[2..]);
            return;
        }

        "test" => {
            cli::test::test_command(&args[2..]);
            return;
        }

        "sim" => {
            cli::sim::sim_command(&args[2..]);
            return;
        }

        "ext" => {
            if let Err(err) = cli::ext::run_ext_command(&args[2..]) {
                eprintln!("Error: {err}");
                process::exit(1);
            }
            return;
        }

        "build" | "--aot" => {
            build_command(&args[2..]);
            return;
        }

        #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
        "pkg" => {
            if let Err(err) = handle_pkg_command(&args) {
                eprintln!("Error: {err}");
                process::exit(1);
            }
            return;
        }

        _ => {}
    }

    let run_args: &[String] = if args[1] == "run" {
        &args[2..]
    } else {
        &args[1..]
    };

    let mut disassemble = false;
    let mut use_mir = false;
    let mut strict_mir = false;
    let mut cranelift_jit = false;
    let mut no_gc = false;
    let mut pure_core = false;
    let mut backend_mode: Option<String> = None;
    let mut jit_mode: Option<String> = None;
    let mut script_file: Option<String> = None;
    let mut capture_failure: Option<String> = env::var("LUST_CAPTURE_FAILURE").ok();
    let mut json_diag = env::var("LUST_ERROR_FORMAT")
        .map(|v| v.eq_ignore_ascii_case("json"))
        .unwrap_or(false)
        || env::var("LUST_DIAG_FORMAT")
            .map(|v| v.eq_ignore_ascii_case("json"))
            .unwrap_or(false);

    let mut i = 0;
    while i < run_args.len() {
        let arg = &run_args[i];
        match arg.as_str() {
            "--help" | "-h" => {
                print_help(&args[0]);
                return;
            }
            "--error-format" | "--diag-format" | "--format" => {
                i += 1;
                let Some(val) = run_args.get(i) else {
                    eprintln!("Error: flag requires a format argument (human, json)");
                    process::exit(1);
                };
                if val.eq_ignore_ascii_case("json") {
                    json_diag = true;
                }
            }
            opt if opt.starts_with("--error-format=") => {
                let val = opt.strip_prefix("--error-format=").unwrap();
                if val.eq_ignore_ascii_case("json") {
                    json_diag = true;
                }
            }
            opt if opt.starts_with("--diag-format=") => {
                let val = opt.strip_prefix("--diag-format=").unwrap();
                if val.eq_ignore_ascii_case("json") {
                    json_diag = true;
                }
            }
            opt if opt.starts_with("--format=") => {
                let val = opt.strip_prefix("--format=").unwrap();
                if val.eq_ignore_ascii_case("json") {
                    json_diag = true;
                }
            }
            "--disassemble" | "-d" => disassemble = true,
            "--mir" => use_mir = true,
            "--strict-mir" => {
                strict_mir = true;
                use_mir = true;
            }
            "--strict" => {
                strict_mir = true;
                use_mir = true;
            }
            "--no-gc" => no_gc = true,
            "--pure-core" | "--no-prelude" => pure_core = true,
            "--capture-failure" | "--capture-on-failure" => {
                i += 1;
                let Some(val) = run_args.get(i) else {
                    eprintln!("Error: --capture-failure requires a path argument");
                    process::exit(1);
                };
                capture_failure = Some(val.clone());
            }
            opt if opt.starts_with("--capture-failure=") => {
                let val = opt.strip_prefix("--capture-failure=").unwrap();
                capture_failure = Some(val.to_string());
            }
            opt if opt.starts_with("--capture-on-failure=") => {
                let val = opt.strip_prefix("--capture-on-failure=").unwrap();
                capture_failure = Some(val.to_string());
            }
            "--panic-for-test" => {
                // Testing probe flag accepted
            }
            "--backend" => {
                i += 1;
                let Some(val) = run_args.get(i) else {
                    eprintln!("Error: --backend requires a mode argument (vm, mir, cranelift, zig, auto)");
                    process::exit(1);
                };
                backend_mode = Some(val.clone());
            }
            opt if opt.starts_with("--backend=") => {
                let val = opt.strip_prefix("--backend=").unwrap();
                backend_mode = Some(val.to_string());
            }
            "--jit" => {
                i += 1;
                let Some(val) = run_args.get(i) else {
                    eprintln!("Error: --jit requires a mode argument (cranelift, auto, off)");
                    process::exit(1);
                };
                jit_mode = Some(val.clone());
            }
            opt if opt.starts_with("--jit=") => {
                let val = opt.strip_prefix("--jit=").unwrap();
                jit_mode = Some(val.to_string());
            }
            other if other.starts_with('-') => {
                eprintln!("Unknown option '{}'", other);
                print_usage(&args[0]);
                process::exit(1);
            }
            pos => {
                if script_file.is_none() {
                    script_file = Some(pos.to_string());
                }
            }
        }
        i += 1;
    }

    let mut is_zig_aot = false;
    let mut is_rust_aot = false;
    if let Some(mode) = backend_mode {
        match mode.to_ascii_lowercase().as_str() {
            "vm" => {
                // Default Bytecode VM
            }
            "mir" | "mir-vm" => {
                use_mir = true;
            }
            "cranelift" | "jit" => {
                use_mir = true;
                cranelift_jit = true;
            }
            "zig" | "aot" => {
                is_zig_aot = true;
            }
            "rust" => {
                is_rust_aot = true;
            }
            "auto" => {
                // Auto: default VM with JIT/MIR
            }
            other => {
                eprintln!(
                    "Error: unknown backend '{}'. Supported backends for run: vm, mir, cranelift, zig, rust, auto",
                    other
                );
                process::exit(1);
            }
        }
    }
    let mut jit_disabled = false;
    if let Some(mode) = jit_mode {
        match mode.to_ascii_lowercase().as_str() {
            "cranelift" => {
                cranelift_jit = true;
            }
            "auto" | "default" => {
                cranelift_jit = false;
            }
            "off" | "none" | "vm" | "bytecode" => {
                cranelift_jit = false;
                jit_disabled = true;
            }
            other => {
                eprintln!(
                    "Error: unknown --jit mode '{}'. Supported modes: cranelift, auto, off",
                    other
                );
                process::exit(1);
            }
        }
    }

    if let Some(file) = script_file {
        if is_rust_aot {
            run_file_rust_aot(&file, pure_core);
        } else if is_zig_aot {
            run_file_zig_aot(&file);
        } else {
            run_file_with_options(
                &file,
                disassemble,
                use_mir,
                strict_mir,
                cranelift_jit,
                jit_disabled,
                no_gc,
                pure_core,
                json_diag,
                capture_failure.as_deref(),
            );
        }
    } else {
        print_usage(&args[0]);
        process::exit(1);
    }
}

fn print_usage(program: &str) {
    eprintln!("Usage: {} [options] <script.lust>", program);
    eprintln!(
        "       {} run <file> [--pure-core|--no-prelude] --backend [vm|cranelift|zig|auto] [--strict]",
        program
    );
    eprintln!(
        "       {} build <file> --backend zig [--cdylib|--staticlib] [--emit-h] [--max-memory <size>] [-o <out>]",
        program
    );
    #[cfg(feature = "bindgen")]
    eprintln!(
        "       {} bindgen <entry.lust> [--out <file.rs>] [--name Bindings]",
        program
    );
    eprintln!("       {} --help", program);
    eprintln!("       {} --version", program);
    #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
    {
        eprintln!("       {} pkg <command> [options]", program);
        eprintln!("       {} login [--token <token>]", program);
        eprintln!("       {} logout", program);
    }
}

fn print_help(program: &str) {
    println!("Lust Programming Language v{}", VERSION);
    println!();
    println!("USAGE:");
    println!(
        "    {} [options] <script.lust>         Run a Lust script",
        program
    );
    println!(
        "    {} run <file> --backend [vm|cranelift|zig|auto] [--strict]",
        program
    );
    println!(
        "                                        Run script with specified backend"
    );
    println!(
        "    {} build <file> --backend zig [--cdylib|--staticlib] [--emit-h] [--max-memory <size>] [-o <out>]",
        program
    );
    println!(
        "                                        Build native binary or C-ABI library via Zig AOT"
    );
    println!(
        "    {} --disassemble <script.lust>     Show bytecode disassembly",
        program
    );
    println!(
        "    {} --mir <script.lust>             Run script using Lust MIR pipeline",
        program
    );
    println!(
        "    {} --strict-mir <script.lust>      Run with strict MIR (reject fallback/unsupported)",
        program
    );
    println!(
        "    {} --no-gc <script.lust>           Run script with cycle collector GC disabled",
        program
    );
    println!(
        "    {} --pure-core, --no-prelude       Run with pure core (no ambient globals, require explicit use std.*)",
        program
    );
    println!(
        "    {} --help, -h                      Show this help message",
        program
    );
    println!(
        "    {} --version, -v                   Show version information",
        program
    );
    println!(
        "    {} --dump-externs <script.lust>    Create extern stubs for rust or Lua 5.1 C API library modules; reads from lust-config.toml",
        program
    );
    #[cfg(feature = "bindgen")]
    println!(
        "    {} bindgen <entry.lust>            Generate checked Rust bindings for @bindgen functions",
        program
    );
    #[cfg(feature = "lua_transpile")]
    {
        println!(
            "    {} --transpile, --tr <path> [...]  Transpile Lua file(s) or directories to Lust",
            program
        );
        println!("        Accepts files, directories (recursive), and shell-expanded globs.");
        println!("        Hidden dirs and node_modules are skipped automatically.");
        println!("        e.g.  {} --transpile src/lua/", program);
        println!("              {} --tr a.lua b.lua --tr c.lua", program);
    }
    #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
    {
        println!(
            "    {} pkg <command> [options]         Manage Lust packages",
            program
        );
    }
    println!();
    println!("EXAMPLES:");
    println!(
        "    {} script.lust                     Run script.lust",
        program
    );
    println!(
        "    {} run script.lust --backend zig   Compile and run via Zig AOT",
        program
    );
    println!(
        "    {} run script.lust --backend cranelift --strict",
        program
    );
    println!(
        "                                        Run via Cranelift JIT (strict MIR)"
    );
    println!(
        "    {} build app.lust --backend zig --cdylib",
        program
    );
    println!(
        "                                        Build shared library via Zig AOT"
    );
    println!(
        "    {} -d script.lust                  Disassemble script.lust",
        program
    );
}

fn write_placeholder_lua_stub(
    output_root: &Path,
    luaopen_symbol: &str,
    dep_name: &str,
) -> Result<Option<PathBuf>, io::Error> {
    let module_name = luaopen_symbol
        .strip_prefix("luaopen_")
        .unwrap_or(luaopen_symbol)
        .replace('_', ".");
    if module_name.is_empty() {
        return Ok(None);
    }
    let relative_path = module_name.replace('.', "/") + ".lust";
    let destination = output_root.join(&relative_path);
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    let contents = format!(
        "-- Placeholder extern stub for Lua module '{module}' from dependency '{dep}'.\n-- TODO: replace with traced API surface.\n\nextern\nend\n",
        module = module_name,
        dep = dep_name
    );
    fs::write(&destination, contents)?;
    Ok(Some(PathBuf::from(relative_path)))
}

fn write_lua_stub(
    output_root: &Path,
    module_name: &str,
    contents: &str,
) -> Result<PathBuf, io::Error> {
    let relative_path = module_name.replace('.', "/") + ".lust";
    let destination = output_root.join(&relative_path);
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&destination, contents)?;
    Ok(PathBuf::from(relative_path))
}

fn lua_module_name(path: &Path) -> String {
    path.with_extension("")
        .to_string_lossy()
        .replace(std::path::MAIN_SEPARATOR, ".")
}

fn print_version() {
    println!("Lust v{} - https://lust-lang.dev", VERSION);
}

/// Recursively collect all `.lua` files under `dir`, skipping hidden directories
/// (names starting with `.`) and `node_modules`.
#[cfg(feature = "lua_transpile")]
fn collect_lua_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(err) => {
            eprintln!(
                "Warning: could not read directory '{}': {}",
                dir.display(),
                err
            );
            return;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();
        if path.is_dir() {
            if name.starts_with('.') || name == "node_modules" {
                continue;
            }
            collect_lua_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("lua") {
            out.push(path);
        }
    }
}

#[cfg(feature = "lua_transpile")]
fn transpile_lua_files(paths: &[String]) {
    if paths.is_empty() {
        eprintln!("Warning: --transpile requires at least one file or directory argument");
        process::exit(1);
    }

    let mut lua_files: Vec<PathBuf> = Vec::new();
    for raw in paths {
        let path = PathBuf::from(raw);
        if path.is_dir() {
            collect_lua_files(&path, &mut lua_files);
        } else if path.extension().and_then(|e| e.to_str()) == Some("lua") {
            lua_files.push(path);
        } else {
            eprintln!(
                "Warning: '{}' is not a .lua file or directory; skipping",
                raw
            );
        }
    }

    if lua_files.is_empty() {
        eprintln!("Warning: no .lua files found in the given paths");
        return;
    }

    let mut ok = 0usize;
    let mut failed = 0usize;
    for lua_path in &lua_files {
        let content = match fs::read_to_string(lua_path) {
            Ok(c) => c,
            Err(err) => {
                eprintln!("Warning: could not read '{}': {}", lua_path.display(), err);
                failed += 1;
                continue;
            }
        };
        let module_name = lua_module_name(&lua_path.with_extension(""));
        match transpile_lua_stub(&content, &module_name) {
            Ok(stub) => {
                let dest = lua_path.with_extension("lust");
                if let Err(err) = fs::write(&dest, stub) {
                    eprintln!("Warning: could not write '{}': {}", dest.display(), err);
                    failed += 1;
                } else {
                    println!("{} -> {}", lua_path.display(), dest.display());
                    ok += 1;
                }
            }
            Err(err) => {
                eprintln!(
                    "Warning: transpilation failed for '{}': {}",
                    lua_path.display(),
                    err
                );
                failed += 1;
            }
        }
    }

    println!("Transpiled {ok} file(s), {failed} warning(s).");
    if failed > 0 {
        process::exit(1);
    }
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn handle_pkg_command(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        print_pkg_usage(&args[0]);
        return Err("pkg requires a subcommand".to_string());
    }

    let program = args[0].clone();
    let subcommand = args[2].as_str();
    let rest = &args[3..];

    match subcommand {
        "add" => handle_pkg_add(&program, rest),
        "remove" => handle_pkg_remove(&program, rest),
        "sync" => handle_pkg_sync(&program, rest),
        "login" => handle_pkg_login(&program, rest),
        "logout" => handle_pkg_logout(&program, rest),
        "publish" => handle_pkg_publish(&program, rest),
        "help" | "--help" | "-h" => {
            print_pkg_usage(&program);
            Ok(())
        }
        other => {
            print_pkg_usage(&program);
            Err(format!("unknown pkg command '{other}'"))
        }
    }
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn print_pkg_usage(program: &str) {
    println!("Package commands:");
    println!("  {program} pkg add <name[@version]> [--registry <url>]");
    println!("  {program} pkg remove <name>");
    println!("  {program} pkg sync [--registry <url>]");
    println!("  {program} pkg login [--token <token>]");
    println!("  {program} pkg logout");
    println!(
        "  {program} pkg publish [--manifest-path <path>] [--token <token>] [--registry <url>] [--readme <path>]"
    );
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn handle_pkg_add(program: &str, args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        print_pkg_usage(program);
        return Err("pkg add requires a package spec".to_string());
    }
    let spec = &args[0];
    let mut registry: Option<String> = None;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--registry" => {
                let value = take_option_value(args, &mut index, "--registry")?;
                registry = Some(value);
            }
            other => {
                return Err(format!("unknown option '{other}'"));
            }
        }
        index += 1;
    }

    let (name, requested_version) = parse_dependency_spec(spec)?;
    let registry_url = resolve_registry_base(registry.as_deref());
    let client = RegistryClient::new(&registry_url).map_err(|err| err.to_string())?;
    let details = client
        .package_details(&name)
        .map_err(|err| err.to_string())?;
    let (download_target, resolved_version) = match requested_version {
        Some(version) => {
            let exists = details.versions.iter().any(|info| info.version == version);
            if !exists {
                return Err(format!(
                    "package '{name}' does not have published version '{version}'"
                ));
            }
            (version.clone(), version)
        }
        None => {
            let latest = details
                .latest_version
                .and_then(|info| {
                    let trimmed = info.version.trim();
                    if trimmed.is_empty() {
                        None
                    } else {
                        Some(info.version.clone())
                    }
                })
                .ok_or_else(|| format!("package '{name}' does not have any published versions"))?;
            ("latest".to_string(), latest)
        }
    };

    let archive: DownloadedArchive = client
        .download_package(&name, &download_target)
        .map_err(|err| err.to_string())?;
    let manager = PackageManager::new(PackageManager::default_root());
    manager.ensure_layout().map_err(|err| err.to_string())?;
    let package_dir = manager.root().join(&name).join(&resolved_version);
    remove_existing_path(&package_dir).map_err(|err| err.to_string())?;
    extract_package_archive(archive.path(), &package_dir)?;

    update_dependency_in_config(&name, &resolved_version)?;
    println!(
        "Added {} {} (stored in {})",
        name,
        resolved_version,
        package_dir.display()
    );
    Ok(())
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn handle_pkg_sync(program: &str, args: &[String]) -> Result<(), String> {
    let mut registry: Option<String> = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--registry" => {
                let value = take_option_value(args, &mut index, "--registry")?;
                registry = Some(value);
            }
            other => {
                print_pkg_usage(program);
                return Err(format!("unknown option '{other}'"));
            }
        }
        index += 1;
    }

    let project_dir = env::current_dir().map_err(|err| err.to_string())?;
    let config = LustConfig::load_from_dir(&project_dir).map_err(|err| err.to_string())?;
    let dependencies = config.dependencies().to_vec();
    if dependencies.is_empty() {
        println!("No dependencies configured; nothing to sync.");
        return Ok(());
    }

    let manager = PackageManager::new(PackageManager::default_root());
    manager.ensure_layout().map_err(|err| err.to_string())?;
    let registry_url = resolve_registry_base(registry.as_deref());
    let client = RegistryClient::new(&registry_url).map_err(|err| err.to_string())?;

    let mut downloaded = 0usize;
    let mut updated = 0usize;
    let mut remote_deps = 0usize;

    for dep in dependencies {
        if dep.path().is_some() {
            continue;
        }
        remote_deps += 1;
        let name = dep.name().to_string();
        let details = client
            .package_details(&name)
            .map_err(|err| err.to_string())?;
        let latest_version = latest_published_version(&details)
            .ok_or_else(|| format!("package '{name}' does not have any published versions"))?;
        if dep.version() != Some(latest_version.as_str()) {
            let previous = dep
                .version()
                .map(|v| format!(" (was {v})"))
                .unwrap_or_default();
            update_dependency_in_config(&name, &latest_version)?;
            println!("Pinned '{}' to {}{}", name, latest_version, previous);
            updated += 1;
        }

        let package_dir = manager.root().join(&name).join(&latest_version);
        if !package_dir.exists() {
            let archive = client
                .download_package(&name, &latest_version)
                .map_err(|err| err.to_string())?;
            extract_package_archive(archive.path(), &package_dir)?;
            println!("Downloaded {} {}", name, latest_version);
            downloaded += 1;
        } else {
            println!("Package {} {} already installed", name, latest_version);
        }
    }

    if remote_deps == 0 {
        println!("No registry dependencies configured; skipping download step.");
    }

    let config = LustConfig::load_from_dir(&project_dir).map_err(|err| err.to_string())?;
    let resolution = resolve_dependencies(&config, &project_dir).map_err(|err| err.to_string())?;
    let prepared =
        prepare_rust_dependencies(&resolution, &project_dir).map_err(|err| err.to_string())?;

    if prepared.is_empty() {
        println!(
            "Sync complete ({} updated, {} downloaded, 0 Rust dependencies prepared).",
            updated, downloaded
        );
    } else {
        for entry in &prepared {
            if entry.stub_roots.is_empty() {
                println!("Prepared Rust dependency '{}'", entry.build.name);
            } else {
                for root in &entry.stub_roots {
                    println!(
                        "Prepared Rust dependency '{}' (stubs -> {})",
                        entry.build.name,
                        root.directory.display()
                    );
                }
            }
        }
        println!(
            "Sync complete ({} updated, {} downloaded, {} Rust dependencies prepared).",
            updated,
            downloaded,
            prepared.len()
        );
    }

    Ok(())
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn handle_pkg_remove(program: &str, args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        print_pkg_usage(program);
        return Err("pkg remove requires a package name".to_string());
    }
    let name = args[0].trim();
    if name.is_empty() {
        return Err("package name cannot be empty".to_string());
    }

    remove_dependency_from_config(name)?;
    println!("Removed dependency '{}'", name);
    Ok(())
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn parse_dependency_spec(spec: &str) -> Result<(String, Option<String>), String> {
    let mut parts = spec.splitn(2, '@');
    let name = parts
        .next()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "package name cannot be empty".to_string())?
        .to_string();
    let version = parts
        .next()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    Ok((name, version))
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn resolve_registry_base(input: Option<&str>) -> String {
    match input {
        Some(value) => {
            if value.ends_with('/') {
                value.to_string()
            } else {
                format!("{value}/")
            }
        }
        None => DEFAULT_BASE_URL.to_string(),
    }
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn extract_package_archive(archive_path: &Path, destination: &Path) -> Result<(), String> {
    remove_existing_path(destination).map_err(|err| err.to_string())?;
    fs::create_dir_all(destination).map_err(|err| err.to_string())?;
    let status = Command::new(resolve_tar_command())
        .arg("-xzf")
        .arg(archive_path)
        .arg("-C")
        .arg(destination)
        .status()
        .map_err(|err| err.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "failed to extract archive with tar (exit code {})",
            status.code().unwrap_or(-1)
        ))
    }
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn resolve_tar_command() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "tar.exe"
    }
    #[cfg(not(target_os = "windows"))]
    {
        "tar"
    }
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn remove_existing_path(path: &Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.is_dir() {
                fs::remove_dir_all(path)
            } else {
                fs::remove_file(path)
            }
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn latest_published_version(details: &PackageDetails) -> Option<String> {
    if let Some(info) = details.latest_version.as_ref() {
        let version = info.version.trim();
        if !version.is_empty() {
            return Some(version.to_string());
        }
    }
    details.versions.iter().find_map(|info| {
        let version = info.version.trim();
        if version.is_empty() {
            None
        } else {
            Some(version.to_string())
        }
    })
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn configuration_path() -> Result<std::path::PathBuf, String> {
    env::current_dir()
        .map(|dir| dir.join("lust-config.toml"))
        .map_err(|err| err.to_string())
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn read_or_create_config(path: &Path) -> Result<Value, String> {
    if path.exists() {
        let content = fs::read_to_string(path).map_err(|err| err.to_string())?;
        toml::from_str(&content).map_err(|err| err.to_string())
    } else {
        Ok(Value::Table(Map::new()))
    }
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn ensure_dependencies_table(doc: &mut Value) -> Result<&mut Map<String, Value>, String> {
    let table = doc
        .as_table_mut()
        .ok_or_else(|| "configuration root must be a table".to_string())?;
    if table.contains_key("dependencies") {
        let value = table
            .get_mut("dependencies")
            .ok_or_else(|| "[dependencies] entry missing".to_string())?;
        return value
            .as_table_mut()
            .ok_or_else(|| "[dependencies] must be a table".to_string());
    }

    let settings_has_dependencies = if let Some(value) = table.get("settings") {
        let settings_table = value
            .as_table()
            .ok_or_else(|| "[settings] must be a table".to_string())?;
        settings_table.contains_key("dependencies")
    } else {
        false
    };

    if settings_has_dependencies {
        let settings_value = table
            .get_mut("settings")
            .ok_or_else(|| "[settings] entry missing".to_string())?;
        let settings_table = settings_value
            .as_table_mut()
            .ok_or_else(|| "[settings] must be a table".to_string())?;
        let deps_value = settings_table
            .get_mut("dependencies")
            .ok_or_else(|| "[settings.dependencies] entry missing".to_string())?;
        return deps_value
            .as_table_mut()
            .ok_or_else(|| "[settings.dependencies] must be a table".to_string());
    }

    let deps_entry = table
        .entry("dependencies".to_string())
        .or_insert_with(|| Value::Table(Map::new()));
    deps_entry
        .as_table_mut()
        .ok_or_else(|| "[dependencies] must be a table".to_string())
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn update_dependency_in_config(name: &str, version: &str) -> Result<(), String> {
    let path = configuration_path()?;
    let mut doc = read_or_create_config(&path)?;
    {
        let deps = ensure_dependencies_table(&mut doc)?;
        deps.insert(name.to_string(), Value::String(version.to_string()));
    }
    let content = toml::to_string_pretty(&doc).map_err(|err| err.to_string())?;
    fs::write(&path, content).map_err(|err| err.to_string())?;
    Ok(())
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn remove_dependency_from_config(name: &str) -> Result<(), String> {
    let path = configuration_path()?;
    if !path.exists() {
        return Err("no lust-config.toml found in the current directory".to_string());
    }
    let mut doc = read_or_create_config(&path)?;
    let mut removed = false;
    {
        let table = doc
            .as_table_mut()
            .ok_or_else(|| "configuration root must be a table".to_string())?;

        if let Some(value) = table.get_mut("dependencies") {
            let deps = value
                .as_table_mut()
                .ok_or_else(|| "[dependencies] must be a table".to_string())?;
            if deps.remove(name).is_some() {
                removed = true;
                if deps.is_empty() {
                    table.remove("dependencies");
                }
            }
        }

        if !removed {
            let settings = match table.get_mut("settings") {
                Some(value) => value
                    .as_table_mut()
                    .ok_or_else(|| "[settings] must be a table".to_string())?,
                None => return Err(format!("dependency '{name}' not found")),
            };
            let deps = match settings.get_mut("dependencies") {
                Some(value) => value
                    .as_table_mut()
                    .ok_or_else(|| "[settings.dependencies] must be a table".to_string())?,
                None => return Err(format!("dependency '{name}' not found")),
            };
            if deps.remove(name).is_none() {
                return Err(format!("dependency '{name}' not found"));
            }
            if deps.is_empty() {
                settings.remove("dependencies");
            }
        }
    }
    let content = toml::to_string_pretty(&doc).map_err(|err| err.to_string())?;
    fs::write(&path, content).map_err(|err| err.to_string())?;
    Ok(())
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn handle_pkg_login(program: &str, args: &[String]) -> Result<(), String> {
    let mut token_arg: Option<String> = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--token" => {
                let value = take_option_value(args, &mut index, "--token")?;
                token_arg = Some(value);
            }
            "--help" | "-h" => {
                println!("Usage: {program} pkg login [--token <token>]");
                return Ok(());
            }
            other => {
                return Err(format!("unknown login option '{other}'"));
            }
        }
        index += 1;
    }

    let mut token = token_arg.unwrap_or_default();
    if token.is_empty() {
        print!("Enter API token: ");
        io::stdout().flush().map_err(|err| err.to_string())?;
        io::stdin()
            .read_line(&mut token)
            .map_err(|err| err.to_string())?;
    }
    let token = token.trim();
    if token.is_empty() {
        return Err("token cannot be empty".to_string());
    }

    save_credentials(token).map_err(|err| err.to_string())?;
    let path = credentials_file().map_err(|err| err.to_string())?;
    println!("Saved token to {}", path.display());
    Ok(())
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn handle_pkg_logout(program: &str, args: &[String]) -> Result<(), String> {
    if let Some(flag) = args.first() {
        if flag == "--help" || flag == "-h" {
            println!("Usage: {program} pkg logout");
            return Ok(());
        }
        return Err(format!(
            "pkg logout does not take any arguments (unexpected '{flag}')"
        ));
    }

    let had_credentials = load_credentials().map_err(|err| err.to_string())?.is_some();
    clear_credentials().map_err(|err| err.to_string())?;
    if had_credentials {
        println!("Cleared stored Lust package token.");
    } else {
        println!("No stored token was found.");
    }
    Ok(())
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn handle_pkg_publish(program: &str, args: &[String]) -> Result<(), String> {
    let mut manifest_path: Option<PathBuf> = None;
    let mut token_override: Option<String> = None;
    let mut registry: Option<String> = None;
    let mut readme_path: Option<PathBuf> = None;

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--manifest-path" => {
                let value = take_option_value(args, &mut index, "--manifest-path")?;
                manifest_path = Some(PathBuf::from(value));
            }
            "--token" => {
                let value = take_option_value(args, &mut index, "--token")?;
                token_override = Some(value);
            }
            "--registry" => {
                let value = take_option_value(args, &mut index, "--registry")?;
                registry = Some(value);
            }
            "--readme" => {
                let value = take_option_value(args, &mut index, "--readme")?;
                readme_path = Some(PathBuf::from(value));
            }
            "--help" | "-h" => {
                println!(
                    "Usage: {program} pkg publish [--manifest-path <path>] [--token <token>] [--registry <url>] [--readme <path>]"
                );
                return Ok(());
            }
            other => {
                return Err(format!("unknown publish option '{other}'"));
            }
        }
        index += 1;
    }

    let manifest = if let Some(path) = manifest_path {
        PackageManifest::discover(path.as_path()).map_err(|err| err.to_string())?
    } else {
        PackageManifest::discover(Path::new(".")).map_err(|err| err.to_string())?
    };

    let readme = if let Some(path) = readme_path {
        Some(
            fs::read_to_string(&path)
                .map_err(|err| format!("failed to read readme '{}': {err}", path.display()))?,
        )
    } else {
        None
    };

    let token = match token_override {
        Some(token) => token,
        None => load_credentials()
            .map_err(|err| err.to_string())?
            .map(|creds| creds.token().to_string())
            .ok_or_else(|| "no stored token; run 'lust pkg login' first".to_string())?,
    };

    let registry_url = resolve_registry_base(registry.as_deref());
    let client = RegistryClient::new(&registry_url).map_err(|err| err.to_string())?;

    let archive = build_package_archive(manifest.root()).map_err(|err| err.to_string())?;
    let response = client
        .publish(&manifest, &token, &archive, readme.as_deref())
        .map_err(|err| err.to_string())?;

    println!("Published {} {}", response.package, response.version);
    println!("Artifact SHA256: {}", response.artifact_sha256);
    println!("Download URL: {}", response.download_url);
    Ok(())
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn take_option_value(args: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn dump_externs(filename: &str) {
    let config = match LustConfig::load_for_entry(filename) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Error loading configuration: {}", e);
            process::exit(1);
        }
    };
    let entry_path = Path::new(filename);
    let project_dir = entry_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let resolution = match resolve_dependencies(&config, &project_dir) {
        Ok(res) => res,
        Err(err) => {
            eprintln!("Error resolving dependencies: {}", err);
            process::exit(1);
        }
    };
    if resolution.rust().is_empty() && resolution.lua().is_empty() {
        println!("No external dependencies configured; nothing to dump.");
        return;
    }

    if !resolution.lua().is_empty() {
        let names = resolution
            .lua()
            .iter()
            .map(|dep| dep.name.clone())
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "Lua dependencies detected ({}) — Lua extern dumping is not implemented yet.",
            names
        );
    }

    let output_root = project_dir.join("externs");

    let mut wrote_any = false;
    if let Err(e) = fs::create_dir_all(&output_root) {
        eprintln!(
            "Error creating extern output directory '{}': {}",
            output_root.display(),
            e
        );
        process::exit(1);
    }

    for dep in resolution.lua() {
        let mut any_stub = false;
        if !dep.luaopen_symbols.is_empty() && dep.library_path.is_file() {
            let spec = LuaModuleSpec::new(dep.library_path.clone(), dep.luaopen_symbols.clone());
            match trace_luaopen(&spec) {
                Ok(traces) => {
                    for result in traces {
                        if let Some(handle) = result.returns.iter().find_map(|v| match v {
                            LuaValue::Table(h) => Some(h.clone()),
                            _ => None,
                        }) {
                            let stub = render_table_stub(&result.module, &handle);
                            match write_lua_stub(&output_root, &result.module, &stub) {
                                Ok(path) => {
                                    println!(
                                        "Wrote Lua traced extern stub for '{}' ({}) -> {}",
                                        dep.name,
                                        result.module,
                                        output_root.join(path).display()
                                    );
                                    any_stub = true;
                                }
                                Err(err) => {
                                    eprintln!(
                                        "Failed to write traced Lua stub for '{}' ({}): {}",
                                        dep.name, result.module, err
                                    );
                                }
                            }
                        } else {
                            match write_placeholder_lua_stub(
                                &output_root,
                                &format!("luaopen_{}", result.module.replace('.', "_")),
                                &dep.name,
                            ) {
                                Ok(Some(path)) => {
                                    println!(
                                        "Lua module '{}' did not return a table; wrote placeholder extern stub -> {}",
                                        result.module,
                                        output_root.join(&path).display()
                                    );
                                    any_stub = true;
                                }
                                Ok(None) => {}
                                Err(err) => eprintln!(
                                    "Failed to write placeholder stub for '{}' ({}): {}",
                                    dep.name, result.module, err
                                ),
                            }
                        }
                    }
                }
                Err(err) => {
                    eprintln!(
                        "Failed to trace luaopen_* for '{}': {}; falling back to placeholders",
                        dep.name, err
                    );
                    for symbol in &dep.luaopen_symbols {
                        match write_placeholder_lua_stub(&output_root, symbol, &dep.name) {
                            Ok(Some(stub_path)) => {
                                println!(
                                    "Wrote Lua placeholder extern stub for '{}' -> {}",
                                    dep.name,
                                    stub_path.display()
                                );
                                any_stub = true;
                            }
                            Ok(None) => {}
                            Err(err) => {
                                eprintln!(
                                    "Failed to write Lua placeholder stub for '{}' symbol '{}': {}",
                                    dep.name, symbol, err
                                );
                            }
                        }
                    }
                }
            }
        } else {
            for symbol in &dep.luaopen_symbols {
                match write_placeholder_lua_stub(&output_root, symbol, &dep.name) {
                    Ok(Some(stub_path)) => {
                        println!(
                            "Wrote Lua placeholder extern stub for '{}' -> {}",
                            dep.name,
                            stub_path.display()
                        );
                        any_stub = true;
                    }
                    Ok(None) => {}
                    Err(err) => {
                        eprintln!(
                            "Failed to write Lua placeholder stub for '{}' symbol '{}': {}",
                            dep.name, symbol, err
                        );
                    }
                }
            }
        }
        for lua_file in &dep.lua_files {
            let full_path = dep.library_path.join(lua_file);
            #[cfg(feature = "lua_transpile")]
            {
                match fs::read_to_string(&full_path) {
                    Ok(content) => {
                        let module_name = lua_module_name(lua_file);
                        match transpile_lua_stub(&content, &module_name) {
                            Ok(stub) => {
                                let destination = output_root.join(lua_file).with_extension("lust");
                                if let Some(parent) = destination.parent() {
                                    if let Err(err) = fs::create_dir_all(parent) {
                                        eprintln!(
                                            "Failed to create stub directory '{}': {}",
                                            parent.display(),
                                            err
                                        );
                                        continue;
                                    }
                                }
                                if let Err(err) = fs::write(&destination, stub) {
                                    eprintln!(
                                        "Failed to write Lua transpiled stub '{}': {}",
                                        destination.display(),
                                        err
                                    );
                                    continue;
                                }
                                println!(
                                    "Wrote Lua transpiled stub for '{}' -> {}",
                                    dep.name,
                                    destination.display()
                                );
                                any_stub = true;
                            }
                            Err(err) => {
                                eprintln!(
                                    "Failed to transpile Lua file '{}' from '{}': {}",
                                    full_path.display(),
                                    dep.name,
                                    err
                                );
                            }
                        }
                    }
                    Err(err) => {
                        eprintln!(
                            "Failed to read Lua file '{}' for '{}': {}",
                            full_path.display(),
                            dep.name,
                            err
                        );
                    }
                }
            }
            #[cfg(not(feature = "lua_transpile"))]
            {
                eprintln!(
                    "Skipping Lua transpilation for '{}' from '{}'; recompile with --features lua_transpile to enable.",
                    full_path.display(),
                    dep.name
                );
            }
        }
        if !any_stub {
            println!(
                "Lua dependency '{}' detected but no luaopen_* symbols or Lua sources were exported; skipping stub generation",
                dep.name
            );
        }
        wrote_any |= any_stub;
    }

    for dep in resolution.rust() {
        let (build, stubs) = match collect_rust_dependency_artifacts(dep) {
            Ok(result) => result,
            Err(err) => {
                eprintln!(
                    "Failed to gather externs for '{}': {}",
                    dep.crate_dir.display(),
                    err
                );
                process::exit(1);
            }
        };
        if stubs.is_empty() {
            println!(
                "Warning: dependency '{}' did not expose any extern metadata or stub files",
                build.name
            );
            continue;
        }
        let written = match write_stub_files(&build.name, &stubs, &output_root) {
            Ok(paths) => paths,
            Err(err) => {
                eprintln!("Failed to write extern stubs for '{}': {}", build.name, err);
                process::exit(1);
            }
        };
        if let Some(cache_dir) = &dep.cache_stub_dir {
            if let Err(err) = fs::create_dir_all(cache_dir) {
                eprintln!(
                    "Failed to create extern cache directory '{}': {}",
                    cache_dir.display(),
                    err
                );
            } else if let Err(err) = write_stub_files(&build.name, &stubs, cache_dir) {
                eprintln!(
                    "Failed to write extern stubs for '{}' to cache: {}",
                    build.name, err
                );
            }
        }
        for path in &written {
            println!(
                "Wrote extern stub for '{}' -> {}",
                build.name,
                output_root.join(path).display()
            );
        }
        wrote_any = true;
    }

    if wrote_any {
        println!("Extern stubs available under {}", output_root.display());
    } else {
        println!("Completed; no extern stubs were generated.");
    }
}

#[cfg(not(all(feature = "packages", not(target_arch = "wasm32"))))]
fn dump_externs(_: &str) {
    eprintln!("This build of the Lust CLI was compiled without package support.");
    process::exit(1);
}

fn build_command(args: &[String]) {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        eprintln!(
            "Usage: lust build <file> --backend zig [--cdylib|--staticlib] [--emit-h] [--max-memory <size>] [-o <out>]"
        );
        eprintln!("       lust build [options] <entry.lust>");
        eprintln!();
        eprintln!("OPTIONS:");
        eprintln!("    --backend <name>          Backend compiler: zig (default), rust, mir-zig, mir");
        eprintln!("    -o, --output <path>       Specify output binary or library path");
        eprintln!("    --cdylib, --dylib         Build dynamic shared library (.dylib, .so, .dll)");
        eprintln!("    --staticlib               Build static library (.a, .lib)");
        eprintln!("    --emit-h [path]           Generate C header file (.h)");
        eprintln!("    --emit-zig                Generate Zig source file without compiling");
        eprintln!("    --emit-rs                 Generate Rust source file without compiling (--backend rust)");
        eprintln!("    --emit=rust-crate         Generate Cargo library crate (Cargo.toml + src/lib.rs) depending on lust-aot-rt");
        eprintln!("    --link-crate <dir|rlib>   Link external Rust crate (Zero-FFI rlib + ThinLTO for --backend rust; C-ABI for zig)");
        eprintln!("    --gas-metering <mode>     Gas metering for Rust AOT: none (default), loop-chunked, exact");
        eprintln!("    --gas-limit <units>       Gas limit for Rust AOT execution");
        eprintln!("    --release-fast            Optimize for speed (default)");
        eprintln!("    --release-safe            Optimize for speed with safety checks");
        eprintln!("    --release-small           Optimize for binary size");
        eprintln!("    --debug                   Compile in debug mode");
        eprintln!("    --strip                   Omit debug symbols");
        eprintln!("    --max-memory <size>       Limit execution memory (e.g. 64MB, 1GB)");
        eprintln!("    --tree-shake              Enable dead code elimination (default: true)");
        eprintln!("    --no-tree-shake           Disable dead code elimination");
        eprintln!("    --target <triple>         Cross-compile target triple");
        eprintln!("    --no-std                  Emit #![no_std] Rust AOT code (StaticSlabAllocator<65536> preamble)");
        eprintln!("    --cache-dir <dir>         AOT compilation cache directory");
        return;
    }

    let mut entry_file: Option<String> = None;
    let mut backend = "zig".to_string();
    let mut backend_explicit = false;
    let mut output_bin = None;
    let mut emit_zig = false;
    let mut emit_c = false;
    let mut emit_rs = false;
    let mut emit_rust_crate = false;
    let mut no_std = false;
    let mut gas_metering = lust::codegen_zig::aot::GasMetering::None;
    let mut gas_limit: Option<u64> = None;
    let mut emit_header_requested = false;
    let mut opts = lust::codegen_zig::CompileOptions::default();
    let mut link_libraries: Vec<PathBuf> = Vec::new();
    let mut link_dirs: Vec<PathBuf> = Vec::new();
    let mut allow_ffi = true;
    let mut link_crates: Vec<PathBuf> = Vec::new();
    let mut json_diag = env::var("LUST_ERROR_FORMAT")
        .map(|v| v.eq_ignore_ascii_case("json"))
        .unwrap_or(false)
        || env::var("LUST_DIAG_FORMAT")
            .map(|v| v.eq_ignore_ascii_case("json"))
            .unwrap_or(false);

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--error-format" | "--diag-format" | "--format" => {
                index += 1;
                let Some(val) = args.get(index) else {
                    eprintln!("Error: flag requires a format argument (human, json)");
                    process::exit(1);
                };
                if val.eq_ignore_ascii_case("json") {
                    json_diag = true;
                }
            }
            opt if opt.starts_with("--error-format=") => {
                let val = opt.strip_prefix("--error-format=").unwrap();
                if val.eq_ignore_ascii_case("json") {
                    json_diag = true;
                }
            }
            opt if opt.starts_with("--diag-format=") => {
                let val = opt.strip_prefix("--diag-format=").unwrap();
                if val.eq_ignore_ascii_case("json") {
                    json_diag = true;
                }
            }
            opt if opt.starts_with("--format=") => {
                let val = opt.strip_prefix("--format=").unwrap();
                if val.eq_ignore_ascii_case("json") {
                    json_diag = true;
                }
            }
            "--backend" => {
                index += 1;
                let Some(val) = args.get(index) else {
                    eprintln!("Error: --backend requires a backend name (zig, rust, mir-zig, mir)");
                    process::exit(1);
                };
                backend = val.clone();
                backend_explicit = true;
            }
            opt if opt.starts_with("--backend=") => {
                backend = opt.strip_prefix("--backend=").unwrap().to_string();
                backend_explicit = true;
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
                opts.target_kind = lust::codegen_zig::BuildTargetKind::SharedLibrary;
            }
            "--staticlib" => {
                opts.target_kind = lust::codegen_zig::BuildTargetKind::StaticLibrary;
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
            "--release-fast" => opts.release_mode = "ReleaseFast".to_string(),
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
                    eprintln!("Error: invalid memory size '{}'", val);
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
            opt if opt.starts_with("--target=") => {
                let val = opt.strip_prefix("--target=").unwrap();
                opts.target_triple = Some(val.to_string());
            }
            "--no-std" | "--nostd" => {
                no_std = true;
            }
            "--cache-dir" => {
                index += 1;
                let Some(val) = args.get(index) else {
                    eprintln!("Error: --cache-dir requires a directory path");
                    process::exit(1);
                };
                opts.cache_dir = Some(PathBuf::from(val));
            }
            "--emit-c" => {
                emit_c = true;
            }
            "--emit-rs" => {
                emit_rs = true;
            }
            "--emit-rust-crate" => {
                emit_rust_crate = true;
            }
            "--emit" => {
                index += 1;
                let Some(val) = args.get(index) else {
                    eprintln!(
                        "Error: --emit requires a kind (rust-crate, rs, zig, c)"
                    );
                    process::exit(1);
                };
                match val.to_ascii_lowercase().as_str() {
                    "rust-crate" | "rust_crate" => emit_rust_crate = true,
                    "rs" | "rust" => emit_rs = true,
                    "zig" => emit_zig = true,
                    "c" => emit_c = true,
                    other => {
                        eprintln!(
                            "Error: unknown --emit kind '{}'. Supported: rust-crate, rs, zig, c",
                            other
                        );
                        process::exit(1);
                    }
                }
            }
            opt if opt.starts_with("--emit=") => {
                let val = opt.strip_prefix("--emit=").unwrap();
                match val.to_ascii_lowercase().as_str() {
                    "rust-crate" | "rust_crate" => emit_rust_crate = true,
                    "rs" | "rust" => emit_rs = true,
                    "zig" => emit_zig = true,
                    "c" => emit_c = true,
                    other => {
                        eprintln!(
                            "Error: unknown --emit kind '{}'. Supported: rust-crate, rs, zig, c",
                            other
                        );
                        process::exit(1);
                    }
                }
            }
            "--gas-metering" => {
                index += 1;
                let Some(val) = args.get(index) else {
                    eprintln!("Error: --gas-metering requires a mode (none, loop-chunked, exact)");
                    process::exit(1);
                };
                gas_metering = match val.to_ascii_lowercase().as_str() {
                    "none" | "off" => lust::codegen_zig::aot::GasMetering::None,
                    "loop-chunked" | "loop" => lust::codegen_zig::aot::GasMetering::LoopChunked,
                    "exact" => lust::codegen_zig::aot::GasMetering::Exact,
                    other => {
                        eprintln!(
                            "Error: unknown --gas-metering mode '{}'. Supported: none, loop-chunked, exact",
                            other
                        );
                        process::exit(1);
                    }
                };
            }
            opt if opt.starts_with("--gas-metering=") => {
                let val = opt.strip_prefix("--gas-metering=").unwrap();
                gas_metering = match val.to_ascii_lowercase().as_str() {
                    "none" | "off" => lust::codegen_zig::aot::GasMetering::None,
                    "loop-chunked" | "loop" => lust::codegen_zig::aot::GasMetering::LoopChunked,
                    "exact" => lust::codegen_zig::aot::GasMetering::Exact,
                    other => {
                        eprintln!(
                            "Error: unknown --gas-metering mode '{}'. Supported: none, loop-chunked, exact",
                            other
                        );
                        process::exit(1);
                    }
                };
            }
            "--gas-limit" => {
                index += 1;
                let Some(val) = args.get(index) else {
                    eprintln!("Error: --gas-limit requires a positive integer argument");
                    process::exit(1);
                };
                match val.parse::<u64>() {
                    Ok(v) => {
                        gas_limit = Some(v);
                        if gas_metering == lust::codegen_zig::aot::GasMetering::None {
                            gas_metering = lust::codegen_zig::aot::GasMetering::Exact;
                        }
                    }
                    Err(_) => {
                        eprintln!("Error: invalid --gas-limit '{}'", val);
                        process::exit(1);
                    }
                }
            }
            opt if opt.starts_with("--gas-limit=") => {
                let val = opt.strip_prefix("--gas-limit=").unwrap();
                match val.parse::<u64>() {
                    Ok(v) => {
                        gas_limit = Some(v);
                        if gas_metering == lust::codegen_zig::aot::GasMetering::None {
                            gas_metering = lust::codegen_zig::aot::GasMetering::Exact;
                        }
                    }
                    Err(_) => {
                        eprintln!("Error: invalid --gas-limit '{}'", val);
                        process::exit(1);
                    }
                }
            }
            "--deny-ffi" | "--pure" => {
                allow_ffi = false;
            }
            "--allow-ffi" => {
                allow_ffi = true;
            }
            "--link" | "--link-lib" | "-l" => {
                index += 1;
                let Some(val) = args.get(index) else {
                    eprintln!("Error: --link requires a library path or name");
                    process::exit(1);
                };
                link_libraries.push(PathBuf::from(val));
            }
            opt if opt.starts_with("--link=") => {
                let val = opt.strip_prefix("--link=").unwrap();
                link_libraries.push(PathBuf::from(val));
            }
            opt if opt.starts_with("--link-lib=") => {
                let val = opt.strip_prefix("--link-lib=").unwrap();
                link_libraries.push(PathBuf::from(val));
            }
            "--link-crate" => {
                index += 1;
                let Some(val) = args.get(index) else {
                    eprintln!("Error: --link-crate requires a crate directory");
                    process::exit(1);
                };
                link_crates.push(PathBuf::from(val));
            }
            opt if opt.starts_with("--link-crate=") => {
                let val = opt.strip_prefix("--link-crate=").unwrap();
                link_crates.push(PathBuf::from(val));
            }
            "--link-dir" | "-L" => {
                index += 1;
                let Some(val) = args.get(index) else {
                    eprintln!("Error: --link-dir requires a directory path");
                    process::exit(1);
                };
                link_dirs.push(PathBuf::from(val));
            }
            opt if opt.starts_with("--link-dir=") => {
                let val = opt.strip_prefix("--link-dir=").unwrap();
                link_dirs.push(PathBuf::from(val));
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

    let Some(entry_arg) = entry_file else {
        eprintln!("Error: lust build requires an entry file");
        process::exit(1);
    };

    if emit_rust_crate && !backend_explicit {
        backend = "rust".to_string();
    }

    match backend.to_ascii_lowercase().as_str() {
        "zig" | "rust" | "mir-zig" | "mir" | "cranelift-aot" => {}
        other => {
            eprintln!(
                "Error: unknown backend '{}'. Supported backends for build: zig, rust, mir-zig, mir, cranelift-aot",
                other
            );
            process::exit(1);
        }
    }

    let is_rust_backend = backend.eq_ignore_ascii_case("rust");
    let is_cranelift_aot_backend = backend.eq_ignore_ascii_case("cranelift-aot");

    // Preserve legacy Zig/WASI-P2 offline-blocked rejection while lifting the hardcoded ban
    // for `--backend rust` (which performs real `rustc --print target-libdir` detection and
    // emits full `#![no_std]` `.rs` with `TargetStdlibNotInstalledOffline`, `NG-W13-2` / `F-11`).
    if !is_rust_backend && !is_cranelift_aot_backend && opts.target_triple.as_deref() == Some("wasm32-wasip2") {
        eprintln!("offline-blocked: target 'wasm32-wasip2' is not installed offline");
        process::exit(1);
    }

    let entry_path = PathBuf::from(&entry_arg);
    let project_dir = entry_path.parent().unwrap_or_else(|| Path::new("."));
    let mut config = lust::config::LustConfig::load_for_entry(&entry_path).unwrap_or_default();
    if !allow_ffi {
        config.capability_mask_mut().allow_ffi = false;
    } else {
        allow_ffi = config.capability_mask().allow_ffi;
    }

    #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
    {
        if !config.dependencies().is_empty() {
            if let Ok(dep_res) = resolve_dependencies(&config, project_dir) {
                if let Ok(prepared) = prepare_rust_dependencies(&dep_res, project_dir) {
                    for dep in prepared {
                        link_libraries.push(dep.build.library_path);
                    }
                }
            }
        }
    }

    if !allow_ffi && (!link_libraries.is_empty() || !link_dirs.is_empty() || !link_crates.is_empty()) {
        eprintln!("Error: CapabilityDenied: external library linking requires 'ffi' capability");
        process::exit(1);
    }

    let mut linked_rust_crates: Vec<lust::codegen_zig::aot::compiler::RustLinkedCrate> = Vec::new();
    let mut rust_crate_deps: Vec<lust::codegen_zig::aot::compiler::RustCrateDependency> = Vec::new();

    #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
    {
        if is_rust_backend {
            for crate_path in &link_crates {
                if emit_rust_crate && crate_path.is_dir() {
                    if !crate_path.join("Cargo.toml").exists() {
                        eprintln!(
                            "Error: missing external library at '{}'",
                            crate_path.display()
                        );
                        process::exit(1);
                    }
                    let package_name = match lust::packages::read_crate_name(crate_path) {
                        Ok(name) => name,
                        Err(err) => {
                            eprintln!("Error: {}", err);
                            process::exit(1);
                        }
                    };
                    let crate_name = match lust::packages::read_crate_lib_name(crate_path) {
                        Ok(name) => name,
                        Err(err) => {
                            eprintln!("Error: {}", err);
                            process::exit(1);
                        }
                    };
                    rust_crate_deps.push(lust::codegen_zig::aot::compiler::RustCrateDependency {
                        package_name,
                        crate_name,
                        path: crate_path.clone(),
                    });
                } else {
                    match lust::packages::prepare_external_rlib(
                        crate_path,
                        allow_ffi,
                        opts.target_triple.as_deref(),
                        opts.release_mode.starts_with("Release"),
                    ) {
                        Ok(rlib) => {
                            linked_rust_crates.push(
                                lust::codegen_zig::aot::compiler::RustLinkedCrate {
                                    crate_name: rlib.crate_name,
                                    rlib_path: rlib.rlib_path,
                                    dependency_dirs: rlib.dependency_dirs,
                                },
                            );
                        }
                        Err(err) => {
                            eprintln!("Error: {}", err);
                            process::exit(1);
                        }
                    }
                }
            }
        } else {
            for crate_dir in &link_crates {
                match lust::packages::build_and_prepare_external_crate(
                    crate_dir,
                    allow_ffi,
                    opts.target_triple.as_deref(),
                    opts.release_mode.starts_with("Release"),
                ) {
                    Ok(out) => {
                        link_libraries.push(out.library_path);
                    }
                    Err(err) => {
                        eprintln!("Error: {}", err);
                        process::exit(1);
                    }
                }
            }
        }

        for lib_path in &link_libraries {
            if let Err(err) = lust::packages::verify_external_library(lib_path, allow_ffi) {
                eprintln!("Error: {}", err);
                process::exit(1);
            }
        }
    }
    if emit_header_requested && opts.emit_header.is_none() {
        opts.emit_header = Some(entry_path.with_extension("h"));
    }

    let mut loader = ModuleLoader::new(entry_path.parent().unwrap_or_else(|| Path::new(".")));
    let program = match loader.load_program_from_entry(&entry_arg) {
        Ok(p) => p,
        Err(e) => {
            if json_diag {
                eprint!("{}", cli::diag::json(format!("Error loading program: {e}"), "error", None, None));
            } else {
                eprintln!("Error loading program: {}", e);
            }
            process::exit(1);
        }
    };

    let mut all_items = Vec::new();
    for module in &program.modules {
        all_items.extend(module.items.clone());
    }

    let is_library = emit_rust_crate
        || matches!(
            opts.target_kind,
            lust::codegen_zig::BuildTargetKind::SharedLibrary
                | lust::codegen_zig::BuildTargetKind::StaticLibrary
        );

    let filtered_items = if opts.tree_shake {
        lust::codegen_zig::tree_shake_items(&all_items, is_library)
    } else {
        all_items.clone()
    };

    let stem = entry_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    let is_wasm = opts
        .target_triple
        .as_ref()
        .map(|t| t.contains("wasm32"))
        .unwrap_or(false);

    let default_extension = if is_wasm {
        "wasm"
    } else {
        match opts.target_kind {
            lust::codegen_zig::BuildTargetKind::Executable => "",
            lust::codegen_zig::BuildTargetKind::SharedLibrary => {
                if cfg!(target_os = "macos") {
                    "dylib"
                } else if cfg!(target_os = "windows") {
                    "dll"
                } else {
                    "so"
                }
            }
            lust::codegen_zig::BuildTargetKind::StaticLibrary => {
                if cfg!(target_os = "windows") {
                    "lib"
                } else {
                    "a"
                }
            }
        }
    };

    let target_bin = output_bin.clone().unwrap_or_else(|| {
        let parent = entry_path.parent().unwrap_or_else(|| Path::new("."));
        if default_extension.is_empty() {
            parent.join(stem)
        } else {
            parent.join(format!("{}.{}", stem, default_extension))
        }
    });

    if is_cranelift_aot_backend {
        let mut imports_map = lust::hashbrown::HashMap::new();
        for m in &program.modules {
            imports_map.insert(m.path.clone(), m.imports.clone());
        }
        let mut typechecker = TypeChecker::with_config(&config);
        typechecker.set_imports_by_module(imports_map);
        if let Err(e) = typechecker.check_program(&program.modules) {
            if json_diag {
                eprint!(
                    "{}",
                    cli::diag::json(format!("Type error: {e}"), "error", None, None)
                );
            } else {
                eprintln!("Type error: {}", e);
            }
            process::exit(1);
        }

        let prepared_items = lust::codegen_zig::aot::prepare_items_for_mir(&filtered_items);
        let mir_module =
            match lust::mir::lowering::AstLowering::lower_module_items("main_module", &prepared_items) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("MIR lowering error: {}", e);
                    process::exit(1);
                }
            };

        match lust::codegen_zig::aot::compile_mir_to_object_file(
            &mir_module,
            &target_bin,
            opts.target_triple.as_deref(),
        ) {
            Ok(obj) => {
                println!("Successfully built Cranelift AOT object: {}", obj.display());
                return;
            }
            Err(e) => {
                eprintln!("{}", e);
                process::exit(1);
            }
        }
    }

    if is_rust_backend {
        // Mandatory NG-W12-4A + NG-W13-1C + NG-W13-2 frontend pipeline:
        // TypeChecker::check_program -> prepare_items_for_mir -> desugar_effects_for_rust_aot
        // -> AstLowering -> MirOptimizer + SROA -> refine_mir_function_with_layouts
        // -> EscapeAnalysis -> cfg_structurer -> Rust emitter
        let mut imports_map = lust::hashbrown::HashMap::new();
        for m in &program.modules {
            imports_map.insert(m.path.clone(), m.imports.clone());
        }
        let mut typechecker = TypeChecker::with_config(&config);
        typechecker.set_imports_by_module(imports_map);
        if let Err(e) = typechecker.check_program(&program.modules) {
            if json_diag {
                eprint!(
                    "{}",
                    cli::diag::json(format!("Type error: {e}"), "error", None, None)
                );
            } else {
                eprintln!("Type error: {}", e);
            }
            process::exit(1);
        }

        let mir_module =
            match lust::codegen_zig::aot::compiler::prepare_and_lower_items_for_rust_aot(
                &filtered_items,
            ) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("MIR lowering error: {}", e);
                    process::exit(1);
                }
            };

        let effective_no_std = no_std
            || opts
                .target_triple
                .as_deref()
                .is_some_and(lust::codegen_zig::aot::is_nostd_target);

        let rust_opts = lust::codegen_zig::aot::RustCompileOptions {
            aot: lust::codegen_zig::aot::AotCompileOptions {
                target_kind: match opts.target_kind {
                    lust::codegen_zig::BuildTargetKind::Executable => {
                        lust::codegen_zig::aot::AotTargetKind::Executable
                    }
                    lust::codegen_zig::BuildTargetKind::SharedLibrary => {
                        lust::codegen_zig::aot::AotTargetKind::SharedLibrary
                    }
                    lust::codegen_zig::BuildTargetKind::StaticLibrary => {
                        lust::codegen_zig::aot::AotTargetKind::StaticLibrary
                    }
                },
                release_mode: opts.release_mode.clone(),
                strip: opts.strip,
                max_memory_bytes: opts.max_memory_bytes,
                target_triple: opts.target_triple.clone(),
            },
            gas_metering,
            gas_limit,
            pure_core: !allow_ffi,
            no_std: effective_no_std,
            cache_dir: opts.cache_dir.clone(),
        };

        if emit_rust_crate {
            let crate_out_dir = output_bin.unwrap_or_else(|| {
                let parent = entry_path.parent().unwrap_or_else(|| Path::new("."));
                parent.join(format!("{stem}_crate"))
            });
            let pkg_name = crate_out_dir
                .file_name()
                .and_then(|s| s.to_str())
                .filter(|s| !s.is_empty())
                .unwrap_or(stem);
            match lust::codegen_zig::aot::AotCompiler::emit_rust_crate(
                &mir_module,
                &all_items,
                &crate_out_dir,
                pkg_name,
                &rust_opts,
                &rust_crate_deps,
            ) {
                Ok(emitted) => {
                    println!(
                        "Generated Rust Cargo crate '{}' at {} ({} + {})",
                        emitted.package_name,
                        emitted.crate_dir.display(),
                        emitted.manifest_path.display(),
                        emitted.lib_rs_path.display()
                    );
                    return;
                }
                Err(e) => {
                    eprintln!("Rust Cargo crate emission error: {}", e);
                    process::exit(1);
                }
            }
        }

        if emit_rs {
            let interop = lust::codegen_zig::aot::rust_emitter::RustInteropOptions {
                linked_crates: linked_rust_crates
                    .iter()
                    .map(|c| c.crate_name.clone())
                    .collect(),
                emit_as_include_module: false,
                export_public_api: is_library,
            };
            let rust_source = match lust::codegen_zig::aot::AotCompiler::emit_rust_module_with_interop(
                &mir_module,
                &all_items,
                &rust_opts,
                &interop,
            ) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Rust AOT codegen error: {}", e);
                    process::exit(1);
                }
            };
            let rs_file = if target_bin.extension().and_then(|e| e.to_str()) == Some("rs") {
                target_bin
            } else {
                entry_path.with_extension("rs")
            };
            if let Some(parent) = rs_file.parent() {
                let _ = fs::create_dir_all(parent);
            }
            if let Err(e) = fs::write(&rs_file, &rust_source) {
                eprintln!("Failed to write {}: {}", rs_file.display(), e);
                process::exit(1);
            }
            println!("Generated Rust source: {}", rs_file.display());
            return;
        }

        let aot_links = lust::codegen_zig::aot::AotLinkOptions {
            link_libraries: link_libraries.clone(),
            link_dirs: link_dirs.clone(),
            allow_ffi,
        };
        let lto_thin = !linked_rust_crates.is_empty();
        let rust_links = lust::codegen_zig::aot::compiler::RustLinkOptions {
            aot_links,
            linked_crates: linked_rust_crates,
            lto_thin,
        };

        match lust::codegen_zig::aot::AotCompiler::compile_module_rust_cross(
            &mir_module,
            &all_items,
            &target_bin,
            &rust_opts,
            &rust_links,
        ) {
            Ok(lust::codegen_zig::aot::RustAotBuildOutcome::Compiled { artifact_path, .. }) => {
                println!("Successfully built AOT artifact: {}", artifact_path.display());
                return;
            }
            Ok(lust::codegen_zig::aot::RustAotBuildOutcome::TargetStdlibNotInstalledOffline {
                target,
                generated_rs_path,
            }) => {
                eprintln!(
                    "TargetStdlibNotInstalledOffline {{ target: \"{target}\", generated_rs_path: \"{}\" }}",
                    generated_rs_path.display()
                );
                process::exit(2);
            }
            Err(e) => {
                eprintln!("{}", e);
                process::exit(1);
            }
        }
    }

    if backend.to_ascii_lowercase() == "mir" {
        let prepared_items = lust::codegen_zig::aot::prepare_items_for_mir(&filtered_items);
        let mir_module = match lust::mir::lowering::AstLowering::lower_module_items("main_module", &prepared_items) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("MIR lowering error: {}", e);
                process::exit(1);
            }
        };

        if emit_c {
            let emitter_opts = lust::codegen_zig::aot::c_emitter::CEmitterOptions {
                is_library,
                max_memory_bytes: opts.max_memory_bytes,
            };
            let c_source = lust::codegen_zig::aot::c_emitter::emit_c_from_mir(&mir_module, &emitter_opts);
            let c_file = entry_path.with_extension("c");
            if let Err(e) = fs::write(&c_file, &c_source) {
                eprintln!("Failed to write {}: {}", c_file.display(), e);
                process::exit(1);
            }
            println!("Generated C source: {}", c_file.display());
            return;
        }

        let aot_opts = lust::codegen_zig::aot::AotCompileOptions {
            target_kind: match opts.target_kind {
                lust::codegen_zig::BuildTargetKind::Executable => lust::codegen_zig::aot::AotTargetKind::Executable,
                lust::codegen_zig::BuildTargetKind::SharedLibrary => lust::codegen_zig::aot::AotTargetKind::SharedLibrary,
                lust::codegen_zig::BuildTargetKind::StaticLibrary => lust::codegen_zig::aot::AotTargetKind::StaticLibrary,
            },
            release_mode: opts.release_mode.clone(),
            strip: opts.strip,
            max_memory_bytes: opts.max_memory_bytes,
            target_triple: opts.target_triple.clone(),
        };

        let aot_links = lust::codegen_zig::aot::AotLinkOptions {
            link_libraries: link_libraries.clone(),
            link_dirs: link_dirs.clone(),
            allow_ffi,
        };

        match lust::codegen_zig::aot::AotCompiler::compile_module_with_links(
            &mir_module,
            &target_bin,
            &aot_opts,
            &aot_links,
        ) {
            Ok(bin) => {
                println!("Successfully built AOT artifact: {}", bin.display());
                return;
            }
            Err(e) => {
                eprintln!("{}", e);
                process::exit(1);
            }
        }
    }

    let runtime_opts = lust::codegen_zig::analyze_runtime_requirements(
        &filtered_items,
        is_library,
        opts.max_memory_bytes,
    );
    let link_libc = runtime_opts.needs_io || runtime_opts.needs_os;

    let mut codegen = lust::codegen_zig::ZigCodegen::with_options(runtime_opts);
    let zig_source = match codegen.generate_program(&filtered_items) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("AOT Codegen error: {}", e);
            process::exit(1);
        }
    };

    let zig_file = entry_path.with_extension("zig");
    if let Err(e) = fs::write(&zig_file, &zig_source) {
        eprintln!("Failed to write {}: {}", zig_file.display(), e);
        process::exit(1);
    }

    if let Some(header_path) = &opts.emit_header {
        let header_content = lust::codegen_zig::generate_c_header(&filtered_items);
        if let Err(e) = fs::write(header_path, header_content) {
            eprintln!("Failed to write C header {}: {}", header_path.display(), e);
            process::exit(1);
        }
        println!("Generated C header: {}", header_path.display());
    }

    if emit_zig {
        println!("Generated Zig source: {}", zig_file.display());
        return;
    }

    let zig_links = lust::codegen_zig::ZigLinkOptions {
        link_libraries,
        link_dirs,
        allow_ffi,
    };

    match lust::codegen_zig::compile_zig_with_links(
        &zig_file,
        &target_bin,
        &opts,
        link_libc,
        &zig_links,
    ) {
        Ok(bin) => {
            println!("Successfully built AOT artifact: {}", bin.display());
        }
        Err(e) => {
            eprintln!("{}", e);
            process::exit(1);
        }
    }
}

fn bindgen_command(args: &[String]) {
    #[cfg(feature = "bindgen")]
    {
        if args
            .first()
            .is_some_and(|arg| arg == "--help" || arg == "-h")
        {
            eprintln!("Usage: lust bindgen <entry.lust> [--out <file.rs>] [--name Bindings]");
            return;
        }
        let Some(entry_arg) = args.first() else {
            eprintln!("Error: lust bindgen requires an entry file");
            process::exit(1);
        };

        let entry = PathBuf::from(entry_arg);
        let mut output = None;
        let mut bindings_name = "Bindings".to_string();
        let mut index = 1;
        while index < args.len() {
            match args[index].as_str() {
                "--out" | "--output" | "-o" => {
                    index += 1;
                    let Some(value) = args.get(index) else {
                        eprintln!("Error: {} requires a path", args[index - 1]);
                        process::exit(1);
                    };
                    output = Some(PathBuf::from(value));
                }
                "--name" => {
                    index += 1;
                    let Some(value) = args.get(index) else {
                        eprintln!("Error: --name requires a Rust type name");
                        process::exit(1);
                    };
                    bindings_name = value.clone();
                }
                option => {
                    eprintln!("Error: unknown bindgen option '{option}'");
                    eprintln!(
                        "Usage: lust bindgen <entry.lust> [--out <file.rs>] [--name Bindings]"
                    );
                    process::exit(1);
                }
            }
            index += 1;
        }

        let generated = match lust::bindgen::RustBindingsBuilder::new(&entry)
            .bindings_name(bindings_name)
            .generate()
        {
            Ok(generated) => generated,
            Err(err) => {
                eprintln!("Bindgen failed: {err}");
                process::exit(1);
            }
        };
        let output = output.unwrap_or_else(|| {
            let stem = entry
                .file_stem()
                .and_then(|name| name.to_str())
                .unwrap_or("lust");
            let parent = entry
                .parent()
                .filter(|path| !path.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."));
            parent.join(format!("{stem}_bindings.rs"))
        });
        if let Err(err) = generated.write_to(&output) {
            eprintln!("Failed to write '{}': {err}", output.display());
            process::exit(1);
        }
        println!(
            "Generated {} Lust binding(s) -> {}",
            generated
                .source
                .matches("    pub fn ")
                .count()
                .saturating_sub(1),
            output.display()
        );
    }

    #[cfg(not(feature = "bindgen"))]
    {
        let _ = args;
        eprintln!(
            "This build of the Lust CLI does not include bindgen; rebuild with --features bindgen."
        );
        process::exit(1);
    }
}


fn handle_dst_failure_capture(
    cap_path: &str,
    filename: &str,
    source: &str,
    failure: lust::sim::replay::FailureKind,
) {
    let cap_config = lust::sim::replay::CaptureConfig::new(cap_path)
        .with_script(filename, source)
        .with_args(env::args().collect())
        .with_env_vars(env::vars().collect());
    let capture_res = lust::sim::replay::capture_failure_to_bundle(
        &cap_config,
        &failure,
        lust::sim::replay::EventLog::new(),
        lust::sim::replay::global_shadow_vfs().snapshot(),
    );
    match capture_res {
        Ok(bundle) => {
            if let Err(write_err) = lust::sim::replay::write_bundle_and_reproduction_task(
                &bundle,
                Path::new(cap_path),
            ) {
                eprintln!("[DST-CAP] Warning: Failed to write replay bundle: {}", write_err);
            } else {
                eprintln!("[DST-CAP] Failure captured to replay bundle: {}", cap_path);
                eprintln!("[DST-CAP] Reproduce with: lust sim replay {}", cap_path);
            }
        }
        Err(err) => {
            eprintln!("[DST-CAP] Warning: Failed to capture replay bundle: {}", err);
        }
    }
}

fn run_file_with_options(
    filename: &str,
    disassemble: bool,
    use_mir: bool,
    strict_mir: bool,
    cranelift_jit: bool,
    jit_disabled: bool,
    no_gc: bool,
    pure_core: bool,
    json_diag: bool,
    capture_failure: Option<&str>,
) {
    let source = match fs::read_to_string(filename) {
        Ok(content) => content,
        Err(e) => {
            if json_diag {
                eprint!("{}", cli::diag::json(format!("Error reading file '{filename}': {e}"), "error", None, None));
            } else {
                eprintln!("Error reading file '{}': {}", filename, e);
            }
            process::exit(1);
        }
    };

    if let Some(cap_path) = capture_failure {
        let cap_config = lust::sim::replay::CaptureConfig::new(cap_path)
            .with_script(filename, &source)
            .with_args(env::args().collect())
            .with_env_vars(env::vars().collect());
        lust::sim::replay::install_failure_panic_hook(cap_config);
        lust::sim::replay::global_shadow_vfs().record_write(filename, source.as_bytes().to_vec());
    }

    if env::var("LUST_TRIGGER_PANIC_FOR_TEST").is_ok() || env::args().any(|a| a == "--panic-for-test") {
        panic!("FORCED_PANIC_FOR_TEST: Artificial panic triggered to verify DST failure capture");
    }
    let entry_path = Path::new(filename);
    let project_dir = entry_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let mut config = match LustConfig::load_for_entry(filename) {
        Ok(cfg) => cfg,
        Err(e) => {
            if json_diag {
                eprint!("{}", cli::diag::json(format!("Error loading configuration: {e}"), "error", None, None));
            } else {
                eprintln!("Error loading configuration: {}", e);
            }
            process::exit(1);
        }
    };
    if pure_core {
        config.set_prelude_mode(lust::config::PreludeMode::Pure);
    }
    let config_had_use_mir = config.use_mir();
    if use_mir {
        config.set_use_mir(true);
    }
    if strict_mir {
        config.set_strict_mir(true);
    }
    if cranelift_jit {
        config.set_cranelift_jit(true);
        if !config_had_use_mir && !use_mir && !strict_mir {
            config.set_use_mir(false);
        }
    }
    if jit_disabled {
        config.set_jit_enabled(false);
    }
    if no_gc {
        config.set_gc_enabled(false);
    }
    #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
    let dependency_resolution = match resolve_dependencies(&config, &project_dir) {
        Ok(res) => res,
        Err(err) => {
            eprintln!("Error resolving dependencies: {}", err);
            process::exit(1);
        }
    };
    #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
    let prepared_rust = match prepare_rust_dependencies(&dependency_resolution, &project_dir) {
        Ok(list) => list,
        Err(err) => {
            eprintln!("Failed to prepare Rust dependencies: {}", err);
            process::exit(1);
        }
    };
    let (functions, mir_functions, trait_impls, init_funcs, struct_defs) = match compile_program(
        filename,
        &config,
        #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
        Some(&dependency_resolution),
        #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
        Some(&prepared_rust),
    ) {
        Ok(result) => result,
        Err(e) => {
            if let Some(cap_path) = capture_failure {
                handle_dst_failure_capture(
                    cap_path,
                    filename,
                    &source,
                    lust::sim::replay::FailureKind::CompilationError(e.to_string()),
                );
            }
            print_error_with_context(&source, filename, &e, json_diag);
            process::exit(1);
        }
    };
    if disassemble {
        println!("Bytecode Disassembly for '{}':", filename);
        println!("{:=<70}", "");
        for func in &functions {
            println!("{}", func.disassemble());
            println!("{:-<70}", "");
        }

        return;
    }

    let mut vm = VM::with_config(&config);
    vm.load_functions(functions);
    for mir_fn in mir_functions {
        if config.use_mir() {
            if let Err(e) = vm.load_mir_function(mir_fn) {
                if let Some(cap_path) = capture_failure {
                    handle_dst_failure_capture(
                        cap_path,
                        filename,
                        &source,
                        lust::sim::replay::FailureKind::RuntimeError(e.to_string()),
                    );
                }
                print_error_with_context(&source, filename, &e, json_diag);
                process::exit(1);
            }
        } else {
            vm.register_mir_candidate(mir_fn);
        }
    }
    vm.register_structs(&struct_defs);
    for (type_name, trait_name) in trait_impls {
        vm.register_trait_impl(type_name, trait_name);
    }

    #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
    let _loaded_extensions: Vec<LoadedRustModule> =
        match load_prepared_rust_dependencies(&prepared_rust, &mut vm) {
            Ok(mods) => mods,
            Err(err) => {
                eprintln!("Failed to load Rust dependencies: {}", err);
                process::exit(1);
            }
        };
    #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
    {
        for dep in dependency_resolution.lua() {
            if dep.luaopen_symbols.is_empty() {
                continue;
            }
            let spec = LuaModuleSpec::new(dep.library_path.clone(), dep.luaopen_symbols.clone());
            match trace_luaopen(&spec) {
                Ok(results) => {
                    for result in results {
                        for value in &result.returns {
                            if let LuaValue::Table(handle) = value {
                                if let Ok(table_val) = lua_to_lust(value, &vm, result.state.clone())
                                {
                                    vm.set_global(&result.module, table_val);
                                }
                                for (key, val) in handle.borrow().entries.iter() {
                                    if let LuaValue::String(name) = key
                                        && let Ok(converted) =
                                            lua_to_lust(val, &vm, result.state.clone())
                                    {
                                        vm.set_global(
                                            format!("{}.{}", result.module, name),
                                            converted,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                Err(err) => {
                    eprintln!(
                        "Warning: failed to load Lua dependency '{}': {}",
                        dep.name, err
                    );
                }
            }
        }
    }
    //eprintln!("init order: {:?}", init_funcs);
    for (module_path, init) in init_funcs {
        match vm.call(&init, vec![]) {
            Ok(val) => {
                let _display_val = format!("{}", val);
                //eprintln!("init {module_path} returned {display_val}");
                // Lua's `require()` only observes a single return value from a module loader.
                // Our Lua->Lust transpiler represents returns as `Array<LuaValue>` to preserve
                // multi-return semantics, but module values should behave like Lua modules:
                // export a single value (usually a table).
                let module_value = match val {
                    lust::bytecode::Value::Array(arr) => arr
                        .borrow()
                        .first()
                        .cloned()
                        .unwrap_or(lust::bytecode::Value::Nil),
                    other => other,
                };
                vm.set_global(module_path, module_value);
            }
            Err(e) => {
                if let Some(cap_path) = capture_failure {
                    handle_dst_failure_capture(
                        cap_path,
                        filename,
                        &source,
                        lust::sim::replay::FailureKind::RuntimeError(e.to_string()),
                    );
                }
                print_error_with_context(&source, filename, &e, json_diag);
                process::exit(1);
            }
        }
    }

    if let Err(e) = vm.call("__script", vec![]) {
        if let Some(cap_path) = capture_failure {
            handle_dst_failure_capture(
                cap_path,
                filename,
                &source,
                lust::sim::replay::FailureKind::RuntimeError(e.to_string()),
            );
        }
        print_error_with_context(&source, filename, &e, json_diag);
        process::exit(1);
    }

    if std::env::var_os("LUST_JIT_STATS").is_some() {
        eprintln!("JIT stats: {:?}", vm.jit_stats());
        eprintln!(
            "Cranelift stats: entries={} compilations={}",
            vm.cranelift_entry_count, vm.cranelift_compilation_count
        );
    }
}

fn run_file_rust_aot(filename: &str, pure_core: bool) {
    let entry_path = Path::new(filename);
    let parent = entry_path.parent().unwrap_or_else(|| Path::new("."));
    let mut loader = ModuleLoader::new(parent);
    let program = match loader.load_program_from_entry(filename) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error loading program: {}", e);
            process::exit(1);
        }
    };

    let mut config = lust::config::LustConfig::load_for_entry(entry_path).unwrap_or_default();
    if pure_core {
        config.capability_mask_mut().allow_ffi = false;
    }
    let mut imports_map = lust::hashbrown::HashMap::new();
    for m in &program.modules {
        imports_map.insert(m.path.clone(), m.imports.clone());
    }
    let mut typechecker = TypeChecker::with_config(&config);
    typechecker.set_imports_by_module(imports_map);
    if let Err(e) = typechecker.check_program(&program.modules) {
        eprintln!("Type error: {}", e);
        process::exit(1);
    }

    let mut all_items = Vec::new();
    for module in &program.modules {
        all_items.extend(module.items.clone());
    }
    let filtered_items = lust::codegen_zig::tree_shake_items(&all_items, false);
    let prepared_items = lust::codegen_zig::aot::prepare_items_for_mir(&filtered_items);
    let mir_module =
        match lust::mir::lowering::AstLowering::lower_module_items("main_module", &prepared_items) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("MIR lowering error: {}", e);
                process::exit(1);
            }
        };

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let pid = process::id();
    let temp_dir = env::temp_dir().join(format!("lust_run_rust_{}_{}", pid, nanos));
    if let Err(e) = fs::create_dir_all(&temp_dir) {
        eprintln!("Failed to create temporary directory: {}", e);
        process::exit(1);
    }

    let stem = entry_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("script");
    let target_bin = temp_dir.join(if cfg!(target_os = "windows") {
        format!("{}.exe", stem)
    } else {
        stem.to_string()
    });

    let rust_opts = lust::codegen_zig::aot::RustCompileOptions {
        aot: lust::codegen_zig::aot::AotCompileOptions::default(),
        gas_metering: lust::codegen_zig::aot::GasMetering::None,
        gas_limit: None,
        pure_core,
        no_std: false,
        cache_dir: None,
    };

    let compiled_bin =
        match lust::codegen_zig::aot::AotCompiler::compile_module_rust(&mir_module, &target_bin, &rust_opts) {
            Ok(bin) => bin,
            Err(e) => {
                eprintln!("{}", e);
                let _ = fs::remove_dir_all(&temp_dir);
                process::exit(1);
            }
        };

    let status = Command::new(&compiled_bin).status();
    let _ = fs::remove_dir_all(&temp_dir);

    match status {
        Ok(s) => {
            if let Some(code) = s.code() {
                process::exit(code);
            } else {
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt;
                    if let Some(sig) = s.signal() {
                        process::exit(128 + sig);
                    }
                }
                process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("Failed to execute compiled binary: {}", e);
            process::exit(1);
        }
    }
}

fn run_file_zig_aot(filename: &str) {
    let entry_path = Path::new(filename);
    let parent = entry_path.parent().unwrap_or_else(|| Path::new("."));
    let mut loader = ModuleLoader::new(parent);
    let program = match loader.load_program_from_entry(filename) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error loading program: {}", e);
            process::exit(1);
        }
    };

    let mut all_items = Vec::new();
    for module in &program.modules {
        all_items.extend(module.items.clone());
    }

    let is_library = false;
    let filtered_items = lust::codegen_zig::tree_shake_items(&all_items, is_library);

    let runtime_opts = lust::codegen_zig::analyze_runtime_requirements(
        &filtered_items,
        is_library,
        None,
    );
    let link_libc = runtime_opts.needs_io || runtime_opts.needs_os;

    let mut codegen = lust::codegen_zig::ZigCodegen::with_options(runtime_opts);
    let zig_source = match codegen.generate_program(&filtered_items) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("AOT Codegen error: {}", e);
            process::exit(1);
        }
    };

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let pid = process::id();
    let temp_dir = env::temp_dir().join(format!("lust_run_zig_{}_{}", pid, nanos));
    if let Err(e) = fs::create_dir_all(&temp_dir) {
        eprintln!("Failed to create temporary directory: {}", e);
        process::exit(1);
    }

    let stem = entry_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("script");
    let zig_file = temp_dir.join(format!("{}.zig", stem));
    if let Err(e) = fs::write(&zig_file, &zig_source) {
        eprintln!("Failed to write {}: {}", zig_file.display(), e);
        let _ = fs::remove_dir_all(&temp_dir);
        process::exit(1);
    }

    let target_bin = temp_dir.join(if cfg!(target_os = "windows") {
        format!("{}.exe", stem)
    } else {
        stem.to_string()
    });

    let opts = lust::codegen_zig::CompileOptions {
        target_kind: lust::codegen_zig::BuildTargetKind::Executable,
        release_mode: "ReleaseFast".to_string(),
        emit_zig: false,
        emit_header: None,
        max_memory: None,
        max_memory_bytes: None,
        tree_shake: true,
        strip: false,
        target_triple: None,
        cache_dir: None,
    };

    let compiled_bin = match lust::codegen_zig::compile_zig(&zig_file, &target_bin, &opts, link_libc) {
        Ok(bin) => bin,
        Err(e) => {
            eprintln!("{}", e);
            let _ = fs::remove_dir_all(&temp_dir);
            process::exit(1);
        }
    };

    let status = Command::new(&compiled_bin).status();
    let _ = fs::remove_dir_all(&temp_dir);

    match status {
        Ok(s) => {
            if let Some(code) = s.code() {
                process::exit(code);
            } else {
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt;
                    if let Some(sig) = s.signal() {
                        process::exit(128 + sig);
                    }
                }
                process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("Failed to execute compiled binary: {}", e);
            process::exit(1);
        }
    }
}

/// What `compile_program` produces: the compiled functions, the trait
/// impls and the module init functions the VM registers, and the struct
/// definitions.
type CompiledProgram = (
    Vec<lust::bytecode::Function>,
    Vec<lust::mir::ir::MirFunction>,
    Vec<(String, String)>,
    Vec<(String, String)>,
    lust::hashbrown::HashMap<String, lust::ast::StructDef>,
);

fn compile_program(
    entry_filename: &str,
    config: &LustConfig,
    #[cfg(all(feature = "packages", not(target_arch = "wasm32")))] deps: Option<
        &DependencyResolution,
    >,
    #[cfg(all(feature = "packages", not(target_arch = "wasm32")))] prepared_rust: Option<
        &[PreparedRustDependency],
    >,
) -> Result<CompiledProgram, lust::LustError> {
    use lust::hashbrown::{HashMap, HashSet};

    let entry_path = Path::new(entry_filename);
    let entry_dir = entry_path.parent().unwrap_or_else(|| Path::new("."));
    let mut loader = ModuleLoader::new(entry_dir.to_path_buf());
    loader.set_config(config.clone());
    #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
    if let Some(resolution) = deps {
        for dependency in resolution.lust() {
            loader.add_module_root(
                dependency.name.clone(),
                dependency.module_root.clone(),
                dependency.root_module.clone(),
            );
            if let Some(alias) = &dependency.sanitized_name {
                loader.add_module_root(
                    alias.clone(),
                    dependency.module_root.clone(),
                    dependency.root_module.clone(),
                );
            }
        }
    }
    #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
    if let Some(prepared) = prepared_rust {
        let mut seen = hashbrown::HashSet::new();
        for entry in prepared {
            for root in &entry.stub_roots {
                let key = (root.prefix.clone(), root.directory.clone());
                if seen.insert(key.clone()) {
                    loader.add_module_root(root.prefix.clone(), root.directory.clone(), None);
                }
            }
        }
    }
    #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
    if let Some(resolution) = deps {
        let extern_root = entry_dir.join("externs");
        if extern_root.exists() {
            let mut prefixes: HashSet<String> = HashSet::new();
            for dep in resolution.lua() {
                for symbol in &dep.luaopen_symbols {
                    let module = symbol
                        .strip_prefix("luaopen_")
                        .unwrap_or(symbol)
                        .replace('_', ".");
                    if let Some(prefix) = module.split('.').next() {
                        prefixes.insert(prefix.to_string());
                    }
                }
                for lua_file in &dep.lua_files {
                    let module = lua_module_name(lua_file);
                    if let Some(prefix) = module.split('.').next() {
                        prefixes.insert(prefix.to_string());
                    }
                }
            }

            if let Ok(entries) = fs::read_dir(&extern_root) {
                for entry in entries.flatten() {
                    match entry.file_type() {
                        Ok(ft) if ft.is_dir() => {
                            if let Some(prefix) = entry.file_name().to_str() {
                                prefixes.insert(prefix.to_string());
                            }
                        }
                        Ok(ft) if ft.is_file() => {
                            if entry.path().extension().and_then(|e| e.to_str()) == Some("lust")
                                && let Some(stem) =
                                    entry.path().file_stem().and_then(|s| s.to_str())
                            {
                                prefixes.insert(stem.to_string());
                            }
                        }
                        _ => {}
                    }
                }
            }

            for prefix in prefixes {
                // Extern stubs are written under externs/<prefix>/...; register both the
                // bare extern root (for top-level modules) and the nested directory so
                // nested modules resolve to externs/<prefix>/... as well.
                loader.add_module_root(prefix.clone(), extern_root.clone(), None);
                loader.add_module_root(prefix.clone(), extern_root.join(&prefix), None);
            }
        }
    }
    let program = loader.load_program_from_entry(entry_filename)?;
    let mut imports_map: HashMap<String, lust::modules::ModuleImports> = HashMap::new();
    for m in &program.modules {
        imports_map.insert(m.path.clone(), m.imports.clone());
    }

    let mut wrapped_items: Vec<Item> = Vec::new();
    for m in &program.modules {
        wrapped_items.push(Item::new(
            lust::ast::ItemKind::Module {
                name: m.path.clone(),
                items: m.items.clone(),
            },
            Span::new(0, 0, 0, 0),
        ));
    }

    let mut typechecker = TypeChecker::with_config(config);
    typechecker.set_imports_by_module(imports_map.clone());
    typechecker.check_program(&program.modules)?;
    let option_coercions = typechecker.take_option_coercions();
    let checked_array_indices = typechecker.take_checked_array_indices();
    let lambda_return_types = typechecker.take_lambda_return_types();
    let numeric_types = typechecker.take_numeric_types();
    let struct_defs = typechecker.struct_definitions();
    let function_signatures = typechecker.function_signatures();
    let mut compiler = Compiler::new();
    compiler.set_option_coercions(option_coercions);
    compiler.set_checked_array_indices(checked_array_indices);
    compiler.set_lambda_return_types(lambda_return_types);
    compiler.set_numeric_types(numeric_types);
    compiler.configure_stdlib(config);
    compiler.set_imports_by_module(imports_map);
    compiler.set_entry_module(program.entry_module.clone());
    compiler.set_function_signatures(function_signatures);
    compiler.set_struct_defs(struct_defs.clone());
    compiler.set_use_mir(config.use_mir());
    compiler.set_collect_mir(config.use_mir() || config.cranelift_jit());
    compiler.set_strict_mir(config.strict_mir());
    let functions = compiler.compile_module(&wrapped_items)?;
    let mir_functions = compiler.take_mir_functions();
    let trait_impls = compiler.get_trait_impls().to_vec();
    let mut init_funcs: Vec<(String, String)> = Vec::new();
    for m in &program.modules {
        if m.path != program.entry_module
            && let Some(init) = &m.init_function
        {
            let init_name = m
                .imports
                .function_aliases
                .get(init)
                .cloned()
                .unwrap_or_else(|| init.clone());
            init_funcs.push((m.path.clone(), init_name));
        }
    }

    Ok((functions, mir_functions, trait_impls, init_funcs, struct_defs))
}

fn print_error_with_context(source: &str, filename: &str, error: &lust::LustError, json_diag: bool) {
    if json_diag {
        let mut bag = cli::diag::DiagnosticBag::new();
        bag.add(error.clone().into_diagnostic(Some(filename), Some(source)));
        cli::diag::emit_diagnostics(&bag);
        return;
    }
    const RED: &str = "\x1b[31m";
    const BOLD: &str = "\x1b[1m";
    const DIM: &str = "\x1b[90m";
    const RESET: &str = "\x1b[0m";
    match error {
        lust::LustError::LexerError {
            line,
            column,
            message,
            module,
        } => {
            if let Some(module) = module {
                eprintln!("{DIM}in module: {module}{RESET}");
            }
            eprintln!("{RED}{BOLD}error{RESET}: {message}");
            print_source_snippet(source, filename, *line, Some(*column));
        }

        lust::LustError::ParserError {
            line,
            column,
            message,
            module,
        } => {
            if let Some(module) = module {
                eprintln!("{DIM}in module: {module}{RESET}");
            }
            eprintln!("{RED}{BOLD}error{RESET}: {message}");
            print_source_snippet(source, filename, *line, Some(*column));
        }

        lust::LustError::TypeError { message } => {
            eprintln!("{RED}{BOLD}type error{RESET}: {message}");
        }

        lust::LustError::TypeErrorWithSpan {
            message,
            line,
            column,
            module,
        } => {
            if let Some(module) = module {
                eprintln!("{DIM}in module: {module}{RESET}");
            }
            eprintln!("{RED}{BOLD}type error{RESET}: {message}");
            print_source_snippet(source, filename, *line, Some(*column));
        }

        lust::LustError::CompileError(msg) => {
            eprintln!("{RED}{BOLD}compile error{RESET}: {msg}");
        }

        lust::LustError::CompileErrorWithSpan {
            message,
            line,
            column,
            module,
        } => {
            if let Some(module) = module {
                eprintln!("{DIM}in module: {module}{RESET}");
            }
            eprintln!("{RED}{BOLD}compile error{RESET}: {message}");
            print_source_snippet(source, filename, *line, Some(*column));
        }

        lust::LustError::RuntimeErrorWithTrace {
            message,
            function: _,
            line,
            stack_trace,
        } => {
            eprintln!("{RED}{BOLD}runtime error{RESET}: {message}");
            if *line > 0 {
                print_source_snippet(source, filename, *line, None);
            } else {
                eprintln!("{DIM} --> {filename}{RESET}");
            }

            if !stack_trace.is_empty() {
                eprintln!("Stack trace:");
                for (i, frame) in stack_trace.iter().enumerate() {
                    if frame.line > 0 {
                        eprintln!("  [{i}] {} (line {})", frame.function, frame.line);
                    } else {
                        eprintln!("  [{i}] {}", frame.function);
                    }
                }
            }
        }

        lust::LustError::RuntimeError { message } => {
            eprintln!("{RED}{BOLD}runtime error{RESET}: {message}");
        }

        lust::LustError::Unknown(msg) => {
            eprintln!("{RED}{BOLD}error{RESET}: {msg}");
        }

        lust::LustError::BlockingHostCallViolation(msg) => {
            eprintln!("{RED}{BOLD}blocking host call violation{RESET}: {msg}");
        }
    }
}

fn print_source_snippet(source: &str, filename: &str, line: usize, column: Option<usize>) {
    const DIM: &str = "\x1b[90m";
    const RESET: &str = "\x1b[0m";
    let lines: Vec<&str> = source.split('\n').collect();
    let line_idx = line.saturating_sub(1);
    let code_line = lines.get(line_idx).copied().unwrap_or("");
    match column {
        Some(col) if col > 0 => {
            eprintln!("{DIM} --> {}:{}:{}{RESET}", filename, line, col);
        }

        _ => {
            eprintln!("{DIM} --> {}:{}{RESET}", filename, line);
        }
    }

    eprintln!(" {} | {}", line, code_line);
    if let Some(col) = column
        && col > 0
    {
        let mut marker = String::new();
        marker.push(' ');
        marker.push_str(&" ".repeat(line.to_string().len()));
        marker.push_str(" | ");
        marker.push_str(&" ".repeat(col.saturating_sub(1)));
        marker.push('^');
        eprintln!("{}", marker);
    }
}

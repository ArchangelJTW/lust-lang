#![cfg(feature = "std")]
#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
use lust::{
    build_local_module, collect_stub_files, load_local_module, stub_files_from_exports,
    write_stub_files, LoadedRustModule,
};
use lust::{Compiler, Item, LustConfig, ModuleLoader, Span, TypeChecker, VM};
use std::env;
use std::fs;
#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
use std::path::Path;
use std::process;
const VERSION: &str = env!("CARGO_PKG_VERSION");
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage(&args[0]);
        process::exit(1);
    }

    match args[1].as_str() {
        "--help" | "-h" => {
            print_help(&args[0]);
        }

        "--version" | "-v" => {
            print_version();
        }

        "--disassemble" | "-d" => {
            if args.len() < 3 {
                eprintln!("Error: --disassemble requires a file argument");
                process::exit(1);
            }

            run_file(&args[2], true);
        }

        "--dump-externs" => {
            if args.len() < 3 {
                eprintln!("Error: --dump-externs requires a file argument");
                process::exit(1);
            }

            dump_externs(&args[2]);
        }

        filename => {
            run_file(filename, false);
        }
    }
}

fn print_usage(program: &str) {
    eprintln!("Usage: {} [options] <script.lust>", program);
    eprintln!("       {} --help", program);
    eprintln!("       {} --version", program);
}

fn print_help(program: &str) {
    println!("Lust Programming Language v{}", VERSION);
    println!();
    println!("USAGE:");
    println!(
        "    {} <script.lust>                   Run a Lust script",
        program
    );
    println!(
        "    {} --disassemble <script.lust>     Show bytecode disassembly",
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
        "    {} --dump-externs <script.lust>    Create extern stubs for rust library modules",
        program
    );
    println!();
    println!("EXAMPLES:");
    println!(
        "    {} script.lust                     Run script.lust",
        program
    );
    println!(
        "    {} -d script.lust                  Disassemble script.lust",
        program
    );
}

fn print_version() {
    println!("Lust v{} - https://lust-lang.dev", VERSION);
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
    let modules: Vec<_> = config.rust_modules().collect();
    if modules.is_empty() {
        println!("No Rust modules configured; nothing to dump.");
        return;
    }

    let entry_path = Path::new(filename);
    let project_dir = entry_path.parent().unwrap_or_else(|| Path::new("."));
    let output_root = project_dir.join("externs");
    if let Err(e) = fs::create_dir_all(&output_root) {
        eprintln!(
            "Error creating extern output directory '{}': {}",
            output_root.display(),
            e
        );
        process::exit(1);
    }

    let mut wrote_any = false;
    for module in modules {
        let build = match build_local_module(module.path()) {
            Ok(build) => build,
            Err(err) => {
                eprintln!(
                    "Failed to build Rust module '{}': {}",
                    module.path().display(),
                    err
                );
                process::exit(1);
            }
        };
        let mut preview_vm = VM::new();
        let preview_module = match load_local_module(&mut preview_vm, &build) {
            Ok(lib) => lib,
            Err(err) => {
                eprintln!(
                    "Failed to load Rust module '{}': {}",
                    module.path().display(),
                    err
                );
                process::exit(1);
            }
        };
        let exports = preview_vm.take_exported_natives();
        preview_vm.clear_native_functions();
        drop(preview_module);
        let mut stubs = stub_files_from_exports(&exports);
        let extern_dir = module.externs_dir();
        let manual_stubs = match collect_stub_files(module.path(), extern_dir.as_deref()) {
            Ok(stubs) => stubs,
            Err(err) => {
                eprintln!(
                    "Failed to collect extern stubs for '{}': {}",
                    module.path().display(),
                    err
                );
                process::exit(1);
            }
        };
        if !manual_stubs.is_empty() {
            stubs.extend(manual_stubs);
        }
        if stubs.is_empty() {
            println!(
                "Warning: module '{}' did not expose any extern metadata or stub files",
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
        if written.is_empty() {
            println!(
                "Warning: module '{}' produced no extern files despite signatures",
                build.name
            );
            continue;
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

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn load_local_extensions(
    config: &LustConfig,
    vm: &mut VM,
) -> Result<Vec<LoadedRustModule>, String> {
    let mut loaded = Vec::new();
    for module in config.rust_modules() {
        let build = build_local_module(module.path())
            .map_err(|err| format!("{}: {}", module.path().display(), err))?;
        let loaded_module = load_local_module(vm, &build)
            .map_err(|err| format!("{}: {}", module.path().display(), err))?;
        loaded.push(loaded_module);
    }

    Ok(loaded)
}

fn run_file(filename: &str, disassemble: bool) {
    let source = match fs::read_to_string(filename) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", filename, e);
            process::exit(1);
        }
    };
    let config = match LustConfig::load_for_entry(filename) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Error loading configuration: {}", e);
            process::exit(1);
        }
    };
    let (functions, trait_impls, init_funcs, struct_defs) = match compile_program(filename, &config)
    {
        Ok(result) => result,
        Err(e) => {
            print_error_with_context(&source, filename, &e);
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
    vm.register_structs(&struct_defs);
    for (type_name, trait_name) in trait_impls {
        vm.register_trait_impl(type_name, trait_name);
    }

    #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
    let _loaded_extensions = match load_local_extensions(&config, &mut vm) {
        Ok(mods) => mods,
        Err(err) => {
            eprintln!("Failed to load Rust extensions: {}", err);
            process::exit(1);
        }
    };
    for init in init_funcs {
        if let Err(e) = vm.call(&init, vec![]) {
            print_error_with_context(&source, filename, &e);
            process::exit(1);
        }
    }

    if let Err(e) = vm.call("__script", vec![]) {
        print_error_with_context(&source, filename, &e);
        process::exit(1);
    }
}

fn compile_program(
    entry_filename: &str,
    config: &LustConfig,
) -> Result<
    (
        Vec<lust::bytecode::Function>,
        Vec<(String, String)>,
        Vec<String>,
        hashbrown::HashMap<String, lust::ast::StructDef>,
    ),
    lust::LustError,
> {
    let mut loader = ModuleLoader::new(".");
    let program = loader.load_program_from_entry(entry_filename)?;
    use hashbrown::HashMap;
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
    let struct_defs = typechecker.struct_definitions();
    let mut compiler = Compiler::new();
    compiler.configure_stdlib(config);
    compiler.set_imports_by_module(imports_map);
    compiler.set_entry_module(program.entry_module.clone());
    let functions = compiler.compile_module(&wrapped_items)?;
    let trait_impls = compiler.get_trait_impls().to_vec();
    let mut init_funcs: Vec<String> = Vec::new();
    for m in &program.modules {
        if m.path != program.entry_module {
            if let Some(init) = &m.init_function {
                init_funcs.push(init.clone());
            }
        }
    }

    Ok((functions, trait_impls, init_funcs, struct_defs))
}

fn print_error_with_context(source: &str, filename: &str, error: &lust::LustError) {
    const RED: &str = "\x1b[31m";
    const BOLD: &str = "\x1b[1m";
    const DIM: &str = "\x1b[90m";
    const RESET: &str = "\x1b[0m";
    match error {
        lust::LustError::LexerError {
            line,
            column,
            message,
            module: _,
        } => {
            eprintln!("{RED}{BOLD}error{RESET}: {message}");
            print_source_snippet(source, filename, *line, Some(*column));
        }

        lust::LustError::ParserError {
            line,
            column,
            message,
            module: _,
        } => {
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
            module: _,
        } => {
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
            module: _,
        } => {
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
    if let Some(col) = column {
        if col > 0 {
            let mut marker = String::new();
            marker.push_str(" ");
            marker.push_str(&" ".repeat(line.to_string().len()));
            marker.push_str(" | ");
            marker.push_str(&" ".repeat(col.saturating_sub(1)));
            marker.push('^');
            eprintln!("{}", marker);
        }
    }
}

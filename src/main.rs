#![cfg(feature = "std")]
#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
use lust::packages::{
    build_local_module, build_package_archive, clear_credentials, collect_stub_files,
    credentials_file, load_credentials, load_local_module, resolve_dependencies, save_credentials,
    stub_files_from_exports, write_stub_files, BuildOptions, DependencyResolution,
    DownloadedArchive, LocalBuildOutput, PackageManager, PackageManifest, RegistryClient,
    ResolvedRustDependency, StubFile, DEFAULT_BASE_URL,
};
#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
use lust::LoadedRustModule;
use lust::{Compiler, Item, LustConfig, ModuleLoader, Span, TypeChecker, VM};
use std::{
    env, fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::{self, Command},
};
#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
use toml::{self, map::Map, Value};
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

        #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
        "pkg" => {
            if let Err(err) = handle_pkg_command(&args) {
                eprintln!("Error: {err}");
                process::exit(1);
            }
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
    #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
    {
        println!(
            "    {} pkg <command> [options]         Manage Lust packages",
            program
        );
        println!(
            "    {} login [--token <token>]         Authenticate with the registry",
            program
        );
        println!(
            "    {} logout                          Clear registry credentials",
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
        "    {} -d script.lust                  Disassemble script.lust",
        program
    );
}

fn print_version() {
    println!("Lust v{} - https://lust-lang.dev", VERSION);
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
    println!("  {program} pkg login [--token <token>]");
    println!("  {program} pkg logout");
    println!("  {program} pkg publish [--manifest-path <path>] [--token <token>] [--registry <url>] [--readme <path>]");
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
    if package_dir.exists() {
        fs::remove_dir_all(&package_dir).map_err(|err| err.to_string())?;
    }
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
    if destination.exists() {
        fs::remove_dir_all(destination).map_err(|err| err.to_string())?;
    }
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
fn ensure_dependencies_table<'a>(doc: &'a mut Value) -> Result<&'a mut Map<String, Value>, String> {
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
    if resolution.rust().is_empty() {
        println!("No Rust dependencies configured; nothing to dump.");
        return;
    }

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

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn collect_rust_dependency_artifacts(
    dep: &ResolvedRustDependency,
) -> Result<(LocalBuildOutput, Vec<StubFile>), String> {
    let build = build_local_module(
        &dep.crate_dir,
        BuildOptions {
            features: &dep.features,
            default_features: dep.default_features,
        },
    )
    .map_err(|err| format!("{}: {}", dep.crate_dir.display(), err))?;

    let mut preview_vm = VM::new();
    let preview_module = load_local_module(&mut preview_vm, &build)
        .map_err(|err| format!("{}: {}", dep.crate_dir.display(), err))?;
    let exports = preview_vm.take_exported_natives();
    preview_vm.clear_native_functions();
    drop(preview_module);

    let mut stubs = stub_files_from_exports(&exports);
    let manual_stubs = collect_stub_files(&dep.crate_dir, dep.externs_override.as_deref())
        .map_err(|err| format!("{}: {}", dep.crate_dir.display(), err))?;
    if !manual_stubs.is_empty() {
        stubs.extend(manual_stubs);
    }

    Ok((build, stubs))
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn load_rust_dependencies(
    deps: &DependencyResolution,
    project_dir: &Path,
    vm: &mut VM,
) -> Result<Vec<LoadedRustModule>, String> {
    let mut loaded = Vec::new();
    if deps.rust().is_empty() {
        return Ok(loaded);
    }

    let extern_root = project_dir.join("externs");
    fs::create_dir_all(&extern_root)
        .map_err(|err| format!("{}: {}", extern_root.display(), err))?;

    for dep in deps.rust() {
        let (build, stubs) = collect_rust_dependency_artifacts(dep)?;
        if !stubs.is_empty() {
            write_stub_files(&build.name, &stubs, &extern_root)
                .map_err(|err| format!("{}: {}", extern_root.display(), err))?;
            if let Some(cache_dir) = &dep.cache_stub_dir {
                fs::create_dir_all(cache_dir).map_err(|err| {
                    format!(
                        "failed to create extern cache '{}': {}",
                        cache_dir.display(),
                        err
                    )
                })?;
                write_stub_files(&build.name, &stubs, cache_dir)
                    .map_err(|err| format!("{}: {}", cache_dir.display(), err))?;
            }
        }
        let loaded_module = load_local_module(vm, &build)
            .map_err(|err| format!("{}: {}", dep.crate_dir.display(), err))?;
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
    let entry_path = Path::new(filename);
    let project_dir = entry_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let config = match LustConfig::load_for_entry(filename) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Error loading configuration: {}", e);
            process::exit(1);
        }
    };
    #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
    let dependency_resolution = match resolve_dependencies(&config, &project_dir) {
        Ok(res) => res,
        Err(err) => {
            eprintln!("Error resolving dependencies: {}", err);
            process::exit(1);
        }
    };
    let (functions, trait_impls, init_funcs, struct_defs) = match compile_program(
        filename,
        &config,
        #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
        Some(&dependency_resolution),
    ) {
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
    let _loaded_extensions =
        match load_rust_dependencies(&dependency_resolution, &project_dir, &mut vm) {
            Ok(mods) => mods,
            Err(err) => {
                eprintln!("Failed to load Rust dependencies: {}", err);
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
    #[cfg(all(feature = "packages", not(target_arch = "wasm32")))] deps: Option<
        &DependencyResolution,
    >,
) -> Result<
    (
        Vec<lust::bytecode::Function>,
        Vec<(String, String)>,
        Vec<String>,
        hashbrown::HashMap<String, lust::ast::StructDef>,
    ),
    lust::LustError,
> {
    let entry_path = Path::new(entry_filename);
    let entry_dir = entry_path.parent().unwrap_or_else(|| Path::new("."));
    let mut loader = ModuleLoader::new(entry_dir.to_path_buf());
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
    let option_coercions = typechecker.take_option_coercions();
    let struct_defs = typechecker.struct_definitions();
    let mut compiler = Compiler::new();
    compiler.set_option_coercions(option_coercions);
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

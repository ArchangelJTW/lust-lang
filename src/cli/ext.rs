//! `lust ext` CLI commands: new, check, stubs.

use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ExtNewOptions {
    pub name: String,
    pub template: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ExtCheckResult {
    pub name: String,
    pub valid: bool,
    pub exports_count: usize,
    pub issues: Vec<String>,
}

pub fn ext_new(options: &ExtNewOptions) -> Result<PathBuf, String> {
    let target_dir = options.path.join(&options.name);
    if target_dir.exists() {
        return Err(format!("Directory already exists: {}", target_dir.display()));
    }

    fs::create_dir_all(target_dir.join("src"))
        .map_err(|e| format!("Failed to create directory: {e}"))?;

    let name = &options.name;
    let cargo_toml = format!(
        r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib", "staticlib"]

[dependencies]
lust-ext = {{ version = "0.1.0" }}
lust-ext-abi = {{ version = "0.1.0" }}
"#
    );
    fs::write(target_dir.join("Cargo.toml"), cargo_toml)
        .map_err(|e| format!("Failed to write Cargo.toml: {e}"))?;

    let src_lib = match options.template.as_str() {
        "c-abi" => format!(
            r#"use lust_ext_abi::*;

#[no_mangle]
pub extern "C" fn lust_ext_manifest_v1() -> *const LustExtManifestV1 {{
    static MANIFEST: LustExtManifestV1 = LustExtManifestV1 {{
        abi_version: LUST_EXT_ABI_VERSION,
        module_name: concat!("{name}", "\0").as_ptr() as *const core::ffi::c_char,
        exports: core::ptr::null(),
        exports_len: 0,
    }};
    &MANIFEST
}}
"#
        ),
        _ => format!(
            r#"use lust_ext::prelude::*;

#[lust_module(name = "{name}", abi = "both", capability = "pure", deterministic = true)]
pub mod {name} {{
    use super::*;

    #[lust_fn(pure, doc = "Example native function for {name}")]
    pub fn double(val: i64) -> i64 {{
        val * 2
    }}
}}
"#
        ),
    };

    fs::write(target_dir.join("src/lib.rs"), src_lib)
        .map_err(|e| format!("Failed to write src/lib.rs: {e}"))?;

    Ok(target_dir)
}

pub fn ext_check(dir: &Path) -> Result<ExtCheckResult, String> {
    let cargo_toml_path = dir.join("Cargo.toml");
    if !cargo_toml_path.exists() {
        return Err(format!("Cargo.toml not found in {}", dir.display()));
    }

    let content = fs::read_to_string(&cargo_toml_path)
        .map_err(|e| format!("Failed to read Cargo.toml: {e}"))?;

    let mut issues = Vec::new();
    let has_cdylib = content.contains("\"cdylib\"") || content.contains("'cdylib'");
    if !has_cdylib {
        issues.push("Cargo.toml crate-type does not include cdylib".to_string());
    }

    let src_lib_path = dir.join("src/lib.rs");
    let mut exports_count = 0;
    let mut name = dir.file_name().and_then(|s| s.to_str()).unwrap_or("ext").to_string();

    if src_lib_path.exists() {
        let src = fs::read_to_string(&src_lib_path).unwrap_or_default();
        let has_lust_module = src.contains("#[lust_module");
        let has_manifest = src.contains("lust_ext_manifest_v1");
        let has_legacy_register = src.contains("lust_extension_register");

        if !has_lust_module && !has_manifest && !has_legacy_register {
            issues.push("No #[lust_module], lust_ext_manifest_v1 or lust_extension_register found in src/lib.rs".to_string());
        }

        // Count approximate exports
        exports_count = src.matches("#[lust_fn").count();
        if exports_count == 0 && (has_manifest || has_legacy_register) {
            exports_count = 1;
        }

        // Extract module name if available
        if let Some(pos) = src.find("name = \"") {
            let after = &src[pos + 8..];
            if let Some(end) = after.find('"') {
                name = after[..end].to_string();
            }
        }
    } else {
        issues.push("src/lib.rs not found".to_string());
    }

    let valid = issues.is_empty();
    Ok(ExtCheckResult {
        name,
        valid,
        exports_count,
        issues,
    })
}

pub fn ext_stubs(dir: &Path) -> Result<String, String> {
    let check = ext_check(dir)?;
    let mut stub = String::new();
    stub.push_str(&format!("--- Auto-generated Lust interface for {}\n", check.name));
    stub.push_str(&format!("module {}\n\n", check.name));

    let src_lib_path = dir.join("src/lib.rs");
    if src_lib_path.exists() {
        let src = fs::read_to_string(&src_lib_path).unwrap_or_default();
        for line in src.lines() {
            if line.trim().starts_with("pub fn ") {
                let sig = line.trim();
                stub.push_str(&format!("--- {}\n", sig));
                stub.push_str(&format!("extern function {};\n\n", sig.trim_prefix("pub fn ")));
            }
        }
    }

    Ok(stub)
}

pub fn run_ext_command(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err("Usage: lust ext <new|check|stubs> [options]".to_string());
    }

    match args[0].as_str() {
        "help" | "--help" | "-h" => {
            println!("Usage: lust ext <new|check|stubs> [options]");
            Ok(())
        }
        "new" => {
            if args.len() < 2 {
                return Err("Usage: lust ext new <name> [--template <basic|c-abi>]".to_string());
            }
            let name = &args[1];
            let mut template = "basic".to_string();
            let mut i = 2;
            while i < args.len() {
                if args[i] == "--template" && i + 1 < args.len() {
                    template = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            let opts = ExtNewOptions {
                name: name.clone(),
                template,
                path: PathBuf::from("."),
            };
            let created = ext_new(&opts)?;
            println!("Created extension at {}", created.display());
            Ok(())
        }
        "check" => {
            let path = if args.len() > 1 {
                PathBuf::from(&args[1])
            } else {
                PathBuf::from(".")
            };
            let res = ext_check(&path)?;
            if res.valid {
                println!("Extension '{}' is valid ({} export(s)).", res.name, res.exports_count);
                Ok(())
            } else {
                eprintln!("Extension '{}' check failed:", res.name);
                for issue in &res.issues {
                    eprintln!(" - {issue}");
                }
                Err("Extension check failed".to_string())
            }
        }
        "stubs" => {
            let path = if args.len() > 1 {
                PathBuf::from(&args[1])
            } else {
                PathBuf::from(".")
            };
            let stub = ext_stubs(&path)?;
            println!("{stub}");
            Ok(())
        }
        other => Err(format!("Unknown ext subcommand: {other}")),
    }
}

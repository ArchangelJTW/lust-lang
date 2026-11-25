use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    println!("cargo:rustc-link-search={}", manifest_dir.display());
    println!("cargo:rerun-if-changed=link.x");
    println!("cargo:rerun-if-changed=src/default_lust_kernel.rs");
    println!("cargo:rerun-if-env-changed=LUST_RV32_KERNEL_RS");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let dst = out_dir.join("lust_kernel.rs");

    let src = match env::var_os("LUST_RV32_KERNEL_RS") {
        Some(p) if !p.is_empty() => {
            let p = PathBuf::from(p);
            println!("cargo:rerun-if-changed={}", p.display());
            p
        }
        _ => manifest_dir.join("src/default_lust_kernel.rs"),
    };

    fs::copy(&src, &dst).unwrap_or_else(|e| {
        panic!("failed to copy kernel {} -> {}: {e}", src.display(), dst.display());
    });
}

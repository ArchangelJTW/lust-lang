//! Integration & parity test suite for `NG-W13-1C`:
//! Zero-FFI Cargo interop in Rust AOT:
//! 1. `lust build --emit=rust-crate -o <dir>` generating a standalone offline-buildable Cargo crate
//!    (`Cargo.toml` + `src/lib.rs`) depending on `lust-aot-rt` (and optional local `--link-crate` deps).
//! 2. `lust build --backend rust --link-crate <crate_dir_or_rlib>` generating direct typed
//!    `extern crate <ext>; <ext>::<fn>(...)` calls with zero `LustExtVal` C-ABI marshaling and
//!    passing `-C lto=thin --extern <ext>=<path>.rlib` to `rustc`.
//! 3. `lust::bindgen::RustAotBuilder` (`RustAotBuilder::new(lust_path).emit_to_out_dir(out_file)`)
//!    generating `include!()`-ready Rust AOT modules for `build.rs`.
//! 4. Strict `TASK_ALLOWS_UNSAFE=0` (zero `unsafe` in emitted Rust and in this test),
//!    Hazard 9 panic-freedom (zero `.unwrap(` / `panic!(` in emitted Rust),
//!    and capability-gate preservation (`--deny-ffi` / `--pure`).

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::tempdir;

use lust::bindgen::RustAotBuilder;
use lust::codegen_zig::aot::compiler::{RustLinkOptions, RustLinkedCrate};
use lust::codegen_zig::aot::{
    AotCompiler, AotLinkOptions, GasMetering, RustCompileOptions, ensure_cached_aot_rt_rlib,
};

fn lust_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_lust"))
}

fn rustc_bin() -> String {
    std::env::var("RUSTC")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "rustc".to_string())
}

fn apply_cargo_env(cmd: &mut Command) {
    if let Ok(rustup) = std::env::var("RUSTUP_HOME") {
        cmd.env("RUSTUP_HOME", rustup);
    }
    if let Ok(cargo_home) = std::env::var("CARGO_HOME") {
        cmd.env("CARGO_HOME", cargo_home);
    }
}

fn assert_zero_unsafe_and_panic_free(code: &str, label: &str) {
    assert!(
        !code.contains("unsafe"),
        "[{label}] emitted Rust must contain ZERO `unsafe` (TASK_ALLOWS_UNSAFE=0):\n{code}"
    );
    assert!(
        !code.contains(".unwrap("),
        "[{label}] emitted Rust must contain ZERO `.unwrap(` calls (Hazard 9):\n{code}"
    );
    assert!(
        !code.contains("panic!("),
        "[{label}] emitted Rust must contain ZERO `panic!(` calls (Hazard 9):\n{code}"
    );
}

fn create_fast_math_crate(dir: &Path) -> PathBuf {
    let crate_dir = dir.join("fast_math");
    fs::create_dir_all(crate_dir.join("src")).expect("mkdir fast_math/src");
    fs::write(
        crate_dir.join("Cargo.toml"),
        r#"[package]
name = "fast-math"
version = "0.1.0"
edition = "2024"

[lib]
name = "fast_math"
crate-type = ["rlib"]

[workspace]
"#,
    )
    .expect("write fast_math Cargo.toml");

    fs::write(
        crate_dir.join("src/lib.rs"),
        r#"//! Pure-Rust Zero-FFI library crate (no C-ABI manifest, no LustExtVal, no unsafe).

#[inline]
pub fn fast_add(a: i64, b: i64) -> i64 {
    a.wrapping_add(b)
}

#[inline]
pub fn scale_f64(x: f64, factor: f64) -> f64 {
    x * factor
}

#[inline]
pub fn is_positive(n: i64) -> bool {
    n > 0
}

#[inline]
pub fn greet(name: &str) -> String {
    format!("hello, {name}!")
}
"#,
    )
    .expect("write fast_math lib.rs");

    crate_dir
}

// ---------------------------------------------------------------------------
// 1. `lust build --emit=rust-crate -o <dir>` + offline cargo build + consumer
// ---------------------------------------------------------------------------

#[test]
fn test_emit_rust_crate_cli_generates_buildable_cargo_package_and_consumer_interop() {
    let tmp = tempdir().expect("tempdir");
    let script = tmp.path().join("math_lib.lust");
    fs::write(
        &script,
        r##"
function fib(n: int): int
    if n <= 1 then
        return n
    end
    local a: int = 0
    local b: int = 1
    local i: int = 2
    while i <= n do
        local next: int = a + b
        a = b
        b = next
        i = i + 1
    end
    return b
end

function route_weight(score: float, bonus: float): float
    return score * 2.5 + bonus
end

function format_tag(prefix: string, id: int): string
    return prefix .. "#" .. tostring(id)
end
"##,
    )
    .expect("write math_lib.lust");

    let out_crate_dir = tmp.path().join("lust_math_crate");
    let out = Command::new(lust_bin())
        .arg("build")
        .arg(&script)
        .arg("--emit=rust-crate")
        .arg("-o")
        .arg(&out_crate_dir)
        .output()
        .expect("invoke lust build --emit=rust-crate");

    assert!(
        out.status.success(),
        "lust build --emit=rust-crate failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let manifest_path = out_crate_dir.join("Cargo.toml");
    let lib_rs_path = out_crate_dir.join("src/lib.rs");
    assert!(manifest_path.exists(), "Cargo.toml must be generated");
    assert!(lib_rs_path.exists(), "src/lib.rs must be generated");

    let manifest = fs::read_to_string(&manifest_path).expect("read Cargo.toml");
    assert!(
        manifest.contains("lust-aot-rt = { path ="),
        "Cargo.toml must depend on lust-aot-rt:\n{manifest}"
    );
    assert!(
        manifest.contains("name = \"lust_math_crate\""),
        "Cargo.toml package name mismatch:\n{manifest}"
    );

    let lib_rs = fs::read_to_string(&lib_rs_path).expect("read src/lib.rs");
    assert_zero_unsafe_and_panic_free(&lib_rs, "emit_rust_crate");
    assert!(
        lib_rs.contains("pub fn fib("),
        "emitted crate must export `pub fn fib`:\n{lib_rs}"
    );
    assert!(
        lib_rs.contains("pub fn route_weight("),
        "emitted crate must export `pub fn route_weight`:\n{lib_rs}"
    );
    assert!(
        lib_rs.contains("pub fn format_tag("),
        "emitted crate must export `pub fn format_tag`:\n{lib_rs}"
    );
    assert!(
        !lib_rs.contains("\nfn main()"),
        "emitted library crate must not define `fn main()`:\n{lib_rs}"
    );

    // Build the generated Cargo crate completely offline
    let cargo_target = tmp.path().join("cargo_target");
    let mut build_cmd = Command::new("cargo");
    apply_cargo_env(&mut build_cmd);
    let build_out = build_cmd
        .arg("build")
        .arg("--offline")
        .arg("--manifest-path")
        .arg(&manifest_path)
        .arg("--target-dir")
        .arg(&cargo_target)
        .output()
        .expect("cargo build emitted rust crate");
    assert!(
        build_out.status.success(),
        "cargo build --offline on emitted crate failed:\nstdout: {}\nstderr: {}\nlib.rs:\n{lib_rs}",
        String::from_utf8_lossy(&build_out.stdout),
        String::from_utf8_lossy(&build_out.stderr)
    );

    // Build & run a consumer Cargo binary crate that imports and calls the emitted crate directly
    let consumer_dir = tmp.path().join("consumer_app");
    fs::create_dir_all(consumer_dir.join("src")).expect("mkdir consumer src");
    let emitted_canon = fs::canonicalize(&out_crate_dir).expect("canonicalize emitted crate");
    fs::write(
        consumer_dir.join("Cargo.toml"),
        format!(
            r#"[package]
name = "consumer_app"
version = "0.1.0"
edition = "2024"

[dependencies]
lust_math_crate = {{ path = "{}" }}

[workspace]
"#,
            emitted_canon.display()
        ),
    )
    .expect("write consumer Cargo.toml");

    fs::write(
        consumer_dir.join("src/main.rs"),
        r#"fn main() {
    let f10 = lust_math_crate::fib(10);
    let w = lust_math_crate::route_weight(4.0, 3.0);
    let tag = lust_math_crate::format_tag("route", 7);
    println!("fib={f10};w={w};tag={tag}");
}
"#,
    )
    .expect("write consumer main.rs");

    let mut run_cmd = Command::new("cargo");
    apply_cargo_env(&mut run_cmd);
    let run_out = run_cmd
        .arg("run")
        .arg("--offline")
        .arg("--quiet")
        .arg("--manifest-path")
        .arg(consumer_dir.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(&cargo_target)
        .output()
        .expect("cargo run consumer_app");
    assert!(
        run_out.status.success(),
        "consumer cargo run failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&run_out.stdout),
        String::from_utf8_lossy(&run_out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&run_out.stdout).trim(),
        "fib=55;w=13;tag=route#7"
    );
}

// ---------------------------------------------------------------------------
// 2. `--backend rust --link-crate <crate_dir>` Zero-FFI direct rlib + ThinLTO
// ---------------------------------------------------------------------------

#[test]
fn test_link_crate_dir_zero_ffi_direct_rlib_and_thin_lto() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");
    let fast_math_dir = create_fast_math_crate(tmp.path());

    let script = tmp.path().join("use_fast_math.lust");
    fs::write(
        &script,
        r#"
extern "rust"
    function fast_math.fast_add(a: int, b: int): int
    function fast_math.scale_f64(x: float, factor: float): float
    function fast_math.is_positive(n: int): bool
    function fast_math.greet(name: string): string
end

function main(): int
    local acc: int = 0
    local i: int = 1
    while i <= 10 do
        acc = fast_math.fast_add(acc, i)
        i = i + 1
    end
    local scaled: float = fast_math.scale_f64(3.5, 4.0)
    local pos: bool = fast_math.is_positive(acc)
    local msg: string = fast_math.greet("lust")
    println("acc=" .. tostring(acc) .. ";scaled=" .. tostring(scaled) .. ";pos=" .. tostring(pos) .. ";msg=" .. msg)
    return 0
end
"#,
    )
    .expect("write use_fast_math.lust");

    // 2a. Inspect generated Rust source via `--emit-rs --backend rust --link-crate <dir>`
    let emitted_rs = tmp.path().join("use_fast_math.rs");
    let mut emit_cmd = Command::new(lust_bin());
    apply_cargo_env(&mut emit_cmd);
    let emit_out = emit_cmd
        .arg("build")
        .arg(&script)
        .arg("--backend")
        .arg("rust")
        .arg("--emit-rs")
        .arg("--release-fast")
        .arg("--link-crate")
        .arg(&fast_math_dir)
        .arg("-o")
        .arg(&emitted_rs)
        .output()
        .expect("invoke lust build --backend rust --emit-rs --link-crate");
    assert!(
        emit_out.status.success(),
        "--emit-rs --link-crate failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&emit_out.stdout),
        String::from_utf8_lossy(&emit_out.stderr)
    );
    let rs_src = fs::read_to_string(&emitted_rs).expect("read emitted rs");
    assert_zero_unsafe_and_panic_free(&rs_src, "link_crate_zero_ffi");
    assert!(
        rs_src.contains("extern crate fast_math;"),
        "must emit `extern crate fast_math;`:\n{rs_src}"
    );
    assert!(
        rs_src.contains("fast_math::fast_add(")
            && rs_src.contains("fast_math::scale_f64(")
            && rs_src.contains("fast_math::is_positive(")
            && rs_src.contains("fast_math::greet("),
        "must emit direct typed Rust calls into fast_math:\n{rs_src}"
    );
    assert!(
        !rs_src.contains("LustExtVal") && !rs_src.contains("lust_ext_manifest_v1"),
        "Zero-FFI path must NOT box values into LustExtVal:\n{rs_src}"
    );

    // 2b. Full native binary build with `-C lto=thin --extern fast_math=...rlib`
    let out_bin = tmp.path().join("app_zero_ffi");
    let mut build_cmd = Command::new(lust_bin());
    apply_cargo_env(&mut build_cmd);
    let build_out = build_cmd
        .arg("build")
        .arg(&script)
        .arg("--backend")
        .arg("rust")
        .arg("--release-fast")
        .arg("--cache-dir")
        .arg(&cache_dir)
        .arg("--link-crate")
        .arg(&fast_math_dir)
        .arg("-o")
        .arg(&out_bin)
        .output()
        .expect("invoke lust build --backend rust --link-crate");
    assert!(
        build_out.status.success(),
        "lust build --backend rust --link-crate failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&build_out.stdout),
        String::from_utf8_lossy(&build_out.stderr)
    );
    assert!(out_bin.exists(), "compiled native binary must exist");

    let run = Command::new(&out_bin).output().expect("run app_zero_ffi");
    assert!(
        run.status.success(),
        "app_zero_ffi execution failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).trim(),
        "acc=55;scaled=14;pos=true;msg=hello, lust!"
    );
}

// ---------------------------------------------------------------------------
// 3. `--backend rust --link-crate <path/to/lib.rlib>` direct prebuilt rlib
// ---------------------------------------------------------------------------

#[test]
fn test_link_crate_direct_prebuilt_rlib_path_with_thin_lto() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");

    // Precompile a standalone pure-Rust `.rlib` directly with `rustc`
    let ext_rs = tmp.path().join("geo_calc.rs");
    let ext_rlib = tmp.path().join("libgeo_calc.rlib");
    fs::write(
        &ext_rs,
        r#"#[inline]
pub fn manhattan(x1: i64, y1: i64, x2: i64, y2: i64) -> i64 {
    (x1 - x2).abs() + (y1 - y2).abs()
}
"#,
    )
    .expect("write geo_calc.rs");

    let rustc_out = Command::new(rustc_bin())
        .arg("--edition=2024")
        .arg("--crate-name")
        .arg("geo_calc")
        .arg("--crate-type=rlib")
        .arg("-C")
        .arg("opt-level=3")
        .arg("-C")
        .arg("panic=abort")
        .arg(&ext_rs)
        .arg("-o")
        .arg(&ext_rlib)
        .output()
        .expect("compile libgeo_calc.rlib");
    assert!(
        rustc_out.status.success(),
        "rustc libgeo_calc.rlib failed: {}",
        String::from_utf8_lossy(&rustc_out.stderr)
    );

    let lust_src = r#"
extern "rust"
    function geo_calc.manhattan(x1: int, y1: int, x2: int, y2: int): int
end

function main(): int
    local d: int = geo_calc.manhattan(10, 25, 3, 5)
    println(d)
    return 0
end
"#;
    let script = tmp.path().join("geo_app.lust");
    fs::write(&script, lust_src).expect("write geo_app.lust");

    // 3a. CLI `--backend rust --link-crate <libgeo_calc.rlib>`
    let cli_bin = tmp.path().join("geo_cli_bin");
    let cli_out = Command::new(lust_bin())
        .arg("build")
        .arg(&script)
        .arg("--backend")
        .arg("rust")
        .arg("--release-fast")
        .arg("--cache-dir")
        .arg(&cache_dir)
        .arg("--link-crate")
        .arg(&ext_rlib)
        .arg("-o")
        .arg(&cli_bin)
        .output()
        .expect("invoke lust build --backend rust --link-crate <.rlib>");
    assert!(
        cli_out.status.success(),
        "CLI --link-crate <.rlib> failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&cli_out.stdout),
        String::from_utf8_lossy(&cli_out.stderr)
    );
    let run_cli = Command::new(&cli_bin).output().expect("run geo_cli_bin");
    assert!(run_cli.status.success());
    assert_eq!(String::from_utf8_lossy(&run_cli.stdout).trim(), "27");

    // 3b. Rust API `AotCompiler::compile_source_rust_with_rlibs`
    let api_bin = tmp.path().join("geo_api_bin");
    let opts = RustCompileOptions {
        cache_dir: Some(cache_dir),
        ..RustCompileOptions::default()
    };
    let links = RustLinkOptions {
        aot_links: AotLinkOptions::default(),
        linked_crates: vec![RustLinkedCrate {
            crate_name: "geo_calc".to_string(),
            rlib_path: ext_rlib,
            dependency_dirs: Vec::new(),
        }],
        lto_thin: true,
    };
    AotCompiler::compile_source_rust_with_rlibs(lust_src, &api_bin, &opts, &links)
        .expect("compile_source_rust_with_rlibs");
    let run_api = Command::new(&api_bin).output().expect("run geo_api_bin");
    assert!(run_api.status.success());
    assert_eq!(String::from_utf8_lossy(&run_api.stdout).trim(), "27");
}

// ---------------------------------------------------------------------------
// 4. Capability denial & fail-closed checks for `--backend rust --link-crate`
// ---------------------------------------------------------------------------

#[test]
fn test_link_crate_rust_backend_capability_denial_and_missing_rlib() {
    let tmp = tempdir().expect("tempdir");
    let script = tmp.path().join("simple.lust");
    fs::write(&script, "function main(): int\n    return 0\nend\n").expect("write");

    let dummy_rlib = tmp.path().join("libdummy.rlib");
    fs::write(&dummy_rlib, b"dummy").expect("write dummy");

    // --deny-ffi must fail closed
    let deny = Command::new(lust_bin())
        .arg("build")
        .arg(&script)
        .arg("--backend")
        .arg("rust")
        .arg("--deny-ffi")
        .arg("--link-crate")
        .arg(&dummy_rlib)
        .output()
        .expect("invoke --deny-ffi");
    assert!(!deny.status.success());
    let deny_err = String::from_utf8_lossy(&deny.stderr);
    assert!(
        deny_err.contains("CapabilityDenied"),
        "expected CapabilityDenied, got: {deny_err}"
    );

    // --pure must fail closed
    let pure = Command::new(lust_bin())
        .arg("build")
        .arg(&script)
        .arg("--backend")
        .arg("rust")
        .arg("--pure")
        .arg("--link-crate")
        .arg(&dummy_rlib)
        .output()
        .expect("invoke --pure");
    assert!(!pure.status.success());
    assert!(String::from_utf8_lossy(&pure.stderr).contains("CapabilityDenied"));

    // Missing rlib must fail closed
    let missing = Command::new(lust_bin())
        .arg("build")
        .arg(&script)
        .arg("--backend")
        .arg("rust")
        .arg("--link-crate")
        .arg("/nonexistent/path/libmissing.rlib")
        .output()
        .expect("invoke missing rlib");
    assert!(!missing.status.success());
    let miss_err = String::from_utf8_lossy(&missing.stderr);
    assert!(
        miss_err.contains("missing external library"),
        "expected missing external library, got: {miss_err}"
    );

    // Non-.rlib regular file must fail closed
    let bad_file = tmp.path().join("not_an_rlib.txt");
    fs::write(&bad_file, b"hello").expect("write txt");
    let bad = Command::new(lust_bin())
        .arg("build")
        .arg(&script)
        .arg("--backend")
        .arg("rust")
        .arg("--link-crate")
        .arg(&bad_file)
        .output()
        .expect("invoke bad file");
    assert!(!bad.status.success());
    let bad_err = String::from_utf8_lossy(&bad.stderr);
    assert!(
        bad_err.contains(".rlib"),
        "expected .rlib diagnostic, got: {bad_err}"
    );
}

// ---------------------------------------------------------------------------
// 5. `--emit=rust-crate` with `--link-crate <dir>` emits path dependency
// ---------------------------------------------------------------------------

#[test]
fn test_emit_rust_crate_with_linked_local_crate_dependency() {
    let tmp = tempdir().expect("tempdir");
    let fast_math_dir = create_fast_math_crate(tmp.path());

    let script = tmp.path().join("combo.lust");
    fs::write(
        &script,
        r#"
extern "rust"
    function fast_math.fast_add(a: int, b: int): int
end

function add_plus_one(a: int, b: int): int
    return fast_math.fast_add(a, b) + 1
end
"#,
    )
    .expect("write combo.lust");

    let out_crate_dir = tmp.path().join("combo_crate");
    let out = Command::new(lust_bin())
        .arg("build")
        .arg(&script)
        .arg("--emit=rust-crate")
        .arg("--link-crate")
        .arg(&fast_math_dir)
        .arg("-o")
        .arg(&out_crate_dir)
        .output()
        .expect("invoke --emit=rust-crate --link-crate");
    assert!(
        out.status.success(),
        "--emit=rust-crate --link-crate failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let manifest = fs::read_to_string(out_crate_dir.join("Cargo.toml")).expect("read Cargo.toml");
    assert!(manifest.contains("lust-aot-rt = { path ="));
    assert!(
        manifest.contains("fast-math = { path ="),
        "generated Cargo.toml must list linked crate dependency:\n{manifest}"
    );
    let lib_rs = fs::read_to_string(out_crate_dir.join("src/lib.rs")).expect("read src/lib.rs");
    assert_zero_unsafe_and_panic_free(&lib_rs, "emit_rust_crate_linked");
    assert!(lib_rs.contains("extern crate fast_math;"));
    assert!(lib_rs.contains("fast_math::fast_add("));

    let cargo_target = tmp.path().join("cargo_target");
    let mut build_cmd = Command::new("cargo");
    apply_cargo_env(&mut build_cmd);
    let build_out = build_cmd
        .arg("build")
        .arg("--offline")
        .arg("--manifest-path")
        .arg(out_crate_dir.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(&cargo_target)
        .output()
        .expect("cargo build combo_crate");
    assert!(
        build_out.status.success(),
        "cargo build --offline on combo_crate failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&build_out.stdout),
        String::from_utf8_lossy(&build_out.stderr)
    );
}

// ---------------------------------------------------------------------------
// 6. `RustAotBuilder` (`src/bindgen.rs`) for `build.rs` + `include!()`
// ---------------------------------------------------------------------------

#[test]
fn test_rust_aot_builder_emit_to_out_dir_and_include_macro_end_to_end() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");
    let out_dir = tmp.path().join("fake_out_dir");
    fs::create_dir_all(&out_dir).expect("mkdir fake_out_dir");

    let script = tmp.path().join("router.lust");
    fs::write(
        &script,
        r#"
function route_tier(latency_ms: int, tokens: int): int
    if latency_ms < 50 and tokens < 2000 then
        return 1
    elseif tokens < 8000 then
        return 2
    else
        return 3
    end
end

function spin_sum(n: int): int
    local acc: int = 0
    local i: int = 1
    while i <= n do
        acc = acc + i
        i = i + 1
    end
    return acc
end
"#,
    )
    .expect("write router.lust");

    // 6a. Without OUT_DIR set in current process, emit_to_out_dir returns a clear error
    if std::env::var_os("OUT_DIR").is_none() {
        let err = RustAotBuilder::new(&script)
            .emit_to_out_dir("router_aot.rs")
            .expect_err("emit_to_out_dir without OUT_DIR must error");
        assert!(
            err.to_string().contains("OUT_DIR"),
            "expected OUT_DIR diagnostic, got: {err}"
        );
    }

    // 6b. Test `RustAotBuilder::new(lust_path).emit_to_out_dir("router_aot.rs")` inside
    //     a child invocation with `OUT_DIR` configured (zero `unsafe` env mutation!).
    let current_exe = std::env::current_exe().expect("current_exe");
    let child = Command::new(&current_exe)
        .env("OUT_DIR", &out_dir)
        .env("LUST_W13_1C_BUILDER_ENTRY", &script)
        .arg("--exact")
        .arg("test_rust_aot_builder_emit_to_out_dir_child_probe")
        .arg("--ignored")
        .arg("--nocapture")
        .output()
        .expect("spawn child emit_to_out_dir probe");
    assert!(
        child.status.success(),
        "emit_to_out_dir child probe failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&child.stdout),
        String::from_utf8_lossy(&child.stderr)
    );
    let child_stdout = String::from_utf8_lossy(&child.stdout);
    assert!(
        child_stdout.contains("cargo:rerun-if-changed="),
        "emit_to_out_dir must print cargo:rerun-if-changed directives:\n{child_stdout}"
    );

    let generated_rs_path = out_dir.join("router_aot.rs");
    assert!(
        generated_rs_path.exists(),
        "router_aot.rs must exist in OUT_DIR"
    );
    let generated_src = fs::read_to_string(&generated_rs_path).expect("read router_aot.rs");
    assert_zero_unsafe_and_panic_free(&generated_src, "RustAotBuilder");
    assert!(
        !generated_src.contains("#!["),
        "include!()-ready Rust AOT module must NOT contain inner `#![...]` crate attributes:\n{generated_src}"
    );
    assert!(
        generated_src.contains("pub fn route_tier("),
        "emitted module must expose `pub fn route_tier`:\n{generated_src}"
    );
    assert!(
        generated_src.contains("pub fn spin_sum_with_ctx("),
        "gas-metered function must expose `pub fn spin_sum_with_ctx`:\n{generated_src}"
    );

    // 6c. Compile a real host Rust binary using `include!(concat!(env!("OUT_DIR"), "/router_aot.rs"))`
    let rt_opts = RustCompileOptions {
        cache_dir: Some(cache_dir),
        ..RustCompileOptions::default()
    };
    let rt_rlib = ensure_cached_aot_rt_rlib(&rt_opts).expect("cached liblust_aot_rt.rlib");

    let host_rs = tmp.path().join("host_driver.rs");
    let host_bin = tmp.path().join("host_driver_bin");
    fs::write(
        &host_rs,
        r#"extern crate lust_aot_rt;

pub mod router_aot {
    include!(concat!(env!("OUT_DIR"), "/router_aot.rs"));
}

fn main() {
    let t1 = router_aot::route_tier(25, 1000).expect("t1");
    let t2 = router_aot::route_tier(120, 4000).expect("t2");
    let t3 = router_aot::route_tier(120, 12000).expect("t3");

    let mut generous_ctx = lust_aot_rt::AotCtx::with_gas_limit(100_000);
    let sum_ok = router_aot::spin_sum_with_ctx(&mut generous_ctx, 100).expect("spin_sum ok");

    let mut tight_ctx = lust_aot_rt::AotCtx::with_gas_limit(3);
    let gas_err = router_aot::spin_sum_with_ctx(&mut tight_ctx, 1000)
        .expect_err("tight gas limit must trigger OutOfGas");

    println!("tiers={t1},{t2},{t3};sum={sum_ok};err={}", gas_err.error_class());
}
"#,
    )
    .expect("write host_driver.rs");

    let compile_host = Command::new(rustc_bin())
        .env("OUT_DIR", &out_dir)
        .arg("--edition=2024")
        .arg("-C")
        .arg("opt-level=3")
        .arg("-C")
        .arg("panic=abort")
        .arg("--extern")
        .arg(format!("lust_aot_rt={}", rt_rlib.display()))
        .arg(&host_rs)
        .arg("-o")
        .arg(&host_bin)
        .output()
        .expect("compile host_driver.rs with include!(OUT_DIR/router_aot.rs)");
    assert!(
        compile_host.status.success(),
        "host_driver compilation failed:\nstderr: {}\ngenerated router_aot.rs:\n{generated_src}",
        String::from_utf8_lossy(&compile_host.stderr)
    );

    let run_host = Command::new(&host_bin).output().expect("run host_driver_bin");
    assert!(
        run_host.status.success(),
        "host_driver execution failed: {}",
        String::from_utf8_lossy(&run_host.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&run_host.stdout).trim(),
        "tiers=1,2,3;sum=5050;err=OutOfGas"
    );
}

#[test]
#[ignore = "Helper probe invoked by test_rust_aot_builder_emit_to_out_dir_and_include_macro_end_to_end with OUT_DIR set"]
fn test_rust_aot_builder_emit_to_out_dir_child_probe() {
    let entry = std::env::var("LUST_W13_1C_BUILDER_ENTRY")
        .expect("LUST_W13_1C_BUILDER_ENTRY must be set by parent test");
    let out_path = RustAotBuilder::new(&entry)
        .gas_metering(GasMetering::LoopChunked)
        .emit_to_out_dir("router_aot.rs")
        .expect("RustAotBuilder::emit_to_out_dir must succeed");
    assert!(out_path.exists(), "emitted file must exist");
}

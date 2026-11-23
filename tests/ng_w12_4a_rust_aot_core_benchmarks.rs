//! Integration & parity test suite for `NG-W12-4A`:
//! Core Rust AOT Emitter (`src/aot/rust_emitter.rs` Tiers 1-3),
//! cached `liblust_aot_rt.rlib` compilation (`src/aot/compiler.rs`),
//! and `lust build --backend rust` CLI (`src/main.rs`).

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use tempfile::tempdir;

use lust::codegen_zig::aot::{
    AotCompileOptions, AotCompiler, AotTargetKind, RustCompileOptions, ensure_cached_aot_rt_rlib,
};

const ALL_10_BENCHMARKS: [&str; 10] = [
    "array",
    "calls",
    "fib",
    "fields",
    "floatmath",
    "methods",
    "nested",
    "sieve",
    "strings",
    "tree",
];

fn lust_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_lust"))
}

fn benchmark_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks")
        .join("suite")
        .join(format!("{name}.lust"))
}

fn run_vm_stdout(script_path: &Path) -> String {
    let out = Command::new(lust_bin())
        .arg("run")
        .arg(script_path)
        .arg("--backend")
        .arg("vm")
        .arg("--jit=off")
        .output()
        .expect("invoke lust vm");
    assert!(
        out.status.success(),
        "lust vm failed on {}:\nstdout: {}\nstderr: {}",
        script_path.display(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf8 stdout")
}

fn run_jit_stdout(script_path: &Path) -> String {
    let out = Command::new(lust_bin())
        .arg("run")
        .arg(script_path)
        .arg("--backend")
        .arg("cranelift")
        .output()
        .expect("invoke lust cranelift jit");
    assert!(
        out.status.success(),
        "lust cranelift jit failed on {}:\nstdout: {}\nstderr: {}",
        script_path.display(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf8 stdout")
}

fn build_rust_aot_cli(
    script_path: &Path,
    out_bin: &Path,
    cache_dir: &Path,
    extra_args: &[&str],
) -> (std::process::Output, std::time::Duration) {
    let start = Instant::now();
    let mut cmd = Command::new(lust_bin());
    cmd.arg("build")
        .arg(script_path)
        .arg("--backend")
        .arg("rust")
        .arg("--release-fast")
        .arg("--cache-dir")
        .arg(cache_dir)
        .arg("-o")
        .arg(out_bin);
    for a in extra_args {
        cmd.arg(a);
    }
    let out = cmd.output().expect("invoke lust build --backend rust");
    (out, start.elapsed())
}

// ---------------------------------------------------------------------------
// Test 1: All 10/10 benchmarks/suite/*.lust compile via `lust build --backend rust`
//         and produce bit-exact stdout parity with JIT/VM + rlib cache verification.
// ---------------------------------------------------------------------------

#[test]
fn test_all_10_benchmarks_rust_aot_bit_exact_stdout_and_rlib_cache() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");

    // Pre-warm the liblust_aot_rt.rlib cache once and verify cache hit reuse.
    let warm_opts = RustCompileOptions {
        aot: AotCompileOptions {
            target_kind: AotTargetKind::Executable,
            release_mode: "ReleaseFast".to_string(),
            strip: false,
            max_memory_bytes: None,
            target_triple: None,
        },
        cache_dir: Some(cache_dir.clone()),
        ..RustCompileOptions::default()
    };
    let rlib_first = ensure_cached_aot_rt_rlib(&warm_opts).expect("build rlib cache");
    assert!(rlib_first.exists(), "cached liblust_aot_rt.rlib must exist");
    let first_mtime = fs::metadata(&rlib_first)
        .expect("rlib metadata")
        .modified()
        .expect("rlib mtime");

    let t_hit = Instant::now();
    let rlib_second = ensure_cached_aot_rt_rlib(&warm_opts).expect("rlib cache hit");
    let hit_elapsed = t_hit.elapsed();
    assert_eq!(rlib_first, rlib_second);
    let second_mtime = fs::metadata(&rlib_second)
        .expect("rlib metadata 2")
        .modified()
        .expect("rlib mtime 2");
    assert_eq!(
        first_mtime, second_mtime,
        "second ensure_cached_aot_rt_rlib must reuse existing .rlib without rebuilding"
    );
    assert!(
        hit_elapsed.as_millis() < 200,
        "cached rlib lookup should be fast, took {} ms",
        hit_elapsed.as_millis()
    );

    // Known exact outputs (also cross-checked live bit-exact against Cranelift JIT below)
    let expected_exact: [(&str, &str); 10] = [
        ("array", "4999995000000\n"),
        ("calls", "99999990000000\n"),
        ("fib", "832040\n"),
        ("fields", "50000005000000\n"),
        ("floatmath", "199999960.06314665\n"),
        ("methods", "49999995000000\n"),
        ("nested", "23135141\n"),
        ("sieve", "664579\n"),
        ("strings", "100000\n"),
        ("tree", "10738090560\n"),
    ];

    for (bench, expected_stdout) in expected_exact {
        let script = benchmark_path(bench);
        let out_bin = tmp.path().join(format!("{bench}_rust_aot"));

        let (build_out, _build_dur) = build_rust_aot_cli(&script, &out_bin, &cache_dir, &[]);
        assert!(
            build_out.status.success(),
            "lust build --backend rust failed for '{bench}.lust':\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&build_out.stdout),
            String::from_utf8_lossy(&build_out.stderr)
        );
        assert!(out_bin.exists(), "built binary missing for '{bench}'");

        let run_out = Command::new(&out_bin)
            .output()
            .unwrap_or_else(|e| panic!("failed to execute {bench} binary: {e}"));
        assert!(
            run_out.status.success(),
            "Rust AOT binary for '{bench}' failed with {:?}:\nstderr: {}",
            run_out.status,
            String::from_utf8_lossy(&run_out.stderr)
        );
        let aot_stdout = String::from_utf8(run_out.stdout).expect("utf8 stdout");

        // Compare bit-exact against Cranelift JIT (full 10/10 benchmark scale)
        let jit_stdout = run_jit_stdout(&script);
        assert_eq!(
            aot_stdout, jit_stdout,
            "bit-exact stdout mismatch between Rust AOT and Cranelift JIT on '{bench}.lust'"
        );
        assert_eq!(
            aot_stdout, expected_stdout,
            "stdout mismatch against exact oracle on '{bench}.lust'"
        );
    }

    // Also verify cached rlib was NOT rebuilt during any of the 10 builds
    let final_mtime = fs::metadata(&rlib_first)
        .expect("rlib metadata final")
        .modified()
        .expect("rlib mtime final");
    assert_eq!(
        first_mtime, final_mtime,
        "cached liblust_aot_rt.rlib must be reused across all 10 benchmark builds"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Direct Bytecode VM (`--backend vm --jit=off`) bit-exact parity across
//         all 10 benchmark algorithms (full-scale on fast programs + VM-friendly scale).
// ---------------------------------------------------------------------------

#[test]
fn test_vm_vs_rust_aot_bit_exact_parity_all_10_benchmark_algorithms() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");

    // Scale down only the 10M loop counts so debug-mode Bytecode VM runs in milliseconds
    // while exercising every single benchmark's exact AST/MIR/codegen structure.
    let scaled_replacements: [(&str, Option<(&str, &str)>); 10] = [
        ("array", Some(("1000000", "2000"))),
        ("calls", Some(("10000000", "5000"))),
        ("fib", Some(("fib(30)", "fib(18)"))),
        ("fields", Some(("10000000", "5000"))),
        ("floatmath", Some(("10000000", "5000"))),
        ("methods", Some(("10000000", "5000"))),
        ("nested", Some(("3000", "60"))),
        ("sieve", Some(("10000000", "5000"))),
        ("strings", Some(("100000", "2000"))),
        ("tree", Some(("build(14, 1)", "build(7, 1)"))),
    ];

    for (bench, repl) in scaled_replacements {
        let mut src = fs::read_to_string(benchmark_path(bench)).expect("read benchmark");
        if let Some((from, to)) = repl {
            src = src.replace(from, to);
        }
        if bench == "tree" {
            src = src.replace("2000", "10").replace("1000", "100");
        }
        let script = tmp.path().join(format!("{bench}_vm_parity.lust"));
        fs::write(&script, &src).expect("write scaled script");

        let vm_stdout = run_vm_stdout(&script);
        let out_bin = tmp.path().join(format!("{bench}_vm_parity_bin"));
        let (build_out, _) = build_rust_aot_cli(&script, &out_bin, &cache_dir, &[]);
        assert!(
            build_out.status.success(),
            "build failed for scaled '{bench}': {}",
            String::from_utf8_lossy(&build_out.stderr)
        );
        let run_out = Command::new(&out_bin).output().expect("run aot bin");
        assert!(run_out.status.success());
        let aot_stdout = String::from_utf8(run_out.stdout).expect("utf8");

        assert_eq!(
            aot_stdout, vm_stdout,
            "bit-exact stdout mismatch between Rust AOT and Bytecode VM on '{bench}'"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3: `--emit-rs` Codegen Quality & Invariants Audit (Tiers 1-3, Hazard 9,
//         zero unsafe, zero unwrap, Infallible vs AotError, Arena, Vec<T, &LustBudgetAlloc>)
// ---------------------------------------------------------------------------

#[test]
fn test_emit_rs_tier_invariants_and_panic_free_safety() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");

    for bench in ALL_10_BENCHMARKS {
        let script = benchmark_path(bench);
        let rs_out = tmp.path().join(format!("{bench}.rs"));
        let (out, _) = build_rust_aot_cli(&script, &rs_out, &cache_dir, &["--emit-rs"]);
        assert!(
            out.status.success(),
            "--emit-rs failed for '{bench}': {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let code = fs::read_to_string(&rs_out).expect("read emitted .rs");

        // Invariant: Zero unsafe blocks and zero .unwrap() / panic!() (TASK_ALLOWS_UNSAFE=0, Hazard 9)
        assert!(
            !code.contains("unsafe"),
            "emitted Rust for '{bench}' must contain ZERO unsafe blocks"
        );
        assert!(
            !code.contains(".unwrap("),
            "emitted Rust for '{bench}' must contain ZERO .unwrap() calls (Hazard 9)"
        );
        assert!(
            !code.contains("panic!("),
            "emitted Rust for '{bench}' must contain ZERO panic!() calls (Hazard 9)"
        );

        match bench {
            "fib" => {
                assert!(
                    code.contains("Result<i64, core::convert::Infallible>"),
                    "fib() must have zero-cost Infallible signature:\n{code}"
                );
            }
            "nested" => {
                assert!(
                    code.contains("% 7_i64"),
                    "nested.lust must emit direct constant modulo `% 7_i64`:\n{code}"
                );
            }
            "strings" => {
                assert!(
                    code.contains("lust_aot_rt::string_append_int"),
                    "strings.lust must fuse `s = s .. tostring(i % 10)` into `string_append_int`:\n{code}"
                );
            }
            "sieve" => {
                assert!(
                    code.contains("alloc::vec::Vec<bool, &'a lust_aot_rt::LustBudgetAlloc>"),
                    "sieve.lust must emit 1-byte bool Vec<bool, &LustBudgetAlloc>:\n{code}"
                );
                assert!(
                    code.contains("if let Some(&__elem) = usize::try_from"),
                    "sieve.lust must fold IfArrayGetOk into `if let Some(&__elem)`:\n{code}"
                );
            }
            "array" => {
                assert!(
                    code.contains("alloc::vec::Vec<i64, &'a lust_aot_rt::LustBudgetAlloc>"),
                    "array.lust must emit Vec<i64, &LustBudgetAlloc>:\n{code}"
                );
                assert!(
                    code.contains("if let Some(&__elem) = usize::try_from"),
                    "array.lust must fold IfArrayGetOk:\n{code}"
                );
            }
            "tree" => {
                assert!(
                    code.contains("lust_aot_rt::LustChunkArena"),
                    "tree.lust must allocate recursive Node boxes in LustChunkArena:\n{code}"
                );
                assert!(
                    code.contains(".as_deref()"),
                    "tree.lust must borrow Option<Box<Node>> via .as_deref() without cloning:\n{code}"
                );
                assert!(
                    code.contains("Result<i64, core::convert::Infallible>"),
                    "tree.lust::sum must be Infallible:\n{code}"
                );
            }
            "fields" => {
                assert!(
                    code.contains("#[repr(C)]"),
                    "fields.lust C-flat struct Point must be #[repr(C)]:\n{code}"
                );
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Test 4: Runtime Error Parity in Child Processes (DivisionByZero, IndexOutOfBounds,
//         OutOfMemory via --max-memory, OutOfGas via --gas-metering) & Edge-Case Math
// ---------------------------------------------------------------------------

#[test]
fn test_rust_aot_runtime_errors_and_edge_case_math_parity() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");

    // 4A. Edge-case integer math parity against VM (i64::MIN / -1, i64::MIN % -1, floordiv, pow, shifts)
    let math_src = r#"
function check_div(a: int, b: int): int
    return a / b
end
function check_rem(a: int, b: int): int
    return a % b
end
function check_fdiv(a: int, b: int): int
    return a // b
end
local min_i: int = -9223372036854775807 - 1
println(check_div(min_i, -1))
println(check_rem(min_i, -1))
println(check_fdiv(-7, 2))
println(check_fdiv(7, -2))
println(check_fdiv(-7, -2))
println(2 ^ 10)
"#;
    let math_script = tmp.path().join("edge_math.lust");
    fs::write(&math_script, math_src).expect("write edge_math");
    let vm_out = run_vm_stdout(&math_script);
    let math_bin = tmp.path().join("edge_math_bin");
    let (build_out, _) = build_rust_aot_cli(&math_script, &math_bin, &cache_dir, &[]);
    assert!(
        build_out.status.success(),
        "edge_math build failed: {}",
        String::from_utf8_lossy(&build_out.stderr)
    );
    let run_out = Command::new(&math_bin).output().expect("run edge_math");
    assert!(run_out.status.success());
    assert_eq!(String::from_utf8(run_out.stdout).unwrap(), vm_out);

    // 4B. Division by zero returns non-zero exit + structured error (never aborts/panics)
    let div0_script = tmp.path().join("div0.lust");
    fs::write(
        &div0_script,
        "function f(a: int, b: int): int\n    return a / b\nend\nprintln(f(10, 0))\n",
    )
    .expect("write div0");
    let div0_bin = tmp.path().join("div0_bin");
    let (b_div0, _) = build_rust_aot_cli(&div0_script, &div0_bin, &cache_dir, &[]);
    assert!(b_div0.status.success());
    let r_div0 = Command::new(&div0_bin).output().expect("run div0");
    assert!(!r_div0.status.success());
    assert_eq!(r_div0.status.code(), Some(1));
    let div0_err = String::from_utf8_lossy(&r_div0.stderr);
    assert!(
        div0_err.contains("Division by zero"),
        "expected Division by zero, got: {div0_err}"
    );

    // 4C. Memory limit (--max-memory 256) trips OutOfMemory cleanly via LustBudgetAlloc
    let oom_script = tmp.path().join("oom.lust");
    fs::write(
        &oom_script,
        "local a: Array<int> = []\nlocal i: int = 0\nwhile i < 10000 do\n    array.push(a, i)\n    i = i + 1\nend\nprintln(array.len(a))\n",
    )
    .expect("write oom");
    let oom_bin = tmp.path().join("oom_bin");
    let (b_oom, _) =
        build_rust_aot_cli(&oom_script, &oom_bin, &cache_dir, &["--max-memory", "256"]);
    assert!(
        b_oom.status.success(),
        "oom build failed: {}",
        String::from_utf8_lossy(&b_oom.stderr)
    );
    let r_oom = Command::new(&oom_bin).output().expect("run oom");
    assert!(!r_oom.status.success());
    assert_eq!(r_oom.status.code(), Some(1));
    let oom_err = String::from_utf8_lossy(&r_oom.stderr);
    assert!(
        oom_err.contains("Out of memory"),
        "expected Out of memory error, got: {oom_err}"
    );

    // 4D. Gas limit (--gas-metering exact --gas-limit 50) trips OutOfGas cleanly
    let oog_script = tmp.path().join("oog.lust");
    fs::write(
        &oog_script,
        "local i: int = 0\nwhile i < 1000 do\n    i = i + 1\nend\nprintln(i)\n",
    )
    .expect("write oog");
    let oog_bin = tmp.path().join("oog_bin");
    let (b_oog, _) = build_rust_aot_cli(
        &oog_script,
        &oog_bin,
        &cache_dir,
        &["--gas-metering", "exact", "--gas-limit", "50"],
    );
    assert!(
        b_oog.status.success(),
        "oog build failed: {}",
        String::from_utf8_lossy(&b_oog.stderr)
    );
    let r_oog = Command::new(&oog_bin).output().expect("run oog");
    assert!(!r_oog.status.success());
    assert_eq!(r_oog.status.code(), Some(1));
    let oog_err = String::from_utf8_lossy(&r_oog.stderr);
    assert!(
        oog_err.contains("Out of gas"),
        "expected Out of gas error, got: {oog_err}"
    );
}

// ---------------------------------------------------------------------------
// Test 5: `lust run --backend rust` CLI integration & AotCompiler Rust API
// ---------------------------------------------------------------------------

#[test]
fn test_lust_run_backend_rust_and_api_compile_source() {
    let tmp = tempdir().expect("tempdir");
    let script = benchmark_path("fib");

    let run_out = Command::new(lust_bin())
        .arg("run")
        .arg(&script)
        .arg("--backend")
        .arg("rust")
        .output()
        .expect("lust run --backend rust");
    assert!(
        run_out.status.success(),
        "lust run --backend rust failed: {}",
        String::from_utf8_lossy(&run_out.stderr)
    );
    assert_eq!(String::from_utf8(run_out.stdout).unwrap(), "832040\n");

    // Direct Rust API compile_source_rust with non-inlined recursive struct & read-only array param
    let api_bin = tmp.path().join("api_test_bin");
    let opts = RustCompileOptions {
        cache_dir: Some(tmp.path().join("api_cache")),
        ..RustCompileOptions::default()
    };
    let compiled = AotCompiler::compile_source_rust(
        r#"
function sum_slice(a: Array<int>): int
    local s: int = 0
    local i: int = 0
    while i < array.len(a) do
        if a[i] is Ok(v) then
            s = s + v
        end
        i = i + 1
    end
    return s
end
local arr: Array<int> = [10, 20, 30, 40]
println(sum_slice(arr))
"#,
        &api_bin,
        &opts,
    )
    .expect("AotCompiler::compile_source_rust");
    let api_run = Command::new(&compiled).output().expect("run api_test_bin");
    assert!(api_run.status.success());
    assert_eq!(String::from_utf8(api_run.stdout).unwrap(), "100\n");
}

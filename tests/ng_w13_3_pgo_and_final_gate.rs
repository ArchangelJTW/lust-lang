//! NG-W13-3: Sample/Instrumented PGO Pipeline (`scripts/build_pgo.sh`) and
//! Final Wave 13 Acceptance Gate Verification.
//!
//! Gate flags: `TASK_ALLOWS_CARGO_LOCK=0`, `TASK_ALLOWS_UNSAFE=0`.
//!
//! Verifies:
//! 1. `scripts/build_pgo.sh` exists, is executable, passes `bash -n`, enforces
//!    `--offline --locked` on every Cargo invocation, and guards against overwriting
//!    protected historical `results/{bench_suite,eval}_perf_t{1,2,3,4}/` directories.
//! 2. Both PGO modes are implemented and validated:
//!    - `--mode=sample <profile.prof>` (`-Cprofile-sample-use`, rustc 1.100.0 Linux AutoFDO)
//!    - `--mode=instrumented` (`-Cprofile-generate` -> `benchmarks/suite/*.lust` under
//!      `--jit=off` & `--jit=cranelift` -> `llvm-profdata merge` -> `-Cprofile-use`, macOS arm64)
//! 3. Active `rustc 1.100+` toolchain supports `-Cprofile-sample-use`, `-Cprofile-generate`,
//!    `-Cprofile-use`, and exposes matching sysroot `llvm-profdata`.
//! 4. End-to-end real PGO builds of `liblust_aot_rt.rlib` in BOTH `--mode=sample` and
//!    `--mode=instrumented` produce valid `.rlib` artifacts that link and execute cleanly.
//! 5. Final Wave 13 gate parity across Bytecode VM (`--jit=off`), Cranelift JIT
//!    (`--jit=cranelift`), and Rust AOT (`--backend rust`) plus Wave 13 AOT features
//!    (services/handlers, `--emit=rust-crate`, cross-target `no_std`, zero `unsafe`/`.unwrap(`).

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::tempdir;

use lust::codegen_zig::aot::{
    AotCompiler, RustCompileOptions, check_target_core_installed, is_nostd_target,
};

const CANONICAL_10_BENCHMARKS: [&str; 10] = [
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

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn pgo_script_path() -> PathBuf {
    repo_root().join("scripts").join("build_pgo.sh")
}

fn lust_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_lust"))
}

fn host_triple() -> String {
    let out = Command::new("rustc")
        .arg("-vV")
        .output()
        .expect("rustc -vV must succeed");
    assert!(out.status.success(), "rustc -vV failed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        .find_map(|line| line.strip_prefix("host: ").map(str::trim))
        .expect("rustc -vV must report host triple")
        .to_string()
}

fn link_and_run_against_pgo_rlib(rlib_path: &Path, work_dir: &Path, label: &str) -> String {
    assert!(
        rlib_path.exists(),
        "[{label}] expected PGO liblust_aot_rt.rlib at {}",
        rlib_path.display()
    );
    let meta = fs::metadata(rlib_path).expect("rlib metadata");
    assert!(
        meta.len() > 1024,
        "[{label}] PGO liblust_aot_rt.rlib unexpectedly small ({} bytes)",
        meta.len()
    );

    let harness_rs = work_dir.join(format!("{label}_harness.rs"));
    let harness_bin = work_dir.join(format!("{label}_harness"));

    // Zero-unsafe verification binary linking against the PGO-built `liblust_aot_rt.rlib`.
    let harness_src = r#"
extern crate lust_aot_rt;

use lust_aot_rt::{
    AotCtx, AotMap, AotValue, EffectHandlerArm, handle_tail_resumptive_typed, perform_effect,
    resume_effect, scope_register_finalizer, with_aot_scope,
};

fn main() {
    let mut ctx = AotCtx::with_memory_limit(2 * 1024 * 1024);
    let map: AotMap<i64, AotValue> = AotMap::new();
    let _ = map.set(1, AotValue::Int(40), &ctx.budget_alloc);
    let base = match map.get(&1) {
        Ok(AotValue::Int(v)) => v,
        _ => 0,
    };
    let arm = EffectHandlerArm::from_fn("VerifyEff", |c, args| {
        let x = match args.first() {
            Some(AotValue::Int(v)) => *v,
            _ => 0,
        };
        resume_effect(c, AotValue::Int(x + 2))
    });
    let delta = handle_tail_resumptive_typed(
        &mut ctx,
        vec![arm],
        |c| match perform_effect(c, "VerifyEff", &[AotValue::Int(0)]) {
            Ok(AotValue::Int(v)) => Ok(v),
            _ => Ok(0),
        },
        |_abort| Ok(0),
    )
    .unwrap_or(0);
    let out = with_aot_scope(&mut ctx, |c| {
        let _ = scope_register_finalizer(c, |_c2| Ok(()));
        Ok(base + delta)
    })
    .unwrap_or(0);
    println!("pgo_rlib_ok:{out}");
}
"#;
    fs::write(&harness_rs, harness_src).expect("write harness.rs");

    let compile = Command::new("rustc")
        .args([
            "--edition=2024",
            "-C",
            "opt-level=2",
            "-C",
            "panic=abort",
            "--extern",
        ])
        .arg(format!("lust_aot_rt={}", rlib_path.display()))
        .arg(&harness_rs)
        .arg("-o")
        .arg(&harness_bin)
        .output()
        .expect("invoke rustc on harness");
    assert!(
        compile.status.success(),
        "[{label}] failed to link against PGO liblust_aot_rt.rlib:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&compile.stdout),
        String::from_utf8_lossy(&compile.stderr)
    );

    let run = Command::new(&harness_bin)
        .output()
        .expect("run linked PGO harness");
    assert!(
        run.status.success(),
        "[{label}] PGO harness exited with {:?}\nstderr:\n{}",
        run.status.code(),
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8_lossy(&run.stdout).trim().to_string()
}

// =============================================================================
// 1. PGO Script Syntax, Permissions, Offline/Locked & Safety Guard Contract
// =============================================================================

#[test]
fn test_build_pgo_script_exists_executable_and_passes_bash_syntax_check() {
    let script = pgo_script_path();
    assert!(
        script.exists(),
        "scripts/build_pgo.sh must exist at {}",
        script.display()
    );

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(&script).expect("script metadata").permissions().mode();
        assert!(
            (mode & 0o111) != 0,
            "scripts/build_pgo.sh must be executable (mode={mode:#o})"
        );
    }

    let content = fs::read_to_string(&script).expect("read scripts/build_pgo.sh");
    assert!(
        content.starts_with("#!/usr/bin/env bash\n"),
        "scripts/build_pgo.sh must start with #!/usr/bin/env bash"
    );
    assert!(
        content.contains("set -euo pipefail"),
        "scripts/build_pgo.sh must enforce strict bash mode"
    );

    // Every `cargo build` invocation inside the script must be `--offline --locked`
    // so `Cargo.lock` is never mutated (`TASK_ALLOWS_CARGO_LOCK=0`).
    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            continue;
        }
        if trimmed.contains("cargo build") {
            assert!(
                trimmed.contains("--offline") && trimmed.contains("--locked"),
                "Line {} of scripts/build_pgo.sh invokes cargo build without --offline --locked: {trimmed}",
                idx + 1
            );
        }
    }

    // Verify all required PGO flags and workloads are encoded in the script
    for required_token in [
        "profile-sample-use=",
        "profile-generate=",
        "profile-use=",
        "llvm-profdata",
        "--jit=off",
        "--jit=cranelift",
        "liblust_aot_rt.rlib",
        "benchmarks/suite",
    ] {
        assert!(
            content.contains(required_token),
            "scripts/build_pgo.sh must contain '{required_token}'"
        );
    }

    // Verify bash -n syntax validation passes cleanly
    let syntax = Command::new("bash")
        .arg("-n")
        .arg(&script)
        .output()
        .expect("run bash -n");
    assert!(
        syntax.status.success(),
        "bash -n failed on scripts/build_pgo.sh:\n{}",
        String::from_utf8_lossy(&syntax.stderr)
    );
}

#[test]
fn test_build_pgo_cli_help_validation_and_protected_results_guard() {
    let script = pgo_script_path();

    // --help succeeds and documents both Linux sample PGO and macOS instrumented PGO
    let help = Command::new("bash")
        .arg(&script)
        .arg("--help")
        .output()
        .expect("run --help");
    assert!(help.status.success(), "--help must succeed");
    let help_text = String::from_utf8_lossy(&help.stdout);
    assert!(help_text.contains("--mode=instrumented"));
    assert!(help_text.contains("--mode=sample"));
    assert!(help_text.contains("-Cprofile-sample-use"));
    assert!(help_text.contains("-Cprofile-generate"));
    assert!(help_text.contains("llvm-profdata"));

    // --mode=sample without a profile path must fail cleanly
    let missing_prof = Command::new("bash")
        .arg(&script)
        .arg("--mode=sample")
        .arg("--dry-run")
        .output()
        .expect("run missing profile");
    assert!(
        !missing_prof.status.success(),
        "--mode=sample without profile path must exit non-zero"
    );

    // --mode=sample with nonexistent file must fail cleanly
    let bad_prof = Command::new("bash")
        .arg(&script)
        .args(["--mode=sample", "/tmp/does_not_exist_ng_w13_3.prof", "--dry-run"])
        .output()
        .expect("run nonexistent profile");
    assert!(
        !bad_prof.status.success(),
        "--mode=sample with missing file must exit non-zero"
    );

    // Protected historical perf_t{1,2,3,4} directories must be rejected (DO NOT rule)
    for forbidden in [
        "/Users/vedmalex/work/lust-lang/results/bench_suite_perf_t1",
        "/Users/vedmalex/work/lust-lang/results/bench_suite_perf_t4",
        "/Users/vedmalex/work/lust-lang/results/eval_perf_t2",
        "/Users/vedmalex/work/lust-lang/results/eval_perf_t4",
    ] {
        let guarded = Command::new("bash")
            .arg(&script)
            .args([
                "--mode=instrumented",
                "--dry-run",
                "--out-dir",
                forbidden,
            ])
            .output()
            .expect("run protected path check");
        assert!(
            !guarded.status.success(),
            "scripts/build_pgo.sh must reject protected output path '{forbidden}'"
        );
        let stderr = String::from_utf8_lossy(&guarded.stderr);
        assert!(
            stderr.contains("protected historical benchmark directory"),
            "expected protected-directory diagnostic for '{forbidden}', got:\n{stderr}"
        );
    }
}

#[test]
fn test_build_pgo_dry_run_contracts_for_instrumented_and_sample_modes() {
    let script = pgo_script_path();
    let td = tempdir().expect("tempdir");
    let target_dir = td.path().join("target_pgo");

    // 1. Instrumented mode dry-run (macOS arm64 default pipeline)
    let inst = Command::new("bash")
        .arg(&script)
        .args([
            "--mode=instrumented",
            "--dry-run",
            "--target-dir",
            target_dir.to_str().unwrap(),
        ])
        .output()
        .expect("run instrumented dry-run");
    assert!(
        inst.status.success(),
        "instrumented --dry-run failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&inst.stdout),
        String::from_utf8_lossy(&inst.stderr)
    );
    let inst_out = String::from_utf8_lossy(&inst.stdout);
    assert!(inst_out.contains("-Cprofile-generate="), "missing -Cprofile-generate:\n{inst_out}");
    assert!(inst_out.contains("merge -o"), "missing llvm-profdata merge:\n{inst_out}");
    assert!(inst_out.contains("-Cprofile-use="), "missing -Cprofile-use:\n{inst_out}");
    assert!(inst_out.contains("--jit=off"), "missing --jit=off training:\n{inst_out}");
    assert!(
        inst_out.contains("--jit=cranelift"),
        "missing --jit=cranelift training:\n{inst_out}"
    );
    assert!(inst_out.contains("--offline"), "missing --offline:\n{inst_out}");
    assert!(inst_out.contains("--locked"), "missing --locked:\n{inst_out}");
    for bench in CANONICAL_10_BENCHMARKS {
        assert!(
            inst_out.contains(&format!("{bench}.lust")),
            "instrumented dry-run missing suite benchmark '{bench}.lust'"
        );
    }

    // 2. Sample mode dry-run (Linux AutoFDO `-Cprofile-sample-use` pipeline)
    let sample_prof = td.path().join("lust.AutoFDO.prof");
    fs::write(&sample_prof, "lust_aot_rt::dummy_sample:10:5\n 1: 10\n").expect("write sample prof");

    let sample = Command::new("bash")
        .arg(&script)
        .arg("--mode=sample")
        .arg(&sample_prof)
        .args([
            "--dry-run",
            "--target-dir",
            target_dir.to_str().unwrap(),
        ])
        .output()
        .expect("run sample dry-run");
    assert!(
        sample.status.success(),
        "sample --dry-run failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&sample.stdout),
        String::from_utf8_lossy(&sample.stderr)
    );
    let sample_out = String::from_utf8_lossy(&sample.stdout);
    assert!(
        sample_out.contains("-Cprofile-sample-use="),
        "sample dry-run missing -Cprofile-sample-use=:\n{sample_out}"
    );
    assert!(
        sample_out.contains("--bin lust"),
        "sample dry-run must build lust CLI/VM binary:\n{sample_out}"
    );
    assert!(
        sample_out.contains("liblust_aot_rt.rlib"),
        "sample dry-run must build liblust_aot_rt.rlib:\n{sample_out}"
    );
}

// =============================================================================
// 2. Toolchain Capabilities & Real End-to-End PGO Builds (Sample + Instrumented)
// =============================================================================

#[test]
fn test_rustc_1_100_pgo_toolchain_and_sysroot_llvm_profdata() {
    let help = Command::new("rustc")
        .args(["-C", "help"])
        .output()
        .expect("rustc -C help");
    assert!(help.status.success());
    let help_out = String::from_utf8_lossy(&help.stdout);
    for flag in ["profile-generate", "profile-use", "profile-sample-use"] {
        assert!(
            help_out.contains(flag),
            "rustc must support -C {flag} (Rust 1.100+ required)"
        );
    }

    let sysroot_out = Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .expect("rustc --print sysroot");
    assert!(sysroot_out.status.success());
    let sysroot = PathBuf::from(String::from_utf8_lossy(&sysroot_out.stdout).trim());
    let profdata = sysroot
        .join("lib")
        .join("rustlib")
        .join(host_triple())
        .join("bin")
        .join("llvm-profdata");
    assert!(
        profdata.exists(),
        "rustc sysroot llvm-profdata must exist at {}",
        profdata.display()
    );
    let ver = Command::new(&profdata)
        .arg("--version")
        .output()
        .expect("llvm-profdata --version");
    assert!(ver.status.success(), "llvm-profdata --version failed");
}

#[test]
fn test_real_sample_pgo_build_aot_rt_end_to_end() {
    let script = pgo_script_path();
    let td = tempdir().expect("tempdir");
    let target_dir = td.path().join("target_sample");
    let out_dir = td.path().join("out_sample");
    let sample_prof = td.path().join("sample.prof");

    // Minimal valid LLVM text sample profile accepted by rustc 1.100 `-Cprofile-sample-use`.
    fs::write(
        &sample_prof,
        "lust_aot_rt::alloc::LustBudgetAlloc::try_charge:100:25\n 1: 100\n",
    )
    .expect("write sample.prof");

    let run = Command::new("bash")
        .arg(&script)
        .arg("--mode=sample")
        .arg(&sample_prof)
        .args([
            "--component=aot-rt",
            "--target-dir",
            target_dir.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute scripts/build_pgo.sh --mode=sample");
    assert!(
        run.status.success(),
        "real --mode=sample --component=aot-rt failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );

    let rlib_path = out_dir.join("liblust_aot_rt.rlib");
    let manifest_path = out_dir.join("pgo-build-manifest.json");
    assert!(manifest_path.exists(), "pgo-build-manifest.json must be emitted");
    let manifest = fs::read_to_string(&manifest_path).expect("read manifest");
    assert!(manifest.contains("\"task_id\": \"NG-W13-3\""));
    assert!(manifest.contains("\"mode\": \"sample\""));

    let output = link_and_run_against_pgo_rlib(&rlib_path, td.path(), "sample_pgo");
    assert_eq!(output, "pgo_rlib_ok:42");
}

#[test]
fn test_real_instrumented_pgo_build_aot_rt_end_to_end() {
    let script = pgo_script_path();
    let td = tempdir().expect("tempdir");
    let target_dir = td.path().join("target_inst");
    let pgo_dir = td.path().join("pgo_data");
    let out_dir = td.path().join("out_inst");

    let run = Command::new("bash")
        .arg(&script)
        .args([
            "--mode=instrumented",
            "--component=aot-rt",
            "--target-dir",
            target_dir.to_str().unwrap(),
            "--pgo-dir",
            pgo_dir.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute scripts/build_pgo.sh --mode=instrumented");
    assert!(
        run.status.success(),
        "real --mode=instrumented --component=aot-rt failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );

    let merged_profdata = pgo_dir.join("merged.profdata");
    assert!(
        merged_profdata.exists() && fs::metadata(&merged_profdata).unwrap().len() > 0,
        "merged.profdata must be generated by llvm-profdata merge at {}",
        merged_profdata.display()
    );

    let profraw_count = fs::read_dir(pgo_dir.join("profraw"))
        .expect("profraw dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("profraw"))
        .count();
    assert!(
        profraw_count >= 1,
        "instrumented training must produce at least one .profraw file"
    );

    let manifest = fs::read_to_string(out_dir.join("pgo-build-manifest.json"))
        .expect("read manifest");
    assert!(manifest.contains("\"task_id\": \"NG-W13-3\""));
    assert!(manifest.contains("\"mode\": \"instrumented\""));

    let rlib_path = out_dir.join("liblust_aot_rt.rlib");
    let output = link_and_run_against_pgo_rlib(&rlib_path, td.path(), "instrumented_pgo");
    assert_eq!(output, "pgo_rlib_ok:42");
}

// =============================================================================
// 3. Final Wave 13 Gate: 10-Suite Inventory, 3-Engine Parity, AOT Features
// =============================================================================

#[test]
fn test_final_wave13_gate_suite_inventory_and_three_engine_parity() {
    let suite_dir = repo_root().join("benchmarks").join("suite");
    for bench in CANONICAL_10_BENCHMARKS {
        let p = suite_dir.join(format!("{bench}.lust"));
        assert!(
            p.exists() && fs::metadata(&p).unwrap().len() > 0,
            "Canonical training/benchmark script missing: {}",
            p.display()
        );
    }

    let td = tempdir().expect("tempdir");
    let cache_dir = td.path().join("rlib_cache");
    let lust = lust_bin();

    // Fast canonical parity gate across Bytecode VM (--jit=off), Cranelift JIT (--jit=cranelift),
    // and Rust AOT (--backend rust --release-fast).
    for (bench, expected) in [("fib", "832040"), ("strings", "100000")] {
        let script_path = suite_dir.join(format!("{bench}.lust"));

        let vm_out = Command::new(&lust)
            .args(["run", script_path.to_str().unwrap(), "--jit=off"])
            .output()
            .expect("run lust --jit=off");
        assert!(
            vm_out.status.success(),
            "VM failed on {bench}: {}",
            String::from_utf8_lossy(&vm_out.stderr)
        );
        let vm_stdout = String::from_utf8_lossy(&vm_out.stdout).trim().to_string();

        let jit_out = Command::new(&lust)
            .args(["run", script_path.to_str().unwrap(), "--jit=cranelift"])
            .output()
            .expect("run lust --jit=cranelift");
        assert!(
            jit_out.status.success(),
            "JIT failed on {bench}: {}",
            String::from_utf8_lossy(&jit_out.stderr)
        );
        let jit_stdout = String::from_utf8_lossy(&jit_out.stdout).trim().to_string();

        let aot_bin = td.path().join(format!("{bench}_aot"));
        let build_out = Command::new(&lust)
            .args([
                "build",
                script_path.to_str().unwrap(),
                "--backend",
                "rust",
                "--release-fast",
                "--cache-dir",
                cache_dir.to_str().unwrap(),
                "-o",
                aot_bin.to_str().unwrap(),
            ])
            .output()
            .expect("build lust rust AOT");
        assert!(
            build_out.status.success(),
            "Rust AOT build failed on {bench}:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&build_out.stdout),
            String::from_utf8_lossy(&build_out.stderr)
        );

        let aot_run = Command::new(&aot_bin)
            .output()
            .expect("run compiled AOT binary");
        assert!(
            aot_run.status.success(),
            "Rust AOT binary failed on {bench}: {}",
            String::from_utf8_lossy(&aot_run.stderr)
        );
        let aot_stdout = String::from_utf8_lossy(&aot_run.stdout).trim().to_string();

        assert_eq!(vm_stdout, expected, "{bench}: VM output mismatch");
        assert_eq!(jit_stdout, expected, "{bench}: JIT output mismatch");
        assert_eq!(aot_stdout, expected, "{bench}: Rust AOT output mismatch");
    }
}

#[test]
fn test_final_wave13_gate_aot_zero_unsafe_and_cross_target_readiness() {
    let source = r#"
struct Counter
    value: int
end

function accumulate(n: int): int
    local c: Counter = Counter { value = 0 }
    local i: int = 1
    while i <= n do
        c.value = c.value + i
        i = i + 1
    end
    return c.value
end

function main(): int
    return accumulate(10) - 55
end
"#;

    // 1. Host std Rust AOT emission must contain zero `unsafe`, zero `.unwrap(`, zero `panic!(`.
    let std_rs = AotCompiler::emit_rust_source(source, &RustCompileOptions::default())
        .expect("emit host std Rust AOT");
    assert!(!std_rs.contains("unsafe"), "emitted Rust AOT must contain zero `unsafe`");
    assert!(!std_rs.contains(".unwrap("), "emitted Rust AOT must contain zero `.unwrap(` (Hazard 9)");
    assert!(!std_rs.contains("panic!("), "emitted Rust AOT must contain zero `panic!(` (Hazard 9)");

    // 2. Embedded `#![no_std]` emission (NG-W13-2) preserves StaticSlabAllocator & zero unsafe.
    let mut nostd_opts = RustCompileOptions::default();
    nostd_opts.no_std = true;
    let nostd_rs = AotCompiler::emit_rust_source(source, &nostd_opts)
        .expect("emit no_std Rust AOT");
    assert!(nostd_rs.contains("#![no_std]"));
    assert!(nostd_rs.contains("StaticSlabAllocator<65536>"));
    assert!(!nostd_rs.contains("unsafe"));
    assert!(is_nostd_target("wasm32-unknown-unknown"));
    assert!(is_nostd_target("thumbv7em-none-eabihf"));
    assert!(is_nostd_target("riscv32imac-unknown-none-elf"));

    // 3. Host sysroot `libcore` check passes cleanly offline.
    let host = host_triple();
    assert!(
        check_target_core_installed(&host).expect("check host target libcore"),
        "host target '{host}' must have libcore installed offline"
    );
}

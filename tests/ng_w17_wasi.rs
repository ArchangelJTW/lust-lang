//! Live WASI Preview 1 & Preview 2 execution parity, capability/budget controls,
//! and root-crate vs `lust-aot-rt` WASI constraint verification (NG-W17-8).

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use lust::codegen_zig::aot::compiler::RustLinkOptions;
use lust::codegen_zig::aot::{AotCompiler, RustAotBuildOutcome, RustCompileOptions};

const TARGET_DIR: &str = "/tmp/lust_eval_target/w17-8";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn lust_bin() -> &'static str {
    env!("CARGO_BIN_EXE_lust")
}

fn run_wasi_runner(args: &[&str]) -> Output {
    let runner_bin = Path::new(TARGET_DIR).join("debug/lust-wasi-runner");
    Command::new(runner_bin)
        .args(args)
        .output()
        .expect("invoke lust-wasi-runner")
}

fn build_wasi_runner() {
    let runner = Command::new("cargo")
        .current_dir(root())
        .args(["build", "--locked", "-p", "lust-wasi-runner", "--target-dir", TARGET_DIR])
        .output()
        .expect("build WASI runner");
    assert!(
        runner.status.success(),
        "runner build failed: {}",
        String::from_utf8_lossy(&runner.stderr)
    );
}

#[test]
fn wasi_aot_stdout_and_exit_match_native() {
    build_wasi_runner();
    let temp = tempfile::tempdir().unwrap();
    let targets = ["wasm32-wasip1", "wasm32-wasip2"];
    for target in targets {
        for fixture in ["success", "error"] {
            let source = root()
                .join("tests/fixtures/w17_wasi")
                .join(format!("{fixture}.rs"));
            let native = temp.path().join(format!("{fixture}-native"));
            let wasm = temp.path().join(format!("{fixture}-{target}.wasm"));
            let built = Command::new("rustc")
                .args([
                    "--edition=2024",
                    "-C",
                    "opt-level=2",
                    "-C",
                    "panic=abort",
                    source.to_str().unwrap(),
                    "-o",
                ])
                .arg(&native)
                .output()
                .unwrap();
            assert!(
                built.status.success(),
                "native build: {}",
                String::from_utf8_lossy(&built.stderr)
            );
            let wasm_build = Command::new("rustc")
                .args([
                    "--edition=2024",
                    "-C",
                    "opt-level=2",
                    "-C",
                    "panic=abort",
                    "--target",
                    target,
                    source.to_str().unwrap(),
                    "-o",
                ])
                .arg(&wasm)
                .output()
                .unwrap();
            assert!(
                wasm_build.status.success(),
                "{target} build: {}",
                String::from_utf8_lossy(&wasm_build.stderr)
            );
            let expected = Command::new(&native).output().unwrap();
            let actual = run_wasi_runner(&[wasm.to_str().unwrap()]);
            assert_eq!(
                actual.stdout,
                expected.stdout,
                "stdout mismatch: {target}/{fixture}; runner stderr: {}",
                String::from_utf8_lossy(&actual.stderr)
            );
            assert_eq!(
                actual.status.code(),
                expected.status.code(),
                "exit mismatch: {target}/{fixture}; runner stderr: {}",
                String::from_utf8_lossy(&actual.stderr)
            );
        }
    }
}

fn compile_lust_source_for_target(
    source: &str,
    out_path: &Path,
    target_triple: Option<&str>,
    cache_dir: &Path,
) -> PathBuf {
    let mut opts = RustCompileOptions::default();
    opts.aot.target_triple = target_triple.map(str::to_string);
    opts.cache_dir = Some(cache_dir.to_path_buf());
    let links = RustLinkOptions::default();
    match AotCompiler::compile_source_rust_cross(source, out_path, &opts, &links)
        .unwrap_or_else(|e| panic!("AOT compile failed for target={target_triple:?}: {e}"))
    {
        RustAotBuildOutcome::Compiled { artifact_path, .. } => artifact_path,
        RustAotBuildOutcome::TargetStdlibNotInstalledOffline { target, .. } => {
            panic!("expected installed offline target '{target}' for WASI parity test")
        }
    }
}

#[test]
fn wasi_real_lust_aot_parity_against_native_and_vm() {
    build_wasi_runner();
    let temp = tempfile::tempdir().unwrap();
    let cache_dir = temp.path().join("aot-cache");

    let cases = [
        ("compute_ok", "compute_ok.lust", 0),
        ("error_exit", "error_exit.lust", 1),
    ];

    for (name, file_name, expected_exit) in cases {
        let lust_path = root().join("tests/fixtures/w17_wasi").join(file_name);
        let source = fs::read_to_string(&lust_path).expect("read .lust fixture");

        // 1. Execute via Lust VM (`lust run`)
        let vm_out = Command::new(lust_bin())
            .arg("run")
            .arg(&lust_path)
            .output()
            .expect("run lust VM");
        assert_eq!(
            vm_out.status.code(),
            Some(expected_exit),
            "VM exit code mismatch for {name}; stderr: {}",
            String::from_utf8_lossy(&vm_out.stderr)
        );

        // 2. Compile Lust -> Rust AOT (`lust-aot-rt`) for native host
        let native_bin = temp.path().join(format!("{name}-native"));
        let native_artifact = compile_lust_source_for_target(&source, &native_bin, None, &cache_dir);
        let native_out = Command::new(&native_artifact)
            .output()
            .expect("run native AOT binary");
        assert_eq!(
            native_out.status.code(),
            Some(expected_exit),
            "Native AOT exit code mismatch for {name}; stderr: {}",
            String::from_utf8_lossy(&native_out.stderr)
        );
        assert_eq!(
            native_out.stdout, vm_out.stdout,
            "Native AOT vs VM stdout mismatch for {name}"
        );

        // 3. Compile Lust -> Rust AOT (`lust-aot-rt`) for wasm32-wasip1 & wasm32-wasip2
        for target in ["wasm32-wasip1", "wasm32-wasip2"] {
            let wasm_path = temp.path().join(format!("{name}-{target}.wasm"));
            let wasm_artifact =
                compile_lust_source_for_target(&source, &wasm_path, Some(target), &cache_dir);
            let wasm_out = run_wasi_runner(&[wasm_artifact.to_str().unwrap()]);
            assert_eq!(
                wasm_out.status.code(),
                Some(expected_exit),
                "{target} exit code mismatch for {name}; runner stderr: {}",
                String::from_utf8_lossy(&wasm_out.stderr)
            );
            assert_eq!(
                wasm_out.stdout,
                native_out.stdout,
                "{target} vs Native AOT stdout mismatch for {name}; runner stderr: {}",
                String::from_utf8_lossy(&wasm_out.stderr)
            );
            assert_eq!(
                wasm_out.stdout,
                vm_out.stdout,
                "{target} vs VM stdout mismatch for {name}"
            );
        }
    }
}

#[test]
fn wasi_runner_capability_and_budget_controls() {
    build_wasi_runner();
    let temp = tempfile::tempdir().unwrap();
    let sandbox_dir = temp.path().join("sandbox");
    fs::create_dir_all(&sandbox_dir).unwrap();
    fs::write(sandbox_dir.join("hello.txt"), "wasi-sandbox-ok").unwrap();

    let probe_src = root().join("tests/fixtures/w17_wasi/cap_probe.rs");

    for target in ["wasm32-wasip1", "wasm32-wasip2"] {
        let wasm = temp.path().join(format!("cap_probe-{target}.wasm"));
        let built = Command::new("rustc")
            .args([
                "--edition=2024",
                "-C",
                "opt-level=2",
                "-C",
                "panic=abort",
                "--target",
                target,
                probe_src.to_str().unwrap(),
                "-o",
            ])
            .arg(&wasm)
            .output()
            .unwrap();
        assert!(
            built.status.success(),
            "cap_probe build failed for {target}: {}",
            String::from_utf8_lossy(&built.stderr)
        );
        let wasm_str = wasm.to_str().unwrap();

        // 1. Default isolation: host PATH and host /etc/hosts are NOT exposed; no preopened dir.
        let iso = run_wasi_runner(&[wasm_str]);
        assert!(iso.status.success(), "{target} iso failed: {}", String::from_utf8_lossy(&iso.stderr));
        assert_eq!(
            String::from_utf8_lossy(&iso.stdout).trim(),
            "host_path=false;injected=MISSING;fs_ok=false;host_leak=false",
            "{target} default isolation failed"
        );

        // 2. Explicit --env and --dir-ro grant scoped access only.
        let dir_spec = format!("{}::/sandbox", sandbox_dir.display());
        let scoped = run_wasi_runner(&[
            "--env",
            "LUST_WASI_TEST_ENV=scoped42",
            "--dir-ro",
            &dir_spec,
            wasm_str,
        ]);
        assert!(
            scoped.status.success(),
            "{target} scoped failed: {}",
            String::from_utf8_lossy(&scoped.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&scoped.stdout).trim(),
            "host_path=false;injected=scoped42;fs_ok=true;host_leak=false",
            "{target} scoped env/dir-ro failed"
        );

        // 3. --deny-stdio suppresses guest stdout.
        let denied = run_wasi_runner(&["--deny-stdio", wasm_str, "--", "stdio"]);
        assert!(denied.status.success());
        assert!(
            denied.stdout.is_empty(),
            "{target} --deny-stdio must suppress stdout, got: {:?}",
            String::from_utf8_lossy(&denied.stdout)
        );

        // 4. --fuel budget traps infinite loop deterministically.
        let fuel_out = run_wasi_runner(&["--fuel", "5000", wasm_str, "--", "spin"]);
        assert!(!fuel_out.status.success(), "{target} --fuel must trap spin loop");
        let fuel_err = String::from_utf8_lossy(&fuel_out.stderr);
        assert!(
            fuel_err.contains("trapped") || fuel_err.contains("fuel"),
            "{target} --fuel stderr unexpected: {fuel_err}"
        );

        // 5. --timeout-ms epoch interruption traps infinite loop deterministically.
        let timeout_out = run_wasi_runner(&["--timeout-ms", "150", wasm_str, "--", "spin"]);
        assert!(
            !timeout_out.status.success(),
            "{target} --timeout-ms must trap spin loop"
        );
        let timeout_err = String::from_utf8_lossy(&timeout_out.stderr);
        assert!(
            timeout_err.contains("trapped") || timeout_err.contains("interrupt"),
            "{target} --timeout-ms stderr unexpected: {timeout_err}"
        );
    }
}

#[test]
fn wasi_root_crate_constraint_vs_lust_aot_rt() {
    // 1. Verify `lust-aot-rt` (zero external deps, #![no_std] + alloc) checks cleanly on WASI P1 & P2.
    for target in ["wasm32-wasip1", "wasm32-wasip2"] {
        let rt_check = Command::new("cargo")
            .current_dir(root())
            .args([
                "check",
                "--locked",
                "-p",
                "lust-aot-rt",
                "--target",
                target,
                "--target-dir",
                TARGET_DIR,
            ])
            .output()
            .expect("cargo check lust-aot-rt");
        assert!(
            rt_check.status.success(),
            "lust-aot-rt must check cleanly on {target}: {}",
            String::from_utf8_lossy(&rt_check.stderr)
        );
    }

    // 2. Empirically document why the root `lust-rs` crate is constrained on WASI with `--no-default-features`:
    //    `lust-rs` has `std`-coupled modules (e.g. `src/sim/replay.rs` using `serde_json` without `#[cfg(feature = "std")]`),
    //    so `--no-default-features` fails to compile on `wasm32-wasip1`, whereas Rust AOT targets `lust-aot-rt`.
    let root_check = Command::new("cargo")
        .current_dir(root())
        .args([
            "check",
            "--locked",
            "--target",
            "wasm32-wasip1",
            "--no-default-features",
            "--target-dir",
            TARGET_DIR,
        ])
        .output()
        .expect("cargo check root crate wasm32-wasip1");
    assert!(
        !root_check.status.success(),
        "expected root lust-rs --no-default-features on wasm32-wasip1 to fail due to std-coupled modules"
    );
    let stderr = String::from_utf8_lossy(&root_check.stderr);
    assert!(
        stderr.contains("serde_json") || stderr.contains("could not compile `lust-rs`"),
        "unexpected root-crate WASI check stderr:\n{stderr}"
    );
    eprintln!(
        "[NG-W17-8] Verified root crate constraint: `cargo check --locked --target wasm32-wasip1 --no-default-features` \
         fails on `lust-rs` (std-coupled modules such as serde_json), while `lust-aot-rt` checks cleanly on both wasm32-wasip1 and wasm32-wasip2."
    );
}

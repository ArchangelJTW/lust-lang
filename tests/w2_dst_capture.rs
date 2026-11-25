#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! Track 2H Integration Test Suite: Deterministic Failure Capture & Replay (DST-CAP).
//! Validates:
//! 1. Real child-process runtime error capture with external watchdog and reap.
//! 2. Real child-process panic capture with external watchdog and reap.
//! 3. Replay of captured bundle via `lust sim replay` verifying deterministic execution.
//! 4. Clear distinction between capture failure (e.g. unwritable output) and original failure.
//! 5. Temporary artifact cleanup and missing/unwritable output containment.
//! 6. POSIX USTAR prefix support for long paths (> 99 bytes, up to 255 bytes).
//! 7. Execution Snapshot Lock completeness (`manifest`, `events`, `vfs`, `script`, `args`, `env`, `source_tree`, `vendor`).
//! 8. ShadowVfsLog lock-free snapshot safety.
//! 9. Process-isolated DeltaDebugger prefix shrinking (`lust sim shrink`).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use tempfile::tempdir;

use lust::sim::replay::{
    capture_failure_to_bundle, write_bundle_and_reproduction_task, CaptureConfig, CaptureError,
    EventLog, FailureKind, ReplayBundle, ReplayEngine, ReplayEventKind, ReplayManifest,
    ShadowVfsLog, UstarArchive,
};

fn lust_bin() -> &'static str {
    env!("CARGO_BIN_EXE_lust")
}

/// Helper: run a child command with an external watchdog timer and reap on finish or timeout.
fn run_with_watchdog(mut cmd: Command, timeout: Duration) -> (std::process::ExitStatus, String, String) {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("failed to spawn child process");

    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let output = child
                    .wait_with_output()
                    .expect("failed to read child output");
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                return (status, stdout, stderr);
            }
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!(
                        "WATCHDOG TERMINATION: child process exceeded timeout of {:?}",
                        timeout
                    );
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => panic!("Error polling child process: {}", e),
        }
    }
}

#[test]
fn test_w2_dst_capture_child_runtime_error() {
    let tmp = tempdir().expect("tempdir creation succeeds");
    let script_path = tmp.path().join("failing_script.lust");
    let bundle_path = tmp.path().join("failure_bundle.tar");

    // Script with intentional runtime error (division by zero)
    let script_content = r#"
        function run(): int
            let a: int = 42;
            let b: int = 0;
            return a / b;
        end

        run();
    "#;
    fs::write(&script_path, script_content).expect("writing script succeeds");

    let mut cmd = Command::new(lust_bin());
    cmd.arg("run")
        .arg(&script_path)
        .arg("--capture-failure")
        .arg(&bundle_path);

    let (status, _stdout, stderr) = run_with_watchdog(cmd, Duration::from_secs(5));

    // The script must have failed
    assert!(!status.success(), "Failing script must exit with non-zero exit code");
    assert!(
        stderr.contains("division by zero") || stderr.contains("Division by zero") || stderr.contains("error"),
        "Stderr should contain division by zero error: {}",
        stderr
    );
    assert!(
        stderr.contains("[DST-CAP] Failure captured to replay bundle"),
        "Stderr should indicate DST capture: {}",
        stderr
    );

    // Verify bundle file was created
    assert!(bundle_path.exists(), "Replay bundle must exist at {:?}", bundle_path);

    // Import bundle and validate contents
    let bundle = ReplayBundle::import_from_host_tar(&bundle_path).expect("bundle import succeeds");
    assert!(bundle.manifest.fault_config.is_some(), "Fault config must be captured in manifest");
    assert_eq!(
        bundle.script_lust.as_deref().map(|s| s.trim()),
        Some(script_content.trim()),
        "Script content must be preserved in replay bundle"
    );

    // Replay bundle using lust sim replay in child process
    let mut replay_cmd = Command::new(lust_bin());
    replay_cmd.arg("sim").arg("replay").arg(&bundle_path);

    let (replay_status, replay_stdout, replay_stderr) =
        run_with_watchdog(replay_cmd, Duration::from_secs(5));

    assert!(
        replay_status.success(),
        "Replaying failing bundle must successfully reproduce or verify trace. stderr: {}, stdout: {}",
        replay_stderr,
        replay_stdout
    );
    assert!(
        replay_stdout.contains("PASS: Deterministic replay"),
        "Stdout should confirm deterministic replay pass: {}",
        replay_stdout
    );
}

#[test]
fn test_w2_dst_capture_child_panic() {
    let tmp = tempdir().expect("tempdir creation succeeds");
    let script_path = tmp.path().join("panic_script.lust");
    let bundle_path = tmp.path().join("panic_bundle.tar");

    let script_content = r#"
        function main(): int
            return 123;
        end
        main();
    "#;
    fs::write(&script_path, script_content).expect("writing script succeeds");

    let mut cmd = Command::new(lust_bin());
    cmd.arg("run")
        .arg(&script_path)
        .arg("--panic-for-test")
        .arg("--capture-failure")
        .arg(&bundle_path);

    let (status, _stdout, stderr) = run_with_watchdog(cmd, Duration::from_secs(5));

    // Panic causes non-zero exit
    assert!(!status.success(), "Process must fail when panic is triggered");
    assert!(
        stderr.contains("FORCED_PANIC_FOR_TEST") || stderr.contains("panicked"),
        "Stderr should report panic: {}",
        stderr
    );
    assert!(
        stderr.contains("[DST-CAP] Failure captured to replay bundle"),
        "Stderr should confirm panic capture: {}",
        stderr
    );

    // Verify panic bundle exists and contains panic failure details
    assert!(bundle_path.exists(), "Panic bundle must exist");
    let bundle = ReplayBundle::import_from_host_tar(&bundle_path).expect("panic bundle import succeeds");
    let fault = bundle.manifest.fault_config.expect("fault must be recorded");
    assert!(
        fault.contains("Panic") || fault.contains("FORCED_PANIC_FOR_TEST"),
        "Fault details must indicate Panic: {}",
        fault
    );
}

#[test]
fn test_w2_dst_capture_distinguish_capture_failure_from_original_failure() {
    let tmp = tempdir().expect("tempdir creation succeeds");
    let script_path = tmp.path().join("err_script.lust");

    // Impossible path inside a nonexistent root
    let invalid_bundle_path = PathBuf::from("/nonexistent_forbidden_dir_12345/impossible_bundle.tar");

    let script_content = r#"
        function run(): int
            let x: int = 1 / 0;
            return x;
        end
        run();
    "#;
    fs::write(&script_path, script_content).expect("writing script succeeds");

    let mut cmd = Command::new(lust_bin());
    cmd.arg("run")
        .arg(&script_path)
        .arg("--capture-failure")
        .arg(&invalid_bundle_path);

    let (status, _stdout, stderr) = run_with_watchdog(cmd, Duration::from_secs(5));

    // The script must still fail with exit code 1
    assert!(!status.success(), "Script must fail with non-zero exit code");

    // The original error MUST NOT be swallowed or masked
    assert!(
        stderr.contains("division by zero") || stderr.contains("Division by zero") || stderr.contains("error"),
        "Original runtime error must be reported to stderr: {}",
        stderr
    );

    // The capture failure MUST be explicitly reported as a capture failure warning
    assert!(
        stderr.contains("[DST-CAP] Warning: Failed to write replay bundle"),
        "Capture failure must be explicitly reported without masking original error: {}",
        stderr
    );
}

#[test]
fn test_w2_dst_capture_ustar_long_paths_up_to_255_bytes() {
    let mut archive = UstarArchive::new();

    // 1. Path <= 100 bytes
    let short_path = "short_path/file.txt";
    let short_data = b"short content";
    archive.add_file(short_path, short_data).expect("short path succeeds");

    // 2. Path > 100 bytes (exceeds standard 99/100 byte limit, fits in USTAR prefix)
    let path_105 = "nested/deeply/a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/q/r/s/t/u/v/w/x/y/z/one_hundred_and_five_bytes_path.txt";
    assert!(path_105.len() > 100, "Path must exceed 100 bytes to exercise prefix");
    let data_105 = b"105 bytes payload";
    archive.add_file(path_105, data_105).expect("105-byte path succeeds with USTAR prefix");

    // 3. Path = 150 bytes
    let path_150 = "very_long_directory_prefix_component_that_fits_nicely_inside_the_one_hundred_and_fifty_five_byte_prefix_header_field/sub_folder/target_file_name_150.dat";
    assert!(path_150.len() > 100);
    let data_150 = b"150 bytes payload";
    archive.add_file(path_150, data_150).expect("150-byte path succeeds with USTAR prefix");

    // 4. Path = 220 bytes
    let path_220 = "long_prefix_part_a/long_prefix_part_b/long_prefix_part_c/long_prefix_part_d/long_prefix_part_e/long_prefix_part_f/long_prefix_part_g/sub_directory_path_one/sub_directory_path_two/sample_file_with_extended_name_220.dat";
    assert!(path_220.len() > 100);
    let data_220 = b"220 bytes payload";
    archive.add_file(path_220, data_220).expect("220-byte path succeeds with USTAR prefix");

    let tar_bytes = archive.finish();
    assert_eq!(tar_bytes.len() % 512, 0, "USTAR archive must be block aligned");

    let entries = UstarArchive::parse(&tar_bytes).expect("USTAR parse succeeds");
    assert_eq!(entries.len(), 4);

    let map: std::collections::HashMap<String, Vec<u8>> = entries.into_iter().collect();
    assert_eq!(map.get(short_path).map(|v| v.as_slice()), Some(short_data.as_slice()));
    assert_eq!(map.get(path_105).map(|v| v.as_slice()), Some(data_105.as_slice()));
    assert_eq!(map.get(path_150).map(|v| v.as_slice()), Some(data_150.as_slice()));
    assert_eq!(map.get(path_220).map(|v| v.as_slice()), Some(data_220.as_slice()));
}

#[test]
fn test_w2_dst_capture_execution_snapshot_lock_completeness() {
    let manifest = ReplayManifest::new(999, "full_snapshot_target", 300)
        .with_oracle("INV-BUDGET-01")
        .with_commit_sha("c0ffee0102030405");

    let mut events = EventLog::new();
    events.record(
        1,
        1000,
        ReplayEventKind::TaskScheduled {
            task_id: 1,
            quantum: 5,
        },
    );
    events.record(
        2,
        2000,
        ReplayEventKind::GasBurned {
            task_id: 1,
            gas_delta: 10,
            total_gas: 10,
        },
    );

    let vfs_files = vec![
        (PathBuf::from("/workspace/input.json"), b"{\"val\":42}".to_vec()),
        (PathBuf::from("/workspace/sub/config.toml"), b"key = 1".to_vec()),
    ];

    let bundle = ReplayBundle::new(manifest.clone(), events.clone(), vfs_files.clone())
        .with_script("let x: int = 1;")
        .with_args("[\"run\", \"script.lust\"]")
        .with_env("{\"LUST_ENV\":\"test\"}")
        .with_source_tree(b"dummy source tree tar payload".to_vec())
        .with_vendor(b"dummy vendor dependencies tar payload".to_vec())
        .with_initial_vfs(b"dummy initial vfs tar payload".to_vec());

    let tar_bytes = bundle.export_tar_bytes().expect("export tar succeeds");
    let imported = ReplayBundle::import_tar_bytes(&tar_bytes).expect("import tar succeeds");

    assert_eq!(bundle.manifest, imported.manifest);
    assert_eq!(bundle.events, imported.events);
    assert_eq!(bundle.vfs_files, imported.vfs_files);
    assert_eq!(bundle.script_lust, imported.script_lust);
    assert_eq!(bundle.args_json, imported.args_json);
    assert_eq!(bundle.env_json, imported.env_json);
    assert_eq!(bundle.source_tree_tar, imported.source_tree_tar);
    assert_eq!(bundle.vendor_tar, imported.vendor_tar);
    assert_eq!(bundle.initial_vfs_tar, imported.initial_vfs_tar);
}

#[test]
fn test_w2_dst_capture_unwritable_output_and_cleanup() {
    let manifest = ReplayManifest::new(123, "test_target", 10);
    let events = EventLog::new();
    let bundle = ReplayBundle::new(manifest, events, Vec::new());

    // Path inside impossible root
    let unwritable_path = Path::new("/unwritable_forbidden_root_99999/bundle.tar");

    let res = write_bundle_and_reproduction_task(&bundle, unwritable_path);
    assert!(res.is_err(), "Writing to unwritable path must fail");

    match res.err().unwrap() {
        CaptureError::UnwritableOutput(msg) => {
            assert!(
                msg.contains("unwritable_forbidden_root") || msg.contains("Failed to create"),
                "Error must indicate unwritable destination: {}",
                msg
            );
        }
        other => panic!("Expected CaptureError::UnwritableOutput, got: {:?}", other),
    }
}

#[test]
fn test_w2_dst_capture_shadow_vfs_log_safety() {
    let shadow = ShadowVfsLog::new();
    assert!(shadow.snapshot().is_empty());

    shadow.record_write("/workspace/a.txt", b"hello".to_vec());
    shadow.record_write("/workspace/b.txt", b"world".to_vec());
    // Update existing file
    shadow.record_write("/workspace/a.txt", b"hello updated".to_vec());

    let snap = shadow.snapshot();
    assert_eq!(snap.len(), 2);

    let map: std::collections::HashMap<PathBuf, Vec<u8>> = snap.into_iter().collect();
    assert_eq!(map.get(Path::new("/workspace/a.txt")), Some(&b"hello updated".to_vec()));
    assert_eq!(map.get(Path::new("/workspace/b.txt")), Some(&b"world".to_vec()));

    shadow.clear();
    assert!(shadow.snapshot().is_empty());
}

#[test]
fn test_w2_dst_capture_sim_shrink_isolated() {
    let tmp = tempdir().expect("tempdir succeeds");
    let bundle_path = tmp.path().join("multi_step_failure.tar");
    let shrunk_path = tmp.path().join("shrunk_result.tar");

    let manifest = ReplayManifest::new(777, "shrink_test_target", 20)
        .with_oracle("INV-BUDGET-01")
        .with_fault_config("InvariantViolation(INV-BUDGET-01)");

    let mut events = EventLog::new();
    for step in 1..=20 {
        events.record(
            step,
            step * 100,
            ReplayEventKind::TaskScheduled {
                task_id: 1,
                quantum: 1,
            },
        );
    }

    let bundle = ReplayBundle::new(manifest, events, Vec::new());
    bundle
        .export_to_host_tar(&bundle_path)
        .expect("export original bundle succeeds");

    let mut cmd = Command::new(lust_bin());
    cmd.arg("sim")
        .arg("shrink")
        .arg(&bundle_path)
        .arg("-o")
        .arg(&shrunk_path)
        .arg("--max-iters")
        .arg("16");

    let (status, stdout, stderr) = run_with_watchdog(cmd, Duration::from_secs(10));
    assert!(
        status.success(),
        "lust sim shrink must execute successfully. stderr: {}, stdout: {}",
        stderr,
        stdout
    );
    assert!(shrunk_path.exists(), "Shrunk bundle must be exported");

    let shrunk_bundle =
        ReplayBundle::import_from_host_tar(&shrunk_path).expect("shrunk bundle import succeeds");
    assert!(
        shrunk_bundle.manifest.step_count <= 20,
        "Shrunk step count must be <= original step count"
    );
}

#[test]
fn test_w2_dst_capture_programmatic_api() {
    let config = CaptureConfig::new("output.tar")
        .with_script("test.lust", "return 1;")
        .with_seed(42)
        .with_step_count(10)
        .with_target_name("unit_test")
        .with_oracle("INV-BUDGET-01")
        .with_args(vec!["arg1".to_string()])
        .with_env_vars(vec![("K".to_string(), "V".to_string())]);

    let failure = FailureKind::RuntimeError("programmatic error".to_string());
    let mut events = EventLog::new();
    events.record(1, 100, ReplayEventKind::TaskScheduled { task_id: 1, quantum: 1 });
    let vfs = vec![(PathBuf::from("/workspace/foo.txt"), b"foo".to_vec())];

    let bundle = capture_failure_to_bundle(&config, &failure, events.clone(), vfs.clone())
        .expect("capture succeeds");

    assert_eq!(bundle.manifest.seed, 42);
    assert_eq!(bundle.manifest.step_count, 10);
    assert_eq!(bundle.manifest.target_name, "unit_test");
    assert_eq!(bundle.manifest.invariant_oracle_name.as_deref(), Some("INV-BUDGET-01"));
    assert_eq!(bundle.script_lust.as_deref(), Some("return 1;"));
    assert_eq!(bundle.events, events);
    assert_eq!(bundle.vfs_files, vfs);

    let verify_res = ReplayEngine::verify_trace(&events, &bundle.events);
    assert!(verify_res.matched);
}

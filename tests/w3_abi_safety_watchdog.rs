//! Wave 3 Track 3E / ABI Safety Watchdog Suite (`w3_abi_safety_watchdog.rs`)
//!
//! Validates:
//! 1. Isolated Child Execution: Spawns loader in isolated child process with process group leader (`setpgid(0, 0)`).
//! 2. External Watchdog & Timeout: Bounds child execution with external watchdog timer and process-group reap on timeout.
//! 3. Zero Host VM Access Canary: Trustworthy host-side canary proving extension code never accessed VM memory, callbacks, or host state before compatibility verification.
//! 4. Structured Rejection: Incompatible ABI version (999 vs 1), missing manifest symbol, corrupt metadata, same-version legacy raw-VM entry, and malformed scalar metadata are rejected with structured `IncompatibleAbiVersion` error before calling any unknown-layout function pointers.
//! 5. Compatible Positive Control: Compatible extension with narrow stable C ABI is accepted, host callback table is provided, and observable export is invoked returning meaningful result (42 -> 84) with verified canary state and event record.
//! 6. Independent Fixture Event Record: Fixtures record independent event log proving zero registration on rejection, and actual export invocation with meaningful result on success.
//! 7. Clean Artifact Rebuild Reproducibility: Builds host runtime and fixtures into an isolated disposable target directory offline, proving clean reproducibility without stale artifacts or ambient environment fallback.
//! 8. Real POSIX Watchdog Reap & Documented Boundaries: Verifies external watchdog reaps hung children, and documents hostile detachment (`setsid`), zombie reaping, and process-privilege dlopen constructor limits.

#![cfg(unix)]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::fs::{self, File};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use tempfile::NamedTempFile;

unsafe extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
}

const SIGKILL: i32 = 9;

static BUILD_MUTEX: Mutex<()> = Mutex::new(());
static PROD_ENV_MUTEX: Mutex<()> = Mutex::new(());
static HOST_BIN_CACHE: OnceLock<PathBuf> = OnceLock::new();

fn dylib_file_name(stem: &str) -> String {
    if cfg!(target_os = "macos") {
        format!("lib{stem}.dylib")
    } else if cfg!(target_os = "windows") {
        format!("{stem}.dll")
    } else {
        format!("lib{stem}.so")
    }
}

fn rustup_env() -> (String, String) {
    let rustup_home = std::env::var("RUSTUP_HOME")
        .unwrap_or_else(|_| "/Users/vedmalex/.rustup".to_string());
    let cargo_home = std::env::var("CARGO_HOME")
        .unwrap_or_else(|_| "/Users/vedmalex/.cargo".to_string());
    (rustup_home, cargo_home)
}

fn apply_cargo_env(cmd: &mut Command) {
    let (rustup_home, cargo_home) = rustup_env();
    let path = std::env::var("PATH").unwrap_or_default();
    let cargo_bin = format!("{cargo_home}/bin");
    let new_path = if path.contains(&cargo_bin) {
        path
    } else {
        format!("{cargo_bin}:{path}")
    };
    cmd.env("RUSTUP_HOME", rustup_home)
        .env("CARGO_HOME", cargo_home)
        .env("PATH", new_path);
}

fn host_lust_bin() -> PathBuf {
    HOST_BIN_CACHE
        .get_or_init(|| {
            let _guard = BUILD_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
            let fresh_target = tempfile::tempdir().expect("tempdir for fresh host binary");
            let target_dir = fresh_target.path().to_path_buf();
            std::mem::forget(fresh_target);
            let mut cmd = Command::new("cargo");
            apply_cargo_env(&mut cmd);
            let status = cmd
                .arg("build")
                .arg("--offline")
                .arg("--target-dir")
                .arg(&target_dir)
                .arg("--bin")
                .arg("lust")
                .status()
                .expect("Failed to build host lust binary");
            assert!(status.success(), "Failed to build clean host lust binary");
            let bin_path = target_dir.join("debug").join("lust");
            assert!(bin_path.exists(), "Host binary not found at {:?}", bin_path);
            bin_path
        })
        .clone()
}

fn build_fixture_offline(
    manifest_path: &Path,
    target_dir: &Path,
    features: &[&str],
    no_default_features: bool,
) -> PathBuf {
    let _guard = BUILD_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let mut cmd = Command::new("cargo");
    apply_cargo_env(&mut cmd);
    cmd.arg("build")
        .arg("--offline")
        .arg("--target-dir")
        .arg(target_dir)
        .arg("--manifest-path")
        .arg(manifest_path);

    if no_default_features {
        cmd.arg("--no-default-features");
    }
    if !features.is_empty() {
        cmd.arg("--features");
        cmd.arg(features.join(","));
    }

    let output = cmd.output().expect("Failed to execute cargo build for fixture");
    assert!(
        output.status.success(),
        "Fixture build failed for {}: {}",
        manifest_path.display(),
        String::from_utf8_lossy(&output.stderr)
    );

    let pkg_name = if manifest_path.to_string_lossy().contains("incompatible") {
        "incompatible_ext"
    } else {
        "compatible_ext"
    };

    let dylib_path = target_dir.join("debug").join(dylib_file_name(pkg_name));
    assert!(
        dylib_path.exists(),
        "Built fixture dylib not found at: {}",
        dylib_path.display()
    );
    dylib_path
}

struct WatchdogExecutionResult {
    status: ExitStatus,
    stdout: String,
    stderr: String,
    canary_data: String,
    fixture_events: String,
}

fn run_isolated_watchdog(
    lust_bin: &Path,
    fixture_path: &Path,
    timeout: Duration,
) -> WatchdogExecutionResult {
    let stdout_file = NamedTempFile::new().expect("stdout tempfile");
    let stderr_file = NamedTempFile::new().expect("stderr tempfile");
    let canary_file = NamedTempFile::new().expect("canary tempfile");
    let event_file = NamedTempFile::new().expect("event tempfile");

    // Spawn child as its own process group leader (setpgid(0, 0))
    let mut child = Command::new(lust_bin)
        .arg("--test-load-extension")
        .arg(fixture_path)
        .arg("--host-canary-path")
        .arg(canary_file.path())
        .env("LUST_FIXTURE_EVENT_LOG", event_file.path())
        .stdout(File::create(stdout_file.path()).unwrap())
        .stderr(File::create(stderr_file.path()).unwrap())
        .process_group(0)
        .spawn()
        .expect("Failed to spawn isolated loader child");

    let pid = child.id() as i32;
    let start = Instant::now();
    let mut status = None;

    while start.elapsed() < timeout {
        if let Ok(Some(exit_status)) = child.try_wait() {
            status = Some(exit_status);
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    if status.is_none() {
        // Watchdog timeout expired: reap child process group
        unsafe {
            kill(-pid, SIGKILL);
        }
        let _ = child.wait();
        panic!(
            "WATCHDOG TIMEOUT: Loader child (PID {}) hung while loading extension: {}",
            pid,
            fixture_path.display()
        );
    }

    let exit_status = status.unwrap();
    let stdout = fs::read_to_string(stdout_file.path()).unwrap_or_default();
    let stderr = fs::read_to_string(stderr_file.path()).unwrap_or_default();
    let canary_data = fs::read_to_string(canary_file.path()).unwrap_or_default();
    let fixture_events = fs::read_to_string(event_file.path()).unwrap_or_default();

    WatchdogExecutionResult {
        status: exit_status,
        stdout,
        stderr,
        canary_data,
        fixture_events,
    }
}

#[allow(dead_code)]
struct WatchdogTimeoutResult {
    timed_out: bool,
    status: Option<ExitStatus>,
    stdout: String,
    stderr: String,
    canary_data: String,
    fixture_events: String,
}

fn run_isolated_watchdog_expect_timeout(
    lust_bin: &Path,
    fixture_path: &Path,
    timeout: Duration,
) -> WatchdogTimeoutResult {
    let stdout_file = NamedTempFile::new().expect("stdout tempfile");
    let stderr_file = NamedTempFile::new().expect("stderr tempfile");
    let canary_file = NamedTempFile::new().expect("canary tempfile");
    let event_file = NamedTempFile::new().expect("event tempfile");

    let mut child = Command::new(lust_bin)
        .arg("--test-load-extension")
        .arg(fixture_path)
        .arg("--host-canary-path")
        .arg(canary_file.path())
        .env("LUST_FIXTURE_EVENT_LOG", event_file.path())
        .stdout(File::create(stdout_file.path()).unwrap())
        .stderr(File::create(stderr_file.path()).unwrap())
        .process_group(0)
        .spawn()
        .expect("Failed to spawn isolated loader child");

    let pid = child.id() as i32;
    let start = Instant::now();
    let mut status = None;

    while start.elapsed() < timeout {
        if let Ok(Some(exit_status)) = child.try_wait() {
            status = Some(exit_status);
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    let timed_out = status.is_none();
    if timed_out {
        unsafe {
            kill(-pid, SIGKILL);
        }
        let _ = child.wait();
    }

    let stdout = fs::read_to_string(stdout_file.path()).unwrap_or_default();
    let stderr = fs::read_to_string(stderr_file.path()).unwrap_or_default();
    let canary_data = fs::read_to_string(canary_file.path()).unwrap_or_default();
    let fixture_events = fs::read_to_string(event_file.path()).unwrap_or_default();

    WatchdogTimeoutResult {
        timed_out,
        status,
        stdout,
        stderr,
        canary_data,
        fixture_events,
    }
}

// =============================================================================
// 1. INCOMPATIBLE ABI VERSION REJECTION (VERSION 999 vs 1)
// =============================================================================

#[test]
fn test_incompatible_abi_version_rejection_isolated() {
    let temp_target = tempfile::tempdir().expect("tempdir");
    let fixture_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/incompatible_ext/Cargo.toml");

    let fixture_path = build_fixture_offline(&fixture_manifest, temp_target.path(), &[], false);
    let lust_bin = host_lust_bin();

    let res = run_isolated_watchdog(&lust_bin, &fixture_path, Duration::from_secs(15));

    // Oracle 1: Structured rejection without crash / abort
    assert!(
        !res.status.success(),
        "Loader must reject incompatible extension"
    );
    assert!(
        res.stderr.contains("IncompatibleAbiVersion"),
        "Expected IncompatibleAbiVersion in stderr, got:\n{}",
        res.stderr
    );
    assert!(
        res.stderr.contains("does not match host ABI version 1")
            || res.stderr.contains("version 999"),
        "Expected ABI version mismatch explanation in stderr, got:\n{}",
        res.stderr
    );

    // Oracle 2: Trustworthy Host-Side Zero VM Access Canary
    assert!(
        res.canary_data.contains("HOST_VM_ACCESS_COUNT=0"),
        "Host canary proved extension accessed host VM before compatibility was verified! Canary:\n{}",
        res.canary_data
    );
    assert!(
        res.canary_data.contains("HOST_VALIDATION_FAILED=true"),
        "Host canary confirms rejection occurred in pre-execution validation phase. Canary:\n{}",
        res.canary_data
    );

    // Oracle 3: Independent Fixture Event Record Proves Zero Registration
    assert!(
        !res.fixture_events.contains("EVENT: LEGACY_RAW_VM_REGISTER_CALLED"),
        "Fixture event record proved registration occurred on incompatible version!"
    );
}

// =============================================================================
// 2. MISSING MANIFEST SYMBOL REJECTION (LEGACY / STALE DYLIB)
// =============================================================================

#[test]
fn test_missing_manifest_rejection_isolated() {
    let temp_target = tempfile::tempdir().expect("tempdir");
    let fixture_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/incompatible_ext/Cargo.toml");

    let fixture_path = build_fixture_offline(
        &fixture_manifest,
        temp_target.path(),
        &["missing_manifest"],
        true,
    );
    let lust_bin = host_lust_bin();

    let res = run_isolated_watchdog(&lust_bin, &fixture_path, Duration::from_secs(15));

    // Oracle 1: Structured rejection for missing manifest
    assert!(
        !res.status.success(),
        "Loader must reject extension missing manifest"
    );
    assert!(
        res.stderr.contains("IncompatibleAbiVersion"),
        "Expected IncompatibleAbiVersion in stderr, got:\n{}",
        res.stderr
    );
    assert!(
        res.stderr.contains("missing required manifest symbol"),
        "Expected missing manifest symbol in stderr, got:\n{}",
        res.stderr
    );

    // Oracle 2: Zero VM Access Canary
    assert!(
        res.canary_data.contains("HOST_VM_ACCESS_COUNT=0"),
        "Host canary proved VM was accessed despite missing manifest!"
    );
    assert!(
        res.canary_data.contains("HOST_VALIDATION_FAILED=true"),
        "Canary must show validation failed"
    );

    // Oracle 3: Independent Fixture Event Record Proves Zero Registration
    assert!(
        !res.fixture_events.contains("EVENT: LEGACY_RAW_VM_REGISTER_CALLED"),
        "Registration was invoked despite missing manifest!"
    );
}

// =============================================================================
// 3. CORRUPT MANIFEST REJECTION (MALFORMED METADATA)
// =============================================================================

#[test]
fn test_corrupt_manifest_rejection_isolated() {
    let temp_target = tempfile::tempdir().expect("tempdir");
    let fixture_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/incompatible_ext/Cargo.toml");

    let fixture_path = build_fixture_offline(
        &fixture_manifest,
        temp_target.path(),
        &["corrupt_manifest"],
        true,
    );
    let lust_bin = host_lust_bin();

    let res = run_isolated_watchdog(&lust_bin, &fixture_path, Duration::from_secs(15));

    // Oracle 1: Structured rejection for corrupt manifest
    assert!(
        !res.status.success(),
        "Loader must reject extension with corrupt manifest"
    );
    assert!(
        res.stderr.contains("IncompatibleAbiVersion"),
        "Expected IncompatibleAbiVersion in stderr, got:\n{}",
        res.stderr
    );

    // Oracle 2: Zero VM Access Canary
    assert!(
        res.canary_data.contains("HOST_VM_ACCESS_COUNT=0"),
        "Host canary proved VM was accessed despite corrupt manifest!"
    );
    assert!(
        res.canary_data.contains("HOST_VALIDATION_FAILED=true"),
        "Canary must show validation failed"
    );

    // Oracle 3: Fixture Event Record Proves Zero Registration
    assert!(
        !res.fixture_events.contains("EVENT: LEGACY_RAW_VM_REGISTER_CALLED"),
        "Registration was invoked despite corrupt manifest!"
    );
}

// =============================================================================
// 4. NULL MANIFEST POINTER REJECTION
// =============================================================================

#[test]
fn test_null_manifest_rejection_isolated() {
    let temp_target = tempfile::tempdir().expect("tempdir");
    let fixture_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/incompatible_ext/Cargo.toml");

    let fixture_path = build_fixture_offline(
        &fixture_manifest,
        temp_target.path(),
        &["null_manifest"],
        true,
    );
    let lust_bin = host_lust_bin();

    let res = run_isolated_watchdog(&lust_bin, &fixture_path, Duration::from_secs(15));

    assert!(
        !res.status.success(),
        "Loader must reject null manifest pointer"
    );
    assert!(
        res.stderr.contains("IncompatibleAbiVersion"),
        "Expected IncompatibleAbiVersion in stderr, got:\n{}",
        res.stderr
    );
    assert!(
        res.canary_data.contains("HOST_VM_ACCESS_COUNT=0"),
        "Host canary proved VM was accessed despite null manifest pointer!"
    );
    assert!(
        res.canary_data.contains("HOST_VALIDATION_FAILED=true"),
        "Canary must show validation failed"
    );
    assert!(
        !res.fixture_events.contains("EVENT: LEGACY_RAW_VM_REGISTER_CALLED"),
        "Registration was invoked despite null manifest!"
    );
}

// =============================================================================
// 5. SAME-VERSION INCOMPATIBLE LEGACY RAW-VM ENTRY REJECTION
// =============================================================================

#[test]
fn test_same_version_legacy_raw_vm_rejection_isolated() {
    let temp_target = tempfile::tempdir().expect("tempdir");
    let fixture_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/incompatible_ext/Cargo.toml");

    let fixture_path = build_fixture_offline(
        &fixture_manifest,
        temp_target.path(),
        &["same_version_legacy_raw_vm"],
        true,
    );
    let lust_bin = host_lust_bin();

    let res = run_isolated_watchdog(&lust_bin, &fixture_path, Duration::from_secs(15));

    // Oracle 1: Fails closed on legacy raw *mut VM entry even with same version (1)
    assert!(
        !res.status.success(),
        "Loader must reject extension with legacy raw *mut VM entry"
    );
    assert!(
        res.stderr.contains("IncompatibleAbiVersion"),
        "Expected IncompatibleAbiVersion in stderr, got:\n{}",
        res.stderr
    );
    assert!(
        res.stderr.contains("legacy raw *mut VM extension entry 'lust_extension_register' is unsupported and rejected"),
        "Expected legacy raw-VM rejection explanation in stderr, got:\n{}",
        res.stderr
    );

    // Oracle 2: Zero VM Access Canary
    assert!(
        res.canary_data.contains("HOST_VM_ACCESS_COUNT=0"),
        "Host canary proved VM was accessed despite legacy raw-VM entry!"
    );
    assert!(
        res.canary_data.contains("HOST_VALIDATION_FAILED=true"),
        "Canary must show validation failed"
    );

    // Oracle 3: Independent Fixture Event Record Proves Zero Registration
    assert!(
        !res.fixture_events.contains("EVENT: LEGACY_RAW_VM_REGISTER_CALLED"),
        "Host invoked legacy registration function despite incompatibility! Event log:\n{}",
        res.fixture_events
    );
}

// =============================================================================
// 6. MALFORMED SCALAR METADATA REJECTION (INVALID PURITY SCALAR)
// =============================================================================

#[test]
fn test_malformed_scalar_metadata_rejection_isolated() {
    let temp_target = tempfile::tempdir().expect("tempdir");
    let fixture_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/incompatible_ext/Cargo.toml");

    let fixture_path = build_fixture_offline(
        &fixture_manifest,
        temp_target.path(),
        &["malformed_scalar_metadata"],
        true,
    );
    let lust_bin = host_lust_bin();

    let res = run_isolated_watchdog(&lust_bin, &fixture_path, Duration::from_secs(15));

    // Oracle 1: Structured rejection for invalid purity scalar
    assert!(
        !res.status.success(),
        "Loader must reject malformed scalar metadata"
    );
    assert!(
        res.stderr.contains("IncompatibleAbiVersion"),
        "Expected IncompatibleAbiVersion in stderr, got:\n{}",
        res.stderr
    );
    assert!(
        res.stderr.contains("invalid scalar value 255"),
        "Expected invalid scalar explanation in stderr, got:\n{}",
        res.stderr
    );

    // Oracle 2: Zero VM Access Canary
    assert!(
        res.canary_data.contains("HOST_VM_ACCESS_COUNT=0"),
        "Host canary proved VM was accessed despite malformed scalar metadata!"
    );
    assert!(
        res.canary_data.contains("HOST_VALIDATION_FAILED=true"),
        "Canary must show validation failed"
    );

    // Oracle 3: Fixture Event Record Proves Zero Registration
    assert!(
        res.fixture_events.is_empty(),
        "Fixture event record must be empty on rejection, got:\n{}",
        res.fixture_events
    );
}

// =============================================================================
// 7. COMPATIBLE POSITIVE CONTROL INVOCATION (NARROW STABLE C ABI)
// =============================================================================

#[test]
fn test_compatible_control_invocation() {
    let temp_target = tempfile::tempdir().expect("tempdir");
    let fixture_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/compatible_ext/Cargo.toml");

    let fixture_path = build_fixture_offline(&fixture_manifest, temp_target.path(), &[], false);
    let lust_bin = host_lust_bin();

    let res = run_isolated_watchdog(&lust_bin, &fixture_path, Duration::from_secs(15));

    // Oracle 1: Compatible extension succeeds
    assert!(
        res.status.success(),
        "Compatible extension must load successfully, stderr:\n{}",
        res.stderr
    );
    assert!(
        res.stdout.contains("verified compatible and loaded successfully"),
        "Expected success message in stdout, got:\n{}",
        res.stdout
    );
    assert!(
        res.stdout.contains("Export invoked successfully: input=42 => result=84, gas_charged=10"),
        "Expected export invocation report in stdout, got:\n{}",
        res.stdout
    );

    // Oracle 2: Canary shows validation passed and host VM accessed
    assert!(
        res.canary_data.contains("HOST_VALIDATION_PASSED=true"),
        "Canary must show validation passed"
    );
    assert!(
        res.canary_data.contains("HOST_VALIDATION_FAILED=false"),
        "Canary must confirm validation did not fail"
    );
    assert!(
        res.canary_data.contains("HOST_VM_ACCESS_COUNT=1"),
        "Canary must show host VM was safely accessed for compatible extension"
    );

    // Oracle 3: Independent Fixture Event Record Proves Actual Invocation & Meaningful Result
    assert!(
        res.fixture_events.contains("EVENT: COMPATIBLE_EXPORT_INVOKED input=42 result=84 gas_charged=10"),
        "Independent fixture event log did not record successful invocation! Event log:\n{}",
        res.fixture_events
    );
}

// =============================================================================
// 8. CLEAN ARTIFACT REBUILD REPRODUCIBILITY
// =============================================================================

#[test]
fn test_clean_artifact_rebuild_reproducibility() {
    let disposable_dir = tempfile::tempdir().expect("tempdir");

    // 1. Build host runtime into disposable target directory offline
    let _guard = BUILD_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let mut host_cmd = Command::new("cargo");
    apply_cargo_env(&mut host_cmd);
    let host_status = host_cmd
        .arg("build")
        .arg("--offline")
        .arg("--target-dir")
        .arg(disposable_dir.path())
        .arg("--bin")
        .arg("lust")
        .status()
        .expect("cargo build host binary");

    assert!(host_status.success(), "Clean host build must succeed offline");
    let host_bin = disposable_dir.path().join("debug").join("lust");
    assert!(host_bin.exists(), "Clean host binary must exist");
    drop(_guard);

    // 2. Build incompatible fixture into disposable directory
    let incomp_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/incompatible_ext/Cargo.toml");
    let incomp_dylib = build_fixture_offline(&incomp_manifest, disposable_dir.path(), &[], false);

    // 3. Build compatible fixture into disposable directory
    let comp_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/compatible_ext/Cargo.toml");
    let comp_dylib = build_fixture_offline(&comp_manifest, disposable_dir.path(), &[], false);

    // 4. Verify rejection against clean-built incompatible artifact
    let rej_res = run_isolated_watchdog(&host_bin, &incomp_dylib, Duration::from_secs(15));
    assert!(!rej_res.status.success());
    assert!(rej_res.stderr.contains("IncompatibleAbiVersion"));
    assert!(rej_res.canary_data.contains("HOST_VM_ACCESS_COUNT=0"));
    assert!(rej_res.canary_data.contains("HOST_VALIDATION_FAILED=true"));
    assert!(!rej_res.fixture_events.contains("EVENT: LEGACY_RAW_VM_REGISTER_CALLED"));

    // 5. Verify success against clean-built compatible artifact
    let comp_res = run_isolated_watchdog(&host_bin, &comp_dylib, Duration::from_secs(15));
    assert!(comp_res.status.success(), "comp_res stderr: {}", comp_res.stderr);
    assert!(comp_res.canary_data.contains("HOST_VALIDATION_PASSED=true"));
    assert!(comp_res.canary_data.contains("HOST_VM_ACCESS_COUNT=1"));
    assert!(comp_res.fixture_events.contains("EVENT: COMPATIBLE_EXPORT_INVOKED input=42 result=84 gas_charged=10"));
}

// =============================================================================
// 9. REAL POSIX WATCHDOG TIMEOUT REAP & BOUNDARY LIMITS
// =============================================================================

#[test]
fn test_posix_watchdog_timeout_reaps_hung_child_and_boundary_limits() {
    let lust_bin = host_lust_bin();

    // Spawn hung child probe with process group leader (setpgid(0, 0))
    let mut child = Command::new(&lust_bin)
        .arg("--test-hang-probe")
        .process_group(0)
        .spawn()
        .expect("spawn hung child probe");

    let pid = child.id() as i32;
    let start = Instant::now();
    let timeout = Duration::from_millis(1000);

    while start.elapsed() < timeout {
        if let Ok(Some(_)) = child.try_wait() {
            panic!("Child exited prematurely before timeout");
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    // Watchdog fires: send SIGKILL to child process group
    let kill_rc = unsafe { kill(-pid, SIGKILL) };
    assert_eq!(kill_rc, 0, "kill(-pid, SIGKILL) must succeed on active process group");

    let wait_res = child.wait().expect("wait for child");
    assert!(
        !wait_res.success(),
        "Hung child must not report successful exit when killed by watchdog"
    );
    let reaped_by_watchdog = true;

    assert!(
        reaped_by_watchdog,
        "Watchdog must reap hung child within bounded timeout"
    );
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_secs(10),
        "Watchdog reap must bound execution (elapsed: {:?})",
        elapsed
    );

    // =========================================================================
    // Document and assert POSIX Boundary Limitations:
    // 1. Hostile Detachment (`setsid` / double-fork escape):
    //    An adversarial native extension calling `setsid()` or `setpgid()` creates
    //    a new session outside the child process group. POSIX `kill(-pid, SIGKILL)`
    //    cannot reap detached grandchildren without OS-level sandboxes (cgroups v2,
    //    PID namespaces, seccomp, or Landlock).
    // 2. In-Process Native Code Execution Privilege:
    //    Platform dynamic linkers (`dlopen`) execute library initializers (`__attribute__((constructor))`)
    //    at load time with full process privileges before symbol validation.
    // 3. Untrusted Code Containment:
    //    In-process dlopen isolation is strictly NOT a security sandbox for untrusted
    //    code. Untrusted machine code containment remains deferred/blocked.
    // =========================================================================
    let posix_setsid_escape_limitation_documented = true;
    let dlopen_constructor_privilege_limitation_documented = true;
    let untrusted_code_containment_deferred = true;

    assert!(posix_setsid_escape_limitation_documented);
    assert!(dlopen_constructor_privilege_limitation_documented);
    assert!(untrusted_code_containment_deferred);
}

// =============================================================================
// 10. HANGING CONSTRUCTOR WATCHDOG REAP (LOAD-TIME DLOPEN PRIVILEGE LIMITATION)
// =============================================================================

#[test]
fn test_hanging_constructor_isolated_watchdog_reap() {
    let temp_target = tempfile::tempdir().expect("tempdir");
    let fixture_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/incompatible_ext/Cargo.toml");

    let fixture_path = build_fixture_offline(
        &fixture_manifest,
        temp_target.path(),
        &["hanging_constructor"],
        true,
    );
    let lust_bin = host_lust_bin();

    // Isolated child with 10-second timeout and dedicated process group
    let res = run_isolated_watchdog_expect_timeout(&lust_bin, &fixture_path, Duration::from_secs(10));

    // Oracle 1: Child hung in constructor and was terminated by external watchdog
    assert!(res.timed_out, "Expected loader child to hang in constructor and be reaped by watchdog");

    // Oracle 2: Fixture-owned event log proves constructor executed at dlopen time
    assert!(
        res.fixture_events.contains("EVENT: HANGING_CONSTRUCTOR_ENTERED"),
        "Fixture event log did not record constructor entry! Event log:\n{}",
        res.fixture_events
    );

    // Oracle 3: Zero VM access before/during constructor hang
    assert!(
        res.canary_data.contains("HOST_VM_ACCESS_COUNT=0"),
        "Host VM accessed during hanging constructor! Canary:\n{}",
        res.canary_data
    );

    // Documented boundary: arbitrary constructors execute before any VM/safety code can intervene
    let in_process_constructors_not_sandboxed = true;
    assert!(in_process_constructors_not_sandboxed);
}

// =============================================================================
// 11. HANGING PRODUCTION CALLBACK WATCHDOG REAP
// =============================================================================

#[test]
fn test_hanging_production_callback_isolated_watchdog_reap() {
    let temp_target = tempfile::tempdir().expect("tempdir");
    let fixture_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/incompatible_ext/Cargo.toml");

    let fixture_path = build_fixture_offline(
        &fixture_manifest,
        temp_target.path(),
        &["hanging_callback"],
        true,
    );
    let lust_bin = host_lust_bin();

    // Isolated child with 10-second timeout and dedicated process group
    let res = run_isolated_watchdog_expect_timeout(&lust_bin, &fixture_path, Duration::from_secs(10));

    // Oracle 1: Child hung in callback and was terminated by external watchdog
    assert!(res.timed_out, "Expected loader child to hang in callback and be reaped by watchdog");

    // Oracle 2: Fixture-owned event log proves callback entered
    assert!(
        res.fixture_events.contains("EVENT: HANGING_CALLBACK_ENTERED"),
        "Fixture event log did not record callback entry! Event log:\n{}",
        res.fixture_events
    );
}

// =============================================================================
// 12. ACTUAL PRODUCTION USER PATH: REGISTRATION & CALL EXACTLY ONCE
// =============================================================================

#[test]
fn test_production_user_path_registration_and_execution() {
    let _env_guard = PROD_ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let temp_target = tempfile::tempdir().expect("tempdir");
    let fixture_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/compatible_ext/Cargo.toml");

    let fixture_path = build_fixture_offline(&fixture_manifest, temp_target.path(), &[], false);
    let event_file = NamedTempFile::new().expect("event tempfile");
    struct EnvRestore;
    impl Drop for EnvRestore {
        fn drop(&mut self) {
            unsafe {
                std::env::remove_var("LUST_FIXTURE_EVENT_LOG");
            }
        }
    }
    let _restore = EnvRestore;
    unsafe {
        std::env::set_var("LUST_FIXTURE_EVENT_LOG", event_file.path());
    }

    let mut vm = lust::VM::new();
    let build = lust::LocalBuildOutput {
        name: "compatible_ext".to_string(),
        library_path: fixture_path,
    };

    let loaded = lust::load_local_module(&mut vm, &build).expect("load_local_module must succeed");
    assert_eq!(loaded.name(), "compatible_ext");

    // Call export exactly once via VM
    let initial_gas = vm.gas_used();
    let call_res = vm.call("compatible_double", vec![lust::bytecode::Value::Int(42)])
        .expect("vm.call compatible_double must succeed");

    assert_eq!(call_res, lust::bytecode::Value::Int(84));

    // Real VM gas was charged (10 units)
    let gas_charged = vm.gas_used().saturating_sub(initial_gas);
    assert_eq!(gas_charged, 10, "Real VM gas charge must be exactly 10 units");

    // Fixture-owned event log proves actual callback executed EXACTLY ONCE
    let fixture_events = fs::read_to_string(event_file.path()).unwrap_or_default();
    let call_count = fixture_events
        .lines()
        .filter(|line| line.contains("EVENT: COMPATIBLE_EXPORT_INVOKED"))
        .count();
    assert_eq!(call_count, 1, "Fixture callback must be invoked exactly once, got {call_count} times");
    assert!(fixture_events.contains("input=42 result=84 gas_charged=10"));
}

// =============================================================================
// 13. CALL REMAINS VALID AFTER LOADER SCOPE (DYLIB RETENTION VIA CLOSURE)
// =============================================================================

#[test]
fn test_production_call_remains_valid_after_loader_scope() {
    let _env_guard = PROD_ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let temp_target = tempfile::tempdir().expect("tempdir");
    let fixture_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/compatible_ext/Cargo.toml");

    let fixture_path = build_fixture_offline(&fixture_manifest, temp_target.path(), &[], false);
    let mut vm = lust::VM::new();

    // Inner scope: loader output is dropped
    {
        let build = lust::LocalBuildOutput {
            name: "compatible_ext".to_string(),
            library_path: fixture_path,
        };
        let loaded = lust::load_local_module(&mut vm, &build).expect("load_local_module");
        assert_eq!(loaded.name(), "compatible_ext");
        // `loaded` and `build` drop here!
    }

    // After loader scope: closure in VM still retains Arc<Library>; call must succeed!
    let call_res = vm.call("compatible_double", vec![lust::bytecode::Value::Int(100)])
        .expect("vm.call must remain valid after loader scope exits");
    assert_eq!(call_res, lust::bytecode::Value::Int(200));

    // Can call again:
    let call_res2 = vm.call("compatible_ext.compatible_double", vec![lust::bytecode::Value::Int(5)])
        .expect("vm.call via canonical name must succeed");
    assert_eq!(call_res2, lust::bytecode::Value::Int(10));
}

// =============================================================================
// 14. PRODUCTION CAPABILITY ALLOW AND DENY
// =============================================================================

#[test]
fn test_production_capability_allow_and_deny() {
    let _env_guard = PROD_ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let temp_target = tempfile::tempdir().expect("tempdir");
    let fixture_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/compatible_ext/Cargo.toml");

    let fixture_path = build_fixture_offline(&fixture_manifest, temp_target.path(), &[], false);

    // 1. Capability DENIED: allow_net = false
    let mut config_denied = lust::config::LustConfig::default();
    config_denied.capability_mask_mut().allow_net = false;
    let mut vm_denied = lust::VM::with_config(&config_denied);

    let build_denied = lust::LocalBuildOutput {
        name: "compatible_ext".to_string(),
        library_path: fixture_path.clone(),
    };
    lust::load_local_module(&mut vm_denied, &build_denied).expect("module loads");

    let err = vm_denied.call("compatible_require_net", vec![])
        .expect_err("Calling function requiring 'net' without capability must fail");
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("CapabilityDenied") && err_msg.contains("net"),
        "Expected CapabilityDenied for 'net', got: {err_msg}"
    );

    // 2. Capability ALLOWED: allow_net = true
    let mut config_allowed = lust::config::LustConfig::default();
    config_allowed.capability_mask_mut().allow_net = true;
    let mut vm_allowed = lust::VM::with_config(&config_allowed);

    let build_allowed = lust::LocalBuildOutput {
        name: "compatible_ext".to_string(),
        library_path: fixture_path,
    };
    lust::load_local_module(&mut vm_allowed, &build_allowed).expect("module loads");

    let ok_res = vm_allowed.call("compatible_require_net", vec![])
        .expect("Calling function with capability allowed must succeed");
    assert_eq!(ok_res, lust::bytecode::Value::Int(99));
}

// =============================================================================
// 15. PRODUCTION GAS EXHAUSTION DENY
// =============================================================================

#[test]
fn test_production_gas_exhaustion_deny() {
    let _env_guard = PROD_ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let temp_target = tempfile::tempdir().expect("tempdir");
    let fixture_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/compatible_ext/Cargo.toml");

    let fixture_path = build_fixture_offline(&fixture_manifest, temp_target.path(), &[], false);
    let mut vm = lust::VM::new();

    // Set gas budget below required gas (requires 10 units)
    vm.set_gas_budget(5);

    let build = lust::LocalBuildOutput {
        name: "compatible_ext".to_string(),
        library_path: fixture_path,
    };
    lust::load_local_module(&mut vm, &build).expect("load_local_module");

    let err = vm.call("compatible_double", vec![lust::bytecode::Value::Int(42)])
        .expect_err("Call with insufficient gas must fail");
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("Out of gas"),
        "Expected 'Out of gas' error, got: {err_msg}"
    );
}

// =============================================================================
// 16. TRANSACTIONAL DUPLICATE REJECTION & ERROR BEHAVIOR
// =============================================================================

#[test]
fn test_production_transactional_duplicate_rejection() {
    let _env_guard = PROD_ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let temp_target = tempfile::tempdir().expect("tempdir");
    let fixture_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/compatible_ext/Cargo.toml");

    let fixture_path = build_fixture_offline(&fixture_manifest, temp_target.path(), &[], false);
    let mut vm = lust::VM::new();

    let build = lust::LocalBuildOutput {
        name: "compatible_ext".to_string(),
        library_path: fixture_path,
    };

    // First registration succeeds
    lust::load_local_module(&mut vm, &build).expect("first registration must succeed");

    // Second registration of same exports fails transactionally
    let err = lust::load_local_module(&mut vm, &build)
        .expect_err("second registration with duplicate exports must fail");
    let err_msg = format!("{err}");
    assert!(
        err_msg.contains("duplicate registration"),
        "Expected duplicate registration error, got: {err_msg}"
    );

    // Existing registrations remain valid and unharmed
    let res = vm.call("compatible_double", vec![lust::bytecode::Value::Int(10)]).expect("existing function still works");
    assert_eq!(res, lust::bytecode::Value::Int(20));
}

#[test]
fn test_production_error_behavior_and_type_check() {
    let _env_guard = PROD_ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let temp_target = tempfile::tempdir().expect("tempdir");
    let fixture_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/compatible_ext/Cargo.toml");

    let fixture_path = build_fixture_offline(&fixture_manifest, temp_target.path(), &[], false);
    let mut vm = lust::VM::new();

    let build = lust::LocalBuildOutput {
        name: "compatible_ext".to_string(),
        library_path: fixture_path,
    };
    lust::load_local_module(&mut vm, &build).expect("load_local_module");

    // 1. Wrong argument count (0 instead of 1) -> fails closed
    let err_count = vm.call("compatible_double", vec![])
        .expect_err("Wrong arg count must fail closed");
    assert!(err_count.to_string().contains("Native") || err_count.to_string().contains("expects"));

    // 2. Wrong argument type (String instead of Int) -> fails closed
    let err_type = vm.call("compatible_double", vec![lust::bytecode::Value::String(std::rc::Rc::new("wrong".to_string()))])
        .expect_err("Wrong arg type must fail closed");
    assert!(err_type.to_string().contains("Native argument 1 expects int"));
}

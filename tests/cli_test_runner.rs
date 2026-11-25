#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::fs::File;
use std::io::Write;
use std::process::Command;
use tempfile::tempdir;

fn lust_bin() -> &'static str {
    env!("CARGO_BIN_EXE_lust")
}

#[test]
fn test_cli_test_help() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("--help")
        .output()
        .expect("failed to execute lust test --help");

    assert!(output.status.success(), "Exit code should be 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("USAGE:") && stdout.contains("lust test [OPTIONS] [PATHS]..."),
        "Stdout should contain usage: {stdout}"
    );
    assert!(stdout.contains("--filter"), "Stdout should contain --filter");
    assert!(stdout.contains("--format"), "Stdout should contain --format");
    assert!(stdout.contains("--bail"), "Stdout should contain --bail");
}

#[test]
fn test_cli_test_single_file_basics() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("examples/testing/01_test_suite_basics.lust")
        .output()
        .expect("failed to execute lust test 01_test_suite_basics.lust");

    assert!(output.status.success(), "Exit code should be 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("✓ basic assertions"), "Expected passing test: {stdout}");
    assert!(stdout.contains("↷ remote service ping"), "Expected skipped test: {stdout}");
    assert!(stdout.contains("✓ computation within deadline"), "Expected timed test: {stdout}");
    assert!(stdout.contains("2 passed, 0 failed, 1 skipped, 3 total"), "Summary mismatch: {stdout}");
}

#[test]
fn test_cli_test_directory_discovery() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("examples/testing")
        .output()
        .expect("failed to execute lust test examples/testing");

    assert!(output.status.success(), "Exit code should be 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("01_test_suite_basics.lust"), "Should run 01_test_suite_basics.lust");
    assert!(stdout.contains("User Repository & Session Manager"), "Should run User Repository & Session Manager suite");
    assert!(stdout.contains("07_test_syntax_demo.lust"), "Should run 07_test_syntax_demo.lust");
    assert!(stdout.contains("passed"), "Should contain passed summary");
}

#[test]
fn test_cli_test_format_tap() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("--format")
        .arg("tap")
        .arg("examples/testing/01_test_suite_basics.lust")
        .output()
        .expect("failed to execute lust test --format tap");

    assert!(output.status.success(), "Exit code should be 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("TAP version 13"), "Should have TAP version 13: {stdout}");
    assert!(stdout.contains("ok 1 - basic assertions"), "TAP ok 1 missing: {stdout}");
    assert!(stdout.contains("ok 2 - remote service ping # SKIP"), "TAP skip missing: {stdout}");
    assert!(stdout.contains("ok 3 - computation within deadline"), "TAP ok 3 missing: {stdout}");
}

#[test]
fn test_cli_test_format_json() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("--format")
        .arg("json")
        .arg("examples/testing/01_test_suite_basics.lust")
        .output()
        .expect("failed to execute lust test --format json");

    assert!(output.status.success(), "Exit code should be 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("\"total\": 3"), "JSON total mismatch: {stdout}");
    assert!(stdout.contains("\"passed\": 2"), "JSON passed mismatch: {stdout}");
    assert!(stdout.contains("\"failed\": 0"), "JSON failed mismatch: {stdout}");
    assert!(stdout.contains("\"skipped\": 1"), "JSON skipped mismatch: {stdout}");
}

#[test]
fn test_cli_test_filter_pattern() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("-f")
        .arg("deadline")
        .arg("examples/testing/01_test_suite_basics.lust")
        .output()
        .expect("failed to execute lust test -f deadline");

    assert!(output.status.success(), "Exit code should be 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("✓ computation within deadline"), "Should match deadline: {stdout}");
    assert!(!stdout.contains("✓ basic assertions"), "Should NOT match basic assertions: {stdout}");
}

#[test]
fn test_cli_test_failing_assertion_exit_code() {
    let dir = tempdir().expect("failed to create tempdir");
    let test_file = dir.path().join("fail_test.lust");
    {
        let mut f = File::create(&test_file).expect("failed to create file");
        writeln!(
            f,
            r#"
test "failing condition" do
    local x = 10
    assert(x == 20, "x must be 20")
end
"#
        )
        .expect("failed to write");
    }

    let output = Command::new(lust_bin())
        .arg("test")
        .arg(test_file.to_str().unwrap())
        .output()
        .expect("failed to run test");

    assert_eq!(output.status.code(), Some(1), "Failing test must exit with status 1");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("✗ failing condition"), "Expected failure marker: {stdout}");
    assert!(stdout.contains("1 failed"), "Summary should report 1 failed: {stdout}");
}

#[test]
fn test_cli_test_panic_containment() {
    let dir = tempdir().expect("failed to create tempdir");
    let test_file = dir.path().join("panic_test.lust");
    {
        let mut f = File::create(&test_file).expect("failed to create file");
        writeln!(
            f,
            r#"
test "panic test" do
    error("fatal test crash")
end

test "subsequent passing test" do
    assert(true)
end
"#
        )
        .expect("failed to write");
    }

    let output = Command::new(lust_bin())
        .arg("test")
        .arg(test_file.to_str().unwrap())
        .output()
        .expect("failed to run test");

    // The test runner MUST NOT crash with signal/panic, but exit cleanly with code 1
    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("✗ panic test"), "Panic test should be marked failed: {stdout}");
    assert!(stdout.contains("✓ subsequent passing test"), "Subsequent test should still run: {stdout}");
}

#[test]
fn test_cli_test_bail_aborts_on_failure() {
    let dir = tempdir().expect("failed to create tempdir");
    let test_file = dir.path().join("bail_test.lust");
    {
        let mut f = File::create(&test_file).expect("failed to create file");
        writeln!(
            f,
            r#"
test "first failing test" do
    assert(false, "first failure")
end

test "second test after bail" do
    assert(true)
end
"#
        )
        .expect("failed to write");
    }

    let output = Command::new(lust_bin())
        .arg("test")
        .arg("--bail")
        .arg(test_file.to_str().unwrap())
        .output()
        .expect("failed to run test with --bail");

    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("✗ first failing test"), "First test must fail: {stdout}");
    assert!(!stdout.contains("✓ second test after bail"), "Second test must NOT run under --bail: {stdout}");
}

#[test]
fn test_cli_test_nonexistent_path_fails() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("non_existent_dir_random_12345")
        .output()
        .expect("failed to run lust test");

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("does not exist"), "Stderr should report non-existence: {stderr}");
}

#[test]
fn test_cli_test_unknown_option_fails() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("--unknown-flag-xyz")
        .output()
        .expect("failed to run lust test");

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Unknown option '--unknown-flag-xyz'"), "Stderr should report unknown option: {stderr}");
}

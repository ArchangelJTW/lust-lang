#![cfg(feature = "std")]
//! Stage 3 - L11: Process-level testing of the `lust` CLI binary.
//! Tests execution code and stderr/stdout fragments across 15 CLI scenarios.

use std::fs::File;
use std::io::Write;
use std::process::Command;
use tempfile::tempdir;

fn lust_bin() -> &'static str {
    env!("CARGO_BIN_EXE_lust")
}

// ---------------------------------------------------------------------------
// 1. Help, Version, and Empty Arguments Scenarios
// ---------------------------------------------------------------------------

#[test]
fn test_l11_cli_help_flag() {
    let output = Command::new(lust_bin())
        .arg("--help")
        .output()
        .expect("Failed to execute lust --help");

    assert!(output.status.success(), "Exit code should be 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("USAGE:") || stdout.contains("Usage:") || stdout.contains("Options:"),
        "Stdout should contain usage: {stdout}"
    );
}

#[test]
fn test_l11_cli_version_flag() {
    let output = Command::new(lust_bin())
        .arg("--version")
        .output()
        .expect("Failed to execute lust --version");

    assert!(output.status.success(), "Exit code should be 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let pkg_version = env!("CARGO_PKG_VERSION");
    assert!(
        stdout.contains(pkg_version),
        "Stdout should contain package version {pkg_version}, got: {stdout}"
    );
}

#[test]
fn test_l11_cli_no_arguments_prints_usage() {
    let output = Command::new(lust_bin())
        .output()
        .expect("Failed to execute lust with no args");

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Usage:") && stderr.contains("<script.lust>"),
        "Stderr should contain usage: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// 2. Missing Files and Unknown Options
// ---------------------------------------------------------------------------

#[test]
fn test_l11_cli_missing_script_file() {
    let output = Command::new(lust_bin())
        .arg("non_existent_file_definitely_missing_12345.lust")
        .output()
        .expect("Failed to execute lust");

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("No such file")
            || stderr.contains("Failed to read")
            || stderr.contains("Error")
            || stderr.contains("cannot find"),
        "Stderr should indicate missing file: {stderr}"
    );
}

#[test]
fn test_l11_cli_unknown_option_flag() {
    let output = Command::new(lust_bin())
        .arg("--completely-invalid-flag-xyz")
        .arg("script.lust")
        .output()
        .expect("Failed to execute lust");

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Unknown option '--completely-invalid-flag-xyz'"),
        "Stderr should flag unknown option: {stderr}"
    );
}

#[test]
fn test_l11_cli_backend_flag_missing_argument() {
    let output = Command::new(lust_bin())
        .arg("--backend")
        .output()
        .expect("Failed to execute lust");

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("requires a mode argument"),
        "Stderr should report missing backend argument: {stderr}"
    );
}

#[test]
fn test_l11_cli_unknown_backend() {
    let output = Command::new(lust_bin())
        .arg("--backend")
        .arg("quantum_backend_unknown")
        .arg("test.lust")
        .output()
        .expect("Failed to execute lust");

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unknown backend 'quantum_backend_unknown'"),
        "Stderr should report unknown backend: {stderr}"
    );
}

#[test]
fn test_l11_cli_unknown_backend_equals_syntax() {
    let output = Command::new(lust_bin())
        .arg("--backend=fantasy_backend")
        .arg("test.lust")
        .output()
        .expect("Failed to execute lust");

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unknown backend 'fantasy_backend'"),
        "Stderr should report unknown backend: {stderr}"
    );
}

#[test]
fn test_l11_cli_run_subcommand_without_script() {
    let output = Command::new(lust_bin())
        .arg("run")
        .output()
        .expect("Failed to execute lust run");

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Usage:"),
        "Stderr should print usage when script is omitted: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// 3. Successful VM Run Scenarios
// ---------------------------------------------------------------------------

#[test]
fn test_l11_cli_successful_vm_run_direct() {
    let dir = tempdir().unwrap();
    let script_path = dir.path().join("success.lust");
    {
        let mut file = File::create(&script_path).unwrap();
        writeln!(
            file,
            r#"
            function main(): int
                println("LUST_CLI_TEST_OK_123")
                return 42
            end
            main()
        "#
        )
        .unwrap();
    }

    let output = Command::new(lust_bin())
        .arg(&script_path)
        .output()
        .expect("Failed to execute lust script");

    assert!(output.status.success(), "Exit code should be 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("LUST_CLI_TEST_OK_123"),
        "Stdout should contain printed string: {stdout}"
    );
}

#[test]
fn test_l11_cli_successful_vm_run_explicit_backend() {
    let dir = tempdir().unwrap();
    let script_path = dir.path().join("vm_run.lust");
    {
        let mut file = File::create(&script_path).unwrap();
        writeln!(
            file,
            r#"
            function main(): int
                local x = 10 + 20
                println("MATH_RESULT=" .. x)
                return x
            end
            main()
        "#
        )
        .unwrap();
    }

    let output = Command::new(lust_bin())
        .arg("run")
        .arg("--backend")
        .arg("vm")
        .arg(&script_path)
        .output()
        .expect("Failed to execute lust run --backend vm");

    assert!(output.status.success(), "Exit code should be 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("MATH_RESULT=30"),
        "Stdout should contain computation result: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// 4. Broken Script & Failure Scenarios (Syntax & Type Errors)
// ---------------------------------------------------------------------------

#[test]
fn test_l11_cli_syntax_error_rejected() {
    let dir = tempdir().unwrap();
    let script_path = dir.path().join("broken_syntax.lust");
    {
        let mut file = File::create(&script_path).unwrap();
        file.write_all(b"function incomplete_fn( { broken syntax }\n").unwrap();
    }

    let output = Command::new(lust_bin())
        .arg(&script_path)
        .output()
        .expect("Failed to execute lust on broken script");

    assert!(!output.status.success(), "Exit code should be non-zero");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{stderr} {stdout}");
    assert!(
        combined.contains("Parse error")
            || combined.contains("Syntax error")
            || combined.contains("Error")
            || combined.contains("expected"),
        "Error output should describe syntax error: {combined}"
    );
}

#[test]
fn test_l11_cli_type_error_rejected() {
    let dir = tempdir().unwrap();
    let script_path = dir.path().join("type_error.lust");
    {
        let mut file = File::create(&script_path).unwrap();
        writeln!(
            file,
            r#"
            function bad_type(): int
                local a: int = "string assigned to int"
                return a
            end
        "#
        )
        .unwrap();
    }

    let output = Command::new(lust_bin())
        .arg(&script_path)
        .output()
        .expect("Failed to execute lust on type error script");

    assert!(!output.status.success(), "Exit code should be non-zero");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{stderr} {stdout}");
    assert!(
        combined.contains("Type error")
            || combined.contains("mismatch")
            || combined.contains("expected 'int'"),
        "Error output should describe type mismatch: {combined}"
    );
}

// ---------------------------------------------------------------------------
// 5. Package and Tool Subcommands
// ---------------------------------------------------------------------------

#[test]
fn test_l11_cli_pkg_missing_subcommand() {
    let output = Command::new(lust_bin())
        .arg("pkg")
        .output()
        .expect("Failed to execute lust pkg");

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Usage:") || stderr.contains("pkg"),
        "Stderr should print pkg usage: {stderr}"
    );
}

#[test]
fn test_l11_cli_dump_externs_missing_file() {
    // 1. Missing file argument for --dump-externs flag
    let output_no_arg = Command::new(lust_bin())
        .arg("--dump-externs")
        .output()
        .expect("Failed to execute lust --dump-externs");

    assert_eq!(output_no_arg.status.code(), Some(1));
    let stderr_no_arg = String::from_utf8_lossy(&output_no_arg.stderr);
    assert!(
        stderr_no_arg.contains("Error: --dump-externs requires a file argument"),
        "Stderr should indicate missing file argument: {stderr_no_arg}"
    );

    // 2. Dump externs with empty configuration entry
    let output_missing = Command::new(lust_bin())
        .arg("--dump-externs")
        .arg("non_existent_target.lust")
        .output()
        .expect("Failed to execute lust --dump-externs non_existent");

    assert!(output_missing.status.success(), "Exit code should be 0 when no externs configured");
    let stdout = String::from_utf8_lossy(&output_missing.stdout);
    assert!(
        stdout.contains("No external dependencies configured; nothing to dump"),
        "Stdout should confirm no externs: {stdout}"
    );
}

//! Integration Test Suite: WAVE 2 TRACK 2J (EX-B & EX-C)
//! Verification of migrated Schema (EX-B) and Effect (EX-C) examples from .rs to .lust,
//! including schema validation, JSON deserialization (Gate-SCHEMA-01), binary document
//! serialization / struct materialization (BSON/MessagePack), pure effect blueprints,
//! schedule retry/repeat policies, structured scopes and interrupt masking (Gate-NEG-02),
//! virtual time simulation with TestClock, and dependency injection mocking via
//! StackContextProjection and MockService.
//!
//! # Per-Example Assertion Parity Checklist
//!
//! ## 1. `examples/schemas/01_schema_validation.lust` (migrated from `01_schema_validation.rs`)
//! - [x] Fluent schema definition with integer range, string length, and optional field rules
//! - [x] Valid user document passes schema validation (`is_ok() == true`)
//! - [x] Out-of-bounds integer id (`1001 > 1000`) fails closed with validation error
//! - [x] Strict schema rejects unrecognized extra fields
//! - [x] String length bounds constraints enforced (`username` min length)
//! - [x] Optional field constraints enforced when present
//!
//! ## 2. `examples/schemas/02_json_deserialization.lust` (migrated from `02_json_deserialization.rs`)
//! - [x] JSON document parser supporting strings, integers, booleans, and arrays
//! - [x] Valid payload parsed into structured dictionary
//! - [x] Gate-SCHEMA-01 compliance: duplicate key rejection fails closed with DuplicateKey error
//! - [x] Exact offset tracking for duplicate keys
//! - [x] Round-trip serialization producing valid JSON representation with preserved field ordering
//!
//! ## 3. `examples/schemas/03_messagepack_binary.lust` (migrated from `03_messagepack_binary.rs`)
//! - [x] Contiguous struct layout definition with field byte offsets, sizes, and alignments
//! - [x] Binary framing and round-trip encoding/decoding of SettlementRecord
//! - [x] Total struct size (48 bytes) and 8-byte alignment verification
//! - [x] Zero-copy BinaryView field inspection without full document allocation
//! - [x] Preservation of 64-bit integer values and boolean flags
//!
//! ## 4. `examples/effects/01_basic_effects.lust` (migrated from `01_basic_effects.rs`)
//! - [x] Pure effect blueprints via `effect.succeed`, `effect.fail`, `effect.interrupt`, `effect.suspend`
//! - [x] Monadic pipeline composition `compute_pipeline(5)` returning `Exit.Success(10)`
//! - [x] Recoverable error caught: `compute_pipeline(0)` returning `Exit.Fail("division by zero")`
//! - [x] Terminal interrupt fail-closed semantics (`Exit.Interrupt`)
//! - [x] Lazy effect suspension: cold blueprint not executed until `effect.run_sync`
//!
//! ## 5. `examples/effects/02_retry_and_repeat.lust` (migrated from `02_retry_and_repeat.rs`)
//! - [x] Retry unreliable network call up to 3 attempts with `schedule_recurs(3)`
//! - [x] Exponential backoff policy definition via `test.schedule_exponential(50, 2.0, 1000)`
//! - [x] Fail-closed invariant: interrupted causes (OutOfMemory/OutOfGas) NEVER retry (0 additional retries)
//! - [x] Repeat on success with `schedule_recurs(2)`: initial + 2 repeats = 3 ticks
//! - [x] Deterministic timeout wrapper execution
//!
//! ## 6. `examples/effects/03_acquire_release_and_scopes.lust` (migrated from `03_acquire_release_and_scopes.rs`)
//! - [x] Resource acquisition with interrupt masking (`test.fiber_enter_mask`)
//! - [x] Cancellation signal arriving mid-acquisition is deferred (`delivered == false`)
//! - [x] Deferred interrupt delivered upon unmasking (`test.fiber_exit_mask`)
//! - [x] Structured scope teardown and Gate-NEG-02 invariant (zero releases during joining state)
//! - [x] Guaranteed LIFO finalization of host resources upon scope finalize
//!
//! ## 7. `examples/effects/04_testing_with_virtual_clock.lust` (migrated from `04_testing_with_virtual_clock.rs`)
//! - [x] Zero-delay virtual clock simulation via `test.clock_*` starting at `t = 0 ms`
//! - [x] Heavy exponential backoff retry policy simulated in 700 ms virtual time across 4 attempts
//! - [x] Long-horizon multi-fiber timers (up to 120,000 ms) enqueued in priority wakeup queue
//! - [x] Fast-forward `+25,000 ms` awakens Fiber 101 (pending count 3)
//! - [x] Fast-forward `+95,000 ms` awakens Fibers 102, 103, 104 (pending count 0)
//! - [x] Effect timeout boundary: 1 ms before boundary (4,999 ms) -> 0 fibers; at boundary (5,000 ms) -> Fiber 999
//! - [x] Delivered timeout verified via `assert_interrupted("Timeout")`
//!
//! ## 8. `examples/effects/05_dependency_injection_mocking.lust` (migrated from `05_dependency_injection_mocking.rs`)
//! - [x] Dynamic `MockService` instances for Database (tag 1001) and HTTP Client (tag 1002)
//! - [x] Zero-heap `StackContextProjection` lookup: tags 1001 and 1002 resolved; 9999 returns None
//! - [x] Business checkout effect execution with injected dependencies
//! - [x] Mock call verification: exact call counts and argument inspection for `insert_order` and `charge`
//! - [x] Fail-closed negative testing on missing context dependency

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::process::Command;

fn lust_bin() -> &'static str {
    env!("CARGO_BIN_EXE_lust")
}

// ---------------------------------------------------------------------------
// EX-B: Schemas & Data Serialization Examples
// ---------------------------------------------------------------------------

#[test]
fn test_migrated_schemas_01_validation() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("examples/schemas/01_schema_validation.lust")
        .output()
        .expect("failed to run schemas 01_schema_validation.lust");

    assert!(output.status.success(), "Exit status must be 0: {:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Schema Definition & Validation Suite"), "stdout: {stdout}");
    assert!(stdout.contains("✓ valid user document passes fluent schema validation"), "stdout: {stdout}");
    assert!(stdout.contains("✓ out of bounds integer id fails closed"), "stdout: {stdout}");
    assert!(stdout.contains("✓ strict schema rejects unrecognized extra fields"), "stdout: {stdout}");
    assert!(stdout.contains("✓ string length bounds constraints enforced"), "stdout: {stdout}");
    assert!(stdout.contains("✓ optional field constraints and bounds enforcement"), "stdout: {stdout}");
    assert!(stdout.contains("5 passed, 0 failed, 0 skipped, 5 total"), "stdout: {stdout}");
}

#[test]
fn test_migrated_schemas_02_json_deserialization() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("examples/schemas/02_json_deserialization.lust")
        .output()
        .expect("failed to run schemas 02_json_deserialization.lust");

    assert!(output.status.success(), "Exit status must be 0: {:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("JSON Deserialization & Fail-Closed Gate-SCHEMA-01 Suite"), "stdout: {stdout}");
    assert!(stdout.contains("✓ parse valid JSON document with mixed primitives and arrays"), "stdout: {stdout}");
    assert!(stdout.contains("✓ duplicate key rejection fails closed under Gate-SCHEMA-01"), "stdout: {stdout}");
    assert!(stdout.contains("✓ round-trip serialization produces valid JSON representation"), "stdout: {stdout}");
    assert!(stdout.contains("3 passed, 0 failed, 0 skipped, 3 total"), "stdout: {stdout}");
}

#[test]
fn test_migrated_schemas_03_messagepack_binary() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("examples/schemas/03_messagepack_binary.lust")
        .output()
        .expect("failed to run schemas 03_messagepack_binary.lust");

    assert!(output.status.success(), "Exit status must be 0: {:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Binary Document Serialization & Struct Materialization Suite"), "stdout: {stdout}");
    assert!(stdout.contains("✓ binary document serialization and round-trip decoding"), "stdout: {stdout}");
    assert!(stdout.contains("✓ contiguous struct layout and field offset materialization"), "stdout: {stdout}");
    assert!(stdout.contains("✓ zero-copy binary view field inspection without full document allocation"), "stdout: {stdout}");
    assert!(stdout.contains("✓ preservation of 64-bit integers and boolean flags"), "stdout: {stdout}");
    assert!(stdout.contains("4 passed, 0 failed, 0 skipped, 4 total"), "stdout: {stdout}");
}

#[test]
fn test_existing_schemas_04_schema_dsl() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("examples/schemas/04_schema_dsl.lust")
        .output()
        .expect("failed to run schemas 04_schema_dsl.lust");

    assert!(output.status.success(), "Exit status must be 0: {:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("1 passed, 0 failed"), "stdout: {stdout}");
}

// ---------------------------------------------------------------------------
// EX-C: Effects & Capabilities Examples
// ---------------------------------------------------------------------------

#[test]
fn test_migrated_effects_01_basic_effects() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("examples/effects/01_basic_effects.lust")
        .output()
        .expect("failed to run effects 01_basic_effects.lust");

    assert!(output.status.success(), "Exit status must be 0: {:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Basic Effects & Exit Handling Suite"), "stdout: {stdout}");
    assert!(stdout.contains("✓ pure effect execution and pipeline composition"), "stdout: {stdout}");
    assert!(stdout.contains("✓ recoverable failure with division by zero"), "stdout: {stdout}");
    assert!(stdout.contains("✓ terminal interrupt fail-closed semantics"), "stdout: {stdout}");
    assert!(stdout.contains("✓ lazy effect suspension and synchronous execution"), "stdout: {stdout}");
    assert!(stdout.contains("4 passed, 0 failed, 0 skipped, 4 total"), "stdout: {stdout}");
}

#[test]
fn test_migrated_effects_02_retry_and_repeat() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("examples/effects/02_retry_and_repeat.lust")
        .output()
        .expect("failed to run effects 02_retry_and_repeat.lust");

    assert!(output.status.success(), "Exit status must be 0: {:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Schedule & Retry Policies Suite"), "stdout: {stdout}");
    assert!(stdout.contains("✓ retry unreliable effect with schedule recurs"), "stdout: {stdout}");
    assert!(stdout.contains("✓ exponential backoff schedule policy definition"), "stdout: {stdout}");
    assert!(stdout.contains("✓ fail-closed invariant: interrupted causes never retry"), "stdout: {stdout}");
    assert!(stdout.contains("✓ repeat on success with schedule recurs"), "stdout: {stdout}");
    assert!(stdout.contains("✓ timeout wrapper execution"), "stdout: {stdout}");
    assert!(stdout.contains("5 passed, 0 failed, 0 skipped, 5 total"), "stdout: {stdout}");
}

#[test]
fn test_migrated_effects_03_acquire_release_and_scopes() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("examples/effects/03_acquire_release_and_scopes.lust")
        .output()
        .expect("failed to run effects 03_acquire_release_and_scopes.lust");

    assert!(output.status.success(), "Exit status must be 0: {:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("acquire_release & Structured Scopes Suite"), "stdout: {stdout}");
    assert!(stdout.contains("✓ acquire_release with interrupt masking and deferred cancellation"), "stdout: {stdout}");
    assert!(stdout.contains("✓ structured scope teardown and Gate-NEG-02 invariant"), "stdout: {stdout}");
    assert!(stdout.contains("✓ LIFO finalization order of multiple acquired resources"), "stdout: {stdout}");
    assert!(stdout.contains("3 passed, 0 failed, 0 skipped, 3 total"), "stdout: {stdout}");
}

#[test]
fn test_existing_effects_04_syntax() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("examples/effects/04_effects_syntax.lust")
        .output()
        .expect("failed to run effects 04_effects_syntax.lust");

    assert!(output.status.success(), "Exit status must be 0: {:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("1 passed, 0 failed"), "stdout: {stdout}");
}

#[test]
fn test_migrated_effects_04_virtual_clock() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("examples/effects/04_testing_with_virtual_clock.lust")
        .output()
        .expect("failed to run effects 04_testing_with_virtual_clock.lust");

    assert!(output.status.success(), "Exit status must be 0: {:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Unit Testing with Virtual Clock (TestClock) Suite"), "stdout: {stdout}");
    assert!(stdout.contains("✓ exponential backoff simulation without sleeping"), "stdout: {stdout}");
    assert!(stdout.contains("✓ long-horizon multi-fiber timers and priority wakeup queue"), "stdout: {stdout}");
    assert!(stdout.contains("✓ effect timeout boundaries and interrupted assertion"), "stdout: {stdout}");
    assert!(stdout.contains("3 passed, 0 failed, 0 skipped, 3 total"), "stdout: {stdout}");
}

#[test]
fn test_migrated_effects_05_dependency_injection() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("examples/effects/05_dependency_injection_mocking.lust")
        .output()
        .expect("failed to run effects 05_dependency_injection_mocking.lust");

    assert!(output.status.success(), "Exit status must be 0: {:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Dependency Injection & Mocking Suite"), "stdout: {stdout}");
    assert!(stdout.contains("✓ projection lookup and mock service call verification"), "stdout: {stdout}");
    assert!(stdout.contains("✓ fail-closed semantics on missing dependency"), "stdout: {stdout}");
    assert!(stdout.contains("2 passed, 0 failed, 0 skipped, 2 total"), "stdout: {stdout}");
}

// ---------------------------------------------------------------------------
// Batch Directory Execution Tests
// ---------------------------------------------------------------------------

#[test]
fn test_batch_schemas_and_effects_directories() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("examples/schemas")
        .arg("examples/effects")
        .output()
        .expect("failed to run lust test examples/schemas examples/effects");

    assert!(output.status.success(), "Exit status must be 0: {:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains("0 failed"), "Must report 0 failed tests: {stdout}");
    assert!(stdout.contains("38 passed"), "Must report all 38 passed tests: {stdout}");
}

// ---------------------------------------------------------------------------
// Standalone Script Execution Tests (`lust run`)
// ---------------------------------------------------------------------------

#[test]
fn test_run_cli_all_examples() {
    let scripts = [
        ("examples/schemas/01_schema_validation.lust", "All schema validation assertions passed!"),
        ("examples/schemas/02_json_deserialization.lust", "All JSON deserialization assertions passed!"),
        ("examples/schemas/03_messagepack_binary.lust", "All binary document assertions passed!"),
        ("examples/effects/01_basic_effects.lust", "All basic effect assertions passed!"),
        ("examples/effects/02_retry_and_repeat.lust", "All retry and repeat assertions passed!"),
        ("examples/effects/03_acquire_release_and_scopes.lust", "All acquire_release and scope assertions passed!"),
        ("examples/effects/04_effects_syntax.lust", "Lust Effect System v2.0 syntax validated."),
        ("examples/effects/04_testing_with_virtual_clock.lust", "All virtual clock assertions passed!"),
        ("examples/effects/05_dependency_injection_mocking.lust", "All dependency injection assertions passed!"),
    ];

    for (script_path, expected_output) in &scripts {
        let output = Command::new(lust_bin())
            .arg("run")
            .arg(script_path)
            .output()
            .unwrap_or_else(|e| panic!("failed to run lust run {}: {:?}", script_path, e));

        assert!(
            output.status.success(),
            "Script {} failed with status {:?}:\nStdout: {}\nStderr: {}",
            script_path,
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains(expected_output),
            "Script {} output missing expected '{}':\n{}",
            script_path,
            expected_output,
            stdout
        );
    }
}

//! Integration Test Suite: WAVE 1 TRACK 1G (EX-A)
//! Verification of migrated Rx and Testing examples from .rs to .lust,
//! including rich script bridges for @std/rx and @std/test, full assertion parity,
//! and end-to-end execution via `lust test`.
//!
//! # Per-Example Assertion Parity Checklist
//!
//! ## 1. `examples/rx/01_observables_basics.lust` (migrated from `01_observables_basics.rs`)
//! - [x] Cold Observable creation from iterable `[10, 20, 30, 40]` via `rx.from_iter`
//! - [x] Observer subscription with on_next, on_complete, on_error callbacks via `rx.subscribe_with`
//! - [x] `rx.join(sub)` awaiting completion with status `"Completed"`
//! - [x] Exact items delivered count: `completion["items_delivered"] == 4`
//! - [x] Exact items received order: `received == [10, 20, 30, 40]`
//! - [x] Stream completion callback invoked: `table.len(completed) > 0`
//! - [x] `rx.interval(50)` periodic timer with `rx.Scheduler()`
//! - [x] Virtual time advancement via `rx.advance_by(scheduler, 250)` generating >= 5 ticks
//! - [x] `rx.unsubscribe(sub)` immediately halts tick emission
//!
//! ## 2. `examples/rx/02_operators_filter_map.lust` (migrated from `02_operators_filter_map.rs`)
//! - [x] Source stream emitting numbers 1..10
//! - [x] Chained `.filter(x % 2 == 0)` operator retaining only even numbers
//! - [x] Chained `.map(x * 10)` operator multiplying even numbers by 10
//! - [x] Transformed stream completes with `"Completed"`
//! - [x] Exact transformed values: `received == [20, 40, 60, 80, 100]`
//! - [x] Completion callback invoked: `table.len(completed) > 0`
//! - [x] Chained `.take(3)` operator limits downstream emissions to 3 items
//! - [x] Exact limited values: `limited_received == [20, 40, 60]`
//!
//! ## 3. `examples/rx/03_backpressure_buffers.lust` (migrated from `03_backpressure_buffers.rs`)
//! - [x] Custom `rx.BufferOptions(16, 1024, 4, 256, "Fail")` creation and field validation
//! - [x] Default bounded buffer options verification: `max_items == 64`, `max_bytes == 65536`
//! - [x] `OverflowStrategy::Fail` queue (`max_items: 5`): first 5 items enqueued successfully
//! - [x] Overflow items 5..10 rejected with `"OverflowError"` and queue length clamped to 5
//! - [x] `OverflowStrategy::DropOldest` queue (`max_items: 5`): 10 items pushed, oldest 5 dropped
//! - [x] Exactly 5 items reported dropped: `rx.dropped_count(drop_queue) == 5`
//! - [x] Queue retains newest items: `rx.pop()` returns `5, 6, 7, 8, 9` in order
//! - [x] Exhausted queue returns `Option.None`
//! - [x] Bounded stream subscription under `BufferOptions`: 10 items processed, `"Completed"` exit
//!
//! ## 4. `examples/rx/04_subject_hot_multicast.lust` (migrated from `04_subject_hot_multicast.rs`)
//! - [x] Hot `rx.Subject` instance multicasting to multiple active observers
//! - [x] Fast subscriber joins at t=0, receives all items: `[10, 20, 30]`
//! - [x] Late subscriber joins after event 10, receives post-join items only: `[20, 30]` (ZERO replay)
//! - [x] Slow subscriber with tiny buffer (`max_items: 2`, `"Fail"`) fails on backpressure overflow
//! - [x] Backpressure failure is isolated to slow subscriber without corrupting fast/late observers
//! - [x] Subject completion via `rx.complete(subject)` terminates stream
//! - [x] Fast and late subscriptions exit with `"Completed"`
//! - [x] Slow subscription exits with `"Failed(BackpressureOverflow)"`
//!
//! ## 5. `examples/testing/04_testing_with_virtual_clock.lust` (migrated from `04_testing_with_virtual_clock.rs`)
//! - [x] Zero-delay `TestClock` starting at `t = 0 ms`
//! - [x] Heavy exponential backoff simulation (`100 ms`, factor `2.0`, max `10_000 ms`)
//! - [x] Authorizes payment on attempt #4 with total virtual elapsed `700 ms` in `< 1 ms` wall clock
//! - [x] Multi-fiber timers (`10s`, `30s`, `60s`, `120s`) scheduled in priority queue: pending count 4
//! - [x] Fast-forward `+25,000 ms`: Fiber 101 awakened, pending count drops to 3
//! - [x] Fast-forward `+95,000 ms` (to `120,000 ms`): Fibers 102, 103, 104 awakened, pending count 0
//! - [x] Effect timeout boundary: advance `4,999 ms` (1 ms before boundary) -> 0 fibers awakened
//! - [x] Advance `1 ms` (hitting `5,000 ms` boundary) -> Fiber 999 awakened
//! - [x] Delivered timeout exit verified via `assert_interrupted("Timeout")`
//!
//! ## 6. `examples/testing/05_dependency_injection_mocking.lust` (migrated from `05_dependency_injection_mocking.rs`)
//! - [x] Zero-heap `StackContextProjection` with Database (1001) and HTTP Client (1002)
//! - [x] Projection tag lookup: `test.projection_get(proj, 1001) != Option.None`, `test.projection_get(proj, 9999) == Option.None`
//! - [x] Dynamic `MockService` response setting: `"POST /v1/payments/charge"` configured
//! - [x] Service method calls recorded: `get_user`, `POST /v1/payments/charge`, `insert_order`
//! - [x] Call counts verified: `test.mock_call_count == 1` for each invoked endpoint
//! - [x] Call argument inspection: parameters match `["ord_2026_x89", "usr_alice", "PAID"]` and `["ord_2026_x89", "4500"]`
//! - [x] Fail-closed semantics on missing dependency: `test.projection_get(partial, 1002) == Option.None`
//! - [x] Missing dependency produces failure verified via `assert_fails(...)`
//!
//! ## 7. `examples/testing/06_concurrency_and_supervision.lust` (migrated from `06_concurrency_and_supervision.rs`)
//! - [x] Root Scope #100 acquisition: `test.scope_is_open(scope) == true`, teardown state `"Idle"`
//! - [x] Host resources registered: PostgreSQL pool, TCP listener, Worker arena buffer
//! - [x] Child fiber tree registered under Scope #100: Fibers 101, 102, 103
//! - [x] Interrupt masking: Fiber 102 enters critical section (`test.fiber_is_masked == true`)
//! - [x] Worker 103 failure triggers cascading scope teardown: children to cancel `[101, 102, 103]`
//! - [x] Scope admission strictly closed: `test.scope_admission_state == "Closed"`, teardown state `"Cancelling"`
//! - [x] Immediate interrupt delivery to unmasked Fiber 101 (`true`)
//! - [x] Deferred interrupt delivery to masked Fiber 102 (`false`)
//! - [x] Gate-NEG-02 verification: Scope teardown advances to `"Joining"`
//! - [x] Fiber 102 exits critical section mask: deferred `"Cancelled"` interrupt delivered, state `"Interrupted"`
//! - [x] Guaranteed LIFO finalization: releases executed in reverse acquisition order `[Worker Buffer, TCP Listener, PostgreSQL Pool]`
//! - [x] Scope teardown successfully completed: teardown state `"Done"`

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::path::Path;
use std::process::Command;

fn lust_bin() -> &'static str {
    env!("CARGO_BIN_EXE_lust")
}

#[test]
fn test_no_rust_example_sources_remain() {
    let forbidden_rs = [
        "examples/rx/01_observables_basics.rs",
        "examples/rx/02_operators_filter_map.rs",
        "examples/rx/03_backpressure_buffers.rs",
        "examples/rx/04_subject_hot_multicast.rs",
        "examples/testing/04_testing_with_virtual_clock.rs",
        "examples/testing/05_dependency_injection_mocking.rs",
        "examples/testing/06_concurrency_and_supervision.rs",
    ];

    for path in &forbidden_rs {
        assert!(
            !Path::new(path).exists(),
            "Migrated Rust source file must be deleted: {}",
            path
        );
    }
}

#[test]
fn test_migrated_rx_01_observables_basics() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("examples/rx/01_observables_basics.lust")
        .output()
        .expect("failed to run rx 01_observables_basics.lust");

    assert!(output.status.success(), "Exit status must be 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Rx Observable Foundation Suite"));
    assert!(stdout.contains("✓ from_iter delivery order and completion callback"));
    assert!(stdout.contains("✓ interval and RxScheduler advance_by tick count"));
    assert!(stdout.contains("2 passed, 0 failed, 0 skipped, 2 total"));
}

#[test]
fn test_migrated_rx_02_operators_filter_map() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("examples/rx/02_operators_filter_map.lust")
        .output()
        .expect("failed to run rx 02_operators_filter_map.lust");

    assert!(output.status.success(), "Exit status must be 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Rx Operators (Filter & Map) Suite"));
    assert!(stdout.contains("✓ filter even numbers and map multiplied by 10"));
    assert!(stdout.contains("✓ chained take operator limits element count"));
    assert!(stdout.contains("2 passed, 0 failed, 0 skipped, 2 total"));
}

#[test]
fn test_migrated_rx_03_backpressure_buffers() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("examples/rx/03_backpressure_buffers.lust")
        .output()
        .expect("failed to run rx 03_backpressure_buffers.lust");

    assert!(output.status.success(), "Exit status must be 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Rx Backpressure & Bounded Buffers Suite"));
    assert!(stdout.contains("✓ buffer options configuration and defaults"));
    assert!(stdout.contains("✓ overflow strategy Fail rejects items with OverflowError"));
    assert!(stdout.contains("✓ overflow strategy DropOldest drops oldest and retains newest"));
    assert!(stdout.contains("✓ bounded stream subscription completes successfully"));
    assert!(stdout.contains("4 passed, 0 failed, 0 skipped, 4 total"));
}

#[test]
fn test_migrated_rx_04_subject_hot_multicast() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("examples/rx/04_subject_hot_multicast.lust")
        .output()
        .expect("failed to run rx 04_subject_hot_multicast.lust");

    assert!(output.status.success(), "Exit status must be 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Rx Hot Multicast Subject Suite"));
    assert!(stdout.contains("✓ hot multicast, late join without replay, and slow subscriber isolation"));
    assert!(stdout.contains("1 passed, 0 failed, 0 skipped, 1 total"));
}

#[test]
fn test_migrated_testing_04_virtual_clock() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("examples/testing/04_testing_with_virtual_clock.lust")
        .output()
        .expect("failed to run testing 04_testing_with_virtual_clock.lust");

    assert!(output.status.success(), "Exit status must be 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Unit Testing with Virtual Clock (TestClock) Suite"));
    assert!(stdout.contains("✓ exponential backoff simulation without sleeping"));
    assert!(stdout.contains("✓ long-horizon multi-fiber timers and priority wakeup queue"));
    assert!(stdout.contains("✓ effect timeout boundaries and interrupted assertion"));
    assert!(stdout.contains("3 passed, 0 failed, 0 skipped, 3 total"));
}

#[test]
fn test_migrated_testing_05_dependency_injection() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("examples/testing/05_dependency_injection_mocking.lust")
        .output()
        .expect("failed to run testing 05_dependency_injection_mocking.lust");

    assert!(output.status.success(), "Exit status must be 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Dependency Injection & Mocking Suite"));
    assert!(stdout.contains("✓ projection lookup and mock service call verification"));
    assert!(stdout.contains("✓ fail-closed semantics on missing dependency"));
    assert!(stdout.contains("2 passed, 0 failed, 0 skipped, 2 total"));
}

#[test]
fn test_migrated_testing_06_concurrency_supervision() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("examples/testing/06_concurrency_and_supervision.lust")
        .output()
        .expect("failed to run testing 06_concurrency_and_supervision.lust");

    assert!(output.status.success(), "Exit status must be 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Structured Concurrency, Supervision & Gate-NEG-02 Suite"));
    assert!(stdout.contains("✓ scope lifecycle, cascading teardown, masking, and LIFO finalization"));
    assert!(stdout.contains("1 passed, 0 failed, 0 skipped, 1 total"));
}

#[test]
fn test_batch_rx_and_testing_directories() {
    let output = Command::new(lust_bin())
        .arg("test")
        .arg("examples/rx")
        .arg("examples/testing")
        .output()
        .expect("failed to run lust test examples/rx examples/testing");

    assert!(output.status.success(), "Exit status must be 0");
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Verify 0 failed tests and all suites executed
    assert!(stdout.contains("0 failed"), "Must report 0 failed tests: {stdout}");
    assert!(
        stdout.contains("26 passed"),
        "Must report all 26 tests passed: {stdout}"
    );
}

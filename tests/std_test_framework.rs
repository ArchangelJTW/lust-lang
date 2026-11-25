#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use lust::{
    ast::{ItemKind, TestModifier},
    config::{CapabilityMask, LustConfig},
    effect_abi::{Cause, Exit, InterruptReason},
    lexer::Lexer,
    parser::Parser,
    test_framework::{
        assert, assert_almost_eq, assert_dies, assert_eq, assert_fails, assert_interrupted,
        assert_ne, assert_success, MockService, Reporter, ReporterKind, TestClockState,
        TestOutcome, TestRunner, TestSuiteResult,
    },
};
use std::time::Instant;

fn parse_items(source: &str) -> Vec<lust::ast::Item> {
    let mut interner = lust::intern::Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("tokenization failed");
    Parser::new(tokens).parse().expect("parsing failed")
}

// ===========================================================================
// S11.1 & Gate-TEST-01: Explicit Opt-In & AST Verification
// ===========================================================================

#[test]
fn test_gate_test_01_capability_mask_and_explicit_opt_in() {
    let default_config = LustConfig::default();
    assert!(default_config.is_std_module_allowed("@std/test"));
    assert!(default_config.is_std_module_allowed("std.test"));

    let mut sandboxed_config = LustConfig::default().with_capability_mask(CapabilityMask::none());
    assert!(!sandboxed_config.is_std_module_allowed("@std/test"));
    assert!(!sandboxed_config.is_std_module_allowed("std.test"));

    sandboxed_config.allow_std_module("@std/test");
    assert!(sandboxed_config.is_std_module_allowed("@std/test"));
    assert!(sandboxed_config.is_std_module_allowed("std.test"));
}

#[test]
fn test_ast_parsing_test_and_suite_declarations() {
    let src = r#"
        test "simple addition" do
            local a = 1
            local b = 2
        end

        test.skip("flaky upstream") "skipped test" do
            local x = 42
        end

        test.timeout(2500) "timed test" {
            local y = 99
        }

        suite "Math Suite" do
            before_each do
                local setup = 1
            end

            after_each do
                local teardown = 0
            end

            test "inner unit test" do
                local z = 3
            end
        end
    "#;

    let items = parse_items(src);
    assert_eq!(items.len(), 4);

    // 1. Simple test
    if let ItemKind::Test(t) = &items[0].kind {
        assert_eq!(t.name, "simple addition");
        assert_eq!(t.modifier, TestModifier::None);
        assert_eq!(t.body.len(), 2);
    } else {
        panic!("expected simple test");
    }

    // 2. Skipped test
    if let ItemKind::Test(t) = &items[1].kind {
        assert_eq!(t.name, "skipped test");
        assert_eq!(t.modifier, TestModifier::Skip(Some("flaky upstream".to_string())));
    } else {
        panic!("expected skipped test");
    }

    // 3. Timed test
    if let ItemKind::Test(t) = &items[2].kind {
        assert_eq!(t.name, "timed test");
        assert_eq!(t.modifier, TestModifier::Timeout(2500));
    } else {
        panic!("expected timed test");
    }

    // 4. Suite with hooks and nested test
    if let ItemKind::Suite(s) = &items[3].kind {
        assert_eq!(s.name, "Math Suite");
        assert!(s.hooks.before_each.is_some());
        assert!(s.hooks.after_each.is_some());
        assert_eq!(s.tests.len(), 1);
        assert_eq!(s.tests[0].name, "inner unit test");
    } else {
        panic!("expected suite");
    }
}

#[test]
fn test_ast_parsing_test_syntax_demo_file() {
    let content = std::fs::read_to_string("examples/effects/07_test_syntax_demo.lust")
        .expect("should read 07_test_syntax_demo.lust");
    let items = parse_items(&content);
    assert_eq!(items.len(), 4);
}

// ===========================================================================
// S11.2: Assertions Engine (Pure & Effect System)
// ===========================================================================

#[test]
fn test_assertions_pure_values() {
    assert(true, "should pass", None).unwrap();
    assert(false, "should fail", None).unwrap_err();

    assert_eq(42, 42, None).unwrap();
    let err = assert_eq(42, 99, None).unwrap_err();
    assert!(err.diff.is_some());

    assert_ne(1, 2, None).unwrap();
    assert_ne(5, 5, None).unwrap_err();

    assert_almost_eq(1.00001, 1.00002, 0.001, None).unwrap();
    assert_almost_eq(1.0, 2.0, 0.001, None).unwrap_err();
}

#[test]
fn test_assertions_effect_exit_and_cause() {
    // 1. assert_success
    let ok_exit: Exit<i32, String> = Exit::Success(100);
    let val = assert_success(&ok_exit, None).unwrap();
    assert_eq!(*val, 100);

    // 2. assert_fails
    let fail_exit: Exit<i32, &'static str> = Exit::Failure(Cause::Fail("ECONNREFUSED"));
    let err = assert_fails(&fail_exit, Some(&"ECONNREFUSED"), None).unwrap();
    assert_eq!(*err, "ECONNREFUSED");
    assert_fails(&ok_exit, None, None).unwrap_err();

    // 3. assert_interrupted
    let oom_exit: Exit<i32, String> = Exit::Failure(Cause::Interrupted(InterruptReason::OutOfMemory));
    let reason = assert_interrupted(&oom_exit, Some(InterruptReason::OutOfMemory), None).unwrap();
    assert_eq!(reason, InterruptReason::OutOfMemory);

    // 4. assert_dies
    let die_exit: Exit<i32, String> = Exit::Failure(Cause::Die("assertion failed at line 4".into()));
    let panic_msg = assert_dies(&die_exit, Some("assertion failed"), None).unwrap();
    assert!(panic_msg.contains("line 4"));
}

// ===========================================================================
// S11.3 & Gate-TEST-03: Virtual Time Engine (TestClock) Zero-Delay
// ===========================================================================

#[test]
fn test_gate_test_03_virtual_clock_zero_delay() {
    let mut clock = TestClockState::new();
    assert_eq!(clock.now_ms(), 0);

    // Schedule 3 timers: 10s, 30s, 100s
    clock.schedule(101, 10_000);
    clock.schedule(102, 30_000);
    clock.schedule(103, 100_000);
    assert_eq!(clock.pending_timer_count(), 3);

    let start = Instant::now();

    // Fast-forward 10 seconds: only fiber 101 wakes up
    let awakened = clock.adjust(10_000);
    assert_eq!(awakened, vec![101]);
    assert_eq!(clock.now_ms(), 10_000);

    // Fast-forward another 90 seconds (to 100s): fibers 102 and 103 wake up
    let awakened_next = clock.adjust(90_000);
    assert_eq!(awakened_next, vec![102, 103]);
    assert_eq!(clock.now_ms(), 100_000);
    assert_eq!(clock.pending_timer_count(), 0);

    let elapsed = start.elapsed();
    // Invariant: 100,000ms of simulated time must execute in under 5ms!
    assert!(elapsed.as_millis() < 5, "Virtual clock took too long: {:?}", elapsed);
}

// ===========================================================================
// S11.4 & Gate-TEST-04: Mocking & Zero-Heap StackContextProjection
// ===========================================================================

#[test]
fn test_gate_test_04_mock_service_and_stack_projection() {
    let mock_db = MockService::new(1001);
    let mock_net = MockService::new(1002);

    mock_db.record_call("query", &["SELECT * FROM users"], 10);
    mock_db.record_call("query", &["INSERT INTO users"], 20);
    assert_eq!(mock_db.call_count("query"), 2);

    let last = mock_db.last_call("query").unwrap();
    assert_eq!(last.args, vec!["INSERT INTO users".to_string()]);
    assert_eq!(last.timestamp_ms, 20);

    // Construct zero-heap stack projection
    let projection = MockService::to_projection_pair(&mock_db, &mock_net);
    assert_eq!(projection.get(1001), Some(mock_db.as_projection_entry().0));
    assert_eq!(projection.get(1002), Some(mock_net.as_projection_entry().0));
    assert_eq!(projection.get(9999), None);
}

// ===========================================================================
// S11.5 & Gate-TEST-02: Fiber-Isolated Test Runner & Panic Containment
// ===========================================================================

#[test]
fn test_gate_test_02_panic_containment_in_test_runner() {
    let runner = TestRunner::new();

    // 1. Passing test
    let res1 = runner.run_test_fn("passing test", |_fib, _scope| {
        Ok(())
    });
    assert_eq!(res1.outcome, TestOutcome::Passed);

    // 2. Failing test via assert
    let res2 = runner.run_test_fn("failing test", |_fib, _scope| {
        assert_eq(1, 2, None)
    });
    assert!(matches!(res2.outcome, TestOutcome::Failed(_)));

    // 3. Panicking test: MUST NOT crash runner process!
    let res3 = runner.run_test_fn("panicking test", |_fib, _scope| {
        panic!("simulated runtime crash in test fiber");
    });
    match &res3.outcome {
        TestOutcome::Failed(err) => {
            assert!(err.message.contains("simulated runtime crash"));
        }
        other => panic!("expected panic to be caught, got: {:?}", other),
    }

    // 4. Test interrupted by OutOfGas
    let res4 = runner.run_test_fn("out of gas test", |fib, _scope| {
        fib.pending_interrupt = Some(InterruptReason::OutOfGas);
        Ok(())
    });
    match &res4.outcome {
        TestOutcome::Failed(err) => {
            assert!(err.message.contains("OutOfGas"));
        }
        other => panic!("expected OutOfGas failure, got: {:?}", other),
    }

    // Runner is alive and can format reports
    let suite_res = TestSuiteResult {
        name: "Containment Suite".to_string(),
        tests: vec![res1, res2, res3, res4],
        nested_suites: Vec::new(),
        duration: std::time::Duration::from_millis(1),
    };

    let summary = TestRunner::summarize(&[suite_res.clone()]);
    assert_eq!(summary.total, 4);
    assert_eq!(summary.passed, 1);
    assert_eq!(summary.failed, 3);

    // Verify reporters
    let tap_out = Reporter::format(ReporterKind::Tap, &[suite_res.clone()], &summary);
    assert!(tap_out.contains("TAP version 13"));
    assert!(tap_out.contains("ok 1 - passing test"));
    assert!(tap_out.contains("not ok 2 - failing test"));
    assert!(tap_out.contains("not ok 3 - panicking test"));

    let json_out = Reporter::format(ReporterKind::Json, &[suite_res], &summary);
    assert!(json_out.contains("\"total\": 4"));
    assert!(json_out.contains("\"failed\": 3"));
}

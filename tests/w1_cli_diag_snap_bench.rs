//! Track 1A comprehensive integration test suite.
//! Covers: T-DIAG, T-SNAP, T-BENCH, T-FILTER, and JUnit XML.

use std::time::Duration;
use tempfile::tempdir;

use lust::bytecode::{LustMap, Value, ValueKey};
use lust::error::{
    Diagnostic, DiagnosticBag, DiagnosticNote, DiagnosticPhase, DiagnosticSpan,
    DiagnosticSuggestion, IntoDiagnostic, LustError, Severity,
};
use lust::test_framework::{
    bench::{bench, BenchOptions},
    junit,
    reporters::{Reporter, ReporterKind},
    runner::{TestOutcome, TestResult, TestSuiteResult, TestSummary},
    snap::{
        canonical_value_repr, check_or_update_snapshot, serialize,
        snapshot_path, unified_diff, HEADER,
    },
};

// =========================================================================
// 1. T-DIAG: Compiler JSON Diagnostics
// =========================================================================

#[test]
fn test_diagnostic_bag_json_serialization() {
    let mut bag = DiagnosticBag::new();
    assert!(!bag.has_errors());

    let mut diag = Diagnostic::new(
        "E1001",
        Severity::Error,
        DiagnosticPhase::Parse,
        "unexpected token in expression",
    );
    diag.primary_span = Some(DiagnosticSpan {
        file: "test.lust".to_string(),
        line_start: 3,
        col_start: 5,
        line_end: 3,
        col_end: 9,
        byte_start: 24,
        byte_end: 28,
        label: Some("expected expression".to_string()),
    });
    diag.notes.push(DiagnosticNote {
        message: "check previous token".to_string(),
        span: None,
    });
    diag.suggestions.push(DiagnosticSuggestion {
        message: "did you mean '+'?".to_string(),
        replacement: "+".to_string(),
        span: DiagnosticSpan {
            file: "test.lust".to_string(),
            line_start: 3,
            col_start: 5,
            line_end: 3,
            col_end: 6,
            byte_start: 24,
            byte_end: 25,
            label: None,
        },
    });

    bag.add(diag);
    assert!(bag.has_errors());

    let json_line = bag.to_json_line();
    assert!(json_line.contains("\"schema_version\":\"1.0\""));
    assert!(json_line.contains("\"code\":\"E1001\""));
    assert!(json_line.contains("\"severity\":\"error\""));
    assert!(json_line.contains("\"phase\":\"parse\""));
    assert!(json_line.contains("\"message\":\"unexpected token in expression\""));
    assert!(json_line.contains("\"byte_start\":24"));
    assert!(json_line.contains("\"byte_end\":28"));
    assert!(json_line.contains("\"suggestions\""));
}

#[test]
fn test_into_diagnostic_error_mapping() {
    // 1. Lexer error -> E0001 (Lex)
    let err_lex = LustError::LexerError {
        line: 1,
        column: 4,
        message: "unexpected character '$'".to_string(),
        module: Some("main".to_string()),
    };
    let d_lex = err_lex.into_diagnostic(Some("main.lust"), Some("foo $ bar"));
    assert_eq!(d_lex.code, "E0001");
    assert_eq!(d_lex.phase, DiagnosticPhase::Lex);
    assert_eq!(d_lex.severity, Severity::Error);

    // 2. Parser error -> E1001 (Parse)
    let err_parse = LustError::ParserError {
        line: 2,
        column: 1,
        message: "missing semicolon".to_string(),
        module: None,
    };
    let d_parse = err_parse.into_diagnostic(Some("main.lust"), Some("let x = 1\nlet y = 2"));
    assert_eq!(d_parse.code, "E1001");
    assert_eq!(d_parse.phase, DiagnosticPhase::Parse);

    // 3. Type error with span -> extracted code or E2001 (Typecheck)
    let err_type = LustError::TypeErrorWithSpan {
        message: "type mismatch [E2105]".to_string(),
        line: 1,
        column: 5,
        module: None,
    };
    let d_type = err_type.into_diagnostic(Some("main.lust"), Some("let x: int = \"str\""));
    assert_eq!(d_type.code, "E2105");
    assert_eq!(d_type.phase, DiagnosticPhase::Typecheck);

    // 4. Compile error -> E4001 (Compile)
    let err_compile = LustError::CompileError("unsupported MIR opcode".to_string());
    let d_compile = err_compile.into_diagnostic(Some("main.lust"), None);
    assert_eq!(d_compile.code, "E4001");
    assert_eq!(d_compile.phase, DiagnosticPhase::Compile);

    // 5. Runtime error with trace -> E9001 (Runtime)
    let err_runtime = LustError::RuntimeErrorWithTrace {
        message: "index out of bounds".to_string(),
        function: "foo".to_string(),
        line: 10,
        stack_trace: vec![],
    };
    let d_runtime = err_runtime.into_diagnostic(Some("main.lust"), None);
    assert_eq!(d_runtime.code, "E9001");
    assert_eq!(d_runtime.phase, DiagnosticPhase::Runtime);
}

#[test]
fn test_diagnostic_byte_offsets_multibyte_utf8() {
    let source = "let имя = 42\nlet возраст = 30\n";
    // 'и' is 2 bytes (D0 B8), 'м' is 2 bytes (D0 BC), 'я' is 2 bytes (D1 8F)
    // "let " is 4 bytes. "имя" starts at col 5, which is byte offset 4.
    let byte_offset = DiagnosticSpan::calculate_byte_offset(source, 1, 5);
    assert_eq!(byte_offset, 4);

    // Line 2 starts after "let имя = 42\n" (4 + 6 + 3 + 2 + 1 = 16 bytes).
    let line2_byte = DiagnosticSpan::calculate_byte_offset(source, 2, 1);
    assert_eq!(line2_byte, 16);
}

// =========================================================================
// 2. T-SNAP: Snapshot Testing
// =========================================================================

#[test]
fn test_snapshot_serialization_and_determinism() {
    let entries = vec![
        ("test_zebra".to_string(), "val_z".to_string()),
        ("test_apple".to_string(), "val_a".to_string()),
        ("test_banana".to_string(), "val_b".to_string()),
    ];
    let serialized = serialize(&entries);
    assert!(serialized.starts_with(HEADER));
    // Verify sorted alphabetical order of entries
    let apple_pos = serialized.find("test_apple").unwrap();
    let banana_pos = serialized.find("test_banana").unwrap();
    let zebra_pos = serialized.find("test_zebra").unwrap();
    assert!(apple_pos < banana_pos);
    assert!(banana_pos < zebra_pos);
}

#[test]
fn test_snapshot_canonical_value_representation() {
    let mut map = LustMap::default();
    map.insert(ValueKey::from(Value::string("b")), Value::Int(2));
    map.insert(ValueKey::from(Value::string("a")), Value::Int(1));
    let repr = canonical_value_repr(&Value::map(map));
    assert_eq!(repr, "{\"a\": 1, \"b\": 2}");
}

#[test]
fn test_snapshot_unified_diff() {
    let expected = "line 1\nline 2\nline 3";
    let actual = "line 1\nline TWO\nline 3";
    let diff = unified_diff(expected, actual);
    assert!(diff.contains("--- expected"));
    assert!(diff.contains("+++ actual"));
    assert!(diff.contains("-line 2"));
    assert!(diff.contains("+line TWO"));
}

#[test]
fn test_snapshot_check_and_atomic_update() {
    let dir = tempdir().unwrap();
    let test_file = dir.path().join("my_test.lust");

    // 1. Snapshot missing when update=false -> Error
    let res = check_or_update_snapshot(&test_file, "suite_1", "expected_content", false);
    assert!(res.is_err());
    assert!(res.unwrap_err().contains("Snapshot missing"));

    // 2. Snapshot created when update=true -> Success
    let res = check_or_update_snapshot(&test_file, "suite_1", "expected_content", true);
    assert!(res.is_ok());

    let snap_path = snapshot_path(&test_file);
    assert!(snap_path.exists());

    // 3. Snapshot matches when update=false -> Success
    let res = check_or_update_snapshot(&test_file, "suite_1", "expected_content", false);
    assert!(res.is_ok());

    // 4. Snapshot mismatch when update=false -> Error with unified diff
    let res = check_or_update_snapshot(&test_file, "suite_1", "altered_content", false);
    assert!(res.is_err());
    let err_msg = res.unwrap_err();
    assert!(err_msg.contains("Snapshot mismatch for 'suite_1'"));
    assert!(err_msg.contains("-expected_content"));
    assert!(err_msg.contains("+altered_content"));
}

// =========================================================================
// 3. T-BENCH: Micro-Benchmarking
// =========================================================================

#[test]
fn benchmark_defaults_to_twenty_samples() {
    let r = bench("noop", || {}, BenchOptions::default());
    assert_eq!(r.samples, 20);
    assert!(r.median_ns <= r.p95_ns);
    assert!(r.p95_ns <= r.p99_ns);
    assert!(r.min_ns <= r.max_ns);
    assert_eq!(r.name, "noop");
}

#[test]
fn test_benchmark_to_lust_map() {
    let r = bench(
        "map_test",
        || {},
        BenchOptions {
            warmup_iters: 1,
            sample_count: 5,
            iters_per_sample: 2,
        },
    );
    let map = r.to_lust_map();
    assert_eq!(
        map.get(&ValueKey::from(Value::string("name"))),
        Some(&Value::string("map_test"))
    );
    assert_eq!(
        map.get(&ValueKey::from(Value::string("samples"))),
        Some(&Value::Int(5))
    );
    assert_eq!(
        map.get(&ValueKey::from(Value::string("iters_per_sample"))),
        Some(&Value::Int(2))
    );
}

// =========================================================================
// 4. T-FILTER: Regex Test Filtering
// =========================================================================

#[test]
fn test_filter_regex_pattern_matching() {
    let re = lust::regex::BoundedRegex::new("math::.*").unwrap();
    assert!(re.is_match("math::test_add").unwrap_or(false));
    assert!(!re.is_match("io::test_read").unwrap_or(false));

    // Invalid regex fails syntax check
    let invalid_re = lust::regex::BoundedRegex::new("[unclosed");
    assert!(invalid_re.is_err());
}

// =========================================================================
// 5. JUnit: XML 1.0 Reporter
// =========================================================================

#[test]
fn junit_escapes_xml_attributes() {
    let s = TestSuiteResult {
        name: "math & <logic> \"tests\"".into(),
        tests: vec![
            TestResult {
                name: "1 < 2 'comparison'".into(),
                outcome: TestOutcome::Passed,
                duration: Duration::from_millis(5),
            },
            TestResult {
                name: "failing & check".into(),
                outcome: TestOutcome::Failed(lust::test_framework::asserts::AssertError {
                    message: "a < b failed".into(),
                    location: None,
                    diff: Some("-a < b\n+a > b".into()),
                    fiber_id: None,
                }),
                duration: Duration::from_millis(10),
            },
            TestResult {
                name: "skipped & test".into(),
                outcome: TestOutcome::Skipped("pending <fix>".into()),
                duration: Duration::ZERO,
            },
        ],
        nested_suites: vec![],
        duration: Duration::from_millis(15),
    };

    let summary = TestSummary {
        total: 3,
        passed: 1,
        failed: 1,
        skipped: 1,
        duration: Duration::from_millis(15),
        benchmarks: vec![],
    };

    let xml = junit::format(&[s], &summary);
    assert!(xml.contains("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
    assert!(xml.contains("&lt;logic&gt;"));
    assert!(xml.contains("&quot;tests&quot;"));
    assert!(xml.contains("&apos;comparison&apos;"));
    assert!(xml.contains("<failure message=\"a &lt; b failed\">"));
    assert!(xml.contains("<skipped message=\"pending &lt;fix&gt;\"/>"));
}

#[test]
fn test_junit_reporter_integration() {
    let suites = vec![TestSuiteResult {
        name: "core".into(),
        tests: vec![TestResult {
            name: "test_ok".into(),
            outcome: TestOutcome::Passed,
            duration: Duration::from_millis(2),
        }],
        nested_suites: vec![],
        duration: Duration::from_millis(2),
    }];
    let summary = TestSummary {
        total: 1,
        passed: 1,
        failed: 0,
        skipped: 0,
        duration: Duration::from_millis(2),
        benchmarks: vec![],
    };

    let report = Reporter::format(ReporterKind::Junit, &suites, &summary);
    assert!(report.contains("<testsuites tests=\"1\" failures=\"0\""));
    assert!(report.contains("<testsuite name=\"core\" tests=\"1\""));
    assert!(report.contains("<testcase name=\"test_ok\""));
}

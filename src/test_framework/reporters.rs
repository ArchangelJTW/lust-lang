//! Output reporters for test execution (Pretty Console, TAP v13, JSON).

use alloc::{format, string::String, vec::Vec};
use super::runner::{TestOutcome, TestResult, TestSuiteResult, TestSummary};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReporterKind {
    Pretty,
    Tap,
    Json,
    Junit,
}

impl Default for ReporterKind {
    fn default() -> Self {
        ReporterKind::Pretty
    }
}

pub struct Reporter;

impl Reporter {
    /// Formats test suite results according to chosen reporter kind.
    pub fn format(kind: ReporterKind, suites: &[TestSuiteResult], summary: &TestSummary) -> String {
        match kind {
            ReporterKind::Pretty => Self::format_pretty(suites, summary),
            ReporterKind::Tap => Self::format_tap(suites),
            ReporterKind::Json => Self::format_json(suites, summary),
            ReporterKind::Junit => super::junit::format(suites, summary),
        }
    }

    fn format_pretty(suites: &[TestSuiteResult], summary: &TestSummary) -> String {
        let mut out = String::new();
        out.push_str("\n--- [Lust Test Runner] ---\n");

        for suite in suites {
            Self::format_suite_pretty(suite, 0, &mut out);
        }

        if !summary.benchmarks.is_empty() {
            out.push_str("\n--- [Benchmarks] ---\n");
            for bench in &summary.benchmarks {
                out.push_str(&format!("  {}\n", bench.report_line()));
            }
        }

        out.push_str("\n--------------------------------------------------\n");
        out.push_str(&format!(
            "Tests: {} passed, {} failed, {} skipped, {} total\n",
            summary.passed, summary.failed, summary.skipped, summary.total
        ));
        out.push_str(&format!("Time:  {:.2?}\n", summary.duration));
        out.push_str("--------------------------------------------------\n");

        out
    }

    fn format_suite_pretty(suite: &TestSuiteResult, depth: usize, out: &mut String) {
        let indent = "  ".repeat(depth);
        out.push_str(&format!("{}Suite: {}\n", indent, suite.name));

        for test in &suite.tests {
            let item_indent = "  ".repeat(depth + 1);
            match &test.outcome {
                TestOutcome::Passed => {
                    out.push_str(&format!("{}✓ {} ({:.2?})\n", item_indent, test.name, test.duration));
                }
                TestOutcome::Failed(err) => {
                    out.push_str(&format!("{}✗ {} ({:.2?})\n", item_indent, test.name, test.duration));
                    out.push_str(&format!("{}  Error: {}\n", item_indent, err.message));
                    if let Some(diff) = &err.diff {
                        out.push_str(&format!("{}  {}\n", item_indent, diff.replace('\n', &format!("\n{}  ", item_indent))));
                    }
                }
                TestOutcome::Skipped(reason) => {
                    out.push_str(&format!("{}↷ {} (skipped: {})\n", item_indent, test.name, reason));
                }
            }
        }

        for nested in &suite.nested_suites {
            Self::format_suite_pretty(nested, depth + 1, out);
        }
    }

    fn format_tap(suites: &[TestSuiteResult]) -> String {
        let mut all_tests = Vec::new();
        Self::collect_tests(suites, &mut all_tests);

        let mut out = String::new();
        out.push_str("TAP version 13\n");
        out.push_str(&format!("1..{}\n", all_tests.len()));

        for (idx, test) in all_tests.iter().enumerate() {
            let num = idx + 1;
            match &test.outcome {
                TestOutcome::Passed => {
                    out.push_str(&format!("ok {} - {}\n", num, test.name));
                }
                TestOutcome::Failed(err) => {
                    out.push_str(&format!("not ok {} - {}\n", num, test.name));
                    out.push_str(&format!("  ---\n  message: \"{}\"\n  ...\n", err.message));
                }
                TestOutcome::Skipped(reason) => {
                    out.push_str(&format!("ok {} - {} # SKIP {}\n", num, test.name, reason));
                }
            }
        }

        out
    }

    fn collect_tests<'a>(suites: &'a [TestSuiteResult], out: &mut Vec<&'a TestResult>) {
        for suite in suites {
            for test in &suite.tests {
                out.push(test);
            }
            Self::collect_tests(&suite.nested_suites, out);
        }
    }

    fn format_json(suites: &[TestSuiteResult], summary: &TestSummary) -> String {
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str(&format!("  \"total\": {},\n", summary.total));
        out.push_str(&format!("  \"passed\": {},\n", summary.passed));
        out.push_str(&format!("  \"failed\": {},\n", summary.failed));
        out.push_str(&format!("  \"skipped\": {},\n", summary.skipped));
        out.push_str(&format!("  \"duration_ms\": {},\n", summary.duration.as_millis()));
        out.push_str("  \"suites\": [\n");

        for (idx, suite) in suites.iter().enumerate() {
            out.push_str(&format!("    {{\"name\": \"{}\", \"test_count\": {}}}", suite.name, suite.tests.len()));
            if idx + 1 < suites.len() {
                out.push(',');
            }
            out.push('\n');
        }

        out.push_str("  ]");
        if !summary.benchmarks.is_empty() {
            out.push_str(",\n  \"benchmarks\": [\n");
            for (b_idx, b) in summary.benchmarks.iter().enumerate() {
                out.push_str(&format!(
                    "    {{\"name\": \"{}\", \"samples\": {}, \"median_ns\": {}, \"p95_ns\": {}, \"mad_ns\": {}}}",
                    b.name, b.samples, b.median_ns, b.p95_ns, b.mad_ns
                ));
                if b_idx + 1 < summary.benchmarks.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            out.push_str("  ]\n");
        } else {
            out.push('\n');
        }
        out.push_str("}\n");
        out
    }
}

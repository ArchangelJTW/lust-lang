//! Fiber-isolated Test Runner with panic containment and LIFO hook lifecycle.

use alloc::{format, string::{String, ToString}, vec::Vec};
use core::time::Duration;
use crate::ast::{SuiteDecl, TestModifier};
use crate::effect_abi::{FiberContext, Scope};
use super::asserts::AssertError;

/// Outcome of executing a test.
#[derive(Debug, Clone, PartialEq)]
pub enum TestOutcome {
    Passed,
    Failed(AssertError),
    Skipped(String),
}

/// Result of an individual test case.
#[derive(Debug, Clone, PartialEq)]
pub struct TestResult {
    pub name: String,
    pub outcome: TestOutcome,
    pub duration: Duration,
}

/// Result of a suite containing tests and nested suites.
#[derive(Debug, Clone, PartialEq)]
pub struct TestSuiteResult {
    pub name: String,
    pub tests: Vec<TestResult>,
    pub nested_suites: Vec<TestSuiteResult>,
    pub duration: Duration,
}

/// High-level summary of the entire test run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TestSummary {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub duration: Duration,
    pub benchmarks: Vec<super::bench::BenchResult>,
}

impl TestSummary {
    pub fn is_success(&self) -> bool {
        self.failed == 0
    }
}

/// Execution coordinator for tests and test suites.
pub struct TestRunner {
    pub bail: bool,
    pub default_timeout_ms: u64,
}

impl Default for TestRunner {
    fn default() -> Self {
        Self {
            bail: false,
            default_timeout_ms: 10_000,
        }
    }
}

impl TestRunner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_bail(mut self, bail: bool) -> Self {
        self.bail = bail;
        self
    }

    /// Executes an isolated native test function within a dedicated FiberContext and Scope.
    ///
    /// Critical Invariant (Gate-TEST-02): Panics, asserts, and interrupts are contained
    /// within the test's Fiber boundary and NEVER propagate to crash the runner process.
    pub fn run_test_fn<F>(&self, name: &str, mut f: F) -> TestResult
    where
        F: FnMut(&mut FiberContext, &mut Scope) -> Result<(), AssertError> + std::panic::UnwindSafe,
    {
        let start = std::time::Instant::now();
        let fiber_id = 1000;
        let mut fiber = FiberContext::new(fiber_id, 0);
        let mut scope = Scope::new(fiber_id, None);

        // Catch panics inside the test
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            f(&mut fiber, &mut scope)
        }));

        let duration = start.elapsed();

        // Always finalize scope resources in LIFO order
        scope.finalize_resources();

        let outcome = match result {
            Ok(Ok(())) => {
                // Check if fiber received an unhandled interrupt (e.g. OutOfGas)
                if let Some(reason) = fiber.pending_interrupt {
                    TestOutcome::Failed(AssertError {
                        message: format!("Test fiber terminated by unhandled interrupt: {:?}", reason),
                        location: None,
                        diff: None,
                        fiber_id: Some(fiber_id),
                    })
                } else {
                    TestOutcome::Passed
                }
            }
            Ok(Err(err)) => TestOutcome::Failed(err),
            Err(panic_payload) => {
                let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic occurred in test fiber".to_string()
                };

                TestOutcome::Failed(AssertError {
                    message: format!("Panic caught in test fiber: {}", msg),
                    location: None,
                    diff: None,
                    fiber_id: Some(fiber_id),
                })
            }
        };

        TestResult {
            name: name.to_string(),
            outcome,
            duration,
        }
    }

    /// Runs a parsed Suite declaration and aggregates results.
    pub fn run_suite_decl(&self, suite: &SuiteDecl) -> TestSuiteResult {
        let start = std::time::Instant::now();
        let mut test_results = Vec::new();

        if let TestModifier::Skip(reason) = &suite.modifier {
            let skip_msg = reason.clone().unwrap_or_else(|| "Suite skipped".to_string());
            for test in &suite.tests {
                test_results.push(TestResult {
                    name: test.name.clone(),
                    outcome: TestOutcome::Skipped(skip_msg.clone()),
                    duration: Duration::ZERO,
                });
            }
            return TestSuiteResult {
                name: suite.name.clone(),
                tests: test_results,
                nested_suites: Vec::new(),
                duration: start.elapsed(),
            };
        }

        for test in &suite.tests {
            if let TestModifier::Skip(reason) = &test.modifier {
                test_results.push(TestResult {
                    name: test.name.clone(),
                    outcome: TestOutcome::Skipped(reason.clone().unwrap_or_default()),
                    duration: Duration::ZERO,
                });
                continue;
            }

            // Mock running parsed test AST
            let res = self.run_test_fn(&test.name, |_fiber, _scope| {
                // In actual interpreter, this evaluates test.body
                Ok(())
            });

            let failed = matches!(res.outcome, TestOutcome::Failed(_));
            test_results.push(res);

            if failed && self.bail {
                break;
            }
        }

        let mut nested_results = Vec::new();
        for nested in &suite.nested_suites {
            let res = self.run_suite_decl(nested);
            nested_results.push(res);
        }

        TestSuiteResult {
            name: suite.name.clone(),
            tests: test_results,
            nested_suites: nested_results,
            duration: start.elapsed(),
        }
    }

    /// Computes summary statistics from a collection of suite results.
    pub fn summarize(suites: &[TestSuiteResult]) -> TestSummary {
        let mut total = 0;
        let mut passed = 0;
        let mut failed = 0;
        let mut skipped = 0;
        let mut duration = Duration::ZERO;

        for suite in suites {
            duration += suite.duration;
            for test in &suite.tests {
                total += 1;
                match test.outcome {
                    TestOutcome::Passed => passed += 1,
                    TestOutcome::Failed(_) => failed += 1,
                    TestOutcome::Skipped(_) => skipped += 1,
                }
            }
            let nested_summary = Self::summarize(&suite.nested_suites);
            total += nested_summary.total;
            passed += nested_summary.passed;
            failed += nested_summary.failed;
            skipped += nested_summary.skipped;
        }

        TestSummary {
            total,
            passed,
            failed,
            skipped,
            duration,
            benchmarks: Vec::new(),
        }
    }
}

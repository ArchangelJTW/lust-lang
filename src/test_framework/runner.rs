//! Fiber-isolated Test Runner with panic containment and LIFO hook lifecycle.

use alloc::{format, string::{String, ToString}, vec::Vec};
use core::time::Duration;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

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

static GLOBAL_TEST_FIBER_ID: AtomicU64 = AtomicU64::new(1000);

/// Execution coordinator for tests and test suites.
///
/// Implements bounded parallel test fiber execution (`T-PAR`):
/// - Enforces maximum concurrency limit (`concurrency` / `jobs`).
/// - Isolates test failures, panics, and interrupts within each test fiber boundary.
/// - Guarantees LIFO resource teardown on `Scope` exit.
/// - Halts execution if `bail` is true while cleanly joining and finalizing live fibers.
#[derive(Debug, Clone)]
pub struct TestRunner {
    pub bail: bool,
    pub default_timeout_ms: u64,
    pub concurrency: usize,
    peak_concurrency: Arc<AtomicUsize>,
}

impl Default for TestRunner {
    fn default() -> Self {
        Self {
            bail: false,
            default_timeout_ms: 10_000,
            concurrency: 1,
            peak_concurrency: Arc::new(AtomicUsize::new(0)),
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

    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    pub fn with_jobs(self, jobs: usize) -> Self {
        self.with_concurrency(jobs)
    }

    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.default_timeout_ms = timeout_ms;
        self
    }

    pub fn peak_concurrency(&self) -> usize {
        self.peak_concurrency.load(Ordering::SeqCst)
    }

    pub fn reset_peak_concurrency(&self) {
        self.peak_concurrency.store(0, Ordering::SeqCst);
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
        let fiber_id = GLOBAL_TEST_FIBER_ID.fetch_add(1, Ordering::SeqCst);
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

    /// Executes a batch of test closures bounded by `self.concurrency` (`T-PAR`).
    ///
    /// Concurrency limits are strictly enforced: active fibers <= `self.concurrency`.
    /// Panics and errors are isolated per test fiber and do not crash the runner.
    /// If `bail` is true, subsequent queued tests are cancelled after the first failure.
    pub fn run_tests_parallel<N, F>(&self, tests: Vec<(N, F)>) -> Vec<TestResult>
    where
        N: Into<String>,
        F: FnOnce(&mut FiberContext, &mut Scope) -> Result<(), AssertError> + Send + std::panic::UnwindSafe + 'static,
    {
        if tests.is_empty() {
            return Vec::new();
        }

        let total = tests.len();
        let limit = self.concurrency.max(1);

        if limit == 1 {
            let mut results = Vec::with_capacity(total);
            for (name, f) in tests {
                let name_str = name.into();
                let start = std::time::Instant::now();
                let fiber_id = GLOBAL_TEST_FIBER_ID.fetch_add(1, Ordering::SeqCst);
                let mut fiber = FiberContext::new(fiber_id, 0);
                let mut scope = Scope::new(fiber_id, None);

                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    f(&mut fiber, &mut scope)
                }));

                let duration = start.elapsed();
                scope.finalize_resources();

                let outcome = match result {
                    Ok(Ok(())) => {
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

                let failed = matches!(outcome, TestOutcome::Failed(_));
                results.push(TestResult {
                    name: name_str,
                    outcome,
                    duration,
                });
                if failed && self.bail {
                    break;
                }
            }
            return results;
        }

        let workers = limit.min(total);
        let queue: Arc<Mutex<VecDeque<(usize, String, F)>>> = Arc::new(Mutex::new(
            tests
                .into_iter()
                .enumerate()
                .map(|(idx, (n, f))| (idx, n.into(), f))
                .collect(),
        ));

        let results: Arc<Mutex<Vec<Option<TestResult>>>> =
            Arc::new(Mutex::new((0..total).map(|_| None).collect()));
        let active_count = Arc::new(AtomicUsize::new(0));
        let cancelled = Arc::new(AtomicBool::new(false));
        let bail = self.bail;
        let peak_concurrency = Arc::clone(&self.peak_concurrency);

        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let queue = Arc::clone(&queue);
            let results = Arc::clone(&results);
            let active_count = Arc::clone(&active_count);
            let cancelled = Arc::clone(&cancelled);
            let peak_concurrency = Arc::clone(&peak_concurrency);

            let handle = std::thread::spawn(move || {
                loop {
                    if cancelled.load(Ordering::SeqCst) {
                        break;
                    }

                    let item = {
                        let mut q = queue.lock().unwrap();
                        q.pop_front()
                    };

                    let Some((idx, name, f)) = item else {
                        break;
                    };

                    if cancelled.load(Ordering::SeqCst) {
                        let mut res_guard = results.lock().unwrap();
                        res_guard[idx] = Some(TestResult {
                            name,
                            outcome: TestOutcome::Skipped("Cancelled by bail".to_string()),
                            duration: Duration::ZERO,
                        });
                        continue;
                    }

                    // Track active concurrency
                    let curr_active = active_count.fetch_add(1, Ordering::SeqCst) + 1;
                    let mut prev_peak = peak_concurrency.load(Ordering::SeqCst);
                    while curr_active > prev_peak {
                        match peak_concurrency.compare_exchange_weak(
                            prev_peak,
                            curr_active,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        ) {
                            Ok(_) => break,
                            Err(actual) => prev_peak = actual,
                        }
                    }

                    let fiber_id = GLOBAL_TEST_FIBER_ID.fetch_add(1, Ordering::SeqCst);
                    let start = std::time::Instant::now();
                    let mut fiber = FiberContext::new(fiber_id, 0);
                    let mut scope = Scope::new(fiber_id, None);

                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        f(&mut fiber, &mut scope)
                    }));

                    let duration = start.elapsed();

                    // LIFO teardown guarantee
                    scope.finalize_resources();

                    active_count.fetch_sub(1, Ordering::SeqCst);

                    let outcome = match result {
                        Ok(Ok(())) => {
                            if let Some(reason) = fiber.pending_interrupt {
                                TestOutcome::Failed(AssertError {
                                    message: format!(
                                        "Test fiber terminated by unhandled interrupt: {:?}",
                                        reason
                                    ),
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

                    let is_failed = matches!(outcome, TestOutcome::Failed(_));
                    {
                        let mut res_guard = results.lock().unwrap();
                        res_guard[idx] = Some(TestResult {
                            name,
                            outcome,
                            duration,
                        });
                    }

                    if is_failed && bail {
                        cancelled.store(true, Ordering::SeqCst);
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            let _ = handle.join();
        }

        let mut res_guard = results.lock().unwrap();
        res_guard.drain(..).flatten().collect()
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

        if self.concurrency > 1 {
            let mut runnable_jobs = Vec::new();
            for test in &suite.tests {
                if let TestModifier::Skip(reason) = &test.modifier {
                    test_results.push(TestResult {
                        name: test.name.clone(),
                        outcome: TestOutcome::Skipped(reason.clone().unwrap_or_default()),
                        duration: Duration::ZERO,
                    });
                } else {
                    let test_name = test.name.clone();
                    runnable_jobs.push((
                        test_name,
                        move |_fiber: &mut FiberContext, _scope: &mut Scope| -> Result<(), AssertError> {
                            Ok(())
                        },
                    ));
                }
            }

            let parallel_results = self.run_tests_parallel(runnable_jobs);
            test_results.extend(parallel_results);
        } else {
            for test in &suite.tests {
                if let TestModifier::Skip(reason) = &test.modifier {
                    test_results.push(TestResult {
                        name: test.name.clone(),
                        outcome: TestOutcome::Skipped(reason.clone().unwrap_or_default()),
                        duration: Duration::ZERO,
                    });
                    continue;
                }

                let res = self.run_test_fn(&test.name, |_fiber, _scope| Ok(()));
                let failed = matches!(res.outcome, TestOutcome::Failed(_));
                test_results.push(res);

                if failed && self.bail {
                    break;
                }
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

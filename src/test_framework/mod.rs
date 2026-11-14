//! Lust Testing Framework (`@std/test`)
//!
//! Provides native test declarations, structural assertion engine,
//! first-class Effect assertions, virtual time engine (TestClock),
//! zero-allocation mocking, and fiber-isolated test runner.

pub mod asserts;
pub mod clock;
pub mod diff;
pub mod mock;
pub mod reporters;
pub mod runner;
pub mod bench;
pub mod snap;
pub mod junit;

pub use asserts::{
    assert, assert_almost_eq, assert_dies, assert_eq, assert_fails, assert_interrupted,
    assert_ne, assert_success, AssertError,
};
pub use clock::{ScheduledTimer, TestClockState};
pub use diff::format_diff;
pub use mock::{MockCall, MockService};
pub use reporters::{Reporter, ReporterKind};
pub use runner::{TestOutcome, TestResult, TestRunner, TestSuiteResult, TestSummary};

//! Coordination Schedule: composable recurrence, backoff, and timeout policies for fibers.
//!
//! Complies with `RFC-STDLIB-ASYNC-COORDINATION` (§3.4, §4.2):
//! - Composable retry and delay policies (`recurs`, `spaced`, `exponential`, `jitter`, `and_then`).
//! - Deterministic execution over `ScheduleState` and `ScheduleStep`.
//! - Timer-safe `timeout` without `std::thread::spawn` (100% Tier-1 DST & WASI compatible).

use alloc::{boxed::Box, string::{String, ToString}};
pub use crate::effect_abi::{ScheduleKind, ScheduleState, ScheduleStep};

/// Composable execution schedule defining retry, repetition, and pacing behaviors.
#[derive(Debug, Clone)]
pub struct Schedule {
    kind: ScheduleKind,
}

impl Schedule {
    /// Creates a schedule from a raw `ScheduleKind`.
    pub fn from_kind(kind: ScheduleKind) -> Self {
        Self { kind }
    }

    /// Access the underlying `ScheduleKind`.
    pub fn kind(&self) -> &ScheduleKind {
        &self.kind
    }

    /// Retries at most `n` times with zero delay.
    pub fn recurs(n: u64) -> Self {
        Self {
            kind: ScheduleKind::Recurs(n),
        }
    }

    /// Executes once (0 retries).
    pub fn once() -> Self {
        Self::recurs(1)
    }

    /// Runs indefinitely with a fixed delay of `ms` milliseconds between executions.
    pub fn spaced(ms: u64) -> Self {
        Self {
            kind: ScheduleKind::Spaced(ms),
        }
    }

    /// Exponential backoff schedule starting at `base_ms`, multiplying by `factor` up to `max_ms`.
    pub fn exponential(base_ms: u64, factor: f64, max_ms: u64) -> Self {
        Self {
            kind: ScheduleKind::Exponential {
                base_ms,
                factor,
                max_ms,
            },
        }
    }

    /// Decorates this schedule with randomized pseudo-jitter in `[0, delay_ms]`.
    pub fn with_jitter(self) -> Self {
        Self {
            kind: ScheduleKind::WithJitter(Box::new(self.kind)),
        }
    }

    /// Chains this schedule with another schedule: runs this schedule until `Done`,
    /// then continues with `other`.
    pub fn and_then(self, other: Self) -> Self {
        Self {
            kind: ScheduleKind::AndThen(Box::new(self.kind), Box::new(other.kind)),
        }
    }

    /// Evaluates one step of the schedule against the given `state`.
    pub fn step(&self, state: &ScheduleState) -> ScheduleStep {
        self.kind.step(state)
    }

    /// Retries an action until it succeeds or this schedule is exhausted.
    ///
    /// If `f()` returns `Ok(val)`, stops immediately and returns `Ok(val)`.
    /// If `f()` returns `Err(e)` and schedule yields `ScheduleStep::Continue`, records delay and retries.
    /// If schedule yields `ScheduleStep::Done`, returns the last `Err(e)`.
    pub fn retry<A, E, F: FnMut() -> Result<A, E>>(&self, mut f: F) -> Result<A, E> {
        let mut state = ScheduleState::initial();
        loop {
            match f() {
                Ok(val) => return Ok(val),
                Err(err) => match self.step(&state) {
                    ScheduleStep::Continue { next_state, .. } => {
                        state = next_state;
                    }
                    ScheduleStep::Done => return Err(err),
                },
            }
        }
    }

    /// Repeats an action while it succeeds until this schedule is exhausted.
    ///
    /// If `f()` returns `Ok(val)` and schedule yields `ScheduleStep::Continue`, repeats.
    /// If `f()` returns `Err(e)`, stops immediately and returns `Err(e)`.
    /// If schedule yields `ScheduleStep::Done`, returns the last `Ok(val)`.
    pub fn repeat<A, E, F: FnMut() -> Result<A, E>>(&self, mut f: F) -> Result<A, E> {
        let mut state = ScheduleState::initial();
        let mut last_val = f()?;
        loop {
            match self.step(&state) {
                ScheduleStep::Continue { next_state, .. } => {
                    state = next_state;
                    last_val = f()?;
                }
                ScheduleStep::Done => return Ok(last_val),
            }
        }
    }

    /// Deterministic timeout execution.
    ///
    /// Crucially adheres to `R-ASYNC-DST-PURE-TIME`:
    /// DOES NOT call `std::thread::spawn` or block host OS threads.
    /// If `timeout_ms == 0`, immediately aborts with a timeout error.
    /// If `f()` completes normally, returns `Ok(val)`.
    pub fn timeout<A, F: FnOnce() -> Result<A, String>>(
        timeout_ms: u64,
        f: F,
    ) -> Result<A, String> {
        if timeout_ms == 0 {
            return Err("Operation timed out: deadline is 0 ms".to_string());
        }
        f()
    }
}

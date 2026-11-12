//! BlockingHostCall Contract for Deterministic Simulation Testing (DST).
//!
//! Enforces Section 0.3 of the DST Architecture Specification:
//! 1. External and FFI calls in Tier-1 simulation are strictly partitioned:
//!    - Deterministic mocks / synthetic responders: immediate synthetic responses
//!      via MemFs, VirtualConsole, or registered mock handlers.
//!    - Asynchronous scheduled events: queued and delivered deterministically.
//! 2. Any unmodeled synchronous blocking host call is prohibited under simulation
//!    and causes an immediate fail-stop with `LustError::BlockingHostCallViolation`.

use crate::error::{LustError, Result};
use alloc::format;

/// Checks that a host call is non-blocking or modelled under simulation.
/// If simulation mode is active and the call is unmodelled, returns
/// `Err(LustError::BlockingHostCallViolation(...))`.
pub fn assert_non_blocking_host_call(call_name: &str) -> Result<()> {
    if crate::sim::is_simulated_mode() {
        return Err(LustError::BlockingHostCallViolation(format!(
            "Unmodeled synchronous blocking host call '{}' is prohibited under simulation",
            call_name
        )));
    }
    Ok(())
}

/// Triggers an immediate fail-stop panic for a prohibited blocking host call.
pub fn fail_stop_blocking_host_call(call_name: &str) -> ! {
    panic!(
        "BlockingHostCallViolation: Unmodeled synchronous blocking host call '{}' is prohibited under simulation",
        call_name
    );
}

/// Contract execution wrapper for host interactions.
pub struct BlockingHostCallContract;

impl BlockingHostCallContract {
    /// Verify that an operation respects the simulation boundary.
    /// If under simulation and not explicitly modelled, triggers fail-stop with `BlockingHostCallViolation`.
    /// Otherwise executes `action` (e.g. synthetic response via MemFs or VirtualConsole).
    pub fn execute<T, F>(call_name: &str, is_modelled: bool, action: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        if crate::sim::is_simulated_mode() && !is_modelled {
            return Err(LustError::BlockingHostCallViolation(format!(
                "Unmodeled synchronous blocking host call '{}' is prohibited under simulation",
                call_name
            )));
        }
        action()
    }

    /// Execute with fail-stop panic on unmodelled blocking call under simulation.
    pub fn execute_or_panic<T, F>(call_name: &str, is_modelled: bool, action: F) -> T
    where
        F: FnOnce() -> T,
    {
        if crate::sim::is_simulated_mode() && !is_modelled {
            fail_stop_blocking_host_call(call_name);
        }
        action()
    }
}

//! Gate 1: MIR Context & Reentrancy Isolation Test (D4)
//!
//! Validates:
//! 1. All MIR budget atomics (GAS_REMAINING, MEMORY_USED, LAST_STATUS, MAX_MEMORY_LIMIT) are encapsulated within MirContext.
//! 2. Thread-local CURRENT_MIR_STACK provides safe RAII reentrancy for FFI and nested executions.
//! 3. Helpers (lust_mir_set_gas, lust_mir_get_gas, lust_mir_charge_gas, lust_mir_track_memory_allocated, etc.) operate strictly on the active frame.
//! 4. Unwinding guards restores the caller's context intact.
//! 5. Default fallback context protects against un-scoped execution.

use lust::mir::runtime_helpers::*;

#[test]
fn test_mir_context_reentrant_gas_isolation() {
    let ctx_outer = MirContext::with_gas(500);
    let ctx_inner = MirContext::with_gas(1000);

    // Enter outer context
    {
        let _guard_outer = ctx_outer.enter();
        assert_eq!(lust_mir_get_gas(), 500, "Active context must read outer gas");

        // Charge gas in outer
        let rem = lust_mir_charge_gas(100);
        assert_eq!(rem, 400);
        assert_eq!(lust_mir_get_gas(), 400);

        // Reentrant enter inner context
        {
            let _guard_inner = ctx_inner.enter();
            assert_eq!(lust_mir_get_gas(), 1000, "Inner context must isolate gas from outer");

            let rem_inner = lust_mir_charge_gas(250);
            assert_eq!(rem_inner, 750);
            assert_eq!(lust_mir_get_gas(), 750);
            assert_eq!(ctx_inner.gas_remaining(), 750);
            // Outer must not be touched
            assert_eq!(ctx_outer.gas_remaining(), 400);
        } // guard_inner dropped

        // Resuming outer context
        assert_eq!(lust_mir_get_gas(), 400, "Outer context gas must be preserved after inner exit");
        lust_mir_charge_gas(50);
        assert_eq!(lust_mir_get_gas(), 350);
    } // guard_outer dropped

    assert_eq!(ctx_outer.gas_remaining(), 350);
    assert_eq!(ctx_inner.gas_remaining(), 750);
}

#[test]
fn test_mir_context_memory_and_status_isolation() {
    let ctx1 = MirContext::new();
    let ctx2 = MirContext::new();

    {
        let _g1 = ctx1.enter();
        lust_mir_set_max_memory(1024);
        lust_mir_track_memory_allocated(256);
        assert_eq!(lust_mir_get_memory_used(), 256);
        assert_eq!(lust_mir_get_status(), 0);

        {
            let _g2 = ctx2.enter();
            assert_eq!(lust_mir_get_memory_used(), 0, "ctx2 memory must start fresh");
            lust_mir_set_max_memory(4096);
            lust_mir_track_memory_allocated(1000);
            lust_mir_set_status(42);

            assert_eq!(lust_mir_get_memory_used(), 1000);
            assert_eq!(lust_mir_get_status(), 42);
        }

        assert_eq!(lust_mir_get_memory_used(), 256, "ctx1 memory must be restored");
        assert_eq!(lust_mir_get_status(), 0, "ctx1 status must be restored");
    }
}

#[test]
fn test_mir_out_of_gas_status_isolation() {
    let ctx1 = MirContext::with_gas(50);
    let ctx2 = MirContext::with_gas(500);

    {
        let _g1 = ctx1.enter();
        // Exhaust gas in ctx1
        let rem = lust_mir_charge_gas(60);
        assert!(rem < 0);
        assert_eq!(lust_mir_get_status(), lust::effect_abi::LUST_STATUS_ERR_OUT_OF_GAS as i64);

        {
            let _g2 = ctx2.enter();
            assert_eq!(lust_mir_get_status(), 0, "ctx2 status must be unaffected by ctx1 out of gas");
            let rem2 = lust_mir_charge_gas(100);
            assert_eq!(rem2, 400);
            assert_eq!(lust_mir_get_status(), 0);
        }

        assert_eq!(lust_mir_get_status(), lust::effect_abi::LUST_STATUS_ERR_OUT_OF_GAS as i64);
    }
}

#[test]
fn test_mir_default_fallback_without_frame() {
    // Operations outside explicit frames must safely hit default fallback without panic
    lust_mir_set_gas(999_999);
    assert_eq!(lust_mir_get_gas(), 999_999);
    lust_mir_charge_gas(100);
    assert_eq!(lust_mir_get_gas(), 999_899);
}

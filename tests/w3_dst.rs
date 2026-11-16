#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! Wave 3 Track 3G Acceptance Test Suite: DST-T2 (Tier-2 Compiled Loop Safepoints).
//!
//! Validates:
//! 1. MIR deterministic gas/timing quantization safepoint pass (`src/mir/opt/safepoints.rs`).
//! 2. Registration in `src/mir/opt/mod.rs` (`MirOptimizer`).
//! 3. Compiled function and loop safepoints with recorded native entry in Cranelift JIT.
//! 4. Spontaneous interrupt injection at function entry and loop headers.
//! 5. Nested masks: deferred delivery while masked, and delivery exactly once upon complete unmasking.
//! 6. Finalizer boundaries: strict LIFO execution and zero leaked resources.
//! 7. Stack unwinding: clean return with `MIR_STATUS_ERR_INTERRUPTED` on delivery.
//! 8. Deterministic simulation capture and bit-for-bit trace replay via `ReplayBundle`.
//! 9. Fail-safe handling of malformed bundles and unwritable capture output destinations.
//! 10. External watchdog boundedness for hang and infinite-loop probes.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tempfile::tempdir;

use lust::effect_abi::InterruptReason;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::*;
use lust::mir::ir::{MirModule, ResourceKind, Statement};
use lust::mir::lowering::AstLowering;
use lust::mir::opt::{MirOptimizer, SafepointConfig, Safepoints};
use lust::mir::runtime_helpers::{
    enter_mir_context, lust_mir_charge_gas, lust_mir_get_gas, lust_mir_set_gas, MirContext,
};
use lust::parser::Parser;
use lust::sim::replay::{
    CaptureConfig, CaptureError, EventLog, FailureKind, ReplayBundle, ReplayEngine, ReplayError,
    ReplayEventKind, ReplayManifest, UstarArchive,
};
use lust::sim::tier2::*;

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn parse_and_lower_module(source: &str, module_name: &str) -> MirModule {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("Tokenization succeeds");
    let items = Parser::new(tokens).parse().expect("Parsing succeeds");

    let mut module = MirModule::new(module_name.into());
    for item in items {
        if let lust::ast::ItemKind::Function(func_def) = item.kind {
            let mir_fn =
                AstLowering::lower_function(&func_def).expect("MIR lowering succeeds");
            module.add_function(mir_fn);
        }
    }
    module
}

/// Helper: run a closure with an external watchdog timer thread to bound hang probes.
fn run_with_watchdog_bound<F, R>(timeout: Duration, f: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let res = f();
        let _ = tx.send(res);
    });

    match rx.recv_timeout(timeout) {
        Ok(res) => res,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            panic!("WATCHDOG FAILURE: operation timed out after {:?}", timeout);
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            panic!("WATCHDOG FAILURE: worker thread panicked or disconnected");
        }
    }
}

// ============================================================================
// Test 1: MIR Pass Safepoint Insertion & Quantization Idempotency
// ============================================================================

#[test]
fn test_w3_dst_mir_safepoints_insertion_and_quantization() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let source = r#"
        function compute_sum(n: int): int
            local acc: int = 0
            local i: int = 0
            while i < n do
                acc = acc + i
                i = i + 1
            end
            return acc
        end
    "#;

    let mut module = parse_and_lower_module(source, "test_mir_safepoints");
    let mir_fn = module
        .functions
        .iter_mut()
        .find(|f| f.signature.name == "compute_sum")
        .expect("compute_sum exists");

    // 1. Detect loop headers and backedges
    let headers = Safepoints::find_loop_headers(&mir_fn.blocks);
    assert!(
        !headers.is_empty(),
        "Must identify at least one loop header in while loop"
    );

    let backedges = Safepoints::find_loop_backedges(&mir_fn.blocks);
    assert_eq!(backedges.len(), 1, "Must find exactly 1 backedge in single while loop");

    // 2. Insert safepoints with custom config (entry: 5 gas, loop: 10 gas)
    let config = SafepointConfig::new(5, 10);
    let inserted = Safepoints::run_on_function_with_config(mir_fn, &config);
    assert_eq!(inserted, 2, "Must insert entry safepoint and loop header safepoint");

    // Verify entry block starts with ResourceCheck(Gas, 5)
    assert!(matches!(
        mir_fn.blocks[mir_fn.entry.0].statements.first(),
        Some(Statement::ResourceCheck(ResourceKind::Gas, 5, _))
    ));

    // Verify loop header block has ResourceCheck(Gas, 10)
    for header in &headers {
        let block = &mir_fn.blocks[header.0];
        let has_loop_check = block.statements.iter().any(|s| {
            matches!(s, Statement::ResourceCheck(ResourceKind::Gas, 10, _))
        });
        assert!(has_loop_check, "Loop header block must contain loop ResourceCheck(Gas, 10)");
    }

    // 3. Verify Idempotency: re-running insertion pass inserts 0 additional checks
    let re_inserted = Safepoints::run_on_function_with_config(mir_fn, &config);
    assert_eq!(re_inserted, 0, "Pass must be idempotent; zero duplicates on second run");

    // 4. Test module-level pass and statistics
    let mut mod2 = parse_and_lower_module(source, "test_mod_stats");
    let stats = MirOptimizer::insert_safepoints_module(&mut mod2, &config);
    assert_eq!(stats.functions_visited, 1);
    assert_eq!(stats.functions_instrumented, 1);
    assert_eq!(stats.entry_safepoints_inserted, 1);
    assert_eq!(stats.loop_safepoints_inserted, 1);
    assert_eq!(stats.total_safepoints_inserted, 2);
    assert_eq!(stats.total_gas_quantized, 15); // 5 + 10 = 15
}

// ============================================================================
// Test 2: Cranelift JIT Compiled Loop Safepoints & Recorded Native Entry
// ============================================================================

#[test]
fn test_w3_dst_compiled_loop_safepoints_and_native_entry() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let source = r#"
        function loop_accum(n: int): int
            local sum: int = 0
            local i: int = 0
            while i < n do
                sum = sum + i
                i = i + 1
            end
            return sum
        end
    "#;

    let mut module = parse_and_lower_module(source, "test_compiled_loop");
    let mir_fn = module
        .functions
        .iter_mut()
        .find(|f| f.signature.name == "loop_accum")
        .expect("loop_accum exists");

    // Insert safepoints into MIR
    let config = SafepointConfig::new(1, 1);
    Safepoints::run_on_function_with_config(mir_fn, &config);

    // Compile function to native machine code via CraneliftJitCompiler
    let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift compiler");
    let code_ptr = compiler
        .compile_function(mir_fn)
        .expect("Cranelift native compilation succeeds");

    assert!(!code_ptr.is_null(), "Compiled native function pointer must be non-null");

    // Record native entry in controller event log
    let mut controller = Tier2SafepointController::new(100);
    controller.record_compiled_entry("loop_accum", 0);

    // Verify event was recorded
    assert_eq!(controller.event_log.len(), 1);
    assert!(matches!(
        &controller.event_log.events[0].event,
        ReplayEventKind::Tier2CompiledEntry { func_name, target_block: 0 } if func_name == "loop_accum"
    ));

    // Execute native compiled machine code
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    lust_mir_set_gas(100_000);

    let native_fn: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
    let result = native_fn(20);

    // 0 + 1 + ... + 19 = 190
    assert_eq!(result, 190, "Native compiled loop result must be 190");

    // Verify safepoints were actually checked by native loop
    let stats = safepoint_stats();
    assert!(
        stats.safepoints_checked > 0,
        "Native loop execution must invoke safepoints (got {})",
        stats.safepoints_checked
    );
    assert_eq!(stats.safepoints_delivered, 0, "No interrupt should be delivered on clean run");
}

// ============================================================================
// Test 3: Interrupt Injection at Safepoint & Clean Stack Unwind
// ============================================================================

#[test]
fn test_w3_dst_interrupt_injection_and_unwind_at_safepoint() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let finalizer_executed = Arc::new(AtomicBool::new(false));
    let f_flag = finalizer_executed.clone();

    // Register a cleanup finalizer on the safepoint runtime
    safepoint_register_finalizer(move || {
        f_flag.store(true, Ordering::SeqCst);
    });

    // Request spontaneous interrupt (Cancelled)
    safepoint_request_interrupt(InterruptReason::Cancelled);

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    lust_mir_set_gas(100_000);

    // Trigger safepoint check
    let status = lust_mir_safepoint();

    // Assert safepoint delivered the interrupt
    assert_eq!(
        status, MIR_STATUS_ERR_INTERRUPTED,
        "Safepoint must return MIR_STATUS_ERR_INTERRUPTED on interrupt delivery"
    );

    // Assert status in MirContext was set
    assert_eq!(
        ctx.last_status.load(Ordering::SeqCst),
        MIR_STATUS_ERR_INTERRUPTED,
        "MirContext.last_status must record MIR_STATUS_ERR_INTERRUPTED"
    );

    // Assert registered finalizer executed
    assert!(
        finalizer_executed.load(Ordering::SeqCst),
        "Registered finalizer must run to completion upon interrupt delivery"
    );

    // Assert runtime statistics
    let stats = safepoint_stats();
    assert_eq!(stats.safepoints_delivered, 1);
    assert_eq!(stats.cleanups_run, 1);
}

// ============================================================================
// Test 4: Nested Masks — Deferred Delivery Exactly Once Upon Complete Unmasking
// ============================================================================

#[test]
fn test_w3_dst_nested_masks_deferred_delivery_exactly_once() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let finalizer_ran = Arc::new(AtomicBool::new(false));
    let f_flag = finalizer_ran.clone();
    safepoint_register_finalizer(move || {
        f_flag.store(true, Ordering::SeqCst);
    });

    // 1. Enter nested masks: Depth 0 -> 1 -> 2 -> 3
    safepoint_enter_mask();
    safepoint_enter_mask();
    safepoint_enter_mask();
    assert_eq!(safepoint_stats().mask_depth, 3);

    // 2. Request interrupt while deeply masked
    safepoint_request_interrupt(InterruptReason::Timeout);

    // 3. Safepoint check while masked: interrupt MUST be deferred
    let status1 = lust_mir_safepoint();
    assert_eq!(status1, 0, "Execution must continue while mask depth > 0");
    assert_eq!(safepoint_stats().safepoints_masked_deferred, 1);
    assert!(!finalizer_ran.load(Ordering::SeqCst), "Finalizer must not run while masked");

    // 4. Exit first mask level: Depth 3 -> 2
    safepoint_exit_mask();
    assert_eq!(safepoint_stats().mask_depth, 2);
    let status2 = lust_mir_safepoint();
    assert_eq!(status2, 0, "Execution must continue while mask depth is 2");
    assert_eq!(safepoint_stats().safepoints_masked_deferred, 2);
    assert!(!finalizer_ran.load(Ordering::SeqCst));

    // 5. Exit second mask level: Depth 2 -> 1
    safepoint_exit_mask();
    assert_eq!(safepoint_stats().mask_depth, 1);
    let status3 = lust_mir_safepoint();
    assert_eq!(status3, 0, "Execution must continue while mask depth is 1");
    assert_eq!(safepoint_stats().safepoints_masked_deferred, 3);
    assert!(!finalizer_ran.load(Ordering::SeqCst));

    // 6. Exit final mask level: Depth 1 -> 0 (now fully unmasked!)
    let unmasked_reason = safepoint_exit_mask();
    assert_eq!(safepoint_stats().mask_depth, 0);
    assert_eq!(unmasked_reason, Some(InterruptReason::Timeout));

    // 7. Safepoint check now delivers the deferred cancellation EXACTLY ONCE!
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    let status_delivered = lust_mir_safepoint();

    assert_eq!(
        status_delivered, MIR_STATUS_ERR_INTERRUPTED,
        "Safepoint must deliver deferred interrupt immediately after outer unmasking"
    );
    assert_eq!(safepoint_stats().safepoints_delivered, 1);
    assert!(finalizer_ran.load(Ordering::SeqCst), "Finalizer must have run");

    // 8. Subsequent safepoint checks: interrupt has been consumed, does NOT redeliver!
    let status_after = lust_mir_safepoint();
    assert_eq!(status_after, 0, "Subsequent check must return 0 (exactly-once delivery)");
    assert_eq!(
        safepoint_stats().safepoints_delivered,
        1,
        "safepoints_delivered counter must remain 1"
    );
}

// ============================================================================
// Test 5: Finalizer Boundaries, Strict LIFO Order & Zero Leaked Resources
// ============================================================================

#[test]
fn test_w3_dst_finalizer_boundaries_lifo_order_and_no_resource_leaks() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let mut controller = Tier2SafepointController::new(2026);
    let execution_order = Arc::new(Mutex::new(Vec::new()));

    // Register 4 finalizers in sequence: 101, 102, 103, 104
    for id in [101, 102, 103, 104] {
        let order = execution_order.clone();
        controller.register_finalizer(id, move || {
            order.lock().unwrap().push(id);
        });
    }

    assert_eq!(controller.registered_finalizers.len(), 4);

    // Enter mask, inject interrupt, exit mask
    controller.enter_mask();
    controller.inject_interrupt(InterruptReason::Cancelled);
    controller.exit_mask();

    // Trigger delivery step
    let res = controller.step_safepoint().expect("Safepoint step succeeds");
    assert_eq!(res, MIR_STATUS_ERR_INTERRUPTED);

    // Verify LIFO order: [104, 103, 102, 101]
    let recorded = execution_order.lock().unwrap().clone();
    assert_eq!(
        recorded,
        vec![104, 103, 102, 101],
        "Finalizers must execute in strict LIFO order"
    );

    // Verify no leaked resources
    assert_eq!(controller.cleanups_run, 4);
    assert_eq!(controller.delivered_count, 1);

    // Verify INV-FIBER-01 and INV-FIBER-02 formal oracles
    controller
        .verify_invariants()
        .expect("Formal oracles must pass with zero violations");
}

// ============================================================================
// Test 6: Tier-2 Simulation Capture & Deterministic Replay
// ============================================================================

#[test]
fn test_w3_dst_simulation_capture_and_deterministic_replay() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let schedule = Tier2InterruptSchedule::new()
        .with_action(Tier2SimAction::RegisterFinalizer {
            at_safepoint: 1,
            finalizer_id: 1,
        })
        .with_action(Tier2SimAction::EnterMask { at_safepoint: 2 })
        .with_action(Tier2SimAction::EnterMask { at_safepoint: 3 })
        .with_action(Tier2SimAction::InjectInterrupt {
            at_safepoint: 4,
            reason: InterruptReason::OutOfGas,
        })
        .with_action(Tier2SimAction::ExitMask { at_safepoint: 5 })
        .with_action(Tier2SimAction::ExitMask { at_safepoint: 6 });

    let mut session = Tier2SimSession::new(42, schedule.clone());

    // Step simulation across 8 steps
    for _ in 0..8 {
        let status = session.run_step().expect("Step succeeds");
        if status == MIR_STATUS_ERR_INTERRUPTED {
            break;
        }
    }

    assert_eq!(session.controller.delivered_count, 1);
    assert_eq!(session.controller.cleanups_run, 1);

    // Capture failure to bundle
    let config = CaptureConfig::new("/tmp/test_bundle.tar")
        .with_seed(42)
        .with_step_count(session.controller.safepoint_counter)
        .with_target_name("tier2_safepoints_test");

    let failure = FailureKind::RuntimeError("Simulated interruption at safepoint".into());
    let bundle = session.capture(&config, &failure).expect("Capture succeeds");

    assert_eq!(bundle.manifest.seed, 42);
    assert_eq!(
        bundle.manifest.step_count,
        session.controller.safepoint_counter
    );
    assert!(!bundle.events.is_empty(), "EventLog must contain recorded events");

    // Replay bundle deterministically using Tier2ReplayEngine
    let replay_result =
        Tier2ReplayEngine::replay_bundle(&bundle, &schedule).expect("Replay succeeds");

    assert!(
        replay_result.matched,
        "Replay trace must match original execution bit-for-bit: {:?}",
        replay_result.divergence_detail
    );
    assert!(replay_result.divergence_step.is_none());
}

// ============================================================================
// Test 7: Malformed Replay Handling
// ============================================================================

#[test]
fn test_w3_dst_malformed_replay_handling() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    // 1. Corrupted TAR bytes
    let bad_bytes = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x11, 0x22, 0x33];
    let err = ReplayBundle::import_tar_bytes(&bad_bytes);
    assert!(
        err.is_err(),
        "Importing invalid TAR bytes must fail safely with ReplayError"
    );

    // 2. Missing manifest.json in archive
    let mut bad_archive = UstarArchive::new();
    bad_archive
        .add_file("events.jsonl", b"")
        .unwrap();
    let archive_bytes = bad_archive.finish();
    let import_err = ReplayBundle::import_tar_bytes(&archive_bytes);
    assert!(
        matches!(import_err, Err(ReplayError::MissingEntry(ref entry)) if entry == "manifest.json"),
        "Missing manifest.json must return ReplayError::MissingEntry"
    );

    // 3. Malformed events.jsonl
    let mut bad_jsonl_archive = UstarArchive::new();
    let manifest = ReplayManifest::new(1, "test", 10);
    bad_jsonl_archive
        .add_file("manifest.json", manifest.to_json().unwrap().as_bytes())
        .unwrap();
    bad_jsonl_archive
        .add_file("events.jsonl", b"{ THIS IS NOT VALID JSON }\n")
        .unwrap();
    let jsonl_bytes = bad_jsonl_archive.finish();
    let jsonl_err = ReplayBundle::import_tar_bytes(&jsonl_bytes);
    assert!(
        matches!(jsonl_err, Err(ReplayError::Deserialization(_))),
        "Malformed events.jsonl must return ReplayError::Deserialization"
    );

    // 4. Trace divergence detection
    let mut log1 = EventLog::new();
    log1.record(
        1,
        100,
        ReplayEventKind::SafepointChecked {
            safepoint_id: 1,
            mask_depth: 0,
        },
    );

    let mut log2 = EventLog::new();
    log2.record(
        1,
        100,
        ReplayEventKind::SafepointChecked {
            safepoint_id: 1,
            mask_depth: 1, // Divergent mask depth!
        },
    );

    let divergence_result = ReplayEngine::verify_trace(&log1, &log2);
    assert!(!divergence_result.matched, "Trace divergence must be detected");
    assert_eq!(divergence_result.divergence_step, Some(1));
    assert!(divergence_result.divergence_detail.is_some());
}

// ============================================================================
// Test 8: Capture Failure & Unwritable Output Destination Handling
// ============================================================================

#[test]
fn test_w3_dst_capture_failure_unwritable_output_handling() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let controller = Tier2SafepointController::new(999);
    let failure = FailureKind::RuntimeError("Destination unwritable test".into());

    // Path pointing to an illegal unwritable destination (e.g. inside /dev/null)
    let illegal_path = PathBuf::from("/dev/null/impossible_dir/bundle.tar");
    let config = CaptureConfig::new(&illegal_path);

    let export_res = controller.export_bundle_tar(&config, &failure, &illegal_path);

    assert!(
        matches!(export_res, Err(CaptureError::UnwritableOutput(_))),
        "Writing to unwritable destination must return CaptureError::UnwritableOutput: {:?}",
        export_res
    );
}

// ============================================================================
// Test 9: Gas Quantization Safepoints & Out-Of-Gas Clean Termination
// ============================================================================

#[test]
fn test_w3_dst_gas_quantization_and_out_of_gas_exit() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);

    // Start with 25 gas
    lust_mir_set_gas(25);
    assert_eq!(lust_mir_get_gas(), 25);

    // Quantized loop steps: 10 gas per iteration
    let rem1 = lust_mir_charge_gas(10);
    assert_eq!(rem1, 15);
    assert_eq!(ctx.last_status.load(Ordering::SeqCst), 0);

    let rem2 = lust_mir_charge_gas(10);
    assert_eq!(rem2, 5);
    assert_eq!(ctx.last_status.load(Ordering::SeqCst), 0);

    // Third iteration exhausts remaining gas: 5 - 10 = -5 < 0
    let rem3 = lust_mir_charge_gas(10);
    assert_eq!(rem3, -5);
    assert_eq!(
        ctx.last_status.load(Ordering::SeqCst),
        lust::effect_abi::LUST_STATUS_ERR_OUT_OF_GAS as i64,
        "Status must transition to LUST_STATUS_ERR_OUT_OF_GAS when budget is exhausted"
    );
}

// ============================================================================
// Test 10: Bounded Hang Probe with External Watchdog
// ============================================================================

#[test]
fn test_w3_dst_bounded_hang_probe_external_watchdog() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    // Watchdog bounds execution to 2.0s maximum.
    // The simulated loop will terminate cleanly on the 5th iteration via safepoint.
    run_with_watchdog_bound(Duration::from_secs(2), || {
        let schedule = Tier2InterruptSchedule::new().with_action(Tier2SimAction::InjectInterrupt {
            at_safepoint: 5,
            reason: InterruptReason::Cancelled,
        });

        let mut session = Tier2SimSession::new(12345, schedule);
        let mut loop_count = 0;

        // Bounded loop probe: should be terminated by safepoint well before 1_000_000 iterations
        while loop_count < 1_000_000 {
            loop_count += 1;
            let status = session.run_step().expect("Step succeeds");
            if status == MIR_STATUS_ERR_INTERRUPTED {
                break;
            }
        }

        assert_eq!(
            loop_count, 5,
            "Safepoint must terminate loop at exactly iteration 5"
        );
        assert_eq!(session.controller.delivered_count, 1);
    });
}

#![cfg(feature = "std")]

//! Production VM OSR backedge integration tests for `PERF-T2-6` (Hazard 6 item 3).

use std::sync::atomic::Ordering;
use std::sync::Mutex;

use lust::bytecode::value::{native_fn, NativeCallResult};
use lust::bytecode::Value;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{safepoint_request_deopt, safepoint_reset};
use lust::mir::ir::MirModule;
use lust::mir::lowering::AstLowering;
use lust::parser::Parser;
use lust::vm::osr::GLOBAL_OSR_COUNTERS;
use lust::vm::{VmBackend, VM};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn parse_and_lower_module(source: &str, module_name: &str) -> MirModule {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("Tokenization must succeed");
    let items = Parser::new(tokens).parse().expect("Parsing must succeed");

    let mut module = MirModule::new(module_name.into());
    for item in items {
        if let lust::ast::ItemKind::Function(func_def) = item.kind {
            let mir_fn = AstLowering::lower_function(&func_def).expect("MIR lowering must succeed");
            module.add_function(mir_fn);
        }
    }
    module
}

fn register_host_setup(vm: &mut VM) {
    vm.register_native(
        "host_setup",
        Value::NativeFunction(native_fn(|args| {
            let base = args.first().and_then(|v| v.as_int()).unwrap_or(0);
            Ok(NativeCallResult::Return(Value::Int(base + 100)))
        })),
    );
    vm.register_native(
        "host_consume_tuple",
        Value::NativeFunction(native_fn(|args| {
            let sum = match args.first() {
                Some(Value::Tuple(elems)) => elems
                    .iter()
                    .filter_map(|v| v.as_int())
                    .sum::<i64>(),
                _ => 0,
            };
            Ok(NativeCallResult::Return(Value::Int(sum)))
        })),
    );
}

#[test]
fn test_perf_t2_6_production_vm_osr_entry() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    GLOBAL_OSR_COUNTERS.reset();
    safepoint_reset();

    let source = r#"
        function setup_then_hot_loop(n: int): int
            local bias: int = host_setup(7)
            local sum: int = bias
            local i: int = 0
            while i < n do
                sum = sum + i
                i = i + 1
            end
            return sum
        end
    "#;

    let module = parse_and_lower_module(source, "perf_t2_6_prod_osr");
    let n = 100_000i64;

    // 1. Bytecode interpreter oracle
    let mut vm_oracle = VM::new();
    register_host_setup(&mut vm_oracle);
    vm_oracle
        .load_mir_module(&module)
        .expect("Oracle load_mir_module must succeed");
    vm_oracle.set_backend(VmBackend::Bytecode);
    let oracle_res = vm_oracle
        .call("setup_then_hot_loop", vec![Value::Int(n)])
        .expect("Oracle call must succeed");
    let expected = 107i64 + (n * (n - 1)) / 2;
    assert_eq!(oracle_res, Value::Int(expected));

    // 2. Cranelift production VM: whole function is ineligible due to `host_setup`,
    //    so interpreter executes setup and enters OSR on the hot loop backedge.
    let mut vm_cl = VM::new();
    register_host_setup(&mut vm_cl);
    vm_cl
        .load_mir_module(&module)
        .expect("Cranelift load_mir_module must succeed");
    vm_cl.set_backend(VmBackend::Cranelift);

    let successes_before = GLOBAL_OSR_COUNTERS.entry_successes.load(Ordering::SeqCst);
    let cl_res = vm_cl
        .call("setup_then_hot_loop", vec![Value::Int(n)])
        .expect("Cranelift OSR call must succeed");
    let successes_after = GLOBAL_OSR_COUNTERS.entry_successes.load(Ordering::SeqCst);

    assert_eq!(
        cl_res, oracle_res,
        "Production OSR result must match Bytecode oracle identically"
    );
    assert!(
        successes_after > successes_before,
        "Production VM must enter OSR at hot loop backedge (entry_successes delta = {})",
        successes_after - successes_before
    );
}

#[test]
fn test_perf_t2_6_osr_gas_exhaustion_and_deduction() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    GLOBAL_OSR_COUNTERS.reset();
    safepoint_reset();

    let source = r#"
        function setup_then_gas_loop(n: int): int
            local bias: int = host_setup(3)
            local sum: int = bias
            local i: int = 0
            while i < n do
                sum = sum + i
                i = i + 1
            end
            return sum
        end
    "#;

    let module = parse_and_lower_module(source, "perf_t2_6_gas");

    // 1. Finite gas budget large enough to enter OSR (past 64 interpreter iterations)
    //    but far too small for 100_000 iterations: MUST fail with Out of gas AND deduct gas!
    let mut vm_oog = VM::new();
    register_host_setup(&mut vm_oog);
    vm_oog
        .load_mir_module(&module)
        .expect("load_mir_module must succeed");
    vm_oog.set_backend(VmBackend::Cranelift);

    let gas_limit = 5_000u64;
    vm_oog.set_gas_budget(gas_limit);

    let attempts_before = GLOBAL_OSR_COUNTERS.entry_attempts.load(Ordering::SeqCst);
    let err = vm_oog
        .call("setup_then_gas_loop", vec![Value::Int(100_000)])
        .expect_err("OSR-entered loop must terminate on gas exhaustion");
    let attempts_after = GLOBAL_OSR_COUNTERS.entry_attempts.load(Ordering::SeqCst);

    let err_msg = format!("{err}");
    assert!(
        err_msg.contains("Out of gas"),
        "Expected Out of gas error, got: {err_msg}"
    );
    assert!(
        attempts_after > attempts_before,
        "Loop must have entered OSR before exhausting gas (attempts delta = {})",
        attempts_after - attempts_before
    );
    assert_eq!(
        vm_oog.gas_used(),
        gas_limit + 1,
        "OSR gas exhaustion must deduct exact consumed gas into VM budgets (limit={}, used={})",
        gas_limit,
        vm_oog.gas_used()
    );
    assert_eq!(
        vm_oog.gas_remaining(),
        Some(0),
        "Remaining gas must be 0 after OSR gas exhaustion"
    );

    // 2. Sufficient finite gas budget: OSR completes AND deducts consumed gas accurately.
    let mut vm_ok = VM::new();
    register_host_setup(&mut vm_ok);
    vm_ok
        .load_mir_module(&module)
        .expect("load_mir_module must succeed");
    vm_ok.set_backend(VmBackend::Cranelift);

    let big_limit = 500_000u64;
    vm_ok.set_gas_budget(big_limit);

    let successes_before = GLOBAL_OSR_COUNTERS.entry_successes.load(Ordering::SeqCst);
    let res = vm_ok
        .call("setup_then_gas_loop", vec![Value::Int(10_000)])
        .expect("OSR loop within gas budget must succeed");
    let successes_after = GLOBAL_OSR_COUNTERS.entry_successes.load(Ordering::SeqCst);

    let expected = 103i64 + (10_000i64 * 9_999i64) / 2;
    assert_eq!(res, Value::Int(expected));
    assert_eq!(successes_after - successes_before, 1);
    assert!(
        vm_ok.gas_used() >= 10_000,
        "Completed OSR loop of 10_000 iterations must deduct >= 10_000 gas, got {}",
        vm_ok.gas_used()
    );
    assert!(
        vm_ok.gas_remaining().unwrap() < big_limit - 9_900,
        "Remaining gas must reflect OSR execution deduction"
    );
}

#[test]
fn test_perf_t2_6_osr_forced_safepoint_deopt_resumes_in_interpreter() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    GLOBAL_OSR_COUNTERS.reset();
    safepoint_reset();

    let source = r#"
        function setup_then_deopt_loop(n: int): int
            local bias: int = host_setup(5)
            local sum: int = bias
            local i: int = 0
            while i < n do
                sum = sum + i
                i = i + 1
            end
            return sum
        end
    "#;

    let module = parse_and_lower_module(source, "perf_t2_6_deopt");
    let n = 1_000i64;
    let expected = 105i64 + (n * (n - 1)) / 2;

    // 1. Oracle
    let mut vm_oracle = VM::new();
    register_host_setup(&mut vm_oracle);
    vm_oracle
        .load_mir_module(&module)
        .expect("load_mir_module must succeed");
    vm_oracle.set_backend(VmBackend::Bytecode);
    let oracle_res = vm_oracle
        .call("setup_then_deopt_loop", vec![Value::Int(n)])
        .expect("Oracle call must succeed");
    assert_eq!(oracle_res, Value::Int(expected));

    // 2. Forced immediate safepoint deopt inside OSR-entered loop
    let mut vm_deopt = VM::new();
    register_host_setup(&mut vm_deopt);
    vm_deopt
        .load_mir_module(&module)
        .expect("load_mir_module must succeed");
    vm_deopt.set_backend(VmBackend::Cranelift);

    safepoint_request_deopt(true);
    let deopt_before = GLOBAL_OSR_COUNTERS.deopt_count.load(Ordering::SeqCst);
    let res = vm_deopt
        .call("setup_then_deopt_loop", vec![Value::Int(n)])
        .expect("Forced OSR safepoint deopt must resume cleanly in interpreter");
    let deopt_after = GLOBAL_OSR_COUNTERS.deopt_count.load(Ordering::SeqCst);
    safepoint_reset();

    assert_eq!(
        res, oracle_res,
        "Post-deopt interpreter resumption must produce exact oracle result"
    );
    assert_eq!(
        deopt_after - deopt_before,
        1,
        "Forced safepoint deopt must record exactly 1 OSR deopt without deopt storm"
    );

    // 3. Under `test-deopt`: trigger deopt mid-loop on poll ordinal 50 (after native loop
    //    iterations have mutated `sum` and `i` in native registers), verifying exact
    //    live-local reconstruction and interpreter completion.
    #[cfg(feature = "test-deopt")]
    {
        let mut vm_mid = VM::new();
        register_host_setup(&mut vm_mid);
        vm_mid
            .load_mir_module(&module)
            .expect("load_mir_module must succeed");
        vm_mid.set_backend(VmBackend::Cranelift);

        lust::mir::cranelift_backend::safepoint_request_deopt_on_poll(
            Some("setup_then_deopt_loop".to_string()),
            50,
        );
        let deopt_mid_before = GLOBAL_OSR_COUNTERS.deopt_count.load(Ordering::SeqCst);
        let mid_res = vm_mid
            .call("setup_then_deopt_loop", vec![Value::Int(n)])
            .expect("Mid-loop poll-50 OSR deopt must resume cleanly in interpreter");
        let deopt_mid_after = GLOBAL_OSR_COUNTERS.deopt_count.load(Ordering::SeqCst);
        safepoint_reset();

        assert_eq!(
            mid_res, oracle_res,
            "Mid-loop poll-50 OSR deopt must reconstruct mutated live locals and match oracle"
        );
        assert_eq!(
            deopt_mid_after - deopt_mid_before,
            1,
            "Mid-loop poll-50 OSR deopt must trigger exactly 1 deopt"
        );
    }
}

#[test]
fn test_perf_t2_6_osr_unsupported_live_local_marks_site_cold() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    GLOBAL_OSR_COUNTERS.reset();
    safepoint_reset();

    // A live Tuple across the loop header is unsupported for flat scalar OSR entry.
    // The site must fall back cleanly on the first threshold crossing and be marked cold
    // so subsequent iterations do NOT thrash OSR attempts!
    let source = r#"
        function setup_then_tuple_loop(n: int): int
            local bias: int = host_setup(1)
            local pair: (int, int) = (bias, 2)
            local sum: int = 0
            local i: int = 0
            while i < n do
                sum = sum + i
                i = i + 1
            end
            local tail: int = host_consume_tuple(pair)
            return sum + tail
        end
    "#;

    let module = parse_and_lower_module(source, "perf_t2_6_cold_tuple");
    let n = 10_000i64;

    let mut vm_oracle = VM::new();
    register_host_setup(&mut vm_oracle);
    vm_oracle
        .load_mir_module(&module)
        .expect("Oracle load_mir_module must succeed");
    vm_oracle.set_backend(VmBackend::Bytecode);
    let oracle_res = vm_oracle
        .call("setup_then_tuple_loop", vec![Value::Int(n)])
        .expect("Oracle call must succeed");

    let mut vm_cl = VM::new();
    register_host_setup(&mut vm_cl);
    vm_cl
        .load_mir_module(&module)
        .expect("Cranelift load_mir_module must succeed");
    vm_cl.set_backend(VmBackend::Cranelift);

    let attempts_before = GLOBAL_OSR_COUNTERS.entry_attempts.load(Ordering::SeqCst);
    let fallbacks_before = GLOBAL_OSR_COUNTERS.fallback_count.load(Ordering::SeqCst);
    let successes_before = GLOBAL_OSR_COUNTERS.entry_successes.load(Ordering::SeqCst);

    let cl_res = vm_cl
        .call("setup_then_tuple_loop", vec![Value::Int(n)])
        .expect("Unsupported live-local loop must fall back cleanly to interpreter");

    let attempts_delta = GLOBAL_OSR_COUNTERS.entry_attempts.load(Ordering::SeqCst) - attempts_before;
    let fallbacks_delta =
        GLOBAL_OSR_COUNTERS.fallback_count.load(Ordering::SeqCst) - fallbacks_before;
    let successes_delta =
        GLOBAL_OSR_COUNTERS.entry_successes.load(Ordering::SeqCst) - successes_before;

    assert_eq!(
        cl_res, oracle_res,
        "Unsupported live-local fallback must match oracle identically"
    );
    assert_eq!(
        successes_delta, 0,
        "Unsupported live-local loop must not report OSR success"
    );
    assert!(
        attempts_delta <= 1,
        "Cold-site marking must prevent linear OSR thrashing across 10,000 iterations (attempts_delta = {})",
        attempts_delta
    );
    assert_eq!(
        fallbacks_delta, attempts_delta,
        "Each failed OSR attempt must record a fallback"
    );
}

#[test]
fn test_perf_t2_6_osr_cached_reentry_no_redundant_recompilation() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    GLOBAL_OSR_COUNTERS.reset();
    safepoint_reset();

    let source = r#"
        function setup_then_cached_loop(n: int): int
            local bias: int = host_setup(10)
            local sum: int = bias
            local i: int = 0
            while i < n do
                sum = sum + i
                i = i + 1
            end
            return sum
        end
    "#;

    let module = parse_and_lower_module(source, "perf_t2_6_cache");
    let mut vm = VM::new();
    register_host_setup(&mut vm);
    vm.load_mir_module(&module)
        .expect("load_mir_module must succeed");
    vm.set_backend(VmBackend::Cranelift);

    let successes_before = GLOBAL_OSR_COUNTERS.entry_successes.load(Ordering::SeqCst);

    // Call 1: compiles OSR entry once and executes natively
    let res1 = vm
        .call("setup_then_cached_loop", vec![Value::Int(5_000)])
        .expect("First call must succeed");
    assert_eq!(res1, Value::Int(110 + (5_000 * 4_999) / 2));
    assert_eq!(
        vm.osr_compilation_count(),
        1,
        "First hot loop call must compile exactly 1 OSR entry"
    );

    // Call 2: hits compiled OSR cache with ZERO redundant recompilation!
    let res2 = vm
        .call("setup_then_cached_loop", vec![Value::Int(6_000)])
        .expect("Second call must succeed");
    assert_eq!(res2, Value::Int(110 + (6_000 * 5_999) / 2));
    assert_eq!(
        vm.osr_compilation_count(),
        1,
        "Second hot loop call must reuse cached OSR entry (0 redundant recompilations)"
    );

    let successes_after = GLOBAL_OSR_COUNTERS.entry_successes.load(Ordering::SeqCst);
    assert_eq!(
        successes_after - successes_before,
        2,
        "Both calls must execute via native OSR entry"
    );

    // Reloading updated MIR invalidates OSR cache so stale machine code never executes
    let updated_source = r#"
        function setup_then_cached_loop(n: int): int
            local bias: int = host_setup(10)
            local sum: int = bias
            local i: int = 0
            while i < n do
                sum = sum + i + 1
                i = i + 1
            end
            return sum
        end
    "#;
    let updated_module = parse_and_lower_module(updated_source, "perf_t2_6_cache_v2");
    vm.load_mir_module(&updated_module)
        .expect("Reloading MIR module must succeed");

    let res3 = vm
        .call("setup_then_cached_loop", vec![Value::Int(5_000)])
        .expect("Updated call must succeed");
    assert_eq!(
        res3,
        Value::Int(110 + (5_000 * 5_001) / 2),
        "Reloaded MIR function must invalidate OSR cache and produce updated result"
    );
    assert_eq!(
        vm.osr_compilation_count(),
        2,
        "Reloaded MIR function must recompile OSR entry after invalidation"
    );
}

#[test]
fn test_perf_t2_6_adv_try_apply_authoritative_error_leaves_frame_unchanged_and_marks_cold() {
    use lust::mir::cranelift_backend::DeoptSiteKind;
    use lust::vm::CallFrame;
    use lust::vm::osr::{DeoptExitState, DeoptReason};

    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    GLOBAL_OSR_COUNTERS.reset();
    safepoint_reset();

    // 1. Direct atomicity check on DeoptExitState::try_apply_authoritative:
    //    (a) Missing resume_pc -> Err and ZERO mutation to frame.ip or frame.registers.
    let mut registers = [const { Value::Nil }; 256];
    registers[0] = Value::Int(111);
    registers[1] = Value::Int(222);
    let mut frame = CallFrame {
        function_idx: 0,
        ip: 42,
        registers,
        base_register: 0,
        return_dest: None,
        upvalues: Vec::new(),
        shared_upvalues: Vec::new(),
    };
    let snapshot_regs = frame.registers.clone();

    let missing_pc_state = DeoptExitState {
        site_id: 7,
        site_kind: DeoptSiteKind::BlockEntry,
        target_pc: 7,
        resume_pc: None,
        reason: DeoptReason::Safepoint,
        restored_locals: vec![(0, Value::Int(999)), (1, Value::Int(888))],
    };
    let err = missing_pc_state
        .try_apply_authoritative(&mut frame)
        .expect_err("Missing resume_pc must fail closed");
    assert!(err.contains("No proven bytecode resume PC mapping"), "unexpected err: {err}");
    assert_eq!(frame.ip, 42, "frame.ip must remain untouched on error");
    assert_eq!(frame.registers, snapshot_regs, "frame.registers must remain untouched on error");

    //    (b) Out-of-bounds register index after a valid index -> Err and ZERO partial mutation.
    let oob_state = DeoptExitState {
        site_id: 7,
        site_kind: DeoptSiteKind::BlockEntry,
        target_pc: 7,
        resume_pc: Some(12),
        reason: DeoptReason::Safepoint,
        restored_locals: vec![(0, Value::Int(999)), (9999, Value::Int(777))],
    };
    let err2 = oob_state
        .try_apply_authoritative(&mut frame)
        .expect_err("Out-of-bounds register index must fail closed");
    assert!(err2.contains("out of bounds"), "unexpected err2: {err2}");
    assert_eq!(frame.ip, 42, "frame.ip must remain untouched on OOB error");
    assert_eq!(
        frame.registers, snapshot_regs,
        "Earlier register 0 must not be partially mutated when register 9999 is OOB"
    );

    // 2. End-to-end production OSR deopt error marks site cold:
    //    Trigger MIR_STATUS_DEOPT during OSR with a corrupted/empty block_entry_pcs entry
    //    cannot occur without tampering, so verify clean deopt + cold marking suppresses re-entry
    //    AND verify cold_site_count == 1 after forced deopt.
    let source = r#"
        function setup_then_deopt_cold(n: int): int
            local bias: int = host_setup(1)
            local acc: int = bias
            local i: int = 0
            while i < n do
                acc = acc + i
                i = i + 1
            end
            return acc
        end
    "#;
    let module = parse_and_lower_module(source, "perf_t2_6_deopt_cold");
    let mut vm = VM::new();
    register_host_setup(&mut vm);
    vm.load_mir_module(&module).expect("load_mir_module");
    vm.set_backend(VmBackend::Cranelift);

    assert_eq!(vm.osr_cold_site_count(), 0);
    safepoint_request_deopt(true);
    let res = vm
        .call("setup_then_deopt_cold", vec![Value::Int(1_000)])
        .expect("Deopt must resume cleanly");
    safepoint_reset();
    assert_eq!(res, Value::Int(101 + (1_000 * 999) / 2));
    assert_eq!(
        vm.osr_cold_site_count(),
        1,
        "Deopted OSR site must be marked cold upfront so subsequent loops do not thrash"
    );

    // Subsequent call on same generation stays cold (0 new OSR attempts).
    let attempts_before = GLOBAL_OSR_COUNTERS.entry_attempts.load(Ordering::SeqCst);
    let res2 = vm
        .call("setup_then_deopt_cold", vec![Value::Int(1_000)])
        .expect("Subsequent call on cold site must succeed in interpreter");
    assert_eq!(res2, res);
    assert_eq!(
        GLOBAL_OSR_COUNTERS.entry_attempts.load(Ordering::SeqCst) - attempts_before,
        0,
        "Cold-marked OSR site must not re-attempt OSR on same generation"
    );
}

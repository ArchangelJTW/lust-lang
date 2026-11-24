#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use lust::bytecode::{EnumObject, Value};
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::safepoint_request_deopt_on_poll;
use lust::mir::cranelift_backend::{
    lust_mir_queue_new, lust_mir_queue_offer_int, lust_mir_queue_take_int,
    lust_mir_unknown_is_int, lust_mir_unknown_unwrap_int, safepoint_reset,
    take_current_sticky_error, take_deopt_payload, CraneliftJitCompiler, MIR_STATUS_ERR_OOM,
    MIR_STATUS_ERR_OUT_OF_BOUNDS, MIR_STATUS_OK,
};
use lust::mir::runtime_helpers::{
    enter_mir_context, lust_mir_string_eq, lust_mir_string_len, lust_mir_string_sub, MirContext,
};
use lust::vfs::VirtualConsole;
use lust::vm::{reset_effect_handler_state, VmBackend, VM};
use lust::{compile_program_with_config, load_program_from_embedded, EmbeddedModule, LustConfig};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn reset_all_state() {
    safepoint_reset();
    reset_effect_handler_state();
    take_deopt_payload();
    take_current_sticky_error();
}

fn compile_embedded_vm(source: &str, cranelift: bool) -> VM {
    let entries = &[EmbeddedModule {
        module: "main",
        parent: None,
        source: Some(source),
    }];
    let program = load_program_from_embedded(entries, "main").expect("load embedded program");
    let mut cfg = LustConfig::default().with_cranelift_jit(cranelift);
    cfg.set_use_mir(true);
    let mut vm = compile_program_with_config(program, &cfg).expect("compile embedded program");
    vm.console = VirtualConsole::captured();
    vm.set_backend(if cranelift {
        VmBackend::Cranelift
    } else {
        VmBackend::Bytecode
    });
    vm
}

fn read_floatmath_source() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks")
        .join("suite")
        .join("floatmath.lust");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

fn read_schema_json_source() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("benchmarks")
        .join("suite_v2")
        .join("schema_json_decode_pipeline.lust");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

fn floatmath_rust_oracle() -> f64 {
    let mut x = 0.0_f64;
    let mut acc = 0.0_f64;
    let mut i = 0_i64;
    while i < 10_000_000 {
        x += 0.001;
        acc = acc + x * x - acc * 0.5;
        i += 1;
    }
    acc
}

fn median_ms(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    let mid = samples.len() / 2;
    if samples.len() % 2 == 1 {
        samples[mid]
    } else {
        0.5 * (samples[mid - 1] + samples[mid])
    }
}

#[test]
fn test_ng_w16_3_floatmath_preserveall_zero_fp_spills_and_beats_luajit() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = read_floatmath_source();
    let mut vm = compile_embedded_vm(&src, true);

    // Inspect IR to confirm PreserveAll cold trampolines are emitted for FP loops.
    let mir_fn = vm
        .mir_functions
        .get("__script")
        .cloned()
        .expect("__script MIR function");
    let mut ir_compiler = CraneliftJitCompiler::new().expect("compiler");
    let ir = ir_compiler
        .dump_function_ir(&mir_fn)
        .expect("dump_function_ir __script");
    assert!(
        ir.contains("__clif_preserve_all_safepoint_slow"),
        "floatmath IR must route cold safepoint slow path through __clif_preserve_all_safepoint_slow:\n{ir}"
    );
    assert!(
        ir.contains("__clif_preserve_all_gas_slow"),
        "floatmath IR must route cold gas slow path through __clif_preserve_all_gas_slow:\n{ir}"
    );
    assert!(
        ir.contains("preserve_all"),
        "floatmath IR must use preserve_all calling convention for cold trampolines:\n{ir}"
    );

    // Pre-compile before timing execution.
    vm.compile_cranelift().expect("compile_cranelift");

    // Warmup run + exact IEEE-754 parity check.
    let expected_f64 = floatmath_rust_oracle();
    let e0 = vm.cranelift_entry_count;
    vm.call("__script", vec![]).expect("floatmath warmup run");
    assert!(
        vm.cranelift_entry_count > e0,
        "floatmath must execute natively in Cranelift JIT"
    );
    let stdout = vm.console.stdout_string();
    let trimmed = stdout.trim();
    assert_eq!(
        trimmed, "199999960.06314665",
        "floatmath output must match exact IEEE-754 expected value"
    );
    let parsed: f64 = trimmed.parse().expect("parse floatmath stdout");
    assert_eq!(
        parsed.to_bits(),
        expected_f64.to_bits(),
        "floatmath Cranelift JIT output must match Rust IEEE-754 oracle bit-for-bit"
    );

    // Measure median execution time across 5 runs (Cranelift codegen is always opt_level=speed).
    let mut samples = Vec::with_capacity(5);
    for _ in 0..5 {
        vm.console = VirtualConsole::captured();
        let t0 = Instant::now();
        vm.call("__script", vec![]).expect("floatmath timed run");
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(vm.console.stdout_string().trim(), "199999960.06314665");
        samples.push(elapsed_ms);
    }
    let med = median_ms(&mut samples);
    eprintln!("[NG-W16-3] floatmath median = {med:.3} ms (samples={samples:?})");
    assert!(
        med <= 29.0,
        "floatmath median wall-clock {med:.3} ms exceeded 29.0 ms target (samples={samples:?})"
    );
}

#[test]
fn test_ng_w16_3_sec01_exact_gas_preservation_under_preserveall_trampolines() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
        function fp_gas_loop(n: int): float
            local x: float = 0.0
            local acc: float = 0.0
            local i: int = 0
            while i < n do
                x = x + 0.25
                acc = acc + x * 1.5 - acc * 0.25
                i = i + 1
            end
            return acc
        end
    "#;

    // Verify IR uses PreserveAll trampolines for this FP loop.
    let probe_vm = compile_embedded_vm(src, true);
    let mir_fn = probe_vm
        .mir_functions
        .get("fp_gas_loop")
        .cloned()
        .expect("fp_gas_loop MIR");
    let mut ir_comp = CraneliftJitCompiler::new().expect("compiler");
    let ir = ir_comp
        .dump_function_ir(&mir_fn)
        .expect("dump_function_ir fp_gas_loop");
    assert!(
        ir.contains("__clif_preserve_all_safepoint_slow")
            && ir.contains("__clif_preserve_all_gas_slow"),
        "fp_gas_loop must use PreserveAll trampolines:\n{ir}"
    );

    // Exact SEC-01 gas budget checks across boundary budgets 0, 1, 50 -> 51, 63, 64, 65.
    // `vm.call("fp_gas_loop", [49])` charges 1 base call gas + 50 loop header checks = 51 total gas.
    let n = 49_i64;
    let oracle_res = {
        let mut vm_bc = compile_embedded_vm(src, false);
        vm_bc.call("fp_gas_loop", vec![Value::Int(n)]).expect("bc oracle")
    };
    let Value::Float(oracle_f64) = oracle_res else {
        panic!("expected float return from fp_gas_loop, got {oracle_res:?}");
    };

    for gas_limit in [0_u64, 1, 50, 51, 63, 64, 65] {
        reset_all_state();
        let mut vm_clif = compile_embedded_vm(src, true);
        vm_clif.set_gas_budget(gas_limit);
        let e0 = vm_clif.cranelift_entry_count;
        let res = vm_clif.call("fp_gas_loop", vec![Value::Int(n)]);
        assert!(
            vm_clif.cranelift_entry_count > e0,
            "fp_gas_loop must execute in Cranelift JIT for gas_limit={gas_limit}"
        );
        if gas_limit < 51 {
            let err = res.expect_err(&format!(
                "expected OutOfGas for n={n} with gas_limit={gas_limit}"
            ));
            let msg = format!("{err:?}");
            assert!(
                msg.contains("OutOfGas") || msg.contains("gas"),
                "expected OutOfGas error for gas_limit={gas_limit}, got {msg}"
            );
        } else {
            let val = res.unwrap_or_else(|e| {
                panic!("expected Ok for n={n} with gas_limit={gas_limit}, got {e:?}")
            });
            let Value::Float(got_f64) = val else {
                panic!("expected Value::Float, got {val:?}");
            };
            assert_eq!(
                got_f64.to_bits(),
                oracle_f64.to_bits(),
                "bit-exact FP result mismatch for gas_limit={gas_limit}"
            );
            assert_eq!(
                vm_clif.gas_remaining(),
                Some(gas_limit - 51),
                "exact remaining gas mismatch for gas_limit={gas_limit}"
            );
        }
    }

    // Verify crossing Q=64 quantum boundary (n = 129 => 1 base + 130 header checks = 131 gas)
    // with budgets 63, 64, 65, 130 -> 131.
    let n_large = 129_i64;
    let oracle_large = {
        let mut vm_bc = compile_embedded_vm(src, false);
        match vm_bc.call("fp_gas_loop", vec![Value::Int(n_large)]).expect("bc oracle large") {
            Value::Float(v) => v,
            other => panic!("expected float, got {other:?}"),
        }
    };
    for gas_limit in [63_u64, 64, 65, 130, 131] {
        reset_all_state();
        let mut vm_clif = compile_embedded_vm(src, true);
        vm_clif.set_gas_budget(gas_limit);
        let res = vm_clif.call("fp_gas_loop", vec![Value::Int(n_large)]);
        if gas_limit < 131 {
            assert!(
                res.is_err(),
                "expected OutOfGas for n={n_large} with gas_limit={gas_limit}"
            );
        } else {
            let Value::Float(got) = res.expect("expected Ok at gas_limit=131") else {
                panic!("expected float");
            };
            assert_eq!(got.to_bits(), oracle_large.to_bits());
            assert_eq!(vm_clif.gas_remaining(), Some(0));
        }
    }

    // Under test-deopt, force deopt at loop safepoint polls across PreserveAll trampolines.
    #[cfg(feature = "test-deopt")]
    {
        for poll_ordinal in [1_u64, 2, 64, 65] {
            reset_all_state();
            let mut vm_clif = compile_embedded_vm(src, true);
            safepoint_request_deopt_on_poll(Some("fp_gas_loop".to_string()), poll_ordinal);
            let e0 = vm_clif.cranelift_entry_count;
            let val = vm_clif
                .call("fp_gas_loop", vec![Value::Int(n_large)])
                .unwrap_or_else(|e| panic!("deopt at poll {poll_ordinal} failed: {e:?}"));
            assert!(
                vm_clif.cranelift_entry_count > e0,
                "must enter Cranelift JIT before deopt at poll {poll_ordinal}"
            );
            let Value::Float(got) = val else {
                panic!("expected Value::Float after deopt, got {val:?}");
            };
            assert_eq!(
                got.to_bits(),
                oracle_large.to_bits(),
                "bit-exact FP state restoration failed after deopt at poll {poll_ordinal}"
            );
        }
    }
}

#[test]
fn test_ng_w16_3_string_eq_and_1byte_sub_zero_heap_handle_growth() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    // 1. Direct MirContext check over 100,000 iterations of string.sub(s, i, i) == "5".
    {
        let ctx = MirContext::new();
        let _guard = enter_mir_context(&ctx);
        let s_handle = ctx.register_handle(Value::string("0123456789"));
        let five_handle = ctx.intern_string("5");
        let handles_before = ctx.heap_values.borrow().len();

        let mut matches = 0_i64;
        for iter in 0..100_000_i64 {
            let pos = (iter % 10) + 1;
            let ch = lust_mir_string_sub(0, s_handle, pos, pos);
            assert_eq!(ctx.last_status(), MIR_STATUS_OK);
            if lust_mir_string_eq(ch, five_handle) != 0 {
                matches += 1;
            }
        }
        assert_eq!(matches, 10_000);
        let handles_after = ctx.heap_values.borrow().len();
        assert_eq!(
            handles_after - handles_before,
            0,
            "100,000 iterations of 1-byte ASCII string.sub + string.eq must grow heap_values by 0 handles"
        );
    }

    // 2. Full Cranelift JIT execution check over 100,000 iterations.
    let src = r#"
        function count_fives(s: string, reps: int): int
            local i: int = 0
            local hits: int = 0
            while i < reps do
                local pos: int = (i % 10) + 1
                local ch: string = string.sub(s, pos, pos)
                if ch == "5" then
                    hits = hits + 1
                end
                i = i + 1
            end
            return hits
        end
    "#;
    let mut vm = compile_embedded_vm(src, true);
    vm.compile_cranelift().expect("compile_cranelift");

    // Pre-register argument string and run natively; verify zero dirty handle allocations inside loop.
    let e0 = vm.cranelift_entry_count;
    let dirty_before = vm.mir_context.dirty_allocations_count();
    let res = vm
        .call(
            "count_fives",
            vec![Value::string("0123456789"), Value::Int(100_000)],
        )
        .expect("count_fives call");
    assert!(
        vm.cranelift_entry_count > e0,
        "count_fives must execute natively in Cranelift JIT"
    );
    assert_eq!(res, Value::Int(10_000));
    let dirty_after = vm.mir_context.dirty_allocations_count();
    // Only the single marshaled `s` argument handle may be registered at entry; 0 handles inside the 100k loop!
    assert!(
        dirty_after - dirty_before <= 1,
        "100,000 iterations of 1-byte string.sub + eq in Cranelift JIT must not allocate handles per iteration (dirty delta={})",
        dirty_after - dirty_before
    );
    assert_eq!(vm.mir_context.heap_values.borrow().len(), 0);
}

#[test]
fn test_ng_w16_3_schema_json_decode_pipeline_cranelift_under_45ms() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = read_schema_json_source();
    let mut vm = compile_embedded_vm(&src, true);

    // Verify IR of `parse_int` uses inline status check instead of calling lust_mir_get_status after string.sub.
    {
        let parse_int_mir = vm
            .mir_functions
            .get("parse_int")
            .cloned()
            .expect("missing MIR function parse_int");
        let mut ir_comp = CraneliftJitCompiler::new().expect("compiler");
        let ir = ir_comp
            .dump_function_ir(&parse_int_mir)
            .expect("dump_function_ir parse_int");
        // `parse_int` has no MIR-to-MIR calls; only emergency_exit_bb may reference lust_mir_get_status.
        let status_calls = ir.matches("lust_mir_get_status").count();
        assert!(
            status_calls <= 1,
            "parse_int should use inline last_status load instead of calling lust_mir_get_status after string.sub:\n{ir}"
        );
    }

    vm.compile_cranelift().expect("compile_cranelift");

    // Warmup run + exact checksum verification.
    let e0 = vm.cranelift_entry_count;
    vm.call("__script", vec![]).expect("schema_json_decode_pipeline warmup");
    assert!(
        vm.cranelift_entry_count > e0,
        "schema_json_decode_pipeline must run natively in Cranelift JIT"
    );
    assert_eq!(
        vm.console.stdout_string().trim(),
        "8224993",
        "schema_json_decode_pipeline exact checksum mismatch"
    );

    // Timed runs.
    let runs = if cfg!(debug_assertions) { 1 } else { 3 };
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        vm.console = VirtualConsole::captured();
        let t0 = Instant::now();
        vm.call("__script", vec![]).expect("schema_json_decode_pipeline timed");
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(vm.console.stdout_string().trim(), "8224993");
        samples.push(elapsed_ms);
    }
    let med = median_ms(&mut samples);
    eprintln!("[NG-W16-3] schema_json_decode_pipeline median = {med:.3} ms (debug={}, samples={samples:?})", cfg!(debug_assertions));
    if !cfg!(debug_assertions) {
        assert!(
            med <= 45.0,
            "schema_json_decode_pipeline release median {med:.3} ms exceeded 45.0 ms target (samples={samples:?})"
        );
    }
}

#[test]
fn test_ng_w16_3_queue_and_unknown_int_runtime_helpers() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);

    // 1. Native Queue helpers: FIFO ordering, capacity limit, empty/invalid errors.
    let q = lust_mir_queue_new(2);
    assert!(q > 0);
    assert_eq!(ctx.last_status(), MIR_STATUS_OK);
    assert_eq!(lust_mir_queue_offer_int(q, 101), 1);
    assert_eq!(lust_mir_queue_offer_int(q, 202), 1);
    assert_eq!(lust_mir_queue_offer_int(q, 303), 0, "full queue must return 0");
    assert_eq!(ctx.last_status(), MIR_STATUS_OK);
    assert_eq!(lust_mir_queue_take_int(q), 101);
    assert_eq!(lust_mir_queue_take_int(q), 202);
    assert_eq!(ctx.last_status(), MIR_STATUS_OK);

    // Taking from empty queue sets OOB status.
    assert_eq!(lust_mir_queue_take_int(q), 0);
    assert_eq!(ctx.last_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);
    ctx.set_status(MIR_STATUS_OK);

    // Negative capacity sets OOB status.
    assert_eq!(lust_mir_queue_new(-1), 0);
    assert_eq!(ctx.last_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);
    ctx.set_status(MIR_STATUS_OK);

    // 2. Unknown int check/unwrap helpers for both raw Value::Int and LuaValue.Int.
    let raw_int_h = ctx.register_handle(Value::Int(42));
    let lua_int_h = ctx.register_handle(Value::Enum(EnumObject::new(
        "LuaValue",
        "Int",
        Some(vec![Value::Int(-77)]),
    )));
    let lua_str_h = ctx.register_handle(Value::Enum(EnumObject::new(
        "LuaValue",
        "String",
        Some(vec![Value::string("hi")]),
    )));
    let str_h = ctx.intern_string("hello");

    assert_eq!(lust_mir_unknown_is_int(raw_int_h), 1);
    assert_eq!(lust_mir_unknown_unwrap_int(raw_int_h), 42);
    assert_eq!(lust_mir_unknown_is_int(lua_int_h), 1);
    assert_eq!(lust_mir_unknown_unwrap_int(lua_int_h), -77);
    assert_eq!(lust_mir_unknown_is_int(lua_str_h), 0);
    assert_eq!(lust_mir_unknown_is_int(str_h), 0);
    assert_eq!(ctx.last_status(), MIR_STATUS_OK);

    assert_eq!(lust_mir_unknown_unwrap_int(lua_str_h), 0);
    assert_eq!(ctx.last_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);
    ctx.set_status(MIR_STATUS_OK);

    // 3. Zero-clone with_str_handle / with_str_pair + 1-byte ASCII & empty handles.
    let empty_h = ctx.intern_string("");
    let a_h = ctx.intern_string("a");
    assert_eq!(lust_mir_string_len(empty_h), 0);
    assert_eq!(lust_mir_string_len(a_h), 1);
    assert_eq!(ctx.with_str_handle(a_h, |s| s == "a"), Some(true));
    assert_eq!(
        ctx.with_str_pair(a_h, str_h, |a, b| (a.len(), b.len())),
        Some((1, 5))
    );

    // 4. OOM check on queue creation respects memory limit.
    ctx.set_max_memory_limit(ctx.memory_used() + 16);
    assert_eq!(lust_mir_queue_new(64), 0);
    assert_eq!(ctx.last_status(), MIR_STATUS_ERR_OOM);
}

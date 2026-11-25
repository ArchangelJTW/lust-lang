#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::sync::atomic::Ordering;
use std::sync::Mutex;

use lust::bytecode::Value;
use lust::mir::cranelift_backend::{
    reset_native_instrumentation_counters, safepoint_reset, CRANELIFT_NATIVE_ENTRY_COUNT,
};
use lust::{EmbeddedProgram, LustConfig};

static GLOBAL_COUNTER_LOCK: Mutex<()> = Mutex::new(());

fn compile_cranelift_program(src: &str) -> EmbeddedProgram {
    let mut cfg = LustConfig::default().with_cranelift_jit(true);
    cfg.set_use_mir(false);
    EmbeddedProgram::builder()
        .with_config(cfg)
        .module("main", src)
        .entry_module("main")
        .compile()
        .expect("compile Cranelift EmbeddedProgram")
}

#[test]
fn test_ng_w10_3_direct_scalar_call_handle_zero_dirty_allocs() {
    let _lock = GLOBAL_COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();
    reset_native_instrumentation_counters();

    let src = r#"
function inc(x: int): int
    return x + 1
end
"#;
    let mut program = compile_cranelift_program(src);

    let h_val = program
        .get_typed_handle::<Value, Value>("main.inc")
        .expect("get Value TypedHandle");
    let h_i64 = program
        .get_typed_handle::<i64, i64>("main.inc")
        .expect("get i64 TypedHandle");

    assert_eq!(program.cranelift_native_entry_count(), 0);
    assert_eq!(program.vm().mir_context.dirty_allocations_count(), 0);

    // First call warms up Cranelift and executes via direct scalar C-ABI fast-path.
    let g0 = CRANELIFT_NATIVE_ENTRY_COUNT.load(Ordering::SeqCst);
    let res = program
        .call_handle(&h_val, Value::Int(41))
        .expect("call_handle(Value::Int(41))");
    let g1 = CRANELIFT_NATIVE_ENTRY_COUNT.load(Ordering::SeqCst);

    assert_eq!(res, Value::Int(42));
    assert_eq!(program.cranelift_native_entry_count(), 1);
    assert_eq!(program.vm().cranelift_entry_count(), 1);
    assert_eq!(
        g1 - g0,
        1,
        "JIT prologue must record exactly 1 native entry (no double counting)"
    );
    assert_eq!(
        program.vm().mir_context.dirty_allocations_count(),
        0,
        "direct pure-scalar call must leave dirty_allocations_count == 0"
    );
    assert_eq!(
        program.vm().cranelift_is_pure_scalar_abi("main.inc"),
        Some(true)
    );

    // Second (warm) call via typed i64 handle
    let g2 = CRANELIFT_NATIVE_ENTRY_COUNT.load(Ordering::SeqCst);
    let res2 = program
        .call_handle(&h_i64, 99i64)
        .expect("call_handle(99i64)");
    let g3 = CRANELIFT_NATIVE_ENTRY_COUNT.load(Ordering::SeqCst);

    assert_eq!(res2, 100i64);
    assert_eq!(program.cranelift_native_entry_count(), 2);
    assert_eq!(g3 - g2, 1);
    assert_eq!(program.vm().mir_context.dirty_allocations_count(), 0);
}

#[test]
fn test_ng_w10_3_direct_scalar_call_typed_0_to_4_args_all_scalar_types() {
    let _lock = GLOBAL_COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let src = r#"
function ret_int_0(): int
    return 77
end

function ret_float_0(): float
    return 3.25
end

function ret_bool_0(): bool
    return true
end

function ret_unit_0()
    local _x: int = 1 + 2
end

function neg_bool_1(b: bool): bool
    return not b
end

function scale_float_1(x: float): float
    return x * 2.5
end

function unit_bool_1(b: bool)
    local _k: int = 0
    if b then
        _k = 1
    end
end

function mix_2_if(a: int, b: float): float
    return b + 1.5
end

function mix_2_bi(flag: bool, n: int): bool
    if flag then
        return n > 10
    end
    return n < 0
end

function mix_3_ifb(a: int, b: float, flag: bool): float
    if flag then
        return b + 10.0
    end
    return b - 10.0
end

function mix_3_bbi(a: bool, b: bool, c: int): int
    if a and b then
        return c * 2
    end
    if a or b then
        return c + 5
    end
    return c
end

function mix_4_ifbi(a: int, b: float, flag: bool, d: int): float
    if flag then
        return b + 100.0
    end
    return b - 100.0
end

function mix_4_bifb(a: bool, b: int, c: float, d: bool): int
    local acc: int = b
    if a then
        acc = acc + 10
    end
    if d then
        acc = acc + 20
    end
    if c > 0.0 then
        acc = acc + 1
    end
    return acc
end

function mix_4_unit(a: int, b: float, c: bool, d: int)
    local _s: int = a + d
    if c then
        _s = _s + 1
    end
end

function mix_4_ffff_bool(a: float, b: float, c: float, d: float): bool
    return (a + b) < (c + d)
end
"#;

    let mut program = compile_cranelift_program(src);

    let mut expected_entries = 0u64;

    macro_rules! check_typed {
        ($name:expr, $args:expr, $expected:expr, $ret:ty) => {{
            let res: $ret = program
                .call_typed($name, $args)
                .unwrap_or_else(|e| panic!("call_typed({}) failed: {e}", $name));
            assert_eq!(res, $expected, "value mismatch for {}", $name);
            expected_entries += 1;
            assert_eq!(
                program.cranelift_native_entry_count(),
                expected_entries,
                "entry count mismatch for {}",
                $name
            );
            assert_eq!(
                program.vm().cranelift_is_pure_scalar_abi($name),
                Some(true),
                "{} must be marked is_pure_scalar_abi",
                $name
            );
            assert_eq!(
                program.vm().mir_context.dirty_allocations_count(),
                0,
                "{} must have zero dirty allocations",
                $name
            );
        }};
    }

    // 0 args
    check_typed!("main.ret_int_0", (), 77i64, i64);
    check_typed!("main.ret_float_0", (), 3.25f64, f64);
    check_typed!("main.ret_bool_0", (), true, bool);
    check_typed!("main.ret_unit_0", (), (), ());

    // 1 arg
    check_typed!("main.neg_bool_1", true, false, bool);
    check_typed!("main.neg_bool_1", false, true, bool);
    check_typed!("main.scale_float_1", 4.0f64, 10.0f64, f64);
    check_typed!("main.unit_bool_1", true, (), ());

    // 2 args
    check_typed!("main.mix_2_if", (5i64, 2.5f64), 4.0f64, f64);
    check_typed!("main.mix_2_bi", (true, 20i64), true, bool);
    check_typed!("main.mix_2_bi", (true, 5i64), false, bool);
    check_typed!("main.mix_2_bi", (false, -3i64), true, bool);

    // 3 args
    check_typed!("main.mix_3_ifb", (1i64, 5.5f64, true), 15.5f64, f64);
    check_typed!("main.mix_3_ifb", (1i64, 5.5f64, false), -4.5f64, f64);
    check_typed!("main.mix_3_bbi", (true, true, 7i64), 14i64, i64);
    check_typed!("main.mix_3_bbi", (false, true, 7i64), 12i64, i64);
    check_typed!("main.mix_3_bbi", (false, false, 7i64), 7i64, i64);

    // 4 args
    check_typed!("main.mix_4_ifbi", (1i64, 2.0f64, true, 3i64), 102.0f64, f64);
    check_typed!("main.mix_4_ifbi", (1i64, 2.0f64, false, 3i64), -98.0f64, f64);
    check_typed!("main.mix_4_bifb", (true, 100i64, 1.0f64, true), 131i64, i64);
    check_typed!("main.mix_4_bifb", (false, 100i64, -1.0f64, true), 120i64, i64);
    check_typed!("main.mix_4_unit", (1i64, 2.0f64, true, 3i64), (), ());
    check_typed!("main.mix_4_ffff_bool", (1.0f64, 2.0f64, 3.0f64, 4.0f64), true, bool);
    check_typed!("main.mix_4_ffff_bool", (5.0f64, 2.0f64, 3.0f64, 4.0f64), false, bool);
}

#[test]
fn test_ng_w10_3_sec01_gas_limit_enforced_on_direct_scalar_path() {
    let _lock = GLOBAL_COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let src = r#"
function loop_sum(n: int): int
    local acc: int = 0
    local i: int = 0
    while i < n do
        acc = acc + i
        i = i + 1
    end
    return acc
end

function inc(x: int): int
    return x + 1
end
"#;
    let mut program = compile_cranelift_program(src);
    let h_loop = program
        .get_typed_handle::<i64, i64>("main.loop_sum")
        .expect("handle loop_sum");
    let h_inc = program
        .get_typed_handle::<i64, i64>("main.inc")
        .expect("handle inc");

    // Warm up compile so `is_pure_scalar_abi` is populated
    let warm = program.call_handle(&h_loop, 3i64).expect("warm call");
    assert_eq!(warm, 3);
    assert_eq!(
        program.vm().cranelift_is_pure_scalar_abi("main.loop_sum"),
        Some(true)
    );
    assert_eq!(program.vm().mir_context.dirty_allocations_count(), 0);

    // SEC-01: gas_limit = 5 with 10-iteration loop MUST fail OutOfGas via call_handle
    program.reset_gas_counter();
    program.set_gas_budget(5);
    let e_before = program.cranelift_native_entry_count();
    let err = program
        .call_handle(&h_loop, 10i64)
        .expect_err("loop_sum(10) with gas_limit=5 must fail OutOfGas");
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("Out of gas"),
        "expected Out of gas error, got: {err_msg}"
    );
    assert_eq!(
        program.cranelift_native_entry_count(),
        e_before + 1,
        "gas-limited call must execute via Cranelift direct entry"
    );
    assert!(
        program.gas_used() > 5,
        "expected gas_used > 5 after exhaustion, got {}",
        program.gas_used()
    );

    // Zero gas budget fails immediately even on 0-loop scalar function
    program.reset_gas_counter();
    program.set_gas_budget(0);
    let err0 = program
        .call_handle(&h_inc, 41i64)
        .expect_err("inc(41) with gas_limit=0 must fail OutOfGas");
    assert!(
        err0.to_string().contains("Out of gas"),
        "expected Out of gas error for limit=0, got: {err0}"
    );

    // Generous budget succeeds and deducts real gas
    program.reset_gas_counter();
    program.set_gas_budget(100);
    let ok_res = program
        .call_handle(&h_loop, 10i64)
        .expect("loop_sum(10) with gas_limit=100 must succeed");
    assert_eq!(ok_res, 45);
    assert!(
        program.gas_used() > 0 && program.gas_used() <= 100,
        "expected 0 < gas_used <= 100, got {}",
        program.gas_used()
    );
    assert_eq!(program.vm().mir_context.dirty_allocations_count(), 0);
}

#[test]
fn test_ng_w10_3_type_mismatch_and_non_scalar_fallback() {
    let _lock = GLOBAL_COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let src = r#"
function inc(x: int): int
    return x + 1
end

function sum5(a: int, b: int, c: int, d: int, e: int): int
    return a + b + c + d + e
end

function echo_str(s: string): string
    return s
end

function noisy(x: int): int
    print(x)
    return x + 1
end
"#;
    let mut program = compile_cranelift_program(src);

    // Warm up compilation
    let h_inc_val = program
        .get_typed_handle::<Value, Value>("main.inc")
        .expect("handle inc");
    assert_eq!(
        program.call_handle(&h_inc_val, Value::Int(10)).unwrap(),
        Value::Int(11)
    );
    assert_eq!(
        program.vm().cranelift_is_pure_scalar_abi("main.inc"),
        Some(true)
    );
    assert_eq!(
        program.vm().cranelift_is_pure_scalar_abi("main.sum5"),
        Some(false)
    );
    assert_eq!(
        program.vm().cranelift_is_pure_scalar_abi("main.echo_str"),
        Some(false)
    );
    assert_ne!(
        program.vm().cranelift_is_pure_scalar_abi("main.noisy"),
        Some(true)
    );

    // Runtime type mismatch via Value handle must fall back cleanly and return error
    let entries_before = program.cranelift_native_entry_count();
    let bad_float = program.call_handle(&h_inc_val, Value::Float(1.5));
    assert!(
        bad_float.is_err(),
        "passing Float to int function must fail type check"
    );
    let bad_str = program.call_handle(&h_inc_val, Value::string("oops"));
    assert!(
        bad_str.is_err(),
        "passing String to int function must fail type check"
    );
    assert_eq!(
        program.cranelift_native_entry_count(),
        entries_before,
        "type-mismatched calls must not enter Cranelift native code"
    );

    // Non-scalar and >4-arg functions still execute accurately via fallback path
    let s5: i64 = program
        .call_typed("main.sum5", (1i64, 2i64, 3i64, 4i64, 5i64))
        .expect("sum5");
    assert_eq!(s5, 15);

    let echoed: String = program
        .call_typed("main.echo_str", "hello")
        .expect("echo_str");
    assert_eq!(echoed, "hello");

    let noisy_res: i64 = program.call_typed("main.noisy", 41i64).expect("noisy");
    assert_eq!(noisy_res, 42);
}

#[test]
fn test_ng_w10_3_runtime_error_div_by_zero_on_direct_path() {
    let _lock = GLOBAL_COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let src = r#"
function div_it(a: int, b: int): int
    return a // b
end
"#;
    let mut program = compile_cranelift_program(src);
    let h = program
        .get_typed_handle::<(i64, i64), i64>("main.div_it")
        .expect("handle div_it");

    assert_eq!(program.call_handle(&h, (10, 2)).expect("10 // 2"), 5);
    assert_eq!(
        program.vm().cranelift_is_pure_scalar_abi("main.div_it"),
        Some(true)
    );

    let err = program
        .call_handle(&h, (10, 0))
        .expect_err("10 // 0 must error");
    assert!(
        err.to_string().to_lowercase().contains("division")
            || err.to_string().to_lowercase().contains("zero"),
        "expected division by zero error, got: {err}"
    );

    // Subsequent call must succeed cleanly (no stale error status left in MirContext)
    assert_eq!(program.call_handle(&h, (12, 3)).expect("12 // 3"), 4);
    assert_eq!(program.vm().mir_context.dirty_allocations_count(), 0);
}

#[test]
fn test_ng_w10_3_trace_hook_bypasses_direct_fast_path() {
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    let _lock = GLOBAL_COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let src = r#"
function inc(x: int): int
    return x + 1
end
"#;
    let mut program = compile_cranelift_program(src);
    let h = program
        .get_typed_handle::<i64, i64>("main.inc")
        .expect("handle inc");

    assert_eq!(program.call_handle(&h, 1i64).unwrap(), 2);

    let hook_hits = Arc::new(AtomicUsize::new(0));
    let hook_hits_clone = Arc::clone(&hook_hits);
    program
        .vm_mut()
        .set_return_hook(Some(Box::new(move |_, _, _| {
            hook_hits_clone.fetch_add(1, Ordering::SeqCst);
        })));

    let res = program.call_handle(&h, 41i64).expect("call with trace hook");
    program.vm_mut().clear_trace_hooks();

    assert_eq!(res, 42);
    assert!(
        hook_hits.load(Ordering::SeqCst) >= 1,
        "active return trace hook must bypass direct scalar fast-path and fire"
    );

    // Clearing trace hooks restores direct fast-path
    let e0 = program.cranelift_native_entry_count();
    assert_eq!(program.call_handle(&h, 41i64).unwrap(), 42);
    assert_eq!(program.cranelift_native_entry_count(), e0 + 1);
    assert_eq!(program.vm().mir_context.dirty_allocations_count(), 0);
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_ng_w10_3_test_deopt_spike_hook_bypasses_direct_fast_path() {
    let _lock = GLOBAL_COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let src = r#"
function sum_loop(n: int): int
    local acc: int = 0
    local i: int = 0
    while i < n do
        acc = acc + i
        i = i + 1
    end
    return acc
end
"#;
    let mut program = compile_cranelift_program(src);
    let h = program
        .get_typed_handle::<i64, i64>("main.sum_loop")
        .expect("handle sum_loop");

    // Warm up compile
    assert_eq!(program.call_handle(&h, 5i64).unwrap(), 10);
    let sites = program.vm().cranelift_deopt_recipe_sites("main.sum_loop");
    assert!(!sites.is_empty(), "expected loop deopt recipe site");
    let block_id = sites[0].0;

    // Install test_deopt_spike_hook + poll deopt request: direct fast-path must yield to
    // normal Cranelift frame dispatch with NativeInvocationGuard and resume in continuation.
    program.vm_mut().test_deopt_spike_hook = Some(lust::vm::TestDeoptSpikeHook::new(
        "main.sum_loop",
        block_id,
    ));
    lust::mir::cranelift_backend::safepoint_request_deopt_on_poll(
        Some("main.sum_loop".into()),
        1,
    );

    let res = program.call_handle(&h, 10i64).expect("sum_loop with deopt hook");
    safepoint_reset();
    assert_eq!(res, 45);
    assert_eq!(
        program
            .vm()
            .test_deopt_spike_hook
            .as_ref()
            .map(|h| h.triggered_count),
        Some(1),
        "test_deopt_spike_hook must trigger via full Cranelift frame path (not bypassed by direct fast-path)"
    );
    assert_eq!(
        program.vm().last_deopt_source,
        Some(lust::vm::DeoptPayloadSource::NativeExecution)
    );
}

#[test]
fn test_ng_w10_3_direct_scalar_batch_consistency() {
    let _lock = GLOBAL_COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let src = r#"
function step(val: int): int
    return (val * 3 + 7) % 10007
end
"#;
    let mut program = compile_cranelift_program(src);
    let h = program
        .get_typed_handle::<i64, i64>("main.step")
        .expect("handle step");

    let mut st = 1i64;
    let mut expected = 1i64;
    let iters = 1_000i64;
    for i in 0..iters {
        st = program.call_handle(&h, st + i % 10).expect("call_handle");
        expected = ((expected + i % 10) * 3 + 7) % 10007;
    }
    assert_eq!(st, expected);
    assert_eq!(program.cranelift_native_entry_count(), iters as u64);
    assert_eq!(program.vm().mir_context.dirty_allocations_count(), 0);
}

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;
use std::time::Instant;

use lust::bytecode::Value;
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::safepoint_request_deopt_on_poll;
use lust::mir::cranelift_backend::{
    MIR_STATUS_ERR_OOM, MIR_STATUS_OK, analyze_module_call_effects,
    lust_mir_string_append_inplace, lust_mir_string_concat, lust_mir_tostring_bool,
    lust_mir_tostring_float, lust_mir_tostring_int, safepoint_reset, take_current_sticky_error,
    take_deopt_payload,
};
use lust::mir::ir::{
    FunctionSignature, MirBinaryOp, MirConstant, MirFunction, MirModule, MirType, Operand, Place,
    Rvalue, Statement, Terminator,
};
use lust::mir::runtime_helpers::MirContext;
use lust::vfs::VirtualConsole;
use lust::vm::{VM, reset_effect_handler_state};
use lust::{EmbeddedModule, LustConfig, compile_program_with_config, load_program_from_embedded};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn lust_bin() -> &'static str {
    env!("CARGO_BIN_EXE_lust")
}

fn suite_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks")
        .join("suite")
        .join(name)
}

fn reset_all_state() {
    safepoint_reset();
    reset_effect_handler_state();
    take_deopt_payload();
    take_current_sticky_error();
}

fn parse_cranelift_stats(stderr: &str) -> (u64, u64) {
    for line in stderr.lines() {
        if let Some(rest) = line.strip_prefix("Cranelift stats:") {
            let mut entries = None;
            let mut compilations = None;
            for tok in rest.split_whitespace() {
                if let Some(v) = tok.strip_prefix("entries=") {
                    entries = v.parse::<u64>().ok();
                } else if let Some(v) = tok.strip_prefix("compilations=") {
                    compilations = v.parse::<u64>().ok();
                }
            }
            if let (Some(e), Some(c)) = (entries, compilations) {
                return (e, c);
            }
        }
    }
    panic!("Missing `Cranelift stats: entries=... compilations=...` in stderr:\n{stderr}");
}

fn compile_embedded_vm(source: &str, cranelift: bool) -> VM {
    let entries = &[EmbeddedModule {
        module: "main",
        parent: None,
        source: Some(source),
    }];
    let program = load_program_from_embedded(entries, "main").expect("load embedded program");
    let mut cfg = LustConfig::default().with_cranelift_jit(cranelift);
    cfg.set_use_mir(false);
    let mut vm = compile_program_with_config(program, &cfg).expect("compile embedded program");
    vm.console = VirtualConsole::captured();
    vm
}

#[test]
fn test_perf_t2_7_strings_concat_linear_and_cow() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let strings_path = suite_path("strings.lust");

    let vm_out = Command::new(lust_bin())
        .args(["--jit=off"])
        .arg(&strings_path)
        .output()
        .expect("run lust --jit=off strings.lust");
    assert!(
        vm_out.status.success(),
        "lust --jit=off strings.lust failed: {}",
        String::from_utf8_lossy(&vm_out.stderr)
    );
    let vm_stdout = String::from_utf8_lossy(&vm_out.stdout);
    assert_eq!(vm_stdout, "100000\n", "VM oracle stdout mismatch for strings.lust");

    let clif_out = Command::new(lust_bin())
        .args(["--jit=cranelift"])
        .arg(&strings_path)
        .env("LUST_JIT_STATS", "1")
        .output()
        .expect("run lust --jit=cranelift strings.lust");
    assert!(
        clif_out.status.success(),
        "lust --jit=cranelift strings.lust failed: {}",
        String::from_utf8_lossy(&clif_out.stderr)
    );
    let clif_stdout = String::from_utf8_lossy(&clif_out.stdout);
    let clif_stderr = String::from_utf8_lossy(&clif_out.stderr);
    assert_eq!(
        clif_stdout, vm_stdout,
        "strings.lust Cranelift stdout mismatch vs VM oracle"
    );
    let (entries, compilations) = parse_cranelift_stats(&clif_stderr);
    assert_eq!(
        (entries, compilations),
        (1, 1),
        "Expected strings.lust to execute natively in Cranelift (entries=1, compilations=1), got entries={entries} compilations={compilations}\nstderr:\n{clif_stderr}"
    );

    // Linear scaling check: T(100k) / T(10k) < 25 (ruling out O(N^2) ~100x blowup)
    let scaling_src = r#"
function build_digits(n: int): int
    local s: string = ""
    local i: int = 0
    while i < n do
        s = s .. tostring(i % 10)
        i = i + 1
    end
    return string.len(s)
end
"#;
    let mut vm_clif = compile_embedded_vm(scaling_src, true);

    // Warmup to compile Cranelift function
    let warm = vm_clif
        .call("build_digits", vec![Value::Int(100)])
        .expect("warmup build_digits");
    assert_eq!(warm, Value::Int(100));
    assert!(
        vm_clif.cranelift_entry_count >= 1,
        "build_digits must execute natively in Cranelift"
    );

    let mut samples_10k = Vec::with_capacity(3);
    let mut samples_100k = Vec::with_capacity(3);
    for _ in 0..3 {
        let t0 = Instant::now();
        let r10 = vm_clif
            .call("build_digits", vec![Value::Int(10_000)])
            .expect("build_digits(10_000)");
        let dt10 = t0.elapsed().as_nanos() as f64;
        assert_eq!(r10, Value::Int(10_000));
        samples_10k.push(dt10);

        let t1 = Instant::now();
        let r100 = vm_clif
            .call("build_digits", vec![Value::Int(100_000)])
            .expect("build_digits(100_000)");
        let dt100 = t1.elapsed().as_nanos() as f64;
        assert_eq!(r100, Value::Int(100_000));
        samples_100k.push(dt100);
    }
    samples_10k.sort_by(|a, b| a.total_cmp(b));
    samples_100k.sort_by(|a, b| a.total_cmp(b));
    let med_10k = samples_10k[1].max(1.0);
    let med_100k = samples_100k[1];
    let ratio = med_100k / med_10k;
    eprintln!(
        "PERF-T2-7 scaling: T(10k)={:.3}ms T(100k)={:.3}ms ratio={:.2}x",
        med_10k / 1e6,
        med_100k / 1e6,
        ratio
    );
    assert!(
        ratio < 25.0,
        "Expected linear O(N) scaling (T(100k)/T(10k) < 25.0), got ratio={ratio:.2} (10k={med_10k:.0}ns, 100k={med_100k:.0}ns)"
    );
}

#[test]
fn test_perf_t2_7_adv_cow_aliasing_preserves_immutability() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
function test_literal_alias(): string
    local a: string = "x"
    local b: string = a
    a = a .. "y"
    return b .. "|" .. a
end

function test_heap_alias_outgoing(): string
    local a: string = "x" .. "1"
    local b: string = a
    a = a .. "2"
    return b .. "|" .. a
end

function test_heap_alias_incoming(): string
    local b: string = "x" .. "1"
    local a: string = b
    a = a .. "2"
    return b .. "|" .. a
end

function test_loop_alias_capture(): string
    local a: string = "x" .. "1"
    local b: string = ""
    local i: int = 0
    while i < 5 do
        if i == 2 then
            b = a
        end
        a = a .. "z"
        i = i + 1
    end
    return b .. "|" .. a
end

function test_self_concat(): string
    local s: string = "ab"
    s = s .. s
    s = s .. s
    return s
end

function test_tostring_alias(): string
    local a: string = "x" .. "1"
    local b: string = tostring(a)
    a = a .. "2"
    return b .. "|" .. a
end
"#;

    let mut vm_off = compile_embedded_vm(src, false);
    let mut vm_clif = compile_embedded_vm(src, true);

    let cases = [
        ("test_literal_alias", "x|xy"),
        ("test_heap_alias_outgoing", "x1|x12"),
        ("test_heap_alias_incoming", "x1|x12"),
        ("test_loop_alias_capture", "x1zz|x1zzzzz"),
        ("test_self_concat", "abababab"),
        ("test_tostring_alias", "x1|x12"),
    ];

    let entries_before = vm_clif.cranelift_entry_count;
    for (name, expected) in cases {
        let r_off = vm_off.call(name, vec![]).expect("vm_off call");
        let r_clif = vm_clif.call(name, vec![]).expect("vm_clif call");
        assert_eq!(r_off, Value::string(expected), "VM oracle mismatch for {name}");
        assert_eq!(r_clif, r_off, "Cranelift mismatch for {name}");
    }
    let entries_after = vm_clif.cranelift_entry_count;
    assert_eq!(
        entries_after - entries_before,
        cases.len() as u64,
        "All COW aliasing test functions must execute natively in Cranelift"
    );
}

#[test]
fn test_perf_t2_7_adv_oom_rollback_leaves_lhs_unmodified() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    // Part 1: Direct runtime helper OOM rollback test asserting MIR_STATUS_ERR_OOM
    // AND that the lhs heap string content is completely unmodified on OOM.
    {
        let ctx = MirContext::new();
        let _guard = ctx.enter();
        let h_a = ctx.intern_string("hello_");
        let h_b = ctx.intern_string("world");
        let lhs_handle = lust_mir_string_concat(h_a, h_b);
        assert!(lhs_handle > 0, "expected heap string handle > 0");
        assert_eq!(ctx.last_status(), MIR_STATUS_OK);
        assert_eq!(
            ctx.get_handle(lhs_handle),
            Some(Value::string("hello_world"))
        );

        // Set a tight memory limit allowing only 3 more bytes, then attempt to append 10 bytes.
        let used_before = ctx.memory_used();
        ctx.set_max_memory_limit(used_before + 3);
        let chunk = ctx.intern_string("0123456789");
        let res = lust_mir_string_append_inplace(lhs_handle, chunk);
        assert_eq!(res, 0, "OOM append must return 0 handle");
        assert_eq!(
            ctx.last_status(),
            MIR_STATUS_ERR_OOM,
            "OOM append must set MIR_STATUS_ERR_OOM"
        );
        assert_eq!(
            ctx.memory_used(),
            used_before,
            "OOM append must roll back memory_used"
        );
        assert_eq!(
            ctx.get_handle(lhs_handle),
            Some(Value::string("hello_world")),
            "OOM append must leave lhs string completely unmodified"
        );

        // Also verify COW fallback when Rc::strong_count > 1 (external Rc alias).
        ctx.set_status(MIR_STATUS_OK);
        ctx.set_max_memory_limit(i64::MAX);
        let external_alias = ctx.get_handle(lhs_handle).expect("alias handle");
        let suffix = ctx.intern_string("_extra");
        let new_handle = lust_mir_string_append_inplace(lhs_handle, suffix);
        assert!(
            new_handle > 0 && new_handle != lhs_handle,
            "Aliased Rc (strong_count > 1) must COW-allocate a new handle"
        );
        assert_eq!(external_alias, Value::string("hello_world"));
        assert_eq!(
            ctx.get_handle(lhs_handle),
            Some(Value::string("hello_world"))
        );
        assert_eq!(
            ctx.get_handle(new_handle),
            Some(Value::string("hello_world_extra"))
        );

        // Verify tostring helpers also enforce max_memory_limit and roll back memory_used on OOM.
        let used_ts = ctx.memory_used();
        ctx.set_max_memory_limit(used_ts);
        ctx.set_status(MIR_STATUS_OK);
        assert_eq!(lust_mir_tostring_int(5), 0);
        assert_eq!(ctx.last_status(), MIR_STATUS_ERR_OOM);
        assert_eq!(ctx.memory_used(), used_ts);

        ctx.set_status(MIR_STATUS_OK);
        assert_eq!(lust_mir_tostring_int(123456), 0);
        assert_eq!(ctx.last_status(), MIR_STATUS_ERR_OOM);
        assert_eq!(ctx.memory_used(), used_ts);

        ctx.set_status(MIR_STATUS_OK);
        assert_eq!(lust_mir_tostring_float(1.5), 0);
        assert_eq!(ctx.last_status(), MIR_STATUS_ERR_OOM);
        assert_eq!(ctx.memory_used(), used_ts);

        ctx.set_status(MIR_STATUS_OK);
        assert_eq!(lust_mir_tostring_bool(1), 0);
        assert_eq!(ctx.last_status(), MIR_STATUS_ERR_OOM);
        assert_eq!(ctx.memory_used(), used_ts);
    }

    // Part 2: End-to-end VM + Cranelift loop OOM enforcement.
    let src = r#"
function grow_until_oom(n: int): int
    local s: string = ""
    local i: int = 0
    while i < n do
        s = s .. "0123456789"
        i = i + 1
    end
    return string.len(s)
end
"#;
    let mut vm_clif = compile_embedded_vm(src, true);
    // Warmup without tight limit to verify native entry
    let ok = vm_clif
        .call("grow_until_oom", vec![Value::Int(2)])
        .expect("warmup grow_until_oom");
    assert_eq!(ok, Value::Int(20));
    assert!(
        vm_clif.cranelift_entry_count >= 1,
        "grow_until_oom must execute natively in Cranelift"
    );

    vm_clif.set_memory_budget_kb(1).expect("set memory budget");
    let err = vm_clif.call("grow_until_oom", vec![Value::Int(200)]);
    assert!(
        err.is_err(),
        "Expected OOM error when memory budget is exceeded in Cranelift string append loop"
    );
}

#[test]
fn test_perf_t2_7_tostring_parity_int_float_bool_str() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
function fmt_int(x: int): string
    return tostring(x)
end

function fmt_float(x: float): string
    return tostring(x)
end

function fmt_bool(x: bool): string
    return tostring(x)
end

function fmt_str(x: string): string
    return tostring(x)
end
"#;

    let mut vm_off = compile_embedded_vm(src, false);
    let mut vm_clif = compile_embedded_vm(src, true);

    let entries_before = vm_clif.cranelift_entry_count;
    let mut total_calls: u64 = 0;

    for v in [0i64, 1, 9, 10, 42, 255, 256, -1, -123456, i64::MAX, i64::MIN] {
        let r_off = vm_off.call("fmt_int", vec![Value::Int(v)]).unwrap();
        let r_clif = vm_clif.call("fmt_int", vec![Value::Int(v)]).unwrap();
        assert_eq!(r_clif, r_off, "tostring(int {v}) parity mismatch");
        total_calls += 1;
    }

    for v in [0.0f64, -0.0, 1.0, -1.0, 1.5, -3.25, 0.125, 1e20, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let r_off = vm_off.call("fmt_float", vec![Value::Float(v)]).unwrap();
        let r_clif = vm_clif.call("fmt_float", vec![Value::Float(v)]).unwrap();
        assert_eq!(r_clif, r_off, "tostring(float {v}) parity mismatch");
        total_calls += 1;
    }

    for v in [true, false] {
        let r_off = vm_off.call("fmt_bool", vec![Value::Bool(v)]).unwrap();
        let r_clif = vm_clif.call("fmt_bool", vec![Value::Bool(v)]).unwrap();
        assert_eq!(r_clif, r_off, "tostring(bool {v}) parity mismatch");
        total_calls += 1;
    }

    for s in ["", "hello", "with spaces", "12345"] {
        let r_off = vm_off.call("fmt_str", vec![Value::string(s)]).unwrap();
        let r_clif = vm_clif.call("fmt_str", vec![Value::string(s)]).unwrap();
        assert_eq!(r_clif, r_off, "tostring(str {s:?}) parity mismatch");
        total_calls += 1;
    }

    let entries_after = vm_clif.cranelift_entry_count;
    assert_eq!(
        entries_after - entries_before,
        total_calls,
        "All tostring parity calls must execute natively in Cranelift"
    );
}

#[test]
fn test_perf_t2_7_b07_strings_compiles_native_and_matches_vm() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
function b07_strings(n: int): int
    local acc: string = ""
    local i: int = 0
    while i < n do
        if i % 3 == 0 then
            acc = acc .. "abc"
        else
            acc = acc .. "xy"
        end
        i = i + 1
    end
    return string.len(acc)
end
"#;
    let mut vm_off = compile_embedded_vm(src, false);
    let mut vm_clif = compile_embedded_vm(src, true);

    for n in [0i64, 1, 50, 500, 5_000] {
        let r_off = vm_off.call("b07_strings", vec![Value::Int(n)]).unwrap();
        let r_clif = vm_clif.call("b07_strings", vec![Value::Int(n)]).unwrap();
        assert_eq!(r_clif, r_off, "b07_strings({n}) parity mismatch");
    }
    assert_eq!(
        vm_clif.cranelift_entry_count, 5,
        "All b07_strings calls must execute natively in Cranelift"
    );
}

#[test]
fn test_perf_t2_7_non_failing_whitelist_excludes_concat_and_tostring() {
    let span = lust::ast::Span::dummy();

    let mut fn_concat = MirFunction::new(
        FunctionSignature {
            name: "do_concat".into(),
            param_types: vec![MirType::String, MirType::String],
            return_type: MirType::String,
            is_public: true,
            effects: None,
        },
        span,
    );
    let p0 = fn_concat.add_param(Some("a".into()), MirType::String, span);
    let p1 = fn_concat.add_param(Some("b".into()), MirType::String, span);
    let res = fn_concat.add_local(Some("r".into()), MirType::String, true, span);
    let bb0 = fn_concat.entry;
    fn_concat.blocks[bb0.0].statements.push(Statement::Assign(
        Place::Local(res),
        Rvalue::BinaryOp(
            MirBinaryOp::Concat,
            Operand::Copy(Place::Local(p0)),
            Operand::Copy(Place::Local(p1)),
        ),
        span,
    ));
    fn_concat.blocks[bb0.0].terminator =
        Some(Terminator::Return(Some(Operand::Move(Place::Local(res))), span));

    let mut fn_tostring = MirFunction::new(
        FunctionSignature {
            name: "do_tostring".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::String,
            is_public: true,
            effects: None,
        },
        span,
    );
    let pi = fn_tostring.add_param(Some("x".into()), MirType::Int, span);
    let ts_res = fn_tostring.add_local(Some("s".into()), MirType::String, true, span);
    let bb_ts0 = fn_tostring.entry;
    let bb_ts1 = fn_tostring.new_block();
    fn_tostring.blocks[bb_ts0.0].terminator = Some(Terminator::Call {
        callee: "tostring".into(),
        args: vec![Operand::Copy(Place::Local(pi))],
        destination: Place::Local(ts_res),
        success: bb_ts1,
        error: None,
        span,
    });
    fn_tostring.blocks[bb_ts1.0].terminator = Some(Terminator::Return(
        Some(Operand::Move(Place::Local(ts_res))),
        span,
    ));

    let mut fn_pure = MirFunction::new(
        FunctionSignature {
            name: "pure_add".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let pa = fn_pure.add_param(Some("a".into()), MirType::Int, span);
    let pr = fn_pure.add_local(Some("r".into()), MirType::Int, true, span);
    let bb_p0 = fn_pure.entry;
    fn_pure.blocks[bb_p0.0].statements.push(Statement::Assign(
        Place::Local(pr),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(pa)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        span,
    ));
    fn_pure.blocks[bb_p0.0].terminator =
        Some(Terminator::Return(Some(Operand::Move(Place::Local(pr))), span));

    let mut module = MirModule::new("perf_t2_7_test".into());
    module.add_function(fn_concat);
    module.add_function(fn_tostring);
    module.add_function(fn_pure);

    let effects = analyze_module_call_effects(&module, false);
    assert!(
        effects["do_concat"].1,
        "MirBinaryOp::Concat allocates and can OOM, so can_fail_transitively MUST be true"
    );
    assert!(
        effects["do_tostring"].1,
        "tostring allocates and can OOM, so can_fail_transitively MUST be true"
    );
    assert!(
        !effects["pure_add"].1,
        "sanity check: pure_add should remain non-failing"
    );
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_perf_t2_7_adv_deopt_mid_concat_loop() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
function build_digits_deopt(n: int): string
    local s: string = ""
    local i: int = 0
    while i < n do
        s = s .. tostring(i % 10)
        i = i + 1
    end
    return s
end
"#;
    let mut vm_off = compile_embedded_vm(src, false);
    let mut vm_clif = compile_embedded_vm(src, true);

    let expected = vm_off
        .call("build_digits_deopt", vec![Value::Int(50)])
        .expect("vm_off build_digits_deopt");

    safepoint_request_deopt_on_poll(Some("build_digits_deopt".to_string()), 17);
    let actual = vm_clif
        .call("build_digits_deopt", vec![Value::Int(50)])
        .expect("vm_clif build_digits_deopt with mid-loop deopt");
    assert_eq!(
        actual, expected,
        "Deopt mid-loop during COW string concat must resume in bytecode VM with exact parity"
    );
    assert!(
        vm_clif.cranelift_entry_count >= 1,
        "build_digits_deopt must have entered Cranelift before deopt"
    );
}

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

use lust::bytecode::Value;
use lust::mir::cranelift_backend::{
    safepoint_reset, take_current_sticky_error, take_deopt_payload,
};
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::{get_current_sticky_error, safepoint_request_deopt_on_poll};
use lust::mir::ir::Terminator;
use lust::mir::verifier::MirVerifier;
use lust::vfs::VirtualConsole;
#[cfg(feature = "test-deopt")]
use lust::vm::DeoptPayloadSource;
use lust::vm::{VM, reset_effect_handler_state};
use lust::{EmbeddedModule, LustConfig, compile_program_with_config, load_program_from_embedded};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn lust_bin() -> &'static str {
    env!("CARGO_BIN_EXE_lust")
}

fn methods_lust_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks")
        .join("suite")
        .join("methods.lust")
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

fn find_fn_name(vm: &VM, suffix: &str) -> String {
    let mut idx = 0;
    while let Some(name) = vm.function_name(idx) {
        if name.ends_with(suffix) {
            return name.to_string();
        }
        idx += 1;
    }
    panic!("Function ending with `{suffix}` not found in VM");
}

// =========================================================================
// Core PERF-T2-3 Test: methods.lust Cranelift + Inlining + CallMethod Dispatch
// =========================================================================

#[test]
fn test_perf_t2_3_user_struct_method_cranelift() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let methods_path = methods_lust_path();
    let methods_src =
        std::fs::read_to_string(&methods_path).expect("read benchmarks/suite/methods.lust");

    // 1. CLI execution: `--jit=cranelift` with `LUST_JIT_STATS=1` vs `--jit=off`
    let out_jit = Command::new(lust_bin())
        .args(["--jit=cranelift"])
        .arg(&methods_path)
        .env("LUST_JIT_STATS", "1")
        .env_remove("LUST_JIT")
        .env_remove("LUST_JIT_BACKEND")
        .env_remove("LUST_BACKEND")
        .output()
        .expect("run lust --jit=cranelift methods.lust");
    assert!(
        out_jit.status.success(),
        "lust --jit=cranelift methods.lust failed: {}",
        String::from_utf8_lossy(&out_jit.stderr)
    );
    let jit_stdout = String::from_utf8_lossy(&out_jit.stdout);
    let jit_stderr = String::from_utf8_lossy(&out_jit.stderr);
    assert_eq!(
        jit_stdout, "49999995000000\n",
        "methods.lust --jit=cranelift stdout mismatch"
    );
    let (cli_entries, cli_compilations) = parse_cranelift_stats(&jit_stderr);
    assert!(
        cli_entries >= 1 && cli_compilations >= 1,
        "Expected methods.lust to compile and enter Cranelift (entries >= 1, compilations >= 1), got entries={cli_entries}, compilations={cli_compilations}\nstderr:\n{jit_stderr}"
    );

    let out_vm = Command::new(lust_bin())
        .args(["--jit=off"])
        .arg(&methods_path)
        .env_remove("LUST_JIT")
        .env_remove("LUST_JIT_BACKEND")
        .env_remove("LUST_BACKEND")
        .output()
        .expect("run lust --jit=off methods.lust");
    assert!(
        out_vm.status.success(),
        "lust --jit=off methods.lust failed: {}",
        String::from_utf8_lossy(&out_vm.stderr)
    );
    assert_eq!(
        out_jit.stdout, out_vm.stdout,
        "exact byte parity required between --jit=cranelift and --jit=off on methods.lust"
    );

    // 2. Embedded VM API inspection: Counter:bump lowered to MIR, c:bump(i) lowered to
    //    Terminator::Call, and PERF-T1-4 inlines Counter:bump into __script.
    let mut vm = compile_embedded_vm(&methods_src, true);
    let bump_name = vm
        .mir_functions
        .keys()
        .find(|k| k.ends_with("Counter:bump"))
        .cloned()
        .unwrap_or_else(|| {
            panic!(
                "Expected Counter:bump to lower to MIR in vm.mir_functions, found keys: {:?}",
                vm.mir_functions.keys().collect::<Vec<_>>()
            )
        });
    let script_mir = vm
        .mir_functions
        .get("__script")
        .expect("__script must lower to MIR");
    let pre_inline_calls_bump = script_mir.blocks.iter().any(|b| {
        matches!(&b.terminator, Some(Terminator::Call { callee, .. }) if callee == &bump_name)
    });
    assert!(
        pre_inline_calls_bump,
        "Pre-inline __script MIR must lower c:bump(i) to Terminator::Call {{ callee: \"{bump_name}\" }}"
    );

    vm.compile_cranelift().expect("compile_cranelift");
    assert!(
        vm.cranelift_compiled.contains_key(&bump_name),
        "Expected `{bump_name}` to compile to Cranelift"
    );
    assert!(
        vm.cranelift_compiled.contains_key("__script"),
        "Expected `__script` to compile to Cranelift"
    );

    let post_inline_script = vm
        .cranelift_compiled_mir("__script")
        .expect("post-inline compiled MIR for __script");
    MirVerifier::verify_fn(&post_inline_script).expect("post-inline __script MIR must verify");
    let post_inline_calls_bump = post_inline_script.blocks.iter().any(|b| {
        matches!(&b.terminator, Some(Terminator::Call { callee, .. }) if callee == &bump_name)
    });
    assert!(
        !post_inline_calls_bump,
        "Expected PERF-T1-4 to inline `{bump_name}` into `__script`, leaving zero Terminator::Call to `{bump_name}`"
    );

    // Execute a fast embedded variant to verify native entry & exact stdout
    let fast_methods_src = methods_src.replace("10000000", "1000");
    let mut vm_fast = compile_embedded_vm(&fast_methods_src, true);
    vm_fast.call("__script", vec![]).expect("VM::call __script");
    let vm_stdout = String::from_utf8(vm_fast.console.stdout_bytes()).expect("utf8 stdout");
    assert_eq!(vm_stdout, "499500\n");
    assert!(
        vm_fast.cranelift_entry_count >= 1 && vm_fast.cranelift_compilation_count >= 1,
        "VM API methods script must compile & enter Cranelift (entries={}, compilations={})",
        vm_fast.cranelift_entry_count,
        vm_fast.cranelift_compilation_count
    );

    // 3. Direct Instruction::CallMethod dispatch from an interpreted caller (non-inlined!)
    //    to a Cranelift-compiled method mutating self.field, verifying writeback to caller's Value::Struct.
    let direct_callmethod_src = r#"
struct Counter
    value: int
end

impl Counter
    function bump(self, d: int): int
        self.value = self.value + d
        return self.value
    end
end

function interpreted_driver(): int
    -- Lambda prevents `interpreted_driver` from lowering to MIR, forcing AST->Bytecode
    -- Instruction::CallMethod dispatch into Cranelift-compiled `Counter:bump`.
    local make_inc = function(k: int): int return k + 1 end
    local c: Counter = Counter { value = 10 }
    local r1: int = c:bump(make_inc(4))
    local r2: int = c:bump(make_inc(9))
    local r3: int = c:bump(make_inc(14))
    return r1 + r2 + r3 + c.value
end
"#;
    let mut vm_oracle = compile_embedded_vm(direct_callmethod_src, false);
    let mut vm_clif = compile_embedded_vm(direct_callmethod_src, true);

    let driver_name = find_fn_name(&vm_clif, "interpreted_driver");
    let bump_fn = vm_clif
        .mir_functions
        .keys()
        .find(|k| k.ends_with("Counter:bump"))
        .cloned()
        .expect("Counter:bump in mir_functions");

    // Ensure caller itself is NOT compiled to Cranelift so Instruction::CallMethod executes!
    assert!(
        !vm_clif.mir_functions.contains_key(&driver_name),
        "interpreted_driver must stay on AST->Bytecode to exercise Instruction::CallMethod"
    );
    vm_clif.compile_cranelift().expect("compile_cranelift");
    assert!(
        vm_clif.cranelift_compiled.contains_key(&bump_fn),
        "Counter:bump must be compiled to Cranelift"
    );

    let expected = vm_oracle
        .call(&driver_name, vec![])
        .expect("oracle interpreted_driver");
    // c starts at 10; +5 => 15, +10 => 25, +15 => 40; return 15 + 25 + 40 + 40 = 120
    assert_eq!(expected, Value::Int(120));

    let entries_before = vm_clif.cranelift_entry_count;
    let actual = vm_clif
        .call(&driver_name, vec![])
        .expect("clif CallMethod interpreted_driver");
    assert_eq!(actual, expected, "CallMethod writeback result mismatch");
    assert_eq!(
        vm_clif.cranelift_entry_count,
        entries_before + 3,
        "All 3 Instruction::CallMethod calls (1st cache-miss + 2 cache-hits) must enter Cranelift natively"
    );
}

// =========================================================================
// Adversarial Negative Test 1: Non-Struct Receivers (Option, Effect/Closure, Dynamic/Unknown)
// =========================================================================

#[test]
fn test_perf_t2_3_adv_non_struct_method_receivers_fallback_cleanly() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    // A. Option receiver `opt:unwrap()` (Some succeeds, None errors cleanly — no mis-dispatch!)
    let option_src = r#"
function unwrap_opt(opt: Option<int>): int
    return opt:unwrap()
end

function test_some(): int
    local o: Option<int> = Option.Some(42)
    return unwrap_opt(o)
end

function test_none(): int
    local o: Option<int> = Option.None
    return unwrap_opt(o)
end
"#;
    let mut vm_opt_bc = compile_embedded_vm(option_src, false);
    let mut vm_opt_clif = compile_embedded_vm(option_src, true);
    let test_some_fn = find_fn_name(&vm_opt_clif, "test_some");
    let test_none_fn = find_fn_name(&vm_opt_clif, "test_none");

    assert_eq!(
        vm_opt_clif.call(&test_some_fn, vec![]).expect("clif test_some"),
        vm_opt_bc.call(&test_some_fn, vec![]).expect("bc test_some")
    );
    assert_eq!(
        vm_opt_clif.call(&test_some_fn, vec![]).expect("clif test_some"),
        Value::Int(42)
    );
    assert!(
        vm_opt_bc.call(&test_none_fn, vec![]).is_err(),
        "bc Option.None:unwrap() must error"
    );
    assert!(
        vm_opt_clif.call(&test_none_fn, vec![]).is_err(),
        "clif Option.None:unwrap() must fall back cleanly and error, not mis-dispatch"
    );

    // B. Effect / Closure receiver (`e:provide(...)`, `e:run_sync()`) falls back cleanly
    let closure_provide_src = r#"
struct Env
    seed: int
end

function run_with_env(e: Effect<int, never>, env: Env): int
    return e:provide(env):run_sync()
end

function run_closure_provide(): int
    local env: Env = Env { seed = 1 }
    return (effect do
        return 77
    end):provide(env):run_sync()
end
"#;
    let mut vm_eff_bc = compile_embedded_vm(closure_provide_src, false);
    let mut vm_eff_clif = compile_embedded_vm(closure_provide_src, true);
    let run_with_env_fn = find_fn_name(&vm_eff_clif, "run_with_env");
    assert!(
        !vm_eff_clif.mir_functions.contains_key(&run_with_env_fn),
        "Effect receiver `e:provide(env)` must NOT mis-dispatch as struct method; must fall back cleanly"
    );
    let eff_fn = find_fn_name(&vm_eff_clif, "run_closure_provide");
    assert_eq!(
        vm_eff_clif.call(&eff_fn, vec![]).expect("clif provide"),
        vm_eff_bc.call(&eff_fn, vec![]).expect("bc provide")
    );
    assert_eq!(
        vm_eff_clif.call(&eff_fn, vec![]).expect("clif provide"),
        Value::Int(77)
    );

    // C. Unknown/dynamic receiver calling user struct method via CallMethod still works & matches oracle
    let dynamic_src = r#"
struct Counter
    value: int
end

impl Counter
    function bump(self, d: int): int
        self.value = self.value + d
        return self.value
    end
end

function bump_dynamic(obj: unknown, d: int): unknown
    return obj:bump(d)
end

function run_dynamic(): int
    local c: Counter = Counter { value = 5 }
    local r = bump_dynamic(c, 37)
    return c.value
end
"#;
    let mut vm_dyn_bc = compile_embedded_vm(dynamic_src, false);
    let mut vm_dyn_clif = compile_embedded_vm(dynamic_src, true);
    let dyn_fn = find_fn_name(&vm_dyn_clif, "run_dynamic");
    let bc_res = vm_dyn_bc.call(&dyn_fn, vec![]).expect("bc run_dynamic");
    let clif_res = vm_dyn_clif.call(&dyn_fn, vec![]).expect("clif run_dynamic");
    assert_eq!(clif_res, bc_res);
    assert_eq!(clif_res, Value::Int(42));
}

// =========================================================================
// Adversarial Negative Test 2: Self Mutation Writeback (Inline + Non-Inline + Aliases + Deopt)
// =========================================================================

#[test]
fn test_perf_t2_3_adv_self_mutation_writeback_inline_non_inline_and_deopt() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
struct Vec2
    x: int
    y: int
end

impl Vec2
    function shift(self, dx: int, dy: int): int
        self.x = self.x + dx
        self.y = self.y + dy
        return self.x + self.y
    end

    -- Loop-bearing method is NOT inlined by PERF-T1-4 (Guard 2), exercising
    -- direct native-to-native method calls AND direct CallMethod writeback.
    function shift_loop(self, n: int): int
        local i: int = 1
        while i <= n do
            self.x = self.x + i
            self.y = self.y + i * 2
            i = i + 1
        end
        return self.x * 1000 + self.y
    end
end

function run_inlined_and_aliased(n: int): int
    local v: Vec2 = Vec2 { x = 1, y = 2 }
    local alias: Vec2 = v
    local i: int = 0
    local last: int = 0
    while i < n do
        last = v:shift(1, 2)
        i = i + 1
    end
    return last + alias.x * 10 + alias.y
end

function run_non_inlined_native_call(n: int): int
    local v: Vec2 = Vec2 { x = 3, y = 4 }
    local alias: Vec2 = v
    local res: int = v:shift_loop(n)
    return res + alias.x + alias.y
end

function run_interpreted_callmethod_alias(n: int): int
    local id = function(k: int): int return k end
    local v: Vec2 = Vec2 { x = 5, y = 6 }
    local alias: Vec2 = v
    local res: int = v:shift_loop(id(n))
    local s: int = alias:shift(id(2), id(3))
    return res + s + v.x + v.y
end
"#;

    let mut vm_bc = compile_embedded_vm(src, false);
    let mut vm_clif = compile_embedded_vm(src, true);
    vm_clif.compile_cranelift().expect("compile_cranelift");

    let inlined_fn = find_fn_name(&vm_clif, "run_inlined_and_aliased");
    let non_inlined_fn = find_fn_name(&vm_clif, "run_non_inlined_native_call");
    let interp_cm_fn = find_fn_name(&vm_clif, "run_interpreted_callmethod_alias");
    let shift_loop_fn = vm_clif
        .mir_functions
        .keys()
        .find(|k| k.ends_with("Vec2:shift_loop"))
        .cloned()
        .expect("Vec2:shift_loop in mir_functions");

    // Verify Vec2:shift_loop was NOT inlined into run_non_inlined_native_call (loop guard)
    let non_inlined_mir = vm_clif
        .cranelift_compiled_mir(&non_inlined_fn)
        .expect("run_non_inlined_native_call compiled MIR");
    assert!(
        non_inlined_mir.blocks.iter().any(|b| {
            matches!(&b.terminator, Some(Terminator::Call { callee, .. }) if callee == &shift_loop_fn)
        }),
        "loop-bearing method Vec2:shift_loop must stay un-inlined"
    );

    // 1. Inlined method + alias writeback parity
    let exp1 = vm_bc.call(&inlined_fn, vec![Value::Int(15)]).expect("bc 1");
    let act1 = vm_clif
        .call(&inlined_fn, vec![Value::Int(15)])
        .expect("clif 1");
    assert_eq!(act1, exp1, "inlined method + alias mismatch");

    // 2. Non-inlined native-to-native method call + alias writeback parity
    let exp2 = vm_bc
        .call(&non_inlined_fn, vec![Value::Int(10)])
        .expect("bc 2");
    let act2 = vm_clif
        .call(&non_inlined_fn, vec![Value::Int(10)])
        .expect("clif 2");
    assert_eq!(act2, exp2, "non-inlined native method + alias mismatch");

    // 3. Interpreted Instruction::CallMethod into Cranelift method + alias writeback parity
    let exp3 = vm_bc
        .call(&interp_cm_fn, vec![Value::Int(10)])
        .expect("bc 3");
    let entries_before = vm_clif.cranelift_entry_count;
    let act3 = vm_clif
        .call(&interp_cm_fn, vec![Value::Int(10)])
        .expect("clif 3");
    assert_eq!(act3, exp3, "interpreted CallMethod + alias mismatch");
    assert!(
        vm_clif.cranelift_entry_count >= entries_before + 2,
        "both v:shift_loop and alias:shift via CallMethod must enter Cranelift"
    );

    // 4. Under `test-deopt`, poll-1 deopt mid-loop inside `run_inlined_and_aliased` must
    //    reconstruct both aliased Vec2 locals and finish with exact oracle result.
    #[cfg(feature = "test-deopt")]
    {
        reset_all_state();
        safepoint_request_deopt_on_poll(Some(inlined_fn.clone()), 5);
        let entries_pre = vm_clif.cranelift_entry_count;
        let deopt_res = vm_clif
            .call(&inlined_fn, vec![Value::Int(15)])
            .expect("poll-5 deopt on inlined method loop must resume cleanly");
        safepoint_reset();

        assert_eq!(vm_clif.cranelift_entry_count, entries_pre + 1);
        assert_eq!(
            vm_clif.last_deopt_source,
            Some(DeoptPayloadSource::NativeExecution)
        );
        assert_eq!(deopt_res, exp1, "post-deopt inlined method result mismatch");
        assert!(take_deopt_payload().is_none());
        assert!(get_current_sticky_error().is_none());
    }
}

//! Integration & regression tests for Wave 15 Task `NG-W15-4`:
//! - CLI `Struct.decode(raw)` registration (`register_struct_value_decoders` / `register_module_struct_decoders`)
//! - `:unwrap()`, `:unwrap_or(d)`, `:is_some()`, `:is_none()`, `:is_ok()`, `:is_err()` & fused `map.get_unwrap`
//! - User `Enum` constructors (`Op.Reset`, `Op.Lit(v)`, `Op.Add(a, b)`) & `if op is Variant(...)` MIR lowering + SROA
//! - `q.offer(...)` struct/dynamic field-call resolution (`BENCH-NG-12`)
//! - First-class function parameter call `f(x)` in `MirToBytecodeCompiler` (`BENCH-NG-04`)
//! - Dynamic `unknown` field/method/type-tag lowering (`BENCH-NG-05`, `BENCH-NG-09`, `BENCH-NG-12`)
//! - Full 15/15 Next-Gen Suite v2 MIR lowering + Rust AOT build & exact oracle checksum parity

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::tempdir;

use lust::bytecode::{Instruction, Value};
use lust::codegen_zig::aot::compiler::prepare_and_lower_items_for_rust_aot;
use lust::codegen_zig::aot::{prepare_items_for_mir, AotCompiler, RustCompileOptions};
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::ir::{MirModule, MirType, Terminator};
use lust::mir::lowering::AstLowering;
use lust::mir::opt::MirOptimizer;
use lust::mir::verifier::MirVerifier;
use lust::mir::vm_backend::{
    register_module_struct_decoders, register_struct_value_decoders, MirToBytecodeCompiler,
};
use lust::modules::ModuleLoader;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::VM;

fn suite_v2_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("benchmarks/suite_v2")
        .join(name)
}

fn parse_and_lower(source: &str, name: &str) -> MirModule {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize().expect("lex");
    let items = Parser::new(tokens).parse().expect("parse");
    let prepared = prepare_items_for_mir(&items);
    let module = AstLowering::lower_module_items(name, &prepared)
        .unwrap_or_else(|e| panic!("MIR lowering failed for {name}: {e}"));
    MirVerifier::verify_module(&module)
        .unwrap_or_else(|e| panic!("MirVerifier failed for {name}: {e:?}"));
    module
}

fn load_and_lower_file(path: &std::path::Path, name: &str) -> (MirModule, Vec<lust::ast::Item>) {
    let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let mut loader = ModuleLoader::new(parent);
    let program = loader
        .load_program_from_entry(path.to_str().expect("utf8 path"))
        .unwrap_or_else(|e| panic!("ModuleLoader failed for {name}: {e}"));
    let mut all_items = Vec::new();
    for m in &program.modules {
        all_items.extend(m.items.clone());
    }
    let filtered = lust::codegen_zig::tree_shake_items(&all_items, false);
    let module = prepare_and_lower_items_for_rust_aot(&filtered)
        .unwrap_or_else(|e| panic!("MIR lowering failed for {name}: {e}"));
    MirVerifier::verify_module(&module)
        .unwrap_or_else(|e| panic!("MirVerifier failed for {name}: {e:?}"));
    (module, all_items)
}

fn compile_and_run_rust_aot_file(path: &std::path::Path, name: &str) -> String {
    let (module, all_items) = load_and_lower_file(path, name);
    let dir = tempdir().expect("tempdir");
    let bin_path = dir.path().join(name);
    let opts = RustCompileOptions::default();
    let built = AotCompiler::compile_module_rust_with_ast(&module, &all_items, &bin_path, &opts)
        .unwrap_or_else(|e| panic!("Rust AOT compilation failed for {name}: {e}"));

    let out = Command::new(&built)
        .output()
        .unwrap_or_else(|e| panic!("failed to execute {name}: {e}"));
    assert!(
        out.status.success(),
        "{name} exited with {:?}, stderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn compile_and_run_rust_aot_source_via_loader(source: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{name}.lust"));
    fs::write(&src_path, source).expect("write temp lust file");
    compile_and_run_rust_aot_file(&src_path, name)
}

#[test]
fn test_ng_w15_4_cli_struct_decode_and_mir_unwrap_chaining() {
    // 1. Struct.decode(raw) through MIR lowering + MIR-VM execution + register_struct_value_decoders
    let decode_src = r#"
struct TelemetryEvent
    ts: int
    device_id: int
    metric: int
    active: bool
end

function eval_decode(raw: string): int
    local ev: TelemetryEvent = TelemetryEvent.decode(raw)
    if ev.active then
        return ev.metric * ev.device_id + (ev.ts % 1000)
    else
        return ev.metric + ev.device_id
    end
end
"#;
    let mut interner = Interner::new();
    let tokens = Lexer::new(decode_src, &mut interner).tokenize().expect("lex");
    let items = Parser::new(tokens).parse().expect("parse");
    let config = LustConfig::default();
    let mut tc = TypeChecker::with_config(&config);
    tc.check_module(&items).expect("typecheck decode_src");
    let struct_defs = tc.struct_definitions();

    let module = AstLowering::lower_module_items("decode_mod", &items).expect("lower decode_mod");
    MirVerifier::verify_module(&module).expect("verify decode_mod");
    let bc_fns = MirToBytecodeCompiler::compile_module(&module).expect("compile decode_mod");

    let mut vm = VM::new();
    vm.load_functions(bc_fns);
    vm.register_structs(&struct_defs);
    register_struct_value_decoders(&mut vm, &struct_defs);

    let raw_a = r#"{"ts":1700000100,"device_id":42,"metric":950,"active":true}"#;
    let raw_b = r#"{"ts":1700000200,"device_id":19,"metric":310,"active":false}"#;
    assert_eq!(
        vm.call("eval_decode", vec![Value::string(raw_a.to_string())])
            .expect("eval_decode a"),
        Value::Int(40000)
    );
    assert_eq!(
        vm.call("eval_decode", vec![Value::string(raw_b.to_string())])
            .expect("eval_decode b"),
        Value::Int(329)
    );

    // Also verify direct CLI binary execution with `--mir` if `CARGO_BIN_EXE_lust` is available.
    if let Some(lust_bin) = option_env!("CARGO_BIN_EXE_lust") {
        let dir = tempdir().expect("tempdir");
        let script = dir.path().join("cli_decode.lust");
        fs::write(
            &script,
            r#"
struct TelemetryEvent
    ts: int
    device_id: int
    metric: int
    active: bool
end

function run_decode(raw: string): int
    local ev: TelemetryEvent = TelemetryEvent.decode(raw)
    if ev.active then
        return ev.metric * ev.device_id + (ev.ts % 1000)
    else
        return 0
    end
end

local raw: string = "{\"ts\":1700000100,\"device_id\":42,\"metric\":950,\"active\":true}"
println(run_decode(raw))
"#,
        )
        .expect("write cli_decode.lust");

        let out = Command::new(lust_bin)
            .arg("--mir")
            .arg(&script)
            .output()
            .expect("run lust --mir");
        assert!(
            out.status.success(),
            "lust --mir cli_decode.lust failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "40000");
    }

    // 2. Two-line `local found: Option<int> = map.get(m, k); local cur: int = found:unwrap()`
    //    must fuse in-place to `"map.get_unwrap"`, plus `:unwrap_or()`, `:is_some()`, `:is_none()`.
    let unwrap_src = r#"
function test_map_unwrap_and_option(n: int): int
    local m: Map<int, int> = {}
    local i: int = 0
    while i < n do
        map.set(m, i, i * 10 + 3)
        i = i + 1
    end

    local acc: int = 0
    i = 0
    while i < n do
        local found: Option<int> = map.get(m, i)
        local cur: int = found:unwrap()
        acc = acc + cur
        i = i + 1
    end

    local miss: int = map.get(m, 99999):unwrap_or(77)
    local opt_hit: Option<int> = map.get(m, 0)
    local opt_miss: Option<int> = map.get(m, 99999)
    local bonus: int = 0
    if opt_hit:is_some() then
        bonus = bonus + 100
    end
    if opt_miss:is_none() then
        bonus = bonus + 200
    end
    return acc + miss + bonus
end
"#;
    let unwrap_mod = parse_and_lower(unwrap_src, "unwrap_mod");
    let f = unwrap_mod
        .functions
        .iter()
        .find(|f| f.signature.name == "test_map_unwrap_and_option")
        .expect("test_map_unwrap_and_option");

    let has_fused_unwrap = f.blocks.iter().any(|bb| {
        matches!(
            &bb.terminator,
            Some(Terminator::Call { callee, .. }) if callee == "map.get_unwrap"
        )
    });
    let has_fused_unwrap_or = f.blocks.iter().any(|bb| {
        matches!(
            &bb.terminator,
            Some(Terminator::Call { callee, .. }) if callee == "map.get_unwrap_or"
        )
    });
    assert!(
        has_fused_unwrap,
        "two-line `local found = map.get(m, k); found:unwrap()` must fuse into `map.get_unwrap`"
    );
    assert!(
        has_fused_unwrap_or,
        "`map.get(m, k):unwrap_or(d)` must lower to `map.get_unwrap_or`"
    );

    let bc_fns = MirToBytecodeCompiler::compile_module(&unwrap_mod).expect("compile unwrap_mod");
    let mut vm2 = VM::new();
    vm2.load_functions(bc_fns);
    register_module_struct_decoders(&mut vm2, &unwrap_mod);
    // n = 5 -> sum(3, 13, 23, 33, 43) = 115; + 77 + 100 + 200 = 492
    let res = vm2
        .call("test_map_unwrap_and_option", vec![Value::Int(5)])
        .expect("execute test_map_unwrap_and_option");
    assert_eq!(res, Value::Int(492));
}

#[test]
fn test_ng_w15_4_user_enum_scalar_sroa_and_mir_vm() {
    let src = r#"
enum Op
    Lit(int)
    Add(int, int)
    Mul(int, int)
    Xor(int, int)
    Reset
end

function make_op(step: int): Op
    local kind: int = step % 5
    if kind == 0 then
        return Op.Lit(step % 97)
    elseif kind == 1 then
        return Op.Add(step % 131, (step * 7) % 131)
    elseif kind == 2 then
        return Op.Mul((step % 19) + 1, (step % 23) + 1)
    elseif kind == 3 then
        return Op.Xor(step, step * 3)
    else
        return Op.Reset
    end
end

function apply_op(acc: int, op: Op): int
    if op is Lit(v) then
        return (acc + v) % 1000000007
    elseif op is Add(a, b) then
        return (acc + a + b) % 1000000007
    elseif op is Mul(a, b) then
        return (acc + a * b) % 1000000007
    elseif op is Xor(a, b) then
        return (acc + (a ~ b)) % 1000000007
    elseif op is Reset then
        return acc / 2
    else
        return acc
    end
end

function run_ops(limit: int): int
    local state: int = 1
    local i: int = 0
    while i < limit do
        local op: Op = make_op(i)
        state = apply_op(state, op)
        i = i + 1
    end
    return state
end
"#;

    let mut module = parse_and_lower(src, "enum_op_mod");
    assert!(
        module.find_struct_layout("__Enum_Op").is_some(),
        "scalar user enum `Op` must register synthetic `__Enum_Op` layout"
    );
    assert!(
        module.enum_defs.iter().any(|e| e.name == "Op"),
        "module.enum_defs must retain `Op` definition"
    );

    // Execute via MIR-VM before optimization to verify MIR semantics directly.
    let bc_fns = MirToBytecodeCompiler::compile_module(&module).expect("compile enum_op_mod");
    let fn_names: Vec<String> = bc_fns.iter().map(|f| f.name.clone()).collect();
    let mut vm = VM::new();
    vm.load_functions(bc_fns);
    for (idx, name) in fn_names.iter().enumerate() {
        vm.set_global(name.clone(), Value::Function(idx));
    }
    let vm_res = vm
        .call("run_ops", vec![Value::Int(200)])
        .expect("run_ops in MIR-VM");

    // Optimize module: inlining + NG-W15-2 multi-def SROA should decompose `__Enum_Op` in `run_ops`.
    MirOptimizer::optimize_module(&mut module);
    MirVerifier::verify_module(&module).expect("verify optimized enum_op_mod");
    let opt_bc_fns =
        MirToBytecodeCompiler::compile_module(&module).expect("compile optimized enum_op_mod");
    let opt_fn_names: Vec<String> = opt_bc_fns.iter().map(|f| f.name.clone()).collect();
    let mut vm_opt = VM::new();
    vm_opt.load_functions(opt_bc_fns);
    for (idx, name) in opt_fn_names.iter().enumerate() {
        vm_opt.set_global(name.clone(), Value::Function(idx));
    }
    let opt_res = vm_opt
        .call("run_ops", vec![Value::Int(200)])
        .expect("run_ops optimized in MIR-VM");
    assert_eq!(vm_res, opt_res, "optimized enum MIR-VM must match unoptimized");

    // Also verify non-scalar (heterogeneous) user enum lowers via MirType::Enum + enum.tag/payload.
    let non_scalar_src = r#"
enum Msg
    Empty
    Text(string)
    Pair(int, string)
end

function inspect_msg(m: Msg): int
    if m is Text(s) then
        return string.len(s)
    elseif m is Pair(n, s) then
        return n + string.len(s)
    elseif m is Empty then
        return 0
    else
        return -1
    end
end

function build_and_inspect(): int
    local a: Msg = Msg.Text("hello")
    local b: Msg = Msg.Pair(10, "ab")
    local c: Msg = Msg.Empty
    return inspect_msg(a) + inspect_msg(b) + inspect_msg(c)
end
"#;
    let non_scalar_mod = parse_and_lower(non_scalar_src, "non_scalar_enum_mod");
    assert!(
        non_scalar_mod.find_struct_layout("__Enum_Msg").is_none(),
        "non-scalar enum `Msg` must NOT use scalar struct layout"
    );
    let inspect_fn = non_scalar_mod
        .functions
        .iter()
        .find(|f| f.signature.name == "inspect_msg")
        .expect("inspect_msg");
    assert!(
        inspect_fn.blocks.iter().any(|b| matches!(
            &b.terminator,
            Some(Terminator::Call { callee, .. }) if callee == "enum.tag"
        )),
        "non-scalar enum match must emit enum.tag"
    );
}

#[test]
fn test_ng_w15_4_first_class_param_call_and_dynamic_unknown_ops_in_mir_vm() {
    // 1. First-class function parameter call `f(f(x))` in MirToBytecodeCompiler (`BENCH-NG-04`)
    let fn_param_src = r#"
function inc3(v: int): int
    return v * 2 + 1
end

function apply_twice(f: function(int): int, x: int): int
    return f(f(x))
end

function run_apply(x: int): int
    return apply_twice(inc3, x)
end
"#;
    let fn_mod = parse_and_lower(fn_param_src, "fn_param_mod");
    let apply_fn = fn_mod
        .functions
        .iter()
        .find(|f| f.signature.name == "apply_twice")
        .expect("apply_twice");
    assert!(
        matches!(
            apply_fn.signature.param_types[0],
            MirType::Function { .. }
        ),
        "param `f: function(int): int` must lower to MirType::Function, got {:?}",
        apply_fn.signature.param_types[0]
    );

    let bc_apply = MirToBytecodeCompiler::compile(apply_fn).expect("compile apply_twice");
    assert!(
        !bc_apply.chunk.instructions.iter().any(|instr| matches!(
            instr,
            Instruction::LoadGlobal(_, idx)
                if bc_apply.chunk.constants.get(*idx as usize)
                    == Some(&Value::string("f".to_string()))
        )),
        "MirToBytecodeCompiler must NOT emit LoadGlobal(\"f\") for parameter `f`:\n{}",
        bc_apply.disassemble()
    );

    let bc_fns = MirToBytecodeCompiler::compile_module(&fn_mod).expect("compile fn_param_mod");
    let fn_names: Vec<String> = bc_fns.iter().map(|f| f.name.clone()).collect();
    let mut vm = VM::new();
    vm.load_functions(bc_fns);
    for (idx, name) in fn_names.iter().enumerate() {
        vm.set_global(name.clone(), Value::Function(idx));
    }
    // inc3(5) = 11; inc3(11) = 23
    assert_eq!(
        vm.call("apply_twice", vec![vm.get_global("inc3").unwrap(), Value::Int(5)])
            .expect("apply_twice(inc3, 5)"),
        Value::Int(23)
    );

    // 2. Direct field-call `q.offer(x)` & `q.take()` on Queue (`BENCH-NG-12`) + `unknown` ops (`BENCH-NG-05`)
    let q_src = r#"
use @std/sync.{Queue}

function test_direct_q_offer(): int
    local q = Queue(8, "suspend")
    q.offer(10)
    q.offer(32)
    local a: int = q.take()
    local b: int = q.take()
    return a + b
end

println(test_direct_q_offer())
"#;
    let q_dir = tempdir().expect("tempdir");
    let q_file = q_dir.path().join("q_field_call.lust");
    fs::write(&q_file, q_src).expect("write q_field_call.lust");
    let (q_mod, _) = load_and_lower_file(&q_file, "q_field_call_mod");
    let q_fn = q_mod
        .functions
        .iter()
        .find(|f| {
            f.signature.name == "test_direct_q_offer"
                || f.signature.name.ends_with(".test_direct_q_offer")
        })
        .expect("test_direct_q_offer");
    assert!(
        q_fn.blocks.iter().any(|b| matches!(
            &b.terminator,
            Some(Terminator::Call { callee, .. }) if callee == "__mir_unknown_get_field"
        )),
        "direct `q.offer(...)` must resolve field via __mir_unknown_get_field"
    );

    let q_aot_src = r#"
use @std/sync.{Queue}

local q = Queue(8, "suspend")
q.offer(10)
q.offer(32)
local a: int = q.take()
local b: int = q.take()
println(a + b)
"#;
    assert_eq!(
        compile_and_run_rust_aot_source_via_loader(q_aot_src, "direct_q_offer"),
        "42"
    );
}

#[test]
fn test_ng_w15_4_all_15_suite_v2_benchmarks_lower_to_mir_and_compile_in_rust_aot() {
    let cases: [(&str, &str, &str); 15] = [
        ("BENCH-NG-01", "map_int_churn.lust", "608439931"),
        ("BENCH-NG-02", "map_str_intern_hist.lust", "379372375"),
        ("BENCH-NG-03", "array_mut_pipeline.lust", "500808394"),
        ("BENCH-NG-04", "closure_upvalue_combinators.lust", "947911"),
        ("BENCH-NG-05", "megamorphic_method_ic.lust", "24980750"),
        ("BENCH-NG-06", "adt_enum_eval.lust", "2262427"),
        ("BENCH-NG-07", "sroa_escape_structs.lust", "990177872"),
        ("BENCH-NG-08", "cycle_collector_pressure.lust", "250029089"),
        ("BENCH-NG-09", "string_parse_tokenize.lust", "948525981"),
        ("BENCH-NG-10", "regex_match_extract.lust", "240000"),
        ("BENCH-NG-11", "defer_effect_thunk.lust", "466147248"),
        ("BENCH-NG-12", "fiber_channel_pingpong.lust", "799919916"),
        ("BENCH-NG-13", "schema_json_decode_pipeline.lust", "8224993"),
        ("BENCH-NG-14", "loop_3level_call_barrier.lust", "321195209"),
        ("BENCH-NG-15", "ffi_struct_array_pingpong.lust", "151024993"),
    ];

    for (bench_id, filename, expected_checksum) in cases {
        let path = suite_v2_path(filename);
        let got = compile_and_run_rust_aot_file(&path, bench_id);
        assert_eq!(
            got, expected_checksum,
            "{bench_id} ({filename}) Rust AOT checksum mismatch: got={got}, expected={expected_checksum}"
        );
        eprintln!("[NG-W15-4] {bench_id} ({filename}) Rust AOT PASS: stdout={got}");
    }
}

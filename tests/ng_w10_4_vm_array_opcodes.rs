#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! NG-W10-4: Direct VM array opcodes (`ArrayLen`, `ArrayPush`, `ArrayGetUnwrap`),
//! unboxed `LustArray` fast paths across all six array instructions in `'frame_fast: loop`
//! and `'dispatch: loop` (`F-05`), and `self.arg_scratch` reuse for `Value::NativeFunction`.

use std::path::PathBuf;

use lust::bytecode::compiler::Compiler;
use lust::bytecode::value::{NativeCallResult, UnboxedArrayData, native_fn};
use lust::bytecode::{Function, Instruction, OpCode, Value};
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vfs::VirtualConsole;
use lust::vm::VM;

fn compile_source(
    source: &str,
    use_mir: bool,
) -> (
    Vec<Function>,
    hashbrown::HashMap<String, lust::ast::StructDef>,
) {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("lex should succeed");
    let items = Parser::new(tokens).parse().expect("parse should succeed");
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    config.set_use_mir(use_mir);
    let mut checker = TypeChecker::with_config(&config);
    checker
        .check_module(&items)
        .expect("typecheck should succeed");
    let struct_defs = checker.struct_definitions();
    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_option_coercions(checker.take_option_coercions());
    compiler.set_checked_array_indices(checker.take_checked_array_indices());
    compiler.set_lambda_return_types(checker.take_lambda_return_types());
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    let funcs = compiler
        .compile_module(&items)
        .expect("compile should succeed");
    (funcs, struct_defs)
}

fn count_load_global_named(func: &Function, target: &str) -> usize {
    func.chunk
        .instructions
        .iter()
        .filter(|instr| match instr {
            Instruction::LoadGlobal(_, const_idx) => func
                .chunk
                .constants
                .get(*const_idx as usize)
                .and_then(Value::as_string)
                == Some(target),
            _ => false,
        })
        .count()
}

fn load_sieve_source() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks")
        .join("suite")
        .join("sieve.lust");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e))
}

#[test]
fn test_ng_w10_4_sieve_disassembly_has_direct_opcodes_and_zero_load_global_array() {
    let sieve_src = load_sieve_source();

    // 1. AST -> Bytecode path
    let (ast_funcs, _) = compile_source(&sieve_src, false);
    let script = ast_funcs
        .iter()
        .find(|f| f.name == "__script")
        .expect("__script function must exist");
    let disasm = script.disassemble();

    assert!(
        disasm.contains("ArrayPush"),
        "sieve.lust disassembly must contain ArrayPush:\n{disasm}"
    );
    assert!(
        disasm.contains("ArrayIndexOk"),
        "sieve.lust disassembly must contain ArrayIndexOk:\n{disasm}"
    );
    assert!(
        disasm.contains("SetIndex"),
        "sieve.lust disassembly must contain SetIndex:\n{disasm}"
    );
    assert_eq!(
        count_load_global_named(script, "array"),
        0,
        "sieve.lust must emit 0 LoadGlobal(\"array\") instructions:\n{disasm}"
    );

    // 2. MIR -> Bytecode path
    let (mir_funcs, _) = compile_source(&sieve_src, true);
    let mir_script = mir_funcs
        .iter()
        .find(|f| f.name == "__script")
        .expect("__script function must exist (MIR)");
    let mir_disasm = mir_script.disassemble();
    assert!(
        mir_disasm.contains("ArrayPush"),
        "MIR sieve.lust disassembly must contain ArrayPush:\n{mir_disasm}"
    );
    assert!(
        mir_disasm.contains("SetIndex"),
        "MIR sieve.lust disassembly must contain SetIndex:\n{mir_disasm}"
    );
    assert_eq!(
        count_load_global_named(mir_script, "array"),
        0,
        "MIR sieve.lust must emit 0 LoadGlobal(\"array\") instructions:\n{mir_disasm}"
    );
}

#[test]
fn test_ng_w10_4_f05_sieve_10k_stays_unboxed_bool_with_zero_materialization() {
    let sieve_fn_src = r#"
function run_sieve_flags(n: int): Array<bool>
    local flags: Array<bool> = []
    local i: int = 0
    while i <= n do
        array.push(flags, true)
        i = i + 1
    end
    local count: int = 0
    i = 2
    while i <= n do
        if flags[i] is Ok(f) then
            if f then
                count = count + 1
                local j: int = i * i
                while j <= n do
                    flags[j] = false
                    j = j + i
                end
            end
        end
        i = i + 1
    end
    return flags
end

function run_sieve_count(n: int): int
    local flags: Array<bool> = []
    local i: int = 0
    while i <= n do
        array.push(flags, true)
        i = i + 1
    end
    local count: int = 0
    i = 2
    while i <= n do
        if flags[i] is Ok(f) then
            if f then
                count = count + 1
                local j: int = i * i
                while j <= n do
                    flags[j] = false
                    j = j + i
                end
            end
        end
        i = i + 1
    end
    return count
end
"#;

    for use_mir in [false, true] {
        let (funcs, struct_defs) = compile_source(sieve_fn_src, use_mir);
        let run_fn = funcs
            .iter()
            .find(|f| f.name == "run_sieve_flags")
            .expect("run_sieve_flags must exist");
        assert_eq!(
            count_load_global_named(run_fn, "array"),
            0,
            "run_sieve_flags (use_mir={use_mir}) must have 0 LoadGlobal(\"array\")"
        );

        let mut config = LustConfig::default();
        config.set_jit_enabled(false);
        let mut vm = VM::with_config(&config);
        vm.register_structs(&struct_defs);
        vm.load_functions(funcs);

        let count = vm
            .call("run_sieve_count", vec![Value::Int(10_000)])
            .expect("run_sieve_count(10_000) should succeed");
        assert_eq!(
            count,
            Value::Int(1229),
            "prime count up to 10_000 must be 1229 (use_mir={use_mir})"
        );

        let flags_val = vm
            .call("run_sieve_flags", vec![Value::Int(10_000)])
            .expect("run_sieve_flags(10_000) should succeed");

        let Value::Array(flags) = flags_val else {
            panic!("expected Array<bool>, got {flags_val:?}");
        };

        assert_eq!(flags.len_fast(), 10_001);
        assert!(
            flags.is_unboxed(),
            "F-05 regression (use_mir={use_mir}): flags must remain unboxed after full sieve"
        );

        // Backing `elements: Vec<Value>` must NEVER have been materialized or allocated!
        {
            let raw_elems = flags.borrow_elements_without_sync();
            assert!(
                raw_elems.is_empty(),
                "F-05 regression (use_mir={use_mir}): elements must remain empty"
            );
            assert_eq!(
                raw_elems.capacity(),
                0,
                "F-05 regression (use_mir={use_mir}): elements capacity must remain 0 (zero materialization)"
            );
        }

        flags.with_unboxed_data(|data| match data {
            UnboxedArrayData::Bool(bytes) => {
                assert_eq!(bytes.len(), 10_001);
                assert_eq!(core::mem::size_of_val(bytes.as_slice()), 10_001);
                assert_eq!(bytes[2], 1, "2 is prime");
                assert_eq!(bytes[3], 1, "3 is prime");
                assert_eq!(bytes[4], 0, "4 is composite");
                assert_eq!(bytes[9973], 1, "9973 is prime");
                assert_eq!(bytes[10_000], 0, "10000 is composite");
            }
            other => panic!("expected UnboxedArrayData::Bool, got {other:?}"),
        });
    }

    // Also run the exact benchmark script `benchmarks/suite/sieve.lust` (with N = 10_000)
    // in `--jit=off` mode and inspect `__root::flags` directly!
    let sieve_10k_script = load_sieve_source().replace("10000000", "10000");
    let (funcs, struct_defs) = compile_source(&sieve_10k_script, false);
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    let mut vm = VM::with_config(&config);
    vm.console = VirtualConsole::captured();
    vm.register_structs(&struct_defs);
    vm.load_functions(funcs);
    vm.call("__script", vec![])
        .expect("sieve 10k script should succeed");
    let stdout = vm.console.stdout_string();
    assert_eq!(stdout.trim(), "1229");

    let script_flags = vm
        .get_global("__root::flags")
        .expect("__root::flags global must exist after running sieve.lust");
    let Value::Array(flags) = script_flags else {
        panic!("expected __root::flags to be Value::Array, got {script_flags:?}");
    };
    assert_eq!(flags.len_fast(), 10_001);
    assert_eq!(
        flags.borrow_elements_without_sync().capacity(),
        0,
        "F-05 regression on sieve.lust script: elements capacity must be 0"
    );
    flags.with_unboxed_data(|data| match data {
        UnboxedArrayData::Bool(bytes) => assert_eq!(bytes.len(), 10_001),
        other => panic!("expected UnboxedArrayData::Bool on sieve.lust script, got {other:?}"),
    });
}

#[test]
fn test_ng_w10_4_direct_opcodes_len_push_get_unwrap_set_stay_unboxed() {
    let src = r#"
function exercise_arrays(): (int, float, int, Array<int>, Array<float>, Array<bool>)
    local ints: Array<int> = []
    local floats: Array<float> = []
    local bools: Array<bool> = []
    local i: int = 0
    while i < 64 do
        array.push(ints, i * 2)
        array.push(floats, (i * 1.0) + 0.5)
        array.push(bools, (i % 2) == 0)
        i = i + 1
    end

    array.set(ints, 10, 999)
    array.set(floats, 10, 12.25)
    array.set(bools, 10, false)

    local int_sum: int = array.len(ints) + array.get(ints, 10):unwrap() + array.get(ints, 5).unwrap()
    local float_val: float = array.get(floats, 10):unwrap() + array.get(floats, 1):unwrap()
    local bool_hits: int = 0
    if array.get(bools, 0):unwrap() then
        bool_hits = bool_hits + 1
    end
    if not array.get(bools, 10):unwrap() then
        bool_hits = bool_hits + 1
    end
    return int_sum, float_val, bool_hits, ints, floats, bools
end
"#;

    let (funcs, struct_defs) = compile_source(src, false);
    let f = funcs
        .iter()
        .find(|f| f.name == "exercise_arrays")
        .expect("exercise_arrays must exist");
    let disasm = f.disassemble();
    assert!(disasm.contains("ArrayPush"), "missing ArrayPush:\n{disasm}");
    assert!(disasm.contains("ArrayLen"), "missing ArrayLen:\n{disasm}");
    assert!(
        disasm.contains("ArrayGetUnwrap"),
        "missing ArrayGetUnwrap:\n{disasm}"
    );
    assert!(disasm.contains("SetIndex"), "missing SetIndex:\n{disasm}");
    assert_eq!(
        count_load_global_named(f, "array"),
        0,
        "must have 0 LoadGlobal(\"array\"):\n{disasm}"
    );

    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    let mut vm = VM::with_config(&config);
    vm.register_structs(&struct_defs);
    vm.load_functions(funcs);

    let res = vm
        .call("exercise_arrays", vec![])
        .expect("exercise_arrays should succeed");
    let Value::Tuple(items) = res else {
        panic!("expected tuple result, got {res:?}");
    };
    assert_eq!(items[0], Value::Int(64 + 999 + 10));
    assert_eq!(items[1], Value::Float(12.25 + 1.5));
    assert_eq!(items[2], Value::Int(2));

    let Value::Array(ints) = &items[3] else {
        panic!("expected ints array");
    };
    let Value::Array(floats) = &items[4] else {
        panic!("expected floats array");
    };
    let Value::Array(bools) = &items[5] else {
        panic!("expected bools array");
    };

    assert_eq!(ints.borrow_elements_without_sync().capacity(), 0);
    assert_eq!(floats.borrow_elements_without_sync().capacity(), 0);
    assert_eq!(bools.borrow_elements_without_sync().capacity(), 0);

    ints.with_unboxed_data(|d| assert!(matches!(d, UnboxedArrayData::Int(v) if v.len() == 64)));
    floats.with_unboxed_data(|d| assert!(matches!(d, UnboxedArrayData::Float(v) if v.len() == 64)));
    bools.with_unboxed_data(|d| assert!(matches!(d, UnboxedArrayData::Bool(v) if v.len() == 64)));
}

#[test]
fn test_ng_w10_4_local_array_shadowing_does_not_emit_array_opcodes() {
    let src = r#"
struct FakeArray
    len: function(Array<int>): int
end

function fake_len(a: Array<int>): int
    return 999
end

function test_shadow(): int
    local array: FakeArray = FakeArray { len = fake_len }
    local xs: Array<int> = [1, 2]
    return array.len(xs)
end
"#;

    let (funcs, struct_defs) = compile_source(src, false);
    let f = funcs
        .iter()
        .find(|f| f.name == "test_shadow")
        .expect("test_shadow must exist");
    let disasm = f.disassemble();
    assert!(
        !disasm.contains("ArrayLen"),
        "shadowed local `array` must NOT emit ArrayLen:\n{disasm}"
    );

    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    let mut vm = VM::with_config(&config);
    vm.register_structs(&struct_defs);
    vm.load_functions(funcs);

    let res = vm
        .call("test_shadow", vec![])
        .expect("test_shadow should succeed");
    assert_eq!(res, Value::Int(999));
}

#[test]
fn test_ng_w10_4_bounds_errors_and_memory_budget_on_array_push() {
    let src = r#"
function oob_get(): int
    local a: Array<int> = []
    array.push(a, 10)
    return array.get(a, 5):unwrap()
end

function neg_get(): int
    local a: Array<int> = []
    array.push(a, 10)
    return array.get(a, -1):unwrap()
end

function oob_set(): int
    local a: Array<int> = []
    array.push(a, 10)
    array.set(a, 5, 99)
    return 0
end

function push_many(n: int): int
    local a: Array<int> = []
    local i: int = 0
    while i < n do
        array.push(a, i)
        i = i + 1
    end
    return array.len(a)
end
"#;

    let (funcs, struct_defs) = compile_source(src, false);
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    let mut vm = VM::with_config(&config);
    vm.register_structs(&struct_defs);
    vm.load_functions(funcs);

    let err1 = vm.call("oob_get", vec![]).expect_err("oob_get must fail");
    assert!(err1.to_string().contains("out of bounds"), "got: {err1}");

    let err2 = vm.call("neg_get", vec![]).expect_err("neg_get must fail");
    assert!(err2.to_string().contains("out of bounds"), "got: {err2}");

    let err3 = vm.call("oob_set", vec![]).expect_err("oob_set must fail");
    assert!(err3.to_string().contains("out of bounds"), "got: {err3}");

    vm.set_memory_budget_bytes(128)
        .expect("set_memory_budget_bytes should succeed");
    let err4 = vm
        .call("push_many", vec![Value::Int(1000)])
        .expect_err("push_many under 128-byte budget must fail with OOM");
    assert!(
        err4.to_string().contains("Out of memory budget"),
        "expected Out of memory budget, got: {err4}"
    );
}

#[test]
fn test_ng_w10_4_instruction_encode_decode_roundtrip() {
    let push = Instruction::ArrayPush(2, 7);
    assert_eq!(push.opcode(), OpCode::ArrayPush);
    assert_eq!(push.defined_register(), None);
    assert!(push.reads_register(2));
    assert!(push.reads_register(7));
    assert!(!push.reads_register(3));
    assert_eq!(format!("{push}"), "ArrayPush R2, R7");
    assert_eq!(Instruction::decode(&push.encode()), Ok(push));

    let get_unwrap = Instruction::ArrayGetUnwrap(4, 1, 9);
    assert_eq!(get_unwrap.opcode(), OpCode::ArrayGetUnwrap);
    assert_eq!(get_unwrap.defined_register(), Some(4));
    assert!(get_unwrap.reads_register(1));
    assert!(get_unwrap.reads_register(9));
    assert!(!get_unwrap.reads_register(4));
    assert_eq!(format!("{get_unwrap}"), "ArrayGetUnwrap R4, R1, R9");
    assert_eq!(Instruction::decode(&get_unwrap.encode()), Ok(get_unwrap));
}

#[test]
fn test_ng_w10_4_native_function_arg_scratch_reuse_and_reentrancy() {
    let src = r#"
extern
    function host_sum(a: int, b: int, c: int): int
    function host_reenter(n: int): int
end

function double_it(x: int): int
    return x * 2
end

function run_native_loop(n: int): int
    local acc: int = 0
    local i: int = 0
    while i < n do
        acc = acc + host_sum(i, i + 1, 1)
        if (i % 10) == 0 then
            acc = acc + host_reenter(i)
        end
        i = i + 1
    end
    return acc
end
"#;

    let (funcs, struct_defs) = compile_source(src, false);
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    let mut vm = VM::with_config(&config);
    vm.register_structs(&struct_defs);
    vm.load_functions(funcs);

    vm.register_native(
        "host_sum",
        Value::NativeFunction(native_fn(|args: &[Value]| {
            let sum: i64 = args.iter().filter_map(Value::as_int).sum();
            Ok(NativeCallResult::Return(Value::Int(sum)))
        })),
    );
    vm.register_native(
        "host_reenter",
        Value::NativeFunction(native_fn(|args: &[Value]| {
            let n = args[0].as_int().unwrap_or(0);
            let res = VM::with_current(|vm| {
                vm.call("double_it", vec![Value::Int(n)])
                    .map_err(|e| e.to_string())
            })?;
            Ok(NativeCallResult::Return(res))
        })),
    );

    let res = vm
        .call("run_native_loop", vec![Value::Int(50)])
        .expect("run_native_loop should succeed");
    let expected: i64 = (0..50)
        .map(|i| (i + (i + 1) + 1) + if i % 10 == 0 { i * 2 } else { 0 })
        .sum();
    assert_eq!(res, Value::Int(expected));
}

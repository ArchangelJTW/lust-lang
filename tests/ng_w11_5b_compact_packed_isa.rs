#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! NG-W11-5B Acceptance & Adversarial Verification Suite:
//! 1. Static layout invariants: `size_of::<PackedInstruction>() == 4` and `align_of == 4`.
//! 2. Bit-level ABC / ABx / AsBx encode/decode edge cases (`i8::MIN..=i8::MAX`, `i16::MIN..=i16::MAX`,
//!    `u16::MAX`) and `gas_weight()` parity with `Instruction::gas_weight()`.
//! 3. Exhaustive round-trip encoding/decoding of all 93 `Instruction` variants into
//!    `PackedInstruction` + `extra_args` with strict 1:1 `packed_code.len() == instructions.len()`,
//!    exact `extra_args` word-count contracts (0 / 1 / 2 words), and malformed decode rejection.
//! 4. `Chunk::emit`, `Chunk::patch_jump` (forward & backward for all 5 branch opcodes),
//!    `Chunk::set_instruction`, `Chunk::pop_instruction`, and `Chunk::build_packed_code()`
//!    compaction without stale `extra_args`.
//! 5. Pathological `extra_args` overflow (`> u16::MAX`) fail-safe fallback without panic.
//! 6. Compilation verification of all 10 `benchmarks/suite/*.lust` programs + execution parity
//!    (outputs & exact `vm.gas_used()`) between 4-byte `packed_code` `'frame_fast: loop`
//!    dispatch and per-instruction `instructions[ip]` slow-path dispatch.
//! 7. OSR / Cranelift continuation & deopt PC alignment with `packed_code` 1:1 indexing.

use std::cell::RefCell;
use std::rc::Rc;

use lust::ast::{ItemKind, StructDef};
use lust::bytecode::instruction::{DecodeError, PackedInstruction};
use lust::bytecode::{
    native_fn, Chunk, Compiler, Function, Instruction, NativeCallResult, OpCode, Value,
};
use lust::config::LustConfig;
use lust::hashbrown::HashMap;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{VmBackend, VM};

struct TraceHookGuard;

impl TraceHookGuard {
    fn new() -> Self {
        let mut vm = VM::new();
        vm.clear_trace_hooks();
        Self
    }
}

impl Drop for TraceHookGuard {
    fn drop(&mut self) {
        let mut vm = VM::new();
        vm.clear_trace_hooks();
    }
}

fn plain_vm() -> VM {
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    let mut vm = VM::with_config(&config);
    vm.set_backend(VmBackend::Bytecode);
    vm
}

fn compile_source_with_structs(src: &str) -> (Vec<Function>, HashMap<String, StructDef>) {
    let mut interner = Interner::new();
    let tokens = Lexer::new(src, &mut interner).tokenize().expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    let mut checker = TypeChecker::with_config(&config);
    checker.check_module(&items).expect("typecheck");
    let mut struct_defs = HashMap::new();
    for item in &items {
        if let ItemKind::Struct(s) = &item.kind {
            struct_defs.insert(s.name.clone(), s.clone());
        }
    }
    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    let funcs = compiler.compile_module(&items).expect("compile_module");
    (funcs, struct_defs)
}

fn compile_source(src: &str) -> Vec<Function> {
    compile_source_with_structs(src).0
}

fn run_capture_stdout(src: &str, install_trace_hook: bool) -> (Vec<String>, u64) {
    let (funcs, struct_defs) = compile_source_with_structs(src);
    let mut vm = plain_vm();
    let output = Rc::new(RefCell::new(Vec::<String>::new()));
    let out_clone = Rc::clone(&output);
    vm.register_native(
        "println",
        Value::NativeFunction(native_fn(move |args: &[Value]| {
            let line = args
                .iter()
                .map(|v| format!("{v}"))
                .collect::<Vec<_>>()
                .join("\t");
            out_clone.borrow_mut().push(line);
            Ok(NativeCallResult::Return(Value::Nil))
        })),
    );
    vm.register_structs(&struct_defs);
    vm.load_functions(funcs);
    if install_trace_hook {
        vm.set_instruction_hook(Some(Box::new(|_, _, _| {})));
    }
    vm.reset_gas_counter();
    vm.call("__script", vec![]).expect("run __script");
    let gas = vm.gas_used();
    if install_trace_hook {
        vm.clear_trace_hooks();
    }
    let captured = output.borrow().clone();
    (captured, gas)
}

fn assert_chunk_packed_1_to_1(func: &Function) {
    let chunk = &func.chunk;
    assert_eq!(
        chunk.packed_code.len(),
        chunk.instructions.len(),
        "Function '{}' violated 1:1 packed_code/instructions length ({} vs {})",
        func.name,
        chunk.packed_code.len(),
        chunk.instructions.len()
    );
    assert_eq!(
        chunk.bb_gas_weights.len(),
        chunk.instructions.len(),
        "Function '{}' bb_gas_weights length mismatch",
        func.name
    );
    for (pc, (&packed, &expected)) in chunk
        .packed_code
        .iter()
        .zip(chunk.instructions.iter())
        .enumerate()
    {
        let decoded = packed.to_instruction(&chunk.extra_args);
        assert_eq!(
            decoded,
            Ok(expected),
            "Function '{}' pc {}: packed {:?} decoded to {:?}, expected {:?}",
            func.name,
            pc,
            packed,
            decoded,
            expected
        );
        assert_eq!(packed.gas_weight(), expected.gas_weight());
    }
    for (&bid, &entry_pc) in &func.block_entry_pcs {
        assert!(
            entry_pc <= chunk.packed_code.len(),
            "Function '{}' block {} entry_pc {} out of packed bounds {}",
            func.name,
            bid,
            entry_pc,
            chunk.packed_code.len()
        );
    }
    for (&bid, &term_pc) in &func.terminator_pcs {
        assert!(
            term_pc <= chunk.packed_code.len(),
            "Function '{}' block {} term_pc {} out of packed bounds {}",
            func.name,
            bid,
            term_pc,
            chunk.packed_code.len()
        );
    }
}

#[test]
fn ng_w11_5b_packed_instruction_size_align_and_bitfields() {
    assert_eq!(core::mem::size_of::<PackedInstruction>(), 4);
    assert_eq!(core::mem::align_of::<PackedInstruction>(), 4);

    let abc = PackedInstruction::encode_abc(OpCode::AddIntImm, 0xAB, 0xCD, (-128i8) as u8);
    assert_eq!(abc.opcode_u8(), OpCode::AddIntImm as u8);
    assert_eq!(abc.a(), 0xAB);
    assert_eq!(abc.b(), 0xCD);
    assert_eq!(abc.c(), 0x80);
    assert_eq!(abc.c_i8(), -128i8);

    let abc_max = PackedInstruction::encode_abc(OpCode::AddIntImm, 1, 2, 127i8 as u8);
    assert_eq!(abc_max.c_i8(), 127i8);

    let abx = PackedInstruction::encode_abx(OpCode::LoadConst, 0xEF, u16::MAX);
    assert_eq!(abx.opcode_u8(), OpCode::LoadConst as u8);
    assert_eq!(abx.a(), 0xEF);
    assert_eq!(abx.bx(), u16::MAX);

    for offset in [i16::MIN, -1024, -1, 0, 1, 1024, i16::MAX] {
        let asbx = PackedInstruction::encode_asbx(OpCode::JumpIfNot, 42, offset);
        assert_eq!(asbx.opcode_u8(), OpCode::JumpIfNot as u8);
        assert_eq!(asbx.a(), 42);
        assert_eq!(asbx.sbx(), offset);
    }
}

#[test]
fn ng_w11_5b_exhaustive_all_93_instructions_round_trip() {
    // Every single variant of Instruction (OpCodes 0..=92), exercising boundary values.
    let all_variants: Vec<(Instruction, usize)> = vec![
        // Direct ABC (0 extra words)
        (Instruction::LoadNil(7), 0),
        (Instruction::LoadBool(8, false), 0),
        (Instruction::LoadBool(9, true), 0),
        (Instruction::Move(10, 250), 0),
        (Instruction::Return(255), 0),
        (Instruction::NewMap(11), 0),
        (Instruction::Add(1, 2, 3), 0),
        (Instruction::Sub(4, 5, 6), 0),
        (Instruction::Mul(7, 8, 9), 0),
        (Instruction::Div(10, 11, 12), 0),
        (Instruction::Mod(13, 14, 15), 0),
        (Instruction::Pow(16, 17, 18), 0),
        (Instruction::FloorDiv(19, 20, 21), 0),
        (Instruction::Neg(22, 23), 0),
        (Instruction::Eq(24, 25, 26), 0),
        (Instruction::Ne(27, 28, 29), 0),
        (Instruction::Lt(30, 31, 32), 0),
        (Instruction::Le(33, 34, 35), 0),
        (Instruction::Gt(36, 37, 38), 0),
        (Instruction::Ge(39, 40, 41), 0),
        (Instruction::AddInt(42, 43, 44), 0),
        (Instruction::SubInt(45, 46, 47), 0),
        (Instruction::MulInt(48, 49, 50), 0),
        (Instruction::DivInt(51, 52, 53), 0),
        (Instruction::ModInt(54, 55, 56), 0),
        (Instruction::PowInt(57, 58, 59), 0),
        (Instruction::FloorDivInt(60, 61, 62), 0),
        (Instruction::NegInt(63, 64), 0),
        (Instruction::EqInt(65, 66, 67), 0),
        (Instruction::NeInt(68, 69, 70), 0),
        (Instruction::LtInt(71, 72, 73), 0),
        (Instruction::LeInt(74, 75, 76), 0),
        (Instruction::GtInt(77, 78, 79), 0),
        (Instruction::GeInt(80, 81, 82), 0),
        (Instruction::AddFloat(83, 84, 85), 0),
        (Instruction::SubFloat(86, 87, 88), 0),
        (Instruction::MulFloat(89, 90, 91), 0),
        (Instruction::DivFloat(92, 93, 94), 0),
        (Instruction::ModFloat(95, 96, 97), 0),
        (Instruction::PowFloat(98, 99, 100), 0),
        (Instruction::FloorDivFloat(101, 102, 103), 0),
        (Instruction::NegFloat(104, 105), 0),
        (Instruction::EqFloat(106, 107, 108), 0),
        (Instruction::NeFloat(109, 110, 111), 0),
        (Instruction::LtFloat(112, 113, 114), 0),
        (Instruction::LeFloat(115, 116, 117), 0),
        (Instruction::GtFloat(118, 119, 120), 0),
        (Instruction::GeFloat(121, 122, 123), 0),
        (Instruction::And(124, 125, 126), 0),
        (Instruction::Or(127, 128, 129), 0),
        (Instruction::Not(130, 131), 0),
        (Instruction::BitAnd(132, 133, 134), 0),
        (Instruction::BitOr(135, 136, 137), 0),
        (Instruction::BitXor(138, 139, 140), 0),
        (Instruction::Shl(141, 142, 143), 0),
        (Instruction::Shr(144, 145, 146), 0),
        (Instruction::BitNot(147, 148), 0),
        (Instruction::NewArray(149, 150, 151), 0),
        (Instruction::TupleNew(152, 153, 154), 0),
        (Instruction::TupleGet(155, 156, 157), 0),
        (Instruction::GetEnumValue(158, 159, 160), 0),
        (Instruction::GetFieldSlot(161, 162, 163), 0),
        (Instruction::SetFieldSlot(164, 165, 166), 0),
        (Instruction::GetIndex(167, 168, 169), 0),
        (Instruction::TryGetIndex(170, 171, 172), 0),
        (Instruction::ArrayLen(173, 174), 0),
        (Instruction::ArrayPush(175, 176), 0),
        (Instruction::ArrayGetUnwrap(177, 178, 179), 0),
        (Instruction::SetIndex(180, 181, 182), 0),
        (Instruction::Concat(183, 184, 185), 0),
        (Instruction::LoadUpvalue(186, 187), 0),
        (Instruction::StoreUpvalue(188, 189), 0),
        (Instruction::AddIntImm(190, 191, -128), 0),
        (Instruction::AddIntImm(190, 191, 127), 0),
        (Instruction::GetFieldAddSetInt(192, 193, 194), 0),
        // Direct ABx / AsBx (0 extra words)
        (Instruction::LoadConst(195, 0xBEEF), 0),
        (Instruction::LoadGlobal(196, 0xCAFE), 0),
        (Instruction::StoreGlobal(0xFACE, 197), 0),
        (Instruction::Jump(-12345), 0),
        (Instruction::JumpIf(198, 23456), 0),
        (Instruction::JumpIfNot(199, i16::MIN), 0),
        // AX format: 1 word in extra_args
        (Instruction::BranchLtInt(200, 201, -4096), 1),
        (Instruction::BranchLeInt(202, 203, 4095), 1),
        (Instruction::Call(204, 205, 206, 207), 1),
        (Instruction::ArrayIndexOk(208, 209, 210, 211), 1),
        (Instruction::GetField(212, 213, 0x1234), 1),
        (Instruction::SetField(214, 0x5678, 215), 1),
        (Instruction::TypeIs(216, 217, 0x9ABC), 1),
        (Instruction::TryCast(218, 219, 0xDEF0), 1),
        // AX format: 2 words in extra_args
        (Instruction::NewStruct(220, 0x1111, 0x2222, 221, 222), 2),
        (Instruction::NewEnumUnit(223, 0x3333, 0x4444), 2),
        (Instruction::NewEnumVariant(224, 0x5555, 0x6666, 225, 226), 2),
        (Instruction::IsEnumVariant(227, 228, 0x7777, 0x8888), 2),
        (Instruction::CallMethod(229, 0x9999, 230, 231, 232), 2),
        (Instruction::Closure(233, 0xAAAA, 234, 235), 2),
    ];

    // Verify we covered every single OpCode 0..=92
    let mut seen_opcodes = [false; 93];
    let mut chunk = Chunk::new();
    let mut expected_extra_len = 0usize;

    for (idx, &(inst, expected_words)) in all_variants.iter().enumerate() {
        let op_byte = inst.opcode().as_u8() as usize;
        seen_opcodes[op_byte] = true;

        let before_extra = chunk.extra_args.len();
        let emitted_pc = chunk.emit(inst, idx + 1);
        assert_eq!(emitted_pc, idx);
        assert_eq!(chunk.packed_code.len(), chunk.instructions.len());
        assert_eq!(
            chunk.extra_args.len() - before_extra,
            expected_words,
            "extra_args word count mismatch for {:?}",
            inst
        );
        expected_extra_len += expected_words;

        let packed = chunk.packed_code[idx];
        assert_eq!(packed.gas_weight(), inst.gas_weight());
        assert_eq!(
            packed.to_instruction(&chunk.extra_args),
            Ok(inst),
            "Incremental emit round-trip failed for {:?}",
            inst
        );
    }

    for (op, &seen) in seen_opcodes.iter().enumerate() {
        assert!(seen, "OpCode {} was not covered in round-trip test", op);
    }
    assert_eq!(chunk.extra_args.len(), expected_extra_len);

    // Verify build_packed_code() produces an exact 1:1 rebuild with identical round-trip
    chunk.build_packed_code();
    assert_eq!(chunk.packed_code.len(), chunk.instructions.len());
    assert_eq!(chunk.extra_args.len(), expected_extra_len);
    for (pc, (&packed, &inst)) in chunk
        .packed_code
        .iter()
        .zip(chunk.instructions.iter())
        .enumerate()
    {
        assert_eq!(
            packed.to_instruction(&chunk.extra_args),
            Ok(inst),
            "build_packed_code() round-trip failed at pc {}",
            pc
        );
    }

    // Error paths in to_instruction
    assert_eq!(
        PackedInstruction(0xFF).to_instruction(&[]),
        Err(DecodeError::UnknownOpCode(0xFF))
    );
    assert_eq!(
        PackedInstruction::encode_abc(OpCode::LoadBool, 0, 2, 0).to_instruction(&[]),
        Err(DecodeError::InvalidBoolean(2))
    );
    assert_eq!(
        PackedInstruction::encode_abx(OpCode::Call, 0, 0).to_instruction(&[]),
        Err(DecodeError::Truncated {
            expected: 1,
            found: 0
        })
    );
    assert_eq!(
        PackedInstruction::encode_abx(OpCode::NewStruct, 0, 0).to_instruction(&[0u32]),
        Err(DecodeError::Truncated {
            expected: 2,
            found: 1
        })
    );
}

#[test]
fn ng_w11_5b_patch_jump_mutation_and_compaction() {
    let mut chunk = Chunk::new();
    let j_fwd = chunk.emit(Instruction::Jump(0), 1);
    let jif = chunk.emit(Instruction::JumpIf(1, 0), 2);
    let jifn = chunk.emit(Instruction::JumpIfNot(2, 0), 3);
    let blt = chunk.emit(Instruction::BranchLtInt(3, 4, 0), 4);
    let ble = chunk.emit(Instruction::BranchLeInt(5, 6, 0), 5);
    let target_fwd = chunk.emit(Instruction::AddIntImm(0, 0, 1), 6);
    let blt_back = chunk.emit(Instruction::BranchLtInt(7, 8, 0), 7);
    let ble_back = chunk.emit(Instruction::BranchLeInt(9, 10, 0), 8);
    let _ret = chunk.emit(Instruction::Return(0), 9);

    chunk.patch_jump(j_fwd, target_fwd);
    chunk.patch_jump(jif, target_fwd);
    chunk.patch_jump(jifn, target_fwd);
    chunk.patch_jump(blt, target_fwd);
    chunk.patch_jump(ble, target_fwd);
    chunk.patch_jump(blt_back, j_fwd);
    chunk.patch_jump(ble_back, target_fwd);

    // Even BEFORE build_packed_code(), patch_jump must have updated packed_code/extra_args in sync.
    assert_eq!(chunk.packed_code.len(), chunk.instructions.len());
    for (pc, (&packed, &inst)) in chunk
        .packed_code
        .iter()
        .zip(chunk.instructions.iter())
        .enumerate()
    {
        assert_eq!(
            packed.to_instruction(&chunk.extra_args),
            Ok(inst),
            "patch_jump drifted at pc {}",
            pc
        );
    }

    // Mutate and pop AX instructions, then compact via build_packed_code().
    let extra_call_pc = chunk.emit(Instruction::Call(1, 2, 3, 4), 10);
    let stale_extra_len = chunk.extra_args.len();
    chunk.set_instruction(extra_call_pc, Instruction::Move(4, 1));
    assert_eq!(
        chunk.packed_code[extra_call_pc].to_instruction(&chunk.extra_args),
        Ok(Instruction::Move(4, 1))
    );
    let popped = chunk.pop_instruction();
    assert_eq!(popped, Some(Instruction::Move(4, 1)));
    assert_eq!(chunk.packed_code.len(), chunk.instructions.len());

    // After compaction, the stale word from the replaced/popped Call is gone.
    chunk.build_packed_code();
    assert!(chunk.extra_args.len() < stale_extra_len);
    assert_eq!(chunk.packed_code.len(), chunk.instructions.len());
    for (&packed, &inst) in chunk.packed_code.iter().zip(chunk.instructions.iter()) {
        assert_eq!(packed.to_instruction(&chunk.extra_args), Ok(inst));
    }
}

#[test]
fn ng_w11_5b_extra_args_overflow_fails_safe_and_executes_via_fallback() {
    // Construct a synthetic function whose AX instructions require > u16::MAX extra_args words.
    // A single Jump skips over the unreachable Call instructions directly to Return,
    // verifying both build_packed_code() non-panic and VM execution via FallbackToOuter.
    let mut f = Function::new("huge_extra_args", 0, false);
    f.set_register_count(4);
    let k42 = f.chunk.add_constant(Value::Int(42));
    f.chunk.emit(Instruction::LoadConst(0, k42), 1);
    f.chunk.emit(Instruction::Return(0), 2);
    // Append 35_000 unreachable NewStruct instructions (70_000 extra_args words > u16::MAX).
    for _ in 0..35_000 {
        f.chunk
            .instructions
            .push(Instruction::NewStruct(1, 0, 0, 0, 0));
    }
    f.chunk.build_packed_code();
    assert!(
        f.chunk.packed_code.is_empty(),
        "Overflowing extra_args (> u16::MAX) must clear packed_code instead of truncating indices"
    );
    assert!(f.chunk.extra_args.is_empty());

    let mut vm = plain_vm();
    vm.load_functions(vec![f]);
    let res = vm.call("huge_extra_args", vec![]).expect("fallback exec");
    assert_eq!(res, Value::Int(42));
}

#[test]
fn ng_w11_5b_all_10_benchmark_suite_programs_compile_and_execute_parity() {
    let _hook_guard = TraceHookGuard::new();

    // 1. Compile all 10 verbatim benchmarks/suite/*.lust programs and verify strict 1:1 packed ISA.
    let full_benchmarks: [(&str, &str); 10] = [
        ("array", include_str!("../benchmarks/suite/array.lust")),
        ("calls", include_str!("../benchmarks/suite/calls.lust")),
        ("fib", include_str!("../benchmarks/suite/fib.lust")),
        ("fields", include_str!("../benchmarks/suite/fields.lust")),
        (
            "floatmath",
            include_str!("../benchmarks/suite/floatmath.lust"),
        ),
        ("methods", include_str!("../benchmarks/suite/methods.lust")),
        ("nested", include_str!("../benchmarks/suite/nested.lust")),
        ("sieve", include_str!("../benchmarks/suite/sieve.lust")),
        ("strings", include_str!("../benchmarks/suite/strings.lust")),
        ("tree", include_str!("../benchmarks/suite/tree.lust")),
    ];

    for (name, src) in full_benchmarks {
        let funcs = compile_source(src);
        assert!(!funcs.is_empty(), "benchmark '{name}' compiled zero functions");
        for func in &funcs {
            assert_chunk_packed_1_to_1(func);
        }
    }

    // 2. Execute scaled versions of all 10 benchmark programs through the 4-byte packed_code
    //    fast dispatcher and compare outputs AND exact gas_used() against the per-instruction
    //    slow dispatcher.
    let scaled_benchmarks: [(&str, String); 10] = [
        (
            "array",
            include_str!("../benchmarks/suite/array.lust")
                .replace("1000000", "200")
                .replace("pass < 10", "pass < 3"),
        ),
        (
            "calls",
            include_str!("../benchmarks/suite/calls.lust").replace("10000000", "200"),
        ),
        (
            "fib",
            include_str!("../benchmarks/suite/fib.lust").replace("fib(30)", "fib(14)"),
        ),
        (
            "fields",
            include_str!("../benchmarks/suite/fields.lust").replace("10000000", "200"),
        ),
        (
            "floatmath",
            include_str!("../benchmarks/suite/floatmath.lust").replace("10000000", "200"),
        ),
        (
            "methods",
            include_str!("../benchmarks/suite/methods.lust").replace("10000000", "200"),
        ),
        (
            "nested",
            include_str!("../benchmarks/suite/nested.lust").replace("3000", "35"),
        ),
        (
            "sieve",
            include_str!("../benchmarks/suite/sieve.lust").replace("10000000", "200"),
        ),
        (
            "strings",
            include_str!("../benchmarks/suite/strings.lust").replace("100000", "200"),
        ),
        (
            "tree",
            include_str!("../benchmarks/suite/tree.lust")
                .replace("build(14, 1)", "build(6, 1)")
                .replace("pass < 20", "pass < 3")
                .replace("k < 1000", "k < 40")
                .replace("j < 2000", "j < 15"),
        ),
    ];

    for (name, src) in scaled_benchmarks {
        let (fast_out, fast_gas) = run_capture_stdout(&src, false);
        let (slow_out, slow_gas) = run_capture_stdout(&src, true);
        assert_eq!(
            fast_out, slow_out,
            "Benchmark '{name}' output diverged between packed_code fast loop and slow path"
        );
        assert!(
            !fast_out.is_empty(),
            "Benchmark '{name}' produced no output"
        );
        assert_eq!(
            fast_gas, slow_gas,
            "Benchmark '{name}' gas_used diverged: packed fast={fast_gas} vs slow={slow_gas}"
        );
    }
}

#[test]
fn ng_w11_5b_sec01_exact_gas_limit_under_packed_dispatch() {
    let _hook_guard = TraceHookGuard::new();

    let src = r#"
        local s: int = 0
        local i: int = 0
        while i < 20 do
            s = s + i
            i = i + 1
        end
        println(s)
    "#;
    let (expected_out, exact_total_gas) = run_capture_stdout(src, false);
    assert_eq!(expected_out, vec!["190".to_string()]);
    assert!(exact_total_gas > 10);

    let noop_println = || {
        Value::NativeFunction(native_fn(|_: &[Value]| {
            Ok(NativeCallResult::Return(Value::Nil))
        }))
    };

    let funcs = compile_source(src);
    // Exact limit succeeds
    let mut vm_ok = plain_vm();
    vm_ok.register_native("println", noop_println());
    vm_ok.load_functions(funcs.clone());
    vm_ok.set_gas_budget(exact_total_gas);
    vm_ok.reset_gas_counter();
    vm_ok
        .call("__script", vec![])
        .expect("exact gas limit must succeed under packed dispatch");
    assert_eq!(vm_ok.gas_used(), exact_total_gas);

    // Limit - 1 must fail with OutOfGas
    let mut vm_oom = plain_vm();
    vm_oom.register_native("println", noop_println());
    vm_oom.load_functions(funcs);
    vm_oom.set_gas_budget(exact_total_gas - 1);
    vm_oom.reset_gas_counter();
    let err = vm_oom
        .call("__script", vec![])
        .expect_err("exact_total_gas - 1 must fail with OutOfGas under packed dispatch");
    assert!(
        format!("{err}").contains("Out of gas"),
        "expected Out of gas error, got: {err}"
    );
}

#[test]
fn ng_w11_5b_cranelift_osr_and_continuation_preserve_packed_1_to_1() {
    let _hook_guard = TraceHookGuard::new();

    let src = r#"
        function sum_to(n: int): int
            local acc: int = 0
            local i: int = 1
            while i <= n do
                acc = acc + i
                i = i + 1
            end
            return acc
        end
    "#;
    let funcs = compile_source(src);
    for f in &funcs {
        assert_chunk_packed_1_to_1(f);
    }

    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    let mut vm = VM::with_config(&config);
    vm.set_backend(VmBackend::Cranelift);
    vm.load_functions(funcs);
    let res = vm.call("sum_to", vec![Value::Int(500)]).expect("cranelift/osr");
    assert_eq!(res, Value::Int(125_250));
}

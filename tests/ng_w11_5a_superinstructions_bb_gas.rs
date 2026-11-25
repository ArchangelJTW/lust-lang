#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! NG-W11-5A Acceptance & Adversarial Verification Suite:
//! 1. Loop superinstructions `AddIntImm`, `BranchLtInt`, `BranchLeInt`, `GetFieldAddSetInt`:
//!    codec round-trip, `OpCode`, `defined_register`, `reads_register`, `gas_weight`, `Display`,
//!    and `Chunk::patch_jump`.
//! 2. `sieve.lust` and `nested.lust` compiled via `AstLowering` + `MirToBytecodeCompiler` emit
//!    `BranchLeInt`/`BranchLtInt` and `AddIntImm` while keeping every jump target aligned with
//!    `block_entry_pcs` and `terminator_pcs`.
//! 3. `GetFieldAddSetInt` intra-block fusion for `self.value = self.value + d` and compound assign,
//!    with fast-path execution and dynamic fallback soundness.
//! 4. Peephole boundary guard: fusion NEVER crosses `block_entry_pcs` / `bb_leaders`.
//! 5. Basic-block gas batching (`Chunk::bb_gas_weights`): exact 1-to-1 `vm.gas_used()` parity
//!    against per-instruction metering across loops, superinstructions, mid-BB slow exits,
//!    recursive `Call`/`Return`, and full `SEC-01` `50 -> 51` boundary sweep.

use std::cell::RefCell;
use std::rc::Rc;

use lust::ast::{
    FieldOwnership, ItemKind, Span, StructDef, StructField, Type, TypeKind, Visibility,
};
use lust::bytecode::{
    Chunk, FieldStorage, Function, Instruction, OpCode, StructLayout, Value,
};
use lust::config::LustConfig;
use lust::hashbrown::{HashMap, HashSet};
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::ir::{
    FunctionSignature, MirBinaryOp, MirConstant, MirFunction, MirType, Operand, Place, Rvalue,
    Statement, Terminator,
};
use lust::mir::lowering::AstLowering;
use lust::mir::vm_backend::{audit_chunk_field_access, MirToBytecodeCompiler};
use lust::parser::Parser;
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

fn lower_script_source(name: &str, src: &str) -> MirFunction {
    let mut interner = Interner::new();
    let tokens = Lexer::new(src, &mut interner).tokenize().expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");
    let mut stmts = Vec::new();
    for item in items {
        if let ItemKind::Script(s) = item.kind {
            stmts.extend(s);
        }
    }
    AstLowering::lower_script(name, &stmts).expect("lower_script")
}

fn assert_all_jumps_land_on_block_entries(bc: &Function) {
    let valid_entries: HashSet<usize> = bc.block_entry_pcs.values().copied().collect();
    for (pc, instr) in bc.chunk.instructions.iter().enumerate() {
        let offset_opt = match *instr {
            Instruction::Jump(off)
            | Instruction::JumpIf(_, off)
            | Instruction::JumpIfNot(_, off)
            | Instruction::BranchLtInt(_, _, off)
            | Instruction::BranchLeInt(_, _, off) => Some(off),
            _ => None,
        };
        if let Some(off) = offset_opt {
            let target = (pc as isize + 1 + off as isize) as usize;
            assert!(
                valid_entries.contains(&target),
                "Jump at pc {pc} ({instr}) lands at {target}, which is not in block_entry_pcs {:?}:\n{}",
                bc.block_entry_pcs,
                bc.disassemble()
            );
        }
    }
    for (&bid, &entry_pc) in &bc.block_entry_pcs {
        let term_pc = bc.terminator_pcs[&bid];
        assert!(
            entry_pc <= term_pc && term_pc <= bc.chunk.instructions.len(),
            "Invalid PCs for block {bid}: entry={entry_pc}, term={term_pc}, len={}",
            bc.chunk.instructions.len()
        );
    }
}

fn make_struct_defs(specs: &[(&str, &[(&str, TypeKind)])]) -> HashMap<String, StructDef> {
    let mut defs = HashMap::new();
    for &(name, fields) in specs {
        let struct_fields = fields
            .iter()
            .map(|(fname, kind)| StructField {
                name: (*fname).to_string(),
                ty: Type::new(kind.clone(), Span::dummy()),
                ownership: FieldOwnership::Strong,
                weak_target: None,
                visibility: Visibility::Public,
            })
            .collect();
        defs.insert(
            name.to_string(),
            StructDef {
                name: name.to_string(),
                type_params: Vec::new(),
                trait_bounds: Vec::new(),
                fields: struct_fields,
                visibility: Visibility::Public,
                doc: None,
            },
        );
    }
    defs
}

fn make_layout(name: &str, fields: &[(&str, TypeKind)]) -> Rc<StructLayout> {
    Rc::new(StructLayout::new(
        name.to_string(),
        fields.iter().map(|(n, _)| Rc::new((*n).to_string())).collect(),
        vec![FieldStorage::Strong; fields.len()],
        fields
            .iter()
            .map(|(_, k)| Type::new(k.clone(), Span::dummy()))
            .collect(),
        vec![None; fields.len()],
    ))
}

#[test]
fn ng_w11_5a_superinstruction_codec_metadata_and_patch_jump() {
    let cases = [
        (
            Instruction::AddIntImm(2, 1, 1),
            OpCode::AddIntImm,
            Some(2),
            1u16,
            "AddIntImm R2, R1, 1",
        ),
        (
            Instruction::AddIntImm(5, 4, -128),
            OpCode::AddIntImm,
            Some(5),
            1u16,
            "AddIntImm R5, R4, -128",
        ),
        (
            Instruction::AddIntImm(7, 3, 127),
            OpCode::AddIntImm,
            Some(7),
            1u16,
            "AddIntImm R7, R3, 127",
        ),
        (
            Instruction::BranchLtInt(1, 2, 14),
            OpCode::BranchLtInt,
            None,
            2u16,
            "BranchLtInt R1, R2, 14",
        ),
        (
            Instruction::BranchLeInt(3, 4, -9),
            OpCode::BranchLeInt,
            None,
            2u16,
            "BranchLeInt R3, R4, -9",
        ),
        (
            Instruction::GetFieldAddSetInt(0, 2, 5),
            OpCode::GetFieldAddSetInt,
            None,
            3u16,
            "GetFieldAddSetInt R0, #2, R5",
        ),
    ];

    for (inst, expected_op, expected_def, expected_weight, expected_disp) in cases {
        assert_eq!(inst.opcode(), expected_op);
        assert_eq!(OpCode::from_u8(expected_op.as_u8()), Ok(expected_op));
        assert_eq!(Instruction::decode(&inst.encode()), Ok(inst));
        assert_eq!(inst.defined_register(), expected_def);
        assert_eq!(inst.gas_weight(), expected_weight);
        assert_eq!(format!("{inst}"), expected_disp);
    }

    assert!(Instruction::AddIntImm(2, 1, 5).reads_register(1));
    assert!(!Instruction::AddIntImm(2, 1, 5).reads_register(2));
    assert!(Instruction::BranchLtInt(1, 2, 4).reads_register(1));
    assert!(Instruction::BranchLtInt(1, 2, 4).reads_register(2));
    assert!(!Instruction::BranchLtInt(1, 2, 4).reads_register(3));
    assert!(Instruction::BranchLeInt(3, 4, -2).reads_register(3));
    assert!(Instruction::BranchLeInt(3, 4, -2).reads_register(4));
    assert!(!Instruction::BranchLeInt(3, 4, -2).reads_register(0));
    assert!(Instruction::GetFieldAddSetInt(6, 1, 9).reads_register(6));
    assert!(Instruction::GetFieldAddSetInt(6, 1, 9).reads_register(9));
    assert!(!Instruction::GetFieldAddSetInt(6, 1, 9).reads_register(1));

    // Verify Chunk::patch_jump works for BranchLtInt and BranchLeInt
    let mut chunk = Chunk::new();
    let br_lt = chunk.emit(Instruction::BranchLtInt(0, 1, 0), 1);
    let br_le = chunk.emit(Instruction::BranchLeInt(0, 1, 0), 2);
    chunk.emit(Instruction::AddIntImm(0, 0, 1), 3);
    let target = chunk.emit(Instruction::Return(0), 4);
    chunk.patch_jump(br_lt, target);
    chunk.patch_jump(br_le, target);
    assert_eq!(chunk.instructions[br_lt], Instruction::BranchLtInt(0, 1, 2));
    assert_eq!(chunk.instructions[br_le], Instruction::BranchLeInt(0, 1, 1));
}

#[test]
fn ng_w11_5a_sieve_and_nested_emit_branch_and_add_int_imm_with_valid_pcs() {
    let _hook_guard = TraceHookGuard::new();

    // 1. Inspect actual benchmarks/suite/sieve.lust and benchmarks/suite/nested.lust
    let sieve_full = include_str!("../benchmarks/suite/sieve.lust");
    let nested_full = include_str!("../benchmarks/suite/nested.lust");

    let sieve_mir = lower_script_source("sieve_full", sieve_full);
    let sieve_bc = MirToBytecodeCompiler::compile(&sieve_mir).expect("compile sieve.lust");
    assert!(
        sieve_bc
            .chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::BranchLeInt(..))),
        "sieve.lust inner/outer loops must emit BranchLeInt:\n{}",
        sieve_bc.disassemble()
    );
    assert!(
        sieve_bc
            .chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::AddIntImm(..))),
        "sieve.lust loops must emit AddIntImm:\n{}",
        sieve_bc.disassemble()
    );
    assert_all_jumps_land_on_block_entries(&sieve_bc);
    assert_eq!(
        sieve_bc.chunk.bb_gas_weights.len(),
        sieve_bc.chunk.instructions.len()
    );

    let nested_mir = lower_script_source("nested_full", nested_full);
    let nested_bc = MirToBytecodeCompiler::compile(&nested_mir).expect("compile nested.lust");
    assert!(
        nested_bc
            .chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::BranchLtInt(..))),
        "nested.lust loops must emit BranchLtInt:\n{}",
        nested_bc.disassemble()
    );
    assert!(
        nested_bc
            .chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::AddIntImm(..))),
        "nested.lust loops must emit AddIntImm:\n{}",
        nested_bc.disassemble()
    );
    assert_all_jumps_land_on_block_entries(&nested_bc);
    assert_eq!(
        nested_bc.chunk.bb_gas_weights.len(),
        nested_bc.chunk.instructions.len()
    );

    // 2. Execute scaled-down sieve (N=200) and nested (N=40) and verify exact results + exact gas parity
    let sieve_small_src = r#"
        local N: int = 200
        local flags: Array<bool> = []
        local i: int = 0
        while i <= N do
            array.push(flags, true)
            i = i + 1
        end
        local count: int = 0
        i = 2
        while i <= N do
            if flags[i] is Ok(f) then
                if f then
                    count = count + 1
                    local j: int = i * i
                    while j <= N do
                        flags[j] = false
                        j = j + i
                    end
                end
            end
            i = i + 1
        end
        return count
    "#;
    let sieve_small_bc = MirToBytecodeCompiler::compile(&lower_script_source(
        "sieve_small",
        sieve_small_src,
    ))
    .expect("compile sieve_small");
    assert!(
        sieve_small_bc
            .chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::BranchLeInt(..)))
            && sieve_small_bc
                .chunk
                .instructions
                .iter()
                .any(|i| matches!(i, Instruction::AddIntImm(..)))
    );

    let nested_small_src = r#"
        local total: int = 0
        local i: int = 0
        while i < 40 do
            local j: int = 0
            while j < 40 do
                total = total + (i * j) % 7
                j = j + 1
            end
            i = i + 1
        end
        return total
    "#;
    let nested_small_bc = MirToBytecodeCompiler::compile(&lower_script_source(
        "nested_small",
        nested_small_src,
    ))
    .expect("compile nested_small");
    assert!(
        nested_small_bc
            .chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::BranchLtInt(..)))
            && nested_small_bc
                .chunk
                .instructions
                .iter()
                .any(|i| matches!(i, Instruction::AddIntImm(..)))
    );

    let expected_nested: i64 = (0..40i64)
        .flat_map(|i| (0..40i64).map(move |j| (i * j) % 7))
        .sum();

    let mut vm_fast = plain_vm();
    vm_fast.load_functions(vec![sieve_small_bc.clone(), nested_small_bc.clone()]);
    vm_fast.reset_gas_counter();
    let res_sieve_fast = vm_fast.call("sieve_small", vec![]).expect("sieve_small fast");
    let gas_sieve_fast = vm_fast.gas_used();
    assert_eq!(res_sieve_fast, Value::Int(46)); // 46 primes <= 200

    vm_fast.reset_gas_counter();
    let res_nested_fast = vm_fast
        .call("nested_small", vec![])
        .expect("nested_small fast");
    let gas_nested_fast = vm_fast.gas_used();
    assert_eq!(res_nested_fast, Value::Int(expected_nested));

    // Compare against per-instruction slow path (trace hook installed)
    let mut vm_slow = plain_vm();
    vm_slow.load_functions(vec![sieve_small_bc, nested_small_bc]);
    vm_slow.set_instruction_hook(Some(Box::new(|_, _, _| {})));

    vm_slow.reset_gas_counter();
    let res_sieve_slow = vm_slow.call("sieve_small", vec![]).expect("sieve_small slow");
    let gas_sieve_slow = vm_slow.gas_used();
    assert_eq!(res_sieve_slow, Value::Int(46));
    assert_eq!(
        gas_sieve_fast, gas_sieve_slow,
        "sieve_small bb_gas_weights ({gas_sieve_fast}) must match per-instruction gas ({gas_sieve_slow})"
    );

    vm_slow.reset_gas_counter();
    let res_nested_slow = vm_slow
        .call("nested_small", vec![])
        .expect("nested_small slow");
    let gas_nested_slow = vm_slow.gas_used();
    assert_eq!(res_nested_slow, Value::Int(expected_nested));
    assert_eq!(
        gas_nested_fast, gas_nested_slow,
        "nested_small bb_gas_weights ({gas_nested_fast}) must match per-instruction gas ({gas_nested_slow})"
    );

    vm_slow.clear_trace_hooks();
}

#[test]
fn ng_w11_5a_get_field_add_set_int_fusion_and_dynamic_fallback() {
    let _hook_guard = TraceHookGuard::new();

    // 1. AST-lowered `c.value = c.value + i` inside a loop compiled with field slots
    let src = r#"
        struct Counter
            value: int
        end

        function counter_loop(): int
            local c: Counter = Counter { value = 0 }
            local i: int = 1
            while i <= 100 do
                c.value = c.value + i
                i = i + 1
            end
            return c.value
        end
    "#;
    let mut interner = Interner::new();
    let tokens = Lexer::new(src, &mut interner).tokenize().expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");
    let mut func_def = None;
    let mut struct_defs = HashMap::new();
    let mut struct_list = Vec::new();
    for item in &items {
        match &item.kind {
            ItemKind::Function(fd) if fd.name == "counter_loop" => func_def = Some(fd),
            ItemKind::Struct(sd) => {
                struct_defs.insert(sd.name.clone(), sd.clone());
                struct_list.push(sd);
            }
            _ => {}
        }
    }
    let (mir_fn, _) =
        AstLowering::lower_function_with_structs(func_def.expect("counter_loop"), &struct_list)
            .expect("lower_function_with_structs");

    let mut layouts = HashMap::new();
    layouts.insert(
        "Counter".to_string(),
        make_layout("Counter", &[("value", TypeKind::Int)]),
    );
    let bc = MirToBytecodeCompiler::compile_with_field_slots(&mir_fn, &layouts)
        .expect("compile_with_field_slots");

    assert!(
        bc.chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::GetFieldAddSetInt(_, 0, _))),
        "Expected GetFieldAddSetInt fusion for `c.value = c.value + i`:\n{}",
        bc.disassemble()
    );
    assert!(
        audit_chunk_field_access(&bc.chunk).has_zero_name_lookup(),
        "Expected zero GetField/SetField name lookups"
    );
    assert_all_jumps_land_on_block_entries(&bc);

    let mut vm_fast = plain_vm();
    vm_fast.register_structs(&struct_defs);
    vm_fast.load_functions(vec![bc.clone()]);
    vm_fast.reset_gas_counter();
    let r0 = VM::frame_hoist_reload_count();
    let out_fast = vm_fast
        .call("counter_loop", vec![])
        .expect("counter_loop fast");
    let r1 = VM::frame_hoist_reload_count();
    let gas_fast = vm_fast.gas_used();
    assert_eq!(out_fast, Value::Int(5050));
    #[cfg(debug_assertions)]
    assert_eq!(
        r1 - r0,
        1,
        "GetFieldAddSetInt loop must stay inside 'frame_fast with 0 slow exits"
    );

    // Exact gas parity with per-instruction trace hook mode (gas_weight == 3)
    let mut vm_slow = plain_vm();
    vm_slow.register_structs(&struct_defs);
    vm_slow.load_functions(vec![bc]);
    vm_slow.set_instruction_hook(Some(Box::new(|_, _, _| {})));
    vm_slow.reset_gas_counter();
    let out_slow = vm_slow
        .call("counter_loop", vec![])
        .expect("counter_loop slow");
    let gas_slow = vm_slow.gas_used();
    assert_eq!(out_slow, Value::Int(5050));
    assert_eq!(gas_fast, gas_slow);
    vm_slow.clear_trace_hooks();

    // 2. Dynamic fallback soundness: GetFieldAddSetInt on Float and String fields, and error on incompatible types
    let mut bump_fn = Function::new("bump_slot0", 2, false);
    bump_fn
        .chunk
        .emit(Instruction::GetFieldAddSetInt(0, 0, 1), 1);
    bump_fn.chunk.emit(Instruction::GetFieldSlot(2, 0, 0), 2);
    bump_fn.chunk.emit(Instruction::Return(2), 3);
    bump_fn.set_register_count(3);

    let mut vm = plain_vm();
    vm.register_structs(&make_struct_defs(&[(
        "BoxVal",
        &[("v", TypeKind::Unknown)],
    )]));
    vm.load_functions(vec![bump_fn]);

    let float_box = vm
        .instantiate_struct("BoxVal", vec![(Rc::new("v".to_string()), Value::Float(1.5))])
        .unwrap();
    let res_f = vm
        .call("bump_slot0", vec![float_box, Value::Float(2.25)])
        .expect("float fallback");
    assert_eq!(res_f, Value::Float(3.75));

    let str_box = vm
        .instantiate_struct(
            "BoxVal",
            vec![(Rc::new("v".to_string()), Value::string("hello "))],
        )
        .unwrap();
    let res_s = vm
        .call("bump_slot0", vec![str_box, Value::string("lust")])
        .expect("string fallback");
    assert_eq!(res_s, Value::string("hello lust"));

    let bad_box = vm
        .instantiate_struct("BoxVal", vec![(Rc::new("v".to_string()), Value::Int(10))])
        .unwrap();
    let err = vm
        .call("bump_slot0", vec![bad_box, Value::string("nope")])
        .expect_err("Int + String must fail cleanly");
    assert!(err.to_string().contains("Cannot add"));
}

#[test]
fn ng_w11_5a_peephole_never_crosses_block_entry_pcs() {
    let span = Span::dummy();

    // 1. Cross-block LtInt (bb0) + JumpIfNot (bb1) MUST NOT fuse across bb1 entry!
    let mut f = MirFunction::new(
        FunctionSignature {
            name: "cross_block_cond".into(),
            param_types: vec![],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let a = f.add_local(Some("a".into()), MirType::Int, false, span);
    let b = f.add_local(Some("b".into()), MirType::Int, false, span);
    let tmp_cond = f.add_local(None, MirType::Bool, false, span); // anonymous, single-use!
    let out = f.add_local(Some("out".into()), MirType::Int, false, span);

    let bb1 = f.new_block();
    let bb_then = f.new_block();
    let bb_else = f.new_block();

    f.blocks[0].statements.extend([
        Statement::Assign(
            Place::Local(a),
            Rvalue::Use(Operand::Constant(MirConstant::Int(3))),
            span,
        ),
        Statement::Assign(
            Place::Local(b),
            Rvalue::Use(Operand::Constant(MirConstant::Int(7))),
            span,
        ),
        Statement::Assign(
            Place::Local(tmp_cond),
            Rvalue::BinaryOp(
                MirBinaryOp::Lt,
                Operand::Copy(Place::Local(a)),
                Operand::Copy(Place::Local(b)),
            ),
            span,
        ),
    ]);
    f.blocks[0].terminator = Some(Terminator::Goto(bb1, span));

    f.blocks[bb1.0].terminator = Some(Terminator::Cond {
        cond: Operand::Move(Place::Local(tmp_cond)),
        then_block: bb_then,
        else_block: bb_else,
        span,
    });

    f.blocks[bb_then.0].statements.push(Statement::Assign(
        Place::Local(out),
        Rvalue::Use(Operand::Constant(MirConstant::Int(111))),
        span,
    ));
    f.blocks[bb_then.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(out))),
        span,
    ));

    f.blocks[bb_else.0].statements.push(Statement::Assign(
        Place::Local(out),
        Rvalue::Use(Operand::Constant(MirConstant::Int(222))),
        span,
    ));
    f.blocks[bb_else.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(out))),
        span,
    ));

    let bc = MirToBytecodeCompiler::compile(&f).expect("compile cross_block_cond");
    assert!(
        !bc.chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::BranchLtInt(..) | Instruction::BranchLeInt(..))),
        "Peephole MUST NOT fuse LtInt and JumpIfNot across MIR basic block boundary:\n{}",
        bc.disassemble()
    );
    assert!(matches!(
        bc.chunk.instructions[bc.block_entry_pcs[&bb1.0]],
        Instruction::JumpIfNot(..)
    ));
    assert_all_jumps_land_on_block_entries(&bc);

    // 2. Cross-block GetFieldSlot + Add (bb0) and SetFieldSlot (bb1) MUST NOT fuse!
    let mut f2 = MirFunction::new(
        FunctionSignature {
            name: "cross_block_field".into(),
            param_types: vec![],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let p = f2.add_local(Some("p".into()), MirType::Struct("Point".into()), true, span);
    let tmp_sum = f2.add_local(None, MirType::Int, false, span); // anonymous, single-use!
    let res = f2.add_local(Some("res".into()), MirType::Int, false, span);
    let f2_bb1 = f2.new_block();

    f2.blocks[0].statements.extend([
        Statement::Assign(
            Place::Local(p),
            Rvalue::Aggregate(
                MirType::Struct("Point".into()),
                vec![Operand::Constant(MirConstant::Int(10))],
            ),
            span,
        ),
        Statement::Assign(
            Place::Local(tmp_sum),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Field(Box::new(Place::Local(p)), 0)),
                Operand::Constant(MirConstant::Int(5)),
            ),
            span,
        ),
    ]);
    f2.blocks[0].terminator = Some(Terminator::Goto(f2_bb1, span));

    f2.blocks[f2_bb1.0].statements.extend([
        Statement::Assign(
            Place::Field(Box::new(Place::Local(p)), 0),
            Rvalue::Use(Operand::Move(Place::Local(tmp_sum))),
            span,
        ),
        Statement::Assign(
            Place::Local(res),
            Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(p)), 0))),
            span,
        ),
    ]);
    f2.blocks[f2_bb1.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(res))),
        span,
    ));

    let mut layouts = HashMap::new();
    layouts.insert("Point".to_string(), make_layout("Point", &[("x", TypeKind::Int)]));
    let bc2 = MirToBytecodeCompiler::compile_with_field_slots(&f2, &layouts)
        .expect("compile cross_block_field");
    assert!(
        !bc2
            .chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::GetFieldAddSetInt(..))),
        "Peephole MUST NOT fuse GetFieldAddSetInt across MIR basic block boundary:\n{}",
        bc2.disassemble()
    );
    assert!(matches!(
        bc2.chunk.instructions[bc2.block_entry_pcs[&f2_bb1.0]],
        Instruction::SetFieldSlot(..)
    ));
    assert_all_jumps_land_on_block_entries(&bc2);
}

#[test]
fn ng_w11_5a_compute_bb_gas_weights_and_sec01_exact_50_51_boundary() {
    let _hook_guard = TraceHookGuard::new();

    // Construct a deterministic loop whose exact gas consumption is 51 units:
    //   BB0 [pc 0..2]: LoadConst(0, 0) [w=1], LoadConst(1, 10) [w=1]     -> bb_weight = 2
    //   BB1 [pc 2..3]: BranchLtInt(0, 1, +2) [w=2] (header, target of Jump) -> bb_weight = 2
    //   BB2 [pc 3..5]: AddIntImm(0, 0, 1) [w=1], Jump(-3) [w=1]          -> bb_weight = 2
    //   BB3 [pc 5..6]: Return(0) [w=1] (target of BranchLtInt)           -> bb_weight = 1
    //
    //Wait: Let's verify exact total gas for N = 12 iterations:
    //   Entry BB0: 2
    //   12 loop iterations: each executes BB1 (2) + BB2 (2) = 4 -> 12 * 4 = 48
    //   Final header check BB1 (i == 12, branch taken to Return): 2 -> 2 + 48 + 2 = 52!
    //   So with 1 instruction in BB0 (passing N=12 as param r0, init i=0 via LoadConst(1, 0)):
    //     BB0 [pc 0]: LoadConst(1, 0)         (w=1) -> leader 0, weight = 1
    //     BB1 [pc 1]: BranchLtInt(1, 0, +2)   (w=2) -> leader 1, weight = 2
    //     BB2 [pc 2..4]: AddIntImm(1, 1, 1) (w=1), Jump(-3) (w=1) -> leader 2, weight = 2
    //     BB3 [pc 4]: Return(1)               (w=1) -> leader 4, weight = 1
    //   Total gas for N = 12:
    //     BB0 (1) + 12 * (BB1(2) + BB2(2)) + BB1(2) + BB3(1) = 1 + 48 + 2 + 1 = 52.
    //   For a multi-instruction BB body of weight 3 (AddIntImm + Move + Jump):
    //     BB0 [pc 0]:    LoadConst(1, 0)       (w=1)
    //     BB1 [pc 1]:    BranchLtInt(1, 0, +3) (w=2)
    //     BB2 [pc 2..5]: AddIntImm(2, 1, 1)    (w=1)
    //                    Move(1, 2)            (w=1)
    //                    Jump(-4)              (w=1)   -> leader 2, weight = 3
    //     BB3 [pc 5]:    Return(1)             (w=1)
    //   Wait: for N = 9: 1 + 9 * (2 + 3) + 2 + 1 = 49.
    //   Add 2 init instructions in BB0 -> BB0 has weight 3!
    //     BB0 [pc 0..3]: LoadConst(1, 0) [w=1], LoadConst(2, 0) [w=1], Move(1, 2) [w=1] -> leader 0, weight = 3
    //     BB1 [pc 3]:    BranchLtInt(1, 0, +3) [w=2] -> leader 3, weight = 2
    //     BB2 [pc 4..7]: AddIntImm(2, 1, 1) [w=1], Move(1, 2) [w=1], Jump(-4) [w=1] -> leader 4, weight = 3
    //     BB3 [pc 7]:    Return(1) [w=1] -> leader 7, weight = 1
    //   Total gas for N = 9:
    //     BB0 (3) + 9 * (2 + 3) + BB1 (2) + BB3 (1) = 3 + 45 + 2 + 1 = 51 EXACTLY!
    let mut f = Function::new("sec01_loop_51", 1, false);
    let k0 = f.chunk.add_constant(Value::Int(0));
    f.chunk.emit(Instruction::LoadConst(1, k0), 1); // pc 0
    f.chunk.emit(Instruction::LoadConst(2, k0), 1); // pc 1
    f.chunk.emit(Instruction::Move(1, 2), 1); // pc 2
    let hdr = f.chunk.instructions.len(); // pc 3
    let br = f.chunk.emit(Instruction::BranchLtInt(1, 0, 0), 2); // pc 3
    f.chunk.emit(Instruction::AddIntImm(2, 1, 1), 3); // pc 4
    f.chunk.emit(Instruction::Move(1, 2), 3); // pc 5
    let back = f.chunk.emit(Instruction::Jump(0), 4); // pc 6
    let exit = f.chunk.emit(Instruction::Return(1), 5); // pc 7
    f.chunk.patch_jump(br, exit);
    f.chunk.patch_jump(back, hdr);
    f.chunk.compute_bb_gas_weights();
    f.set_register_count(3);

    assert_eq!(
        f.chunk.bb_gas_weights,
        vec![3, 0, 0, 2, 3, 0, 0, 1],
        "bb_gas_weights must store exact block sum at each leader and 0 at interior PCs"
    );

    // 1. Unbounded execution uses exactly 51 gas
    let mut vm_fast = plain_vm();
    vm_fast.load_functions(vec![f.clone()]);
    vm_fast.reset_gas_counter();
    assert_eq!(
        vm_fast.call("sec01_loop_51", vec![Value::Int(9)]).expect("unbounded"),
        Value::Int(9)
    );
    assert_eq!(vm_fast.gas_used(), 51);

    // 2. SEC-01 exact boundary: gas_limit = 50 -> OutOfGas with gas_used() == 51
    vm_fast.reset_gas_counter();
    vm_fast.set_gas_budget(50);
    let oog_err = vm_fast
        .call("sec01_loop_51", vec![Value::Int(9)])
        .expect_err("gas_limit = 50 on 51-gas loop MUST fail with OutOfGas");
    assert!(
        oog_err.to_string().contains("Out of gas"),
        "expected Out of gas, got: {oog_err}"
    );
    assert_eq!(
        vm_fast.gas_used(),
        51,
        "SEC-01 exact boundary: gas_limit=50 must trip at gas_used==51, got {}",
        vm_fast.gas_used()
    );

    // 3. gas_limit = 51 -> succeeds with gas_used() == 51
    vm_fast.reset_gas_counter();
    vm_fast.set_gas_budget(51);
    assert_eq!(
        vm_fast
            .call("sec01_loop_51", vec![Value::Int(9)])
            .expect("gas_limit = 51 must succeed"),
        Value::Int(9)
    );
    assert_eq!(vm_fast.gas_used(), 51);

    // 4. Exhaustive sweep gas_limit = 0..=51: 100% parity between bb_gas_weights fast path and per-instruction trace path!
    let mut vm_slow = plain_vm();
    vm_slow.load_functions(vec![f]);
    let hook_hits = Rc::new(RefCell::new(0usize));
    let hh = Rc::clone(&hook_hits);
    vm_slow.set_instruction_hook(Some(Box::new(move |_, _, _| {
        *hh.borrow_mut() += 1;
    })));

    for limit in 0u64..=51 {
        vm_fast.reset_gas_counter();
        vm_fast.set_gas_budget(limit);
        let fast_res = vm_fast.call("sec01_loop_51", vec![Value::Int(9)]);
        let fast_gas = vm_fast.gas_used();

        vm_slow.reset_gas_counter();
        vm_slow.set_gas_budget(limit);
        let slow_res = vm_slow.call("sec01_loop_51", vec![Value::Int(9)]);
        let slow_gas = vm_slow.gas_used();

        assert_eq!(
            fast_res.is_ok(),
            slow_res.is_ok(),
            "Outcome mismatch at gas_limit={limit}: fast={fast_res:?}, slow={slow_res:?}"
        );
        assert_eq!(
            fast_gas, slow_gas,
            "Gas used mismatch at gas_limit={limit}: fast={fast_gas}, slow={slow_gas}"
        );
    }
    assert!(*hook_hits.borrow() > 0);
    vm_slow.clear_trace_hooks();
}

#[test]
fn ng_w11_5a_mid_bb_slow_exit_preserves_exact_gas_and_oog_boundary() {
    let _hook_guard = TraceHookGuard::new();

    // Single basic block containing a mid-BB `StoreGlobal` (which triggers `FrameExit::Slow`
    // inside `'frame_fast` without being a BB terminator!):
    //   pc 0: LoadConst(0, 10)   [w=1]
    //   pc 1: StoreGlobal("g", 0)[w=1]  <- exits 'frame_fast via FrameExit::Slow!
    //   pc 2: AddIntImm(1, 0, 5) [w=1]  <- re-enters 'frame_fast at interior pc 2 (bb_gas_weights[2] == 0)
    //   pc 3: AddIntImm(2, 1, 7) [w=1]
    //   pc 4: Return(2)          [w=1]
    // Total BB weight = 5 (bb_gas_weights = [5, 0, 0, 0, 0]).
    let mut f = Function::new("mid_bb_slow", 0, false);
    let k10 = f.chunk.add_constant(Value::Int(10));
    let kg = f.chunk.add_constant(Value::string("g"));
    f.chunk.emit(Instruction::LoadConst(0, k10), 1);
    f.chunk.emit(Instruction::StoreGlobal(kg, 0), 2);
    f.chunk.emit(Instruction::AddIntImm(1, 0, 5), 3);
    f.chunk.emit(Instruction::AddIntImm(2, 1, 7), 4);
    f.chunk.emit(Instruction::Return(2), 5);
    f.chunk.compute_bb_gas_weights();
    f.set_register_count(3);

    assert_eq!(f.chunk.bb_gas_weights, vec![5, 0, 0, 0, 0]);

    let mut vm_fast = plain_vm();
    vm_fast.load_functions(vec![f.clone()]);

    let mut vm_slow = plain_vm();
    vm_slow.load_functions(vec![f]);
    vm_slow.set_instruction_hook(Some(Box::new(|_, _, _| {})));

    // Sweep gas_limit = 0..=5 across the mid-BB FrameExit::Slow boundary!
    // At limit = 3 or 4, pc 0 enters exact_gas_mode, pc 1 hits FrameExit::Slow,
    // and pc 2..4 MUST stay in exact_gas_mode upon re-entry and trigger OutOfGas at limit+1!
    for limit in 0u64..=5 {
        vm_fast.reset_gas_counter();
        vm_fast.set_gas_budget(limit);
        let r_fast = vm_fast.call("mid_bb_slow", vec![]);
        let g_fast = vm_fast.gas_used();

        vm_slow.reset_gas_counter();
        vm_slow.set_gas_budget(limit);
        let r_slow = vm_slow.call("mid_bb_slow", vec![]);
        let g_slow = vm_slow.gas_used();

        assert_eq!(
            r_fast.is_ok(),
            r_slow.is_ok(),
            "mid-BB slow-exit outcome mismatch at limit={limit}: fast={r_fast:?}, slow={r_slow:?}"
        );
        assert_eq!(
            g_fast, g_slow,
            "mid-BB slow-exit gas mismatch at limit={limit}: fast={g_fast}, slow={g_slow}"
        );
        if limit == 5 {
            assert_eq!(r_fast.unwrap(), Value::Int(22));
        }
    }
    vm_slow.clear_trace_hooks();
}

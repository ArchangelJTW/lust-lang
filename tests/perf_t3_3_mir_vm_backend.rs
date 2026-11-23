#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::rc::Rc;

use lust::ast::{Span, StructDef, StructField, FieldOwnership, Type, TypeKind, Visibility};
use lust::bytecode::{
    FieldStorage, Function, Instruction, LustMap, NativeCallResult, StructLayout, Value, ValueKey,
    native_fn,
};
use lust::hashbrown::HashMap;
use lust::mir::ir::{
    BasicBlock, BlockId, FunctionSignature, MirBinaryOp, MirConstant, MirFunction, MirType,
    MirUnaryOp, Operand, Place, Rvalue, Statement, Terminator,
};
use lust::mir::vm_backend::MirToBytecodeCompiler;
use lust::vm::VM;

fn make_point_layout() -> Rc<StructLayout> {
    let span = Span::dummy();
    Rc::new(StructLayout::new(
        "Point".to_string(),
        vec![Rc::new("x".to_string()), Rc::new("y".to_string())],
        vec![FieldStorage::Strong, FieldStorage::Strong],
        vec![Type::new(TypeKind::Int, span), Type::new(TypeKind::Int, span)],
        vec![None, None],
    ))
}

fn make_point_struct_defs() -> HashMap<String, StructDef> {
    let span = Span::dummy();
    let mut defs = HashMap::new();
    defs.insert(
        "Point".to_string(),
        StructDef {
            name: "Point".to_string(),
            type_params: vec![],
            trait_bounds: vec![],
            fields: vec![
                StructField {
                    name: "x".to_string(),
                    ty: Type::new(TypeKind::Int, span),
                    ownership: FieldOwnership::Strong,
                    weak_target: None,
                    visibility: Visibility::Public,
                },
                StructField {
                    name: "y".to_string(),
                    ty: Type::new(TypeKind::Int, span),
                    ownership: FieldOwnership::Strong,
                    weak_target: None,
                    visibility: Visibility::Public,
                },
            ],
            visibility: Visibility::Public,
            doc: None,
        },
    );
    defs
}

fn assert_no_fallthrough_jump0(bc: &Function) {
    for (pc, instr) in bc.chunk.instructions.iter().enumerate() {
        assert!(
            !matches!(instr, Instruction::Jump(0)),
            "Unexpected fallthrough Jump(0) at pc {pc} in '{}':\n{}",
            bc.name,
            bc.disassemble()
        );
    }
}

fn assert_all_jumps_land_on_block_entries(bc: &Function) {
    let valid_entries: lust::hashbrown::HashSet<usize> =
        bc.block_entry_pcs.values().copied().collect();
    for (pc, instr) in bc.chunk.instructions.iter().enumerate() {
        let offset_opt = match *instr {
            Instruction::Jump(off)
            | Instruction::JumpIf(_, off)
            | Instruction::JumpIfNot(_, off) => Some(off),
            _ => None,
        };
        if let Some(off) = offset_opt {
            let target = (pc as isize + 1 + off as isize) as usize;
            assert!(
                valid_entries.contains(&target),
                "Jump at pc {pc} ({instr}) lands at {target}, which is not a valid block entry PC in {:?}:\n{}",
                bc.block_entry_pcs,
                bc.disassemble()
            );
        }
    }
}

/// Canonical PERF-T3-3 RED->GREEN test:
/// Compiles a typed integer loop via `MirToBytecodeCompiler` and asserts:
/// (a) emits `AddInt`/`LtInt`/`LeInt` instead of generic `Add`/`Lt`/`Le`,
/// (b) emits zero fallthrough `Jump(0)`,
/// (c) `block_entry_pcs` and `terminator_pcs` are exact for every block,
/// (d) execution produces exact results.
#[test]
fn test_perf_t3_3_mir_vm_typed_opcodes_and_no_fallthrough_jump0() {
    let span = Span::dummy();
    // Locally initialized `n = 10`, `acc = 0`, `i = 1`, `bonus = 0`:
    // proves real typed loops emit AddInt/LtInt/LeInt cleanly without trusting unchecked parameter types.
    let mut f = MirFunction::new(
        FunctionSignature {
            name: "typed_int_loop".into(),
            param_types: vec![],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let n = f.add_local(Some("n".into()), MirType::Int, false, span);
    let acc = f.add_local(Some("acc".into()), MirType::Int, true, span);
    let i = f.add_local(Some("i".into()), MirType::Int, true, span);
    let cond_le = f.add_local(Some("cond_le".into()), MirType::Bool, false, span);
    let cond_lt = f.add_local(Some("cond_lt".into()), MirType::Bool, false, span);
    let bonus = f.add_local(Some("bonus".into()), MirType::Int, true, span);

    let bb_hdr = f.new_block(); // bb1
    let bb_body = f.new_block(); // bb2
    let bb_then = f.new_block(); // bb3
    let bb_step = f.new_block(); // bb4
    let bb_exit = f.new_block(); // bb5

    // bb0: n = 10; acc = 0; i = 1; bonus = 0; goto bb_hdr (fallthrough!)
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(n),
        Rvalue::Use(Operand::Constant(MirConstant::Int(10))),
        span,
    ));
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        span,
    ));
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::Use(Operand::Constant(MirConstant::Int(1))),
        span,
    ));
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(bonus),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        span,
    ));
    f.blocks[0].terminator = Some(Terminator::Goto(bb_hdr, span));

    // bb1 (bb_hdr): cond_le = i <= n; if cond_le then bb_body (fallthrough!) else bb_exit
    f.blocks[bb_hdr.0].statements.push(Statement::Assign(
        Place::Local(cond_le),
        Rvalue::BinaryOp(
            MirBinaryOp::LtEq,
            Operand::Copy(Place::Local(i)),
            Operand::Copy(Place::Local(n)),
        ),
        span,
    ));
    f.blocks[bb_hdr.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond_le)),
        then_block: bb_body,
        else_block: bb_exit,
        span,
    });

    // bb2 (bb_body): cond_lt = i < 5; if cond_lt then bb_then (fallthrough!) else bb_step
    f.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(cond_lt),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(i)),
            Operand::Constant(MirConstant::Int(5)),
        ),
        span,
    ));
    f.blocks[bb_body.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond_lt)),
        then_block: bb_then,
        else_block: bb_step,
        span,
    });

    // bb3 (bb_then): bonus = bonus + 10; goto bb_step (fallthrough!)
    f.blocks[bb_then.0].statements.push(Statement::Assign(
        Place::Local(bonus),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(bonus)),
            Operand::Constant(MirConstant::Int(10)),
        ),
        span,
    ));
    f.blocks[bb_then.0].terminator = Some(Terminator::Goto(bb_step, span));

    // bb4 (bb_step): acc = acc + i; i = i + 1; goto bb_hdr (backedge)
    f.blocks[bb_step.0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(acc)),
            Operand::Copy(Place::Local(i)),
        ),
        span,
    ));
    f.blocks[bb_step.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        span,
    ));
    f.blocks[bb_step.0].terminator = Some(Terminator::Goto(bb_hdr, span));

    // bb5 (bb_exit): acc = acc + bonus; return acc
    f.blocks[bb_exit.0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(acc)),
            Operand::Copy(Place::Local(bonus)),
        ),
        span,
    ));
    f.blocks[bb_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(acc))),
        span,
    ));

    let bc = MirToBytecodeCompiler::compile(&f).expect("compile typed_int_loop");
    let insns = &bc.chunk.instructions;

    // (a) Typed Int opcodes emitted instead of generic Add/Lt/Le
    assert!(
        insns.iter().any(|i| matches!(i, Instruction::AddInt(..))),
        "Expected AddInt in compiled loop:\n{}",
        bc.disassemble()
    );
    assert!(
        insns.iter().any(|i| matches!(i, Instruction::LtInt(..))),
        "Expected LtInt in compiled loop:\n{}",
        bc.disassemble()
    );
    assert!(
        insns.iter().any(|i| matches!(i, Instruction::LeInt(..))),
        "Expected LeInt in compiled loop:\n{}",
        bc.disassemble()
    );
    assert!(
        !insns.iter().any(|i| matches!(
            i,
            Instruction::Add(..) | Instruction::Lt(..) | Instruction::Le(..)
        )),
        "Generic Add/Lt/Le must not be emitted for proven Int operands:\n{}",
        bc.disassemble()
    );

    // (b) Zero fallthrough Jump(0)
    assert_no_fallthrough_jump0(&bc);
    assert_all_jumps_land_on_block_entries(&bc);

    // (c) Exact block_entry_pcs and terminator_pcs for every block
    assert_eq!(bc.block_entry_pcs.len(), f.blocks.len());
    assert_eq!(bc.terminator_pcs.len(), f.blocks.len());
    for b in &f.blocks {
        let entry_pc = bc.block_entry_pcs[&b.id.0];
        let term_pc = bc.terminator_pcs[&b.id.0];
        assert!(
            entry_pc <= term_pc && term_pc <= insns.len(),
            "Invalid PCs for block {:?}: entry={entry_pc}, term={term_pc}, len={}",
            b.id,
            insns.len()
        );
    }
    // bb0 has 4 LoadConst (plus any NG-W9-5 LICM-hoisted loop constants) and elided Goto -> bb1 entry equals bb0 terminator PC.
    assert_eq!(bc.block_entry_pcs[&0], 0);
    assert!(bc.terminator_pcs[&0] >= 4);
    assert_eq!(bc.block_entry_pcs[&bb_hdr.0], bc.terminator_pcs[&0]);
    assert!(matches!(insns[bc.block_entry_pcs[&bb_hdr.0]], Instruction::LeInt(..)));
    assert!(matches!(
        insns[bc.terminator_pcs[&bb_hdr.0]],
        Instruction::JumpIfNot(..)
    ));
    // Verify backedge in bb_step jumps directly to bb_hdr entry PC
    let step_term_pc = bc.terminator_pcs[&bb_step.0];
    let Instruction::Jump(back_off) = insns[step_term_pc] else {
        panic!("Expected backedge Jump at bb_step terminator");
    };
    let back_target = (step_term_pc as isize + 1 + back_off as isize) as usize;
    assert_eq!(back_target, bc.block_entry_pcs[&bb_hdr.0]);

    // (d) Exact execution result
    let mut vm = VM::new();
    vm.load_functions(vec![bc]);
    // sum(1..=10) = 55; i < 5 triggers for i = 1,2,3,4 -> bonus = 40; total = 95
    assert_eq!(
        vm.call("typed_int_loop", vec![]).expect("run typed_int_loop"),
        Value::Int(95)
    );
}

#[test]
fn test_perf_t3_3_typed_int_and_float_full_op_matrix() {
    let span = Span::dummy();

    // 1. Int ops on locally-initialized Int locals: Sub, Mul, Div, Mod, FloorDiv, Pow, Neg, Eq, Ne, Gt, Ge
    let mut fi = MirFunction::new(
        FunctionSignature {
            name: "int_ops".into(),
            param_types: vec![],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let a = fi.add_local(Some("a".into()), MirType::Int, false, span);
    let b = fi.add_local(Some("b".into()), MirType::Int, false, span);
    let t1 = fi.add_local(None, MirType::Int, false, span);
    let t2 = fi.add_local(None, MirType::Int, false, span);
    let t3 = fi.add_local(None, MirType::Int, false, span);
    let t4 = fi.add_local(None, MirType::Int, false, span);
    let t5 = fi.add_local(None, MirType::Int, false, span);
    let t6 = fi.add_local(None, MirType::Int, false, span);
    let t7 = fi.add_local(None, MirType::Int, false, span);
    let c_eq = fi.add_local(None, MirType::Bool, false, span);
    let c_ne = fi.add_local(None, MirType::Bool, false, span);
    let c_gt = fi.add_local(None, MirType::Bool, false, span);
    let c_ge = fi.add_local(None, MirType::Bool, false, span);
    let out = fi.add_local(None, MirType::Int, false, span);

    fi.blocks[0].statements.extend([
        Statement::Assign(Place::Local(a), Rvalue::Use(Operand::Constant(MirConstant::Int(11))), span),
        Statement::Assign(Place::Local(b), Rvalue::Use(Operand::Constant(MirConstant::Int(4))), span),
        Statement::Assign(
            Place::Local(t1),
            Rvalue::BinaryOp(MirBinaryOp::Sub, Operand::Copy(Place::Local(a)), Operand::Copy(Place::Local(b))),
            span,
        ),
        Statement::Assign(
            Place::Local(t2),
            Rvalue::BinaryOp(MirBinaryOp::Mul, Operand::Copy(Place::Local(t1)), Operand::Constant(MirConstant::Int(3))),
            span,
        ),
        Statement::Assign(
            Place::Local(t3),
            Rvalue::BinaryOp(MirBinaryOp::Div, Operand::Copy(Place::Local(t2)), Operand::Constant(MirConstant::Int(2))),
            span,
        ),
        Statement::Assign(
            Place::Local(t4),
            Rvalue::BinaryOp(MirBinaryOp::Mod, Operand::Copy(Place::Local(a)), Operand::Copy(Place::Local(b))),
            span,
        ),
        Statement::Assign(
            Place::Local(t5),
            Rvalue::BinaryOp(MirBinaryOp::FloorDiv, Operand::Constant(MirConstant::Int(-7)), Operand::Constant(MirConstant::Int(2))),
            span,
        ),
        Statement::Assign(
            Place::Local(t6),
            Rvalue::BinaryOp(MirBinaryOp::Pow, Operand::Constant(MirConstant::Int(2)), Operand::Constant(MirConstant::Int(4))),
            span,
        ),
        Statement::Assign(
            Place::Local(t7),
            Rvalue::UnaryOp(MirUnaryOp::Neg, Operand::Copy(Place::Local(t5))),
            span,
        ),
        Statement::Assign(
            Place::Local(c_eq),
            Rvalue::BinaryOp(MirBinaryOp::Eq, Operand::Copy(Place::Local(a)), Operand::Copy(Place::Local(b))),
            span,
        ),
        Statement::Assign(
            Place::Local(c_ne),
            Rvalue::BinaryOp(MirBinaryOp::NotEq, Operand::Copy(Place::Local(a)), Operand::Copy(Place::Local(b))),
            span,
        ),
        Statement::Assign(
            Place::Local(c_gt),
            Rvalue::BinaryOp(MirBinaryOp::Gt, Operand::Copy(Place::Local(a)), Operand::Copy(Place::Local(b))),
            span,
        ),
        Statement::Assign(
            Place::Local(c_ge),
            Rvalue::BinaryOp(MirBinaryOp::GtEq, Operand::Copy(Place::Local(a)), Operand::Copy(Place::Local(b))),
            span,
        ),
        Statement::Assign(
            Place::Local(out),
            Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(t3)), Operand::Copy(Place::Local(t4))),
            span,
        ),
        Statement::Assign(
            Place::Local(out),
            Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(out)), Operand::Copy(Place::Local(t6))),
            span,
        ),
        Statement::Assign(
            Place::Local(out),
            Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(out)), Operand::Copy(Place::Local(t7))),
            span,
        ),
    ]);
    fi.blocks[0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(out))), span));

    let bci = MirToBytecodeCompiler::compile(&fi).expect("compile int_ops");
    let ins = &bci.chunk.instructions;
    assert!(ins.iter().any(|i| matches!(i, Instruction::SubInt(..))));
    assert!(ins.iter().any(|i| matches!(i, Instruction::MulInt(..))));
    assert!(ins.iter().any(|i| matches!(i, Instruction::DivInt(..))));
    assert!(ins.iter().any(|i| matches!(i, Instruction::ModInt(..))));
    assert!(ins.iter().any(|i| matches!(i, Instruction::FloorDivInt(..))));
    assert!(ins.iter().any(|i| matches!(i, Instruction::PowInt(..))));
    assert!(ins.iter().any(|i| matches!(i, Instruction::NegInt(..))));
    assert!(ins.iter().any(|i| matches!(i, Instruction::EqInt(..))));
    assert!(ins.iter().any(|i| matches!(i, Instruction::NeInt(..))));
    assert!(ins.iter().any(|i| matches!(i, Instruction::GtInt(..))));
    assert!(ins.iter().any(|i| matches!(i, Instruction::GeInt(..))));

    // 2. Float ops on locally-initialized Float locals + IEEE-754 NaN & signed-zero semantics
    let build_float_ops = |name: &str, init_a: f64, init_b: f64| {
        let mut ff = MirFunction::new(
            FunctionSignature {
                name: name.into(),
                param_types: vec![],
                return_type: MirType::Bool,
                is_public: true,
                effects: None,
            },
            span,
        );
        let fa = ff.add_local(Some("fa".into()), MirType::Float, false, span);
        let fb = ff.add_local(Some("fb".into()), MirType::Float, false, span);
        let f1 = ff.add_local(None, MirType::Float, false, span);
        let f2 = ff.add_local(None, MirType::Float, false, span);
        let f3 = ff.add_local(None, MirType::Float, false, span);
        let f4 = ff.add_local(None, MirType::Float, false, span);
        let f5 = ff.add_local(None, MirType::Float, false, span);
        let feq = ff.add_local(None, MirType::Bool, false, span);
        let fne = ff.add_local(None, MirType::Bool, false, span);
        let flt = ff.add_local(None, MirType::Bool, false, span);
        let fle = ff.add_local(None, MirType::Bool, false, span);
        let fgt = ff.add_local(None, MirType::Bool, false, span);
        let fge = ff.add_local(None, MirType::Bool, false, span);

        ff.blocks[0].statements.extend([
            Statement::Assign(Place::Local(fa), Rvalue::Use(Operand::Constant(MirConstant::Float(init_a))), span),
            Statement::Assign(Place::Local(fb), Rvalue::Use(Operand::Constant(MirConstant::Float(init_b))), span),
            Statement::Assign(
                Place::Local(f1),
                Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(fa)), Operand::Copy(Place::Local(fb))),
                span,
            ),
            Statement::Assign(
                Place::Local(f2),
                Rvalue::BinaryOp(MirBinaryOp::Sub, Operand::Copy(Place::Local(f1)), Operand::Constant(MirConstant::Float(1.0))),
                span,
            ),
            Statement::Assign(
                Place::Local(f3),
                Rvalue::BinaryOp(MirBinaryOp::Mul, Operand::Copy(Place::Local(f2)), Operand::Constant(MirConstant::Float(2.0))),
                span,
            ),
            Statement::Assign(
                Place::Local(f4),
                Rvalue::BinaryOp(MirBinaryOp::Div, Operand::Copy(Place::Local(f3)), Operand::Constant(MirConstant::Float(2.0))),
                span,
            ),
            Statement::Assign(
                Place::Local(f5),
                Rvalue::UnaryOp(MirUnaryOp::Neg, Operand::Copy(Place::Local(f4))),
                span,
            ),
            Statement::Assign(
                Place::Local(flt),
                Rvalue::BinaryOp(MirBinaryOp::Lt, Operand::Copy(Place::Local(fa)), Operand::Copy(Place::Local(fb))),
                span,
            ),
            Statement::Assign(
                Place::Local(fle),
                Rvalue::BinaryOp(MirBinaryOp::LtEq, Operand::Copy(Place::Local(fa)), Operand::Copy(Place::Local(fb))),
                span,
            ),
            Statement::Assign(
                Place::Local(fgt),
                Rvalue::BinaryOp(MirBinaryOp::Gt, Operand::Copy(Place::Local(fa)), Operand::Copy(Place::Local(fb))),
                span,
            ),
            Statement::Assign(
                Place::Local(fge),
                Rvalue::BinaryOp(MirBinaryOp::GtEq, Operand::Copy(Place::Local(fa)), Operand::Copy(Place::Local(fb))),
                span,
            ),
            Statement::Assign(
                Place::Local(feq),
                Rvalue::BinaryOp(MirBinaryOp::Eq, Operand::Copy(Place::Local(fa)), Operand::Copy(Place::Local(fb))),
                span,
            ),
            Statement::Assign(
                Place::Local(fne),
                Rvalue::BinaryOp(MirBinaryOp::NotEq, Operand::Copy(Place::Local(fa)), Operand::Copy(Place::Local(fb))),
                span,
            ),
        ]);
        ff.blocks[0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(fne))), span));
        ff
    };

    let bcf_nan = MirToBytecodeCompiler::compile(&build_float_ops("float_ops_nan", f64::NAN, f64::NAN))
        .expect("compile float_ops_nan");
    let bcf_zero = MirToBytecodeCompiler::compile(&build_float_ops("float_ops_zero", -0.0, 0.0))
        .expect("compile float_ops_zero");

    for bcf in [&bcf_nan, &bcf_zero] {
        let fins = &bcf.chunk.instructions;
        assert!(fins.iter().any(|i| matches!(i, Instruction::AddFloat(..))));
        assert!(fins.iter().any(|i| matches!(i, Instruction::SubFloat(..))));
        assert!(fins.iter().any(|i| matches!(i, Instruction::MulFloat(..))));
        assert!(fins.iter().any(|i| matches!(i, Instruction::DivFloat(..))));
        assert!(fins.iter().any(|i| matches!(i, Instruction::NegFloat(..))));
        assert!(fins.iter().any(|i| matches!(i, Instruction::EqFloat(..))));
        assert!(fins.iter().any(|i| matches!(i, Instruction::NeFloat(..))));
        assert!(fins.iter().any(|i| matches!(i, Instruction::LtFloat(..))));
        assert!(fins.iter().any(|i| matches!(i, Instruction::LeFloat(..))));
        assert!(fins.iter().any(|i| matches!(i, Instruction::GtFloat(..))));
        assert!(fins.iter().any(|i| matches!(i, Instruction::GeFloat(..))));
    }

    let mut vm = VM::new();
    vm.load_functions(vec![bci, bcf_nan, bcf_zero]);

    // a=11, b=4: t1=7, t2=21, t3=10, t4=3, t5=-4, t6=16, t7=4 -> out = 10 + 3 + 16 + 4 = 33
    assert_eq!(vm.call("int_ops", vec![]).expect("int_ops"), Value::Int(33));
    // IEEE-754 NaN != NaN is true; -0.0 != 0.0 is false
    assert_eq!(
        vm.call("float_ops_nan", vec![]).expect("float_ops NaN"),
        Value::Bool(true)
    );
    assert_eq!(
        vm.call("float_ops_zero", vec![]).expect("float_ops signed zero"),
        Value::Bool(false)
    );
}

#[test]
fn test_perf_t3_3_adv_mixed_unknown_and_string_never_emit_typed_int_opcodes() {
    let span = Span::dummy();
    let mut f = MirFunction::new(
        FunctionSignature {
            name: "mixed_and_unknown".into(),
            param_types: vec![
                MirType::Unknown,
                MirType::Int,
                MirType::Float,
                MirType::Array(Box::new(MirType::Unknown)),
                MirType::Array(Box::new(MirType::Int)),
            ],
            return_type: MirType::Float,
            is_public: true,
            effects: None,
        },
        span,
    );
    let unk = f.params[0];
    let iv = f.params[1];
    let fv = f.params[2];
    let unk_arr = f.params[3];
    let int_arr = f.params[4];

    let idx0 = f.add_local(None, MirType::Int, false, span);
    let r_unk_add = f.add_local(None, MirType::Unknown, false, span);
    let r_unk_lt = f.add_local(None, MirType::Bool, false, span);
    let r_mix_add = f.add_local(None, MirType::Float, false, span);
    let r_mix_le = f.add_local(None, MirType::Bool, false, span);
    let r_arr_unk = f.add_local(None, MirType::Unknown, false, span);
    let r_arr_int = f.add_local(None, MirType::Int, false, span);
    let r_str = f.add_local(None, MirType::String, false, span);

    f.blocks[0].statements.extend([
        Statement::Assign(Place::Local(idx0), Rvalue::Use(Operand::Constant(MirConstant::Int(0))), span),
        // Unknown + Int -> MUST remain generic Add
        Statement::Assign(
            Place::Local(r_unk_add),
            Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(unk)), Operand::Copy(Place::Local(iv))),
            span,
        ),
        // Int < Unknown -> MUST remain generic Lt
        Statement::Assign(
            Place::Local(r_unk_lt),
            Rvalue::BinaryOp(MirBinaryOp::Lt, Operand::Copy(Place::Local(iv)), Operand::Copy(Place::Local(unk))),
            span,
        ),
        // Int + Float -> MUST remain generic Add
        Statement::Assign(
            Place::Local(r_mix_add),
            Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(iv)), Operand::Copy(Place::Local(fv))),
            span,
        ),
        // Float <= Int -> MUST remain generic Le
        Statement::Assign(
            Place::Local(r_mix_le),
            Rvalue::BinaryOp(MirBinaryOp::LtEq, Operand::Copy(Place::Local(fv)), Operand::Copy(Place::Local(iv))),
            span,
        ),
        // Array<Unknown>[0] + 1 -> MUST remain generic Add
        Statement::Assign(
            Place::Local(r_arr_unk),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Index(Box::new(Place::Local(unk_arr)), idx0)),
                Operand::Constant(MirConstant::Int(1)),
            ),
            span,
        ),
        // String + String -> Concat (not AddInt)
        Statement::Assign(
            Place::Local(r_str),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Constant(MirConstant::String("hi".into())),
                Operand::Constant(MirConstant::String("!".into())),
            ),
            span,
        ),
        // Array<Int>[0] + 1 -> fail-closed mutable array index stays generic Add!
        Statement::Assign(
            Place::Local(r_arr_int),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Index(Box::new(Place::Local(int_arr)), idx0)),
                Operand::Constant(MirConstant::Int(1)),
            ),
            span,
        ),
        // Final return: r_mix_add + r_unk_add (Float + Unknown -> generic Add)
        Statement::Assign(
            Place::Local(r_mix_add),
            Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(r_mix_add)), Operand::Copy(Place::Local(r_unk_add))),
            span,
        ),
        Statement::Assign(
            Place::Local(r_mix_add),
            Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(r_mix_add)), Operand::Copy(Place::Local(r_arr_int))),
            span,
        ),
    ]);
    f.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(r_mix_add))),
        span,
    ));

    let bc = MirToBytecodeCompiler::compile(&f).expect("compile mixed_and_unknown");
    let insns = &bc.chunk.instructions;

    let add_int_count = insns.iter().filter(|i| matches!(i, Instruction::AddInt(..))).count();
    let generic_add_count = insns.iter().filter(|i| matches!(i, Instruction::Add(..))).count();
    assert_eq!(
        add_int_count, 0,
        "Param/unknown/mixed/mutable-array Add ops must NEVER emit AddInt; got {add_int_count}:\n{}",
        bc.disassemble()
    );
    assert_eq!(
        generic_add_count, 6,
        "Unknown/mixed/param/array Add ops must stay generic Add; got {generic_add_count}:\n{}",
        bc.disassemble()
    );
    assert!(!insns.iter().any(|i| matches!(i, Instruction::LtInt(..) | Instruction::LeInt(..))));
    assert!(insns.iter().any(|i| matches!(i, Instruction::Lt(..))));
    assert!(insns.iter().any(|i| matches!(i, Instruction::Le(..))));
    assert!(insns.iter().any(|i| matches!(i, Instruction::Concat(..))));

    let mut vm = VM::new();
    vm.load_functions(vec![bc]);
    // Pass runtime Float for `unk` and inside `unk_arr` — would fail if wrongly specialized to AddInt!
    let res = vm
        .call(
            "mixed_and_unknown",
            vec![
                Value::Float(2.5),
                Value::Int(4),
                Value::Float(1.5),
                Value::array(vec![Value::Float(3.5)]),
                Value::array(vec![Value::Int(10)]),
            ],
        )
        .expect("mixed/unknown runtime execution must succeed");
    // r_unk_add = 2.5 + 4 = 6.5; r_mix_add = 4 + 1.5 = 5.5; r_arr_int = 11 -> 5.5 + 6.5 + 11 = 23.0
    assert_eq!(res, Value::Float(23.0));
}

#[test]
fn test_perf_t3_3_cond_else_fallthrough_switch_call_and_empty_blocks_no_jump0() {
    let span = Span::dummy();

    // Helper callee: double(x) = x + x
    let mut callee = MirFunction::new(
        FunctionSignature {
            name: "double_it".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let cx = callee.params[0];
    let cout = callee.add_local(None, MirType::Int, false, span);
    callee.blocks[0].statements.push(Statement::Assign(
        Place::Local(cout),
        Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(cx)), Operand::Copy(Place::Local(cx))),
        span,
    ));
    callee.blocks[0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(cout))), span));

    // Caller exercises:
    // - Cond with else_block == next_block_id (emits JumpIf, zero Jump(0))
    // - Empty fallthrough block (Goto to next_block_id)
    // - Call with success == next_block_id (zero Jump(0))
    // - Switch with default == next_block_id (zero Jump(0))
    let mut f = MirFunction::new(
        FunctionSignature {
            name: "flow_test".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let x = f.params[0];
    let is_neg = f.add_local(None, MirType::Bool, false, span);
    let res = f.add_local(None, MirType::Int, true, span);

    let bb_else_empty = f.new_block(); // bb1: else_block of bb0 -> fallthrough!
    let bb_call = f.new_block();       // bb2: empty bb1 falls through here!
    let bb_switch = f.new_block();     // bb3: Call success falls through here!
    let bb_default = f.new_block();    // bb4: Switch default falls through here!
    let bb_case1 = f.new_block();      // bb5
    let bb_neg = f.new_block();        // bb6

    // bb0: if x < 0 then bb_neg else bb_else_empty (else_block is next!)
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(is_neg),
        Rvalue::BinaryOp(MirBinaryOp::Lt, Operand::Copy(Place::Local(x)), Operand::Constant(MirConstant::Int(0))),
        span,
    ));
    f.blocks[0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(is_neg)),
        then_block: bb_neg,
        else_block: bb_else_empty,
        span,
    });

    // bb1 (empty): goto bb_call (fallthrough!)
    f.blocks[bb_else_empty.0].terminator = Some(Terminator::Goto(bb_call, span));

    // bb2: res = call double_it(x) -> bb_switch (fallthrough!)
    f.blocks[bb_call.0].terminator = Some(Terminator::Call {
        callee: "double_it".into(),
        args: vec![Operand::Copy(Place::Local(x))],
        destination: Place::Local(res),
        success: bb_switch,
        error: None,
        span,
    });

    // bb3: switch res [ 2 => bb_case1 ] default => bb_default (fallthrough!)
    f.blocks[bb_switch.0].terminator = Some(Terminator::Switch {
        discr: Operand::Copy(Place::Local(res)),
        cases: vec![(2, bb_case1)],
        default: bb_default,
        span,
    });

    // bb4: return res + 100
    f.blocks[bb_default.0].statements.push(Statement::Assign(
        Place::Local(res),
        Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(res)), Operand::Constant(MirConstant::Int(100))),
        span,
    ));
    f.blocks[bb_default.0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(res))), span));

    // bb5: return 777
    f.blocks[bb_case1.0].terminator = Some(Terminator::Return(Some(Operand::Constant(MirConstant::Int(777))), span));

    // bb6: return -1
    f.blocks[bb_neg.0].terminator = Some(Terminator::Return(Some(Operand::Constant(MirConstant::Int(-1))), span));

    let bc_callee = MirToBytecodeCompiler::compile(&callee).expect("compile callee");
    let mut direct_indices = HashMap::new();
    direct_indices.insert("double_it".to_string(), 0usize);
    let bc_caller = MirToBytecodeCompiler::compile_with_direct_callee_indices(&f, &direct_indices)
        .expect("compile caller");

    assert_no_fallthrough_jump0(&bc_callee);
    assert_no_fallthrough_jump0(&bc_caller);
    assert_all_jumps_land_on_block_entries(&bc_caller);

    // bb0 must use JumpIf (else fallthrough)
    assert!(
        matches!(bc_caller.chunk.instructions[bc_caller.terminator_pcs[&0]], Instruction::JumpIf(..)),
        "Cond with else == next_block_id should emit JumpIf:\n{}",
        bc_caller.disassemble()
    );
    // Empty bb1 has identical entry/terminator PC equal to bb2 entry PC
    assert_eq!(bc_caller.block_entry_pcs[&bb_else_empty.0], bc_caller.terminator_pcs[&bb_else_empty.0]);
    assert_eq!(bc_caller.block_entry_pcs[&bb_else_empty.0], bc_caller.block_entry_pcs[&bb_call.0]);

    let mut vm = VM::new();
    vm.load_functions(vec![bc_callee, bc_caller]);
    assert_eq!(vm.call("flow_test", vec![Value::Int(-5)]).expect("x=-5"), Value::Int(-1));
    assert_eq!(vm.call("flow_test", vec![Value::Int(1)]).expect("x=1"), Value::Int(777));
    assert_eq!(vm.call("flow_test", vec![Value::Int(5)]).expect("x=5"), Value::Int(110));
}

#[test]
fn test_perf_t3_3_native_struct_opcodes_with_layout_and_unregistered_fallback() {
    let span = Span::dummy();

    let build_struct_fn = |fn_name: &str, struct_name: &str| {
        let mut f = MirFunction::new(
            FunctionSignature {
                name: fn_name.into(),
                param_types: vec![MirType::Int],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            span,
        );
        let n = f.params[0];
        let p = f.add_local(Some("p".into()), MirType::Struct(struct_name.into()), true, span);
        let alias = f.add_local(Some("alias".into()), MirType::Struct(struct_name.into()), false, span);
        let sum = f.add_local(Some("sum".into()), MirType::Int, false, span);
        let out = f.add_local(Some("out".into()), MirType::Int, false, span);

        f.blocks[0].statements.extend([
            Statement::Assign(
                Place::Local(p),
                Rvalue::Aggregate(
                    MirType::Struct(struct_name.into()),
                    vec![Operand::Copy(Place::Local(n)), Operand::Constant(MirConstant::Int(10))],
                ),
                span,
            ),
            Statement::Assign(
                Place::Local(alias),
                Rvalue::Use(Operand::Copy(Place::Local(p))),
                span,
            ),
            // sum = p.x + p.y (direct Place::Field operands!)
            Statement::Assign(
                Place::Local(sum),
                Rvalue::BinaryOp(
                    MirBinaryOp::Add,
                    Operand::Copy(Place::Field(Box::new(Place::Local(p)), 0)),
                    Operand::Copy(Place::Field(Box::new(Place::Local(p)), 1)),
                ),
                span,
            ),
            // p.y = sum
            Statement::Assign(
                Place::Field(Box::new(Place::Local(p)), 1),
                Rvalue::Use(Operand::Copy(Place::Local(sum))),
                span,
            ),
            // out = alias.y (proves shared Value::Struct mutation)
            Statement::Assign(
                Place::Local(out),
                Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(alias)), 1))),
                span,
            ),
        ]);
        f.blocks[0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(out))), span));
        f
    };

    let f_point = build_struct_fn("point_native", "Point");
    let f_anon = build_struct_fn("anon_fallback", "UnregisteredStruct");

    let mut layouts = HashMap::new();
    layouts.insert("Point".to_string(), make_point_layout());

    // 1. Registered struct layout -> emits NewStruct, GetField, SetField, while mutable field ops stay generic Add
    let bc_point = MirToBytecodeCompiler::compile_with_struct_layouts(&f_point, &layouts)
        .expect("compile point_native");
    let pins = &bc_point.chunk.instructions;
    assert!(
        pins.iter().any(|i| matches!(i, Instruction::NewStruct(..))),
        "Expected NewStruct for registered Point:\n{}",
        bc_point.disassemble()
    );
    assert!(
        pins.iter().any(|i| matches!(i, Instruction::GetField(..))),
        "Expected GetField for registered Point:\n{}",
        bc_point.disassemble()
    );
    assert!(
        pins.iter().any(|i| matches!(i, Instruction::SetField(..))),
        "Expected SetField for registered Point:\n{}",
        bc_point.disassemble()
    );
    assert!(
        pins.iter().any(|i| matches!(i, Instruction::Add(..))),
        "Mutable struct field addition must stay generic Add (fail-closed):\n{}",
        bc_point.disassemble()
    );
    assert!(
        !pins.iter().any(|i| matches!(i, Instruction::AddInt(..))),
        "Mutable struct field addition must NOT emit AddInt:\n{}",
        bc_point.disassemble()
    );
    assert!(
        !pins.iter().any(|i| matches!(
            i,
            Instruction::NewArray(..) | Instruction::GetIndex(..) | Instruction::SetIndex(..)
        )),
        "Registered Point must not fall back to array/index opcodes:\n{}",
        bc_point.disassemble()
    );

    // 2. Unregistered struct -> preserves fallback to NewArray/GetIndex/SetIndex
    let bc_anon = MirToBytecodeCompiler::compile_with_struct_layouts(&f_anon, &layouts)
        .expect("compile anon_fallback");
    let ains = &bc_anon.chunk.instructions;
    assert!(ains.iter().any(|i| matches!(i, Instruction::NewArray(..))));
    assert!(ains.iter().any(|i| matches!(i, Instruction::GetIndex(..))));
    assert!(ains.iter().any(|i| matches!(i, Instruction::SetIndex(..))));
    assert!(!ains.iter().any(|i| matches!(
        i,
        Instruction::NewStruct(..) | Instruction::GetField(..) | Instruction::SetField(..)
    )));

    // 3. Default `compile()` without layouts also preserves NewArray/GetIndex/SetIndex
    let bc_default = MirToBytecodeCompiler::compile(&f_point).expect("compile default");
    assert!(bc_default.chunk.instructions.iter().any(|i| matches!(i, Instruction::NewArray(..))));

    // 4. Execute both native struct and fallback functions on VM
    let mut vm = VM::new();
    vm.register_structs(&make_point_struct_defs());
    vm.load_functions(vec![bc_point, bc_anon]);
    assert_eq!(
        vm.call("point_native", vec![Value::Int(32)]).expect("run point_native"),
        Value::Int(42)
    );
    assert_eq!(
        vm.call("anon_fallback", vec![Value::Int(32)]).expect("run anon_fallback"),
        Value::Int(42)
    );
}

#[test]
fn test_perf_t3_3_deopt_and_continuation_pcs_exact_with_elided_fallthrough() {
    let span = Span::dummy();
    let mut f = MirFunction::new(
        FunctionSignature {
            name: "deopt_pc_check".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let n = f.params[0];
    let acc = f.add_local(Some("acc".into()), MirType::Int, true, span);
    let i = f.add_local(Some("i".into()), MirType::Int, true, span);
    let cond = f.add_local(Some("cond".into()), MirType::Bool, false, span);
    let eff_res = f.add_local(Some("eff_res".into()), MirType::Int, false, span);

    let bb_hdr = f.new_block();  // bb1 (loop header, BlockEntry deopt site)
    let bb_call = f.new_block(); // bb2 (PreTerminator Call deopt site, then_block fallthrough)
    let bb_cont = f.new_block(); // bb3 (Call success fallthrough)
    let bb_exit = f.new_block(); // bb4

    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        span,
    ));
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        span,
    ));
    f.blocks[0].terminator = Some(Terminator::Goto(bb_hdr, span));

    f.blocks[bb_hdr.0].statements.push(Statement::Assign(
        Place::Local(cond),
        Rvalue::BinaryOp(MirBinaryOp::Lt, Operand::Copy(Place::Local(i)), Operand::Copy(Place::Local(n))),
        span,
    ));
    f.blocks[bb_hdr.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond)),
        then_block: bb_call,
        else_block: bb_exit,
        span,
    });

    f.blocks[bb_call.0].terminator = Some(Terminator::Call {
        callee: "step_cb".into(),
        args: vec![Operand::Copy(Place::Local(i))],
        destination: Place::Local(eff_res),
        success: bb_cont,
        error: None,
        span,
    });

    f.blocks[bb_cont.0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(acc)), Operand::Copy(Place::Local(eff_res))),
        span,
    ));
    f.blocks[bb_cont.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(i)), Operand::Constant(MirConstant::Int(1))),
        span,
    ));
    f.blocks[bb_cont.0].terminator = Some(Terminator::Goto(bb_hdr, span));

    f.blocks[bb_exit.0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(acc))), span));

    let bc = MirToBytecodeCompiler::compile(&f).expect("compile deopt_pc_check");
    assert_no_fallthrough_jump0(&bc);
    assert_all_jumps_land_on_block_entries(&bc);

    // PreTerminator PC for bb_call must point to the start of the Call sequence, and
    // the instruction right after Instruction::Call must be bb_cont's block_entry_pc!
    let call_term_pc = bc.terminator_pcs[&bb_call.0];
    let cont_entry_pc = bc.block_entry_pcs[&bb_cont.0];
    assert!(call_term_pc < cont_entry_pc);
    assert!(
        matches!(bc.chunk.instructions[cont_entry_pc - 1], Instruction::Call(..)),
        "With elided fallthrough after Call, instruction immediately before bb_cont must be Call:\n{}",
        bc.disassemble()
    );

    let mut vm = VM::new();
    vm.register_native(
        "step_cb",
        Value::NativeFunction(lust::bytecode::native_fn(|args: &[Value]| {
            let v = args[0].as_int().unwrap_or(0);
            Ok(lust::bytecode::NativeCallResult::Return(Value::Int(v * 2 + 1)))
        })),
    );
    vm.load_functions(vec![bc]);
    // Sum of (2*i + 1) for i=0..5 = 1+3+5+7+9 = 25
    assert_eq!(
        vm.call("deopt_pc_check", vec![Value::Int(5)]).expect("run deopt_pc_check"),
        Value::Int(25)
    );
}

/// Adversarial soundness test for AstLowering placeholder `MirType::Int` temps:
/// `AstLowering` creates unnamed expression temporaries with placeholder `MirType::Int`
/// even when computing float expressions like `x * x - acc * 0.5`.
/// `MirToBytecodeCompiler` MUST refine types via forward dataflow and emit `MulFloat`/`SubFloat`,
/// NEVER mis-emitting `SubInt` on float temporaries!
#[test]
fn test_perf_t3_3_adv_placeholder_int_temp_holding_float_and_reassigned_local() {
    let span = Span::dummy();
    let mut f = MirFunction::new(
        FunctionSignature {
            name: "placeholder_float_temps".into(),
            param_types: vec![],
            return_type: MirType::Float,
            is_public: true,
            effects: None,
        },
        span,
    );
    let x = f.add_local(Some("x".into()), MirType::Float, false, span);
    let acc = f.add_local(Some("acc".into()), MirType::Float, false, span);
    // Deliberately declare temps as placeholder `MirType::Int` (exactly as AstLowering does!)
    let t_xx = f.add_local(None, MirType::Int, true, span);
    let t_half = f.add_local(None, MirType::Int, true, span);
    let t_diff = f.add_local(None, MirType::Int, true, span);
    // Also a local reassigned from Int to Float -> must widen to Unknown (generic Add)
    let dyn_v = f.add_local(Some("dyn_v".into()), MirType::Int, true, span);
    let out = f.add_local(None, MirType::Int, true, span);

    f.blocks[0].statements.extend([
        Statement::Assign(Place::Local(x), Rvalue::Use(Operand::Constant(MirConstant::Float(2.0))), span),
        Statement::Assign(Place::Local(acc), Rvalue::Use(Operand::Constant(MirConstant::Float(6.0))), span),
        Statement::Assign(
            Place::Local(t_xx),
            Rvalue::BinaryOp(MirBinaryOp::Mul, Operand::Copy(Place::Local(x)), Operand::Copy(Place::Local(x))),
            span,
        ),
        Statement::Assign(
            Place::Local(t_half),
            Rvalue::BinaryOp(MirBinaryOp::Mul, Operand::Copy(Place::Local(acc)), Operand::Constant(MirConstant::Float(0.5))),
            span,
        ),
        Statement::Assign(
            Place::Local(t_diff),
            Rvalue::BinaryOp(MirBinaryOp::Sub, Operand::Copy(Place::Local(t_xx)), Operand::Copy(Place::Local(t_half))),
            span,
        ),
        Statement::Assign(
            Place::Local(dyn_v),
            Rvalue::Use(Operand::Constant(MirConstant::Int(1))),
            span,
        ),
        Statement::Assign(
            Place::Local(dyn_v),
            Rvalue::Use(Operand::Constant(MirConstant::Float(0.25))),
            span,
        ),
        Statement::Assign(
            Place::Local(out),
            Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(t_diff)), Operand::Copy(Place::Local(dyn_v))),
            span,
        ),
    ]);
    f.blocks[0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(out))), span));

    let bc = MirToBytecodeCompiler::compile(&f).expect("compile placeholder_float_temps");
    let insns = &bc.chunk.instructions;
    assert!(
        insns.iter().any(|i| matches!(i, Instruction::MulFloat(..))),
        "Expected MulFloat for float temps:\n{}",
        bc.disassemble()
    );
    assert!(
        insns.iter().any(|i| matches!(i, Instruction::SubFloat(..))),
        "Expected SubFloat for float temp subtraction:\n{}",
        bc.disassemble()
    );
    assert!(
        !insns.iter().any(|i| matches!(
            i,
            Instruction::MulInt(..) | Instruction::SubInt(..) | Instruction::AddInt(..)
        )),
        "Placeholder Int temps holding Float must NEVER emit Int opcodes:\n{}",
        bc.disassemble()
    );
    assert!(
        insns.iter().any(|i| matches!(i, Instruction::Add(..))),
        "Reassigned Int->Float local must widen to generic Add:\n{}",
        bc.disassemble()
    );

    let mut vm = VM::new();
    vm.load_functions(vec![bc]);
    // x=2.0, acc=6.0 -> t_xx=4.0, t_half=3.0, t_diff=1.0, dyn_v=0.25 -> out=1.25
    assert_eq!(
        vm.call("placeholder_float_temps", vec![])
            .expect("placeholder_float_temps must execute cleanly"),
        Value::Float(1.25)
    );
}

/// Adversarial test 1 (RED->GREEN):
/// Conditional initialization `if c { x = 5 }; return x == 1` (plus transitive copy `y = x; return y == 1`).
/// `x` may be uninitialized (`Value::Nil`) on the false branch, so comparisons MUST stay generic `Eq`.
#[test]
fn test_perf_t3_3_adv_conditional_uninit_nil_stays_generic() {
    let span = Span::dummy();

    // Direct read of conditionally initialized local: `if c { x = 5 }; return x == 1`
    let mut f = MirFunction::new(
        FunctionSignature {
            name: "cond_uninit_direct".into(),
            param_types: vec![MirType::Bool],
            return_type: MirType::Bool,
            is_public: true,
            effects: None,
        },
        span,
    );
    let c = f.params[0];
    let x = f.add_local(Some("x".into()), MirType::Int, true, span);
    let res = f.add_local(Some("res".into()), MirType::Bool, false, span);

    let bb_then = f.new_block(); // bb1
    let bb_merge = f.new_block(); // bb2

    f.blocks[0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(c)),
        then_block: bb_then,
        else_block: bb_merge,
        span,
    });
    f.blocks[bb_then.0].statements.push(Statement::Assign(
        Place::Local(x),
        Rvalue::Use(Operand::Constant(MirConstant::Int(5))),
        span,
    ));
    f.blocks[bb_then.0].terminator = Some(Terminator::Goto(bb_merge, span));
    f.blocks[bb_merge.0].statements.push(Statement::Assign(
        Place::Local(res),
        Rvalue::BinaryOp(
            MirBinaryOp::Eq,
            Operand::Copy(Place::Local(x)),
            Operand::Constant(MirConstant::Int(5)),
        ),
        span,
    ));
    f.blocks[bb_merge.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(res))),
        span,
    ));

    // Transitive propagation: `if c { x = 1 }; y = x; return y == 1`
    let mut f_trans = MirFunction::new(
        FunctionSignature {
            name: "cond_uninit_transitive".into(),
            param_types: vec![MirType::Bool],
            return_type: MirType::Bool,
            is_public: true,
            effects: None,
        },
        span,
    );
    let tc = f_trans.params[0];
    let tx = f_trans.add_local(Some("x".into()), MirType::Int, true, span);
    let ty = f_trans.add_local(Some("y".into()), MirType::Int, true, span);
    let tres = f_trans.add_local(Some("res".into()), MirType::Bool, false, span);
    let t_then = f_trans.new_block();
    let t_merge = f_trans.new_block();

    f_trans.blocks[0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(tc)),
        then_block: t_then,
        else_block: t_merge,
        span,
    });
    f_trans.blocks[t_then.0].statements.push(Statement::Assign(
        Place::Local(tx),
        Rvalue::Use(Operand::Constant(MirConstant::Int(1))),
        span,
    ));
    f_trans.blocks[t_then.0].terminator = Some(Terminator::Goto(t_merge, span));
    f_trans.blocks[t_merge.0].statements.extend([
        Statement::Assign(
            Place::Local(ty),
            Rvalue::Use(Operand::Copy(Place::Local(tx))),
            span,
        ),
        Statement::Assign(
            Place::Local(tres),
            Rvalue::BinaryOp(
                MirBinaryOp::Eq,
                Operand::Copy(Place::Local(ty)),
                Operand::Constant(MirConstant::Int(1)),
            ),
            span,
        ),
    ]);
    f_trans.blocks[t_merge.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(tres))),
        span,
    ));

    let bc = MirToBytecodeCompiler::compile(&f).expect("compile cond_uninit_direct");
    let bc_trans =
        MirToBytecodeCompiler::compile(&f_trans).expect("compile cond_uninit_transitive");

    for compiled in [&bc, &bc_trans] {
        let insns = &compiled.chunk.instructions;
        assert!(
            insns.iter().any(|i| matches!(i, Instruction::Eq(..))),
            "Expected generic Eq for conditionally uninit local in '{}':\n{}",
            compiled.name,
            compiled.disassemble()
        );
        assert!(
            !insns.iter().any(|i| matches!(i, Instruction::EqInt(..))),
            "Must NOT emit EqInt for conditionally uninit local in '{}':\n{}",
            compiled.name,
            compiled.disassemble()
        );
    }

    let mut vm = VM::new();
    vm.load_functions(vec![bc, bc_trans]);
    assert_eq!(
        vm.call("cond_uninit_direct", vec![Value::Bool(false)])
            .expect("cond_uninit_direct(false) must not trap on Nil"),
        Value::Bool(false)
    );
    assert_eq!(
        vm.call("cond_uninit_direct", vec![Value::Bool(true)])
            .expect("cond_uninit_direct(true)"),
        Value::Bool(true)
    );
    assert_eq!(
        vm.call("cond_uninit_transitive", vec![Value::Bool(false)])
            .expect("cond_uninit_transitive(false) must not trap on Nil"),
        Value::Bool(false)
    );
    assert_eq!(
        vm.call("cond_uninit_transitive", vec![Value::Bool(true)])
            .expect("cond_uninit_transitive(true)"),
        Value::Bool(true)
    );
}

/// Adversarial test 2 (RED->GREEN):
/// Aliased array mutation (`a = [1]; b = a; b[0] = "s"; return a[0] == 1`) AND
/// mutable struct field overwrite (`s.x = "s"; return s.x == 1`) MUST stay generic `Eq` (no `EqInt`).
#[test]
fn test_perf_t3_3_adv_aliased_array_and_struct_field_mutation_stay_generic() {
    let span = Span::dummy();

    // 1. Aliased array mutation
    let mut f_arr = MirFunction::new(
        FunctionSignature {
            name: "aliased_array_mutation".into(),
            param_types: vec![],
            return_type: MirType::Bool,
            is_public: true,
            effects: None,
        },
        span,
    );
    let a = f_arr.add_local(Some("a".into()), MirType::Array(Box::new(MirType::Int)), true, span);
    let b = f_arr.add_local(Some("b".into()), MirType::Array(Box::new(MirType::Int)), true, span);
    let idx0 = f_arr.add_local(Some("idx0".into()), MirType::Int, false, span);
    let out_arr = f_arr.add_local(Some("out".into()), MirType::Bool, false, span);

    f_arr.blocks[0].statements.extend([
        Statement::Assign(
            Place::Local(idx0),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            span,
        ),
        Statement::Assign(
            Place::Local(a),
            Rvalue::Aggregate(
                MirType::Array(Box::new(MirType::Int)),
                vec![Operand::Constant(MirConstant::Int(1))],
            ),
            span,
        ),
        Statement::Assign(
            Place::Local(b),
            Rvalue::Use(Operand::Copy(Place::Local(a))),
            span,
        ),
        Statement::Assign(
            Place::Index(Box::new(Place::Local(b)), idx0),
            Rvalue::Use(Operand::Constant(MirConstant::String("s".into()))),
            span,
        ),
        Statement::Assign(
            Place::Local(out_arr),
            Rvalue::BinaryOp(
                MirBinaryOp::Eq,
                Operand::Copy(Place::Index(Box::new(Place::Local(a)), idx0)),
                Operand::Constant(MirConstant::Int(1)),
            ),
            span,
        ),
    ]);
    f_arr.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(out_arr))),
        span,
    ));

    // 2. Mutable struct field overwritten with string
    let mut f_struct = MirFunction::new(
        FunctionSignature {
            name: "mutable_struct_field_mutation".into(),
            param_types: vec![],
            return_type: MirType::Bool,
            is_public: true,
            effects: None,
        },
        span,
    );
    let s = f_struct.add_local(Some("s".into()), MirType::Struct("Point".into()), true, span);
    let out_s = f_struct.add_local(Some("out".into()), MirType::Bool, false, span);

    f_struct.blocks[0].statements.extend([
        Statement::Assign(
            Place::Local(s),
            Rvalue::Aggregate(
                MirType::Struct("Point".into()),
                vec![
                    Operand::Constant(MirConstant::Int(1)),
                    Operand::Constant(MirConstant::Int(2)),
                ],
            ),
            span,
        ),
        Statement::Assign(
            Place::Field(Box::new(Place::Local(s)), 0),
            Rvalue::Use(Operand::Constant(MirConstant::String("s".into()))),
            span,
        ),
        Statement::Assign(
            Place::Local(out_s),
            Rvalue::BinaryOp(
                MirBinaryOp::Eq,
                Operand::Copy(Place::Field(Box::new(Place::Local(s)), 0)),
                Operand::Constant(MirConstant::Int(1)),
            ),
            span,
        ),
    ]);
    f_struct.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(out_s))),
        span,
    ));

    let mut layouts = HashMap::new();
    layouts.insert("Point".to_string(), make_point_layout());

    let bc_arr = MirToBytecodeCompiler::compile(&f_arr).expect("compile aliased_array_mutation");
    let bc_struct = MirToBytecodeCompiler::compile_with_struct_layouts(&f_struct, &layouts)
        .expect("compile mutable_struct_field_mutation");

    assert!(
        bc_arr.chunk.instructions.iter().any(|i| matches!(i, Instruction::Eq(..))),
        "Expected generic Eq for aliased array element:\n{}",
        bc_arr.disassemble()
    );
    assert!(
        !bc_arr.chunk.instructions.iter().any(|i| matches!(i, Instruction::EqInt(..))),
        "Must NOT emit EqInt for mutable array index:\n{}",
        bc_arr.disassemble()
    );

    assert!(
        bc_struct.chunk.instructions.iter().any(|i| matches!(i, Instruction::NewStruct(..))),
        "Expected native NewStruct:\n{}",
        bc_struct.disassemble()
    );
    assert!(
        bc_struct.chunk.instructions.iter().any(|i| matches!(i, Instruction::SetField(..))),
        "Expected native SetField:\n{}",
        bc_struct.disassemble()
    );
    assert!(
        bc_struct.chunk.instructions.iter().any(|i| matches!(i, Instruction::GetField(..))),
        "Expected native GetField:\n{}",
        bc_struct.disassemble()
    );
    assert!(
        bc_struct.chunk.instructions.iter().any(|i| matches!(i, Instruction::Eq(..))),
        "Expected generic Eq for mutable struct field:\n{}",
        bc_struct.disassemble()
    );
    assert!(
        !bc_struct.chunk.instructions.iter().any(|i| matches!(i, Instruction::EqInt(..))),
        "Must NOT emit EqInt for mutable struct field:\n{}",
        bc_struct.disassemble()
    );

    let mut vm = VM::new();
    vm.register_structs(&make_point_struct_defs());
    vm.load_functions(vec![bc_arr, bc_struct]);

    assert_eq!(
        vm.call("aliased_array_mutation", vec![])
            .expect("aliased_array_mutation must execute cleanly"),
        Value::Bool(false)
    );
    assert_eq!(
        vm.call("mutable_struct_field_mutation", vec![])
            .expect("mutable_struct_field_mutation must execute cleanly"),
        Value::Bool(false)
    );
}

/// Adversarial test 3 (RED->GREEN):
/// Shadowed global `math.floor` returning `"not_int"` during `Rvalue::Cast(1.5, MirType::Int)`:
/// `cast_res == 1` MUST emit generic `Eq` (no `EqInt`) and return `Value::Bool(false)`.
#[test]
fn test_perf_t3_3_adv_shadowed_math_floor_cast_int_stays_generic() {
    let span = Span::dummy();
    let mut f = MirFunction::new(
        FunctionSignature {
            name: "shadowed_cast_int".into(),
            param_types: vec![],
            return_type: MirType::Bool,
            is_public: true,
            effects: None,
        },
        span,
    );
    let casted = f.add_local(Some("casted".into()), MirType::Int, false, span);
    let out = f.add_local(Some("out".into()), MirType::Bool, false, span);

    f.blocks[0].statements.extend([
        Statement::Assign(
            Place::Local(casted),
            Rvalue::Cast(Operand::Constant(MirConstant::Float(1.5)), MirType::Int),
            span,
        ),
        Statement::Assign(
            Place::Local(out),
            Rvalue::BinaryOp(
                MirBinaryOp::Eq,
                Operand::Copy(Place::Local(casted)),
                Operand::Constant(MirConstant::Int(1)),
            ),
            span,
        ),
    ]);
    f.blocks[0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(out))), span));

    let bc = MirToBytecodeCompiler::compile(&f).expect("compile shadowed_cast_int");
    assert!(
        bc.chunk.instructions.iter().any(|i| matches!(i, Instruction::Eq(..))),
        "Expected generic Eq after dynamic Cast(_, Int):\n{}",
        bc.disassemble()
    );
    assert!(
        !bc.chunk.instructions.iter().any(|i| matches!(i, Instruction::EqInt(..))),
        "Must NOT emit EqInt after dynamic Cast(_, Int):\n{}",
        bc.disassemble()
    );

    let mut vm = VM::new();
    let mut fake_math = LustMap::default();
    fake_math.insert(
        ValueKey::from_value(&Value::string("floor".to_string())),
        Value::NativeFunction(native_fn(|_args: &[Value]| {
            Ok(NativeCallResult::Return(Value::string("not_int".to_string())))
        })),
    );
    vm.set_global("math", Value::map(fake_math));
    vm.load_functions(vec![bc]);

    assert_eq!(
        vm.call("shadowed_cast_int", vec![])
            .expect("shadowed_cast_int must execute cleanly"),
        Value::Bool(false)
    );
}

/// Adversarial test 4 (RED->GREEN):
/// Dynamic parameter declared `MirType::Int` passed wrong runtime types (`String("hi")`, `Float(1.0)`):
/// `x == 1` MUST stay generic `Eq` (no `EqInt`), matching dynamic VM `Value::PartialEq` semantics,
/// and `x + 1` MUST stay generic `Add` (no `AddInt`), returning `Float(2.5)` for `Float(1.5)`.
#[test]
fn test_perf_t3_3_adv_dynamic_param_wrong_runtime_type_stays_generic() {
    let span = Span::dummy();
    let mut f_eq = MirFunction::new(
        FunctionSignature {
            name: "dyn_param_eq".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Bool,
            is_public: true,
            effects: None,
        },
        span,
    );
    let x = f_eq.params[0];
    let out = f_eq.add_local(Some("out".into()), MirType::Bool, false, span);
    f_eq.blocks[0].statements.push(Statement::Assign(
        Place::Local(out),
        Rvalue::BinaryOp(
            MirBinaryOp::Eq,
            Operand::Copy(Place::Local(x)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        span,
    ));
    f_eq.blocks[0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(out))), span));

    let mut f_add = MirFunction::new(
        FunctionSignature {
            name: "dyn_param_add".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let ax = f_add.params[0];
    let aout = f_add.add_local(Some("out".into()), MirType::Int, false, span);
    f_add.blocks[0].statements.push(Statement::Assign(
        Place::Local(aout),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(ax)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        span,
    ));
    f_add.blocks[0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(aout))), span));

    let bc_eq = MirToBytecodeCompiler::compile(&f_eq).expect("compile dyn_param_eq");
    assert!(
        bc_eq.chunk.instructions.iter().any(|i| matches!(i, Instruction::Eq(..))),
        "Expected generic Eq for dynamically typed param:\n{}",
        bc_eq.disassemble()
    );
    assert!(
        !bc_eq.chunk.instructions.iter().any(|i| matches!(i, Instruction::EqInt(..))),
        "Must NOT emit EqInt for unchecked MirType::Int param:\n{}",
        bc_eq.disassemble()
    );

    let bc_add = MirToBytecodeCompiler::compile(&f_add).expect("compile dyn_param_add");
    assert!(
        bc_add.chunk.instructions.iter().any(|i| matches!(i, Instruction::Add(..))),
        "Expected generic Add for dynamically typed param:\n{}",
        bc_add.disassemble()
    );
    assert!(
        !bc_add.chunk.instructions.iter().any(|i| matches!(i, Instruction::AddInt(..))),
        "Must NOT emit AddInt for unchecked MirType::Int param:\n{}",
        bc_add.disassemble()
    );

    let mut vm = VM::new();
    vm.load_functions(vec![bc_eq, bc_add]);
    assert_eq!(
        vm.call("dyn_param_eq", vec![Value::string("hi".to_string())])
            .expect("dyn_param_eq(\"hi\") must succeed"),
        Value::Bool(Value::string("hi".to_string()) == Value::Int(1))
    );
    assert_eq!(
        vm.call("dyn_param_eq", vec![Value::Float(1.0)])
            .expect("dyn_param_eq(1.0) must succeed without EqInt type error"),
        Value::Bool(Value::Float(1.0) == Value::Int(1))
    );
    assert_eq!(
        vm.call("dyn_param_eq", vec![Value::Int(1)])
            .expect("dyn_param_eq(1) must succeed"),
        Value::Bool(true)
    );
    assert_eq!(
        vm.call("dyn_param_add", vec![Value::Float(1.5)])
            .expect("dyn_param_add(1.5) must coerce via generic Add"),
        Value::Float(2.5)
    );
    assert_eq!(
        vm.call("dyn_param_add", vec![Value::Int(41)])
            .expect("dyn_param_add(41)"),
        Value::Int(42)
    );
}

/// Adversarial test 5 (RED->GREEN):
/// 1) Unreachable block assigning `Float` (and reading an uninit local) MUST NOT poison reachable `Int` locals.
/// 2) Comprehensive CFG terminator & layout edge cases:
///    empty block, self-loop (`Goto(self)` NOT elided), block 0 fallthrough, trailing block,
///    `Cond` (then-fallthrough, else-fallthrough, `then == else`),
///    `Switch` (default-fallthrough & non-fallthrough), and
///    `Call` (success-fallthrough & non-fallthrough), with exact `block_entry_pcs`, `terminator_pcs`,
///    and branch jump targets.
#[test]
fn test_perf_t3_3_adv_unreachable_block_and_cfg_terminator_edge_cases() {
    let span = Span::dummy();

    // Part A: Unreachable block (with non-dense BlockId(99)) assigning Float to reachable Int local `v`
    let mut f_unreach = MirFunction::new(
        FunctionSignature {
            name: "unreachable_no_poison".into(),
            param_types: vec![],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let v = f_unreach.add_local(Some("v".into()), MirType::Int, true, span);
    let out = f_unreach.add_local(Some("out".into()), MirType::Int, false, span);

    // bb0 (reachable): v = 10; out = v + 5; return out
    f_unreach.blocks[0].statements.extend([
        Statement::Assign(
            Place::Local(v),
            Rvalue::Use(Operand::Constant(MirConstant::Int(10))),
            span,
        ),
        Statement::Assign(
            Place::Local(out),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(v)),
                Operand::Constant(MirConstant::Int(5)),
            ),
            span,
        ),
    ]);
    f_unreach.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(out))),
        span,
    ));

    // Unreachable block with non-dense BlockId(99) assigning Float to `v` and reading uninit `out`
    let mut bb_dead = BasicBlock::new(BlockId(99));
    bb_dead.statements.push(Statement::Assign(
        Place::Local(v),
        Rvalue::Use(Operand::Constant(MirConstant::Float(3.5))),
        span,
    ));
    bb_dead.terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(v))),
        span,
    ));
    f_unreach.blocks.push(bb_dead);

    let bc_unreach =
        MirToBytecodeCompiler::compile(&f_unreach).expect("compile unreachable_no_poison");
    assert!(
        bc_unreach.chunk.instructions.iter().any(|i| matches!(i, Instruction::AddInt(..))),
        "Unreachable block must NOT poison reachable Int local `v`:\n{}",
        bc_unreach.disassemble()
    );
    assert!(
        !bc_unreach.chunk.instructions.iter().any(|i| matches!(i, Instruction::Add(..))),
        "Expected AddInt (not generic Add) when only unreachable block writes Float:\n{}",
        bc_unreach.disassemble()
    );

    // Part B: Comprehensive CFG terminator edge cases (including non-dense BlockIds & self-loop)
    let mut callee = MirFunction::new(
        FunctionSignature {
            name: "id_plus_one".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let cx = callee.params[0];
    let cres = callee.add_local(None, MirType::Int, false, span);
    callee.blocks[0].statements.push(Statement::Assign(
        Place::Local(cres),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(cx)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        span,
    ));
    callee.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(cres))),
        span,
    ));

    let mut f_cfg = MirFunction::new(
        FunctionSignature {
            name: "cfg_edge_cases".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let mode = f_cfg.params[0];
    let c_zero = f_cfg.add_local(None, MirType::Bool, false, span);
    let c_one = f_cfg.add_local(None, MirType::Bool, false, span);
    let call_out = f_cfg.add_local(None, MirType::Int, true, span);

    // Blocks in layout order (mixing dense and non-dense IDs):
    // idx 0: bb0 (block 0 fallthrough Goto -> bb1)
    // idx 1: bb1 (empty block, Goto -> bb2 fallthrough)
    // idx 2: bb2 (Cond then-fallthrough: if mode == 0 then bb3 else bb4)
    // idx 3: bb3 (Cond then == else fallthrough: if c_zero then bb4 else bb4)
    // idx 4: bb4 (Cond else-fallthrough: if mode == 1 then bb_selfloop(90) else bb5)
    // idx 5: bb5 (Call success-fallthrough -> bb6)
    // idx 6: bb6 (Switch default-fallthrough: case 11 => bb8, default => bb7)
    // idx 7: bb7 (Call non-fallthrough -> bb9)
    // idx 8: bb8 (Switch non-fallthrough: case 20 => bb9, default => bb10)
    // idx 9: bb9 (Cond then == else non-fallthrough -> bb10)
    // idx 10: bb_selfloop(90) (Self-loop: Goto(bb_selfloop) — MUST NOT be elided!)
    // idx 11: bb10 (Trailing block: Return call_out)
    let bb1 = BlockId(1);
    let bb2 = BlockId(2);
    let bb3 = BlockId(3);
    let bb4 = BlockId(4);
    let bb5 = BlockId(5);
    let bb6 = BlockId(6);
    let bb7 = BlockId(7);
    let bb8 = BlockId(8);
    let bb9 = BlockId(9);
    let bb_selfloop = BlockId(90);
    let bb10 = BlockId(10);

    // bb0: block 0 fallthrough
    f_cfg.blocks[0].statements.push(Statement::Assign(
        Place::Local(call_out),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        span,
    ));
    f_cfg.blocks[0].terminator = Some(Terminator::Goto(bb1, span));

    // bb1: empty block fallthrough
    let mut b1 = BasicBlock::new(bb1);
    b1.terminator = Some(Terminator::Goto(bb2, span));

    // bb2: Cond then-fallthrough (then = bb3, else = bb4)
    let mut b2 = BasicBlock::new(bb2);
    b2.statements.push(Statement::Assign(
        Place::Local(c_zero),
        Rvalue::BinaryOp(
            MirBinaryOp::Eq,
            Operand::Copy(Place::Local(mode)),
            Operand::Constant(MirConstant::Int(0)),
        ),
        span,
    ));
    b2.terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(c_zero)),
        then_block: bb3,
        else_block: bb4,
        span,
    });

    // bb3: Cond then == else (fallthrough to bb4)
    let mut b3 = BasicBlock::new(bb3);
    b3.terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(c_zero)),
        then_block: bb4,
        else_block: bb4,
        span,
    });

    // bb4: Cond else-fallthrough (then = bb_selfloop, else = bb5)
    let mut b4 = BasicBlock::new(bb4);
    b4.statements.push(Statement::Assign(
        Place::Local(c_one),
        Rvalue::BinaryOp(
            MirBinaryOp::Eq,
            Operand::Copy(Place::Local(mode)),
            Operand::Constant(MirConstant::Int(999)),
        ),
        span,
    ));
    b4.terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(c_one)),
        then_block: bb_selfloop,
        else_block: bb5,
        span,
    });

    // bb5: Call success-fallthrough -> bb6
    let mut b5 = BasicBlock::new(bb5);
    b5.terminator = Some(Terminator::Call {
        callee: "id_plus_one".into(),
        args: vec![Operand::Copy(Place::Local(mode))],
        destination: Place::Local(call_out),
        success: bb6,
        error: None,
        span,
    });

    // bb6: Switch default-fallthrough (11 => bb8, default => bb7)
    let mut b6 = BasicBlock::new(bb6);
    b6.terminator = Some(Terminator::Switch {
        discr: Operand::Copy(Place::Local(call_out)),
        cases: vec![(11, bb8)],
        default: bb7,
        span,
    });

    // bb7: Call non-fallthrough -> bb9
    let mut b7 = BasicBlock::new(bb7);
    b7.terminator = Some(Terminator::Call {
        callee: "id_plus_one".into(),
        args: vec![Operand::Copy(Place::Local(call_out))],
        destination: Place::Local(call_out),
        success: bb9,
        error: None,
        span,
    });

    // bb8: Switch non-fallthrough (11 => bb9, default => bb10)
    let mut b8 = BasicBlock::new(bb8);
    b8.terminator = Some(Terminator::Switch {
        discr: Operand::Copy(Place::Local(call_out)),
        cases: vec![(11, bb9)],
        default: bb10,
        span,
    });

    // bb9: Cond then == else non-fallthrough -> bb10
    let mut b9 = BasicBlock::new(bb9);
    b9.terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(c_zero)),
        then_block: bb10,
        else_block: bb10,
        span,
    });

    // bb_selfloop: empty self-loop `Goto(bb_selfloop)` MUST NOT be elided!
    let mut b_self = BasicBlock::new(bb_selfloop);
    b_self.terminator = Some(Terminator::Goto(bb_selfloop, span));

    // bb10: trailing block
    let mut b10 = BasicBlock::new(bb10);
    b10.terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(call_out))),
        span,
    ));

    f_cfg
        .blocks
        .extend([b1, b2, b3, b4, b5, b6, b7, b8, b9, b_self, b10]);

    let bc_callee = MirToBytecodeCompiler::compile(&callee).expect("compile id_plus_one");
    let mut direct = HashMap::new();
    direct.insert("id_plus_one".to_string(), 0usize);
    let bc_cfg = MirToBytecodeCompiler::compile_with_direct_callee_indices(&f_cfg, &direct)
        .expect("compile cfg_edge_cases");
    let insns = &bc_cfg.chunk.instructions;

    assert_no_fallthrough_jump0(&bc_cfg);
    assert_all_jumps_land_on_block_entries(&bc_cfg);
    assert_eq!(bc_cfg.block_entry_pcs.len(), f_cfg.blocks.len());
    assert_eq!(bc_cfg.terminator_pcs.len(), f_cfg.blocks.len());

    // Helper to resolve jump target PC at instruction index `pc`
    let jump_target = |pc: usize| -> usize {
        match insns[pc] {
            Instruction::Jump(off)
            | Instruction::JumpIf(_, off)
            | Instruction::JumpIfNot(_, off) => (pc as isize + 1 + off as isize) as usize,
            other => panic!("Expected jump instruction at pc {pc}, got {other:?}"),
        }
    };

    // 1. Block 0 fallthrough -> bb1 starts right at bb0 terminator PC
    assert_eq!(bc_cfg.block_entry_pcs[&0], 0);
    assert_eq!(bc_cfg.terminator_pcs[&0], bc_cfg.block_entry_pcs[&bb1.0]);

    // 2. Empty block bb1 fallthrough -> bb1 entry == bb1 term == bb2 entry
    assert_eq!(bc_cfg.block_entry_pcs[&bb1.0], bc_cfg.terminator_pcs[&bb1.0]);
    assert_eq!(bc_cfg.block_entry_pcs[&bb1.0], bc_cfg.block_entry_pcs[&bb2.0]);

    // 3. bb2 Cond then-fallthrough -> JumpIfNot to bb4
    let b2_term = bc_cfg.terminator_pcs[&bb2.0];
    assert!(matches!(insns[b2_term], Instruction::JumpIfNot(..)));
    assert_eq!(jump_target(b2_term), bc_cfg.block_entry_pcs[&bb4.0]);
    assert_eq!(b2_term + 1, bc_cfg.block_entry_pcs[&bb3.0]);

    // 4. bb3 Cond then == else fallthrough -> zero jump emitted, falls directly to bb4
    assert_eq!(bc_cfg.block_entry_pcs[&bb3.0], bc_cfg.block_entry_pcs[&bb4.0]);

    // 5. bb4 Cond else-fallthrough -> JumpIf to bb_selfloop, falls through to bb5
    let b4_term = bc_cfg.terminator_pcs[&bb4.0];
    assert!(matches!(insns[b4_term], Instruction::JumpIf(..)));
    assert_eq!(jump_target(b4_term), bc_cfg.block_entry_pcs[&bb_selfloop.0]);
    assert_eq!(b4_term + 1, bc_cfg.block_entry_pcs[&bb5.0]);

    // 6. bb5 Call success-fallthrough -> instruction right before bb6 is Call (no Jump)
    let b6_entry = bc_cfg.block_entry_pcs[&bb6.0];
    assert!(matches!(insns[b6_entry - 1], Instruction::Call(..)));

    // 7. bb6 Switch default-fallthrough -> instruction right before bb7 is JumpIf to bb8 (no default Jump)
    let b7_entry = bc_cfg.block_entry_pcs[&bb7.0];
    assert!(matches!(insns[b7_entry - 1], Instruction::JumpIf(..)));
    assert_eq!(jump_target(b7_entry - 1), bc_cfg.block_entry_pcs[&bb8.0]);

    // 8. bb7 Call non-fallthrough -> trailing Jump to bb9 right before bb8
    let b8_entry = bc_cfg.block_entry_pcs[&bb8.0];
    assert!(matches!(insns[b8_entry - 1], Instruction::Jump(..)));
    assert_eq!(jump_target(b8_entry - 1), bc_cfg.block_entry_pcs[&bb9.0]);

    // 9. bb8 Switch non-fallthrough -> trailing Jump to bb10 right before bb9
    let b9_entry = bc_cfg.block_entry_pcs[&bb9.0];
    assert!(matches!(insns[b9_entry - 1], Instruction::Jump(..)));
    assert_eq!(jump_target(b9_entry - 1), bc_cfg.block_entry_pcs[&bb10.0]);

    // 10. bb9 Cond then == else non-fallthrough -> Jump directly to bb10
    let b9_term = bc_cfg.terminator_pcs[&bb9.0];
    assert!(matches!(insns[b9_term], Instruction::Jump(..)));
    assert_eq!(jump_target(b9_term), bc_cfg.block_entry_pcs[&bb10.0]);

    // 11. bb_selfloop: Goto(self) MUST NOT be elided; emits Jump(-1) jumping back to itself!
    let self_entry = bc_cfg.block_entry_pcs[&bb_selfloop.0];
    assert_eq!(self_entry, bc_cfg.terminator_pcs[&bb_selfloop.0]);
    assert!(
        matches!(insns[self_entry], Instruction::Jump(-1)),
        "Self-loop Goto(self) must emit Jump(-1), got {:?}:\n{}",
        insns[self_entry],
        bc_cfg.disassemble()
    );
    assert_eq!(jump_target(self_entry), self_entry);

    // 12. Trailing block bb10: Return at final instruction
    let b10_entry = bc_cfg.block_entry_pcs[&bb10.0];
    assert_eq!(b10_entry, insns.len() - 1);
    assert!(matches!(insns[b10_entry], Instruction::Return(..)));

    // Execute reachable runtime paths to confirm CFG fidelity
    let mut vm = VM::new();
    vm.load_functions(vec![bc_callee, bc_unreach, bc_cfg]);
    assert_eq!(
        vm.call("unreachable_no_poison", vec![]).expect("unreachable_no_poison"),
        Value::Int(15)
    );
    // mode = 0: bb0->bb1->bb2->bb3->bb4->bb5 (call_out = 1) -> bb6 default->bb7 (call_out = 2) -> bb9 -> bb10 => 2
    assert_eq!(
        vm.call("cfg_edge_cases", vec![Value::Int(0)]).expect("mode=0"),
        Value::Int(2)
    );
    // mode = 10: bb0->bb1->bb2->bb4->bb5 (call_out = 11) -> bb6 case 11->bb8 case 11->bb9->bb10 => 11
    assert_eq!(
        vm.call("cfg_edge_cases", vec![Value::Int(10)]).expect("mode=10"),
        Value::Int(11)
    );
}



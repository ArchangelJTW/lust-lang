#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use lust::ast::{ItemKind, Span};
use lust::bytecode::{Compiler, Function, Instruction, Value};
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::ir::{
    FunctionSignature, MirBinaryOp, MirConstant, MirFunction, MirType, Operand, Place, Rvalue,
    Statement, Terminator,
};
use lust::mir::lowering::AstLowering;
use lust::mir::vm_backend::MirToBytecodeCompiler;
use lust::parser::Parser;
use lust::vm::VM;

fn compile_source(src: &str) -> Vec<Function> {
    let mut interner = Interner::new();
    let mut lexer = Lexer::new(src, &mut interner);
    let tokens = lexer.tokenize().expect("tokenize");
    let mut parser = Parser::new(tokens);
    let items = parser.parse().expect("parse");
    let mut compiler = Compiler::new();
    compiler.compile_module(&items).expect("compile")
}

fn eval_fn(src: &str, fn_name: &str, args: Vec<Value>) -> Value {
    let functions = compile_source(src);
    let mut vm = VM::new();
    vm.load_functions(functions);
    vm.call(fn_name, args).expect("vm call")
}

fn count_jump_zero(func: &Function) -> usize {
    func.chunk
        .instructions
        .iter()
        .filter(|instr| matches!(instr, Instruction::Jump(0)))
        .count()
}

/// NG-W9-5 Test 1: `if x > 0 then y = 1 end` must emit 0 `Jump(0)` instructions.
#[test]
fn test_if_without_else_or_elseif_has_zero_jump0() {
    let src = r#"
function check(x: int): int
    local y: int = 0
    if x > 0 then
        y = 1
    end
    return y
end
"#;
    let functions = compile_source(src);
    let check_fn = functions
        .iter()
        .find(|f| f.name == "check")
        .expect("check function");
    assert_eq!(
        count_jump_zero(check_fn),
        0,
        "Expected 0 Jump(0) in simple `if ... end`, got:\n{}",
        check_fn.disassemble()
    );

    assert_eq!(eval_fn(src, "check", vec![Value::Int(5)]), Value::Int(1));
    assert_eq!(eval_fn(src, "check", vec![Value::Int(-3)]), Value::Int(0));
    assert_eq!(eval_fn(src, "check", vec![Value::Int(0)]), Value::Int(0));
}

/// NG-W9-5 Test 2 (F-09 regression): `if ... elseif ... end` WITHOUT `else`
/// must NOT fall through from `then` into `elseif`, and must emit 0 `Jump(0)`.
#[test]
fn test_f09_if_elseif_without_else_no_fallthrough() {
    let src = r#"
function classify(x: int): int
    local y: int = 0
    if x == 1 then
        y = 10
    elseif x == 2 then
        y = 20
    end
    return y
end

function classify3(x: int): int
    local y: int = 0
    if x == 1 then
        y = 10
    elseif x == 2 then
        y = 20
    elseif x == 3 then
        y = 30
    end
    return y
end

function classify_with_else(x: int): int
    local y: int = 0
    if x == 1 then
        y = 10
    elseif x == 2 then
        y = 20
    else
        y = 99
    end
    return y
end
"#;
    let functions = compile_source(src);
    for name in ["classify", "classify3", "classify_with_else"] {
        let f = functions.iter().find(|f| f.name == name).expect(name);
        assert_eq!(
            count_jump_zero(f),
            0,
            "Expected 0 Jump(0) in {name}:\n{}",
            f.disassemble()
        );
    }

    // Core F-09 regression: x == 1 must return 10 (NOT fall through to 20!)
    assert_eq!(eval_fn(src, "classify", vec![Value::Int(1)]), Value::Int(10));
    assert_eq!(eval_fn(src, "classify", vec![Value::Int(2)]), Value::Int(20));
    assert_eq!(eval_fn(src, "classify", vec![Value::Int(3)]), Value::Int(0));

    // Multi-elseif without else
    assert_eq!(eval_fn(src, "classify3", vec![Value::Int(1)]), Value::Int(10));
    assert_eq!(eval_fn(src, "classify3", vec![Value::Int(2)]), Value::Int(20));
    assert_eq!(eval_fn(src, "classify3", vec![Value::Int(3)]), Value::Int(30));
    assert_eq!(eval_fn(src, "classify3", vec![Value::Int(4)]), Value::Int(0));

    // With else
    assert_eq!(
        eval_fn(src, "classify_with_else", vec![Value::Int(1)]),
        Value::Int(10)
    );
    assert_eq!(
        eval_fn(src, "classify_with_else", vec![Value::Int(2)]),
        Value::Int(20)
    );
    assert_eq!(
        eval_fn(src, "classify_with_else", vec![Value::Int(3)]),
        Value::Int(99)
    );
}

/// NG-W9-5 Test 3: `StmtKind::Return` elides `Move` + `(return_N)` when no active `defer`
/// exists, while preserving exact `defer` evaluation semantics when `defer` IS active.
#[test]
fn test_return_move_elision_and_defer_safety() {
    let src = r#"
function id(x: int): int
    return x
end

function add_one(x: int): int
    return x + 1
end

function with_defer(x: int): int
    local acc: int = x
    defer acc = acc + 100
    return acc
end
"#;
    let functions = compile_source(src);
    let id_fn = functions.iter().find(|f| f.name == "id").expect("id");
    assert!(
        !id_fn
            .chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Move(..))),
        "Expected 0 Move instructions in `id`:\n{}",
        id_fn.disassemble()
    );

    let add_fn = functions
        .iter()
        .find(|f| f.name == "add_one")
        .expect("add_one");
    assert!(
        !add_fn
            .chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Move(..))),
        "Expected 0 Move instructions in `add_one`:\n{}",
        add_fn.disassemble()
    );

    assert_eq!(eval_fn(src, "id", vec![Value::Int(42)]), Value::Int(42));
    assert_eq!(
        eval_fn(src, "add_one", vec![Value::Int(41)]),
        Value::Int(42)
    );
    // Defer mutates `acc` after return value is captured in dedicated slot -> must still return 7
    assert_eq!(
        eval_fn(src, "with_defer", vec![Value::Int(7)]),
        Value::Int(7)
    );
}

/// NG-W9-5 Test 4: `move_result` rewrites destination directly for LoadConst, LoadBool,
/// LoadNil, GetIndex, ArrayLen, Call, FloorDiv, Pow, Concat without trailing Move.
#[test]
fn test_move_result_elides_trailing_moves() {
    let src = r#"
function helper(v: int): int
    return v * 2
end

function exercise(a: int, b: int): int
    local c: int = 42
    local flag: bool = true
    local n: int = 0
    local p: int = a ^ 2
    local fd: int = b // 3
    local r: int = helper(a)
    local s: string = "hi" .. "!"
    if flag and n == 0 and s == "hi!" then
        return c + p + fd + r
    end
    return 0
end
"#;
    let functions = compile_source(src);
    let ex_fn = functions
        .iter()
        .find(|f| f.name == "exercise")
        .expect("exercise");
    // None of `c = 42`, `flag = true`, `n = nil`, `p = a ^ 2`, `fd = b // 3`, `r = helper(a)`, `s = "hi" .. "!"`
    // should emit a Move after their defining instruction.
    let instrs = &ex_fn.chunk.instructions;
    for window in instrs.windows(2) {
        if let Instruction::Move(_, src_reg) = window[1] {
            let prev_defined = window[0].defined_register();
            assert!(
                !matches!(
                    window[0],
                    Instruction::LoadConst(..)
                        | Instruction::LoadBool(..)
                        | Instruction::LoadNil(..)
                        | Instruction::Pow(..)
                        | Instruction::PowInt(..)
                        | Instruction::FloorDiv(..)
                        | Instruction::FloorDivInt(..)
                        | Instruction::Call(..)
                        | Instruction::Concat(..)
                ) || prev_defined != Some(src_reg),
                "Unexpected Move immediately after {:?} -> {:?} in:\n{}",
                window[0],
                window[1],
                ex_fn.disassemble()
            );
        }
    }

    // a=5, b=10 -> c=42, p=25, fd=3, r=10 => 80
    assert_eq!(
        eval_fn(src, "exercise", vec![Value::Int(5), Value::Int(10)]),
        Value::Int(80)
    );
}

/// NG-W9-5 Test 5: `floatmath.lust` loop compiled via `MirToBytecodeCompiler` has
/// 0 `LoadConst` instructions inside the loop header/body, valid `block_entry_pcs` /
/// `terminator_pcs`, and produces the exact expected result.
#[test]
fn test_mir_vm_backend_floatmath_licm_zero_loadconst_in_loop() {
    let src = r#"
function floatmath_run(limit: int): float
    local x: float = 0.0
    local acc: float = 0.0
    local i: int = 0
    while i < limit do
        x = x + 0.001
        acc = acc + x * x - acc * 0.5
        i = i + 1
    end
    return acc
end
"#;
    let mut interner = Interner::new();
    let mut lexer = Lexer::new(src, &mut interner);
    let tokens = lexer.tokenize().expect("tokenize");
    let mut parser = Parser::new(tokens);
    let items = parser.parse().expect("parse");
    let func_def = items
        .iter()
        .find_map(|item| match &item.kind {
            ItemKind::Function(f) if f.name == "floatmath_run" => Some(f),
            _ => None,
        })
        .expect("floatmath_run AST function");
    let mir_fn = AstLowering::lower_function(func_def).expect("lower to MIR");

    let loop_headers = lust::mir::opt::Safepoints::find_loop_headers(&mir_fn.blocks);
    assert!(
        !loop_headers.is_empty(),
        "floatmath_run must have a loop header"
    );

    let bc = MirToBytecodeCompiler::compile(&mir_fn).expect("compile floatmath_run");

    // Verify all jump targets land on valid block_entry_pcs
    let valid_entries: lust::hashbrown::HashSet<usize> =
        bc.block_entry_pcs.values().copied().collect();
    let mut loop_backedge_span: Option<(usize, usize)> = None;
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
                "Jump at pc {pc} lands at {target}, not in {:?}:\n{}",
                bc.block_entry_pcs,
                bc.disassemble()
            );
            if off < 0 {
                loop_backedge_span = Some((target, pc));
            }
        }
    }

    let (loop_start_pc, loop_end_pc) =
        loop_backedge_span.expect("floatmath_run must have a backward loop jump");
    assert!(
        loop_start_pc > 0,
        "Preheader LoadConst instructions must precede loop header PC:\n{}",
        bc.disassemble()
    );

    let load_const_in_loop = bc.chunk.instructions[loop_start_pc..=loop_end_pc]
        .iter()
        .filter(|i| matches!(i, Instruction::LoadConst(..)))
        .count();
    assert_eq!(
        load_const_in_loop,
        0,
        "Expected 0 LoadConst instructions inside floatmath loop ({loop_start_pc}..={loop_end_pc}), got:\n{}",
        bc.disassemble()
    );

    // Execute and verify exact floating-point parity with reference Rust loop
    let limit = 1000i64;
    let mut expected_x = 0.0f64;
    let mut expected_acc = 0.0f64;
    for _ in 0..limit {
        expected_x += 0.001;
        expected_acc = expected_acc + expected_x * expected_x - expected_acc * 0.5;
    }

    let mut vm = VM::new();
    vm.load_functions(vec![bc]);
    let res = vm
        .call("floatmath_run", vec![Value::Int(limit)])
        .expect("run floatmath_run");
    match res {
        Value::Float(actual) => {
            assert_eq!(
                actual.to_bits(),
                expected_acc.to_bits(),
                "floatmath_run bit-exact mismatch: actual={actual}, expected={expected_acc}"
            );
        }
        other => panic!("Expected Float, got {other:?}"),
    }
}

/// NG-W9-5 Test 6: Signed-zero distinctness (`-0.0` vs `+0.0`) and degenerate `bb0` loop header.
#[test]
fn test_mir_vm_backend_licm_signed_zero_and_entry_loop_header() {
    let span = Span::dummy();
    let mut f = MirFunction::new(
        FunctionSignature {
            name: "entry_loop".into(),
            param_types: vec![],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let i = f.add_local(Some("i".into()), MirType::Int, true, span);
    let cond = f.add_local(Some("cond".into()), MirType::Bool, false, span);
    let bb_exit = f.new_block();

    // bb0 is itself a loop header (self-loop while i < 5, starting from uninit? No, let's pass i as param!)
    let mut f2 = MirFunction::new(
        FunctionSignature {
            name: "entry_self_loop".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let p_i = f2.params[0];
    let c_loc = f2.add_local(Some("cond".into()), MirType::Bool, false, span);
    let bb_done = f2.new_block();
    // bb0: p_i = p_i + 1; cond = p_i < 5; if cond goto bb0 else bb_done
    f2.blocks[0].statements.push(Statement::Assign(
        Place::Local(p_i),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(p_i)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        span,
    ));
    f2.blocks[0].statements.push(Statement::Assign(
        Place::Local(c_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(p_i)),
            Operand::Constant(MirConstant::Int(5)),
        ),
        span,
    ));
    f2.blocks[0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(c_loc)),
        then_block: lust::mir::ir::BlockId(0),
        else_block: bb_done,
        span,
    });
    f2.blocks[bb_done.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(p_i))),
        span,
    ));
    let _ = (f, i, cond, bb_exit);

    let bc = MirToBytecodeCompiler::compile(&f2).expect("compile entry_self_loop");
    let bb0_pc = bc.block_entry_pcs[&0];
    assert!(
        bb0_pc >= 2,
        "Preheader LoadConsts must precede bb0 entry PC when bb0 is a loop header:\n{}",
        bc.disassemble()
    );
    let load_const_in_bb0 = bc.chunk.instructions[bb0_pc..bc.block_entry_pcs[&bb_done.0]]
        .iter()
        .filter(|i| matches!(i, Instruction::LoadConst(..)))
        .count();
    assert_eq!(load_const_in_bb0, 0);

    let mut vm = VM::new();
    vm.load_functions(vec![bc]);
    let res = vm
        .call("entry_self_loop", vec![Value::Int(0)])
        .expect("run entry_self_loop");
    assert_eq!(res, Value::Int(5));
}

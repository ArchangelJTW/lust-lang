#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! NG-W14-2 Acceptance & Adversarial Verification Suite:
//! 1. Default AST bytecode compiler (`use_mir = false`) emits `BranchLtInt`, `BranchLeInt`,
//!    and `AddIntImm` for canonical `while` loops and preserves exact runtime + gas parity.
//! 2. Default AST bytecode compiler hoists loop-invariant scalar constants (`Int`/`Float`)
//!    out of simple `while` loops while preserving `defer`/`errdefer`, closures, `and`/`or`,
//!    `reduce`, and fail-closed dynamic/unknown typing.
//! 3. MIR bytecode compilation in `Compiler::compile_module` (`use_mir = true`, non-deopt path)
//!    wires `use_field_slots` while preserving `direct_callee_indices`, `block_entry_pcs`,
//!    `terminator_pcs`, and `LuaTable` exclusion.

use lust::ast::{Span, StructDef, Type, TypeKind};
use lust::bytecode::compiler::Compiler;
use lust::bytecode::{FieldStorage, Function, Instruction, StructLayout, Value};
use lust::config::LustConfig;
use lust::hashbrown::{HashMap, HashSet};
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::ir::{
    FunctionSignature, MirFunction, MirType, Operand, Place, Rvalue, Statement, Terminator,
};
use lust::mir::vm_backend::{audit_chunk_field_access, MirToBytecodeCompiler};
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{VmBackend, VM};
use std::rc::Rc;

fn compile_module_with_mode(source: &str, use_mir: bool) -> (Vec<Function>, HashMap<String, StructDef>) {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize().expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    config.set_use_mir(use_mir);

    let mut checker = TypeChecker::with_config(&config);
    checker.check_module(&items).expect("typecheck");
    let struct_defs = checker.struct_definitions();

    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_option_coercions(checker.take_option_coercions());
    compiler.set_checked_array_indices(checker.take_checked_array_indices());
    compiler.set_lambda_return_types(checker.take_lambda_return_types());
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    compiler.set_struct_defs(struct_defs.clone());

    let functions = compiler.compile_module(&items).expect("compile_module");
    (functions, struct_defs)
}

fn plain_vm(struct_defs: &HashMap<String, StructDef>, functions: Vec<Function>) -> VM {
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    let mut vm = VM::with_config(&config);
    vm.set_backend(VmBackend::Bytecode);
    vm.register_structs(struct_defs);
    vm.load_functions(functions);
    vm
}

fn find_fn<'a>(functions: &'a [Function], name: &str) -> &'a Function {
    functions
        .iter()
        .find(|f| f.name == name)
        .unwrap_or_else(|| panic!("function '{name}' not found"))
}

fn assert_mir_pcs_valid(bc: &Function) {
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
                "Jump at pc {pc} ({instr}) lands at {target}, not in block_entry_pcs {:?}:\n{}",
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

#[test]
fn test_ng_w14_2_ast_emits_loop_superinstructions() {
    let src = r#"
function count_lt(n: int): int
    local i: int = 0
    while i < n do
        i = i + 1
    end
    return i
end

function sum_le(n: int): int
    local i: int = 1
    local acc: int = 0
    while i <= n do
        acc = acc + i
        i += 1
    end
    return acc
end

function countdown_gt(n: int): int
    local i: int = n
    local steps: int = 0
    while i > 0 do
        steps = steps + 1
        i = i - 1
    end
    return steps
end

function countdown_ge(n: int): int
    local i: int = n
    local steps: int = 0
    while i >= 0 do
        steps += 1
        i -= 1
    end
    return steps
end

function dynamic_while(n: unknown): int
    local i: unknown = 0
    local steps: int = 0
    while i < n do
        i = i + 1
        steps = steps + 1
    end
    return steps
end
"#;

    let (functions, struct_defs) = compile_module_with_mode(src, false);

    // 1. Canonical `while i < n do i = i + 1 end` must emit BranchLtInt + AddIntImm
    let f_lt = find_fn(&functions, "count_lt");
    assert!(
        f_lt.chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::BranchLtInt(..))),
        "AST compiler must emit BranchLtInt for `while i < n`:\n{}",
        f_lt.disassemble()
    );
    assert!(
        f_lt.chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::AddIntImm(_, _, 1))),
        "AST compiler must emit AddIntImm(_, _, 1) for `i = i + 1`:\n{}",
        f_lt.disassemble()
    );
    assert!(
        !f_lt
            .chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::LtInt(..) | Instruction::JumpIfNot(..) | Instruction::AddInt(..))),
        "count_lt should fuse LtInt+JumpIfNot and AddInt:\n{}",
        f_lt.disassemble()
    );

    // 2. `while i <= n` + `i += 1` must emit BranchLeInt + AddIntImm
    let f_le = find_fn(&functions, "sum_le");
    assert!(
        f_le.chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::BranchLeInt(..))),
        "AST compiler must emit BranchLeInt for `while i <= n`:\n{}",
        f_le.disassemble()
    );
    assert!(
        f_le.chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::AddIntImm(_, _, 1))),
        "AST compiler must emit AddIntImm for `i += 1`:\n{}",
        f_le.disassemble()
    );

    // 3. `while i > 0` + `i = i - 1` must emit BranchLtInt + AddIntImm(_, _, -1)
    let f_gt = find_fn(&functions, "countdown_gt");
    assert!(
        f_gt.chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::BranchLtInt(..))),
        "AST compiler must emit BranchLtInt for `while i > 0`:\n{}",
        f_gt.disassemble()
    );
    assert!(
        f_gt.chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::AddIntImm(_, _, -1))),
        "AST compiler must emit AddIntImm(_, _, -1) for `i = i - 1`:\n{}",
        f_gt.disassemble()
    );

    // 4. `while i >= 0` + `i -= 1` must emit BranchLeInt + AddIntImm(_, _, -1)
    let f_ge = find_fn(&functions, "countdown_ge");
    assert!(
        f_ge.chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::BranchLeInt(..))),
        "AST compiler must emit BranchLeInt for `while i >= 0`:\n{}",
        f_ge.disassemble()
    );
    assert!(
        f_ge.chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::AddIntImm(_, _, -1))),
        "AST compiler must emit AddIntImm(_, _, -1) for `i -= 1`:\n{}",
        f_ge.disassemble()
    );

    // 5. Fail-closed: `unknown` loop condition/add MUST NOT emit BranchLtInt/BranchLeInt
    let f_dyn = find_fn(&functions, "dynamic_while");
    assert!(
        !f_dyn
            .chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::BranchLtInt(..) | Instruction::BranchLeInt(..))),
        "unknown operands must never emit BranchLtInt/BranchLeInt:\n{}",
        f_dyn.disassemble()
    );

    // 6. Exact runtime parity + bb_gas_weights vs per-instruction gas parity + SEC-01 budget check
    let mut vm_fast = plain_vm(&struct_defs, functions.clone());
    assert_eq!(vm_fast.call("count_lt", vec![Value::Int(100)]).unwrap(), Value::Int(100));
    assert_eq!(vm_fast.call("count_lt", vec![Value::Int(0)]).unwrap(), Value::Int(0));
    assert_eq!(vm_fast.call("sum_le", vec![Value::Int(100)]).unwrap(), Value::Int(5050));
    assert_eq!(vm_fast.call("countdown_gt", vec![Value::Int(10)]).unwrap(), Value::Int(10));
    assert_eq!(vm_fast.call("countdown_ge", vec![Value::Int(10)]).unwrap(), Value::Int(11));
    assert_eq!(vm_fast.call("dynamic_while", vec![Value::Float(3.5)]).unwrap(), Value::Int(4));

    vm_fast.reset_gas_counter();
    let _ = vm_fast.call("sum_le", vec![Value::Int(50)]).unwrap();
    let gas_fast = vm_fast.gas_used();

    let mut vm_slow = plain_vm(&struct_defs, functions);
    vm_slow.set_instruction_hook(Some(Box::new(|_, _, _| {})));
    vm_slow.reset_gas_counter();
    assert_eq!(vm_slow.call("sum_le", vec![Value::Int(50)]).unwrap(), Value::Int(1275));
    let gas_slow = vm_slow.gas_used();
    vm_slow.clear_trace_hooks();
    assert_eq!(
        gas_fast, gas_slow,
        "AST superinstruction bb_gas_weights ({gas_fast}) must match per-instruction gas ({gas_slow})"
    );

    // SEC-01 exact boundary check
    vm_fast.reset_gas_counter();
    vm_fast.set_gas_budget(gas_fast - 1);
    let err = vm_fast
        .call("sum_le", vec![Value::Int(50)])
        .expect_err("budget = gas - 1 must fail with Out of gas");
    assert!(err.to_string().contains("Out of gas"), "unexpected error: {err}");
    vm_fast.reset_gas_counter();
    vm_fast.set_gas_budget(gas_fast);
    assert_eq!(vm_fast.call("sum_le", vec![Value::Int(50)]).unwrap(), Value::Int(1275));
}

#[test]
fn test_ng_w14_2_ast_hoists_scalar_constants_and_preserves_semantics() {
    let src = r#"
function floatmath_small(): float
    local x: float = 0.0
    local acc: float = 0.0
    local i: int = 0
    while i < 500 do
        x = x + 0.001
        acc = acc + x * x - acc * 0.5
        i = i + 1
    end
    return acc
end

function nested_small(): int
    local total: int = 0
    local i: int = 0
    while i < 20 do
        local j: int = 0
        while j < 20 do
            total = total + (i * j) % 7
            j = j + 1
        end
        i = i + 1
    end
    return total
end

function with_defer(n: int): int
    local sum: int = 0
    local i: int = 0
    while i < n do
        defer sum = sum + 10
        sum = sum + i
        i = i + 1
    end
    return sum
end

function with_closure(n: int): int
    local funcs: Array<function(): int> = []
    local i: int = 0
    while i < n do
        local captured: int = i
        array.push(funcs, function(): int return captured + 100 end)
        i = i + 1
    end
    local total: int = 0
    local j: int = 0
    while j < array.len(funcs) do
        if funcs[j] is Ok(f) then
            total = total + f()
        end
        j = j + 1
    end
    return total
end

function with_logic_and_reduce(): int
    local total: int = 0
    local i: int = 0
    local arr: Array<int> = [1, 2, 3, 4]
    while i < 5 do
        local a: bool = false
        local b: bool = a or (i < 10)
        if b and (i < 10) then
            local sum: int = 0
            local k: int = 0
            while k < array.len(arr) do
                if arr[k] is Ok(v) then
                    sum = sum + v
                end
                k = k + 1
            end
            total = total + sum
        end
        i = i + 1
    end
    return total
end
"#;

    let (functions, struct_defs) = compile_module_with_mode(src, false);

    // Verify floatmath_small has ZERO LoadConst inside the loop [BranchLtInt .. backedge Jump]
    let f_fm = find_fn(&functions, "floatmath_small");
    let br_pc = f_fm
        .chunk
        .instructions
        .iter()
        .position(|i| matches!(i, Instruction::BranchLtInt(..)))
        .expect("floatmath_small must emit BranchLtInt");
    let back_pc = f_fm
        .chunk
        .instructions
        .iter()
        .rposition(|i| matches!(i, Instruction::Jump(off) if *off < 0))
        .expect("floatmath_small must have backedge Jump");
    let loop_load_consts = f_fm.chunk.instructions[br_pc..=back_pc]
        .iter()
        .filter(|i| matches!(i, Instruction::LoadConst(..)))
        .count();
    assert_eq!(
        loop_load_consts, 0,
        "floatmath_small loop body must have 0 LoadConst after LICM + AddIntImm:\n{}",
        f_fm.disassemble()
    );

    // Verify nested_small inner loop has ZERO LoadConst inside [inner BranchLtInt .. inner backedge Jump]
    let f_nested = find_fn(&functions, "nested_small");
    let branch_pcs: Vec<usize> = f_nested
        .chunk
        .instructions
        .iter()
        .enumerate()
        .filter_map(|(pc, i)| matches!(i, Instruction::BranchLtInt(..)).then_some(pc))
        .collect();
    assert_eq!(branch_pcs.len(), 2, "nested_small must emit 2 BranchLtInt instructions");
    let inner_br = branch_pcs[1];
    let inner_back = f_nested.chunk.instructions[inner_br..]
        .iter()
        .position(|i| matches!(i, Instruction::Jump(off) if *off < 0))
        .map(|rel| inner_br + rel)
        .expect("nested_small inner loop must have backedge Jump");
    let inner_load_consts = f_nested.chunk.instructions[inner_br..=inner_back]
        .iter()
        .filter(|i| matches!(i, Instruction::LoadConst(..)))
        .count();
    assert_eq!(
        inner_load_consts, 0,
        "nested_small inner loop must have 0 LoadConst after LICM + AddIntImm:\n{}",
        f_nested.disassemble()
    );

    // Execute all functions and verify exact parity
    let mut vm = plain_vm(&struct_defs, functions);

    let mut expected_x = 0.0f64;
    let mut expected_acc = 0.0f64;
    for _ in 0..500 {
        expected_x += 0.001;
        expected_acc = expected_acc + expected_x * expected_x - expected_acc * 0.5;
    }
    let Value::Float(got_acc) = vm.call("floatmath_small", vec![]).unwrap() else {
        panic!("expected float result from floatmath_small");
    };
    assert!((got_acc - expected_acc).abs() < 1e-12, "got {got_acc}, expected {expected_acc}");

    let expected_nested: i64 = (0..20i64)
        .flat_map(|i| (0..20i64).map(move |j| (i * j) % 7))
        .sum();
    assert_eq!(
        vm.call("nested_small", vec![]).unwrap(),
        Value::Int(expected_nested)
    );

    // with_defer(4): sum_{i=0..4} (i + 10) = 6 + 40 = 46
    assert_eq!(vm.call("with_defer", vec![Value::Int(4)]).unwrap(), Value::Int(46));

    // with_closure(4): (0+100) + (1+100) + (2+100) + (3+100) = 406
    assert_eq!(vm.call("with_closure", vec![Value::Int(4)]).unwrap(), Value::Int(406));

    // with_logic_and_reduce(): 5 * (1+2+3+4) = 50
    assert_eq!(vm.call("with_logic_and_reduce", vec![]).unwrap(), Value::Int(50));
}

#[test]
fn test_ng_w14_2_mir_bytecode_enables_field_slots() {
    let src = r#"
struct Point
    x: int
    y: int
end

struct Counter
    value: int
end

impl Counter
    function bump(self, d: int): int
        self.value = self.value + d
        return self.value
    end
end

function run_fields(n: int): int
    local p: Point = Point { x = 0, y = 0 }
    local i: int = 1
    while i <= n do
        p.x = i
        p.y = p.y + i
        i = i + 1
    end
    return p.x + p.y
end

function run_methods(n: int): int
    local c: Counter = Counter { value = 0 }
    local last: int = 0
    local i: int = 1
    while i <= n do
        last = c:bump(i)
        i = i + 1
    end
    return last
end
"#;

    let (functions, struct_defs) = compile_module_with_mode(src, true);

    let f_fields = find_fn(&functions, "run_fields");
    assert!(
        f_fields
            .chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::NewStruct(..))),
        "MIR bytecode run_fields must emit NewStruct:\n{}",
        f_fields.disassemble()
    );
    assert!(
        f_fields
            .chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::GetFieldSlot(..))),
        "MIR bytecode run_fields must emit GetFieldSlot:\n{}",
        f_fields.disassemble()
    );
    assert!(
        f_fields
            .chunk
            .instructions
            .iter()
            .any(|i| matches!(
                i,
                Instruction::SetFieldSlot(..) | Instruction::GetFieldAddSetInt(..)
            )),
        "MIR bytecode run_fields must emit SetFieldSlot / GetFieldAddSetInt:\n{}",
        f_fields.disassemble()
    );
    assert!(
        audit_chunk_field_access(&f_fields.chunk).has_zero_name_lookup(),
        "MIR bytecode run_fields must have zero GetField/SetField name lookups:\n{}",
        f_fields.disassemble()
    );
    assert_mir_pcs_valid(f_fields);

    let f_bump = find_fn(&functions, "Counter:bump");
    assert!(
        f_bump
            .chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::GetFieldSlot(..)))
            && f_bump.chunk.instructions.iter().any(|i| matches!(
                i,
                Instruction::SetFieldSlot(..) | Instruction::GetFieldAddSetInt(..)
            )),
        "MIR bytecode Counter:bump must emit slot field opcodes:\n{}",
        f_bump.disassemble()
    );
    assert!(
        audit_chunk_field_access(&f_bump.chunk).has_zero_name_lookup(),
        "MIR bytecode Counter:bump must have zero name lookups:\n{}",
        f_bump.disassemble()
    );
    assert_mir_pcs_valid(f_bump);

    // Direct callee index must still be preserved for run_methods -> Counter:bump
    let f_methods = find_fn(&functions, "run_methods");
    assert!(
        !f_methods
            .chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::LoadGlobal(..))),
        "run_methods must preserve direct_callee_indices (no LoadGlobal for Counter:bump):\n{}",
        f_methods.disassemble()
    );
    assert_mir_pcs_valid(f_methods);

    // Execute both functions on VM and assert exact results
    let mut vm = plain_vm(&struct_defs, functions);
    // p.x = 100, p.y = 5050 -> 5150
    assert_eq!(
        vm.call("run_fields", vec![Value::Int(100)]).expect("run_fields"),
        Value::Int(5150)
    );
    assert_eq!(
        vm.call("run_methods", vec![Value::Int(100)]).expect("run_methods"),
        Value::Int(5050)
    );

    // Verify LuaTable is strictly excluded from GetFieldSlot/SetFieldSlot/GetFieldAddSetInt
    let span = Span::dummy();
    let mut lua_fn = MirFunction::new(
        FunctionSignature {
            name: "lua_table_exclusion".into(),
            param_types: vec![MirType::Struct("LuaTable".into())],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let t = lua_fn.params[0];
    let out = lua_fn.add_local(Some("out".into()), MirType::Int, false, span);
    lua_fn.blocks[0].statements.extend([
        Statement::Assign(
            Place::Field(Box::new(Place::Local(t)), 0),
            Rvalue::Use(Operand::Constant(lust::mir::ir::MirConstant::Int(7))),
            span,
        ),
        Statement::Assign(
            Place::Local(out),
            Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(t)), 0))),
            span,
        ),
    ]);
    lua_fn.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(out))),
        span,
    ));

    let mut layouts = HashMap::new();
    layouts.insert(
        "LuaTable".to_string(),
        Rc::new(StructLayout::new(
            "LuaTable".to_string(),
            vec![Rc::new("table".to_string()), Rc::new("metamethods".to_string())],
            vec![FieldStorage::Strong, FieldStorage::Strong],
            vec![Type::new(TypeKind::Unknown, span), Type::new(TypeKind::Unknown, span)],
            vec![None, None],
        )),
    );
    let callee_indices = HashMap::new();
    let bc_lua = MirToBytecodeCompiler::compile_with_field_slots_and_callee_indices(
        &lua_fn,
        &callee_indices,
        &layouts,
    )
    .expect("compile lua_table_exclusion");
    assert!(
        !bc_lua.chunk.instructions.iter().any(|i| matches!(
            i,
            Instruction::GetFieldSlot(..)
                | Instruction::SetFieldSlot(..)
                | Instruction::GetFieldAddSetInt(..)
        )),
        "LuaTable MUST be excluded from GetFieldSlot/SetFieldSlot/GetFieldAddSetInt:\n{}",
        bc_lua.disassemble()
    );
    assert!(
        bc_lua.chunk.instructions.iter().any(|i| matches!(i, Instruction::GetField(..)))
            && bc_lua.chunk.instructions.iter().any(|i| matches!(i, Instruction::SetField(..))),
        "LuaTable must fall back to name-based GetField/SetField:\n{}",
        bc_lua.disassemble()
    );
}

#[test]
fn test_ng_w14_2_ast_and_mir_parity_with_break_continue_and_outer_defer() {
    let src = r#"
function break_continue_loop(n: int): int
    local i: int = 0
    local acc: int = 0
    while i < n do
        i = i + 1
        if i % 2 == 0 then
            continue
        end
        if i > 15 then
            break
        end
        acc = acc + i * 3
    end
    return acc
end

function outer_defer_early_return(target: int): int
    local log: Array<int> = []
    defer array.push(log, 999)
    local i: int = 0
    while i < 50 do
        if i == target then
            return i * 10
        end
        i = i + 1
    end
    return -1
end

function non_loop_int_ops(n: int): bool
    local value = n + 1 + 2
    value += 3
    return value < n * 4
end
"#;

    let (ast_fns, ast_structs) = compile_module_with_mode(src, false);
    let (mir_fns, mir_structs) = compile_module_with_mode(src, true);

    // Non-loop int ops must keep AddInt (preserving existing unit test invariant outside loops)
    let f_non_loop = find_fn(&ast_fns, "non_loop_int_ops");
    assert!(
        f_non_loop
            .chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::AddInt(d, l, _) if d == l)),
        "non-loop int compound assign should preserve AddInt:\n{}",
        f_non_loop.disassemble()
    );

    // break_continue_loop must emit BranchLtInt + AddIntImm and hoist 2, 15, 3
    let f_bc_ast = find_fn(&ast_fns, "break_continue_loop");
    assert!(
        f_bc_ast
            .chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::BranchLtInt(..)))
            && f_bc_ast
                .chunk
                .instructions
                .iter()
                .any(|i| matches!(i, Instruction::AddIntImm(_, _, 1))),
        "break_continue_loop must emit BranchLtInt + AddIntImm:\n{}",
        f_bc_ast.disassemble()
    );

    let mut vm_ast = plain_vm(&ast_structs, ast_fns);
    let mut vm_mir = plain_vm(&mir_structs, mir_fns);

    // Odd numbers in 1..=15: 1+3+5+7+9+11+13+15 = 64 -> * 3 = 192
    for n in [0, 1, 5, 15, 30] {
        let a = vm_ast
            .call("break_continue_loop", vec![Value::Int(n)])
            .expect("ast break_continue_loop");
        let m = vm_mir
            .call("break_continue_loop", vec![Value::Int(n)])
            .expect("mir break_continue_loop");
        assert_eq!(a, m, "AST vs MIR parity mismatch for n={n}");
    }
    assert_eq!(
        vm_ast.call("break_continue_loop", vec![Value::Int(30)]).unwrap(),
        Value::Int(192)
    );
    assert_eq!(
        vm_ast
            .call("outer_defer_early_return", vec![Value::Int(7)])
            .unwrap(),
        Value::Int(70)
    );
    assert_eq!(
        vm_ast
            .call("outer_defer_early_return", vec![Value::Int(100)])
            .unwrap(),
        Value::Int(-1)
    );
    assert_eq!(
        vm_ast
            .call("non_loop_int_ops", vec![Value::Int(4)])
            .unwrap(),
        Value::Bool(true)
    );
}


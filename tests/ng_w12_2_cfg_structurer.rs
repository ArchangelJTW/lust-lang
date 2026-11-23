#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::fs;
use std::path::PathBuf;

use hashbrown::HashSet;
use lust::ast::Span;
use lust::codegen_zig::aot::cfg_structurer::{
    StructuredNode, analyze_cfg, structure_mir_function,
};
use lust::codegen_zig::aot::prepare_items_for_mir;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::ir::*;
use lust::mir::lowering::AstLowering;
use lust::mir::opt::MirOptimizer;
use lust::parser::Parser;

fn lower_and_optimize(module_name: &str, source: &str) -> MirModule {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .unwrap_or_else(|e| panic!("tokenize {module_name}: {e}"));
    let items = Parser::new(tokens)
        .parse()
        .unwrap_or_else(|e| panic!("parse {module_name}: {e}"));
    let prepared = prepare_items_for_mir(&items);
    let mut module = AstLowering::lower_module_items(module_name, &prepared)
        .unwrap_or_else(|e| panic!("lower_module_items {module_name}: {e}"));
    MirOptimizer::optimize_module(&mut module);
    module
}

fn benchmark_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks")
        .join("suite")
        .join(format!("{name}.lust"))
}

// ---------------------------------------------------------------------------
// Semantic Equivalence Oracle: Flat MIR CFG vs StructuredNode Tree
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvalVal {
    Int(i64),
    Bool(bool),
    Unit,
}

impl EvalVal {
    fn as_int(self) -> i64 {
        match self {
            EvalVal::Int(v) => v,
            EvalVal::Bool(b) => i64::from(b),
            EvalVal::Unit => 0,
        }
    }

    fn as_bool(self) -> bool {
        match self {
            EvalVal::Bool(b) => b,
            EvalVal::Int(v) => v != 0,
            EvalVal::Unit => false,
        }
    }
}

fn eval_operand(op: &Operand, locals: &[EvalVal]) -> EvalVal {
    match op {
        Operand::Copy(Place::Local(id)) | Operand::Move(Place::Local(id)) => locals[id.0],
        Operand::Constant(MirConstant::Int(i)) => EvalVal::Int(*i),
        Operand::Constant(MirConstant::Bool(b)) => EvalVal::Bool(*b),
        Operand::Constant(MirConstant::Unit) => EvalVal::Unit,
        other => panic!("unsupported operand in oracle: {other:?}"),
    }
}

fn exec_block_statements(block: &BasicBlock, locals: &mut [EvalVal]) {
    for stmt in &block.statements {
        match stmt {
            Statement::Assign(Place::Local(dst), rval, _) => {
                let val = match rval {
                    Rvalue::Use(op) => eval_operand(op, locals),
                    Rvalue::BinaryOp(op, lhs, rhs) => {
                        let l = eval_operand(lhs, locals);
                        let r = eval_operand(rhs, locals);
                        match op {
                            MirBinaryOp::Add => EvalVal::Int(l.as_int().wrapping_add(r.as_int())),
                            MirBinaryOp::Sub => EvalVal::Int(l.as_int().wrapping_sub(r.as_int())),
                            MirBinaryOp::Mul => EvalVal::Int(l.as_int().wrapping_mul(r.as_int())),
                            MirBinaryOp::Eq => EvalVal::Bool(l.as_int() == r.as_int()),
                            MirBinaryOp::NotEq => EvalVal::Bool(l.as_int() != r.as_int()),
                            MirBinaryOp::Lt => EvalVal::Bool(l.as_int() < r.as_int()),
                            MirBinaryOp::LtEq => EvalVal::Bool(l.as_int() <= r.as_int()),
                            MirBinaryOp::Gt => EvalVal::Bool(l.as_int() > r.as_int()),
                            MirBinaryOp::GtEq => EvalVal::Bool(l.as_int() >= r.as_int()),
                            MirBinaryOp::And => EvalVal::Bool(l.as_bool() && r.as_bool()),
                            MirBinaryOp::Or => EvalVal::Bool(l.as_bool() || r.as_bool()),
                            other => panic!("unsupported binop in oracle: {other:?}"),
                        }
                    }
                    Rvalue::UnaryOp(MirUnaryOp::Neg, op) => {
                        EvalVal::Int(eval_operand(op, locals).as_int().wrapping_neg())
                    }
                    Rvalue::UnaryOp(MirUnaryOp::Not, op) => {
                        EvalVal::Bool(!eval_operand(op, locals).as_bool())
                    }
                    other => panic!("unsupported rvalue in oracle: {other:?}"),
                };
                locals[dst.0] = val;
            }
            Statement::StorageLive(_, _)
            | Statement::StorageDead(_, _)
            | Statement::ResourceCheck(_, _, _) => {}
            other => panic!("unsupported statement in oracle: {other:?}"),
        }
    }
}

fn eval_flat_mir(func: &MirFunction, args: &[EvalVal]) -> (EvalVal, Vec<BlockId>) {
    let mut locals = vec![EvalVal::Int(0); func.locals.len()];
    for (p, &val) in func.params.iter().zip(args.iter()) {
        locals[p.0] = val;
    }
    let mut bb = func.entry;
    let mut trace = Vec::new();
    let mut steps = 0usize;
    loop {
        steps += 1;
        assert!(steps < 100_000, "flat MIR oracle exceeded step budget");
        trace.push(bb);
        let block = &func.blocks[bb.0];
        exec_block_statements(block, &mut locals);
        match block.terminator.as_ref().expect("terminator") {
            Terminator::Goto(target, _) => bb = *target,
            Terminator::Cond {
                cond,
                then_block,
                else_block,
                ..
            } => {
                bb = if eval_operand(cond, &locals).as_bool() {
                    *then_block
                } else {
                    *else_block
                };
            }
            Terminator::Switch {
                discr,
                cases,
                default,
                ..
            } => {
                let dv = eval_operand(discr, &locals).as_int();
                bb = cases
                    .iter()
                    .find(|(k, _)| *k == dv)
                    .map(|(_, t)| *t)
                    .unwrap_or(*default);
            }
            Terminator::Return(opt, _) => {
                let res = opt
                    .as_ref()
                    .map_or(EvalVal::Unit, |o| eval_operand(o, &locals));
                return (res, trace);
            }
            other => panic!("unsupported terminator in flat oracle: {other:?}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flow {
    Normal,
    Break(BlockId),
    Continue(BlockId),
    Return(EvalVal),
}

fn eval_structured_node(
    func: &MirFunction,
    node: &StructuredNode,
    locals: &mut [EvalVal],
    trace: &mut Vec<BlockId>,
    steps: &mut usize,
) -> Flow {
    *steps += 1;
    assert!(*steps < 100_000, "structured oracle exceeded step budget");
    match node {
        StructuredNode::BasicBlock(bb) => {
            trace.push(*bb);
            exec_block_statements(&func.blocks[bb.0], locals);
            Flow::Normal
        }
        StructuredNode::Seq(items) => {
            for item in items {
                let flow = eval_structured_node(func, item, locals, trace, steps);
                if flow != Flow::Normal {
                    return flow;
                }
            }
            Flow::Normal
        }
        StructuredNode::If {
            cond_block,
            cond_op,
            then_branch,
            else_branch,
        } => {
            trace.push(*cond_block);
            exec_block_statements(&func.blocks[cond_block.0], locals);
            if eval_operand(cond_op, locals).as_bool() {
                eval_structured_node(func, then_branch, locals, trace, steps)
            } else if let Some(eb) = else_branch {
                eval_structured_node(func, eb, locals, trace, steps)
            } else {
                Flow::Normal
            }
        }
        StructuredNode::Switch {
            discr_block,
            discr_op,
            cases,
            default,
        } => {
            trace.push(*discr_block);
            exec_block_statements(&func.blocks[discr_block.0], locals);
            let dv = eval_operand(discr_op, locals).as_int();
            if let Some((_, arm)) = cases.iter().find(|(k, _)| *k == dv) {
                eval_structured_node(func, arm, locals, trace, steps)
            } else {
                eval_structured_node(func, default, locals, trace, steps)
            }
        }
        StructuredNode::WhileLoop {
            label,
            header_block,
            cond_op,
            negate_cond,
            body,
        } => loop {
            trace.push(*header_block);
            exec_block_statements(&func.blocks[header_block.0], locals);
            let raw = eval_operand(cond_op, locals).as_bool();
            let keep_going = if *negate_cond { !raw } else { raw };
            if !keep_going {
                return Flow::Normal;
            }
            match eval_structured_node(func, body, locals, trace, steps) {
                Flow::Normal => {}
                Flow::Continue(lbl) if lbl == *label => {}
                Flow::Break(lbl) if lbl == *label => return Flow::Normal,
                other => return other,
            }
        },
        StructuredNode::Loop { label, body } => loop {
            match eval_structured_node(func, body, locals, trace, steps) {
                Flow::Normal => {}
                Flow::Continue(lbl) if lbl == *label => {}
                Flow::Break(lbl) if lbl == *label => return Flow::Normal,
                other => return other,
            }
        },
        StructuredNode::LabeledBlock { label, body } => {
            match eval_structured_node(func, body, locals, trace, steps) {
                Flow::Break(lbl) if lbl == *label => Flow::Normal,
                other => other,
            }
        }
        StructuredNode::Break(lbl) => Flow::Break(*lbl),
        StructuredNode::Continue(lbl) => Flow::Continue(*lbl),
        StructuredNode::Return(bb, opt) => {
            trace.push(*bb);
            exec_block_statements(&func.blocks[bb.0], locals);
            let res = opt
                .as_ref()
                .map_or(EvalVal::Unit, |o| eval_operand(o, locals));
            Flow::Return(res)
        }
        StructuredNode::IfArrayGetOk { .. } => {
            panic!("IfArrayGetOk not expected in scalar oracle test")
        }
        StructuredNode::StateMachineFallback { .. } => {
            panic!("StateMachineFallback must not occur in reducible oracle test")
        }
    }
}

fn eval_structured(func: &MirFunction, tree: &StructuredNode, args: &[EvalVal]) -> (EvalVal, Vec<BlockId>) {
    let mut locals = vec![EvalVal::Int(0); func.locals.len()];
    for (p, &val) in func.params.iter().zip(args.iter()) {
        locals[p.0] = val;
    }
    let mut trace = Vec::new();
    let mut steps = 0usize;
    match eval_structured_node(func, tree, &mut locals, &mut trace, &mut steps) {
        Flow::Return(v) => (v, trace),
        other => panic!("structured execution did not return: {other:?}"),
    }
}

fn assert_all_reachable_blocks_preserved(func: &MirFunction, tree: &StructuredNode) {
    let analysis = analyze_cfg(func);
    let mut referenced = HashSet::new();
    tree.collect_referenced_blocks(&mut referenced);
    for (idx, &is_reach) in analysis.reachable.iter().enumerate() {
        if is_reach {
            assert!(
                referenced.contains(&BlockId(idx)),
                "function '{}': reachable block bb{} was dropped by cfg_structurer!\ntree = {:#?}",
                func.signature.name,
                idx,
                tree
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 1: All 10/10 benchmarks/suite/*.lust structure with 0 StateMachineFallback
// ---------------------------------------------------------------------------

#[test]
fn test_all_10_benchmark_suite_programs_structure_with_zero_fallback() {
    let benchmarks = [
        "strings",
        "fields",
        "tree",
        "fib",
        "nested",
        "methods",
        "calls",
        "array",
        "floatmath",
        "sieve",
    ];

    let mut total_functions = 0usize;
    let mut total_while_loops = 0usize;

    for bench in &benchmarks {
        let path = benchmark_path(bench);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        let module = lower_and_optimize(bench, &source);
        assert!(
            !module.functions.is_empty(),
            "benchmark {bench} lowered to 0 MIR functions"
        );

        for func in &module.functions {
            total_functions += 1;
            let analysis = analyze_cfg(func);
            assert!(
                analysis.is_reducible,
                "benchmark '{bench}' function '{}' is unexpectedly irreducible: {:?}",
                func.signature.name, analysis.irreducible_edges
            );

            let tree = structure_mir_function(func);
            assert_eq!(
                tree.count_state_machine_fallbacks(),
                0,
                "benchmark '{bench}' function '{}' emitted StateMachineFallback!\ntree = {:#?}",
                func.signature.name,
                tree
            );
            assert_all_reachable_blocks_preserved(func, &tree);

            total_while_loops += count_while_loops(&tree);
        }
    }

    assert!(total_functions >= 10, "expected at least 10 benchmark functions");
    assert!(
        total_while_loops >= 10,
        "expected native WhileLoop nodes across benchmarks, got {total_while_loops}"
    );
}

fn count_while_loops(node: &StructuredNode) -> usize {
    match node {
        StructuredNode::WhileLoop { body, .. } => 1 + count_while_loops(body),
        StructuredNode::Seq(items) => items.iter().map(count_while_loops).sum(),
        StructuredNode::If {
            then_branch,
            else_branch,
            ..
        }
        | StructuredNode::IfArrayGetOk {
            then_branch,
            else_branch,
            ..
        } => {
            count_while_loops(then_branch)
                + else_branch.as_deref().map_or(0, count_while_loops)
        }
        StructuredNode::Switch { cases, default, .. } => {
            cases.iter().map(|(_, n)| count_while_loops(n)).sum::<usize>()
                + count_while_loops(default)
        }
        StructuredNode::Loop { body, .. } | StructuredNode::LabeledBlock { body, .. } => {
            count_while_loops(body)
        }
        StructuredNode::BasicBlock(_)
        | StructuredNode::Break(_)
        | StructuredNode::Continue(_)
        | StructuredNode::Return(_, _)
        | StructuredNode::StateMachineFallback { .. } => 0,
    }
}

fn collect_if_array_get_ok<'a>(
    node: &'a StructuredNode,
    out: &mut Vec<(BlockId, LocalId, LocalId, LocalId)>,
) {
    match node {
        StructuredNode::IfArrayGetOk {
            header_block,
            arr_local,
            idx_local,
            binding_local,
            then_branch,
            else_branch,
        } => {
            out.push((*header_block, *arr_local, *idx_local, *binding_local));
            collect_if_array_get_ok(then_branch, out);
            if let Some(eb) = else_branch {
                collect_if_array_get_ok(eb, out);
            }
        }
        StructuredNode::Seq(items) => {
            for item in items {
                collect_if_array_get_ok(item, out);
            }
        }
        StructuredNode::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_if_array_get_ok(then_branch, out);
            if let Some(eb) = else_branch {
                collect_if_array_get_ok(eb, out);
            }
        }
        StructuredNode::Switch { cases, default, .. } => {
            for (_, n) in cases {
                collect_if_array_get_ok(n, out);
            }
            collect_if_array_get_ok(default, out);
        }
        StructuredNode::WhileLoop { body, .. }
        | StructuredNode::Loop { body, .. }
        | StructuredNode::LabeledBlock { body, .. } => {
            collect_if_array_get_ok(body, out);
        }
        StructuredNode::BasicBlock(_)
        | StructuredNode::Break(_)
        | StructuredNode::Continue(_)
        | StructuredNode::Return(_, _)
        | StructuredNode::StateMachineFallback { .. } => {}
    }
}

// ---------------------------------------------------------------------------
// Test 2: IfArrayGetOk recognized in sieve.lust, array.lust, and tree.lust
// ---------------------------------------------------------------------------

#[test]
fn test_if_array_get_ok_recognized_in_sieve_array_tree() {
    for bench in ["sieve", "array", "tree"] {
        let path = benchmark_path(bench);
        let source = fs::read_to_string(&path).expect("read benchmark");
        let module = lower_and_optimize(bench, &source);

        let mut found_nodes = Vec::new();
        for func in &module.functions {
            let tree = structure_mir_function(func);
            assert_eq!(tree.count_state_machine_fallbacks(), 0);
            let mut fn_matches = Vec::new();
            collect_if_array_get_ok(&tree, &mut fn_matches);
            for (hdr, arr_local, idx_local, binding_local) in fn_matches {
                assert!(hdr.0 < func.blocks.len());
                assert!(arr_local.0 < func.locals.len());
                assert!(idx_local.0 < func.locals.len());
                assert!(binding_local.0 < func.locals.len());
                assert!(
                    matches!(func.locals[arr_local.0].ty, MirType::Array(_) | MirType::Unknown),
                    "benchmark '{bench}' fn '{}': arr_local {:?} has unexpected type {:?}",
                    func.signature.name,
                    arr_local,
                    func.locals[arr_local.0].ty
                );
                found_nodes.push((func.signature.name.clone(), hdr, arr_local, idx_local, binding_local));
            }
        }

        assert!(
            !found_nodes.is_empty(),
            "benchmark '{bench}.lust' did not recognize any StructuredNode::IfArrayGetOk!"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3: Inlined functions with two early returns (Hazard 14)
// ---------------------------------------------------------------------------

#[test]
fn test_inlined_two_early_returns_sequential_and_nested_labeled_block_break() {
    // Part A: Sequential two early returns inlined into `main`.
    let src_seq = r#"
        function classify(x: int): int
            if x < 0 then
                return -1
            end
            if x == 0 then
                return 0
            end
            return 1
        end

        function main(a: int, b: int, c: int): int
            local r1: int = classify(a)
            local r2: int = classify(b)
            local r3: int = classify(c)
            return r1 * 100 + r2 * 10 + r3
        end
    "#;
    let mod_seq = lower_and_optimize("inlined_seq", src_seq);
    let main_seq = mod_seq
        .functions
        .iter()
        .find(|f| f.signature.name == "main")
        .expect("main");

    // Verify `classify` was actually inlined (no remaining Call to classify).
    let has_classify_call = main_seq.blocks.iter().any(|b| {
        matches!(
            &b.terminator,
            Some(Terminator::Call { callee, .. }) if callee == "classify"
        )
    });
    assert!(!has_classify_call, "expected classify to be inlined into main");

    let tree_seq = structure_mir_function(main_seq);
    assert_eq!(tree_seq.count_state_machine_fallbacks(), 0);
    assert_all_reachable_blocks_preserved(main_seq, &tree_seq);

    for (a, b, c, expected) in [
        (-5, 0, 7, -99),
        (3, -2, 0, 90),
        (0, 0, 0, 0),
        (1, 2, 3, 111),
    ] {
        let args = [EvalVal::Int(a), EvalVal::Int(b), EvalVal::Int(c)];
        let (flat_v, flat_trace) = eval_flat_mir(main_seq, &args);
        let (struct_v, struct_trace) = eval_structured(main_seq, &tree_seq, &args);
        assert_eq!(flat_v, EvalVal::Int(expected));
        assert_eq!(struct_v, flat_v);
        assert_eq!(struct_trace, flat_trace);
    }

    // Part B: Hazard 14 — Nested branches with two early returns bypassing a shared merge
    // block (`r + x`) before the inlined join block (`v + 1`), requiring `LabeledBlock` + `Break`.
    let src_nested = r#"
        function tricky(x: int, y: int): int
            local r: int = 0
            if x > 0 then
                if y > 0 then
                    return 100
                end
                r = 10
            else
                if y < 0 then
                    return -100
                end
                r = 20
            end
            return r + x
        end

        function main(x: int, y: int): int
            local v: int = tricky(x, y)
            return v + 1
        end
    "#;
    let mod_nested = lower_and_optimize("inlined_nested", src_nested);
    let main_nested = mod_nested
        .functions
        .iter()
        .find(|f| f.signature.name == "main")
        .expect("main");

    let has_tricky_call = main_nested.blocks.iter().any(|b| {
        matches!(
            &b.terminator,
            Some(Terminator::Call { callee, .. }) if callee == "tricky"
        )
    });
    assert!(!has_tricky_call, "expected tricky to be inlined into main");

    let tree_nested = structure_mir_function(main_nested);
    assert_eq!(
        tree_nested.count_state_machine_fallbacks(),
        0,
        "Hazard 14 inlined early-return CFG must have 0 StateMachineFallback:\n{:#?}",
        tree_nested
    );
    assert!(
        tree_nested.count_labeled_blocks() >= 1,
        "expected LabeledBlock for Hazard 14 early returns bypassing shared merge:\n{:#?}",
        tree_nested
    );
    assert!(
        tree_nested.count_breaks() >= 2,
        "expected at least 2 Break nodes for the two early returns:\n{:#?}",
        tree_nested
    );
    assert_all_reachable_blocks_preserved(main_nested, &tree_nested);

    for (x, y, expected) in [
        (1, 2, 101),    // x > 0, y > 0 -> early return 100 -> +1 = 101
        (5, -1, 16),    // x > 0, y <= 0 -> r = 10 -> 10 + 5 + 1 = 16
        (-3, -2, -99),  // x <= 0, y < 0 -> early return -100 -> +1 = -99
        (-4, 0, 17),    // x <= 0, y >= 0 -> r = 20 -> 20 + (-4) + 1 = 17
    ] {
        let args = [EvalVal::Int(x), EvalVal::Int(y)];
        let (flat_v, flat_trace) = eval_flat_mir(main_nested, &args);
        let (struct_v, struct_trace) = eval_structured(main_nested, &tree_nested, &args);
        assert_eq!(flat_v, EvalVal::Int(expected));
        assert_eq!(struct_v, flat_v);
        assert_eq!(struct_trace, flat_trace);
    }
}

// ---------------------------------------------------------------------------
// Test 4: Cooper-Harvey-Kennedy ipdom, loop nesting tree, & WhileLoop header statements
// ---------------------------------------------------------------------------

#[test]
fn test_chk_ipdom_loop_nesting_tree_and_header_statements_preserved() {
    let nested_src = fs::read_to_string(benchmark_path("nested")).expect("read nested.lust");
    let nested_mod = lower_and_optimize("nested", &nested_src);
    let main_fn = nested_mod
        .functions
        .iter()
        .find(|f| f.signature.name == "main")
        .expect("main");

    let analysis = analyze_cfg(main_fn);
    assert!(analysis.is_reducible);
    assert_eq!(
        analysis.loops.len(),
        2,
        "nested.lust should have exactly 2 natural loops (outer i, inner j)"
    );

    let outer = analysis
        .loops
        .values()
        .find(|l| l.parent_loop.is_none())
        .expect("outer loop");
    let inner = analysis
        .loops
        .values()
        .find(|l| l.parent_loop == Some(outer.header))
        .expect("inner loop nested in outer");
    assert!(outer.child_loops.contains(&inner.header));
    assert!(outer.body_set.len() > inner.body_set.len());

    let tree = structure_mir_function(main_fn);
    assert_eq!(tree.count_state_machine_fallbacks(), 0);
    assert_eq!(count_while_loops(&tree), 2);

    // Verify both WhileLoop header blocks retain their condition & ResourceCheck statements.
    let mut header_blocks = Vec::new();
    collect_while_headers(&tree, &mut header_blocks);
    assert_eq!(header_blocks.len(), 2);
    for hdr in header_blocks {
        assert!(
            !main_fn.blocks[hdr.0].statements.is_empty(),
            "WhileLoop header_block {:?} should carry condition/gas statements",
            hdr
        );
    }
}

fn collect_while_headers(node: &StructuredNode, out: &mut Vec<BlockId>) {
    match node {
        StructuredNode::WhileLoop {
            header_block, body, ..
        } => {
            out.push(*header_block);
            collect_while_headers(body, out);
        }
        StructuredNode::Seq(items) => {
            for item in items {
                collect_while_headers(item, out);
            }
        }
        StructuredNode::If {
            then_branch,
            else_branch,
            ..
        }
        | StructuredNode::IfArrayGetOk {
            then_branch,
            else_branch,
            ..
        } => {
            collect_while_headers(then_branch, out);
            if let Some(eb) = else_branch {
                collect_while_headers(eb, out);
            }
        }
        StructuredNode::Switch { cases, default, .. } => {
            for (_, n) in cases {
                collect_while_headers(n, out);
            }
            collect_while_headers(default, out);
        }
        StructuredNode::Loop { body, .. } | StructuredNode::LabeledBlock { body, .. } => {
            collect_while_headers(body, out);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Test 5: Short-circuit CFG, Multi-Exit Loop, Continue/Break, and Switch
// ---------------------------------------------------------------------------

#[test]
fn test_short_circuit_multi_exit_loop_and_switch_semantic_equivalence() {
    let span = Span::dummy();

    // 5A. Short-circuit `if (a && b) then T else F; M`:
    //     bb0: if a then bb1 else bb3(F)
    //     bb1: if b then bb2(T) else bb3(F)
    //     bb2(T): r = 42; goto bb4(M)
    //     bb3(F): r = 7; goto bb4(M)
    //     bb4(M): return r
    {
        let sig = FunctionSignature {
            name: "short_circuit".into(),
            param_types: vec![MirType::Bool, MirType::Bool],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut f = MirFunction::new(sig, span);
        let a = f.params[0];
        let b = f.params[1];
        let r = f.return_local;
        let bb0 = BlockId(0);
        let bb1 = f.new_block();
        let bb2 = f.new_block();
        let bb3 = f.new_block();
        let bb4 = f.new_block();

        f.block_mut(bb0).terminator = Some(Terminator::Cond {
            cond: Operand::Copy(Place::Local(a)),
            then_block: bb1,
            else_block: bb3,
            span,
        });
        f.block_mut(bb1).terminator = Some(Terminator::Cond {
            cond: Operand::Copy(Place::Local(b)),
            then_block: bb2,
            else_block: bb3,
            span,
        });
        f.block_mut(bb2).statements.push(Statement::Assign(
            Place::Local(r),
            Rvalue::Use(Operand::Constant(MirConstant::Int(42))),
            span,
        ));
        f.block_mut(bb2).terminator = Some(Terminator::Goto(bb4, span));

        f.block_mut(bb3).statements.push(Statement::Assign(
            Place::Local(r),
            Rvalue::Use(Operand::Constant(MirConstant::Int(7))),
            span,
        ));
        f.block_mut(bb3).terminator = Some(Terminator::Goto(bb4, span));

        f.block_mut(bb4).terminator =
            Some(Terminator::Return(Some(Operand::Copy(Place::Local(r))), span));

        let analysis = analyze_cfg(&f);
        assert!(analysis.is_reducible);
        assert_eq!(analysis.ipdom[bb0.0], Some(bb4));
        assert_eq!(analysis.ipdom[bb1.0], Some(bb4));

        let tree = structure_mir_function(&f);
        assert_eq!(tree.count_state_machine_fallbacks(), 0);
        assert_all_reachable_blocks_preserved(&f, &tree);

        for av in [false, true] {
            for bv in [false, true] {
                let args = [EvalVal::Bool(av), EvalVal::Bool(bv)];
                let (fv, ft) = eval_flat_mir(&f, &args);
                let (sv, st) = eval_structured(&f, &tree, &args);
                assert_eq!(sv, fv);
                assert_eq!(st, ft);
            }
        }
    }

    // 5B. Multi-exit while loop with both `continue` and secondary labeled exit:
    //     bb0: i = 0; acc = 0; goto bb1(hdr)
    //     bb1(hdr): cond = i < n; if cond then bb2 else bb6(normal_exit)
    //     bb2: if i == target then bb7(special_exit) else bb3
    //     bb3: if i == skip then bb4(continue_bb) else bb5(add_bb)
    //     bb4: i = i + 1; goto bb1(hdr)
    //     bb5: acc = acc + i; i = i + 1; goto bb1(hdr)
    //     bb6(normal_exit): acc = acc + 1000; goto bb8(merge)
    //     bb7(special_exit): acc = acc + 500; goto bb8(merge)
    //     bb8(merge): return acc
    {
        let sig = FunctionSignature {
            name: "multi_exit_loop".into(),
            param_types: vec![MirType::Int, MirType::Int, MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut f = MirFunction::new(sig, span);
        let n_loc = f.params[0];
        let target_loc = f.params[1];
        let skip_loc = f.params[2];
        let acc = f.return_local;
        let i_loc = f.add_local(Some("i".into()), MirType::Int, true, span);
        let c_loop = f.add_local(None, MirType::Bool, true, span);
        let c_target = f.add_local(None, MirType::Bool, true, span);
        let c_skip = f.add_local(None, MirType::Bool, true, span);

        let bb0 = BlockId(0);
        let bb1 = f.new_block();
        let bb2 = f.new_block();
        let bb3 = f.new_block();
        let bb4 = f.new_block();
        let bb5 = f.new_block();
        let bb6 = f.new_block();
        let bb7 = f.new_block();
        let bb8 = f.new_block();

        f.block_mut(bb0).statements.push(Statement::Assign(
            Place::Local(i_loc),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            span,
        ));
        f.block_mut(bb0).statements.push(Statement::Assign(
            Place::Local(acc),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            span,
        ));
        f.block_mut(bb0).terminator = Some(Terminator::Goto(bb1, span));

        f.block_mut(bb1).statements.push(Statement::Assign(
            Place::Local(c_loop),
            Rvalue::BinaryOp(
                MirBinaryOp::Lt,
                Operand::Copy(Place::Local(i_loc)),
                Operand::Copy(Place::Local(n_loc)),
            ),
            span,
        ));
        f.block_mut(bb1).terminator = Some(Terminator::Cond {
            cond: Operand::Copy(Place::Local(c_loop)),
            then_block: bb2,
            else_block: bb6,
            span,
        });

        f.block_mut(bb2).statements.push(Statement::Assign(
            Place::Local(c_target),
            Rvalue::BinaryOp(
                MirBinaryOp::Eq,
                Operand::Copy(Place::Local(i_loc)),
                Operand::Copy(Place::Local(target_loc)),
            ),
            span,
        ));
        f.block_mut(bb2).terminator = Some(Terminator::Cond {
            cond: Operand::Copy(Place::Local(c_target)),
            then_block: bb7,
            else_block: bb3,
            span,
        });

        f.block_mut(bb3).statements.push(Statement::Assign(
            Place::Local(c_skip),
            Rvalue::BinaryOp(
                MirBinaryOp::Eq,
                Operand::Copy(Place::Local(i_loc)),
                Operand::Copy(Place::Local(skip_loc)),
            ),
            span,
        ));
        f.block_mut(bb3).terminator = Some(Terminator::Cond {
            cond: Operand::Copy(Place::Local(c_skip)),
            then_block: bb4,
            else_block: bb5,
            span,
        });

        f.block_mut(bb4).statements.push(Statement::Assign(
            Place::Local(i_loc),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(i_loc)),
                Operand::Constant(MirConstant::Int(1)),
            ),
            span,
        ));
        f.block_mut(bb4).terminator = Some(Terminator::Goto(bb1, span));

        f.block_mut(bb5).statements.push(Statement::Assign(
            Place::Local(acc),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(acc)),
                Operand::Copy(Place::Local(i_loc)),
            ),
            span,
        ));
        f.block_mut(bb5).statements.push(Statement::Assign(
            Place::Local(i_loc),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(i_loc)),
                Operand::Constant(MirConstant::Int(1)),
            ),
            span,
        ));
        f.block_mut(bb5).terminator = Some(Terminator::Goto(bb1, span));

        f.block_mut(bb6).statements.push(Statement::Assign(
            Place::Local(acc),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(acc)),
                Operand::Constant(MirConstant::Int(1000)),
            ),
            span,
        ));
        f.block_mut(bb6).terminator = Some(Terminator::Goto(bb8, span));

        f.block_mut(bb7).statements.push(Statement::Assign(
            Place::Local(acc),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(acc)),
                Operand::Constant(MirConstant::Int(500)),
            ),
            span,
        ));
        f.block_mut(bb7).terminator = Some(Terminator::Goto(bb8, span));

        f.block_mut(bb8).terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(acc))),
            span,
        ));

        let analysis = analyze_cfg(&f);
        assert!(analysis.is_reducible);
        assert_eq!(analysis.loops[&bb1].exit_targets.len(), 2);

        let tree = structure_mir_function(&f);
        assert_eq!(tree.count_state_machine_fallbacks(), 0);
        assert_all_reachable_blocks_preserved(&f, &tree);

        for (n, target, skip) in [(6, 99, 3), (6, 4, 2), (6, 0, 1), (0, 0, 0)] {
            let args = [EvalVal::Int(n), EvalVal::Int(target), EvalVal::Int(skip)];
            let (fv, ft) = eval_flat_mir(&f, &args);
            let (sv, st) = eval_structured(&f, &tree, &args);
            assert_eq!(sv, fv);
            assert_eq!(st, ft);
        }
    }

    // 5C. Switch with two cases sharing a target block (zero code duplication):
    {
        let sig = FunctionSignature {
            name: "switch_shared".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut f = MirFunction::new(sig, span);
        let x = f.params[0];
        let r = f.return_local;
        let bb0 = BlockId(0);
        let bb_shared = f.new_block();
        let bb_other = f.new_block();
        let bb_join = f.new_block();

        f.block_mut(bb0).terminator = Some(Terminator::Switch {
            discr: Operand::Copy(Place::Local(x)),
            cases: vec![(1, bb_shared), (2, bb_shared)],
            default: bb_other,
            span,
        });
        f.block_mut(bb_shared).statements.push(Statement::Assign(
            Place::Local(r),
            Rvalue::Use(Operand::Constant(MirConstant::Int(10))),
            span,
        ));
        f.block_mut(bb_shared).terminator = Some(Terminator::Goto(bb_join, span));

        f.block_mut(bb_other).statements.push(Statement::Assign(
            Place::Local(r),
            Rvalue::Use(Operand::Constant(MirConstant::Int(20))),
            span,
        ));
        f.block_mut(bb_other).terminator = Some(Terminator::Goto(bb_join, span));

        f.block_mut(bb_join).terminator =
            Some(Terminator::Return(Some(Operand::Copy(Place::Local(r))), span));

        let tree = structure_mir_function(&f);
        assert_eq!(tree.count_state_machine_fallbacks(), 0);
        assert_all_reachable_blocks_preserved(&f, &tree);

        for xv in [0, 1, 2, 3] {
            let args = [EvalVal::Int(xv)];
            let (fv, ft) = eval_flat_mir(&f, &args);
            let (sv, st) = eval_structured(&f, &tree, &args);
            assert_eq!(sv, fv);
            assert_eq!(st, ft);
        }
    }
}

// ---------------------------------------------------------------------------
// Test 6: Genuinely irreducible CFG isolates to local StateMachineFallback
// ---------------------------------------------------------------------------

#[test]
fn test_irreducible_cfg_uses_local_state_machine_fallback_and_preserves_exit() {
    let span = Span::dummy();
    let sig = FunctionSignature {
        name: "irreducible".into(),
        param_types: vec![MirType::Bool, MirType::Bool, MirType::Bool],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut f = MirFunction::new(sig, span);
    let c0 = f.params[0];
    let c1 = f.params[1];
    let c2 = f.params[2];
    let bb0 = BlockId(0);
    let bb1 = f.new_block();
    let bb2 = f.new_block();
    let bb3 = f.new_block();

    // Two-entry cycle: bb0 -> {bb1, bb2}; bb1 -> {bb2, bb3}; bb2 -> {bb1, bb3}; bb3 -> return 1
    f.block_mut(bb0).terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(c0)),
        then_block: bb1,
        else_block: bb2,
        span,
    });
    f.block_mut(bb1).terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(c1)),
        then_block: bb2,
        else_block: bb3,
        span,
    });
    f.block_mut(bb2).terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(c2)),
        then_block: bb1,
        else_block: bb3,
        span,
    });
    f.block_mut(bb3).terminator = Some(Terminator::Return(
        Some(Operand::Constant(MirConstant::Int(1))),
        span,
    ));

    let analysis = analyze_cfg(&f);
    assert!(!analysis.is_reducible);
    assert!(!analysis.irreducible_edges.is_empty());
    assert_eq!(analysis.ipdom[bb0.0], Some(bb3));

    let tree = structure_mir_function(&f);
    assert_eq!(tree.count_state_machine_fallbacks(), 1);
    assert_all_reachable_blocks_preserved(&f, &tree);

    match &tree {
        StructuredNode::Seq(items) => {
            assert_eq!(items.len(), 2);
            assert!(matches!(
                &items[0],
                StructuredNode::StateMachineFallback {
                    entry: BlockId(0),
                    exit: Some(BlockId(3)),
                    blocks,
                } if blocks.contains(&bb0) && blocks.contains(&bb1) && blocks.contains(&bb2)
            ));
            assert!(matches!(&items[1], StructuredNode::Return(BlockId(3), _)));
        }
        other => panic!("expected Seq([StateMachineFallback, Return]), got {other:#?}"),
    }
}

// ---------------------------------------------------------------------------
// Test 7: Negated-condition WhileLoop, General Loop, and Nested Loop Labeled Break/Continue
// ---------------------------------------------------------------------------

#[test]
fn test_negated_while_general_loop_and_nested_labeled_break_continue() {
    let span = Span::dummy();

    // 7A. Header-negated WhileLoop (`if done then exit else body`):
    {
        let sig = FunctionSignature {
            name: "negated_while".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut f = MirFunction::new(sig, span);
        let n = f.params[0];
        let acc = f.return_local;
        let i = f.add_local(Some("i".into()), MirType::Int, true, span);
        let done = f.add_local(None, MirType::Bool, true, span);

        let bb0 = BlockId(0);
        let bb_hdr = f.new_block();
        let bb_body = f.new_block();
        let bb_exit = f.new_block();

        f.block_mut(bb0).statements.push(Statement::Assign(
            Place::Local(i),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            span,
        ));
        f.block_mut(bb0).statements.push(Statement::Assign(
            Place::Local(acc),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            span,
        ));
        f.block_mut(bb0).terminator = Some(Terminator::Goto(bb_hdr, span));

        f.block_mut(bb_hdr).statements.push(Statement::Assign(
            Place::Local(done),
            Rvalue::BinaryOp(
                MirBinaryOp::GtEq,
                Operand::Copy(Place::Local(i)),
                Operand::Copy(Place::Local(n)),
            ),
            span,
        ));
        // `then_block` is EXIT, `else_block` is BODY => negate_cond == true!
        f.block_mut(bb_hdr).terminator = Some(Terminator::Cond {
            cond: Operand::Copy(Place::Local(done)),
            then_block: bb_exit,
            else_block: bb_body,
            span,
        });

        f.block_mut(bb_body).statements.push(Statement::Assign(
            Place::Local(acc),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(acc)),
                Operand::Copy(Place::Local(i)),
            ),
            span,
        ));
        f.block_mut(bb_body).statements.push(Statement::Assign(
            Place::Local(i),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(i)),
                Operand::Constant(MirConstant::Int(1)),
            ),
            span,
        ));
        f.block_mut(bb_body).terminator = Some(Terminator::Goto(bb_hdr, span));

        f.block_mut(bb_exit).terminator =
            Some(Terminator::Return(Some(Operand::Copy(Place::Local(acc))), span));

        let tree = structure_mir_function(&f);
        assert_eq!(tree.count_state_machine_fallbacks(), 0);
        assert_all_reachable_blocks_preserved(&f, &tree);

        // Ensure `WhileLoop { negate_cond: true, .. }` was emitted.
        let mut found_negated_while = false;
        if let StructuredNode::Seq(items) = &tree {
            for item in items {
                if let StructuredNode::WhileLoop {
                    header_block,
                    negate_cond,
                    ..
                } = item
                {
                    assert_eq!(*header_block, bb_hdr);
                    assert!(*negate_cond);
                    found_negated_while = true;
                }
            }
        }
        assert!(found_negated_while, "expected WhileLoop with negate_cond=true: {tree:#?}");

        for nv in [0, 1, 5, 10] {
            let args = [EvalVal::Int(nv)];
            let (fv, ft) = eval_flat_mir(&f, &args);
            let (sv, st) = eval_structured(&f, &tree, &args);
            assert_eq!(sv, fv);
            assert_eq!(st, ft);
        }
    }

    // 7B. Nested loops with labeled `break 'outer` and `continue 'outer` from inside inner loop:
    //     bb0: i = 0; acc = 0; goto bb_outer_hdr
    //     bb_outer_hdr: if i < n then bb_outer_body else bb_outer_exit
    //     bb_outer_body: j = 0; goto bb_inner_hdr
    //     bb_inner_hdr: if j < m then bb_inner_body else bb_outer_inc
    //     bb_inner_body: if i + j == stop_sum then bb_outer_exit else bb_check_cont
    //     bb_check_cont: if j == skip_j then bb_outer_inc else bb_inner_inc
    //     bb_inner_inc: acc = acc + 1; j = j + 1; goto bb_inner_hdr
    //     bb_outer_inc: i = i + 1; goto bb_outer_hdr
    //     bb_outer_exit: return acc
    {
        let sig = FunctionSignature {
            name: "nested_labeled_break_continue".into(),
            param_types: vec![MirType::Int, MirType::Int, MirType::Int, MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut f = MirFunction::new(sig, span);
        let n = f.params[0];
        let m = f.params[1];
        let stop_sum = f.params[2];
        let skip_j = f.params[3];
        let acc = f.return_local;
        let i = f.add_local(Some("i".into()), MirType::Int, true, span);
        let j = f.add_local(Some("j".into()), MirType::Int, true, span);
        let sum = f.add_local(None, MirType::Int, true, span);
        let c1 = f.add_local(None, MirType::Bool, true, span);
        let c2 = f.add_local(None, MirType::Bool, true, span);
        let c3 = f.add_local(None, MirType::Bool, true, span);
        let c4 = f.add_local(None, MirType::Bool, true, span);

        let bb0 = BlockId(0);
        let bb_outer_hdr = f.new_block();
        let bb_outer_body = f.new_block();
        let bb_inner_hdr = f.new_block();
        let bb_inner_body = f.new_block();
        let bb_check_cont = f.new_block();
        let bb_inner_inc = f.new_block();
        let bb_outer_inc = f.new_block();
        let bb_outer_exit = f.new_block();

        f.block_mut(bb0).statements.push(Statement::Assign(
            Place::Local(i),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            span,
        ));
        f.block_mut(bb0).statements.push(Statement::Assign(
            Place::Local(acc),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            span,
        ));
        f.block_mut(bb0).terminator = Some(Terminator::Goto(bb_outer_hdr, span));

        f.block_mut(bb_outer_hdr).statements.push(Statement::Assign(
            Place::Local(c1),
            Rvalue::BinaryOp(
                MirBinaryOp::Lt,
                Operand::Copy(Place::Local(i)),
                Operand::Copy(Place::Local(n)),
            ),
            span,
        ));
        f.block_mut(bb_outer_hdr).terminator = Some(Terminator::Cond {
            cond: Operand::Copy(Place::Local(c1)),
            then_block: bb_outer_body,
            else_block: bb_outer_exit,
            span,
        });

        f.block_mut(bb_outer_body).statements.push(Statement::Assign(
            Place::Local(j),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            span,
        ));
        f.block_mut(bb_outer_body).terminator = Some(Terminator::Goto(bb_inner_hdr, span));

        f.block_mut(bb_inner_hdr).statements.push(Statement::Assign(
            Place::Local(c2),
            Rvalue::BinaryOp(
                MirBinaryOp::Lt,
                Operand::Copy(Place::Local(j)),
                Operand::Copy(Place::Local(m)),
            ),
            span,
        ));
        f.block_mut(bb_inner_hdr).terminator = Some(Terminator::Cond {
            cond: Operand::Copy(Place::Local(c2)),
            then_block: bb_inner_body,
            else_block: bb_outer_inc,
            span,
        });

        f.block_mut(bb_inner_body).statements.push(Statement::Assign(
            Place::Local(sum),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(i)),
                Operand::Copy(Place::Local(j)),
            ),
            span,
        ));
        f.block_mut(bb_inner_body).statements.push(Statement::Assign(
            Place::Local(c3),
            Rvalue::BinaryOp(
                MirBinaryOp::Eq,
                Operand::Copy(Place::Local(sum)),
                Operand::Copy(Place::Local(stop_sum)),
            ),
            span,
        ));
        f.block_mut(bb_inner_body).terminator = Some(Terminator::Cond {
            cond: Operand::Copy(Place::Local(c3)),
            then_block: bb_outer_exit,
            else_block: bb_check_cont,
            span,
        });

        f.block_mut(bb_check_cont).statements.push(Statement::Assign(
            Place::Local(c4),
            Rvalue::BinaryOp(
                MirBinaryOp::Eq,
                Operand::Copy(Place::Local(j)),
                Operand::Copy(Place::Local(skip_j)),
            ),
            span,
        ));
        f.block_mut(bb_check_cont).terminator = Some(Terminator::Cond {
            cond: Operand::Copy(Place::Local(c4)),
            then_block: bb_outer_inc,
            else_block: bb_inner_inc,
            span,
        });

        f.block_mut(bb_inner_inc).statements.push(Statement::Assign(
            Place::Local(acc),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(acc)),
                Operand::Constant(MirConstant::Int(1)),
            ),
            span,
        ));
        f.block_mut(bb_inner_inc).statements.push(Statement::Assign(
            Place::Local(j),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(j)),
                Operand::Constant(MirConstant::Int(1)),
            ),
            span,
        ));
        f.block_mut(bb_inner_inc).terminator = Some(Terminator::Goto(bb_inner_hdr, span));

        f.block_mut(bb_outer_inc).statements.push(Statement::Assign(
            Place::Local(i),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(i)),
                Operand::Constant(MirConstant::Int(1)),
            ),
            span,
        ));
        f.block_mut(bb_outer_inc).terminator = Some(Terminator::Goto(bb_outer_hdr, span));

        f.block_mut(bb_outer_exit).terminator =
            Some(Terminator::Return(Some(Operand::Copy(Place::Local(acc))), span));

        let analysis = analyze_cfg(&f);
        assert!(analysis.is_reducible);
        assert_eq!(analysis.loops.len(), 2);
        assert_eq!(analysis.loops[&bb_inner_hdr].parent_loop, Some(bb_outer_hdr));

        let tree = structure_mir_function(&f);
        assert_eq!(tree.count_state_machine_fallbacks(), 0);
        assert_all_reachable_blocks_preserved(&f, &tree);

        for (nv, mv, stopv, skipv) in [
            (4, 4, 99, 99),
            (4, 4, 3, 99),
            (4, 4, 99, 2),
            (5, 5, 4, 3),
        ] {
            let args = [
                EvalVal::Int(nv),
                EvalVal::Int(mv),
                EvalVal::Int(stopv),
                EvalVal::Int(skipv),
            ];
            let (fv, ft) = eval_flat_mir(&f, &args);
            let (sv, st) = eval_structured(&f, &tree, &args);
            assert_eq!(sv, fv);
            assert_eq!(st, ft);
        }
    }
}


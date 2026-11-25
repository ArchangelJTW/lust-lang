#![cfg(feature = "std")]
#![forbid(unsafe_code)]

//! NG-W14-1 Verification Suite:
//! 1. Inner-loop `Q=64` strip-mining on nested loops (innermost natural loops prioritized for Q=64
//!    while enclosing outer loops remain on standard non-quantized counter promotion).
//! 2. True batch gas & safepoint reconciliation in `emit_quantized_loop_header_check`:
//!    fast path updates ONLY `quantum_rem -= 1; brif(q < 0, refill_bb, cont_bb)` with exact
//!    reconciliation on quantum refill, normal loop exits, early break/return, OOG/OOB traps,
//!    and forced `test-deopt` mid-quantum exits.
//! 3. Exact `SEC-01` gas thresholds (`0, 1, 63, 64, 65`) with zero observable gas overshoot.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;

use lust::ast::Span;
use lust::bytecode::Value;
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{
    lust_mir_get_status, promoted_loop_cfg_verifications, reset_promoted_loop_cfg_verifications,
    reset_unbounded_gas_replenish_count, safepoint_reset, safepoint_stats,
    set_unbounded_gas_chunk_override, unbounded_gas_replenish_count, CraneliftJitCompiler,
    GasFastState, SafepointFastState, MIR_STATUS_ERR_OUT_OF_BOUNDS,
};
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::safepoint_request_deopt_on_poll;
use lust::mir::ir::{
    BasicBlock, BlockId, FunctionSignature, LocalDecl, LocalId, MirBinaryOp, MirConstant,
    MirFunction, MirModule, MirType, Operand, Place, ResourceKind, Rvalue, Statement, Terminator,
};
use lust::mir::lowering::AstLowering;
use lust::mir::verifier::MirVerifier;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{VmBackend, VM};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn parse_and_lower_module(source: &str, module_name: &str) -> MirModule {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("Tokenization must succeed");
    let items = Parser::new(tokens).parse().expect("Parsing must succeed");

    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);

    AstLowering::lower_module_items(module_name, &items).expect("MIR module lowering must succeed")
}

struct ClifIrView {
    blocks: HashMap<String, Vec<String>>,
    cold_blocks: HashSet<String>,
    block_order: Vec<String>,
    succs: HashMap<String, Vec<String>>,
}

impl ClifIrView {
    fn parse(ir: &str) -> Self {
        let mut blocks: HashMap<String, Vec<String>> = HashMap::new();
        let mut cold_blocks: HashSet<String> = HashSet::new();
        let mut block_order: Vec<String> = Vec::new();
        let mut cur_block: Option<String> = None;

        for raw_line in ir.lines() {
            let line = raw_line.trim();
            if line.starts_with("block") && line.contains(':') {
                let bname = line
                    .split(|c: char| c == '(' || c == ':' || c.is_whitespace())
                    .next()
                    .unwrap()
                    .to_string();
                if line.contains("cold") {
                    cold_blocks.insert(bname.clone());
                }
                block_order.push(bname.clone());
                blocks.entry(bname.clone()).or_default();
                cur_block = Some(bname);
                continue;
            }
            if line == "}" {
                cur_block = None;
                continue;
            }
            if let Some(ref bname) = cur_block {
                if !line.is_empty() && !line.starts_with(';') {
                    blocks
                        .entry(bname.clone())
                        .or_default()
                        .push(line.to_string());
                }
            }
        }

        let mut succs: HashMap<String, Vec<String>> = HashMap::new();
        for b in &block_order {
            let mut s = Vec::new();
            if let Some(lines) = blocks.get(b) {
                for l in lines {
                    if l.starts_with("jump ") || l.starts_with("brif ") || l.starts_with("br_table ") {
                        let bytes = l.as_bytes();
                        let mut pos = 0;
                        while let Some(rel) = l[pos..].find("block") {
                            let start = pos + rel;
                            let mut end = start + 5;
                            while end < bytes.len() && bytes[end].is_ascii_digit() {
                                end += 1;
                            }
                            if end > start + 5 {
                                let tgt = &l[start..end];
                                if !s.iter().any(|x| x == tgt) {
                                    s.push(tgt.to_string());
                                }
                            }
                            pos = end;
                        }
                    }
                }
            }
            succs.insert(b.clone(), s);
        }

        Self {
            blocks,
            cold_blocks,
            block_order,
            succs,
        }
    }

    fn hot_cycle_blocks(&self) -> HashSet<String> {
        let hot_nodes: HashSet<&String> = self
            .block_order
            .iter()
            .filter(|b| !self.cold_blocks.contains(*b))
            .collect();

        let mut in_cycle = HashSet::new();
        for &start in &hot_nodes {
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();
            if let Some(s_list) = self.succs.get(start) {
                for nxt in s_list {
                    if hot_nodes.contains(nxt) && visited.insert(nxt.clone()) {
                        queue.push_back(nxt.clone());
                    }
                }
            }
            while let Some(cur) = queue.pop_front() {
                if cur == *start {
                    in_cycle.insert(start.clone());
                    break;
                }
                if let Some(s_list) = self.succs.get(&cur) {
                    for nxt in s_list {
                        if hot_nodes.contains(nxt) && visited.insert(nxt.clone()) {
                            queue.push_back(nxt.clone());
                        }
                    }
                }
            }
        }
        in_cycle
    }
}

fn push_local(
    f: &mut MirFunction,
    name: &str,
    ty: MirType,
    is_mut: bool,
    is_param: bool,
) -> LocalId {
    let id = LocalId(f.locals.len());
    let mut decl = LocalDecl::new(id, Some(name.into()), ty, is_mut, Span::dummy());
    decl.is_param = is_param;
    f.locals.push(decl);
    if is_param {
        f.params.push(id);
    }
    id
}

fn push_block(f: &mut MirFunction) -> BlockId {
    let id = BlockId(f.blocks.len());
    f.blocks.push(BasicBlock::new(id));
    id
}

/// Builds a nested loop MIR function with configurable outer and inner header gas amounts.
fn build_nested_loop_fn(name: &str, outer_amt: u64, inner_amt: u64) -> MirFunction {
    let mut f = MirFunction::new(
        FunctionSignature {
            name: name.into(),
            param_types: vec![MirType::Int, MirType::Int],
            return_type: MirType::Int,
            effects: None,
            is_public: true,
        },
        Span::dummy(),
    );
    f.locals.clear();
    f.params.clear();
    let outer_n = push_local(&mut f, "outer_n", MirType::Int, false, true);
    let inner_n = push_local(&mut f, "inner_n", MirType::Int, false, true);
    let i = push_local(&mut f, "i", MirType::Int, true, false);
    let j = push_local(&mut f, "j", MirType::Int, true, false);
    let acc = push_local(&mut f, "acc", MirType::Int, true, false);
    let c_outer = push_local(&mut f, "c_outer", MirType::Bool, true, false);
    let c_inner = push_local(&mut f, "c_inner", MirType::Bool, true, false);

    let bb0 = f.entry;
    let bb_outer_hdr = push_block(&mut f);
    let bb_outer_body = push_block(&mut f);
    let bb_inner_hdr = push_block(&mut f);
    let bb_inner_body = push_block(&mut f);
    let bb_outer_latch = push_block(&mut f);
    let bb_exit = push_block(&mut f);

    f.blocks[bb0.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[bb0.0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[bb0.0].terminator = Some(Terminator::Goto(bb_outer_hdr, Span::dummy()));

    // Outer header
    f.blocks[bb_outer_hdr.0]
        .statements
        .push(Statement::ResourceCheck(
            ResourceKind::Gas,
            outer_amt,
            Span::dummy(),
        ));
    f.blocks[bb_outer_hdr.0].statements.push(Statement::Assign(
        Place::Local(c_outer),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(i)),
            Operand::Copy(Place::Local(outer_n)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_outer_hdr.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(c_outer)),
        then_block: bb_outer_body,
        else_block: bb_exit,
        span: Span::dummy(),
    });

    // Outer body initializes j = 0 -> inner header
    f.blocks[bb_outer_body.0].statements.push(Statement::Assign(
        Place::Local(j),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[bb_outer_body.0].terminator = Some(Terminator::Goto(bb_inner_hdr, Span::dummy()));

    // Inner header
    f.blocks[bb_inner_hdr.0]
        .statements
        .push(Statement::ResourceCheck(
            ResourceKind::Gas,
            inner_amt,
            Span::dummy(),
        ));
    f.blocks[bb_inner_hdr.0].statements.push(Statement::Assign(
        Place::Local(c_inner),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(j)),
            Operand::Copy(Place::Local(inner_n)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_inner_hdr.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(c_inner)),
        then_block: bb_inner_body,
        else_block: bb_outer_latch,
        span: Span::dummy(),
    });

    // Inner body
    f.blocks[bb_inner_body.0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(acc)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_inner_body.0].statements.push(Statement::Assign(
        Place::Local(j),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(j)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_inner_body.0].terminator = Some(Terminator::Goto(bb_inner_hdr, Span::dummy()));

    // Outer latch
    f.blocks[bb_outer_latch.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_outer_latch.0].terminator = Some(Terminator::Goto(bb_outer_hdr, Span::dummy()));

    f.blocks[bb_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(acc))),
        Span::dummy(),
    ));
    f
}

// ============================================================================
// 1. IR CONTRACT: Inner-Loop Q=64 Enabled on Nested Loops + True Batch Fast Path
// ============================================================================

#[test]
fn test_ng_w14_1_nested_inner_loop_enables_q64() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();
    reset_promoted_loop_cfg_verifications();

    // (a) Nested `while` loop: outer keeps standard counter promotion (zero ssub_overflow),
    //     and inner loop MUST enable Q=64 quantization (`iconst.i64 63`)!
    let nested_src = r#"
        function nested_sum(outer: int, inner: int): int
            local i: int = 0
            local total: int = 0
            while i < outer do
                local j: int = 0
                while j < inner do
                    total = total + 1
                    j = j + 1
                end
                i = i + 1
            end
            return total
        end
    "#;
    let nested_mod = parse_and_lower_module(nested_src, "ng_w14_1_nested_ir");
    let mut compiler = CraneliftJitCompiler::new().expect("create compiler");
    let nested_ir = compiler
        .dump_function_ir(&nested_mod.functions[0])
        .expect("dump nested_sum IR");

    assert!(
        !nested_ir.contains("ssub_overflow"),
        "Outer and inner promoted loops must not emit ssub_overflow:\n{nested_ir}"
    );
    assert!(
        nested_ir.contains("iconst.i64 63"),
        "NG-W14-1: Inner loop of nested while MUST enable Q=64 quantization (iconst.i64 63):\n{nested_ir}"
    );
    assert!(
        promoted_loop_cfg_verifications() > 0,
        "Mechanical CFG exit-completeness verifier must execute"
    );

    // (b) Single `floatmath`-style loop: true batch fast path must collapse the 4-block diamond
    //     so the hot cycle updates ONLY `quantum_rem` (-1) and `i` (+1) — NOT `sp_cnt + 1` or `gas - 1`!
    let float_src = r#"
        function float_loop(n: int): float
            local x: float = 0.0
            local acc: float = 0.0
            local i: int = 0
            while i < n do
                x = x + 0.001
                acc = acc + x * x - acc * 0.5
                i = i + 1
            end
            return acc
        end
    "#;
    let float_mod = parse_and_lower_module(float_src, "ng_w14_1_float_ir");
    let mut float_compiler = CraneliftJitCompiler::new().expect("create compiler");
    let float_ir = float_compiler
        .dump_function_ir(&float_mod.functions[0])
        .expect("dump float_loop IR");
    assert!(
        float_ir.contains("iconst.i64 63"),
        "Single loop must arm Q=64 quantum:\n{float_ir}"
    );
    let parsed_float = ClifIrView::parse(&float_ir);
    let hot_blocks = parsed_float.hot_cycle_blocks();
    assert!(
        !hot_blocks.is_empty(),
        "Expected hot cycle blocks in float_loop:\n{float_ir}"
    );
    let mut hot_iadd_plus_1 = 0usize;
    let mut hot_iadd_minus_1 = 0usize;
    let mut hot_loads = 0usize;
    for b in &hot_blocks {
        if let Some(lines) = parsed_float.blocks.get(b) {
            for l in lines {
                // Cranelift lowers `iadd_imm_s(x, 1)` to `iconst.i64 1` + `iadd ... ; vN = 1`
                if (l.contains("iadd_imm ") && l.ends_with(", 1"))
                    || (l.contains(" = iadd") && l.ends_with("= 1"))
                {
                    hot_iadd_plus_1 += 1;
                }
                if (l.contains("iadd_imm ") && l.ends_with(", -1"))
                    || (l.contains(" = iadd") && l.ends_with("= -1"))
                {
                    hot_iadd_minus_1 += 1;
                }
                if l.contains(" = load.") {
                    hot_loads += 1;
                }
            }
        }
    }
    assert_eq!(
        hot_iadd_plus_1, 1,
        "Hot cycle of float_loop must have ONLY `i = i + 1` (zero per-iter `sp_cnt + 1`), found {hot_iadd_plus_1} in {hot_blocks:?}:\n{float_ir}"
    );
    assert_eq!(
        hot_iadd_minus_1, 1,
        "Hot cycle of float_loop must have ONLY `quantum_rem - 1` (zero per-iter `gas - 1`), found {hot_iadd_minus_1} in {hot_blocks:?}:\n{float_ir}"
    );
    assert_eq!(
        hot_loads, 0,
        "Hot cycle of float_loop must have ZERO memory loads (slow_flags/gas/sp), found {hot_loads}:\n{float_ir}"
    );

    // (c) Nested peeled-header array loop (`array.lust`-style): inner loop enables Q=64
    //     AND produces exact parity + exact gas/safepoint reconciliation!
    let nested_arr_src = r#"
        function nested_arr_sum(arr: Array<int>, passes: int, n: int): int
            local total: int = 0
            local p: int = 0
            while p < passes do
                local j: int = 0
                while j < n do
                    total = total + arr[j]
                    j = j + 1
                end
                p = p + 1
            end
            return total
        end
    "#;
    let nested_arr_mod = parse_and_lower_module(nested_arr_src, "ng_w14_1_nested_arr");
    let mut arr_compiler = CraneliftJitCompiler::new().expect("create compiler");
    let nested_arr_ir = arr_compiler
        .dump_function_ir(&nested_arr_mod.functions[0])
        .expect("dump nested_arr_sum IR");
    assert!(
        nested_arr_ir.contains("iconst.i64 63"),
        "Nested peeled-header array inner loop MUST enable Q=64:\n{nested_arr_ir}"
    );
    assert!(
        !nested_arr_ir.contains("ssub_overflow"),
        "Nested peeled-header array loop must not emit ssub_overflow:\n{nested_arr_ir}"
    );

    // Verify runtime parity + exact gas/safepoint counts on sibling inner loops & 3-deep nested loops
    let complex_nested_src = r#"
        function sibling_and_deep(a: int, b: int, c: int): int
            local total: int = 0
            local i: int = 0
            while i < a do
                local j: int = 0
                while j < b do
                    total = total + 1
                    j = j + 1
                end
                local k: int = 0
                while k < c do
                    total = total + 2
                    k = k + 1
                end
                i = i + 1
            end
            return total
        end

        function three_deep(a: int, b: int, c: int): int
            local total: int = 0
            local i: int = 0
            while i < a do
                local j: int = 0
                while j < b do
                    local k: int = 0
                    while k < c do
                        total = total + 1
                        k = k + 1
                    end
                    j = j + 1
                end
                i = i + 1
            end
            return total
        end
    "#;
    let complex_mod = parse_and_lower_module(complex_nested_src, "ng_w14_1_complex_nested");

    // sibling_and_deep(3, 100, 75):
    //   outer headers: 3 + 1 = 4
    //   inner1 headers: 3 * (100 + 1) = 303
    //   inner2 headers: 3 * (75 + 1) = 228
    //   total loop checks = 535 (+ 1 VM entry charge = 536 gas_used); result = 750
    {
        safepoint_reset();
        let mut vm = VM::new();
        vm.load_mir_module(&complex_mod).expect("load complex_mod");
        vm.set_backend(VmBackend::Cranelift);
        vm.set_gas_budget(100_000);
        let res = vm
            .call(
                "sibling_and_deep",
                vec![Value::Int(3), Value::Int(100), Value::Int(75)],
            )
            .expect("sibling_and_deep must succeed");
        assert_eq!(res, Value::Int(750));
        assert_eq!(vm.gas_used(), 536);
        assert_eq!(vm.gas_remaining(), Some(100_000 - 536));
        assert_eq!(safepoint_stats().safepoints_checked, 535);
    }

    // three_deep(2, 3, 100):
    //   outer: 2 + 1 = 3
    //   mid: 2 * (3 + 1) = 8
    //   inner: 2 * 3 * (100 + 1) = 606
    //   total loop checks = 617 (+ 1 VM entry charge = 618 gas_used); result = 600
    {
        safepoint_reset();
        let mut vm = VM::new();
        vm.load_mir_module(&complex_mod).expect("load complex_mod");
        vm.set_backend(VmBackend::Cranelift);
        vm.set_gas_budget(100_000);
        let res = vm
            .call(
                "three_deep",
                vec![Value::Int(2), Value::Int(3), Value::Int(100)],
            )
            .expect("three_deep must succeed");
        assert_eq!(res, Value::Int(600));
        assert_eq!(vm.gas_used(), 618);
        assert_eq!(vm.gas_remaining(), Some(100_000 - 618));
        assert_eq!(safepoint_stats().safepoints_checked, 617);
    }

    let _ = core::mem::offset_of!(GasFastState, gas_remaining);
    let _ = core::mem::offset_of!(SafepointFastState, safepoints_checked);
}

// ============================================================================
// 2. SEC-01 EXACT GAS THRESHOLDS (0, 1, 63, 64, 65) & NORMAL/EARLY/ERROR EXIT RECONCILIATION
// ============================================================================

#[test]
fn test_ng_w14_1_sec01_exact_gas_thresholds_and_normal_exit() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let src = r#"
        function single_loop(n: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                acc = acc + i
                i = i + 1
            end
            return acc
        end

        function nested_loop(outer: int, inner: int): int
            local i: int = 0
            local total: int = 0
            while i < outer do
                local j: int = 0
                while j < inner do
                    total = total + 1
                    j = j + 1
                end
                i = i + 1
            end
            return total
        end

        function nested_early_break(outer: int, inner: int, stop_j: int): int
            local i: int = 0
            local total: int = 0
            while i < outer do
                local j: int = 0
                while j < inner do
                    if j == stop_j then
                        break
                    end
                    total = total + 1
                    j = j + 1
                end
                i = i + 1
            end
            return total
        end

        function nested_early_return(outer: int, inner: int, stop_total: int): int
            local i: int = 0
            local total: int = 0
            while i < outer do
                local j: int = 0
                while j < inner do
                    if total == stop_total then
                        return total
                    end
                    total = total + 1
                    j = j + 1
                end
                i = i + 1
            end
            return total
        end

        function nested_arr_oob(arr: Array<int>, outer: int, inner: int): int
            local i: int = 0
            local total: int = 0
            while i < outer do
                local j: int = 0
                while j < inner do
                    total = total + arr[j]
                    j = j + 1
                end
                i = i + 1
            end
            return total
        end
    "#;
    let module = parse_and_lower_module(src, "ng_w14_1_sec01_mod");
    MirVerifier::verify_module(&module).expect("verify sec01_mod");

    // (a) Exact SEC-01 OOG thresholds (`0, 1, 63, 64, 65` + surrounding boundaries) on single & nested loops
    for budget in [0u64, 1, 2, 62, 63, 64, 65, 66, 127, 128, 129, 130] {
        // VM entry charges 1 base gas before entering native code:
        // - budget == 0: entry consumes 1 (rem=-1) + first header traps immediately (rem=-2, sp=1, gas_used=2)
        // - budget >= 1: entry consumes 1 (rem=budget-1) + `budget` headers -> exact trap at gas_used=budget+1, sp=budget
        let expected_oog_gas_used = if budget == 0 { 2 } else { budget + 1 };
        let expected_oog_sp = if budget == 0 { 1 } else { budget };

        // 1. Single loop OOG: zero overshoot!
        {
            safepoint_reset();
            let mut vm = VM::new();
            vm.load_mir_module(&module).expect("load module");
            vm.set_backend(VmBackend::Cranelift);
            vm.set_gas_budget(budget);

            let err = vm
                .call("single_loop", vec![Value::Int(500)])
                .expect_err("single_loop must fail OutOfGas");
            assert!(
                err.to_string().contains("Out of gas"),
                "single_loop budget={budget}: got {err}"
            );
            assert_eq!(
                vm.gas_used(),
                expected_oog_gas_used,
                "single_loop exact OOG gas_used at budget={budget}"
            );
            assert_eq!(vm.gas_remaining(), Some(0));
            assert_eq!(
                safepoint_stats().safepoints_checked,
                expected_oog_sp,
                "single_loop exact safepoints_checked at budget={budget}"
            );
        }

        // 2. Single loop exact-fit normal exit (when budget >= 2):
        //    1 VM entry + (budget - 1) loop header checks = `budget` total gas -> succeeds with 0 remaining!
        if budget >= 2 {
            let n = (budget - 2) as i64;
            safepoint_reset();
            let mut vm = VM::new();
            vm.load_mir_module(&module).expect("load module");
            vm.set_backend(VmBackend::Cranelift);
            vm.set_gas_budget(budget);

            let res = vm
                .call("single_loop", vec![Value::Int(n)])
                .expect("single_loop exact-fit must succeed");
            assert_eq!(res, Value::Int(n * (n - 1) / 2));
            assert_eq!(
                vm.gas_used(),
                budget,
                "single_loop exact-fit gas_used at budget={budget}"
            );
            assert_eq!(vm.gas_remaining(), Some(0));
            assert_eq!(
                safepoint_stats().safepoints_checked,
                budget - 1,
                "single_loop exact-fit safepoints_checked at budget={budget}"
            );
        }

        // 3. Nested loop OOG across inner Q=64 boundaries: zero overshoot!
        {
            safepoint_reset();
            let mut vm = VM::new();
            vm.load_mir_module(&module).expect("load module");
            vm.set_backend(VmBackend::Cranelift);
            vm.set_gas_budget(budget);

            let err = vm
                .call("nested_loop", vec![Value::Int(10), Value::Int(200)])
                .expect_err("nested_loop must fail OutOfGas");
            assert!(
                err.to_string().contains("Out of gas"),
                "nested_loop budget={budget}: got {err}"
            );
            assert_eq!(
                vm.gas_used(),
                expected_oog_gas_used,
                "nested_loop exact OOG gas_used at budget={budget}"
            );
            assert_eq!(vm.gas_remaining(), Some(0));
            assert_eq!(
                safepoint_stats().safepoints_checked,
                expected_oog_sp,
                "nested_loop exact safepoints_checked at budget={budget}"
            );
        }
    }

    // (b) Normal exit mid-quantum reconciliation on single loop (n = 17 -> 18 headers + 1 entry = 19 gas_used)
    {
        safepoint_reset();
        let mut vm = VM::new();
        vm.load_mir_module(&module).expect("load module");
        vm.set_backend(VmBackend::Cranelift);
        vm.set_gas_budget(10_000);
        let res = vm
            .call("single_loop", vec![Value::Int(17)])
            .expect("single_loop(17) must succeed");
        assert_eq!(res, Value::Int(17 * 16 / 2));
        assert_eq!(vm.gas_used(), 19);
        assert_eq!(vm.gas_remaining(), Some(10_000 - 19));
        assert_eq!(safepoint_stats().safepoints_checked, 18);
    }

    // (c) Normal exit reconciliation on nested loops across 0-iter, 1-iter, mid-quantum, exact-quantum, multi-quantum!
    for (outer, inner) in [(0i64, 100i64), (5, 0), (5, 1), (4, 17), (3, 63), (3, 64), (5, 150)] {
        let expected_headers = ((outer + 1) + outer * (inner + 1)) as u64;
        let expected_gas_used = 1 + expected_headers;
        safepoint_reset();
        let mut vm = VM::new();
        vm.load_mir_module(&module).expect("load module");
        vm.set_backend(VmBackend::Cranelift);
        vm.set_gas_budget(100_000);
        let res = vm
            .call("nested_loop", vec![Value::Int(outer), Value::Int(inner)])
            .expect("nested_loop must succeed");
        assert_eq!(res, Value::Int(outer * inner));
        assert_eq!(
            vm.gas_used(),
            expected_gas_used,
            "nested_loop({outer}, {inner}) exact gas_used reconciliation failed"
        );
        assert_eq!(vm.gas_remaining(), Some(100_000 - expected_gas_used));
        assert_eq!(
            safepoint_stats().safepoints_checked,
            expected_headers,
            "nested_loop({outer}, {inner}) exact safepoints_checked reconciliation failed"
        );
    }

    // (d) Mid-quantum `break` inside nested inner loop: outer=4, inner=100, break at j==17
    //     Each outer iter: 18 inner header checks (j=0..=17); outer: 5 checks => 77 gas checks (+1 entry = 78)
    {
        safepoint_reset();
        let mut vm = VM::new();
        vm.load_mir_module(&module).expect("load module");
        vm.set_backend(VmBackend::Cranelift);
        vm.set_gas_budget(100_000);
        let res = vm
            .call(
                "nested_early_break",
                vec![Value::Int(4), Value::Int(100), Value::Int(17)],
            )
            .expect("nested_early_break must succeed");
        assert_eq!(res, Value::Int(4 * 17));
        assert_eq!(vm.gas_used(), 78);
        assert_eq!(vm.gas_remaining(), Some(100_000 - 78));
    }

    // (e) Mid-quantum early `return` directly out of nested inner loop: outer=5, inner=100, return at total==217 (i=2, j=17)
    //     i=0: 1 outer + 101 inner = 102
    //     i=1: 1 outer + 101 inner = 102
    //     i=2: 1 outer + 18 inner (j=0..=17) = 19
    //     Total = 223 checks (+1 entry = 224 gas_used), result = 217
    {
        // Confirm nested_early_return actually enables Q=64 on its inner loop!
        let early_ret_fn = module
            .functions
            .iter()
            .find(|f| f.signature.name == "nested_early_return")
            .expect("nested_early_return fn");
        let mut ir_comp = CraneliftJitCompiler::new().expect("compiler");
        let early_ret_ir = ir_comp
            .dump_function_ir(early_ret_fn)
            .expect("dump nested_early_return IR");
        assert!(
            early_ret_ir.contains("iconst.i64 63"),
            "nested_early_return inner loop must arm Q=64 to test mid-quantum return reconciliation:\n{early_ret_ir}"
        );

        safepoint_reset();
        let mut vm = VM::new();
        vm.load_mir_module(&module).expect("load module");
        vm.set_backend(VmBackend::Cranelift);
        vm.set_gas_budget(100_000);
        let res = vm
            .call(
                "nested_early_return",
                vec![Value::Int(5), Value::Int(100), Value::Int(217)],
            )
            .expect("nested_early_return must succeed");
        assert_eq!(res, Value::Int(217));
        assert_eq!(vm.gas_used(), 224);
        assert_eq!(vm.gas_remaining(), Some(100_000 - 224));
        assert_eq!(safepoint_stats().safepoints_checked, 223);
    }

    // (f) Mid-quantum OOB trap inside nested inner loop must reconcile exact gas & safepoints!
    //     arr has len 17; nested_arr_oob(arr, outer=3, inner=100) traps at i=0, j=17!
    //     Checks executed before OOB: 1 outer + 18 inner (j=0..=17) = 19 checks (+ 1 VM entry = 20 gas_used).
    {
        safepoint_reset();
        let mut vm = VM::new();
        vm.load_mir_module(&module).expect("load module");
        vm.set_backend(VmBackend::Cranelift);
        vm.set_gas_budget(10_000);
        let arr = Value::array((0..17).map(Value::Int).collect());
        let err = vm
            .call("nested_arr_oob", vec![arr, Value::Int(3), Value::Int(100)])
            .expect_err("nested_arr_oob must trap OOB at j=17");
        assert!(
            err.to_string().to_lowercase().contains("bound")
                || err.to_string().to_lowercase().contains("index")
                || lust_mir_get_status() == MIR_STATUS_ERR_OUT_OF_BOUNDS,
            "Expected OOB error, got: {err}"
        );
        assert_eq!(
            vm.gas_used(),
            20,
            "Mid-quantum OOB inside nested inner loop must reconcile exact gas_used"
        );
        assert_eq!(safepoint_stats().safepoints_checked, 19);
    }

    // (g) Non-unit gas amounts on nested loop (outer amt=3, inner amt=5):
    //     exact OOG boundary AND exact normal-exit reconciliation!
    {
        let amt_fn = build_nested_loop_fn("nested_amt3_5", 3, 5);
        let mut amt_mod = MirModule::new("ng_w14_1_amt_mod".into());
        amt_mod.add_function(amt_fn);
        MirVerifier::verify_module(&amt_mod).expect("verify nested_amt3_5");

        // Normal exit: outer=3, inner=70 -> outer checks = 4 (*3=12), inner checks = 3*71=213 (*5=1065),
        // total loop gas = 1077 (+ 1 VM entry = 1078 gas_used)
        safepoint_reset();
        let mut vm = VM::new();
        vm.load_mir_module(&amt_mod).expect("load amt_mod");
        vm.set_backend(VmBackend::Cranelift);
        vm.set_gas_budget(50_000);
        let res = vm
            .call("nested_amt3_5", vec![Value::Int(3), Value::Int(70)])
            .expect("nested_amt3_5 must succeed");
        assert_eq!(res, Value::Int(210));
        assert_eq!(vm.gas_used(), 1078);
        assert_eq!(vm.gas_remaining(), Some(50_000 - 1078));
        assert_eq!(safepoint_stats().safepoints_checked, 4 + 213);

        // OOG boundary sweep around 63*5 = 315:
        for budget in [1u64, 4, 8, 9, 316, 319, 320, 324, 325] {
            safepoint_reset();
            let mut vm = VM::new();
            vm.load_mir_module(&amt_mod).expect("load amt_mod");
            vm.set_backend(VmBackend::Cranelift);
            vm.set_gas_budget(budget);
            let err = vm
                .call("nested_amt3_5", vec![Value::Int(5), Value::Int(200)])
                .expect_err("nested_amt3_5 must hit OOG");
            assert!(err.to_string().contains("Out of gas"), "budget={budget}: {err}");
            assert_eq!(vm.gas_remaining(), Some(0));
            let rem = vm.mir_context.gas_remaining();
            assert!(
                rem <= 0 && rem >= -5,
                "Non-unit nested OOG at budget={budget} must not overshoot: rem={rem}"
            );
        }
    }

    // (h) Unbounded mode with small replenishment chunk (chunk=64) on nested loop
    {
        safepoint_reset();
        let mut vm = VM::new();
        vm.load_mir_module(&module).expect("load module");
        vm.set_backend(VmBackend::Cranelift);

        reset_unbounded_gas_replenish_count();
        let _chunk_guard = set_unbounded_gas_chunk_override(Some(64));
        let res = vm
            .call("nested_loop", vec![Value::Int(4), Value::Int(150)])
            .expect("Unbounded nested loop with chunk=64 must succeed");
        assert_eq!(res, Value::Int(600));
        assert!(
            unbounded_gas_replenish_count() >= 8,
            "Expected >= 8 replenishments, got {}",
            unbounded_gas_replenish_count()
        );
        // 5 outer + 4 * 151 inner = 609 safepoints
        assert_eq!(safepoint_stats().safepoints_checked, 609);
    }
}

// ============================================================================
// 3. TEST-DEOPT: Forced Deopt at Iteration 17 (Mid-Quantum) & Quantum Boundaries
// ============================================================================

#[cfg(feature = "test-deopt")]
#[test]
fn test_ng_w14_1_forced_deopt_reconciles_quantum_gas() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let src = r#"
        function single_sum(n: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                acc = acc + i
                i = i + 1
            end
            return acc
        end

        function nested_sum(outer: int, inner: int): int
            local i: int = 0
            local total: int = 0
            while i < outer do
                local j: int = 0
                while j < inner do
                    total = total + i + j
                    j = j + 1
                end
                i = i + 1
            end
            return total
        end

        function nested_arr_sum(arr: Array<int>, passes: int, n: int): int
            local total: int = 0
            local p: int = 0
            while p < passes do
                local j: int = 0
                while j < n do
                    total = total + arr[j]
                    j = j + 1
                end
                p = p + 1
            end
            return total
        end
    "#;
    let module = parse_and_lower_module(src, "ng_w14_1_deopt_mod");

    // (a) Full VM deopt at iteration 17 (mid-quantum) + quantum boundaries (63, 64, 65)
    //     on single_sum, nested_sum, and peeled-header nested_arr_sum!
    for ordinal in [17u64, 63, 64, 65] {
        // 1. Single loop: n = 80
        let expected_single = 80 * 79 / 2;
        safepoint_reset();
        let mut vm = VM::new();
        vm.load_mir_module(&module).expect("load deopt_mod");
        vm.set_backend(VmBackend::Cranelift);
        vm.set_gas_budget(100_000);
        safepoint_request_deopt_on_poll(None, ordinal);
        let res = vm
            .call("single_sum", vec![Value::Int(80)])
            .expect("single_sum deopt + VM resume must succeed");
        assert_eq!(res, Value::Int(expected_single));
        assert!(vm.cranelift_entry_count > 0);
        assert!(
            vm.gas_used() > ordinal,
            "gas_used must include native prefix + VM tail (ordinal={ordinal}, gas_used={})",
            vm.gas_used()
        );

        // 2. Nested loop: outer = 3, inner = 30
        //    sum_{i=0..3} sum_{j=0..30} (i + j) = 3 * (30*29/2) + 30 * (0+1+2) = 1305 + 90 = 1395
        safepoint_reset();
        let mut vm_nested = VM::new();
        vm_nested.load_mir_module(&module).expect("load deopt_mod");
        vm_nested.set_backend(VmBackend::Cranelift);
        vm_nested.set_gas_budget(100_000);
        safepoint_request_deopt_on_poll(None, ordinal);
        let res_nested = vm_nested
            .call("nested_sum", vec![Value::Int(3), Value::Int(30)])
            .expect("nested_sum deopt + VM resume must succeed");
        assert_eq!(res_nested, Value::Int(1395));
        assert!(vm_nested.cranelift_entry_count > 0);

        // Verify exact VM parity against pure bytecode execution
        let mut vm_bc = VM::new();
        vm_bc.load_mir_module(&module).expect("load bc");
        vm_bc.set_backend(VmBackend::Bytecode);
        let res_bc = vm_bc
            .call("nested_sum", vec![Value::Int(3), Value::Int(30)])
            .expect("bc nested_sum");
        assert_eq!(res_nested, res_bc);

        // 3. Nested peeled-header array loop: passes = 3, n = 30
        safepoint_reset();
        let mut vm_arr = VM::new();
        vm_arr.load_mir_module(&module).expect("load deopt_mod");
        vm_arr.set_backend(VmBackend::Cranelift);
        vm_arr.set_gas_budget(100_000);
        safepoint_request_deopt_on_poll(None, ordinal);
        let arr = Value::array((1..=30).map(Value::Int).collect());
        let res_arr = vm_arr
            .call("nested_arr_sum", vec![arr, Value::Int(3), Value::Int(30)])
            .expect("nested_arr_sum deopt + VM resume must succeed");
        assert_eq!(res_arr, Value::Int(3 * (30 * 31 / 2)));
        assert!(vm_arr.cranelift_entry_count > 0);
    }

    // (b) Exact post-deopt safepoint & gas reconciliation at iteration 17 (tail < OSR threshold):
    //     both single_sum(25) and nested_sum(2, 12) stay below OSR re-entry after deopting at 17,
    //     asserting exact safepoints_checked == 17 and strictly lower gas than deopting at 1!
    {
        safepoint_reset();
        let mut vm_deopt_17 = VM::new();
        vm_deopt_17.load_mir_module(&module).expect("load deopt_mod");
        vm_deopt_17.set_backend(VmBackend::Cranelift);
        vm_deopt_17.set_gas_budget(10_000);
        safepoint_request_deopt_on_poll(None, 17);
        let res17 = vm_deopt_17
            .call("single_sum", vec![Value::Int(25)])
            .expect("single_sum(25) deopt at 17 must succeed");
        assert_eq!(res17, Value::Int(25 * 24 / 2));
        assert_eq!(safepoint_stats().safepoints_checked, 17);
        let gas_deopt_17 = vm_deopt_17.gas_used();

        safepoint_reset();
        let mut vm_deopt_1 = VM::new();
        vm_deopt_1.load_mir_module(&module).expect("load deopt_mod");
        vm_deopt_1.set_backend(VmBackend::Cranelift);
        vm_deopt_1.set_gas_budget(10_000);
        safepoint_request_deopt_on_poll(None, 1);
        let res1 = vm_deopt_1
            .call("single_sum", vec![Value::Int(25)])
            .expect("single_sum(25) deopt at 1 must succeed");
        assert_eq!(res1, Value::Int(25 * 24 / 2));
        assert_eq!(safepoint_stats().safepoints_checked, 1);
        let gas_deopt_1 = vm_deopt_1.gas_used();

        assert!(
            gas_deopt_17 < gas_deopt_1,
            "Deopting at ordinal 17 must consume fewer total VM gas units than deopting at 1 (17={gas_deopt_17}, 1={gas_deopt_1})"
        );

        // Nested loop short-tail deopt at ordinal 17 (outer=2, inner=12 -> 1 + 13 + 1 + 13 + 1 = 29 headers total)
        safepoint_reset();
        let mut vm_nested_17 = VM::new();
        vm_nested_17.load_mir_module(&module).expect("load deopt_mod");
        vm_nested_17.set_backend(VmBackend::Cranelift);
        vm_nested_17.set_gas_budget(10_000);
        safepoint_request_deopt_on_poll(None, 17);
        let res_n17 = vm_nested_17
            .call("nested_sum", vec![Value::Int(2), Value::Int(12)])
            .expect("nested_sum(2, 12) deopt at 17 must succeed");
        // i=0: 0+1+..+11 = 66; i=1: 1+2+..+12 = 78 -> total = 144
        assert_eq!(res_n17, Value::Int(144));
        assert_eq!(safepoint_stats().safepoints_checked, 17);
    }

    safepoint_reset();
}

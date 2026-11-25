#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use lust::ast::Span;
use lust::bytecode::Value;
use lust::effect_abi::InterruptReason;
use lust::{
    EmbeddedModule, LustConfig, compile_program_with_config, load_program_from_embedded,
};
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{
    analyze_module_call_effects, get_current_sticky_error, lust_mir_alloc_array_typed,
    lust_mir_get_status, lust_mir_safepoint, lust_mir_safepoint_state_ptr, lust_mir_set_gas,
    lust_mir_set_status, push_native_invocation, reset_native_instrumentation_counters,
    safepoint_enter_mask, safepoint_exit_mask, safepoint_register_finalizer,
    safepoint_request_deopt, safepoint_request_interrupt, safepoint_reset,
    safepoint_set_gas_budget_active, safepoint_stats, take_current_sticky_error,
    take_deopt_payload, CraneliftJitCompiler, CRANELIFT_NATIVE_ENTRY_COUNT, MIR_ARRAY_ELEM_TAG_INT,
    MIR_STATUS_DEOPT, MIR_STATUS_ERR_DIV_BY_ZERO, MIR_STATUS_ERR_INTERRUPTED,
    MIR_STATUS_ERR_OUT_OF_BOUNDS, MIR_STATUS_ERR_OUT_OF_GAS, MIR_STATUS_OK, SLOW_FLAG_DEOPT,
    SLOW_FLAG_FINALIZERS, SLOW_FLAG_GAS_BUDGET, SLOW_FLAG_INTERRUPT, SLOW_FLAG_MASKED,
};
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::{safepoint_request_deopt_on_poll, SLOW_FLAG_TEST_DEOPT};
use lust::mir::ir::{
    FunctionSignature, MirBinaryOp, MirConstant, MirFunction, MirModule, MirType, Operand, Place,
    ResourceKind, Rvalue, Statement, Terminator,
};
use lust::mir::lowering::AstLowering;
use lust::mir::opt::MirOptimizer;
use lust::mir::runtime_helpers::{enter_mir_context, MirContext};
use lust::parser::Parser;

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn lower_module_from_source(src: &str, name: &str) -> MirModule {
    let mut interner = Interner::new();
    let tokens = Lexer::new(src, &mut interner)
        .tokenize()
        .expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");
    let mut module = MirModule::new(name.into());
    for item in &items {
        if let lust::ast::ItemKind::Function(func) = &item.kind {
            let mut mir_fn = AstLowering::lower_function(func).expect("lower_function");
            MirOptimizer::optimize_function(&mut mir_fn);
            module.add_function(mir_fn);
        }
    }
    module
}

fn lower_single_function(src: &str) -> MirFunction {
    let module = lower_module_from_source(src, "perf_t1_2_and_3");
    module
        .functions
        .into_iter()
        .next()
        .expect("expected at least one function")
}

/// Build a MIR loop function:
/// ```text
/// fn loop_with_cb(n: int) -> int {
///     i = 0;
///     while i < n {
///         _ = cb_name(i);
///         i = i + 1;
///     }
///     return i;
/// }
/// ```
fn build_loop_calling_callback(fn_name: &str, cb_name: &str) -> MirFunction {
    let sig = FunctionSignature {
        name: fn_name.into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut f = MirFunction::new(sig, Span::dummy());
    let n = f.params[0];
    let i = f.add_local(Some("i".into()), MirType::Int, true, Span::dummy());
    let cond = f.add_local(Some("cond".into()), MirType::Bool, true, Span::dummy());
    let cb_ret = f.add_local(Some("cb_ret".into()), MirType::Int, true, Span::dummy());

    let bb_hdr = f.new_block(); // bb1 (loop header)
    let bb_body = f.new_block(); // bb2
    let bb_after_cb = f.new_block(); // bb3
    let bb_exit = f.new_block(); // bb4

    // bb0: i = 0; goto bb_hdr
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    // bb_hdr: cond = i < n; brif cond -> bb_body else bb_exit
    f.blocks[bb_hdr.0].statements.push(Statement::Assign(
        Place::Local(cond),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(i)),
            Operand::Copy(Place::Local(n)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_hdr.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond)),
        then_block: bb_body,
        else_block: bb_exit,
        span: Span::dummy(),
    });

    // bb_body: cb_ret = call cb_name(i) -> bb_after_cb
    f.blocks[bb_body.0].terminator = Some(Terminator::Call {
        callee: cb_name.into(),
        args: vec![Operand::Copy(Place::Local(i))],
        destination: Place::Local(cb_ret),
        success: bb_after_cb,
        error: None,
        span: Span::dummy(),
    });

    // bb_after_cb: i = i + 1; goto bb_hdr
    f.blocks[bb_after_cb.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_after_cb.0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    // bb_exit: return i
    f.blocks[bb_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(i))),
        Span::dummy(),
    ));

    f
}

struct ClifIrBlocks {
    entry_block: String,
    blocks: HashMap<String, Vec<String>>,
    cold_blocks: HashSet<String>,
    sym_map: HashMap<String, String>,
    block_order: Vec<String>,
}

impl ClifIrBlocks {
    fn parse(ir: &str) -> Self {
        let mut sym_map: HashMap<String, String> = HashMap::new();
        let mut blocks: HashMap<String, Vec<String>> = HashMap::new();
        let mut cold_blocks: HashSet<String> = HashSet::new();
        let mut block_order: Vec<String> = Vec::new();
        let mut cur_block: Option<String> = None;

        for raw_line in ir.lines() {
            let line = raw_line.trim();
            if let Some(rest) = line.strip_prefix("; sym ") {
                if let Some((fref, sym)) = rest.split_once(" = ") {
                    sym_map.insert(fref.trim().to_string(), sym.trim().to_string());
                }
                continue;
            }
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

        let entry_block = block_order.first().cloned().unwrap_or_default();
        Self {
            entry_block,
            blocks,
            cold_blocks,
            sym_map,
            block_order,
        }
    }

    fn extract_call_sym(&self, line: &str) -> Option<String> {
        let idx = line.find("call ")?;
        let after = &line[idx + 5..];
        let callee_ref = after
            .split(|c: char| c == '(' || c.is_whitespace())
            .next()?
            .trim();
        Some(
            self.sym_map
                .get(callee_ref)
                .cloned()
                .unwrap_or_else(|| callee_ref.to_string()),
        )
    }

    fn calls_in_block(&self, block: &str) -> Vec<String> {
        self.blocks
            .get(block)
            .map(|lines| {
                lines
                    .iter()
                    .filter_map(|l| self.extract_call_sym(l))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn all_call_sites(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for b in &self.block_order {
            for sym in self.calls_in_block(b) {
                out.push((b.clone(), sym));
            }
        }
        out
    }

    #[allow(dead_code)]
    fn reachable_call_sites(&self) -> Vec<(String, String)> {
        let extract_targets = |line: &str| -> Vec<String> {
            let mut out = Vec::new();
            for tok in line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
                if tok.starts_with("block")
                    && tok.len() > 5
                    && tok[5..].chars().all(|c| c.is_ascii_digit())
                {
                    out.push(tok.to_string());
                }
            }
            out
        };
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        if !self.entry_block.is_empty() {
            visited.insert(self.entry_block.clone());
            queue.push_back(self.entry_block.clone());
        }
        let mut out = Vec::new();
        while let Some(b) = queue.pop_front() {
            if let Some(lines) = self.blocks.get(&b) {
                for l in lines {
                    if let Some(sym) = self.extract_call_sym(l) {
                        out.push((b.clone(), sym));
                    }
                    if l.starts_with("jump ")
                        || l.starts_with("brif ")
                        || l.starts_with("br_table ")
                    {
                        for tgt in extract_targets(l) {
                            if visited.insert(tgt.clone()) {
                                queue.push_back(tgt);
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

fn read_slow_flags() -> u64 {
    unsafe { (*lust_mir_safepoint_state_ptr()).slow_flags }
}

// ============================================================================
// PART 1: PERF-T1-2 Core & 6 Mandatory Adversarial Tests
// ============================================================================

#[test]
fn test_perf_t1_2_safepoint_inline_fast_path_and_equivalence() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();
    assert_eq!(read_slow_flags(), 0);

    let loop_src = r#"
        function sum_loop(n: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                acc = acc + i
                i = i + 1
            end
            return acc
        end
    "#;
    let loop_fn = lower_single_function(loop_src);

    // (a) Verify generated Cranelift IR:
    //   - `lust_mir_safepoint_state_ptr` called ONCE in entry block
    //   - `lust_mir_safepoint_slow` called ONLY in a cold block
    //   - NEVER an unconditional `call lust_mir_safepoint` in any block
    let mut compiler = CraneliftJitCompiler::new().expect("create compiler");
    let ir = compiler
        .dump_function_ir(&loop_fn)
        .expect("dump_function_ir sum_loop");
    let parsed = ClifIrBlocks::parse(&ir);

    let all_sites = parsed.all_call_sites();
    let state_ptr_sites: Vec<&String> = all_sites
        .iter()
        .filter(|(_, sym)| sym == "lust_mir_safepoint_state_ptr")
        .map(|(b, _)| b)
        .collect();
    assert_eq!(
        state_ptr_sites.len(),
        1,
        "Expected `lust_mir_safepoint_state_ptr` called exactly once in hot loop function, got {state_ptr_sites:?}:\n{ir}"
    );
    assert_eq!(
        state_ptr_sites[0], &parsed.entry_block,
        "Expected `lust_mir_safepoint_state_ptr` in entry block {}, got {}:\n{ir}",
        parsed.entry_block, state_ptr_sites[0]
    );

    let slow_sites: Vec<&String> = all_sites
        .iter()
        .filter(|(_, sym)| sym == "lust_mir_safepoint_slow")
        .map(|(b, _)| b)
        .collect();
    assert!(
        !slow_sites.is_empty(),
        "Expected `lust_mir_safepoint_slow` call in cold slow-path block:\n{ir}"
    );
    for b in &slow_sites {
        assert!(
            parsed.cold_blocks.contains(*b),
            "Expected `lust_mir_safepoint_slow` to be placed in a cold block, but {b} is not marked cold:\n{ir}"
        );
    }

    assert!(
        !all_sites.iter().any(|(_, sym)| sym == "lust_mir_safepoint"),
        "Found forbidden unconditional `call lust_mir_safepoint` in loop IR:\n{ir}"
    );

    // Also verify loop-free function has ZERO `lust_mir_safepoint_state_ptr` calls at entry
    let leaf_fn = lower_single_function("function add_one(x: int): int return x + 1 end");
    let mut compiler_leaf = CraneliftJitCompiler::new().expect("create compiler leaf");
    let leaf_ir = compiler_leaf
        .dump_function_ir(&leaf_fn)
        .expect("dump_function_ir add_one");
    let leaf_parsed = ClifIrBlocks::parse(&leaf_ir);
    assert!(
        !leaf_parsed
            .all_call_sites()
            .iter()
            .any(|(_, sym)| sym == "lust_mir_safepoint_state_ptr"
                || sym == "lust_mir_safepoint_slow"
                || sym == "lust_mir_safepoint"),
        "Loop-free function must have zero safepoint entry/slow calls:\n{leaf_ir}"
    );

    // (b) Execute compiled loop and assert exact safepoints_checked == iterations + 1
    let mut exec_compiler = CraneliftJitCompiler::new().expect("create exec compiler");
    let code_ptr = exec_compiler
        .compile_function(&loop_fn)
        .expect("compile_function sum_loop");
    let native_sum_loop: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code_ptr) };

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    safepoint_reset();

    let iters: i64 = 1000;
    let res = native_sum_loop(iters);
    assert_eq!(res, iters * (iters - 1) / 2);

    let stats = safepoint_stats();
    assert_eq!(
        stats.safepoints_checked,
        (iters as u64) + 1,
        "safepoints_checked must equal loop iteration count + 1"
    );
    assert_eq!(stats.safepoints_delivered, 0);
    assert_eq!(stats.safepoints_masked_deferred, 0);

    // Verify SLOW_FLAG_GAS_BUDGET synchronization
    safepoint_set_gas_budget_active(true);
    assert_eq!(read_slow_flags(), SLOW_FLAG_GAS_BUDGET);
    safepoint_set_gas_budget_active(false);
    assert_eq!(read_slow_flags(), 0);
}

static ADV_A_UNMASK_AT_ITER: AtomicI64 = AtomicI64::new(-1);
static ADV_A_LAST_ITER: AtomicI64 = AtomicI64::new(-1);
static ADV_A_NESTED_MODE: AtomicBool = AtomicBool::new(false);

extern "C" fn adv_a_host_tick(iter: i64) -> i64 {
    ADV_A_LAST_ITER.store(iter, Ordering::SeqCst);
    if ADV_A_NESTED_MODE.load(Ordering::SeqCst) {
        // Nested 3 -> 2 -> 1 -> 0: unmask one level on iterations 0, 1, 2
        if iter < 3 {
            let _ = safepoint_exit_mask();
        }
    } else if iter == ADV_A_UNMASK_AT_ITER.load(Ordering::SeqCst) {
        let _ = safepoint_exit_mask();
    }
    0
}

#[test]
fn test_perf_t1_2_adv_a_mask_interrupt_unmask_single_and_nested() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let loop_fn = build_loop_calling_callback("loop_mask_cb", "adv_a_host_tick");
    let mut compiler = unsafe {
        CraneliftJitCompiler::new_with_host_callbacks(&[(
            "adv_a_host_tick",
            vec![MirType::Int],
            MirType::Int,
            adv_a_host_tick as *const u8,
        )])
    }
    .expect("create compiler with host cb");
    let code_ptr = compiler.compile_function(&loop_fn).expect("compile");
    let native_loop: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code_ptr) };

    // Part 1: Single mask 1 -> 0 mid-loop
    {
        safepoint_reset();
        let ctx = MirContext::new();
        let _guard = enter_mir_context(&ctx);
        lust_mir_set_status(MIR_STATUS_OK);

        ADV_A_NESTED_MODE.store(false, Ordering::SeqCst);
        ADV_A_UNMASK_AT_ITER.store(2, Ordering::SeqCst);
        ADV_A_LAST_ITER.store(-1, Ordering::SeqCst);

        assert_eq!(read_slow_flags(), 0);
        safepoint_enter_mask();
        assert_eq!(read_slow_flags(), SLOW_FLAG_MASKED);
        safepoint_request_interrupt(InterruptReason::Timeout);
        assert_eq!(read_slow_flags(), SLOW_FLAG_MASKED | SLOW_FLAG_INTERRUPT);

        // Iterations 0, 1, 2 run while masked -> 3 masked deferrals.
        // At end of iter 2, host_tick calls safepoint_exit_mask() (1 -> 0).
        // slow_flags stays SLOW_FLAG_INTERRUPT != 0, so iter 3 header delivers immediately!
        let ret = native_loop(100);
        assert_eq!(ret, MIR_STATUS_ERR_INTERRUPTED);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_INTERRUPTED);
        assert_eq!(
            ADV_A_LAST_ITER.load(Ordering::SeqCst),
            2,
            "Interrupt must be delivered on the very next loop iteration (iter 3 header) after unmask"
        );
        let stats = safepoint_stats();
        assert_eq!(stats.mask_depth, 0);
        assert_eq!(stats.safepoints_masked_deferred, 3);
        assert_eq!(stats.safepoints_delivered, 1);
        assert_eq!(stats.safepoints_checked, 4);
        assert_eq!(
            read_slow_flags(),
            0,
            "After interrupt consumption with no finalizers/masks, slow_flags must return to 0"
        );
    }

    // Part 2: Nested mask 3 -> 2 -> 1 -> 0 mid-loop
    {
        safepoint_reset();
        let ctx = MirContext::new();
        let _guard = enter_mir_context(&ctx);
        lust_mir_set_status(MIR_STATUS_OK);

        ADV_A_NESTED_MODE.store(true, Ordering::SeqCst);
        ADV_A_LAST_ITER.store(-1, Ordering::SeqCst);

        safepoint_enter_mask();
        safepoint_enter_mask();
        safepoint_enter_mask();
        assert_eq!(safepoint_stats().mask_depth, 3);
        assert_eq!(read_slow_flags(), SLOW_FLAG_MASKED);

        safepoint_request_interrupt(InterruptReason::Cancelled);
        assert_eq!(read_slow_flags(), SLOW_FLAG_MASKED | SLOW_FLAG_INTERRUPT);

        // Iter 0 header (depth=3, deferred #1) -> body unmasks 3->2
        // Iter 1 header (depth=2, deferred #2) -> body unmasks 2->1
        // Iter 2 header (depth=1, deferred #3) -> body unmasks 1->0 (slow_flags = SLOW_FLAG_INTERRUPT!)
        // Iter 3 header (depth=0) -> immediate delivery!
        let ret = native_loop(100);
        assert_eq!(ret, MIR_STATUS_ERR_INTERRUPTED);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_INTERRUPTED);
        assert_eq!(
            ADV_A_LAST_ITER.load(Ordering::SeqCst),
            2,
            "Nested 3->2->1->0 unmask must deliver on the very next loop header (iter 3)"
        );
        let stats = safepoint_stats();
        assert_eq!(stats.mask_depth, 0);
        assert_eq!(stats.safepoints_masked_deferred, 3);
        assert_eq!(stats.safepoints_delivered, 1);
        assert_eq!(stats.safepoints_checked, 4);
        assert_eq!(read_slow_flags(), 0);
    }
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_perf_t1_2_adv_b_test_deopt_poll_ordinal_and_unrelated_fn() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let loop_src = r#"
        function count_loop(n: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                acc = acc + i
                i = i + 1
            end
            return acc
        end
    "#;
    let loop_fn = lower_single_function(loop_src);
    let mut compiler = CraneliftJitCompiler::new().expect("create compiler");
    let code_ptr = compiler.compile_function(&loop_fn).expect("compile");
    let native_count: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code_ptr) };

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);

    // 1. Exact poll ordinal k = 5 on matching function name
    {
        safepoint_reset();
        take_deopt_payload();
        lust_mir_set_status(MIR_STATUS_OK);

        let _inv = push_native_invocation("count_loop".into(), 1, 101);
        safepoint_request_deopt_on_poll(Some("count_loop".into()), 5);
        assert_eq!(read_slow_flags(), SLOW_FLAG_TEST_DEOPT);

        let ret = native_count(20);
        assert_eq!(ret, MIR_STATUS_DEOPT);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_DEOPT);
        assert_eq!(
            safepoint_stats().safepoints_checked,
            5,
            "Deopt on poll ordinal 5 must trigger on exact 5th safepoint check"
        );
        assert_eq!(
            read_slow_flags(),
            0,
            "SLOW_FLAG_TEST_DEOPT must clear immediately when ordinal matches"
        );
        let payload = take_deopt_payload().expect("DeoptPayload must be published");
        assert_eq!(payload.func_name.as_deref(), Some("count_loop"));
    }

    // 2. Unrelated function name must NOT deopt and must complete normally;
    //    dropping unrelated NativeInvocationGuard must clear target if matched or leave clean on reset.
    {
        safepoint_reset();
        take_deopt_payload();
        lust_mir_set_status(MIR_STATUS_OK);

        {
            let _inv = push_native_invocation("count_loop".into(), 1, 102);
            safepoint_request_deopt_on_poll(Some("other_unrelated_fn".into()), 1);
            assert_eq!(read_slow_flags(), SLOW_FLAG_TEST_DEOPT);

            let ret = native_count(10);
            assert_eq!(ret, 45, "Unrelated target fn must not deopt count_loop");
            assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
            assert_eq!(safepoint_stats().safepoints_checked, 11);
            assert!(take_deopt_payload().is_none());
            // Still armed for other_unrelated_fn because count_loop didn't match
            assert_eq!(read_slow_flags(), SLOW_FLAG_TEST_DEOPT);
        }
        // Now push and drop an invocation for other_unrelated_fn — NativeInvocationGuard::drop must clear & sync slow_flags!
        {
            let _inv_other = push_native_invocation("other_unrelated_fn".into(), 1, 103);
            assert_eq!(read_slow_flags(), SLOW_FLAG_TEST_DEOPT);
        }
        assert_eq!(
            read_slow_flags(),
            0,
            "NativeInvocationGuard::drop must clear SLOW_FLAG_TEST_DEOPT via sync_safepoint_fast_flags"
        );
    }
}

#[test]
fn test_perf_t1_2_adv_c_finalizers_and_reentrancy_no_refcell_panic() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    lust_mir_set_status(MIR_STATUS_OK);

    let fin1_saw_checked = Arc::new(AtomicUsize::new(0));
    let fin2_ran = Arc::new(AtomicBool::new(false));

    let f1_ref = fin1_saw_checked.clone();
    safepoint_register_finalizer(move || {
        // Reentrantly read safepoint_stats() AND mutate via safepoint_request_interrupt()!
        // Previously this would panic with RefCell already mutably borrowed.
        let snap = safepoint_stats();
        f1_ref.store(snap.safepoints_checked as usize, Ordering::SeqCst);
        safepoint_request_interrupt(InterruptReason::Cancelled);
    });
    assert_eq!(read_slow_flags(), SLOW_FLAG_FINALIZERS);

    let f2_ref = fin2_ran.clone();
    safepoint_register_finalizer(move || {
        let _ = safepoint_stats();
        f2_ref.store(true, Ordering::SeqCst);
    });
    assert_eq!(read_slow_flags(), SLOW_FLAG_FINALIZERS);

    safepoint_request_interrupt(InterruptReason::Timeout);
    assert_eq!(
        read_slow_flags(),
        SLOW_FLAG_FINALIZERS | SLOW_FLAG_INTERRUPT
    );

    let status = lust_mir_safepoint();
    assert_eq!(status, MIR_STATUS_ERR_INTERRUPTED);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_INTERRUPTED);
    assert_eq!(fin1_saw_checked.load(Ordering::SeqCst), 1);
    assert!(fin2_ran.load(Ordering::SeqCst));

    let stats = safepoint_stats();
    assert_eq!(stats.safepoints_checked, 1);
    assert_eq!(stats.safepoints_delivered, 1);
    assert_eq!(stats.cleanups_run, 2);
    // Finalizer 1 re-armed pending_interrupt(Cancelled) -> slow_flags MUST reflect SLOW_FLAG_INTERRUPT!
    assert_eq!(
        read_slow_flags(),
        SLOW_FLAG_INTERRUPT,
        "Re-armed interrupt from reentrant finalizer must keep SLOW_FLAG_INTERRUPT set"
    );

    // Consuming the re-armed interrupt clears slow_flags completely
    let status2 = lust_mir_safepoint();
    assert_eq!(status2, MIR_STATUS_ERR_INTERRUPTED);
    assert_eq!(safepoint_stats().safepoints_delivered, 2);
    assert_eq!(read_slow_flags(), 0);
}

#[test]
fn test_perf_t1_2_adv_d_nested_native_callee_fail_closed_guard() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();
    take_deopt_payload();
    take_current_sticky_error();

    let src = r#"
        function inner_loop(n: int): int
            local i: int = 0
            local s: int = 0
            while i < n do
                s = s + i
                i = i + 1
            end
            return s
        end

        function outer_caller(n: int): int
            return inner_loop(n)
        end
    "#;
    let module = lower_module_from_source(src, "nested_deopt_mod");
    let mut compiler = CraneliftJitCompiler::new().expect("create compiler");
    let compiled = compiler.compile_module(&module).expect("compile_module");
    let outer: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(compiled["outer_caller"]) };

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    lust_mir_set_status(MIR_STATUS_OK);

    let _inv = push_native_invocation("outer_caller".into(), 1, 1);
    safepoint_request_deopt(true);
    assert_eq!(read_slow_flags(), SLOW_FLAG_DEOPT);

    let ret = outer(10);
    assert_ne!(ret, MIR_STATUS_DEOPT, "Nested native callee must NOT emit MIR_STATUS_DEOPT");
    assert_ne!(
        lust_mir_get_status(),
        MIR_STATUS_OK,
        "Nested native callee deopt must set error status"
    );
    let sticky = get_current_sticky_error().expect("Sticky error must be recorded");
    assert!(
        sticky.contains("Nested native-to-native deopt"),
        "Unexpected sticky error: {sticky}"
    );
    assert!(take_deopt_payload().is_none());
    assert_eq!(
        read_slow_flags(),
        0,
        "Failed nested deopt must clear deopt_requested and sync slow_flags to 0"
    );
    take_current_sticky_error();
}

static ADV_E_MODE: AtomicI64 = AtomicI64::new(0); // 1 = interrupt at iter 5, 2 = deopt at iter 5
static ADV_E_CALLS: AtomicI64 = AtomicI64::new(0);

extern "C" fn adv_e_host_cb(iter: i64) -> i64 {
    ADV_E_CALLS.fetch_add(1, Ordering::SeqCst);
    if iter == 4 {
        match ADV_E_MODE.load(Ordering::SeqCst) {
            1 => safepoint_request_interrupt(InterruptReason::Cancelled),
            2 => safepoint_request_deopt(true),
            _ => {}
        }
    }
    0
}

#[test]
fn test_perf_t1_2_adv_e_reentrant_host_callback_mid_loop_interrupt_and_deopt() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let loop_fn = build_loop_calling_callback("loop_reentrant_cb", "adv_e_host_cb");
    let mut compiler = unsafe {
        CraneliftJitCompiler::new_with_host_callbacks(&[(
            "adv_e_host_cb",
            vec![MirType::Int],
            MirType::Int,
            adv_e_host_cb as *const u8,
        )])
    }
    .expect("create compiler");
    let code_ptr = compiler.compile_function(&loop_fn).expect("compile");
    let native_loop: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code_ptr) };

    // 1. Mid-loop interrupt armed during fast-path execution (slow_flags starts at 0!)
    {
        safepoint_reset();
        let ctx = MirContext::new();
        let _guard = enter_mir_context(&ctx);
        lust_mir_set_status(MIR_STATUS_OK);
        ADV_E_MODE.store(1, Ordering::SeqCst);
        ADV_E_CALLS.store(0, Ordering::SeqCst);
        assert_eq!(read_slow_flags(), 0);

        let ret = native_loop(1000);
        assert_eq!(ret, MIR_STATUS_ERR_INTERRUPTED);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_INTERRUPTED);
        assert_eq!(
            ADV_E_CALLS.load(Ordering::SeqCst),
            5,
            "Host cb ran for iters 0..=4; iter 5 header must stop immediately via inline slow_flags load"
        );
        let stats = safepoint_stats();
        assert_eq!(stats.safepoints_checked, 6);
        assert_eq!(stats.safepoints_delivered, 1);
    }

    // 2. Mid-loop deopt armed during fast-path execution (slow_flags starts at 0!)
    {
        safepoint_reset();
        take_deopt_payload();
        let ctx = MirContext::new();
        let _guard = enter_mir_context(&ctx);
        lust_mir_set_status(MIR_STATUS_OK);
        ADV_E_MODE.store(2, Ordering::SeqCst);
        ADV_E_CALLS.store(0, Ordering::SeqCst);
        assert_eq!(read_slow_flags(), 0);

        let _inv = push_native_invocation("loop_reentrant_cb".into(), 1, 42);
        let ret = native_loop(1000);
        assert_eq!(ret, MIR_STATUS_DEOPT);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_DEOPT);
        assert_eq!(
            ADV_E_CALLS.load(Ordering::SeqCst),
            5,
            "Host cb ran for iters 0..=4; iter 5 header must immediately deopt"
        );
        assert_eq!(safepoint_stats().safepoints_checked, 6);
        let payload = take_deopt_payload().expect("DeoptPayload must be published");
        assert_eq!(payload.func_name.as_deref(), Some("loop_reentrant_cb"));
        // Check captured `i` local equals 5!
        let i_val = payload
            .live_locals
            .iter()
            .find(|(idx, _)| *idx == 2) // param n=0, ret=1, i=2
            .map(|(_, v)| *v);
        assert_eq!(i_val, Some(5), "Captured live local i must equal 5 at deopt");
    }
}

static ADV_F_TOGGLE_AT: AtomicI64 = AtomicI64::new(5);

extern "C" fn adv_f_toggle_mask_cb(iter: i64) -> i64 {
    let t = ADV_F_TOGGLE_AT.load(Ordering::SeqCst);
    if iter == t {
        // Switch from fast path (slow_flags == 0) to slow path (SLOW_FLAG_MASKED) mid-loop!
        safepoint_enter_mask();
    } else if iter == t + 5 {
        // Switch back to fast path (slow_flags == 0) mid-loop!
        let _ = safepoint_exit_mask();
    }
    0
}

#[test]
fn test_perf_t1_2_adv_f_safepoints_checked_exact_parity_fast_and_mixed() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let loop_fn = build_loop_calling_callback("loop_mixed_parity", "adv_f_toggle_mask_cb");
    let mut compiler = unsafe {
        CraneliftJitCompiler::new_with_host_callbacks(&[(
            "adv_f_toggle_mask_cb",
            vec![MirType::Int],
            MirType::Int,
            adv_f_toggle_mask_cb as *const u8,
        )])
    }
    .expect("create compiler");
    let code_ptr = compiler.compile_function(&loop_fn).expect("compile");
    let native_loop: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code_ptr) };

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);

    // 1. Pure fast-path loop (20 iterations -> 21 checks)
    safepoint_reset();
    ADV_F_TOGGLE_AT.store(9999, Ordering::SeqCst);
    assert_eq!(native_loop(20), 20);
    assert_eq!(safepoint_stats().safepoints_checked, 21);

    // 2. Mixed fast-path + slow-path (masked for 5 iterations in the middle) + fast-path:
    //    slow_path must NOT double-increment safepoints_checked!
    safepoint_reset();
    ADV_F_TOGGLE_AT.store(5, Ordering::SeqCst);
    assert_eq!(native_loop(20), 20);
    assert_eq!(
        safepoint_stats().safepoints_checked,
        21,
        "Mixed fast/slow loop must record exact 21 checks (no double counting in lust_mir_safepoint_slow!)"
    );
    assert_eq!(read_slow_flags(), 0);
}

// ============================================================================
// PART 2: PERF-T1-3 Native-to-Native Call Fast Path & Closed-World Allowlist
// ============================================================================

#[test]
fn test_perf_t1_3_internal_call_fast_path_and_nested_deopt_proof() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();
    reset_native_instrumentation_counters();
    take_deopt_payload();
    take_current_sticky_error();

    let src = r#"
        function fib(n: int): int
            if n <= 1 then
                return n
            end
            return fib(n - 1) + fib(n - 2)
        end

        function inner_loop(n: int): int
            local i: int = 0
            local s: int = 0
            while i < n do
                s = s + i
                i = i + 1
            end
            return s
        end

        function outer_calls_loop(n: int): int
            return inner_loop(n)
        end

        function rec_loop(n: int): int
            if n <= 0 then
                return 0
            end
            local i: int = 0
            while i < 2 do
                i = i + 1
            end
            return i + rec_loop(n - 1)
        end

        function mut_a(n: int): int
            if n <= 0 then
                return 0
            end
            return mut_b(n - 1)
        end

        function mut_b(n: int): int
            if n <= 0 then
                return inner_loop(3)
            end
            return mut_a(n - 1)
        end
    "#;
    let module = lower_module_from_source(src, "perf_t1_3_mod");

    // Verify compile-time fixed-point analysis directly
    let effects = analyze_module_call_effects(&module, false);
    assert_eq!(effects["fib"], (false, false), "fib must be pure non-deopt non-fail");
    assert_eq!(effects["inner_loop"], (true, true), "inner_loop has loop safepoint");
    assert_eq!(effects["outer_calls_loop"], (true, true), "outer_calls_loop transitively calls loop");
    assert_eq!(effects["rec_loop"], (true, true), "rec_loop has loop");
    assert_eq!(effects["mut_a"], (true, true), "mut_a -> mut_b -> inner_loop must be can_deopt=true");
    assert_eq!(effects["mut_b"], (true, true), "mut_b -> inner_loop must be can_deopt=true");

    // Inspect generated Cranelift IR via dump_module_ir
    let mut ir_compiler = CraneliftJitCompiler::new().expect("create ir compiler");
    let ir_map = ir_compiler
        .dump_module_ir(&module)
        .expect("dump_module_ir");

    // 1. Public export wrapper `fib` calls `lust_mir_record_native_entry` once and `fib$fast` once
    let fib_wrapper = ClifIrBlocks::parse(ir_map.get("fib").expect("fib wrapper IR"));
    let fib_wrapper_calls: Vec<String> = fib_wrapper
        .all_call_sites()
        .into_iter()
        .map(|(_, s)| s)
        .collect();
    assert_eq!(
        fib_wrapper_calls,
        vec![
            "lust_mir_record_native_entry".to_string(),
            "fib$fast".to_string()
        ],
        "Public wrapper `fib` must call record_native_entry once then `fib$fast`:\n{}",
        ir_map["fib"]
    );

    // 2. Internal body `fib$fast` calls `fib$fast` directly with ZERO of the 4 C-ABI helpers!
    let fib_fast_ir = ir_map.get("fib$fast").expect("fib$fast IR");
    let fib_fast = ClifIrBlocks::parse(fib_fast_ir);
    let fib_fast_reachable: Vec<String> = fib_fast
        .reachable_call_sites()
        .into_iter()
        .map(|(_, s)| s)
        .collect();
    assert!(
        fib_fast_reachable.iter().any(|s| s == "fib$fast"),
        "fib$fast must recursively call fib$fast directly:\n{fib_fast_ir}"
    );
    for forbidden in [
        "lust_mir_record_native_entry",
        "lust_mir_enter_native_callee",
        "lust_mir_exit_native_callee",
        "lust_mir_get_status",
        "lust_mir_safepoint_state_ptr",
        "lust_mir_safepoint_slow",
        "lust_mir_safepoint",
    ] {
        assert!(
            !fib_fast_reachable.iter().any(|s| s == forbidden),
            "fib$fast reachable IR must NOT call {forbidden}:\n{fib_fast_ir}"
        );
    }

    // 3. `outer_calls_loop` MUST retain `enter_native_callee`, `exit_native_callee`, and `get_status`
    let outer_fast_ir = ir_map
        .get("outer_calls_loop$fast")
        .or_else(|| ir_map.get("outer_calls_loop"))
        .expect("outer_calls_loop body IR");
    let outer_fast = ClifIrBlocks::parse(outer_fast_ir);
    let outer_reachable: Vec<String> = outer_fast
        .reachable_call_sites()
        .into_iter()
        .map(|(_, s)| s)
        .collect();
    for required in [
        "lust_mir_enter_native_callee",
        "inner_loop$fast",
        "lust_mir_exit_native_callee",
        "lust_mir_get_status",
    ] {
        assert!(
            outer_reachable.iter().any(|s| s == required),
            "outer_calls_loop$fast must retain {required}:\n{outer_fast_ir}"
        );
    }

    // Compile and execute
    let mut compiler = CraneliftJitCompiler::new().expect("create compiler");
    let compiled = compiler.compile_module(&module).expect("compile_module");

    // (a) fib(10) == 55 and increments CRANELIFT_NATIVE_ENTRY_COUNT by EXACTLY 1 (not 177!)
    let fib_fn: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(compiled["fib"]) };
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    lust_mir_set_status(MIR_STATUS_OK);
    reset_native_instrumentation_counters();

    let fib10 = fib_fn(10);
    assert_eq!(fib10, 55, "fib(10) must equal 55");
    assert_eq!(
        CRANELIFT_NATIVE_ENTRY_COUNT.load(Ordering::SeqCst),
        1,
        "fib(10) must increment CRANELIFT_NATIVE_ENTRY_COUNT by exactly 1"
    );

    // (b) outer_calls_loop -> inner_loop, rec_loop, and mut_a -> mut_b -> inner_loop
    //     MUST ALL reject nested deopt fail-closed!
    //     (mut_a(3) -> mut_b(2) -> mut_a(1) -> mut_b(0) -> inner_loop(3))
    for (fn_name, arg) in [("outer_calls_loop", 10_i64), ("rec_loop", 3_i64), ("mut_a", 3_i64)] {
        let f_ptr: extern "C" fn(i64) -> i64 =
            unsafe { std::mem::transmute(compiled[fn_name]) };
        safepoint_reset();
        take_deopt_payload();
        take_current_sticky_error();
        lust_mir_set_status(MIR_STATUS_OK);
        {
            let _inv = push_native_invocation(fn_name.into(), 1, 77);
            // For rec_loop, deopt triggers in the top frame on poll 1 unless we check nested call;
            // outer_calls_loop and mut_a enter nested callees before hitting inner_loop's safepoint!
            if fn_name != "rec_loop" {
                safepoint_request_deopt(true);
                let ret = f_ptr(arg);
                assert_ne!(ret, MIR_STATUS_DEOPT, "{fn_name}: nested deopt must fail closed");
                assert_ne!(lust_mir_get_status(), MIR_STATUS_OK, "{fn_name}: error status required");
                let sticky = get_current_sticky_error()
                    .unwrap_or_else(|| panic!("{fn_name}: nested deopt sticky error missing"));
                assert!(
                    sticky.contains("Nested native-to-native deopt"),
                    "{fn_name}: unexpected sticky error: {sticky}"
                );
            } else {
                // Verify rec_loop's own IR retains enter/exit_native_callee for recursive self-calls
                let rec_ir = ir_map.get("rec_loop$fast").expect("rec_loop$fast IR");
                let rec_parsed = ClifIrBlocks::parse(rec_ir);
                let rec_calls: Vec<String> = rec_parsed
                    .reachable_call_sites()
                    .into_iter()
                    .map(|(_, s)| s)
                    .collect();
                assert!(rec_calls.iter().any(|s| s == "lust_mir_enter_native_callee"));
                assert!(rec_calls.iter().any(|s| s == "lust_mir_exit_native_callee"));
            }
        }
        take_current_sticky_error();
    }
}

/// Build a 2-function module `{ caller(a, b) -> callee(a, b), callee(a, b) -> ... }`
/// where `callee` starts as a pure whitelisted scalar function (`a + b`) and can be
/// mutated with ONE forbidden construct at a time for boundary falsification.
fn make_pure_caller_callee_module() -> MirModule {
    let mut module = MirModule::new("whitelist_boundary_mod".into());

    // callee(a: int, b: int) -> int { t = a + b; return t }
    let callee_sig = FunctionSignature {
        name: "callee".into(),
        param_types: vec![MirType::Int, MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut callee = MirFunction::new(callee_sig, Span::dummy());
    let ca = callee.params[0];
    let cb = callee.params[1];
    let ct = callee.add_local(Some("t".into()), MirType::Int, true, Span::dummy());
    callee.blocks[0].statements.push(Statement::Assign(
        Place::Local(ct),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(ca)),
            Operand::Copy(Place::Local(cb)),
        ),
        Span::dummy(),
    ));
    callee.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(ct))),
        Span::dummy(),
    ));

    // caller(a: int, b: int) -> int { r = call callee(a, b) -> bb1; bb1: return r + 1 }
    let caller_sig = FunctionSignature {
        name: "caller".into(),
        param_types: vec![MirType::Int, MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut caller = MirFunction::new(caller_sig, Span::dummy());
    let pa = caller.params[0];
    let pb = caller.params[1];
    let pr = caller.add_local(Some("r".into()), MirType::Int, true, Span::dummy());
    let pout = caller.add_local(Some("out".into()), MirType::Int, true, Span::dummy());
    let bb1 = caller.new_block();
    caller.blocks[0].terminator = Some(Terminator::Call {
        callee: "callee".into(),
        args: vec![
            Operand::Copy(Place::Local(pa)),
            Operand::Copy(Place::Local(pb)),
        ],
        destination: Place::Local(pr),
        success: bb1,
        error: None,
        span: Span::dummy(),
    });
    caller.blocks[bb1.0].statements.push(Statement::Assign(
        Place::Local(pout),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(pr)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    caller.blocks[bb1.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(pout))),
        Span::dummy(),
    ));

    module.add_function(callee);
    module.add_function(caller);
    module
}

fn caller_has_status_check_in_ir(module: &MirModule, gas_budget_active: bool) -> bool {
    let mut compiler = CraneliftJitCompiler::new().expect("compiler");
    compiler.set_gas_budget_active(gas_budget_active);
    let ir_map = compiler.dump_module_ir(module).expect("dump_module_ir");
    let ir = ir_map
        .get("caller$fast")
        .or_else(|| ir_map.get("caller"))
        .expect("caller body IR");
    let parsed = ClifIrBlocks::parse(ir);
    parsed
        .reachable_call_sites()
        .iter()
        .any(|(_, s)| s == "lust_mir_get_status")
}

#[test]
fn test_perf_t1_3_closed_world_whitelist_flips_fail_closed() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    // 0. Baseline pure module: both callee and caller have can_fail == false and ZERO status check in caller$fast!
    let base = make_pure_caller_callee_module();
    let base_eff = analyze_module_call_effects(&base, false);
    assert_eq!(base_eff["callee"], (false, false));
    assert_eq!(base_eff["caller"], (false, false));
    assert!(
        !caller_has_status_check_in_ir(&base, false),
        "Pure baseline caller$fast must elide lust_mir_get_status"
    );

    // 1. Active gas_budget_active = true flips BOTH callee & caller to can_fail = true and retains status check!
    let gas_flag_eff = analyze_module_call_effects(&base, true);
    assert_eq!(gas_flag_eff["callee"].1, true, "gas_budget_active=true must force can_fail=true");
    assert_eq!(gas_flag_eff["caller"].1, true, "gas_budget_active=true must force can_fail=true");
    assert!(
        caller_has_status_check_in_ir(&base, true),
        "gas_budget_active=true must retain lust_mir_get_status in caller$fast"
    );

    // 2. Individually inject ONE forbidden BinaryOp at a time: Div, Mod, FloorDiv, Pow
    for op in [
        MirBinaryOp::Div,
        MirBinaryOp::Mod,
        MirBinaryOp::FloorDiv,
        MirBinaryOp::Pow,
    ] {
        let mut m = make_pure_caller_callee_module();
        let ca = m.functions[0].params[0];
        let cb = m.functions[0].params[1];
        let ct = lust::mir::ir::LocalId(3);
        m.functions[0].blocks[0].statements[0] = Statement::Assign(
            Place::Local(ct),
            Rvalue::BinaryOp(
                op,
                Operand::Copy(Place::Local(ca)),
                Operand::Copy(Place::Local(cb)),
            ),
            Span::dummy(),
        );
        let eff = analyze_module_call_effects(&m, false);
        assert!(
            eff["callee"].1 && eff["caller"].1,
            "Injecting {op:?} must flip can_fail_transitively to true for callee & caller (got {eff:?})"
        );
        assert!(
            caller_has_status_check_in_ir(&m, false),
            "Injecting {op:?} must retain lust_mir_get_status in caller$fast"
        );

        // Runtime propagation proof for DIV_BY_ZERO on Div/Mod/FloorDiv
        if matches!(op, MirBinaryOp::Div | MirBinaryOp::Mod | MirBinaryOp::FloorDiv) {
            let mut comp = CraneliftJitCompiler::new().expect("compiler");
            let map = comp.compile_module(&m).expect("compile");
            let caller_fn: extern "C" fn(i64, i64) -> i64 =
                unsafe { std::mem::transmute(map["caller"]) };
            let ctx = MirContext::new();
            let _g = enter_mir_context(&ctx);
            lust_mir_set_status(MIR_STATUS_OK);
            let _ = caller_fn(10, 0);
            assert_eq!(
                lust_mir_get_status(),
                MIR_STATUS_ERR_DIV_BY_ZERO,
                "Nested {op:?} by zero must propagate MIR_STATUS_ERR_DIV_BY_ZERO through caller"
            );
        }
    }

    // 3. Inject Statement::ResourceCheck(ResourceKind::Gas, 10) -> flips fail-closed & propagates OUT_OF_GAS
    {
        let mut m = make_pure_caller_callee_module();
        m.functions[0].blocks[0].statements.insert(
            0,
            Statement::ResourceCheck(ResourceKind::Gas, 10, Span::dummy()),
        );
        let eff = analyze_module_call_effects(&m, false);
        assert!(eff["callee"].1 && eff["caller"].1);
        assert!(caller_has_status_check_in_ir(&m, false));

        let mut comp = CraneliftJitCompiler::new().expect("compiler");
        let map = comp.compile_module(&m).expect("compile");
        let caller_fn: extern "C" fn(i64, i64) -> i64 =
            unsafe { std::mem::transmute(map["caller"]) };
        let ctx = MirContext::new();
        let _g = enter_mir_context(&ctx);
        lust_mir_set_status(MIR_STATUS_OK);
        lust_mir_set_gas(5); // < 10 -> OUT_OF_GAS inside callee!
        let _ = caller_fn(1, 2);
        assert_eq!(
            lust_mir_get_status(),
            MIR_STATUS_ERR_OUT_OF_GAS,
            "Nested ResourceCheck(Gas) exhaustion must propagate MIR_STATUS_ERR_OUT_OF_GAS through caller"
        );
    }

    // 4. Inject Place::Index -> flips fail-closed & propagates OUT_OF_BOUNDS across nested call
    {
        let mut m = MirModule::new("idx_fail_mod".into());
        let callee_sig = FunctionSignature {
            name: "callee".into(),
            param_types: vec![MirType::Array(Box::new(MirType::Int)), MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut callee = MirFunction::new(callee_sig, Span::dummy());
        let arr = callee.params[0];
        let idx = callee.params[1];
        let v = callee.add_local(Some("v".into()), MirType::Int, true, Span::dummy());
        callee.blocks[0].statements.push(Statement::Assign(
            Place::Local(v),
            Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(arr)), idx))),
            Span::dummy(),
        ));
        callee.blocks[0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(v))),
            Span::dummy(),
        ));

        let caller_sig = FunctionSignature {
            name: "caller".into(),
            param_types: vec![MirType::Array(Box::new(MirType::Int)), MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut caller = MirFunction::new(caller_sig, Span::dummy());
        let pa = caller.params[0];
        let pi = caller.params[1];
        let pr = caller.add_local(Some("r".into()), MirType::Int, true, Span::dummy());
        let bb1 = caller.new_block();
        caller.blocks[0].terminator = Some(Terminator::Call {
            callee: "callee".into(),
            args: vec![
                Operand::Copy(Place::Local(pa)),
                Operand::Copy(Place::Local(pi)),
            ],
            destination: Place::Local(pr),
            success: bb1,
            error: None,
            span: Span::dummy(),
        });
        caller.blocks[bb1.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(pr))),
            Span::dummy(),
        ));
        m.add_function(callee);
        m.add_function(caller);

        let eff = analyze_module_call_effects(&m, false);
        assert!(eff["callee"].1 && eff["caller"].1);
        assert!(caller_has_status_check_in_ir(&m, false));

        let mut comp = CraneliftJitCompiler::new().expect("compiler");
        let map = comp.compile_module(&m).expect("compile");
        let caller_fn: extern "C" fn(i64, i64) -> i64 =
            unsafe { std::mem::transmute(map["caller"]) };
        let ctx = MirContext::new();
        let _g = enter_mir_context(&ctx);
        let arr_ptr = lust_mir_alloc_array_typed(2, MIR_ARRAY_ELEM_TAG_INT);
        lust_mir_set_status(MIR_STATUS_OK);
        let _ = caller_fn(arr_ptr as i64, 99); // OOB!
        assert_eq!(
            lust_mir_get_status(),
            MIR_STATUS_ERR_OUT_OF_BOUNDS,
            "Nested Place::Index OOB must propagate MIR_STATUS_ERR_OUT_OF_BOUNDS through caller"
        );
    }

    // 5. Inject Place::Field (Tuple field read) -> flips fail-closed
    {
        let mut m = MirModule::new("field_fail_mod".into());
        let tup_ty = MirType::Tuple(vec![MirType::Int, MirType::Int]);
        let callee_sig = FunctionSignature {
            name: "callee".into(),
            param_types: vec![tup_ty.clone()],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut callee = MirFunction::new(callee_sig, Span::dummy());
        let tup = callee.params[0];
        let v = callee.add_local(Some("v".into()), MirType::Int, true, Span::dummy());
        callee.blocks[0].statements.push(Statement::Assign(
            Place::Local(v),
            Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(tup)), 0))),
            Span::dummy(),
        ));
        callee.blocks[0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(v))),
            Span::dummy(),
        ));

        let caller_sig = FunctionSignature {
            name: "caller".into(),
            param_types: vec![tup_ty],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut caller = MirFunction::new(caller_sig, Span::dummy());
        let pt = caller.params[0];
        let pr = caller.add_local(Some("r".into()), MirType::Int, true, Span::dummy());
        let bb1 = caller.new_block();
        caller.blocks[0].terminator = Some(Terminator::Call {
            callee: "callee".into(),
            args: vec![Operand::Copy(Place::Local(pt))],
            destination: Place::Local(pr),
            success: bb1,
            error: None,
            span: Span::dummy(),
        });
        caller.blocks[bb1.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(pr))),
            Span::dummy(),
        ));
        m.add_function(callee);
        m.add_function(caller);

        let eff = analyze_module_call_effects(&m, false);
        assert!(eff["callee"].1 && eff["caller"].1, "Place::Field must flip can_fail=true");
        assert!(caller_has_status_check_in_ir(&m, false));
    }

    // 6. Inject Rvalue::Aggregate (array allocation inside callee) -> flips fail-closed
    {
        let mut m = make_pure_caller_callee_module();
        let arr_loc = m.functions[0].add_local(
            Some("arr".into()),
            MirType::Array(Box::new(MirType::Int)),
            true,
            Span::dummy(),
        );
        let ca = m.functions[0].params[0];
        m.functions[0].blocks[0].statements.insert(
            0,
            Statement::Assign(
                Place::Local(arr_loc),
                Rvalue::Aggregate(
                    MirType::Array(Box::new(MirType::Int)),
                    vec![Operand::Copy(Place::Local(ca))],
                ),
                Span::dummy(),
            ),
        );
        let eff = analyze_module_call_effects(&m, false);
        assert!(
            eff["callee"].1 && eff["caller"].1,
            "Rvalue::Aggregate must flip can_fail=true"
        );
        assert!(caller_has_status_check_in_ir(&m, false));
    }

    // 7. Inject unknown / external callee -> flips BOTH can_deopt and can_fail to true!
    {
        let mut m = make_pure_caller_callee_module();
        let ca = m.functions[0].params[0];
        let ext_r = m.functions[0].add_local(Some("ext_r".into()), MirType::Int, true, Span::dummy());
        let bb_after = m.functions[0].new_block();
        m.functions[0].blocks[0].statements.clear();
        m.functions[0].blocks[0].terminator = Some(Terminator::Call {
            callee: "some_unknown_external_fn".into(),
            args: vec![Operand::Copy(Place::Local(ca))],
            destination: Place::Local(ext_r),
            success: bb_after,
            error: None,
            span: Span::dummy(),
        });
        m.functions[0].blocks[bb_after.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(ext_r))),
            Span::dummy(),
        ));

        let eff = analyze_module_call_effects(&m, false);
        assert_eq!(
            eff["callee"],
            (true, true),
            "Unknown/external callee must flip both can_deopt and can_fail to true"
        );
        assert_eq!(
            eff["caller"],
            (true, true),
            "Caller of function with unknown/external callee must transitively flip to (true, true)"
        );
    }
}

#[test]
fn test_perf_t1_2_safepoint_reset_clears_all_armed_flags_and_counter() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    lust_mir_set_status(MIR_STATUS_OK);

    // 1. Exercise delivered interrupt + finalizer cleanup so safepoints_delivered > 0 and cleanups_run > 0.
    safepoint_register_finalizer(|| {});
    safepoint_request_interrupt(InterruptReason::Cancelled);
    let delivered_status = lust_mir_safepoint();
    assert_eq!(delivered_status, MIR_STATUS_ERR_INTERRUPTED);
    lust_mir_set_status(MIR_STATUS_OK);

    // 2. Exercise masked deferral so mask_depth > 0 and safepoints_masked_deferred > 0.
    safepoint_enter_mask();
    safepoint_request_interrupt(InterruptReason::Timeout);
    let masked_status = lust_mir_safepoint();
    assert_eq!(masked_status, MIR_STATUS_OK);

    // 3. Arm ALL remaining slow flags simultaneously.
    safepoint_request_deopt(true);
    safepoint_register_finalizer(|| {});
    safepoint_set_gas_budget_active(true);
    #[cfg(feature = "test-deopt")]
    safepoint_request_deopt_on_poll(Some("target".into()), 3);

    #[allow(unused_mut)]
    let mut expected_flags = SLOW_FLAG_MASKED
        | SLOW_FLAG_INTERRUPT
        | SLOW_FLAG_DEOPT
        | SLOW_FLAG_FINALIZERS
        | SLOW_FLAG_GAS_BUDGET;
    #[cfg(feature = "test-deopt")]
    {
        expected_flags |= SLOW_FLAG_TEST_DEOPT;
    }

    let (flags_before, checked_before) = unsafe {
        let ptr = lust_mir_safepoint_state_ptr();
        ((*ptr).slow_flags, (*ptr).safepoints_checked)
    };
    assert_eq!(
        flags_before, expected_flags,
        "All slow flags must be armed before safepoint_reset()"
    );
    assert_ne!(flags_before, 0);
    assert!(checked_before > 0);
    let stats_before = safepoint_stats();
    assert!(stats_before.mask_depth > 0);
    assert!(stats_before.safepoints_checked > 0);
    assert!(stats_before.safepoints_masked_deferred > 0);
    assert!(stats_before.safepoints_delivered > 0);
    assert!(stats_before.cleanups_run > 0);

    // 4. Call safepoint_reset() and assert EVERY flag and counter is zeroed.
    safepoint_reset();

    let (flags_after, checked_after) = unsafe {
        let ptr = lust_mir_safepoint_state_ptr();
        ((*ptr).slow_flags, (*ptr).safepoints_checked)
    };
    assert_eq!(flags_after, 0, "safepoint_reset() must clear all slow_flags");
    assert_eq!(
        checked_after, 0,
        "safepoint_reset() must zero SafepointFastState.safepoints_checked"
    );

    let stats_after = safepoint_stats();
    assert_eq!(stats_after.mask_depth, 0);
    assert_eq!(stats_after.safepoints_checked, 0);
    assert_eq!(stats_after.safepoints_masked_deferred, 0);
    assert_eq!(stats_after.safepoints_delivered, 0);
    assert_eq!(stats_after.cleanups_run, 0);
}

#[test]
fn test_perf_t1_3_compile_without_gas_then_enable_vm_gas_budget() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();
    take_deopt_payload();
    take_current_sticky_error();

    let src = r#"
function fib(n: int): int
    if n <= 1 then
        return n
    end
    return fib(n - 1) + fib(n - 2)
end

function gas_helper(n: int): int
    if n <= 0 then
        return 1
    end
    return n + gas_helper(n - 1)
end

function gas_caller(n: int): int
    local a: int = gas_helper(n)
    local b: int = gas_helper(n)
    return a + b
end
"#;

    let entries = &[EmbeddedModule {
        module: "main",
        parent: None,
        source: Some(src),
    }];
    let program = load_program_from_embedded(entries, "main").expect("load embedded program");
    let mut cfg = LustConfig::default().with_cranelift_jit(true);
    cfg.set_use_mir(false);
    let mut vm = compile_program_with_config(program, &cfg).expect("compile embedded program");

    let fib_name = vm
        .mir_functions
        .keys()
        .find(|k| k.ends_with("fib"))
        .cloned()
        .expect("fib in mir_functions");
    let gas_helper_name = vm
        .mir_functions
        .keys()
        .find(|k| k.ends_with("gas_helper"))
        .cloned()
        .expect("gas_helper in mir_functions");
    let gas_caller_name = vm
        .mir_functions
        .keys()
        .find(|k| k.ends_with("gas_caller"))
        .cloned()
        .expect("gas_caller in mir_functions");

    // Inject Statement::ResourceCheck(ResourceKind::Gas, 10) into gas_helper before JIT compilation.
    vm.mir_functions
        .get_mut(&gas_helper_name)
        .expect("gas_helper MIR")
        .blocks[0]
        .statements
        .insert(0, Statement::ResourceCheck(ResourceKind::Gas, 10, Span::dummy()));

    // Confirm no gas budget is active at compile time.
    assert!(vm.gas_remaining().is_none());
    vm.compile_cranelift()
        .expect("compile_cranelift with no gas budget");
    let compiled_count_initial = vm.cranelift_compilation_count;
    assert!(compiled_count_initial >= 3);

    // Verify gas_helper was NOT inlined into gas_caller (it is self-recursive) and that
    // analyze_module_call_effects(gas_budget_active=false) marks gas_helper & gas_caller can_fail=true
    // while fib remains (false, false).
    let fib_mir = vm.cranelift_compiled_mir(&fib_name).expect("compiled fib MIR");
    let helper_mir = vm
        .cranelift_compiled_mir(&gas_helper_name)
        .expect("compiled gas_helper MIR");
    let caller_mir = vm
        .cranelift_compiled_mir(&gas_caller_name)
        .expect("compiled gas_caller MIR");
    assert!(
        caller_mir.blocks.iter().any(|b| {
            matches!(&b.terminator, Some(Terminator::Call { callee, .. }) if callee == &gas_helper_name)
        }),
        "gas_caller must retain native-to-native call to gas_helper"
    );
    let mut check_mod = MirModule::new("gas_transition_check".into());
    check_mod.add_function(fib_mir);
    check_mod.add_function(helper_mir);
    check_mod.add_function(caller_mir);
    let effects = analyze_module_call_effects(&check_mod, false);
    assert_eq!(effects[&fib_name], (false, false));
    assert!(effects[&gas_helper_name].1);
    assert!(effects[&gas_caller_name].1);

    // First calls with NO gas budget succeed cleanly via Cranelift JIT.
    assert_eq!(
        vm.call(&fib_name, vec![Value::Int(10)])
            .expect("fib without gas budget"),
        Value::Int(55)
    );
    assert_eq!(
        vm.call(&gas_caller_name, vec![Value::Int(3)])
            .expect("gas_caller without gas budget"),
        Value::Int(14)
    );

    // Now enable a tight gas budget on the SAME VM WITHOUT recompiling!
    // (a.1) fib with budget = 0 fails OutOfGas via VM-entry base charge.
    vm.set_gas_budget(0);
    let entries_before = vm.cranelift_entry_count;
    let fib_zero_err = vm
        .call(&fib_name, vec![Value::Int(6)])
        .expect_err("fib with budget=0 must fail OutOfGas");
    assert_eq!(vm.cranelift_entry_count, entries_before + 1);
    assert_eq!(
        vm.cranelift_compilation_count, compiled_count_initial,
        "Enabling gas budget must not have required recompilation"
    );
    assert!(
        fib_zero_err.to_string().contains("Out of gas"),
        "Expected Out of gas error for fib(budget=0), got: {fib_zero_err}"
    );

    // (a.2) fib with budget = 10 succeeds and consumes exactly 1 VM-entry base gas.
    vm.reset_gas_counter();
    vm.set_gas_budget(10);
    let entries_before = vm.cranelift_entry_count;
    let fib_res = vm
        .call(&fib_name, vec![Value::Int(10)])
        .expect("fib with budget=10 must succeed");
    assert_eq!(fib_res, Value::Int(55));
    assert_eq!(vm.cranelift_entry_count, entries_before + 1);
    assert_eq!(vm.gas_used(), 1);
    assert_eq!(vm.gas_remaining(), Some(9));
    assert_eq!(vm.cranelift_compilation_count, compiled_count_initial);

    // (b.1) gas_caller calling ResourceCheck(Gas, 10) callee with budget = 5 MUST propagate
    // OutOfGas accurately AND abort immediately after the first failing call (used == 1 + 10 = 11,
    // never reaching the second gas_helper call!).
    vm.reset_gas_counter();
    vm.set_gas_budget(5);
    let entries_before = vm.cranelift_entry_count;
    let gas_caller_err = vm
        .call(&gas_caller_name, vec![Value::Int(3)])
        .expect_err("gas_caller with budget=5 must propagate OutOfGas from gas_helper");
    assert_eq!(vm.cranelift_entry_count, entries_before + 1);
    assert_eq!(vm.cranelift_compilation_count, compiled_count_initial);
    assert!(
        gas_caller_err.to_string().contains("Out of gas"),
        "Expected Out of gas error for gas_caller(budget=5), got: {gas_caller_err}"
    );
    assert_eq!(
        vm.gas_used(),
        11,
        "gas_caller must abort immediately on first ResourceCheck(Gas, 10) exhaustion (1 base + 10 helper)"
    );

    // (b.2) With sufficient budget (100), gas_caller succeeds and accounts for exact gas (1 + 4*10 + 4*10 = 81).
    vm.reset_gas_counter();
    vm.set_gas_budget(100);
    let entries_before = vm.cranelift_entry_count;
    let gas_caller_ok = vm
        .call(&gas_caller_name, vec![Value::Int(3)])
        .expect("gas_caller with budget=100 must succeed");
    assert_eq!(gas_caller_ok, Value::Int(14));
    assert_eq!(vm.cranelift_entry_count, entries_before + 1);
    assert_eq!(vm.gas_used(), 81);
    assert_eq!(vm.cranelift_compilation_count, compiled_count_initial);
}

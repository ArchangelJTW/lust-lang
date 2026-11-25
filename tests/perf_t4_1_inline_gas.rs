#![cfg(feature = "std")]

//! Phase 1 RED -> Phase 2 GREEN & Adversarial Verification Suite for `PERF-T4-1`:
//! Inline `Statement::ResourceCheck(ResourceKind::Gas)` & SIMD `vl.gas_check`
//! with compiler-enforced `!Send + !Sync` `MirContext`, `GasFastState`,
//! explicit unbounded replenishment, and RAII `GasScopeGuard`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use lust::ast::Span;
use lust::bytecode::value::{native_fn, NativeCallResult};
use lust::bytecode::Value;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{
    analyze_module_call_effects, lust_mir_alloc_array_typed, lust_mir_charge_gas, lust_mir_get_gas,
    lust_mir_get_status, lust_mir_set_gas, lust_mir_set_status,
    reset_unbounded_gas_replenish_count, safepoint_reset, set_unbounded_gas_chunk_override,
    unbounded_gas_replenish_count, CraneliftJitCompiler, MIR_ARRAY_ELEM_TAG_INT,
    MIR_STATUS_ERR_OUT_OF_GAS, MIR_STATUS_OK,
};
use lust::mir::ir::{
    BasicBlock, BlockId, FunctionSignature, LocalDecl, LocalId, MirBinaryOp, MirConstant,
    MirFunction, MirModule, MirType, Operand, Place, ResourceKind, Rvalue, Statement, Terminator,
};
use lust::mir::lowering::AstLowering;
use lust::mir::opt::simd::recognize_vectorized_loop_blocks;
use lust::mir::opt::{SafepointConfig, Safepoints, SimdOptimizer};
use lust::mir::runtime_helpers::{enter_mir_context, MirContext};
use lust::mir::verifier::MirVerifier;
use lust::parser::Parser;
use lust::vm::osr::GLOBAL_OSR_COUNTERS;
use lust::vm::{current_vm_ptr, CurrentVmGuard, VmBackend, VM};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

// Compile-time `!Send` and `!Sync` negative trait assertion on BOTH `MirContext`
// and `Arc<MirContext>` (Hazard 7 item 1: zero-dependency `AmbiguousIfImpl` check).
// If `MirContext` or `Arc<MirContext>` were `Send` or `Sync`, this fails to compile!
const _: fn() = || {
    trait AmbiguousIfImpl<A> {
        fn check() {}
    }
    impl<T: ?Sized> AmbiguousIfImpl<()> for T {}
    impl<T: ?Sized + Send> AmbiguousIfImpl<u8> for T {}
    impl<T: ?Sized + Sync> AmbiguousIfImpl<u16> for T {}

    let _ = <MirContext as AmbiguousIfImpl<_>>::check;
    let _ = <Arc<MirContext> as AmbiguousIfImpl<_>>::check;
};

fn parse_and_lower_module(source: &str, module_name: &str) -> MirModule {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("Tokenization must succeed");
    let items = Parser::new(tokens).parse().expect("Parsing must succeed");

    let mut module = MirModule::new(module_name.into());
    for item in items {
        if let lust::ast::ItemKind::Function(func_def) = item.kind {
            let mir_fn = AstLowering::lower_function(&func_def).expect("MIR lowering must succeed");
            module.add_function(mir_fn);
        }
    }
    module
}

fn lower_single_function(source: &str) -> MirFunction {
    let module = parse_and_lower_module(source, "single_fn_mod");
    module
        .functions
        .into_iter()
        .next()
        .expect("Expected single function")
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

    fn all_call_sites(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for b in &self.block_order {
            if let Some(lines) = self.blocks.get(b) {
                for l in lines {
                    if let Some(sym) = self.extract_call_sym(l) {
                        out.push((b.clone(), sym));
                    }
                }
            }
        }
        out
    }

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
                    if l.starts_with("jump ") || l.starts_with("brif ") || l.starts_with("br_table ") {
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

fn push_local(f: &mut MirFunction, name: &str, ty: MirType, is_mut: bool, is_param: bool) -> LocalId {
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

fn build_empty_fn(name: &str, param_types: Vec<MirType>, return_type: MirType) -> (MirFunction, Vec<LocalId>) {
    let mut f = MirFunction::new(
        FunctionSignature {
            name: name.into(),
            param_types: param_types.clone(),
            return_type,
            effects: None,
            is_public: true,
        },
        Span::dummy(),
    );
    // `MirFunction::new` pre-populates `f.params` / `f.locals` for signature params if any;
    // normalize cleanly so `f.params` matches `param_types` 1-for-1.
    f.locals.clear();
    f.params.clear();
    let mut param_ids = Vec::with_capacity(param_types.len());
    for (idx, ty) in param_types.into_iter().enumerate() {
        let id = push_local(&mut f, &format!("p{idx}"), ty, false, true);
        param_ids.push(id);
    }
    (f, param_ids)
}

/// Builds a loop function with explicit `ResourceCheck(Gas, gas_per_iter)` at loop header.
fn build_gas_loop_fn(name: &str, gas_per_iter: u64) -> MirFunction {
    let (mut f, params) = build_empty_fn(name, vec![MirType::Int], MirType::Int);
    let n = params[0];
    let i = push_local(&mut f, "i", MirType::Int, true, false);
    let cond = push_local(&mut f, "cond", MirType::Bool, true, false);

    let bb0 = f.entry;
    let bb_hdr = push_block(&mut f);
    let bb_body = push_block(&mut f);
    let bb_exit = push_block(&mut f);

    f.blocks[bb0.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[bb0.0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    f.blocks[bb_hdr.0].statements.push(Statement::ResourceCheck(
        ResourceKind::Gas,
        gas_per_iter,
        Span::dummy(),
    ));
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

    f.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_body.0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    f.blocks[bb_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(i))),
        Span::dummy(),
    ));
    f
}

/// Builds a loop function that invokes an external host callback `cb(i)` on each iteration
/// and checks gas `ResourceCheck(Gas, 1)` at the loop header.
fn build_gas_loop_with_callback(name: &str, cb_name: &str) -> MirFunction {
    let (mut f, params) = build_empty_fn(name, vec![MirType::Int], MirType::Int);
    let n = params[0];
    let i = push_local(&mut f, "i", MirType::Int, true, false);
    let cond = push_local(&mut f, "cond", MirType::Bool, true, false);
    let cb_ret = push_local(&mut f, "cb_ret", MirType::Int, true, false);

    let bb0 = f.entry;
    let bb_hdr = push_block(&mut f);
    let bb_body = push_block(&mut f);
    let bb_after_cb = push_block(&mut f);
    let bb_exit = push_block(&mut f);

    f.blocks[bb0.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[bb0.0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    f.blocks[bb_hdr.0].statements.push(Statement::ResourceCheck(
        ResourceKind::Gas,
        1,
        Span::dummy(),
    ));
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

    f.blocks[bb_body.0].terminator = Some(Terminator::Call {
        callee: cb_name.into(),
        args: vec![Operand::Copy(Place::Local(i))],
        destination: Place::Local(cb_ret),
        success: bb_after_cb,
        error: None,
        span: Span::dummy(),
    });

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

    f.blocks[bb_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(i))),
        Span::dummy(),
    ));
    f
}

/// Builds a single-block function that charges `amt` gas and returns `x + 1`.
fn build_single_gas_charge_fn(name: &str, amt: u64) -> MirFunction {
    let (mut f, params) = build_empty_fn(name, vec![MirType::Int], MirType::Int);
    let x = params[0];
    let out = push_local(&mut f, "out", MirType::Int, true, false);

    let bb0 = f.entry;
    f.blocks[bb0.0]
        .statements
        .push(Statement::ResourceCheck(ResourceKind::Gas, amt, Span::dummy()));
    f.blocks[bb0.0].statements.push(Statement::Assign(
        Place::Local(out),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(x)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb0.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(out))),
        Span::dummy(),
    ));
    f
}

/// Builds a canonical vectorizable binary array-add loop `out[i] = a[i] + b[i]` for `0..n`.
fn build_simd_add_loop(name: &str) -> MirFunction {
    let arr_ty = MirType::Array(Box::new(MirType::Int));
    let (mut f, params) = build_empty_fn(
        name,
        vec![arr_ty.clone(), arr_ty.clone(), arr_ty, MirType::Int],
        MirType::Int,
    );
    let (a, b, out, n) = (params[0], params[1], params[2], params[3]);
    let i = push_local(&mut f, "i", MirType::Int, true, false);
    let cond = push_local(&mut f, "cond", MirType::Bool, true, false);
    let va = push_local(&mut f, "va", MirType::Int, true, false);
    let vb = push_local(&mut f, "vb", MirType::Int, true, false);
    let sum = push_local(&mut f, "sum", MirType::Int, true, false);

    let bb0 = f.entry;
    let bb_hdr = push_block(&mut f);
    let bb_body = push_block(&mut f);
    let bb_exit = push_block(&mut f);

    f.blocks[bb0.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[bb0.0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

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

    f.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(va),
        Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(a)), i))),
        Span::dummy(),
    ));
    f.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(vb),
        Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(b)), i))),
        Span::dummy(),
    ));
    f.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(sum),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(va)),
            Operand::Copy(Place::Local(vb)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Index(Box::new(Place::Local(out)), i),
        Rvalue::Use(Operand::Copy(Place::Local(sum))),
        Span::dummy(),
    ));
    f.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_body.0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    f.blocks[bb_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(n))),
        Span::dummy(),
    ));
    f
}

fn alloc_mir_int_array(vals: &[i64]) -> *mut i64 {
    let ptr = lust_mir_alloc_array_typed(vals.len() as i64, MIR_ARRAY_ELEM_TAG_INT);
    assert!(!ptr.is_null(), "array allocation must succeed");
    for (idx, &v) in vals.iter().enumerate() {
        unsafe {
            *ptr.add(2 + idx) = v;
        }
    }
    ptr
}

fn read_mir_int_array(ptr: *const i64, len: usize) -> Vec<i64> {
    let mut out = Vec::with_capacity(len);
    for idx in 0..len {
        out.push(unsafe { *ptr.add(2 + idx) });
    }
    out
}

fn assert_inline_gas_ir_contract(ir: &str, label: &str) {
    let parsed = ClifIrBlocks::parse(ir);
    let all_sites = parsed.all_call_sites();

    let fast_ptr_sites: Vec<&String> = all_sites
        .iter()
        .filter(|(_, sym)| sym == "lust_mir_gas_fast_ptr")
        .map(|(b, _)| b)
        .collect();
    assert_eq!(
        fast_ptr_sites.len(),
        1,
        "[{label}] Expected `lust_mir_gas_fast_ptr` called exactly once, got {fast_ptr_sites:?}:\n{ir}"
    );
    assert_eq!(
        fast_ptr_sites[0], &parsed.entry_block,
        "[{label}] Expected `lust_mir_gas_fast_ptr` in entry block {}, got {}:\n{ir}",
        parsed.entry_block, fast_ptr_sites[0]
    );

    let slow_sites: Vec<&String> = all_sites
        .iter()
        .filter(|(_, sym)| sym == "lust_mir_gas_slow")
        .map(|(b, _)| b)
        .collect();
    assert!(
        !slow_sites.is_empty(),
        "[{label}] Expected `lust_mir_gas_slow` call in cold slow-path block:\n{ir}"
    );
    for b in &slow_sites {
        assert!(
            parsed.cold_blocks.contains(*b),
            "[{label}] Expected `lust_mir_gas_slow` in a cold block, but {b} is not cold:\n{ir}"
        );
    }

    assert!(
        !all_sites.iter().any(|(_, sym)| sym == "lust_mir_charge_gas"),
        "[{label}] Found forbidden unconditional `call lust_mir_charge_gas` in IR:\n{ir}"
    );
}

// ============================================================================
// CORE TEST: IR Contract (Scalar + SIMD + OSR), Exact Gas Parity & Unbounded
// ============================================================================

#[test]
fn test_perf_t4_1_inline_gas_ir_and_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    // (a.1) Scalar hot loop IR inspection
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
    let mut compiler = CraneliftJitCompiler::new().expect("create compiler");
    let ir = compiler
        .dump_function_ir(&loop_fn)
        .expect("dump_function_ir sum_loop");
    assert_inline_gas_ir_contract(&ir, "scalar_hot_loop");

    // (a.2) Gas-free leaf function has ZERO gas fast-ptr / slow-path calls
    let leaf_fn = lower_single_function("function add_one(x: int): int return x + 1 end");
    let mut leaf_compiler = CraneliftJitCompiler::new().expect("create leaf compiler");
    let leaf_ir = leaf_compiler
        .dump_function_ir(&leaf_fn)
        .expect("dump_function_ir add_one");
    let leaf_parsed = ClifIrBlocks::parse(&leaf_ir);
    let leaf_calls = leaf_parsed.all_call_sites();
    assert!(
        !leaf_calls.iter().any(|(_, s)| s == "lust_mir_gas_fast_ptr" || s == "lust_mir_gas_slow"),
        "Gas-free function must not emit gas fast_ptr or slow calls:\n{leaf_ir}"
    );

    // (a.3) SIMD vl.gas_check IR inspection
    let mut simd_fn = build_simd_add_loop("simd_add_metered_ir");
    assert!(SimdOptimizer::run_on_function(&mut simd_fn));
    let config = SafepointConfig::new(1, 5);
    Safepoints::run_on_function_with_config(&mut simd_fn, &config);
    let recognized = recognize_vectorized_loop_blocks(&simd_fn);
    assert_eq!(recognized.len(), 1);
    assert_eq!(recognized[0].gas_check, Some(5));
    let mut simd_compiler = CraneliftJitCompiler::new().expect("simd compiler");
    let simd_ir = simd_compiler
        .dump_function_ir(&simd_fn)
        .expect("dump_function_ir simd_add_metered_ir");
    assert_inline_gas_ir_contract(&simd_ir, "simd_vl_gas_check");

    // (a.4) OSR entry IR inspection
    let osr_loop_fn = build_gas_loop_fn("osr_ir_loop", 1);
    let mut osr_compiler = CraneliftJitCompiler::new().expect("osr compiler");
    let osr_ir = osr_compiler
        .dump_osr_entry_ir(&osr_loop_fn, BlockId(1), &[LocalId(0), LocalId(1)])
        .expect("dump_osr_entry_ir");
    assert_inline_gas_ir_contract(&osr_ir, "osr_entry");

    // (b) Exact gas parity across budgets 0, 1, SEC-01 50 -> 51, and standalone 1_000_000 - 11
    let bounded_src = r#"
        function bounded_loop(n: int): int
            local i: int = 0
            while i < n do
                i = i + 1
            end
            return i
        end
    "#;
    let bounded_mod = parse_and_lower_module(bounded_src, "bounded_gas_parity");
    let bounded_fn = bounded_mod.functions[0].clone();

    // Standalone 1_000_000 - 11 parity (`src/mir/mod.rs:700-704`)
    {
        let mut jit = CraneliftJitCompiler::new().expect("jit");
        let ptr = jit.compile_function(&bounded_fn).expect("compile bounded_loop");
        let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(ptr) };
        lust_mir_set_gas(1_000_000);
        assert_eq!(f(10), 10);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
        assert_eq!(lust_mir_get_gas(), 1_000_000 - 11);

        // Budget 0: OOG on first header check
        lust_mir_set_gas(0);
        assert_eq!(f(0), MIR_STATUS_ERR_OUT_OF_GAS);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_GAS);
        assert_eq!(lust_mir_get_gas(), -1);

        // Budget 1: 0-iter loop succeeds (1 header check), 1-iter loop fails (needs 2 checks)
        lust_mir_set_gas(1);
        assert_eq!(f(0), 0);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
        assert_eq!(lust_mir_get_gas(), 0);

        lust_mir_set_gas(1);
        assert_eq!(f(1), MIR_STATUS_ERR_OUT_OF_GAS);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_GAS);
        assert_eq!(lust_mir_get_gas(), -1);

        // SEC-01 exact boundary: 50 checks succeeds with 0 remaining; 51 checks fails at -1
        lust_mir_set_gas(50);
        assert_eq!(f(49), 49);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
        assert_eq!(lust_mir_get_gas(), 0);

        lust_mir_set_gas(50);
        assert_eq!(f(50), MIR_STATUS_ERR_OUT_OF_GAS);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_GAS);
        assert_eq!(lust_mir_get_gas(), -1);
    }

    // VM-level SEC-01 50 -> 51 parity (entry base charge 1 + loop header checks)
    {
        let mut vm = VM::new();
        vm.load_mir_module(&bounded_mod).expect("load bounded_mod");
        vm.set_backend(VmBackend::Cranelift);

        // 1 entry + 49 header checks = 50 gas -> succeeds
        vm.set_gas_budget(50);
        assert_eq!(
            vm.call("bounded_loop", vec![Value::Int(48)]).expect("50 gas succeeds"),
            Value::Int(48)
        );
        assert_eq!(vm.gas_used(), 50);
        assert_eq!(vm.gas_remaining(), Some(0));

        // 1 entry + 50 header checks = 51 gas with budget 50 -> fails Out of gas
        vm.reset_gas_counter();
        vm.set_gas_budget(50);
        let err = vm
            .call("bounded_loop", vec![Value::Int(49)])
            .expect_err("51st gas check must fail Out of gas");
        assert!(err.to_string().contains("Out of gas"), "unexpected err: {err}");
        assert_eq!(vm.gas_used(), 51);
        assert_eq!(vm.gas_remaining(), Some(0));
    }

    // (c.1) Unbounded VM execution with small test chunk seam (chunk = 64) over 500 iterations
    {
        let mut vm = VM::new();
        vm.load_mir_module(&bounded_mod).expect("load bounded_mod");
        vm.set_backend(VmBackend::Cranelift);

        reset_unbounded_gas_replenish_count();
        let _chunk_guard = set_unbounded_gas_chunk_override(Some(64));
        let res = vm
            .call("bounded_loop", vec![Value::Int(500)])
            .expect("Unbounded VM with chunk=64 must complete 500 iterations cleanly");
        assert_eq!(res, Value::Int(500));
        assert!(
            unbounded_gas_replenish_count() >= 7,
            "Expected >= 7 replenishments for 501 checks with chunk=64, got {}",
            unbounded_gas_replenish_count()
        );
        assert!(
            !vm.mir_context.gas_unbounded(),
            "GasScopeGuard must restore gas_unbounded=false after return"
        );
    }

    // (c.2) Unbounded VM run exceeding 1_000_000_000 gas units without override (real UNBOUNDED_GAS_CHUNK)
    {
        let heavy_fn = build_gas_loop_fn("heavy_unbounded_loop", 1_000_000);
        let mut heavy_mod = MirModule::new("heavy_unbounded_mod".into());
        heavy_mod.add_function(heavy_fn);
        MirVerifier::verify_module(&heavy_mod).expect("verify heavy_mod");

        let mut vm = VM::new();
        vm.load_mir_module(&heavy_mod).expect("load heavy_mod");
        vm.set_backend(VmBackend::Cranelift);

        reset_unbounded_gas_replenish_count();
        let res = vm
            .call("heavy_unbounded_loop", vec![Value::Int(1_500)])
            .expect("Unbounded VM run exceeding 1_000_000_000 gas units must succeed without OOG");
        assert_eq!(res, Value::Int(1_500));
        assert!(
            unbounded_gas_replenish_count() >= 1,
            "Crossing 1_000_000_000 default gas must invoke real UNBOUNDED_GAS_CHUNK replenishment"
        );
        assert!(!vm.mir_context.gas_unbounded());
    }
}

// ============================================================================
// ADVERSARIAL 1: Nested Native-to-Native Calls Sharing MirContext + OOG Propagation
// ============================================================================

#[test]
fn test_perf_t4_1_adv1_nested_native_calls_gas_and_oog_propagation() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let source = r#"
        function callee_loop(n: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                acc = acc + 1
                i = i + 1
            end
            return acc
        end

        function caller_nested(n: int): int
            local a: int = callee_loop(n)
            local b: int = callee_loop(n)
            return a + b
        end
    "#;
    let module = parse_and_lower_module(source, "nested_gas_mod");
    MirVerifier::verify_module(&module).expect("verify nested_gas_mod");

    // 1. Direct Cranelift ABI: both calls share MirContext gas
    let mut jit = CraneliftJitCompiler::new().expect("jit");
    let compiled = jit.compile_module(&module).expect("compile nested_gas_mod");
    let caller_fn: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(compiled["caller_nested"]) };

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);

    // Each callee_loop(5) checks gas 6 times -> two calls consume 12 gas total
    ctx.set_status(MIR_STATUS_OK);
    ctx.set_gas(12);
    assert_eq!(caller_fn(5), 10);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
    assert_eq!(ctx.gas_remaining(), 0);

    // With gas = 4, first callee_loop(5) exhausts gas at check #5 (4 -> -1);
    // caller_nested MUST abort immediately without executing the second call!
    ctx.set_status(MIR_STATUS_OK);
    ctx.set_gas(4);
    assert_eq!(caller_fn(5), MIR_STATUS_ERR_OUT_OF_GAS);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_GAS);
    assert_eq!(
        ctx.gas_remaining(),
        -1,
        "Caller must propagate callee OOG immediately without invoking second callee"
    );

    // 2. VM-level verification with finite gas budget
    let mut vm = VM::new();
    vm.load_mir_module(&module).expect("load nested_gas_mod");
    vm.set_backend(VmBackend::Cranelift);

    // 1 entry + 6 + 6 = 13 gas
    vm.set_gas_budget(13);
    assert_eq!(
        vm.call("caller_nested", vec![Value::Int(5)]).expect("13 gas ok"),
        Value::Int(10)
    );
    assert_eq!(vm.gas_used(), 13);

    // Budget 5: 1 entry + 5 checks in first callee = 6 gas -> OOG before second callee
    vm.reset_gas_counter();
    vm.set_gas_budget(5);
    let err = vm
        .call("caller_nested", vec![Value::Int(5)])
        .expect_err("budget 5 must propagate callee OOG");
    assert!(err.to_string().contains("Out of gas"), "unexpected err: {err}");
    assert_eq!(vm.gas_used(), 6);
}

// ============================================================================
// ADVERSARIAL 2: Reentrant Host Callback & Same-VM Callback No-Refund Invariant
// ============================================================================

static CB_MODE: AtomicI64 = AtomicI64::new(0);
static CB_INVOCATIONS: AtomicUsize = AtomicUsize::new(0);
static CB_INNER_GAS_LEFT: AtomicI64 = AtomicI64::new(0);

extern "C" fn reentrant_gas_host_cb(iter: i64) -> i64 {
    CB_INVOCATIONS.fetch_add(1, Ordering::SeqCst);
    match CB_MODE.load(Ordering::SeqCst) {
        // Mode 1: Push a nested MirContextGuard, charge gas inside inner context, pop on return.
        1 => {
            let ctx_inner = MirContext::with_gas(200);
            {
                let _inner_guard = ctx_inner.enter();
                lust_mir_charge_gas(50);
                CB_INNER_GAS_LEFT.store(ctx_inner.gas_remaining(), Ordering::SeqCst);
            }
            0
        }
        // Mode 2: At iter == 2, tighten current MirContext gas to 3 via `lust_mir_set_gas(3)`.
        2 => {
            if iter == 2 {
                lust_mir_set_gas(3);
            }
            0
        }
        // Mode 3: Re-enter the active VM via `current_vm_ptr()` and call Cranelift `inner_work(15)`.
        3 => {
            if iter == 4 {
                if let Some(vm_ptr) = current_vm_ptr() {
                    let vm = unsafe { &mut *vm_ptr };
                    if vm.call("inner_work", vec![Value::Int(15)]).is_err() {
                        lust_mir_set_status(MIR_STATUS_ERR_OUT_OF_GAS);
                    }
                }
            }
            0
        }
        _ => 0,
    }
}

#[test]
fn test_perf_t4_1_adv2_reentrant_host_callback_and_same_vm_no_refund() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let cb_loop_fn = build_gas_loop_with_callback("cb_gas_loop", "reentrant_gas_host_cb");
    let mut cb_mod = MirModule::new("cb_gas_mod".into());
    cb_mod.add_function(cb_loop_fn.clone());
    MirVerifier::verify_module(&cb_mod).expect("verify cb_gas_mod");

    // Verify IR reloads `lust_mir_gas_fast_ptr` after the host callback call
    let mut ir_compiler = unsafe {
        CraneliftJitCompiler::new_with_host_callbacks(&[(
            "reentrant_gas_host_cb",
            vec![MirType::Int],
            MirType::Int,
            reentrant_gas_host_cb as *const u8,
        )])
    }
    .expect("ir compiler with host cb");
    let ir = ir_compiler
        .dump_function_ir(&cb_loop_fn)
        .expect("dump cb_gas_loop IR");
    let parsed = ClifIrBlocks::parse(&ir);
    let fast_ptr_count = parsed
        .all_call_sites()
        .iter()
        .filter(|(_, sym)| sym == "lust_mir_gas_fast_ptr")
        .count();
    assert!(
        fast_ptr_count >= 2,
        "Expected entry + post-callback reload of `lust_mir_gas_fast_ptr`, got {fast_ptr_count}:\n{ir}"
    );

    let mut compiler = unsafe {
        CraneliftJitCompiler::new_with_host_callbacks(&[(
            "reentrant_gas_host_cb",
            vec![MirType::Int],
            MirType::Int,
            reentrant_gas_host_cb as *const u8,
        )])
    }
    .expect("compiler with host cb");
    let code_ptr = compiler
        .compile_function(&cb_loop_fn)
        .expect("compile cb_gas_loop");
    let native_loop: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code_ptr) };

    // Case 1: Nested MirContextGuard inside host callback isolates inner gas & preserves outer gas
    {
        let ctx_outer = MirContext::with_gas(100);
        let _guard = ctx_outer.enter();
        CB_MODE.store(1, Ordering::SeqCst);
        CB_INVOCATIONS.store(0, Ordering::SeqCst);
        CB_INNER_GAS_LEFT.store(0, Ordering::SeqCst);

        let ret = native_loop(10);
        assert_eq!(ret, 10);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
        assert_eq!(CB_INVOCATIONS.load(Ordering::SeqCst), 10);
        assert_eq!(CB_INNER_GAS_LEFT.load(Ordering::SeqCst), 150);
        // Outer loop executed 11 header checks (10 iters + exit check) -> 100 - 11 == 89
        assert_eq!(
            ctx_outer.gas_remaining(),
            89,
            "Outer MirContext gas must be completely isolated from nested MirContextGuard charges"
        );
    }

    // Case 2: Mid-loop host callback calling `lust_mir_set_gas(3)` takes effect on next header check
    {
        let ctx_outer = MirContext::with_gas(10_000);
        let _guard = ctx_outer.enter();
        CB_MODE.store(2, Ordering::SeqCst);
        CB_INVOCATIONS.store(0, Ordering::SeqCst);

        let ret = native_loop(100);
        assert_eq!(ret, MIR_STATUS_ERR_OUT_OF_GAS);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_GAS);
        // Iters 0,1,2 run (at iter 2 cb sets gas=3); then iters 3,4,5 consume 3->2->1->0,
        // and iter 6 header check consumes 0->-1 (OOG). Total callbacks = 6 (iters 0..=5).
        assert_eq!(CB_INVOCATIONS.load(Ordering::SeqCst), 6);
        assert_eq!(ctx_outer.gas_remaining(), -1);
    }

    // Case 3: Same-VM reentrant callback calling `vm.call("inner_work", ...)` under finite budget
    // MUST NOT refund inner callback gas to the outer native loop!
    {
        let inner_src = r#"
            function inner_work(n: int): int
                local i: int = 0
                while i < n do
                    i = i + 1
                end
                return i
            end
        "#;
        let inner_mod = parse_and_lower_module(inner_src, "inner_reenter_mod");
        let mut vm = VM::new();
        vm.load_mir_module(&inner_mod).expect("load inner_mod");
        vm.set_backend(VmBackend::Cranelift);
        vm.compile_cranelift().expect("compile inner_work");

        // Total VM budget = 100.
        // Outer native loop runs iters 0..=4 (5 header checks -> 100 -> 95), then at iter 4
        // `reentrant_gas_host_cb` invokes `vm.call("inner_work", [15])` on the SAME VM:
        //   inner_work consumes 1 base + 16 header checks = 17 gas -> vm.gas_remaining() == 83.
        //   GasScopeGuard syncs vm.mir_context gas to <= 83 (never refunding inner's 17 gas!).
        // Outer loop then finishes iters 5..=9 + exit check (6 more checks -> 83 - 6 = 77).
        vm.set_gas_budget(100);
        vm.mir_context.set_gas(100);
        vm.mir_context.set_status(MIR_STATUS_OK);

        CB_MODE.store(3, Ordering::SeqCst);
        CB_INVOCATIONS.store(0, Ordering::SeqCst);

        let ret = {
            let _vm_guard = CurrentVmGuard::from_vm(&mut vm);
            native_loop(10)
        };
        assert_eq!(ret, 10);
        assert_eq!(vm.mir_context.last_status(), MIR_STATUS_OK);
        assert_eq!(
            vm.gas_used(),
            17,
            "Inner Cranelift VM call must have charged 17 gas (1 entry + 16 checks)"
        );
        assert_eq!(
            vm.mir_context.gas_remaining(),
            77,
            "Inner callback gas (17) must NOT be refunded to outer native loop (100 - 17 - 6 = 77)"
        );
    }
}

// ============================================================================
// ADVERSARIAL 3: Extreme Integer Boundaries Never Wrap Positive
// ============================================================================

#[test]
fn test_perf_t4_1_adv3_extreme_integer_boundaries_never_wrap_positive() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let fn_amt1 = build_single_gas_charge_fn("charge_one", 1);
    let fn_amt0 = build_single_gas_charge_fn("charge_zero", 0);
    let fn_amt10 = build_single_gas_charge_fn("charge_ten", 10);
    let fn_amt_huge = build_single_gas_charge_fn("charge_huge", u64::MAX);

    let mut module = MirModule::new("extreme_gas_mod".into());
    module.add_function(fn_amt1);
    module.add_function(fn_amt0);
    module.add_function(fn_amt10);
    module.add_function(fn_amt_huge);
    MirVerifier::verify_module(&module).expect("verify extreme_gas_mod");

    let mut jit = CraneliftJitCompiler::new().expect("jit");
    let compiled = jit.compile_module(&module).expect("compile");
    let f1: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(compiled["charge_one"]) };
    let f0: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(compiled["charge_zero"]) };
    let f10: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(compiled["charge_ten"]) };
    let fhuge: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(compiled["charge_huge"]) };

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);

    // 1. gas_remaining = i64::MIN, amt = 1 -> MUST saturate to i64::MIN and fail OOG (never wrap positive!)
    ctx.set_status(MIR_STATUS_OK);
    ctx.set_gas(i64::MIN);
    assert_eq!(f1(41), MIR_STATUS_ERR_OUT_OF_GAS);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_GAS);
    assert_eq!(lust_mir_get_gas(), i64::MIN);

    // 2. gas_remaining = i64::MIN + 5, amt = 10 -> MUST saturate to i64::MIN and fail OOG
    ctx.set_status(MIR_STATUS_OK);
    ctx.set_gas(i64::MIN + 5);
    assert_eq!(f10(41), MIR_STATUS_ERR_OUT_OF_GAS);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_GAS);
    assert_eq!(lust_mir_get_gas(), i64::MIN);

    // 3. gas_remaining = 0, amt = 0 -> succeeds cleanly with 0 remaining
    ctx.set_status(MIR_STATUS_OK);
    ctx.set_gas(0);
    assert_eq!(f0(41), 42);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
    assert_eq!(lust_mir_get_gas(), 0);

    // 4. gas_remaining = 0, amt = 1 -> fails OOG with -1 remaining
    ctx.set_status(MIR_STATUS_OK);
    ctx.set_gas(0);
    assert_eq!(f1(41), MIR_STATUS_ERR_OUT_OF_GAS);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_GAS);
    assert_eq!(lust_mir_get_gas(), -1);

    // 5. gas_remaining = -1, amt = 0 -> fails OOG with -1 remaining
    ctx.set_status(MIR_STATUS_OK);
    ctx.set_gas(-1);
    assert_eq!(f0(41), MIR_STATUS_ERR_OUT_OF_GAS);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_GAS);
    assert_eq!(lust_mir_get_gas(), -1);

    // 6. gas_remaining = i64::MAX, amt = u64::MAX (> i64::MAX) -> saturates to i64::MIN and fails OOG
    ctx.set_status(MIR_STATUS_OK);
    ctx.set_gas(i64::MAX);
    assert_eq!(fhuge(41), MIR_STATUS_ERR_OUT_OF_GAS);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_GAS);
    assert_eq!(lust_mir_get_gas(), i64::MIN);

    // 7. Direct helper `lust_mir_charge_gas` / `ctx.charge_gas` matches exact overflow-safe behavior
    ctx.set_status(MIR_STATUS_OK);
    ctx.set_gas(i64::MIN);
    assert_eq!(lust_mir_charge_gas(1), i64::MIN);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_GAS);
    assert_eq!(ctx.gas_remaining(), i64::MIN);
}

// ============================================================================
// ADVERSARIAL 4: `can_fail_transitively` Blocks PERF-T1-3 Fast Path for Gas Callee
// ============================================================================

#[test]
fn test_perf_t4_1_adv4_can_fail_transitively_blocks_perf_t1_3_fast_path() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let src = r#"
        function pure_leaf(x: int): int
            return x + 1
        end

        function gas_leaf(x: int): int
            return x + 1
        end

        function caller_pure(x: int): int
            return pure_leaf(x)
        end

        function caller_gas(x: int): int
            return gas_leaf(x)
        end
    "#;
    let mut module = parse_and_lower_module(src, "can_fail_gas_mod");
    // Inject ResourceCheck(Gas, 1) into `gas_leaf` ONLY
    module
        .functions
        .iter_mut()
        .find(|f| f.signature.name == "gas_leaf")
        .expect("gas_leaf")
        .blocks[0]
        .statements
        .insert(0, Statement::ResourceCheck(ResourceKind::Gas, 1, Span::dummy()));
    MirVerifier::verify_module(&module).expect("verify can_fail_gas_mod");

    let effects = analyze_module_call_effects(&module, false);
    assert_eq!(effects["pure_leaf"], (false, false));
    assert_eq!(effects["caller_pure"], (false, false));
    assert!(
        effects["gas_leaf"].1,
        "Gas-bearing leaf MUST have can_fail_transitively == true"
    );
    assert!(
        effects["caller_gas"].1,
        "Caller of gas-bearing leaf MUST have can_fail_transitively == true"
    );

    let mut compiler = CraneliftJitCompiler::new().expect("compiler");
    let ir_map = compiler.dump_module_ir(&module).expect("dump_module_ir");

    let pure_ir = ir_map
        .get("caller_pure$fast")
        .or_else(|| ir_map.get("caller_pure"))
        .expect("caller_pure IR");
    let pure_calls = ClifIrBlocks::parse(pure_ir).reachable_call_sites();
    assert!(
        !pure_calls.iter().any(|(_, s)| s == "lust_mir_get_status"),
        "pure_leaf caller should elide reachable lust_mir_get_status:\n{pure_ir}"
    );

    let gas_ir = ir_map
        .get("caller_gas$fast")
        .or_else(|| ir_map.get("caller_gas"))
        .expect("caller_gas IR");
    let gas_calls = ClifIrBlocks::parse(gas_ir).reachable_call_sites();
    assert!(
        gas_calls.iter().any(|(_, s)| s == "lust_mir_get_status"),
        "caller_gas MUST retain reachable lust_mir_get_status check after calling gas_leaf$fast:\n{gas_ir}"
    );
}

// ============================================================================
// ADVERSARIAL 5: SIMD Vector + Scalar Tail Early OOG Exit Parity
// ============================================================================

#[test]
fn test_perf_t4_1_adv5_simd_vector_and_scalar_tail_early_oog_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let config = SafepointConfig::new(1, 5);

    // Reference unvectorized loop with identical gas metering
    let mut reference = build_simd_add_loop("simd_ref_loop");
    Safepoints::run_on_function_with_config(&mut reference, &config);

    // Vectorized loop (vector chunks + scalar tail) with late gas metering (`vl.gas_check == Some(5)`)
    let mut vectorized = build_simd_add_loop("simd_vec_loop");
    assert!(SimdOptimizer::run_on_function(&mut vectorized));
    Safepoints::run_on_function_with_config(&mut vectorized, &config);
    let rec = recognize_vectorized_loop_blocks(&vectorized);
    assert_eq!(rec.len(), 1);
    assert_eq!(rec[0].gas_check, Some(5));

    let mut module = MirModule::new("simd_gas_parity_mod".into());
    module.add_function(reference);
    module.add_function(vectorized);
    MirVerifier::verify_module(&module).expect("verify simd_gas_parity_mod");

    let mut jit = CraneliftJitCompiler::new().expect("jit");
    let compiled = jit.compile_module(&module).expect("compile");
    let ref_fn: extern "C" fn(i64, i64, i64, i64) -> i64 =
        unsafe { std::mem::transmute(compiled["simd_ref_loop"]) };
    let vec_fn: extern "C" fn(i64, i64, i64, i64) -> i64 =
        unsafe { std::mem::transmute(compiled["simd_vec_loop"]) };

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);

    // n = 10 exercises both 4-lane vector range (0..8) and 2-element scalar tail (8..10)
    let n = 10usize;
    let a = alloc_mir_int_array(&[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    let b = alloc_mir_int_array(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

    for budget in 0..=65i64 {
        let out_ref = alloc_mir_int_array(&vec![-777; n]);
        lust_mir_set_gas(budget);
        let ret_ref = ref_fn(a as i64, b as i64, out_ref as i64, n as i64);
        let obs_ref = (
            ret_ref,
            lust_mir_get_status(),
            lust_mir_get_gas(),
            read_mir_int_array(out_ref, n),
        );

        let out_vec = alloc_mir_int_array(&vec![-777; n]);
        lust_mir_set_gas(budget);
        let ret_vec = vec_fn(a as i64, b as i64, out_vec as i64, n as i64);
        let obs_vec = (
            ret_vec,
            lust_mir_get_status(),
            lust_mir_get_gas(),
            read_mir_int_array(out_vec, n),
        );

        assert_eq!(
            obs_vec, obs_ref,
            "SIMD vector+tail early OOG mismatch at budget={budget}: vec={obs_vec:?} != ref={obs_ref:?}"
        );
    }
}

// ============================================================================
// ADVERSARIAL 6: Production OSR Finite OOG & Unbounded Small-Chunk Replenishment
// ============================================================================

#[test]
fn test_perf_t4_1_adv6_production_osr_finite_oog_and_unbounded_replenish() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    GLOBAL_OSR_COUNTERS.reset();
    safepoint_reset();

    let source = r#"
        function osr_gas_test_loop(n: int): int
            local bias: int = host_setup(5)
            local sum: int = bias
            local i: int = 0
            while i < n do
                sum = sum + i
                i = i + 1
            end
            return sum
        end
    "#;
    let module = parse_and_lower_module(source, "perf_t4_1_osr_mod");

    let register_setup = |vm: &mut VM| {
        vm.register_native(
            "host_setup",
            Value::NativeFunction(native_fn(|args| {
                let base = args.first().and_then(|v| v.as_int()).unwrap_or(0);
                Ok(NativeCallResult::Return(Value::Int(base + 100)))
            })),
        );
    };

    // (a) Finite gas exhaustion inside production OSR loop
    {
        let mut vm = VM::new();
        register_setup(&mut vm);
        vm.load_mir_module(&module).expect("load module");
        vm.set_backend(VmBackend::Cranelift);

        let limit = 2_000u64;
        vm.set_gas_budget(limit);
        let attempts_before = GLOBAL_OSR_COUNTERS.entry_attempts.load(Ordering::SeqCst);
        let err = vm
            .call("osr_gas_test_loop", vec![Value::Int(50_000)])
            .expect_err("Finite OSR loop must terminate on gas exhaustion");
        let attempts_after = GLOBAL_OSR_COUNTERS.entry_attempts.load(Ordering::SeqCst);

        assert!(err.to_string().contains("Out of gas"), "unexpected err: {err}");
        assert!(attempts_after > attempts_before, "Must have entered OSR");
        assert_eq!(vm.gas_used(), limit + 1);
        assert_eq!(vm.gas_remaining(), Some(0));
        assert!(!vm.mir_context.gas_unbounded());
    }

    // (b) Unbounded production OSR loop with small chunk seam (chunk = 64) over 500 iterations
    {
        let mut vm = VM::new();
        register_setup(&mut vm);
        vm.load_mir_module(&module).expect("load module");
        vm.set_backend(VmBackend::Cranelift);

        let n = 500i64;
        let expected = 105i64 + (n * (n - 1)) / 2;

        reset_unbounded_gas_replenish_count();
        let _chunk_guard = set_unbounded_gas_chunk_override(Some(64));
        let successes_before = GLOBAL_OSR_COUNTERS.entry_successes.load(Ordering::SeqCst);
        let res = vm
            .call("osr_gas_test_loop", vec![Value::Int(n)])
            .expect("Unbounded OSR loop with chunk=64 must complete 500 iterations cleanly");
        let successes_after = GLOBAL_OSR_COUNTERS.entry_successes.load(Ordering::SeqCst);

        assert_eq!(res, Value::Int(expected));
        assert!(
            successes_after > successes_before,
            "Production OSR must execute natively (entry_successes delta = {})",
            successes_after - successes_before
        );
        assert!(
            unbounded_gas_replenish_count() >= 5,
            "Expected >= 5 unbounded OSR gas replenishments with chunk=64, got {}",
            unbounded_gas_replenish_count()
        );
        assert!(
            !vm.mir_context.gas_unbounded(),
            "GasScopeGuard must restore gas_unbounded=false after OSR return"
        );
    }
}

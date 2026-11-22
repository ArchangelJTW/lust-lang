#![cfg(feature = "std")]

//! Phase 1 RED -> Phase 2 GREEN & Adversarial Verification Suite for `PERF-T4-2`:
//! Pure Inner-Loop Counter Register Promotion (`gas_remaining` & `safepoints_checked`)
//! and Non-Speculative Guarded-Preheader Array `(data_ptr, len)` & Struct Base Hoisting.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;

use lust::ast::Span;
use lust::bytecode::Value;
use lust::config::LustConfig;
use lust::effect_abi::InterruptReason;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{
    lust_mir_alloc_array_typed, lust_mir_alloc_cflat_struct, lust_mir_array_push,
    lust_mir_get_gas, lust_mir_get_status, lust_mir_set_gas, lust_mir_set_status,
    promoted_loop_cfg_verifications, reset_promoted_loop_cfg_verifications,
    reset_unbounded_gas_replenish_count, safepoint_enter_mask, safepoint_exit_mask,
    safepoint_request_interrupt, safepoint_reset, safepoint_stats,
    set_unbounded_gas_chunk_override, unbounded_gas_replenish_count,
    CraneliftJitCompiler, GasFastState, SafepointFastState, MIR_ARRAY_ELEM_TAG_INT,
    MIR_STATUS_ERR_DIV_BY_ZERO, MIR_STATUS_ERR_INTERRUPTED,
    MIR_STATUS_ERR_OUT_OF_BOUNDS, MIR_STATUS_ERR_OUT_OF_GAS, MIR_STATUS_OK,
};
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::{
    safepoint_request_deopt_on_poll, take_deopt_payload, MIR_STATUS_DEOPT,
};
use lust::mir::ir::{
    BasicBlock, BlockId, FunctionSignature, LocalDecl, LocalId, MirBinaryOp, MirConstant,
    MirFunction, MirModule, MirType, Operand, Place, ResourceKind, Rvalue, Statement, Terminator,
};
use lust::mir::lowering::AstLowering;
use lust::mir::runtime_helpers::{enter_mir_context, MirContext};
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

fn lower_single_function_with_layouts(
    source: &str,
) -> (MirFunction, Vec<lust::mir::ir::MirStructLayout>) {
    let module = parse_and_lower_module(source, "single_fn_mod");
    let layouts = module.struct_layouts.clone();
    let func = module
        .functions
        .into_iter()
        .next()
        .expect("Expected single function");
    (func, layouts)
}

fn lower_single_function(source: &str) -> MirFunction {
    lower_single_function_with_layouts(source).0
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

fn build_empty_fn(
    name: &str,
    param_types: Vec<MirType>,
    return_type: MirType,
) -> (MirFunction, Vec<LocalId>) {
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
    f.locals.clear();
    f.params.clear();
    let mut param_ids = Vec::with_capacity(param_types.len());
    for (idx, ty) in param_types.into_iter().enumerate() {
        let id = push_local(&mut f, &format!("p{idx}"), ty, false, true);
        param_ids.push(id);
    }
    (f, param_ids)
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

struct ClifIrBlocks {
    entry_block: String,
    blocks: HashMap<String, Vec<String>>,
    block_params: HashMap<String, Vec<String>>,
    cold_blocks: HashSet<String>,
    sym_map: HashMap<String, String>,
    block_order: Vec<String>,
    succs: HashMap<String, Vec<String>>,
    /// Directed value-copy edges: `val -> aliases` propagated across block params and copies.
    val_aliases: HashMap<String, Vec<String>>,
}

impl ClifIrBlocks {
    fn parse(ir: &str) -> Self {
        let mut sym_map: HashMap<String, String> = HashMap::new();
        let mut blocks: HashMap<String, Vec<String>> = HashMap::new();
        let mut block_params: HashMap<String, Vec<String>> = HashMap::new();
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
                let mut params = Vec::new();
                if let (Some(lp), Some(rp)) = (line.find('('), line.find(')')) {
                    if lp < rp {
                        for p in line[lp + 1..rp].split(',') {
                            if let Some(v) = p.split(':').next() {
                                let vt = v.trim();
                                if !vt.is_empty() {
                                    params.push(vt.to_string());
                                }
                            }
                        }
                    }
                }
                block_params.insert(bname.clone(), params);
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
        let mut val_aliases: HashMap<String, Vec<String>> = HashMap::new();
        for b in &block_order {
            let mut s = Vec::new();
            if let Some(lines) = blocks.get(b) {
                for l in lines {
                    // Cranelift value alias syntax: `vA -> vB` (vA is an alias for vB)
                    if let Some((lhs, rhs)) = l.split_once(" -> ") {
                        let a = lhs.trim();
                        let b_val = rhs.split_whitespace().next().unwrap_or("").trim();
                        if a.starts_with('v') && b_val.starts_with('v') {
                            val_aliases
                                .entry(b_val.to_string())
                                .or_default()
                                .push(a.to_string());
                            val_aliases
                                .entry(a.to_string())
                                .or_default()
                                .push(b_val.to_string());
                        }
                    }
                    if l.starts_with("jump ") || l.starts_with("brif ") || l.starts_with("br_table ") {
                        // Parse `blockN(vA, vB, ...)` invocations to link arg -> formal param.
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
                                if end < bytes.len() && bytes[end] == b'(' {
                                    if let Some(close_rel) = l[end + 1..].find(')') {
                                        let args_str = &l[end + 1..end + 1 + close_rel];
                                        if let Some(formals) = block_params.get(tgt) {
                                            for (arg, formal) in args_str.split(',').zip(formals.iter()) {
                                                let a = arg.trim();
                                                if a.starts_with('v') {
                                                    val_aliases
                                                        .entry(a.to_string())
                                                        .or_default()
                                                        .push(formal.clone());
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            pos = end;
                        }
                    }
                }
            }
            succs.insert(b.clone(), s);
        }

        let entry_block = block_order.first().cloned().unwrap_or_default();
        Self {
            entry_block,
            blocks,
            block_params,
            cold_blocks,
            sym_map,
            block_order,
            succs,
            val_aliases,
        }
    }

    /// Returns all SSA value names that hold the same pointer as `root_var` (propagated via block params).
    fn alias_set(&self, root_var: &str) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(root_var.to_string());
        queue.push_back(root_var.to_string());
        while let Some(v) = queue.pop_front() {
            if let Some(nexts) = self.val_aliases.get(&v) {
                for n in nexts {
                    if visited.insert(n.clone()) {
                        queue.push_back(n.clone());
                    }
                }
            }
        }
        visited
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

    /// Returns the SSA value name defined by `vN = call <sym>()` in entry_block, if any.
    fn find_call_result_var(&self, target_sym: &str) -> Option<String> {
        let lines = self.blocks.get(&self.entry_block)?;
        for l in lines {
            if self.extract_call_sym(l).as_deref() == Some(target_sym) {
                if let Some((lhs, _)) = l.split_once(" = ") {
                    return Some(lhs.trim().to_string());
                }
            }
        }
        None
    }

    /// Returns all non-cold blocks participating in any directed cycle in the non-cold subgraph.
    fn hot_cycle_blocks(&self) -> HashSet<String> {
        let hot_nodes: HashSet<&String> = self
            .block_order
            .iter()
            .filter(|b| !self.cold_blocks.contains(*b))
            .collect();

        let mut in_cycle = HashSet::new();
        for &start in &hot_nodes {
            // BFS from start's hot successors to see if `start` is reachable again.
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

    fn is_store_line(line: &str) -> bool {
        line.starts_with("store ") || line.starts_with("store.")
    }

    fn line_accesses_any_ptr(line: &str, ptr_vars: &HashSet<String>, offset: i32) -> bool {
        ptr_vars
            .iter()
            .any(|p| Self::line_accesses_ptr(line, p, offset))
    }

    fn line_accesses_ptr(line: &str, ptr_var: &str, offset: i32) -> bool {
        let has_ptr = line
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .any(|tok| tok == ptr_var);
        if !has_ptr {
            return false;
        }
        if offset == 0 {
            !line.contains(&format!("{ptr_var}+"))
        } else {
            line.contains(&format!("{ptr_var}+{offset}"))
        }
    }
}

// ============================================================================
// 1. CLIF IR INSPECTION TESTS (Pillar 1 + Pillar 2)
// ============================================================================

#[test]
fn test_perf_t4_2_ir_integer_loop_zero_hot_counter_stores() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let off_gas = core::mem::offset_of!(GasFastState, gas_remaining) as i32;
    let off_sp = core::mem::offset_of!(SafepointFastState, safepoints_checked) as i32;

    let src = r#"
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
    let mir_fn = lower_single_function(src);
    let mut compiler = CraneliftJitCompiler::new().expect("create compiler");
    let ir = compiler
        .dump_function_ir(&mir_fn)
        .expect("dump_function_ir sum_loop");
    let parsed = ClifIrBlocks::parse(&ir);

    let gas_ptr = parsed
        .find_call_result_var("lust_mir_gas_fast_ptr")
        .expect("gas_ptr must be defined at entry");
    let sp_ptr = parsed
        .find_call_result_var("lust_mir_safepoint_state_ptr")
        .expect("sp_ptr must be defined at entry");

    let gas_ptrs = parsed.alias_set(&gas_ptr);
    let sp_ptrs = parsed.alias_set(&sp_ptr);

    let hot_blocks = parsed.hot_cycle_blocks();
    assert!(
        !hot_blocks.is_empty(),
        "Expected non-empty hot cycle blocks in sum_loop IR:\n{ir}"
    );

    let mut hot_gas_stores = Vec::new();
    let mut hot_sp_stores = Vec::new();
    let mut hot_gas_loads = Vec::new();
    let mut hot_sp_loads = Vec::new();
    let mut outside_gas_loads = 0usize;
    let mut outside_sp_loads = 0usize;
    let mut outside_gas_stores = 0usize;
    let mut outside_sp_stores = 0usize;

    for b in &parsed.block_order {
        let is_hot = hot_blocks.contains(b);
        if let Some(lines) = parsed.blocks.get(b) {
            for l in lines {
                if ClifIrBlocks::is_store_line(l) {
                    if ClifIrBlocks::line_accesses_any_ptr(l, &gas_ptrs, off_gas) {
                        if is_hot {
                            hot_gas_stores.push((b.clone(), l.clone()));
                        } else {
                            outside_gas_stores += 1;
                        }
                    }
                    if ClifIrBlocks::line_accesses_any_ptr(l, &sp_ptrs, off_sp) {
                        if is_hot {
                            hot_sp_stores.push((b.clone(), l.clone()));
                        } else {
                            outside_sp_stores += 1;
                        }
                    }
                } else if l.contains(" = load.") {
                    if ClifIrBlocks::line_accesses_any_ptr(l, &gas_ptrs, off_gas) {
                        if is_hot {
                            hot_gas_loads.push((b.clone(), l.clone()));
                        } else {
                            outside_gas_loads += 1;
                        }
                    }
                    if ClifIrBlocks::line_accesses_any_ptr(l, &sp_ptrs, off_sp) {
                        if is_hot {
                            hot_sp_loads.push((b.clone(), l.clone()));
                        } else {
                            outside_sp_loads += 1;
                        }
                    }
                }
            }
        }
    }

    assert!(
        hot_gas_stores.is_empty(),
        "Hot loop must have ZERO gas_remaining stores, found {hot_gas_stores:?}\nIR:\n{ir}"
    );
    assert!(
        hot_sp_stores.is_empty(),
        "Hot loop must have ZERO safepoints_checked stores, found {hot_sp_stores:?}\nIR:\n{ir}"
    );
    assert!(
        hot_gas_loads.is_empty(),
        "Hot loop must have ZERO gas_remaining loads, found {hot_gas_loads:?}\nIR:\n{ir}"
    );
    assert!(
        hot_sp_loads.is_empty(),
        "Hot loop must have ZERO safepoints_checked loads, found {hot_sp_loads:?}\nIR:\n{ir}"
    );
    assert!(
        outside_gas_loads >= 1 && outside_sp_loads >= 1,
        "Preheader/reload must load promoted counters (gas={outside_gas_loads}, sp={outside_sp_loads})\nIR:\n{ir}"
    );
    assert!(
        outside_gas_stores >= 1 && outside_sp_stores >= 1,
        "Exit/slow flush blocks must store promoted counters (gas={outside_gas_stores}, sp={outside_sp_stores})\nIR:\n{ir}"
    );
}

#[test]
fn test_perf_t4_2_ir_array_loop_hoists_header_and_cap_redirect() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let off_gas = core::mem::offset_of!(GasFastState, gas_remaining) as i32;
    let off_sp = core::mem::offset_of!(SafepointFastState, safepoints_checked) as i32;

    let src = r#"
        function sum_array(arr: Array<int>, n: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                acc = acc + arr[i]
                i = i + 1
            end
            return acc
        end
    "#;
    let mir_fn = lower_single_function(src);
    let mut compiler = CraneliftJitCompiler::new().expect("create compiler");
    let ir = compiler
        .dump_function_ir(&mir_fn)
        .expect("dump_function_ir sum_array");
    let parsed = ClifIrBlocks::parse(&ir);

    let gas_ptr = parsed
        .find_call_result_var("lust_mir_gas_fast_ptr")
        .expect("gas_ptr must be defined");
    let sp_ptr = parsed
        .find_call_result_var("lust_mir_safepoint_state_ptr")
        .expect("sp_ptr must be defined");

    let gas_ptrs = parsed.alias_set(&gas_ptr);
    let sp_ptrs = parsed.alias_set(&sp_ptr);

    // Extract parameter 0 (`arr`) SSA name from entry block header (`blockN(v0: i64, ...):`)
    let arr_param_var = parsed
        .block_params
        .get(&parsed.entry_block)
        .and_then(|ps| ps.first())
        .cloned()
        .expect("Expected entry block param v0 for arr");
    let arr_ptrs = parsed.alias_set(&arr_param_var);

    let hot_blocks = parsed.hot_cycle_blocks();
    assert!(
        !hot_blocks.is_empty(),
        "Expected non-empty hot cycle blocks in sum_array IR:\n{ir}"
    );

    let mut hot_header_reloads = Vec::new();
    let mut hot_counter_stores = Vec::new();
    let mut preheader_cap_loads = 0usize;

    for b in &parsed.block_order {
        let is_hot = hot_blocks.contains(b);
        if let Some(lines) = parsed.blocks.get(b) {
            for l in lines {
                if is_hot && ClifIrBlocks::is_store_line(l) {
                    if ClifIrBlocks::line_accesses_any_ptr(l, &gas_ptrs, off_gas)
                        || ClifIrBlocks::line_accesses_any_ptr(l, &sp_ptrs, off_sp)
                    {
                        hot_counter_stores.push((b.clone(), l.clone()));
                    }
                }
                if l.contains(" = load.") {
                    // Check for MirArrayHeader.len (+0) or MirArrayHeader.cap (+8) loads on arr_ptrs
                    let loads_arr_len = ClifIrBlocks::line_accesses_any_ptr(l, &arr_ptrs, 0);
                    let loads_arr_cap = ClifIrBlocks::line_accesses_any_ptr(l, &arr_ptrs, 8);
                    if is_hot && (loads_arr_len || loads_arr_cap) {
                        hot_header_reloads.push((b.clone(), l.clone()));
                    }
                    if !is_hot && loads_arr_cap {
                        preheader_cap_loads += 1;
                    }
                }
            }
        }
    }

    assert!(
        hot_counter_stores.is_empty(),
        "Hot array loop must have ZERO counter stores, found {hot_counter_stores:?}\nIR:\n{ir}"
    );
    assert!(
        hot_header_reloads.is_empty(),
        "Hot array loop must have ZERO MirArrayHeader len/cap reloads on {arr_param_var}, found {hot_header_reloads:?}\nIR:\n{ir}"
    );
    assert!(
        preheader_cap_loads >= 1,
        "Expected cap load hoisted into guarded preheader outside hot cycle\nIR:\n{ir}"
    );
}

#[test]
fn test_perf_t4_2_ir_struct_loop_hoists_base_null_check() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let src = r#"
        struct Counter
            val: int
            step: int
        end

        function sum_struct(c: Counter, n: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                acc = acc + c.val + c.step
                i = i + 1
            end
            return acc
        end
    "#;
    let (mir_fn, layouts) = lower_single_function_with_layouts(src);
    let mut compiler = CraneliftJitCompiler::new().expect("create compiler");
    compiler.register_struct_layouts(&layouts);
    let ir = compiler
        .dump_function_ir(&mir_fn)
        .expect("dump_function_ir sum_struct");
    let parsed = ClifIrBlocks::parse(&ir);

    let struct_param_var = parsed
        .block_params
        .get(&parsed.entry_block)
        .and_then(|ps| ps.first())
        .cloned()
        .expect("Expected entry block param v0 for struct c");
    let struct_ptrs = parsed.alias_set(&struct_param_var);

    let hot_blocks = parsed.hot_cycle_blocks();
    assert!(!hot_blocks.is_empty(), "Expected hot cycle blocks:\n{ir}");

    let mut hot_null_checks = Vec::new();
    let mut preheader_null_checks = 0usize;
    for b in &parsed.block_order {
        let is_hot = hot_blocks.contains(b);
        if let Some(lines) = parsed.blocks.get(b) {
            for l in lines {
                if l.contains("icmp")
                    && l.contains(" eq ")
                    && struct_ptrs.iter().any(|sp| l.contains(&format!("{sp},")))
                {
                    if is_hot {
                        hot_null_checks.push((b.clone(), l.clone()));
                    } else {
                        preheader_null_checks += 1;
                    }
                }
            }
        }
    }
    assert!(
        hot_null_checks.is_empty(),
        "Hot struct loop must hoist null check out of hot cycle, found {hot_null_checks:?}\nIR:\n{ir}"
    );
    assert!(
        preheader_null_checks >= 1,
        "Expected struct null check hoisted into guarded preheader outside hot cycle\nIR:\n{ir}"
    );
}

// ============================================================================
// 2. EXACT COUNTER PARITY TESTS (gas_remaining AND safepoints_checked)
// ============================================================================

#[test]
fn test_perf_t4_2_exact_counter_parity_normal_zero_break_return_nested() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);

    let src = r#"
        function sum_loop(n: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                acc = acc + i
                i = i + 1
            end
            return acc
        end

        function break_loop(n: int, limit: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                if i == limit then
                    break
                end
                acc = acc + i
                i = i + 1
            end
            return acc
        end

        function early_ret_loop(n: int, limit: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                if i == limit then
                    return acc
                end
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
    "#;
    let module = parse_and_lower_module(src, "parity_mod");
    MirVerifier::verify_module(&module).expect("verify parity_mod");

    let mut jit = CraneliftJitCompiler::new().expect("jit");
    let compiled = jit.compile_module(&module).expect("compile parity_mod");

    let sum_fn: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(compiled["sum_loop"]) };
    let break_fn: extern "C" fn(i64, i64) -> i64 =
        unsafe { std::mem::transmute(compiled["break_loop"]) };
    let ret_fn: extern "C" fn(i64, i64) -> i64 =
        unsafe { std::mem::transmute(compiled["early_ret_loop"]) };
    let nested_fn: extern "C" fn(i64, i64) -> i64 =
        unsafe { std::mem::transmute(compiled["nested_loop"]) };

    // 1. Normal exit (N = 100): 101 header checks
    safepoint_reset();
    lust_mir_set_status(MIR_STATUS_OK);
    lust_mir_set_gas(10_000);
    assert_eq!(sum_fn(100), 4950);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
    assert_eq!(lust_mir_get_gas(), 10_000 - 101);
    assert_eq!(safepoint_stats().safepoints_checked, 101);

    // 2. Zero-iteration exit (N = 0): 1 header check
    safepoint_reset();
    lust_mir_set_status(MIR_STATUS_OK);
    lust_mir_set_gas(10_000);
    assert_eq!(sum_fn(0), 0);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
    assert_eq!(lust_mir_get_gas(), 10_000 - 1);
    assert_eq!(safepoint_stats().safepoints_checked, 1);

    // 3. Early break at i == 37: 38 header gas checks (i=0..=37); `after_loop_bb` has
    //    target <= src on break jump so `find_loop_headers` adds 1 post-loop safepoint (39 total).
    safepoint_reset();
    lust_mir_set_status(MIR_STATUS_OK);
    lust_mir_set_gas(10_000);
    assert_eq!(break_fn(100, 37), 37 * 36 / 2);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
    assert_eq!(lust_mir_get_gas(), 10_000 - 38);
    assert_eq!(safepoint_stats().safepoints_checked, 39);

    // 4. Early return at i == 25: 26 header checks (i=0..=25)
    safepoint_reset();
    lust_mir_set_status(MIR_STATUS_OK);
    lust_mir_set_gas(10_000);
    assert_eq!(ret_fn(100, 25), 25 * 24 / 2);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
    assert_eq!(lust_mir_get_gas(), 10_000 - 26);
    assert_eq!(safepoint_stats().safepoints_checked, 26);

    // 5. Nested call-free loops (outer = 7, inner = 5):
    //    outer checks = 7 + 1 = 8; inner checks = 7 * (5 + 1) = 42; total = 50
    safepoint_reset();
    lust_mir_set_status(MIR_STATUS_OK);
    lust_mir_set_gas(10_000);
    assert_eq!(nested_fn(7, 5), 35);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
    assert_eq!(lust_mir_get_gas(), 10_000 - 50);
    assert_eq!(safepoint_stats().safepoints_checked, 50);
}

#[test]
fn test_perf_t4_2_exact_counter_parity_oob_before_and_after_gas_and_div_zero() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);

    // (a) OOB AFTER header gas check (normal Lust lowering)
    let oob_after_src = r#"
        function read_oob(arr: Array<int>, n: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                acc = acc + arr[i]
                i = i + 1
            end
            return acc
        end
    "#;
    let oob_after_fn = lower_single_function(oob_after_src);

    // (b) OOB BEFORE gas check (synthetic MIR: header has safepoint, body does arr[i] BEFORE ResourceCheck(Gas, 1))
    let arr_ty = MirType::Array(Box::new(MirType::Int));
    let (mut oob_before_fn, params) =
        build_empty_fn("oob_before_gas", vec![arr_ty, MirType::Int], MirType::Int);
    {
        let (arr, n) = (params[0], params[1]);
        let i = push_local(&mut oob_before_fn, "i", MirType::Int, true, false);
        let acc = push_local(&mut oob_before_fn, "acc", MirType::Int, true, false);
        let cond = push_local(&mut oob_before_fn, "cond", MirType::Bool, true, false);
        let v = push_local(&mut oob_before_fn, "v", MirType::Int, true, false);

        let bb0 = oob_before_fn.entry;
        let bb_hdr = push_block(&mut oob_before_fn);
        let bb_body = push_block(&mut oob_before_fn);
        let bb_exit = push_block(&mut oob_before_fn);

        oob_before_fn.blocks[bb0.0].statements.push(Statement::Assign(
            Place::Local(i),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));
        oob_before_fn.blocks[bb0.0].statements.push(Statement::Assign(
            Place::Local(acc),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));
        oob_before_fn.blocks[bb0.0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

        oob_before_fn.blocks[bb_hdr.0].statements.push(Statement::Assign(
            Place::Local(cond),
            Rvalue::BinaryOp(
                MirBinaryOp::Lt,
                Operand::Copy(Place::Local(i)),
                Operand::Copy(Place::Local(n)),
            ),
            Span::dummy(),
        ));
        oob_before_fn.blocks[bb_hdr.0].terminator = Some(Terminator::Cond {
            cond: Operand::Copy(Place::Local(cond)),
            then_block: bb_body,
            else_block: bb_exit,
            span: Span::dummy(),
        });

        // Index BEFORE ResourceCheck(Gas, 1)!
        oob_before_fn.blocks[bb_body.0].statements.push(Statement::Assign(
            Place::Local(v),
            Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(arr)), i))),
            Span::dummy(),
        ));
        oob_before_fn.blocks[bb_body.0]
            .statements
            .push(Statement::ResourceCheck(ResourceKind::Gas, 1, Span::dummy()));
        oob_before_fn.blocks[bb_body.0].statements.push(Statement::Assign(
            Place::Local(acc),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(acc)),
                Operand::Copy(Place::Local(v)),
            ),
            Span::dummy(),
        ));
        oob_before_fn.blocks[bb_body.0].statements.push(Statement::Assign(
            Place::Local(i),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(i)),
                Operand::Constant(MirConstant::Int(1)),
            ),
            Span::dummy(),
        ));
        oob_before_fn.blocks[bb_body.0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

        oob_before_fn.blocks[bb_exit.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(acc))),
            Span::dummy(),
        ));
    }

    // (c) Div-by-zero inside promoted loop
    let div_zero_src = r#"
        function div_in_loop(n: int, zero_at: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                acc = acc + (100 / (zero_at - i))
                i = i + 1
            end
            return acc
        end
    "#;
    let div_zero_fn = lower_single_function(div_zero_src);

    let mut module = MirModule::new("err_flush_mod".into());
    module.add_function(oob_after_fn);
    module.add_function(oob_before_fn);
    module.add_function(div_zero_fn);
    MirVerifier::verify_module(&module).expect("verify err_flush_mod");

    let mut jit = CraneliftJitCompiler::new().expect("jit");
    let compiled = jit.compile_module(&module).expect("compile err_flush_mod");

    let f_after: extern "C" fn(i64, i64) -> i64 =
        unsafe { std::mem::transmute(compiled["read_oob"]) };
    let f_before: extern "C" fn(i64, i64) -> i64 =
        unsafe { std::mem::transmute(compiled["oob_before_gas"]) };
    let f_div: extern "C" fn(i64, i64) -> i64 =
        unsafe { std::mem::transmute(compiled["div_in_loop"]) };

    let arr = alloc_mir_int_array(&[10, 20, 30, 40, 50]);

    // (a) OOB at i=5 after 6 header gas checks & 6 safepoints
    safepoint_reset();
    lust_mir_set_status(MIR_STATUS_OK);
    lust_mir_set_gas(1_000);
    assert_eq!(f_after(arr as i64, 10), MIR_STATUS_ERR_OUT_OF_BOUNDS);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);
    assert_eq!(lust_mir_get_gas(), 1_000 - 6);
    assert_eq!(safepoint_stats().safepoints_checked, 6);

    // (b) OOB at i=5 BEFORE body gas check -> 5 prior gas checks, 6 safepoints!
    safepoint_reset();
    lust_mir_set_status(MIR_STATUS_OK);
    lust_mir_set_gas(1_000);
    assert_eq!(f_before(arr as i64, 10), MIR_STATUS_ERR_OUT_OF_BOUNDS);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);
    assert_eq!(lust_mir_get_gas(), 1_000 - 5);
    assert_eq!(safepoint_stats().safepoints_checked, 6);

    // (c) Div-by-zero at i=5 -> 6 header gas checks & 6 safepoints
    safepoint_reset();
    lust_mir_set_status(MIR_STATUS_OK);
    lust_mir_set_gas(1_000);
    assert_eq!(f_div(10, 5), MIR_STATUS_ERR_DIV_BY_ZERO);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_DIV_BY_ZERO);
    assert_eq!(lust_mir_get_gas(), 1_000 - 6);
    assert_eq!(safepoint_stats().safepoints_checked, 6);

    ctx.clear_to_baselines(0, 0);
}

#[test]
fn test_perf_t4_2_exact_counter_parity_oog_and_unbounded_replenish() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let src = r#"
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
    let module = parse_and_lower_module(src, "oog_unbounded_mod");

    // 1. Exact OOG 50 -> 51 boundary (direct native)
    {
        let ctx = MirContext::new();
        let _guard = enter_mir_context(&ctx);
        let mut jit = CraneliftJitCompiler::new().expect("jit");
        let code = jit
            .compile_function(&module.functions[0])
            .expect("compile sum_loop");
        let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code) };

        safepoint_reset();
        lust_mir_set_status(MIR_STATUS_OK);
        lust_mir_set_gas(50);
        assert_eq!(f(49), 49 * 48 / 2);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
        assert_eq!(lust_mir_get_gas(), 0);
        assert_eq!(safepoint_stats().safepoints_checked, 50);

        safepoint_reset();
        lust_mir_set_status(MIR_STATUS_OK);
        lust_mir_set_gas(50);
        assert_eq!(f(50), MIR_STATUS_ERR_OUT_OF_GAS);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_GAS);
        assert_eq!(lust_mir_get_gas(), -1);
        assert_eq!(safepoint_stats().safepoints_checked, 51);
    }

    // 2. Unbounded replenishment with small chunk (64 over 500 iterations)
    {
        safepoint_reset();
        let mut vm = VM::new();
        vm.load_mir_module(&module).expect("load module");
        vm.set_backend(VmBackend::Cranelift);

        reset_unbounded_gas_replenish_count();
        let _chunk_guard = set_unbounded_gas_chunk_override(Some(64));
        let res = vm
            .call("sum_loop", vec![Value::Int(500)])
            .expect("unbounded replenishment must succeed");
        assert_eq!(res, Value::Int(500 * 499 / 2));
        assert!(
            unbounded_gas_replenish_count() >= 7,
            "Expected >= 7 replenishments, got {}",
            unbounded_gas_replenish_count()
        );
        assert_eq!(safepoint_stats().safepoints_checked, 501);
    }
}

#[test]
fn test_perf_t4_2_exact_counter_parity_masked_interrupt_and_unmask() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);

    let src = r#"
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
    let mir_fn = lower_single_function(src);
    let mut jit = CraneliftJitCompiler::new().expect("jit");
    let code = jit.compile_function(&mir_fn).expect("compile sum_loop");
    let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code) };

    // Masked interrupt: slow_safepoint_bb runs EVERY iteration, flushes BOTH counters,
    // returns 0 (continue), reloads BOTH counters, and finishes cleanly!
    safepoint_reset();
    safepoint_enter_mask();
    safepoint_request_interrupt(InterruptReason::Cancelled);
    lust_mir_set_status(MIR_STATUS_OK);
    lust_mir_set_gas(5_000);

    assert_eq!(f(40), 40 * 39 / 2);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
    assert_eq!(lust_mir_get_gas(), 5_000 - 41);
    let st = safepoint_stats();
    assert_eq!(st.safepoints_checked, 41);
    assert_eq!(st.safepoints_masked_deferred, 41);
    assert_eq!(st.safepoints_delivered, 0);

    // Unmask: next call delivers interrupt at poll #1 BEFORE gas check!
    safepoint_exit_mask();
    lust_mir_set_status(MIR_STATUS_OK);
    lust_mir_set_gas(5_000);
    assert_eq!(f(40), MIR_STATUS_ERR_INTERRUPTED);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_INTERRUPTED);
    assert_eq!(
        lust_mir_get_gas(),
        5_000,
        "Interrupt at loop header safepoint fires BEFORE ResourceCheck(Gas)"
    );
    let st2 = safepoint_stats();
    assert_eq!(st2.safepoints_checked, 42);
    assert_eq!(st2.safepoints_delivered, 1);

    safepoint_reset();
}

#[test]
fn test_perf_t4_2_redirected_array_and_struct_hoisting_correctness() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);

    let src = r#"
        struct Pair
            a: int
            b: int
        end

        function inc_and_sum(arr: Array<int>, n: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                arr[i] = arr[i] + 1
                acc = acc + arr[i]
                i = i + 1
            end
            return acc
        end

        function sum_pair(p: Pair, n: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                acc = acc + p.a + p.b
                i = i + 1
            end
            return acc
        end
    "#;
    let module = parse_and_lower_module(src, "hoist_correct_mod");
    let mut jit = CraneliftJitCompiler::new().expect("jit");
    let compiled = jit.compile_module(&module).expect("compile hoist_correct_mod");

    let arr_fn: extern "C" fn(i64, i64) -> i64 =
        unsafe { std::mem::transmute(compiled["inc_and_sum"]) };
    let pair_fn: extern "C" fn(i64, i64) -> i64 =
        unsafe { std::mem::transmute(compiled["sum_pair"]) };

    // Create an array with cap=2, then push past cap so cap < 0 (PERF-T2-2 redirect!)
    let arr = alloc_mir_int_array(&[10, 20]);
    lust_mir_array_push(arr, 30);
    lust_mir_array_push(arr, 40);
    lust_mir_array_push(arr, 50);
    unsafe {
        assert!(
            *arr.add(1) < 0,
            "Expected redirected array with cap < 0 after push growth"
        );
    }

    safepoint_reset();
    lust_mir_set_status(MIR_STATUS_OK);
    lust_mir_set_gas(1_000);
    // (11 + 21 + 31 + 41 + 51) = 155
    assert_eq!(arr_fn(arr as i64, 5), 155);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
    assert_eq!(lust_mir_get_gas(), 1_000 - 6);
    assert_eq!(safepoint_stats().safepoints_checked, 6);

    // Struct hoisting on valid C-flat struct + null struct trap on N>=1
    let name = b"Pair";
    let pair_ptr = lust_mir_alloc_cflat_struct(16, name.as_ptr(), name.len() as i64);
    unsafe {
        *pair_ptr = 7;
        *pair_ptr.add(1) = 13;
    }
    safepoint_reset();
    lust_mir_set_status(MIR_STATUS_OK);
    lust_mir_set_gas(1_000);
    assert_eq!(pair_fn(pair_ptr as i64, 10), 200);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
    assert_eq!(lust_mir_get_gas(), 1_000 - 11);
    assert_eq!(safepoint_stats().safepoints_checked, 11);

    // Null struct with N=5 traps OOB cleanly on iteration 1 (after 1 header gas + 1 sp)
    safepoint_reset();
    lust_mir_set_status(MIR_STATUS_OK);
    lust_mir_set_gas(1_000);
    assert_eq!(pair_fn(0, 5), MIR_STATUS_ERR_OUT_OF_BOUNDS);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);
    assert_eq!(lust_mir_get_gas(), 1_000 - 1);
    assert_eq!(safepoint_stats().safepoints_checked, 1);

    ctx.clear_to_baselines(0, 0);
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_perf_t4_2_forced_test_deopt_at_poll_k_exact_counters_and_resume() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let src = r#"
        function sum_arr(arr: Array<int>, n: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                acc = acc + arr[i]
                i = i + 1
            end
            return acc
        end
    "#;
    let module = parse_and_lower_module(src, "deopt_promoted_mod");

    // 1. Direct native deopt on poll #1 (inside peeled_header_bb!) AND poll #7 (inside steady-state header_bb!)
    {
        let ctx = MirContext::new();
        let _guard = enter_mir_context(&ctx);
        let mut jit = CraneliftJitCompiler::new().expect("jit");
        let code = jit
            .compile_function(&module.functions[0])
            .expect("compile sum_arr");
        let f: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(code) };
        let arr = alloc_mir_int_array(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        for poll_k in [1u64, 7u64] {
            safepoint_reset();
            let _ = take_deopt_payload();
            safepoint_request_deopt_on_poll(None, poll_k);
            lust_mir_set_status(MIR_STATUS_OK);
            lust_mir_set_gas(1_000);

            assert_eq!(f(arr as i64, 10), MIR_STATUS_DEOPT);
            assert_eq!(lust_mir_get_status(), MIR_STATUS_DEOPT);
            // Poll #k occurs at loop header BEFORE header's gas check, so exactly (k - 1) gas checks ran!
            assert_eq!(
                lust_mir_get_gas(),
                1_000 - (poll_k as i64 - 1),
                "Exact gas_remaining mismatch on poll {poll_k}"
            );
            assert_eq!(
                safepoint_stats().safepoints_checked,
                poll_k,
                "Exact safepoints_checked mismatch on poll {poll_k}"
            );
            let payload = take_deopt_payload().expect("DeoptPayload must be published");
            assert!(payload.target_pc > 0, "Expected loop header target_pc");
            // Verify exact live locals captured at poll #k (i = k - 1, acc = sum(1..=k-1))
            let expected_i = (poll_k - 1) as i64;
            let expected_acc = expected_i * (expected_i + 1) / 2;
            let captured_i = payload
                .live_locals
                .iter()
                .find(|(loc_idx, _)| {
                    module.functions[0]
                        .locals
                        .get(*loc_idx)
                        .and_then(|l| l.name.as_deref())
                        == Some("i")
                })
                .map(|(_, v)| *v);
            let captured_acc = payload
                .live_locals
                .iter()
                .find(|(loc_idx, _)| {
                    module.functions[0]
                        .locals
                        .get(*loc_idx)
                        .and_then(|l| l.name.as_deref())
                        == Some("acc")
                })
                .map(|(_, v)| *v);
            assert_eq!(captured_i, Some(expected_i), "Exact local `i` mismatch on poll {poll_k}");
            assert_eq!(
                captured_acc,
                Some(expected_acc),
                "Exact local `acc` mismatch on poll {poll_k}"
            );
        }
        ctx.clear_to_baselines(0, 0);
    }

    // 2. Full VM-level deopt + interpreter resume parity (poll #1 inside peeled header AND poll #6 inside steady header)
    let mut gas_by_poll = Vec::new();
    for poll_k in [1u64, 6u64] {
        safepoint_reset();
        let mut vm_deopt = VM::new();
        vm_deopt.load_mir_module(&module).expect("load deopt");
        vm_deopt.set_backend(VmBackend::Cranelift);
        vm_deopt.set_gas_budget(1_000);
        safepoint_request_deopt_on_poll(None, poll_k);
        let deopt_arr = Value::array((1..=10).map(Value::Int).collect());
        let deopt_res = vm_deopt
            .call("sum_arr", vec![deopt_arr, Value::Int(10)])
            .expect("deopt + resume must succeed");
        assert_eq!(deopt_res, Value::Int(55), "Result parity mismatch on poll {poll_k}");
        assert!(vm_deopt.cranelift_entry_count > 0);
        assert_eq!(
            safepoint_stats().safepoints_checked,
            poll_k,
            "VM native prefix safepoints_checked must equal exact poll_k={poll_k}"
        );
        gas_by_poll.push(vm_deopt.gas_used());
    }
    assert!(
        gas_by_poll[1] < gas_by_poll[0],
        "Deopting later (poll 6) runs more native iterations before interpreter resume than poll 1: {gas_by_poll:?}"
    );
    safepoint_reset();
}

// ============================================================================
// 3. ADVERSARIAL SAFETY TESTS
// ============================================================================

#[test]
fn test_perf_t4_2_adv1_zero_iteration_loop_never_speculates_forged_or_null_array_or_struct() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);

    let src = r#"
        struct Node
            x: int
        end

        function zero_iter_arr(arr: Array<int>, n: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                acc = acc + arr[i]
                i = i + 1
            end
            return acc
        end

        function zero_iter_struct(nd: Node, n: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                acc = acc + nd.x
                i = i + 1
            end
            return acc
        end
    "#;
    let module = parse_and_lower_module(src, "zero_iter_spec_mod");
    let mut jit = CraneliftJitCompiler::new().expect("jit");
    let compiled = jit
        .compile_module(&module)
        .expect("compile zero_iter_spec_mod");

    let arr_fn: extern "C" fn(i64, i64) -> i64 =
        unsafe { std::mem::transmute(compiled["zero_iter_arr"]) };
    let struct_fn: extern "C" fn(i64, i64) -> i64 =
        unsafe { std::mem::transmute(compiled["zero_iter_struct"]) };

    let empty_arr = alloc_mir_int_array(&[]);

    // Null (0), forged unmapped pointer (0xdead_beef), and empty array MUST NEVER
    // be dereferenced or validated on a 0-iteration loop!
    for bad_ptr in [0i64, 0xdead_beef_i64, empty_arr as i64] {
        safepoint_reset();
        lust_mir_set_status(MIR_STATUS_OK);
        lust_mir_set_gas(500);
        assert_eq!(arr_fn(bad_ptr, 0), 0);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
        assert_eq!(lust_mir_get_gas(), 499);
        assert_eq!(safepoint_stats().safepoints_checked, 1);
    }

    for bad_ptr in [0i64, 0xdead_beef_i64] {
        safepoint_reset();
        lust_mir_set_status(MIR_STATUS_OK);
        lust_mir_set_gas(500);
        assert_eq!(struct_fn(bad_ptr, 0), 0);
        assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
        assert_eq!(lust_mir_get_gas(), 499);
        assert_eq!(safepoint_stats().safepoints_checked, 1);
    }

    ctx.clear_to_baselines(0, 0);
}

#[test]
fn test_perf_t4_2_adv2_alias_mediated_growth_and_call_in_loop_disables_promotion() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let src = r#"
        function alias_grow_and_read(a: Array<int>, n: int): int
            local b: Array<int> = a
            local i: int = 0
            local acc: int = 0
            while i < n do
                array.push(b, i * 10)
                acc = acc + a[i]
                i = i + 1
            end
            return acc
        end
    "#;
    let module = parse_and_lower_module(src, "alias_grow_mod");

    // Verify IR keeps counter updates in memory (fail-closed because loop contains array.push call!)
    let off_sp = core::mem::offset_of!(SafepointFastState, safepoints_checked) as i32;
    let mut ir_compiler = CraneliftJitCompiler::new().expect("ir compiler");
    let ir = ir_compiler
        .dump_function_ir(&module.functions[0])
        .expect("dump_function_ir alias_grow_and_read");
    let parsed = ClifIrBlocks::parse(&ir);
    let sp_ptr = parsed
        .find_call_result_var("lust_mir_safepoint_state_ptr")
        .expect("sp_ptr");
    let sp_ptrs = parsed.alias_set(&sp_ptr);
    let hot_blocks = parsed.hot_cycle_blocks();
    // Both safepoints_checked store (+8) and gas_remaining store (trusted store right before gas_slow brif)
    // must remain inside hot_blocks when Terminator::Call disables counter promotion.
    let has_hot_sp_store = hot_blocks.iter().any(|b| {
        parsed.blocks.get(b).is_some_and(|lines| {
            lines
                .iter()
                .any(|l| ClifIrBlocks::is_store_line(l) && ClifIrBlocks::line_accesses_any_ptr(l, &sp_ptrs, off_sp))
        })
    });
    let has_hot_trusted_gas_store = hot_blocks.iter().any(|b| {
        parsed.blocks.get(b).is_some_and(|lines| {
            lines.iter().any(|l| l.contains("ssub_overflow"))
                && lines.iter().any(|l| ClifIrBlocks::is_store_line(l) && l.contains("notrap aligned"))
        })
    });
    assert!(
        has_hot_sp_store && has_hot_trusted_gas_store,
        "Loop containing Terminator::Call(array.push) MUST disable counter promotion fail-closed (sp={has_hot_sp_store}, gas={has_hot_trusted_gas_store}):\n{ir}"
    );

    // Behavioral verification: growing `b` reallocates & redirects `a`, and `a[i]` reads every pushed value!
    safepoint_reset();
    let mut vm = VM::new();
    vm.load_mir_module(&module).expect("load alias_grow_mod");
    vm.set_backend(VmBackend::Cranelift);
    vm.set_gas_budget(1_000);
    let arr = Value::array(Vec::new());
    let res = vm
        .call("alias_grow_and_read", vec![arr, Value::Int(20)])
        .expect("alias_grow_and_read must succeed");
    // sum_{i=0..19} (i * 10) = 190 * 10 = 1900
    assert_eq!(res, Value::Int(1900));
    // 1 entry base charge + 21 header gas checks = 22
    assert_eq!(vm.gas_used(), 22);
    assert_eq!(safepoint_stats().safepoints_checked, 21);
}

#[test]
fn test_perf_t4_2_adv3_irreducible_multi_entry_cfg_fallback() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);

    // Build synthetic irreducible multi-entry CFG:
    //   bb0 (entry): if start_left != 0 -> bb1 else -> bb2
    //   bb1: gas(1); n = n - 1; acc = acc + 1; if n > 0 -> bb2 else -> bb3
    //   bb2: gas(1); n = n - 1; acc = acc + 10; if n > 0 -> bb1 else -> bb3
    //   bb3: return acc
    let (mut f, params) = build_empty_fn(
        "irreducible_cycle",
        vec![MirType::Int, MirType::Int],
        MirType::Int,
    );
    let (start_left, n_in) = (params[0], params[1]);
    let n = push_local(&mut f, "n", MirType::Int, true, false);
    let acc = push_local(&mut f, "acc", MirType::Int, true, false);
    let c0 = push_local(&mut f, "c0", MirType::Bool, true, false);
    let c1 = push_local(&mut f, "c1", MirType::Bool, true, false);
    let c2 = push_local(&mut f, "c2", MirType::Bool, true, false);

    let bb0 = f.entry;
    let bb1 = push_block(&mut f);
    let bb2 = push_block(&mut f);
    let bb3 = push_block(&mut f);

    f.blocks[bb0.0].statements.push(Statement::Assign(
        Place::Local(n),
        Rvalue::Use(Operand::Copy(Place::Local(n_in))),
        Span::dummy(),
    ));
    f.blocks[bb0.0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[bb0.0].statements.push(Statement::Assign(
        Place::Local(c0),
        Rvalue::BinaryOp(
            MirBinaryOp::NotEq,
            Operand::Copy(Place::Local(start_left)),
            Operand::Constant(MirConstant::Int(0)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb0.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(c0)),
        then_block: bb1,
        else_block: bb2,
        span: Span::dummy(),
    });

    // bb1
    f.blocks[bb1.0]
        .statements
        .push(Statement::ResourceCheck(ResourceKind::Gas, 1, Span::dummy()));
    f.blocks[bb1.0].statements.push(Statement::Assign(
        Place::Local(n),
        Rvalue::BinaryOp(
            MirBinaryOp::Sub,
            Operand::Copy(Place::Local(n)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb1.0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(acc)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb1.0].statements.push(Statement::Assign(
        Place::Local(c1),
        Rvalue::BinaryOp(
            MirBinaryOp::Gt,
            Operand::Copy(Place::Local(n)),
            Operand::Constant(MirConstant::Int(0)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb1.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(c1)),
        then_block: bb2,
        else_block: bb3,
        span: Span::dummy(),
    });

    // bb2
    f.blocks[bb2.0]
        .statements
        .push(Statement::ResourceCheck(ResourceKind::Gas, 1, Span::dummy()));
    f.blocks[bb2.0].statements.push(Statement::Assign(
        Place::Local(n),
        Rvalue::BinaryOp(
            MirBinaryOp::Sub,
            Operand::Copy(Place::Local(n)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb2.0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(acc)),
            Operand::Constant(MirConstant::Int(10)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb2.0].statements.push(Statement::Assign(
        Place::Local(c2),
        Rvalue::BinaryOp(
            MirBinaryOp::Gt,
            Operand::Copy(Place::Local(n)),
            Operand::Constant(MirConstant::Int(0)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb2.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(c2)),
        then_block: bb1,
        else_block: bb3,
        span: Span::dummy(),
    });

    // bb3
    f.blocks[bb3.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(acc))),
        Span::dummy(),
    ));

    // Must NOT promote counters (irreducible multi-entry CFG -> fail-closed!)
    let off_gas = core::mem::offset_of!(GasFastState, gas_remaining) as i32;
    let mut jit = CraneliftJitCompiler::new().expect("jit");
    let ir = jit
        .dump_function_ir(&f)
        .expect("dump_function_ir irreducible_cycle");
    let parsed = ClifIrBlocks::parse(&ir);
    let gas_ptr = parsed
        .find_call_result_var("lust_mir_gas_fast_ptr")
        .expect("gas_ptr");
    let gas_ptrs = parsed.alias_set(&gas_ptr);
    let hot_blocks = parsed.hot_cycle_blocks();
    let has_hot_gas_store = hot_blocks.iter().any(|b| {
        parsed.blocks.get(b).is_some_and(|lines| {
            lines
                .iter()
                .any(|l| ClifIrBlocks::is_store_line(l) && ClifIrBlocks::line_accesses_any_ptr(l, &gas_ptrs, off_gas))
        })
    });
    assert!(
        has_hot_gas_store,
        "Irreducible multi-entry CFG MUST NOT promote counters:\n{ir}"
    );

    let mut exec_jit = CraneliftJitCompiler::new().expect("exec jit");
    let code = exec_jit
        .compile_function(&f)
        .expect("compile irreducible_cycle");
    let func: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(code) };

    // start_left=1, n=5 -> bb1(+1), bb2(+10), bb1(+1), bb2(+10), bb1(+1) = 23, gas = 5
    safepoint_reset();
    lust_mir_set_status(MIR_STATUS_OK);
    lust_mir_set_gas(100);
    assert_eq!(func(1, 5), 23);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
    assert_eq!(lust_mir_get_gas(), 95);

    // start_left=0, n=4 -> bb2(+10), bb1(+1), bb2(+10), bb1(+1) = 22, gas = 4
    safepoint_reset();
    lust_mir_set_status(MIR_STATUS_OK);
    lust_mir_set_gas(100);
    assert_eq!(func(0, 4), 22);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);
    assert_eq!(lust_mir_get_gas(), 96);
}

#[test]
fn test_perf_t4_2_adv4_compile_time_cfg_exit_completeness_verifier_runs() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_promoted_loop_cfg_verifications();

    let src = r#"
        struct Point
            x: int
            y: int
        end

        function f1(n: int): int
            local i: int = 0
            local s: int = 0
            while i < n do
                s = s + i
                i = i + 1
            end
            return s
        end

        function f2(arr: Array<int>, n: int): int
            local i: int = 0
            local s: int = 0
            while i < n do
                if i == 10 then
                    return s
                end
                s = s + arr[i]
                i = i + 1
            end
            return s
        end

        function f3(p: Point, n: int): int
            local i: int = 0
            local s: int = 0
            while i < n do
                s = s + p.x + p.y
                i = i + 1
            end
            return s
        end

        function f4(n: int, m: int): int
            local i: int = 0
            local s: int = 0
            while i < n do
                local j: int = 0
                while j < m do
                    s = s + 1
                    j = j + 1
                end
                i = i + 1
            end
            return s
        end
    "#;
    let module = parse_and_lower_module(src, "cfg_verifier_mod");
    let mut jit = CraneliftJitCompiler::new().expect("jit");
    let _ = jit.compile_module(&module).expect("compile cfg_verifier_mod");

    assert!(
        promoted_loop_cfg_verifications() >= 4,
        "Expected compile-time CFG exit-completeness verifier to run on all 4 promoted functions, got {}",
        promoted_loop_cfg_verifications()
    );
}


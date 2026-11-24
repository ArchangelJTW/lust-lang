#![cfg(feature = "std")]
#![forbid(unsafe_code)]

//! NG-W15-6 Verification Suite:
//! 1. Wave 14 `array.lust` & `sieve.lust` JIT drift recovery:
//!    - `peeled_header_bb` runs inline 1-iteration check and defers `Q=64` quantum arming to
//!      backedge header `clif_blocks[hspec.hdr]` (zero cold-block quantum refill on 0-trip loops).
//!    - Dominance-verified `find_loop_headers` / `find_loop_backedges` / `is_loop_backedge` in
//!      `cranelift_backend` eliminate false `if`-merge loop headers in `sieve.lust`.
//!    - Proven-in-bounds `UnsignedRangeCheck` (`if arr[j] is Ok(v)`, `if flags[i] is Ok(f)`,
//!      `if stack[mid] is Ok(v)` with `mid = n / 2` and `n >= 4`) folds to `iconst(1)` +
//!      unconditional `jump(then_bb)` while retaining fail-closed bounds checks on unproven accesses.
//! 2. Cranelift native codegen wiring for typed `Map<K, V>` (`lust_mir_map_set_typed`,
//!    `lust_mir_map_get_typed`, `lust_mir_map_has_typed`, `lust_mir_map_get_int_unwrap`),
//!    multi-field `Enum` (`lust_mir_alloc_enum2`, `lust_mir_enum_payload_at`),
//!    `string.sub` / `string.byte` / `string.byte_at` / `string.gsub` / `string.split` / `string.find`,
//!    and `Array<Struct>`.
//! 3. Mandatory `SEC-01` exact gas-preservation gate (`gas_limit = 0, 1, 50 -> 51, 63, 64, 65`)
//!    across 0-trip, 1-trip, and multi-trip peeled-header hoisted loops.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;

use lust::bytecode::Value;
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{
    find_loop_backedges, find_loop_headers, is_loop_backedge, promoted_loop_cfg_verifications,
    reset_promoted_loop_cfg_verifications, safepoint_reset, safepoint_stats,
    take_current_sticky_error, take_deopt_payload, CraneliftJitCompiler,
};
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::safepoint_request_deopt_on_poll;
use lust::mir::ir::{MirFunction, MirModule};
use lust::mir::lowering::AstLowering;
use lust::mir::opt::{JitInlineConfig, MirOptimizer};
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{reset_effect_handler_state, VmBackend, VM};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn reset_all() {
    safepoint_reset();
    reset_effect_handler_state();
    take_deopt_payload();
    take_current_sticky_error();
}

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

fn parse_and_lower_single_fn(source: &str, module_name: &str) -> MirFunction {
    let module = parse_and_lower_module(source, module_name);
    module
        .functions
        .into_iter()
        .next()
        .expect("expected at least one function")
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
        let mut cold_blocks = HashSet::new();
        let mut block_order = Vec::new();
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
                    if l.starts_with("jump ") || l.starts_with("brif ") || l.starts_with("br_table ")
                    {
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

    fn count_icmp_in_blocks(&self, blocks: &HashSet<String>, cc: &str) -> usize {
        let p1 = format!("icmp {cc} ");
        let p2 = format!("icmp.i64 {cc} ");
        let mut count = 0usize;
        for b in blocks {
            if let Some(lines) = self.blocks.get(b) {
                for l in lines {
                    if l.contains(&p1) || l.contains(&p2) {
                        count += 1;
                    }
                }
            }
        }
        count
    }

    fn count_non_push_ult_in_blocks(&self, blocks: &HashSet<String>) -> usize {
        let mut count = 0usize;
        for b in blocks {
            if let Some(lines) = self.blocks.get(b) {
                for (idx, l) in lines.iter().enumerate() {
                    if l.contains("icmp ult ") || l.contains("icmp.i64 ult ") {
                        // Inline `array.push` capacity check is immediately preceded by
                        // `band ..., 0x7fff_ffff_ffff_ffff` (`cap_raw & i64::MAX`).
                        let is_inline_push_cap = idx > 0
                            && lines[idx - 1].contains("band ")
                            && idx >= 2
                            && lines[idx - 2].contains("0x7fff_ffff_ffff_ffff");
                        if !is_inline_push_cap {
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }

    fn count_iconst_63(&self) -> usize {
        let mut count = 0usize;
        for b in &self.block_order {
            if let Some(lines) = self.blocks.get(b) {
                for l in lines {
                    if l.contains("iconst.i64 63") {
                        count += 1;
                    }
                }
            }
        }
        count
    }
}

fn run_mir_fn_cranelift_and_vm(
    source: &str,
    mod_name: &str,
    fn_name: &str,
    args: Vec<Value>,
) -> Value {
    let mut module = parse_and_lower_module(source, mod_name);
    MirOptimizer::optimize_module_for_jit(&mut module, &JitInlineConfig::default());

    reset_all();
    let mut vm_jit = VM::new();
    vm_jit.load_mir_module(&module).expect("load module jit");
    vm_jit.set_backend(VmBackend::Cranelift);
    let jit_res = vm_jit
        .call(fn_name, args.clone())
        .unwrap_or_else(|e| panic!("Cranelift JIT call '{fn_name}' failed: {e:?}"));
    if vm_jit.cranelift_entry_count == 0 {
        let f_opt = module.functions.iter().find(|f| f.signature.name == fn_name);
        let diag = f_opt.map(|f| {
            let refined = lust::mir::cranelift_backend::jit_engine::refine_mir_function(f);
            let mut c = CraneliftJitCompiler::new().unwrap();
            let compile_res = c.dump_function_ir(&refined);
            format!(
                "raw_locals={:?}\nrefined_locals={:?}\nblocks={:?}\ncompile_res={:?}",
                f.locals,
                refined.locals,
                refined.blocks,
                compile_res.map(|_| "OK")
            )
        });
        panic!(
            "Function '{fn_name}' must execute natively via Cranelift JIT (entries={}, compiled={:?})\n{}",
            vm_jit.cranelift_entry_count,
            vm_jit.cranelift_compiled.keys().collect::<Vec<_>>(),
            diag.unwrap_or_default()
        );
    }

    reset_all();
    let mut vm_bc = VM::new();
    vm_bc.load_mir_module(&module).expect("load module bc");
    vm_bc.set_backend(VmBackend::Bytecode);
    let bc_res = vm_bc
        .call(fn_name, args.clone())
        .unwrap_or_else(|e| panic!("VM call '{fn_name}' failed: {e:?}"));

    assert_eq!(
        jit_res, bc_res,
        "Cranelift JIT != Bytecode VM parity mismatch for '{fn_name}'({args:?})"
    );
    jit_res
}

// ============================================================================
// 1. Wave 14 `array.lust` & `sieve.lust` Drift Recovery + `BENCH-NG-03` Bounds Folding
// ============================================================================

#[test]
fn test_ng_w15_6_sieve_and_array_proven_in_bounds_elides_range_check_and_peeled_refill() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();
    reset_promoted_loop_cfg_verifications();

    // (a) `sieve.lust` pattern:
    //   1. Dominance-verified `find_loop_headers` returns ONLY the 3 real `while` headers (no false if-merge headers).
    //   2. `peeled_header_bb` does NOT emit a second `Q=64` arming site (only 1 `arm_quantum_bb` for the inner loop's backedge header).
    //   3. Proven-in-bounds `if flags[i] is Ok(f)` folds to `iconst(1)` + unconditional `jump` (0 hot `icmp ult`, 0 hot `icmp uge`).
    let sieve_src = r#"
function sieve_bench(limit: int): int
    local N: int = 10000
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
    return count + limit - limit
end
"#;
    let sieve_fn = parse_and_lower_single_fn(sieve_src, "ng_w15_6_sieve");
    let headers = find_loop_headers(&sieve_fn.blocks);
    let backedges = find_loop_backedges(&sieve_fn.blocks);
    assert_eq!(
        headers.len(),
        3,
        "Dominance-verified cranelift_backend::find_loop_headers must find ONLY the 3 real `while` headers in sieve (got {headers:?}, backedges={backedges:?})"
    );
    for &(from, to) in &backedges {
        assert!(is_loop_backedge(&sieve_fn.blocks, from, to));
        assert!(!is_loop_backedge(&sieve_fn.blocks, to, from));
    }

    let mut sieve_comp = CraneliftJitCompiler::new().expect("compiler");
    let sieve_ir = sieve_comp
        .dump_function_ir(&sieve_fn)
        .expect("dump sieve IR");
    let sieve_parsed = ClifIrView::parse(&sieve_ir);
    let sieve_hot = sieve_parsed.hot_cycle_blocks();
    assert!(!sieve_hot.is_empty());
    assert_eq!(
        sieve_parsed.count_non_push_ult_in_blocks(&sieve_hot),
        0,
        "NG-W15-6: proven-in-bounds `if flags[i] is Ok(f)` must fold to iconst(1)+jump (0 hot range-check `icmp ult`):\n{sieve_ir}"
    );
    assert_eq!(
        sieve_parsed.count_icmp_in_blocks(&sieve_hot, "uge"),
        0,
        "NG-W15-6: proven-in-bounds `flags[j] = false` must elide bounds check (0 hot `icmp uge`):\n{sieve_ir}"
    );
    // Two quantized loops in `sieve_bench` (the `array.push` init loop + the inner sieve loop).
    // Each `emit_quantized_loop_header_check` emits 3 `iconst.i64 63` constants inside `arm_quantum_bb`
    // (threshold, c63, sp+63). Deferring Q=64 from `peeled_header_bb` to backedge header gives 2*3 = 6
    // (whereas duplicating Q=64 in `peeled_header_bb` would produce 3*3 = 9).
    assert_eq!(
        sieve_parsed.count_iconst_63(),
        6,
        "NG-W15-6: peeled_header_bb must NOT emit quantized refill; expected 2 Q=64 arm sites (6 x iconst.i64 63, not 9):\n{sieve_ir}"
    );
    assert!(promoted_loop_cfg_verifications() > 0);

    let sieve_res = run_mir_fn_cranelift_and_vm(
        sieve_src,
        "ng_w15_6_sieve_run",
        "sieve_bench",
        vec![Value::Int(0)],
    );
    assert_eq!(sieve_res, Value::Int(1229));

    // (b) `array.lust` pattern:
    //   1. Proven-in-bounds `if arr[j] is Ok(v)` inside `while j < n` folds to `iconst(1)` + unconditional `jump` (0 hot `icmp ult`, 0 hot `icmp uge`).
    //   2. `peeled_header_bb` defers `Q=64` arming to backedge header (exactly 1 Q=64 arm site = 3 x `iconst.i64 63`).
    let array_src = r#"
function array_bench(n_elems: int, passes: int): int
    local arr: Array<int> = []
    local i: int = 0
    while i < n_elems do
        array.push(arr, i)
        i = i + 1
    end
    local total: int = 0
    local pass: int = 0
    while pass < passes do
        local j: int = 0
        local n: int = array.len(arr)
        while j < n do
            if arr[j] is Ok(v) then
                total = total + v
            end
            j = j + 1
        end
        pass = pass + 1
    end
    return total
end
"#;
    let array_fn = parse_and_lower_single_fn(array_src, "ng_w15_6_array");
    let mut array_comp = CraneliftJitCompiler::new().expect("compiler");
    let array_ir = array_comp
        .dump_function_ir(&array_fn)
        .expect("dump array IR");
    let array_parsed = ClifIrView::parse(&array_ir);
    let array_hot = array_parsed.hot_cycle_blocks();
    assert!(!array_hot.is_empty());
    assert_eq!(
        array_parsed.count_non_push_ult_in_blocks(&array_hot),
        0,
        "NG-W15-6: proven-in-bounds `if arr[j] is Ok(v)` in array_bench must fold to iconst(1)+jump (0 hot range-check `icmp ult`):\n{array_ir}"
    );
    assert_eq!(
        array_parsed.count_icmp_in_blocks(&array_hot, "uge"),
        0,
        "NG-W15-6: proven-in-bounds `arr[j]` load in array_bench must have 0 hot `icmp uge`:\n{array_ir}"
    );
    // Two quantized loops in `array_bench` (the `array.push` init loop + the inner `while j < n` loop):
    // 2 * 3 = 6 `iconst.i64 63` (NOT 9 with `peeled_header_bb` quantized).
    assert_eq!(
        array_parsed.count_iconst_63(),
        6,
        "NG-W15-6: peeled_header_bb in array_bench must defer Q=64 arming to backedge header (6 x iconst.i64 63, not 9):\n{array_ir}"
    );

    let array_res = run_mir_fn_cranelift_and_vm(
        array_src,
        "ng_w15_6_array_run",
        "array_bench",
        vec![Value::Int(2000), Value::Int(5)],
    );
    assert_eq!(array_res, Value::Int(5 * (2000 * 1999 / 2)));

    // (c) `BENCH-NG-03` (`array_mut_pipeline.lust`) midpoint range check folding:
    //   `n = array.len(stack); if n >= 4 then mid = n / 2; if stack[mid] is Ok(v) then stack[mid] = ...`
    let ng03_src = r#"
function array_mut_pipeline_fn(iters: int): int
    local stack: Array<int> = []
    local i: int = 0
    local checksum: int = 0
    while i < iters do
        array.push(stack, (i * 17 + 5) % 10007)
        local n: int = array.len(stack)
        if n >= 4 then
            local mid: int = n / 2
            if stack[mid] is Ok(v) then
                stack[mid] = (v + i) % 10007
                checksum = (checksum + v) % 1000000007
            end
        end
        i = i + 1
    end
    return checksum + array.len(stack)
end
"#;
    let ng03_fn = parse_and_lower_single_fn(ng03_src, "ng_w15_6_ng03");
    let mut ng03_comp = CraneliftJitCompiler::new().expect("compiler");
    let ng03_ir = ng03_comp.dump_function_ir(&ng03_fn).expect("dump ng03 IR");
    let ng03_parsed = ClifIrView::parse(&ng03_ir);
    let ng03_hot = ng03_parsed.hot_cycle_blocks();
    assert_eq!(
        ng03_parsed.count_non_push_ult_in_blocks(&ng03_hot),
        0,
        "NG-W15-6: proven-in-bounds `if stack[mid] is Ok(v)` (`n >= 4`, `mid = n / 2`) must fold (0 hot range-check `icmp ult`):\n{ng03_ir}"
    );
    let ng03_res = run_mir_fn_cranelift_and_vm(
        ng03_src,
        "ng_w15_6_ng03_run",
        "array_mut_pipeline_fn",
        vec![Value::Int(5000)],
    );
    // Exact full-scale oracle checksum for BENCH-NG-03 (`iters = 500000`) is 500808394.
    let ng03_full = run_mir_fn_cranelift_and_vm(
        ng03_src,
        "ng_w15_6_ng03_full",
        "array_mut_pipeline_fn",
        vec![Value::Int(500000)],
    );
    assert_eq!(ng03_full, Value::Int(500808394));
    assert!(matches!(ng03_res, Value::Int(_)));

    // (d) Fail-closed check: `mid = n / 2` WITHOUT `n >= 1` guard (e.g. empty array `n == 0`)
    //     MUST NOT fold and must take the `Err` branch cleanly!
    let unsafe_mid_src = r#"
function empty_mid_check(): int
    local stack: Array<int> = []
    local n: int = array.len(stack)
    local mid: int = n / 2
    if stack[mid] is Ok(v) then
        return v
    end
    return -12345
end
"#;
    let empty_mid_res = run_mir_fn_cranelift_and_vm(
        unsafe_mid_src,
        "ng_w15_6_fail_closed_mid",
        "empty_mid_check",
        vec![],
    );
    assert_eq!(empty_mid_res, Value::Int(-12345));
}

// ============================================================================
// 2. Cranelift Codegen Wiring for Typed Map, Multi-Field Enum, String, and Suite v2
// ============================================================================

#[test]
fn test_ng_w15_6_cranelift_compiles_and_runs_suite_v2_map_enum_and_string_benchmarks() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    // (a) Typed Map<int, int> (`BENCH-NG-01` `map_int_churn` exact kernel + key=1 handle collision guard!)
    let ng01_src = r#"
function map_int_churn_fn(): int
    local m: Map<int, int> = {}
    local n_keys: int = 16384
    local i: int = 0
    while i < n_keys do
        local k: int = (i * 73) % n_keys
        map.set(m, k, i * 3 + 1)
        i = i + 1
    end
    local acc: int = 0
    local step: int = 0
    while step < 200000 do
        local k: int = (step * 131 + acc) & 16383
        local found: Option<int> = map.get(m, k)
        local cur: int = found:unwrap()
        local next_v: int = (cur + step) % 1000003
        map.set(m, k, next_v)
        acc = (acc + next_v) % 1000000007
        step = step + 1
    end
    return acc + map.len(m)
end
"#;
    let ng01_res = run_mir_fn_cranelift_and_vm(
        ng01_src,
        "ng_w15_6_ng01",
        "map_int_churn_fn",
        vec![],
    );
    assert_eq!(
        ng01_res,
        Value::Int(608439931),
        "BENCH-NG-01 Cranelift checksum mismatch"
    );

    // (b) Typed Map<string, int> + Array<string> (`BENCH-NG-02` `map_str_intern_hist`)
    let ng02_src = r#"
function map_str_intern_hist_fn(): int
    local hist: Map<string, int> = {}
    local keys: Array<string> = []
    local k: int = 0
    while k < 256 do
        array.push(keys, "tok_key_" .. tostring(k))
        map.set(hist, "tok_key_" .. tostring(k), 0)
        k = k + 1
    end
    local i: int = 0
    local total: int = 0
    while i < 150000 do
        local idx: int = (i * 41 + (i / 7)) % 256
        if keys[idx] is Ok(key_str) then
            local found: Option<int> = map.get(hist, key_str)
            local c: int = found:unwrap()
            local next_c: int = c + 1
            map.set(hist, key_str, next_c)
            total = (total + next_c * (idx + 1)) % 1000000007
        end
        i = i + 1
    end
    return total
end
"#;
    let ng02_res = run_mir_fn_cranelift_and_vm(
        ng02_src,
        "ng_w15_6_ng02",
        "map_str_intern_hist_fn",
        vec![],
    );
    assert_eq!(
        ng02_res,
        Value::Int(379372375),
        "BENCH-NG-02 Cranelift checksum mismatch"
    );

    // (c) `map.has` & `map.get_unwrap_or` (including i64::MIN sentinel edge case!)
    let map_ops_src = r#"
function test_map_extras(): int
    local m: Map<int, int> = {}
    map.set(m, 1, 42)
    map.set(m, 2, 100)
    local h1: int = 0
    if map.has(m, 1) then
        h1 = 1
    end
    local h9: int = 0
    if map.has(m, 9) then
        h9 = 1
    end
    local v1: int = map.get(m, 1):unwrap_or(7)
    local v9: int = map.get(m, 9):unwrap_or(7)
    return h1 * 1000 + h9 * 100 + v1 + v9
end
"#;
    let map_ops_res = run_mir_fn_cranelift_and_vm(
        map_ops_src,
        "ng_w15_6_map_extras",
        "test_map_extras",
        vec![],
    );
    assert_eq!(map_ops_res, Value::Int(1049));

    // (d) User `enum Op` (`BENCH-NG-06` `adt_enum_eval`)
    let ng06_src = r#"
enum Op
    Lit(int)
    Add(int, int)
    Mul(int, int)
    Xor(int, int)
    Reset
end

function make_op(step: int): Op
    local kind: int = step % 5
    if kind == 0 then
        return Op.Lit(step % 97)
    elseif kind == 1 then
        return Op.Add(step % 131, (step * 7) % 131)
    elseif kind == 2 then
        return Op.Mul((step % 19) + 1, (step % 23) + 1)
    elseif kind == 3 then
        return Op.Xor(step, step * 3)
    else
        return Op.Reset
    end
end

function apply_op(acc: int, op: Op): int
    if op is Lit(v) then
        return (acc + v) % 1000000007
    elseif op is Add(a, b) then
        return (acc + a + b) % 1000000007
    elseif op is Mul(a, b) then
        return (acc + a * b) % 1000000007
    elseif op is Xor(a, b) then
        return (acc + (a ~ b)) % 1000000007
    else
        return (acc / 2)
    end
end

function adt_enum_eval_fn(): int
    local state: int = 1
    local i: int = 0
    while i < 1000000 do
        local op: Op = make_op(i)
        state = apply_op(state, op)
        i = i + 1
    end
    return state
end
"#;
    let ng06_res = run_mir_fn_cranelift_and_vm(
        ng06_src,
        "ng_w15_6_ng06",
        "adt_enum_eval_fn",
        vec![],
    );
    assert_eq!(
        ng06_res,
        Value::Int(2262427),
        "BENCH-NG-06 Cranelift checksum mismatch"
    );

    // (e) Multi-field Enum runtime helpers (`enum.alloc2`, `enum.tag`, `enum.payload_at`) direct MIR test
    {
        use lust::ast::Span;
        use lust::mir::ir::{
            BlockId, FunctionSignature, LocalDecl, LocalId, MirBinaryOp, MirConstant,
            MirType, Operand, Place, Rvalue, Statement, Terminator,
        };
        let mut f = MirFunction::new(
            FunctionSignature {
                name: "enum2_direct".into(),
                param_types: vec![MirType::Int, MirType::Int],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            Span::dummy(),
        );
        f.locals.clear();
        f.params.clear();
        let a = LocalId(0);
        f.locals.push(LocalDecl::param(a, Some("a".into()), MirType::Int, Span::dummy()));
        f.params.push(a);
        let b = LocalId(1);
        f.locals.push(LocalDecl::param(b, Some("b".into()), MirType::Int, Span::dummy()));
        f.params.push(b);
        let e = f.add_local(Some("e".into()), MirType::Enum("Pair".into()), true, Span::dummy());
        let tag = f.add_local(Some("tag".into()), MirType::Int, true, Span::dummy());
        let p0 = f.add_local(Some("p0".into()), MirType::Int, true, Span::dummy());
        let p1 = f.add_local(Some("p1".into()), MirType::Int, true, Span::dummy());
        let s1 = f.add_local(Some("s1".into()), MirType::Int, true, Span::dummy());
        let s2 = f.add_local(Some("s2".into()), MirType::Int, true, Span::dummy());

        let bb0 = BlockId(0);
        let bb1 = f.new_block();
        let bb2 = f.new_block();
        let bb3 = f.new_block();
        let bb4 = f.new_block();

        f.blocks[bb0.0].terminator = Some(Terminator::Call {
            callee: "enum.alloc2".into(),
            args: vec![
                Operand::Constant(MirConstant::Int(3)),
                Operand::Copy(Place::Local(a)),
                Operand::Copy(Place::Local(b)),
                Operand::Constant(MirConstant::Int(2)),
            ],
            destination: Place::Local(e),
            success: bb1,
            error: None,
            span: Span::dummy(),
        });
        f.blocks[bb1.0].terminator = Some(Terminator::Call {
            callee: "enum.tag".into(),
            args: vec![Operand::Copy(Place::Local(e))],
            destination: Place::Local(tag),
            success: bb2,
            error: None,
            span: Span::dummy(),
        });
        f.blocks[bb2.0].terminator = Some(Terminator::Call {
            callee: "enum.payload_at".into(),
            args: vec![Operand::Copy(Place::Local(e)), Operand::Constant(MirConstant::Int(0))],
            destination: Place::Local(p0),
            success: bb3,
            error: None,
            span: Span::dummy(),
        });
        f.blocks[bb3.0].terminator = Some(Terminator::Call {
            callee: "enum.payload_at".into(),
            args: vec![Operand::Copy(Place::Local(e)), Operand::Constant(MirConstant::Int(1))],
            destination: Place::Local(p1),
            success: bb4,
            error: None,
            span: Span::dummy(),
        });
        f.blocks[bb4.0].statements.push(Statement::Assign(
            Place::Local(s1),
            Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(p0)), Operand::Copy(Place::Local(p1))),
            Span::dummy(),
        ));
        f.blocks[bb4.0].statements.push(Statement::Assign(
            Place::Local(s2),
            Rvalue::BinaryOp(MirBinaryOp::Mul, Operand::Copy(Place::Local(tag)), Operand::Copy(Place::Local(s1))),
            Span::dummy(),
        ));
        f.blocks[bb4.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(s2))),
            Span::dummy(),
        ));

        let mut mod_e = MirModule::new("ng_w15_6_enum2".into());
        mod_e.add_function(f);
        reset_all();
        let mut vm = VM::new();
        vm.load_mir_module(&mod_e).expect("load enum2");
        vm.set_backend(VmBackend::Cranelift);
        let res = vm
            .call("enum2_direct", vec![Value::Int(17), Value::Int(25)])
            .expect("enum2_direct must execute natively");
        assert_eq!(res, Value::Int(3 * (17 + 25)));
        assert!(vm.cranelift_entry_count > 0);
    }

    // (f) `string.gsub` (`BENCH-NG-10` `regex_match_extract`)
    let ng10_src = r##"
function regex_match_extract_fn(): int
    local i: int = 0
    local matched_len: int = 0
    while i < 20000 do
        local line: string = "user_" .. tostring(100 + (i % 900)) .. "_req_" .. tostring(i * 7)
        local res: Array<string> = string.gsub(line, "[0-9]+", "#")
        if res[0] is Ok(out) then
            matched_len = matched_len + string.len(out)
        end
        i = i + 1
    end
    return matched_len
end
"##;
    let ng10_res = run_mir_fn_cranelift_and_vm(
        ng10_src,
        "ng_w15_6_ng10",
        "regex_match_extract_fn",
        vec![],
    );
    assert_eq!(
        ng10_res,
        Value::Int(240000),
        "BENCH-NG-10 Cranelift checksum mismatch"
    );

    // (g) `string.sub` + `string.split` (`BENCH-NG-13` `schema_json_decode_pipeline`)
    let ng13_src = r#"
struct TelemetryEvent
    ts: int
    device_id: int
    metric: int
    active: bool
end

function parse_int(s: string): int
    local n: int = string.len(s)
    local idx: int = 1
    local acc: int = 0
    while idx <= n do
        local ch: string = string.sub(s, idx, idx)
        local d: int = 0
        if ch == "1" then d = 1
        elseif ch == "2" then d = 2
        elseif ch == "3" then d = 3
        elseif ch == "4" then d = 4
        elseif ch == "5" then d = 5
        elseif ch == "6" then d = 6
        elseif ch == "7" then d = 7
        elseif ch == "8" then d = 8
        elseif ch == "9" then d = 9
        end
        acc = acc * 10 + d
        idx = idx + 1
    end
    return acc
end

function decode_telemetry_event(raw: string): TelemetryEvent
    local n: int = string.len(raw)
    local inner: string = string.sub(raw, 2, n - 1)
    local fields: Array<string> = string.split(inner, ",")
    local ts: int = 0
    local device_id: int = 0
    local metric: int = 0
    local active: bool = false
    local idx: int = 0
    local fcount: int = array.len(fields)
    while idx < fcount do
        if fields[idx] is Ok(entry) then
            local kv: Array<string> = string.split(entry, ":")
            if kv[0] is Ok(k) then
                if kv[1] is Ok(v) then
                    if k == "\"ts\"" then
                        ts = parse_int(v)
                    elseif k == "\"device_id\"" then
                        device_id = parse_int(v)
                    elseif k == "\"metric\"" then
                        metric = parse_int(v)
                    elseif k == "\"active\"" then
                        active = (v == "true")
                    end
                end
            end
        end
        idx = idx + 1
    end
    return TelemetryEvent {
        ts = ts,
        device_id = device_id,
        metric = metric,
        active = active
    }
end

function schema_decode_fn(iters: int): int
    local payload_a: string = "{\"ts\":1700000100,\"device_id\":42,\"metric\":950,\"active\":true}"
    local payload_b: string = "{\"ts\":1700000200,\"device_id\":19,\"metric\":310,\"active\":false}"
    local total: int = 0
    local i: int = 0
    while i < iters do
        local raw: string = payload_a
        if (i & 1) == 1 then
            raw = payload_b
        end
        local ev: TelemetryEvent = decode_telemetry_event(raw)
        if ev.active then
            total = (total + ev.metric * ev.device_id + (ev.ts % 1000)) % 1000000007
        else
            total = (total + ev.metric + ev.device_id) % 1000000007
        end
        i = i + 1
    end
    return total
end
"#;
    // Verify exact parity on 200 iterations and verify full-scale 50,000-iter oracle checksum `8224993`!
    let ng13_small = run_mir_fn_cranelift_and_vm(
        ng13_src,
        "ng_w15_6_ng13_small",
        "schema_decode_fn",
        vec![Value::Int(200)],
    );
    assert!(matches!(ng13_small, Value::Int(_)));
    let mut ng13_mod = parse_and_lower_module(ng13_src, "ng_w15_6_ng13_full");
    MirOptimizer::optimize_module_for_jit(&mut ng13_mod, &JitInlineConfig::default());
    reset_all();
    let mut vm13 = VM::new();
    vm13.load_mir_module(&ng13_mod).expect("load ng13");
    vm13.set_backend(VmBackend::Cranelift);
    let ng13_full = vm13
        .call("schema_decode_fn", vec![Value::Int(50000)])
        .expect("ng13 full must run in Cranelift");
    assert_eq!(ng13_full, Value::Int(8224993));
    assert!(vm13.cranelift_entry_count > 0);

    // (h) `string.sub` + `string.byte` + `string.find` in Cranelift JIT
    //     First verify `string.sub` exact Cranelift == Bytecode VM parity:
    let str_sub_src = r#"
function test_string_sub_parity(): int
    local s: string = "alpha=123;beta=456"
    local sub1: string = string.sub(s, 7, 9)
    local sub2: string = string.sub(s, -3)
    local ok1: int = 0
    local ok2: int = 0
    if sub1 == "123" then ok1 = 10 end
    if sub2 == "456" then ok2 = 20 end
    return ok1 + ok2 + string.len(sub1) + string.len(sub2)
end
"#;
    let str_sub_res = run_mir_fn_cranelift_and_vm(
        str_sub_src,
        "ng_w15_6_str_sub",
        "test_string_sub_parity",
        vec![],
    );
    assert_eq!(str_sub_res, Value::Int(36));

    //     Next verify `string.sub` + `string.byte` (array & scalar) + `string.find` (array & scalar)
    //     natively in Cranelift JIT against the unboxed `Array<int>` / `int` MIR & AOT contract
    //     (Note: legacy Bytecode VM `stdlib.rs` wraps multi-return `string.byte`/`find` in `LuaValue::Int`).
    let str_ops_src = r#"
function test_string_sub_byte_find(): int
    local s: string = "alpha=123;beta=456"
    local sub: string = string.sub(s, 7, 9)
    local b_arr: Array<int> = string.byte(sub, 1, 3)
    local bsum: int = 0
    if b_arr[0] is Ok(b0) then bsum = bsum + b0 end
    if b_arr[1] is Ok(b1) then bsum = bsum + b1 end
    if b_arr[2] is Ok(b2) then bsum = bsum + b2 end
    local found: Array<int> = string.find(s, "beta", 1, true)
    local fpos: int = 0
    if found[0] is Ok(p0) then fpos = fpos + p0 end
    if found[1] is Ok(p1) then fpos = fpos + p1 end
    local miss: Array<int> = string.find(s, "zzzz", 1, true)
    local mlen: int = array.len(miss)
    local b_first: int = string.byte(sub, 1)
    local f_first: int = string.find(s, "beta", 1, true)
    return bsum * 10000 + fpos * 100 + b_first + f_first + mlen * 10000000
end
"#;
    let mut str_ops_mod = parse_and_lower_module(str_ops_src, "ng_w15_6_str_ops");
    MirOptimizer::optimize_module_for_jit(&mut str_ops_mod, &JitInlineConfig::default());
    reset_all();
    let mut vm_str = VM::new();
    vm_str.load_mir_module(&str_ops_mod).expect("load str_ops");
    vm_str.set_backend(VmBackend::Cranelift);
    let str_ops_res = vm_str
        .call("test_string_sub_byte_find", vec![])
        .expect("test_string_sub_byte_find must execute natively in Cranelift");
    assert!(
        vm_str.cranelift_entry_count > 0,
        "test_string_sub_byte_find must run natively via Cranelift JIT"
    );
    // '1'(49)+'2'(50)+'3'(51) = 150; "beta" at 1-based [11, 14] -> fpos=25; b_first=49; f_first=11;
    // miss (`string_find_values` returns `[Value::Nil]` -> len=1):
    // 1*10000000 + 150*10000 + 25*100 + 49 + 11 = 11502560
    assert_eq!(
        str_ops_res,
        Value::Int(11502560),
        "Unexpected test_string_sub_byte_find result: {str_ops_res:?}"
    );

    // (i) `Array<Struct>` (`BENCH-NG-08` `cycle_collector_pressure.lust` kernel)
    //     Uses `compile_program_with_config` so recursive `struct Cell` metadata is registered
    //     in `VM::struct_metadata` (matching `lust run --cranelift`).
    let ng08_src = r#"
struct Cell
    val: int
    next: Option<Cell>
end

function cycle_pool_fn(pool_n: int, iters: int): int
    local live_pool: Array<Cell> = []
    local k: int = 0
    while k < pool_n do
        local c: Cell = Cell { val = k, next = Option.None }
        array.push(live_pool, c)
        k = k + 1
    end
    local total: int = 0
    local i: int = 0
    while i < iters do
        local a: Cell = Cell { val = i, next = Option.None }
        local b: Cell = Cell { val = i + 1, next = Option.Some(a) }
        a.next = Option.Some(b)
        if a.next is Some(nb) then
            total = (total + nb.val) % 1000000007
        end
        if (i & 1) == 0 then
            a.next = Option.None
        end
        i = i + 1
    end
    return total + array.len(live_pool)
end
"#;
    {
        let entries = &[lust::EmbeddedModule {
            module: "main",
            parent: None,
            source: Some(ng08_src),
        }];
        let prog_jit =
            lust::load_program_from_embedded(entries, "main").expect("load embedded ng08 jit");
        let mut cfg_jit = LustConfig::default().with_cranelift_jit(true);
        cfg_jit.set_use_mir(false);
        reset_all();
        let mut vm_jit =
            lust::compile_program_with_config(prog_jit, &cfg_jit).expect("compile ng08 jit");
        let fn_name_jit = if vm_jit.function_by_name("main.cycle_pool_fn").is_some() {
            "main.cycle_pool_fn"
        } else {
            "cycle_pool_fn"
        };
        let ng08_jit = vm_jit
            .call(fn_name_jit, vec![Value::Int(64), Value::Int(500)])
            .expect("cycle_pool_fn must execute natively in Cranelift");
        assert!(
            vm_jit.cranelift_entry_count > 0,
            "cycle_pool_fn (Array<Cell>) must execute natively via Cranelift JIT (entries={}, compiled={:?})",
            vm_jit.cranelift_entry_count,
            vm_jit.cranelift_compiled.keys().collect::<Vec<_>>()
        );

        let prog_bc =
            lust::load_program_from_embedded(entries, "main").expect("load embedded ng08 bc");
        let mut cfg_bc = LustConfig::default().with_cranelift_jit(false);
        cfg_bc.set_use_mir(false);
        reset_all();
        let mut vm_bc =
            lust::compile_program_with_config(prog_bc, &cfg_bc).expect("compile ng08 bc");
        let fn_name_bc = if vm_bc.function_by_name("main.cycle_pool_fn").is_some() {
            "main.cycle_pool_fn"
        } else {
            "cycle_pool_fn"
        };
        let ng08_bc = vm_bc
            .call(fn_name_bc, vec![Value::Int(64), Value::Int(500)])
            .expect("cycle_pool_fn vm");
        assert_eq!(ng08_jit, ng08_bc);
        // sum_{i=0..500} (i + 1) + 64 = 125250 + 64 = 125314
        assert_eq!(ng08_jit, Value::Int(125314));
    }
}

// ============================================================================
// 3. Mandatory `SEC-01` Exact Gas Preservation Gate (`0, 1, 50 -> 51, 63, 64, 65`)
// ============================================================================

#[test]
fn test_ng_w15_6_sec01_exact_gas_preservation_peeled_header_and_quantized_backedge() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    // Hoisted array loops exercising:
    //  - 0-trip peeled inner loop (like `sieve.lust` for primes `i > sqrt(N)`)
    //  - 1-trip peeled inner loop (enters `peeled_header_bb` + 1 body + 1 backedge into `clif_blocks[hdr]`)
    //  - multi-trip peeled inner loop crossing Q=64 quantum boundary
    let src = r#"
function hoisted_single(arr: Array<int>, n: int): int
    local j: int = 0
    local total: int = 0
    while j < n do
        total = total + arr[j]
        j = j + 1
    end
    return total
end

function hoisted_nested_mixed(arr: Array<int>, outer: int, inner: int): int
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
    let module = parse_and_lower_module(src, "ng_w15_6_sec01");

    // Verify hoisted_nested_mixed arms Q=64 on backedge header ONLY (3 x iconst.i64 63)
    let mut comp = CraneliftJitCompiler::new().expect("compiler");
    let nested_ir = comp
        .dump_function_ir(&module.functions[1])
        .expect("dump hoisted_nested_mixed IR");
    let parsed_nested = ClifIrView::parse(&nested_ir);
    assert_eq!(
        parsed_nested.count_iconst_63(),
        3,
        "hoisted_nested_mixed must arm Q=64 strictly on backedge header (not peeled_header_bb):\n{nested_ir}"
    );

    let arr100 = Value::array((1..=100).map(Value::Int).collect());

    // (a) Exact `50 -> 51` gas boundary on `hoisted_single(arr, 49)`:
    //     1 VM entry + (49 + 1) = 50 loop header checks => exact total gas needed = 51!
    //     - gas_limit in [0, 1, 50] => OutOfGas (0 remaining, zero overshoot!)
    //     - gas_limit == 51         => Ok(49 * 50 / 2 = 1225), gas_remaining == Some(0)!
    for budget in [0u64, 1, 50, 51, 63, 64, 65] {
        reset_all();
        let mut vm = VM::new();
        vm.load_mir_module(&module).expect("load sec01");
        vm.set_backend(VmBackend::Cranelift);
        vm.set_gas_budget(budget);

        let res = vm.call("hoisted_single", vec![arr100.clone(), Value::Int(49)]);
        if budget <= 50 {
            let err = res.expect_err(&format!(
                "SEC-01: hoisted_single(49) must trap OutOfGas at budget={budget}"
            ));
            assert!(
                err.to_string().contains("Out of gas"),
                "budget={budget}: expected Out of gas, got {err}"
            );
            assert_eq!(vm.gas_remaining(), Some(0));
            let expected_sp = if budget == 0 { 1 } else { budget };
            let expected_gas_used = if budget == 0 { 2 } else { budget + 1 };
            assert_eq!(
                vm.gas_used(),
                expected_gas_used,
                "SEC-01: zero gas overshoot at budget={budget}"
            );
            assert_eq!(
                safepoint_stats().safepoints_checked,
                expected_sp,
                "SEC-01: exact safepoints_checked at budget={budget}"
            );
        } else {
            let val = res.expect(&format!(
                "SEC-01: hoisted_single(49) must succeed at budget={budget}"
            ));
            assert_eq!(val, Value::Int(49 * 50 / 2));
            assert_eq!(vm.gas_used(), 51);
            assert_eq!(vm.gas_remaining(), Some(budget - 51));
            assert_eq!(safepoint_stats().safepoints_checked, 50);
        }
    }

    // (b) Exact Q=64 boundary (`63, 64, 65`) on `hoisted_single(arr, 63)` (needs 1 + 64 = 65 gas)
    //     and on `hoisted_nested_mixed(arr, 4, 20)` (needs 1 + 5 + 4*21 = 90 gas):
    for budget in [0u64, 1, 50, 51, 63, 64, 65] {
        // 1. hoisted_single(arr, 63): exact boundary between 64 (OOG) and 65 (Ok)!
        reset_all();
        let mut vm_s = VM::new();
        vm_s.load_mir_module(&module).expect("load sec01");
        vm_s.set_backend(VmBackend::Cranelift);
        vm_s.set_gas_budget(budget);
        let res_s = vm_s.call("hoisted_single", vec![arr100.clone(), Value::Int(63)]);
        if budget < 65 {
            let err = res_s.expect_err(&format!(
                "SEC-01: hoisted_single(63) must trap OutOfGas at budget={budget}"
            ));
            assert!(err.to_string().contains("Out of gas"));
            assert_eq!(vm_s.gas_remaining(), Some(0));
            let expected_sp = if budget == 0 { 1 } else { budget };
            let expected_gas_used = if budget == 0 { 2 } else { budget + 1 };
            assert_eq!(vm_s.gas_used(), expected_gas_used);
            assert_eq!(safepoint_stats().safepoints_checked, expected_sp);
        } else {
            let val = res_s.expect("hoisted_single(63) must succeed at budget=65");
            assert_eq!(val, Value::Int(63 * 64 / 2));
            assert_eq!(vm_s.gas_used(), 65);
            assert_eq!(vm_s.gas_remaining(), Some(0));
            assert_eq!(safepoint_stats().safepoints_checked, 64);
        }

        // 2. hoisted_nested_mixed(arr, 5, 20) (needs 1 + 6 + 5*21 = 112 gas):
        //    must trap with ZERO overshoot across ALL budgets [0, 1, 50, 51, 63, 64, 65]!
        reset_all();
        let mut vm_n = VM::new();
        vm_n.load_mir_module(&module).expect("load sec01");
        vm_n.set_backend(VmBackend::Cranelift);
        vm_n.set_gas_budget(budget);
        let err_n = vm_n
            .call(
                "hoisted_nested_mixed",
                vec![arr100.clone(), Value::Int(5), Value::Int(20)],
            )
            .expect_err(&format!(
                "SEC-01: hoisted_nested_mixed(5, 20) must trap OutOfGas at budget={budget}"
            ));
        assert!(err_n.to_string().contains("Out of gas"));
        assert_eq!(vm_n.gas_remaining(), Some(0));
        let expected_sp = if budget == 0 { 1 } else { budget };
        let expected_gas_used = if budget == 0 { 2 } else { budget + 1 };
        assert_eq!(
            vm_n.gas_used(),
            expected_gas_used,
            "SEC-01: hoisted_nested_mixed zero overshoot at budget={budget}"
        );
        assert_eq!(safepoint_stats().safepoints_checked, expected_sp);
    }

    // (c) Exact gas & safepoint reconciliation across 0-trip, 1-trip, 2-trip, and Q=64-crossing inner loops!
    for (outer, inner) in [(5i64, 0i64), (5, 1), (5, 2), (3, 63), (3, 64), (3, 65), (2, 100)] {
        let expected_headers = ((outer + 1) + outer * (inner + 1)) as u64;
        let expected_gas_used = 1 + expected_headers;
        let expected_val = outer * (inner * (inner + 1) / 2);

        reset_all();
        let mut vm_jit = VM::new();
        vm_jit.load_mir_module(&module).expect("load sec01");
        vm_jit.set_backend(VmBackend::Cranelift);
        vm_jit.set_gas_budget(50_000);
        let res_jit = vm_jit
            .call(
                "hoisted_nested_mixed",
                vec![arr100.clone(), Value::Int(outer), Value::Int(inner)],
            )
            .expect("hoisted_nested_mixed jit");
        assert_eq!(res_jit, Value::Int(expected_val));
        assert_eq!(
            vm_jit.gas_used(),
            expected_gas_used,
            "hoisted_nested_mixed({outer}, {inner}) exact gas_used"
        );
        assert_eq!(vm_jit.gas_remaining(), Some(50_000 - expected_gas_used));
        assert_eq!(
            safepoint_stats().safepoints_checked,
            expected_headers,
            "hoisted_nested_mixed({outer}, {inner}) exact safepoints_checked"
        );

        // Parity with Bytecode VM
        reset_all();
        let mut vm_bc = VM::new();
        vm_bc.load_mir_module(&module).expect("load sec01 bc");
        vm_bc.set_backend(VmBackend::Bytecode);
        let res_bc = vm_bc
            .call(
                "hoisted_nested_mixed",
                vec![arr100.clone(), Value::Int(outer), Value::Int(inner)],
            )
            .expect("hoisted_nested_mixed bc");
        assert_eq!(res_jit, res_bc);
    }

    // (d) Exact 50 -> 51 boundary on nested peeled loop `hoisted_nested_mixed(arr, 4, 10)`:
    //     1 VM entry + (4 + 1) outer + 4 * (10 + 1) inner = 1 + 5 + 44 = 50 headers + 1 entry?
    //     Wait: outer=5, inner=8 => outer checks=6, inner checks=5*9=45 -> wait:
    //     outer=4, inner=10 => outer checks = 5, inner checks = 4*11 = 44 => total headers = 49, total gas = 50!
    //     outer=5, inner=8  => outer checks = 6, inner checks = 5*9  = 45 => total headers = 51, total gas = 52!
    //     outer=1, inner=47 => outer checks = 2, inner checks = 1*48 = 48 => total headers = 50, total gas = 51!
    for budget in [50u64, 51u64] {
        reset_all();
        let mut vm = VM::new();
        vm.load_mir_module(&module).expect("load sec01");
        vm.set_backend(VmBackend::Cranelift);
        vm.set_gas_budget(budget);
        let res = vm.call(
            "hoisted_nested_mixed",
            vec![arr100.clone(), Value::Int(1), Value::Int(47)],
        );
        if budget == 50 {
            let err = res.expect_err("nested (1, 47) must trap OutOfGas at budget=50");
            assert!(err.to_string().contains("Out of gas"));
            assert_eq!(vm.gas_used(), 51);
            assert_eq!(vm.gas_remaining(), Some(0));
            assert_eq!(safepoint_stats().safepoints_checked, 50);
        } else {
            let val = res.expect("nested (1, 47) must succeed at budget=51");
            assert_eq!(val, Value::Int(47 * 48 / 2));
            assert_eq!(vm.gas_used(), 51);
            assert_eq!(vm.gas_remaining(), Some(0));
            assert_eq!(safepoint_stats().safepoints_checked, 50);
        }
    }
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_ng_w15_6_peeled_header_and_backedge_deopt_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    let src = r#"
function hoisted_nested_deopt(arr: Array<int>, outer: int, inner: int): int
    local i: int = 0
    local total: int = 0
    while i < outer do
        local j: int = 0
        while j < inner do
            if arr[j] is Ok(v) then
                total = total + v
            end
            j = j + 1
        end
        i = i + 1
    end
    return total
end
"#;
    let module = parse_and_lower_module(src, "ng_w15_6_deopt_mod");
    let arr = Value::array((1..=70).map(Value::Int).collect());
    let expected = 3 * (70 * 71 / 2);

    // Poll 1 = outer header; Poll 2 = inner peeled_header_bb; Poll 3 = inner backedge Q=64 arm;
    // Polls 17, 63, 64, 65 = mid-quantum & boundary polls!
    for poll_k in [1u64, 2, 3, 17, 63, 64, 65] {
        reset_all();
        let mut vm = VM::new();
        vm.load_mir_module(&module).expect("load deopt mod");
        vm.set_backend(VmBackend::Cranelift);
        vm.set_gas_budget(100_000);
        safepoint_request_deopt_on_poll(None, poll_k);
        let res = vm
            .call(
                "hoisted_nested_deopt",
                vec![arr.clone(), Value::Int(3), Value::Int(70)],
            )
            .unwrap_or_else(|e| panic!("Deopt resume failed at poll {poll_k}: {e:?}"));
        assert_eq!(res, Value::Int(expected), "Deopt parity mismatch at poll {poll_k}");
    }
}

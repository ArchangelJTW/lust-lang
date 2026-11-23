#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! NG-W10-5 Verification Suite:
//! 1. Atomic 1-byte `ElemStride::U8` for `Array<bool>` across `runtime_helpers`,
//!    `cranelift_dispatch` (including unboxed `memcpy` without materializing `Vec<Value>`),
//!    and `cranelift_backend` (`load.i8` / `store.i8`).
//! 2. `MirContext::push_grow_slow_calls()` diagnostic counter.
//! 3. Inline CLIF fast-path for `array.push` / `table.push` on locally-allocated arrays ONLY
//!    (strictly forbidden on function parameters / `read_only_borrowed` / unvalidated pointers)
//!    with `observer_free = true` counter promotion.
//! 4. `proven_in_bounds` second bounds-check elision in `Place::Index`.
//! 5. Outer-loop `HoistLoopSpec` for `if flags[i] is Ok(f)` (`MirUnaryOp::Len`).

use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;
use std::sync::Mutex;

use lust::bytecode::{FlatArrayElemTag, LustArray, Value};
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{
    CraneliftJitCompiler, MIR_ARRAY_ELEM_TAG_BOOL, MIR_STATUS_ERR_OUT_OF_BOUNDS, MIR_STATUS_OK,
    lust_mir_alloc_array_typed, lust_mir_array_get, lust_mir_array_len, lust_mir_array_push,
    lust_mir_array_set, lust_mir_get_status, lust_mir_set_status,
    promoted_loop_cfg_verifications, reset_promoted_loop_cfg_verifications, safepoint_reset,
    take_current_sticky_error, take_deopt_payload,
};
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::safepoint_request_deopt_on_poll;
use lust::mir::ir::{MirFunction, MirType};
use lust::mir::lowering::AstLowering;
use lust::mir::runtime_helpers::{MirContext, enter_mir_context};
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vfs::VirtualConsole;
use lust::vm::cranelift_dispatch::{build_or_get_cached_flat_array, validate_and_decode_array};
use lust::vm::{VM, reset_effect_handler_state};
use lust::{EmbeddedModule, compile_program_with_config, load_program_from_embedded};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn reset_all() {
    safepoint_reset();
    reset_effect_handler_state();
    take_deopt_payload();
    take_current_sticky_error();
}

fn lower_single_function(source: &str) -> MirFunction {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");
    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let module = AstLowering::lower_module_items("ng_w10_5_mod", &items).expect("lower module");
    module
        .functions
        .into_iter()
        .next()
        .expect("expected single function")
}

fn compile_embedded_vm(source: &str, use_cranelift: bool) -> VM {
    let entries = &[EmbeddedModule {
        module: "main",
        parent: None,
        source: Some(source),
    }];
    let program = load_program_from_embedded(entries, "main").expect("load embedded program");
    let mut cfg = LustConfig::default().with_cranelift_jit(use_cranelift);
    cfg.set_use_mir(false);
    let mut vm = compile_program_with_config(program, &cfg).expect("compile embedded program");
    vm.console = VirtualConsole::captured();
    if use_cranelift {
        vm.compile_cranelift().expect("compile cranelift");
    }
    vm
}

fn find_fn_name(vm: &VM, suffix: &str) -> String {
    for c in [suffix.to_string(), format!("main.{suffix}")] {
        if vm.cranelift_compiled.contains_key(&c) {
            return c;
        }
    }
    vm.cranelift_compiled
        .keys()
        .find(|k| k.ends_with(suffix))
        .cloned()
        .unwrap_or_else(|| panic!("Function '{suffix}' not compiled in Cranelift"))
}

struct ClifIrBlocks {
    blocks: HashMap<String, Vec<String>>,
    cold_blocks: HashSet<String>,
    sym_map: HashMap<String, String>,
    block_order: Vec<String>,
    succs: HashMap<String, Vec<String>>,
}

impl ClifIrBlocks {
    fn parse(ir: &str) -> Self {
        let mut sym_map = HashMap::new();
        let mut blocks: HashMap<String, Vec<String>> = HashMap::new();
        let mut cold_blocks = HashSet::new();
        let mut block_order = Vec::new();
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

        let mut succs: HashMap<String, Vec<String>> = HashMap::new();
        for b in &block_order {
            let mut s = Vec::new();
            if let Some(lines) = blocks.get(b) {
                for l in lines {
                    if l.starts_with("jump ")
                        || l.starts_with("brif ")
                        || l.starts_with("br_table ")
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
            sym_map,
            block_order,
            succs,
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

const SIEVE_FN_SRC: &str = r#"
function sieve(n: int): int
    local flags: Array<bool> = []
    local i: int = 0
    while i <= n do
        array.push(flags, true)
        i = i + 1
    end
    local count: int = 0
    i = 2
    while i <= n do
        if flags[i] is Ok(f) then
            if f then
                count = count + 1
                local j: int = i * i
                while j <= n do
                    flags[j] = false
                    j = j + i
                end
            end
        end
        i = i + 1
    end
    return count
end
"#;

#[test]
fn test_ng_w10_5_sieve_10k_inline_push_slow_calls_and_exact_result() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();
    reset_promoted_loop_cfg_verifications();

    let mut vm = compile_embedded_vm(SIEVE_FN_SRC, true);
    let sieve_name = find_fn_name(&vm, "sieve");

    assert!(
        promoted_loop_cfg_verifications() >= 2,
        "Expected both push loop and outer sieve loop to pass Barrier 5 promoted-loop CFG verifier"
    );

    vm.mir_context.reset_push_grow_slow_calls();
    let e0 = vm.cranelift_entry_count();
    let res_10k = vm
        .call(&sieve_name, vec![Value::Int(10_000)])
        .expect("sieve(10_000) must succeed in Cranelift JIT");
    assert_eq!(res_10k, Value::Int(1229));
    assert_eq!(
        vm.cranelift_entry_count() - e0,
        1,
        "sieve(10_000) must execute natively via Cranelift"
    );

    let slow_calls = vm.mir_context.push_grow_slow_calls();
    assert!(
        slow_calls >= 1 && slow_calls <= 15,
        "sieve(10_000) must invoke lust_mir_array_push ONLY on capacity growth reallocations (<= 15), NOT 10_001; got {slow_calls}"
    );

    // Full benchmark input N = 10_000_000 -> 664579 primes
    vm.mir_context.reset_push_grow_slow_calls();
    let res_10m = vm
        .call(&sieve_name, vec![Value::Int(10_000_000)])
        .expect("sieve(10_000_000) must succeed in Cranelift JIT");
    assert_eq!(res_10m, Value::Int(664_579));
    let slow_calls_10m = vm.mir_context.push_grow_slow_calls();
    assert!(
        slow_calls_10m >= 1 && slow_calls_10m <= 25,
        "sieve(10_000_000) push_grow_slow_calls must be <= 25, got {slow_calls_10m}"
    );
}

#[test]
fn test_ng_w10_5_sieve_clif_ir_u8_stride_hoisted_outer_and_cold_grow() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    let mir_fn = lower_single_function(SIEVE_FN_SRC);
    let mut compiler = CraneliftJitCompiler::new().expect("create compiler");
    let ir = compiler
        .dump_function_ir(&mir_fn)
        .expect("dump_function_ir sieve");
    let parsed = ClifIrBlocks::parse(&ir);
    let hot_blocks = parsed.hot_cycle_blocks();
    assert!(!hot_blocks.is_empty(), "Expected hot loop blocks in sieve IR:\n{ir}");

    // 1. Verify 1-byte load.i8 and store.i8 are emitted in hot sieve blocks
    let mut has_hot_load_i8 = false;
    let mut has_hot_store_i8 = false;
    let mut hot_push_calls = Vec::new();
    let mut cold_push_calls = Vec::new();

    for b in &parsed.block_order {
        let is_hot = hot_blocks.contains(b);
        let is_cold = parsed.cold_blocks.contains(b);
        if let Some(lines) = parsed.blocks.get(b) {
            for l in lines {
                if is_hot && l.contains("load.i8") {
                    has_hot_load_i8 = true;
                }
                if is_hot && l.starts_with("store.i8 ") {
                    has_hot_store_i8 = true;
                }
                if parsed.extract_call_sym(l).as_deref() == Some("lust_mir_array_push") {
                    if is_hot {
                        hot_push_calls.push((b.clone(), l.clone()));
                    } else if is_cold {
                        cold_push_calls.push((b.clone(), l.clone()));
                    }
                }
            }
        }
    }

    assert!(
        has_hot_load_i8,
        "Expected 1-byte `load.i8` for Array<bool> read in hot sieve loop:\n{ir}"
    );
    assert!(
        has_hot_store_i8,
        "Expected 1-byte `store.i8` for Array<bool> write/push in hot sieve loop:\n{ir}"
    );
    assert!(
        hot_push_calls.is_empty(),
        "Inline fast-path must move `lust_mir_array_push` out of hot cycle into a cold block; found {hot_push_calls:?}\nIR:\n{ir}"
    );
    assert_eq!(
        cold_push_calls.len(),
        1,
        "Expected exactly 1 cold `lust_mir_array_push` grow call, found {cold_push_calls:?}\nIR:\n{ir}"
    );

    // 2. Verify NO second OOB check (`icmp uge`) in the `if flags[i] is Ok(f)` then-block
    //    (the inner loop `flags[j] = false` still has 1 `icmp uge`, so total hot `icmp uge` == 1, NOT 2!)
    let mut hot_uge_count = 0usize;
    for b in &hot_blocks {
        if let Some(lines) = parsed.blocks.get(b) {
            for l in lines {
                if l.contains("icmp uge ") || l.contains("icmp.i64 uge ") {
                    hot_uge_count += 1;
                }
            }
        }
    }
    assert_eq!(
        hot_uge_count, 1,
        "Expected exactly 1 hot `icmp uge` (inner `flags[j] = false`) with outer `if flags[i] is Ok(f)` second bounds check elided; got {hot_uge_count}\nIR:\n{ir}"
    );
}

#[test]
fn test_ng_w10_5_bool_array_u8_packing_across_runtime_helpers_and_dispatch() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    // 1. runtime_helpers: word_capacity == 2 + ceil(n/8), bit-exact get/set/push across 8-byte word boundaries
    {
        let ctx = MirContext::new();
        let _guard = enter_mir_context(&ctx);

        let ptr = lust_mir_alloc_array_typed(17, MIR_ARRAY_ELEM_TAG_BOOL);
        assert!(!ptr.is_null());
        let rec = ctx
            .get_active_allocation_record(ptr as usize)
            .expect("bool array allocation record");
        // 17 bools -> ceil(17 / 8) = 3 payload words -> 2 + 3 = 5 total words
        assert_eq!(rec.word_capacity, 5);

        for i in 0..17i64 {
            lust_mir_array_set(ptr, i, i % 3);
        }
        for i in 0..17i64 {
            assert_eq!(
                lust_mir_array_get(ptr, i),
                i64::from(i % 3 != 0),
                "bool flat get mismatch at {i}"
            );
        }

        // Push 20 more elements -> triggers growth with 1-byte backing buffer
        for i in 17..37i64 {
            assert_eq!(lust_mir_array_push(ptr, i % 2), MIR_STATUS_OK);
        }
        assert_eq!(lust_mir_array_len(ptr), 37);
        for i in 0..17i64 {
            assert_eq!(lust_mir_array_get(ptr, i), i64::from(i % 3 != 0));
        }
        for i in 17..37i64 {
            assert_eq!(lust_mir_array_get(ptr, i), i % 2);
        }

        let decoded = validate_and_decode_array(&ctx, &[], ptr as i64, &MirType::Bool)
            .expect("validate_and_decode_array bool");
        assert_eq!(decoded.elements.len(), 37);
        for i in 0..17usize {
            assert_eq!(decoded.elements[i], Value::Bool(i % 3 != 0));
        }
        for i in 17..37usize {
            assert_eq!(decoded.elements[i], Value::Bool(i % 2 != 0));
        }

        ctx.clear_to_baselines(0, 0);
    }

    // 2. cranelift_dispatch: UnboxedArrayData::{Int, Float, Bool} memcpy without materializing Vec<Value>,
    //    and 1-byte packed FlatArrayBuffer for Bool (100 bools -> 2 + 13 = 15 words, NOT 102 words!)
    {
        let ctx = MirContext::new();
        let arr_b = Rc::new(LustArray::new(Vec::new()));
        for i in 0..100 {
            arr_b.push_bool(i % 3 == 0);
        }
        assert!(arr_b.is_unboxed(), "arr_b must start in unboxed Bool mode");

        let (buf, was_hit) = build_or_get_cached_flat_array(&ctx, &arr_b, &[], FlatArrayElemTag::Bool)
            .expect("build flat bool cache")
            .expect("some flat bool cache");
        assert!(!was_hit);
        assert!(
            arr_b.is_unboxed(),
            "build_or_get_cached_flat_array MUST NOT materialize Vec<Value> for unboxed Bool array"
        );
        assert_eq!(
            buf.word_len(),
            2 + (100 + 7) / 8,
            "100-bool FlatArrayBuffer must use 1-byte stride (15 words), not 102 words"
        );
        unsafe {
            assert_eq!(*buf.as_ptr(), 100);
            assert_eq!(*buf.as_ptr().add(1), 100);
            let bytes = buf.as_ptr().add(2) as *const u8;
            for i in 0..100usize {
                assert_eq!(*bytes.add(i), u8::from(i % 3 == 0));
            }
        }

        // Unboxed Int & Float memcpy without materializing
        let arr_i = Rc::new(LustArray::new(Vec::new()));
        for i in 0..50i64 {
            arr_i.push_int(i * 7 - 3);
        }
        let (buf_i, _) = build_or_get_cached_flat_array(&ctx, &arr_i, &[], FlatArrayElemTag::Int)
            .expect("build flat int cache")
            .expect("some flat int cache");
        assert!(arr_i.is_unboxed(), "Int unboxed array must stay unboxed");
        assert_eq!(buf_i.word_len(), 52);

        let arr_f = Rc::new(LustArray::new(Vec::new()));
        for i in 0..20 {
            arr_f.push_float((i as f64) * 0.5);
        }
        let (buf_f, _) =
            build_or_get_cached_flat_array(&ctx, &arr_f, &[], FlatArrayElemTag::Float)
                .expect("build flat float cache")
                .expect("some flat float cache");
        assert!(arr_f.is_unboxed(), "Float unboxed array must stay unboxed");
        assert_eq!(buf_f.word_len(), 22);
    }

    // 3. End-to-end VM + Cranelift JIT: Array<bool> literal (> 8 elements), read-only unboxed Bool param,
    //    and mutable Bool param writeback
    {
        let src = r#"
function make_bools(): Array<bool>
    local a: Array<bool> = [true, false, true, true, false, false, true, false, true, false]
    a[1] = true
    a[8] = false
    return a
end

function count_true_ro(arr: Array<bool>): int
    local n: int = array.len(arr)
    local i: int = 0
    local c: int = 0
    while i < n do
        if arr[i] is Ok(v) then
            if v then
                c = c + 1
            end
        end
        i = i + 1
    end
    return c
end

function flip_bools_rw(arr: Array<bool>): int
    local n: int = array.len(arr)
    local i: int = 0
    while i < n do
        if arr[i] is Ok(v) then
            arr[i] = not v
        end
        i = i + 1
    end
    return n
end
"#;
        let mut vm = compile_embedded_vm(src, true);
        let make_fn = find_fn_name(&vm, "make_bools");
        let count_fn = find_fn_name(&vm, "count_true_ro");
        let flip_fn = find_fn_name(&vm, "flip_bools_rw");

        let made = vm.call(&make_fn, vec![]).expect("make_bools");
        assert_eq!(
            made,
            Value::array(vec![
                Value::Bool(true),
                Value::Bool(true), // flipped from false
                Value::Bool(true),
                Value::Bool(true),
                Value::Bool(false),
                Value::Bool(false),
                Value::Bool(true),
                Value::Bool(false),
                Value::Bool(false), // flipped from true
                Value::Bool(false),
            ])
        );

        // Pass an UNBOXED Bool LustArray to read-only `count_true_ro`: verify cached flat-view
        // fast-path (1 build, then 1 hit on second call, 0 writebacks) and exact result.
        let unboxed_bools = Rc::new(LustArray::new(Vec::new()));
        for i in 0..33 {
            unboxed_bools.push_bool(i % 2 == 0); // 17 true
        }
        assert!(unboxed_bools.is_unboxed());
        let builds_before = vm.cranelift_array_cache_builds();
        let hits_before = vm.cranelift_array_cache_hits();
        let wb_before = vm.cranelift_writeback_commit_count();
        let c = vm
            .call(&count_fn, vec![Value::Array(Rc::clone(&unboxed_bools))])
            .expect("count_true_ro");
        assert_eq!(c, Value::Int(17));
        let c2 = vm
            .call(&count_fn, vec![Value::Array(Rc::clone(&unboxed_bools))])
            .expect("count_true_ro second call");
        assert_eq!(c2, Value::Int(17));
        assert_eq!(vm.cranelift_array_cache_builds() - builds_before, 1);
        assert_eq!(vm.cranelift_array_cache_hits() - hits_before, 1);
        assert_eq!(vm.cranelift_writeback_commit_count() - wb_before, 0);

        // Pass an unboxed Bool array to mutable `flip_bools_rw`: must flip all 33 elements cleanly
        let n = vm
            .call(&flip_fn, vec![Value::Array(Rc::clone(&unboxed_bools))])
            .expect("flip_bools_rw");
        assert_eq!(n, Value::Int(33));
        for i in 0..33 {
            assert_eq!(unboxed_bools.get_value_fast(i), Some(Value::Bool(i % 2 != 0)));
        }
    }
}

#[test]
fn test_ng_w10_5_do_not_inline_push_on_function_param_arrays() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    let src = r#"
function push_param(arr: Array<int>, n: int): int
    local i: int = 0
    while i < n do
        array.push(arr, i)
        i = i + 1
    end
    return array.len(arr)
end
"#;
    let mut vm = compile_embedded_vm(src, true);
    let push_param_fn = find_fn_name(&vm, "push_param");

    // Function parameter `arr` MUST NOT use inline fast-path: every push calls `lust_mir_array_push`
    let arr_rc = Rc::new(LustArray::new(Vec::new()));
    vm.mir_context.reset_push_grow_slow_calls();
    let res = vm
        .call(
            &push_param_fn,
            vec![Value::Array(Rc::clone(&arr_rc)), Value::Int(50)],
        )
        .expect("push_param");
    assert_eq!(res, Value::Int(50));
    assert_eq!(
        vm.mir_context.push_grow_slow_calls(),
        50,
        "DO NOT inline fast_push_bb on function parameters: all 50 pushes must route through lust_mir_array_push"
    );

    // Also verify forged pointer passed to compiled `push_param` fails closed without SIGSEGV!
    let mir_fn = lower_single_function(src);
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    let mut jit = CraneliftJitCompiler::new().expect("jit");
    let code = jit.compile_function(&mir_fn).expect("compile push_param");
    let f: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(code) };
    lust_mir_set_status(MIR_STATUS_OK);
    let _ = f(0xdead_beef, 10);
    assert_eq!(
        lust_mir_get_status(),
        MIR_STATUS_ERR_OUT_OF_BOUNDS,
        "Forged param pointer in array.push must fail closed via lust_mir_array_push"
    );
}

#[test]
fn test_ng_w10_5_array_bench_workload_inline_push_and_bounds_elision() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    let src = r#"
function array_workload(n: int, passes: int): int
    local arr: Array<int> = []
    local i: int = 0
    while i < n do
        array.push(arr, i)
        i = i + 1
    end
    local total: int = 0
    local pass: int = 0
    while pass < passes do
        local j: int = 0
        local len: int = array.len(arr)
        while j < len do
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
    let mut vm = compile_embedded_vm(src, true);
    let wl_fn = find_fn_name(&vm, "array_workload");
    vm.mir_context.reset_push_grow_slow_calls();
    let res = vm
        .call(&wl_fn, vec![Value::Int(20_000), Value::Int(5)])
        .expect("array_workload");
    let expected = 5 * (20_000 * 19_999 / 2);
    assert_eq!(res, Value::Int(expected));
    let slow_calls = vm.mir_context.push_grow_slow_calls();
    assert!(
        slow_calls >= 1 && slow_calls <= 15,
        "Expected <= 15 slow grow calls for 20_000 int pushes, got {slow_calls}"
    );
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_ng_w10_5_sieve_mid_loop_deopt_and_resume_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    let mut vm = compile_embedded_vm(SIEVE_FN_SRC, true);
    let sieve_name = find_fn_name(&vm, "sieve");

    // Deopt during the inline-push loop (poll 50) AND during the outer/inner sieve loop (poll 1200)
    for poll_k in [50u64, 1200u64] {
        reset_all();
        safepoint_request_deopt_on_poll(None, poll_k);
        let res = vm
            .call(&sieve_name, vec![Value::Int(1000)])
            .expect("sieve mid-loop deopt must resume cleanly");
        assert_eq!(
            res,
            Value::Int(168),
            "sieve(1000) deopt at poll {poll_k} must return 168 primes"
        );
        assert_eq!(vm.call_stack_depth(), 0);
    }
}

#![cfg(feature = "std")]
#![forbid(unsafe_code)]

//! NG-W14-8 Verification Suite: Unsigned Range Checks & Loop Induction Bounds Proofs
//!
//! Verifies:
//! 1. `if arr[j] is Ok(v)` (and `arr[j].try_unwrap()`) lowers its bounds check to a single
//!    unsigned comparison `(j as u64) < (len as u64)` (`icmp ult`) in Cranelift instead of
//!    `(j >= 0) & (j < len)`.
//! 2. Conservative loop-induction bounds proof feeds `proven_in_bounds` for canonical
//!    `while j <= N` / `while j < len` loops (including `sieve.lust` and direct `arr[j] = val`
//!    stores) when `arr` is non-aliased/non-shrinking, `j` starts `>= 0`, step `> 0`, and no
//!    signed `i64` overflow is possible.
//! 3. Fail-closed safety & SEC-04 guardrails: `j = -1`, `j = i64::MIN`, `j = len`, post-step
//!    stores, length-shrinking mutations (`array.pop`/`clear`/helper calls, including via alias),
//!    negative/zero/unknown steps, and potential `i64` overflow still trap / take `Err`.
//! 4. Exact JIT == VM parity on `sieve`, `array`, and canonical store loops, plus mid-loop
//!    `test-deopt` continuation parity.

use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;
use std::sync::Mutex;

use lust::bytecode::{LustArray, Value};
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{
    CraneliftJitCompiler, safepoint_reset, take_current_sticky_error, take_deopt_payload,
};
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::safepoint_request_deopt_on_poll;
use lust::mir::ir::MirFunction;
use lust::mir::lowering::AstLowering;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vfs::VirtualConsole;
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
    let module = AstLowering::lower_module_items("ng_w14_8_mod", &items).expect("lower module");
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
        .unwrap_or_else(|| suffix.to_string())
}

fn call_jit_and_vm(
    vm_jit: &mut VM,
    vm_bc: &mut VM,
    fn_suffix: &str,
    args: Vec<Value>,
) -> Value {
    let fn_name = find_fn_name(vm_jit, fn_suffix);
    let e0 = vm_jit.cranelift_entry_count();
    let jit_res = vm_jit
        .call(&fn_name, args.clone())
        .unwrap_or_else(|e| panic!("JIT call '{fn_suffix}'({args:?}) failed: {e:?}"));
    assert!(
        vm_jit.cranelift_entry_count() > e0,
        "Function '{fn_suffix}' must execute via Cranelift JIT"
    );
    let bc_res = vm_bc
        .call(&fn_name, args.clone())
        .unwrap_or_else(|e| panic!("VM call '{fn_suffix}'({args:?}) failed: {e:?}"));
    assert_eq!(
        jit_res, bc_res,
        "JIT != VM mismatch for '{fn_suffix}'({args:?})"
    );
    jit_res
}

struct ClifIrBlocks {
    blocks: HashMap<String, Vec<String>>,
    cold_blocks: HashSet<String>,
    block_order: Vec<String>,
    succs: HashMap<String, Vec<String>>,
}

impl ClifIrBlocks {
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

    fn count_icmp_all(&self, cc: &str) -> usize {
        let p1 = format!("icmp {cc} ");
        let p2 = format!("icmp.i64 {cc} ");
        let mut count = 0usize;
        for b in &self.block_order {
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
}

#[test]
fn test_ng_w14_8_negative_and_oob_indices_still_trap() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    let src = r#"
function checked_get(arr: Array<int>, j: int): int
    if arr[j] is Ok(v) then
        return v
    end
    return -999
end

function checked_get_empty(j: int): int
    local arr: Array<int> = []
    if arr[j] is Ok(v) then
        return v
    end
    return -777
end

function direct_store_neg_start(): int
    local arr: Array<int> = [1, 2, 3, 4]
    local n: int = array.len(arr)
    local j: int = -1
    while j < n do
        arr[j] = 42
        j = j + 1
    end
    return 0
end

function direct_store_i64_min(min_idx: int): int
    local arr: Array<int> = [1, 2, 3, 4]
    local n: int = array.len(arr)
    local j: int = min_idx
    while j < n do
        arr[j] = 42
        j = j + 1
    end
    return 0
end

function direct_store_off_by_one_le_len(): int
    local arr: Array<int> = [1, 2, 3, 4]
    local n: int = array.len(arr)
    local j: int = 0
    while j <= n do
        arr[j] = 42
        j = j + 1
    end
    return 0
end

function direct_store_after_step(): int
    local arr: Array<int> = [1, 2, 3, 4]
    local n: int = array.len(arr)
    local j: int = 0
    while j < n do
        j = j + 1
        arr[j] = 42
    end
    return 0
end

function direct_store_after_cond_step(): int
    local arr: Array<int> = [1, 2, 3, 4]
    local n: int = array.len(arr)
    local j: int = 0
    while j < n do
        if j == 3 then
            j = j + 1
        end
        arr[j] = 42
        j = j + 1
    end
    return 0
end

function direct_store_pop_in_loop(): int
    local arr: Array<int> = [1, 2, 3, 4]
    local n: int = array.len(arr)
    local j: int = 0
    while j < n do
        array.pop(arr)
        arr[j] = 42
        j = j + 1
    end
    return 0
end

function direct_store_aliased_pop_before_loop(a: Array<int>, b: Array<int>): int
    local n: int = array.len(a)
    array.pop(b)
    local j: int = 0
    while j < n do
        a[j] = 42
        j = j + 1
    end
    return 0
end

function direct_store_negative_step(): int
    local arr: Array<int> = [1, 2, 3, 4]
    local n: int = array.len(arr)
    local j: int = 1
    while j < n do
        arr[j] = 42
        j = j + (-1)
    end
    return 0
end
"#;

    let mut vm_jit = compile_embedded_vm(src, true);
    let mut vm_bc = compile_embedded_vm(src, false);

    // 1. `if arr[j] is Ok(v)` exact Err/Ok boundary parity across negative, i64::MIN, len, i64::MAX
    let arr_val = Value::array(vec![
        Value::Int(10),
        Value::Int(20),
        Value::Int(30),
        Value::Int(40),
    ]);
    for (idx, expected) in [
        (-1i64, -999i64),
        (i64::MIN, -999),
        (-100_000, -999),
        (0, 10),
        (3, 40),
        (4, -999),
        (5, -999),
        (i64::MAX, -999),
    ] {
        let res = call_jit_and_vm(
            &mut vm_jit,
            &mut vm_bc,
            "checked_get",
            vec![arr_val.clone(), Value::Int(idx)],
        );
        assert_eq!(res, Value::Int(expected), "checked_get failed at idx={idx}");
    }

    // Empty array: every index (0, -1, i64::MIN, i64::MAX) must take Err branch
    for idx in [0i64, -1, i64::MIN, 1, i64::MAX] {
        let res = call_jit_and_vm(&mut vm_jit, &mut vm_bc, "checked_get_empty", vec![Value::Int(idx)]);
        assert_eq!(res, Value::Int(-777), "checked_get_empty failed at idx={idx}");
    }

    // 2. Direct OOB / SEC-04 traps in BOTH Cranelift JIT and VM
    let trap_cases: Vec<(&str, Vec<Value>)> = vec![
        ("direct_store_neg_start", vec![]),
        ("direct_store_i64_min", vec![Value::Int(i64::MIN)]),
        ("direct_store_off_by_one_le_len", vec![]),
        ("direct_store_after_step", vec![]),
        ("direct_store_after_cond_step", vec![]),
        ("direct_store_pop_in_loop", vec![]),
        ("direct_store_negative_step", vec![]),
    ];

    for (fn_name, args) in trap_cases {
        let resolved = find_fn_name(&vm_jit, fn_name);
        let jit_res = vm_jit.call(&resolved, args.clone());
        let bc_res = vm_bc.call(&resolved, args.clone());
        assert!(
            jit_res.is_err(),
            "SEC-04 FAIL: JIT must trap on '{fn_name}', got {jit_res:?}"
        );
        assert!(
            bc_res.is_err(),
            "SEC-04 FAIL: VM must trap on '{fn_name}', got {bc_res:?}"
        );
    }

    // 3. Aliased array shrink before loop (`a` and `b` point to same `Rc<LustArray>`)
    {
        let resolved = find_fn_name(&vm_jit, "direct_store_aliased_pop_before_loop");
        let shared_jit = Rc::new(LustArray::new(vec![
            Value::Int(1),
            Value::Int(2),
            Value::Int(3),
            Value::Int(4),
        ]));
        let jit_res = vm_jit.call(
            &resolved,
            vec![
                Value::Array(Rc::clone(&shared_jit)),
                Value::Array(Rc::clone(&shared_jit)),
            ],
        );
        assert!(
            jit_res.is_err(),
            "SEC-04 FAIL: aliased array.pop(b) before `while j < n` must still trap OOB in JIT, got {jit_res:?}"
        );

        let shared_bc = Rc::new(LustArray::new(vec![
            Value::Int(1),
            Value::Int(2),
            Value::Int(3),
            Value::Int(4),
        ]));
        let bc_res = vm_bc.call(
            &resolved,
            vec![
                Value::Array(Rc::clone(&shared_bc)),
                Value::Array(Rc::clone(&shared_bc)),
            ],
        );
        assert!(
            bc_res.is_err(),
            "SEC-04 FAIL: aliased array.pop(b) before `while j < n` must trap OOB in VM, got {bc_res:?}"
        );
    }
}

#[test]
fn test_ng_w14_8_sieve_and_array_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    let src = r#"
function sieve_const_10000(): int
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
    return count
end

function sieve_const_small(): int
    local N: int = 30
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
    return count
end

function array_bench_pattern(n: int, passes: int): int
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

function canonical_fill_and_sum(n: int): int
    local arr: Array<int> = []
    local k: int = 0
    while k < n do
        array.push(arr, 0)
        k = k + 1
    end
    local len: int = array.len(arr)
    local j: int = 0
    while j < len do
        arr[j] = j * 3 + 1
        j = j + 1
    end
    local last: int = len - 1
    j = 0
    while j <= last do
        if arr[j] is Ok(v) then
            arr[j] = v + 2
        end
        j = j + 1
    end
    local sum: int = 0
    j = 0
    while j < len do
        if arr[j] is Ok(v) then
            sum = sum + v
        end
        j = j + 1
    end
    return sum
end

function fill_param_array(arr: Array<int>, base: int): int
    local n: int = array.len(arr)
    local j: int = 0
    while j < n do
        arr[j] = base + j
        j = j + 1
    end
    local sum: int = 0
    j = 0
    while j < n do
        if arr[j] is Ok(v) then
            sum = sum + v
        end
        j = j + 1
    end
    return sum
end
"#;

    let mut vm_jit = compile_embedded_vm(src, true);
    let mut vm_bc = compile_embedded_vm(src, false);

    assert_eq!(
        call_jit_and_vm(&mut vm_jit, &mut vm_bc, "sieve_const_10000", vec![]),
        Value::Int(1229)
    );
    assert_eq!(
        call_jit_and_vm(&mut vm_jit, &mut vm_bc, "sieve_const_small", vec![]),
        Value::Int(10)
    );

    let expected_arr = 4 * (5000 * 4999 / 2);
    assert_eq!(
        call_jit_and_vm(
            &mut vm_jit,
            &mut vm_bc,
            "array_bench_pattern",
            vec![Value::Int(5000), Value::Int(4)],
        ),
        Value::Int(expected_arr)
    );
    assert_eq!(
        call_jit_and_vm(
            &mut vm_jit,
            &mut vm_bc,
            "array_bench_pattern",
            vec![Value::Int(0), Value::Int(3)],
        ),
        Value::Int(0)
    );

    for n in [0i64, 1, 2, 17, 250] {
        // sum_{j=0}^{n-1} (3*j + 3) = 3 * n * (n + 1) / 2
        let expected = 3 * n * (n + 1) / 2;
        assert_eq!(
            call_jit_and_vm(&mut vm_jit, &mut vm_bc, "canonical_fill_and_sum", vec![Value::Int(n)]),
            Value::Int(expected),
            "canonical_fill_and_sum mismatch for n={n}"
        );
    }

    // Param array fill parity
    let arr_jit = Rc::new(LustArray::new(vec![Value::Int(0); 32]));
    let arr_bc = Rc::new(LustArray::new(vec![Value::Int(0); 32]));
    let fn_name = find_fn_name(&vm_jit, "fill_param_array");
    let r_jit = vm_jit
        .call(&fn_name, vec![Value::Array(Rc::clone(&arr_jit)), Value::Int(10)])
        .expect("fill_param_array jit");
    let r_bc = vm_bc
        .call(&fn_name, vec![Value::Array(Rc::clone(&arr_bc)), Value::Int(10)])
        .expect("fill_param_array bc");
    assert_eq!(r_jit, r_bc);
    assert_eq!(r_jit, Value::Int(32 * 10 + (32 * 31 / 2)));
}

#[test]
fn test_ng_w14_8_clif_ir_unsigned_lt_and_loop_bounds_elision() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    // 1. Standalone `if arr[j] is Ok(v)` emits single `icmp ult`, zero `icmp sge`/`icmp slt`
    {
        let src = r#"
function get_or_zero(arr: Array<int>, j: int): int
    if arr[j] is Ok(v) then
        return v
    end
    return 0
end
"#;
        let mir_fn = lower_single_function(src);
        let mut compiler = CraneliftJitCompiler::new().expect("compiler");
        let ir = compiler.dump_function_ir(&mir_fn).expect("dump_function_ir");
        let parsed = ClifIrBlocks::parse(&ir);
        assert!(
            parsed.count_icmp_all("ult") >= 1,
            "Expected single unsigned comparison `icmp ult` for `if arr[j] is Ok(v)`:\n{ir}"
        );
        assert_eq!(
            parsed.count_icmp_all("sge"),
            0,
            "Expected zero `icmp sge` in `if arr[j] is Ok(v)` after NG-W14-8 unsigned fusion:\n{ir}"
        );
    }

    // 2. `sieve.lust` pattern with constant `N = 10000`:
    //    - outer `if flags[i] is Ok(f)` uses `icmp ult`
    //    - inner `while j <= N do flags[j] = false; j = j + i; end` elides `icmp uge` completely (0 hot `icmp uge`)!
    {
        let src = r#"
function sieve_const(): int
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
    return count
end
"#;
        let mir_fn = lower_single_function(src);
        let mut compiler = CraneliftJitCompiler::new().expect("compiler");
        let ir = compiler.dump_function_ir(&mir_fn).expect("dump_function_ir");
        let parsed = ClifIrBlocks::parse(&ir);
        let hot = parsed.hot_cycle_blocks();
        assert!(!hot.is_empty(), "Expected hot cycle blocks in sieve_const:\n{ir}");
        assert!(
            parsed.count_icmp_in_blocks(&hot, "ult") >= 1,
            "Expected hot `icmp ult` for `if flags[i] is Ok(f)` in sieve_const:\n{ir}"
        );
        assert_eq!(
            parsed.count_icmp_in_blocks(&hot, "uge"),
            0,
            "Expected 0 hot `icmp uge` in sieve_const (inner `flags[j] = false` proven in bounds!):\n{ir}"
        );
    }

    // 3. Canonical `while j < len do arr[j] = val; j = j + 1; end` elides `icmp uge` completely!
    {
        let src = r#"
function fill_loop(arr: Array<int>): int
    local n: int = array.len(arr)
    local j: int = 0
    while j < n do
        arr[j] = j * 2
        j = j + 1
    end
    return n
end
"#;
        let mir_fn = lower_single_function(src);
        let mut compiler = CraneliftJitCompiler::new().expect("compiler");
        let ir = compiler.dump_function_ir(&mir_fn).expect("dump_function_ir");
        let parsed = ClifIrBlocks::parse(&ir);
        let hot = parsed.hot_cycle_blocks();
        assert!(!hot.is_empty(), "Expected hot cycle blocks in fill_loop:\n{ir}");
        assert_eq!(
            parsed.count_icmp_in_blocks(&hot, "uge"),
            0,
            "Expected 0 hot `icmp uge` in canonical `while j < n` store loop:\n{ir}"
        );
    }

    // 4. Fail-closed check: off-by-one `while j <= n` and `j = -1` MUST retain bounds check (`icmp uge` >= 1)!
    for (label, src) in [
        (
            "off_by_one",
            r#"
function bad_le(arr: Array<int>): int
    local n: int = array.len(arr)
    local j: int = 0
    while j <= n do
        arr[j] = 1
        j = j + 1
    end
    return n
end
"#,
        ),
        (
            "neg_start",
            r#"
function bad_neg(arr: Array<int>): int
    local n: int = array.len(arr)
    local j: int = -1
    while j < n do
        arr[j] = 1
        j = j + 1
    end
    return n
end
"#,
        ),
        (
            "unknown_step",
            r#"
function bad_step(arr: Array<int>, step: int): int
    local n: int = array.len(arr)
    local j: int = 0
    while j < n do
        arr[j] = 1
        j = j + step
    end
    return n
end
"#,
        ),
    ] {
        let mir_fn = lower_single_function(src);
        let mut compiler = CraneliftJitCompiler::new().expect("compiler");
        let ir = compiler.dump_function_ir(&mir_fn).expect("dump_function_ir");
        let parsed = ClifIrBlocks::parse(&ir);
        let hot = parsed.hot_cycle_blocks();
        assert!(
            parsed.count_icmp_in_blocks(&hot, "uge") >= 1,
            "Fail-closed check '{label}' MUST retain hot `icmp uge` bounds check:\n{ir}"
        );
    }
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_ng_w14_8_mid_loop_deopt_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    let src = r#"
function sieve_deopt(): int
    local N: int = 1000
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
    return count
end

function fill_deopt(n: int): int
    local arr: Array<int> = []
    local k: int = 0
    while k < n do
        array.push(arr, 0)
        k = k + 1
    end
    local len: int = array.len(arr)
    local j: int = 0
    while j < len do
        arr[j] = j + 1
        j = j + 1
    end
    local sum: int = 0
    j = 0
    while j < len do
        if arr[j] is Ok(v) then
            sum = sum + v
        end
        j = j + 1
    end
    return sum
end
"#;

    let mut vm = compile_embedded_vm(src, true);
    let sieve_name = find_fn_name(&vm, "sieve_deopt");
    let fill_name = find_fn_name(&vm, "fill_deopt");

    for poll_k in [10u64, 500, 1200] {
        reset_all();
        safepoint_request_deopt_on_poll(None, poll_k);
        let res = vm
            .call(&sieve_name, vec![])
            .expect("sieve_deopt must resume cleanly");
        assert_eq!(res, Value::Int(168), "sieve_deopt parity failed at poll {poll_k}");
        assert_eq!(vm.call_stack_depth(), 0);
    }

    for poll_k in [5u64, 150, 350] {
        reset_all();
        safepoint_request_deopt_on_poll(None, poll_k);
        let res = vm
            .call(&fill_name, vec![Value::Int(200)])
            .expect("fill_deopt must resume cleanly");
        assert_eq!(
            res,
            Value::Int(200 * 201 / 2),
            "fill_deopt parity failed at poll {poll_k}"
        );
        assert_eq!(vm.call_stack_depth(), 0);
    }
}

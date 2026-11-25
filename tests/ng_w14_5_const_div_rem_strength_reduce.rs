#![cfg(feature = "std")]
#![forbid(unsafe_code)]

//! NG-W14-5 Verification Suite: Constant Signed `sdiv`/`srem` Strength Reduction
//!
//! Verifies:
//! 1. Exact Lust/Rust truncated-toward-zero signed division/remainder parity (JIT == VM == Rust)
//!    over boundary dividends `[i64::MIN, ..., -100, -7, -2, -1, 0, 1, 2, 7, 100, ..., i64::MAX]`
//!    and divisors `[-7, -4, -3, -2, -1, 0, 1, 2, 3, 4, 7, 8, -8, 16, -16, 2^62, -2^62, i64::MIN, i64::MAX]`,
//!    across compile-time constants, function-wide constant locals across loops, and dynamic operands.
//! 2. Exact `rhs == 0` runtime error (`MIR_STATUS_ERR_DIV_BY_ZERO`) and exact `i64::MIN / -1` /
//!    `i64::MIN % -1` non-trapping wrapping semantics in an isolated child process with watchdog.
//! 3. Pre-opt CLIF IR verification that power-of-two constants (`±2^k`, `1`, `i64::MIN`) emit
//!    branchless shift/mask arithmetic (zero `sdiv`/`srem`), constant locals across loops elide
//!    the `rhs == 0` and `rhs == -1` guard blocks, and `v % ±2^k == 0` folds to `(v & (2^k - 1)) == 0`.
//! 4. Exact `test-deopt` mid-loop continuation parity on constant-mod loops.

use std::collections::{HashMap, HashSet, VecDeque};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use lust::bytecode::compiler::Compiler;
use lust::bytecode::Value;
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{safepoint_reset, CraneliftJitCompiler};
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::safepoint_request_deopt_on_poll;
use lust::mir::ir::{MirFunction, MirModule};
use lust::mir::lowering::AstLowering;
use lust::mir::opt::MirOptimizer;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{VmBackend, VM};

static TEST_MUTEX: Mutex<()> = Mutex::new(());
const CHILD_ENV_KEY: &str = "LUST_NG_W14_5_CHILD_WORKER";

fn lower_module_from_source(src: &str) -> (MirModule, Vec<lust::ast::Item>) {
    let mut interner = Interner::new();
    let tokens = Lexer::new(src, &mut interner)
        .tokenize()
        .expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");
    let mut module = MirModule::new("ng_w14_5".into());
    for item in &items {
        if let lust::ast::ItemKind::Function(func) = &item.kind {
            let mut mir_fn = AstLowering::lower_function(func).expect("lower_function");
            MirOptimizer::optimize_function(&mut mir_fn);
            module.add_function(mir_fn);
        }
    }
    (module, items)
}

fn lower_single_function(src: &str) -> MirFunction {
    let (module, _) = lower_module_from_source(src);
    module
        .functions
        .into_iter()
        .next()
        .expect("expected at least one function")
}

fn create_vms(src: &str) -> (VM, VM) {
    let (module, items) = lower_module_from_source(src);

    let mut vm_clif = VM::new();
    vm_clif.load_mir_module(&module).expect("load_mir_module");
    vm_clif.set_backend(VmBackend::Cranelift);

    let mut vm_bc = VM::new();
    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let struct_defs = checker.struct_definitions();
    vm_clif.register_structs(&struct_defs);
    vm_bc.register_structs(&struct_defs);

    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    if let Ok(bc_functions) = compiler.compile_module(&items) {
        vm_bc.load_functions(bc_functions);
    } else {
        vm_bc.load_mir_module(&module).expect("load_mir_module bc");
    }
    vm_bc.set_backend(VmBackend::Bytecode);

    (vm_clif, vm_bc)
}

struct ClifIrSummary {
    reachable_lines: Vec<String>,
    reachable_calls: Vec<String>,
}

fn analyze_clif_ir(ir: &str) -> ClifIrSummary {
    let mut sym_map: HashMap<String, String> = HashMap::new();
    let mut blocks: HashMap<String, Vec<String>> = HashMap::new();
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

    let extract_call_sym = |line: &str| -> Option<String> {
        let idx = line.find("call ")?;
        let after = &line[idx + 5..];
        let callee_ref = after
            .split(|c: char| c == '(' || c.is_whitespace())
            .next()?
            .trim();
        Some(
            sym_map
                .get(callee_ref)
                .cloned()
                .unwrap_or_else(|| callee_ref.to_string()),
        )
    };

    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<String> = VecDeque::new();
    if let Some(entry) = block_order.first() {
        visited.insert(entry.clone());
        queue.push_back(entry.clone());
    }
    let mut reachable_lines = Vec::new();
    let mut reachable_calls = Vec::new();

    while let Some(b) = queue.pop_front() {
        if let Some(lines) = blocks.get(&b) {
            for l in lines {
                reachable_lines.push(l.clone());
                if let Some(sym) = extract_call_sym(l) {
                    reachable_calls.push(sym);
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

    ClifIrSummary {
        reachable_lines,
        reachable_calls,
    }
}

fn has_sdiv(lines: &[String]) -> bool {
    lines
        .iter()
        .any(|l| l.contains(" sdiv ") || l.contains("sdiv."))
}

fn has_srem(lines: &[String]) -> bool {
    lines
        .iter()
        .any(|l| l.contains(" srem ") || l.contains("srem."))
}

#[test]
fn test_ng_w14_5_signed_mod_div_edge_cases() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let src = r#"
        function div_c_neg7(x: int): int return x / -7 end
        function div_c_neg4(x: int): int return x / -4 end
        function div_c_neg3(x: int): int return x / -3 end
        function div_c_neg2(x: int): int return x / -2 end
        function div_c_neg1(x: int): int return x / -1 end
        function div_c_0(x: int): int return x / 0 end
        function div_c_1(x: int): int return x / 1 end
        function div_c_2(x: int): int return x / 2 end
        function div_c_3(x: int): int return x / 3 end
        function div_c_4(x: int): int return x / 4 end
        function div_c_7(x: int): int return x / 7 end
        function div_c_8(x: int): int return x / 8 end
        function div_c_neg8(x: int): int return x / -8 end
        function div_c_16(x: int): int return x / 16 end
        function div_c_neg16(x: int): int return x / -16 end
        function div_c_p62(x: int): int return x / 4611686018427387904 end
        function div_c_neg_p62(x: int): int return x / -4611686018427387904 end
        function div_c_min(x: int): int return x / -9223372036854775808 end
        function div_c_max(x: int): int return x / 9223372036854775807 end

        function mod_c_neg7(x: int): int return x % -7 end
        function mod_c_neg4(x: int): int return x % -4 end
        function mod_c_neg3(x: int): int return x % -3 end
        function mod_c_neg2(x: int): int return x % -2 end
        function mod_c_neg1(x: int): int return x % -1 end
        function mod_c_0(x: int): int return x % 0 end
        function mod_c_1(x: int): int return x % 1 end
        function mod_c_2(x: int): int return x % 2 end
        function mod_c_3(x: int): int return x % 3 end
        function mod_c_4(x: int): int return x % 4 end
        function mod_c_7(x: int): int return x % 7 end
        function mod_c_8(x: int): int return x % 8 end
        function mod_c_neg8(x: int): int return x % -8 end
        function mod_c_16(x: int): int return x % 16 end
        function mod_c_neg16(x: int): int return x % -16 end
        function mod_c_p62(x: int): int return x % 4611686018427387904 end
        function mod_c_neg_p62(x: int): int return x % -4611686018427387904 end
        function mod_c_min(x: int): int return x % -9223372036854775808 end
        function mod_c_max(x: int): int return x % 9223372036854775807 end

        function div_dyn(a: int, b: int): int return a / b end
        function mod_dyn(a: int, b: int): int return a % b end

        function mod_local_loop_7(x: int): int
            local d: int = 7
            local i: int = 0
            local acc: int = 0
            while i < 1 do
                acc = x % d
                i = i + 1
            end
            return acc
        end

        function div_local_loop_neg2(x: int): int
            local d: int = -2
            local i: int = 0
            local acc: int = 0
            while i < 1 do
                acc = x / d
                i = i + 1
            end
            return acc
        end

        function mod_local_loop_0(x: int): int
            local d: int = 0
            local i: int = 0
            local acc: int = 0
            while i < 1 do
                acc = x % d
                i = i + 1
            end
            return acc
        end

        function div_local_loop_neg1(x: int): int
            local d: int = -1
            local i: int = 0
            local acc: int = 0
            while i < 1 do
                acc = x / d
                i = i + 1
            end
            return acc
        end

        function is_div_2(x: int): bool return x % 2 == 0 end
        function is_not_div_2(x: int): bool return x % 2 != 0 end
        function is_div_neg2(x: int): bool return x % -2 == 0 end
        function is_div_4(x: int): bool return x % 4 == 0 end
        function is_not_div_4(x: int): bool return x % 4 != 0 end
        function is_div_neg8(x: int): bool return x % -8 == 0 end
        function is_div_min(x: int): bool return x % -9223372036854775808 == 0 end
        function is_div_2_zero_lhs(x: int): bool return 0 == x % 2 end

        function mod2_mut_after_soundness(x: int): bool
            local r: int = x % 2
            x = x + 1
            return r == 0
        end

        function mod2_reused_rem(x: int): int
            local r: int = x % 2
            if r == 0 then
                return r + 100
            end
            return r * 10
        end

        function nested_mod7_kernel(n: int): int
            local total: int = 0
            local i: int = 0
            while i < n do
                local j: int = 0
                while j < n do
                    total = total + (i * j) % 7
                    j = j + 1
                end
                i = i + 1
            end
            return total
        end

        function count_even_loop(n: int): int
            local c: int = 0
            local i: int = -n
            while i <= n do
                if i % 2 == 0 then
                    c = c + 1
                end
                i = i + 1
            end
            return c
        end
    "#;

    let (mut vm_clif, mut vm_bc) = create_vms(src);

    let dividends: [i64; 21] = [
        i64::MIN,
        i64::MIN + 1,
        -100,
        -9,
        -8,
        -7,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        7,
        8,
        9,
        100,
        i64::MAX - 1,
        i64::MAX,
    ];

    let const_divisors: [(&str, &str, i64); 19] = [
        ("div_c_neg7", "mod_c_neg7", -7),
        ("div_c_neg4", "mod_c_neg4", -4),
        ("div_c_neg3", "mod_c_neg3", -3),
        ("div_c_neg2", "mod_c_neg2", -2),
        ("div_c_neg1", "mod_c_neg1", -1),
        ("div_c_0", "mod_c_0", 0),
        ("div_c_1", "mod_c_1", 1),
        ("div_c_2", "mod_c_2", 2),
        ("div_c_3", "mod_c_3", 3),
        ("div_c_4", "mod_c_4", 4),
        ("div_c_7", "mod_c_7", 7),
        ("div_c_8", "mod_c_8", 8),
        ("div_c_neg8", "mod_c_neg8", -8),
        ("div_c_16", "mod_c_16", 16),
        ("div_c_neg16", "mod_c_neg16", -16),
        ("div_c_p62", "mod_c_p62", 1i64 << 62),
        ("div_c_neg_p62", "mod_c_neg_p62", -(1i64 << 62)),
        ("div_c_min", "mod_c_min", i64::MIN),
        ("div_c_max", "mod_c_max", i64::MAX),
    ];

    for &lhs in &dividends {
        for &(div_fn, mod_fn, rhs) in &const_divisors {
            if rhs == 0 {
                let e_before = vm_clif.cranelift_entry_count;
                assert!(
                    vm_clif.call(div_fn, vec![Value::Int(lhs)]).is_err(),
                    "JIT {div_fn}({lhs}) must trap on div-by-zero"
                );
                assert!(vm_clif.cranelift_entry_count > e_before);
                assert!(
                    vm_bc.call(div_fn, vec![Value::Int(lhs)]).is_err(),
                    "VM {div_fn}({lhs}) must trap on div-by-zero"
                );

                let e_before = vm_clif.cranelift_entry_count;
                assert!(
                    vm_clif.call(mod_fn, vec![Value::Int(lhs)]).is_err(),
                    "JIT {mod_fn}({lhs}) must trap on mod-by-zero"
                );
                assert!(vm_clif.cranelift_entry_count > e_before);
                assert!(
                    vm_bc.call(mod_fn, vec![Value::Int(lhs)]).is_err(),
                    "VM {mod_fn}({lhs}) must trap on mod-by-zero"
                );

                // Dynamic divisor 0
                assert!(vm_clif.call("div_dyn", vec![Value::Int(lhs), Value::Int(0)]).is_err());
                assert!(vm_bc.call("div_dyn", vec![Value::Int(lhs), Value::Int(0)]).is_err());
                assert!(vm_clif.call("mod_dyn", vec![Value::Int(lhs), Value::Int(0)]).is_err());
                assert!(vm_bc.call("mod_dyn", vec![Value::Int(lhs), Value::Int(0)]).is_err());
                continue;
            }

            let expected_div = lhs.wrapping_div(rhs);
            let expected_rem = lhs.wrapping_rem(rhs);
            if !(lhs == i64::MIN && rhs == -1) {
                assert_eq!(lhs.checked_div(rhs), Some(expected_div));
                assert_eq!(lhs.checked_rem(rhs), Some(expected_rem));
            }

            // Constant div
            let e_before = vm_clif.cranelift_entry_count;
            let jit_div = vm_clif
                .call(div_fn, vec![Value::Int(lhs)])
                .unwrap_or_else(|e| panic!("JIT {div_fn}({lhs}) failed: {e}"));
            assert!(vm_clif.cranelift_entry_count > e_before);
            let bc_div = vm_bc
                .call(div_fn, vec![Value::Int(lhs)])
                .unwrap_or_else(|e| panic!("VM {div_fn}({lhs}) failed: {e}"));
            assert_eq!(jit_div, Value::Int(expected_div), "JIT {div_fn}({lhs})");
            assert_eq!(jit_div, bc_div, "JIT vs VM parity {div_fn}({lhs})");

            // Constant mod
            let e_before = vm_clif.cranelift_entry_count;
            let jit_mod = vm_clif
                .call(mod_fn, vec![Value::Int(lhs)])
                .unwrap_or_else(|e| panic!("JIT {mod_fn}({lhs}) failed: {e}"));
            assert!(vm_clif.cranelift_entry_count > e_before);
            let bc_mod = vm_bc
                .call(mod_fn, vec![Value::Int(lhs)])
                .unwrap_or_else(|e| panic!("VM {mod_fn}({lhs}) failed: {e}"));
            assert_eq!(jit_mod, Value::Int(expected_rem), "JIT {mod_fn}({lhs})");
            assert_eq!(jit_mod, bc_mod, "JIT vs VM parity {mod_fn}({lhs})");

            // Dynamic div & mod
            let jit_dyn_div = vm_clif
                .call("div_dyn", vec![Value::Int(lhs), Value::Int(rhs)])
                .expect("JIT div_dyn");
            let bc_dyn_div = vm_bc
                .call("div_dyn", vec![Value::Int(lhs), Value::Int(rhs)])
                .expect("VM div_dyn");
            assert_eq!(jit_dyn_div, Value::Int(expected_div));
            assert_eq!(jit_dyn_div, bc_dyn_div);

            let jit_dyn_mod = vm_clif
                .call("mod_dyn", vec![Value::Int(lhs), Value::Int(rhs)])
                .expect("JIT mod_dyn");
            let bc_dyn_mod = vm_bc
                .call("mod_dyn", vec![Value::Int(lhs), Value::Int(rhs)])
                .expect("VM mod_dyn");
            assert_eq!(jit_dyn_mod, Value::Int(expected_rem));
            assert_eq!(jit_dyn_mod, bc_dyn_mod);
        }

        // Cross-block constant-local loops
        let j7 = vm_clif.call("mod_local_loop_7", vec![Value::Int(lhs)]).unwrap();
        let b7 = vm_bc.call("mod_local_loop_7", vec![Value::Int(lhs)]).unwrap();
        assert_eq!(j7, Value::Int(lhs.wrapping_rem(7)));
        assert_eq!(j7, b7);

        let jn2 = vm_clif.call("div_local_loop_neg2", vec![Value::Int(lhs)]).unwrap();
        let bn2 = vm_bc.call("div_local_loop_neg2", vec![Value::Int(lhs)]).unwrap();
        assert_eq!(jn2, Value::Int(lhs.wrapping_div(-2)));
        assert_eq!(jn2, bn2);

        let jn1 = vm_clif.call("div_local_loop_neg1", vec![Value::Int(lhs)]).unwrap();
        let bn1 = vm_bc.call("div_local_loop_neg1", vec![Value::Int(lhs)]).unwrap();
        assert_eq!(jn1, Value::Int(lhs.wrapping_div(-1)));
        assert_eq!(jn1, bn1);

        assert!(vm_clif.call("mod_local_loop_0", vec![Value::Int(lhs)]).is_err());
        assert!(vm_bc.call("mod_local_loop_0", vec![Value::Int(lhs)]).is_err());

        // Power-of-two divisibility peephole cases
        let div_checks: [(&str, bool); 9] = [
            ("is_div_2", lhs.wrapping_rem(2) == 0),
            ("is_not_div_2", lhs.wrapping_rem(2) != 0),
            ("is_div_neg2", lhs.wrapping_rem(-2) == 0),
            ("is_div_4", lhs.wrapping_rem(4) == 0),
            ("is_not_div_4", lhs.wrapping_rem(4) != 0),
            ("is_div_neg8", lhs.wrapping_rem(-8) == 0),
            ("is_div_min", lhs.wrapping_rem(i64::MIN) == 0),
            ("is_div_2_zero_lhs", lhs.wrapping_rem(2) == 0),
            ("mod2_mut_after_soundness", lhs.wrapping_rem(2) == 0),
        ];
        for (fname, expected) in div_checks {
            let jr = vm_clif.call(fname, vec![Value::Int(lhs)]).unwrap();
            let br = vm_bc.call(fname, vec![Value::Int(lhs)]).unwrap();
            assert_eq!(jr, Value::Bool(expected), "JIT {fname}({lhs})");
            assert_eq!(jr, br, "VM parity {fname}({lhs})");
        }

        let jr_reused = vm_clif.call("mod2_reused_rem", vec![Value::Int(lhs)]).unwrap();
        let br_reused = vm_bc.call("mod2_reused_rem", vec![Value::Int(lhs)]).unwrap();
        let expected_reused = if lhs.wrapping_rem(2) == 0 {
            100
        } else {
            lhs.wrapping_rem(2) * 10
        };
        assert_eq!(jr_reused, Value::Int(expected_reused));
        assert_eq!(jr_reused, br_reused);
    }

    // Kernel parity: nested `% 7` and `tree` `v % 2 == 0`
    let jit_nested = vm_clif.call("nested_mod7_kernel", vec![Value::Int(200)]).unwrap();
    let bc_nested = vm_bc.call("nested_mod7_kernel", vec![Value::Int(200)]).unwrap();
    assert_eq!(jit_nested, bc_nested);

    let jit_even = vm_clif.call("count_even_loop", vec![Value::Int(500)]).unwrap();
    let bc_even = vm_bc.call("count_even_loop", vec![Value::Int(500)]).unwrap();
    assert_eq!(jit_even, Value::Int(501));
    assert_eq!(jit_even, bc_even);
}

#[test]
fn test_ng_w14_5_ir_strength_reduction_and_elision() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let mut compiler = CraneliftJitCompiler::new().expect("create compiler");

    // 1. Power-of-two, ±1, and i64::MIN divisors must lower directly to branchless integer ops
    //    (zero sdiv / srem in pre-opt CLIF IR, zero status/helper calls).
    let pow2_cases = [
        "function mod2(x: int): int return x % 2 end",
        "function mod_neg2(x: int): int return x % -2 end",
        "function div2(x: int): int return x / 2 end",
        "function div_neg2(x: int): int return x / -2 end",
        "function mod4(x: int): int return x % 4 end",
        "function div4(x: int): int return x / 4 end",
        "function mod1(x: int): int return x % 1 end",
        "function div1(x: int): int return x / 1 end",
        "function mod_min(x: int): int return x % -9223372036854775808 end",
        "function div_min(x: int): int return x / -9223372036854775808 end",
    ];
    for src in pow2_cases {
        let mir_fn = lower_single_function(src);
        let ir = compiler.dump_function_ir(&mir_fn).expect("dump_function_ir");
        let summary = analyze_clif_ir(&ir);
        assert!(
            !has_sdiv(&summary.reachable_lines) && !has_srem(&summary.reachable_lines),
            "Expected branchless strength-reduced IR (zero sdiv/srem) for `{src}`:\n{ir}"
        );
        assert!(
            !summary
                .reachable_calls
                .iter()
                .any(|c| c == "lust_mir_set_status" || c == "lust_mir_div_int" || c == "lust_mir_rem_int"),
            "Unexpected runtime div/status call for `{src}`:\n{ir}"
        );
    }

    // 2. Constant local across loop (`local d: int = 7`) must elide both `rhs == 0` and `rhs == -1`
    //    runtime branches (zero `lust_mir_set_status` calls and zero `srem`/`sdiv` in optimized IR).
    let local_loop_src = r#"
        function sum_mod_local7(n: int): int
            local d: int = 7
            local s: int = 0
            local i: int = 0
            while i < n do
                s = s + (i * 3) % d
                i = i + 1
            end
            return s
        end
    "#;
    let local_loop_fn = lower_single_function(local_loop_src);
    let pre_ir = compiler
        .dump_function_ir(&local_loop_fn)
        .expect("dump_function_ir sum_mod_local7");
    let pre_summary = analyze_clif_ir(&pre_ir);
    assert!(
        !pre_summary
            .reachable_calls
            .iter()
            .any(|c| c == "lust_mir_set_status"),
        "Constant local `d = 7` must elide `rhs == 0` trap path in CLIF IR:\n{pre_ir}"
    );

    // 3. Optimized CLIF IR for `% 7`, `/ 3`, `% -7`, `/ -7`, and `sum_mod_local7` must contain
    //    zero `sdiv`/`srem` instructions and use magic-multiplier `smulhi`.
    let mut opt_compiler = CraneliftJitCompiler::new().expect("create opt compiler");
    let odd_const_cases = [
        ("function mod7(x: int): int return x % 7 end", true),
        ("function div3(x: int): int return x / 3 end", true),
        ("function mod_neg7(x: int): int return x % -7 end", true),
        ("function div_neg7(x: int): int return x / -7 end", true),
        (local_loop_src, true),
    ];
    for (src, expect_smulhi) in odd_const_cases {
        let mir_fn = lower_single_function(src);
        let opt_ir = opt_compiler
            .dump_optimized_function_ir(&mir_fn)
            .expect("dump_optimized_function_ir");
        let summary = analyze_clif_ir(&opt_ir);
        assert!(
            !has_sdiv(&summary.reachable_lines) && !has_srem(&summary.reachable_lines),
            "Optimized CLIF IR for `{src}` must contain zero sdiv/srem:\n{opt_ir}"
        );
        if expect_smulhi {
            assert!(
                summary.reachable_lines.iter().any(|l| l.contains("smulhi")),
                "Optimized CLIF IR for `{src}` must strength-reduce via smulhi:\n{opt_ir}"
            );
        }
    }

    // 4. `v % 2 == 0` peephole must fold directly to `band` + `icmp` (zero `isub`/`srem` in optimized IR).
    let is_even_fn = lower_single_function("function is_even(v: int): bool return v % 2 == 0 end");
    let opt_even_ir = opt_compiler
        .dump_optimized_function_ir(&is_even_fn)
        .expect("dump_optimized_function_ir is_even");
    let even_summary = analyze_clif_ir(&opt_even_ir);
    assert!(
        !has_srem(&even_summary.reachable_lines)
            && !even_summary.reachable_lines.iter().any(|l| l.contains(" isub "))
            && even_summary.reachable_lines.iter().any(|l| l.contains("band")),
        "Expected `v % 2 == 0` to fold to `(v & 1) == 0` with zero srem/isub:\n{opt_even_ir}"
    );
}

fn run_trap_safety_child_worker() {
    let src = r#"
        function div_c0(x: int): int return x / 0 end
        function mod_c0(x: int): int return x % 0 end
        function div_c_neg1(x: int): int return x / -1 end
        function mod_c_neg1(x: int): int return x % -1 end
        function div_c_min(x: int): int return x / -9223372036854775808 end
        function mod_c_min(x: int): int return x % -9223372036854775808 end
        function div_dyn(a: int, b: int): int return a / b end
        function mod_dyn(a: int, b: int): int return a % b end
        function div_local_neg1(x: int): int
            local d: int = -1
            local i: int = 0
            local out: int = 0
            while i < 1 do
                out = x / d
                i = i + 1
            end
            return out
        end
        function mod_local_0(x: int): int
            local d: int = 0
            local i: int = 0
            local out: int = 0
            while i < 1 do
                out = x % d
                i = i + 1
            end
            return out
        end
    "#;
    let (mut vm_clif, mut vm_bc) = create_vms(src);

    // Non-trapping wrapping semantics on i64::MIN / -1 and i64::MIN % -1
    assert_eq!(vm_clif.call("div_c_neg1", vec![Value::Int(i64::MIN)]).unwrap(), Value::Int(i64::MIN));
    assert_eq!(vm_clif.call("mod_c_neg1", vec![Value::Int(i64::MIN)]).unwrap(), Value::Int(0));
    assert_eq!(vm_clif.call("div_local_neg1", vec![Value::Int(i64::MIN)]).unwrap(), Value::Int(i64::MIN));
    assert_eq!(vm_clif.call("div_dyn", vec![Value::Int(i64::MIN), Value::Int(-1)]).unwrap(), Value::Int(i64::MIN));
    assert_eq!(vm_clif.call("mod_dyn", vec![Value::Int(i64::MIN), Value::Int(-1)]).unwrap(), Value::Int(0));

    // Non-trapping i64::MIN / i64::MIN and i64::MIN % i64::MIN
    assert_eq!(vm_clif.call("div_c_min", vec![Value::Int(i64::MIN)]).unwrap(), Value::Int(1));
    assert_eq!(vm_clif.call("mod_c_min", vec![Value::Int(i64::MIN)]).unwrap(), Value::Int(0));
    assert_eq!(vm_clif.call("div_c_min", vec![Value::Int(i64::MAX)]).unwrap(), Value::Int(0));
    assert_eq!(vm_clif.call("mod_c_min", vec![Value::Int(i64::MAX)]).unwrap(), Value::Int(i64::MAX));

    // Clean runtime error (zero SIGFPE/SIGILL) on all div/mod-by-zero paths
    for (fname, args) in [
        ("div_c0", vec![Value::Int(i64::MIN)]),
        ("mod_c0", vec![Value::Int(i64::MIN)]),
        ("div_dyn", vec![Value::Int(i64::MIN), Value::Int(0)]),
        ("mod_dyn", vec![Value::Int(i64::MIN), Value::Int(0)]),
        ("mod_local_0", vec![Value::Int(42)]),
    ] {
        assert!(vm_clif.call(fname, args.clone()).is_err(), "JIT {fname} must error");
        assert!(vm_bc.call(fname, args).is_err(), "VM {fname} must error");
    }

    println!("NG_W14_5_CHILD_TRAP_SAFETY_OK");
}

#[test]
fn test_ng_w14_5_trap_safety_child_process() {
    if std::env::var_os(CHILD_ENV_KEY).is_some() {
        run_trap_safety_child_worker();
        return;
    }

    let exe = std::env::current_exe().expect("current_exe");
    let mut child = Command::new(exe)
        .arg("--exact")
        .arg("test_ng_w14_5_trap_safety_child_process")
        .arg("--nocapture")
        .env(CHILD_ENV_KEY, "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn child process");

    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if let Some(_status) = child.try_wait().expect("try_wait") {
            let output = child.wait_with_output().expect("wait_with_output");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success(),
                "Child process trapped/failed (status={:?}):\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}",
                output.status
            );
            assert!(
                stdout.contains("NG_W14_5_CHILD_TRAP_SAFETY_OK"),
                "Missing sentinel output:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            );
            return;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("Watchdog timeout (15s): child process hung");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_ng_w14_5_deopt_parity_in_const_mod_loop() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let src = r#"
        function mix_mod_div_loop(n: int): int
            local d7: int = 7
            local d2: int = 2
            local acc: int = 0
            local i: int = -n
            while i < n do
                local m7: int = (i * 3) % d7
                local q2: int = i / d2
                if i % 2 == 0 then
                    acc = acc + m7 + q2
                else
                    acc = acc - m7 + (i % -4)
                end
                i = i + 1
            end
            return acc
        end
    "#;

    let (mut vm_clif, mut vm_bc) = create_vms(src);
    let expected = vm_bc
        .call("mix_mod_div_loop", vec![Value::Int(120)])
        .expect("VM baseline");

    // Pure JIT run
    let jit_clean = vm_clif
        .call("mix_mod_div_loop", vec![Value::Int(120)])
        .expect("JIT clean");
    assert_eq!(jit_clean, expected);

    // Forced mid-loop deopt -> VM continuation must produce identical result
    safepoint_reset();
    safepoint_request_deopt_on_poll(Some("mix_mod_div_loop".to_string()), 1);
    let jit_deopt = vm_clif
        .call("mix_mod_div_loop", vec![Value::Int(120)])
        .expect("JIT deopt continuation");
    safepoint_reset();
    assert_eq!(jit_deopt, expected, "Forced deopt continuation must match VM exactly");
}

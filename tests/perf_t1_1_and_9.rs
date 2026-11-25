#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use cranelift_codegen::settings::OptLevel;
use lust::bytecode::Value;
use lust::bytecode::compiler::Compiler;
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{
    CraneliftJitCompiler, MIR_STATUS_ERR_DIV_BY_ZERO, MIR_STATUS_OK, lust_mir_get_status,
    lust_mir_set_status,
};
use lust::mir::ir::{MirFunction, MirModule};
use lust::mir::lowering::AstLowering;
use lust::mir::opt::MirOptimizer;
use lust::mir::runtime_helpers::{MirContext, enter_mir_context};
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{VM, VmBackend};

const CHILD_ENV_KEY: &str = "LUST_PERF_T1_9_CHILD_WORKER";

fn lower_module_from_source(src: &str) -> (MirModule, Vec<lust::ast::Item>) {
    let mut interner = Interner::new();
    let tokens = Lexer::new(src, &mut interner)
        .tokenize()
        .expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");
    let mut module = MirModule::new("perf_t1_1_and_9".into());
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
    reachable_blocks: Vec<String>,
    reachable_lines: Vec<String>,
    reachable_calls: Vec<String>,
    all_calls: Vec<String>,
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
                blocks.entry(bname.clone()).or_default().push(line.to_string());
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

    let mut all_calls = Vec::new();
    for bname in &block_order {
        if let Some(lines) = blocks.get(bname) {
            for l in lines {
                if let Some(sym) = extract_call_sym(l) {
                    all_calls.push(sym);
                }
            }
        }
    }

    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<String> = VecDeque::new();
    if let Some(entry) = block_order.first() {
        visited.insert(entry.clone());
        queue.push_back(entry.clone());
    }
    let mut reachable_blocks = Vec::new();
    let mut reachable_lines = Vec::new();
    let mut reachable_calls = Vec::new();

    while let Some(b) = queue.pop_front() {
        reachable_blocks.push(b.clone());
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
        reachable_blocks,
        reachable_lines,
        reachable_calls,
        all_calls,
    }
}

#[test]
fn test_perf_t1_1_cranelift_opt_level_speed() {
    let compiler = CraneliftJitCompiler::new().expect("create CraneliftJitCompiler");
    assert_eq!(
        compiler.isa_opt_level(),
        OptLevel::Speed,
        "PERF-T1-1 requires Cranelift ISA opt_level = OptLevel::Speed (got {:?})",
        compiler.isa_opt_level()
    );
    assert_eq!(compiler.isa_opt_level_str(), "speed");

    // Adversarial IEEE-754 parity check under OptLevel::Speed (signed zero, NaN, inf)
    let src = r#"
        function neg_zero(x: float): float return -x end
        function add_f(a: float, b: float): float return a + b end
        function div_f(a: float, b: float): float return a / b end
    "#;
    let (mut vm_clif, mut vm_bc) = create_vms(src);

    let clif_nz = vm_clif.call("neg_zero", vec![Value::Float(0.0)]).unwrap();
    let bc_nz = vm_bc.call("neg_zero", vec![Value::Float(0.0)]).unwrap();
    match (clif_nz, bc_nz) {
        (Value::Float(a), Value::Float(b)) => {
            assert_eq!(a.to_bits(), b.to_bits(), "signed-zero bit parity");
            assert_eq!(a.to_bits(), (-0.0_f64).to_bits());
        }
        other => panic!("unexpected float result: {other:?}"),
    }

    let clif_nan = vm_clif
        .call("div_f", vec![Value::Float(0.0), Value::Float(0.0)])
        .unwrap();
    let bc_nan = vm_bc
        .call("div_f", vec![Value::Float(0.0), Value::Float(0.0)])
        .unwrap();
    match (clif_nan, bc_nan) {
        (Value::Float(a), Value::Float(b)) => {
            assert!(a.is_nan() && b.is_nan(), "0.0 / 0.0 must be NaN in both engines");
        }
        other => panic!("unexpected NaN result: {other:?}"),
    }

    let clif_sum = vm_clif
        .call("add_f", vec![Value::Float(1.5), Value::Float(-0.0)])
        .unwrap();
    let bc_sum = vm_bc
        .call("add_f", vec![Value::Float(1.5), Value::Float(-0.0)])
        .unwrap();
    assert_eq!(clif_sum, bc_sum);
}

#[test]
fn test_perf_t1_9_inline_int_div_mod_semantics_and_ir() {
    // (a) Cranelift IR inspection for constant non-zero / non-(-1) divisors
    let mut compiler = CraneliftJitCompiler::new().expect("create compiler");

    let mod7_fn = lower_single_function("function mod7(x: int): int return x % 7 end");
    let mod7_ir = compiler
        .dump_function_ir(&mod7_fn)
        .expect("dump_function_ir mod7");
    let mod7_summary = analyze_clif_ir(&mod7_ir);
    assert!(
        mod7_summary
            .reachable_lines
            .iter()
            .any(|l| l.contains(" srem ") || l.contains("srem.")),
        "Expected direct `srem` instruction for `x % 7` in Cranelift IR:\n{mod7_ir}"
    );
    assert!(
        !mod7_summary.all_calls.iter().any(|c| c == "lust_mir_rem_int"),
        "Found forbidden `lust_mir_rem_int` call for `x % 7`:\n{mod7_ir}"
    );
    assert!(
        !mod7_summary.all_calls.iter().any(|c| c == "lust_mir_div_int"),
        "Found forbidden `lust_mir_div_int` call for `x % 7`:\n{mod7_ir}"
    );
    assert!(
        !mod7_summary
            .reachable_calls
            .iter()
            .any(|c| c == "lust_mir_get_status"),
        "Found forbidden `lust_mir_get_status` call in reachable blocks for `x % 7` (reachable blocks {:?}):\n{mod7_ir}",
        mod7_summary.reachable_blocks
    );

    let div3_fn = lower_single_function("function div3(x: int): int return x / 3 end");
    let div3_ir = compiler
        .dump_function_ir(&div3_fn)
        .expect("dump_function_ir div3");
    let div3_summary = analyze_clif_ir(&div3_ir);
    assert!(
        div3_summary
            .reachable_lines
            .iter()
            .any(|l| l.contains(" sdiv ") || l.contains("sdiv.")),
        "Expected direct `sdiv` instruction for `x / 3` in Cranelift IR:\n{div3_ir}"
    );
    assert!(
        !div3_summary.all_calls.iter().any(|c| c == "lust_mir_div_int"),
        "Found forbidden `lust_mir_div_int` call for `x / 3`:\n{div3_ir}"
    );
    assert!(
        !div3_summary.all_calls.iter().any(|c| c == "lust_mir_rem_int"),
        "Found forbidden `lust_mir_rem_int` call for `x / 3`:\n{div3_ir}"
    );
    assert!(
        !div3_summary
            .reachable_calls
            .iter()
            .any(|c| c == "lust_mir_get_status"),
        "Found forbidden `lust_mir_get_status` call in reachable blocks for `x / 3`:\n{div3_ir}"
    );

    // Constant -1 must use ineg / iconst 0 with zero sdiv/srem/helper calls
    let div_neg1_fn = lower_single_function("function div_neg1(x: int): int return x / -1 end");
    let div_neg1_ir = compiler
        .dump_function_ir(&div_neg1_fn)
        .expect("dump_function_ir div_neg1");
    let div_neg1_summary = analyze_clif_ir(&div_neg1_ir);
    assert!(
        !div_neg1_summary
            .reachable_lines
            .iter()
            .any(|l| l.contains(" sdiv ") || l.contains("sdiv.")),
        "`x / -1` must use ineg, not trapping sdiv:\n{div_neg1_ir}"
    );
    assert!(
        !div_neg1_summary
            .all_calls
            .iter()
            .any(|c| c == "lust_mir_div_int" || c == "lust_mir_rem_int"),
        "`x / -1` must not call div/rem helpers:\n{div_neg1_ir}"
    );

    let mod_neg1_fn = lower_single_function("function mod_neg1(x: int): int return x % -1 end");
    let mod_neg1_ir = compiler
        .dump_function_ir(&mod_neg1_fn)
        .expect("dump_function_ir mod_neg1");
    let mod_neg1_summary = analyze_clif_ir(&mod_neg1_ir);
    assert!(
        !mod_neg1_summary
            .reachable_lines
            .iter()
            .any(|l| l.contains(" srem ") || l.contains("srem.")),
        "`x % -1` must return constant 0, not trapping srem:\n{mod_neg1_ir}"
    );

    // Constant 0 followed by dead code using variables must still verify cleanly in Cranelift
    let div_zero_dead_fn = lower_single_function(
        r#"
        function div_zero_dead(x: int): int
            local y: int = x / 0
            return y + x
        end
        "#,
    );
    let _div_zero_ir = compiler
        .dump_function_ir(&div_zero_dead_fn)
        .expect("constant div by 0 with dead continuation must verify cleanly");

    // (b) Semantics parity: Cranelift JIT vs Bytecode VM bit-for-bit
    let src = r#"
        function mod_c2(x: int): int return x % 2 end
        function mod_c_neg2(x: int): int return x % -2 end
        function mod_c7(x: int): int return x % 7 end
        function mod_c_neg1(x: int): int return x % -1 end
        function mod_c0(x: int): int return x % 0 end

        function div_c2(x: int): int return x / 2 end
        function div_c_neg2(x: int): int return x / -2 end
        function div_c3(x: int): int return x / 3 end
        function div_c_neg1(x: int): int return x / -1 end
        function div_c0(x: int): int return x / 0 end
        function div_c0_dead(x: int): int
            local y: int = x / 0
            return y + x
        end

        function div_dyn(a: int, b: int): int return a / b end
        function mod_dyn(a: int, b: int): int return a % b end
        function floordiv_dyn(a: int, b: int): int return a // b end
    "#;
    let (mut vm_clif, mut vm_bc) = create_vms(src);

    let ok_cases: Vec<(&str, Vec<Value>, i64)> = vec![
        // Required edge cases (constant RHS)
        ("mod_c2", vec![Value::Int(-7)], -1),
        ("mod_c_neg2", vec![Value::Int(7)], 1),
        ("mod_c_neg2", vec![Value::Int(-7)], -1),
        ("mod_c7", vec![Value::Int(100)], 2),
        ("mod_c7", vec![Value::Int(-100)], -2),
        ("mod_c_neg1", vec![Value::Int(i64::MIN)], 0),
        ("mod_c_neg1", vec![Value::Int(42)], 0),
        ("div_c2", vec![Value::Int(-7)], -3),
        ("div_c_neg2", vec![Value::Int(7)], -3),
        ("div_c_neg2", vec![Value::Int(-7)], 3),
        ("div_c3", vec![Value::Int(100)], 33),
        ("div_c3", vec![Value::Int(-100)], -33),
        ("div_c_neg1", vec![Value::Int(i64::MIN)], i64::MIN),
        ("div_c_neg1", vec![Value::Int(i64::MAX)], -i64::MAX),
        ("div_c_neg1", vec![Value::Int(0)], 0),
        // Required edge cases (dynamic RHS)
        ("mod_dyn", vec![Value::Int(-7), Value::Int(2)], -1),
        ("mod_dyn", vec![Value::Int(7), Value::Int(-2)], 1),
        ("mod_dyn", vec![Value::Int(-7), Value::Int(-2)], -1),
        ("div_dyn", vec![Value::Int(-7), Value::Int(2)], -3),
        ("div_dyn", vec![Value::Int(7), Value::Int(-2)], -3),
        ("div_dyn", vec![Value::Int(-7), Value::Int(-2)], 3),
        ("div_dyn", vec![Value::Int(i64::MIN), Value::Int(-1)], i64::MIN),
        ("mod_dyn", vec![Value::Int(i64::MIN), Value::Int(-1)], 0),
        ("div_dyn", vec![Value::Int(i64::MIN), Value::Int(2)], i64::MIN / 2),
        ("mod_dyn", vec![Value::Int(i64::MIN), Value::Int(2)], 0),
        ("div_dyn", vec![Value::Int(i64::MAX), Value::Int(-1)], -i64::MAX),
        ("mod_dyn", vec![Value::Int(i64::MAX), Value::Int(-1)], 0),
        ("div_dyn", vec![Value::Int(0), Value::Int(7)], 0),
        ("mod_dyn", vec![Value::Int(0), Value::Int(7)], 0),
        // FloorDiv unchanged parity
        ("floordiv_dyn", vec![Value::Int(-7), Value::Int(2)], -4),
        ("floordiv_dyn", vec![Value::Int(7), Value::Int(-2)], -4),
        ("floordiv_dyn", vec![Value::Int(-7), Value::Int(-2)], 3),
        ("floordiv_dyn", vec![Value::Int(i64::MIN), Value::Int(-1)], i64::MIN),
    ];

    for (func, args, expected) in ok_cases {
        let entries_before = vm_clif.cranelift_entry_count;
        let clif_res = vm_clif
            .call(func, args.clone())
            .unwrap_or_else(|e| panic!("Cranelift {func}({args:?}) failed: {e}"));
        let bc_res = vm_bc
            .call(func, args.clone())
            .unwrap_or_else(|e| panic!("VM {func}({args:?}) failed: {e}"));
        assert_eq!(clif_res, Value::Int(expected), "Cranelift {func}({args:?})");
        assert_eq!(clif_res, bc_res, "VM parity mismatch for {func}({args:?})");
        assert!(
            vm_clif.cranelift_entry_count > entries_before,
            "{func} must execute via native Cranelift JIT"
        );
    }

    // Division/modulo by zero must error in both Cranelift and VM
    let zero_cases: Vec<(&str, Vec<Value>)> = vec![
        ("div_c0", vec![Value::Int(42)]),
        ("div_c0_dead", vec![Value::Int(42)]),
        ("mod_c0", vec![Value::Int(42)]),
        ("div_dyn", vec![Value::Int(42), Value::Int(0)]),
        ("mod_dyn", vec![Value::Int(42), Value::Int(0)]),
        ("div_dyn", vec![Value::Int(i64::MIN), Value::Int(0)]),
        ("mod_dyn", vec![Value::Int(i64::MIN), Value::Int(0)]),
        ("floordiv_dyn", vec![Value::Int(42), Value::Int(0)]),
    ];
    for (func, args) in zero_cases {
        let entries_before = vm_clif.cranelift_entry_count;
        let clif_err = vm_clif.call(func, args.clone());
        let bc_err = vm_bc.call(func, args.clone());
        assert!(
            clif_err.is_err(),
            "Cranelift {func}({args:?}) must fail with division/modulo by zero, got {clif_err:?}"
        );
        assert!(
            bc_err.is_err(),
            "VM {func}({args:?}) must fail with division/modulo by zero, got {bc_err:?}"
        );
        assert!(
            vm_clif.cranelift_entry_count > entries_before,
            "{func} must enter native Cranelift JIT before reporting div-by-zero"
        );
    }
}

fn run_adversarial_child_worker() {
    let src = r#"
        function div_dyn(a: int, b: int): int return a / b end
        function mod_dyn(a: int, b: int): int return a % b end
        function div_c_neg1(a: int): int return a / -1 end
        function mod_c_neg1(a: int): int return a % -1 end
        function div_c0(a: int): int return a / 0 end
        function mod_c0(a: int): int return a % 0 end
    "#;
    let (module, _) = lower_module_from_source(src);
    let mut compiler = CraneliftJitCompiler::new().expect("create compiler");
    let compiled = compiler.compile_module(&module).expect("compile_module");

    let div_dyn: extern "C" fn(i64, i64) -> i64 =
        unsafe { std::mem::transmute(compiled["div_dyn"]) };
    let mod_dyn: extern "C" fn(i64, i64) -> i64 =
        unsafe { std::mem::transmute(compiled["mod_dyn"]) };
    let div_c_neg1: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(compiled["div_c_neg1"]) };
    let mod_c_neg1: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(compiled["mod_c_neg1"]) };
    let div_c0: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(compiled["div_c0"]) };
    let mod_c0: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(compiled["mod_c0"]) };

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);

    // 1. Dynamic (i64::MIN, -1) must NOT trap with SIGFPE/SIGILL
    lust_mir_set_status(MIR_STATUS_OK);
    assert_eq!(div_dyn(i64::MIN, -1), i64::MIN);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);

    lust_mir_set_status(MIR_STATUS_OK);
    assert_eq!(mod_dyn(i64::MIN, -1), 0);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);

    // 2. Dynamic b = 0 must NOT trap and must set MIR_STATUS_ERR_DIV_BY_ZERO
    lust_mir_set_status(MIR_STATUS_OK);
    let _ = div_dyn(42, 0);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_DIV_BY_ZERO);

    lust_mir_set_status(MIR_STATUS_OK);
    let _ = mod_dyn(42, 0);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_DIV_BY_ZERO);

    lust_mir_set_status(MIR_STATUS_OK);
    let _ = div_dyn(i64::MIN, 0);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_DIV_BY_ZERO);

    lust_mir_set_status(MIR_STATUS_OK);
    let _ = mod_dyn(i64::MIN, 0);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_DIV_BY_ZERO);

    // 3. Constant -1 and constant 0 direct ABI checks
    lust_mir_set_status(MIR_STATUS_OK);
    assert_eq!(div_c_neg1(i64::MIN), i64::MIN);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);

    lust_mir_set_status(MIR_STATUS_OK);
    assert_eq!(mod_c_neg1(i64::MIN), 0);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_OK);

    lust_mir_set_status(MIR_STATUS_OK);
    let _ = div_c0(42);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_DIV_BY_ZERO);

    lust_mir_set_status(MIR_STATUS_OK);
    let _ = mod_c0(42);
    assert_eq!(lust_mir_get_status(), MIR_STATUS_ERR_DIV_BY_ZERO);

    println!("CHILD_TRAP_AVOIDANCE_OK");
}

#[test]
fn test_perf_t1_9_adversarial_dynamic_trap_avoidance_child_process() {
    if std::env::var_os(CHILD_ENV_KEY).is_some() {
        run_adversarial_child_worker();
        return;
    }

    let exe = std::env::current_exe().expect("current_exe");
    let mut child = Command::new(exe)
        .arg("--exact")
        .arg("test_perf_t1_9_adversarial_dynamic_trap_avoidance_child_process")
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
                "Child process trapped/failed (status={:?}, possible SIGFPE/SIGILL):\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}",
                output.status
            );
            assert!(
                stdout.contains("CHILD_TRAP_AVOIDANCE_OK"),
                "Child process missing sentinel output:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            );
            return;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("Watchdog timeout (15s): child process hung during dynamic div/mod trap test");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

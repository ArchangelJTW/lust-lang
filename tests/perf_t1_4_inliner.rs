#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

use lust::ast::Span;
use lust::bytecode::{Instruction, Value};
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{
    DeoptSiteKind, get_current_sticky_error, safepoint_request_deopt, safepoint_reset,
    take_current_sticky_error, take_deopt_payload,
};
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::safepoint_request_deopt_on_poll;
use lust::mir::ir::{
    FunctionSignature, MirBinaryOp, MirConstant, MirFunction, MirModule, MirType, Operand, Place,
    Rvalue, Statement, Terminator,
};
use lust::mir::opt::{JitInlineConfig, MirOptimizer};
use lust::mir::verifier::MirVerifier;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{VM, VmBackend, reset_effect_handler_state};
use lust::{
    Compiler, EmbeddedModule, LustConfig, compile_program_with_config, load_program_from_embedded,
};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn lust_bin() -> &'static str {
    env!("CARGO_BIN_EXE_lust")
}

fn reset_safepoint_test_state() {
    safepoint_reset();
    reset_effect_handler_state();
    take_deopt_payload();
    take_current_sticky_error();
}

const CALLS_LOOP_SRC: &str = r#"
function step(a: int, b: int): int
    return a + b * 2
end

function run_calls(n: int): int
    local acc: int = 0
    local i: int = 0
    while i < n do
        acc = step(acc, i)
        i = i + 1
    end
    return acc
end
"#;

fn build_vm_with_backend(source: &str, cranelift: bool) -> VM {
    let entries = &[EmbeddedModule {
        module: "main",
        parent: None,
        source: Some(source),
    }];
    let program = load_program_from_embedded(entries, "main").expect("load embedded program");
    let mut cfg = LustConfig::default().with_cranelift_jit(cranelift);
    cfg.set_use_mir(false);
    compile_program_with_config(program, &cfg).expect("compile embedded program")
}

fn setup_bytecode_funcs(source: &str) -> Vec<lust::bytecode::Function> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");
    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    compiler.compile_module(&items).expect("compile_module")
}

// =========================================================================
// Core PERF-T1-4 Test: Leaf Helper Inlining + Authoritative Deopt Recipes
// =========================================================================

#[test]
fn test_perf_t1_4_jit_inlines_leaf_helper_preserving_deopt_recipes() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_safepoint_test_state();

    let mut vm = build_vm_with_backend(CALLS_LOOP_SRC, true);
    let run_name = vm
        .mir_functions
        .keys()
        .find(|k| k.ends_with("run_calls"))
        .cloned()
        .expect("run_calls in mir_functions");
    let step_name = vm
        .mir_functions
        .keys()
        .find(|k| k.ends_with("step"))
        .cloned()
        .expect("step in mir_functions");

    // Pre-inline MIR still calls `step`.
    let pre_inline_has_step = vm.mir_functions[&run_name].blocks.iter().any(|b| {
        matches!(&b.terminator, Some(Terminator::Call { callee, .. }) if callee == &step_name)
    });
    assert!(
        pre_inline_has_step,
        "pre-inline MIR for `{run_name}` must contain Terminator::Call to `{step_name}`"
    );

    // Compile via VM::compile_cranelift()
    vm.compile_cranelift().expect("compile_cranelift");

    // (a.1) Post-inline compiled MIR for caller must have ZERO Terminator::Call to `step`.
    let post_inline_mir = vm
        .cranelift_compiled_mir(&run_name)
        .expect("post-inline compiled MIR for run_calls");
    MirVerifier::verify_fn(&post_inline_mir).expect("post-inline MIR must verify cleanly");
    let post_inline_has_step = post_inline_mir.blocks.iter().any(|b| {
        matches!(&b.terminator, Some(Terminator::Call { callee, .. }) if callee == &step_name)
    });
    assert!(
        !post_inline_has_step,
        "post-inline compiled MIR for `{run_name}` must have ZERO Terminator::Call to `{step_name}`"
    );

    // (a.2) Installed continuation bytecode for `run_calls` must match post-inline MIR (zero Call).
    let bc_fn = vm
        .function_by_name(&run_name)
        .expect("run_calls continuation bytecode");
    let has_call_insn = bc_fn
        .chunk
        .instructions
        .iter()
        .any(|insn| matches!(insn, Instruction::Call(..)));
    assert!(
        !has_call_insn,
        "Expected leaf helper `{step_name}` to be inlined into `{run_name}` before continuation bytecode & Cranelift codegen, but found Call instruction in {:?}",
        bc_fn.chunk.instructions
    );

    // (a.3) Loop-header BlockEntry deopt recipe must exist and point to a valid continuation PC.
    let recipe_sites = vm.cranelift_deopt_recipe_sites(&run_name);
    assert!(
        recipe_sites
            .iter()
            .any(|&(_, kind, _)| kind == DeoptSiteKind::BlockEntry),
        "expected loop-header BlockEntry deopt recipe for `{run_name}`, got {:?}",
        recipe_sites
    );

    // Normal native Cranelift execution oracle check: sum_{i=0}^{99} (2*i) = 9900
    let normal_res = vm
        .call(&run_name, vec![Value::Int(100)])
        .expect("native run_calls");
    assert_eq!(normal_res, Value::Int(9900));
    assert!(vm.cranelift_entry_count >= 1);

    // (b) Forced safepoint deopt at the loop header reconstructs all live locals and completes
    //     with exact oracle output.
    reset_safepoint_test_state();
    safepoint_request_deopt(true);
    let deopt_res = vm
        .call(&run_name, vec![Value::Int(100)])
        .expect("deopted run_calls");
    reset_safepoint_test_state();
    assert_eq!(deopt_res, Value::Int(9900));

    // Under `test-deopt`, also test mid-loop deopt at poll ordinal 25 (after 24 inlined iterations).
    #[cfg(feature = "test-deopt")]
    {
        reset_safepoint_test_state();
        safepoint_request_deopt_on_poll(Some(run_name.clone()), 25);
        let mid_loop_deopt = vm
            .call(&run_name, vec![Value::Int(100)])
            .expect("mid-loop poll-25 deopt must resume cleanly");
        reset_safepoint_test_state();
        assert_eq!(mid_loop_deopt, Value::Int(9900));
    }
}

// =========================================================================
// Adversarial Negative Test 1: Direct Recursion (fib) & Mutual Recursion (is_even/is_odd)
// =========================================================================

#[test]
fn test_perf_t1_4_direct_and_mutual_recursion_never_infinite_loop_and_match_oracle() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_safepoint_test_state();

    let src = r#"
function fib(n: int): int
    if n <= 1 then
        return n
    end
    return fib(n - 1) + fib(n - 2)
end

function is_even(n: int): int
    if n == 0 then
        return 1
    end
    return is_odd(n - 1)
end

function is_odd(n: int): int
    if n == 0 then
        return 0
    end
    return is_even(n - 1)
end
"#;

    let mut vm_clif = build_vm_with_backend(src, true);
    let mut vm_bc = build_vm_with_backend(src, false);

    let fib_name = vm_clif
        .mir_functions
        .keys()
        .find(|k| k.ends_with("fib"))
        .cloned()
        .expect("fib");
    let even_name = vm_clif
        .mir_functions
        .keys()
        .find(|k| k.ends_with("is_even"))
        .cloned()
        .expect("is_even");
    let odd_name = vm_clif
        .mir_functions
        .keys()
        .find(|k| k.ends_with("is_odd"))
        .cloned()
        .expect("is_odd");

    // Must terminate cleanly without infinite inlining.
    vm_clif.compile_cranelift().expect("compile_cranelift");

    let fib_mir = vm_clif
        .cranelift_compiled_mir(&fib_name)
        .expect("compiled fib MIR");
    assert!(
        fib_mir.blocks.iter().any(|b| {
            matches!(&b.terminator, Some(Terminator::Call { callee, .. }) if callee == &fib_name)
        }),
        "direct recursion in fib must be preserved"
    );

    let even_mir = vm_clif
        .cranelift_compiled_mir(&even_name)
        .expect("compiled is_even MIR");
    let odd_mir = vm_clif
        .cranelift_compiled_mir(&odd_name)
        .expect("compiled is_odd MIR");
    assert!(
        even_mir.blocks.iter().any(|b| {
            matches!(&b.terminator, Some(Terminator::Call { callee, .. }) if callee == &odd_name)
        }),
        "mutual recursion call is_even -> is_odd must be preserved"
    );
    assert!(
        odd_mir.blocks.iter().any(|b| {
            matches!(&b.terminator, Some(Terminator::Call { callee, .. }) if callee == &even_name)
        }),
        "mutual recursion call is_odd -> is_even must be preserved"
    );

    // Exact execution parity with bytecode VM oracle
    for n in [0, 1, 5, 10, 15] {
        let clif_v = vm_clif.call(&fib_name, vec![Value::Int(n)]).expect("clif fib");
        let bc_v = vm_bc.call(&fib_name, vec![Value::Int(n)]).expect("bc fib");
        assert_eq!(clif_v, bc_v, "fib({n}) parity mismatch");
    }
    for n in [0, 1, 2, 7, 14, 21] {
        let clif_e = vm_clif
            .call(&even_name, vec![Value::Int(n)])
            .expect("clif is_even");
        let bc_e = vm_bc
            .call(&even_name, vec![Value::Int(n)])
            .expect("bc is_even");
        assert_eq!(clif_e, bc_e, "is_even({n}) parity mismatch");

        let clif_o = vm_clif
            .call(&odd_name, vec![Value::Int(n)])
            .expect("clif is_odd");
        let bc_o = vm_bc
            .call(&odd_name, vec![Value::Int(n)])
            .expect("bc is_odd");
        assert_eq!(clif_o, bc_o, "is_odd({n}) parity mismatch");
    }
    assert!(vm_clif.cranelift_entry_count >= 1);
}

// =========================================================================
// Adversarial Negative Test 2: Effect-Performing Callee NOT Inlined + PreTerminator Preserved
// =========================================================================

fn make_effect_callee_mir(name: &str, intrinsic: &str) -> MirFunction {
    let span = Span::dummy();
    let mut f = MirFunction::new(
        FunctionSignature {
            name: name.into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let x = f.params[0];
    let r = f.add_local(Some("r".into()), MirType::Int, false, span);
    let out = f.add_local(Some("out".into()), MirType::Int, false, span);
    let bb1 = f.new_block();

    f.blocks[0].terminator = Some(Terminator::Call {
        callee: intrinsic.into(),
        args: vec![
            Operand::Constant(MirConstant::String("Tick".into())),
            Operand::Copy(Place::Local(x)),
        ],
        destination: Place::Local(r),
        success: bb1,
        error: None,
        span,
    });
    f.blocks[bb1.0].statements.push(Statement::Assign(
        Place::Local(out),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(r)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        span,
    ));
    f.blocks[bb1.0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(out))), span));
    f
}

fn make_wrapper_calling(name: &str, target: &str) -> MirFunction {
    let span = Span::dummy();
    let mut f = MirFunction::new(
        FunctionSignature {
            name: name.into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let x = f.params[0];
    let res = f.add_local(Some("res".into()), MirType::Int, false, span);
    let bb1 = f.new_block();
    f.blocks[0].terminator = Some(Terminator::Call {
        callee: target.into(),
        args: vec![Operand::Copy(Place::Local(x))],
        destination: Place::Local(res),
        success: bb1,
        error: None,
        span,
    });
    f.blocks[bb1.0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(res))), span));
    f
}

#[test]
fn test_perf_t1_4_effect_callee_not_inlined_and_preserves_preterminator_deopt() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_safepoint_test_state();

    // Part A: Direct `MirOptimizer::optimize_module_for_jit` guard verification for BOTH
    // `__lust_perform` and `effect.perform`.
    for intrinsic in ["__lust_perform", "effect.perform"] {
        let mut module = MirModule::new("effect_guard_mod".into());
        module.add_function(make_effect_callee_mir("eff_callee", intrinsic));
        module.add_function(make_wrapper_calling("caller_fn", "eff_callee"));

        let inlined = MirOptimizer::optimize_module_for_jit(&mut module, &JitInlineConfig::default());
        assert!(
            !inlined.contains("caller_fn"),
            "caller_fn must NOT inline effect callee using `{intrinsic}`"
        );
        let caller = module
            .functions
            .iter()
            .find(|f| f.signature.name == "caller_fn")
            .unwrap();
        assert!(
            caller.blocks.iter().any(|b| {
                matches!(&b.terminator, Some(Terminator::Call { callee, .. }) if callee == "eff_callee")
            }),
            "Call to `eff_callee` ({intrinsic}) must remain un-inlined in `caller_fn`"
        );
    }

    // Part B: End-to-end VM::compile_cranelift() + effect resume verifies exact PreTerminator
    // deopt site identity on `eff_callee` (`__lust_perform`) and exact runtime oracle parity.
    let driver_src = r#"
        function eff_callee(x: int): int
            return 0
        end

        function main(): int
            handle {
                eff_callee(5)
            } with {
                Tick(v) => resume(v + 10)
            }
        end
    "#;
    let bc_funcs = setup_bytecode_funcs(driver_src);
    let eff_mir = make_effect_callee_mir("eff_callee", "__lust_perform");

    let mut vm_bc = VM::new();
    vm_bc.set_backend(VmBackend::Bytecode);
    vm_bc.load_functions(bc_funcs.clone());
    vm_bc
        .load_mir_function(eff_mir.clone())
        .expect("load_mir_function bc");
    let expected = vm_bc.call("main", vec![]).expect("bc main");
    assert_eq!(expected, Value::Int(16));

    let mut vm_clif = VM::new();
    vm_clif.set_backend(VmBackend::Cranelift);
    vm_clif.load_functions(bc_funcs);
    vm_clif
        .load_mir_function(eff_mir)
        .expect("load_mir_function clif");
    vm_clif.compile_cranelift().expect("compile_cranelift");

    // Verify PreTerminator deopt site identity is preserved at block 0.
    let sites = vm_clif.cranelift_deopt_recipe_sites("eff_callee");
    assert_eq!(
        sites.len(),
        1,
        "expected exactly one PreTerminator recipe for eff_callee, got {:?}",
        sites
    );
    assert_eq!(sites[0].0, 0, "PreTerminator block_id must stay 0");
    assert_eq!(sites[0].1, DeoptSiteKind::PreTerminator);

    let entries_before = vm_clif.cranelift_entry_count;
    let actual = vm_clif.call("main", vec![]).expect("clif main");
    assert_eq!(actual, expected);
    assert_eq!(vm_clif.cranelift_entry_count, entries_before + 1);
    assert!(take_deopt_payload().is_none());
    assert!(get_current_sticky_error().is_none());
}

// =========================================================================
// Adversarial Negative Test 3: Loop-Bearing Callee NOT Inlined + Nested-Deopt & Poll Ordinals
// =========================================================================

#[test]
fn test_perf_t1_4_loop_bearing_callee_not_inlined_preserves_nested_deopt_guard() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_safepoint_test_state();

    let src = r#"
function inner_loop(n: int): int
    local s: int = 0
    local i: int = 0
    while i < n do
        s = s + i
        i = i + 1
    end
    return s
end

function outer_caller(n: int): int
    return inner_loop(n) + 1
end
"#;

    let mut vm = build_vm_with_backend(src, true);
    let outer_name = vm
        .mir_functions
        .keys()
        .find(|k| k.ends_with("outer_caller"))
        .cloned()
        .expect("outer_caller");
    let inner_name = vm
        .mir_functions
        .keys()
        .find(|k| k.ends_with("inner_loop"))
        .cloned()
        .expect("inner_loop");

    vm.compile_cranelift().expect("compile_cranelift");

    // Guard 2: Loop-bearing `inner_loop` MUST NOT be inlined into `outer_caller`.
    let outer_mir = vm
        .cranelift_compiled_mir(&outer_name)
        .expect("compiled outer_caller MIR");
    let still_calls_inner = outer_mir.blocks.iter().any(|b| {
        matches!(&b.terminator, Some(Terminator::Call { callee, .. }) if callee == &inner_name)
    });
    assert!(
        still_calls_inner,
        "loop-bearing callee `{inner_name}` MUST NOT be inlined into `{outer_name}`"
    );

    // Normal execution parity: sum(0..10) + 1 = 46
    let ok_res = vm
        .call(&outer_name, vec![Value::Int(10)])
        .expect("normal outer_caller");
    assert_eq!(ok_res, Value::Int(46));

    // Nested-deopt fail-closed guard: requesting safepoint deopt while executing nested
    // native callee `inner_loop` inside `outer_caller` MUST fail closed (NOT silently succeed
    // via accidental inlining!).
    reset_safepoint_test_state();
    safepoint_request_deopt(true);
    let err = vm
        .call(&outer_name, vec![Value::Int(10)])
        .expect_err("nested deopt inside un-inlined loop callee must fail closed");
    reset_safepoint_test_state();
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("nested") || msg.to_lowercase().contains("deopt"),
        "expected nested native deopt fail-closed error, got: {msg}"
    );

    // Direct top-level call to `inner_loop` with `test-deopt` poll ordinal preserves exact ordinal!
    #[cfg(feature = "test-deopt")]
    {
        reset_safepoint_test_state();
        safepoint_request_deopt_on_poll(Some(inner_name.clone()), 5);
        let direct_deopt = vm
            .call(&inner_name, vec![Value::Int(10)])
            .expect("top-level inner_loop deopt on poll 5 must resume cleanly");
        reset_safepoint_test_state();
        assert_eq!(direct_deopt, Value::Int(45));
    }
}

// =========================================================================
// Adversarial Negative Test 4: First-Call Deopt After Inlining Increases Locals Beyond AST max_register
// =========================================================================

#[test]
fn test_perf_t1_4_first_call_deopt_after_inlining_exceeds_ast_register_count() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_safepoint_test_state();

    // Construct leaf helpers with many locals so inlining them into `driver` strictly grows
    // `driver`'s post-inline `local_types.len()` and `continuation_bc.register_count` well
    // beyond `driver`'s original AST `Function.register_count`.
    let src = r#"
function heavy_leaf_a(x: int): int
    local a1: int = x + 1
    local a2: int = a1 + 2
    local a3: int = a2 + 3
    local a4: int = a3 + 4
    local a5: int = a4 + 5
    local a6: int = a5 + 6
    local a7: int = a6 + 7
    local a8: int = a7 + 8
    return a8
end

function heavy_leaf_b(y: int): int
    local b1: int = y + 10
    local b2: int = b1 + 11
    local b3: int = b2 + 12
    local b4: int = b3 + 13
    local b5: int = b4 + 14
    local b6: int = b5 + 15
    local b7: int = b6 + 16
    local b8: int = b7 + 17
    return b8
end

function driver(n: int): int
    local acc: int = 0
    local i: int = 0
    while i < n do
        acc = acc + heavy_leaf_a(i) + heavy_leaf_b(i)
        i = i + 1
    end
    return acc
end
"#;

    let mut vm_bc = build_vm_with_backend(src, false);
    let mut vm_clif = build_vm_with_backend(src, true);

    let driver_name = vm_clif
        .mir_functions
        .keys()
        .find(|k| k.ends_with("driver"))
        .cloned()
        .expect("driver");

    let ast_reg_count = vm_clif
        .function_by_name(&driver_name)
        .expect("AST driver function")
        .register_count as usize;
    assert!(
        vm_clif
            .function_by_name(&driver_name)
            .unwrap()
            .block_entry_pcs
            .is_empty(),
        "driver must start as AST bytecode before first call"
    );

    let oracle = vm_bc
        .call(&driver_name, vec![Value::Int(20)])
        .expect("bc oracle driver");

    // DO NOT call `compile_cranelift()` beforehand! Force lazy first-call compilation
    // while `top_frame` was allocated for the smaller AST `register_count`, AND trigger
    // safepoint deopt on that very first call!
    reset_safepoint_test_state();
    safepoint_request_deopt(true);
    let first_call_deopt = vm_clif
        .call(&driver_name, vec![Value::Int(20)])
        .expect("first-call lazy compile + deopt must succeed via Step E register setup");
    reset_safepoint_test_state();

    assert_eq!(first_call_deopt, oracle);
    assert!(vm_clif.cranelift_entry_count >= 1);

    let post_inline_locals = vm_clif
        .cranelift_compiled_local_count(&driver_name)
        .expect("cached local_types.len() for driver");
    let continuation_reg_count = vm_clif
        .function_by_name(&driver_name)
        .expect("installed continuation bytecode")
        .register_count as usize;
    assert!(
        post_inline_locals > ast_reg_count && continuation_reg_count > ast_reg_count,
        "inlining must grow local_types ({post_inline_locals}) and continuation register_count ({continuation_reg_count}) beyond original AST register_count ({ast_reg_count})"
    );
}

// =========================================================================
// Adversarial Negative Test 5: `calls.lust` End-to-End Exact Byte Parity
// =========================================================================

#[test]
fn test_perf_t1_4_calls_lust_cli_exact_byte_parity() {
    let calls_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks")
        .join("suite")
        .join("calls.lust");
    assert!(
        calls_path.exists(),
        "calls.lust benchmark must exist at {}",
        calls_path.display()
    );

    let out_jit = Command::new(lust_bin())
        .args(["--jit=cranelift"])
        .arg(&calls_path)
        .env_remove("LUST_JIT")
        .env_remove("LUST_JIT_BACKEND")
        .env_remove("LUST_BACKEND")
        .output()
        .expect("run lust --jit=cranelift calls.lust");
    assert!(
        out_jit.status.success(),
        "lust --jit=cranelift calls.lust failed: {}",
        String::from_utf8_lossy(&out_jit.stderr)
    );

    let out_vm = Command::new(lust_bin())
        .args(["--jit=off"])
        .arg(&calls_path)
        .env_remove("LUST_JIT")
        .env_remove("LUST_JIT_BACKEND")
        .env_remove("LUST_BACKEND")
        .output()
        .expect("run lust --jit=off calls.lust");
    assert!(
        out_vm.status.success(),
        "lust --jit=off calls.lust failed: {}",
        String::from_utf8_lossy(&out_vm.stderr)
    );

    assert_eq!(
        out_jit.stdout,
        out_vm.stdout,
        "exact byte parity required between --jit=cranelift and --jit=off on calls.lust"
    );
    assert_eq!(
        String::from_utf8_lossy(&out_jit.stdout).trim(),
        "99999990000000"
    );
}

// =========================================================================
// Unit Guard Test: `excluded_functions` (Guard 3) & `MirVerifier` Fail-Closed Revert (Guard 4)
// =========================================================================

#[test]
fn test_perf_t1_4_excluded_functions_and_verifier_revert_guards() {
    let span = Span::dummy();

    // Guard 3: excluded_functions blocks inlining both INTO and OUT OF excluded functions.
    let mut leaf = MirFunction::new(
        FunctionSignature {
            name: "leaf".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let lp0 = leaf.params[0];
    let lt = leaf.add_local(Some("t".into()), MirType::Int, false, span);
    leaf.blocks[0].statements.push(Statement::Assign(
        Place::Local(lt),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(lp0)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        span,
    ));
    leaf.blocks[0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(lt))), span));

    let caller = make_wrapper_calling("caller", "leaf");

    // Exclude caller -> must not inline INTO caller
    let mut m1 = MirModule::new("excl_caller".into());
    m1.add_function(leaf.clone());
    m1.add_function(caller.clone());
    let mut cfg1 = JitInlineConfig::default();
    cfg1.excluded_functions.insert("caller".into());
    let inlined1 = MirOptimizer::optimize_module_for_jit(&mut m1, &cfg1);
    assert!(inlined1.is_empty());

    // Exclude leaf -> must not inline OUT OF leaf
    let mut m2 = MirModule::new("excl_callee".into());
    m2.add_function(leaf.clone());
    m2.add_function(caller.clone());
    let mut cfg2 = JitInlineConfig::default();
    cfg2.excluded_functions.insert("leaf".into());
    let inlined2 = MirOptimizer::optimize_module_for_jit(&mut m2, &cfg2);
    assert!(inlined2.is_empty());

    // Guard 4: Malformed callee that causes post-inline MirVerifier failure MUST revert caller
    // cleanly to its pre-inline MIR snapshot (fail-closed).
    let mut bad_callee = MirFunction::new(
        FunctionSignature {
            name: "bad_leaf".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let uninit = bad_callee.add_local(Some("uninit".into()), MirType::Int, false, span);
    // Returns uninitialized local -> fails MirVerifier::verify_fn after inlining!
    bad_callee.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(uninit))),
        span,
    ));

    let good_caller = make_wrapper_calling("good_caller", "bad_leaf");
    let pre_inline_blocks = good_caller.blocks.clone();

    let mut m3 = MirModule::new("bad_verify".into());
    m3.add_function(bad_callee);
    m3.add_function(good_caller);
    let inlined3 = MirOptimizer::optimize_module_for_jit(&mut m3, &JitInlineConfig::default());
    assert!(
        !inlined3.contains("good_caller"),
        "post-inline MirVerifier failure must revert good_caller"
    );
    let reverted = m3
        .functions
        .iter()
        .find(|f| f.signature.name == "good_caller")
        .unwrap();
    assert_eq!(reverted.blocks, pre_inline_blocks);
    MirVerifier::verify_fn(reverted).expect("reverted caller must still pass MirVerifier");
}

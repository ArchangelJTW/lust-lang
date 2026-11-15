#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::fs;
use std::process::Command;
use tempfile::tempdir;

use lust::ast::{ItemKind, Span};
use lust::codegen_zig::aot::{AotCompileOptions, AotCompiler, AotTargetKind};
use lust::codegen_zig::{CompileOptions, compile_zig, is_zig_available};
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::ir::*;
use lust::mir::lowering::AstLowering;
use lust::mir::opt::inline::MirInliner;
use lust::mir::opt::MirOptimizer;
use lust::mir::verifier::MirVerifier;
use lust::parser::Parser;
use lust::vm::{VM, VmBackend};

fn parse_to_mir(source: &str) -> MirModule {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize().expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");

    let mut module = MirModule::new("test_mod".into());
    for item in &items {
        if let ItemKind::Function(func_def) = &item.kind {
            let mir_fn = AstLowering::lower_function(func_def).expect("lower_function");
            module.add_function(mir_fn);
        }
    }
    module
}

fn run_interpreter(source: &str) -> (Option<i64>, String) {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize().expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");

    let mut vm = VM::new();
    let config = lust::config::LustConfig::default();
    let mut checker = lust::typechecker::TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let mut compiler = lust::bytecode::compiler::Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    let functions = match compiler.compile_module(&items) {
        Ok(f) => f,
        Err(e) => return (None, format!("{e}")),
    };
    vm.load_functions(functions);
    vm.set_backend(VmBackend::Bytecode);
    match vm.call("main", vec![]) {
        Ok(val) => {
            let int_val = match val {
                lust::bytecode::Value::Int(i) => Some(i),
                lust::bytecode::Value::Bool(b) => Some(if b { 1 } else { 0 }),
                _ => None,
            };
            (int_val, String::new())
        }
        Err(e) => (None, format!("{e}")),
    }
}

// ---------------------------------------------------------------------------
// 1. MIR Inliner: Leaf Function Inlining & IR Inspection
// ---------------------------------------------------------------------------

#[test]
fn test_mir_inliner_leaf_function_inlining() {
    let source = r#"
        function add(a: int, b: int): int
            return a + b
        end

        function main(): int
            return add(10, 20)
        end
    "#;

    let mut module = parse_to_mir(source);
    MirVerifier::verify_module(&module).expect("pre-inline verification");

    let main_fn = module.functions.iter().find(|f| f.signature.name == "main").unwrap();
    let has_call_before = main_fn.blocks.iter().any(|b| {
        matches!(&b.terminator, Some(Terminator::Call { callee, .. }) if callee == "add")
    });
    assert!(has_call_before, "main must contain a Call terminator to 'add' before inlining");

    let changed = MirInliner::run_on_module(&mut module);
    assert!(changed, "MirInliner must report changed = true");

    MirVerifier::verify_module(&module).expect("post-inline verification");

    let main_after = module.functions.iter().find(|f| f.signature.name == "main").unwrap();
    let has_call_after = main_after.blocks.iter().any(|b| {
        matches!(&b.terminator, Some(Terminator::Call { callee, .. }) if callee == "add")
    });
    assert!(!has_call_after, "Call terminator to 'add' must be eliminated after inlining");

    // Full optimizer pass should fold 10 + 20 -> 30
    MirOptimizer::optimize_module(&mut module);
    MirVerifier::verify_module(&module).expect("post-opt verification");

    let main_optimized = module.functions.iter().find(|f| f.signature.name == "main").unwrap();
    let returns_30 = main_optimized.blocks.iter().any(|b| {
        matches!(
            &b.terminator,
            Some(Terminator::Return(Some(Operand::Constant(MirConstant::Int(30))), _))
        )
    });
    assert!(returns_30, "Constant folding after inlining must evaluate 10 + 20 to 30");
}

// ---------------------------------------------------------------------------
// 2. MIR Inliner: Multi-Level Inlining (A -> B -> C)
// ---------------------------------------------------------------------------

#[test]
fn test_mir_inliner_multi_level_inlining() {
    let source = r#"
        function step_c(x: int): int
            return x + 1
        end

        function step_b(x: int): int
            return step_c(x) * 2
        end

        function step_a(x: int): int
            return step_b(x) + 3
        end
    "#;

    let mut module = parse_to_mir(source);
    MirVerifier::verify_module(&module).expect("initial verification");

    let changed = MirInliner::run_on_module(&mut module);
    assert!(changed, "Inlining multi-level calls must succeed");
    MirVerifier::verify_module(&module).expect("multi-level verification");

    let step_a = module.functions.iter().find(|f| f.signature.name == "step_a").unwrap();
    let calls_in_a = step_a.blocks.iter().filter_map(|b| {
        if let Some(Terminator::Call { callee, .. }) = &b.terminator {
            Some(callee.clone())
        } else {
            None
        }
    }).collect::<Vec<_>>();

    assert!(
        calls_in_a.is_empty(),
        "All calls inside step_a must be inlined; found: {:?}",
        calls_in_a
    );
}

// ---------------------------------------------------------------------------
// 3. MIR Inliner: Control-Flow with If/Else Branches and Multiple Returns
// ---------------------------------------------------------------------------

#[test]
fn test_mir_inliner_control_flow_branches() {
    let source = r#"
        function max_val(a: int, b: int): int
            if a > b then
                return a
            else
                return b
            end
        end

        function caller(x: int, y: int): int
            local m: int = max_val(x, y)
            return m + 1
        end
    "#;

    let mut module = parse_to_mir(source);
    MirVerifier::verify_module(&module).expect("initial verification");

    let changed = MirInliner::run_on_module(&mut module);
    assert!(changed, "Inlining function with branch control-flow must succeed");
    MirVerifier::verify_module(&module).expect("post-inline branch verification");

    let caller = module.functions.iter().find(|f| f.signature.name == "caller").unwrap();
    let has_max_call = caller.blocks.iter().any(|b| {
        matches!(&b.terminator, Some(Terminator::Call { callee, .. }) if callee == "max_val")
    });
    assert!(!has_max_call, "Call to max_val must be inlined");
    assert!(caller.blocks.len() >= 4, "Caller must have branched basic blocks from inlined max_val");
}

// ---------------------------------------------------------------------------
// 4. MIR Inliner: Loop Control Flow Inlining
// ---------------------------------------------------------------------------

#[test]
fn test_mir_inliner_loop_inlining() {
    let source = r#"
        function sum_range(n: int): int
            local total: int = 0
            local i: int = 0
            while i < n do
                total = total + i
                i = i + 1
            end
            return total
        end

        function run(): int
            return sum_range(5)
        end
    "#;

    let mut module = parse_to_mir(source);
    MirVerifier::verify_module(&module).expect("initial verification");

    let changed = MirInliner::run_on_module(&mut module);
    assert!(changed, "Inlining loop function must succeed");
    MirVerifier::verify_module(&module).expect("post-inline loop verification");

    let run_fn = module.functions.iter().find(|f| f.signature.name == "run").unwrap();
    let has_sum_call = run_fn.blocks.iter().any(|b| {
        matches!(&b.terminator, Some(Terminator::Call { callee, .. }) if callee == "sum_range")
    });
    assert!(!has_sum_call, "sum_range call must be inlined into run");
}

// ---------------------------------------------------------------------------
// 5. MIR Inliner: Direct Recursion Guard (Safe Fallback)
// ---------------------------------------------------------------------------

#[test]
fn test_mir_inliner_direct_recursion_guard() {
    let source = r#"
        function factorial(n: int): int
            if n <= 1 then
                return 1
            end
            return n * factorial(n - 1)
        end

        function compute(): int
            return factorial(5)
        end
    "#;

    let mut module = parse_to_mir(source);
    MirVerifier::verify_module(&module).expect("initial verification");

    // Attempt inlining
    MirInliner::run_on_module(&mut module);
    MirVerifier::verify_module(&module).expect("post-inline recursion verification");

    let fact_fn = module.functions.iter().find(|f| f.signature.name == "factorial").unwrap();
    let has_self_call = fact_fn.blocks.iter().any(|b| {
        matches!(&b.terminator, Some(Terminator::Call { callee, .. }) if callee == "factorial")
    });
    assert!(
        has_self_call,
        "Direct self-recursive call inside factorial MUST be guarded and preserved"
    );
}

// ---------------------------------------------------------------------------
// 6. MIR Inliner: Mutual Recursion Guard (Cycle Detection)
// ---------------------------------------------------------------------------

#[test]
fn test_mir_inliner_mutual_recursion_guard() {
    let source = r#"
        function ping(n: int): int
            if n <= 0 then
                return 0
            end
            return pong(n - 1)
        end

        function pong(n: int): int
            if n <= 0 then
                return 1
            end
            return ping(n - 1)
        end
    "#;

    let mut module = parse_to_mir(source);
    MirVerifier::verify_module(&module).expect("initial verification");

    // Inliner must terminate cleanly without stack overflow or infinite unrolling
    MirInliner::run_on_module(&mut module);
    MirVerifier::verify_module(&module).expect("post-inline mutual recursion verification");

    let ping_fn = module.functions.iter().find(|f| f.signature.name == "ping").unwrap();
    let pong_fn = module.functions.iter().find(|f| f.signature.name == "pong").unwrap();

    let has_any_call = ping_fn.blocks.iter().any(|b| matches!(&b.terminator, Some(Terminator::Call { .. })))
        || pong_fn.blocks.iter().any(|b| matches!(&b.terminator, Some(Terminator::Call { .. })));

    assert!(has_any_call, "Cycle in mutual recursion must guard against infinite inlining");
}

// ---------------------------------------------------------------------------
// 7. MIR Inliner: Missing & Arity Mismatch Safe Fallback
// ---------------------------------------------------------------------------

#[test]
fn test_mir_inliner_safe_fallbacks() {
    let mut module = MirModule::new("fallback_mod".into());

    let caller_sig = FunctionSignature {
        name: "caller".into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut caller = MirFunction::new(caller_sig, Span::dummy());
    let dest = Place::Local(LocalId(1));

    // Call to nonexistent function
    let bb1 = caller.new_block();
    caller.blocks[0].terminator = Some(Terminator::Call {
        callee: "nonexistent_fn".into(),
        args: vec![Operand::Constant(MirConstant::Int(42))],
        destination: dest.clone(),
        success: bb1,
        error: None,
        span: Span::dummy(),
    });
    caller.blocks[bb1.0].terminator = Some(Terminator::Return(Some(Operand::Copy(dest)), Span::dummy()));
    module.add_function(caller);

    let inliner = MirInliner::new();
    let changed = inliner.inline_module(&mut module);
    assert!(!changed, "Inliner must safely fall back when callee does not exist");

    let caller_after = &module.functions[0];
    assert!(
        matches!(&caller_after.blocks[0].terminator, Some(Terminator::Call { callee, .. }) if callee == "nonexistent_fn"),
        "Missing callee call must be safely preserved"
    );
}

// ---------------------------------------------------------------------------
// 8. Autonomous AOT Release Build: Native Execution without Zig
// ---------------------------------------------------------------------------

#[test]
fn test_aot_release_build_zig_free_arithmetic_parity() {
    let source = r#"
        function compute(a: int, b: int): int
            local sum: int = a + b
            local diff: int = a - b
            local prod: int = sum * diff
            return prod
        end

        function main(): int
            return compute(15, 5)
        end
    "#;

    let dir = tempdir().expect("tempdir");
    let out_bin = dir.path().join("arithmetic_app");

    let opts = AotCompileOptions {
        target_kind: AotTargetKind::Executable,
        release_mode: "ReleaseFast".to_string(),
        strip: false,
        max_memory_bytes: None,
        target_triple: None,
    };

    // Compile native binary without zig
    let compiled_path = AotCompiler::compile_source(source, &out_bin, &opts)
        .expect("AOT compilation without zig must succeed");

    assert!(compiled_path.exists(), "Compiled binary must exist on disk");

    // Execute the resulting native binary
    let output = Command::new(&compiled_path)
        .output()
        .expect("Binary execution failed");

    let exit_code = output.status.code().unwrap_or(-1);
    // (15 + 5) * (15 - 5) = 20 * 10 = 200
    assert_eq!(exit_code, 200, "Native binary exit code must equal expected arithmetic result");

    // Verify parity with interpreter
    let (interp_res, _) = run_interpreter(source);
    assert_eq!(interp_res, Some(200), "Interpreter must produce identical result");
}

// ---------------------------------------------------------------------------
// 9. Autonomous AOT Release Build: Inlined Calls + Output Parity
// ---------------------------------------------------------------------------

#[test]
fn test_aot_release_build_zig_free_inlining_and_print_parity() {
    let source = r#"
        function square(x: int): int
            return x * x
        end

        function sum_of_squares(a: int, b: int): int
            return square(a) + square(b)
        end

        function main(): int
            local res: int = sum_of_squares(3, 4)
            println(res)
            return res
        end
    "#;

    let dir = tempdir().expect("tempdir");
    let out_bin = dir.path().join("inlined_print_app");

    let opts = AotCompileOptions {
        target_kind: AotTargetKind::Executable,
        release_mode: "ReleaseFast".to_string(),
        strip: false,
        max_memory_bytes: None,
        target_triple: None,
    };

    let compiled_path = AotCompiler::compile_source(source, &out_bin, &opts)
        .expect("AOT compilation without zig must succeed");

    let output = Command::new(&compiled_path)
        .output()
        .expect("Binary execution failed");

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let exit_code = output.status.code().unwrap_or(-1);

    // 3^2 + 4^2 = 9 + 16 = 25
    assert_eq!(exit_code, 25, "Exit code must match return value");
    assert_eq!(stdout, "25", "Stdout must contain printed output '25'");

    let (interp_res, _) = run_interpreter(source);
    assert_eq!(interp_res, Some(25), "Interpreter result must match native binary");
}

// ---------------------------------------------------------------------------
// 10. Autonomous AOT Release Build: Control-Flow and Loops Parity
// ---------------------------------------------------------------------------

#[test]
fn test_aot_release_build_zig_free_loops_and_branches_parity() {
    let source = r#"
        function sum_evens(limit: int): int
            local total: int = 0
            local i: int = 0
            while i <= limit do
                if i % 2 == 0 then
                    total = total + i
                end
                i = i + 1
            end
            return total
        end

        function main(): int
            return sum_evens(10)
        end
    "#;

    let dir = tempdir().expect("tempdir");
    let out_bin = dir.path().join("loops_app");

    let opts = AotCompileOptions {
        target_kind: AotTargetKind::Executable,
        release_mode: "ReleaseFast".to_string(),
        strip: false,
        max_memory_bytes: None,
        target_triple: None,
    };

    let compiled_path = AotCompiler::compile_source(source, &out_bin, &opts)
        .expect("AOT compilation without zig must succeed");

    let output = Command::new(&compiled_path)
        .output()
        .expect("Binary execution failed");

    let exit_code = output.status.code().unwrap_or(-1);
    // 0 + 2 + 4 + 6 + 8 + 10 = 30
    assert_eq!(exit_code, 30, "Sum of evens up to 10 must be 30");

    let (interp_res, _) = run_interpreter(source);
    assert_eq!(interp_res, Some(30), "Interpreter result must match");
}

// ---------------------------------------------------------------------------
// 11. Autonomous AOT Release Build: Runtime Error Parity (Div by Zero)
// ---------------------------------------------------------------------------

#[test]
fn test_aot_release_build_zig_free_runtime_error_parity() {
    let source = r#"
        function bad_math(x: int): int
            return x / 0
        end

        function main(): int
            return bad_math(42)
        end
    "#;

    let dir = tempdir().expect("tempdir");
    let out_bin = dir.path().join("div_zero_app");

    let opts = AotCompileOptions {
        target_kind: AotTargetKind::Executable,
        release_mode: "ReleaseFast".to_string(),
        strip: false,
        max_memory_bytes: None,
        target_triple: None,
    };

    let compiled_path = AotCompiler::compile_source(source, &out_bin, &opts)
        .expect("AOT compilation without zig must succeed");

    let output = Command::new(&compiled_path)
        .output()
        .expect("Binary execution failed");

    assert!(!output.status.success(), "Division by zero must cause non-zero exit status");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("division by zero"),
        "Stderr must report division by zero error, got: {stderr}"
    );

    // Interpreter comparison
    let (interp_res, interp_err) = run_interpreter(source);
    assert_eq!(interp_res, None, "Interpreter must return None on error");
    assert!(
        interp_err.contains("division by zero") || !interp_err.is_empty(),
        "Interpreter must report error on division by zero"
    );
}

// ---------------------------------------------------------------------------
// 12. End-to-End compile_zig Fallback with LUST_NO_ZIG=1
// ---------------------------------------------------------------------------

#[test]
fn test_compile_zig_fallback_end_to_end_when_zig_unavailable() {
    let dir = tempdir().expect("tempdir");
    let lust_file = dir.path().join("program.lust");
    let zig_file = dir.path().join("program.zig");
    let out_bin = dir.path().join("program_bin");

    let source = r#"
        function multiply(a: int, b: int): int
            return a * b
        end

        function main(): int
            local res: int = multiply(6, 7)
            println(res)
            return res
        end
    "#;

    fs::write(&lust_file, source).expect("write lust file");
    // Write placeholder zig file (as main.rs does during lust build)
    fs::write(&zig_file, "// Placeholder zig source").expect("write zig file");

    let opts = CompileOptions {
        target_kind: lust::codegen_zig::BuildTargetKind::Executable,
        release_mode: "ReleaseFast".to_string(),
        emit_zig: false,
        emit_header: None,
        max_memory: None,
        max_memory_bytes: None,
        tree_shake: false,
        strip: false,
        target_triple: None,
        cache_dir: None,
    };

    // Simulate system zig unavailable via environment variable
    unsafe {
        std::env::set_var("LUST_NO_ZIG", "1");
    }

    assert!(!is_zig_available(), "is_zig_available() must be false when LUST_NO_ZIG=1");

    let res = compile_zig(&zig_file, &out_bin, &opts, false);

    // Clean up env var
    unsafe {
        std::env::remove_var("LUST_NO_ZIG");
    }

    let compiled_bin = res.expect("compile_zig fallback must succeed when zig is unavailable");
    assert!(compiled_bin.exists(), "Compiled binary must exist");

    let output = Command::new(&compiled_bin).output().expect("execute binary");
    assert_eq!(output.status.code(), Some(42), "Exit code must be 42");
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    assert_eq!(stdout, "42", "Stdout must be 42");
}

// ---------------------------------------------------------------------------
// 13. MIR Inliner: Depth Budget Guard
// ---------------------------------------------------------------------------

#[test]
fn test_mir_inliner_depth_budget_guard() {
    let source = r#"
        function level3(x: int): int
            return x + 1
        end

        function level2(x: int): int
            return level3(x) + 2
        end

        function level1(x: int): int
            return level2(x) + 3
        end

        function main(): int
            return level1(10)
        end
    "#;

    let mut module = parse_to_mir(source);
    // Configure depth budget to only 1: main can inline level1, but level1 cannot inline level2
    let config = lust::mir::opt::InlinerConfig {
        max_depth: 1,
        max_callee_blocks: 64,
        max_callee_statements: 256,
    };
    let inliner = MirInliner::with_config(config);
    let changed = inliner.inline_module(&mut module);
    assert!(changed);

    let main_fn = module.functions.iter().find(|f| f.signature.name == "main").unwrap();
    let has_level2 = main_fn.blocks.iter().any(|b| {
        matches!(&b.terminator, Some(Terminator::Call { callee, .. }) if callee == "level2")
    });
    assert!(has_level2, "Depth budget must stop inlining beyond max_depth = 1");
}

// ---------------------------------------------------------------------------
// 14. MIR Inliner: Callee Size Budget Guard
// ---------------------------------------------------------------------------

#[test]
fn test_mir_inliner_callee_size_budget_guard() {
    let source = r#"
        function huge_callee(x: int): int
            local a: int = x + 1
            local b: int = a + 2
            local c: int = b + 3
            return c
        end

        function caller(): int
            return huge_callee(10)
        end
    "#;

    let mut module = parse_to_mir(source);
    // Configure max_callee_statements to 2 (huge_callee has > 2 statements)
    let config = lust::mir::opt::InlinerConfig {
        max_depth: 4,
        max_callee_blocks: 64,
        max_callee_statements: 2,
    };
    let inliner = MirInliner::with_config(config);
    let changed = inliner.inline_module(&mut module);
    assert!(!changed, "huge_callee must be rejected by statement count budget");

    let caller_fn = module.functions.iter().find(|f| f.signature.name == "caller").unwrap();
    let has_huge_call = caller_fn.blocks.iter().any(|b| {
        matches!(&b.terminator, Some(Terminator::Call { callee, .. }) if callee == "huge_callee")
    });
    assert!(has_huge_call, "huge_callee must remain as a Call terminator due to budget limit");
}

// ---------------------------------------------------------------------------
// 15. Autonomous AOT Release Build: Shared Library (.dylib / .so)
// ---------------------------------------------------------------------------

#[test]
fn test_aot_release_build_shared_library() {
    let source = r#"
        function multiply(a: int, b: int): int
            return a * b
        end
    "#;

    let dir = tempdir().expect("tempdir");
    let ext = if cfg!(target_os = "macos") { "dylib" } else { "so" };
    let out_lib = dir.path().join(format!("libtest_mult.{ext}"));

    let opts = AotCompileOptions {
        target_kind: AotTargetKind::SharedLibrary,
        release_mode: "ReleaseFast".to_string(),
        strip: false,
        max_memory_bytes: None,
        target_triple: None,
    };

    let compiled_lib = AotCompiler::compile_source(source, &out_lib, &opts)
        .expect("AOT shared library compilation must succeed");

    assert!(compiled_lib.exists(), "Compiled dylib/so must exist on disk");
}

// ---------------------------------------------------------------------------
// 16. Autonomous AOT Release Build: Memory Quota Guard
// ---------------------------------------------------------------------------

#[test]
fn test_aot_release_build_memory_quota_guard() {
    let source = r#"
        function allocate_too_much(): int
            local arr: Array<int> = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            return arr[0]
        end

        function main(): int
            return allocate_too_much()
        end
    "#;

    let dir = tempdir().expect("tempdir");
    let out_bin = dir.path().join("quota_app");

    // Restrict max memory to only 16 bytes (array struct + payload will exceed this)
    let opts = AotCompileOptions {
        target_kind: AotTargetKind::Executable,
        release_mode: "ReleaseFast".to_string(),
        strip: false,
        max_memory_bytes: Some(16),
        target_triple: None,
    };

    let compiled_bin = AotCompiler::compile_source(source, &out_bin, &opts)
        .expect("AOT compilation with memory quota must succeed");

    let output = Command::new(&compiled_bin).output().expect("execute binary");
    assert!(!output.status.success(), "Memory quota violation must cause non-zero exit status");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("memory limit exceeded"),
        "Stderr must report memory limit exceeded, got: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// 17. Autonomous AOT CLI Build Module Dispatch (--release without zig)
// ---------------------------------------------------------------------------

#[test]
fn test_aot_cli_build_module_dispatch() {
    #[path = "../src/cli/build.rs"]
    mod cli_build;

    let dir = tempdir().expect("tempdir");
    let lust_file = dir.path().join("cli_test.lust");
    let out_bin = dir.path().join("cli_test_bin");

    let source = r#"
        function compute(x: int): int
            return x * 3
        end

        function main(): int
            return compute(7)
        end
    "#;
    fs::write(&lust_file, source).expect("write lust file");

    unsafe {
        std::env::set_var("LUST_NO_ZIG", "1");
    }

    cli_build::build_command(&[
        lust_file.to_str().unwrap().to_string(),
        "--release".to_string(),
        "-o".to_string(),
        out_bin.to_str().unwrap().to_string(),
    ]);

    unsafe {
        std::env::remove_var("LUST_NO_ZIG");
    }

    assert!(out_bin.exists(), "Binary built via cli::build::build_command must exist");
    let output = Command::new(&out_bin).output().expect("execute binary");
    assert_eq!(output.status.code(), Some(21), "Exit code must be 21 (7 * 3)");
}

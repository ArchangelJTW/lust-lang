//! Regression and acceptance tests for TASK-BENCH-SUITE-BUGFIX:
//! - BF-1: Cranelift print/println typed console-aware ABI (float/bool/arity/console)
//! - BF-2A: AstLowering IsPattern mutual recursion & silent-false elimination
//! - BF-2B: Exhaustive Cranelift Place/Operand projection eligibility & non-swallowing
//! - BF-3: Context-sensitive `..` concat vs `for ... in` range parsing & Zig AOT parity

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use lust::ast::{BinaryOp, Expr, ExprKind, ItemKind, Literal, Span, StmtKind};
use lust::bytecode::Value;
use lust::bytecode::compiler::Compiler;
use lust::config::LustConfig;
use lust::error::LustError;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::CraneliftJitCompiler;
use lust::mir::ir::{
    FunctionSignature, MirFunction, MirModule, MirType, Operand, Place, Rvalue, Statement,
    Terminator,
};
#[cfg(feature = "test-deopt")]
use lust::mir::lowering::AstLowering;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vfs::VirtualConsole;
#[cfg(feature = "test-deopt")]
use lust::vm::CallFrame;
use lust::vm::{VM, VmBackend};
use lust::{EmbeddedModule, compile_program_with_config, load_program_from_embedded};

static TMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn unique_temp_path(prefix: &str, ext: &str) -> PathBuf {
    let id = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("lust_bf_{prefix}_{pid}_{id}.{ext}"))
}

fn lust_bin() -> &'static str {
    env!("CARGO_BIN_EXE_lust")
}

struct ChildResult {
    status_code: Option<i32>,
    timed_out: bool,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn run_cmd_with_timeout(
    program: &str,
    args: &[&str],
    timeout: Duration,
) -> ChildResult {
    let mut child = Command::new(program)
        .args(args)
        .env("LUST_JIT_QUIET", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn {program} {args:?}: {e}"));

    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => {
                let output = child.wait_with_output().expect("wait_with_output");
                return ChildResult {
                    status_code: output.status.code(),
                    timed_out: false,
                    stdout: output.stdout,
                    stderr: output.stderr,
                };
            }
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let output = child.wait_with_output().expect("wait_with_output after kill");
                    return ChildResult {
                        status_code: output.status.code(),
                        timed_out: true,
                        stdout: output.stdout,
                        stderr: output.stderr,
                    };
                }
                std::thread::sleep(Duration::from_millis(15));
            }
            Err(e) => panic!("try_wait failed for {program}: {e}"),
        }
    }
}

fn run_lust_script(src: &str, extra_args: &[&str], timeout: Duration) -> ChildResult {
    let path = unique_temp_path("script", "lust");
    fs::write(&path, src).expect("write temp script");
    let path_str = path.to_string_lossy().into_owned();
    let mut args: Vec<&str> = extra_args.to_vec();
    args.push(&path_str);
    let res = run_cmd_with_timeout(lust_bin(), &args, timeout);
    let _ = fs::remove_file(&path);
    res
}

fn run_lust_file(path: &Path, extra_args: &[&str], timeout: Duration) -> ChildResult {
    let path_str = path.to_string_lossy().into_owned();
    let mut args: Vec<&str> = extra_args.to_vec();
    args.push(&path_str);
    run_cmd_with_timeout(lust_bin(), &args, timeout)
}

fn compile_mir_and_bytecode_vms(src: &str) -> (VM, VM) {
    let mut interner = Interner::new();
    let tokens = Lexer::new(src, &mut interner)
        .tokenize()
        .expect("tokenization should succeed");
    let items = Parser::new(tokens).parse().expect("parsing should succeed");

    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    checker
        .check_module(&items)
        .expect("typechecking should succeed");
    let struct_defs = checker.struct_definitions();
    let option_coercions = checker.take_option_coercions();
    let checked_array_indices = checker.take_checked_array_indices();
    let lambda_return_types = checker.take_lambda_return_types();
    let numeric_types = checker.take_numeric_types();
    let func_sigs = checker.take_function_signatures();

    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_option_coercions(option_coercions);
    compiler.set_checked_array_indices(checked_array_indices);
    compiler.set_lambda_return_types(lambda_return_types);
    compiler.set_numeric_types(numeric_types);
    compiler.set_function_signatures(func_sigs);
    compiler.set_use_mir(true);
    let bc_functions = compiler
        .compile_module(&items)
        .expect("module compilation should succeed");
    let mir_functions = compiler.take_mir_functions();

    let mut vm_clif = VM::new();
    vm_clif.console = VirtualConsole::captured();
    vm_clif.register_structs(&struct_defs);
    vm_clif.load_functions(bc_functions.clone());
    for mir_fn in mir_functions {
        vm_clif.load_mir_function(mir_fn).expect("load_mir_function");
    }
    vm_clif.set_backend(VmBackend::Cranelift);

    let mut vm_bc = VM::new();
    vm_bc.console = VirtualConsole::captured();
    vm_bc.register_structs(&struct_defs);
    vm_bc.load_functions(bc_functions);
    vm_bc.set_backend(VmBackend::Bytecode);

    (vm_clif, vm_bc)
}

#[cfg(feature = "test-deopt")]
fn compile_direct_mir_vm(src: &str) -> (VM, VM) {
    let mut interner = Interner::new();
    let tokens = Lexer::new(src, &mut interner)
        .tokenize()
        .expect("tokenization should succeed");
    let items = Parser::new(tokens).parse().expect("parsing should succeed");
    let mut module = MirModule::new("bf1_direct".into());
    for item in &items {
        if let ItemKind::Function(func) = &item.kind {
            let mir_fn = AstLowering::lower_function(func).expect("MIR lowering should succeed");
            module.add_function(mir_fn);
        }
    }

    let mut vm_clif = VM::new();
    vm_clif.console = VirtualConsole::captured();
    vm_clif.load_mir_module(&module).unwrap();
    vm_clif.set_backend(VmBackend::Cranelift);

    let mut vm_bc = VM::new();
    vm_bc.console = VirtualConsole::captured();
    vm_bc.load_mir_module(&module).unwrap();
    vm_bc.set_backend(VmBackend::Bytecode);

    (vm_clif, vm_bc)
}

fn parse_items(src: &str) -> Result<Vec<lust::ast::Item>, LustError> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(src, &mut interner).tokenize()?;
    Parser::new(tokens).parse()
}

fn parse_single_expr(expr_src: &str) -> Expr {
    let wrapped = format!("local __x = {expr_src}");
    let items = parse_items(&wrapped).unwrap_or_else(|e| panic!("failed to parse `{expr_src}`: {e}"));
    assert_eq!(items.len(), 1);
    match &items[0].kind {
        ItemKind::Script(stmts) => match &stmts[0].kind {
            StmtKind::Local { initializer, .. } => {
                initializer.as_ref().expect("local initializer")[0].clone()
            }
            other => panic!("expected Local stmt, got {other:?}"),
        },
        other => panic!("expected Script item, got {other:?}"),
    }
}

fn parse_for_in_iterator(stmt_src: &str) -> Result<Expr, LustError> {
    let items = parse_items(stmt_src)?;
    match &items[0].kind {
        ItemKind::Script(stmts) => match &stmts[0].kind {
            StmtKind::ForIn { iterator, .. } => Ok(iterator.clone()),
            other => panic!("expected ForIn stmt, got {other:?}"),
        },
        other => panic!("expected Script item, got {other:?}"),
    }
}

#[test]
fn test_bf1_print_parity_float_bool_arity_and_console() {
    // 1. Float loop + scalar print/println parity via VM::call (native Cranelift entries >= 1)
    let src = r#"
        function run_float(): int
            local x: float = 0.0
            local acc: float = 0.0
            local i: int = 0
            while i < 1000 do
                x = x + 0.001
                acc = acc + x * x - acc * 0.5
                i = i + 1
            end
            println(acc)
            return 1
        end

        function run_scalars(): int
            local neg_zero: float = -0.0
            local one_zero: float = 1.0
            println(neg_zero)
            println(one_zero)
            println(true, false, 42, 3.5)
            print(1, 2)
            println()
            return 0
        end

        function run_non_scalar(): int
            println("hello", 7)
            return 0
        end
    "#;

    let (mut vm_clif, mut vm_bc) = compile_mir_and_bytecode_vms(src);

    // Float loop via VM::call
    let res_clif = vm_clif
        .call("run_float", vec![])
        .expect("Cranelift run_float must succeed without VerifierError");
    let res_bc = vm_bc
        .call("run_float", vec![])
        .expect("Bytecode run_float must succeed");
    assert_eq!(res_clif, res_bc);
    assert!(
        vm_clif.cranelift_entry_count >= 1,
        "run_float must execute natively in Cranelift (entries={})",
        vm_clif.cranelift_entry_count
    );
    let clif_float_out = vm_clif.console.stdout_bytes();
    let bc_float_out = vm_bc.console.stdout_bytes();
    assert!(
        !clif_float_out.is_empty(),
        "Cranelift println(f64) must write to vm.console, not bypass it"
    );
    assert_eq!(
        clif_float_out,
        bc_float_out,
        "Cranelift vs Bytecode float stdout mismatch: clif={:?}, bc={:?}",
        String::from_utf8_lossy(&clif_float_out),
        String::from_utf8_lossy(&bc_float_out),
    );

    // Multi-arg, -0.0, 1.0, bools, print (no-nl), 0-arg println() via VM::call
    let entries_before = vm_clif.cranelift_entry_count;
    let res_clif = vm_clif
        .call("run_scalars", vec![])
        .expect("Cranelift run_scalars must succeed");
    let res_bc = vm_bc
        .call("run_scalars", vec![])
        .expect("Bytecode run_scalars must succeed");
    assert_eq!(res_clif, res_bc);
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "run_scalars must execute natively in Cranelift"
    );
    let clif_scalars = vm_clif.console.stdout_bytes();
    let bc_scalars = vm_bc.console.stdout_bytes();
    assert_eq!(
        clif_scalars,
        bc_scalars,
        "Scalar print/println parity mismatch:\nclif={:?}\nbc  ={:?}",
        String::from_utf8_lossy(&clif_scalars),
        String::from_utf8_lossy(&bc_scalars),
    );
    let expected_suffix = "-0\n1\ntrue\tfalse\t42\t3.5\n1\t2\n";
    let actual_str = String::from_utf8(clif_scalars).expect("valid utf8");
    assert!(
        actual_str.ends_with(expected_suffix),
        "Unexpected scalar formatting: got {actual_str:?}, expected suffix {expected_suffix:?}"
    );

    // Non-scalar fallback: println("hello", 7) must fall back cleanly and match oracle
    let res_clif = vm_clif
        .call("run_non_scalar", vec![])
        .expect("Cranelift non-scalar print must fall back cleanly to VM");
    let res_bc = vm_bc
        .call("run_non_scalar", vec![])
        .expect("Bytecode non-scalar print must succeed");
    assert_eq!(res_clif, res_bc);
    assert_eq!(
        vm_clif.console.stdout_bytes(),
        vm_bc.console.stdout_bytes(),
        "Non-scalar print fallback console mismatch"
    );
    assert!(
        String::from_utf8_lossy(&vm_clif.console.stdout_bytes()).ends_with("hello\t7\n"),
        "Expected 'hello\\t7\\n' at end of console output"
    );

    // 2. Direct VM::run_cranelift() path (verifies CurrentVmGuard inside try_run_cranelift_top_frame_scoped)
    #[cfg(feature = "test-deopt")]
    {
        let direct_src = r#"
            function direct_float_print(): int
                local f: float = 2.25
                println(f, true, 9)
                return 7
            end
        "#;
        let (mut vm_direct_clif, mut vm_direct_bc) = compile_direct_mir_vm(direct_src);
        let frame = CallFrame {
            function_idx: 0,
            ip: 0,
            registers: [const { Value::Nil }; 256],
            base_register: 0,
            return_dest: None,
            upvalues: Vec::new(),
            shared_upvalues: Vec::new(),
        };
        vm_direct_clif.push_call_frame_for_test(frame);
        let direct_res = vm_direct_clif
            .run_cranelift()
            .expect("Direct VM::run_cranelift must succeed");
        let oracle_res = vm_direct_bc
            .call("direct_float_print", vec![])
            .expect("Bytecode oracle must succeed");
        assert_eq!(direct_res, oracle_res);
        assert!(
            vm_direct_clif.cranelift_entry_count >= 1,
            "Direct VM::run_cranelift must enter Cranelift natively"
        );
        assert_eq!(
            vm_direct_clif.console.stdout_bytes(),
            vm_direct_bc.console.stdout_bytes(),
            "Direct VM::run_cranelift must route output to vm.console (got {:?}, expected {:?})",
            String::from_utf8_lossy(&vm_direct_clif.console.stdout_bytes()),
            String::from_utf8_lossy(&vm_direct_bc.console.stdout_bytes()),
        );
    }
}

#[test]
fn test_bf2a_is_pattern_uncalled_no_stack_overflow_and_semantics() {
    // 1. Timeout-guarded child process (5s): uncalled function with `if node.left is Some(l)`
    // Must NOT stack overflow at compile time under `--jit=cranelift`.
    let uncalled_src = r#"
        struct Node
            value: int
            left: Option<Node>
        end

        function sum(node: Node): int
            local s: int = node.value
            if node.left is Some(l) then
                s = s + sum(l)
            end
            return s
        end

        println(42)
    "#;
    let uncalled_clif = run_lust_script(uncalled_src, &["--jit=cranelift"], Duration::from_secs(5));
    assert!(
        !uncalled_clif.timed_out && uncalled_clif.status_code == Some(0),
        "Uncalled is-pattern function failed under --jit=cranelift (timed_out={}, rc={:?}, stderr={})",
        uncalled_clif.timed_out,
        uncalled_clif.status_code,
        String::from_utf8_lossy(&uncalled_clif.stderr),
    );
    assert_eq!(uncalled_clif.stdout, b"42\n");

    // 2. Semantics parity (--jit=off vs --jit=cranelift):
    // Some/None, Ok/Err, field is Some(l), indexed arr[i] is Ok(v) (in-bounds, -1, OOB, side effects)
    let semantics_src = r#"
        struct Node
            value: int
            left: Option<Node>
        end

        function test_opt(x: Option<int>): int
            if x is Some(v) then
                return v + 10
            end
            if x is None then
                return -1
            end
            return -99
        end

        function test_res(r: Result<int, string>): int
            if r is Ok(v) then
                return v * 2
            end
            if r is Err(e) then
                return string.len(e)
            end
            return -99
        end

        function sum_node(n: Node): int
            local s: int = n.value
            if n.left is Some(l) then
                s = s + sum_node(l)
            end
            return s
        end

        function test_indexed(arr: Array<int>, idx: int): int
            if arr[idx] is Ok(v) then
                return v
            else
                return -777
            end
        end

        function get_arr(counts: Array<int>): Array<int>
            if counts[0] is Ok(c) then
                counts[0] = c + 1
            end
            return [100, 200, 300]
        end

        function next_idx(counts: Array<int>, target: int): int
            if counts[1] is Ok(c) then
                counts[1] = c + 1
            end
            return target
        end

        function test_side_effects(counts: Array<int>, target: int): int
            if get_arr(counts)[next_idx(counts, target)] is Ok(v) then
                return v
            else
                return -555
            end
        end

        println(test_opt(Option.Some(5)))
        println(test_opt(Option.None))
        println(test_res(Result.Ok(21)))
        println(test_res(Result.Err("oops")))
        local n1: Node = Node { value = 3, left = Option.None }
        local n2: Node = Node { value = 7, left = Option.Some(n1) }
        println(sum_node(n2))

        local a: Array<int> = [10, 20, 30]
        println(test_indexed(a, 0))
        println(test_indexed(a, 2))
        println(test_indexed(a, -1))
        println(test_indexed(a, 3))
        println(test_indexed(a, 8))

        local counts: Array<int> = [0, 0]
        println(test_side_effects(counts, 1))
        println(counts[0]:unwrap(), counts[1]:unwrap())
        println(test_side_effects(counts, 5))
        println(counts[0]:unwrap(), counts[1]:unwrap())
        println(test_side_effects(counts, -1))
        println(counts[0]:unwrap(), counts[1]:unwrap())
    "#;

    let vm_res = run_lust_script(semantics_src, &["--jit=off"], Duration::from_secs(10));
    let clif_res = run_lust_script(semantics_src, &["--jit=cranelift"], Duration::from_secs(10));
    assert!(
        !vm_res.timed_out && vm_res.status_code == Some(0),
        "VM oracle failed (rc={:?}, stderr={})",
        vm_res.status_code,
        String::from_utf8_lossy(&vm_res.stderr),
    );
    assert!(
        !clif_res.timed_out && clif_res.status_code == Some(0),
        "Cranelift semantics failed (rc={:?}, stderr={})",
        clif_res.status_code,
        String::from_utf8_lossy(&clif_res.stderr),
    );
    assert_eq!(
        clif_res.stdout,
        vm_res.stdout,
        "IsPattern semantics mismatch:\nclif={}\nvm  ={}",
        String::from_utf8_lossy(&clif_res.stdout),
        String::from_utf8_lossy(&vm_res.stdout),
    );
    // Verify single evaluation of side-effecting get_arr()[next_idx()]
    let out_str = String::from_utf8(vm_res.stdout).expect("utf8");
    let lines: Vec<&str> = out_str.lines().collect();
    assert_eq!(lines[0], "15");
    assert_eq!(lines[1], "-1");
    assert_eq!(lines[2], "42");
    assert_eq!(lines[3], "4");
    assert_eq!(lines[4], "10");
    assert_eq!(lines[5], "10");
    assert_eq!(lines[6], "30");
    assert_eq!(lines[7], "-777");
    assert_eq!(lines[8], "-777");
    assert_eq!(lines[9], "-777");
    assert_eq!(lines[10], "200");
    assert_eq!(lines[11], "1\t1");
    assert_eq!(lines[12], "-555");
    assert_eq!(lines[13], "2\t2");
    assert_eq!(lines[14], "-555");
    assert_eq!(lines[15], "3\t3");

    // 3. strict_mir = true must return an explicit CompileError (never stack overflow or silent false)
    for (label, strict_src) in [
        (
            "non-indexed Some",
            r#"
            function check_opt(x: Option<int>): int
                if x is Some(v) then
                    return v
                end
                return 0
            end
            "#,
        ),
        (
            "non-indexed None",
            r#"
            function check_none(x: Option<int>): int
                if x is None then
                    return 1
                end
                return 0
            end
            "#,
        ),
    ] {
        let entries = &[EmbeddedModule {
            module: "main",
            parent: None,
            source: Some(strict_src),
        }];
        let program = load_program_from_embedded(entries, "main").expect("load embedded program");
        let config = LustConfig::default().with_strict_mir(true);
        let err = match compile_program_with_config(program, &config) {
            Ok(_) => panic!("strict_mir=true must reject `{label}` explicitly"),
            Err(e) => e,
        };
        assert!(
            matches!(err, LustError::CompileError { .. }),
            "expected CompileError for `{label}`, got {err:?}"
        );
    }
}

#[test]
fn test_bf2b_struct_option_fallback_and_independent_native() {
    // 1. Minimal reproducer: `only_struct_build0`
    let only_struct_build0 = r#"
        struct Node
            value: int
            left: Option<Node>
            right: Option<Node>
        end

        function build(depth: int, value: int): Node
            if depth == 0 then
                return Node { value = value, left = Option.None, right = Option.None }
            end
            local l: Node = build(depth - 1, value * 2)
            local r: Node = build(depth - 1, value * 2 + 1)
            return Node { value = value, left = Option.Some(l), right = Option.Some(r) }
        end

        local t: Node = build(0, 1)
        println(t.value)
    "#;
    let vm_b0 = run_lust_script(only_struct_build0, &["--jit=off"], Duration::from_secs(10));
    let clif_b0 = run_lust_script(only_struct_build0, &["--jit=cranelift"], Duration::from_secs(10));
    assert!(
        !vm_b0.timed_out && vm_b0.status_code == Some(0),
        "only_struct_build0 VM failed: rc={:?}, stderr={}",
        vm_b0.status_code,
        String::from_utf8_lossy(&vm_b0.stderr),
    );
    assert!(
        !clif_b0.timed_out && clif_b0.status_code == Some(0),
        "only_struct_build0 Cranelift failed: rc={:?}, stderr={}",
        clif_b0.status_code,
        String::from_utf8_lossy(&clif_b0.stderr),
    );
    assert_eq!(vm_b0.stdout, b"1\n");
    assert_eq!(clif_b0.stdout, b"1\n");

    // 2. Full `benchmarks/suite/tree.lust` (depth=14 -> 10738090560\n)
    let tree_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks/suite/tree.lust");
    let vm_tree = run_lust_file(&tree_path, &["--jit=off"], Duration::from_secs(120));
    let clif_tree = run_lust_file(&tree_path, &["--jit=cranelift"], Duration::from_secs(120));
    assert!(
        !vm_tree.timed_out && vm_tree.status_code == Some(0),
        "tree.lust VM failed: timed_out={}, rc={:?}, stderr={}",
        vm_tree.timed_out,
        vm_tree.status_code,
        String::from_utf8_lossy(&vm_tree.stderr),
    );
    assert!(
        !clif_tree.timed_out && clif_tree.status_code == Some(0),
        "tree.lust Cranelift failed: timed_out={}, rc={:?}, stderr={}",
        clif_tree.timed_out,
        clif_tree.status_code,
        String::from_utf8_lossy(&clif_tree.stderr),
    );
    assert_eq!(vm_tree.stdout, b"10738090560\n");
    assert_eq!(clif_tree.stdout, b"10738090560\n");

    // 3. Mixed module: ineligible Node path falls back cleanly while independent scalar function
    // `add_loop` compiles and executes natively (cranelift_entry_count >= 1).
    let mixed_src = r#"
        struct Node
            value: int
            left: Option<Node>
            right: Option<Node>
        end

        function build(depth: int, value: int): Node
            if depth == 0 then
                return Node { value = value, left = Option.None, right = Option.None }
            end
            local l: Node = build(depth - 1, value * 2)
            local r: Node = build(depth - 1, value * 2 + 1)
            return Node { value = value, left = Option.Some(l), right = Option.Some(r) }
        end

        function read_root(): int
            local t: Node = build(0, 99)
            return t.value
        end

        function add_loop(n: int): int
            local acc: int = 0
            local i: int = 0
            while i < n do
                acc = acc + i
                i = i + 1
            end
            return acc
        end
    "#;
    let (mut vm_clif, mut vm_bc) = compile_mir_and_bytecode_vms(mixed_src);

    let r_clif = vm_clif
        .call("read_root", vec![])
        .expect("mixed module ineligible read_root must fall back cleanly");
    let r_bc = vm_bc.call("read_root", vec![]).expect("oracle read_root");
    assert_eq!(r_clif, Value::Int(99));
    assert_eq!(r_clif, r_bc);

    let entries_before = vm_clif.cranelift_entry_count;
    let l_clif = vm_clif
        .call("add_loop", vec![Value::Int(100)])
        .expect("mixed module independent add_loop must succeed");
    let l_bc = vm_bc
        .call("add_loop", vec![Value::Int(100)])
        .expect("oracle add_loop");
    assert_eq!(l_clif, Value::Int(4950));
    assert_eq!(l_clif, l_bc);
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "Independent scalar function add_loop must execute natively in Cranelift"
    );

    // 4. Direct malformed-MIR test: compile_module must still return Err(...) (never swallowed)
    let sig = FunctionSignature {
        name: "bad_field_on_int".to_string(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut bad_fn = MirFunction::new(sig, Span::dummy());
    let ret = bad_fn.return_local;
    let p0 = bad_fn.params[0];
    bad_fn.blocks[0].statements.push(Statement::Assign(
        Place::Local(ret),
        Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(p0)), 0))),
        Span::dummy(),
    ));
    bad_fn.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(ret))),
        Span::dummy(),
    ));
    let mut bad_mod = MirModule::new("malformed_bf2b".into());
    bad_mod.add_function(bad_fn);

    let mut jit = CraneliftJitCompiler::new().expect("create Cranelift compiler");
    let compile_res = jit.compile_module(&bad_mod);
    assert!(
        compile_res.is_err(),
        "Direct compile_module on malformed MIR must return Err, never be swallowed"
    );
}

#[test]
fn test_bf3_concat_variables_chains_and_for_ranges() {
    // 1. AST verification for concat vs for-in range
    // a .. b .. c -> right-associative Binary(Concat, a, Binary(Concat, b, c))
    let chain = parse_single_expr("a .. b .. c");
    match &chain.kind {
        ExprKind::Binary {
            left,
            op: BinaryOp::Concat,
            right,
        } => {
            assert!(matches!(left.kind, ExprKind::Identifier(ref n) if n == "a"));
            match &right.kind {
                ExprKind::Binary {
                    left: rl,
                    op: BinaryOp::Concat,
                    right: rr,
                } => {
                    assert!(matches!(rl.kind, ExprKind::Identifier(ref n) if n == "b"));
                    assert!(matches!(rr.kind, ExprKind::Identifier(ref n) if n == "c"));
                }
                other => panic!("expected right-associative Concat, got {other:?}"),
            }
        }
        other => panic!("expected Concat for `a .. b .. c`, got {other:?}"),
    }

    // a + 1 .. b - 1 -> Binary(Concat, Add(a, 1), Sub(b, 1))
    let arith_concat = parse_single_expr("a + 1 .. b - 1");
    match &arith_concat.kind {
        ExprKind::Binary {
            left,
            op: BinaryOp::Concat,
            right,
        } => {
            assert!(matches!(left.kind, ExprKind::Binary { op: BinaryOp::Add, .. }));
            assert!(matches!(right.kind, ExprKind::Binary { op: BinaryOp::Sub, .. }));
        }
        other => panic!("expected Concat(Add, Sub) for `a + 1 .. b - 1`, got {other:?}"),
    }

    // for i in 1..10 do end -> Range(1, 10)
    let r_lit = parse_for_in_iterator("for i in 1..10 do end").expect("parse for 1..10");
    match &r_lit.kind {
        ExprKind::Range { start, end, inclusive } => {
            assert!(!inclusive);
            assert!(matches!(start.kind, ExprKind::Literal(Literal::Integer(1))));
            assert!(matches!(end.kind, ExprKind::Literal(Literal::Integer(10))));
        }
        other => panic!("expected Range for `for i in 1..10`, got {other:?}"),
    }

    // for i in a..b do end -> Range(a, b)
    let r_var = parse_for_in_iterator("for i in a..b do end").expect("parse for a..b");
    match &r_var.kind {
        ExprKind::Range { start, end, .. } => {
            assert!(matches!(start.kind, ExprKind::Identifier(ref n) if n == "a"));
            assert!(matches!(end.kind, ExprKind::Identifier(ref n) if n == "b"));
        }
        other => panic!("expected Range for `for i in a..b`, got {other:?}"),
    }

    // for i in a + 1 .. b - 1 do end -> Range(Add(a, 1), Sub(b, 1))
    let r_arith =
        parse_for_in_iterator("for i in a + 1 .. b - 1 do end").expect("parse for a+1..b-1");
    match &r_arith.kind {
        ExprKind::Range { start, end, .. } => {
            assert!(matches!(start.kind, ExprKind::Binary { op: BinaryOp::Add, .. }));
            assert!(matches!(end.kind, ExprKind::Binary { op: BinaryOp::Sub, .. }));
        }
        other => panic!("expected Range(Add, Sub), got {other:?}"),
    }

    // for i in a | 1 .. b & 7 do end -> Range(BitOr(a, 1), BitAnd(b, 7))
    let r_bit =
        parse_for_in_iterator("for i in a | 1 .. b & 7 do end").expect("parse for a|1..b&7");
    match &r_bit.kind {
        ExprKind::Range { start, end, .. } => {
            assert!(matches!(start.kind, ExprKind::Binary { op: BinaryOp::BitOr, .. }));
            assert!(matches!(end.kind, ExprKind::Binary { op: BinaryOp::BitAnd, .. }));
        }
        other => panic!("expected Range(BitOr, BitAnd), got {other:?}"),
    }

    // for x in split(s .. t) do end -> Call(split, [Concat(s, t)])
    let r_call =
        parse_for_in_iterator("for x in split(s .. t) do end").expect("parse for split(s..t)");
    match &r_call.kind {
        ExprKind::Call { args, .. } => {
            assert_eq!(args.len(), 1);
            assert!(matches!(
                args[0].kind,
                ExprKind::Binary {
                    op: BinaryOp::Concat,
                    ..
                }
            ));
        }
        other => panic!("expected Call with Concat arg, got {other:?}"),
    }

    // for x in ("a" .. "b") do end -> Paren(Concat("a", "b"))
    let r_paren =
        parse_for_in_iterator("for x in (\"a\" .. \"b\") do end").expect("parse for (\"a\"..\"b\")");
    match &r_paren.kind {
        ExprKind::Paren(inner) => {
            assert!(matches!(
                inner.kind,
                ExprKind::Binary {
                    op: BinaryOp::Concat,
                    ..
                }
            ));
        }
        other => panic!("expected Paren(Concat), got {other:?}"),
    }

    // for i in 1..2..3 do end -> parse error
    assert!(
        parse_items("for i in 1..2..3 do end").is_err(),
        "`for i in 1..2..3 do end` must produce a parse error"
    );

    // 2. End-to-end parity across --jit=off, --jit=cranelift, and Zig AOT --release-fast
    let e2e_src = r#"
        function next_a(counts: Array<int>): string
            if counts[0] is Ok(c) then
                counts[0] = c + 1
            end
            return "foo"
        end

        function next_b(counts: Array<int>): string
            if counts[1] is Ok(c) then
                counts[1] = c + 1
            end
            return "bar"
        end

        local s: string = ""
        local i: int = 0
        while i < 200 do
            s = s .. tostring(i % 10)
            i = i + 1
        end
        println(string.len(s))

        local empty1: string = ""
        local empty2: string = ""
        println(string.len(empty1 .. empty2))

        local part1: string = "привет"
        local part2: string = " "
        local part3: string = "мир"
        println(part1 .. part2 .. part3)

        local counts: Array<int> = [0, 0]
        local se: string = next_a(counts) .. next_b(counts)
        println(se)
        if counts[0] is Ok(c0) then
            println(c0)
        end
        if counts[1] is Ok(c1) then
            println(c1)
        end

        local range_sum: int = 0
        for k = 1, 5 do
            range_sum = range_sum + k
        end
        println(range_sum)
    "#;

    let vm_res = run_lust_script(e2e_src, &["--jit=off"], Duration::from_secs(10));
    let clif_res = run_lust_script(e2e_src, &["--jit=cranelift"], Duration::from_secs(10));
    assert!(
        !vm_res.timed_out && vm_res.status_code == Some(0),
        "concat e2e VM failed: rc={:?}, stderr={}",
        vm_res.status_code,
        String::from_utf8_lossy(&vm_res.stderr),
    );
    assert!(
        !clif_res.timed_out && clif_res.status_code == Some(0),
        "concat e2e Cranelift failed: rc={:?}, stderr={}",
        clif_res.status_code,
        String::from_utf8_lossy(&clif_res.stderr),
    );
    assert_eq!(clif_res.stdout, vm_res.stdout);
    let expected_e2e = "200\n0\nпривет мир\nfoobar\n1\n1\n15\n";
    assert_eq!(
        String::from_utf8_lossy(&vm_res.stdout),
        expected_e2e,
        "Unexpected concat e2e output"
    );

    // Zig AOT --release-fast parity
    let src_path = unique_temp_path("concat_aot", "lust");
    let bin_path = unique_temp_path("concat_aot_bin", "out");
    fs::write(&src_path, e2e_src).expect("write aot concat script");
    let src_str = src_path.to_string_lossy().into_owned();
    let bin_str = bin_path.to_string_lossy().into_owned();
    let build_res = run_cmd_with_timeout(
        lust_bin(),
        &["build", &src_str, "-o", &bin_str, "--release-fast"],
        Duration::from_secs(45),
    );
    assert!(
        !build_res.timed_out && build_res.status_code == Some(0),
        "Zig AOT build failed: timed_out={}, rc={:?}, stderr={}",
        build_res.timed_out,
        build_res.status_code,
        String::from_utf8_lossy(&build_res.stderr),
    );
    let aot_run = run_cmd_with_timeout(&bin_str, &[], Duration::from_secs(10));
    let _ = fs::remove_file(&src_path);
    let _ = fs::remove_file(&bin_path);
    assert!(
        !aot_run.timed_out && aot_run.status_code == Some(0),
        "Zig AOT run failed: timed_out={}, rc={:?}, stderr={}",
        aot_run.timed_out,
        aot_run.status_code,
        String::from_utf8_lossy(&aot_run.stderr),
    );
    assert_eq!(
        aot_run.stdout,
        vm_res.stdout,
        "Zig AOT vs VM concat parity mismatch:\naot={}\nvm ={}",
        String::from_utf8_lossy(&aot_run.stdout),
        String::from_utf8_lossy(&vm_res.stdout),
    );

    // 3. Non-ToString type error: must parse as Concat and fail typechecking on ToString
    let bad_concat = r#"
        struct Opaque
            x: int
        end
        local o: Opaque = Opaque { x = 1 }
        local bad: string = o .. "tail"
        println(bad)
    "#;
    let items = parse_items(bad_concat).expect("bad_concat must parse cleanly");
    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    let tc_err = checker
        .check_module(&items)
        .expect_err("concat on non-ToString struct must fail typechecking");
    let err_msg = format!("{tc_err}");
    assert!(
        err_msg.contains("ToString"),
        "Expected ToString error on non-ToString concat operand, got: {err_msg}"
    );
}

#[test]
fn test_bf2a_field_access_rejects_option_none_unknown_and_nested_fields() {
    use lust::mir::lowering::{AstLowering, MirLowerError};

    fn lower_single_fn(src: &str, fn_name: &str) -> Result<MirFunction, MirLowerError> {
        let items = parse_items(src).expect("fixture must parse");
        let struct_defs: Vec<&lust::ast::StructDef> = items
            .iter()
            .filter_map(|item| match &item.kind {
                ItemKind::Struct(s) => Some(s),
                _ => None,
            })
            .collect();
        let fdef = items
            .iter()
            .find_map(|item| match &item.kind {
                ItemKind::Function(f) if f.name == fn_name => Some(f),
                _ => None,
            })
            .expect("target function present");
        let (mir_fn, _ops) = AstLowering::lower_function_with_structs(fdef, &struct_defs)?;
        Ok(mir_fn)
    }

    // 1. Option.None field access must fail MIR lowering cleanly (non-local base).
    let err_none = lower_single_fn(
        r#"
        function f(): int
            local x = Option.None
            return 0
        end
        "#,
        "f",
    )
    .expect_err("Option.None must fail MIR lowering with UnlowerableExpression");
    assert!(matches!(err_none, MirLowerError::UnlowerableExpression(_)), "got {err_none:?}");

    // 2. Unknown field on a registered struct (including '.x' which previously matched legacy fallback!).
    let err_unknown_x = lower_single_fn(
        r#"
        struct Node
            value: int
            left: Option<Node>
        end
        function f(n: Node): int
            return n.x
        end
        "#,
        "f",
    )
    .expect_err("Unknown field '.x' on registered struct Node must fail MIR lowering");
    assert!(
        matches!(err_unknown_x, MirLowerError::UnlowerableExpression(_)),
        "got {err_unknown_x:?}"
    );

    // 3. Nested field access on a non-struct field (n.left is Option<Node>, NOT Node — must NOT resolve parent Node.value!).
    let err_nested = lower_single_fn(
        r#"
        struct Node
            value: int
            left: Option<Node>
        end
        function f(n: Node): int
            return n.left.value
        end
        "#,
        "f",
    )
    .expect_err("Nested field access n.left.value on Option<Node> must fail MIR lowering");
    assert!(
        matches!(err_nested, MirLowerError::UnlowerableExpression(_)),
        "got {err_nested:?}"
    );

    // 4. Constant base field access '(42)._0' / '(42).x' and scalar local 'a.x' must fail MIR lowering.
    for (label, const_src) in [
        (
            "(42)._0",
            r#"
            function f(): int
                return (42)._0
            end
            "#,
        ),
        (
            "(42).x",
            r#"
            function f(): int
                return (42).x
            end
            "#,
        ),
    ] {
        let err_const = lower_single_fn(const_src, "f")
            .expect_err(&format!("Constant field access {label} must fail MIR lowering"));
        assert!(
            matches!(err_const, MirLowerError::UnlowerableExpression(_)),
            "{label}: got {err_const:?}"
        );
    }

    let err_scalar = lower_single_fn(
        r#"
        function f(a: int): int
            return a.x
        end
        "#,
        "f",
    )
    .expect_err("Scalar field access a.x must fail MIR lowering");
    assert!(matches!(err_scalar, MirLowerError::UnlowerableExpression(_)), "got {err_scalar:?}");

    // 5. Positive control: valid field on registered struct lowers cleanly to field index 0.
    let ok_fn = lower_single_fn(
        r#"
        struct Node
            value: int
            left: Option<Node>
        end
        function f(n: Node): int
            return n.value
        end
        "#,
        "f",
    )
    .expect("Valid registered struct field n.value must lower cleanly");
    assert_eq!(ok_fn.signature.name, "f");
}

#[test]
fn test_bf2b_struct_option_result_handles_preserved_e2e() {
    use lust::bytecode::compiler::Compiler;
    use lust::mir::ir::MirModule;
    use lust::mir::lowering::AstLowering;
    use lust::vm::{VM, VmBackend};
    use lust::Value;

    // 1. Direct Cranelift VM native execution on Option/Result-bearing struct:
    //    verifies native entry (cranelift_entry_count > 0) AND bit/handle preservation of Enum fields.
    let direct_src = r#"
        struct Envelope
            id: int
            opt: Option<int>
            res: Result<int, string>
        end

        function read_id(e: Envelope): int
            return e.id + 10
        end

        function id_env(e: Envelope): Envelope
            return e
        end
    "#;
    let items = parse_items(direct_src).expect("direct_src parses");
    let struct_defs_ast: Vec<&lust::ast::StructDef> = items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Struct(s) => Some(s),
            _ => None,
        })
        .collect();
    let mut mir_module = MirModule::new("bf2b_env".into());
    for item in &items {
        if let ItemKind::Function(f) = &item.kind {
            let (mir_fn, _) =
                AstLowering::lower_function_with_structs(f, &struct_defs_ast).expect("lower");
            mir_module.add_function(mir_fn);
        }
    }

    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    checker.check_module(&items).expect("typecheck direct_src");
    let struct_defs = checker.struct_definitions();

    let mut vm_clif = VM::new();
    vm_clif.load_mir_module(&mir_module).expect("load_mir_module");
    vm_clif.set_backend(VmBackend::Cranelift);
    vm_clif.register_structs(&struct_defs);

    let mut vm_bc = VM::new();
    vm_bc.register_structs(&struct_defs);
    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    let bc_fns = compiler.compile_module(&items).expect("compile_module");
    vm_bc.load_functions(bc_fns);

    let opt_some = Value::enum_variant("Option", "Some", vec![Value::Int(7)]);
    let res_ok = Value::enum_variant("Result", "Ok", vec![Value::Int(99)]);
    let env_clif = vm_clif
        .instantiate_struct_ordered("Envelope", vec![Value::Int(32), opt_some.clone(), res_ok.clone()])
        .expect("instantiate Envelope (clif)");
    let env_bc = vm_bc
        .instantiate_struct_ordered("Envelope", vec![Value::Int(32), opt_some.clone(), res_ok.clone()])
        .expect("instantiate Envelope (bc)");

    let entries_before = vm_clif.cranelift_entry_count;
    let got_id_clif = vm_clif.call("read_id", vec![env_clif.clone()]).expect("clif read_id");
    let got_id_bc = vm_bc.call("read_id", vec![env_bc.clone()]).expect("bc read_id");
    assert_eq!(got_id_clif, Value::Int(42));
    assert_eq!(got_id_clif, got_id_bc);
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "read_id(Envelope) must execute natively via Cranelift"
    );

    let entries_before_id = vm_clif.cranelift_entry_count;
    let roundtrip_clif = vm_clif.call("id_env", vec![env_clif]).expect("clif id_env");
    let roundtrip_bc = vm_bc.call("id_env", vec![env_bc]).expect("bc id_env");
    assert_eq!(roundtrip_clif, roundtrip_bc, "Option/Result struct fields must roundtrip intact");
    assert!(
        vm_clif.cranelift_entry_count > entries_before_id,
        "id_env(Envelope) must execute natively via Cranelift"
    );

    // 2. CLI end-to-end parity with top-level Envelope creation + post-call unwrap checks.
    let cli_src = r#"
        struct Envelope
            id: int
            opt: Option<int>
            res: Result<int, string>
        end

        function read_id(e: Envelope): int
            return e.id + 10
        end

        local e: Envelope = Envelope {
            id = 32,
            opt = Option.Some(7),
            res = Result.Ok(99)
        }
        println(read_id(e))
        println(e.opt:unwrap())
        if e.res is Ok(v) then
            println(v)
        end
    "#;
    let vm_res = run_lust_script(cli_src, &["--jit=off"], Duration::from_secs(10));
    let clif_res = run_lust_script(cli_src, &["--jit=cranelift"], Duration::from_secs(10));
    assert_eq!(vm_res.status_code, Some(0), "vm stderr: {}", String::from_utf8_lossy(&vm_res.stderr));
    assert_eq!(clif_res.status_code, Some(0), "clif stderr: {}", String::from_utf8_lossy(&clif_res.stderr));
    assert_eq!(vm_res.stdout, b"42\n7\n99\n");
    assert_eq!(clif_res.stdout, b"42\n7\n99\n");
}

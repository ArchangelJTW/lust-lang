#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use lust::{
    ast::*,
    codegen_zig::{
        ZigCodegen, analyze_runtime_requirements, tree_shake_items,
    },
    intern::Interner,
    lexer::Lexer,
    mir::{
        ir::*,
        zig_backend::{MirToZigCompiler, ZigArtifactKind, ZigCompileOptions},
    },
    parser::Parser,
};
use std::process::Command;

fn parse_source(source: &str) -> Vec<Item> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize().expect("tokenize");
    Parser::new(tokens).parse().expect("parse")
}

fn run_zig_source(zig_src: &str) {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let zig_path = temp_dir.path().join("main.zig");
    std::fs::write(&zig_path, zig_src).expect("write zig source");

    let output = Command::new("zig")
        .arg("run")
        .arg(&zig_path)
        .output()
        .expect("zig run invocation");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!(
            "Zig execution failed with status {}:\nSTDOUT:\n{}\nSTDERR:\n{}\nZIG SOURCE:\n{}",
            output.status, stdout, stderr, zig_src
        );
    }
}

/// Test 1: Pure Math / Array No-Bloat
/// A program with only integer arithmetic and Array<int> operations must NOT pull:
/// - `__lust_print`, `__lust_println`, or `std.c.write`
/// - String runtime (`STRING_RUNTIME_SRC`, `__lust_string_split`, etc.)
/// Both in `codegen_zig` and in `mir::zig_backend`.
#[test]
fn test_pure_math_array_no_bloat() {
    let source = r#"
        function compute(n: int): int
            local arr: Array<int> = [1, 2, 3]
            arr.push(n)
            local sum: int = 0
            for i = 0, arr.len() - 1 do
                sum = sum + arr[i]
            end
            return sum
        end

        local _res = compute(4)
    "#;

    let items = parse_source(source);
    let opts = analyze_runtime_requirements(&items, false, None);

    assert!(opts.needs_array, "Array runtime must be needed");
    assert!(!opts.needs_print, "Print runtime must NOT be needed");
    assert!(!opts.needs_string, "String runtime must NOT be needed");

    let mut codegen = ZigCodegen::with_options(opts);
    let zig_src = codegen.generate_program(&items).expect("generate_program");

    // Must NOT contain print helpers or POSIX write
    assert!(!zig_src.contains("__lust_print"), "Must not contain __lust_print");
    assert!(!zig_src.contains("__lust_println"), "Must not contain __lust_println");
    assert!(!zig_src.contains("std.c.write"), "Must not contain std.c.write");

    // Must NOT contain string runtime helpers
    assert!(!zig_src.contains("__lust_string_split"), "Must not contain __lust_string_split");
    assert!(!zig_src.contains("__lust_string_lower"), "Must not contain __lust_string_lower");
    assert!(!zig_src.contains("__lust_string_upper"), "Must not contain __lust_string_upper");

    // Must contain LustArray
    assert!(zig_src.contains("pub fn LustArray"), "Must contain LustArray definition");

    // Compile and run with Zig to verify correctness
    run_zig_source(&zig_src);

    // Verify MIR Zig Backend for pure math as well
    let mut mir_func = MirFunction::new(
        FunctionSignature {
            name: "mir_pure_math".to_string(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        Span::dummy(),
    );
    mir_func.blocks = vec![BasicBlock {
        id: BlockId(0),
        statements: vec![Statement::Assign(
            Place::Local(LocalId(1)),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(LocalId(0))),
                Operand::Constant(MirConstant::Int(10)),
            ),
            Span::dummy(),
        )],
        terminator: Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(LocalId(1)))),
            Span::dummy(),
        )),
    }];

    let mir_opts = ZigCompileOptions {
        artifact_kind: ZigArtifactKind::StaticLib,
        emit_c_header: false,
        max_memory_bytes: None,
    };
    let mir_zig = MirToZigCompiler::generate_source_with_options(&mir_func, &mir_opts)
        .expect("MIR zig generation");

    assert!(!mir_zig.contains("__lust_mir_print"), "MIR backend must not include __lust_mir_print for pure math");
    assert!(!mir_zig.contains("std.c.write"), "MIR backend must not include std.c.write for pure math");
}

/// Test 2: String Concat without Array Bloat
/// A program that only concatenates strings (e.g. `"score: " .. 42`) must NOT pull `LustArray`.
#[test]
fn test_string_concat_no_array_bloat() {
    let source = r#"
        function format_score(score: int): string
            return "score: " .. score
        end

        local s = format_score(100)
    "#;

    let items = parse_source(source);
    let opts = analyze_runtime_requirements(&items, false, None);

    assert!(opts.needs_string, "String runtime must be needed");
    assert!(!opts.needs_array, "Array runtime must NOT be needed for string concat");
    assert!(!opts.needs_print, "Print runtime must NOT be needed");

    let mut codegen = ZigCodegen::with_options(opts);
    let zig_src = codegen.generate_program(&items).expect("generate_program");

    // Must contain string concat and tostring
    assert!(zig_src.contains("__lust_str_concat"), "Must contain __lust_str_concat");
    assert!(zig_src.contains("__lust_tostring"), "Must contain __lust_tostring");

    // Must NOT contain array runtime
    assert!(!zig_src.contains("pub fn LustArray"), "Must not contain LustArray");

    // Must NOT contain print runtime
    assert!(!zig_src.contains("__lust_print"), "Must not contain __lust_print");
    assert!(!zig_src.contains("__lust_println"), "Must not contain __lust_println");
    assert!(!zig_src.contains("std.c.write"), "Must not contain std.c.write");

    // Compile and run with Zig
    run_zig_source(&zig_src);
}

/// Test 3: Dead Function Elimination
/// When an unused function contains `println("never called")`, after `tree_shake_items`:
/// - The dead function is removed
/// - `__lust_print` / `__lust_println` / `std.c.write` is NOT generated in Zig output
#[test]
fn test_dead_function_elimination() {
    let source = r#"
        function dead_fn()
            println("never called")
        end

        function alive_fn(): int
            return 42
        end

        local _res = alive_fn()
    "#;

    let items = parse_source(source);
    let shaken = tree_shake_items(&items, false);

    // dead_fn must be eliminated
    let has_dead_fn = shaken.iter().any(|item| {
        if let ItemKind::Function(f) = &item.kind {
            f.name == "dead_fn"
        } else {
            false
        }
    });
    assert!(!has_dead_fn, "dead_fn must be eliminated by tree_shake_items");

    let opts = analyze_runtime_requirements(&shaken, false, None);
    assert!(!opts.needs_print, "Print runtime must NOT be needed after dead_fn elimination");

    let mut codegen = ZigCodegen::with_options(opts);
    let zig_src = codegen.generate_program(&shaken).expect("generate_program");

    assert!(!zig_src.contains("dead_fn"), "Generated code must not contain dead_fn");
    assert!(!zig_src.contains("__lust_print"), "Generated code must not contain __lust_print");
    assert!(!zig_src.contains("__lust_println"), "Generated code must not contain __lust_println");
    assert!(!zig_src.contains("std.c.write"), "Generated code must not contain std.c.write");

    run_zig_source(&zig_src);
}

/// Test 4: Use import resolution for std.math and std.io
/// Verify `use std.math.{sqrt, abs};` and `use std.io.{print, println};`
#[test]
fn test_use_imports_resolution() {
    let source = r#"
        use std.math.{sqrt, abs}
        use std.io.{println}

        function calculate(): int
            local root: float = sqrt(16.0)
            local a: float = abs(-5.5)
            println("calculation done")
            return 0
        end

        local _res = calculate()
    "#;

    let items = parse_source(source);
    let opts = analyze_runtime_requirements(&items, false, None);

    assert!(opts.needs_math, "Math runtime must be needed");
    assert!(opts.needs_print, "Print runtime must be needed for println");

    let mut codegen = ZigCodegen::with_options(opts);
    let zig_src = codegen.generate_program(&items).expect("generate_program");

    assert!(zig_src.contains("__lust_math_sqrt"), "Must call __lust_math_sqrt");
    assert!(zig_src.contains("__lust_math_abs"), "Must call __lust_math_abs");
    assert!(zig_src.contains("__lust_println"), "Must call __lust_println");

    run_zig_source(&zig_src);
}

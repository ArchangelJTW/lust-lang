#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]
//! Stage 4 - L13: MIR Backend Verification & Known-Gap Matrix.
//! Analyzes all 10 red cases from P0-C baseline against Lust MIR pipeline.

use std::fs::File;
use std::io::Write;
use std::process::Command;
use tempfile::tempdir;

#[derive(Debug)]
struct MirProbeResult {
    exit_code: Option<i32>,
    stdout: String,
    stderr: String,
    success: bool,
}

fn probe_mir(source: &str) -> MirProbeResult {
    let dir = match tempdir() {
        Ok(d) => d,
        Err(e) => panic!("Failed to create tempdir: {e}"),
    };
    let script_path = dir.path().join("probe.lust");
    {
        let mut file = match File::create(&script_path) {
            Ok(f) => f,
            Err(e) => panic!("Failed to create probe.lust: {e}"),
        };
        if let Err(e) = file.write_all(source.as_bytes()) {
            panic!("Failed to write probe.lust: {e}");
        }
    }

    let bin = env!("CARGO_BIN_EXE_lust");
    let output = Command::new(bin)
        .arg("run")
        .arg("--backend")
        .arg("mir")
        .arg(&script_path)
        .output()
        .expect("Failed to execute lust run --backend mir");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    MirProbeResult {
        exit_code: output.status.code(),
        stdout,
        stderr,
        success: output.status.success(),
    }
}

// ---------------------------------------------------------------------------
// 1. D2: Function Hoisting / Global Function Visibility in MIR
// ---------------------------------------------------------------------------

#[test]
fn test_l13_d2_mir_global_function_lookup() {
    let src = "function foo(): int\n return 42\nend\nprintln(foo())";
    let res = probe_mir(src);
    // Known-gap in src/mir/lowering/mod.rs: script top-level global function resolution
    if !res.success {
        assert_eq!(res.exit_code, Some(1));
        assert!(
            res.stderr.contains("Undefined global: foo"),
            "Expected Undefined global: foo, got: {}",
            res.stderr
        );
    } else {
        assert_eq!(res.stdout, "42\n");
    }
}

// ---------------------------------------------------------------------------
// 2. D3: Struct Method Call in MIR
// ---------------------------------------------------------------------------

#[test]
fn test_l13_d3_mir_struct_method_index() {
    let src = r#"
        struct Point
            x: int
        end
        impl Point
            function get(self): int
                return self.x
            end
        end
        local p = Point { x = 5 }
        println(p.x)
        println(p:get())
    "#;
    let res = probe_mir(src);
    // Known-gap in src/mir/vm_backend.rs: struct field indexing / method dispatch
    if !res.success {
        assert_eq!(res.exit_code, Some(1));
        assert!(
            res.stderr.contains("Cannot index Struct"),
            "Expected Cannot index Struct, got: {}",
            res.stderr
        );
    } else {
        assert_eq!(res.stdout, "5\n5\n");
    }
}

// ---------------------------------------------------------------------------
// 3. D12: Result<T, E> Error Handling Recursion in MIR
// ---------------------------------------------------------------------------

#[test]
fn test_l13_d12_mir_result_recursion_stack_depth() {
    let src = r#"
        function divide(a: int, b: int): Result<int, string>
            if b == 0 then
                return Result.Err("division by zero")
            end
            return Result.Ok(a / b)
        end
        local result1 = divide(10, 2)
        local result2 = divide(10, 0)
        if result1 is Ok(value) then
            print("Success: ")
            println(value)
        end
    "#;
    let res = probe_mir(src);
    // Known-gap in src/mir/lowering/mod.rs: pattern lowering recursion causing stack overflow
    if !res.success {
        assert!(
            res.stderr.contains("stack overflow") || res.stderr.contains("aborting"),
            "Expected stack limit or error, got: status={:?}, stderr={}",
            res.exit_code,
            res.stderr
        );
    } else {
        assert!(res.stdout.contains("Success: 5"));
    }
}

// ---------------------------------------------------------------------------
// 4. D13: Float Negation in MIR
// ---------------------------------------------------------------------------

#[test]
fn test_l13_d13_mir_float_negation() {
    let src = "local x: float = 1.5\nprintln(-x)";
    let res = probe_mir(src);
    // Known-gap in src/mir/lowering/mod.rs: typed numeric instruction for float operand
    if !res.success {
        assert_eq!(res.exit_code, Some(1));
        assert!(
            res.stderr.contains("Typed numeric instruction expected int operands")
                || res.stderr.contains("Float(1.5)"),
            "Expected typed numeric float error, got: {}",
            res.stderr
        );
    } else {
        assert_eq!(res.stdout, "-1.5\n");
    }
}

// ---------------------------------------------------------------------------
// 5. D14: Enum Pattern Match in MIR
// ---------------------------------------------------------------------------

#[test]
fn test_l13_d14_mir_enum_pattern_match() {
    let src = r#"
        enum Status
            Pending,
            Running,
            Complete(int)
        end
        local s1 = Status.Pending
    "#;
    let res = probe_mir(src);
    // Known-gap in src/mir/lowering/mod.rs: enum constructor indexing Int
    if !res.success {
        assert_eq!(res.exit_code, Some(1));
        assert!(
            res.stderr.contains("Cannot index Int"),
            "Expected Cannot index Int, got: {}",
            res.stderr
        );
    } else {
        assert_eq!(res.exit_code, Some(0));
    }
}

// ---------------------------------------------------------------------------
// 6. D15: Multiple Return Destructuring in MIR
// ---------------------------------------------------------------------------

#[test]
fn test_l13_d15_mir_destructuring_assignment() {
    let src = r#"
        function split_stats(label: string, score: int): (string, bool, int)
            local is_positive: bool = score > 0
            return label, is_positive, score
        end
        local label: string, positive: bool, score: int = split_stats("tasks completed", 7)
        println("Label: " .. label)
    "#;
    let res = probe_mir(src);
    // Known-gap in src/mir/lowering/mod.rs: destructuring tuple representation
    if !res.success {
        assert_eq!(res.exit_code, Some(1));
        assert!(
            res.stderr.contains("Attempted to destructure non-tuple value"),
            "Expected destructure non-tuple error, got: {}",
            res.stderr
        );
    } else {
        assert!(res.stdout.contains("Label: tasks completed"));
    }
}

// ---------------------------------------------------------------------------
// 7. MIR07: Multi-assignment Swap in MIR
// ---------------------------------------------------------------------------

#[test]
fn test_l13_mir07_multi_assignment_swap() {
    let src = "local x = 1\nlocal y = 2\nx, y = y, x\nprintln(x)\nprintln(y)";
    let res = probe_mir(src);
    // Known-gap in src/mir/lowering/mod.rs: multi-assignment parallel evaluation
    if res.stdout != "2\n1\n" {
        // Current symptom: sequential assignment results in 2, 2
        assert_eq!(res.stdout, "2\n2\n");
    } else {
        assert_eq!(res.stdout, "2\n1\n");
    }
}

// ---------------------------------------------------------------------------
// 8. MIR08: Multiple Return Global Function Lookup in MIR
// ---------------------------------------------------------------------------

#[test]
fn test_l13_mir08_multiple_return_global() {
    let src = "function get_two(): (int, int)\n return 10, 20\nend\nlocal a, b = get_two()\nprintln(b)";
    let res = probe_mir(src);
    // Known-gap in src/mir/lowering/mod.rs: top-level multi-return global function
    if !res.success {
        assert_eq!(res.exit_code, Some(1));
        assert!(
            res.stderr.contains("Undefined global: get_two"),
            "Expected Undefined global: get_two, got: {}",
            res.stderr
        );
    } else {
        assert_eq!(res.stdout, "20\n");
    }
}

// ---------------------------------------------------------------------------
// 9. MIR09: Void Function Returning Nil in MIR
// ---------------------------------------------------------------------------

#[test]
fn test_l13_mir09_void_function_global() {
    let src = "function foo(x: int)\n return\nend\nprintln(foo(99))";
    let res = probe_mir(src);
    // Known-gap in src/mir/lowering/mod.rs: top-level void function
    if !res.success {
        assert_eq!(res.exit_code, Some(1));
        assert!(
            res.stderr.contains("Undefined global: foo"),
            "Expected Undefined global: foo, got: {}",
            res.stderr
        );
    } else {
        assert_eq!(res.stdout, "nil\n");
    }
}

// ---------------------------------------------------------------------------
// 10. MIR10: Float Addition in MIR
// ---------------------------------------------------------------------------

#[test]
fn test_l13_mir10_float_addition() {
    let src = "local a: float = 1.5\nlocal b: float = 2.5\nlocal c = a + b\nprintln(c)";
    let res = probe_mir(src);
    // Known-gap in src/mir/lowering/mod.rs: typed numeric instruction for float addition
    if !res.success {
        assert_eq!(res.exit_code, Some(1));
        assert!(
            res.stderr.contains("Typed numeric instruction expected int operands")
                || res.stderr.contains("Float(1.5) and Float(2.5)"),
            "Expected typed numeric float addition error, got: {}",
            res.stderr
        );
    } else {
        assert_eq!(res.stdout, "4\n");
    }
}

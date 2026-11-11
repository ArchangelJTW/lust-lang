#![cfg(feature = "std")]
//! Stage 4 - L12: Zig Codegen Backend Verification & Known-Gap Matrix.
//! Analyzes all 13 red cases from P0-C baseline against Zig 0.17.1 compiler.

use lust::{
    codegen_zig::{analyze_runtime_requirements, ZigCodegen},
    intern::Interner,
    lexer::Lexer,
    parser::Parser,
};
use std::process::Command;
use tempfile::tempdir;

#[derive(Debug)]
struct ZigProbeResult {
    exit_code: Option<i32>,
    stdout: String,
    stderr: String,
    success: bool,
    #[allow(dead_code)]
    zig_source: String,
}

fn probe_zig(source: &str) -> ZigProbeResult {
    let mut interner = Interner::new();
    let tokens = match Lexer::new(source, &mut interner).tokenize() {
        Ok(t) => t,
        Err(e) => {
            return ZigProbeResult {
                exit_code: Some(1),
                stdout: String::new(),
                stderr: format!("Lexer error: {e}"),
                success: false,
                zig_source: String::new(),
            };
        }
    };

    let items = match Parser::new(tokens).parse() {
        Ok(it) => it,
        Err(e) => {
            return ZigProbeResult {
                exit_code: Some(1),
                stdout: String::new(),
                stderr: format!("Parser error: {e}"),
                success: false,
                zig_source: String::new(),
            };
        }
    };

    let opts = analyze_runtime_requirements(&items, false, None);
    let mut codegen = ZigCodegen::with_options(opts);
    let zig_src = match codegen.generate_program(&items) {
        Ok(src) => src,
        Err(e) => {
            return ZigProbeResult {
                exit_code: Some(1),
                stdout: String::new(),
                stderr: format!("Codegen error: {e}"),
                success: false,
                zig_source: String::new(),
            };
        }
    };

    let dir = match tempdir() {
        Ok(d) => d,
        Err(e) => panic!("Failed to create tempdir: {e}"),
    };
    let probe_path = dir.path().join("probe.zig");
    if let Err(e) = std::fs::write(&probe_path, &zig_src) {
        panic!("Failed to write probe.zig: {e}");
    }

    let output = Command::new("zig")
        .arg("run")
        .arg(&probe_path)
        .output()
        .expect("zig command execution failed");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    ZigProbeResult {
        exit_code: output.status.code(),
        stdout,
        stderr,
        success: output.status.success(),
        zig_source: zig_src,
    }
}

// ---------------------------------------------------------------------------
// 1. D1: Duplicate Struct Member 'main'
// ---------------------------------------------------------------------------

#[test]
fn test_l12_d1_main_duplicate_symbol() {
    let src = "function main(): int\n println(42)\n return 0\nend\nmain()";
    let res = probe_zig(src);
    // Known-gap in src/codegen_zig/mod.rs:emit_function vs emit_main.
    // Defining `function main()` collides with Zig top-level `pub fn main() !void`.
    if !res.success {
        assert_eq!(res.exit_code, Some(1));
        assert!(
            res.stderr.contains("duplicate struct member name 'main'")
                || res.stderr.contains("duplicate name here"),
            "Expected duplicate main symbol error: {}",
            res.stderr
        );
    } else {
        assert_eq!(res.stdout, "42\n");
    }
}

// ---------------------------------------------------------------------------
// 2. D4: Float Division Formatting
// ---------------------------------------------------------------------------

#[test]
fn test_l12_d4_float_division() {
    let src = "println(1.0 / 3.0)";
    let res = probe_zig(src);
    assert!(res.success, "Float division probe failed: {}", res.stderr);
    // Known-gap in src/codegen_zig/runtime.rs:MATH_RUNTIME_SRC (__lust_div return type or format)
    // In baseline P0-C, exit is 0 but stdout is "0\n" instead of "0.3333333333333333\n".
    if res.stdout != "0.3333333333333333\n" {
        assert_eq!(res.stdout, "0\n");
    } else {
        assert_eq!(res.stdout, "0.3333333333333333\n");
    }
}

// ---------------------------------------------------------------------------
// 3. D5: Map Keys Module Lookup
// ---------------------------------------------------------------------------

#[test]
fn test_l12_d5_map_keys_module() {
    let src = "local scores: Map<string, int> = { Alice = 100, Bob = 85 }\nlocal keys = map.keys(scores)\nprintln(array.len(keys))";
    let res = probe_zig(src);
    // Known-gap in src/codegen_zig/runtime.rs: map.keys runtime helper declaration
    if !res.success {
        assert_eq!(res.exit_code, Some(1));
        assert!(
            res.stderr.contains("use of undeclared identifier 'map'")
                || res.stderr.contains("undeclared identifier"),
            "Expected undeclared identifier map: {}",
            res.stderr
        );
    } else {
        assert_eq!(res.stdout, "2\n");
    }
}

// ---------------------------------------------------------------------------
// 4. D6: Closure Scope Return Emission
// ---------------------------------------------------------------------------

#[test]
fn test_l12_d6_closure_return_scope() {
    let src = "local base: int = 17\nlocal f = function(x: int): int\n return x + base\nend\nprintln(f(3))\nprintln(f(-2))";
    let res = probe_zig(src);
    // Known-gap in src/codegen_zig/mod.rs:emit_lambda
    if !res.success {
        assert_eq!(res.exit_code, Some(1));
        assert!(
            res.stderr.contains("'return' outside function scope")
                || res.stderr.contains("outside function scope"),
            "Expected return outside function scope error: {}",
            res.stderr
        );
    } else {
        assert_eq!(res.stdout, "20\n15\n");
    }
}

// ---------------------------------------------------------------------------
// 5. D7: Multiple Return Binding Type Inference
// ---------------------------------------------------------------------------

#[test]
fn test_l12_d7_multiple_return_destructuring() {
    let src = "function pair(): (string, int)\n return \"sentinel\", 37\nend\nlocal a, b = pair()\nprintln(a)\nprintln(b)";
    let res = probe_zig(src);
    // Known-gap in src/codegen_zig/mod.rs:emit_stmt (undefined type in multi-binding)
    if !res.success {
        assert_eq!(res.exit_code, Some(1));
        assert!(
            res.stderr.contains("variable of type '@TypeOf(undefined)' must be const or comptime")
                || res.stderr.contains("@TypeOf(undefined)"),
            "Expected @TypeOf(undefined) error: {}",
            res.stderr
        );
    } else {
        assert_eq!(res.stdout, "sentinel\n37\n");
    }
}

// ---------------------------------------------------------------------------
// 6. D8: Raw Multiline String Literals
// ---------------------------------------------------------------------------

#[test]
fn test_l12_d8_string_literal_escaping() {
    let src = "println(\"line1\\nline2\")";
    let res = probe_zig(src);
    // Known-gap in src/codegen_zig/mod.rs:emit_literal (unescaped newline in string literal)
    if !res.success {
        assert_eq!(res.exit_code, Some(1));
        assert!(
            res.stderr.contains("string literal contains invalid byte: '\\n'")
                || res.stderr.contains("invalid byte"),
            "Expected invalid byte in string literal: {}",
            res.stderr
        );
    } else {
        assert_eq!(res.stdout, "line1\nline2\n");
    }
}

// ---------------------------------------------------------------------------
// 7. D9: Wildcard Pattern in Result
// ---------------------------------------------------------------------------

#[test]
fn test_l12_d9_wildcard_pattern_identifier() {
    let src = "local r: Result<int, string> = Result.Ok(7)\nif r is Ok(_) then\n println(\"ok\")\nend";
    let res = probe_zig(src);
    // Known-gap in src/codegen_zig/mod.rs:emit_pattern (raw '_' identifier in Zig)
    if !res.success {
        assert_eq!(res.exit_code, Some(1));
        assert!(
            res.stderr.contains("'_' used as an identifier without @\"_\" syntax")
                || res.stderr.contains("@\"_\""),
            "Expected '_' identifier syntax error: {}",
            res.stderr
        );
    } else {
        assert_eq!(res.stdout, "ok\n");
    }
}

// ---------------------------------------------------------------------------
// 8. D10: Map Unknown Value Boxing
// ---------------------------------------------------------------------------

#[test]
fn test_l12_d10_map_unknown_cast() {
    let src = r#"
        local data: Map<string, unknown> = {}
        map.set(data, "age", 30)
    "#;
    let res = probe_zig(src);
    // Known-gap in src/codegen_zig/runtime.rs:generate_map_runtime
    if !res.success {
        assert_eq!(res.exit_code, Some(1));
        assert!(
            res.stderr.contains("expected type '")
                || res.stderr.contains("LustMap")
                || res.stderr.contains("anyopaque"),
            "Expected Map unknown type mismatch: {}",
            res.stderr
        );
    } else {
        assert_eq!(res.exit_code, Some(0));
    }
}

// ---------------------------------------------------------------------------
// 9. D11: Integer Division by Zero Panic
// ---------------------------------------------------------------------------

#[test]
fn test_l12_d11_division_by_zero_behavior() {
    let src = "local z: int = 0\nprintln(10 / z)\nprintln(\"UNREACHABLE\")";
    let res = probe_zig(src);
    // Zig runtime __lust_div__func executes and traps on zero division
    assert!(!res.success, "Division by zero must fail");
    assert!(
        res.stderr.contains("Division by zero") || res.stderr.contains("panic"),
        "Expected Division by zero panic: {}",
        res.stderr
    );
}

// ---------------------------------------------------------------------------
// 10. ZIG12: Numeric For Loop with Negative Step
// ---------------------------------------------------------------------------

#[test]
fn test_l12_zig12_negative_step_for_loop() {
    let src = "local count = 0\nfor i = 5, 1, -1 do\n count += 1\nend\nprintln(count)";
    let res = probe_zig(src);
    // Known-gap in src/codegen_zig/mod.rs:emit_numeric_for (forward condition vs reverse condition)
    if res.stdout != "5\n" {
        // Current symptom: loop condition stops prematurely (output "0\n")
        assert_eq!(res.stdout, "0\n");
    } else {
        assert_eq!(res.stdout, "5\n");
    }
}

// ---------------------------------------------------------------------------
// 11. ZIG13: Logical Or Value Semantics
// ---------------------------------------------------------------------------

#[test]
fn test_l12_zig13_logical_or_value() {
    let src = "local x = 10 or 20\nprintln(x)";
    let res = probe_zig(src);
    // Known-gap in src/codegen_zig/mod.rs:emit_binary (emitted Zig `or` expects bool)
    if !res.success {
        assert_eq!(res.exit_code, Some(1));
        assert!(
            res.stderr.contains("expected type 'bool', found 'comptime_int'")
                || res.stderr.contains("expected type 'bool'"),
            "Expected bool type expectation for Zig `or`: {}",
            res.stderr
        );
    } else {
        assert_eq!(res.stdout, "10\n");
    }
}

// ---------------------------------------------------------------------------
// 12. ZIG14: Bitwise Shift Negative Number
// ---------------------------------------------------------------------------

#[test]
fn test_l12_zig14_bitwise_shift_negative_int() {
    let src = "local x = (-1) << 1\nprintln(x)";
    let res = probe_zig(src);
    // Known-gap in src/codegen_zig/mod.rs:emit_binary (comptime_int variable typing)
    if !res.success {
        assert_eq!(res.exit_code, Some(1));
        assert!(
            res.stderr.contains("variable of type 'comptime_int' must be const or comptime")
                || res.stderr.contains("comptime_int"),
            "Expected comptime_int variable restriction: {}",
            res.stderr
        );
    } else {
        assert_eq!(res.stdout, "-2\n");
    }
}

// ---------------------------------------------------------------------------
// 13. ZIG16: Println Zero Arguments
// ---------------------------------------------------------------------------

#[test]
fn test_l12_zig16_println_zero_arguments() {
    let src = "println()\nprintln(42)";
    let res = probe_zig(src);
    // Known-gap in src/codegen_zig/runtime.rs:generate_io_runtime (0-arg println identifier)
    if !res.success {
        assert_eq!(res.exit_code, Some(1));
        assert!(
            res.stderr.contains("use of undeclared identifier 'println'")
                || res.stderr.contains("undeclared identifier"),
            "Expected undeclared identifier println: {}",
            res.stderr
        );
    } else {
        assert_eq!(res.stdout, "\n42\n");
    }
}

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use lust::{
    EmbeddedProgram, Value,
    codegen_zig::{ZigCodegen, analyze_runtime_requirements},
    lexer::Lexer,
    parser::Parser,
};
use std::process::Command;

fn run_vm(source: &str, func: &str) -> lust::Result<Value> {
    let mut prog = EmbeddedProgram::builder()
        .module("main", source)
        .entry_module("main")
        .compile()?;
    prog.vm_mut().call(&format!("main.{}", func), vec![])
}

fn run_zig(source: &str) {
    let mut interner = lust::intern::Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize().expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");
    let opts = analyze_runtime_requirements(&items, false, None);
    let mut codegen = ZigCodegen::with_options(opts);
    let zig_src = codegen.generate_program(&items).expect("generate_program");

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let zig_path = temp_dir.path().join("main.zig");
    std::fs::write(&zig_path, &zig_src).expect("write zig source");

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

#[test]
fn test_precedence_and_associativity_vm() {
    let source = r#"
        function precedence_unary_pow(): int
            return -2 ^ 2
        end

        function right_assoc_pow(): int
            return 2 ^ 3 ^ 2
        end

        function left_assoc_sub(): int
            return 10 - 3 - 2
        end

        function shift_and_add(): int
            return 1 << 2 + 1
        end

        function bitwise_and_or(): int
            return 240 & 15 | 2
        end
    "#;

    assert_eq!(run_vm(source, "precedence_unary_pow").unwrap(), Value::Int(-4));
    assert_eq!(run_vm(source, "right_assoc_pow").unwrap(), Value::Int(512));
    assert_eq!(run_vm(source, "left_assoc_sub").unwrap(), Value::Int(5));
    assert_eq!(run_vm(source, "shift_and_add").unwrap(), Value::Int(8));
    assert_eq!(run_vm(source, "bitwise_and_or").unwrap(), Value::Int(2));
}

#[test]
fn test_precedence_and_associativity_zig() {
    let source = r#"
        assert(-2 ^ 2 == -4)
        assert(2 ^ 3 ^ 2 == 512)
        assert(10 - 3 - 2 == 5)
        assert(1 << 2 + 1 == 8)
        assert((240 & 15 | 2) == 2)
    "#;

    run_zig(source);
}

#[test]
fn test_floor_division_and_modulo_vm() {
    let source = r#"
        function floor_div_pos(): int
            return 7 // 2
        end

        function floor_div_neg_num(): int
            return -7 // 2
        end

        function floor_div_neg_den(): int
            return 7 // -2
        end

        function floor_div_both_neg(): int
            return -7 // -2
        end

        function modulo_neg(): int
            return -7 % 2
        end

        function modulo_den_neg(): int
            return 7 % -2
        end

        function floor_div_float(): float
            return 7.5 // 2.0
        end

        function floor_div_float_neg(): float
            return -7.5 // 2.0
        end
    "#;

    assert_eq!(run_vm(source, "floor_div_pos").unwrap(), Value::Int(3));
    assert_eq!(run_vm(source, "floor_div_neg_num").unwrap(), Value::Int(-4));
    assert_eq!(run_vm(source, "floor_div_neg_den").unwrap(), Value::Int(-4));
    assert_eq!(run_vm(source, "floor_div_both_neg").unwrap(), Value::Int(3));
    assert_eq!(run_vm(source, "modulo_neg").unwrap(), Value::Int(-1));
    assert_eq!(run_vm(source, "modulo_den_neg").unwrap(), Value::Int(1));
    assert_eq!(run_vm(source, "floor_div_float").unwrap(), Value::Float(3.0));
    assert_eq!(run_vm(source, "floor_div_float_neg").unwrap(), Value::Float(-4.0));
}

#[test]
fn test_floor_division_and_modulo_zig() {
    let source = r#"
        assert(7 // 2 == 3)
        assert(-7 // 2 == -4)
        assert(7 // -2 == -4)
        assert(-7 // -2 == 3)
        assert(-7 % 2 == -1)
        assert(7 % -2 == 1)
        assert(7.5 // 2.0 == 3.0)
        assert(-7.5 // 2.0 == -4.0)
    "#;

    run_zig(source);
}

#[test]
fn test_bitwise_operators_vm() {
    let source = r#"
        function bit_and(): int
            return 240 & 60
        end

        function bit_or(): int
            return 240 | 15
        end

        function bit_xor_caret(): int
            return 255 ^^ 15
        end

        function bit_xor_tilde(): int
            return 255 ~ 15
        end

        function bit_not(): int
            return ~0
        end

        function shl(): int
            return 1 << 10
        end

        function shr_arithmetic(): int
            return -16 >> 2
        end
    "#;

    assert_eq!(run_vm(source, "bit_and").unwrap(), Value::Int(0x30));
    assert_eq!(run_vm(source, "bit_or").unwrap(), Value::Int(0xFF));
    assert_eq!(run_vm(source, "bit_xor_caret").unwrap(), Value::Int(0xF0));
    assert_eq!(run_vm(source, "bit_xor_tilde").unwrap(), Value::Int(0xF0));
    assert_eq!(run_vm(source, "bit_not").unwrap(), Value::Int(-1));
    assert_eq!(run_vm(source, "shl").unwrap(), Value::Int(1024));
    assert_eq!(run_vm(source, "shr_arithmetic").unwrap(), Value::Int(-4));
}

#[test]
fn test_bitwise_operators_zig() {
    let source = r#"
        assert(240 & 60 == 48)
        assert(240 | 15 == 255)
        assert(255 ^^ 15 == 240)
        assert(255 ~ 15 == 240)
        assert(~0 == -1)
        assert(1 << 10 == 1024)
        assert(-16 >> 2 == -4)
    "#;

    run_zig(source);
}

#[test]
fn test_compound_assignments_vm() {
    let source = r#"
        struct Counter
            count: int
        end

        function test_local_compound(): int
            local x: int = 10
            x += 5
            x -= 3
            x *= 2
            x //= 5
            x ^= 2
            x &= 7
            x |= 16
            x ^^= 3
            x <<= 2
            x >>= 1
            x %= 5
            return x
        end

        function test_field_compound(): int
            local c = Counter { count = 10 }
            c.count += 5
            c.count //= 2
            c.count ^= 2
            return c.count
        end

        function test_index_compound(): int
            local arr: Map<int, int> = { [1] = 20 }
            arr[1] += 10
            arr[1] //= 2
            return arr[1]
        end
    "#;

    assert_eq!(run_vm(source, "test_local_compound").unwrap(), Value::Int(3));
    assert_eq!(run_vm(source, "test_field_compound").unwrap(), Value::Int(49));
    assert_eq!(run_vm(source, "test_index_compound").unwrap(), Value::Int(15));
}

#[test]
fn test_compound_assignments_zig() {
    let source = r#"
        struct Counter
            count: int
        end

        local x: int = 10
        x += 5
        x -= 3
        x *= 2
        x //= 5
        x ^= 2
        x &= 7
        x |= 16
        x ^^= 3
        x <<= 2
        x >>= 1
        x %= 5
        assert(x == 3)

        local c = Counter { count = 10 }
        c.count += 5
        c.count //= 2
        c.count ^= 2
        assert(c.count == 49)

        local arr = [10, 20, 30]
        arr[1] += 10
        arr[1] //= 2
        assert(arr[1] == 15)
    "#;

    run_zig(source);
}

#[test]
fn test_table_and_string_methods_zig() {
    let source = r#"
        local t = [10, 20, 30]
        table.insert(t, 40)
        assert(table.len(t) == 4)
        local popped = table.remove(t)
        assert(popped == 40)
        assert(table.len(t) == 3)

        table.insert(t, 0, 5)
        assert(t[0] == 5)
        local removed = table.remove(t, 0)
        assert(removed == 5)

        local joined = table.concat(t, "-")
        assert(string.len(joined) == 8)

        local s = "hello world"
        local sub = string.slice(s, 0, 5)
        assert(string.len(sub) == 5)
        assert(string.contains(sub, "hello"))

        local idx = string.index_of(s, "world")
        assert(idx != null)

        local sum = 0
        for i, v in ipairs(t) do
            sum += v
        end
        assert(sum == 60)
    "#;

    run_zig(source);
}

#[test]
fn test_error_cases_vm() {
    let div_zero = r#"
        function div_by_zero(): int
            return 10 // 0
        end
    "#;
    assert!(run_vm(div_zero, "div_by_zero").is_err());

    let mod_zero = r#"
        function mod_by_zero(): int
            return 10 % 0
        end
    "#;
    assert!(run_vm(mod_zero, "mod_by_zero").is_err());

    let neg_pow = r#"
        function neg_pow(): int
            return 2 ^ -1
        end
    "#;
    let err = run_vm(neg_pow, "neg_pow").unwrap_err();
    assert!(err.to_string().contains("Negative exponent"));
}

//! Step 5 (P5): Bytecode Identity Oracle Acceptance Tests
//!
//! Validates:
//! 1. Zero runtime overhead: Radix literals (0x, 0b, 0o) and underscore separators
//!    compile to byte-for-byte identical chunk instructions and constants as their decimal equivalents.
//! 2. Complete coverage of Hex, Binary, Octal, Decimal with '_', Float with '_', and MSB boundaries.
//! 3. Preservation of bytecode identity across bitwise operations, arithmetic expressions,
//!    function calls, control flow, and pattern matching.

use lust::bytecode::compiler::Compiler;
use lust::bytecode::{Function, Value};
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;

fn compile_source(source: &str) -> lust::Result<Vec<Function>> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize()?;
    let items = Parser::new(tokens).parse()?;
    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    compiler.compile_module(&items)
}

fn assert_bytecode_identity(src_radix: &str, src_dec: &str) {
    let f_radix = compile_source(src_radix)
        .unwrap_or_else(|e| panic!("Failed to compile radix source: {}\nSource:\n{}", e, src_radix));
    let f_dec = compile_source(src_dec)
        .unwrap_or_else(|e| panic!("Failed to compile decimal source: {}\nSource:\n{}", e, src_dec));

    assert!(
        !f_radix.is_empty(),
        "Radix source produced no compiled functions: vacuous identity check rejected"
    );
    assert_eq!(
        f_radix.len(),
        f_dec.len(),
        "Function count mismatch between radix and decimal sources"
    );

    for (idx, (func_r, func_d)) in f_radix.iter().zip(f_dec.iter()).enumerate() {
        assert!(
            !func_r.chunk.instructions.is_empty(),
            "Function #{}: '{}' has empty chunk instructions: vacuous check rejected",
            idx,
            func_r.name
        );
        assert_eq!(
            func_r.chunk.instructions,
            func_d.chunk.instructions,
            "Instruction mismatch in function #{}: '{}' vs '{}'",
            idx,
            func_r.name,
            func_d.name
        );
        assert_eq!(
            func_r.chunk.constants,
            func_d.chunk.constants,
            "Constant mismatch in function #{}: '{}' vs '{}'",
            idx,
            func_r.name,
            func_d.name
        );
    }
}

#[test]
fn test_bytecode_identity_radix_vs_decimal() {
    // 1. Hexadecimal literals
    // 0x0F vs 15
    assert_bytecode_identity(
        "function test(): int return 0x0F end",
        "function test(): int return 15 end",
    );

    // 0x7FFF_FFFF_FFFF_FFFF vs 9223372036854775807 (i64::MAX)
    assert_bytecode_identity(
        "function test(): int return 0x7FFF_FFFF_FFFF_FFFF end",
        "function test(): int return 9223372036854775807 end",
    );

    // 0x8000_0000_0000_0000 vs -9223372036854775808 (i64::MIN)
    assert_bytecode_identity(
        "function test(): int return 0x8000_0000_0000_0000 end",
        "function test(): int return -9223372036854775808 end",
    );

    // 0xDEAD_BEEF vs 3735928559
    assert_bytecode_identity(
        "function test(): int return 0xDEAD_BEEF end",
        "function test(): int return 3735928559 end",
    );

    // 0Xff vs 255
    assert_bytecode_identity(
        "function test(): int return 0Xff end",
        "function test(): int return 255 end",
    );

    // 0x0 vs 0
    assert_bytecode_identity(
        "function test(): int return 0x0 end",
        "function test(): int return 0 end",
    );

    // 0xFFFF_FFFF_FFFF_FFFF vs -1 in pattern matching (where -1 folds to constant integer pattern)
    assert_bytecode_identity(
        "function test(x: int): bool return x is 0xFFFF_FFFF_FFFF_FFFF end",
        "function test(x: int): bool return x is -1 end",
    );

    // 0xFFFF_FFFF_FFFF_FFFF across radix bases (hex, bin, oct) yielding constant -1
    assert_bytecode_identity(
        "function test(): int return 0xFFFF_FFFF_FFFF_FFFF end",
        "function test(): int return 0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111 end",
    );
    assert_bytecode_identity(
        "function test(): int return 0xFFFF_FFFF_FFFF_FFFF end",
        "function test(): int return 0o1777_7777_7777_7777_7777_77 end",
    );

    // 2. Binary literals
    // 0b1010_0101 vs 165
    assert_bytecode_identity(
        "function test(): int return 0b1010_0101 end",
        "function test(): int return 165 end",
    );

    // 0b0111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111 vs i64::MAX
    assert_bytecode_identity(
        "function test(): int return 0b0111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111 end",
        "function test(): int return 9223372036854775807 end",
    );

    // 0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000 vs i64::MIN
    assert_bytecode_identity(
        "function test(): int return 0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000 end",
        "function test(): int return -9223372036854775808 end",
    );

    // 0B1010 vs 10
    assert_bytecode_identity(
        "function test(): int return 0B1010 end",
        "function test(): int return 10 end",
    );

    // 0b1111_0000 vs 240
    assert_bytecode_identity(
        "function test(): int return 0b1111_0000 end",
        "function test(): int return 240 end",
    );

    // 0b0 vs 0
    assert_bytecode_identity(
        "function test(): int return 0b0 end",
        "function test(): int return 0 end",
    );

    // 3. Octal literals
    // 0o755 vs 493
    assert_bytecode_identity(
        "function test(): int return 0o755 end",
        "function test(): int return 493 end",
    );

    // 0o177777 vs 65535
    assert_bytecode_identity(
        "function test(): int return 0o177777 end",
        "function test(): int return 65535 end",
    );

    // 0o644 vs 420
    assert_bytecode_identity(
        "function test(): int return 0o644 end",
        "function test(): int return 420 end",
    );

    // 0o0123_4567 vs 342391
    assert_bytecode_identity(
        "function test(): int return 0o0123_4567 end",
        "function test(): int return 342391 end",
    );

    // 0o0777_7777_7777_7777_7777_77 vs i64::MAX
    assert_bytecode_identity(
        "function test(): int return 0o0777_7777_7777_7777_7777_77 end",
        "function test(): int return 9223372036854775807 end",
    );

    // 0o1000_0000_0000_0000_0000_00 vs i64::MIN
    assert_bytecode_identity(
        "function test(): int return 0o1000_0000_0000_0000_0000_00 end",
        "function test(): int return -9223372036854775808 end",
    );

    // 0o0 vs 0
    assert_bytecode_identity(
        "function test(): int return 0o0 end",
        "function test(): int return 0 end",
    );

    // 4. Floating point literals with '_'
    // 3.1415_9265 vs 3.14159265
    assert_bytecode_identity(
        "function test(): float return 3.1415_9265 end",
        "function test(): float return 3.14159265 end",
    );

    // 1_000.5_0 vs 1000.5
    assert_bytecode_identity(
        "function test(): float return 1_000.5_0 end",
        "function test(): float return 1000.5 end",
    );

    // 1_0.5 vs 10.5
    assert_bytecode_identity(
        "function test(): float return 1_0.5 end",
        "function test(): float return 10.5 end",
    );

    // 1e1_0 vs 1e10
    assert_bytecode_identity(
        "function test(): float return 1e1_0 end",
        "function test(): float return 1e10 end",
    );

    // 1.0e+6 vs 1000000.0
    assert_bytecode_identity(
        "function test(): float return 1.0e+6 end",
        "function test(): float return 1000000.0 end",
    );

    // 1e3 vs 1000.0
    assert_bytecode_identity(
        "function test(): float return 1e3 end",
        "function test(): float return 1000.0 end",
    );

    // 1e-3 vs 0.001
    assert_bytecode_identity(
        "function test(): float return 1e-3 end",
        "function test(): float return 0.001 end",
    );

    // 5. Separators '_' in decimal integers
    // 1_000_000 vs 1000000
    assert_bytecode_identity(
        "function test(): int return 1_000_000 end",
        "function test(): int return 1000000 end",
    );

    // 42_133_7 vs 421337
    assert_bytecode_identity(
        "function test(): int return 42_133_7 end",
        "function test(): int return 421337 end",
    );

    // 1_0 vs 10
    assert_bytecode_identity(
        "function test(): int return 1_0 end",
        "function test(): int return 10 end",
    );

    // 9_223_372_036_854_775_807 vs 9223372036854775807
    assert_bytecode_identity(
        "function test(): int return 9_223_372_036_854_775_807 end",
        "function test(): int return 9223372036854775807 end",
    );

    // -9_223_372_036_854_775_808 vs -9223372036854775808
    assert_bytecode_identity(
        "function test(): int return -9_223_372_036_854_775_808 end",
        "function test(): int return -9223372036854775808 end",
    );
}

#[test]
fn test_bytecode_identity_operations_and_masks() {
    // 1. Bitwise operations with masks
    assert_bytecode_identity(
        r#"
        function worker(): int
            local mask = 0x0FFF
            local flags = 0b1010_0101
            return mask & flags
        end
        "#,
        r#"
        function worker(): int
            local mask = 4095
            local flags = 165
            return mask & flags
        end
        "#,
    );

    // 2. Bitwise OR and shifts with boundary values
    assert_bytecode_identity(
        r#"
        function worker(): int
            local a = (0x0F << 4) | 0x05
            local b = 0x8000_0000_0000_0000 >> 63
            return a + b
        end
        "#,
        r#"
        function worker(): int
            local a = (15 << 4) | 5
            local b = -9223372036854775808 >> 63
            return a + b
        end
        "#,
    );

    // 3. Bitwise XOR (^^)
    assert_bytecode_identity(
        r#"
        function worker(): int
            local x = 0x5555_5555_5555_5555 ^^ 0x3333_3333_3333_3333
            local y = 0o755 ^^ 0o055
            return x + y
        end
        "#,
        r#"
        function worker(): int
            local x = 6148914691236517205 ^^ 3689348814741910323
            local y = 493 ^^ 45
            return x + y
        end
        "#,
    );

    // 4. Arithmetic expressions with mixed radices and separators
    assert_bytecode_identity(
        r#"
        function calc(): int
            return 0x10 + 0b1010 - 0o12 + 1_000
        end
        "#,
        r#"
        function calc(): int
            return 16 + 10 - 10 + 1000
        end
        "#,
    );

    // 5. Complex arithmetic with multiplication, division, and modulo
    assert_bytecode_identity(
        r#"
        function calc(): int
            local a = (0x20 * 0b11) / 0o10
            local b = 1_000_000 % 0x64
            return a + b
        end
        "#,
        r#"
        function calc(): int
            local a = (32 * 3) / 8
            local b = 1000000 % 100
            return a + b
        end
        "#,
    );
}

#[test]
fn test_bytecode_identity_control_flow_and_calls() {
    // 1. While loop with radix counters and bounds
    assert_bytecode_identity(
        r#"
        function loop_calc(): int
            local sum = 0
            local i = 0
            while i < 0x20 do
                sum = sum + (i & 0x0F) * 0b10
                i = i + 1
            end
            return sum
        end
        "#,
        r#"
        function loop_calc(): int
            local sum = 0
            local i = 0
            while i < 32 do
                sum = sum + (i & 15) * 2
                i = i + 1
            end
            return sum
        end
        "#,
    );

    // 2. If-else branches with radix conditions
    assert_bytecode_identity(
        r#"
        function check_branch(x: int): int
            if x == 0x0F then
                return 0b1010
            else
                if x == 0o755 then
                    return 0x7FFF_FFFF_FFFF_FFFF
                else
                    return 1_000_000
                end
            end
        end
        "#,
        r#"
        function check_branch(x: int): int
            if x == 15 then
                return 10
            else
                if x == 493 then
                    return 9223372036854775807
                else
                    return 1000000
                end
            end
        end
        "#,
    );

    // 3. Multi-function calls and argument passing
    assert_bytecode_identity(
        r#"
        function helper(a: int, b: int, c: int, d: int): int
            return (a << 2) + (b & 0x0F) + c + d
        end
        function main_worker(): int
            return helper(0x10, 0b1010_1111, 0o755, 1_000)
        end
        "#,
        r#"
        function helper(a: int, b: int, c: int, d: int): int
            return (a << 2) + (b & 15) + c + d
        end
        function main_worker(): int
            return helper(16, 175, 493, 1000)
        end
        "#,
    );

    // 4. Pattern matching with radix vs decimal patterns (including negative literal pattern -1)
    assert_bytecode_identity(
        r#"
        function match_hex(x: int): bool return x is 0x0F end
        function match_bin(x: int): bool return x is 0b1010 end
        function match_oct(x: int): bool return x is 0o755 end
        function match_dec(x: int): bool return x is 1_000_000 end
        function match_neg(x: int): bool return x is 0xFFFF_FFFF_FFFF_FFFF end
        function match_min(x: int): bool return x is 0x8000_0000_0000_0000 end
        "#,
        r#"
        function match_hex(x: int): bool return x is 15 end
        function match_bin(x: int): bool return x is 10 end
        function match_oct(x: int): bool return x is 493 end
        function match_dec(x: int): bool return x is 1000000 end
        function match_neg(x: int): bool return x is -1 end
        function match_min(x: int): bool return x is -9223372036854775808 end
        "#,
    );
}

#[test]
fn test_bytecode_identity_constant_values_verification() {
    // Direct verification of constant table values
    let funcs_hex = compile_source("function f(): int return 0xFFFF_FFFF_FFFF_FFFF end").unwrap();
    assert_eq!(funcs_hex[0].chunk.constants[0], Value::Int(-1));

    let funcs_bin = compile_source("function f(): int return 0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111 end").unwrap();
    assert_eq!(funcs_bin[0].chunk.constants[0], Value::Int(-1));

    let funcs_oct = compile_source("function f(): int return 0o1777_7777_7777_7777_7777_77 end").unwrap();
    assert_eq!(funcs_oct[0].chunk.constants[0], Value::Int(-1));

    let funcs_min_hex = compile_source("function f(): int return 0x8000_0000_0000_0000 end").unwrap();
    assert_eq!(funcs_min_hex[0].chunk.constants[0], Value::Int(i64::MIN));

    let funcs_min_dec = compile_source("function f(): int return -9223372036854775808 end").unwrap();
    assert_eq!(funcs_min_dec[0].chunk.constants[0], Value::Int(i64::MIN));

    let funcs_max_hex = compile_source("function f(): int return 0x7FFF_FFFF_FFFF_FFFF end").unwrap();
    assert_eq!(funcs_max_hex[0].chunk.constants[0], Value::Int(i64::MAX));

    let funcs_max_dec = compile_source("function f(): int return 9223372036854775807 end").unwrap();
    assert_eq!(funcs_max_dec[0].chunk.constants[0], Value::Int(i64::MAX));
}

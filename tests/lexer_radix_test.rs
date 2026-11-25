#![cfg(feature = "std")]

use lust::ast::{ExprKind, ItemKind, Literal, StmtKind};
use lust::intern::Interner;
use lust::lexer::{Lexer, Token, TokenKind};
use lust::number::{parse_float_literal, parse_integer_literal, NumberParseError};
use lust::{Compiler, Function, LustConfig, LustError, Parser, TypeChecker};

fn tokenize(source: &str) -> Result<Vec<Token>, LustError> {
    let mut interner = Interner::new();
    Lexer::new(source, &mut interner).tokenize()
}

fn parse_program(source: &str) -> Result<Vec<lust::Item>, LustError> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize()?;
    Parser::new(tokens).parse()
}

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

// =========================================================================
// Category 1: Hexadecimal Valid & MSB Boundary Vectors
// =========================================================================
#[test]
fn test_category_1_hexadecimal_valid_and_msb_boundary_vectors() {
    let cases = [
        ("0x0", 0i64),
        ("0x0F", 15i64),
        ("0Xff", 255i64),
        ("0xDEAD_BEEF", 3735928559i64),
        ("0x7FFF_FFFF_FFFF_FFFF", i64::MAX),
        ("0x8000_0000_0000_0000", i64::MIN),
        ("0xFFFF_FFFF_FFFF_FFFF", -1i64),
    ];

    for (src, expected) in cases {
        let tokens = tokenize(src).unwrap_or_else(|e| panic!("Failed to tokenize '{}': {:?}", src, e));
        assert_eq!(tokens[0].kind, TokenKind::Integer, "Token kind must be Integer for '{}'", src);
        assert_eq!(tokens[0].lexeme, src, "Lexeme must match source for '{}'", src);
        let parsed = parse_integer_literal(&tokens[0].lexeme)
            .unwrap_or_else(|e| panic!("Failed to parse integer literal '{}': {:?}", src, e));
        assert_eq!(parsed, expected, "Mismatch for '{}'", src);
    }
}

// =========================================================================
// Category 2: Binary Valid & Boundary Vectors
// =========================================================================
#[test]
fn test_category_2_binary_valid_and_boundary_vectors() {
    let cases = [
        ("0b0", 0i64),
        ("0b1", 1i64),
        ("0B1010", 10i64),
        ("0b1111_0000", 240i64),
        ("0b0111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111", i64::MAX),
        ("0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000", i64::MIN),
        ("0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111", -1i64),
    ];

    for (src, expected) in cases {
        let tokens = tokenize(src).unwrap_or_else(|e| panic!("Failed to tokenize '{}': {:?}", src, e));
        assert_eq!(tokens[0].kind, TokenKind::Integer, "Token kind must be Integer for '{}'", src);
        assert_eq!(tokens[0].lexeme, src, "Lexeme must match source for '{}'", src);
        let parsed = parse_integer_literal(&tokens[0].lexeme)
            .unwrap_or_else(|e| panic!("Failed to parse integer literal '{}': {:?}", src, e));
        assert_eq!(parsed, expected, "Mismatch for '{}'", src);
    }
}

// =========================================================================
// Category 3: Octal Valid & Boundary Vectors
// =========================================================================
#[test]
fn test_category_3_octal_valid_and_boundary_vectors() {
    let cases = [
        ("0o0", 0i64),
        ("0o7", 7i64),
        ("0O755", 493i64),
        ("0o644", 420i64),
        ("0o177777", 65535i64),
        ("0o0123_4567", 342391i64),
        ("0o0777_7777_7777_7777_7777_77", i64::MAX),
        ("0o1000_0000_0000_0000_0000_00", i64::MIN),
        ("0o1777_7777_7777_7777_7777_77", -1i64),
    ];

    for (src, expected) in cases {
        let tokens = tokenize(src).unwrap_or_else(|e| panic!("Failed to tokenize '{}': {:?}", src, e));
        assert_eq!(tokens[0].kind, TokenKind::Integer, "Token kind must be Integer for '{}'", src);
        assert_eq!(tokens[0].lexeme, src, "Lexeme must match source for '{}'", src);
        let parsed = parse_integer_literal(&tokens[0].lexeme)
            .unwrap_or_else(|e| panic!("Failed to parse integer literal '{}': {:?}", src, e));
        assert_eq!(parsed, expected, "Mismatch for '{}'", src);
    }
}

// =========================================================================
// Category 4: Decimal Integer with '_' Separator
// =========================================================================
#[test]
fn test_category_4_decimal_integer_with_underscore_separator() {
    let cases = [
        ("1_000_000", 1000000i64),
        ("42_133_7", 421337i64),
        ("1_0", 10i64),
        ("9_223_372_036_854_775_807", i64::MAX),
    ];

    for (src, expected) in cases {
        let tokens = tokenize(src).unwrap_or_else(|e| panic!("Failed to tokenize '{}': {:?}", src, e));
        assert_eq!(tokens[0].kind, TokenKind::Integer, "Token kind must be Integer for '{}'", src);
        assert_eq!(tokens[0].lexeme, src, "Lexeme must match source for '{}'", src);
        let parsed = parse_integer_literal(&tokens[0].lexeme)
            .unwrap_or_else(|e| panic!("Failed to parse integer literal '{}': {:?}", src, e));
        assert_eq!(parsed, expected, "Mismatch for '{}'", src);
    }
}

// =========================================================================
// Category 5: Floating Point with '_' and Exponents
// =========================================================================
#[test]
fn test_category_5_floating_point_with_underscore_and_exponents() {
    let cases = [
        ("3.1415_9265", 3.14159265f64),
        ("1_000.5_0", 1000.5f64),
        ("1_0.5", 10.5f64),
        ("1e3", 1000.0f64),
        ("1e-3", 0.001f64),
        ("1.0e+6", 1000000.0f64),
        ("1e1_0", 1e10f64),
    ];

    for (src, expected) in cases {
        let tokens = tokenize(src).unwrap_or_else(|e| panic!("Failed to tokenize '{}': {:?}", src, e));
        assert_eq!(tokens[0].kind, TokenKind::Float, "Token kind must be Float for '{}'", src);
        assert_eq!(tokens[0].lexeme, src, "Lexeme must match source for '{}'", src);
        let parsed = parse_float_literal(&tokens[0].lexeme)
            .unwrap_or_else(|e| panic!("Failed to parse float literal '{}': {:?}", src, e));
        assert!((parsed - expected).abs() < 1e-12, "Mismatch for '{}': got {}, expected {}", src, parsed, expected);
    }
}

// =========================================================================
// Category 6: Range Operator Disambiguation
// =========================================================================
#[test]
fn test_category_6_range_operator_disambiguation() {
    // 0..10 -> [Integer("0"), DoubleDot, Integer("10")]
    let tokens = tokenize("0..10").expect("tokenize 0..10");
    assert_eq!(tokens[0].kind, TokenKind::Integer);
    assert_eq!(tokens[0].lexeme, "0");
    assert_eq!(tokens[1].kind, TokenKind::DoubleDot);
    assert_eq!(tokens[2].kind, TokenKind::Integer);
    assert_eq!(tokens[2].lexeme, "10");

    // 0..=10 -> [Integer("0"), DoubleDot, Equal, Integer("10")]
    let tokens = tokenize("0..=10").expect("tokenize 0..=10");
    assert_eq!(tokens[0].kind, TokenKind::Integer);
    assert_eq!(tokens[0].lexeme, "0");
    assert_eq!(tokens[1].kind, TokenKind::DoubleDot);
    assert_eq!(tokens[2].kind, TokenKind::Equal);
    assert_eq!(tokens[3].kind, TokenKind::Integer);
    assert_eq!(tokens[3].lexeme, "10");

    // 1_0..2_0 -> [Integer("1_0"), DoubleDot, Integer("2_0")]
    let tokens = tokenize("1_0..2_0").expect("tokenize 1_0..2_0");
    assert_eq!(tokens[0].kind, TokenKind::Integer);
    assert_eq!(tokens[0].lexeme, "1_0");
    assert_eq!(tokens[1].kind, TokenKind::DoubleDot);
    assert_eq!(tokens[2].kind, TokenKind::Integer);
    assert_eq!(tokens[2].lexeme, "2_0");

    // Disambiguation: 1.method() does not consume dot as float
    let tokens = tokenize("1.method()").expect("tokenize 1.method()");
    assert_eq!(tokens[0].kind, TokenKind::Integer);
    assert_eq!(tokens[0].lexeme, "1");
    assert_eq!(tokens[1].kind, TokenKind::Dot);
    assert_eq!(tokens[2].kind, TokenKind::Identifier);

    // Disambiguation: 1. followed by space
    let tokens = tokenize("1. ").expect("tokenize '1. '");
    assert_eq!(tokens[0].kind, TokenKind::Integer);
    assert_eq!(tokens[0].lexeme, "1");
    assert_eq!(tokens[1].kind, TokenKind::Dot);
}

// =========================================================================
// Category 7: Radix Prefix Syntax Teeth (Missing Digits & Leading '_')
// =========================================================================
#[test]
fn test_category_7_radix_prefix_syntax_teeth() {
    let missing_digit_cases = ["0x", "0b", "0o", "0x ", "0b;", "0o)"];
    for src in missing_digit_cases {
        let err = tokenize(src).unwrap_err();
        match err {
            LustError::LexerError { message, .. } => {
                assert!(
                    message.contains("Expected at least one valid digit after radix prefix"),
                    "For '{}', got unexpected message: {}",
                    src,
                    message
                );
            }
            other => panic!("Expected LexerError for '{}', got: {:?}", src, other),
        }
    }

    let leading_underscore_cases = ["0x_1", "0b_0", "0o_7"];
    for src in leading_underscore_cases {
        let err = tokenize(src).unwrap_err();
        match err {
            LustError::LexerError { message, .. } => {
                assert!(
                    message.contains("Radix prefix cannot be immediately followed by an underscore"),
                    "For '{}', got unexpected message: {}",
                    src,
                    message
                );
            }
            other => panic!("Expected LexerError for '{}', got: {:?}", src, other),
        }
    }
}

// =========================================================================
// Category 8: Digit Domain & Malformed Token Teeth
// =========================================================================
#[test]
fn test_category_8_digit_domain_and_malformed_token_teeth() {
    let radix_digit_cases = ["0xG12", "0b102", "0b10foo", "0x1_G", "0o78"];
    for src in radix_digit_cases {
        let err = tokenize(src).unwrap_err();
        match err {
            LustError::LexerError { message, .. } => {
                assert!(
                    message.contains("Invalid digit for radix literal"),
                    "For '{}', got unexpected message: {}",
                    src,
                    message
                );
            }
            other => panic!("Expected LexerError for '{}', got: {:?}", src, other),
        }
    }

    let decimal_suffix_cases = ["100abc", "42xyz"];
    for src in decimal_suffix_cases {
        let err = tokenize(src).unwrap_err();
        match err {
            LustError::LexerError { message, .. } => {
                assert!(
                    message.contains("Invalid trailing characters in number literal"),
                    "For '{}', got unexpected message: {}",
                    src,
                    message
                );
            }
            other => panic!("Expected LexerError for '{}', got: {:?}", src, other),
        }
    }
}

// =========================================================================
// Category 9: Underscore Placement Teeth (Double, Trailing, Dot/Exp Adjacency)
// =========================================================================
#[test]
fn test_category_9_underscore_placement_teeth() {
    // Consecutive underscores
    let consecutive_cases = ["1__0", "0x1__0", "0b1__0", "0o1__0", "1.0__1", "1e1__0"];
    for src in consecutive_cases {
        let err = tokenize(src).unwrap_err();
        match err {
            LustError::LexerError { message, .. } => {
                assert!(
                    message.contains("Consecutive underscores in number literal are not allowed"),
                    "For '{}', got unexpected message: {}",
                    src,
                    message
                );
            }
            other => panic!("Expected LexerError for '{}', got: {:?}", src, other),
        }
    }

    // Trailing underscores in integer / radix
    let trailing_integer_cases = ["100_", "0xFF_", "0b10_", "0o77_"];
    for src in trailing_integer_cases {
        let err = tokenize(src).unwrap_err();
        match err {
            LustError::LexerError { message, .. } => {
                assert!(
                    message.contains("Trailing underscore in number literal is not allowed"),
                    "For '{}', got unexpected message: {}",
                    src,
                    message
                );
            }
            other => panic!("Expected LexerError for '{}', got: {:?}", src, other),
        }
    }

    // Trailing underscore in float
    let trailing_float_cases = ["1.5_", "100.0_"];
    for src in trailing_float_cases {
        let err = tokenize(src).unwrap_err();
        match err {
            LustError::LexerError { message, .. } => {
                assert!(
                    message.contains("Trailing underscore in float literal is not allowed"),
                    "For '{}', got unexpected message: {}",
                    src,
                    message
                );
            }
            other => panic!("Expected LexerError for '{}', got: {:?}", src, other),
        }
    }

    // Underscore preceding / following decimal point
    let dot_adjacent_cases = [
        ("1._0", "Underscore immediately following decimal point is not allowed"),
        ("1_.0", "Underscore immediately preceding decimal point is not allowed"),
    ];
    for (src, expected_msg) in dot_adjacent_cases {
        let err = tokenize(src).unwrap_err();
        match err {
            LustError::LexerError { message, .. } => {
                assert!(
                    message.contains(expected_msg),
                    "For '{}', expected message containing '{}', got: {}",
                    src,
                    expected_msg,
                    message
                );
            }
            other => panic!("Expected LexerError for '{}', got: {:?}", src, other),
        }
    }

    // Underscore preceding / following exponent
    let exp_adjacent_cases = [
        ("1.0_e3", "Underscore immediately preceding exponent is not allowed"),
        ("1_e3", "Underscore immediately preceding exponent is not allowed"),
        ("1e_3", "Underscore immediately following exponent indicator or sign is not allowed"),
        ("1e+_3", "Underscore immediately following exponent indicator or sign is not allowed"),
        ("1e-_3", "Underscore immediately following exponent indicator or sign is not allowed"),
        ("1e3_", "Trailing underscore in number literal is not allowed"),
    ];
    for (src, expected_msg) in exp_adjacent_cases {
        let err = tokenize(src).unwrap_err();
        match err {
            LustError::LexerError { message, .. } => {
                assert!(
                    message.contains(expected_msg),
                    "For '{}', expected message containing '{}', got: {}",
                    src,
                    expected_msg,
                    message
                );
            }
            other => panic!("Expected LexerError for '{}', got: {:?}", src, other),
        }
    }
}

// =========================================================================
// Category 10: Numeric Overflow Teeth
// =========================================================================
#[test]
fn test_category_10_numeric_overflow_teeth() {
    // 1. parse_integer_literal overflow
    assert_eq!(
        parse_integer_literal("0x1_0000_0000_0000_0000").unwrap_err(),
        NumberParseError::Overflow("exceeds u64::MAX")
    );
    assert_eq!(
        parse_integer_literal("0b1_00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000").unwrap_err(),
        NumberParseError::Overflow("exceeds u64::MAX")
    );
    assert_eq!(
        parse_integer_literal("0o2000_0000_0000_0000_0000_00").unwrap_err(),
        NumberParseError::Overflow("exceeds u64::MAX")
    );
    assert_eq!(
        parse_integer_literal("9223372036854775808").unwrap_err(),
        NumberParseError::Overflow("exceeds i64::MAX")
    );

    // 2. Parser compile failure on standalone > i64::MAX or > u64::MAX
    let err_dec = compile_source("function f(): int\nreturn 9223372036854775808\nend").unwrap_err();
    assert!(
        format!("{}", err_dec).contains("exceeds i64::MAX"),
        "Expected decimal overflow in parser, got: {}",
        err_dec
    );

    let err_hex = compile_source("function f(): int\nreturn 0x1_0000_0000_0000_0000\nend").unwrap_err();
    assert!(
        format!("{}", err_hex).contains("exceeds u64::MAX"),
        "Expected radix overflow in parser, got: {}",
        err_hex
    );

    // 3. Successful parse of -9223372036854775808 into Literal::Integer(i64::MIN) via parse_unary
    let items = parse_program("function f(): int\nreturn -9223372036854775808\nend").expect("parse i64::MIN program");
    if let ItemKind::Function(f) = &items[0].kind {
        if let StmtKind::Return(exprs) = &f.body[0].kind {
            match exprs[0].kind {
                ExprKind::Literal(Literal::Integer(val)) => {
                    assert_eq!(val, i64::MIN, "Must parse directly into i64::MIN");
                }
                ref other => panic!("Expected Literal(Integer(i64::MIN)), got: {:?}", other),
            }
        } else {
            panic!("Expected return statement");
        }
    } else {
        panic!("Expected function item");
    }

    // Also compile to bytecode to verify chunk constant
    let f_min = compile_source("function f(): int\nreturn -9223372036854775808\nend").expect("compile i64::MIN");
    assert_eq!(f_min[0].chunk.constants[0], lust::Value::Int(i64::MIN));
}

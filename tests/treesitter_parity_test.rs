#![cfg(feature = "std")]

use lust::intern::Interner;
use lust::lexer::{Lexer, Token, TokenKind};
use lust::LustError;
use tree_sitter::{Node, Parser};

/// Helper to convert (1-based line, 1-based column) from Lexer into byte offset in source.
fn line_col_to_byte_offset(source: &str, target_line: usize, target_col: usize) -> usize {
    let mut line = 1;
    let mut col = 1;
    for (idx, ch) in source.char_indices() {
        if line == target_line && col == target_col {
            return idx;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    source.len()
}

/// Recursively collect all nodes with `kind() == "number"` in Tree-sitter AST.
fn collect_ts_number_nodes<'a>(
    node: Node<'a>,
    source: &'a str,
    out: &mut Vec<(usize, usize, &'a str)>,
) {
    if node.kind() == "number" {
        out.push((
            node.start_byte(),
            node.end_byte(),
            &source[node.start_byte()..node.end_byte()],
        ));
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_ts_number_nodes(child, source, out);
    }
}

/// Recursively check if tree contains any ERROR node.
fn tree_has_error_node(node: Node<'_>) -> bool {
    if node.is_error() || node.kind() == "ERROR" {
        return true;
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if tree_has_error_node(child) {
            return true;
        }
    }
    false
}

fn make_ts_parser() -> Parser {
    let mut parser = Parser::new();
    parser
        .set_language(tree_sitter_lust::language())
        .expect("Failed to load Lust tree-sitter grammar");
    parser
}

#[test]
fn test_treesitter_rust_lexer_parity_valid_corpus() {
    let mut parser = make_ts_parser();

    // Corpus of at least 40 valid expressions:
    // hex, binary, octal, decimal with _, boundary i64::MAX / i64::MIN, floats with _, exponents,
    // and complex expressions containing numbers.
    let corpus: &[&str] = &[
        // Hexadecimal
        "0x0",
        "0x0F",
        "0Xff",
        "0xDEAD_BEEF",
        "0x7FFF_FFFF_FFFF_FFFF",
        "0x8000_0000_0000_0000",
        "0xFFFF_FFFF_FFFF_FFFF",
        "0x12_34_56_78",
        "0xCAFE_BABE",
        "0x0123_4567_89ab_cdef",
        // Binary
        "0b0",
        "0b1",
        "0B1010",
        "0b1111_0000",
        "0b1010_0101",
        "0b0111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111",
        "0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111",
        "0b0000_0001",
        "0b1100_0011",
        // Octal
        "0o0",
        "0o7",
        "0O755",
        "0o644",
        "0o177777",
        "0o0123_4567",
        "0o77_77",
        "0o52",
        // Decimal integers with _ and boundaries
        "0",
        "42",
        "1_000_000",
        "42_133_7",
        "1_0",
        "9223372036854775807",
        "9_223_372_036_854_775_807",
        "-9223372036854775808",
        "-9_223_372_036_854_775_808",
        "123_456_789",
        "999_999",
        // Floating point with _ and exponents
        "3.1415_9265",
        "1_000.5_0",
        "1_0.5",
        "0.0",
        "1e3",
        "1e-3",
        "1.0e+6",
        "1e1_0",
        "1.5e-1_0",
        "2.718_281_828",
        "100.001",
        "1e+5",
        "0.123_456",
        "123_456.789_012",
        // Complex expressions with multiple numbers
        "0x1_0",
        "1.5e-4",
        "1.foo()",
        "local x = 0xDEAD_BEEF;",
        "local y: int = 0b1010_1010 + 0o755;",
        "local z = 1_000_000 * 3.1415_9265;",
        "return 0x8000_0000_0000_0000;",
        "print(1e3, 1.0e+6, 0xFFFF_FFFF_FFFF_FFFF);",
    ];

    assert!(
        corpus.len() >= 40,
        "Corpus must contain at least 40 expressions, got {}",
        corpus.len()
    );

    for &source in corpus {
        // 1. Rust lexer tokenization
        let mut interner = Interner::new();
        let tokens = Lexer::new(source, &mut interner)
            .tokenize()
            .unwrap_or_else(|e| panic!("Rust lexer failed for '{}': {:?}", source, e));

        let rust_numbers: Vec<(usize, usize, &str)> = tokens
            .iter()
            .filter(|t| t.kind == TokenKind::Integer || t.kind == TokenKind::Float)
            .map(|t| {
                let start = line_col_to_byte_offset(source, t.line, t.column);
                let end = start + t.lexeme.len();
                assert_eq!(
                    &source[start..end],
                    t.lexeme.as_str(),
                    "Lexer token lexeme must match source slice for '{}'",
                    source
                );
                (start, end, &source[start..end])
            })
            .collect();

        // 2. Tree-sitter parsing
        let tree = parser
            .parse(source, None)
            .unwrap_or_else(|| panic!("Tree-sitter failed to parse '{}'", source));
        let root = tree.root_node();
        let mut ts_numbers = Vec::new();
        collect_ts_number_nodes(root, source, &mut ts_numbers);

        // Sort both by start_byte just in case
        assert_eq!(
            rust_numbers.len(),
            ts_numbers.len(),
            "Number node count mismatch for '{}'. Rust: {:?}, TS: {:?}",
            source,
            rust_numbers,
            ts_numbers
        );

        for (i, (r_num, ts_num)) in rust_numbers.iter().zip(ts_numbers.iter()).enumerate() {
            assert_eq!(
                r_num.0, ts_num.0,
                "Start byte mismatch at index {} for '{}': Rust {}, TS {}",
                i, source, r_num.0, ts_num.0
            );
            assert_eq!(
                r_num.1, ts_num.1,
                "End byte mismatch at index {} for '{}': Rust {}, TS {}",
                i, source, r_num.1, ts_num.1
            );
            assert_eq!(
                r_num.2, ts_num.2,
                "Slice text mismatch at index {} for '{}': Rust '{}', TS '{}'",
                i, source, r_num.2, ts_num.2
            );
        }
    }
}

#[test]
fn test_treesitter_range_disambiguation() {
    let mut parser = make_ts_parser();

    let range_cases = ["0..10", "0..=10", "1_0..2_0"];

    for &source in &range_cases {
        // Rust lexer
        let mut interner = Interner::new();
        let tokens = Lexer::new(source, &mut interner)
            .tokenize()
            .unwrap_or_else(|e| panic!("Rust lexer failed on range '{}': {:?}", source, e));

        let num_tokens: Vec<&Token> = tokens
            .iter()
            .filter(|t| t.kind == TokenKind::Integer || t.kind == TokenKind::Float)
            .collect();
        assert_eq!(
            num_tokens.len(),
            2,
            "Rust lexer must find exactly 2 number tokens in '{}'",
            source
        );

        // Tree-sitter
        let tree = parser
            .parse(source, None)
            .unwrap_or_else(|| panic!("Tree-sitter failed to parse range '{}'", source));
        let root = tree.root_node();
        let mut ts_numbers = Vec::new();
        collect_ts_number_nodes(root, source, &mut ts_numbers);

        assert_eq!(
            ts_numbers.len(),
            2,
            "Tree-sitter must produce strictly 2 'number' nodes for '{}'",
            source
        );

        // Verify the two numbers match
        for (r_tok, ts_num) in num_tokens.iter().zip(ts_numbers.iter()) {
            let start = line_col_to_byte_offset(source, r_tok.line, r_tok.column);
            let end = start + r_tok.lexeme.len();
            assert_eq!(start, ts_num.0, "Start byte mismatch for range '{}'", source);
            assert_eq!(end, ts_num.1, "End byte mismatch for range '{}'", source);
            assert_eq!(r_tok.lexeme.as_str(), ts_num.2, "Lexeme mismatch for range '{}'", source);
        }

        // Verify separation: there must be characters between the two numbers (the '..' or '..=')
        let between_slice = &source[ts_numbers[0].1..ts_numbers[1].0];
        assert!(
            between_slice.starts_with(".."),
            "Range expression '{}' must have '..' separating the two numbers, found '{}'",
            source,
            between_slice
        );
    }
}

#[test]
fn test_treesitter_invalid_literals_rejection() {
    let mut parser = make_ts_parser();

    // Invalid literals where Rust lexer returns LexerError
    let invalid_cases = [
        "0x1_G",
        "0b102",
        "1__0",
        "0x_1",
        "0b_0",
        "0o_7",
        "100_",
        "0xFF_",
        "1._0",
        "1_.0",
        "1e_3",
        "1e+_3",
        "0x",
        "0b",
        "0o",
    ];

    for &source in &invalid_cases {
        // 1. Rust lexer must return a LexerError
        let mut interner = Interner::new();
        let lex_result = Lexer::new(source, &mut interner).tokenize();
        assert!(
            matches!(lex_result, Err(LustError::LexerError { .. })),
            "Rust lexer must return LexerError for invalid literal '{}', got {:?}",
            source,
            lex_result
        );

        // 2. Tree-sitter must either:
        //    a) contain an ERROR node, OR
        //    b) NOT parse the entire invalid string as a single 'number' node
        let tree = parser
            .parse(source, None)
            .unwrap_or_else(|| panic!("Tree-sitter failed to run on '{}'", source));
        let root = tree.root_node();
        let has_err = tree_has_error_node(root);

        let mut ts_numbers = Vec::new();
        collect_ts_number_nodes(root, source, &mut ts_numbers);

        let full_as_number = ts_numbers.len() == 1
            && ts_numbers[0].0 == 0
            && ts_numbers[0].1 == source.trim().len();

        assert!(
            has_err || !full_as_number,
            "Tree-sitter must not accept invalid literal '{}' as a valid single number without error (has_err={}, full_as_number={})",
            source,
            has_err,
            full_as_number
        );
    }
}

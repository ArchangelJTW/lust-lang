#![cfg(feature = "std")]
//! Stage 3 - L9: Coverage and correctness tests for AST, Lexer Tokens, and Typechecker Expression Operations & Patterns.

use lust::ast::{
    expr::{BinaryOp, Expr, ExprKind, Literal, Pattern, StructLiteralField, UnaryOp},
    items::{EnumDef, EnumVariant, Visibility},
    Span, Type, TypeKind,
};
use lust::intern::Interner;
use lust::lexer::{Lexer, Token, TokenKind};
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use std::collections::HashSet;

fn dummy_span() -> Span {
    Span::dummy()
}

fn check_source(source: &str) -> lust::error::Result<()> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize()?;
    let items = Parser::new(tokens).parse()?;
    TypeChecker::new().check_module(&items)
}

// ---------------------------------------------------------------------------
// 1. Lexer Tokens: TokenKind display, keywords, and allocation flags
// ---------------------------------------------------------------------------

#[test]
fn test_l9_token_construction_and_methods() {
    let t_alloc = Token::new(TokenKind::Identifier, "my_var".to_string(), 1, 5);
    assert!(t_alloc.has_allocated_lexeme());
    assert_eq!(t_alloc.lexeme, "my_var");
    assert_eq!(t_alloc.line, 1);
    assert_eq!(t_alloc.column, 5);

    let t_simple = Token::simple(TokenKind::Plus, 2, 10);
    assert!(!t_simple.has_allocated_lexeme());
    assert!(t_simple.lexeme.is_empty());
    assert_eq!(t_simple.line, 2);
    assert_eq!(t_simple.column, 10);
}

#[test]
fn test_l9_token_keyword_table() {
    let valid_keywords = [
        ("local", TokenKind::Local),
        ("function", TokenKind::Function),
        ("return", TokenKind::Return),
        ("if", TokenKind::If),
        ("then", TokenKind::Then),
        ("else", TokenKind::Else),
        ("elseif", TokenKind::Elseif),
        ("end", TokenKind::End),
        ("while", TokenKind::While),
        ("do", TokenKind::Do),
        ("for", TokenKind::For),
        ("in", TokenKind::In),
        ("break", TokenKind::Break),
        ("continue", TokenKind::Continue),
        ("struct", TokenKind::Struct),
        ("enum", TokenKind::Enum),
        ("trait", TokenKind::Trait),
        ("impl", TokenKind::Impl),
        ("as", TokenKind::As),
        ("is", TokenKind::Is),
        ("true", TokenKind::True),
        ("false", TokenKind::False),
        ("and", TokenKind::And),
        ("or", TokenKind::Or),
        ("not", TokenKind::Not),
        ("extern", TokenKind::Extern),
        ("use", TokenKind::Use),
        ("module", TokenKind::Module),
        ("defer", TokenKind::Defer),
        ("errdefer", TokenKind::ErrDefer),
        ("scoped", TokenKind::Scoped),
    ];

    for (word, expected_kind) in valid_keywords {
        let kw = TokenKind::keyword(word);
        assert_eq!(kw, Some(expected_kind), "Keyword {word} mismatch");
    }

    assert_eq!(TokenKind::keyword("non_existent_keyword"), None);
    assert_eq!(TokenKind::keyword(""), None);
    assert_eq!(TokenKind::keyword("let"), None);
}

#[test]
fn test_l9_token_kind_display_all_variants() {
    let variants = [
        (TokenKind::Integer, "integer"),
        (TokenKind::Float, "float"),
        (TokenKind::String, "string"),
        (TokenKind::True, "true"),
        (TokenKind::False, "false"),
        (TokenKind::Identifier, "identifier"),
        (TokenKind::Local, "local"),
        (TokenKind::Function, "function"),
        (TokenKind::Return, "return"),
        (TokenKind::If, "if"),
        (TokenKind::Then, "then"),
        (TokenKind::Else, "else"),
        (TokenKind::Elseif, "elseif"),
        (TokenKind::End, "end"),
        (TokenKind::While, "while"),
        (TokenKind::Do, "do"),
        (TokenKind::For, "for"),
        (TokenKind::In, "in"),
        (TokenKind::Break, "break"),
        (TokenKind::Continue, "continue"),
        (TokenKind::Struct, "struct"),
        (TokenKind::Enum, "enum"),
        (TokenKind::Trait, "trait"),
        (TokenKind::Impl, "impl"),
        (TokenKind::As, "as"),
        (TokenKind::Is, "is"),
        (TokenKind::Extern, "extern"),
        (TokenKind::Use, "use"),
        (TokenKind::Module, "module"),
        (TokenKind::Defer, "defer"),
        (TokenKind::ErrDefer, "errdefer"),
        (TokenKind::Scoped, "scoped"),
        (TokenKind::EffectKw, "effect"),
        (TokenKind::Bang, "!"),
        (TokenKind::Plus, "+"),
        (TokenKind::Minus, "-"),
        (TokenKind::Star, "*"),
        (TokenKind::Slash, "/"),
        (TokenKind::Percent, "%"),
        (TokenKind::Caret, "^"),
        (TokenKind::DoubleDot, ".."),
        (TokenKind::Equal, "="),
        (TokenKind::DoubleEqual, "=="),
        (TokenKind::NotEqual, "~="),
        (TokenKind::Less, "<"),
        (TokenKind::LessEqual, "<="),
        (TokenKind::Greater, ">"),
        (TokenKind::GreaterEqual, ">="),
        (TokenKind::And, "and"),
        (TokenKind::Or, "or"),
        (TokenKind::Not, "not"),
        (TokenKind::PlusEqual, "+="),
        (TokenKind::MinusEqual, "-="),
        (TokenKind::StarEqual, "*="),
        (TokenKind::SlashEqual, "/="),
        (TokenKind::DoubleSlash, "//"),
        (TokenKind::DoubleSlashEqual, "//="),
        (TokenKind::PercentEqual, "%="),
        (TokenKind::CaretEqual, "^="),
        (TokenKind::AmpersandEqual, "&="),
        (TokenKind::PipeEqual, "|="),
        (TokenKind::DoubleCaret, "^^"),
        (TokenKind::DoubleCaretEqual, "^^="),
        (TokenKind::DoubleLess, "<<"),
        (TokenKind::DoubleLessEqual, "<<="),
        (TokenKind::DoubleGreater, ">>"),
        (TokenKind::DoubleGreaterEqual, ">>="),
        (TokenKind::Tilde, "~"),
        (TokenKind::Ampersand, "&"),
        (TokenKind::Pipe, "|"),
        (TokenKind::PipeGreater, "|>"),
        (TokenKind::Question, "?"),
        (TokenKind::LeftParen, "("),
        (TokenKind::RightParen, ")"),
        (TokenKind::LeftBrace, "{"),
        (TokenKind::RightBrace, "}"),
        (TokenKind::LeftBracket, "["),
        (TokenKind::RightBracket, "]"),
        (TokenKind::Comma, ","),
        (TokenKind::Colon, ":"),
        (TokenKind::DoubleColon, "::"),
        (TokenKind::Semicolon, ";"),
        (TokenKind::Dot, "."),
        (TokenKind::Arrow, "->"),
        (TokenKind::FatArrow, "=>"),
        (TokenKind::Comment, "comment"),
        (TokenKind::DocComment, "doc comment"),
        (TokenKind::Whitespace, "whitespace"),
        (TokenKind::Newline, "newline"),
        (TokenKind::Eof, "EOF"),
    ];

    for (kind, expected_str) in variants {
        assert_eq!(
            kind.to_string(),
            expected_str,
            "Display for {:?} did not match",
            kind
        );
    }
}

// ---------------------------------------------------------------------------
// 2. AST: Expr, ExprKind, Operators, Pattern Display & Visitors
// ---------------------------------------------------------------------------

#[test]
fn test_l9_ast_operators_display() {
    let bin_ops = [
        (BinaryOp::Add, "+"),
        (BinaryOp::Sub, "-"),
        (BinaryOp::Mul, "*"),
        (BinaryOp::Div, "/"),
        (BinaryOp::Mod, "%"),
        (BinaryOp::FloorDiv, "//"),
        (BinaryOp::Pow, "^"),
        (BinaryOp::Eq, "=="),
        (BinaryOp::Ne, "!="),
        (BinaryOp::Lt, "<"),
        (BinaryOp::Le, "<="),
        (BinaryOp::Gt, ">"),
        (BinaryOp::Ge, ">="),
        (BinaryOp::And, "and"),
        (BinaryOp::Or, "or"),
        (BinaryOp::BitAnd, "&"),
        (BinaryOp::BitOr, "|"),
        (BinaryOp::BitXor, "^^"),
        (BinaryOp::Shl, "<<"),
        (BinaryOp::Shr, ">>"),
        (BinaryOp::Concat, ".."),
        (BinaryOp::Range, ".."),
    ];
    for (op, expected) in bin_ops {
        assert_eq!(op.to_string(), expected, "BinaryOp {op:?} Display mismatch");
    }

    let un_ops = [
        (UnaryOp::Neg, "-"),
        (UnaryOp::Not, "not"),
        (UnaryOp::BitNot, "~"),
    ];
    for (op, expected) in un_ops {
        assert_eq!(op.to_string(), expected, "UnaryOp {op:?} Display mismatch");
    }
}

#[test]
fn test_l9_ast_literal_and_pattern_display() {
    assert_eq!(Literal::Integer(12345).to_string(), "12345");
    assert_eq!(Literal::Float(3.1415).to_string(), "3.1415");
    assert_eq!(
        Literal::String("hello \"world\"\n".to_string()).to_string(),
        "\"hello \\\"world\\\"\\n\""
    );
    assert_eq!(Literal::Bool(true).to_string(), "true");
    assert_eq!(Literal::Bool(false).to_string(), "false");

    assert_eq!(Pattern::Wildcard.to_string(), "_");
    assert_eq!(Pattern::Literal(Literal::Integer(42)).to_string(), "42");
    assert_eq!(Pattern::Identifier("my_id".to_string()).to_string(), "my_id");

    // Enum pattern with empty enum_name (unqualified) and empty bindings
    let pat_enum_unqualified = Pattern::Enum {
        enum_name: String::new(),
        variant: "None".to_string(),
        bindings: vec![],
    };
    assert_eq!(pat_enum_unqualified.to_string(), "None");

    // Enum pattern with qualified name and bindings
    let pat_enum_qualified = Pattern::Enum {
        enum_name: "Option".to_string(),
        variant: "Some".to_string(),
        bindings: vec![Pattern::Identifier("val".to_string())],
    };
    assert_eq!(pat_enum_qualified.to_string(), "Option.Some(val)");

    // Struct pattern with field shorthand and explicit pattern
    let pat_struct = Pattern::Struct {
        name: "Point".to_string(),
        fields: vec![
            ("x".to_string(), Pattern::Identifier("x".to_string())),
            ("y".to_string(), Pattern::Literal(Literal::Integer(0))),
        ],
    };
    assert_eq!(pat_struct.to_string(), "Point { x, y = 0 }");

    // TypeCheck pattern
    let ty = Type::new(TypeKind::Int, dummy_span());
    let pat_type = Pattern::TypeCheck(ty);
    assert_eq!(pat_type.to_string(), "as int");
}

#[test]
fn test_l9_expr_return_spec() {
    let span = dummy_span();

    // 1. Non-lambda expr returns None
    let non_lambda = Expr::new(ExprKind::Literal(Literal::Integer(10)), span);
    assert_eq!(non_lambda.return_spec(), None);

    // 2. Lambda without return type
    let lambda_no_ret = Expr::new(
        ExprKind::Lambda {
            params: vec![],
            return_type: None,
            body: Box::new(Expr::new(ExprKind::Literal(Literal::Integer(0)), span)),
        },
        span,
    );
    let spec = lambda_no_ret.return_spec().expect("Should return ReturnSpec");
    assert_eq!(spec.return_type, None);
    assert_eq!(spec.error_type, None);
    assert!(spec.requirements.is_empty());

    // 3. Lambda with standard return type
    let int_ty = Type::new(TypeKind::Int, span);
    let lambda_std = Expr::new(
        ExprKind::Lambda {
            params: vec![],
            return_type: Some(int_ty.clone()),
            body: Box::new(Expr::new(ExprKind::Literal(Literal::Integer(0)), span)),
        },
        span,
    );
    let spec_std = lambda_std.return_spec().expect("Should return ReturnSpec");
    assert_eq!(spec_std.return_type, Some(int_ty));
    assert_eq!(spec_std.error_type, None);
    assert!(spec_std.requirements.is_empty());

    // 4. Lambda with Effect return type and Never error
    let eff_never = Type::new(
        TypeKind::Effect {
            success: Box::new(Type::new(TypeKind::String, span)),
            error: Box::new(Type::new(TypeKind::Never, span)),
            requirements: vec![],
        },
        span,
    );
    let lambda_eff_never = Expr::new(
        ExprKind::Lambda {
            params: vec![],
            return_type: Some(eff_never),
            body: Box::new(Expr::new(ExprKind::Literal(Literal::Integer(0)), span)),
        },
        span,
    );
    let spec_never = lambda_eff_never.return_spec().unwrap();
    assert_eq!(
        spec_never.return_type,
        Some(Type::new(TypeKind::String, span))
    );
    assert_eq!(spec_never.error_type, None); // Never converts to None

    // 5. Lambda with Effect return type, concrete error, and requirements
    let req_service = Type::new(TypeKind::Named("Database".to_string()), span);
    let eff_error = Type::new(
        TypeKind::Effect {
            success: Box::new(Type::new(TypeKind::Int, span)),
            error: Box::new(Type::new(TypeKind::Named("IoError".to_string()), span)),
            requirements: vec![req_service.clone()],
        },
        span,
    );
    let lambda_eff_error = Expr::new(
        ExprKind::Lambda {
            params: vec![],
            return_type: Some(eff_error),
            body: Box::new(Expr::new(ExprKind::Literal(Literal::Integer(0)), span)),
        },
        span,
    );
    let spec_err = lambda_eff_error.return_spec().unwrap();
    assert_eq!(spec_err.return_type, Some(Type::new(TypeKind::Int, span)));
    assert_eq!(
        spec_err.error_type,
        Some(Type::new(TypeKind::Named("IoError".to_string()), span))
    );
    assert_eq!(spec_err.requirements, vec![req_service]);
}

// ---------------------------------------------------------------------------
// 3. Full AST Node Visitor / Walker covering every single ExprKind variant
// ---------------------------------------------------------------------------

struct AstCoverageVisitor {
    visited_kinds: HashSet<&'static str>,
}

impl AstCoverageVisitor {
    fn new() -> Self {
        Self {
            visited_kinds: HashSet::new(),
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        match &expr.kind {
            ExprKind::Literal(_) => {
                self.visited_kinds.insert("Literal");
            }
            ExprKind::Identifier(_) => {
                self.visited_kinds.insert("Identifier");
            }
            ExprKind::Binary { left, right, .. } => {
                self.visited_kinds.insert("Binary");
                self.visit_expr(left);
                self.visit_expr(right);
            }
            ExprKind::Unary { operand, .. } => {
                self.visited_kinds.insert("Unary");
                self.visit_expr(operand);
            }
            ExprKind::Call { callee, args, .. } => {
                self.visited_kinds.insert("Call");
                self.visit_expr(callee);
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            ExprKind::MethodCall { receiver, args, .. } => {
                self.visited_kinds.insert("MethodCall");
                self.visit_expr(receiver);
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            ExprKind::FieldAccess { object, .. } => {
                self.visited_kinds.insert("FieldAccess");
                self.visit_expr(object);
            }
            ExprKind::Index { object, index } => {
                self.visited_kinds.insert("Index");
                self.visit_expr(object);
                self.visit_expr(index);
            }
            ExprKind::Array(elems) => {
                self.visited_kinds.insert("Array");
                for elem in elems {
                    self.visit_expr(elem);
                }
            }
            ExprKind::Map(pairs) => {
                self.visited_kinds.insert("Map");
                for (k, v) in pairs {
                    self.visit_expr(k);
                    self.visit_expr(v);
                }
            }
            ExprKind::Tuple(items) => {
                self.visited_kinds.insert("Tuple");
                for it in items {
                    self.visit_expr(it);
                }
            }
            ExprKind::StructLiteral { fields, .. } => {
                self.visited_kinds.insert("StructLiteral");
                for f in fields {
                    self.visit_expr(&f.value);
                }
            }
            ExprKind::EnumConstructor { args, .. } => {
                self.visited_kinds.insert("EnumConstructor");
                for a in args {
                    self.visit_expr(a);
                }
            }
            ExprKind::Lambda { body, .. } => {
                self.visited_kinds.insert("Lambda");
                self.visit_expr(body);
            }
            ExprKind::Paren(inner) => {
                self.visited_kinds.insert("Paren");
                self.visit_expr(inner);
            }
            ExprKind::Cast { expr, .. } => {
                self.visited_kinds.insert("Cast");
                self.visit_expr(expr);
            }
            ExprKind::TypeCheck { expr, .. } => {
                self.visited_kinds.insert("TypeCheck");
                self.visit_expr(expr);
            }
            ExprKind::IsPattern { expr, .. } => {
                self.visited_kinds.insert("IsPattern");
                self.visit_expr(expr);
            }
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.visited_kinds.insert("If");
                self.visit_expr(condition);
                self.visit_expr(then_branch);
                if let Some(eb) = else_branch {
                    self.visit_expr(eb);
                }
            }
            ExprKind::Block(stmts) => {
                self.visited_kinds.insert("Block");
                for stmt in stmts {
                    if let lust::ast::stmt::StmtKind::Expr(e) = &stmt.kind {
                        self.visit_expr(e);
                    }
                }
            }
            ExprKind::EffectBlock(stmts) => {
                self.visited_kinds.insert("EffectBlock");
                for stmt in stmts {
                    if let lust::ast::stmt::StmtKind::Expr(e) = &stmt.kind {
                        self.visit_expr(e);
                    }
                }
            }
            ExprKind::Return(vals) => {
                self.visited_kinds.insert("Return");
                for v in vals {
                    self.visit_expr(v);
                }
            }
            ExprKind::Range {
                start,
                end,
                inclusive: _,
            } => {
                self.visited_kinds.insert("Range");
                self.visit_expr(start);
                self.visit_expr(end);
            }
        }
    }
}

#[test]
fn test_l9_ast_visitor_covers_every_expr_variant() {
    let span = dummy_span();
    let lit_int = || Expr::new(ExprKind::Literal(Literal::Integer(1)), span);
    let lit_str = || {
        Expr::new(
            ExprKind::Literal(Literal::String("hello".to_string())),
            span,
        )
    };

    let all_nodes = vec![
        lit_int(),
        Expr::new(ExprKind::Identifier("x".to_string()), span),
        Expr::new(
            ExprKind::Binary {
                left: Box::new(lit_int()),
                op: BinaryOp::Add,
                right: Box::new(lit_int()),
            },
            span,
        ),
        Expr::new(
            ExprKind::Unary {
                op: UnaryOp::Neg,
                operand: Box::new(lit_int()),
            },
            span,
        ),
        Expr::new(
            ExprKind::Call {
                callee: Box::new(Expr::new(ExprKind::Identifier("foo".to_string()), span)),
                type_args: None,
                args: vec![lit_int()],
            },
            span,
        ),
        Expr::new(
            ExprKind::MethodCall {
                receiver: Box::new(lit_int()),
                method: "bar".to_string(),
                type_args: None,
                args: vec![],
            },
            span,
        ),
        Expr::new(
            ExprKind::FieldAccess {
                object: Box::new(lit_int()),
                field: "baz".to_string(),
            },
            span,
        ),
        Expr::new(
            ExprKind::Index {
                object: Box::new(lit_int()),
                index: Box::new(lit_int()),
            },
            span,
        ),
        Expr::new(ExprKind::Array(vec![lit_int()]), span),
        Expr::new(ExprKind::Map(vec![(lit_str(), lit_int())]), span),
        Expr::new(ExprKind::Tuple(vec![lit_int(), lit_str()]), span),
        Expr::new(
            ExprKind::StructLiteral {
                name: "Point".to_string(),
                fields: vec![StructLiteralField {
                    name: "x".to_string(),
                    value: lit_int(),
                    span,
                }],
            },
            span,
        ),
        Expr::new(
            ExprKind::EnumConstructor {
                enum_name: "Color".to_string(),
                variant: "Red".to_string(),
                args: vec![],
            },
            span,
        ),
        Expr::new(
            ExprKind::Lambda {
                params: vec![("a".to_string(), Some(Type::new(TypeKind::Int, span)))],
                return_type: Some(Type::new(TypeKind::Int, span)),
                body: Box::new(lit_int()),
            },
            span,
        ),
        Expr::new(ExprKind::Paren(Box::new(lit_int())), span),
        Expr::new(
            ExprKind::Cast {
                expr: Box::new(lit_int()),
                target_type: Type::new(TypeKind::Float, span),
            },
            span,
        ),
        Expr::new(
            ExprKind::TypeCheck {
                expr: Box::new(lit_int()),
                check_type: Type::new(TypeKind::Int, span),
            },
            span,
        ),
        Expr::new(
            ExprKind::IsPattern {
                expr: Box::new(lit_int()),
                pattern: Pattern::Wildcard,
            },
            span,
        ),
        Expr::new(
            ExprKind::If {
                condition: Box::new(Expr::new(
                    ExprKind::Literal(Literal::Bool(true)),
                    span,
                )),
                then_branch: Box::new(lit_int()),
                else_branch: Some(Box::new(lit_int())),
            },
            span,
        ),
        Expr::new(
            ExprKind::Block(vec![lust::ast::stmt::Stmt {
                kind: lust::ast::stmt::StmtKind::Expr(lit_int()),
                span,
            }]),
            span,
        ),
        Expr::new(
            ExprKind::EffectBlock(vec![lust::ast::stmt::Stmt {
                kind: lust::ast::stmt::StmtKind::Expr(lit_int()),
                span,
            }]),
            span,
        ),
        Expr::new(ExprKind::Return(vec![lit_int()]), span),
        Expr::new(
            ExprKind::Range {
                start: Box::new(lit_int()),
                end: Box::new(lit_int()),
                inclusive: true,
            },
            span,
        ),
    ];

    let mut visitor = AstCoverageVisitor::new();
    for node in &all_nodes {
        visitor.visit_expr(node);
    }

    let expected_variants = [
        "Literal",
        "Identifier",
        "Binary",
        "Unary",
        "Call",
        "MethodCall",
        "FieldAccess",
        "Index",
        "Array",
        "Map",
        "Tuple",
        "StructLiteral",
        "EnumConstructor",
        "Lambda",
        "Paren",
        "Cast",
        "TypeCheck",
        "IsPattern",
        "If",
        "Block",
        "EffectBlock",
        "Return",
        "Range",
    ];

    for var in expected_variants {
        assert!(
            visitor.visited_kinds.contains(var),
            "Variant {var} was not visited by AstCoverageVisitor"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. TypeChecker: Direct Binary & Unary Expression Operations
// ---------------------------------------------------------------------------

#[test]
fn test_l9_typecheck_binary_arithmetic_oracles() {
    let mut checker = TypeChecker::new();
    let span = dummy_span();

    let e_int = Expr::new(ExprKind::Literal(Literal::Integer(10)), span);
    let e_float = Expr::new(ExprKind::Literal(Literal::Float(2.5)), span);
    let e_str = Expr::new(
        ExprKind::Literal(Literal::String("abc".to_string())),
        span,
    );
    let e_bool = Expr::new(ExprKind::Literal(Literal::Bool(true)), span);

    // int + int -> int
    let ty = checker
        .check_binary_expr(span, &e_int, &BinaryOp::Add, &e_int)
        .unwrap();
    assert_eq!(ty.kind, TypeKind::Int);

    // int + float -> float
    let ty = checker
        .check_binary_expr(span, &e_int, &BinaryOp::Add, &e_float)
        .unwrap();
    assert_eq!(ty.kind, TypeKind::Float);

    // float * float -> float
    let ty = checker
        .check_binary_expr(span, &e_float, &BinaryOp::Mul, &e_float)
        .unwrap();
    assert_eq!(ty.kind, TypeKind::Float);

    // int // int -> int
    let ty = checker
        .check_binary_expr(span, &e_int, &BinaryOp::FloorDiv, &e_int)
        .unwrap();
    assert_eq!(ty.kind, TypeKind::Int);

    // int ^ float -> float
    let ty = checker
        .check_binary_expr(span, &e_int, &BinaryOp::Pow, &e_float)
        .unwrap();
    assert_eq!(ty.kind, TypeKind::Float);

    // Exact Error Oracle: int + string fails
    let err = checker
        .check_binary_expr(span, &e_int, &BinaryOp::Add, &e_str)
        .unwrap_err();
    assert!(
        err.to_string().contains("requires numeric types"),
        "Expected numeric types error, got: {err}"
    );

    // Exact Error Oracle: bool * float fails
    let err = checker
        .check_binary_expr(span, &e_bool, &BinaryOp::Mul, &e_float)
        .unwrap_err();
    assert!(
        err.to_string().contains("requires numeric types"),
        "Expected numeric types error, got: {err}"
    );
}

#[test]
fn test_l9_typecheck_binary_bitwise_oracles() {
    let mut checker = TypeChecker::new();
    let span = dummy_span();

    let e_int1 = Expr::new(ExprKind::Literal(Literal::Integer(7)), span);
    let e_int2 = Expr::new(ExprKind::Literal(Literal::Integer(3)), span);
    let e_float = Expr::new(ExprKind::Literal(Literal::Float(1.5)), span);

    for op in [
        BinaryOp::BitAnd,
        BinaryOp::BitOr,
        BinaryOp::BitXor,
        BinaryOp::Shl,
        BinaryOp::Shr,
    ] {
        let ty = checker
            .check_binary_expr(span, &e_int1, &op, &e_int2)
            .unwrap();
        assert_eq!(ty.kind, TypeKind::Int);

        // Bitwise with float must fail with exact error message
        let err = checker
            .check_binary_expr(span, &e_int1, &op, &e_float)
            .unwrap_err();
        assert!(
            err.to_string().contains("requires integer types"),
            "Op {op:?} error mismatch: {err}"
        );
    }
}

#[test]
fn test_l9_typecheck_binary_comparison_oracles() {
    let mut checker = TypeChecker::new();
    let span = dummy_span();

    let e_int = Expr::new(ExprKind::Literal(Literal::Integer(1)), span);
    let e_str = Expr::new(
        ExprKind::Literal(Literal::String("a".to_string())),
        span,
    );

    for op in [
        BinaryOp::Eq,
        BinaryOp::Ne,
        BinaryOp::Lt,
        BinaryOp::Le,
        BinaryOp::Gt,
        BinaryOp::Ge,
    ] {
        let ty = checker
            .check_binary_expr(span, &e_int, &op, &e_int)
            .unwrap();
        assert_eq!(ty.kind, TypeKind::Bool);

        // Incompatible comparison int and string
        let err = checker
            .check_binary_expr(span, &e_int, &op, &e_str)
            .unwrap_err();
        assert!(
            err.to_string().contains("requires compatible types"),
            "Op {op:?} mismatch: {err}"
        );
    }
}

#[test]
fn test_l9_typecheck_binary_range_rejection() {
    let mut checker = TypeChecker::new();
    let span = dummy_span();
    let e_int = Expr::new(ExprKind::Literal(Literal::Integer(1)), span);

    let err = checker
        .check_binary_expr(span, &e_int, &BinaryOp::Range, &e_int)
        .unwrap_err();
    assert!(
        err.to_string().contains("Range operator is not supported"),
        "Range rejection mismatch: {err}"
    );
}

#[test]
fn test_l9_typecheck_unary_oracles() {
    let mut checker = TypeChecker::new();
    let span = dummy_span();

    let e_int = Expr::new(ExprKind::Literal(Literal::Integer(5)), span);
    let e_float = Expr::new(ExprKind::Literal(Literal::Float(2.5)), span);
    let e_bool = Expr::new(ExprKind::Literal(Literal::Bool(true)), span);
    let e_str = Expr::new(
        ExprKind::Literal(Literal::String("x".to_string())),
        span,
    );

    // Unary Negation
    assert_eq!(
        checker
            .check_unary_expr(&UnaryOp::Neg, &e_int)
            .unwrap()
            .kind,
        TypeKind::Int
    );
    assert_eq!(
        checker
            .check_unary_expr(&UnaryOp::Neg, &e_float)
            .unwrap()
            .kind,
        TypeKind::Float
    );
    let err = checker.check_unary_expr(&UnaryOp::Neg, &e_str).unwrap_err();
    assert!(err.to_string().contains("requires numeric type"));

    // Unary Not
    assert_eq!(
        checker
            .check_unary_expr(&UnaryOp::Not, &e_bool)
            .unwrap()
            .kind,
        TypeKind::Bool
    );

    // Unary BitNot
    assert_eq!(
        checker
            .check_unary_expr(&UnaryOp::BitNot, &e_int)
            .unwrap()
            .kind,
        TypeKind::Int
    );
    let err = checker
        .check_unary_expr(&UnaryOp::BitNot, &e_float)
        .unwrap_err();
    assert!(err.to_string().contains("requires integer type"));
}

// ---------------------------------------------------------------------------
// 5. TypeChecker: Pattern Validation (patterns.rs)
// ---------------------------------------------------------------------------

#[test]
fn test_l9_pattern_validation_oracles() {
    let mut checker = TypeChecker::new();
    let span = dummy_span();

    // Register a test enum
    let color_enum = EnumDef {
        name: "Color".to_string(),
        visibility: Visibility::Public,
        type_params: vec![],
        trait_bounds: vec![],
        variants: vec![
            EnumVariant {
                name: "Red".to_string(),
                fields: None,
            },
            EnumVariant {
                name: "Rgb".to_string(),
                fields: Some(vec![
                    Type::new(TypeKind::Int, span),
                    Type::new(TypeKind::Int, span),
                    Type::new(TypeKind::Int, span),
                ]),
            },
        ],
        doc: None,
    };
    checker.register_external_enum(color_enum).unwrap();

    let color_type = Type::new(TypeKind::Named("Color".to_string()), span);
    let int_type = Type::new(TypeKind::Int, span);

    // 1. Valid Wildcard, Literal, Identifier
    checker
        .validate_is_pattern(&Pattern::Wildcard, &color_type)
        .unwrap();
    checker
        .validate_is_pattern(&Pattern::Literal(Literal::Integer(1)), &int_type)
        .unwrap();
    checker
        .validate_is_pattern(&Pattern::Identifier("c".to_string()), &color_type)
        .unwrap();

    // 2. Valid Enum Variant (zero bindings)
    let pat_red = Pattern::Enum {
        enum_name: "Color".to_string(),
        variant: "Red".to_string(),
        bindings: vec![],
    };
    checker
        .validate_is_pattern(&pat_red, &color_type)
        .expect("Valid Red variant should pass");

    // 3. Valid Enum Variant with correct bindings
    let pat_rgb = Pattern::Enum {
        enum_name: "Color".to_string(),
        variant: "Rgb".to_string(),
        bindings: vec![
            Pattern::Identifier("r".to_string()),
            Pattern::Identifier("g".to_string()),
            Pattern::Identifier("b".to_string()),
        ],
    };
    checker
        .validate_is_pattern(&pat_rgb, &color_type)
        .expect("Valid Rgb variant should pass");

    // 4. Undefined enum error
    let pat_undef = Pattern::Enum {
        enum_name: "NonExistent".to_string(),
        variant: "Foo".to_string(),
        bindings: vec![],
    };
    let undef_type = Type::new(TypeKind::Named("NonExistent".to_string()), span);
    let err = checker
        .validate_is_pattern(&pat_undef, &undef_type)
        .unwrap_err();
    assert!(err.to_string().contains("Undefined enum 'NonExistent'"));

    // 5. Pattern on non-enum type
    let err = checker
        .validate_is_pattern(&pat_red, &int_type)
        .unwrap_err();
    assert!(err.to_string().contains("Cannot use enum pattern on non-enum type 'int'"));

    // 6. Unknown variant on enum
    let pat_unknown_var = Pattern::Enum {
        enum_name: "Color".to_string(),
        variant: "Blue".to_string(),
        bindings: vec![],
    };
    let err = checker
        .validate_is_pattern(&pat_unknown_var, &color_type)
        .unwrap_err();
    assert!(err.to_string().contains("Enum 'Color' has no variant 'Blue'"));

    // 7. Mismatched binding count
    let pat_mismatched_bindings = Pattern::Enum {
        enum_name: "Color".to_string(),
        variant: "Rgb".to_string(),
        bindings: vec![Pattern::Identifier("r".to_string())], // 1 instead of 3
    };
    let err = checker
        .validate_is_pattern(&pat_mismatched_bindings, &color_type)
        .unwrap_err();
    assert!(err.to_string().contains("expects 3 bindings, got 1"));
}

// ---------------------------------------------------------------------------
// 6. TypeChecker: Collections (collections.rs - Array, Map, Indexing)
// ---------------------------------------------------------------------------

#[test]
fn test_l9_typecheck_collections_and_indexing_oracles() {
    let mut checker = TypeChecker::new();
    let span = dummy_span();

    let e_int = Expr::new(ExprKind::Literal(Literal::Integer(1)), span);
    let e_str = Expr::new(
        ExprKind::Literal(Literal::String("key".to_string())),
        span,
    );
    let e_bool = Expr::new(ExprKind::Literal(Literal::Bool(true)), span);

    // 1. Array literals
    // Empty array without hint -> Array<unknown>
    let ty_empty = checker.check_array_literal(&[], None).unwrap();
    match ty_empty.kind {
        TypeKind::Array(inner) => assert_eq!(inner.kind, TypeKind::Unknown),
        other => panic!("Expected Array<unknown>, got {other:?}"),
    }

    // Empty array with hint -> Array<int>
    let expected_arr_int = Type::new(
        TypeKind::Array(Box::new(Type::new(TypeKind::Int, span))),
        span,
    );
    let ty_empty_hint = checker
        .check_array_literal(&[], Some(&expected_arr_int))
        .unwrap();
    assert_eq!(ty_empty_hint, expected_arr_int);

    // Non-empty array homogeneous
    let ty_arr = checker
        .check_array_literal(&[e_int.clone(), e_int.clone()], None)
        .unwrap();
    assert_eq!(
        ty_arr,
        Type::new(
            TypeKind::Array(Box::new(Type::new(TypeKind::Int, span))),
            span
        )
    );

    // Array with union hint mismatch error
    let union_elem = Type::new(
        TypeKind::Union(vec![
            Type::new(TypeKind::Int, span),
            Type::new(TypeKind::String, span),
        ]),
        span,
    );
    let union_arr = Type::new(TypeKind::Array(Box::new(union_elem)), span);
    let err = checker
        .check_array_literal(&[e_int.clone(), e_bool.clone()], Some(&union_arr))
        .unwrap_err();
    assert!(
        err.to_string().contains("does not match any type in union"),
        "Expected union mismatch error, got: {err}"
    );

    // 2. Map literals
    let ty_empty_map = checker.check_map_literal(&[], None).unwrap();
    match ty_empty_map.kind {
        TypeKind::Map(k, v) => {
            assert_eq!(k.kind, TypeKind::Unknown);
            assert_eq!(v.kind, TypeKind::Unknown);
        }
        other => panic!("Expected Map<unknown, unknown>, got {other:?}"),
    }

    let ty_map = checker
        .check_map_literal(&[(e_str.clone(), e_int.clone())], None)
        .unwrap();
    match ty_map.kind {
        TypeKind::Map(k, v) => {
            assert_eq!(k.kind, TypeKind::String);
            assert_eq!(v.kind, TypeKind::Int);
        }
        other => panic!("Expected Map<string, int>, got {other:?}"),
    }

    // 3. Indexing oracles
    // Array indexing with integer succeeds and yields Result<elem, IndexError>
    let e_arr = Expr::new(ExprKind::Array(vec![e_int.clone()]), span);
    let idx_ty = checker.check_index_expr(span, &e_arr, &e_int).unwrap();
    match idx_ty.kind {
        TypeKind::Result(ok_ty, err_ty) => {
            assert_eq!(ok_ty.kind, TypeKind::Int);
            assert_eq!(err_ty.kind, TypeKind::Named("IndexError".to_string()));
        }
        other => panic!("Expected Result<int, IndexError>, got {other:?}"),
    }

    // Array indexing with string fails
    let err = checker.check_index_expr(span, &e_arr, &e_str).unwrap_err();
    assert!(
        err.to_string().contains("expected 'int', got 'string'")
            || err.to_string().contains("Type mismatch"),
        "Unexpected error: {err}"
    );

    // Map indexing with string succeeds
    let e_map = Expr::new(ExprKind::Map(vec![(e_str.clone(), e_int.clone())]), span);
    let map_val_ty = checker.check_index_expr(span, &e_map, &e_str).unwrap();
    assert_eq!(map_val_ty.kind, TypeKind::Int);

    // Map indexing with wrong key type fails
    let err = checker.check_index_expr(span, &e_map, &e_int).unwrap_err();
    assert!(
        err.to_string().contains("Type mismatch")
            || err.to_string().contains("expected 'string', got 'int'"),
        "Unexpected error: {err}"
    );

    // Non-indexable type indexing fails
    let err = checker.check_index_expr(span, &e_int, &e_int).unwrap_err();
    assert!(err.to_string().contains("Cannot index type 'int'"));
}

// ---------------------------------------------------------------------------
// 7. Full Module Parsing + Typecheck Integration Oracles
// ---------------------------------------------------------------------------

#[test]
fn test_l9_module_typechecker_integration_oracles() {
    // Valid module with typed functions, structs, enums
    let valid_source = r#"
        struct Point
            x: int,
            y: int
        end

        function add_points(p1: Point, p2: Point): Point
            return Point { x = p1.x + p2.x, y = p1.y + p2.y }
        end

        function main(): int
            local p = add_points(Point { x = 1, y = 2 }, Point { x = 3, y = 4 })
            return p.x + p.y
        end
    "#;
    check_source(valid_source).expect("Valid Lust code should typecheck cleanly");

    // Invalid module: mismatched return type
    let bad_ret = r#"
        function fail(): int
            return "not an integer"
        end
    "#;
    let err = check_source(bad_ret).unwrap_err();
    assert!(
        err.to_string().contains("type") || err.to_string().contains("mismatch"),
        "Expected type mismatch error, got: {err}"
    );

    // Invalid module: field not found on struct
    let bad_field = r#"
        struct S
            a: int
        end
        function test(s: S): int
            return s.non_existent_field
        end
    "#;
    let err = check_source(bad_field).unwrap_err();
    assert!(
        err.to_string().contains("has no field 'non_existent_field'"),
        "Expected field error, got: {err}"
    );
}

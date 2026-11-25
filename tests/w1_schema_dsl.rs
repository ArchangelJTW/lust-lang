#![cfg(feature = "std")]

use lust::{
    EmbeddedProgram, Value,
    ast::ItemKind,
    intern::Interner,
    lexer::Lexer,
    parser::Parser,
    typechecker::TypeChecker,
    mir::{AstLowering, MirConstant, Operand},
};

fn parse_program(source: &str) -> Vec<lust::ast::Item> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("tokenization should succeed");
    Parser::new(tokens)
        .parse()
        .expect("parsing should succeed")
}

fn typecheck_program(source: &str) -> lust::Result<()> {
    let items = parse_program(source);
    let mut tc = TypeChecker::new();
    tc.check_module(&items)
}

fn run_program(source: &str, func: &str) -> lust::Result<Value> {
    let mut prog = EmbeddedProgram::builder()
        .module("main", source)
        .entry_module("main")
        .compile()?;
    prog.vm_mut().call(&format!("main.{}", func), vec![])
}

#[test]
fn test_contextual_schema_and_type_identifiers() {
    // Verifies that 'local schema = ...' and 'local type = ...' parse and typecheck as normal variables
    let src = r#"
        function test_identifiers(): int
            local schema = 123
            local type = 456
            return schema + type
        end
    "#;
    let res = run_program(src, "test_identifiers").expect("execution should succeed");
    assert_eq!(res, Value::Int(579));
}

#[test]
fn test_schema_dsl_parsing_and_instantiation() {
    let src = r#"
        schema User
            id: int @min(1)
            username: string @min(3) @max(32) @regex("^[a-z0-9_]+$")
            email: string @email
            nickname: string?
        end

        function make_user(): string
            local u = User {
                id = 42,
                username = "vedmalex",
                email = "alex@lust-lang.org",
                nickname = Option.None
            }
            return u.username
        end
    "#;
    let res = run_program(src, "make_user").expect("instantiation and field access should succeed");
    if let Value::String(s) = res {
        assert_eq!(&*s, "vedmalex");
    } else {
        panic!("expected string, got {:?}", res);
    }
}

#[test]
fn test_type_alias_declaration() {
    let src = r#"
        type UserId = int
        type Nickname = string?

        function get_id(id: UserId): int
            return id
        end
    "#;
    let items = parse_program(src);
    println!("DEBUG ITEMS: {:#?}", items);
    let res = typecheck_program(src);
    assert!(res.is_ok(), "type aliases should typecheck cleanly: {:?}", res);
}

#[test]
fn test_typeof_and_infer_type_operators() {
    let src = r#"
        schema User
            id: int
            username: string
        end

        type InferredUser = Infer<typeof User>
        type InferredOut = InferOutput<typeof User>
        type InferredIn = InferInput<typeof User>

        function check_inferred(): int
            local u: InferredUser = User { id = 1, username = "test" }
            return u.id
        end
    "#;
    let res = run_program(src, "check_inferred").expect("Infer<typeof User> should typecheck and run");
    assert_eq!(res, Value::Int(1));
}

#[test]
fn test_infer_type_mismatch_fails_compile_time() {
    let src = r#"
        schema User
            id: int
            username: string
        end

        function bad_assignment()
            local bad: Infer<typeof User> = "not a user"
        end
    "#;
    let res = typecheck_program(src);
    assert!(res.is_err(), "mismatched type assignment to Infer<typeof User> must fail typecheck");
}

#[test]
fn test_valid_regex_literal_compiles() {
    let src = r#"
        use "@std/regex" as regex

        function get_pattern(): Regex
            local r = regex!"^[a-zA-Z0-9_]+$"
            return r
        end
    "#;
    let res = run_program(src, "get_pattern").expect("valid regex! literal should compile and run");
    if let Value::String(s) = res {
        assert_eq!(&*s, "^[a-zA-Z0-9_]+$");
    } else {
        panic!("expected string/regex, got {:?}", res);
    }
}

#[test]
fn test_invalid_regex_literal_fails_compile_time() {
    let src = r#"
        use "@std/regex" as regex

        function bad_regex()
            local r = regex!"[unclosed"
        end
    "#;
    let res = typecheck_program(src);
    assert!(res.is_err(), "invalid regex! pattern must fail at compile time (typecheck)");
    let err = res.unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("Invalid regex pattern") || msg.contains("regex"),
        "error message should cite regex failure: {msg}"
    );
}

#[test]
fn test_invalid_schema_regex_constraint_fails_compile_time() {
    let src = r#"
        schema BadSchema
            name: string @regex("[unclosed")
        end
    "#;
    let res = typecheck_program(src);
    assert!(res.is_err(), "invalid regex in schema constraint must fail at compile time");
    let err = res.unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("Invalid regex pattern") || msg.contains("schema constraint"),
        "error message should cite schema regex constraint error: {msg}"
    );
}

#[test]
fn test_mir_lowering_regex_literal_not_unit() {
    let src = r#"
        use "@std/regex" as regex

        function test_mir(): Regex
            local p = regex!"\\d+"
            return p
        end
    "#;
    let items = parse_program(src);
    let mut tc = TypeChecker::new();
    tc.check_module(&items).expect("typechecking");

    // Lower to MIR function
    let func = items.into_iter().find_map(|item| match item.kind {
        ItemKind::Function(f) if f.name == "test_mir" => Some(f),
        _ => None,
    }).expect("test_mir function found");

    let mir = AstLowering::lower_function(&func).expect("mir lowering should succeed");

    // Scan all basic blocks for assignments from regex literal
    let mut found_string_const = false;
    let mut found_unit_fallback = false;

    for bb in &mir.blocks {
        for stmt in &bb.statements {
            if let lust::mir::Statement::Assign(_, rvalue, _) = stmt {
                if let lust::mir::Rvalue::Use(Operand::Constant(c)) = rvalue {
                    match c {
                        MirConstant::String(s) if s == "\\d+" => {
                            found_string_const = true;
                        }
                        MirConstant::Unit => {
                            found_unit_fallback = true;
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    assert!(found_string_const, "MIR lowering should produce MirConstant::String for regex literal");
    assert!(!found_unit_fallback, "MIR lowering must not have fallen back to MirConstant::Unit");
}

#[test]
fn test_schema_methods_validate_and_transform() {
    let src = r#"
        use "@std/schema" as s

        function test_schema_pipeline(): int
            local NameSchema = s.string().nonempty()
            local Transformed = NameSchema.transform(function(val: string): int return 42 end)
            local result: int = Transformed.validate("input")
            return result
        end
    "#;
    let res = typecheck_program(src);
    assert!(res.is_ok(), "schema transform and validate pipeline should typecheck: {:?}", res);
}

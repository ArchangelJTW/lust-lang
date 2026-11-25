#![cfg(feature = "std")]

use lust::{
    EmbeddedProgram, Value,
    ast::{ExprKind, ItemKind, StmtKind},
    intern::Interner,
    lexer::{Lexer, TokenKind},
    parser::Parser,
};

fn run_vm(source: &str, func: &str) -> lust::Result<Value> {
    let mut prog = EmbeddedProgram::builder()
        .module("main", source)
        .entry_module("main")
        .compile()?;
    prog.vm_mut().call(&format!("main.{}", func), vec![])
}

fn parse_program(source: &str) -> Vec<lust::ast::Item> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("tokenization should succeed");
    Parser::new(tokens)
        .parse()
        .expect("parsing should succeed")
}

#[test]
fn test_pipe_lexer_tokens() {
    let mut interner = Interner::new();
    let source = "| |= |> |> |";
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("tokenization");
    let kinds: Vec<TokenKind> = tokens.into_iter().map(|t| t.kind).collect();
    assert_eq!(
        kinds,
        vec![
            TokenKind::Pipe,
            TokenKind::PipeEqual,
            TokenKind::PipeGreater,
            TokenKind::PipeGreater,
            TokenKind::Pipe,
            TokenKind::Eof,
        ]
    );
}

#[test]
fn test_pipe_ast_desugaring_unary_and_call() {
    let source = r#"
        function test()
            local a = x |> f
            local b = x |> g(1, 2)
            local c = x |> f |> g(1)
        end
    "#;
    let items = parse_program(source);
    assert_eq!(items.len(), 1);

    if let ItemKind::Function(ref f) = items[0].kind {
        let body = &f.body;
        assert_eq!(body.len(), 3);

        // a = x |> f  => Call { callee: f, args: [x] }
        if let StmtKind::Local { initializer: Some(ref inits), .. } = body[0].kind {
            if let ExprKind::Call { ref callee, ref args, .. } = inits[0].kind {
                assert!(matches!(callee.kind, ExprKind::Identifier(ref name) if name == "f"));
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0].kind, ExprKind::Identifier(ref name) if name == "x"));
            } else {
                panic!("Expected Call expression for a");
            }
        } else {
            panic!("Expected Local statement for a");
        }

        // b = x |> g(1, 2) => Call { callee: g, args: [x, 1, 2] }
        if let StmtKind::Local { initializer: Some(ref inits), .. } = body[1].kind {
            if let ExprKind::Call { ref callee, ref args, .. } = inits[0].kind {
                assert!(matches!(callee.kind, ExprKind::Identifier(ref name) if name == "g"));
                assert_eq!(args.len(), 3);
                assert!(matches!(args[0].kind, ExprKind::Identifier(ref name) if name == "x"));
            } else {
                panic!("Expected Call expression for b");
            }
        } else {
            panic!("Expected Local statement for b");
        }

        // c = x |> f |> g(1) => Call { callee: g, args: [Call { callee: f, args: [x] }, 1] }
        if let StmtKind::Local { initializer: Some(ref inits), .. } = body[2].kind {
            if let ExprKind::Call { ref callee, ref args, .. } = inits[0].kind {
                assert!(matches!(callee.kind, ExprKind::Identifier(ref name) if name == "g"));
                assert_eq!(args.len(), 2);
                if let ExprKind::Call { callee: ref inner_callee, args: ref inner_args, .. } = args[0].kind {
                    assert!(matches!(inner_callee.kind, ExprKind::Identifier(ref name) if name == "f"));
                    assert_eq!(inner_args.len(), 1);
                } else {
                    panic!("Expected inner Call expression for c's first argument");
                }
            } else {
                panic!("Expected Call expression for c");
            }
        } else {
            panic!("Expected Local statement for c");
        }
    } else {
        panic!("Expected Function item");
    }
}

#[test]
fn test_pipe_operator_vm_unary() {
    let source = r#"
        function double(x: int): int
            return x * 2
        end

        function test(): int
            return 21 |> double
        end
    "#;
    let res = run_vm(source, "test").expect("execution should succeed");
    assert_eq!(res, Value::Int(42));
}

#[test]
fn test_pipe_operator_vm_thread_first_with_args() {
    let source = r#"
        function add(a: int, b: int): int
            return a + b
        end

        function sub(a: int, b: int): int
            return a - b
        end

        function test(): int
            return 50 |> sub(20) |> add(12)
        end
    "#;
    let res = run_vm(source, "test").expect("execution should succeed");
    // 50 |> sub(20) => sub(50, 20) = 30
    // 30 |> add(12) => add(30, 12) = 42
    assert_eq!(res, Value::Int(42));
}

#[test]
fn test_pipe_operator_vm_multiline_pipeline() {
    let source = r#"
        function double(x: int): int
            return x * 2
        end

        function add(a: int, b: int): int
            return a + b
        end

        function sub(a: int, b: int): int
            return a - b
        end

        function test(): int
            local res = 10
                |> double
                |> add(5)
                |> sub(3)
            return res
        end
    "#;
    let res = run_vm(source, "test").expect("execution should succeed");
    // 10 * 2 = 20; 20 + 5 = 25; 25 - 3 = 22
    assert_eq!(res, Value::Int(22));
}

#[test]
fn test_pipe_operator_vm_precedence_arithmetic_first() {
    let source = r#"
        function double(x: int): int
            return x * 2
        end

        function test(): int
            return 2 + 3 |> double
        end
    "#;
    let res = run_vm(source, "test").expect("execution should succeed");
    // (2 + 3) |> double => double(5) = 10
    assert_eq!(res, Value::Int(10));
}

#[test]
fn test_pipe_operator_vm_lambda_pipeline() {
    let source = r#"
        function apply_twice(x: int, f: function(int): int): int
            return f(f(x))
        end

        function test(): int
            local inc = function(n: int): int return n + 1 end
            return 10 |> apply_twice(inc)
        end
    "#;
    let res = run_vm(source, "test").expect("execution should succeed");
    // 10 |> apply_twice(inc) => apply_twice(10, inc) => inc(inc(10)) = 12
    assert_eq!(res, Value::Int(12));
}

#[test]
fn test_pipe_operator_vm_direct_anonymous_lambda() {
    let source = r#"
        function test(): int
            return 10 |> (function(x: int): int return x * 3 end)
        end
    "#;
    let res = run_vm(source, "test").expect("execution should succeed");
    assert_eq!(res, Value::Int(30));
}

#[test]
fn test_pipe_operator_vm_long_mixed_chain() {
    let source = r#"
        function double(x: int): int return x * 2 end
        function add(a: int, b: int): int return a + b end
        function mul(a: int, b: int): int return a * b end
        function sub(a: int, b: int): int return a - b end

        function test(): int
            return 2 |> double |> add(5) |> mul(3) |> sub(1)
        end
    "#;
    let res = run_vm(source, "test").expect("execution should succeed");
    // (((2 * 2) + 5) * 3) - 1 = (9 * 3) - 1 = 26
    assert_eq!(res, Value::Int(26));
}

#[test]
fn test_pipe_operator_vm_logical_expression_pipe() {
    let source = r#"
        function bool_to_int(b: bool): int
            if b then
                return 1
            else
                return 0
            end
        end

        function test(): int
            return (5 > 2 and 10 < 20) |> bool_to_int
        end
    "#;
    let res = run_vm(source, "test").expect("execution should succeed");
    assert_eq!(res, Value::Int(1));
}

#[test]
fn test_pipe_operator_chains_left_to_right() {
    let source = r#"
        function sub(a: int, b: int): int
            return a - b
        end

        function test(): int
            -- Left-to-right: sub(sub(100, 30), 10) = sub(70, 10) = 60
            -- If right-to-left: sub(100, sub(30, 10)) = sub(100, 20) = 80
            return 100 |> sub(30) |> sub(10)
        end
    "#;
    let res = run_vm(source, "test").expect("execution should succeed");
    assert_eq!(res, Value::Int(60));

    // Also verify AST nesting for left-to-right chaining:
    // a |> f |> g |> h  =>  h(g(f(a)))
    let ast_source = r#"
        function test()
            local res = a |> f |> g |> h
        end
    "#;
    let items = parse_program(ast_source);
    if let ItemKind::Function(ref func) = items[0].kind {
        if let StmtKind::Local { initializer: Some(ref inits), .. } = func.body[0].kind {
            // Outermost call is h
            if let ExprKind::Call { ref callee, ref args, .. } = inits[0].kind {
                assert!(matches!(callee.kind, ExprKind::Identifier(ref name) if name == "h"));
                assert_eq!(args.len(), 1);
                // Middle call is g
                if let ExprKind::Call { callee: ref mid_callee, args: ref mid_args, .. } = args[0].kind {
                    assert!(matches!(mid_callee.kind, ExprKind::Identifier(ref name) if name == "g"));
                    assert_eq!(mid_args.len(), 1);
                    // Innermost call is f
                    if let ExprKind::Call { callee: ref inner_callee, args: ref inner_args, .. } = mid_args[0].kind {
                        assert!(matches!(inner_callee.kind, ExprKind::Identifier(ref name) if name == "f"));
                        assert_eq!(inner_args.len(), 1);
                        assert!(matches!(inner_args[0].kind, ExprKind::Identifier(ref name) if name == "a"));
                    } else {
                        panic!("Expected innermost Call to f");
                    }
                } else {
                    panic!("Expected middle Call to g");
                }
            } else {
                panic!("Expected outermost Call to h");
            }
        }
    }
}

#[test]
fn test_pipe_operator_with_existing_expression_grammar() {
    let source = r#"
        function double(x: int): int
            return x * 2
        end

        function add(a: int, b: int): int
            return a + b
        end

        function abs(x: int): int
            if x < 0 then
                return -x
            else
                return x
            end
        end

        function test(): int
            -- Arithmetic precedence: 1 + 2 * 3 = 7 |> double = 14
            local r1 = 1 + 2 * 3 |> double

            -- Unary operator precedence: -15 |> abs = 15 |> double = 30
            local r2 = -15 |> abs |> double

            -- Bitwise precedence: (5 | 2) = 7 |> double = 14
            local r3 = 5 | 2 |> double

            -- Pipe nested within arithmetic
            local r4 = (10 |> double) + (5 |> double)

            return r1 + r2 + r3 + r4
        end
    "#;
    let res = run_vm(source, "test").expect("execution should succeed");
    // r1 = 14, r2 = 30, r3 = 14, r4 = 20 + 10 = 30; total = 14 + 30 + 14 + 30 = 88
    assert_eq!(res, Value::Int(88));
}

#[test]
fn test_pipe_non_callable_same_error_as_direct_call_compile_time() {
    // 1. Calling a typed non-callable variable
    let pipe_source = r#"
        function test(): int
            local not_a_func: int = 42
            return 10 |> not_a_func
        end
    "#;
    let direct_source = r#"
        function test(): int
            local not_a_func: int = 42
            return not_a_func(10)
        end
    "#;

    let pipe_err = run_vm(pipe_source, "test").unwrap_err();
    let direct_err = run_vm(direct_source, "test").unwrap_err();

    assert_eq!(
        pipe_err.to_string(),
        direct_err.to_string(),
        "Compile-time type error for non-callable pipe must match direct call error exactly"
    );

    // 2. Calling a literal non-callable
    let pipe_lit = r#"
        function test(): int
            return 10 |> 42
        end
    "#;
    let direct_lit = r#"
        function test(): int
            return 42(10)
        end
    "#;

    let pipe_lit_err = run_vm(pipe_lit, "test").unwrap_err();
    let direct_lit_err = run_vm(direct_lit, "test").unwrap_err();

    assert_eq!(
        pipe_lit_err.to_string(),
        direct_lit_err.to_string(),
        "Literal non-callable pipe error must match direct call error exactly"
    );
}

#[test]
fn test_pipe_non_callable_same_error_as_direct_call_runtime() {
    // Calling a dynamic/any variable that holds a non-callable value at runtime
    let pipe_source = r#"
        function test(): any
            local f: any = 42
            return 10 |> f
        end
    "#;
    let direct_source = r#"
        function test(): any
            local f: any = 42
            return f(10)
        end
    "#;

    let pipe_err = run_vm(pipe_source, "test").unwrap_err();
    let direct_err = run_vm(direct_source, "test").unwrap_err();

    assert_eq!(
        pipe_err.to_string(),
        direct_err.to_string(),
        "Runtime error for non-callable pipe must match direct call error exactly"
    );
}

#[test]
fn test_pipe_strict_mir_zero_cost_pipeline() {
    let source = r#"
        function double(x: int): int
            return x * 2
        end

        function add_five(x: int): int
            return x + 5
        end

        function compute(x: int): int
            return x |> double |> add_five
        end
    "#;

    let mut prog = EmbeddedProgram::builder()
        .with_config(lust::LustConfig::default().with_strict_mir(true))
        .module("main", source)
        .entry_module("main")
        .compile()
        .expect("Strict MIR compilation should succeed");

    let result = prog
        .vm_mut()
        .call("main.compute", vec![Value::Int(20)])
        .expect("Execution under strict MIR should succeed");
    assert_eq!(result, Value::Int(45));
}

#[cfg(feature = "cranelift")]
#[test]
fn test_pipe_cranelift_jit_lowered_call_execution() {
    use lust::ast::ItemKind;
    use lust::mir::lowering::AstLowering;
    use lust::mir::cranelift_backend::jit_engine::CraneliftJitCompiler;
    use lust::mir::ir::MirModule;

    let source = r#"
        function double(x: int): int
            return x * 2
        end

        function add_five(x: int): int
            return x + 5
        end

        function pipeline(x: int): int
            return x |> double |> add_five
        end
    "#;

    let items = parse_program(source);
    let mut module = MirModule::new("test_pipe_jit".into());

    for item in items {
        if let ItemKind::Function(func_def) = item.kind {
            let mir_fn = AstLowering::lower_function(&func_def).expect("lower function to MIR");
            module.add_function(mir_fn);
        }
    }

    let mut jit = CraneliftJitCompiler::new().expect("JIT initialization");
    let func_ptrs = jit.compile_module(&module).expect("Cranelift JIT compilation");
    let pipe_ptr = *func_ptrs.get("pipeline").expect("pipeline function found");
    let native_pipeline: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(pipe_ptr) };

    // 21 |> double (42) |> add_five (47)
    let res = native_pipeline(21);
    assert_eq!(res, 47);
}





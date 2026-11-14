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

fn try_parse_program(source: &str) -> lust::Result<Vec<lust::ast::Item>> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize()?;
    Parser::new(tokens).parse()
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

#[test]
fn test_pipe_placeholder_arbitrary_positions() {
    let source = r#"
        function sub(a: int, b: int): int
            return a - b
        end

        function clamp(min: int, val: int, max: int): int
            if val < min then return min end
            if val > max then return max end
            return val
        end

        function test(): int
            -- first pos: 50 |> sub(_, 20) => sub(50, 20) = 30
            local r1 = 50 |> sub(_, 20)
            -- last pos: 50 |> sub(100, _) => sub(100, 50) = 50
            local r2 = 50 |> sub(100, _)
            -- middle pos: 25 |> clamp(0, _, 100) => clamp(0, 25, 100) = 25
            local r3 = 25 |> clamp(0, _, 100)
            -- middle pos clamped low: -5 |> clamp(0, _, 100) => 0
            local r4 = -5 |> clamp(0, _, 100)
            -- middle pos clamped high: 150 |> clamp(0, _, 100) => 100
            local r5 = 150 |> clamp(0, _, 100)
            return r1 + r2 + r3 + r4 + r5
        end
    "#;
    let res = run_vm(source, "test").expect("execution should succeed");
    // 30 + 50 + 25 + 0 + 100 = 205
    assert_eq!(res, Value::Int(205));
}

#[test]
fn test_pipe_method_call_placeholder() {
    let source = r#"
        struct Calculator
            factor: int
        end

        impl Calculator
            function compute(self, base: int, extra: int): int
                return (base - extra) * self.factor
            end
        end

        function test(): int
            local calc = Calculator { factor = 2 }
            -- 30 piped into extra: calc:compute(100, 30) = (100 - 30) * 2 = 140
            local r1 = 30 |> calc:compute(100, _)
            -- 50 piped into base: calc:compute(50, 20) = (50 - 20) * 2 = 60
            local r2 = 50 |> calc:compute(_, 20)
            return r1 + r2
        end
    "#;
    let res = run_vm(source, "test").expect("execution should succeed");
    // 140 + 60 = 200
    assert_eq!(res, Value::Int(200));
}

#[test]
fn test_pipe_unwrap_option_and_result() {
    let source = r#"
        function make_some(x: int): Option<int>
            return Option.Some(x * 2)
        end

        function make_none(x: int): Option<int>
            return Option.None
        end

        function make_ok(x: int): Result<int, string>
            return Result.Ok(x + 10)
        end

        function test_unwrap_some(): int ! string
            return 21 |> make_some?
        end

        function test_unwrap_ok(): int ! string
            return 32 |> make_ok?
        end
    "#;
    let res_some = run_vm(source, "test_unwrap_some").expect("unwrap some should succeed");
    assert_eq!(res_some, Value::Int(42));

    let res_ok = run_vm(source, "test_unwrap_ok").expect("unwrap ok should succeed");
    assert_eq!(res_ok, Value::Int(42));

    let fail_source = r#"
        function make_none(x: int): Option<int>
            return Option.None
        end

        function test_fail(): int ! string
            return 10 |> make_none?
        end
    "#;
    assert!(run_vm(fail_source, "test_fail").is_err());
}

#[test]
fn test_pipe_unwrap_chained_with_placeholder() {
    let source = r#"
        function sub_res(a: int, b: int): Result<int, string>
            return Result.Ok(a - b)
        end

        function mul_res(a: int, b: int): Result<int, string>
            return Result.Ok(a * b)
        end

        function test(): int ! string
            -- 20 |> sub_res(100, _)? => Result.Ok(100 - 20) = 80
            -- 80 |> mul_res(2)? => Result.Ok(80 * 2) = 160
            return 20 |> sub_res(100, _)? |> mul_res(2)?
        end
    "#;
    let res = run_vm(source, "test").expect("execution should succeed");
    assert_eq!(res, Value::Int(160));
}

#[test]
fn test_currying_partial_application() {
    let source = r#"
        function add(a: int, b: int): int
            return a + b
        end

        function sub(a: int, b: int): int
            return a - b
        end

        function clamp(min: int, val: int, max: int): int
            if val < min then return min end
            if val > max then return max end
            return val
        end

        function test(): int
            local add10 = add(10, _)
            local r1 = add10(5) -- 15

            local sub_from_100 = sub(100, _)
            local r2 = sub_from_100(30) -- 70

            local sub_20 = sub(_, 20)
            local r3 = sub_20(50) -- 30

            local clamp_pct = clamp(0, _, 100)
            local r4 = clamp_pct(150) -- 100
            local r5 = clamp_pct(-20) -- 0
            local r6 = clamp_pct(42) -- 42

            local mul = function(a: int, b: int): int return a * b end
            local curried_mul = mul(_, _)
            local r7 = curried_mul(6, 7) -- 42

            return r1 + r2 + r3 + r4 + r5 + r6 + r7
        end
    "#;
    let res = run_vm(source, "test").expect("execution should succeed");
    // 15 + 70 + 30 + 100 + 0 + 42 + 42 = 299
    assert_eq!(res, Value::Int(299));
}

#[test]
fn test_currying_with_pipe_interop() {
    let source = r#"
        struct Calculator
            factor: int
        end

        impl Calculator
            function compute(self, base: int, extra: int): int
                return (base - extra) * self.factor
            end
        end

        function add(a: int, b: int): int
            return a + b
        end

        function test(): int
            -- 1. Pipe value into a curried function
            local add5 = add(5, _)
            local r1 = 37 |> add5

            -- 2. Pipe directly into placeholder
            local r2 = 37 |> add(5, _)

            -- 3. Pipe into method currying
            local calc = Calculator { factor = 2 }
            local m = calc:compute(10, _)
            local r3 = 5 |> m

            return r1 + r2 + r3
        end
    "#;
    let res = run_vm(source, "test").expect("execution should succeed");
    // (37 + 5) + (5 + 37) + (10 - 5) * 2 = 42 + 42 + 10 = 94
    assert_eq!(res, Value::Int(94));
}

#[test]
fn test_pipe_side_effects_evaluated_exactly_once() {
    let source = r#"
        struct BoxState
            count: int
        end

        impl BoxState
            function tick(self): int
                self.count += 1
                return 25
            end
        end

        function sub(a: int, b: int): int
            return a - b
        end

        function make_opt(x: int): Option<int>
            return Option.Some(x + 5)
        end

        function test(): int ! string
            local state = BoxState { count = 0 }

            -- state:tick() piped into placeholder: tick() must execute exactly once
            local r1 = state:tick() |> sub(100, _)
            local c1 = state.count

            -- state:tick() piped into unwrap: tick() must execute exactly once
            local r2 = state:tick() |> make_opt?
            local c2 = state.count

            return r1 + r2 + c1 + c2
        end
    "#;
    let res = run_vm(source, "test").expect("execution should succeed");
    // r1 = 100 - 25 = 75
    // c1 = 1
    // r2 = 25 + 5 = 30
    // c2 = 2
    // total = 75 + 30 + 1 + 2 = 108
    assert_eq!(res, Value::Int(108));
}

#[test]
fn test_pipe_parse_failures() {
    // 1. Multiple placeholders in single pipe step
    let src1 = "function t() return 10 |> f(_, _) end";
    let err1 = try_parse_program(src1).unwrap_err();
    assert!(
        err1.to_string().contains("Multiple placeholders"),
        "Expected multiple placeholders error, got: {}",
        err1
    );

    // 2. Multiple placeholders with extra arguments
    let src2 = "function t() return 10 |> f(1, _, _) end";
    let err2 = try_parse_program(src2).unwrap_err();
    assert!(
        err2.to_string().contains("Multiple placeholders"),
        "Expected multiple placeholders error, got: {}",
        err2
    );

    // 3. Multiple placeholders in method call
    let src3 = "function t() return 10 |> obj:m(_, _) end";
    let err3 = try_parse_program(src3).unwrap_err();
    assert!(
        err3.to_string().contains("Multiple placeholders"),
        "Expected multiple placeholders error, got: {}",
        err3
    );

    // 4. Placeholder as pipe source
    let src4 = "function t() return _ |> f end";
    let err4 = try_parse_program(src4).unwrap_err();
    assert!(
        err4.to_string().contains("pipe source"),
        "Expected pipe source error, got: {}",
        err4
    );

    // 5. Placeholder as pipe target without call
    let src5 = "function t() return 10 |> _ end";
    let err5 = try_parse_program(src5).unwrap_err();
    assert!(
        err5.to_string().contains("pipe target"),
        "Expected pipe target error, got: {}",
        err5
    );

    // 6. Standalone placeholder in assignment
    let src6 = "function t() local x = _ end";
    let err6 = try_parse_program(src6).unwrap_err();
    assert!(
        err6.to_string().contains("Placeholder '_' is only valid in call argument positions"),
        "Expected standalone placeholder error, got: {}",
        err6
    );

    // 7. Standalone placeholder in binary op
    let src7 = "function t() return 1 + _ end";
    let err7 = try_parse_program(src7).unwrap_err();
    assert!(
        err7.to_string().contains("Placeholder '_' is only valid in call argument positions"),
        "Expected placeholder binary op error, got: {}",
        err7
    );

    // 8. Invalid token after pipe
    let src8 = "function t() return 10 |> ? end";
    assert!(try_parse_program(src8).is_err());

    // 9. Multiple parenthesized placeholders
    let src9 = "function t() return 10 |> f((_), (_)) end";
    let err9 = try_parse_program(src9).unwrap_err();
    assert!(
        err9.to_string().contains("Multiple placeholders"),
        "Expected multiple placeholders error, got: {}",
        err9
    );

    // 10. Multiple nested parenthesized placeholders
    let src10 = "function t() return 10 |> f(((_)), _) end";
    let err10 = try_parse_program(src10).unwrap_err();
    assert!(
        err10.to_string().contains("Multiple placeholders"),
        "Expected multiple placeholders error, got: {}",
        err10
    );

    // 11. Parenthesized placeholder as pipe target without call
    let src11 = "function t() return 10 |> ((_)) end";
    let err11 = try_parse_program(src11).unwrap_err();
    assert!(
        err11.to_string().contains("pipe target")
            || err11
                .to_string()
                .contains("Placeholder '_' is only valid in call argument positions"),
        "Expected pipe target or invalid placeholder error, got: {}",
        err11
    );

    // 12. Parenthesized placeholder as pipe source
    let src12 = "function t() return ((_)) |> f end";
    let err12 = try_parse_program(src12).unwrap_err();
    assert!(
        err12.to_string().contains("pipe source")
            || err12
                .to_string()
                .contains("Placeholder '_' is only valid in call argument positions"),
        "Expected pipe source or invalid placeholder error, got: {}",
        err12
    );
}

#[test]
fn test_pipe_placeholder_strict_mir_zero_cost() {
    let source = r#"
        function sub(a: int, b: int): int
            return a - b
        end

        function compute(x: int): int
            return x |> sub(100, _)
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
        .call("main.compute", vec![Value::Int(30)])
        .expect("Execution under strict MIR should succeed");
    // 100 - 30 = 70
    assert_eq!(result, Value::Int(70));
}

#[test]
fn test_pipe_placeholder_cranelift_jit_execution() {
    use lust::ast::ItemKind;
    use lust::mir::lowering::AstLowering;
    use lust::mir::cranelift_backend::jit_engine::CraneliftJitCompiler;
    use lust::mir::ir::MirModule;

    let source = r#"
        function sub(a: int, b: int): int
            return a - b
        end

        function add(a: int, b: int): int
            return a + b
        end

        function pipeline(x: int): int
            return x |> sub(100, _) |> add(12)
        end
    "#;

    let items = parse_program(source);
    let mut module = MirModule::new("test_pipe_jit_placeholder".into());

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

    // 100 - 30 = 70; 70 + 12 = 82
    let res = native_pipeline(30);
    assert_eq!(res, 82);
}

#[test]
fn test_pipe_parenthesized_targets_and_nested_parens() {
    let source = r#"
        function sub(a: int, b: int): int
            return a - b
        end

        function add(a: int, b: int): int
            return a + b
        end

        function make_some(x: int): Option<int>
            return Option.Some(x * 2)
        end

        function test(): int ! string
            -- 1. Parens around target call with placeholder
            local r1 = 10 |> (sub(100, _)) -- 90

            -- 2. Double parens around target call
            local r2 = 10 |> ((sub(100, _))) -- 90

            -- 3. Parens around unwrap target
            local r3 = 21 |> (make_some?) -- 42

            -- 4. Parens around callee with unwrap outside
            local r4 = 21 |> (make_some)? -- 42

            -- 5. Nested parens around placeholder in call arg
            local r5 = 10 |> sub(100, ((_))) -- 90

            -- 6. Currying with nested parens around placeholder
            local curried_add = add((((_))), 5)
            local r6 = curried_add(10) -- 15

            return r1 + r2 + r3 + r4 + r5 + r6
        end
    "#;
    let res = run_vm(source, "test").expect("execution should succeed");
    // 90 + 90 + 42 + 42 + 90 + 15 = 369
    assert_eq!(res, Value::Int(369));
}

#[test]
fn test_pipe_method_call_unwrap() {
    let source = r#"
        struct Calculator
            factor: int
        end

        impl Calculator
            function compute_opt(self, base: int, extra: int): Option<int>
                return Option.Some((base - extra) * self.factor)
            end

            function default_opt(self, base: int): Option<int>
                return Option.Some(base * self.factor)
            end
        end

        function test(): int ! string
            local calc = Calculator { factor = 3 }

            -- method call unwrap with placeholder: (100 - 20) * 3 = 240
            local r1 = 20 |> calc:compute_opt(100, _)?

            -- method call unwrap with thread-first: 10 * 3 = 30
            local r2 = 10 |> calc:default_opt()?

            return r1 + r2
        end
    "#;
    let res = run_vm(source, "test").expect("execution should succeed");
    // 240 + 30 = 270
    assert_eq!(res, Value::Int(270));
}

#[test]
fn test_pipe_function_call_empty_parens_unwrap() {
    let source = r#"
        function make_opt(x: int): Option<int>
            return Option.Some(x + 100)
        end

        function test(): int ! string
            -- unwrap with empty parens: make_opt(50)? => 150
            return 50 |> make_opt()?
        end
    "#;
    let res = run_vm(source, "test").expect("execution should succeed");
    assert_eq!(res, Value::Int(150));
}





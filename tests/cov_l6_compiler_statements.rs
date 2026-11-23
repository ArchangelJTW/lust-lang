use lust::bytecode::compiler::Compiler;
use lust::bytecode::instruction::{Instruction, OpCode};
use lust::bytecode::{Function, Value};
use lust::error::LustError;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::VM;
use lust::LustConfig;

// =========================================================================
// TEST HARNESS & INDEPENDENT ORACLES
// =========================================================================

/// Compiles source using the full pipeline: Lexer -> Parser -> TypeChecker -> Compiler.
fn compile_source(source: &str) -> Result<Vec<Function>, LustError> {
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

/// Compiles source without type checker, directly parsing and compiling the module.
fn compile_direct(source: &str) -> Result<Vec<Function>, LustError> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize()?;
    let items = Parser::new(tokens).parse()?;
    let mut compiler = Compiler::new();
    compiler.compile_module(&items)
}

/// Executes source code in VM and returns the result of the main/script function.
#[allow(dead_code)]
fn run_vm(source: &str) -> Result<Value, LustError> {
    let functions = compile_source(source)?;
    let mut vm = VM::new();
    vm.load_functions(functions);
    vm.call("__script", vec![])
}

/// Independent Oracle: Verifies that a function chunk contains the expected opcode subsequence in order.
fn assert_has_opcode_subsequence(
    source: &str,
    func_name: &str,
    instructions: &[Instruction],
    expected: &[OpCode],
) {
    let actual: Vec<OpCode> = instructions.iter().map(|i| i.opcode()).collect();
    let mut actual_idx = 0;
    for &exp in expected {
        let mut found = false;
        while actual_idx < actual.len() {
            if actual[actual_idx] == exp {
                actual_idx += 1;
                found = true;
                break;
            }
            actual_idx += 1;
        }
        if !found {
            panic!(
                "Opcode oracle failure for function '{}'!\nSource snippet:\n{}\nExpected opcode: {:?}\nActual sequence: {:?}",
                func_name, source, exp, actual
            );
        }
    }
}

/// Helper to get a function by name or substring.
fn get_function<'a>(functions: &'a [Function], name_part: &str) -> &'a Function {
    functions
        .iter()
        .find(|f| f.name.contains(name_part))
        .unwrap_or_else(|| {
            let names: Vec<&str> = functions.iter().map(|f| f.name.as_str()).collect();
            panic!(
                "Function matching '{}' not found in compiled functions: {:?}",
                name_part, names
            )
        })
}

// =========================================================================
// 1. RETURN STATEMENT ORACLE TESTS
// =========================================================================

#[test]
fn test_return_outside_of_function_compiler_diagnostic() {
    let mut compiler = Compiler::new();
    let stmt = lust::ast::Stmt::new(
        lust::ast::StmtKind::Return(vec![]),
        lust::ast::Span::dummy(),
    );
    let result = compiler.compile_stmt(&stmt);
    match result {
        Err(LustError::CompileError(msg)) => {
            assert!(
                msg.contains("'return' outside of function"),
                "Expected diagnostic to mention return outside of function, got: {}",
                msg
            );
        }
        other => panic!("Expected CompileError for return outside function, got: {:?}", other),
    }
}

#[test]
fn test_return_outside_of_function_typechecker_diagnostic() {
    let expr = lust::ast::Expr::new(
        lust::ast::ExprKind::Return(vec![]),
        lust::ast::Span::dummy(),
    );
    let mut checker = TypeChecker::new();
    let err = checker.check_expr(&expr).expect_err("Return outside function must fail in typechecker");
    let err_str = format!("{:?}", err);
    assert!(
        err_str.contains("'return' outside of function"),
        "Expected 'return' outside of function, got: {}",
        err_str
    );
}

#[test]
fn test_return_variations_and_opcode_oracle() {
    let source = r#"
function f_nil()
    return
end

function f_single(): int
    return 42
end

function f_multi(): (int, int, int)
    return 1, 2, 3
end
"#;
    let functions = compile_source(source).unwrap_or_else(|e| panic!("Source failed:\n{}\nErr: {:?}", source, e));

    let fn_nil = get_function(&functions, "f_nil");
    assert_has_opcode_subsequence(source, "f_nil", &fn_nil.chunk.instructions, &[OpCode::Return]);
    assert!(matches!(
        fn_nil.chunk.instructions.last(),
        Some(Instruction::Return(255))
    ));

    let fn_single = get_function(&functions, "f_single");
    assert_has_opcode_subsequence(
        source,
        "f_single",
        &fn_single.chunk.instructions,
        &[OpCode::LoadConst, OpCode::Return],
    );

    let fn_multi = get_function(&functions, "f_multi");
    assert_has_opcode_subsequence(
        source,
        "f_multi",
        &fn_multi.chunk.instructions,
        &[OpCode::TupleNew, OpCode::Return],
    );
}

#[test]
fn test_return_error_constructors_and_static_err_path() {
    let source = r#"
function make_err(): Result<int, string>
    return Err("something failed")
end

function make_res_err(): Result<int, string>
    return Result.Err("explicit prefix")
end
"#;
    let functions = compile_source(source).unwrap_or_else(|e| panic!("Source failed:\n{}\nErr: {:?}", source, e));

    let f1 = get_function(&functions, "make_err");
    assert_has_opcode_subsequence(
        source,
        "make_err",
        &f1.chunk.instructions,
        &[OpCode::NewEnumVariant, OpCode::Return],
    );

    let f2 = get_function(&functions, "make_res_err");
    assert_has_opcode_subsequence(
        source,
        "make_res_err",
        &f2.chunk.instructions,
        &[OpCode::NewEnumVariant, OpCode::Return],
    );
}

#[test]
fn test_return_effect_thunk_escape_rejection() {
    let source = r#"
function bad_escape()
    local th = effect do return 42 end
    return th
end
"#;
    let err = compile_direct(source).expect_err("Returning an effect thunk must fail compilation");
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("cannot escape or be used as a value"),
        "Source:\n{}\nExpected cannot escape error, got: {}",
        source,
        msg
    );
}

// =========================================================================
// 2. CLOSURES & UPVALUE ORACLE TESTS
// =========================================================================

#[test]
fn test_closure_capturing_mutable_local_and_sharing() {
    let source = r#"
function make_counter()
    local count = 10
    local inc = function()
        count += 1
        return count
    end
    local get = function()
        return count
    end
    return inc, get
end
"#;
    let functions = compile_source(source).unwrap_or_else(|e| panic!("Source failed:\n{}\nErr: {:?}", source, e));

    let counter_fn = get_function(&functions, "make_counter");
    assert_has_opcode_subsequence(
        source,
        "make_counter",
        &counter_fn.chunk.instructions,
        &[OpCode::Closure, OpCode::Closure, OpCode::Return],
    );

    let inc_lambda = functions.iter().find(|f| f.name.starts_with("<lambda@")).unwrap();
    assert_has_opcode_subsequence(
        source,
        &inc_lambda.name,
        &inc_lambda.chunk.instructions,
        &[OpCode::LoadUpvalue, OpCode::AddInt, OpCode::StoreUpvalue],
    );
}

#[test]
fn test_deeply_nested_closures_parent_and_grandparent_capture() {
    let source = r#"
function outer(a: int)
    return function(b: int)
        return function(c: int)
            return a + b + c
        end
    end
end
"#;
    let functions = compile_source(source).unwrap_or_else(|e| panic!("Source failed:\n{}\nErr: {:?}", source, e));

    let outer_fn = get_function(&functions, "outer");
    assert_has_opcode_subsequence(
        source,
        "outer",
        &outer_fn.chunk.instructions,
        &[OpCode::Closure, OpCode::Return],
    );

    let lambdas: Vec<&Function> = functions
        .iter()
        .filter(|f| f.name.starts_with("<lambda@"))
        .collect();
    assert!(
        lambdas.len() >= 2,
        "Expected at least 2 compiled lambdas for nested closure, got {}",
        lambdas.len()
    );

    let inner_lambda = lambdas.last().unwrap();
    assert_has_opcode_subsequence(
        source,
        &inner_lambda.name,
        &inner_lambda.chunk.instructions,
        &[OpCode::LoadUpvalue, OpCode::LoadUpvalue, OpCode::AddInt, OpCode::Return],
    );
}

#[test]
fn test_closure_free_variables_full_ast_traversal() {
    let source = r#"
struct ObjContainer
    x: int
end

function comprehensive_closure_env()
    local a = 1
    local b = 2
    local arr = [1, 2]
    local obj = ObjContainer { x = 10 }

    local f = function()
        local sum = a + b
        local neg = -sum
        local fld = obj.x
        local idx = arr[0]
        local arr_lit = [a, b]
        local map_lit = { [1] = 2 }
        local tup_lit = (a, b)
        local casted = a as float
        local check = a is int

        if a > 0 then
            sum = sum + 1
        elseif a == 0 then
            sum = sum + 2
        else
            sum = sum + 3
        end

        while sum < 10 do
            sum = sum + 1
            if sum == 5 then continue end
            if sum == 9 then break end
        end

        for i = 1, 3, 1 do
            sum = sum + i
        end

        for x in arr do
            sum = sum + x
        end

        scoped do
            local scoped_v = sum * 2
        end

        local nested = function()
            return a + sum
        end

        return nested() + fld + idx + casted as int
    end
    return f()
end
"#;
    let functions = compile_source(source).unwrap_or_else(|e| panic!("Source failed:\n{}\nErr: {:?}", source, e));
    assert!(!functions.is_empty(), "Must compile all functions in comprehensive closure env");
}

// =========================================================================
// 3. METHODS, TRAITS & SELF ORACLE TESTS
// =========================================================================

#[test]
fn test_method_compilation_implicit_and_explicit_self() {
    let source = r#"
struct Vector2
    x: int
    y: int
end

impl Vector2
    function new(x: int, y: int): Vector2
        return Vector2 { x = x, y = y }
    end

    function norm_sq(self): int
        return self.x * self.x + self.y * self.y
    end

    function add(self: Vector2, other: Vector2): Vector2
        return Vector2 { x = self.x + other.x, y = self.y + other.y }
    end
end
"#;
    let functions = compile_source(source).unwrap_or_else(|e| panic!("Source failed:\n{}\nErr: {:?}", source, e));

    let static_m = get_function(&functions, "Vector2.new");
    assert!(!static_m.is_method, "Vector2.new must have is_method == false");

    let implicit_self = get_function(&functions, "Vector2:norm_sq");
    assert!(implicit_self.is_method, "Vector2:norm_sq must have is_method == true");

    let explicit_self = get_function(&functions, "Vector2:add");
    assert!(explicit_self.is_method, "Vector2:add must have is_method == true");
}

#[test]
fn test_trait_methods_and_default_implementations() {
    let source = r#"
struct Entity
    id: int
end

trait Identifiable
    function get_id(self): int
    function has_valid_id(self): bool
        return self.id > 0
    end
end

impl Identifiable for Entity
    function get_id(self): int
        return self.id
    end
end
"#;
    let functions = compile_source(source).unwrap_or_else(|e| panic!("Source failed:\n{}\nErr: {:?}", source, e));

    let entity_get_id = get_function(&functions, "Entity:get_id");
    assert!(entity_get_id.is_method, "Entity:get_id must be an instance method");
    assert_has_opcode_subsequence(
        source,
        "Entity:get_id",
        &entity_get_id.chunk.instructions,
        &[OpCode::GetField, OpCode::Return],
    );
}

#[test]
fn test_map_filter_reduce_methods_opcodes() {
    let source = r#"
function run_transforms(arr: Array<int>): int
    local doubled = arr:map(function(x: int): int return x * 2 end)
    local filtered = doubled:filter(function(x: int): bool return x > 2 end)
    local total = filtered:reduce(0, function(acc: int, x: int): int return acc + x end)
    return total
end
"#;
    let functions = compile_source(source).unwrap_or_else(|e| panic!("Source failed:\n{}\nErr: {:?}", source, e));
    let run_fn = get_function(&functions, "run_transforms");

    assert_has_opcode_subsequence(
        source,
        "run_transforms",
        &run_fn.chunk.instructions,
        &[
            OpCode::NewArray,
            OpCode::ArrayLen,
            OpCode::SubInt,
            OpCode::LeInt,
            OpCode::GetIndex,
            OpCode::Call,
            OpCode::CallMethod,
            OpCode::AddInt,
        ],
    );

    assert_has_opcode_subsequence(
        source,
        "run_transforms",
        &run_fn.chunk.instructions,
        &[
            OpCode::NewArray,
            OpCode::ArrayLen,
            OpCode::JumpIfNot,
            OpCode::CallMethod,
        ],
    );

    assert_has_opcode_subsequence(
        source,
        "run_transforms",
        &run_fn.chunk.instructions,
        &[
            OpCode::ArrayLen,
            OpCode::LeInt,
            OpCode::GetIndex,
            OpCode::Call,
            OpCode::AddInt,
        ],
    );
}

#[test]
fn test_effect_thunk_synchronous_run_and_captures() {
    let source = r#"
function run_effects(): int
    local base: int = 100
    local th = effect do
        local temp: int = base + 5
        return temp * 2
    end
    return th:run_sync()
end
"#;
    let functions = compile_source(source).unwrap_or_else(|e| panic!("Source failed:\n{}\nErr: {:?}", source, e));
    let run_fn = get_function(&functions, "run_effects");

    assert_has_opcode_subsequence(
        source,
        "run_effects",
        &run_fn.chunk.instructions,
        &[
            OpCode::LoadConst,
            OpCode::LoadNil,
            OpCode::AddInt,
            OpCode::MulInt,
            OpCode::Move,
            OpCode::Return,
        ],
    );
}

#[test]
fn test_effect_thunk_compiler_rejections_and_boundaries() {
    let cases = vec![
        (
            "local th = effect do return 1 end\nlocal copy = th",
            "cannot be aliased or copied",
        ),
        (
            "local th = effect do return 1 end\nth = 2",
            "cannot be reassigned or mutated",
        ),
        (
            "local th = effect do return 1 end\nth += 1",
            "cannot be reassigned or mutated",
        ),
        (
            "local th = effect do local f = function() return 1 end\nreturn f() end",
            "Lambdas are not supported inside effect thunk",
        ),
        (
            "local th = effect do local inner = effect do return 1 end\nreturn 2 end",
            "Nested effect blocks are not supported",
        ),
        (
            "local x: int = 1\nlocal th = effect do x = 2\nreturn x end",
            "Write to captured variable 'x' is not supported in effect thunk",
        ),
        (
            "local x: int = 1\nlocal th = effect do x += 2\nreturn x end",
            "Write to captured variable 'x' is not supported in effect thunk",
        ),
        (
            "local th = effect do return end",
            "Effect thunk return must return an integer value",
        ),
        (
            "local th = effect do defer 1\nreturn 2 end",
            "Cleanups (defer/errdefer) are not supported inside effect thunk",
        ),
        (
            "local th = effect do errdefer 1\nreturn 2 end",
            "Cleanups (defer/errdefer) are not supported inside effect thunk",
        ),
        (
            "local s = \"not an int\"\nlocal th = effect do return s end",
            "Unsupported capture type",
        ),
    ];

    for (code, expected_err) in cases {
        let err = compile_direct(code).expect_err(&format!("Code should fail compilation:\n{}", code));
        let err_msg = format!("{:?}", err);
        assert!(
            err_msg.contains(expected_err),
            "Code snippet:\n{}\nExpected error containing: '{}', got: '{}'",
            code,
            expected_err,
            err_msg
        );
    }
}

// =========================================================================
// 4. STATEMENT KINDS: WHILE, FOR, IF, ASSIGNMENTS
// =========================================================================

#[test]
fn test_break_and_continue_outside_loop_diagnostics() {
    let break_stmt = lust::ast::Stmt::new(lust::ast::StmtKind::Break, lust::ast::Span::dummy());
    let continue_stmt = lust::ast::Stmt::new(lust::ast::StmtKind::Continue, lust::ast::Span::dummy());

    let mut compiler = Compiler::new();
    let break_err = compiler.compile_stmt(&break_stmt).expect_err("break outside loop must error");
    assert!(
        format!("{:?}", break_err).contains("'break' outside of loop"),
        "Expected 'break' outside of loop, got: {:?}",
        break_err
    );

    let continue_err = compiler.compile_stmt(&continue_stmt).expect_err("continue outside loop must error");
    assert!(
        format!("{:?}", continue_err).contains("'continue' outside of loop"),
        "Expected 'continue' outside of loop, got: {:?}",
        continue_err
    );
}

#[test]
fn test_while_loop_break_and_continue_opcodes() {
    let source = r#"
function run_while(): int
    local i = 0
    while i < 10 do
        i += 1
        if i == 3 then
            continue
        end
        if i == 7 then
            break
        end
    end
    return i
end
"#;
    let functions = compile_source(source).unwrap_or_else(|e| panic!("Source failed:\n{}\nErr: {:?}", source, e));
    let fn_while = get_function(&functions, "run_while");

    assert_has_opcode_subsequence(
        source,
        "run_while",
        &fn_while.chunk.instructions,
        &[
            OpCode::LtInt,
            OpCode::JumpIfNot,
            OpCode::AddInt,
            OpCode::EqInt,
            OpCode::JumpIfNot,
            OpCode::Jump,
            OpCode::EqInt,
            OpCode::JumpIfNot,
            OpCode::Jump,
            OpCode::Jump,
            OpCode::Return,
        ],
    );
}

#[test]
fn test_for_numeric_loop_variations() {
    let source = r#"
function for_loops(dynamic_step: int): int
    local sum = 0

    for i = 1, 10, 2 do
        sum += i
    end

    for i = 10, 1, -1 do
        sum += i
    end

    for i = 1, 5 do
        sum += i
    end

    for i = 1, 10, dynamic_step do
        sum += i
        if sum > 100 then
            break
        else
            continue
        end
    end

    return sum
end
"#;
    let functions = compile_source(source).unwrap_or_else(|e| panic!("Source failed:\n{}\nErr: {:?}", source, e));
    let fn_for = get_function(&functions, "for_loops");

    assert_has_opcode_subsequence(
        source,
        "for_loops",
        &fn_for.chunk.instructions,
        &[OpCode::LeInt, OpCode::JumpIfNot, OpCode::AddInt, OpCode::Jump],
    );

    assert_has_opcode_subsequence(
        source,
        "for_loops",
        &fn_for.chunk.instructions,
        &[OpCode::GeInt, OpCode::JumpIfNot, OpCode::AddInt, OpCode::Jump],
    );

    assert_has_opcode_subsequence(
        source,
        "for_loops",
        &fn_for.chunk.instructions,
        &[
            OpCode::LtInt,
            OpCode::Not,
            OpCode::GeInt,
            OpCode::LeInt,
            OpCode::And,
            OpCode::Or,
            OpCode::JumpIfNot,
        ],
    );
}

#[test]
fn test_for_in_loop_variations() {
    let source = r#"
function test_for_in(arr: Array<int>): int
    local total = 0

    for x in 1..10 do
        total += x
    end

    for elem in arr do
        total += elem
    end

    for elem in arr:iter() do
        total += elem
        if total > 50 then break else continue end
    end

    return total
end
"#;
    let functions = compile_source(source).unwrap_or_else(|e| panic!("Source failed:\n{}\nErr: {:?}", source, e));
    let fn_for_in = get_function(&functions, "test_for_in");

    assert_has_opcode_subsequence(
        source,
        "test_for_in",
        &fn_for_in.chunk.instructions,
        &[OpCode::LtInt, OpCode::JumpIfNot, OpCode::AddInt, OpCode::Jump],
    );

    assert_has_opcode_subsequence(
        source,
        "test_for_in",
        &fn_for_in.chunk.instructions,
        &[
            OpCode::ArrayLen,
            OpCode::LtInt,
            OpCode::JumpIfNot,
            OpCode::GetIndex,
            OpCode::AddInt,
            OpCode::Jump,
        ],
    );
}

#[test]
fn test_for_in_inclusive_range_ast_direct() {
    let start = lust::ast::Expr::new(
        lust::ast::ExprKind::Literal(lust::ast::Literal::Integer(1)),
        lust::ast::Span::dummy(),
    );
    let end = lust::ast::Expr::new(
        lust::ast::ExprKind::Literal(lust::ast::Literal::Integer(10)),
        lust::ast::Span::dummy(),
    );
    let iterator = lust::ast::Expr::new(
        lust::ast::ExprKind::Range {
            start: Box::new(start),
            end: Box::new(end),
            inclusive: true,
        },
        lust::ast::Span::dummy(),
    );
    let stmt = lust::ast::Stmt::new(
        lust::ast::StmtKind::ForIn {
            variables: vec!["i".to_string()],
            iterator,
            body: vec![],
        },
        lust::ast::Span::dummy(),
    );
    let item = lust::ast::Item::new(
        lust::ast::ItemKind::Script(vec![stmt]),
        lust::ast::Span::dummy(),
    );
    let mut compiler = Compiler::new();
    let functions = compiler.compile_module(&[item]).expect("compile inclusive for-in loop should succeed");

    let script_fn = get_function(&functions, "__script");
    assert_has_opcode_subsequence("inclusive_range_ast", "__script", &script_fn.chunk.instructions, &[OpCode::LeInt]);
}

#[test]
fn test_if_then_elseif_else_and_pattern_branches() {
    let source = r#"
function classify(opt: Option<int>, n: int): int
    if opt is Some(val) then
        return val + 1
    elseif opt is None then
        return 0
    elseif n > 100 then
        return 100
    elseif n < 0 then
        return -1
    else
        return n
    end
end
"#;
    let functions = compile_source(source).unwrap_or_else(|e| panic!("Source failed:\n{}\nErr: {:?}", source, e));
    let fn_classify = get_function(&functions, "classify");

    assert_has_opcode_subsequence(
        source,
        "classify",
        &fn_classify.chunk.instructions,
        &[
            OpCode::IsEnumVariant,
            OpCode::JumpIfNot,
            OpCode::GetEnumValue,
            OpCode::Return,
            OpCode::IsEnumVariant,
            OpCode::JumpIfNot,
            OpCode::Return,
            OpCode::GtInt,
            OpCode::JumpIfNot,
            OpCode::Return,
            OpCode::LtInt,
            OpCode::JumpIfNot,
            OpCode::Return,
            OpCode::Return,
        ],
    );
}

#[test]
fn test_compound_assignments_and_string_concat_optimization() {
    let source = r#"
struct ObjContainer
    val: int
end

function compound_ops(arr: Array<int>, obj: ObjContainer): int
    local x = 100
    x += 10
    x -= 5
    x *= 2
    x /= 3
    x %= 7
    x &= 15
    x |= 4
    x ^= 2
    x <<= 2
    x >>= 1

    obj.val += 50
    arr[0] += 25

    local s = "hello"
    s = s .. " world"

    return x + obj.val + arr[0]
end
"#;
    let functions = compile_source(source).unwrap_or_else(|e| panic!("Source failed:\n{}\nErr: {:?}", source, e));
    let fn_comp = get_function(&functions, "compound_ops");

    assert_has_opcode_subsequence(
        source,
        "compound_ops",
        &fn_comp.chunk.instructions,
        &[
            OpCode::AddInt,
            OpCode::SubInt,
            OpCode::MulInt,
            OpCode::DivInt,
            OpCode::ModInt,
            OpCode::BitAnd,
            OpCode::BitOr,
            OpCode::PowInt,
            OpCode::Shl,
            OpCode::Shr,
            OpCode::GetField,
            OpCode::AddInt,
            OpCode::SetField,
            OpCode::GetIndex,
            OpCode::Add,
            OpCode::SetIndex,
            OpCode::Concat,
            OpCode::Return,
        ],
    );
}

#[test]
fn test_invalid_compound_assignment_target_rejection() {
    let source = "10 += 5";
    let err = compile_direct(source).expect_err("Compound assignment to literal must fail");
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("Invalid compound assignment target"),
        "Source:\n{}\nExpected 'Invalid compound assignment target', got: {}",
        source,
        msg
    );
}

#[test]
fn test_local_bindings_and_tuple_unpacking() {
    let source = r#"
function locals_test(): int
    local a = 1
    local b: int = 2
    local c = a
    local x, y, z = 10, 20, 30
    local uninit
    return a + b + c + x + y + z
end
"#;
    let functions = compile_source(source).unwrap_or_else(|e| panic!("Source failed:\n{}\nErr: {:?}", source, e));
    let fn_locals = get_function(&functions, "locals_test");

    assert_has_opcode_subsequence(
        source,
        "locals_test",
        &fn_locals.chunk.instructions,
        &[
            OpCode::LoadConst,
            OpCode::LoadConst,
            OpCode::TupleNew,
            OpCode::TupleGet,
            OpCode::Move,
            OpCode::TupleGet,
            OpCode::Move,
            OpCode::TupleGet,
            OpCode::Move,
            OpCode::LoadNil,
            OpCode::Return,
        ],
    );
}

#[test]
fn test_multiple_assignment_tuple_unpacking() {
    let source = r#"
function swap_test(): (int, int)
    local a = 1
    local b = 2
    a, b = b, a
    return a, b
end
"#;
    let functions = compile_source(source).unwrap_or_else(|e| panic!("Source failed:\n{}\nErr: {:?}", source, e));
    let fn_swap = get_function(&functions, "swap_test");

    assert_has_opcode_subsequence(
        source,
        "swap_test",
        &fn_swap.chunk.instructions,
        &[
            OpCode::TupleNew,
            OpCode::TupleGet,
            OpCode::Move,
            OpCode::TupleGet,
            OpCode::Move,
            OpCode::TupleNew,
            OpCode::Return,
        ],
    );
}

#[test]
fn test_defer_and_scoped_blocks_opcodes() {
    let source = r#"
function defer_scope(): int
    local res = 0
    scoped do
        local inside = 10
        res += inside
    end
    defer res += 1
    errdefer res += 2
    return res
end
"#;
    let functions = compile_source(source).unwrap_or_else(|e| panic!("Source failed:\n{}\nErr: {:?}", source, e));
    let fn_def = get_function(&functions, "defer_scope");

    assert_has_opcode_subsequence(
        source,
        "defer_scope",
        &fn_def.chunk.instructions,
        &[OpCode::AddInt, OpCode::AddInt, OpCode::Return],
    );
}

// =========================================================================
// 5. EFFECT THUNK DEEP TRAVERSAL TESTS
// =========================================================================

#[test]
fn test_effect_thunk_all_statement_and_expression_forms() {
    let source = r#"
function effect_thunk_full(): int
    local base: int = 10
    local th = effect do
        local l_val: int = base + 1
        l_val = l_val + 2
        l_val += 3
        local arr = [1, 2, 3]
        local tup = (4, 5)
        local map_data = { [1] = 2 }
        local p = l_val as int
        local is_ok = l_val is int

        if l_val > 20 then
            l_val += 1
        elseif l_val == 20 then
            l_val += 2
        else
            l_val += 3
        end

        while l_val < 30 do
            l_val += 1
            if l_val == 25 then continue end
            if l_val == 29 then break end
        end

        for i = 1, 3, 1 do
            l_val += i
        end

        for elem in 1..3 do
            l_val += elem
        end

        scoped do
            local inner = l_val + 1
        end

        return l_val
    end

    return th:run_sync()
end
"#;
    let functions = compile_source(source).unwrap_or_else(|e| panic!("Source failed:\n{}\nErr: {:?}", source, e));
    assert!(!functions.is_empty(), "Effect thunk with full construct set must compile");
}

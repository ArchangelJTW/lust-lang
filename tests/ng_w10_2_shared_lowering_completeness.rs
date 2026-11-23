#![cfg(feature = "std")]

use lust::ast::{
    BinaryOp, Expr, ExprKind, FunctionDef, FunctionParam, Item, ItemKind, Literal, Span, Stmt,
    StmtKind, Type, TypeKind, Visibility,
};
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::ir::{
    MirCleanupKind, MirModule, MirType, Operand, Place, Rvalue, Statement, Terminator,
};
use lust::mir::lowering::AstLowering;
use lust::mir::verifier::MirVerifier;
use lust::mir::vm_backend::{MirToBytecodeCompiler, MirVmBackendError};
use lust::parser::Parser;

fn parse_and_lower_module(source: &str, module_name: &str) -> MirModule {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("lex source");
    let items = Parser::new(tokens).parse().expect("parse source");
    let module = AstLowering::lower_module_items(module_name, &items).expect("lower module to MIR");
    MirVerifier::verify_module(&module).expect("MIR module must pass MirVerifier");
    module
}

/// Test 1: All 4 previously-failing Phase 1 benchmark functions (`b02_float_array`,
/// `b08_array_rw`, `b08_map_rw`, `b11_array_sum`) from `phase1_benches.rs` lower cleanly
/// with 0 skipped functions, direct `Place::Index` for `array.get:unwrap()` / `array.set()`,
/// and `map.new`/`map.set`/`map.get` for `ExprKind::Map` + `map.get:unwrap()`.
#[test]
fn test_ng_w10_2_phase1_benches_zero_skipped_functions() {
    let source = r#"
function b02_float_array(n: int): float
    local a: Array<float> = []
    local b: Array<float> = []
    local i: int = 0
    while i < n do
        array.push(a, (i % 97) / 97.0)
        array.push(b, (i % 89) / 89.0)
        i = i + 1
    end
    local total: float = 0.0
    i = 0
    while i < n do
        local va: float = array.get(a, i):unwrap()
        local vb: float = array.get(b, i):unwrap()
        total = total + (va * vb + 0.5)
        i = i + 1
    end
    return total
end

function b08_array_rw(n: int): int
    local arr: Array<int> = []
    local i: int = 0
    while i < n do
        array.push(arr, (i * 7 + 3) % 1009)
        i = i + 1
    end
    local sum: int = 0
    i = 0
    while i < n do
        local idx: int = (i * 13) % n
        local v: int = array.get(arr, idx):unwrap()
        array.set(arr, idx, v + 1)
        sum = sum + v
        i = i + 1
    end
    return sum
end

function b08_map_rw(n: int): int
    local m: Map<int, int> = {}
    local i: int = 0
    while i < n do
        map.set(m, i % 128, (i * 7 + 3) % 1009)
        i = i + 1
    end
    local sum: int = 0
    i = 0
    while i < n do
        local k: int = (i * 13) % 128
        local v: int = map.get(m, k):unwrap()
        map.set(m, k, v + 1)
        sum = sum + v
        i = i + 1
    end
    return sum
end

function b11_array_sum(arr: Array<int>): int
    local n: int = array.len(arr)
    local i: int = 0
    local s: int = 0
    while i < n do
        s = s + array.get(arr, i):unwrap()
        i = i + 1
    end
    return s
end
"#;

    let module = parse_and_lower_module(source, "phase1_subset");
    let names: Vec<&str> = module
        .functions
        .iter()
        .map(|f| f.signature.name.as_str())
        .collect();
    assert_eq!(
        module.functions.len(),
        4,
        "Expected all 4 functions lowered with 0 skipped, got: {names:?}"
    );
    for expected in ["b02_float_array", "b08_array_rw", "b08_map_rw", "b11_array_sum"] {
        assert!(
            names.contains(&expected),
            "Missing lowered MIR function '{expected}' in {names:?}"
        );
    }

    // Verify `b08_array_rw` lowered `array.get(arr, idx):unwrap()` and `array.set(arr, idx, v + 1)`
    // directly to `Place::Index` read and write, with zero calls to `array.get` or `array.set`.
    let b08_arr = module
        .functions
        .iter()
        .find(|f| f.signature.name == "b08_array_rw")
        .expect("b08_array_rw");
    let has_index_read = b08_arr.blocks.iter().any(|bb| {
        bb.statements.iter().any(|stmt| {
            matches!(
                stmt,
                Statement::Assign(
                    Place::Local(_),
                    Rvalue::Use(Operand::Copy(Place::Index(..))),
                    _
                )
            )
        })
    });
    let has_index_write = b08_arr.blocks.iter().any(|bb| {
        bb.statements.iter().any(|stmt| {
            matches!(
                stmt,
                Statement::Assign(Place::Index(..), Rvalue::Use(_), _)
            )
        })
    });
    let calls_array_get_or_set = b08_arr.blocks.iter().any(|bb| {
        matches!(
            &bb.terminator,
            Some(Terminator::Call { callee, .. }) if callee == "array.get" || callee == "array.set"
        )
    });
    assert!(has_index_read, "b08_array_rw must emit Place::Index read for array.get:unwrap()");
    assert!(has_index_write, "b08_array_rw must emit Place::Index write for array.set()");
    assert!(
        !calls_array_get_or_set,
        "b08_array_rw must not leave dynamic array.get/array.set calls"
    );

    // Verify `b08_map_rw` lowered `{}` -> `map.new` and `map.get(m, k):unwrap()` -> `map.get`.
    let b08_map = module
        .functions
        .iter()
        .find(|f| f.signature.name == "b08_map_rw")
        .expect("b08_map_rw");
    let has_map_new = b08_map.blocks.iter().any(|bb| {
        matches!(&bb.terminator, Some(Terminator::Call { callee, .. }) if callee == "map.new")
    });
    let has_map_get = b08_map.blocks.iter().any(|bb| {
        matches!(&bb.terminator, Some(Terminator::Call { callee, .. }) if callee == "map.get")
    });
    assert!(has_map_new, "b08_map_rw must emit Terminator::Call to map.new for {{}}");
    assert!(has_map_get, "b08_map_rw must emit Terminator::Call to map.get for map.get:unwrap()");

    // Verify `b02_float_array` also lowered `array.get:unwrap()` to `Place::Index` with Float element type.
    let b02 = module
        .functions
        .iter()
        .find(|f| f.signature.name == "b02_float_array")
        .expect("b02_float_array");
    let has_float_index_read = b02.blocks.iter().any(|bb| {
        bb.statements.iter().any(|stmt| {
            matches!(
                stmt,
                Statement::Assign(
                    Place::Local(dst),
                    Rvalue::Use(Operand::Copy(Place::Index(..))),
                    _
                ) if b02.locals[dst.0].ty == MirType::Float
            )
        })
    });
    assert!(
        has_float_index_read,
        "b02_float_array must lower array.get(a, i):unwrap() to typed Float Place::Index read"
    );
}

/// Test 2: `ItemKind::Impl` lowers methods into `MirModule` with qualified name `"Counter:bump"`.
#[test]
fn test_ng_w10_2_impl_block_lowers_counter_bump() {
    let source = r#"
struct Counter
    value: int
end

impl Counter
    function bump(self, delta: int): int
        self.value = self.value + delta
        return self.value
    end
end

function run_counter(n: int): int
    local c: Counter = Counter { value = 0 }
    return c:bump(n)
end
"#;

    let module = parse_and_lower_module(source, "counter_mod");
    let bump = module
        .functions
        .iter()
        .find(|f| f.signature.name == "Counter:bump")
        .expect("ItemKind::Impl must produce MirFunction 'Counter:bump'");
    assert_eq!(bump.params.len(), 2, "Counter:bump must have (self, delta) params");
    assert_eq!(
        bump.signature.param_types[0],
        MirType::Struct("Counter".to_string()),
        "self param must resolve to MirType::Struct(\"Counter\")"
    );
    assert_eq!(bump.signature.param_types[1], MirType::Int);
    assert_eq!(bump.signature.return_type, MirType::Int);
}

/// Test 3: `if flags[i] is Ok(f)` records `(flags_local, i_local)` in `MirFunction::proven_in_bounds`
/// for `then_block` ONLY (and NOT for `else_block` or `merge_block`).
#[test]
fn test_ng_w10_2_proven_in_bounds_on_then_block_only() {
    let source = r#"
function check_flag(flags: Array<bool>, i: int): int
    local acc: int = 0
    if flags[i] is Ok(f) then
        if f then
            acc = acc + 1
        end
    else
        acc = acc - 1
    end
    return acc
end
"#;

    let module = parse_and_lower_module(source, "bounds_mod");
    let func = module
        .functions
        .iter()
        .find(|f| f.signature.name == "check_flag")
        .expect("check_flag");

    let flags_local = func
        .locals
        .iter()
        .find(|l| l.name.as_deref() == Some("flags"))
        .expect("flags param local")
        .id;
    let i_local = func
        .locals
        .iter()
        .find(|l| l.name.as_deref() == Some("i"))
        .expect("i param local")
        .id;

    // Locate the Cond terminator for `if flags[i] is Ok(f)` (in entry block 0).
    let (then_block, else_block) = match &func.blocks[func.entry.0].terminator {
        Some(Terminator::Cond {
            then_block,
            else_block,
            ..
        }) => (*then_block, *else_block),
        other => panic!("Expected Terminator::Cond in entry block, got {other:?}"),
    };

    let then_entries = func
        .proven_in_bounds
        .get(&then_block.0)
        .expect("then_block must have proven_in_bounds entry");
    assert!(
        then_entries.contains(&(flags_local, i_local)),
        "then_block proven_in_bounds must contain ({flags_local}, {i_local}), got {then_entries:?}"
    );

    // DO NOT invariant: else_block and merge_block must NOT have proven_in_bounds.
    assert!(
        !func.proven_in_bounds.contains_key(&else_block.0),
        "else_block must NOT have proven_in_bounds"
    );
    assert_eq!(
        func.proven_in_bounds.len(),
        1,
        "Only the single then_block of `if flags[i] is Ok(f)` may carry proven_in_bounds; merge_block and else_block must remain absent"
    );
}

/// Test 4: A module with custom `enum Shape`, a closure capturing `offset`
/// (`fn(x: int): int => x + offset` / `function(x: int): int return x + offset end`),
/// and an `errdefer { ... }` block lowers cleanly to `MirModule` with populated
/// `enum_defs`, `Rvalue::Closure` (+ lifted `__lust_lambda_0`), and `MirCleanupKind::ErrDefer`.
#[test]
fn test_ng_w10_2_enum_closure_and_errdefer_module_lowering() {
    let source = r#"
enum Shape
    Point
    Circle(float)
    Rect(int, int)
end

function make_adder_with_cleanup(offset: int): int
    local cleaned: int = 0
    errdefer {
        cleaned = 1
    }
    local add = function(x: int): int
        return x + offset
    end
    return offset
end
"#;

    let module = parse_and_lower_module(source, "aot_completeness_mod");

    // 4a. Verify `module.enum_defs` has `Shape` with all 3 variants and field types.
    let shape_enum = module
        .enum_defs
        .iter()
        .find(|e| e.name == "Shape")
        .expect("module.enum_defs must contain 'Shape'");
    assert_eq!(
        shape_enum.variants,
        vec![
            ("Point".to_string(), vec![]),
            ("Circle".to_string(), vec![MirType::Float]),
            ("Rect".to_string(), vec![MirType::Int, MirType::Int]),
        ]
    );

    // 4b. Verify `make_adder_with_cleanup` has `Rvalue::Closure` capturing `offset`
    //     and `cleanup_blocks` containing `MirCleanupKind::ErrDefer`.
    let main_fn = module
        .functions
        .iter()
        .find(|f| f.signature.name == "make_adder_with_cleanup")
        .expect("make_adder_with_cleanup");

    let mut found_closure: Option<(String, Vec<(String, MirType)>)> = None;
    for bb in &main_fn.blocks {
        for stmt in &bb.statements {
            if let Statement::Assign(
                _,
                Rvalue::Closure {
                    func_name,
                    captures,
                },
                _,
            ) = stmt
            {
                found_closure = Some((
                    func_name.clone(),
                    captures
                        .iter()
                        .map(|(n, _, ty)| (n.clone(), ty.clone()))
                        .collect(),
                ));
            }
        }
    }
    let (lifted_name, captured) =
        found_closure.expect("make_adder_with_cleanup must emit Rvalue::Closure");
    assert_eq!(lifted_name, "__lust_lambda_0");
    assert_eq!(captured, vec![("offset".to_string(), MirType::Int)]);

    // Verify lifted `__lust_lambda_0` function exists in `module.functions` with captures first.
    let lifted_fn = module
        .functions
        .iter()
        .find(|f| f.signature.name == "__lust_lambda_0")
        .expect("module.functions must contain lifted '__lust_lambda_0'");
    assert_eq!(lifted_fn.params.len(), 2, "lifted lambda must take (offset, x)");
    assert_eq!(
        lifted_fn.locals[lifted_fn.params[0].0].name.as_deref(),
        Some("offset"),
        "captured variable must be first parameter of lifted lambda"
    );
    assert_eq!(
        lifted_fn.locals[lifted_fn.params[1].0].name.as_deref(),
        Some("x")
    );
    assert_eq!(lifted_fn.signature.return_type, MirType::Int);

    // 4c. Verify `cleanup_blocks` has `MirCleanupKind::ErrDefer` pointing to a cleanup basic block.
    let errdefer_entry = main_fn
        .cleanup_blocks
        .iter()
        .find(|c| c.kind == MirCleanupKind::ErrDefer)
        .expect("main_fn.cleanup_blocks must record MirCleanupKind::ErrDefer");
    assert!(
        main_fn.blocks[errdefer_entry.block.0].is_cleanup,
        "cleanup block must be marked is_cleanup = true"
    );

    // 4d. Verify fail-closed behavior in VM backend on `Rvalue::Closure`.
    let vm_err = MirToBytecodeCompiler::compile(main_fn)
        .expect_err("VM backend must fail closed on Rvalue::Closure");
    assert_eq!(
        vm_err,
        MirVmBackendError::UnsupportedRvalue("Rvalue::Closure".to_string())
    );

    // 4e. Also verify expression-bodied `ExprKind::Lambda` (`fn(x: int): int => x + offset`) directly.
    let span = Span::dummy();
    let expr_lambda = Expr::new(
        ExprKind::Lambda {
            params: vec![("x".to_string(), Some(Type::new(TypeKind::Int, span)))],
            return_type: Some(Type::new(TypeKind::Int, span)),
            body: Box::new(Expr::new(
                ExprKind::Binary {
                    left: Box::new(Expr::new(ExprKind::Identifier("x".to_string()), span)),
                    op: BinaryOp::Add,
                    right: Box::new(Expr::new(ExprKind::Identifier("offset".to_string()), span)),
                },
                span,
            )),
        },
        span,
    );
    let wrapper_item = Item::new(
        ItemKind::Function(FunctionDef {
            name: "expr_lambda_host".to_string(),
            type_params: vec![],
            trait_bounds: vec![],
            params: vec![FunctionParam {
                name: "offset".to_string(),
                ty: Type::new(TypeKind::Int, span),
                is_self: false,
            }],
            return_type: Some(Type::new(TypeKind::Int, span)),
            body: vec![
                Stmt::new(
                    StmtKind::Local {
                        bindings: vec![lust::ast::LocalBinding {
                            name: "f".to_string(),
                            type_annotation: None,
                            span,
                        }],
                        initializer: Some(vec![expr_lambda]),
                    },
                    span,
                ),
                Stmt::new(
                    StmtKind::Return(vec![Expr::new(
                        ExprKind::Literal(Literal::Integer(0)),
                        span,
                    )]),
                    span,
                ),
            ],
            is_method: false,
            visibility: Visibility::Public,
            doc: None,
            effects: None,
        }),
        span,
    );
    let expr_mod = AstLowering::lower_module_items("expr_lambda_mod", &[wrapper_item])
        .expect("lower expression-bodied lambda");
    MirVerifier::verify_module(&expr_mod).expect("verify expression-bodied lambda module");
    assert!(
        expr_mod
            .functions
            .iter()
            .any(|f| f.signature.name == "__lust_lambda_0"),
        "expression-bodied lambda must lift to __lust_lambda_0"
    );
}

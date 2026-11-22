//! Wave 1 Track 1F focused Cranelift type parity checks.
//! Comprehensive verification of Float arithmetic/comparisons/casts,
//! compound values (String, Array, Tuple) marshaling, in-place mutation,
//! zero-leak handle ownership cleanup, and Bytecode interpreter oracle parity.

use std::rc::Rc;
use lust::bytecode::{LustMap, Value, ValueKey};
use lust::bytecode::compiler::Compiler;
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::ir::MirModule;
use lust::mir::lowering::AstLowering;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{VM, VmBackend};

fn compile_module_from_source(src: &str) -> (MirModule, Vec<lust::ast::Item>) {
    let mut interner = Interner::new();
    let tokens = Lexer::new(src, &mut interner)
        .tokenize()
        .expect("tokenization should succeed");
    let items = Parser::new(tokens).parse().expect("parsing should succeed");
    let mut module = MirModule::new("parity_suite".into());
    for item in &items {
        if let lust::ast::ItemKind::Function(func) = &item.kind {
            let mir_fn = AstLowering::lower_function(func)
                .expect("MIR lowering should succeed");
            module.add_function(mir_fn);
        }
    }
    (module, items)
}

fn create_vms(src: &str) -> (VM, VM) {
    let (module, items) = compile_module_from_source(src);

    let mut vm_clif = VM::new();
    vm_clif.load_mir_module(&module).unwrap();
    vm_clif.set_backend(VmBackend::Cranelift);

    let mut vm_bc = VM::new();
    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let struct_defs = checker.struct_definitions();
    vm_clif.register_structs(&struct_defs);
    vm_bc.register_structs(&struct_defs);
    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    if let Ok(bc_functions) = compiler.compile_module(&items) {
        vm_bc.load_functions(bc_functions);
    } else {
        vm_bc.load_mir_module(&module).unwrap();
    }
    vm_bc.set_backend(VmBackend::Bytecode);

    (vm_clif, vm_bc)
}

#[test]
fn float_constants_compile_to_cranelift_f64() {
    let src = "function f(): float return 1.25 end";
    let (mut vm_clif, mut vm_bc) = create_vms(src);

    let res_clif = vm_clif.call("f", vec![]).unwrap();
    let res_bc = vm_bc.call("f", vec![]).unwrap();

    assert_eq!(res_clif, Value::Float(1.25));
    assert_eq!(res_clif, res_bc);
    assert!(
        vm_clif.cranelift_entry_count > 0,
        "Cranelift JIT dispatch must execute natively, not fallback"
    );
}

#[test]
fn float_arithmetic_operations_parity() {
    let src = r#"
        function add_f(a: float, b: float): float return a + b end
        function sub_f(a: float, b: float): float return a - b end
        function mul_f(a: float, b: float): float return a * b end
        function div_f(a: float, b: float): float return a / b end
        function floordiv_f(a: float, b: float): float return a // b end
        function rem_f(a: float, b: float): float return a % b end
        function pow_f(a: float, b: float): float return a ^ b end
        function neg_f(a: float): float return -a end
    "#;
    let (mut vm_clif, mut vm_bc) = create_vms(src);

    let cases = vec![
        ("add_f", vec![Value::Float(3.5), Value::Float(1.25)], Value::Float(4.75)),
        ("sub_f", vec![Value::Float(3.5), Value::Float(1.25)], Value::Float(2.25)),
        ("mul_f", vec![Value::Float(2.5), Value::Float(4.0)], Value::Float(10.0)),
        ("div_f", vec![Value::Float(7.5), Value::Float(2.5)], Value::Float(3.0)),
        ("floordiv_f", vec![Value::Float(7.8), Value::Float(2.5)], Value::Float(3.0)),
        ("rem_f", vec![Value::Float(7.5), Value::Float(2.0)], Value::Float(1.5)),
        ("pow_f", vec![Value::Float(2.0), Value::Float(3.0)], Value::Float(8.0)),
        ("neg_f", vec![Value::Float(42.5)], Value::Float(-42.5)),
    ];

    for (func, args, expected) in cases {
        let entries_before = vm_clif.cranelift_entry_count;
        let res_clif = vm_clif.call(func, args.clone()).unwrap();
        let res_bc = vm_bc.call(func, args).unwrap();

        assert_eq!(res_clif, expected, "Func {} failed", func);
        assert_eq!(res_clif, res_bc, "Oracle mismatch for {}", func);
        assert!(
            vm_clif.cranelift_entry_count > entries_before,
            "Func {} must execute via Cranelift JIT",
            func
        );
    }
}

#[test]
fn float_comparison_parity() {
    let src = r#"
        function lt_f(a: float, b: float): bool return a < b end
        function lte_f(a: float, b: float): bool return a <= b end
        function gt_f(a: float, b: float): bool return a > b end
        function gte_f(a: float, b: float): bool return a >= b end
        function eq_f(a: float, b: float): bool return a == b end
        function neq_f(a: float, b: float): bool return a != b end
    "#;
    let (mut vm_clif, mut vm_bc) = create_vms(src);

    let test_triples = vec![
        (1.5, 2.5),
        (2.5, 2.5),
        (3.5, 2.5),
    ];

    for (x, y) in test_triples {
        for op in &["lt_f", "lte_f", "gt_f", "gte_f", "eq_f", "neq_f"] {
            let args = vec![Value::Float(x), Value::Float(y)];
            let entries_before = vm_clif.cranelift_entry_count;
            let res_clif = vm_clif.call(op, args.clone()).unwrap();
            let res_bc = vm_bc.call(op, args).unwrap();

            assert_eq!(res_clif, res_bc, "Comparison mismatch for {} on ({}, {})", op, x, y);
            assert!(
                vm_clif.cranelift_entry_count > entries_before,
                "Comparison {} must execute via Cranelift",
                op
            );
        }
    }
}

#[test]
fn float_type_casting_parity() {
    use lust::ast::Span;
    use lust::mir::ir::{FunctionSignature, MirFunction, MirType, Operand, Place, Rvalue, Statement, Terminator};

    fn make_cast_fn(name: &str, from_ty: MirType, to_ty: MirType) -> MirFunction {
        let sig = FunctionSignature {
            name: name.to_string(),
            param_types: vec![from_ty],
            return_type: to_ty.clone(),
            is_public: true,
            effects: None,
        };
        let mut f = MirFunction::new(sig, Span::dummy());
        let ret = f.return_local;
        let a = f.params[0];
        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(ret),
            Rvalue::Cast(Operand::Copy(Place::Local(a)), to_ty),
            Span::dummy(),
        ));
        f.blocks[0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(ret))),
            Span::dummy(),
        ));
        f
    }

    let mut vm_clif = VM::new();
    vm_clif.load_mir_function(make_cast_fn("int_to_float", MirType::Int, MirType::Float)).unwrap();
    vm_clif.load_mir_function(make_cast_fn("float_to_int", MirType::Float, MirType::Int)).unwrap();
    vm_clif.set_backend(VmBackend::Cranelift);

    let res_f = vm_clif.call("int_to_float", vec![Value::Int(42)]).unwrap();
    assert_eq!(res_f, Value::Float(42.0));
    assert!(
        vm_clif.cranelift_entry_count > 0,
        "Cranelift JIT must execute int_to_float"
    );

    let entries_before = vm_clif.cranelift_entry_count;
    let res_i = vm_clif.call("float_to_int", vec![Value::Float(42.75)]).unwrap();
    assert_eq!(res_i, Value::Int(42));
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "Cranelift JIT must execute float_to_int"
    );
}

#[test]
fn string_compound_marshaling_and_operations() {
    let src = r#"
        function concat_str(a: string, b: string): string return a + b end
        function eq_str(a: string, b: string): bool return a == b end
        function neq_str(a: string, b: string): bool return a != b end
        function const_str(): string return "lust-native" end
    "#;
    let (mut vm_clif, mut vm_bc) = create_vms(src);

    // Concat (Cranelift native string concatenation via helper)
    let entries_before = vm_clif.cranelift_entry_count;
    let args = vec![Value::string("Hello, "), Value::string("Cranelift!")];
    let res_clif = vm_clif.call("concat_str", args).unwrap();
    assert_eq!(res_clif, Value::string("Hello, Cranelift!"));
    assert!(vm_clif.cranelift_entry_count > entries_before);

    // Equality
    let eq_res = vm_clif.call("eq_str", vec![Value::string("abc"), Value::string("abc")]).unwrap();
    let eq_res_bc = vm_bc.call("eq_str", vec![Value::string("abc"), Value::string("abc")]).unwrap();
    assert_eq!(eq_res, Value::Bool(true));
    assert_eq!(eq_res, eq_res_bc);

    let neq_res = vm_clif.call("eq_str", vec![Value::string("abc"), Value::string("xyz")]).unwrap();
    let neq_res_bc = vm_bc.call("eq_str", vec![Value::string("abc"), Value::string("xyz")]).unwrap();
    assert_eq!(neq_res, Value::Bool(false));
    assert_eq!(neq_res, neq_res_bc);

    let neq_res2 = vm_clif.call("neq_str", vec![Value::string("abc"), Value::string("xyz")]).unwrap();
    let neq_res2_bc = vm_bc.call("neq_str", vec![Value::string("abc"), Value::string("xyz")]).unwrap();
    assert_eq!(neq_res2, Value::Bool(true));
    assert_eq!(neq_res2, neq_res2_bc);

    // String constant return
    let const_res = vm_clif.call("const_str", vec![]).unwrap();
    let const_res_bc = vm_bc.call("const_str", vec![]).unwrap();
    assert_eq!(const_res, Value::string("lust-native"));
    assert_eq!(const_res, const_res_bc);
}

#[test]
fn array_marshaling_indexing_and_mutation() {
    let src = r#"
        function sum_int_array(arr: Array<int>): int
            return arr[0] + arr[1]
        end

        function mutate_first(arr: Array<int>): int
            arr[0] = 999
            return arr[0]
        end

        function make_array(): Array<int>
            return [10, 20, 30]
        end

        function sum_float_array(arr: Array<float>): float
            return arr[0] + arr[1]
        end
    "#;
    let (mut vm_clif, mut vm_bc) = create_vms(src);

    // Reading array
    let arr_val = Value::array(vec![Value::Int(15), Value::Int(27)]);
    let sum_res = vm_clif.call("sum_int_array", vec![arr_val]).unwrap();
    assert_eq!(sum_res, Value::Int(42));

    // Float array
    let float_arr = Value::array(vec![Value::Float(1.5), Value::Float(2.5)]);
    let float_sum = vm_clif.call("sum_float_array", vec![float_arr]).unwrap();
    assert_eq!(float_sum, Value::Float(4.0));

    // In-place mutation
    let orig_cell = Rc::new(lust::bytecode::LustArray::new(vec![Value::Int(1), Value::Int(2)]));
    let mut_arg = Value::Array(orig_cell.clone());
    let mut_res = vm_clif.call("mutate_first", vec![mut_arg]).unwrap();
    assert_eq!(mut_res, Value::Int(999));
    assert_eq!(orig_cell.borrow()[0], Value::Int(999), "Array must be mutated in-place");

    // Returning array
    let ret_arr = vm_clif.call("make_array", vec![]).unwrap();
    let ret_arr_bc = vm_bc.call("make_array", vec![]).unwrap();
    assert_eq!(ret_arr, Value::array(vec![Value::Int(10), Value::Int(20), Value::Int(30)]));
    assert_eq!(ret_arr, ret_arr_bc);
}

#[test]
fn tuple_and_compound_marshaling() {
    let src = r#"
        function sum_tuple(t: (int, int)): int
            return t[0] + t[1]
        end

        function make_tuple(): (int, int)
            return (7, 11)
        end
    "#;
    let (mut vm_clif, mut vm_bc) = create_vms(src);

    let tuple_val = Value::Tuple(Rc::new(vec![Value::Int(7), Value::Int(11)]));
    let sum_res = vm_clif.call("sum_tuple", vec![tuple_val]).unwrap();
    assert_eq!(sum_res, Value::Int(18));

    let ret_tuple = vm_clif.call("make_tuple", vec![]).unwrap();
    let ret_tuple_bc = vm_bc.call("make_tuple", vec![]).unwrap();
    assert_eq!(ret_tuple, Value::Tuple(Rc::new(vec![Value::Int(7), Value::Int(11)])));
    assert_eq!(ret_tuple, ret_tuple_bc);
}

#[test]
fn ownership_cleanup_zero_leak() {
    let src = r#"
        function concat_str(a: string, b: string): string return a + b end
        function div_int(a: int, b: int): int return a / b end
        function div_float(a: float, b: float): float return a // b end
    "#;
    let (mut vm_clif, _) = create_vms(src);

    let baseline_handles = vm_clif.mir_context.handle_count();

    // 1. Successful execution loop - no handle leakage
    for i in 0..50 {
        let s1 = Value::string(format!("str_{}", i));
        let s2 = Value::string(format!("_val_{}", i));
        let _ = vm_clif.call("concat_str", vec![s1, s2]).unwrap();
        assert_eq!(
            vm_clif.mir_context.handle_count(),
            baseline_handles,
            "Iteration {} leaked handles!",
            i
        );
    }

    // 2. Division by zero in integer - error path must clean handles
    let err_i = vm_clif.call("div_int", vec![Value::Int(10), Value::Int(0)]);
    assert!(err_i.is_err(), "Division by zero should return error");
    assert_eq!(
        vm_clif.mir_context.handle_count(),
        baseline_handles,
        "Error path must restore handle count to baseline"
    );

    // 3. Division by zero in float floordiv - error path must clean handles
    let err_f = vm_clif.call("div_float", vec![Value::Float(10.0), Value::Float(0.0)]);
    assert!(err_f.is_err(), "Float floordiv by zero should return error");
    assert_eq!(
        vm_clif.mir_context.handle_count(),
        baseline_handles,
        "Float error path must restore handle count to baseline"
    );
}

#[test]
fn gas_budget_and_oom_cleanup() {
    let src = r#"
        function looping_concat(a: string, b: string): string
            return a + b
        end
    "#;
    let (mut vm_clif, _) = create_vms(src);
    let baseline_handles = vm_clif.mir_context.handle_count();

    // Configure tiny gas budget
    vm_clif.set_gas_budget(1);

    let s1 = Value::string("Alpha");
    let s2 = Value::string("Beta");
    let res = vm_clif.call("looping_concat", vec![s1, s2]);
    // May succeed on 1 gas or fail with OutOfGas, but either way handle count must be zero
    let _ = res;
    assert_eq!(
        vm_clif.mir_context.handle_count(),
        baseline_handles,
        "Handles must be clean regardless of gas budget result"
    );
}

#[test]
fn struct_marshaling_passing_and_returning_parity() {
    let src = r#"
        struct Point
            x: int,
            y: int
        end

        function id_point(p: Point): Point
            return p
        end

        function make_point(a: int, b: int): Point
            return Point { x = a, y = b }
        end

        function sum_point(p: Point): int
            local dummy = Point { x = 0, y = 0 }
            return p.x + p.y
        end

        function mutate_point(p: Point): int
            local dummy = Point { x = 0, y = 0 }
            p.x = 999
            return p.x
        end
    "#;
    let (mut vm_clif, mut vm_bc) = create_vms(src);

    let pt_clif = vm_clif
        .instantiate_struct_ordered("Point", vec![Value::Int(10), Value::Int(20)])
        .unwrap();
    let pt_bc = vm_bc
        .instantiate_struct_ordered("Point", vec![Value::Int(10), Value::Int(20)])
        .unwrap();

    // 1. Pass and return Struct
    let entries_before = vm_clif.cranelift_entry_count;
    let res_clif = vm_clif.call("id_point", vec![pt_clif.clone()]).unwrap();
    let res_bc = vm_bc.call("id_point", vec![pt_bc.clone()]).unwrap();

    assert_eq!(res_clif, res_bc, "Struct id_point mismatch");
    assert_eq!(res_clif, pt_clif);
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "id_point must execute via Cranelift JIT"
    );

    // 2. Make struct and return
    let entries_before = vm_clif.cranelift_entry_count;
    let make_clif = vm_clif.call("make_point", vec![Value::Int(15), Value::Int(27)]).unwrap();
    let make_bc = vm_bc.call("make_point", vec![Value::Int(15), Value::Int(27)]).unwrap();

    assert_eq!(make_clif, make_bc, "make_point result mismatch");
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "make_point must execute via Cranelift JIT"
    );

    // 3. Field access and summation
    let entries_before = vm_clif.cranelift_entry_count;
    let sum_clif = vm_clif.call("sum_point", vec![pt_clif.clone()]).unwrap();
    let sum_bc = vm_bc.call("sum_point", vec![pt_bc.clone()]).unwrap();

    assert_eq!(sum_clif, Value::Int(30));
    assert_eq!(sum_clif, sum_bc);
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "sum_point must execute via Cranelift JIT"
    );

    // 4. In-place struct mutation
    let orig_pt = vm_clif
        .instantiate_struct_ordered("Point", vec![Value::Int(1), Value::Int(2)])
        .unwrap();
    let entries_before = vm_clif.cranelift_entry_count;
    let mut_res = vm_clif.call("mutate_point", vec![orig_pt.clone()]).unwrap();
    assert_eq!(mut_res, Value::Int(999));
    if let Value::Struct(ref s) = orig_pt {
        assert_eq!(s.fields.borrow()[0], Value::Int(999), "Struct must be mutated in-place");
    } else {
        panic!("Expected Value::Struct");
    }
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "mutate_point must execute via Cranelift JIT"
    );
}

#[test]
fn struct_float_fields_parity() {
    let src = r#"
        struct Vec2
            x: float,
            y: float
        end

        function id_vec(v: Vec2): Vec2
            return v
        end

        function make_vec(x: float, y: float): Vec2
            return Vec2 { x = x, y = y }
        end
    "#;
    let (mut vm_clif, mut vm_bc) = create_vms(src);

    let v_clif = vm_clif
        .instantiate_struct_ordered("Vec2", vec![Value::Float(1.5), Value::Float(2.5)])
        .unwrap();
    let v_bc = vm_bc
        .instantiate_struct_ordered("Vec2", vec![Value::Float(1.5), Value::Float(2.5)])
        .unwrap();

    let entries_before = vm_clif.cranelift_entry_count;
    let res_clif = vm_clif.call("id_vec", vec![v_clif.clone()]).unwrap();
    let res_bc = vm_bc.call("id_vec", vec![v_bc]).unwrap();

    assert_eq!(res_clif, res_bc, "Float struct id_vec mismatch");
    assert_eq!(res_clif, v_clif);
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "id_vec must execute via Cranelift JIT"
    );

    let entries_before = vm_clif.cranelift_entry_count;
    let make_clif = vm_clif
        .call("make_vec", vec![Value::Float(3.25), Value::Float(4.75)])
        .unwrap();
    let make_bc = vm_bc
        .call("make_vec", vec![Value::Float(3.25), Value::Float(4.75)])
        .unwrap();

    assert_eq!(make_clif, make_bc, "make_vec mismatch");
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "make_vec must execute via Cranelift JIT"
    );
}

#[test]
fn map_marshaling_passing_and_returning_parity() {
    let src = r#"
        function id_map(m: Map<string, int>): Map<string, int>
            return m
        end

        function map_access(m: Map<string, int>, k: string): Option<int>
            return map.get(m, k)
        end

        function map_mutate(m: Map<string, int>, k: string, v: int): Option<int>
            map.set(m, k, v)
            return map.get(m, k)
        end

        function map_size(m: Map<string, int>): int
            return map.len(m)
        end
    "#;
    let (mut vm_clif, mut vm_bc) = create_vms(src);

    let mut m1 = LustMap::default();
    m1.insert(ValueKey::from(Value::string("alpha")), Value::Int(10));
    m1.insert(ValueKey::from(Value::string("beta")), Value::Int(20));
    let map_val_clif = Value::map(m1.clone());
    let map_val_bc = Value::map(m1);

    // 1. Pass and return Map
    let entries_before = vm_clif.cranelift_entry_count;
    let res_clif = vm_clif.call("id_map", vec![map_val_clif.clone()]).unwrap();
    let res_bc = vm_bc.call("id_map", vec![map_val_bc.clone()]).unwrap();

    assert_eq!(res_clif, res_bc, "id_map mismatch");
    assert_eq!(res_clif, map_val_clif);
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "id_map must execute via Cranelift JIT"
    );

    // 2. Map get access
    let entries_before = vm_clif.cranelift_entry_count;
    let val_clif = vm_clif
        .call("map_access", vec![map_val_clif.clone(), Value::string("alpha")])
        .unwrap();
    let val_bc = vm_bc
        .call("map_access", vec![map_val_bc.clone(), Value::string("alpha")])
        .unwrap();

    assert_eq!(val_clif, Value::some(Value::Int(10)));
    assert_eq!(val_clif, val_bc);
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "map_access must execute via Cranelift JIT"
    );

    let missing_clif = vm_clif
        .call("map_access", vec![map_val_clif.clone(), Value::string("nonexistent")])
        .unwrap();
    let missing_bc = vm_bc
        .call("map_access", vec![map_val_bc.clone(), Value::string("nonexistent")])
        .unwrap();
    assert_eq!(missing_clif, Value::none());
    assert_eq!(missing_clif, missing_bc);

    // 3. Map size
    let entries_before = vm_clif.cranelift_entry_count;
    let size_clif = vm_clif.call("map_size", vec![map_val_clif.clone()]).unwrap();
    let size_bc = vm_bc.call("map_size", vec![map_val_bc.clone()]).unwrap();

    assert_eq!(size_clif, Value::Int(2));
    assert_eq!(size_clif, size_bc);
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "map_size must execute via Cranelift JIT"
    );

    // 4. Map mutation
    let entries_before = vm_clif.cranelift_entry_count;
    let mut_res_clif = vm_clif
        .call("map_mutate", vec![map_val_clif.clone(), Value::string("gamma"), Value::Int(30)])
        .unwrap();
    let mut_res_bc = vm_bc
        .call("map_mutate", vec![map_val_bc.clone(), Value::string("gamma"), Value::Int(30)])
        .unwrap();
    assert_eq!(mut_res_clif, Value::some(Value::Int(30)));
    assert_eq!(mut_res_clif, mut_res_bc);
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "map_mutate must execute via Cranelift JIT"
    );
    if let Value::Map(ref m) = map_val_clif {
        assert_eq!(
            m.borrow().get(&ValueKey::from(Value::string("gamma"))),
            Some(&Value::Int(30)),
            "Map must be mutated in-place"
        );
    } else {
        panic!("Expected Value::Map");
    }
}

#[test]
fn enum_marshaling_passing_and_returning_parity() {
    use lust::ast::Span;
    use lust::mir::ir::{FunctionSignature, MirFunction, MirType, Operand, Place, Terminator};

    let src = r#"
        function id_opt(x: Option<int>): Option<int>
            return x
        end

        function id_res(x: Result<int, string>): Result<int, string>
            return x
        end

        function make_some(x: int): Option<int>
            return Some(x)
        end

        function make_none(): Option<int>
            return Option.None()
        end

        function make_ok(x: int): Result<int, string>
            return Ok(x)
        end
    "#;
    let (mut vm_clif, mut vm_bc) = create_vms(src);

    // 1. Pass and return Option::Some
    let some_val = Value::some(Value::Int(42));
    let entries_before = vm_clif.cranelift_entry_count;
    let res_clif = vm_clif.call("id_opt", vec![some_val.clone()]).unwrap();
    let res_bc = vm_bc.call("id_opt", vec![some_val.clone()]).unwrap();

    assert_eq!(res_clif, some_val);
    assert_eq!(res_clif, res_bc);
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "id_opt must execute via Cranelift JIT"
    );

    // 2. Pass and return Option::None
    let none_val = Value::none();
    let entries_before = vm_clif.cranelift_entry_count;
    let res_clif = vm_clif.call("id_opt", vec![none_val.clone()]).unwrap();
    let res_bc = vm_bc.call("id_opt", vec![none_val.clone()]).unwrap();

    assert_eq!(res_clif, none_val);
    assert_eq!(res_clif, res_bc);
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "id_opt (None) must execute via Cranelift JIT"
    );

    // 3. Pass and return Result::Ok
    let ok_val = Value::ok(Value::Int(100));
    let entries_before = vm_clif.cranelift_entry_count;
    let res_clif = vm_clif.call("id_res", vec![ok_val.clone()]).unwrap();
    let res_bc = vm_bc.call("id_res", vec![ok_val.clone()]).unwrap();

    assert_eq!(res_clif, ok_val);
    assert_eq!(res_clif, res_bc);
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "id_res (Ok) must execute via Cranelift JIT"
    );

    // 4. Pass and return Result::Err
    let err_val = Value::err(Value::string("failure"));
    let entries_before = vm_clif.cranelift_entry_count;
    let res_clif = vm_clif.call("id_res", vec![err_val.clone()]).unwrap();
    let res_bc = vm_bc.call("id_res", vec![err_val.clone()]).unwrap();

    assert_eq!(res_clif, err_val);
    assert_eq!(res_clif, res_bc);
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "id_res (Err) must execute via Cranelift JIT"
    );

    // 5. Native construct Some, None, Ok
    let some_res = vm_clif.call("make_some", vec![Value::Int(77)]).unwrap();
    let some_bc = vm_bc.call("make_some", vec![Value::Int(77)]).unwrap();
    assert_eq!(some_res, Value::some(Value::Int(77)));
    assert_eq!(some_res, some_bc);

    let none_res = vm_clif.call("make_none", vec![]).unwrap();
    let none_bc = vm_bc.call("make_none", vec![]).unwrap();
    assert_eq!(none_res, Value::none());
    assert_eq!(none_res, none_bc);

    let ok_res = vm_clif.call("make_ok", vec![Value::Int(88)]).unwrap();
    let ok_bc = vm_bc.call("make_ok", vec![Value::Int(88)]).unwrap();
    assert_eq!(ok_res, Value::ok(Value::Int(88)));
    assert_eq!(ok_res, ok_bc);

    // 6. MIR functions calling Cranelift builtins: enum.tag and enum.payload
    let tag_fn = {
        let sig = FunctionSignature {
            name: "mir_enum_tag".to_string(),
            param_types: vec![MirType::Enum("Option".into())],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut f = MirFunction::new(sig, Span::dummy());
        let ret = f.return_local;
        let a = f.params[0];
        let succ = f.new_block();
        f.blocks[0].terminator = Some(Terminator::Call {
            callee: "enum.tag".to_string(),
            args: vec![Operand::Copy(Place::Local(a))],
            destination: Place::Local(ret),
            success: succ,
            error: None,
            span: Span::dummy(),
        });
        f.blocks[succ.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(ret))),
            Span::dummy(),
        ));
        f
    };

    let payload_fn = {
        let sig = FunctionSignature {
            name: "mir_enum_payload".to_string(),
            param_types: vec![MirType::Enum("Option".into())],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut f = MirFunction::new(sig, Span::dummy());
        let ret = f.return_local;
        let a = f.params[0];
        let succ = f.new_block();
        f.blocks[0].terminator = Some(Terminator::Call {
            callee: "enum.payload".to_string(),
            args: vec![Operand::Copy(Place::Local(a))],
            destination: Place::Local(ret),
            success: succ,
            error: None,
            span: Span::dummy(),
        });
        f.blocks[succ.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(ret))),
            Span::dummy(),
        ));
        f
    };

    vm_clif.load_mir_function(tag_fn).unwrap();
    vm_clif.load_mir_function(payload_fn).unwrap();

    let entries_before = vm_clif.cranelift_entry_count;
    let tag_some = vm_clif.call("mir_enum_tag", vec![Value::some(Value::Int(55))]).unwrap();
    assert_eq!(tag_some, Value::Int(1));
    assert!(vm_clif.cranelift_entry_count > entries_before);

    let entries_before = vm_clif.cranelift_entry_count;
    let tag_none = vm_clif.call("mir_enum_tag", vec![Value::none()]).unwrap();
    assert_eq!(tag_none, Value::Int(0));
    assert!(vm_clif.cranelift_entry_count > entries_before);

    let entries_before = vm_clif.cranelift_entry_count;
    let payload_res = vm_clif.call("mir_enum_payload", vec![Value::some(Value::Int(55))]).unwrap();
    assert_eq!(payload_res, Value::Int(55));
    assert!(vm_clif.cranelift_entry_count > entries_before);
}

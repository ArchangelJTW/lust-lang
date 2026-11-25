//! Acceptance test suite for Wave 0 Track 0F: J-CLIF-INT
//!
//! Verifies:
//! 1. Wiring of `CraneliftJitCompiler` into `VM::run` via `VmBackend::Cranelift`.
//! 2. Acceptance scalar suite executed through actual native Cranelift entry.
//! 3. Runtime execution path & counters (`cranelift_entry_count`, `gas_remaining`, `cranelift_compilation_count`).
//! 4. 100% Interpreter Oracle Parity between `VmBackend::Bytecode` and `VmBackend::Cranelift`.
//! 5. Baseline path remains completely unchanged when `VmBackend::Bytecode` is active.

use lust::bytecode::Value;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::ir::MirModule;
use lust::mir::lowering::AstLowering;
use lust::parser::Parser;
use lust::vm::{VM, VmBackend};
use std::sync::Mutex;

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn parse_and_lower_module(source: &str, module_name: &str) -> MirModule {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("Tokenization must succeed");
    let items = Parser::new(tokens).parse().expect("Parsing must succeed");

    let mut module = MirModule::new(module_name.into());
    for item in items {
        if let lust::ast::ItemKind::Function(func_def) = item.kind {
            let mir_fn = AstLowering::lower_function(&func_def).expect("MIR lowering must succeed");
            module.add_function(mir_fn);
        }
    }
    module
}

#[test]
fn test_scalar_arithmetic_parity_and_entry_counters() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let source = r#"
        function compute_arithmetic(a: int, b: int): int
            local sum: int = a + b
            local diff: int = a - b
            local prod: int = a * b
            local quot: int = a / b
            local rem: int = a % b
            local fdiv: int = a // b
            local p: int = a ^ b
            return sum + diff + prod + quot + rem + fdiv + p
        end
    "#;

    let module = parse_and_lower_module(source, "test_arithmetic");
    let mut vm = VM::new();
    vm.load_mir_module(&module).expect("Load MIR module");

    // 1. Cranelift JIT Mode
    vm.set_backend(VmBackend::Cranelift);
    assert_eq!(vm.backend(), VmBackend::Cranelift);

    let gas_before = vm.mir_context.gas_remaining();
    let entries_before = vm.cranelift_entry_count;

    let args = vec![Value::Int(7), Value::Int(3)];
    let clif_result = vm.call("compute_arithmetic", args).expect("Cranelift execution");

    let entries_after = vm.cranelift_entry_count;
    let gas_after = vm.mir_context.gas_remaining();

    // Verify runtime path and counters
    assert_eq!(entries_after, entries_before + 1, "Cranelift entry counter must increment by 1");
    assert!(gas_after < gas_before, "Gas must be charged during Cranelift execution");

    // Expected:
    // sum = 10, diff = 4, prod = 21, quot = 2, rem = 1, fdiv = 2, p = 7^3 = 343
    // total = 10 + 4 + 21 + 2 + 1 + 2 + 343 = 383
    assert_eq!(clif_result, Value::Int(383));

    // 2. Baseline Interpreter Oracle Parity
    vm.set_backend(VmBackend::Bytecode);
    assert_eq!(vm.backend(), VmBackend::Bytecode);

    let baseline_entries = vm.cranelift_entry_count;
    let interp_result = vm.call("compute_arithmetic", vec![Value::Int(7), Value::Int(3)])
        .expect("Interpreter execution");

    assert_eq!(vm.cranelift_entry_count, baseline_entries, "Baseline interpreter must not increment Cranelift entry counter");
    assert_eq!(clif_result, interp_result, "Cranelift result must match Interpreter Oracle parity 100%");
}

#[test]
fn test_scalar_bitwise_parity_and_entry_counters() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let source = r#"
        function compute_bitwise(a: int, b: int): int
            local band_val: int = a & b
            local bor_val: int = a | b
            local bxor_val: int = a ^^ b
            local bnot_val: int = ~a
            local shl_val: int = a << 2
            local shr_val: int = a >> 1
            return band_val + bor_val + bxor_val + bnot_val + shl_val + shr_val
        end
    "#;

    let module = parse_and_lower_module(source, "test_bitwise");
    let mut vm = VM::new();
    vm.load_mir_module(&module).expect("Load MIR module");

    // Test Cranelift execution
    vm.set_backend(VmBackend::Cranelift);
    let clif_res = vm.call("compute_bitwise", vec![Value::Int(42), Value::Int(15)])
        .expect("Cranelift bitwise execution");

    // Test Baseline Bytecode VM Oracle
    vm.set_backend(VmBackend::Bytecode);
    let vm_res = vm.call("compute_bitwise", vec![Value::Int(42), Value::Int(15)])
        .expect("Bytecode bitwise execution");

    assert_eq!(clif_res, vm_res, "Bitwise parity check failed");
}

#[test]
fn test_scalar_comparisons_and_branching_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let source = r#"
        function classify_value(x: int, y: int): int
            if x == y then
                return 100
            elseif x > y and x <= 50 then
                return 200 + x
            elseif x < y or y == 0 then
                return 300 + y
            else
                return 400
            end
        end
    "#;

    let module = parse_and_lower_module(source, "test_branching");
    let mut vm = VM::new();
    vm.load_mir_module(&module).expect("Load MIR module");

    let test_cases = [
        (10, 10), // equal -> 100
        (25, 10), // x > y and x <= 50 -> 225
        (5, 20),  // x < y -> 320
        (60, 10), // else -> 400
    ];

    for (x, y) in test_cases {
        vm.set_backend(VmBackend::Cranelift);
        let clif = vm.call("classify_value", vec![Value::Int(x), Value::Int(y)]).unwrap();

        vm.set_backend(VmBackend::Bytecode);
        let interp = vm.call("classify_value", vec![Value::Int(x), Value::Int(y)]).unwrap();

        assert_eq!(clif, interp, "Parity check failed for classify_value({}, {})", x, y);
    }
}

#[test]
fn test_scalar_loops_break_continue_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let source = r#"
        function loop_scalar(limit: int): int
            local acc: int = 0
            for i = 1, limit do
                if i == 3 then
                    continue
                end
                if i > 7 then
                    break
                end
                acc = acc + (i * 10)
            end
            return acc
        end
    "#;

    let module = parse_and_lower_module(source, "test_loop");
    let mut vm = VM::new();
    vm.load_mir_module(&module).expect("Load MIR module");

    // Cranelift
    vm.set_backend(VmBackend::Cranelift);
    let clif_val = vm.call("loop_scalar", vec![Value::Int(10)]).expect("Cranelift loop");

    // Interpreter Oracle
    vm.set_backend(VmBackend::Bytecode);
    let interp_val = vm.call("loop_scalar", vec![Value::Int(10)]).expect("Interpreter loop");

    // i values added: 1 (10), 2 (20), 4 (40), 5 (50), 6 (60), 7 (70) = 250
    assert_eq!(clif_val, Value::Int(250));
    assert_eq!(clif_val, interp_val, "Loop with break and continue parity failed");
}

#[test]
fn test_scalar_math_builtins_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let source = r#"
        function test_math_intrinsics(val: int, low: int, high: int): int
            local a: int = math.abs(val)
            local m1: int = math.min(a, high)
            local m2: int = math.max(m1, low)
            local c: int = math.clamp(val, low, high)
            return a + m1 + m2 + c
        end
    "#;

    let module = parse_and_lower_module(source, "test_math");
    let mut vm = VM::new();
    vm.load_mir_module(&module).expect("Load MIR module");

    let test_inputs = [
        (-25, 0, 100),
        (50, 10, 30),
        (-5, -20, -10),
    ];

    for (val, low, high) in test_inputs {
        vm.set_backend(VmBackend::Cranelift);
        let clif = vm.call("test_math_intrinsics", vec![Value::Int(val), Value::Int(low), Value::Int(high)]).unwrap();

        vm.set_backend(VmBackend::Bytecode);
        let interp = vm.call("test_math_intrinsics", vec![Value::Int(val), Value::Int(low), Value::Int(high)]).unwrap();

        assert_eq!(clif, interp, "Math intrinsics parity failed for ({}, {}, {})", val, low, high);
    }
}

#[test]
fn test_scalar_multi_function_calls_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let source = r#"
        function square(x: int): int
            return x * x
        end

        function sum_of_squares(a: int, b: int): int
            return square(a) + square(b)
        end
    "#;

    let module = parse_and_lower_module(source, "test_multi_fn");
    let mut vm = VM::new();
    vm.load_mir_module(&module).expect("Load MIR module");

    // Cranelift
    vm.set_backend(VmBackend::Cranelift);
    let clif = vm.call("sum_of_squares", vec![Value::Int(3), Value::Int(4)]).expect("Cranelift sum_of_squares");

    // Interpreter
    vm.set_backend(VmBackend::Bytecode);
    let interp = vm.call("sum_of_squares", vec![Value::Int(3), Value::Int(4)]).expect("Interpreter sum_of_squares");

    assert_eq!(clif, Value::Int(25));
    assert_eq!(clif, interp, "Multi-function call parity failed");
}

#[test]
fn test_baseline_path_unchanged_when_bytecode_mode() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    unsafe {
        std::env::remove_var("LUST_JIT_BACKEND");
        std::env::remove_var("LUST_BACKEND");
    }
    let source = r#"
        function baseline_calc(x: int): int
            local acc: int = 0
            local i: int = 0
            while i < x do
                acc = acc + i
                i = i + 1
            end
            return acc
        end
    "#;

    let module = parse_and_lower_module(source, "test_baseline");
    let mut vm = VM::new();
    vm.load_mir_module(&module).expect("Load MIR module");

    // By default, backend is Bytecode
    assert_eq!(vm.backend(), VmBackend::Bytecode);

    let entries_start = vm.cranelift_entry_count;
    assert_eq!(entries_start, 0);

    let res = vm.call("baseline_calc", vec![Value::Int(10)]).expect("Baseline run");
    assert_eq!(res, Value::Int(45));

    // Cranelift counters must remain untouched
    assert_eq!(vm.cranelift_entry_count, 0, "Bytecode mode must never enter Cranelift");
}

#[test]
fn test_direct_vm_run_execution() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let source = r#"
        function direct_run(a: int, b: int): int
            return a * 10 + b
        end
    "#;

    let module = parse_and_lower_module(source, "test_direct");
    let mut vm = VM::new();
    vm.load_mir_module(&module).expect("Load MIR module");

    // Call via vm.call in Cranelift mode (which routes through VM::run)
    vm.set_backend(VmBackend::Cranelift);
    let res = vm.call("direct_run", vec![Value::Int(5), Value::Int(7)]).expect("Direct call");
    assert_eq!(res, Value::Int(57));
    assert!(vm.cranelift_entry_count >= 1);
}

#[test]
fn test_env_var_backend_activation() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    unsafe {
        std::env::set_var("LUST_JIT_BACKEND", "cranelift");
    }
    let vm = VM::new();
    assert_eq!(vm.backend(), VmBackend::Cranelift, "LUST_JIT_BACKEND=cranelift must initialize VM in Cranelift mode");
    unsafe {
        std::env::remove_var("LUST_JIT_BACKEND");
    }
}

#[test]
fn test_parity_across_wide_scalar_range() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let source = r#"
        function compute_edge_cases(x: int, y: int): int
            local q: int = x // y
            local r: int = x % y
            local b: int = (x & 0xFF) ^^ (y & 0xFF)
            local sh: int = (x & 15) << (y & 3)
            return q + r + b + sh
        end
    "#;

    let module = parse_and_lower_module(source, "test_edge_cases");
    let mut vm = VM::new();
    vm.load_mir_module(&module).expect("Load MIR module");

    let cases = [
        (100, 7),
        (-100, 7),
        (100, -7),
        (-100, -7),
        (255, 3),
        (1024, 5),
    ];

    for (x, y) in cases {
        vm.set_backend(VmBackend::Cranelift);
        let clif = vm.call("compute_edge_cases", vec![Value::Int(x), Value::Int(y)]).unwrap();

        vm.set_backend(VmBackend::Bytecode);
        let interp = vm.call("compute_edge_cases", vec![Value::Int(x), Value::Int(y)]).unwrap();

        assert_eq!(clif, interp, "Parity check failed for edge cases ({}, {})", x, y);
    }
}

#[test]
fn test_boolean_return_and_conditions_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let source = r#"
        function is_in_range(x: int, min_val: int, max_val: int): bool
            return x >= min_val and x <= max_val
        end
    "#;

    let module = parse_and_lower_module(source, "test_bool");
    let mut vm = VM::new();
    vm.load_mir_module(&module).expect("Load MIR module");

    let inputs = [
        (5, 1, 10, true),
        (0, 1, 10, false),
        (10, 1, 10, true),
        (11, 1, 10, false),
    ];

    for (x, min_v, max_v, expected) in inputs {
        vm.set_backend(VmBackend::Cranelift);
        let clif = vm.call("is_in_range", vec![Value::Int(x), Value::Int(min_v), Value::Int(max_v)]).unwrap();

        vm.set_backend(VmBackend::Bytecode);
        let interp = vm.call("is_in_range", vec![Value::Int(x), Value::Int(min_v), Value::Int(max_v)]).unwrap();

        assert_eq!(clif, Value::Bool(expected));
        assert_eq!(clif, interp, "Boolean parity check failed for ({}, {}, {})", x, min_v, max_v);
    }
}

#[test]
fn test_nullary_and_many_params_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let source = r#"
        function get_magic_constant(): int
            return 1337
        end

        function sum_five(a: int, b: int, c: int, d: int, e: int): int
            return a + b + c + d + e
        end
    "#;

    let module = parse_and_lower_module(source, "test_params");
    let mut vm = VM::new();
    vm.load_mir_module(&module).expect("Load MIR module");

    // Nullary (0 params)
    vm.set_backend(VmBackend::Cranelift);
    let clif_magic = vm.call("get_magic_constant", vec![]).expect("Cranelift nullary call");
    vm.set_backend(VmBackend::Bytecode);
    let interp_magic = vm.call("get_magic_constant", vec![]).expect("Interp nullary call");
    assert_eq!(clif_magic, Value::Int(1337));
    assert_eq!(clif_magic, interp_magic);

    // 5 params
    vm.set_backend(VmBackend::Cranelift);
    let clif_five = vm.call("sum_five", vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4), Value::Int(5)])
        .expect("Cranelift 5 params call");
    vm.set_backend(VmBackend::Bytecode);
    let interp_five = vm.call("sum_five", vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4), Value::Int(5)])
        .expect("Interp 5 params call");
    assert_eq!(clif_five, Value::Int(15));
    assert_eq!(clif_five, interp_five);
}

#[test]
fn test_fallback_to_interpreter_for_unlowered_bytecode() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let mut vm = VM::new();

    let source = "function bytecode_only(x: int): int return x * 10 end";
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize().expect("Lexer");
    let items = Parser::new(tokens).parse().expect("Parser");
    let mut compiler = lust::Compiler::new();
    let functions = compiler.compile_module(&items).expect("Bytecode compile");
    vm.load_functions(functions);

    // In Cranelift mode, since bytecode_only is not in mir_functions, it should transparently fall back
    vm.set_backend(VmBackend::Cranelift);
    let entries_before = vm.cranelift_entry_count;
    let res = vm.call("bytecode_only", vec![Value::Int(6)]).expect("Fallback run");
    assert_eq!(res, Value::Int(60));
    assert_eq!(vm.cranelift_entry_count, entries_before, "Fallback to interpreter must not increment Cranelift entry count");
}

mod track_0f_remediation_tests {
    use lust::ast::Span;
    use lust::bytecode::value::Value;
    use lust::error::LustError;
    use lust::mir::ir::{
        BlockId, FunctionSignature, MirBinaryOp, MirConstant, MirFunction, MirType, MirUnaryOp,
        Operand, Place, Rvalue, Statement, Terminator,
    };
    use lust::vm::{VM, VmBackend};
    use std::process::Command;

    fn make_const_fn(name: &str, val: i64) -> MirFunction {
        let sig = FunctionSignature {
            name: name.to_string(),
            param_types: vec![],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut f = MirFunction::new(sig, Span::dummy());
        f.blocks[0].terminator = Some(Terminator::Return(
            Some(Operand::Constant(MirConstant::Int(val))),
            Span::dummy(),
        ));
        f
    }

    fn make_binop_fn(name: &str, op: MirBinaryOp) -> MirFunction {
        let sig = FunctionSignature {
            name: name.to_string(),
            param_types: vec![MirType::Int, MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut f = MirFunction::new(sig, Span::dummy());
        let ret = f.return_local;
        let a = f.params[0];
        let b = f.params[1];
        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(ret),
            Rvalue::BinaryOp(op, Operand::Copy(Place::Local(a)), Operand::Copy(Place::Local(b))),
            Span::dummy(),
        ));
        f.blocks[0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(ret))),
            Span::dummy(),
        ));
        f
    }

    #[test]
    fn test_cranelift_div_rem_floordiv_by_zero_returns_runtime_error_no_panic() {
        let mut vm = VM::new();
        vm.load_mir_function(make_binop_fn("clif_div", MirBinaryOp::Div))
            .unwrap();
        vm.load_mir_function(make_binop_fn("clif_mod", MirBinaryOp::Mod))
            .unwrap();
        vm.load_mir_function(make_binop_fn("clif_fdiv", MirBinaryOp::FloorDiv))
            .unwrap();
        vm.set_backend(VmBackend::Cranelift);

        for (fn_name, recovery_expected) in [("clif_div", 5), ("clif_mod", 0), ("clif_fdiv", 5)] {
            let err = vm
                .call(fn_name, vec![Value::Int(10), Value::Int(0)])
                .expect_err("division/mod/floordiv by zero must return Err instead of panicking");
            assert!(
                matches!(err, LustError::RuntimeError { .. }),
                "expected RuntimeError for {fn_name}, got: {err:?}"
            );
            assert!(vm.is_idle(), "VM call stack must remain clean after {fn_name} error");
            let ok = vm
                .call(fn_name, vec![Value::Int(10), Value::Int(2)])
                .expect("valid call after error must succeed");
            assert_eq!(ok, Value::Int(recovery_expected));
        }
    }

    #[test]
    fn test_cranelift_pow_negative_exponent_returns_runtime_error_no_panic() {
        let mut vm = VM::new();
        vm.load_mir_function(make_binop_fn("clif_pow", MirBinaryOp::Pow))
            .unwrap();
        vm.set_backend(VmBackend::Cranelift);

        let err = vm
            .call("clif_pow", vec![Value::Int(2), Value::Int(-1)])
            .expect_err("negative exponent must return Err instead of panicking");
        assert!(matches!(err, LustError::RuntimeError { .. }));
        assert!(vm.is_idle());

        let ok = vm
            .call("clif_pow", vec![Value::Int(2), Value::Int(5)])
            .unwrap();
        assert_eq!(ok, Value::Int(32));
    }

    #[test]
    fn test_cranelift_array_alloc_oom_returns_runtime_error_no_panic() {
        let arr_ty = MirType::Array(Box::new(MirType::Int));
        let sig = FunctionSignature {
            name: "clif_alloc_arr".to_string(),
            param_types: vec![],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut f = MirFunction::new(sig, Span::dummy());
        let arr_local = f.add_local(Some("arr".into()), arr_ty.clone(), false, Span::dummy());
        let ret = f.return_local;
        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(arr_local),
            Rvalue::Aggregate(
                arr_ty,
                vec![
                    Operand::Constant(MirConstant::Int(1)),
                    Operand::Constant(MirConstant::Int(2)),
                    Operand::Constant(MirConstant::Int(3)),
                    Operand::Constant(MirConstant::Int(4)),
                ],
            ),
            Span::dummy(),
        ));
        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(ret),
            Rvalue::UnaryOp(MirUnaryOp::Len, Operand::Copy(Place::Local(arr_local))),
            Span::dummy(),
        ));
        f.blocks[0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(ret))),
            Span::dummy(),
        ));

        let mut vm = VM::new();
        vm.load_mir_function(f.clone()).unwrap();
        vm.set_memory_budget_bytes(8).unwrap();
        vm.set_backend(VmBackend::Cranelift);

        let err = vm
            .call("clif_alloc_arr", vec![])
            .expect_err("array alloc exceeding 8-byte budget must fail gracefully with Err");
        assert!(matches!(err, LustError::RuntimeError { .. }));
        assert!(vm.is_idle());

        let mut vm_ok = VM::new();
        vm_ok.load_mir_function(f).unwrap();
        vm_ok.set_memory_budget_bytes(4096).unwrap();
        vm_ok.set_backend(VmBackend::Cranelift);
        let res = vm_ok.call("clif_alloc_arr", vec![]).unwrap();
        assert_eq!(res, Value::Int(4));
        assert!(vm_ok.memory_used_bytes() >= 48);
    }

    #[test]
    fn test_cranelift_incremental_load_and_reload_cache_invalidation() {
        let mut vm = VM::new();
        vm.load_mir_function(make_const_fn("inc_f1", 10)).unwrap();
        vm.set_backend(VmBackend::Cranelift);

        assert_eq!(vm.call("inc_f1", vec![]).unwrap(), Value::Int(10));
        let comp1 = vm.cranelift_compilation_count;
        assert!(comp1 >= 1);

        let sig_f2 = FunctionSignature {
            name: "inc_f2".to_string(),
            param_types: vec![],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut f2 = MirFunction::new(sig_f2, Span::dummy());
        let tmp = f2.add_local(Some("tmp".into()), MirType::Int, false, Span::dummy());
        let ret = f2.return_local;
        let bb1: BlockId = f2.new_block();
        f2.blocks[0].terminator = Some(Terminator::Call {
            callee: "inc_f1".into(),
            args: vec![],
            destination: Place::Local(tmp),
            success: bb1,
            error: None,
            span: Span::dummy(),
        });
        f2.block_mut(bb1).statements.push(Statement::Assign(
            Place::Local(ret),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(tmp)),
                Operand::Constant(MirConstant::Int(5)),
            ),
            Span::dummy(),
        ));
        f2.block_mut(bb1).terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(ret))),
            Span::dummy(),
        ));

        vm.load_mir_function(f2).unwrap();
        assert_eq!(vm.call("inc_f2", vec![]).unwrap(), Value::Int(15));
        let comp2 = vm.cranelift_compilation_count;
        assert!(comp2 > comp1);

        vm.load_mir_function(make_const_fn("inc_f1", 42)).unwrap();
        assert_eq!(vm.call("inc_f1", vec![]).unwrap(), Value::Int(42));
        assert_eq!(vm.call("inc_f2", vec![]).unwrap(), Value::Int(47));
        assert!(vm.cranelift_compilation_count > comp2);
    }

    #[test]
    fn test_cli_run_jit_cranelift_with_stats() {
        let bin = env!("CARGO_BIN_EXE_lust");
        let script_path = std::env::temp_dir().join(format!(
            "lust_track0f_cli_cranelift_{}.lust",
            std::process::id()
        ));
        let script = "\
function fib(n: int): int
    if n <= 1 then
        return n
    end
    return fib(n - 1) + fib(n - 2)
end

local res: int = fib(10)
print(res)
";
        std::fs::write(&script_path, script).expect("write temp lust script");

        let output = Command::new(bin)
            .args(["run", "--jit=cranelift", script_path.to_str().unwrap()])
            .env("LUST_JIT_STATS", "1")
            .output()
            .expect("execute lust CLI with --jit=cranelift");

        let _ = std::fs::remove_file(&script_path);

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "CLI --jit=cranelift failed: status={:?}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            stdout,
            stderr
        );
        assert!(
            stdout.contains("55"),
            "expected fib(10)=55 on stdout, got: {stdout}"
        );
        let stats_line = stderr
            .lines()
            .find(|l| l.contains("Cranelift stats:"))
            .unwrap_or_else(|| panic!("expected Cranelift stats on stderr, got:\n{stderr}"));
        let entries: u64 = stats_line
            .split("entries=")
            .nth(1)
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("failed to parse entries from '{stats_line}'"));
        let compilations: u64 = stats_line
            .split("compilations=")
            .nth(1)
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("failed to parse compilations from '{stats_line}'"));
        assert!(entries >= 1, "expected entries >= 1, got {entries} ({stats_line})");
        assert!(
            compilations >= 1,
            "expected compilations >= 1, got {compilations} ({stats_line})"
        );
    }
}

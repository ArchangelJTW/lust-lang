//! NG-W9-4: Surgical VM interpreter fast-loop fixes.
//!
//! Verifies:
//! 1. Exact per-instruction gas accounting (`vm.gas_used() == expected_exact_instruction_count`)
//!    across a chain of `Call` and `Return` instructions (preserving `'frame_fast` vs `else` invariant).
//! 2. `vm.cycle_collections_count() == 0` after 1,000+ constructions of `Option::Some(Value::Int(i))`
//!    and scalar tuples (via `CycleCollector::is_leaf`).
//! 3. `MutationM1Guard::enable()` / `MUTATION_M1_SKIP_GAS` works via `SLOW_PATH_SKIP_GAS` without
//!    affecting `SLOW_PATH_PREAMBLE_MASK`.
//! 4. O(1) recursion depth tracking (`CallMeta.active_depth`) scales cleanly on deep direct recursion (B05).

use std::sync::atomic::Ordering;

use lust::bytecode::compiler::Compiler;
use lust::bytecode::{Function, Instruction, Value};
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{
    MUTATION_M1_EXECUTION_COUNT, MUTATION_M1_SKIP_GAS, MutationM1Guard, SLOW_PATH_SKIP_GAS, VM,
};

fn compile_source(source: &str) -> lust::Result<Vec<Function>> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize()?;
    let items = Parser::new(tokens).parse()?;
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    compiler.compile_module(&items)
}

fn plain_vm() -> VM {
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    VM::with_config(&config)
}

static M1_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn test_ng_w9_4_exact_gas_on_call_and_return_chain() {
    let _lock = M1_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    MUTATION_M1_SKIP_GAS.store(false, Ordering::SeqCst);
    let mut vm = plain_vm();

    // Function 0: leaf(x: int) -> int
    //   0: LoadConst R1, 10
    //   1: AddInt    R2, R0, R1
    //   2: Return    R2
    // Total = 3 instructions
    let mut leaf = Function::new("leaf".to_string(), 1, false);
    leaf.register_count = 3;
    let c10 = leaf.chunk.add_constant(Value::Int(10));
    leaf.chunk.emit(Instruction::LoadConst(1, c10), 1);
    leaf.chunk.emit(Instruction::AddInt(2, 0, 1), 1);
    leaf.chunk.emit(Instruction::Return(2), 1);

    // Function 1: mid(x: int) -> int
    //   0: LoadConst R1, Function(0)
    //   1: Call      R1, first_arg=0, arg_count=1, dest=2
    //   2: LoadConst R3, 5
    //   3: AddInt    R4, R2, R3
    //   4: Return    R4
    // Total = 5 instructions in mid + 3 in leaf = 8
    let mut mid = Function::new("mid".to_string(), 1, false);
    mid.register_count = 5;
    let c_leaf = mid.chunk.add_constant(Value::Function(0));
    let c5 = mid.chunk.add_constant(Value::Int(5));
    mid.chunk.emit(Instruction::LoadConst(1, c_leaf), 1);
    mid.chunk.emit(Instruction::Call(1, 0, 1, 2), 1);
    mid.chunk.emit(Instruction::LoadConst(3, c5), 1);
    mid.chunk.emit(Instruction::AddInt(4, 2, 3), 1);
    mid.chunk.emit(Instruction::Return(4), 1);

    // Function 2: top() -> int
    //   0: LoadConst R0, Function(1)
    //   1: LoadConst R1, 27
    //   2: Call      R0, first_arg=1, arg_count=1, dest=2
    //   3: Return    R2
    // Total = 4 instructions in top + 5 in mid + 3 in leaf = 12
    let mut top = Function::new("top".to_string(), 0, false);
    top.register_count = 3;
    let c_mid = top.chunk.add_constant(Value::Function(1));
    let c27 = top.chunk.add_constant(Value::Int(27));
    top.chunk.emit(Instruction::LoadConst(0, c_mid), 1);
    top.chunk.emit(Instruction::LoadConst(1, c27), 1);
    top.chunk.emit(Instruction::Call(0, 1, 1, 2), 1);
    top.chunk.emit(Instruction::Return(2), 1);

    vm.load_functions(vec![leaf, mid, top]);
    vm.reset_gas_counter();

    let res = vm.call("top", vec![]).expect("call chain must succeed");
    assert_eq!(res, Value::Int(42));
    assert_eq!(
        vm.gas_used(),
        12,
        "exact instruction count across Call/Return chain must be preserved"
    );
}

#[test]
fn test_ng_w9_4_option_some_scalar_zero_cycle_collections() {
    let _lock = M1_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    MUTATION_M1_SKIP_GAS.store(false, Ordering::SeqCst);
    let src = r#"
        function wrap_loop(n: int): int
            local acc = 0
            local i = 0
            while i < n do
                local opt: Option<int> = Option.Some(i)
                acc = acc + opt:unwrap()
                i = i + 1
            end
            return acc
        end
    "#;
    let funcs = compile_source(src).expect("compile must succeed");
    let mut vm = plain_vm();
    vm.load_functions(funcs);

    let tracked_before = vm.cycle_collector_tracked_count();
    let res = vm
        .call("wrap_loop", vec![Value::Int(1_000)])
        .expect("wrap_loop must succeed");
    assert_eq!(res, Value::Int(499_500));
    assert_eq!(
        vm.cycle_collections_count(),
        0,
        "1,000 Option::Some(Int) allocations must trigger 0 cycle collections"
    );
    assert_eq!(
        vm.cycle_collector_tracked_count(),
        tracked_before,
        "scalar-payload Option::Some must not register any new containers in CycleCollector"
    );

    // Sanity check that cycle_collections_count() increments on a real collection.
    vm.force_collect_cycles();
    assert_eq!(vm.cycle_collections_count(), 1);
}

#[test]
fn test_ng_w9_4_mutation_m1_via_slow_path_skip_gas() {
    let _lock = M1_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    MUTATION_M1_SKIP_GAS.store(false, Ordering::SeqCst);
    MUTATION_M1_EXECUTION_COUNT.store(0, Ordering::SeqCst);

    let src = r#"
        function sum_n(n: int): int
            local acc = 0
            local i = 0
            while i < n do
                acc = acc + i
                i = i + 1
            end
            return acc
        end
    "#;
    let funcs = compile_source(src).expect("compile must succeed");
    let mut vm = plain_vm();
    vm.load_functions(funcs);

    // Baseline: SLOW_PATH_SKIP_GAS is clear, gas is charged.
    vm.reset_gas_counter();
    let base_res = vm.call("sum_n", vec![Value::Int(20)]).expect("baseline ok");
    assert_eq!(base_res, Value::Int(190));
    assert!(vm.gas_used() > 0, "baseline must consume gas");
    assert_eq!(vm.slow_path_flags() & SLOW_PATH_SKIP_GAS, 0);

    // Under MutationM1Guard: SLOW_PATH_SKIP_GAS is set on run_inner entry, gas is skipped.
    {
        let _guard = MutationM1Guard::enable();
        vm.reset_gas_counter();
        let before_witness = MUTATION_M1_EXECUTION_COUNT.load(Ordering::SeqCst);
        let m1_res = vm.call("sum_n", vec![Value::Int(20)]).expect("m1 ok");
        assert_eq!(m1_res, Value::Int(190));
        assert_eq!(vm.gas_used(), 0, "M1 must skip gas charging");
        assert!(
            MUTATION_M1_EXECUTION_COUNT.load(Ordering::SeqCst) > before_witness,
            "M1 execution witness must increment"
        );
        assert_ne!(
            vm.slow_path_flags() & SLOW_PATH_SKIP_GAS,
            0,
            "SLOW_PATH_SKIP_GAS must be synced while MutationM1Guard is active"
        );
    }

    // After guard drop: SLOW_PATH_SKIP_GAS clears on next call and gas resumes.
    vm.reset_gas_counter();
    let after_res = vm
        .call("sum_n", vec![Value::Int(20)])
        .expect("post-guard ok");
    assert_eq!(after_res, Value::Int(190));
    assert!(vm.gas_used() > 0, "gas charging must resume after guard drop");
    assert_eq!(vm.slow_path_flags() & SLOW_PATH_SKIP_GAS, 0);
}

#[test]
fn test_ng_w9_4_b05_deep_recursion_o1_depth_tracking() {
    let _lock = M1_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    MUTATION_M1_SKIP_GAS.store(false, Ordering::SeqCst);
    let src = r#"
        function rec_sum(n: int): int
            if n <= 0 then
                return 0
            end
            return n + rec_sum(n - 1)
        end
    "#;
    let funcs = compile_source(src).expect("compile must succeed");
    let mut vm = plain_vm();
    vm.load_functions(funcs);

    let res = vm
        .call("rec_sum", vec![Value::Int(500)])
        .expect("depth-500 direct recursion must succeed");
    assert_eq!(res, Value::Int(125_250));
    assert_eq!(
        vm.call_stack_depth(),
        0,
        "call stack must unwind cleanly after depth-500 recursion"
    );

    // Repeat to verify active_depth unwinds cleanly back to 0 and supports subsequent calls.
    let res2 = vm
        .call("rec_sum", vec![Value::Int(500)])
        .expect("second depth-500 direct recursion must succeed");
    assert_eq!(res2, Value::Int(125_250));
    assert_eq!(vm.call_stack_depth(), 0);
}

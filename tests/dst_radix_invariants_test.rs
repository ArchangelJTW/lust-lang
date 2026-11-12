//! Step 5 (P5): DST Determinism & Invariant Acceptance Tests
//!
//! Validates:
//! 1. Deterministic simulation execution via MinimalSimScheduler and VM::with_seed(seed).
//! 2. Bytecode equivalence between radix (0x, 0b, 0o) and decimal forms produces:
//!    - Strictly identical execution outcomes: `outcome_a == outcome_b`
//!    - Strictly identical gas consumption: `sched_a.vm.gas_used() == sched_b.vm.gas_used()`
//!    - Strictly identical scheduler steps: `sched_a.step_counter == sched_b.step_counter`
//! 3. Coverage across bitwise masks, boundary values (i64::MIN, i64::MAX, -1), loops,
//!    multi-function call chains, and multi-fiber execution.
//! 4. Invariant stability across varied PRNG seed distributions.

use lust::bytecode::compiler::Compiler;
use lust::bytecode::{Function, Value};
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::parser::Parser;
use lust::sim::{MinimalSimScheduler, StepOutcome};
use lust::typechecker::TypeChecker;
use lust::vm::VM;

fn compile_source(source: &str) -> lust::Result<Vec<Function>> {
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

struct SimulationResult {
    outcome: StepOutcome,
    gas_used: u64,
    step_counter: u64,
}

fn run_single_fiber_simulation(source: &str, seed: u64, fn_name: &str) -> SimulationResult {
    let funcs = compile_source(source).expect("compile source");
    let fn_index = funcs
        .iter()
        .position(|f| f.name == fn_name)
        .unwrap_or_else(|| panic!("Function '{}' not found in compiled module", fn_name));

    let mut vm = VM::with_seed(seed);
    vm.load_functions(funcs);
    let mut scheduler = MinimalSimScheduler::new(vm);
    let handle = scheduler
        .spawn_fiber(Value::Function(fn_index), vec![])
        .expect("spawn fiber must succeed");
    let outcome = scheduler
        .step_fiber(handle, None)
        .expect("step fiber must succeed");

    SimulationResult {
        outcome,
        gas_used: scheduler.vm.gas_used(),
        step_counter: scheduler.step_counter,
    }
}

fn assert_sim_radix_parity(
    src_radix: &str,
    src_dec: &str,
    seed: u64,
    fn_name: &str,
    expected_val: Option<Value>,
) {
    let res_radix = run_single_fiber_simulation(src_radix, seed, fn_name);
    let res_dec = run_single_fiber_simulation(src_dec, seed, fn_name);

    if let Some(ref expected) = expected_val {
        assert_eq!(
            res_radix.outcome,
            StepOutcome::Completed(Some(expected.clone())),
            "Radix outcome did not match expected value"
        );
        assert_eq!(
            res_dec.outcome,
            StepOutcome::Completed(Some(expected.clone())),
            "Decimal outcome did not match expected value"
        );
    }

    assert_eq!(
        res_radix.outcome, res_dec.outcome,
        "Outcomes between radix and decimal must be identical (seed {})",
        seed
    );
    assert_eq!(
        res_radix.gas_used, res_dec.gas_used,
        "Gas used between radix and decimal must be identical (seed {}): {} vs {}",
        seed, res_radix.gas_used, res_dec.gas_used
    );
    assert_eq!(
        res_radix.step_counter, res_dec.step_counter,
        "Step counter between radix and decimal must be identical (seed {}): {} vs {}",
        seed, res_radix.step_counter, res_dec.step_counter
    );
    assert!(res_radix.gas_used > 0, "Execution must consume gas");
    assert!(
        res_radix.step_counter > 0,
        "Execution must record simulation steps"
    );
}

#[test]
fn test_dst_bitwise_masks_invariants() {
    // 1. Bitwise mask: 0x0FFF & 0b1010_0101 vs 4095 & 165
    let src_radix_1 = r#"
    function worker(): int
        local mask = 0x0FFF
        local flags = 0b1010_0101
        return mask & flags
    end
    "#;
    let src_dec_1 = r#"
    function worker(): int
        local mask = 4095
        local flags = 165
        return mask & flags
    end
    "#;
    assert_sim_radix_parity(src_radix_1, src_dec_1, 100, "worker", Some(Value::Int(165)));

    // 2. Shift and OR: (0x0F << 4) | 0x05 vs (15 << 4) | 5
    let src_radix_2 = r#"
    function worker(): int
        local a = (0x0F << 4) | 0x05
        return a
    end
    "#;
    let src_dec_2 = r#"
    function worker(): int
        local a = (15 << 4) | 5
        return a
    end
    "#;
    assert_sim_radix_parity(src_radix_2, src_dec_2, 100, "worker", Some(Value::Int(245)));

    // 3. 0xDEAD_BEEF & 0xFFFF vs 3735928559 & 65535
    let src_radix_3 = r#"
    function worker(): int
        return 0xDEAD_BEEF & 0xFFFF
    end
    "#;
    let src_dec_3 = r#"
    function worker(): int
        return 3735928559 & 65535
    end
    "#;
    assert_sim_radix_parity(src_radix_3, src_dec_3, 100, "worker", Some(Value::Int(48879)));

    // 4. Octal bitwise: 0o755 & 0o777 vs 493 & 511
    let src_radix_4 = r#"
    function worker(): int
        return 0o755 & 0o777
    end
    "#;
    let src_dec_4 = r#"
    function worker(): int
        return 493 & 511
    end
    "#;
    assert_sim_radix_parity(src_radix_4, src_dec_4, 100, "worker", Some(Value::Int(493)));

    // 5. Bitwise XOR (^^)
    let src_radix_5 = r#"
    function worker(): int
        local x = 0x5555_5555_5555_5555 ^^ 0x3333_3333_3333_3333
        return x
    end
    "#;
    let src_dec_5 = r#"
    function worker(): int
        local x = 6148914691236517205 ^^ 3689348814741910323
        return x
    end
    "#;
    assert_sim_radix_parity(
        src_radix_5,
        src_dec_5,
        100,
        "worker",
        Some(Value::Int(0x5555_5555_5555_5555 ^ 0x3333_3333_3333_3333)),
    );
}

#[test]
fn test_dst_boundary_values_invariants() {
    // 1. i64::MIN arithmetic shift right by 63: 0x8000_... vs -9223372036854775808
    let src_min_radix = r#"
    function worker(): int
        local val = 0x8000_0000_0000_0000 >> 63
        return val
    end
    "#;
    let src_min_dec = r#"
    function worker(): int
        local val = -9223372036854775808 >> 63
        return val
    end
    "#;
    assert_sim_radix_parity(src_min_radix, src_min_dec, 42, "worker", Some(Value::Int(-1)));

    // 2. i64::MIN direct return
    let src_min_val_radix = r#"
    function worker(): int
        return 0x8000_0000_0000_0000
    end
    "#;
    let src_min_val_dec = r#"
    function worker(): int
        return -9223372036854775808
    end
    "#;
    assert_sim_radix_parity(
        src_min_val_radix,
        src_min_val_dec,
        42,
        "worker",
        Some(Value::Int(i64::MIN)),
    );

    // 3. i64::MAX direct return: 0x7FFF_FFFF_FFFF_FFFF vs 9223372036854775807
    let src_max_radix = r#"
    function worker(): int
        return 0x7FFF_FFFF_FFFF_FFFF
    end
    "#;
    let src_max_dec = r#"
    function worker(): int
        return 9223372036854775807
    end
    "#;
    assert_sim_radix_parity(
        src_max_radix,
        src_max_dec,
        42,
        "worker",
        Some(Value::Int(i64::MAX)),
    );

    // 4. Bitwise AND between i64::MIN and i64::MAX must be 0
    let src_min_max_radix = r#"
    function worker(): int
        return 0x8000_0000_0000_0000 & 0x7FFF_FFFF_FFFF_FFFF
    end
    "#;
    let src_min_max_dec = r#"
    function worker(): int
        return -9223372036854775808 & 9223372036854775807
    end
    "#;
    assert_sim_radix_parity(src_min_max_radix, src_min_max_dec, 42, "worker", Some(Value::Int(0)));

    // 5. -1 across hex and binary forms
    let src_neg1_hex = r#"
    function worker(): int
        return 0xFFFF_FFFF_FFFF_FFFF
    end
    "#;
    let src_neg1_bin = r#"
    function worker(): int
        return 0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111
    end
    "#;
    assert_sim_radix_parity(src_neg1_hex, src_neg1_bin, 42, "worker", Some(Value::Int(-1)));
}

#[test]
fn test_dst_loops_and_iterative_computation() {
    // 1. Single while loop with radix counter and bound
    let src_loop_radix = r#"
    function loop_calc(): int
        local sum = 0
        local i = 0
        while i < 0x20 do
            sum = sum + (i & 0x0F) * 0b10
            i = i + 1
        end
        return sum
    end
    "#;
    let src_loop_dec = r#"
    function loop_calc(): int
        local sum = 0
        local i = 0
        while i < 32 do
            sum = sum + (i & 15) * 2
            i = i + 1
        end
        return sum
    end
    "#;
    // (0+1+..+15) = 120 * 2 = 240, executed twice (for 0..15 and 16..31) -> 240 * 2 = 480
    assert_sim_radix_parity(src_loop_radix, src_loop_dec, 777, "loop_calc", Some(Value::Int(480)));

    // 2. Nested while loops with mixed radices and separators
    let src_nested_radix = r#"
    function nested_calc(): int
        local total = 0
        local i = 0
        while i < 0x0A do
            local j = 0
            while j < 0b101 do
                total = total + (i * 0b10) + (j & 0x03)
                j = j + 1
            end
            i = i + 1
        end
        return total
    end
    "#;
    let src_nested_dec = r#"
    function nested_calc(): int
        local total = 0
        local i = 0
        while i < 10 do
            local j = 0
            while j < 5 do
                total = total + (i * 2) + (j & 3)
                j = j + 1
            end
            i = i + 1
        end
        return total
    end
    "#;
    assert_sim_radix_parity(src_nested_radix, src_nested_dec, 777, "nested_calc", Some(Value::Int(510)));
}

#[test]
fn test_dst_function_call_chains() {
    let src_chain_radix = r#"
    function step1(x: int): int
        return (x << 0b10) | 0x03
    end
    function step2(y: int): int
        return step1(y) + 0o10
    end
    function step3(z: int): int
        return step2(z * 0b11) & 0x0FFF
    end
    function main_worker(): int
        return step3(0x15)
    end
    "#;

    let src_chain_dec = r#"
    function step1(x: int): int
        return (x << 2) | 3
    end
    function step2(y: int): int
        return step1(y) + 8
    end
    function step3(z: int): int
        return step2(z * 3) & 4095
    end
    function main_worker(): int
        return step3(21)
    end
    "#;

    // z = 21 -> z * 3 = 63
    // step1(63) = (63 << 2) | 3 = 252 | 3 = 255
    // step2(63) = 255 + 8 = 263
    // step3(21) = 263 & 4095 = 263
    assert_sim_radix_parity(
        src_chain_radix,
        src_chain_dec,
        2024,
        "main_worker",
        Some(Value::Int(263)),
    );
}

#[test]
fn test_dst_seed_stability_invariants() {
    let seeds = [0u64, 1, 7, 42, 100, 256, 1337, 65535, 999999, u64::MAX];

    let src_radix = r#"
    function worker(): int
        local mask = 0x0FFF
        local flags = 0b1010_0101
        local shift = 0o04
        return (mask & flags) << shift
    end
    "#;
    let src_dec = r#"
    function worker(): int
        local mask = 4095
        local flags = 165
        local shift = 4
        return (mask & flags) << shift
    end
    "#;
    // (165 << 4) = 2640

    for &seed in &seeds {
        assert_sim_radix_parity(
            src_radix,
            src_dec,
            seed,
            "worker",
            Some(Value::Int(2640)),
        );
    }
}

#[test]
fn test_dst_multi_fiber_concurrent_determinism() {
    let src_radix = r#"
    function fiber_a(): int
        local mask = 0x0FFF
        local flags = 0b1010_0101
        return mask & flags
    end

    function fiber_b(): int
        local sum = 0
        local i = 0
        while i < 0x10 do
            sum = sum + i * 0b10
            i = i + 1
        end
        return sum
    end
    "#;

    let src_dec = r#"
    function fiber_a(): int
        local mask = 4095
        local flags = 165
        return mask & flags
    end

    function fiber_b(): int
        local sum = 0
        local i = 0
        while i < 16 do
            sum = sum + i * 2
            i = i + 1
        end
        return sum
    end
    "#;

    let seed = 314159;

    // Radix multi-fiber run
    let funcs_radix = compile_source(src_radix).expect("compile radix");
    let idx_radix_a = funcs_radix
        .iter()
        .position(|f| f.name == "fiber_a")
        .expect("fiber_a in radix funcs");
    let idx_radix_b = funcs_radix
        .iter()
        .position(|f| f.name == "fiber_b")
        .expect("fiber_b in radix funcs");

    let mut vm_radix = VM::with_seed(seed);
    vm_radix.load_functions(funcs_radix);
    let mut sched_radix = MinimalSimScheduler::new(vm_radix);

    let h_radix_a = sched_radix
        .spawn_fiber(Value::Function(idx_radix_a), vec![])
        .expect("spawn radix a");
    let h_radix_b = sched_radix
        .spawn_fiber(Value::Function(idx_radix_b), vec![])
        .expect("spawn radix b");

    let outcome_radix_a = sched_radix
        .step_fiber(h_radix_a, None)
        .expect("step radix a");
    let outcome_radix_b = sched_radix
        .step_fiber(h_radix_b, None)
        .expect("step radix b");

    // Decimal multi-fiber run
    let funcs_dec = compile_source(src_dec).expect("compile dec");
    let idx_dec_a = funcs_dec
        .iter()
        .position(|f| f.name == "fiber_a")
        .expect("fiber_a in dec funcs");
    let idx_dec_b = funcs_dec
        .iter()
        .position(|f| f.name == "fiber_b")
        .expect("fiber_b in dec funcs");

    let mut vm_dec = VM::with_seed(seed);
    vm_dec.load_functions(funcs_dec);
    let mut sched_dec = MinimalSimScheduler::new(vm_dec);

    let h_dec_a = sched_dec
        .spawn_fiber(Value::Function(idx_dec_a), vec![])
        .expect("spawn dec a");
    let h_dec_b = sched_dec
        .spawn_fiber(Value::Function(idx_dec_b), vec![])
        .expect("spawn dec b");

    let outcome_dec_a = sched_dec
        .step_fiber(h_dec_a, None)
        .expect("step dec a");
    let outcome_dec_b = sched_dec
        .step_fiber(h_dec_b, None)
        .expect("step dec b");

    // Check fiber outcomes:
    // fiber_a: 4095 & 165 = 165
    // fiber_b: (0+..+15) * 2 = 120 * 2 = 240
    assert_eq!(outcome_radix_a, StepOutcome::Completed(Some(Value::Int(165))));
    assert_eq!(outcome_dec_a, StepOutcome::Completed(Some(Value::Int(165))));
    assert_eq!(outcome_radix_b, StepOutcome::Completed(Some(Value::Int(240))));
    assert_eq!(outcome_dec_b, StepOutcome::Completed(Some(Value::Int(240))));

    // Multi-fiber determinism invariants:
    assert_eq!(
        outcome_radix_a, outcome_dec_a,
        "Fiber A outcomes must match"
    );
    assert_eq!(
        outcome_radix_b, outcome_dec_b,
        "Fiber B outcomes must match"
    );
    assert_eq!(
        sched_radix.vm.gas_used(),
        sched_dec.vm.gas_used(),
        "Total scheduler gas used must match between radix and decimal multi-fiber runs"
    );
    assert_eq!(
        sched_radix.step_counter,
        sched_dec.step_counter,
        "Total scheduler steps must match between radix and decimal multi-fiber runs"
    );
    assert!(
        sched_radix.vm.gas_used() > 0,
        "Total scheduler gas used must be positive"
    );
    assert!(
        sched_radix.step_counter > 0,
        "Total scheduler steps must be positive"
    );
}

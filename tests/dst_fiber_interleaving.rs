//! Gate 4 Acceptance Test: Preemptive Fiber Interleaving and Deterministic Race Condition Detection (D4, Gate 4)
//!
//! Validates:
//! 1. Modeled race condition between two concurrent fibers sharing a mutable data structure (array counter).
//! 2. Sequential/cooperative execution produces the expected atomic sum (no race condition).
//! 3. `PreemptiveExplorationScheduler` explores interleavings across pseudo-random seeds.
//! 4. Proves that within 500 seeds, `PreemptiveExplorationScheduler` detects the race condition (lost update: counter < expected).
//! 5. Captures the exact failing seed and re-runs the scheduler with that seed.
//! 6. Proves deterministic reproducibility: the re-run with the failing seed produces the exact identical race condition output and execution trace.

use std::cell::RefCell;
use std::rc::Rc;

use lust::bytecode::{Function, Instruction, Value};
use lust::sim::PreemptiveExplorationScheduler;
use lust::vm::TaskState;

fn create_counter_incrementer_function(increments: i64) -> Function {
    let mut func = Function::new("increment_counter".to_string(), 1, false);
    func.register_count = 8;
    let c_idx = func.chunk.add_constant(Value::Int(0));
    let c_limit = func.chunk.add_constant(Value::Int(increments));
    let c_step = func.chunk.add_constant(Value::Int(1));
    let c_zero = func.chunk.add_constant(Value::Int(0));

    // Setup:
    // R0 = arg 0 (shared Array)
    // R1 = 0 (counter index)
    // R2 = 0 (loop variable i)
    // R3 = increments (limit)
    // R4 = 1 (step)
    func.chunk.emit(Instruction::LoadConst(1, c_idx), 1);
    func.chunk.emit(Instruction::LoadConst(2, c_zero), 1);
    func.chunk.emit(Instruction::LoadConst(3, c_limit), 1);
    func.chunk.emit(Instruction::LoadConst(4, c_step), 1);

    // Loop header:
    let loop_head = func.chunk.instructions.len();
    func.chunk.emit(Instruction::Lt(5, 2, 3), 2);
    let exit_jump = func.chunk.emit(Instruction::JumpIfNot(5, 0), 2);

    // Non-atomic Read-Modify-Write sequence:
    // 1. Read: R6 = R0[R1]
    func.chunk.emit(Instruction::GetIndex(6, 0, 1), 3);
    // 2. Modify: R6 = R6 + R4
    func.chunk.emit(Instruction::Add(6, 6, 4), 3);
    // 3. Write: R0[R1] = R6
    func.chunk.emit(Instruction::SetIndex(0, 1, 6), 3);

    // Increment loop index: R2 = R2 + R4
    func.chunk.emit(Instruction::Add(2, 2, 4), 4);
    let back_jump = func.chunk.emit(Instruction::Jump(0), 4);
    func.chunk.patch_jump(back_jump, loop_head);

    let loop_exit = func.chunk.instructions.len();
    func.chunk.patch_jump(exit_jump, loop_exit);
    func.chunk.emit(Instruction::Return(0), 5);

    func
}

#[test]
fn test_sequential_execution_has_no_race_condition() {
    let mut scheduler = PreemptiveExplorationScheduler::with_seed(100);
    let func = create_counter_incrementer_function(5);
    scheduler.vm.load_functions(vec![func]);

    let shared_array = Value::Array(Rc::new(RefCell::new(vec![Value::Int(0)])));
    let func_val = Value::Function(0);

    // Fiber 1 runs to completion first
    let h1 = scheduler
        .vm
        .create_task_value(func_val.clone(), vec![shared_array.clone()])
        .unwrap();
    scheduler.vm.resume_task_handle(h1, None).unwrap();
    assert_eq!(
        scheduler.vm.get_task_instance(h1).unwrap().state,
        TaskState::Completed
    );

    // Fiber 2 runs to completion second
    let h2 = scheduler
        .vm
        .create_task_value(func_val, vec![shared_array.clone()])
        .unwrap();
    scheduler.vm.resume_task_handle(h2, None).unwrap();
    assert_eq!(
        scheduler.vm.get_task_instance(h2).unwrap().state,
        TaskState::Completed
    );

    // Without preemption/interleaving, 5 + 5 = exactly 10
    if let Value::Array(arr) = &shared_array {
        let final_count = arr.borrow()[0].clone();
        assert_eq!(final_count, Value::Int(10));
    } else {
        panic!("shared_array must be Value::Array");
    }
}

#[test]
fn test_preemptive_interleaving_detects_race_condition_within_500_seeds_and_reproduces() {
    let increments_per_fiber = 5;
    let expected_atomic_total = increments_per_fiber * 2; // 10

    let mut failing_seed: Option<u64> = None;
    let mut failing_counter_value: Option<i64> = None;
    let mut failing_execution_log = Vec::new();

    // Search across up to 500 seeds to find an interleaving that triggers the lost update race condition
    for seed in 0..500 {
        let mut scheduler = PreemptiveExplorationScheduler::with_seed(seed)
            .with_max_quantum(3)
            .with_step_limit(50_000);

        let func = create_counter_incrementer_function(increments_per_fiber);
        scheduler.vm.load_functions(vec![func]);

        let shared_array = Value::Array(Rc::new(RefCell::new(vec![Value::Int(0)])));
        let func_val = Value::Function(0);

        let _h1 = scheduler
            .spawn_fiber(func_val.clone(), vec![shared_array.clone()])
            .unwrap();
        let _h2 = scheduler
            .spawn_fiber(func_val, vec![shared_array.clone()])
            .unwrap();

        let steps = scheduler.run_until_quiescent_or_limit(10_000).unwrap();
        assert!(steps > 0);

        let counter_val = if let Value::Array(arr) = &shared_array {
            match arr.borrow()[0] {
                Value::Int(v) => v,
                _ => panic!("Expected Value::Int in shared array"),
            }
        } else {
            panic!("Expected Value::Array");
        };

        if counter_val < expected_atomic_total {
            failing_seed = Some(seed);
            failing_counter_value = Some(counter_val);
            failing_execution_log = scheduler.execution_log.clone();
            break;
        }
    }

    // Assertion 1: A race condition was discovered within 500 seeds
    assert!(
        failing_seed.is_some(),
        "PreemptiveExplorationScheduler failed to discover race condition within 500 seeds"
    );

    let seed = failing_seed.unwrap();
    let expected_race_val = failing_counter_value.unwrap();
    println!(
        "Race condition discovered at seed {}: final counter = {} (expected atomic = {})",
        seed, expected_race_val, expected_atomic_total
    );

    // Assertion 2: Re-running with the exact same seed reproduces identical race condition
    let mut replay_scheduler = PreemptiveExplorationScheduler::with_seed(seed)
        .with_max_quantum(3)
        .with_step_limit(50_000);

    let func = create_counter_incrementer_function(increments_per_fiber);
    replay_scheduler.vm.load_functions(vec![func]);

    let replay_shared_array = Value::Array(Rc::new(RefCell::new(vec![Value::Int(0)])));
    let func_val = Value::Function(0);

    let _h1 = replay_scheduler
        .spawn_fiber(func_val.clone(), vec![replay_shared_array.clone()])
        .unwrap();
    let _h2 = replay_scheduler
        .spawn_fiber(func_val, vec![replay_shared_array.clone()])
        .unwrap();

    let replay_steps = replay_scheduler
        .run_until_quiescent_or_limit(10_000)
        .unwrap();
    assert!(replay_steps > 0);

    let replay_counter_val = if let Value::Array(arr) = &replay_shared_array {
        match arr.borrow()[0] {
            Value::Int(v) => v,
            _ => panic!("Expected Value::Int in shared array"),
        }
    } else {
        panic!("Expected Value::Array");
    };

    // Verify counter value matches down to the exact integer
    assert_eq!(
        replay_counter_val, expected_race_val,
        "Replay did not reproduce identical race condition counter value!"
    );

    // Verify step count and execution trace match 100% deterministically
    assert_eq!(
        replay_scheduler.execution_log.len(),
        failing_execution_log.len(),
        "Execution log length mismatch between initial run and replay"
    );
    assert_eq!(
        replay_scheduler.execution_log, failing_execution_log,
        "Execution log traces differ between initial run and replay"
    );
}

//! Example: Deterministic Preemptive Fiber Scheduling in Simulation Testing (DST).
//! Demonstrates running multiple concurrent fibers under `PreemptiveExplorationScheduler`
//! with a pseudo-random seed, step limits, and proves 100% deterministic reproducibility.

use lust::bytecode::{Function, Instruction, TaskHandle, Value};
use lust::sim::PreemptiveExplorationScheduler;
use lust::vm::TaskState;

fn build_counter_fiber(limit: i64) -> Function {
    let mut func = Function::new("counter_fiber".to_string(), 0, false);
    func.register_count = 4;
    let c_limit = func.chunk.add_constant(Value::Int(limit));
    let c_one = func.chunk.add_constant(Value::Int(1));
    let c_zero = func.chunk.add_constant(Value::Int(0));

    // R0 = 0 (counter), R1 = limit, R2 = 1 (step)
    func.chunk.emit(Instruction::LoadConst(0, c_zero), 1);
    func.chunk.emit(Instruction::LoadConst(1, c_limit), 1);
    func.chunk.emit(Instruction::LoadConst(2, c_one), 1);

    // Loop header
    let loop_head = func.chunk.instructions.len();
    func.chunk.emit(Instruction::Lt(3, 0, 1), 2);
    let exit_jump = func.chunk.emit(Instruction::JumpIfNot(3, 0), 2);

    func.chunk.emit(Instruction::Add(0, 0, 2), 3);
    let back_jump = func.chunk.emit(Instruction::Jump(0), 4);
    func.chunk.patch_jump(back_jump, loop_head);

    let loop_exit = func.chunk.instructions.len();
    func.chunk.patch_jump(exit_jump, loop_exit);
    func.chunk.emit(Instruction::Return(0), 5);

    func
}

fn run_simulation_run(seed: u64) -> (usize, Vec<(TaskHandle, u64, TaskState)>) {
    let mut scheduler = PreemptiveExplorationScheduler::with_seed(seed)
        .with_max_quantum(2)
        .with_step_limit(10_000);

    let func = build_counter_fiber(10);
    scheduler.vm.load_functions(vec![func]);

    let _h1 = scheduler
        .spawn_fiber(Value::Function(0), vec![])
        .expect("Fiber 1 spawn succeeds");
    let _h2 = scheduler
        .spawn_fiber(Value::Function(0), vec![])
        .expect("Fiber 2 spawn succeeds");

    let steps = scheduler
        .run_until_quiescent_or_limit(10_000)
        .expect("Simulation must complete without deadlock");

    (steps, scheduler.execution_log.clone())
}

fn main() {
    println!("--- [Deterministic Simulation Scheduler Example] ---");

    let seed = 42;
    println!("Running initial simulation with seed {}...", seed);
    let (steps1, log1) = run_simulation_run(seed);
    println!("✓ Run 1 finished in {} simulation steps", steps1);

    println!("Re-running simulation with identical seed {}...", seed);
    let (steps2, log2) = run_simulation_run(seed);
    println!("✓ Run 2 finished in {} simulation steps", steps2);

    // Assert absolute deterministic reproducibility (DST Invariant)
    assert_eq!(steps1, steps2, "Step count must be perfectly identical");
    assert_eq!(log1, log2, "Execution logs must match byte-for-byte");
    println!("✓ Deterministic reproducibility confirmed: 100% trace match");

    // Assert that a different seed produces a divergent execution trace
    let seed_alt = 43;
    println!("Running simulation with divergent seed {}...", seed_alt);
    let (_steps3, log3) = run_simulation_run(seed_alt);
    assert_ne!(
        log1, log3,
        "Different seeds must produce divergent execution traces"
    );
    println!(
        "✓ Divergence verified: seed {} produced distinct scheduling decisions from seed {}",
        seed_alt, seed
    );

    println!("All simulation scheduler assertions passed!");
}

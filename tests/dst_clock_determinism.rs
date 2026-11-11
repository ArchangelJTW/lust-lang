//! Gate 2: Deterministic Clock & Virtual Time Advancement Test (DST Phase 2, D1)
//!
//! Validates:
//! 1. WallClock and MonotonicClock strict separation.
//! 2. TimerControl virtual time advancement.
//! 3. Deterministic 24-hour virtual time scenario completing in < 100 scheduler steps without blocking OS thread.
//! 4. Bit-for-bit replay and step bound watchdog.

use core::time::Duration;
use lust::bytecode::{Function, Instruction, NativeCallResult, Value, native_fn};
use lust::sim::{LustClock, MinimalSimScheduler, StepOutcome};
use lust::vm::{TaskState, VM};

#[test]
fn test_wall_and_monotonic_clock_separation() {
    let epoch_start = 1_700_000_000_000_000_000u64; // arbitrary epoch
    let mono_start = 1_000_000_000u64;
    let clock = LustClock::simulated(epoch_start, mono_start);

    assert!(clock.is_simulated());
    assert_eq!(clock.now_epoch_nanos(), epoch_start);
    assert_eq!(clock.now_nanos(), mono_start);
    assert_eq!(clock.now_epoch_secs_f64(), 1_700_000_000.0);

    // Advance by 500ms
    clock.advance(Duration::from_millis(500));
    assert_eq!(clock.now_epoch_nanos(), epoch_start + 500_000_000);
    assert_eq!(clock.now_nanos(), mono_start + 500_000_000);
    assert_eq!(clock.now_epoch_secs_f64(), 1_700_000_000.5);

    // Monotonic clock is strictly non-decreasing
    let mono_1 = clock.now_nanos();
    clock.advance(Duration::from_micros(1));
    let mono_2 = clock.now_nanos();
    assert!(mono_2 > mono_1);
}

#[test]
fn test_timer_queue_scheduling_and_cancellation() {
    let clock = LustClock::simulated(0, 0);

    let flag1 = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag2 = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    let f1 = flag1.clone();
    let f2 = flag2.clone();

    // Schedule timer 1 for 100ms and timer 2 for 200ms
    let tid1 = clock.schedule_timer(100_000_000, 1, move || {
        f1.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    let _tid2 = clock.schedule_timer(200_000_000, 1, move || {
        f2.store(true, std::sync::atomic::Ordering::SeqCst);
    });

    assert_eq!(clock.next_deadline(), Some(100_000_000));

    // Cancel timer 1
    assert!(clock.cancel_timer(tid1, 1));
    // Timer 1 cancelled: next deadline is now timer 2 (200ms)
    assert_eq!(clock.next_deadline(), Some(200_000_000));

    // Advance to 150ms: timer 1 was cancelled, timer 2 not yet expired
    clock.advance(Duration::from_millis(150));
    let expired = clock.poll_expired_timers(clock.now_nanos());
    assert_eq!(expired.len(), 0);
    assert!(!flag1.load(std::sync::atomic::Ordering::SeqCst));
    assert!(!flag2.load(std::sync::atomic::Ordering::SeqCst));

    // Advance to 250ms: timer 2 expires
    clock.advance(Duration::from_millis(100));
    let expired = clock.poll_expired_timers(clock.now_nanos());
    assert_eq!(expired.len(), 1);
    for act in expired {
        act();
    }
    assert!(!flag1.load(std::sync::atomic::Ordering::SeqCst));
    assert!(flag2.load(std::sync::atomic::Ordering::SeqCst));
}

#[test]
fn test_24_hour_delay_scenario_deterministic_within_100_steps() {
    // 24 hours in seconds: 86400
    // Real time execution must be instantaneous (< 100ms) and < 100 scheduler steps.
    let real_start = std::time::Instant::now();

    let mut config = lust::config::LustConfig::default();
    config.enable_module("os");
    let mut vm = VM::with_config_and_seed(&config, 42);
    vm.clock = std::sync::Arc::new(lust::sim::LustClock::simulated(0, 0));

    // Create worker function that sleeps for 24 hours (86,400s), then returns 42
    let mut func = Function::new("sleep_24h_worker".to_string(), 0, false);
    func.register_count = 8;

    let os_sleep = match vm.get_global("os").expect("os module must exist") {
        Value::Map(map) => {
            let key = lust::bytecode::ValueKey::from(Value::string("sleep"));
            map.borrow().get(&key).cloned().expect("os.sleep must exist")
        }
        _ => panic!("os is not a map"),
    };

    let c_sleep = func.chunk.add_constant(os_sleep);
    let c_duration = func.chunk.add_constant(Value::Float(86_400.0));
    let c_result = func.chunk.add_constant(Value::Int(42));

    func.chunk.emit(Instruction::LoadConst(0, c_sleep), 1);
    func.chunk.emit(Instruction::LoadConst(1, c_duration), 1);
    // Call sleep(86400.0) -> yields with 24h timer
    func.chunk.emit(Instruction::Call(0, 1, 1, 2), 1);
    func.chunk.emit(Instruction::LoadConst(3, c_result), 1);
    func.chunk.emit(Instruction::Return(3), 1);

    vm.load_functions(vec![func]);

    let mut scheduler = MinimalSimScheduler::new(vm);
    let handle = scheduler
        .spawn_fiber(Value::Function(0), vec![])
        .expect("spawn must succeed");

    // Execute with virtual time stepping
    let outcome = scheduler
        .run_task_with_virtual_time(handle, 100)
        .expect("24-hour simulation must succeed");

    assert_eq!(outcome, StepOutcome::Completed(Some(Value::Int(42))));

    // Verification 1: Scheduler step counter must be strictly < 100 steps
    assert!(
        scheduler.step_counter < 100,
        "Steps used: {} must be < 100",
        scheduler.step_counter
    );

    // Verification 2: Virtual clock must have advanced by exactly 24 hours (86,400 seconds)
    let virtual_elapsed_nanos = scheduler.vm.clock.now_nanos();
    assert_eq!(
        virtual_elapsed_nanos,
        86_400 * 1_000_000_000,
        "Virtual clock did not advance to 24 hours"
    );

    // Verification 3: Physical wall clock elapsed time must be < 100ms
    let real_elapsed = real_start.elapsed();
    assert!(
        real_elapsed < Duration::from_millis(100),
        "Real time took {:?}, must not block host OS thread!",
        real_elapsed
    );

    // Verification 4: Task state in VM must be Completed
    let task = scheduler.vm.get_task_instance(handle).expect("task must exist");
    assert_eq!(task.state, TaskState::Completed);
}

#[test]
fn test_bit_for_bit_virtual_replay_across_runs() {
    fn run_simulation() -> (u64, u64, Option<Value>) {
        let mut vm = VM::with_sim_clock(999, 100, 100);
        let mut func = Function::new("deterministic_flow".to_string(), 0, false);
        func.register_count = 8;

        let step_fn = native_fn(|args| {
            let increment = args.first().and_then(|v| v.as_int()).unwrap_or(1);
            if let Some(vm_ptr) = lust::vm::current_vm_ptr() {
                let vm = unsafe { &mut *vm_ptr };
                vm.clock.advance(Duration::from_millis(increment as u64 * 100));
            }
            Ok(NativeCallResult::Return(Value::Int(increment * 2)))
        });

        let c_step = func.chunk.add_constant(Value::NativeFunction(step_fn));
        let c_val = func.chunk.add_constant(Value::Int(5));
        func.chunk.emit(Instruction::LoadConst(0, c_step), 1);
        func.chunk.emit(Instruction::LoadConst(1, c_val), 1);
        func.chunk.emit(Instruction::Call(0, 1, 1, 2), 1);
        func.chunk.emit(Instruction::Return(2), 1);

        vm.load_functions(vec![func]);

        let mut scheduler = MinimalSimScheduler::new(vm);
        let handle = scheduler
            .spawn_fiber(Value::Function(0), vec![])
            .expect("spawn");
        let outcome = scheduler.run_task_with_virtual_time(handle, 50).expect("run");

        let ret = match outcome {
            StepOutcome::Completed(v) => v,
            _ => None,
        };

        (scheduler.step_counter, scheduler.vm.clock.now_nanos(), ret)
    }

    let run1 = run_simulation();
    let run2 = run_simulation();
    let run3 = run_simulation();

    assert_eq!(run1, run2);
    assert_eq!(run2, run3);
    assert_eq!(run1.2, Some(Value::Int(10)));
}

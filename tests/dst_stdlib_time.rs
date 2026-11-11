//! Gate 2: stdlib os.time, os.date, and nonblocking os.sleep under SimClock (DST Phase 2, D1)
//!
//! Validates:
//! 1. `os.time()` is routed through `WallClock` of current VM instance.
//! 2. `os.date()` formats deterministic date strings and date tables (`"*t"`).
//! 3. `os.sleep()` is nonblocking under `SimClock`: yields fiber with timer, resumes on deadline,
//!    completes without blocking host OS thread.

use core::time::Duration;
use lust::bytecode::{Function, Instruction, Value, ValueKey};
use lust::sim::{MinimalSimScheduler, StepOutcome};
use lust::vm::{TaskState, VM};

fn get_os_fn(vm: &VM, name: &str) -> Value {
    match vm.get_global("os").expect("os module must exist") {
        Value::Map(map) => {
            let key = ValueKey::from(Value::string(name));
            map.borrow().get(&key).cloned().unwrap_or_else(|| panic!("os.{} must exist", name))
        }
        _ => panic!("os is not a map"),
    }
}

fn get_map_field(map_val: &Value, name: &str) -> Option<Value> {
    match map_val {
        Value::Map(map) => {
            let key = ValueKey::from(Value::string(name));
            map.borrow().get(&key).cloned()
        }
        _ => None,
    }
}

fn make_sim_vm(seed: u64, epoch_nanos: u64, mono_nanos: u64) -> VM {
    let mut config = lust::config::LustConfig::default();
    config.enable_module("os");
    let mut vm = VM::with_config_and_seed(&config, seed);
    vm.clock = std::sync::Arc::new(lust::sim::LustClock::simulated(epoch_nanos, mono_nanos));
    vm
}

#[test]
fn test_stdlib_os_time_deterministic_with_sim_clock() {
    // Epoch timestamp: 2026-11-11 12:00:00 UTC = 1_794_408_000 seconds
    let epoch_nanos = 1_794_408_000 * 1_000_000_000u64;
    let mut vm = make_sim_vm(42, epoch_nanos, 0);

    let os_time_fn = get_os_fn(&vm, "time");

    let _vm_guard = lust::vm::CurrentVmGuard::new(&mut vm as *mut VM);
    let mut func = Function::new("test_time".to_string(), 0, false);
    func.register_count = 4;
    let c_fn = func.chunk.add_constant(os_time_fn);
    func.chunk.emit(Instruction::LoadConst(0, c_fn), 1);
    func.chunk.emit(Instruction::Call(0, 0, 0, 1), 1);
    func.chunk.emit(Instruction::Return(1), 1);

    vm.load_functions(vec![func]);

    let mut scheduler = MinimalSimScheduler::new(vm);
    let h1 = scheduler.spawn_fiber(Value::Function(0), vec![]).unwrap();
    let out1 = scheduler.step_fiber(h1, None).unwrap();
    assert_eq!(
        out1,
        StepOutcome::Completed(Some(Value::Float(1_794_408_000.0)))
    );

    // Advance clock by 3600 seconds (1 hour)
    scheduler.advance_time(Duration::from_secs(3600));

    // Call os.time() again
    let h2 = scheduler.spawn_fiber(Value::Function(0), vec![]).unwrap();
    let out2 = scheduler.step_fiber(h2, None).unwrap();
    assert_eq!(
        out2,
        StepOutcome::Completed(Some(Value::Float(1_794_411_600.0)))
    );
}

#[test]
fn test_stdlib_os_date_formatting_and_table() {
    // 2026-11-11 12:30:45 UTC = 1_794_409_845 seconds
    let epoch_secs = 1_794_409_845i64;
    let epoch_nanos = (epoch_secs as u64) * 1_000_000_000;
    let mut vm = make_sim_vm(42, epoch_nanos, 0);

    let os_date_fn = get_os_fn(&vm, "date");

    let _vm_guard = lust::vm::CurrentVmGuard::new(&mut vm as *mut VM);

    // Test 1: os.date() without arguments -> returns "%Y-%m-%d %H:%M:%S"
    let mut func1 = Function::new("test_date_default".to_string(), 0, false);
    func1.register_count = 4;
    let c_fn = func1.chunk.add_constant(os_date_fn.clone());
    func1.chunk.emit(Instruction::LoadConst(0, c_fn), 1);
    func1.chunk.emit(Instruction::Call(0, 0, 0, 1), 1);
    func1.chunk.emit(Instruction::Return(1), 1);

    // Test 2: os.date("%Y-%m-%d") -> custom format string
    let mut func2 = Function::new("test_date_custom".to_string(), 0, false);
    func2.register_count = 4;
    let c_fn = func2.chunk.add_constant(os_date_fn.clone());
    let c_fmt = func2.chunk.add_constant(Value::string("%Y-%m-%d"));
    func2.chunk.emit(Instruction::LoadConst(0, c_fn), 1);
    func2.chunk.emit(Instruction::LoadConst(1, c_fmt), 1);
    func2.chunk.emit(Instruction::Call(0, 1, 1, 2), 1);
    func2.chunk.emit(Instruction::Return(2), 1);

    // Test 3: os.date("*t") -> returns table of components
    let mut func3 = Function::new("test_date_table".to_string(), 0, false);
    func3.register_count = 4;
    let c_fn = func3.chunk.add_constant(os_date_fn);
    let c_fmt_t = func3.chunk.add_constant(Value::string("*t"));
    func3.chunk.emit(Instruction::LoadConst(0, c_fn), 1);
    func3.chunk.emit(Instruction::LoadConst(1, c_fmt_t), 1);
    func3.chunk.emit(Instruction::Call(0, 1, 1, 2), 1);
    func3.chunk.emit(Instruction::Return(2), 1);

    vm.load_functions(vec![func1, func2, func3]);

    let mut scheduler = MinimalSimScheduler::new(vm);

    // Run test 1: default format
    let h1 = scheduler.spawn_fiber(Value::Function(0), vec![]).unwrap();
    let out1 = scheduler.step_fiber(h1, None).unwrap();
    assert_eq!(
        out1,
        StepOutcome::Completed(Some(Value::string("2026-11-11 15:10:45")))
    );

    // Run test 2: custom format
    let h2 = scheduler.spawn_fiber(Value::Function(1), vec![]).unwrap();
    let out2 = scheduler.step_fiber(h2, None).unwrap();
    assert_eq!(
        out2,
        StepOutcome::Completed(Some(Value::string("2026-11-11")))
    );

    // Run test 3: "*t" table
    let h3 = scheduler.spawn_fiber(Value::Function(2), vec![]).unwrap();
    let out3 = scheduler.step_fiber(h3, None).unwrap();
    match out3 {
        StepOutcome::Completed(Some(map_val)) => {
            assert_eq!(get_map_field(&map_val, "year"), Some(Value::Int(2026)));
            assert_eq!(get_map_field(&map_val, "month"), Some(Value::Int(11)));
            assert_eq!(get_map_field(&map_val, "day"), Some(Value::Int(11)));
            assert_eq!(get_map_field(&map_val, "hour"), Some(Value::Int(15)));
            assert_eq!(get_map_field(&map_val, "min"), Some(Value::Int(10)));
            assert_eq!(get_map_field(&map_val, "sec"), Some(Value::Int(45)));
            assert_eq!(get_map_field(&map_val, "wday"), Some(Value::Int(4))); // Wednesday = 4
        }
        other => panic!("Expected completed table map, got {:?}", other),
    }
}

#[test]
fn test_stdlib_os_sleep_nonblocking_under_sim_clock() {
    let start_instant = std::time::Instant::now();
    let mut vm = make_sim_vm(100, 0, 0);

    let os_sleep_fn = get_os_fn(&vm, "sleep");

    let mut func = Function::new("test_sleep_fiber".to_string(), 0, false);
    func.register_count = 8;
    let c_fn = func.chunk.add_constant(os_sleep_fn);
    // Request a 100-second sleep (would freeze test for 1m40s if blocking!)
    let c_dur = func.chunk.add_constant(Value::Float(100.0));
    let c_ret = func.chunk.add_constant(Value::Int(777));

    func.chunk.emit(Instruction::LoadConst(0, c_fn), 1);
    func.chunk.emit(Instruction::LoadConst(1, c_dur), 1);
    // Call os.sleep(100.0)
    func.chunk.emit(Instruction::Call(0, 1, 1, 2), 1);
    func.chunk.emit(Instruction::LoadConst(3, c_ret), 1);
    func.chunk.emit(Instruction::Return(3), 1);

    vm.load_functions(vec![func]);

    let mut scheduler = MinimalSimScheduler::new(vm);
    let handle = scheduler
        .spawn_fiber(Value::Function(0), vec![])
        .expect("spawn must succeed");

    // First step: fiber executes up to os.sleep(100.0) and yields with a timer
    let outcome1 = scheduler.step_fiber(handle, None).expect("step 1 must succeed");
    match outcome1 {
        StepOutcome::YieldedTimer(timer_id) => {
            assert!(timer_id.as_u64() > 0);
        }
        other => panic!("Expected YieldedTimer, got {:?}", other),
    }

    // Verify task state in VM
    let task = scheduler.vm.get_task_instance(handle).unwrap();
    assert_eq!(task.state, TaskState::Yielded);
    assert!(task.timer_id.is_some());

    // Host OS thread must NOT have been blocked!
    let elapsed_initial = start_instant.elapsed();
    assert!(
        elapsed_initial < Duration::from_millis(50),
        "os.sleep blocked OS thread! Elapsed: {:?}",
        elapsed_initial
    );

    // Verify next timer deadline is exactly 100 seconds (100 * 1e9 nanos)
    assert_eq!(
        scheduler.next_deadline(),
        Some(100 * 1_000_000_000),
        "Timer deadline must be 100 seconds"
    );

    // Advance virtual time to deadline and resume fiber
    let stepped = scheduler.advance_to_next_deadline();
    assert!(stepped);
    assert_eq!(scheduler.vm.clock.now_nanos(), 100 * 1_000_000_000);

    // Resume fiber
    let outcome2 = scheduler.step_fiber(handle, None).expect("step 2 must succeed");
    assert_eq!(outcome2, StepOutcome::Completed(Some(Value::Int(777))));

    let final_elapsed = start_instant.elapsed();
    assert!(
        final_elapsed < Duration::from_millis(100),
        "Full 100-second simulated sleep took {:?}, must be under 100ms!",
        final_elapsed
    );
}

#[test]
fn test_stdlib_os_sleep_zero_delay_nonblocking() {
    let mut vm = make_sim_vm(101, 1_000_000_000, 500_000_000);
    let os_sleep_fn = get_os_fn(&vm, "sleep");

    let mut func = Function::new("test_sleep_zero".to_string(), 0, false);
    func.register_count = 8;
    let c_fn = func.chunk.add_constant(os_sleep_fn);
    let c_dur = func.chunk.add_constant(Value::Float(0.0));
    let c_ret = func.chunk.add_constant(Value::Int(42));

    func.chunk.emit(Instruction::LoadConst(0, c_fn), 1);
    func.chunk.emit(Instruction::LoadConst(1, c_dur), 1);
    func.chunk.emit(Instruction::Call(0, 1, 1, 2), 1);
    func.chunk.emit(Instruction::LoadConst(3, c_ret), 1);
    func.chunk.emit(Instruction::Return(3), 1);

    vm.load_functions(vec![func]);

    let mut scheduler = MinimalSimScheduler::new(vm);
    let handle = scheduler
        .spawn_fiber(Value::Function(0), vec![])
        .expect("spawn must succeed");

    // First step: fiber executes os.sleep(0.0) and yields with a zero-delay timer
    let outcome1 = scheduler.step_fiber(handle, None).expect("step 1 must succeed");
    match outcome1 {
        StepOutcome::YieldedTimer(timer_id) => {
            assert!(timer_id.as_u64() > 0);
        }
        other => panic!("Expected YieldedTimer for os.sleep(0.0), got {:?}", other),
    }

    // Advance to next deadline: next == now, so advance_to_next_deadline polls expired timers and returns true
    let advanced = scheduler.advance_to_next_deadline();
    assert!(advanced, "advance_to_next_deadline must return true for zero-delay timer");

    // Resume fiber
    let outcome2 = scheduler.step_fiber(handle, None).expect("step 2 must succeed");
    assert_eq!(outcome2, StepOutcome::Completed(Some(Value::Int(42))));

    // Also verify end-to-end via run_task_with_virtual_time with os.sleep(0.0)
    let mut vm2 = make_sim_vm(102, 1_000_000_000, 500_000_000);
    let os_sleep_fn2 = get_os_fn(&vm2, "sleep");
    let mut func_e2e = Function::new("test_sleep_zero_e2e".to_string(), 0, false);
    func_e2e.register_count = 8;
    let c_fn2 = func_e2e.chunk.add_constant(os_sleep_fn2);
    let c_dur2 = func_e2e.chunk.add_constant(Value::Float(0.0));
    let c_ret2 = func_e2e.chunk.add_constant(Value::Int(99));
    func_e2e.chunk.emit(Instruction::LoadConst(0, c_fn2), 1);
    func_e2e.chunk.emit(Instruction::LoadConst(1, c_dur2), 1);
    func_e2e.chunk.emit(Instruction::Call(0, 1, 1, 2), 1);
    func_e2e.chunk.emit(Instruction::LoadConst(3, c_ret2), 1);
    func_e2e.chunk.emit(Instruction::Return(3), 1);
    vm2.load_functions(vec![func_e2e]);

    let mut scheduler2 = MinimalSimScheduler::new(vm2);
    let handle2 = scheduler2
        .spawn_fiber(Value::Function(0), vec![])
        .expect("spawn must succeed");
    let outcome_e2e = scheduler2
        .run_task_with_virtual_time(handle2, 10)
        .expect("run_task_with_virtual_time must not hang on os.sleep(0.0)");
    assert_eq!(outcome_e2e, StepOutcome::Completed(Some(Value::Int(99))));
}


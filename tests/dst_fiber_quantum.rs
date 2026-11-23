//! Gate 4 Acceptance Test: Fuel-based Fiber Preemption and FiberContinuation (D4, Gate 4)
//!
//! Validates:
//! 1. Fuel-based quantum instruction preemption:
//!    - Setting quantum to N instructions preempts the running fiber after exactly N instructions.
//!    - Fiber transitions to `TaskState::Yielded(YieldReason::QuantumExhausted)`.
//! 2. Exact state preservation:
//!    - Top frame's `ip` points to the next unexecuted instruction.
//!    - All register contents are preserved across preemption.
//!    - Full `FiberContinuation` is captured on the `TaskInstance`.
//! 3. Lossless resumption:
//!    - Resuming from `FiberContinuation` continues execution seamlessly.
//!    - Multiple quantum steps cleanly drive the fiber to `TaskState::Completed` with the correct result.
//! 4. Shared upvalues are preserved across preemption and resumption.

use lust::bytecode::{Function, Instruction, Upvalue, Value};
use lust::vm::{TaskState, VM, YieldReason};

fn create_accumulator_function(limit: i64) -> Function {
    let mut func = Function::new("accumulator".to_string(), 0, false);
    func.register_count = 8;
    let c_limit = func.chunk.add_constant(Value::Int(limit));
    let c_step = func.chunk.add_constant(Value::Int(1));
    let c_zero = func.chunk.add_constant(Value::Int(0));

    // Setup:
    // 0: R0 = 0 (accumulator)
    // 1: R1 = 1 (step)
    // 2: R2 = limit
    func.chunk.emit(Instruction::LoadConst(0, c_zero), 1);
    func.chunk.emit(Instruction::LoadConst(1, c_step), 1);
    func.chunk.emit(Instruction::LoadConst(2, c_limit), 1);

    // Loop head at ip = 3:
    // 3: R3 = R0 < R2
    // 4: JumpIfNot(R3, offset: +3) -> goes to 8 (Return)
    // 5: R0 = R0 + R1
    // 6: Jump(offset: -4) -> goes to 3
    // 7: Nop
    // 8: Return(R0)
    let loop_head = 3;
    func.chunk.emit(Instruction::Lt(3, 0, 2), 2);
    let exit_jump = func.chunk.emit(Instruction::JumpIfNot(3, 0), 2);
    func.chunk.emit(Instruction::Add(0, 0, 1), 3);
    let back_jump = func.chunk.emit(Instruction::Jump(0), 4);
    func.chunk.patch_jump(back_jump, loop_head);
    let loop_exit = func.chunk.instructions.len();
    func.chunk.patch_jump(exit_jump, loop_exit);
    func.chunk.emit(Instruction::Return(0), 5);

    func
}

#[test]
fn test_fuel_based_preemption_and_exact_quantum_accounting() {
    let mut vm = VM::with_seed(42);
    let func = create_accumulator_function(10);
    vm.load_functions(vec![func]);

    let handle = vm
        .create_task_value(Value::Function(0), vec![])
        .expect("spawn accumulator fiber");

    // Initial state: Ready
    let task_initial = vm.get_task_instance(handle).unwrap();
    assert_eq!(task_initial.state, TaskState::Ready);
    assert!(task_initial.continuation.is_none());

    // Step 1: Exactly 3 instructions (the setup instructions)
    let state1 = vm
        .resume_task_handle_with_quantum(handle, 3, None)
        .expect("quantum step 1 must succeed");
    assert_eq!(state1, TaskState::Yielded(YieldReason::QuantumExhausted));

    let task1 = vm.get_task_instance(handle).unwrap();
    assert_eq!(task1.state, TaskState::Yielded(YieldReason::QuantumExhausted));

    // Verify FiberContinuation captured
    let cont1 = task1.continuation.as_ref().expect("continuation captured");
    assert_eq!(cont1.ip, 3, "IP must point to loop head at instruction 3");
    assert_eq!(cont1.call_stack.len(), 1);
    let cont1_regs = cont1.call_stack[0].saved_registers().expect("saved registers captured");
    assert_eq!(cont1_regs[0], Value::Int(0)); // accumulator = 0
    assert_eq!(cont1_regs[1], Value::Int(1)); // step = 1
    assert_eq!(cont1_regs[2], Value::Int(10)); // limit = 10

    // Step 2: One full loop iteration (4 instructions: Lt, JumpIfNot, Add, Jump)
    let state2 = vm
        .resume_task_handle_with_quantum(handle, 4, None)
        .expect("quantum step 2 must succeed");
    assert_eq!(state2, TaskState::Yielded(YieldReason::QuantumExhausted));

    let task2 = vm.get_task_instance(handle).unwrap();
    let cont2 = task2.continuation.as_ref().unwrap();
    assert_eq!(cont2.ip, 3, "After 1 iteration, IP returns to loop head");
    assert_eq!(
        cont2.call_stack[0].saved_registers().expect("saved registers captured")[0],
        Value::Int(1),
        "Accumulator R0 must be 1"
    );

    // Step 3: Run with a larger quantum (e.g. 100 instructions) to complete the loop
    let state3 = vm
        .resume_task_handle_with_quantum(handle, 100, None)
        .expect("quantum step 3 must succeed");
    assert_eq!(state3, TaskState::Completed);

    let task3 = vm.get_task_instance(handle).unwrap();
    assert_eq!(task3.last_result, Some(Value::Int(10)));
}

#[test]
fn test_fiber_continuation_preservation_and_explicit_restore() {
    let mut vm = VM::with_seed(12345);
    let func = create_accumulator_function(20);
    vm.load_functions(vec![func]);

    let handle = vm
        .create_task_value(Value::Function(0), vec![])
        .expect("spawn fiber");

    // Execute 3 setup instructions + 3 loop iterations (4 * 3 = 12) = 15 instructions
    let state = vm
        .resume_task_handle_with_quantum(handle, 15, None)
        .expect("step 15 instructions");
    assert_eq!(state, TaskState::Yielded(YieldReason::QuantumExhausted));

    // Capture and extract the continuation
    let task = vm.get_task_instance_mut(handle).unwrap();
    let cont = task.take_continuation().expect("continuation present");
    assert_eq!(cont.ip, 3);
    assert_eq!(
        cont.call_stack[0].saved_registers().expect("saved registers captured")[0],
        Value::Int(3)
    );

    // Modify a register in the continuation (e.g. inject intermediate accumulator value 15)
    let mut modified_cont = cont.clone();
    modified_cont.call_stack[0]
        .saved_registers_mut()
        .expect("saved registers present")[0] = Value::Int(15);

    // Restore modified continuation and resume
    vm.resume_from_continuation(handle, modified_cont, None)
        .expect("resume from continuation");

    let task_final = vm.get_task_instance(handle).unwrap();
    assert_eq!(task_final.state, TaskState::Completed);
    // Since we set accumulator to 15, and limit is 20: 15..20 takes 5 iterations -> final is 20
    assert_eq!(task_final.last_result, Some(Value::Int(20)));
}

#[test]
fn test_fiber_continuation_preserves_shared_upvalues() {
    let mut vm = VM::with_seed(999);

    // Build a function that manipulates an upvalue:
    // U0: shared counter
    // Function body:
    // 0: LoadUpvalue(R0, 0)
    // 1: Add(R0, R0, R1)
    // 2: StoreUpvalue(0, R0)
    // 3: Return(R0)
    let mut func = Function::new("upvalue_worker".to_string(), 0, false);
    func.register_count = 4;
    let c_one = func.chunk.add_constant(Value::Int(1));
    func.chunk.emit(Instruction::LoadConst(1, c_one), 1);
    func.chunk.emit(Instruction::LoadUpvalue(0, 0), 1);
    func.chunk.emit(Instruction::Add(0, 0, 1), 1);
    func.chunk.emit(Instruction::StoreUpvalue(0, 0), 1);
    func.chunk.emit(Instruction::Return(0), 1);

    vm.load_functions(vec![func]);

    // Provide shared upvalue initialized to 50
    let upval = Upvalue::new(Value::Int(50));
    let closure = Value::Closure(std::rc::Rc::new(lust::bytecode::value::ClosureObject {
        function_idx: 0,
        upvalues: vec![upval.clone()],
    }));

    let handle = vm
        .create_task_value(closure, vec![])
        .expect("spawn upvalue worker");

    // Step with quantum = 3 (executes LoadConst, LoadUpvalue, Add) -> R0 becomes 51, before StoreUpvalue!
    let state = vm
        .resume_task_handle_with_quantum(handle, 3, None)
        .expect("step 3 ops");
    assert_eq!(state, TaskState::Yielded(YieldReason::QuantumExhausted));

    // Verify continuation captured shared upvalues
    let task = vm.get_task_instance(handle).unwrap();
    let cont = task.continuation.as_ref().unwrap();
    assert_eq!(cont.upvalues.len(), 1);
    assert_eq!(cont.upvalues[0].borrow().get(), Value::Int(50));
    assert_eq!(
        cont.call_stack[0].saved_registers().expect("saved registers captured")[0],
        Value::Int(51)
    );

    // Step remaining 2 instructions (StoreUpvalue, Return)
    let state2 = vm
        .resume_task_handle_with_quantum(handle, 5, None)
        .expect("finish remaining");
    assert_eq!(state2, TaskState::Completed);

    // The shared upvalue is now mutated to 51
    assert_eq!(upval.get(), Value::Int(51));
    let task_done = vm.get_task_instance(handle).unwrap();
    assert_eq!(task_done.last_result, Some(Value::Int(51)));
}

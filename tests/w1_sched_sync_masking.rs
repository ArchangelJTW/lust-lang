use lust::effect_abi::{FiberContext, FiberExecutionState, InterruptReason};
use lust::vm::{TaskId, TaskState, WaitKey, YieldReason, VM};

// INV-FIBER-02: interrupt masking defers cooperative cancellation/timeout until
// the outermost mask exits; resource exhaustion remains immediately terminal.
#[test]
fn inv_fiber_02_masking_and_park_wake_oracle() {
    let mut masked = FiberContext::new(11, 0);
    masked.enter_mask();
    masked.enter_mask();
    assert!(!masked.request_interrupt(InterruptReason::Cancelled));
    assert_eq!(masked.pending_interrupt, Some(InterruptReason::Cancelled));
    assert_eq!(masked.state, FiberExecutionState::Ready);
    assert_eq!(masked.exit_mask(), None);
    assert_eq!(masked.exit_mask(), Some(InterruptReason::Cancelled));
    assert!(masked.request_interrupt(InterruptReason::Cancelled));
    assert_eq!(masked.state, FiberExecutionState::Interrupted);
    assert!(masked.exit_mask().is_none());

    for reason in [InterruptReason::OutOfGas, InterruptReason::OutOfMemory] {
        let mut fiber = FiberContext::new(12, 0);
        fiber.enter_mask();
        assert!(fiber.request_interrupt(reason));
        assert_eq!(fiber.state, FiberExecutionState::Interrupted);
    }

    // Use real task instances so parking captures a continuation and only an
    // explicit wake can make a parked fiber runnable again.
    let mut tasks = VM::new();
    let key = WaitKey(7);
    let handles: Vec<_> = (0..3).map(|_| tasks.create_native_future_task()).collect();
    let ids: Vec<_> = handles.iter().map(|h| TaskId(h.0)).collect();
    for handle in &handles {
        assert!(tasks.park_task(TaskId(handle.0), key));
        let task = tasks.get_task_instance(*handle).unwrap();
        assert_eq!(task.state, TaskState::Yielded(YieldReason::Parked(key)));
        assert!(task.continuation().is_some());
    }
    assert_eq!(tasks.wake_one(key).map(|h| TaskId(h.0)), Some(ids[0]));
    assert_eq!(tasks.get_task_instance(handles[0]).unwrap().state, TaskState::Ready);
    assert_eq!(tasks.get_task_instance(handles[1]).unwrap().state, TaskState::Yielded(YieldReason::Parked(key)));
    assert_eq!(tasks.wake_all(key).into_iter().map(|h| TaskId(h.0)).collect::<Vec<_>>(), vec![ids[1], ids[2]]);
    assert!(tasks.wake_all(key).is_empty());
}

#[test]
fn deterministic_seeded_simulation_replays_interleavings() {
    use lust::{embed::EmbeddedProgram, sim::PreemptiveExplorationScheduler};

    // Empty functions still create real scheduler work; step traces include
    // chosen fiber, quantum, and resulting state, not merely seed metadata.
    let source = "function worker()\nend";
    let seeds = [0, 1, 7, 42, u64::MAX];
    let mut observed = Vec::new();
    for seed in seeds {
        let mut scheduler = PreemptiveExplorationScheduler::with_seed(seed).with_max_quantum(5);
        let mut program = EmbeddedProgram::builder().entry_module("main").module("main", source).compile().expect("compile worker");
        let function = program.function_handle("worker").expect("worker function").as_value().clone();
        scheduler.vm = core::mem::replace(program.vm_mut(), VM::new());
        scheduler.spawn_fiber(function.clone(), Vec::new()).expect("spawn first");
        scheduler.spawn_fiber(function, Vec::new()).expect("spawn second");
        let mut trace = Vec::new();
        while let Some((handle, state)) = scheduler.step_one().expect("scheduler step") {
            trace.push((handle, state));
        }
        assert_eq!(trace.len(), 2, "seed {seed} should execute both fibers");
        observed.push(trace);
    }

    for (seed, expected) in seeds.into_iter().zip(observed) {
        let mut replay = PreemptiveExplorationScheduler::with_seed(seed).with_max_quantum(5);
        let mut program = EmbeddedProgram::builder().entry_module("main").module("main", source).compile().expect("compile worker");
        let function = program.function_handle("worker").expect("worker function").as_value().clone();
        replay.vm = core::mem::replace(program.vm_mut(), VM::new());
        replay.spawn_fiber(function.clone(), Vec::new()).expect("spawn first");
        replay.spawn_fiber(function, Vec::new()).expect("spawn second");
        let mut actual = Vec::new();
        while let Some((handle, state)) = replay.step_one().expect("scheduler step") {
            actual.push((handle, state));
        }
        assert_eq!(actual, expected, "seed {seed} must replay identically");
    }
}

#[test]
fn test_waiter_registration_cancellation_cleanup_and_race_safety() {
    let mut vm = VM::new();
    let key = vm.next_wait_key();
    let h1 = vm.create_native_future_task();
    let h2 = vm.create_native_future_task();
    let h3 = vm.create_native_future_task();
    let id1 = TaskId(h1.0);
    let id2 = TaskId(h2.0);
    let id3 = TaskId(h3.0);

    // 1. Register all three tasks as waiting on `key`
    assert!(vm.park_task(id1, key));
    assert!(vm.park_task(id2, key));
    assert!(vm.park_task(id3, key));
    assert!(vm.is_task_waiting(id1));
    assert!(vm.is_task_waiting(id2));
    assert!(vm.is_task_waiting(id3));

    // 2. Cancellation cleanup: cancel task 2 while waiting
    vm.cancel_task(id2, InterruptReason::Cancelled);
    let t2 = vm.get_task_instance(h2).unwrap();
    assert_eq!(t2.state, TaskState::Stopped);
    // Task 2 must be unregistered from waiters
    assert!(!vm.is_task_waiting(id2));
    assert!(vm.is_task_waiting(id1));
    assert!(vm.is_task_waiting(id3));

    // 3. Waking: wake_one should wake task 1 (FIFO)
    let woken1 = vm.wake_one(key);
    assert_eq!(woken1, Some(id1));
    assert_eq!(vm.get_task_instance(h1).unwrap().state, TaskState::Ready);
    assert!(!vm.is_task_waiting(id1));

    // 4. Next wake_one must skip cancelled task 2 and wake task 3 without lost wakeup!
    let woken2 = vm.wake_one(key);
    assert_eq!(woken2, Some(id3));
    assert_eq!(vm.get_task_instance(h3).unwrap().state, TaskState::Ready);
    assert!(!vm.is_task_waiting(id3));

    // No more waiters left
    assert_eq!(vm.wake_one(key), None);
    assert!(vm.wake_all(key).is_empty());

    // 5. Verify waking a stopped task can never restore it to Ready
    assert!(!vm.park_task(id2, key), "terminal task cannot be parked");
    let woken_empty = vm.wake_all(key);
    assert!(woken_empty.is_empty());
    assert_eq!(vm.get_task_instance(h2).unwrap().state, TaskState::Stopped);
}

#[test]
fn test_inv_mask_02_budget_interrupt_non_maskable_preempts_before_instruction() {
    for reason in [InterruptReason::OutOfGas, InterruptReason::OutOfMemory] {
        let mut vm = VM::new();
        let id = vm.task_manager_mut().next_id();
        let mut task = lust::vm::TaskInstance::new_native_future(id);
        task.state = TaskState::Running;
        // Enter mask
        task.enter_mask();
        assert!(task.fiber_context.is_masked());

        // Request non-maskable budget interrupt while masked
        task.request_interrupt(reason);
        // While running, dispatch safe-point will poll it. State is Still Running until poll_interrupt.
        assert_eq!(task.state, TaskState::Running);

        vm.task_manager_mut().insert(task);
        vm.set_current_task_id(Some(id));

        // Before any VM instruction executes, dispatch-loop poll_interrupt delivers the budget interrupt!
        let run_res = vm.run();
        assert!(run_res.is_err(), "budget interrupt must immediately error in dispatch");

        let task_after = vm.task_manager().get(id).unwrap();
        assert_eq!(task_after.state, TaskState::Stopped, "INV-MASK-02: must transition to Stopped");
        assert_eq!(task_after.fiber_context.state, FiberExecutionState::Interrupted);
        assert!(task_after.error.as_ref().unwrap().to_string().contains(&format!("{:?}", reason)));
    }

    // Contrast with maskable interrupt (Cancelled/Timeout) while masked:
    {
        let mut vm = VM::new();
        let id = vm.task_manager_mut().next_id();
        let mut task = lust::vm::TaskInstance::new_native_future(id);
        task.state = TaskState::Running;
        task.enter_mask();
        task.request_interrupt(InterruptReason::Cancelled);
        // Cancelled is deferred while masked!
        assert_eq!(task.fiber_context.pending_interrupt, Some(InterruptReason::Cancelled));
        assert_eq!(task.fiber_context.state, FiberExecutionState::Ready);
    }
}

#[test]
fn test_inv_mask_04_mask_underflow_safety() {
    let mut ctx = FiberContext::new(77, 0);
    assert_eq!(ctx.interrupt_mask_depth, 0);
    assert!(!ctx.is_masked());

    // Underflow call 1: exit_mask when depth is already 0
    let res1 = ctx.exit_mask();
    assert_eq!(res1, None);
    assert_eq!(ctx.interrupt_mask_depth, 0, "saturating_sub prevents underflow below 0");
    assert!(!ctx.is_masked());

    // Underflow call 2: repeated exit_mask
    let res2 = ctx.exit_mask();
    assert_eq!(res2, None);
    assert_eq!(ctx.interrupt_mask_depth, 0);

    // Normal mask entry and exit
    ctx.enter_mask();
    assert_eq!(ctx.interrupt_mask_depth, 1);
    assert!(ctx.is_masked());

    let res3 = ctx.exit_mask();
    assert_eq!(res3, None);
    assert_eq!(ctx.interrupt_mask_depth, 0);
    assert!(!ctx.is_masked());

    // Underflow call 3 after normal cycle
    let res4 = ctx.exit_mask();
    assert_eq!(res4, None);
    assert_eq!(ctx.interrupt_mask_depth, 0);

    // TaskInstance level underflow check
    let mut task = lust::vm::TaskInstance::new_native_future(TaskId(100));
    assert_eq!(task.fiber_context.interrupt_mask_depth, 0);
    assert_eq!(task.exit_mask(), None);
    assert_eq!(task.fiber_context.interrupt_mask_depth, 0);
}

#[test]
fn test_sync_primitives_end_to_end() {
    use lust::effect_abi::sync::{BoundedQueue, Deferred, Latch, QueueError, Ref, Semaphore};
    use lust::effect_abi::Exit;

    let mut vm = VM::new();

    // 1. Ref<T>
    let mut r = Ref::new(10);
    assert_eq!(r.get(), 10);
    r.set(20);
    assert_eq!(r.get(), 20);
    assert_eq!(r.update(|x| x + 5), 25);
    assert_eq!(r.get(), 25);
    let (prev_flag, new_val) = r.modify(|x| (x > 20, x * 2));
    assert!(prev_flag);
    assert_eq!(new_val, 50);
    assert_eq!(r.get(), 50);

    // 2. Deferred<T, E>
    let def_key = vm.next_wait_key();
    let mut def = Deferred::<i64, String>::new(def_key);
    assert!(!def.is_completed());
    assert_eq!(def.poll(), None);

    let h_w1 = vm.create_native_future_task();
    let h_w2 = vm.create_native_future_task();
    let id_w1 = TaskId(h_w1.0);
    let id_w2 = TaskId(h_w2.0);

    // Both await and park on uncompleted Deferred
    assert_eq!(def.await_or_park_vm(id_w1, &mut vm), Err(def_key));
    assert_eq!(def.await_or_park_vm(id_w2, &mut vm), Err(def_key));
    assert_eq!(vm.get_task_instance(h_w1).unwrap().state, TaskState::Yielded(YieldReason::Parked(def_key)));
    assert_eq!(vm.get_task_instance(h_w2).unwrap().state, TaskState::Yielded(YieldReason::Parked(def_key)));

    // Complete and wake all waiters
    let (completed, woken) = def.complete_and_wake(Exit::Success(42), &mut vm);
    assert!(completed);
    assert_eq!(woken.len(), 2);
    assert_eq!(vm.get_task_instance(h_w1).unwrap().state, TaskState::Ready);
    assert_eq!(vm.get_task_instance(h_w2).unwrap().state, TaskState::Ready);
    assert!(def.is_completed());
    assert_eq!(def.poll(), Some(&Exit::Success(42)));

    // Subsequent await on already-completed returns value immediately
    let h_w3 = vm.create_native_future_task();
    let id_w3 = TaskId(h_w3.0);
    assert_eq!(def.await_or_park_vm(id_w3, &mut vm), Ok(Some(&Exit::Success(42))));
    assert_eq!(vm.get_task_instance(h_w3).unwrap().state, TaskState::Yielded(YieldReason::Explicit));

    // Deferred single-assignment: second complete is rejected
    assert!(!def.complete(Exit::Success(99)));
    assert_eq!(def.poll(), Some(&Exit::Success(42)));

    // 3. Semaphore with cancellation and with_permit
    let sem_key = vm.next_wait_key();
    let mut sem = Semaphore::with_key(2, sem_key);
    assert_eq!(sem.available_permits(), 2);
    assert!(sem.try_acquire(1));
    assert_eq!(sem.available_permits(), 1);
    assert!(sem.try_acquire(1));
    assert_eq!(sem.available_permits(), 0);
    assert!(!sem.try_acquire(1));

    // Park waiter 1 and waiter 2
    let s_w1 = vm.create_native_future_task();
    let s_w2 = vm.create_native_future_task();
    let s_id1 = TaskId(s_w1.0);
    let s_id2 = TaskId(s_w2.0);

    assert!(!sem.acquire_or_park_vm(s_id1, 1, &mut vm));
    assert!(!sem.acquire_or_park_vm(s_id2, 1, &mut vm));
    assert!(vm.is_task_waiting(s_id1));
    assert!(vm.is_task_waiting(s_id2));

    // Cancel waiter 1 while waiting
    sem.cancel_waiter_vm(s_id1, &mut vm);
    assert!(!vm.is_task_waiting(s_id1));

    // Release 1 permit: must NOT leak permit to cancelled waiter 1; must wake waiter 2!
    let woken_sem = sem.release_vm(1, &mut vm);
    assert_eq!(woken_sem, vec![s_id2]);
    assert_eq!(vm.get_task_instance(s_w2).unwrap().state, TaskState::Ready);
    assert_eq!(sem.available_permits(), 0);

    // with_permit RAII guard
    sem.release_vm(1, &mut vm);
    assert_eq!(sem.available_permits(), 1);
    let res = sem.with_permit(1, |s| {
        assert_eq!(s.available_permits(), 0);
        "success"
    });
    assert_eq!(res, Some("success"));
    assert_eq!(sem.available_permits(), 1, "permit released after guard scope");

    // 4. BoundedQueue<T> offer/take/cancel/close
    let offer_k = vm.next_wait_key();
    let take_k = vm.next_wait_key();
    let mut q = BoundedQueue::with_keys(2, offer_k, take_k);
    assert_eq!(q.len(), 0);
    assert!(q.offer_vm(100, &mut vm).unwrap());
    assert!(q.offer_vm(200, &mut vm).unwrap());
    assert!(q.is_full());

    // Queue is full: producer parks
    let q_p1 = vm.create_native_future_task();
    let q_pid1 = TaskId(q_p1.0);
    assert!(!q.offer_or_park_vm(q_pid1, 300, &mut vm).unwrap());
    assert_eq!(vm.get_task_instance(q_p1).unwrap().state, TaskState::Yielded(YieldReason::Parked(offer_k)));

    // Consumer takes an element: frees space and wakes producer
    let val1 = q.take_vm(&mut vm).unwrap();
    assert_eq!(val1, Some(100));
    assert_eq!(vm.get_task_instance(q_p1).unwrap().state, TaskState::Ready);

    // Queue consumer park when empty
    let val2 = q.take_vm(&mut vm).unwrap();
    assert_eq!(val2, Some(200));
    assert!(q.is_empty());

    let q_c1 = vm.create_native_future_task();
    let q_cid1 = TaskId(q_c1.0);
    assert_eq!(q.take_or_park_vm(q_cid1, &mut vm).unwrap(), None);
    assert_eq!(vm.get_task_instance(q_c1).unwrap().state, TaskState::Yielded(YieldReason::Parked(take_k)));

    // Cancel consumer waiter
    q.cancel_waiter_vm(q_cid1, &mut vm);
    assert!(!vm.is_task_waiting(q_cid1));

    // Offer after cancellation: item stays in queue without being lost to cancelled consumer
    assert!(q.offer_vm(400, &mut vm).unwrap());
    assert_eq!(q.len(), 1);
    assert_eq!(q.take_vm(&mut vm).unwrap(), Some(400));

    // Close queue with parked waiters
    let q_p2 = vm.create_native_future_task();
    let q_c2 = vm.create_native_future_task();
    let q_pid2 = TaskId(q_p2.0);
    let q_cid2 = TaskId(q_c2.0);
    q.offer_vm(1, &mut vm).unwrap();
    q.offer_vm(2, &mut vm).unwrap();
    q.offer_or_park_vm(q_pid2, 3, &mut vm).unwrap();
    let val = q.take_vm(&mut vm).unwrap();
    assert_eq!(val, Some(1));
    let _ = q.take_vm(&mut vm).unwrap();
    let _ = q.take_or_park_vm(q_cid2, &mut vm).unwrap();

    let woken_close = q.close_vm(&mut vm);
    assert!(q.is_closed());
    assert!(woken_close.contains(&q_cid2));
    assert_eq!(q.offer_vm(999, &mut vm), Err(QueueError::Closed));
    assert_eq!(q.take_vm(&mut vm), Err(QueueError::Closed));

    // 5. Latch
    let latch_k = vm.next_wait_key();
    let mut latch = Latch::with_key(2, latch_k);
    assert!(!latch.is_open());
    let l_t1 = vm.create_native_future_task();
    let l_id1 = TaskId(l_t1.0);
    assert!(!latch.wait_or_park_vm(l_id1, &mut vm));
    assert_eq!(vm.get_task_instance(l_t1).unwrap().state, TaskState::Yielded(YieldReason::Parked(latch_k)));

    let woken_l1 = latch.count_down_vm(&mut vm);
    assert!(woken_l1.is_empty());
    assert!(!latch.is_open());

    let woken_l2 = latch.count_down_vm(&mut vm);
    assert_eq!(woken_l2, vec![l_id1]);
    assert!(latch.is_open());
    assert_eq!(vm.get_task_instance(l_t1).unwrap().state, TaskState::Ready);
    assert!(latch.wait_or_park_vm(l_id1, &mut vm));
}

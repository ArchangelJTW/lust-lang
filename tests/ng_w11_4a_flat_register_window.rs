//! NG-W11-4A Acceptance & Regression Suite:
//! Flat VM Register Window (`vm_registers: Vec<Value, VmAllocHandle>`),
//! compact `CallFrame` (`<= 96` bytes), `non_plain_mask` selective Rc cleanup,
//! Hazard 10 deopt/Cranelift top-frame window expansion, and fiber suspend/resume isolation.

use std::mem::size_of;
use std::rc::Rc;

use lust::bytecode::{Function, Instruction, Value};
use lust::intern::Interner;
use lust::lexer::Lexer;
#[cfg(feature = "test-deopt")]
use lust::mir::ir::MirModule;
#[cfg(feature = "test-deopt")]
use lust::mir::lowering::AstLowering;
use lust::parser::Parser;
use lust::vm::{CallFrame, TaskState, VmBackend, VM, YieldReason};

fn compile_and_load_vm(source: &str) -> VM {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize().expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");
    let mut compiler = lust::Compiler::new();
    let functions = compiler.compile_module(&items).expect("compile_module");
    let mut vm = VM::new();
    vm.load_functions(functions);
    vm
}

#[test]
fn test_call_frame_size_is_at_most_96_bytes() {
    let sz = size_of::<CallFrame>();
    assert!(
        sz <= 96,
        "CallFrame must be <= 96 bytes (got {sz} bytes)"
    );
}

#[test]
fn test_vm_registers_budgeted_by_vm_alloc_handle_and_refunded_on_drop() {
    let alloc_handle = {
        let mut vm = VM::new();
        let handle = vm.vm_alloc().clone();

        // Pre-allocated with capacity >= 4096 backed by VmAllocHandle.
        let init_cap = vm.vm_registers_capacity();
        assert!(
            init_cap >= 4096,
            "vm_registers must be pre-allocated with capacity >= 4096 (got {init_cap})"
        );
        let init_used = handle.inner().used_bytes();
        let expected_min_bytes = 4096 * size_of::<Value>();
        assert!(
            init_used >= expected_min_bytes,
            "VmAllocHandle must track vm_registers backing buffer (>= {expected_min_bytes} B, got {init_used} B)"
        );

        // Push enough frames to grow vm_registers past initial 4096 slots (e.g. 20 * 256 = 5120).
        for i in 0..20 {
            let idx = vm.push_call_frame(0, 256, None);
            assert_eq!(idx, i);
            vm.frame_registers_mut(idx)[0] = Value::Int(i as i64);
        }
        assert!(vm.vm_registers_len() >= 5120);
        let grown_used = handle.inner().used_bytes();
        assert!(
            grown_used > init_used,
            "VmAllocHandle used_bytes must increase when vm_registers grows (before={init_used}, after={grown_used})"
        );
        assert!(handle.inner().peak_bytes() >= grown_used);

        // Pop all frames cleanly.
        for _ in 0..20 {
            assert!(vm.pop_call_frame_public().is_some());
            assert_eq!(
                vm.last_frame_pop_cleanup_iters(),
                0,
                "Pure-scalar frames must perform 0 cleanup iterations on pop"
            );
        }

        handle
    };

    // After VM drops, vm_registers buffer must be fully refunded to 0 bytes.
    assert_eq!(
        alloc_handle.inner().used_bytes(),
        0,
        "VmAllocHandle used_bytes must return to 0 after VM drop"
    );
}

#[test]
fn test_non_plain_mask_zero_iters_on_scalar_and_exact_rc_cleanup_on_non_plain() {
    let mut vm = VM::new();

    // 1. Pure-scalar frame: write Int/Float/Bool/Nil -> non_plain_mask == 0 -> 0 cleanup iterations.
    let scalar_idx = vm.push_call_frame(0, 16, None);
    vm.write_frame_register(scalar_idx, 0, Value::Int(42));
    vm.write_frame_register(scalar_idx, 1, Value::Float(3.5));
    vm.write_frame_register(scalar_idx, 2, Value::Bool(true));
    vm.write_frame_register(scalar_idx, 3, Value::Nil);
    assert_eq!(vm.top_frame_registers()[0], Value::Int(42));
    let popped_scalar = vm.pop_call_frame_public().expect("pop scalar frame");
    assert_eq!(popped_scalar.non_plain_mask, 0);
    assert_eq!(
        vm.last_frame_pop_cleanup_iters(),
        0,
        "Pure-scalar frame pop MUST perform 0 cleanup iterations"
    );

    // 2. Low-slot non-plain frame: write 2 Rc<String> values into slots 2 and 17.
    let rc_a = Rc::new("alpha_owned".to_string());
    let rc_b = Rc::new("beta_owned".to_string());
    let low_idx = vm.push_call_frame(0, 32, None);
    vm.write_frame_register(low_idx, 0, Value::Int(7));
    vm.write_frame_register(low_idx, 2, Value::String(Rc::clone(&rc_a)));
    vm.write_frame_register(low_idx, 17, Value::String(Rc::clone(&rc_b)));
    assert_eq!(Rc::strong_count(&rc_a), 2);
    assert_eq!(Rc::strong_count(&rc_b), 2);

    let popped_low = vm.pop_call_frame_public().expect("pop low non-plain frame");
    assert_eq!(popped_low.non_plain_mask, (1u64 << 2) | (1u64 << 17));
    assert_eq!(
        vm.last_frame_pop_cleanup_iters(),
        2,
        "Only the 2 marked non-plain slots must be cleared"
    );
    assert_eq!(
        Rc::strong_count(&rc_a),
        1,
        "Rc in slot 2 must be dropped immediately on frame pop"
    );
    assert_eq!(
        Rc::strong_count(&rc_b),
        1,
        "Rc in slot 17 must be dropped immediately on frame pop"
    );

    // 3. Overflow-slot (>= 64) non-plain frame: write Rc<String> into slot 70 (sets bit 63).
    let rc_hi = Rc::new("high_slot_owned".to_string());
    let hi_idx = vm.push_call_frame(0, 80, None);
    vm.write_frame_register(hi_idx, 70, Value::String(Rc::clone(&rc_hi)));
    assert_eq!(Rc::strong_count(&rc_hi), 2);

    let popped_hi = vm.pop_call_frame_public().expect("pop high non-plain frame");
    assert_ne!(popped_hi.non_plain_mask & (1u64 << 63), 0);
    assert!(vm.last_frame_pop_cleanup_iters() > 0);
    assert_eq!(
        Rc::strong_count(&rc_hi),
        1,
        "Rc in overflow slot 70 (>= 64) must be dropped immediately on frame pop"
    );
}

#[test]
fn test_bytecode_execution_scalar_vs_heap_frame_cleanup_and_recursion() {
    let source = r#"
        function sum_scalar(n: int): int
            local acc: int = 0
            local i: int = 0
            while i < n do
                acc = acc + i
                i = i + 1
            end
            return acc
        end

        function fib(n: int): int
            if n <= 1 then
                return n
            end
            return fib(n - 1) + fib(n - 2)
        end

        function build_str(a: string, b: string): string
            local c: string = a + "_" + b
            return c
        end
    "#;

    let mut vm = compile_and_load_vm(source);
    vm.set_backend(VmBackend::Bytecode);

    let res = vm.call("sum_scalar", vec![Value::Int(100)]).expect("sum_scalar");
    assert_eq!(res, Value::Int(4950));
    assert_eq!(
        vm.last_frame_pop_cleanup_iters(),
        0,
        "Pure-scalar Lust function return must perform 0 cleanup iterations"
    );

    let fib_res = vm.call("fib", vec![Value::Int(15)]).expect("fib(15)");
    assert_eq!(fib_res, Value::Int(610));
    assert_eq!(vm.call_stack_depth(), 0);
    assert_eq!(vm.last_frame_pop_cleanup_iters(), 0);

    let str_res = vm
        .call("build_str", vec![Value::string("hello"), Value::string("world")])
        .expect("build_str");
    assert_eq!(str_res, Value::string("hello_world"));
    assert!(
        vm.last_frame_pop_cleanup_iters() > 0,
        "Heap-string Lust function return must clear its non-plain slots"
    );
}

#[test]
fn test_interleaved_fibers_isolate_register_windows_across_suspend_resume() {
    let mut vm = VM::with_seed(77);

    // Build a simple accumulator function taking (start, limit) -> sum
    let mut func = Function::new("fiber_acc".to_string(), 2, false);
    func.register_count = 6;
    let c_one = func.chunk.add_constant(Value::Int(1));
    // R0 = start (acc), R1 = limit, R2 = step (1), R3 = cond
    func.chunk.emit(Instruction::LoadConst(2, c_one), 1);
    let loop_head = func.chunk.instructions.len();
    func.chunk.emit(Instruction::Lt(3, 0, 1), 2);
    let exit_j = func.chunk.emit(Instruction::JumpIfNot(3, 0), 2);
    func.chunk.emit(Instruction::Add(0, 0, 2), 3);
    let back_j = func.chunk.emit(Instruction::Jump(0), 4);
    func.chunk.patch_jump(back_j, loop_head);
    let exit_ip = func.chunk.instructions.len();
    func.chunk.patch_jump(exit_j, exit_ip);
    func.chunk.emit(Instruction::Return(0), 5);
    vm.load_functions(vec![func]);

    let f1 = vm
        .create_task_value(Value::Function(0), vec![Value::Int(10), Value::Int(15)])
        .expect("create fiber 1");
    let f2 = vm
        .create_task_value(Value::Function(0), vec![Value::Int(100), Value::Int(104)])
        .expect("create fiber 2");

    // Step fiber 1 by 5 ops (1 setup + 1 iteration -> R0 = 11)
    assert_eq!(
        vm.resume_task_handle_with_quantum(f1, 5, None).unwrap(),
        TaskState::Yielded(YieldReason::QuantumExhausted)
    );
    // Step fiber 2 by 5 ops (1 setup + 1 iteration -> R0 = 101)
    assert_eq!(
        vm.resume_task_handle_with_quantum(f2, 5, None).unwrap(),
        TaskState::Yielded(YieldReason::QuantumExhausted)
    );

    // Verify saved register snapshots remain strictly isolated
    let c1 = vm.get_task_instance(f1).unwrap().continuation.as_ref().unwrap();
    let c2 = vm.get_task_instance(f2).unwrap().continuation.as_ref().unwrap();
    assert_eq!(c1.call_stack[0].saved_registers().unwrap()[0], Value::Int(11));
    assert_eq!(c2.call_stack[0].saved_registers().unwrap()[0], Value::Int(101));

    // Finish both fibers in interleaved order
    assert_eq!(
        vm.resume_task_handle_with_quantum(f1, 100, None).unwrap(),
        TaskState::Completed
    );
    assert_eq!(
        vm.resume_task_handle_with_quantum(f2, 100, None).unwrap(),
        TaskState::Completed
    );
    assert_eq!(vm.get_task_instance(f1).unwrap().last_result, Some(Value::Int(15)));
    assert_eq!(vm.get_task_instance(f2).unwrap().last_result, Some(Value::Int(104)));
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_hazard_10_cranelift_deopt_expands_undersized_top_frame_window() {
    let source = r#"
        function many_locals(n: int): int
            local a: int = n + 1
            local b: int = a + 2
            local c: int = b + 3
            local d: int = c + 4
            local e: int = d + 5
            local f: int = e + 6
            local g: int = f + 7
            local i: int = 0
            local acc: int = g
            while i < n do
                acc = acc + i
                i = i + 1
            end
            return acc
        end
    "#;

    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize().expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");
    let mut module = MirModule::new("hazard10_expand".into());
    for item in &items {
        if let lust::ast::ItemKind::Function(func_def) = &item.kind {
            let mir_fn = AstLowering::lower_function(func_def).expect("lower MIR");
            module.add_function(mir_fn);
        }
    }

    let mut vm_clif = VM::new();
    vm_clif.load_mir_module(&module).expect("load_mir_module");
    vm_clif.set_backend(VmBackend::Cranelift);

    let mut vm_bc = VM::new();
    vm_bc.load_mir_module(&module).expect("load_mir_module");
    vm_bc.set_backend(VmBackend::Bytecode);
    let expected = vm_bc.call("many_locals", vec![Value::Int(10)]).expect("oracle");

    // Push an intentionally undersized top frame (reg_count = 2, smaller than MIR locals)
    // and invoke run_cranelift(): Hazard 10 must expand the top frame window cleanly.
    let mut small_frame = CallFrame::new_detached(0, 0, 2);
    small_frame.saved_registers_mut().unwrap()[0] = Value::Int(10);
    vm_clif.push_call_frame_for_test(small_frame);
    let actual = vm_clif
        .run_cranelift()
        .expect("run_cranelift must expand undersized top frame window cleanly");
    assert_eq!(actual, expected);
}

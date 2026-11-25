use std::rc::Rc;

use lust::LustConfig;
use lust::ast::{Span, Type, TypeKind};
use lust::bytecode::{
    FieldStorage, Function, Instruction, OpCode, StructLayout, StructObject, Value,
};
use lust::typechecker::FunctionSignature;
use lust::vm::{TaskState, VM, VmBackend, YieldReason};

fn bytecode_vm(functions: Vec<Function>) -> VM {
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    let mut vm = VM::with_config(&config);
    vm.set_backend(VmBackend::Bytecode);
    vm.load_functions(functions);
    vm
}

fn integer_signature(param_count: usize) -> FunctionSignature {
    let int = Type::new(TypeKind::Int, Span::dummy());
    FunctionSignature {
        params: vec![int.clone(); param_count],
        return_type: int,
        is_method: false,
        type_params: Vec::new(),
        trait_bounds: Vec::new(),
        effects: None,
    }
}

fn recursive_sum() -> Function {
    let mut function = Function::new("sum", 1, false);
    function.set_register_count(255);
    function.set_signature(integer_signature(1));
    let chunk = &mut function.chunk;
    let zero = chunk.add_constant(Value::Int(0));
    let recurse = chunk.add_constant(Value::Function(0));
    chunk.emit(Instruction::AddIntImm(254, 0, 100), 1);
    chunk.emit(Instruction::LoadConst(1, zero), 1);
    chunk.emit(Instruction::EqInt(2, 0, 1), 1);
    let base_case = chunk.emit(Instruction::JumpIf(2, 0), 1);
    chunk.emit(Instruction::LoadConst(3, recurse), 1);
    chunk.emit(Instruction::AddIntImm(4, 0, -1), 1);
    chunk.emit(Instruction::Call(3, 4, 1, 5), 1);
    chunk.emit(Instruction::AddInt(254, 254, 5), 1);
    let ret = chunk.emit(Instruction::Return(254), 1);
    chunk.patch_jump(base_case, ret);
    function
}

#[test]
fn test_ng_w14_3_nested_frames_stride_oom_and_fiber_isolation() {
    let function = recursive_sum();
    let mut vm = bytecode_vm(vec![function.clone()]);
    assert_eq!(
        vm.call("sum", vec![Value::Int(15)]).unwrap(),
        Value::Int(1720)
    );
    assert_eq!(vm.vm_registers_len(), 16 * 256);
    assert_eq!(vm.call_stack_depth(), 0);

    let alloc = vm.vm_alloc().clone();
    alloc.inner().set_limit_bytes(alloc.inner().used_bytes());
    let error = vm.call("sum", vec![Value::Int(16)]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Out of memory allocating VM register window")
    );
    assert_eq!(vm.call_stack_depth(), 0);
    assert_eq!(vm.vm_registers_len(), 16 * 256);
    alloc.inner().set_limit_bytes(usize::MAX);

    vm.set_memory_budget_bytes(0).unwrap();
    let error = vm.call("sum", vec![Value::Int(16)]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Out of memory allocating VM register window")
    );
    assert_eq!(vm.call_stack_depth(), 0);
    vm.clear_memory_budget().unwrap();
    assert_eq!(
        vm.call("sum", vec![Value::Int(16)]).unwrap(),
        Value::Int(1836)
    );

    let mut fiber_vm = bytecode_vm(vec![function]);
    for (n, expected) in [(3, 406), (7, 828)] {
        let task = fiber_vm
            .create_task_value(Value::Function(0), vec![Value::Int(n)])
            .unwrap();
        // Seven instructions reach the nested call. The eighth sets R254 in the child.
        assert_eq!(
            fiber_vm
                .resume_task_handle_with_quantum(task, 8, None)
                .unwrap(),
            TaskState::Yielded(YieldReason::QuantumExhausted)
        );
        let saved = &fiber_vm
            .get_task_instance(task)
            .unwrap()
            .continuation
            .as_ref()
            .unwrap()
            .call_stack;
        assert_eq!(saved.len(), 2);
        assert_eq!(saved[1].base_reg, saved[0].base_reg + 256);
        assert_eq!(
            saved[0].saved_registers().unwrap()[254],
            Value::Int(n + 100)
        );
        assert_eq!(saved[1].saved_registers().unwrap()[254], Value::Int(n + 99));

        // A second suspension forces the dense restore path with both frames live.
        assert_eq!(
            fiber_vm
                .resume_task_handle_with_quantum(task, 1, None)
                .unwrap(),
            TaskState::Yielded(YieldReason::QuantumExhausted)
        );
        let saved = &fiber_vm
            .get_task_instance(task)
            .unwrap()
            .continuation
            .as_ref()
            .unwrap()
            .call_stack;
        assert_eq!(saved[1].base_reg, saved[0].base_reg + 256);
        assert_eq!(
            saved[0].saved_registers().unwrap()[254],
            Value::Int(n + 100)
        );
        assert_eq!(saved[1].saved_registers().unwrap()[254], Value::Int(n + 99));

        assert_eq!(
            fiber_vm
                .resume_task_handle_with_quantum(task, 100, None)
                .unwrap(),
            TaskState::Completed
        );
        assert_eq!(
            fiber_vm.get_task_instance(task).unwrap().last_result,
            Some(Value::Int(expected))
        );
    }
}

#[test]
fn test_ng_w14_3_edge_bb_gas_exact_sec01() {
    let mut callee = Function::new("identity", 1, false);
    callee.set_register_count(1);
    callee.set_signature(integer_signature(1));
    callee.chunk.emit(Instruction::Return(0), 1);

    let mut function = Function::new("run", 1, false);
    function.set_register_count(4);
    let chunk = &mut function.chunk;
    let zero = chunk.add_constant(Value::Int(0));
    let identity = chunk.add_constant(Value::Function(1));
    let global = chunk.add_constant(Value::string("loop_start"));
    chunk.emit(Instruction::LoadConst(1, zero), 1);
    chunk.emit(Instruction::LoadConst(2, identity), 1);
    chunk.emit(Instruction::Call(2, 1, 1, 1), 1);
    chunk.emit(Instruction::LoadBool(3, true), 2);
    chunk.emit(Instruction::StoreGlobal(global, 1), 2);
    chunk.emit(Instruction::Move(2, 1), 2);
    chunk.emit(Instruction::Move(1, 2), 2);
    let header = chunk.emit(Instruction::BranchLtInt(1, 0, 0), 3);
    chunk.emit(Instruction::AddIntImm(2, 1, 1), 4);
    chunk.emit(Instruction::Move(1, 2), 4);
    let backedge = chunk.emit(Instruction::Jump(0), 4);
    let exit = chunk.emit(Instruction::Return(1), 5);
    chunk.patch_jump(header, exit);
    chunk.patch_jump(backedge, header);

    // Entry + callee + fallthrough (8), eight iterations (40), exit (3).
    let mut packed = bytecode_vm(vec![function.clone(), callee.clone()]);
    let mut exact = bytecode_vm(vec![function, callee]);
    for limit in 0..=51 {
        packed.reset_gas_counter();
        packed.set_gas_budget(limit);
        let actual = packed.call("run", vec![Value::Int(8)]);
        exact.reset_gas_counter();
        exact.set_gas_budget(limit);
        exact.set_instruction_hook(Some(Box::new(|_, _, _| {})));
        let expected = exact.call("run", vec![Value::Int(8)]);
        exact.clear_trace_hooks();
        assert_eq!(
            actual.as_ref().ok(),
            expected.as_ref().ok(),
            "limit {limit}"
        );
        assert_eq!(packed.gas_used(), exact.gas_used(), "limit {limit}");
        if limit < 51 {
            assert!(actual.unwrap_err().to_string().contains("Out of gas"));
        } else {
            assert_eq!(actual.unwrap(), Value::Int(8));
        }
        if limit >= 50 {
            assert_eq!(packed.gas_used(), 51);
        }
    }
}

#[test]
fn a_runtime_error_charges_only_the_executed_prefix() {
    let mut function = Function::new("divide", 2, false);
    function.set_register_count(3);
    function.chunk.emit(Instruction::DivInt(2, 0, 1), 1);
    function.chunk.emit(Instruction::Return(2), 2);
    let mut vm = bytecode_vm(vec![function]);
    let error = vm
        .call("divide", vec![Value::Int(1), Value::Int(0)])
        .unwrap_err();
    assert!(
        error
            .to_string()
            .to_lowercase()
            .contains("division by zero")
    );
    assert_eq!(vm.gas_used(), 1);
}

#[test]
fn test_ng_w14_3_compact_field_ops_and_weak_gc() {
    let node_type = Type::new(TypeKind::Named("Node".into()), Span::dummy());
    let layout = Rc::new(StructLayout::new(
        "Node".into(),
        vec![Rc::new("number".into()), Rc::new("child".into())],
        vec![FieldStorage::Strong, FieldStorage::Weak],
        vec![
            Type::new(TypeKind::String, Span::dummy()),
            node_type.clone(),
        ],
        vec![None, Some(node_type)],
    ));
    let parent = Value::Struct(StructObject::new(
        "Node",
        layout.clone(),
        vec![Value::string("before"), Value::none()],
    ));
    let child = StructObject::new("Node", layout, vec![Value::string("child"), Value::none()]);
    let weak_child = Rc::downgrade(&child);
    let mut function = Function::new("mutate", 2, false);
    function.set_register_count(6);
    let chunk = &mut function.chunk;
    for i in 0..255 {
        chunk.add_constant(Value::Int(i));
    }
    let number = chunk.add_constant(Value::string("number"));
    let weak = chunk.add_constant(Value::string("child"));
    assert_eq!((number, weak), (255, 256));
    let prefix = chunk.add_constant(Value::string("value:"));
    let after = chunk.add_constant(Value::string("after"));

    chunk.emit(Instruction::LoadConst(2, prefix), 1);
    chunk.emit(Instruction::Concat(3, 2, 0), 1);
    chunk.emit(Instruction::LoadConst(4, after), 1);
    let set_small = chunk.emit(Instruction::SetField(0, number, 4), 1);
    let set_wide = chunk.emit(Instruction::SetField(0, weak, 1), 1);
    let get_wide = chunk.emit(Instruction::GetField(4, 0, weak), 1);
    chunk.emit(Instruction::Concat(5, 2, 0), 1);
    chunk.emit(Instruction::Concat(3, 3, 5), 1);
    chunk.emit(Instruction::Return(3), 1);

    let mut tostring = Function::new("Node:to_string", 1, false);
    tostring.set_register_count(2);
    for i in 0..255 {
        tostring.chunk.add_constant(Value::Int(i));
    }
    let number = tostring.chunk.add_constant(Value::string("number"));
    let get_small = tostring.chunk.emit(Instruction::GetField(1, 0, number), 1);
    tostring.chunk.emit(Instruction::Return(1), 1);

    assert_eq!(chunk.extra_args.len(), 2);
    assert_eq!(tostring.chunk.extra_args.len(), 0);
    assert_eq!(
        (
            chunk.packed_code[set_small].b(),
            chunk.packed_code[set_small].c()
        ),
        (4, 255)
    );
    assert_eq!(
        (
            tostring.chunk.packed_code[get_small].b(),
            tostring.chunk.packed_code[get_small].c()
        ),
        (0, 255)
    );
    assert_eq!(
        chunk.packed_code[set_wide].opcode_u8(),
        OpCode::SetField as u8
    );
    assert_eq!(
        chunk.packed_code[get_wide].opcode_u8(),
        OpCode::GetField as u8
    );
    for (chunk, pc) in [
        (&*chunk, set_small),
        (&*chunk, set_wide),
        (&*chunk, get_wide),
        (&tostring.chunk, get_small),
    ] {
        let packed = chunk.packed_code[pc];
        assert_eq!(
            packed.to_instruction(&chunk.extra_args).unwrap(),
            chunk.instructions[pc]
        );
        let legacy = chunk.instructions[pc].encode();
        assert_eq!(
            Instruction::decode(&legacy).unwrap(),
            chunk.instructions[pc]
        );
    }

    let mut read = Function::new("read_child", 1, false);
    read.set_register_count(2);
    let child_name = read.chunk.add_constant(Value::string("child"));
    read.chunk.emit(Instruction::GetField(1, 0, child_name), 1);
    read.chunk.emit(Instruction::Return(1), 1);
    let mut vm = bytecode_vm(vec![function, tostring, read]);
    vm.set_gc_enabled(true);
    assert_eq!(
        vm.call("mutate", vec![parent.clone(), Value::Struct(child.clone())])
            .unwrap(),
        Value::string("value:beforevalue:after")
    );
    assert_eq!(
        vm.call("read_child", vec![parent.clone()]).unwrap(),
        Value::some(Value::Struct(child.clone()))
    );
    drop(child);
    vm.force_collect_cycles();
    assert!(weak_child.upgrade().is_none());
    assert_eq!(vm.call("read_child", vec![parent]).unwrap(), Value::none());
}

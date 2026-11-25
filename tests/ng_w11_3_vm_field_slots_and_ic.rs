#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use lust::ast::{FieldOwnership, Span, StructDef, StructField, Type, TypeKind, Visibility};
use lust::bytecode::chunk::ChunkIcSlot;
use lust::bytecode::{FieldStorage, Function, Instruction, StructLayout, Value};
use lust::hashbrown::HashMap;
use lust::mir::ir::{
    FunctionSignature, MirBinaryOp, MirConstant, MirFunction, MirType, Operand, Place, Rvalue,
    Statement, Terminator,
};
use lust::mir::vm_backend::{audit_chunk_field_access, MirToBytecodeCompiler};
use lust::vm::VM;
use std::rc::Rc;

fn make_struct_defs(specs: &[(&str, &[(&str, TypeKind)])]) -> HashMap<String, StructDef> {
    let mut defs = HashMap::new();
    for &(name, fields) in specs {
        let mut struct_fields = Vec::new();
        for &(fname, ref kind) in fields {
            struct_fields.push(StructField {
                name: fname.to_string(),
                ty: Type::new(kind.clone(), Span::dummy()),
                ownership: FieldOwnership::Strong,
                weak_target: None,
                visibility: Visibility::Public,
            });
        }
        defs.insert(
            name.to_string(),
            StructDef {
                name: name.to_string(),
                type_params: Vec::new(),
                trait_bounds: Vec::new(),
                fields: struct_fields,
                visibility: Visibility::Public,
                doc: None,
            },
        );
    }
    defs
}

fn make_layout(name: &str, fields: &[(&str, TypeKind)]) -> Rc<StructLayout> {
    let field_names = fields.iter().map(|(n, _)| Rc::new((*n).to_string())).collect();
    let field_storage = vec![FieldStorage::Strong; fields.len()];
    let field_types = fields
        .iter()
        .map(|(_, k)| Type::new(k.clone(), Span::dummy()))
        .collect();
    let weak_targets = vec![None; fields.len()];
    Rc::new(StructLayout::new(
        name.to_string(),
        field_names,
        field_storage,
        field_types,
        weak_targets,
    ))
}

#[test]
fn ng_w11_3_frame_fast_no_slow_exits_and_ic_populated() {
    // Function `fast_loop(n)`:
    //   r0 = n, r1 = i (0), r2 = acc (0), r3 = 1
    //   loop while i < n:
    //     r4 = LoadGlobal("g_step")
    //     r5 = NewStruct("Point", [x=r1, y=r4])
    //     r6 = GetField(r5, "x")
    //     SetField(r5, "y", r6)
    //     r7 = GetFieldSlot(r5, 1)
    //     SetFieldSlot(r5, 0, r4)
    //     r8 = GetFieldSlot(r5, 0)
    //     r9 = NewEnumUnit("Option", "None")
    //     r10 = NewEnumVariant("Option", "Some", [r7])
    //     r11 = GetEnumValue(r10, 0)
    //     r2 = r2 + r11 + r8
    //     r1 = r1 + 1
    //   return r2
    let mut f = Function::new("fast_loop", 1, false);
    let k_zero = f.chunk.add_constant(Value::Int(0));
    let k_one = f.chunk.add_constant(Value::Int(1));
    let k_g_step = f.chunk.add_constant(Value::string("g_step"));
    let k_point = f.chunk.add_constant(Value::string("Point"));
    // Contiguous field names for NewStruct:
    let first_field_name_idx = f.chunk.constants.len() as u16;
    f.chunk.constants.push(Value::string("x"));
    f.chunk.constants.push(Value::string("y"));
    let k_x = first_field_name_idx;
    let k_y = first_field_name_idx + 1;
    let k_option = f.chunk.add_constant(Value::string("Option"));
    let k_none = f.chunk.add_constant(Value::string("None"));
    let k_some = f.chunk.add_constant(Value::string("Some"));

    f.chunk.emit(Instruction::LoadConst(1, k_zero), 1); // r1 = 0
    f.chunk.emit(Instruction::LoadConst(2, k_zero), 1); // r2 = 0
    f.chunk.emit(Instruction::LoadConst(3, k_one), 1); // r3 = 1

    let loop_head = f.chunk.instructions.len();
    f.chunk.emit(Instruction::LtInt(12, 1, 0), 2);
    let jump_exit = f.chunk.emit(Instruction::JumpIfNot(12, 0), 2);

    let load_global_ip = f.chunk.emit(Instruction::LoadGlobal(4, k_g_step), 3);
    f.chunk.emit(Instruction::Move(13, 1), 3);
    f.chunk.emit(Instruction::Move(14, 4), 3);
    let new_struct_ip = f.chunk.emit(
        Instruction::NewStruct(5, k_point, first_field_name_idx, 13, 2),
        4,
    );
    let get_field_ip = f.chunk.emit(Instruction::GetField(6, 5, k_x), 5);
    let set_field_ip = f.chunk.emit(Instruction::SetField(5, k_y, 6), 6);
    f.chunk.emit(Instruction::GetFieldSlot(7, 5, 1), 7);
    f.chunk.emit(Instruction::SetFieldSlot(5, 0, 4), 8);
    f.chunk.emit(Instruction::GetFieldSlot(8, 5, 0), 9);
    let new_enum_unit_ip = f.chunk.emit(Instruction::NewEnumUnit(9, k_option, k_none), 10);
    let new_enum_var_ip = f
        .chunk
        .emit(Instruction::NewEnumVariant(10, k_option, k_some, 7, 1), 11);
    f.chunk.emit(Instruction::GetEnumValue(11, 10, 0), 12);
    f.chunk.emit(Instruction::AddInt(2, 2, 11), 13);
    f.chunk.emit(Instruction::AddInt(2, 2, 8), 14);
    f.chunk.emit(Instruction::AddInt(1, 1, 3), 15);
    let jump_back = f.chunk.emit(Instruction::Jump(0), 16);
    f.chunk.patch_jump(jump_back, loop_head);

    let exit_pc = f.chunk.instructions.len();
    f.chunk.patch_jump(jump_exit, exit_pc);
    f.chunk.emit(Instruction::Return(2), 17);
    f.set_register_count(16);

    let mut vm = VM::new();
    vm.set_gc_enabled(false);
    vm.register_structs(&make_struct_defs(&[(
        "Point",
        &[("x", TypeKind::Int), ("y", TypeKind::Int)],
    )]));
    vm.set_global("g_step", Value::Int(5));
    vm.load_functions(vec![f]);

    let r0 = VM::frame_hoist_reload_count();
    let res_small = vm.call("fast_loop", vec![Value::Int(10)]).expect("run 10");
    let r1 = VM::frame_hoist_reload_count();
    let res_large = vm
        .call("fast_loop", vec![Value::Int(1000)])
        .expect("run 1000");
    let r2 = VM::frame_hoist_reload_count();

    // sum_{i=0}^{n-1} (i + 5) = n*(n-1)/2 + 5*n
    assert_eq!(res_small, Value::Int(10 * 9 / 2 + 5 * 10));
    assert_eq!(res_large, Value::Int(1000 * 999 / 2 + 5 * 1000));

    // In debug builds, frame_hoist_reload_count increments on every FrameExit::Slow.
    // Equal deltas for 10 vs 1000 iterations prove ZERO Slow exits inside the loop!
    assert_eq!(
        r1 - r0,
        r2 - r1,
        "expected identical frame reload counts for 10 vs 1000 iterations (0 slow exits inside loop)"
    );

    let func = vm.function_by_name("fast_loop").expect("fast_loop");
    let slots = func.chunk.ic_slots.borrow();
    assert!(matches!(
        slots[load_global_ip],
        ChunkIcSlot::GlobalSlot { .. }
    ));
    assert!(matches!(
        slots[new_struct_ip],
        ChunkIcSlot::NewStructTemplate {
            is_identity: true,
            ..
        }
    ));
    assert!(matches!(
        slots[get_field_ip],
        ChunkIcSlot::FieldSlot { slot_idx: 0, .. }
    ));
    assert!(matches!(
        slots[set_field_ip],
        ChunkIcSlot::FieldSlot { slot_idx: 1, .. }
    ));
    assert!(matches!(slots[new_enum_unit_ip], ChunkIcSlot::EnumUnit { .. }));
    assert!(matches!(
        slots[new_enum_var_ip],
        ChunkIcSlot::EnumVariantNames { .. }
    ));
}

#[test]
fn ng_w11_3_polymorphic_get_and_set_field_ic_updates_correctly() {
    // Function `read_and_bump_x(obj, delta)`:
    //   r2 = GetField(r0, "x")
    //   r3 = r2 + r1
    //   SetField(r0, "x", r3)
    //   r4 = GetField(r0, "x")
    //   return r4
    let mut f = Function::new("read_and_bump_x", 2, false);
    let k_x = f.chunk.add_constant(Value::string("x"));
    let get1_ip = f.chunk.emit(Instruction::GetField(2, 0, k_x), 1);
    f.chunk.emit(Instruction::Add(3, 2, 1), 2);
    let set_ip = f.chunk.emit(Instruction::SetField(0, k_x, 3), 3);
    let get2_ip = f.chunk.emit(Instruction::GetField(4, 0, k_x), 4);
    f.chunk.emit(Instruction::Return(4), 5);
    f.set_register_count(6);

    let mut vm = VM::new();
    vm.register_structs(&make_struct_defs(&[
        ("A", &[("x", TypeKind::Int)]),
        ("B", &[("y", TypeKind::Int), ("x", TypeKind::Int)]),
    ]));
    vm.load_functions(vec![f]);

    let a_obj = vm
        .instantiate_struct("A", vec![(Rc::new("x".to_string()), Value::Int(10))])
        .expect("A");
    let b_obj = vm
        .instantiate_struct(
            "B",
            vec![
                (Rc::new("y".to_string()), Value::Int(999)),
                (Rc::new("x".to_string()), Value::Int(20)),
            ],
        )
        .expect("B");

    // 1. Call with A { x: 10 } -> IC caches slot 0
    let res_a1 = vm
        .call("read_and_bump_x", vec![a_obj.clone(), Value::Int(5)])
        .expect("call A");
    assert_eq!(res_a1, Value::Int(15));
    {
        let slots = vm
            .function_by_name("read_and_bump_x")
            .unwrap()
            .chunk
            .ic_slots
            .borrow();
        assert!(matches!(
            slots[get1_ip],
            ChunkIcSlot::FieldSlot { slot_idx: 0, .. }
        ));
        assert!(matches!(
            slots[set_ip],
            ChunkIcSlot::FieldSlot { slot_idx: 0, .. }
        ));
        assert!(matches!(
            slots[get2_ip],
            ChunkIcSlot::FieldSlot { slot_idx: 0, .. }
        ));
    }

    // 2. Call with B { y: 999, x: 20 } -> IC refreshes to slot 1 and preserves y = 999
    let res_b1 = vm
        .call("read_and_bump_x", vec![b_obj.clone(), Value::Int(7)])
        .expect("call B");
    assert_eq!(res_b1, Value::Int(27));
    assert_eq!(b_obj.struct_get_field("y"), Some(Value::Int(999)));
    assert_eq!(b_obj.struct_get_field("x"), Some(Value::Int(27)));
    {
        let slots = vm
            .function_by_name("read_and_bump_x")
            .unwrap()
            .chunk
            .ic_slots
            .borrow();
        assert!(matches!(
            slots[get1_ip],
            ChunkIcSlot::FieldSlot { slot_idx: 1, .. }
        ));
        assert!(matches!(
            slots[set_ip],
            ChunkIcSlot::FieldSlot { slot_idx: 1, .. }
        ));
        assert!(matches!(
            slots[get2_ip],
            ChunkIcSlot::FieldSlot { slot_idx: 1, .. }
        ));
    }

    // 3. Switch back to A -> IC refreshes back to slot 0
    let res_a2 = vm
        .call("read_and_bump_x", vec![a_obj.clone(), Value::Int(3)])
        .expect("call A again");
    assert_eq!(res_a2, Value::Int(18));
    assert_eq!(a_obj.struct_get_field("x"), Some(Value::Int(18)));
    {
        let slots = vm
            .function_by_name("read_and_bump_x")
            .unwrap()
            .chunk
            .ic_slots
            .borrow();
        assert!(matches!(
            slots[get1_ip],
            ChunkIcSlot::FieldSlot { slot_idx: 0, .. }
        ));
    }
}

#[test]
fn ng_w11_3_load_global_ic_invalidation_on_store_global_and_set_global() {
    // Function `read_g()`: LoadGlobal(0, "counter"); Return(0)
    let mut read_fn = Function::new("read_g", 0, false);
    let k_counter = read_fn.chunk.add_constant(Value::string("counter"));
    let load_ip = read_fn.chunk.emit(Instruction::LoadGlobal(0, k_counter), 1);
    read_fn.chunk.emit(Instruction::Return(0), 2);
    read_fn.set_register_count(2);

    // Function `write_g(val)`: StoreGlobal("counter", 0); Return(0)
    let mut write_fn = Function::new("write_g", 1, false);
    let k_counter_w = write_fn.chunk.add_constant(Value::string("counter"));
    write_fn.chunk.emit(Instruction::StoreGlobal(k_counter_w, 0), 1);
    write_fn.chunk.emit(Instruction::Return(0), 2);
    write_fn.set_register_count(2);

    let mut vm = VM::new();
    vm.set_global("counter", Value::Int(100));
    vm.load_functions(vec![read_fn, write_fn]);

    let gen_before = vm.globals_generation();
    assert_eq!(vm.call("read_g", vec![]).expect("read 1"), Value::Int(100));
    let cached_gen_1 = match vm.function_by_name("read_g").unwrap().chunk.ic_slots.borrow()[load_ip]
    {
        ChunkIcSlot::GlobalSlot { generation, .. } => generation,
        ref other => panic!("expected GlobalSlot IC, got {other:?}"),
    };
    assert_eq!(cached_gen_1, gen_before);

    // Mutate via StoreGlobal opcode -> must bump globals_generation and invalidate LoadGlobal IC
    assert_eq!(
        vm.call("write_g", vec![Value::Int(250)]).expect("write_g"),
        Value::Int(250)
    );
    let gen_after_store = vm.globals_generation();
    assert!(
        gen_after_store > gen_before,
        "StoreGlobal must bump globals_generation"
    );

    assert_eq!(vm.call("read_g", vec![]).expect("read 2"), Value::Int(250));
    let cached_gen_2 = match vm.function_by_name("read_g").unwrap().chunk.ic_slots.borrow()[load_ip]
    {
        ChunkIcSlot::GlobalSlot { generation, .. } => generation,
        ref other => panic!("expected refreshed GlobalSlot IC, got {other:?}"),
    };
    assert_eq!(cached_gen_2, gen_after_store);

    // Mutate via host VM::set_global -> must stay synchronized and bump globals_generation
    vm.set_global("counter", Value::Int(999));
    let gen_after_host = vm.globals_generation();
    assert!(gen_after_host > gen_after_store);
    assert_eq!(vm.call("read_g", vec![]).expect("read 3"), Value::Int(999));
    let cached_gen_3 = match vm.function_by_name("read_g").unwrap().chunk.ic_slots.borrow()[load_ip]
    {
        ChunkIcSlot::GlobalSlot { generation, .. } => generation,
        ref other => panic!("expected refreshed GlobalSlot IC after set_global, got {other:?}"),
    };
    assert_eq!(cached_gen_3, gen_after_host);
}

#[test]
fn ng_w11_3_option_none_singleton_reuses_same_rc_pointer() {
    let mut f1 = Function::new("make_none_a", 0, false);
    let k_opt = f1.chunk.add_constant(Value::string("Option"));
    let k_none = f1.chunk.add_constant(Value::string("None"));
    f1.chunk.emit(Instruction::NewEnumUnit(0, k_opt, k_none), 1);
    f1.chunk.emit(Instruction::Return(0), 2);
    f1.set_register_count(1);

    let mut f2 = Function::new("make_none_b", 0, false);
    let k_opt2 = f2.chunk.add_constant(Value::string("Option"));
    let k_none2 = f2.chunk.add_constant(Value::string("None"));
    f2.chunk
        .emit(Instruction::NewEnumUnit(0, k_opt2, k_none2), 1);
    f2.chunk.emit(Instruction::Return(0), 2);
    f2.set_register_count(1);

    let mut vm = VM::new();
    vm.load_functions(vec![f1, f2]);

    let v1 = vm.call("make_none_a", vec![]).expect("none a1");
    let v2 = vm.call("make_none_a", vec![]).expect("none a2");
    let v3 = vm.call("make_none_b", vec![]).expect("none b1");

    let (Value::Enum(e1), Value::Enum(e2), Value::Enum(e3)) = (v1, v2, v3) else {
        panic!("expected Option::None Enum values");
    };
    assert!(Rc::ptr_eq(&e1, &e2), "Option::None must share singleton Rc");
    assert!(
        Rc::ptr_eq(&e1, &e3),
        "Option::None across functions must share thread-local singleton Rc"
    );
}

#[test]
fn ng_w11_3_mir_backend_field_slots_and_instruction_codec_roundtrip() {
    // 1. Instruction encode/decode/Display/defined_register/reads_register round-trip
    let get_slot = Instruction::GetFieldSlot(3, 1, 7);
    let set_slot = Instruction::SetFieldSlot(1, 7, 4);
    assert_eq!(Instruction::decode(&get_slot.encode()), Ok(get_slot));
    assert_eq!(Instruction::decode(&set_slot.encode()), Ok(set_slot));
    assert_eq!(get_slot.defined_register(), Some(3));
    assert_eq!(set_slot.defined_register(), None);
    assert!(get_slot.reads_register(1));
    assert!(!get_slot.reads_register(3));
    assert!(set_slot.reads_register(1));
    assert!(set_slot.reads_register(4));
    assert!(!set_slot.reads_register(2));
    assert_eq!(format!("{get_slot}"), "GetFieldSlot R3, R1, #7");
    assert_eq!(format!("{set_slot}"), "SetFieldSlot R1, #7, R4");

    // 2. MIR function operating on Point { x, y }
    let span = Span::dummy();
    let mut f = MirFunction::new(
        FunctionSignature {
            name: "slot_point_fn".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let arg_x = f.params[0];
    let p = f.add_local(Some("p".into()), MirType::Struct("Point".into()), true, span);
    let sum = f.add_local(Some("sum".into()), MirType::Int, false, span);
    let out = f.add_local(Some("out".into()), MirType::Int, false, span);
    f.blocks[0].statements.extend([
        Statement::Assign(
            Place::Local(p),
            Rvalue::Aggregate(
                MirType::Struct("Point".into()),
                vec![
                    Operand::Copy(Place::Local(arg_x)),
                    Operand::Constant(MirConstant::Int(10)),
                ],
            ),
            span,
        ),
        Statement::Assign(
            Place::Local(sum),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Field(Box::new(Place::Local(p)), 0)),
                Operand::Copy(Place::Field(Box::new(Place::Local(p)), 1)),
            ),
            span,
        ),
        Statement::Assign(
            Place::Field(Box::new(Place::Local(p)), 1),
            Rvalue::Use(Operand::Copy(Place::Local(sum))),
            span,
        ),
        Statement::Assign(
            Place::Local(out),
            Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(p)), 1))),
            span,
        ),
    ]);
    f.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(out))),
        span,
    ));

    let mut layouts = HashMap::new();
    layouts.insert(
        "Point".to_string(),
        make_layout("Point", &[("x", TypeKind::Int), ("y", TypeKind::Int)]),
    );

    let bc = MirToBytecodeCompiler::compile_with_field_slots(&f, &layouts)
        .expect("compile_with_field_slots");
    assert!(
        bc.chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::GetFieldSlot(_, _, 0)))
            && bc
                .chunk
                .instructions
                .iter()
                .any(|i| matches!(i, Instruction::GetFieldSlot(_, _, 1))),
        "expected GetFieldSlot for slots 0 and 1:\n{}",
        bc.disassemble()
    );
    assert!(
        bc.chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::SetFieldSlot(_, 1, _))),
        "expected SetFieldSlot for slot 1:\n{}",
        bc.disassemble()
    );
    assert!(
        audit_chunk_field_access(&bc.chunk).has_zero_name_lookup(),
        "GetFieldSlot/SetFieldSlot must have zero name lookups"
    );

    let mut vm = VM::new();
    vm.register_structs(&make_struct_defs(&[(
        "Point",
        &[("x", TypeKind::Int), ("y", TypeKind::Int)],
    )]));
    vm.load_functions(vec![bc]);
    assert_eq!(
        vm.call("slot_point_fn", vec![Value::Int(32)])
            .expect("execute slot_point_fn"),
        Value::Int(42)
    );
}

#[test]
fn ng_w11_3_call_method_monomorphic_ic_and_polymorphic_switch() {
    // Method A:score(self, bonus) -> self.x + bonus
    let mut a_score = Function::new("A:score", 2, true);
    a_score.chunk.emit(Instruction::GetFieldSlot(2, 0, 0), 1);
    a_score.chunk.emit(Instruction::AddInt(3, 2, 1), 2);
    a_score.chunk.emit(Instruction::Return(3), 3);
    a_score.set_register_count(4);

    // Method B:score(self, bonus) -> self.x * 2 + bonus
    let mut b_score = Function::new("B:score", 2, true);
    let k_two = b_score.chunk.add_constant(Value::Int(2));
    b_score.chunk.emit(Instruction::GetFieldSlot(2, 0, 1), 1);
    b_score.chunk.emit(Instruction::LoadConst(3, k_two), 2);
    b_score.chunk.emit(Instruction::MulInt(4, 2, 3), 3);
    b_score.chunk.emit(Instruction::AddInt(5, 4, 1), 4);
    b_score.chunk.emit(Instruction::Return(5), 5);
    b_score.set_register_count(6);

    // Caller `call_score(obj, bonus)`:
    let mut caller = Function::new("call_score", 2, false);
    let k_score = caller.chunk.add_constant(Value::string("score"));
    let call_ip = caller
        .chunk
        .emit(Instruction::CallMethod(0, k_score, 1, 1, 2), 1);
    caller.chunk.emit(Instruction::Return(2), 2);
    caller.set_register_count(4);

    let mut vm = VM::new();
    vm.register_structs(&make_struct_defs(&[
        ("A", &[("x", TypeKind::Int)]),
        ("B", &[("y", TypeKind::Int), ("x", TypeKind::Int)]),
    ]));
    vm.load_functions(vec![a_score, b_score, caller]);

    let a_obj = vm
        .instantiate_struct("A", vec![(Rc::new("x".to_string()), Value::Int(10))])
        .unwrap();
    let b_obj = vm
        .instantiate_struct(
            "B",
            vec![
                (Rc::new("y".to_string()), Value::Int(0)),
                (Rc::new("x".to_string()), Value::Int(10)),
            ],
        )
        .unwrap();

    assert_eq!(
        vm.call("call_score", vec![a_obj.clone(), Value::Int(3)])
            .unwrap(),
        Value::Int(13)
    );
    assert!(matches!(
        vm.function_by_name("call_score").unwrap().chunk.ic_slots.borrow()[call_ip],
        ChunkIcSlot::MethodCall {
            target_func_idx: 0,
            ..
        }
    ));

    // Second call hits IC directly
    assert_eq!(
        vm.call("call_score", vec![a_obj, Value::Int(4)]).unwrap(),
        Value::Int(14)
    );

    // Switch receiver to B -> updates IC to target_func_idx 1
    assert_eq!(
        vm.call("call_score", vec![b_obj, Value::Int(5)]).unwrap(),
        Value::Int(25)
    );
    assert!(matches!(
        vm.function_by_name("call_score").unwrap().chunk.ic_slots.borrow()[call_ip],
        ChunkIcSlot::MethodCall {
            target_func_idx: 1,
            ..
        }
    ));
}

#[test]
fn ng_w11_3_lua_table_never_populates_struct_layout_ic() {
    // Verify LuaTable is excluded from StructLayout slot IC in GetField/SetField/NewStruct
    let mut f = Function::new("lua_table_probe", 1, false);
    let k_foo = f.chunk.add_constant(Value::string("foo"));
    let k_val = f.chunk.add_constant(Value::Int(77));
    f.chunk.emit(Instruction::LoadConst(1, k_val), 1);
    let set_ip = f.chunk.emit(Instruction::SetField(0, k_foo, 1), 2);
    let get_ip = f.chunk.emit(Instruction::GetField(2, 0, k_foo), 3);
    f.chunk.emit(Instruction::Return(2), 4);
    f.set_register_count(4);

    let mut vm = VM::new();
    vm.register_structs(&make_struct_defs(&[(
        "LuaTable",
        &[
            ("table", TypeKind::Unknown),
            ("metamethods", TypeKind::Unknown),
        ],
    )]));
    vm.load_functions(vec![f]);

    let table_map = Value::map(HashMap::with_hasher(Default::default()));
    let lua_table = vm
        .instantiate_struct(
            "LuaTable",
            vec![
                (Rc::new("table".to_string()), table_map),
                (Rc::new("metamethods".to_string()), Value::Nil),
            ],
        )
        .expect("LuaTable");

    let res = vm
        .call("lua_table_probe", vec![lua_table])
        .expect("lua_table_probe");
    assert_eq!(res, Value::Int(77));

    let slots = vm
        .function_by_name("lua_table_probe")
        .unwrap()
        .chunk
        .ic_slots
        .borrow();
    assert!(
        matches!(slots.get(set_ip), None | Some(ChunkIcSlot::Empty)),
        "LuaTable SetField must NEVER populate StructLayout IC, got {:?}",
        slots.get(set_ip)
    );
    assert!(
        matches!(slots.get(get_ip), None | Some(ChunkIcSlot::Empty)),
        "LuaTable GetField must NEVER populate StructLayout IC, got {:?}",
        slots.get(get_ip)
    );

    // Also verify MirToBytecodeCompiler does NOT emit GetFieldSlot/SetFieldSlot for LuaTable
    let span = Span::dummy();
    let mut mf = MirFunction::new(
        FunctionSignature {
            name: "lua_mir".into(),
            param_types: vec![MirType::Struct("LuaTable".into())],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let t = mf.params[0];
    let out = mf.add_local(Some("out".into()), MirType::Int, false, span);
    mf.blocks[0].statements.push(Statement::Assign(
        Place::Local(out),
        Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(t)), 0))),
        span,
    ));
    mf.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(out))),
        span,
    ));
    let mut layouts = HashMap::new();
    layouts.insert(
        "LuaTable".to_string(),
        make_layout(
            "LuaTable",
            &[("table", TypeKind::Unknown), ("metamethods", TypeKind::Unknown)],
        ),
    );
    let bc = MirToBytecodeCompiler::compile_with_field_slots(&mf, &layouts)
        .expect("compile lua_mir");
    assert!(
        !bc.chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::GetFieldSlot(..) | Instruction::SetFieldSlot(..))),
        "MirToBytecodeCompiler must NOT emit GetFieldSlot/SetFieldSlot for LuaTable"
    );
}

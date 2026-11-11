//! Test suite for L5: Bytecode Value & Instruction Dispatch.
//!
//! Oracles:
//! - Exact mathematical / IEEE-754 semantics for number boundaries and hash equality.
//! - Independent type classification and tag reflection.
//! - Coinductive cycle detection for recursive tables/arrays/structs in Debug, Display, and PartialEq.
//! - Bit-level binary encoding/decoding roundtrip for all 85 instruction opcodes.
//! - Graceful error returns on truncated or corrupted byte streams without panicking.
//! - Disassembler and register liveness oracle.

use lust::ast::{Span, Type, TypeKind};
use lust::bytecode::instruction::{DecodeError, Instruction, OpCode};
use lust::bytecode::value::{
    jit_array_get_safe, jit_array_index_ok_safe, jit_array_index_result_safe, jit_array_len_safe,
    jit_array_push_safe, jit_array_set_index_safe, jit_call_method_safe, jit_concat_safe,
    jit_drop_values, jit_drop_values_masked, jit_drop_vec_bool, jit_drop_vec_float,
    jit_drop_vec_int, jit_enum_is_some_safe, jit_enum_unwrap_safe, jit_get_enum_value_safe,
    jit_get_field_indexed_int_fast, jit_get_field_indexed_safe, jit_get_field_keyed,
    jit_get_field_safe, jit_guard_function_identity, jit_guard_native_function,
    jit_guard_struct_layout, jit_init_nil, jit_is_enum_variant_safe, jit_move_safe,
    jit_new_array_safe, jit_new_enum_unit_safe, jit_new_enum_variant_safe,
    jit_new_option_some_fast, jit_rebox_array_bool, jit_rebox_array_float, jit_rebox_array_int,
    jit_release_value, jit_replace_bool, jit_replace_float32_bits, jit_replace_float_bits,
    jit_replace_int, jit_replace_int32, jit_replace_nil, jit_retain_value, jit_return_value,
    jit_set_field_indexed_int_fast, jit_set_field_indexed_safe, jit_set_field_safe,
    jit_set_field_strong_safe, jit_set_global_safe, jit_try_cast_safe, jit_type_is_safe,
    jit_unbox_array_bool, jit_unbox_array_float, jit_unbox_array_int, jit_value_is_truthy,
    jit_vec_bool_push, jit_vec_float_push, jit_vec_int_push, native_fn, ClosureObject,
    EnumObject, FieldStorage, IteratorState, JitVecSlot, LustMap, Name, NativeCallResult,
    StructLayout, StructObject, TaskHandle, Upvalue, Value, ValueKey, ValueTag, ValueType,
    WeakStructRef,
};
use lust::bytecode::Chunk;
use lust::number::{LustFloat, LustInt, NumericType};
use lust::vm::VM;
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

fn hash_of<T: Hash>(val: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    val.hash(&mut hasher);
    hasher.finish()
}

// ============================================================================
// 1. Number boundaries, ValueKey hash consistency, and Type Classification
// ============================================================================

#[test]
fn test_number_boundaries_eq_hash_and_classification() {
    let int_samples: &[LustInt] = &[
        0,
        1,
        -1,
        42,
        -42,
        LustInt::MAX,
        LustInt::MIN,
        (1 << 53) - 1, // max safe float int
        -(1 << 53) + 1,
    ];

    for &i in int_samples {
        let v = Value::Int(i);
        assert_eq!(
            v.type_of(),
            ValueType::Int,
            "input Int({}) type_of expected ValueType::Int",
            i
        );
        assert_eq!(
            v.tag(),
            ValueTag::Int,
            "input Int({}) tag expected ValueTag::Int",
            i
        );
        assert!(v.is_plain(), "input Int({}) must be plain", i);
        assert_eq!(
            v.as_int(),
            Some(i),
            "input Int({}) as_int expected Some({})",
            i,
            i
        );
        assert_eq!(
            v,
            Value::Int(i),
            "Int({}) must equal itself",
            i
        );

        let k1 = ValueKey::from(i);
        let k2 = ValueKey::from(Value::Int(i));
        assert_eq!(k1, k2, "ValueKey from int vs from Value::Int must be equal for {}", i);
        assert_eq!(
            hash_of(&k1),
            hash_of(&k2),
            "ValueKey hashes must be identical for {}",
            i
        );
    }

    // Boundary floats
    let zero = 0.0f64;
    let neg_zero = -0.0f64;
    let nan1 = f64::NAN;
    let nan2 = f64::from_bits(0x7ff8000000000001); // distinct NaN bit pattern
    let inf = f64::INFINITY;
    let neg_inf = f64::NEG_INFINITY;
    let max_float = f64::MAX;

    let float_samples = [zero, neg_zero, 1.0, -1.0, 3.14159, inf, neg_inf, max_float];
    for &f in &float_samples {
        let v = Value::Float(f);
        assert_eq!(
            v.type_of(),
            ValueType::Float,
            "input Float({}) type_of expected ValueType::Float",
            f
        );
        assert_eq!(
            v.tag(),
            ValueTag::Float,
            "input Float({}) tag expected ValueTag::Float",
            f
        );
        assert!(v.is_plain(), "input Float({}) must be plain", f);
        assert_eq!(
            v.as_float(),
            Some(f),
            "input Float({}) as_float expected Some({})",
            f,
            f
        );
    }

    // Zero vs -0.0 hash consistency oracle:
    // In IEEE 754, 0.0 == -0.0. Therefore, ValueKey(0.0) == ValueKey(-0.0),
    // and their hashes MUST be identical according to Rust's Hash contract.
    let k_zero = ValueKey::from(zero);
    let k_neg_zero = ValueKey::from(neg_zero);
    assert_eq!(
        k_zero, k_neg_zero,
        "ValueKey(0.0) must equal ValueKey(-0.0) per IEEE-754 semantics"
    );
    assert_eq!(
        hash_of(&k_zero),
        hash_of(&k_neg_zero),
        "hash of ValueKey(0.0) and ValueKey(-0.0) must be equal to preserve Hash invariant"
    );

    // Map retrieval across 0.0 and -0.0
    let mut map = LustMap::default();
    map.insert(k_zero.clone(), Value::string("zero_entry"));
    assert_eq!(
        map.get(&k_neg_zero),
        Some(&Value::string("zero_entry")),
        "looking up -0.0 in map inserted with 0.0 must return the entry"
    );

    // NaN equality and hash consistency oracle:
    // In ValueKey, NaNs are unified so that NaN keys can be stored and retrieved consistently.
    let k_nan1 = ValueKey::from(nan1);
    let k_nan2 = ValueKey::from(nan2);
    assert_eq!(
        k_nan1, k_nan2,
        "ValueKey(NaN) must equal another ValueKey(NaN) for key lookup stability"
    );
    assert_eq!(
        hash_of(&k_nan1),
        hash_of(&k_nan2),
        "hash of distinct NaN bit patterns must be equal in ValueKey"
    );

    map.insert(k_nan1.clone(), Value::Int(999));
    assert_eq!(
        map.get(&k_nan2),
        Some(&Value::Int(999)),
        "looking up NaN2 in map inserted with NaN1 must succeed"
    );

    // Distinctness between Int(0) and Float(0.0)
    assert_ne!(
        Value::Int(0),
        Value::Float(0.0),
        "Value::Int(0) and Value::Float(0.0) are distinct typed variants in Lust"
    );
    let k_int_zero = ValueKey::from(0i64);
    assert_ne!(
        k_int_zero, k_zero,
        "ValueKey(0) and ValueKey(0.0) must not be equal because types differ"
    );
}

// ============================================================================
// 2. Cross-Type Classification and Equality Matrix
// ============================================================================

#[test]
fn test_cross_type_equality_and_classification_matrix() {
    let int_type = lust::ast::Type::new(lust::ast::TypeKind::Int, lust::ast::Span::dummy());
    let dummy_layout = Rc::new(StructLayout::new(
        "Point".to_string(),
        vec![Rc::new("x".to_string()), Rc::new("y".to_string())],
        vec![FieldStorage::Strong, FieldStorage::Strong],
        vec![int_type.clone(), int_type],
        vec![None, None],
    ));
    let struct_obj = StructObject::new("Point", dummy_layout.clone(), vec![Value::Int(10), Value::Int(20)]);
    let weak_ref = WeakStructRef::new(&struct_obj);

    let dummy_fn = native_fn(|args| Ok(NativeCallResult::Return(args.first().cloned().unwrap_or(Value::Nil))));
    let closure_obj = Rc::new(ClosureObject {
        function_idx: 42,
        upvalues: vec![Upvalue::new(Value::Int(0))],
    });

    let distinct_values: Vec<(&'static str, Value, ValueType, ValueTag, bool)> = vec![
        ("Nil", Value::Nil, ValueType::Nil, ValueTag::Nil, false),
        ("Bool(false)", Value::Bool(false), ValueType::Bool, ValueTag::Bool, false),
        ("Bool(true)", Value::Bool(true), ValueType::Bool, ValueTag::Bool, true),
        ("Int(42)", Value::Int(42), ValueType::Int, ValueTag::Int, true),
        ("Float(42.0)", Value::Float(42.0), ValueType::Float, ValueTag::Float, true),
        ("String", Value::string("lust"), ValueType::String, ValueTag::String, true),
        ("Array", Value::array(vec![Value::Int(1)]), ValueType::Array, ValueTag::Array, true),
        ("Tuple", Value::tuple(vec![Value::Int(1)]), ValueType::Tuple, ValueTag::Tuple, true),
        ("Map", Value::map(LustMap::default()), ValueType::Map, ValueTag::Map, true),
        ("Struct", Value::Struct(struct_obj.clone()), ValueType::Struct, ValueTag::Struct, true),
        ("WeakStruct", Value::WeakStruct(weak_ref), ValueType::Struct, ValueTag::Struct, true),
        ("EnumUnit", Value::enum_unit("Option", "None"), ValueType::Enum, ValueTag::Enum, false),
        ("EnumVariant", Value::enum_variant("Option", "Some", vec![Value::Int(1)]), ValueType::Enum, ValueTag::Enum, true),
        ("Function", Value::Function(0), ValueType::Function, ValueTag::Function, true),
        ("NativeFunction", Value::NativeFunction(dummy_fn), ValueType::NativeFunction, ValueTag::NativeFunction, true),
        ("Closure", Value::Closure(closure_obj), ValueType::Closure, ValueTag::Closure, true),
        ("Task", Value::Task(TaskHandle(77)), ValueType::Task, ValueTag::Task, true),
    ];

    for (name, val, expected_type, expected_tag, expected_truthy) in &distinct_values {
        assert_eq!(
            val.type_of(),
            *expected_type,
            "type_of mismatch for variant {}",
            name
        );
        assert_eq!(
            val.tag(),
            *expected_tag,
            "tag mismatch for variant {}",
            name
        );
        assert_eq!(
            val.is_truthy(),
            *expected_truthy,
            "is_truthy mismatch for variant {}",
            name
        );
        assert_eq!(
            val.to_bool(),
            *expected_truthy,
            "to_bool mismatch for variant {}",
            name
        );

        // Fast clone equivalence
        let cloned = val.fast_clone();
        if !matches!(val, Value::NativeFunction(_) | Value::Iterator(_)) {
            assert_eq!(
                val, &cloned,
                "fast_clone of {} must be equal to original",
                name
            );
        }
    }

    // Cross-equality: distinct type pairs should not equal each other
    for (i, (name1, val1, _, _, _)) in distinct_values.iter().enumerate() {
        for (j, (name2, val2, _, _, _)) in distinct_values.iter().enumerate() {
            if i != j {
                // WeakStruct upgrades to Struct, so WeakStruct and Struct can equal if pointing to same object
                if (name1.contains("WeakStruct") && name2.contains("Struct"))
                    || (name1.contains("Struct") && name2.contains("WeakStruct"))
                {
                    continue;
                }
                assert_ne!(
                    val1, val2,
                    "cross-type equality between {} and {} must be false",
                    name1, name2
                );
            }
        }
    }
}

// ============================================================================
// 3. Table-Driven Value Conversions & Display Formatting
// ============================================================================

#[test]
fn test_table_driven_from_and_display() {
    let cases: Vec<(Value, &'static str)> = vec![
        (Value::from(false), "false"),
        (Value::from(true), "true"),
        (Value::from(0i64), "0"),
        (Value::from(-100i64), "-100"),
        (Value::from(1.25f64), "1.25"),
        (Value::from("hello"), "hello"),
        (Value::from("world".to_string()), "world"),
        (Value::from(Rc::new("rc_str".to_string())), "rc_str"),
        (Value::from(vec![Value::Int(1), Value::Int(2), Value::Int(3)]), "[1, 2, 3]"),
        (Value::tuple(vec![Value::Int(10), Value::string("x")]), "(10, x)"),
        (Value::enum_unit("Status", "Pending"), "Status.Pending"),
        (Value::enum_variant("Status", "Error", vec![Value::string("timeout")]), "Status.Error(timeout)"),
        (Value::Function(12), "<function@12>"),
        (Value::Task(TaskHandle(42)), "<task 42>"),
        (Value::none(), "Option.None"),
        (Value::some(Value::Int(7)), "Option.Some(7)"),
        (Value::ok(Value::Int(1)), "Result.Ok(1)"),
        (Value::err(Value::string("fail")), "Result.Err(fail)"),
    ];

    for (val, expected_display) in cases {
        let actual_display = format!("{}", val);
        assert_eq!(
            actual_display, expected_display,
            "Display format mismatch for value {:?}",
            val
        );
        let to_str = val.to_string();
        assert_eq!(
            to_str, expected_display,
            "to_string mismatch for value {:?}",
            val
        );
    }
}

// ============================================================================
// 4. Map / Table: Empty, Collisions, and Tombstone Deletions
// ============================================================================

#[test]
fn test_table_empty_collisions_and_tombstones() {
    let map_val = Value::map(LustMap::default());
    assert_eq!(map_val.map_len(), Some(0), "empty map must have length 0");
    assert_eq!(format!("{}", map_val), "{}", "empty map display must be {{}}");
    assert_eq!(format!("{:?}", map_val), "Map({})", "empty map debug must be Map({{}})");

    // Inserting entries
    let key_a = ValueKey::string("alpha");
    let key_b = ValueKey::string("beta");
    let key_c = ValueKey::string("gamma");

    assert!(map_val.map_set(key_a.clone(), Value::Int(100)).is_ok());
    assert!(map_val.map_set(key_b.clone(), Value::Int(200)).is_ok());
    assert!(map_val.map_set(key_c.clone(), Value::Int(300)).is_ok());
    assert_eq!(map_val.map_len(), Some(3), "map length after 3 inserts must be 3");

    assert_eq!(map_val.map_get(&key_a), Some(Value::Int(100)));
    assert_eq!(map_val.map_get(&key_b), Some(Value::Int(200)));
    assert_eq!(map_val.map_get(&key_c), Some(Value::Int(300)));
    assert_eq!(map_val.map_has(&key_b), Some(true));

    // Tombstone / deletion
    let removed = map_val.map_delete(&key_b).unwrap();
    assert_eq!(removed, Some(Value::Int(200)), "removing key_b must return previous value");
    assert_eq!(map_val.map_len(), Some(2), "map length after removal must be 2");
    assert_eq!(map_val.map_get(&key_b), None, "removed key_b must return None on get");
    assert_eq!(map_val.map_has(&key_b), Some(false));

    // Remaining keys still intact
    assert_eq!(map_val.map_get(&key_a), Some(Value::Int(100)));
    assert_eq!(map_val.map_get(&key_c), Some(Value::Int(300)));

    // Re-inserting previously deleted key
    assert!(map_val.map_set(key_b.clone(), Value::Int(205)).is_ok());
    assert_eq!(map_val.map_len(), Some(3), "map length after re-insertion must be 3");
    assert_eq!(map_val.map_get(&key_b), Some(Value::Int(205)));

    // Colliding hash keys simulation:
    // Insert integer keys that might share buckets or collision behavior
    for i in 0..100 {
        let k = ValueKey::from(i as LustInt);
        map_val.map_set(k, Value::Int(i * 10)).unwrap();
    }
    assert_eq!(map_val.map_len(), Some(103)); // 3 string keys + 100 int keys
    for i in 0..100 {
        let k = ValueKey::from(i as LustInt);
        assert_eq!(map_val.map_get(&k), Some(Value::Int(i * 10)));
    }
}

// ============================================================================
// 5. Cyclic Data Structures: Infinite Recursion Immunity
// ============================================================================

#[test]
fn test_cyclic_data_structures_never_overflow_or_hang() {
    // 5.1 Cyclic Array: array containing itself
    let cyclic_arr = Value::array(vec![Value::Int(1)]);
    cyclic_arr.array_push(cyclic_arr.clone()).unwrap();
    assert_eq!(cyclic_arr.array_len(), Some(2));

    // Display must format cycle gracefully without overflowing stack
    let disp = format!("{}", cyclic_arr);
    assert!(
        disp.contains("[...]"),
        "cyclic array display must contain cycle guard marker '[...]', got: {}",
        disp
    );

    // Debug must format cycle gracefully without overflowing stack
    let dbg = format!("{:?}", cyclic_arr);
    assert!(
        dbg.contains("Array([...])"),
        "cyclic array debug must contain cycle guard marker 'Array([...])', got: {}",
        dbg
    );

    // Equality must terminate and return true for self-reference
    assert_eq!(
        cyclic_arr, cyclic_arr,
        "cyclic array must equal itself without infinite recursion"
    );

    // 5.2 Mutual Cyclic Arrays: a -> b, b -> a
    let a = Value::array(vec![Value::Int(10)]);
    let b = Value::array(vec![Value::Int(10)]);
    a.array_push(b.clone()).unwrap();
    b.array_push(a.clone()).unwrap();

    let a_disp = format!("{}", a);
    assert!(a_disp.contains("[...]"), "mutual cycle display must terminate: {}", a_disp);
    assert_eq!(a, b, "mutually symmetric cyclic arrays must equal each other");

    // Asymmetric cyclic arrays must terminate and return false
    let c = Value::array(vec![Value::Int(99)]); // different element
    c.array_push(c.clone()).unwrap();
    assert_ne!(a, c, "asymmetric cyclic arrays must not be equal");

    // 5.3 Cyclic Map
    let cyclic_map = Value::map(LustMap::default());
    cyclic_map
        .map_set(ValueKey::string("self"), cyclic_map.clone())
        .unwrap();

    let map_disp = format!("{}", cyclic_map);
    assert!(
        map_disp.contains("{...}"),
        "cyclic map display must contain '{{...}}', got: {}",
        map_disp
    );

    let map_dbg = format!("{:?}", cyclic_map);
    assert!(
        map_dbg.contains("Map({{{...}}})") || map_dbg.contains("Map"),
        "cyclic map debug must terminate, got: {}",
        map_dbg
    );

    assert_eq!(
        cyclic_map, cyclic_map,
        "cyclic map must equal itself without infinite recursion"
    );

    // 5.4 Cyclic Struct
    let int_type = lust::ast::Type::new(lust::ast::TypeKind::Int, lust::ast::Span::dummy());
    let next_type = lust::ast::Type::new(lust::ast::TypeKind::Named("Node".to_string()), lust::ast::Span::dummy());
    let node_layout = Rc::new(StructLayout::new(
        "Node".to_string(),
        vec![Rc::new("val".to_string()), Rc::new("next".to_string())],
        vec![FieldStorage::Strong, FieldStorage::Strong],
        vec![int_type, next_type],
        vec![None, None],
    ));
    let node = StructObject::new("Node", node_layout, vec![Value::Int(1), Value::Nil]);
    let node_val = Value::Struct(node);
    // Introduce self-loop into field 'next'
    if let Value::Struct(ref s) = node_val {
        s.fields.borrow_mut()[1] = node_val.clone();
    }

    let node_disp = format!("{}", node_val);
    assert!(
        node_disp.contains("Node { ... }"),
        "cyclic struct display must contain cycle guard, got: {}",
        node_disp
    );

    let node_dbg = format!("{:?}", node_val);
    assert!(
        node_dbg.contains("fields: [...]"),
        "cyclic struct debug must contain cycle guard, got: {}",
        node_dbg
    );

    assert_eq!(
        node_val, node_val,
        "cyclic struct must equal itself without infinite recursion"
    );
}

// ============================================================================
// 6. Instruction Binary Encoding / Decoding Exhaustive Roundtrip
// ============================================================================

#[test]
fn test_instruction_encoding_decoding_roundtrip_exhaustive() {
    let instructions: Vec<Instruction> = vec![
        Instruction::LoadNil(5),
        Instruction::LoadBool(1, false),
        Instruction::LoadBool(2, true),
        Instruction::LoadConst(3, 1024),
        Instruction::LoadGlobal(4, 2048),
        Instruction::StoreGlobal(512, 7),
        Instruction::Move(10, 20),
        Instruction::Add(1, 2, 3),
        Instruction::Sub(4, 5, 6),
        Instruction::Mul(7, 8, 9),
        Instruction::Div(10, 11, 12),
        Instruction::Mod(13, 14, 15),
        Instruction::Pow(16, 17, 18),
        Instruction::FloorDiv(19, 20, 21),
        Instruction::Neg(22, 23),
        Instruction::Eq(24, 25, 26),
        Instruction::Ne(27, 28, 29),
        Instruction::Lt(30, 31, 32),
        Instruction::Le(33, 34, 35),
        Instruction::Gt(36, 37, 38),
        Instruction::Ge(39, 40, 41),
        Instruction::AddInt(1, 2, 3),
        Instruction::SubInt(4, 5, 6),
        Instruction::MulInt(7, 8, 9),
        Instruction::DivInt(10, 11, 12),
        Instruction::ModInt(13, 14, 15),
        Instruction::PowInt(16, 17, 18),
        Instruction::FloorDivInt(19, 20, 21),
        Instruction::NegInt(22, 23),
        Instruction::EqInt(24, 25, 26),
        Instruction::NeInt(27, 28, 29),
        Instruction::LtInt(30, 31, 32),
        Instruction::LeInt(33, 34, 35),
        Instruction::GtInt(36, 37, 38),
        Instruction::GeInt(39, 40, 41),
        Instruction::AddFloat(1, 2, 3),
        Instruction::SubFloat(4, 5, 6),
        Instruction::MulFloat(7, 8, 9),
        Instruction::DivFloat(10, 11, 12),
        Instruction::ModFloat(13, 14, 15),
        Instruction::PowFloat(16, 17, 18),
        Instruction::FloorDivFloat(19, 20, 21),
        Instruction::NegFloat(22, 23),
        Instruction::EqFloat(24, 25, 26),
        Instruction::NeFloat(27, 28, 29),
        Instruction::LtFloat(30, 31, 32),
        Instruction::LeFloat(33, 34, 35),
        Instruction::GtFloat(36, 37, 38),
        Instruction::GeFloat(39, 40, 41),
        Instruction::And(1, 2, 3),
        Instruction::Or(4, 5, 6),
        Instruction::Not(7, 8),
        Instruction::BitAnd(9, 10, 11),
        Instruction::BitOr(12, 13, 14),
        Instruction::BitXor(15, 16, 17),
        Instruction::Shl(18, 19, 20),
        Instruction::Shr(21, 22, 23),
        Instruction::BitNot(24, 25),
        Instruction::Jump(-320),
        Instruction::Jump(500),
        Instruction::JumpIf(2, -15),
        Instruction::JumpIfNot(3, 45),
        Instruction::Call(0, 1, 4, 5),
        Instruction::Return(8),
        Instruction::NewArray(1, 2, 10),
        Instruction::NewMap(4),
        Instruction::NewStruct(0, 100, 200, 3, 5),
        Instruction::NewEnumUnit(1, 55, 66),
        Instruction::NewEnumVariant(2, 77, 88, 3, 2),
        Instruction::TupleNew(1, 4, 3),
        Instruction::TupleGet(2, 3, 1),
        Instruction::IsEnumVariant(0, 1, 300, 400),
        Instruction::GetEnumValue(5, 6, 2),
        Instruction::GetField(1, 2, 5000),
        Instruction::SetField(2, 5000, 3),
        Instruction::GetIndex(1, 2, 3),
        Instruction::TryGetIndex(4, 5, 6),
        Instruction::ArrayIndexOk(1, 2, 3, 4),
        Instruction::ArrayLen(10, 11),
        Instruction::SetIndex(1, 2, 3),
        Instruction::Concat(7, 8, 9),
        Instruction::CallMethod(1, 600, 2, 3, 4),
        Instruction::TypeIs(1, 2, 700),
        Instruction::TryCast(3, 4, 800),
        Instruction::LoadUpvalue(5, 6),
        Instruction::StoreUpvalue(7, 8),
        Instruction::Closure(1, 999, 2, 3),
    ];

    assert_eq!(instructions.len(), 87); // 85 unique opcodes, plus both bool values and jump signs

    for (idx, instr) in instructions.iter().enumerate() {
        let encoded = instr.encode();
        assert_eq!(
            encoded.len(),
            Instruction::ENCODED_SIZE,
            "encoded size for instruction #{} ({:?}) must be 8 bytes",
            idx,
            instr
        );

        let decoded = Instruction::decode(&encoded).unwrap_or_else(|e| {
            panic!(
                "failed to decode instruction #{}: {:?}, error: {:?}",
                idx, instr, e
            )
        });

        assert_eq!(
            &decoded, instr,
            "roundtrip decode mismatch for instruction #{}: expected {:?}, got {:?}",
            idx, instr, decoded
        );
    }
}

// ============================================================================
// 7. Hostile & Corrupted Instruction Stream Decoding
// ============================================================================

#[test]
fn test_corrupted_instruction_stream_never_panics() {
    // 7.1 Truncated streams
    for len in 0..Instruction::ENCODED_SIZE {
        let short_bytes = vec![0u8; len];
        let result = Instruction::decode(&short_bytes);
        assert_eq!(
            result,
            Err(DecodeError::Truncated {
                expected: Instruction::ENCODED_SIZE,
                found: len,
            }),
            "stream of length {} must return Truncated error",
            len
        );
    }

    // 7.2 Unknown opcodes
    for op in 85..=255u8 {
        let mut bytes = [0u8; 8];
        bytes[0] = op;
        let result = Instruction::decode(&bytes);
        assert_eq!(
            result,
            Err(DecodeError::UnknownOpCode(op)),
            "opcode 0x{:02x} ({}) must return UnknownOpCode error",
            op,
            op
        );
    }

    // 7.3 Invalid boolean encoding in LoadBool (opcode 1)
    for invalid_bool in 2..=255u8 {
        let mut bytes = [0u8; 8];
        bytes[0] = OpCode::LoadBool.as_u8();
        bytes[1] = 0; // dest reg
        bytes[2] = invalid_bool;
        let result = Instruction::decode(&bytes);
        assert_eq!(
            result,
            Err(DecodeError::InvalidBoolean(invalid_bool)),
            "LoadBool with byte 0x{:02x} must return InvalidBoolean error",
            invalid_bool
        );
    }

    // 7.4 Fuzz pseudo-random hostile streams: must NEVER panic
    let mut state: u64 = 0xdeadbeefcafebabe;
    for _ in 0..20_000 {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let bytes = state.to_le_bytes();

        let _ = Instruction::decode(&bytes); // must not panic
    }
}

// ============================================================================
// 8. Disassembler, Register Liveness, and Specialization Contracts
// ============================================================================

#[test]
fn test_instruction_disassembler_and_register_contracts() {
    let mut chunk = Chunk::new();
    let k0 = chunk.add_constant(Value::Int(42));
    let k1 = chunk.add_constant(Value::string("pi"));

    chunk.emit(Instruction::LoadConst(0, k0), 10);
    chunk.emit(Instruction::LoadConst(1, k1), 10);
    chunk.emit(Instruction::AddInt(2, 0, 1), 11);
    chunk.emit(Instruction::Return(2), 12);

    let disasm = chunk.disassemble("test_proc");
    assert!(
        disasm.contains("===== test_proc ====="),
        "disassembly header missing: {}",
        disasm
    );
    assert!(
        disasm.contains("K0: 42"),
        "constant K0 missing in disasm: {}",
        disasm
    );
    assert!(
        disasm.contains("K1: pi"),
        "constant K1 missing in disasm: {}",
        disasm
    );
    assert!(
        disasm.contains("LoadConst R0, K0"),
        "LoadConst instruction missing in disasm: {}",
        disasm
    );
    assert!(
        disasm.contains("AddInt R2, R0, R1"),
        "AddInt instruction missing in disasm: {}",
        disasm
    );
    assert!(
        disasm.contains("Return R2"),
        "Return instruction missing in disasm: {}",
        disasm
    );

    // Register contracts: defined_register
    assert_eq!(Instruction::Add(5, 1, 2).defined_register(), Some(5));
    assert_eq!(Instruction::Move(7, 3).defined_register(), Some(7));
    assert_eq!(Instruction::Return(5).defined_register(), None);
    assert_eq!(Instruction::Jump(10).defined_register(), None);
    assert_eq!(Instruction::StoreGlobal(0, 4).defined_register(), None);

    // Register contracts: reads_register
    let add = Instruction::Add(0, 1, 2);
    assert!(add.reads_register(1));
    assert!(add.reads_register(2));
    assert!(!add.reads_register(0));
    assert!(!add.reads_register(3));

    let call = Instruction::Call(0, 4, 3, 1); // func=0, args=4..6, dest=1
    assert!(call.reads_register(0));
    assert!(call.reads_register(4));
    assert!(call.reads_register(5));
    assert!(call.reads_register(6));
    assert!(!call.reads_register(7));
    assert!(!call.reads_register(1));

    // Specialization contracts
    let generic_add = Instruction::Add(0, 1, 2);
    let specialized_int = generic_add.specialize_numeric(NumericType::Int);
    assert_eq!(specialized_int, Instruction::AddInt(0, 1, 2));

    let specialized_float = generic_add.specialize_numeric(NumericType::Float);
    assert_eq!(specialized_float, Instruction::AddFloat(0, 1, 2));

    let inverted = specialized_int.numeric_specialization();
    assert_eq!(inverted, Some((generic_add, NumericType::Int, 1, 2)));

    let inverted_float = specialized_float.numeric_specialization();
    assert_eq!(inverted_float, Some((generic_add, NumericType::Float, 1, 2)));

    // Negation specialization contract: operand register repeated in both positions
    let generic_neg = Instruction::Neg(3, 4);
    let specialized_neg_int = generic_neg.specialize_numeric(NumericType::Int);
    assert_eq!(specialized_neg_int, Instruction::NegInt(3, 4));
    assert_eq!(
        specialized_neg_int.numeric_specialization(),
        Some((generic_neg, NumericType::Int, 4, 4))
    );
}

// ============================================================================
// 9. Display & Formatter for All Instruction Categories
// ============================================================================

#[test]
fn test_instruction_display_format_coverage() {
    let cases: Vec<(Instruction, &'static str)> = vec![
        (Instruction::LoadNil(0), "LoadNil R0"),
        (Instruction::LoadBool(1, true), "LoadBool R1, true"),
        (Instruction::LoadConst(2, 5), "LoadConst R2, K5"),
        (Instruction::LoadGlobal(3, 6), "LoadGlobal R3, K6"),
        (Instruction::StoreGlobal(7, 4), "StoreGlobal K7, R4"),
        (Instruction::Move(5, 6), "Move R5, R6"),
        (Instruction::Add(1, 2, 3), "Add R1, R2, R3"),
        (Instruction::Jump(-10), "Jump -10"),
        (Instruction::JumpIf(1, 20), "JumpIf R1, 20"),
        (Instruction::JumpIfNot(2, -5), "JumpIfNot R2, -5"),
        (Instruction::Call(0, 1, 0, 2), "Call R0, <no args>, R2"),
        (Instruction::Call(0, 1, 2, 2), "Call R0, R1..R2, R2"),
        (Instruction::NewArray(0, 1, 0), "NewArray R0, <no elements>"),
        (Instruction::NewArray(0, 1, 3), "NewArray R0, R1..R3"),
        (Instruction::NewMap(5), "NewMap R5"),
        (Instruction::NewStruct(0, 1, 2, 3, 0), "NewStruct R0, K1, <no fields>, R3..R3"),
        (Instruction::NewStruct(0, 1, 2, 3, 2), "NewStruct R0, K1, K2..K3, R3..R4"),
        (Instruction::NewEnumUnit(1, 2, 3), "NewEnumUnit R1, K2, K3"),
        (Instruction::NewEnumVariant(1, 2, 3, 4, 0), "NewEnumVariant R1, K2, K3, <no values>"),
        (Instruction::NewEnumVariant(1, 2, 3, 4, 2), "NewEnumVariant R1, K2, K3, R4..R5"),
        (Instruction::TupleNew(0, 1, 0), "TupleNew R0, <empty>"),
        (Instruction::TupleNew(0, 1, 2), "TupleNew R0, R1..R2"),
        (Instruction::TupleGet(0, 1, 2), "TupleGet R0, R1, 2"),
        (Instruction::IsEnumVariant(0, 1, 2, 3), "IsEnumVariant R0, R1, K2, K3"),
        (Instruction::GetEnumValue(0, 1, 2), "GetEnumValue R0, R1, 2"),
        (Instruction::GetField(0, 1, 2), "GetField R0, R1, K2"),
        (Instruction::SetField(1, 2, 3), "SetField R1, K2, R3"),
        (Instruction::GetIndex(0, 1, 2), "GetIndex R0, R1, R2"),
        (Instruction::TryGetIndex(0, 1, 2), "TryGetIndex R0, R1, R2"),
        (Instruction::ArrayIndexOk(0, 1, 2, 3), "ArrayIndexOk R0, R1, R2, R3"),
        (Instruction::ArrayLen(0, 1), "ArrayLen R0, R1"),
        (Instruction::SetIndex(0, 1, 2), "SetIndex R0, R1, R2"),
        (Instruction::Concat(0, 1, 2), "Concat R0, R1, R2"),
        (Instruction::CallMethod(0, 1, 2, 0, 3), "CallMethod R0, K1, <no args>, R3"),
        (Instruction::CallMethod(0, 1, 2, 2, 3), "CallMethod R0, K1, R2..R3, R3"),
        (Instruction::TypeIs(0, 1, 2), "TypeIs R0, R1, K2"),
        (Instruction::TryCast(0, 1, 2), "TryCast R0, R1, K2"),
        (Instruction::LoadUpvalue(0, 1), "LoadUpvalue R0, U1"),
        (Instruction::StoreUpvalue(1, 2), "StoreUpvalue U1, R2"),
        (Instruction::Closure(0, 1, 2, 0), "Closure R0, F1, <no upvalues>"),
        (Instruction::Closure(0, 1, 2, 2), "Closure R0, F1, R2..R3"),
    ];

    for (instr, expected) in cases {
        let actual = format!("{}", instr);
        assert_eq!(
            actual, expected,
            "Instruction display mismatch: expected '{}', got '{}'",
            expected, actual
        );
    }
}

// ============================================================================
// 10. StructLayout, Field Queries, and Weak Reference Conversion
// ============================================================================

#[test]
fn test_struct_layout_and_storage_semantics() {
    let int_type = Type::new(TypeKind::Int, Span::dummy());
    let person_type = Type::new(TypeKind::Named("Person".to_string()), Span::dummy());

    let field_names = vec![
        Rc::new("id".to_string()),
        Rc::new("owner".to_string()),
        Rc::new("name".to_string()),
    ];
    let field_storage = vec![
        FieldStorage::Strong,
        FieldStorage::Weak,
        FieldStorage::Strong,
    ];
    let field_types = vec![int_type.clone(), person_type.clone(), Type::new(TypeKind::String, Span::dummy())];
    let weak_targets = vec![None, Some(person_type), None];

    let layout = StructLayout::new(
        "Person".to_string(),
        field_names.clone(),
        field_storage,
        field_types,
        weak_targets,
    );

    // Metadata queries
    assert_eq!(layout.name(), "Person");
    assert_eq!(layout.name_interned().as_str(), "Person");
    assert!(layout.has_weak_fields());
    assert_eq!(layout.field_names(), &field_names[..]);
    assert_eq!(layout.field_storage(0), FieldStorage::Strong);
    assert_eq!(layout.field_storage(1), FieldStorage::Weak);
    assert_eq!(layout.field_storage(2), FieldStorage::Strong);
    assert!(!layout.is_weak(0));
    assert!(layout.is_weak(1));
    assert!(!layout.is_weak(2));
    assert_eq!(layout.field_type(0), &int_type);
    assert!(layout.weak_target(1).is_some());
    assert!(layout.weak_target(0).is_none());

    // Index lookup: pointer equality fast-path vs string fallback
    assert_eq!(layout.index_of_rc(&field_names[0]), Some(0));
    assert_eq!(layout.index_of_rc(&field_names[1]), Some(1));
    // Different Rc instance with same string
    let other_rc = Rc::new("id".to_string());
    assert_eq!(layout.index_of_rc(&other_rc), Some(0));
    assert_eq!(layout.index_of_str("owner"), Some(1));
    assert_eq!(layout.index_of_str("unknown"), None);
    assert_eq!(layout.index_of_rc(&Rc::new("unknown".to_string())), None);

    // Strong field canonicalize and materialize: identity pass-through
    let strong_val = Value::Int(42);
    let can_strong = layout.canonicalize_field_value(0, strong_val.clone()).unwrap();
    assert_eq!(can_strong, strong_val);
    let mat_strong = layout.materialize_field_value(0, strong_val.clone());
    assert_eq!(mat_strong, strong_val);

    // Weak field: canonicalize with Value::Struct -> coerces to Option.Some(WeakStruct)
    let dummy_layout = Rc::new(StructLayout::new(
        "Person".to_string(),
        vec![],
        vec![],
        vec![],
        vec![],
    ));
    let target_obj = StructObject::new("Person", dummy_layout.clone(), vec![]);
    let can_struct = layout.canonicalize_field_value(1, Value::Struct(target_obj.clone())).unwrap();
    match &can_struct {
        Value::Enum(obj) => {
            assert_eq!(obj.enum_name.as_str(), "Option");
            assert_eq!(obj.variant.as_str(), "Some");
            let inner = obj.values.as_ref().unwrap().first().unwrap();
            assert!(matches!(inner, Value::WeakStruct(_)));
        }
        _ => panic!("expected Option.Some for weak field canonicalization"),
    }

    // Weak field: canonicalize with Option.Some(Struct)
    let some_struct = Value::some(Value::Struct(target_obj.clone()));
    let can_some = layout.canonicalize_field_value(1, some_struct).unwrap();
    match &can_some {
        Value::Enum(obj) => {
            assert_eq!(obj.variant.as_str(), "Some");
            assert!(matches!(obj.values.as_ref().unwrap().first().unwrap(), Value::WeakStruct(_)));
        }
        _ => panic!("expected Option.Some(WeakStruct)"),
    }

    // Weak field: canonicalize with Option.Some(WeakStruct)
    let weak_ref = WeakStructRef::new(&target_obj);
    let some_weak = Value::some(Value::WeakStruct(weak_ref.clone()));
    let can_weak = layout.canonicalize_field_value(1, some_weak).unwrap();
    assert!(matches!(&can_weak, Value::Enum(obj) if obj.variant.as_str() == "Some"));

    // Weak field: canonicalize with Option.None or Nil -> Option.None
    let can_none = layout.canonicalize_field_value(1, Value::none()).unwrap();
    assert_eq!(can_none, Value::none());
    let can_nil = layout.canonicalize_field_value(1, Value::Nil).unwrap();
    assert_eq!(can_nil, Value::none());

    // Weak field: invalid non-struct value passed
    let err_int = layout.canonicalize_field_value(1, Value::some(Value::Int(100)));
    assert!(err_int.is_err());
    assert!(err_int.unwrap_err().contains("expects a struct reference"));

    // Weak field: invalid variant on Option enum
    let err_variant = layout.canonicalize_field_value(1, Value::enum_unit("Option", "Custom"));
    assert!(err_variant.is_err());
    assert!(err_variant.unwrap_err().contains("must store Option values; received variant 'Custom'"));

    // Materialize: alive weak struct inside Option.Some -> upgrades to Option.Some(Struct)
    let mat_alive = layout.materialize_field_value(1, Value::some(Value::WeakStruct(weak_ref.clone())));
    assert_eq!(mat_alive, Value::some(Value::Struct(target_obj.clone())));

    // Materialize: dead weak struct inside Option.Some -> becomes Option.None
    let dead_weak = {
        let temp_obj = StructObject::new("Person", dummy_layout.clone(), vec![]);
        WeakStructRef::new(&temp_obj)
    };
    let mat_dead = layout.materialize_field_value(1, Value::some(Value::WeakStruct(dead_weak)));
    assert_eq!(mat_dead, Value::none());

    // Materialize: Option.None or Nil -> Option.None
    assert_eq!(layout.materialize_field_value(1, Value::none()), Value::none());
    assert_eq!(layout.materialize_field_value(1, Value::Nil), Value::none());
    // Materialize other non-weak variant in Option.Some
    assert_eq!(layout.materialize_field_value(1, Value::some(Value::Int(99))), Value::some(Value::Int(99)));
}

// ============================================================================
// 11. StructObject, WeakStructRef Lifecycle & Debug Formatting
// ============================================================================

#[test]
fn test_struct_object_and_weak_struct_ref_lifecycle() {
    let dummy_layout = Rc::new(StructLayout::new(
        "Hero".to_string(),
        vec![Rc::new("hp".to_string())],
        vec![FieldStorage::Strong],
        vec![Type::new(TypeKind::Int, Span::dummy())],
        vec![None],
    ));
    let hero = StructObject::new("Hero", dummy_layout.clone(), vec![Value::Int(100)]);
    let weak = WeakStructRef::new(&hero);

    // When alive
    assert!(weak.upgrade().is_some());
    assert_eq!(weak.upgrade().unwrap(), Value::Struct(hero.clone()));
    assert_eq!(weak.struct_name().unwrap().as_str(), "Hero");

    // Debug formatting of StructObject
    let dbg = format!("{:?}", hero);
    assert!(dbg.contains("Struct { name: \"Hero\", fields: [Int(100)] }"), "got: {}", dbg);

    // Weak struct debug when alive
    let val_weak = Value::WeakStruct(weak.clone());
    let dbg_weak = format!("{:?}", val_weak);
    assert!(dbg_weak.contains("WeakStruct(Struct"), "got: {}", dbg_weak);

    // Dropping strong reference
    drop(hero);
    assert!(weak.upgrade().is_none());
    assert!(weak.struct_name().is_none());

    // Weak struct debug when dangling
    let dbg_dangling = format!("{:?}", Value::WeakStruct(weak.clone()));
    assert_eq!(dbg_dangling, "WeakStruct(<dangling>)");

    // Display when dangling
    let disp_dangling = format!("{}", Value::WeakStruct(weak));
    assert_eq!(disp_dangling, "nil");
}

// ============================================================================
// 12. Value Struct Field Accessors, Bounds Checks, and Error Formatting
// ============================================================================

#[test]
fn test_value_struct_field_accessors_and_errors() {
    let layout = Rc::new(StructLayout::new(
        "Vec2".to_string(),
        vec![Rc::new("x".to_string()), Rc::new("y".to_string())],
        vec![FieldStorage::Strong, FieldStorage::Strong],
        vec![Type::new(TypeKind::Int, Span::dummy()), Type::new(TypeKind::Int, Span::dummy())],
        vec![None, None],
    ));
    let point = Value::Struct(StructObject::new("Vec2", layout.clone(), vec![Value::Int(10), Value::Int(20)]));

    // Get field
    assert_eq!(point.struct_get_field("x"), Some(Value::Int(10)));
    assert_eq!(point.struct_get_field("y"), Some(Value::Int(20)));
    assert_eq!(point.struct_get_field("z"), None);
    assert_eq!(point.struct_get_field_rc(&Rc::new("x".to_string())), Some(Value::Int(10)));
    assert_eq!(point.struct_get_field_rc(&Rc::new("z".to_string())), None);
    assert_eq!(point.struct_get_field_indexed(0), Some(Value::Int(10)));
    assert_eq!(point.struct_get_field_indexed(1), Some(Value::Int(20)));
    assert_eq!(point.struct_get_field_indexed(2), None);

    // Set field
    assert!(point.struct_set_field("x", Value::Int(11)).is_ok());
    assert_eq!(point.struct_get_field("x"), Some(Value::Int(11)));

    assert!(point.struct_set_field_rc(&Rc::new("y".to_string()), Value::Int(22)).is_ok());
    assert_eq!(point.struct_get_field("y"), Some(Value::Int(22)));

    assert!(point.struct_set_field_indexed(0, Value::Int(33)).is_ok());
    assert_eq!(point.struct_get_field("x"), Some(Value::Int(33)));

    // Non-existent field errors
    let err_str = point.struct_set_field("bad", Value::Int(0));
    assert_eq!(err_str, Err("Struct 'Vec2' has no field 'bad'".to_string()));

    let err_rc = point.struct_set_field_rc(&Rc::new("bad".to_string()), Value::Int(0));
    assert_eq!(err_rc, Err("Struct 'Vec2' has no field 'bad'".to_string()));

    let err_idx = point.struct_set_field_indexed(5, Value::Int(0));
    assert_eq!(err_idx, Err("Struct 'Vec2' field index 5 out of bounds (len 2)".to_string()));

    // Accessors on non-struct values
    let non_struct = Value::Int(42);
    assert_eq!(non_struct.struct_get_field("x"), None);
    assert_eq!(non_struct.struct_get_field_rc(&Rc::new("x".to_string())), None);
    assert_eq!(non_struct.struct_get_field_indexed(0), None);
    assert_eq!(
        non_struct.struct_set_field("x", Value::Int(1)),
        Err("Attempted to set field on non-struct value".to_string())
    );
    assert_eq!(
        non_struct.struct_set_field_rc(&Rc::new("x".to_string()), Value::Int(1)),
        Err("Attempted to set field on non-struct value".to_string())
    );
    assert_eq!(
        non_struct.struct_set_field_indexed(0, Value::Int(1)),
        Err("Attempted to set field on non-struct value".to_string())
    );
}

// ============================================================================
// 13. Value Array & Map Accessors and Non-Array/Map Invariants
// ============================================================================

#[test]
fn test_value_array_and_map_accessors_and_errors() {
    // Array methods
    let arr = Value::array(vec![Value::Int(1), Value::Int(2)]);
    assert_eq!(arr.as_array(), Some(vec![Value::Int(1), Value::Int(2)]));
    assert_eq!(arr.array_len(), Some(2));
    assert_eq!(arr.array_get(0), Some(Value::Int(1)));
    assert_eq!(arr.array_get(1), Some(Value::Int(2)));
    assert_eq!(arr.array_get(2), None);

    assert!(arr.array_push(Value::Int(3)).is_ok());
    assert_eq!(arr.array_len(), Some(3));
    assert_eq!(arr.array_pop(), Ok(Some(Value::Int(3))));
    assert_eq!(arr.array_pop(), Ok(Some(Value::Int(2))));
    assert_eq!(arr.array_pop(), Ok(Some(Value::Int(1))));
    assert_eq!(arr.array_pop(), Ok(None));
    assert_eq!(arr.array_len(), Some(0));

    // Array methods on non-array
    let non_arr = Value::Bool(false);
    assert_eq!(non_arr.as_array(), None);
    assert_eq!(non_arr.array_len(), None);
    assert_eq!(non_arr.array_get(0), None);
    assert_eq!(non_arr.array_push(Value::Int(1)), Err("Cannot push to non-array".to_string()));
    assert_eq!(non_arr.array_pop(), Err("Cannot pop from non-array".to_string()));

    // Map methods on non-map
    assert_eq!(non_arr.as_map(), None);
    assert_eq!(non_arr.map_len(), None);
    assert_eq!(non_arr.map_get(&ValueKey::from(1i64)), None);
    assert_eq!(non_arr.map_has(&ValueKey::from(1i64)), None);
    assert_eq!(
        non_arr.map_set(ValueKey::from(1i64), Value::Nil),
        Err("Cannot set key on non-map".to_string())
    );
    assert_eq!(
        non_arr.map_delete(&ValueKey::from(1i64)),
        Err("Cannot delete key from non-map".to_string())
    );
}

// ============================================================================
// 14. Value Tuple, Enum, String, Task, and Numeric Query Methods
// ============================================================================

#[test]
fn test_value_tuple_enum_string_conversions_and_queries() {
    // String & Task accessors
    let s = Value::string("lust-lang");
    assert_eq!(s.as_string(), Some("lust-lang"));
    assert_eq!(s.as_string_rc(), Some(Rc::new("lust-lang".to_string())));
    assert_eq!(s.as_task_handle(), None);

    let t = Value::task(TaskHandle(101));
    assert_eq!(t.as_task_handle(), Some(TaskHandle(101)));
    assert_eq!(t.as_string(), None);
    assert_eq!(t.as_string_rc(), None);

    // Tuple accessors
    let tup = Value::tuple(vec![Value::Int(1), Value::string("two"), Value::Bool(true)]);
    assert_eq!(tup.tuple_len(), Some(3));
    assert_eq!(tup.tuple_get(0), Some(Value::Int(1)));
    assert_eq!(tup.tuple_get(1), Some(Value::string("two")));
    assert_eq!(tup.tuple_get(2), Some(Value::Bool(true)));
    assert_eq!(tup.tuple_get(3), None);

    let non_tup = Value::Int(7);
    assert_eq!(non_tup.tuple_len(), None);
    assert_eq!(non_tup.tuple_get(0), None);

    // Enum unit & variant accessors
    let unit = Value::enum_unit("Option", "None");
    assert_eq!(unit.as_enum(), Some(("Option", "None", None)));
    assert!(unit.is_enum_variant("Option", "None"));
    assert!(!unit.is_enum_variant("Option", "Some"));
    assert!(unit.is_enum_variant("", "None")); // empty enum_name matches any enum

    let variant = Value::enum_variant("Result", "Err", vec![Value::string("fatal")]);
    assert!(variant.is_enum_variant("Result", "Err"));
    assert!(!variant.is_enum_variant("Result", "Ok"));
    assert!(!variant.is_enum_variant("Option", "Err"));
    let (e_name, v_name, payload) = variant.as_enum().unwrap();
    assert_eq!(e_name, "Result");
    assert_eq!(v_name, "Err");
    assert_eq!(payload.unwrap(), &[Value::string("fatal")]);

    assert_eq!(non_tup.as_enum(), None);
    assert!(!non_tup.is_enum_variant("Result", "Ok"));

    // Option & Result constructors
    assert_eq!(Value::some(Value::Int(5)), Value::enum_variant("Option", "Some", vec![Value::Int(5)]));
    assert_eq!(Value::none(), Value::enum_unit("Option", "None"));
    assert_eq!(Value::ok(Value::Int(5)), Value::enum_variant("Result", "Ok", vec![Value::Int(5)]));
    assert_eq!(Value::err(Value::Int(5)), Value::enum_variant("Result", "Err", vec![Value::Int(5)]));

    // Numeric conversions
    assert_eq!(Value::Float(42.0).as_int(), Some(42));
    assert_eq!(Value::Int(42).as_float(), Some(42.0));
    assert_eq!(Value::string("bad").as_int(), None);
    assert_eq!(Value::string("bad").as_float(), None);
}

// ============================================================================
// 15. Name Interning, Hashing, Display, and Trait Implementations
// ============================================================================

#[test]
fn test_name_interning_hashing_and_traits() {
    let n1 = Name::from("symbol");
    let n2 = Name::from(&"symbol".to_string());
    let n3 = Name::from("symbol".to_string());

    // Interning oracle: pointer identity
    assert_eq!(n1, n2);
    assert_eq!(n2, n3);
    assert_eq!(n1.inner_ptr(), n2.inner_ptr());
    assert_eq!(n1.as_str(), "symbol");
    assert_eq!(&*n1, "symbol");
    assert_eq!(n1.as_ref(), "symbol");
    let borrowed: &str = std::borrow::Borrow::borrow(&n1);
    assert_eq!(borrowed, "symbol");

    // PartialEq cross-type with &str and String
    assert_eq!(n1, "symbol");
    assert_eq!(n1, "symbol".to_string());
    assert_eq!("symbol", &*n1);
    assert_eq!("symbol".to_string(), String::from(n1.clone()));

    // Hash consistency
    assert_eq!(hash_of(&n1), hash_of(&n2));

    // Display & Debug
    assert_eq!(format!("{}", n1), "symbol");
    assert_eq!(format!("{:?}", n1), "\"symbol\"");

    // From<Name> for String
    let string_out: String = String::from(n1.clone());
    assert_eq!(string_out, "symbol");
}

// ============================================================================
// 16. ValueKey LuaValue Unwrapping and All-Variant Hashing
// ============================================================================

#[test]
fn test_value_key_lua_unwrapping_and_variant_hashing() {
    // LuaValue unwrapping in ValueKey equality & hash
    let lua_nil = Value::enum_unit("LuaValue", "Nil");
    let lua_bool = Value::enum_variant("LuaValue", "Bool", vec![Value::Bool(true)]);
    let lua_int = Value::enum_variant("LuaValue", "Int", vec![Value::Int(42)]);
    let lua_num = Value::enum_variant("LuaValue", "Number", vec![Value::Int(42)]);
    let lua_str = Value::enum_variant("LuaValue", "String", vec![Value::string("lua")]);
    let lua_tbl = Value::enum_variant("LuaValue", "Table", vec![]);
    let lua_fn = Value::enum_variant("LuaValue", "Function", vec![]);
    let lua_other = Value::enum_variant("LuaValue", "UserData", vec![]);

    let k_nil = ValueKey::from(lua_nil);
    assert_eq!(k_nil, ValueKey::from(Value::Nil));
    assert_eq!(hash_of(&k_nil), hash_of(&ValueKey::from(Value::Nil)));

    let k_bool = ValueKey::from(lua_bool);
    assert_eq!(k_bool, ValueKey::from(true));
    assert_eq!(hash_of(&k_bool), hash_of(&ValueKey::from(true)));

    let k_int = ValueKey::from(lua_int);
    assert_eq!(k_int, ValueKey::from(42i64));
    assert_eq!(hash_of(&k_int), hash_of(&ValueKey::from(42i64)));

    let k_num = ValueKey::from(lua_num);
    assert_eq!(k_num, ValueKey::from(42i64));

    let k_str = ValueKey::from(lua_str);
    assert_eq!(k_str, ValueKey::from("lua"));
    assert_eq!(hash_of(&k_str), hash_of(&ValueKey::from("lua")));

    let k_tbl = ValueKey::from(lua_tbl);
    assert_eq!(k_tbl, ValueKey::from(Value::Nil));

    let k_fn = ValueKey::from(lua_fn);
    assert_eq!(k_fn, ValueKey::from(Value::Nil));

    let _k_other = ValueKey::from(lua_other);

    // ValueKey::from constructors
    let _ = ValueKey::from(false);
    let _ = ValueKey::from("str");
    let _ = ValueKey::from("str".to_string());
    let _ = ValueKey::from(Rc::new("str".to_string()));

    // Hashing each value variant into ValueKey
    let dummy_fn = native_fn(|_| Ok(NativeCallResult::Return(Value::Nil)));
    let dummy_closure = Rc::new(ClosureObject { function_idx: 1, upvalues: vec![] });
    let dummy_layout = Rc::new(StructLayout::new("S".to_string(), vec![], vec![], vec![], vec![]));
    let dummy_struct = StructObject::new("S", dummy_layout, vec![]);
    let dummy_weak = WeakStructRef::new(&dummy_struct);
    let dummy_iter = Rc::new(RefCell::new(IteratorState::Array { items: vec![], index: 0 }));

    let all_keys = vec![
        ValueKey::from(Value::Nil),
        ValueKey::from(Value::Bool(true)),
        ValueKey::from(Value::Int(123)),
        ValueKey::from(Value::Float(12.5)),
        ValueKey::from(Value::string("key")),
        ValueKey::from(Value::array(vec![Value::Int(1)])),
        ValueKey::from(Value::tuple(vec![Value::Int(1)])),
        ValueKey::from(Value::map(LustMap::default())),
        ValueKey::from(Value::Struct(dummy_struct.clone())),
        ValueKey::from(Value::WeakStruct(dummy_weak)),
        ValueKey::from(Value::enum_unit("Color", "Red")),
        ValueKey::from(Value::enum_variant("Color", "Rgb", vec![Value::Int(1)])),
        ValueKey::from(Value::Function(10)),
        ValueKey::from(Value::NativeFunction(dummy_fn)),
        ValueKey::from(Value::Closure(dummy_closure)),
        ValueKey::from(Value::Iterator(dummy_iter)),
        ValueKey::from(Value::Task(TaskHandle(55))),
    ];

    for key in &all_keys {
        let h = hash_of(key);
        assert_eq!(h, hash_of(key), "hash must be deterministic for key {:?}", key);
    }
}

// ============================================================================
// 17. Value PartialEq Cross-Layout, Weak References, and Iterators
// ============================================================================

#[test]
fn test_value_partial_eq_cross_layout_and_weak_semantics() {
    let layout_a = Rc::new(StructLayout::new(
        "Point".to_string(),
        vec![Rc::new("x".to_string()), Rc::new("y".to_string())],
        vec![FieldStorage::Strong, FieldStorage::Strong],
        vec![Type::new(TypeKind::Int, Span::dummy()), Type::new(TypeKind::Int, Span::dummy())],
        vec![None, None],
    ));
    let layout_b = Rc::new(StructLayout::new(
        "Point".to_string(),
        vec![Rc::new("x".to_string()), Rc::new("y".to_string())],
        vec![FieldStorage::Strong, FieldStorage::Strong],
        vec![Type::new(TypeKind::Int, Span::dummy()), Type::new(TypeKind::Int, Span::dummy())],
        vec![None, None],
    ));
    let layout_c = Rc::new(StructLayout::new(
        "Point".to_string(),
        vec![Rc::new("x".to_string()), Rc::new("z".to_string())],
        vec![FieldStorage::Strong, FieldStorage::Strong],
        vec![Type::new(TypeKind::Int, Span::dummy()), Type::new(TypeKind::Int, Span::dummy())],
        vec![None, None],
    ));

    let p_a = Value::Struct(StructObject::new("Point", layout_a.clone(), vec![Value::Int(1), Value::Int(2)]));
    let p_b = Value::Struct(StructObject::new("Point", layout_b.clone(), vec![Value::Int(1), Value::Int(2)]));
    let p_c = Value::Struct(StructObject::new("Point", layout_c.clone(), vec![Value::Int(1), Value::Int(2)]));
    let p_diff = Value::Struct(StructObject::new("Point", layout_b.clone(), vec![Value::Int(1), Value::Int(99)]));

    // Two distinct layout allocations with matching field names and values are equal!
    assert_eq!(p_a, p_b, "cross-layout structs with identical field names/values must be equal");
    // Different field names -> not equal
    assert_ne!(p_a, p_c, "cross-layout structs with differing field names must not be equal");
    // Different values -> not equal
    assert_ne!(p_a, p_diff, "cross-layout structs with differing values must not be equal");

    // WeakStruct PartialEq
    if let Value::Struct(ref s) = p_a {
        let weak_alive1 = WeakStructRef::new(s);
        let weak_alive2 = WeakStructRef::new(s);
        assert_eq!(Value::WeakStruct(weak_alive1.clone()), Value::WeakStruct(weak_alive2));
        assert_eq!(Value::WeakStruct(weak_alive1.clone()), p_a);
        assert_eq!(p_a, Value::WeakStruct(weak_alive1.clone()));
    }

    // Dead WeakStruct
    let dead_weak1 = {
        let s = StructObject::new("Point", layout_a.clone(), vec![Value::Int(1), Value::Int(2)]);
        WeakStructRef::new(&s)
    };
    let dead_weak2 = {
        let s = StructObject::new("Point", layout_a.clone(), vec![Value::Int(1), Value::Int(2)]);
        WeakStructRef::new(&s)
    };
    // (None, None) => true
    assert_eq!(Value::WeakStruct(dead_weak1.clone()), Value::WeakStruct(dead_weak2));
    // Dead weak struct equals Nil
    assert_eq!(Value::WeakStruct(dead_weak1.clone()), Value::Nil);
    assert_eq!(Value::Nil, Value::WeakStruct(dead_weak1.clone()));
    assert_ne!(Value::WeakStruct(dead_weak1), Value::Int(0));

    // Iterators are never equal
    let it1 = Value::Iterator(Rc::new(RefCell::new(IteratorState::Array { items: vec![], index: 0 })));
    let it2 = Value::Iterator(Rc::new(RefCell::new(IteratorState::Array { items: vec![], index: 0 })));
    assert_ne!(it1, it2);

    // Tasks: same handle vs different
    assert_eq!(Value::Task(TaskHandle(1)), Value::Task(TaskHandle(1)));
    assert_ne!(Value::Task(TaskHandle(1)), Value::Task(TaskHandle(2)));
}

// ============================================================================
// 18. Upvalue, NativeCallResult, and ValueTag Coverage
// ============================================================================

#[test]
fn test_upvalue_native_call_result_and_debug() {
    let upval = Upvalue::new(Value::Int(10));
    assert_eq!(upval.get(), Value::Int(10));
    upval.set(Value::Int(20));
    assert_eq!(upval.get(), Value::Int(20));
    let dbg = format!("{:?}", upval);
    assert!(dbg.contains("Upvalue(Int(20))"), "got: {}", dbg);

    // NativeCallResult
    let res_ret = NativeCallResult::Return(Value::Int(1));
    let res_yield = NativeCallResult::Yield(Value::Int(2));
    let res_stop = NativeCallResult::Stop(Value::Int(3));
    let res_from: NativeCallResult = Value::Int(42).into();
    assert!(matches!(res_from, NativeCallResult::Return(Value::Int(42))));
    let _ = format!("{:?} {:?} {:?}", res_ret, res_yield, res_stop);

    // ValueTag as_u8
    let tags = [
        (ValueTag::Nil, 0u8),
        (ValueTag::Bool, 1u8),
        (ValueTag::Int, 2u8),
        (ValueTag::Float, 3u8),
        (ValueTag::String, 4u8),
        (ValueTag::Array, 5u8),
        (ValueTag::Tuple, 6u8),
        (ValueTag::Map, 7u8),
        (ValueTag::Struct, 8u8),
        (ValueTag::Enum, 9u8),
        (ValueTag::Function, 10u8),
        (ValueTag::NativeFunction, 11u8),
        (ValueTag::Closure, 12u8),
        (ValueTag::Iterator, 13u8),
        (ValueTag::Task, 14u8),
    ];
    for (tag, expected_u8) in tags {
        assert_eq!(tag.as_u8(), expected_u8, "tag {:?} must match as_u8 {}", tag, expected_u8);
    }
}

// ============================================================================
// 19. Safe JIT Primitive Replacement and Frame Dropping Helpers
// ============================================================================

#[test]
fn test_jit_replace_and_return_value_helpers() {
    unsafe {
        // jit_replace_int
        assert_eq!(jit_replace_int(std::ptr::null_mut(), 10), 0);
        let mut slot = Value::Nil;
        assert_eq!(jit_replace_int(&mut slot, 42), 1);
        assert_eq!(slot, Value::Int(42));

        // jit_replace_int32
        assert_eq!(jit_replace_int32(std::ptr::null_mut(), 10), 0);
        assert_eq!(jit_replace_int32(&mut slot, -15), 1);
        assert_eq!(slot, Value::Int(-15));

        // jit_replace_float_bits
        assert_eq!(jit_replace_float_bits(std::ptr::null_mut(), 0), 0);
        let bits = 3.14159f64.to_bits();
        assert_eq!(jit_replace_float_bits(&mut slot, bits), 1);
        assert_eq!(slot, Value::Float(3.14159));

        // jit_replace_float32_bits
        assert_eq!(jit_replace_float32_bits(std::ptr::null_mut(), 0), 0);
        let bits32 = 2.5f32.to_bits();
        assert_eq!(jit_replace_float32_bits(&mut slot, bits32), 1);
        assert_eq!(slot, Value::Float(2.5));

        // jit_replace_bool
        assert_eq!(jit_replace_bool(std::ptr::null_mut(), 1), 0);
        assert_eq!(jit_replace_bool(&mut slot, 1), 1);
        assert_eq!(slot, Value::Bool(true));
        assert_eq!(jit_replace_bool(&mut slot, 0), 1);
        assert_eq!(slot, Value::Bool(false));

        // jit_replace_nil
        assert_eq!(jit_replace_nil(std::ptr::null_mut()), 0);
        slot = Value::Int(99);
        assert_eq!(jit_replace_nil(&mut slot), 1);
        assert_eq!(slot, Value::Nil);

        // jit_init_nil
        assert_eq!(jit_init_nil(std::ptr::null_mut()), 0);
        let mut uninit_slot: std::mem::MaybeUninit<Value> = std::mem::MaybeUninit::uninit();
        assert_eq!(jit_init_nil(uninit_slot.as_mut_ptr()), 1);
        let inited = uninit_slot.assume_init();
        assert_eq!(inited, Value::Nil);

        // jit_return_value
        let mut dest = Value::Int(1);
        jit_return_value(std::ptr::null_mut(), &mut dest);
        assert_eq!(dest, Value::Nil);

        let mut src = Value::string("return_val");
        jit_return_value(&mut src, &mut dest);
        assert_eq!(dest, Value::string("return_val"));
        assert_eq!(src, Value::Nil);

        // jit_drop_values_masked
        jit_drop_values_masked(std::ptr::null_mut(), 5, 0);
        let mut values = [
            Value::string("reg0"),
            Value::string("reg1"),
            Value::string("reg2"),
        ];
        // Mask register 1 (bit 1 set)
        jit_drop_values_masked(values.as_mut_ptr(), 3, 0b010);
        // After drop, only slot 1 is still logically owned; forget values so double-drop doesn't occur
        std::mem::forget(values);

        // jit_drop_values
        jit_drop_values(std::ptr::null_mut(), 0);
        let mut values2 = [Value::string("a"), Value::string("b")];
        jit_drop_values(values2.as_mut_ptr(), 2);
        std::mem::forget(values2);
    }
}

// ============================================================================
// 20. Safe JIT Array Operations and Bounds Handling
// ============================================================================

#[test]
fn test_jit_array_operations_safe() {
    let mut vm = VM::new();
    let vm_ptr = &mut vm as *mut VM;

    unsafe {
        // jit_array_len_safe
        assert_eq!(jit_array_len_safe(std::ptr::null()), -1);
        let not_arr = Value::Int(5);
        assert_eq!(jit_array_len_safe(&not_arr), -1);
        let arr = Value::array(vec![Value::Int(10), Value::Int(20), Value::Int(30)]);
        assert_eq!(jit_array_len_safe(&arr), 3);

        // jit_new_array_safe
        assert_eq!(jit_new_array_safe(vm_ptr, std::ptr::null(), 0, std::ptr::null_mut()), 0);
        let mut new_arr_out = Value::Nil;
        // Empty array
        assert_eq!(jit_new_array_safe(vm_ptr, std::ptr::null(), 0, &mut new_arr_out), 1);
        assert_eq!(new_arr_out.array_len(), Some(0));

        // Array with elements
        let elems = [Value::Int(100), Value::Int(200)];
        assert_eq!(jit_new_array_safe(vm_ptr, elems.as_ptr(), 2, &mut new_arr_out), 1);
        assert_eq!(new_arr_out.array_len(), Some(2));
        assert_eq!(new_arr_out.array_get(0), Some(Value::Int(100)));

        // jit_array_push_safe
        assert_eq!(jit_array_push_safe(vm_ptr, std::ptr::null(), std::ptr::null()), 0);
        assert_eq!(jit_array_push_safe(vm_ptr, &not_arr, &elems[0]), 0);
        assert_eq!(jit_array_push_safe(vm_ptr, &new_arr_out, &Value::Int(300)), 1);
        assert_eq!(new_arr_out.array_len(), Some(3));

        // jit_array_get_safe
        let mut out = Value::Nil;
        assert_eq!(jit_array_get_safe(std::ptr::null_mut(), &new_arr_out, &Value::Int(0), &mut out), 0);
        assert_eq!(jit_array_get_safe(vm_ptr, &not_arr, &Value::Int(0), &mut out), 0);
        assert_eq!(jit_array_get_safe(vm_ptr, &new_arr_out, &Value::string("bad"), &mut out), 0);
        // Valid get
        assert_eq!(jit_array_get_safe(vm_ptr, &new_arr_out, &Value::Int(1), &mut out), 1);
        assert_eq!(out, Value::Int(200));
        // Out of bounds get
        assert_eq!(jit_array_get_safe(vm_ptr, &new_arr_out, &Value::Int(99), &mut out), 0);

        // jit_array_index_ok_safe
        let mut val_out = Value::Nil;
        let mut cond_out = Value::Nil;
        assert_eq!(jit_array_index_ok_safe(std::ptr::null(), &Value::Int(0), &mut val_out, &mut cond_out), 0);
        assert_eq!(jit_array_index_ok_safe(&not_arr, &Value::Int(0), &mut val_out, &mut cond_out), 0);
        // Valid in-bounds
        assert_eq!(jit_array_index_ok_safe(&new_arr_out, &Value::Int(0), &mut val_out, &mut cond_out), 1);
        assert_eq!(val_out, Value::Int(100));
        assert_eq!(cond_out, Value::Bool(true));
        // Out-of-bounds
        assert_eq!(jit_array_index_ok_safe(&new_arr_out, &Value::Int(99), &mut val_out, &mut cond_out), 1);
        assert_eq!(val_out, Value::Nil);
        assert_eq!(cond_out, Value::Bool(false));

        // jit_array_set_index_safe
        assert_eq!(jit_array_set_index_safe(std::ptr::null(), &Value::Int(0), &Value::Int(1)), 0);
        assert_eq!(jit_array_set_index_safe(&not_arr, &Value::Int(0), &Value::Int(1)), 0);
        assert_eq!(jit_array_set_index_safe(&new_arr_out, &Value::Int(99), &Value::Int(1)), 0);
        // Valid set
        assert_eq!(jit_array_set_index_safe(&new_arr_out, &Value::Int(0), &Value::Int(777)), 1);
        assert_eq!(new_arr_out.array_get(0), Some(Value::Int(777)));
    }
}

// ============================================================================
// 21. Safe JIT Specialized Vec Slots (int, bool, float unbox/rebox)
// ============================================================================

#[test]
fn test_jit_specialized_vec_slots() {
    unsafe {
        // Int array unbox/rebox/push/drop
        let arr_int = Value::array(vec![Value::Int(10), Value::Int(20)]);
        let mut slot_int = JitVecSlot {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
            array: std::ptr::null(),
        };
        assert_eq!(jit_unbox_array_int(std::ptr::null(), &mut slot_int), 0);
        assert_eq!(jit_unbox_array_int(&arr_int, &mut slot_int), 1);
        assert_eq!(slot_int.len, 2);
        assert_eq!(jit_vec_int_push(&mut slot_int.ptr, &mut slot_int.len, &mut slot_int.cap, 30), 1);
        assert_eq!(slot_int.len, 3);
        assert_eq!(jit_rebox_array_int(&mut slot_int), 1);
        assert_eq!(arr_int.as_array(), Some(vec![Value::Int(10), Value::Int(20), Value::Int(30)]));
        jit_drop_vec_int(&mut slot_int);

        // Bool array unbox/rebox/push/drop
        let arr_bool = Value::array(vec![Value::Bool(true), Value::Bool(false)]);
        let mut slot_bool = JitVecSlot {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
            array: std::ptr::null(),
        };
        assert_eq!(jit_unbox_array_bool(std::ptr::null(), &mut slot_bool), 0);
        assert_eq!(jit_unbox_array_bool(&arr_bool, &mut slot_bool), 1);
        assert_eq!(slot_bool.len, 2);
        let mut bool_ptr = slot_bool.ptr as *mut u8;
        assert_eq!(jit_vec_bool_push(&mut bool_ptr, &mut slot_bool.len, &mut slot_bool.cap, 1), 1);
        slot_bool.ptr = bool_ptr as *mut LustInt;
        assert_eq!(jit_rebox_array_bool(&mut slot_bool), 1);
        assert_eq!(arr_bool.as_array(), Some(vec![Value::Bool(true), Value::Bool(false), Value::Bool(true)]));
        jit_drop_vec_bool(&mut slot_bool);

        // Float array unbox/rebox/push/drop
        let arr_fl = Value::array(vec![Value::Float(1.5), Value::Float(2.5)]);
        let mut slot_fl = JitVecSlot {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
            array: std::ptr::null(),
        };
        assert_eq!(jit_unbox_array_float(std::ptr::null(), &mut slot_fl), 0);
        assert_eq!(jit_unbox_array_float(&arr_fl, &mut slot_fl), 1);
        assert_eq!(slot_fl.len, 2);
        let mut float_ptr = slot_fl.ptr as *mut LustFloat;
        assert_eq!(jit_vec_float_push(&mut float_ptr, &mut slot_fl.len, &mut slot_fl.cap, 3.5), 1);
        slot_fl.ptr = float_ptr as *mut LustInt;
        assert_eq!(jit_rebox_array_float(&mut slot_fl), 1);
        assert_eq!(arr_fl.as_array(), Some(vec![Value::Float(1.5), Value::Float(2.5), Value::Float(3.5)]));
        jit_drop_vec_float(&mut slot_fl);
    }
}

// ============================================================================
// 22. Safe JIT Enum & Struct Field Helpers
// ============================================================================

#[test]
fn test_jit_enum_and_struct_field_helpers() {
    let mut vm = VM::new();
    let vm_ptr = &mut vm as *mut VM;

    unsafe {
        let mut out = Value::Nil;

        // jit_enum_is_some_safe
        assert_eq!(jit_enum_is_some_safe(std::ptr::null(), &mut out), 0);
        let opt_some = Value::some(Value::Int(42));
        let opt_none = Value::none();
        assert_eq!(jit_enum_is_some_safe(&opt_some, &mut out), 1);
        assert_eq!(out, Value::Bool(true));
        assert_eq!(jit_enum_is_some_safe(&opt_none, &mut out), 1);
        assert_eq!(out, Value::Bool(false));

        // jit_enum_unwrap_safe
        assert_eq!(jit_enum_unwrap_safe(vm_ptr, &opt_some, &mut out), 1);
        assert_eq!(out, Value::Int(42));
        let res_ok = Value::ok(Value::string("success"));
        assert_eq!(jit_enum_unwrap_safe(vm_ptr, &res_ok, &mut out), 1);
        assert_eq!(out, Value::string("success"));
        assert_eq!(jit_enum_unwrap_safe(vm_ptr, &opt_none, &mut out), 0);

        // jit_value_is_truthy
        assert_eq!(jit_value_is_truthy(std::ptr::null()), 0);
        assert_eq!(jit_value_is_truthy(&Value::Nil), 0);
        assert_eq!(jit_value_is_truthy(&Value::Bool(false)), 0);
        assert_eq!(jit_value_is_truthy(&opt_none), 0);
        assert_eq!(jit_value_is_truthy(&Value::Bool(true)), 1);
        assert_eq!(jit_value_is_truthy(&Value::Int(1)), 1);

        // jit_new_enum_unit_safe
        let e_name = "State";
        let v_name = "Idle";
        assert_eq!(
            jit_new_enum_unit_safe(
                vm_ptr,
                e_name.as_ptr(),
                e_name.len(),
                v_name.as_ptr(),
                v_name.len(),
                &mut out
            ),
            1
        );
        assert_eq!(out, Value::enum_unit("State", "Idle"));

        // jit_new_enum_variant_safe
        let payload = [Value::Int(77)];
        let v2_name = "Active";
        assert_eq!(
            jit_new_enum_variant_safe(
                vm_ptr,
                e_name.as_ptr(),
                e_name.len(),
                v2_name.as_ptr(),
                v2_name.len(),
                payload.as_ptr(),
                1,
                &mut out
            ),
            1
        );
        assert_eq!(out, Value::enum_variant("State", "Active", vec![Value::Int(77)]));

        // jit_new_option_some_fast
        assert_eq!(jit_new_option_some_fast(vm_ptr, &Value::Int(999), &mut out), 1);
        assert_eq!(out, Value::some(Value::Int(999)));

        // jit_is_enum_variant_safe
        assert_eq!(
            jit_is_enum_variant_safe(&out, "Option".as_ptr(), 6, "Some".as_ptr(), 4),
            1
        );
        assert_eq!(
            jit_is_enum_variant_safe(&out, "Option".as_ptr(), 6, "None".as_ptr(), 4),
            0
        );

        // jit_get_enum_value_safe
        assert_eq!(jit_get_enum_value_safe(&out, 0, &mut out), 1);
        assert_eq!(out, Value::Int(999));

        // jit_concat_safe
        let str_left = Value::string("foo");
        let str_right = Value::string("bar");
        assert_eq!(jit_concat_safe(vm_ptr, &str_left, &str_right, &mut out), 1);
        assert_eq!(out, Value::string("foobar"));

        // jit_set_field_strong_safe and indexed safe helpers
        let layout = Rc::new(StructLayout::new(
            "Point".to_string(),
            vec![Rc::new("x".to_string()), Rc::new("y".to_string())],
            vec![FieldStorage::Strong, FieldStorage::Strong],
            vec![Type::new(TypeKind::Int, Span::dummy()), Type::new(TypeKind::Int, Span::dummy())],
            vec![None, None],
        ));
        let point_val = Value::Struct(StructObject::new("Point", layout.clone(), vec![Value::Int(10), Value::Int(20)]));

        assert_eq!(jit_set_field_strong_safe(&point_val, 0, &Value::Int(15)), 1);
        assert_eq!(point_val.struct_get_field("x"), Some(Value::Int(15)));
        assert_eq!(jit_set_field_strong_safe(&point_val, 99, &Value::Int(15)), 0);

        // jit_get_field_safe
        assert_eq!(jit_get_field_safe(&point_val, "x".as_ptr(), 1, &mut out), 1);
        assert_eq!(out, Value::Int(15));
        assert_eq!(jit_get_field_safe(&point_val, "z".as_ptr(), 1, &mut out), 0);

        // jit_set_field_safe
        assert_eq!(jit_set_field_safe(&point_val, "y".as_ptr(), 1, &Value::Int(25)), 1);
        assert_eq!(point_val.struct_get_field("y"), Some(Value::Int(25)));

        // jit_get_field_indexed_safe and jit_set_field_indexed_safe
        assert_eq!(jit_get_field_indexed_safe(&point_val, 1, &mut out), 1);
        assert_eq!(out, Value::Int(25));
        assert_eq!(jit_set_field_indexed_safe(&point_val, 1, &Value::Int(30)), 1);
        assert_eq!(point_val.struct_get_field("y"), Some(Value::Int(30)));

        // jit_get_field_indexed_int_fast and jit_set_field_indexed_int_fast
        assert_eq!(jit_get_field_indexed_int_fast(&point_val, 0, &mut out), 1);
        assert_eq!(out, Value::Int(15));
        assert_eq!(jit_set_field_indexed_int_fast(&point_val, 0, &Value::Int(50)), 1);
        assert_eq!(point_val.struct_get_field("x"), Some(Value::Int(50)));

        // jit_get_field_keyed on Map
        let map_val = Value::map(LustMap::default());
        map_val.map_set(ValueKey::string("key"), Value::Int(42)).unwrap();
        let k = ValueKey::string("key");
        assert_eq!(jit_get_field_keyed(&map_val, "key".as_ptr(), 3, &k, &mut out), 1);
        assert_eq!(out, Value::Int(42));

        // jit_retain_value, jit_release_value, jit_move_safe, jit_set_global_safe
        let mut target = Value::Nil;
        assert_eq!(jit_move_safe(&out, &mut target), 1);
        assert_eq!(target, Value::Int(42));
        jit_retain_value(&target);
        jit_release_value(&mut target);
        assert_eq!(target, Value::Nil);

        let g_name = "my_global";
        assert_eq!(
            jit_set_global_safe(vm_ptr, g_name.as_ptr(), g_name.len(), &Value::Int(9999)),
            1
        );
        assert_eq!(vm.get_global("my_global"), Some(Value::Int(9999)));
    }
}

// ============================================================================
// 23. Safe JIT Guards and Builtin Method Dispatch
// ============================================================================

#[test]
fn test_jit_guards_and_method_calls_safe() {
    let mut vm = VM::new();
    let vm_ptr = &mut vm as *mut VM;

    unsafe {
        // jit_guard_struct_layout
        let layout = Rc::new(StructLayout::new(
            "Point".to_string(),
            vec![],
            vec![],
            vec![],
            vec![],
        ));
        let struct_val = Value::Struct(StructObject::new("Point", layout.clone(), vec![]));
        let layout_ptr = Rc::as_ptr(&layout) as *const ();
        assert_eq!(jit_guard_struct_layout(&struct_val, layout_ptr), 1);
        let dummy_ptr = &() as *const ();
        assert_eq!(jit_guard_struct_layout(&struct_val, dummy_ptr), 0);
        assert_eq!(jit_guard_struct_layout(&Value::Nil, layout_ptr), 0);

        // jit_guard_native_function
        let native = native_fn(|_| Ok(NativeCallResult::Return(Value::Nil)));
        let native_val = Value::NativeFunction(native.clone());
        let native_ptr = Rc::as_ptr(&native) as *const ();
        assert_eq!(jit_guard_native_function(&native_val, native_ptr, 0), 1);
        assert_eq!(jit_guard_native_function(&native_val, dummy_ptr, 0), 0);
        assert_eq!(jit_guard_native_function(&Value::Nil, native_ptr, 0), 0);

        // jit_guard_function_identity
        let func_val = Value::Function(42);
        assert_eq!(jit_guard_function_identity(&func_val, 0, 42, std::ptr::null(), 0), 1);
        assert_eq!(jit_guard_function_identity(&func_val, 0, 99, std::ptr::null(), 0), 0);

        // jit_call_method_safe: Option methods (is_some, is_none, unwrap)
        let opt_some = Value::some(Value::Int(10));
        let mut out = Value::Nil;
        assert_eq!(
            jit_call_method_safe(
                vm_ptr,
                &opt_some,
                "is_some".as_ptr(),
                7,
                std::ptr::null(),
                0,
                0,
                &mut out
            ),
            1
        );
        assert_eq!(out, Value::Bool(true));

        assert_eq!(
            jit_call_method_safe(
                vm_ptr,
                &opt_some,
                "is_none".as_ptr(),
                7,
                std::ptr::null(),
                0,
                0,
                &mut out
            ),
            1
        );
        assert_eq!(out, Value::Bool(false));

        assert_eq!(
            jit_call_method_safe(
                vm_ptr,
                &opt_some,
                "unwrap".as_ptr(),
                6,
                std::ptr::null(),
                0,
                0,
                &mut out
            ),
            1
        );
        assert_eq!(out, Value::Int(10));

        // jit_call_method_safe: Result methods (is_ok, is_err, unwrap, unwrap_or)
        let res_err = Value::err(Value::string("failure"));
        assert_eq!(
            jit_call_method_safe(
                vm_ptr,
                &res_err,
                "is_ok".as_ptr(),
                5,
                std::ptr::null(),
                0,
                0,
                &mut out
            ),
            1
        );
        assert_eq!(out, Value::Bool(false));

        assert_eq!(
            jit_call_method_safe(
                vm_ptr,
                &res_err,
                "is_err".as_ptr(),
                6,
                std::ptr::null(),
                0,
                0,
                &mut out
            ),
            1
        );
        assert_eq!(out, Value::Bool(true));

        let default_arg = [Value::Int(42)];
        assert_eq!(
            jit_call_method_safe(
                vm_ptr,
                &res_err,
                "unwrap_or".as_ptr(),
                9,
                default_arg.as_ptr(),
                1,
                0,
                &mut out
            ),
            1
        );
        assert_eq!(out, Value::Int(42));

        // Iterator next
        let iter_val = Value::Iterator(Rc::new(RefCell::new(IteratorState::Array {
            items: vec![Value::Int(1)],
            index: 0,
        })));
        assert_eq!(
            jit_call_method_safe(
                vm_ptr,
                &iter_val,
                "next".as_ptr(),
                4,
                std::ptr::null(),
                0,
                0,
                &mut out
            ),
            1
        );
        assert_eq!(out, Value::some(Value::Int(1)));
        assert_eq!(
            jit_call_method_safe(
                vm_ptr,
                &iter_val,
                "next".as_ptr(),
                4,
                std::ptr::null(),
                0,
                0,
                &mut out
            ),
            1
        );
        assert_eq!(out, Value::none());

        // jit_array_index_result_safe
        let test_arr = Value::array(vec![Value::Int(77)]);
        assert_eq!(jit_array_index_result_safe(vm_ptr, &test_arr, &Value::Int(0), &mut out), 1);
        assert_eq!(out, Value::ok(Value::Int(77)));
        assert_eq!(jit_array_index_result_safe(vm_ptr, &test_arr, &Value::Int(5), &mut out), 1);
        assert!(matches!(out, Value::Enum(ref obj) if obj.variant == "Err"));

        // jit_type_is_safe & jit_try_cast_safe
        let val_int = Value::Int(10);
        assert_eq!(jit_type_is_safe(vm_ptr, &val_int, "int".as_ptr(), 3), 1);
        assert_eq!(jit_type_is_safe(vm_ptr, &val_int, "string".as_ptr(), 6), 0);
        assert_eq!(jit_try_cast_safe(vm_ptr, &val_int, "int".as_ptr(), 3, &mut out), 1);
        assert_eq!(out, Value::some(Value::Int(10)));
        assert_eq!(jit_try_cast_safe(vm_ptr, &val_int, "string".as_ptr(), 6, &mut out), 1);
        assert_eq!(out, Value::none());

        // EnumObject debug
        let enum_obj = EnumObject::new("Status", "Ready", Some(vec![Value::Int(1)]));
        assert!(format!("{:?}", enum_obj).contains("Enum { enum: \"Status\", variant: \"Ready\""));
    }
}


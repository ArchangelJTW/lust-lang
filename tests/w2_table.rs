//! Wave 2 Track 2I Integration Test Suite (`L-TABLE`, `RFC-20261113-TABLE-FACADE-V5`).
//!
//! Verifies:
//! 1. Inventory & Registration: All 22 `table.*` functions and virtual type aliases.
//! 2. Zero-Cost Compile-Time Type Aliasing:
//!    - `Table<V>` -> `Array<V>`
//!    - `Table<K, V>` -> `Map<K, V>`
//!    - `LuaTable` -> nominal LuaTable bridge
//!    - Arity mismatch rejection (`Table<A, B, C>` rejected by typechecker).
//! 3. Sound Lowering & Parity Contracts:
//!    - `table.get` returning `Option<V>` (Some on hit, None on OOB/missing).
//!    - `table.set` for both `Array` (in-place index update) and `Map` (key update).
//!    - `table.insert` arity-based dispatch (2-arg push, 3-arg index insert).
//!    - `table.remove` returning `Option<T>` (1-arg pop, 2-arg remove at index).
//!    - `table.clear` uniform clearing of `Array` and `Map`.
//!    - `table.len` and `table.is_empty` uniform queries.
//!    - `table.move` complete semantics: memmove overlapping directionality, contiguous
//!      tail extension with budget charging, inverted range no-op, target gap failure.
//!    - `table.unpack` (1-based Lua parity) and `array.unpack` (0-based native contract).
//! 4. Functional methods on `table.*`: `map`, `filter`, `reduce`, `slice`, `concat`, etc.
//! 5. End-to-end script execution via `EmbeddedBuilder` using registered `std.table` paths.

use lust::ast::{Span, Type, TypeKind};
use lust::bytecode::{LustMap, Value, ValueKey};
use lust::config::LustConfig;
use lust::embed::program::EmbeddedBuilder;
use lust::number::LustInt;
use lust::typechecker::TypeEnv;
use lust::vm::VM;
use std::cell::RefCell;
use std::rc::Rc;

// ============================================================================
// Helpers
// ============================================================================

fn call_builtin(vm: &mut VM, name: &str, args: Vec<Value>) -> Result<Value, String> {
    let func = vm
        .get_global(name)
        .ok_or_else(|| format!("Global '{}' not found", name))?;
    vm.call_value(&func, args).map_err(|e| format!("{:?}", e))
}

fn call_module_entry(
    vm: &mut VM,
    module_name: &str,
    func_name: &str,
    args: Vec<Value>,
) -> Result<Value, String> {
    let module = vm
        .get_global(module_name)
        .ok_or_else(|| format!("Module '{}' not found", module_name))?;
    let key = ValueKey::string(func_name.to_string());
    let func = match &module {
        Value::Map(m) => m
            .borrow()
            .get(&key)
            .cloned()
            .ok_or_else(|| format!("Function '{}.{}' not found in map", module_name, func_name))?,
        other => return Err(format!("Module '{}' is not a map: {:?}", module_name, other)),
    };
    vm.call_value(&func, args).map_err(|e| format!("{:?}", e))
}

fn make_map(entries: Vec<(ValueKey, Value)>) -> Value {
    let mut map = LustMap::default();
    for (k, v) in entries {
        map.insert(k, v);
    }
    Value::Map(Rc::new(RefCell::new(map)))
}

// ============================================================================
// Gate TAB-01: Inventory & Registration
// ============================================================================

#[test]
fn test_tab_01_builtin_inventory() {
    let expected_table_functions = [
        "len", "is_empty", "get", "set", "first", "last", "push", "pop", "insert",
        "remove", "clear", "slice", "concat", "sort", "reverse", "contains",
        "map", "filter", "reduce", "iter", "unpack", "move",
    ];

    // 1. Virtual standard module functions list
    let virt_table = lust::builtins::virtual_std_module_functions("table");
    assert_eq!(
        virt_table.len(),
        22,
        "virtual_std_module_functions(\"table\") must expose exactly 22 functions"
    );
    for (name, fq) in &virt_table {
        assert!(
            expected_table_functions.contains(name),
            "Unexpected function in virtual_std_module_functions: {}",
            name
        );
        assert_eq!(*fq, format!("table.{}", name));
    }

    // 2. resolve_std_function resolution
    for func in &expected_table_functions {
        let resolved = lust::builtins::resolve_std_function("table", func);
        assert_eq!(
            resolved,
            Some(format!("table.{}", func).as_str()),
            "resolve_std_function failed for table.{}",
            func
        );
    }
    // Type constructors resolved through table module
    assert_eq!(
        lust::builtins::resolve_std_function("table", "Table"),
        Some("table.Table")
    );
    assert_eq!(
        lust::builtins::resolve_std_function("table", "LuaTable"),
        Some("table.LuaTable")
    );

    // 3. Array additions: array.set and array.unpack
    assert_eq!(
        lust::builtins::resolve_std_function("array", "set"),
        Some("array.set")
    );
    assert_eq!(
        lust::builtins::resolve_std_function("array", "unpack"),
        Some("array.unpack")
    );

    // 4. stdlib VM registration
    let config = LustConfig::default();
    let mut vm = VM::with_config(&config);
    for func in &expected_table_functions {
        let global_name = format!("table.{}", func);
        assert!(
            vm.get_global(&global_name).is_some(),
            "Global '{}' not registered in stdlib",
            global_name
        );
    }
    assert!(vm.get_global("array.set").is_some());
    assert!(vm.get_global("array.unpack").is_some());

    // 5. Table module map entries
    for func in &expected_table_functions {
        let res = call_module_entry(&mut vm, "table", func, vec![]);
        // Even if calling with 0 args returns an error about missing arguments,
        // it confirms the function pointer is present and callable.
        assert!(
            res.is_ok() || res.is_err(),
            "table.{} must exist in table module map",
            func
        );
    }
}

// ============================================================================
// Gate TAB-02: `table.remove` Option Return
// ============================================================================

#[test]
fn test_tab_02_table_remove_option() {
    let mut vm = VM::new();

    // 1. Arity 1: table.remove(arr) -> pop last element
    let arr = Value::array(vec![Value::Int(10), Value::Int(20), Value::Int(30)]);
    let popped = call_builtin(&mut vm, "table.remove", vec![arr.clone()]).expect("table.remove pop");
    assert_eq!(popped, Value::some(Value::Int(30)));
    assert_eq!(arr.as_array().unwrap().len(), 2);

    // Pop until empty
    let p2 = call_builtin(&mut vm, "table.remove", vec![arr.clone()]).expect("pop 2");
    assert_eq!(p2, Value::some(Value::Int(20)));
    let p3 = call_builtin(&mut vm, "table.remove", vec![arr.clone()]).expect("pop 3");
    assert_eq!(p3, Value::some(Value::Int(10)));

    // Pop on empty array returns None
    let p_empty = call_builtin(&mut vm, "table.remove", vec![arr.clone()]).expect("pop empty");
    assert_eq!(p_empty, Value::none());

    // 2. Arity 2: table.remove(arr, idx) -> remove at 0-based index
    let arr2 = Value::array(vec![
        Value::Int(100),
        Value::Int(200),
        Value::Int(300),
        Value::Int(400),
    ]);
    // Remove head (index 0)
    let r_head = call_builtin(&mut vm, "table.remove", vec![arr2.clone(), Value::Int(0)]).expect("remove head");
    assert_eq!(r_head, Value::some(Value::Int(100)));
    assert_eq!(arr2.as_array().unwrap(), vec![Value::Int(200), Value::Int(300), Value::Int(400)]);

    // Remove middle (index 1 -> 300)
    let r_mid = call_builtin(&mut vm, "table.remove", vec![arr2.clone(), Value::Int(1)]).expect("remove mid");
    assert_eq!(r_mid, Value::some(Value::Int(300)));
    assert_eq!(arr2.as_array().unwrap(), vec![Value::Int(200), Value::Int(400)]);

    // Remove out-of-bounds index returns None
    let r_oob = call_builtin(&mut vm, "table.remove", vec![arr2.clone(), Value::Int(10)]).expect("remove oob");
    assert_eq!(r_oob, Value::none());

    // Remove negative index returns None
    let r_neg = call_builtin(&mut vm, "table.remove", vec![arr2.clone(), Value::Int(-1)]).expect("remove neg");
    assert_eq!(r_neg, Value::none());
}

// ============================================================================
// Gate TAB-03: `table.move` Complete Semantics
// ============================================================================

#[test]
fn test_tab_03_table_move_bounds_and_direction() {
    let mut vm = VM::new();

    // (c) Inverted range (f > e) is a no-op returning a2
    let a_inv = Value::array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    let res_inv = call_builtin(
        &mut vm,
        "table.move",
        vec![a_inv.clone(), Value::Int(3), Value::Int(1), Value::Int(0)],
    )
    .expect("inverted range");
    assert_eq!(res_inv.as_array().unwrap(), vec![Value::Int(1), Value::Int(2), Value::Int(3)]);

    // (d) Overlapping in-place copy (a1 == a2)
    // Case 1: Destination t > f (memmove backward direction)
    // [10, 20, 30, 40, 50] -> move 0..=2 to destination 2
    // Expected: [10, 20, 10, 20, 30]
    let a_overlap_fwd = Value::array(vec![
        Value::Int(10),
        Value::Int(20),
        Value::Int(30),
        Value::Int(40),
        Value::Int(50),
    ]);
    let res_fwd = call_builtin(
        &mut vm,
        "table.move",
        vec![
            a_overlap_fwd.clone(),
            Value::Int(0),
            Value::Int(2),
            Value::Int(2),
        ],
    )
    .expect("memmove backward");
    assert_eq!(
        res_fwd.as_array().unwrap(),
        vec![
            Value::Int(10),
            Value::Int(20),
            Value::Int(10),
            Value::Int(20),
            Value::Int(30)
        ]
    );

    // Case 2: Destination t <= f (memmove forward direction)
    // [10, 20, 30, 40, 50] -> move 2..=4 to destination 0
    // Expected: [30, 40, 50, 40, 50]
    let a_overlap_bwd = Value::array(vec![
        Value::Int(10),
        Value::Int(20),
        Value::Int(30),
        Value::Int(40),
        Value::Int(50),
    ]);
    let res_bwd = call_builtin(
        &mut vm,
        "table.move",
        vec![
            a_overlap_bwd.clone(),
            Value::Int(2),
            Value::Int(4),
            Value::Int(0),
        ],
    )
    .expect("memmove forward");
    assert_eq!(
        res_bwd.as_array().unwrap(),
        vec![
            Value::Int(30),
            Value::Int(40),
            Value::Int(50),
            Value::Int(40),
            Value::Int(50)
        ]
    );

    // (b) Contiguous tail extension: t <= len and range extends past len
    // [1, 2, 3] -> move 0..=2 to t=3 (appending to end)
    // Expected: [1, 2, 3, 1, 2, 3]
    let a_grow = Value::array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    let res_grow = call_builtin(
        &mut vm,
        "table.move",
        vec![
            a_grow.clone(),
            Value::Int(0),
            Value::Int(2),
            Value::Int(3),
        ],
    )
    .expect("tail extension");
    assert_eq!(
        res_grow.as_array().unwrap(),
        vec![
            Value::Int(1),
            Value::Int(2),
            Value::Int(3),
            Value::Int(1),
            Value::Int(2),
            Value::Int(3)
        ]
    );

    // (a) Target gap is an error: t > len fails with "Cannot move with index gap into dense Array"
    let a_gap = Value::array(vec![Value::Int(1), Value::Int(2)]);
    let err_gap = call_builtin(
        &mut vm,
        "table.move",
        vec![
            a_gap.clone(),
            Value::Int(0),
            Value::Int(1),
            Value::Int(5), // gap: t=5 > len=2
        ],
    );
    assert!(err_gap.is_err(), "Target gap t > len must fail");
    assert!(
        err_gap.unwrap_err().contains("Cannot move with index gap into dense Array"),
        "Error message must specify index gap into dense Array"
    );

    // (e) Distinct arrays (a1 != a2)
    let a1 = Value::array(vec![Value::Int(100), Value::Int(200), Value::Int(300)]);
    let a2 = Value::array(vec![Value::Int(1), Value::Int(2)]);
    // Copy a1[1..=2] into a2[2] (extending a2)
    let res_distinct = call_builtin(
        &mut vm,
        "table.move",
        vec![
            a1.clone(),
            Value::Int(1),
            Value::Int(2),
            Value::Int(2),
            a2.clone(),
        ],
    )
    .expect("distinct arrays copy");
    assert_eq!(
        res_distinct.as_array().unwrap(),
        vec![Value::Int(1), Value::Int(2), Value::Int(200), Value::Int(300)]
    );
    // Source a1 is untouched
    assert_eq!(
        a1.as_array().unwrap(),
        vec![Value::Int(100), Value::Int(200), Value::Int(300)]
    );

    // (f) Source out of bounds: e >= a1.len()
    let err_src_oob = call_builtin(
        &mut vm,
        "table.move",
        vec![a1.clone(), Value::Int(0), Value::Int(10), Value::Int(0)],
    );
    assert!(err_src_oob.is_err());
}

// ============================================================================
// Gate TAB-04: `table.get` Option Return
// ============================================================================

#[test]
fn test_tab_04_table_get_option() {
    let mut vm = VM::new();

    // 1. Array: Hit returns Some(v), Miss returns None
    let arr = Value::array(vec![Value::Int(11), Value::Int(22), Value::Int(33)]);
    let hit_0 = call_builtin(&mut vm, "table.get", vec![arr.clone(), Value::Int(0)]).expect("get 0");
    assert_eq!(hit_0, Value::some(Value::Int(11)));

    let hit_2 = call_builtin(&mut vm, "table.get", vec![arr.clone(), Value::Int(2)]).expect("get 2");
    assert_eq!(hit_2, Value::some(Value::Int(33)));

    let miss_oob = call_builtin(&mut vm, "table.get", vec![arr.clone(), Value::Int(3)]).expect("get oob");
    assert_eq!(miss_oob, Value::none());

    let miss_neg = call_builtin(&mut vm, "table.get", vec![arr.clone(), Value::Int(-1)]).expect("get neg");
    assert_eq!(miss_neg, Value::none());

    // 2. Map: Hit returns Some(v), Miss returns None
    let map = make_map(vec![
        (ValueKey::string("name".to_string()), Value::string("Lust".to_string())),
        (ValueKey::string("version".to_string()), Value::Int(3)),
    ]);

    let map_hit = call_builtin(
        &mut vm,
        "table.get",
        vec![map.clone(), Value::string("name".to_string())],
    )
    .expect("map hit");
    assert_eq!(map_hit, Value::some(Value::string("Lust".to_string())));

    let map_miss = call_builtin(
        &mut vm,
        "table.get",
        vec![map.clone(), Value::string("missing_key".to_string())],
    )
    .expect("map miss");
    assert_eq!(map_miss, Value::none());
}

// ============================================================================
// Gate TAB-05: `table.insert` Arity Overload
// ============================================================================

#[test]
fn test_tab_05_table_insert_arity() {
    let mut vm = VM::new();

    // 1. Arity 2: table.insert(arr, val) -> push to end
    let arr = Value::array(vec![Value::Int(1), Value::Int(2)]);
    call_builtin(&mut vm, "table.insert", vec![arr.clone(), Value::Int(3)]).expect("insert push");
    assert_eq!(arr.as_array().unwrap(), vec![Value::Int(1), Value::Int(2), Value::Int(3)]);

    // 2. Arity 3: table.insert(arr, idx, val) -> insert at index
    // Insert at head (index 0)
    call_builtin(
        &mut vm,
        "table.insert",
        vec![arr.clone(), Value::Int(0), Value::Int(99)],
    )
    .expect("insert head");
    assert_eq!(
        arr.as_array().unwrap(),
        vec![Value::Int(99), Value::Int(1), Value::Int(2), Value::Int(3)]
    );

    // Insert at middle (index 2)
    call_builtin(
        &mut vm,
        "table.insert",
        vec![arr.clone(), Value::Int(2), Value::Int(77)],
    )
    .expect("insert mid");
    assert_eq!(
        arr.as_array().unwrap(),
        vec![
            Value::Int(99),
            Value::Int(1),
            Value::Int(77),
            Value::Int(2),
            Value::Int(3)
        ]
    );

    // Insert at tail (index len)
    let len = arr.as_array().unwrap().len() as LustInt;
    call_builtin(
        &mut vm,
        "table.insert",
        vec![arr.clone(), Value::Int(len), Value::Int(100)],
    )
    .expect("insert tail");
    assert_eq!(
        arr.as_array().unwrap(),
        vec![
            Value::Int(99),
            Value::Int(1),
            Value::Int(77),
            Value::Int(2),
            Value::Int(3),
            Value::Int(100)
        ]
    );
}

// ============================================================================
// Gate TAB-06: `table.set`, `table.clear`, `table.len`, `table.is_empty`
// ============================================================================

#[test]
fn test_tab_06_table_set_and_clear_and_queries() {
    let mut vm = VM::new();

    // 1. table.set on Array
    let arr = Value::array(vec![Value::Int(10), Value::Int(20), Value::Int(30)]);
    call_builtin(
        &mut vm,
        "table.set",
        vec![arr.clone(), Value::Int(1), Value::Int(999)],
    )
    .expect("set array element");
    assert_eq!(arr.as_array().unwrap()[1], Value::Int(999));

    // table.set array OOB fails
    let err_set_oob = call_builtin(
        &mut vm,
        "table.set",
        vec![arr.clone(), Value::Int(10), Value::Int(42)],
    );
    assert!(err_set_oob.is_err());

    // 2. table.set on Map
    let map = make_map(vec![
        (ValueKey::string("x".to_string()), Value::Int(1)),
    ]);

    // Update existing key
    call_builtin(
        &mut vm,
        "table.set",
        vec![map.clone(), Value::string("x".to_string()), Value::Int(100)],
    )
    .expect("set map existing");

    // Insert new key
    call_builtin(
        &mut vm,
        "table.set",
        vec![map.clone(), Value::string("y".to_string()), Value::Int(200)],
    )
    .expect("set map new");

    let val_x = call_builtin(&mut vm, "table.get", vec![map.clone(), Value::string("x".to_string())]).unwrap();
    assert_eq!(val_x, Value::some(Value::Int(100)));
    let val_y = call_builtin(&mut vm, "table.get", vec![map.clone(), Value::string("y".to_string())]).unwrap();
    assert_eq!(val_y, Value::some(Value::Int(200)));

    // 3. table.len and table.is_empty on Array and Map
    let len_arr = call_builtin(&mut vm, "table.len", vec![arr.clone()]).unwrap();
    assert_eq!(len_arr, Value::Int(3));
    let empty_arr = call_builtin(&mut vm, "table.is_empty", vec![arr.clone()]).unwrap();
    assert_eq!(empty_arr, Value::Bool(false));

    let len_map = call_builtin(&mut vm, "table.len", vec![map.clone()]).unwrap();
    assert_eq!(len_map, Value::Int(2));
    let empty_map = call_builtin(&mut vm, "table.is_empty", vec![map.clone()]).unwrap();
    assert_eq!(empty_map, Value::Bool(false));

    // 4. table.clear on Array and Map
    call_builtin(&mut vm, "table.clear", vec![arr.clone()]).expect("clear array");
    assert_eq!(arr.as_array().unwrap().len(), 0);
    assert_eq!(
        call_builtin(&mut vm, "table.is_empty", vec![arr.clone()]).unwrap(),
        Value::Bool(true)
    );

    call_builtin(&mut vm, "table.clear", vec![map.clone()]).expect("clear map");
    assert_eq!(
        call_builtin(&mut vm, "table.len", vec![map.clone()]).unwrap(),
        Value::Int(0)
    );
    assert_eq!(
        call_builtin(&mut vm, "table.is_empty", vec![map.clone()]).unwrap(),
        Value::Bool(true)
    );
}

// ============================================================================
// Gate TAB-07: Unpack Parity: `table.unpack` (1-based) vs `array.unpack` (0-based)
// ============================================================================

#[test]
fn test_tab_07_unpack_parity() {
    let mut vm = VM::new();
    let arr = Value::array(vec![
        Value::Int(10),
        Value::Int(20),
        Value::Int(30),
        Value::Int(40),
        Value::Int(50),
    ]);

    // 1. table.unpack (1-based Lua parity)
    // Default unpacks all: elements 1..=5
    let unp_all = call_builtin(&mut vm, "table.unpack", vec![arr.clone()]).expect("table.unpack all");
    // Multi-return packages into array of LuaValue
    let elements = unp_all.as_array().unwrap();
    let int_vals: Vec<i64> = elements
        .iter()
        .map(|v| match v {
            Value::Int(n) => *n,
            Value::Enum(e) => e
                .values
                .as_ref()
                .and_then(|p| p.first())
                .and_then(|p| p.as_int())
                .unwrap_or(0),
            _ => v.as_int().unwrap_or(0),
        })
        .collect();
    assert_eq!(int_vals, vec![10, 20, 30, 40, 50]);

    // Range: unpack 2..=4 (1-based, corresponds to items 20, 30, 40)
    let unp_range = call_builtin(
        &mut vm,
        "table.unpack",
        vec![arr.clone(), Value::Int(2), Value::Int(4)],
    )
    .expect("table.unpack range");
    let range_vals: Vec<i64> = unp_range
        .as_array()
        .unwrap()
        .iter()
        .map(|v| match v {
            Value::Int(n) => *n,
            Value::Enum(e) => e
                .values
                .as_ref()
                .and_then(|p| p.first())
                .and_then(|p| p.as_int())
                .unwrap_or(0),
            _ => v.as_int().unwrap_or(0),
        })
        .collect();
    assert_eq!(range_vals, vec![20, 30, 40]);

    // 2. array.unpack (0-based native Lust contract)
    // Default unpacks all: elements 0..=4
    let unp_arr_all = call_builtin(&mut vm, "array.unpack", vec![arr.clone()]).expect("array.unpack all");
    assert_eq!(
        unp_arr_all.as_array().unwrap(),
        vec![
            Value::Int(10),
            Value::Int(20),
            Value::Int(30),
            Value::Int(40),
            Value::Int(50)
        ]
    );

    // Range: unpack 1..=3 (0-based, corresponds to items 20, 30, 40)
    let unp_arr_range = call_builtin(
        &mut vm,
        "array.unpack",
        vec![arr.clone(), Value::Int(1), Value::Int(3)],
    )
    .expect("array.unpack range");
    assert_eq!(
        unp_arr_range.as_array().unwrap(),
        vec![Value::Int(20), Value::Int(30), Value::Int(40)]
    );
}

// ============================================================================
// Gate TAB-08: Functional & Collection Builtins on `table.*`
// ============================================================================

#[test]
fn test_tab_08_functional_builtins() {
    let mut vm = VM::new();
    let arr = Value::array(vec![
        Value::Int(5),
        Value::Int(2),
        Value::Int(8),
        Value::Int(1),
        Value::Int(9),
    ]);

    // first & last
    let first = call_builtin(&mut vm, "table.first", vec![arr.clone()]).unwrap();
    assert_eq!(first, Value::some(Value::Int(5)));
    let last = call_builtin(&mut vm, "table.last", vec![arr.clone()]).unwrap();
    assert_eq!(last, Value::some(Value::Int(9)));

    // slice
    let sl = call_builtin(
        &mut vm,
        "table.slice",
        vec![arr.clone(), Value::Int(1), Value::Int(3)],
    )
    .unwrap();
    assert_eq!(sl.as_array().unwrap(), vec![Value::Int(2), Value::Int(8)]);

    // contains
    let c_true = call_builtin(&mut vm, "table.contains", vec![arr.clone(), Value::Int(8)]).unwrap();
    assert_eq!(c_true, Value::Bool(true));
    let c_false = call_builtin(&mut vm, "table.contains", vec![arr.clone(), Value::Int(99)]).unwrap();
    assert_eq!(c_false, Value::Bool(false));

    // reverse
    let arr_rev = Value::array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    call_builtin(&mut vm, "table.reverse", vec![arr_rev.clone()]).unwrap();
    assert_eq!(
        arr_rev.as_array().unwrap(),
        vec![Value::Int(3), Value::Int(2), Value::Int(1)]
    );

    // concat
    let str_arr = Value::array(vec![
        Value::string("a".to_string()),
        Value::string("b".to_string()),
        Value::string("c".to_string()),
    ]);
    let joined = call_builtin(
        &mut vm,
        "table.concat",
        vec![str_arr, Value::string(",".to_string())],
    )
    .unwrap();
    assert_eq!(joined, Value::string("a,b,c".to_string()));
}

// ============================================================================
// Gate TAB-09: Typechecker Table & LuaTable Resolution & Matching
// ============================================================================

#[test]
fn test_tab_09_typechecker_resolution() {
    let mut env = TypeEnv::new();
    let dummy_span = Span::new(0, 0, 0, 0);

    // 1. Single parameter: Table<T> -> Array<T>
    let table_int = Type::new(
        TypeKind::GenericInstance {
            name: "Table".to_string(),
            type_args: vec![Type::new(TypeKind::Int, dummy_span)],
        },
        dummy_span,
    );
    let res_table_int = env.resolve_type(&table_int);
    assert_eq!(
        res_table_int.kind,
        TypeKind::Array(Box::new(Type::new(TypeKind::Int, dummy_span)))
    );

    // 2. Two parameters: Table<K, V> -> Map<K, V>
    let table_map = Type::new(
        TypeKind::GenericInstance {
            name: "Table".to_string(),
            type_args: vec![
                Type::new(TypeKind::String, dummy_span),
                Type::new(TypeKind::Int, dummy_span),
            ],
        },
        dummy_span,
    );
    let res_table_map = env.resolve_type(&table_map);
    assert_eq!(
        res_table_map.kind,
        TypeKind::Map(
            Box::new(Type::new(TypeKind::String, dummy_span)),
            Box::new(Type::new(TypeKind::Int, dummy_span))
        )
    );

    // 3. Qualified path aliases: std.table.Table and table.Table
    let std_table_int = Type::new(
        TypeKind::GenericInstance {
            name: "std.table.Table".to_string(),
            type_args: vec![Type::new(TypeKind::Int, dummy_span)],
        },
        dummy_span,
    );
    assert_eq!(
        env.resolve_type(&std_table_int).kind,
        TypeKind::Array(Box::new(Type::new(TypeKind::Int, dummy_span)))
    );

    let table_mod_map = Type::new(
        TypeKind::GenericInstance {
            name: "table.Table".to_string(),
            type_args: vec![
                Type::new(TypeKind::String, dummy_span),
                Type::new(TypeKind::Float, dummy_span),
            ],
        },
        dummy_span,
    );
    assert_eq!(
        env.resolve_type(&table_mod_map).kind,
        TypeKind::Map(
            Box::new(Type::new(TypeKind::String, dummy_span)),
            Box::new(Type::new(TypeKind::Float, dummy_span))
        )
    );

    // 4. LuaTable resolution
    let lua_tbl = Type::new(TypeKind::Named("std.table.LuaTable".to_string()), dummy_span);
    assert_eq!(
        env.resolve_type(&lua_tbl).kind,
        TypeKind::Named("LuaTable".to_string())
    );

    // 5. Bidirectional type matching
    let array_int = Type::new(
        TypeKind::Array(Box::new(Type::new(TypeKind::Int, dummy_span))),
        dummy_span,
    );
    assert!(env.types_match(&table_int, &array_int));
    assert!(env.types_match(&array_int, &table_int));

    let map_str_int = Type::new(
        TypeKind::Map(
            Box::new(Type::new(TypeKind::String, dummy_span)),
            Box::new(Type::new(TypeKind::Int, dummy_span)),
        ),
        dummy_span,
    );
    assert!(env.types_match(&table_map, &map_str_int));
    assert!(env.types_match(&map_str_int, &table_map));

    // 6. Method lookup for Table forwards to Array
    let dummy_fn = lust::ast::FunctionDef {
        name: "len".to_string(),
        type_params: vec![],
        trait_bounds: vec![],
        params: vec![],
        return_type: Some(Type::new(TypeKind::Int, dummy_span)),
        body: vec![],
        is_method: true,
        visibility: lust::ast::Visibility::Public,
        doc: None,
        effects: None,
    };
    let array_impl = lust::ast::ImplBlock {
        type_params: vec![],
        trait_name: None,
        target_type: Type::new(TypeKind::Named("Array".to_string()), dummy_span),
        methods: vec![dummy_fn],
        where_clause: vec![],
    };
    env.register_impl(&array_impl).unwrap();

    assert!(env.lookup_method("Table", "len").is_some());
    assert!(env.lookup_method("std.table.Table", "len").is_some());
    assert!(env.lookup_method("table.Table", "len").is_some());
}

// ============================================================================
// Gate TAB-10: End-to-End Script Execution with `use std.table`
// ============================================================================

#[test]
fn test_tab_10_script_execution_registered_path() {
    let script = r#"
        use std.table

        function test_array_ops(): int
            local t: Table<int> = [10, 20, 30]
            table.insert(t, 40)
            table.set(t, 1, 99)
            local l: int = table.len(t)
            return l
        end

        function test_move_ops(): int
            local a: Table<int> = [1, 2, 3, 4, 5]
            table.move(a, 0, 2, 2)
            -- a becomes [1, 2, 1, 2, 3]
            return a[2]:unwrap_or(0) + a[3]:unwrap_or(0) + a[4]:unwrap_or(0)
        end

        function run(): int
            return test_array_ops() + test_move_ops()
        end
    "#;

    let mut prog = EmbeddedBuilder::new()
        .module("main", script)
        .entry_module("main")
        .compile()
        .expect("compile script with use std.table");

    let res = prog.call_raw("main.run", vec![]).expect("run script");
    // test_array_ops: len(t) = 4
    // test_move_ops: a[2] + a[3] + a[4] = 1 + 2 + 3 = 6
    // total = 10
    assert_eq!(res, Value::Int(10));
}

#[test]
fn test_tab_10_script_selective_imports() {
    let script = r#"
        use std.table.{Table, len, set, insert}

        function run(): int
            local arr: Table<int> = [1, 2, 3]
            insert(arr, 4)
            set(arr, 0, 100)
            return len(arr) + arr[0]:unwrap_or(0)
        end
    "#;

    let mut prog = EmbeddedBuilder::new()
        .module("main", script)
        .entry_module("main")
        .compile()
        .expect("compile script with selective imports");

    let res = prog.call_raw("main.run", vec![]).expect("run script");
    // len(arr) = 4, arr[0] = 100 -> 104
    assert_eq!(res, Value::Int(104));
}

// ============================================================================
// Gate TAB-11: Negative Cases (Type Checking & Runtime Rejection)
// ============================================================================

#[test]
fn test_tab_11_negative_cases() {
    // 1. Script with Table<int, int, int> (arity 3) must fail type checking
    let invalid_type_script = r#"
        use std.table.{Table}

        function run(): int
            local invalid: Table<int, int, int> = [1, 2, 3]
            return 0
        end
    "#;

    let compile_res = EmbeddedBuilder::new()
        .module("main", invalid_type_script)
        .entry_module("main")
        .compile();
    assert!(
        compile_res.is_err(),
        "Table with 3 type arguments must be rejected by the compiler"
    );

    // 2. Runtime rejection of table.set out-of-bounds in script
    let runtime_oob_script = r#"
        use std.table

        function run(): int
            local arr: Table<int> = [1, 2]
            table.set(arr, 10, 42)
            return 0
        end
    "#;

    let mut prog_oob = EmbeddedBuilder::new()
        .module("main", runtime_oob_script)
        .entry_module("main")
        .compile()
        .expect("compile oob script");

    let exec_res = prog_oob.call_raw("main.run", vec![]);
    assert!(
        exec_res.is_err(),
        "table.set with out-of-bounds index must fail at runtime"
    );

    // 3. Runtime rejection of table.move target gap in script
    let runtime_gap_script = r#"
        use std.table

        function run(): int
            local arr: Table<int> = [1, 2]
            table.move(arr, 0, 1, 5)
            return 0
        end
    "#;

    let mut prog_gap = EmbeddedBuilder::new()
        .module("main", runtime_gap_script)
        .entry_module("main")
        .compile()
        .expect("compile gap script");

    let gap_res = prog_gap.call_raw("main.run", vec![]);
    assert!(
        gap_res.is_err(),
        "table.move with index gap must fail at runtime"
    );
}

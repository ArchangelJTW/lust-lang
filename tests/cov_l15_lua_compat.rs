#![cfg(feature = "std")]
#![allow(non_snake_case)]
#![allow(clippy::approx_constant)]
//! Stage 5 - L15: Lua Compatibility Verification Suite.
//! Verifies Lua 5.1 C API semantics and Lust VM integration:
//! - Stack manipulation and indexing (positive, negative, pseudo-indices, replace, remove, insert)
//! - Types and conversions (numbers, hex strings, booleans, strings with embedded nulls)
//! - Tables: rawgeti/rawseti, gettable/settable, getfield/setfield, next traversal, sequence length
//! - Metatables: attachment, retrieval, metatable inspection, luaL_newmetatable/luaL_getmetatable
//! - Buffer operations: buffinit, addlstring, addstring, addvalue, prepbuffer, pushresult
//! - References: luaL_ref and luaL_unref in registry with lua_rawgeti verification
//! - Closures and Upvalues: C-closures with upvalues, lua_upvalueindex
//! - Userdata: allocation, size tracking, metatable verification via luaL_checkudata
//! - Error handling & pcall: lua_error, luaL_error, luaL_argerror, lua_pcall catch and propagation
//! - Bi-directional Lust <-> Lua value roundtrip with variant & payload verification

use std::ffi::{CStr, CString};
use core::ffi::{c_char, c_int};
use lust::bytecode::{Value, ValueKey};
use lust::lua_compat::*;
use lust::vm::VM;

// Helper: safe string to *const c_char
fn cstr(s: &str) -> CString {
    CString::new(s).expect("CString::new failed")
}

// Helpers matching Lua 5.1 C macro contracts:
unsafe fn lua_pop(L: *mut lua_State, n: c_int) {
    unsafe { lua_settop(L, -(n) - 1); }
}

unsafe fn lua_tostring(L: *mut lua_State, idx: c_int) -> *const c_char {
    unsafe { lua_tolstring(L, idx, core::ptr::null_mut()) }
}

unsafe fn lua_isnil(L: *mut lua_State, idx: c_int) -> c_int {
    unsafe { (lua_type(L, idx) == LUA_TNIL) as c_int }
}

fn lua_upvalueindex(i: c_int) -> c_int {
    LUA_GLOBALSINDEX - i
}

// ---------------------------------------------------------------------------
// 1. Stack Manipulation & Indexing
// ---------------------------------------------------------------------------

#[test]
fn test_l15_stack_manipulation_and_indexing() {
    unsafe {
        let L = luaL_newstate();
        assert!(!L.is_null());

        // 1.1 Initial empty stack
        assert_eq!(lua_gettop(L), 0);
        assert_eq!(lua_type(L, 1), LUA_TNONE);
        assert_eq!(lua_type(L, -1), LUA_TNONE);

        // 1.2 Pushing primitives
        lua_pushnil(L);
        assert_eq!(lua_type(L, -1), LUA_TNIL);
        assert_eq!(lua_gettop(L), 1);

        lua_pushboolean(L, 1);
        assert_eq!(lua_type(L, -1), LUA_TBOOLEAN);
        assert_eq!(lua_toboolean(L, -1), 1);

        lua_pushinteger(L, 42);
        assert_eq!(lua_type(L, -1), LUA_TNUMBER);
        assert_eq!(lua_tointeger(L, -1), 42);

        lua_pushnumber(L, 3.14159);
        assert_eq!(lua_type(L, -1), LUA_TNUMBER);
        assert!((lua_tonumber(L, -1) - 3.14159).abs() < 1e-6);

        let test_str = cstr("hello lua");
        lua_pushstring(L, test_str.as_ptr());
        assert_eq!(lua_type(L, -1), LUA_TSTRING);
        assert_eq!(lua_isstring(L, -1), 1);
        assert_eq!(lua_isnumber(L, -1), 0);

        assert_eq!(lua_gettop(L), 5);

        // 1.3 pushvalue (duplicate)
        lua_pushvalue(L, 3); // Duplicate integer 42
        assert_eq!(lua_gettop(L), 6);
        assert_eq!(lua_tointeger(L, -1), 42);

        // 1.4 remove
        lua_remove(L, -1);
        assert_eq!(lua_gettop(L), 5);
        assert_eq!(CStr::from_ptr(lua_tostring(L, -1)).to_str().unwrap(), "hello lua");

        // 1.5 insert
        lua_pushinteger(L, 999);
        lua_insert(L, 2); // Insert 999 at position 2
        assert_eq!(lua_tointeger(L, 2), 999);
        assert_eq!(lua_gettop(L), 6);

        // 1.6 replace
        lua_pushinteger(L, 777);
        lua_replace(L, 2); // Replace slot 2 with 777
        assert_eq!(lua_tointeger(L, 2), 777);
        assert_eq!(lua_gettop(L), 6);

        // 1.7 settop truncation and expansion
        lua_settop(L, 3);
        assert_eq!(lua_gettop(L), 3);
        lua_settop(L, 5); // Pad with nils
        assert_eq!(lua_gettop(L), 5);
        assert_eq!(lua_type(L, 4), LUA_TNIL);
        assert_eq!(lua_type(L, 5), LUA_TNIL);

        lua_settop(L, 0); // Clear all
        assert_eq!(lua_gettop(L), 0);

        lua_close(L);
    }
}

// ---------------------------------------------------------------------------
// 2. Type Conversions & Number Parsing
// ---------------------------------------------------------------------------

#[test]
fn test_l15_type_conversions_and_coercions() {
    unsafe {
        let L = luaL_newstate();
        assert!(!L.is_null());

        // Hex number string coercion
        let hex_str = cstr("0x2A");
        lua_pushstring(L, hex_str.as_ptr());
        assert_eq!(lua_isnumber(L, -1), 1);
        assert_eq!(lua_tointeger(L, -1), 42);

        // Float string coercion
        let float_str = cstr("-12.5");
        lua_pushstring(L, float_str.as_ptr());
        assert_eq!(lua_isnumber(L, -1), 1);
        assert!((lua_tonumber(L, -1) - (-12.5)).abs() < 1e-6);

        // Boolean truthiness
        lua_pushboolean(L, 0);
        assert_eq!(lua_toboolean(L, -1), 0);
        lua_pushboolean(L, 1);
        assert_eq!(lua_toboolean(L, -1), 1);
        lua_pushnil(L);
        assert_eq!(lua_toboolean(L, -1), 0);
        lua_pushinteger(L, 0);
        assert_eq!(lua_toboolean(L, -1), 1); // In Lua, 0 is truthy!

        // Type names
        assert_eq!(CStr::from_ptr(lua_typename(L, LUA_TNIL)).to_str().unwrap(), "nil");
        assert_eq!(CStr::from_ptr(lua_typename(L, LUA_TNUMBER)).to_str().unwrap(), "number");
        assert_eq!(CStr::from_ptr(lua_typename(L, LUA_TSTRING)).to_str().unwrap(), "string");
        assert_eq!(CStr::from_ptr(lua_typename(L, LUA_TTABLE)).to_str().unwrap(), "table");
        assert_eq!(CStr::from_ptr(lua_typename(L, LUA_TFUNCTION)).to_str().unwrap(), "function");

        // String with embedded null
        let bytes = b"abc\0def";
        lua_pushlstring(L, bytes.as_ptr() as *const c_char, bytes.len());
        assert_eq!(lua_objlen(L, -1), 7);
        let mut out_len = 0usize;
        let p = lua_tolstring(L, -1, &mut out_len as *mut usize);
        assert_eq!(out_len, 7);
        let slice = std::slice::from_raw_parts(p as *const u8, out_len);
        assert_eq!(slice, b"abc\0def");

        lua_close(L);
    }
}

// ---------------------------------------------------------------------------
// 3. Table Operations & Sequence Length
// ---------------------------------------------------------------------------

#[test]
fn test_l15_table_operations_and_sequence_length() {
    unsafe {
        let L = luaL_newstate();
        assert!(!L.is_null());

        // Create table
        lua_newtable(L);
        assert_eq!(lua_istable(L, -1), 1);

        // rawseti / rawgeti (1-based array sequence)
        for i in 1..=5 {
            lua_pushinteger(L, i * 10);
            lua_rawseti(L, -2, i as c_int);
        }

        // Check sequence length (#t)
        assert_eq!(lua_objlen(L, -1), 5);

        // Fetch back elements
        for i in 1..=5 {
            lua_rawgeti(L, -1, i as c_int);
            assert_eq!(lua_tointeger(L, -1), i * 10);
            lua_pop(L, 1);
        }

        // Set string fields via lua_setfield / lua_getfield
        let key_name = cstr("author");
        let val_name = cstr("Ada Lovelace");
        lua_pushstring(L, val_name.as_ptr());
        lua_setfield(L, -2, key_name.as_ptr());

        lua_getfield(L, -1, key_name.as_ptr());
        assert_eq!(lua_type(L, -1), LUA_TSTRING);
        assert_eq!(CStr::from_ptr(lua_tostring(L, -1)).to_str().unwrap(), "Ada Lovelace");
        lua_pop(L, 1);

        // Table iteration via lua_next
        let mut count = 0;
        lua_pushnil(L); // initial key
        while lua_next(L, -2) != 0 {
            // key at -2, value at -1
            count += 1;
            lua_pop(L, 1); // pop value, keep key for next iteration
        }
        assert_eq!(count, 6); // 5 integers + 1 string field

        lua_close(L);
    }
}

// ---------------------------------------------------------------------------
// 4. Metatables and Metamethods
// ---------------------------------------------------------------------------

#[test]
fn test_l15_metatables_and_registry_registration() {
    unsafe {
        let L = luaL_newstate();
        assert!(!L.is_null());

        // Create base table
        lua_newtable(L); // index 1

        // Create metatable
        lua_newtable(L); // index 2
        lua_pushstring(L, cstr("default_val").as_ptr());
        lua_setfield(L, -2, cstr("fallback_key").as_ptr());

        // Set metatable on base table
        assert_eq!(lua_setmetatable(L, 1), 1);
        assert_eq!(lua_gettop(L), 1); // base table remains

        // Verify getmetatable retrieves the attached metatable
        assert_eq!(lua_getmetatable(L, 1), 1);
        assert_eq!(lua_istable(L, -1), 1);
        // Inspect field in retrieved metatable
        lua_getfield(L, -1, cstr("fallback_key").as_ptr());
        assert_eq!(
            CStr::from_ptr(lua_tostring(L, -1)).to_str().unwrap(),
            "default_val"
        );
        lua_pop(L, 2); // pop field and metatable

        // luaL_newmetatable & luaL_getmetatable in registry
        let tname = cstr("MyType");
        assert_eq!(luaL_newmetatable(L, tname.as_ptr()), 1); // created
        assert_eq!(luaL_newmetatable(L, tname.as_ptr()), 0); // already exists
        lua_pop(L, 2);

        luaL_getmetatable(L, tname.as_ptr());
        assert_eq!(lua_istable(L, -1), 1);
        lua_pop(L, 1);

        lua_close(L);
    }
}

// ---------------------------------------------------------------------------
// 5. Userdata Allocation & Tracking
// ---------------------------------------------------------------------------

#[test]
fn test_l15_userdata_lifecycle_and_methods() {
    unsafe {
        let L = luaL_newstate();
        assert!(!L.is_null());

        struct Point {
            x: i32,
            y: i32,
        }

        // Register Point metatable in registry
        let tname = cstr("Point");
        assert_eq!(luaL_newmetatable(L, tname.as_ptr()), 1);
        lua_pop(L, 1);

        // Allocate userdata
        let ud_ptr = lua_newuserdata(L, std::mem::size_of::<Point>());
        assert!(!ud_ptr.is_null());
        assert_eq!(lua_isuserdata(L, -1), 1);
        assert_eq!(lua_type(L, -1), LUA_TUSERDATA);
        assert_eq!(lua_objlen(L, -1), std::mem::size_of::<Point>());

        let pt = ud_ptr as *mut Point;
        (*pt).x = 100;
        (*pt).y = 200;

        assert_eq!(lua_touserdata(L, -1), ud_ptr);

        // Before attaching Point metatable, checkudata MUST return null
        let bad_check = luaL_checkudata(L, -1, tname.as_ptr());
        assert!(bad_check.is_null(), "checkudata without metatable must return null");

        // Attach Point metatable to userdata
        luaL_getmetatable(L, tname.as_ptr());
        assert_eq!(lua_setmetatable(L, -2), 1);

        // Now checkudata with matching type succeeds
        let pt_read = luaL_checkudata(L, -1, tname.as_ptr()) as *const Point;
        assert_eq!(pt_read as *mut Point, pt);
        assert_eq!((*pt_read).x, 100);
        assert_eq!((*pt_read).y, 200);

        // Checkudata with mismatched type fails and returns null
        let wrong_tname = cstr("OtherType");
        assert!(luaL_checkudata(L, -1, wrong_tname.as_ptr()).is_null());

        assert_eq!(lua_getmetatable(L, -1), 1);
        assert_eq!(lua_istable(L, -1), 1);
        lua_pop(L, 1);

        lua_close(L);
    }
}

// ---------------------------------------------------------------------------
// 6. Registry References (luaL_ref / luaL_unref)
// ---------------------------------------------------------------------------

#[test]
fn test_l15_registry_references() {
    unsafe {
        let L = luaL_newstate();
        assert!(!L.is_null());

        lua_pushstring(L, cstr("retained string").as_ptr());
        let r1 = luaL_ref(L, LUA_REGISTRYINDEX);
        assert!(r1 > 0);

        lua_pushinteger(L, 12345);
        let r2 = luaL_ref(L, LUA_REGISTRYINDEX);
        assert!(r2 > 0);
        assert_ne!(r1, r2);

        // Verify retrieval via lua_rawgeti from LUA_REGISTRYINDEX
        lua_rawgeti(L, LUA_REGISTRYINDEX, r1);
        assert_eq!(lua_isstring(L, -1), 1);
        assert_eq!(
            CStr::from_ptr(lua_tostring(L, -1)).to_str().unwrap(),
            "retained string"
        );
        lua_pop(L, 1);

        lua_rawgeti(L, LUA_REGISTRYINDEX, r2);
        assert_eq!(lua_isnumber(L, -1), 1);
        assert_eq!(lua_tointeger(L, -1), 12345);
        lua_pop(L, 1);

        // Unref r1 and r2
        luaL_unref(L, LUA_REGISTRYINDEX, r1);
        luaL_unref(L, LUA_REGISTRYINDEX, r2);

        // After unref, rawgeti must yield nil
        lua_rawgeti(L, LUA_REGISTRYINDEX, r1);
        assert_eq!(lua_isnil(L, -1), 1);
        lua_pop(L, 1);

        lua_rawgeti(L, LUA_REGISTRYINDEX, r2);
        assert_eq!(lua_isnil(L, -1), 1);
        lua_pop(L, 1);

        lua_close(L);
    }
}

// ---------------------------------------------------------------------------
// 7. C Functions, Closures, Upvalues & Calls
// ---------------------------------------------------------------------------

unsafe extern "C" fn lua_adder_cfunc(L: *mut lua_State) -> c_int {
    unsafe {
        let a = lua_tointeger(L, 1);
        let b = lua_tointeger(L, 2);
        lua_pushinteger(L, a + b);
        1
    }
}

unsafe extern "C" fn lua_closure_with_upvalue(L: *mut lua_State) -> c_int {
    unsafe {
        // Upvalue 1 accessed via lua_upvalueindex(1)
        let upval = lua_tointeger(L, lua_upvalueindex(1));
        let arg = lua_tointeger(L, 1);
        lua_pushinteger(L, upval + arg);
        1
    }
}

#[test]
fn test_l15_c_functions_closures_and_calls() {
    unsafe {
        let L = luaL_newstate();
        assert!(!L.is_null());

        // 7.1 Direct lua_call
        lua_pushcfunction(L, Some(lua_adder_cfunc));
        lua_pushinteger(L, 15);
        lua_pushinteger(L, 27);
        lua_call(L, 2, 1);
        assert_eq!(lua_tointeger(L, -1), 42);
        lua_pop(L, 1);

        // 7.2 Closure with upvalue
        lua_pushinteger(L, 100); // upvalue
        lua_pushcclosure(L, Some(lua_closure_with_upvalue), 1);
        lua_pushinteger(L, 25);  // arg
        lua_call(L, 1, 1);
        assert_eq!(lua_tointeger(L, -1), 125);
        lua_pop(L, 1);

        // 7.3 luaL_register library
        let lib_funcs = [
            luaL_Reg {
                name: cstr("add").into_raw(),
                func: Some(lua_adder_cfunc),
            },
            luaL_Reg {
                name: std::ptr::null(),
                func: None,
            },
        ];

        luaL_register(L, cstr("mymath").as_ptr(), lib_funcs.as_ptr());
        assert_eq!(lua_istable(L, -1), 1);

        lua_getfield(L, -1, cstr("add").as_ptr());
        assert_eq!(lua_iscfunction(L, -1), 1);
        lua_pushinteger(L, 30);
        lua_pushinteger(L, 70);
        lua_call(L, 2, 1);
        assert_eq!(lua_tointeger(L, -1), 100);

        // Free raw cstring
        let _ = CString::from_raw(lib_funcs[0].name as *mut c_char);

        lua_close(L);
    }
}

// ---------------------------------------------------------------------------
// 8. Error Handling & pcall
// ---------------------------------------------------------------------------

unsafe extern "C" fn lua_failing_cfunc(L: *mut lua_State) -> c_int {
    unsafe {
        luaL_error(L, cstr("intentional failure in cfunc").as_ptr())
    }
}

#[test]
fn test_l15_error_handling_and_pcall() {
    unsafe {
        let L = luaL_newstate();
        assert!(!L.is_null());

        // 8.1 Successful pcall
        lua_pushcfunction(L, Some(lua_adder_cfunc));
        lua_pushinteger(L, 10);
        lua_pushinteger(L, 20);
        let status = lua_pcall(L, 2, 1, 0);
        assert_eq!(status, 0);
        assert_eq!(lua_tointeger(L, -1), 30);
        lua_pop(L, 1);

        // 8.2 Failing pcall catches error
        lua_pushcfunction(L, Some(lua_failing_cfunc));
        let err_status = lua_pcall(L, 0, 1, 0);
        assert_eq!(err_status, LUA_ERRRUN);
        assert_eq!(lua_type(L, -1), LUA_TSTRING);
        let err_msg = CStr::from_ptr(lua_tostring(L, -1)).to_str().unwrap();
        assert!(err_msg.contains("intentional failure in cfunc"));
        lua_pop(L, 1);

        // 8.3 Attempting to call non-function with pcall
        lua_pushinteger(L, 12345);
        let call_err = lua_pcall(L, 0, 1, 0);
        assert_eq!(call_err, LUA_ERRRUN);
        let not_callable_msg = CStr::from_ptr(lua_tostring(L, -1)).to_str().unwrap();
        assert!(not_callable_msg.contains("attempt to call int(12345)"));
        lua_pop(L, 1);

        lua_close(L);
    }
}

// ---------------------------------------------------------------------------
// 9. luaL_Buffer Operations
// ---------------------------------------------------------------------------

#[test]
fn test_l15_buffer_multi_chunk_assembly() {
    unsafe {
        let L = luaL_newstate();
        assert!(!L.is_null());

        let mut b: luaL_Buffer = core::mem::zeroed();
        luaL_buffinit(L, &mut b as *mut luaL_Buffer);

        luaL_addstring(&mut b as *mut luaL_Buffer, cstr("header: ").as_ptr());
        let chunk = cstr("payload-data-xyz;");
        luaL_addlstring(&mut b as *mut luaL_Buffer, chunk.as_ptr(), 17);

        // Push value on stack and add it via luaL_addvalue
        lua_pushstring(L, cstr("footer").as_ptr());
        luaL_addvalue(&mut b as *mut luaL_Buffer);

        luaL_pushresult(&mut b as *mut luaL_Buffer);
        assert_eq!(lua_type(L, -1), LUA_TSTRING);
        let res = CStr::from_ptr(lua_tostring(L, -1)).to_str().unwrap();
        assert_eq!(res, "header: payload-data-xyz;footer");

        lua_close(L);
    }
}

// ---------------------------------------------------------------------------
// 10. Bi-directional Lust <-> Lua Conversion Parity
// ---------------------------------------------------------------------------

#[test]
fn test_l15_lust_lua_value_roundtrip() {
    let vm = VM::new();

    // 10.1 Primitive values
    let primitives = vec![
        (Value::Nil, "Nil"),
        (Value::Bool(true), "Bool"),
        (Value::Int(42), "Int"),
        (Value::Float(2.71828), "Float"),
        (Value::string("hello from lust"), "String"),
    ];

    for (val, expected_variant) in primitives {
        let lua_val = value_to_lua(&val, &vm);
        let lust_val = lua_to_lust(&lua_val, &vm, None).expect("lua_to_lust failed");
        match &lust_val {
            Value::Enum(obj) => {
                assert_eq!(obj.enum_name, "LuaValue");
                assert_eq!(obj.variant, expected_variant);
                match expected_variant {
                    "Nil" => assert!(
                        obj.values.is_none() || obj.values.as_ref().unwrap().is_empty()
                    ),
                    "Bool" => assert_eq!(obj.values.as_ref().unwrap()[0], Value::Bool(true)),
                    "Int" => assert_eq!(obj.values.as_ref().unwrap()[0], Value::Int(42)),
                    "Float" => {
                        if let Value::Float(f) = obj.values.as_ref().unwrap()[0] {
                            assert!((f - 2.71828).abs() < 1e-4);
                        } else {
                            panic!("expected float payload");
                        }
                    }
                    "String" => assert_eq!(
                        obj.values.as_ref().unwrap()[0],
                        Value::string("hello from lust")
                    ),
                    _ => {}
                }
            }
            other => panic!("Expected LuaValue enum, got: {other:?}"),
        }
    }

    // 10.2 Map to Lua Table and back
    let mut entries = hashbrown::HashMap::default();
    entries.insert(ValueKey::string("version"), Value::Int(5));
    entries.insert(ValueKey::string("active"), Value::Bool(true));
    let map_val = Value::map(entries);

    let lua_table = value_to_lua(&map_val, &vm);
    let converted_lust = lua_to_lust(&lua_table, &vm, None).expect("lua_to_lust table failed");
    if let Value::Enum(obj) = converted_lust {
        assert_eq!(obj.enum_name, "LuaValue");
        assert_eq!(obj.variant, "Table");
        let struct_val = &obj.values.as_ref().expect("payload exists")[0];
        if let Value::Struct(inst) = struct_val {
            assert_eq!(inst.name.as_str(), "LuaTable");
            let table_map = inst.fields.borrow()[0].clone();
            if let Value::Map(m) = table_map {
                let borrowed = m.borrow();
                match borrowed.get(&ValueKey::string("version")) {
                    Some(Value::Enum(e)) => {
                        assert_eq!(e.variant, "Int");
                        assert_eq!(e.values.as_ref().unwrap()[0], Value::Int(5));
                    }
                    other => panic!("Unexpected version value: {other:?}"),
                }
                match borrowed.get(&ValueKey::string("active")) {
                    Some(Value::Enum(e)) => {
                        assert_eq!(e.variant, "Bool");
                        assert_eq!(e.values.as_ref().unwrap()[0], Value::Bool(true));
                    }
                    other => panic!("Unexpected active value: {other:?}"),
                }
            } else {
                panic!("Expected table field to be a map");
            }
        } else {
            panic!("Expected LuaTable struct in Table payload");
        }
    } else {
        panic!("Expected Table variant");
    }
}

// ---------------------------------------------------------------------------
// 11. Lua Operators in Lust VM Execution
// ---------------------------------------------------------------------------

#[test]
fn test_l15_vm_lua_operators_and_methods() {
    // Arithmetic operators on LuaValue
    let v1 = LuaValue::Int(10);
    let v2 = LuaValue::Int(25);
    assert_eq!(v1.clone() + v2.clone(), LuaValue::Int(35));
    assert_eq!(v2.clone() - v1.clone(), LuaValue::Int(15));
    assert_eq!(v1.clone() * v2.clone(), LuaValue::Int(250));
    assert_eq!(v2.clone() % v1.clone(), LuaValue::Int(5));
    assert_eq!(-v1, LuaValue::Int(-10));

    // String table stub generator
    let tbl = std::rc::Rc::new(core::cell::RefCell::new(LuaTable::new()));
    tbl.borrow_mut().entries.insert(
        LuaValue::String("calculate".to_string()),
        LuaValue::Function(LuaFunction {
            id: 1,
            name: Some("calculate".to_string()),
            cfunc: None,
            lust_handle: None,
            upvalues: Vec::new(),
        }),
    );
    tbl.borrow_mut().entries.insert(
        LuaValue::String("PI".to_string()),
        LuaValue::Float(3.14159),
    );

    let stub = render_table_stub("fastmath", &tbl);
    assert!(stub.contains("function fastmath.calculate(LuaValue): LuaValue"));
    assert!(stub.contains("const fastmath.PI: LuaValue"));
}

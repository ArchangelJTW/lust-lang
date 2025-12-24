use super::task::{TaskInstance, TaskKind, TaskState};
use super::VM;
use crate::bytecode::value::IteratorState;
use crate::bytecode::{NativeCallResult, Value, ValueKey};
use crate::LustError;
use alloc::format;
use alloc::rc::Rc;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;

pub(super) fn install_core_builtins(vm: &mut VM) {
    for (name, value) in core_entries(vm) {
        vm.register_native(name, value);
    }
}

pub(super) fn core_entries(vm: &VM) -> Vec<(&'static str, Value)> {
    vec![
        ("error", create_error_fn()),
        ("assert", create_assert_fn()),
        ("tostring", create_tostring_fn()),
        ("tonumber", create_tonumber_fn()),
        ("setmetatable", create_setmetatable_fn()),
        ("unpack", super::stdlib::create_table_unpack_fn()),
        ("select", super::stdlib::create_select_fn()),
        ("pairs", create_pairs_fn()),
        ("ipairs", create_ipairs_fn()),
        ("task", create_task_module(vm)),
        ("lua", create_lua_module(vm)),
    ]
}

fn lua_truthy(value: &Value) -> bool {
    !matches!(value, Value::Nil | Value::Bool(false))
}

pub(super) fn unwrap_lua_value(value: Value) -> Value {
    if let Value::Enum {
        enum_name,
        variant,
        values,
    } = &value
    {
        if enum_name == "LuaValue" {
            return match variant.as_str() {
                "Nil" => Value::Nil,
                "Bool" => values
                    .as_ref()
                    .and_then(|v| v.get(0))
                    .cloned()
                    .unwrap_or(Value::Bool(false)),
                "Function" => {
                    let handle = values
                        .as_ref()
                        .and_then(|vals| vals.get(0))
                        .and_then(|v| v.struct_get_field("handle"))
                        .and_then(|v| v.as_int())
                        .map(|i| i as usize);
                    if let Some(handle) = handle {
                        if let Some(func) = crate::lua_compat::lookup_lust_function(handle) {
                            return func;
                        }
                        #[cfg(feature = "std")]
                        if std::env::var_os("LUST_LUA_SOCKET_TRACE").is_some() {
                            eprintln!("[lua-socket] unwrap missing handle={}", handle);
                        }
                    }
                    Value::Nil
                }
                "Int" | "Float" | "String" | "Table" | "Userdata" | "LightUserdata" => values
                    .as_ref()
                    .and_then(|v| v.get(0))
                    .cloned()
                    .unwrap_or(Value::Nil),
                _ => value,
            };
        }
    }
    value
}

fn create_error_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        let message = args
            .get(0)
            .cloned()
            .map(unwrap_lua_value)
            .map(|v| format!("{}", v))
            .unwrap_or_else(|| "error".to_string());
        Err(message)
    }))
}

fn create_assert_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        let cond = args
            .get(0)
            .cloned()
            .map(unwrap_lua_value)
            .unwrap_or(Value::Bool(false));
        if lua_truthy(&cond) {
            return Ok(NativeCallResult::Return(cond));
        }
        let message = args
            .get(1)
            .cloned()
            .map(unwrap_lua_value)
            .map(|v| format!("{}", v))
            .unwrap_or_else(|| "assertion failed".to_string());
        Err(message)
    }))
}

fn parse_base_arg(arg: Option<Value>) -> Option<u32> {
    arg.map(unwrap_lua_value).and_then(|val| match val {
        Value::Int(i) => u32::try_from(i).ok(),
        Value::Float(f) => u32::try_from(f as i64).ok(),
        _ => None,
    })
}

fn create_tonumber_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        let value = args
            .get(0)
            .cloned()
            .map(unwrap_lua_value)
            .unwrap_or(Value::Nil);
        let base = parse_base_arg(args.get(1).cloned());
        let result = match value {
            Value::Int(_) | Value::Float(_) => value,
            Value::Bool(b) => Value::Int(if b { 1 } else { 0 }),
            Value::String(s) => {
                if let Some(base) = base {
                    if let Ok(parsed) = i64::from_str_radix(s.as_str(), base) {
                        Value::Int(parsed)
                    } else {
                        Value::Nil
                    }
                } else if let Ok(i) = s.as_str().parse::<i64>() {
                    Value::Int(i)
                } else if let Ok(f) = s.as_str().parse::<f64>() {
                    Value::Float(f)
                } else {
                    Value::Nil
                }
            }
            _ => Value::Nil,
        };
        Ok(NativeCallResult::Return(result))
    }))
}

fn setmetatable_value(table: Value, meta: Value) -> Result<Value, String> {
    if let Some(Value::Map(metamethods_rc)) = table.struct_get_field("metamethods") {
        const META_KEY: &str = "__lust_metatable";
        let mut metamethods = metamethods_rc.borrow_mut();
        metamethods.clear();

        if meta == Value::Nil {
            return Ok(table);
        }

        // Store the full metatable for getmetatable() and copy all metamethod entries (keys starting with "__")
        // into the fast lookup map used by the VM.
        let meta_lua = VM::with_current(|vm| to_lua_value(vm, meta.clone()))?;
        metamethods.insert(ValueKey::string(META_KEY.to_string()), meta_lua);

        // Extract metamethods from the metatable - check both 'table' and 'metamethods' fields
        let mut meta_pairs: Vec<(ValueKey, Value)> = if let Some(map) = meta.as_map() {
            map.into_iter().collect()
        } else if let Some(Value::Map(map)) = meta.struct_get_field("table") {
            map.borrow()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        } else {
            Vec::new()
        };

        // Also extract from the metamethods field (transpiled Lua tables put metamethods here)
        if let Some(Value::Map(meta_metamethods)) = meta.struct_get_field("metamethods") {
            let additional: Vec<(ValueKey, Value)> = meta_metamethods
                .borrow()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            meta_pairs.extend(additional);
        }

        for (key, value) in meta_pairs {
            let key_value = key.to_value();
            let Some(key_str) = key_value.as_string().map(|s| s.to_string()) else {
                continue;
            };
            if !key_str.starts_with("__") {
                continue;
            }
            let value_lua = VM::with_current(|vm| to_lua_value(vm, value))?;
            metamethods.insert(ValueKey::string(key_str), value_lua);
        }
        Ok(table)
    } else {
        Err("setmetatable expects LuaTable values".to_string())
    }
}

fn create_setmetatable_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        if args.len() != 2 {
            return Err("setmetatable(table, meta) requires 2 args".to_string());
        }
        let table = unwrap_lua_value(args[0].clone());
        let meta = unwrap_lua_value(args[1].clone());
        setmetatable_value(table, meta).map(NativeCallResult::Return)
    }))
}

fn collect_map_pairs(value: &Value) -> Vec<(ValueKey, Value)> {
    if let Some(map) = value.as_map() {
        map.into_iter().collect()
    } else if let Some(inner) = value.struct_get_field("table").and_then(|v| v.as_map()) {
        inner.into_iter().collect()
    } else if let Some(arr) = value.as_array() {
        arr.into_iter()
            .enumerate()
            .map(|(i, v)| (ValueKey::from_value(&Value::Int((i as i64) + 1)), v.clone()))
            .collect()
    } else {
        Vec::new()
    }
}

fn create_pairs_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        let target = unwrap_lua_value(args.get(0).cloned().unwrap_or(Value::Nil));
        let items = collect_map_pairs(&target);
        let iter = IteratorState::MapPairs { items, index: 0 };
        Ok(NativeCallResult::Return(Value::Iterator(Rc::new(
            RefCell::new(iter),
        ))))
    }))
}

fn create_ipairs_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        let target = unwrap_lua_value(args.get(0).cloned().unwrap_or(Value::Nil));
        let items = if let Some(arr) = target.as_array() {
            arr.into_iter()
                .enumerate()
                .map(|(i, v)| (ValueKey::from_value(&Value::Int((i as i64) + 1)), v.clone()))
                .collect()
        } else {
            collect_map_pairs(&target)
        };
        let iter = IteratorState::MapPairs { items, index: 0 };
        Ok(NativeCallResult::Return(Value::Iterator(Rc::new(
            RefCell::new(iter),
        ))))
    }))
}

pub(super) fn create_tostring_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        if args.is_empty() {
            return Err("tostring() requires at least one argument".to_string());
        }

        Ok(NativeCallResult::Return(Value::string(format!(
            "{}",
            args[0]
        ))))
    }))
}

pub(super) fn create_task_module(vm: &VM) -> Value {
    let entries = [
        (string_key("run"), create_task_run_fn()),
        (string_key("create"), create_task_create_fn()),
        (string_key("status"), create_task_status_fn()),
        (string_key("info"), create_task_info_fn()),
        (string_key("resume"), create_task_resume_fn()),
        (string_key("yield"), create_task_yield_fn()),
        (string_key("stop"), create_task_stop_fn()),
        (string_key("restart"), create_task_restart_fn()),
        (string_key("current"), create_task_current_fn()),
    ];
    vm.map_with_entries(entries)
}

pub(super) fn string_key(name: &str) -> ValueKey {
    ValueKey::from(name.to_string())
}

fn create_lua_module(vm: &VM) -> Value {
    let entries = [
        (string_key("nil"), Value::enum_unit("LuaValue", "Nil")),
        (
            string_key("require"),
            Value::NativeFunction(Rc::new(|args: &[Value]| {
                if args.len() != 1 {
                    return Err("lua.require(name) requires 1 argument".to_string());
                }
                let name = unwrap_lua_value(args[0].clone());
                let Some(name) = name.as_string() else {
                    return Err("lua.require(name) requires a string module name".to_string());
                };
                VM::with_current(|vm| {
                    if let Some(value) = vm.get_global(name) {
                        Ok(NativeCallResult::Return(value))
                    } else {
                        Ok(NativeCallResult::Return(Value::enum_unit(
                            "LuaValue", "Nil",
                        )))
                    }
                })
            })),
        ),
        (
            string_key("table"),
            Value::NativeFunction(Rc::new(|_| {
                VM::with_current(|vm| {
                    let table = vm.new_map_value();
                    let metamethods = vm.new_map_value();
                    vm.instantiate_struct(
                        "LuaTable",
                        vec![
                            (Rc::new("table".to_string()), table),
                            (Rc::new("metamethods".to_string()), metamethods),
                        ],
                    )
                    .map(NativeCallResult::Return)
                    .map_err(|e| e.to_string())
                })
            })),
        ),
        (
            string_key("to_value"),
            Value::NativeFunction(Rc::new(|args: &[Value]| {
                let value = args.get(0).cloned().unwrap_or(Value::Nil);
                VM::with_current(|vm| {
                    let converted = match value.clone() {
                        Value::Enum { enum_name, .. } if enum_name == "LuaValue" => value,
                        Value::Nil => Value::enum_unit("LuaValue", "Nil"),
                        Value::Bool(b) => {
                            Value::enum_variant("LuaValue", "Bool", vec![Value::Bool(b)])
                        }
                        Value::Int(i) => {
                            Value::enum_variant("LuaValue", "Int", vec![Value::Int(i)])
                        }
                        Value::Float(f) => {
                            Value::enum_variant("LuaValue", "Float", vec![Value::Float(f)])
                        }
                        Value::String(s) => {
                            Value::enum_variant("LuaValue", "String", vec![Value::String(s)])
                        }
                        Value::Struct { name, .. } if name == "LuaTable" => {
                            Value::enum_variant("LuaValue", "Table", vec![value])
                        }
                        Value::Struct { name, .. } if name == "LuaFunction" => {
                            Value::enum_variant("LuaValue", "Function", vec![value])
                        }
                        Value::Struct { name, .. } if name == "LuaUserdata" => {
                            Value::enum_variant("LuaValue", "Userdata", vec![value])
                        }
                        Value::Struct { name, .. } if name == "LuaThread" => {
                            Value::enum_variant("LuaValue", "Thread", vec![value])
                        }
                        Value::Map(map) => {
                            let table = Value::Map(map.clone());
                            let metamethods = vm.new_map_value();
                            let lua_table = vm
                                .instantiate_struct(
                                    "LuaTable",
                                    vec![
                                        (Rc::new("table".to_string()), table),
                                        (Rc::new("metamethods".to_string()), metamethods),
                                    ],
                                )
                                .map_err(|e| e.to_string())?;
                            Value::enum_variant("LuaValue", "Table", vec![lua_table])
                        }
                        Value::Function(_) | Value::Closure { .. } | Value::NativeFunction(_) => {
                            let handle = crate::lua_compat::register_lust_function(value.clone());
                            let lua_fn = vm
                                .instantiate_struct(
                                    "LuaFunction",
                                    vec![(
                                        Rc::new("handle".to_string()),
                                        Value::Int(handle as i64),
                                    )],
                                )
                                .map_err(|e| e.to_string())?;
                            Value::enum_variant("LuaValue", "Function", vec![lua_fn])
                        }
                        other => {
                            // Fallback: wrap opaque values as LightUserdata via integer handle.
                            Value::enum_variant(
                                "LuaValue",
                                "LightUserdata",
                                vec![Value::Int(other.type_of() as i64)],
                            )
                        }
                    };
                    Ok(NativeCallResult::Return(converted))
                })
            })),
        ),
        (
            string_key("unwrap"),
            Value::NativeFunction(Rc::new(|args: &[Value]| {
                let value = args.get(0).cloned().unwrap_or(Value::Nil);
                Ok(NativeCallResult::Return(unwrap_lua_value(value)))
            })),
        ),
        (
            string_key("is_truthy"),
            Value::NativeFunction(Rc::new(|args: &[Value]| {
                let value = args.get(0).cloned().unwrap_or(Value::Nil);
                let unwrapped = unwrap_lua_value(value);
                let is_truthy = lua_truthy(&unwrapped);
                Ok(NativeCallResult::Return(Value::Bool(is_truthy)))
            })),
        ),
        (
            string_key("setmetatable"),
            Value::NativeFunction(Rc::new(|args: &[Value]| {
                if args.len() != 2 {
                    return Err("lua.setmetatable(table, meta) requires 2 args".to_string());
                }
                let table = args[0].clone();
                let meta = args[1].clone();
                VM::with_current(|_vm| {
                    setmetatable_value(table, meta).map(NativeCallResult::Return)
                })
            })),
        ),
        (
            string_key("getmetatable"),
            Value::NativeFunction(Rc::new(|args: &[Value]| {
                if args.len() != 1 {
                    return Err("lua.getmetatable(table) requires 1 arg".to_string());
                }
                let table = args[0].clone();
                VM::with_current(|_vm| {
                    const META_KEY: &str = "__lust_metatable";
                    let table = unwrap_lua_value(table);
                    if let Some(metamethods) = table.struct_get_field("metamethods") {
                        if let Some(map) = metamethods.as_map() {
                            if let Some(meta) =
                                map.get(&ValueKey::string(META_KEY.to_string())).cloned()
                            {
                                // Lua: if metatable has __metatable, return that instead.
                                if let Value::Enum {
                                    enum_name,
                                    variant,
                                    values,
                                } = &meta
                                {
                                    if enum_name == "LuaValue" && variant == "Table" {
                                        if let Some(inner) =
                                            values.as_ref().and_then(|vals| vals.get(0))
                                        {
                                            if let Some(Value::Map(meta_map)) =
                                                inner.struct_get_field("table")
                                            {
                                                if let Some(protect) = meta_map
                                                    .borrow()
                                                    .get(&ValueKey::string(
                                                        "__metatable".to_string(),
                                                    ))
                                                    .cloned()
                                                {
                                                    return Ok(NativeCallResult::Return(protect));
                                                }
                                            }
                                        }
                                    }
                                }
                                return Ok(NativeCallResult::Return(meta));
                            }
                        }
                    }
                    Ok(NativeCallResult::Return(Value::enum_unit(
                        "LuaValue", "Nil",
                    )))
                })
            })),
        ),
        (string_key("socket_protect"), create_lua_socket_protect_fn()),
        (string_key("socket_skip"), create_lua_socket_skip_fn()),
        (string_key("socket_newtry"), create_lua_socket_newtry_fn()),
        (string_key("socket_try"), create_lua_socket_try_fn()),
        // Operator helpers for LuaValue
        (
            string_key("op_neg"),
            Value::NativeFunction(Rc::new(|args: &[Value]| {
                let a = args.get(0).cloned().unwrap_or(Value::Nil);
                Ok(NativeCallResult::Return(lua_op_neg(a)))
            })),
        ),
        (
            string_key("op_add"),
            Value::NativeFunction(Rc::new(|args: &[Value]| {
                let a = args.get(0).cloned().unwrap_or(Value::Nil);
                let b = args.get(1).cloned().unwrap_or(Value::Nil);
                Ok(NativeCallResult::Return(lua_op_binary(a, b, |x, y| x + y)))
            })),
        ),
        (
            string_key("op_sub"),
            Value::NativeFunction(Rc::new(|args: &[Value]| {
                let a = args.get(0).cloned().unwrap_or(Value::Nil);
                let b = args.get(1).cloned().unwrap_or(Value::Nil);
                Ok(NativeCallResult::Return(lua_op_binary(a, b, |x, y| x - y)))
            })),
        ),
        (
            string_key("op_mul"),
            Value::NativeFunction(Rc::new(|args: &[Value]| {
                let a = args.get(0).cloned().unwrap_or(Value::Nil);
                let b = args.get(1).cloned().unwrap_or(Value::Nil);
                Ok(NativeCallResult::Return(lua_op_binary(a, b, |x, y| x * y)))
            })),
        ),
        (
            string_key("op_div"),
            Value::NativeFunction(Rc::new(|args: &[Value]| {
                let a = args.get(0).cloned().unwrap_or(Value::Nil);
                let b = args.get(1).cloned().unwrap_or(Value::Nil);
                Ok(NativeCallResult::Return(lua_op_binary(a, b, |x, y| x / y)))
            })),
        ),
        (
            string_key("op_mod"),
            Value::NativeFunction(Rc::new(|args: &[Value]| {
                let a = args.get(0).cloned().unwrap_or(Value::Nil);
                let b = args.get(1).cloned().unwrap_or(Value::Nil);
                Ok(NativeCallResult::Return(lua_op_binary(a, b, |x, y| x % y)))
            })),
        ),
        (
            string_key("op_concat"),
            Value::NativeFunction(Rc::new(|args: &[Value]| {
                let a = args.get(0).cloned().unwrap_or(Value::Nil);
                let b = args.get(1).cloned().unwrap_or(Value::Nil);
                Ok(NativeCallResult::Return(lua_op_concat(a, b)))
            })),
        ),
        (
            string_key("op_eq"),
            Value::NativeFunction(Rc::new(|args: &[Value]| {
                let a = unwrap_lua_value(args.get(0).cloned().unwrap_or(Value::Nil));
                let b = unwrap_lua_value(args.get(1).cloned().unwrap_or(Value::Nil));
                Ok(NativeCallResult::Return(Value::Bool(a == b)))
            })),
        ),
        (
            string_key("op_ne"),
            Value::NativeFunction(Rc::new(|args: &[Value]| {
                let a = unwrap_lua_value(args.get(0).cloned().unwrap_or(Value::Nil));
                let b = unwrap_lua_value(args.get(1).cloned().unwrap_or(Value::Nil));
                Ok(NativeCallResult::Return(Value::Bool(a != b)))
            })),
        ),
        (
            string_key("op_lt"),
            Value::NativeFunction(Rc::new(|args: &[Value]| {
                let a = unwrap_lua_value(args.get(0).cloned().unwrap_or(Value::Nil));
                let b = unwrap_lua_value(args.get(1).cloned().unwrap_or(Value::Nil));
                let result = match (a, b) {
                    (Value::Int(x), Value::Int(y)) => x < y,
                    (Value::Float(x), Value::Float(y)) => x < y,
                    (Value::Int(x), Value::Float(y)) => (x as f64) < y,
                    (Value::Float(x), Value::Int(y)) => x < (y as f64),
                    (Value::String(x), Value::String(y)) => x < y,
                    _ => false,
                };
                Ok(NativeCallResult::Return(Value::Bool(result)))
            })),
        ),
        (
            string_key("op_le"),
            Value::NativeFunction(Rc::new(|args: &[Value]| {
                let a = unwrap_lua_value(args.get(0).cloned().unwrap_or(Value::Nil));
                let b = unwrap_lua_value(args.get(1).cloned().unwrap_or(Value::Nil));
                let result = match (a, b) {
                    (Value::Int(x), Value::Int(y)) => x <= y,
                    (Value::Float(x), Value::Float(y)) => x <= y,
                    (Value::Int(x), Value::Float(y)) => (x as f64) <= y,
                    (Value::Float(x), Value::Int(y)) => x <= (y as f64),
                    (Value::String(x), Value::String(y)) => x <= y,
                    _ => false,
                };
                Ok(NativeCallResult::Return(Value::Bool(result)))
            })),
        ),
        (
            string_key("op_gt"),
            Value::NativeFunction(Rc::new(|args: &[Value]| {
                let a = unwrap_lua_value(args.get(0).cloned().unwrap_or(Value::Nil));
                let b = unwrap_lua_value(args.get(1).cloned().unwrap_or(Value::Nil));
                let result = match (a, b) {
                    (Value::Int(x), Value::Int(y)) => x > y,
                    (Value::Float(x), Value::Float(y)) => x > y,
                    (Value::Int(x), Value::Float(y)) => (x as f64) > y,
                    (Value::Float(x), Value::Int(y)) => x > (y as f64),
                    (Value::String(x), Value::String(y)) => x > y,
                    _ => false,
                };
                Ok(NativeCallResult::Return(Value::Bool(result)))
            })),
        ),
        (
            string_key("op_ge"),
            Value::NativeFunction(Rc::new(|args: &[Value]| {
                let a = unwrap_lua_value(args.get(0).cloned().unwrap_or(Value::Nil));
                let b = unwrap_lua_value(args.get(1).cloned().unwrap_or(Value::Nil));
                let result = match (a, b) {
                    (Value::Int(x), Value::Int(y)) => x >= y,
                    (Value::Float(x), Value::Float(y)) => x >= y,
                    (Value::Int(x), Value::Float(y)) => (x as f64) >= y,
                    (Value::Float(x), Value::Int(y)) => x >= (y as f64),
                    (Value::String(x), Value::String(y)) => x >= y,
                    _ => false,
                };
                Ok(NativeCallResult::Return(Value::Bool(result)))
            })),
        ),
        (
            string_key("call_method"),
            Value::NativeFunction(Rc::new(|args: &[Value]| {
                let obj = args.get(0).cloned().unwrap_or(Value::Nil);
                let method_name_val = args.get(1).cloned().unwrap_or(Value::Nil);
                let method_args: Vec<Value> = args.iter().skip(2).cloned().collect();

                VM::with_current(|vm| {
                    // Get method name as string
                    let method_name = match unwrap_lua_value(method_name_val) {
                        Value::String(s) => s.to_string(),
                        _ => return Err("call_method expects string method name".to_string()),
                    };

                    // Unwrap LuaValue if needed
                    let unwrapped_obj = unwrap_lua_value(obj.clone());

                    // #[cfg(feature = "std")]
                    // eprintln!("[call_method] obj type: {:?}, unwrapped type: {:?}, method: {}",
                    //     obj.type_of(), unwrapped_obj.type_of(), method_name);

                    // Try to find method on object
                    let method = if let Value::Struct { name, .. } = &unwrapped_obj {
                        // #[cfg(feature = "std")]
                        // eprintln!("[call_method] Struct name: {}", name);

                        if name == "LuaUserdata" {
                            // For userdata, look up metatable from Lua state registry
                            // Get userdata ID and state pointer, then look up the method
                            let id_opt = unwrapped_obj
                                .struct_get_field("handle")
                                .and_then(|id_val| id_val.as_int());
                            let state_ptr_opt = unwrapped_obj
                                .struct_get_field("state")
                                .and_then(|state_val| state_val.as_int());

                            match (id_opt, state_ptr_opt) {
                                (Some(id), Some(state_ptr)) if state_ptr != 0 => {
                                    // #[cfg(feature = "std")]
                                    // eprintln!("[call_method] Looking for userdata #{} method in state {:#x}",
                                    //     id, state_ptr);

                                    // SAFETY: The state pointer was stored when the userdata was created
                                    // and should remain valid as long as the lua_State hasn't been closed.
                                    // For C libraries loaded at startup, the state lives for the program duration.
                                    let found_method = unsafe {
                                        let state = state_ptr as *mut crate::lua_compat::lua_State;
                                        if state.is_null() {
                                            None
                                        } else {
                                            let lua_state = &(*state).state;

                                            // Look up metatable from userdata_metatables
                                            lua_state.userdata_metatables.get(&(id as usize))
                                                .and_then(|metatable| {
                                                    // Check if metatable has an __index table
                                                    let metatable_borrow = metatable.borrow();
                                                    let index_table = metatable_borrow.entries.get(
                                                        &crate::lua_compat::LuaValue::String("__index".to_string())
                                                    );

                                                    if let Some(crate::lua_compat::LuaValue::Table(index_tbl)) = index_table {
                                                        // Look up method in __index table
                                                        let method_key = crate::lua_compat::LuaValue::String(method_name.clone());
                                                        let index_borrow = index_tbl.borrow();
                                                        let method_val = index_borrow.entries.get(&method_key)?;

                                                        // #[cfg(feature = "std")]
                                                        // eprintln!("[call_method] Found method {} in __index", method_name);

                                                        // For C functions from metatables, we need to register them
                                                        // directly since they don't have lust_handle set
                                                        if let crate::lua_compat::LuaValue::Function(func) = method_val {
                                                            if let Some(cfunc) = func.cfunc {
                                                                // Register the C function by creating a NativeFunction wrapper
                                                                let state_ptr_copy = state;
                                                                let _cfunc_name = func.name.clone();
                                                                let cfunc_upvalues = func.upvalues.clone();

                                                                let native = Value::NativeFunction(alloc::rc::Rc::new(move |args: &[Value]| {
                                                                    use crate::vm::VM;
                                                                    VM::with_current(|vm| {
                                                                        // unsafe {
                                                                            if state_ptr_copy.is_null() {
                                                                                return Err("null lua_State pointer".to_string());
                                                                            }

                                                                            let lua_state = &mut (*state_ptr_copy).state;

                                                                            // Save current state
                                                                            let saved_stack = core::mem::take(&mut lua_state.stack);
                                                                            let saved_upvalues = core::mem::replace(
                                                                                &mut lua_state.current_upvalues,
                                                                                cfunc_upvalues.clone()
                                                                            );

                                                                            // Push arguments onto Lua stack
                                                                            for arg in args {
                                                                                lua_state.push(crate::lua_compat::value_to_lua(arg, vm));
                                                                            }

                                                                            // Call the C function
                                                                            let result_count = (cfunc)(state_ptr_copy);

                                                                            // Collect results
                                                                            let mut results = Vec::new();
                                                                            if result_count > 0 {
                                                                                for _ in 0..result_count {
                                                                                    if let Some(val) = lua_state.pop() {
                                                                                        results.push(val);
                                                                                    }
                                                                                }
                                                                                results.reverse();
                                                                            }

                                                                            // Restore state
                                                                            lua_state.stack = saved_stack;
                                                                            lua_state.current_upvalues = saved_upvalues;

                                                                            // Convert results to Lust values
                                                                            // #[cfg(feature = "std")]
                                                                            // eprintln!("[C function] Returned {} values from C", results.len());
                                                                            // #[cfg(feature = "std")]
                                                                            // for (i, r) in results.iter().enumerate() {
                                                                            //     eprintln!("[C function] Result[{}]: {:?}", i, r);
                                                                            // }

                                                                            let lust_results: Result<Vec<Value>, String> = results.iter()
                                                                                .map(|v| crate::lua_compat::lua_to_lust(v, vm, None))
                                                                                .collect();

                                                                            match lust_results {
                                                                                Ok(vals) if vals.len() == 1 => Ok(crate::bytecode::value::NativeCallResult::Return(vals[0].clone())),
                                                                                Ok(vals) if vals.is_empty() => Ok(crate::bytecode::value::NativeCallResult::Return(Value::Nil)),
                                                                                Ok(vals) => {
                                                                                    let arr = alloc::rc::Rc::new(core::cell::RefCell::new(vals));
                                                                                    Ok(crate::bytecode::value::NativeCallResult::Return(Value::Array(arr)))
                                                                                }
                                                                                Err(e) => Err(e),
                                                                            }
                                                                        // }
                                                                    })
                                                                }));

                                                                // Register and return the native function
                                                                let _handle = crate::lua_compat::register_lust_function(native.clone());
                                                                return Some(native);
                                                            }
                                                        }

                                                        // Fallback to normal conversion for non-C functions
                                                        crate::lua_compat::lua_to_lust(method_val, vm, None).ok()
                                                    } else {
                                                        None
                                                    }
                                                })
                                        }
                                    };

                                    if found_method.is_some() {
                                        found_method
                                    } else {
                                        // #[cfg(feature = "std")]
                                        // eprintln!("[call_method] Method {} not found in userdata metatable", method_name);
                                        None
                                    }
                                }
                                _ => {
                                    // #[cfg(feature = "std")]
                                    // eprintln!("[call_method] Invalid userdata: id={:?}, state={:?}", id_opt, state_ptr_opt);
                                    None
                                }
                            }
                        } else if name == "LuaTable" {
                            // For LuaTable, check table field first, then metatable.__index
                            // #[cfg(feature = "std")]
                            // eprintln!("[call_method] LuaTable lookup for method: {}", method_name);

                            // First try direct field access on the table itself
                            unwrapped_obj.struct_get_field("table")
                                .and_then(|table_map| {
                                    // #[cfg(feature = "std")]
                                    // eprintln!("[call_method] Checking table map directly for {}", method_name);
                                    if let Value::Map(m) = table_map {
                                        let result = m.borrow().get(&ValueKey::from(Value::string(method_name.clone()))).cloned();
                                        // #[cfg(feature = "std")]
                                        // eprintln!("[call_method] Direct table lookup result: {:?}", result.is_some());
                                        result
                                    } else {
                                        None
                                    }
                                })
                                .or_else(|| {
                                    // Check table["__index"][method_name]
                                    // #[cfg(feature = "std")]
                                    // eprintln!("[call_method] Checking table[__index] for {}", method_name);

                                    let index_val = unwrapped_obj.struct_get_field("table")
                                        .and_then(|table_map| {
                                            if let Value::Map(m) = table_map {
                                                m.borrow().get(&ValueKey::from(Value::string("__index"))).cloned()
                                            } else {
                                                None
                                            }
                                        });

                                    // #[cfg(feature = "std")]
                                    // if let Some(ref idx) = index_val {
                                    //     eprintln!("[call_method] Found table[__index], type: {:?}", idx.type_of());
                                    // } else {
                                    //     eprintln!("[call_method] No table[__index] found");
                                    // }

                                    index_val.and_then(|index_val| {
                                            // index_val could be a LuaValue.Table enum or a Map
                                            match &index_val {
                                                Value::Map(m) => {
                                                    // #[cfg(feature = "std")]
                                                    // eprintln!("[call_method] table[__index] is a Map, looking for {}", method_name);
                                                    let result = m.borrow().get(&ValueKey::from(Value::string(method_name.clone()))).cloned();
                                                    // #[cfg(feature = "std")]
                                                    // eprintln!("[call_method] Map lookup result: {:?}", result.is_some());
                                                    result
                                                }
                                                Value::Enum { variant, values, .. } if variant == "Table" => {
                                                    // #[cfg(feature = "std")]
                                                    // eprintln!("[call_method] table[__index] is a LuaValue.Table enum");
                                                    values.as_ref()
                                                        .and_then(|vals| vals.get(0).cloned())
                                                        .and_then(|table_struct| {
                                                            table_struct.struct_get_field("table")
                                                                .and_then(|table_map| {
                                                                    if let Value::Map(m) = table_map {
                                                                        m.borrow().get(&ValueKey::from(Value::string(method_name.clone()))).cloned()
                                                                    } else {
                                                                        None
                                                                    }
                                                                })
                                                        })
                                                }
                                                _ => None
                                            }
                                        })
                                })
                                .or_else(|| {
                                    // Fallback: check metatable's __index
                                    // #[cfg(feature = "std")]
                                    // eprintln!("[call_method] Checking LuaTable metamethods");

                                    unwrapped_obj.struct_get_field("metamethods")
                                        .and_then(|metamethods| {
                                            // #[cfg(feature = "std")]
                                            // eprintln!("[call_method] metamethods type: {:?}", metamethods.type_of());

                                            if let Value::Map(map) = metamethods {
                                                let index_val = map.borrow().get(&ValueKey::from(Value::string("__index"))).cloned();

                                                // #[cfg(feature = "std")]
                                                // if let Some(ref idx) = index_val {
                                                //     eprintln!("[call_method] Found __index, type: {:?}", idx.type_of());
                                                // } else {
                                                //     eprintln!("[call_method] No __index in metamethods");
                                                // }

                                                index_val.and_then(|index_table| {
                                                    // #[cfg(feature = "std")]
                                                    // eprintln!("[call_method] __index value: {:?}", index_table);

                                                    // __index could be a table or an enum wrapping a table
                                                    match &index_table {
                                                        Value::Struct { name, .. } if name == "LuaTable" => {
                                                            index_table.struct_get_field(&method_name)
                                                        }
                                                        Value::Enum { variant, values, .. } if variant == "Table" => {
                                                            // Unwrap LuaValue.Table enum
                                                            values.as_ref()
                                                                .and_then(|vals| vals.get(0).cloned())
                                                                .and_then(|table_struct| {
                                                                    table_struct.struct_get_field("table")
                                                                        .and_then(|table_map| {
                                                                            if let Value::Map(m) = table_map {
                                                                                m.borrow().get(&ValueKey::from(Value::string(method_name.clone()))).cloned()
                                                                            } else {
                                                                                None
                                                                            }
                                                                        })
                                                                })
                                                        }
                                                        _ => None
                                                    }
                                                })
                                            } else {
                                                None
                                            }
                                        })
                                })
                        } else {
                            unwrapped_obj.struct_get_field(&method_name)
                        }
                    } else {
                        None
                    };

                    let method = method.ok_or_else(|| {
                        format!("Method '{}' not found on {:?}", method_name, obj.type_of())
                    })?;

                    // Call the method with obj as first argument
                    let mut call_args = vec![obj];
                    call_args.extend(method_args);

                    vm.call_value(&method, call_args)
                        .map(|v| NativeCallResult::Return(v))
                        .map_err(|e| e.to_string())
                })
            })),
        ),
        (
            string_key("table_from_entries"),
            Value::NativeFunction(Rc::new(|args: &[Value]| {
                let entries_array = args.get(0).cloned().unwrap_or(Value::Nil);

                VM::with_current(|vm| {
                    // Create new table map
                    let table_map = vm.new_map_value();

                    // Extract array of tuples and populate the table
                    if let Value::Array(entries) = &entries_array {
                        for entry in entries.borrow().iter() {
                            // Each entry should be a 2-element tuple
                            if let Value::Tuple(fields) = entry {
                                if fields.len() >= 2 {
                                    let key = fields[0].clone();
                                    let value = fields[1].clone();

                                    // Key stays as-is (for map key), value gets wrapped in LuaValue
                                    let lua_value = to_lua_value(vm, value)?;

                                    // Insert into the map using the unwrapped key
                                    if let Value::Map(map) = &table_map {
                                        use crate::bytecode::ValueKey;
                                        map.borrow_mut().insert(ValueKey::from(key), lua_value);
                                    }
                                }
                            }
                        }
                    }

                    // Create empty metamethods map
                    let metamethods = vm.new_map_value();

                    // Return LuaTable struct
                    let lua_table = vm
                        .instantiate_struct(
                            "LuaTable",
                            vec![
                                (Rc::new("table".to_string()), table_map),
                                (Rc::new("metamethods".to_string()), metamethods),
                            ],
                        )
                        .map_err(|e| e.to_string())?;

                    Ok(NativeCallResult::Return(lua_table))
                })
            })),
        ),
    ];
    vm.map_with_entries(entries)
}

fn pack_lua_values(vm: &mut VM, values: Vec<Value>) -> Result<Value, String> {
    let mut packed = Vec::with_capacity(values.len());
    for value in values {
        packed.push(to_lua_value(vm, value)?);
    }
    Ok(Value::array(packed))
}

fn to_lua_value(vm: &mut VM, value: Value) -> Result<Value, String> {
    Ok(match value.clone() {
        Value::Enum { enum_name, .. } if enum_name == "LuaValue" => value,
        Value::Nil => Value::enum_unit("LuaValue", "Nil"),
        Value::Bool(b) => Value::enum_variant("LuaValue", "Bool", vec![Value::Bool(b)]),
        Value::Int(i) => Value::enum_variant("LuaValue", "Int", vec![Value::Int(i)]),
        Value::Float(f) => Value::enum_variant("LuaValue", "Float", vec![Value::Float(f)]),
        Value::String(s) => Value::enum_variant("LuaValue", "String", vec![Value::String(s)]),
        Value::Struct { name, .. } if name == "LuaTable" => {
            Value::enum_variant("LuaValue", "Table", vec![value])
        }
        Value::Struct { name, .. } if name == "LuaFunction" => {
            Value::enum_variant("LuaValue", "Function", vec![value])
        }
        Value::Struct { name, .. } if name == "LuaUserdata" => {
            Value::enum_variant("LuaValue", "Userdata", vec![value])
        }
        Value::Struct { name, .. } if name == "LuaThread" => {
            Value::enum_variant("LuaValue", "Thread", vec![value])
        }
        Value::Map(map) => {
            let table = Value::Map(map.clone());
            let metamethods = vm.new_map_value();
            let lua_table = vm
                .instantiate_struct(
                    "LuaTable",
                    vec![
                        (Rc::new("table".to_string()), table),
                        (Rc::new("metamethods".to_string()), metamethods),
                    ],
                )
                .map_err(|e| e.to_string())?;
            Value::enum_variant("LuaValue", "Table", vec![lua_table])
        }
        Value::Function(_) | Value::Closure { .. } | Value::NativeFunction(_) => {
            let handle = crate::lua_compat::register_lust_function(value.clone());
            let lua_fn = vm
                .instantiate_struct(
                    "LuaFunction",
                    vec![(Rc::new("handle".to_string()), Value::Int(handle as i64))],
                )
                .map_err(|e| e.to_string())?;
            Value::enum_variant("LuaValue", "Function", vec![lua_fn])
        }
        other => Value::enum_variant(
            "LuaValue",
            "LightUserdata",
            vec![Value::Int(other.type_of() as i64)],
        ),
    })
}

fn lua_value_error_message(err: LustError) -> String {
    match err {
        LustError::RuntimeError { message } => message,
        other => other.to_string(),
    }
}

fn resolve_callable_for_pcall(value: Value) -> Result<Value, String> {
    if let Value::Struct { name, .. } = &value {
        if name == "LuaFunction" {
            let handle = value
                .struct_get_field("handle")
                .and_then(|v| v.as_int())
                .map(|v| v as usize)
                .ok_or_else(|| "LuaFunction missing handle".to_string())?;
            return crate::lua_compat::lookup_lust_function(handle).ok_or_else(|| {
                format!("LuaFunction handle {} was not registered with VM", handle)
            });
        }
    }
    Ok(value)
}

fn create_lua_socket_protect_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        let func = args.get(0).cloned().unwrap_or(Value::Nil);
        #[cfg(feature = "std")]
        if std::env::var_os("LUST_LUA_SOCKET_TRACE").is_some() {
            eprintln!("[lua-socket] protect setup arg0={:?}", func.type_of());
        }
        VM::with_current(move |_vm| {
            let func = resolve_callable_for_pcall(func)?;
            let wrapper = Value::NativeFunction(Rc::new(move |call_args: &[Value]| {
                let func = func.clone();
                VM::with_current(move |vm| match vm.call_value(&func, call_args.to_vec()) {
                    Ok(value) => {
                        #[cfg(feature = "std")]
                        if std::env::var_os("LUST_LUA_SOCKET_TRACE").is_some() {
                            eprintln!("[lua-socket] protect ok -> {:?}", value.type_of());
                        }
                        let packed = match value {
                            Value::Array(_) => value,
                            other => pack_lua_values(vm, vec![other])?,
                        };
                        Ok(NativeCallResult::Return(packed))
                    }
                    Err(err) => {
                        let msg = lua_value_error_message(err);
                        #[cfg(feature = "std")]
                        if std::env::var_os("LUST_LUA_SOCKET_TRACE").is_some() {
                            eprintln!("[lua-socket] protect err -> {}", msg);
                        }
                        let packed = pack_lua_values(vm, vec![Value::Nil, Value::string(msg)])?;
                        #[cfg(feature = "std")]
                        if std::env::var_os("LUST_LUA_SOCKET_TRACE").is_some() {
                            eprintln!("[lua-socket] protect err ret = {}", packed);
                        }
                        Ok(NativeCallResult::Return(packed))
                    }
                })
            }));
            Ok(NativeCallResult::Return(wrapper))
        })
    }))
}

fn create_lua_socket_skip_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        let count = unwrap_lua_value(args.get(0).cloned().unwrap_or(Value::Nil));
        let skip = count
            .as_int()
            .or_else(|| count.as_float().map(|f| f as i64))
            .unwrap_or(0)
            .max(0) as usize;
        let values: Vec<Value> = if args.len() == 2 {
            if let Some(arr) = args[1].as_array() {
                arr
            } else {
                vec![args[1].clone()]
            }
        } else if args.len() > 2 {
            args[1..].to_vec()
        } else {
            Vec::new()
        };
        let remaining = if skip >= values.len() {
            Vec::new()
        } else {
            values[skip..].to_vec()
        };
        VM::with_current(|vm| {
            let packed = pack_lua_values(vm, remaining)?;
            Ok(NativeCallResult::Return(packed))
        })
    }))
}

fn create_lua_socket_newtry_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        let finalizer = args.get(0).cloned();

        // Create a "try" function that captures the finalizer
        let try_fn = Value::NativeFunction(Rc::new(move |try_args: &[Value]| {
            let first = unwrap_lua_value(try_args.get(0).cloned().unwrap_or(Value::Nil));

            // If first argument is nil or false, call finalizer and throw error
            if !lua_truthy(&first) {
                // Call the finalizer if it exists
                if let Some(ref fin) = finalizer {
                    let _ = VM::with_current(|vm| {
                        let _ = vm.call_value(fin, Vec::new());
                        Ok::<(), String>(())
                    });
                }

                // Throw the error (last argument, or second argument if there are only 2)
                let err_msg = if try_args.len() > 1 {
                    let err_val = unwrap_lua_value(try_args[try_args.len() - 1].clone());
                    match err_val {
                        Value::String(s) => s.to_string(),
                        other => alloc::format!("{:?}", other),
                    }
                } else {
                    "operation failed".to_string()
                };
                return Err(err_msg);
            }

            // Return all arguments
            if try_args.len() == 1 {
                Ok(NativeCallResult::Return(try_args[0].clone()))
            } else {
                let arr = Rc::new(RefCell::new(try_args.to_vec()));
                Ok(NativeCallResult::Return(Value::Array(arr)))
            }
        }));

        Ok(NativeCallResult::Return(try_fn))
    }))
}

fn create_lua_socket_try_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        let first = unwrap_lua_value(args.get(0).cloned().unwrap_or(Value::Nil));

        // If first argument is nil or false, throw error
        if !lua_truthy(&first) {
            let err_msg = if args.len() > 1 {
                let err_val = unwrap_lua_value(args[1].clone());
                match err_val {
                    Value::String(s) => s.to_string(),
                    other => alloc::format!("{:?}", other),
                }
            } else {
                "operation failed".to_string()
            };
            return Err(err_msg);
        }

        // Return all arguments
        if args.len() == 1 {
            Ok(NativeCallResult::Return(first))
        } else {
            let arr = Rc::new(RefCell::new(args.to_vec()));
            Ok(NativeCallResult::Return(Value::Array(arr)))
        }
    }))
}

fn create_task_run_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        if args.is_empty() {
            return Err("task.run() requires a function".to_string());
        }

        let func = args[0].clone();
        let rest: Vec<Value> = args.iter().skip(1).cloned().collect();
        VM::with_current(move |vm| {
            vm.spawn_task_value(func, rest)
                .map(|handle| NativeCallResult::Return(Value::task(handle)))
                .map_err(|e| e.to_string())
        })
    }))
}

fn create_task_create_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        if args.is_empty() {
            return Err("task.create() requires a function".to_string());
        }

        let func = args[0].clone();
        let rest: Vec<Value> = args.iter().skip(1).cloned().collect();
        VM::with_current(move |vm| {
            vm.create_task_value(func, rest)
                .map(|handle| NativeCallResult::Return(Value::task(handle)))
                .map_err(|e| e.to_string())
        })
    }))
}

fn create_task_status_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        if args.len() != 1 {
            return Err("task.status() requires a task handle".to_string());
        }

        let handle = args[0]
            .as_task_handle()
            .ok_or_else(|| "task.status() requires a task handle value".to_string())?;
        VM::with_current(move |vm| {
            let task = vm.get_task_instance(handle).map_err(|e| e.to_string())?;
            Ok(NativeCallResult::Return(task_state_to_status_value(
                &task.state,
            )))
        })
    }))
}

fn create_task_info_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        if args.len() != 1 {
            return Err("task.info() requires a task handle".to_string());
        }

        let handle = args[0]
            .as_task_handle()
            .ok_or_else(|| "task.info() requires a task handle value".to_string())?;
        VM::with_current(move |vm| {
            let task = vm.get_task_instance(handle).map_err(|e| e.to_string())?;
            let info = build_task_info_value(vm, task).map_err(|e| e.to_string())?;
            Ok(NativeCallResult::Return(info))
        })
    }))
}

fn create_task_resume_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        if args.is_empty() {
            return Err("task.resume() requires a task handle".to_string());
        }

        let handle = args[0]
            .as_task_handle()
            .ok_or_else(|| "task.resume() requires a task handle value".to_string())?;
        let resume_value = args.get(1).cloned();
        VM::with_current(move |vm| {
            let kind = vm
                .get_task_instance(handle)
                .map(|task| task.kind().clone())
                .map_err(|e| e.to_string())?;

            if matches!(kind, TaskKind::NativeFuture { .. }) {
                if resume_value.is_some() {
                    return Err(
                        "task.resume() does not accept a resume value for native async tasks"
                            .to_string(),
                    );
                }

                let task = vm.get_task_instance(handle).map_err(|e| e.to_string())?;
                let info = build_task_info_value(vm, task).map_err(|e| e.to_string())?;
                return Ok(NativeCallResult::Return(info));
            }

            vm.resume_task_handle(handle, resume_value.clone())
                .map_err(|e| e.to_string())?;
            let task = vm.get_task_instance(handle).map_err(|e| e.to_string())?;
            let info = build_task_info_value(vm, task).map_err(|e| e.to_string())?;
            Ok(NativeCallResult::Return(info))
        })
    }))
}

fn create_task_yield_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        let value = args.get(0).cloned().unwrap_or(Value::Nil);
        Ok(NativeCallResult::Yield(value))
    }))
}

fn create_task_stop_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        if args.len() != 1 {
            return Err("task.stop() requires a task handle".to_string());
        }

        let handle = args[0]
            .as_task_handle()
            .ok_or_else(|| "task.stop() requires a task handle value".to_string())?;
        VM::with_current(move |vm| {
            let before = vm
                .get_task_instance(handle)
                .map(|task| task.state.clone())
                .map_err(|e| e.to_string())?;
            vm.stop_task_handle(handle).map_err(|e| e.to_string())?;
            let after = vm
                .get_task_instance(handle)
                .map(|task| task.state.clone())
                .map_err(|e| e.to_string())?;
            let changed = after == TaskState::Stopped && before != TaskState::Stopped;
            Ok(NativeCallResult::Return(Value::Bool(changed)))
        })
    }))
}

fn create_task_restart_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        if args.len() != 1 {
            return Err("task.restart() requires a task handle".to_string());
        }

        let handle = args[0]
            .as_task_handle()
            .ok_or_else(|| "task.restart() requires a task handle value".to_string())?;
        VM::with_current(move |vm| {
            vm.restart_task_handle(handle).map_err(|e| e.to_string())?;
            let task = vm.get_task_instance(handle).map_err(|e| e.to_string())?;
            let info = build_task_info_value(vm, task).map_err(|e| e.to_string())?;
            Ok(NativeCallResult::Return(info))
        })
    }))
}

fn create_task_current_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        if !args.is_empty() {
            return Err("task.current() takes no arguments".to_string());
        }

        VM::with_current(|vm| {
            let value = match vm.current_task_handle() {
                Some(handle) => Value::some(Value::task(handle)),
                None => Value::none(),
            };
            Ok(NativeCallResult::Return(value))
        })
    }))
}

fn task_state_to_status_value(state: &TaskState) -> Value {
    match state {
        TaskState::Ready => Value::enum_unit("TaskStatus", "Ready"),
        TaskState::Running => Value::enum_unit("TaskStatus", "Running"),
        TaskState::Yielded => Value::enum_unit("TaskStatus", "Yielded"),
        TaskState::Completed => Value::enum_unit("TaskStatus", "Completed"),
        TaskState::Failed => Value::enum_unit("TaskStatus", "Failed"),
        TaskState::Stopped => Value::enum_unit("TaskStatus", "Stopped"),
    }
}

fn build_task_info_value(vm: &VM, task: &TaskInstance) -> Result<Value, LustError> {
    let last_yield = match &task.last_yield {
        Some(value) => Value::some(value.clone()),
        None => Value::none(),
    };
    let last_result = match &task.last_result {
        Some(value) => Value::some(value.clone()),
        None => Value::none(),
    };
    let error = match task.error.as_ref() {
        Some(err) => Value::some(Value::string(err.to_string())),
        None => Value::none(),
    };
    vm.instantiate_struct(
        "TaskInfo",
        vec![
            (
                Rc::new("state".to_string()),
                task_state_to_status_value(&task.state),
            ),
            (Rc::new("last_yield".to_string()), last_yield),
            (Rc::new("last_result".to_string()), last_result),
            (Rc::new("error".to_string()), error),
        ],
    )
}

// Helper functions for LuaValue operations
fn lua_op_neg(a: Value) -> Value {
    match unwrap_lua_value(a.clone()) {
        Value::Int(x) => Value::enum_variant("LuaValue", "Int", vec![Value::Int(-x)]),
        Value::Float(x) => Value::enum_variant("LuaValue", "Float", vec![Value::Float(-x)]),
        _ => a, // Return original value if not numeric
    }
}

fn lua_op_binary<F>(a: Value, b: Value, op: F) -> Value
where
    F: Fn(crate::lua_compat::LuaValue, crate::lua_compat::LuaValue) -> crate::lua_compat::LuaValue,
{
    let lua_a = value_to_rust_luavalue(&a);
    let lua_b = value_to_rust_luavalue(&b);
    let result = op(lua_a, lua_b);
    rust_luavalue_to_value(result)
}

fn lua_op_concat(a: Value, b: Value) -> Value {
    let a_str = match unwrap_lua_value(a.clone()) {
        Value::String(s) => s.to_string(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Nil => String::new(), // Treat nil as empty string, like Lua
        other => {
            #[cfg(feature = "std")]
            eprintln!(
                "[WARN] lua_op_concat: cannot concatenate {:?}, treating as empty string",
                other
            );
            String::new()
        }
    };
    let b_str = match unwrap_lua_value(b.clone()) {
        Value::String(s) => s.to_string(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Nil => String::new(), // Treat nil as empty string, like Lua
        other => {
            #[cfg(feature = "std")]
            eprintln!(
                "[WARN] lua_op_concat: cannot concatenate {:?}, treating as empty string",
                other
            );
            String::new()
        }
    };
    Value::enum_variant(
        "LuaValue",
        "String",
        vec![Value::String(Rc::new(a_str + &b_str))],
    )
}

fn value_to_rust_luavalue(v: &Value) -> crate::lua_compat::LuaValue {
    use crate::lua_compat::LuaValue;
    match unwrap_lua_value(v.clone()) {
        Value::Nil => LuaValue::Nil,
        Value::Bool(b) => LuaValue::Bool(b),
        Value::Int(i) => LuaValue::Int(i),
        Value::Float(f) => LuaValue::Float(f),
        Value::String(s) => LuaValue::String(s.to_string()),
        _ => LuaValue::Nil,
    }
}

fn rust_luavalue_to_value(lv: crate::lua_compat::LuaValue) -> Value {
    use crate::lua_compat::LuaValue;
    match lv {
        LuaValue::Nil => Value::enum_unit("LuaValue", "Nil"),
        LuaValue::Bool(b) => Value::enum_variant("LuaValue", "Bool", vec![Value::Bool(b)]),
        LuaValue::Int(i) => Value::enum_variant("LuaValue", "Int", vec![Value::Int(i)]),
        LuaValue::Float(f) => Value::enum_variant("LuaValue", "Float", vec![Value::Float(f)]),
        LuaValue::String(s) => {
            Value::enum_variant("LuaValue", "String", vec![Value::String(Rc::new(s))])
        }
        _ => Value::enum_unit("LuaValue", "Nil"),
    }
}

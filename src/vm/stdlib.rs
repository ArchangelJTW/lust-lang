use super::VM;
use super::corelib::{string_key, unwrap_lua_value};
use crate::LustInt;
use crate::bytecode::value::EnumObject;
use crate::bytecode::value::ValueKey;
use crate::bytecode::{NativeCallResult, Value, native_fn};
use crate::config::LustConfig;
#[cfg(not(target_arch = "wasm32"))]
use crate::lua_compat::register_lust_function;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng, make_rng};
use regex::Regex;
use std::io::{self, Read, Write};
use std::path::Path;
use std::rc::Rc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub fn create_stdlib(config: &LustConfig, vm: &VM) -> Vec<(&'static str, Value)> {
    let mut stdlib = vec![
        ("print", create_print_fn()),
        ("println", create_println_fn()),
        ("type", create_type_fn()),
        ("select", create_select_fn()),
        ("random", create_math_random_fn()),
        ("randomseed", create_math_randomseed_fn()),
        ("string", create_string_module(vm)),
        ("array", create_array_module(vm)),
        ("table", create_table_module(vm)),
        ("map", create_map_module(vm)),
        ("math", create_math_module(vm)),
        ("effect", create_effect_module(vm)),
        ("bit", create_bit_module(vm)),
        ("bitwise", create_bit_module(vm)),

        // Direct qualified entries for imported symbols
        ("effect.succeed", create_effect_succeed_fn()),
        ("effect.fail", create_effect_fail_fn()),
        ("effect.die", create_effect_die_fn()),
        ("effect.interrupt", create_effect_interrupt_fn()),
        ("effect.from_result", create_effect_from_result_fn()),
        ("effect.run_sync", create_effect_run_sync_fn()),
        ("effect.suspend", create_effect_suspend_fn()),
        ("math.abs", create_math_abs_fn()),
        ("math.floor", create_math_floor_fn()),
        ("math.ceil", create_math_ceil_fn()),
        ("math.round", create_math_round_fn()),
        ("math.sqrt", create_math_sqrt_fn()),
        ("math.sin", create_math_sin_fn()),
        ("math.cos", create_math_cos_fn()),
        ("math.tan", create_math_tan_fn()),
        ("math.asin", create_math_asin_fn()),
        ("math.acos", create_math_acos_fn()),
        ("math.atan", create_math_atan_fn()),
        ("math.atan2", create_math_atan2_fn()),
        ("math.min", create_math_min_fn()),
        ("math.max", create_math_max_fn()),
        ("math.clamp", create_math_clamp_fn()),
        ("math.deg", create_math_deg_fn()),
        ("math.rad", create_math_rad_fn()),
        ("math.exp", create_math_exp_fn()),
        ("math.log", create_math_log_fn()),
        ("math.tointeger", create_math_tointeger_fn()),
        ("math.tofloat", create_math_tofloat_fn()),
        ("math.mod", create_math_mod_fn()),
        ("math.random", create_math_random_fn()),
        ("math.randomseed", create_math_randomseed_fn()),
        ("math.pow", create_math_pow_fn()),
        ("bit.band", create_bit_band_fn()),
        ("bit.bor", create_bit_bor_fn()),
        ("bit.bxor", create_bit_bxor_fn()),
        ("bit.bnot", create_bit_bnot_fn()),
        ("bit.lshift", create_bit_lshift_fn()),
        ("bit.rshift", create_bit_rshift_fn()),
        ("bit.arshift", create_bit_arshift_fn()),
        ("bitwise.band", create_bit_band_fn()),
        ("bitwise.bor", create_bit_bor_fn()),
        ("bitwise.bxor", create_bit_bxor_fn()),
        ("bitwise.bnot", create_bit_bnot_fn()),
        ("bitwise.lshift", create_bit_lshift_fn()),
        ("bitwise.rshift", create_bit_rshift_fn()),
        ("bitwise.arshift", create_bit_arshift_fn()),

        ("table.len", create_array_len_fn()),
        ("table.is_empty", create_array_is_empty_fn()),
        ("table.get", create_array_get_fn()),
        ("table.first", create_array_first_fn()),
        ("table.last", create_array_last_fn()),
        ("table.push", create_array_push_fn()),
        ("table.pop", create_array_pop_fn()),
        ("table.insert", create_table_insert_fn()),
        ("table.remove", create_table_remove_fn()),
        ("table.clear", create_array_clear_fn()),
        ("table.slice", create_array_slice_fn()),
        ("table.concat", create_array_concat_fn()),
        ("table.sort", create_array_sort_fn()),
        ("table.reverse", create_array_reverse_fn()),
        ("table.contains", create_array_contains_fn()),
        ("table.map", create_array_map_fn()),
        ("table.filter", create_array_filter_fn()),
        ("table.reduce", create_array_reduce_fn()),
        ("table.iter", create_array_iter_fn()),
        ("table.unpack", create_table_unpack_fn()),

        ("array.len", create_array_len_fn()),
        ("array.is_empty", create_array_is_empty_fn()),
        ("array.get", create_array_get_fn()),
        ("array.first", create_array_first_fn()),
        ("array.last", create_array_last_fn()),
        ("array.push", create_array_push_fn()),
        ("array.pop", create_array_pop_fn()),
        ("array.insert", create_array_insert_fn()),
        ("array.remove", create_array_remove_fn()),
        ("array.clear", create_array_clear_fn()),
        ("array.slice", create_array_slice_fn()),
        ("array.concat", create_array_concat_fn()),
        ("array.sort", create_array_sort_fn()),
        ("array.reverse", create_array_reverse_fn()),
        ("array.contains", create_array_contains_fn()),
        ("array.map", create_array_map_fn()),
        ("array.filter", create_array_filter_fn()),
        ("array.reduce", create_array_reduce_fn()),
        ("array.iter", create_array_iter_fn()),

        ("string.len", create_string_len_fn()),
        ("string.sub", create_string_sub_fn()),
        ("string.find", create_string_find_fn()),
        ("string.lower", create_string_lower_fn()),
        ("string.upper", create_string_upper_fn()),
        ("string.rep", create_string_rep_fn()),
        ("string.reverse", create_string_reverse_fn()),
        ("string.byte", create_string_byte_fn()),
        ("string.char", create_string_char_fn()),
        ("string.format", create_string_format_fn()),
        ("string.split", create_string_split_fn()),
        ("string.trim", create_string_trim_fn()),
        ("string.starts_with", create_string_starts_with_fn()),
        ("string.ends_with", create_string_ends_with_fn()),
        ("string.contains", create_string_contains_fn()),
        ("string.replace", create_string_replace_fn()),
        ("string.match", create_string_match_fn()),
        ("string.gsub", create_string_gsub_fn()),
        ("string.trim_start", create_string_trim_start_fn()),
        ("string.trim_end", create_string_trim_end_fn()),
        ("string.is_empty", create_string_is_empty_fn()),
        ("string.chars", create_string_chars_fn()),
        ("string.lines", create_string_lines_fn()),
        ("string.slice", create_string_slice_fn()),
        ("string.index_of", create_string_index_of_fn()),

        ("map.len", create_map_len_fn()),
        ("map.is_empty", create_map_is_empty_fn()),
        ("map.get", create_map_get_fn()),
        ("map.set", create_map_set_fn()),
        ("map.delete", create_map_delete_fn()),
        ("map.clear", create_map_clear_fn()),
        ("map.has", create_map_has_fn()),
        ("map.keys", create_map_keys_fn()),
        ("map.values", create_map_values_fn()),
        ("map.iter", create_map_iter_fn()),
    ];
    if config.is_module_enabled("io") {
        stdlib.push(("io", create_io_module(vm)));
    }

    if config.is_module_enabled("os") {
        stdlib.push(("os", create_os_module(vm)));
    }

    stdlib
}

fn create_print_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let mut text = String::new();
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                text.push('\t');
            }
            use core::fmt::Write;
            let _ = write!(text, "{}", arg);
        }

        if let Some(vm_ptr) = crate::vm::current_vm_ptr() {
            unsafe {
                (*vm_ptr).console.write_stdout(text.as_bytes());
            }
        } else {
            print!("{}", text);
        }

        Ok(NativeCallResult::Return(Value::Nil))
    }))
}

fn create_println_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let mut text = String::new();
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                text.push('\t');
            }
            use core::fmt::Write;
            let _ = write!(text, "{}", arg);
        }
        text.push('\n');

        if let Some(vm_ptr) = crate::vm::current_vm_ptr() {
            unsafe {
                (*vm_ptr).console.write_stdout(text.as_bytes());
            }
        } else {
            print!("{}", text);
        }

        Ok(NativeCallResult::Return(Value::Nil))
    }))
}

fn create_type_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("type() requires at least one argument".to_string());
        }

        let value = &args[0];

        // Special handling for LuaValue enum - return Lua type names
        if let Value::Enum(object) = value
            && object.enum_name == "LuaValue"
        {
            let EnumObject { variant, .. } = object.as_ref();
            let lua_type = match variant.as_str() {
                "Nil" => "nil",
                "Bool" => "boolean",
                "Int" | "Float" => "number",
                "String" => "string",
                "Table" => "table",
                "Function" => "function",
                "Userdata" | "LightUserdata" => "userdata",
                "Thread" => "thread",
                _ => "unknown",
            };
            return Ok(NativeCallResult::Return(Value::enum_variant(
                "LuaValue",
                "String",
                vec![Value::string(lua_type)],
            )));
        }

        // Regular Lust types - also wrap in LuaValue for Lua compat
        let type_name = match value {
            Value::Nil => "nil",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Tuple(_) => "tuple",
            Value::Map(_) => "map",
            Value::Struct { .. } | Value::WeakStruct(_) => "struct",
            Value::Enum(_) => "enum",
            Value::Function(_) => "function",
            Value::NativeFunction(_) => "function",
            Value::Closure(_) => "function",
            Value::Iterator(_) => "iterator",
            Value::Task(_) => "task",
        };
        Ok(NativeCallResult::Return(Value::enum_variant(
            "LuaValue",
            "String",
            vec![Value::string(type_name)],
        )))
    }))
}

pub(crate) fn create_select_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("select expects at least one argument".to_string());
        }
        let selector = unwrap_lua_value(args[0].clone());
        let mut values: Vec<Value> = Vec::new();
        for arg in args.iter().skip(1) {
            let val = unwrap_lua_value(arg.clone());
            if let Some(arr) = val.as_array() {
                values.extend(arr);
            } else {
                values.push(val);
            }
        }
        if let Some(s) = selector.as_string() {
            if s == "#" {
                return Ok(NativeCallResult::Return(
                    Value::Int(values.len() as LustInt),
                ));
            } else {
                return Err("select expects '#' or an index as the first argument".to_string());
            }
        }
        let raw_idx = if let Some(i) = selector.as_int() {
            i
        } else if let Some(f) = selector.as_float() {
            f as LustInt
        } else {
            return Err("select expects '#' or an integer as the first argument".to_string());
        };
        let len = values.len() as isize;
        let mut start = if raw_idx < 0 {
            len + raw_idx as isize + 1
        } else {
            raw_idx as isize
        };
        if start < 1 {
            start = 1;
        }
        let start_idx = (start - 1) as usize;
        if start_idx >= values.len() {
            return return_lua_values(Vec::new());
        }
        return_lua_values(values[start_idx..].to_vec())
    }))
}

fn create_io_module(vm: &VM) -> Value {
    let entries = [
        (string_key("read_file"), create_io_read_file_fn()),
        (string_key("read_dir"), create_io_read_dir_fn()),
        (
            string_key("read_file_bytes"),
            create_io_read_file_bytes_fn(),
        ),
        (string_key("write_file"), create_io_write_file_fn()),
        (string_key("read_stdin"), create_io_read_stdin_fn()),
        (string_key("read_line"), create_io_read_line_fn()),
        (string_key("write_stdout"), create_io_write_stdout_fn()),
    ];
    vm.map_with_entries(entries)
}

fn create_io_read_file_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() != 1 {
            return Ok(NativeCallResult::Return(Value::err(Value::string(
                "io.read_file(path) requires a single string path",
            ))));
        }

        let path = match args[0].as_string() {
            Some(p) => p,
            None => {
                return Ok(NativeCallResult::Return(Value::err(Value::string(
                    "io.read_file(path) requires a string path",
                ))));
            }
        };
        let vfs = crate::vfs::current_vfs();
        match vfs.read_to_string(Path::new(path)) {
            Ok(contents) => Ok(NativeCallResult::Return(Value::ok(Value::string(contents)))),
            Err(err) => Ok(NativeCallResult::Return(Value::err(Value::string(
                err.to_string(),
            )))),
        }
    }))
}

fn create_io_read_file_bytes_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() != 1 {
            return Ok(NativeCallResult::Return(Value::err(Value::string(
                "io.read_file_bytes(path) requires a single string path",
            ))));
        }

        let path = match args[0].as_string() {
            Some(p) => p,
            None => {
                return Ok(NativeCallResult::Return(Value::err(Value::string(
                    "io.read_file_bytes(path) requires a string path",
                ))));
            }
        };

        let vfs = crate::vfs::current_vfs();
        match vfs.read(Path::new(path)) {
            Ok(bytes) => {
                let values: Vec<Value> = bytes
                    .into_iter()
                    .map(|b| Value::Int(b as LustInt))
                    .collect();
                Ok(NativeCallResult::Return(Value::ok(Value::array(values))))
            }

            Err(err) => Ok(NativeCallResult::Return(Value::err(Value::string(
                err.to_string(),
            )))),
        }
    }))
}

fn create_io_read_dir_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() != 1 {
            return Ok(NativeCallResult::Return(Value::err(Value::string(
                "io.read_dir(path) requires a single string path",
            ))));
        }

        let path = match args[0].as_string() {
            Some(path) => path,
            None => {
                return Ok(NativeCallResult::Return(Value::err(Value::string(
                    "io.read_dir(path) requires a string path",
                ))));
            }
        };

        let vfs = crate::vfs::current_vfs();
        match vfs.read_dir(Path::new(path)) {
            Ok(entries) => {
                let names = entries
                    .into_iter()
                    .map(|entry| {
                        Value::string(
                            entry
                                .file_name()
                                .map(|n| n.to_string_lossy().into_owned())
                                .unwrap_or_default(),
                        )
                    })
                    .collect::<Vec<_>>();
                Ok(NativeCallResult::Return(Value::ok(Value::array(names))))
            }
            Err(err) => Ok(NativeCallResult::Return(Value::err(Value::string(
                err.to_string(),
            )))),
        }
    }))
}

fn create_io_write_file_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Ok(NativeCallResult::Return(Value::err(Value::string(
                "io.write_file(path, contents) requires a path and value",
            ))));
        }

        let path = match args[0].as_string() {
            Some(p) => p,
            None => {
                return Ok(NativeCallResult::Return(Value::err(Value::string(
                    "io.write_file(path, contents) requires a string path",
                ))));
            }
        };
        let contents = if let Some(s) = args[1].as_string() {
            s.to_string()
        } else {
            format!("{}", args[1])
        };
        let vfs = crate::vfs::current_vfs();
        match vfs.write(Path::new(path), contents.as_bytes()) {
            Ok(_) => Ok(NativeCallResult::Return(Value::ok(Value::Nil))),
            Err(err) => Ok(NativeCallResult::Return(Value::err(Value::string(
                err.to_string(),
            )))),
        }
    }))
}

fn create_io_read_stdin_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if !args.is_empty() {
            return Ok(NativeCallResult::Return(Value::err(Value::string(
                "io.read_stdin() takes no arguments",
            ))));
        }

        if let Some(vm_ptr) = crate::vm::current_vm_ptr() {
            let mut buf = Vec::new();
            unsafe {
                let mut temp = [0u8; 1024];
                loop {
                    match (*vm_ptr).console.read_stdin(&mut temp) {
                        Ok(0) => break,
                        Ok(n) => buf.extend_from_slice(&temp[..n]),
                        Err(e) => {
                            return Ok(NativeCallResult::Return(Value::err(Value::string(
                                e.to_string(),
                            ))));
                        }
                    }
                }
            }
            let s = String::from_utf8_lossy(&buf).into_owned();
            Ok(NativeCallResult::Return(Value::ok(Value::string(s))))
        } else {
            let mut buffer = String::new();
            match io::stdin().read_to_string(&mut buffer) {
                Ok(_) => Ok(NativeCallResult::Return(Value::ok(Value::string(buffer)))),
                Err(err) => Ok(NativeCallResult::Return(Value::err(Value::string(
                    err.to_string(),
                )))),
            }
        }
    }))
}

fn create_io_read_line_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if !args.is_empty() {
            return Ok(NativeCallResult::Return(Value::err(Value::string(
                "io.read_line() takes no arguments",
            ))));
        }

        if let Some(vm_ptr) = crate::vm::current_vm_ptr() {
            match unsafe { (*vm_ptr).console.read_line_stdin() } {
                Ok(line) => Ok(NativeCallResult::Return(Value::ok(Value::string(line)))),
                Err(err) => Ok(NativeCallResult::Return(Value::err(Value::string(
                    err.to_string(),
                )))),
            }
        } else {
            let mut line = String::new();
            match io::stdin().read_line(&mut line) {
                Ok(_) => Ok(NativeCallResult::Return(Value::ok(Value::string(line)))),
                Err(err) => Ok(NativeCallResult::Return(Value::err(Value::string(
                    err.to_string(),
                )))),
            }
        }
    }))
}

fn create_io_write_stdout_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if let Some(vm_ptr) = crate::vm::current_vm_ptr() {
            unsafe {
                for arg in args {
                    let s = format!("{}", arg);
                    (*vm_ptr).console.write_stdout(s.as_bytes());
                }
            }
        } else {
            let mut stdout = io::stdout();
            for arg in args {
                if let Err(err) = write!(stdout, "{}", arg) {
                    return Ok(NativeCallResult::Return(Value::err(Value::string(
                        err.to_string(),
                    ))));
                }
            }

            if let Err(err) = stdout.flush() {
                return Ok(NativeCallResult::Return(Value::err(Value::string(
                    err.to_string(),
                ))));
            }
        }

        Ok(NativeCallResult::Return(Value::ok(Value::Nil)))
    }))
}

fn create_os_module(vm: &VM) -> Value {
    let entries = [
        (string_key("time"), create_os_time_fn()),
        (string_key("date"), create_os_date_fn()),
        (string_key("sleep"), create_os_sleep_fn()),
        (string_key("create_file"), create_os_create_file_fn()),
        (string_key("create_dir"), create_os_create_dir_fn()),
        (string_key("remove_file"), create_os_remove_file_fn()),
        (string_key("remove_dir"), create_os_remove_dir_fn()),
        (string_key("rename"), create_os_rename_fn()),
        (string_key("get_env"), create_os_get_env_fn()),
        (string_key("getenv"), create_os_get_env_fn()),
        (string_key("set_env"), create_os_set_env_fn()),
        (string_key("setenv"), create_os_set_env_fn()),
    ];
    vm.map_with_entries(entries)
}

fn create_os_time_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if !args.is_empty() {
            return Ok(NativeCallResult::Return(Value::err(Value::string(
                "os.time() takes no arguments",
            ))));
        }

        let seconds = if let Some(vm_ptr) = crate::vm::current_vm_ptr() {
            unsafe {
                let vm = &*vm_ptr;
                vm.clock.now_epoch_secs_f64()
            }
        } else {
            let now = SystemTime::now();
            match now.duration_since(UNIX_EPOCH) {
                Ok(duration) => duration.as_secs_f64(),
                Err(err) => -(err.duration().as_secs_f64()),
            }
        };

        Ok(NativeCallResult::Return(Value::Float(seconds)))
    }))
}

fn create_os_date_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let format_opt = if !args.is_empty() {
            match &args[0] {
                Value::Nil => None,
                Value::String(s) => Some(s.as_str()),
                _ => {
                    return Ok(NativeCallResult::Return(Value::err(Value::string(
                        "os.date(format, time) expects a string format",
                    ))));
                }
            }
        } else {
            None
        };

        let epoch_seconds = if args.len() >= 2 {
            match args[1].as_float() {
                Some(s) => s as i64,
                None => match args[1].as_int() {
                    Some(i) => i,
                    None => {
                        return Ok(NativeCallResult::Return(Value::err(Value::string(
                            "os.date(format, time) expects numeric time",
                        ))));
                    }
                },
            }
        } else if let Some(vm_ptr) = crate::vm::current_vm_ptr() {
            unsafe {
                let vm = &*vm_ptr;
                (vm.clock.now_epoch_nanos() / 1_000_000_000) as i64
            }
        } else {
            let now = SystemTime::now();
            match now.duration_since(UNIX_EPOCH) {
                Ok(d) => d.as_secs() as i64,
                Err(e) => -(e.duration().as_secs() as i64),
            }
        };

        let civil = crate::sim::clock::epoch_secs_to_civil(epoch_seconds);
        let fmt_str = format_opt.unwrap_or("%Y-%m-%d %H:%M:%S");

        if fmt_str == "*t" || fmt_str == "!*t" {
            if let Some(vm_ptr) = crate::vm::current_vm_ptr() {
                let vm = unsafe { &*vm_ptr };
                let entries = [
                    (string_key("year"), Value::Int(civil.year as i64)),
                    (string_key("month"), Value::Int(civil.month as i64)),
                    (string_key("day"), Value::Int(civil.day as i64)),
                    (string_key("hour"), Value::Int(civil.hour as i64)),
                    (string_key("min"), Value::Int(civil.min as i64)),
                    (string_key("sec"), Value::Int(civil.sec as i64)),
                    (string_key("wday"), Value::Int(civil.wday as i64)),
                    (string_key("yday"), Value::Int(civil.yday as i64)),
                    (string_key("isdst"), Value::Bool(civil.isdst)),
                ];
                return Ok(NativeCallResult::Return(vm.map_with_entries(entries)));
            } else {
                return Ok(NativeCallResult::Return(Value::string(civil.format_default())));
            }
        }

        let formatted = civil.format(fmt_str);
        Ok(NativeCallResult::Return(Value::string(formatted)))
    }))
}

fn create_os_sleep_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() != 1 {
            return Ok(NativeCallResult::Return(Value::err(Value::string(
                "os.sleep(seconds) requires a single float duration",
            ))));
        }

        let seconds = match args[0].as_float() {
            Some(value) => value,
            None => {
                return Ok(NativeCallResult::Return(Value::err(Value::string(
                    "os.sleep(seconds) requires a float duration",
                ))));
            }
        };

        if !seconds.is_finite() || seconds < 0.0 {
            return Ok(NativeCallResult::Return(Value::err(Value::string(
                "os.sleep(seconds) requires a finite, non-negative duration",
            ))));
        }

        if seconds > (u64::MAX as f64) {
            return Ok(NativeCallResult::Return(Value::err(Value::string(
                "os.sleep(seconds) duration is too large",
            ))));
        }

        Ok(NativeCallResult::YieldTimer(
            Duration::from_secs_f64(seconds),
            Value::ok(Value::Nil),
        ))
    }))
}

fn create_os_create_file_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() != 1 {
            return Ok(NativeCallResult::Return(Value::err(Value::string(
                "os.create_file(path) requires a single string path",
            ))));
        }

        let path = match args[0].as_string() {
            Some(p) => p,
            None => {
                return Ok(NativeCallResult::Return(Value::err(Value::string(
                    "os.create_file(path) requires a string path",
                ))));
            }
        };
        let vfs = crate::vfs::current_vfs();
        let path_obj = Path::new(path);
        if !vfs.exists(path_obj) {
            match vfs.write(path_obj, &[]) {
                Ok(_) => Ok(NativeCallResult::Return(Value::ok(Value::Nil))),
                Err(err) => Ok(NativeCallResult::Return(Value::err(Value::string(
                    err.to_string(),
                )))),
            }
        } else {
            Ok(NativeCallResult::Return(Value::ok(Value::Nil)))
        }
    }))
}

fn create_os_create_dir_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() != 1 {
            return Ok(NativeCallResult::Return(Value::err(Value::string(
                "os.create_dir(path) requires a single string path",
            ))));
        }

        let path = match args[0].as_string() {
            Some(p) => p,
            None => {
                return Ok(NativeCallResult::Return(Value::err(Value::string(
                    "os.create_dir(path) requires a string path",
                ))));
            }
        };
        let vfs = crate::vfs::current_vfs();
        match vfs.create_dir_all(Path::new(path)) {
            Ok(_) => Ok(NativeCallResult::Return(Value::ok(Value::Nil))),
            Err(err) => Ok(NativeCallResult::Return(Value::err(Value::string(
                err.to_string(),
            )))),
        }
    }))
}

fn create_os_remove_file_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() != 1 {
            return Ok(NativeCallResult::Return(Value::err(Value::string(
                "os.remove_file(path) requires a single string path",
            ))));
        }

        let path = match args[0].as_string() {
            Some(p) => p,
            None => {
                return Ok(NativeCallResult::Return(Value::err(Value::string(
                    "os.remove_file(path) requires a string path",
                ))));
            }
        };
        let vfs = crate::vfs::current_vfs();
        match vfs.remove_file(Path::new(path)) {
            Ok(_) => Ok(NativeCallResult::Return(Value::ok(Value::Nil))),
            Err(err) => Ok(NativeCallResult::Return(Value::err(Value::string(
                err.to_string(),
            )))),
        }
    }))
}

fn create_os_remove_dir_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() != 1 {
            return Ok(NativeCallResult::Return(Value::err(Value::string(
                "os.remove_dir(path) requires a single string path",
            ))));
        }

        let path = match args[0].as_string() {
            Some(p) => p,
            None => {
                return Ok(NativeCallResult::Return(Value::err(Value::string(
                    "os.remove_dir(path) requires a string path",
                ))));
            }
        };
        let vfs = crate::vfs::current_vfs();
        let path_obj = Path::new(path);
        let entries = match vfs.read_dir(path_obj) {
            Ok(e) => e,
            Err(err) => {
                return Ok(NativeCallResult::Return(Value::err(Value::string(
                    err.to_string(),
                ))));
            }
        };
        if !entries.is_empty() {
            return Ok(NativeCallResult::Return(Value::err(Value::string(
                "Directory not empty",
            ))));
        }
        match vfs.remove_dir_all(path_obj) {
            Ok(_) => Ok(NativeCallResult::Return(Value::ok(Value::Nil))),
            Err(err) => Ok(NativeCallResult::Return(Value::err(Value::string(
                err.to_string(),
            )))),
        }
    }))
}

fn create_os_rename_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() != 2 {
            return Ok(NativeCallResult::Return(Value::err(Value::string(
                "os.rename(from, to) requires two string paths",
            ))));
        }

        let from = match args[0].as_string() {
            Some(f) => f,
            None => {
                return Ok(NativeCallResult::Return(Value::err(Value::string(
                    "os.rename(from, to) requires string paths",
                ))));
            }
        };
        let to = match args[1].as_string() {
            Some(t) => t,
            None => {
                return Ok(NativeCallResult::Return(Value::err(Value::string(
                    "os.rename(from, to) requires string paths",
                ))));
            }
        };
        let vfs = crate::vfs::current_vfs();
        match vfs.rename(Path::new(from), Path::new(to)) {
            Ok(_) => Ok(NativeCallResult::Return(Value::ok(Value::Nil))),
            Err(err) => Ok(NativeCallResult::Return(Value::err(Value::string(
                err.to_string(),
            )))),
        }
    }))
}

fn create_os_get_env_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() != 1 {
            return Ok(NativeCallResult::Return(Value::err(Value::string(
                "os.get_env(name) requires a single string name",
            ))));
        }

        let name = match args[0].as_string() {
            Some(p) => p,
            None => {
                return Ok(NativeCallResult::Return(Value::err(Value::string(
                    "os.get_env(name) requires a string name",
                ))));
            }
        };

        let val_opt = if let Some(vm_ptr) = crate::vm::current_vm_ptr() {
            unsafe { (*vm_ptr).env.get(name) }
        } else {
            std::env::var(name).ok()
        };

        match val_opt {
            Some(value) => Ok(NativeCallResult::Return(Value::ok(Value::some(
                Value::string(value),
            )))),
            None => Ok(NativeCallResult::Return(Value::ok(Value::none()))),
        }
    }))
}

fn create_os_set_env_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() != 2 {
            return Ok(NativeCallResult::Return(Value::err(Value::string(
                "os.set_env(name, value) requires two string arguments",
            ))));
        }

        let name = match args[0].as_string() {
            Some(p) => p,
            None => {
                return Ok(NativeCallResult::Return(Value::err(Value::string(
                    "os.set_env(name, value) requires a string name",
                ))));
            }
        };

        let val = match args[1].as_string() {
            Some(v) => v,
            None => {
                return Ok(NativeCallResult::Return(Value::err(Value::string(
                    "os.set_env(name, value) requires a string value",
                ))));
            }
        };

        if let Some(vm_ptr) = crate::vm::current_vm_ptr() {
            unsafe {
                (*vm_ptr).env.set(name, val);
            }
        }
        Ok(NativeCallResult::Return(Value::ok(Value::Nil)))
    }))
}

fn create_string_module(vm: &VM) -> Value {
    let entries = [
        (string_key("len"), create_string_len_fn()),
        (string_key("lower"), create_string_lower_fn()),
        (string_key("upper"), create_string_upper_fn()),
        (string_key("sub"), create_string_sub_fn()),
        (string_key("byte"), create_string_byte_fn()),
        (string_key("char"), create_string_char_fn()),
        (string_key("find"), create_string_find_fn()),
        (string_key("match"), create_string_match_fn()),
        (string_key("gsub"), create_string_gsub_fn()),
        (string_key("format"), create_string_format_fn()),
        (string_key("rep"), create_string_rep_fn()),
        (string_key("reverse"), create_string_reverse_fn()),
        (string_key("split"), create_string_split_fn()),
        (string_key("trim"), create_string_trim_fn()),
        (string_key("trim_start"), create_string_trim_start_fn()),
        (string_key("trim_end"), create_string_trim_end_fn()),
        (string_key("replace"), create_string_replace_fn()),
        (string_key("starts_with"), create_string_starts_with_fn()),
        (string_key("ends_with"), create_string_ends_with_fn()),
        (string_key("contains"), create_string_contains_fn()),
        (string_key("is_empty"), create_string_is_empty_fn()),
        (string_key("chars"), create_string_chars_fn()),
        (string_key("lines"), create_string_lines_fn()),
        (string_key("slice"), create_string_slice_fn()),
        (string_key("index_of"), create_string_index_of_fn()),
    ];
    vm.map_with_entries(entries)
}

fn create_string_len_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let input = args.first().cloned().unwrap_or(Value::Nil);
        let value = unwrap_lua_value(input);
        match value {
            Value::Nil => Ok(NativeCallResult::Return(Value::Int(0))), // Nil has length 0
            Value::String(s) => Ok(NativeCallResult::Return(Value::Int(s.len() as LustInt))),
            other => Err(format!("string.len expects a string, got {:?}", other)),
        }
    }))
}

fn create_string_lower_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let value = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let s = value
            .as_string()
            .ok_or_else(|| "string.lower expects a string".to_string())?;
        Ok(NativeCallResult::Return(Value::string(s.to_lowercase())))
    }))
}

fn create_string_upper_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let value = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let s = value
            .as_string()
            .ok_or_else(|| "string.upper expects a string".to_string())?;
        Ok(NativeCallResult::Return(Value::string(s.to_uppercase())))
    }))
}

fn create_string_sub_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let value = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let source = value
            .as_string()
            .ok_or_else(|| "string.sub expects a string".to_string())?;
        let start = args
            .get(1)
            .map(|v| unwrap_lua_value(v.clone()).as_int().unwrap_or(1))
            .unwrap_or(1);
        let end = args.get(2).map(|v| {
            unwrap_lua_value(v.clone())
                .as_int()
                .unwrap_or(source.len() as LustInt)
        });
        let (start_idx, end_idx) = normalize_range(start, end, source.len());
        if start_idx >= source.len() || start_idx >= end_idx {
            return Ok(NativeCallResult::Return(Value::string("")));
        }
        let slice = &source.as_bytes()[start_idx..end_idx.min(source.len())];
        Ok(NativeCallResult::Return(Value::string(
            String::from_utf8_lossy(slice),
        )))
    }))
}

fn create_string_byte_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let value = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let source = value
            .as_string()
            .ok_or_else(|| "string.byte expects a string".to_string())?;
        let start = args
            .get(1)
            .map(|v| unwrap_lua_value(v.clone()).as_int().unwrap_or(1))
            .unwrap_or(1);
        let end = args
            .get(2)
            .map(|v| unwrap_lua_value(v.clone()).as_int().unwrap_or(start));
        let (start_idx, end_idx) = normalize_range(start, end, source.len());
        let bytes = source.as_bytes();
        if start_idx >= bytes.len() || start_idx >= end_idx {
            return return_lua_values(vec![lua_nil()]);
        }
        let mut values = Vec::new();
        for b in &bytes[start_idx..end_idx.min(bytes.len())] {
            values.push(Value::Int(*b as LustInt));
        }
        return_lua_values(values)
    }))
}

fn create_string_char_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let mut output = String::new();
        for arg in args {
            let raw = unwrap_lua_value(arg.clone());
            let code = raw
                .as_int()
                .or_else(|| raw.as_float().map(|f| f as LustInt))
                .ok_or_else(|| "string.char expects numeric arguments".to_string())?;
            if !(0..=255).contains(&code) {
                return Err("string.char codepoints must be in [0,255]".to_string());
            }
            if let Some(ch) = char::from_u32(code as u32) {
                output.push(ch);
            }
        }
        Ok(NativeCallResult::Return(Value::string(output)))
    }))
}

fn create_string_find_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let subject = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let pattern_val = unwrap_lua_value(args.get(1).cloned().unwrap_or(Value::Nil));
        let haystack = subject
            .as_string()
            .ok_or_else(|| "string.find expects a string subject".to_string())?;
        let pattern = pattern_val
            .as_string()
            .ok_or_else(|| "string.find expects a pattern string".to_string())?;
        let start = args
            .get(2)
            .map(|v| unwrap_lua_value(v.clone()).as_int().unwrap_or(1))
            .unwrap_or(1);
        let plain = args
            .get(3)
            .map(|v| matches!(unwrap_lua_value(v.clone()), Value::Bool(true)))
            .unwrap_or(false);
        let (offset, _) = normalize_range(start, None, haystack.len());
        if offset > haystack.len() {
            return return_lua_values(vec![lua_nil()]);
        }
        let slice = haystack.get(offset..).unwrap_or("");
        if plain {
            if let Some(pos) = slice.find(pattern) {
                let begin = offset + pos;
                let end = begin + pattern.len().saturating_sub(1);
                return return_lua_values(vec![
                    Value::Int((begin as LustInt) + 1),
                    Value::Int((end as LustInt) + 1),
                ]);
            }
            return return_lua_values(vec![lua_nil()]);
        }
        let regex = lua_pattern_to_regex(pattern)?;
        if let Some(caps) = regex.captures(slice)
            && let Some(mat) = caps.get(0)
        {
            let begin = offset + mat.start();
            let end = offset + mat.end().saturating_sub(1);
            let mut results: Vec<Value> = vec![
                Value::Int((begin as LustInt) + 1),
                Value::Int((end as LustInt) + 1),
            ];
            for idx in 1..caps.len() {
                if let Some(c) = caps.get(idx) {
                    results.push(Value::string(c.as_str()));
                } else {
                    results.push(lua_nil());
                }
            }
            return return_lua_values(results);
        }
        return_lua_values(vec![lua_nil()])
    }))
}

fn create_string_match_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let subject = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let pattern_val = unwrap_lua_value(args.get(1).cloned().unwrap_or(Value::Nil));
        let haystack = subject
            .as_string()
            .ok_or_else(|| "string.match expects a string subject".to_string())?;
        let pattern = pattern_val
            .as_string()
            .ok_or_else(|| "string.match expects a pattern string".to_string())?;
        let start = args
            .get(2)
            .map(|v| unwrap_lua_value(v.clone()).as_int().unwrap_or(1))
            .unwrap_or(1);
        let (offset, _) = normalize_range(start, None, haystack.len());
        if offset > haystack.len() {
            return Ok(NativeCallResult::Return(Value::Nil));
        }
        let slice = haystack.get(offset..).unwrap_or("");
        let regex = lua_pattern_to_regex(pattern)?;
        let Some(caps) = regex.captures(slice) else {
            return Ok(NativeCallResult::Return(Value::Nil));
        };
        let Some(mat) = caps.get(0) else {
            return Ok(NativeCallResult::Return(Value::Nil));
        };

        let capture_count = caps.len().saturating_sub(1);
        if capture_count == 0 {
            return Ok(NativeCallResult::Return(Value::string(mat.as_str())));
        }
        if capture_count == 1 {
            if let Some(c) = caps.get(1) {
                return Ok(NativeCallResult::Return(Value::string(c.as_str())));
            }
            return Ok(NativeCallResult::Return(Value::Nil));
        }

        let mut results = Vec::with_capacity(capture_count);
        for idx in 1..caps.len() {
            if let Some(c) = caps.get(idx) {
                results.push(Value::string(c.as_str()));
            } else {
                results.push(Value::Nil);
            }
        }
        return_lua_values(results)
    }))
}

fn create_string_gsub_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let subject = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let pattern_val = unwrap_lua_value(args.get(1).cloned().unwrap_or(Value::Nil));
        let repl = args.get(2).cloned().unwrap_or(Value::Nil);
        let limit = args
            .get(3)
            .map(|v| unwrap_lua_value(v.clone()).as_int().unwrap_or(-1))
            .unwrap_or(-1);
        let text = subject
            .as_string()
            .ok_or_else(|| "string.gsub expects a string subject".to_string())?;
        let pattern = pattern_val
            .as_string()
            .ok_or_else(|| "string.gsub expects a pattern string".to_string())?;
        let regex = lua_pattern_to_regex(pattern)?;
        let max_repls = if limit < 0 { i64::MAX } else { limit };
        enum Replacer {
            String(String),
            Func(Value),
            Other(Value),
        }
        let replacer = match &repl {
            Value::Enum(object)
                if object.enum_name == "LuaValue" && object.variant == "Function" =>
            {
                Replacer::Func(repl.clone())
            }
            Value::Function(_) | Value::NativeFunction(_) | Value::Closure(_) => {
                Replacer::Func(repl.clone())
            }
            _ => {
                let unwrapped = unwrap_lua_value(repl.clone());
                match unwrapped {
                    Value::String(s) => Replacer::String(s.to_string()),
                    other => Replacer::Other(other),
                }
            }
        };
        let mut last_end = 0;
        let mut count: i64 = 0;
        let mut output = String::new();
        for caps in regex.captures_iter(text) {
            if count >= max_repls {
                break;
            }
            let mat = caps.get(0).unwrap();
            output.push_str(&text[last_end..mat.start()]);
            let replacement = match &replacer {
                Replacer::String(template) => build_template_replacement(template.as_str(), &caps),
                Replacer::Func(func_val) => VM::with_current(|vm| {
                    let mut call_args = Vec::new();
                    if caps.len() > 1 {
                        for idx in 1..caps.len() {
                            if let Some(c) = caps.get(idx) {
                                call_args.push(to_lua_value(vm, Value::string(c.as_str()))?);
                            } else {
                                call_args.push(lua_nil());
                            }
                        }
                    } else if let Some(m) = caps.get(0) {
                        call_args.push(to_lua_value(vm, Value::string(m.as_str()))?);
                    }
                    let result = vm
                        .call_value(func_val, call_args)
                        .map_err(|e| e.to_string())?;
                    let first = unwrap_first_return(result);
                    Ok(first.to_string())
                })?,
                Replacer::Other(other) => other.to_string(),
            };
            output.push_str(&replacement);
            last_end = mat.end();
            count += 1;
        }
        output.push_str(&text[last_end..]);
        return_lua_values(vec![Value::string(output), Value::Int(count)])
    }))
}

fn create_string_format_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("string.format requires a format string".to_string());
        }
        let fmt_val = unwrap_lua_value(args[0].clone());
        let fmt = fmt_val
            .as_string()
            .ok_or_else(|| "string.format expects a string format".to_string())?;
        let rendered = render_format(fmt, &args[1..])?;
        Ok(NativeCallResult::Return(Value::string(rendered)))
    }))
}

fn create_string_rep_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("string.rep requires at least a string and count".to_string());
        }
        let val = unwrap_lua_value(args[0].clone());
        let s = val
            .as_string()
            .ok_or_else(|| "string.rep expects a string".to_string())?;
        let n = args
            .get(1)
            .and_then(|v| unwrap_lua_value(v.clone()).as_int())
            .unwrap_or(0);
        if n <= 0 {
            return Ok(NativeCallResult::Return(Value::string("")));
        }
        let sep_val = args.get(2).map(|v| unwrap_lua_value(v.clone()));
        let sep = sep_val
            .as_ref()
            .and_then(|v| v.as_string())
            .unwrap_or_default();
        let mut out = String::new();
        for i in 0..n {
            if i > 0 && !sep.is_empty() {
                out.push_str(sep);
            }
            out.push_str(s);
        }
        Ok(NativeCallResult::Return(Value::string(out)))
    }))
}

fn create_string_reverse_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let s = val
            .as_string()
            .ok_or_else(|| "string.reverse expects a string".to_string())?;
        let reversed: String = s.chars().rev().collect();
        Ok(NativeCallResult::Return(Value::string(reversed)))
    }))
}

fn create_string_split_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let s = val
            .as_string()
            .ok_or_else(|| "string.split expects a string".to_string())?;
        let delim_val = args.get(1).map(|v| unwrap_lua_value(v.clone()));
        let delimiter = delim_val
            .as_ref()
            .and_then(|v| v.as_string())
            .unwrap_or_default();
        let parts: Vec<Value> = if delimiter.is_empty() {
            s.chars().map(|c| Value::string(c.to_string())).collect()
        } else {
            s.split(delimiter).map(Value::string).collect()
        };
        Ok(NativeCallResult::Return(Value::array(parts)))
    }))
}

fn create_string_trim_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let s = val
            .as_string()
            .ok_or_else(|| "string.trim expects a string".to_string())?;
        Ok(NativeCallResult::Return(Value::string(s.trim())))
    }))
}

fn create_string_trim_start_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let s = val
            .as_string()
            .ok_or_else(|| "string.trim_start expects a string".to_string())?;
        Ok(NativeCallResult::Return(Value::string(s.trim_start())))
    }))
}

fn create_string_trim_end_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let s = val
            .as_string()
            .ok_or_else(|| "string.trim_end expects a string".to_string())?;
        Ok(NativeCallResult::Return(Value::string(s.trim_end())))
    }))
}

fn create_string_replace_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let s = val
            .as_string()
            .ok_or_else(|| "string.replace expects a string".to_string())?;
        let from_val = args.get(1).map(|v| unwrap_lua_value(v.clone()));
        let from = from_val
            .as_ref()
            .and_then(|v| v.as_string())
            .ok_or_else(|| "string.replace expects a 'from' string".to_string())?;
        let to_val = args.get(2).map(|v| unwrap_lua_value(v.clone()));
        let to = to_val
            .as_ref()
            .and_then(|v| v.as_string())
            .ok_or_else(|| "string.replace expects a 'to' string".to_string())?;
        Ok(NativeCallResult::Return(Value::string(s.replace(from, to))))
    }))
}

fn create_string_starts_with_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let s = val
            .as_string()
            .ok_or_else(|| "string.starts_with expects a string".to_string())?;
        let p_val = args.get(1).map(|v| unwrap_lua_value(v.clone()));
        let prefix = p_val
            .as_ref()
            .and_then(|v| v.as_string())
            .ok_or_else(|| "string.starts_with expects a prefix string".to_string())?;
        Ok(NativeCallResult::Return(Value::Bool(s.starts_with(prefix))))
    }))
}

fn create_string_ends_with_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let s = val
            .as_string()
            .ok_or_else(|| "string.ends_with expects a string".to_string())?;
        let s_val = args.get(1).map(|v| unwrap_lua_value(v.clone()));
        let suffix = s_val
            .as_ref()
            .and_then(|v| v.as_string())
            .ok_or_else(|| "string.ends_with expects a suffix string".to_string())?;
        Ok(NativeCallResult::Return(Value::Bool(s.ends_with(suffix))))
    }))
}

fn create_string_contains_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let s = val
            .as_string()
            .ok_or_else(|| "string.contains expects a string".to_string())?;
        let sub_val = args.get(1).map(|v| unwrap_lua_value(v.clone()));
        let substr = sub_val
            .as_ref()
            .and_then(|v| v.as_string())
            .ok_or_else(|| "string.contains expects a substring".to_string())?;
        Ok(NativeCallResult::Return(Value::Bool(s.contains(substr))))
    }))
}

fn create_string_is_empty_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let s = val
            .as_string()
            .ok_or_else(|| "string.is_empty expects a string".to_string())?;
        Ok(NativeCallResult::Return(Value::Bool(s.is_empty())))
    }))
}

fn create_string_chars_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let s = val
            .as_string()
            .ok_or_else(|| "string.chars expects a string".to_string())?;
        let chars: Vec<Value> = s.chars().map(|c| Value::string(c.to_string())).collect();
        Ok(NativeCallResult::Return(Value::array(chars)))
    }))
}

fn create_string_lines_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let s = val
            .as_string()
            .ok_or_else(|| "string.lines expects a string".to_string())?;
        let lines: Vec<Value> = s.lines().map(Value::string).collect();
        Ok(NativeCallResult::Return(Value::array(lines)))
    }))
}

fn create_string_slice_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let value = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let source = value
            .as_string()
            .ok_or_else(|| "string.slice expects a string as first argument".to_string())?;
        let start = args
            .get(1)
            .and_then(Value::as_int)
            .ok_or_else(|| "string.slice expects an integer start index".to_string())?;
        let end = args
            .get(2)
            .and_then(Value::as_int)
            .ok_or_else(|| "string.slice expects an integer end index".to_string())?;

        let len = source.len();
        let start_idx = (start.max(0) as usize).min(len);
        let end_idx = (end.max(0) as usize).min(len);

        if start_idx >= end_idx {
            return Ok(NativeCallResult::Return(Value::string("")));
        }
        let slice = &source.as_bytes()[start_idx..end_idx];
        Ok(NativeCallResult::Return(Value::string(
            String::from_utf8_lossy(slice),
        )))
    }))
}

fn create_string_index_of_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let value = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let source = value
            .as_string()
            .ok_or_else(|| "string.index_of expects a string as first argument".to_string())?;
        let substr_val = unwrap_lua_value(args.get(1).cloned().unwrap_or(Value::Nil));
        let substr = substr_val
            .as_string()
            .ok_or_else(|| "string.index_of expects a substring as second argument".to_string())?;

        match source.find(&substr) {
            Some(idx) => Ok(NativeCallResult::Return(Value::some(Value::Int(idx as LustInt)))),
            None => Ok(NativeCallResult::Return(Value::none())),
        }
    }))
}

fn create_array_module(vm: &VM) -> Value {
    let entries = [
        (string_key("len"), create_array_len_fn()),
        (string_key("is_empty"), create_array_is_empty_fn()),
        (string_key("get"), create_array_get_fn()),
        (string_key("first"), create_array_first_fn()),
        (string_key("last"), create_array_last_fn()),
        (string_key("push"), create_array_push_fn()),
        (string_key("pop"), create_array_pop_fn()),
        (string_key("insert"), create_array_insert_fn()),
        (string_key("remove"), create_array_remove_fn()),
        (string_key("clear"), create_array_clear_fn()),
        (string_key("slice"), create_array_slice_fn()),
        (string_key("concat"), create_array_concat_fn()),
        (string_key("sort"), create_array_sort_fn()),
        (string_key("reverse"), create_array_reverse_fn()),
        (string_key("contains"), create_array_contains_fn()),
        (string_key("map"), create_array_map_fn()),
        (string_key("filter"), create_array_filter_fn()),
        (string_key("reduce"), create_array_reduce_fn()),
        (string_key("iter"), create_array_iter_fn()),
    ];
    vm.map_with_entries(entries)
}

fn create_array_len_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = args.first().cloned().unwrap_or(Value::Nil);
        match val {
            Value::Array(arr) => {
                let len = crate::bytecode::value::get_specialized_cached_len(&arr)
                    .unwrap_or_else(|| arr.borrow().len());
                Ok(NativeCallResult::Return(Value::Int(len as LustInt)))
            }
            other => Err(format!(
                "array.len expects an array, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_array_is_empty_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = args.first().cloned().unwrap_or(Value::Nil);
        match val {
            Value::Array(arr) => {
                let is_empty = crate::bytecode::value::get_specialized_cached_len(&arr)
                    .map(|l| l == 0)
                    .unwrap_or_else(|| arr.borrow().is_empty());
                Ok(NativeCallResult::Return(Value::Bool(is_empty)))
            }
            other => Err(format!(
                "array.is_empty expects an array, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_array_get_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = args.first().cloned().unwrap_or(Value::Nil);
        let index = args
            .get(1)
            .and_then(Value::as_int)
            .ok_or_else(|| "array.get expects an integer index".to_string())?;
        match val {
            Value::Array(arr) => {
                if let Some(elem) = crate::bytecode::value::get_specialized_cached_element(&arr, index) {
                    Ok(NativeCallResult::Return(Value::some(elem)))
                } else if crate::bytecode::value::get_specialized_cached_len(&arr).is_some() {
                    Ok(NativeCallResult::Return(Value::none()))
                } else {
                    let borrowed = arr.borrow();
                    if index < 0 || (index as usize) >= borrowed.len() {
                        Ok(NativeCallResult::Return(Value::none()))
                    } else {
                        Ok(NativeCallResult::Return(Value::some(
                            borrowed[index as usize].clone(),
                        )))
                    }
                }
            }
            other => Err(format!(
                "array.get expects an array, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_array_first_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = args.first().cloned().unwrap_or(Value::Nil);
        match val {
            Value::Array(arr) => {
                if let Some(elem) = crate::bytecode::value::get_specialized_cached_element(&arr, 0) {
                    Ok(NativeCallResult::Return(Value::some(elem)))
                } else if let Some(len) = crate::bytecode::value::get_specialized_cached_len(&arr) {
                    if len == 0 {
                        Ok(NativeCallResult::Return(Value::none()))
                    } else {
                        crate::bytecode::value::sync_specialized_array(&arr);
                        let borrowed = arr.borrow();
                        Ok(NativeCallResult::Return(
                            borrowed.first().cloned().map(Value::some).unwrap_or_else(Value::none),
                        ))
                    }
                } else {
                    let borrowed = arr.borrow();
                    Ok(NativeCallResult::Return(
                        borrowed
                            .first()
                            .cloned()
                            .map(Value::some)
                            .unwrap_or_else(Value::none),
                    ))
                }
            }
            other => Err(format!(
                "array.first expects an array, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_array_last_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = args.first().cloned().unwrap_or(Value::Nil);
        match val {
            Value::Array(arr) => {
                if let Some(len) = crate::bytecode::value::get_specialized_cached_len(&arr) {
                    if len == 0 {
                        Ok(NativeCallResult::Return(Value::none()))
                    } else if let Some(elem) = crate::bytecode::value::get_specialized_cached_element(&arr, (len - 1) as i64) {
                        Ok(NativeCallResult::Return(Value::some(elem)))
                    } else {
                        crate::bytecode::value::sync_specialized_array(&arr);
                        let borrowed = arr.borrow();
                        Ok(NativeCallResult::Return(
                            borrowed.last().cloned().map(Value::some).unwrap_or_else(Value::none),
                        ))
                    }
                } else {
                    let borrowed = arr.borrow();
                    Ok(NativeCallResult::Return(
                        borrowed
                            .last()
                            .cloned()
                            .map(Value::some)
                            .unwrap_or_else(Value::none),
                    ))
                }
            }
            other => Err(format!(
                "array.last expects an array, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_array_push_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("array.push requires an array and a value".to_string());
        }
        match &args[0] {
            Value::Array(arr) => {
                let mut borrowed = arr.borrow_mut();
                let len = borrowed.len();
                let cap = borrowed.capacity();
                if len == cap {
                    let new_cap = if cap == 0 { 4 } else { cap.saturating_mul(2) };
                    VM::with_current(|vm| {
                        vm.charge_vec_growth::<Value>(cap, new_cap)
                            .map_err(|e| e.to_string())
                    })?;
                }
                if !crate::vm::cycle::CycleCollector::is_leaf(&args[1]) {
                    let _ = VM::with_current(|vm| {
                        vm.cycle_collector.record_mutation();
                        Ok::<(), String>(())
                    });
                }
                borrowed.push(args[1].clone());
                crate::bytecode::value::invalidate_specialized_array_cache(arr.as_ptr() as usize);
                Ok(NativeCallResult::Return(Value::Nil))
            }
            other => Err(format!(
                "array.push expects an array, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_array_pop_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = args.first().cloned().unwrap_or(Value::Nil);
        match val {
            Value::Array(arr) => {
                crate::bytecode::value::invalidate_specialized_array_cache(arr.as_ptr() as usize);
                let popped = arr.borrow_mut().pop();
                Ok(NativeCallResult::Return(
                    popped.map(Value::some).unwrap_or_else(Value::none),
                ))
            }
            other => Err(format!(
                "array.pop expects an array, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_array_insert_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 3 {
            return Err("array.insert requires array, index, and value".to_string());
        }
        let index = args[1]
            .as_int()
            .ok_or_else(|| "array.insert expects an integer index".to_string())?;
        match &args[0] {
            Value::Array(arr) => {
                let mut borrowed = arr.borrow_mut();
                let len = borrowed.len();
                let cap = borrowed.capacity();
                if len == cap {
                    let new_cap = if cap == 0 { 4 } else { cap.saturating_mul(2) };
                    VM::with_current(|vm| {
                        vm.charge_vec_growth::<Value>(cap, new_cap)
                            .map_err(|e| e.to_string())
                    })?;
                }
                if !crate::vm::cycle::CycleCollector::is_leaf(&args[2]) {
                    let _ = VM::with_current(|vm| {
                        vm.cycle_collector.record_mutation();
                        Ok::<(), String>(())
                    });
                }
                let idx = (index.max(0) as usize).min(borrowed.len());
                borrowed.insert(idx, args[2].clone());
                crate::bytecode::value::invalidate_specialized_array_cache(arr.as_ptr() as usize);
                Ok(NativeCallResult::Return(Value::Nil))
            }
            other => Err(format!(
                "array.insert expects an array, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_array_remove_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("array.remove requires array and index".to_string());
        }
        let index = args[1]
            .as_int()
            .ok_or_else(|| "array.remove expects an integer index".to_string())?;
        match &args[0] {
            Value::Array(arr) => {
                crate::bytecode::value::invalidate_specialized_array_cache(arr.as_ptr() as usize);
                let mut borrowed = arr.borrow_mut();
                if index >= 0 && (index as usize) < borrowed.len() {
                    let val = borrowed.remove(index as usize);
                    Ok(NativeCallResult::Return(Value::some(val)))
                } else {
                    Ok(NativeCallResult::Return(Value::none()))
                }
            }
            other => Err(format!(
                "array.remove expects an array, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_array_clear_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = args.first().cloned().unwrap_or(Value::Nil);
        match val {
            Value::Array(arr) => {
                crate::bytecode::value::invalidate_specialized_array_cache(arr.as_ptr() as usize);
                arr.borrow_mut().clear();
                Ok(NativeCallResult::Return(Value::Nil))
            }
            other => Err(format!(
                "array.clear expects an array, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_array_slice_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 3 {
            return Err("array.slice requires array, start, and end indices".to_string());
        }
        let start = args[1]
            .as_int()
            .ok_or_else(|| "array.slice expects an integer start index".to_string())?;
        let end = args[2]
            .as_int()
            .ok_or_else(|| "array.slice expects an integer end index".to_string())?;
        match &args[0] {
            Value::Array(arr) => {
                let borrowed = arr.borrow();
                let s = (start.max(0) as usize).min(borrowed.len());
                let e = (end.max(0) as usize).min(borrowed.len());
                if s > e {
                    return Ok(NativeCallResult::Return(Value::array(Vec::new())));
                }
                let sliced = borrowed[s..e].to_vec();
                Ok(NativeCallResult::Return(Value::array(sliced)))
            }
            other => Err(format!(
                "array.slice expects an array, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_array_concat_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = args.first().cloned().unwrap_or(Value::Nil);
        let sep_val = args.get(1).map(|v| unwrap_lua_value(v.clone()));
        let sep = sep_val
            .as_ref()
            .and_then(|v| v.as_string())
            .unwrap_or_default();
        match val {
            Value::Array(arr) => {
                let borrowed = arr.borrow();
                let parts: Vec<String> = borrowed.iter().map(|v| format!("{}", v)).collect();
                Ok(NativeCallResult::Return(Value::string(parts.join(sep))))
            }
            other => Err(format!(
                "array.concat expects an array, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_array_sort_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = args.first().cloned().unwrap_or(Value::Nil);
        let comp = args.get(1).cloned();
        match val {
            Value::Array(arr) => {
                let mut borrowed = arr.borrow_mut();
                if let Some(comp_fn) = comp.filter(|c| !matches!(c, Value::Nil)) {
                    let mut sort_err = None;
                    borrowed.sort_by(|a, b| {
                        if sort_err.is_some() {
                            return std::cmp::Ordering::Equal;
                        }
                        let res = VM::with_current(|vm| {
                            vm.call_value(&comp_fn, vec![a.clone(), b.clone()])
                                .map_err(|e| e.to_string())
                        });
                        match res {
                            Ok(ret) => {
                                let v = unwrap_first_return(ret);
                                if v.is_truthy() {
                                    std::cmp::Ordering::Less
                                } else {
                                    std::cmp::Ordering::Greater
                                }
                            }
                            Err(e) => {
                                sort_err = Some(e);
                                std::cmp::Ordering::Equal
                            }
                        }
                    });
                    if let Some(e) = sort_err {
                        return Err(e);
                    }
                } else {
                    borrowed.sort_by(|a, b| match (a, b) {
                        (Value::Int(i1), Value::Int(i2)) => i1.cmp(i2),
                        (Value::Float(f1), Value::Float(f2)) => {
                            f1.partial_cmp(f2).unwrap_or(std::cmp::Ordering::Equal)
                        }
                        (Value::Int(i), Value::Float(f)) => (*i as f64)
                            .partial_cmp(f)
                            .unwrap_or(std::cmp::Ordering::Equal),
                        (Value::Float(f), Value::Int(i)) => f
                            .partial_cmp(&(*i as f64))
                            .unwrap_or(std::cmp::Ordering::Equal),
                        (Value::String(s1), Value::String(s2)) => s1.cmp(s2),
                        _ => format!("{}", a).cmp(&format!("{}", b)),
                    });
                }
                Ok(NativeCallResult::Return(Value::Nil))
            }
            other => Err(format!(
                "array.sort expects an array, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_array_reverse_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = args.first().cloned().unwrap_or(Value::Nil);
        match val {
            Value::Array(arr) => {
                arr.borrow_mut().reverse();
                Ok(NativeCallResult::Return(Value::Nil))
            }
            other => Err(format!(
                "array.reverse expects an array, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_array_contains_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("array.contains requires an array and a value".to_string());
        }
        match &args[0] {
            Value::Array(arr) => {
                let target = &args[1];
                let contains = arr.borrow().iter().any(|v| v == target);
                Ok(NativeCallResult::Return(Value::Bool(contains)))
            }
            other => Err(format!(
                "array.contains expects an array, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_array_map_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("array.map requires an array and a function".to_string());
        }
        match &args[0] {
            Value::Array(arr) => {
                let func = &args[1];
                let borrowed = arr.borrow().clone();
                let mut mapped = Vec::with_capacity(borrowed.len());
                for item in borrowed {
                    let res = VM::with_current(|vm| {
                        vm.call_value(func, vec![item]).map_err(|e| e.to_string())
                    })?;
                    mapped.push(unwrap_first_return(res));
                }
                Ok(NativeCallResult::Return(Value::array(mapped)))
            }
            other => Err(format!(
                "array.map expects an array, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_array_filter_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("array.filter requires an array and a predicate function".to_string());
        }
        match &args[0] {
            Value::Array(arr) => {
                let func = &args[1];
                let borrowed = arr.borrow().clone();
                let mut filtered = Vec::new();
                for item in borrowed {
                    let res = VM::with_current(|vm| {
                        vm.call_value(func, vec![item.clone()])
                            .map_err(|e| e.to_string())
                    })?;
                    let keep = unwrap_first_return(res);
                    if keep.is_truthy() {
                        filtered.push(item);
                    }
                }
                Ok(NativeCallResult::Return(Value::array(filtered)))
            }
            other => Err(format!(
                "array.filter expects an array, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_array_reduce_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 3 {
            return Err(
                "array.reduce requires array, initial value, and a reducer function".to_string(),
            );
        }
        match &args[0] {
            Value::Array(arr) => {
                let mut acc = args[1].clone();
                let func = &args[2];
                let borrowed = arr.borrow().clone();
                for item in borrowed {
                    let res = VM::with_current(|vm| {
                        vm.call_value(func, vec![acc.clone(), item])
                            .map_err(|e| e.to_string())
                    })?;
                    acc = unwrap_first_return(res);
                }
                Ok(NativeCallResult::Return(acc))
            }
            other => Err(format!(
                "array.reduce expects an array, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_array_iter_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = args.first().cloned().unwrap_or(Value::Nil);
        match val {
            Value::Array(arr) => {
                let items = arr.borrow().clone();
                let iter = crate::bytecode::value::IteratorState::Array { items, index: 0 };
                Ok(NativeCallResult::Return(Value::Iterator(Rc::new(
                    std::cell::RefCell::new(iter),
                ))))
            }
            other => Err(format!(
                "array.iter expects an array, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_table_insert_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("table.insert requires at least 2 arguments".to_string());
        }
        let target = unwrap_lua_value(args[0].clone());
        if args.len() == 2 {
            // table.insert(arr, val) -> behaves as push
            let val = args[1].clone();
            match target {
                Value::Array(arr) => {
                    let mut borrowed = arr.borrow_mut();
                    let len = borrowed.len();
                    let cap = borrowed.capacity();
                    if len == cap {
                        let new_cap = if cap == 0 { 4 } else { cap.saturating_mul(2) };
                        VM::with_current(|vm| {
                            vm.charge_vec_growth::<Value>(cap, new_cap)
                                .map_err(|e| e.to_string())
                        })?;
                    }
                    if !crate::vm::cycle::CycleCollector::is_leaf(&val) {
                        let _ = VM::with_current(|vm| {
                            vm.cycle_collector.record_mutation();
                            Ok::<(), String>(())
                        });
                    }
                    borrowed.push(val);
                    crate::bytecode::value::invalidate_specialized_array_cache(arr.as_ptr() as usize);
                    Ok(NativeCallResult::Return(Value::Nil))
                }
                other => Err(format!(
                    "table.insert expects an array as first argument, got {:?}",
                    other.type_of()
                )),
            }
        } else {
            // table.insert(arr, idx, val) -> behaves as insert
            let index = args[1]
                .as_int()
                .ok_or_else(|| "table.insert expects an integer index when called with 3 arguments".to_string())?;
            let val = args[2].clone();
            match target {
                Value::Array(arr) => {
                    let mut borrowed = arr.borrow_mut();
                    let len = borrowed.len();
                    let cap = borrowed.capacity();
                    if len == cap {
                        let new_cap = if cap == 0 { 4 } else { cap.saturating_mul(2) };
                        VM::with_current(|vm| {
                            vm.charge_vec_growth::<Value>(cap, new_cap)
                                .map_err(|e| e.to_string())
                        })?;
                    }
                    if !crate::vm::cycle::CycleCollector::is_leaf(&val) {
                        let _ = VM::with_current(|vm| {
                            vm.cycle_collector.record_mutation();
                            Ok::<(), String>(())
                        });
                    }
                    let idx = (index.max(0) as usize).min(borrowed.len());
                    borrowed.insert(idx, val);
                    crate::bytecode::value::invalidate_specialized_array_cache(arr.as_ptr() as usize);
                    Ok(NativeCallResult::Return(Value::Nil))
                }
                other => Err(format!(
                    "table.insert expects an array as first argument, got {:?}",
                    other.type_of()
                )),
            }
        }
    }))
}

fn create_table_remove_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("table.remove requires at least 1 argument".to_string());
        }
        let target = unwrap_lua_value(args[0].clone());
        if args.len() == 1 {
            // table.remove(arr) -> behaves as pop
            match target {
                Value::Array(arr) => {
                    crate::bytecode::value::invalidate_specialized_array_cache(arr.as_ptr() as usize);
                    let popped = arr.borrow_mut().pop();
                    Ok(NativeCallResult::Return(
                        popped.map(Value::some).unwrap_or_else(Value::none),
                    ))
                }
                other => Err(format!(
                    "table.remove expects an array as first argument, got {:?}",
                    other.type_of()
                )),
            }
        } else {
            // table.remove(arr, idx) -> behaves as remove
            let index = args[1]
                .as_int()
                .ok_or_else(|| "table.remove expects an integer index when called with 2 arguments".to_string())?;
            match target {
                Value::Array(arr) => {
                    crate::bytecode::value::invalidate_specialized_array_cache(arr.as_ptr() as usize);
                    let mut borrowed = arr.borrow_mut();
                    if index >= 0 && (index as usize) < borrowed.len() {
                        let val = borrowed.remove(index as usize);
                        Ok(NativeCallResult::Return(Value::some(val)))
                    } else {
                        Ok(NativeCallResult::Return(Value::none()))
                    }
                }
                other => Err(format!(
                    "table.remove expects an array as first argument, got {:?}",
                    other.type_of()
                )),
            }
        }
    }))
}

fn create_table_module(vm: &VM) -> Value {
    let entries = [
        (string_key("len"), create_array_len_fn()),
        (string_key("is_empty"), create_array_is_empty_fn()),
        (string_key("get"), create_array_get_fn()),
        (string_key("first"), create_array_first_fn()),
        (string_key("last"), create_array_last_fn()),
        (string_key("push"), create_array_push_fn()),
        (string_key("pop"), create_array_pop_fn()),
        (string_key("insert"), create_table_insert_fn()),
        (string_key("remove"), create_table_remove_fn()),
        (string_key("clear"), create_array_clear_fn()),
        (string_key("slice"), create_array_slice_fn()),
        (string_key("concat"), create_array_concat_fn()),
        (string_key("sort"), create_array_sort_fn()),
        (string_key("reverse"), create_array_reverse_fn()),
        (string_key("contains"), create_array_contains_fn()),
        (string_key("map"), create_array_map_fn()),
        (string_key("filter"), create_array_filter_fn()),
        (string_key("reduce"), create_array_reduce_fn()),
        (string_key("iter"), create_array_iter_fn()),
        (string_key("unpack"), create_table_unpack_fn()),
    ];
    vm.map_with_entries(entries)
}

fn create_effect_module(vm: &VM) -> Value {
    let entries = [
        (string_key("succeed"), create_effect_succeed_fn()),
        (string_key("fail"), create_effect_fail_fn()),
        (string_key("die"), create_effect_die_fn()),
        (string_key("interrupt"), create_effect_interrupt_fn()),
        (string_key("from_result"), create_effect_from_result_fn()),
        (string_key("run_sync"), create_effect_run_sync_fn()),
        (string_key("suspend"), create_effect_suspend_fn()),
    ];
    vm.map_with_entries(entries)
}

fn create_effect_succeed_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = args.first().cloned().unwrap_or(Value::Nil);
        let entries = vec![
            (string_key("__effect"), Value::Bool(true)),
            (string_key("__kind"), Value::string("succeed")),
            (string_key("__value"), val),
        ];
        Ok(NativeCallResult::Return(Value::map(entries.into_iter().collect())))
    }))
}

fn create_effect_fail_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let err = args.first().cloned().unwrap_or(Value::Nil);
        let entries = vec![
            (string_key("__effect"), Value::Bool(true)),
            (string_key("__kind"), Value::string("fail")),
            (string_key("__error"), err),
        ];
        Ok(NativeCallResult::Return(Value::map(entries.into_iter().collect())))
    }))
}

fn create_effect_die_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let msg = args.first().cloned().unwrap_or(Value::string("defect"));
        let entries = vec![
            (string_key("__effect"), Value::Bool(true)),
            (string_key("__kind"), Value::string("die")),
            (string_key("__defect"), msg),
        ];
        Ok(NativeCallResult::Return(Value::map(entries.into_iter().collect())))
    }))
}

fn create_effect_interrupt_fn() -> Value {
    Value::NativeFunction(native_fn(|_args: &[Value]| {
        let entries = vec![
            (string_key("__effect"), Value::Bool(true)),
            (string_key("__kind"), Value::string("interrupt")),
        ];
        Ok(NativeCallResult::Return(Value::map(entries.into_iter().collect())))
    }))
}

fn create_effect_from_result_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let res = args.first().cloned().unwrap_or(Value::Nil);
        if let Value::Enum(e) = &res {
            if e.variant.as_str() == "Ok" {
                let val = e.values.as_ref().and_then(|v| v.first()).cloned().unwrap_or(Value::Nil);
                return Ok(NativeCallResult::Return(Value::map(vec![
                    (string_key("__effect"), Value::Bool(true)),
                    (string_key("__kind"), Value::string("succeed")),
                    (string_key("__value"), val),
                ].into_iter().collect())));
            } else if e.variant.as_str() == "Err" {
                let err = e.values.as_ref().and_then(|v| v.first()).cloned().unwrap_or(Value::Nil);
                return Ok(NativeCallResult::Return(Value::map(vec![
                    (string_key("__effect"), Value::Bool(true)),
                    (string_key("__kind"), Value::string("fail")),
                    (string_key("__error"), err),
                ].into_iter().collect())));
            }
        }
        Ok(NativeCallResult::Return(Value::map(vec![
            (string_key("__effect"), Value::Bool(true)),
            (string_key("__kind"), Value::string("succeed")),
            (string_key("__value"), res),
        ].into_iter().collect())))
    }))
}

fn create_effect_suspend_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("effect.suspend requires a function/thunk".to_string());
        }
        let thunk = args[0].clone();
        let entries = vec![
            (string_key("__effect"), Value::Bool(true)),
            (string_key("__kind"), Value::string("suspend")),
            (string_key("__thunk"), thunk),
        ];
        Ok(NativeCallResult::Return(Value::map(entries.into_iter().collect())))
    }))
}

fn create_effect_run_sync_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("effect.run_sync requires an effect argument".to_string());
        }
        let eff = &args[0];
        match eff {
            Value::Closure(_) | Value::Function(_) => {
                let res = VM::with_current(|vm| {
                    vm.call_value(eff, vec![]).map_err(|e| e.to_string())
                });
                match res {
                    Ok(val) => {
                        if let Value::Enum(e) = &val {
                            if e.enum_name == "Exit" {
                                return Ok(NativeCallResult::Return(val));
                            }
                        }
                        Ok(NativeCallResult::Return(Value::enum_variant(
                            "Exit",
                            "Success",
                            vec![val],
                        )))
                    }
                    Err(e) => {
                        if e.to_lowercase().contains("out of gas") || e.contains("Interrupt") {
                            Ok(NativeCallResult::Return(Value::enum_variant(
                                "Exit",
                                "Interrupt",
                                vec![],
                            )))
                        } else {
                            Ok(NativeCallResult::Return(Value::enum_variant(
                                "Exit",
                                "Fail",
                                vec![Value::string(e)],
                            )))
                        }
                    }
                }
            }
            Value::Map(m) => {
                let borrowed = m.borrow();
                let kind = borrowed.get(&string_key("__kind"));
                match kind {
                    Some(Value::String(s)) if s.as_str() == "succeed" => {
                        let val = borrowed.get(&string_key("__value")).cloned().unwrap_or(Value::Nil);
                        Ok(NativeCallResult::Return(Value::enum_variant(
                            "Exit",
                            "Success",
                            vec![val],
                        )))
                    }
                    Some(Value::String(s)) if s.as_str() == "fail" => {
                        let err = borrowed.get(&string_key("__error")).cloned().unwrap_or(Value::Nil);
                        Ok(NativeCallResult::Return(Value::enum_variant(
                            "Exit",
                            "Fail",
                            vec![err],
                        )))
                    }
                    Some(Value::String(s)) if s.as_str() == "die" => {
                        let msg = borrowed.get(&string_key("__defect")).cloned().unwrap_or(Value::string("defect"));
                        Ok(NativeCallResult::Return(Value::enum_variant(
                            "Exit",
                            "Die",
                            vec![msg],
                        )))
                    }
                    Some(Value::String(s)) if s.as_str() == "interrupt" => {
                        Ok(NativeCallResult::Return(Value::enum_variant(
                            "Exit",
                            "Interrupt",
                            vec![],
                        )))
                    }
                    Some(Value::String(s)) if s.as_str() == "suspend" => {
                        if let Some(thunk) = borrowed.get(&string_key("__thunk")).cloned() {
                            let res = VM::with_current(|vm| {
                                vm.call_value(&thunk, vec![]).map_err(|e| e.to_string())
                            });
                            match res {
                                Ok(val) => {
                                    if let Value::Enum(e) = &val {
                                        if e.enum_name == "Exit" {
                                            return Ok(NativeCallResult::Return(val));
                                        }
                                    }
                                    Ok(NativeCallResult::Return(Value::enum_variant(
                                        "Exit",
                                        "Success",
                                        vec![val],
                                    )))
                                }
                                Err(e) => {
                                    if e.to_lowercase().contains("out of gas") || e.contains("Interrupt") {
                                        Ok(NativeCallResult::Return(Value::enum_variant(
                                            "Exit",
                                            "Interrupt",
                                            vec![],
                                        )))
                                    } else {
                                        Ok(NativeCallResult::Return(Value::enum_variant(
                                            "Exit",
                                            "Fail",
                                            vec![Value::string(e)],
                                        )))
                                    }
                                }
                            }
                        } else {
                            Err("effect.run_sync: invalid suspend effect without __thunk".to_string())
                        }
                    }
                    _ => {
                        Ok(NativeCallResult::Return(Value::enum_variant(
                            "Exit",
                            "Success",
                            vec![eff.clone()],
                        )))
                    }
                }
            }
            Value::Enum(e) if e.enum_name == "Exit" => {
                Ok(NativeCallResult::Return(eff.clone()))
            }
            other => {
                Ok(NativeCallResult::Return(Value::enum_variant(
                    "Exit",
                    "Success",
                    vec![other.clone()],
                )))
            }
        }
    }))
}

fn create_map_module(vm: &VM) -> Value {
    let entries = [
        (string_key("len"), create_map_len_fn()),
        (string_key("is_empty"), create_map_is_empty_fn()),
        (string_key("get"), create_map_get_fn()),
        (string_key("set"), create_map_set_fn()),
        (string_key("has"), create_map_has_fn()),
        (string_key("delete"), create_map_delete_fn()),
        (string_key("clear"), create_map_clear_fn()),
        (string_key("keys"), create_map_keys_fn()),
        (string_key("values"), create_map_values_fn()),
        (string_key("iter"), create_map_iter_fn()),
    ];
    vm.map_with_entries(entries)
}

fn create_map_len_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = args.first().cloned().unwrap_or(Value::Nil);
        match val {
            Value::Map(m) => Ok(NativeCallResult::Return(Value::Int(
                m.borrow().len() as LustInt
            ))),
            other => Err(format!("map.len expects a map, got {:?}", other.type_of())),
        }
    }))
}

fn create_map_is_empty_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = args.first().cloned().unwrap_or(Value::Nil);
        match val {
            Value::Map(m) => Ok(NativeCallResult::Return(Value::Bool(m.borrow().is_empty()))),
            other => Err(format!(
                "map.is_empty expects a map, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_map_get_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("map.get requires map and key".to_string());
        }
        match &args[0] {
            Value::Map(m) => {
                let key = ValueKey::from_value(&args[1]);
                let borrowed = m.borrow();
                match borrowed.get(&key) {
                    Some(v) => Ok(NativeCallResult::Return(Value::some(v.clone()))),
                    None => Ok(NativeCallResult::Return(Value::none())),
                }
            }
            other => Err(format!("map.get expects a map, got {:?}", other.type_of())),
        }
    }))
}

fn create_map_set_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 3 {
            return Err("map.set requires map, key, and value".to_string());
        }
        match &args[0] {
            Value::Map(m) => {
                let key = ValueKey::from_value(&args[1]);
                let mut borrowed = m.borrow_mut();
                if !borrowed.contains_key(&key) {
                    VM::with_current(|vm| {
                        vm.budgets
                            .charge_map_entry_estimate()
                            .map_err(|e| e.to_string())
                    })?;
                }
                if !crate::vm::cycle::CycleCollector::is_leaf(&args[2]) {
                    let _ = VM::with_current(|vm| {
                        vm.cycle_collector.record_mutation();
                        Ok::<(), String>(())
                    });
                }
                borrowed.insert(key, args[2].clone());
                Ok(NativeCallResult::Return(Value::Nil))
            }
            other => Err(format!("map.set expects a map, got {:?}", other.type_of())),
        }
    }))
}

fn create_map_has_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("map.has requires map and key".to_string());
        }
        match &args[0] {
            Value::Map(m) => {
                let key = ValueKey::from_value(&args[1]);
                let has = m.borrow().contains_key(&key);
                Ok(NativeCallResult::Return(Value::Bool(has)))
            }
            other => Err(format!("map.has expects a map, got {:?}", other.type_of())),
        }
    }))
}

fn create_map_delete_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("map.delete requires map and key".to_string());
        }
        match &args[0] {
            Value::Map(m) => {
                let key = ValueKey::from_value(&args[1]);
                let removed = m.borrow_mut().remove(&key);
                Ok(NativeCallResult::Return(
                    removed.map(Value::some).unwrap_or_else(Value::none),
                ))
            }
            other => Err(format!(
                "map.delete expects a map, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_map_clear_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = args.first().cloned().unwrap_or(Value::Nil);
        match val {
            Value::Map(m) => {
                m.borrow_mut().clear();
                Ok(NativeCallResult::Return(Value::Nil))
            }
            other => Err(format!(
                "map.clear expects a map, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_map_keys_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = args.first().cloned().unwrap_or(Value::Nil);
        match val {
            Value::Map(m) => {
                let keys: Vec<Value> = m.borrow().keys().map(|k| k.to_value()).collect();
                Ok(NativeCallResult::Return(Value::array(keys)))
            }
            other => Err(format!("map.keys expects a map, got {:?}", other.type_of())),
        }
    }))
}

fn create_map_values_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = args.first().cloned().unwrap_or(Value::Nil);
        match val {
            Value::Map(m) => {
                let values: Vec<Value> = m.borrow().values().cloned().collect();
                Ok(NativeCallResult::Return(Value::array(values)))
            }
            other => Err(format!(
                "map.values expects a map, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_map_iter_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = args.first().cloned().unwrap_or(Value::Nil);
        match val {
            Value::Map(m) => {
                let items: Vec<(ValueKey, Value)> = m
                    .borrow()
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                let iter = crate::bytecode::value::IteratorState::MapPairs { items, index: 0 };
                Ok(NativeCallResult::Return(Value::Iterator(Rc::new(
                    std::cell::RefCell::new(iter),
                ))))
            }
            other => Err(format!("map.iter expects a map, got {:?}", other.type_of())),
        }
    }))
}

fn create_math_module(vm: &VM) -> Value {
    let entries = [
        (string_key("pi"), Value::Float(std::f64::consts::PI)),
        (string_key("huge"), Value::Float(f64::INFINITY)),
        (string_key("maxinteger"), Value::Int(LustInt::MAX)),
        (string_key("mininteger"), Value::Int(LustInt::MIN)),
        (string_key("abs"), create_math_abs_fn()),
        (string_key("floor"), create_math_floor_fn()),
        (string_key("ceil"), create_math_ceil_fn()),
        (string_key("round"), create_math_round_fn()),
        (string_key("sqrt"), create_math_sqrt_fn()),
        (string_key("sin"), create_math_sin_fn()),
        (string_key("cos"), create_math_cos_fn()),
        (string_key("tan"), create_math_tan_fn()),
        (string_key("asin"), create_math_asin_fn()),
        (string_key("acos"), create_math_acos_fn()),
        (string_key("atan"), create_math_atan_fn()),
        (string_key("atan2"), create_math_atan2_fn()),
        (string_key("min"), create_math_min_fn()),
        (string_key("max"), create_math_max_fn()),
        (string_key("clamp"), create_math_clamp_fn()),
        (string_key("random"), create_math_random_fn()),
        (string_key("randomseed"), create_math_randomseed_fn()),
        (string_key("deg"), create_math_deg_fn()),
        (string_key("rad"), create_math_rad_fn()),
        (string_key("exp"), create_math_exp_fn()),
        (string_key("log"), create_math_log_fn()),
        (string_key("tointeger"), create_math_tointeger_fn()),
        (string_key("tofloat"), create_math_tofloat_fn()),
        (string_key("mod"), create_math_mod_fn()),
        (string_key("pow"), create_math_pow_fn()),
    ];
    vm.map_with_entries(entries)
}

fn create_math_abs_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        match val {
            Value::Int(i) => Ok(NativeCallResult::Return(Value::Int(i.wrapping_abs()))),
            Value::Float(f) => Ok(NativeCallResult::Return(Value::Float(f.abs()))),
            other => Err(format!(
                "math.abs expects a number, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_math_floor_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        match val {
            Value::Int(i) => Ok(NativeCallResult::Return(Value::Int(i))),
            Value::Float(f) => Ok(NativeCallResult::Return(Value::Int(f.floor() as LustInt))),
            other => Err(format!(
                "math.floor expects a number, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_math_ceil_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        match val {
            Value::Int(i) => Ok(NativeCallResult::Return(Value::Int(i))),
            Value::Float(f) => Ok(NativeCallResult::Return(Value::Int(f.ceil() as LustInt))),
            other => Err(format!(
                "math.ceil expects a number, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_math_round_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        match val {
            Value::Int(i) => Ok(NativeCallResult::Return(Value::Int(i))),
            Value::Float(f) => Ok(NativeCallResult::Return(Value::Float(f.round()))),
            other => Err(format!(
                "math.round expects a number, got {:?}",
                other.type_of()
            )),
        }
    }))
}

fn create_math_sqrt_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let num = val
            .as_float()
            .or_else(|| val.as_int().map(|i| i as f64))
            .ok_or_else(|| "math.sqrt expects a number".to_string())?;
        Ok(NativeCallResult::Return(Value::Float(num.sqrt())))
    }))
}

fn create_math_sin_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let num = val
            .as_float()
            .or_else(|| val.as_int().map(|i| i as f64))
            .ok_or_else(|| "math.sin expects a number".to_string())?;
        Ok(NativeCallResult::Return(Value::Float(num.sin())))
    }))
}

fn create_math_cos_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let num = val
            .as_float()
            .or_else(|| val.as_int().map(|i| i as f64))
            .ok_or_else(|| "math.cos expects a number".to_string())?;
        Ok(NativeCallResult::Return(Value::Float(num.cos())))
    }))
}

fn create_math_tan_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let num = val
            .as_float()
            .or_else(|| val.as_int().map(|i| i as f64))
            .ok_or_else(|| "math.tan expects a number".to_string())?;
        Ok(NativeCallResult::Return(Value::Float(num.tan())))
    }))
}

fn create_math_asin_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let num = val
            .as_float()
            .or_else(|| val.as_int().map(|i| i as f64))
            .ok_or_else(|| "math.asin expects a number".to_string())?;
        Ok(NativeCallResult::Return(Value::Float(num.asin())))
    }))
}

fn create_math_acos_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let num = val
            .as_float()
            .or_else(|| val.as_int().map(|i| i as f64))
            .ok_or_else(|| "math.acos expects a number".to_string())?;
        Ok(NativeCallResult::Return(Value::Float(num.acos())))
    }))
}

fn create_math_atan_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let num = val
            .as_float()
            .or_else(|| val.as_int().map(|i| i as f64))
            .ok_or_else(|| "math.atan expects a number".to_string())?;
        Ok(NativeCallResult::Return(Value::Float(num.atan())))
    }))
}

fn create_math_atan2_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("math.atan2 requires y and x arguments".to_string());
        }
        let y_val = unwrap_lua_value(args[0].clone());
        let x_val = unwrap_lua_value(args[1].clone());
        let y = y_val
            .as_float()
            .or_else(|| y_val.as_int().map(|i| i as f64))
            .ok_or_else(|| "math.atan2 expects number arguments".to_string())?;
        let x = x_val
            .as_float()
            .or_else(|| x_val.as_int().map(|i| i as f64))
            .ok_or_else(|| "math.atan2 expects number arguments".to_string())?;
        Ok(NativeCallResult::Return(Value::Float(y.atan2(x))))
    }))
}

fn create_math_min_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("math.min requires at least one argument".to_string());
        }
        let mut min_val = unwrap_lua_value(args[0].clone());
        for arg in &args[1..] {
            let next = unwrap_lua_value(arg.clone());
            match (&min_val, &next) {
                (Value::Int(a), Value::Int(b)) => {
                    if b < a {
                        min_val = Value::Int(*b);
                    }
                }
                (Value::Float(a), Value::Float(b)) => {
                    if b < a {
                        min_val = Value::Float(*b);
                    }
                }
                (Value::Int(a), Value::Float(b)) => {
                    if *b < (*a as f64) {
                        min_val = Value::Float(*b);
                    }
                }
                (Value::Float(a), Value::Int(b)) => {
                    if (*b as f64) < *a {
                        min_val = Value::Int(*b);
                    }
                }
                _ => return Err("math.min expects numbers".to_string()),
            }
        }
        Ok(NativeCallResult::Return(min_val))
    }))
}

fn create_math_max_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("math.max requires at least one argument".to_string());
        }
        let mut max_val = unwrap_lua_value(args[0].clone());
        for arg in &args[1..] {
            let next = unwrap_lua_value(arg.clone());
            match (&max_val, &next) {
                (Value::Int(a), Value::Int(b)) => {
                    if b > a {
                        max_val = Value::Int(*b);
                    }
                }
                (Value::Float(a), Value::Float(b)) => {
                    if b > a {
                        max_val = Value::Float(*b);
                    }
                }
                (Value::Int(a), Value::Float(b)) => {
                    if *b > (*a as f64) {
                        max_val = Value::Float(*b);
                    }
                }
                (Value::Float(a), Value::Int(b)) => {
                    if (*b as f64) > *a {
                        max_val = Value::Int(*b);
                    }
                }
                _ => return Err("math.max expects numbers".to_string()),
            }
        }
        Ok(NativeCallResult::Return(max_val))
    }))
}

fn create_math_clamp_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 3 {
            return Err("math.clamp requires x, min, and max arguments".to_string());
        }
        let x = unwrap_lua_value(args[0].clone());
        let min = unwrap_lua_value(args[1].clone());
        let max = unwrap_lua_value(args[2].clone());
        match (&x, &min, &max) {
            (Value::Int(xi), Value::Int(mini), Value::Int(maxi)) => Ok(NativeCallResult::Return(
                Value::Int((*xi).max(*mini).min(*maxi)),
            )),
            _ => {
                let xf = x
                    .as_float()
                    .or_else(|| x.as_int().map(|i| i as f64))
                    .ok_or_else(|| "math.clamp expects numbers".to_string())?;
                let minf = min
                    .as_float()
                    .or_else(|| min.as_int().map(|i| i as f64))
                    .ok_or_else(|| "math.clamp expects numbers".to_string())?;
                let maxf = max
                    .as_float()
                    .or_else(|| max.as_int().map(|i| i as f64))
                    .ok_or_else(|| "math.clamp expects numbers".to_string())?;
                Ok(NativeCallResult::Return(Value::Float(
                    xf.max(minf).min(maxf),
                )))
            }
        }
    }))
}

fn create_math_deg_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let num = val
            .as_float()
            .or_else(|| val.as_int().map(|i| i as f64))
            .ok_or_else(|| "math.deg expects a number".to_string())?;
        Ok(NativeCallResult::Return(Value::Float(num.to_degrees())))
    }))
}

fn create_math_rad_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let num = val
            .as_float()
            .or_else(|| val.as_int().map(|i| i as f64))
            .ok_or_else(|| "math.rad expects a number".to_string())?;
        Ok(NativeCallResult::Return(Value::Float(num.to_radians())))
    }))
}

fn create_math_exp_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let num = val
            .as_float()
            .or_else(|| val.as_int().map(|i| i as f64))
            .ok_or_else(|| "math.exp expects a number".to_string())?;
        Ok(NativeCallResult::Return(Value::Float(num.exp())))
    }))
}

fn create_math_log_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let num = val
            .as_float()
            .or_else(|| val.as_int().map(|i| i as f64))
            .ok_or_else(|| "math.log expects a number".to_string())?;
        let base = args
            .get(1)
            .map(|v| unwrap_lua_value(v.clone()))
            .and_then(|v| v.as_float().or_else(|| v.as_int().map(|i| i as f64)));
        let result = match base {
            Some(b) => num.log(b),
            None => num.ln(),
        };
        Ok(NativeCallResult::Return(Value::Float(result)))
    }))
}

fn create_math_pow_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("math.pow requires base and exponent arguments".to_string());
        }
        let base_val = unwrap_lua_value(args[0].clone());
        let exp_val = unwrap_lua_value(args[1].clone());
        let base = base_val
            .as_float()
            .or_else(|| base_val.as_int().map(|i| i as f64))
            .ok_or_else(|| "math.pow expects number arguments".to_string())?;
        let exp = exp_val
            .as_float()
            .or_else(|| exp_val.as_int().map(|i| i as f64))
            .ok_or_else(|| "math.pow expects number arguments".to_string())?;
        Ok(NativeCallResult::Return(Value::Float(base.powf(exp))))
    }))
}

fn create_bit_band_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("bit.band requires at least one argument".to_string());
        }
        let mut acc: LustInt = -1;
        for (idx, arg) in args.iter().enumerate() {
            let val = unwrap_lua_value(arg.clone());
            let n = coerce_int(&val).ok_or_else(|| {
                format!("bit.band expects integer arguments, got arg {}", idx + 1)
            })?;
            acc &= n;
        }
        Ok(NativeCallResult::Return(Value::Int(acc)))
    }))
}

fn create_bit_bor_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("bit.bor requires at least one argument".to_string());
        }
        let mut acc: LustInt = 0;
        for (idx, arg) in args.iter().enumerate() {
            let val = unwrap_lua_value(arg.clone());
            let n = coerce_int(&val).ok_or_else(|| {
                format!("bit.bor expects integer arguments, got arg {}", idx + 1)
            })?;
            acc |= n;
        }
        Ok(NativeCallResult::Return(Value::Int(acc)))
    }))
}

fn create_bit_bxor_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("bit.bxor requires at least one argument".to_string());
        }
        let mut acc: LustInt = 0;
        for (idx, arg) in args.iter().enumerate() {
            let val = unwrap_lua_value(arg.clone());
            let n = coerce_int(&val).ok_or_else(|| {
                format!("bit.bxor expects integer arguments, got arg {}", idx + 1)
            })?;
            acc ^= n;
        }
        Ok(NativeCallResult::Return(Value::Int(acc)))
    }))
}

fn create_bit_bnot_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        let n = coerce_int(&val)
            .ok_or_else(|| "bit.bnot expects an integer argument".to_string())?;
        Ok(NativeCallResult::Return(Value::Int(!n)))
    }))
}

fn create_bit_lshift_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("bit.lshift requires value and shift count".to_string());
        }
        let a = coerce_int(&unwrap_lua_value(args[0].clone()))
            .ok_or_else(|| "bit.lshift expects integer value".to_string())?;
        let n = coerce_int(&unwrap_lua_value(args[1].clone()))
            .ok_or_else(|| "bit.lshift expects integer shift count".to_string())?;
        let shift = (n & 63) as u32;
        Ok(NativeCallResult::Return(Value::Int(a.wrapping_shl(shift))))
    }))
}

fn create_bit_rshift_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("bit.rshift requires value and shift count".to_string());
        }
        let a = coerce_int(&unwrap_lua_value(args[0].clone()))
            .ok_or_else(|| "bit.rshift expects integer value".to_string())?;
        let n = coerce_int(&unwrap_lua_value(args[1].clone()))
            .ok_or_else(|| "bit.rshift expects integer shift count".to_string())?;
        let shift = (n & 63) as u32;
        let unsigned = a as u64;
        Ok(NativeCallResult::Return(Value::Int((unsigned.wrapping_shr(shift)) as LustInt)))
    }))
}

fn create_bit_arshift_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("bit.arshift requires value and shift count".to_string());
        }
        let a = coerce_int(&unwrap_lua_value(args[0].clone()))
            .ok_or_else(|| "bit.arshift expects integer value".to_string())?;
        let n = coerce_int(&unwrap_lua_value(args[1].clone()))
            .ok_or_else(|| "bit.arshift expects integer shift count".to_string())?;
        let shift = (n & 63) as u32;
        Ok(NativeCallResult::Return(Value::Int(a.wrapping_shr(shift))))
    }))
}

fn create_bit_module(vm: &VM) -> Value {
    let entries = [
        (string_key("band"), create_bit_band_fn()),
        (string_key("bor"), create_bit_bor_fn()),
        (string_key("bxor"), create_bit_bxor_fn()),
        (string_key("bnot"), create_bit_bnot_fn()),
        (string_key("lshift"), create_bit_lshift_fn()),
        (string_key("rshift"), create_bit_rshift_fn()),
        (string_key("arshift"), create_bit_arshift_fn()),
    ];
    vm.map_with_entries(entries)
}

fn create_math_tointeger_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        match val {
            Value::Int(i) => Ok(NativeCallResult::Return(Value::some(Value::Int(i)))),
            Value::Float(f) => {
                if f.fract() == 0.0 && f >= (LustInt::MIN as f64) && f <= (LustInt::MAX as f64) {
                    Ok(NativeCallResult::Return(Value::some(Value::Int(
                        f as LustInt,
                    ))))
                } else {
                    Ok(NativeCallResult::Return(Value::none()))
                }
            }
            Value::String(s) => {
                if let Ok(i) = s.trim().parse::<LustInt>() {
                    Ok(NativeCallResult::Return(Value::some(Value::Int(i))))
                } else {
                    Ok(NativeCallResult::Return(Value::none()))
                }
            }
            _ => Ok(NativeCallResult::Return(Value::none())),
        }
    }))
}

fn create_math_tofloat_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let val = unwrap_lua_value(args.first().cloned().unwrap_or(Value::Nil));
        match val {
            Value::Float(f) => Ok(NativeCallResult::Return(Value::some(Value::Float(f)))),
            Value::Int(i) => Ok(NativeCallResult::Return(Value::some(Value::Float(
                i as f64,
            )))),
            Value::String(s) => {
                if let Ok(f) = s.trim().parse::<f64>() {
                    Ok(NativeCallResult::Return(Value::some(Value::Float(f))))
                } else {
                    Ok(NativeCallResult::Return(Value::none()))
                }
            }
            _ => Ok(NativeCallResult::Return(Value::none())),
        }
    }))
}

fn create_math_mod_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("math.mod expects 2 integer arguments".to_string());
        }
        let a = unwrap_lua_value(args[0].clone())
            .as_int()
            .ok_or_else(|| "math.mod expects an integer as first argument".to_string())?;
        let b = unwrap_lua_value(args[1].clone())
            .as_int()
            .ok_or_else(|| "math.mod expects an integer as second argument".to_string())?;
        if b == 0 {
            return Err("math.mod: division by zero".to_string());
        }
        if b == -1 {
            return Ok(NativeCallResult::Return(Value::Int(0)));
        }
        let r = a % b;
        let res = if (r > 0 && b < 0) || (r < 0 && b > 0) {
            r + b
        } else {
            r
        };
        Ok(NativeCallResult::Return(Value::Int(res)))
    }))
}

#[derive(Clone)]
enum TableData {
    Array(Value),
    Map(Value),
}

fn table_data(value: &Value) -> Option<TableData> {
    if value.as_array().is_some() {
        return Some(TableData::Array(value.clone()));
    }

    if value.as_map().is_some() {
        return Some(TableData::Map(value.clone()));
    }

    if let Some(map) = value.struct_get_field("table")
        && map.as_map().is_some()
    {
        return Some(TableData::Map(map));
    }

    None
}

fn read_sequence(data: &TableData) -> Vec<Value> {
    match data {
        TableData::Array(val) => val.as_array().unwrap_or_default(),
        TableData::Map(val) => {
            let map = val.as_map().unwrap_or_default();
            let mut seq: Vec<Value> = Vec::new();
            let mut idx: LustInt = 1;
            loop {
                let key = ValueKey::from_value(&Value::Int(idx));
                if let Some(val) = map.get(&key) {
                    seq.push(val.clone());
                    idx += 1;
                } else {
                    break;
                }
            }
            seq
        }
    }
}

pub(crate) fn create_table_unpack_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("table.unpack expects a table/array".to_string());
        }
        let table_val = unwrap_lua_value(args[0].clone());
        let Some(data) = table_data(&table_val) else {
            return Err("table.unpack expects a table/array".to_string());
        };
        let seq = read_sequence(&data);
        let start = args
            .get(1)
            .and_then(|v| unwrap_lua_value(v.clone()).as_int())
            .unwrap_or(1);
        let end = args
            .get(2)
            .and_then(|v| unwrap_lua_value(v.clone()).as_int())
            .unwrap_or(seq.len() as LustInt);
        let start_idx = (start - 1).max(0) as usize;
        let end_idx = end.max(0) as usize;
        let mut values: Vec<Value> = Vec::new();
        for (i, val) in seq.iter().enumerate() {
            if i < start_idx || i >= end_idx {
                continue;
            }
            values.push(val.clone());
        }
        return_lua_values(values)
    }))
}

fn create_math_random_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let lower = args
            .first()
            .map(|v| unwrap_lua_value(v.clone()))
            .filter(|v| !matches!(v, Value::Nil));
        let upper = args
            .get(1)
            .map(|v| unwrap_lua_value(v.clone()))
            .filter(|v| !matches!(v, Value::Nil));
        let value = with_rng_mut(|rng| match (lower.as_ref(), upper.as_ref()) {
            (None, _) => Value::Float(rng.random::<f64>()),
            (Some(max), None) => {
                let hi = coerce_int(max).unwrap_or(1);
                let upper_bound = if hi < 1 { 1 } else { hi };
                Value::Int(rng.random_range(1..=upper_bound))
            }
            (Some(min), Some(max)) => {
                let lo = coerce_int(min).unwrap_or(1);
                let hi = coerce_int(max).unwrap_or(lo);
                let (start, end) = if lo <= hi { (lo, hi) } else { (hi, lo) };
                Value::Int(rng.random_range(start..=end))
            }
        })?;
        Ok(NativeCallResult::Return(value))
    }))
}

thread_local! {
    static FALLBACK_RNG: std::cell::RefCell<StdRng> = std::cell::RefCell::new(make_rng());
}

fn create_math_randomseed_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let seed_val = args
            .first()
            .map(|v| unwrap_lua_value(v.clone()))
            .unwrap_or(Value::Int(0));
        let seed = coerce_int(&seed_val).unwrap_or(0) as u64;
        if let Some(vm_ptr) = crate::vm::current_vm_ptr() {
            let vm = unsafe { &*vm_ptr };
            vm.set_seed(seed);
        } else {
            FALLBACK_RNG.with(|rng| {
                *rng.borrow_mut() = StdRng::seed_from_u64(seed);
            });
        }
        Ok(NativeCallResult::Return(Value::Nil))
    }))
}

fn coerce_int(value: &Value) -> Option<LustInt> {
    match value {
        Value::Int(i) => Some(*i),
        Value::Float(f) => Some(*f as LustInt),
        Value::Bool(b) => Some(if *b { 1 } else { 0 }),
        _ => None,
    }
}

fn with_rng_mut<F, R>(f: F) -> Result<R, String>
where
    F: FnOnce(&mut dyn rand::Rng) -> R,
{
    if let Some(vm_ptr) = crate::vm::current_vm_ptr() {
        let vm = unsafe { &*vm_ptr };
        let mut rng = vm.rng.borrow_mut();
        Ok(f(&mut **rng))
    } else {
        FALLBACK_RNG.with(|rng| Ok(f(&mut *rng.borrow_mut())))
    }
}

fn render_format(fmt: &str, args: &[Value]) -> Result<String, String> {
    let mut out = String::new();
    let mut arg_idx = 0;
    let mut chars = fmt.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '%' {
            out.push(ch);
            continue;
        }
        if let Some('%') = chars.peek() {
            chars.next();
            out.push('%');
            continue;
        }
        let mut zero_pad = false;
        if let Some('0') = chars.peek() {
            zero_pad = true;
            chars.next();
        }
        let mut width_str = String::new();
        while let Some(next) = chars.peek() {
            if next.is_ascii_digit() {
                width_str.push(*next);
                chars.next();
            } else {
                break;
            }
        }
        let mut precision: Option<usize> = None;
        if let Some('.') = chars.peek() {
            chars.next();
            let mut prec = String::new();
            while let Some(next) = chars.peek() {
                if next.is_ascii_digit() {
                    prec.push(*next);
                    chars.next();
                } else {
                    break;
                }
            }
            if !prec.is_empty() {
                precision = prec.parse().ok();
            }
        }
        let spec = chars
            .next()
            .ok_or_else(|| "incomplete format specifier".to_string())?;
        let width = if width_str.is_empty() {
            None
        } else {
            width_str.parse().ok()
        };
        let arg = args.get(arg_idx).cloned().unwrap_or(Value::Nil);
        arg_idx += 1;
        let raw = unwrap_lua_value(arg);
        let formatted = match spec {
            's' => raw.to_string(),
            'd' | 'i' | 'u' => {
                let num = raw
                    .as_int()
                    .or_else(|| raw.as_float().map(|f| f as LustInt))
                    .unwrap_or(0);
                pad_value(format!("{}", num), width, zero_pad)
            }
            'x' => {
                let num = raw
                    .as_int()
                    .or_else(|| raw.as_float().map(|f| f as LustInt))
                    .unwrap_or(0);
                pad_value(format!("{:x}", num), width, zero_pad)
            }
            'X' => {
                let num = raw
                    .as_int()
                    .or_else(|| raw.as_float().map(|f| f as LustInt))
                    .unwrap_or(0);
                pad_value(format!("{:X}", num), width, zero_pad)
            }
            'c' => {
                let num = raw
                    .as_int()
                    .or_else(|| raw.as_float().map(|f| f as LustInt))
                    .unwrap_or(0);
                if let Some(ch) = char::from_u32(num as u32) {
                    ch.to_string()
                } else {
                    "".to_string()
                }
            }
            'f' | 'g' => {
                let num = raw
                    .as_float()
                    .or_else(|| raw.as_int().map(|i| i as f64))
                    .unwrap_or(0.0);
                if let Some(p) = precision {
                    pad_value(format!("{:.*}", p, num), width, zero_pad)
                } else {
                    pad_value(format!("{}", num), width, zero_pad)
                }
            }
            other => pad_value(format!("{}", other), width, zero_pad),
        };
        out.push_str(&formatted);
    }
    Ok(out)
}

fn pad_value(value: String, width: Option<usize>, zero_pad: bool) -> String {
    if let Some(w) = width
        && value.len() < w
    {
        let mut padded = String::new();
        let pad_char = if zero_pad { '0' } else { ' ' };
        for _ in 0..(w - value.len()) {
            padded.push(pad_char);
        }
        padded.push_str(&value);
        return padded;
    }
    value
}

fn normalize_range(start: LustInt, end: Option<LustInt>, len: usize) -> (usize, usize) {
    let len_i = len as LustInt;
    let mut s = if start < 0 { len_i + start + 1 } else { start };
    let mut e = end.unwrap_or(len_i);
    if e < 0 {
        e = len_i + e + 1;
    }
    if s < 1 {
        s = 1;
    }
    if e < 0 {
        e = 0;
    }
    if e > len_i {
        e = len_i;
    }
    if s > e {
        return (len, len);
    }
    (s.saturating_sub(1) as usize, e.max(0) as usize)
}

fn lua_pattern_to_regex(pattern: &str) -> Result<Regex, String> {
    fn is_regex_meta(ch: char) -> bool {
        matches!(
            ch,
            '.' | '+' | '*' | '?' | '^' | '$' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '\\'
        )
    }
    let mut out = String::new();
    let mut chars = pattern.chars().peekable();
    let mut in_class = false;
    while let Some(ch) = chars.next() {
        match ch {
            '%' => {
                if let Some(next) = chars.next() {
                    let translated = match next {
                        'a' => Some(if in_class { "A-Za-z" } else { "[A-Za-z]" }),
                        'c' => Some(if in_class { "\\p{Cc}" } else { "[\\p{Cc}]" }),
                        'd' => Some(if in_class { "0-9" } else { "[0-9]" }),
                        'l' => Some(if in_class { "a-z" } else { "[a-z]" }),
                        'u' => Some(if in_class { "A-Z" } else { "[A-Z]" }),
                        'w' => Some(if in_class {
                            "A-Za-z0-9_"
                        } else {
                            "[A-Za-z0-9_]"
                        }),
                        'x' => Some(if in_class { "A-Fa-f0-9" } else { "[A-Fa-f0-9]" }),
                        's' => Some(if in_class { "\\s" } else { "[\\s]" }),
                        'p' => Some(if in_class { "\\p{P}" } else { "[\\p{P}]" }),
                        'z' => Some(if in_class { "\\x00" } else { "[\\x00]" }),
                        '%' => Some("%"),
                        _ => None,
                    };
                    if let Some(rep) = translated {
                        out.push_str(rep);
                    } else {
                        if is_regex_meta(next) {
                            out.push('\\');
                        }
                        out.push(next);
                    }
                } else {
                    out.push('%');
                }
            }
            '[' => {
                in_class = true;
                out.push('[');
            }
            ']' => {
                in_class = false;
                out.push(']');
            }
            '.' | '+' | '*' | '?' => out.push(ch),
            '^' | '$' | '(' | ')' => out.push(ch),
            '{' | '}' | '|' | '\\' => {
                out.push('\\');
                out.push(ch);
            }
            '-' => {
                if in_class {
                    out.push('-');
                } else {
                    out.push_str("*?");
                }
            }
            other => {
                if !in_class && is_regex_meta(other) {
                    out.push('\\');
                }
                out.push(other);
            }
        }
    }
    Regex::new(&out).map_err(|e| e.to_string())
}

fn build_template_replacement(template: &str, caps: &regex::Captures) -> String {
    let mut out = String::new();
    let mut chars = template.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '%' {
            if let Some(next) = chars.next() {
                if next == '%' {
                    out.push('%');
                    continue;
                }
                if let Some(d) = next.to_digit(10) {
                    let idx = d as usize;
                    if idx == 0 {
                        if let Some(m) = caps.get(0) {
                            out.push_str(m.as_str());
                        }
                    } else if let Some(m) = caps.get(idx) {
                        out.push_str(m.as_str());
                    }
                    continue;
                }
                out.push(next);
            } else {
                out.push('%');
            }
        } else {
            out.push(ch);
        }
    }
    out
}

fn to_lua_value(vm: &VM, value: Value) -> Result<Value, String> {
    if let Value::Enum(object) = &value
        && object.enum_name == "LuaValue"
    {
        return Ok(value);
    }
    Ok(match value.clone() {
        Value::Nil => Value::enum_unit("LuaValue", "Nil"),
        Value::Bool(b) => Value::enum_variant("LuaValue", "Bool", vec![Value::Bool(b)]),
        Value::Int(i) => Value::enum_variant("LuaValue", "Int", vec![Value::Int(i)]),
        Value::Float(f) => Value::enum_variant("LuaValue", "Float", vec![Value::Float(f)]),
        Value::String(s) => Value::enum_variant("LuaValue", "String", vec![Value::String(s)]),
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
        #[cfg(not(target_arch = "wasm32"))]
        Value::Function(_) | Value::Closure { .. } | Value::NativeFunction(_) => {
            let handle = register_lust_function(value.clone());
            let lua_fn = vm
                .instantiate_struct(
                    "LuaFunction",
                    vec![(Rc::new("handle".to_string()), Value::Int(handle as LustInt))],
                )
                .map_err(|e| e.to_string())?;
            Value::enum_variant("LuaValue", "Function", vec![lua_fn])
        }
        other => Value::enum_variant(
            "LuaValue",
            "LightUserdata",
            vec![Value::Int(other.type_of() as LustInt)],
        ),
    })
}

fn return_lua_values(values: Vec<Value>) -> Result<NativeCallResult, String> {
    VM::with_current(|vm| pack_lua_values(vm, values).map(NativeCallResult::Return))
}

fn pack_lua_values(vm: &VM, values: Vec<Value>) -> Result<Value, String> {
    let mut packed = Vec::with_capacity(values.len());
    for value in values {
        packed.push(to_lua_value(vm, value)?);
    }
    Ok(Value::array(packed))
}

fn unwrap_first_return(value: Value) -> Value {
    if let Value::Array(arr) = value {
        if let Some(first) = arr.borrow().first() {
            return unwrap_lua_value(first.clone());
        }
        return Value::Nil;
    }
    unwrap_lua_value(value)
}

fn lua_nil() -> Value {
    Value::enum_unit("LuaValue", "Nil")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stdlib_defaults_without_optional_modules() {
        let vm = VM::with_config(&LustConfig::default());
        let stdlib = create_stdlib(&LustConfig::default(), &vm);
        assert!(!stdlib.iter().any(|(name, _)| *name == "io"));
        assert!(!stdlib.iter().any(|(name, _)| *name == "os"));
        assert!(stdlib.iter().any(|(name, _)| *name == "string"));
        assert!(stdlib.iter().any(|(name, _)| *name == "array"));
        assert!(stdlib.iter().any(|(name, _)| *name == "map"));
        assert!(stdlib.iter().any(|(name, _)| *name == "math"));
    }

    #[test]
    fn stdlib_includes_optional_modules_when_configured() {
        let cfg = LustConfig::from_toml_str(
            r#"
                [settings]
                stdlib_modules = ["io", "os"]
            "#,
        )
        .expect("parse");
        let vm = VM::with_config(&cfg);
        let stdlib = create_stdlib(&cfg, &vm);
        assert!(stdlib.iter().any(|(name, _)| *name == "io"));
        assert!(stdlib.iter().any(|(name, _)| *name == "os"));
        assert!(stdlib.iter().any(|(name, _)| *name == "string"));
        assert!(stdlib.iter().any(|(name, _)| *name == "table"));
    }

    #[test]
    fn test_table_insert_push_and_remove_pop() {
        let mut vm = VM::with_config(&LustConfig::default());
        let _guard = crate::vm::CurrentVmGuard::from_vm(&mut vm);

        let stdlib = create_stdlib(&LustConfig::default(), &vm);
        let table_mod = stdlib.iter().find(|(name, _)| *name == "table").unwrap().1.clone();

        let insert_fn = table_mod.map_get(&string_key("insert")).unwrap();
        let remove_fn = table_mod.map_get(&string_key("remove")).unwrap();
        let concat_fn = table_mod.map_get(&string_key("concat")).unwrap();

        let arr = Value::array(Vec::new());

        // table.insert(arr, 10) (push)
        let res1 = match &insert_fn {
            Value::NativeFunction(f) => f(&[arr.clone(), Value::Int(10)]).unwrap(),
            _ => panic!("expected native fn"),
        };
        assert!(matches!(res1, NativeCallResult::Return(Value::Nil)));

        // table.insert(arr, 20) (push)
        let _ = match &insert_fn {
            Value::NativeFunction(f) => f(&[arr.clone(), Value::Int(20)]).unwrap(),
            _ => panic!("expected native fn"),
        };

        // table.insert(arr, 1, 15) (insert at index 1)
        let _ = match &insert_fn {
            Value::NativeFunction(f) => f(&[arr.clone(), Value::Int(1), Value::Int(15)]).unwrap(),
            _ => panic!("expected native fn"),
        };

        // Check concat: "10, 15, 20"
        let concat_res = match &concat_fn {
            Value::NativeFunction(f) => f(&[arr.clone(), Value::string(", ")]).unwrap(),
            _ => panic!("expected native fn"),
        };
        match concat_res {
            NativeCallResult::Return(Value::String(s)) => assert_eq!(&*s, "10, 15, 20"),
            other => panic!("expected string, got {:?}", other),
        }

        // table.remove(arr) -> pop -> Some(20)
        let pop_res = match &remove_fn {
            Value::NativeFunction(f) => f(&[arr.clone()]).unwrap(),
            _ => panic!("expected native fn"),
        };
        match pop_res {
            NativeCallResult::Return(Value::Enum(e)) if e.variant.as_str() == "Some" => {
                assert_eq!(e.values.as_ref().unwrap().first(), Some(&Value::Int(20)));
            }
            other => panic!("expected Some(20), got {:?}", other),
        }

        // table.remove(arr, 0) -> remove at 0 -> Some(10)
        let remove_0_res = match &remove_fn {
            Value::NativeFunction(f) => f(&[arr.clone(), Value::Int(0)]).unwrap(),
            _ => panic!("expected native fn"),
        };
        match remove_0_res {
            NativeCallResult::Return(Value::Enum(e)) if e.variant.as_str() == "Some" => {
                assert_eq!(e.values.as_ref().unwrap().first(), Some(&Value::Int(10)));
            }
            other => panic!("expected Some(10), got {:?}", other),
        }

        // Only 15 remains
        let remove_last = match &remove_fn {
            Value::NativeFunction(f) => f(&[arr.clone()]).unwrap(),
            _ => panic!("expected native fn"),
        };
        match remove_last {
            NativeCallResult::Return(Value::Enum(e)) if e.variant.as_str() == "Some" => {
                assert_eq!(e.values.as_ref().unwrap().first(), Some(&Value::Int(15)));
            }
            other => panic!("expected Some(15), got {:?}", other),
        }

        // Now empty: table.remove(arr) -> None
        let remove_empty = match &remove_fn {
            Value::NativeFunction(f) => f(&[arr.clone()]).unwrap(),
            _ => panic!("expected native fn"),
        };
        match remove_empty {
            NativeCallResult::Return(Value::Enum(e)) => {
                assert_eq!(e.variant.as_str(), "None");
            }
            other => panic!("expected None, got {:?}", other),
        }
    }

    #[test]
    fn test_math_mod() {
        let vm = VM::with_config(&LustConfig::default());
        let stdlib = create_stdlib(&LustConfig::default(), &vm);
        let math_mod = stdlib.iter().find(|(name, _)| *name == "math").unwrap().1.clone();
        let mod_fn = math_mod.map_get(&string_key("mod")).unwrap();

        let call_mod = |a: i64, b: i64| match &mod_fn {
            Value::NativeFunction(f) => match f(&[Value::Int(a), Value::Int(b)]).unwrap() {
                NativeCallResult::Return(Value::Int(res)) => res,
                other => panic!("expected int, got {:?}", other),
            },
            _ => panic!("expected native fn"),
        };

        // 5 // 2 == 2,  5 = 2*2 + 1
        assert_eq!(call_mod(5, 2), 1);
        // -5 // 2 == -3, -5 = -3*2 + 1
        assert_eq!(call_mod(-5, 2), 1);
        // 5 // -2 == -3, 5 = -3*(-2) + (-1)
        assert_eq!(call_mod(5, -2), -1);
        // -5 // -2 == 2, -5 = 2*(-2) + (-1)
        assert_eq!(call_mod(-5, -2), -1);

        // Division by zero
        match &mod_fn {
            Value::NativeFunction(f) => {
                assert!(f(&[Value::Int(5), Value::Int(0)]).is_err());
            }
            _ => panic!("expected native fn"),
        }
    }

    #[test]
    fn test_string_slice_and_index_of() {
        let vm = VM::with_config(&LustConfig::default());
        let stdlib = create_stdlib(&LustConfig::default(), &vm);
        let str_mod = stdlib.iter().find(|(name, _)| *name == "string").unwrap().1.clone();
        let slice_fn = str_mod.map_get(&string_key("slice")).unwrap();
        let index_of_fn = str_mod.map_get(&string_key("index_of")).unwrap();

        // string.slice("hello world", 0, 5) -> "hello"
        let res = match &slice_fn {
            Value::NativeFunction(f) => f(&[Value::string("hello world"), Value::Int(0), Value::Int(5)]).unwrap(),
            _ => panic!("expected native fn"),
        };
        match res {
            NativeCallResult::Return(Value::String(s)) => assert_eq!(&*s, "hello"),
            other => panic!("expected string, got {:?}", other),
        }

        // string.slice("hello world", 6, 11) -> "world"
        let res = match &slice_fn {
            Value::NativeFunction(f) => f(&[Value::string("hello world"), Value::Int(6), Value::Int(11)]).unwrap(),
            _ => panic!("expected native fn"),
        };
        match res {
            NativeCallResult::Return(Value::String(s)) => assert_eq!(&*s, "world"),
            other => panic!("expected string, got {:?}", other),
        }

        // string.slice inverted indices -> ""
        let res = match &slice_fn {
            Value::NativeFunction(f) => f(&[Value::string("hello"), Value::Int(4), Value::Int(2)]).unwrap(),
            _ => panic!("expected native fn"),
        };
        match res {
            NativeCallResult::Return(Value::String(s)) => assert_eq!(s.as_str(), ""),
            other => panic!("expected string, got {:?}", other),
        }

        // string.index_of("hello world", "world") -> Some(6)
        let res = match &index_of_fn {
            Value::NativeFunction(f) => f(&[Value::string("hello world"), Value::string("world")]).unwrap(),
            _ => panic!("expected native fn"),
        };
        match res {
            NativeCallResult::Return(Value::Enum(e)) if e.variant.as_str() == "Some" => {
                assert_eq!(e.values.as_ref().unwrap().first(), Some(&Value::Int(6)));
            }
            other => panic!("expected Some(6), got {:?}", other),
        }

        // string.index_of("hello world", "notfound") -> None
        let res = match &index_of_fn {
            Value::NativeFunction(f) => f(&[Value::string("hello world"), Value::string("notfound")]).unwrap(),
            _ => panic!("expected native fn"),
        };
        match res {
            NativeCallResult::Return(Value::Enum(e)) => {
                assert_eq!(e.variant.as_str(), "None");
            }
            other => panic!("expected None, got {:?}", other),
        }
    }

    #[test]
    fn test_ipairs_0_based_array() {
        use crate::bytecode::value::IteratorState;
        let vm = VM::with_config(&LustConfig::default());
        let corelib = crate::vm::corelib::core_entries(&vm);
        let ipairs_fn = corelib.iter().find(|(name, _)| *name == "ipairs").unwrap().1.clone();

        let arr = Value::array(vec![Value::Int(10), Value::Int(20), Value::Int(30)]);
        let iter_res = match &ipairs_fn {
            Value::NativeFunction(f) => f(&[arr]).unwrap(),
            _ => panic!("expected native fn"),
        };

        match iter_res {
            NativeCallResult::Return(Value::Iterator(it)) => {
                let borrowed = it.borrow();
                match &*borrowed {
                    IteratorState::MapPairs { items, .. } => {
                        assert_eq!(items.len(), 3);
                        assert_eq!(items[0].0.to_value(), Value::Int(0));
                        assert_eq!(items[0].1, Value::Int(10));
                        assert_eq!(items[1].0.to_value(), Value::Int(1));
                        assert_eq!(items[1].1, Value::Int(20));
                        assert_eq!(items[2].0.to_value(), Value::Int(2));
                        assert_eq!(items[2].1, Value::Int(30));
                    }
                    _ => panic!("expected MapPairs"),
                }
            }
            other => panic!("expected iterator, got {:?}", other),
        }
    }
}

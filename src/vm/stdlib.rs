use super::corelib::string_key;
use super::VM;
use crate::bytecode::{NativeCallResult, Value};
use crate::config::LustConfig;
use crate::LustInt;
use std::fs;
use std::io::{self, Read, Write};
use std::rc::Rc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
pub fn create_stdlib(config: &LustConfig, vm: &VM) -> Vec<(&'static str, Value)> {
    let mut stdlib = vec![
        ("print", create_print_fn()),
        ("println", create_println_fn()),
        ("type", create_type_fn()),
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
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                print!("\t");
            }

            print!("{}", arg);
        }

        Ok(NativeCallResult::Return(Value::Nil))
    }))
}

fn create_println_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                print!("\t");
            }

            print!("{}", arg);
        }

        println!();
        Ok(NativeCallResult::Return(Value::Nil))
    }))
}

fn create_type_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        if args.is_empty() {
            return Err("type() requires at least one argument".to_string());
        }

        let type_name = match &args[0] {
            Value::Nil => "nil",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Tuple(_) => "tuple",
            Value::Map(_) => "map",
            Value::Struct { .. } | Value::WeakStruct(_) => "struct",
            Value::Enum { .. } => "enum",
            Value::Function(_) => "function",
            Value::NativeFunction(_) => "function",
            Value::Closure { .. } => "function",
            Value::Iterator(_) => "iterator",
            Value::Task(_) => "task",
        };
        Ok(NativeCallResult::Return(Value::string(type_name)))
    }))
}

fn create_io_module(vm: &VM) -> Value {
    let entries = [
        (string_key("read_file"), create_io_read_file_fn()),
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
    Value::NativeFunction(Rc::new(|args: &[Value]| {
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
                ))))
            }
        };
        match fs::read_to_string(path) {
            Ok(contents) => Ok(NativeCallResult::Return(Value::ok(Value::string(contents)))),
            Err(err) => Ok(NativeCallResult::Return(Value::err(Value::string(
                err.to_string(),
            )))),
        }
    }))
}

fn create_io_read_file_bytes_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
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
                ))))
            }
        };

        match fs::read(path) {
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

fn create_io_write_file_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
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
                ))))
            }
        };
        let contents = if let Some(s) = args[1].as_string() {
            s.to_string()
        } else {
            format!("{}", args[1])
        };
        match fs::write(path, contents) {
            Ok(_) => Ok(NativeCallResult::Return(Value::ok(Value::Nil))),
            Err(err) => Ok(NativeCallResult::Return(Value::err(Value::string(
                err.to_string(),
            )))),
        }
    }))
}

fn create_io_read_stdin_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        if !args.is_empty() {
            return Ok(NativeCallResult::Return(Value::err(Value::string(
                "io.read_stdin() takes no arguments",
            ))));
        }

        let mut buffer = String::new();
        match io::stdin().read_to_string(&mut buffer) {
            Ok(_) => Ok(NativeCallResult::Return(Value::ok(Value::string(buffer)))),
            Err(err) => Ok(NativeCallResult::Return(Value::err(Value::string(
                err.to_string(),
            )))),
        }
    }))
}

fn create_io_read_line_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        if !args.is_empty() {
            return Ok(NativeCallResult::Return(Value::err(Value::string(
                "io.read_line() takes no arguments",
            ))));
        }

        let mut line = String::new();
        match io::stdin().read_line(&mut line) {
            Ok(_) => Ok(NativeCallResult::Return(Value::ok(Value::string(line)))),
            Err(err) => Ok(NativeCallResult::Return(Value::err(Value::string(
                err.to_string(),
            )))),
        }
    }))
}

fn create_io_write_stdout_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
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

        Ok(NativeCallResult::Return(Value::ok(Value::Nil)))
    }))
}

fn create_os_module(vm: &VM) -> Value {
    let entries = [
        (string_key("time"), create_os_time_fn()),
        (string_key("sleep"), create_os_sleep_fn()),
        (string_key("create_file"), create_os_create_file_fn()),
        (string_key("create_dir"), create_os_create_dir_fn()),
        (string_key("remove_file"), create_os_remove_file_fn()),
        (string_key("remove_dir"), create_os_remove_dir_fn()),
        (string_key("rename"), create_os_rename_fn()),
    ];
    vm.map_with_entries(entries)
}

fn create_os_time_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
        if !args.is_empty() {
            return Ok(NativeCallResult::Return(Value::err(Value::string(
                "os.time() takes no arguments",
            ))));
        }

        let now = SystemTime::now();
        let seconds = match now.duration_since(UNIX_EPOCH) {
            Ok(duration) => duration.as_secs_f64(),
            Err(err) => -(err.duration().as_secs_f64()),
        };

        Ok(NativeCallResult::Return(Value::Float(seconds)))
    }))
}

fn create_os_sleep_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
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
                ))))
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

        thread::sleep(Duration::from_secs_f64(seconds));

        Ok(NativeCallResult::Return(Value::ok(Value::Nil)))
    }))
}

fn create_os_create_file_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
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
                ))))
            }
        };
        match fs::OpenOptions::new().write(true).create(true).open(path) {
            Ok(_) => Ok(NativeCallResult::Return(Value::ok(Value::Nil))),
            Err(err) => Ok(NativeCallResult::Return(Value::err(Value::string(
                err.to_string(),
            )))),
        }
    }))
}

fn create_os_create_dir_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
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
                ))))
            }
        };
        match fs::create_dir_all(path) {
            Ok(_) => Ok(NativeCallResult::Return(Value::ok(Value::Nil))),
            Err(err) => Ok(NativeCallResult::Return(Value::err(Value::string(
                err.to_string(),
            )))),
        }
    }))
}

fn create_os_remove_file_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
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
                ))))
            }
        };
        match fs::remove_file(path) {
            Ok(_) => Ok(NativeCallResult::Return(Value::ok(Value::Nil))),
            Err(err) => Ok(NativeCallResult::Return(Value::err(Value::string(
                err.to_string(),
            )))),
        }
    }))
}

fn create_os_remove_dir_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
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
                ))))
            }
        };
        match fs::remove_dir_all(path) {
            Ok(_) => Ok(NativeCallResult::Return(Value::ok(Value::Nil))),
            Err(err) => Ok(NativeCallResult::Return(Value::err(Value::string(
                err.to_string(),
            )))),
        }
    }))
}

fn create_os_rename_fn() -> Value {
    Value::NativeFunction(Rc::new(|args: &[Value]| {
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
                ))))
            }
        };
        let to = match args[1].as_string() {
            Some(t) => t,
            None => {
                return Ok(NativeCallResult::Return(Value::err(Value::string(
                    "os.rename(from, to) requires string paths",
                ))))
            }
        };
        match fs::rename(from, to) {
            Ok(_) => Ok(NativeCallResult::Return(Value::ok(Value::Nil))),
            Err(err) => Ok(NativeCallResult::Return(Value::err(Value::string(
                err.to_string(),
            )))),
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stdlib_defaults_without_optional_modules() {
        let stdlib = create_stdlib(&LustConfig::default());
        assert!(!stdlib.iter().any(|(name, _)| *name == "io"));
        assert!(!stdlib.iter().any(|(name, _)| *name == "os"));
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
        let stdlib = create_stdlib(&cfg);
        assert!(stdlib.iter().any(|(name, _)| *name == "io"));
        assert!(stdlib.iter().any(|(name, _)| *name == "os"));
    }
}

use super::{
    task::{TaskInstance, TaskState},
    VM,
};
use crate::bytecode::{NativeCallResult, Value, ValueKey};
use crate::config::LustConfig;
use crate::{LustError, LustInt};
use hashbrown::HashMap;
use std::fs;
use std::io::{self, Read, Write};
use std::rc::Rc;
pub fn create_stdlib(config: &LustConfig) -> Vec<(&'static str, Value)> {
    let mut stdlib = vec![
        ("print", create_print_fn()),
        ("println", create_println_fn()),
        ("type", create_type_fn()),
        ("tostring", create_tostring_fn()),
        ("task", create_task_module()),
    ];
    if config.is_module_enabled("io") {
        stdlib.push(("io", create_io_module()));
    }

    if config.is_module_enabled("os") {
        stdlib.push(("os", create_os_module()));
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
            Value::Table(_) => "table",
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

fn create_tostring_fn() -> Value {
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

fn create_task_module() -> Value {
    let mut entries: HashMap<ValueKey, Value> = HashMap::new();
    entries.insert(string_key("run"), create_task_run_fn());
    entries.insert(string_key("create"), create_task_create_fn());
    entries.insert(string_key("status"), create_task_status_fn());
    entries.insert(string_key("info"), create_task_info_fn());
    entries.insert(string_key("resume"), create_task_resume_fn());
    entries.insert(string_key("yield"), create_task_yield_fn());
    entries.insert(string_key("stop"), create_task_stop_fn());
    entries.insert(string_key("restart"), create_task_restart_fn());
    entries.insert(string_key("current"), create_task_current_fn());
    Value::table(entries)
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

fn create_io_module() -> Value {
    let mut entries: HashMap<ValueKey, Value> = HashMap::new();
    entries.insert(string_key("read_file"), create_io_read_file_fn());
    entries.insert(
        string_key("read_file_bytes"),
        create_io_read_file_bytes_fn(),
    );
    entries.insert(string_key("write_file"), create_io_write_file_fn());
    entries.insert(string_key("read_stdin"), create_io_read_stdin_fn());
    entries.insert(string_key("read_line"), create_io_read_line_fn());
    entries.insert(string_key("write_stdout"), create_io_write_stdout_fn());
    Value::table(entries)
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

fn create_os_module() -> Value {
    let mut entries: HashMap<ValueKey, Value> = HashMap::new();
    entries.insert(string_key("create_file"), create_os_create_file_fn());
    entries.insert(string_key("create_dir"), create_os_create_dir_fn());
    entries.insert(string_key("remove_file"), create_os_remove_file_fn());
    entries.insert(string_key("remove_dir"), create_os_remove_dir_fn());
    entries.insert(string_key("rename"), create_os_rename_fn());
    Value::table(entries)
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

fn string_key(name: &str) -> ValueKey {
    ValueKey::String(Rc::new(name.to_string()))
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
        let cfg =
            LustConfig::from_toml_str("\"enabled modules\" = [\"io\", \"os\"]").expect("parse");
        let stdlib = create_stdlib(&cfg);
        assert!(stdlib.iter().any(|(name, _)| *name == "io"));
        assert!(stdlib.iter().any(|(name, _)| *name == "os"));
    }
}

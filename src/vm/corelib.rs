use super::task::{TaskInstance, TaskState};
use super::VM;
use crate::bytecode::{NativeCallResult, Value, ValueKey};
use crate::LustError;
use alloc::format;
use alloc::rc::Rc;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use hashbrown::HashMap;

pub(super) fn install_core_builtins(vm: &mut VM) {
    for (name, value) in core_entries() {
        vm.register_native(name, value);
    }
}

pub(super) fn core_entries() -> Vec<(&'static str, Value)> {
    vec![
        ("tostring", create_tostring_fn()),
        ("task", create_task_module()),
    ]
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

pub(super) fn create_task_module() -> Value {
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

pub(super) fn string_key(name: &str) -> ValueKey {
    ValueKey::String(Rc::new(name.to_string()))
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

use super::*;
use crate::bytecode::{LustMap, ValueKey};
use crate::vm::task::TaskKind;
use crate::LustInt;
use alloc::{format, string::ToString};
use core::{array, cell::RefCell, mem};
impl VM {
    pub(super) fn run_task_internal(
        &mut self,
        task_id: TaskId,
        resume_value: Option<Value>,
    ) -> Result<()> {
        let mut task = match self.task_manager.detach(task_id) {
            Some(task) => task,
            None => {
                return Err(LustError::RuntimeError {
                    message: format!("Invalid task handle {}", task_id.as_u64()),
                })
            }
        };
        if matches!(task.kind(), TaskKind::NativeFuture { .. }) {
            let message = format!(
                "Task {} is managed by the host runtime and cannot be resumed manually",
                task_id.as_u64()
            );
            self.task_manager.attach(task);
            return Err(LustError::RuntimeError { message });
        }

        match task.state {
            TaskState::Completed | TaskState::Failed | TaskState::Stopped => {
                let message = format!(
                    "Task {} cannot be resumed (state: {})",
                    task_id.as_u64(),
                    task.state.as_str()
                );
                self.task_manager.attach(task);
                return Err(LustError::RuntimeError { message });
            }

            TaskState::Running => {
                self.task_manager.attach(task);
                return Err(LustError::RuntimeError {
                    message: format!("Task {} is already running", task_id.as_u64()),
                });
            }

            _ => {}
        }

        task.state = TaskState::Running;
        task.last_yield = None;
        let mut resume_value_opt = resume_value;
        if let Some(dest) = task.yield_dest.take() {
            let value = resume_value_opt.take().unwrap_or(Value::Nil);
            if let Some(frame) = task.call_stack.last_mut() {
                frame.registers[dest as usize] = value;
            }
        } else if resume_value_opt.is_some() {
            let message = format!(
                "Task {} is not waiting for a resume value",
                task_id.as_u64()
            );
            self.task_manager.attach(task);
            return Err(LustError::RuntimeError { message });
        }

        mem::swap(&mut self.call_stack, &mut task.call_stack);
        mem::swap(
            &mut self.pending_return_value,
            &mut task.pending_return_value,
        );
        mem::swap(&mut self.pending_return_dest, &mut task.pending_return_dest);
        self.current_task = Some(task_id);
        self.last_task_signal = None;
        let run_result = self.run();
        let signal = self.last_task_signal.take();
        self.current_task = None;
        let mut error_result: Option<LustError> = None;
        match run_result {
            Ok(value) => {
                if let Some(signal) = signal {
                    match signal {
                        TaskSignal::Yield {
                            dest,
                            value: yielded,
                        } => {
                            task.state = TaskState::Yielded;
                            task.last_yield = Some(yielded);
                            task.last_result = None;
                            task.yield_dest = Some(dest);
                        }

                        TaskSignal::Stop { value: stop_value } => {
                            task.state = TaskState::Stopped;
                            task.last_result = Some(stop_value);
                            task.last_yield = None;
                            task.call_stack.clear();
                            task.pending_return_value = None;
                            task.pending_return_dest = None;
                        }
                    }
                } else {
                    task.state = TaskState::Completed;
                    task.last_result = Some(value);
                    task.last_yield = None;
                }
            }

            Err(err) => {
                let annotated = self.annotate_runtime_error(err);
                task.state = TaskState::Failed;
                task.error = Some(annotated.clone());
                task.last_yield = None;
                error_result = Some(annotated);
            }
        }

        mem::swap(&mut self.call_stack, &mut task.call_stack);
        mem::swap(
            &mut self.pending_return_value,
            &mut task.pending_return_value,
        );
        mem::swap(&mut self.pending_return_dest, &mut task.pending_return_dest);
        self.task_manager.attach(task);
        if let Some(err) = error_result {
            Err(err)
        } else {
            Ok(())
        }
    }

    pub(super) fn task_id_from_handle(&self, handle: TaskHandle) -> Result<TaskId> {
        let id = TaskId(handle.id());
        if self.task_manager.contains(id) {
            Ok(id)
        } else {
            Err(LustError::RuntimeError {
                message: format!("Invalid task handle {}", handle.id()),
            })
        }
    }

    pub(super) fn prepare_task_frame(
        &mut self,
        func: Value,
        args: Vec<Value>,
    ) -> Result<CallFrame> {
        match func {
            Value::Function(func_idx) => {
                let function = &self.functions[func_idx];
                if args.len() != function.param_count as usize {
                    return Err(LustError::RuntimeError {
                        message: format!(
                            "Task entry expects {} arguments, got {}",
                            function.param_count,
                            args.len()
                        ),
                    });
                }

                let mut frame = CallFrame {
                    function_idx: func_idx,
                    ip: 0,
                    registers: array::from_fn(|_| Value::Nil),
                    base_register: 0,
                    return_dest: None,
                    upvalues: Vec::new(),
                };
                for (i, arg) in args.into_iter().enumerate() {
                    frame.registers[i] = arg;
                }

                Ok(frame)
            }

            Value::Closure {
                function_idx,
                upvalues,
            } => {
                let function = &self.functions[function_idx];
                if args.len() != function.param_count as usize {
                    return Err(LustError::RuntimeError {
                        message: format!(
                            "Task entry expects {} arguments, got {}",
                            function.param_count,
                            args.len()
                        ),
                    });
                }

                let captured: Vec<Value> = upvalues.iter().map(|uv| uv.get()).collect();
                let mut frame = CallFrame {
                    function_idx,
                    ip: 0,
                    registers: array::from_fn(|_| Value::Nil),
                    base_register: 0,
                    return_dest: None,
                    upvalues: captured,
                };
                for (i, arg) in args.into_iter().enumerate() {
                    frame.registers[i] = arg;
                }

                Ok(frame)
            }

            other => Err(LustError::RuntimeError {
                message: format!("task.run() expects a function or closure, got {:?}", other),
            }),
        }
    }

    pub(super) fn create_task_value(
        &mut self,
        func: Value,
        args: Vec<Value>,
    ) -> Result<TaskHandle> {
        let frame = self.prepare_task_frame(func, args)?;
        let task_id = self.task_manager.next_id();
        let task = TaskInstance::new(task_id, frame);
        self.task_manager.insert(task);
        Ok(task_id.to_handle())
    }

    pub fn spawn_task_value(&mut self, func: Value, args: Vec<Value>) -> Result<TaskHandle> {
        let handle = self.create_task_value(func, args)?;
        let task_id = TaskId(handle.id());
        if let Err(err) = self.run_task_internal(task_id, None) {
            let _ = self.task_manager.detach(task_id);
            return Err(err);
        }

        Ok(handle)
    }

    pub fn spawn_tick_task(&mut self, function_name: &str) -> Result<TaskHandle> {
        let canonical = if function_name.contains("::") {
            function_name.replace("::", ".")
        } else {
            function_name.to_string()
        };
        let func_idx = self
            .functions
            .iter()
            .position(|f| f.name == canonical)
            .ok_or_else(|| LustError::RuntimeError {
                message: format!("Function not found: {}", function_name),
            })?;

        let yield_fn = self
            .globals
            .get("task")
            .cloned()
            .and_then(|task| match task {
                Value::Map(map) => map
                    .borrow()
                    .get(&ValueKey::string("yield".to_string()))
                    .cloned(),
                _ => None,
            })
            .ok_or_else(|| LustError::RuntimeError {
                message: "Missing corelib 'task.yield' (task module not installed?)".to_string(),
            })?;

        if !matches!(yield_fn, Value::NativeFunction(_)) {
            return Err(LustError::RuntimeError {
                message: "corelib 'task.yield' is not a native function".to_string(),
            });
        }

        let wrapper_name = format!("__jit_tick_driver_{}", func_idx);
        let wrapper_idx = match self.functions.iter().position(|f| f.name == wrapper_name) {
            Some(existing) => existing,
            None => {
                let target = &self.functions[func_idx];
                let arg_count = target.param_count;

                let mut wrapper = Function::new(wrapper_name, 0, false);

                let resume_reg: Register = 0;
                let tick_fn_reg: Register = 1;
                let yield_fn_reg: Register = 2;
                let idx_reg: Register = 3;
                let arg_base: Register = 4;
                let result_reg: Register = arg_base.saturating_add(arg_count);

                let required_registers = (result_reg as u16 + 1).min(256) as u8;
                wrapper.set_register_count(required_registers);

                let tick_const = wrapper.chunk.add_constant(Value::Function(func_idx));
                let yield_const = wrapper.chunk.add_constant(yield_fn);
                let mut index_consts = Vec::new();
                if arg_count > 1 {
                    for i in 0..(arg_count as i64) {
                        index_consts.push(wrapper.chunk.add_constant(Value::Int(i)));
                    }
                }

                wrapper
                    .chunk
                    .emit(Instruction::LoadConst(tick_fn_reg, tick_const), 0);
                wrapper
                    .chunk
                    .emit(Instruction::LoadConst(yield_fn_reg, yield_const), 0);

                // Prime the task so the host can immediately supply the first tick's argument via resume().
                wrapper
                    .chunk
                    .emit(Instruction::Call(yield_fn_reg, 0, 0, resume_reg), 0);

                let loop_start = wrapper.chunk.instructions.len();

                match arg_count {
                    0 => {
                        wrapper
                            .chunk
                            .emit(Instruction::Call(tick_fn_reg, 0, 0, result_reg), 0);
                    }
                    1 => {
                        wrapper.chunk.emit(
                            Instruction::Call(tick_fn_reg, resume_reg, 1, result_reg),
                            0,
                        );
                    }
                    _ => {
                        // For N>1, expect resume() to pass an Array of arguments.
                        for (i, const_idx) in index_consts.iter().enumerate() {
                            wrapper
                                .chunk
                                .emit(Instruction::LoadConst(idx_reg, *const_idx), 0);
                            wrapper.chunk.emit(
                                Instruction::GetIndex(arg_base + i as u8, resume_reg, idx_reg),
                                0,
                            );
                        }
                        wrapper.chunk.emit(
                            Instruction::Call(tick_fn_reg, arg_base, arg_count, result_reg),
                            0,
                        );
                    }
                }

                // Yield the on_tick() result back to the host; resume() will write the next tick arg into resume_reg.
                wrapper.chunk.emit(
                    Instruction::Call(yield_fn_reg, result_reg, 1, resume_reg),
                    0,
                );

                let jump_idx = wrapper.chunk.emit(Instruction::Jump(0), 0);
                wrapper.chunk.patch_jump(jump_idx, loop_start);

                let new_idx = self.functions.len();
                self.functions.push(wrapper);
                new_idx
            }
        };

        self.spawn_task_value(Value::Function(wrapper_idx), Vec::new())
    }

    pub fn tick_task(&mut self, handle: TaskHandle, resume_value: Value) -> Result<Value> {
        self.resume_task_handle(handle, Some(resume_value))?;
        let task = self.get_task_instance(handle)?;
        match task.state {
            TaskState::Yielded => Ok(task.last_yield.clone().unwrap_or(Value::Nil)),
            TaskState::Completed | TaskState::Stopped => Ok(task.last_result.clone().unwrap_or(Value::Nil)),
            TaskState::Ready | TaskState::Running => Ok(Value::Nil),
            TaskState::Failed => Err(task.error.clone().unwrap_or_else(|| LustError::RuntimeError {
                message: "Task failed".to_string(),
            })),
        }
    }

    pub fn resume_task_handle(
        &mut self,
        handle: TaskHandle,
        resume_value: Option<Value>,
    ) -> Result<()> {
        let task_id = self.task_id_from_handle(handle)?;
        self.run_task_internal(task_id, resume_value)
    }

    pub(super) fn stop_task_handle(&mut self, handle: TaskHandle) -> Result<()> {
        let task_id = self.task_id_from_handle(handle)?;
        let mut task = match self.task_manager.detach(task_id) {
            Some(task) => task,
            None => {
                return Err(LustError::RuntimeError {
                    message: format!("Invalid task handle {}", handle.id()),
                })
            }
        };
        match task.state {
            TaskState::Stopped | TaskState::Completed | TaskState::Failed => {
                self.task_manager.attach(task);
                return Ok(());
            }

            TaskState::Running => {
                self.task_manager.attach(task);
                return Err(LustError::RuntimeError {
                    message: format!("Task {} is currently running", handle.id()),
                });
            }

            _ => {}
        }

        task.state = TaskState::Stopped;
        task.call_stack.clear();
        task.pending_return_value = None;
        task.pending_return_dest = None;
        task.yield_dest = None;
        task.last_yield = None;
        self.task_manager.attach(task);
        Ok(())
    }

    pub(super) fn restart_task_handle(&mut self, handle: TaskHandle) -> Result<()> {
        let task_id = self.task_id_from_handle(handle)?;
        let mut task = match self.task_manager.detach(task_id) {
            Some(task) => task,
            None => {
                return Err(LustError::RuntimeError {
                    message: format!("Invalid task handle {}", handle.id()),
                })
            }
        };
        task.reset();
        self.task_manager.insert(task);
        if let Err(err) = self.run_task_internal(task_id, None) {
            return Err(err);
        }

        Ok(())
    }

    pub fn get_task_instance(&self, handle: TaskHandle) -> Result<&TaskInstance> {
        let task_id = self.task_id_from_handle(handle)?;
        self.task_manager
            .get(task_id)
            .ok_or_else(|| LustError::RuntimeError {
                message: format!("Invalid task handle {}", handle.id()),
            })
    }

    pub fn current_task_handle(&self) -> Option<TaskHandle> {
        self.current_task.map(|id| id.to_handle())
    }

    pub fn create_native_future_task(&mut self) -> TaskHandle {
        let id = self.task_manager.next_id();
        let task = TaskInstance::new_native_future(id);
        let handle = task.handle();
        self.task_manager.insert(task);
        handle
    }

    pub fn complete_native_future_task(
        &mut self,
        handle: TaskHandle,
        outcome: std::result::Result<Value, String>,
    ) -> Result<()> {
        let task_id = self.task_id_from_handle(handle)?;
        let mut task = match self.task_manager.detach(task_id) {
            Some(task) => task,
            None => {
                return Err(LustError::RuntimeError {
                    message: format!("Invalid task handle {}", handle.id()),
                })
            }
        };

        match task.kind_mut() {
            TaskKind::NativeFuture { .. } => {
                match outcome {
                    Ok(value) => {
                        task.state = TaskState::Completed;
                        task.last_result = Some(value);
                        task.error = None;
                    }
                    Err(err_msg) => {
                        task.state = TaskState::Failed;
                        task.last_result = None;
                        task.error = Some(LustError::RuntimeError { message: err_msg });
                    }
                }
                task.last_yield = None;
                task.pending_return_value = None;
                task.pending_return_dest = None;
                task.yield_dest = None;
                self.task_manager.attach(task);
                Ok(())
            }

            TaskKind::Script => {
                self.task_manager.attach(task);
                Err(LustError::RuntimeError {
                    message: "Attempted to complete a script task using native future completion"
                        .to_string(),
                })
            }
        }
    }

    pub(super) fn call_builtin_method(
        &mut self,
        object: &Value,
        method_name: &str,
        args: Vec<Value>,
    ) -> Result<Value> {
        #[cfg(feature = "std")]
        if std::env::var_os("LUST_LUA_SOCKET_TRACE").is_some() && method_name == "settimeout" {
            match object {
                Value::Enum {
                    enum_name,
                    variant,
                    ..
                } => {
                    eprintln!(
                        "[lua-socket] CallMethod enum={} variant={} method={}",
                        enum_name, variant, method_name
                    );
                }
                other => {
                    eprintln!(
                        "[lua-socket] CallMethod type={:?} method={}",
                        other.type_of(),
                        method_name
                    );
                }
            }
        }

        if let Value::Enum {
            enum_name,
            variant,
            ..
        } = object
        {
            if enum_name == "LuaValue" && variant == "Userdata" {
                if let Some(result) = self.try_call_lua_dynamic_method(object, method_name, &args)?
                {
                    return Ok(result);
                }
                #[cfg(feature = "std")]
                if std::env::var_os("LUST_LUA_SOCKET_TRACE").is_some() {
                    let indexer = self.lua_index_metamethod(object);
                    eprintln!(
                        "[lua-socket] userdata missing method '{}' indexer={:?} userdata={:?}",
                        method_name,
                        indexer.as_ref().map(|v| v.type_of()),
                        object
                    );
                }
            }
        }

        if let Value::Struct { name, .. } = object {
            if name == "LuaTable" {
                if let Some(result) = self.try_call_lua_dynamic_method(object, method_name, &args)?
                {
                    return Ok(result);
                }
            }
        }

        if let Value::Enum {
            enum_name,
            variant,
            values,
        } = object
        {
            if enum_name == "LuaValue" && variant == "Table" {
                if let Some(inner) = values.as_ref().and_then(|vals| vals.get(0)) {
                    return self.call_builtin_method(inner, method_name, args);
                }
            }
        }

        if let Value::Struct {
            name: struct_name, ..
        } = object
        {
            let mangled_name = format!("{}:{}", struct_name, method_name);
            if let Some(func_idx) = self.functions.iter().position(|f| f.name == mangled_name) {
                let mut method_args = vec![object.clone()];
                method_args.extend(args.clone());
                return self.call_value(&Value::Function(func_idx), method_args);
            }

            let mut candidate_names = vec![mangled_name.clone()];
            if let Some(simple) = struct_name.rsplit(|c| c == '.' || c == ':').next() {
                candidate_names.push(format!("{}:{}", simple, method_name));
            }

            for candidate in candidate_names {
                let mut resolved = None;
                for variant in [candidate.clone(), candidate.replace('.', "::")] {
                    if let Some(value) = self.get_global(&variant) {
                        resolved = Some(value);
                        break;
                    }
                }

                if let Some(global_func) = resolved {
                    let mut method_args = vec![object.clone()];
                    method_args.extend(args.clone());
                    return self.call_value(&global_func, method_args);
                }
            }
        }

        match object {
            Value::Struct { name, .. } if name == "LuaTable" => {
                let Some(map_rc) = lua_table_map_rc(object) else {
                    return Err(LustError::RuntimeError {
                        message: "LuaTable is missing 'table' map field".to_string(),
                    });
                };
                match method_name {
                    "len" => {
                        if !args.is_empty() {
                            return Err(LustError::RuntimeError {
                                message: "len() takes no arguments".to_string(),
                            });
                        }
                        let seq = lua_table_read_sequence(&map_rc.borrow());
                        Ok(Value::Int(seq.len() as LustInt))
                    }
                    "push" => {
                        if args.len() != 1 {
                            return Err(LustError::RuntimeError {
                                message: "push() requires 1 argument (value)".to_string(),
                            });
                        }
                        let value = super::corelib::unwrap_lua_value(args[0].clone());
                        let mut map = map_rc.borrow_mut();
                        let len = lua_table_sequence_len(&map);
                        let key = ValueKey::from_value(&Value::Int((len as LustInt) + 1));
                        map.insert(key, value);
                        Ok(Value::Nil)
                    }
                    "insert" => {
                        if args.len() != 2 {
                            return Err(LustError::RuntimeError {
                                message: "insert() requires 2 arguments (pos, value)".to_string(),
                            });
                        }
                        let pos_raw = super::corelib::unwrap_lua_value(args[0].clone());
                        let pos = pos_raw.as_int().unwrap_or(0).max(1) as usize;
                        let value = super::corelib::unwrap_lua_value(args[1].clone());
                        let mut seq = lua_table_read_sequence(&map_rc.borrow());
                        let idx = pos.saturating_sub(1);
                        if idx > seq.len() {
                            seq.push(value);
                        } else {
                            seq.insert(idx, value);
                        }
                        lua_table_write_sequence(&map_rc, &seq);
                        Ok(Value::Nil)
                    }
                    "remove" => {
                        if args.len() != 1 {
                            return Err(LustError::RuntimeError {
                                message: "remove() requires 1 argument (pos)".to_string(),
                            });
                        }
                        let pos_raw = super::corelib::unwrap_lua_value(args[0].clone());
                        let mut seq = lua_table_read_sequence(&map_rc.borrow());
                        if seq.is_empty() {
                            return Ok(Value::Nil);
                        }
                        let pos = pos_raw.as_int().unwrap_or(seq.len() as LustInt);
                        let idx =
                            ((pos - 1).max(0) as usize).min(seq.len().saturating_sub(1));
                        let removed = seq.remove(idx);
                        lua_table_write_sequence(&map_rc, &seq);
                        Ok(removed)
                    }
                    "concat" => {
                        if args.len() != 3 {
                            return Err(LustError::RuntimeError {
                                message: "concat() requires 3 arguments (sep, i, j)".to_string(),
                            });
                        }
                        let sep_raw = super::corelib::unwrap_lua_value(args[0].clone());
                        let sep = sep_raw.as_string().unwrap_or_default();
                        let seq = lua_table_read_sequence(&map_rc.borrow());
                        let start = super::corelib::unwrap_lua_value(args[1].clone())
                            .as_int()
                            .unwrap_or(1);
                        let end = super::corelib::unwrap_lua_value(args[2].clone())
                            .as_int()
                            .unwrap_or(seq.len() as LustInt);
                        let start_idx = (start - 1).max(0) as usize;
                        let end_idx = end.max(0) as usize;
                        let mut pieces: Vec<String> = Vec::new();
                        for (i, val) in seq.iter().enumerate() {
                            if i < start_idx || i >= end_idx {
                                continue;
                            }
                            let raw = super::corelib::unwrap_lua_value(val.clone());
                            pieces.push(format!("{}", raw));
                        }
                        Ok(Value::string(pieces.join(&sep)))
                    }
                    "unpack" => {
                        if args.len() != 2 {
                            return Err(LustError::RuntimeError {
                                message: "unpack() requires 2 arguments (i, j)".to_string(),
                            });
                        }
                        let unpack = super::stdlib::create_table_unpack_fn();
                        let Value::NativeFunction(func) = unpack else {
                            return Err(LustError::RuntimeError {
                                message: "unpack() builtin is not a native function".to_string(),
                            });
                        };
                        let call_args = vec![object.clone(), args[0].clone(), args[1].clone()];
                        match func(&call_args).map_err(|e| LustError::RuntimeError { message: e })? {
                            NativeCallResult::Return(value) => Ok(value),
                            NativeCallResult::Yield(_) => Err(LustError::RuntimeError {
                                message: "unpack() unexpectedly yielded".to_string(),
                            }),
                            NativeCallResult::Stop(_) => Err(LustError::RuntimeError {
                                message: "unpack() unexpectedly stopped execution".to_string(),
                            }),
                        }
                    }
                    "sort" => {
                        if args.len() != 1 {
                            return Err(LustError::RuntimeError {
                                message: "sort() requires 1 argument (comp)".to_string(),
                            });
                        }
                        let mut seq = lua_table_read_sequence(&map_rc.borrow());
                        seq.sort_by(|a, b| {
                            let la = format!("{}", super::corelib::unwrap_lua_value(a.clone()));
                            let lb = format!("{}", super::corelib::unwrap_lua_value(b.clone()));
                            la.cmp(&lb)
                        });
                        lua_table_write_sequence(&map_rc, &seq);
                        Ok(Value::Nil)
                    }
                    "maxn" => {
                        if !args.is_empty() {
                            return Err(LustError::RuntimeError {
                                message: "maxn() takes no arguments".to_string(),
                            });
                        }
                        let map = map_rc.borrow();
                        let mut max_idx: LustInt = 0;
                        for key in map.keys() {
                            if let Value::Int(i) = key.to_value() {
                                if i > max_idx && i > 0 {
                                    max_idx = i;
                                }
                            }
                        }
                        Ok(Value::Int(max_idx))
                    }
                    _ => Err(LustError::RuntimeError {
                        message: format!("LuaTable has no method '{}'", method_name),
                    }),
                }
            }
            Value::Enum {
                enum_name,
                variant,
                values,
            } if enum_name == "Option" => match method_name {
                "is_some" => Ok(Value::Bool(variant == "Some")),
                "is_none" => Ok(Value::Bool(variant == "None")),
                "unwrap" => {
                    if variant == "Some" {
                        if let Some(vals) = values {
                            if !vals.is_empty() {
                                Ok(vals[0].clone())
                            } else {
                                Err(LustError::RuntimeError {
                                    message: "Option::Some has no value".to_string(),
                                })
                            }
                        } else {
                            Err(LustError::RuntimeError {
                                message: "Option::Some has no value".to_string(),
                            })
                        }
                    } else {
                        Err(LustError::RuntimeError {
                            message: "Called unwrap() on Option::None".to_string(),
                        })
                    }
                }

                "unwrap_or" => {
                    if args.is_empty() {
                        return Err(LustError::RuntimeError {
                            message: "unwrap_or requires a default value".to_string(),
                        });
                    }

                    if variant == "Some" {
                        if let Some(vals) = values {
                            if !vals.is_empty() {
                                Ok(vals[0].clone())
                            } else {
                                Ok(args[0].clone())
                            }
                        } else {
                            Ok(args[0].clone())
                        }
                    } else {
                        Ok(args[0].clone())
                    }
                }

                _ => Err(LustError::RuntimeError {
                    message: format!("Option has no method '{}'", method_name),
                }),
            },
            Value::Enum {
                enum_name,
                variant,
                values,
            } if enum_name == "Result" => match method_name {
                "is_ok" => Ok(Value::Bool(variant == "Ok")),
                "is_err" => Ok(Value::Bool(variant == "Err")),
                "unwrap" => {
                    if variant == "Ok" {
                        if let Some(vals) = values {
                            if !vals.is_empty() {
                                Ok(vals[0].clone())
                            } else {
                                Err(LustError::RuntimeError {
                                    message: "Result::Ok has no value".to_string(),
                                })
                            }
                        } else {
                            Err(LustError::RuntimeError {
                                message: "Result::Ok has no value".to_string(),
                            })
                        }
                    } else {
                        Err(LustError::RuntimeError {
                            message: "Called unwrap() on Result::Err".to_string(),
                        })
                    }
                }

                "unwrap_or" => {
                    if args.is_empty() {
                        return Err(LustError::RuntimeError {
                            message: "unwrap_or requires a default value".to_string(),
                        });
                    }

                    if variant == "Ok" {
                        if let Some(vals) = values {
                            if !vals.is_empty() {
                                Ok(vals[0].clone())
                            } else {
                                Ok(args[0].clone())
                            }
                        } else {
                            Ok(args[0].clone())
                        }
                    } else {
                        Ok(args[0].clone())
                    }
                }

                _ => Err(LustError::RuntimeError {
                    message: format!("Result has no method '{}'", method_name),
                }),
            },
            Value::Array(arr) => match method_name {
                "iter" => {
                    let items = arr.borrow().clone();
                    let iter = crate::bytecode::value::IteratorState::Array { items, index: 0 };
                    Ok(Value::Iterator(Rc::new(RefCell::new(iter))))
                }

                "len" => Ok(Value::Int(int_from_usize(arr.borrow().len()))),
                "get" => {
                    if args.is_empty() {
                        return Err(LustError::RuntimeError {
                            message: "get requires an index argument".to_string(),
                        });
                    }

                    let index = args[0].as_int().ok_or_else(|| LustError::RuntimeError {
                        message: "Array index must be an integer".to_string(),
                    })?;
                    let borrowed = arr.borrow();
                    if index < 0 || index as usize >= borrowed.len() {
                        Ok(Value::none())
                    } else {
                        Ok(Value::some(borrowed[index as usize].clone()))
                    }
                }

                "first" => {
                    let borrowed = arr.borrow();
                    if borrowed.is_empty() {
                        Ok(Value::none())
                    } else {
                        Ok(Value::some(borrowed[0].clone()))
                    }
                }

                "last" => {
                    let borrowed = arr.borrow();
                    if borrowed.is_empty() {
                        Ok(Value::none())
                    } else {
                        Ok(Value::some(borrowed[borrowed.len() - 1].clone()))
                    }
                }

                "push" => {
                    if args.is_empty() {
                        return Err(LustError::RuntimeError {
                            message: "push requires a value argument".to_string(),
                        });
                    }

                    arr.borrow_mut().push(args[0].clone());
                    Ok(Value::Nil)
                }

                "pop" => {
                    let popped = arr.borrow_mut().pop();
                    match popped {
                        Some(val) => Ok(Value::some(val)),
                        None => Ok(Value::none()),
                    }
                }

                "map" => {
                    if args.is_empty() {
                        return Err(LustError::RuntimeError {
                            message: "map requires a function argument".to_string(),
                        });
                    }

                    let func = &args[0];
                    let borrowed = arr.borrow();
                    let mut result = Vec::new();
                    for elem in borrowed.iter() {
                        let mapped_value = self.call_value(func, vec![elem.clone()])?;
                        result.push(mapped_value);
                    }

                    Ok(Value::array(result))
                }

                "filter" => {
                    if args.is_empty() {
                        return Err(LustError::RuntimeError {
                            message: "filter requires a function argument".to_string(),
                        });
                    }

                    let func = &args[0];
                    let borrowed = arr.borrow();
                    let mut result = Vec::new();
                    for elem in borrowed.iter() {
                        let keep = self.call_value(func, vec![elem.clone()])?;
                        if keep.is_truthy() {
                            result.push(elem.clone());
                        }
                    }

                    Ok(Value::array(result))
                }

                "reduce" => {
                    if args.len() < 2 {
                        return Err(LustError::RuntimeError {
                            message: "reduce requires an initial value and function".to_string(),
                        });
                    }

                    let init_value = &args[0];
                    let func = &args[1];
                    let borrowed = arr.borrow();
                    let mut accumulator = init_value.clone();
                    for elem in borrowed.iter() {
                        accumulator = self.call_value(func, vec![accumulator, elem.clone()])?;
                    }

                    Ok(accumulator)
                }

                "slice" => {
                    if args.len() < 2 {
                        return Err(LustError::RuntimeError {
                            message: "slice requires start and end indices".to_string(),
                        });
                    }

                    let start = args[0].as_int().ok_or_else(|| LustError::RuntimeError {
                        message: "Start index must be an integer".to_string(),
                    })? as usize;
                    let end = args[1].as_int().ok_or_else(|| LustError::RuntimeError {
                        message: "End index must be an integer".to_string(),
                    })? as usize;
                    let borrowed = arr.borrow();
                    if start > borrowed.len() || end > borrowed.len() || start > end {
                        return Err(LustError::RuntimeError {
                            message: "Invalid slice indices".to_string(),
                        });
                    }

                    let sliced = borrowed[start..end].to_vec();
                    Ok(Value::array(sliced))
                }

                "clear" => {
                    arr.borrow_mut().clear();
                    Ok(Value::Nil)
                }

                "is_empty" => Ok(Value::Bool(arr.borrow().is_empty())),
                _ => Err(LustError::RuntimeError {
                    message: format!("Array has no method '{}'", method_name),
                }),
            },
            Value::String(s) => match method_name {
                "iter" => {
                    let items: Vec<Value> =
                        s.chars().map(|c| Value::string(c.to_string())).collect();
                    let iter = crate::bytecode::value::IteratorState::Array { items, index: 0 };
                    Ok(Value::Iterator(Rc::new(RefCell::new(iter))))
                }

                "len" => Ok(Value::Int(int_from_usize(s.len()))),
                "substring" => {
                    if args.len() < 2 {
                        return Err(LustError::RuntimeError {
                            message: "substring requires start and end indices".to_string(),
                        });
                    }

                    let start = args[0].as_int().ok_or_else(|| LustError::RuntimeError {
                        message: "Start index must be an integer".to_string(),
                    })? as usize;
                    let end = args[1].as_int().ok_or_else(|| LustError::RuntimeError {
                        message: "End index must be an integer".to_string(),
                    })? as usize;
                    if start > s.len() || end > s.len() || start > end {
                        return Err(LustError::RuntimeError {
                            message: "Invalid substring indices".to_string(),
                        });
                    }

                    Ok(Value::string(&s[start..end]))
                }

                "find" => {
                    if args.is_empty() {
                        return Err(LustError::RuntimeError {
                            message: "find requires a search string".to_string(),
                        });
                    }

                    let search = args[0].as_string().ok_or_else(|| LustError::RuntimeError {
                        message: "Search string must be a string".to_string(),
                    })?;
                    match s.find(search) {
                        Some(pos) => Ok(Value::some(Value::Int(int_from_usize(pos)))),
                        None => Ok(Value::none()),
                    }
                }

                "starts_with" => {
                    if args.is_empty() {
                        return Err(LustError::RuntimeError {
                            message: "starts_with requires a prefix string".to_string(),
                        });
                    }

                    let prefix = args[0].as_string().ok_or_else(|| LustError::RuntimeError {
                        message: "Prefix must be a string".to_string(),
                    })?;
                    Ok(Value::Bool(s.starts_with(prefix)))
                }

                "ends_with" => {
                    if args.is_empty() {
                        return Err(LustError::RuntimeError {
                            message: "ends_with requires a suffix string".to_string(),
                        });
                    }

                    let suffix = args[0].as_string().ok_or_else(|| LustError::RuntimeError {
                        message: "Suffix must be a string".to_string(),
                    })?;
                    Ok(Value::Bool(s.ends_with(suffix)))
                }

                "split" => {
                    if args.is_empty() {
                        return Err(LustError::RuntimeError {
                            message: "split requires a separator string".to_string(),
                        });
                    }

                    let separator = args[0].as_string().ok_or_else(|| LustError::RuntimeError {
                        message: "Separator must be a string".to_string(),
                    })?;
                    let parts: Vec<Value> =
                        s.split(separator).map(|part| Value::string(part)).collect();
                    Ok(Value::array(parts))
                }

                "trim" => Ok(Value::string(s.trim())),
                "trim_start" => Ok(Value::string(s.trim_start())),
                "trim_end" => Ok(Value::string(s.trim_end())),
                "replace" => {
                    if args.len() < 2 {
                        return Err(LustError::RuntimeError {
                            message: "replace requires 'from' and 'to' string arguments"
                                .to_string(),
                        });
                    }

                    let from = args[0].as_string().ok_or_else(|| LustError::RuntimeError {
                        message: "First argument must be a string".to_string(),
                    })?;
                    let to = args[1].as_string().ok_or_else(|| LustError::RuntimeError {
                        message: "Second argument must be a string".to_string(),
                    })?;
                    Ok(Value::string(&s.replace(from, to)))
                }

                "to_upper" => Ok(Value::string(&s.to_uppercase())),
                "to_lower" => Ok(Value::string(&s.to_lowercase())),
                "contains" => {
                    if args.is_empty() {
                        return Err(LustError::RuntimeError {
                            message: "contains requires a search string".to_string(),
                        });
                    }

                    let search = args[0].as_string().ok_or_else(|| LustError::RuntimeError {
                        message: "Search string must be a string".to_string(),
                    })?;
                    Ok(Value::Bool(s.contains(search)))
                }

                "is_empty" => Ok(Value::Bool(s.is_empty())),
                "chars" => {
                    let chars: Vec<Value> =
                        s.chars().map(|c| Value::string(&c.to_string())).collect();
                    Ok(Value::array(chars))
                }

                "lines" => {
                    let lines: Vec<Value> = s.lines().map(|line| Value::string(line)).collect();
                    Ok(Value::array(lines))
                }

                _ => Err(LustError::RuntimeError {
                    message: format!("String has no method '{}'", method_name),
                }),
            },
            Value::Map(map) => {
                use crate::bytecode::ValueKey;
                match method_name {
                    "iter" => {
                        let items: Vec<(ValueKey, Value)> = map
                            .borrow()
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect();
                        let iter =
                            crate::bytecode::value::IteratorState::MapPairs { items, index: 0 };
                        return Ok(Value::Iterator(Rc::new(RefCell::new(iter))));
                    }

                    "len" => Ok(Value::Int(int_from_usize(map.borrow().len()))),
                    "get" => {
                        if args.is_empty() {
                            return Err(LustError::RuntimeError {
                                message: "get requires a key argument".to_string(),
                            });
                        }

                        let key = self.make_hash_key(&args[0])?;
                        match map.borrow().get(&key) {
                            Some(value) => Ok(Value::some(value.clone())),
                            None => Ok(Value::none()),
                        }
                    }

                    "set" => {
                        if args.len() < 2 {
                            return Err(LustError::RuntimeError {
                                message: "set requires key and value arguments".to_string(),
                            });
                        }

                        let key = self.make_hash_key(&args[0])?;
                        let value = args[1].clone();
                        map.borrow_mut().insert(key, value);
                        Ok(Value::Nil)
                    }

                    "has" => {
                        if args.is_empty() {
                            return Err(LustError::RuntimeError {
                                message: "has requires a key argument".to_string(),
                            });
                        }

                        let key = self.make_hash_key(&args[0])?;
                        Ok(Value::Bool(map.borrow().contains_key(&key)))
                    }

                    "delete" => {
                        if args.is_empty() {
                            return Err(LustError::RuntimeError {
                                message: "delete requires a key argument".to_string(),
                            });
                        }

                        let key = self.make_hash_key(&args[0])?;
                        match map.borrow_mut().remove(&key) {
                            Some(value) => Ok(Value::some(value)),
                            None => Ok(Value::none()),
                        }
                    }

                    "keys" => {
                        let keys: Vec<Value> = map.borrow().keys().map(|k| k.to_value()).collect();
                        Ok(Value::array(keys))
                    }

                    "values" => {
                        let values: Vec<Value> = map.borrow().values().cloned().collect();
                        Ok(Value::array(values))
                    }

                    _ => Err(LustError::RuntimeError {
                        message: format!("Map has no method '{}'", method_name),
                    }),
                }
            }

            Value::Iterator(state_rc) => match method_name {
                "iter" => Ok(Value::Iterator(state_rc.clone())),
                "next" => {
                    use crate::bytecode::value::IteratorState;
                    let mut state = state_rc.borrow_mut();
                    match &mut *state {
                        IteratorState::Array { items, index } => {
                            if *index < items.len() {
                                let v = items[*index].clone();
                                *index += 1;
                                Ok(Value::some(v))
                            } else {
                                Ok(Value::none())
                            }
                        }

                        IteratorState::MapPairs { items, index } => {
                            if *index < items.len() {
                                let (k, v) = items[*index].clone();
                                *index += 1;
                                Ok(Value::some(Value::array(vec![k.to_value(), v])))
                            } else {
                                Ok(Value::none())
                            }
                        }
                    }
                }

                _ => Err(LustError::RuntimeError {
                    message: format!("Iterator has no method '{}'", method_name),
                }),
            },
            Value::Float(f) => match method_name {
                "to_int" => {
                    if !args.is_empty() {
                        return Err(LustError::RuntimeError {
                            message: "to_int() takes no arguments".to_string(),
                        });
                    }

                    Ok(Value::Int(int_from_float(*f)))
                }

                "floor" => {
                    if !args.is_empty() {
                        return Err(LustError::RuntimeError {
                            message: "floor() takes no arguments".to_string(),
                        });
                    }

                    Ok(Value::Float(float_floor(*f)))
                }

                "ceil" => {
                    if !args.is_empty() {
                        return Err(LustError::RuntimeError {
                            message: "ceil() takes no arguments".to_string(),
                        });
                    }

                    Ok(Value::Float(float_ceil(*f)))
                }

                "round" => {
                    if !args.is_empty() {
                        return Err(LustError::RuntimeError {
                            message: "round() takes no arguments".to_string(),
                        });
                    }

                    Ok(Value::Float(float_round(*f)))
                }

                "sqrt" => {
                    if !args.is_empty() {
                        return Err(LustError::RuntimeError {
                            message: "sqrt() takes no arguments".to_string(),
                        });
                    }

                    if *f < 0.0 {
                        return Err(LustError::RuntimeError {
                            message: "sqrt() requires a non-negative number".to_string(),
                        });
                    }

                    Ok(Value::Float(float_sqrt(*f)))
                }

                "abs" => {
                    if !args.is_empty() {
                        return Err(LustError::RuntimeError {
                            message: "abs() takes no arguments".to_string(),
                        });
                    }

                    Ok(Value::Float(float_abs(*f)))
                }

                "min" => {
                    if args.len() != 1 {
                        return Err(LustError::RuntimeError {
                            message: "min() requires 1 argument (other)".to_string(),
                        });
                    }
                    let other = args[0]
                        .as_float()
                        .or_else(|| args[0].as_int().map(float_from_int))
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "min() other must be a number".to_string(),
                        })?;
                    Ok(Value::Float(f.min(other)))
                }

                "max" => {
                    if args.len() != 1 {
                        return Err(LustError::RuntimeError {
                            message: "max() requires 1 argument (other)".to_string(),
                        });
                    }
                    let other = args[0]
                        .as_float()
                        .or_else(|| args[0].as_int().map(float_from_int))
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "max() other must be a number".to_string(),
                        })?;
                    Ok(Value::Float(f.max(other)))
                }

                "clamp" => {
                    if args.len() != 2 {
                        return Err(LustError::RuntimeError {
                            message: "clamp() requires 2 arguments (min, max)".to_string(),
                        });
                    }

                    let min = args[0].as_float().ok_or_else(|| LustError::RuntimeError {
                        message: "clamp() min must be a number".to_string(),
                    })?;
                    let max = args[1].as_float().ok_or_else(|| LustError::RuntimeError {
                        message: "clamp() max must be a number".to_string(),
                    })?;
                    if min > max {
                        return Err(LustError::RuntimeError {
                            message: "clamp() min must be less than or equal to max".to_string(),
                        });
                    }

                    Ok(Value::Float(float_clamp(*f, min, max)))
                }

                _ => Err(LustError::RuntimeError {
                    message: format!("Float has no method '{}'", method_name),
                }),
            },
            Value::Int(i) => match method_name {
                "to_float" => {
                    if !args.is_empty() {
                        return Err(LustError::RuntimeError {
                            message: "to_float() takes no arguments".to_string(),
                        });
                    }

                    Ok(Value::Float(float_from_int(*i)))
                }

                "abs" => {
                    if !args.is_empty() {
                        return Err(LustError::RuntimeError {
                            message: "abs() takes no arguments".to_string(),
                        });
                    }

                    Ok(Value::Int(i.abs()))
                }

                "min" => {
                    if args.len() != 1 {
                        return Err(LustError::RuntimeError {
                            message: "min() requires 1 argument (other)".to_string(),
                        });
                    }
                    if let Some(other) = args[0].as_int() {
                        return Ok(Value::Int((*i).min(other)));
                    }
                    if let Some(other) = args[0].as_float() {
                        return Ok(Value::Float(float_from_int(*i).min(other)));
                    }
                    Err(LustError::RuntimeError {
                        message: "min() other must be a number".to_string(),
                    })
                }

                "max" => {
                    if args.len() != 1 {
                        return Err(LustError::RuntimeError {
                            message: "max() requires 1 argument (other)".to_string(),
                        });
                    }
                    if let Some(other) = args[0].as_int() {
                        return Ok(Value::Int((*i).max(other)));
                    }
                    if let Some(other) = args[0].as_float() {
                        return Ok(Value::Float(float_from_int(*i).max(other)));
                    }
                    Err(LustError::RuntimeError {
                        message: "max() other must be a number".to_string(),
                    })
                }

                "clamp" => {
                    if args.len() != 2 {
                        return Err(LustError::RuntimeError {
                            message: "clamp() requires 2 arguments (min, max)".to_string(),
                        });
                    }

                    let min = args[0].as_int().ok_or_else(|| LustError::RuntimeError {
                        message: "clamp() min must be an integer".to_string(),
                    })?;
                    let max = args[1].as_int().ok_or_else(|| LustError::RuntimeError {
                        message: "clamp() max must be an integer".to_string(),
                    })?;
                    if min > max {
                        return Err(LustError::RuntimeError {
                            message: "clamp() min must be less than or equal to max".to_string(),
                        });
                    }

                    Ok(Value::Int((*i).clamp(min, max)))
                }

                _ => Err(LustError::RuntimeError {
                    message: format!("Int has no method '{}'", method_name),
                }),
            },
            _ => Err(LustError::RuntimeError {
                message: format!(
                    "Type {:?} has no method '{}'",
                    object.type_of(),
                    method_name
                ),
            }),
        }
    }

    fn try_call_lua_dynamic_method(
        &mut self,
        receiver: &Value,
        method_name: &str,
        args: &[Value],
    ) -> Result<Option<Value>> {
        let key = Value::string(method_name.to_string());
        let method = self.lua_resolve_index(receiver, &key, 8)?;
        if matches!(method, Value::Nil) {
            return Ok(None);
        }

        let mut call_args = Vec::with_capacity(1 + args.len());
        call_args.push(receiver.clone());
        call_args.extend_from_slice(args);
        let result = self.call_value(&method, call_args)?;
        Ok(Some(result))
    }

    fn lua_resolve_index(&mut self, receiver: &Value, key: &Value, depth: usize) -> Result<Value> {
        if depth == 0 {
            return Ok(Value::Nil);
        }

        if let Some(direct) = self.lua_direct_index(receiver, key) {
            if !matches!(direct, Value::Nil) {
                return Ok(direct);
            }
        }

        let Some(indexer) = self.lua_index_metamethod(receiver) else {
            return Ok(Value::Nil);
        };
        if matches!(indexer, Value::Nil) {
            return Ok(Value::Nil);
        }

        let is_callable = matches!(
            indexer,
            Value::Function(_) | Value::Closure { .. } | Value::NativeFunction(_)
        ) || matches!(
            &indexer,
            Value::Enum {
                enum_name,
                variant,
                ..
            } if enum_name == "LuaValue" && variant == "Function"
        );

        if is_callable {
            self.call_value(&indexer, vec![receiver.clone(), key.clone()])
        } else {
            self.lua_resolve_index(&indexer, key, depth - 1)
        }
    }

    fn lua_direct_index(&self, receiver: &Value, key: &Value) -> Option<Value> {
        if let Value::Enum {
            enum_name,
            variant,
            values,
        } = receiver
        {
            if enum_name == "LuaValue" && variant == "Table" {
                if let Some(inner) = values.as_ref().and_then(|vals| vals.get(0)) {
                    return self.lua_direct_index(inner, key);
                }
            }
        }

        match receiver {
            Value::Struct { name, .. } if name == "LuaTable" => {
                let Some(Value::Map(map_rc)) = receiver.struct_get_field("table") else {
                    return None;
                };
                let raw_key = super::corelib::unwrap_lua_value(key.clone());
                let lookup_key = ValueKey::from_value(&raw_key);
                let value = map_rc.borrow().get(&lookup_key).cloned();
                value
            }
            Value::Map(map_rc) => {
                let raw_key = super::corelib::unwrap_lua_value(key.clone());
                let lookup_key = ValueKey::from_value(&raw_key);
                let value = map_rc.borrow().get(&lookup_key).cloned();
                value
            }
            _ => None,
        }
    }

    fn lua_index_metamethod(&self, receiver: &Value) -> Option<Value> {
        if let Value::Enum {
            enum_name,
            variant,
            values,
        } = receiver
        {
            if enum_name == "LuaValue" && variant == "Table" {
                if let Some(inner) = values.as_ref().and_then(|vals| vals.get(0)) {
                    return self.lua_index_metamethod(inner);
                }
            }
            if enum_name == "LuaValue" && variant == "Userdata" {
                if let Some(inner) = values.as_ref().and_then(|vals| vals.get(0)) {
                    return self.lua_index_metamethod(inner);
                }
            }
        }

        let Value::Struct { name, .. } = receiver else {
            return None;
        };
        let Some(Value::Map(meta_rc)) = receiver.struct_get_field("metamethods") else {
            return None;
        };
        if name != "LuaTable" && name != "LuaUserdata" {
            return None;
        }
        let value = meta_rc
            .borrow()
            .get(&ValueKey::string("__index".to_string()))
            .cloned();
        value
    }
}

fn lua_table_map_rc(table: &Value) -> Option<Rc<RefCell<LustMap>>> {
    match table.struct_get_field("table") {
        Some(Value::Map(map_rc)) => Some(map_rc),
        _ => None,
    }
}

fn lua_table_sequence_len(map: &LustMap) -> usize {
    let mut idx: LustInt = 1;
    loop {
        let key = ValueKey::from_value(&Value::Int(idx));
        if map.contains_key(&key) {
            idx += 1;
        } else {
            break;
        }
    }
    (idx - 1) as usize
}

fn lua_table_read_sequence(map: &LustMap) -> Vec<Value> {
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

fn lua_table_write_sequence(map_rc: &Rc<RefCell<LustMap>>, seq: &[Value]) {
    let mut map = map_rc.borrow_mut();
    let mut idx: LustInt = 1;
    loop {
        let key = ValueKey::from_value(&Value::Int(idx));
        if map.remove(&key).is_some() {
            idx += 1;
        } else {
            break;
        }
    }
    for (i, val) in seq.iter().enumerate() {
        let key = ValueKey::from_value(&Value::Int((i as LustInt) + 1));
        map.insert(key, val.clone());
    }
}

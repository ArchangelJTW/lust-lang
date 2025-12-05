use super::*;
use crate::bytecode::ValueKey;
use core::{array, ptr};
impl VM {
    pub(super) fn push_current_vm(&mut self) {
        let ptr = self as *mut VM;
        crate::vm::push_vm_ptr(ptr);
    }

    pub(super) fn pop_current_vm(&mut self) {
        crate::vm::pop_vm_ptr();
    }

    pub(super) fn run(&mut self) -> Result<Value> {
        loop {
            if let Some(target_depth) = self.call_until_depth {
                if self.call_stack.len() == target_depth {
                    if let Some(return_value) = self.pending_return_value.take() {
                        self.call_until_depth = None;
                        return Ok(return_value);
                    }
                }
            }

            if let Some(return_value) = self.pending_return_value.take() {
                if let Some(dest_reg) = self.pending_return_dest.take() {
                    self.set_register(dest_reg, return_value)?;
                }
            }

            if self.current_task.is_some() {
                if let Some(signal) = self.pending_task_signal.take() {
                    self.last_task_signal = Some(signal);
                    return Ok(Value::Nil);
                }
            }

            if self.call_stack.len() > self.max_stack_depth {
                return Err(LustError::RuntimeError {
                    message: "Stack overflow".to_string(),
                });
            }

            let executing_frame_index =
                self.call_stack
                    .len()
                    .checked_sub(1)
                    .ok_or_else(|| LustError::RuntimeError {
                        message: "Empty call stack".to_string(),
                    })?;
            let frame = self
                .call_stack
                .last_mut()
                .ok_or_else(|| LustError::RuntimeError {
                    message: "Empty call stack".to_string(),
                })?;
            let (instruction, ip_before_execution, func_idx) = {
                let func = &self.functions[frame.function_idx];
                if frame.ip >= func.chunk.instructions.len() {
                    self.call_stack.pop();
                    if self.call_stack.is_empty() {
                        return Ok(Value::Nil);
                    }

                    continue;
                }

                let instruction = func.chunk.instructions[frame.ip];
                frame.ip += 1;
                let ip_before_execution = frame.ip;
                let func_idx = frame.function_idx;
                (instruction, ip_before_execution, func_idx)
            };
            let (should_check_jit, loop_start_ip) = if let Instruction::Jump(offset) = instruction {
                if offset < 0 {
                    let current_frame = self.call_stack.last().unwrap();
                    let jump_target = (current_frame.ip as isize + offset as isize) as usize;
                    (true, jump_target)
                } else {
                    (false, 0)
                }
            } else {
                (false, 0)
            };
            if should_check_jit && self.jit.enabled {
                let count = self.jit.profiler.record_backedge(func_idx, loop_start_ip);
                if let Some(trace_id) = self
                    .jit
                    .root_traces
                    .get(&(func_idx, loop_start_ip))
                    .copied()
                {
                    let frame = self.call_stack.last_mut().unwrap();
                    let registers_ptr = frame.registers.as_mut_ptr();
                    let entry = self.jit.get_trace(trace_id).map(|t| t.entry);
                    if let Some(entry_fn) = entry {
                        crate::jit::log(|| {
                            format!(
                                "▶️  JIT: Executing trace #{} at func {} ip {}",
                                trace_id.0, func_idx, loop_start_ip
                            )
                        });

                        // Capture RSP before and after to detect stack leaks
                        let rsp_before: usize;
                        unsafe { std::arch::asm!("mov {}, rsp", out(reg) rsp_before) };

                        let result = entry_fn(registers_ptr, self as *mut VM, ptr::null());

                        let rsp_after: usize;
                        unsafe { std::arch::asm!("mov {}, rsp", out(reg) rsp_after) };

                        let rsp_diff = rsp_after as isize - rsp_before as isize;
                        crate::jit::log(|| {
                            format!("🎯 JIT: Trace #{} execution result: {} (RSP before: {:x}, after: {:x}, diff: {})",
                            trace_id.0, result, rsp_before, rsp_after, rsp_diff)
                        });

                        if result == 0 {
                            if let Some(frame) = self.call_stack.last_mut() {
                                frame.ip = loop_start_ip;
                            }

                            continue;
                        } else if result > 0 {
                            let guard_index = (result - 1) as usize;
                            let side_trace_id = self
                                .jit
                                .get_trace(trace_id)
                                .and_then(|t| t.guards.get(guard_index))
                                .and_then(|g| g.side_trace);
                            if let Some(side_trace_id) = side_trace_id {
                                crate::jit::log(|| {
                                    format!(
                                        "🌳 JIT: Executing side trace #{} for guard #{}",
                                        side_trace_id.0, guard_index
                                    )
                                });
                                let frame = self.call_stack.last_mut().unwrap();
                                let registers_ptr = frame.registers.as_mut_ptr();
                                let side_entry = self.jit.get_trace(side_trace_id).map(|t| t.entry);
                                if let Some(side_entry_fn) = side_entry {
                                    let side_result =
                                        side_entry_fn(registers_ptr, self as *mut VM, ptr::null());
                                    if side_result == 0 {
                                        crate::jit::log(|| {
                                            format!(
                                                "✅ JIT: Side trace #{} executed successfully",
                                                side_trace_id.0
                                            )
                                        });
                                    } else {
                                        crate::jit::log(|| {
                                            format!(
                                                "⚠️  JIT: Side trace #{} failed, falling back to interpreter",
                                                side_trace_id.0
                                            )
                                        });
                                    }
                                }
                            } else {
                                if let Some(trace) = self.jit.get_trace(trace_id) {
                                    if let Some(g) = trace.guards.get(guard_index) {
                                        if g.bailout_ip != 0 {
                                            continue;
                                        }
                                    }
                                }

                                self.handle_guard_failure(trace_id, guard_index, func_idx)?;
                                self.jit.root_traces.remove(&(func_idx, loop_start_ip));
                            }
                        } else {
                            crate::jit::log(|| {
                                "⚠️  JIT: Trace execution failed (unknown error)".to_string()
                            });
                            if let Some(frame) = self.call_stack.last_mut() {
                                frame.ip = loop_start_ip;
                            }

                            self.jit.root_traces.remove(&(func_idx, loop_start_ip));
                        }
                    }
                } else {
                    let is_side_trace = self.side_trace_context.is_some();
                    if is_side_trace {
                        if let Some(recorder) = &self.trace_recorder {
                            if !recorder.is_recording() {
                                crate::jit::log(|| {
                                    format!(
                                        "📝 JIT: Trace recording complete - {} ops recorded",
                                        recorder.trace.ops.len()
                                    )
                                });
                                let recorder = self.trace_recorder.take().unwrap();
                                let mut trace = recorder.finish();
                                let side_trace_ctx = self.side_trace_context.take().unwrap();
                                let mut optimizer = TraceOptimizer::new();
                                let hoisted_constants = optimizer.optimize(&mut trace);
                                let (parent_trace_id, guard_index) = side_trace_ctx;
                                crate::jit::log(|| {
                                    format!(
                                        "⚙️  JIT: Compiling side trace (parent: #{}, guard: {})...",
                                        parent_trace_id.0, guard_index
                                    )
                                });
                                let trace_id = self.jit.alloc_trace_id();
                                match JitCompiler::new().compile_trace(
                                    &trace,
                                    trace_id,
                                    Some(parent_trace_id),
                                    hoisted_constants.clone(),
                                ) {
                                    Ok(compiled_trace) => {
                                        crate::jit::log(|| {
                                            format!(
                                                "✅ JIT: Side trace #{} compiled successfully!",
                                                trace_id.0
                                            )
                                        });
                                        if let Some(parent) =
                                            self.jit.get_trace_mut(parent_trace_id)
                                        {
                                            if guard_index < parent.guards.len() {
                                                parent.guards[guard_index].side_trace =
                                                    Some(trace_id);
                                                crate::jit::log(|| {
                                                    format!(
                                                        "🔗 JIT: Linked side trace #{} to parent trace #{} guard #{}",
                                                        trace_id.0, parent_trace_id.0, guard_index
                                                    )
                                                });
                                            }
                                        }

                                        self.jit.store_side_trace(compiled_trace);
                                    }

                                    Err(e) => {
                                        crate::jit::log(|| {
                                            format!("❌ JIT: Side trace compilation failed: {}", e)
                                        });
                                    }
                                }
                            }
                        }
                    } else {
                        if let Some(recorder) = &mut self.trace_recorder {
                            if recorder.is_recording() && count > crate::jit::HOT_THRESHOLD + 1 {
                                crate::jit::log(|| {
                                    format!(
                                        "📝 JIT: Trace recording complete - {} ops recorded",
                                        recorder.trace.ops.len()
                                    )
                                });
                                let recorder = self.trace_recorder.take().unwrap();
                                let mut trace = recorder.finish();
                                let mut optimizer = TraceOptimizer::new();
                                let hoisted_constants = optimizer.optimize(&mut trace);
                                crate::jit::log(|| "⚙️  JIT: Compiling root trace...".to_string());
                                let trace_id = self.jit.alloc_trace_id();
                                match JitCompiler::new().compile_trace(
                                    &trace,
                                    trace_id,
                                    None,
                                    hoisted_constants.clone(),
                                ) {
                                    Ok(compiled_trace) => {
                                        crate::jit::log(|| {
                                            format!(
                                                "✅ JIT: Trace #{} compiled successfully!",
                                                trace_id.0
                                            )
                                        });
                                        crate::jit::log(|| {
                                            "🚀 JIT: Future iterations will use native code!"
                                                .to_string()
                                        });
                                        self.jit.store_root_trace(
                                            func_idx,
                                            loop_start_ip,
                                            compiled_trace,
                                        );
                                    }

                                    Err(e) => {
                                        crate::jit::log(|| {
                                            format!("❌ JIT: Trace compilation failed: {}", e)
                                        });
                                    }
                                }
                            }
                        }

                        if count == crate::jit::HOT_THRESHOLD + 1 {
                            crate::jit::log(|| {
                                format!(
                                    "🔥 JIT: Hot loop detected at func {} ip {} - starting trace recording!",
                                    func_idx, loop_start_ip
                                )
                            });
                            let mut recorder =
                                TraceRecorder::new(func_idx, loop_start_ip, MAX_TRACE_LENGTH);
                            // Specialize loop-invariant values at trace entry
                            {
                                let frame = self.call_stack.last().unwrap();
                                let func = &self.functions[func_idx];
                                recorder.specialize_trace_inputs(&frame.registers, func);
                            }
                            self.trace_recorder = Some(recorder);
                            self.skip_next_trace_record = true;
                        }
                    }
                }
            }

            match instruction {
                Instruction::LoadNil(dest) => {
                    self.set_register(dest, Value::Nil)?;
                }

                Instruction::LoadBool(dest, value) => {
                    self.set_register(dest, Value::Bool(value))?;
                }

                Instruction::LoadConst(dest, const_idx) => {
                    let constant = {
                        let func = &self.functions[func_idx];
                        func.chunk.constants[const_idx as usize].clone()
                    };
                    self.set_register(dest, constant)?;
                }

                Instruction::LoadGlobal(dest, name_idx) => {
                    let func = &self.functions[self.call_stack.last().unwrap().function_idx];
                    let name = func.chunk.constants[name_idx as usize]
                        .as_string()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Global name must be a string".to_string(),
                        })?;
                    if let Some(value) = self.globals.get(name) {
                        self.set_register(dest, value.clone())?;
                    } else if let Some(value) = self.natives.get(name) {
                        self.set_register(dest, value.clone())?;
                    } else {
                        if let Some((_, value)) =
                            self.globals.iter().find(|(key, _)| key.as_str() == name)
                        {
                            self.set_register(dest, value.clone())?;
                        } else if let Some((_, value)) =
                            self.natives.iter().find(|(key, _)| key.as_str() == name)
                        {
                            self.set_register(dest, value.clone())?;
                        } else {
                            return Err(LustError::RuntimeError {
                                message: format!("Undefined global: {}", name),
                            });
                        }
                    }
                }

                Instruction::StoreGlobal(name_idx, src) => {
                    let func = &self.functions[self.call_stack.last().unwrap().function_idx];
                    let name = func.chunk.constants[name_idx as usize]
                        .as_string()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Global name must be a string".to_string(),
                        })?;
                    let value = self.get_register(src)?.clone();
                    self.globals.insert(name.to_string(), value);
                }

                Instruction::Move(dest, src) => {
                    let value = self.get_register(src)?.clone();
                    self.set_register(dest, value)?;
                }

                Instruction::Add(dest, lhs, rhs) => {
                    self.binary_op(dest, lhs, rhs, |l, r| match (l, r) {
                        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
                        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
                        (Value::Int(a), Value::Float(b)) => {
                            Ok(Value::Float(float_from_int(*a) + *b))
                        }
                        (Value::Float(a), Value::Int(b)) => {
                            Ok(Value::Float(*a + float_from_int(*b)))
                        }
                        _ => Err(LustError::RuntimeError {
                            message: format!("Cannot add {:?} and {:?}", l, r),
                        }),
                    })?;
                }

                Instruction::Sub(dest, lhs, rhs) => {
                    self.binary_op(dest, lhs, rhs, |l, r| match (l, r) {
                        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
                        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
                        (Value::Int(a), Value::Float(b)) => {
                            Ok(Value::Float(float_from_int(*a) - *b))
                        }
                        (Value::Float(a), Value::Int(b)) => {
                            Ok(Value::Float(*a - float_from_int(*b)))
                        }
                        _ => Err(LustError::RuntimeError {
                            message: format!("Cannot subtract {:?} and {:?}", l, r),
                        }),
                    })?;
                }

                Instruction::Mul(dest, lhs, rhs) => {
                    self.binary_op(dest, lhs, rhs, |l, r| match (l, r) {
                        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
                        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
                        (Value::Int(a), Value::Float(b)) => {
                            Ok(Value::Float(float_from_int(*a) * *b))
                        }
                        (Value::Float(a), Value::Int(b)) => {
                            Ok(Value::Float(*a * float_from_int(*b)))
                        }
                        _ => Err(LustError::RuntimeError {
                            message: format!("Cannot multiply {:?} and {:?}", l, r),
                        }),
                    })?;
                }

                Instruction::Div(dest, lhs, rhs) => {
                    self.binary_op(dest, lhs, rhs, |l, r| match (l, r) {
                        (Value::Int(a), Value::Int(b)) => {
                            if *b == 0 {
                                Err(LustError::RuntimeError {
                                    message: "Division by zero".to_string(),
                                })
                            } else {
                                Ok(Value::Int(a / b))
                            }
                        }

                        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(*a / *b)),
                        (Value::Int(a), Value::Float(b)) => {
                            Ok(Value::Float(float_from_int(*a) / *b))
                        }
                        (Value::Float(a), Value::Int(b)) => {
                            Ok(Value::Float(*a / float_from_int(*b)))
                        }
                        _ => Err(LustError::RuntimeError {
                            message: format!("Cannot divide {:?} and {:?}", l, r),
                        }),
                    })?;
                }

                Instruction::Mod(dest, lhs, rhs) => {
                    self.binary_op(dest, lhs, rhs, |l, r| match (l, r) {
                        (Value::Int(a), Value::Int(b)) => {
                            if *b == 0 {
                                Err(LustError::RuntimeError {
                                    message: "Modulo by zero".to_string(),
                                })
                            } else {
                                Ok(Value::Int(a % b))
                            }
                        }

                        (Value::Float(a), Value::Float(b)) => {
                            if *b == 0.0 {
                                Err(LustError::RuntimeError {
                                    message: "Modulo by zero".to_string(),
                                })
                            } else {
                                Ok(Value::Float(a % b))
                            }
                        }

                        (Value::Int(a), Value::Float(b)) => {
                            if *b == 0.0 {
                                Err(LustError::RuntimeError {
                                    message: "Modulo by zero".to_string(),
                                })
                            } else {
                                Ok(Value::Float(float_from_int(*a) % *b))
                            }
                        }

                        (Value::Float(a), Value::Int(b)) => {
                            if *b == 0 {
                                Err(LustError::RuntimeError {
                                    message: "Modulo by zero".to_string(),
                                })
                            } else {
                                Ok(Value::Float(*a % float_from_int(*b)))
                            }
                        }

                        _ => Err(LustError::RuntimeError {
                            message: format!("Cannot modulo {:?} and {:?}", l, r),
                        }),
                    })?;
                }

                Instruction::Neg(dest, src) => {
                    let value = self.get_register(src)?;
                    let result = match value {
                        Value::Int(i) => Value::Int(-i),
                        Value::Float(f) => Value::Float(-f),
                        _ => {
                            return Err(LustError::RuntimeError {
                                message: format!("Cannot negate {:?}", value),
                            })
                        }
                    };
                    self.set_register(dest, result)?;
                }

                Instruction::Eq(dest, lhs, rhs) => {
                    let left = self.get_register(lhs)?;
                    let right = self.get_register(rhs)?;
                    self.set_register(dest, Value::Bool(left == right))?;
                }

                Instruction::Ne(dest, lhs, rhs) => {
                    let left = self.get_register(lhs)?;
                    let right = self.get_register(rhs)?;
                    self.set_register(dest, Value::Bool(left != right))?;
                }

                Instruction::Lt(dest, lhs, rhs) => {
                    self.comparison_op(dest, lhs, rhs, |l, r| l < r)?;
                }

                Instruction::Le(dest, lhs, rhs) => {
                    self.comparison_op(dest, lhs, rhs, |l, r| l <= r)?;
                }

                Instruction::Gt(dest, lhs, rhs) => {
                    self.comparison_op(dest, lhs, rhs, |l, r| l > r)?;
                }

                Instruction::Ge(dest, lhs, rhs) => {
                    self.comparison_op(dest, lhs, rhs, |l, r| l >= r)?;
                }

                Instruction::And(dest, lhs, rhs) => {
                    let left = self.get_register(lhs)?;
                    let right = self.get_register(rhs)?;
                    let result = Value::Bool(left.is_truthy() && right.is_truthy());
                    self.set_register(dest, result)?;
                }

                Instruction::Or(dest, lhs, rhs) => {
                    let left = self.get_register(lhs)?;
                    let right = self.get_register(rhs)?;
                    let result = Value::Bool(left.is_truthy() || right.is_truthy());
                    self.set_register(dest, result)?;
                }

                Instruction::Not(dest, src) => {
                    let value = self.get_register(src)?;
                    self.set_register(dest, Value::Bool(!value.is_truthy()))?;
                }

                Instruction::Jump(offset) => {
                    let frame = self.call_stack.last_mut().unwrap();
                    frame.ip = (frame.ip as isize + offset as isize) as usize;
                }

                Instruction::JumpIf(cond, offset) => {
                    let condition = self.get_register(cond)?;
                    if condition.is_truthy() {
                        let frame = self.call_stack.last_mut().unwrap();
                        frame.ip = (frame.ip as isize + offset as isize) as usize;
                    }
                }

                Instruction::JumpIfNot(cond, offset) => {
                    let condition = self.get_register(cond)?;
                    if !condition.is_truthy() {
                        let frame = self.call_stack.last_mut().unwrap();
                        frame.ip = (frame.ip as isize + offset as isize) as usize;
                    }
                }

                Instruction::Call(func_reg, first_arg, arg_count, dest_reg) => {
                    let func_value = self.get_register(func_reg)?.clone();
                    match func_value {
                        Value::Function(func_idx) => {
                            let mut args = Vec::new();
                            for i in 0..arg_count {
                                args.push(self.get_register(first_arg + i)?.clone());
                            }

                            let mut frame = CallFrame {
                                function_idx: func_idx,
                                ip: 0,
                                registers: array::from_fn(|_| Value::Nil),
                                base_register: 0,
                                return_dest: Some(dest_reg),
                                upvalues: Vec::new(),
                            };
                            for (i, arg) in args.into_iter().enumerate() {
                                frame.registers[i] = arg;
                            }

                            self.call_stack.push(frame);
                        }

                        Value::Closure {
                            function_idx: func_idx,
                            upvalues,
                        } => {
                            let mut args = Vec::new();
                            for i in 0..arg_count {
                                args.push(self.get_register(first_arg + i)?.clone());
                            }

                            let upvalue_values: Vec<Value> =
                                upvalues.iter().map(|uv| uv.get()).collect();
                            let mut frame = CallFrame {
                                function_idx: func_idx,
                                ip: 0,
                                registers: array::from_fn(|_| Value::Nil),
                                base_register: 0,
                                return_dest: Some(dest_reg),
                                upvalues: upvalue_values,
                            };
                            for (i, arg) in args.into_iter().enumerate() {
                                frame.registers[i] = arg;
                            }

                            self.call_stack.push(frame);
                        }

                        Value::NativeFunction(native_fn) => {
                            let mut args = Vec::new();
                            for i in 0..arg_count {
                                args.push(self.get_register(first_arg + i)?.clone());
                            }

                            self.push_current_vm();
                            let outcome = native_fn(&args);
                            self.pop_current_vm();
                            let outcome =
                                outcome.map_err(|e| LustError::RuntimeError { message: e })?;
                            self.handle_native_call_outcome(dest_reg, outcome)?;
                        }

                        _ => {
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "Cannot call non-function value: {:?}",
                                    func_value
                                ),
                            })
                        }
                    }
                }

                Instruction::Return(value_reg) => {
                    let return_value = if value_reg == 255 {
                        Value::Nil
                    } else {
                        self.get_register(value_reg)?.clone()
                    };
                    let return_dest = self.call_stack.last().unwrap().return_dest;
                    self.call_stack.pop();
                    if self.call_stack.is_empty() {
                        return Ok(return_value);
                    }

                    self.pending_return_value = Some(return_value);
                    self.pending_return_dest = return_dest;
                }

                Instruction::NewArray(dest, first_elem, count) => {
                    let mut elements = Vec::new();
                    for i in 0..count {
                        elements.push(self.get_register(first_elem + i)?.clone());
                    }

                    self.set_register(dest, Value::array(elements))?;
                }

                Instruction::TupleNew(dest, first_elem, count) => {
                    let mut elements = Vec::new();
                    for offset in 0..(count as usize) {
                        let value = self.get_register(first_elem + offset as u8)?.clone();
                        if let Value::Tuple(existing) = value {
                            elements.extend(existing.iter().cloned());
                        } else {
                            elements.push(value);
                        }
                    }

                    self.set_register(dest, Value::tuple(elements))?;
                }

                Instruction::TupleGet(dest, tuple_reg, index) => {
                    let tuple_value = self.get_register(tuple_reg)?.clone();
                    if let Value::Tuple(values) = tuple_value {
                        let idx = index as usize;
                        if idx >= values.len() {
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "Tuple index {} out of bounds (len {})",
                                    idx,
                                    values.len()
                                ),
                            });
                        }

                        let value = values[idx].clone();
                        self.set_register(dest, value)?;
                    } else {
                        return Err(LustError::RuntimeError {
                            message: "Attempted to destructure non-tuple value".to_string(),
                        });
                    }
                }

                Instruction::NewMap(dest) => {
                    self.set_register(dest, self.new_map_value())?;
                }

                Instruction::NewStruct(
                    dest,
                    name_idx,
                    first_field_name_idx,
                    first_field,
                    field_count,
                ) => {
                    let func = &self.functions[self.call_stack.last().unwrap().function_idx];
                    let struct_name = func.chunk.constants[name_idx as usize]
                        .as_string()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Struct name must be a string".to_string(),
                        })?
                        .to_string();
                    let mut fields = Vec::with_capacity(field_count as usize);
                    for i in 0..field_count {
                        let field_name = func.chunk.constants
                            [(first_field_name_idx + i as u16) as usize]
                            .as_string_rc()
                            .ok_or_else(|| LustError::RuntimeError {
                                message: "Field name must be a string".to_string(),
                            })?;
                        let value = self.get_register(first_field + i)?.clone();
                        fields.push((field_name, value));
                    }

                    let struct_value = self.instantiate_struct(&struct_name, fields)?;
                    self.set_register(dest, struct_value)?;
                }

                Instruction::NewEnumUnit(dest, enum_name_idx, variant_idx) => {
                    let func = &self.functions[self.call_stack.last().unwrap().function_idx];
                    let enum_name = func.chunk.constants[enum_name_idx as usize]
                        .as_string()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Enum name must be a string".to_string(),
                        })?
                        .to_string();
                    let variant_name = func.chunk.constants[variant_idx as usize]
                        .as_string()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Variant name must be a string".to_string(),
                        })?
                        .to_string();
                    self.set_register(dest, Value::enum_unit(enum_name, variant_name))?;
                }

                Instruction::NewEnumVariant(
                    dest,
                    enum_name_idx,
                    variant_idx,
                    first_value,
                    value_count,
                ) => {
                    let func = &self.functions[self.call_stack.last().unwrap().function_idx];
                    let enum_name = func.chunk.constants[enum_name_idx as usize]
                        .as_string()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Enum name must be a string".to_string(),
                        })?
                        .to_string();
                    let variant_name = func.chunk.constants[variant_idx as usize]
                        .as_string()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Variant name must be a string".to_string(),
                        })?
                        .to_string();
                    let mut values = Vec::new();
                    for i in 0..value_count {
                        values.push(self.get_register(first_value + i)?.clone());
                    }

                    self.set_register(dest, Value::enum_variant(enum_name, variant_name, values))?;
                }

                Instruction::IsEnumVariant(dest, value_reg, enum_name_idx, variant_idx) => {
                    let value = self.get_register(value_reg)?;
                    let func = &self.functions[self.call_stack.last().unwrap().function_idx];
                    let enum_name = func.chunk.constants[enum_name_idx as usize]
                        .as_string()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Enum name must be a string".to_string(),
                        })?;
                    let variant_name = func.chunk.constants[variant_idx as usize]
                        .as_string()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Variant name must be a string".to_string(),
                        })?;
                    let is_variant = value.is_enum_variant(enum_name, variant_name);
                    self.set_register(dest, Value::Bool(is_variant))?;
                }

                Instruction::GetEnumValue(dest, enum_reg, index) => {
                    let enum_value = self.get_register(enum_reg)?;
                    if let Some((_, _, Some(values))) = enum_value.as_enum() {
                        if (index as usize) < values.len() {
                            self.set_register(dest, values[index as usize].clone())?;
                        } else {
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "Enum value index {} out of bounds (has {} values)",
                                    index,
                                    values.len()
                                ),
                            });
                        }
                    } else {
                        return Err(LustError::RuntimeError {
                            message: "GetEnumValue requires an enum variant with values"
                                .to_string(),
                        });
                    }
                }

                Instruction::GetField(dest, obj, field_idx) => {
                    let object = self.get_register(obj)?;
                    let func = &self.functions[self.call_stack.last().unwrap().function_idx];
                    let field_name = func.chunk.constants[field_idx as usize]
                        .as_string_rc()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Field name must be a string".to_string(),
                        })?;
                    let value = match object {
                        Value::Struct { .. } => object
                            .struct_get_field_rc(&field_name)
                            .unwrap_or(Value::Nil),
                        Value::Map(map) => {
                            use crate::bytecode::ValueKey;
                            let key = ValueKey::from(field_name.clone());
                            map.borrow().get(&key).cloned().unwrap_or(Value::Nil)
                        }

                        _ => {
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "Cannot get field '{}' from {:?}",
                                    field_name.as_str(),
                                    object
                                ),
                            })
                        }
                    };
                    self.set_register(dest, value)?;
                }

                Instruction::SetField(obj_reg, field_idx, value_reg) => {
                    let object = self.get_register(obj_reg)?;
                    let value = self.get_register(value_reg)?.clone();
                    let func = &self.functions[self.call_stack.last().unwrap().function_idx];
                    let field_name = func.chunk.constants[field_idx as usize]
                        .as_string_rc()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Field name must be a string".to_string(),
                        })?;
                    let mut invalidate_key: Option<usize> = None;
                    match object {
                        Value::Struct { .. } => {
                            invalidate_key = Self::struct_cache_key(object);
                            object
                                .struct_set_field_rc(&field_name, value)
                                .map_err(|message| LustError::RuntimeError { message })?;
                        }

                        Value::Map(map) => {
                            use crate::bytecode::ValueKey;
                            let key = ValueKey::from(field_name.clone());
                            map.borrow_mut().insert(key, value);
                        }

                        _ => {
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "Cannot set field '{}' on {:?}",
                                    field_name.as_str(),
                                    object
                                ),
                            })
                        }
                    }

                    if let Some(key) = invalidate_key {
                        self.struct_tostring_cache.remove(&key);
                    }
                }

                Instruction::Concat(dest, lhs, rhs) => {
                    let (left, right) = {
                        let frame =
                            self.call_stack
                                .last_mut()
                                .ok_or_else(|| LustError::RuntimeError {
                                    message: "Empty call stack".to_string(),
                                })?;
                        let left = frame.registers[lhs as usize].clone();
                        let right = frame.registers[rhs as usize].clone();
                        (left, right)
                    };
                    let left_str = self.value_to_string_for_concat(&left)?;
                    let right_str = self.value_to_string_for_concat(&right)?;
                    let mut combined = String::with_capacity(left_str.len() + right_str.len());
                    combined.push_str(left_str.as_ref());
                    combined.push_str(right_str.as_ref());
                    let result = Value::string(combined);
                    self.set_register(dest, result)?;
                }

                Instruction::GetIndex(dest, array_reg, index_reg) => {
                    let collection = self.get_register(array_reg)?.clone();
                    let index = self.get_register(index_reg)?.clone();
                    let result = match collection {
                        Value::Array(arr) => {
                            let idx = index.as_int().ok_or_else(|| LustError::RuntimeError {
                                message: "Array index must be an integer".to_string(),
                            })?;
                            let borrowed = arr.borrow();
                            if idx < 0 || idx as usize >= borrowed.len() {
                                return Err(LustError::RuntimeError {
                                    message: format!(
                                        "Array index {} out of bounds (length: {})",
                                        idx,
                                        borrowed.len()
                                    ),
                                });
                            }

                            borrowed[idx as usize].clone()
                        }

                        Value::Map(map) => {
                            let key = self.make_hash_key(&index)?;
                            map.borrow().get(&key).cloned().unwrap_or(Value::Nil)
                        }

                        _ => {
                            return Err(LustError::RuntimeError {
                                message: format!("Cannot index {:?}", collection.type_of()),
                            })
                        }
                    };
                    self.set_register(dest, result)?;
                }

                Instruction::ArrayLen(dest, array_reg) => {
                    let collection = self.get_register(array_reg)?;
                    match collection {
                        Value::Array(arr) => {
                            let len = int_from_usize(arr.borrow().len());
                            self.set_register(dest, Value::Int(len))?;
                        }

                        _ => {
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "ArrayLen requires array, got {:?}",
                                    collection.type_of()
                                ),
                            });
                        }
                    }
                }

                Instruction::CallMethod(
                    obj_reg,
                    method_name_idx,
                    first_arg,
                    arg_count,
                    dest_reg,
                ) => {
                    let object = self.get_register(obj_reg)?.clone();
                    let method_name = {
                        let func = &self.functions[self.call_stack.last().unwrap().function_idx];
                        func.chunk.constants[method_name_idx as usize]
                            .as_string()
                            .ok_or_else(|| LustError::RuntimeError {
                                message: "Method name must be a string".to_string(),
                            })?
                            .to_string()
                    };
                    if let Value::Struct {
                        name: struct_name, ..
                    } = &object
                    {
                        let mangled_name = format!("{}:{}", struct_name, method_name);
                        if let Some(func_idx) =
                            self.functions.iter().position(|f| f.name == mangled_name)
                        {
                            let mut frame = CallFrame {
                                function_idx: func_idx,
                                ip: 0,
                                registers: array::from_fn(|_| Value::Nil),
                                base_register: 0,
                                return_dest: Some(dest_reg),
                                upvalues: Vec::new(),
                            };
                            frame.registers[0] = object.clone();
                            for i in 0..arg_count {
                                frame.registers[(i + 1) as usize] =
                                    self.get_register(first_arg + i)?.clone();
                            }

                            self.call_stack.push(frame);
                            continue;
                        }

                        let mut candidate_names = vec![mangled_name.clone()];
                        if let Some(simple) = struct_name.rsplit(|c| c == '.' || c == ':').next() {
                            candidate_names.push(format!("{}:{}", simple, method_name));
                        }

                        let mut handled = false;
                        for candidate in candidate_names {
                            let mut resolved = None;
                            for variant in [candidate.clone(), candidate.replace('.', "::")] {
                                if let Some((_name, value)) =
                                    self.globals.iter().find(|(name, _)| *name == &variant)
                                {
                                    resolved = Some(value.clone());
                                    break;
                                }
                                if let Some((_name, value)) =
                                    self.natives.iter().find(|(name, _)| *name == &variant)
                                {
                                    resolved = Some(value.clone());
                                    break;
                                }
                            }

                            if let Some(global_func) = resolved {
                                let mut call_args = Vec::with_capacity(1 + arg_count as usize);
                                call_args.push(object.clone());
                                for i in 0..arg_count {
                                    call_args.push(self.get_register(first_arg + i)?.clone());
                                }
                                let result = self.call_value(&global_func, call_args)?;
                                self.set_register(dest_reg, result)?;
                                handled = true;
                                break;
                            }
                        }
                        if handled {
                            continue;
                        }
                    }

                    let mut args = Vec::new();
                    for i in 0..arg_count {
                        args.push(self.get_register(first_arg + i)?.clone());
                    }

                    let result = self.call_builtin_method(&object, &method_name, args)?;
                    self.set_register(dest_reg, result)?;
                }

                Instruction::Closure(dest, func_idx, first_upvalue_reg, upvalue_count) => {
                    use crate::bytecode::Upvalue;
                    let mut upvalues = Vec::new();
                    for i in 0..upvalue_count {
                        let value = self.get_register(first_upvalue_reg + i)?.clone();
                        upvalues.push(Upvalue::new(value));
                    }

                    let closure = Value::Closure {
                        function_idx: func_idx as usize,
                        upvalues: Rc::new(upvalues),
                    };
                    self.set_register(dest, closure)?;
                }

                Instruction::LoadUpvalue(dest, upvalue_idx) => {
                    let frame = self
                        .call_stack
                        .last()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Empty call stack".to_string(),
                        })?;
                    if (upvalue_idx as usize) < frame.upvalues.len() {
                        let value = frame.upvalues[upvalue_idx as usize].clone();
                        self.set_register(dest, value)?;
                    } else {
                        return Err(LustError::RuntimeError {
                            message: format!(
                                "Upvalue index {} out of bounds (have {} upvalues)",
                                upvalue_idx,
                                frame.upvalues.len()
                            ),
                        });
                    }
                }

                Instruction::StoreUpvalue(upvalue_idx, src) => {
                    let value = self.get_register(src)?.clone();
                    let frame =
                        self.call_stack
                            .last_mut()
                            .ok_or_else(|| LustError::RuntimeError {
                                message: "Empty call stack".to_string(),
                            })?;
                    if (upvalue_idx as usize) < frame.upvalues.len() {
                        frame.upvalues[upvalue_idx as usize] = value;
                    } else {
                        return Err(LustError::RuntimeError {
                            message: format!(
                                "Upvalue index {} out of bounds (have {} upvalues)",
                                upvalue_idx,
                                frame.upvalues.len()
                            ),
                        });
                    }
                }

                Instruction::SetIndex(collection_reg, index_reg, value_reg) => {
                    let collection = self.get_register(collection_reg)?.clone();
                    let index = self.get_register(index_reg)?.clone();
                    let value = self.get_register(value_reg)?.clone();
                    match collection {
                        Value::Array(arr) => {
                            let idx = index.as_int().ok_or_else(|| LustError::RuntimeError {
                                message: "Array index must be an integer".to_string(),
                            })?;
                            let mut borrowed = arr.borrow_mut();
                            if idx < 0 || idx as usize >= borrowed.len() {
                                return Err(LustError::RuntimeError {
                                    message: format!(
                                        "Array index {} out of bounds (length: {})",
                                        idx,
                                        borrowed.len()
                                    ),
                                });
                            }

                            borrowed[idx as usize] = value;
                        }

                        Value::Map(map) => {
                            let key = self.make_hash_key(&index)?;
                            map.borrow_mut().insert(key, value);
                        }

                        _ => {
                            return Err(LustError::RuntimeError {
                                message: format!("Cannot index {:?}", collection.type_of()),
                            })
                        }
                    }
                }

                Instruction::TypeIs(dest, value_reg, type_name_idx) => {
                    let value = self.get_register(value_reg)?.clone();
                    let func = &self.functions[self.call_stack.last().unwrap().function_idx];
                    let type_name = func.chunk.constants[type_name_idx as usize]
                        .as_string()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Type name must be a string".to_string(),
                        })?
                        .to_string();
                    let matches = self.value_is_type(&value, &type_name);
                    self.set_register(dest, Value::Bool(matches))?;
                }
            }

            if self.jit.enabled {
                if let Some(recorder) = &mut self.trace_recorder {
                    if recorder.is_recording() {
                        if self.skip_next_trace_record {
                            self.skip_next_trace_record = false;
                        } else {
                            let function = &self.functions[func_idx];
                            let registers_opt =
                                if let Some(frame) = self.call_stack.get(executing_frame_index) {
                                    Some(&frame.registers)
                                } else if executing_frame_index > 0 {
                                    self.call_stack
                                        .get(executing_frame_index - 1)
                                        .map(|frame| &frame.registers)
                                } else {
                                    None
                                };
                            if let Some(registers) = registers_opt {
                                if let Err(e) = recorder.record_instruction(
                                    instruction,
                                    ip_before_execution,
                                    registers,
                                    function,
                                    func_idx,
                                    &self.functions,
                                ) {
                                    crate::jit::log(|| format!("⚠️  JIT: {}", e));
                                    self.trace_recorder = None;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pub(super) fn binary_op<F>(
        &mut self,
        dest: Register,
        lhs: Register,
        rhs: Register,
        op: F,
    ) -> Result<()>
    where
        F: FnOnce(&Value, &Value) -> Result<Value>,
    {
        let left = self.get_register(lhs)?;
        let right = self.get_register(rhs)?;
        let result = op(left, right)?;
        self.set_register(dest, result)
    }

    pub(super) fn comparison_op<F>(
        &mut self,
        dest: Register,
        lhs: Register,
        rhs: Register,
        op: F,
    ) -> Result<()>
    where
        F: FnOnce(LustFloat, LustFloat) -> bool,
    {
        let left = self.get_register(lhs)?;
        let right = self.get_register(rhs)?;
        let result = match (left, right) {
            (Value::Int(a), Value::Int(b)) => op(float_from_int(*a), float_from_int(*b)),
            (Value::Float(a), Value::Float(b)) => op(*a, *b),
            (Value::Int(a), Value::Float(b)) => op(float_from_int(*a), *b),
            (Value::Float(a), Value::Int(b)) => op(*a, float_from_int(*b)),
            _ => {
                return Err(LustError::RuntimeError {
                    message: format!("Cannot compare {:?} and {:?}", left, right),
                })
            }
        };
        self.set_register(dest, Value::Bool(result))
    }

    pub(super) fn value_is_type(&self, value: &Value, type_name: &str) -> bool {
        if let Some(matches) = self.match_function_type(value, type_name) {
            return matches;
        }

        let value_type_name = match value {
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::String(_) => "string",
            Value::Bool(_) => "bool",
            Value::Nil => "nil",
            Value::Array(_) => "Array",
            Value::Tuple(_) => "Tuple",
            Value::Map(_) => "Map",
            Value::Struct { name, .. } => name.as_str(),
            Value::WeakStruct(weak) => weak.struct_name(),
            Value::Enum { enum_name, .. } => enum_name.as_str(),
            Value::Function(_) | Value::NativeFunction(_) | Value::Closure { .. } => "function",
            Value::Iterator(_) => "Iterator",
            Value::Task(_) => "task",
        };
        if value_type_name == type_name {
            return true;
        }

        if type_name.starts_with("Array") && matches!(value, Value::Array(_)) {
            return true;
        }

        if type_name.starts_with("Map") && matches!(value, Value::Map(_)) {
            return true;
        }

        if type_name.starts_with("Tuple") && matches!(value, Value::Tuple(_)) {
            return true;
        }

        if type_name == "Option"
            && matches!(value, Value::Enum { enum_name, .. } if enum_name == "Option")
        {
            return true;
        }

        if type_name == "Result"
            && matches!(value, Value::Enum { enum_name, .. } if enum_name == "Result")
        {
            return true;
        }

        if type_name == "unknown" {
            return true;
        }

        if let Some(_) = self
            .trait_impls
            .get(&(value_type_name.to_string(), type_name.to_string()))
        {
            return true;
        }

        false
    }

    fn value_trait_name(&self, value: &Value) -> String {
        match value {
            Value::Int(_) => "int".to_string(),
            Value::Float(_) => "float".to_string(),
            Value::String(_) => "string".to_string(),
            Value::Bool(_) => "bool".to_string(),
            Value::Nil => "nil".to_string(),
            Value::Array(_) => "Array".to_string(),
            Value::Tuple(_) => "Tuple".to_string(),
            Value::Map(_) => "Map".to_string(),
            Value::Struct { name, .. } => name.clone(),
            Value::WeakStruct(weak) => weak.struct_name().to_string(),
            Value::Enum { enum_name, .. } => enum_name.clone(),
            Value::Function(_) | Value::NativeFunction(_) | Value::Closure { .. } => {
                "function".to_string()
            }
            Value::Iterator(_) => "Iterator".to_string(),
            Value::Task(_) => "task".to_string(),
        }
    }

    fn invoke_hashkey(&mut self, value: &Value, type_name: &str) -> Result<Value> {
        let mut candidates = vec![format!("{}:{}", type_name, HASH_KEY_METHOD)];
        if let Some(last) = type_name.rsplit('.').next() {
            if last != type_name {
                candidates.push(format!("{}:{}", last, HASH_KEY_METHOD));
            }
        }

        for candidate in candidates {
            if let Some(idx) = self.functions.iter().position(|f| f.name == candidate) {
                return self.call_value(&Value::Function(idx), vec![value.clone()]);
            }
        }

        Err(LustError::RuntimeError {
            message: format!(
                "HashKey trait declared but method '{}' not found for type '{}'",
                HASH_KEY_METHOD, type_name
            ),
        })
    }

    pub(super) fn make_hash_key(&mut self, value: &Value) -> Result<ValueKey> {
        let type_name = self.value_trait_name(value);
        if self.type_has_hashkey(&type_name) {
            let hashed = self.invoke_hashkey(value, &type_name)?;
            Ok(ValueKey::with_hashed(value.clone(), hashed))
        } else {
            Ok(ValueKey::from_value(value))
        }
    }

    fn match_function_type(&self, value: &Value, type_name: &str) -> Option<bool> {
        let wants_signature = type_name.starts_with("function(");
        let wants_generic = type_name == "function";
        if !wants_signature && !wants_generic {
            return None;
        }

        let matches = match value {
            Value::Function(idx) => self.function_signature_matches(*idx, type_name),
            Value::Closure { function_idx, .. } => {
                self.function_signature_matches(*function_idx, type_name)
            }
            Value::NativeFunction(_) => wants_generic,
            _ => false,
        };
        Some(matches)
    }

    fn function_signature_matches(&self, func_idx: usize, type_name: &str) -> bool {
        if type_name == "function" {
            return true;
        }

        self.functions
            .get(func_idx)
            .and_then(|func| func.signature.as_ref())
            .map(|signature| signature.to_string() == type_name)
            .unwrap_or(false)
    }

    pub(super) fn get_register(&self, reg: Register) -> Result<&Value> {
        let frame = self
            .call_stack
            .last()
            .ok_or_else(|| LustError::RuntimeError {
                message: "Empty call stack".to_string(),
            })?;
        Ok(&frame.registers[reg as usize])
    }

    pub(super) fn set_register(&mut self, reg: Register, value: Value) -> Result<()> {
        self.observe_value(&value);
        let frame = self
            .call_stack
            .last_mut()
            .ok_or_else(|| LustError::RuntimeError {
                message: "Empty call stack".to_string(),
            })?;
        frame.registers[reg as usize] = value;
        self.maybe_collect_cycles();
        Ok(())
    }

    pub(super) fn handle_native_call_outcome(
        &mut self,
        dest: Register,
        outcome: NativeCallResult,
    ) -> Result<()> {
        match outcome {
            NativeCallResult::Return(value) => self.set_register(dest, value),
            NativeCallResult::Yield(value) => {
                if self.current_task.is_some() {
                    self.set_register(dest, Value::Nil)?;
                    self.pending_task_signal = Some(TaskSignal::Yield { dest, value });
                    Ok(())
                } else {
                    Err(LustError::RuntimeError {
                        message: "task.yield() can only be used inside a task".to_string(),
                    })
                }
            }

            NativeCallResult::Stop(value) => {
                if self.current_task.is_some() {
                    self.set_register(dest, Value::Nil)?;
                    self.pending_task_signal = Some(TaskSignal::Stop { value });
                    Ok(())
                } else {
                    Err(LustError::RuntimeError {
                        message: "task.stop() can only be used inside a task".to_string(),
                    })
                }
            }
        }
    }

    pub fn value_to_string_for_concat(&mut self, value: &Value) -> Result<Rc<String>> {
        match value {
            Value::String(s) => Ok(s.clone()),
            Value::Struct { name, .. } => self.invoke_tostring(value, name),
            Value::Enum { enum_name, .. } => self.invoke_tostring(value, enum_name),
            _ => Ok(Rc::new(value.to_string())),
        }
    }

    #[inline(never)]
    pub fn call_value(&mut self, func: &Value, args: Vec<Value>) -> Result<Value> {
        match func {
            Value::Function(func_idx) => {
                let mut frame = CallFrame {
                    function_idx: *func_idx,
                    ip: 0,
                    registers: array::from_fn(|_| Value::Nil),
                    base_register: 0,
                    return_dest: None,
                    upvalues: Vec::new(),
                };
                for (i, arg) in args.into_iter().enumerate() {
                    frame.registers[i] = arg;
                }

                let stack_depth_before = self.call_stack.len();
                self.call_stack.push(frame);
                let previous_target = self.call_until_depth;
                self.call_until_depth = Some(stack_depth_before);
                let run_result = self.run();
                self.call_until_depth = previous_target;
                run_result
            }

            Value::Closure {
                function_idx: func_idx,
                upvalues,
            } => {
                let upvalue_values: Vec<Value> = upvalues.iter().map(|uv| uv.get()).collect();
                let mut frame = CallFrame {
                    function_idx: *func_idx,
                    ip: 0,
                    registers: array::from_fn(|_| Value::Nil),
                    base_register: 0,
                    return_dest: None,
                    upvalues: upvalue_values,
                };
                for (i, arg) in args.into_iter().enumerate() {
                    frame.registers[i] = arg;
                }

                let stack_depth_before = self.call_stack.len();
                self.call_stack.push(frame);
                let previous_target = self.call_until_depth;
                self.call_until_depth = Some(stack_depth_before);
                let run_result = self.run();
                self.call_until_depth = previous_target;
                run_result
            }

            Value::NativeFunction(native_fn) => {
                self.push_current_vm();
                let outcome = native_fn(&args);
                self.pop_current_vm();
                let outcome = outcome.map_err(|e| LustError::RuntimeError { message: e })?;
                match outcome {
                    NativeCallResult::Return(value) => Ok(value),
                    NativeCallResult::Yield(_) | NativeCallResult::Stop(_) => {
                        Err(LustError::RuntimeError {
                            message: "Yielding or stopping is not allowed from this context"
                                .to_string(),
                        })
                    }
                }
            }

            _ => Err(LustError::RuntimeError {
                message: format!("Cannot call non-function value: {:?}", func),
            }),
        }
    }
}

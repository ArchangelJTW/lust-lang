use super::*;
use crate::bytecode::ValueKey;
use core::ptr;

/// How thoroughly a call's arguments are checked against the signature.
#[derive(Clone, Copy)]
enum ArgCheck {
    /// Full validation, including container contents (host and dynamic calls).
    Deep,
    /// Kinds only; the typechecker proved the rest (bytecode calls).
    Shallow,
}
impl VM {
    fn is_loop_in_hierarchy(
        &self,
        function_idx: usize,
        loop_start_ip: usize,
        backedge_ip: usize,
    ) -> bool {
        let instructions = &self.functions[function_idx].chunk.instructions;
        let jump_target =
            |ip: usize, offset: i16| (ip as isize + 1 + offset as isize).max(0) as usize;
        let contains_nested_loop = instructions
            .iter()
            .enumerate()
            .skip(loop_start_ip)
            .take(backedge_ip.saturating_sub(loop_start_ip))
            .any(|(ip, instruction)| {
                let Instruction::Jump(offset) = instruction else {
                    return false;
                };
                *offset < 0 && jump_target(ip, *offset) > loop_start_ip
            });
        let is_nested_loop = instructions
            .iter()
            .enumerate()
            .skip(backedge_ip.saturating_add(1))
            .any(|(ip, instruction)| {
                let Instruction::Jump(offset) = instruction else {
                    return false;
                };
                *offset < 0 && jump_target(ip, *offset) < loop_start_ip
            });
        contains_nested_loop || is_nested_loop
    }

    fn lua_table_map(value: &Value) -> Option<Value> {
        if let Value::Enum {
            enum_name,
            variant,
            values,
        } = value
            && enum_name == "LuaValue"
            && variant == "Table"
            && let Some(inner) = values.as_ref().and_then(|vals| vals.first())
            && let Some(map) = inner.struct_get_field("table")
        {
            return Some(map);
        }

        if let Value::Struct { name, .. } = value
            && name == "LuaTable"
            && let Some(map) = value.struct_get_field("table")
        {
            return Some(map);
        }

        None
    }

    fn lua_table_key_value(value: &Value) -> Value {
        if let Value::Enum {
            enum_name,
            variant,
            values,
        } = value
            && enum_name == "LuaValue"
        {
            return match variant.as_str() {
                "Nil" => Value::Nil,
                "Bool" | "Int" | "Float" | "String" | "Table" | "Function" | "LightUserdata"
                | "Userdata" | "Thread" => values
                    .as_ref()
                    .and_then(|v| v.first())
                    .cloned()
                    .unwrap_or(Value::Nil),
                _ => value.clone(),
            };
        }
        value.clone()
    }

    pub(super) fn push_current_vm(&mut self) {
        let ptr = self as *mut VM;
        crate::vm::push_vm_ptr(ptr);
    }

    pub(super) fn pop_current_vm(&mut self) {
        crate::vm::pop_vm_ptr();
    }

    pub(super) fn run(&mut self) -> Result<Value> {
        'dispatch: loop {
            if let Some(target_depth) = self.call_until_depth
                && self.call_stack.len() == target_depth
                && let Some(return_value) = self.pending_return_value.take()
            {
                self.call_until_depth = None;
                return Ok(return_value);
            }

            if self.pending_return_value.is_some()
                && let Some(return_value) = self.pending_return_value.take()
                && let Some(dest_reg) = self.pending_return_dest.take()
            {
                self.set_register(dest_reg, return_value)?;
            }

            if self.current_task.is_some()
                && let Some(signal) = self.pending_task_signal.take()
            {
                self.last_task_signal = Some(signal);
                return Ok(Value::Nil);
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
                    if let Some(frame) = self.call_stack.pop() {
                        self.recycle_frame(frame);
                    }
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
                let backedge_ip = ip_before_execution.saturating_sub(1);
                let loop_in_hierarchy =
                    self.is_loop_in_hierarchy(func_idx, loop_start_ip, backedge_ip);
                if self
                    .trace_recorder
                    .as_ref()
                    .is_some_and(|recorder| !recorder.is_recording())
                {
                    self.abandon_trace_recording();
                }
                let root_trace_id = if self.trace_recorder.is_none() {
                    self.jit
                        .root_traces
                        .get(&(func_idx, loop_start_ip))
                        .copied()
                } else {
                    None
                };
                if let Some(trace_id) = root_trace_id {
                    let frame = self.call_stack.last_mut().unwrap();
                    let registers_ptr = frame.registers.as_mut_ptr();
                    if let Some(trace) = self.jit.trace_handle(trace_id) {
                        self.jit.record_native_entry();
                        crate::jit::log(|| {
                            format!(
                                "▶️  JIT: Executing trace #{} at func {} ip {}",
                                trace_id.0, func_idx, loop_start_ip
                            )
                        });

                        let trace_gas_cost = {
                            let cost = trace.trace.ops.len()
                                + trace.trace.preamble.len()
                                + trace.trace.postamble.len();
                            core::cmp::max(1, cost) as u64
                        };
                        self.budgets.charge_gas(trace_gas_cost)?;
                        self.pending_jit_error = None;

                        // Capture RSP before and after to detect stack leaks (x86_64 only)
                        #[cfg(target_arch = "x86_64")]
                        let rsp_before: usize;
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            core::arch::asm!("mov {}, rsp", out(reg) rsp_before)
                        };

                        let vm_ptr = self as *mut VM;
                        let result = trace.execute(registers_ptr, vm_ptr, ptr::null());
                        drop(trace);

                        #[cfg(target_arch = "x86_64")]
                        let rsp_after: usize;
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            core::arch::asm!("mov {}, rsp", out(reg) rsp_after)
                        };

                        #[cfg(target_arch = "x86_64")]
                        let rsp_diff = rsp_after as isize - rsp_before as isize;
                        crate::jit::log(|| {
                            #[cfg(target_arch = "x86_64")]
                            return format!(
                                "🎯 JIT: Trace #{} execution result: {} (RSP before: {:x}, after: {:x}, diff: {})",
                                trace_id.0, result, rsp_before, rsp_after, rsp_diff
                            );
                            #[cfg(not(target_arch = "x86_64"))]
                            return format!(
                                "🎯 JIT: Trace #{} execution result: {}",
                                trace_id.0, result
                            );
                        });

                        if result == 0 {
                            if self.current_task.is_some() && self.pending_task_signal.is_some() {
                                if let Some(frame) = self.call_stack.last_mut() {
                                    frame.ip = ip_before_execution.saturating_sub(1);
                                }
                                continue;
                            }

                            if let Some(frame) = self.call_stack.last_mut() {
                                frame.ip = loop_start_ip;
                            }

                            continue;
                        } else if result > 0 {
                            self.jit.record_guard_exit();
                            let guard_index = (result - 1) as usize;
                            let bailout_ip = self
                                .jit
                                .get_trace(trace_id)
                                .and_then(|trace| trace.guards.get(guard_index))
                                .map(|guard| guard.bailout_ip);

                            crate::jit::log(|| {
                                let kind = self
                                    .jit
                                    .get_trace(trace_id)
                                    .and_then(|trace| trace.guards.get(guard_index))
                                    .map(|guard| format!("{:?}", guard.kind));
                                format!(
                                    "↩️  JIT: guard #{guard_index} exit {kind:?} → ip {bailout_ip:?}"
                                )
                            });
                            // A nested loop's trace may have bailed out
                            // somewhere other than the guard's own ip.
                            let bailout_ip = self.nested_loop_exit_ip.take().or(bailout_ip);
                            if let Some(bailout_ip) = bailout_ip
                                && let Some(frame) = self.call_stack.last_mut()
                            {
                                frame.ip = bailout_ip;
                            }
                            if let Some(error) = self.pending_jit_error.take() {
                                return Err(error);
                            }

                            self.handle_guard_failure(trace_id, guard_index, func_idx)?;
                            // A loop-condition exit is how a trace normally
                            // ends, and a nested-loop exit is where an outer
                            // trace hands the inner loop to its own trace;
                            // both resume at a known ip with the registers
                            // written back, so the trace stays valid for the
                            // next entry. Any other guard failure means the
                            // trace assumed something that no longer holds.
                            let reusable_exit = self
                                .jit
                                .get_trace(trace_id)
                                .and_then(|trace| trace.guards.get(guard_index))
                                .is_some_and(|guard| {
                                    matches!(
                                        guard.kind,
                                        crate::jit::GuardKind::Truthy { .. }
                                            | crate::jit::GuardKind::Falsy { .. }
                                            | crate::jit::GuardKind::NestedLoop { .. }
                                    )
                                });
                            if !reusable_exit {
                                self.jit.evict_root_trace(func_idx, loop_start_ip);
                            }
                            continue;
                        } else {
                            self.jit.record_execution_failure();
                            if let Some(error) = self.pending_jit_error.take() {
                                return Err(error);
                            }
                            // A result of -(k + 2) names the failing op's
                            // instruction: resume there, so the interpreter
                            // re-executes it (raising its error) with the
                            // trace's earlier side effects intact. A bare -1
                            // has no site and restarts the iteration.
                            let resume_ip = if result <= -2 {
                                self.jit
                                    .get_trace(trace_id)
                                    .and_then(|trace| trace.fail_sites.get((-result - 2) as usize))
                                    .copied()
                            } else {
                                None
                            };
                            crate::jit::log(|| {
                                format!(
                                    "⚠️  JIT: Trace execution failed (result {result}), resuming at ip {:?}",
                                    resume_ip
                                )
                            });
                            if let Some(frame) = self.call_stack.last_mut() {
                                frame.ip = resume_ip.unwrap_or(loop_start_ip);
                            }

                            self.jit.evict_root_trace(func_idx, loop_start_ip);
                            // Re-dispatch from the loop header we just installed.
                            //
                            // Falling through instead would let the interpreter
                            // go on to execute the backward `Jump` that got us
                            // here, applying its offset on top of the rewritten
                            // ip.  That sends ip out of range, pops the frame and
                            // empties the call stack, so the program simply stops
                            // — exit status 0, nothing printed, no error.
                            continue;
                        }
                    }
                } else {
                    if let Some(recorder) = &mut self.trace_recorder {
                        // Only finalise the recording when *this* loop is the
                        // one being recorded.  With a nested loop the inner
                        // back-edge reaches this handler while the recorder is
                        // still part-way through the outer loop's body; taking
                        // that partial body and installing it under the inner
                        // loop's key produced a trace that re-initialised the
                        // inner induction variable, never advanced the outer
                        // one, and therefore never terminated.
                        let recording_this_loop = recorder.trace.function_idx == func_idx
                            && recorder.trace.start_ip == loop_start_ip;
                        if recorder.is_recording() && recording_this_loop {
                            crate::jit::log(|| {
                                format!(
                                    "📝 JIT: Trace recording complete - {} ops recorded",
                                    recorder.trace.ops.len()
                                )
                            });
                            let mut recorder = self.trace_recorder.take().unwrap();
                            recorder.complete_nested_skip_at(backedge_ip);
                            if !recorder.is_recording() {
                                self.jit.recording_aborted(func_idx, loop_start_ip);
                                continue;
                            }
                            let mut trace = recorder.finish();
                            let mut optimizer = TraceOptimizer::new();
                            let hoisted_constants = optimizer.optimize(&mut trace);
                            crate::jit::log(|| "⚙️  JIT: Compiling root trace...".to_string());
                            let trace_id = self.jit.alloc_trace_id();
                            match JitCompiler::new().compile_trace(
                                &trace,
                                trace_id,
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
                                    self.jit.recording_aborted(func_idx, loop_start_ip);
                                }
                            }
                        }
                    }

                    if self.trace_recorder.is_none()
                        && !self
                            .jit
                            .root_traces
                            .contains_key(&(func_idx, loop_start_ip))
                        && self.jit.should_record_root(
                            func_idx,
                            loop_start_ip,
                            count,
                            crate::jit::HOT_THRESHOLD + u32::from(loop_in_hierarchy),
                        )
                    {
                        crate::jit::log(|| {
                            format!(
                                "🔥 JIT: Hot loop detected at func {} ip {} - starting trace recording!",
                                func_idx, loop_start_ip
                            )
                        });
                        let mut recorder =
                            TraceRecorder::new(func_idx, loop_start_ip, MAX_TRACE_LENGTH);
                        recorder.set_root_frame_index(self.call_stack.len().saturating_sub(1));
                        recorder.set_intrinsics(&self.jit.intrinsics);
                        // Specialize loop-invariant values at trace entry
                        if !self
                            .jit
                            .no_specialize_sites
                            .contains(&(func_idx, loop_start_ip))
                        {
                            let frame = self.call_stack.last().unwrap();
                            let func = &self.functions[func_idx];
                            recorder.specialize_trace_inputs(&frame.registers, func);
                        }
                        self.trace_recorder = Some(recorder);
                        self.jit.recording_started();
                        self.skip_next_trace_record = true;
                    }
                }
            }

            self.budgets.charge_gas(1)?;
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
                        func.chunk.constants[const_idx as usize].fast_clone()
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
                    self.globals_version = self.globals_version.wrapping_add(1);
                }

                Instruction::Move(dest, src) => {
                    let value = self.get_register(src)?.clone();
                    self.set_register(dest, value)?;
                }

                Instruction::AddInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Int(a + b)))?;
                }
                Instruction::SubInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Int(a - b)))?;
                }
                Instruction::MulInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Int(a * b)))?;
                }
                Instruction::DivInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| {
                        if b == 0 {
                            Err(LustError::RuntimeError {
                                message: "Division by zero".to_string(),
                            })
                        } else {
                            Ok(Value::Int(a.wrapping_div(b)))
                        }
                    })?;
                }
                Instruction::ModInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| {
                        if b == 0 {
                            Err(LustError::RuntimeError {
                                message: "Modulo by zero".to_string(),
                            })
                        } else {
                            Ok(Value::Int(a.wrapping_rem(b)))
                        }
                    })?;
                }
                Instruction::NegInt(dest, src) => {
                    self.int_binary_op(dest, src, src, |a, _| Ok(Value::Int(-a)))?;
                }
                Instruction::EqInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a == b)))?;
                }
                Instruction::NeInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a != b)))?;
                }
                Instruction::LtInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a < b)))?;
                }
                Instruction::LeInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a <= b)))?;
                }
                Instruction::GtInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a > b)))?;
                }
                Instruction::GeInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a >= b)))?;
                }
                Instruction::AddFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Float(a + b)))?;
                }
                Instruction::SubFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Float(a - b)))?;
                }
                Instruction::MulFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Float(a * b)))?;
                }
                Instruction::DivFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Float(a / b)))?;
                }
                Instruction::ModFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| {
                        if b == 0.0 {
                            Err(LustError::RuntimeError {
                                message: "Modulo by zero".to_string(),
                            })
                        } else {
                            Ok(Value::Float(a % b))
                        }
                    })?;
                }
                Instruction::NegFloat(dest, src) => {
                    self.float_binary_op(dest, src, src, |a, _| Ok(Value::Float(-a)))?;
                }
                Instruction::EqFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a == b)))?;
                }
                Instruction::NeFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a != b)))?;
                }
                Instruction::LtFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a < b)))?;
                }
                Instruction::LeFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a <= b)))?;
                }
                Instruction::GtFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a > b)))?;
                }
                Instruction::GeFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a >= b)))?;
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
                                Ok(Value::Int(a.wrapping_div(*b)))
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
                                Ok(Value::Int(a.wrapping_rem(*b)))
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
                            });
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
                    self.comparison_op(dest, lhs, rhs, |l, r| l < r, |l, r| l < r)?;
                }

                Instruction::Le(dest, lhs, rhs) => {
                    self.comparison_op(dest, lhs, rhs, |l, r| l <= r, |l, r| l <= r)?;
                }

                Instruction::Gt(dest, lhs, rhs) => {
                    self.comparison_op(dest, lhs, rhs, |l, r| l > r, |l, r| l > r)?;
                }

                Instruction::Ge(dest, lhs, rhs) => {
                    self.comparison_op(dest, lhs, rhs, |l, r| l >= r, |l, r| l >= r)?;
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
                    let truthy = match condition {
                        Value::Bool(b) => *b,
                        other => other.is_truthy(),
                    };
                    if truthy {
                        let frame = self.call_stack.last_mut().unwrap();
                        frame.ip = (frame.ip as isize + offset as isize) as usize;
                    }
                }

                Instruction::JumpIfNot(cond, offset) => {
                    let condition = self.get_register(cond)?;
                    let truthy = match condition {
                        Value::Bool(b) => *b,
                        other => other.is_truthy(),
                    };
                    if !truthy {
                        let frame = self.call_stack.last_mut().unwrap();
                        frame.ip = (frame.ip as isize + offset as isize) as usize;
                    }
                }

                Instruction::Call(func_reg, first_arg, arg_count, dest_reg)
                    if self.plain_function_callee(func_reg) =>
                {
                    let frame = self.bytecode_call_frame(func_reg, first_arg, arg_count, dest_reg)?;
                    let callee_idx = frame.function_idx;
                    self.call_stack.push(frame);
                    if self.jit.enabled
                        && self.trace_recorder.is_none()
                        && let Some(finished) = self.run_compiled_function(callee_idx)?
                    {
                        return Ok(finished);
                    }
                }

                Instruction::Call(func_reg, first_arg, arg_count, dest_reg) => {
                    let func_value = self.get_register(func_reg)?.clone();
                    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
                    let mut func_value = func_value;
                    // if let Value::Enum { enum_name, variant, .. } = &func_value {
                    //     if enum_name == "LuaValue" && variant == "Table" {
                    //         eprintln!("DEBUG: Instruction::Call with LuaValue.Table");
                    //     }
                    // }

                    // Check if this is a table/userdata with __call metamethod (Lua compat)
                    let needs_call_value = {
                        let mut check_value = &func_value;
                        // Unwrap LuaValue enum if needed
                        if let Value::Enum {
                            enum_name,
                            variant,
                            values,
                        } = &func_value
                            && enum_name == "LuaValue"
                            && (variant == "Table" || variant == "Userdata")
                            && let Some(inner) = values.as_ref().and_then(|v| v.first())
                        {
                            check_value = inner;
                        }
                        // Check if it's a LuaTable/LuaUserdata struct with metamethods
                        if let Value::Struct { name, .. } = check_value {
                            (name == "LuaTable" || name == "LuaUserdata")
                                && check_value.struct_get_field("metamethods").is_some()
                        } else {
                            false
                        }
                    };

                    if needs_call_value {
                        // eprintln!("DEBUG Instruction::Call: Delegating to call_value for table/userdata");
                        // Delegate to call_value which handles __call metamethods
                        let mut args = Vec::new();
                        for i in 0..arg_count {
                            args.push(self.get_register(first_arg + i)?.clone());
                        }
                        let result = self.call_value(&func_value, args)?;
                        self.set_register(dest_reg, result)?;
                        continue;
                    } else {
                        // Check what type we're actually dealing with
                        if let Value::Enum {
                            enum_name, variant, ..
                        } = &func_value
                            && enum_name == "LuaValue"
                            && (variant == "Table" || variant == "Userdata")
                        {
                            #[cfg(feature = "std")]
                            eprintln!(
                                "DEBUG Instruction::Call: Have LuaValue.{} but needs_call_value=false",
                                variant
                            );
                        }
                    }

                    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
                    loop {
                        let handle = match &func_value {
                            Value::Enum {
                                enum_name,
                                variant,
                                values,
                            } if enum_name == "LuaValue" && variant == "Function" => values
                                .as_ref()
                                .and_then(|vals| vals.first())
                                .and_then(|v| v.struct_get_field("handle"))
                                .and_then(|v| v.as_int())
                                .map(|i| i as usize),
                            _ => None,
                        };
                        let Some(handle) = handle else { break };
                        let inner =
                            crate::lua_compat::lookup_lust_function(handle).ok_or_else(|| {
                                LustError::RuntimeError {
                                    message: format!(
                                        "LuaValue function handle {} was not registered with VM",
                                        handle
                                    ),
                                }
                            })?;
                        func_value = inner;
                    }
                    match func_value {
                        Value::Function(func_idx) => {
                            let mut args = core::mem::take(&mut self.arg_scratch);
                            args.clear();
                            for i in 0..arg_count {
                                args.push(self.get_register(first_arg + i)?.clone());
                            }

                            let frame = self.make_checked_call_frame(
                                func_idx,
                                Some(dest_reg),
                                args,
                                Vec::new(),
                            )?;
                            self.call_stack.push(frame);
                        }

                        Value::Closure {
                            function_idx: func_idx,
                            upvalues,
                        } => {
                            let mut args = core::mem::take(&mut self.arg_scratch);
                            args.clear();
                            for i in 0..arg_count {
                                args.push(self.get_register(first_arg + i)?.clone());
                            }

                            let upvalue_values: Vec<Value> =
                                upvalues.iter().map(|uv| uv.get()).collect();
                            let frame = self.make_checked_call_frame(
                                func_idx,
                                Some(dest_reg),
                                args,
                                upvalue_values,
                            )?;
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
                            });
                        }
                    }
                }

                Instruction::Return(value_reg) => {
                    let return_value = if value_reg == 255 {
                        Value::Nil
                    } else {
                        self.get_register(value_reg)?.fast_clone()
                    };
                    if let Some(finished) = self.finish_return(return_value)? {
                        return Ok(finished);
                    }
                }

                Instruction::NewArray(dest, first_elem, count) => {
                    let element_count = count as usize;
                    self.budgets.charge_value_vec(element_count)?;
                    let mut elements = Vec::with_capacity(element_count);
                    for i in 0..count {
                        elements.push(self.get_register(first_elem + i)?.clone());
                    }

                    self.set_register(dest, Value::array(elements))?;
                }

                Instruction::TupleNew(dest, first_elem, count) => {
                    let mut parts = Vec::with_capacity(count as usize);
                    let mut total_elements: usize = 0;
                    for offset in 0..(count as usize) {
                        let value = self.get_register(first_elem + offset as u8)?.clone();
                        total_elements = total_elements.saturating_add(match &value {
                            Value::Tuple(existing) => existing.len(),
                            _ => 1,
                        });
                        parts.push(value);
                    }

                    self.budgets.charge_value_vec(total_elements)?;
                    let mut elements = Vec::with_capacity(total_elements);
                    for value in parts {
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
                    self.budgets.charge_value_vec(field_count as usize)?;
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
                    self.budgets.charge_value_vec(value_count as usize)?;
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
                    let mut values = Vec::with_capacity(value_count as usize);
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
                    let value = if let Some(map_val) = Self::lua_table_map(object) {
                        if let Value::Map(map) = map_val {
                            let raw_key_value = Value::String(field_name.clone());
                            let key =
                                self.make_hash_key(&Self::lua_table_key_value(&raw_key_value))?;
                            let borrowed = map.borrow();
                            borrowed.get(&key).cloned().unwrap_or(Value::Nil)
                        } else {
                            Value::Nil
                        }
                    } else {
                        match object {
                            Value::Struct { .. } => object
                                .struct_get_field_rc(&field_name)
                                .unwrap_or(Value::Nil),
                            Value::Map(map) => {
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
                                });
                            }
                        }
                    };
                    self.set_register(dest, value)?;
                }

                Instruction::SetField(obj_reg, field_idx, value_reg) => {
                    let object = self.get_register(obj_reg)?.clone();
                    let value = self.get_register(value_reg)?.clone();
                    let func = &self.functions[self.call_stack.last().unwrap().function_idx];
                    let field_name = func.chunk.constants[field_idx as usize]
                        .as_string_rc()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Field name must be a string".to_string(),
                        })?;
                    let mut invalidate_key: Option<usize> = None;
                    if let Some(map_val) = Self::lua_table_map(&object) {
                        if let Value::Map(map) = map_val {
                            let raw_key_value = Value::String(field_name.clone());
                            let key =
                                self.make_hash_key(&Self::lua_table_key_value(&raw_key_value))?;
                            if self.budgets.mem_budget_enabled() && !map.borrow().contains_key(&key)
                            {
                                self.budgets.charge_map_entry_estimate()?;
                            }
                            map.borrow_mut().insert(key, value);
                        } else {
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "Cannot set field '{}' on {:?}",
                                    field_name.as_str(),
                                    object
                                ),
                            });
                        }
                    } else {
                        match &object {
                            Value::Struct { .. } => {
                                invalidate_key = Self::struct_cache_key(&object);
                                object
                                    .struct_set_field_rc(&field_name, value)
                                    .map_err(|message| LustError::RuntimeError { message })?;
                            }

                            Value::Map(map) => {
                                let key = ValueKey::from(field_name.clone());
                                if self.budgets.mem_budget_enabled()
                                    && !map.borrow().contains_key(&key)
                                {
                                    self.budgets.charge_map_entry_estimate()?;
                                }
                                map.borrow_mut().insert(key, value);
                            }

                            _ => {
                                return Err(LustError::RuntimeError {
                                    message: format!(
                                        "Cannot set field '{}' on {:?}",
                                        field_name.as_str(),
                                        object
                                    ),
                                });
                            }
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
                    let cap = left_str.len().saturating_add(right_str.len());
                    self.budgets.charge_mem_bytes(cap)?;
                    let mut combined = String::with_capacity(cap);
                    combined.push_str(left_str.as_ref());
                    combined.push_str(right_str.as_ref());
                    let result = Value::string(combined);
                    self.set_register(dest, result)?;
                }

                Instruction::GetIndex(dest, array_reg, index_reg) => {
                    let collection = self.get_register(array_reg)?.clone();
                    let index = self.get_register(index_reg)?.clone();
                    let result = if let Some(map_val) = Self::lua_table_map(&collection) {
                        if let Value::Map(map) = map_val {
                            let raw_key = self.make_hash_key(&Self::lua_table_key_value(&index))?;
                            let borrowed = map.borrow();
                            borrowed.get(&raw_key).cloned().unwrap_or(Value::Nil)
                        } else {
                            return Err(LustError::RuntimeError {
                                message: format!("Cannot index {:?}", collection.type_of()),
                            });
                        }
                    } else {
                        match collection {
                            Value::Array(arr) => {
                                let idx =
                                    index.as_int().ok_or_else(|| LustError::RuntimeError {
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
                                });
                            }
                        }
                    };
                    self.set_register(dest, result)?;
                }

                Instruction::TryGetIndex(dest, collection_reg, index_reg) => {
                    let collection = self.get_register(collection_reg)?.clone();
                    let index = self.get_register(index_reg)?.clone();
                    let result = if let Some(map_val) = Self::lua_table_map(&collection) {
                        if let Value::Map(map) = map_val {
                            let raw_key = self.make_hash_key(&Self::lua_table_key_value(&index))?;
                            let borrowed = map.borrow();
                            borrowed.get(&raw_key).cloned().unwrap_or(Value::Nil)
                        } else {
                            return Err(LustError::RuntimeError {
                                message: format!("Cannot index {:?}", collection.type_of()),
                            });
                        }
                    } else {
                        match &collection {
                            Value::Array(_) => {
                                let index =
                                    index.as_int().ok_or_else(|| LustError::RuntimeError {
                                        message: "Array index must be an integer".to_string(),
                                    })?;
                                self.array_index_result(&collection, index)?
                            }
                            Value::Map(map) => {
                                let key = self.make_hash_key(&index)?;
                                map.borrow().get(&key).cloned().unwrap_or(Value::Nil)
                            }
                            _ => {
                                return Err(LustError::RuntimeError {
                                    message: format!("Cannot index {:?}", collection.type_of()),
                                });
                            }
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
                ) => 'method: {
                    let object = self.get_register(obj_reg)?.clone();
                    // Fast path: a user-defined struct method already resolved
                    // at this call site.
                    if let Value::Struct { layout, .. } = &object {
                        let key = (
                            Rc::as_ptr(layout) as usize,
                            func_idx,
                            method_name_idx as u16,
                        );
                        if let Some(&target) = self.method_cache.get(&key) {
                            let mut args = core::mem::take(&mut self.arg_scratch);
                            args.clear();
                            args.push(object.clone());
                            for i in 0..arg_count {
                                args.push(self.get_register(first_arg + i)?.clone());
                            }
                            let frame = self.make_checked_call_frame(
                                target,
                                Some(dest_reg),
                                args,
                                Vec::new(),
                            )?;
                            // Fall through to the trace recorder below: it
                            // inlines user-defined struct methods like calls.
                            self.call_stack.push(frame);
                            break 'method;
                        }
                    }
                    let method_name = {
                        let func = &self.functions[self.call_stack.last().unwrap().function_idx];
                        func.chunk.constants[method_name_idx as usize]
                            .as_string()
                            .ok_or_else(|| LustError::RuntimeError {
                                message: "Method name must be a string".to_string(),
                            })?
                            .to_string()
                    };
                    let object_type_name = match &object {
                        Value::Struct { name, .. } => Some(name.as_str()),
                        Value::Enum { enum_name, .. } => Some(enum_name.as_str()),
                        _ => None,
                    };
                    if let Some(struct_name) = object_type_name {
                        let mangled_name = format!("{}:{}", struct_name, method_name);
                        if let Some(func_idx) =
                            self.functions.iter().position(|f| f.name == mangled_name)
                        {
                            if let Value::Struct { layout, .. } = &object {
                                let caller = self.call_stack.last().unwrap().function_idx;
                                self.method_cache.insert(
                                    (Rc::as_ptr(layout) as usize, caller, method_name_idx as u16),
                                    func_idx,
                                );
                            }
                            let mut args = Vec::with_capacity(1 + arg_count as usize);
                            args.push(object.clone());
                            for i in 0..arg_count {
                                args.push(self.get_register(first_arg + i)?.clone());
                            }
                            let frame =
                                self.make_call_frame(func_idx, Some(dest_reg), args, Vec::new())?;
                            self.call_stack.push(frame);
                            break 'method;
                        }

                        let mut candidate_names = vec![mangled_name.clone()];
                        if let Some(simple) = struct_name.rsplit(['.', ':']).next() {
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
                                if self.trace_recorder.is_some() {
                                    self.abandon_trace_recording();
                                }
                                let result = self.call_value(&global_func, call_args)?;
                                self.set_register(dest_reg, result)?;
                                handled = true;
                                break;
                            }
                        }
                        if handled {
                            continue 'dispatch;
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
                    self.budgets
                        .charge_upvalues_estimate(upvalue_count as usize)?;
                    let mut upvalues = Vec::with_capacity(upvalue_count as usize);
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
                    if let Some(map_val) = Self::lua_table_map(&collection) {
                        if let Value::Map(map) = map_val {
                            let key = self.make_hash_key(&Self::lua_table_key_value(&index))?;
                            if self.budgets.mem_budget_enabled() && !map.borrow().contains_key(&key)
                            {
                                self.budgets.charge_map_entry_estimate()?;
                            }
                            map.borrow_mut().insert(key, value);
                        } else {
                            return Err(LustError::RuntimeError {
                                message: format!("Cannot index {:?}", collection.type_of()),
                            });
                        }
                    } else {
                        match collection {
                            Value::Array(arr) => {
                                let idx =
                                    index.as_int().ok_or_else(|| LustError::RuntimeError {
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
                                if self.budgets.mem_budget_enabled()
                                    && !map.borrow().contains_key(&key)
                                {
                                    self.budgets.charge_map_entry_estimate()?;
                                }
                                map.borrow_mut().insert(key, value);
                            }

                            _ => {
                                return Err(LustError::RuntimeError {
                                    message: format!("Cannot index {:?}", collection.type_of()),
                                });
                            }
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

                Instruction::TryCast(dest, value_reg, type_name_idx) => {
                    let value = self.get_register(value_reg)?.clone();
                    let func = &self.functions[self.call_stack.last().unwrap().function_idx];
                    let type_name = func.chunk.constants[type_name_idx as usize]
                        .as_string()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Cast type name must be a string".to_string(),
                        })?
                        .to_string();
                    let result = if self.value_is_type(&value, &type_name) {
                        self.budgets.charge_value_vec(1)?;
                        Value::some(value)
                    } else {
                        Value::none()
                    };
                    self.set_register(dest, result)?;
                }
            }

            if self.jit.enabled
                && let Some(recorder) = &mut self.trace_recorder
                && recorder.is_recording()
            {
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
                    let frame_pushed = self.call_stack.len() > executing_frame_index + 1;
                    recorder.globals_version = self.globals_version;
                    if let Some(registers) = registers_opt
                        && let Err(e) = recorder.record_instruction_at_frame(
                            executing_frame_index,
                            instruction,
                            ip_before_execution,
                            registers,
                            function,
                            func_idx,
                            &self.functions,
                            frame_pushed,
                        )
                    {
                        crate::jit::log(|| format!("⚠️  JIT: {}", e));
                        self.abandon_trace_recording();
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

    fn int_binary_op<F>(
        &mut self,
        dest: Register,
        lhs: Register,
        rhs: Register,
        op: F,
    ) -> Result<()>
    where
        F: FnOnce(LustInt, LustInt) -> Result<Value>,
    {
        let left = self.get_register(lhs)?;
        let right = self.get_register(rhs)?;
        let (Value::Int(a), Value::Int(b)) = (left, right) else {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Typed numeric instruction expected int operands, got {:?} and {:?}",
                    left, right
                ),
            });
        };
        let result = op(*a, *b)?;
        self.set_register(dest, result)
    }

    fn float_binary_op<F>(
        &mut self,
        dest: Register,
        lhs: Register,
        rhs: Register,
        op: F,
    ) -> Result<()>
    where
        F: FnOnce(LustFloat, LustFloat) -> Result<Value>,
    {
        let left = self.get_register(lhs)?;
        let right = self.get_register(rhs)?;
        let (Value::Float(a), Value::Float(b)) = (left, right) else {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Typed numeric instruction expected float operands, got {:?} and {:?}",
                    left, right
                ),
            });
        };
        let result = op(*a, *b)?;
        self.set_register(dest, result)
    }

    pub(super) fn comparison_op<I, F>(
        &mut self,
        dest: Register,
        lhs: Register,
        rhs: Register,
        int_op: I,
        op: F,
    ) -> Result<()>
    where
        I: FnOnce(LustInt, LustInt) -> bool,
        F: FnOnce(LustFloat, LustFloat) -> bool,
    {
        let left = self.get_register(lhs)?;
        let right = self.get_register(rhs)?;
        let result = match (left, right) {
            (Value::Int(a), Value::Int(b)) => int_op(*a, *b),
            (Value::Float(a), Value::Float(b)) => op(*a, *b),
            (Value::Int(a), Value::Float(b)) => op(float_from_int(*a), *b),
            (Value::Float(a), Value::Int(b)) => op(*a, float_from_int(*b)),
            _ => {
                return Err(LustError::RuntimeError {
                    message: format!("Cannot compare {:?} and {:?}", left, right),
                });
            }
        };
        self.set_register(dest, Value::Bool(result))
    }

    pub(crate) fn value_is_type(&self, value: &Value, type_name: &str) -> bool {
        // Lua compatibility intentionally treats LuaValue as a dynamic carrier. Transpiled
        // values may still be represented by their underlying VM value at call boundaries.
        if type_name == "LuaValue" {
            return true;
        }

        if type_name
            .split('|')
            .map(str::trim)
            .any(|member| member != type_name && self.value_is_type(value, member))
        {
            return true;
        }

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
            Value::Task(_) => "Task",
        };
        if value_type_name == type_name || (matches!(value, Value::Nil) && type_name == "()") {
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

        if (type_name == "Option" || type_name.starts_with("Option<"))
            && matches!(value, Value::Enum { enum_name, .. } if enum_name == "Option")
        {
            return true;
        }

        if (type_name == "Result" || type_name.starts_with("Result<"))
            && matches!(value, Value::Enum { enum_name, .. } if enum_name == "Result")
        {
            return true;
        }

        if type_name == "unknown" {
            return true;
        }

        if self
            .trait_impls
            .get(&(value_type_name.to_string(), type_name.to_string()))
            .is_some()
        {
            return true;
        }

        false
    }

    pub(crate) fn value_matches_type(&self, value: &Value, ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Int => matches!(value, Value::Int(_)),
            TypeKind::Float => matches!(value, Value::Float(_)),
            TypeKind::String => matches!(value, Value::String(_)),
            TypeKind::Bool => matches!(value, Value::Bool(_)),
            TypeKind::Unit => matches!(value, Value::Nil),
            TypeKind::Unknown | TypeKind::Generic(_) | TypeKind::Infer => true,
            TypeKind::Named(expected) => self.value_is_type(value, expected),
            TypeKind::GenericInstance { name, .. } => self.value_is_type(value, name),
            TypeKind::Array(element_type) => match value {
                Value::Array(values) => values
                    .borrow()
                    .iter()
                    .all(|value| self.value_matches_type(value, element_type)),
                _ => false,
            },
            TypeKind::Map(key_type, value_type) => match value {
                Value::Map(entries) => entries.borrow().iter().all(|(key, value)| {
                    self.value_matches_type(&key.to_value(), key_type)
                        && self.value_matches_type(value, value_type)
                }),
                _ => false,
            },
            TypeKind::Tuple(element_types) => match value {
                Value::Tuple(values) => {
                    values.len() == element_types.len()
                        && values
                            .iter()
                            .zip(element_types)
                            .all(|(value, ty)| self.value_matches_type(value, ty))
                }
                _ => false,
            },
            TypeKind::Option(inner) => match value {
                Value::Enum {
                    enum_name,
                    variant,
                    values,
                } if enum_name == "Option" => match variant.as_str() {
                    "None" => values.as_ref().is_none_or(|values| values.is_empty()),
                    "Some" => values.as_ref().is_some_and(|values| {
                        values.len() == 1 && self.value_matches_type(&values[0], inner)
                    }),
                    _ => false,
                },
                _ => false,
            },
            TypeKind::Result(ok_type, err_type) => match value {
                Value::Enum {
                    enum_name,
                    variant,
                    values,
                } if enum_name == "Result" => {
                    let payload_type = match variant.as_str() {
                        "Ok" => ok_type,
                        "Err" => err_type,
                        _ => return false,
                    };
                    values.as_ref().is_some_and(|values| {
                        values.len() == 1 && self.value_matches_type(&values[0], payload_type)
                    })
                }
                _ => false,
            },
            TypeKind::Function {
                params,
                return_type,
            } => match value {
                Value::Function(index)
                | Value::Closure {
                    function_idx: index,
                    ..
                } => self
                    .functions
                    .get(*index)
                    .and_then(|function| function.signature.as_ref())
                    .is_some_and(|signature| {
                        signature.params == *params && signature.return_type == **return_type
                    }),
                Value::NativeFunction(_) => true,
                _ => false,
            },
            TypeKind::Union(types) => types.iter().any(|ty| self.value_matches_type(value, ty)),
            TypeKind::Trait(name) => {
                let value_name = self.value_trait_name(value);
                self.trait_impls
                    .contains_key(&(value_name, name.to_string()))
            }
            TypeKind::TraitBound(names) => {
                let value_name = self.value_trait_name(value);
                names.iter().all(|name| {
                    self.trait_impls
                        .contains_key(&(value_name.clone(), name.clone()))
                })
            }
            TypeKind::Ref(inner) | TypeKind::MutRef(inner) => self.value_matches_type(value, inner),
            TypeKind::Pointer { .. } => true,
        }
    }

    /// The tail of `Instruction::Return` once the value is in hand: check
    /// it against the signature, pop the frame and deliver the value to
    /// the caller. Returns the value itself when the call stack is now
    /// empty and `run` should hand it back.
    fn finish_return(&mut self, return_value: Value) -> Result<Option<Value>> {
        let frame = self.call_stack.last().unwrap();
        let return_dest = frame.return_dest;
        self.validate_function_return(frame.function_idx, &return_value)?;
        if let Some(frame) = self.call_stack.pop() {
            self.recycle_frame(frame);
        }
        if self.call_stack.is_empty() {
            return Ok(Some(return_value));
        }

        // Deliver straight into the caller unless this return ends a host
        // call into the VM, which `run` hands back from the top of the
        // loop.
        if self.call_until_depth == Some(self.call_stack.len()) {
            self.pending_return_value = Some(return_value);
            self.pending_return_dest = return_dest;
        } else if let Some(dest) = return_dest {
            self.set_register(dest, return_value)?;
        }
        Ok(None)
    }

    /// Run the frame just pushed for `func_idx` through the function's
    /// compiled code, compiling it first when it has just become hot (see
    /// `jit::function`). Does nothing when there is no code. Afterwards
    /// the call stack is wherever the native code left it: the frame
    /// popped (it returned; `Some` when that emptied the stack), or one or
    /// more frames — native callees materialized on exit included —
    /// positioned at the instruction the interpreter continues from.
    fn run_compiled_function(&mut self, func_idx: usize) -> Result<Option<Value>> {
        let code = match self.jit.function_code(func_idx) {
            Some(code) => code,
            None => {
                if !self.jit.record_function_entry(func_idx) {
                    return Ok(None);
                }
                self.compile_function(func_idx);
                match self.jit.function_code(func_idx) {
                    Some(code) => code,
                    None => return Ok(None),
                }
            }
        };

        // Native-to-native calls grow the machine stack; give them at most
        // `NATIVE_STACK_RESERVE` below here before they hand calls to the
        // interpreter, whose frames live on the heap.
        let marker = 0u8;
        let sp = &marker as *const u8 as usize;
        let limit = sp.saturating_sub(crate::jit::NATIVE_STACK_RESERVE);
        crate::jit::JIT_STACK_LIMIT.with(|cell| {
            let current = cell.get();
            if current == 0 || limit < current {
                cell.set(limit);
            }
        });

        let cost = code.trace.ops.len();
        self.budgets.charge_gas(core::cmp::max(1, cost) as u64)?;
        self.jit.record_native_entry();
        self.pending_jit_error = None;
        crate::jit::JIT_EXIT_INFO.with(|cell| cell.set(usize::MAX));
        let budget = self.max_stack_depth.saturating_sub(self.call_stack.len());
        crate::jit::JIT_DEPTH_BUDGET.with(|cell| cell.set(budget));
        let registers_ptr = self.call_stack.last_mut().unwrap().registers.as_mut_ptr();
        let vm_ptr = self as *mut VM;
        let result = code.execute(registers_ptr, vm_ptr, ptr::null());
        crate::jit::log(|| format!("🎯 JIT: function {} native result {}", func_idx, result));
        drop(code);

        if (crate::jit::FUNCTION_RETURN_BASE..crate::jit::NATIVE_RETURNED).contains(&result) {
            let reg = (result - crate::jit::FUNCTION_RETURN_BASE) as usize;
            let return_value = if reg == 255 {
                Value::Nil
            } else {
                self.call_stack.last().unwrap().registers[reg].fast_clone()
            };
            return self.finish_return(return_value);
        }

        // An exit, from this function or a native callee whose frames are
        // now on the call stack: resume the innermost frame where the
        // exiting site said.
        let info = crate::jit::JIT_EXIT_INFO.with(|cell| cell.get());
        if info == usize::MAX {
            // Every exit stub of function code records its site; an exit
            // without one is a compiler bug. Evict the code and resume at
            // the function's entry, which at least keeps the interpreter
            // consistent.
            debug_assert!(false, "function {func_idx} exited with {result} and no exit info");
            crate::jit::log(|| {
                format!("❌ JIT: function {func_idx} exited with {result} and no exit info")
            });
            self.jit.evict_function_code(func_idx);
            if let Some(frame) = self.call_stack.last_mut() {
                frame.ip = 0;
            }
            return Ok(None);
        }
        let ip = info & ((1usize << crate::jit::EXIT_KIND_SHIFT) - 1);
        let kind = info >> crate::jit::EXIT_KIND_SHIFT;
        if let Some(error) = self.pending_jit_error.take() {
            return Err(error);
        }
        let frame = self.call_stack.last_mut().unwrap();
        let exited_idx = frame.function_idx;
        frame.ip = ip;
        crate::jit::log(|| {
            format!(
                "↩️  JIT: function {} exit kind {} → func {} ip {}",
                func_idx, kind, exited_idx, ip
            )
        });
        match kind {
            crate::jit::EXIT_KIND_HANDOFF | crate::jit::EXIT_KIND_FAIL => {}
            _ => {
                self.jit.record_guard_exit();
                self.jit.evict_function_code(exited_idx);
            }
        }
        Ok(None)
    }

    /// Translate and compile `func_idx` (see `jit::function`), or mark it
    /// as not compilable.
    fn compile_function(&mut self, func_idx: usize) {
        use crate::jit::function::{FunctionSig, translate};
        let sig_of = |meta: &CallMeta, function: &Function| FunctionSig {
            params: meta.params.iter().map(|kind| kind.value_type()).collect(),
            ret: meta.return_kind.value_type(),
            lua_function: meta.lua_function,
            register_count: function.register_count,
        };
        let Some((function, meta)) = self.functions.get(func_idx).zip(self.call_meta.get(func_idx))
        else {
            return;
        };
        let sig = sig_of(meta, function);
        let functions = &self.functions;
        let call_meta = &self.call_meta;
        let callee_sig = |idx: usize| {
            functions
                .get(idx)
                .zip(call_meta.get(idx))
                .map(|(function, meta)| sig_of(meta, function))
        };
        let Some(trace) = translate(function, func_idx, &sig, &callee_sig) else {
            crate::jit::log(|| format!("🚫 JIT: function {} is not compilable", func_idx));
            self.jit.function_not_compilable(func_idx);
            return;
        };
        let trace_id = self.jit.alloc_trace_id();
        let register_count = function.register_count;
        let entry_table = self.jit.function_entry_table();
        match JitCompiler::new().compile_function(&trace, trace_id, register_count, entry_table) {
            Ok(code) => {
                crate::jit::log(|| format!("✅ JIT: function {} compiled", func_idx));
                self.jit.store_function_code(func_idx, code);
            }
            Err(e) => {
                crate::jit::log(|| format!("❌ JIT: function {} compile failed: {}", func_idx, e));
                self.jit.function_not_compilable(func_idx);
            }
        }
    }

    /// Is the callee register a plain bytecode function (not a closure,
    /// native, Lua value or callable table) whose call can take the fast
    /// path?
    #[inline]
    fn plain_function_callee(&self, func_reg: Register) -> bool {
        let Some(frame) = self.call_stack.last() else {
            return false;
        };
        match frame.registers[func_reg as usize] {
            Value::Function(idx) => self
                .call_meta
                .get(idx)
                .is_some_and(|meta| !meta.lua_function),
            _ => false,
        }
    }

    /// The fast path of `Instruction::Call` for a plain bytecode callee:
    /// same checks as `make_call_frame_with` with `ArgCheck::Shallow`, but
    /// the arguments are copied straight from the caller's registers into
    /// the new frame, with no intermediate buffer and no per-call signature
    /// walk. `plain_function_callee` must have said yes.
    fn bytecode_call_frame(
        &mut self,
        func_reg: Register,
        first_arg: Register,
        arg_count: u8,
        dest_reg: Register,
    ) -> Result<Box<CallFrame>> {
        let caller = self.call_stack.len() - 1;
        let Value::Function(function_idx) = self.call_stack[caller].registers[func_reg as usize]
        else {
            unreachable!("plain_function_callee checked the callee");
        };
        let function = &self.functions[function_idx];
        if arg_count != function.param_count {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Function {} expects {} arguments, got {}",
                    function.name, function.param_count, arg_count
                ),
            });
        }
        let register_count = function.register_count;
        let mut frame = self.take_frame(function_idx, Some(dest_reg), register_count)?;
        for index in 0..arg_count as usize {
            let value = &self.call_stack[caller].registers[first_arg.wrapping_add(index as u8) as usize];
            let expected = self.call_meta[function_idx].params[index];
            if !expected.matches(value) {
                let function = &self.functions[function_idx];
                let ty = function
                    .signature
                    .as_ref()
                    .map(|signature| signature.params[index].to_string())
                    .unwrap_or_default();
                let message = format!(
                    "Function {} argument {} expects {}, got {:?}",
                    function.name,
                    index + 1,
                    ty,
                    value.type_of()
                );
                self.recycle_frame(frame);
                return Err(LustError::RuntimeError { message });
            }
            self.cycle_collector.register_graph(value);
            frame.registers[index] = value.fast_clone();
        }
        let recursive = self
            .call_stack
            .iter()
            .any(|existing| existing.function_idx == function_idx);
        self.jit.record_function_call(recursive);
        Ok(frame)
    }

    pub(super) fn make_checked_call_frame(
        &mut self,
        function_idx: usize,
        return_dest: Option<Register>,
        args: Vec<Value>,
        upvalues: Vec<Value>,
    ) -> Result<Box<CallFrame>> {
        self.make_call_frame_with(function_idx, return_dest, args, upvalues, ArgCheck::Shallow)
    }

    /// Build a frame for a call from the host or a dynamic value: every
    /// argument is validated in full against the signature, including the
    /// elements of containers.
    pub(super) fn make_call_frame(
        &mut self,
        function_idx: usize,
        return_dest: Option<Register>,
        args: Vec<Value>,
        upvalues: Vec<Value>,
    ) -> Result<Box<CallFrame>> {
        self.make_call_frame_with(function_idx, return_dest, args, upvalues, ArgCheck::Deep)
    }

    fn make_call_frame_with(
        &mut self,
        function_idx: usize,
        return_dest: Option<Register>,
        mut args: Vec<Value>,
        upvalues: Vec<Value>,
        check: ArgCheck,
    ) -> Result<Box<CallFrame>> {
        let function = self
            .functions
            .get(function_idx)
            .ok_or_else(|| LustError::RuntimeError {
                message: format!("Invalid function index {}", function_idx),
            })?;
        let is_lua_function = function.signature.as_ref().is_some_and(|signature| {
            let lua_params = signature
                .params
                .iter()
                .all(|ty| matches!(&ty.kind, TypeKind::Named(name) if name == "LuaValue"));
            let lua_multi_return = matches!(
                &signature.return_type.kind,
                TypeKind::Array(inner)
                    if matches!(&inner.kind, TypeKind::Named(name) if name == "LuaValue")
            );
            lua_params && (!signature.params.is_empty() || lua_multi_return)
        });
        if is_lua_function {
            args.resize(function.param_count as usize, Value::Nil);
            args.truncate(function.param_count as usize);
        }
        if args.len() != function.param_count as usize {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Function {} expects {} arguments, got {}",
                    function.name,
                    function.param_count,
                    args.len()
                ),
            });
        }
        if let Some(signature) = &function.signature
            && signature.params.len() == args.len()
        {
            for (index, (value, ty)) in args.iter().zip(&signature.params).enumerate() {
                let ok = match check {
                    ArgCheck::Deep => self.value_matches_type(value, ty),
                    ArgCheck::Shallow => Self::value_matches_type_shallow(value, ty),
                };
                if !ok {
                    return Err(LustError::RuntimeError {
                        message: format!(
                            "Function {} argument {} expects {}, got {:?}",
                            function.name,
                            index + 1,
                            ty,
                            value.type_of()
                        ),
                    });
                }
            }
        }

        let register_count = function.register_count;
        let mut frame = self.take_frame(function_idx, return_dest, register_count)?;
        let recursive = self
            .call_stack
            .iter()
            .any(|existing| existing.function_idx == function_idx);
        self.jit.record_function_call(recursive);
        frame.upvalues = upvalues;
        for (index, arg) in args.drain(..).enumerate() {
            self.observe_value_graph(&arg);
            frame.registers[index] = arg;
        }
        // Keep the (now empty) argument buffer so the next call allocates
        // nothing.
        if args.capacity() > self.arg_scratch.capacity() {
            self.arg_scratch = args;
        }
        Ok(frame)
    }

    /// A frame for `function_idx`, from the pool when one is available.
    /// Fails when pushing it would exceed the stack depth limit (checked
    /// here, where the stack grows, rather than on every instruction).
    pub(super) fn take_frame(
        &mut self,
        function_idx: usize,
        return_dest: Option<Register>,
        register_count: u8,
    ) -> Result<Box<CallFrame>> {
        if self.call_stack.len() >= self.max_stack_depth {
            return Err(LustError::RuntimeError {
                message: "Stack overflow".to_string(),
            });
        }
        Ok(match self.frame_pool.pop() {
            Some(mut frame) => {
                // Pooled frames come back with every register they used
                // reset to Nil (see `recycle_frame`); only the bookkeeping
                // needs setting.
                frame.function_idx = function_idx;
                frame.ip = 0;
                frame.base_register = 0;
                frame.return_dest = return_dest;
                frame
            }
            None => CallFrame::new(function_idx, return_dest, register_count),
        })
    }

    /// An O(1) argument check for calls the typechecker already validated:
    /// scalar kinds and container kinds are verified, contents are not.
    /// Anything the checker treats dynamically (`unknown`, generics,
    /// unions, function types, Lua values) is accepted; typed instructions
    /// still guard the payloads they read.
    fn value_matches_type_shallow(value: &Value, ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Int => matches!(value, Value::Int(_)),
            TypeKind::Float => matches!(value, Value::Float(_)),
            TypeKind::String => matches!(value, Value::String(_)),
            TypeKind::Bool => matches!(value, Value::Bool(_)),
            TypeKind::Unit => matches!(value, Value::Nil),
            TypeKind::Array(_) => matches!(value, Value::Array(_)),
            TypeKind::Map(..) => matches!(value, Value::Map(_)),
            TypeKind::Tuple(_) => matches!(value, Value::Tuple(_)),
            _ => true,
        }
    }

    /// Keep a popped frame for reuse, with the registers it used reset to
    /// Nil (dropping their values now, as dropping the frame would have).
    pub(super) fn recycle_frame(&mut self, mut frame: Box<CallFrame>) {
        if self.frame_pool.len() >= super::FRAME_POOL_LIMIT {
            return;
        }
        let register_count = self
            .functions
            .get(frame.function_idx)
            .map(|f| f.register_count)
            .unwrap_or(u8::MAX);
        frame.reset(frame.function_idx, None, register_count);
        self.frame_pool.push(frame);
    }

    fn validate_function_return(&self, function_idx: usize, value: &Value) -> Result<()> {
        // A return executes bytecode the typechecker validated; check the
        // kind only (precomputed in `CallMeta`). Container contents are
        // checked where they are read, by typed instructions.
        let meta = &self.call_meta[function_idx];
        if meta.return_kind.matches(value)
            || (meta.lua_multi_return && matches!(value, Value::Nil))
        {
            return Ok(());
        }
        let function = &self.functions[function_idx];
        Err(LustError::RuntimeError {
            message: format!(
                "Function {} must return {}, got {:?}",
                function.name,
                function
                    .signature
                    .as_ref()
                    .map(|signature| signature.return_type.to_string())
                    .unwrap_or_default(),
                value.type_of()
            ),
        })
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
        if let Some(last) = type_name.rsplit('.').next()
            && last != type_name
        {
            candidates.push(format!("{}:{}", last, HASH_KEY_METHOD));
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

        let Some(signature) = self
            .functions
            .get(func_idx)
            .and_then(|func| func.signature.as_ref())
        else {
            return false;
        };

        let rendered = signature.to_string();
        if rendered == type_name {
            return true;
        }

        let (Some(rendered_params), Some(expected_params)) = (
            function_type_params(&rendered),
            function_type_params(type_name),
        ) else {
            return false;
        };
        if rendered_params != expected_params {
            return false;
        }

        match &signature.return_type.kind {
            TypeKind::Unit => !type_string_declares_return(type_name),
            _ => false,
        }
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

    #[inline]
    pub(super) fn set_register(&mut self, reg: Register, value: Value) -> Result<()> {
        self.cycle_collector.register_value(&value);
        let frame = self
            .call_stack
            .last_mut()
            .ok_or_else(|| LustError::RuntimeError {
                message: "Empty call stack".to_string(),
            })?;
        let slot = &mut frame.registers[reg as usize];
        if slot.is_plain() {
            // SAFETY: the old value owns nothing, so it needs no drop.
            unsafe { core::ptr::write(slot, value) };
        } else {
            *slot = value;
        }
        self.maybe_collect_cycles();
        Ok(())
    }

    pub(super) fn handle_native_call_outcome(
        &mut self,
        dest: Register,
        outcome: NativeCallResult,
    ) -> Result<()> {
        #[cfg(feature = "std")]
        if lua_socket_trace_enabled()
            && let NativeCallResult::Return(value) = &outcome
            && let Value::Array(arr) = value
        {
            let borrowed = arr.borrow();
            let interesting = borrowed.len() > 1
                && matches!(
                    borrowed.first(),
                    Some(Value::Enum { enum_name, variant, .. })
                        if enum_name == "LuaValue" && variant == "Nil"
                );
            if interesting {
                let func_name = self
                    .call_stack
                    .last()
                    .and_then(|frame| self.functions.get(frame.function_idx))
                    .map(|f| f.name.as_str())
                    .unwrap_or("<unknown>");
                eprintln!(
                    "[lua-socket] native return in {} dest=R{} len={} value={}",
                    func_name,
                    dest,
                    borrowed.len(),
                    value
                );
            }
        }
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
        // Lua compatibility: honor __call metamethod on Lua tables/userdata.
        // Lua semantics: if value has metatable.__call, calling it invokes that function with the
        // receiver as the first argument.
        let mut args = args;
        let maybe_lua_call = {
            let mut current = func;
            let mut receiver_wrapped = func.clone();

            if let Value::Struct { name, .. } = func {
                if name == "LuaTable" {
                    receiver_wrapped = Value::enum_variant("LuaValue", "Table", vec![func.clone()]);
                } else if name == "LuaUserdata" {
                    receiver_wrapped =
                        Value::enum_variant("LuaValue", "Userdata", vec![func.clone()]);
                }
            }

            if let Value::Enum {
                enum_name,
                variant,
                values,
            } = func
                && enum_name == "LuaValue"
                && (variant == "Table" || variant == "Userdata")
                && let Some(inner) = values.as_ref().and_then(|vals| vals.first())
            {
                current = inner;
            }

            if let Value::Struct { name, .. } = current {
                if name == "LuaTable" || name == "LuaUserdata" {
                    if let Some(Value::Map(meta_rc)) = current.struct_get_field("metamethods") {
                        meta_rc
                            .borrow()
                            .get(&ValueKey::string("__call".to_string()))
                            .cloned()
                            .map(|callable| (callable, receiver_wrapped))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        };

        if let Some((callable, receiver)) = maybe_lua_call {
            // eprintln!("DEBUG: __call metamethod found, invoking with receiver");
            let mut call_args = Vec::with_capacity(args.len() + 1);
            call_args.push(receiver);
            call_args.append(&mut args);
            return self.call_value(&callable, call_args);
        } else {
            // if let Value::Struct { name, .. } = func {
            //     if name == "LuaTable" || name == "LuaUserdata" {
            //         eprintln!("DEBUG: Calling LuaTable/Userdata but no __call found in metamethods");
            //     }
            // }
        }

        match func {
            #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
            Value::Enum {
                enum_name,
                variant,
                values,
            } if enum_name == "LuaValue" && variant == "Function" => {
                let handle = values
                    .as_ref()
                    .and_then(|vals| vals.first())
                    .and_then(|v| v.struct_get_field("handle"))
                    .and_then(|v| v.as_int())
                    .map(|i| i as usize)
                    .ok_or_else(|| LustError::RuntimeError {
                        message: "LuaValue function missing handle".to_string(),
                    })?;
                let inner = crate::lua_compat::lookup_lust_function(handle).ok_or_else(|| {
                    LustError::RuntimeError {
                        message: format!(
                            "LuaValue function handle {} was not registered with VM",
                            handle
                        ),
                    }
                })?;
                self.call_value(&inner, args)
            }
            Value::Function(func_idx) => {
                let saved_pending_return_value = self.pending_return_value.clone();
                let saved_pending_return_dest = self.pending_return_dest;
                let saved_pending_task_signal = self.pending_task_signal.clone();
                let saved_last_task_signal = self.last_task_signal.clone();

                let frame = self.make_call_frame(*func_idx, None, args, Vec::new())?;

                let stack_depth_before = self.call_stack.len();
                self.call_stack.push(frame);
                let previous_target = self.call_until_depth;
                self.call_until_depth = Some(stack_depth_before);
                // Compiled code for the function runs it now; `run` then
                // either hands back the pending return value or carries on
                // interpreting from wherever the native code exited.
                let native = if self.jit.enabled && self.trace_recorder.is_none() {
                    self.run_compiled_function(*func_idx).map(|_| ())
                } else {
                    Ok(())
                };
                let run_result = native.and_then(|()| self.run());
                self.call_until_depth = previous_target;
                match run_result {
                    Ok(value) => Ok(value),
                    Err(err) => {
                        let annotated = self.annotate_runtime_error(err);
                        while self.call_stack.len() > stack_depth_before {
                            if let Some(frame) = self.call_stack.pop() {
                                self.recycle_frame(frame);
                            }
                        }
                        self.pending_return_value = saved_pending_return_value;
                        self.pending_return_dest = saved_pending_return_dest;
                        self.pending_task_signal = saved_pending_task_signal;
                        self.last_task_signal = saved_last_task_signal;
                        Err(annotated)
                    }
                }
            }

            Value::Closure {
                function_idx: func_idx,
                upvalues,
            } => {
                let saved_pending_return_value = self.pending_return_value.clone();
                let saved_pending_return_dest = self.pending_return_dest;
                let saved_pending_task_signal = self.pending_task_signal.clone();
                let saved_last_task_signal = self.last_task_signal.clone();

                let upvalue_values: Vec<Value> = upvalues.iter().map(|uv| uv.get()).collect();
                let frame = self.make_call_frame(*func_idx, None, args, upvalue_values)?;

                let stack_depth_before = self.call_stack.len();
                self.call_stack.push(frame);
                let previous_target = self.call_until_depth;
                self.call_until_depth = Some(stack_depth_before);
                let run_result = self.run();
                self.call_until_depth = previous_target;
                match run_result {
                    Ok(value) => Ok(value),
                    Err(err) => {
                        let annotated = self.annotate_runtime_error(err);
                        while self.call_stack.len() > stack_depth_before {
                            if let Some(frame) = self.call_stack.pop() {
                                self.recycle_frame(frame);
                            }
                        }
                        self.pending_return_value = saved_pending_return_value;
                        self.pending_return_dest = saved_pending_return_dest;
                        self.pending_task_signal = saved_pending_task_signal;
                        self.last_task_signal = saved_last_task_signal;
                        Err(annotated)
                    }
                }
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

fn type_string_declares_return(type_name: &str) -> bool {
    if !type_name.starts_with("function(") {
        return true;
    }
    let bytes = type_name.as_bytes();
    let mut depth = 0usize;
    for (index, byte) in bytes.iter().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return bytes[index + 1..].first() == Some(&b':');
                }
            }
            _ => {}
        }
    }
    true
}

fn function_type_params(type_name: &str) -> Option<&str> {
    let rest = type_name.strip_prefix("function(")?;
    let bytes = rest.as_bytes();
    let mut depth = 1usize;
    for (index, byte) in bytes.iter().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&rest[..index]);
                }
            }
            _ => {}
        }
    }
    None
}

/// `LUST_LUA_SOCKET_TRACE` is consulted on every native call; read the
/// environment once rather than per call.
#[cfg(feature = "std")]
fn lua_socket_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("LUST_LUA_SOCKET_TRACE").is_some())
}

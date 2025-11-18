use super::*;
impl VM {
    pub(super) fn build_stack_trace(&self) -> Vec<StackFrame> {
        let mut frames = Vec::new();
        for frame in &self.call_stack {
            let function = &self.functions[frame.function_idx];
            let ip_index = frame.ip.saturating_sub(1);
            let line = function.chunk.lines.get(ip_index).copied().unwrap_or(0);
            frames.push(StackFrame::new(function.name.clone(), line, ip_index));
        }

        frames
    }

    pub(super) fn annotate_runtime_error(&self, err: LustError) -> LustError {
        match err {
            LustError::RuntimeError { message } => {
                if let Some(frame) = self.call_stack.last() {
                    let function = &self.functions[frame.function_idx];
                    let ip_index = frame.ip.saturating_sub(1);
                    let line = function.chunk.lines.get(ip_index).copied().unwrap_or(0);
                    let stack_trace = self.build_stack_trace();
                    LustError::RuntimeErrorWithTrace {
                        message,
                        function: function.name.clone(),
                        line,
                        stack_trace,
                    }
                } else {
                    LustError::RuntimeError { message }
                }
            }

            other => other,
        }
    }

    pub(super) fn invoke_tostring(&mut self, value: &Value, type_name: &str) -> Result<Rc<String>> {
        if self.type_has_tostring(type_name) {
            let cache_key = Self::struct_cache_key(value);
            if let Some(key) = cache_key {
                if let Some(cached) = self.struct_tostring_cache.get(&key) {
                    return Ok(cached.clone());
                }
            }

            let result = self.call_builtin_method(value, TO_STRING_METHOD, Vec::new())?;
            match result {
                Value::String(s) => {
                    let rc = s.clone();
                    if let Some(key) = cache_key {
                        self.struct_tostring_cache.insert(key, rc.clone());
                    }

                    Ok(rc)
                }

                other => Err(LustError::RuntimeError {
                    message: format!(
                        "{}: to_string() must return string, got {:?}",
                        type_name,
                        other.type_of()
                    ),
                }),
            }
        } else {
            Ok(Rc::new(value.to_string()))
        }
    }

    pub(super) fn type_has_tostring(&self, type_name: &str) -> bool {
        let mut candidates = vec![type_name];
        if let Some(last) = type_name.rsplit('.').next() {
            if last != type_name {
                candidates.push(last);
            }
        }

        for candidate in candidates {
            let key = (candidate.to_string(), TO_STRING_TRAIT.to_string());
            if self.trait_impls.contains_key(&key) {
                return true;
            }

            let mangled = format!("{}:{}", candidate, TO_STRING_METHOD);
            if self.functions.iter().any(|f| f.name == mangled) {
                return true;
            }
        }

        false
    }

    pub(super) fn struct_cache_key(value: &Value) -> Option<usize> {
        if let Value::Struct { fields, .. } = value {
            Some(Rc::as_ptr(fields) as usize)
        } else {
            None
        }
    }

    pub(super) fn handle_guard_failure(
        &mut self,
        trace_id: crate::jit::TraceId,
        guard_index: usize,
        _func_idx: usize,
    ) -> Result<()> {
        use crate::jit::{GuardKind, SIDE_EXIT_THRESHOLD};
        if !self.jit.enabled {
            return Ok(());
        }

        let should_record_side_trace = if let Some(trace) = self.jit.get_trace_mut(trace_id) {
            if guard_index < trace.guards.len() {
                let guard = &mut trace.guards[guard_index];
                guard.fail_count += 1;
                crate::jit::log(|| {
                    format!(
                        "⚠️  JIT: Guard #{} failed (count: {})",
                        guard_index, guard.fail_count
                    )
                });
                if guard.fail_count >= SIDE_EXIT_THRESHOLD {
                    if let GuardKind::NestedLoop {
                        function_idx,
                        loop_start_ip,
                    } = guard.kind
                    {
                        if guard.side_trace.is_none() {
                            crate::jit::log(|| {
                                format!(
                                    "🔥 JIT: Hot side exit detected (guard #{}, failed {} times)",
                                    guard_index, guard.fail_count
                                )
                            });
                            crate::jit::log(|| {
                                format!(
                                    "🌳 JIT: Will compile side trace for nested loop at func {} ip {}...",
                                    function_idx, loop_start_ip
                                )
                            });
                            Some((function_idx, loop_start_ip))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        if let Some((function_idx, loop_start_ip)) = should_record_side_trace {
            if self.trace_recorder.is_none() {
                self.side_trace_context = Some((trace_id, guard_index));
                let mut recorder =
                    TraceRecorder::new(function_idx, loop_start_ip, crate::jit::MAX_TRACE_LENGTH);
                // Specialize loop-invariant values at side trace entry
                {
                    let frame = self.call_stack.last().unwrap();
                    let func = &self.functions[function_idx];
                    recorder.specialize_trace_inputs(&frame.registers, func);
                }
                self.trace_recorder = Some(recorder);
            }
        }

        Ok(())
    }
}

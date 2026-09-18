use super::*;
impl VM {
    pub(super) fn abandon_trace_recording(&mut self) {
        if let Some(recorder) = self.trace_recorder.take() {
            if self.side_trace_context.take().is_none() {
                self.jit
                    .recording_aborted(recorder.trace.function_idx, recorder.trace.start_ip);
            } else {
                self.jit.side_recording_aborted();
            }
        } else {
            self.side_trace_context = None;
        }
        self.skip_next_trace_record = false;
    }

    pub(super) fn build_stack_trace(&self) -> Vec<StackFrame> {
        let mut frames = Vec::new();
        for frame in &self.call_stack {
            let function = &self.functions[frame.function_idx];
            let ip_index = frame.ip.saturating_sub(1);
            #[cfg(feature = "std")]
            let line = function.chunk.lines.get(ip_index).copied().unwrap_or(0);
            #[cfg(not(feature = "std"))]
            let line = 0usize;
            frames.push(StackFrame::new(function.name.clone(), line, ip_index));
        }

        frames
    }

    pub(super) fn annotate_runtime_error(&self, err: LustError) -> LustError {
        match err {
            LustError::RuntimeError { message } => {
                if let Some(frame) = self.call_stack.last() {
                    let function = &self.functions[frame.function_idx];
                    #[cfg(feature = "std")]
                    let ip_index = frame.ip.saturating_sub(1);
                    #[cfg(feature = "std")]
                    let line = function.chunk.lines.get(ip_index).copied().unwrap_or(0);
                    #[cfg(not(feature = "std"))]
                    let line = 0usize;
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
            if let Some(key) = cache_key
                && let Some(cached) = self.struct_tostring_cache.get(&key)
            {
                return Ok(cached.clone());
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
        if let Some(last) = type_name.rsplit('.').next()
            && last != type_name
        {
            candidates.push(last);
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

    pub(super) fn type_has_hashkey(&self, type_name: &str) -> bool {
        let mut candidates = vec![type_name];
        if let Some(last) = type_name.rsplit('.').next()
            && last != type_name
        {
            candidates.push(last);
        }

        for candidate in candidates {
            let key = (candidate.to_string(), HASH_KEY_TRAIT.to_string());
            if self.trait_impls.contains_key(&key) {
                return true;
            }

            let mangled = format!("{}:{}", candidate, HASH_KEY_METHOD);
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

    /// Count a guard exit. Nested loops run through their own root trace
    /// from inside the outer trace (`jit_run_nested_loop`), so no guard kind
    /// grows a side trace any more; the side-trace recording that used to
    /// start here after `SIDE_EXIT_THRESHOLD` failures produced a loop trace
    /// whose completion left the frame's ip at the outer back-edge rather
    /// than at the inner loop's exit.
    pub(super) fn handle_guard_failure(
        &mut self,
        trace_id: crate::jit::TraceId,
        guard_index: usize,
        _func_idx: usize,
    ) -> Result<()> {
        if !self.jit.enabled {
            return Ok(());
        }
        if let Some(trace) = self.jit.get_trace_mut(trace_id)
            && let Some(guard) = trace.guards.get_mut(guard_index)
        {
            guard.fail_count += 1;
            crate::jit::log(|| {
                format!(
                    "⚠️  JIT: Guard #{} failed (count: {})",
                    guard_index, guard.fail_count
                )
            });
        }
        Ok(())
    }
}

/// Run the nested loop at `(function_idx, loop_start_ip)` of the frame whose
/// registers are `registers` through the loop's own root trace, from inside
/// the outer loop's native code.
///
/// Returns 0 when the loop ran to its normal exit and execution can carry
/// on natively at `resume_ip`. Returns 1 when the interpreter has to take
/// over: the outer trace exits through its `NestedLoop` guard, whose bailout
/// ip is the inner back-edge (the interpreter then runs the loop itself,
/// compiling it when it gets hot), unless `nested_loop_exit_ip` says where
/// the inner trace bailed out instead. An error raised by the
/// inner trace is left in `pending_jit_error` for the guard-exit path.
///
/// # Safety
/// `vm` is null (from backend unit tests) or the VM executing the outer
/// trace, and `registers` is that VM's current frame.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_run_nested_loop(
    vm: *mut VM,
    registers: *mut Value,
    function_idx: usize,
    loop_start_ip: usize,
    resume_ip: usize,
) -> i32 {
    use crate::jit::GuardKind;
    if vm.is_null() {
        return 1;
    }
    let vm = unsafe { &mut *vm };
    let Some(trace_id) = vm
        .jit
        .root_traces
        .get(&(function_idx, loop_start_ip))
        .copied()
    else {
        return 1;
    };
    loop {
        let Some(trace) = vm.jit.trace_handle(trace_id) else {
            return 1;
        };
        let cost = trace.trace.ops.len() + trace.trace.preamble.len() + trace.trace.postamble.len();
        if let Err(error) = vm.budgets.charge_gas(core::cmp::max(1, cost) as u64) {
            vm.pending_jit_error = Some(error);
            return 1;
        }
        vm.jit.record_native_entry();
        vm.pending_jit_error = None;
        let result = trace.execute(registers, vm as *mut VM, core::ptr::null());
        drop(trace);
        if result == 0 {
            if vm.current_task.is_some() && vm.pending_task_signal.is_some() {
                // Let the interpreter's back-edge handling see the signal.
                return 1;
            }
            continue;
        }
        if result > 0 {
            let guard_index = (result - 1) as usize;
            vm.jit.record_guard_exit();
            let guard = vm
                .jit
                .get_trace(trace_id)
                .and_then(|trace| trace.guards.get(guard_index))
                .map(|guard| (guard.bailout_ip, guard.kind.clone()));
            let _ = vm.handle_guard_failure(trace_id, guard_index, function_idx);
            let Some((bailout_ip, kind)) = guard else {
                return 1;
            };
            let reusable_exit = matches!(
                kind,
                GuardKind::Truthy { .. } | GuardKind::Falsy { .. } | GuardKind::NestedLoop { .. }
            );
            if !reusable_exit {
                vm.jit.evict_root_trace(function_idx, loop_start_ip);
            }
            if reusable_exit && bailout_ip == resume_ip {
                return 0;
            }
            // A `NestedLoop` exit of the inner trace comes from a deeper
            // level of this helper, which has already recorded where the
            // interpreter resumes.
            if !matches!(kind, GuardKind::NestedLoop { .. }) || vm.nested_loop_exit_ip.is_none() {
                vm.nested_loop_exit_ip = Some(bailout_ip);
            }
            return 1;
        }
        // Failure: same recovery as an interpreter-entered trace (see the
        // dispatch loop), with the resume ip handed to the guard-exit path.
        vm.jit.record_execution_failure();
        let resume = if result <= -2 {
            vm.jit
                .get_trace(trace_id)
                .and_then(|trace| trace.fail_sites.get((-result - 2) as usize))
                .copied()
        } else {
            None
        };
        vm.nested_loop_exit_ip = Some(resume.unwrap_or(loop_start_ip));
        vm.jit.evict_root_trace(function_idx, loop_start_ip);
        return 1;
    }
}

/// The record a trace pushes for each inlined call, kept in a chain from
/// the innermost frame outward (x21 on aarch64, r15 on x86_64). The callee
/// registers live directly below it. Layout is shared with the backends'
/// `compile_inline_call`.
#[repr(C)]
pub struct JitInlineRecord {
    /// Registers the callee frame holds.
    pub value_count: usize,
    /// The caller's register array (the trace's own for the outermost
    /// record, the enclosing inline frame's otherwise).
    pub caller_regs: *mut Value,
    /// The enclosing inline record, null for the outermost.
    pub prev: *const JitInlineRecord,
    /// Backend-private (x86_64 keeps its alignment padding here).
    pub reserved: usize,
    pub function_idx: usize,
    /// Caller register the callee's result goes to.
    pub return_dest: usize,
    /// Caller register holding the callee value (a closure's upvalues come
    /// from it).
    pub callee_reg: usize,
    /// Where the caller continues once the callee returns: the instruction
    /// after the call.
    pub caller_resume_ip: usize,
}

/// Turn the inline-call frames a trace is exiting from into interpreter
/// frames, so execution resumes inside the callee rather than re-running
/// the call. `record` is the innermost record and `regs` its frame's
/// registers; each frame's values are moved into a real `CallFrame`. The
/// frames are pushed outermost first, each caller's ip set to its resume
/// ip; the innermost frame's ip is set afterwards by the guard-exit or
/// fail-site path, exactly as for the trace's own frame. Returns the
/// trace's own register array. With no VM (backend unit tests) the values
/// are dropped instead.
///
/// # Safety
/// `record` is a chain of records the trace pushed, `regs` the innermost
/// frame's registers, and `vm` null or the VM executing the trace.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_materialize_inline_frames(
    vm: *mut VM,
    record: *const JitInlineRecord,
    regs: *mut Value,
) -> *mut Value {
    let mut chain = Vec::new();
    let mut record = record;
    let mut regs = regs;
    while !record.is_null() {
        let r = unsafe { &*record };
        chain.push((r, regs));
        regs = r.caller_regs;
        record = r.prev;
    }
    let root_regs = regs;
    if vm.is_null() {
        for (r, regs) in chain {
            for i in 0..r.value_count {
                unsafe { core::ptr::drop_in_place(regs.add(i)) };
            }
        }
        return root_regs;
    }
    let vm = unsafe { &mut *vm };
    for (r, regs) in chain.into_iter().rev() {
        let register_count = vm
            .functions
            .get(r.function_idx)
            .map(|f| f.register_count)
            .unwrap_or(r.value_count as u8);
        let mut frame = vm.take_frame(r.function_idx, Some(r.return_dest as Register), register_count);
        for i in 0..r.value_count {
            frame.registers[i] = unsafe { core::ptr::read(regs.add(i)) };
        }
        if let Value::Closure { upvalues, .. } = unsafe { &*r.caller_regs.add(r.callee_reg) } {
            frame.upvalues = upvalues.iter().map(|uv| uv.get()).collect();
        }
        if let Some(caller) = vm.call_stack.last_mut() {
            caller.ip = r.caller_resume_ip;
        }
        vm.call_stack.push(frame);
    }
    root_regs
}

use super::*;
use crate::VM;
use hashbrown::HashMap;
impl JitCompiler {
    pub fn new() -> Self {
        Self {
            ops: Assembler::new().unwrap(),
            leaked_constants: Vec::new(),
            fail_stack: Vec::new(),
            exit_stack: Vec::new(),
            inline_depth: 0,
            specialization_registry: SpecializationRegistry::new(),
            specialized_values: HashMap::new(),
            next_specialized_id: 0,
        }
    }

    pub(super) fn current_fail_label(&self) -> dynasmrt::DynamicLabel {
        *self
            .fail_stack
            .last()
            .expect("JIT fail label stack is empty")
    }

    pub(super) fn current_exit_label(&self) -> dynasmrt::DynamicLabel {
        *self
            .exit_stack
            .last()
            .expect("JIT exit label stack is empty")
    }

    pub fn compile_trace(
        &mut self,
        trace: &Trace,
        trace_id: TraceId,
        parent: Option<TraceId>,
        hoisted_constants: Vec<(u8, Value)>,
    ) -> Result<CompiledTrace> {
        let stack_size = Self::compute_stack_size(trace);
        let mut guards = Vec::new();
        let mut guard_index = 0i32;
        let exit_label = self.ops.new_dynamic_label();
        let fail_label = self.ops.new_dynamic_label();
        self.exit_stack.push(exit_label);
        self.fail_stack.push(fail_label);
        crate::jit::log(|| format!("🔧 JIT: Emitting prologue with sub rsp, {}", stack_size));
        dynasm!(self.ops
            ; push rbp
            ; mov rbp, rsp
            ; push rbx
            ; push r12
            ; push r13
            ; push r14
            ; push r15
            ; sub rsp, stack_size
            ; xor r15, r15
            ; mov r12, rdi
            ; mov r13, rsi
        );
        for (dest, value) in &hoisted_constants {
            self.compile_load_const(*dest, value)?;
        }

        // Compile preamble (executed once at trace entry)
        jit::log(|| format!("🔧 JIT: Compiling preamble ({} ops)", trace.preamble.len()));
        self.compile_ops(&trace.preamble, &mut guard_index, &mut guards)?;

        // Create a loop_start label AFTER preamble, BEFORE loop body
        let loop_start_label = self.ops.new_dynamic_label();
        dynasm!(self.ops
            ; => loop_start_label
            ; loop_start:
        );

        // Compile main trace body (the loop)
        let compile_result = self.compile_ops(&trace.ops, &mut guard_index, &mut guards);
        compile_result?;

        // At end of loop body, jump back to loop_start to loop
        dynasm!(self.ops
            ; jmp => loop_start_label
        );

        let unwind_label = self.ops.new_dynamic_label();
        let fail_return_label = self.ops.new_dynamic_label();
        dynasm!(self.ops
            ; => exit_label
            ; exit:
        );

        // Compile postamble (executed once at trace exit)
        jit::log(|| {
            format!(
                "🔧 JIT: Compiling postamble ({} ops)",
                trace.postamble.len()
            )
        });
        self.compile_ops(&trace.postamble, &mut guard_index, &mut guards)?;

        // Now pop the label stacks after everything is compiled
        self.exit_stack.pop();
        self.fail_stack.pop();

        // Set return value to 0 (success) AFTER postamble to avoid clobbering
        dynasm!(self.ops
            ; xor eax, eax
        );

        dynasm!(self.ops
            ; add rsp, stack_size
            ; pop r15
            ; pop r14
            ; pop r13
            ; pop r12
            ; pop rbx
            ; pop rbp
            ; ret
            ; => fail_label
            ; fail:
            ; mov eax, DWORD -1
            ; => unwind_label
            ; test r15, r15
            ; je => fail_return_label
            ; mov eax, DWORD [r15]
            ; mov rbx, rax
            ; add rsp, rbx
            ; mov r12, [r15 + 8]
            ; mov r15, [r15 + 16]
            ; add rsp, 24
            ; jmp => unwind_label
            ; => fail_return_label
            ; jmp => exit_label
        );
        let ops = mem::replace(&mut self.ops, Assembler::new().unwrap());
        let exec_buffer = ops.finalize().unwrap();
        let entry_point = exec_buffer.ptr(dynasmrt::AssemblyOffset(0));
        let entry: extern "C" fn(*mut Value, *mut VM, *const Function) -> i32 =
            unsafe { mem::transmute(entry_point) };
        #[cfg(feature = "std")]
        {
            if std::env::var("LUST_JIT_DUMP").is_ok() {
                use std::{fs, path::PathBuf};
                let len = exec_buffer.len();
                let bytes = unsafe { std::slice::from_raw_parts(entry_point as *const u8, len) };
                let mut path = PathBuf::from("target");
                let _ = fs::create_dir_all(&path);
                path.push(format!(
                    "jit_trace_{}_{}.bin",
                    trace_id.0,
                    parent.map(|p| p.0).unwrap_or(trace.function_idx)
                ));
                if let Err(err) = fs::write(&path, bytes) {
                    crate::jit::log(|| {
                        format!("⚠️  JIT: failed to dump trace to {:?}: {}", path, err)
                    });
                } else {
                    crate::jit::log(|| format!("📝 JIT: Dumped trace bytes to {:?}", path));
                }
            }
        }
        Box::leak(Box::new(exec_buffer));
        let leaked_constants = mem::take(&mut self.leaked_constants);
        Ok(CompiledTrace {
            id: trace_id,
            entry,
            trace: trace.clone(),
            guards,
            parent,
            side_traces: Vec::new(),
            leaked_constants,
            hoisted_constants,
        })
    }

    fn compile_ops(
        &mut self,
        ops: &[TraceOp],
        guard_index: &mut i32,
        guards: &mut Vec<Guard>,
    ) -> Result<()> {
        for op in ops {
            match op {
                TraceOp::LoadConst { dest, value } => {
                    self.compile_load_const(*dest, value)?;
                }

                TraceOp::Move { dest, src } => {
                    self.compile_move(*dest, *src)?;
                }

                TraceOp::Add {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_add_specialized(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }

                TraceOp::Sub {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_sub_specialized(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }

                TraceOp::Mul {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_mul_specialized(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }

                TraceOp::Div {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_div_specialized(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }

                TraceOp::Mod {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_mod_specialized(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }

                TraceOp::Neg { dest, src } => {
                    self.compile_neg(*dest, *src)?;
                }

                TraceOp::Lt { dest, lhs, rhs } => {
                    self.compile_lt(*dest, *lhs, *rhs)?;
                }

                TraceOp::Le { dest, lhs, rhs } => {
                    self.compile_le(*dest, *lhs, *rhs)?;
                }

                TraceOp::Gt { dest, lhs, rhs } => {
                    self.compile_gt(*dest, *lhs, *rhs)?;
                }

                TraceOp::Ge { dest, lhs, rhs } => {
                    self.compile_ge(*dest, *lhs, *rhs)?;
                }

                TraceOp::Eq { dest, lhs, rhs } => {
                    self.compile_eq(*dest, *lhs, *rhs)?;
                }

                TraceOp::Ne { dest, lhs, rhs } => {
                    self.compile_ne(*dest, *lhs, *rhs)?;
                }

                TraceOp::And { dest, lhs, rhs } => {
                    self.compile_and(*dest, *lhs, *rhs)?;
                }

                TraceOp::Or { dest, lhs, rhs } => {
                    self.compile_or(*dest, *lhs, *rhs)?;
                }

                TraceOp::Not { dest, src } => {
                    self.compile_not(*dest, *src)?;
                }

                TraceOp::Concat { dest, lhs, rhs } => {
                    self.compile_concat(*dest, *lhs, *rhs)?;
                }

                TraceOp::GetIndex { dest, array, index } => {
                    self.compile_get_index(*dest, *array, *index)?;
                }

                TraceOp::ArrayLen { dest, array } => {
                    self.compile_array_len(*dest, *array)?;
                }

                TraceOp::GuardNativeFunction { register, function } => {
                    let expected_ptr = function.pointer();
                    crate::jit::log(|| format!("🔒 JIT: guard native reg {}", register));
                    let guard = self.compile_guard_native_function(
                        *register,
                        expected_ptr,
                        *guard_index as usize,
                    )?;
                    guards.push(guard);
                    *guard_index += 1;
                }

                TraceOp::GuardFunction {
                    register,
                    function_idx,
                } => {
                    crate::jit::log(|| {
                        format!(
                            "🔒 JIT: guard function reg {} -> idx {}",
                            register, function_idx
                        )
                    });
                    let guard = self.compile_guard_function(
                        *register,
                        *function_idx,
                        *guard_index as usize,
                    )?;
                    guards.push(guard);
                    *guard_index += 1;
                }

                TraceOp::GuardClosure {
                    register,
                    function_idx,
                    upvalues_ptr,
                } => {
                    crate::jit::log(|| {
                        format!(
                            "🔒 JIT: guard closure reg {} -> idx {}",
                            register, function_idx
                        )
                    });
                    let guard = self.compile_guard_closure(
                        *register,
                        *function_idx,
                        *upvalues_ptr,
                        *guard_index as usize,
                    )?;
                    guards.push(guard);
                    *guard_index += 1;
                }

                TraceOp::CallNative {
                    dest,
                    callee,
                    function,
                    first_arg,
                    arg_count,
                } => {
                    let expected_ptr = function.pointer();
                    self.compile_call_native(*dest, *callee, expected_ptr, *first_arg, *arg_count)?;
                }

                TraceOp::CallFunction {
                    dest,
                    callee,
                    function_idx,
                    first_arg,
                    arg_count,
                    is_closure,
                    upvalues_ptr,
                } => {
                    self.compile_call_function(
                        *dest,
                        *callee,
                        *function_idx,
                        *first_arg,
                        *arg_count,
                        *is_closure,
                        *upvalues_ptr,
                    )?;
                }

                TraceOp::InlineCall {
                    dest,
                    callee,
                    trace,
                } => {
                    self.compile_inline_call(*dest, *callee, trace, guard_index, guards)?;
                }

                TraceOp::CallMethod {
                    dest,
                    object,
                    method_name,
                    first_arg,
                    arg_count,
                } => {
                    // Optimize common method calls with specialized JIT helpers
                    match (method_name.as_str(), *arg_count) {
                        ("push", 1) => {
                            self.compile_array_push(*object, *first_arg)?;
                        }
                        ("is_some", 0) => {
                            self.compile_enum_is_some(*dest, *object)?;
                        }
                        ("unwrap", 0) => {
                            self.compile_enum_unwrap(*dest, *object)?;
                        }
                        _ => {
                            self.compile_call_method(
                                *dest,
                                *object,
                                method_name,
                                *first_arg,
                                *arg_count,
                            )?;
                        }
                    }
                }

                TraceOp::GetField {
                    dest,
                    object,
                    field_name,
                    field_index,
                    value_type,
                    is_weak,
                } => {
                    self.compile_get_field(
                        *dest,
                        *object,
                        field_name,
                        *field_index,
                        *value_type,
                        *is_weak,
                    )?;
                }

                TraceOp::SetField {
                    object,
                    field_name,
                    value,
                    field_index,
                    value_type,
                    is_weak,
                } => {
                    self.compile_set_field(
                        *object,
                        field_name,
                        *value,
                        *field_index,
                        *value_type,
                        *is_weak,
                    )?;
                }

                TraceOp::NewArray {
                    dest,
                    first_element,
                    count,
                } => {
                    self.compile_new_array(*dest, *first_element, *count)?;
                }

                TraceOp::NewStruct {
                    dest,
                    struct_name,
                    field_names,
                    field_registers,
                } => {
                    self.compile_new_struct(*dest, struct_name, field_names, field_registers)?;
                }

                TraceOp::NewEnumUnit {
                    dest,
                    enum_name,
                    variant_name,
                } => {
                    self.compile_new_enum_unit(*dest, enum_name, variant_name)?;
                }

                TraceOp::NewEnumVariant {
                    dest,
                    enum_name,
                    variant_name,
                    value_registers,
                } => {
                    self.compile_new_enum_variant(*dest, enum_name, variant_name, value_registers)?;
                }

                TraceOp::IsEnumVariant {
                    dest,
                    value,
                    enum_name,
                    variant_name,
                } => {
                    self.compile_is_enum_variant(*dest, *value, enum_name, variant_name)?;
                }

                TraceOp::GetEnumValue {
                    dest,
                    enum_reg,
                    index,
                } => {
                    self.compile_get_enum_value(*dest, *enum_reg, *index)?;
                }

                TraceOp::Guard {
                    register,
                    expected_type,
                } => {
                    let guard =
                        self.compile_guard(*register, *expected_type, *guard_index as usize)?;
                    guards.push(guard);
                    *guard_index += 1;
                }

                TraceOp::GuardLoopContinue {
                    condition_register,
                    expect_truthy,
                    bailout_ip,
                } => {
                    let guard = self.compile_truth_guard(
                        *condition_register,
                        *expect_truthy,
                        *bailout_ip,
                        *guard_index as usize,
                    )?;
                    guards.push(guard);
                    *guard_index += 1;
                }

                TraceOp::NestedLoopCall {
                    function_idx,
                    loop_start_ip,
                    bailout_ip,
                } => {
                    // Nested loop call - this will be replaced with a direct call to
                    // the compiled inner loop trace once it's compiled.
                    // For now, exit to interpreter which will:
                    // 1. Run the loop in interpreter
                    // 2. Eventually compile it as a hot trace
                    // 3. Later, this guard can become a side trace that calls the compiled loop

                    let exit_label = self.current_exit_label();
                    jit::log(|| {
                        format!(
                            "🔗 JIT: Nested loop at func {} ip {} - exiting to interpreter (guard #{})",
                            function_idx, loop_start_ip, *guard_index
                        )
                    });
                    guards.push(Guard {
                        index: *guard_index as usize,
                        bailout_ip: *bailout_ip,
                        kind: GuardKind::NestedLoop {
                            function_idx: *function_idx,
                            loop_start_ip: *loop_start_ip,
                        },
                        fail_count: 0,
                        side_trace: None,
                    });
                    let current_guard_index = *guard_index;
                    dynasm!(self.ops
                        ; mov eax, DWORD (current_guard_index + 1)
                        ; jmp => exit_label
                    );
                    *guard_index += 1;
                }

                TraceOp::Unbox {
                    specialized_id,
                    source_reg,
                    layout,
                } => {
                    self.compile_unbox(*specialized_id, *source_reg, layout)?;
                }

                TraceOp::Rebox {
                    dest_reg,
                    specialized_id,
                    layout,
                } => {
                    self.compile_rebox(*dest_reg, *specialized_id, layout)?;
                }

                TraceOp::DropSpecialized {
                    specialized_id,
                    layout,
                } => {
                    self.compile_drop_specialized(*specialized_id, layout)?;
                }

                TraceOp::SpecializedOp { op, operands } => {
                    self.compile_specialized_op(op, operands)?;
                }

                TraceOp::Return { .. } => {}
            }
        }

        Ok(())
    }

    fn compute_stack_size(trace: &Trace) -> i32 {
        let specialized_slots = Self::count_specialized_slots(trace) as i32;
        let specialized_bytes =
            SPECIALIZED_STACK_BASE + (specialized_slots * SPECIALIZED_SLOT_SIZE);
        let mut size = MIN_JIT_STACK_SIZE.max(specialized_bytes);
        let remainder = size % 16;
        if remainder != 8 {
            size += (8 - remainder + 16) % 16;
        }
        crate::jit::log(|| {
            format!(
                "🧮 JIT: Trace requires {} specialized slots → stack {} bytes",
                specialized_slots, size
            )
        });
        size
    }

    fn count_specialized_slots(trace: &Trace) -> usize {
        trace
            .preamble
            .iter()
            .chain(trace.ops.iter())
            .chain(trace.postamble.iter())
            .filter(|op| matches!(op, TraceOp::Unbox { .. }))
            .count()
    }

    fn compile_inline_call(
        &mut self,
        dest: u8,
        callee: u8,
        trace: &InlineTrace,
        guard_index: &mut i32,
        guards: &mut Vec<Guard>,
    ) -> Result<()> {
        self.inline_depth += 1;
        let result = (|| -> Result<()> {
            if trace.register_count == 0 {
                crate::jit::log(|| {
                    format!(
                        "⚠️  JIT: Inline fallback for func {} (no registers)",
                        trace.function_idx
                    )
                });
                return self.compile_call_function(
                    dest,
                    callee,
                    trace.function_idx,
                    trace.first_arg,
                    trace.arg_count,
                    trace.is_closure,
                    trace.upvalues_ptr,
                );
            }

            crate::jit::log(|| {
                format!(
                    "✨ JIT: Inlining call to func {} into register R{}",
                    trace.function_idx, dest
                )
            });

            let value_size = mem::size_of::<Value>() as i32;
            let frame_size = trace.register_count as i32 * value_size;
            let align_adjust = ((16 - (frame_size & 15)) & 15) as i32;
            let metadata_size = 32i32;
            let outer_fail = self.current_fail_label();
            let inline_fail = self.ops.new_dynamic_label();
            let inline_end = self.ops.new_dynamic_label();
            extern "C" {
                fn jit_move_safe(src_ptr: *const Value, dest_ptr: *mut Value) -> u8;
            }

            // Save inline metadata (frame size, caller registers, previous inline frame).
            dynasm!(self.ops
                ; sub rsp, metadata_size
            );
            dynasm!(self.ops
                ; mov eax, DWORD frame_size as _
                ; mov [rsp], rax
                ; mov [rsp + 8], r12
                ; mov [rsp + 16], r15
            );
            dynasm!(self.ops
                ; mov eax, DWORD align_adjust as _
                ; mov [rsp + 24], rax
                ; mov r15, rsp
            );
            if align_adjust != 0 {
                dynasm!(self.ops
                    ; sub rsp, align_adjust
                );
            }
            // Allocate space for callee registers.
            dynasm!(self.ops
                ; sub rsp, frame_size
                ; mov r12, rsp
            );

            for reg in 0..trace.register_count {
                self.compile_load_const(reg, &Value::Nil)?;
            }

            // Copy positional arguments into callee registers.
            for (arg_index, src_reg) in trace.arg_registers.iter().enumerate() {
                let src_offset = (*src_reg as i32) * value_size;
                let dest_offset = (arg_index as i32) * value_size;
                dynasm!(self.ops
                    ; mov r14, [r15 + 8]
                    ; lea rdi, [r14 + src_offset]
                    ; lea rsi, [r12 + dest_offset]
                    ; mov rax, QWORD jit_move_safe as *const () as _
                    ; call rax
                    ; test al, al
                    ; jz =>inline_fail
                );
            }

            self.fail_stack.push(inline_fail);
            let inline_result = self.compile_ops(&trace.body, guard_index, guards);
            self.fail_stack.pop();
            inline_result?;

            if let Some(ret_reg) = trace.return_register {
                let ret_offset = (ret_reg as i32) * value_size;
                let dest_offset = (dest as i32) * value_size;
                dynasm!(self.ops
                    ; mov r14, [r15 + 8]
                    ; lea rdi, [r12 + ret_offset]
                    ; lea rsi, [r14 + dest_offset]
                    ; mov rax, QWORD jit_move_safe as *const () as _
                    ; call rax
                    ; test al, al
                    ; jz =>inline_fail
                );
                dynasm!(self.ops
                    ; add rsp, frame_size
                );
                dynasm!(self.ops
                    ; mov eax, DWORD [r15 + 24]
                    ; add rsp, rax
                    ; mov r12, [r15 + 8]
                    ; mov r15, [r15 + 16]
                    ; add rsp, metadata_size
                    ; jmp => inline_end
                );
            } else {
                dynasm!(self.ops
                    ; add rsp, frame_size
                );
                dynasm!(self.ops
                    ; mov eax, DWORD [r15 + 24]
                    ; add rsp, rax
                    ; mov r12, [r15 + 8]
                    ; mov r15, [r15 + 16]
                    ; add rsp, metadata_size
                );
                self.compile_load_const(dest, &Value::Nil)?;
                dynasm!(self.ops
                    ; jmp => inline_end
                );
            }

            dynasm!(self.ops
                ; => inline_fail
                ; mov eax, DWORD [r15]
                ; mov rbx, rax
                ; add rsp, rbx
            );
            dynasm!(self.ops
            ; mov eax, DWORD [r15 + 24]
            ; add rsp, rax
            ; mov r12, [r15 + 8]
            ; mov r15, [r15 + 16]
            ; add rsp, metadata_size
            ; jmp => outer_fail
            ; => inline_end
            );

            Ok(())
        })();
        self.inline_depth -= 1;
        result
    }
}

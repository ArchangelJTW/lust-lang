use super::*;
use crate::VM;
use hashbrown::HashMap;
impl JitCompiler {
    pub fn new() -> Self {
        Self {
            ops: Assembler::new().unwrap(),
            data: Vec::new(),
            fail_stack: Vec::new(),
            exit_stack: Vec::new(),
            inline_depth: 0,
            specialization_registry: SpecializationRegistry::new(),
            specialized_values: HashMap::new(),
            scalar_registers: HashMap::new(),
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

    pub(super) fn retain_value(&mut self, value: Value) -> *const Value {
        let value = Box::new(value);
        let ptr = value.as_ref() as *const Value;
        self.data.push(JitData::Value(value));
        ptr
    }

    pub(super) fn retain_string(&mut self, value: &str) -> (*const u8, usize) {
        let value: Box<str> = value.into();
        let result = (value.as_ptr(), value.len());
        self.data.push(JitData::String(value));
        result
    }

    pub(super) fn retain_string_pointers(&mut self, pointers: Vec<*const u8>) -> *const *const u8 {
        let pointers = pointers.into_boxed_slice();
        let ptr = pointers.as_ptr();
        self.data.push(JitData::StringPointers(pointers));
        ptr
    }

    pub(super) fn retain_string_lengths(&mut self, lengths: Vec<usize>) -> *const usize {
        let lengths = lengths.into_boxed_slice();
        let ptr = lengths.as_ptr();
        self.data.push(JitData::StringLengths(lengths));
        ptr
    }

    pub fn compile_trace(
        &mut self,
        trace: &Trace,
        trace_id: TraceId,
        parent: Option<TraceId>,
        hoisted_constants: Vec<(u8, Value)>,
    ) -> Result<CompiledTrace> {
        self.scalar_registers.clear();
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
        for slot in 0..Self::count_specialized_slots(trace) as i32 {
            let offset = SPECIALIZED_BASE_OFFSET - slot * SPECIALIZED_SLOT_SIZE;
            dynasm!(self.ops
                ; mov QWORD [rbp + offset], 0
                ; mov QWORD [rbp + offset + 8], 0
                ; mov QWORD [rbp + offset + 16], 0
            );
        }
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
            // Postamble helpers may overwrite eax. Preserve the exit reason in
            // a callee-saved register until specialized state is materialized.
            ; mov r14d, eax
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

        dynasm!(self.ops
            ; mov eax, r14d
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
        let data = mem::take(&mut self.data);
        Ok(CompiledTrace {
            id: trace_id,
            entry,
            _executable: exec_buffer,
            _data: data,
            trace: trace.clone(),
            guards,
            parent,
            side_traces: Vec::new(),
            hoisted_constants,
        })
    }

    fn compile_ops(
        &mut self,
        ops: &[TraceOp],
        guard_index: &mut i32,
        guards: &mut Vec<Guard>,
    ) -> Result<()> {
        let mut skip_next = false;
        for (op_index, op) in ops.iter().enumerate() {
            if skip_next {
                skip_next = false;
                continue;
            }
            if let Some(next) = ops.get(op_index + 1) {
                if let TraceOp::GuardLoopContinue {
                    condition_register, ..
                } = next
                {
                    if self.scalar_registers.contains_key(condition_register)
                        && Self::register_overwritten_before_read(
                            &ops[op_index + 2..],
                            *condition_register,
                        )
                    {
                        if let Some(guard) =
                            self.compile_integer_comparison_guard(op, next, *guard_index as usize)?
                        {
                            guards.push(guard);
                            *guard_index += 1;
                            skip_next = true;
                            continue;
                        }
                    }
                }
                if let TraceOp::LoadConst {
                    dest: constant_register,
                    ..
                } = op
                {
                    if self.scalar_registers.contains_key(constant_register)
                        && Self::register_overwritten_before_read(
                            &ops[op_index + 2..],
                            *constant_register,
                        )
                        && self.compile_integer_add_immediate(op, next)?
                    {
                        self.update_scalar_registers(next);
                        skip_next = true;
                        continue;
                    }
                }
            }
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

                TraceOp::Lt {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_lt(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }

                TraceOp::Le {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_le(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }

                TraceOp::Gt {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_gt(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }

                TraceOp::Ge {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_ge(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }

                TraceOp::Eq {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_eq(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }

                TraceOp::Ne {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_ne(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
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

                TraceOp::TryGetIndex { dest, array, index } => {
                    self.compile_try_get_index(*dest, *array, *index)?;
                }

                TraceOp::ArrayIndexOk {
                    value_dest,
                    condition_dest,
                    array,
                    index,
                } => {
                    self.compile_array_index_ok(*value_dest, *condition_dest, *array, *index)?;
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
                    let outer_scalar_registers = mem::take(&mut self.scalar_registers);
                    self.compile_inline_call(*dest, *callee, trace, guard_index, guards)?;
                    self.scalar_registers = outer_scalar_registers;
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

                TraceOp::TypeIs {
                    dest,
                    value,
                    type_name,
                } => {
                    self.compile_type_is(*dest, *value, type_name)?;
                }

                TraceOp::TryCast {
                    dest,
                    value,
                    type_name,
                } => {
                    self.compile_try_cast(*dest, *value, type_name)?;
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
            self.update_scalar_registers(op);
        }

        Ok(())
    }

    fn compile_integer_add_immediate(
        &mut self,
        load: &TraceOp,
        arithmetic: &TraceOp,
    ) -> Result<bool> {
        let TraceOp::LoadConst {
            dest: constant_register,
            value: Value::Int(immediate),
        } = load
        else {
            return Ok(false);
        };
        let TraceOp::Add {
            dest,
            lhs,
            rhs,
            lhs_type: ValueType::Int,
            rhs_type: ValueType::Int,
        } = arithmetic
        else {
            return Ok(false);
        };
        let source = if lhs == constant_register && rhs != constant_register {
            *rhs
        } else if rhs == constant_register && lhs != constant_register {
            *lhs
        } else {
            return Ok(false);
        };

        self.load_to_rax(source);
        if *immediate == 1 {
            dynasm!(self.ops ; inc rax);
        } else if *immediate == -1 {
            dynasm!(self.ops ; dec rax);
        } else if let Ok(immediate) = i32::try_from(*immediate) {
            dynasm!(self.ops ; add rax, immediate);
        } else {
            dynasm!(self.ops
                ; mov rcx, QWORD *immediate
                ; add rax, rcx
            );
        }
        self.store_from_rax(*dest, ValueTag::Int.as_u8());
        Ok(true)
    }

    fn compile_integer_comparison_guard(
        &mut self,
        comparison: &TraceOp,
        guard: &TraceOp,
        guard_index: usize,
    ) -> Result<Option<Guard>> {
        let (condition_register, lhs, rhs, comparison_kind) = match comparison {
            TraceOp::Lt {
                dest,
                lhs,
                rhs,
                lhs_type: ValueType::Int,
                rhs_type: ValueType::Int,
            } => (*dest, *lhs, *rhs, 0),
            TraceOp::Le {
                dest,
                lhs,
                rhs,
                lhs_type: ValueType::Int,
                rhs_type: ValueType::Int,
            } => (*dest, *lhs, *rhs, 1),
            TraceOp::Gt {
                dest,
                lhs,
                rhs,
                lhs_type: ValueType::Int,
                rhs_type: ValueType::Int,
            } => (*dest, *lhs, *rhs, 2),
            TraceOp::Ge {
                dest,
                lhs,
                rhs,
                lhs_type: ValueType::Int,
                rhs_type: ValueType::Int,
            } => (*dest, *lhs, *rhs, 3),
            _ => return Ok(None),
        };
        let TraceOp::GuardLoopContinue {
            condition_register: guarded_register,
            expect_truthy,
            bailout_ip,
        } = guard
        else {
            return Ok(None);
        };
        if condition_register != *guarded_register {
            return Ok(None);
        }

        let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
        let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        let guard_ok = self.ops.new_dynamic_label();
        dynasm!(self.ops
            ; mov rax, [r12 + lhs_offset + 8]
            ; mov rcx, [r12 + rhs_offset + 8]
            ; cmp rax, rcx
        );
        match (comparison_kind, *expect_truthy) {
            (0, true) | (3, false) => dynasm!(self.ops ; jl =>guard_ok),
            (1, true) | (2, false) => dynasm!(self.ops ; jle =>guard_ok),
            (2, true) | (1, false) => dynasm!(self.ops ; jg =>guard_ok),
            (3, true) | (0, false) => dynasm!(self.ops ; jge =>guard_ok),
            _ => unreachable!(),
        }

        // The interpreter resumes at the branch bytecode, which reads this
        // register. Materialize only the uncommon failed result.
        let failed_value = i64::from(!*expect_truthy);
        dynasm!(self.ops ; mov rax, QWORD failed_value);
        self.store_from_rax(condition_register, ValueTag::Bool.as_u8());
        let guard_return_value = (guard_index + 1) as i32;
        let exit_label = self.current_exit_label();
        dynasm!(self.ops
            ; mov eax, DWORD guard_return_value
            ; jmp =>exit_label
            ; =>guard_ok
        );

        Ok(Some(Guard {
            index: guard_index,
            bailout_ip: *bailout_ip,
            kind: if *expect_truthy {
                GuardKind::Truthy {
                    register: condition_register,
                }
            } else {
                GuardKind::Falsy {
                    register: condition_register,
                }
            },
            fail_count: 0,
            side_trace: None,
        }))
    }

    fn register_overwritten_before_read(ops: &[TraceOp], register: u8) -> bool {
        for op in ops {
            if Self::op_reads_register(op, register) {
                return false;
            }
            if Self::op_may_exit(op) {
                return false;
            }
            if Self::op_writes_register(op, register) {
                return true;
            }
        }
        false
    }

    fn op_may_exit(op: &TraceOp) -> bool {
        matches!(
            op,
            TraceOp::Div { .. }
                | TraceOp::Mod { .. }
                | TraceOp::Concat { .. }
                | TraceOp::GetIndex { .. }
                | TraceOp::TryGetIndex { .. }
                | TraceOp::ArrayIndexOk { .. }
                | TraceOp::ArrayLen { .. }
                | TraceOp::GuardNativeFunction { .. }
                | TraceOp::GuardFunction { .. }
                | TraceOp::GuardClosure { .. }
                | TraceOp::CallNative { .. }
                | TraceOp::CallFunction { .. }
                | TraceOp::InlineCall { .. }
                | TraceOp::CallMethod { .. }
                | TraceOp::GetField { .. }
                | TraceOp::SetField { .. }
                | TraceOp::NewArray { .. }
                | TraceOp::NewStruct { .. }
                | TraceOp::NewEnumUnit { .. }
                | TraceOp::NewEnumVariant { .. }
                | TraceOp::IsEnumVariant { .. }
                | TraceOp::TypeIs { .. }
                | TraceOp::TryCast { .. }
                | TraceOp::GetEnumValue { .. }
                | TraceOp::Guard { .. }
                | TraceOp::GuardLoopContinue { .. }
                | TraceOp::NestedLoopCall { .. }
                | TraceOp::Return { .. }
                | TraceOp::Unbox { .. }
                | TraceOp::Rebox { .. }
                | TraceOp::DropSpecialized { .. }
                | TraceOp::SpecializedOp { .. }
        )
    }

    fn op_reads_register(op: &TraceOp, register: u8) -> bool {
        let in_args =
            |first: u8, count: u8| register >= first && register < first.saturating_add(count);
        match op {
            TraceOp::LoadConst { .. } => false,
            TraceOp::Move { src, .. } | TraceOp::Neg { src, .. } => *src == register,
            TraceOp::Add { lhs, rhs, .. }
            | TraceOp::Sub { lhs, rhs, .. }
            | TraceOp::Mul { lhs, rhs, .. }
            | TraceOp::Div { lhs, rhs, .. }
            | TraceOp::Mod { lhs, rhs, .. }
            | TraceOp::Eq { lhs, rhs, .. }
            | TraceOp::Ne { lhs, rhs, .. }
            | TraceOp::Lt { lhs, rhs, .. }
            | TraceOp::Le { lhs, rhs, .. }
            | TraceOp::Gt { lhs, rhs, .. }
            | TraceOp::Ge { lhs, rhs, .. }
            | TraceOp::And { lhs, rhs, .. }
            | TraceOp::Or { lhs, rhs, .. }
            | TraceOp::Concat { lhs, rhs, .. } => *lhs == register || *rhs == register,
            TraceOp::Not { src, .. } => *src == register,
            TraceOp::GetIndex { array, index, .. }
            | TraceOp::TryGetIndex { array, index, .. }
            | TraceOp::ArrayIndexOk { array, index, .. } => {
                *array == register || *index == register
            }
            TraceOp::ArrayLen { array, .. } => *array == register,
            TraceOp::GuardNativeFunction { register: source, .. }
            | TraceOp::GuardFunction { register: source, .. }
            | TraceOp::GuardClosure { register: source, .. }
            | TraceOp::Guard {
                register: source, ..
            } => *source == register,
            TraceOp::CallNative {
                callee,
                first_arg,
                arg_count,
                ..
            }
            | TraceOp::CallFunction {
                callee,
                first_arg,
                arg_count,
                ..
            } => *callee == register || in_args(*first_arg, *arg_count),
            TraceOp::InlineCall { callee, trace, .. } => {
                *callee == register || trace.arg_registers.contains(&register)
            }
            TraceOp::CallMethod {
                object,
                first_arg,
                arg_count,
                ..
            } => *object == register || in_args(*first_arg, *arg_count),
            TraceOp::GetField { object, .. } => *object == register,
            TraceOp::SetField { object, value, .. } => {
                *object == register || *value == register
            }
            TraceOp::NewArray {
                first_element,
                count,
                ..
            } => in_args(*first_element, *count),
            TraceOp::NewStruct {
                field_registers, ..
            } => field_registers.contains(&register),
            TraceOp::NewEnumVariant {
                value_registers, ..
            } => value_registers.contains(&register),
            TraceOp::IsEnumVariant { value, .. }
            | TraceOp::TypeIs { value, .. }
            | TraceOp::TryCast { value, .. } => *value == register,
            TraceOp::GetEnumValue { enum_reg, .. } => *enum_reg == register,
            TraceOp::GuardLoopContinue {
                condition_register, ..
            } => *condition_register == register,
            TraceOp::Return { value } => *value == Some(register),
            TraceOp::Unbox { source_reg, .. } => *source_reg == register,
            TraceOp::SpecializedOp { operands, .. } => operands.iter().any(|operand| {
                matches!(operand, crate::jit::trace::Operand::Register(source) if *source == register)
            }),
            TraceOp::NewEnumUnit { .. }
            | TraceOp::NestedLoopCall { .. }
            | TraceOp::Rebox { .. }
            | TraceOp::DropSpecialized { .. } => false,
        }
    }

    fn op_writes_register(op: &TraceOp, register: u8) -> bool {
        match op {
            TraceOp::ArrayIndexOk {
                value_dest,
                condition_dest,
                ..
            } => *value_dest == register || *condition_dest == register,
            TraceOp::Rebox { dest_reg, .. } => *dest_reg == register,
            TraceOp::LoadConst { dest, .. }
            | TraceOp::Move { dest, .. }
            | TraceOp::Add { dest, .. }
            | TraceOp::Sub { dest, .. }
            | TraceOp::Mul { dest, .. }
            | TraceOp::Div { dest, .. }
            | TraceOp::Mod { dest, .. }
            | TraceOp::Neg { dest, .. }
            | TraceOp::Eq { dest, .. }
            | TraceOp::Ne { dest, .. }
            | TraceOp::Lt { dest, .. }
            | TraceOp::Le { dest, .. }
            | TraceOp::Gt { dest, .. }
            | TraceOp::Ge { dest, .. }
            | TraceOp::And { dest, .. }
            | TraceOp::Or { dest, .. }
            | TraceOp::Not { dest, .. }
            | TraceOp::Concat { dest, .. }
            | TraceOp::GetIndex { dest, .. }
            | TraceOp::TryGetIndex { dest, .. }
            | TraceOp::ArrayLen { dest, .. }
            | TraceOp::CallNative { dest, .. }
            | TraceOp::CallFunction { dest, .. }
            | TraceOp::InlineCall { dest, .. }
            | TraceOp::CallMethod { dest, .. }
            | TraceOp::GetField { dest, .. }
            | TraceOp::NewArray { dest, .. }
            | TraceOp::NewStruct { dest, .. }
            | TraceOp::NewEnumUnit { dest, .. }
            | TraceOp::NewEnumVariant { dest, .. }
            | TraceOp::IsEnumVariant { dest, .. }
            | TraceOp::TypeIs { dest, .. }
            | TraceOp::TryCast { dest, .. }
            | TraceOp::GetEnumValue { dest, .. } => *dest == register,
            _ => false,
        }
    }

    fn update_scalar_registers(&mut self, op: &TraceOp) {
        let scalar_type = |ty: ValueType| {
            matches!(ty, ValueType::Bool | ValueType::Int | ValueType::Float).then_some(ty)
        };
        let set = |registers: &mut HashMap<u8, ValueType>, register, ty| {
            if let Some(ty) = ty {
                registers.insert(register, ty);
            } else {
                registers.remove(&register);
            }
        };

        match op {
            TraceOp::LoadConst { dest, value } => {
                let ty = match value {
                    Value::Bool(_) => Some(ValueType::Bool),
                    Value::Int(_) => Some(ValueType::Int),
                    Value::Float(_) => Some(ValueType::Float),
                    _ => None,
                };
                set(&mut self.scalar_registers, *dest, ty);
            }
            TraceOp::Move { dest, src } => {
                let ty = self.scalar_registers.get(src).copied();
                set(&mut self.scalar_registers, *dest, ty);
            }
            TraceOp::Add {
                dest,
                lhs_type,
                rhs_type,
                ..
            }
            | TraceOp::Sub {
                dest,
                lhs_type,
                rhs_type,
                ..
            }
            | TraceOp::Mul {
                dest,
                lhs_type,
                rhs_type,
                ..
            }
            | TraceOp::Div {
                dest,
                lhs_type,
                rhs_type,
                ..
            } => {
                let ty = match (*lhs_type, *rhs_type) {
                    (ValueType::Int, ValueType::Int) => Some(ValueType::Int),
                    (ValueType::Int | ValueType::Float, ValueType::Int | ValueType::Float) => {
                        Some(ValueType::Float)
                    }
                    _ => None,
                };
                set(&mut self.scalar_registers, *dest, ty);
            }
            TraceOp::Mod {
                dest,
                lhs_type,
                rhs_type,
                ..
            } => {
                let ty = match (*lhs_type, *rhs_type) {
                    (ValueType::Int, ValueType::Int) => Some(ValueType::Int),
                    _ => None,
                };
                set(&mut self.scalar_registers, *dest, ty);
            }
            TraceOp::Neg { dest, src } => {
                let ty = self.scalar_registers.get(src).copied();
                set(&mut self.scalar_registers, *dest, ty);
            }
            TraceOp::Eq { dest, .. }
            | TraceOp::Ne { dest, .. }
            | TraceOp::Lt { dest, .. }
            | TraceOp::Le { dest, .. }
            | TraceOp::Gt { dest, .. }
            | TraceOp::Ge { dest, .. }
            | TraceOp::And { dest, .. }
            | TraceOp::Or { dest, .. }
            | TraceOp::Not { dest, .. }
            | TraceOp::IsEnumVariant { dest, .. }
            | TraceOp::TypeIs { dest, .. } => {
                self.scalar_registers.insert(*dest, ValueType::Bool);
            }
            TraceOp::ArrayLen { dest, .. } => {
                self.scalar_registers.insert(*dest, ValueType::Int);
            }
            TraceOp::GetField { dest, .. } => {
                self.scalar_registers.remove(dest);
            }
            TraceOp::Guard {
                register,
                expected_type,
            } => {
                set(
                    &mut self.scalar_registers,
                    *register,
                    scalar_type(*expected_type),
                );
            }
            TraceOp::ArrayIndexOk {
                value_dest,
                condition_dest,
                ..
            } => {
                self.scalar_registers.remove(value_dest);
                self.scalar_registers
                    .insert(*condition_dest, ValueType::Bool);
            }
            TraceOp::Concat { dest, .. }
            | TraceOp::GetIndex { dest, .. }
            | TraceOp::TryGetIndex { dest, .. }
            | TraceOp::CallNative { dest, .. }
            | TraceOp::CallFunction { dest, .. }
            | TraceOp::InlineCall { dest, .. }
            | TraceOp::CallMethod { dest, .. }
            | TraceOp::NewArray { dest, .. }
            | TraceOp::NewStruct { dest, .. }
            | TraceOp::NewEnumUnit { dest, .. }
            | TraceOp::NewEnumVariant { dest, .. }
            | TraceOp::TryCast { dest, .. }
            | TraceOp::GetEnumValue { dest, .. } => {
                self.scalar_registers.remove(dest);
            }
            TraceOp::Rebox { dest_reg, .. } => {
                self.scalar_registers.remove(dest_reg);
            }
            TraceOp::SpecializedOp { operands, .. } => {
                for operand in operands {
                    if let crate::jit::trace::Operand::Register(register) = operand {
                        self.scalar_registers.remove(register);
                    }
                }
            }
            TraceOp::SetField { .. }
            | TraceOp::GuardNativeFunction { .. }
            | TraceOp::GuardFunction { .. }
            | TraceOp::GuardClosure { .. }
            | TraceOp::GuardLoopContinue { .. }
            | TraceOp::NestedLoopCall { .. }
            | TraceOp::Return { .. }
            | TraceOp::Unbox { .. }
            | TraceOp::DropSpecialized { .. } => {}
        }
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
            let frame_value_count = trace.register_count as i32;
            let align_adjust = ((16 - (frame_size & 15)) & 15) as i32;
            let metadata_size = 32i32;
            let outer_fail = self.current_fail_label();
            let inline_fail = self.ops.new_dynamic_label();
            let inline_end = self.ops.new_dynamic_label();
            extern "C" {
                fn jit_move_safe(src_ptr: *const Value, dest_ptr: *mut Value) -> u8;
                fn jit_init_nil(dest: *mut Value) -> u8;
                fn jit_drop_values(values: *mut Value, len: usize);
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
                let offset = reg as i32 * value_size;
                dynasm!(self.ops
                    ; lea rdi, [r12 + offset]
                    ; mov rax, QWORD jit_init_nil as *const () as _
                    ; call rax
                );
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
                    ; mov rdi, r12
                    ; mov esi, DWORD frame_value_count
                    ; mov rax, QWORD jit_drop_values as *const () as _
                    ; call rax
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
                    ; mov rdi, r12
                    ; mov esi, DWORD frame_value_count
                    ; mov rax, QWORD jit_drop_values as *const () as _
                    ; call rax
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
                ; mov rdi, r12
                ; mov esi, DWORD frame_value_count
                ; mov rax, QWORD jit_drop_values as *const () as _
                ; call rax
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

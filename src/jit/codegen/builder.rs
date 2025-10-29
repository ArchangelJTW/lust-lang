use super::*;
use crate::VM;
impl JitCompiler {
    pub fn new() -> Self {
        Self {
            ops: Assembler::new().unwrap(),
            leaked_constants: Vec::new(),
        }
    }

    pub fn compile_trace(
        &mut self,
        trace: &Trace,
        trace_id: TraceId,
        parent: Option<TraceId>,
        hoisted_constants: Vec<(u8, Value)>,
    ) -> Result<CompiledTrace> {
        let mut guards = Vec::new();
        let mut guard_index = 0i32;
        dynasm!(self.ops
            ; push rbp
            ; mov rbp, rsp
            ; push rbx
            ; push r12
            ; push r13
            ; push r14
            ; push r15
            ; mov r12, rdi
            ; mov r13, rsi
        );
        for (dest, value) in &hoisted_constants {
            self.compile_load_const(*dest, value)?;
        }

        for (_i, op) in trace.ops.iter().enumerate() {
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
                        guard_index as usize,
                    )?;
                    guards.push(guard);
                    guard_index += 1;
                }

                TraceOp::CallNative {
                    dest,
                    callee,
                    function,
                    first_arg,
                    arg_count,
                } => {
                    let expected_ptr = function.pointer();
                    self.compile_call_native(
                        *dest,
                        *callee,
                        expected_ptr,
                        *first_arg,
                        *arg_count,
                    )?;
                }

                TraceOp::CallMethod {
                    dest,
                    object,
                    method_name,
                    first_arg,
                    arg_count,
                } => {
                    self.compile_call_method(*dest, *object, method_name, *first_arg, *arg_count)?;
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

                TraceOp::NewStruct {
                    dest,
                    struct_name,
                    field_names,
                    field_registers,
                } => {
                    self.compile_new_struct(*dest, struct_name, field_names, field_registers)?;
                }

                TraceOp::Guard {
                    register,
                    expected_type,
                } => {
                    let guard =
                        self.compile_guard(*register, *expected_type, guard_index as usize)?;
                    guards.push(guard);
                    guard_index += 1;
                }

                TraceOp::GuardLoopContinue {
                    condition_register,
                    bailout_ip,
                } => {
                    let guard = self.compile_loop_continue_guard(
                        *condition_register,
                        *bailout_ip,
                        guard_index as usize,
                    )?;
                    guards.push(guard);
                    guard_index += 1;
                }

                TraceOp::NestedLoopCall {
                    function_idx,
                    loop_start_ip,
                    bailout_ip,
                } => {
                    jit::log(|| {
                        format!(
                            "🔗 JIT: Nested loop detected at func {} ip {} (guard #{})",
                            function_idx, loop_start_ip, guard_index
                        )
                    });
                    guards.push(Guard {
                        index: guard_index as usize,
                        bailout_ip: *bailout_ip,
                        kind: GuardKind::NestedLoop {
                            function_idx: *function_idx,
                            loop_start_ip: *loop_start_ip,
                        },
                        fail_count: 0,
                        side_trace: None,
                    });
                    let current_guard_index = guard_index;
                    dynasm!(self.ops
                        ; mov eax, DWORD (current_guard_index + 1)
                        ; jmp >exit
                    );
                    guard_index += 1;
                }

                TraceOp::Return { .. } => {}
            }
        }

        dynasm!(self.ops
            ; xor eax, eax
            ; exit:
            ; pop r15
            ; pop r14
            ; pop r13
            ; pop r12
            ; pop rbx
            ; pop rbp
            ; ret
            ; fail:
            ; mov eax, DWORD -1
            ; jmp <exit
        );
        let ops = mem::replace(&mut self.ops, Assembler::new().unwrap());
        let buffer = ops.finalize().unwrap();
        let entry_point = buffer.ptr(dynasmrt::AssemblyOffset(0));
        let entry: extern "C" fn(*mut Value, *mut VM, *const Function) -> i32 =
            unsafe { mem::transmute(entry_point) };
        Box::leak(Box::new(buffer));
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
}

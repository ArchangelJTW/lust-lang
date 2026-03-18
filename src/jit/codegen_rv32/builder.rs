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
        *self.fail_stack.last().expect("JIT fail label stack is empty")
    }

    pub(super) fn current_exit_label(&self) -> dynasmrt::DynamicLabel {
        *self.exit_stack.last().expect("JIT exit label stack is empty")
    }

    /// Total frame size (local + saved-registers area), rounded to 16 bytes.
    fn compute_frame_size(_trace: &Trace) -> i32 {
        let total = SAVED_REGS_SIZE + MIN_JIT_STACK_SIZE;
        (total + 15) & !15
    }

    pub fn compile_trace(
        &mut self,
        trace: &Trace,
        trace_id: TraceId,
        parent: Option<TraceId>,
        hoisted_constants: Vec<(u8, Value)>,
    ) -> Result<CompiledTrace> {
        let frame_size = Self::compute_frame_size(trace);

        // Saved-register offsets from the top of the frame (sp+frame_size).
        // Layout (highest → lowest address):
        //   [frame_size - 4]  ra
        //   [frame_size - 8]  s0  (frame pointer)
        //   [frame_size - 12] s2  (register-array base)
        //   [frame_size - 16] s3  (VM pointer)
        //   [frame_size - 20] s4  (unwind chain)
        let ra_off  = frame_size - 4;
        let s0_off  = frame_size - 8;
        let s2_off  = frame_size - 12;
        let s3_off  = frame_size - 16;
        let s4_off  = frame_size - 20;

        let mut guards: Vec<Guard> = Vec::new();
        let mut guard_index: i32 = 0;

        let exit_label = self.ops.new_dynamic_label();
        let fail_label = self.ops.new_dynamic_label();
        self.exit_stack.push(exit_label);
        self.fail_stack.push(fail_label);

        jit::log(|| {
            format!(
                "🔧 RV32 JIT: emitting prologue, frame_size={}",
                frame_size
            )
        });

        // ── Prologue ────────────────────────────────────────────────────────
        // Entry: a0 = *mut Value (regs), a1 = *mut VM, a2 = *const Function
        dynasm!(self.ops
            ; .arch riscv32i
            ; addi sp, sp, -frame_size
            ; sw ra,  [sp, ra_off]
            ; sw s0,  [sp, s0_off]
            ; sw s2,  [sp, s2_off]
            ; sw s3,  [sp, s3_off]
            ; sw s4,  [sp, s4_off]
            ; addi s0, sp, frame_size   // frame pointer
            ; mv s2, a0                 // register-array base
            ; mv s3, a1                 // VM pointer
            ; mv s4, zero               // unwind chain = NULL
        );

        // Hoisted constants
        for (dest, value) in &hoisted_constants {
            self.compile_load_const(*dest, value)?;
        }

        // Preamble (executed once on trace entry)
        jit::log(|| format!("🔧 RV32 JIT: preamble ({} ops)", trace.preamble.len()));
        self.compile_ops(&trace.preamble, &mut guard_index, &mut guards)?;

        // Loop-start label (after preamble, before loop body)
        let loop_start = self.ops.new_dynamic_label();
        dynasm!(self.ops
            ; .arch riscv32i
            ; => loop_start
            ; loop_start:
        );

        // Main loop body
        jit::log(|| format!("🔧 RV32 JIT: loop body ({} ops)", trace.ops.len()));
        self.compile_ops(&trace.ops, &mut guard_index, &mut guards)?;

        // Back-edge: jump to loop start
        dynasm!(self.ops
            ; .arch riscv32i
            ; j => loop_start
        );

        // ── Exit label (guard exits + normal trace exit) ─────────────────────
        let unwind_label = self.ops.new_dynamic_label();
        let fail_return_label = self.ops.new_dynamic_label();

        dynasm!(self.ops
            ; .arch riscv32i
            ; => exit_label
            ; exit:
        );

        // Postamble (rebox / cleanup, executed at every exit)
        jit::log(|| format!("🔧 RV32 JIT: postamble ({} ops)", trace.postamble.len()));
        self.compile_ops(&trace.postamble, &mut guard_index, &mut guards)?;

        self.exit_stack.pop();
        self.fail_stack.pop();

        // Success return value: 0
        dynasm!(self.ops ; .arch riscv32i ; li a0, 0);

        // ── Common epilogue: restore saved registers and return ───────────────
        dynasm!(self.ops
            ; .arch riscv32i
            ; lw s4, [sp, s4_off]
            ; lw s3, [sp, s3_off]
            ; lw s2, [sp, s2_off]
            ; lw s0, [sp, s0_off]
            ; lw ra,  [sp, ra_off]
            ; addi sp, sp, frame_size
            ; ret

        // ── Fail path (-1) ───────────────────────────────────────────────────
            ; => fail_label
            ; fail:
            ; li a0, -1

        // ── Unwind chain ─────────────────────────────────────────────────────
        // s4 is a linked list of unwind records: { i32 frame_size, *mut Value s2, *unwind next }
        // (12 bytes per record on rv32; mirrors the x86_64 { i64, ptr, ptr } layout but 4-byte)
            ; => unwind_label
            ; unwind_loop:
            ; beqz s4, => fail_return_label // no more frames to unwind
            ; lw t0, [s4, 0]                // t0 = saved frame_size
            ; add sp, sp, t0                // pop the nested JIT frame
            ; lw s2, [s4, 4]               // restore register-array base
            ; lw t1, [s4, 8]               // t1 = next chain ptr
            ; addi sp, sp, 12              // pop the 12-byte unwind record
            ; mv s4, t1
            ; j => unwind_label

            ; => fail_return_label
            ; fail_return:
            ; j => exit_label               // fall through to epilogue (postamble + restore)
        );

        // ── Finalise ─────────────────────────────────────────────────────────
        let ops = mem::replace(&mut self.ops, Assembler::new().unwrap());
        let exec_buf = ops.finalize().unwrap();
        let entry_point = exec_buf.ptr(dynasmrt::AssemblyOffset(0));
        let entry: extern "C" fn(*mut Value, *mut VM, *const Function) -> i32 =
            unsafe { mem::transmute(entry_point) };

        // Keep the buffer alive forever (same strategy as x86_64 backend).
        Box::leak(Box::new(exec_buf));

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
                TraceOp::Add { dest, lhs, rhs, lhs_type, rhs_type } => {
                    self.compile_add_specialized(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }
                TraceOp::Sub { dest, lhs, rhs, lhs_type, rhs_type } => {
                    self.compile_sub_specialized(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }
                TraceOp::Mul { dest, lhs, rhs, lhs_type, rhs_type } => {
                    self.compile_mul_specialized(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }
                TraceOp::Div { dest, lhs, rhs, lhs_type, rhs_type } => {
                    self.compile_div_specialized(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }
                TraceOp::Mod { dest, lhs, rhs, lhs_type, rhs_type } => {
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
                    let ptr = function.pointer();
                    let g = self.compile_guard_native_function(
                        *register, ptr, *guard_index as usize,
                    )?;
                    guards.push(g);
                    *guard_index += 1;
                }
                TraceOp::GuardFunction { register, function_idx } => {
                    let g = self.compile_guard_function(
                        *register, *function_idx, *guard_index as usize,
                    )?;
                    guards.push(g);
                    *guard_index += 1;
                }
                TraceOp::GuardClosure { register, function_idx, upvalues_ptr } => {
                    let g = self.compile_guard_closure(
                        *register, *function_idx, *upvalues_ptr, *guard_index as usize,
                    )?;
                    guards.push(g);
                    *guard_index += 1;
                }
                TraceOp::CallNative { dest, callee, function, first_arg, arg_count } => {
                    let ptr = function.pointer();
                    self.compile_call_native(*dest, *callee, ptr, *first_arg, *arg_count)?;
                }
                TraceOp::CallFunction {
                    dest, callee, function_idx, first_arg, arg_count,
                    is_closure, upvalues_ptr,
                } => {
                    self.compile_call_function(
                        *dest, *callee, *function_idx, *first_arg, *arg_count,
                        *is_closure, *upvalues_ptr,
                    )?;
                }
                TraceOp::InlineCall { dest, callee, trace } => {
                    self.compile_inline_call(*dest, *callee, trace, guard_index, guards)?;
                }
                TraceOp::CallMethod { dest, object, method_name, first_arg, arg_count } => {
                    match (method_name.as_str(), *arg_count) {
                        ("push", 1) => self.compile_array_push(*object, *first_arg)?,
                        ("is_some", 0) => self.compile_enum_is_some(*dest, *object)?,
                        ("unwrap", 0) => self.compile_enum_unwrap(*dest, *object)?,
                        _ => self.compile_call_method(
                            *dest, *object, method_name, *first_arg, *arg_count,
                        )?,
                    }
                }
                TraceOp::GetField { dest, object, field_name, field_index, value_type, is_weak } => {
                    self.compile_get_field(
                        *dest, *object, field_name, *field_index, *value_type, *is_weak,
                    )?;
                }
                TraceOp::SetField { object, field_name, value, field_index, value_type, is_weak } => {
                    self.compile_set_field(
                        *object, field_name, *value, *field_index, *value_type, *is_weak,
                    )?;
                }
                TraceOp::NewArray { dest, first_element, count } => {
                    self.compile_new_array(*dest, *first_element, *count)?;
                }
                TraceOp::NewStruct { dest, struct_name, field_names, field_registers } => {
                    self.compile_new_struct(*dest, struct_name, field_names, field_registers)?;
                }
                TraceOp::NewEnumUnit { dest, enum_name, variant_name } => {
                    self.compile_new_enum_unit(*dest, enum_name, variant_name)?;
                }
                TraceOp::NewEnumVariant { dest, enum_name, variant_name, value_registers } => {
                    self.compile_new_enum_variant(
                        *dest, enum_name, variant_name, value_registers,
                    )?;
                }
                TraceOp::IsEnumVariant { dest, value, enum_name, variant_name } => {
                    self.compile_is_enum_variant(*dest, *value, enum_name, variant_name)?;
                }
                TraceOp::GetEnumValue { dest, enum_reg, index } => {
                    self.compile_get_enum_value(*dest, *enum_reg, *index)?;
                }
                TraceOp::Guard { register, expected_type } => {
                    let g = self.compile_guard(
                        *register, *expected_type, *guard_index as usize,
                    )?;
                    guards.push(g);
                    *guard_index += 1;
                }
                TraceOp::GuardLoopContinue { condition_register, expect_truthy, bailout_ip } => {
                    let g = self.compile_truth_guard(
                        *condition_register, *expect_truthy, *bailout_ip,
                        *guard_index as usize,
                    )?;
                    guards.push(g);
                    *guard_index += 1;
                }
                TraceOp::Unbox { specialized_id, source_reg, layout } => {
                    self.compile_unbox(*specialized_id, *source_reg, layout)?;
                }
                TraceOp::Rebox { dest_reg, specialized_id, layout } => {
                    self.compile_rebox(*dest_reg, *specialized_id, layout)?;
                }
                TraceOp::DropSpecialized { specialized_id, layout } => {
                    self.compile_drop_specialized(*specialized_id, layout)?;
                }
                TraceOp::SpecializedOp { op, operands } => {
                    self.compile_specialized_op(op, operands)?;
                }
                TraceOp::NestedLoopCall { function_idx, loop_start_ip, bailout_ip } => {
                    // Nested loop: exit to interpreter so it can be compiled later.
                    let exit_label = self.current_exit_label();
                    let current_guard_index = *guard_index;
                    guards.push(Guard {
                        index: current_guard_index as usize,
                        bailout_ip: *bailout_ip,
                        kind: GuardKind::NestedLoop {
                            function_idx: *function_idx,
                            loop_start_ip: *loop_start_ip,
                        },
                        fail_count: 0,
                        side_trace: None,
                    });
                    dynasm!(self.ops
                        ; .arch riscv32i
                        ; li a0, current_guard_index + 1
                        ; j => exit_label
                    );
                    *guard_index += 1;
                }
                TraceOp::Return { .. } => {
                    // Return ops are handled by the epilogue; no codegen needed here.
                }
            }
        }
        Ok(())
    }

    /// Inline a nested JIT trace (analogous to the x86_64 version).
    fn compile_inline_call(
        &mut self,
        _dest: u8,
        _callee: u8,
        trace: &InlineTrace,
        guard_index: &mut i32,
        guards: &mut Vec<Guard>,
    ) -> Result<()> {
        // Push a new exit/fail pair for the inlined trace.
        let inner_exit = self.ops.new_dynamic_label();
        let inner_fail = self.ops.new_dynamic_label();
        self.exit_stack.push(inner_exit);
        self.fail_stack.push(inner_fail);
        self.inline_depth += 1;

        self.compile_ops(&trace.body, guard_index, guards)?;

        self.inline_depth -= 1;
        self.exit_stack.pop();
        self.fail_stack.pop();

        // Emit the inner exit/fail labels as no-ops (fall through).
        dynasm!(self.ops
            ; .arch riscv32i
            ; => inner_exit
            ; => inner_fail
        );
        Ok(())
    }
}

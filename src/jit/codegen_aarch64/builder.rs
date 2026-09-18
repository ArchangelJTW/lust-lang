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
            last_fail_island: 0,
            specialization_registry: SpecializationRegistry::new(),
            specialized_values: HashMap::new(),
            scalar_registers: HashMap::new(),
            pins: HashMap::new(),
            pin_active: false,
            dirty_pins: Vec::new(),
            trace_start_ip: 0,
            current_fail_ip: None,
            fail_sites: Vec::new(),
            next_specialized_id: 0,
        }
    }

    /// Where the interpreter resumes when a guard that is not a branch
    /// (type, function identity, struct layout) fails: the instruction the
    /// guard was recorded for, so it is re-executed with the value the guard
    /// rejected. Inside an inlined body the ip is the call's, since the
    /// callee frame is torn down on exit; the call then runs in the
    /// interpreter.
    pub(super) fn guard_bailout_ip(&self) -> usize {
        self.current_fail_ip.unwrap_or(self.trace_start_ip)
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
        self.trace_start_ip = trace.start_ip;
        self.last_fail_island = self.ops.offset().0;
        self.pins = pins::plan(&hoisted_constants, &trace.preamble, &trace.ops);
        crate::jit::log(|| {
            let mut text = format!("📋 JIT(aarch64): trace {:?} preamble:\n", trace_id);
            for op in &trace.preamble {
                text.push_str(&format!("    {op:?}\n"));
            }
            text.push_str("  body:\n");
            for op in &trace.ops {
                text.push_str(&format!("    {op:?}\n"));
            }
            text.push_str(&format!("  pins: {:?}", self.pins));
            text
        });
        self.pin_active = false;
        self.dirty_pins.clear();
        self.current_fail_ip = None;
        self.fail_sites.clear();
        let stack_size = Self::compute_stack_size(trace);
        let mut guards = Vec::new();
        let mut guard_index = 0i32;
        // Exits from the preamble and the loop prologue: nothing pinned yet.
        let exit_label = self.ops.new_dynamic_label();
        // Exits from the loop body: write pinned registers back first.
        let exit_pinned_label = self.ops.new_dynamic_label();
        let after_unwind_label = self.ops.new_dynamic_label();
        let epilogue_label = self.ops.new_dynamic_label();
        let fail_label = self.ops.new_dynamic_label();
        self.exit_stack.push(exit_label);
        self.fail_stack.push(fail_label);
        crate::jit::log(|| {
            format!(
                "🔧 JIT(aarch64): Emitting prologue with sub sp, {} ({} pinned registers)",
                stack_size,
                self.pins.len()
            )
        });

        // Entry: x0 = *mut Value (registers), x1 = *mut VM, x2 = *const Function
        dynasm!(self.ops
            ; .arch aarch64
            ; stp x29, x30, [sp, -16]!
            ; mov x29, sp
            ; stp x19, x20, [sp, -16]!
            ; stp x21, x22, [sp, -16]!
            ; stp x23, x24, [sp, -16]!
            ; stp x25, x26, [sp, -16]!
            ; stp x27, x28, [sp, -16]!
            ; stp d8, d9, [sp, -16]!
            ; stp d10, d11, [sp, -16]!
            ; stp d12, d13, [sp, -16]!
            ; stp d14, d15, [sp, -16]!
        );
        self.emit_sub_sp(stack_size);
        dynasm!(self.ops
            ; .arch aarch64
            ; mov x19, x0
            ; mov x20, x1
            ; mov x21, xzr
        );
        for slot in 0..Self::count_specialized_slots(trace) as i32 {
            let offset = SPECIALIZED_BASE_OFFSET - slot * SPECIALIZED_SLOT_SIZE;
            self.emit_slot_addr(11, offset);
            dynasm!(self.ops
                ; .arch aarch64
                ; str xzr, [x11]
                ; str xzr, [x11, 8]
                ; str xzr, [x11, 16]
                ; str xzr, [x11, 24]
            );
        }
        for (dest, value) in &hoisted_constants {
            self.compile_load_const(*dest, value)?;
        }

        // Compile preamble (executed once at trace entry)
        jit::log(|| format!("🔧 JIT: Compiling preamble ({} ops)", trace.preamble.len()));
        self.compile_ops(&trace.preamble, &mut guard_index, &mut guards)?;

        // `>fail` references so far (preamble) bind here: nothing to write back.
        dynasm!(self.ops
            ; .arch aarch64
            ; b >preamble_fail_skip
            ; fail:
            ; b => fail_label
            ; preamble_fail_skip:
        );

        // Loop prologue: load carried pins.
        let mut pinned: Vec<(u8, pins::Pin)> = self.pins.iter().map(|(r, p)| (*r, *p)).collect();
        pinned.sort_by_key(|(r, _)| *r);
        for (vm_reg, pin) in &pinned {
            if pin.class == pins::PinClass::Carried {
                self.emit_pin_load(*vm_reg, *pin);
            }
        }
        let has_pins = !self.pins.is_empty();
        if has_pins {
            self.pin_active = true;
            // Iteration two onwards arrives with every carried pin possibly
            // newer than memory.
            self.dirty_pins = pinned
                .iter()
                .filter(|(_, p)| p.class == pins::PinClass::Carried)
                .map(|(r, _)| *r)
                .collect();
            self.exit_stack.push(exit_pinned_label);
        }

        // Create a loop_start label AFTER preamble, BEFORE loop body
        self.current_fail_ip = None;
        let loop_start_label = self.ops.new_dynamic_label();
        dynasm!(self.ops
            ; .arch aarch64
            ; => loop_start_label
            ; loop_start:
        );

        // Compile main trace body (the loop)
        let compile_result = self.compile_ops(&trace.ops, &mut guard_index, &mut guards);
        compile_result?;

        // At end of loop body, jump back to loop_start to loop
        dynasm!(self.ops
            ; .arch aarch64
            ; b => loop_start_label
        );
        if has_pins {
            self.exit_stack.pop();
        }
        self.pin_active = false;

        // `>fail` references from the body bind here.
        dynasm!(self.ops
            ; .arch aarch64
            ; fail:
            ; movn w0, 0
            ; b => exit_pinned_label
        );

        // Body exits: preserve the exit code, turn any inline frames into
        // interpreter frames so x19 is the trace's own register array again,
        // then write pinned registers back to it.
        dynasm!(self.ops
            ; .arch aarch64
            ; => exit_pinned_label
            ; mov w22, w0
        );
        self.emit_unwind_inline_frames();
        self.emit_writeback_carried_pins();
        dynasm!(self.ops
            ; .arch aarch64
            ; b => after_unwind_label
            ; => exit_label
            ; exit:
            // Postamble helpers may overwrite w0. Preserve the exit reason in
            // a callee-saved register until specialized state is materialized.
            ; mov w22, w0
        );
        self.emit_unwind_inline_frames();
        dynasm!(self.ops
            ; .arch aarch64
            ; => after_unwind_label
        );

        // Compile postamble (executed once at trace exit)
        jit::log(|| {
            format!(
                "🔧 JIT: Compiling postamble ({} ops)",
                trace.postamble.len()
            )
        });
        self.current_fail_ip = None;
        self.compile_ops(&trace.postamble, &mut guard_index, &mut guards)?;

        // Now pop the label stacks after everything is compiled
        self.exit_stack.pop();
        self.fail_stack.pop();

        // Epilogue: sp is recovered from the frame pointer, so the exit path
        // is valid regardless of how deep an inline frame we came from.
        let saved_below_fp = SAVED_BELOW_FP as u32;
        dynasm!(self.ops
            ; .arch aarch64
            ; mov w0, w22
            ; => epilogue_label
            ; sub sp, x29, #saved_below_fp
            ; ldp d14, d15, [sp], 16
            ; ldp d12, d13, [sp], 16
            ; ldp d10, d11, [sp], 16
            ; ldp d8, d9, [sp], 16
            ; ldp x27, x28, [sp], 16
            ; ldp x25, x26, [sp], 16
            ; ldp x23, x24, [sp], 16
            ; ldp x21, x22, [sp], 16
            ; ldp x19, x20, [sp], 16
            ; ldp x29, x30, [sp], 16
            ; ret
            // A failing postamble must not run the postamble again: return
            // -1 straight through the epilogue. `>fail` references from the
            // postamble bind here.
            ; fail:
            ; movn w0, 0
            ; b => epilogue_label
            // Failures before anything is pinned (preamble).
            ; => fail_label
            ; movn w0, 0
            ; b => exit_label
        );
        crate::jit::log(|| {
            format!("📏 JIT(aarch64): trace code size {} bytes", self.ops.offset().0)
        });
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
            fail_sites: mem::take(&mut self.fail_sites),
            parent,
            side_traces: Vec::new(),
            hoisted_constants,
        })
    }

    /// After an op that can branch to `>fail`: bind those branches to a stub
    /// that exits with this op's fail-site code, so the interpreter resumes
    /// at the instruction the op came from. Without a known ip the branches
    /// fall through to the trace's generic `fail:` (-1).
    fn emit_fail_stub(&mut self) {
        let Some(ip) = self.current_fail_ip else {
            return;
        };
        let code = -((self.fail_sites.len() as i32) + 2);
        self.fail_sites.push(ip);
        let exit_label = self.current_exit_label();
        dynasm!(self.ops
            ; .arch aarch64
            ; b >fail_stub_skip
            ; fail:
        );
        self.emit_mov_imm_i32(0, code);
        dynasm!(self.ops
            ; .arch aarch64
            ; b => exit_label
            ; fail_stub_skip:
        );
    }

    /// Ops that can branch to `>fail`.
    fn op_may_fail(op: &TraceOp) -> bool {
        matches!(
            op,
            TraceOp::LoadConst { .. }
                | TraceOp::Div { .. }
                | TraceOp::Mod { .. }
                | TraceOp::Concat { .. }
                | TraceOp::GetIndex { .. }
                | TraceOp::TryGetIndex { .. }
                | TraceOp::ArrayIndexOk { .. }
                | TraceOp::ArrayLen { .. }
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
                | TraceOp::TryCast { .. }
                | TraceOp::GetEnumValue { .. }
                | TraceOp::Unbox { .. }
                | TraceOp::Rebox { .. }
                | TraceOp::DropSpecialized { .. }
                | TraceOp::SpecializedOp { .. }
        ) && !matches!(
            op,
            TraceOp::LoadConst {
                value: Value::Int(_) | Value::Float(_) | Value::Bool(_),
                ..
            }
        )
    }
    /// Hand the inline-call frame chain in x21 to the interpreter (see
    /// `jit_materialize_inline_frames`): the callee frames become real
    /// frames and execution resumes inside the innermost one. Leaves x19 =
    /// the trace's own register array and x21 = 0. No-op without a chain.
    fn emit_unwind_inline_frames(&mut self) {
        unsafe extern "C" {
            fn jit_materialize_inline_frames(
                vm: *mut VM,
                record: *const u8,
                regs: *mut Value,
            ) -> *mut Value;
        }
        dynasm!(self.ops
            ; .arch aarch64
            ; cbz x21, >unwind_done
            ; mov x0, x20
            ; mov x1, x21
            ; mov x2, x19
        );
        self.emit_call(jit_materialize_inline_frames as *const ());
        dynasm!(self.ops
            ; .arch aarch64
            ; mov x19, x0
            ; mov x21, xzr
            ; unwind_done:
        );
    }

    fn compile_ops(
        &mut self,
        ops: &[TraceOp],
        guard_index: &mut i32,
        guards: &mut Vec<Guard>,
    ) -> Result<()> {
        let mut skip_through: Option<usize> = None;
        for (op_index, op) in ops.iter().enumerate() {
            if skip_through.is_some_and(|last| op_index <= last) {
                continue;
            }
            if let TraceOp::At { ip } = op {
                // Inside an inlined body the markers carry the callee's ips;
                // an exit there resumes inside the callee, whose frame the
                // exit path materializes.
                self.current_fail_ip = Some(*ip);
                continue;
            }
            if self.pin_active && pins::op_touches_register_memory(op) {
                self.flush_dirty_pins();
            }
            // A type guard on a pin whose type is proven at entry is
            // redundant: only typed writes reach it. Other pins keep their
            // guard, which is what proves their type.
            if let TraceOp::Guard {
                register,
                expected_type,
            } = op
                && self
                    .active_pin(*register)
                    .is_some_and(|pin| pin.proven_at_entry && pin.ty == *expected_type)
            {
                self.update_scalar_registers(op);
                continue;
            }
            // Fusion looks at the next real op; `At` markers are transparent.
            let next_index = (op_index + 1..ops.len())
                .find(|&j| !matches!(ops[j], TraceOp::At { .. }));
            if let Some(next_index) = next_index {
                let next = &ops[next_index];
                if let TraceOp::GuardLoopContinue {
                    condition_register, ..
                } = next
                    && self.scalar_registers.contains_key(condition_register)
                    && Self::register_overwritten_before_read(
                        &ops[next_index + 1..],
                        *condition_register,
                    )
                    && let Some(guard) =
                        self.compile_numeric_comparison_guard(op, next, *guard_index as usize)?
                {
                    guards.push(guard);
                    *guard_index += 1;
                    skip_through = Some(next_index);
                    continue;
                }
                if let TraceOp::LoadConst {
                    dest: constant_register,
                    ..
                } = op
                    && self.scalar_registers.contains_key(constant_register)
                    && Self::register_overwritten_before_read(
                        &ops[next_index + 1..],
                        *constant_register,
                    )
                    && (self.compile_integer_add_immediate(op, next)?
                        || self.compile_float_op_immediate(op, next)?)
                {
                    self.update_scalar_registers(next);
                    skip_through = Some(next_index);
                    continue;
                }
            }
            match op {
                TraceOp::At { .. } => unreachable!("markers are consumed above"),
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

                TraceOp::GuardStructLayout { register, layout } => {
                    let guard = self.compile_guard_struct_layout(
                        *register,
                        *layout,
                        *guard_index as usize,
                    )?;
                    guards.push(guard);
                    *guard_index += 1;
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
                    // The callee body addresses its own frame: pins are
                    // suspended, and memory must be current on entry.
                    let outer_scalar_registers = mem::take(&mut self.scalar_registers);
                    let outer_pin_active = self.pin_active;
                    self.pin_active = false;
                    let result = self.compile_inline_call(*dest, *callee, trace, guard_index, guards);
                    self.pin_active = outer_pin_active;
                    self.scalar_registers = outer_scalar_registers;
                    result?;
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
                    resume_ip,
                } => {
                    let guard = self.compile_nested_loop_call(
                        *function_idx,
                        *loop_start_ip,
                        *bailout_ip,
                        *resume_ip,
                        *guard_index as usize,
                    );
                    guards.push(guard);
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
            if Self::op_may_fail(op) {
                self.emit_fail_stub();
            }
            self.maybe_emit_fail_island();
        }

        Ok(())
    }

    /// Plant a `fail:` island if the code since the last one is getting
    /// close to the reach of a conditional branch (see FAIL_ISLAND_INTERVAL).
    /// Only ever called between ops, so no op's own local labels are split.
    fn maybe_emit_fail_island(&mut self) {
        let here = self.ops.offset().0;
        if here - self.last_fail_island < FAIL_ISLAND_INTERVAL {
            return;
        }
        dynasm!(self.ops
            ; .arch aarch64
            ; b >fail_island_skip
            ; fail:
            ; b >fail
            ; fail_island_skip:
        );
        self.last_fail_island = self.ops.offset().0;
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

        let a = self.operand_x(source, 0);
        let d = self.direct_dest_x(*dest).unwrap_or(0);
        if (0..=4095).contains(immediate) {
            let imm = *immediate as u32;
            dynasm!(self.ops ; .arch aarch64 ; add XSP(d), XSP(a), #imm);
        } else if (-4095..0).contains(immediate) {
            let imm = (-*immediate) as u32;
            dynasm!(self.ops ; .arch aarch64 ; sub XSP(d), XSP(a), #imm);
        } else {
            self.emit_mov_imm64(10, *immediate as u64);
            dynasm!(self.ops ; .arch aarch64 ; add X(d), X(a), x10);
        }
        if d == 0 {
            self.store_from_x0(*dest, ValueTag::Int.as_u8());
        }
        Ok(true)
    }

    fn compile_numeric_comparison_guard(
        &mut self,
        comparison: &TraceOp,
        guard: &TraceOp,
        guard_index: usize,
    ) -> Result<Option<Guard>> {
        let (condition_register, lhs, rhs, lhs_type, rhs_type, comparison_kind) = match comparison {
            TraceOp::Lt {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => (*dest, *lhs, *rhs, *lhs_type, *rhs_type, 0),
            TraceOp::Le {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => (*dest, *lhs, *rhs, *lhs_type, *rhs_type, 1),
            TraceOp::Gt {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => (*dest, *lhs, *rhs, *lhs_type, *rhs_type, 2),
            TraceOp::Ge {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => (*dest, *lhs, *rhs, *lhs_type, *rhs_type, 3),
            _ => return Ok(None),
        };
        if !matches!(lhs_type, ValueType::Int | ValueType::Float)
            || !matches!(rhs_type, ValueType::Int | ValueType::Float)
        {
            return Ok(None);
        }
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

        let guard_ok = self.ops.new_dynamic_label();
        let cmp = self.compare_operands(lhs, rhs, lhs_type, rhs_type);
        self.emit_compare(cmp);
        if matches!(cmp, super::comparisons::Compare::Float(..)) {
            // Every ordered comparison is false for NaN. AArch64 condition
            // codes already treat the unordered result correctly, and the
            // complement of each ordered condition is true for NaN, so the
            // failing direction needs no separate unordered check.
            match (comparison_kind, *expect_truthy) {
                (0, true) => dynasm!(self.ops ; .arch aarch64 ; b.mi =>guard_ok),
                (1, true) => dynasm!(self.ops ; .arch aarch64 ; b.ls =>guard_ok),
                (2, true) => dynasm!(self.ops ; .arch aarch64 ; b.gt =>guard_ok),
                (3, true) => dynasm!(self.ops ; .arch aarch64 ; b.ge =>guard_ok),
                (0, false) => dynasm!(self.ops ; .arch aarch64 ; b.pl =>guard_ok),
                (1, false) => dynasm!(self.ops ; .arch aarch64 ; b.hi =>guard_ok),
                (2, false) => dynasm!(self.ops ; .arch aarch64 ; b.le =>guard_ok),
                (3, false) => dynasm!(self.ops ; .arch aarch64 ; b.lt =>guard_ok),
                _ => unreachable!(),
            }
        } else {
            match (comparison_kind, *expect_truthy) {
                (0, true) | (3, false) => dynasm!(self.ops ; .arch aarch64 ; b.lt =>guard_ok),
                (1, true) | (2, false) => dynasm!(self.ops ; .arch aarch64 ; b.le =>guard_ok),
                (2, true) | (1, false) => dynasm!(self.ops ; .arch aarch64 ; b.gt =>guard_ok),
                (3, true) | (0, false) => dynasm!(self.ops ; .arch aarch64 ; b.ge =>guard_ok),
                _ => unreachable!(),
            }
        }

        // The interpreter resumes at the branch bytecode, which reads this
        // register. Materialize only the uncommon failed result.
        let failed_value = u32::from(!*expect_truthy);
        dynasm!(self.ops ; .arch aarch64 ; movz x0, #failed_value);
        self.store_from_x0(condition_register, ValueTag::Bool.as_u8());
        let guard_return_value = (guard_index + 1) as i32;
        self.emit_guard_exit(guard_return_value);
        dynasm!(self.ops ; .arch aarch64 ; =>guard_ok);

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
                | TraceOp::GuardStructLayout { .. }
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
            TraceOp::At { .. } | TraceOp::LoadConst { .. } => false,
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
            | TraceOp::GuardStructLayout { register: source, .. }
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
            TraceOp::NestedLoopCall { .. } => true,
            TraceOp::NewEnumUnit { .. } | TraceOp::Rebox { .. } | TraceOp::DropSpecialized { .. } => {
                false
            }
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
        if matches!(op, TraceOp::At { .. }) {
            return;
        }
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
            TraceOp::At { .. } => {}
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
            | TraceOp::GuardStructLayout { .. }
            | TraceOp::GuardFunction { .. }
            | TraceOp::GuardClosure { .. }
            | TraceOp::GuardLoopContinue { .. }
            | TraceOp::Return { .. }
            | TraceOp::Unbox { .. }
            | TraceOp::DropSpecialized { .. } => {}
            // The inner loop may have written any register.
            TraceOp::NestedLoopCall { .. } => self.scalar_registers.clear(),
        }
    }

    fn compute_stack_size(trace: &Trace) -> i32 {
        let specialized_slots = Self::count_specialized_slots(trace) as i32;
        let specialized_bytes =
            SPECIALIZED_STACK_BASE + (specialized_slots * SPECIALIZED_SLOT_SIZE);
        let size = MIN_JIT_STACK_SIZE.max(specialized_bytes);
        let size = (size + 15) & !15;
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
            let frame_value_count = trace.register_count as i32;
            // Round the frame up so sp stays 16-byte aligned regardless of
            // the Value size.
            let frame_size = (frame_value_count * value_size + 15) & !15;
            let metadata_size = INLINE_METADATA_SIZE as u32;
            let inline_end = self.ops.new_dynamic_label();
            unsafe extern "C" {
                fn jit_move_safe(src_ptr: *const Value, dest_ptr: *mut Value) -> u8;
                fn jit_drop_values(values: *mut Value, len: usize);
            }

            // Push the inline record (see `JitInlineRecord`). It is linked
            // into the unwind chain only once the frame below it is fully
            // built, so a failure while building it exits cleanly.
            let caller_resume_ip = self.current_fail_ip.map_or(0, |ip| ip + 1);
            dynasm!(self.ops
                ; .arch aarch64
                ; sub sp, sp, #metadata_size
            );
            self.emit_mov_imm_i32(0, frame_value_count);
            dynasm!(self.ops
                ; .arch aarch64
                ; str x0, [sp]
                ; str x19, [sp, 8]
                ; str x21, [sp, 16]
                ; str xzr, [sp, 24]
            );
            self.emit_mov_imm64(0, trace.function_idx as u64);
            dynasm!(self.ops ; .arch aarch64 ; str x0, [sp, 32]);
            self.emit_mov_imm64(0, dest as u64);
            dynasm!(self.ops ; .arch aarch64 ; str x0, [sp, 40]);
            self.emit_mov_imm64(0, callee as u64);
            dynasm!(self.ops ; .arch aarch64 ; str x0, [sp, 48]);
            self.emit_mov_imm64(0, caller_resume_ip as u64);
            dynasm!(self.ops ; .arch aarch64 ; str x0, [sp, 56]);
            // Allocate space for callee registers and initialise every one
            // to Nil (discriminant 0) — what `jit_init_nil` does, without a
            // call per register — so the argument moves below find nothing to
            // drop. x19 still addresses the caller's registers here.
            self.emit_sub_sp(frame_size);
            let nil_tag = ValueTag::Nil.as_u8() as u32;
            dynasm!(self.ops ; .arch aarch64 ; movz w13, #nil_tag);
            for reg in 0..trace.register_count {
                dynasm!(self.ops ; .arch aarch64 ; mov x11, sp);
                self.emit_add_imm(11, 11, super::registers::reg_offset(reg));
                dynasm!(self.ops ; .arch aarch64 ; strb w13, [x11]);
            }

            // Copy positional arguments into the callee frame.
            for (arg_index, src_reg) in trace.arg_registers.iter().enumerate() {
                let dest_offset = (arg_index as i32) * value_size;
                self.emit_reg_addr(0, *src_reg);
                dynasm!(self.ops ; .arch aarch64 ; mov x11, sp);
                self.emit_add_imm(1, 11, dest_offset);
                self.emit_call(jit_move_safe as *const ());
                self.emit_fail_if_w0_zero();
            }
            // A failed argument move resumes at the call, in the caller.
            self.emit_fail_stub();

            dynasm!(self.ops
                ; .arch aarch64
                ; mov x19, sp
            );
            // Link the record into the unwind chain.
            dynasm!(self.ops ; .arch aarch64 ; mov x11, sp);
            self.emit_add_imm(21, 11, frame_size);

            let call_ip = self.current_fail_ip;
            let inline_result = self.compile_ops(&trace.body, guard_index, guards);
            inline_result?;
            // A failed result move exits from the callee frame at its last
            // instruction; everything after the frame is popped is the
            // caller's again.
            let callee_last_ip = self.current_fail_ip;
            self.current_fail_ip = call_ip;

            if let Some(ret_reg) = trace.return_register {
                self.current_fail_ip = callee_last_ip;
                let dest_offset = (dest as i32) * value_size;
                dynasm!(self.ops
                    ; .arch aarch64
                    ; ldr x11, [x21, 8]
                );
                self.emit_reg_addr(0, ret_reg);
                self.emit_add_imm(1, 11, dest_offset);
                self.emit_call(jit_move_safe as *const ());
                self.emit_fail_if_w0_zero();
                self.emit_fail_stub();
                self.current_fail_ip = call_ip;
            }

            // Drop callee registers and pop the frame + metadata.
            dynasm!(self.ops
                ; .arch aarch64
                ; mov x0, x19
            );
            self.emit_mov_imm_i32(1, frame_value_count);
            self.emit_call(jit_drop_values as *const ());
            dynasm!(self.ops
                ; .arch aarch64
                ; add sp, x21, #metadata_size
                ; ldr x19, [x21, 8]
                ; ldr x21, [x21, 16]
            );

            if trace.return_register.is_none() {
                self.compile_load_const(dest, &Value::Nil)?;
            }
            dynasm!(self.ops
                ; .arch aarch64
                ; => inline_end
            );

            Ok(())
        })();
        self.inline_depth -= 1;
        result
    }
}

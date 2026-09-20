use super::*;
use crate::VM;
impl JitCompiler {
    pub(super) fn compile_guard(
        &mut self,
        register: u8,
        expected_type: ValueType,
        guard_index: usize,
    ) -> Result<Guard> {
        let expected_tag = match expected_type {
            ValueType::Bool => ValueTag::Bool,
            ValueType::Int => ValueTag::Int,
            ValueType::Float => ValueTag::Float,
            ValueType::String => ValueTag::String,
            ValueType::Array => ValueTag::Array,
            ValueType::Tuple => ValueTag::Tuple,
            ValueType::Struct => ValueTag::Struct,
            ValueType::Plain => {
                return Err(crate::LustError::RuntimeError {
                    message: "a guard cannot expect Plain".into(),
                });
            }
        };
        let expected_discriminant = expected_tag.as_u8() as u32;
        let guard_return_value = (guard_index + 1) as i32;
        // A guard is the proof of a register's type; it must look at memory
        // even when the register is pinned.
        self.load_tag_from_memory(0, register);
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp w0, #expected_discriminant
            ; b.eq >guard_ok
        );
        self.emit_guard_exit(guard_return_value);
        dynasm!(self.ops
            ; .arch aarch64
            ; guard_ok:
        );
        Ok(Guard {
            index: guard_index,
            bailout_ip: self.guard_bailout_ip(),
            kind: match expected_type {
                ValueType::Int => GuardKind::IntType { register },
                ValueType::Float => GuardKind::FloatType { register },
                ValueType::Bool => GuardKind::BoolType { register },
                ValueType::String => GuardKind::IntType { register },
                ValueType::Array => GuardKind::IntType { register },
                ValueType::Tuple => GuardKind::IntType { register },
                ValueType::Struct => GuardKind::IntType { register },
                ValueType::Plain => unreachable!("rejected above"),
            },
            fail_count: 0,
        })
    }

    pub(super) fn compile_guard_function(
        &mut self,
        register: u8,
        function_idx: usize,
        guard_index: usize,
    ) -> Result<Guard> {
        self.compile_guard_function_internal(
            register,
            function_idx,
            core::ptr::null(),
            false,
            guard_index,
        )
    }

    pub(super) fn compile_guard_closure(
        &mut self,
        register: u8,
        function_idx: usize,
        upvalues_ptr: *const (),
        guard_index: usize,
    ) -> Result<Guard> {
        self.compile_guard_function_internal(
            register,
            function_idx,
            upvalues_ptr,
            true,
            guard_index,
        )
    }

    fn compile_guard_function_internal(
        &mut self,
        register: u8,
        function_idx: usize,
        upvalues_ptr: *const (),
        is_closure: bool,
        guard_index: usize,
    ) -> Result<Guard> {
        let guard_return_value = (guard_index + 1) as i32;
        unsafe extern "C" {
            fn jit_guard_function_identity(
                value_ptr: *const Value,
                expected_kind: u8,
                expected_function_idx: usize,
                expected_upvalues: *const (),
                register_index: u8,
            ) -> u8;
        }
        let kind_flag: u32 = if is_closure { 1 } else { 0 };
        self.emit_reg_addr(0, register);
        self.emit_mov_imm32(1, kind_flag);
        self.emit_mov_imm64(2, function_idx as u64);
        self.emit_mov_imm64(3, upvalues_ptr as usize as u64);
        self.emit_mov_imm32(4, register as u32);
        self.emit_call(jit_guard_function_identity as *const ());
        dynasm!(self.ops
            ; .arch aarch64
            ; and w0, w0, 0xff
            ; cbnz w0, >guard_ok
        );
        self.emit_guard_exit(guard_return_value);
        dynasm!(self.ops
            ; .arch aarch64
            ; guard_ok:
        );
        let kind = if is_closure {
            GuardKind::Closure {
                register,
                function_idx,
                upvalues_ptr,
            }
        } else {
            GuardKind::Function {
                register,
                function_idx,
            }
        };
        Ok(Guard {
            index: guard_index,
            bailout_ip: self.guard_bailout_ip(),
            kind,
            fail_count: 0,
        })
    }

    pub(super) fn compile_guard_native_function(
        &mut self,
        register: u8,
        expected_ptr: *const (),
        expected_inner: usize,
        guard_index: usize,
    ) -> Result<Guard> {
        let guard_return_value = (guard_index + 1) as i32;
        unsafe extern "C" {
            fn jit_guard_native_function(
                value_ptr: *const Value,
                expected: *const (),
                register_index: u8,
            ) -> u8;
        }
        if let Some(layout) = jit::layout::ownership_layout() {
            // Inline: the native-function tag, then the allocation pointer.
            let tag = layout.native_tag as u32;
            let offset = layout.single_rc_offset as u32;
            self.load_tag(0, register);
            self.emit_reg_addr(11, register);
            dynasm!(self.ops
                ; .arch aarch64
                ; cmp w0, #tag
                ; b.ne >guard_fail
                ; ldr x9, [x11, #offset]
            );
            self.emit_mov_imm64(10, expected_inner as u64);
            dynasm!(self.ops
                ; .arch aarch64
                ; cmp x9, x10
                ; b.eq >guard_ok
                ; guard_fail:
            );
            self.emit_guard_exit(guard_return_value);
            dynasm!(self.ops
                ; .arch aarch64
                ; guard_ok:
            );
            return Ok(Guard {
                index: guard_index,
                bailout_ip: self.guard_bailout_ip(),
                kind: GuardKind::NativeFunction {
                    register,
                    expected: expected_ptr,
                },
                fail_count: 0,
            });
        }
        self.emit_reg_addr(0, register);
        self.emit_mov_imm64(1, expected_ptr as usize as u64);
        self.emit_mov_imm32(2, register as u32);
        self.emit_call(jit_guard_native_function as *const ());
        dynasm!(self.ops
            ; .arch aarch64
            ; and w0, w0, 0xff
            ; cbnz w0, >guard_ok
        );
        self.emit_guard_exit(guard_return_value);
        dynasm!(self.ops
            ; .arch aarch64
            ; guard_ok:
        );
        Ok(Guard {
            index: guard_index,
            bailout_ip: self.guard_bailout_ip(),
            kind: GuardKind::NativeFunction {
                register,
                expected: expected_ptr,
            },
            fail_count: 0,
        })
    }

    /// `VM::globals_version == version`, else exit (the trace's global
    /// snapshots are stale, so the exit evicts it).
    pub(super) fn compile_guard_globals(&mut self, version: u64, guard_index: usize) -> Guard {
        let guard_return_value = (guard_index + 1) as i32;
        let offset = core::mem::offset_of!(crate::vm::VM, globals_version) as u64;
        self.emit_mov_imm64(10, offset);
        self.emit_mov_imm64(11, version);
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x9, [x20, x10]
            ; cmp x9, x11
            ; b.eq >guard_ok
        );
        self.emit_guard_exit(guard_return_value);
        dynasm!(self.ops
            ; .arch aarch64
            ; guard_ok:
        );
        Guard {
            index: guard_index,
            bailout_ip: self.guard_bailout_ip(),
            kind: GuardKind::Globals { version },
            fail_count: 0,
        }
    }

    pub(super) fn compile_guard_struct_layout(
        &mut self,
        register: u8,
        layout: *const (),
        guard_index: usize,
    ) -> Result<Guard> {
        let guard_return_value = (guard_index + 1) as i32;
        unsafe extern "C" {
            fn jit_guard_struct_layout(value_ptr: *const Value, expected: *const ()) -> u8;
        }
        if let Some(measured) = jit::layout::rc_vec_layout() {
            // Inline: the struct tag, then the layout's allocation pointer
            // (`expected` is the `Rc`'s data pointer, 16 bytes in).
            let struct_tag = ValueTag::Struct.as_u8() as u32;
            let object_offset = measured.struct_fields_offset as u32;
            let layout_offset = measured.struct_layout_offset as u32;
            let inner = (layout as usize).wrapping_sub(16) as u64;
            self.load_tag(0, register);
            self.emit_reg_addr(11, register);
            dynasm!(self.ops
                ; .arch aarch64
                ; cmp w0, #struct_tag
                ; b.ne >guard_fail
                ; ldr x9, [x11, #object_offset]
                ; ldr x9, [x9, #layout_offset]
            );
            self.emit_mov_imm64(10, inner);
            dynasm!(self.ops
                ; .arch aarch64
                ; cmp x9, x10
                ; b.eq >guard_ok
                ; guard_fail:
            );
            self.emit_guard_exit(guard_return_value);
            dynasm!(self.ops
                ; .arch aarch64
                ; guard_ok:
            );
            return Ok(Guard {
                index: guard_index,
                bailout_ip: self.guard_bailout_ip(),
                kind: GuardKind::StructLayout { register, layout },
                fail_count: 0,
            });
        }
        self.emit_reg_addr(0, register);
        self.emit_mov_imm64(1, layout as usize as u64);
        self.emit_call(jit_guard_struct_layout as *const ());
        dynasm!(self.ops
            ; .arch aarch64
            ; and w0, w0, 0xff
            ; cbnz w0, >guard_ok
        );
        self.emit_guard_exit(guard_return_value);
        dynasm!(self.ops
            ; .arch aarch64
            ; guard_ok:
        );
        Ok(Guard {
            index: guard_index,
            bailout_ip: self.guard_bailout_ip(),
            kind: GuardKind::StructLayout { register, layout },
            fail_count: 0,
        })
    }

    /// Run a nested loop through its own root trace (see
    /// `jit_run_nested_loop`) and carry on at `resume_ip`; when the helper
    /// says the interpreter has to take over, exit through this guard.
    /// Inside an inlined body the guard exit is taken unconditionally: the
    /// inner loop then runs from the materialized frame.
    pub(super) fn compile_nested_loop_call(
        &mut self,
        function_idx: usize,
        loop_start_ip: usize,
        bailout_ip: usize,
        resume_ip: usize,
        guard_index: usize,
    ) -> Guard {
        let guard_return_value = (guard_index + 1) as i32;
        unsafe extern "C" {
            fn jit_run_nested_loop(
                vm: *mut VM,
                registers: *mut Value,
                function_idx: usize,
                loop_start_ip: usize,
                resume_ip: usize,
            ) -> i32;
        }
        if self.inline_depth == 0 {
            dynasm!(self.ops
                ; .arch aarch64
                ; mov x0, x20
                ; mov x1, x19
            );
            self.emit_mov_imm64(2, function_idx as u64);
            self.emit_mov_imm64(3, loop_start_ip as u64);
            self.emit_mov_imm64(4, resume_ip as u64);
            self.emit_call(jit_run_nested_loop as *const ());
            dynasm!(self.ops
                ; .arch aarch64
                ; cbz w0, >loop_done
            );
        }
        self.emit_guard_exit(guard_return_value);
        dynasm!(self.ops
            ; .arch aarch64
            ; loop_done:
        );
        Guard {
            index: guard_index,
            bailout_ip,
            kind: GuardKind::NestedLoop {
                function_idx,
                loop_start_ip,
            },
            fail_count: 0,
        }
    }

    pub(super) fn compile_truth_guard(
        &mut self,
        condition_register: u8,
        expect_truthy: bool,
        bailout_ip: usize,
        guard_index: usize,
    ) -> Result<Guard> {
        let guard_return_value = (guard_index + 1) as i32;
        let bool_tag = ValueTag::Bool.as_u8() as u32;
        let scalar_max_tag = ValueTag::Float.as_u8() as u32;
        unsafe extern "C" {
            fn jit_value_is_truthy(value_ptr: *const Value) -> u8;
        }
        if self.scalar_registers.get(&condition_register) == Some(&ValueType::Bool) {
            self.load_bool_payload(0, condition_register);
        } else {
            self.load_tag(0, condition_register);
            dynasm!(self.ops
                ; .arch aarch64
                ; cmp w0, #scalar_max_tag
                ; b.hi >generic_truthiness
                ; cmp w0, #bool_tag
                ; b.eq >load_bool
                // Nil is false; numeric scalars are true regardless of payload.
                ; cmp w0, 0
                ; cset w0, ne
                ; b >truthiness_ready
                ; load_bool:
            );
            self.load_bool_payload(0, condition_register);
            dynasm!(self.ops
                ; .arch aarch64
                ; b >truthiness_ready
                ; generic_truthiness:
            );
            self.emit_reg_addr(0, condition_register);
            self.emit_call(jit_value_is_truthy as *const ());
            dynasm!(self.ops
                ; .arch aarch64
                ; and w0, w0, 0xff
                ; truthiness_ready:
            );
        }
        if expect_truthy {
            dynasm!(self.ops ; .arch aarch64 ; cbnz w0, >guard_ok);
        } else {
            dynasm!(self.ops ; .arch aarch64 ; cbz w0, >guard_ok);
        }
        self.emit_guard_exit(guard_return_value);
        dynasm!(self.ops
            ; .arch aarch64
            ; guard_ok:
        );
        let kind = if expect_truthy {
            GuardKind::Truthy {
                register: condition_register,
            }
        } else {
            GuardKind::Falsy {
                register: condition_register,
            }
        };
        Ok(Guard {
            index: guard_index,
            bailout_ip,
            kind,
            fail_count: 0,
        })
    }
}

// ── Function code: branches, native calls, returns ─────────────────────
impl JitCompiler {
    /// Branch to the function label when the register's truthiness equals
    /// `expect_truthy`.
    pub(super) fn compile_branch_if(&mut self, register: u8, expect_truthy: bool, label: usize) {
        let label = self.function_label(label);
        if self.scalar_registers.get(&register) == Some(&ValueType::Bool) {
            // Only the low byte of a Bool's payload is defined.
            self.load_bool_payload(0, register);
        } else {
            unsafe extern "C" {
                fn jit_value_is_truthy(value_ptr: *const Value) -> u8;
            }
            self.emit_reg_addr(0, register);
            self.emit_call(jit_value_is_truthy as *const ());
            dynasm!(self.ops ; .arch aarch64 ; and w0, w0, 0xff);
        }
        if expect_truthy {
            dynasm!(self.ops ; .arch aarch64 ; cbnz w0, => label);
        } else {
            dynasm!(self.ops ; .arch aarch64 ; cbz w0, => label);
        }
    }

    /// Call another bytecode function's compiled code natively: push an
    /// inline record and a frame for it (exactly as an inlined call does,
    /// so an exit inside the callee materializes every frame), `blr` its
    /// entry with x2 = the record, pop, and check the result. When the
    /// callee has no compiled code or the native stack is nearly full, exit
    /// through a `Call` guard: the interpreter performs the call from the
    /// call instruction and carries on interpreting this function.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn compile_call_direct(
        &mut self,
        dest: u8,
        callee: u8,
        receiver: Option<u8>,
        function_idx: usize,
        first_arg: u8,
        arg_count: u8,
        callee_registers: u8,
        call_ip: usize,
        resume_ip: usize,
        alias_mask: u64,
        result_type: Option<ValueType>,
        guard_index: &mut i32,
        guards: &mut Vec<Guard>,
    ) -> Result<()> {
        unsafe extern "C" {
            fn jit_move_safe(src_ptr: *const Value, dest_ptr: *mut Value) -> u8;
        }
        let value_size = mem::size_of::<Value>() as i32;
        let frame_value_count = callee_registers as i32;
        let frame_size = (frame_value_count * value_size + 15) & !15;
        // The receiver, then the arguments, into callee registers 0...
        let sources: Vec<u8> = receiver
            .into_iter()
            .chain((0..arg_count).map(|index| first_arg.wrapping_add(index)))
            .collect();
        let to_interpreter = self.ops.new_dynamic_label();
        let done = self.ops.new_dynamic_label();
        let epilogue = self
            .function_epilogue
            .expect("function code is compiled inside compile_trace");

        // Native stack limit (see `JIT_STACK_LIMIT`), the interpreter's
        // frame depth limit (`JIT_DEPTH_BUDGET`) and the callee's entry
        // point (from the entry table), all from their cells: the
        // callee-saved registers are for pinned values.
        let slot_offset = function_idx * mem::size_of::<usize>();
        let stack_limit = jit::STACK_LIMIT_OFFSET as u32;
        let depth_budget = jit::DEPTH_BUDGET_OFFSET as u32;
        let entry_table = jit::ENTRY_TABLE_OFFSET as u32;
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x9, [x20, #stack_limit]
            ; mov x10, sp
            ; cmp x10, x9
            ; b.lo => to_interpreter
            ; ldr x10, [x20, #depth_budget]
            ; cbz x10, => to_interpreter
            ; ldr x11, [x20, #entry_table]
        );
        if slot_offset <= 32760 {
            dynasm!(self.ops ; .arch aarch64 ; ldr x16, [x11, #slot_offset as u32]);
        } else {
            self.emit_mov_imm64(9, slot_offset as u64);
            dynasm!(self.ops ; .arch aarch64 ; ldr x16, [x11, x9]);
        }
        dynasm!(self.ops ; .arch aarch64 ; cbz x16, => to_interpreter);

        // Push the record (see `JitInlineRecord`): value_count, caller
        // regs, previous chain, alias mask, function, result register,
        // callee register, where the caller resumes if the frame is
        // materialized.
        let site = self.retain_call_site(crate::vm::JitCallSite {
            value_count: frame_value_count as usize,
            alias_mask: alias_mask as usize,
            function_idx,
            return_dest: dest as usize,
            callee_reg: callee as usize,
            caller_resume_ip: resume_ip,
        });
        self.emit_mov_imm64(0, site as u64);
        dynasm!(self.ops
            ; .arch aarch64
            ; stp x19, x21, [sp, #-(INLINE_METADATA_SIZE)]!
            ; str x0, [sp, 16]
        );

        // The callee frame: every register Nil (tag 0), then the arguments.
        self.emit_sub_sp(frame_size);
        dynasm!(self.ops ; .arch aarch64 ; mov x11, sp);
        for reg in 0..callee_registers {
            let offset = super::registers::reg_offset(reg) as u32;
            dynasm!(self.ops ; .arch aarch64 ; strb wzr, [x11, #offset]);
        }
        // x16 (the entry) survives the argument moves unless one calls a
        // helper; then it is reloaded.
        let mut helper_called = false;
        for (index, src_reg) in sources.into_iter().enumerate() {
            let dest_offset = index as i32 * value_size;
            if index < 64 && alias_mask & (1 << index) != 0 {
                // Aliased: a bitwise copy the callee only reads and does
                // not drop (its record says so).
                self.emit_reg_addr(11, src_reg);
                dynasm!(self.ops ; .arch aarch64 ; mov x9, sp);
                self.emit_add_imm(9, 9, dest_offset);
                for chunk in (0..value_size).step_by(16) {
                    dynasm!(self.ops
                        ; .arch aarch64
                        ; ldp x0, x1, [x11, #chunk]
                        ; stp x0, x1, [x9, #chunk]
                    );
                }
                continue;
            }
            match self.scalar_registers.get(&src_reg).copied() {
                Some(ty @ (ValueType::Int | ValueType::Bool | ValueType::Float)) => {
                    let tag = match ty {
                        ValueType::Int => ValueTag::Int,
                        ValueType::Bool => ValueTag::Bool,
                        _ => ValueTag::Float,
                    }
                    .as_u8() as u32;
                    let payload_offset = (dest_offset + 8) as u32;
                    let tag_offset = dest_offset as u32;
                    if ty == ValueType::Bool {
                        self.load_bool_payload(0, src_reg);
                    } else {
                        self.load_payload(0, src_reg);
                    }
                    dynasm!(self.ops
                        ; .arch aarch64
                        ; mov x11, sp
                        ; movz w13, #tag
                        ; strb w13, [x11, #tag_offset]
                        ; str x0, [x11, #payload_offset]
                    );
                }
                _ => {
                    helper_called = true;
                    self.emit_reg_addr(0, src_reg);
                    dynasm!(self.ops ; .arch aarch64 ; mov x11, sp);
                    self.emit_add_imm(1, 11, dest_offset);
                    self.emit_call(jit_move_safe as *const ());
                    self.emit_fail_if_w0_zero();
                }
            }
        }
        // A failed argument move resumes at the call, in this frame.
        self.emit_fail_stub();

        // Enter the callee: x0 = its registers, x1 = VM, x2 = its record.
        // The entry is reloaded: the argument moves above may have called
        // helpers through x16. The depth budget is spent for the call's
        // duration.
        if helper_called {
            dynasm!(self.ops ; .arch aarch64 ; ldr x11, [x20, #entry_table]);
            if slot_offset <= 32760 {
                dynasm!(self.ops ; .arch aarch64 ; ldr x16, [x11, #slot_offset as u32]);
            } else {
                self.emit_mov_imm64(9, slot_offset as u64);
                dynasm!(self.ops ; .arch aarch64 ; ldr x16, [x11, x9]);
            }
        }
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x9, [x20, #depth_budget]
            ; sub x9, x9, 1
            ; str x9, [x20, #depth_budget]
            ; mov x0, sp
            ; mov x1, x20
            ; mov x11, sp
        );
        self.emit_add_imm(2, 11, frame_size);
        self.emit_reg_addr(3, dest);
        dynasm!(self.ops
            ; .arch aarch64
            ; blr x16
            ; ldr x9, [x20, #depth_budget]
            ; add x9, x9, 1
            ; str x9, [x20, #depth_budget]
        );
        // Pop the frame and record; the callee's epilogue restored x19..x24.
        let pop = (frame_size + INLINE_METADATA_SIZE) as u32;
        if pop <= 4095 {
            dynasm!(self.ops ; .arch aarch64 ; add sp, sp, #pop);
        } else {
            dynasm!(self.ops ; .arch aarch64 ; mov x11, sp);
            self.emit_add_imm(11, 11, pop as i32);
            dynasm!(self.ops ; .arch aarch64 ; mov sp, x11);
        }
        let returned_hi = (jit::NATIVE_RETURNED as u32) >> 16;
        dynasm!(self.ops
            ; .arch aarch64
            ; movz w9, #returned_hi, lsl #16
            ; cmp w0, w9
            ; b.eq => done
            // Anything else is an exit that already materialized every
            // frame (ours included): propagate it.
            ; b => epilogue
            ; => to_interpreter
        );
        let guard_return_value = *guard_index + 1;
        self.exit_is_handoff = true;
        self.emit_guard_exit(guard_return_value);
        self.exit_is_handoff = false;
        guards.push(Guard {
            index: *guard_index as usize,
            bailout_ip: call_ip,
            kind: GuardKind::Call { function_idx },
            fail_count: 0,
        });
        *guard_index += 1;
        dynasm!(self.ops ; .arch aarch64 ; => done);
        if let Some(ty) = result_type {
            // The callee returned its declared scalar result in x1.
            match ty {
                ValueType::Float => {
                    dynasm!(self.ops ; .arch aarch64 ; fmov d0, x1);
                    self.store_d0_as_float(dest);
                }
                ValueType::Int => {
                    dynasm!(self.ops ; .arch aarch64 ; mov x0, x1);
                    self.store_from_x0(dest, ValueTag::Int.as_u8());
                }
                _ => {
                    dynasm!(self.ops ; .arch aarch64 ; mov x0, x1);
                    self.store_from_x0(dest, ValueTag::Bool.as_u8());
                }
            }
        }
        Ok(())
    }

    /// Drop this frame's registers at a native return, except the aliased
    /// arguments (the record's mask; no record when entered from the
    /// interpreter), which are the caller's.
    fn emit_drop_frame(&mut self, register_count: u8) {
        unsafe extern "C" {
            fn jit_drop_values_masked(values: *mut Value, len: usize, mask: u64);
        }
        // Only the registers that may own something here are released,
        // inline: not the scalars, not the parameters a native caller
        // aliased (the record's mask says which, but they own nothing
        // either way — an unaliased one was copied as a scalar).
        if register_count <= 64 {
            for reg in 0..register_count {
                if self.function_alias_params & (1u64 << reg) != 0
                    || self.scalar_registers.contains_key(&reg)
                {
                    continue;
                }
                self.emit_reg_addr(11, reg);
                self.emit_release_at_x11();
            }
            return;
        }
        dynasm!(self.ops
            ; .arch aarch64
            ; mov x0, x19
            ; mov x2, xzr
            ; cbz x21, >no_record
            ; ldr x2, [x21, 16]
            ; ldr x2, [x2, 8]
            ; no_record:
        );
        self.emit_mov_imm_i32(1, i32::from(register_count));
        self.emit_call(jit_drop_values_masked as *const ());
    }

    /// `Return` of function code. Called natively (x21 = our record): move
    /// the value into the caller's result register, drop our registers if
    /// any may own something, and return `NATIVE_RETURNED`. Entered from the
    /// interpreter (x21 = 0): exit with `FUNCTION_RETURN_BASE + register`
    /// and let it perform the return.
    pub(super) fn compile_function_return(
        &mut self,
        value: Option<u8>,
        _guard_index: &mut i32,
        _guards: &mut Vec<Guard>,
    ) -> Result<()> {
        unsafe extern "C" {
            fn jit_return_value(src: *mut Value, dest: *mut Value);
        }
        let (register_count, may_own) = self.function_frame;
        let epilogue = self
            .function_epilogue
            .expect("function code is compiled inside compile_trace");
        let exit_label = self.current_exit_label();
        let interp_return = self.ops.new_dynamic_label();
        dynasm!(self.ops
            ; .arch aarch64
            ; cbz x22, => interp_return
        );
        if let (Some(ty), Some(reg)) = (self.function_result, value) {
            // A declared scalar result goes back in x1 (its payload bits;
            // the translator's guard before the `Return` proved the type)
            // and the caller stores it: no store here, no tag to check.
            if may_own {
                self.emit_drop_frame(register_count);
            }
            if ty == ValueType::Bool {
                self.load_bool_payload(1, reg);
            } else {
                self.load_payload(1, reg);
            }
            let returned_hi = (jit::NATIVE_RETURNED as u32) >> 16;
            dynasm!(self.ops
                ; .arch aarch64
                ; movz w0, #returned_hi, lsl #16
                ; b => epilogue
                ; => interp_return
            );
            let code = jit::FUNCTION_RETURN_BASE + i32::from(reg);
            self.emit_mov_imm_i32(0, code);
            dynasm!(self.ops ; .arch aarch64 ; b => exit_label);
            return Ok(());
        }
        dynasm!(self.ops ; .arch aarch64 ; mov x1, x22);
        // A scalar result of known type is stored directly when the
        // caller's register holds nothing owned (it usually holds Nil or
        // the previous scalar); the helper handles everything else.
        let scalar = value.and_then(|reg| {
            self.scalar_registers
                .get(&reg)
                .copied()
                .filter(|ty| matches!(ty, ValueType::Int | ValueType::Bool | ValueType::Float))
                .map(|ty| (reg, ty))
        });
        let stored = self.ops.new_dynamic_label();
        if let Some((reg, ty)) = scalar {
            let tag = match ty {
                ValueType::Int => ValueTag::Int,
                ValueType::Bool => ValueTag::Bool,
                _ => ValueTag::Float,
            }
            .as_u8() as u32;
            let scalar_max_tag = ValueTag::Float.as_u8() as u32;
            if ty == ValueType::Bool {
                self.load_bool_payload(0, reg);
            } else {
                self.load_payload(0, reg);
            }
            dynasm!(self.ops
                ; .arch aarch64
                ; ldrb w9, [x1]
                ; cmp w9, #scalar_max_tag
                ; b.hi >owned
                ; movz w13, #tag
                ; strb w13, [x1]
                ; str x0, [x1, 8]
                ; b => stored
                ; owned:
            );
        }
        match value {
            Some(reg) => self.emit_reg_addr(0, reg),
            None => dynasm!(self.ops ; .arch aarch64 ; mov x0, xzr),
        }
        self.emit_call(jit_return_value as *const ());
        dynasm!(self.ops ; .arch aarch64 ; => stored);
        if may_own {
            self.emit_drop_frame(register_count);
        }
        let returned_hi = (jit::NATIVE_RETURNED as u32) >> 16;
        dynasm!(self.ops
            ; .arch aarch64
            ; movz w0, #returned_hi, lsl #16
            ; b => epilogue
            ; => interp_return
        );
        let code = jit::FUNCTION_RETURN_BASE + i32::from(value.unwrap_or(255));
        self.emit_mov_imm_i32(0, code);
        dynasm!(self.ops ; .arch aarch64 ; b => exit_label);
        Ok(())
    }
}

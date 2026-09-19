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
        let metadata_size = INLINE_METADATA_SIZE as u32;
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
        let _ = result_type;

        // Native stack limit (x23, see `JIT_STACK_LIMIT`) and the callee's
        // entry point (x24 = the entry table), if it has compiled code.
        let slot_offset = (function_idx * mem::size_of::<usize>()) as u32;
        dynasm!(self.ops
            ; .arch aarch64
            ; mov x9, sp
            ; cmp x9, x23
            ; b.lo => to_interpreter
            // The interpreter's frame depth limit (x26, see
            // `JIT_DEPTH_BUDGET`).
            ; cbz x26, => to_interpreter
        );
        if slot_offset <= 32760 {
            dynasm!(self.ops ; .arch aarch64 ; ldr x16, [x24, #slot_offset]);
        } else {
            self.emit_mov_imm64(11, slot_offset as u64);
            dynasm!(self.ops ; .arch aarch64 ; ldr x16, [x24, x11]);
        }
        dynasm!(self.ops ; .arch aarch64 ; cbz x16, => to_interpreter);

        // Push the record (see `JitInlineRecord`): value_count, caller
        // regs, previous chain, alias mask, function, result register,
        // callee register, where the caller resumes if the frame is
        // materialized.
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
        self.emit_mov_imm64(0, function_idx as u64);
        dynasm!(self.ops ; .arch aarch64 ; str x0, [sp, 32]);
        self.emit_mov_imm64(0, dest as u64);
        dynasm!(self.ops ; .arch aarch64 ; str x0, [sp, 40]);
        self.emit_mov_imm64(0, callee as u64);
        dynasm!(self.ops ; .arch aarch64 ; str x0, [sp, 48]);
        self.emit_mov_imm64(0, (call_ip + 1) as u64);
        dynasm!(self.ops ; .arch aarch64 ; str x0, [sp, 56]);

        // The callee frame: every register Nil (tag 0), then the arguments.
        self.emit_sub_sp(frame_size);
        dynasm!(self.ops ; .arch aarch64 ; mov x11, sp);
        for reg in 0..callee_registers {
            let offset = super::registers::reg_offset(reg) as u32;
            dynasm!(self.ops ; .arch aarch64 ; strb wzr, [x11, #offset]);
        }
        for (index, src_reg) in sources.into_iter().enumerate() {
            let dest_offset = index as i32 * value_size;
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
        // helpers through x16.
        if slot_offset <= 32760 {
            dynasm!(self.ops ; .arch aarch64 ; ldr x16, [x24, #slot_offset]);
        } else {
            self.emit_mov_imm64(11, slot_offset as u64);
            dynasm!(self.ops ; .arch aarch64 ; ldr x16, [x24, x11]);
        }
        dynasm!(self.ops
            ; .arch aarch64
            ; sub x26, x26, 1
            ; mov x0, sp
            ; mov x1, x20
            ; mov x11, sp
        );
        self.emit_add_imm(2, 11, frame_size);
        dynasm!(self.ops
            ; .arch aarch64
            ; blr x16
            ; add x26, x26, 1
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
        Ok(())
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
        let value_size = mem::size_of::<Value>() as u32;
        unsafe extern "C" {
            fn jit_return_value(src: *mut Value, dest: *mut Value);
            fn jit_drop_values(values: *mut Value, len: usize);
        }
        let (register_count, may_own) = self.function_frame;
        let epilogue = self
            .function_epilogue
            .expect("function code is compiled inside compile_trace");
        let exit_label = self.current_exit_label();
        let interp_return = self.ops.new_dynamic_label();
        dynasm!(self.ops
            ; .arch aarch64
            ; cbz x21, => interp_return
            ; ldr x11, [x21, 8]
            ; ldr x9, [x21, 40]
            ; movz w10, #value_size
            ; madd x1, x9, x10, x11
        );
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
            dynasm!(self.ops ; .arch aarch64 ; mov x0, x19);
            self.emit_mov_imm_i32(1, i32::from(register_count));
            self.emit_call(jit_drop_values as *const ());
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

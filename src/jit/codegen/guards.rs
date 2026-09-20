use super::*;
use crate::VM;
impl JitCompiler {
    pub(super) fn compile_guard(
        &mut self,
        register: u8,
        expected_type: ValueType,
        guard_index: usize,
    ) -> Result<Guard> {
        let offset = (register as i32) * (mem::size_of::<Value>() as i32);
        let expected_tag = match expected_type {
            ValueType::Bool => ValueTag::Bool,
            ValueType::Int => ValueTag::Int,
            ValueType::Float => ValueTag::Float,
            ValueType::String => ValueTag::String,
            ValueType::Array => ValueTag::Array,
            ValueType::Tuple => ValueTag::Tuple,
            ValueType::Struct => ValueTag::Struct,
        };
        let expected_discriminant = expected_tag.as_u8() as i8;
        let guard_return_value = (guard_index + 1) as i32;
        dynasm!(self.ops
            ; .arch x64
            ; mov al, [r12 + offset]
            ; cmp al, BYTE expected_discriminant
            ; jne >guard_fail
            ; jmp >guard_ok
            ; guard_fail:
        );
        self.emit_guard_exit(guard_return_value);
        dynasm!(self.ops
            ; .arch x64
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
        let offset = (register as i32) * (mem::size_of::<Value>() as i32);
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
        let kind_flag: i32 = if is_closure { 1 } else { 0 };
        let reg_index = register as i32;
        dynasm!(self.ops
            ; .arch x64
            ; lea rdi, [r12 + offset]
            ; mov esi, DWORD kind_flag
            ; mov rdx, QWORD function_idx as _
            ; mov rcx, QWORD upvalues_ptr as _
            ; mov r8d, DWORD reg_index
            ; mov rax, QWORD jit_guard_function_identity as *const () as _
            ; call rax
            ; test al, al
            ; jz >guard_fail
            ; jmp >guard_ok
            ; guard_fail:
        );
        self.emit_guard_exit(guard_return_value);
        dynasm!(self.ops
            ; .arch x64
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
        let offset = (register as i32) * (mem::size_of::<Value>() as i32);
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
            let tag = layout.single_rc_tags[4] as i8;
            let rc_offset = layout.single_rc_offset as i32;
            dynasm!(self.ops
                ; .arch x64
                ; cmp BYTE [r12 + offset], tag
                ; jne >guard_fail
                ; mov rax, QWORD expected_inner as i64
                ; cmp rax, [r12 + offset + rc_offset]
                ; je >guard_ok
                ; guard_fail:
            );
            self.emit_guard_exit(guard_return_value);
            dynasm!(self.ops
                ; .arch x64
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
        let reg_index = register as i32;
        dynasm!(self.ops
            ; .arch x64
            ; lea rdi, [r12 + offset]
            ; mov rsi, QWORD expected_ptr as _
            ; mov edx, DWORD reg_index
            ; mov rax, QWORD jit_guard_native_function as *const () as _
            ; call rax
            ; test al, al
            ; jz >guard_fail
            ; jmp >guard_ok
            ; guard_fail:
        );
        self.emit_guard_exit(guard_return_value);
        dynasm!(self.ops
            ; .arch x64
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
        let offset = core::mem::offset_of!(crate::vm::VM, globals_version) as i32;
        dynasm!(self.ops
            ; .arch x64
            ; mov rax, QWORD version as i64
            ; cmp rax, [r13 + offset]
            ; je >guard_ok
        );
        self.emit_guard_exit(guard_return_value);
        dynasm!(self.ops
            ; .arch x64
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
        let offset = (register as i32) * (mem::size_of::<Value>() as i32);
        let guard_return_value = (guard_index + 1) as i32;
        unsafe extern "C" {
            fn jit_guard_struct_layout(value_ptr: *const Value, expected: *const ()) -> u8;
        }
        if let Some(measured) = jit::layout::rc_vec_layout() {
            // Inline: the struct tag, then the layout's allocation pointer
            // (`expected` is the `Rc`'s data pointer, 16 bytes in).
            let struct_tag = ValueTag::Struct.as_u8() as i8;
            let layout_offset = measured.struct_layout_offset as i32;
            let inner = (layout as usize).wrapping_sub(16) as i64;
            dynasm!(self.ops
                ; .arch x64
                ; cmp BYTE [r12 + offset], struct_tag
                ; jne >guard_fail
                ; mov rax, QWORD inner
                ; cmp rax, [r12 + offset + layout_offset]
                ; je >guard_ok
                ; guard_fail:
            );
            self.emit_guard_exit(guard_return_value);
            dynasm!(self.ops
                ; .arch x64
                ; guard_ok:
            );
            return Ok(Guard {
                index: guard_index,
                bailout_ip: self.guard_bailout_ip(),
                kind: GuardKind::StructLayout { register, layout },
                fail_count: 0,
            });
        }
        dynasm!(self.ops
            ; .arch x64
            ; lea rdi, [r12 + offset]
            ; mov rsi, QWORD layout as usize as _
            ; mov rax, QWORD jit_guard_struct_layout as *const () as _
            ; call rax
            ; test al, al
            ; jnz >guard_ok
        );
        self.emit_guard_exit(guard_return_value);
        dynasm!(self.ops
            ; .arch x64
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
                ; .arch x64
                ; mov rdi, r13
                ; mov rsi, r12
                ; mov rdx, QWORD function_idx as _
                ; mov rcx, QWORD loop_start_ip as _
                ; mov r8, QWORD resume_ip as _
                ; mov rax, QWORD jit_run_nested_loop as *const () as _
                ; call rax
                ; test eax, eax
                ; jz >loop_done
            );
        }
        dynasm!(self.ops
            ; .arch x64
        );
        self.emit_guard_exit(guard_return_value);
        dynasm!(self.ops
            ; .arch x64
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
        let cond_offset = (condition_register as i32) * (mem::size_of::<Value>() as i32);
        let guard_return_value = (guard_index + 1) as i32;
        let bool_tag = ValueTag::Bool.as_u8() as i8;
        let scalar_max_tag = ValueTag::Float.as_u8() as i8;
        unsafe extern "C" {
            fn jit_value_is_truthy(value_ptr: *const Value) -> u8;
        }
        if self.scalar_registers.get(&condition_register) == Some(&ValueType::Bool) {
            dynasm!(self.ops ; .arch x64 ; cmp BYTE [r12 + cond_offset + 8], 0);
        } else {
            dynasm!(self.ops
                ; .arch x64
                ; mov al, BYTE [r12 + cond_offset]
                ; cmp al, scalar_max_tag
                ; ja >generic_truthiness
                ; cmp al, bool_tag
                ; je >load_bool
                // Nil is false; numeric scalars are true regardless of payload.
                ; test al, al
                ; setnz al
                ; jmp >truthiness_ready
                ; load_bool:
                ; mov al, BYTE [r12 + cond_offset + 8]
                ; jmp >truthiness_ready
                ; generic_truthiness:
                ; lea rdi, [r12 + cond_offset]
                ; mov rax, QWORD jit_value_is_truthy as *const () as _
                ; call rax
                ; truthiness_ready:
                ; test al, al
            );
        }
        if expect_truthy {
            dynasm!(self.ops
                ; .arch x64
                ; jnz >guard_ok
            );
            self.emit_guard_exit(guard_return_value);
            dynasm!(self.ops
                ; .arch x64
                ; guard_ok:
            );
        } else {
            dynasm!(self.ops
                ; .arch x64
                ; jz >guard_ok
            );
            self.emit_guard_exit(guard_return_value);
            dynasm!(self.ops
                ; .arch x64
                ; guard_ok:
            );
        }
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
        let offset = (register as i32) * (mem::size_of::<Value>() as i32);
        if self.scalar_registers.get(&register) == Some(&ValueType::Bool) {
            // Only the low byte of a Bool's payload is defined.
            dynasm!(self.ops ; .arch x64 ; movzx eax, BYTE [r12 + offset + 8]);
        } else {
            unsafe extern "C" {
                fn jit_value_is_truthy(value_ptr: *const Value) -> u8;
            }
            dynasm!(self.ops
                ; .arch x64
                ; lea rdi, [r12 + offset]
                ; mov rax, QWORD jit_value_is_truthy as *const () as _
                ; call rax
                ; movzx eax, al
            );
        }
        dynasm!(self.ops ; .arch x64 ; test rax, rax);
        if expect_truthy {
            dynasm!(self.ops ; .arch x64 ; jnz => label);
        } else {
            dynasm!(self.ops ; .arch x64 ; jz => label);
        }
    }

    /// Call another bytecode function's compiled code natively (see the
    /// aarch64 backend for the protocol): record + frame pushed exactly as
    /// for an inlined call, `call` its entry with rdx = the record, pop,
    /// check the result; without a compiled callee or with the native
    /// stack nearly full, exit through a `Call` guard and let the
    /// interpreter perform the call.
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
        let metadata_size = INLINE_METADATA_SIZE;
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
        let slot_offset = (function_idx * mem::size_of::<usize>()) as i32;
        let site = self.retain_call_site(crate::vm::JitCallSite {
            value_count: frame_value_count as usize,
            alias_mask: alias_mask as usize,
            function_idx,
            return_dest: dest as usize,
            callee_reg: callee as usize,
            caller_resume_ip: resume_ip,
        });
        dynasm!(self.ops
            ; .arch x64
            // Native stack limit and the interpreter's frame depth limit
            // (see `JitCells`).
            ; cmp rsp, [r13 + jit::STACK_LIMIT_OFFSET as i32]
            ; jb => to_interpreter
            ; cmp QWORD [r13 + jit::DEPTH_BUDGET_OFFSET as i32], 0
            ; je => to_interpreter
            // The callee's entry point, if it has compiled code.
            ; mov rax, [r13 + jit::ENTRY_TABLE_OFFSET as i32]
            ; mov rbx, [rax + slot_offset]
            ; test rbx, rbx
            ; jz => to_interpreter
            // The record (see `JitInlineRecord`).
            ; sub rsp, metadata_size
            ; mov [rsp], r12
            ; mov [rsp + 8], r15
            ; mov rax, QWORD site as i64
            ; mov [rsp + 16], rax
            // The callee frame: every register Nil, then the arguments.
            ; sub rsp, frame_size
        );
        for reg in 0..callee_registers {
            let offset = reg as i32 * value_size;
            dynasm!(self.ops ; .arch x64 ; mov BYTE [rsp + offset], 0);
        }
        for (index, src_reg) in sources.into_iter().enumerate() {
            let src_offset = (src_reg as i32) * value_size;
            let dest_offset = index as i32 * value_size;
            if index < 64 && alias_mask & (1 << index) != 0 {
                // Aliased: a bitwise copy the callee only reads and does
                // not drop (its record says so).
                for word in (0..value_size).step_by(8) {
                    dynasm!(self.ops
                        ; .arch x64
                        ; mov rax, [r12 + src_offset + word]
                        ; mov [rsp + dest_offset + word], rax
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
                    .as_u8() as i8;
                    if ty == ValueType::Bool {
                        dynasm!(self.ops ; .arch x64 ; movzx eax, BYTE [r12 + src_offset + 8]);
                    } else {
                        dynasm!(self.ops ; .arch x64 ; mov rax, [r12 + src_offset + 8]);
                    }
                    dynasm!(self.ops
                        ; .arch x64
                        ; mov BYTE [rsp + dest_offset], tag
                        ; mov [rsp + dest_offset + 8], rax
                    );
                }
                _ => {
                    dynasm!(self.ops
                        ; .arch x64
                        ; lea rdi, [r12 + src_offset]
                        ; lea rsi, [rsp + dest_offset]
                        ; mov rax, QWORD jit_move_safe as *const () as _
                        ; call rax
                        ; test al, al
                        ; jz >fail
                    );
                }
            }
        }
        // A failed argument move resumes at the call, in this frame.
        self.emit_fail_stub();

        let returned = jit::NATIVE_RETURNED;
        dynasm!(self.ops
            ; .arch x64
            // Enter the callee: rdi = its registers, rsi = VM, rdx = record.
            ; dec QWORD [r13 + jit::DEPTH_BUDGET_OFFSET as i32]
            ; mov rdi, rsp
            ; mov rsi, r13
            ; lea rdx, [rsp + frame_size]
            ; lea rcx, [r12 + (dest as i32) * value_size]
            ; call rbx
            // Pop the frame and record; the callee's epilogue restored
            // rbx, r12..r15.
            ; add rsp, frame_size + metadata_size
            ; inc QWORD [r13 + jit::DEPTH_BUDGET_OFFSET as i32]
            ; cmp eax, DWORD returned
            ; je => done
            // Anything else is an exit that already materialized every
            // frame (ours included): propagate it.
            ; jmp => epilogue
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
        dynasm!(self.ops ; .arch x64 ; => done);
        if let Some(ty) = result_type {
            // The callee returned its declared scalar result in rdx.
            match ty {
                ValueType::Float => {
                    dynasm!(self.ops ; .arch x64 ; movq xmm0, rdx);
                    self.store_xmm0_as_float(dest);
                }
                ValueType::Int => {
                    dynasm!(self.ops ; .arch x64 ; mov rax, rdx);
                    self.store_from_rax(dest, ValueTag::Int.as_u8());
                }
                _ => {
                    dynasm!(self.ops ; .arch x64 ; mov rax, rdx);
                    self.store_from_rax(dest, ValueTag::Bool.as_u8());
                }
            }
        }
        Ok(())
    }

    /// `Return` of function code (see the aarch64 backend).
    /// Drop this frame's registers at a native return, except the aliased
    /// arguments (the record's mask; no record when entered from the
    /// interpreter), which are the caller's.
    fn emit_drop_frame(&mut self, register_count: u8) {
        unsafe extern "C" {
            fn jit_drop_values_masked(values: *mut Value, len: usize, mask: u64);
        }
        dynasm!(self.ops
            ; .arch x64
            ; mov rdi, r12
            ; mov esi, DWORD i32::from(register_count)
            ; xor edx, edx
            ; test r15, r15
            ; jz >no_record
            ; mov rdx, [r15 + 16]
            ; mov rdx, [rdx + 8]
            ; no_record:
            ; mov rax, QWORD jit_drop_values_masked as *const () as _
            ; call rax
        );
    }

    pub(super) fn compile_function_return(&mut self, value: Option<u8>) -> Result<()> {
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
            ; .arch x64
            ; test r14, r14
            ; jz => interp_return
        );
        if let (Some(ty), Some(reg)) = (self.function_result, value) {
            // A declared scalar result goes back in rdx (its payload bits;
            // the translator's guard before the `Return` proved the type)
            // and the caller stores it: no store here, no tag to check.
            if may_own {
                self.emit_drop_frame(register_count);
            }
            let payload = (reg as i32) * (mem::size_of::<Value>() as i32) + 8;
            if ty == ValueType::Bool {
                dynasm!(self.ops ; .arch x64 ; movzx edx, BYTE [r12 + payload]);
            } else {
                dynasm!(self.ops ; .arch x64 ; mov rdx, [r12 + payload]);
            }
            let returned = jit::NATIVE_RETURNED;
            let code = jit::FUNCTION_RETURN_BASE + i32::from(reg);
            dynasm!(self.ops
                ; .arch x64
                ; mov eax, DWORD returned
                ; jmp => epilogue
                ; => interp_return
                ; mov eax, DWORD code
                ; jmp => exit_label
            );
            return Ok(());
        }
        dynasm!(self.ops ; .arch x64 ; mov rsi, r14);
        // A scalar result of known type is stored directly when the
        // caller's register holds nothing owned; the helper handles the
        // rest.
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
            .as_u8() as i8;
            let scalar_max_tag = ValueTag::Float.as_u8() as i8;
            let offset = (reg as i32) * (mem::size_of::<Value>() as i32);
            dynasm!(self.ops
                ; .arch x64
                ; cmp BYTE [rsi], scalar_max_tag
                ; ja >owned
            );
            if ty == ValueType::Bool {
                dynasm!(self.ops ; .arch x64 ; movzx eax, BYTE [r12 + offset + 8]);
            } else {
                dynasm!(self.ops ; .arch x64 ; mov rax, [r12 + offset + 8]);
            }
            dynasm!(self.ops
                ; .arch x64
                ; mov BYTE [rsi], tag
                ; mov [rsi + 8], rax
                ; jmp => stored
                ; owned:
            );
        }
        match value {
            Some(reg) => {
                let offset = (reg as i32) * (mem::size_of::<Value>() as i32);
                dynasm!(self.ops ; .arch x64 ; lea rdi, [r12 + offset]);
            }
            None => dynasm!(self.ops ; .arch x64 ; xor edi, edi),
        }
        dynasm!(self.ops
            ; .arch x64
            ; mov rax, QWORD jit_return_value as *const () as _
            ; call rax
            ; => stored
        );
        if may_own {
            self.emit_drop_frame(register_count);
        }
        let returned = jit::NATIVE_RETURNED;
        let code = jit::FUNCTION_RETURN_BASE + i32::from(value.unwrap_or(255));
        dynasm!(self.ops
            ; .arch x64
            ; mov eax, DWORD returned
            ; jmp => epilogue
            ; => interp_return
            ; mov eax, DWORD code
            ; jmp => exit_label
        );
        Ok(())
    }
}

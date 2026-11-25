use super::*;
use crate::bytecode::value::JitVecSlot;
use crate::jit::trace::{Operand, SpecializedOpKind};
use crate::number::{LustFloat, LustInt};

fn vec_kind(layout: &SpecializedLayout) -> Option<VecKind> {
    match layout {
        SpecializedLayout::Vec {
            element_layout,
            element_size,
        } => match &**element_layout {
            SpecializedLayout::Scalar { size: 1, .. } => Some(VecKind::Bool),
            SpecializedLayout::Scalar {
                size,
                is_float: false,
                ..
            } if *size == core::mem::size_of::<LustInt>() => Some(VecKind::Int),
            SpecializedLayout::Scalar {
                size,
                is_float: true,
                ..
            } if *size == core::mem::size_of::<LustFloat>() => Some(VecKind::Float),
            _ if *element_size == 1 => Some(VecKind::Bool),
            _ => None,
        },
        _ => None,
    }
}

impl JitCompiler {
    /// Compile Unbox operation - convert Value to specialized representation
    pub(super) fn compile_unbox(
        &mut self,
        specialized_id: usize,
        source_reg: u8,
        layout: &SpecializedLayout,
        guard_index: &mut i32,
        guards: &mut Vec<Guard>,
    ) -> Result<()> {
        match vec_kind(layout) {
            Some(VecKind::Int) => {
                self.compile_unbox_array(specialized_id, source_reg, VecKind::Int, guard_index, guards)
            }
            Some(VecKind::Bool) => {
                self.compile_unbox_array(specialized_id, source_reg, VecKind::Bool, guard_index, guards)
            }
            Some(VecKind::Float) => {
                self.compile_unbox_array(specialized_id, source_reg, VecKind::Float, guard_index, guards)
            }
            None => Err(crate::LustError::RuntimeError {
                message: format!("Unbox not yet implemented for layout {:?}", layout),
            }),
        }
    }

    /// Compile Rebox operation - convert specialized representation back to Value
    pub(super) fn compile_rebox(
        &mut self,
        dest_reg: u8,
        specialized_id: usize,
        layout: &SpecializedLayout,
    ) -> Result<()> {
        match vec_kind(layout) {
            Some(kind) => self.compile_rebox_array(dest_reg, specialized_id, kind),
            None => Err(crate::LustError::RuntimeError {
                message: format!("Rebox not yet implemented for layout {:?}", layout),
            }),
        }
    }

    /// Compile specialized operation
    pub(super) fn compile_specialized_op(
        &mut self,
        op: &SpecializedOpKind,
        operands: &[Operand],
    ) -> Result<()> {
        match op {
            SpecializedOpKind::VecPush => {
                if operands.len() != 2 {
                    return Err(crate::LustError::RuntimeError {
                        message: "VecPush requires 2 operands (vec_id, value)".into(),
                    });
                }
                match (&operands[0], &operands[1]) {
                    (Operand::Specialized(vec_id), Operand::Register(value_reg)) => {
                        let kind = self
                            .specialized_values
                            .get(vec_id)
                            .map(|s| s.kind)
                            .unwrap_or(VecKind::Int);
                        match kind {
                            VecKind::Int => self.compile_vec_int_push(*vec_id, *value_reg)?,
                            VecKind::Bool => self.compile_vec_bool_push(*vec_id, *value_reg)?,
                            VecKind::Float => self.compile_vec_float_push(*vec_id, *value_reg)?,
                        }
                    }
                    _ => {
                        return Err(crate::LustError::RuntimeError {
                            message: "VecPush operands must be (Specialized, Register)".into(),
                        });
                    }
                }
            }
            SpecializedOpKind::VecLen => {
                if operands.len() != 2 {
                    return Err(crate::LustError::RuntimeError {
                        message: "VecLen requires 2 operands (vec_id, dest_reg)".into(),
                    });
                }
                match (&operands[0], &operands[1]) {
                    (Operand::Specialized(vec_id), Operand::Register(dest_reg)) => {
                        self.compile_vec_len(*vec_id, *dest_reg)?;
                    }
                    _ => {
                        return Err(crate::LustError::RuntimeError {
                            message: "VecLen operands must be (Specialized, Register)".into(),
                        });
                    }
                }
            }
            SpecializedOpKind::VecGet => {
                if operands.len() != 3 {
                    return Err(crate::LustError::RuntimeError {
                        message: "VecGet requires 3 operands (dest_reg, vec_id, index)".into(),
                    });
                }
                match (&operands[0], &operands[1], &operands[2]) {
                    (
                        Operand::Register(dest_reg),
                        Operand::Specialized(vec_id),
                        Operand::Register(index_reg),
                    ) => {
                        self.compile_vec_get(*dest_reg, *vec_id, *index_reg)?;
                    }
                    _ => {
                        return Err(crate::LustError::RuntimeError {
                            message: "VecGet operands must be (Register, Specialized, Register)"
                                .into(),
                        });
                    }
                }
            }
            SpecializedOpKind::VecSet => {
                if operands.len() != 3 {
                    return Err(crate::LustError::RuntimeError {
                        message: "VecSet requires 3 operands (vec_id, index, value)".into(),
                    });
                }
                match (&operands[0], &operands[1], &operands[2]) {
                    (
                        Operand::Specialized(vec_id),
                        Operand::Register(index_reg),
                        Operand::Register(value_reg),
                    ) => {
                        self.compile_vec_set(*vec_id, *index_reg, *value_reg)?;
                    }
                    _ => {
                        return Err(crate::LustError::RuntimeError {
                            message: "VecSet operands must be (Specialized, Register, Register)"
                                .into(),
                        });
                    }
                }
            }
            SpecializedOpKind::VecIndexOk => {
                if operands.len() != 4 {
                    return Err(crate::LustError::RuntimeError {
                        message: "VecIndexOk requires 4 operands (value_dest, condition_dest, vec_id, index)"
                            .into(),
                    });
                }
                match (&operands[0], &operands[1], &operands[2], &operands[3]) {
                    (
                        Operand::Register(value_dest),
                        Operand::Register(condition_dest),
                        Operand::Specialized(vec_id),
                        Operand::Register(index_reg),
                    ) => {
                        self.compile_vec_index_ok(
                            *value_dest,
                            *condition_dest,
                            *vec_id,
                            *index_reg,
                        )?;
                    }
                    _ => {
                        return Err(crate::LustError::RuntimeError {
                            message: "VecIndexOk operands must be (Register, Register, Specialized, Register)"
                                .into(),
                        });
                    }
                }
            }
            _ => {
                return Err(crate::LustError::RuntimeError {
                    message: format!("Specialized op {:?} not yet implemented", op),
                });
            }
        }
        Ok(())
    }

    /// Unbox Array<T> - convert Rc<RefCell<Vec<Value>>> to Vec<T> on stack
    fn compile_unbox_array(
        &mut self,
        specialized_id: usize,
        source_reg: u8,
        kind: VecKind,
        guard_index: &mut i32,
        guards: &mut Vec<Guard>,
    ) -> Result<()> {
        jit::log(|| {
            format!(
                "📦 JIT: Unboxing Array<{:?}> from reg {} to specialized #{}",
                kind, source_reg, specialized_id
            )
        });

        let stack_offset = match self.specialized_values.get(&specialized_id) {
            Some(existing) => existing.stack_offset,
            None => {
                let stack_offset = self.allocate_specialized_stack(32, 8);
                self.specialized_values.insert(
                    specialized_id,
                    SpecializedValue { stack_offset, kind },
                );
                stack_offset
            }
        };

        let reg_offset = (source_reg as i32) * mem::size_of::<Value>() as i32;

        unsafe extern "C" {
            fn jit_unbox_array_int(array_value_ptr: *const Value, slot: *mut JitVecSlot) -> u8;
            fn jit_unbox_array_bool(array_value_ptr: *const Value, slot: *mut JitVecSlot) -> u8;
            fn jit_unbox_array_float(array_value_ptr: *const Value, slot: *mut JitVecSlot) -> u8;
        }
        let unbox_fn: *const () = match kind {
            VecKind::Int => jit_unbox_array_int as *const (),
            VecKind::Bool => jit_unbox_array_bool as *const (),
            VecKind::Float => jit_unbox_array_float as *const (),
        };

        let guard_idx = *guard_index as usize;
        *guard_index += 1;
        guards.push(Guard {
            index: guard_idx,
            bailout_ip: self.guard_bailout_ip(),
            kind: GuardKind::Plain {
                register: source_reg,
            },
            fail_count: 0,
        });
        let guard_return_value = (guard_idx + 1) as i32;

        dynasm!(self.ops
            ; .arch x64
            ; lea rdi, [r12 + reg_offset]
            ; lea rsi, [rbp + stack_offset]
            ; mov rax, QWORD unbox_fn as i64
            ; call rax
            ; test al, al
            ; jnz >unbox_ok
        );
        self.emit_guard_exit(guard_return_value);
        dynasm!(self.ops
            ; .arch x64
            ; unbox_ok:
        );

        Ok(())
    }

    /// Rebox Array<T> - convert Vec<T> back to Rc<RefCell<Vec<Value>>>
    fn compile_rebox_array(
        &mut self,
        dest_reg: u8,
        specialized_id: usize,
        kind: VecKind,
    ) -> Result<()> {
        jit::log(|| {
            format!(
                "📦 JIT: Reboxing specialized #{} to Array<{:?}> in reg {}",
                specialized_id, kind, dest_reg
            )
        });

        let spec_value = self
            .specialized_values
            .get(&specialized_id)
            .ok_or_else(|| crate::LustError::RuntimeError {
                message: format!("Specialized value #{} not found", specialized_id),
            })?;

        let stack_offset = spec_value.stack_offset;
        let _ = dest_reg;

        unsafe extern "C" {
            fn jit_rebox_array_int(slot: *mut JitVecSlot) -> u8;
            fn jit_rebox_array_bool(slot: *mut JitVecSlot) -> u8;
            fn jit_rebox_array_float(slot: *mut JitVecSlot) -> u8;
        }
        let rebox_fn: *const () = match kind {
            VecKind::Int => jit_rebox_array_int as *const (),
            VecKind::Bool => jit_rebox_array_bool as *const (),
            VecKind::Float => jit_rebox_array_float as *const (),
        };

        dynasm!(self.ops
            ; .arch x64
            ; lea rdi, [rbp + stack_offset]
            ; mov rax, QWORD rebox_fn as i64
            ; call rax
            ; test al, al
            ; jz >fail
        );

        Ok(())
    }

    /// Specialized push for Vec<LustInt>
    fn compile_vec_int_push(&mut self, vec_id: usize, value_reg: u8) -> Result<()> {
        jit::log(|| {
            format!(
                "⚡ JIT: Specialized int push to vec #{} from reg {}",
                vec_id, value_reg
            )
        });

        let spec_value =
            self.specialized_values
                .get(&vec_id)
                .ok_or_else(|| crate::LustError::RuntimeError {
                    message: format!("Specialized vec #{} not found", vec_id),
                })?;

        let stack_offset = spec_value.stack_offset;
        let value_offset = (value_reg as i32) * mem::size_of::<Value>() as i32 + 8;

        unsafe extern "C" {
            fn jit_vec_int_push(
                vec_ptr: *mut *mut LustInt,
                vec_len: *mut usize,
                vec_cap: *mut usize,
                value: LustInt,
            ) -> u8;
        }
        let push_fn = jit_vec_int_push as *const ();

        dynasm!(self.ops
            ; .arch x64
            ; lea rdi, [rbp + stack_offset]
            ; lea rsi, [rbp + stack_offset + 8]
            ; lea rdx, [rbp + stack_offset + 16]
            ; mov rcx, [r12 + value_offset]
            ; mov rax, QWORD push_fn as i64
            ; call rax
            ; test al, al
            ; jz >fail
        );

        Ok(())
    }

    /// Specialized push for Vec<bool>
    fn compile_vec_bool_push(&mut self, vec_id: usize, value_reg: u8) -> Result<()> {
        jit::log(|| {
            format!(
                "⚡ JIT: Specialized bool push to vec #{} from reg {}",
                vec_id, value_reg
            )
        });

        let spec_value =
            self.specialized_values
                .get(&vec_id)
                .ok_or_else(|| crate::LustError::RuntimeError {
                    message: format!("Specialized vec #{} not found", vec_id),
                })?;

        let stack_offset = spec_value.stack_offset;
        let value_offset = (value_reg as i32) * mem::size_of::<Value>() as i32 + 8;

        unsafe extern "C" {
            fn jit_vec_bool_push(
                vec_ptr: *mut *mut u8,
                vec_len: *mut usize,
                vec_cap: *mut usize,
                value: u8,
            ) -> u8;
        }
        let push_fn = jit_vec_bool_push as *const ();

        dynasm!(self.ops
            ; .arch x64
            ; lea rdi, [rbp + stack_offset]
            ; lea rsi, [rbp + stack_offset + 8]
            ; lea rdx, [rbp + stack_offset + 16]
            ; movzx ecx, BYTE [r12 + value_offset]
            ; mov rax, QWORD push_fn as i64
            ; call rax
            ; test al, al
            ; jz >fail
        );

        Ok(())
    }

    /// Specialized push for Vec<LustFloat>
    fn compile_vec_float_push(&mut self, vec_id: usize, value_reg: u8) -> Result<()> {
        jit::log(|| {
            format!(
                "⚡ JIT: Specialized float push to vec #{} from reg {}",
                vec_id, value_reg
            )
        });

        let spec_value =
            self.specialized_values
                .get(&vec_id)
                .ok_or_else(|| crate::LustError::RuntimeError {
                    message: format!("Specialized vec #{} not found", vec_id),
                })?;

        let stack_offset = spec_value.stack_offset;
        let value_offset = (value_reg as i32) * mem::size_of::<Value>() as i32 + 8;

        unsafe extern "C" {
            fn jit_vec_float_push(
                vec_ptr: *mut *mut LustFloat,
                vec_len: *mut usize,
                vec_cap: *mut usize,
                value: LustFloat,
            ) -> u8;
        }
        let push_fn = jit_vec_float_push as *const ();

        dynasm!(self.ops
            ; .arch x64
            ; lea rdi, [rbp + stack_offset]
            ; lea rsi, [rbp + stack_offset + 8]
            ; lea rdx, [rbp + stack_offset + 16]
            ; movsd xmm0, [r12 + value_offset]
            ; mov rax, QWORD push_fn as i64
            ; call rax
            ; test al, al
            ; jz >fail
        );

        Ok(())
    }

    /// Specialized element load (VecGet)
    fn compile_vec_get(&mut self, dest_reg: u8, vec_id: usize, index_reg: u8) -> Result<()> {
        let specialized = self.specialized_values.get(&vec_id).cloned().ok_or_else(|| {
            crate::LustError::RuntimeError {
                message: format!("Specialized vec #{} not found", vec_id),
            }
        })?;
        let stack_offset = specialized.stack_offset;
        let idx_offset = (index_reg as i32) * mem::size_of::<Value>() as i32 + 8;

        dynasm!(self.ops
            ; .arch x64
            ; mov rcx, [r12 + idx_offset]
            ; cmp rcx, [rbp + stack_offset + 8]
            ; jae >fail
            ; mov rdx, [rbp + stack_offset]
        );

        match specialized.kind {
            VecKind::Bool => {
                dynasm!(self.ops
                    ; .arch x64
                    ; movzx eax, BYTE [rdx + rcx]
                );
                self.store_from_rax(dest_reg, ValueTag::Bool.as_u8());
            }
            VecKind::Int => {
                dynasm!(self.ops
                    ; .arch x64
                    ; mov rax, [rdx + rcx*8]
                );
                self.store_from_rax(dest_reg, ValueTag::Int.as_u8());
            }
            VecKind::Float => {
                dynasm!(self.ops
                    ; .arch x64
                    ; movsd xmm0, [rdx + rcx*8]
                );
                self.store_xmm0_as_float(dest_reg);
            }
        }
        Ok(())
    }

    /// Specialized element store (VecSet)
    fn compile_vec_set(&mut self, vec_id: usize, index_reg: u8, value_reg: u8) -> Result<()> {
        let specialized = self.specialized_values.get(&vec_id).cloned().ok_or_else(|| {
            crate::LustError::RuntimeError {
                message: format!("Specialized vec #{} not found", vec_id),
            }
        })?;
        let stack_offset = specialized.stack_offset;
        let idx_offset = (index_reg as i32) * mem::size_of::<Value>() as i32 + 8;
        let val_offset = (value_reg as i32) * mem::size_of::<Value>() as i32 + 8;

        dynasm!(self.ops
            ; .arch x64
            ; mov rcx, [r12 + idx_offset]
            ; cmp rcx, [rbp + stack_offset + 8]
            ; jae >fail
            ; mov rdx, [rbp + stack_offset]
        );

        match specialized.kind {
            VecKind::Bool => {
                dynasm!(self.ops
                    ; .arch x64
                    ; mov al, BYTE [r12 + val_offset]
                    ; mov BYTE [rdx + rcx], al
                );
            }
            VecKind::Int | VecKind::Float => {
                dynasm!(self.ops
                    ; .arch x64
                    ; mov rax, [r12 + val_offset]
                    ; mov [rdx + rcx*8], rax
                );
            }
        }
        Ok(())
    }

    /// Specialized element load with bounds check (VecIndexOk)
    fn compile_vec_index_ok(
        &mut self,
        value_dest: u8,
        condition_dest: u8,
        vec_id: usize,
        index_reg: u8,
    ) -> Result<()> {
        let specialized = self.specialized_values.get(&vec_id).cloned().ok_or_else(|| {
            crate::LustError::RuntimeError {
                message: format!("Specialized vec #{} not found", vec_id),
            }
        })?;
        let stack_offset = specialized.stack_offset;
        let idx_offset = (index_reg as i32) * mem::size_of::<Value>() as i32 + 8;

        let out_of_range = self.ops.new_dynamic_label();
        let done = self.ops.new_dynamic_label();

        dynasm!(self.ops
            ; .arch x64
            ; mov rcx, [r12 + idx_offset]
            ; cmp rcx, [rbp + stack_offset + 8]
            ; jae => out_of_range
            ; mov rdx, [rbp + stack_offset]
        );

        match specialized.kind {
            VecKind::Bool => {
                dynasm!(self.ops
                    ; .arch x64
                    ; movzx eax, BYTE [rdx + rcx]
                );
                self.store_from_rax(value_dest, ValueTag::Bool.as_u8());
            }
            VecKind::Int => {
                dynasm!(self.ops
                    ; .arch x64
                    ; mov rax, [rdx + rcx*8]
                );
                self.store_from_rax(value_dest, ValueTag::Int.as_u8());
            }
            VecKind::Float => {
                dynasm!(self.ops
                    ; .arch x64
                    ; movsd xmm0, [rdx + rcx*8]
                );
                self.store_xmm0_as_float(value_dest);
            }
        }

        dynasm!(self.ops
            ; .arch x64
            ; mov eax, 1
        );
        self.store_from_rax(condition_dest, ValueTag::Bool.as_u8());

        dynasm!(self.ops
            ; .arch x64
            ; jmp => done
            ; => out_of_range
        );
        self.compile_load_const(value_dest, &Value::Nil)?;
        dynasm!(self.ops
            ; .arch x64
            ; xor eax, eax
        );
        self.store_from_rax(condition_dest, ValueTag::Bool.as_u8());

        dynasm!(self.ops
            ; .arch x64
            ; => done
        );
        self.hot_rax = None;
        self.hot_xmm0 = None;
        Ok(())
    }

    /// Get length of specialized Vec<T>
    fn compile_vec_len(&mut self, vec_id: usize, dest_reg: u8) -> Result<()> {
        jit::log(|| {
            format!(
                "⚡ JIT: Specialized len of vec #{} to reg {}",
                vec_id, dest_reg
            )
        });

        let spec_value =
            self.specialized_values
                .get(&vec_id)
                .ok_or_else(|| crate::LustError::RuntimeError {
                    message: format!("Specialized vec #{} not found", vec_id),
                })?;

        let stack_offset = spec_value.stack_offset;
        dynasm!(self.ops
            ; .arch x64
            ; mov rax, [rbp + stack_offset + 8]
        );
        self.store_from_rax(dest_reg, ValueTag::Int.as_u8());

        Ok(())
    }

    /// Drop a specialized value without reboxing (cleanup for leaked specializations)
    pub(super) fn compile_drop_specialized(
        &mut self,
        specialized_id: usize,
        layout: &SpecializedLayout,
    ) -> Result<()> {
        match vec_kind(layout) {
            Some(kind) => self.compile_drop_vec(specialized_id, kind),
            None => Err(crate::LustError::RuntimeError {
                message: format!("Drop not yet implemented for layout {:?}", layout),
            }),
        }
    }

    /// Drop Vec<T> without reboxing
    fn compile_drop_vec(&mut self, vec_id: usize, kind: VecKind) -> Result<()> {
        jit::log(|| format!("🗑️  JIT: Dropping specialized {:?} vec #{}", kind, vec_id));

        let spec_value =
            self.specialized_values
                .get(&vec_id)
                .ok_or_else(|| crate::LustError::RuntimeError {
                    message: format!("Specialized vec #{} not found", vec_id),
                })?;

        let stack_offset = spec_value.stack_offset;

        unsafe extern "C" {
            fn jit_drop_vec_int(slot: *mut JitVecSlot);
            fn jit_drop_vec_bool(slot: *mut JitVecSlot);
            fn jit_drop_vec_float(slot: *mut JitVecSlot);
        }
        let drop_fn: *const () = match kind {
            VecKind::Int => jit_drop_vec_int as *const (),
            VecKind::Bool => jit_drop_vec_bool as *const (),
            VecKind::Float => jit_drop_vec_float as *const (),
        };

        dynasm!(self.ops
            ; .arch x64
            ; lea rdi, [rbp + stack_offset]
            ; mov rax, QWORD drop_fn as i64
            ; call rax
        );

        Ok(())
    }

    /// Allocate stack space for specialized values
    /// Returns the offset from rbp (negative value)
    fn allocate_specialized_stack(&mut self, _size: usize, _align: usize) -> i32 {
        let allocation_offset = self.specialized_values.len() as i32 * SPECIALIZED_SLOT_SIZE;
        SPECIALIZED_BASE_OFFSET - allocation_offset
    }
}

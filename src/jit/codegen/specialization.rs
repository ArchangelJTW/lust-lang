use super::*;
use crate::jit::trace::{Operand, SpecializedOpKind};
use crate::bytecode::value::JitVecSlot;
use crate::number::LustInt;

impl JitCompiler {
    /// Compile Unbox operation - convert Value to specialized representation
    pub(super) fn compile_unbox(
        &mut self,
        specialized_id: usize,
        source_reg: u8,
        layout: &SpecializedLayout,
    ) -> Result<()> {
        match layout {
            SpecializedLayout::Vec { element_layout, .. } => {
                // Check if it's Vec<LustInt>
                if matches!(**element_layout, SpecializedLayout::Scalar { size, .. } if size == core::mem::size_of::<LustInt>())
                {
                    self.compile_unbox_array_int(specialized_id, source_reg)?;
                } else {
                    return Err(crate::LustError::RuntimeError {
                        message: format!(
                            "Unbox not yet implemented for Vec with element type {:?}",
                            element_layout
                        ),
                    });
                }
            }
            _ => {
                return Err(crate::LustError::RuntimeError {
                    message: format!("Unbox not yet implemented for layout {:?}", layout),
                });
            }
        }
        Ok(())
    }

    /// Compile Rebox operation - convert specialized representation back to Value
    pub(super) fn compile_rebox(
        &mut self,
        dest_reg: u8,
        specialized_id: usize,
        layout: &SpecializedLayout,
    ) -> Result<()> {
        match layout {
            SpecializedLayout::Vec { element_layout, .. } => {
                // Check if it's Vec<LustInt>
                if matches!(**element_layout, SpecializedLayout::Scalar { size, .. } if size == core::mem::size_of::<LustInt>())
                {
                    self.compile_rebox_array_int(dest_reg, specialized_id)?;
                } else {
                    return Err(crate::LustError::RuntimeError {
                        message: format!(
                            "Rebox not yet implemented for Vec with element type {:?}",
                            element_layout
                        ),
                    });
                }
            }
            _ => {
                return Err(crate::LustError::RuntimeError {
                    message: format!("Rebox not yet implemented for layout {:?}", layout),
                });
            }
        }
        Ok(())
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
                        self.compile_vec_int_push(*vec_id, *value_reg)?;
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
                        self.compile_vec_int_len(*vec_id, *dest_reg)?;
                    }
                    _ => {
                        return Err(crate::LustError::RuntimeError {
                            message: "VecLen operands must be (Specialized, Register)".into(),
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

    /// Unbox Array<int> - convert Rc<RefCell<Vec<Value>>> to Vec<LustInt> on stack
    fn compile_unbox_array_int(&mut self, specialized_id: usize, source_reg: u8) -> Result<()> {
        jit::log(|| {
            format!(
                "📦 JIT: Unboxing Array<int> from reg {} to specialized #{}",
                source_reg, specialized_id
            )
        });

        // Allocate a slot for the Vec raw parts and the array reference.
        let stack_offset = self.allocate_specialized_stack(32, 8);

        // Store specialized value info
        self.specialized_values
            .insert(specialized_id, SpecializedValue { stack_offset });

        let reg_offset = (source_reg as i32) * 64;

        // jit_unbox_array_int(array_ptr, slot): the helper releases whatever
        // the slot held (an earlier unbox in an unrolled iteration) and leaves
        // it empty on failure, so the postamble's rebox is a no-op then.
        unsafe extern "C" {
            fn jit_unbox_array_int(array_value_ptr: *const Value, slot: *mut JitVecSlot) -> u8;
        }
        let unbox_fn = jit_unbox_array_int as *const ();

        dynasm!(self.ops
            ; .arch x64
            ; lea rdi, [r12 + reg_offset]
            ; lea rsi, [rbp + stack_offset]
            ; mov rax, QWORD unbox_fn as i64
            ; call rax
            ; test al, al
            ; jz >fail
        );

        Ok(())
    }

    /// Rebox Array<int> - convert Vec<LustInt> back to Rc<RefCell<Vec<Value>>>
    fn compile_rebox_array_int(&mut self, dest_reg: u8, specialized_id: usize) -> Result<()> {
        jit::log(|| {
            format!(
                "📦 JIT: Reboxing specialized #{} to Array<int> in reg {}",
                specialized_id, dest_reg
            )
        });

        let spec_value = self
            .specialized_values
            .get(&specialized_id)
            .ok_or_else(|| crate::LustError::RuntimeError {
                message: format!("Specialized value #{} not found", specialized_id),
            })?;

        let stack_offset = spec_value.stack_offset;

        // jit_rebox_array_int(slot): publishes into the array the slot was
        // unboxed from and empties the slot. `dest_reg` is where the recorder
        // last saw that array; the value itself is found through the slot.
        let _ = dest_reg;
        unsafe extern "C" {
            fn jit_rebox_array_int(slot: *mut JitVecSlot) -> u8;
        }
        let rebox_fn = jit_rebox_array_int as *const ();

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
                "⚡ JIT: Specialized push to vec #{} from reg {}",
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
        let value_offset = (value_reg as i32) * 64 + 8; // +8 to skip tag, get int value

        // Call jit_vec_int_push(vec_ptr_addr, len_addr, cap_addr, value)
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
            // Arg 1: address of vec_ptr (on stack)
            ; lea rdi, [rbp + stack_offset]

            // Arg 2: address of vec_len (on stack)
            ; lea rsi, [rbp + stack_offset + 8]

            // Arg 3: address of vec_cap (on stack)
            ; lea rdx, [rbp + stack_offset + 16]

            // Arg 4: value (LustInt from register)
            ; mov rcx, [r12 + value_offset]

            // Call helper
            ; mov rax, QWORD push_fn as i64
            ; call rax

            // Check return value
            ; test al, al
            ; jz >fail
        );

        Ok(())
    }

    /// Get length of specialized Vec<int>
    fn compile_vec_int_len(&mut self, vec_id: usize, dest_reg: u8) -> Result<()> {
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
            // Read vec_len from JIT stack
            ; mov rax, [rbp + stack_offset + 8]
        );
        self.store_from_rax(dest_reg, 2);

        Ok(())
    }

    /// Drop a specialized value without reboxing (cleanup for leaked specializations)
    pub(super) fn compile_drop_specialized(
        &mut self,
        specialized_id: usize,
        layout: &SpecializedLayout,
    ) -> Result<()> {
        match layout {
            SpecializedLayout::Vec { element_layout, .. } => {
                // Check if it's Vec<LustInt>
                if matches!(**element_layout, SpecializedLayout::Scalar { size, .. } if size == core::mem::size_of::<LustInt>())
                {
                    self.compile_drop_vec_int(specialized_id)?;
                } else {
                    return Err(crate::LustError::RuntimeError {
                        message: format!(
                            "Drop not yet implemented for Vec with element type {:?}",
                            element_layout
                        ),
                    });
                }
            }
            _ => {
                return Err(crate::LustError::RuntimeError {
                    message: format!("Drop not yet implemented for layout {:?}", layout),
                });
            }
        }
        Ok(())
    }

    /// Drop Vec<LustInt> without reboxing
    fn compile_drop_vec_int(&mut self, vec_id: usize) -> Result<()> {
        jit::log(|| format!("🗑️  JIT: Dropping specialized vec #{}", vec_id));

        let spec_value =
            self.specialized_values
                .get(&vec_id)
                .ok_or_else(|| crate::LustError::RuntimeError {
                    message: format!("Specialized vec #{} not found", vec_id),
                })?;

        let stack_offset = spec_value.stack_offset;

        unsafe extern "C" {
            fn jit_drop_vec_int(slot: *mut JitVecSlot);
        }
        let drop_fn = jit_drop_vec_int as *const ();

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
        // Stack layout after prologue:
        // [rbp - 8]: rbx, [rbp - 16]: r12, [rbp - 24]: r13, [rbp - 32]: r14, [rbp - 40]: r15
        // Allocated space: [rbp - 41] to [rbp - (40 + stack_size)]
        //
        // SPECIALIZED_BASE_OFFSET keeps every slot below the saved registers
        // Vec needs ptr, len, cap = 24 bytes
        // Each additional specialized value uses 32 bytes (24 for data + 8 padding for alignment)
        let allocation_offset = self.specialized_values.len() as i32 * SPECIALIZED_SLOT_SIZE;
        SPECIALIZED_BASE_OFFSET - allocation_offset
    }
}

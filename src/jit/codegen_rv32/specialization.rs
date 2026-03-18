// Specialization operations for RV32 JIT.
//
// Specialized Vec<LustInt> metadata is stored on the JIT stack relative to s0:
//   [s0 + stack_offset + 0]  *mut LustInt  (4 bytes, pointer)
//   [s0 + stack_offset + 4]  usize len     (4 bytes on rv32)
//   [s0 + stack_offset + 8]  usize cap     (4 bytes on rv32)
//
// All helper functions accept at most 4 args — they fit in a0-a3.

use super::*;

impl JitCompiler {
    /// Compile Unbox — convert a Value into a specialized representation.
    pub(super) fn compile_unbox(
        &mut self,
        specialized_id: usize,
        source_reg: u8,
        layout: &SpecializedLayout,
    ) -> Result<()> {
        match layout {
            SpecializedLayout::Vec { element_layout, .. } => {
                if matches!(
                    **element_layout,
                    SpecializedLayout::Scalar { size, .. } if size == core::mem::size_of::<LustInt>()
                ) {
                    self.compile_unbox_array_int(specialized_id, source_reg)
                } else {
                    Err(crate::LustError::RuntimeError {
                        message: alloc::format!(
                            "Unbox not yet implemented for Vec with element type {:?}",
                            element_layout
                        ),
                    })
                }
            }
            _ => Err(crate::LustError::RuntimeError {
                message: alloc::format!("Unbox not yet implemented for layout {:?}", layout),
            }),
        }
    }

    /// Compile Rebox — convert a specialized representation back to a Value.
    pub(super) fn compile_rebox(
        &mut self,
        dest_reg: u8,
        specialized_id: usize,
        layout: &SpecializedLayout,
    ) -> Result<()> {
        match layout {
            SpecializedLayout::Vec { element_layout, .. } => {
                if matches!(
                    **element_layout,
                    SpecializedLayout::Scalar { size, .. } if size == core::mem::size_of::<LustInt>()
                ) {
                    self.compile_rebox_array_int(dest_reg, specialized_id)
                } else {
                    Err(crate::LustError::RuntimeError {
                        message: alloc::format!(
                            "Rebox not yet implemented for Vec with element type {:?}",
                            element_layout
                        ),
                    })
                }
            }
            _ => Err(crate::LustError::RuntimeError {
                message: alloc::format!("Rebox not yet implemented for layout {:?}", layout),
            }),
        }
    }

    /// Compile a SpecializedOp.
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
                        self.compile_vec_int_push(*vec_id, *value_reg)
                    }
                    _ => Err(crate::LustError::RuntimeError {
                        message: "VecPush operands must be (Specialized, Register)".into(),
                    }),
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
                        self.compile_vec_int_len(*vec_id, *dest_reg)
                    }
                    _ => Err(crate::LustError::RuntimeError {
                        message: "VecLen operands must be (Specialized, Register)".into(),
                    }),
                }
            }
            _ => Err(crate::LustError::RuntimeError {
                message: alloc::format!("Specialized op {:?} not yet implemented", op),
            }),
        }
    }

    /// Compile DropSpecialized — drop a specialized value without reboxing.
    pub(super) fn compile_drop_specialized(
        &mut self,
        specialized_id: usize,
        layout: &SpecializedLayout,
    ) -> Result<()> {
        match layout {
            SpecializedLayout::Vec { element_layout, .. } => {
                if matches!(
                    **element_layout,
                    SpecializedLayout::Scalar { size, .. } if size == core::mem::size_of::<LustInt>()
                ) {
                    self.compile_drop_vec_int(specialized_id)
                } else {
                    Err(crate::LustError::RuntimeError {
                        message: alloc::format!(
                            "Drop not yet implemented for Vec with element type {:?}",
                            element_layout
                        ),
                    })
                }
            }
            _ => Err(crate::LustError::RuntimeError {
                message: alloc::format!("Drop not yet implemented for layout {:?}", layout),
            }),
        }
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /// Unbox Array<int> — extract the raw Vec<LustInt> parts onto the JIT stack.
    ///
    /// Calls: `jit_unbox_array_int(array_ptr, out_vec_ptr, out_len, out_cap) -> u8`
    fn compile_unbox_array_int(
        &mut self,
        specialized_id: usize,
        source_reg: u8,
    ) -> Result<()> {
        extern "C" {
            fn jit_unbox_array_int(
                array_value_ptr: *const Value,
                out_vec_ptr: *mut *mut LustInt,
                out_len: *mut usize,
                out_cap: *mut usize,
            ) -> u8;
        }

        let stack_offset = self.allocate_specialized_stack();
        self.specialized_values.insert(specialized_id, SpecializedValue { stack_offset });

        // a0 = &array_value (source_reg in register array)
        self.emit_addr_in_t2(source_reg, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a0, t2);

        // a1 = &vec_ptr  on JIT stack (s0 + stack_offset + 0)
        // a2 = &vec_len  on JIT stack (s0 + stack_offset + 4)
        // a3 = &vec_cap  on JIT stack (s0 + stack_offset + 8)
        let off0 = stack_offset;
        let off4 = stack_offset + 4;
        let off8 = stack_offset + 8;
        dynasm!(self.ops ; .arch riscv32i ; addi a1, s0, off0);
        dynasm!(self.ops ; .arch riscv32i ; addi a2, s0, off4);
        dynasm!(self.ops ; .arch riscv32i ; addi a3, s0, off8);

        self.emit_load_fn_ptr(jit_unbox_array_int as *const ());
        self.emit_call_t0();

        // a0 == 0 → failure
        let fail = self.current_fail_label();
        dynasm!(self.ops ; .arch riscv32i ; beqz a0, => fail);

        Ok(())
    }

    /// Rebox Array<int> — reconstruct a Value::Array from Vec<LustInt> parts.
    ///
    /// Calls: `jit_rebox_array_int(vec_ptr, vec_len, vec_cap, out_value_ptr) -> u8`
    fn compile_rebox_array_int(
        &mut self,
        dest_reg: u8,
        specialized_id: usize,
    ) -> Result<()> {
        extern "C" {
            fn jit_rebox_array_int(
                vec_ptr: *mut LustInt,
                vec_len: usize,
                vec_cap: usize,
                out_value_ptr: *mut Value,
            ) -> u8;
        }

        let stack_offset = self
            .specialized_values
            .get(&specialized_id)
            .ok_or_else(|| crate::LustError::RuntimeError {
                message: alloc::format!("Specialized value #{} not found", specialized_id),
            })?
            .stack_offset;

        let off0 = stack_offset;
        let off4 = stack_offset + 4;
        let off8 = stack_offset + 8;

        // a0 = vec_ptr, a1 = vec_len, a2 = vec_cap
        dynasm!(self.ops ; .arch riscv32i ; lw a0, [s0, off0]);
        dynasm!(self.ops ; .arch riscv32i ; lw a1, [s0, off4]);
        dynasm!(self.ops ; .arch riscv32i ; lw a2, [s0, off8]);

        // a3 = &dest_reg Value
        self.emit_addr_in_t2(dest_reg, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a3, t2);

        self.emit_load_fn_ptr(jit_rebox_array_int as *const ());
        self.emit_call_t0();

        let fail = self.current_fail_label();
        dynasm!(self.ops ; .arch riscv32i ; beqz a0, => fail);

        Ok(())
    }

    /// Specialized push for Vec<LustInt>.
    ///
    /// Calls: `jit_vec_int_push(vec_ptr_addr, len_addr, cap_addr, value) -> u8`
    fn compile_vec_int_push(&mut self, vec_id: usize, value_reg: u8) -> Result<()> {
        extern "C" {
            fn jit_vec_int_push(
                vec_ptr: *mut *mut LustInt,
                vec_len: *mut usize,
                vec_cap: *mut usize,
                value: LustInt,
            ) -> u8;
        }

        let stack_offset = self
            .specialized_values
            .get(&vec_id)
            .ok_or_else(|| crate::LustError::RuntimeError {
                message: alloc::format!("Specialized vec #{} not found", vec_id),
            })?
            .stack_offset;

        let off0 = stack_offset;
        let off4 = stack_offset + 4;
        let off8 = stack_offset + 8;

        // a0 = &vec_ptr, a1 = &vec_len, a2 = &vec_cap
        dynasm!(self.ops ; .arch riscv32i ; addi a0, s0, off0);
        dynasm!(self.ops ; .arch riscv32i ; addi a1, s0, off4);
        dynasm!(self.ops ; .arch riscv32i ; addi a2, s0, off8);

        // a3 = LustInt value from register data field
        self.emit_addr_in_t2(value_reg, VALUE_DATA_OFFSET);
        dynasm!(self.ops ; .arch riscv32i ; lw a3, [t2, 0]);

        self.emit_load_fn_ptr(jit_vec_int_push as *const ());
        self.emit_call_t0();

        let fail = self.current_fail_label();
        dynasm!(self.ops ; .arch riscv32i ; beqz a0, => fail);

        Ok(())
    }

    /// Get the length of a specialized Vec<int>, storing it as Value::Int.
    fn compile_vec_int_len(&mut self, vec_id: usize, dest_reg: u8) -> Result<()> {
        let stack_offset = self
            .specialized_values
            .get(&vec_id)
            .ok_or_else(|| crate::LustError::RuntimeError {
                message: alloc::format!("Specialized vec #{} not found", vec_id),
            })?
            .stack_offset;

        let len_off = stack_offset + 4;

        // Read len from JIT stack, write as Value::Int into dest_reg.
        // Value layout: tag (u8) at +0, data (i32) at +VALUE_DATA_OFFSET.
        dynasm!(self.ops ; .arch riscv32i ; lw t0, [s0, len_off]);
        self.emit_addr_in_t2(dest_reg, 0);
        dynasm!(self.ops
            ; .arch riscv32i
            ; li t1, 2              // ValueTag::Int
            ; sb t1, [t2, 0]
            ; sw t0, [t2, VALUE_DATA_OFFSET]
        );

        Ok(())
    }

    /// Drop a Vec<LustInt> without reboxing it (cleanup on bailout).
    ///
    /// Calls: `jit_drop_vec_int(vec_ptr, vec_len, vec_cap)`
    fn compile_drop_vec_int(&mut self, vec_id: usize) -> Result<()> {
        extern "C" {
            fn jit_drop_vec_int(vec_ptr: *mut LustInt, vec_len: usize, vec_cap: usize);
        }

        let stack_offset = self
            .specialized_values
            .get(&vec_id)
            .ok_or_else(|| crate::LustError::RuntimeError {
                message: alloc::format!("Specialized vec #{} not found", vec_id),
            })?
            .stack_offset;

        let off0 = stack_offset;
        let off4 = stack_offset + 4;
        let off8 = stack_offset + 8;

        dynasm!(self.ops ; .arch riscv32i ; lw a0, [s0, off0]);
        dynasm!(self.ops ; .arch riscv32i ; lw a1, [s0, off4]);
        dynasm!(self.ops ; .arch riscv32i ; lw a2, [s0, off8]);

        self.emit_load_fn_ptr(jit_drop_vec_int as *const ());
        self.emit_call_t0();

        Ok(())
    }

    /// Allocate a specialized slot on the JIT stack.
    ///
    /// Returns the offset from `s0` (negative).  Each call bumps down by
    /// `SPECIALIZED_SLOT_SIZE` bytes.  The MIN_JIT_STACK_SIZE in the frame
    /// must be large enough to hold all anticipated slots.
    fn allocate_specialized_stack(&self) -> i32 {
        let n = self.specialized_values.len() as i32;
        SPECIALIZED_BASE_OFFSET - n * SPECIALIZED_SLOT_SIZE
    }
}

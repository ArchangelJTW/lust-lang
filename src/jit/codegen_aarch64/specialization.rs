use super::*;
use crate::jit::trace::{Operand, SpecializedOpKind};
use crate::bytecode::value::JitVecSlot;
use crate::number::LustInt;

fn is_int_vec(layout: &SpecializedLayout) -> Option<bool> {
    match layout {
        SpecializedLayout::Vec { element_layout, .. } => Some(matches!(
            **element_layout,
            SpecializedLayout::Scalar { size, .. } if size == core::mem::size_of::<LustInt>()
        )),
        _ => None,
    }
}

impl JitCompiler {
    fn slot_offset(&self, specialized_id: usize, what: &str) -> Result<i32> {
        self.specialized_values
            .get(&specialized_id)
            .map(|value| value.stack_offset)
            .ok_or_else(|| crate::LustError::RuntimeError {
                message: format!("Specialized {} #{} not found", what, specialized_id),
            })
    }

    /// Compile Unbox operation - convert Value to specialized representation
    pub(super) fn compile_unbox(
        &mut self,
        specialized_id: usize,
        source_reg: u8,
        layout: &SpecializedLayout,
    ) -> Result<()> {
        match is_int_vec(layout) {
            Some(true) => self.compile_unbox_array_int(specialized_id, source_reg),
            Some(false) => Err(crate::LustError::RuntimeError {
                message: format!("Unbox not yet implemented for Vec with layout {:?}", layout),
            }),
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
        match is_int_vec(layout) {
            Some(true) => self.compile_rebox_array_int(dest_reg, specialized_id),
            Some(false) => Err(crate::LustError::RuntimeError {
                message: format!("Rebox not yet implemented for Vec with layout {:?}", layout),
            }),
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
        self.specialized_values
            .insert(specialized_id, SpecializedValue { stack_offset });

        // jit_unbox_array_int(array_ptr, slot): the helper releases whatever
        // the slot held (an earlier unbox in an unrolled iteration) and leaves
        // it empty on failure, so the postamble's rebox is a no-op then.
        unsafe extern "C" {
            fn jit_unbox_array_int(array_value_ptr: *const Value, slot: *mut JitVecSlot) -> u8;
        }
        self.emit_reg_addr(0, source_reg);
        self.emit_slot_addr(1, stack_offset);
        self.emit_call(jit_unbox_array_int as *const ());
        self.emit_fail_if_w0_zero();

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

        let stack_offset = self.slot_offset(specialized_id, "value")?;

        // jit_rebox_array_int(slot): publishes into the array the slot was
        // unboxed from and empties the slot. `dest_reg` is where the recorder
        // last saw that array; the value itself is found through the slot.
        let _ = dest_reg;
        unsafe extern "C" {
            fn jit_rebox_array_int(slot: *mut JitVecSlot) -> u8;
        }
        self.emit_slot_addr(0, stack_offset);
        self.emit_call(jit_rebox_array_int as *const ());
        self.emit_fail_if_w0_zero();

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

        let stack_offset = self.slot_offset(vec_id, "vec")?;

        // Call jit_vec_int_push(vec_ptr_addr, len_addr, cap_addr, value)
        unsafe extern "C" {
            fn jit_vec_int_push(
                vec_ptr: *mut *mut LustInt,
                vec_len: *mut usize,
                vec_cap: *mut usize,
                value: LustInt,
            ) -> u8;
        }

        self.emit_slot_addr(0, stack_offset);
        dynasm!(self.ops
            ; .arch aarch64
            ; add x1, x0, 8
            ; add x2, x0, 16
        );
        self.load_payload(3, value_reg);
        self.emit_call(jit_vec_int_push as *const ());
        self.emit_fail_if_w0_zero();

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

        let stack_offset = self.slot_offset(vec_id, "vec")?;
        self.emit_slot_addr(11, stack_offset);
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x0, [x11, 8]
        );
        self.store_from_x0(dest_reg, ValueTag::Int.as_u8());

        Ok(())
    }

    /// Drop a specialized value without reboxing (cleanup for leaked specializations)
    pub(super) fn compile_drop_specialized(
        &mut self,
        specialized_id: usize,
        layout: &SpecializedLayout,
    ) -> Result<()> {
        match is_int_vec(layout) {
            Some(true) => self.compile_drop_vec_int(specialized_id),
            Some(false) => Err(crate::LustError::RuntimeError {
                message: format!("Drop not yet implemented for Vec with layout {:?}", layout),
            }),
            None => Err(crate::LustError::RuntimeError {
                message: format!("Drop not yet implemented for layout {:?}", layout),
            }),
        }
    }

    /// Drop Vec<LustInt> without reboxing
    fn compile_drop_vec_int(&mut self, vec_id: usize) -> Result<()> {
        jit::log(|| format!("🗑️  JIT: Dropping specialized vec #{}", vec_id));

        let stack_offset = self.slot_offset(vec_id, "vec")?;

        unsafe extern "C" {
            fn jit_drop_vec_int(slot: *mut JitVecSlot);
        }
        self.emit_slot_addr(0, stack_offset);
        self.emit_call(jit_drop_vec_int as *const ());

        Ok(())
    }

    /// Allocate stack space for specialized values.
    /// Returns the offset from x29 (negative value).
    fn allocate_specialized_stack(&mut self, _size: usize, _align: usize) -> i32 {
        // Slots grow downward from SPECIALIZED_BASE_OFFSET below the frame
        // pointer; each is SPECIALIZED_SLOT_SIZE bytes (24 used + padding).
        let allocation_offset = self.specialized_values.len() as i32 * SPECIALIZED_SLOT_SIZE;
        SPECIALIZED_BASE_OFFSET - allocation_offset
    }
}

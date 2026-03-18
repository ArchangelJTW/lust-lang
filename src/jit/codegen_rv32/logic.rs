use super::*;

impl JitCompiler {
    pub(super) fn compile_and(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        self.emit_addr_in_t2(lhs, 0);
        dynasm!(self.ops
            ; .arch riscv32i
            ; lbu t0, [t2, 0]           // lhs tag
            ; beqz t0, >false_result    // Nil → false
            ; li t1, 1
            ; bne t0, t1, >lhs_truthy   // non-Bool → truthy
            ; lw t0, [t2, VALUE_DATA_OFFSET]
            ; beqz t0, >false_result    // Bool(false) → false
            ; lhs_truthy:
        );
        self.emit_addr_in_t2(rhs, 0);
        dynasm!(self.ops
            ; .arch riscv32i
            ; lbu t0, [t2, 0]           // rhs tag
            ; beqz t0, >false_result    // Nil → false
            ; li t1, 1
            ; bne t0, t1, >true_result
            ; lw t0, [t2, VALUE_DATA_OFFSET]
            ; beqz t0, >false_result
            ; true_result:
        );
        self.emit_addr_in_t2(dest, 0);
        dynasm!(self.ops
            ; .arch riscv32i
            ; li t0, 1
            ; sb t0, [t2, 0]            // tag = Bool
            ; li t0, 1
            ; sw t0, [t2, VALUE_DATA_OFFSET]
            ; j >done
            ; false_result:
        );
        self.emit_addr_in_t2(dest, 0);
        dynasm!(self.ops
            ; .arch riscv32i
            ; li t0, 1
            ; sb t0, [t2, 0]
            ; sw zero, [t2, VALUE_DATA_OFFSET]
            ; done:
        );
        Ok(())
    }

    pub(super) fn compile_or(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        self.emit_addr_in_t2(lhs, 0);
        dynasm!(self.ops
            ; .arch riscv32i
            ; lbu t0, [t2, 0]
            ; beqz t0, >check_rhs
            ; li t1, 1
            ; bne t0, t1, >true_result
            ; lw t0, [t2, VALUE_DATA_OFFSET]
            ; bnez t0, >true_result
            ; check_rhs:
        );
        self.emit_addr_in_t2(rhs, 0);
        dynasm!(self.ops
            ; .arch riscv32i
            ; lbu t0, [t2, 0]
            ; beqz t0, >false_result
            ; li t1, 1
            ; bne t0, t1, >true_result
            ; lw t0, [t2, VALUE_DATA_OFFSET]
            ; bnez t0, >true_result
            ; false_result:
        );
        self.emit_addr_in_t2(dest, 0);
        dynasm!(self.ops
            ; .arch riscv32i
            ; li t0, 1
            ; sb t0, [t2, 0]
            ; sw zero, [t2, VALUE_DATA_OFFSET]
            ; j >done
            ; true_result:
        );
        self.emit_addr_in_t2(dest, 0);
        dynasm!(self.ops
            ; .arch riscv32i
            ; li t0, 1
            ; sb t0, [t2, 0]
            ; li t0, 1
            ; sw t0, [t2, VALUE_DATA_OFFSET]
            ; done:
        );
        Ok(())
    }

    pub(super) fn compile_not(&mut self, dest: u8, src: u8) -> Result<()> {
        self.emit_addr_in_t2(src, 0);
        dynasm!(self.ops
            ; .arch riscv32i
            ; lbu t0, [t2, 0]
            ; beqz t0, >true_result     // Nil → true
            ; li t1, 1
            ; bne t0, t1, >false_result // non-Bool → falsy → result true? No, non-nil/non-false is truthy → NOT = false
            ; lw t0, [t2, VALUE_DATA_OFFSET]
            ; beqz t0, >true_result     // Bool(false) → NOT = true
            ; false_result:
        );
        self.emit_addr_in_t2(dest, 0);
        dynasm!(self.ops
            ; .arch riscv32i
            ; li t0, 1
            ; sb t0, [t2, 0]
            ; sw zero, [t2, VALUE_DATA_OFFSET]
            ; j >done
            ; true_result:
        );
        self.emit_addr_in_t2(dest, 0);
        dynasm!(self.ops
            ; .arch riscv32i
            ; li t0, 1
            ; sb t0, [t2, 0]
            ; li t0, 1
            ; sw t0, [t2, VALUE_DATA_OFFSET]
            ; done:
        );
        Ok(())
    }

    pub(super) fn compile_concat(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        extern "C" {
            fn jit_concat_safe(
                vm_ptr: *mut crate::VM,
                left: *const Value,
                right: *const Value,
                out: *mut Value,
            ) -> u8;
        }
        // a0 = vm_ptr, a1 = &lhs, a2 = &rhs, a3 = &dest
        dynasm!(self.ops ; .arch riscv32i ; mv a0, s3);
        self.emit_addr_in_t2(lhs, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a1, t2);
        self.emit_addr_in_t2(rhs, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a2, t2);
        self.emit_addr_in_t2(dest, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a3, t2);
        self.emit_load_fn_ptr(jit_concat_safe as *const ());
        self.emit_call_t0();
        dynasm!(self.ops ; .arch riscv32i ; beqz a0, >fail);
        Ok(())
    }
}

// Comparison operations for RV32 JIT.
//
// RISC-V has no flags register; comparisons use `slt`/`sltu` and branches.
// All comparisons here operate on integer Values (LustInt = i32 under no_std).
// Results are stored as Bool (tag=1) with data=0 (false) or data=1 (true).

use super::*;

impl JitCompiler {
    pub(super) fn compile_lt(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        self.load_data_to_t0(lhs);
        self.load_data_to_t1(rhs);
        dynasm!(self.ops
            ; .arch riscv32i
            ; slt t0, t0, t1    // t0 = (lhs < rhs) ? 1 : 0  (signed)
        );
        self.store_t0_as_bool(dest);
        Ok(())
    }

    pub(super) fn compile_le(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        self.load_data_to_t0(lhs);
        self.load_data_to_t1(rhs);
        dynasm!(self.ops
            ; .arch riscv32i
            ; slt t0, t1, t0    // t0 = (rhs < lhs) ? 1 : 0
            ; xori t0, t0, 1    // t0 = NOT (lhs > rhs)  ≡  lhs <= rhs
        );
        self.store_t0_as_bool(dest);
        Ok(())
    }

    pub(super) fn compile_gt(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        self.load_data_to_t0(lhs);
        self.load_data_to_t1(rhs);
        dynasm!(self.ops
            ; .arch riscv32i
            ; slt t0, t1, t0    // t0 = (rhs < lhs) ? 1 : 0  ≡  lhs > rhs
        );
        self.store_t0_as_bool(dest);
        Ok(())
    }

    pub(super) fn compile_ge(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        self.load_data_to_t0(lhs);
        self.load_data_to_t1(rhs);
        dynasm!(self.ops
            ; .arch riscv32i
            ; slt t0, t0, t1    // t0 = (lhs < rhs) ? 1 : 0
            ; xori t0, t0, 1    // NOT → lhs >= rhs
        );
        self.store_t0_as_bool(dest);
        Ok(())
    }

    pub(super) fn compile_eq(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        self.load_data_to_t0(lhs);
        self.load_data_to_t1(rhs);
        dynasm!(self.ops
            ; .arch riscv32i
            ; xor t0, t0, t1
            ; sltiu t0, t0, 1   // t0 = (t0 == 0) ? 1 : 0
        );
        self.store_t0_as_bool(dest);
        Ok(())
    }

    pub(super) fn compile_ne(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        self.load_data_to_t0(lhs);
        self.load_data_to_t1(rhs);
        dynasm!(self.ops
            ; .arch riscv32i
            ; xor t0, t0, t1
            ; sltu t0, zero, t0 // t0 = (t0 != 0) ? 1 : 0
        );
        self.store_t0_as_bool(dest);
        Ok(())
    }

    /// Store `t0` as a Bool (tag=1, data=t0) into `regs[vm_reg]`.
    pub(super) fn store_t0_as_bool(&mut self, vm_reg: u8) {
        self.emit_addr_in_t2(vm_reg, 0);
        dynasm!(self.ops
            ; .arch riscv32i
            ; li t1, 1          // ValueTag::Bool
            ; sb t1, [t2, 0]
            ; sw t0, [t2, VALUE_DATA_OFFSET]
        );
    }
}

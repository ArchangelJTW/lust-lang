use super::*;

impl JitCompiler {
    /// Emit instructions that leave the address of `Value[vm_reg] + extra` in `t2`.
    /// Uses `li t2, offset; add t2, s2, t2` to handle offsets > 2047.
    pub(super) fn emit_addr_in_t2(&mut self, vm_reg: u8, extra: i32) {
        let offset = (vm_reg as i32) * (mem::size_of::<Value>() as i32) + extra;
        dynasm!(self.ops
            ; .arch riscv32i
            ; li t2, offset
            ; add t2, s2, t2
        );
    }

    /// Load the discriminant byte of `regs[vm_reg]` into `t0` (zero-extended).
    pub(super) fn load_tag_to_t0(&mut self, vm_reg: u8) {
        self.emit_addr_in_t2(vm_reg, 0);
        dynasm!(self.ops
            ; .arch riscv32i
            ; lbu t0, [t2, 0]
        );
    }

    /// Load the 32-bit integer/float data field of `regs[vm_reg]` into `t0`.
    pub(super) fn load_data_to_t0(&mut self, vm_reg: u8) {
        self.emit_addr_in_t2(vm_reg, VALUE_DATA_OFFSET);
        dynasm!(self.ops
            ; .arch riscv32i
            ; lw t0, [t2, 0]
        );
    }

    /// Load the 32-bit data field of `regs[vm_reg]` into `t1`.
    pub(super) fn load_data_to_t1(&mut self, vm_reg: u8) {
        self.emit_addr_in_t2(vm_reg, VALUE_DATA_OFFSET);
        dynasm!(self.ops
            ; .arch riscv32i
            ; lw t1, [t2, 0]
        );
    }

    /// Store `t0` as an integer value (tag=Int=2, data=t0) into `regs[vm_reg]`.
    pub(super) fn store_t0_as_int(&mut self, vm_reg: u8) {
        self.emit_addr_in_t2(vm_reg, 0);
        dynasm!(self.ops
            ; .arch riscv32i
            ; li t1, 2          // ValueTag::Int
            ; sb t1, [t2, 0]
            ; sw t0, [t2, VALUE_DATA_OFFSET]
        );
    }

    /// Store `t0` as a float value (tag=Float=3, data stored via fmv) into `regs[vm_reg]`.
    /// `ft0` must hold the f32 value to store; `t0` is used as a temp.
    pub(super) fn store_ft0_as_float(&mut self, vm_reg: u8) {
        self.emit_addr_in_t2(vm_reg, 0);
        dynasm!(self.ops
            ; .arch riscv32i ; .feature f
            ; li t0, 3          // ValueTag::Float
            ; sb t0, [t2, 0]
            ; fsw ft0, [t2, VALUE_DATA_OFFSET]
        );
    }

    /// Load the f32 data field of `regs[vm_reg]` into `ft0`.
    pub(super) fn load_float_to_ft0(&mut self, vm_reg: u8) {
        self.emit_addr_in_t2(vm_reg, VALUE_DATA_OFFSET);
        dynasm!(self.ops
            ; .arch riscv32i ; .feature f
            ; flw ft0, [t2, 0]
        );
    }

    /// Load the f32 data field of `regs[vm_reg]` into `ft1`.
    pub(super) fn load_float_to_ft1(&mut self, vm_reg: u8) {
        self.emit_addr_in_t2(vm_reg, VALUE_DATA_OFFSET);
        dynasm!(self.ops
            ; .arch riscv32i ; .feature f
            ; flw ft1, [t2, 0]
        );
    }

    /// Emit an indirect call through the address currently in `t0`.
    /// Clobbers `ra`.
    pub(super) fn emit_call_t0(&mut self) {
        dynasm!(self.ops
            ; .arch riscv32i
            ; jalr ra, t0, 0
        );
    }

    /// Load a 32-bit function pointer into `t0`.
    pub(super) fn emit_load_fn_ptr(&mut self, fn_ptr: *const ()) {
        let bits = fn_ptr as u32 as i32;
        dynasm!(self.ops
            ; .arch riscv32i
            ; li t0, bits
        );
    }
}

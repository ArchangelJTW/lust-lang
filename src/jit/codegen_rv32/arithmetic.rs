use super::*;

impl JitCompiler {
    pub(super) fn compile_add_specialized(
        &mut self,
        dest: u8,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> Result<()> {
        if lhs_type == ValueType::Int && rhs_type == ValueType::Int {
            self.load_data_to_t0(lhs);
            self.load_data_to_t1(rhs);
            dynasm!(self.ops ; .arch riscv32i ; .feature mf ; add t0, t0, t1);
            self.store_t0_as_int(dest);
            return Ok(());
        }

        if lhs_type == ValueType::Float && rhs_type == ValueType::Float {
            self.load_float_to_ft0(lhs);
            self.load_float_to_ft1(rhs);
            dynasm!(self.ops ; .arch riscv32i ; .feature mf ; fadd.s ft0, ft0, ft1);
            self.store_ft0_as_float(dest);
            return Ok(());
        }

        if lhs_type == ValueType::Int && rhs_type == ValueType::Float {
            self.load_data_to_t0(lhs);
            self.load_float_to_ft1(rhs);
            dynasm!(self.ops
                ; .arch riscv32i ; .feature mf
                ; fcvt.s.w ft0, t0
                ; fadd.s ft0, ft0, ft1
            );
            self.store_ft0_as_float(dest);
            return Ok(());
        }

        if lhs_type == ValueType::Float && rhs_type == ValueType::Int {
            self.load_float_to_ft0(lhs);
            self.load_data_to_t0(rhs);
            dynasm!(self.ops
                ; .arch riscv32i ; .feature mf
                ; fcvt.s.w ft1, t0
                ; fadd.s ft0, ft0, ft1
            );
            self.store_ft0_as_float(dest);
            return Ok(());
        }

        self.compile_add(dest, lhs, rhs)
    }

    pub(super) fn compile_add(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        let fail_label = self.current_fail_label();
        self.load_tag_to_t0(lhs);
        dynasm!(self.ops
            ; .arch riscv32i ; .feature mf
            ; li t1, 3              // Float tag
            ; beq t0, t1, >float_path
        );
        self.load_tag_to_t0(rhs);
        dynasm!(self.ops
            ; .arch riscv32i ; .feature mf
            ; beq t0, t1, >float_path
        );
        // Int + Int
        self.load_data_to_t0(lhs);
        self.load_data_to_t1(rhs);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; add t0, t0, t1);
        self.store_t0_as_int(dest);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; j >done);
        // Float path
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; float_path:);
        self.load_tag_to_t0(lhs);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; li t1, 3);
        let fail_bits = fail_label;
        dynasm!(self.ops
            ; .arch riscv32i ; .feature mf
            ; beq t0, t1, >lhs_is_float
            ; li t1, 2
            ; bne t0, t1, => fail_bits   // neither float nor int → bail
            ; fcvt.s.w ft0, t0
            ; j >lhs_done
            ; lhs_is_float:
        );
        self.load_float_to_ft0(lhs);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; j >lhs_done ; lhs_done:);
        self.load_tag_to_t0(rhs);
        dynasm!(self.ops
            ; .arch riscv32i ; .feature mf
            ; li t1, 3
            ; beq t0, t1, >rhs_is_float
            ; li t1, 2
            ; bne t0, t1, => fail_bits
        );
        self.load_data_to_t0(rhs);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; fcvt.s.w ft1, t0 ; j >rhs_done ; rhs_is_float:);
        self.load_float_to_ft1(rhs);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; rhs_done:);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; fadd.s ft0, ft0, ft1);
        self.store_ft0_as_float(dest);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; done:);
        Ok(())
    }

    pub(super) fn compile_sub_specialized(
        &mut self,
        dest: u8,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> Result<()> {
        if lhs_type == ValueType::Int && rhs_type == ValueType::Int {
            self.load_data_to_t0(lhs);
            self.load_data_to_t1(rhs);
            dynasm!(self.ops ; .arch riscv32i ; .feature mf ; sub t0, t0, t1);
            self.store_t0_as_int(dest);
            return Ok(());
        }
        if lhs_type == ValueType::Float && rhs_type == ValueType::Float {
            self.load_float_to_ft0(lhs);
            self.load_float_to_ft1(rhs);
            dynasm!(self.ops ; .arch riscv32i ; .feature mf ; fsub.s ft0, ft0, ft1);
            self.store_ft0_as_float(dest);
            return Ok(());
        }
        self.compile_sub(dest, lhs, rhs)
    }

    pub(super) fn compile_sub(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        self.load_data_to_t0(lhs);
        self.load_data_to_t1(rhs);
        self.load_tag_to_t0(lhs); // reuse t0 for tag check
        dynasm!(self.ops
            ; .arch riscv32i ; .feature mf
            ; li t1, 3
            ; beq t0, t1, >float_path
        );
        self.load_data_to_t0(lhs);
        self.load_data_to_t1(rhs);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; sub t0, t0, t1);
        self.store_t0_as_int(dest);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; j >done ; float_path:);
        self.load_float_to_ft0(lhs);
        self.load_float_to_ft1(rhs);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; fsub.s ft0, ft0, ft1);
        self.store_ft0_as_float(dest);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; done:);
        Ok(())
    }

    pub(super) fn compile_mul_specialized(
        &mut self,
        dest: u8,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> Result<()> {
        if lhs_type == ValueType::Int && rhs_type == ValueType::Int {
            self.load_data_to_t0(lhs);
            self.load_data_to_t1(rhs);
            dynasm!(self.ops ; .arch riscv32i ; .feature mf ; mul t0, t0, t1);
            self.store_t0_as_int(dest);
            return Ok(());
        }
        if lhs_type == ValueType::Float && rhs_type == ValueType::Float {
            self.load_float_to_ft0(lhs);
            self.load_float_to_ft1(rhs);
            dynasm!(self.ops ; .arch riscv32i ; .feature mf ; fmul.s ft0, ft0, ft1);
            self.store_ft0_as_float(dest);
            return Ok(());
        }
        self.compile_mul(dest, lhs, rhs)
    }

    pub(super) fn compile_mul(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        self.load_tag_to_t0(lhs);
        dynasm!(self.ops
            ; .arch riscv32i ; .feature mf
            ; li t1, 3
            ; beq t0, t1, >float_path
        );
        self.load_data_to_t0(lhs);
        self.load_data_to_t1(rhs);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; mul t0, t0, t1);
        self.store_t0_as_int(dest);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; j >done ; float_path:);
        self.load_float_to_ft0(lhs);
        self.load_float_to_ft1(rhs);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; fmul.s ft0, ft0, ft1);
        self.store_ft0_as_float(dest);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; done:);
        Ok(())
    }

    pub(super) fn compile_div_specialized(
        &mut self,
        dest: u8,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> Result<()> {
        if lhs_type == ValueType::Int && rhs_type == ValueType::Int {
            self.load_data_to_t0(lhs);
            self.load_data_to_t1(rhs);
            dynasm!(self.ops ; .arch riscv32i ; .feature mf ; div t0, t0, t1);
            self.store_t0_as_int(dest);
            return Ok(());
        }
        if lhs_type == ValueType::Float && rhs_type == ValueType::Float {
            self.load_float_to_ft0(lhs);
            self.load_float_to_ft1(rhs);
            dynasm!(self.ops ; .arch riscv32i ; .feature mf ; fdiv.s ft0, ft0, ft1);
            self.store_ft0_as_float(dest);
            return Ok(());
        }
        self.compile_div(dest, lhs, rhs)
    }

    pub(super) fn compile_div(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        self.load_tag_to_t0(lhs);
        dynasm!(self.ops
            ; .arch riscv32i ; .feature mf
            ; li t1, 3
            ; beq t0, t1, >float_path
        );
        self.load_data_to_t0(lhs);
        self.load_data_to_t1(rhs);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; div t0, t0, t1);
        self.store_t0_as_int(dest);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; j >done ; float_path:);
        self.load_float_to_ft0(lhs);
        self.load_float_to_ft1(rhs);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; fdiv.s ft0, ft0, ft1);
        self.store_ft0_as_float(dest);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; done:);
        Ok(())
    }

    pub(super) fn compile_mod_specialized(
        &mut self,
        dest: u8,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> Result<()> {
        if lhs_type == ValueType::Int && rhs_type == ValueType::Int {
            self.load_data_to_t0(lhs);
            self.load_data_to_t1(rhs);
            dynasm!(self.ops ; .arch riscv32i ; .feature mf ; rem t0, t0, t1);
            self.store_t0_as_int(dest);
            return Ok(());
        }
        self.compile_mod(dest, lhs, rhs)
    }

    pub(super) fn compile_mod(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        self.load_data_to_t0(lhs);
        self.load_data_to_t1(rhs);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; rem t0, t0, t1);
        self.store_t0_as_int(dest);
        Ok(())
    }

    pub(super) fn compile_neg(&mut self, dest: u8, src: u8) -> Result<()> {
        self.load_tag_to_t0(src);
        dynasm!(self.ops
            ; .arch riscv32i ; .feature mf
            ; li t1, 3
            ; beq t0, t1, >float_path
        );
        self.load_data_to_t0(src);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; neg t0, t0);
        self.store_t0_as_int(dest);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; j >done ; float_path:);
        self.load_float_to_ft0(src);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; fneg.s ft0, ft0);
        self.store_ft0_as_float(dest);
        dynasm!(self.ops ; .arch riscv32i ; .feature mf ; done:);
        Ok(())
    }
}

use super::*;

impl JitCompiler {
    /// Load comparison operands. Returns true when they were loaded into
    /// d0/d1 (float compare), false when loaded into x0/x10 (int compare).
    pub(super) fn load_numeric_comparison_operands(
        &mut self,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> bool {
        if lhs_type == ValueType::Int && rhs_type == ValueType::Int {
            self.load_payload(0, lhs);
            self.load_payload(10, rhs);
            return false;
        }

        if lhs_type == ValueType::Int {
            self.load_payload_int_as_f(0, lhs);
        } else {
            self.load_payload_f(0, lhs);
        }
        if rhs_type == ValueType::Int {
            self.load_payload_int_as_f(1, rhs);
        } else {
            self.load_payload_f(1, rhs);
        }
        true
    }

    // Ordered float comparisons must be false when either operand is NaN.
    // After `fcmp`, an unordered result sets C and V (and clears N, Z), so:
    //   mi (N=1)            – less than, false for NaN
    //   ls (C=0 || Z=1)     – less or equal, false for NaN
    //   gt (Z=0 && N==V)    – greater, false for NaN
    //   ge (N==V)           – greater or equal, false for NaN
    //   eq / ne             – NaN is never equal, always not-equal

    pub(super) fn compile_lt(
        &mut self,
        dest: u8,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> Result<()> {
        if self.load_numeric_comparison_operands(lhs, rhs, lhs_type, rhs_type) {
            dynasm!(self.ops ; .arch aarch64 ; fcmp d0, d1 ; cset x0, mi);
        } else {
            dynasm!(self.ops ; .arch aarch64 ; cmp x0, x10 ; cset x0, lt);
        }
        self.store_from_x0(dest, ValueTag::Bool.as_u8());
        Ok(())
    }

    pub(super) fn compile_le(
        &mut self,
        dest: u8,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> Result<()> {
        if self.load_numeric_comparison_operands(lhs, rhs, lhs_type, rhs_type) {
            dynasm!(self.ops ; .arch aarch64 ; fcmp d0, d1 ; cset x0, ls);
        } else {
            dynasm!(self.ops ; .arch aarch64 ; cmp x0, x10 ; cset x0, le);
        }
        self.store_from_x0(dest, ValueTag::Bool.as_u8());
        Ok(())
    }

    pub(super) fn compile_gt(
        &mut self,
        dest: u8,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> Result<()> {
        if self.load_numeric_comparison_operands(lhs, rhs, lhs_type, rhs_type) {
            dynasm!(self.ops ; .arch aarch64 ; fcmp d0, d1 ; cset x0, gt);
        } else {
            dynasm!(self.ops ; .arch aarch64 ; cmp x0, x10 ; cset x0, gt);
        }
        self.store_from_x0(dest, ValueTag::Bool.as_u8());
        Ok(())
    }

    pub(super) fn compile_ge(
        &mut self,
        dest: u8,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> Result<()> {
        if self.load_numeric_comparison_operands(lhs, rhs, lhs_type, rhs_type) {
            dynasm!(self.ops ; .arch aarch64 ; fcmp d0, d1 ; cset x0, ge);
        } else {
            dynasm!(self.ops ; .arch aarch64 ; cmp x0, x10 ; cset x0, ge);
        }
        self.store_from_x0(dest, ValueTag::Bool.as_u8());
        Ok(())
    }

    pub(super) fn compile_eq(
        &mut self,
        dest: u8,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> Result<()> {
        if lhs_type != rhs_type {
            dynasm!(self.ops ; .arch aarch64 ; mov x0, xzr);
        } else if lhs_type == ValueType::Float {
            self.load_numeric_comparison_operands(lhs, rhs, lhs_type, rhs_type);
            dynasm!(self.ops ; .arch aarch64 ; fcmp d0, d1 ; cset x0, eq);
        } else if lhs_type == ValueType::Bool {
            self.load_bool_payload(0, lhs);
            self.load_bool_payload(10, rhs);
            dynasm!(self.ops ; .arch aarch64 ; cmp w0, w10 ; cset x0, eq);
        } else {
            self.load_payload(0, lhs);
            self.load_payload(10, rhs);
            dynasm!(self.ops ; .arch aarch64 ; cmp x0, x10 ; cset x0, eq);
        }
        self.store_from_x0(dest, ValueTag::Bool.as_u8());
        Ok(())
    }

    pub(super) fn compile_ne(
        &mut self,
        dest: u8,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> Result<()> {
        if lhs_type != rhs_type {
            dynasm!(self.ops ; .arch aarch64 ; movz x0, 1);
        } else if lhs_type == ValueType::Float {
            self.load_numeric_comparison_operands(lhs, rhs, lhs_type, rhs_type);
            dynasm!(self.ops ; .arch aarch64 ; fcmp d0, d1 ; cset x0, ne);
        } else if lhs_type == ValueType::Bool {
            self.load_bool_payload(0, lhs);
            self.load_bool_payload(10, rhs);
            dynasm!(self.ops ; .arch aarch64 ; cmp w0, w10 ; cset x0, ne);
        } else {
            self.load_payload(0, lhs);
            self.load_payload(10, rhs);
            dynasm!(self.ops ; .arch aarch64 ; cmp x0, x10 ; cset x0, ne);
        }
        self.store_from_x0(dest, ValueTag::Bool.as_u8());
        Ok(())
    }
}

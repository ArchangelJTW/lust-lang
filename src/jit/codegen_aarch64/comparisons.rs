use super::*;

/// Where a comparison's operands ended up.
#[derive(Clone, Copy)]
pub(super) enum Compare {
    /// Signed integer compare of X(a) with X(b).
    Int(u8, u8),
    /// Float compare of D(a) with D(b).
    Float(u8, u8),
}

impl JitCompiler {
    /// Place comparison operands: pinned registers are used in place, others
    /// load into x0/x10 (ints) or d0/d1 (floats, ints converted).
    pub(super) fn compare_operands(
        &mut self,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> Compare {
        if lhs_type == ValueType::Int && rhs_type == ValueType::Int {
            let (a, b) = self.operand_pair_x(lhs, rhs, 0, 10);
            return Compare::Int(a, b);
        }
        let (a, b) = self.operand_pair_numeric_d(lhs, rhs, lhs_type, rhs_type);
        Compare::Float(a, b)
    }

    /// Emit the `cmp`/`fcmp` for placed operands.
    pub(super) fn emit_compare(&mut self, cmp: Compare) {
        match cmp {
            Compare::Int(a, b) => dynasm!(self.ops ; .arch aarch64 ; cmp X(a), X(b)),
            Compare::Float(a, b) => dynasm!(self.ops ; .arch aarch64 ; fcmp D(a), D(b)),
        }
    }

    /// Materialize a condition as a Bool into registers[dest]: straight into
    /// a carried pin, otherwise via x0 and the store path. `cond` is the
    /// AArch64 condition code number (cset encodes the inverse of it).
    fn finish_bool(&mut self, dest: u8, emit_cset: impl FnOnce(&mut Self, u8)) {
        match self.direct_dest_x(dest) {
            Some(d) => emit_cset(self, d),
            None => {
                emit_cset(self, 0);
                self.store_from_x0(dest, ValueTag::Bool.as_u8());
            }
        }
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
        let cmp = self.compare_operands(lhs, rhs, lhs_type, rhs_type);
        self.emit_compare(cmp);
        match cmp {
            Compare::Float(..) => self.finish_bool(dest, |s, d| dynasm!(s.ops ; .arch aarch64 ; cset X(d), mi)),
            Compare::Int(..) => self.finish_bool(dest, |s, d| dynasm!(s.ops ; .arch aarch64 ; cset X(d), lt)),
        }
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
        let cmp = self.compare_operands(lhs, rhs, lhs_type, rhs_type);
        self.emit_compare(cmp);
        match cmp {
            Compare::Float(..) => self.finish_bool(dest, |s, d| dynasm!(s.ops ; .arch aarch64 ; cset X(d), ls)),
            Compare::Int(..) => self.finish_bool(dest, |s, d| dynasm!(s.ops ; .arch aarch64 ; cset X(d), le)),
        }
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
        let cmp = self.compare_operands(lhs, rhs, lhs_type, rhs_type);
        self.emit_compare(cmp);
        self.finish_bool(dest, |s, d| dynasm!(s.ops ; .arch aarch64 ; cset X(d), gt));
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
        let cmp = self.compare_operands(lhs, rhs, lhs_type, rhs_type);
        self.emit_compare(cmp);
        self.finish_bool(dest, |s, d| dynasm!(s.ops ; .arch aarch64 ; cset X(d), ge));
        Ok(())
    }

    fn compile_equality(
        &mut self,
        dest: u8,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
        equal: bool,
    ) -> Result<()> {
        if lhs_type != rhs_type {
            let value = u32::from(!equal);
            self.finish_bool(dest, |s, d| dynasm!(s.ops ; .arch aarch64 ; movz X(d), #value));
            return Ok(());
        }
        match lhs_type {
            ValueType::Float => {
                let cmp = self.compare_operands(lhs, rhs, lhs_type, rhs_type);
                self.emit_compare(cmp);
            }
            ValueType::Bool => {
                self.load_bool_payload(0, lhs);
                self.load_bool_payload(10, rhs);
                dynasm!(self.ops ; .arch aarch64 ; cmp w0, w10);
            }
            _ => {
                let (a, b) = self.operand_pair_x(lhs, rhs, 0, 10);
                dynasm!(self.ops ; .arch aarch64 ; cmp X(a), X(b));
            }
        }
        if equal {
            self.finish_bool(dest, |s, d| dynasm!(s.ops ; .arch aarch64 ; cset X(d), eq));
        } else {
            self.finish_bool(dest, |s, d| dynasm!(s.ops ; .arch aarch64 ; cset X(d), ne));
        }
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
        self.compile_equality(dest, lhs, rhs, lhs_type, rhs_type, true)
    }

    pub(super) fn compile_ne(
        &mut self,
        dest: u8,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> Result<()> {
        self.compile_equality(dest, lhs, rhs, lhs_type, rhs_type, false)
    }
}

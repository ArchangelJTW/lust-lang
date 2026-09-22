use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl JitCompiler {
    /// X(d) = X(a) <op> X(b) for integer operands. Division by zero fails
    /// the trace.
    fn emit_int_op_regs(&mut self, op: BinOp, d: u8, a: u8, b: u8) {
        match op {
            BinOp::Add => dynasm!(self.ops ; .arch aarch64 ; add X(d), X(a), X(b)),
            BinOp::Sub => dynasm!(self.ops ; .arch aarch64 ; sub X(d), X(a), X(b)),
            BinOp::Mul => dynasm!(self.ops ; .arch aarch64 ; mul X(d), X(a), X(b)),
            BinOp::Div => dynasm!(self.ops
                ; .arch aarch64
                ; cbz X(b), >fail
                ; sdiv X(d), X(a), X(b)
            ),
        }
    }

    /// x0 = x0 <op> x9 for integer operands. Division by zero fails the trace.
    fn emit_int_op(&mut self, op: BinOp) {
        self.emit_int_op_regs(op, 0, 0, 9);
    }

    /// D(d) = D(a) <op> D(b)
    fn emit_float_op_regs(&mut self, op: BinOp, d: u8, a: u8, b: u8) {
        match op {
            BinOp::Add => dynasm!(self.ops ; .arch aarch64 ; fadd D(d), D(a), D(b)),
            BinOp::Sub => dynasm!(self.ops ; .arch aarch64 ; fsub D(d), D(a), D(b)),
            BinOp::Mul => dynasm!(self.ops ; .arch aarch64 ; fmul D(d), D(a), D(b)),
            BinOp::Div => dynasm!(self.ops ; .arch aarch64 ; fdiv D(d), D(a), D(b)),
        }
    }

    /// d0 = d0 <op> d1
    fn emit_float_op(&mut self, op: BinOp) {
        self.emit_float_op_regs(op, 0, 0, 1);
    }

    /// Integer result of `X(a) <op> X(b)` into registers[dest]: straight
    /// into a carried pin, otherwise via x0 and the store path.
    fn finish_int_op(&mut self, op: BinOp, dest: u8, a: u8, b: u8) {
        match self.direct_dest_x(dest) {
            Some(d) => self.emit_int_op_regs(op, d, a, b),
            None => {
                self.emit_int_op_regs(op, 0, a, b);
                self.store_from_x0(dest, ValueTag::Int.as_u8());
            }
        }
    }

    /// Float result of `D(a) <op> D(b)` into registers[dest].
    fn finish_float_op(&mut self, op: BinOp, dest: u8, a: u8, b: u8) {
        match self.direct_dest_d(dest) {
            Some(d) => self.emit_float_op_regs(op, d, a, b),
            None => {
                self.emit_float_op_regs(op, 0, a, b);
                self.store_d0_as_float(dest);
            }
        }
    }

    /// Fuse `LoadConst <number>` with a following float `Add/Sub/Mul/Div`
    /// that consumes it, when the constant register is dead afterwards: the
    /// constant is materialized in d1 and never stored. The constant may be
    /// an Int used in a mixed operation; it is converted at compile time.
    pub(super) fn compile_float_op_immediate(
        &mut self,
        load: &TraceOp,
        arithmetic: &TraceOp,
    ) -> Result<bool> {
        let TraceOp::LoadConst {
            dest: constant_register,
            value,
        } = load
        else {
            return Ok(false);
        };
        let constant = match value {
            Value::Float(f) => *f,
            Value::Int(i) => *i as f64,
            _ => return Ok(false),
        };
        let (op, dest, lhs, rhs, lhs_type, rhs_type) = match arithmetic {
            TraceOp::Add {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => (BinOp::Add, *dest, *lhs, *rhs, *lhs_type, *rhs_type),
            TraceOp::Sub {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => (BinOp::Sub, *dest, *lhs, *rhs, *lhs_type, *rhs_type),
            TraceOp::Mul {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => (BinOp::Mul, *dest, *lhs, *rhs, *lhs_type, *rhs_type),
            TraceOp::Div {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => (BinOp::Div, *dest, *lhs, *rhs, *lhs_type, *rhs_type),
            _ => return Ok(false),
        };
        let numeric = |ty: ValueType| matches!(ty, ValueType::Int | ValueType::Float);
        if !numeric(lhs_type) || !numeric(rhs_type) {
            return Ok(false);
        }
        // Only float results: (Int, Int) stays on the integer path.
        if lhs_type == ValueType::Int && rhs_type == ValueType::Int {
            return Ok(false);
        }
        let constant_is_lhs = lhs == *constant_register && rhs != *constant_register;
        let constant_is_rhs = rhs == *constant_register && lhs != *constant_register;
        if !constant_is_lhs && !constant_is_rhs {
            return Ok(false);
        }
        let (other, other_type) = if constant_is_lhs {
            (rhs, rhs_type)
        } else {
            (lhs, lhs_type)
        };
        // The constant's annotated type must match what we materialize.
        let constant_type = if constant_is_lhs { lhs_type } else { rhs_type };
        if constant_type != ValueType::Float && !matches!(value, Value::Int(_)) {
            return Ok(false);
        }

        self.emit_mov_imm64(0, constant.to_bits());
        dynasm!(self.ops ; .arch aarch64 ; fmov d1, x0);
        let o = self.operand_numeric_d(other, other_type, 0);
        if constant_is_lhs {
            self.finish_float_op(op, dest, 1, o);
        } else {
            self.finish_float_op(op, dest, o, 1);
        }
        Ok(true)
    }

    fn compile_binary_specialized(
        &mut self,
        dest: u8,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
        op: BinOp,
    ) -> Result<()> {
        if lhs_type == ValueType::Int && rhs_type == ValueType::Int {
            let (a, b) = self.operand_pair_x(lhs, rhs, 0, 9);
            self.finish_int_op(op, dest, a, b);
            return Ok(());
        }

        let numeric = |ty: ValueType| matches!(ty, ValueType::Int | ValueType::Float);
        if numeric(lhs_type) && numeric(rhs_type) {
            let (a, b) = self.operand_pair_numeric_d(lhs, rhs, lhs_type, rhs_type);
            self.finish_float_op(op, dest, a, b);
            return Ok(());
        }

        self.compile_binary_generic(dest, lhs, rhs, op)
    }

    /// Runtime-dispatched numeric operation: if either operand is a Float the
    /// result is a Float (ints are converted), otherwise both are treated as
    /// Int payloads. Mirrors the x86_64 backend's generic path.
    fn compile_binary_generic(&mut self, dest: u8, lhs: u8, rhs: u8, op: BinOp) -> Result<()> {
        let float_tag = ValueTag::Float.as_u8() as u32;
        let int_tag = ValueTag::Int.as_u8() as u32;
        self.load_tag(0, lhs);
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp w0, #float_tag
            ; b.eq >float_path
        );
        self.load_tag(0, rhs);
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp w0, #float_tag
            ; b.eq >float_path
        );
        self.load_payload(0, lhs);
        self.load_payload(9, rhs);
        self.emit_int_op(op);
        dynasm!(self.ops
            ; .arch aarch64
            ; b >store_int
            ; float_path:
        );
        self.load_tag(0, lhs);
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp w0, #int_tag
            ; b.ne >lhs_is_float
        );
        self.load_payload_int_as_f(0, lhs);
        dynasm!(self.ops
            ; .arch aarch64
            ; b >rhs_check
            ; lhs_is_float:
        );
        self.load_payload_f(0, lhs);
        dynasm!(self.ops
            ; .arch aarch64
            ; rhs_check:
        );
        self.load_tag(0, rhs);
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp w0, #int_tag
            ; b.ne >rhs_is_float
        );
        self.load_payload_int_as_f(1, rhs);
        dynasm!(self.ops
            ; .arch aarch64
            ; b >do_float_op
            ; rhs_is_float:
        );
        self.load_payload_f(1, rhs);
        dynasm!(self.ops
            ; .arch aarch64
            ; do_float_op:
        );
        self.emit_float_op(op);
        self.store_d0_as_float(dest);
        dynasm!(self.ops
            ; .arch aarch64
            ; b >arith_done
            ; store_int:
        );
        self.store_from_x0(dest, ValueTag::Int.as_u8());
        dynasm!(self.ops
            ; .arch aarch64
            ; arith_done:
        );
        Ok(())
    }

    pub(super) fn compile_add_specialized(
        &mut self,
        dest: u8,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> Result<()> {
        self.compile_binary_specialized(dest, lhs, rhs, lhs_type, rhs_type, BinOp::Add)
    }

    pub(super) fn compile_sub_specialized(
        &mut self,
        dest: u8,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> Result<()> {
        self.compile_binary_specialized(dest, lhs, rhs, lhs_type, rhs_type, BinOp::Sub)
    }

    pub(super) fn compile_mul_specialized(
        &mut self,
        dest: u8,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> Result<()> {
        self.compile_binary_specialized(dest, lhs, rhs, lhs_type, rhs_type, BinOp::Mul)
    }

    pub(super) fn compile_div_specialized(
        &mut self,
        dest: u8,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> Result<()> {
        self.compile_binary_specialized(dest, lhs, rhs, lhs_type, rhs_type, BinOp::Div)
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
            let (a, b) = self.operand_pair_x(lhs, rhs, 0, 9);
            match self.direct_dest_x(dest) {
                Some(d) => self.emit_int_mod_regs(d, a, b),
                None => {
                    self.emit_int_mod_regs(0, a, b);
                    self.store_from_x0(dest, ValueTag::Int.as_u8());
                }
            }
            return Ok(());
        }

        let numeric = |ty: ValueType| matches!(ty, ValueType::Int | ValueType::Float);
        if numeric(lhs_type) && numeric(rhs_type) {
            // fmod takes its arguments in d0/d1 and returns in d0.
            let (a, b) = self.operand_pair_numeric_d(lhs, rhs, lhs_type, rhs_type);
            if a != 0 {
                dynasm!(self.ops ; .arch aarch64 ; fmov d0, D(a));
            }
            if b != 1 {
                dynasm!(self.ops ; .arch aarch64 ; fmov d1, D(b));
            }
            self.emit_float_mod();
            match self.direct_dest_d(dest) {
                Some(d) => dynasm!(self.ops ; .arch aarch64 ; fmov D(d), d0),
                None => self.store_d0_as_float(dest),
            }
            return Ok(());
        }

        self.compile_mod_generic(dest, lhs, rhs)
    }

    /// X(d) = X(a) % X(b) with the interpreter's semantics: sign of the
    /// dividend, modulo by zero fails the trace so the interpreter raises the
    /// error. x10 is scratch.
    fn emit_int_mod_regs(&mut self, d: u8, a: u8, b: u8) {
        dynasm!(self.ops
            ; .arch aarch64
            ; cbz X(b), >fail
            ; sdiv x10, X(a), X(b)
            ; msub X(d), x10, X(b), X(a)
        );
    }

    /// x0 = x0 % x9 (see `emit_int_mod_regs`).
    fn emit_int_mod(&mut self) {
        self.emit_int_mod_regs(0, 0, 9);
    }

    /// d0 = d0 % d1 as Rust's `f64 % f64` (libm `fmod`: exact, sign of the
    /// dividend, NaN for NaN or infinite dividends). A zero divisor fails
    /// the trace like the interpreter's "Modulo by zero"; NaN is not equal
    /// to zero and falls through to fmod, again like the interpreter.
    fn emit_float_mod(&mut self) {
        unsafe extern "C" {
            fn fmod(a: f64, b: f64) -> f64;
        }
        dynasm!(self.ops
            ; .arch aarch64
            ; fcmp d1, 0.0
            ; b.eq >fail
        );
        self.emit_call(fmod as *const ());
    }

    /// Runtime-dispatched modulo for operands of unknown static type: a
    /// Float on either side selects the float path (ints converted),
    /// otherwise both payloads are integers.
    fn compile_mod_generic(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        let float_tag = ValueTag::Float.as_u8() as u32;
        let int_tag = ValueTag::Int.as_u8() as u32;
        self.load_tag(0, lhs);
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp w0, #float_tag
            ; b.eq >float_path
        );
        self.load_tag(0, rhs);
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp w0, #float_tag
            ; b.eq >float_path
        );
        self.load_payload(0, lhs);
        self.load_payload(9, rhs);
        self.emit_int_mod();
        dynasm!(self.ops
            ; .arch aarch64
            ; b >store_int
            ; float_path:
        );
        self.load_tag(0, lhs);
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp w0, #int_tag
            ; b.ne >lhs_is_float
        );
        self.load_payload_int_as_f(0, lhs);
        dynasm!(self.ops
            ; .arch aarch64
            ; b >rhs_check
            ; lhs_is_float:
        );
        self.load_payload_f(0, lhs);
        dynasm!(self.ops
            ; .arch aarch64
            ; rhs_check:
        );
        self.load_tag(0, rhs);
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp w0, #int_tag
            ; b.ne >rhs_is_float
        );
        self.load_payload_int_as_f(1, rhs);
        dynasm!(self.ops
            ; .arch aarch64
            ; b >do_float_mod
            ; rhs_is_float:
        );
        self.load_payload_f(1, rhs);
        dynasm!(self.ops
            ; .arch aarch64
            ; do_float_mod:
        );
        self.emit_float_mod();
        self.store_d0_as_float(dest);
        dynasm!(self.ops
            ; .arch aarch64
            ; b >mod_done
            ; store_int:
        );
        self.store_from_x0(dest, ValueTag::Int.as_u8());
        dynasm!(self.ops
            ; .arch aarch64
            ; mod_done:
        );
        Ok(())
    }
}

use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl JitCompiler {
    /// x0 = x0 <op> x9 for integer operands. Division by zero fails the trace.
    fn emit_int_op(&mut self, op: BinOp) {
        match op {
            BinOp::Add => dynasm!(self.ops ; .arch aarch64 ; add x0, x0, x9),
            BinOp::Sub => dynasm!(self.ops ; .arch aarch64 ; sub x0, x0, x9),
            BinOp::Mul => dynasm!(self.ops ; .arch aarch64 ; mul x0, x0, x9),
            BinOp::Div => dynasm!(self.ops
                ; .arch aarch64
                ; cbz x9, >fail
                ; sdiv x0, x0, x9
            ),
        }
    }

    /// d0 = d0 <op> d1
    fn emit_float_op(&mut self, op: BinOp) {
        match op {
            BinOp::Add => dynasm!(self.ops ; .arch aarch64 ; fadd d0, d0, d1),
            BinOp::Sub => dynasm!(self.ops ; .arch aarch64 ; fsub d0, d0, d1),
            BinOp::Mul => dynasm!(self.ops ; .arch aarch64 ; fmul d0, d0, d1),
            BinOp::Div => dynasm!(self.ops ; .arch aarch64 ; fdiv d0, d0, d1),
        }
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
            self.load_payload(0, lhs);
            self.load_payload(9, rhs);
            self.emit_int_op(op);
            self.store_from_x0(dest, ValueTag::Int.as_u8());
            return Ok(());
        }

        let numeric = |ty: ValueType| matches!(ty, ValueType::Int | ValueType::Float);
        if numeric(lhs_type) && numeric(rhs_type) {
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
            self.emit_float_op(op);
            self.store_d0_as_float(dest);
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
            self.load_payload(0, lhs);
            self.load_payload(9, rhs);
            self.emit_int_mod();
            self.store_from_x0(dest, ValueTag::Int.as_u8());
            return Ok(());
        }

        let numeric = |ty: ValueType| matches!(ty, ValueType::Int | ValueType::Float);
        if numeric(lhs_type) && numeric(rhs_type) {
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
            self.emit_float_mod();
            self.store_d0_as_float(dest);
            return Ok(());
        }

        self.compile_mod_generic(dest, lhs, rhs)
    }

    /// x0 = x0 % x9 with the interpreter's semantics: sign of the dividend,
    /// modulo by zero fails the trace so the interpreter raises the error.
    fn emit_int_mod(&mut self) {
        dynasm!(self.ops
            ; .arch aarch64
            ; cbz x9, >fail
            ; sdiv x10, x0, x9
            ; msub x0, x10, x9, x0
        );
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

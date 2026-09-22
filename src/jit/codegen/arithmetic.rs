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
            self.operands_rax_rbx(lhs, rhs);
            dynasm!(self.ops
                ; .arch x64
                ; add rax, rbx
            );
            self.store_from_rax(dest, 2);
            return Ok(());
        }

        if lhs_type == ValueType::Float && rhs_type == ValueType::Float {
            self.operands_numeric_xmm(lhs, rhs, ValueType::Float, ValueType::Float);
            dynasm!(self.ops ; .arch x64 ; addsd xmm0, xmm1);
            self.store_xmm0_as_float(dest);
            return Ok(());
        }

        if lhs_type == ValueType::Int && rhs_type == ValueType::Float {
            self.operands_numeric_xmm(lhs, rhs, ValueType::Int, ValueType::Float);
            dynasm!(self.ops ; .arch x64 ; addsd xmm0, xmm1);
            self.store_xmm0_as_float(dest);
            return Ok(());
        }

        if lhs_type == ValueType::Float && rhs_type == ValueType::Int {
            self.operands_numeric_xmm(lhs, rhs, ValueType::Float, ValueType::Int);
            dynasm!(self.ops ; .arch x64 ; addsd xmm0, xmm1);
            self.store_xmm0_as_float(dest);
            return Ok(());
        }

        self.compile_add(dest, lhs, rhs)
    }

    pub(super) fn compile_add(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
        let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops
            ; .arch x64
            ; mov al, [r12 + lhs_offset]
            ; cmp al, 3
            ; je >float_path
            ; mov al, [r12 + rhs_offset]
            ; cmp al, 3
            ; je >float_path
            ; mov rax, [r12 + lhs_offset + 8]
            ; mov rbx, [r12 + rhs_offset + 8]
            ; add rax, rbx
            ; jmp >store_int
            ; float_path:
            ; mov al, [r12 + lhs_offset]
            ; cmp al, 2
            ; jne >lhs_is_float
            ; mov rax, [r12 + lhs_offset + 8]
            ; cvtsi2sd xmm0, rax
            ; jmp >rhs_check
            ; lhs_is_float:
            ; movsd xmm0, [r12 + lhs_offset + 8]
            ; rhs_check:
            ; mov al, [r12 + rhs_offset]
            ; cmp al, 2
            ; jne >rhs_is_float
            ; mov rax, [r12 + rhs_offset + 8]
            ; cvtsi2sd xmm1, rax
            ; jmp >do_float_add
            ; rhs_is_float:
            ; movsd xmm1, [r12 + rhs_offset + 8]
            ; do_float_add:
            ; addsd xmm0, xmm1
        );
        self.store_xmm0_as_float(dest);
        dynasm!(self.ops
            ; .arch x64
            ; jmp >done
            ; store_int:
        );
        self.store_from_rax(dest, 2);
        dynasm!(self.ops
            ; .arch x64
            ; done:
        );
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
            self.operands_rax_rbx(lhs, rhs);
            dynasm!(self.ops
                ; .arch x64
                ; sub rax, rbx
            );
            self.store_from_rax(dest, 2);
            return Ok(());
        }

        if lhs_type == ValueType::Float && rhs_type == ValueType::Float {
            self.operands_numeric_xmm(lhs, rhs, ValueType::Float, ValueType::Float);
            dynasm!(self.ops ; .arch x64 ; subsd xmm0, xmm1);
            self.store_xmm0_as_float(dest);
            return Ok(());
        }

        if lhs_type == ValueType::Int && rhs_type == ValueType::Float {
            self.operands_numeric_xmm(lhs, rhs, ValueType::Int, ValueType::Float);
            dynasm!(self.ops ; .arch x64 ; subsd xmm0, xmm1);
            self.store_xmm0_as_float(dest);
            return Ok(());
        }

        if lhs_type == ValueType::Float && rhs_type == ValueType::Int {
            self.operands_numeric_xmm(lhs, rhs, ValueType::Float, ValueType::Int);
            dynasm!(self.ops ; .arch x64 ; subsd xmm0, xmm1);
            self.store_xmm0_as_float(dest);
            return Ok(());
        }

        self.compile_sub(dest, lhs, rhs)
    }

    pub(super) fn compile_sub(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
        let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops
            ; .arch x64
            ; mov al, [r12 + lhs_offset]
            ; cmp al, 3
            ; je >float_path
            ; mov al, [r12 + rhs_offset]
            ; cmp al, 3
            ; je >float_path
            ; mov rax, [r12 + lhs_offset + 8]
            ; mov rbx, [r12 + rhs_offset + 8]
            ; sub rax, rbx
            ; jmp >store_int
            ; float_path:
            ; mov al, [r12 + lhs_offset]
            ; cmp al, 2
            ; jne >lhs_is_float
            ; mov rax, [r12 + lhs_offset + 8]
            ; cvtsi2sd xmm0, rax
            ; jmp >rhs_check
            ; lhs_is_float:
            ; movsd xmm0, [r12 + lhs_offset + 8]
            ; rhs_check:
            ; mov al, [r12 + rhs_offset]
            ; cmp al, 2
            ; jne >rhs_is_float
            ; mov rax, [r12 + rhs_offset + 8]
            ; cvtsi2sd xmm1, rax
            ; jmp >do_float_sub
            ; rhs_is_float:
            ; movsd xmm1, [r12 + rhs_offset + 8]
            ; do_float_sub:
            ; subsd xmm0, xmm1
        );
        self.store_xmm0_as_float(dest);
        dynasm!(self.ops
            ; .arch x64
            ; jmp >done
            ; store_int:
        );
        self.store_from_rax(dest, 2);
        dynasm!(self.ops
            ; .arch x64
            ; done:
        );
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
            self.operands_rax_rbx(lhs, rhs);
            dynasm!(self.ops
                ; .arch x64
                ; imul rax, rbx
            );
            self.store_from_rax(dest, 2);
            return Ok(());
        }

        if lhs_type == ValueType::Float && rhs_type == ValueType::Float {
            self.operands_numeric_xmm(lhs, rhs, ValueType::Float, ValueType::Float);
            dynasm!(self.ops ; .arch x64 ; mulsd xmm0, xmm1);
            self.store_xmm0_as_float(dest);
            return Ok(());
        }

        if lhs_type == ValueType::Int && rhs_type == ValueType::Float {
            self.operands_numeric_xmm(lhs, rhs, ValueType::Int, ValueType::Float);
            dynasm!(self.ops ; .arch x64 ; mulsd xmm0, xmm1);
            self.store_xmm0_as_float(dest);
            return Ok(());
        }

        if lhs_type == ValueType::Float && rhs_type == ValueType::Int {
            self.operands_numeric_xmm(lhs, rhs, ValueType::Float, ValueType::Int);
            dynasm!(self.ops ; .arch x64 ; mulsd xmm0, xmm1);
            self.store_xmm0_as_float(dest);
            return Ok(());
        }

        self.compile_mul(dest, lhs, rhs)
    }

    pub(super) fn compile_mul(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
        let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops
            ; .arch x64
            ; mov al, [r12 + lhs_offset]
            ; cmp al, 3
            ; je >float_path
            ; mov al, [r12 + rhs_offset]
            ; cmp al, 3
            ; je >float_path
            ; mov rax, [r12 + lhs_offset + 8]
            ; mov rbx, [r12 + rhs_offset + 8]
            ; imul rax, rbx
            ; jmp >store_int
            ; float_path:
            ; mov al, [r12 + lhs_offset]
            ; cmp al, 2
            ; jne >lhs_is_float
            ; mov rax, [r12 + lhs_offset + 8]
            ; cvtsi2sd xmm0, rax
            ; jmp >rhs_check
            ; lhs_is_float:
            ; movsd xmm0, [r12 + lhs_offset + 8]
            ; rhs_check:
            ; mov al, [r12 + rhs_offset]
            ; cmp al, 2
            ; jne >rhs_is_float
            ; mov rax, [r12 + rhs_offset + 8]
            ; cvtsi2sd xmm1, rax
            ; jmp >do_float_mul
            ; rhs_is_float:
            ; movsd xmm1, [r12 + rhs_offset + 8]
            ; do_float_mul:
            ; mulsd xmm0, xmm1
        );
        self.store_xmm0_as_float(dest);
        dynasm!(self.ops
            ; .arch x64
            ; jmp >done
            ; store_int:
        );
        self.store_from_rax(dest, 2);
        dynasm!(self.ops
            ; .arch x64
            ; done:
        );
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
            self.operands_idiv(lhs, rhs);
            self.emit_int_div();
            self.store_from_rax(dest, 2);
            return Ok(());
        }

        if lhs_type == ValueType::Float && rhs_type == ValueType::Float {
            self.operands_numeric_xmm(lhs, rhs, ValueType::Float, ValueType::Float);
            dynasm!(self.ops ; .arch x64 ; divsd xmm0, xmm1);
            self.store_xmm0_as_float(dest);
            return Ok(());
        }

        if lhs_type == ValueType::Int && rhs_type == ValueType::Float {
            self.operands_numeric_xmm(lhs, rhs, ValueType::Int, ValueType::Float);
            dynasm!(self.ops ; .arch x64 ; divsd xmm0, xmm1);
            self.store_xmm0_as_float(dest);
            return Ok(());
        }

        if lhs_type == ValueType::Float && rhs_type == ValueType::Int {
            self.operands_numeric_xmm(lhs, rhs, ValueType::Float, ValueType::Int);
            dynasm!(self.ops ; .arch x64 ; divsd xmm0, xmm1);
            self.store_xmm0_as_float(dest);
            return Ok(());
        }

        self.compile_div(dest, lhs, rhs)
    }

    pub(super) fn compile_div(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
        let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops
            ; .arch x64
            ; mov al, [r12 + lhs_offset]
            ; cmp al, 3
            ; je >float_path
            ; mov al, [r12 + rhs_offset]
            ; cmp al, 3
            ; je >float_path
            ; mov rax, [r12 + lhs_offset + 8]
            ; mov rbx, [r12 + rhs_offset + 8]
        );
        self.emit_int_div();
        dynasm!(self.ops
            ; .arch x64
            ; jmp >store_int
            ; float_path:
            ; mov al, [r12 + lhs_offset]
            ; cmp al, 2
            ; jne >lhs_is_float
            ; mov rax, [r12 + lhs_offset + 8]
            ; cvtsi2sd xmm0, rax
            ; jmp >rhs_check
            ; lhs_is_float:
            ; movsd xmm0, [r12 + lhs_offset + 8]
            ; rhs_check:
            ; mov al, [r12 + rhs_offset]
            ; cmp al, 2
            ; jne >rhs_is_float
            ; mov rax, [r12 + rhs_offset + 8]
            ; cvtsi2sd xmm1, rax
            ; jmp >do_float_div
            ; rhs_is_float:
            ; movsd xmm1, [r12 + rhs_offset + 8]
            ; do_float_div:
            ; divsd xmm0, xmm1
        );
        self.store_xmm0_as_float(dest);
        dynasm!(self.ops
            ; .arch x64
            ; jmp >done
            ; store_int:
        );
        self.store_from_rax(dest, 2);
        dynasm!(self.ops
            ; .arch x64
            ; done:
        );
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
            self.operands_idiv(lhs, rhs);
            self.emit_int_mod();
            self.store_from_rax(dest, 2);
            return Ok(());
        }

        let numeric = |ty: ValueType| matches!(ty, ValueType::Int | ValueType::Float);
        if numeric(lhs_type) && numeric(rhs_type) {
            self.operands_numeric_xmm(lhs, rhs, lhs_type, rhs_type);
            self.emit_float_mod();
            self.store_xmm0_as_float(dest);
            return Ok(());
        }

        self.compile_mod(dest, lhs, rhs)
    }

    /// rax = rax / rbx with the interpreter's semantics (`wrapping_div`):
    /// division by zero fails the trace so the interpreter raises the
    /// error, and `MIN / -1` wraps to `MIN`.
    ///
    /// The divisor is tested against -1 because x86 `idiv` raises #DE (a
    /// `SIGFPE`) for `MIN / -1`, whose quotient does not fit the
    /// destination. aarch64 `sdiv` and riscv `div` wrap in hardware, so
    /// only this backend needs the check. `neg rax` gives the wrapped
    /// quotient for any dividend (`0 - x`), `MIN` included.
    fn emit_int_div(&mut self) {
        dynasm!(self.ops
            ; .arch x64
            ; test rbx, rbx
            ; jz >fail
            ; cmp rbx, -1
            ; jne >div_safe
            ; neg rax
            ; jmp >div_done
            ; div_safe:
            ; cqo
            ; idiv rbx
            ; div_done:
        );
    }

    /// rax = rax % rbx with the interpreter's semantics (`wrapping_rem`):
    /// sign of the dividend, modulo by zero fails the trace so the
    /// interpreter raises the error.
    ///
    /// `x % -1` is 0 for every dividend, and taking that path also avoids
    /// the #DE that `idiv` would raise for `MIN % -1` (see
    /// `emit_int_div`).
    fn emit_int_mod(&mut self) {
        dynasm!(self.ops
            ; .arch x64
            ; test rbx, rbx
            ; jz >fail
            ; cmp rbx, -1
            ; jne >mod_safe
            ; xor eax, eax
            ; jmp >mod_done
            ; mod_safe:
            ; cqo
            ; idiv rbx
            ; mov rax, rdx
            ; mod_done:
        );
    }

    /// xmm0 = xmm0 % xmm1 as Rust's `f64 % f64` (libm `fmod`). A zero
    /// divisor fails the trace like the interpreter's "Modulo by zero"; NaN
    /// compares unordered, not equal to zero, and falls through to fmod like
    /// the interpreter.
    fn emit_float_mod(&mut self) {
        unsafe extern "C" {
            fn fmod(a: f64, b: f64) -> f64;
        }
        dynasm!(self.ops
            ; .arch x64
            ; xorpd xmm2, xmm2
            ; ucomisd xmm1, xmm2
            ; jp >divisor_ok
            ; je >fail
            ; divisor_ok:
            ; mov rax, QWORD fmod as *const () as _
            ; call rax
        );
    }

    /// Runtime-dispatched modulo for operands of unknown static type: a
    /// Float on either side selects the float path (ints converted),
    /// otherwise both payloads are integers.
    pub(super) fn compile_mod(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
        let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops
            ; .arch x64
            ; mov al, [r12 + lhs_offset]
            ; cmp al, 3
            ; je >float_path
            ; mov al, [r12 + rhs_offset]
            ; cmp al, 3
            ; je >float_path
            ; mov rax, [r12 + lhs_offset + 8]
            ; mov rbx, [r12 + rhs_offset + 8]
        );
        self.emit_int_mod();
        dynasm!(self.ops
            ; .arch x64
            ; jmp >store_int
            ; float_path:
            ; mov al, [r12 + lhs_offset]
            ; cmp al, 2
            ; jne >lhs_is_float
            ; mov rax, [r12 + lhs_offset + 8]
            ; cvtsi2sd xmm0, rax
            ; jmp >rhs_check
            ; lhs_is_float:
            ; movsd xmm0, [r12 + lhs_offset + 8]
            ; rhs_check:
            ; mov al, [r12 + rhs_offset]
            ; cmp al, 2
            ; jne >rhs_is_float
            ; mov rax, [r12 + rhs_offset + 8]
            ; cvtsi2sd xmm1, rax
            ; jmp >do_float_mod
            ; rhs_is_float:
            ; movsd xmm1, [r12 + rhs_offset + 8]
            ; do_float_mod:
        );
        self.emit_float_mod();
        self.store_xmm0_as_float(dest);
        dynasm!(self.ops
            ; .arch x64
            ; jmp >mod_done
            ; store_int:
        );
        self.store_from_rax(dest, 2);
        dynasm!(self.ops
            ; .arch x64
            ; mod_done:
        );
        Ok(())
    }
}

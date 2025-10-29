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
            self.load_to_rax(lhs);
            self.load_to_rbx(rhs);
            dynasm!(self.ops
                ; add rax, rbx
            );
            self.store_from_rax(dest, 2);
            return Ok(());
        }

        if lhs_type == ValueType::Float && rhs_type == ValueType::Float {
            let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
            let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
            let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
            dynasm!(self.ops
                ; movsd xmm0, [r12 + lhs_offset + 8]
                ; movsd xmm1, [r12 + rhs_offset + 8]
                ; addsd xmm0, xmm1
                ; mov BYTE [r12 + dest_offset], 3
                ; movsd [r12 + dest_offset + 8], xmm0
            );
            return Ok(());
        }

        if lhs_type == ValueType::Int && rhs_type == ValueType::Float {
            let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
            let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
            let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
            dynasm!(self.ops
                ; mov rax, [r12 + lhs_offset + 8]
                ; cvtsi2sd xmm0, rax
                ; movsd xmm1, [r12 + rhs_offset + 8]
                ; addsd xmm0, xmm1
                ; mov BYTE [r12 + dest_offset], 3
                ; movsd [r12 + dest_offset + 8], xmm0
            );
            return Ok(());
        }

        if lhs_type == ValueType::Float && rhs_type == ValueType::Int {
            let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
            let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
            let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
            dynasm!(self.ops
                ; movsd xmm0, [r12 + lhs_offset + 8]
                ; mov rax, [r12 + rhs_offset + 8]
                ; cvtsi2sd xmm1, rax
                ; addsd xmm0, xmm1
                ; mov BYTE [r12 + dest_offset], 3
                ; movsd [r12 + dest_offset + 8], xmm0
            );
            return Ok(());
        }

        self.compile_add(dest, lhs, rhs)
    }

    pub(super) fn compile_add(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
        let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops
            ; mov al, [r12 + lhs_offset]
            ; cmp al, 3
            ; je >float_path
            ; mov al, [r12 + rhs_offset]
            ; cmp al, 3
            ; je >float_path
            ; mov rax, [r12 + lhs_offset + 8]
            ; mov rbx, [r12 + rhs_offset + 8]
            ; add rax, rbx
            ; mov BYTE [r12 + dest_offset], 2
            ; mov [r12 + dest_offset + 8], rax
            ; jmp >done
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
            ; mov BYTE [r12 + dest_offset], 3
            ; movsd [r12 + dest_offset + 8], xmm0
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
            self.load_to_rax(lhs);
            self.load_to_rbx(rhs);
            dynasm!(self.ops
                ; sub rax, rbx
            );
            self.store_from_rax(dest, 2);
            return Ok(());
        }

        if lhs_type == ValueType::Float && rhs_type == ValueType::Float {
            let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
            let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
            let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
            dynasm!(self.ops
                ; movsd xmm0, [r12 + lhs_offset + 8]
                ; movsd xmm1, [r12 + rhs_offset + 8]
                ; subsd xmm0, xmm1
                ; mov BYTE [r12 + dest_offset], 3
                ; movsd [r12 + dest_offset + 8], xmm0
            );
            return Ok(());
        }

        if lhs_type == ValueType::Int && rhs_type == ValueType::Float {
            let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
            let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
            let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
            dynasm!(self.ops
                ; mov rax, [r12 + lhs_offset + 8]
                ; cvtsi2sd xmm0, rax
                ; movsd xmm1, [r12 + rhs_offset + 8]
                ; subsd xmm0, xmm1
                ; mov BYTE [r12 + dest_offset], 3
                ; movsd [r12 + dest_offset + 8], xmm0
            );
            return Ok(());
        }

        if lhs_type == ValueType::Float && rhs_type == ValueType::Int {
            let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
            let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
            let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
            dynasm!(self.ops
                ; movsd xmm0, [r12 + lhs_offset + 8]
                ; mov rax, [r12 + rhs_offset + 8]
                ; cvtsi2sd xmm1, rax
                ; subsd xmm0, xmm1
                ; mov BYTE [r12 + dest_offset], 3
                ; movsd [r12 + dest_offset + 8], xmm0
            );
            return Ok(());
        }

        self.compile_sub(dest, lhs, rhs)
    }

    pub(super) fn compile_sub(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
        let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops
            ; mov al, [r12 + lhs_offset]
            ; cmp al, 3
            ; je >float_path
            ; mov al, [r12 + rhs_offset]
            ; cmp al, 3
            ; je >float_path
            ; mov rax, [r12 + lhs_offset + 8]
            ; mov rbx, [r12 + rhs_offset + 8]
            ; sub rax, rbx
            ; mov BYTE [r12 + dest_offset], 2
            ; mov [r12 + dest_offset + 8], rax
            ; jmp >done
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
            ; mov BYTE [r12 + dest_offset], 3
            ; movsd [r12 + dest_offset + 8], xmm0
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
            self.load_to_rax(lhs);
            self.load_to_rbx(rhs);
            dynasm!(self.ops
                ; imul rax, rbx
            );
            self.store_from_rax(dest, 2);
            return Ok(());
        }

        if lhs_type == ValueType::Float && rhs_type == ValueType::Float {
            let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
            let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
            let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
            dynasm!(self.ops
                ; movsd xmm0, [r12 + lhs_offset + 8]
                ; movsd xmm1, [r12 + rhs_offset + 8]
                ; mulsd xmm0, xmm1
                ; mov BYTE [r12 + dest_offset], 3
                ; movsd [r12 + dest_offset + 8], xmm0
            );
            return Ok(());
        }

        if lhs_type == ValueType::Int && rhs_type == ValueType::Float {
            let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
            let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
            let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
            dynasm!(self.ops
                ; mov rax, [r12 + lhs_offset + 8]
                ; cvtsi2sd xmm0, rax
                ; movsd xmm1, [r12 + rhs_offset + 8]
                ; mulsd xmm0, xmm1
                ; mov BYTE [r12 + dest_offset], 3
                ; movsd [r12 + dest_offset + 8], xmm0
            );
            return Ok(());
        }

        if lhs_type == ValueType::Float && rhs_type == ValueType::Int {
            let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
            let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
            let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
            dynasm!(self.ops
                ; movsd xmm0, [r12 + lhs_offset + 8]
                ; mov rax, [r12 + rhs_offset + 8]
                ; cvtsi2sd xmm1, rax
                ; mulsd xmm0, xmm1
                ; mov BYTE [r12 + dest_offset], 3
                ; movsd [r12 + dest_offset + 8], xmm0
            );
            return Ok(());
        }

        self.compile_mul(dest, lhs, rhs)
    }

    pub(super) fn compile_mul(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
        let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops
            ; mov al, [r12 + lhs_offset]
            ; cmp al, 3
            ; je >float_path
            ; mov al, [r12 + rhs_offset]
            ; cmp al, 3
            ; je >float_path
            ; mov rax, [r12 + lhs_offset + 8]
            ; mov rbx, [r12 + rhs_offset + 8]
            ; imul rax, rbx
            ; mov BYTE [r12 + dest_offset], 2
            ; mov [r12 + dest_offset + 8], rax
            ; jmp >done
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
            ; mov BYTE [r12 + dest_offset], 3
            ; movsd [r12 + dest_offset + 8], xmm0
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
            self.load_to_rax(lhs);
            self.load_to_rbx(rhs);
            dynasm!(self.ops
                ; test rbx, rbx
                ; jz >fail
                ; xor rdx, rdx
                ; idiv rbx
            );
            self.store_from_rax(dest, 2);
            return Ok(());
        }

        if lhs_type == ValueType::Float && rhs_type == ValueType::Float {
            let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
            let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
            let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
            dynasm!(self.ops
                ; movsd xmm0, [r12 + lhs_offset + 8]
                ; movsd xmm1, [r12 + rhs_offset + 8]
                ; divsd xmm0, xmm1
                ; mov BYTE [r12 + dest_offset], 3
                ; movsd [r12 + dest_offset + 8], xmm0
            );
            return Ok(());
        }

        if lhs_type == ValueType::Int && rhs_type == ValueType::Float {
            let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
            let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
            let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
            dynasm!(self.ops
                ; mov rax, [r12 + lhs_offset + 8]
                ; cvtsi2sd xmm0, rax
                ; movsd xmm1, [r12 + rhs_offset + 8]
                ; divsd xmm0, xmm1
                ; mov BYTE [r12 + dest_offset], 3
                ; movsd [r12 + dest_offset + 8], xmm0
            );
            return Ok(());
        }

        if lhs_type == ValueType::Float && rhs_type == ValueType::Int {
            let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
            let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
            let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
            dynasm!(self.ops
                ; movsd xmm0, [r12 + lhs_offset + 8]
                ; mov rax, [r12 + rhs_offset + 8]
                ; cvtsi2sd xmm1, rax
                ; divsd xmm0, xmm1
                ; mov BYTE [r12 + dest_offset], 3
                ; movsd [r12 + dest_offset + 8], xmm0
            );
            return Ok(());
        }

        self.compile_div(dest, lhs, rhs)
    }

    pub(super) fn compile_div(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
        let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops
            ; mov al, [r12 + lhs_offset]
            ; cmp al, 3
            ; je >float_path
            ; mov al, [r12 + rhs_offset]
            ; cmp al, 3
            ; je >float_path
            ; mov rax, [r12 + lhs_offset + 8]
            ; mov rbx, [r12 + rhs_offset + 8]
            ; test rbx, rbx
            ; jz >fail
            ; xor rdx, rdx
            ; idiv rbx
            ; mov BYTE [r12 + dest_offset], 2
            ; mov [r12 + dest_offset + 8], rax
            ; jmp >done
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
            ; mov BYTE [r12 + dest_offset], 3
            ; movsd [r12 + dest_offset + 8], xmm0
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
            self.load_to_rax(lhs);
            self.load_to_rbx(rhs);
            dynasm!(self.ops
                ; test rbx, rbx
                ; jz >fail
                ; xor rdx, rdx
                ; idiv rbx
                ; mov rax, rdx
            );
            self.store_from_rax(dest, 2);
            return Ok(());
        }

        self.compile_mod(dest, lhs, rhs)
    }

    pub(super) fn compile_mod(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
        let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops
            ; mov rax, [r12 + lhs_offset + 8]
            ; mov rbx, [r12 + rhs_offset + 8]
            ; test rbx, rbx
            ; jz >fail
            ; xor rdx, rdx
            ; idiv rbx
            ; mov BYTE [r12 + dest_offset], 2
            ; mov [r12 + dest_offset + 8], rdx
        );
        Ok(())
    }
}

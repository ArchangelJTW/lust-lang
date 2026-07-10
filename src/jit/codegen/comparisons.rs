use super::*;

impl JitCompiler {
    fn load_numeric_comparison_operands(
        &mut self,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> bool {
        let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
        let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        if lhs_type == ValueType::Int && rhs_type == ValueType::Int {
            dynasm!(self.ops
                ; mov rax, [r12 + lhs_offset + 8]
                ; mov rcx, [r12 + rhs_offset + 8]
            );
            return false;
        }

        if lhs_type == ValueType::Int {
            dynasm!(self.ops
                ; mov rax, [r12 + lhs_offset + 8]
                ; cvtsi2sd xmm0, rax
            );
        } else {
            dynasm!(self.ops ; movsd xmm0, [r12 + lhs_offset + 8]);
        }
        if rhs_type == ValueType::Int {
            dynasm!(self.ops
                ; mov rax, [r12 + rhs_offset + 8]
                ; cvtsi2sd xmm1, rax
            );
        } else {
            dynasm!(self.ops ; movsd xmm1, [r12 + rhs_offset + 8]);
        }
        true
    }

    pub(super) fn compile_lt(
        &mut self,
        dest: u8,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> Result<()> {
        if self.load_numeric_comparison_operands(lhs, rhs, lhs_type, rhs_type) {
            dynasm!(self.ops ; ucomisd xmm0, xmm1 ; setb al ; setnp cl ; and al, cl ; movzx rax, al);
        } else {
            dynasm!(self.ops ; cmp rax, rcx ; setl al ; movzx rax, al);
        }
        self.store_from_rax(dest, 1);
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
            dynasm!(self.ops ; ucomisd xmm0, xmm1 ; setbe al ; setnp cl ; and al, cl ; movzx rax, al);
        } else {
            dynasm!(self.ops ; cmp rax, rcx ; setle al ; movzx rax, al);
        }
        self.store_from_rax(dest, 1);
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
            dynasm!(self.ops ; ucomisd xmm0, xmm1 ; seta al ; movzx rax, al);
        } else {
            dynasm!(self.ops ; cmp rax, rcx ; setg al ; movzx rax, al);
        }
        self.store_from_rax(dest, 1);
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
            dynasm!(self.ops ; ucomisd xmm0, xmm1 ; setae al ; movzx rax, al);
        } else {
            dynasm!(self.ops ; cmp rax, rcx ; setge al ; movzx rax, al);
        }
        self.store_from_rax(dest, 1);
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
            dynasm!(self.ops ; xor rax, rax);
        } else if lhs_type == ValueType::Float {
            self.load_numeric_comparison_operands(lhs, rhs, lhs_type, rhs_type);
            dynasm!(self.ops ; ucomisd xmm0, xmm1 ; sete al ; setnp cl ; and al, cl ; movzx rax, al);
        } else {
            let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
            let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
            if lhs_type == ValueType::Bool {
                dynasm!(self.ops
                    ; mov al, [r12 + lhs_offset + 8]
                    ; mov cl, [r12 + rhs_offset + 8]
                    ; cmp al, cl
                    ; sete al
                    ; movzx rax, al
                );
            } else {
                dynasm!(self.ops
                    ; mov rax, [r12 + lhs_offset + 8]
                    ; mov rcx, [r12 + rhs_offset + 8]
                    ; cmp rax, rcx
                    ; sete al
                    ; movzx rax, al
                );
            }
        }
        self.store_from_rax(dest, 1);
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
            dynasm!(self.ops ; mov rax, 1);
        } else if lhs_type == ValueType::Float {
            self.load_numeric_comparison_operands(lhs, rhs, lhs_type, rhs_type);
            dynasm!(self.ops ; ucomisd xmm0, xmm1 ; setne al ; setp cl ; or al, cl ; movzx rax, al);
        } else {
            let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
            let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
            if lhs_type == ValueType::Bool {
                dynasm!(self.ops
                    ; mov al, [r12 + lhs_offset + 8]
                    ; mov cl, [r12 + rhs_offset + 8]
                    ; cmp al, cl
                    ; setne al
                    ; movzx rax, al
                );
            } else {
                dynasm!(self.ops
                    ; mov rax, [r12 + lhs_offset + 8]
                    ; mov rcx, [r12 + rhs_offset + 8]
                    ; cmp rax, rcx
                    ; setne al
                    ; movzx rax, al
                );
            }
        }
        self.store_from_rax(dest, 1);
        Ok(())
    }
}

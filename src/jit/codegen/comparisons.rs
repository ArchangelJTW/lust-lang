use super::*;

impl JitCompiler {
    pub(super) fn load_numeric_comparison_operands(
        &mut self,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> bool {
        if lhs_type == ValueType::Int && rhs_type == ValueType::Int {
            self.operands_rax_rcx(lhs, rhs);
            return false;
        }

        self.operands_numeric_xmm(lhs, rhs, lhs_type, rhs_type);
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
            dynasm!(self.ops ; .arch x64 ; ucomisd xmm0, xmm1 ; setb al ; setnp cl ; and al, cl ; movzx rax, al);
        } else {
            dynasm!(self.ops ; .arch x64 ; cmp rax, rcx ; setl al ; movzx rax, al);
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
            dynasm!(self.ops ; .arch x64 ; ucomisd xmm0, xmm1 ; setbe al ; setnp cl ; and al, cl ; movzx rax, al);
        } else {
            dynasm!(self.ops ; .arch x64 ; cmp rax, rcx ; setle al ; movzx rax, al);
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
            dynasm!(self.ops ; .arch x64 ; ucomisd xmm0, xmm1 ; seta al ; movzx rax, al);
        } else {
            dynasm!(self.ops ; .arch x64 ; cmp rax, rcx ; setg al ; movzx rax, al);
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
            dynasm!(self.ops ; .arch x64 ; ucomisd xmm0, xmm1 ; setae al ; movzx rax, al);
        } else {
            dynasm!(self.ops ; .arch x64 ; cmp rax, rcx ; setge al ; movzx rax, al);
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
            dynasm!(self.ops ; .arch x64 ; xor rax, rax);
        } else if lhs_type == ValueType::Float {
            self.load_numeric_comparison_operands(lhs, rhs, lhs_type, rhs_type);
            dynasm!(self.ops ; .arch x64 ; ucomisd xmm0, xmm1 ; sete al ; setnp cl ; and al, cl ; movzx rax, al);
        } else {
            if lhs_type == ValueType::Bool {
                self.operands_bool_eax_ecx(lhs, rhs);
                dynasm!(self.ops ; .arch x64 ; cmp eax, ecx ; sete al ; movzx rax, al);
            } else {
                self.operands_rax_rcx(lhs, rhs);
                dynasm!(self.ops ; .arch x64 ; cmp rax, rcx ; sete al ; movzx rax, al);
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
            dynasm!(self.ops ; .arch x64 ; mov rax, 1);
        } else if lhs_type == ValueType::Float {
            self.load_numeric_comparison_operands(lhs, rhs, lhs_type, rhs_type);
            dynasm!(self.ops ; .arch x64 ; ucomisd xmm0, xmm1 ; setne al ; setp cl ; or al, cl ; movzx rax, al);
        } else {
            if lhs_type == ValueType::Bool {
                self.operands_bool_eax_ecx(lhs, rhs);
                dynasm!(self.ops ; .arch x64 ; cmp eax, ecx ; setne al ; movzx rax, al);
            } else {
                self.operands_rax_rcx(lhs, rhs);
                dynasm!(self.ops ; .arch x64 ; cmp rax, rcx ; setne al ; movzx rax, al);
            }
        }
        self.store_from_rax(dest, 1);
        Ok(())
    }
}

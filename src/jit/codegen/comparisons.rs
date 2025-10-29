use super::*;
impl JitCompiler {
    pub(super) fn compile_lt(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        self.load_to_rax(lhs);
        dynasm!(self.ops ; push rax);
        let offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops ; mov rcx, [r12 + offset + 8]);
        dynasm!(self.ops
            ; pop rax
            ; cmp rax, rcx
            ; setl al
            ; movzx rax, al
        );
        self.store_from_rax(dest, 1);
        Ok(())
    }

    pub(super) fn compile_le(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        self.load_to_rax(lhs);
        dynasm!(self.ops ; push rax);
        let offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops ; mov rcx, [r12 + offset + 8]);
        dynasm!(self.ops
            ; pop rax
            ; cmp rax, rcx
            ; setle al
            ; movzx rax, al
        );
        self.store_from_rax(dest, 1);
        Ok(())
    }

    pub(super) fn compile_gt(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        self.load_to_rax(lhs);
        dynasm!(self.ops ; push rax);
        let offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops ; mov rcx, [r12 + offset + 8]);
        dynasm!(self.ops
            ; pop rax
            ; cmp rax, rcx
            ; setg al
            ; movzx rax, al
        );
        self.store_from_rax(dest, 1);
        Ok(())
    }

    pub(super) fn compile_ge(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        self.load_to_rax(lhs);
        dynasm!(self.ops ; push rax);
        let offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops ; mov rcx, [r12 + offset + 8]);
        dynasm!(self.ops
            ; pop rax
            ; cmp rax, rcx
            ; setge al
            ; movzx rax, al
        );
        self.store_from_rax(dest, 1);
        Ok(())
    }

    pub(super) fn compile_eq(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        self.load_to_rax(lhs);
        dynasm!(self.ops; push rax);
        let offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops; mov rcx, [r12 + offset + 8]);
        dynasm!(self.ops
            ; pop rax
            ; cmp rax, rcx
            ; sete al
            ; movzx rax, al
        );
        self.store_from_rax(dest, 1);
        Ok(())
    }

    pub(super) fn compile_ne(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        self.load_to_rax(lhs);
        dynasm!(self.ops; push rax);
        let offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops; mov rcx, [r12 + offset + 8]);
        dynasm!(self.ops
            ; pop rax
            ; cmp rax, rcx
            ; setne al
            ; movzx rax, al
        );
        self.store_from_rax(dest, 1);
        Ok(())
    }
}

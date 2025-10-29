use super::*;
impl JitCompiler {
    pub(super) fn load_to_rax(&mut self, vm_reg: u8) {
        let offset = (vm_reg as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops
            ; mov rax, [r12 + offset + 8]
        );
    }

    pub(super) fn load_to_rbx(&mut self, vm_reg: u8) {
        let offset = (vm_reg as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops
            ; mov rbx, [r12 + offset + 8]
        );
    }

    pub(super) fn store_from_rax(&mut self, vm_reg: u8, discriminant: u8) {
        let offset = (vm_reg as i32) * (mem::size_of::<Value>() as i32);
        let disc_i8 = discriminant as i8;
        dynasm!(self.ops
            ; mov BYTE [r12 + offset], disc_i8
            ; mov [r12 + offset + 8], rax
        );
    }
}

use super::*;
impl JitCompiler {
    pub(super) fn compile_neg(&mut self, dest: u8, src: u8) -> Result<()> {
        let src_offset = (src as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops
            ; mov al, [r12 + src_offset]
            ; cmp al, 3
            ; je >float_path
            ; mov rax, [r12 + src_offset + 8]
            ; neg rax
            ; jmp >store_int
            ; float_path:
            ; movsd xmm0, [r12 + src_offset + 8]
            ; mov rax, QWORD 0x8000000000000000u64 as _
            ; movq xmm1, rax
            ; xorpd xmm0, xmm1
        );
        self.store_xmm0_as_float(dest);
        dynasm!(self.ops
            ; jmp >done
            ; store_int:
        );
        self.store_from_rax(dest, 2);
        dynasm!(self.ops
            ; done:
        );
        Ok(())
    }

    pub(super) fn compile_and(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
        let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops
            ; mov al, [r12 + lhs_offset]
            ; cmp al, 0
            ; je >false_result
            ; cmp al, 1
            ; jne >true_for_lhs
            ; mov al, [r12 + lhs_offset + 8]
            ; test al, al
            ; jz >false_result
            ; true_for_lhs:
            ; mov al, [r12 + rhs_offset]
            ; cmp al, 0
            ; je >false_result
            ; cmp al, 1
            ; jne >true_result
            ; mov al, [r12 + rhs_offset + 8]
            ; test al, al
            ; jz >false_result
            ; true_result:
            ; mov rax, 1
            ; jmp >store
            ; false_result:
            ; xor eax, eax
            ; store:
        );
        self.store_from_rax(dest, 1);
        Ok(())
    }

    pub(super) fn compile_or(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
        let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops
            ; mov al, [r12 + lhs_offset]
            ; cmp al, 0
            ; je >check_rhs
            ; cmp al, 1
            ; jne >true_result
            ; mov al, [r12 + lhs_offset + 8]
            ; test al, al
            ; jnz >true_result
            ; check_rhs:
            ; mov al, [r12 + rhs_offset]
            ; cmp al, 0
            ; je >false_result
            ; cmp al, 1
            ; jne >true_result
            ; mov al, [r12 + rhs_offset + 8]
            ; test al, al
            ; jnz >true_result
            ; false_result:
            ; xor eax, eax
            ; jmp >store
            ; true_result:
            ; mov rax, 1
            ; store:
        );
        self.store_from_rax(dest, 1);
        Ok(())
    }

    pub(super) fn compile_not(&mut self, dest: u8, src: u8) -> Result<()> {
        let src_offset = (src as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops
            ; mov al, [r12 + src_offset]
            ; cmp al, 0
            ; je >true_result
            ; cmp al, 1
            ; jne >false_result
            ; mov al, [r12 + src_offset + 8]
            ; test al, al
            ; jz >true_result
            ; false_result:
            ; xor eax, eax
            ; jmp >store
            ; true_result:
            ; mov rax, 1
            ; store:
        );
        self.store_from_rax(dest, 1);
        Ok(())
    }

    pub(super) fn compile_concat(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
        let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        extern "C" {
            fn jit_concat_safe(
                vm_ptr: *mut crate::VM,
                left: *const Value,
                right: *const Value,
                out: *mut Value,
            ) -> u8;
        }

        dynasm!(self.ops
            ; mov rdi, r13
            ; lea rsi, [r12 + lhs_offset]
            ; lea rdx, [r12 + rhs_offset]
            ; lea rcx, [r12 + dest_offset]
            ; mov rax, QWORD jit_concat_safe as *const () as _
            ; call rax
            ; test al, al
            ; jz >fail
        );
        Ok(())
    }
}

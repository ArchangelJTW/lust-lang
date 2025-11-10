use super::*;
impl JitCompiler {
    pub(super) fn compile_neg(&mut self, dest: u8, src: u8) -> Result<()> {
        let src_offset = (src as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops
            ; mov al, [r12 + src_offset]
            ; cmp al, 3
            ; je >float_path
            ; mov rax, [r12 + src_offset + 8]
            ; neg rax
            ; mov BYTE [r12 + dest_offset], 2
            ; mov [r12 + dest_offset + 8], rax
            ; jmp >done
            ; float_path:
            ; movsd xmm0, [r12 + src_offset + 8]
            ; mov rax, QWORD 0x8000000000000000u64 as _
            ; movq xmm1, rax
            ; xorpd xmm0, xmm1
            ; mov BYTE [r12 + dest_offset], 3
            ; movsd [r12 + dest_offset + 8], xmm0
            ; done:
        );
        Ok(())
    }

    pub(super) fn compile_and(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
        let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
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
            ; mov BYTE [r12 + dest_offset], 1
            ; mov QWORD [r12 + dest_offset + 8], 0
            ; mov BYTE [r12 + dest_offset + 8], 1
            ; jmp >done
            ; false_result:
            ; mov BYTE [r12 + dest_offset], 1
            ; mov QWORD [r12 + dest_offset + 8], 0
            ; done:
        );
        Ok(())
    }

    pub(super) fn compile_or(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
        let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
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
            ; mov BYTE [r12 + dest_offset], 1
            ; mov QWORD [r12 + dest_offset + 8], 0
            ; jmp >done
            ; true_result:
            ; mov BYTE [r12 + dest_offset], 1
            ; mov QWORD [r12 + dest_offset + 8], 0
            ; mov BYTE [r12 + dest_offset + 8], 1
            ; done:
        );
        Ok(())
    }

    pub(super) fn compile_not(&mut self, dest: u8, src: u8) -> Result<()> {
        let src_offset = (src as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
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
            ; mov BYTE [r12 + dest_offset], 1
            ; mov QWORD [r12 + dest_offset + 8], 0
            ; jmp >done
            ; true_result:
            ; mov BYTE [r12 + dest_offset], 1
            ; mov QWORD [r12 + dest_offset + 8], 0
            ; mov BYTE [r12 + dest_offset + 8], 1
            ; done:
        );
        Ok(())
    }

    pub(super) fn compile_concat(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        let lhs_offset = (lhs as i32) * (mem::size_of::<Value>() as i32);
        let rhs_offset = (rhs as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        extern "C" {
            fn jit_concat_safe(left: *const Value, right: *const Value, out: *mut Value) -> u8;
        }

        dynasm!(self.ops
            ; lea rdi, [r12 + lhs_offset]
            ; lea rsi, [r12 + rhs_offset]
            ; lea rdx, [r12 + dest_offset]
            ; mov rax, QWORD jit_concat_safe as _
            ; call rax
            ; test al, al
            ; jz >fail
        );
        Ok(())
    }
}

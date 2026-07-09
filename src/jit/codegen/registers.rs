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
        match discriminant {
            1 => {
                extern "C" {
                    fn jit_replace_bool(dest: *mut Value, value: u8) -> u8;
                }
                dynasm!(self.ops
                    ; mov rsi, rax
                    ; lea rdi, [r12 + offset]
                    ; mov rax, QWORD jit_replace_bool as *const () as _
                    ; call rax
                );
            }
            2 => {
                extern "C" {
                    fn jit_replace_int(dest: *mut Value, value: crate::number::LustInt) -> u8;
                }
                dynasm!(self.ops
                    ; mov rsi, rax
                    ; lea rdi, [r12 + offset]
                    ; mov rax, QWORD jit_replace_int as *const () as _
                    ; call rax
                );
            }
            _ => unreachable!("unsupported scalar Value discriminant"),
        }
    }

    pub(super) fn store_xmm0_as_float(&mut self, vm_reg: u8) {
        let offset = (vm_reg as i32) * (mem::size_of::<Value>() as i32);
        extern "C" {
            fn jit_replace_float_bits(dest: *mut Value, bits: u64) -> u8;
        }
        dynasm!(self.ops
            ; movq rsi, xmm0
            ; lea rdi, [r12 + offset]
            ; mov rax, QWORD jit_replace_float_bits as *const () as _
            ; call rax
        );
    }
}

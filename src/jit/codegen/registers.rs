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
        let stored_type = match discriminant {
            1 => ValueType::Bool,
            2 => ValueType::Int,
            _ => unreachable!("unsupported scalar Value discriminant"),
        };
        if self.scalar_registers.get(&vm_reg) == Some(&stored_type) {
            dynasm!(self.ops
                ; mov QWORD [r12 + offset + 8], rax
            );
            return;
        }
        if self.scalar_registers.contains_key(&vm_reg) {
            dynasm!(self.ops
                ; mov BYTE [r12 + offset], discriminant as i8
                ; mov QWORD [r12 + offset + 8], rax
            );
            return;
        }
        let scalar_max_tag = ValueTag::Float.as_u8() as i8;
        match discriminant {
            1 => {
                extern "C" {
                    fn jit_replace_bool(dest: *mut Value, value: u8) -> u8;
                }
                dynasm!(self.ops
                    ; cmp BYTE [r12 + offset], scalar_max_tag
                    ; ja >replace_owned
                    ; mov BYTE [r12 + offset], discriminant as i8
                    ; mov QWORD [r12 + offset + 8], rax
                    ; jmp >done
                    ; replace_owned:
                    ; mov rsi, rax
                    ; lea rdi, [r12 + offset]
                    ; mov rax, QWORD jit_replace_bool as *const () as _
                    ; call rax
                    ; done:
                );
            }
            2 => {
                extern "C" {
                    fn jit_replace_int(dest: *mut Value, value: crate::number::LustInt) -> u8;
                }
                dynasm!(self.ops
                    ; cmp BYTE [r12 + offset], scalar_max_tag
                    ; ja >replace_owned
                    ; mov BYTE [r12 + offset], discriminant as i8
                    ; mov QWORD [r12 + offset + 8], rax
                    ; jmp >done
                    ; replace_owned:
                    ; mov rsi, rax
                    ; lea rdi, [r12 + offset]
                    ; mov rax, QWORD jit_replace_int as *const () as _
                    ; call rax
                    ; done:
                );
            }
            _ => unreachable!("unsupported scalar Value discriminant"),
        }
    }

    pub(super) fn store_xmm0_as_float(&mut self, vm_reg: u8) {
        let offset = (vm_reg as i32) * (mem::size_of::<Value>() as i32);
        let float_tag = ValueTag::Float.as_u8() as i8;
        if self.scalar_registers.get(&vm_reg) == Some(&ValueType::Float) {
            dynasm!(self.ops
                ; movq QWORD [r12 + offset + 8], xmm0
            );
            return;
        }
        if self.scalar_registers.contains_key(&vm_reg) {
            dynasm!(self.ops
                ; mov BYTE [r12 + offset], float_tag
                ; movq QWORD [r12 + offset + 8], xmm0
            );
            return;
        }
        let scalar_max_tag = ValueTag::Float.as_u8() as i8;
        extern "C" {
            fn jit_replace_float_bits(dest: *mut Value, bits: u64) -> u8;
        }
        dynasm!(self.ops
            ; movq rsi, xmm0
            ; cmp BYTE [r12 + offset], scalar_max_tag
            ; ja >replace_owned
            ; mov BYTE [r12 + offset], float_tag
            ; mov QWORD [r12 + offset + 8], rsi
            ; jmp >done
            ; replace_owned:
            ; lea rdi, [r12 + offset]
            ; mov rax, QWORD jit_replace_float_bits as *const () as _
            ; call rax
            ; done:
        );
    }
}

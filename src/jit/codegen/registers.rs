use super::*;
impl JitCompiler {
    pub(super) fn load_to_rax(&mut self, vm_reg: u8) {
        let offset = (vm_reg as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops
            ; .arch x64
            ; mov rax, [r12 + offset + 8]
        );
    }

    pub(super) fn load_to_rbx(&mut self, vm_reg: u8) {
        let offset = (vm_reg as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops
            ; .arch x64
            ; mov rbx, [r12 + offset + 8]
        );
    }

    // ── Operands ─────────────────────────────────────────────────────────
    //
    // An op reads its operands through these, first thing, so a payload
    // the previous op's store left in rax / xmm0 is used in place. Each
    // helper that writes rax or xmm0 forgets what was there, so a second
    // operand read after it loads from memory.

    /// rax = int payload of registers[vm_reg].
    pub(super) fn operand_rax(&mut self, vm_reg: u8) {
        if self.hot_rax_in == Some(vm_reg) {
            return;
        }
        self.hot_rax_in = None;
        self.load_to_rax(vm_reg);
    }

    /// rbx = int payload of registers[vm_reg]; rax is untouched.
    pub(super) fn operand_rbx(&mut self, vm_reg: u8) {
        if self.hot_rax_in == Some(vm_reg) {
            dynasm!(self.ops ; .arch x64 ; mov rbx, rax);
            return;
        }
        self.load_to_rbx(vm_reg);
    }

    /// rcx = int payload of registers[vm_reg]; rax is untouched.
    pub(super) fn operand_rcx(&mut self, vm_reg: u8) {
        let offset = (vm_reg as i32) * (mem::size_of::<Value>() as i32);
        if self.hot_rax_in == Some(vm_reg) {
            dynasm!(self.ops ; .arch x64 ; mov rcx, rax);
            return;
        }
        dynasm!(self.ops ; .arch x64 ; mov rcx, [r12 + offset + 8]);
    }

    /// eax = bool payload of registers[vm_reg] as 0 or 1 (only the low
    /// byte of a Bool's payload is defined in memory; a store leaves the
    /// whole register defined).
    pub(super) fn operand_bool_eax(&mut self, vm_reg: u8) {
        let offset = (vm_reg as i32) * (mem::size_of::<Value>() as i32);
        if self.hot_rax_in == Some(vm_reg) {
            return;
        }
        self.hot_rax_in = None;
        dynasm!(self.ops ; .arch x64 ; movzx eax, BYTE [r12 + offset + 8]);
    }

    /// ecx = bool payload of registers[vm_reg] as 0 or 1; rax is untouched.
    pub(super) fn operand_bool_ecx(&mut self, vm_reg: u8) {
        let offset = (vm_reg as i32) * (mem::size_of::<Value>() as i32);
        if self.hot_rax_in == Some(vm_reg) {
            dynasm!(self.ops ; .arch x64 ; mov ecx, eax);
            return;
        }
        dynasm!(self.ops ; .arch x64 ; movzx ecx, BYTE [r12 + offset + 8]);
    }

    /// xmm0 = float payload of registers[vm_reg].
    pub(super) fn operand_xmm0(&mut self, vm_reg: u8) {
        let offset = (vm_reg as i32) * (mem::size_of::<Value>() as i32);
        if self.hot_xmm0_in == Some(vm_reg) {
            return;
        }
        self.hot_xmm0_in = None;
        dynasm!(self.ops ; .arch x64 ; movsd xmm0, [r12 + offset + 8]);
    }

    /// xmm1 = float payload of registers[vm_reg]; xmm0 is untouched.
    pub(super) fn operand_xmm1(&mut self, vm_reg: u8) {
        let offset = (vm_reg as i32) * (mem::size_of::<Value>() as i32);
        if self.hot_xmm0_in == Some(vm_reg) {
            dynasm!(self.ops ; .arch x64 ; movapd xmm1, xmm0);
            return;
        }
        dynasm!(self.ops ; .arch x64 ; movsd xmm1, [r12 + offset + 8]);
    }

    /// xmm0 = float(int payload of registers[vm_reg]); clobbers rax when
    /// the payload is not already there.
    pub(super) fn operand_int_as_xmm0(&mut self, vm_reg: u8) {
        self.operand_rax(vm_reg);
        self.hot_xmm0_in = None;
        dynasm!(self.ops ; .arch x64 ; cvtsi2sd xmm0, rax);
    }

    /// xmm1 = float(int payload of registers[vm_reg]); clobbers rax when
    /// the payload is not already there. xmm0 is untouched.
    pub(super) fn operand_int_as_xmm1(&mut self, vm_reg: u8) {
        self.operand_rax(vm_reg);
        dynasm!(self.ops ; .arch x64 ; cvtsi2sd xmm1, rax);
    }

    /// rax, rbx = int payloads of registers[lhs], registers[rhs]. The one
    /// already in rax is read first, so it is copied rather than lost to
    /// the other's load.
    pub(super) fn operands_rax_rbx(&mut self, lhs: u8, rhs: u8) {
        if self.hot_rax_in == Some(rhs) && lhs != rhs {
            self.operand_rbx(rhs);
            self.operand_rax(lhs);
        } else {
            self.operand_rax(lhs);
            self.operand_rbx(rhs);
        }
    }

    /// rax, rcx = int payloads of registers[lhs], registers[rhs] (see
    /// `operands_rax_rbx`).
    pub(super) fn operands_rax_rcx(&mut self, lhs: u8, rhs: u8) {
        if self.hot_rax_in == Some(rhs) && lhs != rhs {
            self.operand_rcx(rhs);
            self.operand_rax(lhs);
        } else {
            self.operand_rax(lhs);
            self.operand_rcx(rhs);
        }
    }

    /// eax, ecx = bool payloads of registers[lhs], registers[rhs] as 0 or
    /// 1 (see `operands_rax_rbx`).
    pub(super) fn operands_bool_eax_ecx(&mut self, lhs: u8, rhs: u8) {
        if self.hot_rax_in == Some(rhs) && lhs != rhs {
            self.operand_bool_ecx(rhs);
            self.operand_bool_eax(lhs);
        } else {
            self.operand_bool_eax(lhs);
            self.operand_bool_ecx(rhs);
        }
    }

    /// xmm0, xmm1 = registers[lhs], registers[rhs] as floats, converting
    /// ints. An rhs still in rax or xmm0 is read first (its conversion
    /// or copy leaves xmm0 free for the lhs).
    pub(super) fn operands_numeric_xmm(
        &mut self,
        lhs: u8,
        rhs: u8,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) {
        let rhs_hot = self.hot_rax_in == Some(rhs) || self.hot_xmm0_in == Some(rhs);
        if rhs_hot && lhs != rhs {
            self.operand_numeric_xmm1(rhs, rhs_type);
            self.operand_numeric_xmm0(lhs, lhs_type);
        } else {
            self.operand_numeric_xmm0(lhs, lhs_type);
            self.operand_numeric_xmm1(rhs, rhs_type);
        }
    }

    /// rax, rbx = dividend and divisor for `idiv`, both loaded, dividend
    /// first. Under Rosetta an `idiv` fed from a register the previous op
    /// computed into, or with its two loads the other way round, runs a
    /// `(i * j) % 7` loop 22 → 28 ms; next to the divide the two loads
    /// cost nothing on real hardware, so the divide keeps the plain shape.
    pub(super) fn operands_idiv(&mut self, lhs: u8, rhs: u8) {
        self.hot_rax_in = None;
        self.load_to_rax(lhs);
        self.load_to_rbx(rhs);
    }

    /// xmm0 = registers[vm_reg] as a float, converting an int.
    pub(super) fn operand_numeric_xmm0(&mut self, vm_reg: u8, ty: ValueType) {
        if ty == ValueType::Int {
            self.operand_int_as_xmm0(vm_reg);
        } else {
            self.operand_xmm0(vm_reg);
        }
    }

    /// xmm1 = registers[vm_reg] as a float, converting an int.
    pub(super) fn operand_numeric_xmm1(&mut self, vm_reg: u8, ty: ValueType) {
        if ty == ValueType::Int {
            self.operand_int_as_xmm1(vm_reg);
        } else {
            self.operand_xmm1(vm_reg);
        }
    }

    pub(super) fn store_from_rax(&mut self, vm_reg: u8, discriminant: u8) {
        let offset = (vm_reg as i32) * (mem::size_of::<Value>() as i32);
        let stored_type = match discriminant {
            1 => ValueType::Bool,
            2 => ValueType::Int,
            _ => unreachable!("unsupported scalar Value discriminant"),
        };
        // A Bool arrives as a byte in al; whatever the upper bits hold (an
        // 8-byte payload load of a Bool, say) must not reach the payload
        // or the `u8` helper argument, which the ABI lets the callee read
        // as a full register.
        if stored_type == ValueType::Bool {
            dynasm!(self.ops ; .arch x64 ; movzx eax, al);
        }
        if self.scalar_registers.get(&vm_reg) == Some(&stored_type) {
            dynasm!(self.ops
                ; .arch x64
                ; mov QWORD [r12 + offset + 8], rax
            );
            self.hot_rax = Some(vm_reg);
            return;
        }
        if self.scalar_registers.contains_key(&vm_reg) {
            dynasm!(self.ops
                ; .arch x64
                ; mov BYTE [r12 + offset], discriminant as i8
                ; mov QWORD [r12 + offset + 8], rax
            );
            self.hot_rax = Some(vm_reg);
            return;
        }
        // The slow path below is a call: rax is not the payload afterwards.
        let scalar_max_tag = ValueTag::Float.as_u8() as i8;
        match discriminant {
            1 => {
                unsafe extern "C" {
                    fn jit_replace_bool(dest: *mut Value, value: u8) -> u8;
                }
                dynasm!(self.ops
                    ; .arch x64
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
                unsafe extern "C" {
                    fn jit_replace_int(dest: *mut Value, value: crate::number::LustInt) -> u8;
                }
                dynasm!(self.ops
                    ; .arch x64
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
                ; .arch x64
                ; movq QWORD [r12 + offset + 8], xmm0
            );
            self.hot_xmm0 = Some(vm_reg);
            return;
        }
        if self.scalar_registers.contains_key(&vm_reg) {
            dynasm!(self.ops
                ; .arch x64
                ; mov BYTE [r12 + offset], float_tag
                ; movq QWORD [r12 + offset + 8], xmm0
            );
            self.hot_xmm0 = Some(vm_reg);
            return;
        }
        let scalar_max_tag = ValueTag::Float.as_u8() as i8;
        unsafe extern "C" {
            fn jit_replace_float_bits(dest: *mut Value, bits: u64) -> u8;
        }
        dynasm!(self.ops
            ; .arch x64
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

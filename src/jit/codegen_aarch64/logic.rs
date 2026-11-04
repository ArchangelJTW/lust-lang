use super::*;
impl JitCompiler {
    pub(super) fn compile_neg(&mut self, dest: u8, src: u8) -> Result<()> {
        // With the operand's type known (a pin, or scalar tracking) emit just
        // that path. A pinned destination always has a known operand type,
        // and the runtime-dispatched version below would store both types.
        let known = self
            .active_pin(src)
            .map(|pin| pin.ty)
            .or_else(|| self.scalar_registers.get(&src).copied())
            .or_else(|| self.active_pin(dest).map(|pin| pin.ty));
        match known {
            Some(ValueType::Int) => {
                self.load_payload(0, src);
                dynasm!(self.ops ; .arch aarch64 ; neg x0, x0);
                self.store_from_x0(dest, ValueTag::Int.as_u8());
                return Ok(());
            }
            Some(ValueType::Float) => {
                self.load_payload_f(0, src);
                dynasm!(self.ops ; .arch aarch64 ; fneg d0, d0);
                self.store_d0_as_float(dest);
                return Ok(());
            }
            _ => {}
        }
        let float_tag = ValueTag::Float.as_u8() as u32;
        self.load_tag(0, src);
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp w0, #float_tag
            ; b.eq >float_path
        );
        self.load_payload(0, src);
        dynasm!(self.ops
            ; .arch aarch64
            ; neg x0, x0
            ; b >store_int
            ; float_path:
        );
        self.load_payload_f(0, src);
        dynasm!(self.ops
            ; .arch aarch64
            ; fneg d0, d0
        );
        self.store_d0_as_float(dest);
        dynasm!(self.ops
            ; .arch aarch64
            ; b >neg_done
            ; store_int:
        );
        self.store_from_x0(dest, ValueTag::Int.as_u8());
        dynasm!(self.ops
            ; .arch aarch64
            ; neg_done:
        );
        Ok(())
    }

    /// w0 = truthiness of registers[reg] for Nil/Bool/other-scalar tags:
    /// Nil → 0, Bool → payload, anything else → 1. Branches to `>false_result`
    /// when falsy and falls through when truthy.
    fn emit_branch_if_falsy(&mut self, reg: u8) {
        self.load_tag(0, reg);
        dynasm!(self.ops
            ; .arch aarch64
            ; cbz w0, >false_result
            ; cmp w0, 1
            ; b.ne >truthy
        );
        self.load_bool_payload(0, reg);
        dynasm!(self.ops
            ; .arch aarch64
            ; cbz w0, >false_result
            ; truthy:
        );
    }

    pub(super) fn compile_and(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        self.emit_branch_if_falsy(lhs);
        self.emit_branch_if_falsy(rhs);
        dynasm!(self.ops
            ; .arch aarch64
            ; movz x0, 1
            ; b >store
            ; false_result:
            ; mov x0, xzr
            ; store:
        );
        self.store_from_x0(dest, ValueTag::Bool.as_u8());
        Ok(())
    }

    pub(super) fn compile_or(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        // lhs truthy → true; lhs falsy → evaluate rhs.
        self.load_tag(0, lhs);
        dynasm!(self.ops
            ; .arch aarch64
            ; cbz w0, >check_rhs
            ; cmp w0, 1
            ; b.ne >true_result
        );
        self.load_bool_payload(0, lhs);
        dynasm!(self.ops
            ; .arch aarch64
            ; cbnz w0, >true_result
            ; check_rhs:
        );
        self.emit_branch_if_falsy(rhs);
        dynasm!(self.ops
            ; .arch aarch64
            ; true_result:
            ; movz x0, 1
            ; b >store
            ; false_result:
            ; mov x0, xzr
            ; store:
        );
        self.store_from_x0(dest, ValueTag::Bool.as_u8());
        Ok(())
    }

    pub(super) fn compile_not(&mut self, dest: u8, src: u8) -> Result<()> {
        self.emit_branch_if_falsy(src);
        dynasm!(self.ops
            ; .arch aarch64
            ; mov x0, xzr
            ; b >store
            ; false_result:
            ; movz x0, 1
            ; store:
        );
        self.store_from_x0(dest, ValueTag::Bool.as_u8());
        Ok(())
    }

    pub(super) fn compile_concat(&mut self, dest: u8, lhs: u8, rhs: u8) -> Result<()> {
        unsafe extern "C" {
            fn jit_concat_safe(
                vm_ptr: *mut crate::VM,
                left: *const Value,
                right: *const Value,
                out: *mut Value,
            ) -> u8;
        }

        dynasm!(self.ops ; .arch aarch64 ; mov x0, x20);
        self.emit_reg_addr(1, lhs);
        self.emit_reg_addr(2, rhs);
        self.emit_reg_addr(3, dest);
        self.emit_call(jit_concat_safe as *const ());
        self.emit_fail_if_w0_zero();
        Ok(())
    }

    fn emit_eval_truthiness(&mut self, cond: u8) {
        if let Some(pin) = self.active_pin(cond) {
            match pin.ty {
                ValueType::Bool => {
                    dynasm!(self.ops
                        ; .arch aarch64
                        ; uxtb w0, W(pin.reg)
                    );
                }
                _ => {
                    dynasm!(self.ops
                        ; .arch aarch64
                        ; movz w0, 1
                    );
                }
            }
            return;
        }
        let bool_tag = ValueTag::Bool.as_u8() as u32;
        let scalar_max_tag = ValueTag::Float.as_u8() as u32;
        unsafe extern "C" {
            fn jit_value_is_truthy(value_ptr: *const Value) -> u8;
        }
        if self.hot_x0_in == Some(cond)
            && self.scalar_registers.get(&cond) == Some(&ValueType::Bool)
        {
            // Already in w0 as 0 or 1!
        } else if self.scalar_registers.get(&cond) == Some(&ValueType::Bool) {
            self.load_bool_payload(0, cond);
        } else {
            self.load_tag(0, cond);
            dynasm!(self.ops
                ; .arch aarch64
                ; cmp w0, #scalar_max_tag
                ; b.hi >generic_truthiness
                ; cmp w0, #bool_tag
                ; b.eq >load_bool
                ; cmp w0, 0
                ; cset w0, ne
                ; b >truthiness_ready
                ; load_bool:
            );
            self.load_bool_payload(0, cond);
            dynasm!(self.ops
                ; .arch aarch64
                ; b >truthiness_ready
                ; generic_truthiness:
            );
            self.emit_reg_addr(0, cond);
            self.emit_call(jit_value_is_truthy as *const ());
            dynasm!(self.ops
                ; .arch aarch64
                ; and w0, w0, 0xff
                ; truthiness_ready:
            );
        }
    }

    pub(super) fn compile_select(
        &mut self,
        dest: u8,
        cond: u8,
        on_true: &Value,
        on_false: &Value,
    ) -> Result<()> {
        self.emit_eval_truthiness(cond);
        match (on_true, on_false) {
            (Value::Int(t), Value::Int(f)) => {
                if self.cached_select_imms != Some((*t, *f)) {
                    self.emit_mov_imm64(14, *t as u64);
                    self.emit_mov_imm64(15, *f as u64);
                    self.cached_select_imms = Some((*t, *f));
                }
                match self.direct_dest_x(dest) {
                    Some(d) => {
                        dynasm!(self.ops
                            ; .arch aarch64
                            ; cmp w0, #0
                            ; csel X(d), x14, x15, ne
                        );
                    }
                    None => {
                        dynasm!(self.ops
                            ; .arch aarch64
                            ; cmp w0, #0
                            ; csel x0, x14, x15, ne
                        );
                        self.store_from_x0(dest, ValueTag::Int.as_u8());
                    }
                }
            }
            (Value::Bool(t), Value::Bool(f)) => {
                let t_val = u32::from(*t);
                let f_val = u32::from(*f);
                match self.direct_dest_x(dest) {
                    Some(d) => {
                        dynasm!(self.ops
                            ; .arch aarch64
                            ; movz x9, #t_val
                            ; movz x10, #f_val
                            ; cmp w0, #0
                            ; csel X(d), x9, x10, ne
                        );
                    }
                    None => {
                        dynasm!(self.ops
                            ; .arch aarch64
                            ; movz x9, #t_val
                            ; movz x10, #f_val
                            ; cmp w0, #0
                            ; csel x0, x9, x10, ne
                        );
                        self.store_from_x0(dest, ValueTag::Bool.as_u8());
                    }
                }
            }
            (Value::Float(t), Value::Float(f)) => {
                self.emit_mov_imm64(9, t.to_bits());
                self.emit_mov_imm64(10, f.to_bits());
                self.hot_d0_in = None;
                match self.direct_dest_d(dest) {
                    Some(d) => {
                        dynasm!(self.ops
                            ; .arch aarch64
                            ; fmov d1, x9
                            ; fmov d2, x10
                            ; cmp w0, #0
                            ; fcsel D(d), d1, d2, ne
                        );
                    }
                    None => {
                        dynasm!(self.ops
                            ; .arch aarch64
                            ; fmov d1, x9
                            ; fmov d2, x10
                            ; cmp w0, #0
                            ; fcsel d0, d1, d2, ne
                        );
                        self.store_d0_as_float(dest);
                    }
                }
            }
            _ => {
                return Err(crate::LustError::RuntimeError {
                    message: "Select with unsupported or mismatched types".to_string(),
                });
            }
        }
        Ok(())
    }

    pub(super) fn compile_select_bit_zero(
        &mut self,
        dest: u8,
        src: u8,
        mask: u64,
        on_zero: &Value,
        on_nonzero: &Value,
    ) -> Result<()> {
        let src_reg = self.operand_x(src, 11);
        match (on_zero, on_nonzero) {
            (Value::Int(t), Value::Int(f)) => {
                if self.cached_select_imms != Some((*t, *f)) {
                    self.emit_mov_imm64(14, *t as u64);
                    self.emit_mov_imm64(15, *f as u64);
                    self.cached_select_imms = Some((*t, *f));
                }
                if mask == 1 {
                    dynasm!(self.ops
                        ; .arch aarch64
                        ; ands xzr, X(src_reg), #1
                    );
                } else {
                    self.emit_mov_imm64(12, mask);
                    dynasm!(self.ops
                        ; .arch aarch64
                        ; ands xzr, X(src_reg), x12
                    );
                }
                match self.direct_dest_x(dest) {
                    Some(d) => {
                        dynasm!(self.ops
                            ; .arch aarch64
                            ; csel X(d), x14, x15, eq
                        );
                    }
                    None => {
                        dynasm!(self.ops
                            ; .arch aarch64
                            ; csel x0, x14, x15, eq
                        );
                        self.store_from_x0(dest, ValueTag::Int.as_u8());
                    }
                }
            }
            _ => {
                return Err(crate::LustError::RuntimeError {
                    message: "SelectBitZero with unsupported types".to_string(),
                });
            }
        }
        Ok(())
    }
}

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
}

use super::*;

/// Largest unsigned immediate accepted by `add`/`sub`/`ldrb`/`strb` (12 bits).
const IMM12_MAX: i32 = 4095;

pub(super) fn value_size() -> i32 {
    mem::size_of::<Value>() as i32
}

pub(super) fn reg_offset(vm_reg: u8) -> i32 {
    (vm_reg as i32) * value_size()
}

impl JitCompiler {
    /// The pin for a VM register, if pins are in effect at this point. In
    /// function code the machine register is only known to be current
    /// where the static environment proves the register's type (every
    /// typed write updates it; anything else is in memory, which the
    /// write-through pins keep current).
    pub(super) fn active_pin(&self, vm_reg: u8) -> Option<pins::Pin> {
        if !self.pin_active {
            return None;
        }
        let pin = self.pins.get(&vm_reg).copied()?;
        if self.function_mode && self.scalar_registers.get(&vm_reg) != Some(&pin.ty) {
            return None;
        }
        Some(pin)
    }

    /// The pin a write to a VM register must update, whatever the static
    /// environment knows (see `active_pin`).
    pub(super) fn pin_for_write(&self, vm_reg: u8) -> Option<pins::Pin> {
        if self.pin_active {
            self.pins.get(&vm_reg).copied()
        } else {
            None
        }
    }

    fn pin_tag(pin: &pins::Pin) -> u8 {
        match pin.ty {
            ValueType::Bool => ValueTag::Bool.as_u8(),
            ValueType::Int => ValueTag::Int.as_u8(),
            ValueType::Float => ValueTag::Float.as_u8(),
            _ => unreachable!("only scalars are pinned"),
        }
    }

    pub(super) fn mark_dirty(&mut self, vm_reg: u8) {
        if !self.dirty_pins.contains(&vm_reg) {
            self.dirty_pins.push(vm_reg);
        }
    }

    /// Write a pinned register's payload to the VM register array. The tag
    /// is never written: for a proven pin memory already holds it, and for
    /// a not-yet-guarded carried pin the payload bits are the ones loaded
    /// from memory, so the store changes nothing.
    pub(super) fn emit_pin_writeback(&mut self, vm_reg: u8, pin: pins::Pin) {
        let offset = (reg_offset(vm_reg) + 8) as u32;
        match pin.ty {
            ValueType::Float => dynasm!(self.ops ; .arch aarch64 ; str D(pin.reg), [x19, #offset]),
            _ => dynasm!(self.ops ; .arch aarch64 ; str X(pin.reg), [x19, #offset]),
        }
    }

    /// Load a carried pin's payload bits from the VM register array (loop
    /// prologue). No type check: see `emit_pin_writeback`.
    pub(super) fn emit_pin_load(&mut self, vm_reg: u8, pin: pins::Pin) {
        let offset = (reg_offset(vm_reg) + 8) as u32;
        match pin.ty {
            ValueType::Float => dynasm!(self.ops ; .arch aarch64 ; ldr D(pin.reg), [x19, #offset]),
            // Bools are read through the low byte; the full word keeps the
            // write-back bit-identical.
            _ => dynasm!(self.ops ; .arch aarch64 ; ldr X(pin.reg), [x19, #offset]),
        }
    }

    /// Flush every carried pin that may be newer than memory.
    pub(super) fn flush_dirty_pins(&mut self) {
        let dirty = mem::take(&mut self.dirty_pins);
        for vm_reg in dirty {
            if let Some(pin) = self.pins.get(&vm_reg).copied() {
                self.emit_pin_writeback(vm_reg, pin);
            }
        }
    }

    /// Write back every carried pin (trace exit, after inline frames are
    /// unwound so x19 is the trace's own register array again).
    pub(super) fn emit_writeback_carried_pins(&mut self) {
        let mut carried: Vec<(u8, pins::Pin)> = self
            .pins
            .iter()
            .filter(|(_, pin)| pin.class == pins::PinClass::Carried)
            .map(|(r, pin)| (*r, *pin))
            .collect();
        carried.sort_by_key(|(r, _)| *r);
        for (vm_reg, pin) in carried {
            self.emit_pin_writeback(vm_reg, pin);
        }
    }

    // ── Direct operands ───────────────────────────────────────────────────
    //
    // Ops read pinned registers in place instead of copying them through
    // x0/d0, and write a carried pin in place instead of through x0/d0 and
    // a store. (Write-through pins still go through the store path so
    // memory is updated.)

    /// X register holding the int/bool payload of registers[vm_reg]: the
    /// pin itself, or `scratch` after loading into it.
    pub(super) fn operand_x(&mut self, vm_reg: u8, scratch: u8) -> u8 {
        if let Some(pin) = self.active_pin(vm_reg)
            && pin.ty != ValueType::Float
        {
            return pin.reg;
        }
        // Still in x0 from the previous op's store.
        if self.hot_x0_in == Some(vm_reg) {
            if scratch != 0 {
                dynasm!(self.ops ; .arch aarch64 ; mov X(scratch), x0);
            }
            return scratch;
        }
        self.load_payload(scratch, vm_reg);
        scratch
    }

    /// D register holding the float payload of registers[vm_reg].
    pub(super) fn operand_d(&mut self, vm_reg: u8, scratch: u8) -> u8 {
        if let Some(pin) = self.active_pin(vm_reg)
            && pin.ty == ValueType::Float
        {
            return pin.reg;
        }
        if self.hot_d0_in == Some(vm_reg) {
            if scratch != 0 {
                dynasm!(self.ops ; .arch aarch64 ; fmov D(scratch), d0);
            }
            return scratch;
        }
        self.load_payload_f(scratch, vm_reg);
        scratch
    }

    /// D register holding registers[vm_reg]'s int payload converted to float.
    pub(super) fn operand_int_as_d(&mut self, vm_reg: u8, scratch: u8) -> u8 {
        if self.hot_x0_in == Some(vm_reg) {
            self.clobber_d(scratch);
            dynasm!(self.ops ; .arch aarch64 ; scvtf D(scratch), x0);
            return scratch;
        }
        self.load_payload_int_as_f(scratch, vm_reg);
        scratch
    }

    /// D register holding registers[vm_reg] as a float, converting an int.
    pub(super) fn operand_numeric_d(&mut self, vm_reg: u8, ty: ValueType, scratch: u8) -> u8 {
        if ty == ValueType::Int {
            self.operand_int_as_d(vm_reg, scratch)
        } else {
            self.operand_d(vm_reg, scratch)
        }
    }

    /// The X register a result for registers[vm_reg] can be written into
    /// directly (a carried int/bool pin), marking it dirty.
    pub(super) fn direct_dest_x(&mut self, vm_reg: u8) -> Option<u8> {
        let pin = self.active_pin(vm_reg)?;
        if pin.ty == ValueType::Float || pin.class != pins::PinClass::Carried {
            return None;
        }
        self.mark_dirty(vm_reg);
        Some(pin.reg)
    }

    /// The D register a float result for registers[vm_reg] can be written
    /// into directly (a carried float pin), marking it dirty.
    pub(super) fn direct_dest_d(&mut self, vm_reg: u8) -> Option<u8> {
        let pin = self.active_pin(vm_reg)?;
        if pin.ty != ValueType::Float || pin.class != pins::PinClass::Carried {
            return None;
        }
        self.mark_dirty(vm_reg);
        Some(pin.reg)
    }

    // ── Immediates ────────────────────────────────────────────────────────

    /// X(reg) = value, via movz + movk (skipping zero halves after the first).
    /// Code is about to write X(reg): what x0 held for this op is gone.
    fn clobber_x(&mut self, reg: u8) {
        if reg == 0 {
            self.hot_x0_in = None;
        }
    }

    fn clobber_d(&mut self, reg: u8) {
        if reg == 0 {
            self.hot_d0_in = None;
        }
    }

    pub(super) fn emit_mov_imm64(&mut self, reg: u8, value: u64) {
        self.clobber_x(reg);
        let h0 = (value & 0xffff) as u32;
        let h1 = ((value >> 16) & 0xffff) as u32;
        let h2 = ((value >> 32) & 0xffff) as u32;
        let h3 = ((value >> 48) & 0xffff) as u32;
        dynasm!(self.ops ; .arch aarch64 ; movz X(reg), #h0);
        if h1 != 0 {
            dynasm!(self.ops ; .arch aarch64 ; movk X(reg), #h1, lsl #16);
        }
        if h2 != 0 {
            dynasm!(self.ops ; .arch aarch64 ; movk X(reg), #h2, lsl #32);
        }
        if h3 != 0 {
            dynasm!(self.ops ; .arch aarch64 ; movk X(reg), #h3, lsl #48);
        }
    }

    /// W(reg) = value (zero-extended into the full X register).
    pub(super) fn emit_mov_imm32(&mut self, reg: u8, value: u32) {
        self.clobber_x(reg);
        let lo = value & 0xffff;
        let hi = value >> 16;
        dynasm!(self.ops ; .arch aarch64 ; movz W(reg), #lo);
        if hi != 0 {
            dynasm!(self.ops ; .arch aarch64 ; movk W(reg), #hi, lsl #16);
        }
    }

    /// W(reg) = value as a 32-bit two's complement immediate.
    pub(super) fn emit_mov_imm_i32(&mut self, reg: u8, value: i32) {
        self.clobber_x(reg);
        self.emit_mov_imm32(reg, value as u32);
    }

    /// X(dst) = X(src) + imm for any non-negative imm. Uses x12 when the
    /// immediate does not fit the 12-bit add encoding. `dst`/`src` may be
    /// the same register but must not be x12.
    pub(super) fn emit_add_imm(&mut self, dst: u8, src: u8, imm: i32) {
        self.clobber_x(dst);
        debug_assert!(imm >= 0);
        debug_assert!(dst != 12 && src != 12);
        if imm == 0 {
            if dst != src {
                dynasm!(self.ops ; .arch aarch64 ; mov X(dst), X(src));
            }
        } else if imm <= IMM12_MAX {
            let imm = imm as u32;
            dynasm!(self.ops ; .arch aarch64 ; add XSP(dst), XSP(src), #imm);
        } else {
            self.emit_mov_imm32(12, imm as u32);
            dynasm!(self.ops ; .arch aarch64 ; add X(dst), X(src), x12);
        }
    }

    /// X(dst) = X(src) - imm for any non-negative imm (see `emit_add_imm`).
    pub(super) fn emit_sub_imm(&mut self, dst: u8, src: u8, imm: i32) {
        debug_assert!(imm >= 0);
        debug_assert!(dst != 12 && src != 12);
        if imm == 0 {
            if dst != src {
                dynasm!(self.ops ; .arch aarch64 ; mov X(dst), X(src));
            }
        } else if imm <= IMM12_MAX {
            let imm = imm as u32;
            dynasm!(self.ops ; .arch aarch64 ; sub XSP(dst), XSP(src), #imm);
        } else {
            self.emit_mov_imm32(12, imm as u32);
            dynasm!(self.ops ; .arch aarch64 ; sub X(dst), X(src), x12);
        }
    }

    /// sp = sp - imm (imm must be a multiple of 16).
    pub(super) fn emit_sub_sp(&mut self, imm: i32) {
        debug_assert!(imm >= 0 && imm % 16 == 0);
        if imm == 0 {
        } else if imm <= IMM12_MAX {
            let imm = imm as u32;
            dynasm!(self.ops ; .arch aarch64 ; sub sp, sp, #imm);
        } else {
            self.emit_mov_imm32(12, imm as u32);
            dynasm!(self.ops ; .arch aarch64 ; sub sp, sp, x12);
        }
    }

    // ── Register-array addressing ─────────────────────────────────────────

    /// X(dst) = &registers[vm_reg]
    pub(super) fn emit_reg_addr(&mut self, dst: u8, vm_reg: u8) {
        self.clobber_x(dst);
        self.emit_add_imm(dst, 19, reg_offset(vm_reg));
    }

    /// W(w) = discriminant byte of registers[vm_reg]. For a pinned register
    /// the tag is a compile-time constant (reads only happen after the type
    /// is proven; stores use `load_tag_from_memory`).
    pub(super) fn load_tag(&mut self, w: u8, vm_reg: u8) {
        self.clobber_x(w);
        if let Some(pin) = self.active_pin(vm_reg) {
            let tag = Self::pin_tag(&pin) as u32;
            dynasm!(self.ops ; .arch aarch64 ; movz W(w), #tag);
            return;
        }
        self.load_tag_from_memory(w, vm_reg);
    }

    /// W(w) = the discriminant byte actually in memory, pinned or not. The
    /// first write to a write-through pin may replace an owned value.
    pub(super) fn load_tag_from_memory(&mut self, w: u8, vm_reg: u8) {
        self.clobber_x(w);
        let offset = reg_offset(vm_reg);
        if offset <= IMM12_MAX {
            let offset = offset as u32;
            dynasm!(self.ops ; .arch aarch64 ; ldrb W(w), [x19, #offset]);
        } else {
            self.emit_reg_addr(11, vm_reg);
            dynasm!(self.ops ; .arch aarch64 ; ldrb W(w), [x11]);
        }
    }

    /// W(w) = low payload byte of registers[vm_reg] (bool payload)
    pub(super) fn load_bool_payload(&mut self, w: u8, vm_reg: u8) {
        self.clobber_x(w);
        if let Some(pin) = self.active_pin(vm_reg) {
            match pin.ty {
                ValueType::Float => dynasm!(self.ops ; .arch aarch64 ; fmov W(w), S(pin.reg)),
                _ => dynasm!(self.ops ; .arch aarch64 ; and WSP(w), W(pin.reg), #0xff),
            }
            return;
        }
        let offset = reg_offset(vm_reg) + 8;
        if offset <= IMM12_MAX {
            let offset = offset as u32;
            dynasm!(self.ops ; .arch aarch64 ; ldrb W(w), [x19, #offset]);
        } else {
            self.emit_reg_addr(11, vm_reg);
            dynasm!(self.ops ; .arch aarch64 ; ldrb W(w), [x11, 8]);
        }
    }

    /// registers[vm_reg].tag = tag. Clobbers w13 (and x11 for far registers).
    pub(super) fn store_tag_imm(&mut self, vm_reg: u8, tag: u8) {
        let offset = reg_offset(vm_reg);
        let tag = tag as u32;
        dynasm!(self.ops ; .arch aarch64 ; movz w13, #tag);
        if offset <= IMM12_MAX {
            let offset = offset as u32;
            dynasm!(self.ops ; .arch aarch64 ; strb w13, [x19, #offset]);
        } else {
            self.emit_reg_addr(11, vm_reg);
            dynasm!(self.ops ; .arch aarch64 ; strb w13, [x11]);
        }
    }

    /// X(x) = 64-bit payload of registers[vm_reg]
    pub(super) fn load_payload(&mut self, x: u8, vm_reg: u8) {
        self.clobber_x(x);
        if let Some(pin) = self.active_pin(vm_reg) {
            match pin.ty {
                ValueType::Float => dynasm!(self.ops ; .arch aarch64 ; fmov X(x), D(pin.reg)),
                _ => dynasm!(self.ops ; .arch aarch64 ; mov X(x), X(pin.reg)),
            }
            return;
        }
        let offset = (reg_offset(vm_reg) + 8) as u32;
        dynasm!(self.ops ; .arch aarch64 ; ldr X(x), [x19, #offset]);
    }

    /// registers[vm_reg].payload = X(x)
    pub(super) fn store_payload(&mut self, vm_reg: u8, x: u8) {
        let offset = (reg_offset(vm_reg) + 8) as u32;
        dynasm!(self.ops ; .arch aarch64 ; str X(x), [x19, #offset]);
    }

    /// D(d) = float payload of registers[vm_reg]
    pub(super) fn load_payload_f(&mut self, d: u8, vm_reg: u8) {
        self.clobber_d(d);
        if let Some(pin) = self.active_pin(vm_reg) {
            match pin.ty {
                ValueType::Float => dynasm!(self.ops ; .arch aarch64 ; fmov D(d), D(pin.reg)),
                _ => dynasm!(self.ops ; .arch aarch64 ; fmov D(d), X(pin.reg)),
            }
            return;
        }
        let offset = (reg_offset(vm_reg) + 8) as u32;
        dynasm!(self.ops ; .arch aarch64 ; ldr D(d), [x19, #offset]);
    }

    /// D(d) = float(int payload of registers[vm_reg]); clobbers x11
    pub(super) fn load_payload_int_as_f(&mut self, d: u8, vm_reg: u8) {
        self.clobber_d(d);
        if let Some(pin) = self.active_pin(vm_reg)
            && pin.ty != ValueType::Float
        {
            dynasm!(self.ops ; .arch aarch64 ; scvtf D(d), X(pin.reg));
            return;
        }
        self.load_payload(11, vm_reg);
        dynasm!(self.ops ; .arch aarch64 ; scvtf D(d), x11);
    }

    /// X(x) = &specialized slot at `stack_offset` (a negative offset from x29)
    pub(super) fn emit_slot_addr(&mut self, x: u8, stack_offset: i32) {
        debug_assert!(stack_offset < 0);
        self.emit_sub_imm(x, 29, -stack_offset);
    }

    // ── Helper calls ──────────────────────────────────────────────────────

    /// Call an `extern "C"` helper with arguments already in x0..x7.
    pub(super) fn emit_call(&mut self, function: *const ()) {
        self.hot_x0_in = None;
        self.hot_d0_in = None;
        self.emit_mov_imm64(16, function as usize as u64);
        dynasm!(self.ops ; .arch aarch64 ; blr x16);
    }

    /// Branch to the trace fail path when the helper's `u8` result in w0 is 0.
    pub(super) fn emit_fail_if_w0_zero(&mut self) {
        dynasm!(self.ops
            ; .arch aarch64
            ; and w0, w0, 0xff
            ; cbz w0, >fail
        );
    }

    /// Generic exit with a guard return value (guard_index + 1). Function
    /// code also records where and why it left (see `JIT_EXIT_INFO`), since
    /// the result propagates through native callers unchanged.
    pub(super) fn emit_guard_exit(&mut self, guard_return_value: i32) {
        let exit_label = self.current_exit_label();
        if self.function_mode {
            let kind = if self.exit_is_handoff {
                jit::EXIT_KIND_HANDOFF
            } else {
                jit::EXIT_KIND_GUARD
            };
            let ip = self.current_fail_ip.unwrap_or(usize::MAX >> 16);
            self.emit_exit_info(ip, kind);
        }
        self.emit_mov_imm_i32(0, guard_return_value);
        dynasm!(self.ops ; .arch aarch64 ; b =>exit_label);
    }

    /// `JIT_EXIT_INFO = ip | kind << EXIT_KIND_SHIFT`.
    pub(super) fn emit_exit_info(&mut self, ip: usize, kind: usize) {
        let info = (ip & ((1usize << jit::EXIT_KIND_SHIFT) - 1)) | (kind << jit::EXIT_KIND_SHIFT);
        let offset = jit::EXIT_INFO_OFFSET as u32;
        self.emit_mov_imm64(12, info as u64);
        dynasm!(self.ops ; .arch aarch64 ; str x12, [x20, #offset]);
    }

    // ── Scalar stores (port of store_from_rax / store_xmm0_as_float) ──────

    /// Store x0 as a Bool (discriminant 1) or Int (discriminant 2) into the
    /// VM register, dropping any owned value that previously lived there.
    pub(super) fn store_from_x0(&mut self, vm_reg: u8, discriminant: u8) {
        let stored_type = match discriminant {
            1 => ValueType::Bool,
            2 => ValueType::Int,
            _ => unreachable!("unsupported scalar Value discriminant"),
        };
        // A Bool arrives as a byte in w0; whatever the upper bits hold (an
        // 8-byte payload load of a Bool, say) must not reach the payload
        // or the `u8` helper argument, which the ABI lets the callee read
        // as a full register.
        if stored_type == ValueType::Bool {
            dynasm!(self.ops ; .arch aarch64 ; and x0, x0, #0xff);
        }
        if let Some(pin) = self.pin_for_write(vm_reg) {
            assert_eq!(pin.ty, stored_type, "pinned register written with another type");
            dynasm!(self.ops ; .arch aarch64 ; mov X(pin.reg), x0);
            if pin.class == pins::PinClass::Carried {
                self.mark_dirty(vm_reg);
                return;
            }
            // Local pins write through so memory is always current.
        }
        if self.scalar_registers.get(&vm_reg) == Some(&stored_type) {
            self.store_payload(vm_reg, 0);
            self.hot_x0 = Some(vm_reg);
            return;
        }
        if self.scalar_registers.contains_key(&vm_reg) {
            self.store_tag_imm(vm_reg, discriminant);
            self.store_payload(vm_reg, 0);
            self.hot_x0 = Some(vm_reg);
            return;
        }
        // The slow path below is a call: x0 is not the payload afterwards.
        let scalar_max_tag = ValueTag::Float.as_u8() as u32;
        let replace: *const () = match discriminant {
            1 => {
                unsafe extern "C" {
                    fn jit_replace_bool(dest: *mut Value, value: u8) -> u8;
                }
                jit_replace_bool as *const ()
            }
            2 => {
                unsafe extern "C" {
                    fn jit_replace_int(dest: *mut Value, value: crate::number::LustInt) -> u8;
                }
                jit_replace_int as *const ()
            }
            _ => unreachable!("unsupported scalar Value discriminant"),
        };
        self.load_tag_from_memory(9, vm_reg);
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp w9, #scalar_max_tag
            ; b.hi >replace_owned
        );
        self.store_tag_imm(vm_reg, discriminant);
        self.store_payload(vm_reg, 0);
        dynasm!(self.ops
            ; .arch aarch64
            ; b >store_done
            ; replace_owned:
            ; mov x1, x0
        );
        self.emit_reg_addr(0, vm_reg);
        self.emit_call(replace);
        dynasm!(self.ops
            ; .arch aarch64
            ; store_done:
        );
    }

    /// Store d0 as a Float into the VM register, dropping any owned value
    /// that previously lived there.
    pub(super) fn store_d0_as_float(&mut self, vm_reg: u8) {
        let float_tag = ValueTag::Float.as_u8();
        if let Some(pin) = self.pin_for_write(vm_reg) {
            assert_eq!(pin.ty, ValueType::Float, "pinned register written with another type");
            dynasm!(self.ops ; .arch aarch64 ; fmov D(pin.reg), d0);
            if pin.class == pins::PinClass::Carried {
                self.mark_dirty(vm_reg);
                return;
            }
        }
        if self.scalar_registers.get(&vm_reg) == Some(&ValueType::Float) {
            let offset = (reg_offset(vm_reg) + 8) as u32;
            dynasm!(self.ops ; .arch aarch64 ; str d0, [x19, #offset]);
            self.hot_d0 = Some(vm_reg);
            return;
        }
        if self.scalar_registers.contains_key(&vm_reg) {
            self.store_tag_imm(vm_reg, float_tag);
            let offset = (reg_offset(vm_reg) + 8) as u32;
            dynasm!(self.ops ; .arch aarch64 ; str d0, [x19, #offset]);
            self.hot_d0 = Some(vm_reg);
            return;
        }
        let scalar_max_tag = ValueTag::Float.as_u8() as u32;
        unsafe extern "C" {
            fn jit_replace_float_bits(dest: *mut Value, bits: u64) -> u8;
        }
        self.load_tag_from_memory(9, vm_reg);
        dynasm!(self.ops
            ; .arch aarch64
            ; fmov x1, d0
            ; cmp w9, #scalar_max_tag
            ; b.hi >replace_owned
        );
        self.store_tag_imm(vm_reg, float_tag);
        self.store_payload(vm_reg, 1);
        dynasm!(self.ops
            ; .arch aarch64
            ; b >store_done
            ; replace_owned:
        );
        self.emit_reg_addr(0, vm_reg);
        self.emit_call(jit_replace_float_bits as *const ());
        dynasm!(self.ops
            ; .arch aarch64
            ; store_done:
        );
    }
}

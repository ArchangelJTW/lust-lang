//! Copying values without the runtime: a `Value` is a tag and an 8-byte
//! payload, and every owning variant's payload is one `Rc` allocation
//! pointer, whose strong count is the allocation's first word. Retaining
//! is an increment; releasing decrements unless the count would reach
//! zero, when the runtime drops the value. A weak reference (whose count
//! is the second word, and which may be dangling) always goes to the
//! runtime.

use super::*;

impl JitCompiler {
    /// Branch to `owned` for a value at x11 that owns an allocation; fall
    /// through for one that owns nothing. w9 = its tag afterwards.
    fn emit_ownership_dispatch(
        &mut self,
        own: &jit::layout::OwnershipLayout,
        owned: dynasmrt::DynamicLabel,
        weak: dynasmrt::DynamicLabel,
    ) {
        // Scalars (tags up to Float) are the common case: one compare.
        let scalar_max_tag = ValueTag::Float.as_u8() as u32;
        dynasm!(self.ops
            ; .arch aarch64
            ; ldrb w9, [x11]
            ; cmp w9, #scalar_max_tag
            ; b.ls >plain
        );
        for tag in own.plain_tags {
            if u32::from(tag) > scalar_max_tag {
                dynasm!(self.ops ; .arch aarch64 ; cmp w9, #tag as u32 ; b.eq >plain);
            }
        }
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp w9, #own.weak_tag as u32
            ; b.eq => weak
            ; b => owned
            ; plain:
        );
    }

    /// x11 = address of a `Value`: bump the count it owns. Clobbers x0–x15
    /// (the slow path is a call).
    pub(super) fn emit_retain_at_x11(&mut self) {
        unsafe extern "C" {
            fn jit_retain_value(value: *const Value);
        }
        let Some(own) = jit::layout::ownership_layout() else {
            dynasm!(self.ops ; .arch aarch64 ; mov x0, x11);
            self.emit_call(jit_retain_value as *const ());
            return;
        };
        let done = self.ops.new_dynamic_label();
        let owned = self.ops.new_dynamic_label();
        let slow = self.ops.new_dynamic_label();
        let offset = own.single_rc_offset as u32;
        self.emit_ownership_dispatch(&own, owned, slow);
        dynasm!(self.ops
            ; .arch aarch64
            ; b => done
            ; => owned
            ; ldr x10, [x11, #offset]
            ; ldr x12, [x10]
            ; add x12, x12, 1
            ; str x12, [x10]
            ; b => done
            ; => slow
            ; mov x0, x11
        );
        self.emit_call(jit_retain_value as *const ());
        dynasm!(self.ops ; .arch aarch64 ; => done);
    }

    /// x11 = address of a `Value` about to be overwritten: give up what it
    /// owns. A count that would reach zero sends the value to the runtime,
    /// which drops it and leaves Nil. Clobbers x0–x15.
    pub(super) fn emit_release_at_x11(&mut self) {
        unsafe extern "C" {
            fn jit_release_value(value: *mut Value);
        }
        let Some(own) = jit::layout::ownership_layout() else {
            dynasm!(self.ops ; .arch aarch64 ; mov x0, x11);
            self.emit_call(jit_release_value as *const ());
            return;
        };
        let done = self.ops.new_dynamic_label();
        let owned = self.ops.new_dynamic_label();
        let slow = self.ops.new_dynamic_label();
        let offset = own.single_rc_offset as u32;
        self.emit_ownership_dispatch(&own, owned, slow);
        dynasm!(self.ops
            ; .arch aarch64
            ; b => done
            ; => owned
            ; ldr x10, [x11, #offset]
            ; ldr x12, [x10]
            ; cmp x12, 1
            ; b.ls => slow
            ; sub x12, x12, 1
            ; str x12, [x10]
            ; b => done
            ; => slow
            ; mov x0, x11
        );
        self.emit_call(jit_release_value as *const ());
        dynasm!(self.ops ; .arch aarch64 ; => done);
    }

    /// Copy the `Value` at x11 to the one at x9, bitwise (16 bytes).
    /// Clobbers x0, x1.
    pub(super) fn emit_copy_x11_to_x9(&mut self) {
        let value_size = mem::size_of::<Value>() as i32;
        for chunk in (0..value_size).step_by(16) {
            dynasm!(self.ops
                ; .arch aarch64
                ; ldp x0, x1, [x11, #chunk]
                ; stp x0, x1, [x9, #chunk]
            );
        }
    }

    /// `registers[dest] = clone of the Value at x11`: retain the source,
    /// release the destination's old value, copy. Clobbers x0–x15.
    pub(super) fn emit_clone_x11_into(&mut self, dest: u8) {
        // A destination holding nothing owned needs no release: copy, then
        // retain the copy (the same counts as the source's).
        if self.scalar_registers.contains_key(&dest) {
            self.emit_reg_addr(9, dest);
            self.emit_copy_x11_to_x9();
            self.emit_reg_addr(11, dest);
            self.emit_retain_at_x11();
            self.scalar_registers.remove(&dest);
            return;
        }
        // The source is copied to the stack first: releasing the
        // destination may free the container the source lives in (`t =
        // t.next` with `t` the last reference), and the slow paths are
        // calls that clobber everything else.
        let value_size = mem::size_of::<Value>() as u32;
        dynasm!(self.ops
            ; .arch aarch64
            ; sub sp, sp, #value_size
            ; mov x9, sp
        );
        self.emit_copy_x11_to_x9();
        dynasm!(self.ops ; .arch aarch64 ; mov x11, sp);
        self.emit_retain_at_x11();
        self.emit_reg_addr(11, dest);
        self.emit_release_at_x11();
        self.emit_reg_addr(9, dest);
        dynasm!(self.ops ; .arch aarch64 ; mov x11, sp);
        self.emit_copy_x11_to_x9();
        dynasm!(self.ops ; .arch aarch64 ; add sp, sp, #value_size);
        self.scalar_registers.remove(&dest);
    }
}

//! Copying values without the runtime: a `Value` is 48 bytes plus the
//! reference counts it owns, and for the common variants (one `Rc`, a
//! struct's three, an enum's two names and payload) those counts live at
//! offsets the layout probe measured. Retaining is an increment; releasing
//! decrements unless a count would reach zero, when the runtime drops the
//! value. Variants the probe does not cover (weak references, closures,
//! tasks) always go to the runtime.

use super::*;

impl JitCompiler {
    /// x11 = address of a `Value`: bump the counts it owns. Clobbers x0–x15
    /// (the slow path is a call).
    pub(super) fn emit_retain_at_x11(&mut self) {
        unsafe extern "C" {
            fn jit_retain_value(value: *const Value);
        }
        let Some((own, rc, en)) = Self::ownership() else {
            dynasm!(self.ops ; .arch aarch64 ; mov x0, x11);
            self.emit_call(jit_retain_value as *const ());
            return;
        };
        let done = self.ops.new_dynamic_label();
        let single = self.ops.new_dynamic_label();
        let slow = self.ops.new_dynamic_label();
        dynasm!(self.ops ; .arch aarch64 ; ldrb w9, [x11]);
        for tag in own.plain_tags {
            dynasm!(self.ops ; .arch aarch64 ; cmp w9, #tag as u32 ; b.eq => done);
        }
        for tag in own.single_rc_tags {
            dynasm!(self.ops ; .arch aarch64 ; cmp w9, #tag as u32 ; b.eq => single);
        }
        let struct_tag = own.struct_tag as u32;
        let enum_tag = own.enum_tag as u32;
        let name_offset = rc.struct_name_offset as u32;
        let layout_offset = rc.struct_layout_offset as u32;
        let fields_offset = rc.struct_fields_offset as u32;
        let enum_name_offset = en.enum_name_offset as u32;
        let variant_offset = en.variant_offset as u32;
        let values_offset = en.values_offset as u32;
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp w9, #struct_tag
            ; b.ne >not_struct
            ; ldr x10, [x11, #name_offset]
            ; ldr x12, [x10]
            ; add x12, x12, 1
            ; str x12, [x10]
            ; ldr x10, [x11, #layout_offset]
            ; ldr x12, [x10]
            ; add x12, x12, 1
            ; str x12, [x10]
            ; ldr x10, [x11, #fields_offset]
            ; ldr x12, [x10]
            ; add x12, x12, 1
            ; str x12, [x10]
            ; b => done
            ; not_struct:
            ; cmp w9, #enum_tag
            ; b.ne => slow
            ; ldr x10, [x11, #enum_name_offset]
            ; ldr x12, [x10]
            ; add x12, x12, 1
            ; str x12, [x10]
            ; ldr x10, [x11, #variant_offset]
            ; ldr x12, [x10]
            ; add x12, x12, 1
            ; str x12, [x10]
            ; ldr x10, [x11, #values_offset]
            ; cbz x10, => done
            ; ldr x12, [x10]
            ; add x12, x12, 1
            ; str x12, [x10]
            ; b => done
            ; => single
        );
        let single_offset = own.single_rc_offset as u32;
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x10, [x11, #single_offset]
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
        let Some((own, rc, en)) = Self::ownership() else {
            dynasm!(self.ops ; .arch aarch64 ; mov x0, x11);
            self.emit_call(jit_release_value as *const ());
            return;
        };
        let done = self.ops.new_dynamic_label();
        let single = self.ops.new_dynamic_label();
        let slow = self.ops.new_dynamic_label();
        dynasm!(self.ops ; .arch aarch64 ; ldrb w9, [x11]);
        for tag in own.plain_tags {
            dynasm!(self.ops ; .arch aarch64 ; cmp w9, #tag as u32 ; b.eq => done);
        }
        for tag in own.single_rc_tags {
            dynasm!(self.ops ; .arch aarch64 ; cmp w9, #tag as u32 ; b.eq => single);
        }
        let struct_tag = own.struct_tag as u32;
        let enum_tag = own.enum_tag as u32;
        let name_offset = rc.struct_name_offset as u32;
        let layout_offset = rc.struct_layout_offset as u32;
        let fields_offset = rc.struct_fields_offset as u32;
        let enum_name_offset = en.enum_name_offset as u32;
        let variant_offset = en.variant_offset as u32;
        let values_offset = en.values_offset as u32;
        // Every count is checked before any is decremented, so the slow
        // path always sees the value intact.
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp w9, #struct_tag
            ; b.ne >not_struct
            ; ldr x10, [x11, #name_offset]
            ; ldr x12, [x10]
            ; ldr x13, [x11, #layout_offset]
            ; ldr x14, [x13]
            ; ldr x15, [x11, #fields_offset]
            ; ldr x9, [x15]
            ; cmp x12, 1
            ; b.ls => slow
            ; cmp x14, 1
            ; b.ls => slow
            ; cmp x9, 1
            ; b.ls => slow
            ; sub x12, x12, 1
            ; str x12, [x10]
            ; sub x14, x14, 1
            ; str x14, [x13]
            ; sub x9, x9, 1
            ; str x9, [x15]
            ; b => done
            ; not_struct:
            ; cmp w9, #enum_tag
            ; b.ne => slow
            ; ldr x10, [x11, #enum_name_offset]
            ; ldr x12, [x10]
            ; ldr x13, [x11, #variant_offset]
            ; ldr x14, [x13]
            ; ldr x15, [x11, #values_offset]
            ; mov x9, 2
            ; cbz x15, >no_values
            ; ldr x9, [x15]
            ; no_values:
            ; cmp x12, 1
            ; b.ls => slow
            ; cmp x14, 1
            ; b.ls => slow
            ; cmp x9, 1
            ; b.ls => slow
            ; sub x12, x12, 1
            ; str x12, [x10]
            ; sub x14, x14, 1
            ; str x14, [x13]
            ; cbz x15, => done
            ; sub x9, x9, 1
            ; str x9, [x15]
            ; b => done
            ; => single
        );
        let single_offset = own.single_rc_offset as u32;
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x10, [x11, #single_offset]
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

    /// Copy the `Value` at x11 to the one at x9, bitwise (48 bytes).
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

    fn ownership() -> Option<(
        jit::layout::OwnershipLayout,
        jit::layout::RcVecLayout,
        jit::layout::EnumLayout,
    )> {
        Some((
            jit::layout::ownership_layout()?,
            jit::layout::rc_vec_layout()?,
            jit::layout::enum_layout()?,
        ))
    }
}

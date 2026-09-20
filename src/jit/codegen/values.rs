//! Copying values without the runtime (the x86_64 twin of
//! `codegen_aarch64/values.rs`): a `Value` is a tag and an 8-byte payload,
//! and every owning variant's payload is one `Rc` allocation pointer,
//! whose strong count is the allocation's first word. Retaining is an
//! increment; releasing decrements unless the count would reach zero,
//! when the runtime drops the value. A weak reference (whose count is the
//! second word, and which may be dangling) always goes to the runtime.

use super::*;

impl JitCompiler {
    /// Jump to `owned` for a value at rsi that owns an allocation, to
    /// `weak` for a weak reference; fall through for one that owns
    /// nothing. ecx = its tag afterwards.
    fn emit_ownership_dispatch(
        &mut self,
        own: &jit::layout::OwnershipLayout,
        owned: dynasmrt::DynamicLabel,
        weak: dynasmrt::DynamicLabel,
    ) {
        // Scalars (tags up to Float) are the common case: one compare.
        let scalar_max_tag = ValueTag::Float.as_u8() as i32;
        dynasm!(self.ops
            ; .arch x64
            ; movzx ecx, BYTE [rsi]
            ; cmp ecx, scalar_max_tag
            ; jbe >plain
        );
        for tag in own.plain_tags {
            if i32::from(tag) > scalar_max_tag {
                dynasm!(self.ops ; .arch x64 ; cmp ecx, tag as i32 ; je >plain);
            }
        }
        dynasm!(self.ops
            ; .arch x64
            ; cmp ecx, own.weak_tag as i32
            ; je => weak
            ; jmp => owned
            ; plain:
        );
    }

    fn emit_call_on_rsi(&mut self, helper: *const ()) {
        dynasm!(self.ops
            ; .arch x64
            ; push rsi
            ; sub rsp, 8
            ; mov rdi, rsi
            ; mov rax, QWORD helper as _
            ; call rax
            ; add rsp, 8
            ; pop rsi
        );
    }

    /// rsi = address of a `Value`: bump the count it owns. Clobbers the
    /// caller-saved registers (the slow path is a call); rsi is preserved.
    pub(super) fn emit_retain_at_rsi(&mut self) {
        unsafe extern "C" {
            fn jit_retain_value(value: *const Value);
        }
        let Some(own) = jit::layout::ownership_layout() else {
            self.emit_call_on_rsi(jit_retain_value as *const ());
            return;
        };
        let done = self.ops.new_dynamic_label();
        let owned = self.ops.new_dynamic_label();
        let slow = self.ops.new_dynamic_label();
        let offset = own.single_rc_offset as i32;
        self.emit_ownership_dispatch(&own, owned, slow);
        dynasm!(self.ops
            ; .arch x64
            ; jmp => done
            ; => owned
            ; mov rax, [rsi + offset]
            ; add QWORD [rax], 1
            ; jmp => done
            ; => slow
        );
        self.emit_call_on_rsi(jit_retain_value as *const ());
        dynasm!(self.ops ; .arch x64 ; => done);
    }

    /// rsi = address of a `Value` about to be overwritten: give up what it
    /// owns. A count that would reach zero sends the value to the runtime,
    /// which drops it and leaves Nil. Clobbers the caller-saved registers;
    /// rsi is preserved.
    pub(super) fn emit_release_at_rsi(&mut self) {
        unsafe extern "C" {
            fn jit_release_value(value: *mut Value);
        }
        let Some(own) = jit::layout::ownership_layout() else {
            self.emit_call_on_rsi(jit_release_value as *const ());
            return;
        };
        let done = self.ops.new_dynamic_label();
        let owned = self.ops.new_dynamic_label();
        let slow = self.ops.new_dynamic_label();
        let offset = own.single_rc_offset as i32;
        self.emit_ownership_dispatch(&own, owned, slow);
        dynasm!(self.ops
            ; .arch x64
            ; jmp => done
            ; => owned
            ; mov rax, [rsi + offset]
            ; cmp QWORD [rax], 1
            ; jbe => slow
            ; sub QWORD [rax], 1
            ; jmp => done
            ; => slow
        );
        self.emit_call_on_rsi(jit_release_value as *const ());
        dynasm!(self.ops ; .arch x64 ; => done);
    }

    /// `registers[dest] = clone of the Value at rsi`: the source is copied
    /// to the stack first (releasing the destination may free the
    /// container the source lives in), retained, the destination's old
    /// value released, and the copy stored. Clobbers the caller-saved
    /// registers.
    pub(super) fn emit_clone_rsi_into(&mut self, dest: u8) {
        let value_size = mem::size_of::<Value>() as i32;
        let dest_offset = (dest as i32) * value_size;
        // A destination holding nothing owned needs no release: copy, then
        // retain the copy (the same counts as the source's).
        if self.scalar_registers.contains_key(&dest) {
            for word in (0..value_size).step_by(8) {
                dynasm!(self.ops
                    ; .arch x64
                    ; mov rax, [rsi + word]
                    ; mov [r12 + dest_offset + word], rax
                );
            }
            dynasm!(self.ops ; .arch x64 ; lea rsi, [r12 + dest_offset]);
            self.emit_retain_at_rsi();
            self.scalar_registers.remove(&dest);
            return;
        }
        dynasm!(self.ops ; .arch x64 ; sub rsp, value_size);
        for word in (0..value_size).step_by(8) {
            dynasm!(self.ops
                ; .arch x64
                ; mov rax, [rsi + word]
                ; mov [rsp + word], rax
            );
        }
        dynasm!(self.ops ; .arch x64 ; mov rsi, rsp);
        self.emit_retain_at_rsi();
        dynasm!(self.ops ; .arch x64 ; lea rsi, [r12 + dest_offset]);
        self.emit_release_at_rsi();
        for word in (0..value_size).step_by(8) {
            dynasm!(self.ops
                ; .arch x64
                ; mov rax, [rsp + word]
                ; mov [r12 + dest_offset + word], rax
            );
        }
        dynasm!(self.ops ; .arch x64 ; add rsp, value_size);
        self.scalar_registers.remove(&dest);
    }
}

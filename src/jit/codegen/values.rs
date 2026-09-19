//! Copying values without the runtime (the x86_64 twin of
//! `codegen_aarch64/values.rs`): a `Value` is 48 bytes plus the reference
//! counts it owns, and for the common variants (one `Rc`, a struct's
//! three, an enum's two names and payload) those counts live at offsets
//! the layout probe measured. Retaining is an increment; releasing
//! decrements unless a count would reach zero, when the runtime drops the
//! value. Variants the probe does not cover (weak references, closures,
//! tasks) always go to the runtime.

use super::*;

impl JitCompiler {
    /// rsi = address of a `Value`: bump the counts it owns. Clobbers the
    /// caller-saved registers (the slow path is a call); rsi is preserved.
    pub(super) fn emit_retain_at_rsi(&mut self) {
        unsafe extern "C" {
            fn jit_retain_value(value: *const Value);
        }
        let Some((own, rc, en)) = Self::ownership() else {
            dynasm!(self.ops
                ; .arch x64
                ; push rsi
                ; sub rsp, 8
                ; mov rdi, rsi
                ; mov rax, QWORD jit_retain_value as *const () as _
                ; call rax
                ; add rsp, 8
                ; pop rsi
            );
            return;
        };
        let done = self.ops.new_dynamic_label();
        let single = self.ops.new_dynamic_label();
        let slow = self.ops.new_dynamic_label();
        dynasm!(self.ops ; .arch x64 ; movzx ecx, BYTE [rsi]);
        for tag in own.plain_tags {
            dynasm!(self.ops ; .arch x64 ; cmp ecx, tag as i32 ; je => done);
        }
        for tag in own.single_rc_tags {
            dynasm!(self.ops ; .arch x64 ; cmp ecx, tag as i32 ; je => single);
        }
        let struct_tag = own.struct_tag as i32;
        let enum_tag = own.enum_tag as i32;
        let name_offset = rc.struct_name_offset as i32;
        let layout_offset = rc.struct_layout_offset as i32;
        let fields_offset = rc.struct_fields_offset as i32;
        let enum_name_offset = en.enum_name_offset as i32;
        let variant_offset = en.variant_offset as i32;
        let values_offset = en.values_offset as i32;
        dynasm!(self.ops
            ; .arch x64
            ; cmp ecx, struct_tag
            ; jne >not_struct
            ; mov rax, [rsi + name_offset]
            ; add QWORD [rax], 1
            ; mov rax, [rsi + layout_offset]
            ; add QWORD [rax], 1
            ; mov rax, [rsi + fields_offset]
            ; add QWORD [rax], 1
            ; jmp => done
            ; not_struct:
            ; cmp ecx, enum_tag
            ; jne => slow
            ; mov rax, [rsi + enum_name_offset]
            ; add QWORD [rax], 1
            ; mov rax, [rsi + variant_offset]
            ; add QWORD [rax], 1
            ; mov rax, [rsi + values_offset]
            ; test rax, rax
            ; jz => done
            ; add QWORD [rax], 1
            ; jmp => done
            ; => single
        );
        let single_offset = own.single_rc_offset as i32;
        dynasm!(self.ops
            ; .arch x64
            ; mov rax, [rsi + single_offset]
            ; add QWORD [rax], 1
            ; jmp => done
            ; => slow
            ; push rsi
            ; sub rsp, 8
            ; mov rdi, rsi
            ; mov rax, QWORD jit_retain_value as *const () as _
            ; call rax
            ; add rsp, 8
            ; pop rsi
            ; => done
        );
    }

    /// rsi = address of a `Value` about to be overwritten: give up what it
    /// owns. A count that would reach zero sends the value to the runtime,
    /// which drops it and leaves Nil. Clobbers the caller-saved registers;
    /// rsi is preserved.
    pub(super) fn emit_release_at_rsi(&mut self) {
        unsafe extern "C" {
            fn jit_release_value(value: *mut Value);
        }
        let Some((own, rc, en)) = Self::ownership() else {
            dynasm!(self.ops
                ; .arch x64
                ; push rsi
                ; sub rsp, 8
                ; mov rdi, rsi
                ; mov rax, QWORD jit_release_value as *const () as _
                ; call rax
                ; add rsp, 8
                ; pop rsi
            );
            return;
        };
        let done = self.ops.new_dynamic_label();
        let single = self.ops.new_dynamic_label();
        let slow = self.ops.new_dynamic_label();
        dynasm!(self.ops ; .arch x64 ; movzx ecx, BYTE [rsi]);
        for tag in own.plain_tags {
            dynasm!(self.ops ; .arch x64 ; cmp ecx, tag as i32 ; je => done);
        }
        for tag in own.single_rc_tags {
            dynasm!(self.ops ; .arch x64 ; cmp ecx, tag as i32 ; je => single);
        }
        let struct_tag = own.struct_tag as i32;
        let enum_tag = own.enum_tag as i32;
        let name_offset = rc.struct_name_offset as i32;
        let layout_offset = rc.struct_layout_offset as i32;
        let fields_offset = rc.struct_fields_offset as i32;
        let enum_name_offset = en.enum_name_offset as i32;
        let variant_offset = en.variant_offset as i32;
        let values_offset = en.values_offset as i32;
        // Every count is checked before any is decremented, so the slow
        // path always sees the value intact.
        dynasm!(self.ops
            ; .arch x64
            ; cmp ecx, struct_tag
            ; jne >not_struct
            ; mov rax, [rsi + name_offset]
            ; mov rdx, [rsi + layout_offset]
            ; mov r8, [rsi + fields_offset]
            ; cmp QWORD [rax], 1
            ; jbe => slow
            ; cmp QWORD [rdx], 1
            ; jbe => slow
            ; cmp QWORD [r8], 1
            ; jbe => slow
            ; sub QWORD [rax], 1
            ; sub QWORD [rdx], 1
            ; sub QWORD [r8], 1
            ; jmp => done
            ; not_struct:
            ; cmp ecx, enum_tag
            ; jne => slow
            ; mov rax, [rsi + enum_name_offset]
            ; mov rdx, [rsi + variant_offset]
            ; mov r8, [rsi + values_offset]
            ; cmp QWORD [rax], 1
            ; jbe => slow
            ; cmp QWORD [rdx], 1
            ; jbe => slow
            ; test r8, r8
            ; jz >names_only
            ; cmp QWORD [r8], 1
            ; jbe => slow
            ; sub QWORD [r8], 1
            ; names_only:
            ; sub QWORD [rax], 1
            ; sub QWORD [rdx], 1
            ; jmp => done
            ; => single
        );
        let single_offset = own.single_rc_offset as i32;
        dynasm!(self.ops
            ; .arch x64
            ; mov rax, [rsi + single_offset]
            ; cmp QWORD [rax], 1
            ; jbe => slow
            ; sub QWORD [rax], 1
            ; jmp => done
            ; => slow
            ; push rsi
            ; sub rsp, 8
            ; mov rdi, rsi
            ; mov rax, QWORD jit_release_value as *const () as _
            ; call rax
            ; add rsp, 8
            ; pop rsi
            ; => done
        );
    }

    /// `registers[dest] = clone of the Value at rsi`: the source is copied
    /// to the stack first (releasing the destination may free the
    /// container the source lives in), retained, the destination's old
    /// value released, and the copy stored. Clobbers the caller-saved
    /// registers.
    pub(super) fn emit_clone_rsi_into(&mut self, dest: u8) {
        let value_size = mem::size_of::<Value>() as i32;
        let dest_offset = (dest as i32) * value_size;
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

use super::*;
impl JitCompiler {
    pub(super) fn compile_load_const(&mut self, dest: u8, value: &Value) -> Result<()> {
        match value {
            Value::Int(i) => {
                dynasm!(self.ops
                    ; .arch x64
                    ; mov rax, QWORD *i as _
                );
                self.store_from_rax(dest, 2);
                Ok(())
            }

            Value::Float(f) => {
                let f_bits = f.to_bits();
                dynasm!(self.ops
                    ; .arch x64
                    ; mov rax, QWORD f_bits as _
                    ; movq xmm0, rax
                );
                self.store_xmm0_as_float(dest);
                Ok(())
            }

            Value::Bool(b) => {
                let bool_val = if *b { 1i64 } else { 0i64 };
                dynasm!(self.ops
                    ; .arch x64
                    ; mov rax, QWORD bool_val
                );
                self.store_from_rax(dest, 1);
                Ok(())
            }

            // Plain values that own nothing: tag + payload, with the old
            // value dropped only if it owned something.
            Value::Nil => self.store_plain_constant(dest, 0, value),
            Value::Function(idx) => self.store_plain_constant(dest, *idx as u64, value),

            _ => self.copy_owned_constant(dest, value),
        }
    }

    fn store_plain_constant(&mut self, dest: u8, payload: u64, value: &Value) -> Result<()> {
        let offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        let scalar_max_tag = ValueTag::Float.as_u8() as i8;
        // The discriminant byte of `#[repr(C, u8)] Value` (`ValueTag` is a
        // coarser classification and numbers `Function` differently).
        // SAFETY: reading the first byte of a live `Value`.
        let tag = unsafe { *(value as *const Value as *const u8) } as i8;
        // A destination known to hold nothing owned is simply overwritten.
        if self.scalar_registers.contains_key(&dest) {
            dynasm!(self.ops
                ; .arch x64
                ; mov rax, QWORD payload as _
                ; mov BYTE [r12 + offset], tag
                ; mov [r12 + offset + 8], rax
            );
            return Ok(());
        }
        let done = self.ops.new_dynamic_label();
        dynasm!(self.ops
            ; .arch x64
            ; cmp BYTE [r12 + offset], scalar_max_tag
            ; jbe >plain
        );
        // A function index held from the previous iteration of a loop that
        // reloads it is as plain as a scalar.
        if matches!(value, Value::Function(_)) {
            dynasm!(self.ops ; .arch x64 ; cmp BYTE [r12 + offset], tag ; jne >owned);
        } else {
            dynasm!(self.ops ; .arch x64 ; jmp >owned);
        }
        dynasm!(self.ops
            ; .arch x64
            ; plain:
            ; mov rax, QWORD payload as _
            ; mov BYTE [r12 + offset], tag
            ; mov [r12 + offset + 8], rax
            ; jmp => done
            ; owned:
        );
        self.copy_owned_constant(dest, value)?;
        dynasm!(self.ops ; .arch x64 ; => done);
        Ok(())
    }

    fn copy_owned_constant(&mut self, dest: u8, value: &Value) -> Result<()> {
        let offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        let src_ptr = self.retain_value(value.clone());
        unsafe extern "C" {
            fn jit_move_safe(src_ptr: *const Value, dest_ptr: *mut Value) -> u8;
        }
        dynasm!(self.ops
            ; .arch x64
            ; mov rdi, QWORD src_ptr as _
            ; lea rsi, [r12 + offset]
            ; mov rax, QWORD jit_move_safe as *const () as _
            ; call rax
            ; test al, al
            ; jz >fail
        );
        Ok(())
    }

    pub(super) fn compile_move(&mut self, dest: u8, src: u8) -> Result<()> {
        let src_offset = (src as i32) * (mem::size_of::<Value>() as i32);
        // A source of known scalar type is a payload copy plus a typed
        // store (which drops whatever the destination held).
        match self.scalar_registers.get(&src).copied() {
            Some(ValueType::Float) => {
                self.operand_xmm0(src);
                self.store_xmm0_as_float(dest);
                return Ok(());
            }
            Some(ValueType::Int) => {
                self.operand_rax(src);
                self.store_from_rax(dest, ValueTag::Int.as_u8());
                return Ok(());
            }
            Some(ValueType::Bool) => {
                self.operand_bool_eax(src);
                self.store_from_rax(dest, ValueTag::Bool.as_u8());
                return Ok(());
            }
            _ => {}
        }
        dynasm!(self.ops ; .arch x64 ; lea rsi, [r12 + src_offset]);
        self.emit_clone_rsi_into(dest);
        Ok(())
    }

    pub(super) fn compile_get_index(&mut self, dest: u8, array: u8, index: u8) -> Result<()> {
        let array_offset = (array as i32) * (mem::size_of::<Value>() as i32);
        let index_offset = (index as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        unsafe extern "C" {
            fn jit_array_get_safe(
                vm_ptr: *mut crate::VM,
                array_value: *const Value,
                index_value: *const Value,
                out: *mut Value,
            ) -> u8;
        }

        dynasm!(self.ops
            ; .arch x64
            ; mov al, [r12 + array_offset]
            ; cmp al, 5
            ; jne >fail
            ; mov al, [r12 + index_offset]
            ; cmp al, 2
            ; jne >fail
            ; mov rdi, r13
            ; lea rsi, [r12 + array_offset]
            ; lea rdx, [r12 + index_offset]
            ; lea rcx, [r12 + dest_offset]
            ; mov rax, QWORD jit_array_get_safe as *const () as _
            ; call rax
            ; test al, al
            ; jz >fail
        );
        Ok(())
    }

    pub(super) fn compile_try_get_index(&mut self, dest: u8, array: u8, index: u8) -> Result<()> {
        let array_offset = (array as i32) * (mem::size_of::<Value>() as i32);
        let index_offset = (index as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        unsafe extern "C" {
            fn jit_array_index_result_safe(
                vm_ptr: *mut crate::VM,
                array_value: *const Value,
                index_value: *const Value,
                out: *mut Value,
            ) -> u8;
        }

        dynasm!(self.ops
            ; .arch x64
            ; mov rdi, r13
            ; lea rsi, [r12 + array_offset]
            ; lea rdx, [r12 + index_offset]
            ; lea rcx, [r12 + dest_offset]
            ; mov rax, QWORD jit_array_index_result_safe as *const () as _
            ; call rax
            ; test al, al
            ; jz >fail
        );
        Ok(())
    }

    pub(super) fn compile_array_index_ok(
        &mut self,
        value_dest: u8,
        condition_dest: u8,
        array: u8,
        index: u8,
        value_type: Option<ValueType>,
    ) -> Result<()> {
        let value_size = mem::size_of::<Value>() as i32;
        let array_offset = (array as i32) * value_size;
        let index_offset = (index as i32) * value_size;
        let value_offset = (value_dest as i32) * value_size;
        let condition_offset = (condition_dest as i32) * value_size;
        unsafe extern "C" {
            fn jit_array_index_ok_safe(
                array_value: *const Value,
                index_value: *const Value,
                value_out: *mut Value,
                condition_out: *mut Value,
            ) -> u8;
        }

        if let (Some(layout), true) = (
            jit::layout::rc_vec_layout(),
            self.scalar_registers.get(&index) == Some(&ValueType::Int),
        ) {
            // Inline: bounds from the Vec's length, a scalar element copied
            // as tag + payload. An element that owns something, or a
            // destination that does, goes through the helper.
            let array_tag = ValueTag::Array.as_u8() as i8;
            let scalar_max_tag = ValueTag::Float.as_u8() as i8;
            let rc_offset = layout.array_rc_offset as i32;
            let len_offset = layout.len_offset as i32;
            let ptr_offset = layout.ptr_offset as i32;
            let done = self.ops.new_dynamic_label();
            let out_of_range = self.ops.new_dynamic_label();
            let slow = self.ops.new_dynamic_label();
            dynasm!(self.ops
                ; .arch x64
                ; cmp BYTE [r12 + array_offset], array_tag
                ; jne => slow
                ; mov r8, [r12 + index_offset + 8]
                ; mov r9, [r12 + array_offset + rc_offset]
                ; cmp r8, [r9 + len_offset]
                ; jae => out_of_range
                ; mov r10, [r9 + ptr_offset]
                ; imul r8, r8, value_size
                ; add r10, r8
            );
            if let Some(ty) = value_type {
                // An element of the expected scalar type: a typed store,
                // after which the destination is known to hold that type.
                // Any other element fails to the interpreter.
                let expected_tag = match ty {
                    ValueType::Int => ValueTag::Int,
                    ValueType::Float => ValueTag::Float,
                    _ => ValueTag::Bool,
                }
                .as_u8() as i8;
                dynasm!(self.ops
                    ; .arch x64
                    ; cmp BYTE [r10], expected_tag
                    ; jne >fail
                );
                match ty {
                    ValueType::Float => {
                        dynasm!(self.ops ; .arch x64 ; movq xmm0, QWORD [r10 + 8]);
                        self.store_xmm0_as_float(value_dest);
                    }
                    ValueType::Int => {
                        dynasm!(self.ops ; .arch x64 ; mov rax, [r10 + 8]);
                        self.store_from_rax(value_dest, ValueTag::Int.as_u8());
                    }
                    _ => {
                        dynasm!(self.ops ; .arch x64 ; movzx eax, BYTE [r10 + 8]);
                        self.store_from_rax(value_dest, ValueTag::Bool.as_u8());
                    }
                }
                self.pending_scalar = Some((value_dest, ty));
            } else {
                dynasm!(self.ops
                    ; .arch x64
                    ; cmp BYTE [r10], scalar_max_tag
                    ; ja => slow
                    // The destination must hold nothing owned for a plain copy.
                    ; cmp BYTE [r12 + value_offset], scalar_max_tag
                    ; ja => slow
                    ; mov rax, [r10]
                    ; mov rcx, [r10 + 8]
                    ; mov [r12 + value_offset], rax
                    ; mov [r12 + value_offset + 8], rcx
                );
            }
            dynasm!(self.ops ; .arch x64 ; mov eax, 1);
            self.store_from_rax(condition_dest, ValueTag::Bool.as_u8());
            dynasm!(self.ops
                ; .arch x64
                ; jmp => done
                ; => out_of_range
            );
            self.compile_load_const(value_dest, &Value::Nil)?;
            dynasm!(self.ops ; .arch x64 ; xor eax, eax);
            self.store_from_rax(condition_dest, ValueTag::Bool.as_u8());
            dynasm!(self.ops
                ; .arch x64
                ; jmp => done
                ; => slow
                ; lea rdi, [r12 + array_offset]
                ; lea rsi, [r12 + index_offset]
                ; lea rdx, [r12 + value_offset]
                ; lea rcx, [r12 + condition_offset]
                ; mov rax, QWORD jit_array_index_ok_safe as *const () as _
                ; call rax
                ; test al, al
                ; jz >fail
                ; => done
            );
            return Ok(());
        }

        dynasm!(self.ops
            ; .arch x64
            ; lea rdi, [r12 + array_offset]
            ; lea rsi, [r12 + index_offset]
            ; lea rdx, [r12 + value_offset]
            ; lea rcx, [r12 + condition_offset]
            ; mov rax, QWORD jit_array_index_ok_safe as *const () as _
            ; call rax
            ; test al, al
            ; jz >fail
        );
        Ok(())
    }

    pub(super) fn compile_array_len(&mut self, dest: u8, array: u8) -> Result<()> {
        let array_offset = (array as i32) * (mem::size_of::<Value>() as i32);
        let array_tag = ValueTag::Array.as_u8() as i8;
        unsafe extern "C" {
            fn jit_array_len_safe(array_value: *const Value) -> i64;
        }

        dynasm!(self.ops
            ; .arch x64
            ; cmp BYTE [r12 + array_offset], array_tag
            ; jne >fail
        );
        if let Some(layout) = jit::layout::rc_vec_layout() {
            let rc_offset = layout.array_rc_offset as i32;
            let len_offset = layout.len_offset as i32;
            dynasm!(self.ops
                ; .arch x64
                ; mov r9, [r12 + array_offset + rc_offset]
                ; mov rax, [r9 + len_offset]
            );
            self.store_from_rax(dest, ValueTag::Int.as_u8());
            return Ok(());
        }
        dynasm!(self.ops
            ; .arch x64
            ; lea rdi, [r12 + array_offset]
            ; mov rax, QWORD jit_array_len_safe as *const () as _
            ; call rax
            ; test rax, rax
            ; js >fail
        );
        self.store_from_rax(dest, ValueTag::Int.as_u8());
        Ok(())
    }

    pub(super) fn compile_get_field(
        &mut self,
        dest: u8,
        object: u8,
        field_name: &str,
        field_index: Option<usize>,
        _value_type: Option<ValueType>,
        _is_weak: bool,
    ) -> Result<()> {
        let object_offset = (object as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        unsafe extern "C" {
            fn jit_get_field_keyed(
                object_ptr: *const Value,
                field_name_ptr: *const u8,
                field_name_len: usize,
                key: *const crate::bytecode::ValueKey,
                out: *mut Value,
            ) -> u8;
            fn jit_get_field_indexed_safe(
                object_ptr: *const Value,
                field_index: usize,
                out: *mut Value,
            ) -> u8;
        }

        if let (Some(index), Some(ty), Some(layout), false) = (
            field_index,
            _value_type.filter(|ty| matches!(ty, ValueType::Int | ValueType::Float | ValueType::Bool)),
            jit::layout::rc_vec_layout(),
            _is_weak,
        ) {
            // A scalar field read inline: struct tag, field count, element
            // tag are each checked, and anything unexpected fails to the
            // interpreter, which re-executes the read. The field is a
            // scalar of the recorded type afterwards.
            let expected_tag = match ty {
                ValueType::Int => ValueTag::Int,
                ValueType::Float => ValueTag::Float,
                _ => ValueTag::Bool,
            }
            .as_u8() as i8;
            let struct_tag = ValueTag::Struct.as_u8() as i8;
            let fields_offset = layout.struct_fields_offset as i32;
            let len_offset = layout.struct_len_offset as i32;
            let ptr_offset = layout.struct_ptr_offset as i32;
            let element = (index * mem::size_of::<Value>()) as i32;
            dynasm!(self.ops
                ; .arch x64
                ; cmp BYTE [r12 + object_offset], struct_tag
                ; jne >fail
                ; mov r9, [r12 + object_offset + fields_offset]
                ; cmp QWORD [r9 + len_offset], index as i32
                ; jbe >fail
                ; mov r10, [r9 + ptr_offset]
                ; cmp BYTE [r10 + element], expected_tag
                ; jne >fail
            );
            match ty {
                ValueType::Float => {
                    dynasm!(self.ops ; .arch x64 ; movq xmm0, QWORD [r10 + element + 8]);
                    self.store_xmm0_as_float(dest);
                }
                ValueType::Int => {
                    dynasm!(self.ops ; .arch x64 ; mov rax, [r10 + element + 8]);
                    self.store_from_rax(dest, ValueTag::Int.as_u8());
                }
                _ => {
                    dynasm!(self.ops ; .arch x64 ; movzx eax, BYTE [r10 + element + 8]);
                    self.store_from_rax(dest, ValueTag::Bool.as_u8());
                }
            }
            self.pending_scalar = Some((dest, ty));
            return Ok(());
        }

        if let (Some(index), Some(layout), false) = (field_index, jit::layout::rc_vec_layout(), _is_weak) {
            // Any strong field read inline: struct tag and field count are
            // checked (anything else fails to the interpreter), then the
            // element is cloned into the register without the runtime.
            let struct_tag = ValueTag::Struct.as_u8() as i8;
            let fields_offset = layout.struct_fields_offset as i32;
            let len_offset = layout.struct_len_offset as i32;
            let ptr_offset = layout.struct_ptr_offset as i32;
            let element = (index * mem::size_of::<Value>()) as i32;
            dynasm!(self.ops
                ; .arch x64
                ; cmp BYTE [r12 + object_offset], struct_tag
                ; jne >fail
                ; mov r9, [r12 + object_offset + fields_offset]
                ; cmp QWORD [r9 + len_offset], index as i32
                ; jbe >fail
                ; mov rsi, [r9 + ptr_offset]
                ; add rsi, element
            );
            self.emit_clone_rsi_into(dest);
            return Ok(());
        }

        if let Some(index) = field_index {
            dynasm!(self.ops
                ; .arch x64
                ; lea rdi, [r12 + object_offset]
                ; mov rsi, QWORD index as _
                ; lea rdx, [r12 + dest_offset]
                ; mov rax, QWORD jit_get_field_indexed_safe as *const () as _
                ; call rax
                ; test al, al
                ; jz >fail
            );
        } else {
            let (field_name_ptr, field_name_len) = self.retain_string(field_name);
            let key = self.retain_key(field_name);
            dynasm!(self.ops
                ; .arch x64
                ; lea rdi, [r12 + object_offset]
                ; mov rsi, QWORD field_name_ptr as _
                ; mov rdx, QWORD field_name_len as _
                ; mov rcx, QWORD key as _
                ; lea r8, [r12 + dest_offset]
                ; mov rax, QWORD jit_get_field_keyed as *const () as _
                ; call rax
                ; test al, al
                ; jz >fail
            );
        }

        Ok(())
    }

    pub(super) fn compile_set_field(
        &mut self,
        object: u8,
        field_name: &str,
        value: u8,
        field_index: Option<usize>,
        _value_type: Option<ValueType>,
        _is_weak: bool,
    ) -> Result<()> {
        let object_offset = (object as i32) * (mem::size_of::<Value>() as i32);
        let value_offset = (value as i32) * (mem::size_of::<Value>() as i32);
        unsafe extern "C" {
            fn jit_set_field_safe(
                object_ptr: *const Value,
                field_name_ptr: *const u8,
                field_name_len: usize,
                value_ptr: *const Value,
            ) -> u8;
            fn jit_set_field_indexed_safe(
                object_ptr: *const Value,
                field_index: usize,
                value_ptr: *const Value,
            ) -> u8;
        }

        let value_ty = self
            .scalar_registers
            .get(&value)
            .copied()
            .filter(|ty| matches!(ty, ValueType::Int | ValueType::Float | ValueType::Bool));
        if let (Some(index), Some(ty), Some(layout), false) =
            (field_index, value_ty, jit::layout::rc_vec_layout(), _is_weak)
        {
            // A scalar store inline when the field currently holds the same
            // scalar kind and the struct is not borrowed; any other
            // situation goes through the helper, which does the full
            // canonicalization and borrow handling.
            unsafe extern "C" {
                fn jit_set_field_strong_safe(
                    object_ptr: *const Value,
                    field_index: usize,
                    value_ptr: *const Value,
                ) -> u8;
            }
            let expected_tag = match ty {
                ValueType::Int => ValueTag::Int,
                ValueType::Float => ValueTag::Float,
                _ => ValueTag::Bool,
            }
            .as_u8() as i8;
            let struct_tag = ValueTag::Struct.as_u8() as i8;
            let fields_offset = layout.struct_fields_offset as i32;
            let len_offset = layout.struct_len_offset as i32;
            let ptr_offset = layout.struct_ptr_offset as i32;
            let borrow_offset = layout.struct_borrow_offset as i32;
            let element = (index * mem::size_of::<Value>()) as i32;
            dynasm!(self.ops
                ; .arch x64
                ; cmp BYTE [r12 + object_offset], struct_tag
                ; jne >slow
                ; mov r9, [r12 + object_offset + fields_offset]
                ; cmp QWORD [r9 + borrow_offset], 0
                ; jne >slow
                ; cmp QWORD [r9 + len_offset], index as i32
                ; jbe >slow
                ; mov r10, [r9 + ptr_offset]
                ; cmp BYTE [r10 + element], expected_tag
                ; jne >slow
            );
            match ty {
                ValueType::Bool => {
                    dynasm!(self.ops
                        ; .arch x64
                        ; movzx eax, BYTE [r12 + value_offset + 8]
                        ; mov BYTE [r10 + element + 8], al
                    );
                }
                _ => {
                    dynasm!(self.ops
                        ; .arch x64
                        ; mov rax, [r12 + value_offset + 8]
                        ; mov [r10 + element + 8], rax
                    );
                }
            }
            dynasm!(self.ops
                ; .arch x64
                ; jmp >done
                ; slow:
                ; lea rdi, [r12 + object_offset]
                ; mov rsi, QWORD index as _
                ; lea rdx, [r12 + value_offset]
                ; mov rax, QWORD jit_set_field_strong_safe as *const () as _
                ; call rax
                ; test al, al
                ; jz >fail
                ; done:
            );
            return Ok(());
        }

        if let Some(index) = field_index {
            crate::jit::log(|| {
                format!(
                    "🔧 JIT: SetField using indexed path, field_index={}, is_weak={}",
                    index, _is_weak
                )
            });
            // Use specialized helpers based on whether field is weak or strong
            if _is_weak {
                dynasm!(self.ops
                    ; .arch x64
                    ; lea rdi, [r12 + object_offset]
                    ; mov rsi, QWORD index as _
                    ; lea rdx, [r12 + value_offset]
                    ; mov rax, QWORD jit_set_field_indexed_safe as *const () as _
                    ; call rax
                    ; test al, al
                    ; jz >fail
                );
            } else {
                // Strong field - can skip canonicalization
                unsafe extern "C" {
                    fn jit_set_field_strong_safe(
                        object_ptr: *const Value,
                        field_index: usize,
                        value_ptr: *const Value,
                    ) -> u8;
                }
                dynasm!(self.ops
                    ; .arch x64
                    ; lea rdi, [r12 + object_offset]
                    ; mov rsi, QWORD index as _
                    ; lea rdx, [r12 + value_offset]
                    ; mov rax, QWORD jit_set_field_strong_safe as *const () as _
                    ; call rax
                    ; test al, al
                    ; jz >fail
                );
            }
        } else {
            let (field_name_ptr, field_name_len) = self.retain_string(field_name);
            dynasm!(self.ops
                ; .arch x64
                ; lea rdi, [r12 + object_offset]
                ; mov rsi, QWORD field_name_ptr as _
                ; mov rdx, QWORD field_name_len as _
                ; lea rcx, [r12 + value_offset]
                ; mov rax, QWORD jit_set_field_safe as *const () as _
                ; call rax
                ; test al, al
                ; jz >fail
            );
        }

        Ok(())
    }

    pub(super) fn compile_new_array(
        &mut self,
        dest: u8,
        first_element: u8,
        count: u8,
    ) -> Result<()> {
        let value_size = mem::size_of::<Value>() as i32;
        let dest_offset = (dest as i32) * value_size;
        let first_elem_offset = (first_element as i32) * value_size;
        let count_usize = count as usize;

        unsafe extern "C" {
            fn jit_new_array_safe(
                vm_ptr: *mut crate::VM,
                elements_ptr: *const Value,
                element_count: usize,
                out: *mut Value,
            ) -> u8;
        }

        // r12 is callee-saved per System V ABI, so we don't need to save it
        dynasm!(self.ops
            ; .arch x64
            ; mov rdi, r13                           // vm_ptr
            ; lea rsi, [r12 + first_elem_offset]     // elements_ptr (ignored if count == 0)
            ; mov rdx, QWORD count_usize as _        // element_count
            ; lea rcx, [r12 + dest_offset]           // out_ptr
            ; mov rax, QWORD jit_new_array_safe as *const () as _
            ; call rax
            ; test al, al
            ; jz >fail
        );

        Ok(())
    }

    pub(super) fn compile_array_push(&mut self, array: u8, value: u8) -> Result<()> {
        let array_offset = (array as i32) * (mem::size_of::<Value>() as i32);
        let value_offset = (value as i32) * (mem::size_of::<Value>() as i32);

        unsafe extern "C" {
            fn jit_array_push_safe(
                vm_ptr: *mut crate::VM,
                array_ptr: *const Value,
                value_ptr: *const Value,
            ) -> u8;
        }

        // Guards have already verified the type, so directly call the helper
        dynasm!(self.ops
            ; .arch x64
            ; mov rdi, r13
            ; lea rsi, [r12 + array_offset]
            ; lea rdx, [r12 + value_offset]
            ; mov rax, QWORD jit_array_push_safe as *const () as _
            ; call rax
            ; test al, al
            ; jz >fail
        );

        Ok(())
    }

    pub(super) fn compile_enum_is_some(&mut self, dest: u8, enum_reg: u8) -> Result<()> {
        let enum_offset = (enum_reg as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);

        unsafe extern "C" {
            fn jit_enum_is_some_safe(enum_ptr: *const Value, out_ptr: *mut Value) -> u8;
        }

        dynasm!(self.ops
            ; .arch x64
            ; lea rdi, [r12 + enum_offset]
            ; lea rsi, [r12 + dest_offset]
            ; mov rax, QWORD jit_enum_is_some_safe as *const () as _
            ; call rax
            ; test al, al
            ; jz >fail
        );

        Ok(())
    }

    pub(super) fn compile_enum_unwrap(&mut self, dest: u8, enum_reg: u8) -> Result<()> {
        let enum_offset = (enum_reg as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);

        unsafe extern "C" {
            fn jit_enum_unwrap_safe(
                vm_ptr: *mut crate::VM,
                enum_ptr: *const Value,
                out_ptr: *mut Value,
            ) -> u8;
        }

        dynasm!(self.ops
            ; .arch x64
            ; mov rdi, r13
            ; lea rsi, [r12 + enum_offset]
            ; lea rdx, [r12 + dest_offset]
            ; mov rax, QWORD jit_enum_unwrap_safe as *const () as _
            ; call rax
            ; test al, al
            ; jz >fail
        );

        Ok(())
    }

    /// `JIT_CALL_IP = ip of the call being compiled`, for the helper about
    /// to run it (unknown inside an inlined body, whose frame the
    /// interpreter does not have). Clobbers rax/r10 only.
    fn emit_call_ip(&mut self) {
        let ip = if self.inline_depth == 0 {
            self.current_fail_ip.unwrap_or(usize::MAX)
        } else {
            usize::MAX
        };
        dynasm!(self.ops
            ; .arch x64
            ; mov r10, QWORD ip as i64
            ; mov [r13 + jit::CALL_IP_OFFSET as i32], r10
        );
    }

    pub(super) fn compile_call_native(
        &mut self,
        dest: u8,
        callee: u8,
        expected_ptr: *const (),
        first_arg: u8,
        arg_count: u8,
    ) -> Result<()> {
        let callee_offset = (callee as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        let first_arg_offset = (first_arg as i32) * (mem::size_of::<Value>() as i32);
        let arg_count_i32 = arg_count as i32;
        let exit_label = self.current_exit_label();
        unsafe extern "C" {
            fn jit_call_native_safe(
                vm_ptr: *mut crate::VM,
                callee_ptr: *const Value,
                expected: *const (),
                args_ptr: *const Value,
                arg_count: u8,
                out: *mut Value,
            ) -> u8;
        }

        self.emit_call_ip();
        dynasm!(self.ops
            ; .arch x64
            ; mov rdi, r13
            ; lea rsi, [r12 + callee_offset]
            ; mov rdx, QWORD expected_ptr as _
            ; lea rcx, [r12 + first_arg_offset]
            ; mov r8, DWORD arg_count_i32
            ; lea r9, [r12 + dest_offset]
            ; mov rax, QWORD jit_call_native_safe as *const () as _
            ; call rax
            ; cmp al, BYTE 2
            ; je >native_yield
            ; cmp al, BYTE 3
            ; je >native_yield
            ; test al, al
            ; jz >fail
            ; jmp >native_ok
            ; native_yield:
            ; jmp => exit_label
            ; native_ok:
        );

        Ok(())
    }

    pub(super) fn compile_call_function(
        &mut self,
        dest: u8,
        callee: u8,
        _function_idx: usize,
        first_arg: u8,
        arg_count: u8,
        is_closure: bool,
        upvalues_ptr: Option<*const ()>,
    ) -> Result<()> {
        let _ = (_function_idx, is_closure, upvalues_ptr);
        let callee_offset = (callee as i32) * (mem::size_of::<Value>() as i32);
        let first_arg_offset = (first_arg as i32) * (mem::size_of::<Value>() as i32);
        let arg_count_i32 = arg_count as i32;
        let dest_i32 = dest as i32;
        unsafe extern "C" {
            fn jit_call_function_safe(
                vm_ptr: *mut crate::VM,
                callee_ptr: *const Value,
                args_ptr: *const Value,
                arg_count: u8,
                dest_reg: u8,
                out: *mut Value,
            ) -> u8;
            fn jit_current_registers(vm_ptr: *mut crate::VM) -> *mut Value;
        }

        dynasm!(self.ops
            ; .arch x64
            ; mov rdi, r13
            ; lea rsi, [r12 + callee_offset]
            ; lea rdx, [r12 + first_arg_offset]
            ; mov ecx, DWORD arg_count_i32
            ; mov r8d, DWORD dest_i32
        );
        // Inside an inlined frame the destination lives on our stack, and
        // function code's frame is wherever r12 points: hand the helper
        // its address. At depth zero in a trace the VM frame may move.
        if self.inline_depth > 0 || self.function_mode {
            let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
            dynasm!(self.ops ; .arch x64 ; lea r9, [r12 + dest_offset]);
        } else {
            dynasm!(self.ops ; .arch x64 ; xor r9d, r9d);
        }
        self.emit_call_ip();
        dynasm!(self.ops
            ; .arch x64
            ; mov rax, QWORD jit_call_function_safe as *const () as _
            ; call rax
            ; test al, al
            ; jz >fail
        );

        if self.inline_depth == 0 && !self.function_mode {
            dynasm!(self.ops
                ; .arch x64
                ; mov rdi, r13
                ; mov rax, QWORD jit_current_registers as *const () as _
                ; call rax
                ; test rax, rax
                ; jz >fail
                ; mov r12, rax
            );
        }
        Ok(())
    }

    pub(super) fn compile_call_method(
        &mut self,
        dest: u8,
        object: u8,
        method_name: &str,
        first_arg: u8,
        arg_count: u8,
    ) -> Result<()> {
        let object_offset = (object as i32) * (mem::size_of::<Value>() as i32);
        let dest_i32 = dest as i32;
        unsafe extern "C" {
            fn jit_call_method_safe(
                vm_ptr: *mut crate::VM,
                object_ptr: *const Value,
                method_name_ptr: *const u8,
                method_name_len: usize,
                args_ptr: *const Value,
                arg_count: u8,
                dest_reg: u8,
                out: *mut Value,
            ) -> u8;
            fn jit_current_registers(vm_ptr: *mut crate::VM) -> *mut Value;
        }

        let (method_name_ptr, method_name_len) = self.retain_string(method_name);
        let first_arg_offset = (first_arg as i32) * (mem::size_of::<Value>() as i32);
        let arg_count_i32 = arg_count as i32;
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops
            ; .arch x64
            ; mov rdi, r13
            ; lea rsi, [r12 + object_offset]
            ; mov rdx, QWORD method_name_ptr as _
            ; mov rcx, QWORD method_name_len as _
            ; lea r8, [r12 + first_arg_offset]
            ; mov r9d, DWORD arg_count_i32
            ; sub rsp, 16
            ; mov rax, QWORD dest_i32 as i64
            ; mov [rsp], rax
        );
        // Eighth argument: result pointer for an inlined frame, else null.
        if self.inline_depth > 0 || self.function_mode {
            dynasm!(self.ops ; .arch x64 ; lea rax, [r12 + dest_offset] ; mov [rsp + 8], rax);
        } else {
            dynasm!(self.ops ; .arch x64 ; mov QWORD [rsp + 8], 0);
        }
        dynasm!(self.ops
            ; .arch x64
            ; mov rax, QWORD jit_call_method_safe as *const () as _
            ; call rax
            ; add rsp, 16
            ; test al, al
            ; jz >fail
        );

        if self.inline_depth == 0 && !self.function_mode {
            dynasm!(self.ops
                ; .arch x64
                ; mov rdi, r13
                ; mov rax, QWORD jit_current_registers as *const () as _
                ; call rax
                ; test rax, rax
                ; jz >fail
                ; mov r12, rax
            );
        }
        Ok(())
    }

    pub(super) fn compile_new_struct(
        &mut self,
        dest: u8,
        struct_name: &str,
        field_names: &[String],
        field_registers: &[u8],
    ) -> Result<()> {
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        unsafe extern "C" {
            fn jit_new_struct_safe(
                vm_ptr: *mut crate::VM,
                struct_name_ptr: *const u8,
                struct_name_len: usize,
                field_names_ptr: *const *const u8,
                field_name_lens_ptr: *const usize,
                field_values_ptr: *const Value,
                field_count: usize,
                out: *mut Value,
            ) -> u8;
            fn jit_current_registers(vm_ptr: *mut crate::VM) -> *mut Value;
        }

        let (struct_name_ptr, struct_name_len) = self.retain_string(struct_name);
        let field_count = field_names.len();
        let mut field_name_ptrs: Vec<*const u8> = Vec::new();
        let mut field_name_lens: Vec<usize> = Vec::new();
        for field_name in field_names {
            let (field_name_ptr, field_name_len) = self.retain_string(field_name);
            field_name_ptrs.push(field_name_ptr);
            field_name_lens.push(field_name_len);
        }

        let field_name_ptrs_ptr = self.retain_string_pointers(field_name_ptrs);
        let field_name_lens_ptr = self.retain_string_lengths(field_name_lens);
        let (has_fields, first_field_offset) = if let Some(first) = field_registers.first() {
            (true, (*first as i32) * (mem::size_of::<Value>() as i32))
        } else {
            (false, 0)
        };
        dynasm!(self.ops
            ; .arch x64
            ; mov rdi, r13
            ; mov rsi, QWORD struct_name_ptr as _
            ; mov rdx, QWORD struct_name_len as _
            ; mov rcx, QWORD field_name_ptrs_ptr as _
            ; mov r8, QWORD field_name_lens_ptr as _
        );
        if has_fields {
            dynasm!(self.ops
                ; .arch x64
                ; lea r9, [r12 + first_field_offset]
            );
        } else {
            dynasm!(self.ops
                ; .arch x64
                ; xor r9d, r9d
            );
        }
        dynasm!(self.ops
            ; .arch x64
            ; sub rsp, 16
            ; mov rax, QWORD field_count as _
            ; mov [rsp], rax
            ; lea rax, [r12 + dest_offset]
            ; mov [rsp + 8], rax
            ; mov rax, QWORD jit_new_struct_safe as *const () as _
            ; call rax
            ; add rsp, 16
            ; test al, al
            ; jz >fail
        );
        if self.inline_depth == 0 && !self.function_mode {
            dynasm!(self.ops
                ; .arch x64
                ; mov rdi, r13
                ; mov rax, QWORD jit_current_registers as *const () as _
                ; call rax
                ; test rax, rax
                ; jz >fail
                ; mov r12, rax
            );
        }
        Ok(())
    }

    pub(super) fn compile_new_enum_unit(
        &mut self,
        dest: u8,
        enum_name: &str,
        variant_name: &str,
    ) -> Result<()> {
        // One shared unit value per site: the store is a copy and a count
        // bump.
        let unit = self.retain_value(Value::enum_unit(enum_name, variant_name));
        dynasm!(self.ops ; .arch x64 ; mov rsi, QWORD unit as usize as _);
        self.emit_clone_rsi_into(dest);
        Ok(())
    }

    pub(super) fn compile_new_enum_variant(
        &mut self,
        dest: u8,
        enum_name: &str,
        variant_name: &str,
        value_registers: &[u8],
    ) -> Result<()> {
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        unsafe extern "C" {
            fn jit_new_enum_variant_safe(
                vm_ptr: *mut crate::VM,
                enum_name_ptr: *const u8,
                enum_name_len: usize,
                variant_name_ptr: *const u8,
                variant_name_len: usize,
                values_ptr: *const Value,
                value_count: usize,
                out: *mut Value,
            ) -> u8;
        }
        let (enum_name_ptr, enum_name_len) = self.retain_string(enum_name);
        let (variant_name_ptr, variant_name_len) = self.retain_string(variant_name);
        let (first_value_offset, value_count) = if let Some(first_reg) = value_registers.first() {
            (
                (*first_reg as i32) * (mem::size_of::<Value>() as i32),
                value_registers.len(),
            )
        } else {
            (0, 0)
        };
        dynasm!(self.ops
            ; .arch x64
            ; mov rdi, r13
            ; mov rsi, QWORD enum_name_ptr as _
            ; mov rdx, QWORD enum_name_len as _
            ; mov rcx, QWORD variant_name_ptr as _
            ; mov r8, QWORD variant_name_len as _
            ; lea r9, [r12 + first_value_offset]
            ; sub rsp, 32
            ; mov rax, QWORD value_count as _
            ; mov [rsp], rax
            ; lea rax, [r12 + dest_offset]
            ; mov [rsp + 8], rax
            ; mov rax, QWORD jit_new_enum_variant_safe as *const () as _
            ; call rax
            ; add rsp, 32
            ; test al, al
            ; jz >fail
        );
        Ok(())
    }

    pub(super) fn compile_is_enum_variant(
        &mut self,
        dest: u8,
        value: u8,
        enum_name: &str,
        variant_name: &str,
    ) -> Result<()> {
        let value_offset = (value as i32) * (mem::size_of::<Value>() as i32);
        unsafe extern "C" {
            fn jit_is_enum_variant_safe(
                value_ptr: *const Value,
                enum_name_ptr: *const u8,
                enum_name_len: usize,
                variant_name_ptr: *const u8,
                variant_name_len: usize,
            ) -> u8;
        }
        if let Some(layout) = jit::layout::enum_layout() {
            // Names are interned, so the variant test is a tag check and
            // pointer compares against the interned constants (the enum
            // name too, unless the test leaves it open).
            let variant_ptr = self.retain_name(variant_name);
            let enum_ptr = (!enum_name.is_empty()).then(|| self.retain_name(enum_name));
            let tag = layout.tag as i8;
            let object_offset = layout.object_offset as i32;
            let variant_offset = layout.variant_offset as i32;
            let enum_name_offset = layout.enum_name_offset as i32;
            dynasm!(self.ops
                ; .arch x64
                ; xor eax, eax
                ; cmp BYTE [r12 + value_offset], tag
                ; jne >done
                ; mov r9, [r12 + value_offset + object_offset]
                ; mov rcx, QWORD variant_ptr as i64
                ; cmp rcx, [r9 + variant_offset]
                ; jne >done
            );
            if let Some(enum_ptr) = enum_ptr {
                dynasm!(self.ops
                    ; .arch x64
                    ; mov rcx, QWORD enum_ptr as i64
                    ; cmp rcx, [r9 + enum_name_offset]
                    ; jne >done
                );
            }
            dynasm!(self.ops
                ; .arch x64
                ; mov eax, 1
                ; done:
            );
            self.store_from_rax(dest, 1);
            return Ok(());
        }
        let (enum_name_ptr, enum_name_len) = self.retain_string(enum_name);
        let (variant_name_ptr, variant_name_len) = self.retain_string(variant_name);
        dynasm!(self.ops
            ; .arch x64
            ; lea rdi, [r12 + value_offset]
            ; mov rsi, QWORD enum_name_ptr as _
            ; mov rdx, QWORD enum_name_len as _
            ; mov rcx, QWORD variant_name_ptr as _
            ; mov r8, QWORD variant_name_len as _
            ; mov rax, QWORD jit_is_enum_variant_safe as *const () as _
            ; call rax
            ; movzx rax, al
        );
        self.store_from_rax(dest, 1);
        Ok(())
    }

    pub(super) fn compile_type_is(&mut self, dest: u8, value: u8, type_name: &str) -> Result<()> {
        let value_offset = (value as i32) * (mem::size_of::<Value>() as i32);
        let direct_tag = match type_name {
            "nil" | "()" => Some(ValueTag::Nil),
            "bool" => Some(ValueTag::Bool),
            "int" => Some(ValueTag::Int),
            "float" => Some(ValueTag::Float),
            "string" => Some(ValueTag::String),
            "Array" => Some(ValueTag::Array),
            "Tuple" => Some(ValueTag::Tuple),
            "Map" => Some(ValueTag::Map),
            _ => None,
        };
        if type_name == "unknown" {
            dynasm!(self.ops ; .arch x64 ; mov rax, QWORD 1);
            self.store_from_rax(dest, 1);
            return Ok(());
        }
        if let Some(tag) = direct_tag {
            let tag = tag.as_u8() as i8;
            dynasm!(self.ops
                ; .arch x64
                ; cmp BYTE [r12 + value_offset], tag
                ; sete al
                ; movzx rax, al
            );
            self.store_from_rax(dest, 1);
            return Ok(());
        }

        unsafe extern "C" {
            fn jit_type_is_safe(
                vm_ptr: *mut crate::VM,
                value_ptr: *const Value,
                type_name_ptr: *const u8,
                type_name_len: usize,
            ) -> u8;
        }
        let (type_name_ptr, type_name_len) = self.retain_string(type_name);
        dynasm!(self.ops
            ; .arch x64
            ; mov rdi, r13
            ; lea rsi, [r12 + value_offset]
            ; mov rdx, QWORD type_name_ptr as _
            ; mov rcx, QWORD type_name_len as _
            ; mov rax, QWORD jit_type_is_safe as *const () as _
            ; call rax
            ; movzx rax, al
        );
        self.store_from_rax(dest, 1);
        Ok(())
    }

    pub(super) fn compile_try_cast(&mut self, dest: u8, value: u8, type_name: &str) -> Result<()> {
        let value_offset = (value as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        unsafe extern "C" {
            fn jit_try_cast_safe(
                vm_ptr: *mut crate::VM,
                value_ptr: *const Value,
                type_name_ptr: *const u8,
                type_name_len: usize,
                out: *mut Value,
            ) -> u8;
        }
        let (type_name_ptr, type_name_len) = self.retain_string(type_name);
        dynasm!(self.ops
            ; .arch x64
            ; mov rdi, r13
            ; lea rsi, [r12 + value_offset]
            ; mov rdx, QWORD type_name_ptr as _
            ; mov rcx, QWORD type_name_len as _
            ; lea r8, [r12 + dest_offset]
            ; mov rax, QWORD jit_try_cast_safe as *const () as _
            ; call rax
            ; test al, al
            ; jz >fail
        );
        Ok(())
    }

    /// `dest` = the bits of strong field `index` of the struct in `object`,
    /// with no count taken and nothing released (see
    /// `TraceOp::BorrowField`). The struct tag and field count are checked;
    /// anything else fails to the interpreter, which re-executes the read.
    pub(super) fn compile_borrow_field(&mut self, dest: u8, object: u8, index: usize) -> Result<()> {
        let Some(layout) = jit::layout::rc_vec_layout() else {
            return Err(crate::LustError::RuntimeError {
                message: "a borrow needs the measured struct layout".into(),
            });
        };
        let value_size = mem::size_of::<Value>() as i32;
        let object_offset = (object as i32) * value_size;
        let dest_offset = (dest as i32) * value_size;
        let struct_tag = ValueTag::Struct.as_u8() as i8;
        let fields_offset = layout.struct_fields_offset as i32;
        let len_offset = layout.struct_len_offset as i32;
        let ptr_offset = layout.struct_ptr_offset as i32;
        let element = (index * mem::size_of::<Value>()) as i32;
        dynasm!(self.ops
            ; .arch x64
            ; cmp BYTE [r12 + object_offset], struct_tag
            ; jne >fail
            ; mov r9, [r12 + object_offset + fields_offset]
            ; cmp QWORD [r9 + len_offset], index as i32
            ; jbe >fail
            ; mov r10, [r9 + ptr_offset]
        );
        for word in (0..value_size).step_by(8) {
            dynasm!(self.ops
                ; .arch x64
                ; mov rax, [r10 + element + word]
                ; mov [r12 + dest_offset + word], rax
            );
        }
        Ok(())
    }

    /// `dest` = the bits of payload value `index` of the enum in
    /// `enum_reg`, a borrow of a borrow (see `TraceOp::BorrowEnumValue`).
    pub(super) fn compile_borrow_enum_value(&mut self, dest: u8, enum_reg: u8, index: u8) -> Result<()> {
        let Some(layout) = jit::layout::enum_layout() else {
            return Err(crate::LustError::RuntimeError {
                message: "a borrow needs the measured enum layout".into(),
            });
        };
        let value_size = mem::size_of::<Value>() as i32;
        let enum_offset = (enum_reg as i32) * value_size;
        let dest_offset = (dest as i32) * value_size;
        let tag = layout.tag as i8;
        let object_offset = layout.object_offset as i32;
        let len_offset = layout.values_len_offset as i32;
        let ptr_offset = layout.values_ptr_offset as i32;
        let unit_offset = layout.unit_word_offset as i32;
        let element = (index as usize * mem::size_of::<Value>()) as i32;
        // A unit variant (no payload) fails: its niche word says so.
        dynasm!(self.ops
            ; .arch x64
            ; cmp BYTE [r12 + enum_offset], tag
            ; jne >fail
            ; mov r9, [r12 + enum_offset + object_offset]
            ; mov rax, QWORD layout.unit_word_value as i64
            ; cmp rax, [r9 + unit_offset]
            ; je >fail
            ; cmp QWORD [r9 + len_offset], index as i32
            ; jbe >fail
            ; mov r10, [r9 + ptr_offset]
        );
        for word in (0..value_size).step_by(8) {
            dynasm!(self.ops
                ; .arch x64
                ; mov rax, [r10 + element + word]
                ; mov [r12 + dest_offset + word], rax
            );
        }
        Ok(())
    }

    pub(super) fn compile_get_enum_value(
        &mut self,
        dest: u8,
        enum_reg: u8,
        index: u8,
    ) -> Result<()> {
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        let enum_offset = (enum_reg as i32) * (mem::size_of::<Value>() as i32);
        unsafe extern "C" {
            fn jit_get_enum_value_safe(enum_ptr: *const Value, index: usize, out: *mut Value)
            -> u8;
        }
        let index_usize = index as usize;
        if let Some(layout) = jit::layout::enum_layout() {
            // Inline: an enum with a payload of at least `index + 1` values
            // (anything else fails to the interpreter), the value cloned
            // into the register without the runtime.
            let tag = layout.tag as i8;
            let object_offset = layout.object_offset as i32;
            let len_offset = layout.values_len_offset as i32;
            let ptr_offset = layout.values_ptr_offset as i32;
            let unit_offset = layout.unit_word_offset as i32;
            let element = (index_usize * mem::size_of::<Value>()) as i32;
            // A unit variant (no payload) fails: its niche word says so.
            dynasm!(self.ops
                ; .arch x64
                ; cmp BYTE [r12 + enum_offset], tag
                ; jne >fail
                ; mov r9, [r12 + enum_offset + object_offset]
                ; mov rax, QWORD layout.unit_word_value as i64
                ; cmp rax, [r9 + unit_offset]
                ; je >fail
                ; cmp QWORD [r9 + len_offset], index as i32
                ; jbe >fail
                ; mov rsi, [r9 + ptr_offset]
                ; add rsi, element
            );
            self.emit_clone_rsi_into(dest);
            return Ok(());
        }
        dynasm!(self.ops
            ; .arch x64
            ; lea rdi, [r12 + enum_offset]
            ; mov rsi, QWORD index_usize as _
            ; lea rdx, [r12 + dest_offset]
            ; mov rax, QWORD jit_get_enum_value_safe as *const () as _
            ; call rax
            ; test al, al
            ; jz >fail
        );
        Ok(())
    }
}

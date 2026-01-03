use super::*;
impl JitCompiler {
    pub(super) fn compile_load_const(&mut self, dest: u8, value: &Value) -> Result<()> {
        match value {
            Value::Int(i) => {
                let offset = (dest as i32) * (mem::size_of::<Value>() as i32);
                dynasm!(self.ops
                    ; mov BYTE [r12 + offset], 2
                    ; mov rax, QWORD *i as _
                    ; mov [r12 + offset + 8], rax
                );
                Ok(())
            }

            Value::Float(f) => {
                let offset = (dest as i32) * (mem::size_of::<Value>() as i32);
                let f_bits = f.to_bits();
                dynasm!(self.ops
                    ; mov BYTE [r12 + offset], 3
                    ; mov rax, QWORD f_bits as _
                    ; mov [r12 + offset + 8], rax
                );
                Ok(())
            }

            Value::Bool(b) => {
                let offset = (dest as i32) * (mem::size_of::<Value>() as i32);
                let bool_val = if *b { 1i8 } else { 0i8 };
                dynasm!(self.ops
                    ; mov BYTE [r12 + offset], 1
                    ; mov BYTE [r12 + offset + 8], bool_val
                );
                Ok(())
            }

            Value::String(_) => self.copy_leaked_constant(dest, value),

            _ => self.copy_leaked_constant(dest, value),
        }
    }

    fn copy_leaked_constant(&mut self, dest: u8, value: &Value) -> Result<()> {
        let offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        let leaked_value = Box::leak(Box::new(value.clone()));
        let src_ptr = leaked_value as *const Value;
        self.leaked_constants.push(src_ptr);
        extern "C" {
            fn jit_move_safe(src_ptr: *const Value, dest_ptr: *mut Value) -> u8;
        }
        dynasm!(self.ops
            ; mov rdi, QWORD src_ptr as _
            ; lea rsi, [r12 + offset]
            ; mov rax, QWORD jit_move_safe as _
            ; call rax
            ; test al, al
            ; jz >fail
        );
        Ok(())
    }

    pub(super) fn compile_move(&mut self, dest: u8, src: u8) -> Result<()> {
        let src_offset = (src as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        extern "C" {
            fn jit_move_safe(src_ptr: *const Value, dest_ptr: *mut Value) -> u8;
        }

        dynasm!(self.ops
            ; mov al, [r12 + src_offset]
            ; cmp al, 3
            ; je >fast_copy
            ; cmp al, 2
            ; je >fast_copy
            ; cmp al, 1
            ; je >fast_copy
            ; cmp al, 0
            ; je >fast_copy
            ; jmp >call_helper
            ; fast_copy:
        );
        let value_size = mem::size_of::<Value>() as i32;
        let num_qwords = (value_size + 7) / 8;
        for i in 0..num_qwords {
            let chunk_offset = i * 8;
            dynasm!(self.ops
                ; mov rax, [r12 + src_offset + chunk_offset]
                ; mov [r12 + dest_offset + chunk_offset], rax
            );
        }

        dynasm!(self.ops
            ; jmp >move_done
            ; call_helper:
            ; lea rdi, [r12 + src_offset]
            ; lea rsi, [r12 + dest_offset]
            ; mov rax, QWORD jit_move_safe as _
            ; call rax
            ; move_done:
        );
        Ok(())
    }

    pub(super) fn compile_get_index(&mut self, dest: u8, array: u8, index: u8) -> Result<()> {
        let array_offset = (array as i32) * (mem::size_of::<Value>() as i32);
        let index_offset = (index as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        extern "C" {
            fn jit_array_get_safe(array_value: *const Value, index: i64, out: *mut Value) -> u8;
        }

        dynasm!(self.ops
            ; mov al, [r12 + array_offset]
            ; cmp al, 5
            ; jne >fail
            ; mov al, [r12 + index_offset]
            ; cmp al, 2
            ; jne >fail
            ; lea rdi, [r12 + array_offset]
            ; mov rsi, [r12 + index_offset + 8]
            ; lea rdx, [r12 + dest_offset]
            ; mov rax, QWORD jit_array_get_safe as _
            ; call rax
            ; test al, al
            ; jz >fail
        );
        Ok(())
    }

    pub(super) fn compile_array_len(&mut self, dest: u8, array: u8) -> Result<()> {
        let array_offset = (array as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        extern "C" {
            fn jit_array_len_safe(array_value: *const Value) -> i64;
        }

        dynasm!(self.ops
            ; mov al, [r12 + array_offset]
            ; cmp al, 5
            ; jne >fail
            ; lea rdi, [r12 + array_offset]
            ; mov rax, QWORD jit_array_len_safe as _
            ; call rax
            ; test rax, rax
            ; js >fail
            ; mov BYTE [r12 + dest_offset], 2
            ; mov [r12 + dest_offset + 8], rax
        );
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
        extern "C" {
            fn jit_get_field_safe(
                object_ptr: *const Value,
                field_name_ptr: *const u8,
                field_name_len: usize,
                out: *mut Value,
            ) -> u8;
            fn jit_get_field_indexed_safe(
                object_ptr: *const Value,
                field_index: usize,
                out: *mut Value,
            ) -> u8;
        }

        if let Some(index) = field_index {
            dynasm!(self.ops
                ; lea rdi, [r12 + object_offset]
                ; mov rsi, QWORD index as _
                ; lea rdx, [r12 + dest_offset]
                ; mov rax, QWORD jit_get_field_indexed_safe as _
                ; call rax
                ; test al, al
                ; jz >fail
            );
        } else {
            let field_name_box = Box::leak(Box::new(field_name.to_string()));
            let field_name_ptr = field_name_box.as_ptr();
            let field_name_len = field_name_box.len();
            dynasm!(self.ops
                ; lea rdi, [r12 + object_offset]
                ; mov rsi, QWORD field_name_ptr as _
                ; mov rdx, QWORD field_name_len as _
                ; lea rcx, [r12 + dest_offset]
                ; mov rax, QWORD jit_get_field_safe as _
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
        extern "C" {
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
                    ; lea rdi, [r12 + object_offset]
                    ; mov rsi, QWORD index as _
                    ; lea rdx, [r12 + value_offset]
                    ; mov rax, QWORD jit_set_field_indexed_safe as _
                    ; call rax
                    ; test al, al
                    ; jz >fail
                );
            } else {
                // Strong field - can skip canonicalization
                extern "C" {
                    fn jit_set_field_strong_safe(
                        object_ptr: *const Value,
                        field_index: usize,
                        value_ptr: *const Value,
                    ) -> u8;
                }
                dynasm!(self.ops
                    ; lea rdi, [r12 + object_offset]
                    ; mov rsi, QWORD index as _
                    ; lea rdx, [r12 + value_offset]
                    ; mov rax, QWORD jit_set_field_strong_safe as _
                    ; call rax
                    ; test al, al
                    ; jz >fail
                );
            }
        } else {
            let field_name_box = Box::leak(Box::new(field_name.to_string()));
            let field_name_ptr = field_name_box.as_ptr();
            let field_name_len = field_name_box.len();
            dynasm!(self.ops
                ; lea rdi, [r12 + object_offset]
                ; mov rsi, QWORD field_name_ptr as _
                ; mov rdx, QWORD field_name_len as _
                ; lea rcx, [r12 + value_offset]
                ; mov rax, QWORD jit_set_field_safe as _
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

        extern "C" {
            fn jit_new_array_safe(
                elements_ptr: *const Value,
                element_count: usize,
                out: *mut Value,
            ) -> u8;
        }

        // r12 is callee-saved per System V ABI, so we don't need to save it
        dynasm!(self.ops
            ; lea rdi, [r12 + first_elem_offset]    // elements_ptr (ignored if count == 0)
            ; mov rsi, QWORD count_usize as _        // element_count
            ; lea rdx, [r12 + dest_offset]           // out_ptr
            ; mov rax, QWORD jit_new_array_safe as _
            ; call rax
            ; test al, al
            ; jz >fail
        );

        Ok(())
    }

    pub(super) fn compile_array_push(&mut self, array: u8, value: u8) -> Result<()> {
        let array_offset = (array as i32) * (mem::size_of::<Value>() as i32);
        let value_offset = (value as i32) * (mem::size_of::<Value>() as i32);

        extern "C" {
            fn jit_array_push_safe(array_ptr: *const Value, value_ptr: *const Value) -> u8;
        }

        // Guards have already verified the type, so directly call the helper
        dynasm!(self.ops
            ; lea rdi, [r12 + array_offset]
            ; lea rsi, [r12 + value_offset]
            ; mov rax, QWORD jit_array_push_safe as _
            ; call rax
            ; test al, al
            ; jz >fail
        );

        Ok(())
    }

    pub(super) fn compile_enum_is_some(&mut self, dest: u8, enum_reg: u8) -> Result<()> {
        let enum_offset = (enum_reg as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);

        extern "C" {
            fn jit_enum_is_some_safe(enum_ptr: *const Value, out_ptr: *mut Value) -> u8;
        }

        dynasm!(self.ops
            ; lea rdi, [r12 + enum_offset]
            ; lea rsi, [r12 + dest_offset]
            ; mov rax, QWORD jit_enum_is_some_safe as _
            ; call rax
            ; test al, al
            ; jz >fail
        );

        Ok(())
    }

    pub(super) fn compile_enum_unwrap(&mut self, dest: u8, enum_reg: u8) -> Result<()> {
        let enum_offset = (enum_reg as i32) * (mem::size_of::<Value>() as i32);
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);

        extern "C" {
            fn jit_enum_unwrap_safe(enum_ptr: *const Value, out_ptr: *mut Value) -> u8;
        }

        dynasm!(self.ops
            ; lea rdi, [r12 + enum_offset]
            ; lea rsi, [r12 + dest_offset]
            ; mov rax, QWORD jit_enum_unwrap_safe as _
            ; call rax
            ; test al, al
            ; jz >fail
        );

        Ok(())
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
        extern "C" {
            fn jit_call_native_safe(
                vm_ptr: *mut crate::VM,
                callee_ptr: *const Value,
                expected: *const (),
                args_ptr: *const Value,
                arg_count: u8,
                out: *mut Value,
            ) -> u8;
        }

        dynasm!(self.ops
            ; mov rdi, r13
            ; lea rsi, [r12 + callee_offset]
            ; mov rdx, QWORD expected_ptr as _
            ; lea rcx, [r12 + first_arg_offset]
            ; mov r8, DWORD arg_count_i32
            ; lea r9, [r12 + dest_offset]
            ; mov rax, QWORD jit_call_native_safe as _
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
        extern "C" {
            fn jit_call_function_safe(
                vm_ptr: *mut crate::VM,
                callee_ptr: *const Value,
                args_ptr: *const Value,
                arg_count: u8,
                dest_reg: u8,
            ) -> u8;
            fn jit_current_registers(vm_ptr: *mut crate::VM) -> *mut Value;
        }

        dynasm!(self.ops
            ; mov rdi, r13
            ; lea rsi, [r12 + callee_offset]
            ; lea rdx, [r12 + first_arg_offset]
            ; mov ecx, DWORD arg_count_i32
            ; mov r8d, DWORD dest_i32
            ; mov rax, QWORD jit_call_function_safe as _
            ; call rax
            ; test al, al
            ; jz >fail
        );

        if self.inline_depth == 0 {
            dynasm!(self.ops
                ; mov rdi, r13
                ; mov rax, QWORD jit_current_registers as _
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
        extern "C" {
            fn jit_call_method_safe(
                vm_ptr: *mut crate::VM,
                object_ptr: *const Value,
                method_name_ptr: *const u8,
                method_name_len: usize,
                args_ptr: *const Value,
                arg_count: u8,
                dest_reg: u8,
            ) -> u8;
            fn jit_current_registers(vm_ptr: *mut crate::VM) -> *mut Value;
        }

        let method_name_box = Box::leak(Box::new(method_name.to_string()));
        let method_name_ptr = method_name_box.as_ptr();
        let method_name_len = method_name_box.len();
        let first_arg_offset = (first_arg as i32) * (mem::size_of::<Value>() as i32);
        let arg_count_i32 = arg_count as i32;
        dynasm!(self.ops
            ; mov rdi, r13
            ; lea rsi, [r12 + object_offset]
            ; mov rdx, QWORD method_name_ptr as _
            ; mov rcx, QWORD method_name_len as _
            ; lea r8, [r12 + first_arg_offset]
            ; mov r9d, DWORD arg_count_i32
            ; sub rsp, 16
            ; mov rax, QWORD dest_i32 as i64
            ; mov [rsp], rax
            ; mov rax, QWORD jit_call_method_safe as _
            ; call rax
            ; add rsp, 16
            ; test al, al
            ; jz >fail
        );

        if self.inline_depth == 0 {
            dynasm!(self.ops
                ; mov rdi, r13
                ; mov rax, QWORD jit_current_registers as _
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
        extern "C" {
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

        let struct_name_box = Box::leak(Box::new(struct_name.to_string()));
        let struct_name_ptr = struct_name_box.as_ptr();
        let struct_name_len = struct_name_box.len();
        let field_count = field_names.len();
        let mut field_name_ptrs: Vec<*const u8> = Vec::new();
        let mut field_name_lens: Vec<usize> = Vec::new();
        for field_name in field_names {
            let field_name_box = Box::leak(Box::new(field_name.to_string()));
            field_name_ptrs.push(field_name_box.as_ptr());
            field_name_lens.push(field_name_box.len());
        }

        let field_name_ptrs_box = Box::leak(field_name_ptrs.into_boxed_slice());
        let field_name_lens_box = Box::leak(field_name_lens.into_boxed_slice());
        let field_name_ptrs_ptr = field_name_ptrs_box.as_ptr();
        let field_name_lens_ptr = field_name_lens_box.as_ptr();
        let (has_fields, first_field_offset) = if let Some(first) = field_registers.first() {
            (true, (*first as i32) * (mem::size_of::<Value>() as i32))
        } else {
            (false, 0)
        };
        dynasm!(self.ops
            ; mov rdi, r13
            ; mov rsi, QWORD struct_name_ptr as _
            ; mov rdx, QWORD struct_name_len as _
            ; mov rcx, QWORD field_name_ptrs_ptr as _
            ; mov r8, QWORD field_name_lens_ptr as _
        );
        if has_fields {
            dynasm!(self.ops
                ; lea r9, [r12 + first_field_offset]
            );
        } else {
            dynasm!(self.ops
                ; xor r9d, r9d
            );
        }
        dynasm!(self.ops
            ; sub rsp, 16
            ; mov rax, QWORD field_count as _
            ; mov [rsp], rax
            ; lea rax, [r12 + dest_offset]
            ; mov [rsp + 8], rax
            ; mov rax, QWORD jit_new_struct_safe as _
            ; call rax
            ; add rsp, 16
            ; test al, al
            ; jz >fail
        );
        if self.inline_depth == 0 {
            dynasm!(self.ops
                ; mov rdi, r13
                ; mov rax, QWORD jit_current_registers as _
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
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        extern "C" {
            fn jit_new_enum_unit_safe(
                enum_name_ptr: *const u8,
                enum_name_len: usize,
                variant_name_ptr: *const u8,
                variant_name_len: usize,
                out: *mut Value,
            ) -> u8;
        }
        let enum_name_box = Box::leak(Box::new(enum_name.to_string()));
        let enum_name_ptr = enum_name_box.as_ptr();
        let enum_name_len = enum_name_box.len();
        let variant_name_box = Box::leak(Box::new(variant_name.to_string()));
        let variant_name_ptr = variant_name_box.as_ptr();
        let variant_name_len = variant_name_box.len();
        dynasm!(self.ops
            ; mov rdi, QWORD enum_name_ptr as _
            ; mov rsi, QWORD enum_name_len as _
            ; mov rdx, QWORD variant_name_ptr as _
            ; mov rcx, QWORD variant_name_len as _
            ; lea r8, [r12 + dest_offset]
            ; mov rax, QWORD jit_new_enum_unit_safe as _
            ; call rax
            ; test al, al
            ; jz >fail
        );
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
        extern "C" {
            fn jit_new_enum_variant_safe(
                enum_name_ptr: *const u8,
                enum_name_len: usize,
                variant_name_ptr: *const u8,
                variant_name_len: usize,
                values_ptr: *const Value,
                value_count: usize,
                out: *mut Value,
            ) -> u8;
        }
        let enum_name_box = Box::leak(Box::new(enum_name.to_string()));
        let enum_name_ptr = enum_name_box.as_ptr();
        let enum_name_len = enum_name_box.len();
        let variant_name_box = Box::leak(Box::new(variant_name.to_string()));
        let variant_name_ptr = variant_name_box.as_ptr();
        let variant_name_len = variant_name_box.len();
        let (first_value_offset, value_count) = if let Some(first_reg) = value_registers.first() {
            (
                (*first_reg as i32) * (mem::size_of::<Value>() as i32),
                value_registers.len(),
            )
        } else {
            (0, 0)
        };
        dynasm!(self.ops
            ; mov rdi, QWORD enum_name_ptr as _
            ; mov rsi, QWORD enum_name_len as _
            ; mov rdx, QWORD variant_name_ptr as _
            ; mov rcx, QWORD variant_name_len as _
            ; lea r8, [r12 + first_value_offset]
            ; mov r9, QWORD value_count as _
            ; sub rsp, 16
            ; lea rax, [r12 + dest_offset]
            ; mov [rsp], rax
            ; mov rax, QWORD jit_new_enum_variant_safe as _
            ; call rax
            ; add rsp, 16
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
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        let value_offset = (value as i32) * (mem::size_of::<Value>() as i32);
        extern "C" {
            fn jit_is_enum_variant_safe(
                value_ptr: *const Value,
                enum_name_ptr: *const u8,
                enum_name_len: usize,
                variant_name_ptr: *const u8,
                variant_name_len: usize,
            ) -> u8;
        }
        let enum_name_box = Box::leak(Box::new(enum_name.to_string()));
        let enum_name_ptr = enum_name_box.as_ptr();
        let enum_name_len = enum_name_box.len();
        let variant_name_box = Box::leak(Box::new(variant_name.to_string()));
        let variant_name_ptr = variant_name_box.as_ptr();
        let variant_name_len = variant_name_box.len();
        dynasm!(self.ops
            ; lea rdi, [r12 + value_offset]
            ; mov rsi, QWORD enum_name_ptr as _
            ; mov rdx, QWORD enum_name_len as _
            ; mov rcx, QWORD variant_name_ptr as _
            ; mov r8, QWORD variant_name_len as _
            ; mov rax, QWORD jit_is_enum_variant_safe as _
            ; call rax
            ; test al, al
            ; jnz >store_true
            ; mov BYTE [r12 + dest_offset], 1
            ; mov QWORD [r12 + dest_offset + 8], 0
            ; jmp >store_done
            ; store_true:
            ; mov BYTE [r12 + dest_offset], 1
            ; mov QWORD [r12 + dest_offset + 8], 0
            ; mov BYTE [r12 + dest_offset + 8], 1
            ; store_done:
        );
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
        extern "C" {
            fn jit_get_enum_value_safe(enum_ptr: *const Value, index: usize, out: *mut Value)
                -> u8;
        }
        let index_usize = index as usize;
        dynasm!(self.ops
            ; lea rdi, [r12 + enum_offset]
            ; mov rsi, QWORD index_usize as _
            ; lea rdx, [r12 + dest_offset]
            ; mov rax, QWORD jit_get_enum_value_safe as _
            ; call rax
            ; test al, al
            ; jz >fail
        );
        Ok(())
    }
}

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
        let value_size = mem::size_of::<Value>() as i32;
        let num_qwords = (value_size + 7) / 8;
        for i in 0..num_qwords {
            let chunk_offset = i * 8;
            let src_addr = (src_ptr as usize + chunk_offset as usize) as i64;
            dynasm!(self.ops
                ; mov rax, QWORD src_addr as _
                ; mov rax, [rax]
                ; mov [r12 + offset + chunk_offset], rax
            );
        }
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
            ; sub rsp, 8
            ; mov rax, QWORD jit_move_safe as _
            ; call rax
            ; add rsp, 8
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
            ; sub rsp, 8
            ; mov rax, QWORD jit_array_get_safe as _
            ; call rax
            ; add rsp, 8
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
            ; sub rsp, 8
            ; mov rax, QWORD jit_array_len_safe as _
            ; call rax
            ; add rsp, 8
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
        value_type: Option<ValueType>,
        is_weak: bool,
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
            fn jit_get_field_indexed_int_fast(
                object_ptr: *const Value,
                field_index: usize,
                out: *mut Value,
            ) -> u8;
        }

        if let Some(index) = field_index {
            if !is_weak && matches!(value_type, Some(ValueType::Int)) {
                dynasm!(self.ops
                    ; lea rdi, [r12 + object_offset]
                    ; mov rsi, QWORD index as _
                    ; lea rdx, [r12 + dest_offset]
                    ; sub rsp, 8
                    ; mov rax, QWORD jit_get_field_indexed_int_fast as _
                    ; call rax
                    ; add rsp, 8
                    ; test al, al
                    ; jz >fail
                );
                return Ok(());
            }

            dynasm!(self.ops
                ; lea rdi, [r12 + object_offset]
                ; mov rsi, QWORD index as _
                ; lea rdx, [r12 + dest_offset]
                ; sub rsp, 8
                ; mov rax, QWORD jit_get_field_indexed_safe as _
                ; call rax
                ; add rsp, 8
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
                ; sub rsp, 8
                ; mov rax, QWORD jit_get_field_safe as _
                ; call rax
                ; add rsp, 8
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
        value_type: Option<ValueType>,
        is_weak: bool,
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
            fn jit_set_field_indexed_int_fast(
                object_ptr: *const Value,
                field_index: usize,
                value_ptr: *const Value,
            ) -> u8;
        }

        if let Some(index) = field_index {
            if !is_weak && matches!(value_type, Some(ValueType::Int)) {
                dynasm!(self.ops
                    ; lea rdi, [r12 + object_offset]
                    ; mov rsi, QWORD index as _
                    ; lea rdx, [r12 + value_offset]
                    ; sub rsp, 8
                    ; mov rax, QWORD jit_set_field_indexed_int_fast as _
                    ; call rax
                    ; add rsp, 8
                    ; test al, al
                    ; jz >fail
                );
                return Ok(());
            }

            dynasm!(self.ops
                ; lea rdi, [r12 + object_offset]
                ; mov rsi, QWORD index as _
                ; lea rdx, [r12 + value_offset]
                ; sub rsp, 8
                ; mov rax, QWORD jit_set_field_indexed_safe as _
                ; call rax
                ; add rsp, 8
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
                ; lea rcx, [r12 + value_offset]
                ; sub rsp, 8
                ; mov rax, QWORD jit_set_field_safe as _
                ; call rax
                ; add rsp, 8
                ; test al, al
                ; jz >fail
            );
        }

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
            ; sub rsp, 8
            ; mov rax, QWORD jit_call_native_safe as _
            ; call rax
            ; add rsp, 8
            ; test al, al
            ; jz >fail
        );

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
        let dest_offset = (dest as i32) * (mem::size_of::<Value>() as i32);
        extern "C" {
            fn jit_call_method_safe(
                object_ptr: *const Value,
                method_name_ptr: *const u8,
                method_name_len: usize,
                args_ptr: *const Value,
                arg_count: u8,
                out: *mut Value,
            ) -> u8;
        }

        let method_name_box = Box::leak(Box::new(method_name.to_string()));
        let method_name_ptr = method_name_box.as_ptr();
        let method_name_len = method_name_box.len();
        let first_arg_offset = (first_arg as i32) * (mem::size_of::<Value>() as i32);
        let arg_count_i32 = arg_count as i32;
        dynasm!(self.ops
            ; lea rdi, [r12 + object_offset]
            ; mov rsi, QWORD method_name_ptr as _
            ; mov rdx, QWORD method_name_len as _
            ; lea rcx, [r12 + first_arg_offset]
            ; mov r8, DWORD arg_count_i32
            ; lea r9, [r12 + dest_offset]
            ; sub rsp, 8
            ; mov rax, QWORD jit_call_method_safe as _
            ; call rax
            ; add rsp, 8
            ; test al, al
            ; jz >fail
        );
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
                struct_name_ptr: *const u8,
                struct_name_len: usize,
                field_names_ptr: *const *const u8,
                field_name_lens_ptr: *const usize,
                field_values_ptr: *const Value,
                field_count: usize,
                out: *mut Value,
            ) -> u8;
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
        let first_field_reg = field_registers[0];
        let first_field_offset = (first_field_reg as i32) * (mem::size_of::<Value>() as i32);
        dynasm!(self.ops
            ; mov rdi, QWORD struct_name_ptr as _
            ; mov rsi, QWORD struct_name_len as _
            ; mov rdx, QWORD field_name_ptrs_ptr as _
            ; mov rcx, QWORD field_name_lens_ptr as _
            ; lea r8, [r12 + first_field_offset]
            ; mov r9, QWORD field_count as _
            ; sub rsp, 8
            ; lea rax, [r12 + dest_offset]
            ; push rax
            ; mov rax, QWORD jit_new_struct_safe as _
            ; call rax
            ; add rsp, 16
            ; test al, al
            ; jz >fail
        );
        Ok(())
    }
}

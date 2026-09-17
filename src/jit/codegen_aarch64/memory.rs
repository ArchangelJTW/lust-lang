use super::*;
impl JitCompiler {
    pub(super) fn compile_load_const(&mut self, dest: u8, value: &Value) -> Result<()> {
        match value {
            Value::Int(i) => {
                self.emit_mov_imm64(0, *i as u64);
                self.store_from_x0(dest, ValueTag::Int.as_u8());
                Ok(())
            }

            Value::Float(f) => {
                self.emit_mov_imm64(0, f.to_bits());
                dynasm!(self.ops ; .arch aarch64 ; fmov d0, x0);
                self.store_d0_as_float(dest);
                Ok(())
            }

            Value::Bool(b) => {
                let bool_val = u32::from(*b);
                dynasm!(self.ops ; .arch aarch64 ; movz x0, #bool_val);
                self.store_from_x0(dest, ValueTag::Bool.as_u8());
                Ok(())
            }

            _ => self.copy_owned_constant(dest, value),
        }
    }

    fn copy_owned_constant(&mut self, dest: u8, value: &Value) -> Result<()> {
        let src_ptr = self.retain_value(value.clone());
        unsafe extern "C" {
            fn jit_move_safe(src_ptr: *const Value, dest_ptr: *mut Value) -> u8;
        }
        self.emit_mov_imm64(0, src_ptr as usize as u64);
        self.emit_reg_addr(1, dest);
        self.emit_call(jit_move_safe as *const ());
        self.emit_fail_if_w0_zero();
        Ok(())
    }

    pub(super) fn compile_move(&mut self, dest: u8, src: u8) -> Result<()> {
        unsafe extern "C" {
            fn jit_move_safe(src_ptr: *const Value, dest_ptr: *mut Value) -> u8;
        }
        self.emit_reg_addr(0, src);
        self.emit_reg_addr(1, dest);
        self.emit_call(jit_move_safe as *const ());
        Ok(())
    }

    pub(super) fn compile_get_index(&mut self, dest: u8, array: u8, index: u8) -> Result<()> {
        let array_tag = ValueTag::Array.as_u8() as u32;
        let int_tag = ValueTag::Int.as_u8() as u32;
        unsafe extern "C" {
            fn jit_array_get_safe(
                vm_ptr: *mut crate::VM,
                array_value: *const Value,
                index_value: *const Value,
                out: *mut Value,
            ) -> u8;
        }

        self.load_tag(0, array);
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp w0, #array_tag
            ; b.ne >fail
        );
        self.load_tag(0, index);
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp w0, #int_tag
            ; b.ne >fail
            ; mov x0, x20
        );
        self.emit_reg_addr(1, array);
        self.emit_reg_addr(2, index);
        self.emit_reg_addr(3, dest);
        self.emit_call(jit_array_get_safe as *const ());
        self.emit_fail_if_w0_zero();
        Ok(())
    }

    pub(super) fn compile_try_get_index(&mut self, dest: u8, array: u8, index: u8) -> Result<()> {
        unsafe extern "C" {
            fn jit_array_index_result_safe(
                vm_ptr: *mut crate::VM,
                array_value: *const Value,
                index_value: *const Value,
                out: *mut Value,
            ) -> u8;
        }

        dynasm!(self.ops ; .arch aarch64 ; mov x0, x20);
        self.emit_reg_addr(1, array);
        self.emit_reg_addr(2, index);
        self.emit_reg_addr(3, dest);
        self.emit_call(jit_array_index_result_safe as *const ());
        self.emit_fail_if_w0_zero();
        Ok(())
    }

    pub(super) fn compile_array_index_ok(
        &mut self,
        value_dest: u8,
        condition_dest: u8,
        array: u8,
        index: u8,
    ) -> Result<()> {
        unsafe extern "C" {
            fn jit_array_index_ok_safe(
                array_value: *const Value,
                index_value: *const Value,
                value_out: *mut Value,
                condition_out: *mut Value,
            ) -> u8;
        }

        self.emit_reg_addr(0, array);
        self.emit_reg_addr(1, index);
        self.emit_reg_addr(2, value_dest);
        self.emit_reg_addr(3, condition_dest);
        self.emit_call(jit_array_index_ok_safe as *const ());
        self.emit_fail_if_w0_zero();
        Ok(())
    }

    pub(super) fn compile_array_len(&mut self, dest: u8, array: u8) -> Result<()> {
        let array_tag = ValueTag::Array.as_u8() as u32;
        unsafe extern "C" {
            fn jit_array_len_safe(array_value: *const Value) -> i64;
        }

        self.load_tag(0, array);
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp w0, #array_tag
            ; b.ne >fail
        );
        self.emit_reg_addr(0, array);
        self.emit_call(jit_array_len_safe as *const ());
        dynasm!(self.ops
            ; .arch aarch64
            ; tbnz x0, 63, >fail
        );
        self.store_from_x0(dest, ValueTag::Int.as_u8());
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
        unsafe extern "C" {
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
            self.emit_reg_addr(0, object);
            self.emit_mov_imm64(1, index as u64);
            self.emit_reg_addr(2, dest);
            self.emit_call(jit_get_field_indexed_safe as *const ());
        } else {
            let (field_name_ptr, field_name_len) = self.retain_string(field_name);
            self.emit_reg_addr(0, object);
            self.emit_mov_imm64(1, field_name_ptr as usize as u64);
            self.emit_mov_imm64(2, field_name_len as u64);
            self.emit_reg_addr(3, dest);
            self.emit_call(jit_get_field_safe as *const ());
        }
        self.emit_fail_if_w0_zero();

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
            fn jit_set_field_strong_safe(
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
            // Weak fields need canonicalization; strong fields can skip it.
            let helper: *const () = if _is_weak {
                jit_set_field_indexed_safe as *const ()
            } else {
                jit_set_field_strong_safe as *const ()
            };
            self.emit_reg_addr(0, object);
            self.emit_mov_imm64(1, index as u64);
            self.emit_reg_addr(2, value);
            self.emit_call(helper);
        } else {
            let (field_name_ptr, field_name_len) = self.retain_string(field_name);
            self.emit_reg_addr(0, object);
            self.emit_mov_imm64(1, field_name_ptr as usize as u64);
            self.emit_mov_imm64(2, field_name_len as u64);
            self.emit_reg_addr(3, value);
            self.emit_call(jit_set_field_safe as *const ());
        }
        self.emit_fail_if_w0_zero();

        Ok(())
    }

    pub(super) fn compile_new_array(
        &mut self,
        dest: u8,
        first_element: u8,
        count: u8,
    ) -> Result<()> {
        unsafe extern "C" {
            fn jit_new_array_safe(
                vm_ptr: *mut crate::VM,
                elements_ptr: *const Value,
                element_count: usize,
                out: *mut Value,
            ) -> u8;
        }

        dynasm!(self.ops ; .arch aarch64 ; mov x0, x20);
        self.emit_reg_addr(1, first_element); // ignored by the helper if count == 0
        self.emit_mov_imm64(2, count as u64);
        self.emit_reg_addr(3, dest);
        self.emit_call(jit_new_array_safe as *const ());
        self.emit_fail_if_w0_zero();

        Ok(())
    }

    pub(super) fn compile_array_push(&mut self, array: u8, value: u8) -> Result<()> {
        unsafe extern "C" {
            fn jit_array_push_safe(
                vm_ptr: *mut crate::VM,
                array_ptr: *const Value,
                value_ptr: *const Value,
            ) -> u8;
        }

        // Guards have already verified the type, so directly call the helper
        dynasm!(self.ops ; .arch aarch64 ; mov x0, x20);
        self.emit_reg_addr(1, array);
        self.emit_reg_addr(2, value);
        self.emit_call(jit_array_push_safe as *const ());
        self.emit_fail_if_w0_zero();

        Ok(())
    }

    pub(super) fn compile_enum_is_some(&mut self, dest: u8, enum_reg: u8) -> Result<()> {
        unsafe extern "C" {
            fn jit_enum_is_some_safe(enum_ptr: *const Value, out_ptr: *mut Value) -> u8;
        }

        self.emit_reg_addr(0, enum_reg);
        self.emit_reg_addr(1, dest);
        self.emit_call(jit_enum_is_some_safe as *const ());
        self.emit_fail_if_w0_zero();

        Ok(())
    }

    pub(super) fn compile_enum_unwrap(&mut self, dest: u8, enum_reg: u8) -> Result<()> {
        unsafe extern "C" {
            fn jit_enum_unwrap_safe(
                vm_ptr: *mut crate::VM,
                enum_ptr: *const Value,
                out_ptr: *mut Value,
            ) -> u8;
        }

        dynasm!(self.ops ; .arch aarch64 ; mov x0, x20);
        self.emit_reg_addr(1, enum_reg);
        self.emit_reg_addr(2, dest);
        self.emit_call(jit_enum_unwrap_safe as *const ());
        self.emit_fail_if_w0_zero();

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

        dynasm!(self.ops ; .arch aarch64 ; mov x0, x20);
        self.emit_reg_addr(1, callee);
        self.emit_mov_imm64(2, expected_ptr as usize as u64);
        self.emit_reg_addr(3, first_arg);
        self.emit_mov_imm32(4, arg_count as u32);
        self.emit_reg_addr(5, dest);
        self.emit_call(jit_call_native_safe as *const ());
        // 0 = failure, 1 = ok, 2/3 = yield/task signal (exit with that code).
        dynasm!(self.ops
            ; .arch aarch64
            ; and w0, w0, 0xff
            ; cmp w0, 2
            ; b.eq >native_yield
            ; cmp w0, 3
            ; b.eq >native_yield
            ; cbz w0, >fail
            ; b >native_ok
            ; native_yield:
            ; b => exit_label
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
        unsafe extern "C" {
            fn jit_call_function_safe(
                vm_ptr: *mut crate::VM,
                callee_ptr: *const Value,
                args_ptr: *const Value,
                arg_count: u8,
                dest_reg: u8,
            ) -> u8;
        }

        dynasm!(self.ops ; .arch aarch64 ; mov x0, x20);
        self.emit_reg_addr(1, callee);
        self.emit_reg_addr(2, first_arg);
        self.emit_mov_imm32(3, arg_count as u32);
        self.emit_mov_imm32(4, dest as u32);
        self.emit_call(jit_call_function_safe as *const ());
        self.emit_fail_if_w0_zero();
        self.emit_reload_registers_base();
        Ok(())
    }

    /// The VM may have reallocated its register file during a call; refresh
    /// x19 from the VM unless we are inside an inlined frame (whose registers
    /// live on our own stack).
    fn emit_reload_registers_base(&mut self) {
        unsafe extern "C" {
            fn jit_current_registers(vm_ptr: *mut crate::VM) -> *mut Value;
        }
        if self.inline_depth == 0 {
            dynasm!(self.ops ; .arch aarch64 ; mov x0, x20);
            self.emit_call(jit_current_registers as *const ());
            dynasm!(self.ops
                ; .arch aarch64
                ; cbz x0, >fail
                ; mov x19, x0
            );
        }
    }

    pub(super) fn compile_call_method(
        &mut self,
        dest: u8,
        object: u8,
        method_name: &str,
        first_arg: u8,
        arg_count: u8,
    ) -> Result<()> {
        unsafe extern "C" {
            fn jit_call_method_safe(
                vm_ptr: *mut crate::VM,
                object_ptr: *const Value,
                method_name_ptr: *const u8,
                method_name_len: usize,
                args_ptr: *const Value,
                arg_count: u8,
                dest_reg: u8,
            ) -> u8;
        }

        let (method_name_ptr, method_name_len) = self.retain_string(method_name);
        dynasm!(self.ops ; .arch aarch64 ; mov x0, x20);
        self.emit_reg_addr(1, object);
        self.emit_mov_imm64(2, method_name_ptr as usize as u64);
        self.emit_mov_imm64(3, method_name_len as u64);
        self.emit_reg_addr(4, first_arg);
        self.emit_mov_imm32(5, arg_count as u32);
        self.emit_mov_imm32(6, dest as u32);
        self.emit_call(jit_call_method_safe as *const ());
        self.emit_fail_if_w0_zero();
        self.emit_reload_registers_base();
        Ok(())
    }

    pub(super) fn compile_new_struct(
        &mut self,
        dest: u8,
        struct_name: &str,
        field_names: &[String],
        field_registers: &[u8],
    ) -> Result<()> {
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
        dynasm!(self.ops ; .arch aarch64 ; mov x0, x20);
        self.emit_mov_imm64(1, struct_name_ptr as usize as u64);
        self.emit_mov_imm64(2, struct_name_len as u64);
        self.emit_mov_imm64(3, field_name_ptrs_ptr as usize as u64);
        self.emit_mov_imm64(4, field_name_lens_ptr as usize as u64);
        if let Some(first) = field_registers.first() {
            self.emit_reg_addr(5, *first);
        } else {
            dynasm!(self.ops ; .arch aarch64 ; mov x5, xzr);
        }
        self.emit_mov_imm64(6, field_count as u64);
        self.emit_reg_addr(7, dest);
        self.emit_call(jit_new_struct_safe as *const ());
        self.emit_fail_if_w0_zero();
        self.emit_reload_registers_base();
        Ok(())
    }

    pub(super) fn compile_new_enum_unit(
        &mut self,
        dest: u8,
        enum_name: &str,
        variant_name: &str,
    ) -> Result<()> {
        unsafe extern "C" {
            fn jit_new_enum_unit_safe(
                vm_ptr: *mut crate::VM,
                enum_name_ptr: *const u8,
                enum_name_len: usize,
                variant_name_ptr: *const u8,
                variant_name_len: usize,
                out: *mut Value,
            ) -> u8;
        }
        let (enum_name_ptr, enum_name_len) = self.retain_string(enum_name);
        let (variant_name_ptr, variant_name_len) = self.retain_string(variant_name);
        dynasm!(self.ops ; .arch aarch64 ; mov x0, x20);
        self.emit_mov_imm64(1, enum_name_ptr as usize as u64);
        self.emit_mov_imm64(2, enum_name_len as u64);
        self.emit_mov_imm64(3, variant_name_ptr as usize as u64);
        self.emit_mov_imm64(4, variant_name_len as u64);
        self.emit_reg_addr(5, dest);
        self.emit_call(jit_new_enum_unit_safe as *const ());
        self.emit_fail_if_w0_zero();
        Ok(())
    }

    pub(super) fn compile_new_enum_variant(
        &mut self,
        dest: u8,
        enum_name: &str,
        variant_name: &str,
        value_registers: &[u8],
    ) -> Result<()> {
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
        let (first_value, value_count) = match value_registers.first() {
            Some(first) => (*first, value_registers.len()),
            None => (0, 0),
        };
        dynasm!(self.ops ; .arch aarch64 ; mov x0, x20);
        self.emit_mov_imm64(1, enum_name_ptr as usize as u64);
        self.emit_mov_imm64(2, enum_name_len as u64);
        self.emit_mov_imm64(3, variant_name_ptr as usize as u64);
        self.emit_mov_imm64(4, variant_name_len as u64);
        self.emit_reg_addr(5, first_value);
        self.emit_mov_imm64(6, value_count as u64);
        self.emit_reg_addr(7, dest);
        self.emit_call(jit_new_enum_variant_safe as *const ());
        self.emit_fail_if_w0_zero();
        Ok(())
    }

    pub(super) fn compile_is_enum_variant(
        &mut self,
        dest: u8,
        value: u8,
        enum_name: &str,
        variant_name: &str,
    ) -> Result<()> {
        unsafe extern "C" {
            fn jit_is_enum_variant_safe(
                value_ptr: *const Value,
                enum_name_ptr: *const u8,
                enum_name_len: usize,
                variant_name_ptr: *const u8,
                variant_name_len: usize,
            ) -> u8;
        }
        let (enum_name_ptr, enum_name_len) = self.retain_string(enum_name);
        let (variant_name_ptr, variant_name_len) = self.retain_string(variant_name);
        self.emit_reg_addr(0, value);
        self.emit_mov_imm64(1, enum_name_ptr as usize as u64);
        self.emit_mov_imm64(2, enum_name_len as u64);
        self.emit_mov_imm64(3, variant_name_ptr as usize as u64);
        self.emit_mov_imm64(4, variant_name_len as u64);
        self.emit_call(jit_is_enum_variant_safe as *const ());
        dynasm!(self.ops ; .arch aarch64 ; and x0, x0, 0xff);
        self.store_from_x0(dest, ValueTag::Bool.as_u8());
        Ok(())
    }

    pub(super) fn compile_type_is(&mut self, dest: u8, value: u8, type_name: &str) -> Result<()> {
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
            dynasm!(self.ops ; .arch aarch64 ; movz x0, 1);
            self.store_from_x0(dest, ValueTag::Bool.as_u8());
            return Ok(());
        }
        if let Some(tag) = direct_tag {
            let tag = tag.as_u8() as u32;
            self.load_tag(9, value);
            dynasm!(self.ops
                ; .arch aarch64
                ; cmp w9, #tag
                ; cset x0, eq
            );
            self.store_from_x0(dest, ValueTag::Bool.as_u8());
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
        dynasm!(self.ops ; .arch aarch64 ; mov x0, x20);
        self.emit_reg_addr(1, value);
        self.emit_mov_imm64(2, type_name_ptr as usize as u64);
        self.emit_mov_imm64(3, type_name_len as u64);
        self.emit_call(jit_type_is_safe as *const ());
        dynasm!(self.ops ; .arch aarch64 ; and x0, x0, 0xff);
        self.store_from_x0(dest, ValueTag::Bool.as_u8());
        Ok(())
    }

    pub(super) fn compile_try_cast(&mut self, dest: u8, value: u8, type_name: &str) -> Result<()> {
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
        dynasm!(self.ops ; .arch aarch64 ; mov x0, x20);
        self.emit_reg_addr(1, value);
        self.emit_mov_imm64(2, type_name_ptr as usize as u64);
        self.emit_mov_imm64(3, type_name_len as u64);
        self.emit_reg_addr(4, dest);
        self.emit_call(jit_try_cast_safe as *const ());
        self.emit_fail_if_w0_zero();
        Ok(())
    }

    pub(super) fn compile_get_enum_value(
        &mut self,
        dest: u8,
        enum_reg: u8,
        index: u8,
    ) -> Result<()> {
        unsafe extern "C" {
            fn jit_get_enum_value_safe(enum_ptr: *const Value, index: usize, out: *mut Value)
            -> u8;
        }
        self.emit_reg_addr(0, enum_reg);
        self.emit_mov_imm64(1, index as u64);
        self.emit_reg_addr(2, dest);
        self.emit_call(jit_get_enum_value_safe as *const ());
        self.emit_fail_if_w0_zero();
        Ok(())
    }
}

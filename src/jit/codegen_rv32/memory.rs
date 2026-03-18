// Memory, constant-loading, and C-helper-based operations for the RV32 JIT.
//
// RISC-V 32 calling convention: a0–a7 for integer arguments (8 registers),
// all JIT helper functions have ≤8 arguments so no stack-spilling is needed.
// Functions returning i64 use (a0=lo, a1=hi); we only use a0 for sizes/counts.
// s2, s3, s4 are callee-saved by the called functions (per ABI), so they are
// preserved across calls.

use super::*;

impl JitCompiler {
    pub(super) fn compile_load_const(&mut self, dest: u8, value: &Value) -> Result<()> {
        match value {
            Value::Int(i) => {
                self.emit_addr_in_t2(dest, 0);
                let i_bits = *i as i32;
                dynasm!(self.ops
                    ; .arch riscv32i
                    ; li t0, 2          // ValueTag::Int
                    ; sb t0, [t2, 0]
                    ; li t0, i_bits
                    ; sw t0, [t2, VALUE_DATA_OFFSET]
                );
                Ok(())
            }
            Value::Float(f) => {
                self.emit_addr_in_t2(dest, 0);
                let f_bits = f.to_bits() as i32;
                dynasm!(self.ops
                    ; .arch riscv32i ; .feature f
                    ; li t0, 3          // ValueTag::Float
                    ; sb t0, [t2, 0]
                    ; li t0, f_bits
                    ; fmv.w.x ft0, t0
                    ; fsw ft0, [t2, VALUE_DATA_OFFSET]
                );
                Ok(())
            }
            Value::Bool(b) => {
                self.emit_addr_in_t2(dest, 0);
                let bool_val = if *b { 1i32 } else { 0i32 };
                dynasm!(self.ops
                    ; .arch riscv32i
                    ; li t0, 1          // ValueTag::Bool
                    ; sb t0, [t2, 0]
                    ; li t0, bool_val
                    ; sw t0, [t2, VALUE_DATA_OFFSET]
                );
                Ok(())
            }
            _ => self.copy_leaked_constant(dest, value),
        }
    }

    fn copy_leaked_constant(&mut self, dest: u8, value: &Value) -> Result<()> {
        extern "C" {
            fn jit_move_safe(src_ptr: *const Value, dest_ptr: *mut Value) -> u8;
        }
        let leaked = Box::leak(Box::new(value.clone()));
        let src_ptr = leaked as *const Value;
        self.leaked_constants.push(src_ptr);
        let src_bits = src_ptr as u32 as i32;

        self.emit_addr_in_t2(dest, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a1, t2);
        dynasm!(self.ops ; .arch riscv32i ; li a0, src_bits);
        self.emit_load_fn_ptr(jit_move_safe as *const ());
        self.emit_call_t0();
        dynasm!(self.ops ; .arch riscv32i ; beqz a0, >fail);
        Ok(())
    }

    pub(super) fn compile_move(&mut self, dest: u8, src: u8) -> Result<()> {
        extern "C" {
            fn jit_move_safe(src_ptr: *const Value, dest_ptr: *mut Value) -> u8;
        }
        let value_size = mem::size_of::<Value>() as i32;
        let num_words = (value_size + 3) / 4;

        // Fast path: scalar types (Nil/Bool/Int/Float) → memcpy word by word.
        // Tags 0-3 are scalar; for anything else call the helper.
        self.load_tag_to_t0(src);
        dynasm!(self.ops
            ; .arch riscv32i
            ; li t1, 4
            ; blt t0, t1, >fast_copy
            ; j >call_helper
            ; fast_copy:
        );
        for i in 0..num_words {
            let off = i * 4;
            self.emit_addr_in_t2(src, off);
            dynasm!(self.ops ; .arch riscv32i ; lw t0, [t2, 0]);
            self.emit_addr_in_t2(dest, off);
            dynasm!(self.ops ; .arch riscv32i ; sw t0, [t2, 0]);
        }
        dynasm!(self.ops ; .arch riscv32i ; j >move_done ; call_helper:);
        self.emit_addr_in_t2(src, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a0, t2);
        self.emit_addr_in_t2(dest, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a1, t2);
        self.emit_load_fn_ptr(jit_move_safe as *const ());
        self.emit_call_t0();
        dynasm!(self.ops ; .arch riscv32i ; move_done:);
        Ok(())
    }

    pub(super) fn compile_get_index(
        &mut self,
        dest: u8,
        array: u8,
        index: u8,
    ) -> Result<()> {
        extern "C" {
            fn jit_array_get_safe(
                array_value: *const Value,
                index: i32,
                out: *mut Value,
            ) -> u8;
        }
        // Guard: array must be Array (tag=5), index must be Int (tag=2)
        self.load_tag_to_t0(array);
        dynasm!(self.ops ; .arch riscv32i ; li t1, 5 ; bne t0, t1, >fail);
        self.load_tag_to_t0(index);
        dynasm!(self.ops ; .arch riscv32i ; li t1, 2 ; bne t0, t1, >fail);
        // a0 = &array, a1 = index_i32, a2 = &dest
        self.emit_addr_in_t2(array, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a0, t2);
        self.load_data_to_t0(index);
        dynasm!(self.ops ; .arch riscv32i ; mv a1, t0);
        self.emit_addr_in_t2(dest, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a2, t2);
        self.emit_load_fn_ptr(jit_array_get_safe as *const ());
        self.emit_call_t0();
        dynasm!(self.ops ; .arch riscv32i ; beqz a0, >fail);
        Ok(())
    }

    pub(super) fn compile_array_len(&mut self, dest: u8, array: u8) -> Result<()> {
        extern "C" {
            // Returns i64 on std (a0=lo, a1=hi on rv32); we use a0 only.
            fn jit_array_len_safe(array_value: *const Value) -> i64;
        }
        self.load_tag_to_t0(array);
        dynasm!(self.ops ; .arch riscv32i ; li t1, 5 ; bne t0, t1, >fail);
        self.emit_addr_in_t2(array, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a0, t2);
        self.emit_load_fn_ptr(jit_array_len_safe as *const ());
        self.emit_call_t0();
        // a0 = length (low 32 bits); a0 < 0 means error
        dynasm!(self.ops ; .arch riscv32i ; bltz a0, >fail);
        // Store result as Int
        self.emit_addr_in_t2(dest, 0);
        dynasm!(self.ops
            ; .arch riscv32i
            ; li t0, 2
            ; sb t0, [t2, 0]
            ; sw a0, [t2, VALUE_DATA_OFFSET]
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
            self.emit_addr_in_t2(object, 0);
            dynasm!(self.ops ; .arch riscv32i ; mv a0, t2);
            dynasm!(self.ops ; .arch riscv32i ; li a1, index as i32);
            self.emit_addr_in_t2(dest, 0);
            dynasm!(self.ops ; .arch riscv32i ; mv a2, t2);
            self.emit_load_fn_ptr(jit_get_field_indexed_safe as *const ());
            self.emit_call_t0();
            dynasm!(self.ops ; .arch riscv32i ; beqz a0, >fail);
        } else {
            let name_box = Box::leak(Box::new(field_name.to_string()));
            let name_ptr = name_box.as_ptr() as u32 as i32;
            let name_len = name_box.len() as i32;
            self.emit_addr_in_t2(object, 0);
            dynasm!(self.ops ; .arch riscv32i ; mv a0, t2);
            dynasm!(self.ops ; .arch riscv32i ; li a1, name_ptr);
            dynasm!(self.ops ; .arch riscv32i ; li a2, name_len);
            self.emit_addr_in_t2(dest, 0);
            dynasm!(self.ops ; .arch riscv32i ; mv a3, t2);
            self.emit_load_fn_ptr(jit_get_field_safe as *const ());
            self.emit_call_t0();
            dynasm!(self.ops ; .arch riscv32i ; beqz a0, >fail);
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
        is_weak: bool,
    ) -> Result<()> {
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
            fn jit_set_field_strong_safe(
                object_ptr: *const Value,
                field_index: usize,
                value_ptr: *const Value,
            ) -> u8;
        }
        if let Some(index) = field_index {
            self.emit_addr_in_t2(object, 0);
            dynasm!(self.ops ; .arch riscv32i ; mv a0, t2);
            dynasm!(self.ops ; .arch riscv32i ; li a1, index as i32);
            self.emit_addr_in_t2(value, 0);
            dynasm!(self.ops ; .arch riscv32i ; mv a2, t2);
            let fn_ptr = if is_weak {
                jit_set_field_indexed_safe as *const ()
            } else {
                jit_set_field_strong_safe as *const ()
            };
            self.emit_load_fn_ptr(fn_ptr);
            self.emit_call_t0();
            dynasm!(self.ops ; .arch riscv32i ; beqz a0, >fail);
        } else {
            let name_box = Box::leak(Box::new(field_name.to_string()));
            let name_ptr = name_box.as_ptr() as u32 as i32;
            let name_len = name_box.len() as i32;
            self.emit_addr_in_t2(object, 0);
            dynasm!(self.ops ; .arch riscv32i ; mv a0, t2);
            dynasm!(self.ops ; .arch riscv32i ; li a1, name_ptr);
            dynasm!(self.ops ; .arch riscv32i ; li a2, name_len);
            self.emit_addr_in_t2(value, 0);
            dynasm!(self.ops ; .arch riscv32i ; mv a3, t2);
            self.emit_load_fn_ptr(jit_set_field_safe as *const ());
            self.emit_call_t0();
            dynasm!(self.ops ; .arch riscv32i ; beqz a0, >fail);
        }
        Ok(())
    }

    pub(super) fn compile_new_array(
        &mut self,
        dest: u8,
        first_element: u8,
        count: u8,
    ) -> Result<()> {
        extern "C" {
            fn jit_new_array_safe(
                vm_ptr: *mut crate::VM,
                elements_ptr: *const Value,
                element_count: usize,
                out: *mut Value,
            ) -> u8;
        }
        dynasm!(self.ops ; .arch riscv32i ; mv a0, s3);
        self.emit_addr_in_t2(first_element, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a1, t2);
        dynasm!(self.ops ; .arch riscv32i ; li a2, count as i32);
        self.emit_addr_in_t2(dest, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a3, t2);
        self.emit_load_fn_ptr(jit_new_array_safe as *const ());
        self.emit_call_t0();
        dynasm!(self.ops ; .arch riscv32i ; beqz a0, >fail);
        Ok(())
    }

    pub(super) fn compile_array_push(&mut self, array: u8, value: u8) -> Result<()> {
        extern "C" {
            fn jit_array_push_safe(
                vm_ptr: *mut crate::VM,
                array_ptr: *const Value,
                value_ptr: *const Value,
            ) -> u8;
        }
        dynasm!(self.ops ; .arch riscv32i ; mv a0, s3);
        self.emit_addr_in_t2(array, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a1, t2);
        self.emit_addr_in_t2(value, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a2, t2);
        self.emit_load_fn_ptr(jit_array_push_safe as *const ());
        self.emit_call_t0();
        dynasm!(self.ops ; .arch riscv32i ; beqz a0, >fail);
        Ok(())
    }

    pub(super) fn compile_enum_is_some(&mut self, dest: u8, enum_reg: u8) -> Result<()> {
        extern "C" {
            fn jit_enum_is_some_safe(enum_ptr: *const Value, out_ptr: *mut Value) -> u8;
        }
        self.emit_addr_in_t2(enum_reg, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a0, t2);
        self.emit_addr_in_t2(dest, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a1, t2);
        self.emit_load_fn_ptr(jit_enum_is_some_safe as *const ());
        self.emit_call_t0();
        dynasm!(self.ops ; .arch riscv32i ; beqz a0, >fail);
        Ok(())
    }

    pub(super) fn compile_enum_unwrap(&mut self, dest: u8, enum_reg: u8) -> Result<()> {
        extern "C" {
            fn jit_enum_unwrap_safe(enum_ptr: *const Value, out_ptr: *mut Value) -> u8;
        }
        self.emit_addr_in_t2(enum_reg, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a0, t2);
        self.emit_addr_in_t2(dest, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a1, t2);
        self.emit_load_fn_ptr(jit_enum_unwrap_safe as *const ());
        self.emit_call_t0();
        dynasm!(self.ops ; .arch riscv32i ; beqz a0, >fail);
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
        let exit_label = self.current_exit_label();
        let expected_bits = expected_ptr as u32 as i32;

        dynasm!(self.ops ; .arch riscv32i ; mv a0, s3);
        self.emit_addr_in_t2(callee, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a1, t2);
        dynasm!(self.ops ; .arch riscv32i ; li a2, expected_bits);
        self.emit_addr_in_t2(first_arg, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a3, t2);
        dynasm!(self.ops ; .arch riscv32i ; li a4, arg_count as i32);
        self.emit_addr_in_t2(dest, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a5, t2);
        self.emit_load_fn_ptr(jit_call_native_safe as *const ());
        self.emit_call_t0();
        // a0 = 2 or 3 → yield/stop (exit trace); 0 → error; 1 → ok
        dynasm!(self.ops
            ; .arch riscv32i
            ; li t0, 2
            ; beq a0, t0, >native_yield
            ; li t0, 3
            ; beq a0, t0, >native_yield
            ; beqz a0, >fail
            ; j >native_ok
            ; native_yield:
            ; j => exit_label
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
        _is_closure: bool,
        _upvalues_ptr: Option<*const ()>,
    ) -> Result<()> {
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
        dynasm!(self.ops ; .arch riscv32i ; mv a0, s3);
        self.emit_addr_in_t2(callee, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a1, t2);
        self.emit_addr_in_t2(first_arg, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a2, t2);
        dynasm!(self.ops ; .arch riscv32i ; li a3, arg_count as i32);
        dynasm!(self.ops ; .arch riscv32i ; li a4, dest as i32);
        self.emit_load_fn_ptr(jit_call_function_safe as *const ());
        self.emit_call_t0();
        dynasm!(self.ops ; .arch riscv32i ; beqz a0, >fail);

        if self.inline_depth == 0 {
            // Re-sync s2 (register base) after a call that may allocate a new frame.
            dynasm!(self.ops ; .arch riscv32i ; mv a0, s3);
            self.emit_load_fn_ptr(jit_current_registers as *const ());
            self.emit_call_t0();
            dynasm!(self.ops ; .arch riscv32i ; beqz a0, >fail ; mv s2, a0);
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
        let name_box = Box::leak(Box::new(method_name.to_string()));
        let name_ptr = name_box.as_ptr() as u32 as i32;
        let name_len = name_box.len() as i32;

        dynasm!(self.ops ; .arch riscv32i ; mv a0, s3);
        self.emit_addr_in_t2(object, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a1, t2);
        dynasm!(self.ops ; .arch riscv32i ; li a2, name_ptr);
        dynasm!(self.ops ; .arch riscv32i ; li a3, name_len);
        self.emit_addr_in_t2(first_arg, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a4, t2);
        dynasm!(self.ops ; .arch riscv32i ; li a5, arg_count as i32);
        dynasm!(self.ops ; .arch riscv32i ; li a6, dest as i32);
        self.emit_load_fn_ptr(jit_call_method_safe as *const ());
        self.emit_call_t0();
        dynasm!(self.ops ; .arch riscv32i ; beqz a0, >fail);

        if self.inline_depth == 0 {
            dynasm!(self.ops ; .arch riscv32i ; mv a0, s3);
            self.emit_load_fn_ptr(jit_current_registers as *const ());
            self.emit_call_t0();
            dynasm!(self.ops ; .arch riscv32i ; beqz a0, >fail ; mv s2, a0);
        }
        Ok(())
    }

    pub(super) fn compile_new_struct(
        &mut self,
        dest: u8,
        struct_name: &str,
        field_names: &[alloc::string::String],
        field_registers: &[u8],
    ) -> Result<()> {
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
        let sname_box = Box::leak(Box::new(struct_name.to_string()));
        let sname_ptr = sname_box.as_ptr() as u32 as i32;
        let sname_len = sname_box.len() as i32;
        let field_count = field_names.len() as i32;

        let mut fptrs: Vec<*const u8> = Vec::new();
        let mut flens: Vec<usize> = Vec::new();
        for n in field_names {
            let b = Box::leak(Box::new(n.clone()));
            fptrs.push(b.as_ptr());
            flens.push(b.len());
        }
        let fptrs_box = Box::leak(fptrs.into_boxed_slice());
        let flens_box = Box::leak(flens.into_boxed_slice());
        let fptrs_bits = fptrs_box.as_ptr() as u32 as i32;
        let flens_bits = flens_box.as_ptr() as u32 as i32;

        // All 8 args fit in a0–a7.
        dynasm!(self.ops ; .arch riscv32i ; mv a0, s3);
        dynasm!(self.ops ; .arch riscv32i ; li a1, sname_ptr);
        dynasm!(self.ops ; .arch riscv32i ; li a2, sname_len);
        dynasm!(self.ops ; .arch riscv32i ; li a3, fptrs_bits);
        dynasm!(self.ops ; .arch riscv32i ; li a4, flens_bits);
        if let Some(&first) = field_registers.first() {
            self.emit_addr_in_t2(first, 0);
            dynasm!(self.ops ; .arch riscv32i ; mv a5, t2);
        } else {
            dynasm!(self.ops ; .arch riscv32i ; li a5, 0);
        }
        dynasm!(self.ops ; .arch riscv32i ; li a6, field_count);
        self.emit_addr_in_t2(dest, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a7, t2);
        self.emit_load_fn_ptr(jit_new_struct_safe as *const ());
        self.emit_call_t0();
        dynasm!(self.ops ; .arch riscv32i ; beqz a0, >fail);

        if self.inline_depth == 0 {
            dynasm!(self.ops ; .arch riscv32i ; mv a0, s3);
            self.emit_load_fn_ptr(jit_current_registers as *const ());
            self.emit_call_t0();
            dynasm!(self.ops ; .arch riscv32i ; beqz a0, >fail ; mv s2, a0);
        }
        Ok(())
    }

    pub(super) fn compile_new_enum_unit(
        &mut self,
        dest: u8,
        enum_name: &str,
        variant_name: &str,
    ) -> Result<()> {
        extern "C" {
            fn jit_new_enum_unit_safe(
                vm_ptr: *mut crate::VM,
                enum_name_ptr: *const u8,
                enum_name_len: usize,
                variant_name_ptr: *const u8,
                variant_name_len: usize,
                out: *mut Value,
            ) -> u8;
        }
        let en_box = Box::leak(Box::new(enum_name.to_string()));
        let en_ptr = en_box.as_ptr() as u32 as i32;
        let en_len = en_box.len() as i32;
        let vn_box = Box::leak(Box::new(variant_name.to_string()));
        let vn_ptr = vn_box.as_ptr() as u32 as i32;
        let vn_len = vn_box.len() as i32;

        dynasm!(self.ops ; .arch riscv32i ; mv a0, s3);
        dynasm!(self.ops ; .arch riscv32i ; li a1, en_ptr);
        dynasm!(self.ops ; .arch riscv32i ; li a2, en_len);
        dynasm!(self.ops ; .arch riscv32i ; li a3, vn_ptr);
        dynasm!(self.ops ; .arch riscv32i ; li a4, vn_len);
        self.emit_addr_in_t2(dest, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a5, t2);
        self.emit_load_fn_ptr(jit_new_enum_unit_safe as *const ());
        self.emit_call_t0();
        dynasm!(self.ops ; .arch riscv32i ; beqz a0, >fail);
        Ok(())
    }

    pub(super) fn compile_new_enum_variant(
        &mut self,
        dest: u8,
        enum_name: &str,
        variant_name: &str,
        value_registers: &[u8],
    ) -> Result<()> {
        extern "C" {
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
        let en_box = Box::leak(Box::new(enum_name.to_string()));
        let en_ptr = en_box.as_ptr() as u32 as i32;
        let en_len = en_box.len() as i32;
        let vn_box = Box::leak(Box::new(variant_name.to_string()));
        let vn_ptr = vn_box.as_ptr() as u32 as i32;
        let vn_len = vn_box.len() as i32;
        let val_count = value_registers.len() as i32;

        dynasm!(self.ops ; .arch riscv32i ; mv a0, s3);
        dynasm!(self.ops ; .arch riscv32i ; li a1, en_ptr);
        dynasm!(self.ops ; .arch riscv32i ; li a2, en_len);
        dynasm!(self.ops ; .arch riscv32i ; li a3, vn_ptr);
        dynasm!(self.ops ; .arch riscv32i ; li a4, vn_len);
        if let Some(&first) = value_registers.first() {
            self.emit_addr_in_t2(first, 0);
            dynasm!(self.ops ; .arch riscv32i ; mv a5, t2);
        } else {
            dynasm!(self.ops ; .arch riscv32i ; li a5, 0);
        }
        dynasm!(self.ops ; .arch riscv32i ; li a6, val_count);
        self.emit_addr_in_t2(dest, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a7, t2);
        self.emit_load_fn_ptr(jit_new_enum_variant_safe as *const ());
        self.emit_call_t0();
        dynasm!(self.ops ; .arch riscv32i ; beqz a0, >fail);
        Ok(())
    }

    pub(super) fn compile_is_enum_variant(
        &mut self,
        dest: u8,
        value: u8,
        enum_name: &str,
        variant_name: &str,
    ) -> Result<()> {
        extern "C" {
            fn jit_is_enum_variant_safe(
                value_ptr: *const Value,
                enum_name_ptr: *const u8,
                enum_name_len: usize,
                variant_name_ptr: *const u8,
                variant_name_len: usize,
            ) -> u8;
        }
        let en_box = Box::leak(Box::new(enum_name.to_string()));
        let en_ptr = en_box.as_ptr() as u32 as i32;
        let en_len = en_box.len() as i32;
        let vn_box = Box::leak(Box::new(variant_name.to_string()));
        let vn_ptr = vn_box.as_ptr() as u32 as i32;
        let vn_len = vn_box.len() as i32;

        self.emit_addr_in_t2(value, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a0, t2);
        dynasm!(self.ops ; .arch riscv32i ; li a1, en_ptr);
        dynasm!(self.ops ; .arch riscv32i ; li a2, en_len);
        dynasm!(self.ops ; .arch riscv32i ; li a3, vn_ptr);
        dynasm!(self.ops ; .arch riscv32i ; li a4, vn_len);
        self.emit_load_fn_ptr(jit_is_enum_variant_safe as *const ());
        self.emit_call_t0();
        // a0 = 1 if match, 0 if not
        self.emit_addr_in_t2(dest, 0);
        dynasm!(self.ops
            ; .arch riscv32i
            ; li t0, 1
            ; sb t0, [t2, 0]            // tag = Bool
            ; sw a0, [t2, VALUE_DATA_OFFSET]
        );
        Ok(())
    }

    pub(super) fn compile_get_enum_value(
        &mut self,
        dest: u8,
        enum_reg: u8,
        index: u8,
    ) -> Result<()> {
        extern "C" {
            fn jit_get_enum_value_safe(
                enum_ptr: *const Value,
                index: usize,
                out: *mut Value,
            ) -> u8;
        }
        self.emit_addr_in_t2(enum_reg, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a0, t2);
        dynasm!(self.ops ; .arch riscv32i ; li a1, index as i32);
        self.emit_addr_in_t2(dest, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a2, t2);
        self.emit_load_fn_ptr(jit_get_enum_value_safe as *const ());
        self.emit_call_t0();
        dynasm!(self.ops ; .arch riscv32i ; beqz a0, >fail);
        Ok(())
    }
}

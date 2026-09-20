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

            // Plain values that own nothing: tag + payload, with the old
            // value dropped only if it owned something.
            Value::Nil => self.store_plain_constant(dest, 0, value),
            Value::Function(idx) => self.store_plain_constant(dest, *idx as u64, value),

            _ => self.copy_owned_constant(dest, value),
        }
    }

    fn store_plain_constant(&mut self, dest: u8, payload: u64, value: &Value) -> Result<()> {
        let scalar_max_tag = ValueTag::Float.as_u8() as u32;
        // The discriminant byte of `#[repr(C, u8)] Value` (`ValueTag` is a
        // coarser classification and numbers `Function` differently).
        // SAFETY: reading the first byte of a live `Value`.
        let tag = unsafe { *(value as *const Value as *const u8) } as u32;
        // A destination known to hold nothing owned is simply overwritten.
        if self.scalar_registers.contains_key(&dest) {
            self.emit_mov_imm64(0, payload);
            self.store_tag_imm(dest, tag as u8);
            self.store_payload(dest, 0);
            return Ok(());
        }
        let done = self.ops.new_dynamic_label();
        self.load_tag_from_memory(9, dest);
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp w9, #scalar_max_tag
            ; b.ls >plain
        );
        // A function index held from the previous iteration of a loop that
        // reloads it is as plain as a scalar.
        if matches!(value, Value::Function(_)) {
            dynasm!(self.ops ; .arch aarch64 ; cmp w9, #tag ; b.ne >owned);
        } else {
            dynasm!(self.ops ; .arch aarch64 ; b >owned);
        }
        dynasm!(self.ops ; .arch aarch64 ; plain:);
        self.emit_mov_imm64(0, payload);
        self.store_tag_imm(dest, tag as u8);
        self.store_payload(dest, 0);
        dynasm!(self.ops
            ; .arch aarch64
            ; b => done
            ; owned:
        );
        self.copy_owned_constant(dest, value)?;
        dynasm!(self.ops ; .arch aarch64 ; => done);
        Ok(())
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
        // A pinned destination is only ever written with a scalar of its
        // own type (see pins::plan), so the move is a register copy; the
        // helper below would update memory and leave the pin stale.
        if let Some(pin) = self.pin_for_write(dest) {
            match pin.ty {
                ValueType::Float => {
                    self.load_payload_f(0, src);
                    self.store_d0_as_float(dest);
                }
                ValueType::Int => {
                    self.load_payload(0, src);
                    self.store_from_x0(dest, ValueTag::Int.as_u8());
                }
                _ => {
                    self.load_payload(0, src);
                    self.store_from_x0(dest, ValueTag::Bool.as_u8());
                }
            }
            return Ok(());
        }
        // A source of known scalar type is a payload copy plus a typed
        // store (which drops whatever the destination held).
        match self.scalar_registers.get(&src).copied() {
            Some(ValueType::Float) => {
                self.load_payload_f(0, src);
                self.store_d0_as_float(dest);
                return Ok(());
            }
            Some(ValueType::Int) => {
                self.load_payload(0, src);
                self.store_from_x0(dest, ValueTag::Int.as_u8());
                return Ok(());
            }
            Some(ValueType::Bool) => {
                self.load_bool_payload(0, src);
                self.store_from_x0(dest, ValueTag::Bool.as_u8());
                return Ok(());
            }
            _ => {}
        }
        self.emit_reg_addr(11, src);
        self.emit_clone_x11_into(dest);
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
        value_type: Option<ValueType>,
    ) -> Result<()> {
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
            let array_tag = ValueTag::Array.as_u8() as u32;
            let scalar_max_tag = ValueTag::Float.as_u8() as u32;
            let rc_offset = layout.array_rc_offset as u32;
            let len_offset = layout.len_offset as u32;
            let ptr_offset = layout.ptr_offset as u32;
            let value_size = mem::size_of::<Value>() as u32;
            let done = self.ops.new_dynamic_label();
            let out_of_range = self.ops.new_dynamic_label();
            let slow = self.ops.new_dynamic_label();
            self.load_tag(0, array);
            dynasm!(self.ops
                ; .arch aarch64
                ; cmp w0, #array_tag
                ; b.ne => slow
            );
            self.load_payload(12, index);
            self.emit_reg_addr(11, array);
            dynasm!(self.ops
                ; .arch aarch64
                ; ldr x9, [x11, #rc_offset]
                ; ldr x10, [x9, #len_offset]
                ; cmp x12, x10
                ; b.hs => out_of_range
                ; ldr x10, [x9, #ptr_offset]
                ; movz w13, #value_size
                ; madd x10, x12, x13, x10
                ; ldrb w9, [x10]
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
                .as_u8() as u32;
                dynasm!(self.ops
                    ; .arch aarch64
                    ; cmp w9, #expected_tag
                    ; b.ne >fail
                );
                match ty {
                    ValueType::Float => {
                        dynasm!(self.ops ; .arch aarch64 ; ldr d0, [x10, 8]);
                        self.store_d0_as_float(value_dest);
                    }
                    ValueType::Int => {
                        dynasm!(self.ops ; .arch aarch64 ; ldr x0, [x10, 8]);
                        self.store_from_x0(value_dest, ValueTag::Int.as_u8());
                    }
                    _ => {
                        dynasm!(self.ops ; .arch aarch64 ; ldrb w0, [x10, 8]);
                        self.store_from_x0(value_dest, ValueTag::Bool.as_u8());
                    }
                }
                self.pending_scalar = Some((value_dest, ty));
            } else {
                dynasm!(self.ops
                    ; .arch aarch64
                    ; cmp w9, #scalar_max_tag
                    ; b.hi => slow
                );
                // The destination must hold nothing owned for a plain copy.
                self.load_tag_from_memory(13, value_dest);
                dynasm!(self.ops
                    ; .arch aarch64
                    ; cmp w13, #scalar_max_tag
                    ; b.hi => slow
                    ; ldp x0, x1, [x10]
                );
                self.emit_reg_addr(11, value_dest);
                dynasm!(self.ops
                    ; .arch aarch64
                    ; stp x0, x1, [x11]
                );
            }
            dynasm!(self.ops ; .arch aarch64 ; movz x0, 1);
            self.store_from_x0(condition_dest, ValueTag::Bool.as_u8());
            dynasm!(self.ops
                ; .arch aarch64
                ; b => done
                ; => out_of_range
            );
            self.compile_load_const(value_dest, &Value::Nil)?;
            dynasm!(self.ops ; .arch aarch64 ; mov x0, xzr);
            self.store_from_x0(condition_dest, ValueTag::Bool.as_u8());
            dynasm!(self.ops
                ; .arch aarch64
                ; b => done
                ; => slow
            );
            self.emit_reg_addr(0, array);
            self.emit_reg_addr(1, index);
            self.emit_reg_addr(2, value_dest);
            self.emit_reg_addr(3, condition_dest);
            self.emit_call(jit_array_index_ok_safe as *const ());
            self.emit_fail_if_w0_zero();
            dynasm!(self.ops ; .arch aarch64 ; => done);
            return Ok(());
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
        if let Some(layout) = jit::layout::rc_vec_layout() {
            let rc_offset = layout.array_rc_offset as u32;
            let len_offset = layout.len_offset as u32;
            self.emit_reg_addr(11, array);
            dynasm!(self.ops
                ; .arch aarch64
                ; ldr x9, [x11, #rc_offset]
                ; ldr x0, [x9, #len_offset]
            );
            self.store_from_x0(dest, ValueTag::Int.as_u8());
            return Ok(());
        }
        self.emit_reg_addr(0, array);
        self.emit_call(jit_array_len_safe as *const ());
        // `tbnz` only reaches ±32 KB; traces can be larger, so branch on
        // the flags instead (`b.cond` reaches ±1 MB like every other exit).
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp x0, 0
            ; b.lt >fail
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
            .as_u8() as u32;
            let struct_tag = ValueTag::Struct.as_u8() as u32;
            let fields_offset = layout.struct_fields_offset as u32;
            let len_offset = layout.len_offset as u32;
            let ptr_offset = layout.ptr_offset as u32;
            let element = (index * mem::size_of::<Value>()) as i32;
            self.load_tag(0, object);
            dynasm!(self.ops
                ; .arch aarch64
                ; cmp w0, #struct_tag
                ; b.ne >fail
            );
            self.emit_reg_addr(11, object);
            dynasm!(self.ops
                ; .arch aarch64
                ; ldr x9, [x11, #fields_offset]
                ; ldr x10, [x9, #len_offset]
                ; cmp x10, #index as u32
                ; b.ls >fail
                ; ldr x10, [x9, #ptr_offset]
            );
            self.emit_add_imm(10, 10, element);
            dynasm!(self.ops
                ; .arch aarch64
                ; ldrb w9, [x10]
                ; cmp w9, #expected_tag
                ; b.ne >fail
            );
            match ty {
                ValueType::Float => {
                    dynasm!(self.ops ; .arch aarch64 ; ldr d0, [x10, 8]);
                    self.store_d0_as_float(dest);
                }
                ValueType::Int => {
                    dynasm!(self.ops ; .arch aarch64 ; ldr x0, [x10, 8]);
                    self.store_from_x0(dest, ValueTag::Int.as_u8());
                }
                _ => {
                    dynasm!(self.ops ; .arch aarch64 ; ldrb w0, [x10, 8]);
                    self.store_from_x0(dest, ValueTag::Bool.as_u8());
                }
            }
            self.pending_scalar = Some((dest, ty));
            return Ok(());
        }

        if let (Some(index), Some(layout), false) = (field_index, jit::layout::rc_vec_layout(), _is_weak) {
            // Any strong field read inline: struct tag and field count are
            // checked (anything else fails to the interpreter), then the
            // element is cloned into the register without the runtime.
            let struct_tag = ValueTag::Struct.as_u8() as u32;
            let fields_offset = layout.struct_fields_offset as u32;
            let len_offset = layout.len_offset as u32;
            let ptr_offset = layout.ptr_offset as u32;
            let element = (index * mem::size_of::<Value>()) as i32;
            self.load_tag(0, object);
            dynasm!(self.ops
                ; .arch aarch64
                ; cmp w0, #struct_tag
                ; b.ne >fail
            );
            self.emit_reg_addr(11, object);
            dynasm!(self.ops
                ; .arch aarch64
                ; ldr x9, [x11, #fields_offset]
                ; ldr x10, [x9, #len_offset]
                ; cmp x10, #index as u32
                ; b.ls >fail
                ; ldr x11, [x9, #ptr_offset]
            );
            self.emit_add_imm(11, 11, element);
            self.emit_clone_x11_into(dest);
            return Ok(());
        }

        if let Some(index) = field_index {
            self.emit_reg_addr(0, object);
            self.emit_mov_imm64(1, index as u64);
            self.emit_reg_addr(2, dest);
            self.emit_call(jit_get_field_indexed_safe as *const ());
        } else {
            let (field_name_ptr, field_name_len) = self.retain_string(field_name);
            let key = self.retain_key(field_name);
            self.emit_reg_addr(0, object);
            self.emit_mov_imm64(1, field_name_ptr as usize as u64);
            self.emit_mov_imm64(2, field_name_len as u64);
            self.emit_mov_imm64(3, key as usize as u64);
            self.emit_reg_addr(4, dest);
            self.emit_call(jit_get_field_keyed as *const ());
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
            let expected_tag = match ty {
                ValueType::Int => ValueTag::Int,
                ValueType::Float => ValueTag::Float,
                _ => ValueTag::Bool,
            }
            .as_u8() as u32;
            let struct_tag = ValueTag::Struct.as_u8() as u32;
            let fields_offset = layout.struct_fields_offset as u32;
            let len_offset = layout.len_offset as u32;
            let ptr_offset = layout.ptr_offset as u32;
            let borrow_offset = layout.borrow_offset as u32;
            let element = (index * mem::size_of::<Value>()) as i32;
            let done = self.ops.new_dynamic_label();
            self.load_tag(0, object);
            dynasm!(self.ops
                ; .arch aarch64
                ; cmp w0, #struct_tag
                ; b.ne >slow
            );
            self.emit_reg_addr(11, object);
            dynasm!(self.ops
                ; .arch aarch64
                ; ldr x9, [x11, #fields_offset]
                ; ldr x10, [x9, #borrow_offset]
                ; cbnz x10, >slow
                ; ldr x10, [x9, #len_offset]
                ; cmp x10, #index as u32
                ; b.ls >slow
                ; ldr x10, [x9, #ptr_offset]
            );
            self.emit_add_imm(10, 10, element);
            dynasm!(self.ops
                ; .arch aarch64
                ; ldrb w9, [x10]
                ; cmp w9, #expected_tag
                ; b.ne >slow
            );
            match ty {
                ValueType::Float => {
                    self.load_payload_f(0, value);
                    dynasm!(self.ops ; .arch aarch64 ; str d0, [x10, 8]);
                }
                ValueType::Int => {
                    self.load_payload(0, value);
                    dynasm!(self.ops ; .arch aarch64 ; str x0, [x10, 8]);
                }
                _ => {
                    self.load_bool_payload(0, value);
                    dynasm!(self.ops ; .arch aarch64 ; strb w0, [x10, 8]);
                }
            }
            dynasm!(self.ops
                ; .arch aarch64
                ; b => done
                ; slow:
            );
            self.emit_reg_addr(0, object);
            self.emit_mov_imm64(1, index as u64);
            self.emit_reg_addr(2, value);
            self.emit_call(jit_set_field_indexed_safe as *const ());
            self.emit_fail_if_w0_zero();
            dynasm!(self.ops ; .arch aarch64 ; => done);
            return Ok(());
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

        self.emit_call_ip();
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
                out: *mut Value,
            ) -> u8;
        }

        dynasm!(self.ops ; .arch aarch64 ; mov x0, x20);
        self.emit_reg_addr(1, callee);
        self.emit_reg_addr(2, first_arg);
        self.emit_mov_imm32(3, arg_count as u32);
        self.emit_mov_imm32(4, dest as u32);
        self.emit_call_result_out(5, dest);
        self.emit_call_ip();
        self.emit_call(jit_call_function_safe as *const ());
        self.emit_fail_if_w0_zero();
        self.emit_reload_registers_base();
        Ok(())
    }

    /// Where a call helper should deliver its result. Inside an inlined
    /// frame the destination register lives on the JIT stack, so its address
    /// is stable and is passed directly. At depth zero the VM may reallocate
    /// its register file during the call, so the helper writes the result by
    /// index into the current frame instead (and x19 is reloaded after).
    /// `JIT_CALL_IP = ip of the call being compiled`, for the helper about
    /// to run it (unknown inside an inlined body, whose frame the
    /// interpreter does not have). Clobbers x11/x12 only.
    fn emit_call_ip(&mut self) {
        let ip = if self.inline_depth == 0 {
            self.current_fail_ip.unwrap_or(usize::MAX)
        } else {
            usize::MAX
        };
        let offset = jit::CALL_IP_OFFSET as u32;
        self.emit_mov_imm64(12, ip as u64);
        dynasm!(self.ops ; .arch aarch64 ; str x12, [x20, #offset]);
    }

    fn emit_call_result_out(&mut self, x: u8, dest: u8) {
        // Function code's frame is wherever x19 points (the machine stack
        // when called natively), never looked up through the VM.
        if self.inline_depth > 0 || self.function_mode {
            self.emit_reg_addr(x, dest);
        } else {
            dynasm!(self.ops ; .arch aarch64 ; mov X(x), xzr);
        }
    }

    /// The VM may have reallocated its register file during a call; refresh
    /// x19 from the VM unless we are inside an inlined frame (whose registers
    /// live on our own stack).
    fn emit_reload_registers_base(&mut self) {
        unsafe extern "C" {
            fn jit_current_registers(vm_ptr: *mut crate::VM) -> *mut Value;
        }
        if self.inline_depth == 0 && !self.function_mode {
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
                out: *mut Value,
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
        self.emit_call_result_out(7, dest);
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
        if let Some(layout) = jit::layout::enum_layout() {
            // Names are interned, so the variant test is a tag check and
            // pointer compares against the interned constants (the enum
            // name too, unless the test leaves it open).
            let variant_ptr = self.retain_name(variant_name);
            let enum_ptr = (!enum_name.is_empty()).then(|| self.retain_name(enum_name));
            let tag = layout.tag as u32;
            let variant_offset = layout.variant_offset as u32;
            let enum_name_offset = layout.enum_name_offset as u32;
            self.load_tag(0, value);
            self.emit_reg_addr(11, value);
            dynasm!(self.ops
                ; .arch aarch64
                ; cmp w0, #tag
                ; b.ne >no
                ; ldr x9, [x11, #variant_offset]
            );
            self.emit_mov_imm64(10, variant_ptr as u64);
            dynasm!(self.ops
                ; .arch aarch64
                ; cmp x9, x10
                ; b.ne >no
            );
            if let Some(enum_ptr) = enum_ptr {
                dynasm!(self.ops ; .arch aarch64 ; ldr x9, [x11, #enum_name_offset]);
                self.emit_mov_imm64(10, enum_ptr as u64);
                dynasm!(self.ops
                    ; .arch aarch64
                    ; cmp x9, x10
                    ; b.ne >no
                );
            }
            dynasm!(self.ops
                ; .arch aarch64
                ; movz x0, 1
                ; b >done
                ; no:
                ; mov x0, xzr
                ; done:
            );
            self.store_from_x0(dest, ValueTag::Bool.as_u8());
            return Ok(());
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
        if let Some(layout) = jit::layout::enum_layout() {
            // Inline: an enum with a payload of at least `index + 1` values
            // (anything else fails to the interpreter), the value cloned
            // into the register without the runtime.
            let tag = layout.tag as u32;
            let values_offset = layout.values_offset as u32;
            let len_offset = layout.values_len_offset as u32;
            let ptr_offset = layout.values_ptr_offset as u32;
            let element = (index as usize * mem::size_of::<Value>()) as i32;
            self.load_tag(0, enum_reg);
            dynasm!(self.ops
                ; .arch aarch64
                ; cmp w0, #tag
                ; b.ne >fail
            );
            self.emit_reg_addr(11, enum_reg);
            dynasm!(self.ops
                ; .arch aarch64
                ; ldr x9, [x11, #values_offset]
                ; cbz x9, >fail
                ; ldr x10, [x9, #len_offset]
                ; cmp x10, #index as u32
                ; b.ls >fail
                ; ldr x11, [x9, #ptr_offset]
            );
            self.emit_add_imm(11, 11, element);
            self.emit_clone_x11_into(dest);
            return Ok(());
        }
        self.emit_reg_addr(0, enum_reg);
        self.emit_mov_imm64(1, index as u64);
        self.emit_reg_addr(2, dest);
        self.emit_call(jit_get_enum_value_safe as *const ());
        self.emit_fail_if_w0_zero();
        Ok(())
    }
}

use super::*;
impl JitCompiler {
    pub(super) fn compile_guard(
        &mut self,
        register: u8,
        expected_type: ValueType,
        guard_index: usize,
    ) -> Result<Guard> {
        let expected_tag = match expected_type {
            ValueType::Bool => ValueTag::Bool,
            ValueType::Int => ValueTag::Int,
            ValueType::Float => ValueTag::Float,
            ValueType::String => ValueTag::String,
            ValueType::Array => ValueTag::Array,
            ValueType::Tuple => ValueTag::Tuple,
            ValueType::Struct => ValueTag::Struct,
        };
        let expected_discriminant = expected_tag.as_u8() as u32;
        let guard_return_value = (guard_index + 1) as i32;
        self.load_tag(0, register);
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp w0, #expected_discriminant
            ; b.eq >guard_ok
        );
        self.emit_guard_exit(guard_return_value);
        dynasm!(self.ops
            ; .arch aarch64
            ; guard_ok:
        );
        Ok(Guard {
            index: guard_index,
            bailout_ip: 0,
            kind: match expected_type {
                ValueType::Int => GuardKind::IntType { register },
                ValueType::Float => GuardKind::FloatType { register },
                ValueType::Bool => GuardKind::BoolType { register },
                ValueType::String => GuardKind::IntType { register },
                ValueType::Array => GuardKind::IntType { register },
                ValueType::Tuple => GuardKind::IntType { register },
                ValueType::Struct => GuardKind::IntType { register },
            },
            fail_count: 0,
            side_trace: None,
        })
    }

    pub(super) fn compile_guard_function(
        &mut self,
        register: u8,
        function_idx: usize,
        guard_index: usize,
    ) -> Result<Guard> {
        self.compile_guard_function_internal(
            register,
            function_idx,
            core::ptr::null(),
            false,
            guard_index,
        )
    }

    pub(super) fn compile_guard_closure(
        &mut self,
        register: u8,
        function_idx: usize,
        upvalues_ptr: *const (),
        guard_index: usize,
    ) -> Result<Guard> {
        self.compile_guard_function_internal(
            register,
            function_idx,
            upvalues_ptr,
            true,
            guard_index,
        )
    }

    fn compile_guard_function_internal(
        &mut self,
        register: u8,
        function_idx: usize,
        upvalues_ptr: *const (),
        is_closure: bool,
        guard_index: usize,
    ) -> Result<Guard> {
        let guard_return_value = (guard_index + 1) as i32;
        unsafe extern "C" {
            fn jit_guard_function_identity(
                value_ptr: *const Value,
                expected_kind: u8,
                expected_function_idx: usize,
                expected_upvalues: *const (),
                register_index: u8,
            ) -> u8;
        }
        let kind_flag: u32 = if is_closure { 1 } else { 0 };
        self.emit_reg_addr(0, register);
        self.emit_mov_imm32(1, kind_flag);
        self.emit_mov_imm64(2, function_idx as u64);
        self.emit_mov_imm64(3, upvalues_ptr as usize as u64);
        self.emit_mov_imm32(4, register as u32);
        self.emit_call(jit_guard_function_identity as *const ());
        dynasm!(self.ops
            ; .arch aarch64
            ; and w0, w0, 0xff
            ; cbnz w0, >guard_ok
        );
        self.emit_guard_exit(guard_return_value);
        dynasm!(self.ops
            ; .arch aarch64
            ; guard_ok:
        );
        let kind = if is_closure {
            GuardKind::Closure {
                register,
                function_idx,
                upvalues_ptr,
            }
        } else {
            GuardKind::Function {
                register,
                function_idx,
            }
        };
        Ok(Guard {
            index: guard_index,
            bailout_ip: 0,
            kind,
            fail_count: 0,
            side_trace: None,
        })
    }

    pub(super) fn compile_guard_native_function(
        &mut self,
        register: u8,
        expected_ptr: *const (),
        guard_index: usize,
    ) -> Result<Guard> {
        let guard_return_value = (guard_index + 1) as i32;
        unsafe extern "C" {
            fn jit_guard_native_function(
                value_ptr: *const Value,
                expected: *const (),
                register_index: u8,
            ) -> u8;
        }
        self.emit_reg_addr(0, register);
        self.emit_mov_imm64(1, expected_ptr as usize as u64);
        self.emit_mov_imm32(2, register as u32);
        self.emit_call(jit_guard_native_function as *const ());
        dynasm!(self.ops
            ; .arch aarch64
            ; and w0, w0, 0xff
            ; cbnz w0, >guard_ok
        );
        self.emit_guard_exit(guard_return_value);
        dynasm!(self.ops
            ; .arch aarch64
            ; guard_ok:
        );
        Ok(Guard {
            index: guard_index,
            bailout_ip: 0,
            kind: GuardKind::NativeFunction {
                register,
                expected: expected_ptr,
            },
            fail_count: 0,
            side_trace: None,
        })
    }

    pub(super) fn compile_truth_guard(
        &mut self,
        condition_register: u8,
        expect_truthy: bool,
        bailout_ip: usize,
        guard_index: usize,
    ) -> Result<Guard> {
        let guard_return_value = (guard_index + 1) as i32;
        let bool_tag = ValueTag::Bool.as_u8() as u32;
        let scalar_max_tag = ValueTag::Float.as_u8() as u32;
        unsafe extern "C" {
            fn jit_value_is_truthy(value_ptr: *const Value) -> u8;
        }
        if self.scalar_registers.get(&condition_register) == Some(&ValueType::Bool) {
            self.load_bool_payload(0, condition_register);
        } else {
            self.load_tag(0, condition_register);
            dynasm!(self.ops
                ; .arch aarch64
                ; cmp w0, #scalar_max_tag
                ; b.hi >generic_truthiness
                ; cmp w0, #bool_tag
                ; b.eq >load_bool
                // Nil is false; numeric scalars are true regardless of payload.
                ; cmp w0, 0
                ; cset w0, ne
                ; b >truthiness_ready
                ; load_bool:
            );
            self.load_bool_payload(0, condition_register);
            dynasm!(self.ops
                ; .arch aarch64
                ; b >truthiness_ready
                ; generic_truthiness:
            );
            self.emit_reg_addr(0, condition_register);
            self.emit_call(jit_value_is_truthy as *const ());
            dynasm!(self.ops
                ; .arch aarch64
                ; and w0, w0, 0xff
                ; truthiness_ready:
            );
        }
        if expect_truthy {
            dynasm!(self.ops ; .arch aarch64 ; cbnz w0, >guard_ok);
        } else {
            dynasm!(self.ops ; .arch aarch64 ; cbz w0, >guard_ok);
        }
        self.emit_guard_exit(guard_return_value);
        dynasm!(self.ops
            ; .arch aarch64
            ; guard_ok:
        );
        let kind = if expect_truthy {
            GuardKind::Truthy {
                register: condition_register,
            }
        } else {
            GuardKind::Falsy {
                register: condition_register,
            }
        };
        Ok(Guard {
            index: guard_index,
            bailout_ip,
            kind,
            fail_count: 0,
            side_trace: None,
        })
    }
}

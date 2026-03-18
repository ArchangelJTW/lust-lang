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
        let expected_disc = expected_tag.as_u8() as i32;
        let guard_return_value = (guard_index + 1) as i32;
        let exit_label = self.current_exit_label();

        self.load_tag_to_t0(register);
        dynasm!(self.ops
            ; .arch riscv32i
            ; li t1, expected_disc
            ; beq t0, t1, >guard_ok
            ; guard_fail:
            ; li a0, guard_return_value
            ; j => exit_label
            ; guard_ok:
        );

        Ok(Guard {
            index: guard_index,
            bailout_ip: 0,
            kind: match expected_type {
                ValueType::Int => GuardKind::IntType { register },
                ValueType::Float => GuardKind::FloatType { register },
                ValueType::Bool => GuardKind::BoolType { register },
                _ => GuardKind::IntType { register }, // other types don't have specialized guard kinds
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
        extern "C" {
            fn jit_guard_function_identity(
                value_ptr: *const Value,
                expected_kind: u8,
                expected_function_idx: usize,
                expected_upvalues: *const (),
                register_index: u8,
            ) -> u8;
        }
        let guard_return_value = (guard_index + 1) as i32;
        let kind_flag = if is_closure { 1i32 } else { 0i32 };
        let reg_index = register as i32;
        let exit_label = self.current_exit_label();

        // a0 = value_ptr, a1 = kind, a2 = func_idx, a3 = upvalues, a4 = reg_index
        self.emit_addr_in_t2(register, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a0, t2);
        dynasm!(self.ops ; .arch riscv32i ; li a1, kind_flag);
        dynasm!(self.ops ; .arch riscv32i ; li a2, function_idx as i32);
        let upvalues_bits = upvalues_ptr as u32 as i32;
        dynasm!(self.ops ; .arch riscv32i ; li a3, upvalues_bits);
        dynasm!(self.ops ; .arch riscv32i ; li a4, reg_index);
        self.emit_load_fn_ptr(jit_guard_function_identity as *const ());
        self.emit_call_t0();
        dynasm!(self.ops
            ; .arch riscv32i
            ; bnez a0, >guard_ok
            ; guard_fail:
            ; li a0, guard_return_value
            ; j => exit_label
            ; guard_ok:
        );

        let kind = if is_closure {
            GuardKind::Closure { register, function_idx, upvalues_ptr }
        } else {
            GuardKind::Function { register, function_idx }
        };
        Ok(Guard { index: guard_index, bailout_ip: 0, kind, fail_count: 0, side_trace: None })
    }

    pub(super) fn compile_guard_native_function(
        &mut self,
        register: u8,
        expected_ptr: *const (),
        guard_index: usize,
    ) -> Result<Guard> {
        extern "C" {
            fn jit_guard_native_function(
                value_ptr: *const Value,
                expected: *const (),
                register_index: u8,
            ) -> u8;
        }
        let guard_return_value = (guard_index + 1) as i32;
        let reg_index = register as i32;
        let exit_label = self.current_exit_label();
        let expected_bits = expected_ptr as u32 as i32;

        self.emit_addr_in_t2(register, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a0, t2);
        dynasm!(self.ops ; .arch riscv32i ; li a1, expected_bits);
        dynasm!(self.ops ; .arch riscv32i ; li a2, reg_index);
        self.emit_load_fn_ptr(jit_guard_native_function as *const ());
        self.emit_call_t0();
        dynasm!(self.ops
            ; .arch riscv32i
            ; bnez a0, >guard_ok
            ; guard_fail:
            ; li a0, guard_return_value
            ; j => exit_label
            ; guard_ok:
        );

        Ok(Guard {
            index: guard_index,
            bailout_ip: 0,
            kind: GuardKind::NativeFunction { register, expected: expected_ptr },
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
        extern "C" {
            fn jit_value_is_truthy(value_ptr: *const Value) -> u8;
        }
        let guard_return_value = (guard_index + 1) as i32;
        let exit_label = self.current_exit_label();

        self.emit_addr_in_t2(condition_register, 0);
        dynasm!(self.ops ; .arch riscv32i ; mv a0, t2);
        self.emit_load_fn_ptr(jit_value_is_truthy as *const ());
        self.emit_call_t0();
        // a0 = 0 (falsy) or 1 (truthy)

        if expect_truthy {
            dynasm!(self.ops
                ; .arch riscv32i
                ; bnez a0, >guard_ok
                ; li a0, guard_return_value
                ; j => exit_label
                ; guard_ok:
            );
        } else {
            dynasm!(self.ops
                ; .arch riscv32i
                ; beqz a0, >guard_ok
                ; li a0, guard_return_value
                ; j => exit_label
                ; guard_ok:
            );
        }

        let kind = if expect_truthy {
            GuardKind::Truthy { register: condition_register }
        } else {
            GuardKind::Falsy { register: condition_register }
        };
        Ok(Guard { index: guard_index, bailout_ip, kind, fail_count: 0, side_trace: None })
    }
}

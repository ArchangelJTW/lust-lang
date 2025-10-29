use super::*;
impl JitCompiler {
    pub(super) fn compile_guard(
        &mut self,
        register: u8,
        expected_type: ValueType,
        guard_index: usize,
    ) -> Result<Guard> {
        let offset = (register as i32) * (mem::size_of::<Value>() as i32);
        let expected_tag = match expected_type {
            ValueType::Bool => ValueTag::Bool,
            ValueType::Int => ValueTag::Int,
            ValueType::Float => ValueTag::Float,
            ValueType::String => ValueTag::String,
            ValueType::Array => ValueTag::Array,
            ValueType::Tuple => ValueTag::Tuple,
            ValueType::Struct => ValueTag::Struct,
        };
        let expected_discriminant = expected_tag.as_u8() as i8;
        let guard_return_value = (guard_index + 1) as i32;
        dynasm!(self.ops
            ; mov al, [r12 + offset]
            ; cmp al, BYTE expected_discriminant
            ; jne >guard_fail
            ; jmp >guard_ok
            ; guard_fail:
            ; mov eax, DWORD guard_return_value
            ; jmp >exit
            ; guard_ok:
        );
        Ok(Guard {
            index: guard_index,
            bailout_ip: 0,
            kind: match expected_type {
                ValueType::Int => GuardKind::IntType { register },
                ValueType::Float => GuardKind::FloatType { register },
                ValueType::Bool => GuardKind::Truthy { register },
                ValueType::String => GuardKind::IntType { register },
                ValueType::Array => GuardKind::IntType { register },
                ValueType::Tuple => GuardKind::IntType { register },
                ValueType::Struct => GuardKind::IntType { register },
            },
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
        let offset = (register as i32) * (mem::size_of::<Value>() as i32);
        let guard_return_value = (guard_index + 1) as i32;
        extern "C" {
            fn jit_guard_native_function(
                value_ptr: *const Value,
                expected: *const (),
                register_index: u8,
            ) -> u8;
        }
        let reg_index = register as i32;
        dynasm!(self.ops
            ; lea rdi, [r12 + offset]
            ; mov rsi, QWORD expected_ptr as _
            ; mov edx, DWORD reg_index
            ; sub rsp, 8
            ; mov rax, QWORD jit_guard_native_function as _
            ; call rax
            ; add rsp, 8
            ; test al, al
            ; jz >guard_fail
            ; jmp >guard_ok
            ; guard_fail:
            ; mov eax, DWORD guard_return_value
            ; jmp >exit
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

    pub(super) fn compile_loop_continue_guard(
        &mut self,
        condition_register: u8,
        bailout_ip: usize,
        guard_index: usize,
    ) -> Result<Guard> {
        let cond_offset = (condition_register as i32) * (mem::size_of::<Value>() as i32);
        let guard_return_value = (guard_index + 1) as i32;
        dynasm!(self.ops
            ; mov al, [r12 + cond_offset + 8]
            ; test al, al
            ; jz >loop_exit
            ; jmp >loop_continue
            ; loop_exit:
            ; mov eax, DWORD guard_return_value
            ; jmp >exit
            ; loop_continue:
        );
        Ok(Guard {
            index: guard_index,
            bailout_ip,
            kind: GuardKind::Truthy {
                register: condition_register,
            },
            fail_count: 0,
            side_trace: None,
        })
    }
}

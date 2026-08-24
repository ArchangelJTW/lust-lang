pub(super) use super::specialization::{SpecializationRegistry, SpecializedLayout};
pub(super) use super::trace::InlineTrace;
pub(super) use super::trace::ValueType;
pub(super) use super::{CompiledTrace, Guard, GuardKind, JitData, Trace, TraceId, TraceOp};
pub(super) use crate::bytecode::{Function, Value, ValueTag};
pub(super) use crate::jit;
pub(super) use crate::Result;
pub(super) use alloc::{boxed::Box, vec::Vec};
pub(super) use core::mem;
pub(super) use dynasmrt::{dynasm, x64::Assembler, DynasmApi, DynasmLabelApi};
use hashbrown::HashMap;

/// Minimum stack allocation size for traces. Individual traces can request more
/// space depending on how many specialized values they materialize.
/// Must stay (8 mod 16) to preserve SysV stack alignment guarantees.
pub(super) const MIN_JIT_STACK_SIZE: i32 = 504;

/// Base offset for specialized value allocations (must avoid saved registers at rbp-40)
pub(super) const SPECIALIZED_BASE_OFFSET: i32 = -64;
/// Size (in bytes) reserved per specialized value (ptr + len + cap + padding)
pub(super) const SPECIALIZED_SLOT_SIZE: i32 = 32;
/// Extra stack space required before the first specialized slot to avoid the
/// saved callee-saved registers (rbp-8 through rbp-40).
pub(super) const SPECIALIZED_STACK_BASE: i32 = 64;
mod arithmetic;
mod builder;
mod comparisons;
mod guards;
mod logic;
mod memory;
mod registers;
mod specialization;
/// Tracks a specialized value in the JIT trace
#[derive(Debug, Clone)]
pub(super) struct SpecializedValue {
    pub stack_offset: i32,
}

pub struct JitCompiler {
    pub(super) ops: Assembler,
    pub(super) data: Vec<JitData>,
    fail_stack: Vec<dynasmrt::DynamicLabel>,
    exit_stack: Vec<dynasmrt::DynamicLabel>,
    inline_depth: usize,
    /// Registry for type specializations
    #[allow(dead_code)]
    pub(super) specialization_registry: SpecializationRegistry,
    /// Track active specialized values in trace
    pub(super) specialized_values: HashMap<usize, SpecializedValue>,
    /// Registers proven to contain non-owning scalar values at this point.
    pub(super) scalar_registers: HashMap<u8, ValueType>,
    /// Next ID for specialized values
    #[allow(dead_code)]
    pub(super) next_specialized_id: usize,
}

impl Default for JitCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::rc::Rc;

    #[test]
    fn retained_strings_and_pointer_arrays_survive_compiler_moves() {
        let mut compiler = JitCompiler::new();
        let (first, first_len) = compiler.retain_string("first");
        let (second, second_len) = compiler.retain_string("second");
        let pointers = compiler.retain_string_pointers(vec![first, second]);
        let lengths = compiler.retain_string_lengths(vec![first_len, second_len]);

        let compiler = *Box::new(compiler);
        let retained_data = compiler.data;

        unsafe {
            let pointers = core::slice::from_raw_parts(pointers, 2);
            let lengths = core::slice::from_raw_parts(lengths, 2);
            assert_eq!(
                core::slice::from_raw_parts(pointers[0], lengths[0]),
                b"first"
            );
            assert_eq!(
                core::slice::from_raw_parts(pointers[1], lengths[1]),
                b"second"
            );
        }

        drop(retained_data);
    }

    #[test]
    fn generated_scalar_store_drops_previous_value() {
        let trace = Trace {
            function_idx: 0,
            start_ip: 0,
            preamble: Vec::new(),
            ops: vec![
                TraceOp::LoadConst {
                    dest: 0,
                    value: Value::Int(7),
                },
                TraceOp::NestedLoopCall {
                    function_idx: 0,
                    loop_start_ip: 0,
                    bailout_ip: 0,
                },
            ],
            postamble: Vec::new(),
            inputs: Vec::new(),
            outputs: vec![0],
        };
        let compiled = JitCompiler::new()
            .compile_trace(&trace, TraceId(0), None, Vec::new())
            .unwrap();
        let string = Rc::new("old register value".to_string());
        let mut registers = vec![Value::String(string.clone())];

        compiled.execute(
            registers.as_mut_ptr(),
            core::ptr::null_mut(),
            core::ptr::null(),
        );

        assert_eq!(Rc::strong_count(&string), 1);
        assert!(matches!(registers[0], Value::Int(7)));
    }

    #[test]
    fn generated_guard_returns_its_exit_code() {
        let trace = Trace {
            function_idx: 0,
            start_ip: 0,
            preamble: Vec::new(),
            ops: vec![TraceOp::Guard {
                register: 0,
                expected_type: ValueType::Int,
            }],
            postamble: Vec::new(),
            inputs: vec![0],
            outputs: Vec::new(),
        };
        let compiled = JitCompiler::new()
            .compile_trace(&trace, TraceId(0), None, Vec::new())
            .unwrap();
        let mut registers = vec![Value::Bool(false)];

        let result = compiled.execute(
            registers.as_mut_ptr(),
            core::ptr::null_mut(),
            core::ptr::null(),
        );

        assert_eq!(result, 1);
    }

    #[test]
    fn fused_integer_comparison_materializes_failed_condition_for_bailout() {
        let trace = Trace {
            function_idx: 0,
            start_ip: 0,
            preamble: Vec::new(),
            ops: vec![
                TraceOp::Guard {
                    register: 0,
                    expected_type: ValueType::Int,
                },
                TraceOp::Guard {
                    register: 1,
                    expected_type: ValueType::Int,
                },
                TraceOp::Le {
                    dest: 2,
                    lhs: 0,
                    rhs: 1,
                    lhs_type: ValueType::Int,
                    rhs_type: ValueType::Int,
                },
                TraceOp::GuardLoopContinue {
                    condition_register: 2,
                    expect_truthy: true,
                    bailout_ip: 9,
                },
            ],
            postamble: Vec::new(),
            inputs: vec![0, 1],
            outputs: vec![2],
        };
        let compiled = JitCompiler::new()
            .compile_trace(&trace, TraceId(0), None, Vec::new())
            .unwrap();
        let mut registers = vec![
            Value::Int(3),
            Value::Int(2),
            Value::String(Rc::new("old".into())),
        ];

        let result = compiled.execute(
            registers.as_mut_ptr(),
            core::ptr::null_mut(),
            core::ptr::null(),
        );

        assert_eq!(result, 3);
        assert!(matches!(registers[2], Value::Bool(false)));
        assert_eq!(compiled.guards[2].bailout_ip, 9);
    }

    #[test]
    fn live_comparison_and_constant_temporaries_are_materialized() {
        let trace = Trace {
            function_idx: 0,
            start_ip: 0,
            preamble: Vec::new(),
            ops: vec![
                TraceOp::Lt {
                    dest: 2,
                    lhs: 0,
                    rhs: 1,
                    lhs_type: ValueType::Int,
                    rhs_type: ValueType::Int,
                },
                TraceOp::GuardLoopContinue {
                    condition_register: 2,
                    expect_truthy: true,
                    bailout_ip: 4,
                },
                TraceOp::Move { dest: 3, src: 2 },
                TraceOp::LoadConst {
                    dest: 4,
                    value: Value::Int(1),
                },
                TraceOp::Add {
                    dest: 5,
                    lhs: 0,
                    rhs: 4,
                    lhs_type: ValueType::Int,
                    rhs_type: ValueType::Int,
                },
                TraceOp::Move { dest: 6, src: 4 },
                TraceOp::NestedLoopCall {
                    function_idx: 0,
                    loop_start_ip: 0,
                    bailout_ip: 0,
                },
            ],
            postamble: Vec::new(),
            inputs: vec![0, 1],
            outputs: vec![3, 5, 6],
        };
        let compiled = JitCompiler::new()
            .compile_trace(&trace, TraceId(0), None, Vec::new())
            .unwrap();
        let old_condition = Rc::new("condition".to_string());
        let old_constant = Rc::new("constant".to_string());
        let mut registers = vec![
            Value::Int(1),
            Value::Int(2),
            Value::String(old_condition.clone()),
            Value::Nil,
            Value::String(old_constant.clone()),
            Value::Nil,
            Value::Nil,
        ];

        compiled.execute(
            registers.as_mut_ptr(),
            core::ptr::null_mut(),
            core::ptr::null(),
        );

        assert!(matches!(registers[3], Value::Bool(true)));
        assert!(matches!(registers[5], Value::Int(2)));
        assert!(matches!(registers[6], Value::Int(1)));
        assert_eq!(Rc::strong_count(&old_condition), 1);
        assert_eq!(Rc::strong_count(&old_constant), 1);
    }

    #[test]
    fn elided_temporaries_do_not_cross_intervening_side_exits() {
        let comparison_trace = Trace {
            function_idx: 0,
            start_ip: 0,
            preamble: Vec::new(),
            ops: vec![
                TraceOp::Guard {
                    register: 2,
                    expected_type: ValueType::Bool,
                },
                TraceOp::Lt {
                    dest: 2,
                    lhs: 0,
                    rhs: 1,
                    lhs_type: ValueType::Int,
                    rhs_type: ValueType::Int,
                },
                TraceOp::GuardLoopContinue {
                    condition_register: 2,
                    expect_truthy: true,
                    bailout_ip: 4,
                },
                TraceOp::Guard {
                    register: 3,
                    expected_type: ValueType::Int,
                },
                TraceOp::LoadConst {
                    dest: 2,
                    value: Value::Int(0),
                },
            ],
            postamble: Vec::new(),
            inputs: vec![0, 1, 2, 3],
            outputs: vec![2],
        };
        let compiled = JitCompiler::new()
            .compile_trace(&comparison_trace, TraceId(0), None, Vec::new())
            .unwrap();
        let mut registers = vec![
            Value::Int(1),
            Value::Int(2),
            Value::Bool(false),
            Value::Bool(false),
        ];

        assert_eq!(
            compiled.execute(
                registers.as_mut_ptr(),
                core::ptr::null_mut(),
                core::ptr::null(),
            ),
            3
        );
        assert!(matches!(registers[2], Value::Bool(true)));

        let constant_trace = Trace {
            function_idx: 0,
            start_ip: 0,
            preamble: Vec::new(),
            ops: vec![
                TraceOp::Guard {
                    register: 4,
                    expected_type: ValueType::Int,
                },
                TraceOp::LoadConst {
                    dest: 4,
                    value: Value::Int(1),
                },
                TraceOp::Add {
                    dest: 5,
                    lhs: 0,
                    rhs: 4,
                    lhs_type: ValueType::Int,
                    rhs_type: ValueType::Int,
                },
                TraceOp::Guard {
                    register: 3,
                    expected_type: ValueType::Int,
                },
                TraceOp::LoadConst {
                    dest: 4,
                    value: Value::Int(2),
                },
            ],
            postamble: Vec::new(),
            inputs: vec![0, 3, 4],
            outputs: vec![4, 5],
        };
        let compiled = JitCompiler::new()
            .compile_trace(&constant_trace, TraceId(1), None, Vec::new())
            .unwrap();
        let mut registers = vec![
            Value::Int(1),
            Value::Nil,
            Value::Nil,
            Value::Bool(false),
            Value::Int(99),
            Value::Nil,
        ];

        assert_eq!(
            compiled.execute(
                registers.as_mut_ptr(),
                core::ptr::null_mut(),
                core::ptr::null(),
            ),
            2
        );
        assert!(matches!(registers[4], Value::Int(1)));
        assert!(matches!(registers[5], Value::Int(2)));
    }

    #[test]
    fn generated_failure_returns_negative_exit_code() {
        let trace = Trace {
            function_idx: 0,
            start_ip: 0,
            preamble: Vec::new(),
            ops: vec![TraceOp::Div {
                dest: 0,
                lhs: 1,
                rhs: 2,
                lhs_type: ValueType::Int,
                rhs_type: ValueType::Int,
            }],
            postamble: Vec::new(),
            inputs: vec![1, 2],
            outputs: vec![0],
        };
        let compiled = JitCompiler::new()
            .compile_trace(&trace, TraceId(0), None, Vec::new())
            .unwrap();
        let mut registers = vec![Value::Nil, Value::Int(7), Value::Int(0)];

        let result = compiled.execute(
            registers.as_mut_ptr(),
            core::ptr::null_mut(),
            core::ptr::null(),
        );

        assert_eq!(result, -1);
    }

    #[test]
    fn generated_signed_division_and_modulo_match_integer_semantics() {
        let trace = Trace {
            function_idx: 0,
            start_ip: 0,
            preamble: Vec::new(),
            ops: vec![
                TraceOp::Div {
                    dest: 0,
                    lhs: 2,
                    rhs: 3,
                    lhs_type: ValueType::Int,
                    rhs_type: ValueType::Int,
                },
                TraceOp::Mod {
                    dest: 1,
                    lhs: 2,
                    rhs: 3,
                    lhs_type: ValueType::Int,
                    rhs_type: ValueType::Int,
                },
                TraceOp::NestedLoopCall {
                    function_idx: 0,
                    loop_start_ip: 0,
                    bailout_ip: 0,
                },
            ],
            postamble: Vec::new(),
            inputs: vec![2, 3],
            outputs: vec![0, 1],
        };
        let compiled = JitCompiler::new()
            .compile_trace(&trace, TraceId(0), None, Vec::new())
            .unwrap();
        let mut registers = vec![Value::Nil, Value::Nil, Value::Int(-7), Value::Int(2)];

        let result = compiled.execute(
            registers.as_mut_ptr(),
            core::ptr::null_mut(),
            core::ptr::null(),
        );

        assert_eq!(result, 1);
        assert!(matches!(registers[0], Value::Int(-3)));
        assert!(matches!(registers[1], Value::Int(-1)));
    }

    #[test]
    fn generated_scalar_comparisons_match_value_semantics() {
        let trace = Trace {
            function_idx: 0,
            start_ip: 0,
            preamble: Vec::new(),
            ops: vec![
                TraceOp::Lt {
                    dest: 0,
                    lhs: 6,
                    rhs: 7,
                    lhs_type: ValueType::Float,
                    rhs_type: ValueType::Float,
                },
                TraceOp::Ge {
                    dest: 1,
                    lhs: 8,
                    rhs: 9,
                    lhs_type: ValueType::Int,
                    rhs_type: ValueType::Float,
                },
                TraceOp::Eq {
                    dest: 2,
                    lhs: 10,
                    rhs: 10,
                    lhs_type: ValueType::Float,
                    rhs_type: ValueType::Float,
                },
                TraceOp::Ne {
                    dest: 3,
                    lhs: 10,
                    rhs: 10,
                    lhs_type: ValueType::Float,
                    rhs_type: ValueType::Float,
                },
                TraceOp::Eq {
                    dest: 4,
                    lhs: 11,
                    rhs: 12,
                    lhs_type: ValueType::Bool,
                    rhs_type: ValueType::Bool,
                },
                TraceOp::Eq {
                    dest: 5,
                    lhs: 8,
                    rhs: 13,
                    lhs_type: ValueType::Int,
                    rhs_type: ValueType::Float,
                },
                TraceOp::NestedLoopCall {
                    function_idx: 0,
                    loop_start_ip: 0,
                    bailout_ip: 0,
                },
            ],
            postamble: Vec::new(),
            inputs: (6..=13).collect(),
            outputs: (0..=5).collect(),
        };
        let compiled = JitCompiler::new()
            .compile_trace(&trace, TraceId(0), None, Vec::new())
            .unwrap();
        let mut registers = vec![
            Value::Nil,
            Value::Nil,
            Value::Nil,
            Value::Nil,
            Value::Nil,
            Value::Nil,
            Value::Float(-2.5),
            Value::Float(-1.0),
            Value::Int(3),
            Value::Float(2.5),
            Value::Float(f64::NAN),
            Value::Bool(true),
            Value::Bool(true),
            Value::Float(3.0),
        ];

        let result = compiled.execute(
            registers.as_mut_ptr(),
            core::ptr::null_mut(),
            core::ptr::null(),
        );

        assert_eq!(result, 1);
        assert!(matches!(registers[0], Value::Bool(true)));
        assert!(matches!(registers[1], Value::Bool(true)));
        assert!(matches!(registers[2], Value::Bool(false)));
        assert!(matches!(registers[3], Value::Bool(true)));
        assert!(matches!(registers[4], Value::Bool(true)));
        assert!(matches!(registers[5], Value::Bool(false)));
    }
}

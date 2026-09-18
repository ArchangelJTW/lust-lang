// AArch64 JIT codegen — requires the `std` feature and target_arch = "aarch64".
//
// This is a port of the x86_64 backend in `src/jit/codegen`. Every `dynasm!`
// invocation begins with `; .arch aarch64` because the proc macro selects its
// default architecture from the *host* at compile time.
//
// Register conventions in generated traces (AAPCS64):
//   x19 – register-array base   (x0 on entry)          callee-saved
//   x20 – VM pointer            (x1 on entry)          callee-saved
//   x21 – inline-frame chain head (0 = none)           callee-saved
//   x22 – exit code preserved across the postamble     callee-saved
//   x23..x28, d8..d15 – pinned VM registers (see pins.rs)  callee-saved
//   x0  – primary scratch / helper result ("rax")
//   x9  – secondary scratch ("rbx")
//   x10 – tertiary scratch ("rcx")
//   x11 – address scratch for out-of-range immediates
//   x12 – immediate scratch used by `emit_add_imm`
//   x13 – tag-byte scratch used by `store_tag_imm`
//   x16 – call target for helper calls (`blr x16`)
//   d0/d1 – float temporaries
//
// Value layout (`#[repr(C, u8)]`, 8-byte aligned): discriminant byte at offset 0,
// payload at offset 8, `mem::size_of::<Value>()` bytes per register.
//
// Frame layout (x29 = frame pointer, grows downward):
//   [x29 + 8]   saved x30 (lr)
//   [x29 + 0]   saved x29
//   [x29 - 8]   saved x20
//   [x29 - 16]  saved x19
//   [x29 - 24]  saved x22
//   [x29 - 32]  saved x21
//   [x29 - 80, x29 - 32)   saved x23..x28
//   [x29 - 144, x29 - 80)  saved d8..d15
//   [x29 - 144 - stack_size, x29 - 144)  local area (specialized slots)
//
// Exits restore `sp` from x29, so an exit taken from inside an inlined call
// frame (which lives below the local area) unwinds correctly. The exit path
// also drains the inline-frame chain in x21, dropping callee registers.

pub(super) use super::specialization::{SpecializationRegistry, SpecializedLayout};
pub(super) use super::trace::InlineTrace;
pub(super) use super::trace::ValueType;
pub(super) use super::{CompiledTrace, Guard, GuardKind, JitData, Trace, TraceId, TraceOp};
pub(super) use crate::Result;
pub(super) use crate::bytecode::{Function, Value, ValueTag};
pub(super) use crate::jit;
pub(super) use alloc::{boxed::Box, vec::Vec};
pub(super) use core::mem;
pub(super) use dynasmrt::{DynasmApi, DynasmLabelApi, aarch64::Assembler, dynasm};
use hashbrown::HashMap;

/// Minimum local-area allocation for traces. Individual traces can request
/// more depending on how many specialized values they materialize.
/// Must be a multiple of 16 to keep `sp` aligned.
pub(super) const MIN_JIT_STACK_SIZE: i32 = 512;

/// Bytes of callee-saved registers stored below x29: x19..x28 (5 pairs)
/// and d8..d15 (4 pairs). x23..x28 and d8..d15 hold pinned VM registers
/// (see `pins`).
pub(super) const SAVED_BELOW_FP: i32 = 32 + 48 + 64;

/// Offset from x29 of the first specialized slot. Slots grow downward from
/// just below the saved registers.
pub(super) const SPECIALIZED_BASE_OFFSET: i32 = -(SAVED_BELOW_FP + 32);
/// Size (in bytes) reserved per specialized value (ptr + len + cap + padding)
pub(super) const SPECIALIZED_SLOT_SIZE: i32 = 32;
/// Local-area bytes needed before the first specialized slot.
pub(super) const SPECIALIZED_STACK_BASE: i32 = 32;

/// Conditional branches and `cbz` reach only ±1 MB. Traces can exceed that
/// (thousands of specialized ops), so `compile_ops` plants a local `fail:`
/// island — a plain `b` (±128 MB) on to the next `fail:` — whenever this
/// many bytes have been emitted since the last one, keeping every `>fail`
/// reference within reach of the island that follows it.
pub(super) const FAIL_ISLAND_INTERVAL: usize = 900 * 1024;

/// Size of the metadata block pushed for each inlined call frame:
/// { value_count: u64, saved_x19: *mut Value, prev_x21: *const u8, pad }
pub(super) const INLINE_METADATA_SIZE: i32 = 32;

mod arithmetic;
mod builder;
mod comparisons;
mod guards;
mod logic;
mod memory;
mod pins;
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
    /// Code offset of the last fail island (see `FAIL_ISLAND_INTERVAL`).
    last_fail_island: usize,
    /// Registry for type specializations
    #[allow(dead_code)]
    pub(super) specialization_registry: SpecializationRegistry,
    /// Track active specialized values in trace
    pub(super) specialized_values: HashMap<usize, SpecializedValue>,
    /// Registers proven to contain non-owning scalar values at this point.
    pub(super) scalar_registers: HashMap<u8, ValueType>,
    /// VM registers held in machine registers for the loop body.
    pins: HashMap<u8, pins::Pin>,
    /// Pins apply only while compiling the loop body (not the preamble,
    /// postamble, or inline-call bodies, which address other storage).
    pin_active: bool,
    /// Carried pins whose machine value may be newer than memory.
    dirty_pins: Vec<u8>,
    /// Bytecode ip of the instruction the ops being compiled came from
    /// (from the last `At` marker), if known.
    current_fail_ip: Option<usize>,
    /// Resume ips for failure exits, indexed by fail-stub number.
    fail_sites: Vec<usize>,
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
                    resume_ip: 0,
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
    fn fused_numeric_guards_preserve_ordering_and_bailout_state() {
        let cases = [
            (
                Value::Float(1.0),
                Value::Float(2.0),
                [true, true, false, false],
            ),
            (
                Value::Float(2.0),
                Value::Float(1.0),
                [false, false, true, true],
            ),
            (
                Value::Float(-0.0),
                Value::Float(0.0),
                [false, true, false, true],
            ),
            (
                Value::Float(f64::INFINITY),
                Value::Float(f64::INFINITY),
                [false, true, false, true],
            ),
            (
                Value::Float(f64::NEG_INFINITY),
                Value::Float(f64::INFINITY),
                [true, true, false, false],
            ),
            (Value::Float(f64::NAN), Value::Float(1.0), [false; 4]),
            (Value::Float(1.0), Value::Float(f64::NAN), [false; 4]),
            (Value::Int(1), Value::Float(2.0), [true, true, false, false]),
            (Value::Float(2.0), Value::Int(1), [false, false, true, true]),
            (Value::Int(1), Value::Float(f64::NAN), [false; 4]),
            (Value::Float(f64::NAN), Value::Int(1), [false; 4]),
            (
                Value::Int(i64::MAX - 1),
                Value::Int(i64::MAX),
                [true, true, false, false],
            ),
        ];
        for (left, right, expected) in cases {
            let value_type = |value: &Value| match value {
                Value::Int(_) => ValueType::Int,
                Value::Float(_) => ValueType::Float,
                _ => unreachable!(),
            };
            let lhs_type = value_type(&left);
            let rhs_type = value_type(&right);
            for (index, comparison) in [
                TraceOp::Lt {
                    dest: 2,
                    lhs: 0,
                    rhs: 1,
                    lhs_type,
                    rhs_type,
                },
                TraceOp::Le {
                    dest: 2,
                    lhs: 0,
                    rhs: 1,
                    lhs_type,
                    rhs_type,
                },
                TraceOp::Gt {
                    dest: 2,
                    lhs: 0,
                    rhs: 1,
                    lhs_type,
                    rhs_type,
                },
                TraceOp::Ge {
                    dest: 2,
                    lhs: 0,
                    rhs: 1,
                    lhs_type,
                    rhs_type,
                },
            ]
            .into_iter()
            .enumerate()
            {
                for expect_truthy in [false, true] {
                    for live_condition in [false, true] {
                        let mut trace = Trace {
                            function_idx: 0,
                            start_ip: 0,
                            preamble: vec![TraceOp::Guard {
                                register: 2,
                                expected_type: ValueType::Bool,
                            }],
                            ops: vec![
                                comparison.clone(),
                                TraceOp::GuardLoopContinue {
                                    condition_register: 2,
                                    expect_truthy,
                                    bailout_ip: 9,
                                },
                                TraceOp::LoadConst {
                                    dest: 2,
                                    value: Value::Int(7),
                                },
                                TraceOp::NestedLoopCall {
                                    function_idx: 0,
                                    loop_start_ip: 0,
                                    bailout_ip: 10,
                                    resume_ip: 0,
                                },
                            ],
                            postamble: Vec::new(),
                            inputs: vec![0, 1, 2],
                            outputs: vec![2, 3],
                        };
                        if live_condition {
                            // A subsequent use must disable boolean elision.
                            trace.ops.insert(2, TraceOp::Move { dest: 3, src: 2 });
                        }
                        let compiled = JitCompiler::new()
                            .compile_trace(&trace, TraceId(0), None, Vec::new())
                            .unwrap();
                        let mut registers = vec![
                            left.clone(),
                            right.clone(),
                            Value::Bool(!expected[index]),
                            Value::Nil,
                        ];
                        let result = compiled.execute(
                            registers.as_mut_ptr(),
                            core::ptr::null_mut(),
                            core::ptr::null(),
                        );
                        if expected[index] == expect_truthy {
                            assert_eq!(result, 3);
                            assert_eq!(registers[2], Value::Int(7));
                            if live_condition {
                                assert_eq!(registers[3], Value::Bool(expected[index]));
                            }
                        } else {
                            assert_eq!(result, 2);
                            assert_eq!(registers[2], Value::Bool(expected[index]));
                            assert_eq!(compiled.guards[1].bailout_ip, 9);
                        }
                    }
                }
            }
        }
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
                    resume_ip: 0,
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
                    resume_ip: 0,
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
    fn generated_float_modulo_matches_interpreter_semantics() {
        // (lhs, rhs, expected) — Rust `%` on f64 (fmod): sign of the dividend.
        let cases = [
            (Value::Float(7.5), Value::Float(2.0), 1.5),
            (Value::Float(-7.5), Value::Float(2.0), -1.5),
            (Value::Float(7.5), Value::Float(-2.0), 1.5),
            (Value::Int(7), Value::Float(2.5), 2.0),
            (Value::Float(7.5), Value::Int(2), 1.5),
            (Value::Float(1e300), Value::Float(3.0), 1e300 % 3.0),
        ];
        for (lhs, rhs, expected) in cases {
            let value_type = |value: &Value| match value {
                Value::Int(_) => ValueType::Int,
                Value::Float(_) => ValueType::Float,
                _ => unreachable!(),
            };
            let trace = Trace {
                function_idx: 0,
                start_ip: 0,
                preamble: Vec::new(),
                ops: vec![
                    TraceOp::Mod {
                        dest: 0,
                        lhs: 1,
                        rhs: 2,
                        lhs_type: value_type(&lhs),
                        rhs_type: value_type(&rhs),
                    },
                    TraceOp::NestedLoopCall {
                        function_idx: 0,
                        loop_start_ip: 0,
                        bailout_ip: 0,
                        resume_ip: 0,
                    },
                ],
                postamble: Vec::new(),
                inputs: vec![1, 2],
                outputs: vec![0],
            };
            let compiled = JitCompiler::new()
                .compile_trace(&trace, TraceId(0), None, Vec::new())
                .unwrap();
            let mut registers = vec![Value::Nil, lhs.clone(), rhs.clone()];
            let result = compiled.execute(
                registers.as_mut_ptr(),
                core::ptr::null_mut(),
                core::ptr::null(),
            );
            assert_eq!(result, 1, "{lhs:?} % {rhs:?}");
            assert_eq!(registers[0], Value::Float(expected), "{lhs:?} % {rhs:?}");
        }

        // Modulo by zero and NaN follow the interpreter: zero fails the
        // trace (the interpreter raises), NaN propagates.
        let run = |lhs: Value, rhs: Value| {
            let trace = Trace {
                function_idx: 0,
                start_ip: 0,
                preamble: Vec::new(),
                ops: vec![
                    TraceOp::Mod {
                        dest: 0,
                        lhs: 1,
                        rhs: 2,
                        lhs_type: ValueType::Float,
                        rhs_type: ValueType::Float,
                    },
                    TraceOp::NestedLoopCall {
                        function_idx: 0,
                        loop_start_ip: 0,
                        bailout_ip: 0,
                        resume_ip: 0,
                    },
                ],
                postamble: Vec::new(),
                inputs: vec![1, 2],
                outputs: vec![0],
            };
            let compiled = JitCompiler::new()
                .compile_trace(&trace, TraceId(0), None, Vec::new())
                .unwrap();
            let mut registers = vec![Value::Nil, lhs, rhs];
            let result = compiled.execute(
                registers.as_mut_ptr(),
                core::ptr::null_mut(),
                core::ptr::null(),
            );
            (result, registers.remove(0))
        };
        assert_eq!(run(Value::Float(1.0), Value::Float(0.0)).0, -1);
        assert_eq!(run(Value::Float(1.0), Value::Float(-0.0)).0, -1);
        let (result, value) = run(Value::Float(1.0), Value::Float(f64::NAN));
        assert_eq!(result, 1);
        assert!(matches!(value, Value::Float(f) if f.is_nan()));
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
                    resume_ip: 0,
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

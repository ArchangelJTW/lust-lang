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
}

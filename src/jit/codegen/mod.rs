pub(super) use super::specialization::{SpecializationRegistry, SpecializedLayout};
pub(super) use super::trace::InlineTrace;
pub(super) use super::trace::ValueType;
pub(super) use super::{CompiledTrace, Guard, GuardKind, Trace, TraceId, TraceOp};
pub(super) use crate::bytecode::{Function, Value, ValueTag};
pub(super) use crate::jit;
pub(super) use crate::Result;
pub(super) use alloc::vec::Vec;
pub(super) use core::mem;
pub(super) use dynasmrt::{dynasm, x64::Assembler, DynasmApi, DynasmLabelApi};
use hashbrown::HashMap;
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
    pub layout: SpecializedLayout,
    pub stack_offset: i32,
    /// Original register it came from (for debugging)
    pub source_reg: Option<u8>,
}

pub struct JitCompiler {
    pub(super) ops: Assembler,
    pub(super) leaked_constants: Vec<*const Value>,
    fail_stack: Vec<dynasmrt::DynamicLabel>,
    exit_stack: Vec<dynasmrt::DynamicLabel>,
    inline_depth: usize,
    /// Registry for type specializations
    pub(super) specialization_registry: SpecializationRegistry,
    /// Track active specialized values in trace
    pub(super) specialized_values: HashMap<usize, SpecializedValue>,
    /// Next ID for specialized values
    pub(super) next_specialized_id: usize,
}

impl Default for JitCompiler {
    fn default() -> Self {
        Self::new()
    }
}

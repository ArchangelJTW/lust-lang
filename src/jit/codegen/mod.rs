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

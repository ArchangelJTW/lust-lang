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

/// JIT stack allocation size in bytes
/// Must be (8 mod 16) to maintain 16-byte alignment after pushing 6 registers (48 bytes)
/// Stack layout: [rbp-8]:rbx, [rbp-16]:r12, [rbp-24]:r13, [rbp-32]:r14, [rbp-40]:r15
/// Usable space: [rbp-41] to [rbp-(40+STACK_SIZE)]
pub(super) const JIT_STACK_SIZE: i32 = 504;

/// Base offset for specialized value allocations (must avoid saved registers at rbp-40)
pub(super) const SPECIALIZED_BASE_OFFSET: i32 = -64;
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

pub(super) use super::trace::InlineTrace;
pub(super) use super::trace::ValueType;
pub(super) use super::{CompiledTrace, Guard, GuardKind, Trace, TraceId, TraceOp};
pub(super) use crate::bytecode::{Function, Value, ValueTag};
pub(super) use crate::jit;
pub(super) use crate::Result;
pub(super) use alloc::vec::Vec;
pub(super) use core::mem;
pub(super) use dynasmrt::{dynasm, x64::Assembler, DynasmApi, DynasmLabelApi};
mod arithmetic;
mod builder;
mod comparisons;
mod guards;
mod logic;
mod memory;
mod registers;
pub struct JitCompiler {
    pub(super) ops: Assembler,
    pub(super) leaked_constants: Vec<*const Value>,
    fail_stack: Vec<dynasmrt::DynamicLabel>,
    exit_stack: Vec<dynasmrt::DynamicLabel>,
    inline_depth: usize,
}

impl Default for JitCompiler {
    fn default() -> Self {
        Self::new()
    }
}

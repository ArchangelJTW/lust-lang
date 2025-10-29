pub(super) use super::trace::ValueType;
pub(super) use super::{CompiledTrace, Guard, GuardKind, Trace, TraceId, TraceOp};
pub(super) use crate::bytecode::{Function, Value, ValueTag};
pub(super) use crate::jit;
pub(super) use crate::Result;
pub(super) use dynasmrt::{dynasm, x64::Assembler, DynasmApi, DynasmLabelApi};
pub(super) use std::mem;
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
}

impl Default for JitCompiler {
    fn default() -> Self {
        Self::new()
    }
}

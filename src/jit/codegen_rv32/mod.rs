// RV32 JIT codegen — requires the `rv32` feature (which implies `std`) and target_arch = "riscv32".
//
// dynasm-rs uses `[base, offset]` syntax for memory references (not the
// traditional RISC-V `offset(base)` syntax).  Every `dynasm!` invocation
// must begin with `; .arch riscv32i` because the proc macro selects its
// default architecture from the *host* at compile time, which is typically
// x86_64 during cross-compilation.
//
// Register conventions in generated traces:
//   s2  (X18) – register-array base  (a0 on entry)
//   s3  (X19) – VM pointer           (a1 on entry)
//   s4  (X20) – unwind-chain head    (zero initially)
//   t0  (X5)  – primary scratch
//   t1  (X6)  – secondary scratch
//   t2  (X7)  – address-computation scratch
//
// Value layout under `no_std` / rv32:
//   LustInt   = i32  (4 bytes, align 4)
//   LustFloat = f32  (4 bytes, align 4)
//   #[repr(C, u8)] enum → discriminant at byte 0, data at byte 4
//   mem::size_of::<Value>() is evaluated at compile time for riscv32.
//
// Floating-point operations require the RV32F extension (f-registers).

pub(super) use super::specialization::{SpecializationRegistry, SpecializedLayout};
pub(super) use super::trace::{InlineTrace, Operand, SpecializedOpKind, ValueType};
pub(super) use super::{CompiledTrace, Guard, GuardKind, Trace, TraceId, TraceOp};
pub(super) use crate::bytecode::{Function, Value, ValueTag};
pub(super) use crate::jit;
pub(super) use crate::number::LustInt;
pub(super) use crate::Result;
pub(super) use alloc::{boxed::Box, format, string::ToString, vec::Vec};
pub(super) use core::mem;
pub(super) use dynasmrt::{dynasm, riscv::Assembler, DynasmApi, DynasmLabelApi};
use hashbrown::HashMap;

/// Byte offset of the data field within a Value on no_std / riscv32.
/// With `#[repr(C, u8)]` and max alignment 4 (i32/f32/pointer), the single
/// byte discriminant is followed by 3 bytes of padding before the data.
pub(super) const VALUE_DATA_OFFSET: i32 = 4;

/// Minimum stack allocation for a JIT frame (local variables area), in bytes.
/// Must keep (total_frame % 16 == 0) for the RISC-V psABI requirement.
pub(super) const MIN_JIT_STACK_SIZE: i32 = 64;

/// Bytes reserved at the top of the frame for callee-saved registers.
/// We save: ra, s0, s2, s3, s4  = 5 × 4 = 20, rounded up to 32 for alignment.
pub(super) const SAVED_REGS_SIZE: i32 = 32;

/// Byte offset from s0 (frame pointer) for the first specialized slot.
/// s0 points to the original sp (top of frame). Saved registers occupy
/// [s0-4..s0-20]; skip the full SAVED_REGS_SIZE (32) bytes, then a 4-byte
/// pad so each slot stays 4-byte aligned.
pub(super) const SPECIALIZED_BASE_OFFSET: i32 = -(SAVED_REGS_SIZE + 4);

/// Bytes per specialized Vec slot: ptr (4) + len (4) + cap (4) = 12, rounded
/// to 16 for alignment.
pub(super) const SPECIALIZED_SLOT_SIZE: i32 = 16;

mod arithmetic;
mod builder;
mod comparisons;
mod guards;
mod logic;
mod memory;
mod registers;
mod specialization;

/// Tracks a specialised value allocated on the JIT stack.
#[derive(Debug, Clone)]
pub(super) struct SpecializedValue {
    pub stack_offset: i32,
}

pub struct JitCompiler {
    pub(super) ops: Assembler,
    pub(super) leaked_constants: Vec<*const Value>,
    fail_stack: Vec<dynasmrt::DynamicLabel>,
    exit_stack: Vec<dynasmrt::DynamicLabel>,
    pub(super) inline_depth: usize,
    pub(super) specialization_registry: SpecializationRegistry,
    pub(super) specialized_values: HashMap<usize, SpecializedValue>,
    pub(super) next_specialized_id: usize,
}

impl Default for JitCompiler {
    fn default() -> Self {
        Self::new()
    }
}

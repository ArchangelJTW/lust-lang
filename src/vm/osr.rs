//! On-Stack Replacement (OSR) and Deoptimization Engine for Lust Runtime.
//!
//! Provides:
//! - Frame state capture, validation, and serialization from interpreter `CallFrame`.
//! - Graceful fallback to interpreter when frame state contains unsupported constructs (e.g. upvalues, closures).
//! - Native OSR entry compilation and invocation via Cranelift.
//! - Deoptimization exit state reconstruction and application back into interpreter `CallFrame`.
//! - Global atomic OSR lifecycle counters (`entry_attempts`, `entry_successes`, `deopt_count`, `fallback_count`).

extern crate alloc;

#[cfg(feature = "std")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(feature = "std"))]
use core::sync::atomic::{AtomicU64, Ordering};

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::bytecode::Value;
use crate::effect_abi::InterruptReason;
use crate::mir::cranelift_backend::{
    take_deopt_payload, CraneliftJitCompiler, DeoptPayload, DeoptSiteKind, MIR_STATUS_DEOPT,
    MIR_STATUS_ERR_INTERRUPTED,
};
use crate::mir::ir::{BlockId, LocalId, MirFunction, MirType};
use crate::mir::runtime_helpers::{enter_mir_context, MirContext};
use crate::vm::CallFrame;

/// Reasons why OSR entry cannot proceed and must fall back to the interpreter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OsrFallbackReason {
    /// Active open or closed upvalues cannot be mapped into flat JIT registers.
    ActiveUpvalues,
    /// A register contains a dynamic or heap type not supported in flat scalar JIT.
    UnsupportedType(String),
    /// Cranelift failed to compile the OSR entry.
    CompilationFailed(String),
    /// Target basic block is invalid or not found.
    InvalidTargetBlock(usize),
    /// Call frame structure is incompatible with OSR.
    UnsupportedFrame(String),
    /// Requested register index is out of bounds.
    MissingRegister(usize),
}

/// Reasons why native execution deoptimized back to the interpreter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeoptReason {
    /// Safepoint check fired and requested deoptimization.
    Safepoint,
    /// Asynchronous interrupt requested execution bailout.
    Interrupt(InterruptReason),
    /// Dynamic type check failed in speculative code.
    TypeCheckFailed,
    /// Speculative assumption was invalidated.
    SpeculationFailed(String),
    /// Explicit deopt instruction.
    Explicit,
}

/// Global atomic telemetry counters for OSR and deoptimization.
#[derive(Debug, Default)]
pub struct OsrCounters {
    pub entry_attempts: AtomicU64,
    pub entry_successes: AtomicU64,
    pub deopt_count: AtomicU64,
    pub fallback_count: AtomicU64,
}

impl OsrCounters {
    pub const fn new() -> Self {
        Self {
            entry_attempts: AtomicU64::new(0),
            entry_successes: AtomicU64::new(0),
            deopt_count: AtomicU64::new(0),
            fallback_count: AtomicU64::new(0),
        }
    }

    pub fn reset(&self) {
        self.entry_attempts.store(0, Ordering::SeqCst);
        self.entry_successes.store(0, Ordering::SeqCst);
        self.deopt_count.store(0, Ordering::SeqCst);
        self.fallback_count.store(0, Ordering::SeqCst);
    }

    pub fn snapshot(&self) -> OsrCountersSnapshot {
        OsrCountersSnapshot {
            entry_attempts: self.entry_attempts.load(Ordering::SeqCst),
            entry_successes: self.entry_successes.load(Ordering::SeqCst),
            deopt_count: self.deopt_count.load(Ordering::SeqCst),
            fallback_count: self.fallback_count.load(Ordering::SeqCst),
        }
    }
}

/// Snapshot of OSR counters for verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OsrCountersSnapshot {
    pub entry_attempts: u64,
    pub entry_successes: u64,
    pub deopt_count: u64,
    pub fallback_count: u64,
}

/// Global OSR counters instance.
pub static GLOBAL_OSR_COUNTERS: OsrCounters = OsrCounters::new();

/// Captured state of an interpreter call frame at an OSR transition point.
#[derive(Debug, Clone, PartialEq)]
pub struct OsrFrameState {
    pub function_idx: usize,
    pub target_pc: usize,
    pub registers: Vec<(usize, Value)>,
    pub has_upvalues: bool,
}

impl OsrFrameState {
    /// Captures live register state from a `CallFrame` at `target_pc`.
    pub fn from_call_frame(frame: &CallFrame, target_pc: usize, live_indices: &[usize]) -> Self {
        let has_upvalues = !frame.upvalues.is_empty() || !frame.shared_upvalues.is_empty();
        let mut registers = Vec::with_capacity(live_indices.len());
        for &idx in live_indices {
            if idx < frame.registers.len() {
                registers.push((idx, frame.registers[idx].clone()));
            }
        }
        Self {
            function_idx: frame.function_idx,
            target_pc,
            registers,
            has_upvalues,
        }
    }

    /// Validates whether this frame can safely enter JIT OSR code.
    ///
    /// Returns `Err(OsrFallbackReason)` if the frame cannot be compiled,
    /// triggering safe fallback to the interpreter without panicking.
    pub fn validate_for_osr(&self) -> Result<(), OsrFallbackReason> {
        if self.has_upvalues {
            return Err(OsrFallbackReason::ActiveUpvalues);
        }
        for (idx, val) in &self.registers {
            match val {
                Value::Int(_) | Value::Float(_) | Value::Bool(_) | Value::Nil => {}
                Value::Closure(_) => {
                    return Err(OsrFallbackReason::UnsupportedType(format!(
                        "Register r{} has unsupported closure",
                        idx
                    )));
                }
                Value::Task(_) => {
                    return Err(OsrFallbackReason::UnsupportedType(format!(
                        "Register r{} has unsupported task handle",
                        idx
                    )));
                }
                Value::Iterator(_) => {
                    return Err(OsrFallbackReason::UnsupportedType(format!(
                        "Register r{} has unsupported iterator",
                        idx
                    )));
                }
                Value::NativeFunction(_) => {
                    return Err(OsrFallbackReason::UnsupportedType(format!(
                        "Register r{} has unsupported native function",
                        idx
                    )));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Converts captured live locals into raw 64-bit integer values to pass as parameters to Cranelift.
    pub fn to_raw_locals(&self, types: &[(usize, MirType)]) -> Result<Vec<i64>, OsrFallbackReason> {
        let mut raw = Vec::with_capacity(self.registers.len());
        for (reg_idx, val) in &self.registers {
            let mir_ty = types
                .iter()
                .find(|(i, _)| i == reg_idx)
                .map(|(_, ty)| ty)
                .unwrap_or(&MirType::Int);
            let raw_val = match (val, mir_ty) {
                (&Value::Int(i), _) => i,
                (&Value::Float(f), _) => f.to_bits() as i64,
                (&Value::Bool(b), _) => {
                    if b {
                        1
                    } else {
                        0
                    }
                }
                (Value::Nil, _) => 0,
                (other, _) => {
                    return Err(OsrFallbackReason::UnsupportedType(format!(
                        "Cannot convert value {:?} of type {:?} to raw scalar",
                        other, mir_ty
                    )));
                }
            };
            raw.push(raw_val);
        }
        Ok(raw)
    }
}

/// Reconstructed interpreter frame state following a deoptimization event.
#[derive(Debug, Clone, PartialEq)]
pub struct DeoptExitState {
    /// Raw site or basic block identifier captured from native execution/safepoint.
    pub site_id: usize,
    /// Authoritative bytecode instruction pointer to resume execution in interpreter, if proven.
    pub resume_pc: Option<usize>,
    /// Target PC (equals resume_pc if mapped, otherwise site_id for backwards compatibility).
    pub target_pc: usize,
    pub restored_locals: Vec<(usize, Value)>,
    pub reason: DeoptReason,
    pub site_kind: DeoptSiteKind,
}

impl DeoptExitState {
    /// Reconstructs deopt state from a raw `DeoptPayload` emitted by the JIT safepoint helper.
    pub fn from_payload(
        payload: DeoptPayload,
        reason: DeoptReason,
        types: &[(usize, MirType)],
    ) -> Self {
        Self::from_payload_with_mapping(payload.clone(), reason.clone(), types, None).unwrap_or_else(|err| {
            panic!("from_payload called with invalid deopt payload or types: {err}. Use try_from_payload for fallible handling.");
        })
    }

    /// Fallible reconstruction of deopt state with exact type validation.
    pub fn try_from_payload(
        payload: DeoptPayload,
        reason: DeoptReason,
        types: &[(usize, MirType)],
    ) -> Result<Self, String> {
        Self::from_payload_with_mapping(payload, reason, types, None)
    }

    /// Reconstructs deopt state with exact type validation and proven bytecode PC mapping.
    pub fn from_payload_with_mapping(
        payload: DeoptPayload,
        reason: DeoptReason,
        types: &[(usize, MirType)],
        block_to_bytecode_pc: Option<&hashbrown::HashMap<usize, usize>>,
    ) -> Result<Self, String> {
        let site_id = payload.target_pc;
        let site_kind = payload.site_kind;
        let resume_pc = block_to_bytecode_pc.and_then(|m| m.get(&site_id).copied());
        let restored_locals = payload
            .try_to_vm_values_with_types(types)
            .map_err(|e| format!("{e}"))?;

        Ok(Self {
            site_id,
            resume_pc,
            target_pc: resume_pc.unwrap_or(site_id),
            restored_locals,
            reason,
            site_kind,
        })
    }

    /// Validates all restored register indices against `frame` bounds and checks for duplicates.
    /// Does NOT mutate the frame.
    pub fn validate_registers(&self, frame: &CallFrame) -> Result<(), String> {
        let mut seen = hashbrown::HashSet::new();
        for (idx, _) in &self.restored_locals {
            if !seen.insert(*idx) {
                return Err(format!("Duplicate local index {} in deopt restored locals", idx));
            }
            if *idx >= frame.registers.len() {
                return Err(format!(
                    "Deopt restored local index {} out of bounds for frame with {} registers",
                    idx,
                    frame.registers.len()
                ));
            }
        }
        Ok(())
    }

    /// Applies the reconstructed registers and instruction pointer back to the `CallFrame`
    /// with exact complete validation before ANY frame/IP mutation.
    ///
    /// If validation fails, returns `Err` and leaves `frame` completely unmodified.
    pub fn apply_to_call_frame(&self, frame: &mut CallFrame) -> Result<(), String> {
        self.validate_registers(frame)?;
        frame.ip = self.resume_pc.unwrap_or(self.target_pc);
        for (idx, val) in &self.restored_locals {
            frame.registers[*idx] = val.clone();
        }
        Ok(())
    }

    /// Applies reconstructed state to `frame` requiring a proven authoritative bytecode resume PC.
    /// Fails closed (returns Err without mutating `frame`) if `self.resume_pc` is None.
    pub fn try_apply_authoritative(&self, frame: &mut CallFrame) -> Result<(), String> {
        self.validate_registers(frame)?;
        let resume_pc = self.resume_pc.ok_or_else(|| {
            format!(
                "No proven bytecode resume PC mapping for deopt site ID {} ({:?}, raw target_pc {})",
                self.site_id, self.site_kind, self.target_pc
            )
        })?;
        frame.ip = resume_pc;
        for (idx, val) in &self.restored_locals {
            frame.registers[*idx] = val.clone();
        }
        Ok(())
    }
}

/// Result of attempting OSR execution.
#[derive(Debug, Clone, PartialEq)]
pub enum OsrExecutionResult {
    /// Native compiled code executed to completion.
    Completed(Value),
    /// Native compiled code triggered deoptimization back to interpreter.
    Deoptimized(DeoptExitState),
    /// OSR entry could not proceed and gracefully fell back to interpreter.
    Fallback(OsrFallbackReason),
    /// Safepoint delivered an unmasked interrupt.
    Interrupted(InterruptReason),
}

/// Invocates a compiled OSR function pointer with raw 64-bit arguments.
///
/// # Safety
/// The caller must ensure that `fn_ptr` is a valid finalized function pointer
/// and matches the calling convention and argument count.
pub unsafe fn invoke_osr_entry(
    fn_ptr: *const u8,
    raw_args: &[i64],
    return_type: &MirType,
) -> Result<Value, String> {
    match raw_args.len() {
        0 => {
            if *return_type == MirType::Float {
                let f: extern "C" fn() -> f64 = unsafe { core::mem::transmute(fn_ptr) };
                Ok(Value::Float(f()))
            } else {
                let f: extern "C" fn() -> i64 = unsafe { core::mem::transmute(fn_ptr) };
                let res = f();
                if *return_type == MirType::Bool {
                    Ok(Value::Bool(res != 0))
                } else {
                    Ok(Value::Int(res))
                }
            }
        }
        1 => {
            if *return_type == MirType::Float {
                let f: extern "C" fn(i64) -> f64 = unsafe { core::mem::transmute(fn_ptr) };
                Ok(Value::Float(f(raw_args[0])))
            } else {
                let f: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(fn_ptr) };
                let res = f(raw_args[0]);
                if *return_type == MirType::Bool {
                    Ok(Value::Bool(res != 0))
                } else {
                    Ok(Value::Int(res))
                }
            }
        }
        2 => {
            if *return_type == MirType::Float {
                let f: extern "C" fn(i64, i64) -> f64 = unsafe { core::mem::transmute(fn_ptr) };
                Ok(Value::Float(f(raw_args[0], raw_args[1])))
            } else {
                let f: extern "C" fn(i64, i64) -> i64 = unsafe { core::mem::transmute(fn_ptr) };
                let res = f(raw_args[0], raw_args[1]);
                if *return_type == MirType::Bool {
                    Ok(Value::Bool(res != 0))
                } else {
                    Ok(Value::Int(res))
                }
            }
        }
        3 => {
            if *return_type == MirType::Float {
                let f: extern "C" fn(i64, i64, i64) -> f64 = unsafe { core::mem::transmute(fn_ptr) };
                Ok(Value::Float(f(raw_args[0], raw_args[1], raw_args[2])))
            } else {
                let f: extern "C" fn(i64, i64, i64) -> i64 = unsafe { core::mem::transmute(fn_ptr) };
                let res = f(raw_args[0], raw_args[1], raw_args[2]);
                if *return_type == MirType::Bool {
                    Ok(Value::Bool(res != 0))
                } else {
                    Ok(Value::Int(res))
                }
            }
        }
        4 => {
            if *return_type == MirType::Float {
                let f: extern "C" fn(i64, i64, i64, i64) -> f64 = unsafe { core::mem::transmute(fn_ptr) };
                Ok(Value::Float(f(raw_args[0], raw_args[1], raw_args[2], raw_args[3])))
            } else {
                let f: extern "C" fn(i64, i64, i64, i64) -> i64 = unsafe { core::mem::transmute(fn_ptr) };
                let res = f(raw_args[0], raw_args[1], raw_args[2], raw_args[3]);
                if *return_type == MirType::Bool {
                    Ok(Value::Bool(res != 0))
                } else {
                    Ok(Value::Int(res))
                }
            }
        }
        5 => {
            if *return_type == MirType::Float {
                let f: extern "C" fn(i64, i64, i64, i64, i64) -> f64 = unsafe { core::mem::transmute(fn_ptr) };
                Ok(Value::Float(f(
                    raw_args[0], raw_args[1], raw_args[2], raw_args[3], raw_args[4],
                )))
            } else {
                let f: extern "C" fn(i64, i64, i64, i64, i64) -> i64 = unsafe { core::mem::transmute(fn_ptr) };
                let res = f(
                    raw_args[0], raw_args[1], raw_args[2], raw_args[3], raw_args[4],
                );
                if *return_type == MirType::Bool {
                    Ok(Value::Bool(res != 0))
                } else {
                    Ok(Value::Int(res))
                }
            }
        }
        n => Err(format!("Unsupported arity {} for OSR native call", n)),
    }
}

/// Orchestrator for compiling and executing OSR entries.
pub struct OsrEngine {
    compiler: CraneliftJitCompiler,
}

impl OsrEngine {
    /// Creates a new `OsrEngine` with an initialized Cranelift JIT compiler.
    pub fn new() -> Result<Self, String> {
        let compiler = CraneliftJitCompiler::new()?;
        Ok(Self { compiler })
    }

    /// Compiles an OSR entry point for `mir_fn` starting at `target_block`.
    pub fn compile_osr(
        &mut self,
        mir_fn: &MirFunction,
        target_block: BlockId,
        live_locals: &[LocalId],
    ) -> Result<*const u8, String> {
        self.compiler
            .compile_osr_entry(mir_fn, target_block, live_locals)
            .map_err(|e| format!("{e}"))
    }

    /// Executes OSR entry from an interpreter `CallFrame`.
    ///
    /// 1. Validates `frame` for OSR safety. If invalid, increments `fallback_count` and returns `Fallback`.
    /// 2. Compiles OSR entry point via Cranelift.
    /// 3. Increments `entry_attempts`.
    /// 4. Executes native machine code.
    /// 5. If `MIR_STATUS_DEOPT` occurs, extracts deopt payload, reconstructs live locals,
    ///    applies them back to `frame`, increments `deopt_count`, and returns `Deoptimized`.
    /// 6. If execution succeeds, increments `entry_successes` and returns `Completed`.
    pub fn execute_osr(
        &mut self,
        mir_fn: &MirFunction,
        target_block: BlockId,
        live_locals: &[LocalId],
        frame: &mut CallFrame,
    ) -> Result<OsrExecutionResult, String> {
        self.execute_osr_with_mapping(mir_fn, target_block, live_locals, frame, None)
    }

    /// Executes OSR entry from an interpreter `CallFrame` with optional block-to-bytecode mapping.
    pub fn execute_osr_with_mapping(
        &mut self,
        mir_fn: &MirFunction,
        target_block: BlockId,
        live_locals: &[LocalId],
        frame: &mut CallFrame,
        block_to_bytecode_pc: Option<&hashbrown::HashMap<usize, usize>>,
    ) -> Result<OsrExecutionResult, String> {
        GLOBAL_OSR_COUNTERS.entry_attempts.fetch_add(1, Ordering::SeqCst);

        let live_indices: Vec<usize> = live_locals.iter().map(|l| l.0).collect();
        let frame_state = OsrFrameState::from_call_frame(frame, target_block.0, &live_indices);

        if let Err(reason) = frame_state.validate_for_osr() {
            GLOBAL_OSR_COUNTERS.fallback_count.fetch_add(1, Ordering::SeqCst);
            return Ok(OsrExecutionResult::Fallback(reason));
        }

        let types: Vec<(usize, MirType)> = live_locals
            .iter()
            .map(|l| {
                (
                    l.0,
                    mir_fn.locals.get(l.0).map(|ld| ld.ty.clone()).unwrap_or(MirType::Int),
                )
            })
            .collect();

        let raw_args = match frame_state.to_raw_locals(&types) {
            Ok(args) => args,
            Err(reason) => {
                GLOBAL_OSR_COUNTERS.fallback_count.fetch_add(1, Ordering::SeqCst);
                return Ok(OsrExecutionResult::Fallback(reason));
            }
        };

        let fn_ptr = match self.compiler.compile_osr_entry(mir_fn, target_block, live_locals) {
            Ok(ptr) => ptr,
            Err(e) => {
                GLOBAL_OSR_COUNTERS.fallback_count.fetch_add(1, Ordering::SeqCst);
                return Ok(OsrExecutionResult::Fallback(
                    OsrFallbackReason::CompilationFailed(format!("{e}")),
                ));
            }
        };

        // Prepare context and clear any existing deopt payload
        take_deopt_payload();
        let mir_ctx = MirContext::new();
        let _guard = enter_mir_context(&mir_ctx);

        let execution_val = unsafe {
            invoke_osr_entry(fn_ptr, &raw_args, &mir_fn.signature.return_type)
        };

        let last_status = mir_ctx.last_status.load(Ordering::SeqCst);
        if last_status == MIR_STATUS_DEOPT {
            GLOBAL_OSR_COUNTERS.deopt_count.fetch_add(1, Ordering::SeqCst);
            if let Some(sticky) = crate::mir::cranelift_backend::take_current_sticky_error() {
                return Err(format!("Native deopt helper error: {}", sticky));
            }
            if let Some(payload) = take_deopt_payload() {
                let deopt_exit = DeoptExitState::from_payload_with_mapping(
                    payload,
                    DeoptReason::Safepoint,
                    &types,
                    block_to_bytecode_pc,
                )
                .map_err(|e| format!("Deopt payload conversion error: {e}"))?;
                if block_to_bytecode_pc.is_some() {
                    deopt_exit
                        .try_apply_authoritative(frame)
                        .map_err(|e| format!("Deopt authoritative frame application error: {e}"))?;
                } else {
                    deopt_exit
                        .apply_to_call_frame(frame)
                        .map_err(|e| format!("Deopt frame application error: {e}"))?;
                }
                return Ok(OsrExecutionResult::Deoptimized(deopt_exit));
            } else {
                return Err("Deopt status set but no deopt payload was captured".into());
            }
        }

        if last_status == MIR_STATUS_ERR_INTERRUPTED {
            return Ok(OsrExecutionResult::Interrupted(InterruptReason::Cancelled));
        }

        let val = execution_val?;
        GLOBAL_OSR_COUNTERS.entry_successes.fetch_add(1, Ordering::SeqCst);
        Ok(OsrExecutionResult::Completed(val))
    }

    /// Executes OSR entry from an interpreter `CallFrame` requiring a proven authoritative block-to-bytecode mapping.
    ///
    /// When native deoptimization occurs, the reconstructed frame is verified against `block_to_bytecode_pc`.
    /// If mapping is missing/mismatched or payload is invalid, a terminal error is returned without
    /// mutating `frame` and without replay.
    pub fn execute_osr_authoritative(
        &mut self,
        mir_fn: &MirFunction,
        target_block: BlockId,
        live_locals: &[LocalId],
        frame: &mut CallFrame,
        block_to_bytecode_pc: &hashbrown::HashMap<usize, usize>,
    ) -> Result<OsrExecutionResult, String> {
        self.execute_osr_with_mapping(
            mir_fn,
            target_block,
            live_locals,
            frame,
            Some(block_to_bytecode_pc),
        )
    }
}

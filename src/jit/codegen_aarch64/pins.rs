//! Register pinning: keep hot, type-stable scalar VM registers in machine
//! registers for the whole loop body instead of loading and storing them
//! around every op.
//!
//! A VM register is eligible when, within the loop body, every write to it
//! is a typed native op producing one fixed scalar type, nothing writes it
//! through a helper, and its type is proven (by the body's own type guard, a
//! preamble guard, a typed read the recorder already relies on, or a
//! definition before any use). Eligible registers
//! are pinned to callee-saved machine registers — ints and bools in
//! x23..x28, floats in d8..d15 — so helper calls leave them intact.
//!
//! Two classes:
//! - `Carried` registers are read before they are written (loop-carried
//!   state and loop invariants). Their payload bits are loaded once before
//!   `loop_start`, without a type check, and written back — payload only,
//!   never the tag — on trace exit after any inline frames are unwound.
//!   Their type is proven either at entry (preamble guard, or a typed read
//!   the recorder already relies on) or by the body's own type guard, which
//!   is kept and reads the memory tag. Until that guard has passed, the
//!   machine register holds exactly the bits memory holds, so writing them
//!   back is a no-op even when the register is not a scalar at all.
//! - `Local` registers are written before they are read. Every write also
//!   goes to memory (write-through), so memory is always current and an
//!   exit taken before the first write of an iteration sees the right value.
//!
//! Before any op that calls a helper with a pointer into the register array,
//! dirty carried pins are flushed. Inline-call bodies address the callee's
//! own frame, so pinning is suspended inside them.

use super::*;
use crate::jit::trace::Operand;
use hashbrown::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PinClass {
    Carried,
    Local,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Pin {
    pub ty: ValueType,
    /// Machine register number: an X register for Int/Bool, a D register
    /// for Float.
    pub reg: u8,
    pub class: PinClass,
    /// The type is known before the body runs, so in-body type guards on
    /// this register are redundant and skipped. Otherwise the body's own
    /// guard is the proof and is emitted as usual.
    pub proven_at_entry: bool,
}

/// Machine registers available for pinning (all callee-saved).
pub(super) const INT_PIN_REGS: [u8; 6] = [23, 24, 25, 26, 27, 28];
pub(super) const FLOAT_PIN_REGS: [u8; 8] = [8, 9, 10, 11, 12, 13, 14, 15];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Access {
    Read,
    Write,
    Guard(ValueType),
}

#[derive(Default)]
struct Candidate {
    first: Option<Access>,
    /// `Some(ty)` for a typed native write, `None` for a write of unknown
    /// type. Any helper write disqualifies outright.
    write_types: Vec<Option<ValueType>>,
    helper_write: bool,
    guard_types: Vec<ValueType>,
    /// Type of the first typed native read, when there is one.
    read_type: Option<ValueType>,
    accesses: usize,
}

fn scalar(ty: ValueType) -> Option<ValueType> {
    matches!(ty, ValueType::Int | ValueType::Float | ValueType::Bool).then_some(ty)
}

/// Static scalar-type environment, mirroring `update_scalar_registers`.
fn env_update(env: &mut HashMap<u8, ValueType>, op: &TraceOp) {
    let set = |env: &mut HashMap<u8, ValueType>, r: u8, ty: Option<ValueType>| match ty {
        Some(ty) => {
            env.insert(r, ty);
        }
        None => {
            env.remove(&r);
        }
    };
    match op {
        TraceOp::At { .. } => {}
        TraceOp::LoadConst { dest, value } => set(env, *dest, const_type(value)),
        TraceOp::Move { dest, src } => {
            let ty = env.get(src).copied();
            set(env, *dest, ty);
        }
        TraceOp::Add {
            dest,
            lhs_type,
            rhs_type,
            ..
        }
        | TraceOp::Sub {
            dest,
            lhs_type,
            rhs_type,
            ..
        }
        | TraceOp::Mul {
            dest,
            lhs_type,
            rhs_type,
            ..
        }
        | TraceOp::Div {
            dest,
            lhs_type,
            rhs_type,
            ..
        } => set(env, *dest, arith_type(*lhs_type, *rhs_type)),
        TraceOp::Mod {
            dest,
            lhs_type,
            rhs_type,
            ..
        } => set(env, *dest, mod_type(*lhs_type, *rhs_type)),
        TraceOp::Neg { dest, src } => {
            let ty = env.get(src).copied();
            set(env, *dest, ty);
        }
        TraceOp::Eq { dest, .. }
        | TraceOp::Ne { dest, .. }
        | TraceOp::Lt { dest, .. }
        | TraceOp::Le { dest, .. }
        | TraceOp::Gt { dest, .. }
        | TraceOp::Ge { dest, .. }
        | TraceOp::And { dest, .. }
        | TraceOp::Or { dest, .. }
        | TraceOp::Not { dest, .. }
        | TraceOp::IsEnumVariant { dest, .. }
        | TraceOp::TypeIs { dest, .. } => set(env, *dest, Some(ValueType::Bool)),
        TraceOp::ArrayLen { dest, .. } => set(env, *dest, Some(ValueType::Int)),
        TraceOp::Guard {
            register,
            expected_type,
        } => set(env, *register, scalar(*expected_type)),
        TraceOp::ArrayIndexOk {
            value_dest,
            condition_dest,
            ..
        } => {
            env.remove(value_dest);
            env.insert(*condition_dest, ValueType::Bool);
        }
        TraceOp::GetField { dest, .. }
        | TraceOp::Concat { dest, .. }
        | TraceOp::GetIndex { dest, .. }
        | TraceOp::TryGetIndex { dest, .. }
        | TraceOp::CallNative { dest, .. }
        | TraceOp::CallFunction { dest, .. }
        | TraceOp::InlineCall { dest, .. }
        | TraceOp::CallMethod { dest, .. }
        | TraceOp::NewArray { dest, .. }
        | TraceOp::NewStruct { dest, .. }
        | TraceOp::NewEnumUnit { dest, .. }
        | TraceOp::NewEnumVariant { dest, .. }
        | TraceOp::TryCast { dest, .. }
        | TraceOp::GetEnumValue { dest, .. } => {
            env.remove(dest);
        }
        TraceOp::Rebox { dest_reg, .. } => {
            env.remove(dest_reg);
        }
        TraceOp::SpecializedOp { operands, .. } => {
            for operand in operands {
                if let Operand::Register(r) = operand {
                    env.remove(r);
                }
            }
        }
        TraceOp::SetField { .. }
        | TraceOp::GuardNativeFunction { .. }
        | TraceOp::GuardFunction { .. }
        | TraceOp::GuardClosure { .. }
        | TraceOp::GuardLoopContinue { .. }
        | TraceOp::NestedLoopCall { .. }
        | TraceOp::Return { .. }
        | TraceOp::Unbox { .. }
        | TraceOp::DropSpecialized { .. } => {}
    }
}

fn const_type(value: &Value) -> Option<ValueType> {
    match value {
        Value::Bool(_) => Some(ValueType::Bool),
        Value::Int(_) => Some(ValueType::Int),
        Value::Float(_) => Some(ValueType::Float),
        _ => None,
    }
}

fn arith_type(lhs: ValueType, rhs: ValueType) -> Option<ValueType> {
    match (lhs, rhs) {
        (ValueType::Int, ValueType::Int) => Some(ValueType::Int),
        (ValueType::Int | ValueType::Float, ValueType::Int | ValueType::Float) => {
            Some(ValueType::Float)
        }
        _ => None,
    }
}

fn mod_type(lhs: ValueType, rhs: ValueType) -> Option<ValueType> {
    match (lhs, rhs) {
        (ValueType::Int, ValueType::Int) => Some(ValueType::Int),
        (ValueType::Int | ValueType::Float, ValueType::Int | ValueType::Float) => {
            Some(ValueType::Float)
        }
        _ => None,
    }
}

/// Which registers an op reads natively (with the static type it reads
/// them as, when the op annotates one) and which it writes, and how.
struct Effects {
    /// (register, type the op reads it as, if typed)
    reads: Vec<(u8, Option<ValueType>)>,
    /// (register, Some(type) for a typed native write, None for unknown)
    native_writes: Vec<(u8, Option<ValueType>)>,
    helper_writes: Vec<u8>,
    guard: Option<(u8, ValueType)>,
}

fn effects(op: &TraceOp, env: &HashMap<u8, ValueType>) -> Effects {
    let mut e = Effects {
        reads: Vec::new(),
        native_writes: Vec::new(),
        helper_writes: Vec::new(),
        guard: None,
    };
    let typed = |ty: ValueType| scalar(ty);
    let range = |first: u8, count: u8| (first..first.saturating_add(count)).map(|r| (r, None));
    match op {
        TraceOp::At { .. } => {}
        TraceOp::LoadConst { dest, value } => match const_type(value) {
            Some(ty) => e.native_writes.push((*dest, Some(ty))),
            None => e.helper_writes.push(*dest),
        },
        TraceOp::Move { dest, src } => {
            let src_ty = env.get(src).copied();
            e.reads.push((*src, src_ty));
            match src_ty {
                Some(ty) => e.native_writes.push((*dest, Some(ty))),
                None => e.helper_writes.push(*dest),
            }
        }
        TraceOp::Add {
            dest,
            lhs,
            rhs,
            lhs_type,
            rhs_type,
        }
        | TraceOp::Sub {
            dest,
            lhs,
            rhs,
            lhs_type,
            rhs_type,
        }
        | TraceOp::Mul {
            dest,
            lhs,
            rhs,
            lhs_type,
            rhs_type,
        }
        | TraceOp::Div {
            dest,
            lhs,
            rhs,
            lhs_type,
            rhs_type,
        } => {
            e.reads.push((*lhs, typed(*lhs_type)));
            e.reads.push((*rhs, typed(*rhs_type)));
            e.native_writes.push((*dest, arith_type(*lhs_type, *rhs_type)));
        }
        TraceOp::Mod {
            dest,
            lhs,
            rhs,
            lhs_type,
            rhs_type,
        } => {
            e.reads.push((*lhs, typed(*lhs_type)));
            e.reads.push((*rhs, typed(*rhs_type)));
            e.native_writes.push((*dest, mod_type(*lhs_type, *rhs_type)));
        }
        TraceOp::Neg { dest, src } => {
            let ty = env.get(src).copied();
            e.reads.push((*src, ty));
            e.native_writes.push((*dest, ty));
        }
        TraceOp::Lt {
            dest,
            lhs,
            rhs,
            lhs_type,
            rhs_type,
        }
        | TraceOp::Le {
            dest,
            lhs,
            rhs,
            lhs_type,
            rhs_type,
        }
        | TraceOp::Gt {
            dest,
            lhs,
            rhs,
            lhs_type,
            rhs_type,
        }
        | TraceOp::Ge {
            dest,
            lhs,
            rhs,
            lhs_type,
            rhs_type,
        }
        | TraceOp::Eq {
            dest,
            lhs,
            rhs,
            lhs_type,
            rhs_type,
        }
        | TraceOp::Ne {
            dest,
            lhs,
            rhs,
            lhs_type,
            rhs_type,
        } => {
            e.reads.push((*lhs, typed(*lhs_type)));
            e.reads.push((*rhs, typed(*rhs_type)));
            e.native_writes.push((*dest, Some(ValueType::Bool)));
        }
        TraceOp::And { dest, lhs, rhs } | TraceOp::Or { dest, lhs, rhs } => {
            e.reads.push((*lhs, None));
            e.reads.push((*rhs, None));
            e.native_writes.push((*dest, Some(ValueType::Bool)));
        }
        TraceOp::Not { dest, src } => {
            e.reads.push((*src, None));
            e.native_writes.push((*dest, Some(ValueType::Bool)));
        }
        TraceOp::Concat { dest, lhs, rhs } => {
            e.reads.push((*lhs, None));
            e.reads.push((*rhs, None));
            e.helper_writes.push(*dest);
        }
        TraceOp::GetIndex { dest, array, index } | TraceOp::TryGetIndex { dest, array, index } => {
            e.reads.push((*array, None));
            e.reads.push((*index, None));
            e.helper_writes.push(*dest);
        }
        TraceOp::ArrayIndexOk {
            value_dest,
            condition_dest,
            array,
            index,
        } => {
            e.reads.push((*array, None));
            e.reads.push((*index, None));
            e.helper_writes.push(*value_dest);
            e.helper_writes.push(*condition_dest);
        }
        TraceOp::ArrayLen { dest, array } => {
            e.reads.push((*array, None));
            e.native_writes.push((*dest, Some(ValueType::Int)));
        }
        TraceOp::GuardNativeFunction { register, .. }
        | TraceOp::GuardFunction { register, .. }
        | TraceOp::GuardClosure { register, .. } => e.reads.push((*register, None)),
        TraceOp::CallNative {
            dest,
            callee,
            first_arg,
            arg_count,
            ..
        }
        | TraceOp::CallFunction {
            dest,
            callee,
            first_arg,
            arg_count,
            ..
        } => {
            e.reads.push((*callee, None));
            e.reads.extend(range(*first_arg, *arg_count));
            e.helper_writes.push(*dest);
        }
        TraceOp::InlineCall { dest, callee, trace } => {
            e.reads.push((*callee, None));
            e.reads.extend(trace.arg_registers.iter().map(|r| (*r, None)));
            e.helper_writes.push(*dest);
        }
        TraceOp::CallMethod {
            dest,
            object,
            first_arg,
            arg_count,
            ..
        } => {
            e.reads.push((*object, None));
            e.reads.extend(range(*first_arg, *arg_count));
            e.helper_writes.push(*dest);
        }
        TraceOp::GetField { dest, object, .. } => {
            e.reads.push((*object, None));
            e.helper_writes.push(*dest);
        }
        TraceOp::SetField { object, value, .. } => {
            e.reads.push((*object, None));
            e.reads.push((*value, None));
        }
        TraceOp::NewArray {
            dest,
            first_element,
            count,
        } => {
            e.reads.extend(range(*first_element, *count));
            e.helper_writes.push(*dest);
        }
        TraceOp::NewStruct {
            dest,
            field_registers,
            ..
        } => {
            e.reads.extend(field_registers.iter().map(|r| (*r, None)));
            e.helper_writes.push(*dest);
        }
        TraceOp::NewEnumUnit { dest, .. } => e.helper_writes.push(*dest),
        TraceOp::NewEnumVariant {
            dest,
            value_registers,
            ..
        } => {
            e.reads.extend(value_registers.iter().map(|r| (*r, None)));
            e.helper_writes.push(*dest);
        }
        TraceOp::IsEnumVariant { dest, value, .. } | TraceOp::TypeIs { dest, value, .. } => {
            e.reads.push((*value, None));
            e.native_writes.push((*dest, Some(ValueType::Bool)));
        }
        TraceOp::TryCast { dest, value, .. } => {
            e.reads.push((*value, None));
            e.helper_writes.push(*dest);
        }
        TraceOp::GetEnumValue { dest, enum_reg, .. } => {
            e.reads.push((*enum_reg, None));
            e.helper_writes.push(*dest);
        }
        TraceOp::Guard {
            register,
            expected_type,
        } => e.guard = Some((*register, *expected_type)),
        TraceOp::GuardLoopContinue {
            condition_register, ..
        } => e.reads.push((*condition_register, None)),
        TraceOp::NestedLoopCall { .. } | TraceOp::DropSpecialized { .. } => {}
        TraceOp::Return { value } => {
            if let Some(r) = value {
                e.reads.push((*r, None));
            }
        }
        TraceOp::Unbox { source_reg, .. } => e.reads.push((*source_reg, None)),
        TraceOp::Rebox { dest_reg, .. } => e.helper_writes.push(*dest_reg),
        TraceOp::SpecializedOp { operands, .. } => {
            for operand in operands {
                if let Operand::Register(r) = operand {
                    // VecLen writes its register natively; VecPush reads.
                    // Be conservative: treat as a helper write.
                    e.helper_writes.push(*r);
                }
            }
        }
    }
    e
}

/// Does this op call a helper that may look at the VM register array?
/// Dirty carried pins must be flushed to memory before it runs.
pub(super) fn op_touches_register_memory(op: &TraceOp) -> bool {
    !matches!(
        op,
        TraceOp::At { .. }
            | TraceOp::LoadConst {
            value: Value::Int(_) | Value::Float(_) | Value::Bool(_),
            ..
        } | TraceOp::Add { .. }
            | TraceOp::Sub { .. }
            | TraceOp::Mul { .. }
            | TraceOp::Div { .. }
            | TraceOp::Mod { .. }
            | TraceOp::Neg { .. }
            | TraceOp::Lt { .. }
            | TraceOp::Le { .. }
            | TraceOp::Gt { .. }
            | TraceOp::Ge { .. }
            | TraceOp::Eq { .. }
            | TraceOp::Ne { .. }
            | TraceOp::And { .. }
            | TraceOp::Or { .. }
            | TraceOp::Not { .. }
            | TraceOp::Guard { .. }
            | TraceOp::GuardLoopContinue { .. }
            | TraceOp::NestedLoopCall { .. }
            | TraceOp::Return { .. }
    )
}

/// Plan the pins for a trace body. `preamble` establishes the scalar types
/// known at loop entry.
pub(super) fn plan(
    hoisted_constants: &[(u8, Value)],
    preamble: &[TraceOp],
    body: &[TraceOp],
) -> HashMap<u8, Pin> {
    // `LUST_JIT_NOPIN=1` disables pinning, for bisecting and benchmarking.
    #[cfg(feature = "std")]
    if std::env::var_os("LUST_JIT_NOPIN").is_some() {
        return HashMap::new();
    }
    let mut entry_env: HashMap<u8, ValueType> = HashMap::new();
    for (dest, value) in hoisted_constants {
        match const_type(value) {
            Some(ty) => {
                entry_env.insert(*dest, ty);
            }
            None => {
                entry_env.remove(dest);
            }
        }
    }
    for op in preamble {
        env_update(&mut entry_env, op);
    }

    let mut env = entry_env.clone();
    let mut candidates: HashMap<u8, Candidate> = HashMap::new();
    let touch = |c: &mut HashMap<u8, Candidate>, r: u8, access: Access| {
        let cand = c.entry(r).or_default();
        cand.accesses += 1;
        if cand.first.is_none() {
            cand.first = Some(access);
        }
    };
    for op in body {
        let e = effects(op, &env);
        if let Some((r, ty)) = e.guard {
            touch(&mut candidates, r, Access::Guard(ty));
            candidates.get_mut(&r).unwrap().guard_types.push(ty);
        }
        for (r, ty) in e.reads {
            touch(&mut candidates, r, Access::Read);
            let cand = candidates.get_mut(&r).unwrap();
            if cand.read_type.is_none() {
                cand.read_type = ty;
            }
        }
        for (r, ty) in e.native_writes {
            touch(&mut candidates, r, Access::Write);
            candidates.get_mut(&r).unwrap().write_types.push(ty);
        }
        for r in e.helper_writes {
            touch(&mut candidates, r, Access::Write);
            candidates.get_mut(&r).unwrap().helper_write = true;
        }
        env_update(&mut env, op);
    }

    let mut eligible: Vec<(u8, Pin, usize)> = Vec::new();
    for (r, cand) in candidates {
        if cand.helper_write {
            continue;
        }
        if cand.write_types.iter().any(|t| t.is_none()) {
            continue;
        }
        let mut ty: Option<ValueType> = None;
        let mut consistent = true;
        for t in cand.write_types.iter().flatten().chain(cand.guard_types.iter()) {
            match ty {
                None => ty = Some(*t),
                Some(prev) if prev == *t => {}
                Some(_) => consistent = false,
            }
        }
        if !consistent {
            continue;
        }
        let Some(first) = cand.first else { continue };
        let (class, proven_at_entry) = match first {
            Access::Write => (PinClass::Local, true),
            Access::Guard(t) => {
                // The body's guard proves the type each iteration.
                if ty.is_some_and(|ty| ty != t) {
                    continue;
                }
                ty = Some(t);
                (PinClass::Carried, false)
            }
            Access::Read => {
                // The type at entry must already be proven: by the preamble,
                // or by a typed read the recorder itself relies on.
                let entry_ty = entry_env.get(&r).copied().or(cand.read_type);
                match (entry_ty, ty) {
                    (Some(a), Some(b)) if a != b => continue,
                    (None, _) => continue,
                    (Some(a), _) => ty = Some(a),
                }
                (PinClass::Carried, true)
            }
        };
        let Some(ty) = ty.and_then(scalar) else { continue };
        eligible.push((
            r,
            Pin {
                ty,
                reg: 0,
                class,
                proven_at_entry,
            },
            cand.accesses,
        ));
    }

    // Hottest first; ties by register number for determinism.
    eligible.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
    let mut ints = INT_PIN_REGS.iter();
    let mut floats = FLOAT_PIN_REGS.iter();
    let mut pins = HashMap::new();
    for (r, mut pin, _) in eligible {
        let reg = match pin.ty {
            ValueType::Float => floats.next(),
            _ => ints.next(),
        };
        let Some(reg) = reg else { continue };
        pin.reg = *reg;
        pins.insert(r, pin);
    }
    pins
}

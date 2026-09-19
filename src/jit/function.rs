//! Whole-function compilation: turn a loop-free bytecode function into
//! trace IR by walking its instructions statically, every branch included,
//! instead of recording one executed path.
//!
//! Only a subset of the instruction set is handled — constants, moves,
//! typed arithmetic and comparisons, forward branches, calls to statically
//! known bytecode functions, returns. A function using anything else is
//! not compiled (its loops still get their own root traces). Types come
//! from the typed opcodes, the callee signatures and a forward dataflow
//! over the (acyclic) control flow; a `Guard` is emitted wherever a typed
//! instruction reads a register the analysis cannot prove.
//!
//! The result is executed by the backends' function mode: `Label`,
//! `Jump` and `BranchIf` are real control flow, `CallDirect` pushes a
//! frame and calls the callee's compiled code natively (or exits to the
//! interpreter at the call when there is none), and `Return` returns.

use super::trace::{Trace, TraceOp, ValueType};
use crate::bytecode::{Function, Instruction, Register, Value};
use crate::number::NumericType;
use alloc::vec::Vec;
use hashbrown::HashMap;

/// What the translator needs to know about a bytecode function: its
/// scalar parameter kinds and scalar return kind, when declared.
#[derive(Debug, Clone, Default)]
pub struct FunctionSig {
    pub params: Vec<Option<ValueType>>,
    pub ret: Option<ValueType>,
    /// Lua-compat functions take the general call path.
    pub lua_function: bool,
    /// Registers the function's frame uses.
    pub register_count: u8,
}

/// Largest function (in instructions) worth compiling whole.
pub const MAX_FUNCTION_LENGTH: usize = 512;

#[derive(Clone, Default)]
struct Env {
    scalars: HashMap<Register, ValueType>,
    /// Registers known to hold a given bytecode function (loaded from a
    /// constant), so a call through them needs no identity guard.
    functions: HashMap<Register, usize>,
}

impl Env {
    fn write(&mut self, reg: Register, ty: Option<ValueType>) {
        match ty {
            Some(ty) => {
                self.scalars.insert(reg, ty);
            }
            None => {
                self.scalars.remove(&reg);
            }
        }
        self.functions.remove(&reg);
    }

    /// Facts both environments agree on.
    fn merge(&self, other: &Env) -> Env {
        Env {
            scalars: self
                .scalars
                .iter()
                .filter(|(reg, ty)| other.scalars.get(*reg) == Some(*ty))
                .map(|(reg, ty)| (*reg, *ty))
                .collect(),
            functions: self
                .functions
                .iter()
                .filter(|(reg, idx)| other.functions.get(*reg) == Some(*idx))
                .map(|(reg, idx)| (*reg, *idx))
                .collect(),
        }
    }

    fn label_scalars(&self) -> Vec<(Register, ValueType)> {
        let mut scalars: Vec<(Register, ValueType)> =
            self.scalars.iter().map(|(r, t)| (*r, *t)).collect();
        scalars.sort_by_key(|(r, _)| *r);
        scalars
    }
}

fn scalar_of(value: &Value) -> Option<ValueType> {
    match value {
        Value::Int(_) => Some(ValueType::Int),
        Value::Float(_) => Some(ValueType::Float),
        Value::Bool(_) => Some(ValueType::Bool),
        _ => None,
    }
}

fn numeric(ty: NumericType) -> ValueType {
    match ty {
        NumericType::Int => ValueType::Int,
        NumericType::Float => ValueType::Float,
    }
}

fn is_scalar(ty: ValueType) -> bool {
    matches!(ty, ValueType::Int | ValueType::Float | ValueType::Bool)
}

fn jump_target(ip: usize, offset: i16) -> Option<usize> {
    let target = ip as i64 + 1 + i64::from(offset);
    usize::try_from(target).ok()
}

struct Translator<'a> {
    function: &'a Function,
    sig: &'a FunctionSig,
    callee_sig: &'a dyn Fn(usize) -> Option<FunctionSig>,
    ops: Vec<TraceOp>,
    env: Env,
    /// Environment arriving at each forward-jump target from its jumps.
    incoming: HashMap<usize, Env>,
    reachable: bool,
    frame_may_own: bool,
}

impl<'a> Translator<'a> {
    fn guard(&mut self, reg: Register, ty: ValueType) {
        if self.env.scalars.get(&reg) != Some(&ty) {
            self.ops.push(TraceOp::Guard {
                register: reg,
                expected_type: ty,
            });
            self.env.scalars.insert(reg, ty);
        }
    }

    fn write(&mut self, reg: Register, ty: Option<ValueType>) {
        if ty.is_none() {
            self.frame_may_own = true;
        }
        self.env.write(reg, ty);
    }

    fn note_jump(&mut self, target: usize) {
        let merged = match self.incoming.get(&target) {
            Some(existing) => existing.merge(&self.env),
            None => self.env.clone(),
        };
        self.incoming.insert(target, merged);
    }

    fn binary(
        &mut self,
        generic: Instruction,
        dest: Register,
        lhs: Register,
        rhs: Register,
        lhs_type: ValueType,
        rhs_type: ValueType,
    ) -> Option<()> {
        let arith = match (lhs_type, rhs_type) {
            (ValueType::Int, ValueType::Int) => ValueType::Int,
            _ => ValueType::Float,
        };
        let (op, result) = match generic {
            Instruction::Add(..) => (
                TraceOp::Add {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                },
                arith,
            ),
            Instruction::Sub(..) => (
                TraceOp::Sub {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                },
                arith,
            ),
            Instruction::Mul(..) => (
                TraceOp::Mul {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                },
                arith,
            ),
            Instruction::Div(..) => (
                TraceOp::Div {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                },
                arith,
            ),
            Instruction::Mod(..) => {
                if lhs_type != rhs_type {
                    return None;
                }
                (
                    TraceOp::Mod {
                        dest,
                        lhs,
                        rhs,
                        lhs_type,
                        rhs_type,
                    },
                    arith,
                )
            }
            Instruction::Eq(..) => (
                TraceOp::Eq {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                },
                ValueType::Bool,
            ),
            Instruction::Ne(..) => (
                TraceOp::Ne {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                },
                ValueType::Bool,
            ),
            Instruction::Lt(..) => (
                TraceOp::Lt {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                },
                ValueType::Bool,
            ),
            Instruction::Le(..) => (
                TraceOp::Le {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                },
                ValueType::Bool,
            ),
            Instruction::Gt(..) => (
                TraceOp::Gt {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                },
                ValueType::Bool,
            ),
            Instruction::Ge(..) => (
                TraceOp::Ge {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                },
                ValueType::Bool,
            ),
            _ => return None,
        };
        self.guard(lhs, lhs_type);
        self.guard(rhs, rhs_type);
        self.ops.push(op);
        self.write(dest, Some(result));
        Some(())
    }

    fn instruction(&mut self, ip: usize, instruction: Instruction) -> Option<()> {
        let constants = &self.function.chunk.constants;
        match instruction {
            Instruction::LoadNil(dest) => {
                self.ops.push(TraceOp::LoadConst {
                    dest,
                    value: Value::Nil,
                });
                self.env.write(dest, None);
            }
            Instruction::LoadBool(dest, value) => {
                self.ops.push(TraceOp::LoadConst {
                    dest,
                    value: Value::Bool(value),
                });
                self.write(dest, Some(ValueType::Bool));
            }
            Instruction::LoadConst(dest, index) => {
                let value = constants.get(index as usize)?.clone();
                let ty = scalar_of(&value);
                let function = match &value {
                    Value::Function(idx) => Some(*idx),
                    _ => None,
                };
                if ty.is_none() && function.is_none() {
                    self.frame_may_own = true;
                }
                self.ops.push(TraceOp::LoadConst { dest, value });
                self.env.write(dest, ty);
                if let Some(idx) = function {
                    self.env.functions.insert(dest, idx);
                }
            }
            Instruction::Move(dest, src) => {
                let ty = self.env.scalars.get(&src).copied();
                let function = self.env.functions.get(&src).copied();
                self.ops.push(TraceOp::Move { dest, src });
                self.write(dest, ty);
                if let Some(idx) = function {
                    self.env.functions.insert(dest, idx);
                }
            }
            Instruction::Not(dest, src) => {
                self.ops.push(TraceOp::Not { dest, src });
                self.write(dest, Some(ValueType::Bool));
            }
            Instruction::Jump(offset) => {
                let target = jump_target(ip, offset)?;
                if target <= ip {
                    return None;
                }
                if target != ip + 1 {
                    self.note_jump(target);
                    self.ops.push(TraceOp::Jump { label: target });
                    self.reachable = false;
                }
            }
            Instruction::JumpIf(cond, offset) | Instruction::JumpIfNot(cond, offset) => {
                let target = jump_target(ip, offset)?;
                if target <= ip {
                    return None;
                }
                let expect_truthy = matches!(instruction, Instruction::JumpIf(..));
                if target != ip + 1 {
                    self.note_jump(target);
                    self.ops.push(TraceOp::BranchIf {
                        condition_register: cond,
                        expect_truthy,
                        label: target,
                    });
                }
            }
            Instruction::Call(callee, first_arg, arg_count, dest) => {
                let function_idx = *self.env.functions.get(&callee)?;
                let sig = (self.callee_sig)(function_idx)?;
                if sig.lua_function || sig.params.len() != arg_count as usize {
                    return None;
                }
                for (index, kind) in sig.params.iter().enumerate() {
                    let reg = first_arg.wrapping_add(index as u8);
                    if let Some(kind) = kind
                        && is_scalar(*kind)
                    {
                        self.guard(reg, *kind);
                    }
                }
                let result_type = sig.ret.filter(|ty| is_scalar(*ty));
                self.ops.push(TraceOp::CallDirect {
                    dest,
                    callee,
                    function_idx,
                    first_arg,
                    arg_count,
                    callee_registers: sig.register_count,
                    call_ip: ip,
                    result_type,
                });
                self.write(dest, result_type);
            }
            Instruction::Return(value) => {
                if value == 255 {
                    self.ops.push(TraceOp::Return { value: None });
                } else {
                    if let Some(kind) = self.sig.ret
                        && is_scalar(kind)
                    {
                        self.guard(value, kind);
                    }
                    self.ops.push(TraceOp::Return { value: Some(value) });
                }
                self.reachable = false;
            }
            other => {
                if let Some((generic, ty, lhs, rhs)) = other.numeric_specialization() {
                    let ty = numeric(ty);
                    match generic {
                        Instruction::Neg(dest, src) => {
                            self.guard(src, ty);
                            self.ops.push(TraceOp::Neg { dest, src });
                            self.write(dest, Some(ty));
                        }
                        Instruction::Add(dest, ..)
                        | Instruction::Sub(dest, ..)
                        | Instruction::Mul(dest, ..)
                        | Instruction::Div(dest, ..)
                        | Instruction::Mod(dest, ..)
                        | Instruction::Eq(dest, ..)
                        | Instruction::Ne(dest, ..)
                        | Instruction::Lt(dest, ..)
                        | Instruction::Le(dest, ..)
                        | Instruction::Gt(dest, ..)
                        | Instruction::Ge(dest, ..) => {
                            self.binary(generic, dest, lhs, rhs, ty, ty)?;
                        }
                        _ => return None,
                    }
                    return Some(());
                }
                // Untyped arithmetic and comparisons: only when the
                // environment already proves both operand types.
                match other {
                    Instruction::Add(dest, lhs, rhs)
                    | Instruction::Sub(dest, lhs, rhs)
                    | Instruction::Mul(dest, lhs, rhs)
                    | Instruction::Div(dest, lhs, rhs)
                    | Instruction::Mod(dest, lhs, rhs)
                    | Instruction::Eq(dest, lhs, rhs)
                    | Instruction::Ne(dest, lhs, rhs)
                    | Instruction::Lt(dest, lhs, rhs)
                    | Instruction::Le(dest, lhs, rhs)
                    | Instruction::Gt(dest, lhs, rhs)
                    | Instruction::Ge(dest, lhs, rhs) => {
                        let lhs_type = self.env.scalars.get(&lhs).copied()?;
                        let rhs_type = self.env.scalars.get(&rhs).copied()?;
                        let numeric = |ty| matches!(ty, ValueType::Int | ValueType::Float);
                        if !numeric(lhs_type) || !numeric(rhs_type) {
                            return None;
                        }
                        self.binary(other, dest, lhs, rhs, lhs_type, rhs_type)?;
                    }
                    _ => return None,
                }
            }
        }
        Some(())
    }
}

/// Translate `function` into function IR, or `None` when it uses
/// something the function compiler does not handle.
pub fn translate(
    function: &Function,
    function_idx: usize,
    sig: &FunctionSig,
    callee_sig: &dyn Fn(usize) -> Option<FunctionSig>,
) -> Option<Trace> {
    let instructions = &function.chunk.instructions;
    if instructions.is_empty() || instructions.len() > MAX_FUNCTION_LENGTH || sig.lua_function {
        return None;
    }
    if !function.upvalues.is_empty() {
        return None;
    }
    let mut env = Env::default();
    let mut frame_may_own = false;
    for (index, kind) in sig.params.iter().enumerate() {
        match kind {
            Some(kind) if is_scalar(*kind) => {
                env.scalars.insert(index as u8, *kind);
            }
            _ => frame_may_own = true,
        }
    }
    if sig.params.len() != function.param_count as usize {
        return None;
    }
    let mut t = Translator {
        function,
        sig,
        callee_sig,
        ops: Vec::new(),
        env,
        incoming: HashMap::new(),
        reachable: true,
        frame_may_own,
    };
    for (ip, instruction) in instructions.iter().enumerate() {
        if let Some(incoming) = t.incoming.remove(&ip) {
            t.env = if t.reachable {
                incoming.merge(&t.env)
            } else {
                incoming
            };
            t.reachable = true;
            let scalars = t.env.label_scalars();
            t.ops.push(TraceOp::Label { id: ip, scalars });
        }
        if !t.reachable {
            continue;
        }
        t.ops.push(TraceOp::At { ip });
        t.instruction(ip, *instruction)?;
    }
    if t.reachable {
        // Fell off the end without a return.
        return None;
    }
    if !t.incoming.is_empty() {
        // A jump past the last instruction.
        return None;
    }
    Some(Trace {
        function_idx,
        start_ip: 0,
        is_function: true,
        frame_may_own: t.frame_may_own,
        preamble: Vec::new(),
        ops: t.ops,
        postamble: Vec::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::{Function, Instruction};

    fn fib_function() -> Function {
        // The bytecode `fib` compiles to (see benchmarks/suite/fib.lust).
        let mut f = Function::new("fib", 1, false);
        f.set_register_count(8);
        let k2 = f.chunk.add_constant(Value::Int(2));
        let k1 = f.chunk.add_constant(Value::Int(1));
        let kf = f.chunk.add_constant(Value::Function(0));
        for ins in [
            Instruction::LoadConst(1, k2),
            Instruction::LtInt(2, 0, 1),
            Instruction::JumpIfNot(2, 2),
            Instruction::Return(0),
            Instruction::Jump(0),
            Instruction::LoadConst(2, k1),
            Instruction::SubInt(3, 0, 2),
            Instruction::Move(1, 3),
            Instruction::LoadConst(2, kf),
            Instruction::Call(2, 1, 1, 3),
            Instruction::LoadConst(5, k2),
            Instruction::SubInt(6, 0, 5),
            Instruction::Move(4, 6),
            Instruction::LoadConst(5, kf),
            Instruction::Call(5, 4, 1, 6),
            Instruction::AddInt(7, 3, 6),
            Instruction::Return(7),
        ] {
            f.chunk.emit(ins, 1);
        }
        f
    }

    #[test]
    fn fib_translates_with_typed_params_and_no_operand_guards() {
        let f = fib_function();
        let sig = FunctionSig {
            params: vec![Some(ValueType::Int)],
            ret: Some(ValueType::Int),
            lua_function: false,
            register_count: 8,
        };
        let sig_for = |idx: usize| (idx == 0).then(|| sig.clone());
        let trace = translate(&f, 0, &sig, &sig_for).expect("fib is compilable");
        assert!(trace.is_function);
        assert!(!trace.frame_may_own);
        let guards = trace
            .ops
            .iter()
            .filter(|op| matches!(op, TraceOp::Guard { .. }))
            .count();
        assert_eq!(guards, 0, "{:?}", trace.ops);
        assert_eq!(
            trace
                .ops
                .iter()
                .filter(|op| matches!(op, TraceOp::CallDirect { .. }))
                .count(),
            2
        );
        assert!(matches!(
            trace.ops.iter().find(|op| matches!(op, TraceOp::Label { .. })),
            Some(TraceOp::Label { id: 5, .. })
        ));
        assert!(trace.ops.iter().any(|op| matches!(
            op,
            TraceOp::BranchIf {
                condition_register: 2,
                expect_truthy: false,
                label: 5
            }
        )));
    }

    #[test]
    fn backward_jumps_and_unknown_instructions_are_rejected() {
        let mut f = Function::new("loop", 0, false);
        f.set_register_count(2);
        f.chunk.emit(Instruction::LoadNil(0), 1);
        f.chunk.emit(Instruction::Jump(-2), 1);
        let sig = FunctionSig::default();
        assert!(translate(&f, 0, &sig, &|_| None).is_none());

        let mut g = Function::new("concat", 2, false);
        g.set_register_count(3);
        g.chunk.emit(Instruction::Concat(2, 0, 1), 1);
        g.chunk.emit(Instruction::Return(2), 1);
        let sig = FunctionSig {
            params: vec![None, None],
            ret: None,
            lua_function: false,
            register_count: 3,
        };
        assert!(translate(&g, 0, &sig, &|_| None).is_none());
    }
}

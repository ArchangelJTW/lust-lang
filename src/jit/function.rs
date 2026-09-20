//! Whole-function compilation: turn a bytecode function into trace IR by
//! walking its instructions statically, every branch included, instead of
//! recording one executed path.
//!
//! Constants, moves, arithmetic and comparisons, branches and loops,
//! calls to statically known bytecode functions and to struct methods,
//! struct fields, arrays, strings, enums, globals (as version-guarded
//! snapshots) and natives are handled; a function using anything else
//! (closures, upvalues, tuples, maps by index) is not compiled. Types come
//! from the typed opcodes, the declared parameter types, struct layouts
//! and a forward dataflow over the control flow, run to a fixpoint around
//! loops; a `Guard` is emitted wherever a typed instruction reads a
//! register the analysis cannot prove.
//!
//! The result is executed by the backends' function mode: `Label`,
//! `Jump` and `BranchIf` are real control flow, `CallDirect` pushes a
//! frame and calls the callee's compiled code natively (or exits to the
//! interpreter at the call when there is none), and `Return` returns.
//! Ops whose helper can fail (an out-of-range index, a non-array) exit to
//! the interpreter at their instruction, which re-executes it.

use super::trace::{Trace, TraceOp, TracedNativeFn, ValueType};
use crate::ast::TypeKind;
use crate::bytecode::{Function, Instruction, Register, StructLayout, Value};
use crate::number::NumericType;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};

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
    /// The declared parameter types, when the function has a signature.
    pub param_types: Vec<Option<TypeKind>>,
    /// Registers (bit i = register i) the function's bytecode writes; an
    /// argument to a register it never writes can be passed by aliasing.
    pub written_registers: u64,
    /// Registers (bit i = register i) the function returns directly. A
    /// `Return` *moves* its register into the caller's destination
    /// (`jit_return_value`), so such a register must own what it holds and
    /// cannot be an uncounted alias.
    pub returned_registers: u64,
}

impl FunctionSig {
    /// May callee register `index` receive a bitwise copy of the caller's
    /// argument, with no reference-count bump?
    ///
    /// Only when the callee neither writes it nor returns it: an aliased
    /// register is a pure borrow for the whole life of the frame, kept
    /// alive by the caller's register, dropped by nobody, and cloned
    /// rather than moved if the frame is materialized. Both the caller
    /// (which builds `alias_mask`) and the callee (which decides whether
    /// its frame may own anything) must agree, so both ask this.
    pub fn can_alias_param(&self, index: usize) -> bool {
        index < 64
            && self.written_registers & (1 << index) == 0
            && self.returned_registers & (1 << index) == 0
    }
}

/// The registers a function's bytecode writes, as a mask (see
/// `FunctionSig::written_registers`). A closure captured from the frame
/// may write through its cells, so a function creating closures counts as
/// writing everything.
pub fn written_registers(function: &Function) -> u64 {
    let mut mask = 0u64;
    for instruction in &function.chunk.instructions {
        if matches!(instruction, Instruction::Closure(..)) {
            return u64::MAX;
        }
        if let Some(reg) = instruction.defined_register() {
            if reg < 64 {
                mask |= 1 << reg;
            } else {
                return u64::MAX;
            }
        }
    }
    mask
}

/// The registers a function returns directly, as a mask (see
/// `FunctionSig::returned_registers`). `Return(255)` returns Nil and owns
/// nothing; a register outside the mask's range is treated as returned.
pub fn returned_registers(function: &Function) -> u64 {
    let mut mask = 0u64;
    for instruction in &function.chunk.instructions {
        if let Instruction::Return(reg) = instruction {
            if *reg == 255 {
                continue;
            }
            if *reg < 64 {
                mask |= 1 << *reg;
            } else {
                return u64::MAX;
            }
        }
    }
    mask
}

/// What the translator can ask the VM about.
pub struct Context<'a> {
    pub callee_sig: &'a dyn Fn(usize) -> Option<FunctionSig>,
    /// The layout registered for a struct type name.
    pub layout_of: &'a dyn Fn(&str) -> Option<Rc<StructLayout>>,
    /// The index of the bytecode function with this (mangled) name.
    pub function_named: &'a dyn Fn(&str) -> Option<usize>,
    /// The current value of a global.
    pub global: &'a dyn Fn(&str) -> Option<Value>,
    /// `VM::globals_version` now; the snapshots are valid while it holds.
    pub globals_version: u64,
    /// Natives with a native-code equivalent (`JitState::intrinsics`).
    pub intrinsics: &'a HashMap<usize, super::Intrinsic>,
}

/// Largest function (in instructions) worth compiling whole.
pub const MAX_FUNCTION_LENGTH: usize = 512;

/// Passes over a function with loops before the environment must have
/// settled (facts only ever disappear, so this is generous).
const MAX_FIXPOINT_PASSES: usize = 8;

#[derive(Clone, Default)]
struct Env {
    scalars: HashMap<Register, ValueType>,
    /// Registers known to hold a given bytecode function (loaded from a
    /// constant), so a call through them needs no identity guard.
    functions: HashMap<Register, usize>,
    /// Registers holding a struct of a known layout, and whether a
    /// `GuardStructLayout` has established that on this path.
    layouts: HashMap<Register, (Rc<StructLayout>, bool)>,
    /// Registers holding an array whose elements are of a scalar type.
    elements: HashMap<Register, ValueType>,
    /// Registers holding a snapshot of a global (a module table, a native)
    /// taken under the globals guard.
    constants: HashMap<Register, Value>,
    /// Registers whose native function identity a guard has established.
    natives_guarded: HashSet<Register>,
    /// A `GuardGlobals` is in force on this path.
    globals_guarded: bool,
}

fn same_constant(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Map(x), Value::Map(y)) => Rc::ptr_eq(x, y),
        (Value::NativeFunction(x), Value::NativeFunction(y)) => Rc::ptr_eq(x, y),
        _ => false,
    }
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
        self.layouts.remove(&reg);
        self.elements.remove(&reg);
        self.constants.remove(&reg);
        self.natives_guarded.remove(&reg);
    }

    /// Facts both environments agree on.
    fn merge(&self, other: &Env) -> Env {
        Env {
            // Two scalars of different types still hold nothing owned.
            scalars: self
                .scalars
                .iter()
                .filter_map(|(reg, ty)| {
                    let other = other.scalars.get(reg)?;
                    Some((*reg, if other == ty { *ty } else { ValueType::Plain }))
                })
                .collect(),
            functions: self
                .functions
                .iter()
                .filter(|(reg, idx)| other.functions.get(*reg) == Some(*idx))
                .map(|(reg, idx)| (*reg, *idx))
                .collect(),
            layouts: self
                .layouts
                .iter()
                .filter_map(|(reg, (layout, guarded))| {
                    let (other_layout, other_guarded) = other.layouts.get(reg)?;
                    Rc::ptr_eq(layout, other_layout)
                        .then(|| (*reg, (Rc::clone(layout), *guarded && *other_guarded)))
                })
                .collect(),
            elements: self
                .elements
                .iter()
                .filter(|(reg, ty)| other.elements.get(*reg) == Some(*ty))
                .map(|(reg, ty)| (*reg, *ty))
                .collect(),
            constants: self
                .constants
                .iter()
                .filter(|(reg, value)| {
                    other
                        .constants
                        .get(*reg)
                        .is_some_and(|o| same_constant(value, o))
                })
                .map(|(reg, value)| (*reg, value.clone()))
                .collect(),
            natives_guarded: self
                .natives_guarded
                .intersection(&other.natives_guarded)
                .copied()
                .collect(),
            globals_guarded: self.globals_guarded && other.globals_guarded,
        }
    }

    /// Does `self` know everything `other` knows (so a label environment
    /// computed as `other` still holds when arriving with `self`)?
    fn covers(&self, other: &Env) -> bool {
        self.merge(other).same_facts(other)
    }

    fn same_facts(&self, other: &Env) -> bool {
        self.scalars == other.scalars
            && self.functions == other.functions
            && self.layouts.len() == other.layouts.len()
            && self.layouts.iter().all(|(reg, (layout, guarded))| {
                other
                    .layouts
                    .get(reg)
                    .is_some_and(|(o, g)| Rc::ptr_eq(layout, o) && guarded == g)
            })
            && self.elements == other.elements
            && self.constants.len() == other.constants.len()
            && self.constants.iter().all(|(reg, value)| {
                other
                    .constants
                    .get(reg)
                    .is_some_and(|o| same_constant(value, o))
            })
            && self.natives_guarded == other.natives_guarded
            && self.globals_guarded == other.globals_guarded
    }

    /// Give `to` the facts `from` has, then restore `from`'s facts from
    /// `previous` (the environment before the op that established them).
    fn move_facts(&mut self, from: Register, to: Register, previous: &Env) {
        let scalar = self.scalars.get(&from).copied();
        let function = self.functions.get(&from).copied();
        let layout = self.layouts.get(&from).cloned();
        let elements = self.elements.get(&from).copied();
        let constant = self.constants.get(&from).cloned();
        let native_guarded = self.natives_guarded.contains(&from);
        self.write(to, scalar);
        if let Some(idx) = function {
            self.functions.insert(to, idx);
        }
        if let Some(layout) = layout {
            self.layouts.insert(to, layout);
        }
        if let Some(elements) = elements {
            self.elements.insert(to, elements);
        }
        if let Some(constant) = constant {
            self.constants.insert(to, constant);
        }
        if native_guarded {
            self.natives_guarded.insert(to);
        }
        self.write(from, previous.scalars.get(&from).copied());
        if let Some(idx) = previous.functions.get(&from) {
            self.functions.insert(from, *idx);
        }
        if let Some(layout) = previous.layouts.get(&from) {
            self.layouts.insert(from, layout.clone());
        }
        if let Some(elements) = previous.elements.get(&from) {
            self.elements.insert(from, *elements);
        }
        if let Some(constant) = previous.constants.get(&from) {
            self.constants.insert(from, constant.clone());
        }
        if previous.natives_guarded.contains(&from) {
            self.natives_guarded.insert(from);
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

fn scalar_kind(kind: &TypeKind) -> Option<ValueType> {
    match kind {
        TypeKind::Int => Some(ValueType::Int),
        TypeKind::Float => Some(ValueType::Float),
        TypeKind::Bool => Some(ValueType::Bool),
        _ => None,
    }
}

fn jump_target(ip: usize, offset: i16) -> Option<usize> {
    let target = ip as i64 + 1 + i64::from(offset);
    usize::try_from(target).ok()
}

/// Every ip some jump lands on.
fn jump_targets(instructions: &[Instruction]) -> Option<HashSet<usize>> {
    let mut targets = HashSet::new();
    for (ip, instruction) in instructions.iter().enumerate() {
        let offset = match instruction {
            Instruction::Jump(offset)
            | Instruction::JumpIf(_, offset)
            | Instruction::JumpIfNot(_, offset) => *offset,
            _ => continue,
        };
        let target = jump_target(ip, offset)?;
        if target >= instructions.len() {
            return None;
        }
        targets.insert(target);
    }
    Some(targets)
}

struct Translator<'a> {
    function: &'a Function,
    sig: &'a FunctionSig,
    ctx: &'a Context<'a>,
    ops: Vec<TraceOp>,
    env: Env,
    /// Environment arriving at each forward-jump target from the jumps
    /// to it seen so far.
    incoming: HashMap<usize, Env>,
    /// Environment the previous pass settled on for each loop header.
    loop_envs: &'a HashMap<usize, Env>,
    /// Back-edge environments seen in this pass.
    back_edges: HashMap<usize, Env>,
    targets: &'a HashSet<usize>,
    reachable: bool,
    frame_may_own: bool,
    /// The environment before the last instruction translated, and
    /// whether that instruction is the last op's only producer (nothing
    /// else — a label, a marker — came between).
    previous_env: Env,
    previous_is_last_op: bool,
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

    /// The register now holds a value of this declared type.
    fn write_typed(&mut self, reg: Register, kind: &TypeKind) {
        self.write(reg, scalar_kind(kind));
        self.note_type(reg, kind);
    }

    /// Non-scalar facts a declared type gives about a register.
    fn note_type(&mut self, reg: Register, kind: &TypeKind) {
        match kind {
            TypeKind::Named(name) => {
                if let Some(layout) = (self.ctx.layout_of)(name) {
                    self.env.layouts.insert(reg, (layout, false));
                }
            }
            TypeKind::Array(element) => {
                if let Some(ty) = scalar_kind(&element.kind) {
                    self.env.elements.insert(reg, ty);
                }
            }
            _ => {}
        }
    }

    fn constant_string(&self, index: u16) -> Option<String> {
        self.function
            .chunk
            .constants
            .get(index as usize)?
            .as_string()
            .map(|s| s.to_string())
    }

    /// The layout a register's struct is known to have, guarded on this
    /// path from here on.
    fn guarded_layout(&mut self, reg: Register) -> Option<Rc<StructLayout>> {
        let (layout, guarded) = self.env.layouts.get(&reg)?.clone();
        if !guarded {
            self.ops.push(TraceOp::GuardStructLayout {
                register: reg,
                layout: Rc::as_ptr(&layout) as *const (),
            });
            self.env.layouts.insert(reg, (Rc::clone(&layout), true));
        }
        Some(layout)
    }

    fn note_jump(&mut self, target: usize) {
        let merged = match self.incoming.get(&target) {
            Some(existing) => existing.merge(&self.env),
            None => self.env.clone(),
        };
        self.incoming.insert(target, merged);
    }

    fn note_back_edge(&mut self, target: usize) {
        let merged = match self.back_edges.get(&target) {
            Some(existing) => existing.merge(&self.env),
            None => self.env.clone(),
        };
        self.back_edges.insert(target, merged);
    }

    /// Something the analysis cannot see is about to run: the globals may
    /// change, so the next global read needs a fresh guard.
    fn opaque_call(&mut self) {
        self.env.globals_guarded = false;
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

    /// A call to a bytecode function whose index is known, with the
    /// arguments in `first_arg..` and an optional receiver before them.
    #[allow(clippy::too_many_arguments)]
    fn call_direct(
        &mut self,
        ip: usize,
        function_idx: usize,
        callee: Register,
        receiver: Option<Register>,
        first_arg: Register,
        arg_count: u8,
        dest: Register,
    ) -> Option<()> {
        let sig = (self.ctx.callee_sig)(function_idx)?;
        let total = arg_count as usize + usize::from(receiver.is_some());
        if sig.lua_function || sig.params.len() != total || sig.register_count == 0 {
            return None;
        }
        let mut sources = Vec::with_capacity(total);
        sources.extend(receiver);
        for index in 0..arg_count {
            sources.push(first_arg.wrapping_add(index));
        }
        let mut alias_mask = 0u64;
        for (index, (reg, kind)) in sources.iter().zip(sig.params.iter()).enumerate() {
            if let Some(kind) = kind
                && is_scalar(*kind)
            {
                self.guard(*reg, *kind);
            } else if sig.can_alias_param(index)
                && !self.env.scalars.get(reg).is_some_and(|ty| is_scalar(*ty))
            {
                // The callee only reads this parameter, and never returns
                // it: pass the caller's value by aliasing rather than
                // cloning and dropping it.
                alias_mask |= 1 << index;
            }
        }
        let result_type = sig.ret.filter(|ty| is_scalar(*ty));
        self.ops.push(TraceOp::CallDirect {
            dest,
            callee,
            receiver,
            function_idx,
            first_arg,
            arg_count,
            callee_registers: sig.register_count,
            call_ip: ip,
            resume_ip: ip + 1,
            alias_mask,
            result_type,
        });
        self.opaque_call();
        self.write(dest, result_type);
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
                let plain = matches!(value, Value::Function(_) | Value::Nil);
                if ty.is_none() && !plain {
                    self.frame_may_own = true;
                }
                // A function constant the register is already proven to
                // hold (reloaded by the bytecode on every iteration of a
                // loop making the same call) need not be stored again.
                if let Some(idx) = function
                    && self.env.functions.get(&dest) == Some(&idx)
                {
                    return Some(());
                }
                self.ops.push(TraceOp::LoadConst { dest, value });
                // A function index or Nil is plain: nothing to drop when
                // the register is written again.
                self.env
                    .write(dest, ty.or(plain.then_some(ValueType::Plain)));
                if let Some(idx) = function {
                    self.env.functions.insert(dest, idx);
                }
            }
            Instruction::LoadGlobal(dest, name_idx) => {
                let name = self.constant_string(name_idx)?;
                let value = (self.ctx.global)(&name)?;
                if !self.env.globals_guarded {
                    self.ops.push(TraceOp::GuardGlobals {
                        version: self.ctx.globals_version,
                    });
                    self.env.globals_guarded = true;
                }
                let ty = scalar_of(&value);
                self.ops.push(TraceOp::LoadConst {
                    dest,
                    value: value.clone(),
                });
                self.write(dest, ty);
                match &value {
                    Value::Function(idx) => {
                        self.env.functions.insert(dest, *idx);
                    }
                    Value::Map(_) | Value::NativeFunction(_) => {
                        self.env.constants.insert(dest, value);
                    }
                    _ => {}
                }
            }
            Instruction::Move(dest, src) => {
                // A temporary moved into a local right after being
                // produced: let the producer write the local and skip the
                // copy (a clone and a drop for an owned value).
                if dest != src
                    && self.previous_is_last_op
                    && self
                        .ops
                        .iter()
                        .rev()
                        .find(|op| !matches!(op, TraceOp::At { .. }))
                        .and_then(|op| op.single_dest())
                        == Some(src)
                    && super::trace::register_dead_after(self.function, ip + 1, src)
                {
                    let previous = self.previous_env.clone();
                    if let Some(op) = self
                        .ops
                        .iter_mut()
                        .rev()
                        .find(|op| !matches!(op, TraceOp::At { .. }))
                    {
                        op.set_dest(dest);
                        // The elided move must not run again after an exit
                        // inside the callee.
                        if let TraceOp::CallDirect { resume_ip, .. } = op {
                            *resume_ip = ip + 1;
                        }
                    }
                    if self.env.scalars.get(&src).is_none() {
                        self.frame_may_own = true;
                    }
                    self.env.move_facts(src, dest, &previous);
                    return Some(());
                }
                let ty = self.env.scalars.get(&src).copied();
                let function = self.env.functions.get(&src).copied();
                let layout = self.env.layouts.get(&src).cloned();
                let elements = self.env.elements.get(&src).copied();
                let constant = self.env.constants.get(&src).cloned();
                let native_guarded = self.env.natives_guarded.contains(&src);
                self.ops.push(TraceOp::Move { dest, src });
                self.write(dest, ty);
                if let Some(idx) = function {
                    self.env.functions.insert(dest, idx);
                }
                if let Some(layout) = layout {
                    self.env.layouts.insert(dest, layout);
                }
                if let Some(elements) = elements {
                    self.env.elements.insert(dest, elements);
                }
                if let Some(constant) = constant {
                    self.env.constants.insert(dest, constant);
                }
                if native_guarded {
                    self.env.natives_guarded.insert(dest);
                }
            }
            Instruction::Not(dest, src) => {
                self.ops.push(TraceOp::Not { dest, src });
                self.write(dest, Some(ValueType::Bool));
            }
            Instruction::And(dest, lhs, rhs) => {
                self.ops.push(TraceOp::And { dest, lhs, rhs });
                self.write(dest, None);
            }
            Instruction::Or(dest, lhs, rhs) => {
                self.ops.push(TraceOp::Or { dest, lhs, rhs });
                self.write(dest, None);
            }
            Instruction::Jump(offset) => {
                let target = jump_target(ip, offset)?;
                if target <= ip {
                    self.note_back_edge(target);
                    self.ops.push(TraceOp::Jump { label: target });
                    self.reachable = false;
                } else if target != ip + 1 {
                    self.note_jump(target);
                    self.ops.push(TraceOp::Jump { label: target });
                    self.reachable = false;
                }
            }
            Instruction::JumpIf(cond, offset) | Instruction::JumpIfNot(cond, offset) => {
                let target = jump_target(ip, offset)?;
                let expect_truthy = matches!(instruction, Instruction::JumpIf(..));
                if target <= ip {
                    self.note_back_edge(target);
                } else if target != ip + 1 {
                    self.note_jump(target);
                } else {
                    return Some(());
                }
                self.ops.push(TraceOp::BranchIf {
                    condition_register: cond,
                    expect_truthy,
                    label: target,
                });
            }
            Instruction::Call(callee, first_arg, arg_count, dest) => {
                if let Some(function_idx) = self.env.functions.get(&callee).copied() {
                    return self.call_direct(
                        ip,
                        function_idx,
                        callee,
                        None,
                        first_arg,
                        arg_count,
                        dest,
                    );
                }
                if let Some(Value::NativeFunction(native)) = self.env.constants.get(&callee) {
                    // A native from a global (or a module table): its
                    // identity is guarded, since the table is mutable.
                    let traced = TracedNativeFn::new(Rc::clone(native));
                    if !self.env.natives_guarded.contains(&callee) {
                        self.ops.push(TraceOp::GuardNativeFunction {
                            register: callee,
                            function: traced.clone(),
                        });
                        self.env.natives_guarded.insert(callee);
                    }
                    let intrinsic = self
                        .ctx
                        .intrinsics
                        .get(&(Rc::as_ptr(native) as *const () as usize))
                        .copied();
                    if intrinsic == Some(super::Intrinsic::ArrayLen) && arg_count == 1 {
                        // `array.len(a)`, with the native's identity guarded
                        // above: the length read inline.
                        self.ops.push(TraceOp::ArrayLen {
                            dest,
                            array: first_arg,
                        });
                        self.write(dest, Some(ValueType::Int));
                        return Some(());
                    }
                    self.ops.push(TraceOp::CallNative {
                        dest,
                        callee,
                        function: traced,
                        first_arg,
                        arg_count,
                    });
                } else {
                    // A function value of unknown identity (a parameter, a
                    // closure): the runtime calls it.
                    self.ops.push(TraceOp::CallFunction {
                        dest,
                        callee,
                        function_idx: 0,
                        first_arg,
                        arg_count,
                        is_closure: false,
                        upvalues_ptr: None,
                    });
                }
                self.opaque_call();
                self.write(dest, None);
            }
            Instruction::CallMethod(object, name_idx, first_arg, arg_count, dest) => {
                let method = self.constant_string(name_idx)?;
                let layout = self.guarded_layout(object)?;
                let mangled = alloc::format!("{}:{}", layout.name(), method);
                let function_idx = (self.ctx.function_named)(&mangled)?;
                return self.call_direct(
                    ip,
                    function_idx,
                    object,
                    Some(object),
                    first_arg,
                    arg_count,
                    dest,
                );
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
            Instruction::GetField(dest, object, name_idx) => {
                let field_name = self.constant_string(name_idx)?;
                if let Some(Value::Map(map)) = self.env.constants.get(&object) {
                    // A module table: the entry read now is what a call
                    // through the result will guard on.
                    let entry = map
                        .borrow()
                        .get(&crate::bytecode::ValueKey::from(field_name.as_str()))
                        .cloned();
                    self.ops.push(TraceOp::GetField {
                        dest,
                        object,
                        field_name,
                        field_index: None,
                        value_type: None,
                        is_weak: false,
                    });
                    self.write(dest, None);
                    if let Some(entry @ Value::NativeFunction(_)) = entry {
                        self.env.constants.insert(dest, entry);
                    }
                    return Some(());
                }
                match self.guarded_layout(object) {
                    Some(layout) => {
                        let index = layout.index_of_str(&field_name)?;
                        let kind = layout.field_type(index).kind.clone();
                        let is_weak = layout.is_weak(index);
                        let value_type = scalar_kind(&kind);
                        self.ops.push(TraceOp::GetField {
                            dest,
                            object,
                            field_name,
                            field_index: Some(index),
                            value_type,
                            is_weak,
                        });
                        self.write_typed(dest, &kind);
                    }
                    None => {
                        self.ops.push(TraceOp::GetField {
                            dest,
                            object,
                            field_name,
                            field_index: None,
                            value_type: None,
                            is_weak: false,
                        });
                        self.write(dest, None);
                    }
                }
            }
            Instruction::SetField(object, name_idx, value) => {
                let field_name = self.constant_string(name_idx)?;
                let value_type = self.env.scalars.get(&value).copied();
                let (field_index, is_weak) = match self.guarded_layout(object) {
                    Some(layout) => {
                        let index = layout.index_of_str(&field_name)?;
                        (Some(index), layout.is_weak(index))
                    }
                    None => (None, false),
                };
                self.ops.push(TraceOp::SetField {
                    object,
                    field_name,
                    value,
                    field_index,
                    value_type,
                    is_weak,
                });
            }
            Instruction::ArrayLen(dest, array) => {
                self.ops.push(TraceOp::ArrayLen { dest, array });
                self.write(dest, Some(ValueType::Int));
            }
            Instruction::GetIndex(dest, array, index) => {
                self.guard(index, ValueType::Int);
                self.ops.push(TraceOp::GetIndex { dest, array, index });
                let element = self.env.elements.get(&array).copied();
                self.write(dest, element);
            }
            Instruction::TryGetIndex(dest, array, index) => {
                // The checked read is only compiled in its matched form,
                // `if a[i] is Ok(v)`: TryGetIndex, IsEnumVariant, JumpIfNot,
                // GetEnumValue, which becomes one ArrayIndexOk plus the
                // branch and the binding move.
                let instructions = &self.function.chunk.instructions;
                let (
                    Some(Instruction::IsEnumVariant(cond, tested, enum_idx, variant_idx)),
                    Some(Instruction::JumpIfNot(branch_cond, offset)),
                    Some(Instruction::GetEnumValue(binding, enum_reg, 0)),
                ) = (
                    instructions.get(ip + 1),
                    instructions.get(ip + 2),
                    instructions.get(ip + 3),
                )
                else {
                    return None;
                };
                let (cond, binding, offset) = (*cond, *binding, *offset);
                if *tested != dest
                    || *branch_cond != cond
                    || *enum_reg != dest
                    || self.constant_string(*enum_idx)? != "Result"
                    || self.constant_string(*variant_idx)? != "Ok"
                    || (ip + 1..=ip + 3).any(|at| self.targets.contains(&at))
                {
                    return None;
                }
                let target = jump_target(ip + 2, offset)?;
                if target <= ip + 2 {
                    return None;
                }
                let value_type = self.env.elements.get(&array).copied();
                self.guard(index, ValueType::Int);
                self.ops.push(TraceOp::ArrayIndexOk {
                    value_dest: dest,
                    condition_dest: cond,
                    array,
                    index,
                    value_type,
                });
                self.write(dest, value_type);
                self.write(cond, Some(ValueType::Bool));
                self.ops.push(TraceOp::At { ip: ip + 2 });
                self.note_jump(target);
                self.ops.push(TraceOp::BranchIf {
                    condition_register: cond,
                    expect_truthy: false,
                    label: target,
                });
                self.ops.push(TraceOp::At { ip: ip + 3 });
                self.ops.push(TraceOp::Move {
                    dest: binding,
                    src: dest,
                });
                self.write(binding, value_type);
            }
            Instruction::NewArray(dest, first_element, count) => {
                self.ops.push(TraceOp::NewArray {
                    dest,
                    first_element,
                    count,
                });
                self.write(dest, None);
                if let Some(kind) = self.function.register_types.get(&dest).cloned() {
                    self.note_type(dest, &kind);
                }
            }
            Instruction::Concat(dest, lhs, rhs) => {
                self.ops.push(TraceOp::Concat { dest, lhs, rhs });
                self.write(dest, None);
            }
            Instruction::NewStruct(dest, name_idx, first_field_name_idx, first_field, field_count) => {
                let struct_name = self.constant_string(name_idx)?;
                let mut field_names = Vec::with_capacity(field_count as usize);
                let mut field_registers = Vec::with_capacity(field_count as usize);
                for i in 0..field_count {
                    field_names.push(self.constant_string(first_field_name_idx + u16::from(i))?);
                    field_registers.push(first_field + i);
                }
                self.ops.push(TraceOp::NewStruct {
                    dest,
                    struct_name: struct_name.clone(),
                    field_names,
                    field_registers,
                });
                self.write(dest, None);
                if let Some(layout) = (self.ctx.layout_of)(&struct_name) {
                    self.env.layouts.insert(dest, (layout, false));
                }
            }
            Instruction::NewEnumUnit(dest, enum_idx, variant_idx) => {
                self.ops.push(TraceOp::NewEnumUnit {
                    dest,
                    enum_name: self.constant_string(enum_idx)?,
                    variant_name: self.constant_string(variant_idx)?,
                });
                self.write(dest, None);
            }
            Instruction::NewEnumVariant(dest, enum_idx, variant_idx, first_value, value_count) => {
                let value_registers = (0..value_count).map(|i| first_value + i).collect();
                self.ops.push(TraceOp::NewEnumVariant {
                    dest,
                    enum_name: self.constant_string(enum_idx)?,
                    variant_name: self.constant_string(variant_idx)?,
                    value_registers,
                });
                self.write(dest, None);
            }
            Instruction::IsEnumVariant(dest, value, enum_idx, variant_idx) => {
                self.ops.push(TraceOp::IsEnumVariant {
                    dest,
                    value,
                    enum_name: self.constant_string(enum_idx)?,
                    variant_name: self.constant_string(variant_idx)?,
                });
                self.write(dest, Some(ValueType::Bool));
            }
            Instruction::GetEnumValue(dest, enum_reg, index) => {
                self.ops.push(TraceOp::GetEnumValue {
                    dest,
                    enum_reg,
                    index,
                });
                self.write(dest, None);
            }
            Instruction::TypeIs(dest, value, type_idx) => {
                self.ops.push(TraceOp::TypeIs {
                    dest,
                    value,
                    type_name: self.constant_string(type_idx)?,
                });
                self.write(dest, Some(ValueType::Bool));
            }
            Instruction::TryCast(dest, value, type_idx) => {
                self.ops.push(TraceOp::TryCast {
                    dest,
                    value,
                    type_name: self.constant_string(type_idx)?,
                });
                self.write(dest, None);
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
                    Instruction::Neg(dest, src) => {
                        let ty = self.env.scalars.get(&src).copied()?;
                        if !matches!(ty, ValueType::Int | ValueType::Float) {
                            return None;
                        }
                        self.ops.push(TraceOp::Neg { dest, src });
                        self.write(dest, Some(ty));
                    }
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

/// The environment a function starts with: its parameters' declared
/// types, and the frame's other registers Nil.
fn entry_env(sig: &FunctionSig, ctx: &Context) -> (Env, bool) {
    let mut env = Env::default();
    let mut frame_may_own = false;
    for (index, kind) in sig.params.iter().enumerate() {
        match kind {
            Some(kind) if is_scalar(*kind) => {
                env.scalars.insert(index as u8, *kind);
            }
            // A parameter the function never writes and never returns owns
            // nothing in a native frame: a compiled caller aliases a
            // non-scalar argument to it and copies a scalar one (the
            // interpreter drops its own frames). Any other register may
            // come to own something — including a parameter that is
            // returned, which is cloned in and moved out.
            _ if sig.can_alias_param(index) => {}
            _ => frame_may_own = true,
        }
        match sig.param_types.get(index) {
            Some(Some(TypeKind::Named(name))) => {
                if let Some(layout) = (ctx.layout_of)(name) {
                    env.layouts.insert(index as u8, (layout, false));
                }
            }
            Some(Some(TypeKind::Array(element))) => {
                if let Some(ty) = scalar_kind(&element.kind) {
                    env.elements.insert(index as u8, ty);
                }
            }
            _ => {}
        }
    }
    // Every other register is `Nil` on entry: the interpreter resets a
    // frame before it copies the arguments in, and a native caller blanks
    // the callee's frame the same way. The first store into such a register
    // has nothing to drop.
    for reg in sig.params.len()..usize::from(sig.register_count) {
        env.scalars.insert(reg as u8, ValueType::Plain);
    }
    (env, frame_may_own)
}

struct Pass {
    ops: Vec<TraceOp>,
    /// The environment each label was compiled with.
    label_envs: HashMap<usize, Env>,
    /// The environment each back edge arrived with.
    back_edges: HashMap<usize, Env>,
    frame_may_own: bool,
    entry_scalars: Vec<(Register, ValueType)>,
}

/// One translation pass with the given loop-header environments, or
/// `None` when the function uses something the function compiler does
/// not handle.
fn translate_pass(
    function: &Function,
    sig: &FunctionSig,
    ctx: &Context,
    targets: &HashSet<usize>,
    loop_envs: &HashMap<usize, Env>,
) -> Option<Pass> {
    let instructions = &function.chunk.instructions;
    let (env, frame_may_own) = entry_env(sig, ctx);
    let entry_scalars = env.label_scalars();
    let mut t = Translator {
        function,
        sig,
        ctx,
        ops: Vec::new(),
        env,
        incoming: HashMap::new(),
        loop_envs,
        back_edges: HashMap::new(),
        targets,
        reachable: true,
        frame_may_own,
        previous_env: Env::default(),
        previous_is_last_op: false,
    };
    let mut label_envs: HashMap<usize, Env> = HashMap::new();
    let mut ip = 0;
    while ip < instructions.len() {
        if targets.contains(&ip) {
            let forward = t.incoming.remove(&ip);
            let mut env = match (forward, t.reachable) {
                (Some(incoming), true) => incoming.merge(&t.env),
                (Some(incoming), false) => incoming,
                (None, true) => t.env.clone(),
                // Only reachable through a back edge: not structured code.
                (None, false) => return None,
            };
            if let Some(loop_env) = t.loop_envs.get(&ip) {
                env = env.merge(loop_env);
            }
            t.env = env;
            t.reachable = true;
            label_envs.insert(ip, t.env.clone());
            let scalars = t.env.label_scalars();
            t.ops.push(TraceOp::Label { id: ip, scalars });
        }
        if !t.reachable {
            ip += 1;
            continue;
        }
        t.ops.push(TraceOp::At { ip });
        let ops_before = t.ops.len();
        let instruction = instructions[ip];
        let before = t.env.clone();
        t.instruction(ip, instruction)?;
        t.previous_env = before;
        // Only a straight-line predecessor's result may be redirected: the
        // last op must be this instruction's, and no label may come
        // between (another path would arrive there).
        t.previous_is_last_op = t.ops.len() > ops_before && !targets.contains(&(ip + 1));
        // The matched checked read consumed three more instructions.
        if matches!(instruction, Instruction::TryGetIndex(..)) {
            ip += 3;
        }
        ip += 1;
    }
    if t.reachable {
        // Fell off the end without a return.
        return None;
    }
    if !t.incoming.is_empty() {
        // A jump past the last instruction.
        return None;
    }
    Some(Pass {
        ops: t.ops,
        label_envs,
        back_edges: t.back_edges,
        frame_may_own: t.frame_may_own,
        entry_scalars,
    })
}

/// Translate `function` into function IR, or `None` when it uses
/// something the function compiler does not handle.
pub fn translate(
    function: &Function,
    function_idx: usize,
    sig: &FunctionSig,
    ctx: &Context,
) -> Option<Trace> {
    let instructions = &function.chunk.instructions;
    if instructions.is_empty() || instructions.len() > MAX_FUNCTION_LENGTH || sig.lua_function {
        return None;
    }
    if !function.upvalues.is_empty() {
        return None;
    }
    if sig.params.len() != function.param_count as usize {
        return None;
    }
    let targets = jump_targets(instructions)?;
    // Loop headers start with everything the entry path knows; each pass
    // narrows them to what the back edges also guarantee, until a pass
    // finds every back edge covered.
    let mut loop_envs: HashMap<usize, Env> = HashMap::new();
    for _ in 0..MAX_FIXPOINT_PASSES {
        let pass = translate_pass(function, sig, ctx, &targets, &loop_envs)?;
        let mut stable = true;
        for (target, env) in &pass.back_edges {
            let label_env = pass.label_envs.get(target)?;
            if !env.covers(label_env) {
                stable = false;
                loop_envs.insert(*target, label_env.merge(env));
            }
        }
        if stable {
            return Some(Trace {
                function_idx,
                start_ip: 0,
                is_function: true,
                frame_may_own: pass.frame_may_own,
                entry_scalars: pass.entry_scalars,
                alias_params: (0..sig.params.len().min(64))
                    .filter(|index| sig.can_alias_param(*index))
                    .fold(0, |mask, index| mask | (1u64 << index)),
                preamble: Vec::new(),
                ops: pass.ops,
                postamble: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::{Function, Instruction};

    fn context<'a>(sig_for: &'a dyn Fn(usize) -> Option<FunctionSig>) -> Context<'a> {
        static NO_INTRINSICS: std::sync::OnceLock<HashMap<usize, super::super::Intrinsic>> =
            std::sync::OnceLock::new();
        Context {
            callee_sig: sig_for,
            layout_of: &|_| None,
            function_named: &|_| None,
            global: &|_| None,
            globals_version: 0,
            intrinsics: NO_INTRINSICS.get_or_init(HashMap::new),
        }
    }

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

    fn int_sig(params: usize, register_count: u8) -> FunctionSig {
        FunctionSig {
            params: vec![Some(ValueType::Int); params],
            ret: Some(ValueType::Int),
            lua_function: false,
            register_count,
            param_types: vec![Some(TypeKind::Int); params],
            written_registers: u64::MAX,
            returned_registers: u64::MAX,
        }
    }

    #[test]
    fn fib_translates_with_typed_params_and_no_operand_guards() {
        let f = fib_function();
        let sig = int_sig(1, 8);
        let sig_for = |idx: usize| (idx == 0).then(|| sig.clone());
        let trace = translate(&f, 0, &sig, &context(&sig_for)).expect("fib is compilable");
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
    fn loops_reach_a_fixpoint_and_keep_typed_counters() {
        // i = 0; x = n; while i < n do i = i + 1; x = x * 2 end; return x
        let mut f = Function::new("pow2", 1, false);
        f.set_register_count(5);
        let k0 = f.chunk.add_constant(Value::Int(0));
        let k1 = f.chunk.add_constant(Value::Int(1));
        let k2 = f.chunk.add_constant(Value::Int(2));
        for ins in [
            Instruction::LoadConst(1, k0), // 0: i = 0
            Instruction::Move(2, 0),       // 1: x = n
            Instruction::LtInt(3, 1, 0),   // 2: header: i < n
            Instruction::JumpIfNot(3, 5),  // 3: exit → 9
            Instruction::LoadConst(4, k1), // 4
            Instruction::AddInt(1, 1, 4),  // 5: i = i + 1
            Instruction::LoadConst(4, k2), // 6
            Instruction::MulInt(2, 2, 4),  // 7: x = x * 2
            Instruction::Jump(-7),         // 8: → 2
            Instruction::Return(2),        // 9
        ] {
            f.chunk.emit(ins, 1);
        }
        let sig = int_sig(1, 5);
        let trace = translate(&f, 0, &sig, &context(&|_| None)).expect("loop is compilable");
        let guards = trace
            .ops
            .iter()
            .filter(|op| matches!(op, TraceOp::Guard { .. }))
            .count();
        assert_eq!(guards, 0, "{:?}", trace.ops);
        let header = trace
            .ops
            .iter()
            .find_map(|op| match op {
                TraceOp::Label { id: 2, scalars } => Some(scalars.clone()),
                _ => None,
            })
            .expect("loop header label");
        // The counters stay Int; the two temporaries hold a Bool (the
        // comparison) or nothing yet, so nothing owned either way.
        assert_eq!(
            header,
            vec![
                (0, ValueType::Int),
                (1, ValueType::Int),
                (2, ValueType::Int),
                (3, ValueType::Plain),
                (4, ValueType::Plain)
            ]
        );
        assert!(
            trace
                .ops
                .iter()
                .any(|op| matches!(op, TraceOp::Jump { label: 2 }))
        );
    }

    #[test]
    fn a_loop_that_changes_a_register_type_narrows_its_header() {
        // x starts as an Int and becomes a Float in the body: the header
        // must not claim x is an Int, and the typed op on it needs a guard.
        let mut f = Function::new("mix", 1, false);
        f.set_register_count(4);
        let k0 = f.chunk.add_constant(Value::Int(0));
        let kf = f.chunk.add_constant(Value::Float(1.5));
        for ins in [
            Instruction::LoadConst(1, k0), // 0: x = 0
            Instruction::LtInt(2, 1, 0),   // 1: header: needs a guard on x
            Instruction::JumpIfNot(2, 2),  // 2: → 5
            Instruction::LoadConst(1, kf), // 3: x = 1.5
            Instruction::Jump(-4),         // 4: → 1
            Instruction::Return(0),        // 5
        ] {
            f.chunk.emit(ins, 1);
        }
        let sig = int_sig(1, 4);
        let trace = translate(&f, 0, &sig, &context(&|_| None)).expect("compilable");
        let header = trace
            .ops
            .iter()
            .find_map(|op| match op {
                TraceOp::Label { id: 1, scalars } => Some(scalars.clone()),
                _ => None,
            })
            .expect("loop header label");
        // x is an Int or a Float — nothing owned, no particular type; the
        // comparison result and the unused register 3 likewise.
        assert_eq!(
            header,
            vec![
                (0, ValueType::Int),
                (1, ValueType::Plain),
                (2, ValueType::Plain),
                (3, ValueType::Plain)
            ]
        );
        assert!(trace.ops.iter().any(|op| matches!(
            op,
            TraceOp::Guard {
                register: 1,
                expected_type: ValueType::Int
            }
        )));
    }

    #[test]
    fn unknown_instructions_are_rejected() {
        let mut g = Function::new("upvalue", 2, false);
        g.set_register_count(3);
        g.chunk.emit(Instruction::LoadUpvalue(2, 0), 1);
        g.chunk.emit(Instruction::Return(2), 1);
        let sig = FunctionSig {
            params: vec![None, None],
            ret: None,
            lua_function: false,
            register_count: 3,
            param_types: vec![None, None],
            written_registers: u64::MAX,
            returned_registers: u64::MAX,
        };
        assert!(translate(&g, 0, &sig, &context(&|_| None)).is_none());
    }
}

/// End-to-end regressions for whole-function compilation: a program is run
/// until its functions are compiled (see `FUNCTION_HOT_THRESHOLD`) and the
/// results are compared against the interpreter's semantics.
#[cfg(all(test, feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
mod compiled_semantics_tests {
    use crate::bytecode::Value;
    use crate::embed::EmbeddedProgram;
    use alloc::rc::Rc;

    fn program(source: &str) -> EmbeddedProgram {
        let mut program = EmbeddedProgram::builder()
            .module("main", source)
            .entry_module("main")
            .compile()
            .expect("compile");
        program.run_entry_script().expect("run entry script");
        program
    }

    /// A function that returns one of its parameters cannot take it as an
    /// uncounted alias: `Return` moves the register into the caller's
    /// destination, so the value would be owned twice (the caller's
    /// original register and the destination) and freed twice.
    #[test]
    fn returning_a_parameter_keeps_its_reference_count() {
        let mut program = program(
            r#"
            function identity(a: Array<int>): Array<int>
                return a
            end
            function wrapper(a: Array<int>): Array<int>
                return identity(a)
            end
            function drive(a: Array<int>): int
                local b: Array<int> = wrapper(a)
                return array.len(b)
            end
        "#,
        );
        let value = Value::array(alloc::vec![Value::Int(42)]);
        let Value::Array(rc) = &value else {
            unreachable!("built an array")
        };
        let before = Rc::strong_count(rc);

        // Enough calls for all three functions to be compiled whole and to
        // call each other natively.
        for _ in 0..80 {
            let result = program
                .call_raw("main.drive", alloc::vec![value.clone()])
                .expect("drive");
            assert_eq!(result.as_int(), Some(1));
            assert_eq!(Rc::strong_count(rc), before);
        }
        assert!(program.jit_stats().functions_compiled >= 1);
        assert_eq!(value.array_len(), Some(1));
    }

    /// `MIN / -1` and `MIN % -1` wrap, as the interpreter does. x86 `idiv`
    /// raises #DE (a `SIGFPE`) for that quotient, so the backend has to
    /// take the divisor apart first; aarch64 and riscv wrap in hardware.
    #[test]
    fn integer_division_overflow_wraps_in_compiled_code() {
        let mut program = program(
            r#"
            function divide(a: int, b: int): int
                return a / b
            end
            function modulo(a: int, b: int): int
                return a % b
            end
            function drive(a: int, b: int, n: int): (int, int)
                local q: int = 0
                local r: int = 0
                local i: int = 0
                while i < n do
                    q = divide(a, b)
                    r = modulo(a, b)
                    i = i + 1
                end
                return q, r
            end
        "#,
        );
        let min = crate::LustInt::MIN;
        for _ in 0..80 {
            let quotient: crate::LustInt = program.call_typed("main.divide", (min, -1)).expect("divide");
            let remainder: crate::LustInt = program.call_typed("main.modulo", (min, -1)).expect("modulo");
            assert_eq!(quotient, min.wrapping_div(-1));
            assert_eq!(remainder, min.wrapping_rem(-1));
        }
        // Again through a traced loop calling both.
        let pair = program
            .call_raw("main.drive", alloc::vec![Value::Int(min), Value::Int(-1), Value::Int(200)])
            .expect("drive");
        let Value::Tuple(values) = &pair else {
            panic!("expected a tuple, got {pair:?}")
        };
        assert_eq!(values[0].as_int(), Some(min.wrapping_div(-1)));
        assert_eq!(values[1].as_int(), Some(min.wrapping_rem(-1)));

        // Division by zero is still an error, not a wrap.
        assert!(program.call_typed::<_, crate::LustInt>("main.divide", (1, 0)).is_err());
        assert!(program.call_typed::<_, crate::LustInt>("main.modulo", (1, 0)).is_err());
    }
}

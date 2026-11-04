use crate::bytecode::{Register, Value};
use crate::jit;
use crate::jit::trace::{Trace, TraceOp};
use alloc::{format, string::ToString, vec, vec::Vec};
use hashbrown::{HashMap, HashSet};
pub struct TraceOptimizer {
    hoisted_constants: Vec<(Register, Value)>,
}

impl TraceOptimizer {
    pub fn new() -> Self {
        Self {
            hoisted_constants: Vec::new(),
        }
    }

    pub fn optimize(&mut self, trace: &mut Trace) -> Vec<(Register, Value)> {
        jit::log(|| "🔧 JIT Optimizer: Starting optimization...".to_string());
        let original_ops = trace.ops.len();
        self.inline_leaf_functions(trace);
        self.fuse_try_get_index_patterns(trace);
        self.fuse_try_cast_patterns(trace);
        self.eliminate_dead_stores_before_select(trace);
        self.optimize_modulo_select(trace);
        self.optimize_constant_modulo(trace);
        self.hoist_constants(trace);
        self.unroll_loop(trace, crate::jit::UNROLL_FACTOR);
        self.hoist_entry_guards(trace);
        self.eliminate_redundant_type_guards(trace);
        self.coalesce_registers(trace);
        let optimized_ops = trace.ops.len();
        let hoisted = self.hoisted_constants.len();
        jit::log(|| {
            format!(
                "✨ JIT Optimizer: Optimized {} ops → {} ops, hoisted {} constants",
                original_ops, optimized_ops, hoisted
            )
        });
        self.hoisted_constants.clone()
    }

    /// Separate `At` markers from the ops they precede, so a pass can match
    /// consecutive ops. `join_markers` puts each marker back in front of
    /// whatever its op became.
    fn split_markers(ops: &[TraceOp]) -> (Vec<TraceOp>, Vec<Option<usize>>) {
        let mut bare = Vec::with_capacity(ops.len());
        let mut markers = Vec::with_capacity(ops.len());
        let mut pending = None;
        for op in ops {
            if let TraceOp::At { ip } = op {
                pending = Some(*ip);
            } else {
                bare.push(op.clone());
                markers.push(pending.take());
            }
        }
        (bare, markers)
    }

    /// The scalar type the first guard on `binding` or `result` in `ops`
    /// expects, looking past guards on other registers and moves that do
    /// not write either; `None` if anything else comes first.
    fn element_guard_type(
        ops: &[TraceOp],
        binding: Register,
        result: Register,
    ) -> Option<crate::jit::trace::ValueType> {
        use crate::jit::trace::ValueType;
        for op in ops {
            match op {
                TraceOp::Guard {
                    register,
                    expected_type,
                } if *register == binding || *register == result => {
                    return matches!(
                        expected_type,
                        ValueType::Int | ValueType::Float | ValueType::Bool
                    )
                    .then_some(*expected_type);
                }
                TraceOp::Guard { .. } => {}
                TraceOp::Move { dest, .. } if *dest != binding && *dest != result => {}
                _ => return None,
            }
        }
        None
    }

    fn push_marker(ops: &mut Vec<TraceOp>, marker: Option<usize>) {
        if let Some(ip) = marker {
            ops.push(TraceOp::At { ip });
        }
    }

    /// A checked index immediately matched as `Ok(value)` can keep its
    /// discriminant and payload in registers instead of allocating a Result.
    fn fuse_try_get_index_patterns(&mut self, trace: &mut Trace) {
        let (bare, markers) = Self::split_markers(&trace.ops);
        let mut ops = Vec::with_capacity(trace.ops.len());
        let mut i = 0;
        while i < bare.len() {
            // Whatever op i becomes, it belongs to op i's instruction.
            Self::push_marker(&mut ops, markers[i]);
            if i + 3 < bare.len()
                && let (
                    TraceOp::TryGetIndex {
                        dest: result_reg,
                        array,
                        index,
                    },
                    TraceOp::IsEnumVariant {
                        dest: condition_reg,
                        value: tested_reg,
                        enum_name,
                        variant_name,
                    },
                    TraceOp::GuardLoopContinue {
                        condition_register,
                        expect_truthy: true,
                        ..
                    },
                    TraceOp::GetEnumValue {
                        dest: binding_reg,
                        enum_reg,
                        index: 0,
                    },
                ) = (&bare[i], &bare[i + 1], &bare[i + 2], &bare[i + 3])
                && tested_reg == result_reg
                && enum_reg == result_reg
                && condition_register == condition_reg
                && enum_name == "Result"
                && variant_name == "Ok"
            {
                // The guard the recorder put on the binding (or the result)
                // right after the match says what element type to expect.
                let value_type =
                    Self::element_guard_type(&bare[i + 4..], *binding_reg, *result_reg);
                ops.push(TraceOp::ArrayIndexOk {
                    value_dest: *result_reg,
                    condition_dest: *condition_reg,
                    array: *array,
                    index: *index,
                    value_type,
                });
                ops.push(bare[i + 2].clone());
                ops.push(TraceOp::Move {
                    dest: *binding_reg,
                    src: *result_reg,
                });
                i += 4;
                continue;
            }

            ops.push(bare[i].clone());
            i += 1;
        }
        trace.ops = ops;
    }

    /// `value as T is Some(x)` lowers through an Option so the value can escape
    /// when needed. In the common immediate-pattern form, constructing that
    /// Option only to test and unpack it is redundant: the type test is the
    /// discriminant, and the successful payload is the original value.
    fn fuse_try_cast_patterns(&mut self, trace: &mut Trace) {
        let (bare, markers) = Self::split_markers(&trace.ops);
        let mut ops = Vec::with_capacity(trace.ops.len());
        let mut i = 0;
        while i < bare.len() {
            Self::push_marker(&mut ops, markers[i]);
            if i + 3 < bare.len()
                && let (
                    TraceOp::TryCast {
                        dest: option_reg,
                        value,
                        type_name,
                    },
                    TraceOp::IsEnumVariant {
                        dest: condition_reg,
                        value: tested_reg,
                        enum_name,
                        variant_name,
                    },
                    TraceOp::GuardLoopContinue {
                        condition_register,
                        expect_truthy: true,
                        ..
                    },
                    TraceOp::GetEnumValue {
                        dest: binding_reg,
                        enum_reg,
                        index: 0,
                    },
                ) = (&bare[i], &bare[i + 1], &bare[i + 2], &bare[i + 3])
                && tested_reg == option_reg
                && enum_reg == option_reg
                && condition_register == condition_reg
                && enum_name == "Option"
                && variant_name == "Some"
            {
                ops.push(TraceOp::TypeIs {
                    dest: *condition_reg,
                    value: *value,
                    type_name: type_name.clone(),
                });
                ops.push(bare[i + 2].clone());
                ops.push(TraceOp::Move {
                    dest: *binding_reg,
                    src: *value,
                });
                i += 4;
                continue;
            }

            ops.push(bare[i].clone());
            i += 1;
        }
        trace.ops = ops;
    }

    fn hoist_constants(&mut self, trace: &mut Trace) {
        // A LoadConst may only be lifted out of the loop body if the register it
        // targets is genuinely loop-invariant.  That requires a whole-body scan
        // *before* deciding anything: walking the ops in order and only looking
        // at what came earlier misses the case where a later op in the same
        // iteration clobbers the register, e.g.
        //
        //     [0] LoadConst  r2 <- 10      ; the `while i < 10` bound
        //     [2] Lt         r3 <- r1, r2
        //     [7] LoadConst  r2 <- 1       ; body reuses r2 as a scratch slot
        //
        // Hoisting op 0 leaves r2 == 1 on every iteration after the first, so
        // the loop silently starts testing `i < 1`.
        let mut clobbered: HashSet<Register> = HashSet::new();
        let mut const_value: HashMap<Register, Value> = HashMap::new();

        // A nested loop runs through its own trace and may write any
        // register of this frame, so nothing in such a body is invariant.
        if trace
            .ops
            .iter()
            .any(|op| matches!(op, TraceOp::NestedLoopCall { .. }))
        {
            return;
        }

        for op in &trace.ops {
            match op {
                TraceOp::CallNative { callee, .. }
                | TraceOp::CallFunction { callee, .. }
                | TraceOp::InlineCall { callee, .. } => {
                    // The callee register is read by the call itself.
                    clobbered.insert(*callee);
                }
                _ => {}
            }

            match op {
                TraceOp::LoadConst { dest, value } => match const_value.get(dest) {
                    // Two different constants into the same register: not invariant.
                    Some(seen) if !values_identical(seen, value) => {
                        clobbered.insert(*dest);
                    }
                    Some(_) => {}
                    None => {
                        const_value.insert(*dest, value.clone());
                    }
                },
                other => {
                    if let TraceOp::ArrayIndexOk {
                        value_dest,
                        condition_dest,
                        ..
                    } = other
                    {
                        clobbered.insert(*value_dest);
                        clobbered.insert(*condition_dest);
                    } else if let TraceOp::SpecializedOp { operands, .. } = other {
                        // A specialized op writes its register operands
                        // (`VecLen`'s length, say); treat every one as
                        // written rather than tell them apart.
                        for operand in operands {
                            if let crate::jit::trace::Operand::Register(register) = operand {
                                clobbered.insert(*register);
                            }
                        }
                    } else if let Some(dest) = Self::dest_of(other) {
                        clobbered.insert(dest);
                    }
                }
            }
        }

        let hoistable: HashSet<Register> = const_value
            .keys()
            .copied()
            .filter(|r| !clobbered.contains(r))
            .collect();

        let mut new_ops = Vec::with_capacity(trace.ops.len());
        let mut already_hoisted: HashSet<Register> = HashSet::new();

        for op in trace.ops.drain(..) {
            match op {
                TraceOp::LoadConst { dest, value } => {
                    if hoistable.contains(&dest) {
                        if already_hoisted.insert(dest) {
                            self.hoisted_constants.push((dest, value));
                        }
                        // Redundant reload of an invariant constant: drop it.
                    } else {
                        new_ops.push(TraceOp::LoadConst { dest, value });
                    }
                }

                other => new_ops.push(other),
            }
        }

        trace.ops = new_ops;
    }

    fn dest_of(op: &TraceOp) -> Option<Register> {
        match op {
            TraceOp::Move { dest, .. }
            | TraceOp::Add { dest, .. }
            | TraceOp::Sub { dest, .. }
            | TraceOp::Mul { dest, .. }
            | TraceOp::Div { dest, .. }
            | TraceOp::Mod { dest, .. }
            | TraceOp::Neg { dest, .. }
            | TraceOp::Eq { dest, .. }
            | TraceOp::Ne { dest, .. }
            | TraceOp::Lt { dest, .. }
            | TraceOp::Le { dest, .. }
            | TraceOp::Gt { dest, .. }
            | TraceOp::Ge { dest, .. }
            | TraceOp::And { dest, .. }
            | TraceOp::Or { dest, .. }
            | TraceOp::Not { dest, .. }
            | TraceOp::Concat { dest, .. }
            | TraceOp::GetIndex { dest, .. }
            | TraceOp::TryGetIndex { dest, .. }
            | TraceOp::ArrayPush { dest, .. }
            | TraceOp::ArrayLen { dest, .. }
            | TraceOp::CallMethod { dest, .. }
            | TraceOp::GetField { dest, .. }
            | TraceOp::NewArray { dest, .. }
            | TraceOp::NewStruct { dest, .. }
            | TraceOp::NewEnumUnit { dest, .. }
            | TraceOp::NewEnumVariant { dest, .. }
            | TraceOp::IsEnumVariant { dest, .. }
            | TraceOp::TypeIs { dest, .. }
            | TraceOp::TryCast { dest, .. }
            | TraceOp::GetEnumValue { dest, .. }
            | TraceOp::CallNative { dest, .. }
            | TraceOp::CallFunction { dest, .. }
            | TraceOp::Select { dest, .. }
            | TraceOp::SelectBitZero { dest, .. }
            | TraceOp::ModImm { dest, .. }
            | TraceOp::InlineCall { dest, .. } => Some(*dest),
            TraceOp::ArrayIndexOk { .. } => None,
            _ => None,
        }
    }

    /// The type of the value `op` writes to its destination, when the op
    /// itself says so (its annotations, not the environment).
    fn annotated_write_type(op: &TraceOp) -> Option<crate::jit::trace::ValueType> {
        use crate::jit::trace::ValueType;
        match op {
            TraceOp::ModImm { .. } => Some(ValueType::Int),
            TraceOp::Select {
                on_true, on_false, ..
            } => match (on_true, on_false) {
                (Value::Int(_), Value::Int(_)) => Some(ValueType::Int),
                (Value::Float(_), Value::Float(_)) => Some(ValueType::Float),
                (Value::Bool(_), Value::Bool(_)) => Some(ValueType::Bool),
                _ => None,
            },
            TraceOp::SelectBitZero {
                on_zero,
                on_nonzero,
                ..
            } => match (on_zero, on_nonzero) {
                (Value::Int(_), Value::Int(_)) => Some(ValueType::Int),
                (Value::Float(_), Value::Float(_)) => Some(ValueType::Float),
                (Value::Bool(_), Value::Bool(_)) => Some(ValueType::Bool),
                _ => None,
            },
            TraceOp::LoadConst { value, .. } => match value {
                Value::Bool(_) => Some(ValueType::Bool),
                Value::Int(_) => Some(ValueType::Int),
                Value::Float(_) => Some(ValueType::Float),
                // A function index or Nil: nothing owned, no particular type.
                Value::Function(_) | Value::Nil => Some(ValueType::Plain),
                _ => None,
            },
            TraceOp::Add {
                lhs_type, rhs_type, ..
            }
            | TraceOp::Sub {
                lhs_type, rhs_type, ..
            }
            | TraceOp::Mul {
                lhs_type, rhs_type, ..
            }
            | TraceOp::Div {
                lhs_type, rhs_type, ..
            } => match (lhs_type, rhs_type) {
                (ValueType::Int, ValueType::Int) => Some(ValueType::Int),
                (ValueType::Int | ValueType::Float, ValueType::Int | ValueType::Float) => {
                    Some(ValueType::Float)
                }
                _ => None,
            },
            TraceOp::Mod {
                lhs_type, rhs_type, ..
            } => match (lhs_type, rhs_type) {
                (ValueType::Int, ValueType::Int) => Some(ValueType::Int),
                _ => None,
            },
            TraceOp::Eq { .. }
            | TraceOp::Ne { .. }
            | TraceOp::Lt { .. }
            | TraceOp::Le { .. }
            | TraceOp::Gt { .. }
            | TraceOp::Ge { .. }
            | TraceOp::Not { .. }
            | TraceOp::IsEnumVariant { .. }
            | TraceOp::TypeIs { .. } => Some(ValueType::Bool),
            TraceOp::ArrayLen { .. } => Some(ValueType::Int),
            TraceOp::CallDirect { result_type, .. } => *result_type,
            TraceOp::InlineCall { trace, .. } => Self::inline_result_type(trace),
            _ => None,
        }
    }

    /// The type an inlined call's result has when its body proves it:
    /// what the body's guards and typed writes say about the register it
    /// returns.
    fn inline_result_type(
        trace: &crate::jit::trace::InlineTrace,
    ) -> Option<crate::jit::trace::ValueType> {
        let return_register = trace.return_register?;
        let mut known: HashMap<Register, crate::jit::trace::ValueType> = HashMap::new();
        for op in &trace.body {
            if let TraceOp::Guard {
                register,
                expected_type,
            } = op
            {
                known.insert(*register, *expected_type);
                continue;
            }
            let mut targets = Self::other_writes(op);
            if let Some(dest) = Self::dest_of(op) {
                targets.push(dest);
            }
            let ty = match op {
                TraceOp::Move { src, .. } => known.get(src).copied(),
                _ => Self::annotated_write_type(op),
            };
            for register in targets {
                match ty {
                    Some(ty) => {
                        known.insert(register, ty);
                    }
                    None => {
                        known.remove(&register);
                    }
                }
            }
        }
        known.get(&return_register).copied()
    }

    /// Every register `op` may write, beyond `dest_of`.
    fn other_writes(op: &TraceOp) -> Vec<Register> {
        match op {
            TraceOp::ArrayIndexOk {
                value_dest,
                condition_dest,
                ..
            } => vec![*value_dest, *condition_dest],
            TraceOp::Rebox { dest_reg, .. } => vec![*dest_reg],
            TraceOp::LoadConst { dest, .. } => vec![*dest],
            TraceOp::Unbox { source_reg, .. } => vec![*source_reg],
            TraceOp::CallDirect { dest, .. } => vec![*dest],
            TraceOp::SpecializedOp { operands, .. } => operands
                .iter()
                .filter_map(|operand| match operand {
                    crate::jit::trace::Operand::Register(register) => Some(*register),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// A type guard the loop re-checks every iteration on a register it
    /// never writes with another type is checked once, at entry: it moves
    /// to the preamble, and the body's copy (now redundant) goes. Only a
    /// guard that comes before any write of its register moves, so entry
    /// is not held to a type the loop would have established itself.
    fn hoist_entry_guards(&mut self, trace: &mut Trace) {
        // The inner loop may write any register.
        if trace
            .ops
            .iter()
            .any(|op| matches!(op, TraceOp::NestedLoopCall { .. }))
        {
            return;
        }
        // The type of each write, with a running environment so a `Move`
        // of a guarded or typed register counts as a typed write.
        let mut writes: HashMap<Register, Vec<Option<crate::jit::trace::ValueType>>> =
            HashMap::new();
        let mut known: HashMap<Register, crate::jit::trace::ValueType> = HashMap::new();
        for op in &trace.ops {
            if let TraceOp::Guard {
                register,
                expected_type,
            } = op
            {
                known.insert(*register, *expected_type);
                continue;
            }
            let ty = match op {
                TraceOp::Move { src, .. } => known.get(src).copied(),
                _ => Self::annotated_write_type(op),
            };
            for register in Self::other_writes(op).into_iter().chain(Self::dest_of(op)) {
                writes.entry(register).or_default().push(ty);
                match ty {
                    Some(ty) => {
                        known.insert(register, ty);
                    }
                    None => {
                        known.remove(&register);
                    }
                }
            }
        }
        let mut written_so_far: HashSet<Register> = HashSet::new();
        let mut hoisted: Vec<TraceOp> = Vec::new();
        let mut ops = Vec::with_capacity(trace.ops.len());
        for op in trace.ops.drain(..) {
            if let TraceOp::Guard {
                register,
                expected_type,
            } = &op
                && !written_so_far.contains(register)
                && writes
                    .get(register)
                    .is_none_or(|types| types.iter().all(|ty| *ty == Some(*expected_type)))
            {
                if !hoisted.iter().any(|h| {
                    matches!(h, TraceOp::Guard { register: r, expected_type: t } if r == register && t == expected_type)
                }) {
                    hoisted.push(op.clone());
                }
                // The body keeps its copy: the backend elides it, and a
                // marker before it still names the ip.
                ops.push(op);
                continue;
            }
            for register in Self::other_writes(&op)
                .into_iter()
                .chain(Self::dest_of(&op))
            {
                written_so_far.insert(register);
            }
            ops.push(op);
        }
        trace.ops = ops;
        // A register the loop writes with scalars before it reads it, and
        // never with anything else, holds nothing owned from the second
        // iteration on. Checking that at entry too lets every store into it
        // skip the tag check (and keep the value in a machine register for
        // the next op). At entry it usually holds the previous iteration's
        // scalar, or Nil; anything else costs one interpreted iteration.
        let mut first_access: HashMap<Register, bool> = HashMap::new(); // true = write
        for op in &trace.ops {
            for register in Self::reads_of(op) {
                first_access.entry(register).or_insert(false);
            }
            for register in Self::other_writes(op).into_iter().chain(Self::dest_of(op)) {
                first_access.entry(register).or_insert(true);
            }
        }
        let guarded: HashSet<Register> = hoisted
            .iter()
            .filter_map(|op| match op {
                TraceOp::Guard { register, .. } => Some(*register),
                _ => None,
            })
            .collect();
        let mut plain: Vec<Register> = first_access
            .iter()
            .filter(|(register, written_first)| {
                **written_first
                    && !guarded.contains(*register)
                    && writes.get(*register).is_some_and(|types| {
                        types.iter().all(|ty| {
                            matches!(
                                ty,
                                Some(
                                    crate::jit::trace::ValueType::Int
                                        | crate::jit::trace::ValueType::Float
                                        | crate::jit::trace::ValueType::Bool
                                        | crate::jit::trace::ValueType::Plain
                                )
                            )
                        })
                    })
            })
            .map(|(register, _)| *register)
            .collect();
        plain.sort_unstable();
        for register in plain {
            hoisted.push(TraceOp::Guard {
                register,
                expected_type: crate::jit::trace::ValueType::Plain,
            });
        }
        if !hoisted.is_empty() {
            jit::log(|| {
                format!(
                    "⬆️  JIT Optimizer: hoisted {} entry guard(s)",
                    hoisted.len()
                )
            });
            let mut preamble = hoisted;
            preamble.append(&mut trace.preamble);
            trace.preamble = preamble;
        }
    }

    /// Every register `op` reads (conservatively: the inputs the op names;
    /// an inlined call reads its arguments and callee).
    fn reads_of(op: &TraceOp) -> Vec<Register> {
        match op {
            TraceOp::Move { src, .. } | TraceOp::Neg { src, .. } | TraceOp::Not { src, .. } => {
                vec![*src]
            }
            TraceOp::Add { lhs, rhs, .. }
            | TraceOp::Sub { lhs, rhs, .. }
            | TraceOp::Mul { lhs, rhs, .. }
            | TraceOp::Div { lhs, rhs, .. }
            | TraceOp::Mod { lhs, rhs, .. }
            | TraceOp::Eq { lhs, rhs, .. }
            | TraceOp::Ne { lhs, rhs, .. }
            | TraceOp::Lt { lhs, rhs, .. }
            | TraceOp::Le { lhs, rhs, .. }
            | TraceOp::Gt { lhs, rhs, .. }
            | TraceOp::Ge { lhs, rhs, .. }
            | TraceOp::And { lhs, rhs, .. }
            | TraceOp::Or { lhs, rhs, .. }
            | TraceOp::Concat { lhs, rhs, .. } => vec![*lhs, *rhs],
            TraceOp::Guard { register, .. }
            | TraceOp::GuardFunction { register, .. }
            | TraceOp::GuardClosure { register, .. }
            | TraceOp::GuardNativeFunction { register, .. }
            | TraceOp::GuardStructLayout { register, .. }
            | TraceOp::TypeIs {
                value: register, ..
            }
            | TraceOp::IsEnumVariant {
                value: register, ..
            }
            | TraceOp::GetEnumValue {
                enum_reg: register, ..
            }
            | TraceOp::GetField {
                object: register, ..
            }
            | TraceOp::ArrayLen {
                array: register, ..
            }
            | TraceOp::TryCast {
                value: register, ..
            } => vec![*register],
            TraceOp::GuardLoopContinue {
                condition_register, ..
            }
            | TraceOp::BranchIf {
                condition_register, ..
            } => vec![*condition_register],
            TraceOp::SetField { object, value, .. } => vec![*object, *value],
            TraceOp::SetIndex {
                array,
                index,
                value,
            } => vec![*array, *index, *value],
            TraceOp::ArrayPush { array, value, .. } => vec![*array, *value],
            TraceOp::GetIndex { array, index, .. }
            | TraceOp::TryGetIndex { array, index, .. }
            | TraceOp::ArrayIndexOk { array, index, .. } => vec![*array, *index],
            TraceOp::Return { value } => value.iter().copied().collect(),
            TraceOp::InlineCall { callee, trace, .. } => {
                let mut reads = trace.arg_registers.clone();
                reads.push(*callee);
                reads
            }
            TraceOp::CallNative {
                callee,
                first_arg,
                arg_count,
                ..
            }
            | TraceOp::CallFunction {
                callee,
                first_arg,
                arg_count,
                ..
            } => {
                let mut reads: Vec<Register> = (0..*arg_count).map(|i| first_arg + i).collect();
                reads.push(*callee);
                reads
            }
            TraceOp::CallMethod {
                object,
                first_arg,
                arg_count,
                ..
            } => {
                let mut reads: Vec<Register> = (0..*arg_count).map(|i| first_arg + i).collect();
                reads.push(*object);
                reads
            }
            TraceOp::CallDirect {
                callee,
                receiver,
                first_arg,
                arg_count,
                ..
            } => {
                let mut reads: Vec<Register> = (0..*arg_count).map(|i| first_arg + i).collect();
                reads.push(*callee);
                reads.extend(receiver.iter().copied());
                reads
            }
            TraceOp::NewArray {
                first_element,
                count,
                ..
            } => (0..*count).map(|i| first_element + i).collect(),
            TraceOp::NewStruct {
                field_registers, ..
            } => field_registers.clone(),
            TraceOp::NewEnumVariant {
                value_registers, ..
            } => value_registers.clone(),
            TraceOp::Select { cond, .. } => vec![*cond],
            TraceOp::SelectBitZero { src, .. } => vec![*src],
            TraceOp::ModImm { src, .. } => vec![*src],
            TraceOp::NewEnumUnit { .. }
            | TraceOp::LoadConst { .. }
            | TraceOp::GuardGlobals { .. }
            | TraceOp::At { .. }
            | TraceOp::Label { .. }
            | TraceOp::Jump { .. }
            | TraceOp::DropSpecialized { .. } => Vec::new(),
            // Specialized ops and reboxing read and write through slots;
            // treat every register as read (no entry guard comes of it).
            _ => (0..=u8::MAX).collect(),
        }
    }

    fn eliminate_redundant_type_guards(&mut self, trace: &mut Trace) {
        let mut known_types: HashMap<Register, crate::jit::trace::ValueType> = HashMap::new();
        let mut ops = Vec::with_capacity(trace.ops.len());

        for op in trace.ops.drain(..) {
            if let TraceOp::Guard {
                register,
                expected_type,
            } = &op
            {
                if known_types.get(register) == Some(expected_type) {
                    continue;
                }
                known_types.insert(*register, *expected_type);
                ops.push(op);
                continue;
            }

            if let TraceOp::ArrayIndexOk {
                value_dest,
                condition_dest,
                ..
            } = &op
            {
                known_types.remove(value_dest);
                known_types.insert(*condition_dest, crate::jit::trace::ValueType::Bool);
            } else if let Some(dest) = Self::dest_of(&op) {
                match Self::result_type(&op, &known_types) {
                    Some(ty) => {
                        known_types.insert(dest, ty);
                    }
                    None => {
                        known_types.remove(&dest);
                    }
                }
            } else if let TraceOp::Rebox { dest_reg, .. } = &op {
                known_types.remove(dest_reg);
            } else if let TraceOp::NestedLoopCall { .. } = &op {
                // The inner loop may have written any register.
                known_types.clear();
            }

            ops.push(op);
        }

        trace.ops = ops;
    }

    fn result_type(
        op: &TraceOp,
        known_types: &HashMap<Register, crate::jit::trace::ValueType>,
    ) -> Option<crate::jit::trace::ValueType> {
        use crate::jit::trace::ValueType;

        match op {
            TraceOp::LoadConst { value, .. } => match value {
                Value::Bool(_) => Some(ValueType::Bool),
                Value::Int(_) => Some(ValueType::Int),
                Value::Float(_) => Some(ValueType::Float),
                Value::String(_) => Some(ValueType::String),
                Value::Array(_) => Some(ValueType::Array),
                Value::Tuple(_) => Some(ValueType::Tuple),
                Value::Struct { .. } | Value::WeakStruct(_) => Some(ValueType::Struct),
                _ => None,
            },
            TraceOp::Move { src, .. } => known_types.get(src).copied(),
            TraceOp::Add {
                lhs_type, rhs_type, ..
            }
            | TraceOp::Sub {
                lhs_type, rhs_type, ..
            }
            | TraceOp::Mul {
                lhs_type, rhs_type, ..
            }
            | TraceOp::Div {
                lhs_type, rhs_type, ..
            } => match (*lhs_type, *rhs_type) {
                (ValueType::Int, ValueType::Int) => Some(ValueType::Int),
                (ValueType::Int | ValueType::Float, ValueType::Int | ValueType::Float) => {
                    Some(ValueType::Float)
                }
                _ => None,
            },
            TraceOp::Mod {
                lhs_type, rhs_type, ..
            } => match (*lhs_type, *rhs_type) {
                (ValueType::Int, ValueType::Int) => Some(ValueType::Int),
                _ => None,
            },
            TraceOp::Neg { src, .. } => known_types
                .get(src)
                .copied()
                .filter(|ty| matches!(ty, ValueType::Int | ValueType::Float)),
            TraceOp::Eq { .. }
            | TraceOp::Ne { .. }
            | TraceOp::Lt { .. }
            | TraceOp::Le { .. }
            | TraceOp::Gt { .. }
            | TraceOp::Ge { .. }
            | TraceOp::And { .. }
            | TraceOp::Or { .. }
            | TraceOp::Not { .. }
            | TraceOp::IsEnumVariant { .. }
            | TraceOp::TypeIs { .. } => Some(ValueType::Bool),
            TraceOp::Concat { .. } => Some(ValueType::String),
            TraceOp::ArrayLen { .. } => Some(ValueType::Int),
            // The value observed while recording does not make a mutable
            // field's runtime type stable. Only a later Guard can do that.
            TraceOp::GetField { .. } => None,
            TraceOp::NewArray { .. } => Some(ValueType::Array),
            TraceOp::NewStruct { .. } => Some(ValueType::Struct),
            TraceOp::Select {
                on_true, on_false, ..
            } => match (on_true, on_false) {
                (Value::Int(_), Value::Int(_)) => Some(ValueType::Int),
                (Value::Float(_), Value::Float(_)) => Some(ValueType::Float),
                (Value::Bool(_), Value::Bool(_)) => Some(ValueType::Bool),
                _ => None,
            },
            TraceOp::SelectBitZero {
                on_zero,
                on_nonzero,
                ..
            } => match (on_zero, on_nonzero) {
                (Value::Int(_), Value::Int(_)) => Some(ValueType::Int),
                (Value::Float(_), Value::Float(_)) => Some(ValueType::Float),
                (Value::Bool(_), Value::Bool(_)) => Some(ValueType::Bool),
                _ => None,
            },
            TraceOp::ModImm { .. } => Some(ValueType::Int),
            _ => None,
        }
    }

    fn is_leaf_inline_trace(trace: &crate::jit::trace::InlineTrace) -> bool {
        if trace.is_closure
            || trace.return_register.is_none()
            || trace.register_count > 32
            || trace.body.len() > 64
        {
            return false;
        }
        for op in &trace.body {
            match op {
                TraceOp::LoadConst { .. }
                | TraceOp::Move { .. }
                | TraceOp::Add { .. }
                | TraceOp::Sub { .. }
                | TraceOp::Mul { .. }
                | TraceOp::Div { .. }
                | TraceOp::Mod { .. }
                | TraceOp::ModImm { .. }
                | TraceOp::Neg { .. }
                | TraceOp::Eq { .. }
                | TraceOp::Ne { .. }
                | TraceOp::Lt { .. }
                | TraceOp::Le { .. }
                | TraceOp::Gt { .. }
                | TraceOp::Ge { .. }
                | TraceOp::And { .. }
                | TraceOp::Or { .. }
                | TraceOp::Not { .. }
                | TraceOp::Select { .. }
                | TraceOp::SelectBitZero { .. }
                | TraceOp::Guard { .. }
                | TraceOp::At { .. } => {}
                _ => return false,
            }
        }
        true
    }

    fn remap_leaf_op(op: &TraceOp, map: &HashMap<Register, Register>) -> TraceOp {
        let r = |reg: &Register| -> Register { *map.get(reg).unwrap_or(reg) };
        match op {
            TraceOp::LoadConst { dest, value } => TraceOp::LoadConst {
                dest: r(dest),
                value: value.clone(),
            },
            TraceOp::Move { dest, src } => TraceOp::Move {
                dest: r(dest),
                src: r(src),
            },
            TraceOp::Add {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => TraceOp::Add {
                dest: r(dest),
                lhs: r(lhs),
                rhs: r(rhs),
                lhs_type: *lhs_type,
                rhs_type: *rhs_type,
            },
            TraceOp::Sub {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => TraceOp::Sub {
                dest: r(dest),
                lhs: r(lhs),
                rhs: r(rhs),
                lhs_type: *lhs_type,
                rhs_type: *rhs_type,
            },
            TraceOp::Mul {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => TraceOp::Mul {
                dest: r(dest),
                lhs: r(lhs),
                rhs: r(rhs),
                lhs_type: *lhs_type,
                rhs_type: *rhs_type,
            },
            TraceOp::Div {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => TraceOp::Div {
                dest: r(dest),
                lhs: r(lhs),
                rhs: r(rhs),
                lhs_type: *lhs_type,
                rhs_type: *rhs_type,
            },
            TraceOp::Mod {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => TraceOp::Mod {
                dest: r(dest),
                lhs: r(lhs),
                rhs: r(rhs),
                lhs_type: *lhs_type,
                rhs_type: *rhs_type,
            },
            TraceOp::ModImm { dest, src, divisor } => TraceOp::ModImm {
                dest: r(dest),
                src: r(src),
                divisor: *divisor,
            },
            TraceOp::Neg { dest, src } => TraceOp::Neg {
                dest: r(dest),
                src: r(src),
            },
            TraceOp::Eq {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => TraceOp::Eq {
                dest: r(dest),
                lhs: r(lhs),
                rhs: r(rhs),
                lhs_type: *lhs_type,
                rhs_type: *rhs_type,
            },
            TraceOp::Ne {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => TraceOp::Ne {
                dest: r(dest),
                lhs: r(lhs),
                rhs: r(rhs),
                lhs_type: *lhs_type,
                rhs_type: *rhs_type,
            },
            TraceOp::Lt {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => TraceOp::Lt {
                dest: r(dest),
                lhs: r(lhs),
                rhs: r(rhs),
                lhs_type: *lhs_type,
                rhs_type: *rhs_type,
            },
            TraceOp::Le {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => TraceOp::Le {
                dest: r(dest),
                lhs: r(lhs),
                rhs: r(rhs),
                lhs_type: *lhs_type,
                rhs_type: *rhs_type,
            },
            TraceOp::Gt {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => TraceOp::Gt {
                dest: r(dest),
                lhs: r(lhs),
                rhs: r(rhs),
                lhs_type: *lhs_type,
                rhs_type: *rhs_type,
            },
            TraceOp::Ge {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => TraceOp::Ge {
                dest: r(dest),
                lhs: r(lhs),
                rhs: r(rhs),
                lhs_type: *lhs_type,
                rhs_type: *rhs_type,
            },
            TraceOp::And { dest, lhs, rhs } => TraceOp::And {
                dest: r(dest),
                lhs: r(lhs),
                rhs: r(rhs),
            },
            TraceOp::Or { dest, lhs, rhs } => TraceOp::Or {
                dest: r(dest),
                lhs: r(lhs),
                rhs: r(rhs),
            },
            TraceOp::Not { dest, src } => TraceOp::Not {
                dest: r(dest),
                src: r(src),
            },
            TraceOp::Select {
                dest,
                cond,
                on_true,
                on_false,
            } => TraceOp::Select {
                dest: r(dest),
                cond: r(cond),
                on_true: on_true.clone(),
                on_false: on_false.clone(),
            },
            TraceOp::SelectBitZero {
                dest,
                src,
                mask,
                on_zero,
                on_nonzero,
            } => TraceOp::SelectBitZero {
                dest: r(dest),
                src: r(src),
                mask: *mask,
                on_zero: on_zero.clone(),
                on_nonzero: on_nonzero.clone(),
            },
            TraceOp::Guard {
                register,
                expected_type,
            } => TraceOp::Guard {
                register: r(register),
                expected_type: *expected_type,
            },
            TraceOp::At { ip } => TraceOp::At { ip: *ip },
            other => other.clone(),
        }
    }

    fn inline_leaf_functions(&mut self, trace: &mut Trace) {
        if !trace
            .ops
            .iter()
            .any(|op| matches!(op, TraceOp::InlineCall { .. }))
        {
            return;
        }

        let mut max_reg = 0u8;
        for op in trace
            .preamble
            .iter()
            .chain(&trace.ops)
            .chain(&trace.postamble)
        {
            if let Some(d) = Self::dest_of(op) {
                max_reg = max_reg.max(d);
            }
            for r in Self::reads_of(op) {
                max_reg = max_reg.max(r);
            }
            for w in Self::other_writes(op) {
                max_reg = max_reg.max(w);
            }
        }

        let mut new_ops = Vec::with_capacity(trace.ops.len() * 2);
        let mut inlined_any = false;

        for op in trace.ops.drain(..) {
            if let TraceOp::InlineCall {
                dest: inline_dest,
                callee,
                trace: inline_trace,
            } = op
            {
                if Self::is_leaf_inline_trace(&inline_trace)
                    && (max_reg as usize + inline_trace.register_count as usize) < 240
                {
                    inlined_any = true;
                    let mut reg_map: HashMap<Register, Register> = HashMap::new();
                    let mut prelude_moves = Vec::new();

                    for i in 0..inline_trace.arg_count {
                        let writes_param = inline_trace.body.iter().any(|callee_op| {
                            Self::dest_of(callee_op) == Some(i)
                                || Self::other_writes(callee_op).contains(&i)
                        });
                        let caller_arg_reg = inline_trace.arg_registers[i as usize];
                        if writes_param {
                            max_reg = max_reg.saturating_add(1);
                            let fresh_reg = max_reg;
                            prelude_moves.push(TraceOp::Move {
                                dest: fresh_reg,
                                src: caller_arg_reg,
                            });
                            reg_map.insert(i, fresh_reg);
                        } else {
                            reg_map.insert(i, caller_arg_reg);
                        }
                    }

                    let ret_reg = inline_trace.return_register.unwrap();
                    let can_direct_return = ret_reg >= inline_trace.arg_count
                        && !inline_trace.arg_registers.contains(&inline_dest);

                    for r in inline_trace.arg_count..inline_trace.register_count {
                        if r == ret_reg && can_direct_return {
                            reg_map.insert(r, inline_dest);
                        } else {
                            max_reg = max_reg.saturating_add(1);
                            reg_map.insert(r, max_reg);
                        }
                    }

                    new_ops.extend(prelude_moves);

                    for callee_op in &inline_trace.body {
                        if matches!(callee_op, TraceOp::At { .. }) {
                            continue;
                        }
                        let remapped = Self::remap_leaf_op(callee_op, &reg_map);
                        new_ops.push(remapped);
                    }

                    let mapped_ret = reg_map.get(&ret_reg).copied().unwrap_or(ret_reg);
                    if mapped_ret != inline_dest {
                        new_ops.push(TraceOp::Move {
                            dest: inline_dest,
                            src: mapped_ret,
                        });
                    }
                    continue;
                } else {
                    new_ops.push(TraceOp::InlineCall {
                        dest: inline_dest,
                        callee,
                        trace: inline_trace,
                    });
                }
            } else {
                new_ops.push(op);
            }
        }

        trace.ops = new_ops;

        if inlined_any {
            crate::jit::log(|| {
                "🌿 JIT Optimizer: Inlined leaf functions into caller trace".to_string()
            });
        }
    }

    fn unroll_loop(&mut self, trace: &mut Trace, factor: usize) {
        if factor <= 1 || trace.ops.is_empty() {
            return;
        }

        if trace.ops.iter().any(|op| {
            matches!(
                op,
                TraceOp::InlineCall { .. } | TraceOp::NestedLoopCall { .. }
            )
        }) {
            return;
        }

        let loop_condition_op = trace.ops.iter().find_map(|op| match op {
            TraceOp::Le { dest, .. }
            | TraceOp::Lt { dest, .. }
            | TraceOp::Ge { dest, .. }
            | TraceOp::Gt { dest, .. } => Some((op.clone(), *dest)),
            _ => None,
        });
        if loop_condition_op.is_none() {
            return;
        }

        // The recorded body already contains the loop's own exit test: the
        // back-edge's JumpIf/JumpIfNot was recorded as a GuardLoopContinue.  An
        // unrolled iteration is therefore just the body again, verbatim.
        //
        // Do not synthesize a fresh comparison, and do not strip comparisons out
        // of the copies.  A trace body also contains comparisons belonging to
        // user code (`if x > 10 then`), indistinguishable from the loop's own
        // test at this level; dropping those leaves the guards that consume them
        // reading a stale register, which silently forces the branch one way.
        let original_ops = trace.ops.clone();
        let mut new_ops = Vec::with_capacity(original_ops.len() * factor);
        for _ in 0..factor {
            new_ops.extend(original_ops.iter().cloned());
        }

        trace.ops = new_ops;
    }

    fn eliminate_dead_stores_before_select(&mut self, trace: &mut Trace) {
        if trace.ops.is_empty() {
            return;
        }
        let mut to_remove = hashbrown::HashSet::new();
        for (i, op) in trace.ops.iter().enumerate() {
            if let TraceOp::Select { dest, .. } | TraceOp::SelectBitZero { dest, .. } = op {
                // Look backward for a LoadConst targeting dest before any read of dest
                for k in (0..i).rev() {
                    let prev = &trace.ops[k];
                    if matches!(prev, TraceOp::At { .. }) {
                        continue;
                    }
                    if let TraceOp::LoadConst {
                        dest: load_dest, ..
                    } = prev
                    {
                        if load_dest == dest {
                            to_remove.insert(k);
                            break;
                        }
                    }
                    if Self::reads_of(prev).contains(dest) || Self::dest_of(prev) == Some(*dest) {
                        break;
                    }
                }
            }
        }
        if !to_remove.is_empty() {
            let mut new_ops = Vec::with_capacity(trace.ops.len() - to_remove.len());
            for (i, op) in trace.ops.drain(..).enumerate() {
                if !to_remove.contains(&i) {
                    new_ops.push(op);
                }
            }
            trace.ops = new_ops;
        }
    }

    fn optimize_modulo_select(&mut self, trace: &mut Trace) {
        use crate::jit::trace::ValueType;

        let mut to_remove = hashbrown::HashSet::new();
        let mut replacements = hashbrown::HashMap::new();

        for (i, op) in trace.ops.iter().enumerate() {
            if let TraceOp::Select {
                dest,
                cond,
                on_true,
                on_false,
            } = op
            {
                // 1. Find the Eq operation defining cond
                let mut eq_info = None;
                for k in (0..i).rev() {
                    let prev = &trace.ops[k];
                    if matches!(prev, TraceOp::At { .. }) {
                        continue;
                    }
                    if let TraceOp::Eq {
                        dest: eq_d,
                        lhs,
                        rhs,
                        lhs_type,
                        rhs_type,
                    } = prev
                    {
                        if *eq_d == *cond
                            && *lhs_type == ValueType::Int
                            && *rhs_type == ValueType::Int
                        {
                            eq_info = Some((k, *lhs, *rhs));
                            break;
                        }
                    }
                    if Self::reads_of(prev).contains(cond) || Self::dest_of(prev) == Some(*cond) {
                        break;
                    }
                }
                let Some((eq_idx, eq_lhs, eq_rhs)) = eq_info else {
                    continue;
                };

                // 2. Check if one of Eq's operands is constant 0, and the other is mod_dest
                let check_zero =
                    |reg: Register, ops: &[TraceOp], before_idx: usize| -> Option<usize> {
                        for k in (0..before_idx).rev() {
                            let prev = &ops[k];
                            if matches!(prev, TraceOp::At { .. }) {
                                continue;
                            }
                            if let TraceOp::LoadConst {
                                dest: d,
                                value: Value::Int(0),
                            } = prev
                            {
                                if *d == reg {
                                    return Some(k);
                                }
                            }
                            if Self::dest_of(prev) == Some(reg) {
                                break;
                            }
                        }
                        None
                    };

                let mut zero_reg = None;
                let mut mod_reg = None;
                let mut zero_idx = None;
                if let Some(k) = check_zero(eq_rhs, &trace.ops, eq_idx) {
                    zero_reg = Some(eq_rhs);
                    mod_reg = Some(eq_lhs);
                    zero_idx = Some(k);
                } else if let Some(k) = check_zero(eq_lhs, &trace.ops, eq_idx) {
                    zero_reg = Some(eq_lhs);
                    mod_reg = Some(eq_rhs);
                    zero_idx = Some(k);
                }

                let Some(mod_reg) = mod_reg else { continue };

                // 3. Find the Mod operation defining mod_reg: mod_reg = src % divisor_reg
                let mut mod_info = None;
                for k in (0..eq_idx).rev() {
                    let prev = &trace.ops[k];
                    if matches!(prev, TraceOp::At { .. }) {
                        continue;
                    }
                    if let TraceOp::Mod {
                        dest: d,
                        lhs: src,
                        rhs: divisor_reg,
                        lhs_type,
                        rhs_type,
                    } = prev
                    {
                        if *d == mod_reg
                            && *lhs_type == ValueType::Int
                            && *rhs_type == ValueType::Int
                        {
                            mod_info = Some((k, *src, *divisor_reg));
                            break;
                        }
                    }
                    if Self::reads_of(prev).contains(&mod_reg)
                        || Self::dest_of(prev) == Some(mod_reg)
                    {
                        break;
                    }
                }
                let Some((mod_idx, src, divisor_reg)) = mod_info else {
                    continue;
                };

                // 4. Check if divisor_reg is constant 2
                let mut divisor_idx = None;
                for k in (0..mod_idx).rev() {
                    let prev = &trace.ops[k];
                    if matches!(prev, TraceOp::At { .. }) {
                        continue;
                    }
                    if let TraceOp::LoadConst {
                        dest: d,
                        value: Value::Int(2),
                    } = prev
                    {
                        if *d == divisor_reg {
                            divisor_idx = Some(k);
                            break;
                        }
                    }
                    if Self::dest_of(prev) == Some(divisor_reg) {
                        break;
                    }
                }
                let Some(divisor_idx) = divisor_idx else {
                    continue;
                };

                // Check downstream reads of intermediate registers
                let downstream_reads = |reg: Register, start: usize| -> bool {
                    for op in &trace.ops[start..] {
                        if Self::reads_of(op).contains(&reg) {
                            return true;
                        }
                        if Self::dest_of(op) == Some(reg) {
                            break;
                        }
                    }
                    false
                };

                if downstream_reads(*cond, i + 1) || downstream_reads(mod_reg, eq_idx + 1) {
                    continue;
                }

                to_remove.insert(divisor_idx);
                to_remove.insert(mod_idx);
                if let Some(z_idx) = zero_idx {
                    if let Some(zr) = zero_reg {
                        if !downstream_reads(zr, eq_idx + 1) {
                            to_remove.insert(z_idx);
                        }
                    }
                }
                to_remove.insert(eq_idx);

                replacements.insert(
                    i,
                    TraceOp::SelectBitZero {
                        dest: *dest,
                        src,
                        mask: 1,
                        on_zero: on_true.clone(),
                        on_nonzero: on_false.clone(),
                    },
                );
            }
        }

        if !to_remove.is_empty() {
            let mut new_ops = Vec::with_capacity(trace.ops.len());
            for (idx, op) in trace.ops.drain(..).enumerate() {
                if let Some(replacement) = replacements.remove(&idx) {
                    new_ops.push(replacement);
                } else if !to_remove.contains(&idx) {
                    new_ops.push(op);
                }
            }
            trace.ops = new_ops;
        }
    }

    fn optimize_constant_modulo(&mut self, trace: &mut Trace) {
        use crate::jit::trace::ValueType;

        let mut to_remove = hashbrown::HashSet::new();
        let mut replacements = hashbrown::HashMap::new();

        for (i, op) in trace.ops.iter().enumerate() {
            if let TraceOp::Mod {
                dest,
                lhs,
                rhs,
                lhs_type: ValueType::Int,
                rhs_type: ValueType::Int,
            } = op
            {
                let mut const_info = None;
                for k in (0..i).rev() {
                    let prev = &trace.ops[k];
                    if matches!(prev, TraceOp::At { .. }) {
                        continue;
                    }
                    if let TraceOp::LoadConst {
                        dest: d,
                        value: Value::Int(c),
                    } = prev
                    {
                        if *d == *rhs && *c > 0 {
                            const_info = Some((k, *c));
                            break;
                        }
                    }
                    if Self::dest_of(prev) == Some(*rhs) {
                        break;
                    }
                }

                if let Some((const_idx, divisor)) = const_info {
                    let downstream_reads = |reg: Register, start: usize| -> bool {
                        for op in &trace.ops[start..] {
                            if Self::reads_of(op).contains(&reg) {
                                return true;
                            }
                            if Self::dest_of(op) == Some(reg) {
                                break;
                            }
                        }
                        false
                    };

                    if !downstream_reads(*rhs, i + 1) {
                        to_remove.insert(const_idx);
                    }

                    replacements.insert(
                        i,
                        TraceOp::ModImm {
                            dest: *dest,
                            src: *lhs,
                            divisor,
                        },
                    );
                }
            }
        }

        if !replacements.is_empty() {
            let mut new_ops = Vec::with_capacity(trace.ops.len());
            for (idx, op) in trace.ops.drain(..).enumerate() {
                if let Some(replacement) = replacements.remove(&idx) {
                    new_ops.push(replacement);
                } else if !to_remove.contains(&idx) {
                    new_ops.push(op);
                }
            }
            trace.ops = new_ops;
        }
    }

    fn coalesce_registers(&mut self, _trace: &mut Trace) {}
}

impl Default for TraceOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Constants are compared structurally; two `LoadConst`s targeting the same
/// register are only interchangeable if they load the very same value.
fn values_identical(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => x.to_bits() == y.to_bits(),
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Nil, Value::Nil) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immediate_try_cast_pattern_does_not_materialize_option() {
        let mut trace = Trace {
            function_idx: 0,
            start_ip: 0,
            is_function: false,
            frame_may_own: true,
            entry_scalars: Vec::new(),
            alias_params: 0,
            borrowed_registers: Vec::new(),
            preamble: Vec::new(),
            ops: vec![
                TraceOp::TryCast {
                    dest: 1,
                    value: 0,
                    type_name: "int".to_string(),
                },
                TraceOp::IsEnumVariant {
                    dest: 2,
                    value: 1,
                    enum_name: "Option".to_string(),
                    variant_name: "Some".to_string(),
                },
                TraceOp::GuardLoopContinue {
                    condition_register: 2,
                    expect_truthy: true,
                    bailout_ip: 10,
                },
                TraceOp::GetEnumValue {
                    dest: 3,
                    enum_reg: 1,
                    index: 0,
                },
            ],
            postamble: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        };

        TraceOptimizer::new().fuse_try_cast_patterns(&mut trace);

        assert!(matches!(
            trace.ops.as_slice(),
            [
                TraceOp::TypeIs {
                    dest: 2,
                    value: 0,
                    type_name,
                },
                TraceOp::GuardLoopContinue {
                    condition_register: 2,
                    expect_truthy: true,
                    bailout_ip: 10,
                },
                TraceOp::Move { dest: 3, src: 0 },
            ] if type_name == "int"
        ));
    }

    #[test]
    fn immediate_checked_index_pattern_does_not_materialize_result() {
        let mut trace = Trace {
            function_idx: 0,
            start_ip: 0,
            is_function: false,
            frame_may_own: true,
            entry_scalars: Vec::new(),
            alias_params: 0,
            borrowed_registers: Vec::new(),
            preamble: Vec::new(),
            ops: vec![
                TraceOp::TryGetIndex {
                    dest: 2,
                    array: 0,
                    index: 1,
                },
                TraceOp::IsEnumVariant {
                    dest: 3,
                    value: 2,
                    enum_name: "Result".to_string(),
                    variant_name: "Ok".to_string(),
                },
                TraceOp::GuardLoopContinue {
                    condition_register: 3,
                    expect_truthy: true,
                    bailout_ip: 10,
                },
                TraceOp::GetEnumValue {
                    dest: 3,
                    enum_reg: 2,
                    index: 0,
                },
            ],
            postamble: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        };

        TraceOptimizer::new().fuse_try_get_index_patterns(&mut trace);

        assert!(matches!(
            trace.ops.as_slice(),
            [
                TraceOp::ArrayIndexOk {
                    value_dest: 2,
                    condition_dest: 3,
                    array: 0,
                    index: 1,
                    ..
                },
                TraceOp::GuardLoopContinue {
                    condition_register: 3,
                    expect_truthy: true,
                    bailout_ip: 10,
                },
                TraceOp::Move { dest: 3, src: 2 },
            ]
        ));
    }

    #[test]
    fn type_guards_are_removed_only_while_register_types_are_known() {
        use crate::jit::trace::ValueType;

        let mut trace = Trace {
            function_idx: 0,
            start_ip: 0,
            is_function: false,
            frame_may_own: true,
            entry_scalars: Vec::new(),
            alias_params: 0,
            borrowed_registers: Vec::new(),
            preamble: Vec::new(),
            ops: vec![
                TraceOp::Guard {
                    register: 0,
                    expected_type: ValueType::Int,
                },
                TraceOp::Add {
                    dest: 2,
                    lhs: 0,
                    rhs: 1,
                    lhs_type: ValueType::Int,
                    rhs_type: ValueType::Int,
                },
                TraceOp::Guard {
                    register: 2,
                    expected_type: ValueType::Int,
                },
                TraceOp::Guard {
                    register: 0,
                    expected_type: ValueType::Int,
                },
                TraceOp::CallMethod {
                    dest: 2,
                    object: 3,
                    method_name: "value".to_string(),
                    first_arg: 0,
                    arg_count: 0,
                },
                TraceOp::Guard {
                    register: 2,
                    expected_type: ValueType::Int,
                },
            ],
            postamble: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        };

        TraceOptimizer::new().eliminate_redundant_type_guards(&mut trace);

        let guarded_registers: Vec<_> = trace
            .ops
            .iter()
            .filter_map(|op| match op {
                TraceOp::Guard { register, .. } => Some(*register),
                _ => None,
            })
            .collect();
        assert_eq!(guarded_registers, vec![0, 2]);
    }
}

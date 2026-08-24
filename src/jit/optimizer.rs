use crate::bytecode::{Register, Value};
use crate::jit;
use crate::jit::trace::{Trace, TraceOp};
use alloc::{format, string::ToString, vec::Vec};
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
        self.fuse_try_get_index_patterns(trace);
        self.fuse_try_cast_patterns(trace);
        self.hoist_constants(trace);
        self.unroll_loop(trace, crate::jit::UNROLL_FACTOR);
        self.eliminate_arithmetic_moves(trace);
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

    /// A checked index immediately matched as `Ok(value)` can keep its
    /// discriminant and payload in registers instead of allocating a Result.
    fn fuse_try_get_index_patterns(&mut self, trace: &mut Trace) {
        let mut ops = Vec::with_capacity(trace.ops.len());
        let mut i = 0;
        while i < trace.ops.len() {
            if i + 3 < trace.ops.len() {
                if let (
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
                ) = (
                    &trace.ops[i],
                    &trace.ops[i + 1],
                    &trace.ops[i + 2],
                    &trace.ops[i + 3],
                ) {
                    if tested_reg == result_reg
                        && enum_reg == result_reg
                        && condition_register == condition_reg
                        && enum_name == "Result"
                        && variant_name == "Ok"
                    {
                        ops.push(TraceOp::ArrayIndexOk {
                            value_dest: *result_reg,
                            condition_dest: *condition_reg,
                            array: *array,
                            index: *index,
                        });
                        ops.push(trace.ops[i + 2].clone());
                        ops.push(TraceOp::Move {
                            dest: *binding_reg,
                            src: *result_reg,
                        });
                        i += 4;
                        continue;
                    }
                }
            }

            ops.push(trace.ops[i].clone());
            i += 1;
        }
        trace.ops = ops;
    }

    /// `value as T is Some(x)` lowers through an Option so the value can escape
    /// when needed. In the common immediate-pattern form, constructing that
    /// Option only to test and unpack it is redundant: the type test is the
    /// discriminant, and the successful payload is the original value.
    fn fuse_try_cast_patterns(&mut self, trace: &mut Trace) {
        let mut ops = Vec::with_capacity(trace.ops.len());
        let mut i = 0;
        while i < trace.ops.len() {
            if i + 3 < trace.ops.len() {
                if let (
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
                ) = (
                    &trace.ops[i],
                    &trace.ops[i + 1],
                    &trace.ops[i + 2],
                    &trace.ops[i + 3],
                ) {
                    if tested_reg == option_reg
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
                        ops.push(trace.ops[i + 2].clone());
                        ops.push(TraceOp::Move {
                            dest: *binding_reg,
                            src: *value,
                        });
                        i += 4;
                        continue;
                    }
                }
            }

            ops.push(trace.ops[i].clone());
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
            | TraceOp::InlineCall { dest, .. } => Some(*dest),
            TraceOp::ArrayIndexOk { .. } => None,
            _ => None,
        }
    }

    fn eliminate_arithmetic_moves(&mut self, trace: &mut Trace) {
        let mut new_ops = Vec::new();
        let mut i = 0;
        while i < trace.ops.len() {
            if i + 1 < trace.ops.len() {
                let current = &trace.ops[i];
                let next = &trace.ops[i + 1];
                if let Some((_, final_dest)) = self.match_arithmetic_move(current, next) {
                    let mut rewritten = current.clone();
                    self.rewrite_arithmetic_dest(&mut rewritten, final_dest);
                    new_ops.push(rewritten);
                    i += 2;
                    continue;
                }
            }

            new_ops.push(trace.ops[i].clone());
            i += 1;
        }

        trace.ops = new_ops;
    }

    fn match_arithmetic_move(&self, op1: &TraceOp, op2: &TraceOp) -> Option<(Register, Register)> {
        let arith_dest = match op1 {
            TraceOp::Add { dest, .. }
            | TraceOp::Sub { dest, .. }
            | TraceOp::Mul { dest, .. }
            | TraceOp::Div { dest, .. }
            | TraceOp::Mod { dest, .. } => *dest,
            _ => return None,
        };
        if let TraceOp::Move {
            dest: move_dest,
            src,
        } = op2
        {
            if *src == arith_dest {
                return Some((arith_dest, *move_dest));
            }
        }

        None
    }

    fn rewrite_arithmetic_dest(&self, op: &mut TraceOp, new_dest: Register) {
        match op {
            TraceOp::Add { dest, .. }
            | TraceOp::Sub { dest, .. }
            | TraceOp::Mul { dest, .. }
            | TraceOp::Div { dest, .. }
            | TraceOp::Mod { dest, .. } => {
                *dest = new_dest;
            }

            _ => {}
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
            _ => None,
        }
    }

    fn unroll_loop(&mut self, trace: &mut Trace, factor: usize) {
        if factor <= 1 || trace.ops.is_empty() {
            return;
        }

        if trace
            .ops
            .iter()
            .any(|op| matches!(op, TraceOp::InlineCall { .. }))
        {
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

    fn coalesce_registers(&mut self, _trace: &mut Trace) {}
}

impl Default for TraceOptimizer {
    fn default() -> Self {
        Self::new()
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

use alloc::{format, string::ToString, vec::Vec};
use crate::bytecode::{Register, Value};
use crate::jit;
use crate::jit::trace::{Trace, TraceOp};
use hashbrown::HashSet;
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
        self.hoist_constants(trace);
        self.unroll_loop(trace, crate::jit::UNROLL_FACTOR);
        self.eliminate_arithmetic_moves(trace);
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

    fn hoist_constants(&mut self, trace: &mut Trace) {
        let mut non_hoistable: HashSet<Register> = HashSet::new();
        for op in &trace.ops {
            if let TraceOp::CallNative { callee, .. } = op {
                non_hoistable.insert(*callee);
            }
        }

        let mut new_ops = Vec::new();
        let mut already_hoisted: HashSet<Register> = HashSet::new();
        let mut assigned: HashSet<Register> = HashSet::new();

        let dest_of = |op: &TraceOp| -> Option<Register> {
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
                | TraceOp::ArrayLen { dest, .. }
                | TraceOp::CallMethod { dest, .. }
                | TraceOp::GetField { dest, .. }
                | TraceOp::NewStruct { dest, .. }
                | TraceOp::CallNative { dest, .. } => Some(*dest),
                _ => None,
            }
        };

        for op in trace.ops.drain(..) {
            match op {
                TraceOp::LoadConst { dest, value } => {
                    if non_hoistable.contains(&dest)
                        || already_hoisted.contains(&dest)
                        || assigned.contains(&dest)
                    {
                        new_ops.push(TraceOp::LoadConst { dest, value });
                    } else {
                        self.hoisted_constants.push((dest, value));
                        already_hoisted.insert(dest);
                    }
                    assigned.insert(dest);
                }

                other => {
                    if let Some(dest) = dest_of(&other) {
                        assigned.insert(dest);
                    }
                    new_ops.push(other);
                }
            }
        }

        trace.ops = new_ops;
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

    fn unroll_loop(&mut self, trace: &mut Trace, factor: usize) {
        if factor <= 1 || trace.ops.is_empty() {
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

        let (loop_cmp_op, cond_reg) = loop_condition_op.unwrap();
        let original_ops = trace.ops.clone();
        let mut new_ops = Vec::new();
        new_ops.extend(original_ops.iter().cloned());
        for _ in 1..factor {
            new_ops.push(loop_cmp_op.clone());
            new_ops.push(TraceOp::GuardLoopContinue {
                condition_register: cond_reg,
                bailout_ip: trace.start_ip,
            });
            for op in &original_ops {
                if !matches!(
                    op,
                    TraceOp::Le { .. }
                        | TraceOp::Lt { .. }
                        | TraceOp::Ge { .. }
                        | TraceOp::Gt { .. }
                ) {
                    new_ops.push(op.clone());
                }
            }
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

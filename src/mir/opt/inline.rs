use crate::ast::Span;
use crate::mir::ir::*;
use alloc::{format, string::String, vec::Vec};
use hashbrown::HashMap;

/// Configuration for MIR-to-MIR inlining pass.
#[derive(Debug, Clone)]
pub struct InlinerConfig {
    /// Maximum depth of inlining along any call path.
    pub max_depth: usize,
    /// Maximum basic blocks allowed in an inlinable callee.
    pub max_callee_blocks: usize,
    /// Maximum statement count allowed in an inlinable callee.
    pub max_callee_statements: usize,
}

impl Default for InlinerConfig {
    fn default() -> Self {
        Self {
            max_depth: 4,
            max_callee_blocks: 64,
            max_callee_statements: 256,
        }
    }
}

/// MIR Inliner Pass: Performs interprocedural inlining of eligible function calls directly in Lust MIR.
pub struct MirInliner {
    config: InlinerConfig,
}

impl Default for MirInliner {
    fn default() -> Self {
        Self::new()
    }
}

impl MirInliner {
    pub fn new() -> Self {
        Self {
            config: InlinerConfig::default(),
        }
    }

    pub fn with_config(config: InlinerConfig) -> Self {
        Self { config }
    }

    /// Convenience static entrypoint to run inlining on a full MIR module until fixpoint.
    pub fn run_on_module(module: &mut MirModule) -> bool {
        Self::new().inline_module(module)
    }

    /// Runs inlining across all functions in `module`.
    pub fn inline_module(&self, module: &mut MirModule) -> bool {
        // Build a lookup table of candidate callees from clean module functions
        let callees: HashMap<String, MirFunction> = module
            .functions
            .iter()
            .map(|f| (f.signature.name.clone(), f.clone()))
            .collect();

        let mut overall_changed = false;
        for func in &mut module.functions {
            overall_changed |= self.inline_function(func, &callees);
        }

        overall_changed
    }

    /// Inlines eligible calls inside a single `caller` function using the provided `callees` registry.
    pub fn inline_function(
        &self,
        caller: &mut MirFunction,
        callees: &HashMap<String, MirFunction>,
    ) -> bool {
        let mut block_stacks: Vec<Vec<String>> =
            vec![vec![caller.signature.name.clone()]; caller.blocks.len()];
        self.inline_function_with_stacks(caller, callees, &mut block_stacks)
    }

    fn inline_function_with_stacks(
        &self,
        caller: &mut MirFunction,
        callees: &HashMap<String, MirFunction>,
        block_stacks: &mut Vec<Vec<String>>,
    ) -> bool {
        let mut changed = false;
        let mut block_idx = 0;

        // Iterate over blocks by index (blocks list grows as new inlined blocks are added)
        while block_idx < caller.blocks.len() {
            let term = caller.blocks[block_idx].terminator.clone();
            if let Some(Terminator::Call {
                callee,
                args,
                destination,
                success,
                error,
                span,
            }) = term
            {
                if let Some(callee_fn) = callees.get(&callee) {
                    let current_stack = if block_idx < block_stacks.len() {
                        block_stacks[block_idx].clone()
                    } else {
                        vec![caller.signature.name.clone()]
                    };

                    if self.is_eligible_callee(callee_fn, &caller.signature.name, &current_stack)
                        && callee_fn.params.len() == args.len()
                    {
                        let mut next_stack = current_stack.clone();
                        next_stack.push(callee.clone());

                        let added_blocks = self.perform_inline(
                            caller,
                            block_idx,
                            callee_fn,
                            &args,
                            destination,
                            success,
                            error,
                            span,
                        );

                        for _ in 0..added_blocks {
                            block_stacks.push(next_stack.clone());
                        }

                        changed = true;
                        // Continue loop: newly added inlined blocks will be scanned for further calls.
                    }
                }
            }
            block_idx += 1;
        }

        if changed {
            while Self::simplify_cfg(caller) {}
            for (i, block) in caller.blocks.iter_mut().enumerate() {
                block.id = BlockId(i);
            }
        }

        changed
    }

    /// Evaluates eligibility of `callee` for inlining:
    /// - Guards against direct recursion (callee calls caller or callee calls callee)
    /// - Guards against mutual recursion via active `call_stack` cycle detection
    /// - Guards against excessive inlining depth
    /// - Guards against oversized functions (callee blocks/statements limits)
    pub fn is_eligible_callee(
        &self,
        callee: &MirFunction,
        caller_name: &str,
        call_stack: &[String],
    ) -> bool {
        // Direct recursion: callee is caller
        if callee.signature.name == caller_name {
            return false;
        }

        // Mutual recursion / cycle in active stack
        if call_stack.contains(&callee.signature.name) {
            return false;
        }

        // Depth budget exceeded (call_stack starts with [caller], so inlining depth is call_stack.len() - 1)
        if call_stack.len() > self.config.max_depth {
            return false;
        }

        // Must have at least one block and a valid entry
        if callee.blocks.is_empty() || callee.entry.0 >= callee.blocks.len() {
            return false;
        }

        // Threshold checks
        if callee.blocks.len() > self.config.max_callee_blocks {
            return false;
        }

        let total_stmts: usize = callee.blocks.iter().map(|b| b.statements.len()).sum();
        if total_stmts > self.config.max_callee_statements {
            return false;
        }

        // Self-recursion check: callee contains a Call to itself
        for block in &callee.blocks {
            if let Some(Terminator::Call {
                callee: inner_callee,
                ..
            }) = &block.terminator
            {
                if inner_callee == &callee.signature.name {
                    return false;
                }
            }
        }

        true
    }

    /// Performs the splicing of `callee` into `caller` at `call_block_idx`.
    /// Returns the number of newly added basic blocks.
    fn perform_inline(
        &self,
        caller: &mut MirFunction,
        call_block_idx: usize,
        callee: &MirFunction,
        call_args: &[Operand],
        destination: Place,
        succ_bb: BlockId,
        err_bb: Option<BlockId>,
        call_span: Span,
    ) -> usize {
        // 1. Allocate fresh locals in caller for every local in callee
        let mut local_map: HashMap<LocalId, LocalId> = HashMap::new();
        for decl in &callee.locals {
            let mapped_name = decl
                .name
                .as_ref()
                .map(|n| format!("{}_{}", callee.signature.name, n));
            let new_local_id =
                caller.add_local(mapped_name, decl.ty.clone(), decl.is_mut, decl.span);
            local_map.insert(decl.id, new_local_id);
        }

        // Special case: Single-block callee can be inlined directly into the call block
        if callee.blocks.len() == 1 {
            let c_block = &callee.blocks[0];

            // Assign parameters directly in call block
            for (i, p_id) in callee.params.iter().enumerate() {
                let mapped_param = local_map[p_id];
                let arg_op = call_args[i].clone();

                caller.blocks[call_block_idx]
                    .statements
                    .push(Statement::StorageLive(mapped_param, call_span));
                caller.blocks[call_block_idx]
                    .statements
                    .push(Statement::Assign(
                        Place::Local(mapped_param),
                        Rvalue::Use(arg_op),
                        call_span,
                    ));
            }

            // Append callee statements directly to call block
            for stmt in &c_block.statements {
                caller.blocks[call_block_idx]
                    .statements
                    .push(remap_statement(stmt, &local_map));
            }

            // Handle terminator directly
            match &c_block.terminator {
                Some(Terminator::Return(opt_ret, s)) => {
                    if let Some(ret_op) = opt_ret {
                        caller.blocks[call_block_idx].statements.push(Statement::Assign(
                            destination,
                            Rvalue::Use(remap_operand(ret_op, &local_map)),
                            *s,
                        ));
                    }
                    caller.blocks[call_block_idx].terminator = Some(Terminator::Goto(succ_bb, *s));
                }
                Some(Terminator::Raise(err_op, s)) => {
                    if let Some(err_target) = err_bb {
                        caller.blocks[call_block_idx].terminator =
                            Some(Terminator::Goto(err_target, *s));
                    } else {
                        caller.blocks[call_block_idx].terminator =
                            Some(Terminator::Raise(remap_operand(err_op, &local_map), *s));
                    }
                }
                Some(Terminator::Unreachable(s)) => {
                    caller.blocks[call_block_idx].terminator = Some(Terminator::Unreachable(*s));
                }
                _ => {
                    caller.blocks[call_block_idx].terminator =
                        Some(Terminator::Goto(succ_bb, call_span));
                }
            }

            return 0;
        }

        // General multi-block inlining
        // 2. Prepend parameter assignments to caller's call block before the jump
        for (i, p_id) in callee.params.iter().enumerate() {
            let mapped_param = local_map[p_id];
            let arg_op = call_args[i].clone();

            caller.blocks[call_block_idx]
                .statements
                .push(Statement::StorageLive(mapped_param, call_span));
            caller.blocks[call_block_idx]
                .statements
                .push(Statement::Assign(
                    Place::Local(mapped_param),
                    Rvalue::Use(arg_op),
                    call_span,
                ));
        }

        // 3. Allocate fresh basic blocks in caller for every basic block in callee
        let mut block_map: HashMap<BlockId, BlockId> = HashMap::new();
        for c_block in &callee.blocks {
            let new_bb = caller.new_block();
            block_map.insert(c_block.id, new_bb);
        }

        // 4. Update the call block's terminator: Goto mapped callee entry block
        let mapped_entry = block_map[&callee.entry];
        caller.blocks[call_block_idx].terminator =
            Some(Terminator::Goto(mapped_entry, call_span));

        // 5. Populate caller's newly allocated blocks with mapped statements and terminators
        for c_block in &callee.blocks {
            let target_bb_id = block_map[&c_block.id];
            let mut stmts = Vec::with_capacity(c_block.statements.len() + 1);

            for stmt in &c_block.statements {
                stmts.push(remap_statement(stmt, &local_map));
            }

            let term = match &c_block.terminator {
                Some(Terminator::Goto(target, s)) => {
                    Some(Terminator::Goto(block_map[target], *s))
                }
                Some(Terminator::Cond {
                    cond,
                    then_block,
                    else_block,
                    span: s,
                }) => Some(Terminator::Cond {
                    cond: remap_operand(cond, &local_map),
                    then_block: block_map[then_block],
                    else_block: block_map[else_block],
                    span: *s,
                }),
                Some(Terminator::Switch {
                    discr,
                    cases,
                    default,
                    span: s,
                }) => Some(Terminator::Switch {
                    discr: remap_operand(discr, &local_map),
                    cases: cases
                        .iter()
                        .map(|(val, target)| (*val, block_map[target]))
                        .collect(),
                    default: block_map[default],
                    span: *s,
                }),
                Some(Terminator::Call {
                    callee: sub_c,
                    args: sub_a,
                    destination: sub_d,
                    success: sub_s,
                    error: sub_e,
                    span: s,
                }) => Some(Terminator::Call {
                    callee: sub_c.clone(),
                    args: sub_a
                        .iter()
                        .map(|a| remap_operand(a, &local_map))
                        .collect(),
                    destination: remap_place(sub_d, &local_map),
                    success: block_map[sub_s],
                    error: sub_e.map(|e| block_map[&e]),
                    span: *s,
                }),
                Some(Terminator::Return(opt_ret, s)) => {
                    // Return from callee: assign return value to caller destination, then jump to success block
                    if let Some(ret_op) = opt_ret {
                        stmts.push(Statement::Assign(
                            destination.clone(),
                            Rvalue::Use(remap_operand(ret_op, &local_map)),
                            *s,
                        ));
                    }
                    Some(Terminator::Goto(succ_bb, *s))
                }
                Some(Terminator::Raise(err_op, s)) => {
                    if let Some(err_target) = err_bb {
                        Some(Terminator::Goto(err_target, *s))
                    } else {
                        Some(Terminator::Raise(remap_operand(err_op, &local_map), *s))
                    }
                }
                Some(Terminator::Unreachable(s)) => Some(Terminator::Unreachable(*s)),
                None => None,
            };

            caller.blocks[target_bb_id.0].statements = stmts;
            caller.blocks[target_bb_id.0].terminator = term;
        }

        callee.blocks.len()
    }

    /// Simplifies the CFG by fusing linear Goto chains where the target has in-degree 1.
    pub fn simplify_cfg(func: &mut MirFunction) -> bool {
        let mut in_degrees = vec![0usize; func.blocks.len()];
        in_degrees[func.entry.0] = 1;
        for block in &func.blocks {
            if let Some(term) = &block.terminator {
                for succ in terminator_successors(term) {
                    if succ.0 < in_degrees.len() {
                        in_degrees[succ.0] += 1;
                    }
                }
            }
        }

        let mut changed = false;
        for i in 0..func.blocks.len() {
            if let Some(Terminator::Goto(target, _)) = func.blocks[i].terminator {
                if target.0 != i
                    && target.0 < func.blocks.len()
                    && in_degrees[target.0] == 1
                    && target != func.entry
                {
                    let target_stmts = func.blocks[target.0].statements.clone();
                    let target_term = func.blocks[target.0].terminator.clone();

                    func.blocks[i].statements.extend(target_stmts);
                    func.blocks[i].terminator = target_term;

                    func.blocks[target.0].statements.clear();
                    func.blocks[target.0].terminator =
                        Some(Terminator::Unreachable(Span::dummy()));

                    changed = true;
                }
            }
        }

        if changed {
            crate::mir::opt::DeadCodeElimination::run_on_function(func);
        }
        for (i, block) in func.blocks.iter_mut().enumerate() {
            block.id = BlockId(i);
        }

        changed
    }
}

fn terminator_successors(term: &Terminator) -> Vec<BlockId> {
    match term {
        Terminator::Goto(target, _) => alloc::vec![*target],
        Terminator::Cond {
            then_block,
            else_block,
            ..
        } => alloc::vec![*then_block, *else_block],
        Terminator::Switch {
            cases, default, ..
        } => {
            let mut succs: Vec<BlockId> = cases.iter().map(|(_, target)| *target).collect();
            succs.push(*default);
            succs
        }
        Terminator::Call {
            success, error, ..
        } => {
            let mut succs = alloc::vec![*success];
            if let Some(err_bb) = error {
                succs.push(*err_bb);
            }
            succs
        }
        Terminator::Return(_, _) | Terminator::Raise(_, _) | Terminator::Unreachable(_) => {
            alloc::vec![]
        }
    }
}

// ---------------------------------------------------------------------------
// Remapping helpers for LocalId translation
// ---------------------------------------------------------------------------

fn remap_place(place: &Place, map: &HashMap<LocalId, LocalId>) -> Place {
    match place {
        Place::Local(id) => Place::Local(*map.get(id).unwrap_or(id)),
        Place::Field(base, idx) => Place::Field(alloc::boxed::Box::new(remap_place(base, map)), *idx),
        Place::Index(base, idx) => Place::Index(
            alloc::boxed::Box::new(remap_place(base, map)),
            *map.get(idx).unwrap_or(idx),
        ),
    }
}

fn remap_operand(op: &Operand, map: &HashMap<LocalId, LocalId>) -> Operand {
    match op {
        Operand::Copy(p) => Operand::Copy(remap_place(p, map)),
        Operand::Move(p) => Operand::Move(remap_place(p, map)),
        Operand::Constant(c) => Operand::Constant(c.clone()),
    }
}

fn remap_rvalue(rval: &Rvalue, map: &HashMap<LocalId, LocalId>) -> Rvalue {
    match rval {
        Rvalue::Use(op) => Rvalue::Use(remap_operand(op, map)),
        Rvalue::BinaryOp(binop, l, r) => {
            Rvalue::BinaryOp(*binop, remap_operand(l, map), remap_operand(r, map))
        }
        Rvalue::UnaryOp(unop, op) => Rvalue::UnaryOp(*unop, remap_operand(op, map)),
        Rvalue::Cast(op, ty) => Rvalue::Cast(remap_operand(op, map), ty.clone()),
        Rvalue::Aggregate(ty, ops) => {
            Rvalue::Aggregate(ty.clone(), ops.iter().map(|o| remap_operand(o, map)).collect())
        }
    }
}

fn remap_statement(stmt: &Statement, map: &HashMap<LocalId, LocalId>) -> Statement {
    match stmt {
        Statement::Assign(p, r, span) => {
            Statement::Assign(remap_place(p, map), remap_rvalue(r, map), *span)
        }
        Statement::StorageLive(id, span) => {
            Statement::StorageLive(*map.get(id).unwrap_or(id), *span)
        }
        Statement::StorageDead(id, span) => {
            Statement::StorageDead(*map.get(id).unwrap_or(id), *span)
        }
        Statement::ResourceCheck(k, amt, span) => Statement::ResourceCheck(*k, *amt, *span),
    }
}

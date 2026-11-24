use crate::ast::Span;
use crate::data::layout::CanonicalTypeTag;
use crate::mir::ir::*;
use alloc::{boxed::Box, format, string::String, vec::Vec};
use hashbrown::{HashMap, HashSet};

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

/// Configuration for JIT-safe MIR-to-MIR inlining pass (`PERF-T1-4` / Hazard 3 Step A).
#[derive(Debug, Clone)]
pub struct JitInlineConfig {
    /// Base inliner limits (depth, callee blocks, callee statements).
    pub base: InlinerConfig,
    /// Functions that must NEVER be inlined INTO or OUT OF
    /// (e.g., `test_deopt_spike_hook.target_function`).
    pub excluded_functions: HashSet<String>,
    /// Maximum total locals allowed in a caller after inlining a callee.
    /// Keeps MIR-VM continuation bytecode register indices safely within `u8` (`< 256` with scratch headroom).
    pub max_caller_locals: usize,
}

impl Default for JitInlineConfig {
    fn default() -> Self {
        Self {
            base: InlinerConfig::default(),
            excluded_functions: HashSet::new(),
            max_caller_locals: 200,
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
        let struct_layouts = collect_module_struct_layouts(module);

        let mut overall_changed = false;
        for func in &mut module.functions {
            overall_changed |=
                self.inline_function_with_layouts(func, &callees, &struct_layouts);
        }

        overall_changed
    }

    /// Inlines eligible calls inside a single `caller` function using the provided `callees` registry.
    pub fn inline_function(
        &self,
        caller: &mut MirFunction,
        callees: &HashMap<String, MirFunction>,
    ) -> bool {
        let mut struct_layouts = HashMap::new();
        register_fn_struct_layouts(&mut struct_layouts, caller);
        for f in callees.values() {
            register_fn_struct_layouts(&mut struct_layouts, f);
        }
        self.inline_function_with_layouts(caller, callees, &struct_layouts)
    }

    fn inline_function_with_layouts(
        &self,
        caller: &mut MirFunction,
        callees: &HashMap<String, MirFunction>,
        struct_layouts: &HashMap<String, MirStructLayout>,
    ) -> bool {
        let mut block_stacks: Vec<Vec<String>> =
            vec![vec![caller.signature.name.clone()]; caller.blocks.len()];
        self.inline_function_with_stacks(caller, callees, struct_layouts, &mut block_stacks)
    }

    fn inline_function_with_stacks(
        &self,
        caller: &mut MirFunction,
        callees: &HashMap<String, MirFunction>,
        struct_layouts: &HashMap<String, MirStructLayout>,
        block_stacks: &mut Vec<Vec<String>>,
    ) -> bool {
        let mut changed = false;

        for _round in 0..8 {
            let mut round_changed = false;

            // Pre-pass: specialize direct non-mutating closures & devirtualize known struct calls
            if self.specialize_caller_calls(
                caller,
                callees,
                struct_layouts,
                block_stacks,
                None,
                None,
                None,
            ) {
                round_changed = true;
                changed = true;
            }

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

                            round_changed = true;
                            changed = true;
                            // Respecialize newly exposed unknown/closure calls immediately so chained
                            // callees (e.g. eval_poly -> NodeA:score, apply_twice -> step_fn) inline.
                            if self.specialize_caller_calls(
                                caller,
                                callees,
                                struct_layouts,
                                block_stacks,
                                None,
                                None,
                                None,
                            ) {
                                round_changed = true;
                            }
                        }
                    }
                }
                block_idx += 1;
            }

            if !round_changed {
                break;
            }
        }

        if changed {
            remove_dead_closure_stmts(caller);
            while Self::simplify_cfg(caller) {}
            renumber_blocks_rpo(caller);
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

    /// Evaluates whether a loop-bearing `callee` is a small single-loop leaf function safe to inline
    /// into a loop call site (`NG-W15-2` / `NG-W16-5`):
    /// - `callee.blocks.len() <= 32` and total non-storage statements across all blocks `<= 128`.
    /// - Exactly one dominance-verified loop header (`Safepoints::find_loop_headers_in_fn(callee).len() == 1`).
    /// - Zero `cleanup_blocks`, zero `is_cleanup` blocks, zero `Rvalue::Closure` statements.
    /// - Every block terminator is `Goto`, `Cond`, `Return`, or a primitive leaf builtin `Call`
    ///   (e.g. `array.len`, `table.len`, `string.len`, `string.sub`, `string.split`, `string.eq`,
    ///   `math.*`) with `error: None` — zero user calls.
    pub fn is_eligible_small_loop_leaf_callee(callee: &MirFunction) -> bool {
        if callee.blocks.is_empty() || callee.entry.0 >= callee.blocks.len() {
            return false;
        }
        if callee.blocks.len() > 32 {
            return false;
        }
        let exec_stmts: usize = callee
            .blocks
            .iter()
            .flat_map(|b| b.statements.iter())
            .filter(|s| !matches!(s, Statement::StorageLive(..) | Statement::StorageDead(..)))
            .count();
        if exec_stmts > 128 {
            return false;
        }
        if !callee.cleanup_blocks.is_empty() || callee.blocks.iter().any(|b| b.is_cleanup) {
            return false;
        }
        if crate::mir::opt::Safepoints::find_loop_headers_in_fn(callee).len() != 1 {
            return false;
        }
        for block in &callee.blocks {
            for stmt in &block.statements {
                if matches!(stmt, Statement::Assign(_, Rvalue::Closure { .. }, _)) {
                    return false;
                }
            }
            match &block.terminator {
                Some(Terminator::Goto(..))
                | Some(Terminator::Cond { .. })
                | Some(Terminator::Return(..)) => {}
                Some(Terminator::Call {
                    callee: inner,
                    error: None,
                    ..
                }) if is_primitive_leaf_builtin_callee(inner) => {}
                _ => return false,
            }
        }
        true
    }

    fn is_eligible_small_loop_callee_with_helpers(
        callee: &MirFunction,
        callees: &HashMap<String, MirFunction>,
    ) -> bool {
        if Self::is_eligible_small_loop_leaf_callee(callee) {
            return true;
        }
        if callee.blocks.is_empty() || callee.entry.0 >= callee.blocks.len() {
            return false;
        }
        if callee.blocks.len() > 32 {
            return false;
        }
        let exec_stmts: usize = callee
            .blocks
            .iter()
            .flat_map(|b| b.statements.iter())
            .filter(|s| !matches!(s, Statement::StorageLive(..) | Statement::StorageDead(..)))
            .count();
        if exec_stmts > 128 {
            return false;
        }
        if !callee.cleanup_blocks.is_empty() || callee.blocks.iter().any(|b| b.is_cleanup) {
            return false;
        }
        if crate::mir::opt::Safepoints::find_loop_headers_in_fn(callee).len() != 1 {
            return false;
        }
        for block in &callee.blocks {
            for stmt in &block.statements {
                if matches!(stmt, Statement::Assign(_, Rvalue::Closure { .. }, _)) {
                    return false;
                }
            }
            match &block.terminator {
                Some(Terminator::Goto(..))
                | Some(Terminator::Cond { .. })
                | Some(Terminator::Return(..)) => {}
                Some(Terminator::Call {
                    callee: inner,
                    error: None,
                    ..
                }) if is_primitive_leaf_builtin_callee(inner)
                    || (inner != &callee.signature.name
                        && callees
                            .get(inner)
                            .is_some_and(Self::is_eligible_small_loop_leaf_callee)) => {}
                _ => return false,
            }
        }
        true
    }

    /// Evaluates JIT-safe eligibility of `callee` for inlining (`PERF-T1-4` / `NG-W15-2`):
    /// - Enforces all base `is_eligible_callee` guards (direct recursion, cycle in `call_stack`,
    ///   `max_depth`, valid entry, `max_callee_blocks`, `max_callee_statements`, self-recursion).
    /// - Guard 1: Never inline a callee containing any `is_effect_or_handler_callee`.
    /// - Guard 2: Loop-bearing callees are rejected unless invoked at an in-loop call site and
    ///   satisfying `is_eligible_small_loop_leaf_callee` (see `is_eligible_callee_for_jit_at_site`).
    /// - Guard 3: Never inline INTO or OUT OF any function in `jit_config.excluded_functions`.
    /// - Rejects `Terminator::Raise` callees and callers exceeding `max_caller_locals` (keeps
    ///   MIR-VM continuation bytecode register indices safely within `u8`).
    pub fn is_eligible_callee_for_jit(
        &self,
        callee: &MirFunction,
        caller_name: &str,
        caller_local_count: usize,
        call_stack: &[String],
        jit_config: &JitInlineConfig,
    ) -> bool {
        self.is_eligible_callee_for_jit_at_site(
            callee,
            caller_name,
            caller_local_count,
            call_stack,
            jit_config,
            false,
            None,
        )
    }

    fn is_eligible_callee_for_jit_at_site(
        &self,
        callee: &MirFunction,
        caller_name: &str,
        caller_local_count: usize,
        call_stack: &[String],
        jit_config: &JitInlineConfig,
        call_site_in_loop: bool,
        callees: Option<&HashMap<String, MirFunction>>,
    ) -> bool {
        if jit_config.excluded_functions.contains(caller_name)
            || jit_config.excluded_functions.contains(&callee.signature.name)
        {
            return false;
        }
        if !self.is_eligible_callee(callee, caller_name, call_stack) {
            return false;
        }
        if callee_has_effect_or_handler_call(callee) {
            return false;
        }
        let small_loop_ok = match callees {
            Some(map) => Self::is_eligible_small_loop_callee_with_helpers(callee, map),
            None => Self::is_eligible_small_loop_leaf_callee(callee),
        };
        if callee_has_loops(callee) && !(call_site_in_loop && small_loop_ok) {
            return false;
        }
        if callee
            .blocks
            .iter()
            .any(|b| matches!(b.terminator, Some(Terminator::Raise(..))))
        {
            return false;
        }
        if caller_local_count.saturating_add(callee.locals.len()) > jit_config.max_caller_locals {
            return false;
        }
        true
    }

    /// Runs JIT-safe inlining across all functions in `module`.
    ///
    /// Enforces all Hazard 3 Step A guards:
    /// - Skips direct and mutual recursion cycles (via call-graph reachability + stack cycle checks).
    /// - Guard 1: Never inlines effect/handler callees (directly or transitively).
    /// - Guard 2: Never inlines loop-bearing callees.
    /// - Guard 3: Never inlines INTO or OUT OF `jit_config.excluded_functions`.
    /// - Guard 4: Verifies every modified `MirFunction` with `MirVerifier::verify_fn`, reverting
    ///   to its pre-inline snapshot on any verification failure (fail-closed).
    ///
    /// Returns the set of function names that received verified inlined code.
    pub fn inline_module_for_jit(
        module: &mut MirModule,
        jit_config: &JitInlineConfig,
    ) -> HashSet<String> {
        let inliner = Self::with_config(jit_config.base.clone());
        let callees: HashMap<String, MirFunction> = module
            .functions
            .iter()
            .map(|f| (f.signature.name.clone(), f.clone()))
            .collect();
        let struct_layouts = collect_module_struct_layouts(module);

        let recursive_fns = compute_recursive_functions(&callees);
        let transitive_effect_fns = compute_transitive_effect_functions(&callees);

        let mut inlined_functions = HashSet::new();
        for func in &mut module.functions {
            if jit_config.excluded_functions.contains(&func.signature.name) {
                continue;
            }
            let pre_inline = func.clone();
            let changed = inliner.inline_function_for_jit(
                func,
                &callees,
                &struct_layouts,
                jit_config,
                &recursive_fns,
                &transitive_effect_fns,
            );
            if changed {
                if crate::mir::verifier::MirVerifier::verify_fn(func).is_err() {
                    *func = pre_inline;
                } else {
                    inlined_functions.insert(func.signature.name.clone());
                }
            }
        }

        inlined_functions
    }

    fn inline_function_for_jit(
        &self,
        caller: &mut MirFunction,
        callees: &HashMap<String, MirFunction>,
        struct_layouts: &HashMap<String, MirStructLayout>,
        jit_config: &JitInlineConfig,
        recursive_fns: &HashSet<String>,
        transitive_effect_fns: &HashSet<String>,
    ) -> bool {
        if jit_config.excluded_functions.contains(&caller.signature.name) {
            return false;
        }
        let mut block_stacks: Vec<Vec<String>> =
            vec![vec![caller.signature.name.clone()]; caller.blocks.len()];
        let mut caller_loop_blocks = compute_natural_loop_blocks(caller);
        let mut changed = false;

        for _round in 0..8 {
            let mut round_changed = false;

            if self.specialize_caller_calls(
                caller,
                callees,
                struct_layouts,
                &mut block_stacks,
                Some(jit_config),
                Some(recursive_fns),
                Some(transitive_effect_fns),
            ) {
                caller_loop_blocks = compute_natural_loop_blocks(caller);
                round_changed = true;
                changed = true;
            }

            let mut block_idx = 0;
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
                    if !recursive_fns.contains(&callee)
                        && !transitive_effect_fns.contains(&callee)
                        && let Some(callee_fn) = callees.get(&callee)
                    {
                        let current_stack = if block_idx < block_stacks.len() {
                            block_stacks[block_idx].clone()
                        } else {
                            vec![caller.signature.name.clone()]
                        };
                        let call_site_in_loop = caller_loop_blocks.contains(&BlockId(block_idx));

                        if self.is_eligible_callee_for_jit_at_site(
                            callee_fn,
                            &caller.signature.name,
                            caller.locals.len(),
                            &current_stack,
                            jit_config,
                            call_site_in_loop,
                            Some(callees),
                        ) && callee_fn.params.len() == args.len()
                        {
                            let mut next_stack = current_stack;
                            next_stack.push(callee);

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
                            caller_loop_blocks = compute_natural_loop_blocks(caller);

                            round_changed = true;
                            changed = true;

                            if self.specialize_caller_calls(
                                caller,
                                callees,
                                struct_layouts,
                                &mut block_stacks,
                                Some(jit_config),
                                Some(recursive_fns),
                                Some(transitive_effect_fns),
                            ) {
                                caller_loop_blocks = compute_natural_loop_blocks(caller);
                            }
                        }
                    }
                }
                block_idx += 1;
            }

            if !round_changed {
                break;
            }
        }

        if changed {
            remove_dead_closure_stmts(caller);
            while Self::simplify_cfg(caller) {}
            renumber_blocks_rpo(caller);
        }

        changed
    }

    /// Evaluates if `callee` has a straight-line control flow (no branches, loops, or cleanups).
    pub fn is_straight_line(callee: &MirFunction) -> bool {
        if callee.blocks.is_empty() || callee.entry.0 >= callee.blocks.len() {
            return false;
        }
        if !crate::mir::opt::Safepoints::find_loop_headers_in_fn(callee).is_empty() {
            return false;
        }
        if !callee.cleanup_blocks.is_empty() || callee.blocks.iter().any(|b| b.is_cleanup) {
            return false;
        }
        let mut in_degrees = alloc::vec![0usize; callee.blocks.len()];
        in_degrees[callee.entry.0] = 1;

        let mut return_count = 0;
        for (idx, block) in callee.blocks.iter().enumerate() {
            match &block.terminator {
                Some(Terminator::Goto(target, _)) => {
                    if target.0 >= callee.blocks.len() || target.0 == idx {
                        return false;
                    }
                    in_degrees[target.0] += 1;
                }
                Some(Terminator::Return(..)) => {
                    return_count += 1;
                }
                _ => return false,
            }
        }

        if return_count != 1 {
            return false;
        }

        for &in_deg in &in_degrees {
            if in_deg != 1 {
                return false;
            }
        }

        true
    }

    /// Performs the splicing of `callee` into `caller` at `call_block_idx` without merging basic blocks.
    /// Returns the number of newly added basic blocks.
    pub fn inline_call_site_unmerged(
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
        self.inline_call_site_unmerged_with_bound_params(
            caller,
            call_block_idx,
            callee,
            &[],
            call_args,
            destination,
            succ_bb,
            err_bb,
            call_span,
        )
    }

    /// Splices `callee` into `caller` at `call_block_idx`, mapping the first `bound_capture_locals.len()`
    /// parameters directly to existing caller locals (for persistent closure capture cells) and
    /// initializing the remaining parameters from `call_args`.
    fn inline_call_site_unmerged_with_bound_params(
        &self,
        caller: &mut MirFunction,
        call_block_idx: usize,
        callee: &MirFunction,
        bound_capture_locals: &[LocalId],
        call_args: &[Operand],
        destination: Place,
        succ_bb: BlockId,
        err_bb: Option<BlockId>,
        call_span: Span,
    ) -> usize {
        // Collect local names referenced as call targets inside `callee` (e.g. `f(x)` in `apply_twice`).
        let mut callee_called_names: HashSet<&str> = HashSet::new();
        for blk in &callee.blocks {
            if let Some(Terminator::Call { callee: sub_c, .. }) = &blk.terminator
                && !sub_c.contains('.')
                && !sub_c.contains(':')
            {
                callee_called_names.insert(sub_c.as_str());
            }
        }
        let existing_caller_names: HashSet<String> = caller
            .locals
            .iter()
            .filter_map(|l| l.name.clone())
            .collect();
        let mut allocated_names: HashSet<String> = HashSet::new();

        // Determine concrete argument types to propagate into `MirType::Unknown` unmutated params.
        let mut concrete_param_types: HashMap<LocalId, MirType> = HashMap::new();
        let n_bound = bound_capture_locals.len();
        for (arg_i, p_id) in callee.params.iter().skip(n_bound).enumerate() {
            if let Some(arg_op) = call_args.get(arg_i)
                && callee.locals[p_id.0].ty == MirType::Unknown
                && !local_is_mutated_in_fn(callee, *p_id)
                && let Some(arg_ty) = operand_mir_type_in_fn(arg_op, caller)
                && is_concrete_specializable_type(&arg_ty)
            {
                concrete_param_types.insert(*p_id, arg_ty);
            }
        }

        // 1. Map bound capture params directly to persistent caller locals; allocate fresh locals for all others.
        let mut local_map: HashMap<LocalId, LocalId> = HashMap::new();
        let mut bound_param_ids: HashSet<LocalId> = HashSet::new();
        for (i, &cap_cell) in bound_capture_locals.iter().enumerate() {
            if let Some(&p_id) = callee.params.get(i) {
                local_map.insert(p_id, cap_cell);
                bound_param_ids.insert(p_id);
            }
        }

        let mut callee_local_callee_remap: HashMap<String, String> = HashMap::new();
        for decl in &callee.locals {
            if bound_param_ids.contains(&decl.id) {
                if let Some(orig_name) = &decl.name
                    && callee_called_names.contains(orig_name.as_str())
                    && let Some(cell_name) = caller.locals[local_map[&decl.id].0].name.clone()
                {
                    callee_local_callee_remap.insert(orig_name.clone(), cell_name);
                }
                continue;
            }
            let mapped_name = decl.name.as_ref().map(|n| {
                let base = format!("{}_{}", callee.signature.name, n);
                if callee_called_names.contains(n.as_str())
                    && (existing_caller_names.contains(&base) || allocated_names.contains(&base))
                {
                    format!("{}_{}_l{}", callee.signature.name, n, caller.locals.len())
                } else {
                    base
                }
            });
            if let Some(ref mn) = mapped_name {
                allocated_names.insert(mn.clone());
                if let Some(orig_name) = &decl.name
                    && callee_called_names.contains(orig_name.as_str())
                {
                    callee_local_callee_remap.insert(orig_name.clone(), mn.clone());
                }
            }
            let local_ty = concrete_param_types
                .get(&decl.id)
                .cloned()
                .unwrap_or_else(|| decl.ty.clone());
            let new_local_id =
                caller.add_local(mapped_name, local_ty, decl.is_mut, decl.span);
            local_map.insert(decl.id, new_local_id);
        }

        // 2. Prepend parameter assignments for non-bound params to caller's call block before the jump
        for (arg_i, p_id) in callee.params.iter().skip(n_bound).enumerate() {
            let mapped_param = local_map[p_id];
            let arg_op = call_args[arg_i].clone();

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
                // Skip StorageLive/StorageDead on bound capture cells so their value persists.
                if matches!(stmt, Statement::StorageLive(id, _) | Statement::StorageDead(id, _) if bound_param_ids.contains(id))
                {
                    continue;
                }
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
                }) => {
                    let remapped_callee = callee_local_callee_remap
                        .get(sub_c)
                        .cloned()
                        .unwrap_or_else(|| sub_c.clone());
                    Some(Terminator::Call {
                        callee: remapped_callee,
                        args: sub_a
                            .iter()
                            .map(|a| remap_operand(a, &local_map))
                            .collect(),
                        destination: remap_place(sub_d, &local_map),
                        success: block_map[sub_s],
                        error: sub_e.map(|e| block_map[&e]),
                        span: *s,
                    })
                }
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

        // Remap callee proven_in_bounds if present
        for (c_bb, pairs) in &callee.proven_in_bounds {
            if let Some(mapped_bb) = block_map.get(&BlockId(*c_bb)) {
                let mapped_pairs: Vec<(LocalId, LocalId)> = pairs
                    .iter()
                    .map(|(a, b)| (*local_map.get(a).unwrap_or(a), *local_map.get(b).unwrap_or(b)))
                    .collect();
                caller
                    .proven_in_bounds
                    .entry(mapped_bb.0)
                    .or_default()
                    .extend(mapped_pairs);
            }
        }

        callee.blocks.len()
    }

    /// Post-inlining specialization sweep (`NG-W16-5`):
    /// 1. Propagates concrete types through single-def copies and integer binary ops.
    /// 2. Devirtualizes `__mir_unknown_get_field` on concrete `MirType::Struct(sname)` receivers
    ///    into direct `Place::Field(recv, field_idx)` loads with refined field types.
    /// 3. Devirtualizes `__mir_unknown_call_method` on concrete `MirType::Struct(sname)` receivers
    ///    into direct `Terminator::Call { callee: "{sname}:{method}", args: [recv, ..] }`.
    /// 4. Specializes direct/inlined calls to locals bound to `Rvalue::Closure { func_name, captures }`:
    ///    - Pure/non-mutating closures: rewrites `call c(args)` -> `call func_name(..captures, ..args)`.
    ///    - Non-escaping closures with mutated scalar captures (e.g. `step_fn` in `BENCH-NG-04`):
    ///      materializes persistent closure-owned capture cells at the `Rvalue::Closure` site and
    ///      inlines every call site directly against those cells.
    #[allow(clippy::too_many_arguments)]
    fn specialize_caller_calls(
        &self,
        caller: &mut MirFunction,
        callees: &HashMap<String, MirFunction>,
        struct_layouts: &HashMap<String, MirStructLayout>,
        block_stacks: &mut Vec<Vec<String>>,
        jit_config: Option<&JitInlineConfig>,
        recursive_fns: Option<&HashSet<String>>,
        transitive_effect_fns: Option<&HashSet<String>>,
    ) -> bool {
        let mut any_changed = false;

        for _pass in 0..6 {
            let mut pass_changed = false;

            // Step A: Single-def concrete type propagation (copies + concrete int arithmetic)
            if propagate_concrete_local_types(caller) {
                pass_changed = true;
            }

            // Step B & C: Devirtualize `__mir_unknown_get_field` and `__mir_unknown_call_method`
            for b_idx in 0..caller.blocks.len() {
                let term = caller.blocks[b_idx].terminator.clone();
                match term {
                    Some(Terminator::Call {
                        ref callee,
                        ref args,
                        ref destination,
                        success,
                        error: None,
                        span,
                    }) if callee == "__mir_unknown_get_field" && args.len() == 2 => {
                        let recv_loc = match &args[0] {
                            Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)) => {
                                Some(*l)
                            }
                            _ => None,
                        };
                        let field_name = match &args[1] {
                            Operand::Constant(MirConstant::String(s)) => Some(s.as_str()),
                            _ => None,
                        };
                        if let (Some(r_loc), Some(fname)) = (recv_loc, field_name)
                            && let MirType::Struct(ref sname) = caller.locals[r_loc.0].ty
                            && !sname.is_empty()
                            && sname != "LuaValue"
                            && let Some(layout) = lookup_struct_layout(struct_layouts, sname)
                            && let Some(f_desc) = layout.field_by_name(fname)
                        {
                            let field_idx = f_desc.ordinal as usize;
                            let field_ty = mir_type_from_canonical_tag(f_desc.tag);
                            let dest = destination.clone();
                            caller.blocks[b_idx].statements.push(Statement::Assign(
                                dest.clone(),
                                Rvalue::Use(Operand::Copy(Place::Field(
                                    Box::new(Place::Local(r_loc)),
                                    field_idx,
                                ))),
                                span,
                            ));
                            caller.blocks[b_idx].terminator = Some(Terminator::Goto(success, span));
                            if let Place::Local(dst_loc) = dest
                                && caller.locals[dst_loc.0].ty == MirType::Unknown
                            {
                                caller.locals[dst_loc.0].ty = field_ty;
                            }
                            pass_changed = true;
                        }
                    }
                    Some(Terminator::Call {
                        ref callee,
                        ref args,
                        ref destination,
                        success,
                        error,
                        span,
                    }) if callee == "__mir_unknown_call_method" && args.len() >= 2 => {
                        let recv_sname = match operand_mir_type_in_fn(&args[0], caller) {
                            Some(MirType::Struct(sname))
                                if !sname.is_empty() && sname != "LuaValue" =>
                            {
                                Some(sname)
                            }
                            _ => None,
                        };
                        let method_name = match &args[1] {
                            Operand::Constant(MirConstant::String(s)) => Some(s.as_str()),
                            _ => None,
                        };
                        if let (Some(sname), Some(mname)) = (recv_sname, method_name)
                            && let Some((direct_callee, callee_fn)) =
                                lookup_method_callee(callees, &sname, mname)
                            && callee_fn.params.len() == 1 + (args.len() - 2)
                        {
                            let mut new_args = Vec::with_capacity(args.len() - 1);
                            new_args.push(args[0].clone());
                            new_args.extend_from_slice(&args[2..]);
                            let ret_ty = callee_fn.signature.return_type.clone();
                            let dest = destination.clone();
                            caller.blocks[b_idx].terminator = Some(Terminator::Call {
                                callee: direct_callee,
                                args: new_args,
                                destination: dest.clone(),
                                success,
                                error,
                                span,
                            });
                            if let Place::Local(dst_loc) = dest
                                && caller.locals[dst_loc.0].ty == MirType::Unknown
                                && ret_ty != MirType::Unknown
                            {
                                caller.locals[dst_loc.0].ty = ret_ty;
                            }
                            pass_changed = true;
                        }
                    }
                    _ => {}
                }
            }

            // Step D: Closure call specialization (`Rvalue::Closure` -> direct call or cell-backed inline)
            if self.specialize_closure_calls(
                caller,
                callees,
                block_stacks,
                jit_config,
                recursive_fns,
                transitive_effect_fns,
            ) {
                pass_changed = true;
            }

            if pass_changed {
                any_changed = true;
            } else {
                break;
            }
        }

        any_changed
    }

    #[allow(clippy::too_many_arguments)]
    fn specialize_closure_calls(
        &self,
        caller: &mut MirFunction,
        callees: &HashMap<String, MirFunction>,
        block_stacks: &mut Vec<Vec<String>>,
        jit_config: Option<&JitInlineConfig>,
        recursive_fns: Option<&HashSet<String>>,
        transitive_effect_fns: Option<&HashSet<String>>,
    ) -> bool {
        // Count local definitions so we only trace single-def closure roots and copy chains.
        let mut def_counts: Vec<usize> = alloc::vec![0usize; caller.locals.len()];
        for &p in &caller.params {
            if p.0 < def_counts.len() {
                def_counts[p.0] += 1;
            }
        }
        let mut closure_defs: HashMap<
            LocalId,
            (usize, usize, String, Vec<(String, Operand, MirType)>, Span),
        > = HashMap::new();
        for (b_idx, blk) in caller.blocks.iter().enumerate() {
            for (s_idx, stmt) in blk.statements.iter().enumerate() {
                if let Statement::Assign(Place::Local(dst), rval, span) = stmt {
                    if dst.0 < def_counts.len() {
                        def_counts[dst.0] += 1;
                    }
                    if let Rvalue::Closure {
                        func_name,
                        captures,
                    } = rval
                    {
                        closure_defs.insert(
                            *dst,
                            (b_idx, s_idx, func_name.clone(), captures.clone(), *span),
                        );
                    }
                }
            }
            if let Some(Terminator::Call {
                destination: Place::Local(dst),
                ..
            }) = &blk.terminator
                && dst.0 < def_counts.len()
            {
                def_counts[dst.0] += 1;
            }
        }
        if closure_defs.is_empty() {
            return false;
        }
        // Keep only single-def closure roots.
        closure_defs.retain(|loc, _| def_counts.get(loc.0).copied().unwrap_or(0) == 1);
        if closure_defs.is_empty() {
            return false;
        }

        // Propagate closure root through single-def copy chains.
        let mut local_to_root: HashMap<LocalId, LocalId> = HashMap::new();
        for &root in closure_defs.keys() {
            local_to_root.insert(root, root);
        }
        loop {
            let mut grew = false;
            for blk in &caller.blocks {
                for stmt in &blk.statements {
                    if let Statement::Assign(
                        Place::Local(dst),
                        Rvalue::Use(
                            Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)),
                        ),
                        _,
                    ) = stmt
                        && def_counts.get(dst.0).copied().unwrap_or(0) == 1
                        && !local_to_root.contains_key(dst)
                        && let Some(&root) = local_to_root.get(src)
                    {
                        local_to_root.insert(*dst, root);
                        grew = true;
                    }
                }
            }
            if !grew {
                break;
            }
        }

        // Map local name -> LocalId (only when unambiguous).
        let name_to_local: HashMap<String, LocalId> = {
            let mut name_counts: HashMap<&str, usize> = HashMap::new();
            let mut map: HashMap<String, LocalId> = HashMap::new();
            for decl in &caller.locals {
                if let Some(ref name) = decl.name {
                    *name_counts.entry(name.as_str()).or_insert(0) += 1;
                    map.insert(name.clone(), decl.id);
                }
            }
            map.retain(|name, _| name_counts.get(name.as_str()).copied().unwrap_or(0) == 1);
            map
        };

        let mut changed = false;

        // Subcase D1: Pure / non-mutating closures -> direct `Terminator::Call { callee: func_name, args: [..caps, ..args] }`
        for b_idx in 0..caller.blocks.len() {
            let Some(Terminator::Call {
                callee,
                args,
                destination,
                success,
                error,
                span,
            }) = caller.blocks[b_idx].terminator.clone()
            else {
                continue;
            };
            if callee.contains('.') || callee.contains(':') || callees.contains_key(&callee) {
                continue;
            }
            let Some(&callee_loc) = name_to_local.get(&callee) else {
                continue;
            };
            let Some(&root) = local_to_root.get(&callee_loc) else {
                continue;
            };
            let Some((_, _, func_name, captures, _)) = closure_defs.get(&root) else {
                continue;
            };
            let Some(lifted_fn) = callees.get(func_name) else {
                continue;
            };
            if lifted_fn.params.len() != captures.len() + args.len() {
                continue;
            }
            let any_cap_mutated = (0..captures.len()).any(|ci| {
                lifted_fn
                    .params
                    .get(ci)
                    .is_some_and(|&pid| local_is_mutated_in_fn(lifted_fn, pid))
            });
            if any_cap_mutated {
                continue;
            }
            // Verify every capture operand is immutable in `caller` (or constant) so passing it at
            // the call site observes the exact value at closure creation.
            let caps_stable = captures.iter().all(|(_, cap_op, _)| match cap_op {
                Operand::Constant(_) => true,
                Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => {
                    def_counts.get(loc.0).copied().unwrap_or(0) <= 1
                }
                _ => false,
            });
            if !caps_stable {
                continue;
            }
            let mut new_args = Vec::with_capacity(captures.len() + args.len());
            for (_, cap_op, _) in captures {
                new_args.push(cap_op.clone());
            }
            new_args.extend(args);
            caller.blocks[b_idx].terminator = Some(Terminator::Call {
                callee: func_name.clone(),
                args: new_args,
                destination,
                success,
                error,
                span,
            });
            changed = true;
        }
        if changed {
            return true;
        }

        // Subcase D2: Non-escaping closures with mutated scalar captures (e.g. `step_fn` in BENCH-NG-04).
        // Materialize closure-owned capture cells at the `Rvalue::Closure` site and inline every call
        // site directly against those cells so mutable upvalue state persists bit-identically.
        let mut roots_sorted: Vec<LocalId> = closure_defs.keys().copied().collect();
        roots_sorted.sort();
        for root in roots_sorted {
            let Some((def_bb, def_stmt_idx, func_name, captures, def_span)) =
                closure_defs.get(&root).cloned()
            else {
                continue;
            };
            if captures.is_empty() {
                continue;
            }
            let Some(lifted_fn) = callees.get(&func_name) else {
                continue;
            };
            if recursive_fns.is_some_and(|r| r.contains(&func_name))
                || transitive_effect_fns.is_some_and(|e| e.contains(&func_name))
            {
                continue;
            }
            if !self.is_eligible_callee(lifted_fn, &caller.signature.name, &[caller.signature.name.clone()])
                || callee_has_effect_or_handler_call(lifted_fn)
                || callee_has_loops(lifted_fn)
                || !lifted_fn.cleanup_blocks.is_empty()
                || lifted_fn
                    .blocks
                    .iter()
                    .any(|b| b.is_cleanup || matches!(b.terminator, Some(Terminator::Raise(..))))
            {
                continue;
            }
            // All captures must be scalar (or immutable) so closure-owned cells match VM semantics.
            let caps_ok = captures.iter().enumerate().all(|(ci, (_, _, cap_ty))| {
                let mutated = lifted_fn
                    .params
                    .get(ci)
                    .is_some_and(|&pid| local_is_mutated_in_fn(lifted_fn, pid));
                !mutated || matches!(cap_ty, MirType::Int | MirType::Float | MirType::Bool)
            });
            if !caps_ok {
                continue;
            }

            // Build alias set for this closure root and verify it NEVER escapes `caller`.
            let alias_set: HashSet<LocalId> = local_to_root
                .iter()
                .filter_map(|(&loc, &r)| (r == root).then_some(loc))
                .collect();
            let mut call_sites: Vec<usize> = Vec::new();
            let mut escapes = false;
            let expected_call_arity = lifted_fn.params.len().saturating_sub(captures.len());

            for (b_idx, blk) in caller.blocks.iter().enumerate() {
                for (s_idx, stmt) in blk.statements.iter().enumerate() {
                    if b_idx == def_bb && s_idx == def_stmt_idx {
                        continue;
                    }
                    match stmt {
                        Statement::StorageLive(_, _)
                        | Statement::StorageDead(_, _)
                        | Statement::ResourceCheck(_, _, _) => {}
                        Statement::Assign(lhs, rval, _) => {
                            let is_alias_copy = matches!(
                                (lhs, rval),
                                (
                                    Place::Local(dst),
                                    Rvalue::Use(
                                        Operand::Copy(Place::Local(src))
                                            | Operand::Move(Place::Local(src))
                                    )
                                ) if alias_set.contains(dst) && alias_set.contains(src)
                            );
                            if !is_alias_copy {
                                let mut used = HashSet::new();
                                stmt.collect_uses(&mut used);
                                if let Place::Local(d) = lhs
                                    && alias_set.contains(d)
                                {
                                    escapes = true;
                                }
                                if used.iter().any(|&u| alias_set.contains(&LocalId(u))) {
                                    escapes = true;
                                }
                            }
                        }
                    }
                    if escapes {
                        break;
                    }
                }
                if escapes {
                    break;
                }
                if let Some(term) = &blk.terminator {
                    let mut used = HashSet::new();
                    term.collect_uses(&mut used);
                    if used.iter().any(|&u| alias_set.contains(&LocalId(u))) {
                        escapes = true;
                        break;
                    }
                    if let Terminator::Call { callee, args, .. } = term
                        && !callee.contains('.')
                        && !callee.contains(':')
                        && !callees.contains_key(callee)
                        && let Some(&callee_loc) = name_to_local.get(callee)
                        && alias_set.contains(&callee_loc)
                    {
                        if args.len() != expected_call_arity {
                            escapes = true;
                            break;
                        }
                        call_sites.push(b_idx);
                    }
                }
            }

            if escapes || call_sites.is_empty() {
                continue;
            }
            // Enforce JIT local budget if configured.
            if let Some(jcfg) = jit_config {
                let extra_locals = captures.len() + call_sites.len() * lifted_fn.locals.len();
                if caller.locals.len().saturating_add(extra_locals) > jcfg.max_caller_locals {
                    continue;
                }
            }

            // Allocate closure-owned capture cells and initialize them right before `Rvalue::Closure`.
            let mut cap_cell_locals: Vec<LocalId> = Vec::with_capacity(captures.len());
            let mut init_stmts: Vec<Statement> = Vec::with_capacity(captures.len() * 2);
            for (ci, (cap_name, cap_op, cap_ty)) in captures.iter().enumerate() {
                let cell_name = format!("__cap_{}_{}_{}", root.0, cap_name, ci);
                let cell_id = caller.add_local(Some(cell_name), cap_ty.clone(), true, def_span);
                cap_cell_locals.push(cell_id);
                init_stmts.push(Statement::StorageLive(cell_id, def_span));
                init_stmts.push(Statement::Assign(
                    Place::Local(cell_id),
                    Rvalue::Use(cap_op.clone()),
                    def_span,
                ));
            }
            // Splice init_stmts right before `def_stmt_idx` in `def_bb`.
            caller.blocks[def_bb]
                .statements
                .splice(def_stmt_idx..def_stmt_idx, init_stmts);

            // Inline every call site against `cap_cell_locals` (without merging blocks mid-loop so
            // existing `call_sites` block indices remain valid).
            for &site_bb in &call_sites {
                let Some(Terminator::Call {
                    args,
                    destination,
                    success,
                    error,
                    span,
                    ..
                }) = caller.blocks[site_bb].terminator.clone()
                else {
                    continue;
                };
                let cur_stack = if site_bb < block_stacks.len() {
                    block_stacks[site_bb].clone()
                } else {
                    vec![caller.signature.name.clone()]
                };
                let mut next_stack = cur_stack;
                next_stack.push(func_name.clone());
                let added = self.inline_call_site_unmerged_with_bound_params(
                    caller,
                    site_bb,
                    lifted_fn,
                    &cap_cell_locals,
                    &args,
                    destination,
                    success,
                    error,
                    span,
                );
                for _ in 0..added {
                    block_stacks.push(next_stack.clone());
                }
            }
            remove_dead_closure_stmts(caller);
            return true;
        }

        false
    }

    /// Performs inlining of `callee` into `caller` at `call_block_idx`.
    /// When `callee` is a single-block or straight-line callee, merges trivial unconditional
    /// `Goto` chains (`pre_call_bb -> cloned_entry_bb -> continuation_bb`) when predecessor
    /// and successor counts are 1, strictly preserving `proven_in_bounds` and deopt/cleanup metadata.
    pub fn inline_call_site(
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
        let is_straight = Self::is_straight_line(callee);
        let added = self.inline_call_site_unmerged(
            caller,
            call_block_idx,
            callee,
            call_args,
            destination,
            succ_bb,
            err_bb,
            call_span,
        );

        if is_straight {
            Self::merge_linear_blocks(caller);
        }

        added
    }

    /// Backwards-compatible alias for `inline_call_site`.
    pub fn perform_inline(
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
        self.inline_call_site(
            caller,
            call_block_idx,
            callee,
            call_args,
            destination,
            succ_bb,
            err_bb,
            call_span,
        )
    }

    /// Simplifies the CFG by merging trivial unconditional `Goto` chains (`A -> B`)
    /// when `A` unconditionally jumps to `B` and `B` has in-degree 1.
    ///
    /// Guardrails:
    /// - Never merges across exception / error handler boundaries.
    /// - Never merges across cleanup boundaries (`from.is_cleanup != to.is_cleanup` or `to` in `cleanup_blocks`).
    /// - Never merges across deopt-point headers or loop headers (`find_loop_headers` or `to == entry`).
    /// - Preserves and remaps `func.proven_in_bounds` and `func.cleanup_blocks`.
    pub fn merge_linear_blocks(func: &mut MirFunction) -> bool {
        let mut overall_changed = false;
        loop {
            let mut in_degrees = alloc::vec![0usize; func.blocks.len()];
            if func.entry.0 < in_degrees.len() {
                in_degrees[func.entry.0] = 1;
            }
            for block in &func.blocks {
                if let Some(term) = &block.terminator {
                    for succ in terminator_successors(term) {
                        if succ.0 < in_degrees.len() {
                            in_degrees[succ.0] += 1;
                        }
                    }
                }
            }

            let loop_headers: hashbrown::HashSet<BlockId> =
                crate::mir::opt::Safepoints::find_loop_headers_in_fn(func);

            let mut error_targets: hashbrown::HashSet<BlockId> = hashbrown::HashSet::new();
            for block in &func.blocks {
                if let Some(Terminator::Call { error: Some(err_bb), .. }) = &block.terminator {
                    error_targets.insert(*err_bb);
                }
            }

            let mut cleanup_entry_blocks: hashbrown::HashSet<BlockId> = hashbrown::HashSet::new();
            for entry in &func.cleanup_blocks {
                cleanup_entry_blocks.insert(entry.block);
            }

            let mut changed = false;
            for i in 0..func.blocks.len() {
                if let Some(Terminator::Goto(target_id, _)) = func.blocks[i].terminator {
                    if target_id.0 != i
                        && target_id.0 < func.blocks.len()
                        && in_degrees[target_id.0] == 1
                        && target_id != func.entry
                        && !loop_headers.contains(&target_id)
                        && !error_targets.contains(&target_id)
                        && !cleanup_entry_blocks.contains(&target_id)
                        && func.blocks[i].is_cleanup == func.blocks[target_id.0].is_cleanup
                        && func.blocks[target_id.0].terminator.is_some()
                    {
                        let target_stmts = core::mem::take(&mut func.blocks[target_id.0].statements);
                        let target_term = func.blocks[target_id.0].terminator.take();

                        func.blocks[i].statements.extend(target_stmts);
                        func.blocks[i].terminator = target_term;

                        func.blocks[target_id.0].terminator =
                            Some(Terminator::Unreachable(Span::dummy()));

                        // Preserve proven_in_bounds:
                        if let Some(target_pairs) = func.proven_in_bounds.remove(&target_id.0) {
                            let i_pairs = func.proven_in_bounds.entry(i).or_default();
                            for pair in target_pairs {
                                if !i_pairs.contains(&pair) {
                                    i_pairs.push(pair);
                                }
                            }
                        }

                        changed = true;
                    }
                }
            }

            if changed {
                overall_changed = true;
                compact_and_renumber_blocks(func);
            } else {
                break;
            }
        }

        overall_changed
    }

    /// Simplifies the CFG by fusing linear Goto chains where the target has in-degree 1.
    pub fn simplify_cfg(func: &mut MirFunction) -> bool {
        Self::merge_linear_blocks(func)
    }
}

fn compact_and_renumber_blocks(func: &mut MirFunction) {
    let mut reachable = hashbrown::HashSet::new();
    let mut worklist = alloc::vec::Vec::new();

    reachable.insert(func.entry);
    worklist.push(func.entry);

    for entry in &func.cleanup_blocks {
        if entry.block.0 < func.blocks.len() && reachable.insert(entry.block) {
            worklist.push(entry.block);
        }
    }
    for (idx, b) in func.blocks.iter().enumerate() {
        if b.is_cleanup && reachable.insert(BlockId(idx)) {
            worklist.push(BlockId(idx));
        }
    }

    while let Some(bb) = worklist.pop() {
        if let Some(block) = func.blocks.get(bb.0) {
            if let Some(term) = &block.terminator {
                for succ in terminator_successors(term) {
                    if succ.0 < func.blocks.len() && reachable.insert(succ) {
                        worklist.push(succ);
                    }
                }
            }
        }
    }

    if reachable.len() == func.blocks.len() {
        for (i, b) in func.blocks.iter_mut().enumerate() {
            b.id = BlockId(i);
        }
        return;
    }

    let mut remap = hashbrown::HashMap::new();
    let mut new_blocks = alloc::vec::Vec::with_capacity(reachable.len());

    for (old_idx, block) in func.blocks.iter().enumerate() {
        let old_id = BlockId(old_idx);
        if reachable.contains(&old_id) {
            let new_id = BlockId(new_blocks.len());
            remap.insert(old_id, new_id);
            let mut new_block = block.clone();
            new_block.id = new_id;
            new_blocks.push(new_block);
        }
    }

    for block in &mut new_blocks {
        if let Some(term) = &mut block.terminator {
            patch_terminator_targets(term, &remap);
        }
    }

    if let Some(&new_entry) = remap.get(&func.entry) {
        func.entry = new_entry;
    }

    for entry in &mut func.cleanup_blocks {
        if let Some(&new_id) = remap.get(&entry.block) {
            entry.block = new_id;
        }
    }

    let mut new_proven = hashbrown::HashMap::new();
    for (old_b_idx, pairs) in &func.proven_in_bounds {
        if let Some(&new_id) = remap.get(&BlockId(*old_b_idx)) {
            new_proven.insert(new_id.0, pairs.clone());
        }
    }
    func.proven_in_bounds = new_proven;

    func.blocks = new_blocks;
}

fn patch_terminator_targets(term: &mut Terminator, remap: &hashbrown::HashMap<BlockId, BlockId>) {
    match term {
        Terminator::Goto(target, _) => {
            if let Some(&new_id) = remap.get(target) {
                *target = new_id;
            }
        }
        Terminator::Cond {
            then_block,
            else_block,
            ..
        } => {
            if let Some(&new_id) = remap.get(then_block) {
                *then_block = new_id;
            }
            if let Some(&new_id) = remap.get(else_block) {
                *else_block = new_id;
            }
        }
        Terminator::Switch { cases, default, .. } => {
            for (_, target) in cases {
                if let Some(&new_id) = remap.get(target) {
                    *target = new_id;
                }
            }
            if let Some(&new_id) = remap.get(default) {
                *default = new_id;
            }
        }
        Terminator::Call { success, error, .. } => {
            if let Some(&new_id) = remap.get(success) {
                *success = new_id;
            }
            if let Some(err_bb) = error {
                if let Some(&new_id) = remap.get(err_bb) {
                    *err_bb = new_id;
                }
            }
        }
        Terminator::Return(_, _) | Terminator::Raise(_, _) | Terminator::Unreachable(_) => {}
    }
}

fn is_jit_effect_or_handler_callee(callee: &str) -> bool {
    #[cfg(feature = "std")]
    {
        crate::mir::cranelift_backend::is_effect_or_handler_callee(callee)
    }
    #[cfg(not(feature = "std"))]
    {
        matches!(
            callee,
            "__lust_handle"
                | "__lust_perform"
                | "perform"
                | "__lust_resume"
                | "resume"
                | "__lust_register_finalizer"
                | "register_finalizer"
                | "effect.perform"
        )
    }
}

fn callee_has_effect_or_handler_call(callee: &MirFunction) -> bool {
    callee.blocks.iter().any(|block| {
        matches!(
            &block.terminator,
            Some(Terminator::Call { callee: inner, .. }) if is_jit_effect_or_handler_callee(inner)
        )
    })
}

fn callee_has_loops(callee: &MirFunction) -> bool {
    !crate::mir::opt::Safepoints::find_loop_headers_in_fn(callee).is_empty()
}

fn is_primitive_leaf_builtin_callee(callee: &str) -> bool {
    matches!(
        callee,
        "array.len"
            | "table.len"
            | "string.len"
            | "string.sub"
            | "string.split"
            | "string.eq"
            | "math.abs"
            | "abs"
            | "math.min"
            | "min"
            | "math.max"
            | "max"
            | "math.clamp"
            | "clamp"
            | "math.floor"
            | "floor"
            | "math.ceil"
            | "ceil"
            | "math.sqrt"
            | "sqrt"
    )
}

fn register_fn_struct_layouts(
    out: &mut HashMap<String, MirStructLayout>,
    func: &MirFunction,
) {
    for layout in &func.struct_layouts {
        out.entry(layout.type_name.clone())
            .or_insert_with(|| layout.clone());
    }
}

fn collect_module_struct_layouts(module: &MirModule) -> HashMap<String, MirStructLayout> {
    let mut out: HashMap<String, MirStructLayout> = HashMap::new();
    for layout in &module.struct_layouts {
        out.entry(layout.type_name.clone())
            .or_insert_with(|| layout.clone());
    }
    for func in &module.functions {
        register_fn_struct_layouts(&mut out, func);
    }
    out
}

fn lookup_struct_layout<'a>(
    layouts: &'a HashMap<String, MirStructLayout>,
    struct_name: &str,
) -> Option<&'a MirStructLayout> {
    if let Some(l) = layouts.get(struct_name) {
        return Some(l);
    }
    let simple = struct_name.rsplit('.').next().unwrap_or(struct_name);
    if let Some(l) = layouts.get(simple) {
        return Some(l);
    }
    let suffix = format!(".{simple}");
    let mut matched: Option<&MirStructLayout> = None;
    for (k, l) in layouts {
        if k.ends_with(&suffix) {
            if matched.is_some() {
                return None;
            }
            matched = Some(l);
        }
    }
    matched
}

fn lookup_method_callee<'a>(
    callees: &'a HashMap<String, MirFunction>,
    struct_name: &str,
    method_name: &str,
) -> Option<(String, &'a MirFunction)> {
    let exact = format!("{struct_name}:{method_name}");
    if let Some(f) = callees.get(&exact) {
        return Some((exact, f));
    }
    let simple = struct_name.rsplit('.').next().unwrap_or(struct_name);
    let simple_key = format!("{simple}:{method_name}");
    if let Some(f) = callees.get(&simple_key) {
        return Some((simple_key, f));
    }
    let suffix = format!(".{simple}:{method_name}");
    let mut found: Option<(String, &MirFunction)> = None;
    for (k, f) in callees {
        if k.ends_with(&suffix) {
            if found.is_some() {
                return None;
            }
            found = Some((k.clone(), f));
        }
    }
    found
}

fn mir_type_from_canonical_tag(tag: CanonicalTypeTag) -> MirType {
    match tag {
        CanonicalTypeTag::Bool => MirType::Bool,
        CanonicalTypeTag::Int64 | CanonicalTypeTag::UInt64 | CanonicalTypeTag::DateTimeUtc => {
            MirType::Int
        }
        CanonicalTypeTag::Float64 => MirType::Float,
        CanonicalTypeTag::StringSlice => MirType::String,
        _ => MirType::Int,
    }
}

fn is_concrete_specializable_type(ty: &MirType) -> bool {
    match ty {
        MirType::Struct(s) => !s.is_empty() && s != "LuaValue",
        MirType::Function { .. } => true,
        _ => false,
    }
}

fn operand_mir_type_in_fn(op: &Operand, func: &MirFunction) -> Option<MirType> {
    match op {
        Operand::Constant(c) => Some(match c {
            MirConstant::Unit => MirType::Unit,
            MirConstant::Bool(_) => MirType::Bool,
            MirConstant::Int(_) => MirType::Int,
            MirConstant::Float(_) => MirType::Float,
            MirConstant::String(_) => MirType::String,
        }),
        Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => {
            func.locals.get(loc.0).map(|d| d.ty.clone())
        }
        _ => None,
    }
}

fn local_is_mutated_in_fn(func: &MirFunction, loc: LocalId) -> bool {
    for blk in &func.blocks {
        for stmt in &blk.statements {
            if matches!(stmt, Statement::Assign(Place::Local(d), _, _) if *d == loc) {
                return true;
            }
        }
        if matches!(
            &blk.terminator,
            Some(Terminator::Call {
                destination: Place::Local(d),
                ..
            }) if *d == loc
        ) {
            return true;
        }
    }
    false
}

fn propagate_concrete_local_types(func: &mut MirFunction) -> bool {
    let mut def_counts = alloc::vec![0usize; func.locals.len()];
    for &p in &func.params {
        if p.0 < def_counts.len() {
            def_counts[p.0] += 1;
        }
    }
    for blk in &func.blocks {
        for stmt in &blk.statements {
            if let Statement::Assign(Place::Local(d), _, _) = stmt
                && d.0 < def_counts.len()
            {
                def_counts[d.0] += 1;
            }
        }
        if let Some(Terminator::Call {
            destination: Place::Local(d),
            ..
        }) = &blk.terminator
            && d.0 < def_counts.len()
        {
            def_counts[d.0] += 1;
        }
    }

    let mut any_changed = false;
    loop {
        let mut changed = false;
        for b_idx in 0..func.blocks.len() {
            let n_stmts = func.blocks[b_idx].statements.len();
            for s_idx in 0..n_stmts {
                let (dst, inferred) = {
                    let stmt = &func.blocks[b_idx].statements[s_idx];
                    let Statement::Assign(Place::Local(d), rval, _) = stmt else {
                        continue;
                    };
                    if func.locals[d.0].ty != MirType::Unknown || def_counts[d.0] != 1 {
                        continue;
                    }
                    let ty = match rval {
                        Rvalue::Use(op) => operand_mir_type_in_fn(op, func).filter(|t| {
                            *t != MirType::Unknown
                                && !matches!(t, MirType::Struct(s) if s == "LuaValue")
                        }),
                        Rvalue::BinaryOp(op, l, r) => {
                            let lt = operand_mir_type_in_fn(l, func);
                            let rt = operand_mir_type_in_fn(r, func);
                            if lt == Some(MirType::Int) && rt == Some(MirType::Int) {
                                Some(match op {
                                    MirBinaryOp::Eq
                                    | MirBinaryOp::NotEq
                                    | MirBinaryOp::Lt
                                    | MirBinaryOp::LtEq
                                    | MirBinaryOp::Gt
                                    | MirBinaryOp::GtEq => MirType::Bool,
                                    _ => MirType::Int,
                                })
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };
                    (*d, ty)
                };
                if let Some(new_ty) = inferred {
                    func.locals[dst.0].ty = new_ty;
                    changed = true;
                    any_changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    any_changed
}

/// Removes dead `Rvalue::Closure` and dead closure-alias copy assignments whose destination
/// local has zero remaining reads (and is not called by local name in any `Terminator::Call`).
fn remove_dead_closure_stmts(func: &mut MirFunction) {
    loop {
        let mut read_locals: HashSet<usize> = HashSet::new();
        let mut called_names: HashSet<String> = HashSet::new();
        for blk in &func.blocks {
            for stmt in &blk.statements {
                stmt.collect_uses(&mut read_locals);
            }
            if let Some(term) = &blk.terminator {
                term.collect_uses(&mut read_locals);
                if let Terminator::Call { callee, .. } = term
                    && !callee.contains('.')
                    && !callee.contains(':')
                {
                    called_names.insert(callee.clone());
                }
            }
        }
        let mut live_by_name: HashSet<usize> = HashSet::new();
        for decl in &func.locals {
            if let Some(ref name) = decl.name
                && called_names.contains(name)
            {
                live_by_name.insert(decl.id.0);
            }
        }

        let mut removed = false;
        for blk in &mut func.blocks {
            let before = blk.statements.len();
            blk.statements.retain(|stmt| {
                let Statement::Assign(Place::Local(dst), rval, _) = stmt else {
                    return true;
                };
                let is_dead_closure_or_fn_copy = matches!(rval, Rvalue::Closure { .. })
                    || (matches!(
                        func.locals.get(dst.0).map(|d| &d.ty),
                        Some(MirType::Function { .. })
                    ) && matches!(rval, Rvalue::Use(_)));
                if is_dead_closure_or_fn_copy
                    && !read_locals.contains(&dst.0)
                    && !live_by_name.contains(&dst.0)
                {
                    return false;
                }
                true
            });
            if blk.statements.len() != before {
                removed = true;
            }
        }
        if !removed {
            // Also drop StorageLive/StorageDead and demote unused dead `MirType::Function` locals
            // to `MirType::Int` so downstream JIT/VM type checks never reject dead closure slots.
            let mut referenced: HashSet<usize> = HashSet::new();
            for &p in &func.params {
                referenced.insert(p.0);
            }
            for blk in &func.blocks {
                for stmt in &blk.statements {
                    if let Statement::Assign(lhs, _, _) = stmt {
                        referenced.insert(lhs.root_local().0);
                    }
                    stmt.collect_uses(&mut referenced);
                }
                if let Some(term) = &blk.terminator {
                    term.collect_uses(&mut referenced);
                    if let Terminator::Call { destination, .. } = term {
                        referenced.insert(destination.root_local().0);
                    }
                }
            }
            let dead_fn_locals: HashSet<usize> = func
                .locals
                .iter()
                .filter(|d| {
                    matches!(d.ty, MirType::Function { .. })
                        && !referenced.contains(&d.id.0)
                        && !live_by_name.contains(&d.id.0)
                })
                .map(|d| d.id.0)
                .collect();
            if !dead_fn_locals.is_empty() {
                for blk in &mut func.blocks {
                    blk.statements.retain(|stmt| {
                        !matches!(
                            stmt,
                            Statement::StorageLive(id, _) | Statement::StorageDead(id, _)
                                if dead_fn_locals.contains(&id.0)
                        )
                    });
                }
                for loc_idx in dead_fn_locals {
                    func.locals[loc_idx].ty = MirType::Int;
                }
            }
            break;
        }
    }
}

/// Computes all blocks belonging to any natural loop in `func` using dominance-verified
/// backedges from `Safepoints::find_loop_backedges_in_fn`.
fn compute_natural_loop_blocks(func: &MirFunction) -> HashSet<BlockId> {
    let n = func.blocks.len();
    if n == 0 || func.entry.0 >= n {
        return HashSet::new();
    }
    let backedges = crate::mir::opt::Safepoints::find_loop_backedges_in_fn(func);
    if backedges.is_empty() {
        return HashSet::new();
    }
    let mut preds: Vec<Vec<BlockId>> = alloc::vec![Vec::new(); n];
    for (idx, block) in func.blocks.iter().enumerate() {
        if let Some(term) = &block.terminator {
            for succ in terminator_successors(term) {
                if succ.0 < n {
                    preds[succ.0].push(BlockId(idx));
                }
            }
        }
    }
    let mut in_loop = HashSet::new();
    for (latch, header) in backedges {
        in_loop.insert(header);
        if latch != header && in_loop.insert(latch) {
            let mut worklist = alloc::vec![latch];
            while let Some(b) = worklist.pop() {
                for &p in &preds[b.0] {
                    if p != header && in_loop.insert(p) {
                        worklist.push(p);
                    }
                }
            }
        }
    }
    in_loop
}

/// Reorders reachable blocks in Reverse Post-Order (RPO) from `func.entry` (with any
/// entry-unreachable cleanup blocks appended in original order) and remaps all block IDs,
/// `func.entry`, `func.cleanup_blocks`, and `func.proven_in_bounds`.
///
/// In a reducible CFG, RPO numbering guarantees every forward edge satisfies `src < dst`
/// and every backedge satisfies `dst <= src`, preventing post-inline spliced return edges
/// (`cloned_exit -> succ_bb`) from appearing as false backedges to downstream index-order passes.
fn renumber_blocks_rpo(func: &mut MirFunction) {
    let n = func.blocks.len();
    if n == 0 || func.entry.0 >= n {
        return;
    }

    let mut visited = alloc::vec![false; n];
    let mut postorder = Vec::with_capacity(n);
    let mut stack: Vec<(usize, Vec<usize>, usize)> = Vec::with_capacity(n);

    let succs_of = |blocks: &[BasicBlock], idx: usize| -> Vec<usize> {
        blocks
            .get(idx)
            .and_then(|b| b.terminator.as_ref())
            .map(|t| {
                terminator_successors(t)
                    .into_iter()
                    .map(|s| s.0)
                    .filter(|&s| s < blocks.len())
                    .collect()
            })
            .unwrap_or_default()
    };

    visited[func.entry.0] = true;
    stack.push((func.entry.0, succs_of(&func.blocks, func.entry.0), 0));

    while let Some((u, succs, next_idx)) = stack.last_mut() {
        if *next_idx < succs.len() {
            let v = succs[*next_idx];
            *next_idx += 1;
            if !visited[v] {
                visited[v] = true;
                let v_succs = succs_of(&func.blocks, v);
                stack.push((v, v_succs, 0));
            }
        } else {
            let u_val = *u;
            stack.pop();
            postorder.push(u_val);
        }
    }

    let mut order = postorder;
    order.reverse();

    // Preserve any cleanup blocks (and blocks reachable from them) not reached from `func.entry`.
    let mut cleanup_worklist: Vec<usize> = Vec::new();
    for entry in &func.cleanup_blocks {
        if entry.block.0 < n && !visited[entry.block.0] {
            visited[entry.block.0] = true;
            cleanup_worklist.push(entry.block.0);
        }
    }
    for (idx, b) in func.blocks.iter().enumerate() {
        if b.is_cleanup && !visited[idx] {
            visited[idx] = true;
            cleanup_worklist.push(idx);
        }
    }
    let mut wi = 0;
    while wi < cleanup_worklist.len() {
        let u = cleanup_worklist[wi];
        wi += 1;
        order.push(u);
        for v in succs_of(&func.blocks, u) {
            if !visited[v] {
                visited[v] = true;
                cleanup_worklist.push(v);
            }
        }
    }

    let is_identity = order.len() == n && order.iter().enumerate().all(|(i, &b)| i == b);
    if is_identity {
        for (i, b) in func.blocks.iter_mut().enumerate() {
            b.id = BlockId(i);
        }
        return;
    }

    let mut remap: HashMap<BlockId, BlockId> = HashMap::with_capacity(order.len());
    let mut new_blocks: Vec<BasicBlock> = Vec::with_capacity(order.len());
    for (new_idx, &old_idx) in order.iter().enumerate() {
        let new_id = BlockId(new_idx);
        remap.insert(BlockId(old_idx), new_id);
        let mut blk = func.blocks[old_idx].clone();
        blk.id = new_id;
        new_blocks.push(blk);
    }

    for block in &mut new_blocks {
        if let Some(term) = &mut block.terminator {
            patch_terminator_targets(term, &remap);
        }
    }

    if let Some(&new_entry) = remap.get(&func.entry) {
        func.entry = new_entry;
    }
    for entry in &mut func.cleanup_blocks {
        if let Some(&new_id) = remap.get(&entry.block) {
            entry.block = new_id;
        }
    }
    let mut new_proven = HashMap::new();
    for (old_b_idx, pairs) in &func.proven_in_bounds {
        if let Some(&new_id) = remap.get(&BlockId(*old_b_idx)) {
            new_proven.insert(new_id.0, pairs.clone());
        }
    }
    func.proven_in_bounds = new_proven;
    func.blocks = new_blocks;
}

/// Computes all module functions that participate in a direct or mutual recursion cycle
/// (i.e., `f` can reach `f` in 1 or more module-local calls).
fn compute_recursive_functions(callees: &HashMap<String, MirFunction>) -> HashSet<String> {
    let mut edges: HashMap<&str, Vec<&str>> = HashMap::with_capacity(callees.len());
    for (name, func) in callees {
        let mut targets = Vec::new();
        for block in &func.blocks {
            if let Some(Terminator::Call { callee, .. }) = &block.terminator
                && callees.contains_key(callee.as_str())
            {
                targets.push(callee.as_str());
            }
        }
        edges.insert(name.as_str(), targets);
    }

    let mut recursive = HashSet::new();
    for start in callees.keys() {
        let start_str = start.as_str();
        let mut visited: HashSet<&str> = HashSet::new();
        let mut stack: Vec<&str> = edges.get(start_str).cloned().unwrap_or_default();
        while let Some(node) = stack.pop() {
            if node == start_str {
                recursive.insert(start.clone());
                break;
            }
            if visited.insert(node)
                && let Some(nexts) = edges.get(node)
            {
                stack.extend(nexts.iter().copied());
            }
        }
    }
    recursive
}

/// Computes all module functions that directly or transitively call an effect/handler intrinsic.
fn compute_transitive_effect_functions(callees: &HashMap<String, MirFunction>) -> HashSet<String> {
    let mut effect_fns: HashSet<String> = callees
        .iter()
        .filter(|(_, func)| callee_has_effect_or_handler_call(func))
        .map(|(name, _)| name.clone())
        .collect();

    loop {
        let mut added = false;
        for (name, func) in callees {
            if effect_fns.contains(name) {
                continue;
            }
            let calls_effect = func.blocks.iter().any(|block| {
                matches!(
                    &block.terminator,
                    Some(Terminator::Call { callee, .. }) if effect_fns.contains(callee)
                )
            });
            if calls_effect {
                effect_fns.insert(name.clone());
                added = true;
            }
        }
        if !added {
            break;
        }
    }
    effect_fns
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
        Rvalue::Closure { func_name, captures } => Rvalue::Closure {
            func_name: func_name.clone(),
            captures: captures
                .iter()
                .map(|(name, op, ty)| (name.clone(), remap_operand(op, map), ty.clone()))
                .collect(),
        },
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

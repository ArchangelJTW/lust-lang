use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};

use crate::ast::Span;
use crate::data::layout::CanonicalTypeTag;
use crate::mir::ir::{
    LocalDecl, LocalId, MirFunction, MirStructLayout, MirType, Operand, Place, Rvalue,
    SroaStructRecipe, Statement, Terminator,
};

/// Maximum total locals allowed in a `MirFunction` after SROA expansion.
/// Keeps MIR-VM continuation bytecode register indices safely below `u8::MAX` (255)
/// with headroom for compiler scratch registers.
pub const SROA_MAX_LOCALS_LIMIT: usize = 250;

/// MIR pass: Scalar Replacement of Aggregates (SROA) for non-escaping structs
/// whose fields are strictly scalar (`Int | Float | Bool | String`).
///
/// Replaces a non-escaping struct local initialized in `func.entry` with one
/// dedicated SSA scalar `LocalDecl` (`is_stack_promoted = false`) per field,
/// rewrites all `Place::Field(Local(orig), k)` projections (including validated
/// immutable inlined-method `self` receiver copies) to `Place::Local(field_locals[k])`,
/// eliminates the struct allocation, and records an [`SroaStructRecipe`] in
/// `func.sroa_decomposed_structs` for cold deoptimization materialization.
pub struct ScalarReplacementOfAggregates;

impl ScalarReplacementOfAggregates {
    pub fn run_on_function(
        func: &mut MirFunction,
        struct_layouts: &HashMap<String, MirStructLayout>,
    ) -> bool {
        if func.blocks.is_empty() || func.entry.0 >= func.blocks.len() || struct_layouts.is_empty()
        {
            return false;
        }

        // Entry block must have zero predecessors so its initialization executes at most once
        // and strictly dominates every reachable block in the function.
        let entry_idx = func.entry.0;
        for block in &func.blocks {
            if let Some(term) = &block.terminator {
                if term.successors().contains(&entry_idx) {
                    return false;
                }
            }
        }

        let mut changed = false;
        let orig_local_count = func.locals.len();

        for local_idx in 0..orig_local_count {
            let local_id = LocalId(local_idx);
            if Self::try_decompose_candidate(func, local_id, struct_layouts) {
                changed = true;
            }
        }

        if Self::run_multi_def_struct_sroa(func, struct_layouts) {
            changed = true;
        }

        changed
    }

    fn lookup_struct_layout<'a>(
        struct_layouts: &'a HashMap<String, MirStructLayout>,
        struct_name: &str,
    ) -> Option<&'a MirStructLayout> {
        if let Some(layout) = struct_layouts.get(struct_name) {
            return Some(layout);
        }
        let mut matched: Option<&'a MirStructLayout> = None;
        for (k, layout) in struct_layouts {
            if Self::struct_names_match(k, struct_name) {
                if matched.is_some() {
                    return None;
                }
                matched = Some(layout);
            }
        }
        matched
    }

    fn resolve_scalar_fields(
        layout: &MirStructLayout,
    ) -> Option<(Vec<MirType>, Vec<String>)> {
        if layout.fields.is_empty() {
            return None;
        }
        let mut field_types = Vec::with_capacity(layout.fields.len());
        let mut field_names = Vec::with_capacity(layout.fields.len());
        for (idx, field) in layout.fields.iter().enumerate() {
            if field.ordinal as usize != idx || field.is_nullable || field.is_optional {
                return None;
            }
            let mir_ty = match field.tag {
                CanonicalTypeTag::Int64 => MirType::Int,
                CanonicalTypeTag::Float64 => MirType::Float,
                CanonicalTypeTag::Bool => MirType::Bool,
                CanonicalTypeTag::StringSlice => MirType::String,
                _ => return None,
            };
            field_types.push(mir_ty);
            field_names.push(field.name.clone());
        }
        Some((field_types, field_names))
    }

    fn try_decompose_candidate(
        func: &mut MirFunction,
        local_id: LocalId,
        struct_layouts: &HashMap<String, MirStructLayout>,
    ) -> bool {
        let Some(decl) = func.locals.get(local_id.0) else {
            return false;
        };

        // Precondition 1: Non-param, non-return struct local with all-scalar fields and headroom <= 250.
        if decl.is_param || local_id == func.return_local || func.params.contains(&local_id) {
            return false;
        }
        if func
            .sroa_decomposed_structs
            .iter()
            .any(|r| r.orig_local == local_id)
        {
            return false;
        }
        let MirType::Struct(struct_name) = &decl.ty else {
            return false;
        };
        if struct_name.is_empty() {
            return false;
        }
        let struct_name = struct_name.clone();
        let orig_span = decl.span;

        let Some(layout) = Self::lookup_struct_layout(struct_layouts, &struct_name) else {
            return false;
        };
        let Some((field_types, field_names)) = Self::resolve_scalar_fields(layout) else {
            return false;
        };
        let n_fields = field_types.len();

        // Hard gate: DO NOT apply SROA if `func.locals.len() + layout.fields.len() > 250`.
        if func.locals.len().saturating_add(n_fields) > SROA_MAX_LOCALS_LIMIT {
            return false;
        }

        // Precondition 2: Find single dominating Aggregate initialization in `func.entry`
        // (either direct or via single-use temporary Aggregate chain in `func.entry`).
        let Some(init_plan) =
            Self::find_entry_init_plan(func, local_id, &struct_name, n_fields)
        else {
            return false;
        };

        // Ensure `local_id` and its temp chain are never referenced in `proven_in_bounds`.
        for pairs in func.proven_in_bounds.values() {
            for &(a, b) in pairs {
                if a == local_id
                    || b == local_id
                    || init_plan.tmp_chain.contains(&a)
                    || init_plan.tmp_chain.contains(&b)
                {
                    return false;
                }
            }
        }

        // Scan all blocks to validate every occurrence of `local_id` and discover
        // any strictly safe immutable inlined-receiver aliases (`alias = Use(Copy/Move(local_id))`).
        let Some(valid_aliases) = Self::validate_all_uses_and_collect_aliases(
            func,
            local_id,
            &struct_name,
            n_fields,
            &init_plan,
        ) else {
            return false;
        };

        // --- All preconditions proven! Perform MIR transformation ---

        // 1. Allocate one SSA scalar LocalDecl per field (`is_stack_promoted: false` strictly!).
        let mut field_locals = Vec::with_capacity(n_fields);
        for (k, field_ty) in field_types.iter().enumerate() {
            let field_local_id = LocalId(func.locals.len());
            func.locals.push(LocalDecl {
                id: field_local_id,
                name: Some(format!("__sroa_{}_{}", local_id.0, k)),
                ty: field_ty.clone(),
                is_mut: true,
                span: orig_span,
                is_stack_promoted: false,
                is_param: false,
            });
            field_locals.push(field_local_id);
        }

        let mut dead_locals: HashSet<LocalId> = HashSet::new();
        dead_locals.insert(local_id);
        for &tmp in &init_plan.tmp_chain {
            dead_locals.insert(tmp);
        }
        let mut rewrite_bases: HashSet<LocalId> = HashSet::new();
        rewrite_bases.insert(local_id);
        let mut remove_alias_stmts: HashSet<(usize, usize)> = HashSet::new();
        for &(alias_id, bb_idx, stmt_idx) in &valid_aliases {
            dead_locals.insert(alias_id);
            rewrite_bases.insert(alias_id);
            remove_alias_stmts.insert((bb_idx, stmt_idx));
        }

        let entry_idx = func.entry.0;
        let remove_entry_stmts: HashSet<usize> = init_plan.copy_stmt_indices.iter().copied().collect();

        // 2. Rewrite statements and terminators across all blocks.
        for (b_idx, block) in func.blocks.iter_mut().enumerate() {
            let mut new_stmts = Vec::with_capacity(block.statements.len() + n_fields * 2);
            for (s_idx, stmt) in block.statements.iter().enumerate() {
                if b_idx == entry_idx && s_idx == init_plan.agg_stmt_idx {
                    for (k, op) in init_plan.field_ops.iter().enumerate() {
                        new_stmts.push(Statement::StorageLive(field_locals[k], init_plan.agg_span));
                        let mut rewritten_op = op.clone();
                        Self::rewrite_operand(&mut rewritten_op, &rewrite_bases, &field_locals);
                        new_stmts.push(Statement::Assign(
                            Place::Local(field_locals[k]),
                            Rvalue::Use(rewritten_op),
                            init_plan.agg_span,
                        ));
                    }
                    continue;
                }
                if b_idx == entry_idx && remove_entry_stmts.contains(&s_idx) {
                    continue;
                }
                if remove_alias_stmts.contains(&(b_idx, s_idx)) {
                    continue;
                }
                match stmt {
                    Statement::StorageLive(id, _) | Statement::StorageDead(id, _)
                        if dead_locals.contains(id) =>
                    {
                        continue;
                    }
                    _ => {}
                }
                let mut cloned = stmt.clone();
                Self::rewrite_statement(&mut cloned, &rewrite_bases, &field_locals);
                new_stmts.push(cloned);
            }
            block.statements = new_stmts;

            if let Some(term) = &mut block.terminator {
                Self::rewrite_terminator(term, &rewrite_bases, &field_locals);
            }
        }

        // 3. Record cold deoptimization reconstruction recipe.
        func.sroa_decomposed_structs.push(SroaStructRecipe {
            orig_local: local_id,
            struct_name,
            field_names,
            field_locals,
        });

        true
    }

    fn find_entry_init_plan(
        func: &MirFunction,
        local_id: LocalId,
        struct_name: &str,
        n_fields: usize,
    ) -> Option<EntryInitPlan> {
        let entry_idx = func.entry.0;
        let entry_block = &func.blocks[entry_idx];

        // Find the unique assignment to `Place::Local(local_id)` in `entry_block`.
        let mut init_stmt_idx: Option<usize> = None;
        for (s_idx, stmt) in entry_block.statements.iter().enumerate() {
            if let Statement::Assign(Place::Local(dst), _, _) = stmt {
                if *dst == local_id {
                    if init_stmt_idx.is_some() {
                        // Multiple whole-struct assignments in entry -> reject.
                        return None;
                    }
                    init_stmt_idx = Some(s_idx);
                }
            }
        }
        let init_stmt_idx = init_stmt_idx?;

        // Ensure `local_id` is never referenced before `init_stmt_idx` in `entry_block`
        // (except `StorageLive(local_id)`).
        for stmt in &entry_block.statements[..init_stmt_idx] {
            if !matches!(stmt, Statement::StorageLive(id, _) if *id == local_id)
                && Self::stmt_mentions_local(stmt, local_id)
            {
                return None;
            }
        }

        match &entry_block.statements[init_stmt_idx] {
            Statement::Assign(
                Place::Local(_),
                Rvalue::Aggregate(MirType::Struct(s_name), ops),
                span,
            ) => {
                if !Self::struct_names_match(s_name, struct_name) || ops.len() != n_fields {
                    return None;
                }
                if ops.iter().any(|op| Self::operand_mentions_local(op, local_id)) {
                    return None;
                }
                Some(EntryInitPlan {
                    init_stmt_idx,
                    agg_stmt_idx: init_stmt_idx,
                    agg_span: *span,
                    field_ops: ops.clone(),
                    tmp_chain: Vec::new(),
                    copy_stmt_indices: Vec::new(),
                })
            }
            Statement::Assign(
                Place::Local(_),
                Rvalue::Use(Operand::Copy(Place::Local(first_tmp)) | Operand::Move(Place::Local(first_tmp))),
                _,
            ) => {
                // Follow single-use temporary chain backwards inside `entry_block`
                // (handles both `MirLowering` struct literals and inlined `Counter.new(...)`).
                let mut cur_tmp = *first_tmp;
                let mut cur_use_idx = init_stmt_idx;
                let mut tmp_chain: Vec<LocalId> = Vec::new();
                let mut copy_stmt_indices: Vec<usize> = vec![init_stmt_idx];

                loop {
                    if cur_tmp == local_id
                        || cur_tmp == func.return_local
                        || func.params.contains(&cur_tmp)
                        || tmp_chain.contains(&cur_tmp)
                    {
                        return None;
                    }
                    let tmp_decl = func.locals.get(cur_tmp.0)?;
                    // Note: unrefined call-destination temporaries from `MirLowering` (e.g.
                    // `local c: Counter = Counter.new(0)` after `MirInliner`) may carry a
                    // placeholder `Int`/`Unknown` declared type before type refinement, while
                    // every hop is strictly a direct `Use(Copy/Move)` from the verified
                    // `Rvalue::Aggregate(MirType::Struct(struct_name), ops)` into `local_id`.
                    if tmp_decl.is_param {
                        return None;
                    }

                    // Locate unique defining statement of `cur_tmp` in `entry_block` before `cur_use_idx`
                    // and verify `cur_tmp` has ZERO other uses anywhere in `func`.
                    let def_idx = Self::verify_single_use_entry_temp(
                        func,
                        entry_idx,
                        cur_tmp,
                        cur_use_idx,
                    )?;
                    tmp_chain.push(cur_tmp);

                    match &entry_block.statements[def_idx] {
                        Statement::Assign(
                            Place::Local(_),
                            Rvalue::Aggregate(MirType::Struct(s_name), ops),
                            span,
                        ) => {
                            if !Self::struct_names_match(s_name, struct_name)
                                || ops.len() != n_fields
                            {
                                return None;
                            }
                            if ops.iter().any(|op| {
                                Self::operand_mentions_local(op, local_id)
                                    || tmp_chain
                                        .iter()
                                        .any(|&t| Self::operand_mentions_local(op, t))
                            }) {
                                return None;
                            }
                            return Some(EntryInitPlan {
                                init_stmt_idx,
                                agg_stmt_idx: def_idx,
                                agg_span: *span,
                                field_ops: ops.clone(),
                                tmp_chain,
                                copy_stmt_indices,
                            });
                        }
                        Statement::Assign(
                            Place::Local(_),
                            Rvalue::Use(
                                Operand::Copy(Place::Local(prev_tmp))
                                | Operand::Move(Place::Local(prev_tmp)),
                            ),
                            _,
                        ) => {
                            copy_stmt_indices.push(def_idx);
                            cur_tmp = *prev_tmp;
                            cur_use_idx = def_idx;
                        }
                        _ => return None,
                    }
                }
            }
            _ => None,
        }
    }

    /// Verifies that `tmp` is only defined once in `entry_idx` before `use_idx`, only read once at
    /// `(entry_idx, use_idx)`, has no `StorageDead(tmp)` strictly between `def_idx` and `use_idx`,
    /// and is never referenced anywhere else in `func`.
    fn verify_single_use_entry_temp(
        func: &MirFunction,
        entry_idx: usize,
        tmp: LocalId,
        use_idx: usize,
    ) -> Option<usize> {
        let mut def_idx: Option<usize> = None;

        for (b_idx, block) in func.blocks.iter().enumerate() {
            for (s_idx, stmt) in block.statements.iter().enumerate() {
                if !Self::stmt_mentions_local(stmt, tmp) {
                    continue;
                }
                if b_idx != entry_idx {
                    return None;
                }
                if s_idx == use_idx {
                    continue;
                }
                match stmt {
                    Statement::StorageLive(id, _) if *id == tmp => {}
                    Statement::StorageDead(id, _) if *id == tmp => {
                        // Must not be killed before it is consumed at `use_idx`.
                        if s_idx < use_idx {
                            return None;
                        }
                    }
                    Statement::Assign(Place::Local(dst), rval, _)
                        if *dst == tmp && !Self::rvalue_mentions_local(rval, tmp) =>
                    {
                        if def_idx.is_some() || s_idx >= use_idx {
                            return None;
                        }
                        def_idx = Some(s_idx);
                    }
                    _ => return None,
                }
            }
            if let Some(term) = &block.terminator {
                if Self::terminator_mentions_local(term, tmp) {
                    return None;
                }
            }
        }

        def_idx
    }

    fn validate_all_uses_and_collect_aliases(
        func: &MirFunction,
        local_id: LocalId,
        struct_name: &str,
        n_fields: usize,
        init_plan: &EntryInitPlan,
    ) -> Option<Vec<(LocalId, usize, usize)>> {
        let entry_idx = func.entry.0;
        let mut candidate_aliases: Vec<(LocalId, usize, usize)> = Vec::new();
        let mut seen_alias_ids: HashSet<LocalId> = HashSet::new();

        for (b_idx, block) in func.blocks.iter().enumerate() {
            for (s_idx, stmt) in block.statements.iter().enumerate() {
                if b_idx == entry_idx && s_idx == init_plan.init_stmt_idx {
                    continue;
                }
                if !Self::stmt_mentions_local(stmt, local_id) {
                    continue;
                }
                match stmt {
                    Statement::StorageLive(id, _) | Statement::StorageDead(id, _)
                        if *id == local_id => {}
                    Statement::Assign(
                        Place::Local(alias_id),
                        Rvalue::Use(
                            Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)),
                        ),
                        _,
                    ) if *src == local_id && *alias_id != local_id => {
                        // Only immutable inlined-parameter copies (`!is_mut`) may fold as aliases.
                        // Mutable user locals (`local b = a`) are rejected so alias semantics
                        // and `Rc::ptr_eq` identity stay untouched.
                        let alias_decl = func.locals.get(alias_id.0)?;
                        let alias_ty_ok = matches!(&alias_decl.ty, MirType::Struct(s) if Self::struct_names_match(s, struct_name));
                        if alias_decl.is_mut
                            || alias_decl.is_param
                            || *alias_id == func.return_local
                            || func.params.contains(alias_id)
                            || init_plan.tmp_chain.contains(alias_id)
                            || !alias_ty_ok
                        {
                            return None;
                        }
                        if b_idx == entry_idx && s_idx <= init_plan.init_stmt_idx {
                            return None;
                        }
                        if !seen_alias_ids.insert(*alias_id) {
                            // Multiple assignments to the same alias local -> reject.
                            return None;
                        }
                        candidate_aliases.push((*alias_id, b_idx, s_idx));
                    }
                    Statement::Assign(place, rval, _) => {
                        if !Self::place_uses_only_direct_fields(place, local_id, n_fields)
                            || !Self::rvalue_uses_only_direct_fields(rval, local_id, n_fields)
                        {
                            return None;
                        }
                    }
                    _ => return None,
                }
            }

            if let Some(term) = &block.terminator {
                if Self::terminator_mentions_local(term, local_id)
                    && !Self::terminator_uses_only_direct_fields(term, local_id, n_fields)
                {
                    return None;
                }
            }
        }

        // Validate every discovered immutable alias across the entire function:
        // it must ONLY be assigned at `(alias_bb, alias_stmt_idx)` and ONLY used via
        // direct `Place::Field(Local(alias_id), k)` (`k < n_fields`) everywhere else.
        for &(alias_id, alias_bb, alias_stmt_idx) in &candidate_aliases {
            for pairs in func.proven_in_bounds.values() {
                for &(a, b) in pairs {
                    if a == alias_id || b == alias_id {
                        return None;
                    }
                }
            }
            for (b_idx, block) in func.blocks.iter().enumerate() {
                for (s_idx, stmt) in block.statements.iter().enumerate() {
                    if b_idx == alias_bb && s_idx == alias_stmt_idx {
                        continue;
                    }
                    if !Self::stmt_mentions_local(stmt, alias_id) {
                        continue;
                    }
                    match stmt {
                        Statement::StorageLive(id, _) | Statement::StorageDead(id, _)
                            if *id == alias_id => {}
                        Statement::Assign(place, rval, _) => {
                            if !Self::place_uses_only_direct_fields(place, alias_id, n_fields)
                                || !Self::rvalue_uses_only_direct_fields(rval, alias_id, n_fields)
                            {
                                return None;
                            }
                        }
                        _ => return None,
                    }
                }
                if let Some(term) = &block.terminator {
                    if Self::terminator_mentions_local(term, alias_id)
                        && !Self::terminator_uses_only_direct_fields(term, alias_id, n_fields)
                    {
                        return None;
                    }
                }
            }
        }

        Some(candidate_aliases)
    }

    // --- Place / Operand / Rvalue / Statement / Terminator inspection helpers ---

    fn struct_names_match(a: &str, b: &str) -> bool {
        if a == b {
            return true;
        }
        let simple_a = a.rsplit_once('.').map(|(_, s)| s).unwrap_or(a);
        let simple_b = b.rsplit_once('.').map(|(_, s)| s).unwrap_or(b);
        simple_a == simple_b && (a == simple_a || b == simple_b)
    }

    fn place_mentions_local(place: &Place, target: LocalId) -> bool {
        match place {
            Place::Local(id) => *id == target,
            Place::Field(base, _) => Self::place_mentions_local(base, target),
            Place::Index(base, idx) => *idx == target || Self::place_mentions_local(base, target),
        }
    }

    fn operand_mentions_local(op: &Operand, target: LocalId) -> bool {
        match op {
            Operand::Copy(p) | Operand::Move(p) => Self::place_mentions_local(p, target),
            Operand::Constant(_) => false,
        }
    }

    fn rvalue_mentions_local(rval: &Rvalue, target: LocalId) -> bool {
        match rval {
            Rvalue::Use(op) | Rvalue::UnaryOp(_, op) | Rvalue::Cast(op, _) => {
                Self::operand_mentions_local(op, target)
            }
            Rvalue::BinaryOp(_, l, r) => {
                Self::operand_mentions_local(l, target) || Self::operand_mentions_local(r, target)
            }
            Rvalue::Aggregate(_, ops) => ops.iter().any(|o| Self::operand_mentions_local(o, target)),
            Rvalue::Closure { captures, .. } => captures
                .iter()
                .any(|(_, o, _)| Self::operand_mentions_local(o, target)),
        }
    }

    fn stmt_mentions_local(stmt: &Statement, target: LocalId) -> bool {
        match stmt {
            Statement::Assign(p, r, _) => {
                Self::place_mentions_local(p, target) || Self::rvalue_mentions_local(r, target)
            }
            Statement::StorageLive(id, _) | Statement::StorageDead(id, _) => *id == target,
            Statement::ResourceCheck(_, _, _) => false,
        }
    }

    fn terminator_mentions_local(term: &Terminator, target: LocalId) -> bool {
        match term {
            Terminator::Goto(_, _) | Terminator::Unreachable(_) => false,
            Terminator::Cond { cond, .. } => Self::operand_mentions_local(cond, target),
            Terminator::Switch { discr, .. } => Self::operand_mentions_local(discr, target),
            Terminator::Call {
                args, destination, ..
            } => {
                Self::place_mentions_local(destination, target)
                    || args.iter().any(|a| Self::operand_mentions_local(a, target))
            }
            Terminator::Return(Some(op), _) | Terminator::Raise(op, _) => {
                Self::operand_mentions_local(op, target)
            }
            Terminator::Return(None, _) => false,
        }
    }

    /// Returns `true` iff every occurrence of `target` inside `place` is strictly a direct
    /// top-level field projection `Place::Field(Box::new(Place::Local(target)), k)` with `k < n_fields`.
    fn place_uses_only_direct_fields(place: &Place, target: LocalId, n_fields: usize) -> bool {
        match place {
            Place::Local(id) => *id != target,
            Place::Field(base, field_idx) => match base.as_ref() {
                Place::Local(id) if *id == target => *field_idx < n_fields,
                _ => !Self::place_mentions_local(base, target),
            },
            Place::Index(base, idx) => *idx != target && !Self::place_mentions_local(base, target),
        }
    }

    fn operand_uses_only_direct_fields(op: &Operand, target: LocalId, n_fields: usize) -> bool {
        match op {
            Operand::Copy(p) | Operand::Move(p) => {
                Self::place_uses_only_direct_fields(p, target, n_fields)
            }
            Operand::Constant(_) => true,
        }
    }

    fn rvalue_uses_only_direct_fields(rval: &Rvalue, target: LocalId, n_fields: usize) -> bool {
        match rval {
            Rvalue::Use(op) | Rvalue::UnaryOp(_, op) | Rvalue::Cast(op, _) => {
                Self::operand_uses_only_direct_fields(op, target, n_fields)
            }
            Rvalue::BinaryOp(_, l, r) => {
                Self::operand_uses_only_direct_fields(l, target, n_fields)
                    && Self::operand_uses_only_direct_fields(r, target, n_fields)
            }
            Rvalue::Aggregate(_, ops) => ops
                .iter()
                .all(|o| Self::operand_uses_only_direct_fields(o, target, n_fields)),
            Rvalue::Closure { captures, .. } => captures
                .iter()
                .all(|(_, o, _)| Self::operand_uses_only_direct_fields(o, target, n_fields)),
        }
    }

    fn terminator_uses_only_direct_fields(
        term: &Terminator,
        target: LocalId,
        n_fields: usize,
    ) -> bool {
        match term {
            Terminator::Goto(_, _) | Terminator::Unreachable(_) => true,
            Terminator::Cond { cond, .. } => {
                Self::operand_uses_only_direct_fields(cond, target, n_fields)
            }
            Terminator::Switch { discr, .. } => {
                Self::operand_uses_only_direct_fields(discr, target, n_fields)
            }
            Terminator::Call {
                args, destination, ..
            } => {
                Self::place_uses_only_direct_fields(destination, target, n_fields)
                    && args
                        .iter()
                        .all(|a| Self::operand_uses_only_direct_fields(a, target, n_fields))
            }
            Terminator::Return(Some(op), _) | Terminator::Raise(op, _) => {
                Self::operand_uses_only_direct_fields(op, target, n_fields)
            }
            Terminator::Return(None, _) => true,
        }
    }

    // --- Place / Operand / Rvalue / Statement / Terminator rewrite helpers ---

    fn rewrite_place(
        place: &mut Place,
        bases: &HashSet<LocalId>,
        field_locals: &[LocalId],
    ) {
        if let Place::Field(base, field_idx) = place {
            if let Place::Local(id) = base.as_ref() {
                if bases.contains(id) && *field_idx < field_locals.len() {
                    *place = Place::Local(field_locals[*field_idx]);
                    return;
                }
            }
        }
        match place {
            Place::Local(_) => {}
            Place::Field(base, _) | Place::Index(base, _) => {
                Self::rewrite_place(base, bases, field_locals);
            }
        }
    }

    fn rewrite_operand(
        op: &mut Operand,
        bases: &HashSet<LocalId>,
        field_locals: &[LocalId],
    ) {
        match op {
            Operand::Copy(p) | Operand::Move(p) => Self::rewrite_place(p, bases, field_locals),
            Operand::Constant(_) => {}
        }
    }

    fn rewrite_rvalue(
        rval: &mut Rvalue,
        bases: &HashSet<LocalId>,
        field_locals: &[LocalId],
    ) {
        match rval {
            Rvalue::Use(op) | Rvalue::UnaryOp(_, op) | Rvalue::Cast(op, _) => {
                Self::rewrite_operand(op, bases, field_locals);
            }
            Rvalue::BinaryOp(_, l, r) => {
                Self::rewrite_operand(l, bases, field_locals);
                Self::rewrite_operand(r, bases, field_locals);
            }
            Rvalue::Aggregate(_, ops) => {
                for op in ops {
                    Self::rewrite_operand(op, bases, field_locals);
                }
            }
            Rvalue::Closure { captures, .. } => {
                for (_, op, _) in captures {
                    Self::rewrite_operand(op, bases, field_locals);
                }
            }
        }
    }

    fn rewrite_statement(
        stmt: &mut Statement,
        bases: &HashSet<LocalId>,
        field_locals: &[LocalId],
    ) {
        if let Statement::Assign(place, rval, _) = stmt {
            Self::rewrite_place(place, bases, field_locals);
            Self::rewrite_rvalue(rval, bases, field_locals);
        }
    }

    fn rewrite_terminator(
        term: &mut Terminator,
        bases: &HashSet<LocalId>,
        field_locals: &[LocalId],
    ) {
        match term {
            Terminator::Goto(_, _) | Terminator::Unreachable(_) => {}
            Terminator::Cond { cond, .. } => Self::rewrite_operand(cond, bases, field_locals),
            Terminator::Switch { discr, .. } => Self::rewrite_operand(discr, bases, field_locals),
            Terminator::Call {
                args, destination, ..
            } => {
                Self::rewrite_place(destination, bases, field_locals);
                for arg in args {
                    Self::rewrite_operand(arg, bases, field_locals);
                }
            }
            Terminator::Return(Some(op), _) | Terminator::Raise(op, _) => {
                Self::rewrite_operand(op, bases, field_locals);
            }
            Terminator::Return(None, _) => {}
        }
    }
    /// Closed-component multi-def & loop-body struct SROA (`NG-W15-2`):
    ///
    /// Decomposes non-escaping connected copy/move components of scalar-field structs
    /// even when locals are defined inside loop bodies or reassigned multiple times
    /// (e.g. `cur = transform(cur, i)` in `BENCH-NG-07`, `TelemetryEvent` in `BENCH-NG-13`,
    /// `OrderTick` in `BENCH-NG-15`).
    ///
    /// Reference-aliasing soundness:
    /// - If the component has ZERO field mutations (`Place::Field(Local(x), k) = ...`),
    ///   every struct instance in the component is immutable and reference copy is
    ///   observationally identical to field-wise value copy.
    /// - If the component has field mutations AND copy edges `dst = Copy/Move(src)`,
    ///   every copied `src` must be a same-block single-def, single-use temporary with
    ///   zero field accesses (strictly rejecting mutable user aliases like `local b = a`).
    fn run_multi_def_struct_sroa(
        func: &mut MirFunction,
        struct_layouts: &HashMap<String, MirStructLayout>,
    ) -> bool {
        if func.blocks.is_empty() || func.entry.0 >= func.blocks.len() || struct_layouts.is_empty()
        {
            return false;
        }

        let mut cleanup_block_set: HashSet<usize> = HashSet::new();
        for (b_idx, blk) in func.blocks.iter().enumerate() {
            if blk.is_cleanup {
                cleanup_block_set.insert(b_idx);
            }
        }
        for entry in &func.cleanup_blocks {
            cleanup_block_set.insert(entry.block.0);
        }

        let mut pib_locals: HashSet<LocalId> = HashSet::new();
        for pairs in func.proven_in_bounds.values() {
            for &(a, b) in pairs {
                pib_locals.insert(a);
                pib_locals.insert(b);
            }
        }

        let mut already_sroa: HashSet<LocalId> = func
            .sroa_decomposed_structs
            .iter()
            .map(|r| r.orig_local)
            .collect();

        // Collect all candidate struct type names appearing in locals or Aggregates.
        let mut candidate_structs: Vec<String> = Vec::new();
        let mut seen_structs: HashSet<String> = HashSet::new();
        for decl in &func.locals {
            if let MirType::Struct(s) = &decl.ty {
                if !s.is_empty() && seen_structs.insert(s.clone()) {
                    candidate_structs.push(s.clone());
                }
            }
        }
        for block in &func.blocks {
            for stmt in &block.statements {
                if let Statement::Assign(_, Rvalue::Aggregate(MirType::Struct(s), _), _) = stmt {
                    if !s.is_empty() && seen_structs.insert(s.clone()) {
                        candidate_structs.push(s.clone());
                    }
                }
            }
        }

        let mut overall_changed = false;

        for struct_name in candidate_structs {
            let Some(layout) = Self::lookup_struct_layout(struct_layouts, &struct_name) else {
                continue;
            };
            let Some((field_types, field_names)) = Self::resolve_scalar_fields(layout) else {
                continue;
            };
            let n_fields = field_types.len();

            // Build undirected copy/move adjacency across all locals in `func`.
            let mut copy_adj: HashMap<LocalId, Vec<LocalId>> = HashMap::new();
            let mut seed_locals: Vec<LocalId> = Vec::new();
            let mut seed_set: HashSet<LocalId> = HashSet::new();

            for (idx, decl) in func.locals.iter().enumerate() {
                let id = LocalId(idx);
                if already_sroa.contains(&id) {
                    continue;
                }
                if let MirType::Struct(s) = &decl.ty {
                    if Self::struct_names_match(s, &struct_name) && seed_set.insert(id) {
                        seed_locals.push(id);
                    }
                }
            }

            for block in &func.blocks {
                for stmt in &block.statements {
                    match stmt {
                        Statement::Assign(
                            Place::Local(dst),
                            Rvalue::Aggregate(MirType::Struct(s), ops),
                            _,
                        ) if Self::struct_names_match(s, &struct_name) && ops.len() == n_fields => {
                            if !already_sroa.contains(dst) && seed_set.insert(*dst) {
                                seed_locals.push(*dst);
                            }
                        }
                        Statement::Assign(
                            Place::Local(dst),
                            Rvalue::Use(
                                Operand::Copy(Place::Local(src))
                                | Operand::Move(Place::Local(src)),
                            ),
                            _,
                        ) if *dst != *src => {
                            copy_adj.entry(*dst).or_default().push(*src);
                            copy_adj.entry(*src).or_default().push(*dst);
                        }
                        _ => {}
                    }
                }
            }

            let mut visited: HashSet<LocalId> = HashSet::new();
            for seed in seed_locals {
                if !visited.insert(seed) {
                    continue;
                }

                // BFS to form the full connected copy/move component.
                let mut comp: Vec<LocalId> = alloc::vec![seed];
                let mut queue_idx = 0;
                while queue_idx < comp.len() {
                    let cur = comp[queue_idx];
                    queue_idx += 1;
                    if let Some(neighbors) = copy_adj.get(&cur) {
                        for &nb in neighbors {
                            if visited.insert(nb) {
                                comp.push(nb);
                            }
                        }
                    }
                }
                comp.sort_unstable_by_key(|id| id.0);
                let comp_set: HashSet<LocalId> = comp.iter().copied().collect();

                // 1. Validate local declarations in `comp`.
                let mut decls_ok = !comp.is_empty();
                for &x in &comp {
                    let Some(decl) = func.locals.get(x.0) else {
                        decls_ok = false;
                        break;
                    };
                    if decl.is_param
                        || func.params.contains(&x)
                        || x == func.return_local
                        || already_sroa.contains(&x)
                        || pib_locals.contains(&x)
                    {
                        decls_ok = false;
                        break;
                    }
                    if let MirType::Struct(other_s) = &decl.ty {
                        if !Self::struct_names_match(other_s, &struct_name) {
                            decls_ok = false;
                            break;
                        }
                    }
                }
                if !decls_ok {
                    continue;
                }

                // 2. Validate every statement and terminator in `func` mentioning any `x in comp`.
                let mut valid = true;
                let mut has_aggregate = false;
                let mut has_field_mutation = false;
                let mut needs_staging = false;
                let mut whole_defs: HashMap<LocalId, Vec<(usize, usize)>> = HashMap::new();
                let mut copy_edges: Vec<(LocalId, LocalId, usize, usize)> = Vec::new();
                let mut whole_copy_reads: HashMap<LocalId, usize> = HashMap::new();
                let mut field_access_counts: HashMap<LocalId, usize> = HashMap::new();

                'blocks: for (b_idx, block) in func.blocks.iter().enumerate() {
                    let is_cleanup_blk = cleanup_block_set.contains(&b_idx);
                    for (s_idx, stmt) in block.statements.iter().enumerate() {
                        if !comp.iter().any(|&x| Self::stmt_mentions_local(stmt, x)) {
                            continue;
                        }
                        if is_cleanup_blk {
                            valid = false;
                            break 'blocks;
                        }
                        match stmt {
                            Statement::StorageLive(id, _) | Statement::StorageDead(id, _)
                                if comp_set.contains(id) => {}
                            Statement::Assign(
                                Place::Local(dst),
                                Rvalue::Aggregate(MirType::Struct(s), ops),
                                _,
                            ) if comp_set.contains(dst) => {
                                if !Self::struct_names_match(s, &struct_name)
                                    || ops.len() != n_fields
                                {
                                    valid = false;
                                    break 'blocks;
                                }
                                for op in ops {
                                    for &x in &comp {
                                        if Self::operand_mentions_local(op, x) {
                                            if !Self::operand_uses_only_direct_fields(
                                                op, x, n_fields,
                                            ) {
                                                valid = false;
                                                break 'blocks;
                                            }
                                            *field_access_counts.entry(x).or_insert(0) += 1;
                                            if x == *dst {
                                                needs_staging = true;
                                            }
                                        }
                                    }
                                }
                                whole_defs.entry(*dst).or_default().push((b_idx, s_idx));
                                has_aggregate = true;
                            }
                            Statement::Assign(
                                Place::Local(dst),
                                Rvalue::Use(
                                    Operand::Copy(Place::Local(src))
                                    | Operand::Move(Place::Local(src)),
                                ),
                                _,
                            ) if comp_set.contains(dst) || comp_set.contains(src) => {
                                if !comp_set.contains(dst) || !comp_set.contains(src) {
                                    valid = false;
                                    break 'blocks;
                                }
                                whole_defs.entry(*dst).or_default().push((b_idx, s_idx));
                                if *dst != *src {
                                    copy_edges.push((*dst, *src, b_idx, s_idx));
                                    *whole_copy_reads.entry(*src).or_insert(0) += 1;
                                }
                            }
                            Statement::Assign(place, rval, _) => {
                                if let Place::Field(base, field_idx) = place {
                                    if let Place::Local(x) = base.as_ref() {
                                        if comp_set.contains(x) {
                                            if *field_idx >= n_fields {
                                                valid = false;
                                                break 'blocks;
                                            }
                                            has_field_mutation = true;
                                            *field_access_counts.entry(*x).or_insert(0) += 1;
                                        }
                                    }
                                }
                                for &x in &comp {
                                    if Self::place_mentions_local(place, x) {
                                        if !Self::place_uses_only_direct_fields(place, x, n_fields)
                                        {
                                            valid = false;
                                            break 'blocks;
                                        }
                                    }
                                    if Self::rvalue_mentions_local(rval, x) {
                                        if !Self::rvalue_uses_only_direct_fields(rval, x, n_fields)
                                        {
                                            valid = false;
                                            break 'blocks;
                                        }
                                        *field_access_counts.entry(x).or_insert(0) += 1;
                                    }
                                }
                            }
                            _ => {
                                valid = false;
                                break 'blocks;
                            }
                        }
                    }

                    if let Some(term) = &block.terminator {
                        if comp.iter().any(|&x| Self::terminator_mentions_local(term, x)) {
                            if is_cleanup_blk {
                                valid = false;
                                break 'blocks;
                            }
                            if let Terminator::Call {
                                destination: Place::Field(base, _),
                                ..
                            } = term
                            {
                                if let Place::Local(x) = base.as_ref() {
                                    if comp_set.contains(x) {
                                        has_field_mutation = true;
                                    }
                                }
                            }
                            for &x in &comp {
                                if Self::terminator_mentions_local(term, x) {
                                    if !Self::terminator_uses_only_direct_fields(term, x, n_fields)
                                    {
                                        valid = false;
                                        break 'blocks;
                                    }
                                    *field_access_counts.entry(x).or_insert(0) += 1;
                                }
                            }
                        }
                    }
                }

                if !valid || !has_aggregate {
                    continue;
                }

                // 3. Enforce reference-aliasing soundness when any field mutation occurs.
                if has_field_mutation && !copy_edges.is_empty() {
                    let mut aliases_sound = true;
                    for &(_dst, src, use_bb, use_stmt) in &copy_edges {
                        let defs = whole_defs.get(&src).map(|v| v.as_slice()).unwrap_or(&[]);
                        let copy_reads = whole_copy_reads.get(&src).copied().unwrap_or(0);
                        let field_uses = field_access_counts.get(&src).copied().unwrap_or(0);
                        if defs.len() != 1
                            || copy_reads != 1
                            || field_uses != 0
                            || defs[0].0 != use_bb
                            || defs[0].1 >= use_stmt
                        {
                            aliases_sound = false;
                            break;
                        }
                    }
                    if !aliases_sound {
                        continue;
                    }
                }

                // 4. Enforce hard 250-local cap.
                let needed_locals = comp
                    .len()
                    .saturating_mul(n_fields)
                    .saturating_add(if needs_staging { n_fields } else { 0 });
                if func.locals.len().saturating_add(needed_locals) > SROA_MAX_LOCALS_LIMIT {
                    continue;
                }

                // --- Component validated! Allocate SSA scalar replacements and rewrite MIR ---
                let mut field_locals_map: HashMap<LocalId, Vec<LocalId>> =
                    HashMap::with_capacity(comp.len());
                for &x in &comp {
                    let x_span = func.locals[x.0].span;
                    let mut fls = Vec::with_capacity(n_fields);
                    for (k, field_ty) in field_types.iter().enumerate() {
                        let fl_id = LocalId(func.locals.len());
                        func.locals.push(LocalDecl {
                            id: fl_id,
                            name: Some(format!("__sroa_{}_{}", x.0, k)),
                            ty: field_ty.clone(),
                            is_mut: true,
                            span: x_span,
                            is_stack_promoted: false,
                            is_param: false,
                        });
                        fls.push(fl_id);
                    }
                    field_locals_map.insert(x, fls);
                }

                let stage_locals: Vec<LocalId> = if needs_staging {
                    let span0 = func.locals[comp[0].0].span;
                    let mut sls = Vec::with_capacity(n_fields);
                    for (k, field_ty) in field_types.iter().enumerate() {
                        let sl_id = LocalId(func.locals.len());
                        func.locals.push(LocalDecl {
                            id: sl_id,
                            name: Some(format!("__sroa_stage_{}_{}", comp[0].0, k)),
                            ty: field_ty.clone(),
                            is_mut: true,
                            span: span0,
                            is_stack_promoted: false,
                            is_param: false,
                        });
                        sls.push(sl_id);
                    }
                    sls
                } else {
                    Vec::new()
                };

                for block in &mut func.blocks {
                    let mut new_stmts = Vec::with_capacity(block.statements.len() + n_fields * 2);
                    for stmt in &block.statements {
                        match stmt {
                            Statement::StorageLive(id, span) if comp_set.contains(id) => {
                                for &fl in &field_locals_map[id] {
                                    new_stmts.push(Statement::StorageLive(fl, *span));
                                }
                            }
                            Statement::StorageDead(id, span) if comp_set.contains(id) => {
                                for &fl in &field_locals_map[id] {
                                    new_stmts.push(Statement::StorageDead(fl, *span));
                                }
                            }
                            Statement::Assign(
                                Place::Local(dst),
                                Rvalue::Aggregate(MirType::Struct(_), ops),
                                span,
                            ) if comp_set.contains(dst) => {
                                let dst_fields = &field_locals_map[dst];
                                let self_ref =
                                    ops.iter().any(|op| Self::operand_mentions_local(op, *dst));
                                if self_ref {
                                    for (k, op) in ops.iter().enumerate() {
                                        let mut rewritten_op = op.clone();
                                        Self::rewrite_operand_multi(
                                            &mut rewritten_op,
                                            &field_locals_map,
                                        );
                                        new_stmts.push(Statement::StorageLive(
                                            stage_locals[k],
                                            *span,
                                        ));
                                        new_stmts.push(Statement::Assign(
                                            Place::Local(stage_locals[k]),
                                            Rvalue::Use(rewritten_op),
                                            *span,
                                        ));
                                    }
                                    for k in 0..n_fields {
                                        new_stmts.push(Statement::Assign(
                                            Place::Local(dst_fields[k]),
                                            Rvalue::Use(Operand::Move(Place::Local(
                                                stage_locals[k],
                                            ))),
                                            *span,
                                        ));
                                        new_stmts.push(Statement::StorageDead(
                                            stage_locals[k],
                                            *span,
                                        ));
                                    }
                                } else {
                                    for (k, op) in ops.iter().enumerate() {
                                        let mut rewritten_op = op.clone();
                                        Self::rewrite_operand_multi(
                                            &mut rewritten_op,
                                            &field_locals_map,
                                        );
                                        new_stmts.push(Statement::Assign(
                                            Place::Local(dst_fields[k]),
                                            Rvalue::Use(rewritten_op),
                                            *span,
                                        ));
                                    }
                                }
                            }
                            Statement::Assign(
                                Place::Local(dst),
                                Rvalue::Use(
                                    Operand::Copy(Place::Local(src))
                                    | Operand::Move(Place::Local(src)),
                                ),
                                span,
                            ) if comp_set.contains(dst) && comp_set.contains(src) => {
                                if *dst != *src {
                                    let dst_fields = &field_locals_map[dst];
                                    let src_fields = &field_locals_map[src];
                                    for k in 0..n_fields {
                                        new_stmts.push(Statement::Assign(
                                            Place::Local(dst_fields[k]),
                                            Rvalue::Use(Operand::Copy(Place::Local(src_fields[k]))),
                                            *span,
                                        ));
                                    }
                                }
                            }
                            _ => {
                                let mut cloned = stmt.clone();
                                Self::rewrite_statement_multi(&mut cloned, &field_locals_map);
                                new_stmts.push(cloned);
                            }
                        }
                    }
                    block.statements = new_stmts;

                    if let Some(term) = &mut block.terminator {
                        Self::rewrite_terminator_multi(term, &field_locals_map);
                    }
                }

                // Record cold deopt reconstruction recipes for all struct-typed locals in `comp`.
                for &x in &comp {
                    already_sroa.insert(x);
                    if let MirType::Struct(s_name) = &func.locals[x.0].ty {
                        func.sroa_decomposed_structs.push(SroaStructRecipe {
                            orig_local: x,
                            struct_name: s_name.clone(),
                            field_names: field_names.clone(),
                            field_locals: field_locals_map[&x].clone(),
                        });
                    }
                }

                overall_changed = true;
            }
        }

        overall_changed
    }

    fn rewrite_place_multi(
        place: &mut Place,
        field_locals_map: &HashMap<LocalId, Vec<LocalId>>,
    ) {
        if let Place::Field(base, field_idx) = place {
            if let Place::Local(id) = base.as_ref() {
                if let Some(fls) = field_locals_map.get(id) {
                    if *field_idx < fls.len() {
                        *place = Place::Local(fls[*field_idx]);
                        return;
                    }
                }
            }
        }
        match place {
            Place::Local(_) => {}
            Place::Field(base, _) | Place::Index(base, _) => {
                Self::rewrite_place_multi(base, field_locals_map);
            }
        }
    }

    fn rewrite_operand_multi(
        op: &mut Operand,
        field_locals_map: &HashMap<LocalId, Vec<LocalId>>,
    ) {
        match op {
            Operand::Copy(p) | Operand::Move(p) => Self::rewrite_place_multi(p, field_locals_map),
            Operand::Constant(_) => {}
        }
    }

    fn rewrite_rvalue_multi(
        rval: &mut Rvalue,
        field_locals_map: &HashMap<LocalId, Vec<LocalId>>,
    ) {
        match rval {
            Rvalue::Use(op) | Rvalue::UnaryOp(_, op) | Rvalue::Cast(op, _) => {
                Self::rewrite_operand_multi(op, field_locals_map);
            }
            Rvalue::BinaryOp(_, l, r) => {
                Self::rewrite_operand_multi(l, field_locals_map);
                Self::rewrite_operand_multi(r, field_locals_map);
            }
            Rvalue::Aggregate(_, ops) => {
                for op in ops {
                    Self::rewrite_operand_multi(op, field_locals_map);
                }
            }
            Rvalue::Closure { captures, .. } => {
                for (_, op, _) in captures {
                    Self::rewrite_operand_multi(op, field_locals_map);
                }
            }
        }
    }

    fn rewrite_statement_multi(
        stmt: &mut Statement,
        field_locals_map: &HashMap<LocalId, Vec<LocalId>>,
    ) {
        if let Statement::Assign(place, rval, _) = stmt {
            Self::rewrite_place_multi(place, field_locals_map);
            Self::rewrite_rvalue_multi(rval, field_locals_map);
        }
    }

    fn rewrite_terminator_multi(
        term: &mut Terminator,
        field_locals_map: &HashMap<LocalId, Vec<LocalId>>,
    ) {
        match term {
            Terminator::Goto(_, _) | Terminator::Unreachable(_) => {}
            Terminator::Cond { cond, .. } => Self::rewrite_operand_multi(cond, field_locals_map),
            Terminator::Switch { discr, .. } => {
                Self::rewrite_operand_multi(discr, field_locals_map)
            }
            Terminator::Call {
                args, destination, ..
            } => {
                Self::rewrite_place_multi(destination, field_locals_map);
                for arg in args {
                    Self::rewrite_operand_multi(arg, field_locals_map);
                }
            }
            Terminator::Return(Some(op), _) | Terminator::Raise(op, _) => {
                Self::rewrite_operand_multi(op, field_locals_map);
            }
            Terminator::Return(None, _) => {}
        }
    }
}

struct EntryInitPlan {
    init_stmt_idx: usize,
    agg_stmt_idx: usize,
    agg_span: Span,
    field_ops: Vec<Operand>,
    tmp_chain: Vec<LocalId>,
    copy_stmt_indices: Vec<usize>,
}

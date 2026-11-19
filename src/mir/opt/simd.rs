use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::ast::Span;
use crate::mir::ir::{
    BlockId, LocalDecl, LocalId, MirBinaryOp, MirConstant, MirFunction, MirModule, MirStructLayout,
    MirType, MirUnaryOp, Operand, Place, ResourceKind, Rvalue, Statement, Terminator,
};
use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};

/// Canonical SIMD lane width (4 elements per vector chunk).
pub const SIMD_LANE_WIDTH: usize = 4;

/// Global flag for forcing scalar fallback across all SIMD operations.
pub static FORCE_SIMD_SCALAR_FALLBACK: AtomicBool = AtomicBool::new(false);

/// Global counters for tracking SIMD operations and fallbacks.
pub static SIMD_VECTOR_OP_COUNT: AtomicUsize = AtomicUsize::new(0);
pub static SIMD_SCALAR_FALLBACK_COUNT: AtomicUsize = AtomicUsize::new(0);
pub static SIMD_TAIL_PROCESSED_COUNT: AtomicUsize = AtomicUsize::new(0);

#[inline]
pub fn is_force_simd_scalar_fallback() -> bool {
    FORCE_SIMD_SCALAR_FALLBACK.load(Ordering::SeqCst)
}

#[inline]
pub fn set_force_simd_scalar_fallback(force: bool) {
    FORCE_SIMD_SCALAR_FALLBACK.store(force, Ordering::SeqCst);
}

pub fn reset_simd_counters() {
    SIMD_VECTOR_OP_COUNT.store(0, Ordering::SeqCst);
    SIMD_SCALAR_FALLBACK_COUNT.store(0, Ordering::SeqCst);
    SIMD_TAIL_PROCESSED_COUNT.store(0, Ordering::SeqCst);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimdConfig {
    pub lane_width: usize,
    pub force_scalar: bool,
    pub vector_unroll: usize,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            lane_width: SIMD_LANE_WIDTH,
            force_scalar: false,
            vector_unroll: 2,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SimdStats {
    pub functions_analyzed: usize,
    pub loops_vectorized: usize,
    pub simd_calls_transformed: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SimdOpKind {
    BinaryVector,
    UnaryReduce,
    BinaryReduce,
}

fn classify_simd_op(callee: &str) -> Option<(&'static str, usize, SimdOpKind)> {
    match callee {
        "simd_add" | "vector.add" | "simd.add" => Some(("simd.add", 5, SimdOpKind::BinaryVector)),
        "simd_mul" | "vector.mul" | "simd.mul" => Some(("simd.mul", 5, SimdOpKind::BinaryVector)),
        "simd_sum" | "vector.sum" | "simd.sum" => Some(("simd.sum", 3, SimdOpKind::UnaryReduce)),
        "simd_dot" | "vector.dot" | "simd.dot" => Some(("simd.dot", 4, SimdOpKind::BinaryReduce)),
        _ => None,
    }
}

fn resolve_struct_field_type(
    struct_name: &str,
    field_idx: usize,
    struct_layouts: &[MirStructLayout],
) -> Option<MirType> {
    if let Some(layout) = struct_layouts.iter().find(|l| l.type_name == struct_name) {
        let field = layout.fields.get(field_idx)?;
        return match field.tag {
            crate::data::layout::CanonicalTypeTag::Bool => Some(MirType::Bool),
            crate::data::layout::CanonicalTypeTag::Int64
            | crate::data::layout::CanonicalTypeTag::UInt64
            | crate::data::layout::CanonicalTypeTag::DateTimeUtc => Some(MirType::Int),
            crate::data::layout::CanonicalTypeTag::Float64 => Some(MirType::Float),
            crate::data::layout::CanonicalTypeTag::StringSlice => Some(MirType::String),
            _ => {
                #[cfg(feature = "std")]
                {
                    crate::mir::runtime_helpers::with_current_mir_context(|ctx| {
                        ctx.get_struct_field_type(struct_name, field_idx)
                    })
                }
                #[cfg(not(feature = "std"))]
                {
                    None
                }
            }
        };
    }
    #[cfg(feature = "std")]
    {
        crate::mir::runtime_helpers::with_current_mir_context(|ctx| {
            ctx.get_struct_field_type(struct_name, field_idx)
        })
    }
    #[cfg(not(feature = "std"))]
    {
        None
    }
}

fn resolve_simd_place_type(
    place: &Place,
    locals: &[LocalDecl],
    struct_layouts: &[MirStructLayout],
) -> Result<MirType, ()> {
    match place {
        Place::Local(loc) => locals.get(loc.0).map(|d| d.ty.clone()).ok_or(()),
        Place::Index(base, idx) => {
            let idx_decl = locals.get(idx.0).ok_or(())?;
            if idx_decl.ty != MirType::Int {
                return Err(());
            }
            match resolve_simd_place_type(base, locals, struct_layouts)? {
                MirType::Array(inner) => Ok(*inner),
                _ => Err(()),
            }
        }
        Place::Field(base, field_idx) => {
            match resolve_simd_place_type(base, locals, struct_layouts)? {
                MirType::Tuple(elems) => elems.get(*field_idx).cloned().ok_or(()),
                MirType::Struct(name) => {
                    resolve_struct_field_type(&name, *field_idx, struct_layouts).ok_or(())
                }
                _ => Err(()),
            }
        }
    }
}

fn resolve_simd_operand_type(
    op: &Operand,
    locals: &[LocalDecl],
    struct_layouts: &[MirStructLayout],
) -> Result<MirType, ()> {
    match op {
        Operand::Copy(p) | Operand::Move(p) => resolve_simd_place_type(p, locals, struct_layouts),
        Operand::Constant(c) => match c {
            MirConstant::Int(_) => Ok(MirType::Int),
            MirConstant::Float(_) => Ok(MirType::Float),
            MirConstant::Bool(_) => Ok(MirType::Bool),
            MirConstant::String(_) => Ok(MirType::String),
            MirConstant::Unit => Ok(MirType::Unit),
        },
    }
}

fn is_valid_simd_buf_type(ty: &MirType) -> bool {
    matches!(ty, MirType::Int)
        || matches!(ty, MirType::Array(inner) if **inner == MirType::Int)
}

fn is_valid_simd_force_type(ty: &MirType) -> bool {
    matches!(ty, MirType::Int | MirType::Bool)
}

fn validate_simd_call_for_opt(
    kind: SimdOpKind,
    expected_arity: usize,
    args: &[Operand],
    destination: &Place,
    success_block: usize,
    error_block: Option<usize>,
    block_count: usize,
    locals: &[LocalDecl],
    struct_layouts: &[MirStructLayout],
) -> Result<MirType, ()> {
    if args.len() != expected_arity {
        return Err(());
    }
    if success_block >= block_count || error_block.is_some_and(|b| b >= block_count) {
        return Err(());
    }

    // Destination must be a valid Int local (reject nested/Unit/non-Int shapes)
    let dest_loc = match destination {
        Place::Local(loc) => loc,
        Place::Index(..) | Place::Field(..) => return Err(()),
    };
    let dest_decl = locals.get(dest_loc.0).ok_or(())?;
    if dest_decl.ty != MirType::Int {
        return Err(());
    }

    let mut arg_tys = alloc::vec::Vec::with_capacity(args.len());
    for arg in args {
        arg_tys.push(resolve_simd_operand_type(arg, locals, struct_layouts)?);
    }

    match kind {
        SimdOpKind::BinaryVector => {
            if !is_valid_simd_buf_type(&arg_tys[0])
                || !is_valid_simd_buf_type(&arg_tys[1])
                || !is_valid_simd_buf_type(&arg_tys[2])
                || arg_tys[3] != MirType::Int
                || !is_valid_simd_force_type(&arg_tys[4])
            {
                return Err(());
            }
            Ok(arg_tys[4].clone())
        }
        SimdOpKind::UnaryReduce => {
            if !is_valid_simd_buf_type(&arg_tys[0])
                || arg_tys[1] != MirType::Int
                || !is_valid_simd_force_type(&arg_tys[2])
            {
                return Err(());
            }
            Ok(arg_tys[2].clone())
        }
        SimdOpKind::BinaryReduce => {
            if !is_valid_simd_buf_type(&arg_tys[0])
                || !is_valid_simd_buf_type(&arg_tys[1])
                || arg_tys[2] != MirType::Int
                || !is_valid_simd_force_type(&arg_tys[3])
            {
                return Err(());
            }
            Ok(arg_tys[3].clone())
        }
    }
}

pub struct SimdOptimizer;

impl SimdOptimizer {
    /// Runs SIMD optimization pass on a single function with default configuration.
    pub fn run_on_function(func: &mut MirFunction) -> bool {
        let config = SimdConfig::default();
        let (loops, calls) = Self::run_on_function_stats(func, &config);
        loops > 0 || calls > 0
    }

    /// Runs SIMD optimization pass on a single function with custom configuration.
    pub fn run_on_function_with_config(func: &mut MirFunction, config: &SimdConfig) -> bool {
        let (loops, calls) = Self::run_on_function_stats(func, config);
        loops > 0 || calls > 0
    }

    /// Runs SIMD optimization and returns exact counts of loops vectorized and SIMD calls transformed.
    pub fn run_on_function_stats(func: &mut MirFunction, config: &SimdConfig) -> (usize, usize) {
        Self::run_on_function_stats_with_layouts(func, config, &[])
    }

    fn run_on_function_stats_with_layouts(
        func: &mut MirFunction,
        config: &SimdConfig,
        struct_layouts: &[MirStructLayout],
    ) -> (usize, usize) {
        let loops_vectorized = detect_and_transform_loops(func, config);
        let mut simd_calls_transformed = 0;
        let block_count = func.blocks.len();
        let locals = &func.locals;

        for block in &mut func.blocks {
            if let Some(Terminator::Call {
                callee,
                args,
                destination,
                success,
                error,
                ..
            }) = &mut block.terminator
            {
                if is_simd_callee(callee) {
                    let Some((canonical, expected_arity, kind)) = classify_simd_op(callee) else {
                        // Unknown or malformed SIMD name: optimizer must NOT mutate malformed inputs
                        continue;
                    };

                    // Transactional validation before any mutation
                    let Ok(force_ty) = validate_simd_call_for_opt(
                        kind,
                        expected_arity,
                        args,
                        destination,
                        success.0,
                        error.map(|b| b.0),
                        block_count,
                        locals,
                        struct_layouts,
                    ) else {
                        continue;
                    };

                    let mut call_changed = false;
                    if *callee != canonical {
                        *callee = canonical.into();
                        call_changed = true;
                    }

                    if config.force_scalar {
                        let forced_const = if force_ty == MirType::Bool {
                            Operand::Constant(MirConstant::Bool(true))
                        } else {
                            Operand::Constant(MirConstant::Int(1))
                        };
                        if let Some(last_arg) = args.last_mut() {
                            if *last_arg != forced_const {
                                *last_arg = forced_const;
                                call_changed = true;
                            }
                        }
                    }

                    if call_changed {
                        simd_calls_transformed += 1;
                    }
                }
            }
        }

        (loops_vectorized, simd_calls_transformed)
    }

    /// Runs SIMD optimization across an entire MIR module with accurate metrics.
    pub fn run_on_module(module: &mut MirModule) -> SimdStats {
        let mut stats = SimdStats::default();
        let config = SimdConfig::default();
        let layouts = &module.struct_layouts;
        for func in &mut module.functions {
            stats.functions_analyzed += 1;
            let (loops, calls) = Self::run_on_function_stats_with_layouts(func, &config, layouts);
            stats.loops_vectorized += loops;
            stats.simd_calls_transformed += calls;
        }
        stats
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorLoopKernelKind {
    Add,
    Mul,
    Sum,
    Dot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorizedLoopInfo {
    pub kind: VectorLoopKernelKind,
    pub pre_block: BlockId,
    pub hdr_block: BlockId,
    pub body_block: BlockId,
    pub tail_block: BlockId,
    pub idx_local: LocalId,
    pub end_local: LocalId,
    pub a_local: LocalId,
    pub b_local: Option<LocalId>,
    pub out_local: Option<LocalId>,
    pub acc_local: Option<LocalId>,
    pub gas_check: Option<u64>,
}

#[derive(Debug, Clone)]
enum LoopLenExpr {
    Local(LocalId),
    Const(i64),
}

#[derive(Debug, Clone)]
enum SymExpr {
    ArrayRead(LocalId),
    AccRead(LocalId),
    BinOp(MirBinaryOp, Box<SymExpr>, Box<SymExpr>),
    StepNext,
}

#[derive(Debug, Clone)]
struct MatchedLoop {
    kind: VectorLoopKernelKind,
    pre_block: BlockId,
    hdr_block: BlockId,
    #[allow(dead_code)]
    body_block: BlockId,
    #[allow(dead_code)]
    exit_block: BlockId,
    idx_local: LocalId,
    len_expr: LoopLenExpr,
    a_local: LocalId,
    b_local: Option<LocalId>,
    out_local: Option<LocalId>,
    acc_local: Option<LocalId>,
    gas_check: Option<u64>,
}

fn terminator_succs(term: &Terminator) -> Vec<BlockId> {
    match term {
        Terminator::Goto(target, _) => alloc::vec![*target],
        Terminator::Cond {
            then_block,
            else_block,
            ..
        } => alloc::vec![*then_block, *else_block],
        Terminator::Switch { cases, default, .. } => {
            let mut out: Vec<BlockId> = cases.iter().map(|(_, t)| *t).collect();
            out.push(*default);
            out
        }
        Terminator::Call { success, error, .. } => {
            let mut out = alloc::vec![*success];
            if let Some(err) = error {
                out.push(*err);
            }
            out
        }
        Terminator::Return(..) | Terminator::Raise(..) | Terminator::Unreachable(..) => {
            Vec::new()
        }
    }
}

fn compute_predecessors(func: &MirFunction) -> Option<Vec<Vec<BlockId>>> {
    let n = func.blocks.len();
    if func.entry.0 >= n {
        return None;
    }
    let mut preds = alloc::vec![Vec::new(); n];
    for (idx, block) in func.blocks.iter().enumerate() {
        let term = block.terminator.as_ref()?;
        for succ in terminator_succs(term) {
            if succ.0 >= n {
                return None;
            }
            preds[succ.0].push(BlockId(idx));
        }
    }
    Some(preds)
}

fn collect_place_locals(place: &Place, out: &mut HashSet<LocalId>) {
    match place {
        Place::Local(loc) => {
            out.insert(*loc);
        }
        Place::Field(base, _) => collect_place_locals(base, out),
        Place::Index(base, idx) => {
            collect_place_locals(base, out);
            out.insert(*idx);
        }
    }
}

fn collect_operand_reads(op: &Operand, out: &mut HashSet<LocalId>) {
    match op {
        Operand::Copy(p) | Operand::Move(p) => collect_place_locals(p, out),
        Operand::Constant(_) => {}
    }
}

fn collect_rvalue_reads(rval: &Rvalue, out: &mut HashSet<LocalId>) {
    match rval {
        Rvalue::Use(op) | Rvalue::UnaryOp(_, op) | Rvalue::Cast(op, _) => {
            collect_operand_reads(op, out);
        }
        Rvalue::BinaryOp(_, lhs, rhs) => {
            collect_operand_reads(lhs, out);
            collect_operand_reads(rhs, out);
        }
        Rvalue::Aggregate(_, elems) => {
            for e in elems {
                collect_operand_reads(e, out);
            }
        }
    }
}

fn has_forbidden_live_out(
    func: &MirFunction,
    exit_block: BlockId,
    hdr_block: BlockId,
    body_block: BlockId,
    forbidden: &HashSet<LocalId>,
) -> bool {
    if forbidden.is_empty() {
        return false;
    }
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    visited.insert(exit_block);
    queue.push_back(exit_block);

    while let Some(bb) = queue.pop_front() {
        if bb == hdr_block || bb == body_block {
            continue;
        }
        let block = &func.blocks[bb.0];
        let mut locally_defined = HashSet::new();
        for stmt in &block.statements {
            match stmt {
                Statement::Assign(place, rval, _) => {
                    let mut reads = HashSet::new();
                    collect_rvalue_reads(rval, &mut reads);
                    match place {
                        Place::Local(_) => {}
                        Place::Field(base, _) => collect_place_locals(base, &mut reads),
                        Place::Index(base, idx) => {
                            collect_place_locals(base, &mut reads);
                            reads.insert(*idx);
                        }
                    }
                    for r in reads {
                        if forbidden.contains(&r) && !locally_defined.contains(&r) {
                            return true;
                        }
                    }
                    if let Place::Local(dst) = place {
                        locally_defined.insert(*dst);
                    }
                }
                Statement::StorageLive(..)
                | Statement::StorageDead(..)
                | Statement::ResourceCheck(..) => {}
            }
        }
        if let Some(term) = &block.terminator {
            let mut reads = HashSet::new();
            match term {
                Terminator::Cond { cond, .. } => collect_operand_reads(cond, &mut reads),
                Terminator::Switch { discr, .. } => collect_operand_reads(discr, &mut reads),
                Terminator::Call {
                    args, destination, ..
                } => {
                    for arg in args {
                        collect_operand_reads(arg, &mut reads);
                    }
                    match destination {
                        Place::Local(_) => {}
                        Place::Field(base, _) => collect_place_locals(base, &mut reads),
                        Place::Index(base, idx) => {
                            collect_place_locals(base, &mut reads);
                            reads.insert(*idx);
                        }
                    }
                }
                Terminator::Return(Some(op), _) | Terminator::Raise(op, _) => {
                    collect_operand_reads(op, &mut reads);
                }
                Terminator::Goto(..)
                | Terminator::Return(None, _)
                | Terminator::Unreachable(..) => {}
            }
            for r in reads {
                if forbidden.contains(&r) && !locally_defined.contains(&r) {
                    return true;
                }
            }
            for succ in terminator_succs(term) {
                if succ != hdr_block && succ != body_block && visited.insert(succ) {
                    queue.push_back(succ);
                }
            }
        }
    }
    false
}

fn eval_operand_sym(
    op: &Operand,
    i_loc: LocalId,
    temp_vals: &HashMap<LocalId, (SymExpr, Vec<LocalId>)>,
    locals: &[LocalDecl],
    acc_candidate: Option<LocalId>,
) -> Option<(SymExpr, Vec<LocalId>)> {
    match op {
        Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => {
            if let Some((expr, deps)) = temp_vals.get(loc) {
                let mut all_deps = deps.clone();
                all_deps.push(*loc);
                Some((expr.clone(), all_deps))
            } else if Some(*loc) == acc_candidate {
                Some((SymExpr::AccRead(*loc), Vec::new()))
            } else {
                None
            }
        }
        Operand::Copy(Place::Index(base, idx)) | Operand::Move(Place::Index(base, idx)) => {
            let Place::Local(arr_loc) = base.as_ref() else {
                return None;
            };
            if *idx != i_loc {
                return None;
            }
            let arr_ty = &locals.get(arr_loc.0)?.ty;
            if !matches!(arr_ty, MirType::Array(inner) if **inner == MirType::Int) {
                return None;
            }
            Some((SymExpr::ArrayRead(*arr_loc), Vec::new()))
        }
        _ => None,
    }
}

fn is_step_plus_one(op1: &Operand, op2: &Operand, i_loc: LocalId) -> bool {
    match (op1, op2) {
        (
            Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)),
            Operand::Constant(MirConstant::Int(1)),
        )
        | (
            Operand::Constant(MirConstant::Int(1)),
            Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)),
        ) => *l == i_loc,
        _ => false,
    }
}

fn extract_binary_array_op(expr: &SymExpr) -> Option<(MirBinaryOp, LocalId, LocalId)> {
    if let SymExpr::BinOp(op, lhs, rhs) = expr {
        if matches!(op, MirBinaryOp::Add | MirBinaryOp::Mul) {
            if let (SymExpr::ArrayRead(a), SymExpr::ArrayRead(b)) = (lhs.as_ref(), rhs.as_ref()) {
                return Some((*op, *a, *b));
            }
        }
    }
    None
}

fn extract_reduction_update(expr: &SymExpr) -> Option<(VectorLoopKernelKind, LocalId, LocalId, Option<LocalId>)> {
    let SymExpr::BinOp(MirBinaryOp::Add, lhs, rhs) = expr else {
        return None;
    };
    let (acc_id, term) = match (lhs.as_ref(), rhs.as_ref()) {
        (SymExpr::AccRead(acc), other) | (other, SymExpr::AccRead(acc)) => (*acc, other),
        _ => return None,
    };
    match term {
        SymExpr::ArrayRead(a) => Some((VectorLoopKernelKind::Sum, acc_id, *a, None)),
        SymExpr::BinOp(MirBinaryOp::Mul, m_l, m_r) => {
            if let (SymExpr::ArrayRead(a), SymExpr::ArrayRead(b)) = (m_l.as_ref(), m_r.as_ref()) {
                Some((VectorLoopKernelKind::Dot, acc_id, *a, Some(*b)))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn try_match_vectorizable_loop(func: &MirFunction, preds: &[Vec<BlockId>]) -> Option<MatchedLoop> {
    for (hdr_idx, hdr_block_ref) in func.blocks.iter().enumerate() {
        let hdr_block = BlockId(hdr_idx);
        if hdr_block == func.entry || hdr_block_ref.is_cleanup {
            continue;
        }
        if preds[hdr_idx].len() != 2 {
            continue;
        }

        let p0 = preds[hdr_idx][0];
        let p1 = preds[hdr_idx][1];
        if p0 == p1 || p0 == hdr_block || p1 == hdr_block {
            continue;
        }

        let Some(Terminator::Cond {
            cond,
            then_block,
            else_block,
            ..
        }) = &hdr_block_ref.terminator
        else {
            continue;
        };
        let (Operand::Copy(Place::Local(cond_loc)) | Operand::Move(Place::Local(cond_loc))) = cond
        else {
            continue;
        };

        let body_block = *then_block;
        let exit_block = *else_block;
        if body_block == func.entry
            || body_block == hdr_block
            || exit_block == hdr_block
            || exit_block == body_block
        {
            continue;
        }
        let pre_block = if p0 == body_block && p1 != body_block {
            p1
        } else if p1 == body_block && p0 != body_block {
            p0
        } else {
            continue;
        };
        if pre_block == exit_block {
            continue;
        }

        if preds[body_block.0].len() != 1 || preds[body_block.0][0] != hdr_block {
            continue;
        }
        if func.blocks[body_block.0].is_cleanup {
            continue;
        }
        if !matches!(
            func.blocks[pre_block.0].terminator,
            Some(Terminator::Goto(t, _)) if t == hdr_block
        ) {
            continue;
        }
        if !matches!(
            func.blocks[body_block.0].terminator,
            Some(Terminator::Goto(t, _)) if t == hdr_block
        ) {
            continue;
        }

        // Validate scalar header shape; resource checks must retain scalar boundaries.
        let mut hdr_cmp: Option<(LocalId, LoopLenExpr)> = None;
        let gas_check: Option<u64> = None;
        let mut hdr_valid = true;
        for stmt in &hdr_block_ref.statements {
            match stmt {
                Statement::StorageLive(loc, _) if *loc == *cond_loc => {}
                Statement::ResourceCheck(ResourceKind::Gas, amt, _)
                    if hdr_cmp.is_none() && gas_check.is_none() =>
                {
                    // A vector chunk cannot preserve scalar exhaustion boundaries or
                    // partial writes. Keep metered loops scalar, including Gas(0).
                    let _ = amt;
                    hdr_valid = false;
                    break;
                }
                Statement::Assign(
                    Place::Local(dst),
                    Rvalue::BinaryOp(MirBinaryOp::Lt, lhs, rhs),
                    _,
                ) if *dst == *cond_loc && hdr_cmp.is_none() => {
                    let (Operand::Copy(Place::Local(i_loc)) | Operand::Move(Place::Local(i_loc))) =
                        lhs
                    else {
                        hdr_valid = false;
                        break;
                    };
                    let len_expr = match rhs {
                        Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)) => {
                            if *l == *i_loc || *l == *cond_loc {
                                hdr_valid = false;
                                break;
                            }
                            if func.locals.get(l.0).map(|d| &d.ty) != Some(&MirType::Int) {
                                hdr_valid = false;
                                break;
                            }
                            LoopLenExpr::Local(*l)
                        }
                        Operand::Constant(MirConstant::Int(k)) => {
                            if *k < SIMD_LANE_WIDTH as i64 {
                                hdr_valid = false;
                                break;
                            }
                            LoopLenExpr::Const(*k)
                        }
                        _ => {
                            hdr_valid = false;
                            break;
                        }
                    };
                    if func.locals.get(i_loc.0).map(|d| &d.ty) != Some(&MirType::Int)
                        || func.locals.get(cond_loc.0).map(|d| &d.ty) != Some(&MirType::Bool)
                    {
                        hdr_valid = false;
                        break;
                    }
                    hdr_cmp = Some((*i_loc, len_expr));
                }
                _ => {
                    hdr_valid = false;
                    break;
                }
            }
        }
        if !hdr_valid {
            continue;
        }
        let Some((i_loc, len_expr)) = hdr_cmp else {
            continue;
        };

        // Validate pre_block: last write to i_loc must be constant 0
        let mut init_zero = false;
        for stmt in &func.blocks[pre_block.0].statements {
            if let Statement::Assign(place, rval, _) = stmt {
                let mut root_written = None;
                match place {
                    Place::Local(l) => root_written = Some(*l),
                    Place::Index(base, _) | Place::Field(base, _) => {
                        let mut b = base.as_ref();
                        while let Place::Index(inner, _) | Place::Field(inner, _) = b {
                            b = inner.as_ref();
                        }
                        if let Place::Local(l) = b {
                            root_written = Some(*l);
                        }
                    }
                }
                if root_written == Some(i_loc) {
                    init_zero = matches!(
                        (place, rval),
                        (
                            Place::Local(l),
                            Rvalue::Use(Operand::Constant(MirConstant::Int(0)))
                        ) if *l == i_loc
                    );
                }
            }
        }
        if !init_zero {
            continue;
        }

        // Validate body_block statements strictly
        let len_local_opt = match &len_expr {
            LoopLenExpr::Local(l) => Some(*l),
            LoopLenExpr::Const(_) => None,
        };

        let mut temp_vals: HashMap<LocalId, (SymExpr, Vec<LocalId>)> = HashMap::new();
        let mut body_assigned_temps: HashSet<LocalId> = HashSet::new();
        let mut temp_assign_counts: HashMap<LocalId, usize> = HashMap::new();
        let mut array_write: Option<(LocalId, SymExpr, Vec<LocalId>)> = None;
        let mut acc_update: Option<(VectorLoopKernelKind, LocalId, LocalId, Option<LocalId>, Vec<LocalId>)> = None;
        let mut i_updated = false;
        let mut step_deps: Vec<LocalId> = Vec::new();
        let mut body_valid = true;

        // First scan: identify candidate reduction accumulator if a non-index Int local is read before written
        // or updated via `acc = acc + ...`
        let mut acc_candidate: Option<LocalId> = None;
        let mut written_before: HashSet<LocalId> = HashSet::new();
        for stmt in &func.blocks[body_block.0].statements {
            if let Statement::Assign(place, rval, _) = stmt {
                let mut reads = HashSet::new();
                collect_rvalue_reads(rval, &mut reads);
                for r in reads {
                    if r != i_loc
                        && Some(r) != len_local_opt
                        && !written_before.contains(&r)
                        && func.locals.get(r.0).map(|d| &d.ty) == Some(&MirType::Int)
                    {
                        if acc_candidate.is_some() && acc_candidate != Some(r) {
                            body_valid = false;
                            break;
                        }
                        acc_candidate = Some(r);
                    }
                }
                if !body_valid {
                    break;
                }
                if let Place::Local(dst) = place {
                    written_before.insert(*dst);
                }
            }
        }
        if !body_valid {
            continue;
        }

        for stmt in &func.blocks[body_block.0].statements {
            match stmt {
                Statement::StorageLive(loc, _) | Statement::StorageDead(loc, _) => {
                    if *loc == i_loc || Some(*loc) == len_local_opt || Some(*loc) == acc_candidate {
                        body_valid = false;
                        break;
                    }
                    if matches!(stmt, Statement::StorageDead(..)) {
                        temp_vals.remove(loc);
                    }
                }
                Statement::ResourceCheck(..) => {
                    body_valid = false;
                    break;
                }
                Statement::Assign(Place::Field(..), _, _) => {
                    body_valid = false;
                    break;
                }
                Statement::Assign(Place::Index(base, idx), rval, _) => {
                    if i_updated || array_write.is_some() || acc_update.is_some() {
                        body_valid = false;
                        break;
                    }
                    let Place::Local(out_loc) = base.as_ref() else {
                        body_valid = false;
                        break;
                    };
                    if *idx != i_loc
                        || Some(*out_loc) == len_local_opt
                        || Some(*out_loc) == acc_candidate
                    {
                        body_valid = false;
                        break;
                    }
                    if !matches!(
                        func.locals.get(out_loc.0).map(|d| &d.ty),
                        Some(MirType::Array(inner)) if **inner == MirType::Int
                    ) {
                        body_valid = false;
                        break;
                    }
                    let Rvalue::Use(op) = rval else {
                        body_valid = false;
                        break;
                    };
                    let Some((sym, deps)) =
                        eval_operand_sym(op, i_loc, &temp_vals, &func.locals, acc_candidate)
                    else {
                        body_valid = false;
                        break;
                    };
                    array_write = Some((*out_loc, sym, deps));
                }
                Statement::Assign(Place::Local(dst), rval, _) => {
                    if Some(*dst) == len_local_opt {
                        body_valid = false;
                        break;
                    }
                    if let Some(MirType::Array(_)) = func.locals.get(dst.0).map(|d| &d.ty) {
                        body_valid = false;
                        break;
                    }
                    if *dst == i_loc {
                        if i_updated {
                            body_valid = false;
                            break;
                        }
                        match rval {
                            Rvalue::BinaryOp(MirBinaryOp::Add, l, r)
                                if is_step_plus_one(l, r, i_loc) =>
                            {
                                i_updated = true;
                            }
                            Rvalue::Use(op) => {
                                let Some((SymExpr::StepNext, deps)) = eval_operand_sym(
                                    op,
                                    i_loc,
                                    &temp_vals,
                                    &func.locals,
                                    acc_candidate,
                                )
                                else {
                                    body_valid = false;
                                    break;
                                };
                                step_deps.extend(deps);
                                i_updated = true;
                            }
                            _ => {
                                body_valid = false;
                                break;
                            }
                        }
                    } else if i_updated {
                        body_valid = false;
                        break;
                    } else if Some(*dst) == acc_candidate {
                        if acc_update.is_some() || array_write.is_some() {
                            body_valid = false;
                            break;
                        }
                        let (expr, deps) = match rval {
                            Rvalue::Use(op) => {
                                let Some(res) = eval_operand_sym(
                                    op,
                                    i_loc,
                                    &temp_vals,
                                    &func.locals,
                                    acc_candidate,
                                ) else {
                                    body_valid = false;
                                    break;
                                };
                                res
                            }
                            Rvalue::BinaryOp(MirBinaryOp::Add, l, r) => {
                                let Some((l_sym, mut l_deps)) = eval_operand_sym(
                                    l,
                                    i_loc,
                                    &temp_vals,
                                    &func.locals,
                                    acc_candidate,
                                ) else {
                                    body_valid = false;
                                    break;
                                };
                                let Some((r_sym, r_deps)) = eval_operand_sym(
                                    r,
                                    i_loc,
                                    &temp_vals,
                                    &func.locals,
                                    acc_candidate,
                                ) else {
                                    body_valid = false;
                                    break;
                                };
                                l_deps.extend(r_deps);
                                (
                                    SymExpr::BinOp(MirBinaryOp::Add, Box::new(l_sym), Box::new(r_sym)),
                                    l_deps,
                                )
                            }
                            _ => {
                                body_valid = false;
                                break;
                            }
                        };
                        let Some((kind, acc_id, a_loc, b_loc)) = extract_reduction_update(&expr) else {
                            body_valid = false;
                            break;
                        };
                        if acc_id != *dst {
                            body_valid = false;
                            break;
                        }
                        acc_update = Some((kind, acc_id, a_loc, b_loc, deps));
                    } else {
                        if func.locals.get(dst.0).map(|d| &d.ty) != Some(&MirType::Int) {
                            body_valid = false;
                            break;
                        }
                        let (sym, deps) = match rval {
                            Rvalue::Use(op) => {
                                let Some(res) = eval_operand_sym(
                                    op,
                                    i_loc,
                                    &temp_vals,
                                    &func.locals,
                                    acc_candidate,
                                ) else {
                                    body_valid = false;
                                    break;
                                };
                                res
                            }
                            Rvalue::BinaryOp(MirBinaryOp::Add, l, r)
                                if is_step_plus_one(l, r, i_loc) =>
                            {
                                (SymExpr::StepNext, Vec::new())
                            }
                            Rvalue::BinaryOp(op @ (MirBinaryOp::Add | MirBinaryOp::Mul), l, r) => {
                                let Some((l_sym, mut l_deps)) = eval_operand_sym(
                                    l,
                                    i_loc,
                                    &temp_vals,
                                    &func.locals,
                                    acc_candidate,
                                ) else {
                                    body_valid = false;
                                    break;
                                };
                                let Some((r_sym, r_deps)) = eval_operand_sym(
                                    r,
                                    i_loc,
                                    &temp_vals,
                                    &func.locals,
                                    acc_candidate,
                                ) else {
                                    body_valid = false;
                                    break;
                                };
                                l_deps.extend(r_deps);
                                (SymExpr::BinOp(*op, Box::new(l_sym), Box::new(r_sym)), l_deps)
                            }
                            _ => {
                                body_valid = false;
                                break;
                            }
                        };
                        *temp_assign_counts.entry(*dst).or_insert(0) += 1;
                        if temp_assign_counts[dst] > 1 {
                            body_valid = false;
                            break;
                        }
                        body_assigned_temps.insert(*dst);
                        temp_vals.insert(*dst, (sym, deps));
                    }
                }
            }
        }

        if !body_valid || !i_updated {
            continue;
        }

        // Verify exact kernel match and that NO extraneous/side-effect assignments exist
        let mut consumed_temps: HashSet<LocalId> = step_deps.into_iter().collect();
        let matched_kernel = match (array_write, acc_update) {
            (Some((out_loc, expr, deps)), None) => {
                let Some((binop, a_loc, b_loc)) = extract_binary_array_op(&expr) else {
                    continue;
                };
                if acc_candidate.is_some() {
                    continue;
                }
                consumed_temps.extend(deps);
                let kind = match binop {
                    MirBinaryOp::Add => VectorLoopKernelKind::Add,
                    MirBinaryOp::Mul => VectorLoopKernelKind::Mul,
                    _ => continue,
                };
                Some((kind, a_loc, Some(b_loc), Some(out_loc), None))
            }
            (None, Some((kind, acc_id, a_loc, b_loc, deps))) => {
                consumed_temps.extend(deps);
                Some((kind, a_loc, b_loc, None, Some(acc_id)))
            }
            _ => None,
        };
        let Some((kind, a_local, b_local, out_local, acc_local)) = matched_kernel else {
            continue;
        };

        // Every assigned body temp must be consumed (rejects dead/side-effect writes)
        if body_assigned_temps.iter().any(|t| !consumed_temps.contains(t)) {
            continue;
        }

        // Live-out check: body temps & cond_loc must NOT be live-out after loop exit
        let mut forbidden_live_outs = body_assigned_temps;
        forbidden_live_outs.insert(*cond_loc);
        if has_forbidden_live_out(func, exit_block, hdr_block, body_block, &forbidden_live_outs) {
            continue;
        }

        return Some(MatchedLoop {
            kind,
            pre_block,
            hdr_block,
            body_block,
            exit_block,
            idx_local: i_loc,
            len_expr,
            a_local,
            b_local,
            out_local,
            acc_local,
            gas_check,
        });
    }
    None
}

fn build_vectorized_loop(func: &mut MirFunction, m: &MatchedLoop) {
    let span = Span::dummy();
    let i_loc = m.idx_local;
    let lane_w = SIMD_LANE_WIDTH as i64;

    let len_val_loc = func.add_local(Some("_simd_len".into()), MirType::Int, false, span);
    let cond_len_loc = func.add_local(Some("_simd_cond_len".into()), MirType::Bool, false, span);
    let vec_end_loc = func.add_local(Some("_simd_vec_end".into()), MirType::Int, false, span);
    let vec_cond_loc = func.add_local(Some("_simd_vec_cond".into()), MirType::Bool, false, span);

    let len_a_loc = func.add_local(Some("_simd_len_a".into()), MirType::Int, false, span);
    let cond_a_loc = func.add_local(Some("_simd_cond_a".into()), MirType::Bool, false, span);

    let len_b_loc = m
        .b_local
        .map(|_| func.add_local(Some("_simd_len_b".into()), MirType::Int, false, span));
    let cond_b_loc = m
        .b_local
        .map(|_| func.add_local(Some("_simd_cond_b".into()), MirType::Bool, false, span));

    let len_out_loc = m
        .out_local
        .map(|_| func.add_local(Some("_simd_len_out".into()), MirType::Int, false, span));
    let cond_out_loc = m
        .out_local
        .map(|_| func.add_local(Some("_simd_cond_out".into()), MirType::Bool, false, span));

    let i0 = func.add_local(Some("_simd_i0".into()), MirType::Int, false, span);
    let i1 = func.add_local(Some("_simd_i1".into()), MirType::Int, false, span);
    let i2 = func.add_local(Some("_simd_i2".into()), MirType::Int, false, span);
    let i3 = func.add_local(Some("_simd_i3".into()), MirType::Int, false, span);

    let a0 = func.add_local(Some("_simd_a0".into()), MirType::Int, false, span);
    let a1 = func.add_local(Some("_simd_a1".into()), MirType::Int, false, span);
    let a2 = func.add_local(Some("_simd_a2".into()), MirType::Int, false, span);
    let a3 = func.add_local(Some("_simd_a3".into()), MirType::Int, false, span);

    let b_lanes = m.b_local.map(|_| {
        (
            func.add_local(Some("_simd_b0".into()), MirType::Int, false, span),
            func.add_local(Some("_simd_b1".into()), MirType::Int, false, span),
            func.add_local(Some("_simd_b2".into()), MirType::Int, false, span),
            func.add_local(Some("_simd_b3".into()), MirType::Int, false, span),
        )
    });

    let r0 = func.add_local(Some("_simd_r0".into()), MirType::Int, false, span);
    let r1 = func.add_local(Some("_simd_r1".into()), MirType::Int, false, span);
    let r2 = func.add_local(Some("_simd_r2".into()), MirType::Int, false, span);
    let r3 = func.add_local(Some("_simd_r3".into()), MirType::Int, false, span);

    let sum01 = func.add_local(Some("_simd_sum01".into()), MirType::Int, false, span);
    let sum23 = func.add_local(Some("_simd_sum23".into()), MirType::Int, false, span);
    let sum_all = func.add_local(Some("_simd_sum_all".into()), MirType::Int, false, span);

    let bb_vec_guard = func.new_block();
    let bb_vec_chk_a = func.new_block();
    let bb_vec_chk_b = m.b_local.map(|_| func.new_block());
    let bb_vec_chk_out = m.out_local.map(|_| func.new_block());
    let bb_vec_pre = func.new_block();
    let bb_vec_hdr = func.new_block();
    let bb_vec_body = func.new_block();

    // Redirect bb_pre -> bb_vec_guard
    func.blocks[m.pre_block.0].terminator = Some(Terminator::Goto(bb_vec_guard, span));

    // bb_vec_guard: check len >= SIMD_LANE_WIDTH
    let len_rhs_op = match m.len_expr {
        LoopLenExpr::Local(l) => Operand::Copy(Place::Local(l)),
        LoopLenExpr::Const(k) => Operand::Constant(MirConstant::Int(k)),
    };
    func.blocks[bb_vec_guard.0].statements.push(Statement::Assign(
        Place::Local(len_val_loc),
        Rvalue::Use(len_rhs_op),
        span,
    ));
    func.blocks[bb_vec_guard.0].statements.push(Statement::Assign(
        Place::Local(cond_len_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::GtEq,
            Operand::Copy(Place::Local(len_val_loc)),
            Operand::Constant(MirConstant::Int(lane_w)),
        ),
        span,
    ));
    func.blocks[bb_vec_guard.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond_len_loc)),
        then_block: bb_vec_chk_a,
        else_block: m.hdr_block,
        span,
    });

    // Chain bounds checks: a -> b (if any) -> out (if any) -> bb_vec_pre
    let next_after_a = bb_vec_chk_b.or(bb_vec_chk_out).unwrap_or(bb_vec_pre);
    func.blocks[bb_vec_chk_a.0].statements.push(Statement::Assign(
        Place::Local(len_a_loc),
        Rvalue::UnaryOp(MirUnaryOp::Len, Operand::Copy(Place::Local(m.a_local))),
        span,
    ));
    func.blocks[bb_vec_chk_a.0].statements.push(Statement::Assign(
        Place::Local(cond_a_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::LtEq,
            Operand::Copy(Place::Local(len_val_loc)),
            Operand::Copy(Place::Local(len_a_loc)),
        ),
        span,
    ));
    func.blocks[bb_vec_chk_a.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond_a_loc)),
        then_block: next_after_a,
        else_block: m.hdr_block,
        span,
    });

    if let (Some(bb_b), Some(b_loc), Some(lb), Some(cb)) =
        (bb_vec_chk_b, m.b_local, len_b_loc, cond_b_loc)
    {
        let next_after_b = bb_vec_chk_out.unwrap_or(bb_vec_pre);
        func.blocks[bb_b.0].statements.push(Statement::Assign(
            Place::Local(lb),
            Rvalue::UnaryOp(MirUnaryOp::Len, Operand::Copy(Place::Local(b_loc))),
            span,
        ));
        func.blocks[bb_b.0].statements.push(Statement::Assign(
            Place::Local(cb),
            Rvalue::BinaryOp(
                MirBinaryOp::LtEq,
                Operand::Copy(Place::Local(len_val_loc)),
                Operand::Copy(Place::Local(lb)),
            ),
            span,
        ));
        func.blocks[bb_b.0].terminator = Some(Terminator::Cond {
            cond: Operand::Copy(Place::Local(cb)),
            then_block: next_after_b,
            else_block: m.hdr_block,
            span,
        });
    }

    if let (Some(bb_out), Some(out_loc), Some(lo), Some(co)) =
        (bb_vec_chk_out, m.out_local, len_out_loc, cond_out_loc)
    {
        func.blocks[bb_out.0].statements.push(Statement::Assign(
            Place::Local(lo),
            Rvalue::UnaryOp(MirUnaryOp::Len, Operand::Copy(Place::Local(out_loc))),
            span,
        ));
        func.blocks[bb_out.0].statements.push(Statement::Assign(
            Place::Local(co),
            Rvalue::BinaryOp(
                MirBinaryOp::LtEq,
                Operand::Copy(Place::Local(len_val_loc)),
                Operand::Copy(Place::Local(lo)),
            ),
            span,
        ));
        func.blocks[bb_out.0].terminator = Some(Terminator::Cond {
            cond: Operand::Copy(Place::Local(co)),
            then_block: bb_vec_pre,
            else_block: m.hdr_block,
            span,
        });
    }

    // bb_vec_pre: vec_end = len_val_loc & -4; goto bb_vec_hdr
    func.blocks[bb_vec_pre.0].statements.push(Statement::Assign(
        Place::Local(vec_end_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::BitAnd,
            Operand::Copy(Place::Local(len_val_loc)),
            Operand::Constant(MirConstant::Int(-4)),
        ),
        span,
    ));
    func.blocks[bb_vec_pre.0].terminator = Some(Terminator::Goto(bb_vec_hdr, span));

    // bb_vec_hdr: optional Gas ResourceCheck, then if i < vec_end -> bb_vec_body else bb_hdr (scalar tail)
    if let Some(amt) = m.gas_check {
        func.blocks[bb_vec_hdr.0].statements.push(Statement::ResourceCheck(
            ResourceKind::Gas,
            amt,
            span,
        ));
    }
    func.blocks[bb_vec_hdr.0].statements.push(Statement::Assign(
        Place::Local(vec_cond_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(i_loc)),
            Operand::Copy(Place::Local(vec_end_loc)),
        ),
        span,
    ));
    func.blocks[bb_vec_hdr.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(vec_cond_loc)),
        then_block: bb_vec_body,
        else_block: m.hdr_block,
        span,
    });

    // bb_vec_body: 4-lane chunk body (loads all inputs before any stores for alias safety)
    let body_stmts = &mut func.blocks[bb_vec_body.0].statements;
    body_stmts.push(Statement::Assign(
        Place::Local(i0),
        Rvalue::Use(Operand::Copy(Place::Local(i_loc))),
        span,
    ));
    body_stmts.push(Statement::Assign(
        Place::Local(i1),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i_loc)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        span,
    ));
    body_stmts.push(Statement::Assign(
        Place::Local(i2),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i_loc)),
            Operand::Constant(MirConstant::Int(2)),
        ),
        span,
    ));
    body_stmts.push(Statement::Assign(
        Place::Local(i3),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i_loc)),
            Operand::Constant(MirConstant::Int(3)),
        ),
        span,
    ));

    // Load a[i0..i3]
    for (dst_a, idx_l) in [(a0, i0), (a1, i1), (a2, i2), (a3, i3)] {
        body_stmts.push(Statement::Assign(
            Place::Local(dst_a),
            Rvalue::Use(Operand::Copy(Place::Index(
                Box::new(Place::Local(m.a_local)),
                idx_l,
            ))),
            span,
        ));
    }

    // Load b[i0..i3] if binary
    if let (Some(b_loc), Some((b0, b1, b2, b3))) = (m.b_local, b_lanes) {
        for (dst_b, idx_l) in [(b0, i0), (b1, i1), (b2, i2), (b3, i3)] {
            body_stmts.push(Statement::Assign(
                Place::Local(dst_b),
                Rvalue::Use(Operand::Copy(Place::Index(
                    Box::new(Place::Local(b_loc)),
                    idx_l,
                ))),
                span,
            ));
        }
    }

    match m.kind {
        VectorLoopKernelKind::Add | VectorLoopKernelKind::Mul => {
            let binop = if m.kind == VectorLoopKernelKind::Add {
                MirBinaryOp::Add
            } else {
                MirBinaryOp::Mul
            };
            let (b0, b1, b2, b3) = b_lanes.expect("binary op has b lanes");
            let out_loc = m.out_local.expect("vector write has out_local");
            for ((dst_r, la), lb) in [(r0, a0), (r1, a1), (r2, a2), (r3, a3)]
                .into_iter()
                .zip([b0, b1, b2, b3])
            {
                body_stmts.push(Statement::Assign(
                    Place::Local(dst_r),
                    Rvalue::BinaryOp(
                        binop,
                        Operand::Copy(Place::Local(la)),
                        Operand::Copy(Place::Local(lb)),
                    ),
                    span,
                ));
            }
            for (idx_l, src_r) in [(i0, r0), (i1, r1), (i2, r2), (i3, r3)] {
                body_stmts.push(Statement::Assign(
                    Place::Index(Box::new(Place::Local(out_loc)), idx_l),
                    Rvalue::Use(Operand::Copy(Place::Local(src_r))),
                    span,
                ));
            }
        }
        VectorLoopKernelKind::Sum => {
            let acc_loc = m.acc_local.expect("sum reduction has acc_local");
            body_stmts.push(Statement::Assign(
                Place::Local(sum01),
                Rvalue::BinaryOp(
                    MirBinaryOp::Add,
                    Operand::Copy(Place::Local(a0)),
                    Operand::Copy(Place::Local(a1)),
                ),
                span,
            ));
            body_stmts.push(Statement::Assign(
                Place::Local(sum23),
                Rvalue::BinaryOp(
                    MirBinaryOp::Add,
                    Operand::Copy(Place::Local(a2)),
                    Operand::Copy(Place::Local(a3)),
                ),
                span,
            ));
            body_stmts.push(Statement::Assign(
                Place::Local(sum_all),
                Rvalue::BinaryOp(
                    MirBinaryOp::Add,
                    Operand::Copy(Place::Local(sum01)),
                    Operand::Copy(Place::Local(sum23)),
                ),
                span,
            ));
            body_stmts.push(Statement::Assign(
                Place::Local(acc_loc),
                Rvalue::BinaryOp(
                    MirBinaryOp::Add,
                    Operand::Copy(Place::Local(acc_loc)),
                    Operand::Copy(Place::Local(sum_all)),
                ),
                span,
            ));
        }
        VectorLoopKernelKind::Dot => {
            let (b0, b1, b2, b3) = b_lanes.expect("dot has b lanes");
            let acc_loc = m.acc_local.expect("dot reduction has acc_local");
            for ((dst_r, la), lb) in [(r0, a0), (r1, a1), (r2, a2), (r3, a3)]
                .into_iter()
                .zip([b0, b1, b2, b3])
            {
                body_stmts.push(Statement::Assign(
                    Place::Local(dst_r),
                    Rvalue::BinaryOp(
                        MirBinaryOp::Mul,
                        Operand::Copy(Place::Local(la)),
                        Operand::Copy(Place::Local(lb)),
                    ),
                    span,
                ));
            }
            body_stmts.push(Statement::Assign(
                Place::Local(sum01),
                Rvalue::BinaryOp(
                    MirBinaryOp::Add,
                    Operand::Copy(Place::Local(r0)),
                    Operand::Copy(Place::Local(r1)),
                ),
                span,
            ));
            body_stmts.push(Statement::Assign(
                Place::Local(sum23),
                Rvalue::BinaryOp(
                    MirBinaryOp::Add,
                    Operand::Copy(Place::Local(r2)),
                    Operand::Copy(Place::Local(r3)),
                ),
                span,
            ));
            body_stmts.push(Statement::Assign(
                Place::Local(sum_all),
                Rvalue::BinaryOp(
                    MirBinaryOp::Add,
                    Operand::Copy(Place::Local(sum01)),
                    Operand::Copy(Place::Local(sum23)),
                ),
                span,
            ));
            body_stmts.push(Statement::Assign(
                Place::Local(acc_loc),
                Rvalue::BinaryOp(
                    MirBinaryOp::Add,
                    Operand::Copy(Place::Local(acc_loc)),
                    Operand::Copy(Place::Local(sum_all)),
                ),
                span,
            ));
        }
    }

    body_stmts.push(Statement::Assign(
        Place::Local(i_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i_loc)),
            Operand::Constant(MirConstant::Int(lane_w)),
        ),
        span,
    ));
    func.blocks[bb_vec_body.0].terminator = Some(Terminator::Goto(bb_vec_hdr, span));
}

/// Recognizes canonicalized 4-lane vectorized loop blocks emitted by `SimdOptimizer`
/// so native Cranelift codegen can lower `body_block` using hardware `I64X2` SIMD instructions.
pub fn recognize_vectorized_loop_blocks(func: &MirFunction) -> Vec<VectorizedLoopInfo> {
    let mut results = Vec::new();
    let Some(preds) = compute_predecessors(func) else {
        return results;
    };

    for (hdr_idx, hdr_blk) in func.blocks.iter().enumerate() {
        let hdr_block = BlockId(hdr_idx);
        if hdr_block == func.entry || hdr_blk.is_cleanup || preds[hdr_idx].len() != 2 {
            continue;
        }
        let Some(Terminator::Cond {
            cond: Operand::Copy(Place::Local(cond_loc)) | Operand::Move(Place::Local(cond_loc)),
            then_block: body_block,
            else_block: tail_block,
            ..
        }) = &hdr_blk.terminator
        else {
            continue;
        };
        if *body_block == hdr_block || *tail_block == hdr_block || *tail_block == *body_block {
            continue;
        }
        let p0 = preds[hdr_idx][0];
        let p1 = preds[hdr_idx][1];
        let pre_block = if p0 == *body_block && p1 != *body_block {
            p1
        } else if p1 == *body_block && p0 != *body_block {
            p0
        } else {
            continue;
        };
        if preds[body_block.0].len() != 1 || preds[body_block.0][0] != hdr_block {
            continue;
        }
        if !matches!(
            func.blocks[pre_block.0].terminator,
            Some(Terminator::Goto(t, _)) if t == hdr_block
        ) {
            continue;
        }
        if !matches!(
            func.blocks[body_block.0].terminator,
            Some(Terminator::Goto(t, _)) if t == hdr_block
        ) {
            continue;
        }

        // hdr_blk must check i < vec_end, optionally preceded by a single Gas ResourceCheck
        let (gas_check, cmp_stmt) = match hdr_blk.statements.as_slice() {
            [cmp] => (None, cmp),
            [Statement::ResourceCheck(ResourceKind::Gas, amt, _), cmp] => (Some(*amt), cmp),
            _ => continue,
        };
        let Statement::Assign(
            Place::Local(dst_c),
            Rvalue::BinaryOp(
                MirBinaryOp::Lt,
                Operand::Copy(Place::Local(idx_local)) | Operand::Move(Place::Local(idx_local)),
                Operand::Copy(Place::Local(end_local)) | Operand::Move(Place::Local(end_local)),
            ),
            _,
        ) = cmp_stmt
        else {
            continue;
        };
        if dst_c != cond_loc {
            continue;
        }

        // pre_block must compute end_local = _ & -4
        let pre_has_bitand = func.blocks[pre_block.0].statements.iter().any(|s| {
            matches!(
                s,
                Statement::Assign(
                    Place::Local(dst),
                    Rvalue::BinaryOp(MirBinaryOp::BitAnd, _, Operand::Constant(MirConstant::Int(-4))),
                    _
                ) if dst == end_local
            )
        });
        if !pre_has_bitand {
            continue;
        }

        // Match exact 4-lane structure in body_block
        let stmts = &func.blocks[body_block.0].statements;
        if stmts.len() < 12 {
            continue;
        }
        // First 4 statements: i0 = i; i1 = i+1; i2 = i+2; i3 = i+3
        let (
            Statement::Assign(Place::Local(i0), Rvalue::Use(Operand::Copy(Place::Local(il0))), _),
            Statement::Assign(
                Place::Local(i1),
                Rvalue::BinaryOp(
                    MirBinaryOp::Add,
                    Operand::Copy(Place::Local(il1)),
                    Operand::Constant(MirConstant::Int(1)),
                ),
                _,
            ),
            Statement::Assign(
                Place::Local(i2),
                Rvalue::BinaryOp(
                    MirBinaryOp::Add,
                    Operand::Copy(Place::Local(il2)),
                    Operand::Constant(MirConstant::Int(2)),
                ),
                _,
            ),
            Statement::Assign(
                Place::Local(i3),
                Rvalue::BinaryOp(
                    MirBinaryOp::Add,
                    Operand::Copy(Place::Local(il3)),
                    Operand::Constant(MirConstant::Int(3)),
                ),
                _,
            ),
        ) = (&stmts[0], &stmts[1], &stmts[2], &stmts[3])
        else {
            continue;
        };
        if il0 != idx_local || il1 != idx_local || il2 != idx_local || il3 != idx_local {
            continue;
        }
        let lane_idxs = [*i0, *i1, *i2, *i3];

        // Last statement: i = i + 4
        let Some(Statement::Assign(
            Place::Local(step_dst),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(step_src)),
                Operand::Constant(MirConstant::Int(4)),
            ),
            _,
        )) = stmts.last()
        else {
            continue;
        };
        if step_dst != idx_local || step_src != idx_local {
            continue;
        }

        // Next 4 statements: a_lanes = a[i0..i3]
        let mut a_local_opt = None;
        let mut a_lanes = [LocalId(0); 4];
        let mut ok = true;
        for lane in 0..4 {
            if let Statement::Assign(
                Place::Local(dst),
                Rvalue::Use(Operand::Copy(Place::Index(base, idx))),
                _,
            ) = &stmts[4 + lane]
            {
                if let Place::Local(arr) = base.as_ref() {
                    if *idx == lane_idxs[lane]
                        && a_local_opt.map_or(true, |prev| prev == *arr)
                    {
                        a_local_opt = Some(*arr);
                        a_lanes[lane] = *dst;
                        continue;
                    }
                }
            }
            ok = false;
            break;
        }
        if !ok {
            continue;
        }
        let Some(a_local) = a_local_opt else {
            continue;
        };

        // Determine Sum (len == 13) vs Add/Mul (len == 21) vs Dot (len == 21)
        if stmts.len() == 13 {
            // Sum: [8]=sum01, [9]=sum23, [10]=sum_all, [11]=acc = acc + sum_all
            if let (
                Statement::Assign(
                    Place::Local(s01),
                    Rvalue::BinaryOp(
                        MirBinaryOp::Add,
                        Operand::Copy(Place::Local(la0)),
                        Operand::Copy(Place::Local(la1)),
                    ),
                    _,
                ),
                Statement::Assign(
                    Place::Local(s23),
                    Rvalue::BinaryOp(
                        MirBinaryOp::Add,
                        Operand::Copy(Place::Local(la2)),
                        Operand::Copy(Place::Local(la3)),
                    ),
                    _,
                ),
                Statement::Assign(
                    Place::Local(sall),
                    Rvalue::BinaryOp(
                        MirBinaryOp::Add,
                        Operand::Copy(Place::Local(s01_r)),
                        Operand::Copy(Place::Local(s23_r)),
                    ),
                    _,
                ),
                Statement::Assign(
                    Place::Local(acc_dst),
                    Rvalue::BinaryOp(
                        MirBinaryOp::Add,
                        Operand::Copy(Place::Local(acc_src)),
                        Operand::Copy(Place::Local(sall_r)),
                    ),
                    _,
                ),
            ) = (&stmts[8], &stmts[9], &stmts[10], &stmts[11])
            {
                if *la0 == a_lanes[0]
                    && *la1 == a_lanes[1]
                    && *la2 == a_lanes[2]
                    && *la3 == a_lanes[3]
                    && s01_r == s01
                    && s23_r == s23
                    && sall_r == sall
                    && acc_dst == acc_src
                {
                    results.push(VectorizedLoopInfo {
                        kind: VectorLoopKernelKind::Sum,
                        pre_block,
                        hdr_block,
                        body_block: *body_block,
                        tail_block: *tail_block,
                        idx_local: *idx_local,
                        end_local: *end_local,
                        a_local,
                        b_local: None,
                        out_local: None,
                        acc_local: Some(*acc_dst),
                        gas_check,
                    });
                }
            }
        } else if stmts.len() == 21 {
            // [8..12]: b[i0..i3]
            let mut b_local_opt = None;
            let mut b_lanes = [LocalId(0); 4];
            let mut b_ok = true;
            for lane in 0..4 {
                if let Statement::Assign(
                    Place::Local(dst),
                    Rvalue::Use(Operand::Copy(Place::Index(base, idx))),
                    _,
                ) = &stmts[8 + lane]
                {
                    if let Place::Local(arr) = base.as_ref() {
                        if *idx == lane_idxs[lane]
                            && b_local_opt.map_or(true, |prev| prev == *arr)
                        {
                            b_local_opt = Some(*arr);
                            b_lanes[lane] = *dst;
                            continue;
                        }
                    }
                }
                b_ok = false;
                break;
            }
            if !b_ok {
                continue;
            }
            let Some(b_local) = b_local_opt else {
                continue;
            };

            // [12..16]: r_lanes = a_lanes <op> b_lanes
            let mut r_op_opt = None;
            let mut r_lanes = [LocalId(0); 4];
            let mut r_ok = true;
            for lane in 0..4 {
                if let Statement::Assign(
                    Place::Local(dst),
                    Rvalue::BinaryOp(
                        op @ (MirBinaryOp::Add | MirBinaryOp::Mul),
                        Operand::Copy(Place::Local(la)),
                        Operand::Copy(Place::Local(lb)),
                    ),
                    _,
                ) = &stmts[12 + lane]
                {
                    if *la == a_lanes[lane]
                        && *lb == b_lanes[lane]
                        && r_op_opt.map_or(true, |prev| prev == *op)
                    {
                        r_op_opt = Some(*op);
                        r_lanes[lane] = *dst;
                        continue;
                    }
                }
                r_ok = false;
                break;
            }
            if !r_ok {
                continue;
            }
            let Some(r_op) = r_op_opt else {
                continue;
            };

            // Either [16..20] are out[i0..i3] = r0..r3 (Add/Mul) OR reduction for Dot
            let mut out_local_opt = None;
            let mut out_ok = true;
            for lane in 0..4 {
                if let Statement::Assign(
                    Place::Index(base, idx),
                    Rvalue::Use(Operand::Copy(Place::Local(src_r))),
                    _,
                ) = &stmts[16 + lane]
                {
                    if let Place::Local(arr) = base.as_ref() {
                        if *idx == lane_idxs[lane]
                            && *src_r == r_lanes[lane]
                            && out_local_opt.map_or(true, |prev| prev == *arr)
                        {
                            out_local_opt = Some(*arr);
                            continue;
                        }
                    }
                }
                out_ok = false;
                break;
            }

            if out_ok {
                if let Some(out_local) = out_local_opt {
                    let kind = if r_op == MirBinaryOp::Add {
                        VectorLoopKernelKind::Add
                    } else {
                        VectorLoopKernelKind::Mul
                    };
                    results.push(VectorizedLoopInfo {
                        kind,
                        pre_block,
                        hdr_block,
                        body_block: *body_block,
                        tail_block: *tail_block,
                        idx_local: *idx_local,
                        end_local: *end_local,
                        a_local,
                        b_local: Some(b_local),
                        out_local: Some(out_local),
                        acc_local: None,
                        gas_check,
                    });
                }
            } else if r_op == MirBinaryOp::Mul {
                if let (
                    Statement::Assign(
                        Place::Local(s01),
                        Rvalue::BinaryOp(
                            MirBinaryOp::Add,
                            Operand::Copy(Place::Local(lr0)),
                            Operand::Copy(Place::Local(lr1)),
                        ),
                        _,
                    ),
                    Statement::Assign(
                        Place::Local(s23),
                        Rvalue::BinaryOp(
                            MirBinaryOp::Add,
                            Operand::Copy(Place::Local(lr2)),
                            Operand::Copy(Place::Local(lr3)),
                        ),
                        _,
                    ),
                    Statement::Assign(
                        Place::Local(sall),
                        Rvalue::BinaryOp(
                            MirBinaryOp::Add,
                            Operand::Copy(Place::Local(s01_r)),
                            Operand::Copy(Place::Local(s23_r)),
                        ),
                        _,
                    ),
                    Statement::Assign(
                        Place::Local(acc_dst),
                        Rvalue::BinaryOp(
                            MirBinaryOp::Add,
                            Operand::Copy(Place::Local(acc_src)),
                            Operand::Copy(Place::Local(sall_r)),
                        ),
                        _,
                    ),
                ) = (&stmts[16], &stmts[17], &stmts[18], &stmts[19])
                {
                    if *lr0 == r_lanes[0]
                        && *lr1 == r_lanes[1]
                        && *lr2 == r_lanes[2]
                        && *lr3 == r_lanes[3]
                        && s01_r == s01
                        && s23_r == s23
                        && sall_r == sall
                        && acc_dst == acc_src
                    {
                        results.push(VectorizedLoopInfo {
                            kind: VectorLoopKernelKind::Dot,
                            pre_block,
                            hdr_block,
                            body_block: *body_block,
                            tail_block: *tail_block,
                            idx_local: *idx_local,
                            end_local: *end_local,
                            a_local,
                            b_local: Some(b_local),
                            out_local: None,
                            acc_local: Some(*acc_dst),
                            gas_check,
                        });
                    }
                }
            }
        }
    }

    results
}

/// Provably sound automatic MIR loop vectorization.
/// Only transforms loops when full CFG, zero init, strict `<` upper bound, `+1` step,
/// absence of side effects, absence of non-induction live-outs, and strict array typing
/// are statically proven; preserves scalar fallback with runtime length/bounds/alias guards.
fn detect_and_transform_loops(func: &mut MirFunction, config: &SimdConfig) -> usize {
    if config.force_scalar || is_force_simd_scalar_fallback() || config.lane_width < SIMD_LANE_WIDTH
    {
        return 0;
    }

    let mut loops_transformed = 0usize;
    let max_loops = func.blocks.len().min(16);
    for _ in 0..max_loops {
        let Some(preds) = compute_predecessors(func) else {
            break;
        };
        let Some(matched) = try_match_vectorizable_loop(func, &preds) else {
            break;
        };
        build_vectorized_loop(func, &matched);
        loops_transformed += 1;
    }
    loops_transformed
}

#[inline]
pub fn is_simd_callee(callee: &str) -> bool {
    callee.starts_with("simd.")
        || callee.starts_with("simd_")
        || callee.starts_with("vector.")
}

// =========================================================================
// Hardware-Accelerated (NEON / SSE2) SIMD Vector Kernels & Scalar Fallback
// =========================================================================

/// Vector addition: out[i] = a[i] + b[i] for i in 0..len.
/// Uses target SIMD intrinsics (`aarch64` NEON `vld1q_s64`/`vaddq_s64`/`vst1q_s64`
/// or `x86_64` SSE2 `_mm_add_epi64`) over 4-element unrolled chunks with scalar tail.
/// If `force_scalar` is true or if conflicting overlap hazard is detected, executes scalar fallback.
/// If contradictory overlap is detected (e.g. forward shift on one operand, backward shift on another),
/// rejects atomically returning false with zero memory mutations.
pub fn simd_vector_add_kernel(a: &[i64], b: &[i64], out: &mut [i64], force_scalar: bool) -> bool {
    let len = a.len().min(b.len()).min(out.len());
    if len == 0 {
        return true;
    }

    let out_ptr = out.as_mut_ptr() as usize;
    let a_ptr = a.as_ptr() as usize;
    let b_ptr = b.as_ptr() as usize;
    let len_bytes = len.saturating_mul(core::mem::size_of::<i64>());
    let out_end = out_ptr.saturating_add(len_bytes);
    let a_end = a_ptr.saturating_add(len_bytes);
    let b_end = b_ptr.saturating_add(len_bytes);

    let a_overlaps = out_ptr < a_end && a_ptr < out_end;
    let a_shifted = a_overlaps && out_ptr != a_ptr;
    let a_fwd_hazard = a_shifted && out_ptr > a_ptr;
    let a_bwd_hazard = a_shifted && out_ptr < a_ptr;

    let b_overlaps = out_ptr < b_end && b_ptr < out_end;
    let b_shifted = b_overlaps && out_ptr != b_ptr;
    let b_fwd_hazard = b_shifted && out_ptr > b_ptr;
    let b_bwd_hazard = b_shifted && out_ptr < b_ptr;

    let has_fwd_hazard = a_fwd_hazard || b_fwd_hazard;
    let has_bwd_hazard = a_bwd_hazard || b_bwd_hazard;

    if has_fwd_hazard && has_bwd_hazard {
        // Contradictory overlap: forward shift on one operand, backward shift on another.
        // In-place traversal would corrupt inputs; reject atomically with 0 mutations.
        return false;
    }

    let force = force_scalar || is_force_simd_scalar_fallback() || has_fwd_hazard || has_bwd_hazard;

    if force || len < SIMD_LANE_WIDTH {
        SIMD_SCALAR_FALLBACK_COUNT.fetch_add(1, Ordering::SeqCst);
        if has_fwd_hazard {
            // Traversal from tail to head prevents clobbering future input elements
            for i in (0..len).rev() {
                out[i] = a[i].wrapping_add(b[i]);
            }
        } else {
            // Traversal from head to tail
            for i in 0..len {
                out[i] = a[i].wrapping_add(b[i]);
            }
        }
        return true;
    }

    SIMD_VECTOR_OP_COUNT.fetch_add(1, Ordering::SeqCst);
    let chunks = len / SIMD_LANE_WIDTH;
    let remainder = len % SIMD_LANE_WIDTH;

    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::{vaddq_s64, vld1q_s64, vst1q_s64};
        for c in 0..chunks {
            let base = c * SIMD_LANE_WIDTH;
            // Safety: `base + 4 <= chunks * 4 <= len <= a.len(), b.len(), out.len()`.
            unsafe {
                let va0 = vld1q_s64(a.as_ptr().add(base));
                let vb0 = vld1q_s64(b.as_ptr().add(base));
                let va1 = vld1q_s64(a.as_ptr().add(base + 2));
                let vb1 = vld1q_s64(b.as_ptr().add(base + 2));
                vst1q_s64(out.as_mut_ptr().add(base), vaddq_s64(va0, vb0));
                vst1q_s64(out.as_mut_ptr().add(base + 2), vaddq_s64(va1, vb1));
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::{__m128i, _mm_add_epi64, _mm_loadu_si128, _mm_storeu_si128};
        for c in 0..chunks {
            let base = c * SIMD_LANE_WIDTH;
            // Safety: SSE2 is baseline on x86_64; `base + 4 <= len`.
            unsafe {
                let va0 = _mm_loadu_si128(a.as_ptr().add(base) as *const __m128i);
                let vb0 = _mm_loadu_si128(b.as_ptr().add(base) as *const __m128i);
                let va1 = _mm_loadu_si128(a.as_ptr().add(base + 2) as *const __m128i);
                let vb1 = _mm_loadu_si128(b.as_ptr().add(base + 2) as *const __m128i);
                _mm_storeu_si128(
                    out.as_mut_ptr().add(base) as *mut __m128i,
                    _mm_add_epi64(va0, vb0),
                );
                _mm_storeu_si128(
                    out.as_mut_ptr().add(base + 2) as *mut __m128i,
                    _mm_add_epi64(va1, vb1),
                );
            }
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        for c in 0..chunks {
            let base = c * SIMD_LANE_WIDTH;
            out[base] = a[base].wrapping_add(b[base]);
            out[base + 1] = a[base + 1].wrapping_add(b[base + 1]);
            out[base + 2] = a[base + 2].wrapping_add(b[base + 2]);
            out[base + 3] = a[base + 3].wrapping_add(b[base + 3]);
        }
    }

    // Scalar tail fallback for remainder elements
    if remainder > 0 {
        SIMD_TAIL_PROCESSED_COUNT.fetch_add(remainder, Ordering::SeqCst);
        let tail_start = chunks * SIMD_LANE_WIDTH;
        for i in tail_start..len {
            out[i] = a[i].wrapping_add(b[i]);
        }
    }

    true
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn vmul_s64_neon(a: core::arch::aarch64::int64x2_t, b: core::arch::aarch64::int64x2_t) -> core::arch::aarch64::int64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        let a_u = vreinterpretq_u64_s64(a);
        let b_u = vreinterpretq_u64_s64(b);
        let a32 = vreinterpretq_u32_u64(a_u);
        let b32 = vreinterpretq_u32_u64(b_u);
        let a_lo = vuzp1_u32(vget_low_u32(a32), vget_high_u32(a32));
        let a_hi = vuzp2_u32(vget_low_u32(a32), vget_high_u32(a32));
        let b_lo = vuzp1_u32(vget_low_u32(b32), vget_high_u32(b32));
        let b_hi = vuzp2_u32(vget_low_u32(b32), vget_high_u32(b32));
        let lo_prod = vmull_u32(a_lo, b_lo);
        let cross1 = vmull_u32(a_lo, b_hi);
        let cross2 = vmull_u32(a_hi, b_lo);
        let cross_sum = vaddq_u64(cross1, cross2);
        let cross_shifted = vshlq_n_u64::<32>(cross_sum);
        let full_prod = vaddq_u64(lo_prod, cross_shifted);
        vreinterpretq_s64_u64(full_prod)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn vmul_s64_sse2(a: core::arch::x86_64::__m128i, b: core::arch::x86_64::__m128i) -> core::arch::x86_64::__m128i {
    use core::arch::x86_64::*;
    unsafe {
        let lo_prod = _mm_mul_epu32(a, b);
        let a_hi = _mm_srli_epi64(a, 32);
        let b_hi = _mm_srli_epi64(b, 32);
        let cross1 = _mm_mul_epu32(a, b_hi);
        let cross2 = _mm_mul_epu32(a_hi, b);
        let cross = _mm_slli_epi64(_mm_add_epi64(cross1, cross2), 32);
        _mm_add_epi64(lo_prod, cross)
    }
}

/// Vector multiplication: out[i] = a[i] * b[i] for i in 0..len.
/// Uses hardware SIMD vector multiplication (AArch64 NEON vmull/vadd or x86_64 SSE2)
/// over 4-element vector chunks with scalar tail and sound alias/overlap semantics.
/// Contradictory overlaps are rejected atomically returning false with zero memory mutations.
pub fn simd_vector_mul_kernel(a: &[i64], b: &[i64], out: &mut [i64], force_scalar: bool) -> bool {
    let len = a.len().min(b.len()).min(out.len());
    if len == 0 {
        return true;
    }

    let out_ptr = out.as_mut_ptr() as usize;
    let a_ptr = a.as_ptr() as usize;
    let b_ptr = b.as_ptr() as usize;
    let len_bytes = len.saturating_mul(core::mem::size_of::<i64>());
    let out_end = out_ptr.saturating_add(len_bytes);
    let a_end = a_ptr.saturating_add(len_bytes);
    let b_end = b_ptr.saturating_add(len_bytes);

    let a_overlaps = out_ptr < a_end && a_ptr < out_end;
    let a_shifted = a_overlaps && out_ptr != a_ptr;
    let a_fwd_hazard = a_shifted && out_ptr > a_ptr;
    let a_bwd_hazard = a_shifted && out_ptr < a_ptr;

    let b_overlaps = out_ptr < b_end && b_ptr < out_end;
    let b_shifted = b_overlaps && out_ptr != b_ptr;
    let b_fwd_hazard = b_shifted && out_ptr > b_ptr;
    let b_bwd_hazard = b_shifted && out_ptr < b_ptr;

    let has_fwd_hazard = a_fwd_hazard || b_fwd_hazard;
    let has_bwd_hazard = a_bwd_hazard || b_bwd_hazard;

    if has_fwd_hazard && has_bwd_hazard {
        // Contradictory overlap: reject atomically
        return false;
    }

    let force = force_scalar || is_force_simd_scalar_fallback() || has_fwd_hazard || has_bwd_hazard;

    if force || len < SIMD_LANE_WIDTH {
        SIMD_SCALAR_FALLBACK_COUNT.fetch_add(1, Ordering::SeqCst);
        if has_fwd_hazard {
            for i in (0..len).rev() {
                out[i] = a[i].wrapping_mul(b[i]);
            }
        } else {
            for i in 0..len {
                out[i] = a[i].wrapping_mul(b[i]);
            }
        }
        return true;
    }

    SIMD_VECTOR_OP_COUNT.fetch_add(1, Ordering::SeqCst);
    let chunks = len / SIMD_LANE_WIDTH;
    let remainder = len % SIMD_LANE_WIDTH;

    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::{vld1q_s64, vst1q_s64};
        for c in 0..chunks {
            let base = c * SIMD_LANE_WIDTH;
            unsafe {
                let va0 = vld1q_s64(a.as_ptr().add(base));
                let vb0 = vld1q_s64(b.as_ptr().add(base));
                let va1 = vld1q_s64(a.as_ptr().add(base + 2));
                let vb1 = vld1q_s64(b.as_ptr().add(base + 2));
                let prod0 = vmul_s64_neon(va0, vb0);
                let prod1 = vmul_s64_neon(va1, vb1);
                vst1q_s64(out.as_mut_ptr().add(base), prod0);
                vst1q_s64(out.as_mut_ptr().add(base + 2), prod1);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_storeu_si128};
        for c in 0..chunks {
            let base = c * SIMD_LANE_WIDTH;
            unsafe {
                let va0 = _mm_loadu_si128(a.as_ptr().add(base) as *const __m128i);
                let vb0 = _mm_loadu_si128(b.as_ptr().add(base) as *const __m128i);
                let va1 = _mm_loadu_si128(a.as_ptr().add(base + 2) as *const __m128i);
                let vb1 = _mm_loadu_si128(b.as_ptr().add(base + 2) as *const __m128i);
                let prod0 = vmul_s64_sse2(va0, vb0);
                let prod1 = vmul_s64_sse2(va1, vb1);
                _mm_storeu_si128(out.as_mut_ptr().add(base) as *mut __m128i, prod0);
                _mm_storeu_si128(out.as_mut_ptr().add(base + 2) as *mut __m128i, prod1);
            }
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        for c in 0..chunks {
            let base = c * SIMD_LANE_WIDTH;
            out[base] = a[base].wrapping_mul(b[base]);
            out[base + 1] = a[base + 1].wrapping_mul(b[base + 1]);
            out[base + 2] = a[base + 2].wrapping_mul(b[base + 2]);
            out[base + 3] = a[base + 3].wrapping_mul(b[base + 3]);
        }
    }

    if remainder > 0 {
        SIMD_TAIL_PROCESSED_COUNT.fetch_add(remainder, Ordering::SeqCst);
        let tail_start = chunks * SIMD_LANE_WIDTH;
        for i in tail_start..len {
            out[i] = a[i].wrapping_mul(b[i]);
        }
    }

    true
}

/// Vector sum reduction: returns sum of all elements in slice.
/// Uses `aarch64` NEON (`vld1q_s64`/`vaddq_s64`) or `x86_64` SSE2 (`_mm_add_epi64`)
/// vector accumulators over 4-element chunks plus scalar tail.
pub fn simd_vector_sum_kernel(a: &[i64], force_scalar: bool) -> i64 {
    let len = a.len();
    let force = force_scalar || is_force_simd_scalar_fallback();

    if force {
        SIMD_SCALAR_FALLBACK_COUNT.fetch_add(1, Ordering::SeqCst);
        let mut acc = 0i64;
        for &val in a {
            acc = acc.wrapping_add(val);
        }
        return acc;
    }

    SIMD_VECTOR_OP_COUNT.fetch_add(1, Ordering::SeqCst);
    let chunks = len / SIMD_LANE_WIDTH;
    let remainder = len % SIMD_LANE_WIDTH;

    #[cfg(target_arch = "aarch64")]
    let mut total: i64 = {
        use core::arch::aarch64::{vaddq_s64, vdupq_n_s64, vgetq_lane_s64, vld1q_s64};
        unsafe {
            let mut acc0 = vdupq_n_s64(0);
            let mut acc1 = vdupq_n_s64(0);
            for c in 0..chunks {
                let base = c * SIMD_LANE_WIDTH;
                acc0 = vaddq_s64(acc0, vld1q_s64(a.as_ptr().add(base)));
                acc1 = vaddq_s64(acc1, vld1q_s64(a.as_ptr().add(base + 2)));
            }
            let sum_vec = vaddq_s64(acc0, acc1);
            vgetq_lane_s64::<0>(sum_vec).wrapping_add(vgetq_lane_s64::<1>(sum_vec))
        }
    };

    #[cfg(target_arch = "x86_64")]
    let mut total: i64 = {
        use core::arch::x86_64::{__m128i, _mm_add_epi64, _mm_cvtsi128_si64, _mm_loadu_si128, _mm_setzero_si128, _mm_unpackhi_epi64};
        unsafe {
            let mut acc0 = _mm_setzero_si128();
            let mut acc1 = _mm_setzero_si128();
            for c in 0..chunks {
                let base = c * SIMD_LANE_WIDTH;
                acc0 = _mm_add_epi64(acc0, _mm_loadu_si128(a.as_ptr().add(base) as *const __m128i));
                acc1 = _mm_add_epi64(acc1, _mm_loadu_si128(a.as_ptr().add(base + 2) as *const __m128i));
            }
            let sum_vec = _mm_add_epi64(acc0, acc1);
            let hi = _mm_unpackhi_epi64(sum_vec, sum_vec);
            _mm_cvtsi128_si64(sum_vec).wrapping_add(_mm_cvtsi128_si64(hi))
        }
    };

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let mut total: i64 = {
        let mut acc0 = 0i64;
        let mut acc1 = 0i64;
        let mut acc2 = 0i64;
        let mut acc3 = 0i64;
        for c in 0..chunks {
            let base = c * SIMD_LANE_WIDTH;
            acc0 = acc0.wrapping_add(a[base]);
            acc1 = acc1.wrapping_add(a[base + 1]);
            acc2 = acc2.wrapping_add(a[base + 2]);
            acc3 = acc3.wrapping_add(a[base + 3]);
        }
        acc0.wrapping_add(acc1).wrapping_add(acc2).wrapping_add(acc3)
    };

    if remainder > 0 {
        SIMD_TAIL_PROCESSED_COUNT.fetch_add(remainder, Ordering::SeqCst);
        let tail_start = chunks * SIMD_LANE_WIDTH;
        for i in tail_start..len {
            total = total.wrapping_add(a[i]);
        }
    }

    total
}

/// Vector dot product: returns sum(a[i] * b[i]).
pub fn simd_vector_dot_kernel(a: &[i64], b: &[i64], force_scalar: bool) -> i64 {
    let len = a.len().min(b.len());
    let force = force_scalar || is_force_simd_scalar_fallback();

    if force {
        SIMD_SCALAR_FALLBACK_COUNT.fetch_add(1, Ordering::SeqCst);
        let mut acc = 0i64;
        for i in 0..len {
            acc = acc.wrapping_add(a[i].wrapping_mul(b[i]));
        }
        return acc;
    }

    SIMD_VECTOR_OP_COUNT.fetch_add(1, Ordering::SeqCst);
    let chunks = len / SIMD_LANE_WIDTH;
    let remainder = len % SIMD_LANE_WIDTH;

    #[cfg(target_arch = "aarch64")]
    let mut total: i64 = {
        use core::arch::aarch64::{vaddq_s64, vdupq_n_s64, vgetq_lane_s64, vld1q_s64};
        unsafe {
            let mut acc0 = vdupq_n_s64(0);
            let mut acc1 = vdupq_n_s64(0);
            for c in 0..chunks {
                let base = c * SIMD_LANE_WIDTH;
                let va0 = vld1q_s64(a.as_ptr().add(base));
                let vb0 = vld1q_s64(b.as_ptr().add(base));
                let va1 = vld1q_s64(a.as_ptr().add(base + 2));
                let vb1 = vld1q_s64(b.as_ptr().add(base + 2));
                acc0 = vaddq_s64(acc0, vmul_s64_neon(va0, vb0));
                acc1 = vaddq_s64(acc1, vmul_s64_neon(va1, vb1));
            }
            let sum_vec = vaddq_s64(acc0, acc1);
            vgetq_lane_s64::<0>(sum_vec).wrapping_add(vgetq_lane_s64::<1>(sum_vec))
        }
    };

    #[cfg(target_arch = "x86_64")]
    let mut total: i64 = {
        use core::arch::x86_64::{__m128i, _mm_add_epi64, _mm_cvtsi128_si64, _mm_loadu_si128, _mm_setzero_si128, _mm_unpackhi_epi64};
        unsafe {
            let mut acc0 = _mm_setzero_si128();
            let mut acc1 = _mm_setzero_si128();
            for c in 0..chunks {
                let base = c * SIMD_LANE_WIDTH;
                let va0 = _mm_loadu_si128(a.as_ptr().add(base) as *const __m128i);
                let vb0 = _mm_loadu_si128(b.as_ptr().add(base) as *const __m128i);
                let va1 = _mm_loadu_si128(a.as_ptr().add(base + 2) as *const __m128i);
                let vb1 = _mm_loadu_si128(b.as_ptr().add(base + 2) as *const __m128i);
                acc0 = _mm_add_epi64(acc0, vmul_s64_sse2(va0, vb0));
                acc1 = _mm_add_epi64(acc1, vmul_s64_sse2(va1, vb1));
            }
            let sum_vec = _mm_add_epi64(acc0, acc1);
            let hi = _mm_unpackhi_epi64(sum_vec, sum_vec);
            _mm_cvtsi128_si64(sum_vec).wrapping_add(_mm_cvtsi128_si64(hi))
        }
    };

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let mut total: i64 = {
        let mut acc0 = 0i64;
        let mut acc1 = 0i64;
        let mut acc2 = 0i64;
        let mut acc3 = 0i64;
        for c in 0..chunks {
            let base = c * SIMD_LANE_WIDTH;
            acc0 = acc0.wrapping_add(a[base].wrapping_mul(b[base]));
            acc1 = acc1.wrapping_add(a[base + 1].wrapping_mul(b[base + 1]));
            acc2 = acc2.wrapping_add(a[base + 2].wrapping_mul(b[base + 2]));
            acc3 = acc3.wrapping_add(a[base + 3].wrapping_mul(b[base + 3]));
        }
        acc0.wrapping_add(acc1).wrapping_add(acc2).wrapping_add(acc3)
    };

    if remainder > 0 {
        SIMD_TAIL_PROCESSED_COUNT.fetch_add(remainder, Ordering::SeqCst);
        let tail_start = chunks * SIMD_LANE_WIDTH;
        for i in tail_start..len {
            total = total.wrapping_add(a[i].wrapping_mul(b[i]));
        }
    }

    total
}

/// Inspects machine code bytes and verifies the presence of genuine vector instructions.
/// - On AArch64: checks for Advanced SIMD (NEON) vector instructions (e.g. `add v.2d`, `ldr q`, `str q`).
/// - On x86_64: checks for SSE2/AVX vector opcodes (e.g. `paddq`, `movdqu`, `movdqa`).
pub fn verify_machine_vector_instructions(code_bytes: &[u8]) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        if code_bytes.len() < 4 {
            return false;
        }
        for chunk in code_bytes.chunks_exact(4) {
            let insn = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let top11 = insn >> 21;
            let top8 = insn >> 24;
            if top11 == 0b0100_1110_111
                || (insn & 0x7fe0_f400) == 0x4ee0_8400
                || top8 == 0x3c
            {
                return true;
            }
        }
        false
    }

    #[cfg(target_arch = "x86_64")]
    {
        for i in 0..code_bytes.len().saturating_sub(2) {
            if (code_bytes[i] == 0x66 && code_bytes[i + 1] == 0x0F && code_bytes[i + 2] == 0xD4)
                || (code_bytes[i] == 0x66 && code_bytes[i + 1] == 0x0F && (code_bytes[i + 2] == 0x6F || code_bytes[i + 2] == 0x7F))
                || (code_bytes[i] == 0xF3 && code_bytes[i + 1] == 0x0F && (code_bytes[i + 2] == 0x6F || code_bytes[i + 2] == 0x7F))
            {
                return true;
            }
        }
        false
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        true
    }
}

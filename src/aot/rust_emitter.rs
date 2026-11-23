//! Structural MIR -> Rust AOT Code Generator (Tiers 1-4, `NG-W12-4A` + `NG-W12-4B`).
//!
//! Translates a refined, escape-analyzed `MirModule` into idiomatic, `#![no_std]`-ready,
//! `unsafe`-free, panic-free Rust source code backed by `lust_aot_rt`:
//! - **Tier 1 (Scalars):** `i64` (wrapping `+`/`-`/`*`, direct const `/` and `%` when
//!   `C != 0 && C != -1`, checked `lust_aot_rt::{div_int, rem_int, floordiv_int, pow_int}`
//!   otherwise), `f64`, `bool`, `()`.
//! - **Tier 2 (C-flat Structs):** `#[repr(C)] #[derive(Clone, Copy, Debug, Default, PartialEq)]`
//!   value structs on the stack / in CPU registers (zero heap allocation).
//! - **Tier 3 (Recursive Structs & Typed Arrays):**
//!   - Recursive structs (`Option<Box<LustStruct_Node<'a>, &'a lust_aot_rt::LustChunkArena<'a>>>`)
//!     with zero-cost borrowed `&LustStruct_Node<'a>` read-only parameters and `if let Some(...)`
//!     borrowing (`IfOptionSome`).
//!   - Local typed arrays `alloc::vec::Vec<T, &'a lust_aot_rt::LustBudgetAlloc>` (`bool` = 1 B,
//!     `i64` = 8 B, `f64` = 8 B) with read-only `&[T]` parameters and `StructuredNode::IfArrayGetOk`
//!     lowered to `if let Some(&v) = usize::try_from(idx).ok().and_then(|__i| arr.get(__i))`.
//! - **Tier 4 (Aliasing, Containers, Enums, Closures, `AotValue`, `defer`/`errdefer`):**
//!   - Local aliasing via `Rc<RefCell<...>>` / `Rc<Cell<...>>` (Hazard 12: zero `E0499`/`E0502`,
//!     panic-free `lust_aot_rt::{ref_borrow, ref_borrow_mut}`).
//!   - Ordered maps `lust_aot_rt::AotMap<K, V>` (`Map<K, V>`) with budget enforcement.
//!   - Native Rust user enums `pub enum LustEnum_<Name>` from `module.enum_defs`.
//!   - Closures (`Rvalue::Closure`) with lifted lambda bodies and shared upvalue cells.
//!   - Dynamic fallback `lust_aot_rt::AotValue` for `MirType::Unknown` / heterogeneous values.
//!   - Deterministic `defer` & `errdefer` via `core::mem::DropGuard` with `DropGuard::dismiss`
//!     on every `Ok(...)` return (Hazard 9: guaranteed execution even under `-C panic=abort`).
//! - **Effect-Driven Signatures:** Functions proven infallible by
//!   `CraneliftJitCompiler::analyze_module_call_effects` (and emitter-level fallibility checks)
//!   return `Result<T, core::convert::Infallible>` (zero-cost register ABI); fallible functions
//!   return `Result<T, lust_aot_rt::AotError>`.

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Write as _;
use hashbrown::{HashMap, HashSet};

use crate::data::layout::{CanonicalTypeTag, LogicalLayoutV1};
use crate::error::{LustError, Result};
use crate::mir::cranelift_backend::jit_engine::{
    analyze_module_call_effects, apply_module_callee_return_types,
    refine_mir_function_with_layouts,
};
use crate::mir::ir::{
    BasicBlock, BlockId, LocalId, MirBinaryOp, MirCleanupEntry, MirCleanupKind, MirConstant,
    MirEnumDef, MirFunction, MirModule, MirStructLayout, MirType, MirUnaryOp, Operand, Place,
    ResourceKind, Rvalue, Statement, Terminator, parse_mir_struct_decode_callee,
};
use crate::mir::opt::{EscapeAnalysis, MirOptimizer, ScalarReplacementOfAggregates};
use crate::mir::verifier::MirVerifier;
use crate::vm::cranelift_dispatch::compute_readonly_params;

use super::cfg_structurer::{StructuredNode, structure_mir_function};

/// Gas metering policy for Rust AOT codegen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GasMetering {
    /// No gas instrumentation (default for `--release-fast`).
    #[default]
    None,
    /// Loop-header / safepoint gas charging.
    LoopChunked,
    /// Exact MIR `Statement::ResourceCheck(ResourceKind::Gas, ..)` charging.
    Exact,
}

/// Configuration for MIR -> Rust AOT emission.
#[derive(Debug, Clone)]
pub struct RustEmitterOptions {
    pub is_library: bool,
    pub max_memory_bytes: Option<usize>,
    pub gas_metering: GasMetering,
    pub gas_limit: Option<u64>,
    pub pure_core: bool,
    pub no_std: bool,
}

impl Default for RustEmitterOptions {
    fn default() -> Self {
        Self {
            is_library: false,
            max_memory_bytes: None,
            gas_metering: GasMetering::None,
            gas_limit: None,
            pure_core: false,
            no_std: false,
        }
    }
}

/// Runs the mandatory MIR optimization, SROA, type refinement, and escape-analysis pipeline
/// required before Rust AOT codegen (`NG-W12-4A` + `NG-W12-4B`).
pub fn prepare_mir_module_for_rust_aot(module: &MirModule) -> MirModule {
    let mut opt_module = module.clone();

    // 0. Canonicalize unqualified callee names against module-qualified function names
    //    (e.g. ModuleLoader qualifies `step` -> `calls.step`, while call sites say `step`).
    //    Doing this before MirOptimizer enables MirInliner, SROA, apply_module_callee_return_types,
    //    compute_readonly_params, and analyze_module_call_effects to match every intra-module call.
    canonicalize_module_callees(&mut opt_module);

    // 1. Run MirOptimizer while preserving `cleanup_blocks` (DCE only walks from `entry`,
    //    and inlining a cleanup-bearing function would lose its `cleanup_blocks`).
    let has_cleanups = opt_module
        .functions
        .iter()
        .any(|f| !f.cleanup_blocks.is_empty());
    let has_closures = opt_module.functions.iter().any(|f| {
        f.blocks.iter().any(|b| {
            b.statements
                .iter()
                .any(|s| matches!(s, Statement::Assign(_, Rvalue::Closure { .. }, _)))
        })
    });

    if !has_cleanups && !has_closures {
        // Tier 1-3 fast path (100% identical to NG-W12-4A for all 10 benchmarks)
        MirOptimizer::optimize_module(&mut opt_module);
    } else {
        // Detach cleanup subgraphs before per-function optimization, then re-attach with
        // remapped BlockIds so DCE never drops `is_cleanup == true` blocks.
        let mut saved_cleanups: Vec<Vec<(MirCleanupKind, Vec<BasicBlock>)>> =
            Vec::with_capacity(opt_module.functions.len());
        for func in &mut opt_module.functions {
            saved_cleanups.push(extract_cleanup_subgraphs(func));
            MirOptimizer::optimize_function(func);
        }
        for (func, cleanups) in opt_module.functions.iter_mut().zip(saved_cleanups) {
            reattach_cleanup_subgraphs(func, cleanups);
        }
    }

    // Build layout map for SROA and type refinement
    let mut struct_layouts: HashMap<String, MirStructLayout> = HashMap::new();
    for layout in &opt_module.struct_layouts {
        struct_layouts.insert(layout.type_name.clone(), layout.clone());
    }

    // 2. Run SROA on every function with fail-closed MirVerifier guard
    //    (skip functions with cleanup_blocks or aliased structs so SROA doesn't discard cleanup blocks
    //    or split aliased reference structs).
    if !struct_layouts.is_empty() {
        for func in &mut opt_module.functions {
            if !func.cleanup_blocks.is_empty() {
                continue;
            }
            let pre_sroa = func.clone();
            if ScalarReplacementOfAggregates::run_on_function(func, &struct_layouts) {
                MirOptimizer::optimize_function(func);
                if MirVerifier::verify_fn(func).is_err() {
                    *func = pre_sroa;
                }
            }
        }
    }

    // 3. Apply module callee return types and refine local/aggregate types with struct layouts
    let callee_rets: HashMap<String, MirType> = opt_module
        .functions
        .iter()
        .map(|f| (f.signature.name.clone(), f.signature.return_type.clone()))
        .collect();
    // Snapshot concrete declared types of `enum.payload` destinations before Cranelift's
    // lattice refiner (which hardcodes `enum.payload -> Int` and can conflict Float payloads into Unknown).
    let pre_refine_enum_payload_tys: Vec<Vec<(LocalId, MirType)>> = opt_module
        .functions
        .iter()
        .map(|f| {
            let mut saved = Vec::new();
            for blk in &f.blocks {
                if let Some(Terminator::Call {
                    callee,
                    destination: Place::Local(dst),
                    ..
                }) = &blk.terminator
                    && callee == "enum.payload"
                {
                    let ty = f.locals[dst.0].ty.clone();
                    if !matches!(ty, MirType::Unknown) {
                        saved.push((*dst, ty));
                    }
                }
            }
            saved
        })
        .collect();

    for func in &mut opt_module.functions {
        apply_module_callee_return_types(func, &callee_rets);
        *func = refine_mir_function_with_layouts(func, &struct_layouts);
    }

    for (func, saved) in opt_module
        .functions
        .iter_mut()
        .zip(pre_refine_enum_payload_tys)
    {
        for (dst, ty) in saved {
            if matches!(func.locals[dst.0].ty, MirType::Unknown) {
                func.locals[dst.0].ty = ty;
            }
        }
    }

    // 3b. Tier 4 type refinement for Map<K, V>, Enum, Closure, and dynamic AotValue locals
    refine_tier4_types(&mut opt_module);

    // 4. Run EscapeAnalysis across the refined module
    EscapeAnalysis::run_on_module(&mut opt_module);

    opt_module
}

fn extract_cleanup_subgraphs(func: &mut MirFunction) -> Vec<(MirCleanupKind, Vec<BasicBlock>)> {
    if func.cleanup_blocks.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(func.cleanup_blocks.len());
    for entry in &func.cleanup_blocks {
        let mut visited: Vec<BlockId> = Vec::new();
        let mut seen: HashSet<BlockId> = HashSet::new();
        let mut queue: VecDeque<BlockId> = VecDeque::new();
        if entry.block.0 < func.blocks.len() {
            seen.insert(entry.block);
            queue.push_back(entry.block);
        }
        while let Some(bb) = queue.pop_front() {
            visited.push(bb);
            if let Some(term) = &func.blocks[bb.0].terminator {
                for succ_idx in term.successors() {
                    let succ = BlockId(succ_idx);
                    if succ.0 < func.blocks.len() && seen.insert(succ) {
                        queue.push_back(succ);
                    }
                }
            }
        }
        let mut remap: HashMap<BlockId, BlockId> = HashMap::new();
        for (new_i, &old_bb) in visited.iter().enumerate() {
            remap.insert(old_bb, BlockId(new_i));
        }
        let mut sub_blocks = Vec::with_capacity(visited.len());
        for (new_i, &old_bb) in visited.iter().enumerate() {
            let mut blk = func.blocks[old_bb.0].clone();
            blk.id = BlockId(new_i);
            blk.is_cleanup = true;
            if let Some(term) = &mut blk.terminator {
                remap_terminator_blocks(term, &remap);
            }
            sub_blocks.push(blk);
        }
        result.push((entry.kind, sub_blocks));
    }
    func.cleanup_blocks.clear();
    result
}

fn reattach_cleanup_subgraphs(
    func: &mut MirFunction,
    cleanups: Vec<(MirCleanupKind, Vec<BasicBlock>)>,
) {
    for (kind, sub_blocks) in cleanups {
        if sub_blocks.is_empty() {
            continue;
        }
        let base = func.blocks.len();
        let mut remap: HashMap<BlockId, BlockId> = HashMap::new();
        for i in 0..sub_blocks.len() {
            remap.insert(BlockId(i), BlockId(base + i));
        }
        for (i, mut blk) in sub_blocks.into_iter().enumerate() {
            blk.id = BlockId(base + i);
            blk.is_cleanup = true;
            if let Some(term) = &mut blk.terminator {
                remap_terminator_blocks(term, &remap);
            }
            func.blocks.push(blk);
        }
        func.cleanup_blocks.push(MirCleanupEntry {
            kind,
            block: BlockId(base),
        });
    }
}

fn remap_terminator_blocks(term: &mut Terminator, remap: &HashMap<BlockId, BlockId>) {
    match term {
        Terminator::Goto(t, _) => {
            if let Some(&n) = remap.get(t) {
                *t = n;
            }
        }
        Terminator::Cond {
            then_block,
            else_block,
            ..
        } => {
            if let Some(&n) = remap.get(then_block) {
                *then_block = n;
            }
            if let Some(&n) = remap.get(else_block) {
                *else_block = n;
            }
        }
        Terminator::Switch { cases, default, .. } => {
            for (_, t) in cases {
                if let Some(&n) = remap.get(t) {
                    *t = n;
                }
            }
            if let Some(&n) = remap.get(default) {
                *default = n;
            }
        }
        Terminator::Call { success, error, .. } => {
            if let Some(&n) = remap.get(success) {
                *success = n;
            }
            if let Some(err_bb) = error
                && let Some(&n) = remap.get(err_bb)
            {
                *err_bb = n;
            }
        }
        Terminator::Return(_, _) | Terminator::Raise(_, _) | Terminator::Unreachable(_) => {}
    }
}

/// Refines Tier 4 MIR types (`Map<K, V>`, `Enum`, `Closure`, `Array<Unknown>`, `map.*` calls)
/// after Cranelift's general lattice refiner.
fn refine_tier4_types(module: &mut MirModule) {
    let enum_names: HashSet<String> = module.enum_defs.iter().map(|e| e.name.clone()).collect();
    let enum_defs_map: HashMap<String, MirEnumDef> = module
        .enum_defs
        .iter()
        .map(|e| (e.name.clone(), e.clone()))
        .collect();

    // 1. Propagate MirType::Function across copies (including Int placeholders from `mir_type_from_ast`)
    //    and infer concrete return types for lifted lambdas (Unknown) or closure-returning functions (Int).
    let mut lambda_ret_updates: HashMap<String, MirType> = HashMap::new();
    for func in &mut module.functions {
        // First propagate MirType::Function from Rvalue::Closure temps to local copies
        for _ in 0..4 {
            let mut changed = false;
            let local_tys: Vec<MirType> = func.locals.iter().map(|d| d.ty.clone()).collect();
            for block in &func.blocks {
                for stmt in &block.statements {
                    if let Statement::Assign(
                        Place::Local(dst),
                        Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))),
                        _,
                    ) = stmt
                        && matches!(local_tys[src.0], MirType::Function { .. })
                        && matches!(local_tys[dst.0], MirType::Unknown | MirType::Int)
                    {
                        func.locals[dst.0].ty = local_tys[src.0].clone();
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        if matches!(func.signature.return_type, MirType::Unknown | MirType::Int) {
            let mut inferred_ret: Option<MirType> = None;
            let mut conflict = false;
            for block in &func.blocks {
                if block.is_cleanup {
                    continue;
                }
                if let Some(Terminator::Return(Some(op), _)) = &block.terminator {
                    let ty = operand_mir_type(op, func);
                    if ty != MirType::Unknown {
                        match &inferred_ret {
                            None => inferred_ret = Some(ty),
                            Some(prev) if *prev == ty => {}
                            Some(_) => {
                                conflict = true;
                                break;
                            }
                        }
                    }
                }
            }
            if !conflict
                && let Some(rt) = inferred_ret
                && (func.signature.return_type == MirType::Unknown
                    || matches!(rt, MirType::Function { .. }))
            {
                func.signature.return_type = rt.clone();
                if let Some(rl) = func.locals.get_mut(func.return_local.0) {
                    rl.ty = rt.clone();
                }
                lambda_ret_updates.insert(func.signature.name.clone(), rt);
            }
        }
    }

    // 2. Per-function Tier 4 refinement
    let module_fn_rets: HashMap<String, MirType> = module
        .functions
        .iter()
        .map(|f| (f.signature.name.clone(), f.signature.return_type.clone()))
        .collect();

    for func in &mut module.functions {
        // 2a-pre. Restore MirType::Function / MirType::Map / MirType::Enum / StructDecode call return types
        //         from module callees if Cranelift's refiner conflicted them with Int placeholders.
        for block in &func.blocks {
            if let Some(Terminator::Call {
                callee,
                destination: Place::Local(dst),
                ..
            }) = &block.terminator
            {
                if let Some(sname) = parse_mir_struct_decode_callee(callee)
                    .or_else(|| callee.strip_prefix("__lust_mir_struct_decode_binary:"))
                {
                    func.locals[dst.0].ty = MirType::Struct(sname.to_string());
                } else if let Some(ret_ty) = module_fn_rets.get(callee)
                    && matches!(
                        ret_ty,
                        MirType::Function { .. } | MirType::Map(_, _) | MirType::Enum(_)
                    )
                {
                    func.locals[dst.0].ty = ret_ty.clone();
                }
            }
        }

        // 2a. Update closure local return types if lifted lambda return type was inferred
        if !lambda_ret_updates.is_empty() {
            for block in &func.blocks {
                for stmt in &block.statements {
                    if let Statement::Assign(
                        Place::Local(dst),
                        Rvalue::Closure { func_name, .. },
                        _,
                    ) = stmt
                        && let Some(new_ret) = lambda_ret_updates.get(func_name)
                        && let Some(decl) = func.locals.get_mut(dst.0)
                        && let MirType::Function { return_type, .. } = &mut decl.ty
                        && **return_type == MirType::Unknown
                    {
                        **return_type = new_ret.clone();
                    }
                }
            }
        }

        // 2b. Propagate closure types from `temp = Closure` / call-return into `local f = Move/Copy(temp)`
        //     and Map/Enum types across simple local copies when destination was Unknown or Int placeholder.
        for _ in 0..4 {
            let mut changed = false;
            let local_tys: Vec<MirType> = func.locals.iter().map(|d| d.ty.clone()).collect();
            for block in &func.blocks {
                for stmt in &block.statements {
                    if let Statement::Assign(
                        Place::Local(dst),
                        Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))),
                        _,
                    ) = stmt
                    {
                        let st = &local_tys[src.0];
                        let dt = &local_tys[dst.0];
                        if matches!(dt, MirType::Unknown) && !matches!(st, MirType::Unknown) {
                            func.locals[dst.0].ty = st.clone();
                            changed = true;
                        } else if matches!(dt, MirType::Int)
                            && matches!(st, MirType::Function { .. } | MirType::Map(_, _))
                        {
                            func.locals[dst.0].ty = st.clone();
                            changed = true;
                        } else if let (MirType::Map(sk, sv), MirType::Map(dk, dv)) = (st, dt) {
                            let new_k = if **dk == MirType::Unknown && **sk != MirType::Unknown {
                                sk.clone()
                            } else {
                                dk.clone()
                            };
                            let new_v = if **dv == MirType::Unknown && **sv != MirType::Unknown {
                                sv.clone()
                            } else {
                                dv.clone()
                            };
                            let unified = MirType::Map(new_k, new_v);
                            if unified != func.locals[dst.0].ty {
                                func.locals[dst.0].ty = unified.clone();
                                changed = true;
                            }
                            if unified != func.locals[src.0].ty {
                                func.locals[src.0].ty = unified;
                                changed = true;
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // 2c. Infer Map<K, V> key/value types from map.set / map.get / map.has / map.delete calls
        let mut map_key_ev: HashMap<LocalId, Option<MirType>> = HashMap::new();
        let mut map_val_ev: HashMap<LocalId, Option<MirType>> = HashMap::new();
        let mut map_key_conflict: HashSet<LocalId> = HashSet::new();
        let mut map_val_conflict: HashSet<LocalId> = HashSet::new();

        for block in &func.blocks {
            if let Some(Terminator::Call { callee, args, .. }) = &block.terminator {
                match callee.as_str() {
                    "map.set" | "table.set" if args.len() >= 3 => {
                        if let Some(mloc) = operand_exact_local(&args[0])
                            && matches!(func.locals[mloc.0].ty, MirType::Map(_, _))
                        {
                            let kt = operand_mir_type(&args[1], func);
                            if kt == MirType::Unknown {
                                map_key_conflict.insert(mloc);
                            } else {
                                match map_key_ev.get(&mloc) {
                                    None => {
                                        map_key_ev.insert(mloc, Some(kt));
                                    }
                                    Some(Some(prev)) if *prev == kt => {}
                                    _ => {
                                        map_key_conflict.insert(mloc);
                                    }
                                }
                            }
                            let vt = operand_mir_type(&args[2], func);
                            if vt == MirType::Unknown {
                                map_val_conflict.insert(mloc);
                            } else {
                                match map_val_ev.get(&mloc) {
                                    None => {
                                        map_val_ev.insert(mloc, Some(vt));
                                    }
                                    Some(Some(prev)) if *prev == vt => {}
                                    _ => {
                                        map_val_conflict.insert(mloc);
                                    }
                                }
                            }
                        }
                    }
                    "map.get" | "map.has" | "map.delete" | "map.remove" if args.len() >= 2 => {
                        if let Some(mloc) = operand_exact_local(&args[0])
                            && matches!(func.locals[mloc.0].ty, MirType::Map(_, _))
                        {
                            let kt = operand_mir_type(&args[1], func);
                            if kt != MirType::Unknown {
                                match map_key_ev.get(&mloc) {
                                    None => {
                                        map_key_ev.insert(mloc, Some(kt));
                                    }
                                    Some(Some(prev)) if *prev == kt => {}
                                    _ => {
                                        map_key_conflict.insert(mloc);
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        for idx in 0..func.locals.len() {
            let loc = LocalId(idx);
            if let MirType::Map(k, v) = func.locals[idx].ty.clone() {
                let final_k = if *k == MirType::Unknown && !map_key_conflict.contains(&loc) {
                    map_key_ev
                        .get(&loc)
                        .and_then(|o| o.clone())
                        .unwrap_or(MirType::Unknown)
                } else {
                    *k
                };
                let final_v = if *v == MirType::Unknown && !map_val_conflict.contains(&loc) {
                    map_val_ev
                        .get(&loc)
                        .and_then(|o| o.clone())
                        .unwrap_or(MirType::Unknown)
                } else {
                    *v
                };
                func.locals[idx].ty = MirType::Map(Box::new(final_k), Box::new(final_v));
            }
        }

        // Propagate refined Map<K, V> back into map.new temps and aliases
        for _ in 0..3 {
            let local_tys: Vec<MirType> = func.locals.iter().map(|d| d.ty.clone()).collect();
            for block in &func.blocks {
                for stmt in &block.statements {
                    if let Statement::Assign(
                        Place::Local(dst),
                        Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))),
                        _,
                    ) = stmt
                        && let (MirType::Map(sk, sv), MirType::Map(dk, dv)) =
                            (&local_tys[src.0], &local_tys[dst.0])
                    {
                        let rk = if **sk != MirType::Unknown { sk } else { dk };
                        let rv = if **sv != MirType::Unknown { sv } else { dv };
                        let unified = MirType::Map(rk.clone(), rv.clone());
                        func.locals[src.0].ty = unified.clone();
                        func.locals[dst.0].ty = unified;
                    }
                }
            }
        }

        // 2d. Fix call return types for map.*, enum.*, EnumName.Variant, and local closure calls
        let name_to_fn_ret: HashMap<String, MirType> = func
            .locals
            .iter()
            .filter_map(|d| {
                if let (Some(n), MirType::Function { return_type, .. }) = (&d.name, &d.ty) {
                    Some((n.clone(), (**return_type).clone()))
                } else {
                    None
                }
            })
            .collect();

        // Collect all locals assigned directly from a `map.get` temp
        let mut map_get_val_types: HashMap<LocalId, MirType> = HashMap::new();
        for block in &func.blocks {
            if let Some(Terminator::Call {
                callee,
                args,
                destination: Place::Local(dst),
                ..
            }) = &block.terminator
            {
                match callee.as_str() {
                    "map.get" | "table.get" if !args.is_empty() => {
                        if let MirType::Map(_, v_ty) = operand_mir_type(&args[0], func) {
                            // `AstLowering` lowers `map.get(m, k):unwrap()` directly to `map.get`,
                            // while `refine_mir_function_with_layouts` forces `Enum("Option")`.
                            // Keep Option only if destination is explicitly `MirType::Option(_)`.
                            if !matches!(func.locals[dst.0].ty, MirType::Option(_)) {
                                func.locals[dst.0].ty = (*v_ty).clone();
                                map_get_val_types.insert(*dst, (*v_ty).clone());
                            }
                        }
                    }
                    "map.has" | "map.is_empty" => {
                        func.locals[dst.0].ty = MirType::Bool;
                    }
                    "map.len" => {
                        func.locals[dst.0].ty = MirType::Int;
                    }
                    "map.delete" | "map.remove" if !args.is_empty() => {
                        if let MirType::Map(_, v_ty) = operand_mir_type(&args[0], func) {
                            func.locals[dst.0].ty = MirType::Option(v_ty);
                        }
                    }
                    "map.keys" if !args.is_empty() => {
                        if let MirType::Map(k_ty, _) = operand_mir_type(&args[0], func) {
                            func.locals[dst.0].ty = MirType::Array(k_ty);
                        }
                    }
                    "map.values" if !args.is_empty() => {
                        if let MirType::Map(_, v_ty) = operand_mir_type(&args[0], func) {
                            func.locals[dst.0].ty = MirType::Array(v_ty);
                        }
                    }
                    "enum.tag" => {
                        func.locals[dst.0].ty = MirType::Int;
                    }
                    "enum.payload" if !args.is_empty() => {
                        if let MirType::Enum(ename) = operand_mir_type(&args[0], func)
                            && let Some(edef) = enum_defs_map.get(&ename)
                        {
                            let fidx = args
                                .get(1)
                                .and_then(|op| match op {
                                    Operand::Constant(MirConstant::Int(i)) if *i >= 0 => {
                                        Some(*i as usize)
                                    }
                                    _ => None,
                                })
                                .unwrap_or(0);
                            let mut candidate: Option<MirType> = None;
                            let mut multi = false;
                            for (_, fields) in &edef.variants {
                                if let Some(ft) = fields.get(fidx) {
                                    match &candidate {
                                        None => candidate = Some(ft.clone()),
                                        Some(prev) if prev == ft => {}
                                        Some(_) => {
                                            multi = true;
                                            break;
                                        }
                                    }
                                }
                            }
                            if !multi && let Some(ft) = candidate {
                                func.locals[dst.0].ty = ft;
                            }
                        }
                    }
                    _ => {
                        // EnumName.Variant or EnumName::Variant constructor calls
                        if let Some((ename, _vname)) = callee
                            .split_once("::")
                            .or_else(|| callee.split_once('.'))
                            && enum_names.contains(ename)
                        {
                            func.locals[dst.0].ty = MirType::Enum(ename.to_string());
                        } else if let Some(ret_ty) = name_to_fn_ret.get(callee) {
                            func.locals[dst.0].ty = ret_ty.clone();
                        }
                    }
                }
            }
        }

        // Restore locals whose type was turned into Conflict/Unknown or Enum("Option")
        // when assigned directly from a refined `map.get` temp (e.g. `local v: int = map.get(m, k):unwrap()`).
        if !map_get_val_types.is_empty() {
            for block in &func.blocks {
                for stmt in &block.statements {
                    if let Statement::Assign(
                        Place::Local(dst),
                        Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))),
                        _,
                    ) = stmt
                        && let Some(v_ty) = map_get_val_types.get(src)
                        && matches!(
                            func.locals[dst.0].ty,
                            MirType::Unknown | MirType::Enum(_)
                        )
                    {
                        func.locals[dst.0].ty = v_ty.clone();
                    }
                }
            }
        }
    }
}

fn canonicalize_module_callees(module: &mut MirModule) {
    let exact_names: HashSet<String> = module
        .functions
        .iter()
        .map(|f| f.signature.name.clone())
        .collect();
    // Map unambiguous short names (`mod.foo` -> `foo`) to their canonical full name.
    let mut short_to_full: HashMap<String, Option<String>> = HashMap::new();
    for f in &module.functions {
        if let Some((_, short)) = f.signature.name.rsplit_once('.') {
            short_to_full
                .entry(short.to_string())
                .and_modify(|slot| *slot = None)
                .or_insert_with(|| Some(f.signature.name.clone()));
        }
    }
    for func in &mut module.functions {
        for block in &mut func.blocks {
            if let Some(Terminator::Call { callee, .. }) = &mut block.terminator
                && !exact_names.contains(callee.as_str())
                && let Some(Some(canonical)) = short_to_full.get(callee.as_str())
            {
                *callee = canonical.clone();
            }
        }
    }
}

/// Generates idiomatic, panic-free, `unsafe`-free Rust source code from a `MirModule`.
pub fn emit_rust_from_mir(module: &MirModule, opts: &RustEmitterOptions) -> Result<String> {
    let prepared = prepare_mir_module_for_rust_aot(module);
    let ctx = ModuleEmitContext::analyze(&prepared, opts);

    let mut out = String::with_capacity(8192);
    emit_file_header(&mut out, opts);
    emit_struct_definitions(&mut out, &ctx);
    emit_enum_definitions(&mut out, &ctx);

    for func in &prepared.functions {
        emit_function(&mut out, func, &ctx)?;
    }

    if !opts.is_library {
        emit_entrypoint(&mut out, &prepared, &ctx);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Module-Level Analysis (Structs, Enums, Aliasing, Closures, Lifetimes, Effects)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SchemaFieldMeta {
    name: String,
    tag: CanonicalTypeTag,
    mir_ty: MirType,
    is_nullable: bool,
    is_optional: bool,
    is_refined_array: bool,
    nested_struct_name: Option<String>,
}

#[derive(Debug, Clone)]
struct StructMeta {
    name: String,
    fields: Vec<(String, MirType)>,
    schema_fields: Vec<SchemaFieldMeta>,
    logical_layout: LogicalLayoutV1,
    resolved_layout: Option<LogicalLayoutV1>,
    has_exact_repr_c_layout: bool,
    has_schema_decoders: bool,
    is_cyclic: bool,
    needs_arena_lt: bool,
    is_copy: bool,
    derives_default: bool,
}

#[derive(Debug, Clone)]
struct EnumMeta {
    name: String,
    variants: Vec<(String, Vec<MirType>)>,
    needs_arena_lt: bool,
    is_copy: bool,
    derives_default: bool,
}

/// Per-local storage strategy (`NG-W12-4B`, Hazard 12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalStorage {
    /// Direct ownership / borrowed ref (Tier 1-3 fast path).
    Direct,
    /// Shared mutable `Copy` upvalue cell `alloc::rc::Rc<core::cell::Cell<T>>`.
    RcCell,
    /// Shared mutable container/struct/non-Copy `alloc::rc::Rc<core::cell::RefCell<T>>`.
    RcRefCell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParamMode {
    /// Passed by value (`i64`, `f64`, `bool`, `()`, `Copy` struct, `AotMap`, `Enum`, `Closure`, `AotValue`, or owned non-Copy value).
    ByValue,
    /// Passed by shared reference `&LustStruct_S<'a>` (read-only non-Copy struct).
    RefStruct,
    /// Passed by mutable reference `&mut LustStruct_S` (mutated struct param, e.g. `self`).
    MutRefStruct,
    /// Passed by shared slice `&[T]` (read-only array).
    RefSlice,
    /// Passed by mutable vector reference `&mut Vec<T, &'a LustBudgetAlloc>`.
    MutVec,
    /// Passed by shared string slice `&str` (read-only string).
    RefStr,
    /// Passed as shared `Rc<Cell<T>>` (mutable Copy upvalue captured by lifted lambda).
    RcCell,
    /// Passed as shared `Rc<RefCell<T>>` (locally aliased or captured container/struct).
    RcRefCell,
}

#[derive(Debug, Clone)]
struct FnMeta {
    rust_name: String,
    can_fail: bool,
    needs_ctx: bool,
    needs_alloc: bool,
    needs_arena: bool,
    param_modes: Vec<ParamMode>,
    local_storage: Vec<LocalStorage>,
    /// Locals referenced inside any `cleanup_blocks` entry of this function.
    cleanup_captured: HashSet<LocalId>,
    return_type: MirType,
}

struct ModuleEmitContext<'m> {
    module: &'m MirModule,
    opts: &'m RustEmitterOptions,
    structs: BTreeMap<String, StructMeta>,
    enums: BTreeMap<String, EnumMeta>,
    fn_metas: HashMap<String, FnMeta>,
    module_uses_arena: bool,
    /// Whether any closure in the module captures a type requiring `'a`.
    closures_need_lt: bool,
}

impl<'m> ModuleEmitContext<'m> {
    fn analyze(module: &'m MirModule, opts: &'m RustEmitterOptions) -> Self {
        let structs = analyze_structs(module);
        let enums = analyze_enums(module, &structs);
        let module_uses_arena = structs.values().any(|s| s.is_cyclic);

        let readonly_params = compute_readonly_params(module);
        let gas_active = opts.gas_metering != GasMetering::None;
        let call_effects = analyze_module_call_effects(module, gas_active);

        let fn_name_set: HashSet<String> = module
            .functions
            .iter()
            .map(|f| f.signature.name.clone())
            .collect();
        let fn_by_name: HashMap<&str, &MirFunction> = module
            .functions
            .iter()
            .map(|f| (f.signature.name.as_str(), f))
            .collect();

        // First pass: determine baseline param modes per function
        let mut param_modes_map: HashMap<String, Vec<ParamMode>> = HashMap::new();
        for func in &module.functions {
            let ro_mask = readonly_params.get(&func.signature.name);
            let modes = compute_fn_param_modes(func, ro_mask, &structs, &enums);
            param_modes_map.insert(func.signature.name.clone(), modes);
        }

        // Second pass (Tier 4): analyze local storage (`RcCell`/`RcRefCell`) for:
        // (a) cleanup_blocks captures, (b) Closure upvalue captures, (c) local aliasing (Hazard 12).
        let mut local_storage_map: HashMap<String, Vec<LocalStorage>> = HashMap::new();
        let mut cleanup_captured_map: HashMap<String, HashSet<LocalId>> = HashMap::new();
        for func in &module.functions {
            local_storage_map.insert(
                func.signature.name.clone(),
                vec![LocalStorage::Direct; func.locals.len()],
            );
            cleanup_captured_map.insert(func.signature.name.clone(), HashSet::new());
        }

        let mut closures_need_lt = false;

        // 2a. Closure capture analysis (enclosing locals + lifted lambda param modes)
        for func in &module.functions {
            let enclosing_name = func.signature.name.clone();
            for block in &func.blocks {
                for stmt in &block.statements {
                    if let Statement::Assign(
                        _,
                        Rvalue::Closure {
                            func_name,
                            captures,
                        },
                        _,
                    ) = stmt
                    {
                        let lifted_fn = fn_by_name.get(func_name.as_str()).copied();
                        for (ci, (_, cap_op, cap_ty)) in captures.iter().enumerate() {
                            if type_needs_lt_raw(cap_ty, &structs, &enums) {
                                closures_need_lt = true;
                            }
                            let lifted_mutates_cap = lifted_fn.is_some_and(|lf| {
                                lf.params
                                    .get(ci)
                                    .is_some_and(|&pid| local_is_mutated_in_fn(lf, pid))
                            });
                            let enclosing_loc = operand_exact_local(cap_op);
                            let enclosing_mutates_cap = enclosing_loc
                                .is_some_and(|loc| local_is_mutated_in_fn(func, loc));
                            let is_ref_container =
                                matches!(cap_ty, MirType::Array(_) | MirType::Struct(_));

                            if lifted_mutates_cap || enclosing_mutates_cap || is_ref_container {
                                let is_copy_scalar = matches!(
                                    cap_ty,
                                    MirType::Int | MirType::Float | MirType::Bool
                                );
                                if is_copy_scalar {
                                    // Match Lust VM `Instruction::Closure` (`Upvalue::new(reg.clone())`):
                                    // scalar upvalues live in a per-closure `Rc<Cell<T>>` so mutations
                                    // persist across calls to the closure without aliasing the outer scalar slot.
                                    if lifted_mutates_cap {
                                        if let Some(lmodes) = param_modes_map.get_mut(func_name)
                                            && ci < lmodes.len()
                                        {
                                            lmodes[ci] = ParamMode::RcCell;
                                        }
                                        if let Some(lf) = lifted_fn
                                            && let Some(&pid) = lf.params.get(ci)
                                            && let Some(lvec) = local_storage_map.get_mut(func_name)
                                            && pid.0 < lvec.len()
                                        {
                                            lvec[pid.0] = LocalStorage::RcCell;
                                        }
                                    }
                                } else if !matches!(
                                    cap_ty,
                                    MirType::Map(_, _) | MirType::Unknown | MirType::Function { .. }
                                ) || lifted_mutates_cap
                                {
                                    // Non-Copy or container/struct (or reassigned Map/AotValue binding)
                                    let is_cyclic_struct = matches!(cap_ty, MirType::Struct(sn) if structs.get(sn).is_some_and(|s| s.is_cyclic));
                                    if !is_cyclic_struct {
                                        if let Some(loc) = enclosing_loc
                                            && let Some(vec_st) =
                                                local_storage_map.get_mut(&enclosing_name)
                                            && loc.0 < vec_st.len()
                                        {
                                            vec_st[loc.0] = LocalStorage::RcRefCell;
                                        }
                                        if let Some(lmodes) = param_modes_map.get_mut(func_name)
                                            && ci < lmodes.len()
                                        {
                                            lmodes[ci] = ParamMode::RcRefCell;
                                        }
                                        if let Some(lf) = lifted_fn
                                            && let Some(&pid) = lf.params.get(ci)
                                            && let Some(lvec) = local_storage_map.get_mut(func_name)
                                            && pid.0 < lvec.len()
                                        {
                                            lvec[pid.0] = LocalStorage::RcRefCell;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2b. Cleanup block (`defer` / `errdefer`) capture analysis
        for func in &module.functions {
            if func.cleanup_blocks.is_empty() {
                continue;
            }
            let captured = collect_cleanup_captured_locals(func);
            let fname = &func.signature.name;
            let param_set: HashSet<LocalId> = func.params.iter().copied().collect();
            for &loc in &captured {
                let ty = &func.locals[loc.0].ty;
                let is_copy_scalar =
                    matches!(ty, MirType::Int | MirType::Float | MirType::Bool);
                let is_cyclic = matches!(ty, MirType::Struct(sn) if structs.get(sn).is_some_and(|s| s.is_cyclic));
                if is_cyclic || matches!(ty, MirType::Unit) {
                    continue;
                }
                if param_set.contains(&loc) {
                    if let Some(pi) = func.params.iter().position(|&p| p == loc) {
                        if is_copy_scalar && local_is_mutated_in_fn(func, loc) {
                            if let Some(st) = local_storage_map.get_mut(fname) {
                                st[loc.0] = LocalStorage::RcCell;
                            }
                        } else if matches!(ty, MirType::Array(_) | MirType::Struct(_)) {
                            if let Some(pms) = param_modes_map.get_mut(fname) {
                                pms[pi] = ParamMode::RcRefCell;
                            }
                            if let Some(st) = local_storage_map.get_mut(fname) {
                                st[loc.0] = LocalStorage::RcRefCell;
                            }
                        }
                    }
                } else if let Some(st) = local_storage_map.get_mut(fname) {
                    if is_copy_scalar {
                        if st[loc.0] == LocalStorage::Direct {
                            st[loc.0] = LocalStorage::RcCell;
                        }
                    } else {
                        st[loc.0] = LocalStorage::RcRefCell;
                    }
                }
            }
            cleanup_captured_map.insert(fname.clone(), captured);
        }

        // 2c. Local Aliasing Analysis (Hazard 12: `local b = a` on Array or mutable Struct)
        for func in &module.functions {
            let fname = &func.signature.name;
            let aliased = detect_aliased_container_locals(func, &structs, &fn_name_set);
            if !aliased.is_empty() {
                let param_set: HashSet<LocalId> = func.params.iter().copied().collect();
                if let Some(st) = local_storage_map.get_mut(fname) {
                    for loc in aliased {
                        if !param_set.contains(&loc) && loc.0 < st.len() {
                            st[loc.0] = LocalStorage::RcRefCell;
                        }
                    }
                }
            }
        }

        // 2d. Propagate ParamMode::RcRefCell if a user function is called with >= 2 RcRefCell args
        //     of the same type (preventing overlapping RefMut/Ref guards across the call boundary)
        //     OR if a callee expects RcRefCell and caller passes a local.
        for _ in 0..3 {
            let mut changed = false;
            for func in &module.functions {
                let caller_st = local_storage_map[&func.signature.name].clone();
                for block in &func.blocks {
                    if let Some(Terminator::Call { callee, args, .. }) = &block.terminator
                        && let Some(callee_fn) = fn_by_name.get(callee.as_str())
                    {
                        // Count RcRefCell args by MirType
                        let mut rc_arg_indices: Vec<usize> = Vec::new();
                        for (ai, arg) in args.iter().enumerate() {
                            if let Some(loc) = operand_exact_local(arg)
                                && caller_st.get(loc.0).copied() == Some(LocalStorage::RcRefCell)
                            {
                                rc_arg_indices.push(ai);
                            }
                        }
                        if rc_arg_indices.len() >= 2 {
                            for &ai in &rc_arg_indices {
                                if let Some(pms) = param_modes_map.get_mut(callee)
                                    && ai < pms.len()
                                    && matches!(
                                        pms[ai],
                                        ParamMode::MutVec
                                            | ParamMode::RefSlice
                                            | ParamMode::MutRefStruct
                                            | ParamMode::RefStruct
                                    )
                                {
                                    pms[ai] = ParamMode::RcRefCell;
                                    if let Some(&pid) = callee_fn.params.get(ai)
                                        && let Some(cst) = local_storage_map.get_mut(callee)
                                    {
                                        cst[pid.0] = LocalStorage::RcRefCell;
                                    }
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // Ensure param_modes::RcRefCell / RcCell agrees with local_storage for every param
        for func in &module.functions {
            let fname = &func.signature.name;
            let pms = &param_modes_map[fname];
            let st = local_storage_map.get_mut(fname).unwrap();
            for (pi, &pid) in func.params.iter().enumerate() {
                match pms[pi] {
                    ParamMode::RcCell => st[pid.0] = LocalStorage::RcCell,
                    ParamMode::RcRefCell => st[pid.0] = LocalStorage::RcRefCell,
                    _ => {}
                }
            }
        }

        // Third pass: determine local requirements (can_fail, needs_ctx, needs_alloc, needs_arena)
        let mut can_fail_map: HashMap<String, bool> = HashMap::new();
        let mut needs_ctx_map: HashMap<String, bool> = HashMap::new();
        let mut needs_alloc_map: HashMap<String, bool> = HashMap::new();
        let mut needs_arena_map: HashMap<String, bool> = HashMap::new();
        let mut direct_callees: HashMap<String, Vec<String>> = HashMap::new();

        for func in &module.functions {
            let name = &func.signature.name;
            let cranelift_can_fail = call_effects
                .get(name)
                .map(|&(_, fl)| fl)
                .unwrap_or(true);
            let pmodes = &param_modes_map[name];
            let lstorage = &local_storage_map[name];

            let local_reqs = scan_function_local_requirements(
                func,
                opts,
                &structs,
                &enums,
                pmodes,
                lstorage,
                module,
            );
            let has_rc_refcell = lstorage.iter().any(|s| *s == LocalStorage::RcRefCell);
            let fn_can_fail = local_reqs.can_fail
                || has_rc_refcell
                || (gas_active && cranelift_can_fail)
                || (opts.max_memory_bytes.is_some()
                    && (local_reqs.needs_alloc
                        || local_reqs.needs_arena
                        || local_reqs.allocates_string));

            can_fail_map.insert(name.clone(), fn_can_fail);
            needs_ctx_map.insert(name.clone(), local_reqs.needs_ctx);
            needs_alloc_map.insert(name.clone(), local_reqs.needs_alloc);
            needs_arena_map.insert(name.clone(), local_reqs.needs_arena);
            direct_callees.insert(name.clone(), local_reqs.user_callees);
        }

        // Propagate can_fail, needs_ctx, needs_alloc, needs_arena transitively over call graph
        let fn_names: Vec<String> = module
            .functions
            .iter()
            .map(|f| f.signature.name.clone())
            .collect();
        loop {
            let mut changed = false;
            for caller in &fn_names {
                let callees = match direct_callees.get(caller) {
                    Some(c) => c.clone(),
                    None => continue,
                };
                for callee in &callees {
                    if let Some(&cf) = can_fail_map.get(callee)
                        && cf
                        && !can_fail_map[caller]
                    {
                        can_fail_map.insert(caller.clone(), true);
                        changed = true;
                    }
                    if let Some(&nc) = needs_ctx_map.get(callee)
                        && nc
                        && !needs_ctx_map[caller]
                    {
                        needs_ctx_map.insert(caller.clone(), true);
                        changed = true;
                    }
                    if let Some(&na) = needs_alloc_map.get(callee)
                        && na
                        && !needs_alloc_map[caller]
                    {
                        needs_alloc_map.insert(caller.clone(), true);
                        changed = true;
                    }
                    if let Some(&nar) = needs_arena_map.get(callee)
                        && nar
                        && !needs_arena_map[caller]
                    {
                        needs_arena_map.insert(caller.clone(), true);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        let mut fn_metas = HashMap::new();
        for func in &module.functions {
            let name = func.signature.name.clone();
            let rust_name = rust_fn_name(&name);
            let param_modes = param_modes_map.remove(&name).unwrap_or_default();
            let local_storage = local_storage_map.remove(&name).unwrap_or_default();
            let cleanup_captured = cleanup_captured_map.remove(&name).unwrap_or_default();
            fn_metas.insert(
                name.clone(),
                FnMeta {
                    rust_name,
                    can_fail: can_fail_map[&name],
                    needs_ctx: needs_ctx_map[&name],
                    needs_alloc: needs_alloc_map[&name],
                    needs_arena: needs_arena_map[&name],
                    param_modes,
                    local_storage,
                    cleanup_captured,
                    return_type: func.signature.return_type.clone(),
                },
            );
        }

        Self {
            module,
            opts,
            structs,
            enums,
            fn_metas,
            module_uses_arena,
            closures_need_lt,
        }
    }

    fn is_copy_type(&self, ty: &MirType) -> bool {
        match ty {
            MirType::Unit | MirType::Bool | MirType::Int | MirType::Float => true,
            MirType::Struct(name) => self.structs.get(name).is_some_and(|s| s.is_copy),
            MirType::Enum(name) => self.enums.get(name).is_some_and(|e| e.is_copy),
            MirType::Option(inner) => match inner.as_ref() {
                MirType::Struct(name) => self
                    .structs
                    .get(name)
                    .is_some_and(|s| s.is_copy && !s.is_cyclic),
                other => self.is_copy_type(other),
            },
            MirType::Tuple(elems) => elems.iter().all(|e| self.is_copy_type(e)),
            _ => false,
        }
    }

    fn type_needs_lifetime(&self, ty: &MirType) -> bool {
        match ty {
            MirType::Array(_) => true,
            MirType::Map(_, v) => self.type_needs_lifetime(v),
            MirType::Struct(name) => self
                .structs
                .get(name)
                .is_some_and(|s| s.needs_arena_lt),
            MirType::Enum(name) => self
                .enums
                .get(name)
                .is_some_and(|e| e.needs_arena_lt),
            MirType::Option(inner) => match inner.as_ref() {
                MirType::Struct(name) => self
                    .structs
                    .get(name)
                    .is_some_and(|s| s.is_cyclic || s.needs_arena_lt),
                other => self.type_needs_lifetime(other),
            },
            MirType::Tuple(elems) => elems.iter().any(|e| self.type_needs_lifetime(e)),
            MirType::Function {
                params,
                return_type,
            } => {
                self.closures_need_lt
                    || params.iter().any(|p| self.type_needs_lifetime(p))
                    || self.type_needs_lifetime(return_type)
            }
            _ => false,
        }
    }

    fn struct_is_cyclic(&self, name: &str) -> bool {
        self.structs.get(name).is_some_and(|s| s.is_cyclic)
    }

    fn rust_map_key_type(&self, k: &MirType) -> String {
        match k {
            MirType::Int => "i64".to_string(),
            MirType::String => "alloc::string::String".to_string(),
            MirType::Bool => "bool".to_string(),
            _ => "lust_aot_rt::AotKey".to_string(),
        }
    }

    fn rust_type(&self, ty: &MirType, lt: &str) -> String {
        match ty {
            MirType::Unit => "()".to_string(),
            MirType::Bool => "bool".to_string(),
            MirType::Int => "i64".to_string(),
            MirType::Float => "f64".to_string(),
            MirType::String => "alloc::string::String".to_string(),
            MirType::Array(elem) => {
                let inner = self.rust_type(elem, lt);
                format!("alloc::vec::Vec<{inner}, &{lt} lust_aot_rt::LustBudgetAlloc>")
            }
            MirType::Map(k, v) => {
                let rk = self.rust_map_key_type(k);
                let rv = self.rust_type(v, lt);
                format!("lust_aot_rt::AotMap<{rk}, {rv}>")
            }
            MirType::Struct(name) => {
                let sname = rust_struct_name(name);
                if self.structs.get(name).is_some_and(|s| s.needs_arena_lt) {
                    format!("{sname}<{lt}>")
                } else {
                    sname
                }
            }
            MirType::Enum(name) => {
                if let Some(emeta) = self.enums.get(name) {
                    let ename = rust_enum_name(name);
                    if emeta.needs_arena_lt {
                        format!("{ename}<{lt}>")
                    } else {
                        ename
                    }
                } else {
                    "lust_aot_rt::AotValue".to_string()
                }
            }
            MirType::Option(inner) => match inner.as_ref() {
                MirType::Struct(name) if self.struct_is_cyclic(name) => {
                    let sname = rust_struct_name(name);
                    format!(
                        "Option<alloc::boxed::Box<{sname}<{lt}>, &{lt} lust_aot_rt::LustChunkArena<{lt}>>>"
                    )
                }
                other => {
                    let inner_ty = self.rust_type(other, lt);
                    format!("Option<{inner_ty}>")
                }
            },
            MirType::Tuple(elems) => {
                if elems.is_empty() {
                    "()".to_string()
                } else {
                    let parts: Vec<String> =
                        elems.iter().map(|e| self.rust_type(e, lt)).collect();
                    if parts.len() == 1 {
                        format!("({},)", parts[0])
                    } else {
                        format!("({})", parts.join(", "))
                    }
                }
            }
            MirType::Function {
                params,
                return_type,
            } => {
                let needs_lt = self.type_needs_lifetime(ty);
                let mut p_strs = Vec::with_capacity(params.len() + 2);
                p_strs.push("&mut lust_aot_rt::AotCtx".to_string());
                if needs_lt {
                    p_strs.push(format!("&{lt} lust_aot_rt::LustBudgetAlloc"));
                } else {
                    p_strs.push("&lust_aot_rt::LustBudgetAlloc".to_string());
                }
                for p in params {
                    p_strs.push(self.rust_type(p, lt));
                }
                let r = self.rust_type(return_type, lt);
                if needs_lt {
                    format!(
                        "alloc::rc::Rc<dyn Fn({}) -> Result<{r}, lust_aot_rt::AotError> + {lt}>",
                        p_strs.join(", ")
                    )
                } else {
                    format!(
                        "alloc::rc::Rc<dyn Fn({}) -> Result<{r}, lust_aot_rt::AotError>>",
                        p_strs.join(", ")
                    )
                }
            }
            _ => "lust_aot_rt::AotValue".to_string(),
        }
    }

    fn rust_schema_field_type(&self, sf: &SchemaFieldMeta, lt: &str) -> String {
        if sf.is_refined_array {
            return self.rust_type(&sf.mir_ty, lt);
        }
        let is_opt = sf.is_optional || sf.is_nullable;
        let base = match sf.tag {
            CanonicalTypeTag::Bool => "bool".to_string(),
            CanonicalTypeTag::Int64 | CanonicalTypeTag::DateTimeUtc => "i64".to_string(),
            CanonicalTypeTag::UInt64 => "u64".to_string(),
            CanonicalTypeTag::Float64 => "f64".to_string(),
            CanonicalTypeTag::ObjectId => "lust_aot_rt::AotObjectId".to_string(),
            CanonicalTypeTag::Decimal128 => "lust_aot_rt::AotDecimal128".to_string(),
            CanonicalTypeTag::StringSlice => "alloc::string::String".to_string(),
            CanonicalTypeTag::BinarySlice => "alloc::vec::Vec<u8>".to_string(),
            CanonicalTypeTag::NestedStruct(_) => return self.rust_type(&sf.mir_ty, lt),
        };
        if is_opt {
            format!("Option<{base}>")
        } else {
            base
        }
    }

    fn rust_schema_field_default_expr(&self, sf: &SchemaFieldMeta) -> String {
        if sf.is_refined_array {
            return self.rust_default_expr(&sf.mir_ty);
        }
        if sf.is_optional || sf.is_nullable {
            return "None".to_string();
        }
        match sf.tag {
            CanonicalTypeTag::Bool => "false".to_string(),
            CanonicalTypeTag::Int64 | CanonicalTypeTag::DateTimeUtc => "0_i64".to_string(),
            CanonicalTypeTag::UInt64 => "0_u64".to_string(),
            CanonicalTypeTag::Float64 => "0.0_f64".to_string(),
            CanonicalTypeTag::ObjectId => "lust_aot_rt::AotObjectId::default()".to_string(),
            CanonicalTypeTag::Decimal128 => "lust_aot_rt::AotDecimal128::default()".to_string(),
            CanonicalTypeTag::StringSlice => "alloc::string::String::new()".to_string(),
            CanonicalTypeTag::BinarySlice => "alloc::vec::Vec::new()".to_string(),
            CanonicalTypeTag::NestedStruct(_) => self.rust_default_expr(&sf.mir_ty),
        }
    }

    fn rust_default_expr(&self, ty: &MirType) -> String {
        match ty {
            MirType::Unit => "()".to_string(),
            MirType::Bool => "false".to_string(),
            MirType::Int => "0_i64".to_string(),
            MirType::Float => "0.0_f64".to_string(),
            MirType::String => "alloc::string::String::new()".to_string(),
            MirType::Array(_) => "alloc::vec::Vec::new_in(__alloc)".to_string(),
            MirType::Map(_, _) => "lust_aot_rt::AotMap::new()".to_string(),
            MirType::Option(_) => "None".to_string(),
            MirType::Struct(name) => {
                let sname = rust_struct_name(name);
                if let Some(meta) = self.structs.get(name) {
                    if meta.derives_default {
                        format!("{sname}::default()")
                    } else {
                        let mut field_inits = Vec::with_capacity(meta.schema_fields.len());
                        for (i, sf) in meta.schema_fields.iter().enumerate() {
                            field_inits.push(format!(
                                "field_{i}: {}",
                                self.rust_schema_field_default_expr(sf)
                            ));
                        }
                        format!("{sname} {{ {} }}", field_inits.join(", "))
                    }
                } else {
                    format!("{sname}::default()")
                }
            }
            MirType::Enum(name) => {
                if let Some(emeta) = self.enums.get(name) {
                    let ename = rust_enum_name(name);
                    if emeta.derives_default {
                        format!("{ename}::default()")
                    } else if let Some((vname, vfields)) = emeta.variants.first() {
                        let rv = sanitize_ident(vname);
                        if vfields.is_empty() {
                            format!("{ename}::{rv}")
                        } else {
                            let inits: Vec<String> = vfields
                                .iter()
                                .map(|ft| self.rust_default_expr(ft))
                                .collect();
                            format!("{ename}::{rv}({})", inits.join(", "))
                        }
                    } else {
                        format!("{ename}::default()")
                    }
                } else {
                    "lust_aot_rt::AotValue::Nil".to_string()
                }
            }
            MirType::Tuple(elems) => {
                let parts: Vec<String> =
                    elems.iter().map(|e| self.rust_default_expr(e)).collect();
                if parts.is_empty() {
                    "()".to_string()
                } else if parts.len() == 1 {
                    format!("({},)", parts[0])
                } else {
                    format!("({})", parts.join(", "))
                }
            }
            MirType::Function {
                params,
                return_type,
            } => {
                let needs_lt = self.type_needs_lifetime(ty);
                let mut p_strs = Vec::with_capacity(params.len() + 2);
                p_strs.push("_: &mut lust_aot_rt::AotCtx".to_string());
                if needs_lt {
                    p_strs.push("_: &'a lust_aot_rt::LustBudgetAlloc".to_string());
                } else {
                    p_strs.push("_: &lust_aot_rt::LustBudgetAlloc".to_string());
                }
                for p in params {
                    p_strs.push(format!("_: {}", self.rust_type(p, "'a")));
                }
                let r = self.rust_type(return_type, "'a");
                let fn_ty = self.rust_type(ty, "'a");
                format!(
                    "(alloc::rc::Rc::new(|{}| -> Result<{r}, lust_aot_rt::AotError> {{ Err(lust_aot_rt::AotError::UserError(alloc::string::String::from(\"uninitialized closure\"))) }}) as {fn_ty})",
                    p_strs.join(", ")
                )
            }
            _ => "lust_aot_rt::AotValue::Nil".to_string(),
        }
    }
}

fn analyze_structs(module: &MirModule) -> BTreeMap<String, StructMeta> {
    let mut type_id_to_name: HashMap<u64, String> = HashMap::new();
    for layout in &module.struct_layouts {
        type_id_to_name.insert(layout.type_id, layout.type_name.clone());
    }

    let mut raw_fields: BTreeMap<String, Vec<(String, MirType)>> = BTreeMap::new();
    let mut raw_schema_fields: BTreeMap<String, Vec<SchemaFieldMeta>> = BTreeMap::new();
    let mut logical_layouts: BTreeMap<String, LogicalLayoutV1> = BTreeMap::new();
    let mut type_ids: BTreeMap<String, u64> = BTreeMap::new();

    for layout in &module.struct_layouts {
        let logical = layout.to_logical_layout();
        let mut fields = Vec::with_capacity(logical.fields.len());
        let mut sfields = Vec::with_capacity(logical.fields.len());
        for f in &logical.fields {
            let mut nested_struct_name = None;
            let base_ty = match f.tag {
                CanonicalTypeTag::Bool => MirType::Bool,
                CanonicalTypeTag::Int64
                | CanonicalTypeTag::UInt64
                | CanonicalTypeTag::DateTimeUtc
                | CanonicalTypeTag::ObjectId
                | CanonicalTypeTag::Decimal128 => MirType::Int,
                CanonicalTypeTag::Float64 => MirType::Float,
                CanonicalTypeTag::StringSlice | CanonicalTypeTag::BinarySlice => MirType::String,
                CanonicalTypeTag::NestedStruct(id) => {
                    if let Some(target_name) = type_id_to_name.get(&id) {
                        nested_struct_name = Some(target_name.clone());
                        MirType::Struct(target_name.clone())
                    } else {
                        MirType::Int
                    }
                }
            };
            let field_ty = if f.is_optional || f.is_nullable {
                MirType::Option(alloc::boxed::Box::new(base_ty))
            } else {
                base_ty
            };
            fields.push((f.name.clone(), field_ty.clone()));
            sfields.push(SchemaFieldMeta {
                name: f.name.clone(),
                tag: f.tag,
                mir_ty: field_ty,
                is_nullable: f.is_nullable,
                is_optional: f.is_optional,
                is_refined_array: false,
                nested_struct_name,
            });
        }
        type_ids.insert(layout.type_name.clone(), layout.type_id);
        logical_layouts.insert(layout.type_name.clone(), logical);
        raw_fields.insert(layout.type_name.clone(), fields);
        raw_schema_fields.insert(layout.type_name.clone(), sfields);
    }

    // Refine struct field types from Rvalue::Aggregate(MirType::Struct(name), ops) in functions
    // (catches Array<T> fields which CanonicalTypeTag defaulted to StringSlice).
    for func in &module.functions {
        for block in &func.blocks {
            for stmt in &block.statements {
                if let Statement::Assign(_, Rvalue::Aggregate(MirType::Struct(sname), ops), _) =
                    stmt
                    && let Some(fields) = raw_fields.get_mut(sname)
                {
                    let mut sfields = raw_schema_fields.get_mut(sname);
                    for (idx, op) in ops.iter().enumerate() {
                        if let Some((_, fty)) = fields.get_mut(idx) {
                            let op_ty = operand_mir_type(op, func);
                            if matches!(op_ty, MirType::Array(_))
                                && matches!(fty, MirType::String)
                            {
                                *fty = op_ty.clone();
                                if let Some(sf) = sfields.as_deref_mut().and_then(|v| v.get_mut(idx)) {
                                    sf.mir_ty = op_ty;
                                    sf.is_refined_array = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Compute struct dependency graph and cycle detection
    let mut deps: HashMap<String, HashSet<String>> = HashMap::new();
    for (sname, fields) in &raw_fields {
        let set = deps.entry(sname.clone()).or_default();
        for (_, fty) in fields {
            collect_struct_deps(fty, set);
        }
    }

    let mut is_cyclic: HashMap<String, bool> = HashMap::new();
    for sname in raw_fields.keys() {
        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        if let Some(neighbors) = deps.get(sname) {
            for n in neighbors {
                stack.push(n.clone());
            }
        }
        let mut cyclic = false;
        while let Some(cur) = stack.pop() {
            if &cur == sname {
                cyclic = true;
                break;
            }
            if visited.insert(cur.clone())
                && let Some(nexts) = deps.get(&cur)
            {
                for n in nexts {
                    stack.push(n.clone());
                }
            }
        }
        is_cyclic.insert(sname.clone(), cyclic);
    }

    // Fixed-point for needs_arena_lt, is_copy, derives_default
    let mut needs_arena_lt: HashMap<String, bool> = HashMap::new();
    let mut is_copy: HashMap<String, bool> = HashMap::new();
    let mut derives_default: HashMap<String, bool> = HashMap::new();

    for (sname, fields) in &raw_fields {
        let has_arr = fields.iter().any(|(_, t)| type_contains_array(t));
        needs_arena_lt.insert(sname.clone(), is_cyclic[sname] || has_arr);
        is_copy.insert(sname.clone(), !is_cyclic[sname] && !has_arr);
        derives_default.insert(sname.clone(), !has_arr);
    }

    loop {
        let mut changed = false;
        for (sname, fields) in &raw_fields {
            let mut cur_lt = needs_arena_lt[sname];
            let mut cur_copy = is_copy[sname];
            let mut cur_def = derives_default[sname];
            for (_, fty) in fields {
                if field_needs_lt(fty, &needs_arena_lt, &is_cyclic) && !cur_lt {
                    cur_lt = true;
                }
                if !field_is_copy(fty, &is_copy, &is_cyclic) && cur_copy {
                    cur_copy = false;
                }
                if !field_derives_default(fty, &derives_default) && cur_def {
                    cur_def = false;
                }
            }
            if cur_lt != needs_arena_lt[sname] {
                needs_arena_lt.insert(sname.clone(), cur_lt);
                changed = true;
            }
            if cur_copy != is_copy[sname] {
                is_copy.insert(sname.clone(), cur_copy);
                changed = true;
            }
            if cur_def != derives_default[sname] {
                derives_default.insert(sname.clone(), cur_def);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Bottom-up fixed point for `has_schema_decoders` and `resolved_layout` / `has_exact_repr_c_layout`.
    let mut has_schema_decoders: HashMap<String, bool> = HashMap::new();
    let mut resolved_layouts: HashMap<String, Option<LogicalLayoutV1>> = HashMap::new();
    let mut exact_repr_c: HashMap<String, bool> = HashMap::new();
    let mut resolved_sizes_by_id: HashMap<u64, (usize, usize)> = HashMap::new();

    for sname in raw_fields.keys() {
        has_schema_decoders.insert(sname.clone(), false);
        resolved_layouts.insert(sname.clone(), None);
        exact_repr_c.insert(sname.clone(), false);
    }

    loop {
        let mut changed = false;
        for (sname, sfields) in &raw_schema_fields {
            let has_refined_arr = sfields.iter().any(|sf| sf.is_refined_array);
            let eligible_base = !is_cyclic[sname]
                && !needs_arena_lt[sname]
                && derives_default[sname]
                && !has_refined_arr;

            if eligible_base && !has_schema_decoders[sname] {
                let nested_ok = sfields.iter().all(|sf| match sf.tag {
                    CanonicalTypeTag::NestedStruct(_) => sf
                        .nested_struct_name
                        .as_deref()
                        .and_then(|n| has_schema_decoders.get(n))
                        .copied()
                        .unwrap_or(false),
                    _ => true,
                });
                if nested_ok {
                    has_schema_decoders.insert(sname.clone(), true);
                    changed = true;
                }
            }

            if eligible_base && !exact_repr_c[sname] {
                let base_layout = &logical_layouts[sname];
                let has_nested = sfields
                    .iter()
                    .any(|sf| matches!(sf.tag, CanonicalTypeTag::NestedStruct(_)));
                let candidate = if !has_nested {
                    Some(base_layout.clone())
                } else {
                    let all_nested_exact = sfields.iter().all(|sf| match sf.tag {
                        CanonicalTypeTag::NestedStruct(_) => sf
                            .nested_struct_name
                            .as_deref()
                            .and_then(|n| exact_repr_c.get(n))
                            .copied()
                            .unwrap_or(false),
                        _ => true,
                    });
                    if all_nested_exact {
                        base_layout.resolve_nested_layouts(&|id| {
                            resolved_sizes_by_id.get(&id).copied()
                        })
                    } else {
                        None
                    }
                };
                if let Some(resolved) = candidate {
                    let is_exact = resolved.is_rust_repr_c_exact();
                    if is_exact {
                        resolved_sizes_by_id
                            .insert(type_ids[sname], (resolved.size_bytes, resolved.align_bytes));
                        exact_repr_c.insert(sname.clone(), true);
                        resolved_layouts.insert(sname.clone(), Some(resolved));
                        changed = true;
                    } else if resolved_layouts[sname].is_none() {
                        resolved_layouts.insert(sname.clone(), Some(resolved));
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    let mut out = BTreeMap::new();
    for (name, fields) in raw_fields {
        let schema_fields = raw_schema_fields.remove(&name).unwrap_or_default();
        let logical_layout = logical_layouts.remove(&name).unwrap();
        let resolved_layout = resolved_layouts.remove(&name).unwrap_or(None);
        out.insert(
            name.clone(),
            StructMeta {
                schema_fields,
                logical_layout,
                resolved_layout,
                has_exact_repr_c_layout: exact_repr_c[&name],
                has_schema_decoders: has_schema_decoders[&name],
                is_cyclic: is_cyclic[&name],
                needs_arena_lt: needs_arena_lt[&name],
                is_copy: is_copy[&name],
                derives_default: derives_default[&name],
                name,
                fields,
            },
        );
    }
    out
}

fn collect_struct_deps(ty: &MirType, out: &mut HashSet<String>) {
    match ty {
        MirType::Struct(name) => {
            out.insert(name.clone());
        }
        MirType::Option(inner) | MirType::Array(inner) => collect_struct_deps(inner, out),
        MirType::Tuple(elems) => {
            for e in elems {
                collect_struct_deps(e, out);
            }
        }
        _ => {}
    }
}

fn type_contains_array(ty: &MirType) -> bool {
    match ty {
        MirType::Array(_) => true,
        MirType::Option(inner) => type_contains_array(inner),
        MirType::Tuple(elems) => elems.iter().any(type_contains_array),
        _ => false,
    }
}

fn field_needs_lt(
    ty: &MirType,
    needs_lt: &HashMap<String, bool>,
    is_cyclic: &HashMap<String, bool>,
) -> bool {
    match ty {
        MirType::Array(_) => true,
        MirType::Struct(name) => *needs_lt.get(name).unwrap_or(&false),
        MirType::Option(inner) => match inner.as_ref() {
            MirType::Struct(name) => {
                *is_cyclic.get(name).unwrap_or(&false) || *needs_lt.get(name).unwrap_or(&false)
            }
            other => field_needs_lt(other, needs_lt, is_cyclic),
        },
        MirType::Tuple(elems) => elems.iter().any(|e| field_needs_lt(e, needs_lt, is_cyclic)),
        _ => false,
    }
}

fn field_is_copy(
    ty: &MirType,
    is_copy: &HashMap<String, bool>,
    is_cyclic: &HashMap<String, bool>,
) -> bool {
    match ty {
        MirType::Unit | MirType::Bool | MirType::Int | MirType::Float => true,
        MirType::Struct(name) => *is_copy.get(name).unwrap_or(&false),
        MirType::Option(inner) => match inner.as_ref() {
            MirType::Struct(name) => {
                !*is_cyclic.get(name).unwrap_or(&false) && *is_copy.get(name).unwrap_or(&false)
            }
            other => field_is_copy(other, is_copy, is_cyclic),
        },
        MirType::Tuple(elems) => elems.iter().all(|e| field_is_copy(e, is_copy, is_cyclic)),
        _ => false,
    }
}

fn field_derives_default(ty: &MirType, derives_def: &HashMap<String, bool>) -> bool {
    match ty {
        MirType::Unit | MirType::Bool | MirType::Int | MirType::Float | MirType::String => true,
        MirType::Option(_) => true, // Option<T>::default() is always None regardless of T
        MirType::Map(_, _) => true, // AotMap::default()
        MirType::Array(_) => false, // Vec<T, &LustBudgetAlloc> is not Default
        MirType::Struct(name) => *derives_def.get(name).unwrap_or(&false),
        MirType::Tuple(elems) => elems.iter().all(|e| field_derives_default(e, derives_def)),
        _ => true,
    }
}

fn analyze_enums(
    module: &MirModule,
    structs: &BTreeMap<String, StructMeta>,
) -> BTreeMap<String, EnumMeta> {
    let mut raw: BTreeMap<String, Vec<(String, Vec<MirType>)>> = BTreeMap::new();
    for edef in &module.enum_defs {
        raw.insert(edef.name.clone(), edef.variants.clone());
    }

    let mut needs_lt: HashMap<String, bool> = HashMap::new();
    let mut is_copy: HashMap<String, bool> = HashMap::new();
    let mut derives_def: HashMap<String, bool> = HashMap::new();

    for ename in raw.keys() {
        needs_lt.insert(ename.clone(), false);
        is_copy.insert(ename.clone(), true);
        derives_def.insert(ename.clone(), true);
    }

    loop {
        let mut changed = false;
        for (ename, variants) in &raw {
            let mut cur_lt = needs_lt[ename];
            let mut cur_copy = is_copy[ename];
            let mut cur_def = derives_def[ename];
            for (vidx, (_, flds)) in variants.iter().enumerate() {
                for ft in flds {
                    if enum_field_needs_lt(ft, structs, &needs_lt) && !cur_lt {
                        cur_lt = true;
                    }
                    if !enum_field_is_copy(ft, structs, &is_copy) && cur_copy {
                        cur_copy = false;
                    }
                    if vidx == 0 && !enum_field_derives_default(ft, structs, &derives_def) && cur_def
                    {
                        cur_def = false;
                    }
                }
            }
            if cur_lt != needs_lt[ename] {
                needs_lt.insert(ename.clone(), cur_lt);
                changed = true;
            }
            if cur_copy != is_copy[ename] {
                is_copy.insert(ename.clone(), cur_copy);
                changed = true;
            }
            if cur_def != derives_def[ename] {
                derives_def.insert(ename.clone(), cur_def);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut out = BTreeMap::new();
    for (name, variants) in raw {
        out.insert(
            name.clone(),
            EnumMeta {
                needs_arena_lt: needs_lt[&name],
                is_copy: is_copy[&name],
                derives_default: derives_def[&name],
                name,
                variants,
            },
        );
    }
    out
}

fn enum_field_needs_lt(
    ty: &MirType,
    structs: &BTreeMap<String, StructMeta>,
    enum_lt: &HashMap<String, bool>,
) -> bool {
    match ty {
        MirType::Array(_) => true,
        MirType::Map(_, v) => enum_field_needs_lt(v, structs, enum_lt),
        MirType::Struct(s) => structs.get(s).is_some_and(|m| m.needs_arena_lt),
        MirType::Enum(e) => *enum_lt.get(e).unwrap_or(&false),
        MirType::Option(inner) => match inner.as_ref() {
            MirType::Struct(s) => structs
                .get(s)
                .is_some_and(|m| m.is_cyclic || m.needs_arena_lt),
            other => enum_field_needs_lt(other, structs, enum_lt),
        },
        MirType::Tuple(elems) => elems
            .iter()
            .any(|e| enum_field_needs_lt(e, structs, enum_lt)),
        _ => false,
    }
}

fn enum_field_is_copy(
    ty: &MirType,
    structs: &BTreeMap<String, StructMeta>,
    enum_copy: &HashMap<String, bool>,
) -> bool {
    match ty {
        MirType::Unit | MirType::Bool | MirType::Int | MirType::Float => true,
        MirType::Struct(s) => structs.get(s).is_some_and(|m| m.is_copy),
        MirType::Enum(e) => *enum_copy.get(e).unwrap_or(&false),
        MirType::Option(inner) => match inner.as_ref() {
            MirType::Struct(s) => structs
                .get(s)
                .is_some_and(|m| m.is_copy && !m.is_cyclic),
            other => enum_field_is_copy(other, structs, enum_copy),
        },
        MirType::Tuple(elems) => elems
            .iter()
            .all(|e| enum_field_is_copy(e, structs, enum_copy)),
        _ => false,
    }
}

fn enum_field_derives_default(
    ty: &MirType,
    structs: &BTreeMap<String, StructMeta>,
    enum_def: &HashMap<String, bool>,
) -> bool {
    match ty {
        MirType::Unit | MirType::Bool | MirType::Int | MirType::Float | MirType::String => true,
        MirType::Option(_) | MirType::Map(_, _) => true,
        MirType::Array(_) | MirType::Function { .. } => false,
        MirType::Struct(s) => structs.get(s).is_some_and(|m| m.derives_default),
        MirType::Enum(e) => *enum_def.get(e).unwrap_or(&true),
        MirType::Tuple(elems) => elems
            .iter()
            .all(|e| enum_field_derives_default(e, structs, enum_def)),
        _ => true,
    }
}

fn type_needs_lt_raw(
    ty: &MirType,
    structs: &BTreeMap<String, StructMeta>,
    enums: &BTreeMap<String, EnumMeta>,
) -> bool {
    match ty {
        MirType::Array(_) => true,
        MirType::Map(_, v) => type_needs_lt_raw(v, structs, enums),
        MirType::Struct(s) => structs.get(s).is_some_and(|m| m.needs_arena_lt),
        MirType::Enum(e) => enums.get(e).is_some_and(|m| m.needs_arena_lt),
        MirType::Option(inner) => match inner.as_ref() {
            MirType::Struct(s) => structs
                .get(s)
                .is_some_and(|m| m.is_cyclic || m.needs_arena_lt),
            other => type_needs_lt_raw(other, structs, enums),
        },
        MirType::Tuple(elems) => elems
            .iter()
            .any(|e| type_needs_lt_raw(e, structs, enums)),
        MirType::Function {
            params,
            return_type,
        } => {
            params.iter().any(|p| type_needs_lt_raw(p, structs, enums))
                || type_needs_lt_raw(return_type, structs, enums)
        }
        _ => false,
    }
}

/// Returns true if `target` is written or mutated anywhere in `func` (including after initialization).
fn local_is_mutated_in_fn(func: &MirFunction, target: LocalId) -> bool {
    let mut assign_count = 0usize;
    let is_param = func.params.contains(&target);
    for block in &func.blocks {
        for stmt in &block.statements {
            if let Statement::Assign(place, _, _) = stmt
                && place.root_local() == target
            {
                match place {
                    Place::Local(_) => {
                        assign_count += 1;
                        if is_param || assign_count > 1 || block.is_cleanup {
                            return true;
                        }
                    }
                    Place::Field(_, _) | Place::Index(_, _) => return true,
                }
            }
        }
        if let Some(Terminator::Call {
            callee,
            args,
            destination,
            ..
        }) = &block.terminator
        {
            if destination.root_local() == target {
                assign_count += 1;
                if is_param || assign_count > 1 || block.is_cleanup {
                    return true;
                }
            }
            if matches!(
                callee.as_str(),
                "array.push"
                    | "table.insert"
                    | "array.set"
                    | "map.set"
                    | "table.set"
                    | "map.delete"
                    | "map.remove"
                    | "map.clear"
            ) && args
                .first()
                .and_then(operand_root_local)
                == Some(target)
            {
                return true;
            }
        }
    }
    false
}

/// Collects all non-cleanup-internal locals referenced inside any `cleanup_blocks` subgraph.
fn collect_cleanup_captured_locals(func: &MirFunction) -> HashSet<LocalId> {
    let mut cleanup_bbs: HashSet<BlockId> = HashSet::new();
    for entry in &func.cleanup_blocks {
        let mut q = VecDeque::new();
        if entry.block.0 < func.blocks.len() && cleanup_bbs.insert(entry.block) {
            q.push_back(entry.block);
        }
        while let Some(bb) = q.pop_front() {
            if let Some(term) = &func.blocks[bb.0].terminator {
                for s in term.successors() {
                    let sb = BlockId(s);
                    if sb.0 < func.blocks.len() && cleanup_bbs.insert(sb) {
                        q.push_back(sb);
                    }
                }
            }
        }
    }

    // Locals referenced in main (non-cleanup) blocks or params
    let mut main_locals: HashSet<LocalId> = func.params.iter().copied().collect();
    for block in &func.blocks {
        if cleanup_bbs.contains(&block.id) || block.is_cleanup {
            continue;
        }
        collect_block_locals(block, &mut main_locals);
    }

    // Locals referenced inside cleanup blocks
    let mut cleanup_locals: HashSet<LocalId> = HashSet::new();
    for &bb in &cleanup_bbs {
        collect_block_locals(&func.blocks[bb.0], &mut cleanup_locals);
    }

    // Captured = locals appearing in BOTH main body/params AND cleanup blocks
    cleanup_locals
        .into_iter()
        .filter(|loc| main_locals.contains(loc))
        .collect()
}

fn collect_block_locals(block: &BasicBlock, out: &mut HashSet<LocalId>) {
    for stmt in &block.statements {
        if let Statement::Assign(place, rval, _) = stmt {
            out.insert(place.root_local());
            let mut tmp = HashSet::new();
            place.collect_uses(&mut tmp);
            rval.collect_uses(&mut tmp);
            for id in tmp {
                out.insert(LocalId(id));
            }
        }
    }
    if let Some(term) = &block.terminator {
        let mut tmp = HashSet::new();
        match term {
            Terminator::Cond { cond, .. } => cond.collect_uses(&mut tmp),
            Terminator::Switch { discr, .. } => discr.collect_uses(&mut tmp),
            Terminator::Call {
                args, destination, ..
            } => {
                out.insert(destination.root_local());
                destination.collect_uses(&mut tmp);
                for a in args {
                    a.collect_uses(&mut tmp);
                }
            }
            Terminator::Return(Some(op), _) | Terminator::Raise(op, _) => {
                op.collect_uses(&mut tmp);
            }
            _ => {}
        }
        for id in tmp {
            out.insert(LocalId(id));
        }
    }
}

/// Detects locally aliased `Array` and mutable `Struct` locals (`NG-W12-4B`, Hazard 12)
/// that must be wrapped in `Rc<RefCell<...>>` to preserve Lust reference semantics
/// without causing Rust borrow-checker conflicts (`E0499`/`E0502`).
fn detect_aliased_container_locals(
    func: &MirFunction,
    structs: &BTreeMap<String, StructMeta>,
    module_fn_names: &HashSet<String>,
) -> HashSet<LocalId> {
    let use_counts = count_local_uses(func);
    let mut def_counts: HashMap<LocalId, usize> = HashMap::new();
    let mut aggregate_or_call_def: HashSet<LocalId> = HashSet::new();

    for block in &func.blocks {
        for stmt in &block.statements {
            if let Statement::Assign(Place::Local(dst), rval, _) = stmt {
                *def_counts.entry(*dst).or_insert(0) += 1;
                if matches!(rval, Rvalue::Aggregate(_, _)) {
                    aggregate_or_call_def.insert(*dst);
                }
            }
        }
        if let Some(Terminator::Call {
            destination: Place::Local(dst),
            ..
        }) = &block.terminator
        {
            *def_counts.entry(*dst).or_insert(0) += 1;
            aggregate_or_call_def.insert(*dst);
        }
    }

    let param_set: HashSet<LocalId> = func.params.iter().copied().collect();

    // Helper: is `loc` an unnamed single-use rvalue materialization temp (e.g. `_tmp = [1, 2]; _a = Move(_tmp)`)?
    let is_pure_init_temp = |loc: LocalId| -> bool {
        !param_set.contains(&loc)
            && func.locals[loc.0].name.is_none()
            && def_counts.get(&loc).copied().unwrap_or(0) == 1
            && aggregate_or_call_def.contains(&loc)
            && use_counts.get(&loc).copied().unwrap_or(0) <= 1
    };

    // Helper: is `loc` a MirInliner-synthesized parameter alias (e.g. `Counter:bump_self`, `count_even_arr`)?
    let is_inlined_callee_param = |loc: LocalId| -> bool {
        if let Some(name) = &func.locals[loc.0].name {
            module_fn_names
                .iter()
                .any(|fn_n| name.len() > fn_n.len() + 1 && name.starts_with(fn_n.as_str()) && name.as_bytes()[fn_n.len()] == b'_')
        } else {
            false
        }
    };

    // Count how many distinct aliases point to each `src`
    let mut alias_edges: Vec<(LocalId, LocalId)> = Vec::new();
    let mut src_inlined_alias_count: HashMap<LocalId, usize> = HashMap::new();
    for block in &func.blocks {
        for stmt in &block.statements {
            if let Statement::Assign(
                Place::Local(dst),
                Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))),
                _,
            ) = stmt
                && dst != src
            {
                let ty = &func.locals[dst.0].ty;
                let is_trackable = match ty {
                    MirType::Array(_) => true,
                    MirType::Struct(sn) => !structs.get(sn).is_some_and(|s| s.is_cyclic),
                    _ => false,
                };
                if !is_trackable || is_pure_init_temp(*src) {
                    continue;
                }
                if *dst == func.return_local {
                    continue;
                }
                alias_edges.push((*dst, *src));
                if is_inlined_callee_param(*dst) {
                    *src_inlined_alias_count.entry(*src).or_insert(0) += 1;
                }
            }
        }
    }

    // Build undirected alias graph ONLY for genuine user aliasing or unsafe multi-alias cases
    let mut adj: HashMap<LocalId, BTreeSet<LocalId>> = HashMap::new();
    for &(dst, src) in &alias_edges {
        let dst_defs = def_counts.get(&dst).copied().unwrap_or(0);
        let src_defs = def_counts.get(&src).copied().unwrap_or(0);
        let safe_inlined_single_alias = is_inlined_callee_param(dst)
            && !param_set.contains(&dst)
            && dst_defs == 1
            && src_defs <= 1
            && src_inlined_alias_count.get(&src).copied().unwrap_or(0) == 1
            && !param_escapes_by_move(func, dst);
        if safe_inlined_single_alias {
            continue;
        }
        // Only promote Struct aliases if the struct is mutated (or passed to a mutating call);
        // always promote Array aliases (since arrays are mutable reference containers in Lust).
        let is_array = matches!(func.locals[dst.0].ty, MirType::Array(_));
        let is_mutated = local_is_mutated_in_fn(func, dst) || local_is_mutated_in_fn(func, src);
        if is_array || is_mutated {
            adj.entry(dst).or_default().insert(src);
            adj.entry(src).or_default().insert(dst);
        }
    }

    // Return all locals in non-trivial connected components, then close transitively over ALL
    // alias_edges (including inlined callee params copying an already-aliased local so they share
    // the same Rc<RefCell> handle instead of cloning into a detached Vec!).
    let mut aliased = HashSet::new();
    for (&node, neighbors) in &adj {
        if !neighbors.is_empty() {
            aliased.insert(node);
            for &n in neighbors {
                aliased.insert(n);
            }
        }
    }
    loop {
        let mut changed = false;
        for &(dst, src) in &alias_edges {
            if aliased.contains(&src) && aliased.insert(dst) {
                changed = true;
            }
            if aliased.contains(&dst) && aliased.insert(src) {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    aliased
}

fn compute_fn_param_modes(
    func: &MirFunction,
    ro_mask: Option<&Vec<bool>>,
    structs: &BTreeMap<String, StructMeta>,
    _enums: &BTreeMap<String, EnumMeta>,
) -> Vec<ParamMode> {
    let mut modes = Vec::with_capacity(func.params.len());
    for (pi, &param_id) in func.params.iter().enumerate() {
        let ty = func
            .locals
            .get(param_id.0)
            .map(|d| &d.ty)
            .unwrap_or(&MirType::Int);
        let is_ro = ro_mask.and_then(|m| m.get(pi).copied()).unwrap_or(false);
        let is_returned_or_stored = param_escapes_by_move(func, param_id);
        let mode = match ty {
            MirType::Unit | MirType::Bool | MirType::Int | MirType::Float => ParamMode::ByValue,
            MirType::Array(_) => {
                if is_ro && !is_returned_or_stored {
                    ParamMode::RefSlice
                } else if !is_returned_or_stored {
                    ParamMode::MutVec
                } else {
                    ParamMode::ByValue
                }
            }
            MirType::Struct(sname) => {
                let is_copy = structs.get(sname).is_some_and(|s| s.is_copy);
                if is_ro {
                    if is_copy {
                        ParamMode::ByValue
                    } else {
                        ParamMode::RefStruct
                    }
                } else if !is_returned_or_stored {
                    ParamMode::MutRefStruct
                } else {
                    ParamMode::ByValue
                }
            }
            MirType::String => {
                if is_ro && !is_returned_or_stored && !param_is_reassigned(func, param_id) {
                    ParamMode::RefStr
                } else {
                    ParamMode::ByValue
                }
            }
            _ => ParamMode::ByValue,
        };
        modes.push(mode);
    }
    modes
}

fn param_is_reassigned(func: &MirFunction, param_id: LocalId) -> bool {
    for block in &func.blocks {
        for stmt in &block.statements {
            if let Statement::Assign(Place::Local(dst), _, _) = stmt
                && *dst == param_id
            {
                return true;
            }
        }
        if let Some(Terminator::Call {
            destination: Place::Local(dst),
            ..
        }) = &block.terminator
            && *dst == param_id
        {
            return true;
        }
    }
    false
}

fn param_escapes_by_move(func: &MirFunction, param_id: LocalId) -> bool {
    for block in &func.blocks {
        for stmt in &block.statements {
            if let Statement::Assign(_, rval, _) = stmt {
                match rval {
                    Rvalue::Aggregate(_, ops) => {
                        for op in ops {
                            if operand_root_local(op) == Some(param_id) {
                                return true;
                            }
                        }
                    }
                    Rvalue::Cast(op, MirType::Option(_)) => {
                        if operand_root_local(op) == Some(param_id) {
                            return true;
                        }
                    }
                    _ => {}
                }
            }
        }
        if let Some(Terminator::Return(Some(op), _)) = &block.terminator
            && operand_root_local(op) == Some(param_id)
        {
            return true;
        }
    }
    false
}

struct LocalFnRequirements {
    can_fail: bool,
    needs_ctx: bool,
    needs_alloc: bool,
    needs_arena: bool,
    allocates_string: bool,
    user_callees: Vec<String>,
}

fn scan_function_local_requirements(
    func: &MirFunction,
    opts: &RustEmitterOptions,
    structs: &BTreeMap<String, StructMeta>,
    enums: &BTreeMap<String, EnumMeta>,
    pmodes: &[ParamMode],
    lstorage: &[LocalStorage],
    module: &MirModule,
) -> LocalFnRequirements {
    let mut req = LocalFnRequirements {
        can_fail: false,
        needs_ctx: false,
        needs_alloc: false,
        needs_arena: false,
        allocates_string: false,
        user_callees: Vec::new(),
    };

    let user_fn_names: HashSet<&str> = module
        .functions
        .iter()
        .map(|f| f.signature.name.as_str())
        .collect();

    let closure_local_names: HashSet<&str> = func
        .locals
        .iter()
        .filter_map(|d| {
            if matches!(d.ty, MirType::Function { .. } | MirType::Unknown) {
                d.name.as_deref()
            } else {
                None
            }
        })
        .collect();

    // Check if any local is an owned Vec (needs __alloc for Vec::new_in) or Map/Enum with Vec
    for (idx, decl) in func.locals.iter().enumerate() {
        let loc_id = LocalId(idx);
        let is_ref_param = func
            .params
            .iter()
            .position(|&p| p == loc_id)
            .and_then(|pi| pmodes.get(pi))
            .is_some_and(|m| {
                matches!(
                    m,
                    ParamMode::RefSlice
                        | ParamMode::MutVec
                        | ParamMode::RefStruct
                        | ParamMode::MutRefStruct
                        | ParamMode::RefStr
                        | ParamMode::RcCell
                        | ParamMode::RcRefCell
                )
            });
        if is_ref_param {
            continue;
        }
        if matches!(decl.ty, MirType::Array(_)) {
            req.needs_alloc = true;
        }
        if let MirType::Struct(sname) = &decl.ty
            && let Some(meta) = structs.get(sname)
            && meta.fields.iter().any(|(_, t)| type_contains_array(t))
        {
            req.needs_alloc = true;
        }
        if let MirType::Enum(ename) = &decl.ty
            && let Some(emeta) = enums.get(ename)
            && !emeta.derives_default
        {
            req.needs_alloc = true;
        }
    }

    // Build proven_in_bounds set so we know which Place::Index reads are already guarded by IfArrayGetOk
    let tree = structure_mir_function(func);
    let mut folded_index_blocks: HashSet<(BlockId, LocalId)> = HashSet::new();
    collect_folded_array_get_bindings(&tree, func, &mut folded_index_blocks);

    for block in &func.blocks {
        for stmt in &block.statements {
            match stmt {
                Statement::ResourceCheck(ResourceKind::Gas, _, _) => {
                    if opts.gas_metering != GasMetering::None {
                        req.needs_ctx = true;
                        req.can_fail = true;
                    }
                }
                Statement::ResourceCheck(ResourceKind::Memory, _, _) => {
                    if opts.max_memory_bytes.is_some() {
                        req.needs_alloc = true;
                        req.can_fail = true;
                    }
                }
                Statement::Assign(place, rval, _) => {
                    if matches!(place, Place::Index(_, _)) {
                        // Checked index write can return IndexOutOfBounds (or Map::set)
                        req.can_fail = true;
                        if let Place::Index(base, _) = place
                            && matches!(place_type_raw(base, func), MirType::Map(_, _) | MirType::Unknown)
                        {
                            req.needs_alloc = true;
                            req.needs_ctx = true;
                        }
                    }
                    // Assigning to/from Unknown (AotValue) may coerce fallibly
                    if let Place::Local(dst) = place
                        && matches!(func.locals[dst.0].ty, MirType::Unknown)
                    {
                        req.can_fail = true;
                    }
                    match rval {
                        Rvalue::BinaryOp(op, lhs, rhs) => {
                            let l_ty = operand_mir_type(lhs, func);
                            let r_ty = operand_mir_type(rhs, func);
                            if matches!(l_ty, MirType::Unknown) || matches!(r_ty, MirType::Unknown) {
                                req.can_fail = true;
                                if *op == MirBinaryOp::Concat {
                                    req.needs_alloc = true;
                                    req.allocates_string = true;
                                }
                            } else {
                                let is_float = is_float_binop(lhs, rhs, func);
                                match op {
                                    MirBinaryOp::Div | MirBinaryOp::Mod if !is_float => {
                                        if !is_safe_const_int_divisor(rhs) {
                                            req.can_fail = true;
                                        }
                                    }
                                    MirBinaryOp::FloorDiv if !is_float => {
                                        req.can_fail = true;
                                    }
                                    MirBinaryOp::Pow if !is_float => {
                                        req.can_fail = true;
                                    }
                                    MirBinaryOp::Concat => {
                                        req.allocates_string = true;
                                        if opts.max_memory_bytes.is_some() {
                                            req.needs_alloc = true;
                                            req.can_fail = true;
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        Rvalue::UnaryOp(op, operand) => {
                            let ty = operand_mir_type(operand, func);
                            if matches!(ty, MirType::Unknown)
                                || (*op == MirUnaryOp::Len && matches!(ty, MirType::Map(_, _)))
                            {
                                req.can_fail = true;
                            }
                        }
                        Rvalue::Use(
                            Operand::Copy(Place::Index(base, _)) | Operand::Move(Place::Index(base, _)),
                        ) => {
                            let is_folded = matches!(place, Place::Local(dst) if folded_index_blocks.contains(&(block.id, *dst)));
                            let base_rc = lstorage.get(base.root_local().0).copied()
                                == Some(LocalStorage::RcRefCell);
                            if !is_folded || base_rc {
                                req.can_fail = true;
                            }
                        }
                        Rvalue::Use(op) => {
                            if matches!(operand_mir_type(op, func), MirType::Unknown) {
                                req.can_fail = true;
                            }
                        }
                        Rvalue::Aggregate(MirType::Array(_), _) => {
                            req.needs_alloc = true;
                            if opts.max_memory_bytes.is_some() {
                                req.can_fail = true;
                            }
                        }
                        Rvalue::Aggregate(MirType::Enum(_), _) => {}
                        Rvalue::Closure { func_name, .. } => {
                            // Propagate arena requirement only if lifted lambda uses arena;
                            // ctx/alloc are passed dynamically at closure call time.
                            if let Some(lf) = module
                                .functions
                                .iter()
                                .find(|f| f.signature.name == *func_name)
                            {
                                if lf
                                    .locals
                                    .iter()
                                    .any(|l| matches!(&l.ty, MirType::Struct(sn) if structs.get(sn).is_some_and(|s| s.is_cyclic)))
                                {
                                    req.needs_arena = true;
                                }
                            }
                        }
                        Rvalue::Cast(op, MirType::Option(inner)) => {
                            if let MirType::Struct(sname) = inner.as_ref()
                                && structs.get(sname).is_some_and(|s| s.is_cyclic)
                                && !matches!(op, Operand::Constant(MirConstant::Int(0)))
                            {
                                req.needs_arena = true;
                                if opts.max_memory_bytes.is_some() {
                                    req.needs_alloc = true;
                                    req.can_fail = true;
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Statement::StorageLive(_, _) | Statement::StorageDead(_, _) => {}
            }
        }

        if let Some(term) = &block.terminator {
            match term {
                Terminator::Call { callee, args, .. } => {
                    if parse_mir_struct_decode_callee(callee).is_some()
                        || callee.starts_with("__lust_mir_struct_decode_binary:")
                    {
                        req.needs_ctx = true;
                        req.can_fail = true;
                    } else if user_fn_names.contains(callee.as_str()) {
                        req.user_callees.push(callee.clone());
                    } else if closure_local_names.contains(callee.as_str()) {
                        req.needs_ctx = true;
                        req.needs_alloc = true;
                        req.can_fail = true;
                    } else {
                        match callee.as_str() {
                            "assert" | "error" => {
                                req.can_fail = true;
                            }
                            "array.push" | "table.insert" => {
                                if args
                                    .first()
                                    .is_some_and(|a| matches!(operand_mir_type(a, func), MirType::Unknown))
                                {
                                    req.needs_ctx = true;
                                    req.can_fail = true;
                                } else if opts.max_memory_bytes.is_some() {
                                    req.needs_alloc = true;
                                    req.can_fail = true;
                                }
                            }
                            "map.new" => {}
                            "map.set" | "table.set" | "map.delete" | "map.remove" | "map.keys"
                            | "map.values" => {
                                req.needs_alloc = true;
                                req.can_fail = true;
                                if args
                                    .first()
                                    .is_some_and(|a| matches!(operand_mir_type(a, func), MirType::Unknown))
                                {
                                    req.needs_ctx = true;
                                }
                            }
                            "map.get" | "table.get" | "map.has" | "map.len" | "map.is_empty"
                            | "map.clear" => {
                                req.can_fail = true;
                            }
                            "array.len" | "table.len" => {
                                if args.first().is_some_and(|a| {
                                    matches!(
                                        operand_mir_type(a, func),
                                        MirType::Map(_, _) | MirType::Unknown
                                    )
                                }) {
                                    req.can_fail = true;
                                }
                            }
                            "enum.tag" | "enum.payload" | "enum.new" | "enum.alloc" => {
                                req.can_fail = true;
                            }
                            "tostring" => {
                                req.allocates_string = true;
                                if opts.max_memory_bytes.is_some() {
                                    req.needs_alloc = true;
                                    req.can_fail = true;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                Terminator::Raise(_, _) | Terminator::Unreachable(_) => {
                    req.can_fail = true;
                }
                _ => {}
            }
        }
    }

    req
}

fn collect_folded_array_get_bindings(
    node: &StructuredNode,
    func: &MirFunction,
    out: &mut HashSet<(BlockId, LocalId)>,
) {
    match node {
        StructuredNode::IfArrayGetOk {
            binding_local,
            then_branch,
            else_branch,
            ..
        } => {
            if let Some(then_bb) = leading_block_of(then_branch) {
                out.insert((then_bb, *binding_local));
            }
            collect_folded_array_get_bindings(then_branch, func, out);
            if let Some(eb) = else_branch {
                collect_folded_array_get_bindings(eb, func, out);
            }
        }
        StructuredNode::Seq(items) => {
            for item in items {
                collect_folded_array_get_bindings(item, func, out);
            }
        }
        StructuredNode::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_folded_array_get_bindings(then_branch, func, out);
            if let Some(eb) = else_branch {
                collect_folded_array_get_bindings(eb, func, out);
            }
        }
        StructuredNode::Switch { cases, default, .. } => {
            for (_, c) in cases {
                collect_folded_array_get_bindings(c, func, out);
            }
            collect_folded_array_get_bindings(default, func, out);
        }
        StructuredNode::WhileLoop { body, .. }
        | StructuredNode::Loop { body, .. }
        | StructuredNode::LabeledBlock { body, .. } => {
            collect_folded_array_get_bindings(body, func, out);
        }
        _ => {}
    }
}

fn leading_block_of(node: &StructuredNode) -> Option<BlockId> {
    match node {
        StructuredNode::BasicBlock(bb) | StructuredNode::Return(bb, _) => Some(*bb),
        StructuredNode::If { cond_block, .. } => Some(*cond_block),
        StructuredNode::IfArrayGetOk { header_block, .. } => Some(*header_block),
        StructuredNode::Switch { discr_block, .. } => Some(*discr_block),
        StructuredNode::WhileLoop { header_block, .. } => Some(*header_block),
        StructuredNode::Seq(items) => items.first().and_then(leading_block_of),
        StructuredNode::Loop { body, .. } | StructuredNode::LabeledBlock { body, .. } => {
            leading_block_of(body)
        }
        StructuredNode::StateMachineFallback { entry, .. } => Some(*entry),
        StructuredNode::Break(_) | StructuredNode::Continue(_) => None,
    }
}

// ---------------------------------------------------------------------------
// File Header, Structs, and Main Entrypoint
// ---------------------------------------------------------------------------

fn emit_file_header(out: &mut String, opts: &RustEmitterOptions) {
    out.push_str("// Generated by Lust Next-Gen Rust AOT Compiler (NG-W12-4A + NG-W12-4B + NG-W12-4C)\n");
    if opts.no_std {
        out.push_str("#![no_std]\n");
    }
    out.push_str("#![allow(unused_variables, unused_mut, unused_assignments, dead_code, non_camel_case_types, non_snake_case, unreachable_code, clippy::all)]\n\n");
    out.push_str("extern crate alloc;\n");
    out.push_str("extern crate lust_aot_rt;\n\n");
}

fn emit_struct_definitions(out: &mut String, ctx: &ModuleEmitContext<'_>) {
    if ctx.structs.is_empty() {
        return;
    }
    out.push_str("// Struct & Schema Definitions (Tiers 2-3 + NG-W12-4C #[repr(C)] Layout Assertions & Decoders)\n");
    for meta in ctx.structs.values() {
        let sname = rust_struct_name(&meta.name);
        if !meta.is_cyclic {
            out.push_str("#[repr(C)]\n");
        }
        if meta.is_copy {
            out.push_str("#[derive(Clone, Copy, Debug, Default, PartialEq)]\n");
        } else if meta.is_cyclic {
            out.push_str("#[derive(Debug, Default)]\n");
        } else if meta.derives_default {
            out.push_str("#[derive(Clone, Debug, Default, PartialEq)]\n");
        } else {
            out.push_str("#[derive(Debug)]\n");
        }
        if meta.needs_arena_lt {
            let _ = writeln!(out, "pub struct {sname}<'a> {{");
        } else {
            let _ = writeln!(out, "pub struct {sname} {{");
        }
        for (i, sf) in meta.schema_fields.iter().enumerate() {
            let rust_ty = ctx.rust_schema_field_type(sf, "'a");
            let _ = writeln!(out, "    pub field_{i}: {rust_ty}, // {}", sf.name);
        }
        out.push_str("}\n\n");

        // Compile-time static layout verification against `LogicalLayoutV1` (`NG-W12-4C`, Gate-SCHEMA-09).
        if meta.has_exact_repr_c_layout {
            let layout = meta
                .resolved_layout
                .as_ref()
                .unwrap_or(&meta.logical_layout);
            for (i, f) in layout.fields.iter().enumerate() {
                let _ = writeln!(
                    out,
                    "const _: () = assert!(core::mem::offset_of!({sname}, field_{i}) == {});",
                    f.byte_offset
                );
            }
            let _ = writeln!(
                out,
                "const _: () = assert!(core::mem::size_of::<{sname}>() == {});",
                layout.size_bytes
            );
            let _ = writeln!(
                out,
                "const _: () = assert!(core::mem::align_of::<{sname}>() == {});\n",
                layout.align_bytes
            );
        }

        if meta.has_schema_decoders {
            emit_schema_decoders(out, meta);
        }
    }
}

fn emit_schema_decoders(out: &mut String, meta: &StructMeta) {
    let sname = rust_struct_name(&meta.name);
    let layout = meta
        .resolved_layout
        .as_ref()
        .unwrap_or(&meta.logical_layout);
    let field_count = meta.schema_fields.len();

    let _ = writeln!(out, "impl {sname} {{");
    let _ = writeln!(
        out,
        "    pub const LOGICAL_SIZE_BYTES: usize = {};",
        layout.size_bytes
    );
    let _ = writeln!(
        out,
        "    pub const LOGICAL_ALIGN_BYTES: usize = {};\n",
        layout.align_bytes
    );

    // 1. Zero-lookup, unaligned-safe binary decoder (`decode_binary` + `decode_binary_at`)
    if meta.has_exact_repr_c_layout {
        out.push_str("    /// Zero-lookup, alignment-agnostic binary schema decoder (`NG-W12-4C`).\n");
        out.push_str("    /// Reads every field at its constant `LogicalLayoutV1` byte offset via `from_le_bytes`.\n");
        out.push_str("    #[inline]\n");
        out.push_str(
            "    pub fn decode_binary(raw: &[u8], ctx: &mut lust_aot_rt::AotCtx) -> Result<Self, lust_aot_rt::AotError> {\n",
        );
        let _ = writeln!(
            out,
            "        lust_aot_rt::schema::check_binary_len(raw, {})?;",
            layout.size_bytes
        );
        out.push_str("        Self::decode_binary_at(raw, 0, ctx)\n");
        out.push_str("    }\n\n");

        out.push_str("    #[inline]\n");
        out.push_str(
            "    pub fn decode_binary_at(raw: &[u8], base_offset: usize, ctx: &mut lust_aot_rt::AotCtx) -> Result<Self, lust_aot_rt::AotError> {\n",
        );
        let _ = writeln!(
            out,
            "        let __end = lust_aot_rt::schema::binary_offset(base_offset, {})?;",
            layout.size_bytes
        );
        out.push_str("        lust_aot_rt::schema::check_binary_len(raw, __end)?;\n");
        out.push_str("        Ok(Self {\n");
        for (i, (sf, lf)) in meta
            .schema_fields
            .iter()
            .zip(layout.fields.iter())
            .enumerate()
        {
            let is_opt = sf.is_optional || sf.is_nullable;
            let off_expr = format!(
                "lust_aot_rt::schema::binary_offset(base_offset, {})?",
                lf.byte_offset
            );
            let val_expr = match (sf.tag, is_opt) {
                (CanonicalTypeTag::Bool, false) => {
                    format!("lust_aot_rt::schema::decode_bin_bool(raw, {off_expr})?")
                }
                (CanonicalTypeTag::Bool, true) => {
                    format!("lust_aot_rt::schema::decode_bin_opt_bool(raw, {off_expr})?")
                }
                (CanonicalTypeTag::Int64 | CanonicalTypeTag::DateTimeUtc, false) => {
                    format!("lust_aot_rt::schema::decode_bin_i64(raw, {off_expr})?")
                }
                (CanonicalTypeTag::UInt64, false) => {
                    format!("lust_aot_rt::schema::decode_bin_u64(raw, {off_expr})?")
                }
                (CanonicalTypeTag::Float64, false) => {
                    format!("lust_aot_rt::schema::decode_bin_f64(raw, {off_expr})?")
                }
                (CanonicalTypeTag::ObjectId, false) => {
                    format!("lust_aot_rt::schema::decode_bin_object_id(raw, {off_expr})?")
                }
                (CanonicalTypeTag::Decimal128, false) => {
                    format!("lust_aot_rt::schema::decode_bin_decimal128(raw, {off_expr})?")
                }
                (CanonicalTypeTag::StringSlice, false) => {
                    format!("lust_aot_rt::schema::decode_bin_string(raw, {off_expr}, ctx)?")
                }
                (CanonicalTypeTag::StringSlice, true) => {
                    format!("lust_aot_rt::schema::decode_bin_opt_string(raw, {off_expr}, ctx)?")
                }
                (CanonicalTypeTag::BinarySlice, false) => {
                    format!("lust_aot_rt::schema::decode_bin_bytes(raw, {off_expr}, ctx)?")
                }
                (CanonicalTypeTag::BinarySlice, true) => {
                    format!("lust_aot_rt::schema::decode_bin_opt_bytes(raw, {off_expr}, ctx)?")
                }
                (CanonicalTypeTag::NestedStruct(_), false) => {
                    if let Some(inner_name) = &sf.nested_struct_name {
                        let inner_rust = rust_struct_name(inner_name);
                        format!("{inner_rust}::decode_binary_at(raw, {off_expr}, ctx)?")
                    } else {
                        "0_i64".to_string()
                    }
                }
                _ => {
                    "return Err(lust_aot_rt::AotError::SchemaDecodeError(\"unsupported binary schema field\"))".to_string()
                }
            };
            let _ = writeln!(out, "            field_{i}: {val_expr},");
        }
        out.push_str("        })\n");
        out.push_str("    }\n\n");
    }

    // 2. Zero-DOM, zero-hash-lookup JSON decoder (`decode_json` + `decode_json_cursor`)
    out.push_str("    /// Zero-DOM, zero-lookup streaming JSON schema decoder (`NG-W12-4C`).\n");
    out.push_str("    #[inline]\n");
    out.push_str(
        "    pub fn decode_json(s: &str, ctx: &mut lust_aot_rt::AotCtx) -> Result<Self, lust_aot_rt::AotError> {\n",
    );
    out.push_str("        let mut __cur = lust_aot_rt::schema::JsonCursor::new(s);\n");
    out.push_str("        let __out = Self::decode_json_cursor(&mut __cur, ctx)?;\n");
    out.push_str("        __cur.expect_eof()?;\n");
    out.push_str("        Ok(__out)\n");
    out.push_str("    }\n\n");

    out.push_str(
        "    pub fn decode_json_cursor(cur: &mut lust_aot_rt::schema::JsonCursor<'_>, ctx: &mut lust_aot_rt::AotCtx) -> Result<Self, lust_aot_rt::AotError> {\n",
    );
    out.push_str("        cur.expect_byte(b'{')?;\n");
    out.push_str("        let mut __out = Self::default();\n");
    let use_bitmask = field_count <= 64;
    if use_bitmask {
        out.push_str("        let mut __seen: u64 = 0;\n");
    } else {
        let _ = writeln!(out, "        let mut __seen = [false; {field_count}];");
    }
    let has_nominal_collision = meta
        .schema_fields
        .iter()
        .any(|sf| matches!(sf.name.as_str(), "_type" | "__type" | "$type"));
    if !has_nominal_collision {
        out.push_str("        let mut __seen_nominal = false;\n");
    }

    out.push_str("        if !cur.consume_byte(b'}') {\n");
    out.push_str("            loop {\n");
    out.push_str("                let __key = cur.parse_key(ctx)?;\n");
    out.push_str("                cur.expect_byte(b':')?;\n");
    out.push_str("                match __key.as_str() {\n");

    if !has_nominal_collision {
        out.push_str("                    \"_type\" | \"__type\" | \"$type\" => {\n");
        out.push_str("                        if __seen_nominal {\n");
        out.push_str("                            return Err(lust_aot_rt::AotError::SchemaDecodeError(\"duplicate nominal type tag\"));\n");
        out.push_str("                        }\n");
        out.push_str("                        __seen_nominal = true;\n");
        out.push_str("                        let __nom = cur.parse_key(ctx)?;\n");
        let _ = writeln!(
            out,
            "                        if __nom.as_str() != {:?} {{",
            meta.name
        );
        out.push_str("                            return Err(lust_aot_rt::AotError::SchemaDecodeError(\"nominal type mismatch\"));\n");
        out.push_str("                        }\n");
        out.push_str("                    }\n");
    }

    for (i, sf) in meta.schema_fields.iter().enumerate() {
        let _ = writeln!(out, "                    {:?} => {{", sf.name);
        if use_bitmask {
            let _ = writeln!(
                out,
                "                        if (__seen & (1_u64 << {i})) != 0 {{"
            );
            out.push_str("                            return Err(lust_aot_rt::AotError::SchemaDecodeError(\"duplicate field in json schema\"));\n");
            out.push_str("                        }\n");
            let _ = writeln!(out, "                        __seen |= 1_u64 << {i};");
        } else {
            let _ = writeln!(out, "                        if __seen[{i}] {{");
            out.push_str("                            return Err(lust_aot_rt::AotError::SchemaDecodeError(\"duplicate field in json schema\"));\n");
            out.push_str("                        }\n");
            let _ = writeln!(out, "                        __seen[{i}] = true;");
        }

        let parse_expr = match sf.tag {
            CanonicalTypeTag::Bool => "cur.parse_bool()?".to_string(),
            CanonicalTypeTag::Int64 | CanonicalTypeTag::DateTimeUtc => {
                "cur.parse_i64()?".to_string()
            }
            CanonicalTypeTag::UInt64 => "cur.parse_u64()?".to_string(),
            CanonicalTypeTag::Float64 => "cur.parse_f64()?".to_string(),
            CanonicalTypeTag::ObjectId => "cur.parse_object_id()?".to_string(),
            CanonicalTypeTag::Decimal128 => "cur.parse_decimal128()?".to_string(),
            CanonicalTypeTag::StringSlice => "cur.parse_string(ctx)?".to_string(),
            CanonicalTypeTag::BinarySlice => "cur.parse_bytes(ctx)?".to_string(),
            CanonicalTypeTag::NestedStruct(_) => {
                if let Some(inner_name) = &sf.nested_struct_name {
                    let inner_rust = rust_struct_name(inner_name);
                    format!("{inner_rust}::decode_json_cursor(cur, ctx)?")
                } else {
                    "cur.parse_i64()?".to_string()
                }
            }
        };

        let is_opt = sf.is_optional || sf.is_nullable;
        if sf.is_nullable {
            out.push_str("                        if cur.consume_null() {\n");
            let _ = writeln!(out, "                            __out.field_{i} = None;");
            out.push_str("                        } else {\n");
            let _ = writeln!(
                out,
                "                            __out.field_{i} = Some({parse_expr});"
            );
            out.push_str("                        }\n");
        } else {
            out.push_str("                        if cur.consume_null() {\n");
            out.push_str("                            return Err(lust_aot_rt::AotError::SchemaDecodeError(\"null value for non-nullable field\"));\n");
            out.push_str("                        }\n");
            if is_opt {
                let _ = writeln!(out, "                        __out.field_{i} = Some({parse_expr});");
            } else {
                let _ = writeln!(out, "                        __out.field_{i} = {parse_expr};");
            }
        }
        out.push_str("                    }\n");
    }

    out.push_str("                    _ => return Err(lust_aot_rt::AotError::SchemaDecodeError(\"unknown field in strict schema\")),\n");
    out.push_str("                }\n");
    out.push_str("                if cur.consume_byte(b',') {\n");
    out.push_str("                    continue;\n");
    out.push_str("                }\n");
    out.push_str("                cur.expect_byte(b'}')?;\n");
    out.push_str("                break;\n");
    out.push_str("            }\n");
    out.push_str("        }\n");

    if use_bitmask {
        let mut req_mask = 0u64;
        for (i, sf) in meta.schema_fields.iter().enumerate() {
            if !sf.is_optional {
                req_mask |= 1u64 << i;
            }
        }
        if req_mask != 0 {
            let _ = writeln!(
                out,
                "        if (__seen & {req_mask}_u64) != {req_mask}_u64 {{"
            );
            out.push_str("            return Err(lust_aot_rt::AotError::SchemaDecodeError(\"missing required field\"));\n");
            out.push_str("        }\n");
        }
    } else {
        for (i, sf) in meta.schema_fields.iter().enumerate() {
            if !sf.is_optional {
                let _ = writeln!(
                    out,
                    "        if !__seen[{i}] {{ return Err(lust_aot_rt::AotError::SchemaDecodeError(\"missing required field\")); }}"
                );
            }
        }
    }

    out.push_str("        Ok(__out)\n");
    out.push_str("    }\n");
    out.push_str("}\n\n");
}

fn emit_enum_definitions(out: &mut String, ctx: &ModuleEmitContext<'_>) {
    if ctx.enums.is_empty() {
        return;
    }
    out.push_str("// User Enum Definitions (Tier 4, NG-W12-4B)\n");
    for meta in ctx.enums.values() {
        let ename = rust_enum_name(&meta.name);
        let lt_decl = if meta.needs_arena_lt { "<'a>" } else { "" };
        if meta.is_copy {
            out.push_str("#[derive(Clone, Copy, Debug, PartialEq)]\n");
        } else {
            out.push_str("#[derive(Clone, Debug, PartialEq)]\n");
        }
        let _ = writeln!(out, "pub enum {ename}{lt_decl} {{");
        for (vname, fields) in &meta.variants {
            let rv = sanitize_ident(vname);
            if fields.is_empty() {
                let _ = writeln!(out, "    {rv},");
            } else {
                let ftys: Vec<String> = fields.iter().map(|t| ctx.rust_type(t, "'a")).collect();
                let _ = writeln!(out, "    {rv}({}),", ftys.join(", "));
            }
        }
        out.push_str("}\n\n");

        if meta.derives_default
            && let Some((v0_name, v0_fields)) = meta.variants.first()
        {
            let rv0 = sanitize_ident(v0_name);
            let _ = writeln!(out, "impl{lt_decl} Default for {ename}{lt_decl} {{");
            out.push_str("    #[inline]\n");
            out.push_str("    fn default() -> Self {\n");
            if v0_fields.is_empty() {
                let _ = writeln!(out, "        Self::{rv0}");
            } else {
                let defs: Vec<String> = v0_fields
                    .iter()
                    .map(|t| ctx.rust_default_expr(t))
                    .collect();
                let _ = writeln!(out, "        Self::{rv0}({})", defs.join(", "));
            }
            out.push_str("    }\n");
            out.push_str("}\n\n");
        }

        let _ = writeln!(out, "impl{lt_decl} {ename}{lt_decl} {{");
        out.push_str("    #[inline]\n");
        out.push_str("    pub fn __lust_tag(&self) -> i64 {\n");
        out.push_str("        match self {\n");
        for (vidx, (vname, fields)) in meta.variants.iter().enumerate() {
            let rv = sanitize_ident(vname);
            if fields.is_empty() {
                let _ = writeln!(out, "            Self::{rv} => {vidx}_i64,");
            } else {
                let _ = writeln!(out, "            Self::{rv}(..) => {vidx}_i64,");
            }
        }
        out.push_str("        }\n");
        out.push_str("    }\n");
        out.push_str("}\n\n");

        let _ = writeln!(
            out,
            "impl{lt_decl} core::fmt::Display for {ename}{lt_decl} {{"
        );
        out.push_str("    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {\n");
        out.push_str("        match self {\n");
        for (vname, fields) in &meta.variants {
            let rv = sanitize_ident(vname);
            if fields.is_empty() {
                let _ = writeln!(
                    out,
                    "            Self::{rv} => write!(f, \"{}.{}\"),",
                    meta.name, vname
                );
            } else {
                let pats: Vec<String> = (0..fields.len()).map(|i| format!("__f{i}")).collect();
                let _ = writeln!(out, "            Self::{rv}({}) => {{", pats.join(", "));
                let _ = writeln!(out, "                write!(f, \"{}.{}(\")?;", meta.name, vname);
                for (i, fty) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push_str("                write!(f, \", \")?;\n");
                    }
                    match fty {
                        MirType::Int
                        | MirType::Float
                        | MirType::Bool
                        | MirType::String
                        | MirType::Enum(_)
                        | MirType::Map(_, _)
                        | MirType::Unknown => {
                            let _ = writeln!(out, "                write!(f, \"{{}}\", __f{i})?;");
                        }
                        _ => {
                            let _ = writeln!(out, "                write!(f, \"{{:?}}\", __f{i})?;");
                        }
                    }
                }
                out.push_str("                write!(f, \")\")\n");
                out.push_str("            }\n");
            }
        }
        out.push_str("        }\n");
        out.push_str("    }\n");
        out.push_str("}\n\n");
    }
}

fn emit_entrypoint(out: &mut String, module: &MirModule, ctx: &ModuleEmitContext<'_>) {
    let main_fn = module
        .functions
        .iter()
        .find(|f| f.signature.name == "main" || f.signature.name.ends_with(".main"));

    out.push_str("fn main() {\n");
    let Some(main) = main_fn else {
        out.push_str("}\n");
        return;
    };

    let meta = &ctx.fn_metas[&main.signature.name];
    let gas_limit_expr = match ctx.opts.gas_limit {
        Some(limit) => format!("Some({limit}_u64)"),
        None => "None".to_string(),
    };
    let mem_limit_expr = match ctx.opts.max_memory_bytes {
        Some(limit) => format!("Some({limit}_usize)"),
        None => "None".to_string(),
    };
    let alloc_expr = match ctx.opts.max_memory_bytes {
        Some(limit) => format!("lust_aot_rt::LustBudgetAlloc::with_limit({limit}_usize)"),
        None => "lust_aot_rt::LustBudgetAlloc::unbounded()".to_string(),
    };

    if meta.needs_ctx {
        if ctx.opts.pure_core {
            let _ = writeln!(out, "    let mut __ctx = lust_aot_rt::AotCtx::pure_core();");
            if let Some(limit) = ctx.opts.gas_limit {
                let _ = writeln!(out, "    __ctx.gas_limit = Some({limit}_u64);");
                let _ = writeln!(
                    out,
                    "    __ctx.gas_remaining = {};",
                    limit.min(i64::MAX as u64)
                );
            }
            if let Some(limit) = ctx.opts.max_memory_bytes {
                let _ = writeln!(
                    out,
                    "    __ctx.budget_alloc = lust_aot_rt::LustBudgetAlloc::with_limit({limit}_usize);"
                );
            }
        } else {
            let _ = writeln!(
                out,
                "    let mut __ctx = lust_aot_rt::AotCtx::with_limits({gas_limit_expr}, {mem_limit_expr});"
            );
        }
    }

    if meta.needs_alloc || meta.needs_arena || ctx.module_uses_arena {
        let _ = writeln!(out, "    let __alloc = {alloc_expr};");
    }
    if meta.needs_arena {
        let _ = writeln!(
            out,
            "    let __arena = lust_aot_rt::LustChunkArena::new(&__alloc);"
        );
    }

    let mut call_args = Vec::new();
    if meta.needs_ctx {
        call_args.push("&mut __ctx");
    }
    if meta.needs_alloc {
        call_args.push("&__alloc");
    }
    if meta.needs_arena {
        call_args.push("&__arena");
    }

    let call_expr = format!("{}({})", meta.rust_name, call_args.join(", "));
    if meta.can_fail {
        let _ = writeln!(out, "    match {call_expr} {{");
        match main.signature.return_type {
            MirType::Int => {
                out.push_str("        Ok(__code) => {\n");
                out.push_str("            if __code != 0 {\n");
                out.push_str("                std::process::exit(__code as i32);\n");
                out.push_str("            }\n");
                out.push_str("        }\n");
            }
            _ => {
                out.push_str("        Ok(_) => {}\n");
            }
        }
        out.push_str("        Err(__err) => {\n");
        out.push_str("            use std::io::Write as _;\n");
        out.push_str("            let _ = writeln!(std::io::stderr(), \"runtime error: {}\", __err);\n");
        out.push_str("            std::process::exit(1);\n");
        out.push_str("        }\n");
        out.push_str("    }\n");
    } else {
        match main.signature.return_type {
            MirType::Int => {
                let _ = writeln!(out, "    let Ok(__code) = {call_expr};");
                out.push_str("    if __code != 0 {\n");
                out.push_str("        std::process::exit(__code as i32);\n");
                out.push_str("    }\n");
            }
            _ => {
                let _ = writeln!(out, "    let Ok(_) = {call_expr};");
            }
        }
    }

    out.push_str("}\n");
}

// ---------------------------------------------------------------------------
// Per-Function Emission
// ---------------------------------------------------------------------------

struct FnEmitter<'a, 'm> {
    func: &'m MirFunction,
    ctx: &'a ModuleEmitContext<'m>,
    meta: &'a FnMeta,
    out: &'a mut String,
    /// Single-def alias resolution for inlined Array / Struct receiver copies (`dst -> root_src`).
    alias_map: HashMap<LocalId, LocalId>,
    /// Statements skipped because they were folded into `IfArrayGetOk`, `IfOptionSome`,
    /// `tostring+Concat` fusion, alias substitution, or stripped inlined `Defer` duplicates.
    skip_stmts: HashSet<(BlockId, usize)>,
    /// Blocks whose `Terminator::Call` is skipped (stripped inlined `Defer` duplicates).
    skip_calls: HashSet<BlockId>,
    /// Blocks whose `Terminator::Call { callee: "tostring", .. }` is fused into an in-place
    /// `string_append_int` / `string_append_float`.
    fused_tostring_calls: HashMap<BlockId, (LocalId, Operand, MirType)>,
    /// Locals whose Rust binding is a scoped `&LustStruct_S<'a>` introduced by `if let Some(...)`.
    scoped_ref_locals: HashSet<LocalId>,
    /// Locals that do not need a top-level `let mut _N` declaration (e.g. aliases, folded bindings).
    omitted_locals: HashSet<LocalId>,
    /// Count of uses per local (for move vs clone decisions).
    local_use_counts: HashMap<LocalId, usize>,
    /// Count of Place::Local definitions per local.
    local_def_counts: HashMap<LocalId, usize>,
    /// Pre-detected `IfOptionSome` patterns keyed by `cond_block`.
    if_option_some: HashMap<BlockId, IfOptionSomePlan>,
    /// Names of active `errdefer` DropGuard variables to dismiss before every `return Ok(...)`.
    errdefer_guards: Vec<String>,
    /// Active cleanup closure index when emitting a `defer`/`errdefer` closure body.
    in_cleanup_closure: Option<usize>,
    /// Remaps captured outer locals to their closure-cloned variable names inside a cleanup closure.
    cleanup_var_remap: HashMap<LocalId, String>,
    /// Maps `_tag_local -> _enum_local` from `_tag = enum.tag(_e)`.
    switch_tag_source: HashMap<LocalId, LocalId>,
    /// Active narrowed enum variant index inside a `Switch` arm matching `_e.__lust_tag()`.
    active_enum_variant: HashMap<LocalId, usize>,
    /// Maps local variable source name -> LocalId (for calling local closure variables).
    local_name_to_id: HashMap<String, LocalId>,
}

#[derive(Debug, Clone)]
struct IfOptionSomePlan {
    src_place: Place,
    binding_local: LocalId,
    struct_name: String,
}

fn emit_function(
    out: &mut String,
    func: &MirFunction,
    ctx: &ModuleEmitContext<'_>,
) -> Result<()> {
    let meta = &ctx.fn_metas[&func.signature.name];
    let tree = structure_mir_function(func);

    let mut local_def_counts: HashMap<LocalId, usize> = HashMap::new();
    let mut switch_tag_source: HashMap<LocalId, LocalId> = HashMap::new();
    for block in &func.blocks {
        if block.is_cleanup {
            continue;
        }
        for stmt in &block.statements {
            if let Statement::Assign(Place::Local(dst), _, _) = stmt {
                *local_def_counts.entry(*dst).or_insert(0) += 1;
            }
        }
        if let Some(Terminator::Call {
            callee,
            args,
            destination: Place::Local(dst),
            ..
        }) = &block.terminator
        {
            *local_def_counts.entry(*dst).or_insert(0) += 1;
            if callee == "enum.tag"
                && let Some(enum_loc) = args.first().and_then(operand_exact_local)
            {
                switch_tag_source.insert(*dst, enum_loc);
            }
        }
    }

    let mut local_name_to_id: HashMap<String, LocalId> = HashMap::new();
    for decl in &func.locals {
        if let Some(name) = &decl.name
            && matches!(decl.ty, MirType::Function { .. } | MirType::Unknown)
        {
            local_name_to_id.insert(name.clone(), decl.id);
        }
    }

    let mut fe = FnEmitter {
        func,
        ctx,
        meta,
        out,
        alias_map: HashMap::new(),
        skip_stmts: HashSet::new(),
        skip_calls: HashSet::new(),
        fused_tostring_calls: HashMap::new(),
        scoped_ref_locals: HashSet::new(),
        omitted_locals: HashSet::new(),
        local_use_counts: count_local_uses(func),
        local_def_counts,
        if_option_some: HashMap::new(),
        errdefer_guards: Vec::new(),
        in_cleanup_closure: None,
        cleanup_var_remap: HashMap::new(),
        switch_tag_source,
        active_enum_variant: HashMap::new(),
        local_name_to_id,
    };

    // Pre-passes on the structured function
    fe.analyze_single_def_aliases();
    fe.analyze_tostring_concat_fusion();
    fe.analyze_structured_folds(&tree);
    fe.analyze_inlined_defer_strips();

    // Emit function signature
    fe.emit_signature();

    // Emit local declarations (including Rc<Cell>/Rc<RefCell> wrappers for Tier 4)
    fe.emit_local_declarations();

    // Emit `defer` / `errdefer` guards via `core::mem::DropGuard` (NG-W12-4B, Hazard 9)
    fe.emit_cleanup_guards()?;

    // Emit structured body
    fe.emit_node(&tree, 1)?;

    // Safety trailing return in case of empty trailing fallthrough on Unit/Int functions
    if !node_always_returns(&tree) {
        let def_ret = fe.ctx.rust_default_expr(&func.signature.return_type);
        if !fe.errdefer_guards.is_empty() {
            let _ = writeln!(fe.out, "    let __ret = {def_ret};");
            for g in fe.errdefer_guards.clone().iter().rev() {
                let _ = writeln!(fe.out, "    core::mem::DropGuard::dismiss({g});");
            }
            fe.out.push_str("    Ok(__ret)\n");
        } else {
            let _ = writeln!(fe.out, "    Ok({def_ret})");
        }
    }

    fe.out.push_str("}\n\n");
    Ok(())
}

impl<'a, 'm> FnEmitter<'a, 'm> {
    fn resolve_local(&self, mut loc: LocalId) -> LocalId {
        let mut hops = 0usize;
        while let Some(&target) = self.alias_map.get(&loc) {
            loc = target;
            hops += 1;
            if hops > self.func.locals.len() {
                break;
            }
        }
        loc
    }

    fn local_storage(&self, loc: LocalId) -> LocalStorage {
        let r = self.resolve_local(loc);
        self.meta
            .local_storage
            .get(r.0)
            .copied()
            .unwrap_or(LocalStorage::Direct)
    }

    fn local_var_name(&self, loc: LocalId) -> String {
        let r = self.resolve_local(loc);
        if let Some(mapped) = self.cleanup_var_remap.get(&r) {
            mapped.clone()
        } else {
            format!("_{}", r.0)
        }
    }

    fn is_param(&self, loc: LocalId) -> Option<(usize, ParamMode)> {
        self.func
            .params
            .iter()
            .position(|&p| p == loc)
            .map(|pi| (pi, self.meta.param_modes[pi]))
    }

    fn local_is_ref(&self, loc: LocalId) -> bool {
        let resolved = self.resolve_local(loc);
        if self.scoped_ref_locals.contains(&resolved) {
            return true;
        }
        if let Some((_, mode)) = self.is_param(resolved) {
            return matches!(
                mode,
                ParamMode::RefStruct
                    | ParamMode::MutRefStruct
                    | ParamMode::RefSlice
                    | ParamMode::MutVec
                    | ParamMode::RefStr
            );
        }
        false
    }

    fn fn_needs_lifetime(&self) -> bool {
        if self.meta.needs_alloc || self.meta.needs_arena {
            return true;
        }
        if self.ctx.type_needs_lifetime(&self.func.signature.return_type) {
            return true;
        }
        for (pi, &pid) in self.func.params.iter().enumerate() {
            let ty = &self.func.locals[pid.0].ty;
            match self.meta.param_modes[pi] {
                ParamMode::MutVec => return true,
                ParamMode::ByValue
                | ParamMode::RefStruct
                | ParamMode::MutRefStruct
                | ParamMode::RcCell
                | ParamMode::RcRefCell => {
                    if self.ctx.type_needs_lifetime(ty) {
                        return true;
                    }
                }
                ParamMode::RefSlice | ParamMode::RefStr => {
                    if let MirType::Array(elem) = ty
                        && self.ctx.type_needs_lifetime(elem)
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn emit_signature(&mut self) {
        let has_lt = self.fn_needs_lifetime();
        let lt_str = if has_lt { "<'a>" } else { "" };

        let mut params = Vec::new();
        if self.meta.needs_ctx {
            params.push("__ctx: &mut lust_aot_rt::AotCtx".to_string());
        }
        if self.meta.needs_alloc {
            params.push("__alloc: &'a lust_aot_rt::LustBudgetAlloc".to_string());
        }
        if self.meta.needs_arena {
            params.push("__arena: &'a lust_aot_rt::LustChunkArena<'a>".to_string());
        }

        for (pi, &pid) in self.func.params.iter().enumerate() {
            let ty = &self.func.locals[pid.0].ty;
            let mode = self.meta.param_modes[pi];
            let p_str = match mode {
                ParamMode::ByValue => {
                    let rty = self.ctx.rust_type(ty, "'a");
                    format!("mut _{}: {rty}", pid.0)
                }
                ParamMode::RefStruct => {
                    let rty = self.ctx.rust_type(ty, "'a");
                    format!("_{}: &{rty}", pid.0)
                }
                ParamMode::MutRefStruct => {
                    let rty = self.ctx.rust_type(ty, "'a");
                    format!("_{}: &mut {rty}", pid.0)
                }
                ParamMode::RefSlice => {
                    let elem = match ty {
                        MirType::Array(inner) => self.ctx.rust_type(inner, "'a"),
                        _ => "i64".to_string(),
                    };
                    format!("_{}: &[{elem}]", pid.0)
                }
                ParamMode::MutVec => {
                    let rty = self.ctx.rust_type(ty, "'a");
                    format!("_{}: &mut {rty}", pid.0)
                }
                ParamMode::RefStr => format!("_{}: &str", pid.0),
                ParamMode::RcCell => {
                    let rty = self.ctx.rust_type(ty, "'a");
                    format!("_{}: alloc::rc::Rc<core::cell::Cell<{rty}>>", pid.0)
                }
                ParamMode::RcRefCell => {
                    let rty = self.ctx.rust_type(ty, "'a");
                    format!(
                        "mut _{}: alloc::rc::Rc<core::cell::RefCell<{rty}>>",
                        pid.0
                    )
                }
            };
            params.push(p_str);
        }

        let ret_ok_ty = self.ctx.rust_type(&self.func.signature.return_type, "'a");
        let err_ty = if self.meta.can_fail {
            "lust_aot_rt::AotError"
        } else {
            "core::convert::Infallible"
        };

        let _ = writeln!(
            self.out,
            "fn {}{lt_str}({}) -> Result<{ret_ok_ty}, {err_ty}> {{",
            self.meta.rust_name,
            params.join(", ")
        );
    }

    fn emit_local_declarations(&mut self) {
        let param_set: HashSet<LocalId> = self.func.params.iter().copied().collect();

        // Shadow any ByValue params promoted to local RcCell/RcRefCell (e.g. reassigned param captured by closure/defer)
        for (pi, &pid) in self.func.params.iter().enumerate() {
            let pmode = self.meta.param_modes[pi];
            let lstorage = self.local_storage(pid);
            if pmode == ParamMode::ByValue {
                match lstorage {
                    LocalStorage::RcCell => {
                        let _ = writeln!(
                            self.out,
                            "    let _{} = alloc::rc::Rc::new(core::cell::Cell::new(_{}));",
                            pid.0, pid.0
                        );
                    }
                    LocalStorage::RcRefCell => {
                        let _ = writeln!(
                            self.out,
                            "    let mut _{} = alloc::rc::Rc::new(core::cell::RefCell::new(_{}));",
                            pid.0, pid.0
                        );
                    }
                    LocalStorage::Direct => {}
                }
            }
        }

        for decl in &self.func.locals {
            if param_set.contains(&decl.id) || self.omitted_locals.contains(&decl.id) {
                continue;
            }
            let rty = self.ctx.rust_type(&decl.ty, "'a");
            let def = self.ctx.rust_default_expr(&decl.ty);
            match self.local_storage(decl.id) {
                LocalStorage::Direct => {
                    let _ = writeln!(self.out, "    let mut _{}: {rty} = {def};", decl.id.0);
                }
                LocalStorage::RcCell => {
                    let _ = writeln!(
                        self.out,
                        "    let _{}: alloc::rc::Rc<core::cell::Cell<{rty}>> = alloc::rc::Rc::new(core::cell::Cell::new({def}));",
                        decl.id.0
                    );
                }
                LocalStorage::RcRefCell => {
                    let _ = writeln!(
                        self.out,
                        "    let mut _{}: alloc::rc::Rc<core::cell::RefCell<{rty}>> = alloc::rc::Rc::new(core::cell::RefCell::new({def}));",
                        decl.id.0
                    );
                }
            }
        }
    }

    /// Detects `AstLowering`-inlined `MirCleanupKind::Defer` duplicates right before `Terminator::Return`
    /// and marks them skipped so `core::mem::DropGuard` (`_lust_defer_N`) executes each `defer`
    /// exactly once on both normal `Ok(...)` exits and `Err(...)` exits in exact LIFO order.
    fn analyze_inlined_defer_strips(&mut self) {
        let defer_entries: Vec<BlockId> = self
            .func
            .cleanup_blocks
            .iter()
            .filter(|c| c.kind == MirCleanupKind::Defer)
            .map(|c| c.block)
            .collect();
        if defer_entries.is_empty() {
            return;
        }

        // Build single-predecessor map for non-cleanup blocks
        let mut preds: Vec<Vec<BlockId>> = vec![Vec::new(); self.func.blocks.len()];
        for blk in &self.func.blocks {
            if blk.is_cleanup {
                continue;
            }
            if let Some(term) = &blk.terminator {
                for succ_idx in term.successors() {
                    if succ_idx < preds.len() {
                        preds[succ_idx].push(blk.id);
                    }
                }
            }
        }

        // Fingerprint each Defer cleanup block as a sequence of atomic actions (calls + assigns)
        let mut defer_fingerprints: Vec<Vec<CleanupActionSig>> = Vec::new();
        for &cbb in &defer_entries {
            defer_fingerprints.push(extract_cleanup_action_sigs(self.func, cbb));
        }

        // For every non-cleanup block ending in Terminator::Return, check if its immediate
        // preceding linear chain ends with the exact reverse-order inlined defer fingerprints.
        let ret_blocks: Vec<BlockId> = self
            .func
            .blocks
            .iter()
            .filter(|b| !b.is_cleanup && matches!(b.terminator, Some(Terminator::Return(_, _))))
            .map(|b| b.id)
            .collect();

        for ret_bb in ret_blocks {
            // Collect linear chain ending at ret_bb (stopping at join points or non-Call/Goto preds)
            let mut chain: Vec<BlockId> = vec![ret_bb];
            let mut cur = ret_bb;
            while preds[cur.0].len() == 1 {
                let p = preds[cur.0][0];
                if matches!(
                    self.func.blocks[p.0].terminator,
                    Some(Terminator::Call { .. } | Terminator::Goto(_, _))
                ) {
                    chain.push(p);
                    cur = p;
                } else {
                    break;
                }
            }
            chain.reverse(); // chronological order up to ret_bb

            // Flatten actions along the chain with their exact (BlockId, ActionLoc)
            let mut chain_actions: Vec<(CleanupActionSig, ChainActionLoc)> = Vec::new();
            for &bb in &chain {
                let blk = &self.func.blocks[bb.0];
                for (sidx, stmt) in blk.statements.iter().enumerate() {
                    if let Some(sig) = stmt_action_sig(stmt, self.func) {
                        chain_actions.push((sig, ChainActionLoc::Stmt(bb, sidx)));
                    }
                }
                if let Some(Terminator::Call { callee, args, .. }) = &blk.terminator {
                    chain_actions.push((
                        call_action_sig(callee, args, self.func),
                        ChainActionLoc::Call(bb),
                    ));
                }
            }

            // Expected suffix in chronological order: defer_fingerprints in REVERSE order
            // (because AstLowering inlines `defer` in LIFO order before Return).
            let mut expected: Vec<CleanupActionSig> = Vec::new();
            for fp in defer_fingerprints.iter().rev() {
                expected.extend(fp.iter().cloned());
            }
            if expected.is_empty() || chain_actions.len() < expected.len() {
                continue;
            }
            let start_idx = chain_actions.len() - expected.len();
            let matches_all = chain_actions[start_idx..]
                .iter()
                .zip(expected.iter())
                .all(|((got, _), exp)| got == exp);
            if matches_all {
                for (_, loc) in &chain_actions[start_idx..] {
                    match *loc {
                        ChainActionLoc::Stmt(bb, sidx) => {
                            self.skip_stmts.insert((bb, sidx));
                        }
                        ChainActionLoc::Call(bb) => {
                            self.skip_calls.insert(bb);
                        }
                    }
                }
            }
        }
    }

    /// Emits `let _lust_defer_N = lust_aot_rt::defer(...)` and
    /// `let _lust_errdefer_N = lust_aot_rt::errdefer(...)` guards backed by `core::mem::DropGuard`.
    fn emit_cleanup_guards(&mut self) -> Result<()> {
        if self.func.cleanup_blocks.is_empty() {
            return Ok(());
        }

        let entries: Vec<MirCleanupEntry> = self.func.cleanup_blocks.clone();
        for (idx, entry) in entries.iter().enumerate() {
            if entry.block.0 >= self.func.blocks.len() {
                continue;
            }

            // Collect locals referenced inside this specific cleanup subgraph
            let mut sub_locals: HashSet<LocalId> = HashSet::new();
            let mut q = VecDeque::new();
            let mut seen = HashSet::new();
            seen.insert(entry.block);
            q.push_back(entry.block);
            while let Some(bb) = q.pop_front() {
                collect_block_locals(&self.func.blocks[bb.0], &mut sub_locals);
                if let Some(term) = &self.func.blocks[bb.0].terminator {
                    for s in term.successors() {
                        let sb = BlockId(s);
                        if sb.0 < self.func.blocks.len() && seen.insert(sb) {
                            q.push_back(sb);
                        }
                    }
                }
            }

            // Capture outer locals that appear in main body / params
            let mut captured_outer: Vec<LocalId> = sub_locals
                .iter()
                .copied()
                .filter(|loc| self.meta.cleanup_captured.contains(loc))
                .collect();
            captured_outer.sort_by_key(|l| l.0);

            // Cleanup-internal locals (created inside cleanup block only)
            let mut internal_locals: Vec<LocalId> = sub_locals
                .iter()
                .copied()
                .filter(|loc| !self.meta.cleanup_captured.contains(loc))
                .collect();
            internal_locals.sort_by_key(|l| l.0);

            let mut remap: HashMap<LocalId, String> = HashMap::new();
            for &loc in &captured_outer {
                let r = self.resolve_local(loc);
                let cg_name = format!("__cg_{idx}_{}", r.0);
                match self.local_storage(r) {
                    LocalStorage::RcCell | LocalStorage::RcRefCell => {
                        let _ = writeln!(
                            self.out,
                            "    let {cg_name} = alloc::rc::Rc::clone(&_{});",
                            r.0
                        );
                    }
                    LocalStorage::Direct => {
                        let ty = &self.func.locals[r.0].ty;
                        if self.ctx.is_copy_type(ty) {
                            if self.local_is_ref(r) {
                                let _ = writeln!(self.out, "    let {cg_name} = *_{};", r.0);
                            } else {
                                let _ = writeln!(self.out, "    let {cg_name} = _{};", r.0);
                            }
                        } else if self.is_param(r).is_some_and(|(_, m)| m == ParamMode::RefStr) {
                            let _ = writeln!(
                                self.out,
                                "    let {cg_name} = alloc::string::String::from(_{});",
                                r.0
                            );
                        } else {
                            let _ = writeln!(self.out, "    let {cg_name} = _{}.clone();", r.0);
                        }
                    }
                }
                remap.insert(r, cg_name);
            }

            // Build structured tree for this cleanup subgraph by cloning a tiny view with entry = entry.block
            let mut sub_fn = self.func.clone();
            sub_fn.entry = entry.block;
            let cleanup_tree = structure_mir_function(&sub_fn);

            let (guard_var, helper_fn) = match entry.kind {
                MirCleanupKind::Defer => (format!("_lust_defer_{idx}"), "lust_aot_rt::defer"),
                MirCleanupKind::ErrDefer => {
                    let g = format!("_lust_errdefer_{idx}");
                    self.errdefer_guards.push(g.clone());
                    (g, "lust_aot_rt::errdefer")
                }
            };

            let _ = writeln!(self.out, "    let {guard_var} = {helper_fn}(move || {{");
            self.out
                .push_str("        let _ = (|| -> Result<(), lust_aot_rt::AotError> {\n");

            // Declare cleanup-internal temporaries inside the closure
            for &iloc in &internal_locals {
                let decl = &self.func.locals[iloc.0];
                let rty = self.ctx.rust_type(&decl.ty, "'a");
                let def = self.ctx.rust_default_expr(&decl.ty);
                let _ = writeln!(self.out, "            let mut _{}: {rty} = {def};", iloc.0);
            }

            let saved_in_cleanup = self.in_cleanup_closure;
            let saved_remap = core::mem::replace(&mut self.cleanup_var_remap, remap);
            self.in_cleanup_closure = Some(idx);

            self.emit_node(&cleanup_tree, 3)?;

            self.in_cleanup_closure = saved_in_cleanup;
            self.cleanup_var_remap = saved_remap;

            self.out.push_str("            Ok(())\n");
            self.out.push_str("        })();\n");
            self.out.push_str("    });\n");
        }
        Ok(())
    }

    /// Detects single-definition local aliases (`dst = Copy/Move(src)`) of Arrays or Structs
    /// (commonly emitted when `MirInliner` inlines a function taking an Array or Struct param)
    /// and resolves `dst -> src` directly to preserve reference semantics with zero cloning.
    /// Locals promoted to `RcCell` or `RcRefCell` (Hazard 12) are excluded and use `Rc::clone`.
    fn analyze_single_def_aliases(&mut self) {
        let param_set: HashSet<LocalId> = self.func.params.iter().copied().collect();
        let mut def_counts: HashMap<LocalId, usize> = HashMap::new();
        let mut single_copy_def: HashMap<LocalId, (BlockId, usize, LocalId)> = HashMap::new();

        for block in &self.func.blocks {
            if block.is_cleanup {
                continue;
            }
            for (sidx, stmt) in block.statements.iter().enumerate() {
                if let Statement::Assign(Place::Local(dst), rval, _) = stmt {
                    *def_counts.entry(*dst).or_insert(0) += 1;
                    if let Rvalue::Use(
                        Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)),
                    ) = rval
                        && dst != src
                    {
                        single_copy_def.insert(*dst, (block.id, sidx, *src));
                    }
                }
            }
            if let Some(Terminator::Call {
                destination: Place::Local(dst),
                ..
            }) = &block.terminator
            {
                *def_counts.entry(*dst).or_insert(0) += 1;
            }
        }

        for (&dst, &(bb, sidx, src)) in &single_copy_def {
            if param_set.contains(&dst) || dst == self.func.return_local {
                continue;
            }
            // Skip any local managed via Rc<Cell> or Rc<RefCell> (Hazard 12)
            if self.meta.local_storage.get(dst.0).copied() != Some(LocalStorage::Direct)
                || self.meta.local_storage.get(src.0).copied() != Some(LocalStorage::Direct)
            {
                continue;
            }
            if def_counts.get(&dst).copied().unwrap_or(0) != 1 {
                continue;
            }
            let src_defs = def_counts.get(&src).copied().unwrap_or(0);
            let src_is_stable = if param_set.contains(&src) {
                src_defs == 0
            } else {
                src_defs == 1
            };
            if !src_is_stable {
                continue;
            }
            let ty = &self.func.locals[dst.0].ty;
            let is_ref_semantic = matches!(ty, MirType::Array(_) | MirType::Struct(_));
            if !is_ref_semantic {
                continue;
            }
            let src_uses = self.local_use_counts.get(&src).copied().unwrap_or(0);
            if src_uses <= 1 && !param_set.contains(&src) {
                continue;
            }
            if param_escapes_by_move(self.func, dst) {
                continue;
            }
            self.alias_map.insert(dst, src);
            self.skip_stmts.insert((bb, sidx));
            self.omitted_locals.insert(dst);
        }
    }

    /// Detects `tmp = call tostring(x)` immediately followed by `s = s .. tmp` in the
    /// single-predecessor `success` block, fusing into in-place `string_append_int`/`float`.
    fn analyze_tostring_concat_fusion(&mut self) {
        let mut pred_counts = vec![0usize; self.func.blocks.len()];
        for block in &self.func.blocks {
            if let Some(term) = &block.terminator {
                for succ in term.successors() {
                    if succ < pred_counts.len() {
                        pred_counts[succ] += 1;
                    }
                }
            }
        }

        for block in &self.func.blocks {
            let Some(Terminator::Call {
                callee,
                args,
                destination: Place::Local(tmp),
                success,
                error: None,
                ..
            }) = &block.terminator
            else {
                continue;
            };
            if callee != "tostring" || args.len() != 1 {
                continue;
            }
            if self.local_use_counts.get(tmp).copied().unwrap_or(0) != 1 {
                continue;
            }
            if pred_counts.get(success.0).copied().unwrap_or(0) != 1 {
                continue;
            }
            let arg_ty = operand_mir_type(&args[0], self.func);
            if !matches!(arg_ty, MirType::Int | MirType::Float) {
                continue;
            }
            let succ_block = &self.func.blocks[success.0];
            for (sidx, stmt) in succ_block.statements.iter().enumerate() {
                match stmt {
                    Statement::StorageLive(_, _) | Statement::StorageDead(_, _) => continue,
                    Statement::Assign(
                        Place::Local(dst_s),
                        Rvalue::BinaryOp(MirBinaryOp::Concat, lhs, rhs),
                        _,
                    ) => {
                        let lhs_is_dst = matches!(
                            lhs,
                            Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l))
                            if l == dst_s
                        );
                        let rhs_is_tmp = matches!(
                            rhs,
                            Operand::Copy(Place::Local(r)) | Operand::Move(Place::Local(r))
                            if r == tmp
                        );
                        if lhs_is_dst
                            && rhs_is_tmp
                            && self.local_storage(*dst_s) == LocalStorage::Direct
                        {
                            self.fused_tostring_calls
                                .insert(block.id, (*dst_s, args[0].clone(), arg_ty));
                            self.skip_stmts.insert((*success, sidx));
                            self.omitted_locals.insert(*tmp);
                        }
                        break;
                    }
                    _ => break,
                }
            }
        }
    }

    /// Walks the `StructuredNode` tree to record statement skips for `IfArrayGetOk` and
    /// `IfOptionSome` (`if node.left is Some(l)`).
    fn analyze_structured_folds(&mut self, node: &StructuredNode) {
        match node {
            StructuredNode::IfArrayGetOk {
                header_block,
                arr_local,
                idx_local,
                binding_local,
                then_branch,
                else_branch,
            } => {
                // Skip the canonical bounds-check temporaries in `header_block` if unused elsewhere
                self.mark_if_array_get_header_skips(*header_block);
                // Skip the redundant `binding_local = arr_local[idx_local]` in the leading block of `then_branch`
                if let Some(then_bb) = leading_block_of(then_branch) {
                    let stmts = &self.func.blocks[then_bb.0].statements;
                    for (sidx, stmt) in stmts.iter().enumerate() {
                        if let Statement::Assign(Place::Local(dst), Rvalue::Use(op), _) = stmt
                            && dst == binding_local
                            && let Operand::Copy(Place::Index(base, idx))
                            | Operand::Move(Place::Index(base, idx)) = op
                            && **base == Place::Local(*arr_local)
                            && idx == idx_local
                        {
                            self.skip_stmts.insert((then_bb, sidx));
                            break;
                        }
                    }
                }
                self.analyze_structured_folds(then_branch);
                if let Some(eb) = else_branch {
                    self.analyze_structured_folds(eb);
                }
            }
            StructuredNode::If {
                cond_block,
                cond_op,
                then_branch,
                else_branch,
            } => {
                self.try_detect_if_option_some(*cond_block, cond_op, then_branch);
                self.analyze_structured_folds(then_branch);
                if let Some(eb) = else_branch {
                    self.analyze_structured_folds(eb);
                }
            }
            StructuredNode::Seq(items) => {
                for item in items {
                    self.analyze_structured_folds(item);
                }
            }
            StructuredNode::Switch { cases, default, .. } => {
                for (_, c) in cases {
                    self.analyze_structured_folds(c);
                }
                self.analyze_structured_folds(default);
            }
            StructuredNode::WhileLoop { body, .. }
            | StructuredNode::Loop { body, .. }
            | StructuredNode::LabeledBlock { body, .. } => {
                self.analyze_structured_folds(body);
            }
            _ => {}
        }
    }

    fn mark_if_array_get_header_skips(&mut self, header_block: BlockId) {
        let Some(Terminator::Cond { cond, .. }) = &self.func.blocks[header_block.0].terminator
        else {
            return;
        };
        let Some(cond_local) = operand_exact_local(cond) else {
            return;
        };
        let stmts = &self.func.blocks[header_block.0].statements;

        let mut last_def: HashMap<LocalId, (usize, &Rvalue)> = HashMap::new();
        for (idx, stmt) in stmts.iter().enumerate() {
            if let Statement::Assign(Place::Local(dst), rval, _) = stmt {
                last_def.insert(*dst, (idx, rval));
            }
        }

        let Some(&(and_idx, Rvalue::BinaryOp(MirBinaryOp::And, lhs, rhs))) =
            last_def.get(&cond_local)
        else {
            return;
        };
        let Some(lhs_loc) = operand_exact_local(lhs) else {
            return;
        };
        let Some(rhs_loc) = operand_exact_local(rhs) else {
            return;
        };
        let Some(&(lhs_idx, lhs_rval)) = last_def.get(&lhs_loc) else {
            return;
        };
        let Some(&(rhs_idx, rhs_rval)) = last_def.get(&rhs_loc) else {
            return;
        };

        // Find optional len_tmp
        let mut to_skip = vec![(and_idx, cond_local), (lhs_idx, lhs_loc), (rhs_idx, rhs_loc)];
        for rval in [lhs_rval, rhs_rval] {
            if let Rvalue::BinaryOp(MirBinaryOp::Lt | MirBinaryOp::Gt, a, b) = rval {
                for op in [a, b] {
                    if let Some(loc) = operand_exact_local(op)
                        && let Some(&(lidx, Rvalue::UnaryOp(MirUnaryOp::Len, _))) =
                            last_def.get(&loc)
                    {
                        to_skip.push((lidx, loc));
                    }
                }
            }
        }

        for (sidx, loc) in to_skip {
            if self.local_use_counts.get(&loc).copied().unwrap_or(0) <= 1 {
                self.skip_stmts.insert((header_block, sidx));
                self.omitted_locals.insert(loc);
            }
        }
    }

    /// Detects `if node.left is Some(l)` lowered by `AstLowering::lower_condition_with_bindings`:
    ///   cond_block:
    ///     _opt_tmp: Option<Struct(S)> = Use(Copy/Move(src_place))
    ///     _cond_tmp: Bool = BinaryOp(NotEq, Copy/Move(_opt_tmp), Constant(Int(0)))
    ///   then_block:
    ///     _binding: Struct(S) = Cast(Copy/Move(_opt_tmp), Struct(S))
    fn try_detect_if_option_some(
        &mut self,
        cond_block: BlockId,
        cond_op: &Operand,
        then_branch: &StructuredNode,
    ) {
        let Some(cond_local) = operand_exact_local(cond_op) else {
            return;
        };
        let stmts = &self.func.blocks[cond_block.0].statements;
        let mut cond_stmt_idx = None;
        let mut opt_local = None;
        let mut is_negated = false;

        for (idx, stmt) in stmts.iter().enumerate() {
            if let Statement::Assign(
                Place::Local(dst),
                Rvalue::BinaryOp(op @ (MirBinaryOp::NotEq | MirBinaryOp::Eq), lhs, rhs),
                _,
            ) = stmt
                && *dst == cond_local
                && matches!(rhs, Operand::Constant(MirConstant::Int(0)))
                && let Some(opt_loc) = operand_exact_local(lhs)
                && matches!(self.func.locals[opt_loc.0].ty, MirType::Option(_))
            {
                cond_stmt_idx = Some(idx);
                opt_local = Some(opt_loc);
                is_negated = *op == MirBinaryOp::Eq;
            }
        }

        let (Some(c_idx), Some(opt_loc)) = (cond_stmt_idx, opt_local) else {
            return;
        };
        if is_negated {
            return;
        }

        let struct_name = match &self.func.locals[opt_loc.0].ty {
            MirType::Option(inner) => match inner.as_ref() {
                MirType::Struct(s) if !s.is_empty() => s.clone(),
                _ => return,
            },
            _ => return,
        };

        // Find `_opt_tmp = Use(Copy/Move(src_place))` in cond_block
        let mut opt_assign_idx = None;
        let mut src_place = None;
        for (idx, stmt) in stmts.iter().enumerate() {
            if idx >= c_idx {
                break;
            }
            if let Statement::Assign(
                Place::Local(dst),
                Rvalue::Use(Operand::Copy(p) | Operand::Move(p)),
                _,
            ) = stmt
                && *dst == opt_loc
            {
                opt_assign_idx = Some(idx);
                src_place = Some(p.clone());
            }
        }
        let (Some(o_idx), Some(src_p)) = (opt_assign_idx, src_place) else {
            return;
        };

        // Find `_binding = Cast(Copy/Move(_opt_tmp), Struct(S))` at start of then_block
        let Some(then_bb) = leading_block_of(then_branch) else {
            return;
        };
        let then_stmts = &self.func.blocks[then_bb.0].statements;
        let mut binding_info = None;
        for (idx, stmt) in then_stmts.iter().enumerate() {
            match stmt {
                Statement::StorageLive(_, _) | Statement::ResourceCheck(_, _, _) => continue,
                Statement::Assign(
                    Place::Local(b_loc),
                    Rvalue::Cast(op, MirType::Struct(s)),
                    _,
                ) if s == &struct_name && operand_exact_local(op) == Some(opt_loc) => {
                    binding_info = Some((idx, *b_loc));
                    break;
                }
                _ => break,
            }
        }
        let Some((b_stmt_idx, binding_local)) = binding_info else {
            return;
        };

        // Ensure `_opt_tmp` is only used in `cond_stmt` and `binding_stmt` (use_count <= 2)
        if self.local_use_counts.get(&opt_loc).copied().unwrap_or(0) > 2 {
            return;
        }
        // Ensure `_binding` is read-only (never mutated or moved out)
        if param_is_reassigned_after(self.func, binding_local, then_bb, b_stmt_idx)
            || param_escapes_by_move(self.func, binding_local)
        {
            return;
        }

        self.skip_stmts.insert((cond_block, o_idx));
        self.skip_stmts.insert((cond_block, c_idx));
        self.skip_stmts.insert((then_bb, b_stmt_idx));
        self.omitted_locals.insert(opt_loc);
        self.omitted_locals.insert(cond_local);
        self.omitted_locals.insert(binding_local);
        self.scoped_ref_locals.insert(binding_local);
        self.if_option_some.insert(
            cond_block,
            IfOptionSomePlan {
                src_place: src_p,
                binding_local,
                struct_name,
            },
        );
    }

    // -----------------------------------------------------------------------
    // StructuredNode Emission
    // -----------------------------------------------------------------------

    fn emit_node(&mut self, node: &StructuredNode, indent: usize) -> Result<()> {
        let pad = "    ".repeat(indent);
        match node {
            StructuredNode::BasicBlock(bb) => {
                self.emit_block_statements(*bb, indent)?;
                if let Some(term) = &self.func.blocks[bb.0].terminator {
                    match term {
                        Terminator::Call {
                            callee,
                            args,
                            destination,
                            ..
                        } => {
                            if self.skip_calls.contains(bb) {
                                // Stripped duplicate inlined `defer` call (handled by DropGuard)
                            } else if let Some((dst_s, int_op, arg_ty)) =
                                self.fused_tostring_calls.get(bb).cloned()
                            {
                                self.emit_fused_string_append(dst_s, &int_op, &arg_ty, indent)?;
                            } else {
                                self.emit_call(callee, args, destination, indent)?;
                            }
                        }
                        Terminator::Raise(err_op, _) => {
                            let e_str = self.emit_operand_as_display_string(err_op);
                            let _ = writeln!(
                                self.out,
                                "{pad}return Err(lust_aot_rt::AotError::UserError({e_str}));"
                            );
                        }
                        Terminator::Unreachable(_) => {
                            if self.in_cleanup_closure.is_some() {
                                let _ = writeln!(self.out, "{pad}return Ok(());");
                            } else if self.meta.can_fail {
                                let _ = writeln!(
                                    self.out,
                                    "{pad}return Err(lust_aot_rt::AotError::UserError(alloc::string::String::from(\"unreachable\")));"
                                );
                            } else {
                                let def = self.ctx.rust_default_expr(&self.func.signature.return_type);
                                let _ = writeln!(self.out, "{pad}return Ok({def});");
                            }
                        }
                        Terminator::Goto(_, _) => {}
                        Terminator::Return(opt, _) => {
                            self.emit_return_expr(opt.as_ref(), indent);
                        }
                        Terminator::Cond { .. } | Terminator::Switch { .. } => {}
                    }
                }
            }
            StructuredNode::Seq(items) => {
                for item in items {
                    self.emit_node(item, indent)?;
                }
            }
            StructuredNode::If {
                cond_block,
                cond_op,
                then_branch,
                else_branch,
            } => {
                self.emit_block_statements(*cond_block, indent)?;
                if let Some(plan) = self.if_option_some.get(cond_block).cloned() {
                    let src_expr = self.emit_place_ref_expr(&plan.src_place);
                    let deref_expr = if self.ctx.struct_is_cyclic(&plan.struct_name) {
                        format!("{src_expr}.as_deref()")
                    } else {
                        format!("{src_expr}.as_ref()")
                    };
                    let _ = writeln!(
                        self.out,
                        "{pad}if let Some(_{}) = {deref_expr} {{",
                        plan.binding_local.0
                    );
                    self.emit_node(then_branch, indent + 1)?;
                    if let Some(eb) = else_branch {
                        let _ = writeln!(self.out, "{pad}}} else {{");
                        self.emit_node(eb, indent + 1)?;
                    }
                    let _ = writeln!(self.out, "{pad}}}");
                } else {
                    let cond_expr = self.emit_operand_bool(cond_op);
                    let _ = writeln!(self.out, "{pad}if {cond_expr} {{");
                    self.emit_node(then_branch, indent + 1)?;
                    if let Some(eb) = else_branch {
                        let _ = writeln!(self.out, "{pad}}} else {{");
                        self.emit_node(eb, indent + 1)?;
                    }
                    let _ = writeln!(self.out, "{pad}}}");
                }
            }
            StructuredNode::IfArrayGetOk {
                header_block,
                arr_local,
                idx_local,
                binding_local,
                then_branch,
                else_branch,
            } => {
                self.emit_block_statements(*header_block, indent)?;
                let arr_id = self.resolve_local(*arr_local);
                let idx_id = self.resolve_local(*idx_local);
                let arr_var = self.local_var_name(arr_id);
                let idx_expr = self.emit_local_read_scalar(idx_id);
                let elem_ty = match &self.func.locals[arr_id.0].ty {
                    MirType::Array(inner) => inner.as_ref().clone(),
                    _ => self.func.locals[binding_local.0].ty.clone(),
                };
                if self.local_storage(arr_id) == LocalStorage::RcRefCell {
                    // Drop the RefCell borrow BEFORE entering `then_branch` so mutations inside
                    // `then_branch` to `arr` or its aliases never cause re-entrant borrow conflicts.
                    let elem_extract = if self.ctx.is_copy_type(&elem_ty) {
                        ".copied()"
                    } else {
                        ".cloned()"
                    };
                    let _ = writeln!(
                        self.out,
                        "{pad}let __opt_elem = {{ let __arr = lust_aot_rt::ref_borrow(&{arr_var})?; usize::try_from({idx_expr}).ok().and_then(|__i| __arr.get(__i){elem_extract}) }};"
                    );
                    let _ = writeln!(self.out, "{pad}if let Some(__elem) = __opt_elem {{");
                    self.emit_assign_local_val(*binding_local, "__elem", indent + 1);
                } else if self.ctx.is_copy_type(&elem_ty) {
                    let _ = writeln!(
                        self.out,
                        "{pad}if let Some(&__elem) = usize::try_from({idx_expr}).ok().and_then(|__i| {arr_var}.get(__i)) {{"
                    );
                    self.emit_assign_local_val(*binding_local, "__elem", indent + 1);
                } else {
                    let _ = writeln!(
                        self.out,
                        "{pad}if let Some(__elem) = usize::try_from({idx_expr}).ok().and_then(|__i| {arr_var}.get(__i)) {{"
                    );
                    self.emit_assign_local_val(*binding_local, "__elem.clone()", indent + 1);
                }
                self.emit_node(then_branch, indent + 1)?;
                if let Some(eb) = else_branch {
                    let _ = writeln!(self.out, "{pad}}} else {{");
                    self.emit_node(eb, indent + 1)?;
                }
                let _ = writeln!(self.out, "{pad}}}");
            }
            StructuredNode::Switch {
                discr_block,
                discr_op,
                cases,
                default,
            } => {
                self.emit_block_statements(*discr_block, indent)?;
                let d_expr = self.emit_operand_value(discr_op);
                let matched_enum_local = operand_exact_local(discr_op)
                    .and_then(|tl| self.switch_tag_source.get(&tl).copied());
                let _ = writeln!(self.out, "{pad}match {d_expr} {{");
                for (val, body) in cases {
                    let _ = writeln!(self.out, "{pad}    {val}_i64 => {{");
                    let prev_variant = if let Some(eloc) = matched_enum_local
                        && *val >= 0
                    {
                        self.active_enum_variant.insert(eloc, *val as usize)
                    } else {
                        None
                    };
                    self.emit_node(body, indent + 2)?;
                    if let Some(eloc) = matched_enum_local {
                        if let Some(pv) = prev_variant {
                            self.active_enum_variant.insert(eloc, pv);
                        } else {
                            self.active_enum_variant.remove(&eloc);
                        }
                    }
                    let _ = writeln!(self.out, "{pad}    }}");
                }
                let _ = writeln!(self.out, "{pad}    _ => {{");
                self.emit_node(default, indent + 2)?;
                let _ = writeln!(self.out, "{pad}    }}");
                let _ = writeln!(self.out, "{pad}}}");
            }
            StructuredNode::WhileLoop {
                label,
                header_block,
                cond_op,
                negate_cond,
                body,
            } => {
                let _ = writeln!(self.out, "{pad}'bb{}: loop {{", label.0);
                self.emit_block_statements(*header_block, indent + 1)?;
                let cond_expr = self.emit_operand_bool(cond_op);
                if *negate_cond {
                    let _ = writeln!(self.out, "{pad}    if {cond_expr} {{ break 'bb{}; }}", label.0);
                } else {
                    let _ = writeln!(self.out, "{pad}    if !({cond_expr}) {{ break 'bb{}; }}", label.0);
                }
                self.emit_node(body, indent + 1)?;
                let _ = writeln!(self.out, "{pad}}}");
            }
            StructuredNode::Loop { label, body } => {
                let _ = writeln!(self.out, "{pad}'bb{}: loop {{", label.0);
                self.emit_node(body, indent + 1)?;
                let _ = writeln!(self.out, "{pad}}}");
            }
            StructuredNode::LabeledBlock { label, body } => {
                let _ = writeln!(self.out, "{pad}'bb{}: {{", label.0);
                self.emit_node(body, indent + 1)?;
                let _ = writeln!(self.out, "{pad}}}");
            }
            StructuredNode::Break(label) => {
                let _ = writeln!(self.out, "{pad}break 'bb{};", label.0);
            }
            StructuredNode::Continue(label) => {
                let _ = writeln!(self.out, "{pad}continue 'bb{};", label.0);
            }
            StructuredNode::Return(bb, opt) => {
                self.emit_block_statements(*bb, indent)?;
                self.emit_return_expr(opt.as_ref(), indent);
            }
            StructuredNode::StateMachineFallback {
                entry,
                blocks,
                exit,
            } => {
                self.emit_state_machine_fallback(*entry, blocks, *exit, indent)?;
            }
        }
        Ok(())
    }

    fn emit_state_machine_fallback(
        &mut self,
        entry: BlockId,
        blocks: &[BlockId],
        exit: Option<BlockId>,
        indent: usize,
    ) -> Result<()> {
        let pad = "    ".repeat(indent);
        let _ = writeln!(self.out, "{pad}let mut __sm_bb: usize = {};", entry.0);
        let _ = writeln!(self.out, "{pad}'__sm_loop: loop {{");
        let _ = writeln!(self.out, "{pad}    match __sm_bb {{");
        for &bb in blocks {
            let _ = writeln!(self.out, "{pad}        {} => {{", bb.0);
            self.emit_block_statements(bb, indent + 3)?;
            if let Some(term) = &self.func.blocks[bb.0].terminator {
                match term {
                    Terminator::Goto(target, _) => {
                        if Some(*target) == exit {
                            let _ = writeln!(self.out, "{pad}            break '__sm_loop;");
                        } else {
                            let _ = writeln!(self.out, "{pad}            __sm_bb = {};", target.0);
                        }
                    }
                    Terminator::Cond {
                        cond,
                        then_block,
                        else_block,
                        ..
                    } => {
                        let c = self.emit_operand_bool(cond);
                        let _ = writeln!(self.out, "{pad}            if {c} {{");
                        if Some(*then_block) == exit {
                            let _ = writeln!(self.out, "{pad}                break '__sm_loop;");
                        } else {
                            let _ = writeln!(
                                self.out,
                                "{pad}                __sm_bb = {};",
                                then_block.0
                            );
                        }
                        let _ = writeln!(self.out, "{pad}            }} else {{");
                        if Some(*else_block) == exit {
                            let _ = writeln!(self.out, "{pad}                break '__sm_loop;");
                        } else {
                            let _ = writeln!(
                                self.out,
                                "{pad}                __sm_bb = {};",
                                else_block.0
                            );
                        }
                        let _ = writeln!(self.out, "{pad}            }}");
                    }
                    Terminator::Call {
                        callee,
                        args,
                        destination,
                        success,
                        ..
                    } => {
                        self.emit_call(callee, args, destination, indent + 3)?;
                        if Some(*success) == exit {
                            let _ = writeln!(self.out, "{pad}            break '__sm_loop;");
                        } else {
                            let _ =
                                writeln!(self.out, "{pad}            __sm_bb = {};", success.0);
                        }
                    }
                    Terminator::Return(opt, _) => {
                        self.emit_return_expr(opt.as_ref(), indent + 3);
                    }
                    Terminator::Raise(err_op, _) => {
                        let e_str = self.emit_operand_as_display_string(err_op);
                        let _ = writeln!(
                            self.out,
                            "{pad}            return Err(lust_aot_rt::AotError::UserError({e_str}));"
                        );
                    }
                    Terminator::Switch {
                        discr,
                        cases,
                        default,
                        ..
                    } => {
                        let d = self.emit_operand_value(discr);
                        let _ = writeln!(self.out, "{pad}            __sm_bb = match {d} {{");
                        for (v, t) in cases {
                            let _ = writeln!(self.out, "{pad}                {v}_i64 => {},", t.0);
                        }
                        let _ = writeln!(self.out, "{pad}                _ => {},", default.0);
                        let _ = writeln!(self.out, "{pad}            }};");
                        if let Some(ex) = exit {
                            let _ = writeln!(
                                self.out,
                                "{pad}            if __sm_bb == {} {{ break '__sm_loop; }}",
                                ex.0
                            );
                        }
                    }
                    Terminator::Unreachable(_) => {
                        let _ = writeln!(self.out, "{pad}            break '__sm_loop;");
                    }
                }
            } else {
                let _ = writeln!(self.out, "{pad}            break '__sm_loop;");
            }
            let _ = writeln!(self.out, "{pad}        }}");
        }
        let _ = writeln!(self.out, "{pad}        _ => break '__sm_loop,");
        let _ = writeln!(self.out, "{pad}    }}");
        let _ = writeln!(self.out, "{pad}}}");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Statements & Expressions
    // -----------------------------------------------------------------------

    fn emit_block_statements(&mut self, bb: BlockId, indent: usize) -> Result<()> {
        let block = &self.func.blocks[bb.0];
        for (sidx, stmt) in block.statements.iter().enumerate() {
            if self.skip_stmts.contains(&(bb, sidx)) {
                continue;
            }
            self.emit_statement(stmt, indent)?;
        }
        Ok(())
    }

    fn emit_statement(&mut self, stmt: &Statement, indent: usize) -> Result<()> {
        let pad = "    ".repeat(indent);
        match stmt {
            Statement::StorageLive(_, _) | Statement::StorageDead(_, _) => {}
            Statement::ResourceCheck(ResourceKind::Gas, amt, _) => {
                if self.ctx.opts.gas_metering != GasMetering::None && self.in_cleanup_closure.is_none()
                {
                    let _ = writeln!(self.out, "{pad}__ctx.charge_gas({amt}_i64)?;");
                }
            }
            Statement::ResourceCheck(ResourceKind::Memory, amt, _) => {
                if self.ctx.opts.max_memory_bytes.is_some() && self.in_cleanup_closure.is_none() {
                    let _ = writeln!(
                        self.out,
                        "{pad}__alloc.try_charge({amt}_usize).map_err(lust_aot_rt::AotError::from)?;"
                    );
                }
            }
            Statement::Assign(place, rval, _) => {
                // Skip writes to omitted locals
                if let Place::Local(dst) = place
                    && self.omitted_locals.contains(dst)
                {
                    return Ok(());
                }

                // Special case: indexed write `base[idx] = val` (Array, RcRefCell Array, Map, AotValue)
                if let Place::Index(base, idx) = place {
                    let base_ty = self.place_mir_type(base);
                    let val_expr = self.emit_rvalue(rval, Some(&self.place_mir_type(place)))?;
                    match base_ty {
                        MirType::Map(k_ty, _) => {
                            let map_expr = self.emit_place_value_expr(base);
                            let key_expr = self.emit_local_as_map_key(*idx, &k_ty);
                            let _ = writeln!(
                                self.out,
                                "{pad}{{ let __val = {val_expr}; {map_expr}.set({key_expr}, __val, __alloc)?; }}"
                            );
                        }
                        MirType::Unknown => {
                            let base_expr = self.emit_place_value_expr(base);
                            let idx_ty = self.func.locals[self.resolve_local(*idx).0].ty.clone();
                            let idx_expr = self.emit_local_read_scalar(*idx);
                            if matches!(idx_ty, MirType::Int) {
                                let _ = writeln!(
                                    self.out,
                                    "{pad}{{ let __val = {val_expr}; {base_expr}.array_set({idx_expr}, __val)?; }}"
                                );
                            } else {
                                let key_expr = self.emit_local_as_map_key(*idx, &MirType::Unknown);
                                let _ = writeln!(
                                    self.out,
                                    "{pad}{{ let __val = {val_expr}; {base_expr}.map_set({key_expr}, __val, __ctx)?; }}"
                                );
                            }
                        }
                        _ => {
                            let idx_expr = self.emit_local_read_scalar(*idx);
                            if let Place::Local(base_loc) = base.as_ref()
                                && self.local_storage(*base_loc) == LocalStorage::RcRefCell
                            {
                                let arr_var = self.local_var_name(*base_loc);
                                let _ = writeln!(self.out, "{pad}{{");
                                let _ = writeln!(self.out, "{pad}    let __val = {val_expr};");
                                let _ = writeln!(
                                    self.out,
                                    "{pad}    let __idx = {idx_expr};"
                                );
                                let _ = writeln!(
                                    self.out,
                                    "{pad}    let mut __arr = lust_aot_rt::ref_borrow_mut(&{arr_var})?;"
                                );
                                let _ = writeln!(self.out, "{pad}    let __len = __arr.len();");
                                let _ = writeln!(
                                    self.out,
                                    "{pad}    match usize::try_from(__idx).ok().and_then(|__i| __arr.get_mut(__i)) {{"
                                );
                                let _ = writeln!(
                                    self.out,
                                    "{pad}        Some(__slot) => *__slot = __val,"
                                );
                                let _ = writeln!(
                                    self.out,
                                    "{pad}        None => return Err(lust_aot_rt::AotError::IndexOutOfBounds {{ index: __idx, len: __len }}),"
                                );
                                let _ = writeln!(self.out, "{pad}    }}");
                                let _ = writeln!(self.out, "{pad}}}");
                            } else {
                                let arr_expr = self.emit_place_mut_expr(base);
                                let _ = writeln!(
                                    self.out,
                                    "{pad}match usize::try_from({idx_expr}).ok().and_then(|__i| {arr_expr}.get_mut(__i)) {{"
                                );
                                let _ = writeln!(
                                    self.out,
                                    "{pad}    Some(__slot) => *__slot = {val_expr},"
                                );
                                let _ = writeln!(
                                    self.out,
                                    "{pad}    None => return Err(lust_aot_rt::AotError::IndexOutOfBounds {{ index: {idx_expr}, len: {arr_expr}.len() }}),"
                                );
                                let _ = writeln!(self.out, "{pad}}}");
                            }
                        }
                    }
                    return Ok(());
                }

                // Special case: field write `s.field_i = val` on `RcRefCell` struct
                if let Place::Field(base, fidx) = place
                    && let Place::Local(base_loc) = base.as_ref()
                    && self.local_storage(*base_loc) == LocalStorage::RcRefCell
                {
                    let target_ty = self.place_mir_type(place);
                    let val_expr = self.emit_rvalue(rval, Some(&target_ty))?;
                    let base_var = self.local_var_name(*base_loc);
                    let _ = writeln!(
                        self.out,
                        "{pad}{{ let __val = {val_expr}; lust_aot_rt::ref_borrow_mut(&{base_var})?.field_{fidx} = __val; }}"
                    );
                    return Ok(());
                }

                // Special case: in-place string append `s = s .. rhs` (Direct storage only)
                if let Place::Local(dst) = place
                    && self.local_storage(*dst) == LocalStorage::Direct
                    && let Rvalue::BinaryOp(MirBinaryOp::Concat, lhs, rhs) = rval
                    && operand_exact_local(lhs).map(|l| self.resolve_local(l))
                        == Some(self.resolve_local(*dst))
                    && matches!(self.func.locals[self.resolve_local(*dst).0].ty, MirType::String)
                {
                    let dst_var = self.local_var_name(*dst);
                    let rhs_str_ref = self.emit_operand_str_ref(rhs);
                    if self.ctx.opts.max_memory_bytes.is_some() {
                        let _ = writeln!(
                            self.out,
                            "{pad}{{ let __add = {rhs_str_ref}; __alloc.try_charge(__add.len()).map_err(lust_aot_rt::AotError::from)?; {dst_var}.push_str(__add); }}"
                        );
                    } else {
                        let _ = writeln!(self.out, "{pad}{dst_var}.push_str({rhs_str_ref});");
                    }
                    return Ok(());
                }

                // Special case: checked indexed read `dst = base[idx]` (Array, RcRefCell Array, Map, AotValue)
                if let Rvalue::Use(
                    Operand::Copy(Place::Index(base, idx)) | Operand::Move(Place::Index(base, idx)),
                ) = rval
                {
                    let base_ty = self.place_mir_type(base);
                    let dst_ty = self.place_mir_type(place);
                    let val_expr = match base_ty {
                        MirType::Map(k_ty, v_ty) => {
                            let map_expr = self.emit_place_value_expr(base);
                            let key_expr = self.emit_local_as_map_key(*idx, &k_ty);
                            let raw = format!("{map_expr}.get(&{key_expr})?");
                            self.coerce_expr_to_type(&raw, &v_ty, &dst_ty)
                        }
                        MirType::Unknown => {
                            let base_expr = self.emit_place_value_expr(base);
                            let idx_ty = self.func.locals[self.resolve_local(*idx).0].ty.clone();
                            let raw = if matches!(idx_ty, MirType::Int) {
                                let idx_expr = self.emit_local_read_scalar(*idx);
                                format!("{base_expr}.array_get({idx_expr})?")
                            } else {
                                let key_expr = self.emit_local_as_map_key(*idx, &MirType::Unknown);
                                format!("{base_expr}.map_get(&{key_expr})?")
                            };
                            self.coerce_expr_to_type(&raw, &MirType::Unknown, &dst_ty)
                        }
                        _ => {
                            let idx_expr = self.emit_local_read_scalar(*idx);
                            let bind = if self.ctx.is_copy_type(&dst_ty) {
                                "*__v"
                            } else {
                                "__v.clone()"
                            };
                            if let Place::Local(base_loc) = base.as_ref()
                                && self.local_storage(*base_loc) == LocalStorage::RcRefCell
                            {
                                let arr_var = self.local_var_name(*base_loc);
                                format!(
                                    "{{ let __idx = {idx_expr}; let __arr = lust_aot_rt::ref_borrow(&{arr_var})?; match usize::try_from(__idx).ok().and_then(|__i| __arr.get(__i)) {{ Some(__v) => {bind}, None => return Err(lust_aot_rt::AotError::IndexOutOfBounds {{ index: __idx, len: __arr.len() }}) }} }}"
                                )
                            } else {
                                let arr_expr = self.emit_place_ref_expr(base);
                                format!(
                                    "match usize::try_from({idx_expr}).ok().and_then(|__i| {arr_expr}.get(__i)) {{ Some(__v) => {bind}, None => return Err(lust_aot_rt::AotError::IndexOutOfBounds {{ index: {idx_expr}, len: {arr_expr}.len() }}) }}"
                                )
                            }
                        }
                    };
                    self.emit_assign_place_val(place, &val_expr, indent);
                    return Ok(());
                }

                // Special case: Array literal `dst = Aggregate(Array(T), elems)`
                if let Rvalue::Aggregate(MirType::Array(elem_ty), elems) = rval {
                    let rust_elem = self.ctx.rust_type(elem_ty, "'a");
                    let vec_expr = if elems.is_empty() {
                        format!("alloc::vec::Vec::<{rust_elem}, _>::new_in(__alloc)")
                    } else {
                        let mut blk = String::new();
                        let _ = writeln!(blk, "{{");
                        let _ = writeln!(
                            blk,
                            "{pad}    let mut __v: alloc::vec::Vec<{rust_elem}, _> = alloc::vec::Vec::new_in(__alloc);"
                        );
                        if self.ctx.opts.max_memory_bytes.is_some() {
                            let _ = writeln!(
                                blk,
                                "{pad}    __v.try_reserve({}).map_err(|_| lust_aot_rt::AotError::from_alloc_failure(__alloc, {} * core::mem::size_of::<{rust_elem}>()))?;",
                                elems.len(),
                                elems.len()
                            );
                        } else {
                            let _ = writeln!(blk, "{pad}    __v.reserve({});", elems.len());
                        }
                        for elem_op in elems {
                            let e_expr = self.emit_operand_owned(elem_op, elem_ty);
                            let _ = writeln!(blk, "{pad}    __v.push({e_expr});");
                        }
                        let _ = write!(blk, "{pad}    __v\n{pad}}}");
                        blk
                    };
                    self.emit_assign_place_val(place, &vec_expr, indent);
                    return Ok(());
                }

                // Special case: Local alias copy `dst = Copy/Move(src)` where BOTH are RcRefCell (Hazard 12)
                if let Place::Local(dst) = place
                    && self.local_storage(*dst) == LocalStorage::RcRefCell
                    && let Rvalue::Use(
                        Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)),
                    ) = rval
                    && self.local_storage(*src) == LocalStorage::RcRefCell
                    && self.func.locals[self.resolve_local(*dst).0].ty
                        == self.func.locals[self.resolve_local(*src).0].ty
                {
                    let dst_var = self.local_var_name(*dst);
                    let src_var = self.local_var_name(*src);
                    let _ = writeln!(self.out, "{pad}{dst_var} = alloc::rc::Rc::clone(&{src_var});");
                    return Ok(());
                }

                let target_ty = self.place_mir_type(place);
                let rhs_expr = self.emit_rvalue(rval, Some(&target_ty))?;
                self.emit_assign_place_val(place, &rhs_expr, indent);
            }
        }
        Ok(())
    }

    fn emit_assign_place_val(&mut self, place: &Place, val_expr: &str, indent: usize) {
        let pad = "    ".repeat(indent);
        match place {
            Place::Local(dst) => self.emit_assign_local_val(*dst, val_expr, indent),
            _ => {
                let dst_expr = self.emit_place_lhs(place);
                let _ = writeln!(self.out, "{pad}{dst_expr} = {val_expr};");
            }
        }
    }

    fn emit_assign_local_val(&mut self, dst: LocalId, val_expr: &str, indent: usize) {
        let pad = "    ".repeat(indent);
        let r = self.resolve_local(dst);
        let var = self.local_var_name(r);
        match self.local_storage(r) {
            LocalStorage::Direct => {
                let _ = writeln!(self.out, "{pad}{var} = {val_expr};");
            }
            LocalStorage::RcCell => {
                let _ = writeln!(self.out, "{pad}{var}.set({val_expr});");
            }
            LocalStorage::RcRefCell => {
                let is_single_init_cleanup = self.meta.cleanup_captured.contains(&r)
                    && self.local_def_counts.get(&r).copied().unwrap_or(0) <= 1;
                if is_single_init_cleanup || self.in_cleanup_closure.is_some() {
                    let _ = writeln!(
                        self.out,
                        "{pad}{{ let __val = {val_expr}; *lust_aot_rt::ref_borrow_mut(&{var})? = __val; }}"
                    );
                } else {
                    let _ = writeln!(
                        self.out,
                        "{pad}{{ let __val = {val_expr}; {var} = alloc::rc::Rc::new(core::cell::RefCell::new(__val)); }}"
                    );
                }
            }
        }
    }

    fn emit_fused_string_append(
        &mut self,
        dst_s: LocalId,
        arg_op: &Operand,
        arg_ty: &MirType,
        indent: usize,
    ) -> Result<()> {
        let pad = "    ".repeat(indent);
        let s_var = self.local_var_name(dst_s);
        let arg_expr = self.emit_operand_value(arg_op);
        match arg_ty {
            MirType::Int => {
                if self.ctx.opts.max_memory_bytes.is_some() {
                    let _ = writeln!(
                        self.out,
                        "{pad}lust_aot_rt::try_string_append_int(&mut {s_var}, {arg_expr}, __alloc)?;"
                    );
                } else {
                    let _ = writeln!(
                        self.out,
                        "{pad}lust_aot_rt::string_append_int(&mut {s_var}, {arg_expr});"
                    );
                }
            }
            MirType::Float => {
                let _ = writeln!(
                    self.out,
                    "{pad}lust_aot_rt::string_append_float(&mut {s_var}, {arg_expr});"
                );
            }
            _ => {}
        }
        Ok(())
    }

    fn emit_rvalue(&self, rval: &Rvalue, expected_ty: Option<&MirType>) -> Result<String> {
        match rval {
            Rvalue::Use(op) => {
                if let Some(ty) = expected_ty {
                    Ok(self.emit_operand_owned(op, ty))
                } else {
                    Ok(self.emit_operand_value(op))
                }
            }
            Rvalue::BinaryOp(op, lhs, rhs) => {
                let raw = self.emit_binop(*op, lhs, rhs)?;
                let l_ty = operand_mir_type(lhs, self.func);
                let r_ty = operand_mir_type(rhs, self.func);
                if (matches!(l_ty, MirType::Unknown) || matches!(r_ty, MirType::Unknown))
                    && !matches!(
                        op,
                        MirBinaryOp::Eq
                            | MirBinaryOp::NotEq
                            | MirBinaryOp::Lt
                            | MirBinaryOp::LtEq
                            | MirBinaryOp::Gt
                            | MirBinaryOp::GtEq
                            | MirBinaryOp::And
                            | MirBinaryOp::Or
                    )
                    && let Some(exp) = expected_ty
                    && !matches!(exp, MirType::Unknown)
                {
                    Ok(self.coerce_expr_to_type(&raw, &MirType::Unknown, exp))
                } else {
                    Ok(raw)
                }
            }
            Rvalue::UnaryOp(op, operand) => {
                let raw = self.emit_unop(*op, operand)?;
                let o_ty = operand_mir_type(operand, self.func);
                if matches!(o_ty, MirType::Unknown)
                    && matches!(op, MirUnaryOp::Neg | MirUnaryOp::BitNot)
                    && let Some(exp) = expected_ty
                    && !matches!(exp, MirType::Unknown)
                {
                    Ok(self.coerce_expr_to_type(&raw, &MirType::Unknown, exp))
                } else {
                    Ok(raw)
                }
            }
            Rvalue::Cast(operand, target_ty) => self.emit_cast(operand, target_ty),
            Rvalue::Aggregate(ty, operands) => match ty {
                MirType::Struct(sname) => {
                    let rust_sname = rust_struct_name(sname);
                    let meta = self.ctx.structs.get(sname);
                    let mut fields = Vec::with_capacity(operands.len());
                    for (i, op) in operands.iter().enumerate() {
                        let fty = meta.and_then(|m| m.fields.get(i).map(|(_, t)| t));
                        let val_expr = if let Some(ft) = fty {
                            self.emit_operand_owned(op, ft)
                        } else {
                            self.emit_operand_value(op)
                        };
                        fields.push(format!("field_{i}: {val_expr}"));
                    }
                    Ok(format!("{rust_sname} {{ {} }}", fields.join(", ")))
                }
                MirType::Enum(ename) => self.emit_enum_aggregate(ename, operands),
                MirType::Tuple(elem_tys) => {
                    let mut parts = Vec::with_capacity(operands.len());
                    for (i, op) in operands.iter().enumerate() {
                        let val = if let Some(et) = elem_tys.get(i) {
                            self.emit_operand_owned(op, et)
                        } else {
                            self.emit_operand_value(op)
                        };
                        parts.push(val);
                    }
                    if parts.is_empty() {
                        Ok("()".to_string())
                    } else if parts.len() == 1 {
                        Ok(format!("({},)", parts[0]))
                    } else {
                        Ok(format!("({})", parts.join(", ")))
                    }
                }
                _ => Ok("()".to_string()),
            },
            Rvalue::Closure {
                func_name,
                captures,
            } => self.emit_closure_rvalue(func_name, captures, expected_ty),
        }
    }

    fn emit_enum_aggregate(&self, ename: &str, operands: &[Operand]) -> Result<String> {
        if let Some(emeta) = self.ctx.enums.get(ename) {
            let rust_ename = rust_enum_name(ename);
            if let Some(Operand::Constant(MirConstant::Int(tag))) = operands.first()
                && *tag >= 0
                && let Some((vname, vfields)) = emeta.variants.get(*tag as usize)
            {
                let rv = sanitize_ident(vname);
                if vfields.is_empty() {
                    return Ok(format!("{rust_ename}::{rv}"));
                }
                let mut args = Vec::with_capacity(vfields.len());
                for (fi, fty) in vfields.iter().enumerate() {
                    if let Some(op) = operands.get(fi + 1) {
                        args.push(self.emit_operand_owned(op, fty));
                    } else {
                        args.push(self.ctx.rust_default_expr(fty));
                    }
                }
                return Ok(format!("{rust_ename}::{rv}({})", args.join(", ")));
            }
            // Dynamic tag fallback
            let tag_expr = operands
                .first()
                .map(|op| self.emit_operand_value(op))
                .unwrap_or_else(|| "0_i64".to_string());
            let mut arms = Vec::new();
            for (vidx, (vname, vfields)) in emeta.variants.iter().enumerate() {
                let rv = sanitize_ident(vname);
                if vfields.is_empty() {
                    arms.push(format!("{vidx}_i64 => {rust_ename}::{rv}"));
                } else {
                    let mut args = Vec::with_capacity(vfields.len());
                    for (fi, fty) in vfields.iter().enumerate() {
                        if let Some(op) = operands.get(fi + 1) {
                            args.push(self.emit_operand_owned(op, fty));
                        } else {
                            args.push(self.ctx.rust_default_expr(fty));
                        }
                    }
                    arms.push(format!(
                        "{vidx}_i64 => {rust_ename}::{rv}({})",
                        args.join(", ")
                    ));
                }
            }
            arms.push(
                "_ => return Err(lust_aot_rt::AotError::UserError(alloc::string::String::from(\"invalid enum tag\")))"
                    .to_string(),
            );
            return Ok(format!("match {tag_expr} {{ {} }}", arms.join(", ")));
        }
        // Unregistered dynamic enum -> AotValue::from_enum
        let tag_expr = operands
            .first()
            .map(|op| self.emit_operand_value(op))
            .unwrap_or_else(|| "0_i64".to_string());
        let payload_elems: Vec<String> = operands
            .iter()
            .skip(1)
            .map(|op| self.emit_operand_as_aot_value(op))
            .collect();
        Ok(format!(
            "lust_aot_rt::AotValue::from_enum(0, ({tag_expr}) as u32, alloc::vec![{}])",
            payload_elems.join(", ")
        ))
    }

    fn emit_closure_rvalue(
        &self,
        func_name: &str,
        captures: &[(String, Operand, MirType)],
        expected_ty: Option<&MirType>,
    ) -> Result<String> {
        let Some(lifted_meta) = self.ctx.fn_metas.get(func_name) else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: missing lifted closure function '{func_name}'"
            )));
        };
        let Some(lifted_fn) = self
            .ctx
            .module
            .functions
            .iter()
            .find(|f| f.signature.name == func_name)
        else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: missing lifted closure MirFunction '{func_name}'"
            )));
        };

        let n_caps = captures.len();
        let explicit_params: Vec<MirType> = lifted_fn
            .params
            .iter()
            .skip(n_caps)
            .map(|pid| lifted_fn.locals[pid.0].ty.clone())
            .collect();
        let ret_ty = lifted_fn.signature.return_type.clone();
        let fn_mir_ty = MirType::Function {
            params: explicit_params.clone(),
            return_type: Box::new(ret_ty.clone()),
        };

        // Bind captured upvalues before `move |...|`
        let mut bindings = Vec::with_capacity(n_caps);
        let mut cap_call_exprs = Vec::with_capacity(n_caps);
        for (ci, (_, cap_op, cap_ty)) in captures.iter().enumerate() {
            let lmode = lifted_meta
                .param_modes
                .get(ci)
                .copied()
                .unwrap_or(ParamMode::ByValue);
            match lmode {
                ParamMode::RcCell | ParamMode::RcRefCell => {
                    if let Some(loc) = operand_exact_local(cap_op)
                        && self.local_storage(loc) != LocalStorage::Direct
                    {
                        let var = self.local_var_name(loc);
                        bindings.push(format!("let __cap_{ci} = alloc::rc::Rc::clone(&{var});"));
                    } else {
                        let v = self.emit_operand_owned(cap_op, cap_ty);
                        if lmode == ParamMode::RcCell {
                            bindings.push(format!(
                                "let __cap_{ci} = alloc::rc::Rc::new(core::cell::Cell::new({v}));"
                            ));
                        } else {
                            bindings.push(format!(
                                "let __cap_{ci} = alloc::rc::Rc::new(core::cell::RefCell::new({v}));"
                            ));
                        }
                    }
                    cap_call_exprs.push(format!("alloc::rc::Rc::clone(&__cap_{ci})"));
                }
                ParamMode::RefStruct => {
                    let v = self.emit_operand_owned(cap_op, cap_ty);
                    bindings.push(format!("let __cap_{ci} = {v};"));
                    cap_call_exprs.push(format!("&__cap_{ci}"));
                }
                ParamMode::RefSlice => {
                    let v = self.emit_operand_owned(cap_op, cap_ty);
                    bindings.push(format!("let __cap_{ci} = {v};"));
                    cap_call_exprs.push(format!("&__cap_{ci}[..]"));
                }
                ParamMode::RefStr => {
                    let v = self.emit_operand_owned(cap_op, cap_ty);
                    bindings.push(format!("let __cap_{ci} = {v};"));
                    cap_call_exprs.push(format!("__cap_{ci}.as_str()"));
                }
                _ => {
                    let v = self.emit_operand_owned(cap_op, cap_ty);
                    bindings.push(format!("let __cap_{ci} = {v};"));
                    if self.ctx.is_copy_type(cap_ty) {
                        cap_call_exprs.push(format!("__cap_{ci}"));
                    } else {
                        cap_call_exprs.push(format!("__cap_{ci}.clone()"));
                    }
                }
            }
        }

        // Dynamic AotValue::Closure target
        if matches!(expected_ty, Some(MirType::Unknown)) {
            let mut call_args = Vec::new();
            if lifted_meta.needs_ctx {
                call_args.push("__c_ctx".to_string());
            }
            if lifted_meta.needs_alloc {
                call_args.push("&__c_alloc".to_string());
            }
            if lifted_meta.needs_arena {
                call_args.push("__arena".to_string());
            }
            call_args.extend(cap_call_exprs);
            for (ai, pty) in explicit_params.iter().enumerate() {
                let raw_a = format!(
                    "__c_args.get({ai}).cloned().unwrap_or(lust_aot_rt::AotValue::Nil)"
                );
                call_args.push(self.coerce_expr_to_type(&raw_a, &MirType::Unknown, pty));
            }
            let raw_call = format!("{}({})", lifted_meta.rust_name, call_args.join(", "));
            let ret_expr = self.coerce_expr_to_type("__res", &ret_ty, &MirType::Unknown);
            let alloc_setup = if lifted_meta.needs_alloc {
                "let __c_alloc = lust_aot_rt::LustBudgetAlloc::unbounded(); "
            } else {
                ""
            };
            return Ok(format!(
                "{{ {} lust_aot_rt::AotValue::closure(move |__c_ctx: &mut lust_aot_rt::AotCtx, __c_args: &[lust_aot_rt::AotValue]| -> Result<lust_aot_rt::AotValue, lust_aot_rt::AotError> {{ {alloc_setup}let __res = {raw_call}.map_err(lust_aot_rt::AotError::from)?; Ok({ret_expr}) }}) }}",
                bindings.join(" ")
            ));
        }

        // Typed closure: Rc<dyn Fn(&mut AotCtx, &LustBudgetAlloc, P0, ...) -> Result<R, AotError>>
        let needs_lt = self.ctx.type_needs_lifetime(&fn_mir_ty);
        let mut sig_params = Vec::with_capacity(explicit_params.len() + 2);
        sig_params.push("__c_ctx: &mut lust_aot_rt::AotCtx".to_string());
        if needs_lt {
            sig_params.push("__c_alloc: &'a lust_aot_rt::LustBudgetAlloc".to_string());
        } else {
            sig_params.push("__c_alloc: &lust_aot_rt::LustBudgetAlloc".to_string());
        }
        for (ai, pty) in explicit_params.iter().enumerate() {
            sig_params.push(format!("mut __arg_{ai}: {}", self.ctx.rust_type(pty, "'a")));
        }

        let mut call_args = Vec::new();
        if lifted_meta.needs_ctx {
            call_args.push("__c_ctx".to_string());
        }
        if lifted_meta.needs_alloc {
            call_args.push("__c_alloc".to_string());
        }
        if lifted_meta.needs_arena {
            call_args.push("__arena".to_string());
        }
        call_args.extend(cap_call_exprs);
        for (ai, pty) in explicit_params.iter().enumerate() {
            let lmode = lifted_meta
                .param_modes
                .get(n_caps + ai)
                .copied()
                .unwrap_or(ParamMode::ByValue);
            let a_expr = match lmode {
                ParamMode::ByValue => format!("__arg_{ai}"),
                ParamMode::RefStruct => format!("&__arg_{ai}"),
                ParamMode::MutRefStruct => format!("&mut __arg_{ai}"),
                ParamMode::RefSlice => format!("&__arg_{ai}[..]"),
                ParamMode::MutVec => format!("&mut __arg_{ai}"),
                ParamMode::RefStr => format!("__arg_{ai}.as_str()"),
                ParamMode::RcCell => {
                    format!("alloc::rc::Rc::new(core::cell::Cell::new(__arg_{ai}))")
                }
                ParamMode::RcRefCell => {
                    let _ = pty;
                    format!("alloc::rc::Rc::new(core::cell::RefCell::new(__arg_{ai}))")
                }
            };
            call_args.push(a_expr);
        }

        let r_rust = self.ctx.rust_type(&ret_ty, "'a");
        let fn_rust_ty = self.ctx.rust_type(&fn_mir_ty, "'a");
        let raw_call = format!("{}({})", lifted_meta.rust_name, call_args.join(", "));
        let body_call = if lifted_meta.can_fail {
            raw_call
        } else {
            format!("{raw_call}.map_err(lust_aot_rt::AotError::from)")
        };

        Ok(format!(
            "{{ {} (alloc::rc::Rc::new(move |{}| -> Result<{r_rust}, lust_aot_rt::AotError> {{ {body_call} }}) as {fn_rust_ty}) }}",
            bindings.join(" "),
            sig_params.join(", ")
        ))
    }

    fn emit_binop(&self, op: MirBinaryOp, lhs: &Operand, rhs: &Operand) -> Result<String> {
        let lhs_ty = operand_mir_type(lhs, self.func);
        let rhs_ty = operand_mir_type(rhs, self.func);

        // Dynamic AotValue fallback when either operand has Unknown type
        if matches!(lhs_ty, MirType::Unknown) || matches!(rhs_ty, MirType::Unknown) {
            let l = self.emit_operand_as_aot_value(lhs);
            let r = self.emit_operand_as_aot_value(rhs);
            let expr = match op {
                MirBinaryOp::Add => format!("({l}).add(&({r}))?"),
                MirBinaryOp::Sub => format!("({l}).sub(&({r}))?"),
                MirBinaryOp::Mul => format!("({l}).mul(&({r}))?"),
                MirBinaryOp::Div => format!("({l}).div(&({r}))?"),
                MirBinaryOp::Mod => format!("({l}).rem(&({r}))?"),
                MirBinaryOp::FloorDiv => format!("({l}).floordiv(&({r}))?"),
                MirBinaryOp::Pow => format!("({l}).pow(&({r}))?"),
                MirBinaryOp::Concat => format!("({l}).concat(&({r}), __alloc)?"),
                MirBinaryOp::Eq => format!("(({l}) == ({r}))"),
                MirBinaryOp::NotEq => format!("(({l}) != ({r}))"),
                MirBinaryOp::Lt => format!("({l}).cmp_lt(&({r}))?"),
                MirBinaryOp::LtEq => format!("({l}).cmp_le(&({r}))?"),
                MirBinaryOp::Gt => format!("({l}).cmp_gt(&({r}))?"),
                MirBinaryOp::GtEq => format!("({l}).cmp_ge(&({r}))?"),
                MirBinaryOp::And => format!("(({l}).is_truthy() && ({r}).is_truthy())"),
                MirBinaryOp::Or => format!("(({l}).is_truthy() || ({r}).is_truthy())"),
                MirBinaryOp::BitAnd => format!(
                    "lust_aot_rt::AotValue::Int(({l}).as_int().ok_or_else(|| lust_aot_rt::AotError::UserError(alloc::string::String::from(\"expected int\")))? & ({r}).as_int().ok_or_else(|| lust_aot_rt::AotError::UserError(alloc::string::String::from(\"expected int\")))?)"
                ),
                MirBinaryOp::BitOr => format!(
                    "lust_aot_rt::AotValue::Int(({l}).as_int().ok_or_else(|| lust_aot_rt::AotError::UserError(alloc::string::String::from(\"expected int\")))? | ({r}).as_int().ok_or_else(|| lust_aot_rt::AotError::UserError(alloc::string::String::from(\"expected int\")))?)"
                ),
                MirBinaryOp::BitXor => format!(
                    "lust_aot_rt::AotValue::Int(({l}).as_int().ok_or_else(|| lust_aot_rt::AotError::UserError(alloc::string::String::from(\"expected int\")))? ^ ({r}).as_int().ok_or_else(|| lust_aot_rt::AotError::UserError(alloc::string::String::from(\"expected int\")))?)"
                ),
                MirBinaryOp::Shl => format!(
                    "lust_aot_rt::AotValue::Int(({l}).as_int().ok_or_else(|| lust_aot_rt::AotError::UserError(alloc::string::String::from(\"expected int\")))?.wrapping_shl((({r}).as_int().ok_or_else(|| lust_aot_rt::AotError::UserError(alloc::string::String::from(\"expected int\")))? & 63) as u32))"
                ),
                MirBinaryOp::Shr => format!(
                    "lust_aot_rt::AotValue::Int(({l}).as_int().ok_or_else(|| lust_aot_rt::AotError::UserError(alloc::string::String::from(\"expected int\")))?.wrapping_shr((({r}).as_int().ok_or_else(|| lust_aot_rt::AotError::UserError(alloc::string::String::from(\"expected int\")))? & 63) as u32))"
                ),
            };
            return Ok(expr);
        }

        // Option comparison against 0 (null-pointer representation in MIR)
        if matches!(lhs_ty, MirType::Option(_))
            && matches!(rhs, Operand::Constant(MirConstant::Int(0)))
        {
            let l = self.emit_operand_ref(lhs);
            return match op {
                MirBinaryOp::Eq => Ok(format!("{l}.is_none()")),
                MirBinaryOp::NotEq => Ok(format!("{l}.is_some()")),
                _ => Err(LustError::Unknown(format!(
                    "Unsupported binary op {op:?} on Option"
                ))),
            };
        }

        // Enum / Map equality
        if matches!(lhs_ty, MirType::Enum(_) | MirType::Map(_, _))
            && matches!(op, MirBinaryOp::Eq | MirBinaryOp::NotEq)
        {
            let l = self.emit_operand_value(lhs);
            let r = self.emit_operand_value(rhs);
            return match op {
                MirBinaryOp::Eq => Ok(format!("(({l}) == ({r}))")),
                MirBinaryOp::NotEq => Ok(format!("(({l}) != ({r}))")),
                _ => unreachable!(),
            };
        }

        // String concatenation (non-in-place)
        if op == MirBinaryOp::Concat {
            let l_ref = self.emit_operand_str_ref(lhs);
            let r_ref = self.emit_operand_str_ref(rhs);
            if self.ctx.opts.max_memory_bytes.is_some() {
                return Ok(format!(
                    "{{ let __la = {l_ref}; let __lb = {r_ref}; let __len = __la.len().saturating_add(__lb.len()); __alloc.try_charge(__len).map_err(lust_aot_rt::AotError::from)?; let mut __s = alloc::string::String::with_capacity(__len); __s.push_str(__la); __s.push_str(__lb); __s }}"
                ));
            }
            return Ok(format!(
                "{{ let __la = {l_ref}; let __lb = {r_ref}; let mut __s = alloc::string::String::with_capacity(__la.len() + __lb.len()); __s.push_str(__la); __s.push_str(__lb); __s }}"
            ));
        }

        // String equality
        if matches!(lhs_ty, MirType::String) || matches!(rhs_ty, MirType::String) {
            let l = self.emit_operand_str_ref(lhs);
            let r = self.emit_operand_str_ref(rhs);
            return match op {
                MirBinaryOp::Eq => Ok(format!("({l} == {r})")),
                MirBinaryOp::NotEq => Ok(format!("({l} != {r})")),
                MirBinaryOp::Lt => Ok(format!("({l} < {r})")),
                MirBinaryOp::LtEq => Ok(format!("({l} <= {r})")),
                MirBinaryOp::Gt => Ok(format!("({l} > {r})")),
                MirBinaryOp::GtEq => Ok(format!("({l} >= {r})")),
                _ => Err(LustError::Unknown(format!(
                    "Unsupported string binary op {op:?}"
                ))),
            };
        }

        let is_float = matches!(lhs_ty, MirType::Float) || matches!(rhs_ty, MirType::Float);
        let l = if is_float && !matches!(lhs_ty, MirType::Float) {
            format!("({} as f64)", self.emit_operand_value(lhs))
        } else {
            self.emit_operand_value(lhs)
        };
        let r = if is_float && !matches!(rhs_ty, MirType::Float) {
            format!("({} as f64)", self.emit_operand_value(rhs))
        } else {
            self.emit_operand_value(rhs)
        };

        if is_float {
            let expr = match op {
                MirBinaryOp::Add => format!("({l} + {r})"),
                MirBinaryOp::Sub => format!("({l} - {r})"),
                MirBinaryOp::Mul => format!("({l} * {r})"),
                MirBinaryOp::Div => format!("({l} / {r})"),
                MirBinaryOp::Mod => format!("({l} % {r})"),
                MirBinaryOp::FloorDiv => format!("({l} / {r}).floor()"),
                MirBinaryOp::Pow => format!("({l}).powf({r})"),
                MirBinaryOp::Eq => format!("({l} == {r})"),
                MirBinaryOp::NotEq => format!("({l} != {r})"),
                MirBinaryOp::Lt => format!("({l} < {r})"),
                MirBinaryOp::LtEq => format!("({l} <= {r})"),
                MirBinaryOp::Gt => format!("({l} > {r})"),
                MirBinaryOp::GtEq => format!("({l} >= {r})"),
                _ => {
                    return Err(LustError::Unknown(format!(
                        "Unsupported float binary op {op:?}"
                    )));
                }
            };
            return Ok(expr);
        }

        let expr = match op {
            MirBinaryOp::Add => format!("({l}).wrapping_add({r})"),
            MirBinaryOp::Sub => format!("({l}).wrapping_sub({r})"),
            MirBinaryOp::Mul => format!("({l}).wrapping_mul({r})"),
            MirBinaryOp::Div => {
                if let Operand::Constant(MirConstant::Int(c)) = rhs
                    && *c != 0
                    && *c != -1
                {
                    format!("(({l}) / {c}_i64)")
                } else {
                    format!("lust_aot_rt::div_int({l}, {r})?")
                }
            }
            MirBinaryOp::Mod => {
                if let Operand::Constant(MirConstant::Int(c)) = rhs
                    && *c != 0
                    && *c != -1
                {
                    format!("(({l}) % {c}_i64)")
                } else {
                    format!("lust_aot_rt::rem_int({l}, {r})?")
                }
            }
            MirBinaryOp::FloorDiv => format!("lust_aot_rt::floordiv_int({l}, {r})?"),
            MirBinaryOp::Pow => format!("lust_aot_rt::pow_int({l}, {r})?"),
            MirBinaryOp::BitAnd => format!("({l} & {r})"),
            MirBinaryOp::BitOr => format!("({l} | {r})"),
            MirBinaryOp::BitXor => format!("({l} ^ {r})"),
            MirBinaryOp::Shl => format!("({l}).wrapping_shl((({r}) & 63) as u32)"),
            MirBinaryOp::Shr => format!("({l}).wrapping_shr((({r}) & 63) as u32)"),
            MirBinaryOp::Eq => format!("({l} == {r})"),
            MirBinaryOp::NotEq => format!("({l} != {r})"),
            MirBinaryOp::Lt => format!("({l} < {r})"),
            MirBinaryOp::LtEq => format!("({l} <= {r})"),
            MirBinaryOp::Gt => format!("({l} > {r})"),
            MirBinaryOp::GtEq => format!("({l} >= {r})"),
            MirBinaryOp::And => {
                let lb = self.emit_operand_bool(lhs);
                let rb = self.emit_operand_bool(rhs);
                format!("({lb} && {rb})")
            }
            MirBinaryOp::Or => {
                let lb = self.emit_operand_bool(lhs);
                let rb = self.emit_operand_bool(rhs);
                format!("({lb} || {rb})")
            }
            MirBinaryOp::Concat => unreachable!(),
        };
        Ok(expr)
    }

    fn emit_unop(&self, op: MirUnaryOp, operand: &Operand) -> Result<String> {
        let ty = operand_mir_type(operand, self.func);
        if matches!(ty, MirType::Unknown) {
            let v = self.emit_operand_value(operand);
            return match op {
                MirUnaryOp::Neg => Ok(format!("({v}).neg()?")),
                MirUnaryOp::Not => Ok(format!("(!({v}).is_truthy())")),
                MirUnaryOp::BitNot => Ok(format!(
                    "lust_aot_rt::AotValue::Int(!({v}).as_int().ok_or_else(|| lust_aot_rt::AotError::UserError(alloc::string::String::from(\"expected int\")))?)"
                )),
                MirUnaryOp::Len => Ok(format!("(({v}).len()? as i64)")),
            };
        }
        match op {
            MirUnaryOp::Neg => {
                let v = self.emit_operand_value(operand);
                if matches!(ty, MirType::Float) {
                    Ok(format!("(-({v}))"))
                } else {
                    Ok(format!("({v}).wrapping_neg()"))
                }
            }
            MirUnaryOp::Not => {
                let b = self.emit_operand_bool(operand);
                Ok(format!("(!({b}))"))
            }
            MirUnaryOp::BitNot => {
                let v = self.emit_operand_value(operand);
                Ok(format!("(!({v}))"))
            }
            MirUnaryOp::Len => {
                if matches!(ty, MirType::Map(_, _)) {
                    let m = self.emit_operand_value(operand);
                    Ok(format!("{m}.len()?"))
                } else if let Some(loc) = operand_exact_local(operand)
                    && self.local_storage(loc) == LocalStorage::RcRefCell
                {
                    let var = self.local_var_name(loc);
                    Ok(format!("(lust_aot_rt::ref_borrow(&{var})?.len() as i64)"))
                } else {
                    let r = self.emit_operand_ref(operand);
                    Ok(format!("({r}.len() as i64)"))
                }
            }
        }
    }

    fn emit_cast(&self, operand: &Operand, target_ty: &MirType) -> Result<String> {
        let src_ty = operand_mir_type(operand, self.func);

        // Coercion to/from AotValue (MirType::Unknown)
        if matches!(target_ty, MirType::Unknown) {
            return Ok(self.emit_operand_as_aot_value(operand));
        }
        if matches!(src_ty, MirType::Unknown) {
            let v = self.emit_operand_value(operand);
            return Ok(self.coerce_expr_to_type(&v, &MirType::Unknown, target_ty));
        }

        // Option<Struct> construction: Cast(0, Option<S>) -> None, Cast(s, Option<S>) -> Some(...)
        if let MirType::Option(inner) = target_ty {
            if matches!(operand, Operand::Constant(MirConstant::Int(0))) {
                return Ok("None".to_string());
            }
            if let MirType::Struct(sname) = inner.as_ref() {
                let val_expr = self.emit_operand_owned(operand, inner);
                if self.ctx.struct_is_cyclic(sname) {
                    let rname = rust_struct_name(sname);
                    if self.ctx.opts.max_memory_bytes.is_some() {
                        return Ok(format!(
                            "Some(alloc::boxed::Box::try_new_in({val_expr}, __arena).map_err(|_| lust_aot_rt::AotError::from_alloc_failure(__alloc, core::mem::size_of::<{rname}<'_>>()))?)"
                        ));
                    }
                    return Ok(format!(
                        "Some(alloc::boxed::Box::new_in({val_expr}, __arena))"
                    ));
                }
                return Ok(format!("Some({val_expr})"));
            }
        }

        if &src_ty == target_ty {
            return Ok(self.emit_operand_owned(operand, target_ty));
        }

        let v = self.emit_operand_value(operand);
        match (src_ty, target_ty) {
            (MirType::Int, MirType::Float) => Ok(format!("({v} as f64)")),
            (MirType::Float, MirType::Int) => Ok(format!("({v} as i64)")),
            (MirType::Int, MirType::Bool) => Ok(format!("({v} != 0)")),
            (MirType::Bool, MirType::Int) => Ok(format!("(if {v} {{ 1_i64 }} else {{ 0_i64 }})")),
            (MirType::Float, MirType::Bool) => Ok(format!("({v} != 0.0)")),
            _ => Ok(self.emit_operand_owned(operand, target_ty)),
        }
    }

    // -----------------------------------------------------------------------
    // Calls & Builtins
    // -----------------------------------------------------------------------

    fn emit_call(
        &mut self,
        callee: &str,
        args: &[Operand],
        destination: &Place,
        indent: usize,
    ) -> Result<()> {
        let pad = "    ".repeat(indent);

        if let Some(struct_name) = parse_mir_struct_decode_callee(callee) {
            let sname = rust_struct_name(struct_name);
            let arg_expr = args
                .first()
                .map(|a| self.emit_operand_str_ref(a))
                .unwrap_or_else(|| "\"\"".to_string());
            let call_expr = format!("{sname}::decode_json({arg_expr}, __ctx)?");
            if self.dest_is_omitted(destination) {
                let _ = writeln!(self.out, "{pad}let _ = {call_expr};");
            } else {
                self.emit_assign_place_val(destination, &call_expr, indent);
            }
            return Ok(());
        }

        if let Some(struct_name) = callee.strip_prefix("__lust_mir_struct_decode_binary:") {
            let sname = rust_struct_name(struct_name);
            let arg_expr = args
                .first()
                .map(|a| self.emit_operand_str_ref(a))
                .unwrap_or_else(|| "\"\"".to_string());
            let call_expr = format!("{sname}::decode_binary(({arg_expr}).as_bytes(), __ctx)?");
            if self.dest_is_omitted(destination) {
                let _ = writeln!(self.out, "{pad}let _ = {call_expr};");
            } else {
                self.emit_assign_place_val(destination, &call_expr, indent);
            }
            return Ok(());
        }

        match callee {
            "print" | "println" => {
                let is_ln = callee == "println";
                self.emit_print_call(args, is_ln, indent);
                return Ok(());
            }
            "assert" => {
                let cond_expr = args
                    .first()
                    .map(|a| self.emit_operand_bool(a))
                    .unwrap_or_else(|| "false".to_string());
                let msg_expr = args
                    .get(1)
                    .map(|m| self.emit_operand_as_display_string(m))
                    .unwrap_or_else(|| {
                        "alloc::string::String::from(\"assertion failed\")".to_string()
                    });
                let _ = writeln!(
                    self.out,
                    "{pad}if !({cond_expr}) {{ return Err(lust_aot_rt::AotError::UserError({msg_expr})); }}"
                );
                return Ok(());
            }
            "error" => {
                let msg_expr = args
                    .first()
                    .map(|m| self.emit_operand_as_display_string(m))
                    .unwrap_or_else(|| "alloc::string::String::from(\"error\")".to_string());
                let _ = writeln!(
                    self.out,
                    "{pad}return Err(lust_aot_rt::AotError::UserError({msg_expr}));"
                );
                return Ok(());
            }
            "array.push" | "table.insert" => {
                if args.len() >= 2 {
                    let arg0_ty = operand_mir_type(&args[0], self.func);
                    if matches!(arg0_ty, MirType::Unknown) {
                        let arr_v = self.emit_operand_value(&args[0]);
                        let val_v = self.emit_operand_as_aot_value(&args[1]);
                        let _ = writeln!(self.out, "{pad}{arr_v}.array_push({val_v}, __ctx)?;");
                        return Ok(());
                    }
                    let elem_ty = match arg0_ty {
                        MirType::Array(inner) => *inner,
                        _ => operand_mir_type(&args[1], self.func),
                    };
                    let val_expr = self.emit_operand_owned(&args[1], &elem_ty);
                    if let Some(loc) = operand_exact_local(&args[0])
                        && self.local_storage(loc) == LocalStorage::RcRefCell
                    {
                        let arr_var = self.local_var_name(loc);
                        let _ = writeln!(self.out, "{pad}{{");
                        let _ = writeln!(self.out, "{pad}    let __val = {val_expr};");
                        let _ = writeln!(
                            self.out,
                            "{pad}    let mut __arr = lust_aot_rt::ref_borrow_mut(&{arr_var})?;"
                        );
                        if self.ctx.opts.max_memory_bytes.is_some() {
                            let rust_elem = self.ctx.rust_type(&elem_ty, "'a");
                            let _ = writeln!(
                                self.out,
                                "{pad}    if __arr.len() == __arr.capacity() {{ __arr.try_reserve(1).map_err(|_| lust_aot_rt::AotError::from_alloc_failure(__alloc, core::mem::size_of::<{rust_elem}>()))?; }}"
                            );
                        }
                        let _ = writeln!(self.out, "{pad}    __arr.push(__val);");
                        let _ = writeln!(self.out, "{pad}}}");
                    } else {
                        let arr_expr = self.emit_operand_mut_ref(&args[0]);
                        if self.ctx.opts.max_memory_bytes.is_some() {
                            let rust_elem = self.ctx.rust_type(&elem_ty, "'a");
                            let _ = writeln!(
                                self.out,
                                "{pad}if {arr_expr}.len() == {arr_expr}.capacity() {{ {arr_expr}.try_reserve(1).map_err(|_| lust_aot_rt::AotError::from_alloc_failure(__alloc, core::mem::size_of::<{rust_elem}>()))?; }}"
                            );
                        }
                        let _ = writeln!(self.out, "{pad}{arr_expr}.push({val_expr});");
                    }
                }
                return Ok(());
            }
            "array.len" | "table.len" | "string.len" | "map.len" => {
                if let Some(arg) = args.first()
                    && !self.dest_is_omitted(destination)
                {
                    let ty = operand_mir_type(arg, self.func);
                    let len_expr = if matches!(ty, MirType::Map(_, _)) {
                        format!("{}.len()?", self.emit_operand_value(arg))
                    } else if matches!(ty, MirType::Unknown) {
                        format!("({}.len()? as i64)", self.emit_operand_value(arg))
                    } else if let Some(loc) = operand_exact_local(arg)
                        && self.local_storage(loc) == LocalStorage::RcRefCell
                    {
                        let var = self.local_var_name(loc);
                        format!("(lust_aot_rt::ref_borrow(&{var})?.len() as i64)")
                    } else {
                        let r = self.emit_operand_ref(arg);
                        format!("({r}.len() as i64)")
                    };
                    self.emit_assign_place_val(destination, &len_expr, indent);
                }
                return Ok(());
            }
            "map.new" => {
                if !self.dest_is_omitted(destination) {
                    let dst_ty = self.place_mir_type(destination);
                    let expr = if matches!(dst_ty, MirType::Unknown) {
                        "lust_aot_rt::AotValue::map()".to_string()
                    } else {
                        "lust_aot_rt::AotMap::new()".to_string()
                    };
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "map.set" | "table.set" if args.len() >= 3 => {
                let m_ty = operand_mir_type(&args[0], self.func);
                match m_ty {
                    MirType::Map(k_ty, v_ty) => {
                        let m_expr = self.emit_operand_value(&args[0]);
                        let k_expr = self.emit_operand_as_map_key(&args[1], &k_ty);
                        let v_expr = self.emit_operand_owned(&args[2], &v_ty);
                        let _ = writeln!(
                            self.out,
                            "{pad}{{ let __k = {k_expr}; let __v = {v_expr}; {m_expr}.set(__k, __v, __alloc)?; }}"
                        );
                    }
                    MirType::Unknown => {
                        let m_expr = self.emit_operand_value(&args[0]);
                        let k_expr = self.emit_operand_as_map_key(&args[1], &MirType::Unknown);
                        let v_expr = self.emit_operand_as_aot_value(&args[2]);
                        let _ = writeln!(
                            self.out,
                            "{pad}{{ let __k = {k_expr}; let __v = {v_expr}; {m_expr}.map_set(__k, __v, __ctx)?; }}"
                        );
                    }
                    _ => {}
                }
                return Ok(());
            }
            "map.get" | "table.get" if args.len() >= 2 => {
                if !self.dest_is_omitted(destination) {
                    let m_ty = operand_mir_type(&args[0], self.func);
                    let dst_ty = self.place_mir_type(destination);
                    let expr = match m_ty {
                        MirType::Map(k_ty, v_ty) => {
                            let m_expr = self.emit_operand_value(&args[0]);
                            let k_expr = self.emit_operand_as_map_key(&args[1], &k_ty);
                            if matches!(dst_ty, MirType::Option(_)) {
                                format!("{{ let __k = {k_expr}; {m_expr}.get_opt(&__k)? }}")
                            } else {
                                let raw = format!("{{ let __k = {k_expr}; {m_expr}.get(&__k)? }}");
                                self.coerce_expr_to_type(&raw, &v_ty, &dst_ty)
                            }
                        }
                        MirType::Unknown => {
                            let m_expr = self.emit_operand_value(&args[0]);
                            let k_expr = self.emit_operand_as_map_key(&args[1], &MirType::Unknown);
                            let raw = format!("{{ let __k = {k_expr}; {m_expr}.map_get(&__k)? }}");
                            self.coerce_expr_to_type(&raw, &MirType::Unknown, &dst_ty)
                        }
                        _ => self.ctx.rust_default_expr(&dst_ty),
                    };
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "map.has" if args.len() >= 2 => {
                if !self.dest_is_omitted(destination) {
                    let m_ty = operand_mir_type(&args[0], self.func);
                    let m_expr = self.emit_operand_value(&args[0]);
                    let expr = match m_ty {
                        MirType::Map(k_ty, _) => {
                            let k_expr = self.emit_operand_as_map_key(&args[1], &k_ty);
                            format!("{{ let __k = {k_expr}; {m_expr}.has(&__k)? }}")
                        }
                        _ => {
                            let k_expr = self.emit_operand_as_map_key(&args[1], &MirType::Unknown);
                            format!("{{ let __k = {k_expr}; {m_expr}.map_has(&__k)? }}")
                        }
                    };
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "map.is_empty" if !args.is_empty() => {
                if !self.dest_is_omitted(destination) {
                    let m_ty = operand_mir_type(&args[0], self.func);
                    let m_expr = self.emit_operand_value(&args[0]);
                    let expr = if matches!(m_ty, MirType::Unknown) {
                        format!("{m_expr}.map_is_empty()?")
                    } else {
                        format!("{m_expr}.is_empty()?")
                    };
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "map.delete" | "map.remove" if args.len() >= 2 => {
                let m_ty = operand_mir_type(&args[0], self.func);
                let m_expr = self.emit_operand_value(&args[0]);
                let expr = match m_ty {
                    MirType::Map(k_ty, _) => {
                        let k_expr = self.emit_operand_as_map_key(&args[1], &k_ty);
                        format!("{{ let __k = {k_expr}; {m_expr}.delete(&__k, __alloc)? }}")
                    }
                    _ => {
                        let k_expr = self.emit_operand_as_map_key(&args[1], &MirType::Unknown);
                        format!("{{ let __k = {k_expr}; {m_expr}.map_remove(&__k, __ctx)? }}")
                    }
                };
                if self.dest_is_omitted(destination)
                    || matches!(self.place_mir_type(destination), MirType::Unit)
                {
                    let _ = writeln!(self.out, "{pad}let _ = {expr};");
                } else {
                    let dst_ty = self.place_mir_type(destination);
                    if matches!(dst_ty, MirType::Option(_)) {
                        self.emit_assign_place_val(destination, &expr, indent);
                    } else {
                        let _ = writeln!(self.out, "{pad}let _ = {expr};");
                    }
                }
                return Ok(());
            }
            "map.clear" if !args.is_empty() => {
                let m_ty = operand_mir_type(&args[0], self.func);
                let m_expr = self.emit_operand_value(&args[0]);
                if matches!(m_ty, MirType::Unknown) {
                    let _ = writeln!(self.out, "{pad}{m_expr}.map_clear()?;");
                } else {
                    let _ = writeln!(self.out, "{pad}{m_expr}.clear()?;");
                }
                return Ok(());
            }
            "map.keys" if !args.is_empty() => {
                if !self.dest_is_omitted(destination) {
                    let m_ty = operand_mir_type(&args[0], self.func);
                    let m_expr = self.emit_operand_value(&args[0]);
                    let expr = if matches!(m_ty, MirType::Unknown) {
                        format!("{m_expr}.map_keys(__alloc)?")
                    } else {
                        format!("{m_expr}.keys_in(__alloc)?")
                    };
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "map.values" if !args.is_empty() => {
                if !self.dest_is_omitted(destination) {
                    let m_ty = operand_mir_type(&args[0], self.func);
                    let m_expr = self.emit_operand_value(&args[0]);
                    let expr = if matches!(m_ty, MirType::Unknown) {
                        format!("{m_expr}.map_values(__alloc)?")
                    } else {
                        format!("{m_expr}.values_in(__alloc)?")
                    };
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "enum.new" | "enum.alloc" => {
                if !self.dest_is_omitted(destination) {
                    let dst_ty = self.place_mir_type(destination);
                    let expr = match &dst_ty {
                        MirType::Enum(ename) => self.emit_enum_aggregate(ename, args)?,
                        _ => self.emit_enum_aggregate("Enum", args)?,
                    };
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "enum.tag" if !args.is_empty() => {
                if !self.dest_is_omitted(destination) {
                    let arg_ty = operand_mir_type(&args[0], self.func);
                    let expr = match &arg_ty {
                        MirType::Enum(ename) if self.ctx.enums.contains_key(ename) => {
                            format!("{}.__lust_tag()", self.emit_operand_value(&args[0]))
                        }
                        _ => format!("{}.enum_tag()?", self.emit_operand_value(&args[0])),
                    };
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "enum.payload" if !args.is_empty() => {
                if !self.dest_is_omitted(destination) {
                    let arg_ty = operand_mir_type(&args[0], self.func);
                    let dst_ty = self.place_mir_type(destination);
                    let fidx = args
                        .get(1)
                        .and_then(|op| match op {
                            Operand::Constant(MirConstant::Int(i)) if *i >= 0 => {
                                Some(*i as usize)
                            }
                            _ => None,
                        })
                        .unwrap_or(0);
                    let expr = match &arg_ty {
                        MirType::Enum(ename) if self.ctx.enums.contains_key(ename) => {
                            let emeta = &self.ctx.enums[ename];
                            let rust_ename = rust_enum_name(ename);
                            let eval_expr = self.emit_operand_value(&args[0]);
                            let narrowed_vidx = operand_exact_local(&args[0])
                                .and_then(|l| self.active_enum_variant.get(&l).copied());
                            let mut arms = Vec::new();
                            for (vidx, (vname, vfields)) in emeta.variants.iter().enumerate() {
                                if let Some(nv) = narrowed_vidx
                                    && nv != vidx
                                {
                                    continue;
                                }
                                let Some(fty) = vfields.get(fidx) else {
                                    continue;
                                };
                                if fty != &dst_ty && !matches!(dst_ty, MirType::Unknown) {
                                    continue;
                                }
                                let rv = sanitize_ident(vname);
                                let pats: Vec<String> = (0..vfields.len())
                                    .map(|i| {
                                        if i == fidx {
                                            "__v".to_string()
                                        } else {
                                            "_".to_string()
                                        }
                                    })
                                    .collect();
                                let extracted = if matches!(dst_ty, MirType::Unknown) {
                                    let owned_v = if self.ctx.is_copy_type(fty) {
                                        "*__v"
                                    } else {
                                        "__v.clone()"
                                    };
                                    self.coerce_expr_to_type(owned_v, fty, &MirType::Unknown)
                                } else if self.ctx.is_copy_type(fty) {
                                    "*__v".to_string()
                                } else {
                                    "__v.clone()".to_string()
                                };
                                arms.push(format!(
                                    "{rust_ename}::{rv}({}) => {extracted}",
                                    pats.join(", ")
                                ));
                            }
                            arms.push(
                                "_ => return Err(lust_aot_rt::AotError::UserError(alloc::string::String::from(\"enum payload variant mismatch\")))"
                                    .to_string(),
                            );
                            format!("match &({eval_expr}) {{ {} }}", arms.join(", "))
                        }
                        _ => {
                            let raw = format!(
                                "{}.enum_payload({fidx}_usize)?",
                                self.emit_operand_value(&args[0])
                            );
                            self.coerce_expr_to_type(&raw, &MirType::Unknown, &dst_ty)
                        }
                    };
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "tostring" => {
                if let Some(arg) = args.first()
                    && !self.dest_is_omitted(destination)
                {
                    let expr = self.emit_tostring_expr(arg);
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "abs" | "math.abs" => {
                if let Some(arg) = args.first()
                    && !self.dest_is_omitted(destination)
                {
                    let ty = operand_mir_type(arg, self.func);
                    let v = self.emit_operand_value(arg);
                    let expr = if matches!(ty, MirType::Float) {
                        format!("({v}).abs()")
                    } else {
                        format!("({v}).wrapping_abs()")
                    };
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "min" | "math.min" => {
                if args.len() >= 2 && !self.dest_is_omitted(destination) {
                    let ty = operand_mir_type(&args[0], self.func);
                    let a = self.emit_operand_value(&args[0]);
                    let b = self.emit_operand_value(&args[1]);
                    let expr = if matches!(ty, MirType::Float) {
                        format!("({a}).min({b})")
                    } else {
                        format!("core::cmp::min({a}, {b})")
                    };
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "max" | "math.max" => {
                if args.len() >= 2 && !self.dest_is_omitted(destination) {
                    let ty = operand_mir_type(&args[0], self.func);
                    let a = self.emit_operand_value(&args[0]);
                    let b = self.emit_operand_value(&args[1]);
                    let expr = if matches!(ty, MirType::Float) {
                        format!("({a}).max({b})")
                    } else {
                        format!("core::cmp::max({a}, {b})")
                    };
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "clamp" | "math.clamp" => {
                if args.len() >= 3 && !self.dest_is_omitted(destination) {
                    let v = self.emit_operand_value(&args[0]);
                    let lo = self.emit_operand_value(&args[1]);
                    let hi = self.emit_operand_value(&args[2]);
                    let expr = format!("({v}).clamp({lo}, {hi})");
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "floor" | "math.floor" | "ceil" | "math.ceil" | "sqrt" | "math.sqrt" | "round"
            | "math.round" => {
                if let Some(arg) = args.first()
                    && !self.dest_is_omitted(destination)
                {
                    let v = self.emit_operand_value(arg);
                    let m = callee.rsplit('.').next().unwrap_or(callee);
                    let expr = format!("({v} as f64).{m}()");
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            _ => {}
        }

        // User Enum Constructor call `EnumName.Variant(...)` or `EnumName::Variant(...)`
        if let Some((ename, vname)) = callee
            .split_once("::")
            .or_else(|| callee.split_once('.'))
            && let Some(emeta) = self.ctx.enums.get(ename)
            && let Some((_vn, vfields)) = emeta.variants.iter().find(|(n, _)| n == vname)
        {
            if !self.dest_is_omitted(destination) {
                let rust_ename = rust_enum_name(ename);
                let rv = sanitize_ident(vname);
                let expr = if vfields.is_empty() {
                    format!("{rust_ename}::{rv}")
                } else {
                    let mut fargs = Vec::with_capacity(vfields.len());
                    for (fi, fty) in vfields.iter().enumerate() {
                        if let Some(op) = args.get(fi) {
                            fargs.push(self.emit_operand_owned(op, fty));
                        } else {
                            fargs.push(self.ctx.rust_default_expr(fty));
                        }
                    }
                    format!("{rust_ename}::{rv}({})", fargs.join(", "))
                };
                self.emit_assign_place_val(destination, &expr, indent);
            }
            return Ok(());
        }

        // Local closure invocation (`Terminator::Call { callee: local_var_name, .. }`)
        if !self.ctx.fn_metas.contains_key(callee)
            && let Some(&cl_loc) = self.local_name_to_id.get(callee)
        {
            let cl_ty = self.func.locals[cl_loc.0].ty.clone();
            let cl_expr = self.emit_operand_value(&Operand::Copy(Place::Local(cl_loc)));
            match cl_ty {
                MirType::Function {
                    params,
                    return_type,
                } => {
                    let mut cargs = Vec::with_capacity(args.len() + 2);
                    cargs.push("__ctx".to_string());
                    cargs.push("__alloc".to_string());
                    for (ai, arg) in args.iter().enumerate() {
                        if let Some(pty) = params.get(ai) {
                            cargs.push(self.emit_operand_owned(arg, pty));
                        } else {
                            cargs.push(self.emit_operand_value(arg));
                        }
                    }
                    let call_expr = format!("({cl_expr})({})?", cargs.join(", "));
                    if self.dest_is_omitted(destination) || *return_type == MirType::Unit {
                        let _ = writeln!(self.out, "{pad}let _ = {call_expr};");
                    } else {
                        let dst_ty = self.place_mir_type(destination);
                        let final_expr =
                            self.coerce_expr_to_type(&call_expr, &return_type, &dst_ty);
                        self.emit_assign_place_val(destination, &final_expr, indent);
                    }
                    return Ok(());
                }
                MirType::Unknown => {
                    let aot_args: Vec<String> = args
                        .iter()
                        .map(|a| self.emit_operand_as_aot_value(a))
                        .collect();
                    let call_expr =
                        format!("({cl_expr}).call(__ctx, &[{}])?", aot_args.join(", "));
                    if self.dest_is_omitted(destination)
                        || matches!(self.place_mir_type(destination), MirType::Unit)
                    {
                        let _ = writeln!(self.out, "{pad}let _ = {call_expr};");
                    } else {
                        let dst_ty = self.place_mir_type(destination);
                        let final_expr =
                            self.coerce_expr_to_type(&call_expr, &MirType::Unknown, &dst_ty);
                        self.emit_assign_place_val(destination, &final_expr, indent);
                    }
                    return Ok(());
                }
                _ => {}
            }
        }

        // User-defined function call within the module
        let Some(callee_meta) = self.ctx.fn_metas.get(callee) else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: unresolved callee '{callee}'"
            )));
        };

        let mut call_args = Vec::new();
        if callee_meta.needs_ctx {
            call_args.push("__ctx".to_string());
        }
        if callee_meta.needs_alloc {
            call_args.push("__alloc".to_string());
        }
        if callee_meta.needs_arena {
            call_args.push("__arena".to_string());
        }

        let callee_fn = self
            .ctx
            .module
            .functions
            .iter()
            .find(|f| f.signature.name == callee);

        // Track any scoped RefCell borrows needed when passing an RcRefCell local to a borrowed param
        let mut pre_borrows: Vec<String> = Vec::new();
        for (i, arg) in args.iter().enumerate() {
            let mode = callee_meta
                .param_modes
                .get(i)
                .copied()
                .unwrap_or(ParamMode::ByValue);
            let expected_ty = callee_fn.and_then(|cf| {
                cf.params
                    .get(i)
                    .and_then(|pid| cf.locals.get(pid.0))
                    .map(|d| &d.ty)
            });
            let arg_rc_storage = operand_exact_local(arg).map(|l| (l, self.local_storage(l)));
            let arg_expr = match mode {
                ParamMode::ByValue => {
                    if let Some(et) = expected_ty {
                        self.emit_operand_owned(arg, et)
                    } else {
                        self.emit_operand_value(arg)
                    }
                }
                ParamMode::RefStruct => {
                    if let Some((loc, LocalStorage::RcRefCell)) = arg_rc_storage {
                        let var = self.local_var_name(loc);
                        pre_borrows.push(format!(
                            "let __arg_g_{i} = lust_aot_rt::ref_borrow(&{var})?;"
                        ));
                        format!("&*__arg_g_{i}")
                    } else {
                        self.emit_operand_ref_borrowed(arg)
                    }
                }
                ParamMode::MutRefStruct => {
                    if let Some((loc, LocalStorage::RcRefCell)) = arg_rc_storage {
                        let var = self.local_var_name(loc);
                        pre_borrows.push(format!(
                            "let mut __arg_g_{i} = lust_aot_rt::ref_borrow_mut(&{var})?;"
                        ));
                        format!("&mut *__arg_g_{i}")
                    } else {
                        self.emit_operand_mut_ref_borrowed(arg)
                    }
                }
                ParamMode::RefSlice => {
                    if let Some((loc, LocalStorage::RcRefCell)) = arg_rc_storage {
                        let var = self.local_var_name(loc);
                        pre_borrows.push(format!(
                            "let __arg_g_{i} = lust_aot_rt::ref_borrow(&{var})?;"
                        ));
                        format!("&__arg_g_{i}[..]")
                    } else {
                        self.emit_operand_slice_ref(arg)
                    }
                }
                ParamMode::MutVec => {
                    if let Some((loc, LocalStorage::RcRefCell)) = arg_rc_storage {
                        let var = self.local_var_name(loc);
                        pre_borrows.push(format!(
                            "let mut __arg_g_{i} = lust_aot_rt::ref_borrow_mut(&{var})?;"
                        ));
                        format!("&mut *__arg_g_{i}")
                    } else {
                        self.emit_operand_mut_ref_borrowed(arg)
                    }
                }
                ParamMode::RefStr => {
                    if let Some((loc, LocalStorage::RcRefCell)) = arg_rc_storage {
                        let var = self.local_var_name(loc);
                        pre_borrows.push(format!(
                            "let __arg_g_{i} = lust_aot_rt::ref_borrow(&{var})?;"
                        ));
                        format!("__arg_g_{i}.as_str()")
                    } else {
                        self.emit_operand_str_ref(arg)
                    }
                }
                ParamMode::RcCell => {
                    if let Some((loc, LocalStorage::RcCell)) = arg_rc_storage {
                        let var = self.local_var_name(loc);
                        format!("alloc::rc::Rc::clone(&{var})")
                    } else {
                        let v = if let Some(et) = expected_ty {
                            self.emit_operand_owned(arg, et)
                        } else {
                            self.emit_operand_value(arg)
                        };
                        format!("alloc::rc::Rc::new(core::cell::Cell::new({v}))")
                    }
                }
                ParamMode::RcRefCell => {
                    if let Some((loc, LocalStorage::RcRefCell)) = arg_rc_storage {
                        let var = self.local_var_name(loc);
                        format!("alloc::rc::Rc::clone(&{var})")
                    } else {
                        let v = if let Some(et) = expected_ty {
                            self.emit_operand_owned(arg, et)
                        } else {
                            self.emit_operand_value(arg)
                        };
                        format!("alloc::rc::Rc::new(core::cell::RefCell::new({v}))")
                    }
                }
            };
            call_args.push(arg_expr);
        }

        let raw_call = format!("{}({})", callee_meta.rust_name, call_args.join(", "));
        let unwrapped_call = if callee_meta.can_fail || self.in_cleanup_closure.is_some() {
            format!("{raw_call}?")
        } else if self.meta.can_fail {
            // `From<Infallible> for AotError` lets `?` convert zero-cost!
            format!("{raw_call}?")
        } else {
            format!("match {raw_call} {{ Ok(__v) => __v }}")
        };

        let call_expr = if pre_borrows.is_empty() {
            unwrapped_call
        } else {
            format!("{{ {} {unwrapped_call} }}", pre_borrows.join(" "))
        };

        if self.dest_is_omitted(destination) || callee_meta.return_type == MirType::Unit {
            let _ = writeln!(self.out, "{pad}let _ = {call_expr};");
        } else {
            let dst_ty = self.place_mir_type(destination);
            let final_expr =
                self.coerce_expr_to_type(&call_expr, &callee_meta.return_type, &dst_ty);
            self.emit_assign_place_val(destination, &final_expr, indent);
        }
        Ok(())
    }

    fn emit_print_call(&mut self, args: &[Operand], is_ln: bool, indent: usize) {
        let pad = "    ".repeat(indent);
        if self.ctx.opts.no_std {
            return;
        }
        if args.is_empty() {
            if is_ln {
                let _ = writeln!(self.out, "{pad}lust_aot_rt::println_str(\"\");");
            }
            return;
        }
        let fn_name = if is_ln {
            "lust_aot_rt::println_str"
        } else {
            "lust_aot_rt::print_str"
        };
        if args.len() == 1 {
            let arg = &args[0];
            let ty = operand_mir_type(arg, self.func);
            match ty {
                MirType::Int => {
                    let v = self.emit_operand_value(arg);
                    let _ = writeln!(
                        self.out,
                        "{pad}{{ let mut __p = alloc::string::String::new(); lust_aot_rt::string_append_int(&mut __p, {v}); {fn_name}(&__p); }}"
                    );
                }
                MirType::Float => {
                    let v = self.emit_operand_value(arg);
                    let _ = writeln!(
                        self.out,
                        "{pad}{{ let __p = lust_aot_rt::format_float({v}); {fn_name}(&__p); }}"
                    );
                }
                MirType::Bool => {
                    let v = self.emit_operand_bool(arg);
                    let _ = writeln!(
                        self.out,
                        "{pad}{fn_name}(if {v} {{ \"true\" }} else {{ \"false\" }});"
                    );
                }
                MirType::String => {
                    let s = self.emit_operand_str_ref(arg);
                    let _ = writeln!(self.out, "{pad}{fn_name}({s});");
                }
                MirType::Unit => {
                    let _ = writeln!(self.out, "{pad}{fn_name}(\"nil\");");
                }
                MirType::Enum(_) | MirType::Map(_, _) | MirType::Unknown => {
                    let v = self.emit_operand_value(arg);
                    let _ = writeln!(
                        self.out,
                        "{pad}{{ let __p = alloc::format!(\"{{}}\", {v}); {fn_name}(&__p); }}"
                    );
                }
                _ => {
                    let v = self.emit_operand_value(arg);
                    let _ = writeln!(
                        self.out,
                        "{pad}{{ let __p = alloc::format!(\"{{:?}}\", {v}); {fn_name}(&__p); }}"
                    );
                }
            }
            return;
        }

        let _ = writeln!(self.out, "{pad}{{");
        let _ = writeln!(
            self.out,
            "{pad}    let mut __p = alloc::string::String::new();"
        );
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                let _ = writeln!(self.out, "{pad}    __p.push('\\t');");
            }
            let ty = operand_mir_type(arg, self.func);
            match ty {
                MirType::Int => {
                    let v = self.emit_operand_value(arg);
                    let _ = writeln!(
                        self.out,
                        "{pad}    lust_aot_rt::string_append_int(&mut __p, {v});"
                    );
                }
                MirType::Float => {
                    let v = self.emit_operand_value(arg);
                    let _ = writeln!(
                        self.out,
                        "{pad}    lust_aot_rt::string_append_float(&mut __p, {v});"
                    );
                }
                MirType::Bool => {
                    let v = self.emit_operand_bool(arg);
                    let _ = writeln!(
                        self.out,
                        "{pad}    __p.push_str(if {v} {{ \"true\" }} else {{ \"false\" }});"
                    );
                }
                MirType::String => {
                    let s = self.emit_operand_str_ref(arg);
                    let _ = writeln!(self.out, "{pad}    __p.push_str({s});");
                }
                MirType::Unit => {
                    let _ = writeln!(self.out, "{pad}    __p.push_str(\"nil\");");
                }
                MirType::Enum(_) | MirType::Map(_, _) | MirType::Unknown => {
                    let v = self.emit_operand_value(arg);
                    let _ = writeln!(
                        self.out,
                        "{pad}    let _ = core::fmt::Write::write_fmt(&mut __p, format_args!(\"{{}}\", {v}));"
                    );
                }
                _ => {
                    let v = self.emit_operand_value(arg);
                    let _ = writeln!(
                        self.out,
                        "{pad}    let _ = core::fmt::Write::write_fmt(&mut __p, format_args!(\"{{:?}}\", {v}));"
                    );
                }
            }
        }
        let _ = writeln!(self.out, "{pad}    {fn_name}(&__p);");
        let _ = writeln!(self.out, "{pad}}}");
    }

    fn emit_tostring_expr(&self, arg: &Operand) -> String {
        let ty = operand_mir_type(arg, self.func);
        match ty {
            MirType::Int => {
                let v = self.emit_operand_value(arg);
                if self.ctx.opts.max_memory_bytes.is_some() {
                    format!(
                        "{{ let mut __s = alloc::string::String::new(); lust_aot_rt::try_string_append_int(&mut __s, {v}, __alloc)?; __s }}"
                    )
                } else {
                    format!(
                        "{{ let mut __s = alloc::string::String::new(); lust_aot_rt::string_append_int(&mut __s, {v}); __s }}"
                    )
                }
            }
            MirType::Float => {
                let v = self.emit_operand_value(arg);
                format!("lust_aot_rt::format_float({v})")
            }
            MirType::Bool => {
                let v = self.emit_operand_bool(arg);
                format!("alloc::string::String::from(if {v} {{ \"true\" }} else {{ \"false\" }})")
            }
            MirType::String => {
                let s = self.emit_operand_str_ref(arg);
                format!("alloc::string::String::from({s})")
            }
            MirType::Unit => "alloc::string::String::from(\"nil\")".to_string(),
            MirType::Enum(_) | MirType::Map(_, _) | MirType::Unknown => {
                let v = self.emit_operand_value(arg);
                format!("alloc::format!(\"{{}}\", {v})")
            }
            _ => {
                let v = self.emit_operand_value(arg);
                format!("alloc::format!(\"{{:?}}\", {v})")
            }
        }
    }

    fn emit_return_expr(&mut self, opt: Option<&Operand>, indent: usize) {
        let pad = "    ".repeat(indent);
        if self.in_cleanup_closure.is_some() {
            let _ = writeln!(self.out, "{pad}return Ok(());");
            return;
        }
        let expr = match opt {
            Some(op) => self.emit_operand_owned(op, &self.func.signature.return_type),
            None => self.ctx.rust_default_expr(&self.func.signature.return_type),
        };
        if !self.errdefer_guards.is_empty() {
            let _ = writeln!(self.out, "{pad}let __ret = {expr};");
            for g in self.errdefer_guards.clone().iter().rev() {
                let _ = writeln!(self.out, "{pad}core::mem::DropGuard::dismiss({g});");
            }
            let _ = writeln!(self.out, "{pad}return Ok(__ret);");
        } else {
            let _ = writeln!(self.out, "{pad}return Ok({expr});");
        }
    }

    // -----------------------------------------------------------------------
    // Place & Operand Formatting Helpers
    // -----------------------------------------------------------------------

    fn dest_is_omitted(&self, place: &Place) -> bool {
        matches!(place, Place::Local(l) if self.omitted_locals.contains(l))
    }

    fn place_mir_type(&self, place: &Place) -> MirType {
        match place {
            Place::Local(id) => {
                let r = self.resolve_local(*id);
                self.func
                    .locals
                    .get(r.0)
                    .map(|d| d.ty.clone())
                    .unwrap_or(MirType::Int)
            }
            Place::Field(base, fidx) => {
                let base_ty = self.place_mir_type(base);
                match base_ty {
                    MirType::Struct(sname) => self
                        .ctx
                        .structs
                        .get(&sname)
                        .and_then(|m| m.fields.get(*fidx))
                        .map(|(_, t)| t.clone())
                        .unwrap_or(MirType::Int),
                    MirType::Tuple(elems) => elems.get(*fidx).cloned().unwrap_or(MirType::Int),
                    _ => MirType::Int,
                }
            }
            Place::Index(base, _) => match self.place_mir_type(base) {
                MirType::Array(inner) => *inner,
                MirType::Map(_, v) => *v,
                MirType::Unknown => MirType::Unknown,
                _ => MirType::Int,
            },
        }
    }

    fn emit_local_read_scalar(&self, loc: LocalId) -> String {
        let r = self.resolve_local(loc);
        let var = self.local_var_name(r);
        match self.local_storage(r) {
            LocalStorage::Direct => var,
            LocalStorage::RcCell => format!("{var}.get()"),
            LocalStorage::RcRefCell => format!("(*lust_aot_rt::ref_borrow(&{var})?)"),
        }
    }

    fn emit_place_lhs(&self, place: &Place) -> String {
        match place {
            Place::Local(id) => self.local_var_name(*id),
            Place::Field(base, fidx) => format!("{}.field_{fidx}", self.emit_place_ref_expr(base)),
            Place::Index(base, idx) => {
                let idx_expr = self.emit_local_read_scalar(*idx);
                format!("{}[({idx_expr}) as usize]", self.emit_place_ref_expr(base))
            }
        }
    }

    fn emit_place_ref_expr(&self, place: &Place) -> String {
        match place {
            Place::Local(id) => self.local_var_name(*id),
            Place::Field(base, fidx) => {
                if let Place::Local(base_loc) = base.as_ref()
                    && self.local_storage(*base_loc) == LocalStorage::RcRefCell
                {
                    let var = self.local_var_name(*base_loc);
                    let fty = self.place_mir_type(place);
                    if self.ctx.is_copy_type(&fty) {
                        format!("lust_aot_rt::ref_borrow(&{var})?.field_{fidx}")
                    } else {
                        format!("lust_aot_rt::ref_borrow(&{var})?.field_{fidx}.clone()")
                    }
                } else {
                    format!("{}.field_{fidx}", self.emit_place_ref_expr(base))
                }
            }
            Place::Index(base, idx) => {
                let idx_expr = self.emit_local_read_scalar(*idx);
                format!("{}[({idx_expr}) as usize]", self.emit_place_ref_expr(base))
            }
        }
    }

    fn emit_place_value_expr(&self, place: &Place) -> String {
        match place {
            Place::Local(id) => {
                let r = self.resolve_local(*id);
                let var = self.local_var_name(r);
                let ty = &self.func.locals[r.0].ty;
                match self.local_storage(r) {
                    LocalStorage::Direct => var,
                    LocalStorage::RcCell => format!("{var}.get()"),
                    LocalStorage::RcRefCell => {
                        if self.ctx.is_copy_type(ty) {
                            format!("(*lust_aot_rt::ref_borrow(&{var})?)")
                        } else {
                            format!("lust_aot_rt::ref_borrow(&{var})?.clone()")
                        }
                    }
                }
            }
            _ => self.emit_place_ref_expr(place),
        }
    }

    fn emit_place_mut_expr(&self, place: &Place) -> String {
        self.emit_place_ref_expr(place)
    }

    fn emit_operand_value(&self, op: &Operand) -> String {
        match op {
            Operand::Constant(c) => emit_constant(c),
            Operand::Copy(p) | Operand::Move(p) => self.emit_place_value_expr(p),
        }
    }

    fn emit_operand_as_aot_value(&self, op: &Operand) -> String {
        let ty = operand_mir_type(op, self.func);
        match op {
            Operand::Constant(c) => emit_constant_for_type(c, &MirType::Unknown),
            _ => {
                let v = self.emit_operand_value(op);
                self.coerce_expr_to_type(&v, &ty, &MirType::Unknown)
            }
        }
    }

    fn emit_operand_as_map_key(&self, op: &Operand, key_ty: &MirType) -> String {
        let src_ty = operand_mir_type(op, self.func);
        match key_ty {
            MirType::Int => self.emit_operand_owned(op, &MirType::Int),
            MirType::Bool => self.emit_operand_bool(op),
            MirType::String => self.emit_operand_owned(op, &MirType::String),
            _ => {
                // Convert to lust_aot_rt::AotKey
                match src_ty {
                    MirType::Int => format!("lust_aot_rt::AotKey::Int({})", self.emit_operand_value(op)),
                    MirType::Bool => format!("lust_aot_rt::AotKey::Bool({})", self.emit_operand_bool(op)),
                    MirType::String => format!(
                        "lust_aot_rt::AotKey::Str(alloc::string::String::from({}))",
                        self.emit_operand_str_ref(op)
                    ),
                    MirType::Unit => "lust_aot_rt::AotKey::Nil".to_string(),
                    MirType::Unknown => format!("({}).to_key()?", self.emit_operand_value(op)),
                    _ => format!("lust_aot_rt::AotKey::from({})", self.emit_operand_value(op)),
                }
            }
        }
    }

    fn emit_local_as_map_key(&self, loc: LocalId, key_ty: &MirType) -> String {
        self.emit_operand_as_map_key(&Operand::Copy(Place::Local(loc)), key_ty)
    }

    fn coerce_expr_to_type(&self, expr: &str, src_ty: &MirType, dst_ty: &MirType) -> String {
        if src_ty == dst_ty {
            return expr.to_string();
        }
        if matches!(dst_ty, MirType::Unknown) {
            return match src_ty {
                MirType::Unit => format!("{{ let _ = {expr}; lust_aot_rt::AotValue::Nil }}"),
                MirType::Bool => format!("lust_aot_rt::AotValue::Bool({expr})"),
                MirType::Int => format!("lust_aot_rt::AotValue::Int({expr})"),
                MirType::Float => format!("lust_aot_rt::AotValue::Float({expr})"),
                MirType::String => format!("lust_aot_rt::AotValue::str({expr})"),
                MirType::Unknown => format!("({expr}).clone()"),
                _ => format!("lust_aot_rt::AotValue::str(alloc::format!(\"{{:?}}\", {expr}))"),
            };
        }
        if matches!(src_ty, MirType::Unknown) {
            return match dst_ty {
                MirType::Unit => format!("{{ let _ = {expr}; () }}"),
                MirType::Bool => format!("({expr}).is_truthy()"),
                MirType::Int => format!(
                    "({expr}).as_int().ok_or_else(|| lust_aot_rt::AotError::UserError(alloc::string::String::from(\"expected int\")))?"
                ),
                MirType::Float => format!("({expr}).as_float_coerced()?"),
                MirType::String => format!("({expr}).to_display_string()"),
                _ => expr.to_string(),
            };
        }
        expr.to_string()
    }

    fn emit_operand_owned(&self, op: &Operand, expected_ty: &MirType) -> String {
        let src_ty = operand_mir_type(op, self.func);
        match op {
            Operand::Constant(c) => emit_constant_for_type(c, expected_ty),
            Operand::Copy(Place::Local(id)) | Operand::Move(Place::Local(id)) => {
                let r = self.resolve_local(*id);
                let var = self.local_var_name(r);
                let is_move = matches!(op, Operand::Move(_));
                let storage = self.local_storage(r);
                if matches!(expected_ty, MirType::Unknown) && !matches!(src_ty, MirType::Unknown) {
                    let v = self.emit_operand_value(op);
                    return self.coerce_expr_to_type(&v, &src_ty, &MirType::Unknown);
                }
                if matches!(src_ty, MirType::Unknown) && !matches!(expected_ty, MirType::Unknown) {
                    let v = self.emit_operand_value(op);
                    return self.coerce_expr_to_type(&v, &MirType::Unknown, expected_ty);
                }
                match storage {
                    LocalStorage::RcCell => format!("{var}.get()"),
                    LocalStorage::RcRefCell => {
                        if self.ctx.is_copy_type(expected_ty) {
                            format!("(*lust_aot_rt::ref_borrow(&{var})?)")
                        } else {
                            format!("lust_aot_rt::ref_borrow(&{var})?.clone()")
                        }
                    }
                    LocalStorage::Direct => {
                        if self.ctx.is_copy_type(expected_ty) {
                            if self.local_is_ref(r) {
                                format!("(*{var})")
                            } else {
                                var
                            }
                        } else {
                            match expected_ty {
                                MirType::String => {
                                    if self.local_is_ref(r) {
                                        format!("alloc::string::String::from({var})")
                                    } else if is_move
                                        && self.local_use_counts.get(&r).copied().unwrap_or(0) <= 1
                                    {
                                        format!("core::mem::take(&mut {var})")
                                    } else {
                                        format!("{var}.clone()")
                                    }
                                }
                                MirType::Array(_) => {
                                    format!(
                                        "core::mem::replace(&mut {var}, alloc::vec::Vec::new_in(__alloc))"
                                    )
                                }
                                MirType::Map(_, _) | MirType::Function { .. } | MirType::Unknown => {
                                    format!("{var}.clone()")
                                }
                                MirType::Enum(_) => {
                                    format!("{var}.clone()")
                                }
                                MirType::Option(_) => {
                                    format!("core::mem::take(&mut {var})")
                                }
                                MirType::Struct(sname) => {
                                    if self
                                        .ctx
                                        .structs
                                        .get(sname)
                                        .is_some_and(|s| s.derives_default)
                                    {
                                        format!("core::mem::take(&mut {var})")
                                    } else {
                                        var
                                    }
                                }
                                _ => format!("{var}.clone()"),
                            }
                        }
                    }
                }
            }
            Operand::Copy(p) | Operand::Move(p) => {
                let expr = self.emit_place_ref_expr(p);
                if matches!(expected_ty, MirType::Unknown) && !matches!(src_ty, MirType::Unknown) {
                    return self.coerce_expr_to_type(&expr, &src_ty, &MirType::Unknown);
                }
                if self.ctx.is_copy_type(expected_ty) {
                    expr
                } else if matches!(expected_ty, MirType::Option(_)) {
                    format!("core::mem::take(&mut {expr})")
                } else {
                    format!("{expr}.clone()")
                }
            }
        }
    }

    fn emit_operand_ref(&self, op: &Operand) -> String {
        match op {
            Operand::Constant(c) => emit_constant(c),
            Operand::Copy(p) | Operand::Move(p) => self.emit_place_ref_expr(p),
        }
    }

    fn emit_operand_ref_borrowed(&self, op: &Operand) -> String {
        match op {
            Operand::Copy(Place::Local(id)) | Operand::Move(Place::Local(id)) => {
                let r = self.resolve_local(*id);
                let var = self.local_var_name(r);
                if self.local_is_ref(r) {
                    var
                } else {
                    format!("&{var}")
                }
            }
            Operand::Copy(p) | Operand::Move(p) => format!("&{}", self.emit_place_ref_expr(p)),
            Operand::Constant(c) => format!("&{}", emit_constant(c)),
        }
    }

    fn emit_operand_mut_ref(&self, op: &Operand) -> String {
        match op {
            Operand::Copy(p) | Operand::Move(p) => self.emit_place_mut_expr(p),
            Operand::Constant(c) => emit_constant(c),
        }
    }

    fn emit_operand_mut_ref_borrowed(&self, op: &Operand) -> String {
        match op {
            Operand::Copy(Place::Local(id)) | Operand::Move(Place::Local(id)) => {
                let r = self.resolve_local(*id);
                let var = self.local_var_name(r);
                if let Some((_, ParamMode::MutRefStruct | ParamMode::MutVec)) = self.is_param(r) {
                    var
                } else {
                    format!("&mut {var}")
                }
            }
            Operand::Copy(p) | Operand::Move(p) => format!("&mut {}", self.emit_place_mut_expr(p)),
            Operand::Constant(c) => emit_constant(c),
        }
    }

    fn emit_operand_slice_ref(&self, op: &Operand) -> String {
        match op {
            Operand::Copy(Place::Local(id)) | Operand::Move(Place::Local(id)) => {
                let r = self.resolve_local(*id);
                let var = self.local_var_name(r);
                if let Some((_, ParamMode::RefSlice)) = self.is_param(r) {
                    var
                } else {
                    format!("&{var}[..]")
                }
            }
            Operand::Copy(p) | Operand::Move(p) => format!("&{}[..]", self.emit_place_ref_expr(p)),
            Operand::Constant(_) => "&[]".to_string(),
        }
    }

    fn emit_operand_str_ref(&self, op: &Operand) -> String {
        match op {
            Operand::Constant(MirConstant::String(s)) => format!("{s:?}"),
            Operand::Constant(c) => format!("{:?}", format!("{c}")),
            Operand::Copy(Place::Local(id)) | Operand::Move(Place::Local(id)) => {
                let r = self.resolve_local(*id);
                let var = self.local_var_name(r);
                if self.local_storage(r) == LocalStorage::RcRefCell {
                    format!("lust_aot_rt::ref_borrow(&{var})?.as_str()")
                } else if let Some((_, ParamMode::RefStr)) = self.is_param(r) {
                    var
                } else {
                    format!("{var}.as_str()")
                }
            }
            Operand::Copy(p) | Operand::Move(p) => format!("{}.as_str()", self.emit_place_ref_expr(p)),
        }
    }

    fn emit_operand_bool(&self, op: &Operand) -> String {
        let ty = operand_mir_type(op, self.func);
        let v = self.emit_operand_value(op);
        match ty {
            MirType::Bool => v,
            MirType::Int => format!("({v} != 0)"),
            MirType::Option(_) => format!("{v}.is_some()"),
            MirType::Unknown => format!("({v}).is_truthy()"),
            _ => v,
        }
    }

    fn emit_operand_as_display_string(&self, op: &Operand) -> String {
        let ty = operand_mir_type(op, self.func);
        match ty {
            MirType::String => {
                format!("alloc::string::String::from({})", self.emit_operand_str_ref(op))
            }
            _ => self.emit_tostring_expr(op),
        }
    }
}

// ---------------------------------------------------------------------------
// Cleanup Inlined-Defer Fingerprint Helpers (`NG-W12-4B`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum CleanupActionSig {
    Call(String, Vec<String>),
    Assign(String),
}

#[derive(Debug, Clone, Copy)]
enum ChainActionLoc {
    Stmt(BlockId, usize),
    Call(BlockId),
}

fn operand_sig(op: &Operand, func: &MirFunction) -> String {
    match op {
        Operand::Constant(c) => format!("const:{c}"),
        Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)) => {
            if let Some(name) = &func.locals[l.0].name {
                format!("var:{name}")
            } else {
                "tmp".to_string()
            }
        }
        Operand::Copy(Place::Field(b, i)) | Operand::Move(Place::Field(b, i)) => {
            format!("field:{}:{i}", b.root_local().0)
        }
        Operand::Copy(Place::Index(b, _)) | Operand::Move(Place::Index(b, _)) => {
            format!("idx:{}", b.root_local().0)
        }
    }
}

fn stmt_action_sig(stmt: &Statement, func: &MirFunction) -> Option<CleanupActionSig> {
    match stmt {
        Statement::Assign(place, rval, _) => {
            let p_sig = match place {
                Place::Local(l) => {
                    if let Some(name) = &func.locals[l.0].name {
                        format!("var:{name}")
                    } else {
                        "tmp".to_string()
                    }
                }
                Place::Field(b, i) => format!("field:{}:{i}", b.root_local().0),
                Place::Index(b, _) => format!("idx:{}", b.root_local().0),
            };
            let r_sig = match rval {
                Rvalue::Use(op) => operand_sig(op, func),
                Rvalue::BinaryOp(op, l, r) => {
                    format!("{op:?}({},{})", operand_sig(l, func), operand_sig(r, func))
                }
                Rvalue::UnaryOp(op, x) => format!("{op:?}({})", operand_sig(x, func)),
                Rvalue::Cast(x, ty) => format!("cast({},{ty})", operand_sig(x, func)),
                Rvalue::Aggregate(ty, ops) => format!("agg({ty},{})", ops.len()),
                Rvalue::Closure { func_name, .. } => format!("closure({func_name})"),
            };
            Some(CleanupActionSig::Assign(format!("{p_sig}={r_sig}")))
        }
        _ => None,
    }
}

fn call_action_sig(callee: &str, args: &[Operand], func: &MirFunction) -> CleanupActionSig {
    let arg_sigs: Vec<String> = args.iter().map(|a| operand_sig(a, func)).collect();
    CleanupActionSig::Call(callee.to_string(), arg_sigs)
}

fn extract_cleanup_action_sigs(func: &MirFunction, start_bb: BlockId) -> Vec<CleanupActionSig> {
    let mut out = Vec::new();
    let mut cur = start_bb;
    let mut seen = HashSet::new();
    while cur.0 < func.blocks.len() && seen.insert(cur) {
        let blk = &func.blocks[cur.0];
        for stmt in &blk.statements {
            if let Some(sig) = stmt_action_sig(stmt, func) {
                out.push(sig);
            }
        }
        match &blk.terminator {
            Some(Terminator::Call {
                callee,
                args,
                success,
                ..
            }) => {
                out.push(call_action_sig(callee, args, func));
                cur = *success;
            }
            Some(Terminator::Goto(next, _)) => {
                cur = *next;
            }
            _ => break,
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

fn emit_constant(c: &MirConstant) -> String {
    match c {
        MirConstant::Unit => "()".to_string(),
        MirConstant::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        MirConstant::Int(i) => {
            if *i == i64::MIN {
                "i64::MIN".to_string()
            } else {
                format!("{i}_i64")
            }
        }
        MirConstant::Float(f) => {
            if f.is_nan() {
                "f64::NAN".to_string()
            } else if f.is_infinite() {
                if *f > 0.0 {
                    "f64::INFINITY".to_string()
                } else {
                    "f64::NEG_INFINITY".to_string()
                }
            } else {
                let s = format!("{f:?}");
                if s.contains('.') || s.contains('e') || s.contains('E') {
                    format!("{s}_f64")
                } else {
                    format!("{s}.0_f64")
                }
            }
        }
        MirConstant::String(s) => format!("alloc::string::String::from({s:?})"),
    }
}

fn emit_constant_for_type(c: &MirConstant, expected_ty: &MirType) -> String {
    if matches!(expected_ty, MirType::Unknown) {
        return match c {
            MirConstant::Unit => "lust_aot_rt::AotValue::Nil".to_string(),
            MirConstant::Bool(b) => format!("lust_aot_rt::AotValue::Bool({b})"),
            MirConstant::Int(i) => {
                let lit = if *i == i64::MIN {
                    "i64::MIN".to_string()
                } else {
                    format!("{i}_i64")
                };
                format!("lust_aot_rt::AotValue::Int({lit})")
            }
            MirConstant::Float(f) => {
                format!(
                    "lust_aot_rt::AotValue::Float({})",
                    emit_constant(&MirConstant::Float(*f))
                )
            }
            MirConstant::String(s) => format!("lust_aot_rt::AotValue::str({s:?})"),
        };
    }
    match (c, expected_ty) {
        (MirConstant::Int(0), MirType::Option(_)) => "None".to_string(),
        (MirConstant::Int(i), MirType::Float) => format!("({i}_i64 as f64)"),
        (MirConstant::Int(i), MirType::Bool) => if *i != 0 { "true" } else { "false" }.to_string(),
        (MirConstant::Int(_), MirType::Unit) => "()".to_string(),
        _ => emit_constant(c),
    }
}

fn operand_mir_type(op: &Operand, func: &MirFunction) -> MirType {
    match op {
        Operand::Constant(c) => match c {
            MirConstant::Unit => MirType::Unit,
            MirConstant::Bool(_) => MirType::Bool,
            MirConstant::Int(_) => MirType::Int,
            MirConstant::Float(_) => MirType::Float,
            MirConstant::String(_) => MirType::String,
        },
        Operand::Copy(Place::Local(id)) | Operand::Move(Place::Local(id)) => func
            .locals
            .get(id.0)
            .map(|d| d.ty.clone())
            .unwrap_or(MirType::Int),
        Operand::Copy(Place::Index(base, _)) | Operand::Move(Place::Index(base, _)) => {
            match place_type_raw(base, func) {
                MirType::Array(inner) => *inner,
                _ => MirType::Int,
            }
        }
        Operand::Copy(Place::Field(base, fidx)) | Operand::Move(Place::Field(base, fidx)) => {
            match place_type_raw(base, func) {
                MirType::Tuple(elems) => elems.get(*fidx).cloned().unwrap_or(MirType::Int),
                _ => MirType::Int,
            }
        }
    }
}

fn place_type_raw(place: &Place, func: &MirFunction) -> MirType {
    match place {
        Place::Local(id) => func
            .locals
            .get(id.0)
            .map(|d| d.ty.clone())
            .unwrap_or(MirType::Int),
        Place::Index(base, _) => match place_type_raw(base, func) {
            MirType::Array(inner) => *inner,
            _ => MirType::Int,
        },
        Place::Field(base, fidx) => match place_type_raw(base, func) {
            MirType::Tuple(elems) => elems.get(*fidx).cloned().unwrap_or(MirType::Int),
            _ => MirType::Int,
        },
    }
}

fn is_float_binop(lhs: &Operand, rhs: &Operand, func: &MirFunction) -> bool {
    matches!(operand_mir_type(lhs, func), MirType::Float)
        || matches!(operand_mir_type(rhs, func), MirType::Float)
}

fn is_safe_const_int_divisor(op: &Operand) -> bool {
    matches!(op, Operand::Constant(MirConstant::Int(c)) if *c != 0 && *c != -1)
}

fn operand_exact_local(op: &Operand) -> Option<LocalId> {
    match op {
        Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)) => Some(*l),
        _ => None,
    }
}

fn operand_root_local(op: &Operand) -> Option<LocalId> {
    match op {
        Operand::Copy(p) | Operand::Move(p) => Some(p.root_local()),
        Operand::Constant(_) => None,
    }
}

fn count_operand_uses(op: &Operand, counts: &mut HashMap<LocalId, usize>) {
    if let Operand::Copy(p) | Operand::Move(p) = op {
        count_place_uses(p, counts, true);
    }
}

fn count_local_uses(func: &MirFunction) -> HashMap<LocalId, usize> {
    let mut counts: HashMap<LocalId, usize> = HashMap::new();
    for block in &func.blocks {
        for stmt in &block.statements {
            if let Statement::Assign(place, rval, _) = stmt {
                if !matches!(place, Place::Local(_)) {
                    count_place_uses(place, &mut counts, true);
                }
                match rval {
                    Rvalue::Use(op) | Rvalue::UnaryOp(_, op) | Rvalue::Cast(op, _) => {
                        count_operand_uses(op, &mut counts);
                    }
                    Rvalue::BinaryOp(_, l, r) => {
                        count_operand_uses(l, &mut counts);
                        count_operand_uses(r, &mut counts);
                    }
                    Rvalue::Aggregate(_, ops) => {
                        for op in ops {
                            count_operand_uses(op, &mut counts);
                        }
                    }
                    Rvalue::Closure { captures, .. } => {
                        for (_, op, _) in captures {
                            count_operand_uses(op, &mut counts);
                        }
                    }
                }
            }
        }
        if let Some(term) = &block.terminator {
            match term {
                Terminator::Cond { cond, .. } => count_operand_uses(cond, &mut counts),
                Terminator::Switch { discr, .. } => count_operand_uses(discr, &mut counts),
                Terminator::Call {
                    args, destination, ..
                } => {
                    for a in args {
                        count_operand_uses(a, &mut counts);
                    }
                    if !matches!(destination, Place::Local(_)) {
                        count_place_uses(destination, &mut counts, true);
                    }
                }
                Terminator::Return(Some(op), _) | Terminator::Raise(op, _) => {
                    count_operand_uses(op, &mut counts);
                }
                _ => {}
            }
        }
    }
    counts
}

fn count_place_uses(place: &Place, counts: &mut HashMap<LocalId, usize>, include_root: bool) {
    match place {
        Place::Local(id) => {
            if include_root {
                *counts.entry(*id).or_insert(0) += 1;
            }
        }
        Place::Field(base, _) => count_place_uses(base, counts, include_root),
        Place::Index(base, idx) => {
            count_place_uses(base, counts, include_root);
            *counts.entry(*idx).or_insert(0) += 1;
        }
    }
}

fn param_is_reassigned_after(
    func: &MirFunction,
    local: LocalId,
    init_bb: BlockId,
    init_sidx: usize,
) -> bool {
    for block in &func.blocks {
        for (sidx, stmt) in block.statements.iter().enumerate() {
            if block.id == init_bb && sidx == init_sidx {
                continue;
            }
            if let Statement::Assign(place, _, _) = stmt
                && place.root_local() == local
            {
                return true;
            }
        }
        if let Some(Terminator::Call { destination, .. }) = &block.terminator
            && destination.root_local() == local
        {
            return true;
        }
    }
    false
}

fn node_always_returns(node: &StructuredNode) -> bool {
    match node {
        StructuredNode::Return(_, _) => true,
        StructuredNode::Seq(items) => items.last().is_some_and(node_always_returns),
        StructuredNode::If {
            then_branch,
            else_branch: Some(eb),
            ..
        }
        | StructuredNode::IfArrayGetOk {
            then_branch,
            else_branch: Some(eb),
            ..
        } => node_always_returns(then_branch) && node_always_returns(eb),
        StructuredNode::Switch { cases, default, .. } => {
            cases.iter().all(|(_, n)| node_always_returns(n)) && node_always_returns(default)
        }
        _ => false,
    }
}

fn sanitize_ident(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else if ch == ':' {
            out.push_str("__");
        } else {
            out.push('_');
        }
    }
    out
}

fn rust_fn_name(name: &str) -> String {
    if name == "main" || name.ends_with(".main") {
        "__lust_fn_main".to_string()
    } else {
        format!("__lust_fn_{}", sanitize_ident(name))
    }
}

fn rust_struct_name(name: &str) -> String {
    format!("LustStruct_{}", sanitize_ident(name))
}

fn rust_enum_name(name: &str) -> String {
    format!("LustEnum_{}", sanitize_ident(name))
}

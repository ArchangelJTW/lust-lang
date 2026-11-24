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
use crate::data::materialize::struct_def_to_layout;
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

/// Extended Cargo & Zero-FFI interop options for Rust AOT emission (`NG-W13-1C`).
#[derive(Debug, Clone, Default)]
pub struct RustInteropOptions {
    /// Sanitized Rust crate identifiers linked directly as `rlib` (emitted as `extern crate <ext>;`).
    pub linked_crates: Vec<String>,
    /// Emit as an `include!()`-compatible module (no crate-level `#![...]` inner attributes and no `fn main()`).
    pub emit_as_include_module: bool,
    /// Emit ergonomic `pub fn` wrapper APIs for user-defined Lust functions (`--emit=rust-crate` and `RustAotBuilder`).
    pub export_public_api: bool,
}

/// Runs the mandatory MIR optimization, SROA, type refinement, and escape-analysis pipeline
/// required before Rust AOT codegen (`NG-W12-4A` + `NG-W12-4B`).
pub fn prepare_mir_module_for_rust_aot(module: &MirModule) -> MirModule {
    prepare_mir_module_for_rust_aot_with_extern_rets(module, &HashMap::new())
}

fn prepare_mir_module_for_rust_aot_with_extern_rets(
    module: &MirModule,
    extern_ret_types: &HashMap<String, MirType>,
) -> MirModule {
    let mut opt_module = module.clone();

    // 0. Canonicalize unqualified callee names against module-qualified function names
    //    (e.g. ModuleLoader qualifies `step` -> `calls.step`, while call sites say `step`).
    //    Doing this before MirOptimizer enables MirInliner, SROA, apply_module_callee_return_types,
    //    compute_readonly_params, and analyze_module_call_effects to match every intra-module call.
    canonicalize_module_callees(&mut opt_module);

    // 0b. Pre-populate return types for `extern` calls (AstLowering defaults unknown callees to Int).
    if !extern_ret_types.is_empty() {
        apply_extern_callee_return_types(&mut opt_module, extern_ret_types);
    }

    // 1. Run MirOptimizer while preserving `cleanup_blocks` and closure/effect lifting invariants
    //    (`AOT-BUDGET-SOUND`: selectively run module-level inlining on all eligible functions
    //    even when sibling functions contain `cleanup_blocks` or `Rvalue::Closure`).
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
        // Partition functions into module-inlining-eligible vs closure/cleanup-bearing.
        let eligible_indices: Vec<usize> = opt_module
            .functions
            .iter()
            .enumerate()
            .filter_map(|(idx, f)| {
                if fn_is_eligible_for_selective_module_opt(f) {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();
        let eligible_set: HashSet<usize> = eligible_indices.iter().copied().collect();

        if eligible_indices.len() >= 2 {
            let mut sub_mod = MirModule {
                name: opt_module.name.clone(),
                functions: eligible_indices
                    .iter()
                    .map(|&idx| opt_module.functions[idx].clone())
                    .collect(),
                struct_layouts: opt_module.struct_layouts.clone(),
                struct_ops: opt_module.struct_ops.clone(),
                enum_defs: opt_module.enum_defs.clone(),
            };
            MirOptimizer::optimize_module(&mut sub_mod);
            for (sub_i, &orig_idx) in eligible_indices.iter().enumerate() {
                opt_module.functions[orig_idx] = sub_mod.functions[sub_i].clone();
            }
        } else {
            for &idx in &eligible_indices {
                MirOptimizer::optimize_function(&mut opt_module.functions[idx]);
            }
        }

        // Detach cleanup subgraphs before per-function optimization on remaining functions,
        // then re-attach with remapped BlockIds so DCE never drops `is_cleanup == true` blocks.
        let mut saved_cleanups: Vec<(usize, Vec<(MirCleanupKind, Vec<BasicBlock>)>)> = Vec::new();
        for (idx, func) in opt_module.functions.iter_mut().enumerate() {
            if eligible_set.contains(&idx) {
                continue;
            }
            saved_cleanups.push((idx, extract_cleanup_subgraphs(func)));
            MirOptimizer::optimize_function(func);
        }
        for (idx, cleanups) in saved_cleanups {
            reattach_cleanup_subgraphs(&mut opt_module.functions[idx], cleanups);
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
    let mut callee_rets: HashMap<String, MirType> = opt_module
        .functions
        .iter()
        .map(|f| (f.signature.name.clone(), f.signature.return_type.clone()))
        .collect();
    for (k, v) in extern_ret_types {
        callee_rets.entry(k.clone()).or_insert_with(|| v.clone());
    }
    // Snapshot declared local types for `enum.payload` and builtin/string calls before Cranelift's
    // lattice refiner (which hardcodes unknown callees -> Int and can conflict declared types into Unknown).
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
    let pre_refine_named_container_tys: Vec<Vec<(LocalId, MirType)>> = opt_module
        .functions
        .iter()
        .map(|f| {
            f.locals
                .iter()
                .filter_map(|d| {
                    if d.name.is_some()
                        && matches!(
                            d.ty,
                            MirType::Array(_) | MirType::Map(_, _) | MirType::String | MirType::Option(_)
                        )
                    {
                        Some((d.id, d.ty.clone()))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .collect();

    let pre_refine_explicit_option_locals: Vec<HashSet<LocalId>> = opt_module
        .functions
        .iter()
        .map(|f| {
            f.locals
                .iter()
                .filter_map(|d| matches!(d.ty, MirType::Option(_)).then_some(d.id))
                .collect()
        })
        .collect();
    let pre_refine_unknown_map_kv: Vec<(HashSet<LocalId>, HashSet<LocalId>)> = opt_module
        .functions
        .iter()
        .map(|f| {
            let mut unk_k = HashSet::new();
            let mut unk_v = HashSet::new();
            for d in &f.locals {
                if let MirType::Map(k, v) = &d.ty {
                    if **k == MirType::Unknown {
                        unk_k.insert(d.id);
                    }
                    if **v == MirType::Unknown {
                        unk_v.insert(d.id);
                    }
                }
            }
            (unk_k, unk_v)
        })
        .collect();

    for func in &mut opt_module.functions {
        apply_module_callee_return_types(func, &callee_rets);
        *func = refine_mir_function_with_layouts(func, &struct_layouts);
    }

    if !extern_ret_types.is_empty() {
        apply_extern_callee_return_types(&mut opt_module, extern_ret_types);
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
    for (func, saved) in opt_module
        .functions
        .iter_mut()
        .zip(pre_refine_named_container_tys)
    {
        for (loc, ty) in saved {
            if matches!(func.locals[loc.0].ty, MirType::Unknown | MirType::Int) {
                func.locals[loc.0].ty = ty;
            }
        }
    }

    // 3a. Normalize built-in `LuaValue` pseudo-struct/enum types to `MirType::Unknown` (`AotValue`)
    //     whenever `LuaValue` is not a user-declared struct/enum layout (`NG-W15-1`).
    normalize_builtin_luavalue_types(&mut opt_module);

    // 3b. Tier 4 type refinement for Map<K, V>, Enum, Closure, and dynamic AotValue locals
    refine_tier4_types(
        &mut opt_module,
        &pre_refine_explicit_option_locals,
        &pre_refine_unknown_map_kv,
    );

    // 4. Run EscapeAnalysis across the refined module
    EscapeAnalysis::run_on_module(&mut opt_module);

    opt_module
}

fn normalize_mir_type_luavalue(
    ty: &mut MirType,
    known_structs: &HashSet<String>,
    known_enums: &HashSet<String>,
) {
    match ty {
        MirType::Struct(name) if name == "LuaValue" && !known_structs.contains(name) => {
            *ty = MirType::Unknown;
        }
        MirType::Enum(name) if name == "LuaValue" && !known_enums.contains(name) => {
            *ty = MirType::Unknown;
        }
        MirType::Array(inner) | MirType::Option(inner) => {
            normalize_mir_type_luavalue(inner, known_structs, known_enums);
        }
        MirType::Map(k, v) => {
            normalize_mir_type_luavalue(k, known_structs, known_enums);
            normalize_mir_type_luavalue(v, known_structs, known_enums);
        }
        MirType::Tuple(elems) => {
            for e in elems {
                normalize_mir_type_luavalue(e, known_structs, known_enums);
            }
        }
        MirType::Function {
            params,
            return_type,
        } => {
            for p in params {
                normalize_mir_type_luavalue(p, known_structs, known_enums);
            }
            normalize_mir_type_luavalue(return_type, known_structs, known_enums);
        }
        _ => {}
    }
}

fn normalize_builtin_luavalue_types(module: &mut MirModule) {
    let known_structs: HashSet<String> = module
        .struct_layouts
        .iter()
        .map(|l| l.type_name.clone())
        .collect();
    let known_enums: HashSet<String> = module.enum_defs.iter().map(|e| e.name.clone()).collect();
    for func in &mut module.functions {
        for p in &mut func.signature.param_types {
            normalize_mir_type_luavalue(p, &known_structs, &known_enums);
        }
        normalize_mir_type_luavalue(&mut func.signature.return_type, &known_structs, &known_enums);
        for loc in &mut func.locals {
            normalize_mir_type_luavalue(&mut loc.ty, &known_structs, &known_enums);
        }
        for block in &mut func.blocks {
            for stmt in &mut block.statements {
                if let Statement::Assign(_, rval, _) = stmt {
                    match rval {
                        Rvalue::Aggregate(agg_ty, _) | Rvalue::Cast(_, agg_ty) => {
                            normalize_mir_type_luavalue(agg_ty, &known_structs, &known_enums);
                        }
                        Rvalue::Closure { captures, .. } => {
                            for (_, _, cty) in captures {
                                normalize_mir_type_luavalue(cty, &known_structs, &known_enums);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

fn fn_is_eligible_for_selective_module_opt(func: &MirFunction) -> bool {
    if !func.cleanup_blocks.is_empty() {
        return false;
    }
    let name = &func.signature.name;
    if name.contains("__lambda_")
        || name.starts_with("__lust_")
        || name.contains("__nontail_")
    {
        return false;
    }
    for block in &func.blocks {
        for stmt in &block.statements {
            if matches!(stmt, Statement::Assign(_, Rvalue::Closure { .. }, _)) {
                return false;
            }
        }
        if let Some(Terminator::Call { callee, .. }) = &block.terminator
            && matches!(
                callee.as_str(),
                "__lust_handle"
                    | "__lust_nontail_handle"
                    | "__lust_yield_op"
                    | "__lust_yield_completed"
                    | "__lust_perform"
                    | "perform"
                    | "__lust_resume"
                    | "resume"
                    | "__lust_nontail_resume"
                    | "__lust_provide"
                    | "__lust_scoped"
                    | "__lust_scope_finalizer"
                    | "__lust_scope_spawn"
                    | "__lust_run_sync"
            )
        {
            return false;
        }
    }
    true
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
fn refine_tier4_types(
    module: &mut MirModule,
    pre_refine_explicit_option_locals: &[HashSet<LocalId>],
    pre_refine_unknown_map_kv: &[(HashSet<LocalId>, HashSet<LocalId>)],
) {
    let enum_names: HashSet<String> = module.enum_defs.iter().map(|e| e.name.clone()).collect();
    let enum_defs_map: HashMap<String, MirEnumDef> = module
        .enum_defs
        .iter()
        .map(|e| (e.name.clone(), e.clone()))
        .collect();

    // 0b. Pre-refine `__lust_resume`/`resume` and `__lust_perform`/`perform` destinations
    //     so lifted handler arms and handle bodies infer their exact return types below.
    for func in &mut module.functions {
        refine_effect_call_local_types_in_fn(func);
    }

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

    for (fn_idx, func) in module.functions.iter_mut().enumerate() {
        // 2a-effect. Propagate concrete return types for `__lust_handle`, `__lust_provide`, `__lust_run_sync`.
        refine_effect_handle_and_provide_types_in_fn(func, &module_fn_rets);

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
                        MirType::Function { .. }
                            | MirType::Map(_, _)
                            | MirType::Enum(_)
                            | MirType::Unit
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
                            && matches!(
                                st,
                                MirType::Function { .. } | MirType::Map(_, _) | MirType::Unit
                            )
                        {
                            func.locals[dst.0].ty = st.clone();
                            changed = true;
                        } else if let (
                            MirType::Function {
                                params: sp,
                                return_type: sr,
                            },
                            MirType::Function {
                                params: dp,
                                return_type: dr,
                            },
                        ) = (st, dt)
                            && sp == dp
                            && **dr == MirType::Unknown
                            && **sr != MirType::Unknown
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
                    "map.get" | "map.get_unwrap" | "map.get_unwrap_or" | "map.has"
                    | "map.delete" | "map.remove"
                        if args.len() >= 2 =>
                    {
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
                let unk_k = pre_refine_unknown_map_kv
                    .get(fn_idx)
                    .is_some_and(|(ks, _)| ks.contains(&loc));
                let unk_v = pre_refine_unknown_map_kv
                    .get(fn_idx)
                    .is_some_and(|(_, vs)| vs.contains(&loc));
                let final_k = if unk_k && map_key_conflict.contains(&loc) {
                    MirType::Unknown
                } else if *k == MirType::Unknown && !map_key_conflict.contains(&loc) {
                    map_key_ev
                        .get(&loc)
                        .and_then(|o| o.clone())
                        .unwrap_or(MirType::Unknown)
                } else {
                    *k
                };
                let final_v = if unk_v && map_val_conflict.contains(&loc) {
                    MirType::Unknown
                } else if *v == MirType::Unknown && !map_val_conflict.contains(&loc) {
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

        // Collect all locals assigned directly from a `map.get` temp, and inspect single-copy target types
        // for polymorphic string/builtin calls (e.g. `local res: Array<string> = string.gsub(...)`).
        let mut copy_dst_types: HashMap<LocalId, MirType> = HashMap::new();
        for block in &func.blocks {
            for stmt in &block.statements {
                if let Statement::Assign(
                    Place::Local(target),
                    Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))),
                    _,
                ) = stmt
                    && !matches!(func.locals[target.0].ty, MirType::Unknown | MirType::Int)
                {
                    copy_dst_types.insert(*src, func.locals[target.0].ty.clone());
                }
            }
        }
        let mut map_get_val_types: HashMap<LocalId, MirType> = HashMap::new();
        let mut call_dst_propagated: HashMap<LocalId, MirType> = HashMap::new();
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
                            // while `refine_mir_function_with_layouts` forces `Option(v_ty)`/`Enum("Option")`.
                            // Keep Option only if destination was ALREADY `MirType::Option(_)` pre-refinement.
                            let was_explicit_opt = pre_refine_explicit_option_locals
                                .get(fn_idx)
                                .is_some_and(|s| s.contains(dst));
                            if !was_explicit_opt {
                                func.locals[dst.0].ty = (*v_ty).clone();
                                map_get_val_types.insert(*dst, (*v_ty).clone());
                            }
                        }
                    }
                    "map.get_unwrap" | "map.get_unwrap_or" if !args.is_empty() => {
                        if let MirType::Map(_, v_ty) = operand_mir_type(&args[0], func) {
                            func.locals[dst.0].ty = (*v_ty).clone();
                            map_get_val_types.insert(*dst, (*v_ty).clone());
                        }
                    }
                    "string.sub" | "string.upper" | "string.lower" | "string.trim"
                    | "string.concat" => {
                        func.locals[dst.0].ty = MirType::String;
                        call_dst_propagated.insert(*dst, MirType::String);
                    }
                    "string.contains" | "string.starts_with" | "string.ends_with" => {
                        func.locals[dst.0].ty = MirType::Bool;
                        call_dst_propagated.insert(*dst, MirType::Bool);
                    }
                    "string.split" => {
                        let resolved = match copy_dst_types.get(dst) {
                            Some(MirType::Array(inner)) if matches!(**inner, MirType::Unknown) => {
                                MirType::Array(Box::new(MirType::Unknown))
                            }
                            Some(MirType::Unknown) => MirType::Unknown,
                            _ => MirType::Array(Box::new(MirType::String)),
                        };
                        func.locals[dst.0].ty = resolved.clone();
                        call_dst_propagated.insert(*dst, resolved);
                    }
                    "string.gsub" => {
                        let resolved = match copy_dst_types.get(dst) {
                            Some(MirType::Array(inner)) if matches!(**inner, MirType::String) => {
                                MirType::Array(Box::new(MirType::String))
                            }
                            Some(MirType::String) => MirType::String,
                            Some(MirType::Unknown) => MirType::Unknown,
                            _ => {
                                if matches!(func.locals[dst.0].ty, MirType::Array(ref inner) if matches!(**inner, MirType::String))
                                {
                                    MirType::Array(Box::new(MirType::String))
                                } else {
                                    MirType::Array(Box::new(MirType::Unknown))
                                }
                            }
                        };
                        func.locals[dst.0].ty = resolved.clone();
                        call_dst_propagated.insert(*dst, resolved);
                    }
                    "string.byte" | "string.find" => {
                        let resolved = match copy_dst_types.get(dst) {
                            Some(MirType::Array(inner)) if matches!(**inner, MirType::Int) => {
                                MirType::Array(Box::new(MirType::Int))
                            }
                            Some(MirType::Int) => MirType::Int,
                            Some(MirType::Unknown) => MirType::Unknown,
                            _ => {
                                if matches!(func.locals[dst.0].ty, MirType::Array(ref inner) if matches!(**inner, MirType::Int))
                                {
                                    MirType::Array(Box::new(MirType::Int))
                                } else {
                                    MirType::Array(Box::new(MirType::Unknown))
                                }
                            }
                        };
                        func.locals[dst.0].ty = resolved.clone();
                        call_dst_propagated.insert(*dst, resolved);
                    }
                    "__mir_unknown_is_int"
                    | "__mir_unknown_is_float"
                    | "__mir_unknown_is_string"
                    | "__mir_unknown_is_bool" => {
                        func.locals[dst.0].ty = MirType::Bool;
                        call_dst_propagated.insert(*dst, MirType::Bool);
                    }
                    "__mir_unknown_unwrap_int" => {
                        func.locals[dst.0].ty = MirType::Int;
                        call_dst_propagated.insert(*dst, MirType::Int);
                    }
                    "__mir_unknown_unwrap_float" => {
                        func.locals[dst.0].ty = MirType::Float;
                        call_dst_propagated.insert(*dst, MirType::Float);
                    }
                    "__mir_unknown_unwrap_string" => {
                        func.locals[dst.0].ty = MirType::String;
                        call_dst_propagated.insert(*dst, MirType::String);
                    }
                    "__mir_unknown_unwrap_bool" => {
                        func.locals[dst.0].ty = MirType::Bool;
                        call_dst_propagated.insert(*dst, MirType::Bool);
                    }
                    "__mir_unknown_get_field" => {
                        let mut refined = MirType::Unknown;
                        if args.len() >= 2
                            && let MirType::Struct(sname) = operand_mir_type(&args[0], func)
                            && let Operand::Constant(MirConstant::String(fname)) = &args[1]
                            && let Some(layout) = module
                                .struct_layouts
                                .iter()
                                .find(|l| l.type_name == sname)
                            && let Some(fdesc) = layout.fields.iter().find(|f| f.name == *fname)
                        {
                            refined = match fdesc.tag {
                                CanonicalTypeTag::Bool => MirType::Bool,
                                CanonicalTypeTag::Int64
                                | CanonicalTypeTag::UInt64
                                | CanonicalTypeTag::DateTimeUtc => MirType::Int,
                                CanonicalTypeTag::Float64 => MirType::Float,
                                CanonicalTypeTag::StringSlice | CanonicalTypeTag::BinarySlice => {
                                    MirType::String
                                }
                                _ => MirType::Unknown,
                            };
                        }
                        func.locals[dst.0].ty = refined.clone();
                        call_dst_propagated.insert(*dst, refined);
                    }
                    "__mir_unknown_call_method" => {
                        let mut refined = MirType::Unknown;
                        if args.len() >= 2
                            && let MirType::Struct(sname) = operand_mir_type(&args[0], func)
                            && let Operand::Constant(MirConstant::String(mname)) = &args[1]
                        {
                            let full = format!("{sname}:{mname}");
                            if let Some(rt) = module_fn_rets.get(&full) {
                                refined = rt.clone();
                            }
                        }
                        func.locals[dst.0].ty = refined.clone();
                        call_dst_propagated.insert(*dst, refined);
                    }
                    "Queue" | "sync.Queue" | "@std/sync.Queue" => {
                        func.locals[dst.0].ty = MirType::Unknown;
                        call_dst_propagated.insert(*dst, MirType::Unknown);
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
                        // Direct `StructName.decode(raw)` call
                        if let Some(sname) = callee.strip_suffix(".decode")
                            && module_fn_rets.get(callee).is_none()
                        {
                            func.locals[dst.0].ty = MirType::Struct(sname.to_string());
                        }
                        // EnumName.Variant or EnumName::Variant constructor calls
                        else if let Some((ename, _vname)) = callee
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

        // Propagate refined string/builtin/unknown call destination types into direct target copies
        if !call_dst_propagated.is_empty() {
            for block in &func.blocks {
                for stmt in &block.statements {
                    if let Statement::Assign(
                        Place::Local(dst),
                        Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))),
                        _,
                    ) = stmt
                        && let Some(cty) = call_dst_propagated.get(src)
                        && matches!(func.locals[dst.0].ty, MirType::Int | MirType::Unknown)
                        && !matches!(cty, MirType::Int)
                    {
                        func.locals[dst.0].ty = cty.clone();
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
                        && (matches!(
                            func.locals[dst.0].ty,
                            MirType::Unknown | MirType::Enum(_)
                        ) || (matches!(func.locals[dst.0].ty, MirType::Option(_))
                            && !pre_refine_explicit_option_locals
                                .get(fn_idx)
                                .is_some_and(|s| s.contains(dst))))
                    {
                        func.locals[dst.0].ty = v_ty.clone();
                    }
                }
            }
        }
    }

    // 3. Re-check lifted lambda return types in case a body lambda contained a nested `__lust_handle`
    //    or `__lust_run_sync` whose return type was just resolved in pass 2.
    let mut second_pass_rets: HashMap<String, MirType> = HashMap::new();
    for func in &mut module.functions {
        if matches!(func.signature.return_type, MirType::Unknown | MirType::Int) {
            let mut inferred: Option<MirType> = None;
            let mut conflict = false;
            for block in &func.blocks {
                if block.is_cleanup {
                    continue;
                }
                if let Some(Terminator::Return(Some(op), _)) = &block.terminator {
                    let ty = operand_mir_type(op, func);
                    if ty != MirType::Unknown {
                        match &inferred {
                            None => inferred = Some(ty),
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
                && let Some(rt) = inferred
                && (func.signature.return_type == MirType::Unknown
                    || matches!(rt, MirType::Unit | MirType::Function { .. }))
            {
                func.signature.return_type = rt.clone();
                if let Some(rl) = func.locals.get_mut(func.return_local.0) {
                    rl.ty = rt.clone();
                }
                second_pass_rets.insert(func.signature.name.clone(), rt);
            }
        }
    }
    if !second_pass_rets.is_empty() {
        let updated_rets: HashMap<String, MirType> = module
            .functions
            .iter()
            .map(|f| (f.signature.name.clone(), f.signature.return_type.clone()))
            .collect();
        for func in &mut module.functions {
            refine_effect_handle_and_provide_types_in_fn(func, &updated_rets);
        }
    }

    // 4. NG-W13-1B: Refine parameterized `EffectYield<OpArg, ResumeVal, BodyRet>` types
    //    and 3-phase `AotScope` builtin destinations across nested continuations & arms.
    refine_nontail_and_scope_types(module);

    // 5. NG-W15-1: Propagate `MirType::Function` to function-typed parameters & local closures
    //    (e.g. `apply_twice(f: function(int): int, x: int): int` in `BENCH-NG-04` where
    //    `mir_type_from_ast` initially lowered `TypeKind::Function` to `MirType::Int`).
    refine_function_typed_params(module);
}

fn refine_function_typed_params(module: &mut MirModule) {
    let module_fn_names: HashSet<String> = module
        .functions
        .iter()
        .map(|f| f.signature.name.clone())
        .collect();

    // Pass A: Propagate `MirType::Function` from call sites (direct module calls AND local closure
    // calls like `apply_twice(step_fn, acc)`) into callee parameters.
    let mut param_updates: HashMap<(String, usize), MirType> = HashMap::new();
    for func in &module.functions {
        // Map local variable name -> (lifted_func_name, n_captures) for closures defined in `func`
        let mut local_closure_target: HashMap<String, (String, usize)> = HashMap::new();
        let mut local_id_to_closure: HashMap<LocalId, (String, usize)> = HashMap::new();
        for block in &func.blocks {
            for stmt in &block.statements {
                if let Statement::Assign(
                    Place::Local(dst),
                    Rvalue::Closure {
                        func_name,
                        captures,
                    },
                    _,
                ) = stmt
                {
                    local_id_to_closure.insert(*dst, (func_name.clone(), captures.len()));
                    if let Some(n) = &func.locals[dst.0].name {
                        local_closure_target.insert(n.clone(), (func_name.clone(), captures.len()));
                    }
                }
            }
        }
        for _ in 0..3 {
            for block in &func.blocks {
                for stmt in &block.statements {
                    if let Statement::Assign(
                        Place::Local(dst),
                        Rvalue::Use(
                            Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)),
                        ),
                        _,
                    ) = stmt
                        && let Some(target) = local_id_to_closure.get(src).cloned()
                    {
                        local_id_to_closure.insert(*dst, target.clone());
                        if let Some(n) = &func.locals[dst.0].name {
                            local_closure_target.insert(n.clone(), target);
                        }
                    }
                }
            }
        }

        for block in &func.blocks {
            if let Some(Terminator::Call { callee, args, .. }) = &block.terminator {
                if module_fn_names.contains(callee.as_str()) {
                    for (ai, arg) in args.iter().enumerate() {
                        let aty = operand_mir_type(arg, func);
                        if matches!(aty, MirType::Function { .. }) {
                            param_updates.entry((callee.clone(), ai)).or_insert(aty);
                        }
                    }
                } else if let Some((lifted_name, n_caps)) = local_closure_target.get(callee) {
                    for (ai, arg) in args.iter().enumerate() {
                        let aty = operand_mir_type(arg, func);
                        if matches!(aty, MirType::Function { .. }) {
                            param_updates
                                .entry((lifted_name.clone(), n_caps + ai))
                                .or_insert(aty);
                        }
                    }
                }
            }
        }
    }

    for func in &mut module.functions {
        let fname = func.signature.name.clone();
        for (pi, &pid) in func.params.iter().enumerate() {
            if let Some(new_ty) = param_updates.get(&(fname.clone(), pi))
                && matches!(func.locals[pid.0].ty, MirType::Int | MirType::Unknown)
            {
                func.locals[pid.0].ty = new_ty.clone();
                if pi < func.signature.param_types.len() {
                    func.signature.param_types[pi] = new_ty.clone();
                }
            }
        }
    }

    // Pass B: Infer `MirType::Function` for any remaining local/parameter invoked directly by name
    // (`Terminator::Call { callee: local_name, .. }`) whose type is still `MirType::Int`.
    for func in &mut module.functions {
        let mut inferred: Vec<(LocalId, MirType)> = Vec::new();
        for block in &func.blocks {
            if let Some(Terminator::Call {
                callee,
                args,
                destination,
                ..
            }) = &block.terminator
            {
                if module_fn_names.contains(callee.as_str()) {
                    continue;
                }
                if let Some(decl) = func
                    .locals
                    .iter()
                    .find(|d| d.name.as_deref() == Some(callee.as_str()))
                    && matches!(decl.ty, MirType::Int)
                {
                    let arg_tys: Vec<MirType> = args
                        .iter()
                        .map(|a| operand_mir_type(a, func))
                        .collect();
                    let ret_ty = match destination {
                        Place::Local(d) => func.locals[d.0].ty.clone(),
                        _ => MirType::Int,
                    };
                    inferred.push((
                        decl.id,
                        MirType::Function {
                            params: arg_tys,
                            return_type: Box::new(ret_ty),
                        },
                    ));
                }
            }
        }
        for (loc, fn_ty) in inferred {
            func.locals[loc.0].ty = fn_ty.clone();
            if let Some(pi) = func.params.iter().position(|&p| p == loc)
                && pi < func.signature.param_types.len()
            {
                func.signature.param_types[pi] = fn_ty;
            }
        }
    }

    // Pass C: Refresh `Rvalue::Closure` local types and copies in callers so closure `params`
    // match the updated lifted function parameter types (e.g. `apply_twice`).
    let lifted_sigs: HashMap<String, (Vec<MirType>, MirType)> = module
        .functions
        .iter()
        .map(|f| {
            let ptys: Vec<MirType> = f
                .params
                .iter()
                .map(|pid| f.locals[pid.0].ty.clone())
                .collect();
            (
                f.signature.name.clone(),
                (ptys, f.signature.return_type.clone()),
            )
        })
        .collect();

    for func in &mut module.functions {
        let mut updated_closures = false;
        for block in &func.blocks {
            for stmt in &block.statements {
                if let Statement::Assign(
                    Place::Local(dst),
                    Rvalue::Closure {
                        func_name,
                        captures,
                    },
                    _,
                ) = stmt
                    && let Some((all_params, ret_ty)) = lifted_sigs.get(func_name)
                {
                    let explicit_params: Vec<MirType> =
                        all_params.iter().skip(captures.len()).cloned().collect();
                    let new_fn_ty = MirType::Function {
                        params: explicit_params,
                        return_type: Box::new(ret_ty.clone()),
                    };
                    if func.locals[dst.0].ty != new_fn_ty
                        && matches!(func.locals[dst.0].ty, MirType::Function { .. })
                    {
                        func.locals[dst.0].ty = new_fn_ty;
                        updated_closures = true;
                    }
                }
            }
        }
        if updated_closures {
            for _ in 0..4 {
                let local_tys: Vec<MirType> = func.locals.iter().map(|d| d.ty.clone()).collect();
                for block in &func.blocks {
                    for stmt in &block.statements {
                        if let Statement::Assign(
                            Place::Local(dst),
                            Rvalue::Use(
                                Operand::Copy(Place::Local(src))
                                | Operand::Move(Place::Local(src)),
                            ),
                            _,
                        ) = stmt
                            && matches!(local_tys[src.0], MirType::Function { .. })
                            && matches!(
                                local_tys[dst.0],
                                MirType::Function { .. } | MirType::Unknown | MirType::Int
                            )
                        {
                            func.locals[dst.0].ty = local_tys[src.0].clone();
                        }
                    }
                }
            }
        }
    }
}

fn trace_mir_closure_and_caps(
    func: &MirFunction,
    op: &Operand,
) -> Option<(String, Vec<(String, Operand, MirType)>)> {
    let mut def_loc: HashMap<LocalId, (BlockId, usize)> = HashMap::new();
    for block in &func.blocks {
        if block.is_cleanup {
            continue;
        }
        for (sidx, stmt) in block.statements.iter().enumerate() {
            if let Statement::Assign(Place::Local(dst), _, _) = stmt {
                def_loc.insert(*dst, (block.id, sidx));
            }
        }
    }
    let mut cur = operand_exact_local(op)?;
    let mut seen = HashSet::new();
    while seen.insert(cur) {
        let &(bb, sidx) = def_loc.get(&cur)?;
        let Statement::Assign(_, rval, _) = &func.blocks[bb.0].statements[sidx] else {
            return None;
        };
        match rval {
            Rvalue::Closure {
                func_name,
                captures,
            } => return Some((func_name.clone(), captures.clone())),
            Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))) => {
                cur = *src;
            }
            _ => return None,
        }
    }
    None
}

fn trace_mir_nontail_handle_site(
    func: &MirFunction,
    args: &[Operand],
) -> Option<(String, Vec<(String, String, usize)>)> {
    if args.len() != 2 {
        return None;
    }
    let (body_fn_name, _) = trace_mir_closure_and_caps(func, &args[0])?;
    let arms_loc = operand_exact_local(&args[1])?;

    let mut def_loc: HashMap<LocalId, (BlockId, usize)> = HashMap::new();
    for block in &func.blocks {
        if block.is_cleanup {
            continue;
        }
        for (sidx, stmt) in block.statements.iter().enumerate() {
            if let Statement::Assign(Place::Local(dst), _, _) = stmt {
                def_loc.insert(*dst, (block.id, sidx));
            }
        }
    }
    let &(arms_bb, arms_sidx) = def_loc.get(&arms_loc)?;
    let Statement::Assign(_, Rvalue::Aggregate(MirType::Array(_), tup_ops), _) =
        &func.blocks[arms_bb.0].statements[arms_sidx]
    else {
        return None;
    };

    let mut arms = Vec::with_capacity(tup_ops.len());
    for tup_op in tup_ops {
        let tup_loc = operand_exact_local(tup_op)?;
        let &(tup_bb, tup_sidx) = def_loc.get(&tup_loc)?;
        let Statement::Assign(_, Rvalue::Aggregate(MirType::Tuple(_), elems), _) =
            &func.blocks[tup_bb.0].statements[tup_sidx]
        else {
            return None;
        };
        if elems.len() < 2 {
            return None;
        }
        let op_name = match &elems[0] {
            Operand::Constant(MirConstant::String(s)) => s.clone(),
            _ => return None,
        };
        let (arm_fn_name, arm_caps) = trace_mir_closure_and_caps(func, &elems[1])?;
        arms.push((op_name, arm_fn_name, arm_caps.len()));
    }
    Some((body_fn_name, arms))
}

fn trace_mir_array_elems(func: &MirFunction, op: &Operand) -> Option<Vec<Operand>> {
    let arr_loc = operand_exact_local(op)?;
    for block in &func.blocks {
        if block.is_cleanup {
            continue;
        }
        for stmt in &block.statements {
            if let Statement::Assign(
                Place::Local(dst),
                Rvalue::Aggregate(MirType::Array(_), elems),
                _,
            ) = stmt
                && *dst == arr_loc
            {
                return Some(elems.clone());
            }
        }
    }
    None
}

/// Refines `OpArg`, `ResumeVal`, `BodyRet`, and scope builtin destination types across
/// `__lust_nontail_handle` / `__lust_yield_op` / `__lust_yield_completed` / `__lust_nontail_resume`
/// / `__lust_scoped` (`NG-W13-1B`).
fn refine_nontail_and_scope_types(module: &mut MirModule) {
    let has_nontail_or_scope = module.functions.iter().any(|f| {
        f.blocks.iter().any(|b| {
            matches!(
                &b.terminator,
                Some(Terminator::Call { callee, .. })
                    if matches!(
                        callee.as_str(),
                        "__lust_nontail_handle"
                            | "__lust_yield_op"
                            | "__lust_yield_completed"
                            | "__lust_nontail_resume"
                            | "__lust_scoped"
                            | "__lust_scope_finalizer"
                            | "__lust_scope_spawn"
                    )
            )
        })
    });
    if !has_nontail_or_scope {
        return;
    }

    // Set scope builtin call destinations to Unit
    for func in &mut module.functions {
        for block in &func.blocks {
            if let Some(Terminator::Call {
                callee,
                destination: Place::Local(dst),
                ..
            }) = &block.terminator
                && matches!(
                    callee.as_str(),
                    "__lust_scoped" | "__lust_scope_finalizer" | "__lust_scope_spawn"
                )
            {
                func.locals[dst.0].ty = MirType::Unit;
            }
        }
    }

    // Multi-pass fixed-point inference so nested sequential continuations propagate `BodyRet`
    // and `ResumeVal` from innermost continuation back to outer handle sites.
    for _ in 0..4 {
        let mut fn_index: HashMap<String, usize> = HashMap::new();
        for (idx, f) in module.functions.iter().enumerate() {
            fn_index.insert(f.signature.name.clone(), idx);
        }

        // Collect all `__lust_nontail_handle` sites: (caller_idx, dst_local, body_fn, arms)
        let mut sites: Vec<(usize, Option<LocalId>, String, Vec<(String, String, usize)>)> =
            Vec::new();
        for (fi, func) in module.functions.iter().enumerate() {
            for block in &func.blocks {
                if let Some(Terminator::Call {
                    callee,
                    args,
                    destination,
                    ..
                }) = &block.terminator
                    && callee == "__lust_nontail_handle"
                    && let Some((body_fn, arms)) = trace_mir_nontail_handle_site(func, args)
                {
                    let dst_loc = match destination {
                        Place::Local(l) => Some(*l),
                        _ => None,
                    };
                    sites.push((fi, dst_loc, body_fn, arms));
                }
            }
        }

        let mut changed = false;
        for (caller_idx, dst_loc, body_fn_name, arms) in sites {
            let Some(&body_idx) = fn_index.get(&body_fn_name) else {
                continue;
            };

            // Inspect body function for `__lust_yield_completed` and `__lust_yield_op`
            let mut body_ret_cand: Option<MirType> = None;
            let mut body_ret_conflict = false;
            let mut resume_val_cand: Option<MirType> = None;
            let mut resume_val_conflict = false;
            let mut op_arg_hints: HashMap<String, Vec<MirType>> = HashMap::new();
            let mut cont_fn_infos: Vec<(String, usize)> = Vec::new();

            let record_ty = |slot: &mut Option<MirType>, conflict: &mut bool, ty: MirType| {
                if ty == MirType::Unknown {
                    return;
                }
                match slot {
                    None => *slot = Some(ty),
                    Some(prev) if *prev == ty => {}
                    Some(_) => *conflict = true,
                }
            };

            {
                let body_fn = &module.functions[body_idx];
                for block in &body_fn.blocks {
                    if let Some(Terminator::Call { callee, args, .. }) = &block.terminator {
                        if callee == "__lust_yield_completed"
                            && let Some(arg0) = args.first()
                        {
                            let t = operand_mir_type(arg0, body_fn);
                            record_ty(&mut body_ret_cand, &mut body_ret_conflict, t);
                        } else if callee == "__lust_yield_op" && args.len() >= 3 {
                            if let Operand::Constant(MirConstant::String(op_name)) = &args[0]
                                && let Some(arg_elems) = trace_mir_array_elems(body_fn, &args[1])
                            {
                                let tys: Vec<MirType> = arg_elems
                                    .iter()
                                    .map(|e| operand_mir_type(e, body_fn))
                                    .collect();
                                op_arg_hints.entry(op_name.clone()).or_insert(tys);
                            }
                            if let Some((cont_name, cont_caps)) =
                                trace_mir_closure_and_caps(body_fn, &args[2])
                            {
                                cont_fn_infos.push((cont_name, cont_caps.len()));
                            }
                        }
                    }
                }
            }

            // Pull BodyRet and ResumeVal from continuation functions
            for (cont_name, n_caps) in &cont_fn_infos {
                if let Some(&ci) = fn_index.get(cont_name) {
                    let cf = &module.functions[ci];
                    if cf.signature.return_type != MirType::Unknown {
                        record_ty(
                            &mut body_ret_cand,
                            &mut body_ret_conflict,
                            cf.signature.return_type.clone(),
                        );
                    }
                    if let Some(&pid) = cf.params.get(*n_caps) {
                        let pty = cf.locals[pid.0].ty.clone();
                        let used = count_local_uses(cf).get(&pid).copied().unwrap_or(0) > 0;
                        if pty != MirType::Unknown && (used || pty != MirType::Int) {
                            record_ty(&mut resume_val_cand, &mut resume_val_conflict, pty);
                        }
                    }
                }
            }

            // Propagate OpArg hints into arm params and pull ResumeVal from `__lust_nontail_resume(v)`
            for (op_name, arm_fn_name, n_caps) in &arms {
                let Some(&arm_idx) = fn_index.get(arm_fn_name) else {
                    continue;
                };
                if let Some(arg_tys) = op_arg_hints.get(op_name) {
                    let arm_fn = &mut module.functions[arm_idx];
                    for (pi, aty) in arg_tys.iter().enumerate() {
                        if *aty != MirType::Unknown
                            && let Some(&pid) = arm_fn.params.get(*n_caps + pi)
                            && matches!(arm_fn.locals[pid.0].ty, MirType::Unknown | MirType::Int)
                            && arm_fn.locals[pid.0].ty != *aty
                        {
                            arm_fn.locals[pid.0].ty = aty.clone();
                            if let Some(sig_p) = arm_fn.signature.param_types.get_mut(*n_caps + pi) {
                                *sig_p = aty.clone();
                            }
                            changed = true;
                        }
                    }
                }
                let arm_fn = &module.functions[arm_idx];
                for block in &arm_fn.blocks {
                    if let Some(Terminator::Call { callee, args, .. }) = &block.terminator
                        && callee == "__lust_nontail_resume"
                        && let Some(arg0) = args.first()
                    {
                        let t = operand_mir_type(arg0, arm_fn);
                        record_ty(&mut resume_val_cand, &mut resume_val_conflict, t);
                    }
                }
            }

            let resolved_body_ret = if body_ret_conflict {
                MirType::Unknown
            } else {
                body_ret_cand.unwrap_or(MirType::Unit)
            };
            let resolved_resume_val = if resume_val_conflict {
                MirType::Unknown
            } else {
                resume_val_cand.unwrap_or(MirType::Unit)
            };

            // Propagate resolved BodyRet into every `__lust_nontail_resume` destination and arm return
            for (_, arm_fn_name, _) in &arms {
                let Some(&arm_idx) = fn_index.get(arm_fn_name) else {
                    continue;
                };
                let arm_fn = &mut module.functions[arm_idx];
                let mut resume_dsts: HashSet<LocalId> = HashSet::new();
                for block in &arm_fn.blocks {
                    if let Some(Terminator::Call {
                        callee,
                        destination: Place::Local(dst),
                        ..
                    }) = &block.terminator
                        && callee == "__lust_nontail_resume"
                    {
                        resume_dsts.insert(*dst);
                        if arm_fn.locals[dst.0].ty != resolved_body_ret {
                            arm_fn.locals[dst.0].ty = resolved_body_ret.clone();
                            changed = true;
                        }
                    }
                }
                if !resume_dsts.is_empty() {
                    // Propagate copy of resume destination into local bindings (e.g. `let r = resume(v)`)
                    for block in &arm_fn.blocks {
                        for stmt in &block.statements {
                            if let Statement::Assign(
                                Place::Local(dst),
                                Rvalue::Use(
                                    Operand::Copy(Place::Local(src))
                                    | Operand::Move(Place::Local(src)),
                                ),
                                _,
                            ) = stmt
                                && resume_dsts.contains(src)
                                && matches!(
                                    arm_fn.locals[dst.0].ty,
                                    MirType::Unknown | MirType::Unit
                                )
                                && resolved_body_ret != MirType::Unknown
                                && arm_fn.locals[dst.0].ty != resolved_body_ret
                            {
                                arm_fn.locals[dst.0].ty = resolved_body_ret.clone();
                                changed = true;
                            }
                        }
                    }
                }
                if matches!(arm_fn.signature.return_type, MirType::Unknown)
                    && resolved_body_ret != MirType::Unknown
                {
                    arm_fn.signature.return_type = resolved_body_ret.clone();
                    if let Some(rl) = arm_fn.locals.get_mut(arm_fn.return_local.0) {
                        rl.ty = resolved_body_ret.clone();
                    }
                    changed = true;
                }
            }

            // Propagate resolved ResumeVal into continuation explicit param if Unknown or unused Int placeholder
            for (cont_name, n_caps) in &cont_fn_infos {
                if let Some(&ci) = fn_index.get(cont_name) {
                    let cf = &mut module.functions[ci];
                    if let Some(&pid) = cf.params.get(*n_caps) {
                        let unused_placeholder = cf.locals[pid.0].ty == MirType::Int
                            && count_local_uses(cf).get(&pid).copied().unwrap_or(0) == 0;
                        if (matches!(cf.locals[pid.0].ty, MirType::Unknown) || unused_placeholder)
                            && resolved_resume_val != MirType::Unknown
                            && cf.locals[pid.0].ty != resolved_resume_val
                        {
                            cf.locals[pid.0].ty = resolved_resume_val.clone();
                            if let Some(sig_p) = cf.signature.param_types.get_mut(*n_caps) {
                                *sig_p = resolved_resume_val.clone();
                            }
                            changed = true;
                        }
                    }
                }
            }

            // Propagate resolved BodyRet into caller's `__lust_nontail_handle` destination & caller return
            if let Some(dst) = dst_loc {
                let caller = &mut module.functions[caller_idx];
                if caller.locals[dst.0].ty != resolved_body_ret
                    && (matches!(caller.locals[dst.0].ty, MirType::Unknown | MirType::Int)
                        || resolved_body_ret != MirType::Unknown)
                {
                    caller.locals[dst.0].ty = resolved_body_ret.clone();
                    changed = true;
                }
                // Re-infer caller return type if caller is a lifted continuation/lambda with Unknown return
                if matches!(caller.signature.return_type, MirType::Unknown) {
                    for block in &caller.blocks {
                        if let Some(Terminator::Return(Some(op), _)) = &block.terminator
                            && operand_exact_local(op) == Some(dst)
                            && resolved_body_ret != MirType::Unknown
                        {
                            caller.signature.return_type = resolved_body_ret.clone();
                            if let Some(rl) = caller.locals.get_mut(caller.return_local.0) {
                                rl.ty = resolved_body_ret.clone();
                            }
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
}

/// Refines `__lust_resume`/`resume` and `__lust_perform`/`perform` call destination types inside `func`.
fn refine_effect_call_local_types_in_fn(func: &mut MirFunction) {
    let mut perform_dsts: Vec<LocalId> = Vec::new();
    for block in &func.blocks {
        if let Some(Terminator::Call {
            callee,
            args,
            destination: Place::Local(dst),
            ..
        }) = &block.terminator
        {
            match callee.as_str() {
                "__lust_resume" | "resume" => {
                    if let Some(arg0) = args.first() {
                        let aty = operand_mir_type(arg0, func);
                        if aty != MirType::Unknown {
                            func.locals[dst.0].ty = aty;
                        }
                    }
                }
                "__lust_perform" | "perform" => {
                    perform_dsts.push(*dst);
                }
                _ => {}
            }
        }
    }

    if perform_dsts.is_empty() {
        return;
    }

    let perf_set: HashSet<LocalId> = perform_dsts.iter().copied().collect();
    let mut inferred_perf_ty: HashMap<LocalId, MirType> = HashMap::new();
    for block in &func.blocks {
        for stmt in &block.statements {
            match stmt {
                Statement::Assign(
                    Place::Local(dst),
                    Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))),
                    _,
                ) if perf_set.contains(src) => {
                    let dst_ty = &func.locals[dst.0].ty;
                    if !matches!(dst_ty, MirType::Unknown | MirType::Int) {
                        inferred_perf_ty.insert(*src, dst_ty.clone());
                    }
                }
                Statement::Assign(_, Rvalue::BinaryOp(MirBinaryOp::Concat, lhs, rhs), _) => {
                    if let Some(l) = operand_exact_local(lhs)
                        && perf_set.contains(&l)
                    {
                        inferred_perf_ty.insert(l, MirType::String);
                    }
                    if let Some(r) = operand_exact_local(rhs)
                        && perf_set.contains(&r)
                    {
                        inferred_perf_ty.insert(r, MirType::String);
                    }
                }
                _ => {}
            }
        }
        if let Some(Terminator::Return(Some(op), _)) = &block.terminator
            && let Some(l) = operand_exact_local(op)
            && perf_set.contains(&l)
            && !matches!(func.signature.return_type, MirType::Unknown)
        {
            inferred_perf_ty
                .entry(l)
                .or_insert_with(|| func.signature.return_type.clone());
        }
    }

    for (loc, ty) in inferred_perf_ty {
        func.locals[loc.0].ty = ty;
    }
}

/// Propagates concrete lifted-thunk return types into `__lust_handle`, `__lust_provide`, and `__lust_run_sync` destinations.
fn refine_effect_handle_and_provide_types_in_fn(
    func: &mut MirFunction,
    module_fn_rets: &HashMap<String, MirType>,
) {
    let mut closure_fn_of: HashMap<LocalId, String> = HashMap::new();
    for block in &func.blocks {
        for stmt in &block.statements {
            if let Statement::Assign(Place::Local(dst), rval, _) = stmt {
                match rval {
                    Rvalue::Closure { func_name, .. } => {
                        closure_fn_of.insert(*dst, func_name.clone());
                    }
                    Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))) => {
                        if let Some(fname) = closure_fn_of.get(src).cloned() {
                            closure_fn_of.insert(*dst, fname);
                        }
                    }
                    _ => {}
                }
            }
        }
        if let Some(Terminator::Call {
            callee,
            args,
            destination: Place::Local(dst),
            ..
        }) = &block.terminator
            && callee == "__lust_provide"
            && let Some(src) = args.first().and_then(operand_exact_local)
            && let Some(fname) = closure_fn_of.get(&src).cloned()
        {
            closure_fn_of.insert(*dst, fname);
        }
    }

    for block in &func.blocks {
        if let Some(Terminator::Call {
            callee,
            args,
            destination: Place::Local(dst),
            ..
        }) = &block.terminator
        {
            match callee.as_str() {
                "__lust_handle" | "__lust_run_sync" => {
                    if let Some(arg0) = args.first() {
                        if let Some(src) = operand_exact_local(arg0)
                            && let Some(fname) = closure_fn_of.get(&src)
                            && let Some(ret_ty) = module_fn_rets.get(fname)
                            && *ret_ty != MirType::Unknown
                        {
                            func.locals[dst.0].ty = ret_ty.clone();
                            continue;
                        }
                        match operand_mir_type(arg0, func) {
                            MirType::Function { return_type, .. }
                            | MirType::Effect { return_type, .. }
                                if *return_type != MirType::Unknown =>
                            {
                                func.locals[dst.0].ty = (*return_type).clone();
                            }
                            _ => {}
                        }
                    }
                }
                "__lust_provide" => {
                    if let Some(arg0) = args.first() {
                        let aty = operand_mir_type(arg0, func);
                        if aty != MirType::Unknown {
                            func.locals[dst.0].ty = aty;
                        }
                    }
                }
                _ => {}
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
    emit_rust_from_mir_with_ast(module, &[], opts)
}

/// Generates idiomatic, panic-free, `unsafe`-free Rust source code from a `MirModule`
/// together with direct AST items (`ServiceDef`, `TraitDef`, `ImplBlock`, `FunctionDef.effects`,
/// and desugared `__lust_handle`/`__lust_perform`/`__lust_resume` calls) without modifying
/// `src/mir/ir.rs` (`NG-W13-1A`, `F-19`).
pub fn emit_rust_from_mir_with_ast(
    module: &MirModule,
    ast_items: &[crate::ast::Item],
    opts: &RustEmitterOptions,
) -> Result<String> {
    emit_rust_from_mir_with_interop(module, ast_items, opts, &RustInteropOptions::default())
}

/// Generates idiomatic, panic-free, `unsafe`-free Rust source code from a `MirModule` + AST items
/// with Cargo/Zero-FFI interop options (`--emit=rust-crate`, `--link-crate`, `RustAotBuilder`,
/// `NG-W13-1C`).
pub fn emit_rust_from_mir_with_interop(
    module: &MirModule,
    ast_items: &[crate::ast::Item],
    opts: &RustEmitterOptions,
    interop: &RustInteropOptions,
) -> Result<String> {
    let ast_meta = AstEffectsMetadata::extract(ast_items);
    let mut module_with_layouts = module.clone();
    register_ast_struct_layouts_in_module(&mut module_with_layouts, ast_items);
    let extern_ret_types = build_extern_return_type_map(&module_with_layouts, &ast_meta);
    let prepared =
        prepare_mir_module_for_rust_aot_with_extern_rets(&module_with_layouts, &extern_ret_types);
    let ctx = ModuleEmitContext::analyze(&prepared, opts, interop, ast_meta);

    let mut out = String::with_capacity(8192);
    emit_file_header(&mut out, opts, interop);
    emit_struct_definitions(&mut out, &ctx);
    emit_enum_definitions(&mut out, &ctx);
    emit_trait_and_service_definitions(&mut out, &ctx);
    emit_suite_v2_runtime_helpers(&mut out, &prepared, &ctx);

    for func in &prepared.functions {
        emit_function(&mut out, func, &ctx)?;
    }

    if interop.export_public_api {
        emit_public_api_wrappers(&mut out, &prepared, &ctx);
    }

    if !opts.is_library && !interop.emit_as_include_module {
        emit_entrypoint(&mut out, &prepared, &ctx);
    }

    Ok(out)
}

fn register_ast_struct_layouts_in_module(module: &mut MirModule, items: &[crate::ast::Item]) {
    for item in items {
        match &item.kind {
            crate::ast::ItemKind::Struct(def) => {
                if !module
                    .struct_layouts
                    .iter()
                    .any(|l| l.type_name == def.name)
                {
                    let layout = struct_def_to_layout(def);
                    module.register_struct_layout(MirStructLayout::from(&layout));
                }
            }
            crate::ast::ItemKind::Module { items: sub, .. } => {
                register_ast_struct_layouts_in_module(module, sub);
            }
            _ => {}
        }
    }
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
    is_mutated_cyclic: bool,
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
    /// True when `!can_fail`, all cleanup blocks are `Defer` (no `ErrDefer`), and every
    /// return block already has the LIFO `AstLowering`-inlined defer suffix (`NG-W15-1`).
    keep_inlined_defers_direct: bool,
    return_type: MirType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum NonTailOpArgMode {
    /// Single arm with 0 explicit parameters -> `OpArg = ()`.
    Unit,
    /// Single arm with 1 explicit typed parameter -> `OpArg = T` (`i64`, `f64`, `bool`, `String`, `()`, `AotValue`).
    Single(MirType),
    /// Multiple arms or multi-arg operation -> `OpArg = (&'static str, alloc::vec::Vec<lust_aot_rt::AotValue>)`.
    TaggedVec,
}

#[derive(Debug, Clone)]
struct NonTailYieldSig {
    op_arg_mode: NonTailOpArgMode,
    resume_val_ty: MirType,
    body_ret_ty: MirType,
}

#[derive(Debug, Clone)]
struct NonTailArmMeta {
    resume_val_ty: MirType,
    body_ret_ty: MirType,
}

struct ModuleEmitContext<'m> {
    module: &'m MirModule,
    opts: &'m RustEmitterOptions,
    interop: &'m RustInteropOptions,
    ast_meta: AstEffectsMetadata,
    structs: BTreeMap<String, StructMeta>,
    /// Deterministic `1..=N` struct runtime type IDs for `AotValue::Struct` packing/dispatch (`NG-W15-1`).
    struct_type_ids: BTreeMap<String, u32>,
    /// Struct names actually referenced by `.decode` / `__lust_mir_struct_decode_binary:` (transitively closed).
    schema_decoded_structs: HashSet<String>,
    enums: BTreeMap<String, EnumMeta>,
    fn_metas: HashMap<String, FnMeta>,
    /// Lifted body thunks of non-tail `handle..with` returning `EffectYield<OpArg, ResumeVal, BodyRet>` (`NG-W13-1B`).
    nontail_body_metas: HashMap<String, NonTailYieldSig>,
    /// Lifted handler arms of non-tail `handle..with` receiving `&mut OneShotResume<ResumeVal, BodyRet>` (`NG-W13-1B`).
    nontail_arm_metas: HashMap<String, NonTailArmMeta>,
    module_uses_arena: bool,
    /// Whether any closure in the module captures a type requiring `'a`.
    closures_need_lt: bool,
}

impl<'m> ModuleEmitContext<'m> {
    fn analyze(
        module: &'m MirModule,
        opts: &'m RustEmitterOptions,
        interop: &'m RustInteropOptions,
        ast_meta: AstEffectsMetadata,
    ) -> Self {
        let structs = analyze_structs(module);
        let mut struct_type_ids = BTreeMap::new();
        for (idx, sname) in structs.keys().enumerate() {
            struct_type_ids.insert(sname.clone(), (idx + 1) as u32);
        }
        let enums = analyze_enums(module, &structs);
        let module_uses_arena = structs
            .values()
            .any(|s| s.is_cyclic && !s.is_mutated_cyclic);

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

        // 2a-effect. For functions containing effect/scope constructs (`__lust_handle`,
        //            `__lust_nontail_handle`, `__lust_yield_op`, `__lust_scoped`,
        //            `__lust_scope_finalizer`, `__lust_scope_spawn`, `__lust_run_sync`),
        //            transitively share mutated scalar locals as `Rc<Cell<T>>` and mutated
        //            non-Copy locals (e.g. `String`) as `Rc<RefCell<T>>` across enclosing
        //            functions and nested/sibling closures (`NG-W13-1A` + `NG-W13-1B`).
        //            Fast path (`NG-W15-1`, `F-02`): if `func` only invokes direct synchronous
        //            `__lust_run_sync` thunks (no handlers/scopes/providers) and a captured scalar
        //            local is not mutated by the lifted thunk, keep it `LocalStorage::Direct` and
        //            pass it `ByValue` at the call site.
        for _ in 0..4 {
            let mut effect_cap_changed = false;
            for func in &module.functions {
                let mut has_non_run_sync_effect = false;
                let mut has_any_run_sync = false;
                let mut all_run_sync_direct = true;
                let mut direct_run_sync_closures: HashSet<String> = HashSet::new();
                for b in &func.blocks {
                    if let Some(Terminator::Call { callee, args, .. }) = &b.terminator {
                        match callee.as_str() {
                            "__lust_handle"
                            | "__lust_nontail_handle"
                            | "__lust_yield_op"
                            | "__lust_scoped"
                            | "__lust_scope_finalizer"
                            | "__lust_scope_spawn"
                            | "__lust_provide" => {
                                has_non_run_sync_effect = true;
                            }
                            "__lust_run_sync" => {
                                has_any_run_sync = true;
                                if args.len() == 1
                                    && let Some(tfn) =
                                        trace_direct_closure_fn_name_in_mir(func, &args[0])
                                {
                                    direct_run_sync_closures.insert(tfn);
                                } else {
                                    all_run_sync_direct = false;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                let has_effect_or_scope = has_non_run_sync_effect || has_any_run_sync;
                if !has_effect_or_scope {
                    continue;
                }
                let direct_run_sync_only =
                    !has_non_run_sync_effect && has_any_run_sync && all_run_sync_direct;

                let enclosing_name = func.signature.name.clone();
                let mut mutated_scalar_caps: HashSet<LocalId> = HashSet::new();
                let mut mutated_noncopy_caps: HashSet<LocalId> = HashSet::new();
                let mut all_closure_caps: Vec<(String, usize, LocalId, MirType)> = Vec::new();
                let mut captured_by_non_direct_closure: HashSet<LocalId> = HashSet::new();
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
                            && !direct_run_sync_closures.contains(func_name)
                        {
                            for (_, cap_op, _) in captures {
                                if let Some(loc) = operand_exact_local(cap_op) {
                                    captured_by_non_direct_closure.insert(loc);
                                }
                            }
                        }
                    }
                }
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
                                let is_scalar =
                                    matches!(cap_ty, MirType::Int | MirType::Float | MirType::Bool);
                                let is_mut_noncopy = matches!(
                                    cap_ty,
                                    MirType::String | MirType::Unknown | MirType::Enum(_)
                                );
                                if !is_scalar && !is_mut_noncopy {
                                    continue;
                                }
                                if let Some(loc) = operand_exact_local(cap_op) {
                                    all_closure_caps.push((
                                        func_name.clone(),
                                        ci,
                                        loc,
                                        cap_ty.clone(),
                                    ));
                                    let lifted_mutates = lifted_fn.is_some_and(|lf| {
                                        lf.params.get(ci).is_some_and(|&pid| {
                                            local_is_mutated_in_fn(lf, pid)
                                                || local_storage_map
                                                    .get(func_name.as_str())
                                                    .and_then(|v| v.get(pid.0))
                                                    .is_some_and(|s| *s != LocalStorage::Direct)
                                        })
                                    });
                                    let enclosing_already_shared = local_storage_map
                                        .get(&enclosing_name)
                                        .and_then(|v| v.get(loc.0))
                                        .is_some_and(|s| *s != LocalStorage::Direct);
                                    let enclosing_mutates = local_is_mutated_in_fn(func, loc);
                                    let can_keep_direct_scalar = direct_run_sync_only
                                        && is_scalar
                                        && direct_run_sync_closures.contains(func_name)
                                        && !lifted_mutates
                                        && !enclosing_already_shared
                                        && !captured_by_non_direct_closure.contains(&loc);
                                    if !can_keep_direct_scalar
                                        && (lifted_mutates
                                            || enclosing_mutates
                                            || enclosing_already_shared)
                                    {
                                        if is_scalar {
                                            mutated_scalar_caps.insert(loc);
                                        } else {
                                            mutated_noncopy_caps.insert(loc);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for loc in &mutated_scalar_caps {
                    if let Some(vec_st) = local_storage_map.get_mut(&enclosing_name)
                        && loc.0 < vec_st.len()
                        && vec_st[loc.0] != LocalStorage::RcCell
                    {
                        vec_st[loc.0] = LocalStorage::RcCell;
                        effect_cap_changed = true;
                    }
                    if let Some(pi) = func.params.iter().position(|&p| p == *loc)
                        && let Some(pms) = param_modes_map.get_mut(&enclosing_name)
                        && pi < pms.len()
                        && pms[pi] != ParamMode::RcCell
                    {
                        pms[pi] = ParamMode::RcCell;
                        effect_cap_changed = true;
                    }
                }
                for loc in &mutated_noncopy_caps {
                    if let Some(vec_st) = local_storage_map.get_mut(&enclosing_name)
                        && loc.0 < vec_st.len()
                        && vec_st[loc.0] != LocalStorage::RcRefCell
                    {
                        vec_st[loc.0] = LocalStorage::RcRefCell;
                        effect_cap_changed = true;
                    }
                    if let Some(pi) = func.params.iter().position(|&p| p == *loc)
                        && let Some(pms) = param_modes_map.get_mut(&enclosing_name)
                        && pi < pms.len()
                        && pms[pi] != ParamMode::RcRefCell
                    {
                        pms[pi] = ParamMode::RcRefCell;
                        effect_cap_changed = true;
                    }
                }
                for (func_name, ci, loc, _) in all_closure_caps {
                    if mutated_scalar_caps.contains(&loc) {
                        if let Some(lmodes) = param_modes_map.get_mut(&func_name)
                            && ci < lmodes.len()
                            && lmodes[ci] != ParamMode::RcCell
                        {
                            lmodes[ci] = ParamMode::RcCell;
                            effect_cap_changed = true;
                        }
                        if let Some(lf) = fn_by_name.get(func_name.as_str()).copied()
                            && let Some(&pid) = lf.params.get(ci)
                            && let Some(lvec) = local_storage_map.get_mut(&func_name)
                            && pid.0 < lvec.len()
                            && lvec[pid.0] != LocalStorage::RcCell
                        {
                            lvec[pid.0] = LocalStorage::RcCell;
                            effect_cap_changed = true;
                        }
                    } else if mutated_noncopy_caps.contains(&loc) {
                        if let Some(lmodes) = param_modes_map.get_mut(&func_name)
                            && ci < lmodes.len()
                            && lmodes[ci] != ParamMode::RcRefCell
                        {
                            lmodes[ci] = ParamMode::RcRefCell;
                            effect_cap_changed = true;
                        }
                        if let Some(lf) = fn_by_name.get(func_name.as_str()).copied()
                            && let Some(&pid) = lf.params.get(ci)
                            && let Some(lvec) = local_storage_map.get_mut(&func_name)
                            && pid.0 < lvec.len()
                            && lvec[pid.0] != LocalStorage::RcRefCell
                        {
                            lvec[pid.0] = LocalStorage::RcRefCell;
                            effect_cap_changed = true;
                        }
                    }
                }
            }
            if !effect_cap_changed {
                break;
            }
        }

        // 2b + 2c + 2d + 3: Monotonic convergence between infallible inlined `Defer` fast path
        // (`NG-W15-1`, `F-02`) and cleanup-guard `RcCell`/`RcRefCell` promotion.
        let (nontail_body_metas, nontail_arm_metas) = analyze_nontail_effect_metas(module);
        let mut direct_defer_candidates: HashSet<String> = module
            .functions
            .iter()
            .filter(|f| {
                !nontail_body_metas.contains_key(&f.signature.name)
                    && !nontail_arm_metas.contains_key(&f.signature.name)
                    && function_inlined_defers_cover_all_returns(f)
            })
            .map(|f| f.signature.name.clone())
            .collect();

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

        let mut promoted_cleanup_fns: HashSet<String> = HashSet::new();
        let mut can_fail_map: HashMap<String, bool> = HashMap::new();
        let mut needs_ctx_map: HashMap<String, bool> = HashMap::new();
        let mut needs_alloc_map: HashMap<String, bool> = HashMap::new();
        let mut needs_arena_map: HashMap<String, bool> = HashMap::new();

        loop {
            // 2b. Cleanup block (`defer` / `errdefer`) capture analysis for functions that need guards
            for func in &module.functions {
                if func.cleanup_blocks.is_empty() {
                    continue;
                }
                let fname = &func.signature.name;
                if direct_defer_candidates.contains(fname) || promoted_cleanup_fns.contains(fname) {
                    continue;
                }
                promoted_cleanup_fns.insert(fname.clone());
                let captured = collect_cleanup_captured_locals(func);
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
                            // Count RcRefCell args and check whether any sibling arg is a Closure/Effect
                            // (which could re-entrantly borrow the same RcRefCell during the call).
                            let mut rc_arg_indices: Vec<usize> = Vec::new();
                            let mut has_callable_arg = false;
                            for (ai, arg) in args.iter().enumerate() {
                                if let Some(loc) = operand_exact_local(arg)
                                    && caller_st.get(loc.0).copied() == Some(LocalStorage::RcRefCell)
                                {
                                    rc_arg_indices.push(ai);
                                }
                                if matches!(
                                    operand_mir_type(arg, func),
                                    MirType::Function { .. } | MirType::Effect { .. } | MirType::Unknown
                                ) {
                                    has_callable_arg = true;
                                }
                            }
                            if rc_arg_indices.len() >= 2
                                || (!rc_arg_indices.is_empty() && has_callable_arg)
                            {
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
            can_fail_map.clear();
            needs_ctx_map.clear();
            needs_alloc_map.clear();
            needs_arena_map.clear();
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

            // Disqualify any direct-defer candidate that turned out to be fallible and re-run 2b/2d/3
            let disqualified: Vec<String> = direct_defer_candidates
                .iter()
                .filter(|fname| can_fail_map.get(*fname).copied().unwrap_or(true))
                .cloned()
                .collect();
            if disqualified.is_empty() {
                break;
            }
            for fname in disqualified {
                direct_defer_candidates.remove(&fname);
            }
        }

        let mut fn_metas = HashMap::new();
        for func in &module.functions {
            let name = func.signature.name.clone();
            let rust_name = rust_fn_name(&name);
            let param_modes = param_modes_map.remove(&name).unwrap_or_default();
            let local_storage = local_storage_map.remove(&name).unwrap_or_default();
            let cleanup_captured = cleanup_captured_map.remove(&name).unwrap_or_default();
            let is_nontail_fn =
                nontail_body_metas.contains_key(&name) || nontail_arm_metas.contains_key(&name);
            let can_fail = can_fail_map[&name] || is_nontail_fn;
            let keep_inlined_defers_direct =
                !can_fail && direct_defer_candidates.contains(&name);
            fn_metas.insert(
                name.clone(),
                FnMeta {
                    rust_name,
                    can_fail,
                    needs_ctx: needs_ctx_map[&name] || is_nontail_fn,
                    needs_alloc: needs_alloc_map[&name],
                    needs_arena: needs_arena_map[&name],
                    param_modes,
                    local_storage,
                    cleanup_captured,
                    keep_inlined_defers_direct,
                    return_type: func.signature.return_type.clone(),
                },
            );
        }

        let mut schema_decoded_structs: HashSet<String> = HashSet::new();
        for func in &module.functions {
            for block in &func.blocks {
                if let Some(Terminator::Call { callee, .. }) = &block.terminator {
                    if let Some(sn) = parse_mir_struct_decode_callee(callee)
                        .or_else(|| callee.strip_prefix("__lust_mir_struct_decode_binary:"))
                    {
                        schema_decoded_structs.insert(sn.to_string());
                    }
                }
            }
        }
        loop {
            let mut changed = false;
            let cur: Vec<String> = schema_decoded_structs.iter().cloned().collect();
            for sn in cur {
                if let Some(meta) = structs.get(&sn) {
                    for sf in &meta.schema_fields {
                        if let Some(nested) = &sf.nested_struct_name
                            && schema_decoded_structs.insert(nested.clone())
                        {
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        Self {
            module,
            opts,
            interop,
            ast_meta,
            structs,
            struct_type_ids,
            schema_decoded_structs,
            enums,
            fn_metas,
            nontail_body_metas,
            nontail_arm_metas,
            module_uses_arena,
            closures_need_lt,
        }
    }

    /// Resolves a MIR callee name to a direct Zero-FFI external Rust crate function call
    /// (`extern crate <ext>; <ext>::<fn>(...)`) when `--link-crate` is active (`NG-W13-1C`).
    fn resolve_extern_crate_call(&self, callee: &str) -> Option<ResolvedExternCrateCall> {
        if self.interop.linked_crates.is_empty() {
            return None;
        }
        let linked_set: HashSet<String> = self
            .interop
            .linked_crates
            .iter()
            .map(|c| sanitize_ident(c))
            .collect();

        let extern_sig = self.ast_meta.extern_fns.iter().find(|e| {
            e.name == callee
                || e.name.rsplit('.').next() == Some(callee)
                || callee.rsplit('.').next() == Some(e.name.as_str())
        });

        let (crate_ident, fn_path) = if let Some((prefix, rest)) =
            callee.split_once("::").or_else(|| callee.split_once('.'))
            && linked_set.contains(&sanitize_ident(prefix))
        {
            (sanitize_ident(prefix), rest.replace('.', "::"))
        } else if let Some(sig) = extern_sig
            && let Some((prefix, rest)) =
                sig.name.split_once("::").or_else(|| sig.name.split_once('.'))
            && linked_set.contains(&sanitize_ident(prefix))
        {
            (sanitize_ident(prefix), rest.replace('.', "::"))
        } else if let Some(sig) = extern_sig
            && linked_set.contains(&sanitize_ident(&sig.abi))
        {
            let simple = sig
                .name
                .rsplit(['.', ':'])
                .next()
                .unwrap_or(&sig.name)
                .to_string();
            (sanitize_ident(&sig.abi), simple)
        } else if let Some(sig) = extern_sig
            && self.interop.linked_crates.len() == 1
        {
            let simple = sig
                .name
                .rsplit(['.', ':'])
                .next()
                .unwrap_or(&sig.name)
                .to_string();
            (sanitize_ident(&self.interop.linked_crates[0]), simple)
        } else {
            return None;
        };

        let param_types: Vec<MirType> = extern_sig
            .map(|sig| {
                sig.params
                    .iter()
                    .map(|(_, ty)| ast_type_to_mir_for_service(ty, self))
                    .collect()
            })
            .unwrap_or_default();
        let return_type = extern_sig
            .and_then(|sig| sig.return_type.as_ref())
            .map(|ty| ast_type_to_mir_for_service(ty, self))
            .unwrap_or(MirType::Unit);

        Some(ResolvedExternCrateCall {
            rust_path: format!("{crate_ident}::{fn_path}"),
            param_types,
            return_type,
        })
    }

    fn nontail_op_arg_rust_type(&self, mode: &NonTailOpArgMode) -> String {
        match mode {
            NonTailOpArgMode::Unit => "()".to_string(),
            NonTailOpArgMode::Single(ty) => self.rust_type(ty, "'static"),
            NonTailOpArgMode::TaggedVec => {
                "(&'static str, alloc::vec::Vec<lust_aot_rt::AotValue>)".to_string()
            }
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
                    .is_some_and(|s| (s.is_cyclic && !s.is_mutated_cyclic) || s.needs_arena_lt),
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

    fn struct_is_arena_cyclic(&self, name: &str) -> bool {
        self.structs
            .get(name)
            .is_some_and(|s| s.is_cyclic && !s.is_mutated_cyclic)
    }

    fn struct_is_rc_cyclic(&self, name: &str) -> bool {
        self.structs.get(name).is_some_and(|s| s.is_mutated_cyclic)
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
                MirType::Struct(name) if self.struct_is_arena_cyclic(name) => {
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

    // Detect cyclic structs whose fields are mutated anywhere in the module (`AOT-BUDGET-SOUND`).
    // Pure immutable recursive trees (e.g. `tree.lust`) remain arena-allocated (`LustChunkArena`),
    // while mutated cyclic structs (e.g. `Cell` in `NG-08`) use `Rc<RefCell<...>>` to preserve
    // reference aliasing identity and immediately reclaim memory when cycles are broken.
    let mut mutated_structs: HashSet<String> = HashSet::new();
    for func in &module.functions {
        for block in &func.blocks {
            for stmt in &block.statements {
                if let Statement::Assign(Place::Field(base, _), _, _) = stmt
                    && let MirType::Struct(sn) = place_type_raw(base, func)
                {
                    mutated_structs.insert(sn);
                }
            }
        }
    }
    let mut is_mutated_cyclic: HashMap<String, bool> = HashMap::new();
    for sname in raw_fields.keys() {
        let mc = is_cyclic[sname] && mutated_structs.contains(sname);
        is_mutated_cyclic.insert(sname.clone(), mc);
    }
    let mut is_arena_cyclic: HashMap<String, bool> = HashMap::new();
    for sname in raw_fields.keys() {
        is_arena_cyclic.insert(sname.clone(), is_cyclic[sname] && !is_mutated_cyclic[sname]);
    }

    // Fixed-point for needs_arena_lt, is_copy, derives_default
    let mut needs_arena_lt: HashMap<String, bool> = HashMap::new();
    let mut is_copy: HashMap<String, bool> = HashMap::new();
    let mut derives_default: HashMap<String, bool> = HashMap::new();

    for (sname, fields) in &raw_fields {
        let has_arr = fields.iter().any(|(_, t)| type_contains_array(t));
        needs_arena_lt.insert(sname.clone(), is_arena_cyclic[sname] || has_arr);
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
                if field_needs_lt(fty, &needs_arena_lt, &is_arena_cyclic) && !cur_lt {
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
                is_mutated_cyclic: is_mutated_cyclic[&name],
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
                .is_some_and(|m| (m.is_cyclic && !m.is_mutated_cyclic) || m.needs_arena_lt),
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
                .is_some_and(|m| (m.is_cyclic && !m.is_mutated_cyclic) || m.needs_arena_lt),
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

/// Returns `true` if `src` can be mutated along any CFG path starting strictly after
/// `(def_bb, def_sidx)` (`dst = Copy/Move(src)`) and reaching a use of `dst` without
/// re-executing `(def_bb, def_sidx)` (`NG-W15-1`).
fn is_src_mutated_while_dst_live(
    func: &MirFunction,
    src: LocalId,
    dst: LocalId,
    def_bb: BlockId,
    def_sidx: usize,
    fn_metas: &HashMap<String, FnMeta>,
) -> bool {
    let stmt_uses_local = |stmt: &Statement, target: LocalId| -> bool {
        if let Statement::Assign(place, rval, _) = stmt {
            let mut tmp = HashSet::new();
            place.collect_uses(&mut tmp);
            rval.collect_uses(&mut tmp);
            tmp.contains(&target.0) || place.root_local() == target
        } else {
            false
        }
    };
    let stmt_mutates_src = |stmt: &Statement| -> bool {
        if let Statement::Assign(place, _, _) = stmt {
            place.root_local() == src
        } else {
            false
        }
    };
    let term_uses_local = |term: &Terminator, target: LocalId| -> bool {
        let mut tmp = HashSet::new();
        match term {
            Terminator::Cond { cond, .. } => cond.collect_uses(&mut tmp),
            Terminator::Switch { discr, .. } => discr.collect_uses(&mut tmp),
            Terminator::Call {
                args, destination, ..
            } => {
                if destination.root_local() == target {
                    return true;
                }
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
        tmp.contains(&target.0)
    };
    let term_mutates_src = |term: &Terminator| -> bool {
        if let Terminator::Call {
            callee,
            args,
            destination,
            ..
        } = term
        {
            if destination.root_local() == src {
                return true;
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
            ) && args.first().and_then(operand_root_local) == Some(src)
            {
                return true;
            }
            if let Some(meta) = fn_metas.get(callee) {
                for (ai, arg) in args.iter().enumerate() {
                    if operand_root_local(arg) == Some(src)
                        && meta.param_modes.get(ai).is_some_and(|m| {
                            matches!(
                                m,
                                ParamMode::MutVec | ParamMode::MutRefStruct | ParamMode::RcRefCell
                            )
                        })
                    {
                        return true;
                    }
                }
            }
        }
        false
    };

    // Fast path: if `src` is never mutated in any statement or call, it cannot mutate while `dst` is live.
    if !local_is_mutated_in_fn(func, src)
        && !func.blocks.iter().any(|b| {
            !b.is_cleanup
                && b.terminator
                    .as_ref()
                    .is_some_and(term_mutates_src)
        })
    {
        return false;
    }

    let mut seen_mut = false;
    let def_block = &func.blocks[def_bb.0];
    for stmt in def_block.statements.iter().skip(def_sidx + 1) {
        if seen_mut && stmt_uses_local(stmt, dst) {
            return true;
        }
        if stmt_mutates_src(stmt) {
            seen_mut = true;
        }
    }
    let mut visited: HashSet<(BlockId, bool)> = HashSet::new();
    let mut worklist: VecDeque<(BlockId, bool)> = VecDeque::new();
    if let Some(term) = &def_block.terminator {
        if seen_mut && term_uses_local(term, dst) {
            return true;
        }
        if term_mutates_src(term) {
            seen_mut = true;
        }
        for succ in term.successors() {
            let sb = BlockId(succ);
            if sb.0 < func.blocks.len() && !func.blocks[sb.0].is_cleanup && visited.insert((sb, seen_mut)) {
                worklist.push_back((sb, seen_mut));
            }
        }
    }

    while let Some((bb, mut cur_mut)) = worklist.pop_front() {
        let block = &func.blocks[bb.0];
        let mut redefined = false;
        for (sidx, stmt) in block.statements.iter().enumerate() {
            if bb == def_bb && sidx == def_sidx {
                redefined = true;
                break;
            }
            if cur_mut && stmt_uses_local(stmt, dst) {
                return true;
            }
            if stmt_mutates_src(stmt) {
                cur_mut = true;
            }
        }
        if redefined {
            continue;
        }
        if let Some(term) = &block.terminator {
            if cur_mut && term_uses_local(term, dst) {
                return true;
            }
            if term_mutates_src(term) {
                cur_mut = true;
            }
            for succ in term.successors() {
                let sb = BlockId(succ);
                if sb.0 < func.blocks.len()
                    && !func.blocks[sb.0].is_cleanup
                    && visited.insert((sb, cur_mut))
                {
                    worklist.push_back((sb, cur_mut));
                }
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
                let is_rc_cyclic = structs.get(sname).is_some_and(|s| s.is_mutated_cyclic);
                if is_rc_cyclic {
                    ParamMode::ByValue
                } else if is_ro {
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
        {
            if meta.fields.iter().any(|(_, t)| type_contains_array(t)) {
                req.needs_alloc = true;
            }
            if meta.is_mutated_cyclic {
                req.can_fail = true;
                if opts.max_memory_bytes.is_some() {
                    req.needs_alloc = true;
                }
            }
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
                                    .any(|l| matches!(&l.ty, MirType::Struct(sn) if structs.get(sn).is_some_and(|s| s.is_cyclic && !s.is_mutated_cyclic)))
                                {
                                    req.needs_arena = true;
                                }
                            }
                        }
                        Rvalue::Cast(op, MirType::Option(inner)) => {
                            if let MirType::Struct(sname) = inner.as_ref()
                                && structs
                                    .get(sname)
                                    .is_some_and(|s| s.is_cyclic && !s.is_mutated_cyclic)
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
                        || callee
                            .strip_suffix(".decode")
                            .is_some_and(|sn| structs.contains_key(sn))
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
                            "map.get" | "table.get" | "map.get_unwrap" | "map.get_unwrap_or"
                            | "map.has" | "map.len" | "map.is_empty" | "map.clear" => {
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
                            "tostring" | "string.sub" | "string.upper" | "string.lower"
                            | "string.trim" | "string.concat" => {
                                req.allocates_string = true;
                                if opts.max_memory_bytes.is_some() {
                                    req.needs_alloc = true;
                                    req.can_fail = true;
                                }
                            }
                            "string.byte" | "string.split" | "string.gsub" | "string.find" => {
                                req.needs_alloc = true;
                                req.allocates_string = true;
                                req.can_fail = true;
                            }
                            "__mir_unknown_unwrap_int"
                            | "__mir_unknown_unwrap_float"
                            | "__mir_unknown_unwrap_string"
                            | "__mir_unknown_unwrap_bool"
                            | "__mir_unknown_get_field" => {
                                req.can_fail = true;
                            }
                            "__mir_unknown_call_method" => {
                                req.needs_ctx = true;
                                req.needs_alloc = true;
                                req.can_fail = true;
                                if structs
                                    .values()
                                    .any(|s| s.is_cyclic && !s.is_mutated_cyclic)
                                {
                                    req.needs_arena = true;
                                }
                            }
                            "Queue" | "sync.Queue" | "@std/sync.Queue" => {
                                req.needs_ctx = true;
                                req.can_fail = true;
                            }
                            "__lust_run_sync" => {
                                if args.len() == 1
                                    && let Some(body_fn) =
                                        trace_direct_closure_fn_name_in_mir(func, &args[0])
                                {
                                    // Pure direct synchronous thunk: inherit needs_ctx/can_fail
                                    // strictly from `body_fn` via call-graph propagation (`NG-W15-1`).
                                    req.user_callees.push(body_fn);
                                } else {
                                    req.needs_ctx = true;
                                    req.can_fail = true;
                                    if let Some(arg0) = args.first()
                                        && let Some(body_fn) =
                                            trace_closure_fn_name_in_mir(func, arg0)
                                    {
                                        req.user_callees.push(body_fn);
                                    }
                                }
                            }
                            "__lust_handle" | "__lust_nontail_handle" | "__lust_perform"
                            | "perform" | "__lust_resume" | "resume" | "__lust_nontail_resume"
                            | "__lust_yield_op" | "__lust_yield_completed" | "__lust_provide"
                            | "__lust_scoped" | "__lust_scope_finalizer"
                            | "__lust_scope_spawn" => {
                                req.needs_ctx = true;
                                req.can_fail = true;
                                // Propagate body thunk requirements directly into enclosing function
                                if matches!(
                                    callee.as_str(),
                                    "__lust_handle" | "__lust_nontail_handle" | "__lust_scoped"
                                ) && let Some(arg0) = args.first()
                                    && let Some(body_fn) = trace_closure_fn_name_in_mir(func, arg0)
                                {
                                    req.user_callees.push(body_fn);
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

const INCLUDE_ALLOW_ATTR: &str = "#[allow(unused_variables, unused_mut, unused_assignments, dead_code, non_camel_case_types, non_snake_case, unreachable_code, clippy::all)]\n";

fn emit_file_header(out: &mut String, opts: &RustEmitterOptions, interop: &RustInteropOptions) {
    out.push_str("// Generated by Lust Next-Gen Rust AOT Compiler (NG-W12-4A + NG-W12-4B + NG-W12-4C + NG-W13-1C + NG-W13-2)\n");
    if !interop.emit_as_include_module {
        if opts.no_std {
            out.push_str("#![no_std]\n");
        }
        out.push_str("#![allow(unused_variables, unused_mut, unused_assignments, dead_code, non_camel_case_types, non_snake_case, unreachable_code, clippy::all)]\n\n");
    }
    out.push_str("extern crate alloc;\n");
    out.push_str("extern crate lust_aot_rt;\n");
    if opts.no_std && !interop.emit_as_include_module {
        // On `wasm32-unknown-unknown` `cdylib` builds, link `std` privately to supply the
        // target global allocator and panic handler without introducing any custom allocator glue.
        // Bare-metal MCU targets (`target_os = "none"`) and host `#![no_std]` `rlib` crates never
        // link `std`.
        out.push_str("#[cfg(all(target_arch = \"wasm32\", target_os = \"unknown\"))]\n");
        out.push_str("extern crate std as __lust_wasm_runtime;\n");
    }
    let mut emitted_crates: HashSet<String> = HashSet::new();
    for ext_crate in &interop.linked_crates {
        let ident = sanitize_ident(ext_crate);
        if !ident.is_empty() && emitted_crates.insert(ident.clone()) {
            let _ = writeln!(out, "extern crate {ident};");
        }
    }
    if opts.no_std && !interop.emit_as_include_module {
        out.push_str("\n// `#![no_std]` Embedded / MCU Preamble with `StaticSlabAllocator<65536>` (`NG-W13-2`)\n");
        out.push_str("pub const LUST_STATIC_SLAB_BYTES: usize = 65536;\n");
        out.push_str("pub type LustStaticSlab = lust_aot_rt::StaticSlabAllocator<65536>;\n\n");
        out.push_str("#[inline]\n");
        out.push_str("pub const fn lust_static_slab_new() -> lust_aot_rt::StaticSlabAllocator<65536> {\n");
        out.push_str("    lust_aot_rt::StaticSlabAllocator::<65536>::new()\n");
        out.push_str("}\n\n");
        out.push_str("#[cfg(all(not(test), target_os = \"none\"))]\n");
        out.push_str("#[panic_handler]\n");
        out.push_str("fn __lust_nostd_panic(_info: &core::panic::PanicInfo<'_>) -> ! {\n");
        out.push_str("    loop {}\n");
        out.push_str("}\n");
    }
    out.push('\n');
}

fn emit_struct_definitions(out: &mut String, ctx: &ModuleEmitContext<'_>) {
    if ctx.structs.is_empty() {
        return;
    }
    out.push_str("// Struct & Schema Definitions (Tiers 2-3 + NG-W12-4C #[repr(C)] Layout Assertions & Decoders)\n");
    for meta in ctx.structs.values() {
        let sname = rust_struct_name(&meta.name);
        if ctx.interop.emit_as_include_module {
            out.push_str(INCLUDE_ALLOW_ATTR);
        }
        if meta.is_mutated_cyclic {
            let lt_decl = if meta.needs_arena_lt { "<'a>" } else { "" };
            let data_name = format!("{sname}_Data");
            let has_budget = ctx.opts.max_memory_bytes.is_some();
            out.push_str("#[derive(Debug)]\n");
            let _ = writeln!(out, "pub struct {data_name}{lt_decl} {{");
            for (i, sf) in meta.schema_fields.iter().enumerate() {
                let rust_ty = ctx.rust_schema_field_type(sf, "'a");
                let _ = writeln!(out, "    pub field_{i}: {rust_ty}, // {}", sf.name);
            }
            if has_budget {
                out.push_str("    pub __charged: usize,\n");
                out.push_str("    pub __budget: lust_aot_rt::LustBudgetAlloc,\n");
            }
            out.push_str("}\n");
            if has_budget {
                let _ = writeln!(out, "impl{lt_decl} Drop for {data_name}{lt_decl} {{");
                out.push_str("    #[inline]\n");
                out.push_str("    fn drop(&mut self) {\n");
                out.push_str("        if self.__charged > 0 {\n");
                out.push_str("            self.__budget.refund(self.__charged);\n");
                out.push_str("        }\n");
                out.push_str("    }\n");
                out.push_str("}\n");
            }
            out.push_str("#[derive(Clone, Debug)]\n");
            let _ = writeln!(
                out,
                "pub struct {sname}{lt_decl}(pub alloc::rc::Rc<core::cell::RefCell<{data_name}{lt_decl}>>);"
            );
            let _ = writeln!(out, "impl{lt_decl} Default for {sname}{lt_decl} {{");
            out.push_str("    #[inline]\n");
            out.push_str("    fn default() -> Self {\n");
            let mut def_inits: Vec<String> = Vec::with_capacity(meta.schema_fields.len() + 2);
            for (i, sf) in meta.schema_fields.iter().enumerate() {
                def_inits.push(format!(
                    "field_{i}: {}",
                    ctx.rust_schema_field_default_expr(sf)
                ));
            }
            if has_budget {
                def_inits.push("__charged: 0".to_string());
                def_inits.push("__budget: lust_aot_rt::LustBudgetAlloc::unbounded()".to_string());
            }
            let _ = writeln!(
                out,
                "        Self(alloc::rc::Rc::new(core::cell::RefCell::new({data_name} {{ {} }})))",
                def_inits.join(", ")
            );
            out.push_str("    }\n");
            out.push_str("}\n");
            let _ = writeln!(out, "impl{lt_decl} PartialEq for {sname}{lt_decl} {{");
            out.push_str("    #[inline]\n");
            out.push_str("    fn eq(&self, other: &Self) -> bool {\n");
            out.push_str("        alloc::rc::Rc::ptr_eq(&self.0, &other.0)\n");
            out.push_str("    }\n");
            out.push_str("}\n\n");
            continue;
        }
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
            let needs_64bit_cfg = !layout.is_c_flat
                || layout
                    .fields
                    .iter()
                    .any(|f| matches!(f.tag, CanonicalTypeTag::Decimal128));
            let cfg_prefix = if needs_64bit_cfg {
                "#[cfg(target_pointer_width = \"64\")]\n"
            } else {
                ""
            };
            for (i, f) in layout.fields.iter().enumerate() {
                let _ = writeln!(
                    out,
                    "{cfg_prefix}const _: () = assert!(core::mem::offset_of!({sname}, field_{i}) == {});",
                    f.byte_offset
                );
            }
            let _ = writeln!(
                out,
                "{cfg_prefix}const _: () = assert!(core::mem::size_of::<{sname}>() == {});",
                layout.size_bytes
            );
            let _ = writeln!(
                out,
                "{cfg_prefix}const _: () = assert!(core::mem::align_of::<{sname}>() == {});\n",
                layout.align_bytes
            );
        }

        if meta.has_schema_decoders
            && (ctx.opts.is_library
                || ctx.interop.export_public_api
                || ctx.schema_decoded_structs.contains(&meta.name))
        {
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
        if ctx.interop.emit_as_include_module {
            out.push_str(INCLUDE_ALLOW_ATTR);
        }
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

// ---------------------------------------------------------------------------
// Direct AST Extraction & Nominal Services / Tail-Resumptive Analysis (NG-W13-1A)
// ---------------------------------------------------------------------------

/// Classification of a single `handle .. with` handler arm body extracted directly from AST.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandlerArmTailKind {
    /// Every `__lust_resume(...)` call in this arm is in tail position (single call per path).
    TailResumptive,
    /// Arm never calls `__lust_resume(...)` (aborts / returns directly).
    Abortive,
    /// Arm calls `__lust_resume(...)` in non-tail position or more than once on a path.
    NonTailOrMultiShot,
}

#[derive(Debug, Clone)]
struct HandleAstSite {
    #[allow(dead_code)]
    arm_kinds: Vec<(String, HandlerArmTailKind)>,
    all_tail_or_abortive: bool,
}

#[derive(Debug, Clone)]
struct AstExternFnMeta {
    abi: String,
    name: String,
    params: Vec<(Option<String>, crate::ast::Type)>,
    return_type: Option<crate::ast::Type>,
}

#[derive(Debug, Clone)]
struct ResolvedExternCrateCall {
    rust_path: String,
    param_types: Vec<MirType>,
    return_type: MirType,
}

#[derive(Debug, Default, Clone)]
struct AstEffectsMetadata {
    services: Vec<crate::ast::ServiceDef>,
    traits: BTreeMap<String, crate::ast::TraitDef>,
    trait_impls: Vec<crate::ast::ImplBlock>,
    inherent_impls: BTreeMap<String, Vec<crate::ast::FunctionDef>>,
    fn_effects: BTreeMap<String, crate::ast::ReturnSpec>,
    handle_sites: Vec<HandleAstSite>,
    extern_fns: Vec<AstExternFnMeta>,
}

impl AstEffectsMetadata {
    fn extract(items: &[crate::ast::Item]) -> Self {
        let mut meta = Self::default();
        meta.collect_items(items);
        meta
    }

    fn collect_items(&mut self, items: &[crate::ast::Item]) {
        for item in items {
            match &item.kind {
                crate::ast::ItemKind::Extern { abi, items: ext_items } => {
                    for ext_item in ext_items {
                        if let crate::ast::ExternItem::Function {
                            name,
                            params,
                            return_type,
                            ..
                        } = ext_item
                        {
                            self.extern_fns.push(AstExternFnMeta {
                                abi: abi.clone(),
                                name: name.clone(),
                                params: params
                                    .iter()
                                    .map(|p| (p.name.clone(), p.ty.clone()))
                                    .collect(),
                                return_type: return_type.clone(),
                            });
                        }
                    }
                }
                crate::ast::ItemKind::Service(s) => {
                    self.services.push(s.clone());
                }
                crate::ast::ItemKind::Trait(t) => {
                    self.traits.insert(t.name.clone(), t.clone());
                }
                crate::ast::ItemKind::Impl(imp) => {
                    for m in &imp.methods {
                        self.scan_stmts_for_handles(&m.body);
                        if let Some(eff) = &m.effects {
                            let target = ast_type_target_name(&imp.target_type);
                            self.fn_effects
                                .insert(format!("{target}:{}", m.name), eff.clone());
                        }
                    }
                    if imp.trait_name.is_some() {
                        self.trait_impls.push(imp.clone());
                    } else {
                        let target = ast_type_target_name(&imp.target_type);
                        self.inherent_impls
                            .entry(target)
                            .or_default()
                            .extend(imp.methods.clone());
                    }
                }
                crate::ast::ItemKind::Function(f) => {
                    if let Some(eff) = &f.effects {
                        self.fn_effects.insert(f.name.clone(), eff.clone());
                    }
                    self.scan_stmts_for_handles(&f.body);
                }
                crate::ast::ItemKind::Script(stmts) => {
                    self.scan_stmts_for_handles(stmts);
                }
                crate::ast::ItemKind::Module { items: sub, .. } => {
                    self.collect_items(sub);
                }
                _ => {}
            }
        }
    }

    fn scan_stmts_for_handles(&mut self, stmts: &[crate::ast::Stmt]) {
        for stmt in stmts {
            match &stmt.kind {
                crate::ast::StmtKind::Local {
                    initializer: Some(exprs),
                    ..
                }
                | crate::ast::StmtKind::Return(exprs) => {
                    for e in exprs {
                        self.scan_expr_for_handles(e);
                    }
                }
                crate::ast::StmtKind::Assign { targets, values } => {
                    for e in targets.iter().chain(values.iter()) {
                        self.scan_expr_for_handles(e);
                    }
                }
                crate::ast::StmtKind::CompoundAssign { target, value, .. } => {
                    self.scan_expr_for_handles(target);
                    self.scan_expr_for_handles(value);
                }
                crate::ast::StmtKind::Expr(e) => self.scan_expr_for_handles(e),
                crate::ast::StmtKind::If {
                    condition,
                    then_block,
                    elseif_branches,
                    else_block,
                } => {
                    self.scan_expr_for_handles(condition);
                    self.scan_stmts_for_handles(then_block);
                    for (ec, eb) in elseif_branches {
                        self.scan_expr_for_handles(ec);
                        self.scan_stmts_for_handles(eb);
                    }
                    if let Some(eb) = else_block {
                        self.scan_stmts_for_handles(eb);
                    }
                }
                crate::ast::StmtKind::While { condition, body } => {
                    self.scan_expr_for_handles(condition);
                    self.scan_stmts_for_handles(body);
                }
                crate::ast::StmtKind::ForNumeric {
                    start,
                    end,
                    step,
                    body,
                    ..
                } => {
                    self.scan_expr_for_handles(start);
                    self.scan_expr_for_handles(end);
                    if let Some(st) = step {
                        self.scan_expr_for_handles(st);
                    }
                    self.scan_stmts_for_handles(body);
                }
                crate::ast::StmtKind::ForIn { iterator, body, .. } => {
                    self.scan_expr_for_handles(iterator);
                    self.scan_stmts_for_handles(body);
                }
                crate::ast::StmtKind::Block(b) | crate::ast::StmtKind::Scoped(b) => {
                    self.scan_stmts_for_handles(b);
                }
                crate::ast::StmtKind::Defer(inner) | crate::ast::StmtKind::ErrDefer(inner) => {
                    self.scan_stmts_for_handles(core::slice::from_ref(inner.as_ref()));
                }
                _ => {}
            }
        }
    }

    fn scan_expr_for_handles(&mut self, expr: &crate::ast::Expr) {
        match &expr.kind {
            crate::ast::ExprKind::Call { callee, args, .. } => {
                if matches!(&callee.kind, crate::ast::ExprKind::Identifier(n) if n == "__lust_handle")
                    && args.len() == 2
                {
                    let mut arm_kinds = Vec::new();
                    if let crate::ast::ExprKind::Array(tuples) = &args[1].kind {
                        for tup in tuples {
                            if let crate::ast::ExprKind::Tuple(elems) = &tup.kind
                                && elems.len() == 2
                            {
                                let eff_name = match &elems[0].kind {
                                    crate::ast::ExprKind::Literal(crate::ast::Literal::String(s)) => {
                                        s.clone()
                                    }
                                    _ => String::new(),
                                };
                                let kind = match &elems[1].kind {
                                    crate::ast::ExprKind::Lambda { body, .. } => {
                                        classify_handler_arm_expr(body)
                                    }
                                    _ => HandlerArmTailKind::NonTailOrMultiShot,
                                };
                                arm_kinds.push((eff_name, kind));
                            }
                        }
                    }
                    let all_tail_or_abortive = arm_kinds.iter().all(|(_, k)| {
                        matches!(
                            k,
                            HandlerArmTailKind::TailResumptive | HandlerArmTailKind::Abortive
                        )
                    });
                    self.handle_sites.push(HandleAstSite {
                        arm_kinds,
                        all_tail_or_abortive,
                    });
                }
                self.scan_expr_for_handles(callee);
                for a in args {
                    self.scan_expr_for_handles(a);
                }
            }
            crate::ast::ExprKind::Binary { left, right, .. }
            | crate::ast::ExprKind::Range {
                start: left,
                end: right,
                ..
            } => {
                self.scan_expr_for_handles(left);
                self.scan_expr_for_handles(right);
            }
            crate::ast::ExprKind::Unary { operand: inner, .. }
            | crate::ast::ExprKind::FieldAccess { object: inner, .. }
            | crate::ast::ExprKind::Paren(inner)
            | crate::ast::ExprKind::Cast { expr: inner, .. }
            | crate::ast::ExprKind::TypeCheck { expr: inner, .. }
            | crate::ast::ExprKind::IsPattern { expr: inner, .. }
            | crate::ast::ExprKind::Lambda { body: inner, .. } => {
                self.scan_expr_for_handles(inner);
            }
            crate::ast::ExprKind::Index { object, index } => {
                self.scan_expr_for_handles(object);
                self.scan_expr_for_handles(index);
            }
            crate::ast::ExprKind::MethodCall { receiver, args, .. } => {
                self.scan_expr_for_handles(receiver);
                for a in args {
                    self.scan_expr_for_handles(a);
                }
            }
            crate::ast::ExprKind::Array(elems)
            | crate::ast::ExprKind::Tuple(elems)
            | crate::ast::ExprKind::EnumConstructor { args: elems, .. }
            | crate::ast::ExprKind::Return(elems) => {
                for e in elems {
                    self.scan_expr_for_handles(e);
                }
            }
            crate::ast::ExprKind::Map(pairs) => {
                for (k, v) in pairs {
                    self.scan_expr_for_handles(k);
                    self.scan_expr_for_handles(v);
                }
            }
            crate::ast::ExprKind::StructLiteral { fields, .. } => {
                for f in fields {
                    self.scan_expr_for_handles(&f.value);
                }
            }
            crate::ast::ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.scan_expr_for_handles(condition);
                self.scan_expr_for_handles(then_branch);
                if let Some(eb) = else_branch {
                    self.scan_expr_for_handles(eb);
                }
            }
            crate::ast::ExprKind::Block(stmts) | crate::ast::ExprKind::EffectBlock(stmts) => {
                self.scan_stmts_for_handles(stmts);
            }
            _ => {}
        }
    }
}

/// Classifies whether an effect handler arm body uses `__lust_resume` only in tail position,
/// never uses `__lust_resume` (abortive), or uses it in non-tail / multi-shot position.
pub fn classify_handler_arm_expr(expr: &crate::ast::Expr) -> HandlerArmTailKind {
    match &expr.kind {
        crate::ast::ExprKind::Paren(inner) => classify_handler_arm_expr(inner),
        crate::ast::ExprKind::Call { callee, args, .. }
            if matches!(&callee.kind, crate::ast::ExprKind::Identifier(n) if n == "__lust_resume" || n == "resume") =>
        {
            if args.iter().any(expr_contains_resume) {
                HandlerArmTailKind::NonTailOrMultiShot
            } else {
                HandlerArmTailKind::TailResumptive
            }
        }
        crate::ast::ExprKind::Block(stmts) => classify_handler_arm_stmts(stmts),
        crate::ast::ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            if expr_contains_resume(condition) {
                return HandlerArmTailKind::NonTailOrMultiShot;
            }
            let tk = classify_handler_arm_expr(then_branch);
            let ek = else_branch
                .as_deref()
                .map(classify_handler_arm_expr)
                .unwrap_or(HandlerArmTailKind::Abortive);
            combine_branch_tail_kinds(tk, ek)
        }
        crate::ast::ExprKind::Return(vals) => {
            if vals.len() == 1 {
                classify_handler_arm_expr(&vals[0])
            } else if vals.iter().any(expr_contains_resume) {
                HandlerArmTailKind::NonTailOrMultiShot
            } else {
                HandlerArmTailKind::Abortive
            }
        }
        _ => {
            if expr_contains_resume(expr) {
                HandlerArmTailKind::NonTailOrMultiShot
            } else {
                HandlerArmTailKind::Abortive
            }
        }
    }
}

fn classify_handler_arm_stmts(stmts: &[crate::ast::Stmt]) -> HandlerArmTailKind {
    if stmts.is_empty() {
        return HandlerArmTailKind::Abortive;
    }
    let last_idx = stmts.len() - 1;
    let mut saw_tail_resume = false;
    for (idx, stmt) in stmts.iter().enumerate() {
        let is_last = idx == last_idx;
        match &stmt.kind {
            crate::ast::StmtKind::Return(vals) => {
                let k = if vals.len() == 1 {
                    classify_handler_arm_expr(&vals[0])
                } else if vals.iter().any(expr_contains_resume) {
                    HandlerArmTailKind::NonTailOrMultiShot
                } else {
                    HandlerArmTailKind::Abortive
                };
                match k {
                    HandlerArmTailKind::NonTailOrMultiShot => {
                        return HandlerArmTailKind::NonTailOrMultiShot;
                    }
                    HandlerArmTailKind::TailResumptive => saw_tail_resume = true,
                    HandlerArmTailKind::Abortive => {}
                }
            }
            crate::ast::StmtKind::Expr(e) if is_last => {
                let k = classify_handler_arm_expr(e);
                match k {
                    HandlerArmTailKind::NonTailOrMultiShot => {
                        return HandlerArmTailKind::NonTailOrMultiShot;
                    }
                    HandlerArmTailKind::TailResumptive => saw_tail_resume = true,
                    HandlerArmTailKind::Abortive => {}
                }
            }
            crate::ast::StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } if is_last => {
                if expr_contains_resume(condition) {
                    return HandlerArmTailKind::NonTailOrMultiShot;
                }
                let mut acc = classify_handler_arm_stmts(then_block);
                for (ec, eb) in elseif_branches {
                    if expr_contains_resume(ec) {
                        return HandlerArmTailKind::NonTailOrMultiShot;
                    }
                    acc = combine_branch_tail_kinds(acc, classify_handler_arm_stmts(eb));
                }
                if let Some(eb) = else_block {
                    acc = combine_branch_tail_kinds(acc, classify_handler_arm_stmts(eb));
                }
                match acc {
                    HandlerArmTailKind::NonTailOrMultiShot => {
                        return HandlerArmTailKind::NonTailOrMultiShot;
                    }
                    HandlerArmTailKind::TailResumptive => saw_tail_resume = true,
                    HandlerArmTailKind::Abortive => {}
                }
            }
            crate::ast::StmtKind::Block(inner) if is_last => {
                let k = classify_handler_arm_stmts(inner);
                match k {
                    HandlerArmTailKind::NonTailOrMultiShot => {
                        return HandlerArmTailKind::NonTailOrMultiShot;
                    }
                    HandlerArmTailKind::TailResumptive => saw_tail_resume = true,
                    HandlerArmTailKind::Abortive => {}
                }
            }
            _ => {
                if stmt_contains_resume(stmt) {
                    return HandlerArmTailKind::NonTailOrMultiShot;
                }
            }
        }
    }
    if saw_tail_resume {
        HandlerArmTailKind::TailResumptive
    } else {
        HandlerArmTailKind::Abortive
    }
}

fn combine_branch_tail_kinds(a: HandlerArmTailKind, b: HandlerArmTailKind) -> HandlerArmTailKind {
    if matches!(a, HandlerArmTailKind::NonTailOrMultiShot)
        || matches!(b, HandlerArmTailKind::NonTailOrMultiShot)
    {
        HandlerArmTailKind::NonTailOrMultiShot
    } else if matches!(a, HandlerArmTailKind::TailResumptive)
        || matches!(b, HandlerArmTailKind::TailResumptive)
    {
        HandlerArmTailKind::TailResumptive
    } else {
        HandlerArmTailKind::Abortive
    }
}

fn expr_contains_resume(expr: &crate::ast::Expr) -> bool {
    match &expr.kind {
        crate::ast::ExprKind::Call { callee, args, .. } => {
            if matches!(&callee.kind, crate::ast::ExprKind::Identifier(n) if n == "__lust_resume" || n == "resume")
            {
                return true;
            }
            expr_contains_resume(callee) || args.iter().any(expr_contains_resume)
        }
        crate::ast::ExprKind::Binary { left, right, .. }
        | crate::ast::ExprKind::Range {
            start: left,
            end: right,
            ..
        } => expr_contains_resume(left) || expr_contains_resume(right),
        crate::ast::ExprKind::Unary { operand: inner, .. }
        | crate::ast::ExprKind::FieldAccess { object: inner, .. }
        | crate::ast::ExprKind::Paren(inner)
        | crate::ast::ExprKind::Cast { expr: inner, .. }
        | crate::ast::ExprKind::TypeCheck { expr: inner, .. }
        | crate::ast::ExprKind::IsPattern { expr: inner, .. }
        | crate::ast::ExprKind::Lambda { body: inner, .. } => expr_contains_resume(inner),
        crate::ast::ExprKind::Index { object, index } => {
            expr_contains_resume(object) || expr_contains_resume(index)
        }
        crate::ast::ExprKind::MethodCall { receiver, args, .. } => {
            expr_contains_resume(receiver) || args.iter().any(expr_contains_resume)
        }
        crate::ast::ExprKind::Array(es)
        | crate::ast::ExprKind::Tuple(es)
        | crate::ast::ExprKind::EnumConstructor { args: es, .. }
        | crate::ast::ExprKind::Return(es) => es.iter().any(expr_contains_resume),
        crate::ast::ExprKind::Map(pairs) => pairs
            .iter()
            .any(|(k, v)| expr_contains_resume(k) || expr_contains_resume(v)),
        crate::ast::ExprKind::StructLiteral { fields, .. } => {
            fields.iter().any(|f| expr_contains_resume(&f.value))
        }
        crate::ast::ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            expr_contains_resume(condition)
                || expr_contains_resume(then_branch)
                || else_branch.as_deref().is_some_and(expr_contains_resume)
        }
        crate::ast::ExprKind::Block(stmts) | crate::ast::ExprKind::EffectBlock(stmts) => {
            stmts.iter().any(stmt_contains_resume)
        }
        _ => false,
    }
}

fn stmt_contains_resume(stmt: &crate::ast::Stmt) -> bool {
    match &stmt.kind {
        crate::ast::StmtKind::Local {
            initializer: Some(es),
            ..
        }
        | crate::ast::StmtKind::Return(es) => es.iter().any(expr_contains_resume),
        crate::ast::StmtKind::Assign { targets, values } => {
            targets.iter().any(expr_contains_resume) || values.iter().any(expr_contains_resume)
        }
        crate::ast::StmtKind::CompoundAssign { target, value, .. } => {
            expr_contains_resume(target) || expr_contains_resume(value)
        }
        crate::ast::StmtKind::Expr(e) => expr_contains_resume(e),
        crate::ast::StmtKind::If {
            condition,
            then_block,
            elseif_branches,
            else_block,
        } => {
            expr_contains_resume(condition)
                || then_block.iter().any(stmt_contains_resume)
                || elseif_branches
                    .iter()
                    .any(|(ec, eb)| expr_contains_resume(ec) || eb.iter().any(stmt_contains_resume))
                || else_block
                    .as_ref()
                    .is_some_and(|eb| eb.iter().any(stmt_contains_resume))
        }
        crate::ast::StmtKind::While { condition, body } => {
            expr_contains_resume(condition) || body.iter().any(stmt_contains_resume)
        }
        crate::ast::StmtKind::ForNumeric {
            start,
            end,
            step,
            body,
            ..
        } => {
            expr_contains_resume(start)
                || expr_contains_resume(end)
                || step.as_ref().is_some_and(expr_contains_resume)
                || body.iter().any(stmt_contains_resume)
        }
        crate::ast::StmtKind::ForIn { iterator, body, .. } => {
            expr_contains_resume(iterator) || body.iter().any(stmt_contains_resume)
        }
        crate::ast::StmtKind::Block(b) | crate::ast::StmtKind::Scoped(b) => {
            b.iter().any(stmt_contains_resume)
        }
        crate::ast::StmtKind::Defer(inner) | crate::ast::StmtKind::ErrDefer(inner) => {
            stmt_contains_resume(inner)
        }
        _ => false,
    }
}

fn ast_type_target_name(ty: &crate::ast::Type) -> String {
    match &ty.kind {
        crate::ast::TypeKind::Named(name)
        | crate::ast::TypeKind::GenericInstance { name, .. }
        | crate::ast::TypeKind::Trait(name) => name.clone(),
        _ => format!("{ty}"),
    }
}

fn ast_type_to_mir_raw(
    ty: &crate::ast::Type,
    struct_names: &HashSet<String>,
    enum_names: &HashSet<String>,
) -> MirType {
    match &ty.kind {
        crate::ast::TypeKind::Int => MirType::Int,
        crate::ast::TypeKind::Float => MirType::Float,
        crate::ast::TypeKind::Bool => MirType::Bool,
        crate::ast::TypeKind::String => MirType::String,
        crate::ast::TypeKind::Unit => MirType::Unit,
        crate::ast::TypeKind::Array(inner) => {
            MirType::Array(Box::new(ast_type_to_mir_raw(inner, struct_names, enum_names)))
        }
        crate::ast::TypeKind::Named(name) => {
            if struct_names.contains(name) {
                MirType::Struct(name.clone())
            } else if enum_names.contains(name) {
                MirType::Enum(name.clone())
            } else {
                MirType::Unknown
            }
        }
        crate::ast::TypeKind::Effect { success, .. } => {
            ast_type_to_mir_raw(success, struct_names, enum_names)
        }
        crate::ast::TypeKind::Option(inner) => {
            MirType::Option(Box::new(ast_type_to_mir_raw(inner, struct_names, enum_names)))
        }
        crate::ast::TypeKind::Tuple(elems) => MirType::Tuple(
            elems
                .iter()
                .map(|e| ast_type_to_mir_raw(e, struct_names, enum_names))
                .collect(),
        ),
        _ => MirType::Unknown,
    }
}

fn build_extern_return_type_map(
    module: &MirModule,
    ast_meta: &AstEffectsMetadata,
) -> HashMap<String, MirType> {
    let mut map = HashMap::new();
    if ast_meta.extern_fns.is_empty() {
        return map;
    }
    let struct_names: HashSet<String> = module
        .struct_layouts
        .iter()
        .map(|s| s.type_name.clone())
        .collect();
    let enum_names: HashSet<String> = module.enum_defs.iter().map(|e| e.name.clone()).collect();
    for ext in &ast_meta.extern_fns {
        let ret_ty = ext
            .return_type
            .as_ref()
            .map(|ty| ast_type_to_mir_raw(ty, &struct_names, &enum_names))
            .unwrap_or(MirType::Unit);
        map.insert(ext.name.clone(), ret_ty.clone());
        if let Some((_, short)) = ext.name.rsplit_once('.') {
            map.entry(short.to_string()).or_insert(ret_ty);
        }
    }
    map
}

fn apply_extern_callee_return_types(
    module: &mut MirModule,
    extern_ret_types: &HashMap<String, MirType>,
) {
    for func in &mut module.functions {
        let mut updated_temps: HashMap<LocalId, MirType> = HashMap::new();
        for block in &func.blocks {
            if let Some(Terminator::Call {
                callee,
                destination: Place::Local(dst),
                ..
            }) = &block.terminator
                && let Some(ret_ty) = extern_ret_types
                    .get(callee)
                    .or_else(|| callee.rsplit_once('.').and_then(|(_, s)| extern_ret_types.get(s)))
                && *ret_ty != MirType::Unknown
            {
                func.locals[dst.0].ty = ret_ty.clone();
                updated_temps.insert(*dst, ret_ty.clone());
            }
        }
        if !updated_temps.is_empty() {
            for block in &func.blocks {
                for stmt in &block.statements {
                    if let Statement::Assign(
                        Place::Local(dst),
                        Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))),
                        _,
                    ) = stmt
                        && let Some(ret_ty) = updated_temps.get(src)
                        && matches!(func.locals[dst.0].ty, MirType::Unknown | MirType::Int)
                        && *ret_ty != MirType::Int
                    {
                        func.locals[dst.0].ty = ret_ty.clone();
                    }
                }
            }
        }
    }
}

fn ast_type_to_mir_for_service(ty: &crate::ast::Type, ctx: &ModuleEmitContext<'_>) -> MirType {
    match &ty.kind {
        crate::ast::TypeKind::Int => MirType::Int,
        crate::ast::TypeKind::Float => MirType::Float,
        crate::ast::TypeKind::Bool => MirType::Bool,
        crate::ast::TypeKind::String => MirType::String,
        crate::ast::TypeKind::Unit => MirType::Unit,
        crate::ast::TypeKind::Array(inner) => {
            MirType::Array(Box::new(ast_type_to_mir_for_service(inner, ctx)))
        }
        crate::ast::TypeKind::Named(name) => {
            if ctx.structs.contains_key(name) {
                MirType::Struct(name.clone())
            } else if ctx.enums.contains_key(name) {
                MirType::Enum(name.clone())
            } else {
                MirType::Unknown
            }
        }
        crate::ast::TypeKind::Effect { success, .. } => ast_type_to_mir_for_service(success, ctx),
        crate::ast::TypeKind::Option(inner) => {
            MirType::Option(Box::new(ast_type_to_mir_for_service(inner, ctx)))
        }
        crate::ast::TypeKind::Tuple(elems) => MirType::Tuple(
            elems
                .iter()
                .map(|e| ast_type_to_mir_for_service(e, ctx))
                .collect(),
        ),
        _ => MirType::Unknown,
    }
}

#[derive(Debug, Clone)]
struct ServiceMethodSig {
    name: String,
    params: Vec<(String, MirType)>,
    return_type: MirType,
}

fn extract_trait_method_sigs(
    tdef: &crate::ast::TraitDef,
    ctx: &ModuleEmitContext<'_>,
) -> Vec<ServiceMethodSig> {
    let mut out = Vec::with_capacity(tdef.methods.len());
    for m in &tdef.methods {
        let params: Vec<(String, MirType)> = m
            .params
            .iter()
            .filter(|p| !(p.is_self || p.name == "self"))
            .map(|p| (p.name.clone(), ast_type_to_mir_for_service(&p.ty, ctx)))
            .collect();
        let ret_spec = m.return_spec();
        let ret_ty = ret_spec
            .return_type
            .as_ref()
            .map(|t| ast_type_to_mir_for_service(t, ctx))
            .unwrap_or(MirType::Unit);
        out.push(ServiceMethodSig {
            name: m.name.clone(),
            params,
            return_type: ret_ty,
        });
    }
    out
}

fn extract_fn_method_sigs(
    fns: &[crate::ast::FunctionDef],
    ctx: &ModuleEmitContext<'_>,
) -> Vec<ServiceMethodSig> {
    let mut out = Vec::with_capacity(fns.len());
    for f in fns {
        let params: Vec<(String, MirType)> = f
            .params
            .iter()
            .filter(|p| !(p.is_self || p.name == "self"))
            .map(|p| (p.name.clone(), ast_type_to_mir_for_service(&p.ty, ctx)))
            .collect();
        let ret_spec = f.return_spec();
        let ret_ty = ret_spec
            .return_type
            .as_ref()
            .map(|t| ast_type_to_mir_for_service(t, ctx))
            .unwrap_or(MirType::Unit);
        out.push(ServiceMethodSig {
            name: f.name.clone(),
            params,
            return_type: ret_ty,
        });
    }
    out
}

fn emit_service_method_delegation(
    out: &mut String,
    target_struct_name: &str,
    sig: &ServiceMethodSig,
    ctx: &ModuleEmitContext<'_>,
) {
    let mir_method_name = format!("{target_struct_name}:{}", sig.name);
    let Some(fmeta) = ctx.fn_metas.get(&mir_method_name) else {
        let def_ret = ctx.rust_default_expr(&sig.return_type);
        let _ = writeln!(out, "        let _ = __ctx;");
        let _ = writeln!(out, "        Ok({def_ret})");
        return;
    };
    let mir_fn = ctx
        .module
        .functions
        .iter()
        .find(|f| f.signature.name == mir_method_name);

    let mut args = Vec::new();
    if fmeta.needs_ctx {
        args.push("__ctx".to_string());
    }
    if fmeta.needs_alloc {
        if fmeta.needs_ctx {
            out.push_str("        let __svc_alloc = __ctx.budget_alloc.share();\n");
            args.push("&__svc_alloc".to_string());
        } else {
            args.push("&__ctx.budget_alloc".to_string());
        }
    }
    if fmeta.needs_arena {
        out.push_str("        let __svc_arena_alloc = __ctx.budget_alloc.share();\n");
        out.push_str(
            "        let __svc_arena = lust_aot_rt::LustChunkArena::new(&__svc_arena_alloc);\n",
        );
        args.push("&__svc_arena".to_string());
    }

    // Check if the MIR method takes `self` as its first parameter
    let takes_self = mir_fn.is_some_and(|mf| {
        mf.params.first().is_some_and(|pid| {
            mf.locals
                .get(pid.0)
                .is_some_and(|d| d.name.as_deref() == Some("self") || matches!(&d.ty, MirType::Struct(sn) if sn == target_struct_name))
        })
    });
    let mut param_offset = 0usize;
    if takes_self {
        let mode = fmeta.param_modes.first().copied().unwrap_or(ParamMode::ByValue);
        let is_copy = ctx
            .structs
            .get(target_struct_name)
            .is_some_and(|s| s.is_copy);
        let self_expr = match mode {
            ParamMode::ByValue => {
                if is_copy {
                    "*self".to_string()
                } else {
                    "self.clone()".to_string()
                }
            }
            ParamMode::RefStruct => "&*self".to_string(),
            ParamMode::MutRefStruct => "self".to_string(),
            ParamMode::RcRefCell => {
                if is_copy {
                    "alloc::rc::Rc::new(core::cell::RefCell::new(*self))".to_string()
                } else {
                    "alloc::rc::Rc::new(core::cell::RefCell::new(self.clone()))".to_string()
                }
            }
            _ => "&*self".to_string(),
        };
        args.push(self_expr);
        param_offset = 1;
    }

    for (idx, (pname, _)) in sig.params.iter().enumerate() {
        let p_ident = format!("__p_{}", sanitize_ident(pname));
        let mode = fmeta
            .param_modes
            .get(param_offset + idx)
            .copied()
            .unwrap_or(ParamMode::ByValue);
        let a_expr = match mode {
            ParamMode::RefStr => format!("{p_ident}.as_str()"),
            ParamMode::RefStruct => format!("&{p_ident}"),
            ParamMode::MutRefStruct => format!("&mut {p_ident}"),
            ParamMode::RefSlice => format!("&{p_ident}[..]"),
            ParamMode::MutVec => format!("&mut {p_ident}"),
            ParamMode::RcCell => format!("alloc::rc::Rc::new(core::cell::Cell::new({p_ident}))"),
            ParamMode::RcRefCell => {
                format!("alloc::rc::Rc::new(core::cell::RefCell::new({p_ident}))")
            }
            ParamMode::ByValue => p_ident,
        };
        args.push(a_expr);
    }

    let raw_call = format!("{}({})", fmeta.rust_name, args.join(", "));
    if fmeta.can_fail {
        let _ = writeln!(out, "        {raw_call}");
    } else {
        let _ = writeln!(
            out,
            "        {raw_call}.map_err(lust_aot_rt::AotError::from)"
        );
    }
}

fn emit_trait_and_service_definitions(out: &mut String, ctx: &ModuleEmitContext<'_>) {
    // 1. Emit Rust traits for AST `ItemKind::Trait` declarations
    if !ctx.ast_meta.traits.is_empty() {
        out.push_str("// Trait Definitions Extracted from AST (NG-W13-1A)\n");
        for (tname, tdef) in &ctx.ast_meta.traits {
            let rust_tname = format!("LustTrait_{}", sanitize_ident(tname));
            let sigs = extract_trait_method_sigs(tdef, ctx);
            if sigs.iter().any(|s| {
                ctx.type_needs_lifetime(&s.return_type)
                    || s.params.iter().any(|(_, pt)| ctx.type_needs_lifetime(pt))
            }) {
                continue;
            }
            let _ = writeln!(out, "pub trait {rust_tname} {{");
            for sig in &sigs {
                let m_ident = sanitize_ident(&sig.name);
                let mut p_strs = vec![
                    "&mut self".to_string(),
                    "__ctx: &mut lust_aot_rt::AotCtx".to_string(),
                ];
                for (pname, pty) in &sig.params {
                    p_strs.push(format!(
                        "__p_{}: {}",
                        sanitize_ident(pname),
                        ctx.rust_type(pty, "'static")
                    ));
                }
                let ret_r = ctx.rust_type(&sig.return_type, "'static");
                let _ = writeln!(
                    out,
                    "    fn {m_ident}({}) -> Result<{ret_r}, lust_aot_rt::AotError>;",
                    p_strs.join(", ")
                );
            }
            out.push_str("}\n\n");

            // Emit `impl LustTrait_<T> for LustStruct_<Target>` for matching impl blocks
            for imp in &ctx.ast_meta.trait_impls {
                if imp.trait_name.as_deref() != Some(tname.as_str()) {
                    continue;
                }
                let target_name = ast_type_target_name(&imp.target_type);
                let Some(smeta) = ctx.structs.get(&target_name) else {
                    continue;
                };
                if smeta.needs_arena_lt {
                    continue;
                }
                let rust_sname = rust_struct_name(&target_name);
                let _ = writeln!(out, "impl {rust_tname} for {rust_sname} {{");
                for sig in &sigs {
                    let m_ident = sanitize_ident(&sig.name);
                    let mut p_strs = vec![
                        "&mut self".to_string(),
                        "__ctx: &mut lust_aot_rt::AotCtx".to_string(),
                    ];
                    for (pname, pty) in &sig.params {
                        p_strs.push(format!(
                            "mut __p_{}: {}",
                            sanitize_ident(pname),
                            ctx.rust_type(pty, "'static")
                        ));
                    }
                    let ret_r = ctx.rust_type(&sig.return_type, "'static");
                    let _ = writeln!(
                        out,
                        "    fn {m_ident}({}) -> Result<{ret_r}, lust_aot_rt::AotError> {{",
                        p_strs.join(", ")
                    );
                    emit_service_method_delegation(out, &target_name, sig, ctx);
                    out.push_str("    }\n");
                }
                out.push_str("}\n\n");
            }
        }
    }

    // 2. Emit Nominal Service traits & safe ZST default providers (`NG-W13-1A`, `F-11`)
    if ctx.ast_meta.services.is_empty() {
        return;
    }
    out.push_str("// Nominal Service Traits & Safe Default::default() ZST Providers (NG-W13-1A, F-11)\n");
    for svc in &ctx.ast_meta.services {
        let s_ident = sanitize_ident(&svc.name);
        let trait_name = format!("LustService_{s_ident}");
        let provider_name = format!("LustDefaultProvider_{s_ident}");

        let target_name_opt = svc.target_type.as_ref().map(ast_type_target_name);
        let mut method_sigs: Vec<ServiceMethodSig> = Vec::new();
        let mut impl_struct_targets: Vec<String> = Vec::new();

        if let Some(ref target_name) = target_name_opt {
            if let Some(tdef) = ctx.ast_meta.traits.get(target_name) {
                method_sigs = extract_trait_method_sigs(tdef, ctx);
                for imp in &ctx.ast_meta.trait_impls {
                    if imp.trait_name.as_deref() == Some(target_name.as_str()) {
                        let st = ast_type_target_name(&imp.target_type);
                        if ctx.structs.contains_key(&st) && !impl_struct_targets.contains(&st) {
                            impl_struct_targets.push(st);
                        }
                    }
                }
            } else {
                if let Some(fns) = ctx.ast_meta.inherent_impls.get(target_name) {
                    method_sigs = extract_fn_method_sigs(fns, ctx);
                }
                if ctx.structs.contains_key(target_name) {
                    impl_struct_targets.push(target_name.clone());
                }
            }
        } else if let Some(tdef) = ctx.ast_meta.traits.get(&svc.name) {
            method_sigs = extract_trait_method_sigs(tdef, ctx);
        } else if let Some(fns) = ctx.ast_meta.inherent_impls.get(&svc.name) {
            method_sigs = extract_fn_method_sigs(fns, ctx);
        }

        // Keep service method signatures lifetime-free so `LustService_<S>` is object/ZST-friendly
        method_sigs.retain(|s| {
            !ctx.type_needs_lifetime(&s.return_type)
                && !s.params.iter().any(|(_, pt)| ctx.type_needs_lifetime(pt))
        });

        let _ = writeln!(out, "pub trait {trait_name} {{");
        let _ = writeln!(
            out,
            "    const SERVICE_NAME: &'static str = {:?};",
            svc.name
        );
        out.push_str("    #[inline]\n");
        out.push_str(
            "    fn service_id(&self, __ctx: &mut lust_aot_rt::AotCtx) -> Result<&'static str, lust_aot_rt::AotError> {\n",
        );
        out.push_str("        let _ = __ctx;\n");
        out.push_str("        Ok(Self::SERVICE_NAME)\n");
        out.push_str("    }\n");
        for sig in &method_sigs {
            let m_ident = sanitize_ident(&sig.name);
            let mut p_strs = vec![
                "&mut self".to_string(),
                "__ctx: &mut lust_aot_rt::AotCtx".to_string(),
            ];
            for (pname, pty) in &sig.params {
                p_strs.push(format!(
                    "__p_{}: {}",
                    sanitize_ident(pname),
                    ctx.rust_type(pty, "'static")
                ));
            }
            let ret_r = ctx.rust_type(&sig.return_type, "'static");
            let _ = writeln!(
                out,
                "    fn {m_ident}({}) -> Result<{ret_r}, lust_aot_rt::AotError>;",
                p_strs.join(", ")
            );
        }
        out.push_str("}\n\n");

        // Safe zero-sized default provider instantiated via `Default::default()` (NO unsafe conjure_zst!)
        out.push_str("#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]\n");
        let _ = writeln!(out, "pub struct {provider_name};");
        let _ = writeln!(
            out,
            "const _: () = assert!(core::mem::size_of::<{provider_name}>() == 0);"
        );
        let _ = writeln!(out, "impl {provider_name} {{");
        out.push_str("    #[inline(always)]\n");
        out.push_str("    pub fn instantiate() -> Self {\n");
        out.push_str("        lust_aot_rt::default_zst_provider::<Self>()\n");
        out.push_str("    }\n");
        out.push_str("}\n\n");

        // Implement `LustService_<S>` for every matching concrete struct target
        for st_name in &impl_struct_targets {
            let Some(smeta) = ctx.structs.get(st_name) else {
                continue;
            };
            if smeta.needs_arena_lt {
                continue;
            }
            let rust_sname = rust_struct_name(st_name);
            let _ = writeln!(out, "impl {trait_name} for {rust_sname} {{");
            for sig in &method_sigs {
                let m_ident = sanitize_ident(&sig.name);
                let mut p_strs = vec![
                    "&mut self".to_string(),
                    "__ctx: &mut lust_aot_rt::AotCtx".to_string(),
                ];
                for (pname, pty) in &sig.params {
                    p_strs.push(format!(
                        "mut __p_{}: {}",
                        sanitize_ident(pname),
                        ctx.rust_type(pty, "'static")
                    ));
                }
                let ret_r = ctx.rust_type(&sig.return_type, "'static");
                let _ = writeln!(
                    out,
                    "    fn {m_ident}({}) -> Result<{ret_r}, lust_aot_rt::AotError> {{",
                    p_strs.join(", ")
                );
                emit_service_method_delegation(out, st_name, sig, ctx);
                out.push_str("    }\n");
            }
            out.push_str("}\n\n");
        }

        // Choose a default-constructible target struct (preferring a 0-field ZST struct first)
        let default_delegate_struct = impl_struct_targets
            .iter()
            .find(|st| {
                ctx.structs.get(*st).is_some_and(|m| {
                    !m.needs_arena_lt && m.derives_default && m.fields.is_empty()
                })
            })
            .or_else(|| {
                impl_struct_targets.iter().find(|st| {
                    ctx.structs
                        .get(*st)
                        .is_some_and(|m| !m.needs_arena_lt && m.derives_default)
                })
            });

        let _ = writeln!(out, "impl {trait_name} for {provider_name} {{");
        for sig in &method_sigs {
            let m_ident = sanitize_ident(&sig.name);
            let mut p_strs = vec![
                "&mut self".to_string(),
                "__ctx: &mut lust_aot_rt::AotCtx".to_string(),
            ];
            let mut fwd_args = vec!["__ctx".to_string()];
            for (pname, pty) in &sig.params {
                let p_ident = format!("__p_{}", sanitize_ident(pname));
                p_strs.push(format!("{p_ident}: {}", ctx.rust_type(pty, "'static")));
                fwd_args.push(p_ident);
            }
            let ret_r = ctx.rust_type(&sig.return_type, "'static");
            let _ = writeln!(
                out,
                "    fn {m_ident}({}) -> Result<{ret_r}, lust_aot_rt::AotError> {{",
                p_strs.join(", ")
            );
            if let Some(target_st) = default_delegate_struct {
                let rust_sname = rust_struct_name(target_st);
                let _ = writeln!(out, "        let mut __target = {rust_sname}::default();");
                let _ = writeln!(
                    out,
                    "        <{rust_sname} as {trait_name}>::{m_ident}(&mut __target, {})",
                    fwd_args.join(", ")
                );
            } else {
                let def_ret = ctx.rust_default_expr(&sig.return_type);
                out.push_str("        let _ = __ctx;\n");
                let _ = writeln!(out, "        Ok({def_ret})");
            }
            out.push_str("    }\n");
        }
        out.push_str("}\n\n");
    }
}

/// Traces an operand directly to a local `Rvalue::Closure { func_name, .. }` ONLY through
/// simple copy/move assignments (never crossing `__lust_provide`). Used for pure synchronous
/// `__lust_run_sync` fast path (`NG-W15-1`, `F-02`).
fn trace_direct_closure_fn_name_in_mir(func: &MirFunction, op: &Operand) -> Option<String> {
    let mut cur = operand_exact_local(op)?;
    let mut visited = HashSet::new();
    while visited.insert(cur) {
        let mut found_next: Option<LocalId> = None;
        for block in &func.blocks {
            if block.is_cleanup {
                continue;
            }
            for stmt in &block.statements {
                if let Statement::Assign(Place::Local(dst), rval, _) = stmt
                    && *dst == cur
                {
                    match rval {
                        Rvalue::Closure { func_name, .. } => return Some(func_name.clone()),
                        Rvalue::Use(
                            Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)),
                        ) => {
                            found_next = Some(*src);
                        }
                        _ => {}
                    }
                }
            }
            if let Some(Terminator::Call {
                callee,
                destination: Place::Local(dst),
                ..
            }) = &block.terminator
                && *dst == cur
                && callee == "__lust_provide"
            {
                return None;
            }
        }
        cur = found_next?;
    }
    None
}

/// Returns `true` when all `func.cleanup_blocks` are `MirCleanupKind::Defer` (no `ErrDefer`),
/// there are no `Raise`/`Unreachable` terminators, and every non-cleanup `Terminator::Return`
/// block ends with the exact LIFO `AstLowering`-inlined `Defer` action suffix (`NG-W15-1`, `F-02`).
fn function_inlined_defers_cover_all_returns(func: &MirFunction) -> bool {
    if func.cleanup_blocks.is_empty() {
        return false;
    }
    if func
        .cleanup_blocks
        .iter()
        .any(|c| c.kind != MirCleanupKind::Defer)
    {
        return false;
    }
    if func.blocks.iter().any(|b| {
        !b.is_cleanup
            && matches!(
                b.terminator,
                Some(Terminator::Raise(_, _) | Terminator::Unreachable(_))
            )
    }) {
        return false;
    }
    let defer_entries: Vec<BlockId> = func
        .cleanup_blocks
        .iter()
        .map(|c| c.block)
        .collect();
    let mut expected: Vec<CleanupActionSig> = Vec::new();
    for &cbb in defer_entries.iter().rev() {
        expected.extend(extract_cleanup_action_sigs(func, cbb));
    }
    if expected.is_empty() {
        return false;
    }

    let mut preds: Vec<Vec<BlockId>> = vec![Vec::new(); func.blocks.len()];
    for blk in &func.blocks {
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

    let ret_blocks: Vec<BlockId> = func
        .blocks
        .iter()
        .filter(|b| !b.is_cleanup && matches!(b.terminator, Some(Terminator::Return(_, _))))
        .map(|b| b.id)
        .collect();
    if ret_blocks.is_empty() {
        return false;
    }

    for ret_bb in ret_blocks {
        let mut chain: Vec<BlockId> = vec![ret_bb];
        let mut cur = ret_bb;
        while preds[cur.0].len() == 1 {
            let p = preds[cur.0][0];
            if matches!(
                func.blocks[p.0].terminator,
                Some(Terminator::Call { .. } | Terminator::Goto(_, _))
            ) {
                chain.push(p);
                cur = p;
            } else {
                break;
            }
        }
        chain.reverse();

        let mut chain_actions: Vec<CleanupActionSig> = Vec::new();
        for &bb in &chain {
            let blk = &func.blocks[bb.0];
            for stmt in &blk.statements {
                if let Some(sig) = stmt_action_sig(stmt, func) {
                    chain_actions.push(sig);
                }
            }
            if let Some(Terminator::Call { callee, args, .. }) = &blk.terminator {
                chain_actions.push(call_action_sig(callee, args, func));
            }
        }
        if chain_actions.len() < expected.len() {
            return false;
        }
        let start_idx = chain_actions.len() - expected.len();
        if !chain_actions[start_idx..]
            .iter()
            .zip(expected.iter())
            .all(|(got, exp)| got == exp)
        {
            return false;
        }
    }
    true
}

/// Emits conditional `#![no_std] + alloc`, zero-`unsafe`, zero-panic runtime helpers
/// for Suite v2 string intrinsics (`string.sub`, `string.byte`, `string.gsub`, `string.split`,
/// `string.find`), `@std/sync.{Queue}`, and dynamic `AotValue` struct field/method dispatch
/// (`NG-W15-1`). Only emits helpers referenced by the module.
fn emit_suite_v2_runtime_helpers(
    out: &mut String,
    module: &MirModule,
    ctx: &ModuleEmitContext<'_>,
) {
    let mut need_range = false;
    let mut need_sub = false;
    let mut need_byte = false;
    let mut need_split = false;
    let mut need_pattern = false;
    let mut need_gsub = false;
    let mut need_find = false;
    let mut need_queue = false;
    let mut need_unknown_field = false;
    let mut need_unknown_method = false;

    for func in &module.functions {
        for block in &func.blocks {
            if let Some(Terminator::Call { callee, .. }) = &block.terminator {
                match callee.as_str() {
                    "string.sub" => {
                        need_range = true;
                        need_sub = true;
                    }
                    "string.byte" => {
                        need_range = true;
                        need_byte = true;
                    }
                    "string.split" => {
                        need_split = true;
                    }
                    "string.gsub" => {
                        need_pattern = true;
                        need_gsub = true;
                    }
                    "string.find" => {
                        need_range = true;
                        need_pattern = true;
                        need_find = true;
                    }
                    "Queue" | "sync.Queue" | "@std/sync.Queue" => {
                        need_queue = true;
                    }
                    "__mir_unknown_get_field" => {
                        need_unknown_field = true;
                    }
                    "__mir_unknown_call_method" => {
                        need_unknown_method = true;
                    }
                    _ => {}
                }
            }
        }
    }

    let allow_attr = if ctx.interop.emit_as_include_module {
        INCLUDE_ALLOW_ATTR
    } else {
        ""
    };

    if need_range {
        let _ = writeln!(out, "{allow_attr}#[inline]");
        out.push_str(
            r#"fn __lust_aot_normalize_range(start: i64, end: Option<i64>, len: usize) -> (usize, usize) {
    let len_i = len as i64;
    let mut s = if start < 0 { len_i + start + 1 } else { start };
    let mut e = match end {
        Some(v) => v,
        None => len_i,
    };
    if e < 0 {
        e = len_i + e + 1;
    }
    if s < 1 {
        s = 1;
    }
    if e < 0 {
        e = 0;
    }
    if e > len_i {
        e = len_i;
    }
    if s > e {
        return (len, len);
    }
    (s.saturating_sub(1) as usize, e.max(0) as usize)
}

"#,
        );
    }

    if need_sub {
        let _ = writeln!(out, "{allow_attr}#[inline(always)]");
        out.push_str(
            r#"fn __lust_aot_string_sub(s: &str, start: i64, end: Option<i64>) -> alloc::string::String {
    let bytes = s.as_bytes();
    let (lo, hi) = __lust_aot_normalize_range(start, end, bytes.len());
    if let Some(valid) = s.get(lo..hi) {
        alloc::string::String::from(valid)
    } else {
        alloc::string::String::from_utf8_lossy(&bytes[lo..hi]).into_owned()
    }
}

"#,
        );
        let _ = writeln!(out, "{allow_attr}#[inline(always)]");
        out.push_str(
            r#"fn __lust_aot_string_sub_cow<'s>(s: &'s str, start: i64, end: Option<i64>) -> alloc::borrow::Cow<'s, str> {
    let bytes = s.as_bytes();
    let (lo, hi) = __lust_aot_normalize_range(start, end, bytes.len());
    if let Some(valid) = s.get(lo..hi) {
        alloc::borrow::Cow::Borrowed(valid)
    } else {
        alloc::string::String::from_utf8_lossy(&bytes[lo..hi])
    }
}

"#,
        );
    }

    if need_byte {
        let _ = writeln!(out, "{allow_attr}#[inline]");
        out.push_str(
            r#"fn __lust_aot_string_byte_vals<'a>(
    s: &str,
    start: i64,
    end: Option<i64>,
    alloc: &'a lust_aot_rt::LustBudgetAlloc,
) -> alloc::vec::Vec<lust_aot_rt::AotValue, &'a lust_aot_rt::LustBudgetAlloc> {
    let bytes = s.as_bytes();
    let (lo, hi) = __lust_aot_normalize_range(start, end, bytes.len());
    let mut out = alloc::vec::Vec::new_in(alloc);
    if lo == hi {
        out.push(lust_aot_rt::AotValue::Nil);
        return out;
    }
    for &b in &bytes[lo..hi] {
        out.push(lust_aot_rt::AotValue::Int(b as i64));
    }
    out
}

"#,
        );
        let _ = writeln!(out, "{allow_attr}#[inline]");
        out.push_str(
            r#"fn __lust_aot_string_byte_ints<'a>(
    s: &str,
    start: i64,
    end: Option<i64>,
    alloc: &'a lust_aot_rt::LustBudgetAlloc,
) -> alloc::vec::Vec<i64, &'a lust_aot_rt::LustBudgetAlloc> {
    let bytes = s.as_bytes();
    let (lo, hi) = __lust_aot_normalize_range(start, end, bytes.len());
    let mut out = alloc::vec::Vec::new_in(alloc);
    for &b in &bytes[lo..hi] {
        out.push(b as i64);
    }
    out
}

"#,
        );
    }

    if need_split {
        let _ = writeln!(out, "{allow_attr}#[inline]");
        out.push_str(
            r#"fn __lust_aot_string_split<'a>(
    s: &str,
    sep: &str,
    alloc: &'a lust_aot_rt::LustBudgetAlloc,
) -> alloc::vec::Vec<alloc::string::String, &'a lust_aot_rt::LustBudgetAlloc> {
    let mut out = alloc::vec::Vec::with_capacity_in(4, alloc);
    if sep.is_empty() {
        for ch in s.chars() {
            let mut cs = alloc::string::String::new();
            cs.push(ch);
            out.push(cs);
        }
    } else {
        for part in s.split(sep) {
            out.push(alloc::string::String::from(part));
        }
    }
    out
}

"#,
        );
        let _ = writeln!(out, "{allow_attr}");
        out.push_str(
            r#"struct __LustSplitSpans<'a> {
    inline_len: usize,
    inline_spans: [(u32, u32); 4],
    overflow: Option<alloc::vec::Vec<(u32, u32), &'a lust_aot_rt::LustBudgetAlloc>>,
}

impl<'a> __LustSplitSpans<'a> {
    #[inline(always)]
    fn new_in(_alloc: &'a lust_aot_rt::LustBudgetAlloc) -> Self {
        Self {
            inline_len: 0,
            inline_spans: [(0, 0); 4],
            overflow: None,
        }
    }

    #[inline(always)]
    fn len(&self) -> usize {
        match &self.overflow {
            Some(v) => v.len(),
            None => self.inline_len,
        }
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> Option<(usize, usize)> {
        match &self.overflow {
            Some(v) => v.get(idx).map(|&(a, b)| (a as usize, b as usize)),
            None => {
                if idx < self.inline_len {
                    let (a, b) = self.inline_spans[idx];
                    Some((a as usize, b as usize))
                } else {
                    None
                }
            }
        }
    }

    #[cold]
    #[inline(never)]
    fn push_cold(&mut self, lo: u32, hi: u32, alloc: &'a lust_aot_rt::LustBudgetAlloc) {
        let pair = (lo, hi);
        if let Some(v) = &mut self.overflow {
            v.push(pair);
            return;
        }
        let mut v = alloc::vec::Vec::with_capacity_in(8, alloc);
        v.extend_from_slice(&self.inline_spans[..self.inline_len]);
        v.push(pair);
        self.inline_len = 0;
        self.overflow = Some(v);
    }

    #[inline(always)]
    fn push(&mut self, lo: usize, hi: usize, alloc: &'a lust_aot_rt::LustBudgetAlloc) {
        if self.overflow.is_none() && self.inline_len < 4 {
            self.inline_spans[self.inline_len] = (lo as u32, hi as u32);
            self.inline_len += 1;
        } else {
            self.push_cold(lo as u32, hi as u32, alloc);
        }
    }
}

#[inline(always)]
fn __lust_aot_string_split_pair_byte<'s>(
    s: &'s str,
    delim: u8,
) -> (Option<&'s str>, Option<&'s str>) {
    let mut it = s.split(delim as char);
    (it.next(), it.next())
}

#[inline(always)]
fn __lust_aot_string_split_pair<'s>(
    s: &'s str,
    sep: &str,
) -> (Option<&'s str>, Option<&'s str>) {
    if sep.is_empty() {
        let mut chars = s.chars();
        let Some(c0) = chars.next() else {
            return (None, None);
        };
        let n0 = c0.len_utf8();
        let p0 = &s[..n0];
        let p1 = chars.next().map(|c1| &s[n0..n0 + c1.len_utf8()]);
        (Some(p0), p1)
    } else {
        let mut it = s.split(sep);
        (it.next(), it.next())
    }
}

#[inline(always)]
fn __lust_aot_string_split_spans_byte<'a>(
    s: &str,
    delim: u8,
    alloc: &'a lust_aot_rt::LustBudgetAlloc,
) -> __LustSplitSpans<'a> {
    let bytes = s.as_bytes();
    let mut out = __LustSplitSpans::new_in(alloc);
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == delim {
            out.push(start, i, alloc);
            start = i + 1;
        }
        i += 1;
    }
    out.push(start, bytes.len(), alloc);
    out
}

#[inline]
fn __lust_aot_string_split_spans<'a>(
    s: &str,
    sep: &str,
    alloc: &'a lust_aot_rt::LustBudgetAlloc,
) -> __LustSplitSpans<'a> {
    let sep_bytes = sep.as_bytes();
    if sep_bytes.len() == 1 && sep_bytes[0] < 128 {
        return __lust_aot_string_split_spans_byte(s, sep_bytes[0], alloc);
    }
    let mut out = __LustSplitSpans::new_in(alloc);
    if sep_bytes.is_empty() {
        let mut pos = 0usize;
        for ch in s.chars() {
            let next = pos + ch.len_utf8();
            out.push(pos, next, alloc);
            pos = next;
        }
    } else {
        let mut start = 0usize;
        for part in s.split(sep) {
            let end = start + part.len();
            out.push(start, end, alloc);
            start = end + sep.len();
        }
    }
    out
}

"#,
        );
    }

    if need_pattern {
        let _ = writeln!(out, "{allow_attr}");
        out.push_str(
            r#"#[derive(Clone)]
enum __LustPatAtom {
    Any,
    Lit(u8),
    Class(u8),
    Set { negated: bool, items: alloc::vec::Vec<__LustSetItem> },
}

#[derive(Clone, Copy)]
enum __LustSetItem {
    Byte(u8),
    Range(u8, u8),
    Class(u8),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum __LustPatQuant {
    One,
    Star,
    Plus,
    LazyStar,
    Opt,
}

#[derive(Clone)]
enum __LustPatTok {
    Atom(__LustPatAtom, __LustPatQuant),
    CapStart(usize),
    CapEnd(usize),
    EndAnchor,
}

#[inline]
fn __lust_pat_match_class(cls: u8, b: u8) -> bool {
    let lower = cls.to_ascii_lowercase();
    let matched = match lower {
        b'a' => b.is_ascii_alphabetic(),
        b'c' => b.is_ascii_control(),
        b'd' => b.is_ascii_digit(),
        b'l' => b.is_ascii_lowercase(),
        b'u' => b.is_ascii_uppercase(),
        b'w' => b.is_ascii_alphanumeric() || b == b'_',
        b'x' => b.is_ascii_hexdigit(),
        b's' => b.is_ascii_whitespace(),
        b'p' => b.is_ascii_punctuation(),
        b'z' => b == 0,
        _ => b == cls,
    };
    if cls.is_ascii_uppercase() && matches!(lower, b'a' | b'c' | b'd' | b'l' | b'u' | b'w' | b'x' | b's' | b'p' | b'z') {
        !matched
    } else {
        matched
    }
}

#[inline]
fn __lust_pat_match_atom(atom: &__LustPatAtom, b: u8) -> bool {
    match atom {
        __LustPatAtom::Any => true,
        __LustPatAtom::Lit(c) => b == *c,
        __LustPatAtom::Class(c) => __lust_pat_match_class(*c, b),
        __LustPatAtom::Set { negated, items } => {
            let mut hit = false;
            for item in items {
                match *item {
                    __LustSetItem::Byte(c) if b == c => { hit = true; break; }
                    __LustSetItem::Range(lo, hi) if b >= lo && b <= hi => { hit = true; break; }
                    __LustSetItem::Class(c) if __lust_pat_match_class(c, b) => { hit = true; break; }
                    _ => {}
                }
            }
            if *negated { !hit } else { hit }
        }
    }
}

fn __lust_pat_compile(pat: &[u8]) -> Result<(bool, alloc::vec::Vec<__LustPatTok>, usize), lust_aot_rt::AotError> {
    let mut i = 0usize;
    let anchored_start = if pat.first() == Some(&b'^') {
        i = 1;
        true
    } else {
        false
    };
    let mut toks = alloc::vec::Vec::new();
    let mut cap_stack = alloc::vec::Vec::new();
    let mut cap_count = 0usize;
    while i < pat.len() {
        let b = pat[i];
        if b == b'(' {
            let idx = cap_count;
            cap_count += 1;
            cap_stack.push(idx);
            toks.push(__LustPatTok::CapStart(idx));
            i += 1;
            continue;
        }
        if b == b')' {
            if let Some(idx) = cap_stack.pop() {
                toks.push(__LustPatTok::CapEnd(idx));
                i += 1;
                continue;
            }
            return Err(lust_aot_rt::AotError::UserError(alloc::string::String::from("unmatched ')' in pattern")));
        }
        if b == b'$' && i + 1 == pat.len() {
            toks.push(__LustPatTok::EndAnchor);
            i += 1;
            continue;
        }
        let atom = if b == b'.' {
            i += 1;
            __LustPatAtom::Any
        } else if b == b'%' {
            i += 1;
            if i < pat.len() {
                let c = pat[i];
                i += 1;
                let lo = c.to_ascii_lowercase();
                if matches!(lo, b'a' | b'c' | b'd' | b'l' | b'u' | b'w' | b'x' | b's' | b'p' | b'z') {
                    __LustPatAtom::Class(c)
                } else {
                    __LustPatAtom::Lit(c)
                }
            } else {
                __LustPatAtom::Lit(b'%')
            }
        } else if b == b'[' {
            i += 1;
            let negated = if i < pat.len() && pat[i] == b'^' {
                i += 1;
                true
            } else {
                false
            };
            let mut items = alloc::vec::Vec::new();
            let mut first = true;
            while i < pat.len() && (first || pat[i] != b']') {
                first = false;
                if pat[i] == b'%' && i + 1 < pat.len() {
                    let c = pat[i + 1];
                    i += 2;
                    let lo = c.to_ascii_lowercase();
                    if matches!(lo, b'a' | b'c' | b'd' | b'l' | b'u' | b'w' | b'x' | b's' | b'p' | b'z') {
                        items.push(__LustSetItem::Class(c));
                    } else {
                        items.push(__LustSetItem::Byte(c));
                    }
                } else if i + 2 < pat.len() && pat[i + 1] == b'-' && pat[i + 2] != b']' {
                    items.push(__LustSetItem::Range(pat[i], pat[i + 2]));
                    i += 3;
                } else {
                    items.push(__LustSetItem::Byte(pat[i]));
                    i += 1;
                }
            }
            if i < pat.len() && pat[i] == b']' {
                i += 1;
            }
            __LustPatAtom::Set { negated, items }
        } else {
            i += 1;
            __LustPatAtom::Lit(b)
        };
        let quant = if i < pat.len() {
            match pat[i] {
                b'*' => { i += 1; __LustPatQuant::Star }
                b'+' => { i += 1; __LustPatQuant::Plus }
                b'-' => { i += 1; __LustPatQuant::LazyStar }
                b'?' => { i += 1; __LustPatQuant::Opt }
                _ => __LustPatQuant::One,
            }
        } else {
            __LustPatQuant::One
        };
        toks.push(__LustPatTok::Atom(atom, quant));
    }
    if !cap_stack.is_empty() {
        return Err(lust_aot_rt::AotError::UserError(alloc::string::String::from("unclosed '(' in pattern")));
    }
    Ok((anchored_start, toks, cap_count))
}

fn __lust_pat_exec(
    hay: &[u8],
    pos: usize,
    toks: &[__LustPatTok],
    ti: usize,
    caps: &mut [(usize, usize)],
) -> Option<usize> {
    if ti >= toks.len() {
        return Some(pos);
    }
    match &toks[ti] {
        __LustPatTok::CapStart(idx) => {
            let prev = caps[*idx];
            caps[*idx].0 = pos;
            if let Some(end) = __lust_pat_exec(hay, pos, toks, ti + 1, caps) {
                return Some(end);
            }
            caps[*idx] = prev;
            None
        }
        __LustPatTok::CapEnd(idx) => {
            let prev = caps[*idx];
            caps[*idx].1 = pos;
            if let Some(end) = __lust_pat_exec(hay, pos, toks, ti + 1, caps) {
                return Some(end);
            }
            caps[*idx] = prev;
            None
        }
        __LustPatTok::EndAnchor => {
            if pos == hay.len() {
                __lust_pat_exec(hay, pos, toks, ti + 1, caps)
            } else {
                None
            }
        }
        __LustPatTok::Atom(atom, quant) => match quant {
            __LustPatQuant::One => {
                if pos < hay.len() && __lust_pat_match_atom(atom, hay[pos]) {
                    __lust_pat_exec(hay, pos + 1, toks, ti + 1, caps)
                } else {
                    None
                }
            }
            __LustPatQuant::Opt => {
                if pos < hay.len() && __lust_pat_match_atom(atom, hay[pos]) {
                    if let Some(end) = __lust_pat_exec(hay, pos + 1, toks, ti + 1, caps) {
                        return Some(end);
                    }
                }
                __lust_pat_exec(hay, pos, toks, ti + 1, caps)
            }
            __LustPatQuant::Star | __LustPatQuant::Plus => {
                let min_k = if *quant == __LustPatQuant::Plus { 1 } else { 0 };
                let mut max_pos = pos;
                while max_pos < hay.len() && __lust_pat_match_atom(atom, hay[max_pos]) {
                    max_pos += 1;
                }
                if max_pos - pos < min_k {
                    return None;
                }
                let mut cur = max_pos;
                loop {
                    if let Some(end) = __lust_pat_exec(hay, cur, toks, ti + 1, caps) {
                        return Some(end);
                    }
                    if cur == pos + min_k {
                        break;
                    }
                    cur -= 1;
                }
                None
            }
            __LustPatQuant::LazyStar => {
                let mut cur = pos;
                loop {
                    if let Some(end) = __lust_pat_exec(hay, cur, toks, ti + 1, caps) {
                        return Some(end);
                    }
                    if cur >= hay.len() || !__lust_pat_match_atom(atom, hay[cur]) {
                        break;
                    }
                    cur += 1;
                }
                None
            }
        },
    }
}

"#,
        );
    }

    if need_gsub {
        let _ = writeln!(out, "{allow_attr}");
        out.push_str(
            r#"fn __lust_aot_string_gsub(
    src: &str,
    pattern: &str,
    repl: &str,
    limit: Option<i64>,
) -> Result<(alloc::string::String, i64), lust_aot_rt::AotError> {
    let max_repl = match limit {
        Some(n) if n <= 0 => return Ok((alloc::string::String::from(src), 0)),
        Some(n) => n,
        None => i64::MAX,
    };
    let (anchored, toks, cap_count) = __lust_pat_compile(pattern.as_bytes())?;
    let hay = src.as_bytes();
    let repl_bytes = repl.as_bytes();
    let repl_has_pct = repl_bytes.contains(&b'%');
    let mut out = alloc::string::String::with_capacity(src.len());
    let mut caps = alloc::vec![(0usize, 0usize); cap_count];
    let mut pos = 0usize;
    let mut span_start = 0usize;
    let mut count = 0_i64;
    while pos <= hay.len() {
        if count >= max_repl {
            break;
        }
        if let Some(end) = __lust_pat_exec(hay, pos, &toks, 0, &mut caps) {
            if pos > span_start {
                if let Some(seg) = src.get(span_start..pos) {
                    out.push_str(seg);
                } else if let Ok(seg) = core::str::from_utf8(&hay[span_start..pos]) {
                    out.push_str(seg);
                }
            }
            count += 1;
            if !repl_has_pct {
                out.push_str(repl);
            } else {
                let mut ri = 0usize;
                while ri < repl_bytes.len() {
                    if repl_bytes[ri] == b'%' && ri + 1 < repl_bytes.len() {
                        let next = repl_bytes[ri + 1];
                        if next == b'%' {
                            out.push('%');
                        } else if next == b'0' {
                            if let Some(seg) = src.get(pos..end) {
                                out.push_str(seg);
                            } else if let Ok(seg) = core::str::from_utf8(&hay[pos..end]) {
                                out.push_str(seg);
                            }
                        } else if next.is_ascii_digit() {
                            let cidx = (next - b'1') as usize;
                            if let Some(&(cs, ce)) = caps.get(cidx)
                                && cs <= ce
                                && ce <= hay.len()
                            {
                                if let Some(seg) = src.get(cs..ce) {
                                    out.push_str(seg);
                                } else if let Ok(seg) = core::str::from_utf8(&hay[cs..ce]) {
                                    out.push_str(seg);
                                }
                            }
                        } else {
                            out.push(next as char);
                        }
                        ri += 2;
                    } else if let Some(seg) = repl.get(ri..)
                        && let Some(ch) = seg.chars().next()
                    {
                        out.push(ch);
                        ri += ch.len_utf8();
                    } else {
                        ri += 1;
                    }
                }
            }
            if end > pos {
                pos = end;
                span_start = pos;
            } else if pos < hay.len() {
                span_start = pos;
                if let Some(seg) = src.get(pos..)
                    && let Some(ch) = seg.chars().next()
                {
                    pos += ch.len_utf8();
                } else {
                    pos += 1;
                }
            } else {
                span_start = hay.len();
                break;
            }
            if anchored {
                break;
            }
        } else {
            if anchored {
                break;
            }
            if pos < hay.len() {
                if let Some(seg) = src.get(pos..)
                    && let Some(ch) = seg.chars().next()
                {
                    pos += ch.len_utf8();
                } else {
                    pos += 1;
                }
            } else {
                break;
            }
        }
    }
    if span_start < hay.len() {
        if let Some(rest) = src.get(span_start..) {
            out.push_str(rest);
        } else if let Ok(rest) = core::str::from_utf8(&hay[span_start..]) {
            out.push_str(rest);
        }
    }
    Ok((out, count))
}

"#,
        );
    }

    if need_find {
        let _ = writeln!(out, "{allow_attr}");
        out.push_str(
            r#"fn __lust_aot_string_find(
    haystack: &str,
    pattern: &str,
    start: i64,
    plain: bool,
) -> Result<Option<(i64, i64, alloc::vec::Vec<alloc::string::String>)>, lust_aot_rt::AotError> {
    let (offset, _) = __lust_aot_normalize_range(start, None, haystack.len());
    let slice = haystack.get(offset..).unwrap_or("");
    if plain {
        return Ok(match slice.find(pattern) {
            Some(pos) => {
                let begin = offset + pos;
                let end = begin + pattern.len().saturating_sub(1);
                Some((begin as i64 + 1, end as i64 + 1, alloc::vec::Vec::new()))
            }
            None => None,
        });
    }
    let (anchored, toks, cap_count) = __lust_pat_compile(pattern.as_bytes())?;
    let hay = slice.as_bytes();
    let mut caps = alloc::vec![(0usize, 0usize); cap_count];
    let limit_pos = if anchored { 0 } else { hay.len() };
    for p in 0..=limit_pos {
        if let Some(end) = __lust_pat_exec(hay, p, &toks, 0, &mut caps) {
            let mut cap_strs = alloc::vec::Vec::with_capacity(cap_count);
            for &(cs, ce) in &caps {
                if cs <= ce && ce <= hay.len() {
                    let s = match core::str::from_utf8(&hay[cs..ce]) {
                        Ok(valid) => alloc::string::String::from(valid),
                        Err(_) => alloc::string::String::from_utf8_lossy(&hay[cs..ce]).into_owned(),
                    };
                    cap_strs.push(s);
                }
            }
            return Ok(Some((
                (offset + p) as i64 + 1,
                (offset + end.saturating_sub(1)) as i64 + 1,
                cap_strs,
            )));
        }
    }
    Ok(None)
}

"#,
        );
    }

    if need_queue {
        let _ = writeln!(out, "{allow_attr}");
        out.push_str(
            r#"fn __lust_aot_new_queue(
    capacity: i64,
    strategy: &str,
    ctx: &lust_aot_rt::AotCtx,
) -> Result<lust_aot_rt::AotValue, lust_aot_rt::AotError> {
    let cap = if capacity <= 0 { 0usize } else { capacity as usize };
    let strategy_code: u8 = match strategy {
        "dropping" => 1,
        "sliding" => 2,
        _ => 0,
    };
    let buf: alloc::rc::Rc<core::cell::RefCell<alloc::collections::VecDeque<lust_aot_rt::AotValue>>> =
        alloc::rc::Rc::new(core::cell::RefCell::new(alloc::collections::VecDeque::with_capacity(cap)));
    let closed: alloc::rc::Rc<core::cell::Cell<bool>> =
        alloc::rc::Rc::new(core::cell::Cell::new(false));
    let obj = lust_aot_rt::AotValue::map();

    {
        let q = alloc::rc::Rc::clone(&buf);
        let cl = alloc::rc::Rc::clone(&closed);
        let offer_fn = lust_aot_rt::AotValue::closure(
            move |_ctx: &mut lust_aot_rt::AotCtx, args: &[lust_aot_rt::AotValue]| -> Result<lust_aot_rt::AotValue, lust_aot_rt::AotError> {
                if cl.get() {
                    return Ok(lust_aot_rt::AotValue::Bool(false));
                }
                let item = args.first().cloned().unwrap_or(lust_aot_rt::AotValue::Nil);
                let mut guard = lust_aot_rt::ref_borrow_mut(&q)?;
                if cap > 0 && guard.len() >= cap {
                    match strategy_code {
                        1 => return Ok(lust_aot_rt::AotValue::Bool(false)),
                        2 => { let _ = guard.pop_front(); }
                        _ => return Err(lust_aot_rt::AotError::UserError(alloc::string::String::from("queue is full"))),
                    }
                }
                guard.push_back(item);
                Ok(lust_aot_rt::AotValue::Bool(true))
            },
        );
        obj.map_set(lust_aot_rt::AotKey::Str(alloc::string::String::from("offer")), offer_fn.clone(), ctx)?;
        obj.map_set(lust_aot_rt::AotKey::Str(alloc::string::String::from("try_offer")), offer_fn, ctx)?;
    }

    {
        let q = alloc::rc::Rc::clone(&buf);
        let take_fn = lust_aot_rt::AotValue::closure(
            move |_ctx: &mut lust_aot_rt::AotCtx, _args: &[lust_aot_rt::AotValue]| -> Result<lust_aot_rt::AotValue, lust_aot_rt::AotError> {
                let mut guard = lust_aot_rt::ref_borrow_mut(&q)?;
                Ok(guard.pop_front().unwrap_or(lust_aot_rt::AotValue::Nil))
            },
        );
        obj.map_set(lust_aot_rt::AotKey::Str(alloc::string::String::from("take")), take_fn.clone(), ctx)?;
        obj.map_set(lust_aot_rt::AotKey::Str(alloc::string::String::from("try_take")), take_fn, ctx)?;
    }

    {
        let q = alloc::rc::Rc::clone(&buf);
        let len_fn = lust_aot_rt::AotValue::closure(
            move |_ctx: &mut lust_aot_rt::AotCtx, _args: &[lust_aot_rt::AotValue]| -> Result<lust_aot_rt::AotValue, lust_aot_rt::AotError> {
                let guard = lust_aot_rt::ref_borrow(&q)?;
                Ok(lust_aot_rt::AotValue::Int(guard.len() as i64))
            },
        );
        obj.map_set(lust_aot_rt::AotKey::Str(alloc::string::String::from("len")), len_fn, ctx)?;
    }

    {
        let cl = alloc::rc::Rc::clone(&closed);
        let close_fn = lust_aot_rt::AotValue::closure(
            move |_ctx: &mut lust_aot_rt::AotCtx, _args: &[lust_aot_rt::AotValue]| -> Result<lust_aot_rt::AotValue, lust_aot_rt::AotError> {
                cl.set(true);
                Ok(lust_aot_rt::AotValue::Nil)
            },
        );
        obj.map_set(lust_aot_rt::AotKey::Str(alloc::string::String::from("close")), close_fn, ctx)?;
    }

    Ok(obj)
}

"#,
        );
    }

    if need_unknown_field || need_unknown_method {
        let copy_structs: Vec<(&String, &StructMeta)> = ctx
            .structs
            .iter()
            .filter(|(_, m)| m.is_copy && !m.is_cyclic && !m.needs_arena_lt)
            .collect();
        if !copy_structs.is_empty() {
            let _ = writeln!(out, "{allow_attr}#[derive(Clone, Copy, Debug)]");
            out.push_str("enum __LustDynStruct {\n");
            for (sname, _) in &copy_structs {
                let vident = sanitize_ident(sname);
                let rname = rust_struct_name(sname);
                let _ = writeln!(out, "    {vident}({rname}),");
            }
            out.push_str("}\n");
            let _ = writeln!(out, "{allow_attr}impl __LustDynStruct {{");
            out.push_str("    #[inline(always)]\n");
            out.push_str("    fn get_field(&self, field: &str) -> Result<lust_aot_rt::AotValue, lust_aot_rt::AotError> {\n");
            out.push_str("        match (self, field) {\n");
            for (sname, meta) in &copy_structs {
                let vident = sanitize_ident(sname);
                for (fi, (fname, fty)) in meta.fields.iter().enumerate() {
                    let packed = coerce_rust_to_aot_value(&format!("__s.field_{fi}"), fty, ctx);
                    let _ = writeln!(
                        out,
                        "            (Self::{vident}(__s), {fname:?}) => Ok({packed}),"
                    );
                }
            }
            out.push_str(
                "            _ => Err(lust_aot_rt::AotError::UserError(alloc::format!(\"unknown field '{}' on dynamic struct\", field))),\n",
            );
            out.push_str("        }\n");
            out.push_str("    }\n");
            out.push_str("}\n\n");
        }
    }

    if need_unknown_field {
        let _ = writeln!(out, "{allow_attr}#[inline]");
        out.push_str(
            "fn __lust_aot_unknown_get_field(obj: &lust_aot_rt::AotValue, field: &str) -> Result<lust_aot_rt::AotValue, lust_aot_rt::AotError> {\n",
        );
        out.push_str("    match obj {\n");
        out.push_str("        lust_aot_rt::AotValue::Struct(rc) => {\n");
        out.push_str("            let s = lust_aot_rt::ref_borrow(rc)?;\n");
        out.push_str("            match (s.type_id, field) {\n");
        for (sname, meta) in &ctx.structs {
            if let Some(&tid) = ctx.struct_type_ids.get(sname) {
                for (fi, (fname, _)) in meta.fields.iter().enumerate() {
                    let _ = writeln!(
                        out,
                        "                ({tid}_u32, {fname:?}) => Ok(s.fields.get({fi}).cloned().unwrap_or(lust_aot_rt::AotValue::Nil)),"
                    );
                }
            }
        }
        out.push_str(
            "                _ => Err(lust_aot_rt::AotError::UserError(alloc::format!(\"unknown field '{}' on struct type {}\", field, s.type_id))),\n",
        );
        out.push_str("            }\n");
        out.push_str("        }\n");
        out.push_str("        lust_aot_rt::AotValue::Map(rc) => {\n");
        out.push_str("            let m = lust_aot_rt::ref_borrow(rc)?;\n");
        out.push_str("            let k = lust_aot_rt::AotKey::Str(alloc::string::String::from(field));\n");
        out.push_str("            Ok(m.get(&k).cloned().unwrap_or(lust_aot_rt::AotValue::Nil))\n");
        out.push_str("        }\n");
        out.push_str(
            "        _ => Err(lust_aot_rt::AotError::UserError(alloc::format!(\"cannot access field '{}' on {}\", field, obj.type_name()))),\n",
        );
        out.push_str("    }\n");
        out.push_str("}\n\n");
    }

    if need_unknown_method {
        let has_arena = ctx.module_uses_arena;
        let arena_param = if has_arena {
            ", __arena: &'a lust_aot_rt::LustChunkArena<'a>"
        } else {
            ""
        };
        let _ = writeln!(out, "{allow_attr}#[inline]");
        let _ = writeln!(
            out,
            "fn __lust_aot_unknown_call_method<'a>(__ctx: &mut lust_aot_rt::AotCtx, __alloc: &'a lust_aot_rt::LustBudgetAlloc{arena_param}, obj: &lust_aot_rt::AotValue, method: &str, args: &[lust_aot_rt::AotValue]) -> Result<lust_aot_rt::AotValue, lust_aot_rt::AotError> {{"
        );
        out.push_str("    match obj {\n");
        out.push_str("        lust_aot_rt::AotValue::Struct(rc) => {\n");
        out.push_str("            let __tid = lust_aot_rt::ref_borrow(rc)?.type_id;\n");
        out.push_str("            match (__tid, method) {\n");
        for func in &module.functions {
            if let Some((sname, mname)) = func.signature.name.split_once(':')
                && let Some(smeta) = ctx.structs.get(sname)
                && !smeta.is_cyclic
                && !smeta.needs_arena_lt
                && let Some(&tid) = ctx.struct_type_ids.get(sname)
                && let Some(fmeta) = ctx.fn_metas.get(&func.signature.name)
            {
                let rust_sname = rust_struct_name(sname);
                let _ = writeln!(out, "                ({tid}_u32, {mname:?}) => {{");
                let _ = writeln!(out, "                    let mut __recv = {{");
                out.push_str("                        let __s = lust_aot_rt::ref_borrow(rc)?;\n");
                let mut finits = Vec::with_capacity(smeta.fields.len());
                for (fi, (_, fty)) in smeta.fields.iter().enumerate() {
                    let raw = format!(
                        "__s.fields.get({fi}).cloned().unwrap_or(lust_aot_rt::AotValue::Nil)"
                    );
                    let conv = coerce_aot_value_to_rust(&raw, fty, ctx);
                    finits.push(format!("field_{fi}: {conv}"));
                }
                let _ = writeln!(
                    out,
                    "                        {rust_sname} {{ {} }}",
                    finits.join(", ")
                );
                out.push_str("                    };\n");

                let mut cargs = Vec::new();
                if fmeta.needs_ctx {
                    cargs.push("__ctx".to_string());
                }
                if fmeta.needs_alloc {
                    cargs.push("__alloc".to_string());
                }
                if fmeta.needs_arena {
                    cargs.push("__arena".to_string());
                }
                let self_mode = fmeta.param_modes.first().copied().unwrap_or(ParamMode::ByValue);
                match self_mode {
                    ParamMode::RefStruct => cargs.push("&__recv".to_string()),
                    ParamMode::MutRefStruct => cargs.push("&mut __recv".to_string()),
                    ParamMode::RcRefCell => {
                        cargs.push("alloc::rc::Rc::new(core::cell::RefCell::new(__recv.clone()))".to_string());
                    }
                    _ => cargs.push("__recv".to_string()),
                }
                for (pi, &pid) in func.params.iter().enumerate().skip(1) {
                    let ai = pi - 1;
                    let pty = &func.locals[pid.0].ty;
                    let pmode = fmeta.param_modes.get(pi).copied().unwrap_or(ParamMode::ByValue);
                    let raw = format!(
                        "args.get({ai}).cloned().unwrap_or(lust_aot_rt::AotValue::Nil)"
                    );
                    let val = coerce_aot_value_to_rust(&raw, pty, ctx);
                    match pmode {
                        ParamMode::RefStr => cargs.push(format!("&({val})")),
                        ParamMode::RefStruct => cargs.push(format!("&({val})")),
                        _ => cargs.push(val),
                    }
                }
                let raw_call = format!("{}({})", fmeta.rust_name, cargs.join(", "));
                let call_res = if fmeta.can_fail {
                    format!("{raw_call}?")
                } else {
                    format!("match {raw_call} {{ Ok(__v) => __v }}")
                };
                let _ = writeln!(out, "                    let __res = {call_res};");
                if self_mode == ParamMode::MutRefStruct {
                    out.push_str("                    {\n");
                    out.push_str("                        let mut __s = lust_aot_rt::ref_borrow_mut(rc)?;\n");
                    for (fi, (_, fty)) in smeta.fields.iter().enumerate() {
                        let packed = coerce_rust_to_aot_value(&format!("__recv.field_{fi}"), fty, ctx);
                        let _ = writeln!(
                            out,
                            "                        if let Some(__slot) = __s.fields.get_mut({fi}) {{ *__slot = {packed}; }}"
                        );
                    }
                    out.push_str("                    }\n");
                }
                let ret_packed = coerce_rust_to_aot_value("__res", &fmeta.return_type, ctx);
                let _ = writeln!(out, "                    Ok({ret_packed})");
                out.push_str("                }\n");
            }
        }
        out.push_str(
            "                _ => Err(lust_aot_rt::AotError::UserError(alloc::format!(\"unknown method '{}' on struct type {}\", method, __tid))),\n",
        );
        out.push_str("            }\n");
        out.push_str("        }\n");
        out.push_str("        lust_aot_rt::AotValue::Map(rc) => {\n");
        out.push_str("            let __fn_val = {\n");
        out.push_str("                let m = lust_aot_rt::ref_borrow(rc)?;\n");
        out.push_str("                let k = lust_aot_rt::AotKey::Str(alloc::string::String::from(method));\n");
        out.push_str("                m.get(&k).cloned().ok_or_else(|| lust_aot_rt::AotError::UserError(alloc::format!(\"unknown method '{}' on map\", method)))?\n");
        out.push_str("            };\n");
        out.push_str("            __fn_val.call(__ctx, args)\n");
        out.push_str("        }\n");
        out.push_str(
            "        _ => Err(lust_aot_rt::AotError::UserError(alloc::format!(\"cannot call method '{}' on {}\", method, obj.type_name()))),\n",
        );
        out.push_str("    }\n");
        out.push_str("}\n\n");
    }
}

fn coerce_aot_value_to_rust(expr: &str, dst_ty: &MirType, ctx: &ModuleEmitContext<'_>) -> String {
    match dst_ty {
        MirType::Unit => format!("{{ let _ = {expr}; () }}"),
        MirType::Bool => format!("({expr}).is_truthy()"),
        MirType::Int => format!(
            "({expr}).as_int().ok_or_else(|| lust_aot_rt::AotError::UserError(alloc::string::String::from(\"expected int\")))?"
        ),
        MirType::Float => format!("({expr}).as_float_coerced()?"),
        MirType::String => format!("({expr}).to_display_string()"),
        MirType::Unknown => format!("({expr}).clone()"),
        MirType::Struct(sname) => {
            if let Some(smeta) = ctx.structs.get(sname)
                && !smeta.is_cyclic
                && !smeta.needs_arena_lt
            {
                let rust_sname = rust_struct_name(sname);
                let mut finits = Vec::with_capacity(smeta.fields.len());
                for (fi, (_, fty)) in smeta.fields.iter().enumerate() {
                    let raw = format!(
                        "__s_ref.fields.get({fi}).cloned().unwrap_or(lust_aot_rt::AotValue::Nil)"
                    );
                    finits.push(format!(
                        "field_{fi}: {}",
                        coerce_aot_value_to_rust(&raw, fty, ctx)
                    ));
                }
                format!(
                    "match &({expr}) {{ lust_aot_rt::AotValue::Struct(__rc) => {{ let __s_ref = lust_aot_rt::ref_borrow(__rc)?; {rust_sname} {{ {} }} }}, _ => return Err(lust_aot_rt::AotError::UserError(alloc::string::String::from(\"expected struct\"))) }}",
                    finits.join(", ")
                )
            } else {
                ctx.rust_default_expr(dst_ty)
            }
        }
        _ => ctx.rust_default_expr(dst_ty),
    }
}

fn coerce_rust_to_aot_value(expr: &str, src_ty: &MirType, ctx: &ModuleEmitContext<'_>) -> String {
    match src_ty {
        MirType::Unit => format!("{{ let _ = {expr}; lust_aot_rt::AotValue::Nil }}"),
        MirType::Bool => format!("lust_aot_rt::AotValue::Bool({expr})"),
        MirType::Int => format!("lust_aot_rt::AotValue::Int({expr})"),
        MirType::Float => format!("lust_aot_rt::AotValue::Float({expr})"),
        MirType::String => format!("lust_aot_rt::AotValue::str({expr})"),
        MirType::Unknown | MirType::Effect { .. } => format!("({expr}).clone()"),
        MirType::Struct(sname) => {
            if let Some(smeta) = ctx.structs.get(sname)
                && let Some(&tid) = ctx.struct_type_ids.get(sname)
            {
                let mut fvals = Vec::with_capacity(smeta.fields.len());
                for (fi, (_, fty)) in smeta.fields.iter().enumerate() {
                    let fexpr = if ctx.is_copy_type(fty) {
                        format!("__st.field_{fi}")
                    } else {
                        format!("__st.field_{fi}.clone()")
                    };
                    fvals.push(coerce_rust_to_aot_value(&fexpr, fty, ctx));
                }
                format!(
                    "{{ let __st = &({expr}); lust_aot_rt::AotValue::from_struct({tid}_u32, alloc::vec![{}]) }}",
                    fvals.join(", ")
                )
            } else {
                format!("lust_aot_rt::AotValue::str(alloc::format!(\"{{:?}}\", {expr}))")
            }
        }
        _ => format!("lust_aot_rt::AotValue::str(alloc::format!(\"{{:?}}\", {expr}))"),
    }
}

/// Traces an operand back to a local `Rvalue::Closure { func_name, .. }` (including through
/// single-copy assignments and `__lust_provide` chains) in `func`.
fn trace_closure_fn_name_in_mir(func: &MirFunction, op: &Operand) -> Option<String> {
    let mut cur = operand_exact_local(op)?;
    let mut visited = HashSet::new();
    while visited.insert(cur) {
        let mut found_next: Option<LocalId> = None;
        for block in &func.blocks {
            for stmt in &block.statements {
                if let Statement::Assign(Place::Local(dst), rval, _) = stmt
                    && *dst == cur
                {
                    match rval {
                        Rvalue::Closure { func_name, .. } => return Some(func_name.clone()),
                        Rvalue::Use(
                            Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)),
                        ) => {
                            found_next = Some(*src);
                        }
                        _ => {}
                    }
                }
            }
            if let Some(Terminator::Call {
                callee,
                args,
                destination: Place::Local(dst),
                ..
            }) = &block.terminator
                && *dst == cur
                && callee == "__lust_provide"
                && let Some(src) = args.first().and_then(operand_exact_local)
            {
                found_next = Some(src);
            }
        }
        cur = found_next?;
    }
    None
}

fn emit_public_api_wrappers(out: &mut String, module: &MirModule, ctx: &ModuleEmitContext<'_>) {
    let mut used_names: HashSet<String> = HashSet::new();
    let mut header_emitted = false;

    for func in &module.functions {
        let raw_name = &func.signature.name;
        if raw_name.starts_with("__")
            || raw_name.contains("__lambda_")
            || raw_name.contains(':')
            || ctx.nontail_body_metas.contains_key(raw_name)
            || ctx.nontail_arm_metas.contains_key(raw_name)
        {
            continue;
        }
        let Some(meta) = ctx.fn_metas.get(raw_name) else {
            continue;
        };
        if ctx.type_needs_lifetime(&meta.return_type) {
            continue;
        }
        if meta.param_modes.iter().any(|m| {
            matches!(
                m,
                ParamMode::MutVec | ParamMode::RcCell | ParamMode::RcRefCell
            )
        }) {
            continue;
        }
        let mut param_has_lt = false;
        for &pid in &func.params {
            let pty = &func.locals[pid.0].ty;
            if matches!(pty, MirType::Array(_)) {
                // Allowed only when passed as `&[T]` (RefSlice), handled below
                continue;
            }
            if ctx.type_needs_lifetime(pty) {
                param_has_lt = true;
                break;
            }
        }
        if param_has_lt {
            continue;
        }

        let short_raw = raw_name.rsplit('.').next().unwrap_or(raw_name);
        let mut pub_name = if short_raw == "main" {
            "lust_main".to_string()
        } else {
            sanitize_ident(short_raw)
        };
        if pub_name.is_empty() || !used_names.insert(pub_name.clone()) {
            let qualified = sanitize_ident(raw_name);
            if qualified.is_empty() || !used_names.insert(qualified.clone()) {
                continue;
            }
            pub_name = qualified;
        }

        if !header_emitted {
            out.push_str("// Public Ergonomic Rust API Wrappers (NG-W13-1C)\n");
            header_emitted = true;
        }

        let mut sig_params: Vec<String> = Vec::with_capacity(func.params.len());
        let mut call_args_base: Vec<String> = Vec::with_capacity(func.params.len());
        for (pi, &pid) in func.params.iter().enumerate() {
            let ty = &func.locals[pid.0].ty;
            let mode = meta.param_modes.get(pi).copied().unwrap_or(ParamMode::ByValue);
            let pname = format!("arg_{pi}");
            match mode {
                ParamMode::ByValue => {
                    let rty = ctx.rust_type(ty, "'static");
                    sig_params.push(format!("{pname}: {rty}"));
                    call_args_base.push(pname);
                }
                ParamMode::RefStruct => {
                    let rty = ctx.rust_type(ty, "'static");
                    sig_params.push(format!("{pname}: &{rty}"));
                    call_args_base.push(pname);
                }
                ParamMode::MutRefStruct => {
                    let rty = ctx.rust_type(ty, "'static");
                    sig_params.push(format!("{pname}: &mut {rty}"));
                    call_args_base.push(pname);
                }
                ParamMode::RefSlice => {
                    let elem = match ty {
                        MirType::Array(inner) => ctx.rust_type(inner, "'static"),
                        _ => "i64".to_string(),
                    };
                    sig_params.push(format!("{pname}: &[{elem}]"));
                    call_args_base.push(pname);
                }
                ParamMode::RefStr => {
                    sig_params.push(format!("{pname}: &str"));
                    call_args_base.push(pname);
                }
                ParamMode::MutVec | ParamMode::RcCell | ParamMode::RcRefCell => {}
            }
        }

        let ret_ty = ctx.rust_type(&meta.return_type, "'static");
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

        // 1. Primary ergonomic `pub fn <name>(...)`
        if ctx.interop.emit_as_include_module {
            out.push_str(INCLUDE_ALLOW_ATTR);
        }
        out.push_str("#[inline]\n");
        let mut full_call_args: Vec<String> = Vec::new();
        if meta.needs_ctx {
            full_call_args.push("&mut __ctx".to_string());
        }
        if meta.needs_alloc {
            full_call_args.push("&__alloc".to_string());
        }
        if meta.needs_arena {
            full_call_args.push("&__arena".to_string());
        }
        full_call_args.extend(call_args_base.iter().cloned());

        if !meta.can_fail {
            let _ = writeln!(
                out,
                "pub fn {pub_name}({}) -> {ret_ty} {{",
                sig_params.join(", ")
            );
            if meta.needs_ctx {
                if ctx.opts.pure_core {
                    out.push_str("    let mut __ctx = lust_aot_rt::AotCtx::pure_core();\n");
                } else {
                    let _ = writeln!(
                        out,
                        "    let mut __ctx = lust_aot_rt::AotCtx::with_limits({gas_limit_expr}, {mem_limit_expr});"
                    );
                }
            }
            if meta.needs_alloc || meta.needs_arena {
                if meta.needs_ctx {
                    out.push_str("    let __alloc = __ctx.budget_alloc.share();\n");
                } else {
                    let _ = writeln!(out, "    let __alloc = {alloc_expr};");
                }
            }
            if meta.needs_arena {
                out.push_str("    let __arena = lust_aot_rt::LustChunkArena::new(&__alloc);\n");
            }
            let _ = writeln!(
                out,
                "    let Ok(__v) = {}({});",
                meta.rust_name,
                full_call_args.join(", ")
            );
            out.push_str("    __v\n");
            out.push_str("}\n\n");
        } else {
            let _ = writeln!(
                out,
                "pub fn {pub_name}({}) -> Result<{ret_ty}, lust_aot_rt::AotError> {{",
                sig_params.join(", ")
            );
            if meta.needs_ctx {
                if ctx.opts.pure_core {
                    out.push_str("    let mut __ctx = lust_aot_rt::AotCtx::pure_core();\n");
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
            if meta.needs_alloc || meta.needs_arena {
                if meta.needs_ctx {
                    out.push_str("    let __alloc = __ctx.budget_alloc.share();\n");
                } else {
                    let _ = writeln!(out, "    let __alloc = {alloc_expr};");
                }
            }
            if meta.needs_arena {
                out.push_str("    let __arena = lust_aot_rt::LustChunkArena::new(&__alloc);\n");
            }
            let _ = writeln!(
                out,
                "    {}({})",
                meta.rust_name,
                full_call_args.join(", ")
            );
            out.push_str("}\n\n");
        }

        // 2. Context-threaded variant `pub fn <name>_with_ctx(ctx: &mut AotCtx, ...)` when ctx or fallibility is used
        if meta.needs_ctx || meta.can_fail {
            let ctx_fn_name = format!("{pub_name}_with_ctx");
            if used_names.insert(ctx_fn_name.clone()) {
                if ctx.interop.emit_as_include_module {
                    out.push_str(INCLUDE_ALLOW_ATTR);
                }
                out.push_str("#[inline]\n");
                let mut ctx_sig_params = vec!["__ctx: &mut lust_aot_rt::AotCtx".to_string()];
                ctx_sig_params.extend(sig_params.iter().cloned());
                let _ = writeln!(
                    out,
                    "pub fn {ctx_fn_name}({}) -> Result<{ret_ty}, lust_aot_rt::AotError> {{",
                    ctx_sig_params.join(", ")
                );
                let mut ctx_call_args = Vec::new();
                if meta.needs_ctx {
                    ctx_call_args.push("__ctx".to_string());
                }
                if meta.needs_alloc || meta.needs_arena {
                    out.push_str("    let __alloc = __ctx.budget_alloc.share();\n");
                }
                if meta.needs_alloc {
                    ctx_call_args.push("&__alloc".to_string());
                }
                if meta.needs_arena {
                    out.push_str("    let __arena = lust_aot_rt::LustChunkArena::new(&__alloc);\n");
                    ctx_call_args.push("&__arena".to_string());
                }
                ctx_call_args.extend(call_args_base.iter().cloned());
                if meta.can_fail {
                    let _ = writeln!(
                        out,
                        "    {}({})",
                        meta.rust_name,
                        ctx_call_args.join(", ")
                    );
                } else {
                    let _ = writeln!(
                        out,
                        "    {}({}).map_err(lust_aot_rt::AotError::from)",
                        meta.rust_name,
                        ctx_call_args.join(", ")
                    );
                }
                out.push_str("}\n\n");
            }
        }
    }
}

fn emit_entrypoint(out: &mut String, module: &MirModule, ctx: &ModuleEmitContext<'_>) {
    let main_fn = module
        .functions
        .iter()
        .find(|f| f.signature.name == "main" || f.signature.name.ends_with(".main"));

    if ctx.opts.no_std {
        out.push_str("pub fn lust_nostd_main() -> i32 {\n");
        out.push_str("    let _slab = lust_static_slab_new();\n");
        let Some(main) = main_fn else {
            out.push_str("    0\n}\n");
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
            if meta.needs_ctx {
                let _ = writeln!(out, "    let __alloc = __ctx.budget_alloc.share();");
            } else {
                let _ = writeln!(out, "    let __alloc = {alloc_expr};");
            }
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
                MirType::Int => out.push_str("        Ok(__code) => __code as i32,\n"),
                _ => out.push_str("        Ok(_) => 0,\n"),
            }
            out.push_str("        Err(_) => 1,\n");
            out.push_str("    }\n");
        } else {
            match main.signature.return_type {
                MirType::Int => {
                    let _ = writeln!(out, "    let Ok(__code) = {call_expr};");
                    out.push_str("    __code as i32\n");
                }
                _ => {
                    let _ = writeln!(out, "    let Ok(_) = {call_expr};");
                    out.push_str("    0\n");
                }
            }
        }
        out.push_str("}\n");
        return;
    }

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
        if meta.needs_ctx {
            let _ = writeln!(out, "    let __alloc = __ctx.budget_alloc.share();");
        } else {
            let _ = writeln!(out, "    let __alloc = {alloc_expr};");
        }
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
    /// Locals defined by `Rvalue::Aggregate` or `Terminator::Call` (`NG-W15-1`).
    aggregate_or_call_def: HashSet<LocalId>,
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
    /// Pre-analyzed tail-resumptive `__lust_handle(body_thunk, arms_array)` call sites (`NG-W13-1A`).
    handle_plans: HashMap<BlockId, HandleCallPlan>,
    /// Pre-analyzed one-shot non-tail `__lust_nontail_handle(body_thunk, arms_array)` call sites (`NG-W13-1B`).
    nontail_handle_plans: HashMap<BlockId, HandleCallPlan>,
    /// Pre-analyzed `__lust_yield_op(op_name, args_arr, cont_closure)` call sites (`NG-W13-1B`).
    nontail_yield_op_plans: HashMap<BlockId, NonTailYieldOpPlan>,
    /// Pre-analyzed `__lust_perform(name, [a0, a1, ...])` inline argument lists (`NG-W13-1A`).
    perform_arg_plans: HashMap<BlockId, Vec<Operand>>,
    /// Pre-analyzed `__lust_run_sync(__lust_provide*(thunk, p))` direct-stack plans (`NG-W13-1A`).
    provide_plans: HashMap<BlockId, ProvideRunPlan>,
    /// Pre-analyzed `__lust_scoped(thunk)` call sites (`NG-W13-1B`).
    scoped_plans: HashMap<BlockId, (String, Vec<(String, Operand, MirType)>)>,
    /// Pre-analyzed `__lust_scope_finalizer` / `__lust_scope_spawn` callback plans (`NG-W13-1B`).
    scope_callback_plans: HashMap<BlockId, (String, Vec<(String, Operand, MirType)>)>,
    /// Destination locals of `__lust_yield_op` / `__lust_yield_completed` (returned directly, omitted from decls).
    yield_return_locals: HashSet<LocalId>,
    /// Pre-batched loop-header gas weights for `GasMetering::LoopChunked` (`AOT-BUDGET-SOUND`).
    loop_chunked_header_gas: HashMap<BlockId, i64>,
    /// Pre-batched single-block gas weights for non-loop blocks under `GasMetering::LoopChunked`.
    block_chunked_gas: HashMap<(BlockId, usize), i64>,
    /// Non-aliased, non-escaping `Map<K, V>` locals lowered directly to unboxed `AotRawMap<K, V>` (`AOT-MAP-HASH`).
    direct_raw_map_locals: HashSet<LocalId>,
    /// Non-escaping `@std/sync.{Queue}` locals lowered directly to unboxed `AotQueue<i64>` (`AOT-DEVIRT`).
    unboxed_queue_locals: HashSet<LocalId>,
    /// Local method handles (`local offer = q.offer`) bound to an unboxed `AotQueue<i64>` (`AOT-DEVIRT`).
    unboxed_queue_methods: HashMap<LocalId, (LocalId, &'static str)>,
    /// Unboxed `AotQueue::take()` calls whose `Unknown` temp is directly consumed by an `Int` local (`call_bb -> int_dst`).
    unboxed_queue_take_redirect: HashMap<BlockId, LocalId>,
    /// Read-only `MirType::String` bindings in `IfArrayGetOk` lowered to zero-clone `&str` (`AOT-SSO-REGEX`).
    borrowed_array_str_locals: HashSet<LocalId>,
    /// Non-escaping `string.split` `Array<string>` locals backed by zero-alloc span buffer `__LustSplitSpans<'a>` -> source `&str` operand (`AOT-SSO-REGEX`).
    small_str_array_locals: HashMap<LocalId, Operand>,
    /// `string.split` locals ONLY read at const indices 0/1 (never `len`), lowered to `Copy` `(Option<&str>, Option<&str>)` (`AOT-SSO-REGEX`).
    split_pair_locals: HashSet<LocalId>,
    /// `WhileLoop`s iterating `0..array.len(split(s, sep))` fused directly to `for elem in s.split(sep)` (`AOT-SSO-REGEX`).
    split_iter_loops: HashMap<BlockId, (Operand, String, LocalId)>,
    /// Single-def constant integer locals (`LocalId -> i64`) for const index resolution.
    const_int_locals: HashMap<LocalId, i64>,
    /// Read-only `MirType::String` locals assigned only from non-mutated strings, stored as `&str` (`AOT-SSO-REGEX`).
    borrowed_select_str_locals: HashSet<LocalId>,
    /// Non-escaping `string.sub` locals backed by `alloc::borrow::Cow<'_, str>` (0 allocs on valid UTF-8, `AOT-SSO-REGEX`).
    cow_str_locals: HashSet<LocalId>,
    /// Direct `AotRawMap` read-modify-write fusion: `get_bb -> (set_bb, map_loc, key_op, val_dst)` (`AOT-MAP-HASH`).
    map_rmw_get_plans: HashMap<BlockId, (BlockId, LocalId, Operand, LocalId)>,
    /// Companion `set_bb -> map_loc` skipped into `*__rmw_slot = val` (`AOT-MAP-HASH`).
    map_rmw_set_plans: HashMap<BlockId, LocalId>,
    /// 1-byte `string.sub(s, i, i)` locals only compared against 1-byte ASCII literals, lowered to `Option<u8>` (`AOT-SSO-REGEX`).
    byte_sub_locals: HashSet<LocalId>,
    /// `string.gsub` `Array<string>` locals only read at index `0`, lowered directly to `String` (`AOT-SSO-REGEX`).
    gsub_first_only_locals: HashSet<LocalId>,
    /// Chained `MirBinaryOp::Concat` + `tostring` expressions fused into a single `String` allocation (`AOT-SSO-REGEX`).
    fused_concat_chains: HashMap<(BlockId, usize), Vec<ConcatPiece>>,
}

#[derive(Debug, Clone)]
enum ConcatPiece {
    Str(Operand),
    Int(Operand),
    Float(Operand),
}

#[derive(Debug, Clone)]
struct NonTailYieldOpPlan {
    op_name: String,
    arg_ops: Vec<Operand>,
    cont_fn_name: String,
    cont_captures: Vec<(String, Operand, MirType)>,
}

/// Scans all `__lust_nontail_handle` sites in `module` and computes exact parameterized
/// `EffectYield<OpArg, ResumeVal, BodyRet>` signatures for every lifted non-tail body and arm (`NG-W13-1B`).
fn analyze_nontail_effect_metas(
    module: &MirModule,
) -> (
    HashMap<String, NonTailYieldSig>,
    HashMap<String, NonTailArmMeta>,
) {
    let mut body_metas: HashMap<String, NonTailYieldSig> = HashMap::new();
    let mut arm_metas: HashMap<String, NonTailArmMeta> = HashMap::new();

    let fn_by_name: HashMap<&str, &MirFunction> = module
        .functions
        .iter()
        .map(|f| (f.signature.name.as_str(), f))
        .collect();

    for func in &module.functions {
        for block in &func.blocks {
            let Some(Terminator::Call {
                callee,
                args,
                destination,
                ..
            }) = &block.terminator
            else {
                continue;
            };
            if callee != "__lust_nontail_handle" {
                continue;
            }
            let Some((body_fn_name, arms)) = trace_mir_nontail_handle_site(func, args) else {
                continue;
            };
            let Some(&body_fn) = fn_by_name.get(body_fn_name.as_str()) else {
                continue;
            };

            let mut body_ret_cand: Option<MirType> = None;
            let mut body_ret_conflict = false;
            let mut resume_val_cand: Option<MirType> = None;
            let mut resume_val_conflict = false;
            let record = |slot: &mut Option<MirType>, conflict: &mut bool, ty: MirType| {
                if ty == MirType::Unknown {
                    return;
                }
                match slot {
                    None => *slot = Some(ty),
                    Some(prev) if *prev == ty => {}
                    Some(_) => *conflict = true,
                }
            };

            for b in &body_fn.blocks {
                if let Some(Terminator::Call {
                    callee: bc,
                    args: bargs,
                    ..
                }) = &b.terminator
                {
                    if bc == "__lust_yield_completed"
                        && let Some(a0) = bargs.first()
                    {
                        record(
                            &mut body_ret_cand,
                            &mut body_ret_conflict,
                            operand_mir_type(a0, body_fn),
                        );
                    } else if bc == "__lust_yield_op"
                        && bargs.len() >= 3
                        && let Some((cont_name, cont_caps)) =
                            trace_mir_closure_and_caps(body_fn, &bargs[2])
                        && let Some(&cf) = fn_by_name.get(cont_name.as_str())
                    {
                        record(
                            &mut body_ret_cand,
                            &mut body_ret_conflict,
                            cf.signature.return_type.clone(),
                        );
                        if let Some(&pid) = cf.params.get(cont_caps.len()) {
                            let pty = cf.locals[pid.0].ty.clone();
                            let used = count_local_uses(cf).get(&pid).copied().unwrap_or(0) > 0;
                            if used || pty != MirType::Int {
                                record(&mut resume_val_cand, &mut resume_val_conflict, pty);
                            }
                        }
                    }
                }
            }

            if let Place::Local(dst) = destination {
                let dst_ty = func.locals[dst.0].ty.clone();
                if body_ret_cand.is_none() && dst_ty != MirType::Unknown {
                    body_ret_cand = Some(dst_ty);
                }
            }

            for (_, arm_fn_name, _) in &arms {
                if let Some(&af) = fn_by_name.get(arm_fn_name.as_str()) {
                    for b in &af.blocks {
                        if let Some(Terminator::Call {
                            callee: ac,
                            args: aargs,
                            ..
                        }) = &b.terminator
                            && ac == "__lust_nontail_resume"
                            && let Some(a0) = aargs.first()
                        {
                            record(
                                &mut resume_val_cand,
                                &mut resume_val_conflict,
                                operand_mir_type(a0, af),
                            );
                        }
                    }
                }
            }

            let body_ret_ty = if body_ret_conflict {
                MirType::Unknown
            } else {
                body_ret_cand.unwrap_or(MirType::Unit)
            };
            let resume_val_ty = if resume_val_conflict {
                MirType::Unknown
            } else {
                resume_val_cand.unwrap_or(MirType::Unit)
            };

            let op_arg_mode = if arms.len() == 1 {
                let (_, ref arm_fn_name, n_caps) = arms[0];
                if let Some(&af) = fn_by_name.get(arm_fn_name.as_str()) {
                    let explicit_count = af.params.len().saturating_sub(n_caps);
                    match explicit_count {
                        0 => NonTailOpArgMode::Unit,
                        1 => {
                            let pty = af.locals[af.params[n_caps].0].ty.clone();
                            NonTailOpArgMode::Single(pty)
                        }
                        _ => NonTailOpArgMode::TaggedVec,
                    }
                } else {
                    NonTailOpArgMode::TaggedVec
                }
            } else {
                NonTailOpArgMode::TaggedVec
            };

            body_metas.insert(
                body_fn_name,
                NonTailYieldSig {
                    op_arg_mode,
                    resume_val_ty: resume_val_ty.clone(),
                    body_ret_ty: body_ret_ty.clone(),
                },
            );
            for (_, arm_fn_name, _) in arms {
                arm_metas.insert(
                    arm_fn_name,
                    NonTailArmMeta {
                        resume_val_ty: resume_val_ty.clone(),
                        body_ret_ty: body_ret_ty.clone(),
                    },
                );
            }
        }
    }

    (body_metas, arm_metas)
}

#[derive(Debug, Clone)]
struct HandleArmPlan {
    effect_name: String,
    arm_fn_name: String,
    captures: Vec<(String, Operand, MirType)>,
}

#[derive(Debug, Clone)]
struct HandleCallPlan {
    body_fn_name: String,
    body_captures: Vec<(String, Operand, MirType)>,
    arms: Vec<HandleArmPlan>,
}

#[derive(Debug, Clone)]
struct ProvideRunPlan {
    thunk_fn_name: String,
    thunk_captures: Vec<(String, Operand, MirType)>,
    /// Provided `(service_key, provider_operand)` pairs ordered outer -> inner.
    providers: Vec<(String, Operand)>,
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
    let mut aggregate_or_call_def: HashSet<LocalId> = HashSet::new();
    let mut switch_tag_source: HashMap<LocalId, LocalId> = HashMap::new();
    for block in &func.blocks {
        if block.is_cleanup {
            continue;
        }
        for stmt in &block.statements {
            if let Statement::Assign(Place::Local(dst), rval, _) = stmt {
                *local_def_counts.entry(*dst).or_insert(0) += 1;
                if matches!(rval, Rvalue::Aggregate(_, _)) {
                    aggregate_or_call_def.insert(*dst);
                }
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
            aggregate_or_call_def.insert(*dst);
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
        aggregate_or_call_def,
        if_option_some: HashMap::new(),
        errdefer_guards: Vec::new(),
        in_cleanup_closure: None,
        cleanup_var_remap: HashMap::new(),
        switch_tag_source,
        active_enum_variant: HashMap::new(),
        local_name_to_id,
        handle_plans: HashMap::new(),
        nontail_handle_plans: HashMap::new(),
        nontail_yield_op_plans: HashMap::new(),
        perform_arg_plans: HashMap::new(),
        provide_plans: HashMap::new(),
        scoped_plans: HashMap::new(),
        scope_callback_plans: HashMap::new(),
        yield_return_locals: HashSet::new(),
        loop_chunked_header_gas: HashMap::new(),
        block_chunked_gas: HashMap::new(),
        direct_raw_map_locals: HashSet::new(),
        unboxed_queue_locals: HashSet::new(),
        unboxed_queue_methods: HashMap::new(),
        unboxed_queue_take_redirect: HashMap::new(),
        borrowed_array_str_locals: HashSet::new(),
        small_str_array_locals: HashMap::new(),
        split_pair_locals: HashSet::new(),
        split_iter_loops: HashMap::new(),
        const_int_locals: HashMap::new(),
        borrowed_select_str_locals: HashSet::new(),
        cow_str_locals: HashSet::new(),
        map_rmw_get_plans: HashMap::new(),
        map_rmw_set_plans: HashMap::new(),
        byte_sub_locals: HashSet::new(),
        gsub_first_only_locals: HashSet::new(),
        fused_concat_chains: HashMap::new(),
    };

    // Pre-passes on the structured function
    fe.analyze_effect_handles();
    fe.analyze_nontail_yields_and_scopes();
    fe.analyze_effect_performs();
    fe.analyze_effect_provides();
    fe.analyze_single_def_aliases();
    fe.analyze_direct_raw_maps();
    fe.analyze_map_rmw_fusion(&tree);
    fe.analyze_unboxed_queues();
    fe.analyze_tostring_concat_fusion();
    fe.analyze_chained_concat_fusion();
    fe.analyze_borrowed_select_str_locals();
    fe.analyze_cow_sub_str_locals();
    fe.analyze_single_byte_sub_locals();
    fe.analyze_structured_folds(&tree);
    fe.analyze_small_str_split_arrays(&tree);
    fe.analyze_gsub_first_only_locals(&tree);
    fe.analyze_loop_chunked_gas(&tree);
    fe.analyze_inlined_defer_strips();

    // Emit function signature
    fe.emit_signature();

    // Emit local declarations (including Rc<Cell>/Rc<RefCell> wrappers for Tier 4)
    fe.emit_local_declarations();

    // Emit `defer` / `errdefer` guards via `core::mem::DropGuard` (NG-W12-4B, Hazard 9)
    fe.emit_cleanup_guards()?;

    // Emit structured body
    fe.emit_node(&tree, 1)?;

    // Safety trailing return in case of empty trailing fallthrough on Unit/Int/EffectYield functions
    if !node_always_returns(&tree) {
        let def_ret = if let Some(ysig) = fe.ctx.nontail_body_metas.get(&func.signature.name) {
            let inner_def = fe.ctx.rust_default_expr(&ysig.body_ret_ty);
            format!("lust_aot_rt::EffectYield::Completed({inner_def})")
        } else {
            fe.ctx.rust_default_expr(&func.signature.return_type)
        };
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
        if let Some(arm_meta) = self.ctx.nontail_arm_metas.get(&self.func.signature.name) {
            let res_ty = self.ctx.rust_type(&arm_meta.resume_val_ty, "'static");
            let ret_ty = self.ctx.rust_type(&arm_meta.body_ret_ty, "'static");
            params.push(format!(
                "__k: &mut lust_aot_rt::OneShotResume<{res_ty}, {ret_ty}>"
            ));
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

        let ret_ok_ty = if let Some(ysig) = self.ctx.nontail_body_metas.get(&self.func.signature.name)
        {
            let op_ty = self.ctx.nontail_op_arg_rust_type(&ysig.op_arg_mode);
            let res_ty = self.ctx.rust_type(&ysig.resume_val_ty, "'static");
            let body_ty = self.ctx.rust_type(&ysig.body_ret_ty, "'static");
            format!("lust_aot_rt::EffectYield<{op_ty}, {res_ty}, {body_ty}>")
        } else {
            self.ctx.rust_type(&self.func.signature.return_type, "'a")
        };
        let err_ty = if self.meta.can_fail {
            "lust_aot_rt::AotError"
        } else {
            "core::convert::Infallible"
        };

        if let Some(eff_spec) = self.ctx.ast_meta.fn_effects.get(&self.func.signature.name)
            && !eff_spec.requirements.is_empty()
        {
            let req_names: Vec<String> = eff_spec
                .requirements
                .iter()
                .map(|r| format!("LustService_{}", sanitize_ident(&ast_type_target_name(r))))
                .collect();
            let _ = writeln!(
                self.out,
                "// Nominal service requirements (NG-W13-1A): {}",
                req_names.join(" + ")
            );
        }

        if self.ctx.interop.emit_as_include_module {
            self.out.push_str(INCLUDE_ALLOW_ATTR);
        }
        if !self.meta.can_fail
            && !self.meta.needs_ctx
            && self.func.signature.name != "main"
            && self.func.blocks.len() <= 16
        {
            self.out.push_str("#[inline]\n");
        }
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
            if self.unboxed_queue_locals.contains(&decl.id) {
                let _ = writeln!(
                    self.out,
                    "    let mut _{}: lust_aot_rt::AotQueue<i64> = lust_aot_rt::AotQueue::new();",
                    decl.id.0
                );
                continue;
            }
            if self.byte_sub_locals.contains(&decl.id) {
                let _ = writeln!(self.out, "    let mut _{}: Option<u8> = None;", decl.id.0);
                continue;
            }
            if self.gsub_first_only_locals.contains(&decl.id) {
                let _ = writeln!(
                    self.out,
                    "    let mut _{}: alloc::string::String = alloc::string::String::new();",
                    decl.id.0
                );
                continue;
            }
            if self.borrowed_select_str_locals.contains(&decl.id) {
                let _ = writeln!(self.out, "    let mut _{}: &str = \"\";", decl.id.0);
                continue;
            }
            if self.cow_str_locals.contains(&decl.id) {
                let _ = writeln!(
                    self.out,
                    "    let mut _{}: alloc::borrow::Cow<'_, str> = alloc::borrow::Cow::Borrowed(\"\");",
                    decl.id.0
                );
                continue;
            }
            if self.split_pair_locals.contains(&decl.id) {
                let _ = writeln!(
                    self.out,
                    "    let mut _{}: (Option<&str>, Option<&str>) = (None, None);",
                    decl.id.0
                );
                continue;
            }
            if self.small_str_array_locals.contains_key(&decl.id) {
                let _ = writeln!(
                    self.out,
                    "    let mut _{}: __LustSplitSpans<'a> = __LustSplitSpans::new_in(__alloc);",
                    decl.id.0
                );
                continue;
            }
            if self.direct_raw_map_locals.contains(&decl.id)
                && let MirType::Map(k, v) = &decl.ty
            {
                let rk = self.ctx.rust_map_key_type(k);
                let rv = self.ctx.rust_type(v, "'a");
                let _ = writeln!(
                    self.out,
                    "    let mut _{}: lust_aot_rt::AotRawMap<{rk}, {rv}> = lust_aot_rt::AotRawMap::new();",
                    decl.id.0
                );
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
        if self.meta.keep_inlined_defers_direct {
            return;
        }
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
        if self.func.cleanup_blocks.is_empty() || self.meta.keep_inlined_defers_direct {
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

    /// Pre-pass (`NG-W13-1A`): traces `__lust_handle(body_thunk, arms_array)` back to its
    /// lifted body closure and `(effect_name, arm_closure)` pairs, skipping the intermediate
    /// heterogeneous tuple/array temps so zero closure-tuple allocations are emitted.
    fn analyze_effect_handles(&mut self) {
        let mut def_loc: HashMap<LocalId, (BlockId, usize)> = HashMap::new();
        for block in &self.func.blocks {
            if block.is_cleanup {
                continue;
            }
            for (sidx, stmt) in block.statements.iter().enumerate() {
                if let Statement::Assign(Place::Local(dst), _, _) = stmt {
                    def_loc.insert(*dst, (block.id, sidx));
                }
            }
        }

        let resolve_closure = |op: &Operand| -> Option<(
            String,
            Vec<(String, Operand, MirType)>,
            Vec<(LocalId, BlockId, usize)>,
        )> {
            let mut cur = operand_exact_local(op)?;
            let mut chain = Vec::new();
            let mut seen = HashSet::new();
            while seen.insert(cur) {
                let &(bb, sidx) = def_loc.get(&cur)?;
                if let Statement::Assign(_, rval, _) = &self.func.blocks[bb.0].statements[sidx] {
                    match rval {
                        Rvalue::Closure {
                            func_name,
                            captures,
                        } => {
                            chain.push((cur, bb, sidx));
                            return Some((func_name.clone(), captures.clone(), chain));
                        }
                        Rvalue::Use(
                            Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)),
                        ) => {
                            chain.push((cur, bb, sidx));
                            cur = *src;
                        }
                        _ => return None,
                    }
                } else {
                    return None;
                }
            }
            None
        };

        for block in &self.func.blocks {
            if block.is_cleanup {
                continue;
            }
            let Some(Terminator::Call { callee, args, .. }) = &block.terminator else {
                continue;
            };
            if !matches!(callee.as_str(), "__lust_handle" | "__lust_nontail_handle")
                || args.len() != 2
            {
                continue;
            }
            let is_nontail = callee == "__lust_nontail_handle";

            let Some((body_fn_name, body_captures, body_chain)) = resolve_closure(&args[0]) else {
                continue;
            };
            let Some(arms_loc) = operand_exact_local(&args[1]) else {
                continue;
            };
            let Some(&(arms_bb, arms_sidx)) = def_loc.get(&arms_loc) else {
                continue;
            };
            let Statement::Assign(_, Rvalue::Aggregate(MirType::Array(_), tup_ops), _) =
                &self.func.blocks[arms_bb.0].statements[arms_sidx]
            else {
                continue;
            };

            let mut arms = Vec::with_capacity(tup_ops.len());
            let mut temps_to_skip: Vec<(LocalId, BlockId, usize)> = Vec::new();
            temps_to_skip.extend(body_chain);
            temps_to_skip.push((arms_loc, arms_bb, arms_sidx));

            let mut all_ok = true;
            for tup_op in tup_ops {
                let Some(tup_loc) = operand_exact_local(tup_op) else {
                    all_ok = false;
                    break;
                };
                let Some(&(tup_bb, tup_sidx)) = def_loc.get(&tup_loc) else {
                    all_ok = false;
                    break;
                };
                let Statement::Assign(_, Rvalue::Aggregate(MirType::Tuple(_), elems), _) =
                    &self.func.blocks[tup_bb.0].statements[tup_sidx]
                else {
                    all_ok = false;
                    break;
                };
                if elems.len() < 2 {
                    all_ok = false;
                    break;
                }
                let effect_name = match &elems[0] {
                    Operand::Constant(MirConstant::String(s)) => s.clone(),
                    _ => {
                        all_ok = false;
                        break;
                    }
                };
                let Some((arm_fn_name, arm_captures, arm_chain)) = resolve_closure(&elems[1])
                else {
                    all_ok = false;
                    break;
                };
                temps_to_skip.push((tup_loc, tup_bb, tup_sidx));
                temps_to_skip.extend(arm_chain);
                arms.push(HandleArmPlan {
                    effect_name,
                    arm_fn_name,
                    captures: arm_captures,
                });
            }

            if all_ok {
                for (loc, bb, sidx) in temps_to_skip {
                    if self.local_use_counts.get(&loc).copied().unwrap_or(0) <= 1
                        && self.local_def_counts.get(&loc).copied().unwrap_or(0) <= 1
                    {
                        self.skip_stmts.insert((bb, sidx));
                        self.omitted_locals.insert(loc);
                    }
                }
                let plan = HandleCallPlan {
                    body_fn_name,
                    body_captures,
                    arms,
                };
                if is_nontail {
                    self.nontail_handle_plans.insert(block.id, plan);
                } else {
                    self.handle_plans.insert(block.id, plan);
                }
            }
        }
    }

    /// Pre-pass (`NG-W13-1B`): traces `__lust_yield_op`, `__lust_yield_completed`,
    /// `__lust_scoped`, `__lust_scope_finalizer`, and `__lust_scope_spawn` back to their
    /// lifted closures and inline argument lists so zero intermediate temps leak into Rust.
    fn analyze_nontail_yields_and_scopes(&mut self) {
        let mut def_loc: HashMap<LocalId, (BlockId, usize)> = HashMap::new();
        for block in &self.func.blocks {
            if block.is_cleanup {
                continue;
            }
            for (sidx, stmt) in block.statements.iter().enumerate() {
                if let Statement::Assign(Place::Local(dst), _, _) = stmt {
                    def_loc.insert(*dst, (block.id, sidx));
                }
            }
        }

        let resolve_closure = |op: &Operand| -> Option<(
            String,
            Vec<(String, Operand, MirType)>,
            Vec<(LocalId, BlockId, usize)>,
        )> {
            let mut cur = operand_exact_local(op)?;
            let mut chain = Vec::new();
            let mut seen = HashSet::new();
            while seen.insert(cur) {
                let &(bb, sidx) = def_loc.get(&cur)?;
                if let Statement::Assign(_, rval, _) = &self.func.blocks[bb.0].statements[sidx] {
                    match rval {
                        Rvalue::Closure {
                            func_name,
                            captures,
                        } => {
                            chain.push((cur, bb, sidx));
                            return Some((func_name.clone(), captures.clone(), chain));
                        }
                        Rvalue::Use(
                            Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)),
                        ) => {
                            chain.push((cur, bb, sidx));
                            cur = *src;
                        }
                        _ => return None,
                    }
                } else {
                    return None;
                }
            }
            None
        };

        for block in &self.func.blocks {
            if block.is_cleanup {
                continue;
            }
            let Some(Terminator::Call {
                callee,
                args,
                destination,
                ..
            }) = &block.terminator
            else {
                continue;
            };
            match callee.as_str() {
                "__lust_yield_completed" => {
                    if let Place::Local(dst) = destination {
                        self.yield_return_locals.insert(*dst);
                    }
                }
                "__lust_yield_op" if args.len() >= 3 => {
                    if let Place::Local(dst) = destination {
                        self.yield_return_locals.insert(*dst);
                    }
                    let op_name = match &args[0] {
                        Operand::Constant(MirConstant::String(s)) => s.clone(),
                        _ => continue,
                    };
                    let Some((cont_fn_name, cont_captures, cont_chain)) =
                        resolve_closure(&args[2])
                    else {
                        continue;
                    };
                    let mut temps_to_skip = cont_chain;
                    let arg_ops = if let Some(arr_loc) = operand_exact_local(&args[1])
                        && let Some(&(arr_bb, arr_sidx)) = def_loc.get(&arr_loc)
                        && let Statement::Assign(_, Rvalue::Aggregate(MirType::Array(_), elems), _) =
                            &self.func.blocks[arr_bb.0].statements[arr_sidx]
                    {
                        temps_to_skip.push((arr_loc, arr_bb, arr_sidx));
                        elems.clone()
                    } else {
                        Vec::new()
                    };
                    for (loc, bb, sidx) in temps_to_skip {
                        if self.local_use_counts.get(&loc).copied().unwrap_or(0) <= 1
                            && self.local_def_counts.get(&loc).copied().unwrap_or(0) <= 1
                        {
                            self.skip_stmts.insert((bb, sidx));
                            self.omitted_locals.insert(loc);
                        }
                    }
                    self.nontail_yield_op_plans.insert(
                        block.id,
                        NonTailYieldOpPlan {
                            op_name,
                            arg_ops,
                            cont_fn_name,
                            cont_captures,
                        },
                    );
                }
                "__lust_scoped" if args.len() == 1 => {
                    if let Some((fn_name, caps, chain)) = resolve_closure(&args[0]) {
                        for (loc, bb, sidx) in chain {
                            if self.local_use_counts.get(&loc).copied().unwrap_or(0) <= 1
                                && self.local_def_counts.get(&loc).copied().unwrap_or(0) <= 1
                            {
                                self.skip_stmts.insert((bb, sidx));
                                self.omitted_locals.insert(loc);
                            }
                        }
                        self.scoped_plans.insert(block.id, (fn_name, caps));
                    }
                }
                "__lust_scope_finalizer" | "__lust_scope_spawn" if args.len() == 1 => {
                    if let Some((fn_name, caps, chain)) = resolve_closure(&args[0]) {
                        for (loc, bb, sidx) in chain {
                            if self.local_use_counts.get(&loc).copied().unwrap_or(0) <= 1
                                && self.local_def_counts.get(&loc).copied().unwrap_or(0) <= 1
                            {
                                self.skip_stmts.insert((bb, sidx));
                                self.omitted_locals.insert(loc);
                            }
                        }
                        self.scope_callback_plans.insert(block.id, (fn_name, caps));
                    }
                }
                _ => {}
            }
        }

        // Propagate `yield_return_locals` across simple local copy temps (e.g. `_4 = _3; return _4;`)
        // so dead post-yield return blocks never emit a mismatched `Completed(_4)` fallback.
        if !self.yield_return_locals.is_empty() {
            for _ in 0..4 {
                let mut changed = false;
                for block in &self.func.blocks {
                    if block.is_cleanup {
                        continue;
                    }
                    for stmt in &block.statements {
                        if let Statement::Assign(
                            Place::Local(dst),
                            Rvalue::Use(
                                Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)),
                            ),
                            _,
                        ) = stmt
                            && self.yield_return_locals.contains(src)
                            && self.yield_return_locals.insert(*dst)
                        {
                            changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }
    }

    /// Pre-pass (`NG-W13-1A`): traces `__lust_perform(name, [a0, a1, ...])` so argument arrays
    /// are passed on the stack as `&[AotValue]` with zero heap `Vec` allocation.
    fn analyze_effect_performs(&mut self) {
        let mut def_loc: HashMap<LocalId, (BlockId, usize)> = HashMap::new();
        for block in &self.func.blocks {
            if block.is_cleanup {
                continue;
            }
            for (sidx, stmt) in block.statements.iter().enumerate() {
                if let Statement::Assign(Place::Local(dst), _, _) = stmt {
                    def_loc.insert(*dst, (block.id, sidx));
                }
            }
        }

        for block in &self.func.blocks {
            if block.is_cleanup {
                continue;
            }
            let Some(Terminator::Call { callee, args, .. }) = &block.terminator else {
                continue;
            };
            if !matches!(callee.as_str(), "__lust_perform" | "perform") || args.len() < 2 {
                continue;
            }
            let Some(arr_loc) = operand_exact_local(&args[1]) else {
                continue;
            };
            if self.local_use_counts.get(&arr_loc).copied().unwrap_or(0) > 1
                || self.local_def_counts.get(&arr_loc).copied().unwrap_or(0) > 1
            {
                continue;
            }
            let Some(&(arr_bb, arr_sidx)) = def_loc.get(&arr_loc) else {
                continue;
            };
            if let Statement::Assign(_, Rvalue::Aggregate(MirType::Array(_), elems), _) =
                &self.func.blocks[arr_bb.0].statements[arr_sidx]
            {
                self.perform_arg_plans.insert(block.id, elems.clone());
                self.skip_stmts.insert((arr_bb, arr_sidx));
                self.omitted_locals.insert(arr_loc);
            }
        }
    }

    /// Pre-pass (`NG-W13-1A`): traces `__lust_run_sync(__lust_provide*(thunk, p))` chains back
    /// to a lifted effect thunk closure so `provide + run_sync` executes directly on the stack
    /// with scoped `with_provided_service` `DropGuard`s and zero `Rc` closure allocation.
    fn analyze_effect_provides(&mut self) {
        let mut assign_def: HashMap<LocalId, (BlockId, usize)> = HashMap::new();
        let mut provide_call_def: HashMap<LocalId, (BlockId, Operand, Operand)> = HashMap::new();
        for block in &self.func.blocks {
            if block.is_cleanup {
                continue;
            }
            for (sidx, stmt) in block.statements.iter().enumerate() {
                if let Statement::Assign(Place::Local(dst), _, _) = stmt {
                    assign_def.insert(*dst, (block.id, sidx));
                }
            }
            if let Some(Terminator::Call {
                callee,
                args,
                destination: Place::Local(dst),
                ..
            }) = &block.terminator
                && callee == "__lust_provide"
                && args.len() == 2
            {
                provide_call_def.insert(*dst, (block.id, args[0].clone(), args[1].clone()));
            }
        }

        for block in &self.func.blocks {
            if block.is_cleanup {
                continue;
            }
            let Some(Terminator::Call { callee, args, .. }) = &block.terminator else {
                continue;
            };
            if callee != "__lust_run_sync" || args.len() != 1 {
                continue;
            }

            let mut cur_loc = match operand_exact_local(&args[0]) {
                Some(l) => l,
                None => continue,
            };
            let mut providers_rev: Vec<(String, Operand)> = Vec::new();
            let mut skip_provide_calls: Vec<(LocalId, BlockId)> = Vec::new();
            let mut skip_assigns: Vec<(LocalId, BlockId, usize)> = Vec::new();
            let mut seen = HashSet::new();
            let mut resolved_thunk: Option<(String, Vec<(String, Operand, MirType)>)> = None;

            while seen.insert(cur_loc) {
                if let Some((p_bb, recv_op, prov_op)) = provide_call_def.get(&cur_loc).cloned() {
                    let svc_key = match operand_mir_type(&prov_op, self.func) {
                        MirType::Struct(sn) => sn,
                        other => format!("{other}"),
                    };
                    providers_rev.push((svc_key, prov_op));
                    skip_provide_calls.push((cur_loc, p_bb));
                    if let Some(next_loc) = operand_exact_local(&recv_op) {
                        cur_loc = next_loc;
                        continue;
                    }
                    break;
                }
                if let Some(&(a_bb, a_sidx)) = assign_def.get(&cur_loc) {
                    if let Statement::Assign(_, rval, _) =
                        &self.func.blocks[a_bb.0].statements[a_sidx]
                    {
                        match rval {
                            Rvalue::Closure {
                                func_name,
                                captures,
                            } => {
                                skip_assigns.push((cur_loc, a_bb, a_sidx));
                                resolved_thunk = Some((func_name.clone(), captures.clone()));
                                break;
                            }
                            Rvalue::Use(
                                Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)),
                            ) => {
                                skip_assigns.push((cur_loc, a_bb, a_sidx));
                                cur_loc = *src;
                                continue;
                            }
                            _ => break,
                        }
                    }
                }
                break;
            }

            if let Some((thunk_fn_name, thunk_captures)) = resolved_thunk {
                providers_rev.reverse();
                for (loc, p_bb) in skip_provide_calls {
                    if self.local_use_counts.get(&loc).copied().unwrap_or(0) <= 1
                        && self.local_def_counts.get(&loc).copied().unwrap_or(0) <= 1
                    {
                        self.skip_calls.insert(p_bb);
                        self.omitted_locals.insert(loc);
                    }
                }
                for (loc, a_bb, a_sidx) in skip_assigns {
                    if self.local_use_counts.get(&loc).copied().unwrap_or(0) <= 1
                        && self.local_def_counts.get(&loc).copied().unwrap_or(0) <= 1
                    {
                        self.skip_stmts.insert((a_bb, a_sidx));
                        self.omitted_locals.insert(loc);
                    }
                }
                self.provide_plans.insert(
                    block.id,
                    ProvideRunPlan {
                        thunk_fn_name,
                        thunk_captures,
                        providers: providers_rev,
                    },
                );
            }
        }
    }

    /// Returns `true` when `loc` is an unnamed single-definition, single-use aggregate or call
    /// materialization temp (`_tmp = [1, 2]; _a = Move(_tmp)`), matching `detect_aliased_container_locals`.
    fn is_pure_init_temp(&self, loc: LocalId) -> bool {
        !self.func.params.contains(&loc)
            && self.func.locals[loc.0].name.is_none()
            && self.local_def_counts.get(&loc).copied().unwrap_or(0) == 1
            && self.aggregate_or_call_def.contains(&loc)
            && self.local_use_counts.get(&loc).copied().unwrap_or(0) <= 1
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
            let ty = &self.func.locals[dst.0].ty;
            let is_ref_semantic = matches!(
                ty,
                MirType::Array(_) | MirType::Struct(_) | MirType::String
            );
            if !is_ref_semantic {
                continue;
            }
            let src_defs = def_counts.get(&src).copied().unwrap_or(0);
            let src_is_stable = if param_set.contains(&src) {
                src_defs == 0
            } else if matches!(ty, MirType::String) {
                // Strings are immutable values; `is_src_mutated_while_dst_live` below guards
                // against `src` being reassigned before the last read of `dst`.
                true
            } else {
                src_defs == 1
            };
            if !src_is_stable {
                continue;
            }
            // Match `detect_aliased_container_locals` (`NG-W15-1`): fold named locals even when
            // syntactically read once inside a loop (e.g. `_41 = Copy(win)` in `BENCH-NG-15`),
            // skipping only pure unnamed single-use init temporaries.
            if self.is_pure_init_temp(src) {
                continue;
            }
            // Do not fold `dst -> src` if `src` is mutated along any path between `dst = Copy(src)`
            // and a use of `dst` without passing through `dst`'s definition again.
            if is_src_mutated_while_dst_live(
                self.func,
                src,
                dst,
                bb,
                sidx,
                &self.ctx.fn_metas,
            ) {
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

    /// Pre-pass (`AOT-MAP-HASH`): identifies non-aliased, non-escaping `Map<K, V>` locals that
    /// can be stored directly as `lust_aot_rt::AotRawMap<K, V>` (bypassing `Rc<RefCell>`).
    fn analyze_direct_raw_maps(&mut self) {
        let param_set: HashSet<LocalId> = self.func.params.iter().copied().collect();
        let mut candidates: HashSet<LocalId> = HashSet::new();
        for decl in &self.func.locals {
            let loc = decl.id;
            if matches!(decl.ty, MirType::Map(_, _))
                && !param_set.contains(&loc)
                && loc != self.func.return_local
                && !self.omitted_locals.contains(&loc)
                && self.local_storage(loc) == LocalStorage::Direct
            {
                candidates.insert(loc);
            }
        }
        if candidates.is_empty() {
            return;
        }

        // Track pure init-temp copies `dst = Copy/Move(src)` between Map locals so we can keep
        // both direct when `src` is a single-use `map.new` init temp, and disqualify genuine aliases.
        let mut disqual: HashSet<LocalId> = HashSet::new();
        let mut init_copy_edges: Vec<(LocalId, LocalId, BlockId, usize)> = Vec::new();
        for block in &self.func.blocks {
            if block.is_cleanup {
                // Any map touched inside a cleanup closure must stay `AotMap` (`Rc<RefCell>`).
                let mut used = HashSet::new();
                collect_block_used_locals(block, &mut used);
                for u in used {
                    disqual.insert(u);
                }
                continue;
            }
            for (sidx, stmt) in block.statements.iter().enumerate() {
                if let Statement::Assign(place, rval, _) = stmt {
                    match rval {
                        Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))) => {
                            if let Place::Local(dst) = place
                                && candidates.contains(dst)
                                && candidates.contains(src)
                            {
                                if self.is_pure_init_temp(*src) {
                                    init_copy_edges.push((*dst, *src, block.id, sidx));
                                } else {
                                    disqual.insert(*dst);
                                    disqual.insert(*src);
                                }
                                continue;
                            }
                            if candidates.contains(src) {
                                disqual.insert(*src);
                            }
                        }
                        Rvalue::UnaryOp(MirUnaryOp::Len, op) => {
                            if let Some(l) = operand_exact_local(op)
                                && candidates.contains(&l)
                            {
                                // Allowed on AotRawMap!
                                continue;
                            }
                        }
                        _ => {
                            let mut tmp = HashSet::new();
                            rval.collect_uses(&mut tmp);
                            for id in tmp {
                                let l = LocalId(id);
                                if candidates.contains(&l) {
                                    disqual.insert(l);
                                }
                            }
                        }
                    }
                    if let Place::Index(base, _) = place {
                        // Indexed write `m[k] = v` is allowed on AotRawMap!
                        if let Place::Local(bl) = base.as_ref()
                            && candidates.contains(bl)
                        {
                            continue;
                        }
                    }
                    if !matches!(place, Place::Local(_)) {
                        let rl = place.root_local();
                        if candidates.contains(&rl) {
                            disqual.insert(rl);
                        }
                    }
                }
            }
            if let Some(term) = &block.terminator {
                match term {
                    Terminator::Call {
                        callee,
                        args,
                        destination,
                        ..
                    } => {
                        let is_direct_map_builtin = matches!(
                            callee.as_str(),
                            "map.new"
                                | "map.set"
                                | "table.set"
                                | "map.get"
                                | "table.get"
                                | "map.get_unwrap"
                                | "map.get_unwrap_or"
                                | "map.has"
                                | "map.len"
                                | "map.is_empty"
                                | "map.delete"
                                | "map.remove"
                                | "map.clear"
                                | "map.keys"
                                | "map.values"
                                | "array.len"
                                | "table.len"
                        );
                        if !is_direct_map_builtin {
                            if let Place::Local(dst) = destination
                                && candidates.contains(dst)
                            {
                                disqual.insert(*dst);
                            }
                            for a in args {
                                let mut tmp = HashSet::new();
                                a.collect_uses(&mut tmp);
                                for id in tmp {
                                    if candidates.contains(&LocalId(id)) {
                                        disqual.insert(LocalId(id));
                                    }
                                }
                            }
                        } else {
                            // In map builtins, only args[0] may be a direct raw map; any map passed as key/value disqualifies.
                            for a in args.iter().skip(1) {
                                let mut tmp = HashSet::new();
                                a.collect_uses(&mut tmp);
                                for id in tmp {
                                    if candidates.contains(&LocalId(id)) {
                                        disqual.insert(LocalId(id));
                                    }
                                }
                            }
                        }
                    }
                    Terminator::Return(Some(op), _) | Terminator::Raise(op, _) => {
                        let mut tmp = HashSet::new();
                        op.collect_uses(&mut tmp);
                        for id in tmp {
                            if candidates.contains(&LocalId(id)) {
                                disqual.insert(LocalId(id));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Propagate disqualification across init-copy edges so both ends agree on AotRawMap vs AotMap.
        loop {
            let mut changed = false;
            for &(dst, src, _, _) in &init_copy_edges {
                if disqual.contains(&dst) && disqual.insert(src) {
                    changed = true;
                }
                if disqual.contains(&src) && disqual.insert(dst) {
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        for loc in candidates {
            if !disqual.contains(&loc) {
                self.direct_raw_map_locals.insert(loc);
            }
        }
    }

    /// Pre-pass (`AOT-MAP-HASH`): fuses sibling `map.get(m, k)` + `map.set(m, k, new_v)` on a
    /// direct `AotRawMap<K, V>` (where `V` is `Copy`) within the same `StructuredNode::Seq`
    /// into a single `AotRawMap::get_mut(k)` lookup (cutting hashing & probing in half on `NG-01`/`NG-02`).
    fn analyze_map_rmw_fusion(&mut self, node: &StructuredNode) {
        if self.direct_raw_map_locals.is_empty() {
            return;
        }
        match node {
            StructuredNode::Seq(items) => {
                // First recurse into children
                for item in items {
                    self.analyze_map_rmw_fusion(item);
                }
                // Scan consecutive `BasicBlock` items in this `Seq` for `get` ... `set` pairs on the same `(map, key)`
                let mut i = 0usize;
                while i < items.len() {
                    if let StructuredNode::BasicBlock(get_bb) = items[i]
                        && let Some(Terminator::Call {
                            callee,
                            args,
                            destination: Place::Local(val_dst),
                            error: None,
                            ..
                        }) = &self.func.blocks[get_bb.0].terminator
                        && matches!(callee.as_str(), "map.get" | "table.get" | "map.get_unwrap")
                        && args.len() >= 2
                        && let Some(m_raw) = operand_exact_local(&args[0])
                        && let Some(k_raw) = operand_exact_local(&args[1])
                    {
                        let m_loc = self.resolve_local(m_raw);
                        let k_loc = self.resolve_local(k_raw);
                        let dst_ty = &self.func.locals[val_dst.0].ty;
                        if self.direct_raw_map_locals.contains(&m_loc)
                            && !matches!(dst_ty, MirType::Option(_))
                            && self.ctx.is_copy_type(dst_ty)
                            && self.local_storage(*val_dst) == LocalStorage::Direct
                        {
                            // Look ahead up to 4 sibling `BasicBlock`s with no other calls and no writes to `m_loc` or `k_loc`
                            let mut j = i + 1;
                            while j < items.len() && j <= i + 4 {
                                let StructuredNode::BasicBlock(cand_bb) = items[j] else {
                                    break;
                                };
                                let blk = &self.func.blocks[cand_bb.0];
                                // Ensure `cand_bb` statements don't touch `m_loc` or write `k_loc`
                                let mut conflict = false;
                                for stmt in &blk.statements {
                                    if let Statement::Assign(place, rval, _) = stmt {
                                        let rl = self.resolve_local(place.root_local());
                                        if rl == m_loc || rl == k_loc {
                                            conflict = true;
                                            break;
                                        }
                                        let mut tmp = HashSet::new();
                                        rval.collect_uses(&mut tmp);
                                        if tmp.iter().any(|&id| self.resolve_local(LocalId(id)) == m_loc) {
                                            conflict = true;
                                            break;
                                        }
                                    }
                                }
                                if conflict {
                                    break;
                                }
                                if let Some(Terminator::Call {
                                    callee: sc,
                                    args: sargs,
                                    error: None,
                                    ..
                                }) = &blk.terminator
                                {
                                    if matches!(sc.as_str(), "map.set" | "table.set")
                                        && sargs.len() >= 3
                                        && operand_exact_local(&sargs[0])
                                            .is_some_and(|l| self.resolve_local(l) == m_loc)
                                        && operand_exact_local(&sargs[1])
                                            .is_some_and(|l| self.resolve_local(l) == k_loc)
                                    {
                                        self.map_rmw_get_plans.insert(
                                            get_bb,
                                            (cand_bb, m_loc, args[1].clone(), *val_dst),
                                        );
                                        self.map_rmw_set_plans.insert(cand_bb, m_loc);
                                        i = j;
                                    }
                                    break;
                                }
                                j += 1;
                            }
                        }
                    }
                    i += 1;
                }
            }
            StructuredNode::If {
                then_branch,
                else_branch,
                ..
            }
            | StructuredNode::IfArrayGetOk {
                then_branch,
                else_branch,
                ..
            } => {
                self.analyze_map_rmw_fusion(then_branch);
                if let Some(eb) = else_branch {
                    self.analyze_map_rmw_fusion(eb);
                }
            }
            StructuredNode::Switch { cases, default, .. } => {
                for (_, c) in cases {
                    self.analyze_map_rmw_fusion(c);
                }
                self.analyze_map_rmw_fusion(default);
            }
            StructuredNode::WhileLoop { body, .. }
            | StructuredNode::Loop { body, .. }
            | StructuredNode::LabeledBlock { body, .. } => {
                self.analyze_map_rmw_fusion(body);
            }
            _ => {}
        }
    }

    /// Pre-pass (`AOT-DEVIRT`): lowers non-escaping `@std/sync.{Queue}` locals (and their local
    /// `offer`/`take`/`len`/`close` method bindings, as in `BENCH-NG-12`) to unboxed `AotQueue<i64>`.
    fn analyze_unboxed_queues(&mut self) {
        let param_set: HashSet<LocalId> = self.func.params.iter().copied().collect();
        let mut queue_inits: Vec<(BlockId, LocalId)> = Vec::new();
        for block in &self.func.blocks {
            if block.is_cleanup {
                continue;
            }
            if let Some(Terminator::Call {
                callee,
                destination: Place::Local(dst),
                ..
            }) = &block.terminator
                && matches!(callee.as_str(), "Queue" | "sync.Queue" | "@std/sync.Queue")
                && !param_set.contains(dst)
                && *dst != self.func.return_local
                && self.local_storage(*dst) == LocalStorage::Direct
            {
                queue_inits.push((block.id, *dst));
            }
        }
        if queue_inits.is_empty() {
            return;
        }

        // Collect single-def copy chains `dst = Copy/Move(src)`
        let mut copy_from: HashMap<LocalId, (BlockId, usize, LocalId)> = HashMap::new();
        for block in &self.func.blocks {
            if block.is_cleanup {
                continue;
            }
            for (sidx, stmt) in block.statements.iter().enumerate() {
                if let Statement::Assign(
                    Place::Local(dst),
                    Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))),
                    _,
                ) = stmt
                    && self.local_def_counts.get(dst).copied().unwrap_or(0) == 1
                {
                    copy_from.insert(*dst, (block.id, sidx, *src));
                }
            }
        }

        for (_init_bb, q_init) in queue_inits {
            // Find any single-def alias of `q_init` (e.g. `_2 = _1`)
            let mut q_aliases: Vec<(LocalId, BlockId, usize)> = Vec::new();
            let mut q_group: HashSet<LocalId> = HashSet::new();
            q_group.insert(q_init);
            for (&dst, &(bb, sidx, src)) in &copy_from {
                if src == q_init && !param_set.contains(&dst) && dst != self.func.return_local {
                    q_aliases.push((dst, bb, sidx));
                    q_group.insert(dst);
                }
            }

            // Collect method-handle extractions `_h = __mir_unknown_get_field(q, "offer"|"take"|...)`
            let mut method_raw: HashMap<LocalId, (BlockId, &'static str)> = HashMap::new();
            let mut method_handles: HashMap<LocalId, &'static str> = HashMap::new();
            let mut method_copy_skips: Vec<(LocalId, BlockId, usize)> = Vec::new();
            let mut valid = true;

            for block in &self.func.blocks {
                if block.is_cleanup {
                    let mut used = HashSet::new();
                    collect_block_used_locals(block, &mut used);
                    if used.iter().any(|u| q_group.contains(u)) {
                        valid = false;
                        break;
                    }
                    continue;
                }
                if let Some(Terminator::Call {
                    callee,
                    args,
                    destination: Place::Local(dst),
                    ..
                }) = &block.terminator
                    && callee == "__mir_unknown_get_field"
                    && args.len() >= 2
                    && let Some(recv) = operand_exact_local(&args[0])
                    && q_group.contains(&recv)
                {
                    let mname: Option<&'static str> = match &args[1] {
                        Operand::Constant(MirConstant::String(s)) => match s.as_str() {
                            "offer" => Some("offer"),
                            "try_offer" => Some("try_offer"),
                            "take" => Some("take"),
                            "try_take" => Some("try_take"),
                            "len" => Some("len"),
                            "close" => Some("close"),
                            _ => None,
                        },
                        _ => None,
                    };
                    if let Some(m) = mname {
                        method_raw.insert(*dst, (block.id, m));
                        method_handles.insert(*dst, m);
                    } else {
                        valid = false;
                        break;
                    }
                }
            }
            if !valid {
                continue;
            }

            // Follow single-def copies of method handles (e.g. `_4 = _3`)
            for (&dst, &(bb, sidx, src)) in &copy_from {
                if let Some(&m) = method_raw.get(&src).map(|(_, m)| m) {
                    if param_set.contains(&dst) || dst == self.func.return_local {
                        valid = false;
                        break;
                    }
                    method_handles.insert(dst, m);
                    method_copy_skips.push((dst, bb, sidx));
                }
            }
            if !valid {
                continue;
            }

            // Verify every use of `q_group` and `method_handles` across the function is a safe
            // `i64` queue operation with zero escape!
            let skip_stmt_set: HashSet<(BlockId, usize)> = q_aliases
                .iter()
                .map(|&(_, bb, sidx)| (bb, sidx))
                .chain(method_copy_skips.iter().map(|&(_, bb, sidx)| (bb, sidx)))
                .collect();

            for block in &self.func.blocks {
                for (sidx, stmt) in block.statements.iter().enumerate() {
                    if skip_stmt_set.contains(&(block.id, sidx)) {
                        continue;
                    }
                    let mut tmp = HashSet::new();
                    if let Statement::Assign(place, rval, _) = stmt {
                        place.collect_uses(&mut tmp);
                        rval.collect_uses(&mut tmp);
                        let rl = place.root_local();
                        if q_group.contains(&rl) || method_handles.contains_key(&rl) {
                            valid = false;
                            break;
                        }
                    }
                    for id in tmp {
                        let l = LocalId(id);
                        if q_group.contains(&l) || method_handles.contains_key(&l) {
                            valid = false;
                            break;
                        }
                    }
                }
                if !valid {
                    break;
                }
                if let Some(term) = &block.terminator {
                    match term {
                        Terminator::Call {
                            callee,
                            args,
                            destination,
                            ..
                        } => {
                            if matches!(callee.as_str(), "Queue" | "sync.Queue" | "@std/sync.Queue")
                                && *destination == Place::Local(q_init)
                            {
                                continue;
                            }
                            if callee == "__mir_unknown_get_field"
                                && args.len() >= 2
                                && operand_exact_local(&args[0])
                                    .is_some_and(|l| q_group.contains(&l))
                            {
                                continue;
                            }
                            if callee == "__mir_unknown_call_method"
                                && args.len() >= 2
                                && operand_exact_local(&args[0])
                                    .is_some_and(|l| q_group.contains(&l))
                            {
                                let ok_method = match &args[1] {
                                    Operand::Constant(MirConstant::String(s)) => match s.as_str() {
                                        "offer" | "try_offer" => {
                                            args.len() == 3
                                                && matches!(
                                                    operand_mir_type(&args[2], self.func),
                                                    MirType::Int
                                                )
                                        }
                                        "take" | "try_take" | "len" | "close" => args.len() == 2,
                                        _ => false,
                                    },
                                    _ => false,
                                };
                                if !ok_method {
                                    valid = false;
                                    break;
                                }
                                continue;
                            }
                            if !self.ctx.fn_metas.contains_key(callee)
                                && let Some(&cl_loc) = self.local_name_to_id.get(callee)
                                && let Some(&m) = method_handles.get(&cl_loc)
                            {
                                let ok_sig = match m {
                                    "offer" | "try_offer" => {
                                        args.len() == 1
                                            && matches!(
                                                operand_mir_type(&args[0], self.func),
                                                MirType::Int
                                            )
                                    }
                                    "take" | "try_take" | "len" | "close" => args.is_empty(),
                                    _ => false,
                                };
                                if !ok_sig {
                                    valid = false;
                                    break;
                                }
                            }
                            for a in args {
                                let mut tmp = HashSet::new();
                                a.collect_uses(&mut tmp);
                                for id in tmp {
                                    let l = LocalId(id);
                                    if q_group.contains(&l) || method_handles.contains_key(&l) {
                                        valid = false;
                                        break;
                                    }
                                }
                            }
                        }
                        Terminator::Return(Some(op), _) | Terminator::Raise(op, _) => {
                            let mut tmp = HashSet::new();
                            op.collect_uses(&mut tmp);
                            for id in tmp {
                                let l = LocalId(id);
                                if q_group.contains(&l) || method_handles.contains_key(&l) {
                                    valid = false;
                                    break;
                                }
                            }
                        }
                        _ => {}
                    }
                }
                if !valid {
                    break;
                }
            }

            if !valid {
                continue;
            }

            // Commit unboxed `AotQueue<i64>` lowering for `q_init`!
            self.unboxed_queue_locals.insert(q_init);
            for (alias_loc, bb, sidx) in q_aliases {
                self.alias_map.insert(alias_loc, q_init);
                self.skip_stmts.insert((bb, sidx));
                self.omitted_locals.insert(alias_loc);
            }
            for (raw_loc, (call_bb, m)) in method_raw {
                self.skip_calls.insert(call_bb);
                self.omitted_locals.insert(raw_loc);
                self.unboxed_queue_methods.insert(raw_loc, (q_init, m));
            }
            for (cp_loc, bb, sidx) in method_copy_skips {
                self.skip_stmts.insert((bb, sidx));
                self.omitted_locals.insert(cp_loc);
                if let Some(&m) = method_handles.get(&cp_loc) {
                    self.unboxed_queue_methods.insert(cp_loc, (q_init, m));
                }
            }

            // Eliminate `AotValue` boxing/drop on `offer()` (dead dst) and `take()` (1-use copy into `Int`)
            for blk in &self.func.blocks {
                if blk.is_cleanup {
                    continue;
                }
                if let Some(Terminator::Call {
                    callee,
                    destination: Place::Local(dst),
                    success,
                    ..
                }) = &blk.terminator
                    && let Some(&cl_loc) = self.local_name_to_id.get(callee)
                    && let Some(&(_, mname)) = self.unboxed_queue_methods.get(&cl_loc)
                {
                    if matches!(mname, "offer" | "try_offer")
                        && self.local_use_counts.get(dst).copied().unwrap_or(0) == 0
                        && self.local_def_counts.get(dst).copied().unwrap_or(0) == 1
                        && *dst != self.func.return_local
                    {
                        self.omitted_locals.insert(*dst);
                    } else if mname == "take"
                        && self.func.locals[dst.0].ty == MirType::Unknown
                        && self.local_use_counts.get(dst).copied().unwrap_or(0) == 1
                        && self.local_def_counts.get(dst).copied().unwrap_or(0) == 1
                        && *dst != self.func.return_local
                        && let Some(succ_blk) = self.func.blocks.get(success.0)
                    {
                        for (sidx, stmt) in succ_blk.statements.iter().enumerate() {
                            if let Statement::Assign(
                                Place::Local(int_dst),
                                Rvalue::Use(
                                    Operand::Copy(Place::Local(src))
                                    | Operand::Move(Place::Local(src)),
                                ),
                                _,
                            ) = stmt
                                && *src == *dst
                                && self.func.locals[int_dst.0].ty == MirType::Int
                                && self.local_storage(*int_dst) == LocalStorage::Direct
                            {
                                self.unboxed_queue_take_redirect.insert(blk.id, *int_dst);
                                self.skip_stmts.insert((succ_blk.id, sidx));
                                self.omitted_locals.insert(*dst);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Pre-pass (`AOT-SSO-REGEX`): detects chained `MirBinaryOp::Concat` trees (including
    /// single-use `tostring(int/float)` leaves in immediately preceding single-predecessor blocks)
    /// and fuses them into a single pre-sized `String` allocation.
    fn analyze_chained_concat_fusion(&mut self) {
        let mut pred_of: Vec<Option<BlockId>> = vec![None; self.func.blocks.len()];
        let mut pred_cnt = vec![0usize; self.func.blocks.len()];
        for blk in &self.func.blocks {
            if let Some(term) = &blk.terminator {
                for succ in term.successors() {
                    if succ < pred_cnt.len() {
                        pred_cnt[succ] += 1;
                        pred_of[succ] = Some(blk.id);
                    }
                }
            }
        }

        // Map single-use `tostring(int/float)` calls `tmp = call tostring(x)` -> (call_bb, arg, arg_ty)
        let mut tostring_defs: HashMap<LocalId, (BlockId, Operand, MirType)> = HashMap::new();
        for blk in &self.func.blocks {
            if blk.is_cleanup || self.fused_tostring_calls.contains_key(&blk.id) {
                continue;
            }
            if let Some(Terminator::Call {
                callee,
                args,
                destination: Place::Local(tmp),
                error: None,
                ..
            }) = &blk.terminator
                && callee == "tostring"
                && args.len() == 1
                && self.local_use_counts.get(tmp).copied().unwrap_or(0) == 1
                && self.local_def_counts.get(tmp).copied().unwrap_or(0) == 1
                && !self.omitted_locals.contains(tmp)
            {
                let aty = operand_mir_type(&args[0], self.func);
                if matches!(aty, MirType::Int | MirType::Float) {
                    tostring_defs.insert(*tmp, (blk.id, args[0].clone(), aty));
                }
            }
        }

        for blk in &self.func.blocks {
            if blk.is_cleanup {
                continue;
            }
            // Collect linear single-predecessor ancestor chain of `blk` (up to 8 hops) for tostring lookup
            let mut linear_ancestors: HashSet<BlockId> = HashSet::new();
            let mut cur = blk.id;
            for _ in 0..8 {
                if pred_cnt[cur.0] == 1
                    && let Some(p) = pred_of[cur.0]
                {
                    if linear_ancestors.insert(p) {
                        cur = p;
                        continue;
                    }
                }
                break;
            }

            // Track local concat definitions inside this block: dst -> (sidx, lhs, rhs)
            let mut concat_defs: HashMap<LocalId, (usize, Operand, Operand)> = HashMap::new();
            for (sidx, stmt) in blk.statements.iter().enumerate() {
                if self.skip_stmts.contains(&(blk.id, sidx)) {
                    continue;
                }
                if let Statement::Assign(
                    Place::Local(dst),
                    Rvalue::BinaryOp(MirBinaryOp::Concat, lhs, rhs),
                    _,
                ) = stmt
                    && self.local_storage(*dst) == LocalStorage::Direct
                    && !self.omitted_locals.contains(dst)
                {
                    // Skip in-place `s = s .. rhs`
                    let is_inplace = matches!(
                        lhs,
                        Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)) if l == dst
                    );
                    if !is_inplace {
                        concat_defs.insert(*dst, (sidx, lhs.clone(), rhs.clone()));
                    }
                }
            }
            if concat_defs.is_empty() {
                continue;
            }

            // Walk statements from bottom to top so outer concat roots consume inner single-use concats
            let mut consumed_stmts: HashSet<usize> = HashSet::new();
            for sidx in (0..blk.statements.len()).rev() {
                if consumed_stmts.contains(&sidx) || self.skip_stmts.contains(&(blk.id, sidx)) {
                    continue;
                }
                let Statement::Assign(
                    Place::Local(_root_dst),
                    Rvalue::BinaryOp(MirBinaryOp::Concat, lhs, rhs),
                    _,
                ) = &blk.statements[sidx]
                else {
                    continue;
                };
                if !concat_defs.contains_key(_root_dst) {
                    continue;
                }

                // Expand concat tree into flat pieces
                let mut pieces: Vec<ConcatPiece> = Vec::new();
                let mut inner_stmt_skips: Vec<(LocalId, usize)> = Vec::new();
                let mut inner_tostring_skips: Vec<(LocalId, BlockId)> = Vec::new();
                let mut stack: Vec<Operand> = vec![rhs.clone(), lhs.clone()];
                let mut expanded_any = false;

                while let Some(op) = stack.pop() {
                    if let Some(loc) = operand_exact_local(&op)
                        && self.local_use_counts.get(&loc).copied().unwrap_or(0) == 1
                        && self.local_def_counts.get(&loc).copied().unwrap_or(0) == 1
                    {
                        if let Some((csidx, clhs, crhs)) = concat_defs.get(&loc)
                            && *csidx < sidx
                            && !consumed_stmts.contains(csidx)
                        {
                            inner_stmt_skips.push((loc, *csidx));
                            stack.push(crhs.clone());
                            stack.push(clhs.clone());
                            expanded_any = true;
                            continue;
                        }
                        if let Some((tbb, targ, taty)) = tostring_defs.get(&loc)
                            && linear_ancestors.contains(tbb)
                            && !self.skip_calls.contains(tbb)
                        {
                            // Ensure `targ` local is not mutated between `tbb` and `blk`
                            let safe_arg = match operand_exact_local(targ) {
                                Some(al) => self.local_def_counts.get(&al).copied().unwrap_or(0) <= 1,
                                None => true,
                            };
                            if safe_arg {
                                inner_tostring_skips.push((loc, *tbb));
                                match taty {
                                    MirType::Int => pieces.push(ConcatPiece::Int(targ.clone())),
                                    MirType::Float => pieces.push(ConcatPiece::Float(targ.clone())),
                                    _ => pieces.push(ConcatPiece::Str(op)),
                                }
                                expanded_any = true;
                                continue;
                            }
                        }
                    }
                    pieces.push(ConcatPiece::Str(op));
                }

                if expanded_any && pieces.len() >= 2 {
                    for (loc, csidx) in inner_stmt_skips {
                        consumed_stmts.insert(csidx);
                        self.skip_stmts.insert((blk.id, csidx));
                        self.omitted_locals.insert(loc);
                    }
                    for (loc, tbb) in inner_tostring_skips {
                        self.skip_calls.insert(tbb);
                        self.omitted_locals.insert(loc);
                    }
                    self.fused_concat_chains.insert((blk.id, sidx), pieces);
                }
            }
        }
    }

    /// Pre-pass (`AOT-SSO-REGEX`): detects read-only `MirType::String` locals that are ONLY
    /// assigned from non-mutated `String` locals or string constants (e.g. `local raw = payload_a;
    /// if cond then raw = payload_b end` in `BENCH-NG-13`) and lowers them to `&str` (`0` clones!).
    fn analyze_borrowed_select_str_locals(&mut self) {
        let param_set: HashSet<LocalId> = self.func.params.iter().copied().collect();
        let fused_mut_targets: HashSet<LocalId> = self
            .fused_tostring_calls
            .values()
            .map(|(dst, _, _)| self.resolve_local(*dst))
            .collect();
        // Identify immutable source `String` locals (defined at most once, outside loops or params)
        let mut loop_blocks: HashSet<BlockId> = HashSet::new();
        for blk in &self.func.blocks {
            if let Some(term) = &blk.terminator {
                for succ in term.successors() {
                    if succ <= blk.id.0 {
                        // Back-edge header..latch
                        for b in succ..=blk.id.0 {
                            loop_blocks.insert(BlockId(b));
                        }
                    }
                }
            }
        }
        let mut stable_str_sources: HashSet<LocalId> = HashSet::new();
        for decl in &self.func.locals {
            if decl.ty != MirType::String || self.local_storage(decl.id) != LocalStorage::Direct {
                continue;
            }
            let defs = self.local_def_counts.get(&decl.id).copied().unwrap_or(0);
            if param_set.contains(&decl.id) && defs == 0 {
                stable_str_sources.insert(decl.id);
            } else if defs == 1 {
                // Defined once outside any loop
                let mut def_in_loop = false;
                for blk in &self.func.blocks {
                    if !loop_blocks.contains(&blk.id) {
                        continue;
                    }
                    for stmt in &blk.statements {
                        if let Statement::Assign(Place::Local(d), _, _) = stmt
                            && *d == decl.id
                        {
                            def_in_loop = true;
                        }
                    }
                    if let Some(Terminator::Call {
                        destination: Place::Local(d),
                        ..
                    }) = &blk.terminator
                        && *d == decl.id
                    {
                        def_in_loop = true;
                    }
                }
                if !def_in_loop {
                    stable_str_sources.insert(decl.id);
                }
            }
        }

        for decl in &self.func.locals {
            let loc = decl.id;
            if decl.ty != MirType::String
                || param_set.contains(&loc)
                || loc == self.func.return_local
                || self.omitted_locals.contains(&loc)
                || fused_mut_targets.contains(&loc)
                || self.local_storage(loc) != LocalStorage::Direct
                || self.local_def_counts.get(&loc).copied().unwrap_or(0) < 2
            {
                continue;
            }
            let mut ok = true;
            for blk in &self.func.blocks {
                if blk.is_cleanup {
                    let mut used = HashSet::new();
                    collect_block_used_locals(blk, &mut used);
                    if used.iter().any(|&u| self.resolve_local(u) == loc) {
                        ok = false;
                        break;
                    }
                    continue;
                }
                for (sidx, stmt) in blk.statements.iter().enumerate() {
                    if self.skip_stmts.contains(&(blk.id, sidx)) {
                        continue;
                    }
                    if let Statement::Assign(place, rval, _) = stmt {
                        if *place == Place::Local(loc) {
                            let valid_rhs = match rval {
                                Rvalue::Use(Operand::Constant(MirConstant::String(_))) => true,
                                Rvalue::Use(
                                    Operand::Copy(Place::Local(src))
                                    | Operand::Move(Place::Local(src)),
                                ) => stable_str_sources.contains(&self.resolve_local(*src)),
                                _ => false,
                            };
                            if !valid_rhs {
                                ok = false;
                                break;
                            }
                        } else if let Rvalue::Aggregate(_, _) | Rvalue::Closure { .. } = rval {
                            let mut tmp = HashSet::new();
                            rval.collect_uses(&mut tmp);
                            if tmp.iter().any(|&id| self.resolve_local(LocalId(id)) == loc) {
                                ok = false;
                                break;
                            }
                        }
                    }
                }
                if !ok {
                    break;
                }
                if let Some(term) = &blk.terminator {
                    match term {
                        Terminator::Call {
                            destination: Place::Local(d),
                            ..
                        } if *d == loc => {
                            ok = false;
                            break;
                        }
                        Terminator::Return(Some(op), _) => {
                            if operand_exact_local(op).is_some_and(|l| self.resolve_local(l) == loc)
                            {
                                ok = false;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
            }
            if ok {
                self.borrowed_select_str_locals.insert(loc);
                self.scoped_ref_locals.insert(loc);
            }
        }
    }

    /// Pre-pass (`AOT-SSO-REGEX`): lowers `inner = string.sub(s, start, end)` (when `s` is a
    /// long-lived `&str` local in `borrowed_select_str_locals` or a `RefStr` param) to
    /// `alloc::borrow::Cow<'_, str>` with zero heap allocation on valid UTF-8 slices!
    fn analyze_cow_sub_str_locals(&mut self) {
        let param_set: HashSet<LocalId> = self.func.params.iter().copied().collect();
        let fused_mut_targets: HashSet<LocalId> = self
            .fused_tostring_calls
            .values()
            .map(|(dst, _, _)| self.resolve_local(*dst))
            .collect();
        let mut sub_defs: Vec<(LocalId, Operand)> = Vec::new();
        for blk in &self.func.blocks {
            if blk.is_cleanup {
                continue;
            }
            if let Some(Terminator::Call {
                callee,
                args,
                destination: Place::Local(dst),
                ..
            }) = &blk.terminator
                && callee == "string.sub"
                && args.len() >= 2
                && self.func.locals[dst.0].ty == MirType::String
                && !param_set.contains(dst)
                && *dst != self.func.return_local
                && !self.omitted_locals.contains(dst)
                && !fused_mut_targets.contains(dst)
                && self.local_storage(*dst) == LocalStorage::Direct
                && self.local_def_counts.get(dst).copied().unwrap_or(0) == 1
            {
                let is_long_lived_src = match &args[0] {
                    Operand::Constant(MirConstant::String(_)) => true,
                    _ => operand_exact_local(&args[0]).is_some_and(|l| {
                        let r = self.resolve_local(l);
                        self.borrowed_select_str_locals.contains(&r)
                            || matches!(self.is_param(r), Some((_, ParamMode::RefStr)))
                    }),
                };
                if is_long_lived_src {
                    sub_defs.push((*dst, args[0].clone()));
                }
            }
        }
        if sub_defs.is_empty() {
            return;
        }

        // Follow single-def pure-init-temp copies `inner = Move(sub_tmp)`
        let mut copy_to_root: HashMap<LocalId, (LocalId, BlockId, usize)> = HashMap::new();
        let cand_set: HashSet<LocalId> = sub_defs.iter().map(|(l, _)| *l).collect();
        for blk in &self.func.blocks {
            if blk.is_cleanup {
                continue;
            }
            for (sidx, stmt) in blk.statements.iter().enumerate() {
                if self.skip_stmts.contains(&(blk.id, sidx)) {
                    continue;
                }
                if let Statement::Assign(
                    Place::Local(dst),
                    Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))),
                    _,
                ) = stmt
                    && cand_set.contains(src)
                    && !param_set.contains(dst)
                    && *dst != self.func.return_local
                    && self.local_storage(*dst) == LocalStorage::Direct
                    && self.local_def_counts.get(dst).copied().unwrap_or(0) == 1
                {
                    copy_to_root.insert(*dst, (*src, blk.id, sidx));
                }
            }
        }
        let copy_skips: HashSet<(BlockId, usize)> = copy_to_root
            .values()
            .map(|&(_, bb, sidx)| (bb, sidx))
            .collect();
        let resolve_c = |loc: LocalId| -> LocalId {
            let r = self.resolve_local(loc);
            copy_to_root.get(&r).map(|&(root, _, _)| root).unwrap_or(r)
        };

        let mut disqual: HashSet<LocalId> = HashSet::new();
        for blk in &self.func.blocks {
            if blk.is_cleanup {
                let mut used = HashSet::new();
                collect_block_used_locals(blk, &mut used);
                for u in used {
                    let r = resolve_c(u);
                    if cand_set.contains(&r) {
                        disqual.insert(r);
                    }
                }
                continue;
            }
            for (sidx, stmt) in blk.statements.iter().enumerate() {
                if self.skip_stmts.contains(&(blk.id, sidx)) || copy_skips.contains(&(blk.id, sidx))
                {
                    continue;
                }
                if let Statement::Assign(place, rval, _) = stmt {
                    if let Rvalue::UnaryOp(MirUnaryOp::Len, _)
                    | Rvalue::BinaryOp(MirBinaryOp::Eq | MirBinaryOp::NotEq, _, _) = rval
                    {
                        continue;
                    }
                    let mut tmp = HashSet::new();
                    place.collect_uses(&mut tmp);
                    rval.collect_uses(&mut tmp);
                    let rl = resolve_c(place.root_local());
                    if cand_set.contains(&rl) {
                        disqual.insert(rl);
                    }
                    for id in tmp {
                        let r = resolve_c(LocalId(id));
                        if cand_set.contains(&r) {
                            disqual.insert(r);
                        }
                    }
                }
            }
            if let Some(term) = &blk.terminator {
                match term {
                    Terminator::Call {
                        callee,
                        args,
                        destination,
                        ..
                    } => {
                        if callee == "string.sub"
                            && matches!(destination, Place::Local(d) if cand_set.contains(d))
                        {
                            continue;
                        }
                        let str_ref_builtin = matches!(
                            callee.as_str(),
                            "string.split" | "string.len" | "string.sub" | "string.gsub" | "string.find"
                        );
                        if !str_ref_builtin {
                            for a in args {
                                let mut tmp = HashSet::new();
                                a.collect_uses(&mut tmp);
                                for id in tmp {
                                    let r = resolve_c(LocalId(id));
                                    if cand_set.contains(&r) {
                                        disqual.insert(r);
                                    }
                                }
                            }
                        }
                    }
                    Terminator::Return(Some(op), _) | Terminator::Raise(op, _) => {
                        let mut tmp = HashSet::new();
                        op.collect_uses(&mut tmp);
                        for id in tmp {
                            let r = resolve_c(LocalId(id));
                            if cand_set.contains(&r) {
                                disqual.insert(r);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        for (c, _) in sub_defs {
            if !disqual.contains(&c) {
                self.cow_str_locals.insert(c);
            }
        }
        for (dst, (root, bb, sidx)) in copy_to_root {
            if self.cow_str_locals.contains(&root) {
                self.alias_map.insert(dst, root);
                self.skip_stmts.insert((bb, sidx));
                self.omitted_locals.insert(dst);
            }
        }
    }

    /// Pre-pass (`AOT-SSO-REGEX`): detects `ch = string.sub(s, idx, idx)` locals that are only
    /// ever compared against 1-byte ASCII string constants (`ch == "1"`, etc., as in `BENCH-NG-13`)
    /// and lowers them to `Option<u8>` with zero `String` allocations.
    fn analyze_single_byte_sub_locals(&mut self) {
        let param_set: HashSet<LocalId> = self.func.params.iter().copied().collect();
        let mut candidates: HashSet<LocalId> = HashSet::new();
        for blk in &self.func.blocks {
            if blk.is_cleanup {
                continue;
            }
            if let Some(Terminator::Call {
                callee,
                args,
                destination: Place::Local(dst),
                ..
            }) = &blk.terminator
                && callee == "string.sub"
                && args.len() >= 3
                && !param_set.contains(dst)
                && *dst != self.func.return_local
                && !self.omitted_locals.contains(dst)
                && self.local_storage(*dst) == LocalStorage::Direct
                && self.local_def_counts.get(dst).copied().unwrap_or(0) == 1
            {
                let same_idx = match (&args[1], &args[2]) {
                    (Operand::Constant(MirConstant::Int(a)), Operand::Constant(MirConstant::Int(b))) => {
                        a == b
                    }
                    _ => {
                        let l1 = operand_exact_local(&args[1]).map(|l| self.resolve_local(l));
                        let l2 = operand_exact_local(&args[2]).map(|l| self.resolve_local(l));
                        l1.is_some() && l1 == l2
                    }
                };
                if same_idx {
                    candidates.insert(*dst);
                }
            }
        }
        if candidates.is_empty() {
            return;
        }

        // Follow single-def pure-init-temp copies `ch = Move/Copy(sub_tmp)` (e.g. `_82 = Move(_81)`)
        // so both the temp and the named `ch` local collapse into a single `Option<u8>` local.
        let mut copy_to_root: HashMap<LocalId, (LocalId, BlockId, usize)> = HashMap::new();
        for blk in &self.func.blocks {
            if blk.is_cleanup {
                continue;
            }
            for (sidx, stmt) in blk.statements.iter().enumerate() {
                if self.skip_stmts.contains(&(blk.id, sidx)) {
                    continue;
                }
                if let Statement::Assign(
                    Place::Local(dst),
                    Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))),
                    _,
                ) = stmt
                    && candidates.contains(src)
                    && !param_set.contains(dst)
                    && *dst != self.func.return_local
                    && self.local_storage(*dst) == LocalStorage::Direct
                    && self.local_def_counts.get(dst).copied().unwrap_or(0) == 1
                {
                    copy_to_root.insert(*dst, (*src, blk.id, sidx));
                }
            }
        }
        let copy_skip_stmts: HashSet<(BlockId, usize)> = copy_to_root
            .values()
            .map(|&(_, bb, sidx)| (bb, sidx))
            .collect();
        let resolve_cand = |loc: LocalId| -> LocalId {
            let r = self.resolve_local(loc);
            if let Some(&(root, _, _)) = copy_to_root.get(&r) {
                root
            } else {
                r
            }
        };

        // Verify every use of a candidate (including through copy_to_root / alias_map) is ONLY
        // `Eq`/`NotEq` against a 1-byte ASCII string constant.
        let mut disqual: HashSet<LocalId> = HashSet::new();
        let is_ascii_1byte = |op: &Operand| -> bool {
            matches!(op, Operand::Constant(MirConstant::String(s)) if s.len() == 1 && s.is_ascii())
        };
        for blk in &self.func.blocks {
            for (sidx, stmt) in blk.statements.iter().enumerate() {
                if self.skip_stmts.contains(&(blk.id, sidx))
                    || copy_skip_stmts.contains(&(blk.id, sidx))
                {
                    continue;
                }
                if let Statement::Assign(_, rval, _) = stmt {
                    if let Rvalue::BinaryOp(MirBinaryOp::Eq | MirBinaryOp::NotEq, lhs, rhs) = rval {
                        let l_res = operand_exact_local(lhs).map(&resolve_cand);
                        let r_res = operand_exact_local(rhs).map(&resolve_cand);
                        if let Some(ll) = l_res
                            && candidates.contains(&ll)
                        {
                            if !is_ascii_1byte(rhs) {
                                disqual.insert(ll);
                            }
                            continue;
                        }
                        if let Some(rl) = r_res
                            && candidates.contains(&rl)
                        {
                            if !is_ascii_1byte(lhs) {
                                disqual.insert(rl);
                            }
                            continue;
                        }
                    }
                    let mut tmp = HashSet::new();
                    rval.collect_uses(&mut tmp);
                    for id in tmp {
                        let r = resolve_cand(LocalId(id));
                        if candidates.contains(&r) {
                            disqual.insert(r);
                        }
                    }
                }
            }
            if let Some(term) = &blk.terminator {
                match term {
                    Terminator::Call { args, .. } => {
                        for a in args {
                            let mut tmp = HashSet::new();
                            a.collect_uses(&mut tmp);
                            for id in tmp {
                                let r = resolve_cand(LocalId(id));
                                if candidates.contains(&r) {
                                    disqual.insert(r);
                                }
                            }
                        }
                    }
                    Terminator::Return(Some(op), _) | Terminator::Raise(op, _) => {
                        let mut tmp = HashSet::new();
                        op.collect_uses(&mut tmp);
                        for id in tmp {
                            let r = resolve_cand(LocalId(id));
                            if candidates.contains(&r) {
                                disqual.insert(r);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        for c in candidates {
            if !disqual.contains(&c) {
                self.byte_sub_locals.insert(c);
            }
        }
        for (dst, (root, bb, sidx)) in copy_to_root {
            if self.byte_sub_locals.contains(&root) {
                self.alias_map.insert(dst, root);
                self.skip_stmts.insert((bb, sidx));
                self.omitted_locals.insert(dst);
            }
        }
    }

    /// Pre-pass (`AOT-SSO-REGEX`): lowers non-escaping `Array<string>` locals created strictly by
    /// `string.split(src, sep)` and read only via borrowed `IfArrayGetOk` / `len()` to zero-alloc
    /// `__LustSplitSpans<'a>` referencing `src` directly (0 byte copies, 0 `from_utf8`, 0 heap allocs!).
    fn analyze_small_str_split_arrays(&mut self, tree: &StructuredNode) {
        let param_set: HashSet<LocalId> = self.func.params.iter().copied().collect();
        let mut split_inits: HashMap<LocalId, Operand> = HashMap::new();
        for blk in &self.func.blocks {
            if blk.is_cleanup {
                continue;
            }
            if let Some(Terminator::Call {
                callee,
                args,
                destination: Place::Local(dst),
                ..
            }) = &blk.terminator
                && callee == "string.split"
                && args.len() >= 2
                && matches!(&self.func.locals[dst.0].ty, MirType::Array(inner) if **inner == MirType::String)
                && !param_set.contains(dst)
                && *dst != self.func.return_local
                && self.local_storage(*dst) == LocalStorage::Direct
                && self.local_def_counts.get(dst).copied().unwrap_or(0) == 1
            {
                // Verify source string is stable (single-def local, borrowed &str, Cow, or constant)
                let src_ok = match &args[0] {
                    Operand::Constant(MirConstant::String(_)) => true,
                    _ => operand_exact_local(&args[0]).is_some_and(|l| {
                        let r = self.resolve_local(l);
                        self.scoped_ref_locals.contains(&r)
                            || self.cow_str_locals.contains(&r)
                            || (self.local_def_counts.get(&r).copied().unwrap_or(0) <= 1
                                && self.local_storage(r) == LocalStorage::Direct)
                    }),
                };
                if src_ok {
                    split_inits.insert(*dst, args[0].clone());
                }
            }
        }
        if split_inits.is_empty() {
            return;
        }

        // Propagate across pure-init-temp copies `arr = Move/Copy(split_tmp)` so we can fold them too!
        let mut group: HashMap<LocalId, Operand> = split_inits.clone();
        let mut copy_to_root: HashMap<LocalId, (LocalId, BlockId, usize)> = HashMap::new();
        for blk in &self.func.blocks {
            if blk.is_cleanup {
                continue;
            }
            for (sidx, stmt) in blk.statements.iter().enumerate() {
                if self.skip_stmts.contains(&(blk.id, sidx)) {
                    continue;
                }
                if let Statement::Assign(
                    Place::Local(dst),
                    Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))),
                    _,
                ) = stmt
                    && let Some(src_str_op) = split_inits.get(src).cloned()
                    && matches!(&self.func.locals[dst.0].ty, MirType::Array(inner) if **inner == MirType::String)
                    && !param_set.contains(dst)
                    && *dst != self.func.return_local
                    && self.local_storage(*dst) == LocalStorage::Direct
                    && self.local_def_counts.get(dst).copied().unwrap_or(0) == 1
                {
                    group.insert(*dst, src_str_op);
                    copy_to_root.insert(*dst, (*src, blk.id, sidx));
                }
            }
        }

        // Collect all `arr_local` used in `IfArrayGetOk` where `binding_local` is borrowed `&str`
        let mut allowed_if_get_arrs: HashSet<LocalId> = HashSet::new();
        let mut disqual: HashSet<LocalId> = HashSet::new();
        collect_if_array_get_info(tree, self, &mut allowed_if_get_arrs, &mut disqual);

        // Verify all remaining uses in blocks are only `IfArrayGetOk` (skipped), `Len`, or `array.len`
        for blk in &self.func.blocks {
            if blk.is_cleanup {
                let mut used = HashSet::new();
                collect_block_used_locals(blk, &mut used);
                for u in used {
                    if group.contains_key(&u) {
                        disqual.insert(u);
                    }
                }
                continue;
            }
            for (sidx, stmt) in blk.statements.iter().enumerate() {
                if self.skip_stmts.contains(&(blk.id, sidx)) {
                    continue;
                }
                if let Statement::Assign(place, rval, _) = stmt {
                    if let Place::Local(dst) = place
                        && group.contains_key(dst)
                        && let Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))) = rval
                        && group.contains_key(src)
                    {
                        continue;
                    }
                    if let Rvalue::UnaryOp(MirUnaryOp::Len, op) = rval
                        && operand_exact_local(op)
                            .is_some_and(|l| group.contains_key(&self.resolve_local(l)))
                    {
                        continue;
                    }
                    let mut tmp = HashSet::new();
                    place.collect_uses(&mut tmp);
                    rval.collect_uses(&mut tmp);
                    let rl = self.resolve_local(place.root_local());
                    if group.contains_key(&rl) {
                        disqual.insert(rl);
                    }
                    for id in tmp {
                        let r = self.resolve_local(LocalId(id));
                        if group.contains_key(&r) {
                            disqual.insert(r);
                        }
                    }
                }
            }
            if let Some(term) = &blk.terminator {
                match term {
                    Terminator::Call {
                        callee,
                        args,
                        destination,
                        ..
                    } => {
                        if callee == "string.split"
                            && matches!(destination, Place::Local(d) if group.contains_key(d))
                        {
                            continue;
                        }
                        if matches!(callee.as_str(), "array.len" | "table.len")
                            && args.len() == 1
                            && operand_exact_local(&args[0])
                                .is_some_and(|l| group.contains_key(&self.resolve_local(l)))
                        {
                            continue;
                        }
                        if let Place::Local(d) = destination
                            && group.contains_key(d)
                        {
                            disqual.insert(*d);
                        }
                        for a in args {
                            let mut tmp = HashSet::new();
                            a.collect_uses(&mut tmp);
                            for id in tmp {
                                let r = self.resolve_local(LocalId(id));
                                if group.contains_key(&r) {
                                    disqual.insert(r);
                                }
                            }
                        }
                    }
                    Terminator::Return(Some(op), _) | Terminator::Raise(op, _) => {
                        let mut tmp = HashSet::new();
                        op.collect_uses(&mut tmp);
                        for id in tmp {
                            let r = self.resolve_local(LocalId(id));
                            if group.contains_key(&r) {
                                disqual.insert(r);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Propagate disqualification across copy edges
        loop {
            let mut changed = false;
            for (&dst, &(src, _, _)) in &copy_to_root {
                if disqual.contains(&dst) && disqual.insert(src) {
                    changed = true;
                }
                if disqual.contains(&src) && disqual.insert(dst) {
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        for (loc, src_str_op) in group {
            if !disqual.contains(&loc) {
                self.small_str_array_locals.insert(loc, src_str_op);
            }
        }
        // Fold pure-init-temp copy `arr = Move(split_tmp)` to eliminate the extra struct copy!
        for (dst, (root, bb, sidx)) in copy_to_root {
            if self.small_str_array_locals.contains_key(&root) {
                self.alias_map.insert(dst, root);
                self.skip_stmts.insert((bb, sidx));
                self.omitted_locals.insert(dst);
            }
        }

        // Promote `small_str_array_locals` that are NEVER passed to `len`/`array.len`, ONLY read
        // at constant indices `0` or `1`, and split from a non-mutated borrowed/const `&str`
        // into `Copy` `(Option<&str>, Option<&str>)` (`split_pair_locals`) — zero `Drop`, zero
        // UTF-8 boundary re-checks, and stops scanning after the 2nd element!
        for blk in &self.func.blocks {
            if blk.is_cleanup {
                continue;
            }
            for stmt in &blk.statements {
                if let Statement::Assign(
                    Place::Local(d),
                    Rvalue::Use(Operand::Constant(MirConstant::Int(v))),
                    _,
                ) = stmt
                    && self.local_def_counts.get(d).copied().unwrap_or(0) == 1
                {
                    self.const_int_locals.insert(*d, *v);
                }
            }
        }
        let mut uses_len_or_bad_idx: HashSet<LocalId> = HashSet::new();
        for blk in &self.func.blocks {
            if blk.is_cleanup {
                continue;
            }
            for (sidx, stmt) in blk.statements.iter().enumerate() {
                if self.skip_stmts.contains(&(blk.id, sidx)) {
                    continue;
                }
                if let Statement::Assign(_, Rvalue::UnaryOp(MirUnaryOp::Len, op), _) = stmt
                    && let Some(l) = operand_exact_local(op)
                {
                    uses_len_or_bad_idx.insert(self.resolve_local(l));
                }
            }
            if let Some(Terminator::Call { callee, args, .. }) = &blk.terminator
                && matches!(callee.as_str(), "array.len" | "table.len")
                && let Some(arg0) = args.first()
                && let Some(l) = operand_exact_local(arg0)
            {
                uses_len_or_bad_idx.insert(self.resolve_local(l));
            }
        }
        fn check_split_pair_indices(
            node: &StructuredNode,
            fe: &FnEmitter<'_, '_>,
            bad: &mut HashSet<LocalId>,
        ) {
            match node {
                StructuredNode::IfArrayGetOk {
                    arr_local,
                    idx_local,
                    then_branch,
                    else_branch,
                    ..
                } => {
                    let r = fe.resolve_local(*arr_local);
                    if fe.small_str_array_locals.contains_key(&r) {
                        let cidx = fe.const_int_locals.get(&fe.resolve_local(*idx_local)).copied();
                        if !matches!(cidx, Some(0 | 1)) {
                            bad.insert(r);
                        }
                    }
                    check_split_pair_indices(then_branch, fe, bad);
                    if let Some(eb) = else_branch {
                        check_split_pair_indices(eb, fe, bad);
                    }
                }
                StructuredNode::Seq(items) => {
                    for it in items {
                        check_split_pair_indices(it, fe, bad);
                    }
                }
                StructuredNode::If {
                    then_branch,
                    else_branch,
                    ..
                } => {
                    check_split_pair_indices(then_branch, fe, bad);
                    if let Some(eb) = else_branch {
                        check_split_pair_indices(eb, fe, bad);
                    }
                }
                StructuredNode::Switch { cases, default, .. } => {
                    for (_, c) in cases {
                        check_split_pair_indices(c, fe, bad);
                    }
                    check_split_pair_indices(default, fe, bad);
                }
                StructuredNode::Loop { body, .. } => check_split_pair_indices(body, fe, bad),
                _ => {}
            }
        }
        check_split_pair_indices(tree, self, &mut uses_len_or_bad_idx);
        let pair_candidates: Vec<LocalId> = self
            .small_str_array_locals
            .iter()
            .filter_map(|(&loc, src_op)| {
                if self.omitted_locals.contains(&loc) || uses_len_or_bad_idx.contains(&loc) {
                    return None;
                }
                let src_borrow_safe = match src_op {
                    Operand::Constant(MirConstant::String(_)) => true,
                    _ => operand_exact_local(src_op).is_some_and(|l| {
                        let r = self.resolve_local(l);
                        self.borrowed_array_str_locals.contains(&r)
                            || self.scoped_ref_locals.contains(&r)
                    }),
                };
                if src_borrow_safe { Some(loc) } else { None }
            })
            .collect();
        for loc in pair_candidates {
            self.split_pair_locals.insert(loc);
        }

        // Fuse `arr = string.split(src, non_empty_const); idx = 0; len = array.len(arr); while idx < len { if arr[idx] is Ok(elem) { body } idx += 1 }`
        // directly into `for elem in src.split(sep) { body }` — zero buffer, zero `Drop`, zero UTF-8 re-checks!
        let mut split_call_info: HashMap<LocalId, (BlockId, Operand, String)> = HashMap::new();
        for blk in &self.func.blocks {
            if blk.is_cleanup {
                continue;
            }
            if let Some(Terminator::Call {
                callee,
                args,
                destination: Place::Local(dst),
                ..
            }) = &blk.terminator
                && callee == "string.split"
                && args.len() >= 2
                && let Operand::Constant(MirConstant::String(sep_lit)) = &args[1]
                && !sep_lit.is_empty()
            {
                let r = self.resolve_local(*dst);
                if self.small_str_array_locals.contains_key(&r) && !self.split_pair_locals.contains(&r)
                {
                    split_call_info.insert(r, (blk.id, args[0].clone(), sep_lit.clone()));
                }
            }
        }
        if !split_call_info.is_empty() {
            // Map `len_loc -> (arr_id, len_call_bb)` for `len = array.len(arr)` (plus single-def copy)
            let mut arr_len_locals: HashMap<LocalId, (LocalId, BlockId)> = HashMap::new();
            for blk in &self.func.blocks {
                if blk.is_cleanup {
                    continue;
                }
                if let Some(Terminator::Call {
                    callee,
                    args,
                    destination: Place::Local(dst),
                    ..
                }) = &blk.terminator
                    && matches!(callee.as_str(), "array.len" | "table.len")
                    && args.len() == 1
                    && let Some(al) = operand_exact_local(&args[0])
                {
                    let ar = self.resolve_local(al);
                    if split_call_info.contains_key(&ar) {
                        arr_len_locals.insert(*dst, (ar, blk.id));
                    }
                }
            }
            let len_copies: Vec<(LocalId, (LocalId, BlockId))> = self
                .func
                .blocks
                .iter()
                .filter(|b| !b.is_cleanup)
                .flat_map(|b| b.statements.iter())
                .filter_map(|s| {
                    if let Statement::Assign(
                        Place::Local(dst),
                        Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))),
                        _,
                    ) = s
                        && let Some(&info) = arr_len_locals.get(src)
                        && self.local_def_counts.get(dst).copied().unwrap_or(0) == 1
                    {
                        Some((*dst, info))
                    } else {
                        None
                    }
                })
                .collect();
            for (d, info) in len_copies {
                arr_len_locals.insert(d, info);
            }

            // Walk structured `WhileLoop`s to find fusable `0..arr.len()` split loops
            fn find_split_iter_loops(
                node: &StructuredNode,
                fe: &FnEmitter<'_, '_>,
                split_call_info: &HashMap<LocalId, (BlockId, Operand, String)>,
                arr_len_locals: &HashMap<LocalId, (LocalId, BlockId)>,
                found: &mut Vec<(BlockId, LocalId, BlockId, BlockId, Operand, String, LocalId)>,
            ) {
                match node {
                    StructuredNode::WhileLoop {
                        label,
                        header_block,
                        cond_op,
                        negate_cond: false,
                        body,
                    } => {
                        if let Some(cond_loc) = operand_exact_local(cond_op) {
                            let hblk = &fe.func.blocks[header_block.0];
                            // Header must compute `cond = (idx < len)` with no other side effects
                            let mut lt_pair = None;
                            let mut header_ok = true;
                            for (sidx, st) in hblk.statements.iter().enumerate() {
                                if fe.skip_stmts.contains(&(*header_block, sidx)) {
                                    continue;
                                }
                                match st {
                                    Statement::StorageLive(_, _) | Statement::StorageDead(_, _) => {}
                                    Statement::Assign(
                                        Place::Local(d),
                                        Rvalue::BinaryOp(MirBinaryOp::Lt, l, r),
                                        _,
                                    ) if *d == cond_loc => {
                                        if let (Some(il), Some(ll)) =
                                            (operand_exact_local(l), operand_exact_local(r))
                                        {
                                            lt_pair = Some((fe.resolve_local(il), fe.resolve_local(ll)));
                                        }
                                    }
                                    _ => {
                                        header_ok = false;
                                        break;
                                    }
                                }
                            }
                            if header_ok
                                && let Some((idx_loc, len_loc)) = lt_pair
                                && let Some(&(arr_id, len_bb)) = arr_len_locals.get(&len_loc)
                                && let Some((split_bb, src_op, sep_lit)) = split_call_info.get(&arr_id)
                                && let StructuredNode::Seq(items) = body.as_ref()
                                && items.len() == 2
                                && let StructuredNode::IfArrayGetOk {
                                    header_block: if_hdr,
                                    arr_local,
                                    idx_local,
                                    binding_local,
                                    then_branch,
                                    else_branch: None,
                                } = &items[0]
                                && fe.resolve_local(*arr_local) == arr_id
                                && fe.resolve_local(*idx_local) == idx_loc
                                && fe.borrowed_array_str_locals.contains(binding_local)
                                && let StructuredNode::BasicBlock(latch_bb) = items[1]
                            {
                                // Verify `if_hdr` has no unskipped statements
                                let if_hdr_clean = fe.func.blocks[if_hdr.0]
                                    .statements
                                    .iter()
                                    .enumerate()
                                    .all(|(si, st)| {
                                        fe.skip_stmts.contains(&(*if_hdr, si))
                                            || matches!(
                                                st,
                                                Statement::StorageLive(_, _)
                                                    | Statement::StorageDead(_, _)
                                            )
                                    });
                                // Verify `latch_bb` ONLY increments `idx_loc` by 1
                                let lblk = &fe.func.blocks[latch_bb.0];
                                let mut inc_tmp: Option<LocalId> = None;
                                let mut did_inc = false;
                                let mut latch_ok = matches!(
                                    lblk.terminator,
                                    Some(Terminator::Goto(_, _)) | None
                                );
                                for (si, st) in lblk.statements.iter().enumerate() {
                                    if fe.skip_stmts.contains(&(latch_bb, si)) {
                                        continue;
                                    }
                                    match st {
                                        Statement::StorageLive(_, _)
                                        | Statement::StorageDead(_, _) => {}
                                        Statement::Assign(
                                            Place::Local(d),
                                            Rvalue::BinaryOp(
                                                MirBinaryOp::Add,
                                                l,
                                                Operand::Constant(MirConstant::Int(1)),
                                            ),
                                            _,
                                        ) if operand_exact_local(l)
                                            .is_some_and(|x| fe.resolve_local(x) == idx_loc) =>
                                        {
                                            if fe.resolve_local(*d) == idx_loc {
                                                did_inc = true;
                                            } else {
                                                inc_tmp = Some(fe.resolve_local(*d));
                                            }
                                        }
                                        Statement::Assign(
                                            Place::Local(d),
                                            Rvalue::Use(
                                                Operand::Copy(Place::Local(s))
                                                | Operand::Move(Place::Local(s)),
                                            ),
                                            _,
                                        ) if fe.resolve_local(*d) == idx_loc
                                            && inc_tmp == Some(fe.resolve_local(*s)) =>
                                        {
                                            did_inc = true;
                                        }
                                        _ => {
                                            latch_ok = false;
                                            break;
                                        }
                                    }
                                }
                                // Verify `idx_loc` is initialized to 0 once and only used in header + IfArrayGetOk + latch,
                                // and `arr_id` is ONLY used in `len_bb` and `IfArrayGetOk`.
                                let mut idx_zero_defs = 0usize;
                                let mut idx_total_defs = 0usize;
                                let mut arr_uses_other = false;
                                for blk in &fe.func.blocks {
                                    for (si, st) in blk.statements.iter().enumerate() {
                                        if fe.skip_stmts.contains(&(blk.id, si)) {
                                            continue;
                                        }
                                        if let Statement::Assign(Place::Local(d), rv, _) = st
                                            && fe.resolve_local(*d) == idx_loc
                                        {
                                            idx_total_defs += 1;
                                            if matches!(
                                                rv,
                                                Rvalue::Use(Operand::Constant(MirConstant::Int(0)))
                                            ) {
                                                idx_zero_defs += 1;
                                            }
                                        }
                                    }
                                    if let Some(Terminator::Call { args, .. }) = &blk.terminator
                                        && blk.id != len_bb
                                        && blk.id != *split_bb
                                    {
                                        for a in args {
                                            if operand_exact_local(a)
                                                .is_some_and(|x| fe.resolve_local(x) == arr_id)
                                            {
                                                arr_uses_other = true;
                                            }
                                        }
                                    }
                                }
                                // Also ensure `idx_loc` is not read inside `then_branch`
                                fn node_reads_local(
                                    n: &StructuredNode,
                                    fe: &FnEmitter<'_, '_>,
                                    target: LocalId,
                                ) -> bool {
                                    let mut used = HashSet::new();
                                    match n {
                                        StructuredNode::BasicBlock(bb) => {
                                            collect_block_used_locals(&fe.func.blocks[bb.0], &mut used);
                                            used.iter().any(|&u| fe.resolve_local(u) == target)
                                        }
                                        StructuredNode::Seq(xs) => {
                                            xs.iter().any(|x| node_reads_local(x, fe, target))
                                        }
                                        StructuredNode::If {
                                            cond_block,
                                            then_branch,
                                            else_branch,
                                            ..
                                        } => {
                                            collect_block_used_locals(
                                                &fe.func.blocks[cond_block.0],
                                                &mut used,
                                            );
                                            used.iter().any(|&u| fe.resolve_local(u) == target)
                                                || node_reads_local(then_branch, fe, target)
                                                || else_branch
                                                    .as_ref()
                                                    .is_some_and(|e| node_reads_local(e, fe, target))
                                        }
                                        StructuredNode::IfArrayGetOk {
                                            header_block,
                                            arr_local,
                                            idx_local,
                                            then_branch,
                                            else_branch,
                                            ..
                                        } => {
                                            fe.resolve_local(*arr_local) == target
                                                || fe.resolve_local(*idx_local) == target
                                                || {
                                                    collect_block_used_locals(
                                                        &fe.func.blocks[header_block.0],
                                                        &mut used,
                                                    );
                                                    used.iter().any(|&u| fe.resolve_local(u) == target)
                                                }
                                                || node_reads_local(then_branch, fe, target)
                                                || else_branch
                                                    .as_ref()
                                                    .is_some_and(|e| node_reads_local(e, fe, target))
                                        }
                                        StructuredNode::Switch {
                                            discr_block,
                                            cases,
                                            default,
                                            ..
                                        } => {
                                            collect_block_used_locals(
                                                &fe.func.blocks[discr_block.0],
                                                &mut used,
                                            );
                                            used.iter().any(|&u| fe.resolve_local(u) == target)
                                                || cases
                                                    .iter()
                                                    .any(|(_, c)| node_reads_local(c, fe, target))
                                                || node_reads_local(default, fe, target)
                                        }
                                        StructuredNode::WhileLoop {
                                            header_block, body, ..
                                        } => {
                                            collect_block_used_locals(
                                                &fe.func.blocks[header_block.0],
                                                &mut used,
                                            );
                                            used.iter().any(|&u| fe.resolve_local(u) == target)
                                                || node_reads_local(body, fe, target)
                                        }
                                        StructuredNode::Loop { body, .. }
                                        | StructuredNode::LabeledBlock { body, .. } => {
                                            node_reads_local(body, fe, target)
                                        }
                                        StructuredNode::Return(bb, _) => {
                                            collect_block_used_locals(&fe.func.blocks[bb.0], &mut used);
                                            used.iter().any(|&u| fe.resolve_local(u) == target)
                                        }
                                        _ => false,
                                    }
                                }
                                if if_hdr_clean
                                    && latch_ok
                                    && did_inc
                                    && idx_zero_defs == 1
                                    && idx_total_defs == 2
                                    && !arr_uses_other
                                    && !node_reads_local(then_branch, fe, idx_loc)
                                    && !node_reads_local(then_branch, fe, arr_id)
                                {
                                    found.push((
                                        *label,
                                        arr_id,
                                        *split_bb,
                                        len_bb,
                                        src_op.clone(),
                                        sep_lit.clone(),
                                        *binding_local,
                                    ));
                                }
                            }
                        }
                        find_split_iter_loops(body, fe, split_call_info, arr_len_locals, found);
                    }
                    StructuredNode::Seq(items) => {
                        for it in items {
                            find_split_iter_loops(it, fe, split_call_info, arr_len_locals, found);
                        }
                    }
                    StructuredNode::If {
                        then_branch,
                        else_branch,
                        ..
                    }
                    | StructuredNode::IfArrayGetOk {
                        then_branch,
                        else_branch,
                        ..
                    } => {
                        find_split_iter_loops(then_branch, fe, split_call_info, arr_len_locals, found);
                        if let Some(eb) = else_branch {
                            find_split_iter_loops(eb, fe, split_call_info, arr_len_locals, found);
                        }
                    }
                    StructuredNode::Switch { cases, default, .. } => {
                        for (_, c) in cases {
                            find_split_iter_loops(c, fe, split_call_info, arr_len_locals, found);
                        }
                        find_split_iter_loops(default, fe, split_call_info, arr_len_locals, found);
                    }
                    StructuredNode::Loop { body, .. }
                    | StructuredNode::LabeledBlock { body, .. } => {
                        find_split_iter_loops(body, fe, split_call_info, arr_len_locals, found);
                    }
                    _ => {}
                }
            }

            let mut found_loops = Vec::new();
            find_split_iter_loops(tree, self, &split_call_info, &arr_len_locals, &mut found_loops);
            // Only fuse when each `arr_id` appears in a single fused loop
            let mut arr_loop_counts: HashMap<LocalId, usize> = HashMap::new();
            for item in &found_loops {
                *arr_loop_counts.entry(item.1).or_insert(0) += 1;
            }
            for (label, arr_id, split_bb, len_bb, src_op, sep_lit, binding_local) in found_loops {
                if arr_loop_counts.get(&arr_id).copied().unwrap_or(0) == 1 {
                    self.skip_calls.insert(split_bb);
                    self.skip_calls.insert(len_bb);
                    self.omitted_locals.insert(arr_id);
                    self.split_iter_loops
                        .insert(label, (src_op, sep_lit, binding_local));
                }
            }
        }
    }

    /// Pre-pass (`AOT-SSO-REGEX`): detects `res = string.gsub(...)` (`Array<string>`) locals that
    /// are ONLY ever read at constant index `0` via `IfArrayGetOk`, lowering them directly to
    /// `alloc::string::String` so `Vec<String>` and unused `count` formatting are completely elided.
    fn analyze_gsub_first_only_locals(&mut self, tree: &StructuredNode) {
        let param_set: HashSet<LocalId> = self.func.params.iter().copied().collect();
        let mut gsub_inits: HashSet<LocalId> = HashSet::new();
        for blk in &self.func.blocks {
            if blk.is_cleanup {
                continue;
            }
            if let Some(Terminator::Call {
                callee,
                destination: Place::Local(dst),
                ..
            }) = &blk.terminator
                && callee == "string.gsub"
                && matches!(&self.func.locals[dst.0].ty, MirType::Array(inner) if **inner == MirType::String)
                && !param_set.contains(dst)
                && *dst != self.func.return_local
                && self.local_storage(*dst) == LocalStorage::Direct
            {
                gsub_inits.insert(*dst);
            }
        }
        if gsub_inits.is_empty() {
            return;
        }

        let mut group: HashSet<LocalId> = gsub_inits.clone();
        let mut copy_edges: Vec<(LocalId, LocalId, BlockId, usize)> = Vec::new();
        for blk in &self.func.blocks {
            if blk.is_cleanup {
                continue;
            }
            for (sidx, stmt) in blk.statements.iter().enumerate() {
                if let Statement::Assign(
                    Place::Local(dst),
                    Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))),
                    _,
                ) = stmt
                    && group.contains(src)
                    && matches!(&self.func.locals[dst.0].ty, MirType::Array(inner) if **inner == MirType::String)
                    && !param_set.contains(dst)
                    && *dst != self.func.return_local
                    && self.local_storage(*dst) == LocalStorage::Direct
                    && self.local_def_counts.get(dst).copied().unwrap_or(0) == 1
                {
                    group.insert(*dst);
                    copy_edges.push((*dst, *src, blk.id, sidx));
                }
            }
        }

        // Collect constant-int definitions `idx = 0`
        let mut const_zero_locals: HashSet<LocalId> = HashSet::new();
        for blk in &self.func.blocks {
            for stmt in &blk.statements {
                if let Statement::Assign(
                    Place::Local(dst),
                    Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
                    _,
                ) = stmt
                    && self.local_def_counts.get(dst).copied().unwrap_or(0) == 1
                {
                    const_zero_locals.insert(*dst);
                }
            }
        }

        let mut disqual: HashSet<LocalId> = HashSet::new();
        check_gsub_if_array_get_zero_only(tree, self, &group, &const_zero_locals, &mut disqual);

        // Ensure no other uses of `group` exist (except skipped bounds-check len / pure init copy)
        for blk in &self.func.blocks {
            if blk.is_cleanup {
                let mut used = HashSet::new();
                collect_block_used_locals(blk, &mut used);
                for u in used {
                    if group.contains(&u) {
                        disqual.insert(u);
                    }
                }
                continue;
            }
            for (sidx, stmt) in blk.statements.iter().enumerate() {
                if self.skip_stmts.contains(&(blk.id, sidx)) {
                    continue;
                }
                if let Statement::Assign(place, rval, _) = stmt {
                    if let Place::Local(dst) = place
                        && group.contains(dst)
                        && let Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))) = rval
                        && group.contains(src)
                    {
                        continue;
                    }
                    // Allow dead `len` temp only if already skipped; otherwise disqualify
                    let mut tmp = HashSet::new();
                    place.collect_uses(&mut tmp);
                    rval.collect_uses(&mut tmp);
                    let rl = self.resolve_local(place.root_local());
                    if group.contains(&rl) {
                        disqual.insert(rl);
                    }
                    for id in tmp {
                        let r = self.resolve_local(LocalId(id));
                        if group.contains(&r) {
                            disqual.insert(r);
                        }
                    }
                }
            }
            if let Some(term) = &blk.terminator {
                match term {
                    Terminator::Call {
                        callee,
                        args,
                        destination,
                        ..
                    } => {
                        if callee == "string.gsub"
                            && matches!(destination, Place::Local(d) if group.contains(d))
                        {
                            continue;
                        }
                        if let Place::Local(d) = destination
                            && group.contains(d)
                        {
                            disqual.insert(*d);
                        }
                        for a in args {
                            let mut tmp = HashSet::new();
                            a.collect_uses(&mut tmp);
                            for id in tmp {
                                let r = self.resolve_local(LocalId(id));
                                if group.contains(&r) {
                                    disqual.insert(r);
                                }
                            }
                        }
                    }
                    Terminator::Return(Some(op), _) | Terminator::Raise(op, _) => {
                        let mut tmp = HashSet::new();
                        op.collect_uses(&mut tmp);
                        for id in tmp {
                            let r = self.resolve_local(LocalId(id));
                            if group.contains(&r) {
                                disqual.insert(r);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        loop {
            let mut changed = false;
            for &(dst, src, _, _) in &copy_edges {
                if disqual.contains(&dst) && disqual.insert(src) {
                    changed = true;
                }
                if disqual.contains(&src) && disqual.insert(dst) {
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        for loc in group {
            if !disqual.contains(&loc) {
                self.gsub_first_only_locals.insert(loc);
            }
        }
        // Fold pure-init-temp copies `res = Move(gsub_tmp)` so buffer reuse in loops works in-place!
        for (dst, src, bb, sidx) in copy_edges {
            if self.gsub_first_only_locals.contains(&src) && gsub_inits.contains(&src) {
                self.alias_map.insert(dst, src);
                self.skip_stmts.insert((bb, sidx));
                self.omitted_locals.insert(dst);
            }
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
                let mut folded_def_stmt: Option<(BlockId, usize)> = None;
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
                            folded_def_stmt = Some((then_bb, sidx));
                            break;
                        }
                    }
                }
                // Borrowed `IfArrayGetOk` (`AOT-SSO-REGEX`): when `binding_local` is a read-only
                // `MirType::String` that never escapes `then_branch` and `arr_local` is not mutated
                // inside `then_branch`, bind `&str` directly with ZERO `.clone()`!
                let arr_id = self.resolve_local(*arr_local);
                let elem_is_str = matches!(
                    &self.func.locals[arr_id.0].ty,
                    MirType::Array(inner) if **inner == MirType::String
                );
                if elem_is_str
                    && folded_def_stmt.is_some()
                    && self.local_storage(arr_id) == LocalStorage::Direct
                    && self.local_storage(*binding_local) == LocalStorage::Direct
                    && !self.func.params.contains(binding_local)
                    && *binding_local != self.func.return_local
                    && self.local_def_counts.get(binding_local).copied().unwrap_or(0) == 1
                    && !param_escapes_by_move(self.func, *binding_local)
                {
                    let mut then_blocks = HashSet::new();
                    collect_node_blocks(then_branch, &mut then_blocks);
                    let mut used_outside = false;
                    for blk in &self.func.blocks {
                        if then_blocks.contains(&blk.id) {
                            continue;
                        }
                        let mut used = HashSet::new();
                        collect_block_used_locals(blk, &mut used);
                        if used
                            .iter()
                            .any(|&u| self.resolve_local(u) == *binding_local)
                        {
                            used_outside = true;
                            break;
                        }
                    }
                    let arr_mutated_in_then =
                        array_mutated_in_blocks(self.func, arr_id, &then_blocks, &self.alias_map);
                    if !used_outside && !arr_mutated_in_then {
                        self.borrowed_array_str_locals.insert(*binding_local);
                        self.omitted_locals.insert(*binding_local);
                        self.scoped_ref_locals.insert(*binding_local);
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
        let mut to_skip = vec![(and_idx, cond_local)];
        let mut branch_rvals: Vec<&Rvalue> = Vec::new();
        for op in [lhs, rhs] {
            if let Some(loc) = operand_exact_local(op)
                && let Some(&(idx, rval)) = last_def.get(&loc)
            {
                to_skip.push((idx, loc));
                branch_rvals.push(rval);
            }
        }

        // Find optional len_tmp
        for rval in branch_rvals {
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
        if !self.ctx.struct_is_rc_cyclic(&struct_name) {
            self.scoped_ref_locals.insert(binding_local);
        }
        self.if_option_some.insert(
            cond_block,
            IfOptionSomePlan {
                src_place: src_p,
                binding_local,
                struct_name,
            },
        );
    }

    /// Pre-pass (`AOT-BUDGET-SOUND`): when `GasMetering::LoopChunked` is active, batches all
    /// `Statement::ResourceCheck(ResourceKind::Gas, amt, _)` inside each loop's header and body
    /// (including straight-line blocks, branch upper bounds, and statically bounded constant-trip
    /// inner loops) into a single loop-header `__ctx.charge_gas(total_weight)?` call.
    fn analyze_loop_chunked_gas(&mut self, node: &StructuredNode) {
        if self.ctx.opts.gas_metering != GasMetering::LoopChunked {
            return;
        }
        match node {
            StructuredNode::WhileLoop {
                label,
                header_block,
                body,
                ..
            } => {
                let mut total_weight: i64 = 0;
                let mut folded_stmts: Vec<(BlockId, usize)> = Vec::new();
                let mut header_w = 0_i64;
                for (sidx, stmt) in self.func.blocks[header_block.0].statements.iter().enumerate() {
                    if let Statement::ResourceCheck(ResourceKind::Gas, amt, _) = stmt {
                        header_w = header_w.saturating_add((*amt as i64).max(1));
                        folded_stmts.push((*header_block, sidx));
                    }
                }
                total_weight = total_weight.saturating_add(header_w);
                let body_w = self.collect_loop_body_gas_weight(body, &mut folded_stmts);
                total_weight = total_weight.saturating_add(body_w);
                if !folded_stmts.is_empty() {
                    for site in folded_stmts {
                        self.skip_stmts.insert(site);
                    }
                    self.loop_chunked_header_gas
                        .insert(*label, total_weight.max(1));
                }
            }
            StructuredNode::Loop { label, body } => {
                let mut folded_stmts: Vec<(BlockId, usize)> = Vec::new();
                let body_w = self.collect_loop_body_gas_weight(body, &mut folded_stmts);
                if !folded_stmts.is_empty() {
                    for site in folded_stmts {
                        self.skip_stmts.insert(site);
                    }
                    self.loop_chunked_header_gas
                        .insert(*label, body_w.max(1));
                }
            }
            StructuredNode::BasicBlock(bb) => {
                self.batch_single_block_gas(*bb);
            }
            StructuredNode::Seq(items) => {
                for item in items {
                    self.analyze_loop_chunked_gas(item);
                }
            }
            StructuredNode::If {
                cond_block,
                then_branch,
                else_branch,
                ..
            } => {
                self.batch_single_block_gas(*cond_block);
                self.analyze_loop_chunked_gas(then_branch);
                if let Some(eb) = else_branch {
                    self.analyze_loop_chunked_gas(eb);
                }
            }
            StructuredNode::IfArrayGetOk {
                header_block,
                then_branch,
                else_branch,
                ..
            } => {
                self.batch_single_block_gas(*header_block);
                self.analyze_loop_chunked_gas(then_branch);
                if let Some(eb) = else_branch {
                    self.analyze_loop_chunked_gas(eb);
                }
            }
            StructuredNode::Switch {
                discr_block,
                cases,
                default,
                ..
            } => {
                self.batch_single_block_gas(*discr_block);
                for (_, c) in cases {
                    self.analyze_loop_chunked_gas(c);
                }
                self.analyze_loop_chunked_gas(default);
            }
            StructuredNode::LabeledBlock { body, .. } => {
                self.analyze_loop_chunked_gas(body);
            }
            StructuredNode::Return(bb, _) => {
                self.batch_single_block_gas(*bb);
            }
            _ => {}
        }
    }

    fn batch_single_block_gas(&mut self, bb: BlockId) {
        let mut first_sidx: Option<usize> = None;
        let mut total: i64 = 0;
        let mut count: usize = 0;
        for (sidx, stmt) in self.func.blocks[bb.0].statements.iter().enumerate() {
            if self.skip_stmts.contains(&(bb, sidx)) {
                continue;
            }
            if let Statement::ResourceCheck(ResourceKind::Gas, amt, _) = stmt {
                if first_sidx.is_none() {
                    first_sidx = Some(sidx);
                } else {
                    self.skip_stmts.insert((bb, sidx));
                }
                total = total.saturating_add((*amt as i64).max(1));
                count += 1;
            }
        }
        if count > 1
            && let Some(first) = first_sidx
        {
            self.block_chunked_gas.insert((bb, first), total);
        }
    }

    /// Recursively collects gas weights inside a loop body that can be folded into the enclosing
    /// loop header under `GasMetering::LoopChunked`. Nested non-constant loops keep their own
    /// chunked header charge via recursive `analyze_loop_chunked_gas`.
    fn collect_loop_body_gas_weight(
        &mut self,
        node: &StructuredNode,
        folded_stmts: &mut Vec<(BlockId, usize)>,
    ) -> i64 {
        match node {
            StructuredNode::BasicBlock(bb) | StructuredNode::Return(bb, _) => {
                let mut w = 0_i64;
                for (sidx, stmt) in self.func.blocks[bb.0].statements.iter().enumerate() {
                    if let Statement::ResourceCheck(ResourceKind::Gas, amt, _) = stmt {
                        w = w.saturating_add((*amt as i64).max(1));
                        folded_stmts.push((*bb, sidx));
                    }
                }
                w
            }
            StructuredNode::Seq(items) => {
                let mut sum = 0_i64;
                for item in items {
                    sum = sum.saturating_add(self.collect_loop_body_gas_weight(item, folded_stmts));
                }
                sum
            }
            StructuredNode::LabeledBlock { body, .. } => {
                self.collect_loop_body_gas_weight(body, folded_stmts)
            }
            StructuredNode::If {
                cond_block,
                then_branch,
                else_branch,
                ..
            } => {
                let mut w = 0_i64;
                for (sidx, stmt) in self.func.blocks[cond_block.0].statements.iter().enumerate() {
                    if let Statement::ResourceCheck(ResourceKind::Gas, amt, _) = stmt {
                        w = w.saturating_add((*amt as i64).max(1));
                        folded_stmts.push((*cond_block, sidx));
                    }
                }
                let tw = self.collect_loop_body_gas_weight(then_branch, folded_stmts);
                let ew = else_branch
                    .as_deref()
                    .map(|eb| self.collect_loop_body_gas_weight(eb, folded_stmts))
                    .unwrap_or(0);
                w.saturating_add(tw.max(ew))
            }
            StructuredNode::IfArrayGetOk {
                header_block,
                then_branch,
                else_branch,
                ..
            } => {
                let mut w = 0_i64;
                for (sidx, stmt) in self.func.blocks[header_block.0].statements.iter().enumerate() {
                    if let Statement::ResourceCheck(ResourceKind::Gas, amt, _) = stmt {
                        w = w.saturating_add((*amt as i64).max(1));
                        folded_stmts.push((*header_block, sidx));
                    }
                }
                let tw = self.collect_loop_body_gas_weight(then_branch, folded_stmts);
                let ew = else_branch
                    .as_deref()
                    .map(|eb| self.collect_loop_body_gas_weight(eb, folded_stmts))
                    .unwrap_or(0);
                w.saturating_add(tw.max(ew))
            }
            StructuredNode::Switch {
                discr_block,
                cases,
                default,
                ..
            } => {
                let mut w = 0_i64;
                for (sidx, stmt) in self.func.blocks[discr_block.0].statements.iter().enumerate() {
                    if let Statement::ResourceCheck(ResourceKind::Gas, amt, _) = stmt {
                        w = w.saturating_add((*amt as i64).max(1));
                        folded_stmts.push((*discr_block, sidx));
                    }
                }
                let mut max_arm = self.collect_loop_body_gas_weight(default, folded_stmts);
                for (_, c) in cases {
                    let cw = self.collect_loop_body_gas_weight(c, folded_stmts);
                    if cw > max_arm {
                        max_arm = cw;
                    }
                }
                w.saturating_add(max_arm)
            }
            StructuredNode::WhileLoop {
                header_block,
                cond_op,
                negate_cond,
                body,
                ..
            } => {
                // If this inner loop has a small statically-bounded constant trip count (e.g. `while k < 4`),
                // fold `trip_count * inner_weight` directly into the outer loop header charge!
                if let Some(trips) = self.estimate_const_inner_loop_trips(*header_block, cond_op, *negate_cond, body) {
                    let mut inner_folded = Vec::new();
                    let mut inner_w = 0_i64;
                    for (sidx, stmt) in self.func.blocks[header_block.0].statements.iter().enumerate() {
                        if let Statement::ResourceCheck(ResourceKind::Gas, amt, _) = stmt {
                            inner_w = inner_w.saturating_add((*amt as i64).max(1));
                            inner_folded.push((*header_block, sidx));
                        }
                    }
                    let bw = self.collect_loop_body_gas_weight(body, &mut inner_folded);
                    inner_w = inner_w.saturating_add(bw);
                    if !inner_folded.is_empty() {
                        folded_stmts.extend(inner_folded);
                        return inner_w.saturating_mul(trips.saturating_add(1));
                    }
                }
                // Otherwise, analyze the inner loop independently with its own chunked header.
                self.analyze_loop_chunked_gas(node);
                0
            }
            StructuredNode::Loop { .. } => {
                self.analyze_loop_chunked_gas(node);
                0
            }
            _ => 0,
        }
    }

    /// Detects a simple bounded constant-trip inner loop (`k < C` or `k <= C` with `k = k + step`,
    /// `step > 0`, no calls or nested loops) initialized to a constant `start` and returns `trips <= 1024`.
    fn estimate_const_inner_loop_trips(
        &self,
        header_block: BlockId,
        cond_op: &Operand,
        negate_cond: bool,
        body: &StructuredNode,
    ) -> Option<i64> {
        if negate_cond {
            return None;
        }
        let cond_loc = operand_exact_local(cond_op)?;
        let mut cmp_info: Option<(LocalId, MirBinaryOp, i64)> = None;
        for stmt in &self.func.blocks[header_block.0].statements {
            if let Statement::Assign(
                Place::Local(dst),
                Rvalue::BinaryOp(op @ (MirBinaryOp::Lt | MirBinaryOp::LtEq), lhs, Operand::Constant(MirConstant::Int(bound))),
                _,
            ) = stmt
                && *dst == cond_loc
                && let Some(iv) = operand_exact_local(lhs)
            {
                cmp_info = Some((iv, *op, *bound));
            }
        }
        let (iv, op, bound) = cmp_info?;
        // Collect all blocks in `body` (rejecting nested loops, calls, or breaks/returns)
        let mut body_bbs = Vec::new();
        if !collect_simple_straight_body_blocks(body, self.func, &mut body_bbs) {
            return None;
        }
        // Check that `iv` is incremented by a positive constant step inside `body_bbs`
        let mut step_val: Option<i64> = None;
        let mut iv_writes = 0usize;
        for &bb in &body_bbs {
            let mut tmp_adds: HashMap<LocalId, i64> = HashMap::new();
            for stmt in &self.func.blocks[bb.0].statements {
                if let Statement::Assign(Place::Local(dst), rval, _) = stmt {
                    if let Rvalue::BinaryOp(MirBinaryOp::Add, lhs, Operand::Constant(MirConstant::Int(s))) = rval
                        && operand_exact_local(lhs) == Some(iv)
                        && *s > 0
                    {
                        tmp_adds.insert(*dst, *s);
                        if *dst == iv {
                            iv_writes += 1;
                            step_val = Some(*s);
                        }
                    } else if *dst == iv {
                        iv_writes += 1;
                        if let Rvalue::Use(op_src) = rval
                            && let Some(src_l) = operand_exact_local(op_src)
                            && let Some(&s) = tmp_adds.get(&src_l)
                        {
                            step_val = Some(s);
                        } else {
                            return None;
                        }
                    }
                }
            }
        }
        let step = step_val?;
        if iv_writes != 1 || step <= 0 {
            return None;
        }
        // Find constant initializer for `iv` outside `body_bbs`
        let body_set: HashSet<BlockId> = body_bbs.into_iter().collect();
        let mut init_val: Option<i64> = None;
        for blk in &self.func.blocks {
            if blk.is_cleanup || blk.id == header_block || body_set.contains(&blk.id) {
                continue;
            }
            for stmt in &blk.statements {
                if let Statement::Assign(Place::Local(dst), rval, _) = stmt
                    && *dst == iv
                {
                    if let Rvalue::Use(Operand::Constant(MirConstant::Int(c))) = rval {
                        if init_val.is_some() && init_val != Some(*c) {
                            return None;
                        }
                        init_val = Some(*c);
                    } else {
                        return None;
                    }
                }
            }
        }
        let start = init_val?;
        let end = if op == MirBinaryOp::LtEq {
            bound.checked_add(1)?
        } else {
            bound
        };
        if end <= start {
            return Some(0);
        }
        let span = end.checked_sub(start)?;
        let trips = (span + step - 1) / step;
        if (0..=1024).contains(&trips) {
            Some(trips)
        } else {
            None
        }
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
                                self.emit_call(*bb, callee, args, destination, indent)?;
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
                    if self.ctx.struct_is_rc_cyclic(&plan.struct_name) {
                        let opt_expr = if matches!(plan.src_place, Place::Field(_, _)) {
                            src_expr
                        } else {
                            format!("{src_expr}.clone()")
                        };
                        let _ = writeln!(
                            self.out,
                            "{pad}if let Some(mut _{}) = {opt_expr} {{",
                            plan.binding_local.0
                        );
                    } else {
                        let deref_expr = if self.ctx.struct_is_arena_cyclic(&plan.struct_name) {
                            format!("{src_expr}.as_deref()")
                        } else {
                            format!("{src_expr}.as_ref()")
                        };
                        let _ = writeln!(
                            self.out,
                            "{pad}if let Some(_{}) = {deref_expr} {{",
                            plan.binding_local.0
                        );
                    }
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
                if self.gsub_first_only_locals.contains(&arr_id) {
                    let _ = writeln!(self.out, "{pad}{{");
                    if self.borrowed_array_str_locals.contains(binding_local) {
                        let _ = writeln!(
                            self.out,
                            "{pad}    let _{}: &str = {arr_var}.as_str();",
                            binding_local.0
                        );
                    } else {
                        self.emit_assign_local_val(
                            *binding_local,
                            &format!("{arr_var}.clone()"),
                            indent + 1,
                        );
                    }
                    self.emit_node(then_branch, indent + 1)?;
                    let _ = writeln!(self.out, "{pad}}}");
                    return Ok(());
                }
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
                } else if self.split_pair_locals.contains(&arr_id) {
                    let cidx = self.const_int_locals.get(&idx_id).copied();
                    let field = if cidx == Some(0) { "0" } else { "1" };
                    let _ = writeln!(
                        self.out,
                        "{pad}if let Some(__elem) = {arr_var}.{field} {{"
                    );
                    if self.borrowed_array_str_locals.contains(binding_local) {
                        let _ = writeln!(
                            self.out,
                            "{pad}    let _{}: &str = __elem;",
                            binding_local.0
                        );
                    } else {
                        self.emit_assign_local_val(
                            *binding_local,
                            "alloc::string::String::from(__elem)",
                            indent + 1,
                        );
                    }
                } else if let Some(src_str_op) = self.small_str_array_locals.get(&arr_id).cloned() {
                    let src_str_expr = self.emit_operand_str_ref(&src_str_op);
                    let _ = writeln!(
                        self.out,
                        "{pad}if let Some((__lo, __hi)) = usize::try_from({idx_expr}).ok().and_then(|__i| {arr_var}.get(__i)) {{"
                    );
                    if self.borrowed_array_str_locals.contains(binding_local) {
                        let _ = writeln!(
                            self.out,
                            "{pad}    let _{}: &str = &({src_str_expr})[__lo..__hi];",
                            binding_local.0
                        );
                    } else {
                        self.emit_assign_local_val(
                            *binding_local,
                            &format!("alloc::string::String::from(&({src_str_expr})[__lo..__hi])"),
                            indent + 1,
                        );
                    }
                } else if self.borrowed_array_str_locals.contains(binding_local) {
                    let _ = writeln!(
                        self.out,
                        "{pad}if let Some(__elem) = usize::try_from({idx_expr}).ok().and_then(|__i| {arr_var}.get(__i)) {{"
                    );
                    let _ = writeln!(
                        self.out,
                        "{pad}    let _{}: &str = __elem.as_str();",
                        binding_local.0
                    );
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
                if let Some((src_op, sep_lit, binding_local)) =
                    self.split_iter_loops.get(label).cloned()
                    && let StructuredNode::Seq(items) = body.as_ref()
                    && let Some(StructuredNode::IfArrayGetOk { then_branch, .. }) = items.first()
                {
                    let src_expr = self.emit_operand_str_ref(&src_op);
                    let sep_arg = if sep_lit.len() == 1 && sep_lit.as_bytes()[0] < 128 {
                        format!("{:?}", sep_lit.as_bytes()[0] as char)
                    } else {
                        format!("{sep_lit:?}")
                    };
                    let _ = writeln!(
                        self.out,
                        "{pad}'bb{}: for _{} in ({src_expr}).split({sep_arg}) {{",
                        label.0, binding_local.0
                    );
                    if let Some(&weight) = self.loop_chunked_header_gas.get(label)
                        && self.in_cleanup_closure.is_none()
                    {
                        let _ = writeln!(self.out, "{pad}    __ctx.charge_gas({weight}_i64)?;");
                    }
                    self.emit_node(then_branch, indent + 1)?;
                    let _ = writeln!(self.out, "{pad}}}");
                    return Ok(());
                }
                let _ = writeln!(self.out, "{pad}'bb{}: loop {{", label.0);
                if let Some(&weight) = self.loop_chunked_header_gas.get(label)
                    && self.in_cleanup_closure.is_none()
                {
                    let _ = writeln!(self.out, "{pad}    __ctx.charge_gas({weight}_i64)?;");
                }
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
                if let Some(&weight) = self.loop_chunked_header_gas.get(label)
                    && self.in_cleanup_closure.is_none()
                {
                    let _ = writeln!(self.out, "{pad}    __ctx.charge_gas({weight}_i64)?;");
                }
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
                        self.emit_call(bb, callee, args, destination, indent + 3)?;
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
            if let Statement::ResourceCheck(ResourceKind::Gas, _, _) = stmt
                && let Some(&batched) = self.block_chunked_gas.get(&(bb, sidx))
            {
                if self.in_cleanup_closure.is_none() {
                    let pad = "    ".repeat(indent);
                    let _ = writeln!(self.out, "{pad}__ctx.charge_gas({batched}_i64)?;");
                }
                continue;
            }
            if let Some(pieces) = self.fused_concat_chains.get(&(bb, sidx)).cloned()
                && let Statement::Assign(place, _, _) = stmt
            {
                self.emit_fused_concat_chain(place, &pieces, indent);
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
                            if matches!(*k_ty, MirType::String) {
                                let k_ref = self.emit_operand_str_ref(&Operand::Copy(Place::Local(*idx)));
                                let _ = writeln!(
                                    self.out,
                                    "{pad}{{ let __val = {val_expr}; {map_expr}.set_borrowed({k_ref}, __val, __alloc)?; }}"
                                );
                            } else {
                                let key_expr = self.emit_local_as_map_key(*idx, &k_ty);
                                let _ = writeln!(
                                    self.out,
                                    "{pad}{{ let __val = {val_expr}; {map_expr}.set({key_expr}, __val, __alloc)?; }}"
                                );
                            }
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

                // Special case: field write `s.field_i = val` on `RcRefCell` struct or `is_mutated_cyclic` struct
                if let Place::Field(base, fidx) = place {
                    let base_ty = self.place_mir_type(base);
                    if let MirType::Struct(sname) = &base_ty
                        && self.ctx.struct_is_rc_cyclic(sname)
                    {
                        let target_ty = self.place_mir_type(place);
                        let val_expr = self.emit_rvalue(rval, Some(&target_ty))?;
                        let base_expr = self.emit_place_value_expr(base);
                        let _ = writeln!(
                            self.out,
                            "{pad}{{ let __val = {val_expr}; lust_aot_rt::ref_borrow_mut(&({base_expr}).0)?.field_{fidx} = __val; }}"
                        );
                        return Ok(());
                    }
                    if let Place::Local(base_loc) = base.as_ref()
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
                }

                // Special case: borrowed `&str` select local assignment `_6 = Copy(_1)` -> `_6 = _1.as_str();`
                if let Place::Local(dst) = place
                    && self.borrowed_select_str_locals.contains(&self.resolve_local(*dst))
                    && let Rvalue::Use(op) = rval
                {
                    let dst_var = self.local_var_name(*dst);
                    let sref = self.emit_operand_str_ref(op);
                    let _ = writeln!(self.out, "{pad}{dst_var} = {sref};");
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
                let is_bool_op = matches!(
                    op,
                    MirBinaryOp::Eq
                        | MirBinaryOp::NotEq
                        | MirBinaryOp::Lt
                        | MirBinaryOp::LtEq
                        | MirBinaryOp::Gt
                        | MirBinaryOp::GtEq
                        | MirBinaryOp::And
                        | MirBinaryOp::Or
                );
                let any_unknown = matches!(l_ty, MirType::Unknown) || matches!(r_ty, MirType::Unknown);
                if any_unknown
                    && !is_bool_op
                    && let Some(exp) = expected_ty
                    && !matches!(exp, MirType::Unknown)
                {
                    Ok(self.coerce_expr_to_type(&raw, &MirType::Unknown, exp))
                } else if let Some(exp) = expected_ty
                    && matches!(exp, MirType::Unknown)
                {
                    let res_ty = if is_bool_op {
                        MirType::Bool
                    } else if any_unknown {
                        MirType::Unknown
                    } else if matches!(op, MirBinaryOp::Concat) {
                        MirType::String
                    } else if matches!(l_ty, MirType::Float) || matches!(r_ty, MirType::Float) {
                        MirType::Float
                    } else {
                        MirType::Int
                    };
                    if matches!(res_ty, MirType::Unknown) {
                        Ok(raw)
                    } else {
                        Ok(self.coerce_expr_to_type(&raw, &res_ty, &MirType::Unknown))
                    }
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
                } else if let Some(exp) = expected_ty
                    && matches!(exp, MirType::Unknown)
                {
                    let res_ty = match op {
                        MirUnaryOp::Not => MirType::Bool,
                        MirUnaryOp::Len => MirType::Int,
                        MirUnaryOp::Neg => {
                            if matches!(o_ty, MirType::Unknown) {
                                MirType::Unknown
                            } else if matches!(o_ty, MirType::Float) {
                                MirType::Float
                            } else {
                                MirType::Int
                            }
                        }
                        MirUnaryOp::BitNot => {
                            if matches!(o_ty, MirType::Unknown) {
                                MirType::Unknown
                            } else {
                                MirType::Int
                            }
                        }
                    };
                    if matches!(res_ty, MirType::Unknown) {
                        Ok(raw)
                    } else {
                        Ok(self.coerce_expr_to_type(&raw, &res_ty, &MirType::Unknown))
                    }
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
                    if meta.is_some_and(|m| m.is_mutated_cyclic) {
                        let data_name = format!("{rust_sname}_Data");
                        if self.ctx.opts.max_memory_bytes.is_some() {
                            let data_ty = if meta.is_some_and(|m| m.needs_arena_lt) {
                                format!("{data_name}<'_>")
                            } else {
                                data_name.clone()
                            };
                            fields.push("__charged: __sz".to_string());
                            fields.push("__budget: __alloc.share()".to_string());
                            Ok(format!(
                                "{{ let __sz = core::mem::size_of::<{data_ty}>(); __alloc.try_charge(__sz).map_err(lust_aot_rt::AotError::from)?; {rust_sname}(alloc::rc::Rc::new(core::cell::RefCell::new({data_name} {{ {} }}))) }}",
                                fields.join(", ")
                            ))
                        } else {
                            Ok(format!(
                                "{rust_sname}(alloc::rc::Rc::new(core::cell::RefCell::new({data_name} {{ {} }})))",
                                fields.join(", ")
                            ))
                        }
                    } else {
                        Ok(format!("{rust_sname} {{ {} }}", fields.join(", ")))
                    }
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
        if matches!(expected_ty, Some(MirType::Unknown | MirType::Effect { .. })) {
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
                "let __c_alloc = __c_ctx.budget_alloc.share(); "
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

        // 1-byte `string.sub(s, i, i)` peephole (`AOT-SSO-REGEX`): `Option<u8> == Some(b)`
        if matches!(op, MirBinaryOp::Eq | MirBinaryOp::NotEq) {
            let cmp = if op == MirBinaryOp::Eq { "==" } else { "!=" };
            if let Some(l) = operand_exact_local(lhs).map(|id| self.resolve_local(id))
                && self.byte_sub_locals.contains(&l)
                && let Operand::Constant(MirConstant::String(lit)) = rhs
                && lit.len() == 1
                && lit.is_ascii()
            {
                let var = self.local_var_name(l);
                let byte = lit.as_bytes()[0];
                return Ok(format!("({var} {cmp} Some({byte}_u8))"));
            }
            if let Some(r) = operand_exact_local(rhs).map(|id| self.resolve_local(id))
                && self.byte_sub_locals.contains(&r)
                && let Operand::Constant(MirConstant::String(lit)) = lhs
                && lit.len() == 1
                && lit.is_ascii()
            {
                let var = self.local_var_name(r);
                let byte = lit.as_bytes()[0];
                return Ok(format!("({var} {cmp} Some({byte}_u8))"));
            }
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
                    if self.operand_is_direct_raw_map(operand) {
                        Ok(format!("({m}.len() as i64)"))
                    } else {
                        Ok(format!("{m}.len()?"))
                    }
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
                if self.ctx.struct_is_arena_cyclic(sname) {
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
        bb: BlockId,
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

        if let Some(struct_name) = callee.strip_suffix(".decode")
            && !self.ctx.fn_metas.contains_key(callee)
            && self
                .ctx
                .structs
                .get(struct_name)
                .is_some_and(|s| s.has_schema_decoders)
        {
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
                        if self.operand_is_direct_raw_map(arg) {
                            format!("({}.len() as i64)", self.emit_operand_value(arg))
                        } else {
                            format!("{}.len()?", self.emit_operand_value(arg))
                        }
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
                    } else if self.place_is_direct_raw_map(destination) {
                        "lust_aot_rt::AotRawMap::new()".to_string()
                    } else {
                        "lust_aot_rt::AotMap::new()".to_string()
                    };
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "map.set" | "table.set" if args.len() >= 3 => {
                if self.map_rmw_set_plans.contains_key(&bb) {
                    let m_ty = operand_mir_type(&args[0], self.func);
                    if let MirType::Map(_, v_ty) = m_ty {
                        let v_expr = self.emit_operand_owned(&args[2], &v_ty);
                        let _ = writeln!(self.out, "{pad}*__rmw_slot = {v_expr};");
                        return Ok(());
                    }
                }
                let m_ty = operand_mir_type(&args[0], self.func);
                match m_ty {
                    MirType::Map(k_ty, v_ty) => {
                        let m_expr = self.emit_operand_value(&args[0]);
                        let v_expr = self.emit_operand_owned(&args[2], &v_ty);
                        if matches!(*k_ty, MirType::String) {
                            let k_ref = self.emit_operand_str_ref(&args[1]);
                            let _ = writeln!(
                                self.out,
                                "{pad}{{ let __v = {v_expr}; {m_expr}.set_borrowed({k_ref}, __v, __alloc)?; }}"
                            );
                        } else {
                            let k_expr = self.emit_operand_as_map_key(&args[1], &k_ty);
                            let _ = writeln!(
                                self.out,
                                "{pad}{{ let __k = {k_expr}; let __v = {v_expr}; {m_expr}.set(__k, __v, __alloc)?; }}"
                            );
                        }
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
                if let Some(&(_, m_loc, ref k_op, val_dst)) = self.map_rmw_get_plans.get(&bb) {
                    let m_ty = self.func.locals[m_loc.0].ty.clone();
                    let m_var = self.local_var_name(m_loc);
                    if let MirType::Map(k_ty, _) = m_ty {
                        if matches!(*k_ty, MirType::String) {
                            let k_ref = self.emit_operand_str_ref(k_op);
                            let _ = writeln!(
                                self.out,
                                "{pad}let __rmw_slot = {m_var}.get_mut({k_ref}).ok_or(lust_aot_rt::AotError::MissingMapKey)?;"
                            );
                        } else {
                            let k_expr = self.emit_operand_as_map_key(k_op, &k_ty);
                            let _ = writeln!(
                                self.out,
                                "{pad}let __rmw_slot = {{ let __k = {k_expr}; {m_var}.get_mut(&__k).ok_or(lust_aot_rt::AotError::MissingMapKey)? }};"
                            );
                        }
                        self.emit_assign_local_val(val_dst, "*__rmw_slot", indent);
                        return Ok(());
                    }
                }
                if !self.dest_is_omitted(destination) {
                    let m_ty = operand_mir_type(&args[0], self.func);
                    let dst_ty = self.place_mir_type(destination);
                    let is_raw = self.operand_is_direct_raw_map(&args[0]);
                    let expr = match m_ty {
                        MirType::Map(k_ty, v_ty) => {
                            let m_expr = self.emit_operand_value(&args[0]);
                            if matches!(*k_ty, MirType::String) {
                                let k_ref = self.emit_operand_str_ref(&args[1]);
                                if matches!(dst_ty, MirType::Option(_)) {
                                    if is_raw {
                                        format!("{m_expr}.get_opt({k_ref})")
                                    } else {
                                        format!("{m_expr}.get_opt({k_ref})?")
                                    }
                                } else {
                                    let raw = if is_raw {
                                        format!("{m_expr}.get_or_err({k_ref})?")
                                    } else {
                                        format!("{m_expr}.get({k_ref})?")
                                    };
                                    self.coerce_expr_to_type(&raw, &v_ty, &dst_ty)
                                }
                            } else {
                                let k_expr = self.emit_operand_as_map_key(&args[1], &k_ty);
                                if matches!(dst_ty, MirType::Option(_)) {
                                    if is_raw {
                                        format!("{{ let __k = {k_expr}; {m_expr}.get_opt(&__k) }}")
                                    } else {
                                        format!("{{ let __k = {k_expr}; {m_expr}.get_opt(&__k)? }}")
                                    }
                                } else {
                                    let raw = if is_raw {
                                        format!("{{ let __k = {k_expr}; {m_expr}.get_or_err(&__k)? }}")
                                    } else {
                                        format!("{{ let __k = {k_expr}; {m_expr}.get(&__k)? }}")
                                    };
                                    self.coerce_expr_to_type(&raw, &v_ty, &dst_ty)
                                }
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
            "map.get_unwrap" if args.len() >= 2 => {
                if let Some(&(_, m_loc, ref k_op, val_dst)) = self.map_rmw_get_plans.get(&bb) {
                    let m_ty = self.func.locals[m_loc.0].ty.clone();
                    let m_var = self.local_var_name(m_loc);
                    if let MirType::Map(k_ty, _) = m_ty {
                        if matches!(*k_ty, MirType::String) {
                            let k_ref = self.emit_operand_str_ref(k_op);
                            let _ = writeln!(
                                self.out,
                                "{pad}let __rmw_slot = {m_var}.get_mut({k_ref}).ok_or(lust_aot_rt::AotError::MissingMapKey)?;"
                            );
                        } else {
                            let k_expr = self.emit_operand_as_map_key(k_op, &k_ty);
                            let _ = writeln!(
                                self.out,
                                "{pad}let __rmw_slot = {{ let __k = {k_expr}; {m_var}.get_mut(&__k).ok_or(lust_aot_rt::AotError::MissingMapKey)? }};"
                            );
                        }
                        self.emit_assign_local_val(val_dst, "*__rmw_slot", indent);
                        return Ok(());
                    }
                }
                if !self.dest_is_omitted(destination) {
                    let m_ty = operand_mir_type(&args[0], self.func);
                    let dst_ty = self.place_mir_type(destination);
                    let is_raw = self.operand_is_direct_raw_map(&args[0]);
                    let expr = match m_ty {
                        MirType::Map(k_ty, v_ty) => {
                            let m_expr = self.emit_operand_value(&args[0]);
                            let raw = if matches!(*k_ty, MirType::String) {
                                let k_ref = self.emit_operand_str_ref(&args[1]);
                                if is_raw {
                                    format!("{m_expr}.get_or_err({k_ref})?")
                                } else {
                                    format!("{m_expr}.get({k_ref})?")
                                }
                            } else {
                                let k_expr = self.emit_operand_as_map_key(&args[1], &k_ty);
                                if is_raw {
                                    format!("{{ let __k = {k_expr}; {m_expr}.get_or_err(&__k)? }}")
                                } else {
                                    format!("{{ let __k = {k_expr}; {m_expr}.get(&__k)? }}")
                                }
                            };
                            self.coerce_expr_to_type(&raw, &v_ty, &dst_ty)
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
            "map.get_unwrap_or" if args.len() >= 3 => {
                if !self.dest_is_omitted(destination) {
                    let m_ty = operand_mir_type(&args[0], self.func);
                    let dst_ty = self.place_mir_type(destination);
                    let is_raw = self.operand_is_direct_raw_map(&args[0]);
                    let expr = match m_ty {
                        MirType::Map(k_ty, v_ty) => {
                            let m_expr = self.emit_operand_value(&args[0]);
                            let def_expr = self.emit_operand_owned(&args[2], &v_ty);
                            let raw = if matches!(*k_ty, MirType::String) {
                                let k_ref = self.emit_operand_str_ref(&args[1]);
                                if is_raw {
                                    format!(
                                        "match {m_expr}.get_opt({k_ref}) {{ Some(__v) => __v, None => {def_expr} }}"
                                    )
                                } else {
                                    format!(
                                        "match {m_expr}.get_opt({k_ref})? {{ Some(__v) => __v, None => {def_expr} }}"
                                    )
                                }
                            } else {
                                let k_expr = self.emit_operand_as_map_key(&args[1], &k_ty);
                                if is_raw {
                                    format!(
                                        "{{ let __k = {k_expr}; match {m_expr}.get_opt(&__k) {{ Some(__v) => __v, None => {def_expr} }} }}"
                                    )
                                } else {
                                    format!(
                                        "{{ let __k = {k_expr}; match {m_expr}.get_opt(&__k)? {{ Some(__v) => __v, None => {def_expr} }} }}"
                                    )
                                }
                            };
                            self.coerce_expr_to_type(&raw, &v_ty, &dst_ty)
                        }
                        MirType::Unknown => {
                            let m_expr = self.emit_operand_value(&args[0]);
                            let k_expr = self.emit_operand_as_map_key(&args[1], &MirType::Unknown);
                            let def_expr = self.emit_operand_as_aot_value(&args[2]);
                            let raw = format!(
                                "{{ let __k = {k_expr}; match {m_expr}.map_get_opt(&__k)? {{ Some(__v) => __v, None => {def_expr} }} }}"
                            );
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
                    let is_raw = self.operand_is_direct_raw_map(&args[0]);
                    let expr = match m_ty {
                        MirType::Map(k_ty, _) => {
                            if matches!(*k_ty, MirType::String) {
                                let k_ref = self.emit_operand_str_ref(&args[1]);
                                if is_raw {
                                    format!("{m_expr}.has({k_ref})")
                                } else {
                                    format!("{m_expr}.has({k_ref})?")
                                }
                            } else {
                                let k_expr = self.emit_operand_as_map_key(&args[1], &k_ty);
                                if is_raw {
                                    format!("{{ let __k = {k_expr}; {m_expr}.has(&__k) }}")
                                } else {
                                    format!("{{ let __k = {k_expr}; {m_expr}.has(&__k)? }}")
                                }
                            }
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
                    let is_raw = self.operand_is_direct_raw_map(&args[0]);
                    let expr = if matches!(m_ty, MirType::Unknown) {
                        format!("{m_expr}.map_is_empty()?")
                    } else if is_raw {
                        format!("{m_expr}.is_empty()")
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
                let is_raw = self.operand_is_direct_raw_map(&args[0]);
                let expr = match m_ty {
                    MirType::Map(k_ty, _) => {
                        if matches!(*k_ty, MirType::String) {
                            let k_ref = self.emit_operand_str_ref(&args[1]);
                            if is_raw {
                                format!("{m_expr}.delete({k_ref}, __alloc)")
                            } else {
                                format!("{m_expr}.delete({k_ref}, __alloc)?")
                            }
                        } else {
                            let k_expr = self.emit_operand_as_map_key(&args[1], &k_ty);
                            if is_raw {
                                format!("{{ let __k = {k_expr}; {m_expr}.delete(&__k, __alloc) }}")
                            } else {
                                format!("{{ let __k = {k_expr}; {m_expr}.delete(&__k, __alloc)? }}")
                            }
                        }
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
                let is_raw = self.operand_is_direct_raw_map(&args[0]);
                if matches!(m_ty, MirType::Unknown) {
                    let _ = writeln!(self.out, "{pad}{m_expr}.map_clear()?;");
                } else if is_raw {
                    let _ = writeln!(self.out, "{pad}{m_expr}.clear();");
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
            "string.upper" if !args.is_empty() => {
                if !self.dest_is_omitted(destination) {
                    let s = self.emit_operand_str_ref(&args[0]);
                    let dst_ty = self.place_mir_type(destination);
                    let raw = format!("({s}).to_uppercase()");
                    let expr = self.coerce_expr_to_type(&raw, &MirType::String, &dst_ty);
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "string.lower" if !args.is_empty() => {
                if !self.dest_is_omitted(destination) {
                    let s = self.emit_operand_str_ref(&args[0]);
                    let dst_ty = self.place_mir_type(destination);
                    let raw = format!("({s}).to_lowercase()");
                    let expr = self.coerce_expr_to_type(&raw, &MirType::String, &dst_ty);
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "string.trim" if !args.is_empty() => {
                if !self.dest_is_omitted(destination) {
                    let s = self.emit_operand_str_ref(&args[0]);
                    let dst_ty = self.place_mir_type(destination);
                    let raw = format!("alloc::string::String::from(({s}).trim())");
                    let expr = self.coerce_expr_to_type(&raw, &MirType::String, &dst_ty);
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "string.contains" if args.len() >= 2 => {
                if !self.dest_is_omitted(destination) {
                    let s = self.emit_operand_str_ref(&args[0]);
                    let needle = self.emit_operand_str_ref(&args[1]);
                    let dst_ty = self.place_mir_type(destination);
                    let raw = format!("({s}).contains({needle})");
                    let expr = self.coerce_expr_to_type(&raw, &MirType::Bool, &dst_ty);
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "string.starts_with" if args.len() >= 2 => {
                if !self.dest_is_omitted(destination) {
                    let s = self.emit_operand_str_ref(&args[0]);
                    let prefix = self.emit_operand_str_ref(&args[1]);
                    let dst_ty = self.place_mir_type(destination);
                    let raw = format!("({s}).starts_with({prefix})");
                    let expr = self.coerce_expr_to_type(&raw, &MirType::Bool, &dst_ty);
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "string.ends_with" if args.len() >= 2 => {
                if !self.dest_is_omitted(destination) {
                    let s = self.emit_operand_str_ref(&args[0]);
                    let suffix = self.emit_operand_str_ref(&args[1]);
                    let dst_ty = self.place_mir_type(destination);
                    let raw = format!("({s}).ends_with({suffix})");
                    let expr = self.coerce_expr_to_type(&raw, &MirType::Bool, &dst_ty);
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "string.concat" if args.len() >= 2 => {
                if !self.dest_is_omitted(destination) {
                    let a = self.emit_operand_str_ref(&args[0]);
                    let b = self.emit_operand_str_ref(&args[1]);
                    let dst_ty = self.place_mir_type(destination);
                    let raw = format!(
                        "{{ let mut __s = alloc::string::String::from({a}); __s.push_str({b}); __s }}"
                    );
                    let expr = self.coerce_expr_to_type(&raw, &MirType::String, &dst_ty);
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "string.sub" if args.len() >= 2 => {
                if !self.dest_is_omitted(destination) {
                    if let Place::Local(dst) = destination
                        && self.byte_sub_locals.contains(&self.resolve_local(*dst))
                    {
                        let s = self.emit_operand_str_ref(&args[0]);
                        let idx_expr = self.emit_operand_owned(&args[1], &MirType::Int);
                        let raw = format!(
                            "{{ let __b = ({s}).as_bytes(); let __i = {idx_expr}; let __adj = if __i >= 1 {{ __i.wrapping_sub(1) as usize }} else if __i < 0 {{ (__b.len() as i64).wrapping_add(__i) as usize }} else {{ usize::MAX }}; __b.get(__adj).copied() }}"
                        );
                        self.emit_assign_place_val(destination, &raw, indent);
                        return Ok(());
                    }
                    if let Place::Local(dst) = destination
                        && self.cow_str_locals.contains(&self.resolve_local(*dst))
                    {
                        let s = self.emit_operand_str_ref(&args[0]);
                        let start = self.emit_operand_owned(&args[1], &MirType::Int);
                        let end = args
                            .get(2)
                            .map(|a| format!("Some({})", self.emit_operand_owned(a, &MirType::Int)))
                            .unwrap_or_else(|| "None".to_string());
                        let raw = format!("__lust_aot_string_sub_cow({s}, {start}, {end})");
                        self.emit_assign_place_val(destination, &raw, indent);
                        return Ok(());
                    }
                    let s = self.emit_operand_str_ref(&args[0]);
                    let start = self.emit_operand_owned(&args[1], &MirType::Int);
                    let end = args
                        .get(2)
                        .map(|a| format!("Some({})", self.emit_operand_owned(a, &MirType::Int)))
                        .unwrap_or_else(|| "None".to_string());
                    let dst_ty = self.place_mir_type(destination);
                    let raw = format!("__lust_aot_string_sub({s}, {start}, {end})");
                    let expr = self.coerce_expr_to_type(&raw, &MirType::String, &dst_ty);
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "string.byte" if !args.is_empty() => {
                if !self.dest_is_omitted(destination) {
                    let s = self.emit_operand_str_ref(&args[0]);
                    let start = args
                        .get(1)
                        .map(|a| self.emit_operand_owned(a, &MirType::Int))
                        .unwrap_or_else(|| "1_i64".to_string());
                    let end = args
                        .get(2)
                        .map(|a| format!("Some({})", self.emit_operand_owned(a, &MirType::Int)))
                        .unwrap_or_else(|| "None".to_string());
                    let dst_ty = self.place_mir_type(destination);
                    let expr = match &dst_ty {
                        MirType::Array(inner) if matches!(**inner, MirType::Int) => {
                            format!("__lust_aot_string_byte_ints({s}, {start}, {end}, __alloc)")
                        }
                        MirType::Array(_) => {
                            format!("__lust_aot_string_byte_vals({s}, {start}, {end}, __alloc)")
                        }
                        MirType::Int => format!(
                            "{{ let __b = __lust_aot_string_byte_ints({s}, {start}, {end}, __alloc); __b.first().copied().unwrap_or(0_i64) }}"
                        ),
                        MirType::Unknown => format!(
                            "{{ let __b = __lust_aot_string_byte_vals({s}, {start}, {end}, __alloc); lust_aot_rt::AotValue::array(__b.into_iter().collect()) }}"
                        ),
                        _ => format!("__lust_aot_string_byte_vals({s}, {start}, {end}, __alloc)"),
                    };
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "string.split" if args.len() >= 2 => {
                if !self.dest_is_omitted(destination) {
                    let s = self.emit_operand_str_ref(&args[0]);
                    let sep = self.emit_operand_str_ref(&args[1]);
                    if let Place::Local(dst) = destination
                        && self.split_pair_locals.contains(&self.resolve_local(*dst))
                    {
                        let expr = if let Operand::Constant(MirConstant::String(sep_lit)) = &args[1]
                            && sep_lit.len() == 1
                            && sep_lit.as_bytes()[0] < 128
                        {
                            format!(
                                "__lust_aot_string_split_pair_byte({s}, {}_u8)",
                                sep_lit.as_bytes()[0]
                            )
                        } else {
                            format!("__lust_aot_string_split_pair({s}, {sep})")
                        };
                        self.emit_assign_place_val(destination, &expr, indent);
                        return Ok(());
                    }
                    if let Place::Local(dst) = destination
                        && self.small_str_array_locals.contains_key(&self.resolve_local(*dst))
                    {
                        let expr = if let Operand::Constant(MirConstant::String(sep_lit)) = &args[1]
                            && sep_lit.len() == 1
                            && sep_lit.as_bytes()[0] < 128
                        {
                            format!(
                                "__lust_aot_string_split_spans_byte({s}, {}_u8, __alloc)",
                                sep_lit.as_bytes()[0]
                            )
                        } else {
                            format!("__lust_aot_string_split_spans({s}, {sep}, __alloc)")
                        };
                        self.emit_assign_place_val(destination, &expr, indent);
                        return Ok(());
                    }
                    let dst_ty = self.place_mir_type(destination);
                    let expr = match &dst_ty {
                        MirType::Array(inner) if matches!(**inner, MirType::Unknown) => format!(
                            "{{ let __parts = __lust_aot_string_split({s}, {sep}, __alloc); let mut __out = alloc::vec::Vec::new_in(__alloc); for __p in __parts {{ __out.push(lust_aot_rt::AotValue::str(__p)); }} __out }}"
                        ),
                        MirType::Unknown => format!(
                            "{{ let __parts = __lust_aot_string_split({s}, {sep}, __alloc); lust_aot_rt::AotValue::array(__parts.into_iter().map(lust_aot_rt::AotValue::str).collect()) }}"
                        ),
                        _ => format!("__lust_aot_string_split({s}, {sep}, __alloc)"),
                    };
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "string.gsub" if args.len() >= 3 => {
                let s = self.emit_operand_str_ref(&args[0]);
                let pat = self.emit_operand_str_ref(&args[1]);
                let repl = self.emit_operand_str_ref(&args[2]);
                let limit_op = args.get(3);
                let limit = limit_op
                    .map(|a| format!("Some({})", self.emit_operand_owned(a, &MirType::Int)))
                    .unwrap_or_else(|| "None".to_string());
                // Compile-time constant ASCII character-class `+` specialization (`AOT-SSO-REGEX`):
                // e.g. `string.gsub(line, "[0-9]+", "#")` emits a zero-alloc O(N) const bitmap loop!
                let const_bm_spec = if limit_op.is_none()
                    && let Operand::Constant(MirConstant::String(pat_lit)) = &args[1]
                    && let Operand::Constant(MirConstant::String(repl_lit)) = &args[2]
                    && !repl_lit.contains('%')
                {
                    try_compile_ascii_class_plus_bitmap(pat_lit).map(|bm| (bm, repl_lit.clone()))
                } else {
                    None
                };
                let gsub_pair_expr = if let Some((bm, repl_lit)) = &const_bm_spec {
                    format!(
                        "{{ const __BM: [u64; 4] = [{}_u64, {}_u64, 0_u64, 0_u64]; let __src: &str = {s}; let __bytes = __src.as_bytes(); let mut __out = alloc::string::String::with_capacity(__src.len()); let mut __pos = 0usize; let mut __span = 0usize; let mut __cnt = 0_i64; while __pos < __bytes.len() {{ let __b = __bytes[__pos]; if __b < 128 && (__BM[(__b >> 6) as usize] & (1_u64 << (__b & 63))) != 0 {{ if __pos > __span {{ __out.push_str(&__src[__span..__pos]); }} __pos += 1; while __pos < __bytes.len() {{ let __b2 = __bytes[__pos]; if __b2 < 128 && (__BM[(__b2 >> 6) as usize] & (1_u64 << (__b2 & 63))) != 0 {{ __pos += 1; }} else {{ break; }} }} __out.push_str({repl_lit:?}); __cnt += 1; __span = __pos; }} else {{ __pos += 1; }} }} if __span < __bytes.len() {{ __out.push_str(&__src[__span..]); }} (__out, __cnt) }}",
                        bm[0], bm[1]
                    )
                } else {
                    format!("__lust_aot_string_gsub({s}, {pat}, {repl}, {limit})?")
                };
                if self.dest_is_omitted(destination) {
                    let _ = writeln!(self.out, "{pad}let _ = {gsub_pair_expr};");
                } else if let Place::Local(dst) = destination
                    && self.gsub_first_only_locals.contains(&self.resolve_local(*dst))
                {
                    let dst_r = self.resolve_local(*dst);
                    let src_same = operand_exact_local(&args[0])
                        .is_some_and(|l| self.resolve_local(l) == dst_r);
                    if let Some((bm, repl_lit)) = &const_bm_spec
                        && self.local_storage(dst_r) == LocalStorage::Direct
                        && !src_same
                    {
                        let dst_var = self.local_var_name(dst_r);
                        let _ = writeln!(
                            self.out,
                            "{pad}{{ const __BM: [u64; 4] = [{}_u64, {}_u64, 0_u64, 0_u64]; let __src: &str = {s}; let __bytes = __src.as_bytes(); {dst_var}.clear(); let mut __pos = 0usize; let mut __span = 0usize; while __pos < __bytes.len() {{ let __b = __bytes[__pos]; if __b < 128 && (__BM[(__b >> 6) as usize] & (1_u64 << (__b & 63))) != 0 {{ if __pos > __span {{ {dst_var}.push_str(&__src[__span..__pos]); }} __pos += 1; while __pos < __bytes.len() {{ let __b2 = __bytes[__pos]; if __b2 < 128 && (__BM[(__b2 >> 6) as usize] & (1_u64 << (__b2 & 63))) != 0 {{ __pos += 1; }} else {{ break; }} }} {dst_var}.push_str({repl_lit:?}); __span = __pos; }} else {{ __pos += 1; }} }} if __span < __bytes.len() {{ {dst_var}.push_str(&__src[__span..]); }} }}",
                            bm[0], bm[1]
                        );
                    } else {
                        let expr = format!("({gsub_pair_expr}).0");
                        self.emit_assign_place_val(destination, &expr, indent);
                    }
                } else {
                    let dst_ty = self.place_mir_type(destination);
                    let expr = match &dst_ty {
                        MirType::Array(inner) if matches!(**inner, MirType::String) => format!(
                            "{{ let (__out, __cnt) = {gsub_pair_expr}; let mut __v = alloc::vec::Vec::with_capacity_in(2, __alloc); __v.push(__out); let mut __cs = alloc::string::String::new(); lust_aot_rt::string_append_int(&mut __cs, __cnt); __v.push(__cs); __v }}"
                        ),
                        MirType::String => format!("({gsub_pair_expr}).0"),
                        MirType::Unknown => format!(
                            "{{ let (__out, __cnt) = {gsub_pair_expr}; lust_aot_rt::AotValue::array(alloc::vec![lust_aot_rt::AotValue::str(__out), lust_aot_rt::AotValue::Int(__cnt)]) }}"
                        ),
                        _ => format!(
                            "{{ let (__out, __cnt) = {gsub_pair_expr}; let mut __v = alloc::vec::Vec::with_capacity_in(2, __alloc); __v.push(lust_aot_rt::AotValue::str(__out)); __v.push(lust_aot_rt::AotValue::Int(__cnt)); __v }}"
                        ),
                    };
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "string.find" if args.len() >= 2 => {
                if !self.dest_is_omitted(destination) {
                    let s = self.emit_operand_str_ref(&args[0]);
                    let pat = self.emit_operand_str_ref(&args[1]);
                    let start = args
                        .get(2)
                        .map(|a| self.emit_operand_owned(a, &MirType::Int))
                        .unwrap_or_else(|| "1_i64".to_string());
                    let plain = args
                        .get(3)
                        .map(|a| self.emit_operand_bool(a))
                        .unwrap_or_else(|| "false".to_string());
                    let dst_ty = self.place_mir_type(destination);
                    let expr = match &dst_ty {
                        MirType::Array(inner) if matches!(**inner, MirType::Int) => format!(
                            "match __lust_aot_string_find({s}, {pat}, {start}, {plain})? {{ Some((__a, __b, _)) => {{ let mut __v = alloc::vec::Vec::new_in(__alloc); __v.push(__a); __v.push(__b); __v }}, None => alloc::vec::Vec::new_in(__alloc) }}"
                        ),
                        MirType::Int => format!(
                            "__lust_aot_string_find({s}, {pat}, {start}, {plain})?.map(|(__a, _, _)| __a).unwrap_or(-1_i64)"
                        ),
                        MirType::Option(_) => format!(
                            "__lust_aot_string_find({s}, {pat}, {start}, {plain})?.map(|(__a, _, _)| __a)"
                        ),
                        MirType::Unknown => format!(
                            "match __lust_aot_string_find({s}, {pat}, {start}, {plain})? {{ Some((__a, __b, __caps)) => {{ let mut __v = alloc::vec![lust_aot_rt::AotValue::Int(__a), lust_aot_rt::AotValue::Int(__b)]; for __c in __caps {{ __v.push(lust_aot_rt::AotValue::str(__c)); }} lust_aot_rt::AotValue::array(__v) }}, None => lust_aot_rt::AotValue::Nil }}"
                        ),
                        _ => format!(
                            "match __lust_aot_string_find({s}, {pat}, {start}, {plain})? {{ Some((__a, __b, __caps)) => {{ let mut __v = alloc::vec::Vec::new_in(__alloc); __v.push(lust_aot_rt::AotValue::Int(__a)); __v.push(lust_aot_rt::AotValue::Int(__b)); for __c in __caps {{ __v.push(lust_aot_rt::AotValue::str(__c)); }} __v }}, None => {{ let mut __v = alloc::vec::Vec::new_in(__alloc); __v.push(lust_aot_rt::AotValue::Nil); __v }} }}"
                        ),
                    };
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "__mir_unknown_is_int" if !args.is_empty() => {
                if !self.dest_is_omitted(destination) {
                    let v = self.emit_operand_as_aot_value(&args[0]);
                    let expr = format!("matches!(&({v}), lust_aot_rt::AotValue::Int(_))");
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "__mir_unknown_is_float" if !args.is_empty() => {
                if !self.dest_is_omitted(destination) {
                    let v = self.emit_operand_as_aot_value(&args[0]);
                    let expr = format!("matches!(&({v}), lust_aot_rt::AotValue::Float(_))");
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "__mir_unknown_is_string" if !args.is_empty() => {
                if !self.dest_is_omitted(destination) {
                    let v = self.emit_operand_as_aot_value(&args[0]);
                    let expr = format!("matches!(&({v}), lust_aot_rt::AotValue::Str(_))");
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "__mir_unknown_is_bool" if !args.is_empty() => {
                if !self.dest_is_omitted(destination) {
                    let v = self.emit_operand_as_aot_value(&args[0]);
                    let expr = format!("matches!(&({v}), lust_aot_rt::AotValue::Bool(_))");
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "__mir_unknown_unwrap_int" if !args.is_empty() => {
                if !self.dest_is_omitted(destination) {
                    let v = self.emit_operand_as_aot_value(&args[0]);
                    let expr = format!(
                        "({v}).as_int().ok_or_else(|| lust_aot_rt::AotError::UserError(alloc::string::String::from(\"expected int\")))?"
                    );
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "__mir_unknown_unwrap_float" if !args.is_empty() => {
                if !self.dest_is_omitted(destination) {
                    let v = self.emit_operand_as_aot_value(&args[0]);
                    let expr = format!("({v}).as_float_coerced()?");
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "__mir_unknown_unwrap_string" if !args.is_empty() => {
                if !self.dest_is_omitted(destination) {
                    let v = self.emit_operand_as_aot_value(&args[0]);
                    let expr = format!(
                        "match &({v}) {{ lust_aot_rt::AotValue::Str(__s) => (**__s).clone(), _ => return Err(lust_aot_rt::AotError::UserError(alloc::string::String::from(\"expected string\"))) }}"
                    );
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "__mir_unknown_unwrap_bool" if !args.is_empty() => {
                if !self.dest_is_omitted(destination) {
                    let v = self.emit_operand_as_aot_value(&args[0]);
                    let expr = format!(
                        "({v}).as_bool().ok_or_else(|| lust_aot_rt::AotError::UserError(alloc::string::String::from(\"expected bool\")))?"
                    );
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "__mir_unknown_get_field" if args.len() >= 2 => {
                // Devirtualize statically-known concrete struct field access (`AOT-DEVIRT`: 0 heap allocs)
                if let MirType::Struct(sname) = operand_mir_type(&args[0], self.func)
                    && let Operand::Constant(MirConstant::String(fname)) = &args[1]
                    && let Some(smeta) = self.ctx.structs.get(&sname)
                    && let Some((fidx, (_, fty))) =
                        smeta.fields.iter().enumerate().find(|(_, (n, _))| n == fname)
                {
                    if !self.dest_is_omitted(destination) {
                        let recv = self.emit_operand_value(&args[0]);
                        let raw = if smeta.is_mutated_cyclic {
                            if self.ctx.is_copy_type(fty) {
                                format!("lust_aot_rt::ref_borrow(&({recv}).0)?.field_{fidx}")
                            } else {
                                format!("lust_aot_rt::ref_borrow(&({recv}).0)?.field_{fidx}.clone()")
                            }
                        } else if self.ctx.is_copy_type(fty) {
                            format!("({recv}).field_{fidx}")
                        } else {
                            format!("({recv}).field_{fidx}.clone()")
                        };
                        let dst_ty = self.place_mir_type(destination);
                        let expr = self.coerce_expr_to_type(&raw, fty, &dst_ty);
                        self.emit_assign_place_val(destination, &expr, indent);
                    }
                    return Ok(());
                }
                let obj = self.emit_operand_as_aot_value(&args[0]);
                let field = self.emit_operand_str_ref(&args[1]);
                let raw = format!("__lust_aot_unknown_get_field(&({obj}), {field})?");
                if self.dest_is_omitted(destination) {
                    let _ = writeln!(self.out, "{pad}let _ = {raw};");
                } else {
                    let dst_ty = self.place_mir_type(destination);
                    let expr = self.coerce_expr_to_type(&raw, &MirType::Unknown, &dst_ty);
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "__mir_unknown_call_method" if args.len() >= 2 => {
                // Direct unboxed `AotQueue<i64>` method dispatch (`AOT-DEVIRT`)
                if let Some(q_raw) = operand_exact_local(&args[0])
                    && self.unboxed_queue_locals.contains(&self.resolve_local(q_raw))
                    && let Operand::Constant(MirConstant::String(mname)) = &args[1]
                {
                    let q_loc = self.resolve_local(q_raw);
                    self.emit_unboxed_queue_method_call(bb, q_loc, mname, &args[2..], destination, indent);
                    return Ok(());
                }
                // Devirtualize statically-known concrete struct method call (`AOT-DEVIRT`: 0 heap allocs)
                if let MirType::Struct(sname) = operand_mir_type(&args[0], self.func)
                    && let Operand::Constant(MirConstant::String(mname)) = &args[1]
                {
                    let full_method = format!("{sname}:{mname}");
                    if let Some(fmeta) = self.ctx.fn_metas.get(&full_method).cloned()
                        && let Some(mfn) = self
                            .ctx
                            .module
                            .functions
                            .iter()
                            .find(|f| f.signature.name == full_method)
                    {
                        let mut cargs = Vec::new();
                        if fmeta.needs_ctx {
                            cargs.push("__ctx".to_string());
                        }
                        if fmeta.needs_alloc {
                            cargs.push("__alloc".to_string());
                        }
                        if fmeta.needs_arena {
                            cargs.push("__arena".to_string());
                        }
                        let self_mode = fmeta.param_modes.first().copied().unwrap_or(ParamMode::ByValue);
                        let self_arg = match self_mode {
                            ParamMode::RefStruct => self.emit_operand_ref_borrowed(&args[0]),
                            ParamMode::MutRefStruct => self.emit_operand_mut_ref_borrowed(&args[0]),
                            _ => self.emit_operand_owned(&args[0], &MirType::Struct(sname.clone())),
                        };
                        cargs.push(self_arg);
                        for (pi, &pid) in mfn.params.iter().enumerate().skip(1) {
                            let pty = &mfn.locals[pid.0].ty;
                            let pmode = fmeta.param_modes.get(pi).copied().unwrap_or(ParamMode::ByValue);
                            let arg_op = args.get(pi + 1);
                            let a_expr = match arg_op {
                                Some(op) => match pmode {
                                    ParamMode::RefStr => self.emit_operand_str_ref(op),
                                    ParamMode::RefStruct => self.emit_operand_ref_borrowed(op),
                                    ParamMode::MutRefStruct => self.emit_operand_mut_ref_borrowed(op),
                                    ParamMode::RefSlice => self.emit_operand_slice_ref(op),
                                    _ => self.emit_operand_owned(op, pty),
                                },
                                None => self.ctx.rust_default_expr(pty),
                            };
                            cargs.push(a_expr);
                        }
                        let raw_call = format!("{}({})", fmeta.rust_name, cargs.join(", "));
                        let call_res = if fmeta.can_fail {
                            format!("{raw_call}?")
                        } else {
                            format!("match {raw_call} {{ Ok(__v) => __v }}")
                        };
                        if self.dest_is_omitted(destination)
                            || matches!(self.place_mir_type(destination), MirType::Unit)
                        {
                            let _ = writeln!(self.out, "{pad}let _ = {call_res};");
                        } else {
                            let dst_ty = self.place_mir_type(destination);
                            let expr = self.coerce_expr_to_type(&call_res, &fmeta.return_type, &dst_ty);
                            self.emit_assign_place_val(destination, &expr, indent);
                        }
                        return Ok(());
                    }
                }
                let obj = self.emit_operand_as_aot_value(&args[0]);
                let method = self.emit_operand_str_ref(&args[1]);
                let rest: Vec<String> = args[2..]
                    .iter()
                    .map(|a| self.emit_operand_as_aot_value(a))
                    .collect();
                let arena_arg = if self.ctx.module_uses_arena {
                    ", __arena"
                } else {
                    ""
                };
                let raw = format!(
                    "__lust_aot_unknown_call_method(__ctx, __alloc{arena_arg}, &({obj}), {method}, &[{}])?",
                    rest.join(", ")
                );
                if self.dest_is_omitted(destination)
                    || matches!(self.place_mir_type(destination), MirType::Unit)
                {
                    let _ = writeln!(self.out, "{pad}let _ = {raw};");
                } else {
                    let dst_ty = self.place_mir_type(destination);
                    let expr = self.coerce_expr_to_type(&raw, &MirType::Unknown, &dst_ty);
                    self.emit_assign_place_val(destination, &expr, indent);
                }
                return Ok(());
            }
            "Queue" | "sync.Queue" | "@std/sync.Queue" => {
                let cap = args
                    .first()
                    .map(|a| self.emit_operand_owned(a, &MirType::Int))
                    .unwrap_or_else(|| "64_i64".to_string());
                let strat = args
                    .get(1)
                    .map(|a| self.emit_operand_str_ref(a))
                    .unwrap_or_else(|| "\"suspend\"".to_string());
                if let Place::Local(dst) = destination
                    && self.unboxed_queue_locals.contains(&self.resolve_local(*dst))
                {
                    let r = self.resolve_local(*dst);
                    let q_var = self.local_var_name(r);
                    let _ = writeln!(
                        self.out,
                        "{pad}{q_var} = lust_aot_rt::AotQueue::<i64>::with_capacity_and_strategy(({cap}).max(0) as usize, {strat});"
                    );
                    return Ok(());
                }
                let raw = format!("__lust_aot_new_queue({cap}, {strat}, __ctx)?");
                if self.dest_is_omitted(destination) {
                    let _ = writeln!(self.out, "{pad}let _ = {raw};");
                } else {
                    let dst_ty = self.place_mir_type(destination);
                    let expr = self.coerce_expr_to_type(&raw, &MirType::Unknown, &dst_ty);
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
            "__lust_handle" => {
                self.emit_effect_handle_call(bb, args, destination, indent)?;
                return Ok(());
            }
            "__lust_nontail_handle" => {
                self.emit_nontail_handle_call(bb, args, destination, indent)?;
                return Ok(());
            }
            "__lust_yield_completed" => {
                self.emit_nontail_yield_completed_call(args, indent)?;
                return Ok(());
            }
            "__lust_yield_op" => {
                self.emit_nontail_yield_op_call(bb, indent)?;
                return Ok(());
            }
            "__lust_nontail_resume" => {
                self.emit_nontail_resume_call(args, destination, indent)?;
                return Ok(());
            }
            "__lust_perform" | "perform" => {
                self.emit_effect_perform_call(bb, args, destination, indent)?;
                return Ok(());
            }
            "__lust_resume" | "resume" => {
                self.emit_effect_resume_call(args, destination, indent)?;
                return Ok(());
            }
            "__lust_provide" => {
                self.emit_effect_provide_call(args, destination, indent)?;
                return Ok(());
            }
            "__lust_run_sync" => {
                self.emit_effect_run_sync_call(bb, args, destination, indent)?;
                return Ok(());
            }
            "__lust_scoped" => {
                self.emit_scoped_call(bb, indent)?;
                return Ok(());
            }
            "__lust_scope_finalizer" => {
                self.emit_scope_finalizer_call(bb, indent)?;
                return Ok(());
            }
            "__lust_scope_spawn" => {
                self.emit_scope_spawn_call(bb, indent)?;
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
            if let Some(&(q_loc, method_name)) = self.unboxed_queue_methods.get(&cl_loc) {
                self.emit_unboxed_queue_method_call(bb, q_loc, method_name, args, destination, indent);
                return Ok(());
            }
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

        // Direct Zero-FFI external Rust crate call (`--backend rust --link-crate`, NG-W13-1C)
        if !self.ctx.fn_metas.contains_key(callee)
            && let Some(ext_call) = self.ctx.resolve_extern_crate_call(callee)
        {
            let mut ext_args: Vec<String> = Vec::with_capacity(args.len());
            for (i, arg) in args.iter().enumerate() {
                let expected = ext_call
                    .param_types
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| operand_mir_type(arg, self.func));
                let a_expr = match expected {
                    MirType::String => self.emit_operand_str_ref(arg),
                    MirType::Array(_) => self.emit_operand_slice_ref(arg),
                    _ => self.emit_operand_owned(arg, &expected),
                };
                ext_args.push(a_expr);
            }
            let call_expr = format!("{}({})", ext_call.rust_path, ext_args.join(", "));
            if self.dest_is_omitted(destination) || ext_call.return_type == MirType::Unit {
                let _ = writeln!(self.out, "{pad}let _ = {call_expr};");
            } else {
                let dst_ty = self.place_mir_type(destination);
                let final_expr =
                    self.coerce_expr_to_type(&call_expr, &ext_call.return_type, &dst_ty);
                self.emit_assign_place_val(destination, &final_expr, indent);
            }
            return Ok(());
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

    fn build_lifted_capture_bindings(
        &self,
        func_name: &str,
        captures: &[(String, Operand, MirType)],
        prefix: &str,
    ) -> Result<(Vec<String>, Vec<String>)> {
        let Some(lifted_meta) = self.ctx.fn_metas.get(func_name) else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: missing lifted effect function '{func_name}'"
            )));
        };
        let mut bindings = Vec::with_capacity(captures.len());
        let mut call_exprs = Vec::with_capacity(captures.len());
        for (ci, (_, cap_op, cap_ty)) in captures.iter().enumerate() {
            let lmode = lifted_meta
                .param_modes
                .get(ci)
                .copied()
                .unwrap_or(ParamMode::ByValue);
            let var_name = format!("{prefix}_{ci}");
            match lmode {
                ParamMode::RcCell | ParamMode::RcRefCell => {
                    if let Some(loc) = operand_exact_local(cap_op)
                        && self.local_storage(loc) != LocalStorage::Direct
                    {
                        let src_var = self.local_var_name(loc);
                        bindings.push(format!("let {var_name} = alloc::rc::Rc::clone(&{src_var});"));
                    } else {
                        let v = self.emit_operand_owned(cap_op, cap_ty);
                        if lmode == ParamMode::RcCell {
                            bindings.push(format!(
                                "let {var_name} = alloc::rc::Rc::new(core::cell::Cell::new({v}));"
                            ));
                        } else {
                            bindings.push(format!(
                                "let {var_name} = alloc::rc::Rc::new(core::cell::RefCell::new({v}));"
                            ));
                        }
                    }
                    call_exprs.push(format!("alloc::rc::Rc::clone(&{var_name})"));
                }
                ParamMode::RefStruct => {
                    let v = self.emit_operand_owned(cap_op, cap_ty);
                    bindings.push(format!("let {var_name} = {v};"));
                    call_exprs.push(format!("&{var_name}"));
                }
                ParamMode::MutRefStruct => {
                    let v = self.emit_operand_owned(cap_op, cap_ty);
                    bindings.push(format!("let mut {var_name} = {v};"));
                    call_exprs.push(format!("&mut {var_name}"));
                }
                ParamMode::RefSlice => {
                    let v = self.emit_operand_owned(cap_op, cap_ty);
                    bindings.push(format!("let {var_name} = {v};"));
                    call_exprs.push(format!("&{var_name}[..]"));
                }
                ParamMode::MutVec => {
                    let v = self.emit_operand_owned(cap_op, cap_ty);
                    bindings.push(format!("let mut {var_name} = {v};"));
                    call_exprs.push(format!("&mut {var_name}"));
                }
                ParamMode::RefStr => {
                    let v = self.emit_operand_owned(cap_op, cap_ty);
                    bindings.push(format!("let {var_name} = {v};"));
                    call_exprs.push(format!("{var_name}.as_str()"));
                }
                ParamMode::ByValue => {
                    let v = self.emit_operand_owned(cap_op, cap_ty);
                    bindings.push(format!("let {var_name} = {v};"));
                    if self.ctx.is_copy_type(cap_ty) {
                        call_exprs.push(var_name);
                    } else {
                        call_exprs.push(format!("{var_name}.clone()"));
                    }
                }
            }
        }
        Ok((bindings, call_exprs))
    }

    fn emit_effect_handle_call(
        &mut self,
        bb: BlockId,
        args: &[Operand],
        destination: &Place,
        indent: usize,
    ) -> Result<()> {
        let pad = "    ".repeat(indent);
        let Some(plan) = self.handle_plans.get(&bb).cloned() else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: unsupported dynamic __lust_handle call at block {bb:?} (args: {})",
                args.len()
            )));
        };

        let Some(body_meta) = self.ctx.fn_metas.get(&plan.body_fn_name).cloned() else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: missing lifted handle body '{}'",
                plan.body_fn_name
            )));
        };

        let all_ast_tail = self
            .ctx
            .ast_meta
            .handle_sites
            .iter()
            .all(|s| s.all_tail_or_abortive);
        let _ = writeln!(
            self.out,
            "{pad}// Tail-resumptive handle..with (ast_tail_verified={all_ast_tail}) -> stack EffectHandlerFrame + DropGuard (NG-W13-1A)"
        );
        let _ = writeln!(self.out, "{pad}let __handle_res = {{");
        let _ = writeln!(
            self.out,
            "{pad}    let mut __eff_arms: alloc::vec::Vec<lust_aot_rt::EffectHandlerArm> = alloc::vec::Vec::with_capacity({});",
            plan.arms.len()
        );

        for (ai, arm) in plan.arms.iter().enumerate() {
            let Some(arm_meta) = self.ctx.fn_metas.get(&arm.arm_fn_name).cloned() else {
                return Err(LustError::Unknown(format!(
                    "Rust AOT emitter: missing lifted handler arm '{}'",
                    arm.arm_fn_name
                )));
            };
            let Some(arm_fn) = self
                .ctx
                .module
                .functions
                .iter()
                .find(|f| f.signature.name == arm.arm_fn_name)
            else {
                return Err(LustError::Unknown(format!(
                    "Rust AOT emitter: missing lifted handler arm MirFunction '{}'",
                    arm.arm_fn_name
                )));
            };

            let (cap_bindings, cap_call_exprs) = self.build_lifted_capture_bindings(
                &arm.arm_fn_name,
                &arm.captures,
                &format!("__arm_{ai}_cap"),
            )?;
            let n_caps = arm.captures.len();
            let explicit_params: Vec<MirType> = arm_fn
                .params
                .iter()
                .skip(n_caps)
                .map(|pid| arm_fn.locals[pid.0].ty.clone())
                .collect();

            let mut arm_call_args = Vec::new();
            if arm_meta.needs_ctx {
                arm_call_args.push("__e_ctx".to_string());
            }
            if arm_meta.needs_alloc {
                arm_call_args.push("&__e_alloc".to_string());
            }
            if arm_meta.needs_arena {
                arm_call_args.push("&__e_arena".to_string());
            }
            arm_call_args.extend(cap_call_exprs);

            let mut arg_bindings = Vec::new();
            for (pi, pty) in explicit_params.iter().enumerate() {
                let raw_a = format!(
                    "__e_args.get({pi}).cloned().unwrap_or(lust_aot_rt::AotValue::Nil)"
                );
                let coerced = self.coerce_expr_to_type(&raw_a, &MirType::Unknown, pty);
                let pmode = arm_meta
                    .param_modes
                    .get(n_caps + pi)
                    .copied()
                    .unwrap_or(ParamMode::ByValue);
                match pmode {
                    ParamMode::RefStr => {
                        arg_bindings.push(format!("let __ea_{pi} = {coerced};"));
                        arm_call_args.push(format!("__ea_{pi}.as_str()"));
                    }
                    ParamMode::RefStruct => {
                        arg_bindings.push(format!("let __ea_{pi} = {coerced};"));
                        arm_call_args.push(format!("&__ea_{pi}"));
                    }
                    ParamMode::MutRefStruct => {
                        arg_bindings.push(format!("let mut __ea_{pi} = {coerced};"));
                        arm_call_args.push(format!("&mut __ea_{pi}"));
                    }
                    ParamMode::RcCell => {
                        arm_call_args.push(format!(
                            "alloc::rc::Rc::new(core::cell::Cell::new({coerced}))"
                        ));
                    }
                    ParamMode::RcRefCell => {
                        arm_call_args.push(format!(
                            "alloc::rc::Rc::new(core::cell::RefCell::new({coerced}))"
                        ));
                    }
                    _ => {
                        arm_call_args.push(coerced);
                    }
                }
            }

            let raw_arm_call = format!("{}({})", arm_meta.rust_name, arm_call_args.join(", "));
            let arm_ret_aot =
                self.coerce_expr_to_type("__arm_out", &arm_meta.return_type, &MirType::Unknown);

            let _ = writeln!(self.out, "{pad}    {{");
            for b in &cap_bindings {
                let _ = writeln!(self.out, "{pad}        {b}");
            }
            let _ = writeln!(
                self.out,
                "{pad}        __eff_arms.push(lust_aot_rt::EffectHandlerArm::from_fn({:?}, move |__e_ctx: &mut lust_aot_rt::AotCtx, __e_args: &[lust_aot_rt::AotValue]| -> Result<lust_aot_rt::AotValue, lust_aot_rt::AotError> {{",
                arm.effect_name
            );
            if arm_meta.needs_alloc || arm_meta.needs_arena {
                let _ = writeln!(
                    self.out,
                    "{pad}            let __e_alloc = __e_ctx.budget_alloc.share();"
                );
            }
            if arm_meta.needs_arena {
                let _ = writeln!(
                    self.out,
                    "{pad}            let __e_arena = lust_aot_rt::LustChunkArena::new(&__e_alloc);"
                );
            }
            for ab in &arg_bindings {
                let _ = writeln!(self.out, "{pad}            {ab}");
            }
            let _ = writeln!(
                self.out,
                "{pad}            let __arm_out = {raw_arm_call}.map_err(lust_aot_rt::AotError::from)?;"
            );
            let _ = writeln!(self.out, "{pad}            Ok({arm_ret_aot})");
            let _ = writeln!(self.out, "{pad}        }}));");
            let _ = writeln!(self.out, "{pad}    }}");
        }

        let (body_cap_bindings, body_cap_exprs) = self.build_lifted_capture_bindings(
            &plan.body_fn_name,
            &plan.body_captures,
            "__body_cap",
        )?;
        for b in &body_cap_bindings {
            let _ = writeln!(self.out, "{pad}    {b}");
        }

        let mut body_call_args = Vec::new();
        if body_meta.needs_ctx {
            body_call_args.push("__h_ctx".to_string());
        }
        if body_meta.needs_alloc {
            body_call_args.push("__alloc".to_string());
        }
        if body_meta.needs_arena {
            body_call_args.push("__arena".to_string());
        }
        body_call_args.extend(body_cap_exprs);

        let body_ret_ty = body_meta.return_type.clone();
        let body_ret_rust = self.ctx.rust_type(&body_ret_ty, "'a");
        let raw_body_call = format!("{}({})", body_meta.rust_name, body_call_args.join(", "));
        let abort_conv = self.coerce_expr_to_type("__abort_val", &MirType::Unknown, &body_ret_ty);

        let _ = writeln!(
            self.out,
            "{pad}    lust_aot_rt::handle_tail_resumptive_typed("
        );
        let _ = writeln!(self.out, "{pad}        __ctx,");
        let _ = writeln!(self.out, "{pad}        __eff_arms,");
        let _ = writeln!(
            self.out,
            "{pad}        |__h_ctx: &mut lust_aot_rt::AotCtx| -> Result<{body_ret_rust}, lust_aot_rt::AotError> {{"
        );
        let _ = writeln!(
            self.out,
            "{pad}            {raw_body_call}.map_err(lust_aot_rt::AotError::from)"
        );
        let _ = writeln!(self.out, "{pad}        }},");
        let _ = writeln!(
            self.out,
            "{pad}        |__abort_val: lust_aot_rt::AotValue| -> Result<{body_ret_rust}, lust_aot_rt::AotError> {{"
        );
        let _ = writeln!(self.out, "{pad}            Ok({abort_conv})");
        let _ = writeln!(self.out, "{pad}        }},");
        let _ = writeln!(self.out, "{pad}    )?");
        let _ = writeln!(self.out, "{pad}}};");

        if !self.dest_is_omitted(destination) && !matches!(self.place_mir_type(destination), MirType::Unit) {
            let dst_ty = self.place_mir_type(destination);
            let final_expr = self.coerce_expr_to_type("__handle_res", &body_ret_ty, &dst_ty);
            self.emit_assign_place_val(destination, &final_expr, indent);
        }
        Ok(())
    }

    fn emit_nontail_handle_call(
        &mut self,
        bb: BlockId,
        args: &[Operand],
        destination: &Place,
        indent: usize,
    ) -> Result<()> {
        let pad = "    ".repeat(indent);
        let Some(plan) = self.nontail_handle_plans.get(&bb).cloned() else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: unsupported dynamic __lust_nontail_handle call at block {bb:?} (args: {})",
                args.len()
            )));
        };
        let Some(body_meta) = self.ctx.fn_metas.get(&plan.body_fn_name).cloned() else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: missing lifted non-tail handle body '{}'",
                plan.body_fn_name
            )));
        };
        let Some(ysig) = self.ctx.nontail_body_metas.get(&plan.body_fn_name).cloned() else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: missing NonTailYieldSig for '{}'",
                plan.body_fn_name
            )));
        };

        let op_arg_rust = self.ctx.nontail_op_arg_rust_type(&ysig.op_arg_mode);
        let resume_rust = self.ctx.rust_type(&ysig.resume_val_ty, "'static");
        let body_ret_rust = self.ctx.rust_type(&ysig.body_ret_ty, "'static");

        let _ = writeln!(
            self.out,
            "{pad}// One-shot non-tail handle..with -> EffectYield<{op_arg_rust}, {resume_rust}, {body_ret_rust}> state-machine (NG-W13-1B)"
        );
        let _ = writeln!(self.out, "{pad}let __handle_res: {body_ret_rust} = {{");

        let (body_cap_bindings, body_cap_exprs) = self.build_lifted_capture_bindings(
            &plan.body_fn_name,
            &plan.body_captures,
            "__nt_body_cap",
        )?;
        for b in &body_cap_bindings {
            let _ = writeln!(self.out, "{pad}    {b}");
        }

        let mut arm_cap_sets: Vec<(Vec<String>, Vec<String>)> = Vec::with_capacity(plan.arms.len());
        for (ai, arm) in plan.arms.iter().enumerate() {
            let (cb, ce) = self.build_lifted_capture_bindings(
                &arm.arm_fn_name,
                &arm.captures,
                &format!("__nt_arm_{ai}_cap"),
            )?;
            for b in &cb {
                let _ = writeln!(self.out, "{pad}    {b}");
            }
            arm_cap_sets.push((cb, ce));
        }

        let mut body_call_args = Vec::new();
        if body_meta.needs_ctx {
            body_call_args.push("__nt_ctx".to_string());
        }
        if body_meta.needs_alloc {
            body_call_args.push("__alloc".to_string());
        }
        if body_meta.needs_arena {
            body_call_args.push("__arena".to_string());
        }
        body_call_args.extend(body_cap_exprs);
        let raw_body_call = format!("{}({})", body_meta.rust_name, body_call_args.join(", "));

        let _ = writeln!(
            self.out,
            "{pad}    lust_aot_rt::handle_non_tail_oneshot::<{op_arg_rust}, {resume_rust}, {body_ret_rust}, _, _>("
        );
        let _ = writeln!(self.out, "{pad}        __ctx,");
        let _ = writeln!(
            self.out,
            "{pad}        |__nt_ctx: &mut lust_aot_rt::AotCtx| -> Result<lust_aot_rt::EffectYield<{op_arg_rust}, {resume_rust}, {body_ret_rust}>, lust_aot_rt::AotError> {{"
        );
        let _ = writeln!(
            self.out,
            "{pad}            {raw_body_call}.map_err(lust_aot_rt::AotError::from)"
        );
        let _ = writeln!(self.out, "{pad}        }},");
        let _ = writeln!(
            self.out,
            "{pad}        |__h_ctx: &mut lust_aot_rt::AotCtx, __op_arg: {op_arg_rust}, __resume_fn: alloc::boxed::Box<dyn FnOnce(&mut lust_aot_rt::AotCtx, {resume_rust}) -> Result<{body_ret_rust}, lust_aot_rt::AotError>>| -> Result<{body_ret_rust}, lust_aot_rt::AotError> {{"
        );
        let _ = writeln!(
            self.out,
            "{pad}            let mut __k = lust_aot_rt::OneShotResume::new(__resume_fn);"
        );

        if matches!(ysig.op_arg_mode, NonTailOpArgMode::TaggedVec) {
            let _ = writeln!(self.out, "{pad}            match __op_arg.0 {{");
            for (ai, arm) in plan.arms.iter().enumerate() {
                let _ = writeln!(self.out, "{pad}                {:?} => {{", arm.effect_name);
                self.emit_nontail_arm_invoke(
                    arm,
                    &arm_cap_sets[ai].1,
                    &ysig,
                    &format!("{pad}                    "),
                )?;
                let _ = writeln!(self.out, "{pad}                }}");
            }
            let _ = writeln!(
                self.out,
                "{pad}                __unknown_op => Err(lust_aot_rt::AotError::UnhandledEffect(alloc::string::String::from(__unknown_op))),"
            );
            let _ = writeln!(self.out, "{pad}            }}");
        } else if let Some(arm) = plan.arms.first() {
            self.emit_nontail_arm_invoke(
                arm,
                &arm_cap_sets[0].1,
                &ysig,
                &format!("{pad}            "),
            )?;
        }

        let _ = writeln!(self.out, "{pad}        }},");
        let _ = writeln!(self.out, "{pad}    )?");
        let _ = writeln!(self.out, "{pad}}};");

        if !self.dest_is_omitted(destination)
            && !matches!(self.place_mir_type(destination), MirType::Unit)
        {
            let dst_ty = self.place_mir_type(destination);
            let final_expr = self.coerce_expr_to_type("__handle_res", &ysig.body_ret_ty, &dst_ty);
            self.emit_assign_place_val(destination, &final_expr, indent);
        }
        Ok(())
    }

    fn emit_nontail_arm_invoke(
        &mut self,
        arm: &HandleArmPlan,
        cap_exprs: &[String],
        ysig: &NonTailYieldSig,
        arm_pad: &str,
    ) -> Result<()> {
        let Some(arm_meta) = self.ctx.fn_metas.get(&arm.arm_fn_name).cloned() else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: missing lifted non-tail arm '{}'",
                arm.arm_fn_name
            )));
        };
        let Some(arm_fn) = self
            .ctx
            .module
            .functions
            .iter()
            .find(|f| f.signature.name == arm.arm_fn_name)
        else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: missing lifted non-tail arm MirFunction '{}'",
                arm.arm_fn_name
            )));
        };

        let n_caps = arm.captures.len();
        let explicit_params: Vec<MirType> = arm_fn
            .params
            .iter()
            .skip(n_caps)
            .map(|pid| arm_fn.locals[pid.0].ty.clone())
            .collect();

        if arm_meta.needs_alloc || arm_meta.needs_arena {
            let _ = writeln!(
                self.out,
                "{arm_pad}let __h_alloc = __h_ctx.budget_alloc.share();"
            );
        }
        if arm_meta.needs_arena {
            let _ = writeln!(
                self.out,
                "{arm_pad}let __h_arena = lust_aot_rt::LustChunkArena::new(&__h_alloc);"
            );
        }

        let mut arm_call_args = Vec::new();
        if arm_meta.needs_ctx {
            arm_call_args.push("__h_ctx".to_string());
        }
        if arm_meta.needs_alloc {
            arm_call_args.push("&__h_alloc".to_string());
        }
        if arm_meta.needs_arena {
            arm_call_args.push("&__h_arena".to_string());
        }
        arm_call_args.push("&mut __k".to_string());
        arm_call_args.extend(cap_exprs.iter().cloned());

        match &ysig.op_arg_mode {
            NonTailOpArgMode::Unit => {}
            NonTailOpArgMode::Single(arg_ty) => {
                if let Some(pty) = explicit_params.first() {
                    let coerced = self.coerce_expr_to_type("__op_arg", arg_ty, pty);
                    let pmode = arm_meta
                        .param_modes
                        .get(n_caps)
                        .copied()
                        .unwrap_or(ParamMode::ByValue);
                    match pmode {
                        ParamMode::RefStr => {
                            let _ = writeln!(self.out, "{arm_pad}let __ea_0 = {coerced};");
                            arm_call_args.push("__ea_0.as_str()".to_string());
                        }
                        ParamMode::RefStruct => {
                            let _ = writeln!(self.out, "{arm_pad}let __ea_0 = {coerced};");
                            arm_call_args.push("&__ea_0".to_string());
                        }
                        ParamMode::MutRefStruct => {
                            let _ = writeln!(self.out, "{arm_pad}let mut __ea_0 = {coerced};");
                            arm_call_args.push("&mut __ea_0".to_string());
                        }
                        ParamMode::RcCell => {
                            arm_call_args.push(format!(
                                "alloc::rc::Rc::new(core::cell::Cell::new({coerced}))"
                            ));
                        }
                        ParamMode::RcRefCell => {
                            arm_call_args.push(format!(
                                "alloc::rc::Rc::new(core::cell::RefCell::new({coerced}))"
                            ));
                        }
                        _ => arm_call_args.push(coerced),
                    }
                }
            }
            NonTailOpArgMode::TaggedVec => {
                for (pi, pty) in explicit_params.iter().enumerate() {
                    let raw_a = format!(
                        "__op_arg.1.get({pi}).cloned().unwrap_or(lust_aot_rt::AotValue::Nil)"
                    );
                    let coerced = self.coerce_expr_to_type(&raw_a, &MirType::Unknown, pty);
                    let pmode = arm_meta
                        .param_modes
                        .get(n_caps + pi)
                        .copied()
                        .unwrap_or(ParamMode::ByValue);
                    match pmode {
                        ParamMode::RefStr => {
                            let _ = writeln!(self.out, "{arm_pad}let __ea_{pi} = {coerced};");
                            arm_call_args.push(format!("__ea_{pi}.as_str()"));
                        }
                        ParamMode::RefStruct => {
                            let _ = writeln!(self.out, "{arm_pad}let __ea_{pi} = {coerced};");
                            arm_call_args.push(format!("&__ea_{pi}"));
                        }
                        ParamMode::MutRefStruct => {
                            let _ = writeln!(self.out, "{arm_pad}let mut __ea_{pi} = {coerced};");
                            arm_call_args.push(format!("&mut __ea_{pi}"));
                        }
                        ParamMode::RcCell => {
                            arm_call_args.push(format!(
                                "alloc::rc::Rc::new(core::cell::Cell::new({coerced}))"
                            ));
                        }
                        ParamMode::RcRefCell => {
                            arm_call_args.push(format!(
                                "alloc::rc::Rc::new(core::cell::RefCell::new({coerced}))"
                            ));
                        }
                        _ => arm_call_args.push(coerced),
                    }
                }
            }
        }

        let raw_arm_call = format!("{}({})", arm_meta.rust_name, arm_call_args.join(", "));
        let ret_coerced =
            self.coerce_expr_to_type("__arm_out", &arm_meta.return_type, &ysig.body_ret_ty);
        let _ = writeln!(
            self.out,
            "{arm_pad}let __arm_out = {raw_arm_call}.map_err(lust_aot_rt::AotError::from)?;"
        );
        let _ = writeln!(self.out, "{arm_pad}Ok({ret_coerced})");
        Ok(())
    }

    fn emit_nontail_yield_completed_call(&mut self, args: &[Operand], indent: usize) -> Result<()> {
        let pad = "    ".repeat(indent);
        let Some(ysig) = self.ctx.nontail_body_metas.get(&self.func.signature.name).cloned() else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: __lust_yield_completed outside non-tail body '{}'",
                self.func.signature.name
            )));
        };
        let val_expr = match args.first() {
            Some(op) => self.emit_operand_owned(op, &ysig.body_ret_ty),
            None => self.ctx.rust_default_expr(&ysig.body_ret_ty),
        };
        if !self.errdefer_guards.is_empty() {
            let _ = writeln!(self.out, "{pad}let __yc_ret = {val_expr};");
            for g in self.errdefer_guards.clone().iter().rev() {
                let _ = writeln!(self.out, "{pad}core::mem::DropGuard::dismiss({g});");
            }
            let _ = writeln!(
                self.out,
                "{pad}return Ok(lust_aot_rt::EffectYield::Completed(__yc_ret));"
            );
        } else {
            let _ = writeln!(
                self.out,
                "{pad}return Ok(lust_aot_rt::EffectYield::Completed({val_expr}));"
            );
        }
        Ok(())
    }

    fn emit_nontail_yield_op_call(&mut self, bb: BlockId, indent: usize) -> Result<()> {
        let pad = "    ".repeat(indent);
        let Some(ysig) = self.ctx.nontail_body_metas.get(&self.func.signature.name).cloned() else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: __lust_yield_op outside non-tail body '{}'",
                self.func.signature.name
            )));
        };
        let Some(plan) = self.nontail_yield_op_plans.get(&bb).cloned() else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: missing NonTailYieldOpPlan at block {bb:?}"
            )));
        };
        let Some(cont_meta) = self.ctx.fn_metas.get(&plan.cont_fn_name).cloned() else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: missing lifted continuation '{}'",
                plan.cont_fn_name
            )));
        };
        let Some(cont_fn) = self
            .ctx
            .module
            .functions
            .iter()
            .find(|f| f.signature.name == plan.cont_fn_name)
        else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: missing lifted continuation MirFunction '{}'",
                plan.cont_fn_name
            )));
        };

        let arg_expr = match &ysig.op_arg_mode {
            NonTailOpArgMode::Unit => "()".to_string(),
            NonTailOpArgMode::Single(arg_ty) => match plan.arg_ops.first() {
                Some(op) => self.emit_operand_owned(op, arg_ty),
                None => self.ctx.rust_default_expr(arg_ty),
            },
            NonTailOpArgMode::TaggedVec => {
                if plan.arg_ops.is_empty() {
                    format!("({:?}, alloc::vec::Vec::new())", plan.op_name)
                } else {
                    let elems: Vec<String> = plan
                        .arg_ops
                        .iter()
                        .map(|op| self.emit_operand_as_aot_value(op))
                        .collect();
                    format!("({:?}, alloc::vec![{}])", plan.op_name, elems.join(", "))
                }
            }
        };

        let (cap_bindings, cap_call_exprs) = self.build_lifted_capture_bindings(
            &plan.cont_fn_name,
            &plan.cont_captures,
            "__k_cap",
        )?;

        let n_caps = plan.cont_captures.len();
        let resume_rust = self.ctx.rust_type(&ysig.resume_val_ty, "'static");
        let body_ret_rust = self.ctx.rust_type(&ysig.body_ret_ty, "'static");

        let mut cont_call_args = Vec::new();
        if cont_meta.needs_ctx {
            cont_call_args.push("__k_ctx".to_string());
        }
        if cont_meta.needs_alloc {
            cont_call_args.push("&__k_alloc".to_string());
        }
        if cont_meta.needs_arena {
            cont_call_args.push("&__k_arena".to_string());
        }
        cont_call_args.extend(cap_call_exprs);

        let mut resumed_param_binding = None;
        if let Some(&pid) = cont_fn.params.get(n_caps) {
            let cont_pty = &cont_fn.locals[pid.0].ty;
            let coerced = self.coerce_expr_to_type("__resumed_val", &ysig.resume_val_ty, cont_pty);
            let pmode = cont_meta
                .param_modes
                .get(n_caps)
                .copied()
                .unwrap_or(ParamMode::ByValue);
            match pmode {
                ParamMode::RefStr => {
                    resumed_param_binding = Some(format!("let __k_res_arg = {coerced};"));
                    cont_call_args.push("__k_res_arg.as_str()".to_string());
                }
                ParamMode::RefStruct => {
                    resumed_param_binding = Some(format!("let __k_res_arg = {coerced};"));
                    cont_call_args.push("&__k_res_arg".to_string());
                }
                ParamMode::MutRefStruct => {
                    resumed_param_binding = Some(format!("let mut __k_res_arg = {coerced};"));
                    cont_call_args.push("&mut __k_res_arg".to_string());
                }
                ParamMode::RcCell => {
                    cont_call_args.push(format!(
                        "alloc::rc::Rc::new(core::cell::Cell::new({coerced}))"
                    ));
                }
                ParamMode::RcRefCell => {
                    cont_call_args.push(format!(
                        "alloc::rc::Rc::new(core::cell::RefCell::new({coerced}))"
                    ));
                }
                _ => cont_call_args.push(coerced),
            }
        }

        let raw_cont_call = format!("{}({})", cont_meta.rust_name, cont_call_args.join(", "));
        let ret_coerced =
            self.coerce_expr_to_type("__k_out", &cont_meta.return_type, &ysig.body_ret_ty);

        let _ = writeln!(self.out, "{pad}{{");
        let _ = writeln!(self.out, "{pad}    let __y_arg = {arg_expr};");
        for b in &cap_bindings {
            let _ = writeln!(self.out, "{pad}    {b}");
        }
        for g in self.errdefer_guards.clone().iter().rev() {
            let _ = writeln!(self.out, "{pad}    core::mem::DropGuard::dismiss({g});");
        }
        let _ = writeln!(
            self.out,
            "{pad}    return Ok(lust_aot_rt::EffectYield::Yielded {{"
        );
        let _ = writeln!(self.out, "{pad}        arg: __y_arg,");
        let _ = writeln!(
            self.out,
            "{pad}        resume_fn: alloc::boxed::Box::new(move |__k_ctx: &mut lust_aot_rt::AotCtx, __resumed_val: {resume_rust}| -> Result<{body_ret_rust}, lust_aot_rt::AotError> {{"
        );
        if cont_meta.needs_alloc || cont_meta.needs_arena {
            let _ = writeln!(
                self.out,
                "{pad}            let __k_alloc = __k_ctx.budget_alloc.share();"
            );
        }
        if cont_meta.needs_arena {
            let _ = writeln!(
                self.out,
                "{pad}            let __k_arena = lust_aot_rt::LustChunkArena::new(&__k_alloc);"
            );
        }
        if let Some(rb) = resumed_param_binding {
            let _ = writeln!(self.out, "{pad}            {rb}");
        }
        let _ = writeln!(
            self.out,
            "{pad}            let __k_out = {raw_cont_call}.map_err(lust_aot_rt::AotError::from)?;"
        );
        let _ = writeln!(self.out, "{pad}            Ok({ret_coerced})");
        let _ = writeln!(self.out, "{pad}        }}),");
        let _ = writeln!(self.out, "{pad}    }});");
        let _ = writeln!(self.out, "{pad}}}");
        Ok(())
    }

    fn emit_nontail_resume_call(
        &mut self,
        args: &[Operand],
        destination: &Place,
        indent: usize,
    ) -> Result<()> {
        let pad = "    ".repeat(indent);
        let Some(arm_meta) = self.ctx.nontail_arm_metas.get(&self.func.signature.name).cloned() else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: __lust_nontail_resume outside non-tail handler arm '{}'",
                self.func.signature.name
            )));
        };
        let val_expr = match args.first() {
            Some(op) => self.emit_operand_owned(op, &arm_meta.resume_val_ty),
            None => self.ctx.rust_default_expr(&arm_meta.resume_val_ty),
        };
        let _ = writeln!(
            self.out,
            "{pad}let __nt_resumed_out = __k.resume(__ctx, {val_expr})?;"
        );
        if !self.dest_is_omitted(destination) {
            let dst_ty = self.place_mir_type(destination);
            if !matches!(dst_ty, MirType::Unit) {
                let coerced =
                    self.coerce_expr_to_type("__nt_resumed_out", &arm_meta.body_ret_ty, &dst_ty);
                self.emit_assign_place_val(destination, &coerced, indent);
            }
        }
        Ok(())
    }

    fn emit_scoped_call(&mut self, bb: BlockId, indent: usize) -> Result<()> {
        let pad = "    ".repeat(indent);
        let Some((thunk_fn_name, thunk_caps)) = self.scoped_plans.get(&bb).cloned() else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: missing scoped block plan at block {bb:?}"
            )));
        };
        let Some(thunk_meta) = self.ctx.fn_metas.get(&thunk_fn_name).cloned() else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: missing lifted scoped thunk '{thunk_fn_name}'"
            )));
        };
        let (cap_bindings, cap_call_exprs) =
            self.build_lifted_capture_bindings(&thunk_fn_name, &thunk_caps, "__sc_cap")?;
        let mut call_args = Vec::new();
        if thunk_meta.needs_ctx {
            call_args.push("__sc_ctx".to_string());
        }
        if thunk_meta.needs_alloc {
            call_args.push("__alloc".to_string());
        }
        if thunk_meta.needs_arena {
            call_args.push("__arena".to_string());
        }
        call_args.extend(cap_call_exprs);
        let raw_call = format!("{}({})", thunk_meta.rust_name, call_args.join(", "));

        let _ = writeln!(
            self.out,
            "{pad}// 3-phase LIFO AotScope (Cancelling -> Joining -> Finalizing) for StmtKind::Scoped (NG-W13-1B)"
        );
        let _ = writeln!(self.out, "{pad}{{");
        for b in &cap_bindings {
            let _ = writeln!(self.out, "{pad}    {b}");
        }
        let _ = writeln!(
            self.out,
            "{pad}    lust_aot_rt::with_aot_scope(__ctx, |__sc_ctx: &mut lust_aot_rt::AotCtx| -> Result<(), lust_aot_rt::AotError> {{"
        );
        let _ = writeln!(
            self.out,
            "{pad}        let _ = {raw_call}.map_err(lust_aot_rt::AotError::from)?;"
        );
        let _ = writeln!(self.out, "{pad}        Ok(())");
        let _ = writeln!(self.out, "{pad}    }})?;");
        let _ = writeln!(self.out, "{pad}}}");
        Ok(())
    }

    fn emit_scope_finalizer_call(&mut self, bb: BlockId, indent: usize) -> Result<()> {
        let pad = "    ".repeat(indent);
        let Some((fin_fn_name, fin_caps)) = self.scope_callback_plans.get(&bb).cloned() else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: missing scope finalizer plan at block {bb:?}"
            )));
        };
        let Some(fin_meta) = self.ctx.fn_metas.get(&fin_fn_name).cloned() else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: missing lifted scope finalizer '{fin_fn_name}'"
            )));
        };
        let (cap_bindings, cap_call_exprs) =
            self.build_lifted_capture_bindings(&fin_fn_name, &fin_caps, "__fin_cap")?;
        let mut call_args = Vec::new();
        if fin_meta.needs_ctx {
            call_args.push("__fin_ctx".to_string());
        }
        if fin_meta.needs_alloc {
            call_args.push("&__fin_alloc".to_string());
        }
        if fin_meta.needs_arena {
            call_args.push("&__fin_arena".to_string());
        }
        call_args.extend(cap_call_exprs);
        let raw_call = format!("{}({})", fin_meta.rust_name, call_args.join(", "));

        let _ = writeln!(self.out, "{pad}{{");
        for b in &cap_bindings {
            let _ = writeln!(self.out, "{pad}    {b}");
        }
        let _ = writeln!(
            self.out,
            "{pad}    lust_aot_rt::scope_register_finalizer(__ctx, move |__fin_ctx: &mut lust_aot_rt::AotCtx| -> Result<(), lust_aot_rt::AotError> {{"
        );
        if fin_meta.needs_alloc || fin_meta.needs_arena {
            let _ = writeln!(
                self.out,
                "{pad}        let __fin_alloc = __fin_ctx.budget_alloc.share();"
            );
        }
        if fin_meta.needs_arena {
            let _ = writeln!(
                self.out,
                "{pad}        let __fin_arena = lust_aot_rt::LustChunkArena::new(&__fin_alloc);"
            );
        }
        let _ = writeln!(
            self.out,
            "{pad}        let _ = {raw_call}.map_err(lust_aot_rt::AotError::from)?;"
        );
        let _ = writeln!(self.out, "{pad}        Ok(())");
        let _ = writeln!(self.out, "{pad}    }})?;");
        let _ = writeln!(self.out, "{pad}}}");
        Ok(())
    }

    fn emit_scope_spawn_call(&mut self, bb: BlockId, indent: usize) -> Result<()> {
        let pad = "    ".repeat(indent);
        let Some((child_fn_name, child_caps)) = self.scope_callback_plans.get(&bb).cloned() else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: missing scope spawn plan at block {bb:?}"
            )));
        };
        let Some(child_meta) = self.ctx.fn_metas.get(&child_fn_name).cloned() else {
            return Err(LustError::Unknown(format!(
                "Rust AOT emitter: missing lifted scope child '{child_fn_name}'"
            )));
        };
        let (cap_bindings, cap_call_exprs) =
            self.build_lifted_capture_bindings(&child_fn_name, &child_caps, "__sp_cap")?;
        let mut call_args = Vec::new();
        if child_meta.needs_ctx {
            call_args.push("__sp_ctx".to_string());
        }
        if child_meta.needs_alloc {
            call_args.push("&__sp_alloc".to_string());
        }
        if child_meta.needs_arena {
            call_args.push("&__sp_arena".to_string());
        }
        call_args.extend(cap_call_exprs);
        let raw_call = format!("{}({})", child_meta.rust_name, call_args.join(", "));

        let _ = writeln!(self.out, "{pad}{{");
        for b in &cap_bindings {
            let _ = writeln!(self.out, "{pad}    {b}");
        }
        let _ = writeln!(
            self.out,
            "{pad}    let _ = lust_aot_rt::scope_spawn_child(__ctx, move |__sp_ctx: &mut lust_aot_rt::AotCtx, _cancelled: bool| -> Result<bool, lust_aot_rt::AotError> {{"
        );
        if child_meta.needs_alloc || child_meta.needs_arena {
            let _ = writeln!(
                self.out,
                "{pad}        let __sp_alloc = __sp_ctx.budget_alloc.share();"
            );
        }
        if child_meta.needs_arena {
            let _ = writeln!(
                self.out,
                "{pad}        let __sp_arena = lust_aot_rt::LustChunkArena::new(&__sp_alloc);"
            );
        }
        let _ = writeln!(
            self.out,
            "{pad}        let _ = {raw_call}.map_err(lust_aot_rt::AotError::from)?;"
        );
        let _ = writeln!(self.out, "{pad}        Ok(true)");
        let _ = writeln!(self.out, "{pad}    }})?;");
        let _ = writeln!(self.out, "{pad}}}");
        Ok(())
    }

    fn emit_effect_perform_call(
        &mut self,
        bb: BlockId,
        args: &[Operand],
        destination: &Place,
        indent: usize,
    ) -> Result<()> {
        let pad = "    ".repeat(indent);
        let eff_name_expr = args
            .first()
            .map(|a| self.emit_operand_str_ref(a))
            .unwrap_or_else(|| "\"\"".to_string());

        let slice_expr = if let Some(inline_ops) = self.perform_arg_plans.get(&bb).cloned() {
            if inline_ops.is_empty() {
                "&[]".to_string()
            } else {
                let elems: Vec<String> = inline_ops
                    .iter()
                    .map(|op| self.emit_operand_as_aot_value(op))
                    .collect();
                format!("&[{}]", elems.join(", "))
            }
        } else if let Some(arg1) = args.get(1) {
            match operand_mir_type(arg1, self.func) {
                MirType::Array(inner) => {
                    let arr_ref = self.emit_operand_slice_ref(arg1);
                    let elem_conv = self.coerce_expr_to_type("__el.clone()", &inner, &MirType::Unknown);
                    format!(
                        "&({arr_ref}.iter().map(|__el| {elem_conv}).collect::<alloc::vec::Vec<lust_aot_rt::AotValue>>())[..]"
                    )
                }
                _ => format!("&[{}]", self.emit_operand_as_aot_value(arg1)),
            }
        } else {
            "&[]".to_string()
        };

        let _ = writeln!(
            self.out,
            "{pad}let __perf_val = lust_aot_rt::perform_effect(__ctx, {eff_name_expr}, {slice_expr})?;"
        );
        if !self.dest_is_omitted(destination) {
            let dst_ty = self.place_mir_type(destination);
            if !matches!(dst_ty, MirType::Unit) {
                let coerced = self.coerce_expr_to_type("__perf_val", &MirType::Unknown, &dst_ty);
                self.emit_assign_place_val(destination, &coerced, indent);
            }
        }
        Ok(())
    }

    fn emit_effect_resume_call(
        &mut self,
        args: &[Operand],
        destination: &Place,
        indent: usize,
    ) -> Result<()> {
        let pad = "    ".repeat(indent);
        let val_expr = args
            .first()
            .map(|a| self.emit_operand_as_aot_value(a))
            .unwrap_or_else(|| "lust_aot_rt::AotValue::Nil".to_string());
        let _ = writeln!(
            self.out,
            "{pad}let __resume_val = lust_aot_rt::resume_effect(__ctx, {val_expr})?;"
        );
        if !self.dest_is_omitted(destination) {
            let dst_ty = self.place_mir_type(destination);
            if !matches!(dst_ty, MirType::Unit) {
                let coerced = self.coerce_expr_to_type("__resume_val", &MirType::Unknown, &dst_ty);
                self.emit_assign_place_val(destination, &coerced, indent);
            }
        }
        Ok(())
    }

    fn emit_effect_provide_call(
        &mut self,
        args: &[Operand],
        destination: &Place,
        indent: usize,
    ) -> Result<()> {
        let pad = "    ".repeat(indent);
        if args.len() < 2 {
            return Err(LustError::Unknown(
                "Rust AOT emitter: __lust_provide expects (effect, provider)".to_string(),
            ));
        }
        let eff_expr = self.emit_operand_as_aot_value(&args[0]);
        let prov_expr = self.emit_operand_as_aot_value(&args[1]);
        let svc_key = match operand_mir_type(&args[1], self.func) {
            MirType::Struct(sn) => sn,
            other => format!("{other}"),
        };
        let wrapped = format!(
            "{{ let __eff_v = {eff_expr}; let __prov_v = {prov_expr}; lust_aot_rt::AotValue::closure(move |__p_ctx: &mut lust_aot_rt::AotCtx, __p_args: &[lust_aot_rt::AotValue]| -> Result<lust_aot_rt::AotValue, lust_aot_rt::AotError> {{ lust_aot_rt::with_provided_service(__p_ctx, {svc_key:?}, __prov_v.clone(), |__inner_ctx| __eff_v.call(__inner_ctx, __p_args)) }}) }}"
        );
        if self.dest_is_omitted(destination) {
            let _ = writeln!(self.out, "{pad}let _ = {wrapped};");
        } else {
            let dst_ty = self.place_mir_type(destination);
            let final_expr = self.coerce_expr_to_type(&wrapped, &MirType::Unknown, &dst_ty);
            self.emit_assign_place_val(destination, &final_expr, indent);
        }
        Ok(())
    }

    fn emit_effect_run_sync_call(
        &mut self,
        bb: BlockId,
        args: &[Operand],
        destination: &Place,
        indent: usize,
    ) -> Result<()> {
        let pad = "    ".repeat(indent);
        if let Some(plan) = self.provide_plans.get(&bb).cloned() {
            let Some(thunk_meta) = self.ctx.fn_metas.get(&plan.thunk_fn_name).cloned() else {
                return Err(LustError::Unknown(format!(
                    "Rust AOT emitter: missing lifted effect thunk '{}'",
                    plan.thunk_fn_name
                )));
            };
            let (cap_bindings, cap_call_exprs) = self.build_lifted_capture_bindings(
                &plan.thunk_fn_name,
                &plan.thunk_captures,
                "__eff_cap",
            )?;
            let mut thunk_args = Vec::new();
            let final_ctx_var = if plan.providers.is_empty() {
                "__ctx".to_string()
            } else {
                format!("__p_ctx_{}", plan.providers.len())
            };
            if thunk_meta.needs_ctx {
                thunk_args.push("__eff_scope_ctx".to_string());
            }
            if thunk_meta.needs_alloc {
                thunk_args.push("__alloc".to_string());
            }
            if thunk_meta.needs_arena {
                thunk_args.push("__arena".to_string());
            }
            thunk_args.extend(cap_call_exprs);

            let raw_call = format!("{}({})", thunk_meta.rust_name, thunk_args.join(", "));
            let ret_rust = self.ctx.rust_type(&thunk_meta.return_type, "'a");

            // Fast path (`NG-W15-1`, `F-02`): pure synchronous lifted thunk with no providers
            // and no `AotCtx` requirement bypasses `lust_aot_rt::with_aot_scope` completely.
            if plan.providers.is_empty() && !thunk_meta.needs_ctx {
                let unwrapped_call = if thunk_meta.can_fail
                    || self.in_cleanup_closure.is_some()
                    || self.meta.can_fail
                {
                    format!("{raw_call}?")
                } else {
                    format!("match {raw_call} {{ Ok(__v) => __v }}")
                };
                if cap_bindings.is_empty() {
                    let _ = writeln!(self.out, "{pad}let __sync_res = {unwrapped_call};");
                } else {
                    let _ = writeln!(self.out, "{pad}let __sync_res = {{");
                    for b in &cap_bindings {
                        let _ = writeln!(self.out, "{pad}    {b}");
                    }
                    let _ = writeln!(self.out, "{pad}    {unwrapped_call}");
                    let _ = writeln!(self.out, "{pad}}};");
                }
            } else {
                let _ = writeln!(self.out, "{pad}let __sync_res = {{");
                for b in &cap_bindings {
                    let _ = writeln!(self.out, "{pad}    {b}");
                }
                for (pi, (svc_key, prov_op)) in plan.providers.iter().enumerate() {
                    let in_ctx = if pi == 0 {
                        "__ctx".to_string()
                    } else {
                        format!("__p_ctx_{pi}")
                    };
                    let out_ctx = format!("__p_ctx_{}", pi + 1);
                    let prov_val = self.emit_operand_as_aot_value(prov_op);
                    let inner_pad = "    ".repeat(indent + 1 + pi);
                    let _ = writeln!(
                        self.out,
                        "{inner_pad}lust_aot_rt::with_provided_service({in_ctx}, {svc_key:?}, {prov_val}, |{out_ctx}: &mut lust_aot_rt::AotCtx| -> Result<{ret_rust}, lust_aot_rt::AotError> {{"
                    );
                }
                let body_pad = "    ".repeat(indent + 1 + plan.providers.len());
                let scope_suffix = if plan.providers.is_empty() { "?" } else { "" };
                let _ = writeln!(
                    self.out,
                    "{body_pad}lust_aot_rt::with_aot_scope({final_ctx_var}, |__eff_scope_ctx: &mut lust_aot_rt::AotCtx| -> Result<{ret_rust}, lust_aot_rt::AotError> {{"
                );
                let _ = writeln!(
                    self.out,
                    "{body_pad}    {raw_call}.map_err(lust_aot_rt::AotError::from)"
                );
                let _ = writeln!(self.out, "{body_pad}}}){scope_suffix}");
                for pi in (0..plan.providers.len()).rev() {
                    let inner_pad = "    ".repeat(indent + 1 + pi);
                    if pi == 0 {
                        let _ = writeln!(self.out, "{inner_pad}}})?");
                    } else {
                        let _ = writeln!(self.out, "{inner_pad}}})");
                    }
                }
                let _ = writeln!(self.out, "{pad}}};");
            }

            if !self.dest_is_omitted(destination) {
                let dst_ty = self.place_mir_type(destination);
                if !matches!(dst_ty, MirType::Unit) {
                    let coerced =
                        self.coerce_expr_to_type("__sync_res", &thunk_meta.return_type, &dst_ty);
                    self.emit_assign_place_val(destination, &coerced, indent);
                }
            }
            return Ok(());
        }

        // Fallback: runtime `AotValue::Closure` effect value wrapped in 3-phase LIFO `AotScope`
        let eff_expr = args
            .first()
            .map(|a| self.emit_operand_as_aot_value(a))
            .unwrap_or_else(|| "lust_aot_rt::AotValue::Nil".to_string());
        let _ = writeln!(
            self.out,
            "{pad}let __sync_val = lust_aot_rt::with_aot_scope(__ctx, |__eff_scope_ctx| ({eff_expr}).call(__eff_scope_ctx, &[]))?;"
        );
        if !self.dest_is_omitted(destination) {
            let dst_ty = self.place_mir_type(destination);
            if !matches!(dst_ty, MirType::Unit) {
                let coerced = self.coerce_expr_to_type("__sync_val", &MirType::Unknown, &dst_ty);
                self.emit_assign_place_val(destination, &coerced, indent);
            }
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
        if let Some(ysig) = self.ctx.nontail_body_metas.get(&self.func.signature.name) {
            if let Some(op) = opt
                && operand_exact_local(op).is_some_and(|l| self.yield_return_locals.contains(&l))
            {
                let _ = writeln!(
                    self.out,
                    "{pad}return Err(lust_aot_rt::AotError::UserError(alloc::string::String::from(\"unreachable yield return\")));"
                );
                return;
            }
            let inner = match opt {
                Some(op) => self.emit_operand_owned(op, &ysig.body_ret_ty),
                None => self.ctx.rust_default_expr(&ysig.body_ret_ty),
            };
            if !self.errdefer_guards.is_empty() {
                let _ = writeln!(self.out, "{pad}let __ret = {inner};");
                for g in self.errdefer_guards.clone().iter().rev() {
                    let _ = writeln!(self.out, "{pad}core::mem::DropGuard::dismiss({g});");
                }
                let _ = writeln!(
                    self.out,
                    "{pad}return Ok(lust_aot_rt::EffectYield::Completed(__ret));"
                );
            } else {
                let _ = writeln!(
                    self.out,
                    "{pad}return Ok(lust_aot_rt::EffectYield::Completed({inner}));"
                );
            }
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

    fn operand_is_direct_raw_map(&self, op: &Operand) -> bool {
        operand_exact_local(op)
            .is_some_and(|l| self.direct_raw_map_locals.contains(&self.resolve_local(l)))
    }

    fn place_is_direct_raw_map(&self, place: &Place) -> bool {
        matches!(place, Place::Local(l) if self.direct_raw_map_locals.contains(&self.resolve_local(*l)))
    }

    fn emit_unboxed_queue_method_call(
        &mut self,
        bb: BlockId,
        q_loc: LocalId,
        method_name: &str,
        args: &[Operand],
        destination: &Place,
        indent: usize,
    ) {
        let pad = "    ".repeat(indent);
        let q_var = self.local_var_name(q_loc);
        let dst_ty = self.place_mir_type(destination);
        match method_name {
            "offer" | "try_offer" => {
                let val_expr = args
                    .first()
                    .map(|a| self.emit_operand_owned(a, &MirType::Int))
                    .unwrap_or_else(|| "0_i64".to_string());
                let raw = format!("{q_var}.offer({val_expr})?");
                if self.dest_is_omitted(destination) || matches!(dst_ty, MirType::Unit) {
                    let _ = writeln!(self.out, "{pad}let _ = {raw};");
                } else {
                    let expr = self.coerce_expr_to_type(&raw, &MirType::Bool, &dst_ty);
                    self.emit_assign_place_val(destination, &expr, indent);
                }
            }
            "take" => {
                let raw = format!("{q_var}.take()?");
                if let Some(&int_dst) = self.unboxed_queue_take_redirect.get(&bb) {
                    self.emit_assign_local_val(int_dst, &raw, indent);
                } else if self.dest_is_omitted(destination) || matches!(dst_ty, MirType::Unit) {
                    let _ = writeln!(self.out, "{pad}let _ = {raw};");
                } else {
                    let expr = self.coerce_expr_to_type(&raw, &MirType::Int, &dst_ty);
                    self.emit_assign_place_val(destination, &expr, indent);
                }
            }
            "try_take" => {
                let raw = format!(
                    "match {q_var}.try_take() {{ Some(__v) => lust_aot_rt::AotValue::Int(__v), None => lust_aot_rt::AotValue::Nil }}"
                );
                if self.dest_is_omitted(destination) || matches!(dst_ty, MirType::Unit) {
                    let _ = writeln!(self.out, "{pad}let _ = {raw};");
                } else {
                    let expr = self.coerce_expr_to_type(&raw, &MirType::Unknown, &dst_ty);
                    self.emit_assign_place_val(destination, &expr, indent);
                }
            }
            "len" => {
                let raw = format!("{q_var}.len()");
                if self.dest_is_omitted(destination) || matches!(dst_ty, MirType::Unit) {
                    let _ = writeln!(self.out, "{pad}let _ = {raw};");
                } else {
                    let expr = self.coerce_expr_to_type(&raw, &MirType::Int, &dst_ty);
                    self.emit_assign_place_val(destination, &expr, indent);
                }
            }
            "close" => {
                let _ = writeln!(self.out, "{pad}{q_var}.close();");
                if !self.dest_is_omitted(destination) && !matches!(dst_ty, MirType::Unit) {
                    let def = self.ctx.rust_default_expr(&dst_ty);
                    self.emit_assign_place_val(destination, &def, indent);
                }
            }
            _ => {}
        }
    }

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
                let base_ty = self.place_mir_type(base);
                if let MirType::Struct(sname) = &base_ty
                    && self.ctx.struct_is_rc_cyclic(sname)
                {
                    let base_expr = self.emit_place_value_expr(base);
                    let fty = self.place_mir_type(place);
                    if self.ctx.is_copy_type(&fty) {
                        format!("lust_aot_rt::ref_borrow(&({base_expr}).0)?.field_{fidx}")
                    } else {
                        format!("lust_aot_rt::ref_borrow(&({base_expr}).0)?.field_{fidx}.clone()")
                    }
                } else if let Place::Local(base_loc) = base.as_ref()
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
        let src_is_aot_val = matches!(src_ty, MirType::Unknown | MirType::Effect { .. });
        let dst_is_aot_val = matches!(dst_ty, MirType::Unknown | MirType::Effect { .. });
        if src_is_aot_val && dst_is_aot_val {
            return format!("({expr}).clone()");
        }
        if dst_is_aot_val {
            return match src_ty {
                MirType::Unit => format!("{{ let _ = {expr}; lust_aot_rt::AotValue::Nil }}"),
                MirType::Bool => format!("lust_aot_rt::AotValue::Bool({expr})"),
                MirType::Int => format!("lust_aot_rt::AotValue::Int({expr})"),
                MirType::Float => format!("lust_aot_rt::AotValue::Float({expr})"),
                MirType::String => format!("lust_aot_rt::AotValue::str({expr})"),
                MirType::Function {
                    params,
                    return_type,
                } => {
                    let mut cargs = vec![
                        "__c_ctx".to_string(),
                        "&__c_alloc".to_string(),
                    ];
                    for (pi, pty) in params.iter().enumerate() {
                        let raw_a = format!(
                            "__c_args.get({pi}).cloned().unwrap_or(lust_aot_rt::AotValue::Nil)"
                        );
                        cargs.push(self.coerce_expr_to_type(&raw_a, &MirType::Unknown, pty));
                    }
                    let ret_aot =
                        self.coerce_expr_to_type("__fn_res", return_type, &MirType::Unknown);
                    format!(
                        "{{ let __fn_ref = ({expr}).clone(); lust_aot_rt::AotValue::closure(move |__c_ctx: &mut lust_aot_rt::AotCtx, __c_args: &[lust_aot_rt::AotValue]| -> Result<lust_aot_rt::AotValue, lust_aot_rt::AotError> {{ let __c_alloc = __c_ctx.budget_alloc.share(); let __fn_res = (__fn_ref)({})?; Ok({ret_aot}) }}) }}",
                        cargs.join(", ")
                    )
                }
                MirType::Struct(_) => coerce_rust_to_aot_value(expr, src_ty, self.ctx),
                MirType::Unknown | MirType::Effect { .. } => format!("({expr}).clone()"),
                _ => format!("lust_aot_rt::AotValue::str(alloc::format!(\"{{:?}}\", {expr}))"),
            };
        }
        if src_is_aot_val {
            return match dst_ty {
                MirType::Unit => format!("{{ let _ = {expr}; () }}"),
                MirType::Bool => format!("({expr}).is_truthy()"),
                MirType::Int => format!(
                    "({expr}).as_int().ok_or_else(|| lust_aot_rt::AotError::UserError(alloc::string::String::from(\"expected int\")))?"
                ),
                MirType::Float => format!("({expr}).as_float_coerced()?"),
                MirType::String => format!("({expr}).to_display_string()"),
                MirType::Struct(_) => coerce_aot_value_to_rust(expr, dst_ty, self.ctx),
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
                if matches!(expected_ty, MirType::Unknown | MirType::Effect { .. })
                    && !matches!(src_ty, MirType::Unknown | MirType::Effect { .. })
                {
                    let v = self.emit_operand_value(op);
                    return self.coerce_expr_to_type(&v, &src_ty, expected_ty);
                }
                if matches!(src_ty, MirType::Unknown | MirType::Effect { .. })
                    && !matches!(expected_ty, MirType::Unknown | MirType::Effect { .. })
                {
                    let v = self.emit_operand_value(op);
                    return self.coerce_expr_to_type(&v, &src_ty, expected_ty);
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
                                    if self.gsub_first_only_locals.contains(&r) {
                                        if is_move
                                            && self.is_pure_init_temp(r)
                                            && self.local_use_counts.get(&r).copied().unwrap_or(0) <= 1
                                        {
                                            format!("core::mem::take(&mut {var})")
                                        } else {
                                            format!("{var}.clone()")
                                        }
                                    } else if self.small_str_array_locals.contains_key(&r) {
                                        format!(
                                            "core::mem::replace(&mut {var}, __LustSplitSpans::new_in(__alloc))"
                                        )
                                    } else if self.is_param(r).is_some_and(|(_, m)| {
                                        matches!(m, ParamMode::RefSlice | ParamMode::MutVec)
                                    }) {
                                        format!(
                                            "{{ let mut __v = alloc::vec::Vec::new_in(__alloc); __v.extend_from_slice({var}); __v }}"
                                        )
                                    } else if is_move
                                        && self.is_pure_init_temp(r)
                                        && self.local_use_counts.get(&r).copied().unwrap_or(0) <= 1
                                    {
                                        format!(
                                            "core::mem::replace(&mut {var}, alloc::vec::Vec::new_in(__alloc))"
                                        )
                                    } else {
                                        format!("{var}.clone()")
                                    }
                                }
                                MirType::Map(_, _) => {
                                    if self.direct_raw_map_locals.contains(&r)
                                        && self.is_pure_init_temp(r)
                                        && self.local_use_counts.get(&r).copied().unwrap_or(0) <= 1
                                    {
                                        format!("core::mem::take(&mut {var})")
                                    } else {
                                        format!("{var}.clone()")
                                    }
                                }
                                MirType::Function { .. } | MirType::Unknown => {
                                    format!("{var}.clone()")
                                }
                                MirType::Enum(_) => {
                                    format!("{var}.clone()")
                                }
                                MirType::Option(inner) => {
                                    if let MirType::Struct(sname) = inner.as_ref()
                                        && self.ctx.struct_is_rc_cyclic(sname)
                                    {
                                        format!("{var}.clone()")
                                    } else {
                                        format!("core::mem::take(&mut {var})")
                                    }
                                }
                                MirType::Struct(sname) => {
                                    if let Some(s) = self.ctx.structs.get(sname) {
                                        if s.is_mutated_cyclic {
                                            format!("{var}.clone()")
                                        } else if s.is_cyclic {
                                            format!("core::mem::take(&mut {var})")
                                        } else if s.derives_default {
                                            if is_move
                                                && self.is_pure_init_temp(r)
                                                && self
                                                    .local_use_counts
                                                    .get(&r)
                                                    .copied()
                                                    .unwrap_or(0)
                                                    <= 1
                                            {
                                                format!("core::mem::take(&mut {var})")
                                            } else {
                                                format!("{var}.clone()")
                                            }
                                        } else {
                                            var
                                        }
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
                } else if let MirType::Option(inner) = expected_ty {
                    if let MirType::Struct(sname) = inner.as_ref()
                        && self.ctx.struct_is_rc_cyclic(sname)
                    {
                        format!("{expr}.clone()")
                    } else {
                        format!("core::mem::take(&mut {expr})")
                    }
                } else if let MirType::Struct(sname) = expected_ty
                    && self.ctx.struct_is_arena_cyclic(sname)
                {
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
                } else if self.cow_str_locals.contains(&r) {
                    format!("{var}.as_ref()")
                } else if self.scoped_ref_locals.contains(&r)
                    || matches!(self.is_param(r), Some((_, ParamMode::RefStr)))
                {
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

    fn emit_fused_concat_chain(
        &mut self,
        place: &Place,
        pieces: &[ConcatPiece],
        indent: usize,
    ) {
        let has_budget = self.ctx.opts.max_memory_bytes.is_some();
        if !has_budget
            && let Place::Local(dst) = place
        {
            let r = self.resolve_local(*dst);
            let self_ref = pieces.iter().any(|p| match p {
                ConcatPiece::Str(op) | ConcatPiece::Int(op) | ConcatPiece::Float(op) => {
                    operand_exact_local(op).is_some_and(|l| self.resolve_local(l) == r)
                }
            });
            if !self_ref
                && self.local_storage(r) == LocalStorage::Direct
                && self.func.locals[r.0].ty == MirType::String
                && !self.scoped_ref_locals.contains(&r)
                && !self.cow_str_locals.contains(&r)
            {
                let pad = "    ".repeat(indent);
                let dst_var = self.local_var_name(r);
                let mut stmt = format!("{pad}{{ {dst_var}.clear(); ");
                for piece in pieces {
                    match piece {
                        ConcatPiece::Str(op) => {
                            let sref = self.emit_operand_str_ref(op);
                            let _ = write!(stmt, "{dst_var}.push_str({sref}); ");
                        }
                        ConcatPiece::Int(op) => {
                            let v = self.emit_operand_value(op);
                            let _ = write!(
                                stmt,
                                "lust_aot_rt::string_append_int(&mut {dst_var}, {v}); "
                            );
                        }
                        ConcatPiece::Float(op) => {
                            let v = self.emit_operand_value(op);
                            let _ = write!(
                                stmt,
                                "lust_aot_rt::string_append_float(&mut {dst_var}, {v}); "
                            );
                        }
                    }
                }
                stmt.push('}');
                let _ = writeln!(self.out, "{stmt}");
                return;
            }
        }
        let mut expr = String::from("{ let mut __s = alloc::string::String::with_capacity(32); ");
        for piece in pieces {
            match piece {
                ConcatPiece::Str(op) => {
                    let sref = self.emit_operand_str_ref(op);
                    if has_budget {
                        let _ = write!(
                            expr,
                            "{{ let __seg = {sref}; __alloc.try_charge(__seg.len()).map_err(lust_aot_rt::AotError::from)?; __s.push_str(__seg); }} "
                        );
                    } else {
                        let _ = write!(expr, "__s.push_str({sref}); ");
                    }
                }
                ConcatPiece::Int(op) => {
                    let v = self.emit_operand_value(op);
                    if has_budget {
                        let _ = write!(
                            expr,
                            "lust_aot_rt::try_string_append_int(&mut __s, {v}, __alloc)?; "
                        );
                    } else {
                        let _ = write!(expr, "lust_aot_rt::string_append_int(&mut __s, {v}); ");
                    }
                }
                ConcatPiece::Float(op) => {
                    let v = self.emit_operand_value(op);
                    let _ = write!(expr, "lust_aot_rt::string_append_float(&mut __s, {v}); ");
                }
            }
        }
        expr.push_str("__s }");
        self.emit_assign_place_val(place, &expr, indent);
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

fn collect_block_used_locals(block: &BasicBlock, out: &mut HashSet<LocalId>) {
    let mut tmp = HashSet::new();
    for stmt in &block.statements {
        if let Statement::Assign(place, rval, _) = stmt {
            out.insert(place.root_local());
            place.collect_uses(&mut tmp);
            rval.collect_uses(&mut tmp);
        }
    }
    if let Some(term) = &block.terminator {
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
    }
    for id in tmp {
        out.insert(LocalId(id));
    }
}

fn collect_node_blocks(node: &StructuredNode, out: &mut HashSet<BlockId>) {
    match node {
        StructuredNode::BasicBlock(bb) | StructuredNode::Return(bb, _) => {
            out.insert(*bb);
        }
        StructuredNode::Seq(items) => {
            for item in items {
                collect_node_blocks(item, out);
            }
        }
        StructuredNode::If {
            cond_block,
            then_branch,
            else_branch,
            ..
        } => {
            out.insert(*cond_block);
            collect_node_blocks(then_branch, out);
            if let Some(eb) = else_branch {
                collect_node_blocks(eb, out);
            }
        }
        StructuredNode::IfArrayGetOk {
            header_block,
            then_branch,
            else_branch,
            ..
        } => {
            out.insert(*header_block);
            collect_node_blocks(then_branch, out);
            if let Some(eb) = else_branch {
                collect_node_blocks(eb, out);
            }
        }
        StructuredNode::Switch {
            discr_block,
            cases,
            default,
            ..
        } => {
            out.insert(*discr_block);
            for (_, c) in cases {
                collect_node_blocks(c, out);
            }
            collect_node_blocks(default, out);
        }
        StructuredNode::WhileLoop {
            header_block, body, ..
        } => {
            out.insert(*header_block);
            collect_node_blocks(body, out);
        }
        StructuredNode::Loop { body, .. } | StructuredNode::LabeledBlock { body, .. } => {
            collect_node_blocks(body, out);
        }
        StructuredNode::StateMachineFallback { blocks, .. } => {
            for &b in blocks {
                out.insert(b);
            }
        }
        StructuredNode::Break(_) | StructuredNode::Continue(_) => {}
    }
}

fn array_mutated_in_blocks(
    func: &MirFunction,
    arr_id: LocalId,
    blocks: &HashSet<BlockId>,
    alias_map: &HashMap<LocalId, LocalId>,
) -> bool {
    let resolve = |mut loc: LocalId| -> LocalId {
        for _ in 0..func.locals.len() {
            if let Some(&next) = alias_map.get(&loc) {
                loc = next;
            } else {
                break;
            }
        }
        loc
    };
    for &bb in blocks {
        let blk = &func.blocks[bb.0];
        for stmt in &blk.statements {
            if let Statement::Assign(place, _, _) = stmt
                && resolve(place.root_local()) == arr_id
            {
                return true;
            }
        }
        if let Some(Terminator::Call {
            callee,
            args,
            destination,
            ..
        }) = &blk.terminator
        {
            if resolve(destination.root_local()) == arr_id {
                return true;
            }
            if matches!(
                callee.as_str(),
                "array.push" | "table.push" | "array.pop" | "table.pop" | "array.clear"
            ) && args
                .first()
                .and_then(operand_exact_local)
                .is_some_and(|l| resolve(l) == arr_id)
            {
                return true;
            }
        }
    }
    false
}

fn collect_if_array_get_info(
    node: &StructuredNode,
    fe: &FnEmitter<'_, '_>,
    allowed: &mut HashSet<LocalId>,
    disqual: &mut HashSet<LocalId>,
) {
    match node {
        StructuredNode::IfArrayGetOk {
            arr_local,
            binding_local,
            then_branch,
            else_branch,
            ..
        } => {
            let arr_id = fe.resolve_local(*arr_local);
            if fe.borrowed_array_str_locals.contains(binding_local) {
                allowed.insert(arr_id);
            } else {
                disqual.insert(arr_id);
            }
            collect_if_array_get_info(then_branch, fe, allowed, disqual);
            if let Some(eb) = else_branch {
                collect_if_array_get_info(eb, fe, allowed, disqual);
            }
        }
        StructuredNode::Seq(items) => {
            for item in items {
                collect_if_array_get_info(item, fe, allowed, disqual);
            }
        }
        StructuredNode::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_if_array_get_info(then_branch, fe, allowed, disqual);
            if let Some(eb) = else_branch {
                collect_if_array_get_info(eb, fe, allowed, disqual);
            }
        }
        StructuredNode::Switch { cases, default, .. } => {
            for (_, c) in cases {
                collect_if_array_get_info(c, fe, allowed, disqual);
            }
            collect_if_array_get_info(default, fe, allowed, disqual);
        }
        StructuredNode::WhileLoop { body, .. }
        | StructuredNode::Loop { body, .. }
        | StructuredNode::LabeledBlock { body, .. } => {
            collect_if_array_get_info(body, fe, allowed, disqual);
        }
        _ => {}
    }
}

fn check_gsub_if_array_get_zero_only(
    node: &StructuredNode,
    fe: &FnEmitter<'_, '_>,
    group: &HashSet<LocalId>,
    const_zero_locals: &HashSet<LocalId>,
    disqual: &mut HashSet<LocalId>,
) {
    match node {
        StructuredNode::IfArrayGetOk {
            arr_local,
            idx_local,
            then_branch,
            else_branch,
            ..
        } => {
            let arr_id = fe.resolve_local(*arr_local);
            let idx_id = fe.resolve_local(*idx_local);
            if group.contains(&arr_id)
                && (!const_zero_locals.contains(&idx_id) || else_branch.is_some())
            {
                disqual.insert(arr_id);
            }
            check_gsub_if_array_get_zero_only(then_branch, fe, group, const_zero_locals, disqual);
            if let Some(eb) = else_branch {
                check_gsub_if_array_get_zero_only(eb, fe, group, const_zero_locals, disqual);
            }
        }
        StructuredNode::Seq(items) => {
            for item in items {
                check_gsub_if_array_get_zero_only(item, fe, group, const_zero_locals, disqual);
            }
        }
        StructuredNode::If {
            then_branch,
            else_branch,
            ..
        } => {
            check_gsub_if_array_get_zero_only(then_branch, fe, group, const_zero_locals, disqual);
            if let Some(eb) = else_branch {
                check_gsub_if_array_get_zero_only(eb, fe, group, const_zero_locals, disqual);
            }
        }
        StructuredNode::Switch { cases, default, .. } => {
            for (_, c) in cases {
                check_gsub_if_array_get_zero_only(c, fe, group, const_zero_locals, disqual);
            }
            check_gsub_if_array_get_zero_only(default, fe, group, const_zero_locals, disqual);
        }
        StructuredNode::WhileLoop { body, .. }
        | StructuredNode::Loop { body, .. }
        | StructuredNode::LabeledBlock { body, .. } => {
            check_gsub_if_array_get_zero_only(body, fe, group, const_zero_locals, disqual);
        }
        _ => {}
    }
}

/// Compiles a simple single-atom ASCII character-class `+` pattern (e.g. `"[0-9]+"`, `"%d+"`,
/// `"[a-zA-Z0-9_]+"`, `"%w+"`, `"%s+"`) into a 128-bit `[u64; 2]` ASCII membership bitmap.
/// Returns `None` for any pattern that can match non-ASCII bytes, captures, anchors, or multiple atoms.
fn try_compile_ascii_class_plus_bitmap(pat: &str) -> Option<[u64; 2]> {
    let bytes = pat.as_bytes();
    if bytes.len() < 2 || *bytes.last()? != b'+' {
        return None;
    }
    let body = &bytes[..bytes.len() - 1];
    let mut bm = [0u64; 2];
    let set_bit = |bm: &mut [u64; 2], b: u8| {
        if b < 128 {
            bm[(b >> 6) as usize] |= 1_u64 << (b & 63);
        }
    };
    let set_class = |bm: &mut [u64; 2], cls: u8| -> bool {
        if cls.is_ascii_uppercase() {
            // Negated classes match non-ASCII bytes >= 128, so reject for ASCII-only fast path.
            return false;
        }
        for b in 0u8..128 {
            let hit = match cls {
                b'a' => b.is_ascii_alphabetic(),
                b'c' => b.is_ascii_control(),
                b'd' => b.is_ascii_digit(),
                b'l' => b.is_ascii_lowercase(),
                b'u' => b.is_ascii_uppercase(),
                b'w' => b.is_ascii_alphanumeric() || b == b'_',
                b'x' => b.is_ascii_hexdigit(),
                b's' => b.is_ascii_whitespace(),
                b'p' => b.is_ascii_punctuation(),
                b'z' => b == 0,
                _ => return false,
            };
            if hit {
                set_bit(bm, b);
            }
        }
        true
    };

    if body.len() == 2 && body[0] == b'%' {
        if set_class(&mut bm, body[1]) {
            return Some(bm);
        }
        return None;
    }
    if body.len() >= 3 && body[0] == b'[' && *body.last()? == b']' {
        let inner = &body[1..body.len() - 1];
        if inner.is_empty() || inner[0] == b'^' {
            return None;
        }
        let mut i = 0usize;
        while i < inner.len() {
            if inner[i] == b'%' && i + 1 < inner.len() {
                let c = inner[i + 1];
                if !set_class(&mut bm, c) {
                    if c < 128 {
                        set_bit(&mut bm, c);
                    } else {
                        return None;
                    }
                }
                i += 2;
            } else if i + 2 < inner.len() && inner[i + 1] == b'-' {
                let lo = inner[i];
                let hi = inner[i + 2];
                if lo >= 128 || hi >= 128 || lo > hi {
                    return None;
                }
                for b in lo..=hi {
                    set_bit(&mut bm, b);
                }
                i += 3;
            } else {
                let c = inner[i];
                if c >= 128 || c == b'[' || c == b']' {
                    return None;
                }
                set_bit(&mut bm, c);
                i += 1;
            }
        }
        if bm[0] != 0 || bm[1] != 0 {
            return Some(bm);
        }
    }
    None
}

fn collect_simple_straight_body_blocks(
    node: &StructuredNode,
    func: &MirFunction,
    out: &mut Vec<BlockId>,
) -> bool {
    match node {
        StructuredNode::BasicBlock(bb) => {
            if matches!(
                func.blocks[bb.0].terminator,
                Some(Terminator::Call { .. } | Terminator::Return(_, _) | Terminator::Raise(_, _))
            ) {
                return false;
            }
            out.push(*bb);
            true
        }
        StructuredNode::Seq(items) => items
            .iter()
            .all(|it| collect_simple_straight_body_blocks(it, func, out)),
        StructuredNode::LabeledBlock { body, .. } => {
            collect_simple_straight_body_blocks(body, func, out)
        }
        StructuredNode::If {
            cond_block,
            then_branch,
            else_branch,
            ..
        } => {
            out.push(*cond_block);
            collect_simple_straight_body_blocks(then_branch, func, out)
                && else_branch
                    .as_deref()
                    .is_none_or(|eb| collect_simple_straight_body_blocks(eb, func, out))
        }
        _ => false,
    }
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

pub mod constant_folding;
pub mod dce;
pub mod escape_analysis;
pub mod inline;
pub mod safepoints;
pub mod simd;

pub use constant_folding::ConstantFolding;
pub use dce::DeadCodeElimination;
pub use escape_analysis::{EscapeAnalysis, EscapeStatus};
pub use inline::{InlinerConfig, JitInlineConfig, MirInliner};
pub use safepoints::{SafepointConfig, SafepointStats, Safepoints};
pub use simd::{SimdConfig, SimdOptimizer, SimdStats, SIMD_LANE_WIDTH};

use crate::mir::ir::{
    BlockId, LocalId, MirFunction, MirModule, Operand, Place, Rvalue, Statement, Terminator,
};

pub struct MirOptimizer;

fn fold_return_constants(func: &mut MirFunction) -> bool {
    let mut changed = false;
    for block in &mut func.blocks {
        let mut constants = hashbrown::HashMap::new();
        for stmt in &block.statements {
            if let Statement::Assign(Place::Local(dst), Rvalue::Use(Operand::Constant(c)), _) = stmt {
                constants.insert(*dst, c.clone());
            }
        }
        if let Some(Terminator::Return(Some(op), _)) = &mut block.terminator {
            match op {
                Operand::Copy(Place::Local(id)) | Operand::Move(Place::Local(id)) => {
                    if let Some(c) = constants.get(id) {
                        *op = Operand::Constant(c.clone());
                        changed = true;
                    }
                }
                _ => {}
            }
        }
    }
    changed
}

impl MirOptimizer {
    pub fn optimize_function(func: &mut MirFunction) {
        let max_passes = 10;
        for _ in 0..max_passes {
            let mut changed = false;
            changed |= ConstantFolding::run_on_function(func);
            changed |= DeadCodeElimination::run_on_function(func);
            changed |= SimdOptimizer::run_on_function(func);
            for (i, block) in func.blocks.iter_mut().enumerate() {
                block.id = BlockId(i);
            }
            changed |= fold_return_constants(func);
            if !changed {
                break;
            }
        }
    }

    pub fn optimize_module(module: &mut MirModule) {
        let max_passes = 10;
        for _ in 0..max_passes {
            let mut changed = false;
            changed |= MirInliner::run_on_module(module);
            for func in &mut module.functions {
                let mut fn_changed = false;
                for _ in 0..max_passes {
                    let mut iter_changed = false;
                    iter_changed |= ConstantFolding::run_on_function(func);
                    iter_changed |= DeadCodeElimination::run_on_function(func);
                    iter_changed |= SimdOptimizer::run_on_function(func);
                    for (i, block) in func.blocks.iter_mut().enumerate() {
                        block.id = BlockId(i);
                    }
                    iter_changed |= fold_return_constants(func);
                    if !iter_changed {
                        break;
                    }
                    fn_changed = true;
                }
                changed |= fn_changed;
            }
            if !changed {
                break;
            }
        }
    }

    /// JIT-safe module optimization (`PERF-T1-4` / Hazard 3 Step A):
    /// 1. Runs `MirInliner::inline_module_for_jit` enforcing all JIT guards (no direct/mutual
    ///    recursion, no effect/handler callees, no loop-bearing callees, `excluded_functions`,
    ///    local count cap, and `MirVerifier::verify_fn` fail-closed revert).
    /// 2. Runs `MirOptimizer::optimize_function` ONLY on functions that received inlined code
    ///    to clean up redundant copies/constants/dead blocks before continuation bytecode,
    ///    deopt recipe construction, and Cranelift codegen.
    /// 3. Re-verifies every optimized function with `MirVerifier::verify_fn`, reverting to the
    ///    pre-inline snapshot on any verification failure (fail-closed).
    ///
    /// Returns the set of function names that retain verified post-inline MIR.
    pub fn optimize_module_for_jit(
        module: &mut MirModule,
        config: &JitInlineConfig,
    ) -> hashbrown::HashSet<alloc::string::String> {
        let pre_inline_snapshots: hashbrown::HashMap<alloc::string::String, MirFunction> = module
            .functions
            .iter()
            .map(|f| (f.signature.name.clone(), f.clone()))
            .collect();

        let mut inlined = MirInliner::inline_module_for_jit(module, config);

        for func in &mut module.functions {
            if !inlined.contains(&func.signature.name) {
                continue;
            }
            Self::optimize_function(func);
            if crate::mir::verifier::MirVerifier::verify_fn(func).is_err() {
                if let Some(orig) = pre_inline_snapshots.get(&func.signature.name) {
                    *func = orig.clone();
                }
                inlined.remove(&func.signature.name);
            }
        }

        inlined
    }

    /// Performs escape analysis on a MIR function local variables.
    pub fn analyze_escapes(
        func: &MirFunction,
    ) -> hashbrown::HashMap<LocalId, EscapeStatus> {
        EscapeAnalysis::analyze_function(func)
    }

    /// Inserts gas and loop safepoints on a single function with default configuration.
    pub fn insert_safepoints(func: &mut MirFunction) -> bool {
        Safepoints::run_on_function(func)
    }

    /// Inserts gas and loop safepoints on a single function with custom configuration.
    pub fn insert_safepoints_with_config(
        func: &mut MirFunction,
        config: &SafepointConfig,
    ) -> usize {
        Safepoints::run_on_function_with_config(func, config)
    }

    /// Inserts gas and loop safepoints across a MIR module with custom configuration.
    pub fn insert_safepoints_module(
        module: &mut MirModule,
        config: &SafepointConfig,
    ) -> SafepointStats {
        Safepoints::run_on_module_with_config(module, config)
    }

    /// Instruments scalar control flow before optimization so gas checks retain
    /// their per-iteration meaning and prevent unsafe loop vectorization.
    pub fn optimize_function_with_safepoints(
        func: &mut MirFunction,
        config: &SafepointConfig,
    ) -> usize {
        let inserted = Safepoints::run_on_function_with_config(func, config);
        Self::optimize_function(func);
        inserted
    }

    /// Instruments scalar module control flow before general optimization.
    pub fn optimize_module_with_safepoints(
        module: &mut MirModule,
        config: &SafepointConfig,
    ) -> SafepointStats {
        let stats = Safepoints::run_on_module_with_config(module, config);
        Self::optimize_module(module);
        stats
    }
}

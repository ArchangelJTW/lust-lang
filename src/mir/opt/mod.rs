pub mod constant_folding;
pub mod dce;
pub mod escape_analysis;
pub mod inline;
pub mod safepoints;

pub use constant_folding::ConstantFolding;
pub use dce::DeadCodeElimination;
pub use escape_analysis::{EscapeAnalysis, EscapeStatus};
pub use inline::{InlinerConfig, MirInliner};
pub use safepoints::{SafepointConfig, SafepointStats, Safepoints};

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

    /// Runs general optimizations followed by deterministic safepoint insertion.
    pub fn optimize_function_with_safepoints(
        func: &mut MirFunction,
        config: &SafepointConfig,
    ) -> usize {
        Self::optimize_function(func);
        Safepoints::run_on_function_with_config(func, config)
    }

    /// Runs general module optimizations followed by deterministic safepoint insertion.
    pub fn optimize_module_with_safepoints(
        module: &mut MirModule,
        config: &SafepointConfig,
    ) -> SafepointStats {
        Self::optimize_module(module);
        Safepoints::run_on_module_with_config(module, config)
    }
}

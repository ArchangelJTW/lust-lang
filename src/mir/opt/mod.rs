pub mod constant_folding;
pub mod dce;
pub mod escape_analysis;
pub mod inline;

pub use constant_folding::ConstantFolding;
pub use dce::DeadCodeElimination;
pub use escape_analysis::{EscapeAnalysis, EscapeStatus};
pub use inline::{InlinerConfig, MirInliner};

use crate::mir::ir::{BlockId, MirFunction, MirModule, Operand, Place, Rvalue, Statement, Terminator};

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
}

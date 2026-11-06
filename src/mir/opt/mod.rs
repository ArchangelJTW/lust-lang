pub mod constant_folding;
pub mod dce;
pub mod escape_analysis;

pub use constant_folding::ConstantFolding;
pub use dce::DeadCodeElimination;
pub use escape_analysis::{EscapeAnalysis, EscapeStatus};

use crate::mir::ir::{MirFunction, MirModule};

pub struct MirOptimizer;

impl MirOptimizer {
    pub fn optimize_function(func: &mut MirFunction) {
        let max_passes = 10;
        for _ in 0..max_passes {
            let mut changed = false;
            changed |= ConstantFolding::run_on_function(func);
            changed |= DeadCodeElimination::run_on_function(func);
            if !changed {
                break;
            }
        }
    }

    pub fn optimize_module(module: &mut MirModule) {
        for func in &mut module.functions {
            Self::optimize_function(func);
        }
    }
}

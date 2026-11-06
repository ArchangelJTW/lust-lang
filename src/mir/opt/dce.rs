use crate::mir::ir::*;
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};

pub struct DeadCodeElimination;

impl DeadCodeElimination {
    pub fn run_on_function(func: &mut MirFunction) -> bool {
        let mut reachable = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(func.entry);
        reachable.insert(func.entry);

        while let Some(bb_id) = queue.pop_front() {
            if let Some(block) = func.blocks.get(bb_id.0) {
                if let Some(term) = &block.terminator {
                    for succ in Self::terminator_successors(term) {
                        if reachable.insert(succ) {
                            queue.push_back(succ);
                        }
                    }
                }
            }
        }

        if reachable.len() == func.blocks.len() {
            return false;
        }

        // Renumber reachable blocks
        let mut remap = HashMap::new();
        let mut new_blocks = Vec::with_capacity(reachable.len());

        for (old_idx, block) in func.blocks.iter().enumerate() {
            let old_id = BlockId(old_idx);
            if reachable.contains(&old_id) {
                let new_id = BlockId(new_blocks.len());
                remap.insert(old_id, new_id);
                new_blocks.push(block.clone());
            }
        }

        // Patch terminator target BlockIds in new blocks
        for block in &mut new_blocks {
            if let Some(term) = &mut block.terminator {
                Self::patch_terminator_targets(term, &remap);
            }
        }

        func.entry = remap.get(&func.entry).copied().unwrap_or(BlockId(0));
        func.blocks = new_blocks;
        true
    }

    fn terminator_successors(term: &Terminator) -> Vec<BlockId> {
        match term {
            Terminator::Goto(target, _) => alloc::vec![*target],
            Terminator::Cond { then_block, else_block, .. } => alloc::vec![*then_block, *else_block],
            Terminator::Switch { cases, default, .. } => {
                let mut succs: Vec<BlockId> = cases.iter().map(|(_, target)| *target).collect();
                succs.push(*default);
                succs
            }
            Terminator::Call { success, error, .. } => {
                let mut succs = alloc::vec![*success];
                if let Some(err_bb) = error {
                    succs.push(*err_bb);
                }
                succs
            }
            Terminator::Return(_, _) | Terminator::Raise(_, _) | Terminator::Unreachable(_) => alloc::vec![],
        }
    }

    fn patch_terminator_targets(term: &mut Terminator, remap: &HashMap<BlockId, BlockId>) {
        match term {
            Terminator::Goto(target, _) => {
                if let Some(&new_id) = remap.get(target) {
                    *target = new_id;
                }
            }
            Terminator::Cond { then_block, else_block, .. } => {
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
}

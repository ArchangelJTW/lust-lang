//! Deterministic Gas & Timing Quantization Safepoint Insertion Pass (DST-T2).
//!
//! Inserts deterministic resource check safepoints (`Statement::ResourceCheck`)
//! at function entries and loop headers across MIR functions.
//!
//! In Tier-2 compiled machine code (Cranelift JIT), these safepoints guarantee:
//! 1. Bounded preemption and gas quantization across all looping constructs.
//! 2. Deterministic execution timing: long-running loops cannot starve the scheduler
//!    or execute unbounded unquantized instruction streams.
//! 3. Idempotent insertion: repeated pass invocations do not duplicate safepoints.

use crate::ast::Span;
use crate::mir::ir::{
    BasicBlock, BlockId, MirFunction, MirModule, ResourceKind, Statement, Terminator,
};
use alloc::vec::Vec;
use hashbrown::HashSet;

/// Configuration options for the MIR safepoint insertion pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafepointConfig {
    /// Gas or resource quantum charged upon function entry.
    pub entry_quantum: u64,
    /// Gas or resource quantum charged upon each loop iteration (at loop header).
    pub loop_quantum: u64,
    /// Whether to insert a safepoint at function entry block.
    pub instrument_entry: bool,
    /// Whether to insert safepoints at loop headers.
    pub instrument_loops: bool,
    /// Kind of resource being checked and quantized (defaults to `ResourceKind::Gas`).
    pub resource_kind: ResourceKind,
}

impl Default for SafepointConfig {
    fn default() -> Self {
        Self {
            entry_quantum: 1,
            loop_quantum: 1,
            instrument_entry: true,
            instrument_loops: true,
            resource_kind: ResourceKind::Gas,
        }
    }
}

impl SafepointConfig {
    pub fn new(entry_quantum: u64, loop_quantum: u64) -> Self {
        Self {
            entry_quantum,
            loop_quantum,
            instrument_entry: true,
            instrument_loops: true,
            resource_kind: ResourceKind::Gas,
        }
    }

    pub fn with_entry_quantum(mut self, q: u64) -> Self {
        self.entry_quantum = q;
        self
    }

    pub fn with_loop_quantum(mut self, q: u64) -> Self {
        self.loop_quantum = q;
        self
    }

    pub fn with_quantum(mut self, q: u64) -> Self {
        self.entry_quantum = q;
        self.loop_quantum = q;
        self
    }

    pub fn with_resource_kind(mut self, kind: ResourceKind) -> Self {
        self.resource_kind = kind;
        self
    }

    pub fn with_instrument_entry(mut self, enable: bool) -> Self {
        self.instrument_entry = enable;
        self
    }

    pub fn with_instrument_loops(mut self, enable: bool) -> Self {
        self.instrument_loops = enable;
        self
    }
}

/// Statistics collected during safepoint insertion across a module or function.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SafepointStats {
    pub functions_visited: usize,
    pub functions_instrumented: usize,
    pub entry_safepoints_inserted: usize,
    pub loop_safepoints_inserted: usize,
    pub total_safepoints_inserted: usize,
    pub total_gas_quantized: u64,
}

/// MIR Safepoint Optimization Pass.
pub struct Safepoints;

impl Safepoints {
    /// Detects all loop headers in the given block slice starting from `BlockId(0)`.
    ///
    /// Uses dominance-verified backedges (`succ` dominates `src`) combined with a DFS
    /// grey-ancestor fallback to remain sound on irreducible control-flow graphs,
    /// eliminating false loop headers caused by index-order forward merge edges
    /// (e.g. `if` merge blocks allocated before nested `then` subgraphs).
    pub fn find_loop_headers(blocks: &[BasicBlock]) -> HashSet<BlockId> {
        Self::find_loop_headers_from_entry(blocks, BlockId(0))
    }

    /// Detects all loop headers reachable from `entry` in `blocks`.
    pub fn find_loop_headers_from_entry(blocks: &[BasicBlock], entry: BlockId) -> HashSet<BlockId> {
        Self::find_loop_backedges_from_entry(blocks, entry)
            .into_iter()
            .map(|(_, header)| header)
            .collect()
    }

    /// Detects all loop headers in `func` using `func.entry` as the CFG root.
    pub(crate) fn find_loop_headers_in_fn(func: &MirFunction) -> HashSet<BlockId> {
        Self::find_loop_headers_from_entry(&func.blocks, func.entry)
    }

    /// Finds all dominance-verified (and DFS grey-ancestor irreducible) loop back-edges
    /// `(from, to)` reachable from `BlockId(0)`.
    pub fn find_loop_backedges(blocks: &[BasicBlock]) -> Vec<(BlockId, BlockId)> {
        Self::find_loop_backedges_from_entry(blocks, BlockId(0))
    }

    /// Finds all dominance-verified (and DFS grey-ancestor irreducible) loop back-edges
    /// `(from, to)` reachable from `entry`.
    pub fn find_loop_backedges_from_entry(
        blocks: &[BasicBlock],
        entry: BlockId,
    ) -> Vec<(BlockId, BlockId)> {
        let n = blocks.len();
        if n == 0 || entry.0 >= n {
            return Vec::new();
        }

        // 1. Collect valid outgoing successor indices per block.
        let mut succs: Vec<Vec<usize>> = Vec::with_capacity(n);
        for block in blocks {
            let mut out = Vec::new();
            if let Some(term) = &block.terminator {
                for succ in Self::terminator_successors(term) {
                    if succ.0 < n {
                        out.push(succ.0);
                    }
                }
            }
            succs.push(out);
        }

        // 2. Iterative DFS from `entry` to compute reachability, RPO order, and
        //    DFS grey-ancestor retreating edges (covers irreducible loops).
        let mut color = alloc::vec![0u8; n]; // 0 = white, 1 = grey, 2 = black
        let mut reachable = alloc::vec![false; n];
        let mut dfs_backedges: HashSet<(usize, usize)> = HashSet::new();
        let mut postorder: Vec<usize> = Vec::with_capacity(n);
        let mut stack: Vec<(usize, usize)> = Vec::with_capacity(n);

        color[entry.0] = 1;
        reachable[entry.0] = true;
        stack.push((entry.0, 0));

        while let Some((u, next_idx)) = stack.last_mut() {
            let u_val = *u;
            if *next_idx < succs[u_val].len() {
                let v = succs[u_val][*next_idx];
                *next_idx += 1;
                if color[v] == 1 {
                    dfs_backedges.insert((u_val, v));
                } else if color[v] == 0 {
                    color[v] = 1;
                    reachable[v] = true;
                    stack.push((v, 0));
                }
            } else {
                color[u_val] = 2;
                postorder.push(u_val);
                stack.pop();
            }
        }

        // 3. Build reachable-only predecessor lists.
        let mut preds: Vec<Vec<usize>> = alloc::vec![Vec::new(); n];
        for (u, out) in succs.iter().enumerate() {
            if !reachable[u] {
                continue;
            }
            for &v in out {
                preds[v].push(u);
            }
        }

        // 4. Iterative bitset dominance over reachable blocks in RPO order.
        let words = n.div_ceil(64);
        let last_word_mask = if n % 64 == 0 {
            !0u64
        } else {
            (1u64 << (n % 64)) - 1
        };
        let mut dom = alloc::vec![0u64; n * words];

        for b in 0..n {
            if !reachable[b] {
                continue;
            }
            let base = b * words;
            if b == entry.0 {
                dom[base + b / 64] = 1u64 << (b % 64);
            } else {
                for w in 0..words {
                    dom[base + w] = if w + 1 == words { last_word_mask } else { !0u64 };
                }
            }
        }

        let mut rpo = postorder;
        rpo.reverse();
        let mut scratch = alloc::vec![0u64; words];

        loop {
            let mut changed = false;
            for &b in &rpo {
                if b == entry.0 || preds[b].is_empty() {
                    continue;
                }
                for w in 0..words {
                    scratch[w] = if w + 1 == words { last_word_mask } else { !0u64 };
                }
                for &p in &preds[b] {
                    let p_base = p * words;
                    for w in 0..words {
                        scratch[w] &= dom[p_base + w];
                    }
                }
                scratch[b / 64] |= 1u64 << (b % 64);

                let b_base = b * words;
                let mut diff = false;
                for w in 0..words {
                    if dom[b_base + w] != scratch[w] {
                        dom[b_base + w] = scratch[w];
                        diff = true;
                    }
                }
                if diff {
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // 5. Emit backedges in deterministic block/successor order:
        //    `src -> succ` is a backedge iff `src` is reachable AND
        //    (`succ` dominates `src` OR `(src, succ)` is a DFS grey-ancestor edge).
        let mut backedges = Vec::new();
        for (src_idx, out) in succs.iter().enumerate() {
            if !reachable[src_idx] {
                continue;
            }
            let src_base = src_idx * words;
            for &succ_idx in out {
                let succ_dominates_src =
                    (dom[src_base + succ_idx / 64] & (1u64 << (succ_idx % 64))) != 0;
                if succ_dominates_src || dfs_backedges.contains(&(src_idx, succ_idx)) {
                    backedges.push((BlockId(src_idx), BlockId(succ_idx)));
                }
            }
        }
        backedges
    }

    /// Finds all dominance-verified loop back-edges in `func` using `func.entry`.
    pub(crate) fn find_loop_backedges_in_fn(func: &MirFunction) -> Vec<(BlockId, BlockId)> {
        Self::find_loop_backedges_from_entry(&func.blocks, func.entry)
    }

    /// Extracts all successor BlockIds from a block terminator.
    pub fn terminator_successors(term: &Terminator) -> Vec<BlockId> {
        match term {
            Terminator::Goto(target, _) => alloc::vec![*target],
            Terminator::Cond {
                then_block,
                else_block,
                ..
            } => alloc::vec![*then_block, *else_block],
            Terminator::Switch { cases, default, .. } => {
                let mut succs = Vec::with_capacity(cases.len() + 1);
                for (_, target) in cases {
                    succs.push(*target);
                }
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
            Terminator::Return(..) | Terminator::Raise(..) | Terminator::Unreachable(..) => {
                Vec::new()
            }
        }
    }

    /// Runs safepoint insertion on a single MIR function with default configuration.
    /// Returns `true` if any safepoint was inserted.
    pub fn run_on_function(func: &mut MirFunction) -> bool {
        Self::run_on_function_with_config(func, &SafepointConfig::default()) > 0
    }

    /// Runs safepoint insertion on a single MIR function with custom configuration.
    /// Returns the number of safepoints inserted.
    pub fn run_on_function_with_config(
        func: &mut MirFunction,
        config: &SafepointConfig,
    ) -> usize {
        if func.blocks.is_empty() {
            return 0;
        }

        let mut inserted = 0;
        let loop_headers = if config.instrument_loops {
            Self::find_loop_headers_in_fn(func)
        } else {
            HashSet::new()
        };

        // 1. Function entry safepoint
        if config.instrument_entry && func.entry.0 < func.blocks.len() {
            let block = &mut func.blocks[func.entry.0];
            let first_check = block.statements.first_mut().and_then(|s| match s {
                Statement::ResourceCheck(k, q, _) if *k == config.resource_kind => Some(q),
                _ => None,
            });
            match first_check {
                Some(q) if *q == config.entry_quantum => {
                    // Already quantized to entry_quantum; idempotent
                }
                Some(q) => {
                    *q = config.entry_quantum;
                    inserted += 1;
                }
                None => {
                    block.statements.insert(
                        0,
                        Statement::ResourceCheck(
                            config.resource_kind,
                            config.entry_quantum,
                            Span::dummy(),
                        ),
                    );
                    inserted += 1;
                }
            }
        }

        // 2. Loop header safepoints
        if config.instrument_loops {
            // Sort loop header indices for deterministic pass order
            let mut sorted_headers: Vec<usize> = loop_headers.into_iter().map(|b| b.0).collect();
            sorted_headers.sort_unstable();

            for header_idx in sorted_headers {
                if header_idx >= func.blocks.len() {
                    continue;
                }

                let is_entry = header_idx == func.entry.0;
                let block = &mut func.blocks[header_idx];

                let target_check = if is_entry {
                    block.statements.get_mut(1).and_then(|s| match s {
                        Statement::ResourceCheck(k, q, _) if *k == config.resource_kind => Some(q),
                        _ => None,
                    })
                } else {
                    block.statements.first_mut().and_then(|s| match s {
                        Statement::ResourceCheck(k, q, _) if *k == config.resource_kind => Some(q),
                        _ => None,
                    })
                };

                match target_check {
                    Some(q) if *q == config.loop_quantum => {
                        // Already quantized to loop_quantum; idempotent
                    }
                    Some(q) => {
                        *q = config.loop_quantum;
                        inserted += 1;
                    }
                    None => {
                        let insert_pos = if is_entry { 1.min(block.statements.len()) } else { 0 };
                        block.statements.insert(
                            insert_pos,
                            Statement::ResourceCheck(
                                config.resource_kind,
                                config.loop_quantum,
                                Span::dummy(),
                            ),
                        );
                        inserted += 1;
                    }
                }
            }
        }

        inserted
    }

    /// Runs safepoint insertion on all functions in a MIR module with default configuration.
    /// Returns `true` if any safepoint was inserted.
    pub fn run_on_module(module: &mut MirModule) -> bool {
        let stats = Self::run_on_module_with_config(module, &SafepointConfig::default());
        stats.total_safepoints_inserted > 0
    }

    /// Runs safepoint insertion on all functions in a MIR module with custom configuration.
    /// Returns full statistics of safepoints inserted.
    pub fn run_on_module_with_config(
        module: &mut MirModule,
        config: &SafepointConfig,
    ) -> SafepointStats {
        let mut stats = SafepointStats {
            functions_visited: module.functions.len(),
            ..Default::default()
        };

        for func in &mut module.functions {
            let initial_has_entry = if config.instrument_entry && func.entry.0 < func.blocks.len() {
                func.blocks[func.entry.0].statements.first().map_or(false, |s| {
                    matches!(s, Statement::ResourceCheck(k, q, _) if *k == config.resource_kind && *q == config.entry_quantum)
                })
            } else {
                true
            };

            let headers = if config.instrument_loops {
                Self::find_loop_headers_in_fn(func)
            } else {
                HashSet::new()
            };
            let mut initial_headers_quantized = 0;
            for h in &headers {
                if h.0 < func.blocks.len() {
                    let block = &func.blocks[h.0];
                    let is_entry = h.0 == func.entry.0;
                    let check = if is_entry {
                        block.statements.get(1)
                    } else {
                        block.statements.first()
                    };
                    if let Some(Statement::ResourceCheck(k, q, _)) = check {
                        if *k == config.resource_kind && *q == config.loop_quantum {
                            initial_headers_quantized += 1;
                        }
                    }
                }
            }

            let inserted = Self::run_on_function_with_config(func, config);
            if inserted > 0 {
                stats.functions_instrumented += 1;
                stats.total_safepoints_inserted += inserted;

                let entry_added = if !initial_has_entry && config.instrument_entry {
                    1
                } else {
                    0
                };
                let loop_added = headers.len().saturating_sub(initial_headers_quantized);

                stats.entry_safepoints_inserted += entry_added;
                stats.loop_safepoints_inserted += loop_added;
                stats.total_gas_quantized += (entry_added as u64) * config.entry_quantum
                    + (loop_added as u64) * config.loop_quantum;
            }
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::ir::{FunctionSignature, MirType, Operand};

    fn make_test_function(name: &str) -> MirFunction {
        let sig = FunctionSignature {
            name: name.into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        MirFunction::new(sig, Span::dummy())
    }

    #[test]
    fn test_loop_header_detection() {
        let mut func = make_test_function("test_loop");
        let mut bb1 = BasicBlock::new(BlockId(1));
        bb1.terminator = Some(Terminator::Cond {
            cond: Operand::Constant(crate::mir::ir::MirConstant::Bool(true)),
            then_block: BlockId(2),
            else_block: BlockId(3),
            span: Span::dummy(),
        });
        func.blocks.push(bb1);

        let mut bb2 = BasicBlock::new(BlockId(2));
        bb2.terminator = Some(Terminator::Goto(BlockId(1), Span::dummy()));
        func.blocks.push(bb2);

        let mut bb3 = BasicBlock::new(BlockId(3));
        bb3.terminator = Some(Terminator::Return(None, Span::dummy()));
        func.blocks.push(bb3);

        func.blocks[0].terminator = Some(Terminator::Goto(BlockId(1), Span::dummy()));

        let headers = Safepoints::find_loop_headers(&func.blocks);
        assert_eq!(headers.len(), 1);
        assert!(headers.contains(&BlockId(1)));

        let backedges = Safepoints::find_loop_backedges(&func.blocks);
        assert_eq!(backedges.len(), 1);
        assert_eq!(backedges[0], (BlockId(2), BlockId(1)));
    }

    #[test]
    fn test_safepoint_insertion_and_idempotency() {
        let mut func = make_test_function("test_insert");
        let mut bb1 = BasicBlock::new(BlockId(1));
        bb1.terminator = Some(Terminator::Cond {
            cond: Operand::Constant(crate::mir::ir::MirConstant::Bool(true)),
            then_block: BlockId(2),
            else_block: BlockId(3),
            span: Span::dummy(),
        });
        func.blocks.push(bb1);

        let mut bb2 = BasicBlock::new(BlockId(2));
        bb2.terminator = Some(Terminator::Goto(BlockId(1), Span::dummy()));
        func.blocks.push(bb2);

        let mut bb3 = BasicBlock::new(BlockId(3));
        bb3.terminator = Some(Terminator::Return(None, Span::dummy()));
        func.blocks.push(bb3);

        func.blocks[0].terminator = Some(Terminator::Goto(BlockId(1), Span::dummy()));

        let config = SafepointConfig::new(5, 10);
        let count1 = Safepoints::run_on_function_with_config(&mut func, &config);
        assert_eq!(count1, 2, "Entry (block 0) and loop header (block 1) instrumented");

        assert_eq!(
            func.blocks[0].statements[0],
            Statement::ResourceCheck(ResourceKind::Gas, 5, Span::dummy())
        );

        assert_eq!(
            func.blocks[1].statements[0],
            Statement::ResourceCheck(ResourceKind::Gas, 10, Span::dummy())
        );

        let count2 = Safepoints::run_on_function_with_config(&mut func, &config);
        assert_eq!(count2, 0, "Pass must be idempotent");
    }
}

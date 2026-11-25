//! Reducible CFG Structurer / Relooper for Rust AOT Codegen (`NG-W12-2`).
//!
//! Reconstructs a structured control-flow tree (`StructuredNode`) from a flat Lust MIR
//! `MirFunction` using:
//! - Forward dominator tree (Cooper-Harvey-Kennedy) and natural loop nesting tree (Step A).
//! - Exact immediate post-dominators (`ipdom`) via Cooper-Harvey-Kennedy on the acyclic
//!   reverse CFG (backedges excluded, virtual exit sink) (Step B).
//! - Recursive dominator-guided region structuring with multi-level `LabeledBlock` / `Break`
//!   (resolving Hazard 14 inlined early returns and arbitrary reducible merge patterns with
//!   zero code duplication and zero `StateMachineFallback`) and `IfArrayGetOk` recognition
//!   (Step C).
//! - Isolated `StateMachineFallback` emission solely for genuinely irreducible subgraphs.

use crate::mir::ir::{
    BlockId, LocalId, MirBinaryOp, MirConstant, MirFunction, MirUnaryOp, Operand, Place, Rvalue,
    Statement, Terminator,
};
use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};

/// Structured control-flow tree reconstructed from a `MirFunction`.
#[derive(Debug, Clone, PartialEq)]
pub enum StructuredNode {
    /// Linear basic block (its statements, and any non-branching terminator such as `Call`,
    /// `Raise`, or `Unreachable`).
    BasicBlock(BlockId),
    /// Sequential list of structured nodes.
    Seq(Vec<StructuredNode>),
    /// Two-way conditional branch. `cond_block` holds all statements evaluated before the
    /// branch condition `cond_op`.
    If {
        cond_block: BlockId,
        cond_op: Operand,
        then_branch: Box<StructuredNode>,
        else_branch: Option<Box<StructuredNode>>,
    },
    /// Specialized bounds-checked array element fetch (`if arr[idx] is Ok(binding)`),
    /// lowering cleanly to `if let Some(&binding) = arr.get(idx as usize)` in Rust AOT.
    /// `header_block` preserves any non-bounds-check statements before the check, and
    /// `then_branch` preserves the full structured body starting at `then_block`.
    IfArrayGetOk {
        header_block: BlockId,
        arr_local: LocalId,
        idx_local: LocalId,
        binding_local: LocalId,
        then_branch: Box<StructuredNode>,
        else_branch: Option<Box<StructuredNode>>,
    },
    /// Multi-way integer switch.
    Switch {
        discr_block: BlockId,
        discr_op: Operand,
        cases: Vec<(i64, StructuredNode)>,
        default: Box<StructuredNode>,
    },
    /// Header-tested natural loop (`while`). `header_block` is ALWAYS preserved so that
    /// condition evaluation statements and gas/resource safepoints in the header are never lost.
    /// When `negate_cond` is `false`, the loop continues while `cond_op` is true; when
    /// `negate_cond` is `true`, the loop continues while `cond_op` is false.
    WhileLoop {
        label: BlockId,
        header_block: BlockId,
        cond_op: Operand,
        negate_cond: bool,
        body: Box<StructuredNode>,
    },
    /// General natural loop (`'bbH: loop { ... }`), exited via `Break(label)`.
    Loop {
        label: BlockId,
        body: Box<StructuredNode>,
    },
    /// Forward-exit labeled block (`'bbM: { ... }`), targeted by `Break(label)`.
    LabeledBlock {
        label: BlockId,
        body: Box<StructuredNode>,
    },
    /// Exits the enclosing `WhileLoop`, `Loop`, or `LabeledBlock` identified by `BlockId`.
    Break(BlockId),
    /// Jumps back to the header of the enclosing `WhileLoop` or `Loop` identified by `BlockId`.
    Continue(BlockId),
    /// Returns from the function after executing statements in `BlockId`.
    Return(BlockId, Option<Operand>),
    /// Local state-machine fallback used only for irreducible CFG subgraphs (or calls with
    /// explicit unwind blocks). Never emitted for reducible CFGs.
    StateMachineFallback {
        entry: BlockId,
        blocks: Vec<BlockId>,
        exit: Option<BlockId>,
    },
}

impl StructuredNode {
    /// Returns `true` if this node is an empty `Seq`.
    pub fn is_empty_seq(&self) -> bool {
        matches!(self, StructuredNode::Seq(items) if items.is_empty())
    }

    /// Counts all `StateMachineFallback` nodes in this subtree.
    pub fn count_state_machine_fallbacks(&self) -> usize {
        match self {
            StructuredNode::StateMachineFallback { .. } => 1,
            StructuredNode::Seq(items) => {
                items.iter().map(Self::count_state_machine_fallbacks).sum()
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
                then_branch.count_state_machine_fallbacks()
                    + else_branch
                        .as_deref()
                        .map_or(0, Self::count_state_machine_fallbacks)
            }
            StructuredNode::Switch { cases, default, .. } => {
                cases
                    .iter()
                    .map(|(_, node)| node.count_state_machine_fallbacks())
                    .sum::<usize>()
                    + default.count_state_machine_fallbacks()
            }
            StructuredNode::WhileLoop { body, .. }
            | StructuredNode::Loop { body, .. }
            | StructuredNode::LabeledBlock { body, .. } => body.count_state_machine_fallbacks(),
            StructuredNode::BasicBlock(_)
            | StructuredNode::Break(_)
            | StructuredNode::Continue(_)
            | StructuredNode::Return(_, _) => 0,
        }
    }

    /// Counts all `IfArrayGetOk` nodes in this subtree.
    pub fn count_if_array_get_ok(&self) -> usize {
        match self {
            StructuredNode::IfArrayGetOk {
                then_branch,
                else_branch,
                ..
            } => {
                1 + then_branch.count_if_array_get_ok()
                    + else_branch
                        .as_deref()
                        .map_or(0, Self::count_if_array_get_ok)
            }
            StructuredNode::Seq(items) => items.iter().map(Self::count_if_array_get_ok).sum(),
            StructuredNode::If {
                then_branch,
                else_branch,
                ..
            } => {
                then_branch.count_if_array_get_ok()
                    + else_branch
                        .as_deref()
                        .map_or(0, Self::count_if_array_get_ok)
            }
            StructuredNode::Switch { cases, default, .. } => {
                cases
                    .iter()
                    .map(|(_, node)| node.count_if_array_get_ok())
                    .sum::<usize>()
                    + default.count_if_array_get_ok()
            }
            StructuredNode::WhileLoop { body, .. }
            | StructuredNode::Loop { body, .. }
            | StructuredNode::LabeledBlock { body, .. } => body.count_if_array_get_ok(),
            StructuredNode::BasicBlock(_)
            | StructuredNode::Break(_)
            | StructuredNode::Continue(_)
            | StructuredNode::Return(_, _)
            | StructuredNode::StateMachineFallback { .. } => 0,
        }
    }

    /// Counts all `LabeledBlock` nodes in this subtree.
    pub fn count_labeled_blocks(&self) -> usize {
        match self {
            StructuredNode::LabeledBlock { body, .. } => 1 + body.count_labeled_blocks(),
            StructuredNode::Seq(items) => items.iter().map(Self::count_labeled_blocks).sum(),
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
                then_branch.count_labeled_blocks()
                    + else_branch
                        .as_deref()
                        .map_or(0, Self::count_labeled_blocks)
            }
            StructuredNode::Switch { cases, default, .. } => {
                cases
                    .iter()
                    .map(|(_, node)| node.count_labeled_blocks())
                    .sum::<usize>()
                    + default.count_labeled_blocks()
            }
            StructuredNode::WhileLoop { body, .. } | StructuredNode::Loop { body, .. } => {
                body.count_labeled_blocks()
            }
            StructuredNode::BasicBlock(_)
            | StructuredNode::Break(_)
            | StructuredNode::Continue(_)
            | StructuredNode::Return(_, _)
            | StructuredNode::StateMachineFallback { .. } => 0,
        }
    }

    /// Counts all `Break` nodes in this subtree.
    pub fn count_breaks(&self) -> usize {
        match self {
            StructuredNode::Break(_) => 1,
            StructuredNode::Seq(items) => items.iter().map(Self::count_breaks).sum(),
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
                then_branch.count_breaks()
                    + else_branch.as_deref().map_or(0, Self::count_breaks)
            }
            StructuredNode::Switch { cases, default, .. } => {
                cases
                    .iter()
                    .map(|(_, node)| node.count_breaks())
                    .sum::<usize>()
                    + default.count_breaks()
            }
            StructuredNode::WhileLoop { body, .. }
            | StructuredNode::Loop { body, .. }
            | StructuredNode::LabeledBlock { body, .. } => body.count_breaks(),
            StructuredNode::BasicBlock(_)
            | StructuredNode::Continue(_)
            | StructuredNode::Return(_, _)
            | StructuredNode::StateMachineFallback { .. } => 0,
        }
    }

    /// Collects all MIR `BlockId`s whose statements are represented by this subtree.
    pub fn collect_referenced_blocks(&self, out: &mut HashSet<BlockId>) {
        match self {
            StructuredNode::BasicBlock(bb) | StructuredNode::Return(bb, _) => {
                out.insert(*bb);
            }
            StructuredNode::Seq(items) => {
                for item in items {
                    item.collect_referenced_blocks(out);
                }
            }
            StructuredNode::If {
                cond_block,
                then_branch,
                else_branch,
                ..
            } => {
                out.insert(*cond_block);
                then_branch.collect_referenced_blocks(out);
                if let Some(eb) = else_branch {
                    eb.collect_referenced_blocks(out);
                }
            }
            StructuredNode::IfArrayGetOk {
                header_block,
                then_branch,
                else_branch,
                ..
            } => {
                out.insert(*header_block);
                then_branch.collect_referenced_blocks(out);
                if let Some(eb) = else_branch {
                    eb.collect_referenced_blocks(out);
                }
            }
            StructuredNode::Switch {
                discr_block,
                cases,
                default,
                ..
            } => {
                out.insert(*discr_block);
                for (_, node) in cases {
                    node.collect_referenced_blocks(out);
                }
                default.collect_referenced_blocks(out);
            }
            StructuredNode::WhileLoop {
                header_block, body, ..
            } => {
                out.insert(*header_block);
                body.collect_referenced_blocks(out);
            }
            StructuredNode::Loop { body, .. } | StructuredNode::LabeledBlock { body, .. } => {
                body.collect_referenced_blocks(out);
            }
            StructuredNode::StateMachineFallback { entry, blocks, .. } => {
                out.insert(*entry);
                for bb in blocks {
                    out.insert(*bb);
                }
            }
            StructuredNode::Break(_) | StructuredNode::Continue(_) => {}
        }
    }
}

/// Natural loop metadata discovered in Step A.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopInfo {
    pub header: BlockId,
    /// All blocks belonging to this natural loop (including `header` and nested loops).
    pub body_blocks: Vec<BlockId>,
    /// Fast membership lookup for `body_blocks`.
    pub body_set: HashSet<BlockId>,
    /// Latch blocks `u` with a backedge `(u -> header)`.
    pub latches: Vec<BlockId>,
    /// All blocks outside `body_set` targeted by edges originating inside `body_set`.
    pub exit_targets: Vec<BlockId>,
    /// Primary exit block of the loop (unique exit, header condition exit, or earliest exit).
    pub primary_exit: Option<BlockId>,
    /// Immediate enclosing natural loop header, if nested.
    pub parent_loop: Option<BlockId>,
    /// Direct child natural loop headers nested inside this loop.
    pub child_loops: Vec<BlockId>,
}

/// Complete CFG analysis result (dominators, post-dominators `ipdom`, natural loop tree,
/// and reducibility status).
#[derive(Debug, Clone)]
pub struct CfgAnalysis {
    pub num_blocks: usize,
    pub entry: BlockId,
    pub reachable: Vec<bool>,
    /// Immediate forward dominator for each block (`None` for entry and unreachable blocks).
    pub idom: Vec<Option<BlockId>>,
    /// Immediate post-dominator on the backedge-free DAG (`VIRTUAL_EXIT` maps to `None`).
    pub ipdom: Vec<Option<BlockId>>,
    /// Forward Reverse Post-Order position for reachable blocks (valid DAG topological order).
    pub fwd_rpo_pos: Vec<usize>,
    /// Natural loop backedges `(latch, header)` where `header` dominates `latch`.
    pub backedges: Vec<(BlockId, BlockId)>,
    /// Retreating edges `(src, dst)` where `dst` does not dominate `src` (empty iff reducible).
    pub irreducible_edges: Vec<(BlockId, BlockId)>,
    /// Natural loops keyed by their header `BlockId`.
    pub loops: HashMap<BlockId, LoopInfo>,
    /// `true` iff the reachable CFG is reducible.
    pub is_reducible: bool,
    /// Forward (non-backedge) incoming edge count per block (counts duplicate switch arms).
    fwd_in_degree: Vec<usize>,
    /// Immediate dominator tree children per block.
    dom_children: Vec<Vec<usize>>,
    /// Isolated irreducible regions keyed by their dominating region entry block.
    irreducible_regions: HashMap<BlockId, (Vec<BlockId>, Option<BlockId>)>,
}

/// Runs Steps A & B of the Cooper-Harvey-Kennedy CFG analysis on `func`:
/// forward dominators, natural loop nesting tree, exact acyclic `ipdom`, and reducibility check.
pub fn analyze_cfg(func: &MirFunction) -> CfgAnalysis {
    let n = func.blocks.len();
    let entry = func.entry;

    if n == 0 || entry.0 >= n {
        return CfgAnalysis {
            num_blocks: n,
            entry,
            reachable: vec![false; n],
            idom: vec![None; n],
            ipdom: vec![None; n],
            fwd_rpo_pos: vec![usize::MAX; n],
            backedges: Vec::new(),
            irreducible_edges: Vec::new(),
            loops: HashMap::new(),
            is_reducible: true,
            fwd_in_degree: vec![0; n],
            dom_children: vec![Vec::new(); n],
            irreducible_regions: HashMap::new(),
        };
    }

    // 1. Extract forward successor lists (preserving order, deduplicated for graph traversals)
    //    and raw successor lists (preserving switch arm multiplicity for merge detection).
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut raw_succs: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (idx, block) in func.blocks.iter().enumerate() {
        if let Some(term) = &block.terminator {
            for s in term.successors() {
                if s < n {
                    raw_succs[idx].push(s);
                    if !succs[idx].contains(&s) {
                        succs[idx].push(s);
                    }
                }
            }
        }
    }

    // 2. Iterative forward DFS from `entry` to compute reachability, RPO, and DFS retreating edges.
    let mut reachable = vec![false; n];
    let mut on_stack = vec![false; n];
    let mut postorder = Vec::with_capacity(n);
    let mut retreating_edges: HashSet<(usize, usize)> = HashSet::new();
    let mut dfs_stack: Vec<(usize, usize)> = Vec::with_capacity(n);

    reachable[entry.0] = true;
    on_stack[entry.0] = true;
    dfs_stack.push((entry.0, 0));

    while let Some((u, next_i)) = dfs_stack.last_mut() {
        let u_val = *u;
        if *next_i < succs[u_val].len() {
            let v = succs[u_val][*next_i];
            *next_i += 1;
            if !reachable[v] {
                reachable[v] = true;
                on_stack[v] = true;
                dfs_stack.push((v, 0));
            } else if on_stack[v] {
                retreating_edges.insert((u_val, v));
            }
        } else {
            on_stack[u_val] = false;
            postorder.push(u_val);
            dfs_stack.pop();
        }
    }

    let mut fwd_rpo = postorder;
    fwd_rpo.reverse();
    let mut fwd_rpo_pos = vec![usize::MAX; n];
    for (pos, &b) in fwd_rpo.iter().enumerate() {
        fwd_rpo_pos[b] = pos;
    }

    // Build reachable predecessor lists.
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for u in 0..n {
        if !reachable[u] {
            continue;
        }
        for &v in &succs[u] {
            preds[v].push(u);
        }
    }

    // 3. Forward Dominators via Cooper-Harvey-Kennedy iterative algorithm.
    let mut raw_idom: Vec<Option<usize>> = vec![None; n];
    raw_idom[entry.0] = Some(entry.0);
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &fwd_rpo {
            if b == entry.0 {
                continue;
            }
            let mut new_idom: Option<usize> = None;
            for &p in &preds[b] {
                if raw_idom[p].is_some() {
                    new_idom = match new_idom {
                        None => Some(p),
                        Some(acc) => Some(intersect(acc, p, &raw_idom, &fwd_rpo_pos)),
                    };
                }
            }
            if let Some(d) = new_idom
                && raw_idom[b] != Some(d)
            {
                raw_idom[b] = Some(d);
                changed = true;
            }
        }
    }

    let mut idom = vec![None; n];
    let mut dom_children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for b in 0..n {
        if b != entry.0
            && let Some(d) = raw_idom[b]
        {
            idom[b] = Some(BlockId(d));
            dom_children[d].push(b);
        }
    }

    // 4. Step A: Classify natural backedges (`header` dominates `latch`) vs irreducible edges.
    let mut backedges_set: HashSet<(usize, usize)> = HashSet::new();
    let mut backedges: Vec<(BlockId, BlockId)> = Vec::new();
    let mut irreducible_edges: Vec<(BlockId, BlockId)> = Vec::new();
    let mut header_latches: HashMap<usize, Vec<usize>> = HashMap::new();

    for &u in &fwd_rpo {
        for &v in &succs[u] {
            if dominates(v, u, &raw_idom, entry.0) {
                if backedges_set.insert((u, v)) {
                    backedges.push((BlockId(u), BlockId(v)));
                    header_latches.entry(v).or_default().push(u);
                }
            } else if retreating_edges.contains(&(u, v)) {
                irreducible_edges.push((BlockId(u), BlockId(v)));
            }
        }
    }

    // Compute forward in-degree (excluding natural backedges and irreducible retreating edges).
    let mut fwd_in_degree = vec![0usize; n];
    for &u in &fwd_rpo {
        for &v in &raw_succs[u] {
            if !backedges_set.contains(&(u, v)) && !retreating_edges.contains(&(u, v)) {
                fwd_in_degree[v] += 1;
            }
        }
    }

    // Compute natural loop bodies via reverse BFS from latches to header.
    let mut loops: HashMap<BlockId, LoopInfo> = HashMap::new();
    for (&hdr, latches) in &header_latches {
        let mut body_set: HashSet<BlockId> = HashSet::new();
        let mut queue: VecDeque<usize> = VecDeque::new();
        body_set.insert(BlockId(hdr));
        for &latch in latches {
            if body_set.insert(BlockId(latch)) {
                queue.push_back(latch);
            }
        }
        while let Some(node) = queue.pop_front() {
            for &p in &preds[node] {
                if body_set.insert(BlockId(p)) {
                    queue.push_back(p);
                }
            }
        }

        let mut body_blocks: Vec<BlockId> = body_set.iter().copied().collect();
        body_blocks.sort_by_key(|b| fwd_rpo_pos[b.0]);

        let mut exit_set: HashSet<BlockId> = HashSet::new();
        for &b in &body_blocks {
            for &s in &succs[b.0] {
                let s_id = BlockId(s);
                if !body_set.contains(&s_id) {
                    exit_set.insert(s_id);
                }
            }
        }
        let mut exit_targets: Vec<BlockId> = exit_set.into_iter().collect();
        exit_targets.sort_by_key(|b| (fwd_rpo_pos[b.0], b.0));

        let primary_exit = choose_primary_loop_exit(func, BlockId(hdr), &body_set, &exit_targets);

        let mut sorted_latches: Vec<BlockId> = latches.iter().map(|&l| BlockId(l)).collect();
        sorted_latches.sort_by_key(|b| (fwd_rpo_pos[b.0], b.0));

        loops.insert(
            BlockId(hdr),
            LoopInfo {
                header: BlockId(hdr),
                body_blocks,
                body_set,
                latches: sorted_latches,
                exit_targets,
                primary_exit,
                parent_loop: None,
                child_loops: Vec::new(),
            },
        );
    }

    // Build natural loop nesting tree: L1 is nested inside L2 iff L1.body ⊂ L2.body.
    let loop_headers: Vec<BlockId> = loops.keys().copied().collect();
    let mut parent_map: HashMap<BlockId, BlockId> = HashMap::new();
    for &h1 in &loop_headers {
        let size1 = loops[&h1].body_set.len();
        let mut best_parent: Option<(BlockId, usize)> = None;
        for &h2 in &loop_headers {
            if h1 == h2 {
                continue;
            }
            let info2 = &loops[&h2];
            if info2.body_set.len() > size1 && info2.body_set.contains(&h1) {
                if best_parent.is_none_or(|(_, best_sz)| info2.body_set.len() < best_sz) {
                    best_parent = Some((h2, info2.body_set.len()));
                }
            }
        }
        if let Some((parent_hdr, _)) = best_parent {
            parent_map.insert(h1, parent_hdr);
        }
    }
    for (child, parent) in parent_map {
        if let Some(info) = loops.get_mut(&child) {
            info.parent_loop = Some(parent);
        }
        if let Some(p_info) = loops.get_mut(&parent) {
            p_info.child_loops.push(child);
        }
    }
    for info in loops.values_mut() {
        info.child_loops.sort_by_key(|b| (fwd_rpo_pos[b.0], b.0));
    }

    // Check exact reducibility using Tarjan's SCC on the backedge-free graph.
    let non_backedge_succs: Vec<Vec<usize>> = (0..n)
        .map(|u| {
            if !reachable[u] {
                Vec::new()
            } else {
                succs[u]
                    .iter()
                    .copied()
                    .filter(|&v| !backedges_set.contains(&(u, v)))
                    .collect()
            }
        })
        .collect();
    let irreducible_sccs = find_nontrivial_sccs(&fwd_rpo, &non_backedge_succs);
    let is_reducible = irreducible_sccs.is_empty() && irreducible_edges.is_empty();

    // 5. Step B: Cooper-Harvey-Kennedy immediate post-dominators (`ipdom`) on the acyclic DAG.
    //    Backedges (and any irreducible retreating edges) are omitted so every loop and
    //    subgraph has a well-defined path to `VIRTUAL_EXIT = n`.
    let virtual_exit = n;
    let mut dag_succs: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    let mut rev_succs: Vec<Vec<usize>> = vec![Vec::new(); n + 1];

    for &b in &fwd_rpo {
        for &s in &succs[b] {
            if !backedges_set.contains(&(b, s)) && !retreating_edges.contains(&(b, s)) {
                dag_succs[b].push(s);
            }
        }
        if dag_succs[b].is_empty() {
            dag_succs[b].push(virtual_exit);
        }
        for &s in &dag_succs[b] {
            rev_succs[s].push(b);
        }
    }

    // Reverse Post-Order from `VIRTUAL_EXIT` over reverse DAG edges `rev_succs`.
    let mut rev_visited = vec![false; n + 1];
    let mut rev_postorder = Vec::with_capacity(n + 1);
    let mut rev_stack: Vec<(usize, usize)> = Vec::with_capacity(n + 1);
    rev_visited[virtual_exit] = true;
    rev_stack.push((virtual_exit, 0));

    while let Some((u, next_i)) = rev_stack.last_mut() {
        let u_val = *u;
        if *next_i < rev_succs[u_val].len() {
            let v = rev_succs[u_val][*next_i];
            *next_i += 1;
            if !rev_visited[v] {
                rev_visited[v] = true;
                rev_stack.push((v, 0));
            }
        } else {
            rev_postorder.push(u_val);
            rev_stack.pop();
        }
    }

    let mut rev_rpo = rev_postorder;
    rev_rpo.reverse();
    let mut rev_rpo_pos = vec![usize::MAX; n + 1];
    for (pos, &node) in rev_rpo.iter().enumerate() {
        rev_rpo_pos[node] = pos;
    }

    let mut raw_ipdom: Vec<Option<usize>> = vec![None; n + 1];
    raw_ipdom[virtual_exit] = Some(virtual_exit);
    let mut pdom_changed = true;
    while pdom_changed {
        pdom_changed = false;
        for &b in &rev_rpo {
            if b == virtual_exit {
                continue;
            }
            let mut new_ipdom: Option<usize> = None;
            for &s in &dag_succs[b] {
                if raw_ipdom[s].is_some() {
                    new_ipdom = match new_ipdom {
                        None => Some(s),
                        Some(acc) => Some(intersect(acc, s, &raw_ipdom, &rev_rpo_pos)),
                    };
                }
            }
            if let Some(p) = new_ipdom
                && raw_ipdom[b] != Some(p)
            {
                raw_ipdom[b] = Some(p);
                pdom_changed = true;
            }
        }
    }

    let mut ipdom: Vec<Option<BlockId>> = vec![None; n];
    for b in 0..n {
        if let Some(p) = raw_ipdom[b]
            && p != virtual_exit
        {
            ipdom[b] = Some(BlockId(p));
        }
    }

    // 6. If irreducible SCCs exist, isolate each irreducible region under its dominating entry
    //    so only that subgraph uses `StateMachineFallback`.
    let mut irreducible_regions: HashMap<BlockId, (Vec<BlockId>, Option<BlockId>)> = HashMap::new();
    for scc in &irreducible_sccs {
        let scc_set: HashSet<usize> = scc.iter().copied().collect();
        let mut entries: Vec<usize> = Vec::new();
        for &node in scc {
            if node == entry.0 || preds[node].iter().any(|p| !scc_set.contains(p)) {
                entries.push(node);
            }
        }
        if entries.is_empty() {
            entries.push(scc[0]);
        }
        let mut region_entry = entries[0];
        for &e in &entries[1..] {
            region_entry = intersect(region_entry, e, &raw_idom, &fwd_rpo_pos);
        }
        let region_exit = ipdom[region_entry];

        // Collect all blocks reachable from `region_entry` before reaching `region_exit`
        // (and within the domain dominated by `region_entry`).
        let mut region_blocks_set: HashSet<usize> = HashSet::new();
        let mut q: VecDeque<usize> = VecDeque::new();
        region_blocks_set.insert(region_entry);
        q.push_back(region_entry);
        while let Some(u) = q.pop_front() {
            for &v in &succs[u] {
                if Some(BlockId(v)) == region_exit {
                    continue;
                }
                if dominates(region_entry, v, &raw_idom, entry.0) && region_blocks_set.insert(v) {
                    q.push_back(v);
                }
            }
        }

        let mut region_blocks: Vec<BlockId> =
            region_blocks_set.into_iter().map(BlockId).collect();
        region_blocks.sort_by_key(|b| (fwd_rpo_pos[b.0], b.0));
        irreducible_regions.insert(BlockId(region_entry), (region_blocks, region_exit));
    }

    CfgAnalysis {
        num_blocks: n,
        entry,
        reachable,
        idom,
        ipdom,
        fwd_rpo_pos,
        backedges,
        irreducible_edges,
        loops,
        is_reducible,
        fwd_in_degree,
        dom_children,
        irreducible_regions,
    }
}

/// Cooper-Harvey-Kennedy lowest-common-ancestor intersection on a dominator / post-dominator tree.
fn intersect(mut b1: usize, mut b2: usize, dom: &[Option<usize>], rpo_pos: &[usize]) -> usize {
    while b1 != b2 {
        while rpo_pos[b1] > rpo_pos[b2] {
            b1 = dom[b1].expect("CFG dominator tree invariant: node must have dominator");
        }
        while rpo_pos[b2] > rpo_pos[b1] {
            b2 = dom[b2].expect("CFG dominator tree invariant: node must have dominator");
        }
    }
    b1
}

/// Returns `true` iff `dom_cand` dominates `node` in the forward dominator tree.
fn dominates(dom_cand: usize, mut node: usize, raw_idom: &[Option<usize>], entry: usize) -> bool {
    loop {
        if node == dom_cand {
            return true;
        }
        if node == entry {
            return false;
        }
        match raw_idom[node] {
            Some(parent) if parent != node => node = parent,
            _ => return false,
        }
    }
}

fn choose_primary_loop_exit(
    func: &MirFunction,
    header: BlockId,
    body_set: &HashSet<BlockId>,
    exit_targets: &[BlockId],
) -> Option<BlockId> {
    if exit_targets.is_empty() {
        return None;
    }
    if exit_targets.len() == 1 {
        return Some(exit_targets[0]);
    }
    // Prefer the direct exit edge of a `while`-style header condition.
    if let Some(Terminator::Cond {
        then_block,
        else_block,
        ..
    }) = &func.blocks[header.0].terminator
    {
        let then_in = body_set.contains(then_block);
        let else_in = body_set.contains(else_block);
        if then_in && !else_in {
            return Some(*else_block);
        }
        if else_in && !then_in {
            return Some(*then_block);
        }
    }
    // Otherwise pick the earliest exit in forward topological order.
    exit_targets.first().copied()
}

/// Iterative Tarjan's SCC finder returning all strongly connected components of size >= 2.
fn find_nontrivial_sccs(nodes: &[usize], adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adj.len();
    let mut index = vec![usize::MAX; n];
    let mut lowlink = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut tarjan_stack: Vec<usize> = Vec::new();
    let mut next_index = 0usize;
    let mut result = Vec::new();

    // Frame: (node, next_edge_idx)
    let mut call_stack: Vec<(usize, usize)> = Vec::new();

    for &start in nodes {
        if index[start] != usize::MAX {
            continue;
        }
        index[start] = next_index;
        lowlink[start] = next_index;
        next_index += 1;
        tarjan_stack.push(start);
        on_stack[start] = true;
        call_stack.push((start, 0));

        while let Some((u, edge_i)) = call_stack.last_mut() {
            let u_val = *u;
            if *edge_i < adj[u_val].len() {
                let v = adj[u_val][*edge_i];
                *edge_i += 1;
                if index[v] == usize::MAX {
                    index[v] = next_index;
                    lowlink[v] = next_index;
                    next_index += 1;
                    tarjan_stack.push(v);
                    on_stack[v] = true;
                    call_stack.push((v, 0));
                } else if on_stack[v] && index[v] < lowlink[u_val] {
                    lowlink[u_val] = index[v];
                }
            } else {
                call_stack.pop();
                if let Some((parent, _)) = call_stack.last()
                    && lowlink[u_val] < lowlink[*parent]
                {
                    lowlink[*parent] = lowlink[u_val];
                }
                if lowlink[u_val] == index[u_val] {
                    let mut comp = Vec::new();
                    while let Some(w) = tarjan_stack.pop() {
                        on_stack[w] = false;
                        comp.push(w);
                        if w == u_val {
                            break;
                        }
                    }
                    if comp.len() >= 2 {
                        result.push(comp);
                    }
                }
            }
        }
    }

    result
}

/// Active enclosing loop scope tracked during Step C recursive structuring.
#[derive(Debug, Clone, Copy)]
struct LoopScope {
    header: BlockId,
    exit: Option<BlockId>,
}

/// Reconstructs a `StructuredNode` control-flow tree for `func`.
pub fn structure_mir_function(func: &MirFunction) -> StructuredNode {
    if func.blocks.is_empty() || func.entry.0 >= func.blocks.len() {
        return StructuredNode::Seq(Vec::new());
    }
    let analysis = analyze_cfg(func);
    let mut builder = Structurer {
        func,
        analysis: &analysis,
        active_expansion: HashSet::new(),
    };
    builder.structure_region(func.entry, None, &[], &[])
}

struct Structurer<'a> {
    func: &'a MirFunction,
    analysis: &'a CfgAnalysis,
    /// Guard against cyclic recursion on malformed/unreducible inputs.
    active_expansion: HashSet<BlockId>,
}

impl<'a> Structurer<'a> {
    /// Step C: Recursive region structurer.
    ///
    /// Invariants:
    /// - `stop_at` is the exact `BlockId` that will execute immediately if control falls
    ///   off the end of this region.
    /// - `active_loops` lists enclosing loops from outermost to innermost.
    /// - `active_merges` lists open enclosing forward `LabeledBlock` merge targets.
    fn structure_region(
        &mut self,
        curr: BlockId,
        stop_at: Option<BlockId>,
        active_loops: &[LoopScope],
        active_merges: &[BlockId],
    ) -> StructuredNode {
        // 1. Reached the immediate fallthrough continuation block.
        if Some(curr) == stop_at {
            return StructuredNode::Seq(Vec::new());
        }

        // 2. Backedge to an active enclosing loop header -> Continue.
        if let Some(scope) = active_loops.iter().rev().find(|l| l.header == curr) {
            return StructuredNode::Continue(scope.header);
        }

        // 3. Exit edge to an active enclosing loop's exit -> Break(loop.header).
        //    Search outermost-to-innermost so exiting multiple loops sharing one exit
        //    jumps directly to the outermost exited loop.
        if let Some(scope) = active_loops.iter().find(|l| l.exit == Some(curr)) {
            return StructuredNode::Break(scope.header);
        }

        // 4. Early forward jump to an enclosing LabeledBlock merge target -> Break(target).
        if active_merges.contains(&curr) {
            return StructuredNode::Break(curr);
        }

        // 5. Irreducible region isolated in Step A/B -> local StateMachineFallback.
        if let Some((blocks, region_exit)) = self.analysis.irreducible_regions.get(&curr) {
            let fallback = StructuredNode::StateMachineFallback {
                entry: curr,
                blocks: blocks.clone(),
                exit: *region_exit,
            };
            if let Some(ex) = *region_exit {
                let tail = self.structure_region(ex, stop_at, active_loops, active_merges);
                return make_seq(vec![fallback, tail]);
            }
            return fallback;
        }

        // 6. Defensive cycle guard (never triggered on reducible CFGs).
        if !self.active_expansion.insert(curr) {
            return StructuredNode::StateMachineFallback {
                entry: curr,
                blocks: vec![curr],
                exit: stop_at,
            };
        }

        let result = if let Some(loop_info) = self.analysis.loops.get(&curr)
            && !active_loops.iter().any(|l| l.header == curr)
        {
            self.structure_loop(loop_info, stop_at, active_loops, active_merges)
        } else {
            self.structure_block_with_merges(curr, stop_at, active_loops, active_merges, false)
        };

        self.active_expansion.remove(&curr);
        result
    }

    /// Structures a natural loop headed by `loop_info.header`, together with any out-of-loop
    /// merge blocks immediately dominated by the header and the loop's continuation `exit_bb`.
    fn structure_loop(
        &mut self,
        loop_info: &LoopInfo,
        stop_at: Option<BlockId>,
        active_loops: &[LoopScope],
        active_merges: &[BlockId],
    ) -> StructuredNode {
        let hdr = loop_info.header;
        let primary_exit = loop_info.primary_exit;

        // Collect out-of-loop merge children immediately dominated by `hdr` (excluding
        // `primary_exit`, which is emitted directly after the loop, and excluding any block
        // already handled by an enclosing scope).
        let out_merges = self.collect_merge_children(
            hdr,
            stop_at,
            active_loops,
            active_merges,
            |m| !loop_info.body_set.contains(&m) && Some(m) != primary_exit,
        );

        let build_loop_and_primary_exit =
            |this: &mut Self, inner_stop: Option<BlockId>, inner_merges: &[BlockId]| {
                let mut next_loops = active_loops.to_vec();
                next_loops.push(LoopScope {
                    header: hdr,
                    exit: primary_exit,
                });

                let loop_node = match &this.func.blocks[hdr.0].terminator {
                    Some(Terminator::Cond {
                        cond,
                        then_block,
                        else_block,
                        ..
                    }) => {
                        let then_in = loop_info.body_set.contains(then_block);
                        let else_in = loop_info.body_set.contains(else_block);

                        if then_in && !else_in && Some(*else_block) == primary_exit {
                            let body = this.structure_region(
                                *then_block,
                                Some(hdr),
                                &next_loops,
                                inner_merges,
                            );
                            StructuredNode::WhileLoop {
                                label: hdr,
                                header_block: hdr,
                                cond_op: cond.clone(),
                                negate_cond: false,
                                body: Box::new(body),
                            }
                        } else if else_in && !then_in && Some(*then_block) == primary_exit {
                            let body = this.structure_region(
                                *else_block,
                                Some(hdr),
                                &next_loops,
                                inner_merges,
                            );
                            StructuredNode::WhileLoop {
                                label: hdr,
                                header_block: hdr,
                                cond_op: cond.clone(),
                                negate_cond: true,
                                body: Box::new(body),
                            }
                        } else {
                            // Both branches stay in the loop (or custom exit structure):
                            // emit `Loop` and structure `hdr` as a branch inside the loop body.
                            let body = this.structure_block_with_merges(
                                hdr,
                                Some(hdr),
                                &next_loops,
                                inner_merges,
                                true,
                            );
                            StructuredNode::Loop {
                                label: hdr,
                                body: Box::new(body),
                            }
                        }
                    }
                    _ => {
                        let body = this.structure_block_with_merges(
                            hdr,
                            Some(hdr),
                            &next_loops,
                            inner_merges,
                            true,
                        );
                        StructuredNode::Loop {
                            label: hdr,
                            body: Box::new(body),
                        }
                    }
                };

                if let Some(exit_bb) = primary_exit {
                    let after_loop =
                        this.structure_region(exit_bb, inner_stop, active_loops, inner_merges);
                    make_seq(vec![loop_node, after_loop])
                } else {
                    loop_node
                }
            };

        self.wrap_with_merge_blocks(
            &out_merges,
            stop_at,
            active_loops,
            active_merges,
            build_loop_and_primary_exit,
        )
    }

    /// Structures a single block `curr` together with all forward merge blocks immediately
    /// dominated by `curr`.
    fn structure_block_with_merges(
        &mut self,
        curr: BlockId,
        stop_at: Option<BlockId>,
        active_loops: &[LoopScope],
        active_merges: &[BlockId],
        inside_own_loop_header: bool,
    ) -> StructuredNode {
        let merge_children = self.collect_merge_children(
            curr,
            stop_at,
            active_loops,
            active_merges,
            |m| {
                if inside_own_loop_header {
                    // When structuring a general `Loop` header's own body, only include
                    // merge children that belong to this loop's body.
                    self.analysis
                        .loops
                        .get(&curr)
                        .is_some_and(|info| info.body_set.contains(&m))
                } else {
                    true
                }
            },
        );

        self.wrap_with_merge_blocks(
            &merge_children,
            stop_at,
            active_loops,
            active_merges,
            |this, inner_stop, inner_merges| {
                this.structure_block_terminator(curr, inner_stop, active_loops, inner_merges)
            },
        )
    }

    /// Wraps a core builder closure inside a stack of merge blocks `merges = [m_1, ..., m_k]`
    /// ordered in forward DAG topological order (`m_1` innermost, `m_k` outermost):
    ///
    /// ```text
    /// 'm_k: {
    ///     ...
    ///     'm_1: {
    ///         build_core(stop_at = Some(m_1), merges = [..., m_k, ..., m_1])
    ///     }
    ///     structure_region(m_1, stop_at = Some(m_2), merges = [..., m_k, ..., m_2])
    ///     ...
    /// }
    /// structure_region(m_k, stop_at = orig_stop_at, merges = orig_merges)
    /// ```
    ///
    /// A `LabeledBlock { label: m_i, .. }` wrapper is emitted only when the inner region
    /// actually contains a `Break(m_i)`; normal diamond fallthroughs stay clean and unwrapped.
    fn wrap_with_merge_blocks<F>(
        &mut self,
        merges: &[BlockId],
        stop_at: Option<BlockId>,
        active_loops: &[LoopScope],
        active_merges: &[BlockId],
        build_core: F,
    ) -> StructuredNode
    where
        F: FnOnce(&mut Self, Option<BlockId>, &[BlockId]) -> StructuredNode,
    {
        if merges.is_empty() {
            return build_core(self, stop_at, active_merges);
        }

        // Push merges outermost-first (`m_k`, ..., `m_1`) onto `inner_merges`.
        let mut inner_merges = active_merges.to_vec();
        for &m in merges.iter().rev() {
            if !inner_merges.contains(&m) {
                inner_merges.push(m);
            }
        }

        let mut acc = build_core(self, Some(merges[0]), &inner_merges);

        for i in 0..merges.len() {
            let m = merges[i];
            // Pop `m` before structuring `m` itself so `m` is expanded rather than emitting `Break(m)`.
            if let Some(pos) = inner_merges.iter().rposition(|&x| x == m) {
                inner_merges.remove(pos);
            }

            if contains_break_to(&acc, m) {
                acc = StructuredNode::LabeledBlock {
                    label: m,
                    body: Box::new(acc),
                };
            }

            let next_stop = merges.get(i + 1).copied().or(stop_at);
            let m_node = self.structure_region(m, next_stop, active_loops, &inner_merges);
            acc = make_seq(vec![acc, m_node]);
        }

        acc
    }

    /// Collects all merge blocks `m` immediately dominated by `curr`, filtered by `pred`,
    /// sorted in forward DAG topological order (`fwd_rpo_pos`), with `ipdom[curr]` placed last
    /// when present so canonical post-dominators form the outermost scope.
    fn collect_merge_children<P>(
        &self,
        curr: BlockId,
        stop_at: Option<BlockId>,
        active_loops: &[LoopScope],
        active_merges: &[BlockId],
        mut pred: P,
    ) -> Vec<BlockId>
    where
        P: FnMut(BlockId) -> bool,
    {
        let mut merges: Vec<BlockId> = self.analysis.dom_children[curr.0]
            .iter()
            .filter_map(|&child_idx| {
                let m = BlockId(child_idx);
                if self.analysis.fwd_in_degree[child_idx] < 2 {
                    return None;
                }
                if Some(m) == stop_at
                    || active_merges.contains(&m)
                    || active_loops
                        .iter()
                        .any(|l| l.header == m || l.exit == Some(m))
                {
                    return None;
                }
                if !pred(m) {
                    return None;
                }
                Some(m)
            })
            .collect();

        let canonical_ipdom = self.analysis.ipdom[curr.0];
        merges.sort_by_key(|m| {
            let is_ipdom = Some(*m) == canonical_ipdom;
            (is_ipdom, self.analysis.fwd_rpo_pos[m.0], m.0)
        });
        merges
    }

    /// Emits the structured node for block `curr`'s terminator (assuming merge blocks have
    /// already been opened by `wrap_with_merge_blocks`).
    fn structure_block_terminator(
        &mut self,
        curr: BlockId,
        stop_at: Option<BlockId>,
        active_loops: &[LoopScope],
        active_merges: &[BlockId],
    ) -> StructuredNode {
        let term = match &self.func.blocks[curr.0].terminator {
            Some(t) => t.clone(),
            None => return StructuredNode::BasicBlock(curr),
        };

        match term {
            Terminator::Goto(target, _) => {
                let next = self.structure_region(target, stop_at, active_loops, active_merges);
                make_seq(vec![StructuredNode::BasicBlock(curr), next])
            }
            Terminator::Call {
                success,
                error: None,
                ..
            } => {
                let next = self.structure_region(success, stop_at, active_loops, active_merges);
                make_seq(vec![StructuredNode::BasicBlock(curr), next])
            }
            Terminator::Call {
                success,
                error: Some(err_bb),
                ..
            } => {
                // Conservatively preserve explicit unwind targets via local fallback.
                let mut blocks = vec![curr];
                if Some(success) != stop_at {
                    blocks.push(success);
                }
                if Some(err_bb) != stop_at && !blocks.contains(&err_bb) {
                    blocks.push(err_bb);
                }
                StructuredNode::StateMachineFallback {
                    entry: curr,
                    blocks,
                    exit: stop_at,
                }
            }
            Terminator::Cond {
                cond,
                then_block,
                else_block,
                ..
            } => {
                if then_block == else_block {
                    let next =
                        self.structure_region(then_block, stop_at, active_loops, active_merges);
                    return make_seq(vec![StructuredNode::BasicBlock(curr), next]);
                }

                // Check for `if arr[idx] is Ok(binding)` pattern (`NG-W12-2` / `NG-W10-2`).
                let can_fold_array_get = Some(then_block) != stop_at
                    && !active_merges.contains(&then_block)
                    && !active_loops
                        .iter()
                        .any(|l| l.header == then_block || l.exit == Some(then_block));

                let array_get_match = if can_fold_array_get {
                    detect_if_array_get_ok(self.func, curr, &cond, then_block)
                } else {
                    None
                };

                let then_node =
                    self.structure_region(then_block, stop_at, active_loops, active_merges);
                let else_node =
                    self.structure_region(else_block, stop_at, active_loops, active_merges);
                let else_branch = if else_node.is_empty_seq() {
                    None
                } else {
                    Some(Box::new(else_node))
                };

                if let Some((arr_local, idx_local, binding_local)) = array_get_match {
                    StructuredNode::IfArrayGetOk {
                        header_block: curr,
                        arr_local,
                        idx_local,
                        binding_local,
                        then_branch: Box::new(then_node),
                        else_branch,
                    }
                } else {
                    StructuredNode::If {
                        cond_block: curr,
                        cond_op: cond,
                        then_branch: Box::new(then_node),
                        else_branch,
                    }
                }
            }
            Terminator::Switch {
                discr,
                cases,
                default,
                ..
            } => {
                let structured_cases = cases
                    .into_iter()
                    .map(|(val, target)| {
                        let node =
                            self.structure_region(target, stop_at, active_loops, active_merges);
                        (val, node)
                    })
                    .collect();
                let default_node =
                    self.structure_region(default, stop_at, active_loops, active_merges);
                StructuredNode::Switch {
                    discr_block: curr,
                    discr_op: discr,
                    cases: structured_cases,
                    default: Box::new(default_node),
                }
            }
            Terminator::Return(opt_op, _) => StructuredNode::Return(curr, opt_op),
            Terminator::Raise(_, _) | Terminator::Unreachable(_) => {
                StructuredNode::BasicBlock(curr)
            }
        }
    }
}

/// Detects the `if arr[idx] is Ok(binding)` pattern on `(header_block, cond_op, then_block)`.
///
/// Primary path: uses `func.proven_in_bounds.get(&then_block.0)` populated by `NG-W10-2`
/// lowering and verifies that `then_block` reads `Place::Index(Place::Local(arr_local), idx_local)`
/// into `Place::Local(binding_local)`.
///
/// Secondary path: recognizes the exact canonical MIR bounds-check sequence in `header_block`
/// (`len = Len(arr)`, `ge_zero = idx >= 0`, `lt_len = idx < len`, `cond = ge_zero && lt_len`,
/// with no intervening mutation of `arr` or `idx`) plus the matching `binding = arr[idx]` read
/// in `then_block`. This preserves `IfArrayGetOk` recognition even when `MirInliner` inlines a
/// callee without propagating `proven_in_bounds` or `DeadCodeElimination` renumbers blocks.
fn detect_if_array_get_ok(
    func: &MirFunction,
    header_block: BlockId,
    cond_op: &Operand,
    then_block: BlockId,
) -> Option<(LocalId, LocalId, LocalId)> {
    // 1. Primary path: `func.proven_in_bounds[then_block]`.
    if let Some(proven_pairs) = func.proven_in_bounds.get(&then_block.0) {
        for &(arr_local, idx_local) in proven_pairs {
            if let Some(binding_local) =
                find_then_array_index_binding(func, then_block, arr_local, idx_local)
            {
                return Some((arr_local, idx_local, binding_local));
            }
        }
    }

    // 2. Secondary path: canonical bounds-check pattern in `header_block` + read in `then_block`.
    if let Some((arr_local, idx_local)) =
        match_header_bounds_check(func, header_block, cond_op)
        && let Some(binding_local) =
            find_then_array_index_binding(func, then_block, arr_local, idx_local)
    {
        return Some((arr_local, idx_local, binding_local));
    }

    None
}

/// Inspects the leading statements of `then_block` for
/// `binding_local = copy/move arr_local[idx_local]`.
fn find_then_array_index_binding(
    func: &MirFunction,
    then_block: BlockId,
    arr_local: LocalId,
    idx_local: LocalId,
) -> Option<LocalId> {
    let block = func.blocks.get(then_block.0)?;
    for stmt in &block.statements {
        match stmt {
            Statement::StorageLive(_, _) | Statement::ResourceCheck(_, _, _) => continue,
            Statement::Assign(Place::Local(binding_local), Rvalue::Use(op), _) => {
                let place = match op {
                    Operand::Copy(p) | Operand::Move(p) => p,
                    Operand::Constant(_) => return None,
                };
                if let Place::Index(base, idx) = place
                    && **base == Place::Local(arr_local)
                    && *idx == idx_local
                {
                    return Some(*binding_local);
                }
                return None;
            }
            _ => return None,
        }
    }
    None
}

/// Matches the canonical MIR bounds check in `header_block`:
/// ```text
/// len_tmp     = Len(copy/move arr)
/// ge_zero_tmp = idx >= 0          (or 0 <= idx)
/// lt_len_tmp  = idx < len_tmp     (or len_tmp > idx)
/// cond_local  = ge_zero_tmp && lt_len_tmp
/// ```
/// ensuring neither `arr` nor `idx` is reassigned after the check.
fn match_header_bounds_check(
    func: &MirFunction,
    header_block: BlockId,
    cond_op: &Operand,
) -> Option<(LocalId, LocalId)> {
    let cond_local = match cond_op {
        Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)) => *l,
        _ => return None,
    };

    let stmts = &func.blocks.get(header_block.0)?.statements;

    // Record the latest definition index and rvalue for each local in `header_block`.
    let mut last_def: HashMap<LocalId, (usize, &Rvalue)> = HashMap::new();
    for (idx, stmt) in stmts.iter().enumerate() {
        if let Statement::Assign(Place::Local(dst), rval, _) = stmt {
            last_def.insert(*dst, (idx, rval));
        }
    }

    let (and_idx, and_rval) = *last_def.get(&cond_local)?;
    let (lhs, rhs) = match and_rval {
        Rvalue::BinaryOp(MirBinaryOp::And, l, r) => (l, r),
        _ => return None,
    };

    let lhs_local = operand_local(lhs)?;
    let rhs_local = operand_local(rhs)?;
    let (lhs_idx, lhs_rval) = *last_def.get(&lhs_local)?;
    let (rhs_idx, rhs_rval) = *last_def.get(&rhs_local)?;
    if lhs_idx >= and_idx || rhs_idx >= and_idx {
        return None;
    }

    let try_pair = |ge_rval: &Rvalue, lt_rval: &Rvalue, ge_pos: usize, lt_pos: usize| {
        let idx_from_ge = match_ge_zero(ge_rval)?;
        let (idx_from_lt, len_local) = match_lt_len(lt_rval)?;
        if idx_from_ge != idx_from_lt {
            return None;
        }
        let (len_pos, len_rval) = *last_def.get(&len_local)?;
        if len_pos >= lt_pos {
            return None;
        }
        let arr_local = match len_rval {
            Rvalue::UnaryOp(MirUnaryOp::Len, op) => operand_local(op)?,
            _ => return None,
        };
        let earliest = len_pos.min(ge_pos).min(lt_pos);
        // Ensure neither `arr_local` nor `idx_local` is mutated between `earliest` and block end.
        for stmt in &stmts[earliest..] {
            if let Statement::Assign(place, _, _) = stmt {
                let root = place.root_local();
                if root == arr_local || root == idx_from_ge {
                    return None;
                }
            }
        }
        Some((arr_local, idx_from_ge))
    };

    try_pair(lhs_rval, rhs_rval, lhs_idx, rhs_idx)
        .or_else(|| try_pair(rhs_rval, lhs_rval, rhs_idx, lhs_idx))
}

fn operand_local(op: &Operand) -> Option<LocalId> {
    match op {
        Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)) => Some(*l),
        _ => None,
    }
}

fn match_ge_zero(rval: &Rvalue) -> Option<LocalId> {
    match rval {
        Rvalue::BinaryOp(MirBinaryOp::GtEq, lhs, Operand::Constant(MirConstant::Int(0))) => {
            operand_local(lhs)
        }
        Rvalue::BinaryOp(MirBinaryOp::LtEq, Operand::Constant(MirConstant::Int(0)), rhs) => {
            operand_local(rhs)
        }
        _ => None,
    }
}

fn match_lt_len(rval: &Rvalue) -> Option<(LocalId, LocalId)> {
    match rval {
        Rvalue::BinaryOp(MirBinaryOp::Lt, lhs, rhs) => {
            Some((operand_local(lhs)?, operand_local(rhs)?))
        }
        Rvalue::BinaryOp(MirBinaryOp::Gt, lhs, rhs) => {
            Some((operand_local(rhs)?, operand_local(lhs)?))
        }
        _ => None,
    }
}

/// Flattens nested `Seq` nodes and eliminates empty `Seq`s.
fn make_seq(nodes: Vec<StructuredNode>) -> StructuredNode {
    let mut flat = Vec::new();
    for node in nodes {
        match node {
            StructuredNode::Seq(inner) => {
                for item in inner {
                    if !item.is_empty_seq() {
                        flat.push(item);
                    }
                }
            }
            other => flat.push(other),
        }
    }
    match flat.len() {
        0 => StructuredNode::Seq(Vec::new()),
        1 => flat.pop().unwrap(),
        _ => StructuredNode::Seq(flat),
    }
}

/// Returns `true` if `node` contains a `StructuredNode::Break(target)` matching `label`.
fn contains_break_to(node: &StructuredNode, label: BlockId) -> bool {
    match node {
        StructuredNode::Break(target) => *target == label,
        StructuredNode::Seq(items) => items.iter().any(|item| contains_break_to(item, label)),
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
            contains_break_to(then_branch, label)
                || else_branch
                    .as_deref()
                    .is_some_and(|eb| contains_break_to(eb, label))
        }
        StructuredNode::Switch { cases, default, .. } => {
            cases.iter().any(|(_, n)| contains_break_to(n, label))
                || contains_break_to(default, label)
        }
        StructuredNode::WhileLoop {
            label: inner_label,
            body,
            ..
        }
        | StructuredNode::Loop {
            label: inner_label,
            body,
        }
        | StructuredNode::LabeledBlock {
            label: inner_label,
            body,
        } => {
            if *inner_label == label {
                false
            } else {
                contains_break_to(body, label)
            }
        }
        StructuredNode::BasicBlock(_)
        | StructuredNode::Continue(_)
        | StructuredNode::Return(_, _)
        | StructuredNode::StateMachineFallback { .. } => false,
    }
}

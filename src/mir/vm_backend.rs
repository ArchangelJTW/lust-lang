use crate::bytecode::{Chunk, Function, Instruction, Register, StructLayout, Value};
use crate::bytecode::value::{native_fn, NativeCallResult};
use crate::data::layout::CanonicalTypeTag;
use crate::data::materialize::{
    decode_json_into_contiguous, ContiguousDecodeOptions, ContiguousStruct, LayoutPerfCounters,
};
use crate::mir::ir::*;
use crate::number::{LustFloat, LustInt, NumericType};
use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;
use hashbrown::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MirVmBackendError {
    #[error("Binary operator '{0:?}' is not supported by MIR VM backend")]
    UnsupportedBinaryOp(MirBinaryOp),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum VmTypeLattice {
    Bottom,
    Known(MirType),
    Top,
}

impl VmTypeLattice {
    fn join(&mut self, incoming: VmTypeLattice) -> bool {
        let next = match (&*self, incoming) {
            (_, VmTypeLattice::Bottom) => return false,
            (VmTypeLattice::Top, _) => return false,
            (_, VmTypeLattice::Top) => VmTypeLattice::Top,
            (VmTypeLattice::Bottom, VmTypeLattice::Known(ty)) => VmTypeLattice::Known(ty),
            (VmTypeLattice::Known(a), VmTypeLattice::Known(b)) if a == &b => return false,
            (
                VmTypeLattice::Known(MirType::Array(a)),
                VmTypeLattice::Known(MirType::Array(b)),
            ) => {
                if **a == MirType::Unknown {
                    return false;
                }
                if *b == MirType::Unknown || *a != b {
                    VmTypeLattice::Known(MirType::Array(Box::new(MirType::Unknown)))
                } else {
                    return false;
                }
            }
            _ => VmTypeLattice::Top,
        };
        if *self != next {
            *self = next;
            true
        } else {
            false
        }
    }
}

pub struct MirToBytecodeCompiler<'a> {
    func: &'a MirFunction,
    local_types: Vec<MirType>,
    chunk: Chunk,
    scratch_start: usize,
    scratch_count: usize,
    max_scratch: usize,
    block_pcs: HashMap<BlockId, usize>,
    terminator_pcs: HashMap<usize, usize>,
    pending_jumps: Vec<(usize, BlockId)>,
    direct_callees: Option<&'a HashSet<String>>,
    direct_callee_indices: Option<&'a HashMap<String, usize>>,
    struct_layouts: Option<&'a HashMap<String, Rc<StructLayout>>>,
}

impl<'a> MirToBytecodeCompiler<'a> {
    pub fn compile(func: &'a MirFunction) -> Result<Function, MirVmBackendError> {
        Self::compile_inner(func, None, None, None)
    }

    pub fn compile_with_direct_callees(
        func: &'a MirFunction,
        direct_callees: &'a HashSet<String>,
    ) -> Result<Function, MirVmBackendError> {
        Self::compile_inner(func, Some(direct_callees), None, None)
    }

    pub fn compile_with_direct_callee_indices(
        func: &'a MirFunction,
        direct_callee_indices: &'a HashMap<String, usize>,
    ) -> Result<Function, MirVmBackendError> {
        Self::compile_inner(func, None, Some(direct_callee_indices), None)
    }

    pub fn compile_with_direct_callees_and_indices(
        func: &'a MirFunction,
        direct_callees: &'a HashSet<String>,
        direct_callee_indices: &'a HashMap<String, usize>,
    ) -> Result<Function, MirVmBackendError> {
        Self::compile_inner(func, Some(direct_callees), Some(direct_callee_indices), None)
    }

    pub fn compile_with_struct_layouts(
        func: &'a MirFunction,
        struct_layouts: &'a HashMap<String, Rc<StructLayout>>,
    ) -> Result<Function, MirVmBackendError> {
        Self::compile_inner(func, None, None, Some(struct_layouts))
    }

    pub fn compile_with_options(
        func: &'a MirFunction,
        direct_callees: Option<&'a HashSet<String>>,
        direct_callee_indices: Option<&'a HashMap<String, usize>>,
        struct_layouts: Option<&'a HashMap<String, Rc<StructLayout>>>,
    ) -> Result<Function, MirVmBackendError> {
        Self::compile_inner(func, direct_callees, direct_callee_indices, struct_layouts)
    }

    fn compile_inner(
        func: &'a MirFunction,
        direct_callees: Option<&'a HashSet<String>>,
        direct_callee_indices: Option<&'a HashMap<String, usize>>,
        struct_layouts: Option<&'a HashMap<String, Rc<StructLayout>>>,
    ) -> Result<Function, MirVmBackendError> {
        let param_count = func.params.len() as u8;
        let mut bytecode_fn = Function::new(func.signature.name.clone(), param_count, false);
        let local_types = Self::infer_local_types(func);

        let mut compiler = MirToBytecodeCompiler {
            func,
            local_types,
            chunk: Chunk::new(),
            scratch_start: func.locals.len(),
            scratch_count: 0,
            max_scratch: 0,
            block_pcs: HashMap::new(),
            terminator_pcs: HashMap::new(),
            pending_jumps: Vec::new(),
            direct_callees,
            direct_callee_indices,
            struct_layouts,
        };

        compiler.compile_blocks()?;
        compiler.patch_all_jumps();

        bytecode_fn.chunk = compiler.chunk;
        bytecode_fn.register_count = (func.locals.len() + compiler.max_scratch) as u8;
        bytecode_fn.terminator_pcs = compiler.terminator_pcs;
        bytecode_fn.block_entry_pcs = compiler
            .block_pcs
            .into_iter()
            .map(|(bid, pc)| (bid.0, pc))
            .collect();
        Ok(bytecode_fn)
    }

    fn next_scratch(&mut self) -> Register {
        let reg = (self.scratch_start + self.scratch_count) as Register;
        self.scratch_count += 1;
        if self.scratch_count > self.max_scratch {
            self.max_scratch = self.scratch_count;
        }
        reg
    }

    fn reset_scratch(&mut self) {
        self.scratch_count = 0;
    }

    fn eval_place_lattice(place: &Place, lattice: &[VmTypeLattice]) -> VmTypeLattice {
        match place {
            Place::Local(loc) => lattice
                .get(loc.0)
                .cloned()
                .unwrap_or(VmTypeLattice::Top),
            // Arrays are aliased mutable `Rc<RefCell<Vec<Value>>>` containers: fail closed.
            Place::Index(_, _) => VmTypeLattice::Top,
            Place::Field(base, field_idx) => match Self::eval_place_lattice(base, lattice) {
                VmTypeLattice::Bottom => VmTypeLattice::Bottom,
                // Tuples are immutable value containers.
                VmTypeLattice::Known(MirType::Tuple(elems)) => match elems.get(*field_idx) {
                    Some(ty) if *ty != MirType::Unknown => VmTypeLattice::Known(ty.clone()),
                    _ => VmTypeLattice::Top,
                },
                // Struct fields are mutable via `SetField` without runtime type enforcement: fail closed.
                _ => VmTypeLattice::Top,
            },
        }
    }

    fn eval_operand_lattice(op: &Operand, lattice: &[VmTypeLattice]) -> VmTypeLattice {
        match op {
            Operand::Constant(c) => VmTypeLattice::Known(match c {
                MirConstant::Int(_) => MirType::Int,
                MirConstant::Float(_) => MirType::Float,
                MirConstant::Bool(_) => MirType::Bool,
                MirConstant::String(_) => MirType::String,
                MirConstant::Unit => MirType::Unit,
            }),
            Operand::Copy(place) | Operand::Move(place) => Self::eval_place_lattice(place, lattice),
        }
    }

    fn eval_rvalue_lattice(
        func: &MirFunction,
        rvalue: &Rvalue,
        lattice: &[VmTypeLattice],
    ) -> VmTypeLattice {
        match rvalue {
            Rvalue::Use(op) => Self::eval_operand_lattice(op, lattice),
            Rvalue::Cast(op, target_ty) => match target_ty {
                MirType::Float => VmTypeLattice::Known(MirType::Float),
                // `Cast(_, Int)` compiles via dynamic `LoadGlobal("math") -> GetField("floor") -> Call`
                // which can return any value if `math` is shadowed: fail closed.
                MirType::Int => VmTypeLattice::Top,
                MirType::Option(_) => VmTypeLattice::Known(target_ty.clone()),
                MirType::Struct(_) => {
                    let is_opt = matches!(
                        op,
                        Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc))
                            if matches!(func.locals.get(loc.0).map(|l| &l.ty), Some(MirType::Option(_)))
                    );
                    if is_opt {
                        VmTypeLattice::Known(target_ty.clone())
                    } else {
                        Self::eval_operand_lattice(op, lattice)
                    }
                }
                _ => Self::eval_operand_lattice(op, lattice),
            },
            Rvalue::BinaryOp(binop, lhs, rhs) => {
                if matches!(binop, MirBinaryOp::Eq | MirBinaryOp::NotEq) {
                    let is_opt = |op: &Operand| {
                        matches!(
                            op,
                            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc))
                                if matches!(func.locals.get(loc.0).map(|l| &l.ty), Some(MirType::Option(_)))
                        )
                    };
                    if (is_opt(lhs) && matches!(rhs, Operand::Constant(MirConstant::Int(0))))
                        || (is_opt(rhs) && matches!(lhs, Operand::Constant(MirConstant::Int(0))))
                    {
                        return VmTypeLattice::Known(MirType::Bool);
                    }
                }
                match binop {
                    MirBinaryOp::Concat => VmTypeLattice::Known(MirType::String),
                    MirBinaryOp::And
                    | MirBinaryOp::Or
                    | MirBinaryOp::Eq
                    | MirBinaryOp::NotEq
                    | MirBinaryOp::Lt
                    | MirBinaryOp::LtEq
                    | MirBinaryOp::Gt
                    | MirBinaryOp::GtEq => VmTypeLattice::Known(MirType::Bool),
                    MirBinaryOp::BitAnd
                    | MirBinaryOp::BitOr
                    | MirBinaryOp::BitXor
                    | MirBinaryOp::Shl
                    | MirBinaryOp::Shr => VmTypeLattice::Known(MirType::Int),
                    MirBinaryOp::Add
                    | MirBinaryOp::Sub
                    | MirBinaryOp::Mul
                    | MirBinaryOp::Div
                    | MirBinaryOp::Mod
                    | MirBinaryOp::FloorDiv
                    | MirBinaryOp::Pow => {
                        let lt = Self::eval_operand_lattice(lhs, lattice);
                        let rt = Self::eval_operand_lattice(rhs, lattice);
                        match (lt, rt) {
                            (VmTypeLattice::Known(MirType::Int), VmTypeLattice::Known(MirType::Int)) => {
                                VmTypeLattice::Known(MirType::Int)
                            }
                            (
                                VmTypeLattice::Known(MirType::Float),
                                VmTypeLattice::Known(MirType::Float | MirType::Int),
                            )
                            | (
                                VmTypeLattice::Known(MirType::Int),
                                VmTypeLattice::Known(MirType::Float),
                            ) => VmTypeLattice::Known(MirType::Float),
                            (
                                VmTypeLattice::Known(MirType::String),
                                VmTypeLattice::Known(MirType::String),
                            ) if *binop == MirBinaryOp::Add => VmTypeLattice::Known(MirType::String),
                            (VmTypeLattice::Bottom, _) | (_, VmTypeLattice::Bottom) => {
                                VmTypeLattice::Bottom
                            }
                            _ => VmTypeLattice::Top,
                        }
                    }
                }
            }
            Rvalue::UnaryOp(unop, op) => match unop {
                MirUnaryOp::Not => VmTypeLattice::Known(MirType::Bool),
                MirUnaryOp::BitNot | MirUnaryOp::Len => VmTypeLattice::Known(MirType::Int),
                MirUnaryOp::Neg => match Self::eval_operand_lattice(op, lattice) {
                    VmTypeLattice::Known(MirType::Int) => VmTypeLattice::Known(MirType::Int),
                    VmTypeLattice::Known(MirType::Float) => VmTypeLattice::Known(MirType::Float),
                    VmTypeLattice::Bottom => VmTypeLattice::Bottom,
                    _ => VmTypeLattice::Top,
                },
            },
            Rvalue::Aggregate(agg_ty, elems) => match agg_ty {
                MirType::Array(_) => {
                    VmTypeLattice::Known(MirType::Array(Box::new(MirType::Unknown)))
                }
                MirType::Tuple(_) => {
                    let mut elem_tys = Vec::with_capacity(elems.len());
                    for elem in elems {
                        match Self::eval_operand_lattice(elem, lattice) {
                            VmTypeLattice::Bottom => return VmTypeLattice::Bottom,
                            VmTypeLattice::Known(ty) => elem_tys.push(ty),
                            VmTypeLattice::Top => elem_tys.push(MirType::Unknown),
                        }
                    }
                    VmTypeLattice::Known(MirType::Tuple(elem_tys))
                }
                _ => VmTypeLattice::Known(agg_ty.clone()),
            },
        }
    }

    /// Computes sound, fail-closed refined runtime types for all locals in `func`.
    ///
    /// 1. Computes CFG reachability from entry block `0` (index `0` in `func.blocks`, resolved via
    ///    `BlockId -> idx` so non-dense or reordered `BlockId`s work cleanly). Unreachable blocks
    ///    never contribute type facts or uninit widening.
    /// 2. Computes forward must-init dataflow over reachable blocks (`func.params` initialized at
    ///    entry; intersection across reachable predecessors) and widens any local that may be read
    ///    before initialization (`Value::Nil`) to `Top` (`MirType::Unknown`).
    /// 3. Initializes all `func.params` to `Top` (fail-closed, because VM calls do not enforce
    ///    `MirFunction` parameter types at runtime) and runs forward fixed-point type refinement
    ///    over reachable blocks only.
    fn infer_local_types(func: &MirFunction) -> Vec<MirType> {
        let num_locals = func.locals.len();
        let num_blocks = func.blocks.len();
        if num_locals == 0 {
            return Vec::new();
        }
        if num_blocks == 0 {
            return vec![MirType::Unknown; num_locals];
        }

        // 1. Map BlockId -> index in func.blocks and compute CFG successors.
        let mut block_id_to_idx: HashMap<BlockId, usize> = HashMap::with_capacity(num_blocks);
        for (idx, block) in func.blocks.iter().enumerate() {
            block_id_to_idx.insert(block.id, idx);
        }

        let mut succs: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
        for (idx, block) in func.blocks.iter().enumerate() {
            let out = &mut succs[idx];
            match &block.terminator {
                Some(Terminator::Goto(target, _)) => {
                    if let Some(&t_idx) = block_id_to_idx.get(target) {
                        out.push(t_idx);
                    }
                }
                Some(Terminator::Cond {
                    then_block,
                    else_block,
                    ..
                }) => {
                    if let Some(&t_idx) = block_id_to_idx.get(then_block) {
                        out.push(t_idx);
                    }
                    if let Some(&e_idx) = block_id_to_idx.get(else_block) {
                        out.push(e_idx);
                    }
                }
                Some(Terminator::Switch { cases, default, .. }) => {
                    for (_, target_bb) in cases {
                        if let Some(&c_idx) = block_id_to_idx.get(target_bb) {
                            out.push(c_idx);
                        }
                    }
                    if let Some(&d_idx) = block_id_to_idx.get(default) {
                        out.push(d_idx);
                    }
                }
                Some(Terminator::Call { success, error, .. }) => {
                    if let Some(&s_idx) = block_id_to_idx.get(success) {
                        out.push(s_idx);
                    }
                    if let Some(err_bb) = error
                        && let Some(&e_idx) = block_id_to_idx.get(err_bb)
                    {
                        out.push(e_idx);
                    }
                }
                Some(Terminator::Return(..) | Terminator::Unreachable(..)) => {}
                Some(Terminator::Raise(..)) | None => {
                    if idx + 1 < num_blocks {
                        out.push(idx + 1);
                    }
                }
            }
        }

        // Reachability starting from entry block index 0.
        let entry_idx = 0usize;
        let mut reachable = vec![false; num_blocks];
        let mut worklist = Vec::with_capacity(num_blocks);
        reachable[entry_idx] = true;
        worklist.push(entry_idx);
        while let Some(b) = worklist.pop() {
            for &s in &succs[b] {
                if !reachable[s] {
                    reachable[s] = true;
                    worklist.push(s);
                }
            }
        }

        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
        for b in 0..num_blocks {
            if !reachable[b] {
                continue;
            }
            for &s in &succs[b] {
                if reachable[s] {
                    preds[s].push(b);
                }
            }
        }

        // 2. Forward must-init dataflow over reachable blocks.
        let mut entry_init = vec![false; num_locals];
        for param in &func.params {
            if param.0 < num_locals {
                entry_init[param.0] = true;
            }
        }

        let mut stmt_gen = vec![vec![false; num_locals]; num_blocks];
        let mut call_gen: Vec<Option<(usize, Option<usize>)>> = vec![None; num_blocks];
        for b in 0..num_blocks {
            if !reachable[b] {
                continue;
            }
            let block = &func.blocks[b];
            for stmt in &block.statements {
                if let Statement::Assign(Place::Local(dst), _, _) = stmt
                    && dst.0 < num_locals
                {
                    stmt_gen[b][dst.0] = true;
                }
            }
            if let Some(Terminator::Call {
                destination: Place::Local(dst),
                success,
                ..
            }) = &block.terminator
                && dst.0 < num_locals
            {
                call_gen[b] = Some((dst.0, block_id_to_idx.get(success).copied()));
            }
        }

        let mut must_init_in = vec![vec![true; num_locals]; num_blocks];
        let mut must_init_after_stmts = vec![vec![true; num_locals]; num_blocks];
        let mut must_init_out = vec![vec![true; num_locals]; num_blocks];
        must_init_in[entry_idx] = entry_init;
        for l in 0..num_locals {
            must_init_after_stmts[entry_idx][l] =
                must_init_in[entry_idx][l] || stmt_gen[entry_idx][l];
            must_init_out[entry_idx][l] = must_init_after_stmts[entry_idx][l];
        }
        if let Some((dst, _)) = call_gen[entry_idx] {
            must_init_out[entry_idx][dst] = true;
        }

        loop {
            let mut changed = false;
            for b in 0..num_blocks {
                if !reachable[b] || b == entry_idx {
                    continue;
                }
                let mut new_in = vec![true; num_locals];
                for &p in &preds[b] {
                    // On Call unwind edges (`error != success`), Call's `destination` is not initialized.
                    let p_out = match call_gen[p] {
                        Some((_, Some(succ_idx))) if succ_idx != b => &must_init_after_stmts[p],
                        _ => &must_init_out[p],
                    };
                    for l in 0..num_locals {
                        new_in[l] &= p_out[l];
                    }
                }
                let mut new_after_stmts = new_in.clone();
                for l in 0..num_locals {
                    new_after_stmts[l] |= stmt_gen[b][l];
                }
                let mut new_out = new_after_stmts.clone();
                if let Some((dst, _)) = call_gen[b] {
                    new_out[dst] = true;
                }
                if new_in != must_init_in[b]
                    || new_after_stmts != must_init_after_stmts[b]
                    || new_out != must_init_out[b]
                {
                    must_init_in[b] = new_in;
                    must_init_after_stmts[b] = new_after_stmts;
                    must_init_out[b] = new_out;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // Walk reachable blocks to detect any read of a potentially uninitialized local.
        let mut may_be_uninit = vec![false; num_locals];
        let mut uses = HashSet::new();
        for b in 0..num_blocks {
            if !reachable[b] {
                continue;
            }
            let block = &func.blocks[b];
            let mut cur_init = must_init_in[b].clone();
            for stmt in &block.statements {
                uses.clear();
                stmt.collect_uses(&mut uses);
                for &u in &uses {
                    if u < num_locals && !cur_init[u] {
                        may_be_uninit[u] = true;
                    }
                }
                if let Statement::Assign(Place::Local(dst), _, _) = stmt
                    && dst.0 < num_locals
                {
                    cur_init[dst.0] = true;
                }
            }
            if let Some(term) = &block.terminator {
                uses.clear();
                term.collect_uses(&mut uses);
                for &u in &uses {
                    if u < num_locals && !cur_init[u] {
                        may_be_uninit[u] = true;
                    }
                }
                if let Terminator::Call {
                    destination: Place::Local(dst),
                    ..
                } = term
                    && dst.0 < num_locals
                {
                    cur_init[dst.0] = true;
                }
            }
        }

        // 3. Forward fixed-point type lattice over reachable blocks only.
        //    - All parameters start at `Top` (unchecked at runtime).
        //    - All potentially uninitialized locals start at `Top` so transitive uses also widen.
        let mut lattice = vec![VmTypeLattice::Bottom; num_locals];
        for param in &func.params {
            if param.0 < num_locals {
                lattice[param.0] = VmTypeLattice::Top;
            }
        }
        for l in 0..num_locals {
            if may_be_uninit[l] {
                lattice[l] = VmTypeLattice::Top;
            }
        }

        let max_iters = num_locals.saturating_mul(4).saturating_add(32);
        let mut converged = false;
        for _ in 0..max_iters {
            let mut changed = false;
            for (b_idx, block) in func.blocks.iter().enumerate() {
                if !reachable[b_idx] {
                    continue;
                }
                for stmt in &block.statements {
                    if let Statement::Assign(Place::Local(dst), rvalue, _) = stmt
                        && dst.0 < num_locals
                    {
                        let rhs_lat = Self::eval_rvalue_lattice(func, rvalue, &lattice);
                        if lattice[dst.0].join(rhs_lat) {
                            changed = true;
                        }
                    }
                }
                if let Some(Terminator::Call {
                    callee,
                    destination: Place::Local(dst),
                    ..
                }) = &block.terminator
                    && dst.0 < num_locals
                {
                    let ret_lat = match callee.as_str() {
                        "array.len" | "table.len" | "map.len" | "string.len" => {
                            VmTypeLattice::Known(MirType::Int)
                        }
                        "tostring" => VmTypeLattice::Known(MirType::String),
                        _ => VmTypeLattice::Top,
                    };
                    if lattice[dst.0].join(ret_lat) {
                        changed = true;
                    }
                }
            }
            if !changed {
                converged = true;
                break;
            }
        }

        if !converged {
            return vec![MirType::Unknown; num_locals];
        }

        lattice
            .into_iter()
            .map(|lat| match lat {
                VmTypeLattice::Known(ty) => ty,
                VmTypeLattice::Bottom | VmTypeLattice::Top => MirType::Unknown,
            })
            .collect()
    }

    fn lookup_struct_layout(&self, name: &str) -> Option<&'a Rc<StructLayout>> {
        if name.is_empty() {
            return None;
        }
        self.struct_layouts.and_then(|layouts| layouts.get(name))
    }

    fn place_type(&self, place: &Place) -> MirType {
        match place {
            Place::Local(loc) => self
                .local_types
                .get(loc.0)
                .cloned()
                .unwrap_or(MirType::Unknown),
            // Arrays are aliased mutable containers: fail closed.
            Place::Index(_, _) => MirType::Unknown,
            Place::Field(base, field_idx) => match self.place_type(base) {
                // Tuples are immutable.
                MirType::Tuple(elems) => elems.get(*field_idx).cloned().unwrap_or(MirType::Unknown),
                // Struct fields are mutable without runtime type enforcement: fail closed.
                _ => MirType::Unknown,
            },
        }
    }

    fn operand_type(&self, op: &Operand) -> MirType {
        match op {
            Operand::Constant(c) => match c {
                MirConstant::Int(_) => MirType::Int,
                MirConstant::Float(_) => MirType::Float,
                MirConstant::Bool(_) => MirType::Bool,
                MirConstant::String(_) => MirType::String,
                MirConstant::Unit => MirType::Unit,
            },
            Operand::Copy(place) | Operand::Move(place) => self.place_type(place),
        }
    }

    fn lookup_struct_field_name(&self, base: &Place, field_idx: usize) -> Option<Rc<String>> {
        let struct_name = match self.place_type(base) {
            MirType::Struct(name) => Some(name),
            _ => match base {
                Place::Local(loc) => match self.func.locals.get(loc.0).map(|l| &l.ty) {
                    Some(MirType::Struct(name)) => Some(name.clone()),
                    _ => None,
                },
                _ => None,
            },
        }?;
        let layout = self.lookup_struct_layout(&struct_name)?;
        layout.field_names().get(field_idx).cloned()
    }

    fn emit_get_field(&mut self, dst: Register, base: &Place, field_idx: usize, line: usize) {
        let base_reg = self.place_to_reg(base);
        if let Some(fname) = self.lookup_struct_field_name(base, field_idx) {
            let fname_val = Value::String(fname);
            let exists = self.chunk.constants.iter().any(|v| v == &fname_val);
            if exists || self.chunk.constants.len() < u16::MAX as usize {
                let field_const_idx = self.chunk.add_constant(fname_val);
                self.chunk
                    .emit(Instruction::GetField(dst, base_reg, field_const_idx), line);
                return;
            }
        }
        let idx_reg = self.next_scratch();
        self.load_constant_into_reg(&MirConstant::Int(field_idx as i64), idx_reg, line);
        self.chunk
            .emit(Instruction::GetIndex(dst, base_reg, idx_reg), line);
    }

    fn emit_set_field(&mut self, base: &Place, field_idx: usize, val_reg: Register, line: usize) {
        let base_reg = self.place_to_reg(base);
        if let Some(fname) = self.lookup_struct_field_name(base, field_idx) {
            let fname_val = Value::String(fname);
            let exists = self.chunk.constants.iter().any(|v| v == &fname_val);
            if exists || self.chunk.constants.len() < u16::MAX as usize {
                let field_const_idx = self.chunk.add_constant(fname_val);
                self.chunk
                    .emit(Instruction::SetField(base_reg, field_const_idx, val_reg), line);
                return;
            }
        }
        let idx_reg = self.next_scratch();
        self.load_constant_into_reg(&MirConstant::Int(field_idx as i64), idx_reg, line);
        self.chunk
            .emit(Instruction::SetIndex(base_reg, idx_reg, val_reg), line);
    }

    fn compile_blocks(&mut self) -> Result<(), MirVmBackendError> {
        for (idx, block) in self.func.blocks.iter().enumerate() {
            let pc = self.chunk.instructions.len();
            self.block_pcs.insert(block.id, pc);

            for stmt in &block.statements {
                self.reset_scratch();
                self.compile_statement(stmt)?;
            }

            self.reset_scratch();
            let term_pc = self.chunk.instructions.len();
            self.terminator_pcs.insert(block.id.0, term_pc);
            let next_block_id = self.func.blocks.get(idx + 1).map(|b| b.id);
            if let Some(term) = &block.terminator {
                self.compile_terminator(term, block.id, next_block_id);
            }
        }
        Ok(())
    }

    fn operand_to_reg(&mut self, op: &Operand, line: usize) -> Register {
        match op {
            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => loc.0 as Register,
            Operand::Copy(Place::Index(base, idx)) | Operand::Move(Place::Index(base, idx)) => {
                let reg = self.next_scratch();
                let base_reg = self.place_to_reg(base);
                let idx_reg = idx.0 as Register;
                self.chunk.emit(Instruction::GetIndex(reg, base_reg, idx_reg), line);
                reg
            }
            Operand::Copy(Place::Field(base, f)) | Operand::Move(Place::Field(base, f)) => {
                let reg = self.next_scratch();
                self.emit_get_field(reg, base, *f, line);
                reg
            }
            Operand::Constant(c) => {
                let reg = self.next_scratch();
                self.load_constant_into_reg(c, reg, line);
                reg
            }
        }
    }

    fn load_constant_into_reg(&mut self, c: &MirConstant, reg: Register, line: usize) {
        match c {
            MirConstant::Int(i) => {
                let const_idx = self.chunk.add_constant(Value::Int(*i as LustInt));
                self.chunk.emit(Instruction::LoadConst(reg, const_idx), line);
            }
            MirConstant::Float(f) => {
                let const_idx = self.chunk.add_constant(Value::Float(*f as LustFloat));
                self.chunk.emit(Instruction::LoadConst(reg, const_idx), line);
            }
            MirConstant::Bool(b) => {
                self.chunk.emit(Instruction::LoadBool(reg, *b), line);
            }
            MirConstant::String(s) => {
                let const_idx = self.chunk.add_constant(Value::string(s.clone()));
                self.chunk.emit(Instruction::LoadConst(reg, const_idx), line);
            }
            MirConstant::Unit => {
                self.chunk.emit(Instruction::LoadNil(reg), line);
            }
        }
    }

    fn place_to_reg(&self, place: &Place) -> Register {
        match place {
            Place::Local(id) => id.0 as Register,
            Place::Index(base, _) => self.place_to_reg(base),
            Place::Field(base, _) => self.place_to_reg(base),
        }
    }

    fn operand_is_option(&self, op: &Operand) -> bool {
        match op {
            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => {
                matches!(
                    self.func.locals.get(loc.0).map(|l| &l.ty),
                    Some(MirType::Option(_))
                )
            }
            _ => false,
        }
    }

    fn compile_rvalue(&mut self, rvalue: &Rvalue, dst_reg: Register, line: usize) -> Result<(), MirVmBackendError> {
        match rvalue {
            Rvalue::Use(op) => {
                match op {
                    Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)) => {
                        if dst_reg != src.0 as Register {
                            self.chunk.emit(Instruction::Move(dst_reg, src.0 as Register), line);
                        }
                    }
                    Operand::Copy(Place::Index(base, idx)) | Operand::Move(Place::Index(base, idx)) => {
                        let base_reg = self.place_to_reg(base);
                        let idx_reg = idx.0 as Register;
                        self.chunk.emit(Instruction::GetIndex(dst_reg, base_reg, idx_reg), line);
                    }
                    Operand::Copy(Place::Field(base, f)) | Operand::Move(Place::Field(base, f)) => {
                        self.emit_get_field(dst_reg, base, *f, line);
                    }
                    Operand::Constant(c) => {
                        self.load_constant_into_reg(c, dst_reg, line);
                    }
                }
            }
            Rvalue::Cast(op, target_ty) => {
                match target_ty {
                    MirType::Option(_) => {
                        let option_idx = self.chunk.add_constant(Value::string("Option".to_string()));
                        if matches!(op, Operand::Constant(MirConstant::Int(0))) {
                            let none_idx = self.chunk.add_constant(Value::string("None".to_string()));
                            self.chunk.emit(
                                Instruction::NewEnumUnit(dst_reg, option_idx, none_idx),
                                line,
                            );
                        } else if self.operand_is_option(op) {
                            let r_op = self.operand_to_reg(op, line);
                            if dst_reg != r_op {
                                self.chunk.emit(Instruction::Move(dst_reg, r_op), line);
                            }
                        } else {
                            let r_op = self.operand_to_reg(op, line);
                            let some_idx = self.chunk.add_constant(Value::string("Some".to_string()));
                            self.chunk.emit(
                                Instruction::NewEnumVariant(dst_reg, option_idx, some_idx, r_op, 1),
                                line,
                            );
                        }
                    }
                    MirType::Struct(_) if self.operand_is_option(op) => {
                        let r_op = self.operand_to_reg(op, line);
                        self.chunk.emit(Instruction::GetEnumValue(dst_reg, r_op, 0), line);
                    }
                    MirType::Float => {
                        let r_op = self.operand_to_reg(op, line);
                        let zero_f_reg = self.next_scratch();
                        self.load_constant_into_reg(&MirConstant::Float(0.0), zero_f_reg, line);
                        self.chunk.emit(Instruction::Add(dst_reg, r_op, zero_f_reg), line);
                    }
                    MirType::Int => {
                        let r_op = self.operand_to_reg(op, line);
                        let obj_idx = self.chunk.add_constant(Value::string("math".to_string()));
                        let field_idx = self.chunk.add_constant(Value::string("floor".to_string()));
                        let func_reg = self.next_scratch();
                        self.chunk.emit(Instruction::LoadGlobal(func_reg, obj_idx), line);
                        self.chunk.emit(Instruction::GetField(func_reg, func_reg, field_idx), line);
                        let arg_reg = self.next_scratch();
                        self.chunk.emit(Instruction::Move(arg_reg, r_op), line);
                        self.chunk.emit(Instruction::Call(func_reg, arg_reg, 1, dst_reg), line);
                    }
                    _ => {
                        let r_op = self.operand_to_reg(op, line);
                        if dst_reg != r_op {
                            self.chunk.emit(Instruction::Move(dst_reg, r_op), line);
                        }
                    }
                }
            }
            Rvalue::BinaryOp(binop, lhs, rhs) => {
                if matches!(binop, MirBinaryOp::Eq | MirBinaryOp::NotEq) {
                    let opt_operand = if self.operand_is_option(lhs)
                        && matches!(rhs, Operand::Constant(MirConstant::Int(0)))
                    {
                        Some(lhs)
                    } else if self.operand_is_option(rhs)
                        && matches!(lhs, Operand::Constant(MirConstant::Int(0)))
                    {
                        Some(rhs)
                    } else {
                        None
                    };
                    if let Some(opt_op) = opt_operand {
                        let r_opt = self.operand_to_reg(opt_op, line);
                        let option_idx = self.chunk.add_constant(Value::string("Option".to_string()));
                        let variant_name = if *binop == MirBinaryOp::NotEq { "Some" } else { "None" };
                        let variant_idx = self.chunk.add_constant(Value::string(variant_name.to_string()));
                        self.chunk.emit(
                            Instruction::IsEnumVariant(dst_reg, r_opt, option_idx, variant_idx),
                            line,
                        );
                        return Ok(());
                    }
                }
                let lhs_ty = self.operand_type(lhs);
                let rhs_ty = self.operand_type(rhs);
                let is_string = lhs_ty == MirType::String || rhs_ty == MirType::String;
                let r_lhs = self.operand_to_reg(lhs, line);
                let r_rhs = self.operand_to_reg(rhs, line);
                if is_string && *binop == MirBinaryOp::Add {
                    self.chunk.emit(Instruction::Concat(dst_reg, r_lhs, r_rhs), line);
                } else {
                    self.compile_binop(*binop, dst_reg, r_lhs, r_rhs, &lhs_ty, &rhs_ty, line)?;
                }
            }
            Rvalue::UnaryOp(unop, op) => {
                let op_ty = self.operand_type(op);
                let r_op = self.operand_to_reg(op, line);
                match unop {
                    MirUnaryOp::Neg => {
                        let base = Instruction::Neg(dst_reg, r_op);
                        let instr = match op_ty {
                            MirType::Int => base.specialize_numeric(NumericType::Int),
                            MirType::Float => base.specialize_numeric(NumericType::Float),
                            _ => base,
                        };
                        self.chunk.emit(instr, line);
                    }
                    MirUnaryOp::Not => {
                        self.chunk.emit(Instruction::Not(dst_reg, r_op), line);
                    }
                    MirUnaryOp::BitNot => {
                        self.chunk.emit(Instruction::BitNot(dst_reg, r_op), line);
                    }
                    MirUnaryOp::Len => {
                        self.chunk.emit(Instruction::ArrayLen(dst_reg, r_op), line);
                    }
                }
            }
            Rvalue::Aggregate(agg_ty, elements) => {
                let count = elements.len() as u8;
                let scratch_base = if count == 0 {
                    0
                } else {
                    let s_base = (self.scratch_start + self.scratch_count) as Register;
                    self.scratch_count += elements.len();
                    if self.scratch_count > self.max_scratch {
                        self.max_scratch = self.scratch_count;
                    }
                    for (i, elem) in elements.iter().enumerate() {
                        let target_reg = s_base + i as u8;
                        match elem {
                            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => {
                                self.chunk.emit(Instruction::Move(target_reg, loc.0 as Register), line);
                            }
                            Operand::Copy(Place::Index(base, idx)) | Operand::Move(Place::Index(base, idx)) => {
                                let base_reg = self.place_to_reg(base);
                                let idx_reg = idx.0 as Register;
                                self.chunk.emit(Instruction::GetIndex(target_reg, base_reg, idx_reg), line);
                            }
                            Operand::Copy(Place::Field(base, f)) | Operand::Move(Place::Field(base, f)) => {
                                self.emit_get_field(target_reg, base, *f, line);
                            }
                            Operand::Constant(c) => {
                                self.load_constant_into_reg(c, target_reg, line);
                            }
                        }
                    }
                    s_base
                };
                match agg_ty {
                    MirType::Tuple(_) => {
                        self.chunk.emit(Instruction::TupleNew(dst_reg, scratch_base, count), line);
                    }
                    MirType::Struct(struct_name) => {
                        if let Some(layout) = self.lookup_struct_layout(struct_name)
                            && layout.field_names().len() == elements.len()
                        {
                            let field_names = layout.field_names();
                            let name_val = Value::string(struct_name.clone());
                            let name_extra =
                                usize::from(!self.chunk.constants.iter().any(|v| v == &name_val));
                            if self
                                .chunk
                                .constants
                                .len()
                                .checked_add(name_extra)
                                .and_then(|n| n.checked_add(field_names.len()))
                                .is_some_and(|end| end <= u16::MAX as usize)
                            {
                                let name_idx = self.chunk.add_constant(name_val);
                                let first_field_name_idx = if count == 0 {
                                    0u16
                                } else {
                                    let start = self.chunk.constants.len();
                                    for fname in field_names {
                                        self.chunk
                                            .constants
                                            .push(Value::String(Rc::clone(fname)));
                                    }
                                    start as u16
                                };
                                self.chunk.emit(
                                    Instruction::NewStruct(
                                        dst_reg,
                                        name_idx,
                                        first_field_name_idx,
                                        scratch_base,
                                        count,
                                    ),
                                    line,
                                );
                                return Ok(());
                            }
                        }
                        self.chunk.emit(Instruction::NewArray(dst_reg, scratch_base, count), line);
                    }
                    _ => {
                        self.chunk.emit(Instruction::NewArray(dst_reg, scratch_base, count), line);
                    }
                }
            }
        }
        Ok(())
    }

    fn compile_statement(&mut self, stmt: &Statement) -> Result<(), MirVmBackendError> {
        match stmt {
            Statement::Assign(Place::Local(dst), rvalue, span) => {
                let dst_reg = dst.0 as Register;
                let line = span.start_line;
                self.compile_rvalue(rvalue, dst_reg, line)?;
            }
            Statement::Assign(Place::Index(base, idx), rvalue, span) => {
                let line = span.start_line;
                let val_reg = match rvalue {
                    Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))) => {
                        src.0 as Register
                    }
                    _ => {
                        let temp_reg = self.next_scratch();
                        self.compile_rvalue(rvalue, temp_reg, line)?;
                        temp_reg
                    }
                };
                let base_reg = self.place_to_reg(base);
                let idx_reg = idx.0 as Register;
                self.chunk.emit(Instruction::SetIndex(base_reg, idx_reg, val_reg), line);
            }
            Statement::Assign(Place::Field(base, f), rvalue, span) => {
                let line = span.start_line;
                let val_reg = match rvalue {
                    Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))) => {
                        src.0 as Register
                    }
                    _ => {
                        let temp_reg = self.next_scratch();
                        self.compile_rvalue(rvalue, temp_reg, line)?;
                        temp_reg
                    }
                };
                self.emit_set_field(base, *f, val_reg, line);
            }
            Statement::StorageLive(_, _) | Statement::StorageDead(_, _) | Statement::ResourceCheck(_, _, _) => {}
        }
        Ok(())
    }

    fn compile_binop(
        &mut self,
        op: MirBinaryOp,
        dst: Register,
        lhs: Register,
        rhs: Register,
        lhs_ty: &MirType,
        rhs_ty: &MirType,
        line: usize,
    ) -> Result<(), MirVmBackendError> {
        let base_instr = match op {
            MirBinaryOp::Add => Instruction::Add(dst, lhs, rhs),
            MirBinaryOp::Sub => Instruction::Sub(dst, lhs, rhs),
            MirBinaryOp::Mul => Instruction::Mul(dst, lhs, rhs),
            MirBinaryOp::Div => Instruction::Div(dst, lhs, rhs),
            MirBinaryOp::Mod => Instruction::Mod(dst, lhs, rhs),
            MirBinaryOp::FloorDiv => Instruction::FloorDiv(dst, lhs, rhs),
            MirBinaryOp::Pow => Instruction::Pow(dst, lhs, rhs),
            MirBinaryOp::BitAnd => Instruction::BitAnd(dst, lhs, rhs),
            MirBinaryOp::BitOr => Instruction::BitOr(dst, lhs, rhs),
            MirBinaryOp::BitXor => Instruction::BitXor(dst, lhs, rhs),
            MirBinaryOp::Shl => Instruction::Shl(dst, lhs, rhs),
            MirBinaryOp::Shr => Instruction::Shr(dst, lhs, rhs),
            MirBinaryOp::Eq => Instruction::Eq(dst, lhs, rhs),
            MirBinaryOp::NotEq => Instruction::Ne(dst, lhs, rhs),
            MirBinaryOp::Lt => Instruction::Lt(dst, lhs, rhs),
            MirBinaryOp::LtEq => Instruction::Le(dst, lhs, rhs),
            MirBinaryOp::Gt => Instruction::Gt(dst, lhs, rhs),
            MirBinaryOp::GtEq => Instruction::Ge(dst, lhs, rhs),
            MirBinaryOp::And => Instruction::And(dst, lhs, rhs),
            MirBinaryOp::Or => Instruction::Or(dst, lhs, rhs),
            MirBinaryOp::Concat => Instruction::Concat(dst, lhs, rhs),
        };
        let instr = match (lhs_ty, rhs_ty) {
            (MirType::Int, MirType::Int) => base_instr.specialize_numeric(NumericType::Int),
            (MirType::Float, MirType::Float) => base_instr.specialize_numeric(NumericType::Float),
            _ => base_instr,
        };
        self.chunk.emit(instr, line);
        Ok(())
    }

    fn compile_terminator(
        &mut self,
        term: &Terminator,
        current_block_id: BlockId,
        next_block_id: Option<BlockId>,
    ) {
        let is_fallthrough = |target: BlockId| {
            next_block_id == Some(target) && target != current_block_id
        };
        match term {
            Terminator::Goto(target, span) => {
                if !is_fallthrough(*target) {
                    let jump_idx = self.chunk.emit(Instruction::Jump(0), span.start_line);
                    self.pending_jumps.push((jump_idx, *target));
                }
            }
            Terminator::Cond { cond, then_block, else_block, span } => {
                let line = span.start_line;
                if then_block == else_block {
                    let _ = self.operand_to_reg(cond, line);
                    if !is_fallthrough(*then_block) {
                        let jump_idx = self.chunk.emit(Instruction::Jump(0), line);
                        self.pending_jumps.push((jump_idx, *then_block));
                    }
                } else if is_fallthrough(*then_block) {
                    let r_cond = self.operand_to_reg(cond, line);
                    let jump_if_not = self.chunk.emit(Instruction::JumpIfNot(r_cond, 0), line);
                    self.pending_jumps.push((jump_if_not, *else_block));
                } else if is_fallthrough(*else_block) {
                    let r_cond = self.operand_to_reg(cond, line);
                    let jump_if = self.chunk.emit(Instruction::JumpIf(r_cond, 0), line);
                    self.pending_jumps.push((jump_if, *then_block));
                } else {
                    let r_cond = self.operand_to_reg(cond, line);
                    let jump_if_not = self.chunk.emit(Instruction::JumpIfNot(r_cond, 0), line);
                    self.pending_jumps.push((jump_if_not, *else_block));

                    let jump_then = self.chunk.emit(Instruction::Jump(0), line);
                    self.pending_jumps.push((jump_then, *then_block));
                }
            }
            Terminator::Switch { discr, cases, default, span } => {
                let line = span.start_line;
                let discr_reg = self.operand_to_reg(discr, line);
                let case_scratch = self.scratch_count;
                for (val, target_bb) in cases {
                    self.scratch_count = case_scratch;
                    let val_reg = self.next_scratch();
                    self.load_constant_into_reg(&MirConstant::Int(*val), val_reg, line);
                    let cmp_reg = self.next_scratch();
                    self.chunk.emit(Instruction::EqInt(cmp_reg, discr_reg, val_reg), line);
                    let jump_target = self.chunk.emit(Instruction::JumpIf(cmp_reg, 0), line);
                    self.pending_jumps.push((jump_target, *target_bb));
                }
                if !is_fallthrough(*default) {
                    let default_jump = self.chunk.emit(Instruction::Jump(0), line);
                    self.pending_jumps.push((default_jump, *default));
                }
            }
            Terminator::Return(opt_op, span) => {
                let r_ret = if let Some(op) = opt_op {
                    self.operand_to_reg(op, span.start_line)
                } else {
                    0
                };
                self.chunk.emit(Instruction::Return(r_ret), span.start_line);
            }
            Terminator::Call { callee, args, destination, success, error: _, span } => {
                let line = span.start_line;
                let arg_count = args.len() as u8;
                let first_arg_reg = if arg_count > 0 {
                    let scratch_base = self.next_scratch();
                    for _ in 1..args.len() {
                        self.next_scratch();
                    }
                    for (i, arg) in args.iter().enumerate() {
                        let target_reg = scratch_base + i as u8;
                        match arg {
                            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => {
                                self.chunk.emit(Instruction::Move(target_reg, loc.0 as Register), line);
                            }
                            Operand::Copy(Place::Index(base, idx)) | Operand::Move(Place::Index(base, idx)) => {
                                let base_reg = self.place_to_reg(base);
                                let idx_reg = idx.0 as Register;
                                self.chunk.emit(Instruction::GetIndex(target_reg, base_reg, idx_reg), line);
                            }
                            Operand::Copy(Place::Field(base, f)) | Operand::Move(Place::Field(base, f)) => {
                                self.emit_get_field(target_reg, base, *f, line);
                            }
                            Operand::Constant(c) => {
                                self.load_constant_into_reg(c, target_reg, line);
                            }
                        }
                    }
                    scratch_base
                } else {
                    0
                };

                let func_reg = self.next_scratch();
                if let Some(&callee_idx) = self
                    .direct_callee_indices
                    .and_then(|map| map.get(callee.as_str()))
                {
                    let fn_const_idx = self.chunk.add_constant(Value::Function(callee_idx));
                    self.chunk.emit(Instruction::LoadConst(func_reg, fn_const_idx), line);
                } else {
                    let is_direct = self
                        .direct_callees
                        .is_some_and(|set| set.contains(callee.as_str()));
                    if !is_direct
                        && let Some((obj_name, field_name)) = callee.split_once('.')
                    {
                        let obj_idx = self.chunk.add_constant(Value::string(obj_name.to_string()));
                        let field_idx = self.chunk.add_constant(Value::string(field_name.to_string()));
                        self.chunk.emit(Instruction::LoadGlobal(func_reg, obj_idx), line);
                        self.chunk.emit(Instruction::GetField(func_reg, func_reg, field_idx), line);
                    } else {
                        let name_idx = self.chunk.add_constant(Value::string(callee.clone()));
                        self.chunk.emit(Instruction::LoadGlobal(func_reg, name_idx), line);
                    }
                }

                let dest_reg = self.place_to_reg(destination);
                self.chunk.emit(Instruction::Call(func_reg, first_arg_reg, arg_count, dest_reg), line);

                if !is_fallthrough(*success) {
                    let jump_idx = self.chunk.emit(Instruction::Jump(0), line);
                    self.pending_jumps.push((jump_idx, *success));
                }
            }
            Terminator::Unreachable(span) => {
                self.chunk.emit(Instruction::Return(0), span.start_line);
            }
            _ => {}
        }
    }

    fn patch_all_jumps(&mut self) {
        for (jump_idx, target) in &self.pending_jumps {
            if let Some(target_pc) = self.block_pcs.get(target) {
                self.chunk.patch_jump(*jump_idx, *target_pc);
            } else {
                panic!("Target basic block {:?} not found during jump patching", target);
            }
        }
    }

    pub fn compile_module(module: &'a MirModule) -> Result<Vec<Function>, MirVmBackendError> {
        let mut out = Vec::with_capacity(module.functions.len());
        for func in &module.functions {
            out.push(Self::compile(func)?);
        }
        Ok(out)
    }
}

/// Structural bytecode audit for Gate-SCHEMA-09 zero name/hash lookup verification.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkFieldAccessAudit {
    pub get_field_count: usize,
    pub set_field_count: usize,
    pub get_index_count: usize,
    pub set_index_count: usize,
}

impl ChunkFieldAccessAudit {
    pub fn has_zero_name_lookup(&self) -> bool {
        self.get_field_count == 0 && self.set_field_count == 0
    }
}

/// Audits a compiled VM bytecode chunk to verify zero `GetField`/`SetField` name lookups.
pub fn audit_chunk_field_access(chunk: &Chunk) -> ChunkFieldAccessAudit {
    let mut audit = ChunkFieldAccessAudit::default();
    for instr in &chunk.instructions {
        match instr {
            Instruction::GetField(..) => audit.get_field_count += 1,
            Instruction::SetField(..) => audit.set_field_count += 1,
            Instruction::GetIndex(..) => audit.get_index_count += 1,
            Instruction::SetIndex(..) => audit.set_index_count += 1,
            _ => {}
        }
    }
    audit
}

fn contiguous_to_vm_slots(s: &ContiguousStruct) -> Result<Value, String> {
    let mut slots = Vec::with_capacity(s.layout.fields.len());
    for field in &s.layout.fields {
        let ord = field.ordinal;
        let off = field.byte_offset;
        if s.is_null(ord) || !s.is_initialized(ord) {
            slots.push(Value::Nil);
            continue;
        }
        let val = match field.tag {
            CanonicalTypeTag::Bool => {
                Value::Bool(s.read_bool_at(ord, off, None).map_err(|e| e.to_string())?.unwrap_or(false))
            }
            CanonicalTypeTag::Int64 | CanonicalTypeTag::DateTimeUtc => {
                Value::Int(s.read_i64_at(ord, off, None).map_err(|e| e.to_string())?.unwrap_or(0) as LustInt)
            }
            CanonicalTypeTag::UInt64 => {
                let u = s.read_u64_at(ord, off, None).map_err(|e| e.to_string())?.unwrap_or(0);
                let i = i64::try_from(u).map_err(|_| "UInt64 overflows VM Int".to_string())?;
                Value::Int(i as LustInt)
            }
            CanonicalTypeTag::Float64 => {
                Value::Float(s.read_f64_at(ord, off, None).map_err(|e| e.to_string())?.unwrap_or(0.0) as LustFloat)
            }
            CanonicalTypeTag::StringSlice => {
                let text = s.read_str_at(ord, off, None).map_err(|e| e.to_string())?.unwrap_or("");
                Value::string(text.to_string())
            }
            CanonicalTypeTag::BinarySlice => {
                let bytes = s.read_bytes_at(ord, off, None).map_err(|e| e.to_string())?.unwrap_or(&[]);
                Value::string(String::from_utf8_lossy(bytes).into_owned())
            }
            CanonicalTypeTag::NestedStruct(_) => {
                if let Some(nested) = s.read_nested_at(ord, None).map_err(|e| e.to_string())? {
                    contiguous_to_vm_slots(nested)?
                } else {
                    Value::Nil
                }
            }
            CanonicalTypeTag::ObjectId => {
                let oid = s.read_object_id_at(ord, off, None).map_err(|e| e.to_string())?.unwrap_or([0u8; 12]);
                let mut hex = String::with_capacity(24);
                for b in oid {
                    use core::fmt::Write;
                    let _ = write!(&mut hex, "{:02x}", b);
                }
                Value::string(hex)
            }
            CanonicalTypeTag::Decimal128 => {
                let dec = s.read_decimal128_at(ord, off, None).map_err(|e| e.to_string())?.unwrap_or(0);
                Value::Int(dec as LustInt)
            }
        };
        slots.push(val);
    }
    Ok(Value::array(slots))
}

/// Registers `StructDecodeInto` native decoders for all layouts in `module` onto `vm`.
pub fn register_module_struct_decoders(vm: &mut crate::vm::VM, module: &MirModule) {
    let counters = Rc::new(RefCell::new(LayoutPerfCounters::default()));
    register_struct_decode_natives_with_counters(vm, &module.struct_layouts, counters);
}

/// Registers `StructDecodeInto` native decoders with shared instrumentation counters.
pub fn register_struct_decode_natives_with_counters(
    vm: &mut crate::vm::VM,
    layouts: &[MirStructLayout],
    counters: Rc<RefCell<LayoutPerfCounters>>,
) {
    let mut nested_map = std::collections::HashMap::new();
    for l in layouts {
        nested_map.insert(l.type_id, l.to_logical_layout());
    }
    let nested_map = Rc::new(nested_map);

    for layout in layouts {
        let callee = mir_struct_decode_callee(&layout.type_name);
        let logical = layout.to_logical_layout();
        let counters_ref = Rc::clone(&counters);
        let nested_ref = Rc::clone(&nested_map);
        let func = native_fn(move |args: &[Value]| {
            let input_str = args
                .first()
                .and_then(|v| v.as_string())
                .ok_or_else(|| "StructDecodeInto expects a JSON string argument".to_string())?;
            let opts = ContiguousDecodeOptions {
                cancel_check: None,
                nested_layouts: Some(nested_ref.as_ref()),
                require_nominal_tag: false,
            };
            let mut c = counters_ref.borrow_mut();
            let decoded = decode_json_into_contiguous(&logical, input_str.as_bytes(), &mut c, &opts)
                .map_err(|e| e.to_string())?;
            let slots = contiguous_to_vm_slots(&decoded)?;
            Ok(NativeCallResult::Return(slots))
        });
        vm.register_native(callee, Value::NativeFunction(func));
    }
}

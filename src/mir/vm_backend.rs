use crate::ast::{FieldOwnership, StructDef};
use crate::bytecode::{
    Chunk, FieldStorage, Function, Instruction, Register, StructLayout, StructObject, Value,
};
use crate::bytecode::value::{native_fn, NativeCallResult};
use crate::data::layout::CanonicalTypeTag;
use crate::data::materialize::{
    decode_json_into_contiguous, struct_def_to_layout, ContiguousDecodeOptions, ContiguousStruct,
    LayoutPerfCounters,
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
    #[error("Rvalue '{0}' is not supported by MIR VM backend")]
    UnsupportedRvalue(alloc::string::String),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum HoistConstKey {
    Int(i64),
    Float(u64),
}

impl HoistConstKey {
    fn from_mir_const(c: &MirConstant) -> Option<Self> {
        match c {
            MirConstant::Int(i) => Some(Self::Int(*i)),
            MirConstant::Float(f) => Some(Self::Float(f.to_bits())),
            _ => None,
        }
    }

    fn to_value(self) -> Value {
        match self {
            Self::Int(i) => Value::Int(i as LustInt),
            Self::Float(bits) => Value::Float(f64::from_bits(bits) as LustFloat),
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
    use_field_slots: bool,
    loop_headers: HashSet<BlockId>,
    in_loop_blocks: HashSet<BlockId>,
    hoisted_consts_map: HashMap<HoistConstKey, Register>,
    hoisted_consts_order: Vec<(HoistConstKey, Register)>,
    local_read_counts: Vec<usize>,
    current_block_start_pc: usize,
    current_block_in_loop: bool,
}

impl<'a> MirToBytecodeCompiler<'a> {
    pub fn compile(func: &'a MirFunction) -> Result<Function, MirVmBackendError> {
        Self::compile_inner(func, None, None, None, false)
    }

    pub fn compile_with_direct_callees(
        func: &'a MirFunction,
        direct_callees: &'a HashSet<String>,
    ) -> Result<Function, MirVmBackendError> {
        Self::compile_inner(func, Some(direct_callees), None, None, false)
    }

    pub fn compile_with_direct_callee_indices(
        func: &'a MirFunction,
        direct_callee_indices: &'a HashMap<String, usize>,
    ) -> Result<Function, MirVmBackendError> {
        Self::compile_inner(func, None, Some(direct_callee_indices), None, false)
    }

    pub fn compile_with_direct_callees_and_indices(
        func: &'a MirFunction,
        direct_callees: &'a HashSet<String>,
        direct_callee_indices: &'a HashMap<String, usize>,
    ) -> Result<Function, MirVmBackendError> {
        Self::compile_inner(func, Some(direct_callees), Some(direct_callee_indices), None, false)
    }

    pub fn compile_with_struct_layouts(
        func: &'a MirFunction,
        struct_layouts: &'a HashMap<String, Rc<StructLayout>>,
    ) -> Result<Function, MirVmBackendError> {
        Self::compile_inner(func, None, None, Some(struct_layouts), false)
    }

    pub fn compile_with_field_slots(
        func: &'a MirFunction,
        struct_layouts: &'a HashMap<String, Rc<StructLayout>>,
    ) -> Result<Function, MirVmBackendError> {
        Self::compile_inner(func, None, None, Some(struct_layouts), true)
    }

    pub fn compile_with_field_slots_and_callee_indices(
        func: &'a MirFunction,
        direct_callee_indices: &'a HashMap<String, usize>,
        struct_layouts: &'a HashMap<String, Rc<StructLayout>>,
    ) -> Result<Function, MirVmBackendError> {
        Self::compile_inner(
            func,
            None,
            Some(direct_callee_indices),
            Some(struct_layouts),
            true,
        )
    }

    pub fn compile_with_options(
        func: &'a MirFunction,
        direct_callees: Option<&'a HashSet<String>>,
        direct_callee_indices: Option<&'a HashMap<String, usize>>,
        struct_layouts: Option<&'a HashMap<String, Rc<StructLayout>>>,
    ) -> Result<Function, MirVmBackendError> {
        Self::compile_inner(
            func,
            direct_callees,
            direct_callee_indices,
            struct_layouts,
            struct_layouts.is_some(),
        )
    }

    fn compile_inner(
        func: &'a MirFunction,
        direct_callees: Option<&'a HashSet<String>>,
        direct_callee_indices: Option<&'a HashMap<String, usize>>,
        struct_layouts: Option<&'a HashMap<String, Rc<StructLayout>>>,
        use_field_slots: bool,
    ) -> Result<Function, MirVmBackendError> {
        let param_count = func.params.len() as u8;
        let mut bytecode_fn = Function::new(func.signature.name.clone(), param_count, false);
        let local_types = Self::infer_local_types(func);

        let (loop_headers, in_loop_blocks, hoisted_consts_map, hoisted_consts_order, scratch_start) =
            if direct_callees.is_none() {
                Self::plan_loop_invariant_consts(func)
            } else {
                let (_, in_loops) = Self::compute_loop_blocks(func);
                (HashSet::new(), in_loops, HashMap::new(), Vec::new(), func.locals.len())
            };

        let local_read_counts = Self::compute_local_read_counts(func);

        let mut compiler = MirToBytecodeCompiler {
            func,
            local_types,
            chunk: Chunk::new(),
            scratch_start,
            scratch_count: 0,
            max_scratch: 0,
            block_pcs: HashMap::new(),
            terminator_pcs: HashMap::new(),
            pending_jumps: Vec::new(),
            direct_callees,
            direct_callee_indices,
            struct_layouts,
            use_field_slots,
            loop_headers,
            in_loop_blocks,
            hoisted_consts_map,
            hoisted_consts_order,
            local_read_counts,
            current_block_start_pc: 0,
            current_block_in_loop: false,
        };

        compiler.compile_blocks()?;
        compiler.patch_all_jumps();
        compiler.chunk.compute_bb_gas_weights();
        compiler.chunk.build_packed_code();

        bytecode_fn.chunk = compiler.chunk;
        bytecode_fn.register_count = (compiler.scratch_start + compiler.max_scratch) as u8;
        bytecode_fn.trusted_signature = Self::is_mir_signature_trusted(func);
        bytecode_fn.terminator_pcs = compiler.terminator_pcs;
        bytecode_fn.block_entry_pcs = compiler
            .block_pcs
            .into_iter()
            .map(|(bid, pc)| (bid.0, pc))
            .collect();
        Ok(bytecode_fn)
    }

    fn is_trusted_mir_type(ty: &MirType) -> bool {
        match ty {
            MirType::Unit | MirType::Bool | MirType::Int | MirType::Float | MirType::String => true,
            MirType::Struct(name) | MirType::Enum(name) => !matches!(
                name.as_str(),
                "LuaValue"
                    | "LuaTable"
                    | "LuaFunction"
                    | "LuaUserdata"
                    | "LuaThread"
                    | "unknown"
                    | "Schema"
                    | "SchemaBuilder"
            ),
            MirType::Array(inner) | MirType::Option(inner) => Self::is_trusted_mir_type(inner),
            MirType::Map(k, v) => Self::is_trusted_mir_type(k) && Self::is_trusted_mir_type(v),
            MirType::Tuple(elems) => elems.iter().all(Self::is_trusted_mir_type),
            MirType::Function {
                params,
                return_type,
            } => {
                params.iter().all(Self::is_trusted_mir_type)
                    && Self::is_trusted_mir_type(return_type)
            }
            MirType::Effect { .. } | MirType::Unknown => false,
        }
    }

    fn is_mir_signature_trusted(func: &MirFunction) -> bool {
        if func.signature.effects.is_some() || func.signature.param_types.len() != func.params.len()
        {
            return false;
        }
        func.signature
            .param_types
            .iter()
            .all(Self::is_trusted_mir_type)
            && Self::is_trusted_mir_type(&func.signature.return_type)
    }

    fn compute_loop_blocks(func: &MirFunction) -> (HashSet<BlockId>, HashSet<BlockId>) {
        if func.blocks.is_empty() {
            return (HashSet::new(), HashSet::new());
        }
        let loop_headers = crate::mir::opt::Safepoints::find_loop_headers(&func.blocks);
        if loop_headers.is_empty() {
            return (loop_headers, HashSet::new());
        }

        let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::with_capacity(func.blocks.len());
        for block in &func.blocks {
            if let Some(term) = &block.terminator {
                for succ in crate::mir::opt::Safepoints::terminator_successors(term) {
                    preds.entry(succ).or_default().push(block.id);
                }
            }
        }

        let backedges = crate::mir::opt::Safepoints::find_loop_backedges(&func.blocks);
        let mut in_loop_blocks: HashSet<BlockId> = HashSet::new();
        for (latch, header) in backedges {
            in_loop_blocks.insert(header);
            if latch != header && in_loop_blocks.insert(latch) {
                let mut worklist = alloc::vec![latch];
                while let Some(b) = worklist.pop() {
                    if let Some(ps) = preds.get(&b) {
                        for &p in ps {
                            if p != header && in_loop_blocks.insert(p) {
                                worklist.push(p);
                            }
                        }
                    }
                }
            }
        }
        (loop_headers, in_loop_blocks)
    }

    fn plan_loop_invariant_consts(
        func: &MirFunction,
    ) -> (
        HashSet<BlockId>,
        HashSet<BlockId>,
        HashMap<HoistConstKey, Register>,
        Vec<(HoistConstKey, Register)>,
        usize,
    ) {
        let mut next_reg = func.locals.len();
        if func.blocks.is_empty() || next_reg >= 250 {
            let (loop_headers, in_loop_blocks) = Self::compute_loop_blocks(func);
            return (loop_headers, in_loop_blocks, HashMap::new(), Vec::new(), next_reg);
        }

        let (loop_headers, in_loop_blocks) = Self::compute_loop_blocks(func);
        if loop_headers.is_empty() {
            return (loop_headers, in_loop_blocks, HashMap::new(), Vec::new(), next_reg);
        }

        let mut hoisted_map: HashMap<HoistConstKey, Register> = HashMap::new();
        let mut hoisted_order: Vec<(HoistConstKey, Register)> = Vec::new();
        let mut record_const = |c: &MirConstant| {
            if next_reg >= 250 {
                return;
            }
            if let Some(key) = HoistConstKey::from_mir_const(c)
                && !hoisted_map.contains_key(&key)
            {
                let reg = next_reg as Register;
                next_reg += 1;
                hoisted_map.insert(key, reg);
                hoisted_order.push((key, reg));
            }
        };

        for block in &func.blocks {
            if !in_loop_blocks.contains(&block.id) {
                continue;
            }
            for stmt in &block.statements {
                if let Statement::Assign(_, rvalue, _) = stmt {
                    match rvalue {
                        Rvalue::Use(op) => {
                            if let Operand::Constant(c) = op {
                                record_const(c);
                            }
                        }
                        Rvalue::Cast(op, target_ty) => {
                            if let Operand::Constant(c) = op {
                                record_const(c);
                            }
                            if matches!(target_ty, MirType::Float) {
                                record_const(&MirConstant::Float(0.0));
                            }
                        }
                        Rvalue::BinaryOp(_, lhs, rhs) => {
                            if let Operand::Constant(c) = lhs {
                                record_const(c);
                            }
                            if let Operand::Constant(c) = rhs {
                                record_const(c);
                            }
                        }
                        Rvalue::UnaryOp(_, op) => {
                            if let Operand::Constant(c) = op {
                                record_const(c);
                            }
                        }
                        Rvalue::Aggregate(_, elems) => {
                            for elem in elems {
                                if let Operand::Constant(c) = elem {
                                    record_const(c);
                                }
                            }
                        }
                        Rvalue::Closure { captures, .. } => {
                            for (_, op, _) in captures {
                                if let Operand::Constant(c) = op {
                                    record_const(c);
                                }
                            }
                        }
                    }
                }
            }
            if let Some(term) = &block.terminator {
                match term {
                    Terminator::Cond { cond, .. } => {
                        if let Operand::Constant(c) = cond {
                            record_const(c);
                        }
                    }
                    Terminator::Switch { discr, cases, .. } => {
                        if let Operand::Constant(c) = discr {
                            record_const(c);
                        }
                        for (val, _) in cases {
                            record_const(&MirConstant::Int(*val));
                        }
                    }
                    Terminator::Return(Some(op), _) => {
                        if let Operand::Constant(c) = op {
                            record_const(c);
                        }
                    }
                    Terminator::Call { args, .. } => {
                        for arg in args {
                            if let Operand::Constant(c) = arg {
                                record_const(c);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        (loop_headers, in_loop_blocks, hoisted_map, hoisted_order, next_reg)
    }

    fn lookup_hoisted_const(&self, c: &MirConstant) -> Option<Register> {
        let key = HoistConstKey::from_mir_const(c)?;
        self.hoisted_consts_map.get(&key).copied()
    }

    fn emit_hoisted_consts(&mut self) {
        let line = 0usize;
        for &(key, reg) in &self.hoisted_consts_order {
            let const_idx = self.chunk.add_constant(key.to_value());
            self.chunk.emit(Instruction::LoadConst(reg, const_idx), line);
        }
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
            Rvalue::Closure { .. } => VmTypeLattice::Top,
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

    fn lookup_struct_name(&self, base: &Place) -> Option<String> {
        match self.place_type(base) {
            MirType::Struct(name) => Some(name),
            _ => match base {
                Place::Local(loc) => match self.func.locals.get(loc.0).map(|l| &l.ty) {
                    Some(MirType::Struct(name)) => Some(name.clone()),
                    _ => None,
                },
                _ => None,
            },
        }
    }

    fn lookup_struct_field_name(&self, base: &Place, field_idx: usize) -> Option<Rc<String>> {
        let struct_name = self.lookup_struct_name(base)?;
        let layout = self.lookup_struct_layout(&struct_name)?;
        layout.field_names().get(field_idx).cloned()
    }

    fn can_emit_struct_field_slot(&self, base: &Place, field_idx: usize) -> bool {
        if !self.use_field_slots || field_idx > u8::MAX as usize {
            return false;
        }
        let Some(struct_name) = self.lookup_struct_name(base) else {
            return false;
        };
        if struct_name == "LuaTable" || struct_name.ends_with(".LuaTable") {
            return false;
        }
        self.lookup_struct_layout(&struct_name)
            .is_some_and(|layout| field_idx < layout.field_names().len())
    }

    fn emit_get_field(&mut self, dst: Register, base: &Place, field_idx: usize, line: usize) {
        let base_reg = self.place_to_reg(base);
        if self.can_emit_struct_field_slot(base, field_idx) {
            self.chunk
                .emit(Instruction::GetFieldSlot(dst, base_reg, field_idx as u8), line);
            return;
        }
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
        if self.can_emit_struct_field_slot(base, field_idx) {
            let slot = field_idx as u8;
            if self.try_fuse_get_field_add_set_int(base_reg, slot, val_reg, line) {
                return;
            }
            self.chunk
                .emit(Instruction::SetFieldSlot(base_reg, slot, val_reg), line);
            return;
        }
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

    fn count_place_reads(place: &Place, counts: &mut [usize]) {
        match place {
            Place::Local(loc) => {
                if let Some(c) = counts.get_mut(loc.0) {
                    *c += 1;
                }
            }
            Place::Index(base, idx) => {
                if let Some(c) = counts.get_mut(idx.0) {
                    *c += 1;
                }
                Self::count_place_reads(base, counts);
            }
            Place::Field(base, _) => {
                Self::count_place_reads(base, counts);
            }
        }
    }

    fn count_operand_reads(op: &Operand, counts: &mut [usize]) {
        match op {
            Operand::Copy(p) | Operand::Move(p) => Self::count_place_reads(p, counts),
            Operand::Constant(_) => {}
        }
    }

    fn count_rvalue_reads(rvalue: &Rvalue, counts: &mut [usize]) {
        match rvalue {
            Rvalue::Use(op) | Rvalue::Cast(op, _) | Rvalue::UnaryOp(_, op) => {
                Self::count_operand_reads(op, counts);
            }
            Rvalue::BinaryOp(_, lhs, rhs) => {
                Self::count_operand_reads(lhs, counts);
                Self::count_operand_reads(rhs, counts);
            }
            Rvalue::Aggregate(_, elems) => {
                for elem in elems {
                    Self::count_operand_reads(elem, counts);
                }
            }
            Rvalue::Closure { captures, .. } => {
                for (_, op, _) in captures {
                    Self::count_operand_reads(op, counts);
                }
            }
        }
    }

    fn compute_local_read_counts(func: &MirFunction) -> Vec<usize> {
        let mut counts = alloc::vec![0usize; func.locals.len()];
        for block in &func.blocks {
            for stmt in &block.statements {
                if let Statement::Assign(place, rvalue, _) = stmt {
                    match place {
                        Place::Local(_) => {}
                        Place::Index(base, idx) => {
                            if let Some(c) = counts.get_mut(idx.0) {
                                *c += 1;
                            }
                            Self::count_place_reads(base, &mut counts);
                        }
                        Place::Field(base, _) => {
                            Self::count_place_reads(base, &mut counts);
                        }
                    }
                    Self::count_rvalue_reads(rvalue, &mut counts);
                }
            }
            if let Some(term) = &block.terminator {
                match term {
                    Terminator::Cond { cond, .. } => Self::count_operand_reads(cond, &mut counts),
                    Terminator::Switch { discr, .. } => {
                        Self::count_operand_reads(discr, &mut counts);
                    }
                    Terminator::Return(Some(op), _) => Self::count_operand_reads(op, &mut counts),
                    Terminator::Call {
                        args, destination, ..
                    } => {
                        for arg in args {
                            Self::count_operand_reads(arg, &mut counts);
                        }
                        match destination {
                            Place::Local(_) => {}
                            Place::Index(base, idx) => {
                                if let Some(c) = counts.get_mut(idx.0) {
                                    *c += 1;
                                }
                                Self::count_place_reads(base, &mut counts);
                            }
                            Place::Field(base, _) => {
                                Self::count_place_reads(base, &mut counts);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        counts
    }

    #[inline]
    fn is_fusible_temp_reg(&self, reg: Register) -> bool {
        let idx = reg as usize;
        if idx >= self.scratch_start {
            return true;
        }
        self.func.locals.get(idx).is_some_and(|l| l.name.is_none())
            && self.local_read_counts.get(idx).copied().unwrap_or(0) == 1
    }

    #[inline]
    fn pop_last_instruction(&mut self) -> Option<Instruction> {
        self.chunk.pop_instruction()
    }

    /// Intra-block peephole fusion for `GetFieldSlot` + (`LoadConst`) + `Add`/`AddInt` + `SetFieldSlot`
    /// into `GetFieldAddSetInt`. Strictly restricted to `self.current_block_start_pc..` so
    /// `block_entry_pcs` / `bb_leaders` are never crossed.
    fn try_fuse_get_field_add_set_int(
        &mut self,
        base_reg: Register,
        slot: u8,
        val_reg: Register,
        line: usize,
    ) -> bool {
        let len = self.chunk.instructions.len();
        if len >= self.current_block_start_pc + 2 {
            let prev2 = self.chunk.instructions[len - 2];
            let prev1 = self.chunk.instructions[len - 1];
            if let (
                Instruction::GetFieldSlot(t1, obj1, slot1),
                Instruction::Add(t2, l, r) | Instruction::AddInt(t2, l, r),
            ) = (prev2, prev1)
                && obj1 == base_reg
                && slot1 == slot
                && l == t1
                && t2 == val_reg
                && t1 != base_reg
                && t1 != r
                && self.is_fusible_temp_reg(t1)
                && self.is_fusible_temp_reg(t2)
            {
                self.pop_last_instruction();
                self.pop_last_instruction();
                self.chunk
                    .emit(Instruction::GetFieldAddSetInt(base_reg, slot, r), line);
                return true;
            }
        }
        if len >= self.current_block_start_pc + 3 {
            let prev3 = self.chunk.instructions[len - 3];
            let prev2 = self.chunk.instructions[len - 2];
            let prev1 = self.chunk.instructions[len - 1];
            if let (
                Instruction::GetFieldSlot(t1, obj1, slot1),
                Instruction::LoadConst(c_reg, k),
                Instruction::Add(t2, l, r) | Instruction::AddInt(t2, l, r),
            ) = (prev3, prev2, prev1)
                && obj1 == base_reg
                && slot1 == slot
                && l == t1
                && r == c_reg
                && t2 == val_reg
                && (c_reg as usize) >= self.scratch_start
                && c_reg != t1
                && c_reg != base_reg
                && t1 != base_reg
                && self.is_fusible_temp_reg(t1)
                && self.is_fusible_temp_reg(t2)
            {
                self.pop_last_instruction();
                self.pop_last_instruction();
                self.pop_last_instruction();
                self.chunk.emit(Instruction::LoadConst(c_reg, k), line);
                self.chunk
                    .emit(Instruction::GetFieldAddSetInt(base_reg, slot, c_reg), line);
                return true;
            }
        }
        false
    }

    /// Intra-block peephole fusion for `LtInt`/`LeInt` + `Terminator::Cond` into
    /// `BranchLtInt` / `BranchLeInt` without crossing `self.current_block_start_pc`.
    /// Updates `self.terminator_pcs[current_block_id]` on fusion and returns `true`.
    fn try_compile_fused_cond_terminator(
        &mut self,
        term: &Terminator,
        current_block_id: BlockId,
        next_block_id: Option<BlockId>,
    ) -> bool {
        let Terminator::Cond {
            cond,
            then_block,
            else_block,
            span,
        } = term
        else {
            return false;
        };
        if then_block == else_block {
            return false;
        }
        let (Operand::Copy(Place::Local(cond_loc)) | Operand::Move(Place::Local(cond_loc))) = cond
        else {
            return false;
        };
        if self.chunk.instructions.len() <= self.current_block_start_pc {
            return false;
        }
        let cond_reg = cond_loc.0 as Register;
        if !self.is_fusible_temp_reg(cond_reg) {
            return false;
        }
        let Some((is_lt, lhs, rhs)) = (match self.chunk.instructions.last().copied() {
            Some(Instruction::LtInt(c, lhs, rhs)) if c == cond_reg => Some((true, lhs, rhs)),
            Some(Instruction::LeInt(c, lhs, rhs)) if c == cond_reg => Some((false, lhs, rhs)),
            _ => None,
        }) else {
            return false;
        };

        let is_fallthrough =
            |target: BlockId| next_block_id == Some(target) && target != current_block_id;
        let line = span.start_line;

        self.pop_last_instruction();
        let term_pc = self.chunk.instructions.len();
        self.terminator_pcs.insert(current_block_id.0, term_pc);

        if is_fallthrough(*then_block) {
            let inst = if is_lt {
                Instruction::BranchLtInt(lhs, rhs, 0)
            } else {
                Instruction::BranchLeInt(lhs, rhs, 0)
            };
            let br_pc = self.chunk.emit(inst, line);
            self.pending_jumps.push((br_pc, *else_block));
        } else if is_fallthrough(*else_block) {
            // BranchLeInt(rhs, lhs, then) jumps iff !(rhs <= lhs) <=> lhs < rhs.
            // BranchLtInt(rhs, lhs, then) jumps iff !(rhs < lhs)  <=> lhs <= rhs.
            let inst = if is_lt {
                Instruction::BranchLeInt(rhs, lhs, 0)
            } else {
                Instruction::BranchLtInt(rhs, lhs, 0)
            };
            let br_pc = self.chunk.emit(inst, line);
            self.pending_jumps.push((br_pc, *then_block));
        } else {
            let inst = if is_lt {
                Instruction::BranchLtInt(lhs, rhs, 0)
            } else {
                Instruction::BranchLeInt(lhs, rhs, 0)
            };
            let br_pc = self.chunk.emit(inst, line);
            self.pending_jumps.push((br_pc, *else_block));

            let jump_then = self.chunk.emit(Instruction::Jump(0), line);
            self.pending_jumps.push((jump_then, *then_block));
        }
        true
    }

    fn compile_blocks(&mut self) -> Result<(), MirVmBackendError> {
        let entry_is_loop_header = self
            .func
            .blocks
            .first()
            .is_some_and(|b| self.loop_headers.contains(&b.id));

        if entry_is_loop_header && !self.hoisted_consts_order.is_empty() {
            self.emit_hoisted_consts();
        }

        for (idx, block) in self.func.blocks.iter().enumerate() {
            let pc = self.chunk.instructions.len();
            self.block_pcs.insert(block.id, pc);
            self.current_block_start_pc = pc;
            self.current_block_in_loop = self.in_loop_blocks.contains(&block.id);

            if idx == 0 && !entry_is_loop_header && !self.hoisted_consts_order.is_empty() {
                self.emit_hoisted_consts();
            }

            for stmt in &block.statements {
                self.reset_scratch();
                self.compile_statement(stmt)?;
            }

            self.reset_scratch();
            let next_block_id = self.func.blocks.get(idx + 1).map(|b| b.id);
            if let Some(term) = &block.terminator {
                if !self.try_compile_fused_cond_terminator(term, block.id, next_block_id) {
                    let term_pc = self.chunk.instructions.len();
                    self.terminator_pcs.insert(block.id.0, term_pc);
                    self.compile_terminator(term, block.id, next_block_id);
                }
            } else {
                let term_pc = self.chunk.instructions.len();
                self.terminator_pcs.insert(block.id.0, term_pc);
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
                if let Some(hoisted_reg) = self.lookup_hoisted_const(c) {
                    return hoisted_reg;
                }
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

    fn place_reads_local(place: &Place, target: LocalId) -> bool {
        match place {
            Place::Local(loc) => *loc == target,
            Place::Index(base, idx) => *idx == target || Self::place_reads_local(base, target),
            Place::Field(base, _) => Self::place_reads_local(base, target),
        }
    }

    fn operand_reads_local(op: &Operand, target: LocalId) -> bool {
        match op {
            Operand::Copy(p) | Operand::Move(p) => Self::place_reads_local(p, target),
            Operand::Constant(_) => false,
        }
    }

    fn rvalue_reads_local(rvalue: &Rvalue, target: LocalId) -> bool {
        match rvalue {
            Rvalue::Use(op) | Rvalue::Cast(op, _) | Rvalue::UnaryOp(_, op) => {
                Self::operand_reads_local(op, target)
            }
            Rvalue::BinaryOp(_, lhs, rhs) => {
                Self::operand_reads_local(lhs, target) || Self::operand_reads_local(rhs, target)
            }
            Rvalue::Aggregate(_, elems) => {
                elems.iter().any(|op| Self::operand_reads_local(op, target))
            }
            Rvalue::Closure { captures, .. } => captures
                .iter()
                .any(|(_, op, _)| Self::operand_reads_local(op, target)),
        }
    }

    fn is_local_read(&self, target: LocalId) -> bool {
        for block in &self.func.blocks {
            for stmt in &block.statements {
                if let Statement::Assign(place, rvalue, _) = stmt {
                    match place {
                        Place::Local(_) => {}
                        Place::Index(base, idx) => {
                            if *idx == target || Self::place_reads_local(base, target) {
                                return true;
                            }
                        }
                        Place::Field(base, _) => {
                            if Self::place_reads_local(base, target) {
                                return true;
                            }
                        }
                    }
                    if Self::rvalue_reads_local(rvalue, target) {
                        return true;
                    }
                }
            }
            if let Some(term) = &block.terminator {
                match term {
                    Terminator::Cond { cond, .. } if Self::operand_reads_local(cond, target) => {
                        return true;
                    }
                    Terminator::Switch { discr, .. }
                        if Self::operand_reads_local(discr, target) =>
                    {
                        return true;
                    }
                    Terminator::Return(Some(op), _) if Self::operand_reads_local(op, target) => {
                        return true;
                    }
                    Terminator::Call { args, .. }
                        if args.iter().any(|a| Self::operand_reads_local(a, target)) =>
                    {
                        return true;
                    }
                    _ => {}
                }
            }
        }
        false
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
                        let src_reg = src.0 as Register;
                        if dst_reg != src_reg {
                            if self.chunk.instructions.len() > self.current_block_start_pc
                                && let Some(Instruction::AddIntImm(prev_dst, prev_src, imm)) =
                                    self.chunk.instructions.last().copied()
                                && prev_dst == src_reg
                                && self.is_fusible_temp_reg(src_reg)
                            {
                                let last_idx = self.chunk.instructions.len() - 1;
                                self.chunk.set_instruction(
                                    last_idx,
                                    Instruction::AddIntImm(dst_reg, prev_src, imm),
                                );
                            } else {
                                self.chunk.emit(Instruction::Move(dst_reg, src_reg), line);
                            }
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
                        if let Some(hoisted_reg) = self.lookup_hoisted_const(c) {
                            if dst_reg != hoisted_reg {
                                self.chunk.emit(Instruction::Move(dst_reg, hoisted_reg), line);
                            }
                        } else {
                            self.load_constant_into_reg(c, dst_reg, line);
                        }
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
                    _ if self.operand_is_option(op) => {
                        let r_op = self.operand_to_reg(op, line);
                        self.chunk.emit(Instruction::GetEnumValue(dst_reg, r_op, 0), line);
                    }
                    MirType::Float => {
                        let r_op = self.operand_to_reg(op, line);
                        let zero_f_reg = if let Some(hoisted_reg) =
                            self.lookup_hoisted_const(&MirConstant::Float(0.0))
                        {
                            hoisted_reg
                        } else {
                            let reg = self.next_scratch();
                            self.load_constant_into_reg(&MirConstant::Float(0.0), reg, line);
                            reg
                        };
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
                let allow_add_int_imm = self.current_block_in_loop
                    || (dst_reg as usize) >= self.scratch_start
                    || self
                        .func
                        .locals
                        .get(dst_reg as usize)
                        .is_some_and(|l| l.name.is_none());
                if allow_add_int_imm && lhs_ty == MirType::Int && rhs_ty == MirType::Int {
                    match *binop {
                        MirBinaryOp::Add => {
                            if let Operand::Constant(MirConstant::Int(v)) = rhs
                                && let Ok(imm) = i8::try_from(*v)
                            {
                                let r_lhs = self.operand_to_reg(lhs, line);
                                self.chunk
                                    .emit(Instruction::AddIntImm(dst_reg, r_lhs, imm), line);
                                return Ok(());
                            }
                            if let Operand::Constant(MirConstant::Int(v)) = lhs
                                && let Ok(imm) = i8::try_from(*v)
                            {
                                let r_rhs = self.operand_to_reg(rhs, line);
                                self.chunk
                                    .emit(Instruction::AddIntImm(dst_reg, r_rhs, imm), line);
                                return Ok(());
                            }
                        }
                        MirBinaryOp::Sub => {
                            if let Operand::Constant(MirConstant::Int(v)) = rhs
                                && let Some(imm) =
                                    v.checked_neg().and_then(|neg| i8::try_from(neg).ok())
                            {
                                let r_lhs = self.operand_to_reg(lhs, line);
                                self.chunk
                                    .emit(Instruction::AddIntImm(dst_reg, r_lhs, imm), line);
                                return Ok(());
                            }
                        }
                        _ => {}
                    }
                }
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
            Rvalue::Closure { .. } => {
                return Err(MirVmBackendError::UnsupportedRvalue(
                    "Rvalue::Closure".into(),
                ));
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
                if self.can_emit_struct_field_slot(base, *f)
                    && let Rvalue::BinaryOp(MirBinaryOp::Add, lhs, rhs) = rvalue
                    && matches!(
                        lhs,
                        Operand::Copy(Place::Field(lhs_base, lhs_f))
                            | Operand::Move(Place::Field(lhs_base, lhs_f))
                            if *lhs_f == *f && lhs_base == base
                    )
                {
                    let base_reg = self.place_to_reg(base);
                    let addend_reg = self.operand_to_reg(rhs, line);
                    self.chunk.emit(
                        Instruction::GetFieldAddSetInt(base_reg, *f as u8, addend_reg),
                        line,
                    );
                    return Ok(());
                }
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
                    let val_reg = if let Some(hoisted_reg) =
                        self.lookup_hoisted_const(&MirConstant::Int(*val))
                    {
                        hoisted_reg
                    } else {
                        let reg = self.next_scratch();
                        self.load_constant_into_reg(&MirConstant::Int(*val), reg, line);
                        reg
                    };
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
                let shadowed_by_direct = self
                    .direct_callee_indices
                    .is_some_and(|map| map.contains_key(callee.as_str()));
                if !shadowed_by_direct {
                    match (callee.as_str(), args.len()) {
                        ("array.len", 1) => {
                            let arr_reg = self.operand_to_reg(&args[0], line);
                            let dest_reg = self.place_to_reg(destination);
                            self.chunk.emit(Instruction::ArrayLen(dest_reg, arr_reg), line);
                            if !is_fallthrough(*success) {
                                let jump_idx = self.chunk.emit(Instruction::Jump(0), line);
                                self.pending_jumps.push((jump_idx, *success));
                            }
                            return;
                        }
                        ("array.push", 2) => {
                            let arr_reg = self.operand_to_reg(&args[0], line);
                            let val_reg = self.operand_to_reg(&args[1], line);
                            self.chunk.emit(Instruction::ArrayPush(arr_reg, val_reg), line);
                            if let Place::Local(dst) = destination
                                && self.is_local_read(*dst)
                            {
                                self.chunk.emit(Instruction::LoadNil(dst.0 as Register), line);
                            }
                            if !is_fallthrough(*success) {
                                let jump_idx = self.chunk.emit(Instruction::Jump(0), line);
                                self.pending_jumps.push((jump_idx, *success));
                            }
                            return;
                        }
                        ("array.set", 3) => {
                            let arr_reg = self.operand_to_reg(&args[0], line);
                            let idx_reg = self.operand_to_reg(&args[1], line);
                            let val_reg = self.operand_to_reg(&args[2], line);
                            self.chunk
                                .emit(Instruction::SetIndex(arr_reg, idx_reg, val_reg), line);
                            if let Place::Local(dst) = destination
                                && self.is_local_read(*dst)
                            {
                                self.chunk.emit(Instruction::LoadNil(dst.0 as Register), line);
                            }
                            if !is_fallthrough(*success) {
                                let jump_idx = self.chunk.emit(Instruction::Jump(0), line);
                                self.pending_jumps.push((jump_idx, *success));
                            }
                            return;
                        }
                        ("array.get_unwrap", 2) => {
                            let arr_reg = self.operand_to_reg(&args[0], line);
                            let idx_reg = self.operand_to_reg(&args[1], line);
                            let dest_reg = self.place_to_reg(destination);
                            self.chunk
                                .emit(Instruction::ArrayGetUnwrap(dest_reg, arr_reg, idx_reg), line);
                            if !is_fallthrough(*success) {
                                let jump_idx = self.chunk.emit(Instruction::Jump(0), line);
                                self.pending_jumps.push((jump_idx, *success));
                            }
                            return;
                        }
                        ("map.new", 0) => {
                            let dest_reg = self.place_to_reg(destination);
                            self.chunk.emit(Instruction::NewMap(dest_reg), line);
                            if !is_fallthrough(*success) {
                                let jump_idx = self.chunk.emit(Instruction::Jump(0), line);
                                self.pending_jumps.push((jump_idx, *success));
                            }
                            return;
                        }
                        ("map.get_unwrap", 2) => {
                            let arg_base = self.next_scratch();
                            let arg1 = self.next_scratch();
                            let m_reg = self.operand_to_reg(&args[0], line);
                            let k_reg = self.operand_to_reg(&args[1], line);
                            self.chunk.emit(Instruction::Move(arg_base, m_reg), line);
                            self.chunk.emit(Instruction::Move(arg1, k_reg), line);

                            let fn_reg = self.next_scratch();
                            let map_idx = self.chunk.add_constant(Value::string("map".to_string()));
                            let get_idx = self.chunk.add_constant(Value::string("get".to_string()));
                            self.chunk.emit(Instruction::LoadGlobal(fn_reg, map_idx), line);
                            self.chunk.emit(Instruction::GetField(fn_reg, fn_reg, get_idx), line);

                            let opt_reg = self.next_scratch();
                            self.chunk.emit(Instruction::Call(fn_reg, arg_base, 2, opt_reg), line);

                            let is_some_reg = self.next_scratch();
                            let opt_name_idx =
                                self.chunk.add_constant(Value::string("Option".to_string()));
                            let some_name_idx =
                                self.chunk.add_constant(Value::string("Some".to_string()));
                            self.chunk.emit(
                                Instruction::IsEnumVariant(
                                    is_some_reg,
                                    opt_reg,
                                    opt_name_idx,
                                    some_name_idx,
                                ),
                                line,
                            );

                            let j_ok = self.chunk.emit(Instruction::JumpIf(is_some_reg, 0), line);
                            let err_fn_reg = self.next_scratch();
                            let err_arg_reg = self.next_scratch();
                            let err_name_idx =
                                self.chunk.add_constant(Value::string("error".to_string()));
                            let err_msg_idx = self
                                .chunk
                                .add_constant(Value::string("called unwrap() on None".to_string()));
                            self.chunk
                                .emit(Instruction::LoadGlobal(err_fn_reg, err_name_idx), line);
                            self.chunk
                                .emit(Instruction::LoadConst(err_arg_reg, err_msg_idx), line);
                            let dest_reg = self.place_to_reg(destination);
                            self.chunk
                                .emit(Instruction::Call(err_fn_reg, err_arg_reg, 1, dest_reg), line);

                            let ok_pc = self.chunk.instructions.len();
                            self.chunk.patch_jump(j_ok, ok_pc);
                            self.chunk
                                .emit(Instruction::GetEnumValue(dest_reg, opt_reg, 0), line);

                            if !is_fallthrough(*success) {
                                let jump_idx = self.chunk.emit(Instruction::Jump(0), line);
                                self.pending_jumps.push((jump_idx, *success));
                            }
                            return;
                        }
                        ("map.get_unwrap_or", 3) => {
                            let arg_base = self.next_scratch();
                            let arg1 = self.next_scratch();
                            let m_reg = self.operand_to_reg(&args[0], line);
                            let k_reg = self.operand_to_reg(&args[1], line);
                            let def_reg = self.operand_to_reg(&args[2], line);
                            self.chunk.emit(Instruction::Move(arg_base, m_reg), line);
                            self.chunk.emit(Instruction::Move(arg1, k_reg), line);

                            let fn_reg = self.next_scratch();
                            let map_idx = self.chunk.add_constant(Value::string("map".to_string()));
                            let get_idx = self.chunk.add_constant(Value::string("get".to_string()));
                            self.chunk.emit(Instruction::LoadGlobal(fn_reg, map_idx), line);
                            self.chunk.emit(Instruction::GetField(fn_reg, fn_reg, get_idx), line);

                            let opt_reg = self.next_scratch();
                            self.chunk.emit(Instruction::Call(fn_reg, arg_base, 2, opt_reg), line);

                            let is_some_reg = self.next_scratch();
                            let opt_name_idx =
                                self.chunk.add_constant(Value::string("Option".to_string()));
                            let some_name_idx =
                                self.chunk.add_constant(Value::string("Some".to_string()));
                            self.chunk.emit(
                                Instruction::IsEnumVariant(
                                    is_some_reg,
                                    opt_reg,
                                    opt_name_idx,
                                    some_name_idx,
                                ),
                                line,
                            );

                            let dest_reg = self.place_to_reg(destination);
                            let j_ok = self.chunk.emit(Instruction::JumpIf(is_some_reg, 0), line);
                            self.chunk.emit(Instruction::Move(dest_reg, def_reg), line);
                            let j_done = self.chunk.emit(Instruction::Jump(0), line);

                            let ok_pc = self.chunk.instructions.len();
                            self.chunk.patch_jump(j_ok, ok_pc);
                            self.chunk
                                .emit(Instruction::GetEnumValue(dest_reg, opt_reg, 0), line);

                            let done_pc = self.chunk.instructions.len();
                            self.chunk.patch_jump(j_done, done_pc);

                            if !is_fallthrough(*success) {
                                let jump_idx = self.chunk.emit(Instruction::Jump(0), line);
                                self.pending_jumps.push((jump_idx, *success));
                            }
                            return;
                        }
                        ("__mir_unknown_get_field", 2) => {
                            if let Operand::Constant(MirConstant::String(field_name)) = &args[1] {
                                let obj_reg = self.operand_to_reg(&args[0], line);
                                let field_idx =
                                    self.chunk.add_constant(Value::string(field_name.clone()));
                                let dest_reg = self.place_to_reg(destination);
                                self.chunk.emit(
                                    Instruction::GetField(dest_reg, obj_reg, field_idx),
                                    line,
                                );
                                if !is_fallthrough(*success) {
                                    let jump_idx = self.chunk.emit(Instruction::Jump(0), line);
                                    self.pending_jumps.push((jump_idx, *success));
                                }
                                return;
                            }
                        }
                        ("__mir_unknown_call_method", n) if n >= 2 => {
                            if let Operand::Constant(MirConstant::String(method_name)) = &args[1] {
                                let obj_reg = self.operand_to_reg(&args[0], line);
                                let rest = &args[2..];
                                let rest_count = rest.len() as u8;
                                let first_arg_reg = if rest_count > 0 {
                                    let scratch_base = self.next_scratch();
                                    for _ in 1..rest.len() {
                                        self.next_scratch();
                                    }
                                    for (i, arg) in rest.iter().enumerate() {
                                        let target_reg = scratch_base + i as u8;
                                        let r = self.operand_to_reg(arg, line);
                                        if r != target_reg {
                                            self.chunk.emit(Instruction::Move(target_reg, r), line);
                                        }
                                    }
                                    scratch_base
                                } else {
                                    0
                                };
                                let method_idx =
                                    self.chunk.add_constant(Value::string(method_name.clone()));
                                let dest_reg = self.place_to_reg(destination);
                                self.chunk.emit(
                                    Instruction::CallMethod(
                                        obj_reg,
                                        method_idx,
                                        first_arg_reg,
                                        rest_count,
                                        dest_reg,
                                    ),
                                    line,
                                );
                                if !is_fallthrough(*success) {
                                    let jump_idx = self.chunk.emit(Instruction::Jump(0), line);
                                    self.pending_jumps.push((jump_idx, *success));
                                }
                                return;
                            }
                        }
                        (
                            "__mir_unknown_is_int"
                            | "__mir_unknown_is_float"
                            | "__mir_unknown_is_string"
                            | "__mir_unknown_is_bool",
                            1,
                        ) => {
                            let variant = match callee.as_str() {
                                "__mir_unknown_is_int" => "Int",
                                "__mir_unknown_is_float" => "Float",
                                "__mir_unknown_is_string" => "String",
                                _ => "Bool",
                            };
                            let val_reg = self.operand_to_reg(&args[0], line);
                            let enum_idx =
                                self.chunk.add_constant(Value::string("LuaValue".to_string()));
                            let var_idx =
                                self.chunk.add_constant(Value::string(variant.to_string()));
                            let dest_reg = self.place_to_reg(destination);
                            self.chunk.emit(
                                Instruction::IsEnumVariant(dest_reg, val_reg, enum_idx, var_idx),
                                line,
                            );
                            if !is_fallthrough(*success) {
                                let jump_idx = self.chunk.emit(Instruction::Jump(0), line);
                                self.pending_jumps.push((jump_idx, *success));
                            }
                            return;
                        }
                        (
                            "__mir_unknown_unwrap_int"
                            | "__mir_unknown_unwrap_float"
                            | "__mir_unknown_unwrap_string"
                            | "__mir_unknown_unwrap_bool",
                            1,
                        ) => {
                            let val_reg = self.operand_to_reg(&args[0], line);
                            let dest_reg = self.place_to_reg(destination);
                            self.chunk
                                .emit(Instruction::GetEnumValue(dest_reg, val_reg, 0), line);
                            if !is_fallthrough(*success) {
                                let jump_idx = self.chunk.emit(Instruction::Jump(0), line);
                                self.pending_jumps.push((jump_idx, *success));
                            }
                            return;
                        }
                        _ => {}
                    }
                }
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
                let local_callee = if !callee.contains('.') && !callee.contains(':') {
                    self.func
                        .locals
                        .iter()
                        .rev()
                        .find(|l| l.name.as_deref() == Some(callee.as_str()))
                        .map(|l| l.id)
                } else {
                    None
                };
                if let Some(loc_id) = local_callee {
                    self.chunk
                        .emit(Instruction::Move(func_reg, loc_id.0 as Register), line);
                } else if self.use_field_slots
                    && let Some(struct_name) = parse_mir_struct_decode_callee(callee)
                {
                    let name_idx = self
                        .chunk
                        .add_constant(Value::string(alloc::format!("{struct_name}.decode")));
                    self.chunk.emit(Instruction::LoadGlobal(func_reg, name_idx), line);
                } else if let Some(&callee_idx) = self
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

fn contiguous_to_vm_struct(
    s: &ContiguousStruct,
    vm_layouts: &HashMap<String, Rc<StructLayout>>,
) -> Result<Value, String> {
    let mut values = Vec::with_capacity(s.layout.fields.len());
    for field in &s.layout.fields {
        let ord = field.ordinal;
        let off = field.byte_offset;
        if s.is_null(ord) || !s.is_initialized(ord) {
            values.push(Value::Nil);
            continue;
        }
        let val = match field.tag {
            CanonicalTypeTag::Bool => Value::Bool(
                s.read_bool_at(ord, off, None)
                    .map_err(|e| e.to_string())?
                    .unwrap_or(false),
            ),
            CanonicalTypeTag::Int64 | CanonicalTypeTag::DateTimeUtc => Value::Int(
                s.read_i64_at(ord, off, None)
                    .map_err(|e| e.to_string())?
                    .unwrap_or(0) as LustInt,
            ),
            CanonicalTypeTag::UInt64 => {
                let u = s
                    .read_u64_at(ord, off, None)
                    .map_err(|e| e.to_string())?
                    .unwrap_or(0);
                let i = i64::try_from(u).map_err(|_| "UInt64 overflows VM Int".to_string())?;
                Value::Int(i as LustInt)
            }
            CanonicalTypeTag::Float64 => Value::Float(
                s.read_f64_at(ord, off, None)
                    .map_err(|e| e.to_string())?
                    .unwrap_or(0.0) as LustFloat,
            ),
            CanonicalTypeTag::StringSlice => {
                let text = s
                    .read_str_at(ord, off, None)
                    .map_err(|e| e.to_string())?
                    .unwrap_or("");
                Value::string(text.to_string())
            }
            CanonicalTypeTag::BinarySlice => {
                let bytes = s
                    .read_bytes_at(ord, off, None)
                    .map_err(|e| e.to_string())?
                    .unwrap_or(&[]);
                Value::string(String::from_utf8_lossy(bytes).into_owned())
            }
            CanonicalTypeTag::NestedStruct(_) => {
                if let Some(nested) = s.read_nested_at(ord, None).map_err(|e| e.to_string())? {
                    contiguous_to_vm_struct(nested, vm_layouts)?
                } else {
                    Value::Nil
                }
            }
            CanonicalTypeTag::ObjectId => {
                let oid = s
                    .read_object_id_at(ord, off, None)
                    .map_err(|e| e.to_string())?
                    .unwrap_or([0u8; 12]);
                let mut hex = String::with_capacity(24);
                for b in oid {
                    use core::fmt::Write;
                    let _ = write!(&mut hex, "{:02x}", b);
                }
                Value::string(hex)
            }
            CanonicalTypeTag::Decimal128 => {
                let dec = s
                    .read_decimal128_at(ord, off, None)
                    .map_err(|e| e.to_string())?
                    .unwrap_or(0);
                Value::Int(dec as LustInt)
            }
        };
        values.push(val);
    }
    if let Some(layout) = vm_layouts.get(&s.layout.type_name) {
        Ok(Value::Struct(StructObject::new(
            s.layout.type_name.clone(),
            Rc::clone(layout),
            values,
        )))
    } else {
        Ok(Value::array(values))
    }
}

/// Registers both MIR (`__mir_struct_decode_into:<Name>`) and VM (`<Name>.decode`)
/// JSON struct decoders onto `vm` from parsed/typechecked `StructDef` definitions.
pub fn register_struct_value_decoders(
    vm: &mut crate::vm::VM,
    struct_defs: &HashMap<String, StructDef>,
) {
    if struct_defs.is_empty() {
        return;
    }

    let mut mir_module = MirModule::new("cli_struct_decoders".to_string());
    let mut vm_layouts: HashMap<String, Rc<StructLayout>> = HashMap::new();
    let mut nested_map = std::collections::HashMap::new();
    let mut def_entries: Vec<(String, crate::data::layout::LogicalLayoutV1)> = Vec::new();

    for (name, def) in struct_defs {
        let logical = struct_def_to_layout(def);
        mir_module.register_struct_layout(MirStructLayout::from(&logical));
        let simple_def_name = def
            .name
            .rsplit('.')
            .next()
            .unwrap_or(def.name.as_str())
            .to_string();
        if simple_def_name != logical.type_name {
            let mut simple_logical = logical.clone();
            simple_logical.type_name = simple_def_name.clone();
            mir_module.register_struct_layout(MirStructLayout::from(&simple_logical));
        }
        let simple_reg_name = name
            .rsplit('.')
            .next()
            .unwrap_or(name.as_str())
            .to_string();
        if simple_reg_name != logical.type_name && simple_reg_name != simple_def_name {
            let mut simple_logical = logical.clone();
            simple_logical.type_name = simple_reg_name.clone();
            mir_module.register_struct_layout(MirStructLayout::from(&simple_logical));
        }
        nested_map.insert(logical.type_id, logical.clone());
        def_entries.push((name.clone(), logical));

        let field_names: Vec<Rc<String>> = def
            .fields
            .iter()
            .map(|field| Rc::new(field.name.clone()))
            .collect();
        let field_storage: Vec<FieldStorage> = def
            .fields
            .iter()
            .map(|field| match field.ownership {
                FieldOwnership::Weak => FieldStorage::Weak,
                FieldOwnership::Strong => FieldStorage::Strong,
            })
            .collect();
        let field_types: Vec<crate::ast::Type> =
            def.fields.iter().map(|field| field.ty.clone()).collect();
        let weak_targets: Vec<Option<crate::ast::Type>> = def
            .fields
            .iter()
            .map(|field| field.weak_target.clone())
            .collect();
        let layout = Rc::new(StructLayout::new(
            def.name.clone(),
            field_names,
            field_storage,
            field_types,
            weak_targets,
        ));
        vm_layouts.insert(def.name.clone(), Rc::clone(&layout));
        vm_layouts.insert(simple_def_name, Rc::clone(&layout));
        vm_layouts.insert(name.clone(), Rc::clone(&layout));
        vm_layouts.insert(simple_reg_name, layout);
    }

    // 1. Register MIR-VM `__mir_struct_decode_into:<Name>` native decoders (slot-array ABI).
    register_module_struct_decoders(vm, &mir_module);

    // 2. Register `<Name>.decode` native decoders returning typed `Value::Struct` instances
    //    for AST bytecode and field-slot MIR bytecode execution.
    let nested_ref = Rc::new(nested_map);
    let vm_layouts_ref = Rc::new(vm_layouts);
    let counters = Rc::new(RefCell::new(LayoutPerfCounters::default()));

    for (reg_name, logical) in def_entries {
        let make_decoder = |logical: crate::data::layout::LogicalLayoutV1| {
            let nested_ref = Rc::clone(&nested_ref);
            let vm_layouts_ref = Rc::clone(&vm_layouts_ref);
            let counters_ref = Rc::clone(&counters);
            native_fn(move |args: &[Value]| {
                let input_str = args
                    .first()
                    .and_then(|v| v.as_string())
                    .ok_or_else(|| "Struct.decode expects a JSON string argument".to_string())?;
                let opts = ContiguousDecodeOptions {
                    cancel_check: None,
                    nested_layouts: Some(nested_ref.as_ref()),
                    require_nominal_tag: false,
                };
                let mut c = counters_ref.borrow_mut();
                let decoded =
                    decode_json_into_contiguous(&logical, input_str.as_bytes(), &mut c, &opts)
                        .map_err(|e| e.to_string())?;
                let val = contiguous_to_vm_struct(&decoded, &vm_layouts_ref)?;
                Ok(NativeCallResult::Return(val))
            })
        };

        let simple = reg_name
            .rsplit('.')
            .next()
            .unwrap_or(reg_name.as_str())
            .to_string();
        vm.register_native(
            alloc::format!("{reg_name}.decode"),
            Value::NativeFunction(make_decoder(logical.clone())),
        );
        if simple != reg_name {
            vm.register_native(
                alloc::format!("{simple}.decode"),
                Value::NativeFunction(make_decoder(logical)),
            );
        }
    }
}

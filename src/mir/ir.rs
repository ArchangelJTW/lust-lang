use crate::ast::Span;
use crate::data::layout::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1, TargetAbi};
use alloc::{boxed::Box, collections::VecDeque, string::String, vec::Vec};
use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(pub usize);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalId(pub usize);

impl fmt::Display for LocalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MirType {
    Unit,
    Bool,
    Int,
    Float,
    String,
    Tuple(Vec<MirType>),
    Array(Box<MirType>),
    Map(Box<MirType>, Box<MirType>),
    Struct(String),
    Enum(String),
    Option(Box<MirType>),
    Function {
        params: Vec<MirType>,
        return_type: Box<MirType>,
    },
    Effect {
        return_type: Box<MirType>,
        error_type: Box<MirType>,
    },
    Unknown,
}

impl fmt::Display for MirType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MirType::Unit => write!(f, "()"),
            MirType::Bool => write!(f, "bool"),
            MirType::Int => write!(f, "int"),
            MirType::Float => write!(f, "float"),
            MirType::String => write!(f, "string"),
            MirType::Tuple(types) => {
                write!(f, "(")?;
                for (i, ty) in types.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{ty}")?;
                }
                write!(f, ")")
            }
            MirType::Array(elem) => write!(f, "[{elem}]"),
            MirType::Map(k, v) => write!(f, "map[{k} -> {v}]"),
            MirType::Struct(name) => write!(f, "struct {name}"),
            MirType::Enum(name) => write!(f, "enum {name}"),
            MirType::Option(inner) => write!(f, "Option<{inner}>"),
            MirType::Function { params, return_type } => {
                write!(f, "fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{p}")?;
                }
                write!(f, ") -> {return_type}")
            }
            MirType::Effect { return_type, error_type } => {
                write!(f, "Effect<{return_type}, {error_type}>")
            }
            MirType::Unknown => write!(f, "unknown"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalDecl {
    pub id: LocalId,
    pub name: Option<String>,
    pub ty: MirType,
    pub is_mut: bool,
    pub span: Span,
    pub is_stack_promoted: bool,
    pub is_param: bool,
}

impl LocalDecl {
    pub fn new(id: LocalId, name: Option<String>, ty: MirType, is_mut: bool, span: Span) -> Self {
        Self {
            id,
            name,
            ty,
            is_mut,
            span,
            is_stack_promoted: false,
            is_param: false,
        }
    }

    pub fn param(id: LocalId, name: Option<String>, ty: MirType, span: Span) -> Self {
        Self {
            id,
            name,
            ty,
            is_mut: false,
            span,
            is_stack_promoted: false,
            is_param: true,
        }
    }

    pub fn with_stack_promoted(mut self, promoted: bool) -> Self {
        self.is_stack_promoted = promoted;
        self
    }

    pub fn with_param(mut self, is_param: bool) -> Self {
        self.is_param = is_param;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MirConstant {
    Unit,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

impl fmt::Display for MirConstant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MirConstant::Unit => write!(f, "()"),
            MirConstant::Bool(b) => write!(f, "{b}"),
            MirConstant::Int(i) => write!(f, "{i}"),
            MirConstant::Float(fl) => write!(f, "{fl:?}"),
            MirConstant::String(s) => write!(f, "{s:?}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Place {
    Local(LocalId),
    Field(Box<Place>, usize),
    Index(Box<Place>, LocalId),
}

impl Place {
    pub fn root_local(&self) -> LocalId {
        match self {
            Place::Local(id) => *id,
            Place::Field(base, _) => base.root_local(),
            Place::Index(base, _) => base.root_local(),
        }
    }

    pub fn collect_uses(&self, set: &mut hashbrown::HashSet<usize>) {
        match self {
            Place::Local(_) => {}
            Place::Field(base, _) => base.collect_uses(set),
            Place::Index(base, idx) => {
                base.collect_uses(set);
                set.insert(idx.0);
            }
        }
    }
}

impl fmt::Display for Place {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Place::Local(id) => write!(f, "{id}"),
            Place::Field(base, idx) => write!(f, "{base}.{idx}"),
            Place::Index(base, idx) => write!(f, "{base}[{idx}]"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    Copy(Place),
    Move(Place),
    Constant(MirConstant),
}

impl Operand {
    pub fn collect_uses(&self, set: &mut hashbrown::HashSet<usize>) {
        match self {
            Operand::Copy(p) | Operand::Move(p) => {
                set.insert(p.root_local().0);
                p.collect_uses(set);
            }
            Operand::Constant(_) => {}
        }
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::Copy(p) => write!(f, "copy {p}"),
            Operand::Move(p) => write!(f, "move {p}"),
            Operand::Constant(c) => write!(f, "const {c}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    FloorDiv,
    Pow,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
    Concat,
}

impl fmt::Display for MirBinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            MirBinaryOp::Add => "+",
            MirBinaryOp::Sub => "-",
            MirBinaryOp::Mul => "*",
            MirBinaryOp::Div => "/",
            MirBinaryOp::Mod => "%",
            MirBinaryOp::FloorDiv => "//",
            MirBinaryOp::Pow => "^",
            MirBinaryOp::BitAnd => "&",
            MirBinaryOp::BitOr => "|",
            MirBinaryOp::BitXor => "^^",
            MirBinaryOp::Shl => "<<",
            MirBinaryOp::Shr => ">>",
            MirBinaryOp::Eq => "==",
            MirBinaryOp::NotEq => "!=",
            MirBinaryOp::Lt => "<",
            MirBinaryOp::LtEq => "<=",
            MirBinaryOp::Gt => ">",
            MirBinaryOp::GtEq => ">=",
            MirBinaryOp::And => "and",
            MirBinaryOp::Or => "or",
            MirBinaryOp::Concat => "..",
        };
        write!(f, "{s}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirUnaryOp {
    Neg,
    Not,
    BitNot,
    Len,
}

impl fmt::Display for MirUnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MirUnaryOp::Neg => write!(f, "-"),
            MirUnaryOp::Not => write!(f, "!"),
            MirUnaryOp::BitNot => write!(f, "~"),
            MirUnaryOp::Len => write!(f, "len "),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Rvalue {
    Use(Operand),
    BinaryOp(MirBinaryOp, Operand, Operand),
    UnaryOp(MirUnaryOp, Operand),
    Cast(Operand, MirType),
    Aggregate(MirType, Vec<Operand>),
    Closure {
        func_name: String,
        captures: Vec<(String, Operand, MirType)>,
    },
}

impl Rvalue {
    pub fn collect_uses(&self, set: &mut hashbrown::HashSet<usize>) {
        match self {
            Rvalue::Use(op) | Rvalue::UnaryOp(_, op) | Rvalue::Cast(op, _) => {
                op.collect_uses(set);
            }
            Rvalue::BinaryOp(_, lhs, rhs) => {
                lhs.collect_uses(set);
                rhs.collect_uses(set);
            }
            Rvalue::Aggregate(_, ops) => {
                for op in ops {
                    op.collect_uses(set);
                }
            }
            Rvalue::Closure { captures, .. } => {
                for (_, op, _) in captures {
                    op.collect_uses(set);
                }
            }
        }
    }
}

impl fmt::Display for Rvalue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Rvalue::Use(op) => write!(f, "{op}"),
            Rvalue::BinaryOp(binop, lhs, rhs) => write!(f, "{lhs} {binop} {rhs}"),
            Rvalue::UnaryOp(unop, op) => write!(f, "{unop}{op}"),
            Rvalue::Cast(op, ty) => write!(f, "{op} as {ty}"),
            Rvalue::Aggregate(ty, operands) => {
                write!(f, "{ty} {{ ")?;
                for (i, op) in operands.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{op}")?;
                }
                write!(f, " }}")
            }
            Rvalue::Closure { func_name, captures } => {
                write!(f, "closure {func_name}(")?;
                for (i, (name, op, ty)) in captures.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{name}: {ty} = {op}")?;
                }
                write!(f, ")")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceKind {
    Gas,
    Memory,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Assign(Place, Rvalue, Span),
    StorageLive(LocalId, Span),
    StorageDead(LocalId, Span),
    ResourceCheck(ResourceKind, u64, Span),
}

impl Statement {
    pub fn collect_uses(&self, set: &mut hashbrown::HashSet<usize>) {
        match self {
            Statement::Assign(place, rval, _) => {
                rval.collect_uses(set);
                match place {
                    Place::Local(_) => {}
                    Place::Field(base, _) => {
                        set.insert(base.root_local().0);
                        base.collect_uses(set);
                    }
                    Place::Index(base, idx) => {
                        set.insert(base.root_local().0);
                        set.insert(idx.0);
                        base.collect_uses(set);
                    }
                }
            }
            _ => {}
        }
    }
}

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Statement::Assign(place, rval, _) => write!(f, "{place} = {rval};"),
            Statement::StorageLive(id, _) => write!(f, "StorageLive({id});"),
            Statement::StorageDead(id, _) => write!(f, "StorageDead({id});"),
            Statement::ResourceCheck(kind, amt, _) => write!(f, "ResourceCheck({kind:?}, {amt});"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Terminator {
    Goto(BlockId, Span),
    Cond {
        cond: Operand,
        then_block: BlockId,
        else_block: BlockId,
        span: Span,
    },
    Switch {
        discr: Operand,
        cases: Vec<(i64, BlockId)>,
        default: BlockId,
        span: Span,
    },
    Call {
        callee: String,
        args: Vec<Operand>,
        destination: Place,
        success: BlockId,
        error: Option<BlockId>,
        span: Span,
    },
    Return(Option<Operand>, Span),
    Raise(Operand, Span),
    Unreachable(Span),
}

impl Terminator {
    pub fn successors(&self) -> Vec<usize> {
        match self {
            Terminator::Goto(target, _) => vec![target.0],
            Terminator::Cond { then_block, else_block, .. } => vec![then_block.0, else_block.0],
            Terminator::Switch { cases, default, .. } => {
                let mut succs = Vec::with_capacity(cases.len() + 1);
                for (_, b) in cases {
                    succs.push(b.0);
                }
                succs.push(default.0);
                succs
            }
            Terminator::Call { success, error, .. } => {
                if let Some(err_b) = error {
                    vec![success.0, err_b.0]
                } else {
                    vec![success.0]
                }
            }
            Terminator::Return(..) | Terminator::Raise(..) | Terminator::Unreachable(..) => vec![],
        }
    }

    pub fn collect_uses(&self, set: &mut hashbrown::HashSet<usize>) {
        match self {
            Terminator::Goto(..) | Terminator::Unreachable(..) => {}
            Terminator::Cond { cond, .. } => {
                cond.collect_uses(set);
            }
            Terminator::Switch { discr, .. } => {
                discr.collect_uses(set);
            }
            Terminator::Call { args, destination, .. } => {
                for arg in args {
                    arg.collect_uses(set);
                }
                if !matches!(destination, Place::Local(_)) {
                    destination.collect_uses(set);
                    set.insert(destination.root_local().0);
                }
            }
            Terminator::Return(Some(op), _) | Terminator::Raise(op, _) => {
                op.collect_uses(set);
            }
            Terminator::Return(None, _) => {}
        }
    }
}

impl fmt::Display for Terminator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Terminator::Goto(target, _) => write!(f, "goto {target};"),
            Terminator::Cond { cond, then_block, else_block, .. } => {
                write!(f, "if {cond} then {then_block} else {else_block};")
            }
            Terminator::Switch { discr, cases, default, .. } => {
                write!(f, "switch {discr} [")?;
                for (val, target) in cases {
                    write!(f, " {val} => {target},")?;
                }
                write!(f, " default => {default} ];")
            }
            Terminator::Call { callee, args, destination, success, error, .. } => {
                write!(f, "{destination} = call {callee}(")?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{a}")?;
                }
                write!(f, ") -> {success}")?;
                if let Some(err) = error {
                    write!(f, " unwind {err}")?;
                }
                write!(f, ";")
            }
            Terminator::Return(val, _) => {
                if let Some(op) = val {
                    write!(f, "return {op};")
                } else {
                    write!(f, "return;")
                }
            }
            Terminator::Raise(err, _) => write!(f, "raise {err};"),
            Terminator::Unreachable(_) => write!(f, "unreachable;"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BasicBlock {
    pub id: BlockId,
    pub statements: Vec<Statement>,
    pub terminator: Option<Terminator>,
    pub is_cleanup: bool,
}

impl BasicBlock {
    pub fn new(id: BlockId) -> Self {
        Self {
            id,
            statements: Vec::new(),
            terminator: None,
            is_cleanup: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionSignature {
    pub name: String,
    pub param_types: Vec<MirType>,
    pub return_type: MirType,
    pub is_public: bool,
    pub effects: Option<crate::ast::ReturnSpec>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirCleanupKind {
    Defer,
    ErrDefer,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MirCleanupEntry {
    pub kind: MirCleanupKind,
    pub block: BlockId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SroaStructRecipe {
    pub orig_local: LocalId,
    pub struct_name: String,
    pub field_names: Vec<String>,
    pub field_locals: Vec<LocalId>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MirFunction {
    pub signature: FunctionSignature,
    pub params: Vec<LocalId>,
    pub return_local: LocalId,
    pub locals: Vec<LocalDecl>,
    pub entry: BlockId,
    pub blocks: Vec<BasicBlock>,
    pub proven_in_bounds: hashbrown::HashMap<usize, Vec<(LocalId, LocalId)>>,
    pub cleanup_blocks: Vec<MirCleanupEntry>,
    pub sroa_decomposed_structs: Vec<SroaStructRecipe>,
}

impl MirFunction {
    pub fn new(signature: FunctionSignature, span: Span) -> Self {
        let mut locals = Vec::new();
        let mut params = Vec::new();
        for (i, p_ty) in signature.param_types.iter().enumerate() {
            let p_id = LocalId(locals.len());
            locals.push(LocalDecl {
                id: p_id,
                name: Some(alloc::format!("arg{i}")),
                ty: p_ty.clone(),
                is_mut: false,
                span,
                is_stack_promoted: false,
                is_param: true,
            });
            params.push(p_id);
        }

        let return_local = LocalId(locals.len());
        let ret_decl = LocalDecl {
            id: return_local,
            name: Some("_return".into()),
            ty: signature.return_type.clone(),
            is_mut: true,
            span,
            is_stack_promoted: false,
            is_param: false,
        };
        locals.push(ret_decl);

        let entry = BlockId(0);
        let blocks = vec![BasicBlock::new(entry)];

        Self {
            signature,
            params,
            return_local,
            locals,
            entry,
            blocks,
            proven_in_bounds: hashbrown::HashMap::new(),
            cleanup_blocks: Vec::new(),
            sroa_decomposed_structs: Vec::new(),
        }
    }

    pub fn add_local(&mut self, name: Option<String>, ty: MirType, is_mut: bool, span: Span) -> LocalId {
        let id = LocalId(self.locals.len());
        self.locals.push(LocalDecl {
            id,
            name,
            ty,
            is_mut,
            span,
            is_stack_promoted: false,
            is_param: false,
        });
        id
    }

    pub fn add_param(&mut self, name: Option<String>, ty: MirType, span: Span) -> LocalId {
        let id = LocalId(self.locals.len());
        self.locals.push(LocalDecl {
            id,
            name,
            ty: ty.clone(),
            is_mut: false,
            span,
            is_stack_promoted: false,
            is_param: true,
        });
        self.params.push(id);
        self.signature.param_types.push(ty);
        id
    }

    pub fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len());
        self.blocks.push(BasicBlock::new(id));
        id
    }

    pub fn block_mut(&mut self, id: BlockId) -> &mut BasicBlock {
        &mut self.blocks[id.0]
    }

    pub fn derive_proven_site_metadata(&self, target_block_id: usize) -> Result<Vec<usize>, String> {
        self.derive_proven_site_metadata_internal(target_block_id, false)
    }

    pub fn derive_proven_site_metadata_at_entry(&self, target_block_id: usize) -> Result<Vec<usize>, String> {
        self.derive_proven_site_metadata_internal(target_block_id, true)
    }

    fn derive_proven_site_metadata_internal(
        &self,
        target_block_id: usize,
        at_entry: bool,
    ) -> Result<Vec<usize>, String> {
        let num_blocks = self.blocks.len();
        if target_block_id >= num_blocks {
            return Err(alloc::format!(
                "Deopt site block {} out of bounds (function '{}' has {} blocks)",
                target_block_id, self.signature.name, num_blocks
            ));
        }

        if self.entry.0 >= num_blocks {
            return Err(alloc::format!(
                "Invalid CFG: entry block {} is out of bounds (function '{}' has {} blocks)",
                self.entry.0, self.signature.name, num_blocks
            ));
        }

        if !at_entry
            && let Some(Terminator::Call { destination, .. }) =
                &self.blocks[target_block_id].terminator
            && !matches!(destination, Place::Local(_))
        {
            return Err(alloc::format!(
                "Cannot derive proven compiler site metadata for deopt site {}: projected call destination {:?} is unsupported at PreTerminator in function '{}'",
                target_block_id, destination, self.signature.name
            ));
        }

        // 1. Strict CFG validation: every block must have a terminator with valid targets
        for (idx, block) in self.blocks.iter().enumerate() {
            let term = match &block.terminator {
                Some(t) => t,
                None => {
                    return Err(alloc::format!(
                        "Invalid CFG: block {} is missing a terminator in function '{}'",
                        idx, self.signature.name
                    ));
                }
            };
            for succ in term.successors() {
                if succ >= num_blocks {
                    return Err(alloc::format!(
                        "Invalid CFG: block {} branches to out-of-bounds block {} in function '{}'",
                        idx, succ, self.signature.name
                    ));
                }
            }
        }

        // 2. Reachability analysis from entry
        let mut reachable = hashbrown::HashSet::new();
        let mut reach_worklist = Vec::new();
        reachable.insert(self.entry.0);
        reach_worklist.push(self.entry.0);
        while let Some(u) = reach_worklist.pop() {
            let term = self.blocks[u].terminator.as_ref().unwrap();
            for succ in term.successors() {
                if reachable.insert(succ) {
                    reach_worklist.push(succ);
                }
            }
        }

        if !reachable.contains(&target_block_id) {
            return Err(alloc::format!(
                "Cannot derive proven compiler site metadata for deopt site {}: block is unreachable in function '{}'",
                target_block_id, self.signature.name
            ));
        }

        // 3. Loop/join/order-independent definite assignment fixed point (Forward dataflow, intersection meet)
        let mut block_inits: Vec<Option<hashbrown::HashSet<usize>>> = vec![None; num_blocks];
        let mut entry_set = hashbrown::HashSet::new();
        for p in &self.params {
            if p.0 < self.locals.len() {
                entry_set.insert(p.0);
            }
        }
        block_inits[self.entry.0] = Some(entry_set);

        let mut da_worklist = Vec::new();
        let mut da_in_worklist = hashbrown::HashSet::new();
        da_worklist.push(self.entry.0);
        da_in_worklist.insert(self.entry.0);

        let max_iterations = num_blocks * (self.locals.len() + 1) * 4 + 100;
        let mut da_iterations = 0;

        while let Some(u) = da_worklist.pop() {
            da_in_worklist.remove(&u);
            da_iterations += 1;
            if da_iterations > max_iterations {
                return Err(alloc::format!(
                    "Definite assignment dataflow did not converge to fixed point in function '{}'",
                    self.signature.name
                ));
            }

            let in_u = match &block_inits[u] {
                Some(s) => s.clone(),
                None => continue,
            };

            let mut out_u = in_u;
            for stmt in &self.blocks[u].statements {
                match stmt {
                    Statement::Assign(Place::Local(loc), _, _) => {
                        out_u.insert(loc.0);
                    }
                    Statement::StorageDead(loc, _) => {
                        out_u.remove(&loc.0);
                    }
                    _ => {}
                }
            }

            let term = self.blocks[u].terminator.as_ref().unwrap();
            match term {
                Terminator::Goto(target, _) => {
                    Self::propagate_da(&mut block_inits, &mut da_worklist, &mut da_in_worklist, target.0, &out_u);
                }
                Terminator::Cond { then_block, else_block, .. } => {
                    Self::propagate_da(&mut block_inits, &mut da_worklist, &mut da_in_worklist, then_block.0, &out_u);
                    Self::propagate_da(&mut block_inits, &mut da_worklist, &mut da_in_worklist, else_block.0, &out_u);
                }
                Terminator::Switch { cases, default, .. } => {
                    for (_, target) in cases {
                        Self::propagate_da(&mut block_inits, &mut da_worklist, &mut da_in_worklist, target.0, &out_u);
                    }
                    Self::propagate_da(&mut block_inits, &mut da_worklist, &mut da_in_worklist, default.0, &out_u);
                }
                Terminator::Call { destination, success, error, .. } => {
                    let mut succ_out = out_u.clone();
                    if let Place::Local(loc) = destination {
                        succ_out.insert(loc.0);
                    }
                    Self::propagate_da(&mut block_inits, &mut da_worklist, &mut da_in_worklist, success.0, &succ_out);
                    if let Some(err_b) = error {
                        Self::propagate_da(&mut block_inits, &mut da_worklist, &mut da_in_worklist, err_b.0, &out_u);
                    }
                }
                _ => {}
            }
        }

        let site_inits = if at_entry {
            match &block_inits[target_block_id] {
                Some(s) => s.clone(),
                None => {
                    return Err(alloc::format!(
                        "Cannot derive proven compiler site metadata for deopt site {}: block has uncomputed inits in function '{}'",
                        target_block_id, self.signature.name
                    ));
                }
            }
        } else {
            let mut cur = match &block_inits[target_block_id] {
                Some(s) => s.clone(),
                None => {
                    return Err(alloc::format!(
                        "Cannot derive proven compiler site metadata for deopt site {}: block has uncomputed inits in function '{}'",
                        target_block_id, self.signature.name
                    ));
                }
            };
            for stmt in &self.blocks[target_block_id].statements {
                match stmt {
                    Statement::Assign(Place::Local(loc), _, _) => {
                        cur.insert(loc.0);
                    }
                    Statement::StorageDead(loc, _) => {
                        cur.remove(&loc.0);
                    }
                    _ => {}
                }
            }
            cur
        };

        // 4. Actual liveness analysis (Backward dataflow, union meet)
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
        for (u, block) in self.blocks.iter().enumerate() {
            let term = block.terminator.as_ref().unwrap();
            for succ in term.successors() {
                preds[succ].push(u);
            }
        }

        let mut live_in: Vec<hashbrown::HashSet<usize>> = vec![hashbrown::HashSet::new(); num_blocks];
        let mut live_at_pre_term: Vec<hashbrown::HashSet<usize>> = vec![hashbrown::HashSet::new(); num_blocks];

        let mut live_worklist: Vec<usize> = (0..num_blocks).collect();
        let mut live_in_worklist: hashbrown::HashSet<usize> = (0..num_blocks).collect();
        let mut live_iterations = 0;

        while let Some(v) = live_worklist.pop() {
            live_in_worklist.remove(&v);
            live_iterations += 1;
            if live_iterations > max_iterations {
                return Err(alloc::format!(
                    "Liveness dataflow did not converge to fixed point in function '{}'",
                    self.signature.name
                ));
            }

            // Compute live_at_pre_term[v] from successors
            let term = self.blocks[v].terminator.as_ref().unwrap();
            let mut pre_term = hashbrown::HashSet::new();
            term.collect_uses(&mut pre_term);

            match term {
                Terminator::Call { destination, success, error, .. } => {
                    for &loc in &live_in[success.0] {
                        if let Place::Local(dst) = destination {
                            if loc != dst.0 {
                                pre_term.insert(loc);
                            }
                        } else {
                            pre_term.insert(loc);
                        }
                    }
                    if let Some(err_b) = error {
                        for &loc in &live_in[err_b.0] {
                            pre_term.insert(loc);
                        }
                    }
                }
                _ => {
                    for succ in term.successors() {
                        for &loc in &live_in[succ] {
                            pre_term.insert(loc);
                        }
                    }
                }
            }
            live_at_pre_term[v] = pre_term;

            // Backward transfer through statements of v:
            let mut cur_live = live_at_pre_term[v].clone();
            for stmt in self.blocks[v].statements.iter().rev() {
                match stmt {
                    Statement::Assign(place, _, _) => {
                        if let Place::Local(loc) = place {
                            cur_live.remove(&loc.0);
                        } else {
                            cur_live.insert(place.root_local().0);
                        }
                        stmt.collect_uses(&mut cur_live);
                    }
                    Statement::StorageDead(loc, _) => {
                        cur_live.remove(&loc.0);
                    }
                    _ => {}
                }
            }

            // Check if live_in[v] changed:
            let mut changed = false;
            for &loc in &cur_live {
                if live_in[v].insert(loc) {
                    changed = true;
                }
            }

            if changed {
                for &p in &preds[v] {
                    if live_in_worklist.insert(p) {
                        live_worklist.push(p);
                    }
                }
            }
        }

        // 5. Authoritative capture metadata at exact site:
        let site_live = if at_entry {
            &live_in[target_block_id]
        } else {
            &live_at_pre_term[target_block_id]
        };

        // 6. Fail-closed validations:
        // A) Every live local at deopt site must be definitely assigned (uninitialized fails closed)
        for &loc_idx in site_live {
            if !site_inits.contains(&loc_idx) {
                return Err(alloc::format!(
                    "Cannot derive proven compiler site metadata for deopt site {}: local {} is live at deopt site but not definitely assigned (uninitialized on some path) in function '{}'",
                    target_block_id, loc_idx, self.signature.name
                ));
            }
        }

        // B) Validate bounds and scalar types for all live locals
        let mut proven_indices: Vec<usize> = site_live.iter().copied().collect();
        proven_indices.sort();

        for &idx in &proven_indices {
            if idx >= self.locals.len() {
                return Err(alloc::format!(
                    "Deopt site metadata derivation error: local index {} exceeds declared locals count {} in function '{}'",
                    idx, self.locals.len(), self.signature.name
                ));
            }
            let ty = &self.locals[idx].ty;
            let is_supported = matches!(
                ty,
                MirType::Int | MirType::Bool | MirType::Float | MirType::Unit | MirType::String
            ) || matches!(
                ty,
                MirType::Array(inner)
                    if matches!(&**inner, MirType::Int | MirType::Bool | MirType::Float)
            ) || matches!(ty, MirType::Struct(name) if !name.is_empty())
                || matches!(
                    ty,
                    MirType::Option(inner)
                        if matches!(&**inner, MirType::Struct(name) if !name.is_empty())
                );
            if !is_supported {
                return Err(alloc::format!(
                    "Cannot derive proven compiler site metadata for deopt site {}: local {} has unsupported deopt type {:?} (Unsupported non-scalar type)",
                    target_block_id, idx, ty
                ));
            }
        }

        Ok(proven_indices)
    }

    fn propagate_da(
        block_inits: &mut [Option<hashbrown::HashSet<usize>>],
        worklist: &mut Vec<usize>,
        in_worklist: &mut hashbrown::HashSet<usize>,
        target: usize,
        out: &hashbrown::HashSet<usize>,
    ) {
        if target >= block_inits.len() {
            return;
        }
        match &mut block_inits[target] {
            None => {
                block_inits[target] = Some(out.clone());
                if in_worklist.insert(target) {
                    worklist.push(target);
                }
            }
            Some(existing) => {
                let old_len = existing.len();
                existing.retain(|id| out.contains(id));
                if existing.len() != old_len {
                    if in_worklist.insert(target) {
                        worklist.push(target);
                    }
                }
            }
        }
    }

    /// Counts total reads of every local across all statements and terminators in the function.
    /// Used by `NG-W14-8` to prove intermediate range-check booleans are single-use dead temps.
    pub fn compute_local_read_counts(&self) -> Vec<usize> {
        let mut counts = vec![0usize; self.locals.len()];
        let count_place_reads = |place: &Place, counts: &mut [usize]| {
            fn walk(p: &Place, counts: &mut [usize], is_root: bool) {
                match p {
                    Place::Local(loc) => {
                        if is_root && loc.0 < counts.len() {
                            counts[loc.0] += 1;
                        }
                    }
                    Place::Field(base, _) => walk(base, counts, true),
                    Place::Index(base, idx) => {
                        walk(base, counts, true);
                        if idx.0 < counts.len() {
                            counts[idx.0] += 1;
                        }
                    }
                }
            }
            walk(place, counts, true);
        };
        let count_operand = |op: &Operand, counts: &mut [usize]| match op {
            Operand::Copy(p) | Operand::Move(p) => count_place_reads(p, counts),
            Operand::Constant(_) => {}
        };
        let count_lhs_place = |place: &Place, counts: &mut [usize]| match place {
            Place::Local(_) => {}
            Place::Field(base, _) => count_place_reads(base, counts),
            Place::Index(base, idx) => {
                count_place_reads(base, counts);
                if idx.0 < counts.len() {
                    counts[idx.0] += 1;
                }
            }
        };

        for block in &self.blocks {
            for stmt in &block.statements {
                if let Statement::Assign(lhs, rval, _) = stmt {
                    count_lhs_place(lhs, &mut counts);
                    match rval {
                        Rvalue::Use(op) | Rvalue::UnaryOp(_, op) | Rvalue::Cast(op, _) => {
                            count_operand(op, &mut counts);
                        }
                        Rvalue::BinaryOp(_, l, r) => {
                            count_operand(l, &mut counts);
                            count_operand(r, &mut counts);
                        }
                        Rvalue::Aggregate(_, ops) => {
                            for op in ops {
                                count_operand(op, &mut counts);
                            }
                        }
                        Rvalue::Closure { captures, .. } => {
                            for (_, op, _) in captures {
                                count_operand(op, &mut counts);
                            }
                        }
                    }
                }
            }
            if let Some(term) = &block.terminator {
                match term {
                    Terminator::Goto(..) | Terminator::Unreachable(..) => {}
                    Terminator::Cond { cond, .. } => count_operand(cond, &mut counts),
                    Terminator::Switch { discr, .. } => count_operand(discr, &mut counts),
                    Terminator::Call { args, destination, .. } => {
                        for arg in args {
                            count_operand(arg, &mut counts);
                        }
                        count_lhs_place(destination, &mut counts);
                    }
                    Terminator::Return(Some(op), _) | Terminator::Raise(op, _) => {
                        count_operand(op, &mut counts);
                    }
                    Terminator::Return(None, _) => {}
                }
            }
        }
        counts
    }

    /// Detects canonical `(idx >= 0) && (idx < len)` sequences within `block_idx` where `len` is
    /// proven `>= 0` and both intermediate bool temporaries are single-use dead temps, allowing
    /// Cranelift (`NG-W14-8`) to emit a single unsigned comparison `(idx as u64) < (len as u64)`.
    pub fn detect_unsigned_range_checks_in_block(
        &self,
        block_idx: usize,
        local_read_counts: &[usize],
        excluded_locals: &hashbrown::HashSet<usize>,
        non_neg_in: &hashbrown::HashSet<usize>,
    ) -> BlockUnsignedRangeFusion {
        let mut fusion = BlockUnsignedRangeFusion {
            skip_stmts: hashbrown::HashSet::new(),
            fused_checks: hashbrown::HashMap::new(),
        };
        let Some(block) = self.blocks.get(block_idx) else {
            return fusion;
        };

        let mut non_neg_locals = non_neg_in.clone();
        let mut ge_zero_defs: hashbrown::HashMap<usize, (usize, LocalId)> =
            hashbrown::HashMap::new();
        let mut lt_len_defs: hashbrown::HashMap<usize, (usize, LocalId, Operand)> =
            hashbrown::HashMap::new();

        for (stmt_idx, stmt) in block.statements.iter().enumerate() {
            match stmt {
                Statement::StorageDead(loc, _) => {
                    non_neg_locals.remove(&loc.0);
                    ge_zero_defs.remove(&loc.0);
                    ge_zero_defs.retain(|_, (_, idx_l)| idx_l.0 != loc.0);
                    lt_len_defs.remove(&loc.0);
                    lt_len_defs.retain(|_, (_, idx_l, len_op)| {
                        idx_l.0 != loc.0 && operand_local_id(len_op) != Some(*loc)
                    });
                }
                Statement::Assign(Place::Local(dst), rval, _) => {
                    let mut matched_fusion = false;
                    if let Rvalue::BinaryOp(MirBinaryOp::And, op_a, op_b) = rval
                        && let (Some(la), Some(lb)) =
                            (operand_local_id(op_a), operand_local_id(op_b))
                        && la != lb
                    {
                        let candidates = [(la, lb), (lb, la)];
                        for (ge_loc, lt_loc) in candidates {
                            let Some(&(s_ge, idx_ge)) = ge_zero_defs.get(&ge_loc.0) else {
                                continue;
                            };
                            let Some((s_lt, idx_lt, len_op)) = lt_len_defs.get(&lt_loc.0).cloned()
                            else {
                                continue;
                            };
                            if idx_ge != idx_lt {
                                continue;
                            }
                            let idx_is_int = self
                                .locals
                                .get(idx_ge.0)
                                .is_some_and(|d| d.ty == MirType::Int);
                            let ge_ok = self
                                .locals
                                .get(ge_loc.0)
                                .is_some_and(|d| !d.is_param && d.ty == MirType::Bool)
                                && !excluded_locals.contains(&ge_loc.0)
                                && local_read_counts.get(ge_loc.0).copied().unwrap_or(0) == 1;
                            let lt_ok = self
                                .locals
                                .get(lt_loc.0)
                                .is_some_and(|d| !d.is_param && d.ty == MirType::Bool)
                                && !excluded_locals.contains(&lt_loc.0)
                                && local_read_counts.get(lt_loc.0).copied().unwrap_or(0) == 1;
                            let len_non_neg = match &len_op {
                                Operand::Constant(MirConstant::Int(c)) => *c >= 0,
                                Operand::Copy(Place::Local(ll))
                                | Operand::Move(Place::Local(ll)) => {
                                    self.locals
                                        .get(ll.0)
                                        .is_some_and(|d| d.ty == MirType::Int)
                                        && non_neg_locals.contains(&ll.0)
                                }
                                _ => false,
                            };
                            if idx_is_int && ge_ok && lt_ok && len_non_neg {
                                fusion.skip_stmts.insert(s_ge);
                                fusion.skip_stmts.insert(s_lt);
                                fusion.fused_checks.insert(
                                    stmt_idx,
                                    UnsignedRangeCheck {
                                        dst: *dst,
                                        idx_local: idx_ge,
                                        len_op,
                                    },
                                );
                                ge_zero_defs.remove(&ge_loc.0);
                                lt_len_defs.remove(&lt_loc.0);
                                matched_fusion = true;
                                break;
                            }
                        }
                    }

                    // Invalidate any prior defs referencing `dst` before recording new def of `dst`
                    ge_zero_defs.remove(&dst.0);
                    ge_zero_defs.retain(|_, (_, idx_l)| idx_l.0 != dst.0);
                    lt_len_defs.remove(&dst.0);
                    lt_len_defs.retain(|_, (_, idx_l, len_op)| {
                        idx_l.0 != dst.0 && operand_local_id(len_op) != Some(*dst)
                    });

                    if !matched_fusion {
                        if let Some(idx_loc) = match_ge_zero_rvalue(rval) {
                            ge_zero_defs.insert(dst.0, (stmt_idx, idx_loc));
                        } else if let Some((idx_loc, len_op)) = match_lt_bound_rvalue(rval) {
                            lt_len_defs.insert(dst.0, (stmt_idx, idx_loc, len_op));
                        }
                    }

                    match rval {
                        Rvalue::UnaryOp(MirUnaryOp::Len, _) => {
                            non_neg_locals.insert(dst.0);
                        }
                        Rvalue::Use(Operand::Constant(MirConstant::Int(c))) if *c >= 0 => {
                            non_neg_locals.insert(dst.0);
                        }
                        Rvalue::Use(
                            Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)),
                        ) if non_neg_locals.contains(&src.0) => {
                            non_neg_locals.insert(dst.0);
                        }
                        _ => {
                            non_neg_locals.remove(&dst.0);
                        }
                    }
                }
                _ => {}
            }
        }

        fusion
    }

    /// Conservative loop-induction bounds proof (`NG-W14-8`):
    ///
    /// Computes:
    /// 1. `non_neg_locals_in[b]`: locals of type `MirType::Int` mathematically proven `>= 0` on
    ///    every path into block `b` (feeds unsigned range-check fusion).
    /// 2. `proven_in_bounds[b]`: `(arr_local, idx_local)` pairs proven `0 <= idx < len(arr)` at
    ///    entry of pre-step loop body block `b` in canonical `while j < len` / `while j <= N`
    ///    loops where `arr` is non-aliased/non-shrinking, `j` starts `>= 0`, `step > 0` is
    ///    loop-invariant, and `max_j_in_body + step_hi <= i64::MAX` (checked in `i128`).
    ///
    /// Fails closed on any call inside the loop, aliasing/length-shrinking mutation, unknown step,
    /// negative/unknown start, off-by-one `<= len`, or potential signed `i64` overflow.
    pub fn compute_loop_induction_bounds_analysis(
        &self,
        entry_block: usize,
        excluded_locals: &hashbrown::HashSet<usize>,
        allow_induction_proofs: bool,
    ) -> LoopBoundsAnalysis {
        let num_blocks = self.blocks.len();
        let empty = LoopBoundsAnalysis {
            proven_in_bounds: hashbrown::HashMap::new(),
            non_neg_locals_in: vec![hashbrown::HashSet::new(); num_blocks],
        };
        if num_blocks == 0 || entry_block >= num_blocks {
            return empty;
        }

        // Phase A1: Reachability & predecessors from entry_block
        let mut reachable = hashbrown::HashSet::new();
        let mut queue = VecDeque::new();
        reachable.insert(entry_block);
        queue.push_back(entry_block);
        while let Some(b) = queue.pop_front() {
            if let Some(term) = &self.blocks[b].terminator {
                for succ in term.successors() {
                    if succ < num_blocks && reachable.insert(succ) {
                        queue.push_back(succ);
                    }
                }
            }
        }
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
        for &b in &reachable {
            if let Some(term) = &self.blocks[b].terminator {
                for succ in term.successors() {
                    if succ < num_blocks && reachable.contains(&succ) && !preds[succ].contains(&b) {
                        preds[succ].push(b);
                    }
                }
            }
        }

        // Phase A2: Exact dominators over reachable blocks
        let mut dom: Vec<hashbrown::HashSet<usize>> = (0..num_blocks)
            .map(|b| {
                if b == entry_block {
                    let mut s = hashbrown::HashSet::new();
                    s.insert(entry_block);
                    s
                } else if reachable.contains(&b) {
                    reachable.clone()
                } else {
                    hashbrown::HashSet::new()
                }
            })
            .collect();
        loop {
            let mut changed = false;
            for b in 0..num_blocks {
                if b == entry_block || !reachable.contains(&b) || preds[b].is_empty() {
                    continue;
                }
                let mut iter = preds[b].iter();
                let first = *iter.next().unwrap();
                let mut new_dom = dom[first].clone();
                for &p in iter {
                    new_dom.retain(|x| dom[p].contains(x));
                }
                new_dom.insert(b);
                if new_dom != dom[b] {
                    dom[b] = new_dom;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // Phase A3: Natural loops from dominating CFG backedges
        let mut nat_loops: hashbrown::HashMap<usize, hashbrown::HashSet<usize>> =
            hashbrown::HashMap::new();
        for &u in &reachable {
            if let Some(term) = &self.blocks[u].terminator {
                for v in term.successors() {
                    if v < num_blocks && reachable.contains(&v) && dom[u].contains(&v) {
                        let body = nat_loops.entry(v).or_default();
                        body.insert(v);
                        if body.insert(u) {
                            let mut stack = vec![u];
                            while let Some(cur) = stack.pop() {
                                for &p in &preds[cur] {
                                    if body.insert(p) {
                                        stack.push(p);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Filter to reducible single-entry natural loops with Cond header
        let mut reducible_loops: Vec<NaturalLoopMeta> = Vec::new();
        for (hdr, body) in nat_loops {
            if !body.iter().all(|&b| dom[b].contains(&hdr)) {
                continue;
            }
            if !body
                .iter()
                .all(|&b| b == hdr || preds[b].iter().all(|p| body.contains(p)))
            {
                continue;
            }
            let Some(Terminator::Cond {
                cond,
                then_block,
                else_block,
                ..
            }) = &self.blocks[hdr].terminator
            else {
                continue;
            };
            let (body_entry, loop_exit, then_is_body) = if body.contains(&then_block.0)
                && then_block.0 != hdr
                && !body.contains(&else_block.0)
            {
                (then_block.0, else_block.0, true)
            } else if body.contains(&else_block.0)
                && else_block.0 != hdr
                && !body.contains(&then_block.0)
            {
                (else_block.0, then_block.0, false)
            } else {
                continue;
            };
            reducible_loops.push(NaturalLoopMeta {
                hdr,
                body_entry,
                loop_exit,
                then_is_body,
                cond: cond.clone(),
                body,
            });
        }
        // Outermost-first (deterministic tie-break by hdr)
        reducible_loops.sort_by(|a, b| b.body.len().cmp(&a.body.len()).then_with(|| a.hdr.cmp(&b.hdr)));
        let mut loop_assigned_by_hdr: hashbrown::HashMap<usize, hashbrown::HashSet<usize>> =
            hashbrown::HashMap::new();
        for lp in &reducible_loops {
            let mut assigned = hashbrown::HashSet::new();
            for &b in &lp.body {
                for stmt in &self.blocks[b].statements {
                    if let Statement::Assign(Place::Local(d), _, _) = stmt {
                        assigned.insert(d.0);
                    }
                }
                if let Some(Terminator::Call {
                    destination: Place::Local(d),
                    ..
                }) = &self.blocks[b].terminator
                {
                    assigned.insert(d.0);
                }
            }
            loop_assigned_by_hdr.insert(lp.hdr, assigned);
        }

        // Phase A4: Identify unaliased locally-allocated arrays (including single-def, single-read,
        // same-block `tmp = Aggregate(Array, ...); arr = Move/Copy(tmp)` ownership transfer).
        let unaliased_local_arrays =
            self.compute_unaliased_local_arrays(excluded_locals);

        // Phase B: Discover canonical array-building push loops (`while i <= N` / `while i < N`)
        let mut push_loop_summaries: hashbrown::HashMap<usize, PushLoopSummary> =
            hashbrown::HashMap::new();
        for lp in &reducible_loops {
            if let Some(summary) = self.match_canonical_push_loop(
                lp,
                &preds,
                &unaliased_local_arrays,
                excluded_locals,
            ) {
                push_loop_summaries.insert(lp.hdr, summary);
            }
        }

        // Phase C: Forward must-dataflow (intervals with finite-height widening + LenFact intersection)
        let mut facts_in: Vec<Option<BlockFacts>> = vec![None; num_blocks];
        facts_in[entry_block] = Some(BlockFacts::default());

        let mut worklist: VecDeque<usize> = VecDeque::new();
        let mut in_worklist: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
        worklist.push_back(entry_block);
        in_worklist.insert(entry_block);

        let mut merge_counts = vec![0usize; num_blocks];
        let max_iters = num_blocks.saturating_mul(self.locals.len() + 4).saturating_mul(6) + 256;
        let mut iters = 0usize;

        while let Some(u) = worklist.pop_front() {
            in_worklist.remove(&u);
            iters += 1;
            if iters > max_iters {
                break;
            }
            let Some(in_facts) = facts_in[u].clone() else {
                continue;
            };
            let out_facts = self.transfer_block_facts(
                u,
                &in_facts,
                &unaliased_local_arrays,
                excluded_locals,
            );
            let Some(term) = &self.blocks[u].terminator else {
                continue;
            };
            match term {
                Terminator::Cond {
                    cond,
                    then_block,
                    else_block,
                    ..
                } => {
                    let cmp_info = match_block_cond_cmp(&self.blocks[u], cond);
                    for (succ, branch_taken) in [(then_block.0, true), (else_block.0, false)] {
                        if succ >= num_blocks || !reachable.contains(&succ) {
                            continue;
                        }
                        let mut edge_facts = out_facts.clone();
                        if let Some(ref cmp) = cmp_info {
                            refine_facts_for_branch(&mut edge_facts, cmp, branch_taken);
                        }
                        if let Some(ps) = push_loop_summaries.get(&u)
                            && succ == ps.exit
                        {
                            self.try_apply_push_loop_exit_fact(
                                ps,
                                &preds,
                                &facts_in,
                                &unaliased_local_arrays,
                                excluded_locals,
                                &mut edge_facts,
                            );
                        }
                        propagate_block_facts(
                            succ,
                            &edge_facts,
                            loop_assigned_by_hdr.get(&succ),
                            &mut facts_in,
                            &mut merge_counts,
                            &mut worklist,
                            &mut in_worklist,
                        );
                    }
                }
                _ => {
                    for succ in term.successors() {
                        if succ >= num_blocks || !reachable.contains(&succ) {
                            continue;
                        }
                        propagate_block_facts(
                            succ,
                            &out_facts,
                            loop_assigned_by_hdr.get(&succ),
                            &mut facts_in,
                            &mut merge_counts,
                            &mut worklist,
                            &mut in_worklist,
                        );
                    }
                }
            }
        }

        // Collect `non_neg_locals_in` from converged intervals
        let mut non_neg_locals_in: Vec<hashbrown::HashSet<usize>> =
            vec![hashbrown::HashSet::new(); num_blocks];
        for b in 0..num_blocks {
            if let Some(ref f) = facts_in[b] {
                for (&loc_idx, &iv) in &f.intervals {
                    if iv.lo >= 0
                        && self
                            .locals
                            .get(loc_idx)
                            .is_some_and(|d| d.ty == MirType::Int)
                    {
                        non_neg_locals_in[b].insert(loc_idx);
                    }
                }
            }
        }

        if !allow_induction_proofs {
            return LoopBoundsAnalysis {
                proven_in_bounds: hashbrown::HashMap::new(),
                non_neg_locals_in,
            };
        }

        // Phase D: Conservative loop-induction bounds proof per reducible loop
        let mut proven_in_bounds: hashbrown::HashMap<usize, Vec<(LocalId, LocalId)>> =
            hashbrown::HashMap::new();

        for lp in &reducible_loops {
            // H1: ZERO Terminator::Call and ZERO Rvalue::Closure anywhere in the loop body
            let has_call_or_closure = lp.body.iter().any(|&b| {
                let blk = &self.blocks[b];
                if matches!(blk.terminator, Some(Terminator::Call { .. })) {
                    return true;
                }
                blk.statements.iter().any(|s| {
                    matches!(s, Statement::Assign(_, Rvalue::Closure { .. }, _))
                })
            });
            if has_call_or_closure {
                continue;
            }

            // H3a: Canonical header comparison `j < bound` or `j <= bound`
            let Some(hdr_cmp) = match_block_cond_cmp(&self.blocks[lp.hdr], &lp.cond) else {
                continue;
            };
            let effective_op = if lp.then_is_body {
                hdr_cmp.op
            } else {
                invert_cmp_op(hdr_cmp.op)
            };

            // Collect all locals assigned anywhere in `lp.body` (and written index roots)
            let mut assigned_in_loop: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
            let mut written_index_roots: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
            let mut has_non_local_index_write = false;
            let mut idx_access_arrays: hashbrown::HashMap<LocalId, Vec<LocalId>> =
                hashbrown::HashMap::new();

            for &b in &lp.body {
                for stmt in &self.blocks[b].statements {
                    if let Statement::Assign(lhs, rval, _) = stmt {
                        if let Place::Local(dst) = lhs {
                            assigned_in_loop.insert(dst.0);
                        }
                        if let Place::Index(base, idx) = lhs {
                            if let Place::Local(arr_loc) = base.as_ref() {
                                written_index_roots.insert(arr_loc.0);
                                let entry = idx_access_arrays.entry(*idx).or_default();
                                if !entry.contains(arr_loc) {
                                    entry.push(*arr_loc);
                                }
                            } else {
                                has_non_local_index_write = true;
                            }
                        }
                        collect_rvalue_index_accesses(rval, &mut idx_access_arrays);
                    }
                }
            }

            // Normalize comparison to `(j_loc, bound_op, is_inclusive)`
            let Some((j_loc, bound_op, is_inclusive)) = normalize_upper_bound_cmp(
                effective_op,
                &hdr_cmp.lhs,
                &hdr_cmp.rhs,
                &self.locals,
                excluded_locals,
                &assigned_in_loop,
            ) else {
                continue;
            };

            let Some(candidate_arrays) = idx_access_arrays.get(&j_loc) else {
                continue;
            };

            // Verify `j_loc` is NOT assigned in `hdr` and has EXACTLY ONE assignment in `body`
            if block_assigns_local(&self.blocks[lp.hdr], j_loc) {
                continue;
            }
            let mut j_assigns: Vec<(usize, usize)> = Vec::new();
            for &b in &lp.body {
                for (s_idx, stmt) in self.blocks[b].statements.iter().enumerate() {
                    if matches!(stmt, Statement::Assign(Place::Local(d), _, _) if *d == j_loc) {
                        j_assigns.push((b, s_idx));
                    }
                }
            }
            if j_assigns.len() != 1 {
                continue;
            }
            let (step_bb, step_stmt_idx) = j_assigns[0];

            // Ensure `step_bb` is NOT inside any strictly nested inner loop of `lp`
            let step_in_inner_loop = reducible_loops.iter().any(|other| {
                other.hdr != lp.hdr
                    && lp.body.contains(&other.hdr)
                    && other.body.len() < lp.body.len()
                    && other.body.contains(&step_bb)
            });
            if step_in_inner_loop {
                continue;
            }

            // H2: Extract canonical positive step `j = j + step` (direct or 2-stmt temp)
            let Some(step_op) =
                extract_induction_step(&self.blocks[step_bb], step_stmt_idx, j_loc)
            else {
                continue;
            };
            let Some(hdr_facts) = facts_in[lp.hdr].as_ref() else {
                continue;
            };
            let step_iv = match &step_op {
                Operand::Constant(MirConstant::Int(c)) => IntInterval { lo: *c, hi: *c },
                Operand::Copy(Place::Local(s_loc)) | Operand::Move(Place::Local(s_loc)) => {
                    if *s_loc == j_loc
                        || excluded_locals.contains(&s_loc.0)
                        || assigned_in_loop.contains(&s_loc.0)
                        || !self
                            .locals
                            .get(s_loc.0)
                            .is_some_and(|d| d.ty == MirType::Int)
                    {
                        continue;
                    }
                    hdr_facts.interval(*s_loc)
                }
                _ => continue,
            };
            if step_iv.lo < 1 {
                continue;
            }

            // H4a: Prove `j_init >= 0` across ALL reachable non-backedge predecessors of `lp.hdr`
            if lp.hdr == entry_block {
                continue;
            }
            let non_backedge_preds: Vec<usize> = preds[lp.hdr]
                .iter()
                .copied()
                .filter(|p| !lp.body.contains(p))
                .collect();
            if non_backedge_preds.is_empty() {
                continue;
            }
            let mut all_entries_non_neg = true;
            for &p in &non_backedge_preds {
                let Some(p_in) = facts_in[p].as_ref() else {
                    all_entries_non_neg = false;
                    break;
                };
                let mut edge_f = self.transfer_block_facts(
                    p,
                    p_in,
                    &unaliased_local_arrays,
                    excluded_locals,
                );
                if let Some(Terminator::Cond {
                    cond,
                    then_block,
                    else_block,
                    ..
                }) = &self.blocks[p].terminator
                    && let Some(cmp) = match_block_cond_cmp(&self.blocks[p], cond)
                {
                    if then_block.0 == lp.hdr && else_block.0 != lp.hdr {
                        refine_facts_for_branch(&mut edge_f, &cmp, true);
                    } else if else_block.0 == lp.hdr && then_block.0 != lp.hdr {
                        refine_facts_for_branch(&mut edge_f, &cmp, false);
                    }
                }
                if edge_f.interval(j_loc).lo < 0 {
                    all_entries_non_neg = false;
                    break;
                }
            }
            if !all_entries_non_neg {
                continue;
            }

            // H4b: Prove NO signed `i64` overflow on `j + step`
            let bound_hi = match &bound_op {
                Operand::Constant(MirConstant::Int(c)) => *c,
                Operand::Copy(Place::Local(b_loc)) | Operand::Move(Place::Local(b_loc)) => {
                    hdr_facts.interval(*b_loc).hi
                }
                _ => continue,
            };
            let max_j_in_body = if is_inclusive {
                bound_hi
            } else if bound_hi == i64::MIN {
                continue;
            } else {
                bound_hi - 1
            };
            if max_j_in_body < 0 {
                continue;
            }
            if (max_j_in_body as i128) + (step_iv.hi as i128) > (i64::MAX as i128) {
                continue;
            }

            // Compute post-step blocks inside `lp.body \ {lp.hdr}` (reachable from successors of `step_bb`)
            let mut post_step: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
            let mut ps_queue: VecDeque<usize> = VecDeque::new();
            if let Some(term) = &self.blocks[step_bb].terminator {
                for succ in term.successors() {
                    if succ != lp.hdr && lp.body.contains(&succ) && post_step.insert(succ) {
                        ps_queue.push_back(succ);
                    }
                }
            }
            while let Some(cur) = ps_queue.pop_front() {
                if let Some(term) = &self.blocks[cur].terminator {
                    for succ in term.successors() {
                        if succ != lp.hdr && lp.body.contains(&succ) && post_step.insert(succ) {
                            ps_queue.push_back(succ);
                        }
                    }
                }
            }

            // H3b + H1b: Check each candidate array accessed by `j_loc`
            for &arr_loc in candidate_arrays {
                if arr_loc == j_loc
                    || excluded_locals.contains(&arr_loc.0)
                    || assigned_in_loop.contains(&arr_loc.0)
                    || !matches!(
                        self.locals.get(arr_loc.0).map(|d| &d.ty),
                        Some(MirType::Array(_))
                    )
                {
                    continue;
                }
                // Shifted-RawWords alias safety for non-unaliased/param arrays
                if !unaliased_local_arrays.contains(&arr_loc.0) {
                    let other_array_written = has_non_local_index_write
                        || written_index_roots.iter().any(|&w| w != arr_loc.0);
                    if other_array_written {
                        continue;
                    }
                }
                let Some(len_facts) = hdr_facts.len_rels.get(&arr_loc.0) else {
                    continue;
                };
                if !len_facts_prove_bound(len_facts, &bound_op, is_inclusive, hdr_facts) {
                    continue;
                }

                // Feed `proven_in_bounds` for all pre-step blocks in `lp.body \ {lp.hdr}`
                for &b in &lp.body {
                    if b == lp.hdr || post_step.contains(&b) {
                        continue;
                    }
                    let entry = proven_in_bounds.entry(b).or_default();
                    if !entry.contains(&(arr_loc, j_loc)) {
                        entry.push((arr_loc, j_loc));
                    }
                }
            }
        }

        LoopBoundsAnalysis {
            proven_in_bounds,
            non_neg_locals_in,
        }
    }

    fn compute_unaliased_local_arrays(
        &self,
        excluded_locals: &hashbrown::HashSet<usize>,
    ) -> hashbrown::HashSet<usize> {
        let n_locals = self.locals.len();
        let mut def_count = vec![0usize; n_locals];
        let mut agg_def_only = vec![true; n_locals];
        let mut single_agg_def_block: Vec<Option<(usize, usize)>> = vec![None; n_locals];
        let mut copy_from: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); n_locals];
        let mut read_count = vec![0usize; n_locals];
        let mut escaped = vec![false; n_locals];

        for &p in &self.params {
            if p.0 < n_locals {
                escaped[p.0] = true;
                agg_def_only[p.0] = false;
            }
        }
        for &ex in excluded_locals {
            if ex < n_locals {
                escaped[ex] = true;
                agg_def_only[ex] = false;
            }
        }

        for (b_idx, block) in self.blocks.iter().enumerate() {
            for (s_idx, stmt) in block.statements.iter().enumerate() {
                if let Statement::Assign(lhs, rval, _) = stmt {
                    match lhs {
                        Place::Local(dst) => {
                            if dst.0 < n_locals {
                                def_count[dst.0] += 1;
                                match rval {
                                    Rvalue::Aggregate(MirType::Array(_), _) => {
                                        if def_count[dst.0] == 1 {
                                            single_agg_def_block[dst.0] = Some((b_idx, s_idx));
                                        } else {
                                            single_agg_def_block[dst.0] = None;
                                        }
                                    }
                                    Rvalue::Use(
                                        Operand::Copy(Place::Local(src))
                                        | Operand::Move(Place::Local(src)),
                                    ) => {
                                        copy_from[dst.0].push((b_idx, s_idx, src.0));
                                        if src.0 < n_locals {
                                            read_count[src.0] += 1;
                                        }
                                        // Do not mark `src` escaped yet; verified below if single-transfer temp.
                                        continue;
                                    }
                                    _ => {
                                        agg_def_only[dst.0] = false;
                                    }
                                }
                            }
                        }
                        Place::Index(base, idx) => {
                            if let Place::Local(b_loc) = base.as_ref() {
                                if b_loc.0 < n_locals {
                                    read_count[b_loc.0] += 1;
                                }
                            } else {
                                mark_place_escaped(base, &mut escaped, &mut read_count);
                            }
                            if idx.0 < n_locals {
                                read_count[idx.0] += 1;
                                escaped[idx.0] = true;
                            }
                        }
                        Place::Field(base, _) => {
                            mark_place_escaped(base, &mut escaped, &mut read_count);
                        }
                    }

                    // Inspect RHS uses
                    match rval {
                        Rvalue::UnaryOp(
                            MirUnaryOp::Len,
                            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)),
                        ) => {
                            if loc.0 < n_locals {
                                read_count[loc.0] += 1;
                            }
                        }
                        Rvalue::Use(
                            Operand::Copy(Place::Index(base, idx))
                            | Operand::Move(Place::Index(base, idx)),
                        ) => {
                            if let Place::Local(b_loc) = base.as_ref() {
                                if b_loc.0 < n_locals {
                                    read_count[b_loc.0] += 1;
                                }
                            } else {
                                mark_place_escaped(base, &mut escaped, &mut read_count);
                            }
                            if idx.0 < n_locals {
                                read_count[idx.0] += 1;
                                escaped[idx.0] = true;
                            }
                        }
                        _ => {
                            mark_rvalue_escaped(rval, &mut escaped, &mut read_count);
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
                        if let Place::Local(dst) = destination {
                            if dst.0 < n_locals {
                                def_count[dst.0] += 1;
                                agg_def_only[dst.0] = false;
                            }
                        } else {
                            mark_place_escaped(destination, &mut escaped, &mut read_count);
                        }
                        let is_safe_array_builtin = matches!(
                            callee.as_str(),
                            "array.push" | "table.push" | "array.len" | "table.len"
                        );
                        for (arg_i, arg) in args.iter().enumerate() {
                            if is_safe_array_builtin
                                && arg_i == 0
                                && let Operand::Copy(Place::Local(loc))
                                    | Operand::Move(Place::Local(loc)) = arg
                            {
                                if loc.0 < n_locals {
                                    read_count[loc.0] += 1;
                                }
                            } else {
                                mark_operand_escaped(arg, &mut escaped, &mut read_count);
                            }
                        }
                    }
                    Terminator::Return(Some(op), _) => {
                        // Returning a local exits the function; only count read, not in-fn alias
                        if let Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) =
                            op
                        {
                            if loc.0 < n_locals {
                                read_count[loc.0] += 1;
                            }
                        } else {
                            mark_operand_escaped(op, &mut escaped, &mut read_count);
                        }
                    }
                    Terminator::Cond { cond, .. } => {
                        mark_operand_escaped(cond, &mut escaped, &mut read_count);
                    }
                    Terminator::Switch { discr, .. } => {
                        mark_operand_escaped(discr, &mut escaped, &mut read_count);
                    }
                    Terminator::Raise(op, _) => {
                        mark_operand_escaped(op, &mut escaped, &mut read_count);
                    }
                    Terminator::Goto(..)
                    | Terminator::Return(None, _)
                    | Terminator::Unreachable(..) => {}
                }
            }
        }

        // Validate copy transfers: a copy `dst = move/copy src` preserves unique ownership ONLY if
        // `src` is a fresh single-def, single-read, same-block Aggregate(Array, _) temporary.
        let mut valid_unaliased = hashbrown::HashSet::new();
        for dst in 0..n_locals {
            if escaped[dst]
                || def_count[dst] == 0
                || !agg_def_only[dst]
                || !matches!(self.locals[dst].ty, MirType::Array(_))
            {
                continue;
            }
            let mut all_copies_are_fresh_transfers = true;
            for &(b_idx, s_idx, src) in &copy_from[dst] {
                let is_fresh_transfer = src < n_locals
                    && src != dst
                    && !escaped[src]
                    && def_count[src] == 1
                    && read_count[src] == 1
                    && matches!(single_agg_def_block[src], Some((sb, ss)) if sb == b_idx && ss < s_idx);
                if !is_fresh_transfer {
                    all_copies_are_fresh_transfers = false;
                    break;
                }
            }
            if all_copies_are_fresh_transfers {
                valid_unaliased.insert(dst);
            }
        }

        // Also ensure no `valid_unaliased` local was copied into ANY OTHER local (making an alias!)
        let mut copied_out_to_other = vec![false; n_locals];
        for dst in 0..n_locals {
            for &(_, _, src) in &copy_from[dst] {
                if src < n_locals && src != dst {
                    copied_out_to_other[src] = true;
                }
            }
        }
        valid_unaliased.retain(|&loc| !copied_out_to_other[loc]);
        valid_unaliased
    }

    fn match_canonical_push_loop(
        &self,
        lp: &NaturalLoopMeta,
        preds: &[Vec<usize>],
        unaliased_local_arrays: &hashbrown::HashSet<usize>,
        excluded_locals: &hashbrown::HashSet<usize>,
    ) -> Option<PushLoopSummary> {
        // Exit block must have `lp.hdr` as its sole predecessor (no early `break` exits!)
        if preds.get(lp.loop_exit).map(|p| p.as_slice()) != Some(&[lp.hdr]) {
            return None;
        }
        let hdr_cmp = match_block_cond_cmp(&self.blocks[lp.hdr], &lp.cond)?;
        let effective_op = if lp.then_is_body {
            hdr_cmp.op
        } else {
            invert_cmp_op(hdr_cmp.op)
        };

        // Walk linear chain from `lp.body_entry` back to `lp.hdr`
        let mut chain_blocks = Vec::new();
        let mut cur = lp.body_entry;
        let mut visited = hashbrown::HashSet::new();
        let mut push_arr: Option<LocalId> = None;
        let mut assigned_in_loop = hashbrown::HashSet::new();

        for stmt in &self.blocks[lp.hdr].statements {
            if let Statement::Assign(Place::Local(d), _, _) = stmt {
                assigned_in_loop.insert(d.0);
            }
        }

        while cur != lp.hdr {
            if !lp.body.contains(&cur) || !visited.insert(cur) {
                return None;
            }
            chain_blocks.push(cur);
            for stmt in &self.blocks[cur].statements {
                if let Statement::Assign(Place::Local(d), _, _) = stmt {
                    assigned_in_loop.insert(d.0);
                }
            }
            match &self.blocks[cur].terminator {
                Some(Terminator::Goto(next, _)) => {
                    cur = next.0;
                }
                Some(Terminator::Call {
                    callee,
                    args,
                    destination,
                    success,
                    error: None,
                    ..
                }) if (callee == "array.push" || callee == "table.push") && args.len() == 2 => {
                    if push_arr.is_some() {
                        return None;
                    }
                    let arr_loc = operand_local_id(&args[0])?;
                    if !unaliased_local_arrays.contains(&arr_loc.0) {
                        return None;
                    }
                    if let Place::Local(d) = destination {
                        assigned_in_loop.insert(d.0);
                    }
                    push_arr = Some(arr_loc);
                    cur = success.0;
                }
                _ => return None,
            }
        }

        // Every block in `lp.body \ {lp.hdr}` must be on the single linear chain
        if chain_blocks.len() + 1 != lp.body.len() {
            return None;
        }
        let arr_loc = push_arr?;
        if assigned_in_loop.contains(&arr_loc.0) {
            return None;
        }

        let (idx_loc, bound_op, is_inclusive) = normalize_upper_bound_cmp(
            effective_op,
            &hdr_cmp.lhs,
            &hdr_cmp.rhs,
            &self.locals,
            excluded_locals,
            &assigned_in_loop,
        )?;
        if idx_loc == arr_loc || block_assigns_local(&self.blocks[lp.hdr], idx_loc) {
            return None;
        }
        if is_inclusive && matches!(bound_op, Operand::Constant(MirConstant::Int(i64::MAX))) {
            return None;
        }

        // Find unique assignment to `idx_loc` in `chain_blocks` and verify step == +1
        let mut idx_assigns = Vec::new();
        for &b in &chain_blocks {
            for (s_idx, stmt) in self.blocks[b].statements.iter().enumerate() {
                if matches!(stmt, Statement::Assign(Place::Local(d), _, _) if *d == idx_loc) {
                    idx_assigns.push((b, s_idx));
                }
            }
        }
        if idx_assigns.len() != 1 {
            return None;
        }
        let (step_bb, step_stmt_idx) = idx_assigns[0];
        let step_op = extract_induction_step(&self.blocks[step_bb], step_stmt_idx, idx_loc)?;
        if !matches!(step_op, Operand::Constant(MirConstant::Int(1))) {
            return None;
        }

        Some(PushLoopSummary {
            hdr: lp.hdr,
            exit: lp.loop_exit,
            body: lp.body.clone(),
            arr: arr_loc,
            idx: idx_loc,
            bound: bound_op,
            is_inclusive,
        })
    }

    fn try_apply_push_loop_exit_fact(
        &self,
        ps: &PushLoopSummary,
        preds: &[Vec<usize>],
        facts_in: &[Option<BlockFacts>],
        unaliased_local_arrays: &hashbrown::HashSet<usize>,
        excluded_locals: &hashbrown::HashSet<usize>,
        edge_facts: &mut BlockFacts,
    ) {
        let non_backedge_preds: Vec<usize> = preds[ps.hdr]
            .iter()
            .copied()
            .filter(|p| !ps.body.contains(p))
            .collect();
        if non_backedge_preds.is_empty() {
            return;
        }
        for &p in &non_backedge_preds {
            let Some(p_in) = facts_in[p].as_ref() else {
                return;
            };
            let p_out =
                self.transfer_block_facts(p, p_in, unaliased_local_arrays, excluded_locals);
            let iv = p_out.interval(ps.idx);
            if iv.lo != 0 || iv.hi != 0 {
                return;
            }
        }
        match &ps.bound {
            Operand::Constant(MirConstant::Int(c)) => {
                if *c >= 0 {
                    let facts = edge_facts.len_rels.entry(ps.arr.0).or_default();
                    if ps.is_inclusive {
                        if *c < i64::MAX {
                            push_unique(facts, LenFact::GtConst(*c));
                            push_unique(facts, LenFact::GeConst(*c + 1));
                        }
                    } else {
                        push_unique(facts, LenFact::GeConst(*c));
                    }
                }
            }
            Operand::Copy(Place::Local(m)) | Operand::Move(Place::Local(m)) => {
                let m_iv = edge_facts.interval(*m);
                let facts = edge_facts.len_rels.entry(ps.arr.0).or_default();
                if ps.is_inclusive {
                    push_unique(facts, LenFact::GtLocal(*m));
                    if m_iv.lo == m_iv.hi && m_iv.lo >= 0 && m_iv.lo < i64::MAX {
                        push_unique(facts, LenFact::GtConst(m_iv.lo));
                        push_unique(facts, LenFact::GeConst(m_iv.lo + 1));
                    }
                } else {
                    push_unique(facts, LenFact::GeLocal(*m));
                    if m_iv.lo == m_iv.hi && m_iv.lo >= 0 {
                        push_unique(facts, LenFact::GeConst(m_iv.lo));
                    }
                }
            }
            _ => {}
        }
    }

    fn transfer_block_facts(
        &self,
        block_idx: usize,
        in_facts: &BlockFacts,
        unaliased_local_arrays: &hashbrown::HashSet<usize>,
        excluded_locals: &hashbrown::HashSet<usize>,
    ) -> BlockFacts {
        let mut facts = in_facts.clone();
        let block = &self.blocks[block_idx];

        for stmt in &block.statements {
            match stmt {
                Statement::StorageDead(loc, _) => {
                    facts.kill_local(*loc);
                }
                Statement::Assign(Place::Local(dst), rval, _) => {
                    // Evaluate RHS intervals and length facts BEFORE killing `dst`
                    let new_iv = if excluded_locals.contains(&dst.0) {
                        None
                    } else {
                        eval_rvalue_interval(rval, &facts)
                    };

                    // Snapshot length facts inherited from RHS before killing `dst`
                    let mut new_self_len_facts: Vec<LenFact> = Vec::new();
                    let mut add_arr_len_facts: Vec<(usize, LenFact)> = Vec::new();

                    match rval {
                        Rvalue::Aggregate(MirType::Array(_), elems) => {
                            new_self_len_facts.push(LenFact::GeConst(elems.len() as i64));
                        }
                        Rvalue::UnaryOp(MirUnaryOp::Len, op) => {
                            if let Some(arr) = operand_local_id(op)
                                && arr != *dst
                                && matches!(
                                    self.locals.get(arr.0).map(|d| &d.ty),
                                    Some(MirType::Array(_))
                                )
                            {
                                add_arr_len_facts.push((arr.0, LenFact::EqLocal(*dst)));
                                add_arr_len_facts.push((arr.0, LenFact::GeLocal(*dst)));
                            }
                        }
                        Rvalue::Use(op) => {
                            if let Some(src) = operand_local_id(op)
                                && src != *dst
                            {
                                // Copy of an array local transfers its current length facts
                                if let Some(src_arr_facts) = facts.len_rels.get(&src.0) {
                                    for lf in src_arr_facts {
                                        if !len_fact_mentions(lf, *dst) {
                                            new_self_len_facts.push(lf.clone());
                                        }
                                    }
                                }
                                // Copy of a bound/length integer local propagates relations
                                for (&arr_idx, arr_facts) in &facts.len_rels {
                                    if arr_idx == dst.0 {
                                        continue;
                                    }
                                    for lf in arr_facts {
                                        match lf {
                                            LenFact::EqLocal(l) if *l == src => {
                                                add_arr_len_facts
                                                    .push((arr_idx, LenFact::EqLocal(*dst)));
                                                add_arr_len_facts
                                                    .push((arr_idx, LenFact::GeLocal(*dst)));
                                            }
                                            LenFact::GeLocal(l) if *l == src => {
                                                add_arr_len_facts
                                                    .push((arr_idx, LenFact::GeLocal(*dst)));
                                            }
                                            LenFact::GtLocal(l) if *l == src => {
                                                add_arr_len_facts
                                                    .push((arr_idx, LenFact::GtLocal(*dst)));
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                        Rvalue::BinaryOp(
                            MirBinaryOp::Sub,
                            lhs,
                            Operand::Constant(MirConstant::Int(c)),
                        ) if *c >= 1 => {
                            if let Some(src) = operand_local_id(lhs)
                                && src != *dst
                            {
                                for (&arr_idx, arr_facts) in &facts.len_rels {
                                    if arr_idx == dst.0 {
                                        continue;
                                    }
                                    if arr_facts.iter().any(|lf| {
                                        matches!(
                                            lf,
                                            LenFact::EqLocal(l) | LenFact::GeLocal(l) if *l == src
                                        )
                                    }) {
                                        add_arr_len_facts.push((arr_idx, LenFact::GtLocal(*dst)));
                                    }
                                }
                            }
                        }
                        _ => {}
                    }

                    // Kill stale facts for `dst`
                    facts.kill_local(*dst);

                    // Commit new interval & length facts
                    if let Some(iv) = new_iv {
                        facts.intervals.insert(dst.0, iv);
                    }
                    if !new_self_len_facts.is_empty() {
                        facts.len_rels.insert(dst.0, new_self_len_facts);
                    }
                    for (arr_idx, lf) in add_arr_len_facts {
                        let entry = facts.len_rels.entry(arr_idx).or_default();
                        push_unique(entry, lf);
                    }
                }
                _ => {}
            }
        }

        if let Some(Terminator::Call {
            callee,
            args,
            destination,
            ..
        }) = &block.terminator
        {
            if let Place::Local(dst) = destination {
                facts.kill_local(*dst);
            }
            if callee == "array.len" || callee == "table.len" {
                if let Place::Local(dst) = destination {
                    if !excluded_locals.contains(&dst.0) {
                        facts.intervals.insert(
                            dst.0,
                            IntInterval {
                                lo: 0,
                                hi: i64::MAX,
                            },
                        );
                    }
                    if let Some(arr) = args.first().and_then(operand_local_id)
                        && arr != *dst
                        && matches!(
                            self.locals.get(arr.0).map(|d| &d.ty),
                            Some(MirType::Array(_))
                        )
                    {
                        let entry = facts.len_rels.entry(arr.0).or_default();
                        push_unique(entry, LenFact::EqLocal(*dst));
                        push_unique(entry, LenFact::GeLocal(*dst));
                    }
                }
            } else if matches!(callee.as_str(), "string.len" | "print" | "println") {
                // Read-only builtins do not mutate array lengths.
                if callee == "string.len"
                    && let Place::Local(dst) = destination
                    && !excluded_locals.contains(&dst.0)
                {
                    facts.intervals.insert(
                        dst.0,
                        IntInterval {
                            lo: 0,
                            hi: i64::MAX,
                        },
                    );
                }
            } else {
                // Any potentially mutating/unknown call (including push/pop/clear/remove/host/user):
                // - Invalidate length facts for ALL non-unaliased arrays (may alias a mutated array!)
                // - Invalidate length facts for any unaliased array passed as an argument.
                let mut arg_locals = hashbrown::HashSet::new();
                for arg in args {
                    arg.collect_uses(&mut arg_locals);
                }
                facts.len_rels.retain(|&arr_idx, _| {
                    unaliased_local_arrays.contains(&arr_idx) && !arg_locals.contains(&arr_idx)
                });
            }
        }

        facts
    }
}

/// Fused unsigned range check `(idx as u64) < (len as u64)` emitted by `NG-W14-8`.
#[derive(Debug, Clone, PartialEq)]
pub struct UnsignedRangeCheck {
    pub dst: LocalId,
    pub idx_local: LocalId,
    pub len_op: Operand,
}

/// Per-block result of detecting fusable unsigned range checks (`NG-W14-8`).
#[derive(Debug, Clone, PartialEq)]
pub struct BlockUnsignedRangeFusion {
    pub skip_stmts: hashbrown::HashSet<usize>,
    pub fused_checks: hashbrown::HashMap<usize, UnsignedRangeCheck>,
}

/// Result of conservative loop-induction bounds analysis (`NG-W14-8`).
#[derive(Debug, Clone, PartialEq)]
pub struct LoopBoundsAnalysis {
    pub proven_in_bounds: hashbrown::HashMap<usize, Vec<(LocalId, LocalId)>>,
    pub non_neg_locals_in: Vec<hashbrown::HashSet<usize>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IntInterval {
    lo: i64,
    hi: i64,
}

impl IntInterval {
    const TOP: Self = Self {
        lo: i64::MIN,
        hi: i64::MAX,
    };

    fn is_top(self) -> bool {
        self.lo == i64::MIN && self.hi == i64::MAX
    }

    fn join(self, other: Self) -> Self {
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    fn widen(old: Self, next: Self) -> Self {
        let lo = if next.lo < old.lo {
            if next.lo >= 0 {
                0
            } else {
                i64::MIN
            }
        } else {
            old.lo
        };
        let hi = if next.hi > old.hi {
            i64::MAX
        } else {
            old.hi
        };
        Self { lo, hi }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LenFact {
    GeConst(i64),
    GtConst(i64),
    GeLocal(LocalId),
    GtLocal(LocalId),
    EqLocal(LocalId),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct BlockFacts {
    intervals: hashbrown::HashMap<usize, IntInterval>,
    len_rels: hashbrown::HashMap<usize, Vec<LenFact>>,
}

impl BlockFacts {
    fn interval(&self, loc: LocalId) -> IntInterval {
        self.intervals
            .get(&loc.0)
            .copied()
            .unwrap_or(IntInterval::TOP)
    }

    fn operand_interval(&self, op: &Operand) -> IntInterval {
        match op {
            Operand::Constant(MirConstant::Int(c)) => IntInterval { lo: *c, hi: *c },
            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => {
                self.interval(*loc)
            }
            _ => IntInterval::TOP,
        }
    }

    fn kill_local(&mut self, loc: LocalId) {
        self.intervals.remove(&loc.0);
        self.len_rels.remove(&loc.0);
        for facts in self.len_rels.values_mut() {
            facts.retain(|lf| !len_fact_mentions(lf, loc));
        }
        self.len_rels.retain(|_, v| !v.is_empty());
    }
}

struct NaturalLoopMeta {
    hdr: usize,
    body_entry: usize,
    loop_exit: usize,
    then_is_body: bool,
    cond: Operand,
    body: hashbrown::HashSet<usize>,
}

struct PushLoopSummary {
    hdr: usize,
    exit: usize,
    body: hashbrown::HashSet<usize>,
    arr: LocalId,
    idx: LocalId,
    bound: Operand,
    is_inclusive: bool,
}

#[derive(Debug, Clone)]
struct CondCmp {
    op: MirBinaryOp,
    lhs: Operand,
    rhs: Operand,
}

fn operand_local_id(op: &Operand) -> Option<LocalId> {
    match op {
        Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)) => Some(*l),
        _ => None,
    }
}

fn match_ge_zero_rvalue(rval: &Rvalue) -> Option<LocalId> {
    match rval {
        Rvalue::BinaryOp(MirBinaryOp::GtEq, lhs, Operand::Constant(MirConstant::Int(0))) => {
            operand_local_id(lhs)
        }
        Rvalue::BinaryOp(MirBinaryOp::LtEq, Operand::Constant(MirConstant::Int(0)), rhs) => {
            operand_local_id(rhs)
        }
        _ => None,
    }
}

fn match_lt_bound_rvalue(rval: &Rvalue) -> Option<(LocalId, Operand)> {
    match rval {
        Rvalue::BinaryOp(MirBinaryOp::Lt, lhs, rhs) => {
            Some((operand_local_id(lhs)?, rhs.clone()))
        }
        Rvalue::BinaryOp(MirBinaryOp::Gt, lhs, rhs) => {
            Some((operand_local_id(rhs)?, lhs.clone()))
        }
        _ => None,
    }
}

fn len_fact_mentions(lf: &LenFact, loc: LocalId) -> bool {
    matches!(
        lf,
        LenFact::GeLocal(l) | LenFact::GtLocal(l) | LenFact::EqLocal(l) if *l == loc
    )
}

fn push_unique<T: PartialEq>(v: &mut Vec<T>, item: T) {
    if !v.contains(&item) {
        v.push(item);
    }
}

fn mark_place_escaped(place: &Place, escaped: &mut [bool], read_count: &mut [usize]) {
    match place {
        Place::Local(loc) => {
            if loc.0 < escaped.len() {
                escaped[loc.0] = true;
                read_count[loc.0] += 1;
            }
        }
        Place::Field(base, _) => mark_place_escaped(base, escaped, read_count),
        Place::Index(base, idx) => {
            mark_place_escaped(base, escaped, read_count);
            if idx.0 < escaped.len() {
                escaped[idx.0] = true;
                read_count[idx.0] += 1;
            }
        }
    }
}

fn mark_operand_escaped(op: &Operand, escaped: &mut [bool], read_count: &mut [usize]) {
    match op {
        Operand::Copy(p) | Operand::Move(p) => mark_place_escaped(p, escaped, read_count),
        Operand::Constant(_) => {}
    }
}

fn mark_rvalue_escaped(rval: &Rvalue, escaped: &mut [bool], read_count: &mut [usize]) {
    match rval {
        Rvalue::Use(op) | Rvalue::UnaryOp(_, op) | Rvalue::Cast(op, _) => {
            mark_operand_escaped(op, escaped, read_count);
        }
        Rvalue::BinaryOp(_, l, r) => {
            mark_operand_escaped(l, escaped, read_count);
            mark_operand_escaped(r, escaped, read_count);
        }
        Rvalue::Aggregate(_, ops) => {
            for op in ops {
                mark_operand_escaped(op, escaped, read_count);
            }
        }
        Rvalue::Closure { captures, .. } => {
            for (_, op, _) in captures {
                mark_operand_escaped(op, escaped, read_count);
            }
        }
    }
}

fn eval_rvalue_interval(rval: &Rvalue, facts: &BlockFacts) -> Option<IntInterval> {
    let iv = match rval {
        Rvalue::Use(op) => facts.operand_interval(op),
        Rvalue::UnaryOp(MirUnaryOp::Len, _) => IntInterval {
            lo: 0,
            hi: i64::MAX,
        },
        Rvalue::BinaryOp(MirBinaryOp::Add, lhs, rhs) => {
            let a = facts.operand_interval(lhs);
            let b = facts.operand_interval(rhs);
            let lo = (a.lo as i128) + (b.lo as i128);
            let hi = (a.hi as i128) + (b.hi as i128);
            if lo >= (i64::MIN as i128) && hi <= (i64::MAX as i128) {
                IntInterval {
                    lo: lo as i64,
                    hi: hi as i64,
                }
            } else {
                IntInterval::TOP
            }
        }
        Rvalue::BinaryOp(MirBinaryOp::Sub, lhs, rhs) => {
            let a = facts.operand_interval(lhs);
            let b = facts.operand_interval(rhs);
            let lo = (a.lo as i128) - (b.hi as i128);
            let hi = (a.hi as i128) - (b.lo as i128);
            if lo >= (i64::MIN as i128) && hi <= (i64::MAX as i128) {
                IntInterval {
                    lo: lo as i64,
                    hi: hi as i64,
                }
            } else {
                IntInterval::TOP
            }
        }
        Rvalue::BinaryOp(MirBinaryOp::Mul, lhs, rhs) => {
            let a = facts.operand_interval(lhs);
            let b = facts.operand_interval(rhs);
            if a.lo >= 0 && b.lo >= 0 {
                let lo = (a.lo as i128) * (b.lo as i128);
                let hi = (a.hi as i128) * (b.hi as i128);
                if hi <= (i64::MAX as i128) {
                    IntInterval {
                        lo: lo as i64,
                        hi: hi as i64,
                    }
                } else {
                    IntInterval::TOP
                }
            } else {
                IntInterval::TOP
            }
        }
        _ => IntInterval::TOP,
    };
    if iv.is_top() { None } else { Some(iv) }
}

fn invert_cmp_op(op: MirBinaryOp) -> MirBinaryOp {
    match op {
        MirBinaryOp::Lt => MirBinaryOp::GtEq,
        MirBinaryOp::LtEq => MirBinaryOp::Gt,
        MirBinaryOp::Gt => MirBinaryOp::LtEq,
        MirBinaryOp::GtEq => MirBinaryOp::Lt,
        other => other,
    }
}

fn match_block_cond_cmp(block: &BasicBlock, cond: &Operand) -> Option<CondCmp> {
    let mut target = operand_local_id(cond)?;
    let mut def_pos = block.statements.len();
    for _ in 0..4 {
        let mut found = None;
        for (i, stmt) in block.statements[..def_pos].iter().enumerate().rev() {
            if let Statement::Assign(Place::Local(dst), rval, _) = stmt
                && *dst == target
            {
                found = Some((i, rval));
                break;
            }
        }
        let (pos, rval) = found?;
        match rval {
            Rvalue::Use(op) => {
                target = operand_local_id(op)?;
                def_pos = pos;
            }
            Rvalue::BinaryOp(
                op @ (MirBinaryOp::Lt | MirBinaryOp::LtEq | MirBinaryOp::Gt | MirBinaryOp::GtEq),
                lhs,
                rhs,
            ) => {
                // Verify neither lhs nor rhs local is modified after `pos`
                let lhs_loc = operand_local_id(lhs);
                let rhs_loc = operand_local_id(rhs);
                for stmt in &block.statements[pos + 1..] {
                    if let Statement::Assign(Place::Local(d), _, _) = stmt {
                        if Some(*d) == lhs_loc || Some(*d) == rhs_loc {
                            return None;
                        }
                    } else if let Statement::StorageDead(d, _) = stmt {
                        if Some(*d) == lhs_loc || Some(*d) == rhs_loc {
                            return None;
                        }
                    }
                }
                return Some(CondCmp {
                    op: *op,
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                });
            }
            _ => return None,
        }
    }
    None
}

fn refine_facts_for_branch(facts: &mut BlockFacts, cmp: &CondCmp, taken: bool) {
    let op = if taken { cmp.op } else { invert_cmp_op(cmp.op) };
    let l_iv = facts.operand_interval(&cmp.lhs);
    let r_iv = facts.operand_interval(&cmp.rhs);

    let (new_l, new_r) = match op {
        MirBinaryOp::Lt => {
            let nl = if r_iv.hi > i64::MIN {
                IntInterval {
                    lo: l_iv.lo,
                    hi: l_iv.hi.min(r_iv.hi - 1),
                }
            } else {
                l_iv
            };
            let nr = if l_iv.lo < i64::MAX {
                IntInterval {
                    lo: r_iv.lo.max(l_iv.lo + 1),
                    hi: r_iv.hi,
                }
            } else {
                r_iv
            };
            (nl, nr)
        }
        MirBinaryOp::LtEq => (
            IntInterval {
                lo: l_iv.lo,
                hi: l_iv.hi.min(r_iv.hi),
            },
            IntInterval {
                lo: r_iv.lo.max(l_iv.lo),
                hi: r_iv.hi,
            },
        ),
        MirBinaryOp::Gt => {
            let nl = if r_iv.lo < i64::MAX {
                IntInterval {
                    lo: l_iv.lo.max(r_iv.lo + 1),
                    hi: l_iv.hi,
                }
            } else {
                l_iv
            };
            let nr = if l_iv.hi > i64::MIN {
                IntInterval {
                    lo: r_iv.lo,
                    hi: r_iv.hi.min(l_iv.hi - 1),
                }
            } else {
                r_iv
            };
            (nl, nr)
        }
        MirBinaryOp::GtEq => (
            IntInterval {
                lo: l_iv.lo.max(r_iv.lo),
                hi: l_iv.hi,
            },
            IntInterval {
                lo: r_iv.lo,
                hi: r_iv.hi.min(l_iv.hi),
            },
        ),
        _ => return,
    };

    if let Some(l_loc) = operand_local_id(&cmp.lhs)
        && new_l.lo <= new_l.hi
        && !new_l.is_top()
    {
        facts.intervals.insert(l_loc.0, new_l);
    }
    if let Some(r_loc) = operand_local_id(&cmp.rhs)
        && new_r.lo <= new_r.hi
        && !new_r.is_top()
    {
        facts.intervals.insert(r_loc.0, new_r);
    }
}

fn propagate_block_facts(
    target: usize,
    edge_facts: &BlockFacts,
    loop_assigned_at_target: Option<&hashbrown::HashSet<usize>>,
    facts_in: &mut [Option<BlockFacts>],
    merge_counts: &mut [usize],
    worklist: &mut VecDeque<usize>,
    in_worklist: &mut hashbrown::HashSet<usize>,
) {
    match &mut facts_in[target] {
        None => {
            facts_in[target] = Some(edge_facts.clone());
            if in_worklist.insert(target) {
                worklist.push_back(target);
            }
        }
        Some(existing) => {
            merge_counts[target] += 1;
            let do_widen_hdr = loop_assigned_at_target.is_some() && merge_counts[target] >= 2;
            let mut changed = false;

            // Join/widen intervals: only locals present in BOTH remain below TOP.
            // Never widen a local that is loop-invariant at `target` (e.g. an outer-loop step
            // variable entering an inner loop header).
            let existing_keys: Vec<usize> = existing.intervals.keys().copied().collect();
            for k in existing_keys {
                let old_iv = existing.intervals[&k];
                match edge_facts.intervals.get(&k).copied() {
                    None => {
                        existing.intervals.remove(&k);
                        changed = true;
                    }
                    Some(new_iv) => {
                        let joined = old_iv.join(new_iv);
                        let widen_k = do_widen_hdr
                            && loop_assigned_at_target.is_some_and(|set| set.contains(&k));
                        let next_iv = if widen_k {
                            IntInterval::widen(old_iv, joined)
                        } else {
                            joined
                        };
                        if next_iv.is_top() {
                            existing.intervals.remove(&k);
                            changed = true;
                        } else if next_iv != old_iv {
                            existing.intervals.insert(k, next_iv);
                            changed = true;
                        }
                    }
                }
            }

            // Intersect len_rels
            let len_keys: Vec<usize> = existing.len_rels.keys().copied().collect();
            for arr_k in len_keys {
                match edge_facts.len_rels.get(&arr_k) {
                    None => {
                        existing.len_rels.remove(&arr_k);
                        changed = true;
                    }
                    Some(edge_lf) => {
                        let cur = existing.len_rels.get_mut(&arr_k).unwrap();
                        let old_len = cur.len();
                        cur.retain(|lf| edge_lf.contains(lf));
                        if cur.len() != old_len {
                            changed = true;
                        }
                        if cur.is_empty() {
                            existing.len_rels.remove(&arr_k);
                        }
                    }
                }
            }

            if changed && in_worklist.insert(target) {
                worklist.push_back(target);
            }
        }
    }
}

fn normalize_upper_bound_cmp(
    op: MirBinaryOp,
    lhs: &Operand,
    rhs: &Operand,
    locals: &[LocalDecl],
    excluded_locals: &hashbrown::HashSet<usize>,
    assigned_in_loop: &hashbrown::HashSet<usize>,
) -> Option<(LocalId, Operand, bool)> {
    let (idx_op, bound_op, is_inclusive) = match op {
        MirBinaryOp::Lt => (lhs, rhs, false),
        MirBinaryOp::LtEq => (lhs, rhs, true),
        MirBinaryOp::Gt => (rhs, lhs, false),
        MirBinaryOp::GtEq => (rhs, lhs, true),
        _ => return None,
    };
    let j_loc = operand_local_id(idx_op)?;
    if excluded_locals.contains(&j_loc.0)
        || !locals.get(j_loc.0).is_some_and(|d| d.ty == MirType::Int)
    {
        return None;
    }
    match bound_op {
        Operand::Constant(MirConstant::Int(_)) => {}
        Operand::Copy(Place::Local(b_loc)) | Operand::Move(Place::Local(b_loc)) => {
            if *b_loc == j_loc
                || excluded_locals.contains(&b_loc.0)
                || assigned_in_loop.contains(&b_loc.0)
                || !locals.get(b_loc.0).is_some_and(|d| d.ty == MirType::Int)
            {
                return None;
            }
        }
        _ => return None,
    }
    Some((j_loc, bound_op.clone(), is_inclusive))
}

fn block_assigns_local(block: &BasicBlock, loc: LocalId) -> bool {
    block
        .statements
        .iter()
        .any(|s| matches!(s, Statement::Assign(Place::Local(d), _, _) if *d == loc))
}

fn extract_induction_step(
    block: &BasicBlock,
    step_stmt_idx: usize,
    j_loc: LocalId,
) -> Option<Operand> {
    let Statement::Assign(Place::Local(dst), rval, _) = block.statements.get(step_stmt_idx)?
    else {
        return None;
    };
    if *dst != j_loc {
        return None;
    }
    let match_add = |rv: &Rvalue| -> Option<Operand> {
        let Rvalue::BinaryOp(MirBinaryOp::Add, l, r) = rv else {
            return None;
        };
        if operand_local_id(l) == Some(j_loc) {
            Some(r.clone())
        } else if operand_local_id(r) == Some(j_loc) {
            Some(l.clone())
        } else {
            None
        }
    };
    if let Some(step) = match_add(rval) {
        return Some(step);
    }
    // 2-statement temporary form: `_tmp = j + step; j = move/copy _tmp;`
    if let Rvalue::Use(op) = rval
        && let Some(tmp_loc) = operand_local_id(op)
        && tmp_loc != j_loc
    {
        for (i, stmt) in block.statements[..step_stmt_idx].iter().enumerate().rev() {
            if let Statement::Assign(Place::Local(d), prev_rval, _) = stmt
                && *d == tmp_loc
            {
                let step = match_add(prev_rval)?;
                let step_loc = operand_local_id(&step);
                for mid in &block.statements[i + 1..step_stmt_idx] {
                    if let Statement::Assign(Place::Local(md), _, _) = mid {
                        if *md == j_loc || *md == tmp_loc || Some(*md) == step_loc {
                            return None;
                        }
                    } else if let Statement::StorageDead(md, _) = mid {
                        if *md == j_loc || *md == tmp_loc || Some(*md) == step_loc {
                            return None;
                        }
                    }
                }
                return Some(step);
            }
        }
    }
    None
}

fn collect_rvalue_index_accesses(
    rval: &Rvalue,
    out: &mut hashbrown::HashMap<LocalId, Vec<LocalId>>,
) {
    let mut inspect_op = |op: &Operand| {
        if let Operand::Copy(Place::Index(base, idx)) | Operand::Move(Place::Index(base, idx)) = op
            && let Place::Local(arr_loc) = base.as_ref()
        {
            let entry = out.entry(*idx).or_default();
            if !entry.contains(arr_loc) {
                entry.push(*arr_loc);
            }
        }
    };
    match rval {
        Rvalue::Use(op) | Rvalue::UnaryOp(_, op) | Rvalue::Cast(op, _) => inspect_op(op),
        Rvalue::BinaryOp(_, l, r) => {
            inspect_op(l);
            inspect_op(r);
        }
        Rvalue::Aggregate(_, ops) => {
            for op in ops {
                inspect_op(op);
            }
        }
        Rvalue::Closure { captures, .. } => {
            for (_, op, _) in captures {
                inspect_op(op);
            }
        }
    }
}

fn len_facts_prove_bound(
    len_facts: &[LenFact],
    bound_op: &Operand,
    is_inclusive: bool,
    hdr_facts: &BlockFacts,
) -> bool {
    match bound_op {
        Operand::Constant(MirConstant::Int(c)) => {
            if *c < 0 {
                return false;
            }
            len_facts.iter().any(|lf| match lf {
                LenFact::GeConst(k) => {
                    if is_inclusive {
                        *k > *c
                    } else {
                        *k >= *c
                    }
                }
                LenFact::GtConst(k) => {
                    if is_inclusive {
                        *k >= *c
                    } else {
                        *k >= c.saturating_sub(1)
                    }
                }
                _ => false,
            })
        }
        Operand::Copy(Place::Local(b_loc)) | Operand::Move(Place::Local(b_loc)) => {
            let direct = len_facts.iter().any(|lf| match lf {
                LenFact::GtLocal(l) if *l == *b_loc => true,
                LenFact::GeLocal(l) | LenFact::EqLocal(l) if !is_inclusive && *l == *b_loc => true,
                _ => false,
            });
            if direct {
                return true;
            }
            let b_hi = hdr_facts.interval(*b_loc).hi;
            if b_hi < 0 {
                return false;
            }
            len_facts.iter().any(|lf| match lf {
                LenFact::GeConst(k) => {
                    if is_inclusive {
                        *k > b_hi
                    } else {
                        *k >= b_hi
                    }
                }
                LenFact::GtConst(k) => {
                    if is_inclusive {
                        *k >= b_hi
                    } else {
                        *k >= b_hi.saturating_sub(1)
                    }
                }
                _ => false,
            })
        }
        _ => false,
    }
}

pub const MIR_STRUCT_DECODE_PREFIX: &str = "__mir_struct_decode_into:";

/// Builds the canonical MIR call target name for `StructDecodeInto`.
pub fn mir_struct_decode_callee(struct_name: &str) -> String {
    alloc::format!("{MIR_STRUCT_DECODE_PREFIX}{struct_name}")
}

/// Extracts the target struct name from a `StructDecodeInto` MIR call target.
pub fn parse_mir_struct_decode_callee(callee: &str) -> Option<&str> {
    callee.strip_prefix(MIR_STRUCT_DECODE_PREFIX)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MirStructFieldLayout {
    pub ordinal: u16,
    pub name: String,
    pub tag: CanonicalTypeTag,
    pub byte_offset: usize,
    pub byte_size: usize,
    pub is_nullable: bool,
    pub is_optional: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MirStructLayout {
    pub type_id: u64,
    pub type_name: String,
    pub fields: Vec<MirStructFieldLayout>,
    pub size_bytes: usize,
    pub align_bytes: usize,
    pub is_c_flat: bool,
    pub target_abi: TargetAbi,
}

impl From<&LogicalLayoutV1> for MirStructLayout {
    fn from(layout: &LogicalLayoutV1) -> Self {
        Self {
            type_id: layout.type_id,
            type_name: layout.type_name.clone(),
            fields: layout
                .fields
                .iter()
                .map(|f| MirStructFieldLayout {
                    ordinal: f.ordinal,
                    name: f.name.clone(),
                    tag: f.tag,
                    byte_offset: f.byte_offset,
                    byte_size: f.byte_size,
                    is_nullable: f.is_nullable,
                    is_optional: f.is_optional,
                })
                .collect(),
            size_bytes: layout.size_bytes,
            align_bytes: layout.align_bytes,
            is_c_flat: layout.is_c_flat,
            target_abi: layout.target_abi,
        }
    }
}

impl MirStructLayout {
    pub fn to_logical_layout(&self) -> LogicalLayoutV1 {
        let fields = self
            .fields
            .iter()
            .map(|f| {
                let mut desc = LogicalFieldDesc::new(f.ordinal, f.name.clone(), f.tag)
                    .with_nullability(f.is_nullable, f.is_optional);
                desc.byte_offset = f.byte_offset;
                desc.byte_size = f.byte_size;
                desc
            })
            .collect();
        LogicalLayoutV1::new_with_abi(self.type_id, self.type_name.clone(), fields, self.target_abi)
    }

    pub fn field_by_name(&self, name: &str) -> Option<&MirStructFieldLayout> {
        self.fields.iter().find(|f| f.name == name)
    }

    pub fn field_by_ordinal(&self, ordinal: u16) -> Option<&MirStructFieldLayout> {
        self.fields.iter().find(|f| f.ordinal == ordinal)
    }
}

/// First-class MIR metadata for `D-LAYOUT` operations (`StructDecodeInto` / `StructLoadField`).
#[derive(Debug, Clone, PartialEq)]
pub enum MirStructOp {
    StructDecodeInto {
        struct_name: String,
        layout: MirStructLayout,
        dst: LocalId,
        src_bytes: Operand,
    },
    StructLoadField {
        struct_name: String,
        base: Place,
        ordinal: u16,
        byte_offset: usize,
        tag: CanonicalTypeTag,
        dst: LocalId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MirEnumDef {
    pub name: String,
    pub variants: Vec<(String, Vec<MirType>)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MirModule {
    pub name: String,
    pub functions: Vec<MirFunction>,
    pub struct_layouts: Vec<MirStructLayout>,
    pub struct_ops: Vec<MirStructOp>,
    pub enum_defs: Vec<MirEnumDef>,
}

impl MirModule {
    pub fn new(name: String) -> Self {
        Self {
            name,
            functions: Vec::new(),
            struct_layouts: Vec::new(),
            struct_ops: Vec::new(),
            enum_defs: Vec::new(),
        }
    }

    pub fn add_function(&mut self, func: MirFunction) {
        self.functions.push(func);
    }

    pub fn register_enum_def(&mut self, enum_def: MirEnumDef) {
        if let Some(existing) = self
            .enum_defs
            .iter_mut()
            .find(|e| e.name == enum_def.name)
        {
            *existing = enum_def;
        } else {
            self.enum_defs.push(enum_def);
        }
    }

    pub fn register_struct_layout(&mut self, layout: MirStructLayout) {
        if let Some(existing) = self
            .struct_layouts
            .iter_mut()
            .find(|l| l.type_name == layout.type_name)
        {
            *existing = layout;
        } else {
            self.struct_layouts.push(layout);
        }
    }

    pub fn find_struct_layout(&self, type_name: &str) -> Option<&MirStructLayout> {
        self.struct_layouts.iter().find(|l| l.type_name == type_name)
    }

    pub fn record_struct_op(&mut self, op: MirStructOp) {
        self.struct_ops.push(op);
    }
}

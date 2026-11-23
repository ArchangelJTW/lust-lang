use crate::ast::Span;
use crate::data::layout::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1, TargetAbi};
use alloc::{boxed::Box, string::String, vec::Vec};
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

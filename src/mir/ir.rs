use crate::ast::Span;
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
    Function {
        params: Vec<MirType>,
        return_type: Box<MirType>,
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
}

impl fmt::Display for MirBinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            MirBinaryOp::Add => "+",
            MirBinaryOp::Sub => "-",
            MirBinaryOp::Mul => "*",
            MirBinaryOp::Div => "/",
            MirBinaryOp::Mod => "%",
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
        };
        write!(f, "{s}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirUnaryOp {
    Neg,
    Not,
    Len,
}

impl fmt::Display for MirUnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MirUnaryOp::Neg => write!(f, "-"),
            MirUnaryOp::Not => write!(f, "!"),
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
}

impl BasicBlock {
    pub fn new(id: BlockId) -> Self {
        Self {
            id,
            statements: Vec::new(),
            terminator: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionSignature {
    pub name: String,
    pub param_types: Vec<MirType>,
    pub return_type: MirType,
    pub is_public: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MirFunction {
    pub signature: FunctionSignature,
    pub params: Vec<LocalId>,
    pub return_local: LocalId,
    pub locals: Vec<LocalDecl>,
    pub entry: BlockId,
    pub blocks: Vec<BasicBlock>,
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
        });
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct MirModule {
    pub name: String,
    pub functions: Vec<MirFunction>,
}

impl MirModule {
    pub fn new(name: String) -> Self {
        Self {
            name,
            functions: Vec::new(),
        }
    }

    pub fn add_function(&mut self, func: MirFunction) {
        self.functions.push(func);
    }
}

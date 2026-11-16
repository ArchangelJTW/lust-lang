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
}

impl LocalDecl {
    pub fn with_stack_promoted(mut self, promoted: bool) -> Self {
        self.is_stack_promoted = promoted;
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
                is_stack_promoted: false,
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
            is_stack_promoted: false,
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

#[derive(Debug, Clone, PartialEq)]
pub struct MirModule {
    pub name: String,
    pub functions: Vec<MirFunction>,
    pub struct_layouts: Vec<MirStructLayout>,
    pub struct_ops: Vec<MirStructOp>,
}

impl MirModule {
    pub fn new(name: String) -> Self {
        Self {
            name,
            functions: Vec::new(),
            struct_layouts: Vec::new(),
            struct_ops: Vec::new(),
        }
    }

    pub fn add_function(&mut self, func: MirFunction) {
        self.functions.push(func);
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

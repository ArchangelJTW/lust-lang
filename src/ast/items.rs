use super::{Expr, Span, Stmt, Type};
use alloc::{string::String, vec::Vec};
#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub kind: ItemKind,
    pub span: Span,
}

impl Item {
    pub fn new(kind: ItemKind, span: Span) -> Self {
        Self { kind, span }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ItemKind {
    Script(Vec<Stmt>),
    Function(FunctionDef),
    Struct(StructDef),
    Enum(EnumDef),
    Trait(TraitDef),
    Impl(ImplBlock),
    TypeAlias {
        name: String,
        type_params: Vec<String>,
        target: Type,
    },
    Module {
        name: String,
        items: Vec<Item>,
    },
    Use {
        public: bool,
        tree: UseTree,
    },
    Const {
        name: String,
        ty: Type,
        value: Expr,
    },
    Static {
        name: String,
        mutable: bool,
        ty: Type,
        value: Expr,
    },
    Extern {
        abi: String,
        items: Vec<ExternItem>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum UseTree {
    Path {
        path: Vec<String>,
        alias: Option<String>,
        import_module: bool,
    },
    Group {
        prefix: Vec<String>,
        items: Vec<UseTreeItem>,
    },
    Glob {
        prefix: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct UseTreeItem {
    pub path: Vec<String>,
    pub alias: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    pub name: String,
    pub type_params: Vec<String>,
    pub trait_bounds: Vec<TraitBound>,
    pub params: Vec<FunctionParam>,
    pub return_type: Option<Type>,
    pub body: Vec<Stmt>,
    pub is_method: bool,
    pub visibility: Visibility,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionParam {
    pub name: String,
    pub ty: Type,
    pub is_self: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructDef {
    pub name: String,
    pub type_params: Vec<String>,
    pub trait_bounds: Vec<TraitBound>,
    pub fields: Vec<StructField>,
    pub visibility: Visibility,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldOwnership {
    Strong,
    Weak,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructField {
    pub name: String,
    pub ty: Type,
    pub visibility: Visibility,
    pub ownership: FieldOwnership,
    pub weak_target: Option<Type>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumDef {
    pub name: String,
    pub type_params: Vec<String>,
    pub trait_bounds: Vec<TraitBound>,
    pub variants: Vec<EnumVariant>,
    pub visibility: Visibility,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant {
    pub name: String,
    pub fields: Option<Vec<Type>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraitDef {
    pub name: String,
    pub type_params: Vec<String>,
    pub methods: Vec<TraitMethod>,
    pub visibility: Visibility,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraitMethod {
    pub name: String,
    pub type_params: Vec<String>,
    pub params: Vec<FunctionParam>,
    pub return_type: Option<Type>,
    pub default_impl: Option<Vec<Stmt>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImplBlock {
    pub type_params: Vec<String>,
    pub trait_name: Option<String>,
    pub target_type: Type,
    pub methods: Vec<FunctionDef>,
    pub where_clause: Vec<TraitBound>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraitBound {
    pub type_param: String,
    pub traits: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    Public,
    Private,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExternItem {
    Function {
        name: String,
        params: Vec<Type>,
        return_type: Option<Type>,
    },
}

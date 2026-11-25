use super::{Span, Stmt, Type};
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
    Module { name: String, items: Vec<Item> },
    Use { public: bool, tree: UseTree },
    Extern { abi: String, items: Vec<ExternItem> },
    Service(ServiceDef),
    Test(TestDecl),
    Suite(SuiteDecl),
    Schema(SchemaDef),
    TypeAlias {
        name: String,
        ty: Type,
        visibility: Visibility,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TestModifier {
    None,
    Skip(Option<String>),
    Only,
    Timeout(u64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TestDecl {
    pub name: String,
    pub modifier: TestModifier,
    pub body: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SuiteHooks {
    pub before_each: Option<Vec<Stmt>>,
    pub after_each: Option<Vec<Stmt>>,
    pub before_all: Option<Vec<Stmt>>,
    pub after_all: Option<Vec<Stmt>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SuiteDecl {
    pub name: String,
    pub modifier: TestModifier,
    pub tests: Vec<TestDecl>,
    pub nested_suites: Vec<SuiteDecl>,
    pub hooks: SuiteHooks,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ServiceDef {
    pub name: String,
    pub target_type: Option<Type>,
    pub doc: Option<String>,
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
    pub doc: Option<String>,
    pub effects: Option<super::ReturnSpec>,
}

impl FunctionDef {
    pub fn return_spec(&self) -> super::ReturnSpec {
        if let Some(effects) = &self.effects {
            effects.clone()
        } else {
            super::ReturnSpec {
                return_type: self.return_type.clone(),
                error_type: None,
                requirements: Vec::new(),
            }
        }
    }

    pub fn error_type(&self) -> Option<Type> {
        self.return_spec().error_type
    }

    pub fn requirements(&self) -> Vec<Type> {
        self.return_spec().requirements
    }
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
    pub doc: Option<String>,
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
    pub doc: Option<String>,
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
    pub doc: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraitMethod {
    pub name: String,
    pub type_params: Vec<String>,
    pub params: Vec<FunctionParam>,
    pub return_type: Option<Type>,
    pub default_impl: Option<Vec<Stmt>>,
    pub doc: Option<String>,
}

impl TraitMethod {
    pub fn return_spec(&self) -> super::ReturnSpec {
        match &self.return_type {
            Some(Type {
                kind: super::TypeKind::Effect { success, error, requirements },
                ..
            }) => super::ReturnSpec {
                return_type: Some((**success).clone()),
                error_type: if matches!(error.kind, super::TypeKind::Never) {
                    None
                } else {
                    Some((**error).clone())
                },
                requirements: requirements.clone(),
            },
            Some(t) => super::ReturnSpec {
                return_type: Some(t.clone()),
                error_type: None,
                requirements: Vec::new(),
            },
            None => super::ReturnSpec {
                return_type: None,
                error_type: None,
                requirements: Vec::new(),
            },
        }
    }

    pub fn error_type(&self) -> Option<Type> {
        self.return_spec().error_type
    }

    pub fn requirements(&self) -> Vec<Type> {
        self.return_spec().requirements
    }
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
pub struct ExternParam {
    pub name: Option<String>,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExternItem {
    Function {
        name: String,
        params: Vec<ExternParam>,
        return_type: Option<Type>,
        doc: Option<String>,
    },
    Const {
        name: String,
        ty: Type,
        doc: Option<String>,
    },
    Struct(StructDef),
    Enum(EnumDef),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SchemaDef {
    pub name: String,
    pub fields: Vec<SchemaField>,
    pub visibility: Visibility,
    pub doc: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SchemaField {
    pub name: String,
    pub ty: Type,
    pub constraints: Vec<SchemaConstraint>,
    pub visibility: Visibility,
    pub doc: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SchemaConstraint {
    pub name: String,
    pub args: Vec<super::Expr>,
    pub span: Span,
}

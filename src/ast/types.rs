use super::Span;
use alloc::{
    boxed::Box,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt;
use core::hash::{Hash, Hasher};

#[derive(Debug, Clone, Eq)]
pub struct Type {
    pub kind: TypeKind,
    pub span: Span,
}

impl PartialEq for Type {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

impl Hash for Type {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
    }
}

impl Type {
    pub fn new(kind: TypeKind, span: Span) -> Self {
        Self { kind, span }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.kind.fmt(f)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeKind {
    Int,
    Float,
    String,
    Bool,
    Named(String),
    Generic(String),
    Array(Box<Type>),
    Map(Box<Type>, Box<Type>),
    Function {
        params: Vec<Type>,
        return_type: Box<Type>,
    },
    Tuple(Vec<Type>),
    Option(Box<Type>),
    Result(Box<Type>, Box<Type>),
    Ref(Box<Type>),
    MutRef(Box<Type>),
    Pointer {
        mutable: bool,
        pointee: Box<Type>,
    },
    GenericInstance {
        name: String,
        type_args: Vec<Type>,
    },
    Unknown,
    Union(Vec<Type>),
    Trait(String),
    TraitBound(Vec<String>),
    Unit,
    Infer,
    Never,
    Effect {
        success: Box<Type>,
        error: Box<Type>,
        requirements: Vec<Type>,
    },
    Typeof(String),
    InferType(Box<Type>),
    InferInput(Box<Type>),
    InferOutput(Box<Type>),
    Schema {
        input: Box<Type>,
        output: Box<Type>,
        fields: Option<Vec<(String, Box<Type>, Box<Type>)>>,
    },
}

impl TypeKind {
    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            TypeKind::Int | TypeKind::Float | TypeKind::String | TypeKind::Bool
        )
    }
}

impl fmt::Display for TypeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeKind::Int => write!(f, "int"),
            TypeKind::Float => write!(f, "float"),
            TypeKind::String => write!(f, "string"),
            TypeKind::Bool => write!(f, "bool"),
            TypeKind::Named(name) => write!(f, "{name}"),
            TypeKind::Generic(name) => write!(f, "{name}"),
            TypeKind::Array(inner) => write!(f, "Array<{}>", inner),
            TypeKind::Map(key, value) => write!(f, "Map<{}, {}>", key, value),
            TypeKind::Function {
                params,
                return_type,
            } => {
                let params = params
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "function({}): {}", params, return_type)
            }

            TypeKind::Tuple(elements) => {
                let elems = elements
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "Tuple<{}>", elems)
            }

            TypeKind::Option(inner) => write!(f, "Option<{}>", inner),
            TypeKind::Result(ok, err) => write!(f, "Result<{}, {}>", ok, err),
            TypeKind::Ref(inner) => write!(f, "&{}", inner),
            TypeKind::MutRef(inner) => write!(f, "&mut {}", inner),
            TypeKind::Pointer { mutable, pointee } => {
                if *mutable {
                    write!(f, "*mut {}", pointee)
                } else {
                    write!(f, "*{}", pointee)
                }
            }

            TypeKind::GenericInstance { name, type_args } => {
                let args = type_args
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "{name}<{}>", args)
            }

            TypeKind::Unknown => write!(f, "unknown"),
            TypeKind::Union(types) => {
                let parts = types
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(" | ");
                write!(f, "{parts}")
            }

            TypeKind::Trait(name) => write!(f, "{name}"),
            TypeKind::TraitBound(traits) => write!(f, "{}", traits.join(" + ")),
            TypeKind::Unit => write!(f, "()"),
            TypeKind::Infer => write!(f, "_"),
            TypeKind::Never => write!(f, "never"),
            TypeKind::Effect {
                success,
                error,
                requirements,
            } => {
                if requirements.is_empty() {
                    if matches!(error.kind, TypeKind::Never) {
                        write!(f, "Effect<{}>", success)
                    } else {
                        write!(f, "Effect<{}, {}>", success, error)
                    }
                } else {
                    let reqs = requirements
                        .iter()
                        .map(|t| t.to_string())
                        .collect::<Vec<_>>()
                        .join(" + ");
                    write!(f, "Effect<{}, {}, {}>", success, error, reqs)
                }
            }
            TypeKind::Typeof(name) => write!(f, "typeof {name}"),
            TypeKind::InferType(inner) => write!(f, "Infer<{inner}>"),
            TypeKind::InferInput(inner) => write!(f, "InferInput<{inner}>"),
            TypeKind::InferOutput(inner) => write!(f, "InferOutput<{inner}>"),
            TypeKind::Schema { input, output, .. } => write!(f, "Schema<{input}, {output}>"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReturnSpec {
    pub return_type: Option<Type>,
    pub error_type: Option<Type>,
    pub requirements: Vec<Type>,
}

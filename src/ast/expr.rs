use super::{Span, Type};
use std::fmt;
#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

impl Expr {
    pub fn new(kind: ExprKind, span: Span) -> Self {
        Self { kind, span }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Literal(Literal),
    Identifier(String),
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
    },
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    MethodCall {
        receiver: Box<Expr>,
        method: String,
        type_args: Option<Vec<Type>>,
        args: Vec<Expr>,
    },
    FieldAccess {
        object: Box<Expr>,
        field: String,
    },
    Index {
        object: Box<Expr>,
        index: Box<Expr>,
    },
    Array(Vec<Expr>),
    Map(Vec<(Expr, Expr)>),
    Tuple(Vec<Expr>),
    StructLiteral {
        name: String,
        fields: Vec<StructLiteralField>,
    },
    EnumConstructor {
        enum_name: String,
        variant: String,
        args: Vec<Expr>,
    },
    Lambda {
        params: Vec<(String, Option<Type>)>,
        return_type: Option<Type>,
        body: Box<Expr>,
    },
    Paren(Box<Expr>),
    Cast {
        expr: Box<Expr>,
        target_type: Type,
    },
    TypeCheck {
        expr: Box<Expr>,
        check_type: Type,
    },
    IsPattern {
        expr: Box<Expr>,
        pattern: Pattern,
    },
    If {
        condition: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Option<Box<Expr>>,
    },
    Block(Vec<super::stmt::Stmt>),
    Return(Vec<Expr>),
    Range {
        start: Box<Expr>,
        end: Box<Expr>,
        inclusive: bool,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructLiteralField {
    pub name: String,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Wildcard,
    Literal(Literal),
    Identifier(String),
    Enum {
        enum_name: String,
        variant: String,
        bindings: Vec<Pattern>,
    },
    Struct {
        name: String,
        fields: Vec<(String, Pattern)>,
    },
    TypeCheck(Type),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    String(String),
    Bool(bool),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Concat,
    Range,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let symbol = match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Mod => "%",
            BinaryOp::Pow => "^",
            BinaryOp::Eq => "==",
            BinaryOp::Ne => "!=",
            BinaryOp::Lt => "<",
            BinaryOp::Le => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::Ge => ">=",
            BinaryOp::And => "and",
            BinaryOp::Or => "or",
            BinaryOp::Concat => "..",
            BinaryOp::Range => "..",
        };
        write!(f, "{symbol}")
    }
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "-"),
            UnaryOp::Not => write!(f, "not"),
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Integer(value) => write!(f, "{value}"),
            Literal::Float(value) => write!(f, "{value}"),
            Literal::String(value) => write!(f, "\"{}\"", value.escape_default()),
            Literal::Bool(value) => write!(f, "{value}"),
        }
    }
}

impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Pattern::Wildcard => write!(f, "_"),
            Pattern::Literal(lit) => write!(f, "{lit}"),
            Pattern::Identifier(name) => write!(f, "{name}"),
            Pattern::Enum {
                enum_name,
                variant,
                bindings,
            } => {
                let qualified = if enum_name.is_empty() {
                    variant.clone()
                } else {
                    format!("{enum_name}.{variant}")
                };
                if bindings.is_empty() {
                    write!(f, "{qualified}")
                } else {
                    let binding_str = bindings
                        .iter()
                        .map(|p| p.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    write!(f, "{qualified}({binding_str})")
                }
            }

            Pattern::Struct { name, fields } => {
                let field_str = fields
                    .iter()
                    .map(|(field_name, pattern)| match pattern {
                        Pattern::Identifier(id) if id == field_name => field_name.clone(),
                        _ => format!("{field_name} = {pattern}"),
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "{name} {{ {field_str} }}")
            }

            Pattern::TypeCheck(ty) => write!(f, "as {ty}"),
        }
    }
}

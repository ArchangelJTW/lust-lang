use super::{Expr, Span, Type};
use alloc::{string::String, vec::Vec};
#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

impl Stmt {
    pub fn new(kind: StmtKind, span: Span) -> Self {
        Self { kind, span }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalBinding {
    pub name: String,
    pub type_annotation: Option<Type>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    Local {
        bindings: Vec<LocalBinding>,
        mutable: bool,
        initializer: Option<Vec<Expr>>,
    },
    Assign {
        targets: Vec<Expr>,
        values: Vec<Expr>,
    },
    CompoundAssign {
        target: Expr,
        op: super::expr::BinaryOp,
        value: Expr,
    },
    Expr(Expr),
    If {
        condition: Expr,
        then_block: Vec<Stmt>,
        elseif_branches: Vec<(Expr, Vec<Stmt>)>,
        else_block: Option<Vec<Stmt>>,
    },
    While {
        condition: Expr,
        body: Vec<Stmt>,
    },
    ForNumeric {
        variable: String,
        start: Expr,
        end: Expr,
        step: Option<Expr>,
        body: Vec<Stmt>,
    },
    ForIn {
        variables: Vec<String>,
        iterator: Expr,
        body: Vec<Stmt>,
    },
    Return(Vec<Expr>),
    Break,
    Continue,
    Block(Vec<Stmt>),
}

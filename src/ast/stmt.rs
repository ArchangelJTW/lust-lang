use super::{Expr, Span, Type};
use alloc::{boxed::Box, string::String, vec::Vec};
#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

impl Stmt {
    pub fn new(kind: StmtKind, span: Span) -> Self {
        Self { kind, span }
    }

    pub fn new_resume(value: Expr, span: Span) -> Self {
        Self::new(StmtKind::Expr(Expr::new_resume(value, span)), span)
    }

    pub fn new_perform(effect_name: String, args: Vec<Expr>, span: Span) -> Self {
        Self::new(StmtKind::Expr(Expr::new_perform(effect_name, args, span)), span)
    }

    pub fn new_handle(body: Expr, arms: Vec<super::expr::HandlerArm>, span: Span) -> Self {
        Self::new(StmtKind::Expr(Expr::new_handle(body, arms, span)), span)
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
    Defer(Box<Stmt>),
    ErrDefer(Box<Stmt>),
    Scoped(Vec<Stmt>),
}

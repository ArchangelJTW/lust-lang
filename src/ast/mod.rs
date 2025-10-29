pub mod expr;
pub mod items;
pub mod stmt;
pub mod types;
pub use expr::{BinaryOp, Expr, ExprKind, Literal, Pattern, StructLiteralField, UnaryOp};
pub use items::{
    EnumDef, EnumVariant, ExternItem, FieldOwnership, FunctionDef, FunctionParam, ImplBlock, Item,
    ItemKind, StructDef, StructField, TraitBound, TraitDef, TraitMethod, UseTree, UseTreeItem,
    Visibility,
};
pub use stmt::{LocalBinding, Stmt, StmtKind};
pub use types::{Type, TypeKind};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
}

impl Span {
    pub fn new(start_line: usize, start_col: usize, end_line: usize, end_col: usize) -> Self {
        Self {
            start_line,
            start_col,
            end_line,
            end_col,
        }
    }

    pub fn dummy() -> Self {
        Self {
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
        }
    }
}

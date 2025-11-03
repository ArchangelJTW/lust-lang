use crate::analysis::ModuleSnapshot;
use crate::utils::span_overlaps_range;
use lust::ast::{Expr, ExprKind, Item, ItemKind, Stmt, StmtKind, TypeKind};
use tower_lsp::lsp_types::{InlayHint, InlayHintKind, InlayHintLabel, Range};

pub(crate) fn collect_inlay_hints_for_module(
    module: &ModuleSnapshot,
    range: &Range,
) -> Vec<InlayHint> {
    let mut hints = Vec::new();
    for item in &module.module.items {
        collect_inlay_hints_from_item(item, module, range, &mut hints);
    }

    hints
}

fn collect_inlay_hints_from_item(
    item: &Item,
    module: &ModuleSnapshot,
    range: &Range,
    hints: &mut Vec<InlayHint>,
) {
    match &item.kind {
        ItemKind::Function(func) => {
            collect_inlay_hints_from_stmts(&func.body, module, range, hints)
        }

        ItemKind::Script(stmts) => collect_inlay_hints_from_stmts(stmts, module, range, hints),
        ItemKind::Impl(impl_block) => {
            for method in &impl_block.methods {
                collect_inlay_hints_from_stmts(&method.body, module, range, hints);
            }
        }

        ItemKind::Module { items, .. } => {
            for child in items {
                collect_inlay_hints_from_item(child, module, range, hints);
            }
        }

        _ => {}
    }
}

fn collect_inlay_hints_from_stmts(
    stmts: &[Stmt],
    module: &ModuleSnapshot,
    range: &Range,
    hints: &mut Vec<InlayHint>,
) {
    for stmt in stmts {
        collect_inlay_hints_from_stmt(stmt, module, range, hints);
    }
}

fn collect_inlay_hints_from_stmt(
    stmt: &Stmt,
    module: &ModuleSnapshot,
    range: &Range,
    hints: &mut Vec<InlayHint>,
) {
    match &stmt.kind {
        StmtKind::Local {
            bindings,
            initializer,
            ..
        } => {
            for binding in bindings {
                if binding.type_annotation.is_some() {
                    continue;
                }

                if !span_overlaps_range(binding.span, range) {
                    continue;
                }

                if let Some(ty) = module.variable_types.get(&binding.span) {
                    if matches!(ty.kind, TypeKind::Infer | TypeKind::Unknown) {
                        continue;
                    }

                    if binding.span.start_line == 0 {
                        continue;
                    }

                    let position = tower_lsp::lsp_types::Position::new(
                        binding.span.start_line.saturating_sub(1) as u32,
                        binding
                            .span
                            .start_col
                            .saturating_sub(1)
                            .saturating_add(binding.name.chars().count())
                            as u32,
                    );
                    let label: InlayHintLabel = format!(": {}", ty).into();
                    hints.push(InlayHint {
                        position,
                        label,
                        kind: Some(InlayHintKind::TYPE),
                        text_edits: None,
                        tooltip: None,
                        padding_left: Some(true),
                        padding_right: None,
                        data: None,
                    });
                }
            }

            if let Some(exprs) = initializer {
                for expr in exprs {
                    collect_inlay_hints_from_expr(expr, module, range, hints);
                }
            }
        }

        StmtKind::Assign { targets, values } => {
            for target in targets {
                collect_inlay_hints_from_expr(target, module, range, hints);
            }

            for value in values {
                collect_inlay_hints_from_expr(value, module, range, hints);
            }
        }

        StmtKind::CompoundAssign { target, value, .. } => {
            collect_inlay_hints_from_expr(target, module, range, hints);
            collect_inlay_hints_from_expr(value, module, range, hints);
        }

        StmtKind::Expr(expr) => collect_inlay_hints_from_expr(expr, module, range, hints),
        StmtKind::If {
            condition,
            then_block,
            elseif_branches,
            else_block,
        } => {
            collect_inlay_hints_from_expr(condition, module, range, hints);
            collect_inlay_hints_from_stmts(then_block, module, range, hints);
            for (expr, block) in elseif_branches {
                collect_inlay_hints_from_expr(expr, module, range, hints);
                collect_inlay_hints_from_stmts(block, module, range, hints);
            }

            if let Some(block) = else_block {
                collect_inlay_hints_from_stmts(block, module, range, hints);
            }
        }

        StmtKind::While { condition, body } => {
            collect_inlay_hints_from_expr(condition, module, range, hints);
            collect_inlay_hints_from_stmts(body, module, range, hints);
        }

        StmtKind::ForNumeric {
            start,
            end,
            step,
            body,
            ..
        } => {
            collect_inlay_hints_from_expr(start, module, range, hints);
            collect_inlay_hints_from_expr(end, module, range, hints);
            if let Some(step) = step {
                collect_inlay_hints_from_expr(step, module, range, hints);
            }

            collect_inlay_hints_from_stmts(body, module, range, hints);
        }

        StmtKind::ForIn { iterator, body, .. } => {
            collect_inlay_hints_from_expr(iterator, module, range, hints);
            collect_inlay_hints_from_stmts(body, module, range, hints);
        }

        StmtKind::Return(exprs) => {
            for expr in exprs {
                collect_inlay_hints_from_expr(expr, module, range, hints);
            }
        }

        StmtKind::Block(stmts) => collect_inlay_hints_from_stmts(stmts, module, range, hints),
        StmtKind::Break | StmtKind::Continue => {}
    }
}

fn collect_inlay_hints_from_expr(
    expr: &Expr,
    module: &ModuleSnapshot,
    range: &Range,
    hints: &mut Vec<InlayHint>,
) {
    match &expr.kind {
        ExprKind::Binary { left, right, .. } => {
            collect_inlay_hints_from_expr(left, module, range, hints);
            collect_inlay_hints_from_expr(right, module, range, hints);
        }

        ExprKind::Unary { operand, .. } => {
            collect_inlay_hints_from_expr(operand, module, range, hints);
        }

        ExprKind::Call { callee, args } => {
            collect_inlay_hints_from_expr(callee, module, range, hints);
            for arg in args {
                collect_inlay_hints_from_expr(arg, module, range, hints);
            }
        }

        ExprKind::MethodCall { receiver, args, .. } => {
            collect_inlay_hints_from_expr(receiver, module, range, hints);
            for arg in args {
                collect_inlay_hints_from_expr(arg, module, range, hints);
            }
        }

        ExprKind::FieldAccess { object, .. } => {
            collect_inlay_hints_from_expr(object, module, range, hints);
        }

        ExprKind::Index { object, index } => {
            collect_inlay_hints_from_expr(object, module, range, hints);
            collect_inlay_hints_from_expr(index, module, range, hints);
        }

        ExprKind::Array(elements) | ExprKind::Tuple(elements) => {
            for element in elements {
                collect_inlay_hints_from_expr(element, module, range, hints);
            }
        }

        ExprKind::Map(entries) => {
            for (key, value) in entries {
                collect_inlay_hints_from_expr(key, module, range, hints);
                collect_inlay_hints_from_expr(value, module, range, hints);
            }
        }

        ExprKind::StructLiteral { fields, .. } => {
            for field in fields {
                collect_inlay_hints_from_expr(&field.value, module, range, hints);
            }
        }

        ExprKind::EnumConstructor { args, .. } => {
            for arg in args {
                collect_inlay_hints_from_expr(arg, module, range, hints);
            }
        }

        ExprKind::Lambda { body, .. } => {
            collect_inlay_hints_from_expr(body, module, range, hints);
        }

        ExprKind::Paren(inner)
        | ExprKind::Cast { expr: inner, .. }
        | ExprKind::TypeCheck { expr: inner, .. }
        | ExprKind::IsPattern { expr: inner, .. } => {
            collect_inlay_hints_from_expr(inner, module, range, hints);
        }

        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_inlay_hints_from_expr(condition, module, range, hints);
            collect_inlay_hints_from_expr(then_branch, module, range, hints);
            if let Some(else_branch) = else_branch {
                collect_inlay_hints_from_expr(else_branch, module, range, hints);
            }
        }

        ExprKind::Block(stmts) => collect_inlay_hints_from_stmts(stmts, module, range, hints),
        ExprKind::Range { start, end, .. } => {
            collect_inlay_hints_from_expr(start, module, range, hints);
            collect_inlay_hints_from_expr(end, module, range, hints);
        }

        ExprKind::Return(values) => {
            for value in values {
                collect_inlay_hints_from_expr(value, module, range, hints);
            }
        }

        ExprKind::Literal(_) | ExprKind::Identifier(_) => {}
    }
}

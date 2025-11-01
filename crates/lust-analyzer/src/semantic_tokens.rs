use crate::utils::{simple_type_name, span_for_identifier};
use hashbrown::{HashMap, HashSet};
use lust::ast::{Expr, ExprKind, ExternItem, Item, ItemKind, Stmt, StmtKind, Type, TypeKind};
use lust::modules::LoadedModule;
use lust::Span;
use tower_lsp::lsp_types::{SemanticToken, SemanticTokenType};
pub(crate) const SEMANTIC_TOKEN_TYPES: &[SemanticTokenType] = &[
    SemanticTokenType::TYPE,
    SemanticTokenType::STRUCT,
    SemanticTokenType::ENUM,
    SemanticTokenType::INTERFACE,
];
pub(crate) const TOKEN_TYPE_TYPE_IDX: u32 = 0;
pub(crate) const TOKEN_TYPE_STRUCT_IDX: u32 = 1;
pub(crate) const TOKEN_TYPE_ENUM_IDX: u32 = 2;
pub(crate) const TOKEN_TYPE_TRAIT_IDX: u32 = 3;
#[derive(Clone, Copy)]
struct RawSemanticToken {
    line: u32,
    start: u32,
    length: u32,
    token_type: u32,
    modifiers: u32,
}

pub(crate) fn collect_semantic_tokens_for_module(
    module: &LoadedModule,
    expr_types: &HashMap<Span, Type>,
    source: &str,
) -> Vec<SemanticToken> {
    let mut raw_tokens = Vec::new();
    let mut seen = HashSet::new();
    let line_offsets = crate::utils::compute_line_offsets(source);
    collect_tokens_from_items(
        &module.items,
        source,
        &line_offsets,
        &mut raw_tokens,
        &mut seen,
    );
    for (span, ty) in expr_types {
        if let TypeKind::Named(name) = &ty.kind {
            push_identifier_token(
                source,
                &line_offsets,
                &mut raw_tokens,
                &mut seen,
                *span,
                name,
                span.start_col.saturating_sub(1),
                TOKEN_TYPE_TYPE_IDX,
            );
        }
    }

    encode_semantic_tokens(raw_tokens)
}

fn encode_semantic_tokens(mut raw: Vec<RawSemanticToken>) -> Vec<SemanticToken> {
    raw.sort_by(|a, b| (a.line, a.start).cmp(&(b.line, b.start)));
    let mut tokens = Vec::with_capacity(raw.len());
    let mut prev_line = 0u32;
    let mut prev_start = 0u32;
    for token in raw {
        let delta_line = token.line.saturating_sub(prev_line);
        let delta_start = if delta_line == 0 {
            token.start.saturating_sub(prev_start)
        } else {
            token.start
        };
        tokens.push(SemanticToken {
            delta_line,
            delta_start,
            length: token.length,
            token_type: token.token_type,
            token_modifiers_bitset: token.modifiers,
        });
        prev_line = token.line;
        prev_start = token.start;
    }

    tokens
}

fn push_token_for_span(
    tokens: &mut Vec<RawSemanticToken>,
    seen: &mut HashSet<(Span, u32)>,
    span: Span,
    token_type: u32,
    length_override: Option<u32>,
) {
    if span.start_line == 0 || span.start_col == 0 {
        return;
    }

    if span.start_line != span.end_line {
        return;
    }

    let key = (span, token_type);
    if !seen.insert(key) {
        return;
    }

    let line = span.start_line.saturating_sub(1) as u32;
    let start = span.start_col.saturating_sub(1) as u32;
    let length = if let Some(len) = length_override {
        len.max(1)
    } else {
        let end = span.end_col.max(span.start_col);
        end.saturating_sub(span.start_col).saturating_add(1) as u32
    };
    if length == 0 {
        return;
    }

    tokens.push(RawSemanticToken {
        line,
        start,
        length,
        token_type,
        modifiers: 0,
    });
}

fn collect_tokens_from_items(
    items: &[Item],
    text: &str,
    line_offsets: &[usize],
    tokens: &mut Vec<RawSemanticToken>,
    seen: &mut HashSet<(Span, u32)>,
) {
    for item in items {
        collect_tokens_from_item(item, text, line_offsets, tokens, seen);
    }
}

fn collect_tokens_from_item(
    item: &Item,
    text: &str,
    line_offsets: &[usize],
    tokens: &mut Vec<RawSemanticToken>,
    seen: &mut HashSet<(Span, u32)>,
) {
    match &item.kind {
        ItemKind::Struct(def) => {
            add_definition_token(
                text,
                line_offsets,
                tokens,
                seen,
                item.span,
                simple_type_name(&def.name),
                TOKEN_TYPE_STRUCT_IDX,
                "struct".len(),
            );
            for field in &def.fields {
                collect_tokens_from_type(&field.ty, text, line_offsets, tokens, seen);
            }
        }

        ItemKind::Enum(def) => {
            add_definition_token(
                text,
                line_offsets,
                tokens,
                seen,
                item.span,
                simple_type_name(&def.name),
                TOKEN_TYPE_ENUM_IDX,
                "enum".len(),
            );
            for variant in &def.variants {
                if let Some(fields) = &variant.fields {
                    for ty in fields {
                        collect_tokens_from_type(ty, text, line_offsets, tokens, seen);
                    }
                }
            }
        }

        ItemKind::Trait(def) => {
            add_definition_token(
                text,
                line_offsets,
                tokens,
                seen,
                item.span,
                simple_type_name(&def.name),
                TOKEN_TYPE_TRAIT_IDX,
                "trait".len(),
            );
            for method in &def.methods {
                for param in &method.params {
                    collect_tokens_from_type(&param.ty, text, line_offsets, tokens, seen);
                }

                if let Some(ret) = &method.return_type {
                    collect_tokens_from_type(ret, text, line_offsets, tokens, seen);
                }
            }
        }

        ItemKind::Function(func) => {
            for param in &func.params {
                collect_tokens_from_type(&param.ty, text, line_offsets, tokens, seen);
            }

            if let Some(ret) = &func.return_type {
                collect_tokens_from_type(ret, text, line_offsets, tokens, seen);
            }

            collect_tokens_from_stmts(&func.body, text, line_offsets, tokens, seen);
        }

        ItemKind::Impl(impl_block) => {
            collect_tokens_from_type(&impl_block.target_type, text, line_offsets, tokens, seen);
            for method in &impl_block.methods {
                for param in &method.params {
                    collect_tokens_from_type(&param.ty, text, line_offsets, tokens, seen);
                }

                if let Some(ret) = &method.return_type {
                    collect_tokens_from_type(ret, text, line_offsets, tokens, seen);
                }

                collect_tokens_from_stmts(&method.body, text, line_offsets, tokens, seen);
            }
        }

        ItemKind::TypeAlias { target, .. } => {
            collect_tokens_from_type(target, text, line_offsets, tokens, seen);
        }

        ItemKind::Const { ty, .. } => {
            collect_tokens_from_type(ty, text, line_offsets, tokens, seen);
        }

        ItemKind::Static { ty, .. } => {
            collect_tokens_from_type(ty, text, line_offsets, tokens, seen);
        }

        ItemKind::Module { items, .. } => {
            collect_tokens_from_items(items, text, line_offsets, tokens, seen);
        }

        ItemKind::Script(stmts) => {
            collect_tokens_from_stmts(stmts, text, line_offsets, tokens, seen);
        }

        ItemKind::Extern { items, .. } => {
            for ExternItem::Function {
                params,
                return_type,
                ..
            } in items
            {
                for ty in params {
                    collect_tokens_from_type(ty, text, line_offsets, tokens, seen);
                }

                if let Some(ret) = return_type {
                    collect_tokens_from_type(ret, text, line_offsets, tokens, seen);
                }
            }
        }

        _ => {}
    }
}

fn add_definition_token(
    text: &str,
    line_offsets: &[usize],
    tokens: &mut Vec<RawSemanticToken>,
    seen: &mut HashSet<(Span, u32)>,
    span: Span,
    name: &str,
    token_type: u32,
    keyword_len: usize,
) {
    let approx_col = span.start_col.saturating_sub(1) + keyword_len;
    push_identifier_token(
        text,
        line_offsets,
        tokens,
        seen,
        span,
        name,
        approx_col,
        token_type,
    );
}

fn collect_tokens_from_stmts(
    stmts: &[Stmt],
    text: &str,
    line_offsets: &[usize],
    tokens: &mut Vec<RawSemanticToken>,
    seen: &mut HashSet<(Span, u32)>,
) {
    for stmt in stmts {
        collect_tokens_from_stmt(stmt, text, line_offsets, tokens, seen);
    }
}

fn collect_tokens_from_stmt(
    stmt: &Stmt,
    text: &str,
    line_offsets: &[usize],
    tokens: &mut Vec<RawSemanticToken>,
    seen: &mut HashSet<(Span, u32)>,
) {
    match &stmt.kind {
        StmtKind::Local {
            bindings,
            initializer,
            ..
        } => {
            for binding in bindings {
                if let Some(ty) = &binding.type_annotation {
                    collect_tokens_from_type(ty, text, line_offsets, tokens, seen);
                }
            }

            if let Some(values) = initializer {
                for expr in values {
                    collect_tokens_from_expr(expr, text, line_offsets, tokens, seen);
                }
            }
        }

        StmtKind::Assign { targets, values } => {
            for expr in targets.iter().chain(values.iter()) {
                collect_tokens_from_expr(expr, text, line_offsets, tokens, seen);
            }
        }

        StmtKind::CompoundAssign { target, value, .. } => {
            collect_tokens_from_expr(target, text, line_offsets, tokens, seen);
            collect_tokens_from_expr(value, text, line_offsets, tokens, seen);
        }

        StmtKind::Expr(expr) => {
            collect_tokens_from_expr(expr, text, line_offsets, tokens, seen);
        }

        StmtKind::Return(exprs) => {
            for expr in exprs {
                collect_tokens_from_expr(expr, text, line_offsets, tokens, seen);
            }
        }

        StmtKind::If {
            condition,
            then_block,
            elseif_branches,
            else_block,
        } => {
            collect_tokens_from_expr(condition, text, line_offsets, tokens, seen);
            collect_tokens_from_stmts(then_block, text, line_offsets, tokens, seen);
            for (expr, block) in elseif_branches {
                collect_tokens_from_expr(expr, text, line_offsets, tokens, seen);
                collect_tokens_from_stmts(block, text, line_offsets, tokens, seen);
            }

            if let Some(block) = else_block {
                collect_tokens_from_stmts(block, text, line_offsets, tokens, seen);
            }
        }

        StmtKind::While { condition, body } => {
            collect_tokens_from_expr(condition, text, line_offsets, tokens, seen);
            collect_tokens_from_stmts(body, text, line_offsets, tokens, seen);
        }

        StmtKind::ForNumeric {
            start,
            end,
            step,
            body,
            ..
        } => {
            collect_tokens_from_expr(start, text, line_offsets, tokens, seen);
            collect_tokens_from_expr(end, text, line_offsets, tokens, seen);
            if let Some(step) = step {
                collect_tokens_from_expr(step, text, line_offsets, tokens, seen);
            }

            collect_tokens_from_stmts(body, text, line_offsets, tokens, seen);
        }

        StmtKind::ForIn { iterator, body, .. } => {
            collect_tokens_from_expr(iterator, text, line_offsets, tokens, seen);
            collect_tokens_from_stmts(body, text, line_offsets, tokens, seen);
        }

        StmtKind::Block(stmts) => {
            collect_tokens_from_stmts(stmts, text, line_offsets, tokens, seen);
        }

        StmtKind::Break | StmtKind::Continue => {}
    }
}

fn collect_tokens_from_expr(
    expr: &Expr,
    text: &str,
    line_offsets: &[usize],
    tokens: &mut Vec<RawSemanticToken>,
    seen: &mut HashSet<(Span, u32)>,
) {
    match &expr.kind {
        ExprKind::Array(elements) | ExprKind::Tuple(elements) => {
            for element in elements {
                collect_tokens_from_expr(element, text, line_offsets, tokens, seen);
            }
        }

        ExprKind::Map(entries) => {
            for (key, value) in entries {
                collect_tokens_from_expr(key, text, line_offsets, tokens, seen);
                collect_tokens_from_expr(value, text, line_offsets, tokens, seen);
            }
        }

        ExprKind::Binary { left, right, .. } => {
            collect_tokens_from_expr(left, text, line_offsets, tokens, seen);
            collect_tokens_from_expr(right, text, line_offsets, tokens, seen);
        }

        ExprKind::Unary { operand, .. } => {
            collect_tokens_from_expr(operand, text, line_offsets, tokens, seen);
        }

        ExprKind::Call { callee, args } => {
            collect_tokens_from_expr(callee, text, line_offsets, tokens, seen);
            for arg in args {
                collect_tokens_from_expr(arg, text, line_offsets, tokens, seen);
            }
        }

        ExprKind::MethodCall { receiver, args, .. } => {
            collect_tokens_from_expr(receiver, text, line_offsets, tokens, seen);
            for arg in args {
                collect_tokens_from_expr(arg, text, line_offsets, tokens, seen);
            }
        }

        ExprKind::FieldAccess { object, .. } => {
            collect_tokens_from_expr(object, text, line_offsets, tokens, seen);
        }

        ExprKind::Index { object, index } => {
            collect_tokens_from_expr(object, text, line_offsets, tokens, seen);
            collect_tokens_from_expr(index, text, line_offsets, tokens, seen);
        }

        ExprKind::StructLiteral { fields, .. } => {
            for field in fields {
                collect_tokens_from_expr(&field.value, text, line_offsets, tokens, seen);
            }
        }

        ExprKind::EnumConstructor {
            enum_name, args, ..
        } => {
            push_identifier_token(
                text,
                line_offsets,
                tokens,
                seen,
                expr.span,
                enum_name,
                expr.span.start_col.saturating_sub(1),
                TOKEN_TYPE_ENUM_IDX,
            );
            for arg in args {
                collect_tokens_from_expr(arg, text, line_offsets, tokens, seen);
            }
        }

        ExprKind::Return(values) => {
            for value in values {
                collect_tokens_from_expr(value, text, line_offsets, tokens, seen);
            }
        }

        ExprKind::Lambda { body, .. } => {
            collect_tokens_from_expr(body, text, line_offsets, tokens, seen);
        }

        ExprKind::Paren(inner)
        | ExprKind::Cast { expr: inner, .. }
        | ExprKind::TypeCheck { expr: inner, .. }
        | ExprKind::IsPattern { expr: inner, .. } => {
            collect_tokens_from_expr(inner, text, line_offsets, tokens, seen);
        }

        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_tokens_from_expr(condition, text, line_offsets, tokens, seen);
            collect_tokens_from_expr(then_branch, text, line_offsets, tokens, seen);
            if let Some(else_branch) = else_branch {
                collect_tokens_from_expr(else_branch, text, line_offsets, tokens, seen);
            }
        }

        ExprKind::Block(stmts) => {
            collect_tokens_from_stmts(stmts, text, line_offsets, tokens, seen);
        }

        ExprKind::Range { start, end, .. } => {
            collect_tokens_from_expr(start, text, line_offsets, tokens, seen);
            collect_tokens_from_expr(end, text, line_offsets, tokens, seen);
        }

        ExprKind::Literal(_) | ExprKind::Identifier(_) => {}
    }
}

fn collect_tokens_from_type(
    ty: &Type,
    text: &str,
    line_offsets: &[usize],
    tokens: &mut Vec<RawSemanticToken>,
    seen: &mut HashSet<(Span, u32)>,
) {
    match &ty.kind {
        TypeKind::Named(name) => {
            push_identifier_token(
                text,
                line_offsets,
                tokens,
                seen,
                ty.span,
                name,
                ty.span.start_col.saturating_sub(1),
                TOKEN_TYPE_TYPE_IDX,
            );
        }

        TypeKind::Trait(name) => {
            push_identifier_token(
                text,
                line_offsets,
                tokens,
                seen,
                ty.span,
                name,
                ty.span.start_col.saturating_sub(1),
                TOKEN_TYPE_TRAIT_IDX,
            );
        }

        TypeKind::Option(inner) => {
            collect_tokens_from_type(inner, text, line_offsets, tokens, seen);
        }

        TypeKind::Ref(inner) | TypeKind::MutRef(inner) => {
            collect_tokens_from_type(inner, text, line_offsets, tokens, seen);
        }

        TypeKind::Result(ok, err) => {
            collect_tokens_from_type(ok, text, line_offsets, tokens, seen);
            collect_tokens_from_type(err, text, line_offsets, tokens, seen);
        }

        TypeKind::Tuple(types) | TypeKind::Union(types) => {
            for inner in types {
                collect_tokens_from_type(inner, text, line_offsets, tokens, seen);
            }
        }

        TypeKind::Map(key, value) => {
            collect_tokens_from_type(key, text, line_offsets, tokens, seen);
            collect_tokens_from_type(value, text, line_offsets, tokens, seen);
        }

        TypeKind::Pointer { pointee, .. } => {
            collect_tokens_from_type(pointee, text, line_offsets, tokens, seen);
        }

        TypeKind::Function {
            params,
            return_type,
        } => {
            for param in params {
                collect_tokens_from_type(param, text, line_offsets, tokens, seen);
            }

            collect_tokens_from_type(return_type, text, line_offsets, tokens, seen);
        }

        TypeKind::GenericInstance { name, type_args } => {
            push_identifier_token(
                text,
                line_offsets,
                tokens,
                seen,
                ty.span,
                name,
                ty.span.start_col.saturating_sub(1),
                TOKEN_TYPE_TYPE_IDX,
            );
            for inner in type_args {
                collect_tokens_from_type(inner, text, line_offsets, tokens, seen);
            }
        }

        TypeKind::Array(inner) => {
            collect_tokens_from_type(inner, text, line_offsets, tokens, seen);
        }

        TypeKind::Table
        | TypeKind::String
        | TypeKind::Int
        | TypeKind::Float
        | TypeKind::Bool
        | TypeKind::Unknown
        | TypeKind::Infer
        | TypeKind::Unit
        | TypeKind::Generic(_)
        | TypeKind::TraitBound(_) => {}
    }
}

fn push_identifier_token(
    text: &str,
    line_offsets: &[usize],
    tokens: &mut Vec<RawSemanticToken>,
    seen: &mut HashSet<(Span, u32)>,
    span: Span,
    name: &str,
    approx_col: usize,
    token_type: u32,
) {
    if let Some(id_span) = span_for_identifier(text, line_offsets, span, name, approx_col) {
        push_token_for_span(
            tokens,
            seen,
            id_span,
            token_type,
            Some(name.chars().count() as u32),
        );
        return;
    }

    if let Some(simple) = name.rsplit('.').next() {
        if simple != name {
            if let Some(id_span) = span_for_identifier(text, line_offsets, span, simple, approx_col)
            {
                push_token_for_span(
                    tokens,
                    seen,
                    id_span,
                    token_type,
                    Some(simple.chars().count() as u32),
                );
            }
        }
    }
}

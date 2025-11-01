use crate::analysis::{
    choose_definition, find_type_for_position, hover_from_definition, location_from_definition,
    AnalysisSnapshot, MethodInfo, ModuleSnapshot, TypeDefinitionKind,
};
use crate::diagnostics::error_to_diagnostics;
use crate::semantic_tokens::SEMANTIC_TOKEN_TYPES;
use crate::utils::{
    analyzer_lust_config, base_type_name, char_at_index, compute_line_offsets,
    extract_word_at_position, identifier_name_at_span, identifier_prefix_range,
    identifier_range_before, identifier_text, is_identifier_char, is_word_char,
    method_display_name, nth_char_byte_index, offset_to_position, position_to_offset,
    prev_char_index, qualify_type_name, simple_type_name, span_contains_position,
    span_from_identifier, span_overlaps_range, span_start_before_or_equal, span_starts_after,
    span_to_range, split_type_member,
};
use hashbrown::{HashMap, HashSet};
use lust::ast::{
    Expr, ExprKind, ExternItem, FunctionParam, Item, ItemKind, Stmt, StmtKind, Type, TypeKind,
    Visibility,
};
use lust::builtins::{self, BuiltinFunction, BuiltinMethod, TypeExpr};
use lust::{Compiler, ModuleLoader, Span, TypeChecker};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::{
    CompletionItem, CompletionItemKind, CompletionOptions, CompletionParams, CompletionResponse,
    CompletionTriggerKind, Diagnostic, Documentation, GotoDefinitionParams, GotoDefinitionResponse,
    Hover, HoverContents, HoverParams, HoverProviderCapability, InitializeParams, InitializeResult,
    InitializedParams, InlayHint, InlayHintKind, InlayHintLabel, InlayHintOptions, InlayHintParams,
    InlayHintServerCapabilities, InsertTextFormat, MarkupContent, MarkupKind, MessageType, OneOf,
    Position, Range, SemanticToken, SemanticTokens, SemanticTokensFullOptions,
    SemanticTokensLegend, SemanticTokensOptions, SemanticTokensParams, SemanticTokensRangeParams,
    SemanticTokensRangeResult, SemanticTokensResult, SemanticTokensServerCapabilities,
    ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind, TextDocumentSyncOptions,
};
use tower_lsp::{async_trait, Client, LanguageServer, LspService, Server};
use url::Url;
#[derive(Clone)]
struct DocumentState {
    text: String,
    version: i32,
}

enum CompletionKind {
    Member,
    Method,
    Pattern,
    Identifier,
    ModulePath,
}

struct CompletionContext {
    kind: CompletionKind,
    object_start: Option<usize>,
    object_end: usize,
    object_name: Option<String>,
    prefix: String,
    path_segments: Vec<String>,
}

fn format_method_signature(method: &MethodInfo) -> String {
    let params = method
        .params
        .iter()
        .map(|param| {
            if param.is_self || param.name == "self" {
                "self".to_string()
            } else if matches!(param.ty.kind, TypeKind::Infer) {
                param.name.clone()
            } else {
                format!("{}: {}", param.name, param.ty)
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    if let Some(ret) = &method.return_type {
        format!("fn {}({}) -> {}", method.name, params, ret)
    } else {
        format!("fn {}({})", method.name, params)
    }
}

fn method_insert_text(method: &MethodInfo) -> (String, Option<InsertTextFormat>) {
    let params: Vec<&FunctionParam> = method
        .params
        .iter()
        .filter(|p| !(p.is_self || p.name == "self"))
        .collect();
    if params.is_empty() {
        (format!("{}()", method.name), None)
    } else {
        let parts = params
            .iter()
            .enumerate()
            .map(|(idx, param)| format!("${{{}:{}}}", idx + 1, param.name))
            .collect::<Vec<_>>()
            .join(", ");
        (
            format!("{}({})", method.name, parts),
            Some(InsertTextFormat::SNIPPET),
        )
    }
}

fn format_function_signature_def(
    name: &str,
    params: &[FunctionParam],
    return_type: &Option<Type>,
) -> String {
    let params_str = params
        .iter()
        .map(|param| {
            if param.is_self || param.name == "self" {
                "self".to_string()
            } else if matches!(param.ty.kind, TypeKind::Infer) {
                param.name.clone()
            } else {
                format!("{}: {}", param.name, param.ty)
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    if let Some(ret) = return_type {
        format!("fn {}({}) -> {}", name, params_str, ret)
    } else {
        format!("fn {}({})", name, params_str)
    }
}

fn type_completion_kind(snapshot: &AnalysisSnapshot, qualified: &str) -> CompletionItemKind {
    if snapshot.has_struct(qualified) {
        CompletionItemKind::STRUCT
    } else if snapshot.has_enum(qualified) {
        CompletionItemKind::ENUM
    } else {
        CompletionItemKind::CLASS
    }
}

fn method_visible(method: &MethodInfo, module_path: Option<&str>) -> bool {
    match method.visibility {
        Visibility::Public => true,
        Visibility::Private => module_path
            .map(|module| module == method.module_path)
            .unwrap_or(false),
    }
}

fn struct_field_completions(
    snapshot: &AnalysisSnapshot,
    module_path: Option<&str>,
    type_name: &str,
    prefix: &str,
) -> Vec<CompletionItem> {
    let mut items = Vec::new();
    if let Some(info) = snapshot.struct_info_for(type_name, module_path) {
        for field in &info.def.fields {
            if !prefix.is_empty() && !field.name.starts_with(prefix) {
                continue;
            }

            let mut item = CompletionItem::default();
            item.label = field.name.clone();
            item.kind = Some(CompletionItemKind::FIELD);
            item.detail = Some(field.ty.to_string());
            items.push(item);
        }
    }

    items
}

fn builtin_enum_variants(type_name: &str) -> Option<Vec<(&'static str, usize)>> {
    match type_name {
        "Option" => Some(vec![("Some", 1), ("None", 0)]),
        "Result" => Some(vec![("Ok", 1), ("Err", 1)]),
        "TaskStatus" => Some(vec![
            ("Ready", 0),
            ("Running", 0),
            ("Yielded", 0),
            ("Completed", 0),
            ("Failed", 0),
            ("Stopped", 0),
        ]),
        _ => None,
    }
}

fn enum_variant_completions(
    snapshot: &AnalysisSnapshot,
    module_path: Option<&str>,
    type_name: &str,
    prefix: &str,
) -> Vec<CompletionItem> {
    let mut items = Vec::new();
    if let Some(info) = snapshot.enum_info_for(type_name, module_path) {
        for variant in &info.def.variants {
            if !prefix.is_empty() && !variant.name.starts_with(prefix) {
                continue;
            }

            let mut item = CompletionItem::default();
            item.label = variant.name.clone();
            item.kind = Some(CompletionItemKind::ENUM_MEMBER);
            if let Some(fields) = &variant.fields {
                let field_types = fields
                    .iter()
                    .map(|ty| ty.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                item.detail = Some(format!("{}({})", variant.name, field_types));
                let snippet_args = fields
                    .iter()
                    .enumerate()
                    .map(|(idx, _)| format!("${{{}:arg{}}}", idx + 1, idx + 1))
                    .collect::<Vec<_>>()
                    .join(", ");
                let snippet = format!("{}({})", variant.name, snippet_args);
                item.insert_text = Some(snippet);
                item.insert_text_format = Some(InsertTextFormat::SNIPPET);
            } else {
                item.detail = Some(format!("{} (unit variant)", variant.name));
                item.insert_text = Some(variant.name.clone());
            }

            items.push(item);
        }

        return items;
    }

    if let Some(builtin) = builtin_enum_variants(simple_type_name(type_name)) {
        for (variant_name, arity) in builtin {
            if !prefix.is_empty() && !variant_name.starts_with(prefix) {
                continue;
            }

            let mut item = CompletionItem::default();
            item.label = variant_name.to_string();
            item.kind = Some(CompletionItemKind::ENUM_MEMBER);
            if arity > 0 {
                item.detail = Some(format!("{} variant", variant_name));
                let snippet_args = (0..arity)
                    .map(|idx| format!("${{{}:arg{}}}", idx + 1, idx + 1))
                    .collect::<Vec<_>>()
                    .join(", ");
                let snippet = format!("{}({})", variant_name, snippet_args);
                item.insert_text = Some(snippet);
                item.insert_text_format = Some(InsertTextFormat::SNIPPET);
            } else {
                item.detail = Some(format!("{} (unit variant)", variant_name));
                item.insert_text = Some(variant_name.to_string());
            }

            items.push(item);
        }
    }

    items
}

fn identifier_completions(
    module: &ModuleSnapshot,
    snapshot: &AnalysisSnapshot,
    file_path: &Path,
    position: Position,
    prefix: &str,
) -> Vec<CompletionItem> {
    let mut items = Vec::new();
    let mut seen = HashSet::new();
    let mut push_item = |name: String, kind: CompletionItemKind, detail: Option<String>| {
        if !prefix.is_empty() && !name.starts_with(prefix) {
            return;
        }

        if !seen.insert(name.clone()) {
            return;
        }

        let mut item = CompletionItem::default();
        item.label = name;
        item.kind = Some(kind);
        if let Some(detail) = detail {
            if !detail.is_empty() {
                item.detail = Some(detail);
            }
        }

        items.push(item);
    };
    if let Some(scope) = find_scope(&module.module.items, &position) {
        let locals = collect_locals_from_scope(&scope, module, &position);
        for local in locals {
            push_item(local.name, CompletionItemKind::VARIABLE, local.type_detail);
        }

        if let ScopeContext::Function { params, .. } = &scope {
            for param in *params {
                if param.name.is_empty() {
                    continue;
                }

                let detail = if matches!(param.ty.kind, TypeKind::Infer | TypeKind::Unknown) {
                    None
                } else {
                    Some(param.ty.to_string())
                };
                push_item(param.name.clone(), CompletionItemKind::VARIABLE, detail);
            }
        }
    }

    let imports = &module.module.imports;
    for (alias, target) in &imports.function_aliases {
        push_item(
            alias.clone(),
            CompletionItemKind::FUNCTION,
            Some(target.clone()),
        );
    }

    for (alias, target) in &imports.type_aliases {
        let kind = type_completion_kind(snapshot, target);
        push_item(alias.clone(), kind, Some(target.clone()));
    }

    for (alias, target) in &imports.module_aliases {
        push_item(
            alias.clone(),
            CompletionItemKind::MODULE,
            Some(target.clone()),
        );
    }

    for item in &module.module.items {
        match &item.kind {
            ItemKind::Function(func) => {
                if func.is_method || func.name.contains(':') {
                    continue;
                }

                let simple = method_display_name(&func.name);
                let detail =
                    format_function_signature_def(&simple, &func.params, &func.return_type);
                push_item(simple, CompletionItemKind::FUNCTION, Some(detail));
            }

            ItemKind::Struct(def) => {
                let simple = simple_type_name(&def.name).to_string();
                push_item(
                    simple,
                    CompletionItemKind::STRUCT,
                    Some("struct".to_string()),
                );
            }

            ItemKind::Enum(def) => {
                let simple = simple_type_name(&def.name).to_string();
                push_item(simple, CompletionItemKind::ENUM, Some("enum".to_string()));
            }

            ItemKind::Trait(def) => {
                let simple = simple_type_name(&def.name).to_string();
                push_item(
                    simple,
                    CompletionItemKind::INTERFACE,
                    Some("trait".to_string()),
                );
            }

            ItemKind::TypeAlias { name, .. } => {
                push_item(
                    name.clone(),
                    CompletionItemKind::CLASS,
                    Some("type alias".to_string()),
                );
            }

            ItemKind::Const { name, .. } => {
                push_item(
                    name.clone(),
                    CompletionItemKind::CONSTANT,
                    Some("const".to_string()),
                );
            }

            ItemKind::Static { name, .. } => {
                push_item(
                    name.clone(),
                    CompletionItemKind::VARIABLE,
                    Some("static".to_string()),
                );
            }

            ItemKind::Module { name, .. } => {
                push_item(
                    name.clone(),
                    CompletionItemKind::MODULE,
                    Some("module".to_string()),
                );
            }

            ItemKind::Extern {
                items: extern_items,
                ..
            } => {
                for ext in extern_items {
                    match ext {
                        ExternItem::Function { name, .. } => {
                            push_item(
                                name.clone(),
                                CompletionItemKind::FUNCTION,
                                Some("extern".to_string()),
                            );
                        }
                    }
                }
            }

            ItemKind::Script(_) | ItemKind::Impl(_) | ItemKind::Use { .. } => {}
        }
    }

    if let Some(defs) = snapshot.definitions_in_file(file_path) {
        for def in defs {
            let kind = match def.kind {
                TypeDefinitionKind::Struct => CompletionItemKind::STRUCT,
                TypeDefinitionKind::Enum => CompletionItemKind::ENUM,
                TypeDefinitionKind::Trait => CompletionItemKind::INTERFACE,
            };
            push_item(def.name.clone(), kind, Some(def.layout.clone()));
        }
    }

    let database = builtins::builtins();
    for func in database.global_functions() {
        let label = func.name;
        if !prefix.is_empty() && !label.starts_with(prefix) {
            continue;
        }

        let owned = label.to_string();
        if !seen.insert(owned.clone()) {
            continue;
        }

        let mut item = function_completion(func, None);
        item.label = owned;
        items.push(item);
    }

    for module in database.modules() {
        let label = module.name().to_string();
        if !prefix.is_empty() && !label.starts_with(prefix) {
            continue;
        }

        if !seen.insert(label.clone()) {
            continue;
        }

        let mut item = CompletionItem::default();
        item.label = label;
        item.kind = Some(CompletionItemKind::MODULE);
        item.detail = Some(module.description().to_string());
        item.documentation = Some(Documentation::MarkupContent(MarkupContent {
            kind: MarkupKind::Markdown,
            value: module.description().to_string(),
        }));
        items.push(item);
    }

    for def in snapshot.all_type_definitions() {
        if !def.file_path.as_os_str().is_empty() {
            continue;
        }

        let label = def.name.clone();
        if !prefix.is_empty() && !label.starts_with(prefix) {
            continue;
        }

        if !seen.insert(label.clone()) {
            continue;
        }

        let kind = match def.kind {
            TypeDefinitionKind::Struct => CompletionItemKind::STRUCT,
            TypeDefinitionKind::Enum => CompletionItemKind::ENUM,
            TypeDefinitionKind::Trait => CompletionItemKind::INTERFACE,
        };
        let mut item = CompletionItem::default();
        item.label = label;
        item.kind = Some(kind);
        if !def.layout.is_empty() {
            item.detail = Some(def.layout.clone());
            item.documentation = completion_documentation(&def.layout, "");
        }
        items.push(item);
    }

    items
}

fn resolve_module_path_for_segments(
    snapshot: &AnalysisSnapshot,
    module: &ModuleSnapshot,
    segments: &[String],
) -> Option<String> {
    if segments.is_empty() {
        return Some(String::new());
    }

    let mut resolved_segments: Vec<String> = Vec::new();
    if let Some(real) = module.module.imports.module_aliases.get(&segments[0]) {
        resolved_segments.extend(real.split('.').map(|s| s.to_string()));
        resolved_segments.extend(segments.iter().skip(1).cloned());
    } else {
        resolved_segments.extend(segments.iter().cloned());
    }

    let resolved_path = resolved_segments.join(".");
    if resolved_path.is_empty()
        || snapshot.module_for_name(&resolved_path).is_some()
        || snapshot.module_children(&resolved_path).is_some()
    {
        Some(resolved_path)
    } else {
        None
    }
}

fn module_alias_member_completions(
    snapshot: &AnalysisSnapshot,
    module: &ModuleSnapshot,
    prefix: &str,
    object_name: &str,
) -> Vec<CompletionItem> {
    let mut items = Vec::new();
    let mut seen = HashSet::new();
    let segments: Vec<String> = object_name
        .split('.')
        .filter(|segment| !segment.is_empty())
        .map(|segment| segment.to_string())
        .collect();
    if segments.is_empty() {
        return items;
    }

    let Some(resolved_path) = resolve_module_path_for_segments(snapshot, module, &segments) else {
        return items;
    };
    let mut push_item = |name: String, kind: CompletionItemKind, detail: Option<String>| {
        if !prefix.is_empty() && !name.starts_with(prefix) {
            return;
        }

        if !seen.insert(name.clone()) {
            return;
        }

        let mut item = CompletionItem::default();
        item.label = name;
        item.kind = Some(kind);
        if let Some(detail) = detail {
            item.detail = Some(detail);
        }

        items.push(item);
    };
    if let Some(children) = snapshot.module_children(&resolved_path) {
        let mut names: Vec<_> = children.iter().cloned().collect();
        names.sort();
        for child in names {
            push_item(
                child,
                CompletionItemKind::MODULE,
                Some("module".to_string()),
            );
        }
    }

    if let Some(target) = snapshot.module_for_name(&resolved_path) {
        let mut functions: Vec<_> = target.module.exports.functions.iter().collect();
        functions.sort_by(|(a, _), (b, _)| a.cmp(b));
        for (name, qualified) in functions {
            push_item(
                name.clone(),
                CompletionItemKind::FUNCTION,
                Some(qualified.clone()),
            );
        }

        let mut types: Vec<_> = target.module.exports.types.iter().collect();
        types.sort_by(|(a, _), (b, _)| a.cmp(b));
        for (name, qualified) in types {
            let kind = type_completion_kind(snapshot, qualified);
            push_item(name.clone(), kind, Some(qualified.clone()));
        }
    }

    items
}

fn module_path_completions(
    snapshot: &AnalysisSnapshot,
    module: &ModuleSnapshot,
    segments: &[String],
    prefix: &str,
) -> Vec<CompletionItem> {
    let mut items = Vec::new();
    let mut seen = HashSet::new();
    let Some(resolved_path) = resolve_module_path_for_segments(snapshot, module, segments) else {
        return items;
    };
    let mut push_item = |name: String, kind: CompletionItemKind, detail: Option<String>| {
        if !prefix.is_empty() && !name.starts_with(prefix) {
            return;
        }

        if !seen.insert(name.clone()) {
            return;
        }

        let mut item = CompletionItem::default();
        item.label = name;
        item.kind = Some(kind);
        if let Some(detail) = detail {
            item.detail = Some(detail);
        }

        items.push(item);
    };
    if let Some(children) = snapshot.module_children(&resolved_path) {
        let mut names: Vec<_> = children.iter().cloned().collect();
        names.sort();
        for child in names {
            push_item(
                child,
                CompletionItemKind::MODULE,
                Some("module".to_string()),
            );
        }
    }

    if let Some(target) = snapshot.module_for_name(&resolved_path) {
        let mut functions: Vec<_> = target.module.exports.functions.iter().collect();
        functions.sort_by(|(a, _), (b, _)| a.cmp(b));
        for (name, qualified) in functions {
            push_item(
                name.clone(),
                CompletionItemKind::FUNCTION,
                Some(qualified.clone()),
            );
        }

        let mut types: Vec<_> = target.module.exports.types.iter().collect();
        types.sort_by(|(a, _), (b, _)| a.cmp(b));
        for (name, qualified) in types {
            let kind = type_completion_kind(snapshot, qualified);
            push_item(name.clone(), kind, Some(qualified.clone()));
        }
    }

    items
}

fn format_type_expr(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Int => "int".to_string(),
        TypeExpr::Float => "float".to_string(),
        TypeExpr::Bool => "bool".to_string(),
        TypeExpr::String => "string".to_string(),
        TypeExpr::Unit => "()".to_string(),
        TypeExpr::Unknown => "unknown".to_string(),
        TypeExpr::Named(name) => (*name).to_string(),
        TypeExpr::Generic(name) => (*name).to_string(),
        TypeExpr::Array(inner) => format!("Array<{}>", format_type_expr(inner)),
        TypeExpr::Map(key, value) => format!(
            "Map<{}, {}>",
            format_type_expr(key),
            format_type_expr(value)
        ),
        TypeExpr::Result(ok, err) => format!(
            "Result<{}, {}>",
            format_type_expr(ok),
            format_type_expr(err)
        ),
        TypeExpr::Option(inner) => format!("Option<{}>", format_type_expr(inner)),
        TypeExpr::Table => "Table".to_string(),
        TypeExpr::SelfType => "Self".to_string(),
        TypeExpr::Function {
            params,
            return_type,
        } => {
            let params = if params.is_empty() {
                String::new()
            } else {
                params
                    .iter()
                    .map(format_type_expr)
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            format!("function({}) -> {}", params, format_type_expr(return_type))
        }
    }
}

fn return_type_text(ty: &TypeExpr) -> Option<String> {
    if matches!(ty, TypeExpr::Unit) {
        None
    } else {
        Some(format_type_expr(ty))
    }
}

fn format_params_with_types(params: &[TypeExpr], names: &[&str]) -> String {
    if params.is_empty() {
        return String::new();
    }

    params
        .iter()
        .enumerate()
        .map(|(idx, ty)| {
            let ty_text = format_type_expr(ty);
            let name = names.get(idx).copied().unwrap_or("_");
            if name.is_empty() {
                ty_text
            } else {
                format!("{}: {}", name, ty_text)
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn completion_documentation(detail: &str, description: &str) -> Option<Documentation> {
    if detail.is_empty() && description.is_empty() {
        return None;
    }

    let mut value = String::new();
    if !detail.is_empty() {
        value.push_str("```lust\n");
        value.push_str(detail);
        value.push_str("\n```");
    }

    if !description.is_empty() {
        if !value.is_empty() {
            value.push_str("\n\n");
        }
        value.push_str(description);
    }

    Some(Documentation::MarkupContent(MarkupContent {
        kind: MarkupKind::Markdown,
        value,
    }))
}

fn build_insert_text(base: &str, param_names: &[&str]) -> (String, Option<InsertTextFormat>) {
    if param_names.is_empty() {
        (format!("{}()", base), None)
    } else {
        let placeholders = param_names
            .iter()
            .enumerate()
            .map(|(idx, name)| format!("${{{}:{}}}", idx + 1, name))
            .collect::<Vec<_>>()
            .join(", ");
        (
            format!("{}({})", base, placeholders),
            Some(InsertTextFormat::SNIPPET),
        )
    }
}

fn function_short_name<'a>(func: &'a BuiltinFunction, module: Option<&str>) -> &'a str {
    if let Some(module) = module {
        let prefix_len = module.len();
        if func.name.len() > prefix_len
            && func.name.starts_with(module)
            && func.name.as_bytes().get(prefix_len) == Some(&b'.')
        {
            &func.name[prefix_len + 1..]
        } else {
            func.name
        }
    } else {
        func.name
    }
}

fn function_detail(func: &BuiltinFunction, module: Option<&str>) -> String {
    let qualified_name = if let Some(module) = module {
        format!("{}.{}", module, function_short_name(func, Some(module)))
    } else {
        func.name.to_string()
    };
    let params = format_params_with_types(&func.signature.params, func.param_names);
    let mut detail = if params.is_empty() {
        format!("{}()", qualified_name)
    } else {
        format!("{}({})", qualified_name, params)
    };
    if let Some(ret) = return_type_text(&func.signature.return_type) {
        detail.push_str(": ");
        detail.push_str(&ret);
    }
    detail
}

fn function_completion(func: &BuiltinFunction, module: Option<&str>) -> CompletionItem {
    let label = if let Some(module) = module {
        function_short_name(func, Some(module)).to_string()
    } else {
        func.name.to_string()
    };
    let (insert_text, insert_format) = build_insert_text(&label, func.param_names);
    let mut item = CompletionItem::default();
    item.label = label;
    item.kind = Some(CompletionItemKind::FUNCTION);
    item.insert_text = Some(insert_text);
    item.insert_text_format = insert_format;
    let detail = function_detail(func, module);
    item.detail = Some(detail.clone());
    item.documentation = completion_documentation(&detail, func.description);
    item
}

fn method_detail(method: &BuiltinMethod) -> String {
    let params = format_params_with_types(&method.signature.params, method.param_names);
    let mut detail = if params.is_empty() {
        format!("fn {}()", method.name)
    } else {
        format!("fn {}({})", method.name, params)
    };
    if let Some(ret) = return_type_text(&method.signature.return_type) {
        detail.push_str(": ");
        detail.push_str(&ret);
    }
    detail
}

fn method_completion(method: &BuiltinMethod) -> CompletionItem {
    let (insert_text, insert_format) = build_insert_text(method.name, method.param_names);
    let mut item = CompletionItem::default();
    item.label = method.name.to_string();
    item.kind = Some(CompletionItemKind::METHOD);
    item.insert_text = Some(insert_text);
    item.insert_text_format = insert_format;
    let detail = method_detail(method);
    item.detail = Some(detail.clone());
    item.documentation = completion_documentation(&detail, method.description);
    item
}

fn builtin_global_completions(object_name: &str, prefix: &str) -> Vec<CompletionItem> {
    let database = builtins::builtins();
    let simple = simple_type_name(object_name);
    let mut module = database.module(simple);
    if module.is_none() {
        let lower = simple.to_ascii_lowercase();
        module = database.module(lower.as_str());
    }
    let module = match module {
        Some(module) => module,
        None => return Vec::new(),
    };

    module
        .functions()
        .iter()
        .filter_map(|func| {
            let short = function_short_name(func, Some(module.name()));
            if prefix.is_empty() || short.starts_with(prefix) {
                Some(function_completion(func, Some(module.name())))
            } else {
                None
            }
        })
        .collect()
}

fn builtin_static_method_completions(_owner: &str, _prefix: &str) -> Vec<CompletionItem> {
    Vec::new()
}

fn builtin_instance_method_completions(owner: &str, prefix: &str) -> Vec<CompletionItem> {
    let database = builtins::builtins();
    let simple = simple_type_name(owner);
    let Some(methods) = database.methods_for(simple) else {
        return Vec::new();
    };

    methods
        .iter()
        .filter_map(|method| {
            if prefix.is_empty() || method.name.starts_with(prefix) {
                Some(method_completion(method))
            } else {
                None
            }
        })
        .collect()
}

fn static_method_completions(
    snapshot: &AnalysisSnapshot,
    owner: &str,
    module_path: Option<&str>,
    prefix: &str,
) -> Vec<CompletionItem> {
    let mut items = Vec::new();
    let mut seen = HashSet::new();
    let owner_simple = simple_type_name(owner);
    if let Some(methods) = snapshot.methods_for_type(owner) {
        for method in methods {
            let method_owner_simple = simple_type_name(&method.owner);
            if method.owner != owner && method_owner_simple != owner_simple {
                continue;
            }

            if method.is_instance {
                continue;
            }

            if !method_visible(method, module_path) {
                continue;
            }

            if !prefix.is_empty() && !method.name.starts_with(prefix) {
                continue;
            }

            if !seen.insert(method.name.clone()) {
                continue;
            }

            let (insert_text, insert_format) = method_insert_text(method);
            let mut item = CompletionItem::default();
            item.label = method.name.clone();
            item.kind = Some(CompletionItemKind::FUNCTION);
            item.detail = Some(format_method_signature(method));
            item.insert_text = Some(insert_text);
            if let Some(format) = insert_format {
                item.insert_text_format = Some(format);
            }

            items.push(item);
        }
    }

    items
}

fn instance_method_completions(
    snapshot: &AnalysisSnapshot,
    owner: &str,
    module_path: Option<&str>,
    prefix: &str,
) -> Vec<CompletionItem> {
    let mut items = Vec::new();
    let mut seen = HashSet::new();
    let owner_simple = simple_type_name(owner);
    if let Some(methods) = snapshot.methods_for_type(owner) {
        for method in methods {
            let method_owner_simple = simple_type_name(&method.owner);
            if method.owner != owner && method_owner_simple != owner_simple {
                continue;
            }

            if !method.is_instance {
                continue;
            }

            if !method_visible(method, module_path) {
                continue;
            }

            if !prefix.is_empty() && !method.name.starts_with(prefix) {
                continue;
            }

            if !seen.insert(method.name.clone()) {
                continue;
            }

            let (insert_text, insert_format) = method_insert_text(method);
            let mut item = CompletionItem::default();
            item.label = method.name.clone();
            item.kind = Some(CompletionItemKind::METHOD);
            item.detail = Some(format_method_signature(method));
            item.insert_text = Some(insert_text);
            if let Some(format) = insert_format {
                item.insert_text_format = Some(format);
            }

            items.push(item);
        }
    }

    items
}

fn type_for_identifier(
    module: &ModuleSnapshot,
    text: &str,
    line_offsets: &[usize],
    name: &str,
    approx_span: Option<Span>,
    target_line: usize,
) -> Option<Type> {
    let try_span = |span: Span| -> Option<Type> {
        module.type_for_span(&span).cloned().or_else(|| {
            if span.start_line == span.end_line {
                let collapsed = Span::new(
                    span.start_line,
                    span.start_col,
                    span.start_line,
                    span.start_col,
                );
                if collapsed != span {
                    module.type_for_span(&collapsed).cloned()
                } else {
                    None
                }
            } else {
                None
            }
        })
    };
    if let Some(span) = approx_span {
        if let Some(ty) = try_span(span) {
            return Some(ty);
        }
    }

    let name_len = name.chars().count();
    if name_len == 0 {
        return None;
    }

    let mut best: Option<(usize, Type)> = None;
    let mut consider = |span: &Span, ty: &Type| {
        if span.start_line == 0 {
            return;
        }

        if let Some(identifier) = identifier_name_at_span(text, line_offsets, *span, name_len) {
            if identifier == name {
                let distance = target_line.abs_diff(span.start_line);
                match &mut best {
                    Some((best_dist, best_ty)) => {
                        if distance < *best_dist {
                            *best_dist = distance;
                            *best_ty = ty.clone();
                        }
                    }

                    None => {
                        best = Some((distance, ty.clone()));
                    }
                }
            }
        }
    };
    for (span, ty) in &module.variable_types {
        consider(span, ty);
    }

    for (span, ty) in &module.expr_types {
        consider(span, ty);
    }

    best.map(|(_, ty)| ty)
}

fn analyze_member_method_context(text: &str, offset: usize) -> Option<CompletionContext> {
    let is_repeated_trigger = |idx: usize, ch: char| -> bool {
        if ch == '.' {
            if let Some((_, prev)) = prev_char_index(text, idx) {
                return prev == '.';
            }
        }

        false
    };
    if let Some((idx, ch)) = char_at_index(text, offset) {
        if ch == '.' || ch == ':' {
            if is_repeated_trigger(idx, ch) {
                return None;
            }

            let kind = if ch == '.' {
                CompletionKind::Member
            } else {
                CompletionKind::Method
            };
            if let Some((object_start, object_end)) = identifier_range_before(text, idx) {
                let object_name = identifier_text(text, (object_start, object_end));
                if !object_name.chars().any(is_identifier_char) {
                    return None;
                }

                return Some(CompletionContext {
                    kind,
                    object_start: Some(object_start),
                    object_end,
                    object_name: Some(object_name),
                    prefix: String::new(),
                    path_segments: Vec::new(),
                });
            }

            return Some(CompletionContext {
                kind,
                object_start: None,
                object_end: idx,
                object_name: None,
                prefix: String::new(),
                path_segments: Vec::new(),
            });
        }
    }

    let (prefix_start, _) = identifier_prefix_range(text, offset);
    let prefix = identifier_text(text, (prefix_start, offset));
    let mut cursor = prefix_start;
    while let Some((idx, ch)) = prev_char_index(text, cursor) {
        if ch.is_whitespace() {
            cursor = idx;
            continue;
        }

        if ch == '.' || ch == ':' {
            let trigger_char = ch;
            let trigger_offset = idx;
            if is_repeated_trigger(trigger_offset, trigger_char) {
                return None;
            }

            if let Some((object_start, object_end)) = identifier_range_before(text, trigger_offset)
            {
                let object_name = identifier_text(text, (object_start, object_end));
                if !object_name.chars().any(is_identifier_char) {
                    return None;
                }

                return Some(CompletionContext {
                    kind: if trigger_char == '.' {
                        CompletionKind::Member
                    } else {
                        CompletionKind::Method
                    },
                    object_start: Some(object_start),
                    object_end,
                    object_name: Some(object_name),
                    prefix,
                    path_segments: Vec::new(),
                });
            }

            return Some(CompletionContext {
                kind: if trigger_char == '.' {
                    CompletionKind::Member
                } else {
                    CompletionKind::Method
                },
                object_start: None,
                object_end: trigger_offset,
                object_name: None,
                prefix,
                path_segments: Vec::new(),
            });
        }

        break;
    }

    None
}

fn analyze_pattern_context(text: &str, offset: usize) -> Option<CompletionContext> {
    let (prefix_start, _) = identifier_prefix_range(text, offset);
    let prefix = identifier_text(text, (prefix_start, offset));
    let (keyword_start, keyword_end) = identifier_range_before(text, prefix_start)?;
    let keyword = identifier_text(text, (keyword_start, keyword_end));
    if keyword != "is" {
        return None;
    }

    let (object_start, object_end) = identifier_range_before(text, keyword_start)?;
    let object_name = identifier_text(text, (object_start, object_end));
    Some(CompletionContext {
        kind: CompletionKind::Pattern,
        object_start: Some(object_start),
        object_end: object_end,
        object_name: Some(object_name),
        prefix,
        path_segments: Vec::new(),
    })
}

fn analyze_identifier_context(text: &str, offset: usize) -> Option<CompletionContext> {
    let (prefix_start, _) = identifier_prefix_range(text, offset);
    let prefix = identifier_text(text, (prefix_start, offset));
    Some(CompletionContext {
        kind: CompletionKind::Identifier,
        object_start: None,
        object_end: prefix_start,
        object_name: None,
        prefix,
        path_segments: Vec::new(),
    })
}

fn analyze_module_path_context(
    text: &str,
    offset: usize,
    module: &ModuleSnapshot,
    position: &Position,
) -> Option<CompletionContext> {
    let (segments, prefix) = parse_module_path_at(text, offset)?;
    if !is_within_use_item(&module.module.items, position) {
        return None;
    }

    Some(CompletionContext {
        kind: CompletionKind::ModulePath,
        object_start: None,
        object_end: offset,
        object_name: None,
        prefix,
        path_segments: segments,
    })
}

fn parse_module_path_at(text: &str, offset: usize) -> Option<(Vec<String>, String)> {
    if text.is_empty() {
        return None;
    }

    let cursor = offset.min(text.len());
    let mut prefix_start = cursor;
    while let Some((idx, ch)) = prev_char_index(text, prefix_start) {
        if is_identifier_char(ch) {
            prefix_start = idx;
        } else {
            break;
        }
    }

    let prefix = text[prefix_start..cursor].to_string();
    let mut segments = Vec::new();
    let mut scan = prefix_start;
    while scan > 0 {
        let mut idx = scan;
        while let Some((prev_idx, prev_ch)) = prev_char_index(text, idx) {
            if prev_ch.is_whitespace() {
                idx = prev_idx;
            } else {
                break;
            }
        }

        if idx == 0 {
            break;
        }

        let (next_idx, ch) = match prev_char_index(text, idx) {
            Some(val) => val,
            None => break,
        };
        match ch {
            '.' => {
                let mut seg_end = next_idx;
                while let Some((w_idx, w_ch)) = prev_char_index(text, seg_end) {
                    if w_ch.is_whitespace() {
                        seg_end = w_idx;
                    } else {
                        break;
                    }
                }

                if seg_end == 0 {
                    break;
                }

                let mut seg_start = seg_end;
                let mut had_char = false;
                while let Some((c_idx, c_ch)) = prev_char_index(text, seg_start) {
                    if is_identifier_char(c_ch) {
                        seg_start = c_idx;
                        had_char = true;
                    } else {
                        break;
                    }
                }

                if !had_char {
                    break;
                }

                segments.push(text[seg_start..seg_end].to_string());
                scan = seg_start;
            }

            '{' | '}' | ',' | '(' | ')' | '[' | ']' => {
                scan = next_idx;
            }

            _ => break,
        }
    }

    segments.reverse();
    Some((segments, prefix))
}

fn is_within_use_item(items: &[Item], position: &Position) -> bool {
    for item in items {
        match &item.kind {
            ItemKind::Use { .. } => {
                if span_contains_position(item.span, position) {
                    return true;
                }
            }

            ItemKind::Module { items: inner, .. } => {
                if span_contains_position(item.span, position)
                    && is_within_use_item(inner, position)
                {
                    return true;
                }
            }

            _ => {}
        }
    }

    false
}

struct LocalInfo {
    name: String,
    type_detail: Option<String>,
}

enum ScopeContext<'a> {
    Function {
        params: &'a [FunctionParam],
        body: &'a [Stmt],
    },
    Script {
        stmts: &'a [Stmt],
    },
}

fn find_scope<'a>(items: &'a [Item], position: &Position) -> Option<ScopeContext<'a>> {
    find_scope_strict(items, position).or_else(|| script_scope_fallback(items, position))
}

fn find_scope_strict<'a>(items: &'a [Item], position: &Position) -> Option<ScopeContext<'a>> {
    for item in items {
        if !span_contains_position(item.span, position) {
            continue;
        }

        match &item.kind {
            ItemKind::Function(func) => {
                return Some(ScopeContext::Function {
                    params: &func.params,
                    body: &func.body,
                });
            }

            ItemKind::Impl(impl_block) => {
                for method in &impl_block.methods {
                    if stmts_contain_position(&method.body, position) {
                        return Some(ScopeContext::Function {
                            params: &method.params,
                            body: &method.body,
                        });
                    }
                }
            }

            ItemKind::Module { items: inner, .. } => {
                if let Some(scope) = find_scope(inner, position) {
                    return Some(scope);
                }
            }

            ItemKind::Script(stmts) => {
                return Some(ScopeContext::Script { stmts });
            }

            _ => {}
        }
    }

    None
}

fn script_scope_fallback<'a>(items: &'a [Item], position: &Position) -> Option<ScopeContext<'a>> {
    let mut best: Option<(usize, usize, ScopeContext<'a>)> = None;
    script_scope_fallback_inner(items, position, &mut best);
    best.map(|(_, _, scope)| scope)
}

fn script_scope_fallback_inner<'a>(
    items: &'a [Item],
    position: &Position,
    best: &mut Option<(usize, usize, ScopeContext<'a>)>,
) {
    let pos_line = position.line as usize + 1;
    let pos_col = position.character as usize + 1;
    for item in items {
        match &item.kind {
            ItemKind::Script(stmts) => {
                let start_line = item.span.start_line;
                let start_col = item.span.start_col;
                let can_use = if start_line == 0 {
                    true
                } else {
                    start_line < pos_line || (start_line == pos_line && start_col <= pos_col)
                };
                if !can_use {
                    continue;
                }

                let key_line = if start_line == 0 { 0 } else { start_line };
                let key_col = if start_col == 0 { 0 } else { start_col };
                let should_update = match best {
                    Some((best_line, best_col, _)) => {
                        key_line > *best_line || (key_line == *best_line && key_col >= *best_col)
                    }

                    None => true,
                };
                if should_update {
                    *best = Some((key_line, key_col, ScopeContext::Script { stmts }));
                }
            }

            ItemKind::Module { items: inner, .. } => {
                script_scope_fallback_inner(inner, position, best);
            }

            _ => {}
        }
    }
}

fn collect_locals_from_scope(
    scope: &ScopeContext,
    module: &ModuleSnapshot,
    position: &Position,
) -> Vec<LocalInfo> {
    let mut locals = Vec::new();
    match scope {
        ScopeContext::Function { body, .. } => {
            collect_locals_from_stmts(body, module, position, &mut locals);
        }

        ScopeContext::Script { stmts } => {
            collect_locals_from_stmts(stmts, module, position, &mut locals);
        }
    }

    locals
}

fn collect_locals_from_stmts(
    stmts: &[Stmt],
    module: &ModuleSnapshot,
    position: &Position,
    results: &mut Vec<LocalInfo>,
) {
    let pos_line = position.line as usize + 1;
    let pos_col = position.character as usize + 1;
    'stmts: for stmt in stmts {
        if stmt.span.start_line != 0 && span_starts_after(stmt.span, pos_line, pos_col) {
            break;
        }

        if !collect_locals_from_stmt(stmt, module, position, pos_line, pos_col, results) {
            break 'stmts;
        }
    }
}

fn collect_locals_from_stmt(
    stmt: &Stmt,
    module: &ModuleSnapshot,
    position: &Position,
    pos_line: usize,
    pos_col: usize,
    results: &mut Vec<LocalInfo>,
) -> bool {
    match &stmt.kind {
        StmtKind::Local { bindings, .. } => {
            for binding in bindings {
                if binding.span.start_line == 0 {
                    continue;
                }

                if span_start_before_or_equal(binding.span, pos_line, pos_col) {
                    let ty = module.type_for_span(&binding.span).map(|t| t.to_string());
                    results.push(LocalInfo {
                        name: binding.name.clone(),
                        type_detail: ty,
                    });
                }
            }
        }

        StmtKind::Block(inner) => {
            if span_contains_position(stmt.span, position) {
                collect_locals_from_stmts(inner, module, position, results);
                return false;
            }

            return true;
        }

        StmtKind::If {
            then_block,
            elseif_branches,
            else_block,
            ..
        } => {
            if span_contains_position(stmt.span, position) {
                if !then_block.is_empty() && stmts_contain_position(then_block, position)
                    || (then_block.is_empty() && span_contains_position(stmt.span, position))
                {
                    collect_locals_from_stmts(then_block, module, position, results);
                    return false;
                }

                for (_, branch) in elseif_branches {
                    if (!branch.is_empty() && stmts_contain_position(branch, position))
                        || (branch.is_empty() && span_contains_position(stmt.span, position))
                    {
                        collect_locals_from_stmts(branch, module, position, results);
                        return false;
                    }
                }

                if let Some(else_block) = else_block {
                    if (!else_block.is_empty() && stmts_contain_position(else_block, position))
                        || (else_block.is_empty() && span_contains_position(stmt.span, position))
                    {
                        collect_locals_from_stmts(else_block, module, position, results);
                    }
                }

                return false;
            }
        }

        StmtKind::While { body, .. } => {
            if span_contains_position(stmt.span, position) {
                if (!body.is_empty() && stmts_contain_position(body, position))
                    || (body.is_empty() && span_contains_position(stmt.span, position))
                {
                    collect_locals_from_stmts(body, module, position, results);
                }

                return false;
            }
        }

        StmtKind::ForNumeric { variable, body, .. } => {
            if span_contains_position(stmt.span, position) {
                if (!body.is_empty() && stmts_contain_position(body, position))
                    || (body.is_empty() && span_contains_position(stmt.span, position))
                {
                    results.push(LocalInfo {
                        name: variable.clone(),
                        type_detail: None,
                    });
                    collect_locals_from_stmts(body, module, position, results);
                }

                return false;
            }
        }

        StmtKind::ForIn {
            variables, body, ..
        } => {
            if span_contains_position(stmt.span, position) {
                if (!body.is_empty() && stmts_contain_position(body, position))
                    || (body.is_empty() && span_contains_position(stmt.span, position))
                {
                    for var in variables {
                        results.push(LocalInfo {
                            name: var.clone(),
                            type_detail: None,
                        });
                    }

                    collect_locals_from_stmts(body, module, position, results);
                }

                return false;
            }
        }

        StmtKind::Expr(_)
        | StmtKind::Return(_)
        | StmtKind::Break
        | StmtKind::Continue
        | StmtKind::Assign { .. }
        | StmtKind::CompoundAssign { .. } => {}
    }

    if span_contains_position(stmt.span, position) {
        return false;
    }

    true
}

fn stmts_contain_position(stmts: &[Stmt], position: &Position) -> bool {
    for stmt in stmts {
        if span_contains_position(stmt.span, position) {
            return true;
        }

        match &stmt.kind {
            StmtKind::Block(inner)
            | StmtKind::While { body: inner, .. }
            | StmtKind::ForNumeric { body: inner, .. }
            | StmtKind::ForIn { body: inner, .. } => {
                if stmts_contain_position(inner, position) {
                    return true;
                }
            }

            StmtKind::If {
                then_block,
                elseif_branches,
                else_block,
                ..
            } => {
                if stmts_contain_position(then_block, position)
                    || elseif_branches
                        .iter()
                        .any(|(_, branch)| stmts_contain_position(branch, position))
                    || else_block
                        .as_ref()
                        .map(|block| stmts_contain_position(block, position))
                        .unwrap_or(false)
                {
                    return true;
                }
            }

            _ => {}
        }
    }

    false
}

fn resolve_type_candidates(type_name: &str, module: &ModuleSnapshot) -> Vec<String> {
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();
    let mut push = |value: String| {
        if seen.insert(value.clone()) {
            candidates.push(value);
        }
    };
    if let Some(fq) = module.module.imports.type_aliases.get(type_name) {
        push(fq.clone());
    }

    if let Some((module_alias, rest)) = type_name.split_once('.') {
        if let Some(real_module) = module.module.imports.module_aliases.get(module_alias) {
            push(format!("{}.{}", real_module, rest));
        }
    }

    push(type_name.to_string());
    if !type_name.contains('.') && !module.module.path.is_empty() {
        push(format!("{}.{}", module.module.path, type_name));
    }

    candidates
}

fn infer_struct_literal_base_name(
    text: &str,
    trigger_offset: usize,
    module: &ModuleSnapshot,
    snapshot: &AnalysisSnapshot,
    module_path: Option<&str>,
) -> Option<String> {
    if trigger_offset == 0 || trigger_offset > text.len() {
        return None;
    }

    let bytes = text.as_bytes();
    let mut idx = trigger_offset;
    while idx > 0 && bytes[idx - 1].is_ascii_whitespace() {
        idx -= 1;
    }

    if idx == 0 || bytes[idx - 1] != b'}' {
        return None;
    }

    let mut cursor = idx - 1;
    let mut depth = 1usize;
    while cursor > 0 {
        cursor -= 1;
        match bytes[cursor] {
            b'}' => depth += 1,
            b'{' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }

            _ => {}
        }
    }

    if depth != 0 || bytes.get(cursor) != Some(&b'{') {
        return None;
    }

    let mut end = cursor;
    while end > 0 && bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }

    let mut start = end;
    while start > 0 {
        let ch = bytes[start - 1];
        if ch.is_ascii_alphanumeric() || ch == b'_' || ch == b'.' {
            start -= 1;
        } else {
            break;
        }
    }

    if start == end {
        return None;
    }

    let type_name = &text[start..end];
    for candidate in resolve_type_candidates(type_name, module) {
        if let Some(info) = snapshot.struct_info_for(&candidate, module_path) {
            let qualified = qualify_type_name(&info.module_path, &info.def.name);
            return Some(qualified);
        }
    }

    None
}

fn infer_constructor_call_base_name(
    text: &str,
    trigger_offset: usize,
    module: &ModuleSnapshot,
    snapshot: &AnalysisSnapshot,
) -> Option<String> {
    if trigger_offset == 0 || trigger_offset > text.len() {
        return None;
    }

    let bytes = text.as_bytes();
    let mut idx = trigger_offset;
    while idx > 0 && bytes[idx - 1].is_ascii_whitespace() {
        idx -= 1;
    }

    if idx == 0 || bytes[idx - 1] != b')' {
        return None;
    }

    let mut cursor = idx - 1;
    let mut depth = 1usize;
    while cursor > 0 {
        cursor -= 1;
        match bytes[cursor] {
            b')' => depth += 1,
            b'(' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }

            _ => {}
        }
    }

    if depth != 0 || bytes.get(cursor) != Some(&b'(') {
        return None;
    }

    let mut end = cursor;
    while end > 0 && bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }

    let mut start = end;
    while start > 0 {
        let ch = bytes[start - 1];
        if ch.is_ascii_alphanumeric() || ch == b'_' || ch == b'.' || ch == b':' {
            start -= 1;
        } else {
            break;
        }
    }

    if start == end {
        return None;
    }

    let call_ident = &text[start..end];
    if let Some((type_part, method_part, is_instance)) = split_type_member(call_ident) {
        if is_instance {
            return None;
        }

        for candidate in resolve_type_candidates(&type_part, module) {
            if let Some(methods) = snapshot.methods_for_type(&candidate) {
                for method in methods {
                    if method.name == method_part && !method.is_instance {
                        if let Some(ret_ty) = &method.return_type {
                            if let Some(base) = base_type_name(ret_ty) {
                                if base.contains('.') {
                                    return Some(base);
                                } else {
                                    let qualified = qualify_type_name(&method.module_path, &base);
                                    return Some(qualified);
                                }
                            }
                        }

                        return Some(method.owner.clone());
                    }
                }
            }
        }
    }

    None
}

fn resolve_base_type_name_for_context(
    module: &ModuleSnapshot,
    snapshot: &AnalysisSnapshot,
    module_path: Option<&str>,
    text: &str,
    line_offsets: &[usize],
    context: &CompletionContext,
) -> Option<String> {
    if text.is_empty() {
        return None;
    }

    let mut value_type: Option<Type> = None;
    let mut identifier_span: Option<Span> = None;
    let mut object_line: Option<usize> = None;
    if let Some(start) = context.object_start {
        let object_position = offset_to_position(text, start, line_offsets);
        object_line = Some(object_position.line as usize + 1);
        if let Some(name) = context.object_name.as_ref() {
            identifier_span = span_from_identifier(text, start, name, line_offsets);
        }

        value_type = find_type_for_position(module, object_position).map(|(_, ty)| ty);
    }

    if value_type.is_none() {
        let mut fallback_offsets = Vec::new();
        if context.object_end > 0 {
            fallback_offsets.push(context.object_end.saturating_sub(1));
        }

        fallback_offsets.push(context.object_end);
        for offset_lookup in fallback_offsets {
            if text.is_empty() {
                break;
            }

            if text.len() == 1 && offset_lookup > 0 {
                continue;
            }

            let capped = offset_lookup.min(text.len().saturating_sub(1));
            let position = offset_to_position(text, capped, line_offsets);
            if let Some((_, ty)) = find_type_for_position(module, position) {
                value_type = Some(ty);
                break;
            }
        }
    }

    if let (Some(name), Some(span), Some(line)) =
        (context.object_name.as_ref(), identifier_span, object_line)
    {
        if let Some(id_ty) = type_for_identifier(module, text, line_offsets, name, Some(span), line)
        {
            value_type = Some(id_ty);
        }
    }

    let mut base_name = value_type.as_ref().and_then(|ty| base_type_name(ty));
    if base_name.is_none() {
        base_name =
            infer_struct_literal_base_name(text, context.object_end, module, snapshot, module_path);
    }

    if base_name.is_none() {
        base_name = infer_constructor_call_base_name(text, context.object_end, module, snapshot);
    }

    base_name
}

fn hover_for_method_call(
    snapshot: &AnalysisSnapshot,
    module: &ModuleSnapshot,
    module_path: Option<&str>,
    text: &str,
    position: Position,
    method_token: &str,
) -> Option<Hover> {
    if method_token.is_empty() {
        return None;
    }

    let line_offsets = compute_line_offsets(text);
    let line_idx = position.line as usize;
    if line_idx >= line_offsets.len() {
        return None;
    }

    let line_start = line_offsets[line_idx];
    let line_end = line_offsets
        .get(line_idx + 1)
        .copied()
        .unwrap_or_else(|| text.len());
    if line_start >= line_end || line_end > text.len() {
        return None;
    }

    let line_slice = &text[line_start..line_end];
    let chars: Vec<char> = line_slice.chars().collect();
    if chars.is_empty() {
        return None;
    }

    let mut char_idx = position.character as usize;
    if char_idx >= chars.len() {
        char_idx = chars.len().saturating_sub(1);
    }

    if !is_word_char(chars[char_idx]) {
        while char_idx > 0 && !is_word_char(chars[char_idx]) {
            char_idx -= 1;
        }

        if !is_word_char(chars[char_idx]) {
            return None;
        }
    }

    let mut start_char = char_idx;
    while start_char > 0 && is_word_char(chars[start_char - 1]) {
        start_char -= 1;
    }

    let mut end_char = char_idx + 1;
    while end_char < chars.len() && is_word_char(chars[end_char]) {
        end_char += 1;
    }

    let line_start_byte = nth_char_byte_index(line_slice, start_char);
    let line_end_byte = nth_char_byte_index(line_slice, end_char);
    if line_start_byte > line_end_byte || line_end_byte > line_slice.len() {
        return None;
    }

    let token_str = &line_slice[line_start_byte..line_end_byte];
    let desired_name = method_token.rsplit('.').next().unwrap_or(method_token);
    let method_segment = token_str.rsplit('.').next().unwrap_or(token_str);
    if method_segment != desired_name {
        return None;
    }

    let prefix_len = token_str.len().saturating_sub(method_segment.len());
    let method_start_byte = line_start_byte + prefix_len;
    let method_end_byte = method_start_byte + method_segment.len();
    let start_offset = line_start + method_start_byte;
    let end_offset = line_start + method_end_byte;
    let next_char = text[end_offset..]
        .chars()
        .skip_while(|c| c.is_whitespace())
        .next();
    if next_char != Some('(') {
        return None;
    }

    let context = analyze_member_method_context(text, end_offset)?;
    if !matches!(
        context.kind,
        CompletionKind::Method | CompletionKind::Member
    ) {
        return None;
    }

    let expect_instance = matches!(context.kind, CompletionKind::Method);
    let base_name = resolve_base_type_name_for_context(
        module,
        snapshot,
        module_path,
        text,
        &line_offsets,
        &context,
    )?;
    let methods = snapshot.methods_for_type(&base_name)?;
    let method = methods
        .iter()
        .find(|m| m.name == desired_name && m.is_instance == expect_instance)?;
    let hover_span = span_from_identifier(text, start_offset, method_segment, &line_offsets);
    let signature = format_method_signature(method);
    let mut body = format!("```lust\n{}\n```", signature);
    body.push_str(&format!("\nDefined on `{}`", method.owner));
    if !method.module_path.is_empty() {
        body.push_str(&format!("\nModule `{}`", method.module_path));
    }

    Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: body,
        }),
        range: hover_span.map(span_to_range),
    })
}

fn collect_inlay_hints_for_module(module: &ModuleSnapshot, range: &Range) -> Vec<InlayHint> {
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

                    let position = Position::new(
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

struct Backend {
    client: Client,
    documents: Arc<RwLock<HashMap<Url, DocumentState>>>,
    last_published: Arc<RwLock<HashMap<Url, HashSet<Url>>>>,
    analysis: Arc<RwLock<Option<AnalysisSnapshot>>>,
}

impl Backend {
    fn new(client: Client) -> Self {
        Self {
            client,
            documents: Arc::new(RwLock::new(HashMap::new())),
            last_published: Arc::new(RwLock::new(HashMap::new())),
            analysis: Arc::new(RwLock::new(None)),
        }
    }

    async fn semantic_tokens_for_path(&self, path: &Path) -> Option<Vec<SemanticToken>> {
        let analysis = self.analysis.read().await;
        analysis
            .as_ref()
            .and_then(|snapshot| snapshot.semantic_tokens_for_path(path))
    }

    async fn document_text(&self, uri: &Url) -> Option<String> {
        let cached = {
            let docs = self.documents.read().await;
            docs.get(uri).map(|doc| doc.text.clone())
        };
        if cached.is_some() {
            return cached;
        }

        let path = uri.to_file_path().ok()?;
        std::fs::read_to_string(path).ok()
    }

    async fn analyze(&self, uri: &Url) {
        let version = {
            let docs = self.documents.read().await;
            docs.get(uri).map(|doc| doc.version)
        };
        let Some(version) = version else {
            return;
        };
        let diagnostics = self.compute_diagnostics(uri).await;
        self.publish_diagnostics(uri.clone(), version, diagnostics)
            .await;
    }

    async fn compute_diagnostics(&self, uri: &Url) -> HashMap<Url, Vec<Diagnostic>> {
        let entry_path = match uri.to_file_path() {
            Ok(path) => path,
            Err(_) => {
                self.client
                    .log_message(
                        MessageType::ERROR,
                        format!("Unsupported document URI scheme: {uri}"),
                    )
                    .await;
                return HashMap::new();
            }
        };
        let overrides = {
            let docs = self.documents.read().await;
            let mut map = HashMap::new();
            for (doc_uri, state) in docs.iter() {
                if let Ok(path) = doc_uri.to_file_path() {
                    map.insert(path, state.text.clone());
                }
            }

            map
        };
        let entry_dir = entry_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let entry_path_str = match entry_path.to_str() {
            Some(s) => s.to_string(),
            None => {
                self.client
                    .log_message(
                        MessageType::ERROR,
                        format!("Non-UTF-8 file path not supported: {:?}", entry_path),
                    )
                    .await;
                return HashMap::new();
            }
        };
        let mut loader = ModuleLoader::new(".");
        loader.set_source_overrides(overrides.clone());
        match loader.load_program_from_entry(&entry_path_str) {
            Ok(program) => {
                let module_path_map: HashMap<String, PathBuf> = program
                    .modules
                    .iter()
                    .map(|m| (m.path.clone(), m.source_path.clone()))
                    .collect();
                let mut imports_map = HashMap::new();
                for module in &program.modules {
                    imports_map.insert(module.path.clone(), module.imports.clone());
                }

                let mut wrapped_items: Vec<Item> = Vec::new();
                for module in &program.modules {
                    wrapped_items.push(Item::new(
                        ItemKind::Module {
                            name: module.path.clone(),
                            items: module.items.clone(),
                        },
                        Span::new(0, 0, 0, 0),
                    ));
                }

                let config = analyzer_lust_config();
                let mut typechecker = TypeChecker::with_config(&config);
                typechecker.set_imports_by_module(imports_map.clone());
                let type_result = typechecker.check_program(&program.modules);
                let struct_defs = typechecker.struct_definitions();
                let enum_defs = typechecker.enum_definitions();
                let type_info = typechecker.take_type_info();
                let snapshot =
                    AnalysisSnapshot::new(&program, type_info, &overrides, struct_defs, enum_defs);
                {
                    let mut analysis = self.analysis.write().await;
                    *analysis = Some(snapshot);
                }

                if let Err(error) = type_result {
                    return self.convert_path_map_to_url_map(error_to_diagnostics(
                        error,
                        &entry_path,
                        &entry_dir,
                        &module_path_map,
                    ));
                }

                let mut compiler = Compiler::new();
                compiler.configure_stdlib(&config);
                compiler.set_imports_by_module(imports_map);
                compiler.set_entry_module(program.entry_module.clone());
                if let Err(error) = compiler.compile_module(&wrapped_items) {
                    return self.convert_path_map_to_url_map(error_to_diagnostics(
                        error,
                        &entry_path,
                        &entry_dir,
                        &module_path_map,
                    ));
                }

                let mut result = HashMap::new();
                if let Ok(url) = Url::from_file_path(&entry_path) {
                    result.insert(url, Vec::new());
                }

                result
            }

            Err(error) => self.convert_path_map_to_url_map(error_to_diagnostics(
                error,
                &entry_path,
                &entry_dir,
                &HashMap::new(),
            )),
        }
    }

    fn convert_path_map_to_url_map(
        &self,
        path_map: HashMap<PathBuf, Vec<Diagnostic>>,
    ) -> HashMap<Url, Vec<Diagnostic>> {
        let mut result = HashMap::new();
        for (path, diagnostics) in path_map {
            if let Ok(url) = Url::from_file_path(&path) {
                result.insert(url, diagnostics);
            }
        }

        result
    }

    async fn publish_diagnostics(
        &self,
        entry_uri: Url,
        entry_version: i32,
        mut new_diagnostics: HashMap<Url, Vec<Diagnostic>>,
    ) {
        new_diagnostics
            .entry(entry_uri.clone())
            .or_insert_with(Vec::new);
        let associated_uris: HashSet<Url> = new_diagnostics.keys().cloned().collect();
        let previous_uris = {
            let mut tracker = self.last_published.write().await;
            tracker
                .insert(entry_uri.clone(), associated_uris.clone())
                .unwrap_or_default()
        };
        let version_lookup = {
            let docs = self.documents.read().await;
            docs.iter()
                .map(|(u, state)| (u.clone(), state.version))
                .collect::<HashMap<_, _>>()
        };
        for (uri, diagnostics) in new_diagnostics {
            let version = version_lookup.get(&uri).copied().or_else(|| {
                if uri == entry_uri {
                    Some(entry_version)
                } else {
                    None
                }
            });
            self.client
                .publish_diagnostics(uri, diagnostics, version)
                .await;
        }

        for uri in previous_uris.difference(&associated_uris) {
            let version = version_lookup.get(uri).copied();
            self.client
                .publish_diagnostics(uri.clone(), Vec::new(), version)
                .await;
        }
    }
}

#[async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _params: InitializeParams) -> Result<InitializeResult> {
        let text_document_sync = TextDocumentSyncOptions {
            open_close: Some(true),
            change: Some(TextDocumentSyncKind::FULL),
            ..Default::default()
        };
        let hover_provider = Some(HoverProviderCapability::Simple(true));
        let definition_provider = Some(OneOf::Left(true));
        let inlay_hint_provider = Some(OneOf::Right(InlayHintServerCapabilities::Options(
            InlayHintOptions::default(),
        )));
        let semantic_tokens_provider = Some(
            SemanticTokensServerCapabilities::SemanticTokensOptions(SemanticTokensOptions {
                legend: SemanticTokensLegend {
                    token_types: SEMANTIC_TOKEN_TYPES.to_vec(),
                    token_modifiers: Vec::new(),
                },
                full: Some(SemanticTokensFullOptions::Bool(true)),
                range: Some(false),
                ..Default::default()
            }),
        );
        let completion_provider = Some(CompletionOptions {
            resolve_provider: Some(false),
            trigger_characters: Some(vec![".".to_string(), ":".to_string()]),
            ..CompletionOptions::default()
        });
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Options(text_document_sync)),
                hover_provider,
                definition_provider,
                inlay_hint_provider,
                semantic_tokens_provider,
                completion_provider,
                ..ServerCapabilities::default()
            },
            ..InitializeResult::default()
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "lust-analyzer initialized")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: tower_lsp::lsp_types::DidOpenTextDocumentParams) {
        let document = params.text_document;
        {
            let mut docs = self.documents.write().await;
            docs.insert(
                document.uri.clone(),
                DocumentState {
                    text: document.text,
                    version: document.version,
                },
            );
        }

        self.analyze(&document.uri).await;
    }

    async fn did_change(&self, params: tower_lsp::lsp_types::DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;
        let version = params.text_document.version;
        if let Some(change) = params.content_changes.last() {
            {
                let mut docs = self.documents.write().await;
                if let Some(doc) = docs.get_mut(&uri) {
                    doc.text = change.text.clone();
                    doc.version = version;
                }
            }

            self.analyze(&uri).await;
        } else {
            self.client
                .log_message(
                    MessageType::WARNING,
                    "didChange event received without content changes",
                )
                .await;
        }
    }

    async fn did_close(&self, params: tower_lsp::lsp_types::DidCloseTextDocumentParams) {
        let uri = params.text_document.uri;
        {
            let mut docs = self.documents.write().await;
            docs.remove(&uri);
        }

        let associated = {
            let mut tracker = self.last_published.write().await;
            tracker.remove(&uri).unwrap_or_default()
        };
        let version_lookup = {
            let docs = self.documents.read().await;
            docs.iter()
                .map(|(u, state)| (u.clone(), state.version))
                .collect::<HashMap<_, _>>()
        };
        for related_uri in associated {
            let version = version_lookup.get(&related_uri).copied();
            self.client
                .publish_diagnostics(related_uri.clone(), Vec::new(), version)
                .await;
        }

        self.client.publish_diagnostics(uri, Vec::new(), None).await;
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;
        let mut text = match self.document_text(&uri).await {
            Some(text) => text,
            None => return Ok(None),
        };
        let file_path = match uri.to_file_path() {
            Ok(path) => path,
            Err(_) => return Ok(None),
        };
        let analysis = self.analysis.read().await;
        let snapshot = match analysis.as_ref() {
            Some(snapshot) => snapshot,
            None => return Ok(None),
        };
        let module = match snapshot.module_for_file(&file_path) {
            Some(module) => module,
            None => return Ok(None),
        };
        let module_path = snapshot
            .module_path_for_file(&file_path)
            .map(|s| s.to_string());
        let mut line_offsets = compute_line_offsets(&text);
        let mut offset = match position_to_offset(&text, position, &line_offsets) {
            Some(value) => value,
            None => return Ok(None),
        };
        if let Some(context_info) = params.context.as_ref() {
            if context_info.trigger_kind == CompletionTriggerKind::TRIGGER_CHARACTER {
                if let Some(trigger_str) = context_info.trigger_character.as_ref() {
                    let mut chars = trigger_str.chars();
                    if let Some(trigger_char) = chars.next() {
                        if chars.next().is_none() {
                            let has_trigger = prev_char_index(&text, offset)
                                .map(|(_, ch)| ch == trigger_char)
                                .unwrap_or(false);
                            if !has_trigger {
                                if offset <= text.len() {
                                    text.insert(offset, trigger_char);
                                } else {
                                    text.push(trigger_char);
                                }

                                line_offsets = compute_line_offsets(&text);
                                offset = match position_to_offset(&text, position, &line_offsets) {
                                    Some(value) => value,
                                    None => return Ok(None),
                                };
                            }
                        }
                    }
                }
            }
        }

        let mut context = analyze_member_method_context(&text, offset)
            .or_else(|| analyze_pattern_context(&text, offset));
        if context.is_none() {
            context = analyze_module_path_context(&text, offset, module, &position);
        }

        if context.is_none() {
            context = analyze_identifier_context(&text, offset);
        }

        let Some(context) = context else {
            return Ok(None);
        };
        let mut items: Vec<CompletionItem> = Vec::new();
        match context.kind {
            CompletionKind::Member | CompletionKind::Method | CompletionKind::Pattern => {
                let base_name = resolve_base_type_name_for_context(
                    module,
                    snapshot,
                    module_path.as_deref(),
                    &text,
                    &line_offsets,
                    &context,
                );
                match context.kind {
                    CompletionKind::Member => {
                        if let Some(owner) = base_name.as_deref() {
                            items.extend(struct_field_completions(
                                snapshot,
                                module_path.as_deref(),
                                owner,
                                &context.prefix,
                            ));
                        }

                        if let Some(object_name) = context.object_name.as_ref() {
                            let candidates = resolve_type_candidates(object_name, module);
                            let mut seen_candidates = HashSet::new();
                            for candidate in candidates {
                                if !seen_candidates.insert(candidate.clone()) {
                                    continue;
                                }

                                items.extend(enum_variant_completions(
                                    snapshot,
                                    module_path.as_deref(),
                                    &candidate,
                                    &context.prefix,
                                ));
                                items.extend(static_method_completions(
                                    snapshot,
                                    &candidate,
                                    module_path.as_deref(),
                                    &context.prefix,
                                ));
                            }

                            items.extend(builtin_global_completions(object_name, &context.prefix));
                            items.extend(builtin_static_method_completions(
                                object_name,
                                &context.prefix,
                            ));
                            items.extend(module_alias_member_completions(
                                snapshot,
                                module,
                                &context.prefix,
                                object_name,
                            ));
                        }
                    }

                    CompletionKind::Method => {
                        if let Some(owner) = base_name.as_deref() {
                            items.extend(instance_method_completions(
                                snapshot,
                                owner,
                                module_path.as_deref(),
                                &context.prefix,
                            ));
                            items.extend(builtin_instance_method_completions(
                                owner,
                                &context.prefix,
                            ));
                        }
                    }

                    CompletionKind::Pattern => {
                        if let Some(owner) = base_name.as_deref() {
                            items.extend(enum_variant_completions(
                                snapshot,
                                module_path.as_deref(),
                                owner,
                                &context.prefix,
                            ));
                        }
                    }

                    _ => {}
                }
            }

            CompletionKind::Identifier => {
                items.extend(identifier_completions(
                    module,
                    snapshot,
                    &file_path,
                    position,
                    &context.prefix,
                ));
            }

            CompletionKind::ModulePath => {
                items.extend(module_path_completions(
                    snapshot,
                    module,
                    &context.path_segments,
                    &context.prefix,
                ));
            }
        }

        if items.is_empty() {
            return Ok(None);
        }

        let mut unique = Vec::new();
        let mut seen_labels = HashSet::new();
        for item in items.into_iter() {
            if seen_labels.insert(item.label.clone()) {
                unique.push(item);
            }
        }

        Ok(Some(CompletionResponse::Array(unique)))
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;
        let file_path = match uri.to_file_path() {
            Ok(path) => path,
            Err(_) => return Ok(None),
        };
        let text = self.document_text(&uri).await;
        let word = text
            .as_ref()
            .and_then(|source| extract_word_at_position(source, position));
        let (method_hover, def_opt, type_opt) = {
            let analysis = self.analysis.read().await;
            let Some(snapshot) = analysis.as_ref() else {
                return Ok(None);
            };
            let module_path = snapshot
                .module_path_for_file(&file_path)
                .map(|s| s.to_string());
            let module = snapshot.module_for_file(&file_path);
            let method_hover = if let (Some(module), Some(source), Some(token)) =
                (module, text.as_ref(), word.as_deref())
            {
                hover_for_method_call(
                    snapshot,
                    module,
                    module_path.as_deref(),
                    source,
                    position,
                    token,
                )
            } else {
                None
            };
            let mut def_clone = None;
            if let Some(defs) = snapshot.definitions_in_file(&file_path) {
                if let Some(def) = defs
                    .iter()
                    .find(|d| span_contains_position(d.span, &position))
                {
                    def_clone = Some(def.clone());
                }
            }

            if def_clone.is_none() {
                if let Some(word) = word.as_ref() {
                    if let Some(def) = snapshot.definition_by_qualified(word) {
                        def_clone = Some(def.clone());
                    } else if let Some(defs) = snapshot.definitions_by_simple(word) {
                        if let Some(def) = choose_definition(defs, module_path.as_deref()) {
                            def_clone = Some(def.clone());
                        }
                    }
                }
            }

            let type_opt = if def_clone.is_none() {
                module
                    .and_then(|module| find_type_for_position(module, position))
                    .map(|(span, ty)| {
                        let type_def = match &ty.kind {
                            TypeKind::Named(name) => snapshot
                                .definition_by_qualified(name)
                                .cloned()
                                .or_else(|| {
                                    if let Some(mp) = module_path.as_ref() {
                                        let qualified = format!("{}.{}", mp, name);
                                        snapshot.definition_by_qualified(&qualified).cloned()
                                    } else {
                                        None
                                    }
                                })
                                .or_else(|| {
                                    snapshot
                                        .definitions_by_simple(name)
                                        .and_then(|defs| {
                                            choose_definition(defs, module_path.as_deref())
                                        })
                                        .cloned()
                                }),
                            _ => None,
                        };
                        (span, ty, type_def)
                    })
            } else {
                None
            };
            (method_hover, def_clone, type_opt)
        };
        if let Some(hover) = method_hover {
            return Ok(Some(hover));
        }

        if let Some(def) = def_opt {
            return Ok(Some(hover_from_definition(&def)));
        }

        if let Some((span, ty, type_def)) = type_opt {
            if let Some(def) = type_def {
                return Ok(Some(hover_from_definition(&def)));
            }

            let hover = Hover {
                contents: HoverContents::Markup(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: format!("`{}`", ty),
                }),
                range: Some(span_to_range(span)),
            };
            return Ok(Some(hover));
        }

        Ok(None)
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;
        let file_path = match uri.to_file_path() {
            Ok(path) => path,
            Err(_) => return Ok(None),
        };
        let text = self.document_text(&uri).await;
        let word = text
            .as_ref()
            .and_then(|source| extract_word_at_position(source, position));
        let def_opt = {
            let analysis = self.analysis.read().await;
            let Some(snapshot) = analysis.as_ref() else {
                return Ok(None);
            };
            let mut def_clone = None;
            if let Some(defs) = snapshot.definitions_in_file(&file_path) {
                if let Some(def) = defs
                    .iter()
                    .find(|d| span_contains_position(d.span, &position))
                {
                    def_clone = Some(def.clone());
                }
            }

            if def_clone.is_none() {
                let module_path = snapshot.module_path_for_file(&file_path);
                if let Some(word) = word.as_ref() {
                    if let Some(def) = snapshot.definition_by_qualified(word) {
                        def_clone = Some(def.clone());
                    } else if let Some(defs) = snapshot.definitions_by_simple(word) {
                        if let Some(def) = choose_definition(defs, module_path) {
                            def_clone = Some(def.clone());
                        }
                    }
                }
            }

            def_clone
        };
        if let Some(def) = def_opt {
            if let Some(location) = location_from_definition(&def) {
                return Ok(Some(GotoDefinitionResponse::Scalar(location)));
            }
        }

        Ok(None)
    }

    async fn inlay_hint(&self, params: InlayHintParams) -> Result<Option<Vec<InlayHint>>> {
        let uri = params.text_document.uri;
        let range = params.range;
        let file_path = match uri.to_file_path() {
            Ok(path) => path,
            Err(_) => return Ok(None),
        };
        let hints = {
            let analysis = self.analysis.read().await;
            let Some(snapshot) = analysis.as_ref() else {
                return Ok(None);
            };
            snapshot
                .module_for_file(&file_path)
                .map(|module| collect_inlay_hints_for_module(module, &range))
        };
        Ok(hints.or(Some(Vec::new())))
    }

    async fn semantic_tokens_full(
        &self,
        params: SemanticTokensParams,
    ) -> Result<Option<SemanticTokensResult>> {
        let uri = params.text_document.uri;
        let path = match uri.to_file_path() {
            Ok(path) => path,
            Err(_) => return Ok(None),
        };
        let tokens = self.semantic_tokens_for_path(&path).await;
        Ok(tokens.map(|data| {
            SemanticTokensResult::Tokens(SemanticTokens {
                result_id: None,
                data,
            })
        }))
    }

    async fn semantic_tokens_range(
        &self,
        _params: SemanticTokensRangeParams,
    ) -> Result<Option<SemanticTokensRangeResult>> {
        Ok(None)
    }
}

pub async fn run() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let (service, socket) = LspService::build(|client| Backend::new(client)).finish();
    Server::new(stdin, stdout, socket).serve(service).await;
}

#[cfg(test)]
mod tests {
    include!("tests.rs");
}

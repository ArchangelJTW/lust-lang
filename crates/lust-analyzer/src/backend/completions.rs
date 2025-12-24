use crate::analysis::{
    find_type_for_position, AnalysisSnapshot, MethodInfo, ModuleSnapshot, TypeDefinitionKind,
};
use crate::utils::{
    base_type_name, char_at_index, identifier_name_at_span, identifier_prefix_range,
    identifier_range_before, identifier_text, is_identifier_char, method_display_name,
    offset_to_position, prev_char_index, qualify_type_name, simple_type_name,
    span_contains_position, span_from_identifier, span_start_before_or_equal, span_starts_after,
    split_type_member,
};
use hashbrown::HashSet;
use lust::ast::{
    ExternItem, FunctionParam, Item, ItemKind, Stmt, StmtKind, Type, TypeKind, Visibility,
};
use lust::builtins::{self, BuiltinFunction, BuiltinMethod, TypeExpr};
use lust::Span;
use std::path::Path;
use tower_lsp::lsp_types::{
    CompletionItem, CompletionItemKind, Documentation, InsertTextFormat, MarkupContent, MarkupKind,
    Position,
};

pub(crate) enum CompletionKind {
    Member,
    Method,
    Pattern,
    Identifier,
    ModulePath,
}

pub(crate) struct CompletionContext {
    pub(crate) kind: CompletionKind,
    pub(crate) object_start: Option<usize>,
    pub(crate) object_end: usize,
    pub(crate) object_name: Option<String>,
    pub(crate) prefix: String,
    pub(crate) path_segments: Vec<String>,
}

pub(crate) fn format_method_signature(method: &MethodInfo) -> String {
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

pub(crate) fn struct_field_completions(
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

pub(crate) fn enum_variant_completions(
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

pub(crate) fn identifier_completions(
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
                        ExternItem::Const { name, .. } => {
                            push_item(
                                name.clone(),
                                CompletionItemKind::CONSTANT,
                                Some("extern const".to_string()),
                            );
                        }
                        ExternItem::Struct(def) => {
                            push_item(
                                simple_type_name(&def.name).to_string(),
                                CompletionItemKind::STRUCT,
                                Some("extern struct".to_string()),
                            );
                        }
                        ExternItem::Enum(def) => {
                            push_item(
                                simple_type_name(&def.name).to_string(),
                                CompletionItemKind::ENUM,
                                Some("extern enum".to_string()),
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
        || snapshot.has_dependency_root(&resolved_path)
    {
        Some(resolved_path)
    } else {
        None
    }
}

pub(crate) fn module_alias_member_completions(
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
            if !is_valid_module_identifier(&child) {
                continue;
            }
            push_item(
                child,
                CompletionItemKind::MODULE,
                Some("module".to_string()),
            );
        }
    }
    if resolved_path.is_empty() {
        let mut dependency_roots: Vec<_> = snapshot.dependency_roots().cloned().collect();
        dependency_roots.sort();
        for child in dependency_roots {
            if !is_valid_module_identifier(&child) {
                continue;
            }
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

pub(crate) fn module_path_completions(
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
            if !is_valid_module_identifier(&child) {
                continue;
            }
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

#[cfg(test)]
pub(crate) fn prewarm_builtins() {
    let _ = builtins::builtins();
}

pub(crate) fn builtin_global_completions(object_name: &str, prefix: &str) -> Vec<CompletionItem> {
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

pub(crate) fn builtin_static_method_completions(
    _owner: &str,
    _prefix: &str,
) -> Vec<CompletionItem> {
    Vec::new()
}

pub(crate) fn builtin_instance_method_completions(
    owner: &str,
    prefix: &str,
) -> Vec<CompletionItem> {
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

pub(crate) fn static_method_completions(
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

pub(crate) fn instance_method_completions(
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

pub(crate) fn type_for_identifier(
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

pub(crate) fn analyze_member_method_context(
    text: &str,
    offset: usize,
) -> Option<CompletionContext> {
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

pub(crate) fn analyze_pattern_context(text: &str, offset: usize) -> Option<CompletionContext> {
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

pub(crate) fn analyze_identifier_context(text: &str, offset: usize) -> Option<CompletionContext> {
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

pub(crate) fn analyze_module_path_context(
    text: &str,
    offset: usize,
    module: &ModuleSnapshot,
    position: &Position,
) -> Option<CompletionContext> {
    let (segments, prefix) = parse_module_path_at(text, offset)?;
    if !is_within_use_item(&module.module.items, position)
        && !appears_like_use_statement(text, offset)
    {
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

fn is_valid_module_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !is_identifier_start(first) {
        return false;
    }

    chars.all(is_identifier_char)
}

fn is_identifier_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

fn appears_like_use_statement(text: &str, offset: usize) -> bool {
    if text.is_empty() {
        return false;
    }

    let cursor = offset.min(text.len());
    let line_start = text[..cursor].rfind('\n').map(|idx| idx + 1).unwrap_or(0);
    let snippet = text[line_start..cursor].trim_start();
    if !snippet.starts_with("use") {
        return false;
    }
    let after_use = snippet.strip_prefix("use").unwrap_or(snippet);
    matches!(
        after_use.chars().next(),
        None | Some(' ') | Some('\t') | Some('{')
    )
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

pub(crate) fn resolve_type_candidates(type_name: &str, module: &ModuleSnapshot) -> Vec<String> {
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

pub(crate) fn infer_struct_literal_base_name(
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

pub(crate) fn infer_constructor_call_base_name(
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

pub(crate) fn resolve_base_type_name_for_context(
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

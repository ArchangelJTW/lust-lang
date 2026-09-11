use crate::analysis::AnalysisSnapshot;
use crate::utils::{simple_type_name, span_to_range};
use lust::ast::{Item, ItemKind};
use std::path::Path;
use tower_lsp::lsp_types::{
    DocumentSymbol, DocumentSymbolResponse, Location, Position, Range, SymbolInformation,
    SymbolKind,
};
use url::Url;

pub(crate) fn document_symbols(
    snapshot: &AnalysisSnapshot,
    file_path: &Path,
) -> Option<DocumentSymbolResponse> {
    let module_snapshot = snapshot.module_for_file(file_path)?;
    let mut symbols = Vec::new();

    for item in &module_snapshot.module.items {
        collect_item_symbols(item, &mut symbols);
    }

    Some(DocumentSymbolResponse::Nested(symbols))
}

fn collect_item_symbols(item: &Item, symbols: &mut Vec<DocumentSymbol>) {
    #[allow(deprecated)]
    match &item.kind {
        ItemKind::Struct(def) => {
            let simple = simple_type_name(&def.name).to_string();
            let range = span_to_range(item.span);
            let mut children = Vec::new();
            for field in &def.fields {
                let frange = span_to_range(field.ty.span);
                children.push(DocumentSymbol {
                    name: field.name.clone(),
                    detail: Some(format!("{}", field.ty)),
                    kind: SymbolKind::FIELD,
                    tags: None,
                    deprecated: None,
                    range: frange,
                    selection_range: frange,
                    children: None,
                });
            }
            symbols.push(DocumentSymbol {
                name: simple,
                detail: Some("struct".to_string()),
                kind: SymbolKind::STRUCT,
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: if children.is_empty() {
                    None
                } else {
                    Some(children)
                },
            });
        }

        ItemKind::Enum(def) => {
            let simple = simple_type_name(&def.name).to_string();
            let range = span_to_range(item.span);
            let mut children = Vec::new();
            for variant in &def.variants {
                let detail = variant.fields.as_ref().map(|f| {
                    let types = f
                        .iter()
                        .map(|t| t.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("({types})")
                });
                children.push(DocumentSymbol {
                    name: variant.name.clone(),
                    detail,
                    kind: SymbolKind::ENUM_MEMBER,
                    tags: None,
                    deprecated: None,
                    range,
                    selection_range: range,
                    children: None,
                });
            }
            symbols.push(DocumentSymbol {
                name: simple,
                detail: Some("enum".to_string()),
                kind: SymbolKind::ENUM,
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: if children.is_empty() {
                    None
                } else {
                    Some(children)
                },
            });
        }

        ItemKind::Trait(def) => {
            let simple = simple_type_name(&def.name).to_string();
            let range = span_to_range(item.span);
            let mut children = Vec::new();
            for method in &def.methods {
                let mname = simple_type_name(&method.name).to_string();
                let detail = method.return_type.as_ref().map(|r| format!("-> {r}"));
                children.push(DocumentSymbol {
                    name: mname,
                    detail,
                    kind: SymbolKind::METHOD,
                    tags: None,
                    deprecated: None,
                    range,
                    selection_range: range,
                    children: None,
                });
            }
            symbols.push(DocumentSymbol {
                name: simple,
                detail: Some("trait".to_string()),
                kind: SymbolKind::INTERFACE,
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: if children.is_empty() {
                    None
                } else {
                    Some(children)
                },
            });
        }

        ItemKind::Impl(impl_block) => {
            let target_name = format!("{}", impl_block.target_type);
            let label = if let Some(trait_name) = &impl_block.trait_name {
                format!("impl {trait_name} for {target_name}")
            } else {
                format!("impl {target_name}")
            };
            let range = span_to_range(item.span);
            let mut children = Vec::new();
            for method in &impl_block.methods {
                let mname = simple_type_name(&method.name).to_string();
                let detail = method.return_type.as_ref().map(|r| format!("-> {r}"));
                children.push(DocumentSymbol {
                    name: mname,
                    detail,
                    kind: SymbolKind::METHOD,
                    tags: None,
                    deprecated: None,
                    range,
                    selection_range: range,
                    children: None,
                });
            }
            symbols.push(DocumentSymbol {
                name: label,
                detail: None,
                kind: SymbolKind::NAMESPACE,
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: if children.is_empty() {
                    None
                } else {
                    Some(children)
                },
            });
        }

        ItemKind::Function(func) => {
            let name = simple_type_name(&func.name).to_string();
            let range = span_to_range(item.span);
            let detail = func.return_type.as_ref().map(|r| format!("-> {r}"));
            symbols.push(DocumentSymbol {
                name,
                detail,
                kind: if func.is_method {
                    SymbolKind::METHOD
                } else {
                    SymbolKind::FUNCTION
                },
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: None,
            });
        }

        ItemKind::Module { items, name } => {
            let range = span_to_range(item.span);
            let mut children = Vec::new();
            for child in items {
                collect_item_symbols(child, &mut children);
            }
            symbols.push(DocumentSymbol {
                name: name.clone(),
                detail: Some("module".to_string()),
                kind: SymbolKind::MODULE,
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: if children.is_empty() {
                    None
                } else {
                    Some(children)
                },
            });
        }

        _ => {}
    }
}

pub(crate) fn workspace_symbols(
    snapshot: &AnalysisSnapshot,
    query: &str,
) -> Vec<SymbolInformation> {
    let query_lower = query.to_lowercase();
    let mut results = Vec::new();

    let matches_query =
        |name: &str| -> bool { query.is_empty() || name.to_lowercase().contains(&query_lower) };

    #[allow(deprecated)]
    for info in snapshot.all_structs() {
        let simple = simple_type_name(&info.def.name);
        if matches_query(simple) {
            if let Ok(uri) = Url::from_file_path(&info.file_path) {
                results.push(SymbolInformation {
                    name: simple.to_string(),
                    kind: SymbolKind::STRUCT,
                    tags: None,
                    deprecated: None,
                    location: Location {
                        uri,
                        range: span_to_range(info.span),
                    },
                    container_name: if info.module_path.is_empty() {
                        None
                    } else {
                        Some(info.module_path.clone())
                    },
                });
            }
        }
    }

    #[allow(deprecated)]
    for info in snapshot.all_enums() {
        let simple = simple_type_name(&info.def.name);
        if matches_query(simple) {
            if let Ok(uri) = Url::from_file_path(&info.file_path) {
                results.push(SymbolInformation {
                    name: simple.to_string(),
                    kind: SymbolKind::ENUM,
                    tags: None,
                    deprecated: None,
                    location: Location {
                        uri,
                        range: span_to_range(info.span),
                    },
                    container_name: if info.module_path.is_empty() {
                        None
                    } else {
                        Some(info.module_path.clone())
                    },
                });
            }
        }
        for variant in &info.def.variants {
            if matches_query(&variant.name) {
                if let Ok(uri) = Url::from_file_path(&info.file_path) {
                    results.push(SymbolInformation {
                        name: variant.name.clone(),
                        kind: SymbolKind::ENUM_MEMBER,
                        tags: None,
                        deprecated: None,
                        location: Location {
                            uri,
                            range: Range::new(Position::new(0, 0), Position::new(0, 0)),
                        },
                        container_name: Some(simple.to_string()),
                    });
                }
            }
        }
    }

    #[allow(deprecated)]
    for info in snapshot.all_traits() {
        let simple = simple_type_name(&info.def.name);
        if matches_query(simple) {
            if let Ok(uri) = Url::from_file_path(&info.file_path) {
                results.push(SymbolInformation {
                    name: simple.to_string(),
                    kind: SymbolKind::INTERFACE,
                    tags: None,
                    deprecated: None,
                    location: Location {
                        uri,
                        range: span_to_range(info.span),
                    },
                    container_name: if info.module_path.is_empty() {
                        None
                    } else {
                        Some(info.module_path.clone())
                    },
                });
            }
        }
    }

    #[allow(deprecated)]
    for info in snapshot.all_functions() {
        if matches_query(&info.name) {
            if let Ok(uri) = Url::from_file_path(&info.file_path) {
                results.push(SymbolInformation {
                    name: info.name.clone(),
                    kind: SymbolKind::FUNCTION,
                    tags: None,
                    deprecated: None,
                    location: Location {
                        uri,
                        range: span_to_range(info.span),
                    },
                    container_name: if info.module_path.is_empty() {
                        None
                    } else {
                        Some(info.module_path.clone())
                    },
                });
            }
        }
    }

    #[allow(deprecated)]
    for info in snapshot.all_methods() {
        if matches_query(&info.name) {
            if let Ok(uri) = Url::from_file_path(&info.file_path) {
                results.push(SymbolInformation {
                    name: info.name.clone(),
                    kind: SymbolKind::METHOD,
                    tags: None,
                    deprecated: None,
                    location: Location {
                        uri,
                        range: span_to_range(info.span),
                    },
                    container_name: Some(info.owner.clone()),
                });
            }
        }
    }

    results
}

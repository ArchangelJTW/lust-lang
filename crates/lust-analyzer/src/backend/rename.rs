use crate::analysis::AnalysisSnapshot;
use crate::utils::{is_reserved_keyword, is_valid_identifier_name};
use std::collections::HashMap;
use std::path::Path;
use tower_lsp::lsp_types::{Position, PrepareRenameResponse, TextEdit, WorkspaceEdit};
use url::Url;

pub(crate) fn prepare_rename(
    snapshot: &AnalysisSnapshot,
    file_path: &Path,
    position: Position,
) -> Option<PrepareRenameResponse> {
    let occ = snapshot.find_symbol_at_position(file_path, &position)?;
    if !occ.symbol.is_renameable() {
        return None;
    }
    Some(PrepareRenameResponse::RangeWithPlaceholder {
        range: occ.range,
        placeholder: occ.symbol.display_name().to_string(),
    })
}

pub(crate) fn rename_symbol(
    snapshot: &AnalysisSnapshot,
    file_path: &Path,
    position: Position,
    new_name: &str,
) -> Result<Option<WorkspaceEdit>, String> {
    let new_name = new_name.trim();
    if new_name.is_empty() {
        return Err("Symbol name cannot be empty".to_string());
    }
    if !is_valid_identifier_name(new_name) {
        return Err(format!("'{new_name}' is not a valid Lust identifier"));
    }
    if is_reserved_keyword(new_name) {
        return Err(format!("'{new_name}' is a reserved keyword in Lust"));
    }

    let occ = match snapshot.find_symbol_at_position(file_path, &position) {
        Some(o) => o,
        None => return Ok(None),
    };

    if !occ.symbol.is_renameable() {
        return Err(format!(
            "Cannot rename built-in symbol '{}'",
            occ.symbol.display_name()
        ));
    }

    let occurrences = snapshot.symbol_references(&occ.symbol);
    if occurrences.is_empty() {
        return Ok(None);
    }

    let mut changes_map: HashMap<Url, Vec<TextEdit>> = HashMap::new();
    for occurrence in occurrences {
        if let Ok(uri) = Url::from_file_path(&occurrence.file_path) {
            let edits = changes_map.entry(uri).or_default();
            if !edits.iter().any(|e| e.range == occurrence.range) {
                edits.push(TextEdit {
                    range: occurrence.range,
                    new_text: new_name.to_string(),
                });
            }
        }
    }

    // Sort edits in descending order of range for deterministic and safe application
    for edits in changes_map.values_mut() {
        edits.sort_by(|a, b| {
            b.range
                .start
                .line
                .cmp(&a.range.start.line)
                .then_with(|| b.range.start.character.cmp(&a.range.start.character))
        });
    }

    Ok(Some(WorkspaceEdit {
        changes: Some(changes_map),
        document_changes: None,
        change_annotations: None,
    }))
}

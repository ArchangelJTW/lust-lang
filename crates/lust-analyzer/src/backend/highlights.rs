use crate::analysis::AnalysisSnapshot;
use std::path::Path;
use tower_lsp::lsp_types::{DocumentHighlight, DocumentHighlightKind, Position};

pub(crate) fn document_highlights(
    snapshot: &AnalysisSnapshot,
    file_path: &Path,
    position: Position,
) -> Option<Vec<DocumentHighlight>> {
    let occ = snapshot.find_symbol_at_position(file_path, &position)?;
    let occurrences = snapshot.symbol_references(&occ.symbol);
    let mut highlights = Vec::new();
    for occurrence in occurrences {
        if occurrence.file_path == file_path {
            let kind = if occurrence.is_definition {
                Some(DocumentHighlightKind::WRITE)
            } else {
                Some(DocumentHighlightKind::READ)
            };
            highlights.push(DocumentHighlight {
                range: occurrence.range,
                kind,
            });
        }
    }
    Some(highlights)
}

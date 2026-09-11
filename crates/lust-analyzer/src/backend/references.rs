use crate::analysis::AnalysisSnapshot;
use std::path::Path;
use tower_lsp::lsp_types::{Location, Position};
use url::Url;

pub(crate) fn find_references(
    snapshot: &AnalysisSnapshot,
    file_path: &Path,
    position: Position,
    include_declaration: bool,
) -> Option<Vec<Location>> {
    let occ = snapshot.find_symbol_at_position(file_path, &position)?;
    let occurrences = snapshot.symbol_references(&occ.symbol);
    let mut locations = Vec::new();
    for occurrence in occurrences {
        if !include_declaration && occurrence.is_definition {
            continue;
        }
        if let Ok(uri) = Url::from_file_path(&occurrence.file_path) {
            locations.push(Location {
                uri,
                range: occurrence.range,
            });
        }
    }
    Some(locations)
}

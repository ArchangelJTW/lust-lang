use lust::LustError;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tower_lsp::lsp_types::{Diagnostic, DiagnosticSeverity, Position, Range};
pub(crate) fn error_to_diagnostics(
    error: LustError,
    entry_path: &Path,
    entry_dir: &Path,
    module_paths: &HashMap<String, PathBuf>,
) -> HashMap<PathBuf, Vec<Diagnostic>> {
    let mut result: HashMap<PathBuf, Vec<Diagnostic>> = HashMap::new();
    let mut push = |path: PathBuf, diagnostic: Diagnostic| {
        result.entry(path).or_default().push(diagnostic);
    };
    match error {
        LustError::LexerError {
            line,
            column,
            message,
            module,
        } => {
            let path = resolve_module_path(entry_path, entry_dir, module_paths, module.as_deref());
            push(path, make_diagnostic(message, point_range(line, column)));
        }

        LustError::ParserError {
            line,
            column,
            message,
            module,
        } => {
            let path = resolve_module_path(entry_path, entry_dir, module_paths, module.as_deref());
            push(path, make_diagnostic(message, point_range(line, column)));
        }

        LustError::TypeError { message } => {
            push(
                entry_path.to_path_buf(),
                make_diagnostic(message, point_range(1, 1)),
            );
        }

        LustError::TypeErrorWithSpan {
            message,
            line,
            column,
            module,
        } => {
            let path = resolve_module_path(entry_path, entry_dir, module_paths, module.as_deref());
            push(path, make_diagnostic(message, point_range(line, column)));
        }

        LustError::CompileError(message) => {
            push(
                entry_path.to_path_buf(),
                make_diagnostic(message, point_range(1, 1)),
            );
        }

        LustError::CompileErrorWithSpan {
            message,
            line,
            column,
            module,
        } => {
            let path = resolve_module_path(entry_path, entry_dir, module_paths, module.as_deref());
            push(path, make_diagnostic(message, point_range(line, column)));
        }

        LustError::Unknown(message) => {
            push(
                entry_path.to_path_buf(),
                make_diagnostic(message, point_range(1, 1)),
            );
        }

        other => {
            push(
                entry_path.to_path_buf(),
                make_diagnostic(other.to_string(), point_range(1, 1)),
            );
        }
    }

    result
}

fn make_diagnostic(message: String, range: Range) -> Diagnostic {
    Diagnostic {
        range,
        severity: Some(DiagnosticSeverity::ERROR),
        code: None,
        code_description: None,
        source: Some("lust-analyzer".to_string()),
        message,
        related_information: None,
        tags: None,
        data: None,
    }
}

fn point_range(line: usize, column: usize) -> Range {
    let line = line.saturating_sub(1) as u32;
    let column = column.saturating_sub(1) as u32;
    Range {
        start: Position::new(line, column),
        end: Position::new(line, column),
    }
}

fn resolve_module_path(
    entry_path: &Path,
    entry_dir: &Path,
    module_paths: &HashMap<String, PathBuf>,
    module: Option<&str>,
) -> PathBuf {
    if let Some(module_name) = module {
        if let Some(path) = module_paths.get(module_name) {
            return path.clone();
        }

        build_module_path(entry_dir, module_name)
    } else {
        entry_path.to_path_buf()
    }
}

fn build_module_path(base_dir: &Path, module: &str) -> PathBuf {
    let mut path = base_dir.to_path_buf();
    for segment in module.split('.') {
        path.push(segment);
    }

    path.set_extension("lust");
    path
}

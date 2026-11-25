//! Structured compiler/runtime diagnostics for command line consumers.

pub use lust::error::{DiagnosticBag, IntoDiagnostic};
use std::fmt::Display;

/// Escapes special characters for JSON strings.
pub fn escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

/// Legacy single-line JSON diagnostic formatter.
pub fn json(
    message: impl Display,
    kind: &str,
    line: Option<usize>,
    column: Option<usize>,
) -> String {
    let mut out = format!(
        "{{\"kind\":\"{}\",\"message\":\"{}\"",
        escape(kind),
        escape(&message.to_string())
    );
    if let Some(v) = line {
        out.push_str(&format!(",\"line\":{v}"));
    }
    if let Some(v) = column {
        out.push_str(&format!(",\"column\":{v}"));
    }
    out.push_str("}\n");
    out
}

/// Emits a single diagnostic or bag to stderr as a single-line JSON string.
pub fn emit_diagnostics(bag: &DiagnosticBag) {
    eprintln!("{}", bag.to_json_line());
}

/// Emits a LustError as structured JSON diagnostics to stderr.
#[allow(dead_code)]
pub fn emit_error_diagnostic(source: Option<&str>, filename: &str, error: &lust::LustError) {
    let mut bag = DiagnosticBag::new();
    bag.add(error.clone().into_diagnostic(Some(filename), source));
    emit_diagnostics(&bag);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_and_json_helpers() {
        assert_eq!(escape("a\\b\"c\nd\re"), "a\\\\b\\\"c\\nd\\re");
        let formatted = json("syntax error", "error", Some(12), Some(4));
        assert!(formatted.contains("\"kind\":\"error\""));
        assert!(formatted.contains("\"message\":\"syntax error\""));
        assert!(formatted.contains("\"line\":12"));
        assert!(formatted.contains("\"column\":4"));
    }
}

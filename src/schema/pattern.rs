//! Regex pattern translation and mode distinction between Lust and JSON Schema (DS-91).
//!
//! - Lust `.regex(...)` has full-match semantics (`\A(?:pattern)\z`).
//! - JSON Schema `pattern` has ECMA-262 search-mode semantics (`contains_pattern`).

use crate::schema::interop::{SchemaExportError, SchemaImportError};
use std::fmt;

/// Execution mode for pattern matching in schema validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternMode {
    /// Full-match mode (Lust `.regex(...)` default: string matches `\A(?:pattern)\z`).
    FullMatch,
    /// Search mode (JSON Schema `pattern`: string contains a match for `pattern`).
    Search,
}

impl fmt::Display for PatternMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatternMode::FullMatch => write!(f, "FullMatch"),
            PatternMode::Search => write!(f, "Search"),
        }
    }
}

/// Verify dialect portability and wrap a Lust full-match regex pattern for JSON Schema export.
///
/// Fails closed with `SchemaExportError::NonPortablePattern` if inline flags or multiline
/// modifiers alter `^...$` semantics.
pub fn export_regex_pattern(pattern: &str) -> Result<String, SchemaExportError> {
    // Check for inline flags or multiline modifiers like (?m), (?i), (?s), (?x), (?U)
    if pattern.contains("(?m") || pattern.contains("(?i") || pattern.contains("(?s")
        || pattern.contains("(?x") || pattern.contains("(?U") || pattern.contains("(?")
    {
        return Err(SchemaExportError::NonPortablePattern(format!(
            "Inline flags in pattern '{}' are non-portable to ECMA-262 JSON Schema pattern",
            pattern
        )));
    }

    // Verify pattern compiles in bounded regex engine
    if let Err(e) = crate::regex::BoundedRegex::new(pattern) {
        return Err(SchemaExportError::NonPortablePattern(format!(
            "Pattern '{}' cannot be compiled by bounded engine: {}",
            pattern, e
        )));
    }

    // Wrap in ^(?:...)$ to enforce full-match semantics in JSON Schema search-mode validator
    if pattern.starts_with('^') && pattern.ends_with('$') {
        Ok(pattern.to_string())
    } else {
        Ok(format!("^(?:{})$", pattern))
    }
}

/// Import a JSON Schema `pattern` into a search-mode pattern plan.
///
/// Compiles into search-mode (`PatternMode::Search`) bounded DFA RegexPlan.
pub fn import_regex_pattern(pattern: &str) -> Result<String, SchemaImportError> {
    if let Err(e) = crate::regex::BoundedRegex::new(pattern) {
        return Err(SchemaImportError::NonPortablePattern(format!(
            "JSON Schema pattern '{}' cannot be compiled into bounded DFA: {}",
            pattern, e
        )));
    }
    Ok(pattern.to_string())
}

/// Execute pattern match against text according to `PatternMode`.
pub fn matches_pattern(pattern: &str, mode: PatternMode, text: &str) -> Result<bool, crate::regex::RegexError> {
    match mode {
        PatternMode::Search => matches_search(pattern, text),
        PatternMode::FullMatch => matches_full(pattern, text),
    }
}

/// Check if text contains a match for pattern (JSON Schema search-mode).
pub fn matches_search(pattern: &str, text: &str) -> Result<bool, crate::regex::RegexError> {
    let re = crate::regex::BoundedRegex::new(pattern)?;
    let found = re.find(text)?;
    Ok(found.is_some())
}

/// Check if text fully matches pattern (Lust full-match mode).
pub fn matches_full(pattern: &str, text: &str) -> Result<bool, crate::regex::RegexError> {
    let anchored_pattern = if pattern.starts_with('^') && pattern.ends_with('$') {
        pattern.to_string()
    } else {
        format!("^(?:{})$", pattern)
    };

    match crate::regex::BoundedRegex::new(&anchored_pattern) {
        Ok(re) => {
            let found = re.find(text)?;
            if let Some(m) = found {
                Ok(m.start == 0 && m.end == text.len())
            } else {
                Ok(false)
            }
        }
        Err(_) => {
            // Fallback to unanchored find with exact bounds check
            let re = crate::regex::BoundedRegex::new(pattern)?;
            let found = re.find(text)?;
            if let Some(m) = found {
                Ok(m.start == 0 && m.end == text.len())
            } else {
                Ok(false)
            }
        }
    }
}

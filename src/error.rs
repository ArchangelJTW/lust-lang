use alloc::{format, string::String, string::ToString, vec::Vec};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use crate::ast::Span;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackFrame {
    pub function: String,
    pub line: usize,
    pub ip: usize,
}

impl StackFrame {
    pub fn new(function: impl Into<String>, line: usize, ip: usize) -> Self {
        Self {
            function: function.into(),
            line,
            ip,
        }
    }
}

#[derive(Error, Debug, Clone, PartialEq)]
pub enum LustError {
    #[error("Lexer error at line {line}, column {column}: {message}")]
    LexerError {
        line: usize,
        column: usize,
        message: String,
        module: Option<String>,
    },
    #[error("Parser error at line {line}, column {column}: {message}")]
    ParserError {
        line: usize,
        column: usize,
        message: String,
        module: Option<String>,
    },
    #[error("Type error: {message}")]
    TypeError { message: String },
    #[error("Type error at line {line}, column {column}: {message}")]
    TypeErrorWithSpan {
        message: String,
        line: usize,
        column: usize,
        module: Option<String>,
    },
    #[error("Compile error: {0}")]
    CompileError(String),
    #[error("Compile error at line {line}, column {column}: {message}")]
    CompileErrorWithSpan {
        message: String,
        line: usize,
        column: usize,
        module: Option<String>,
    },
    #[error("Runtime error: {message}")]
    RuntimeError { message: String },
    #[error(
        "Runtime error at line {line} in {function}: {message}\n{}",
        format_stack_trace(stack_trace)
    )]
    RuntimeErrorWithTrace {
        message: String,
        function: String,
        line: usize,
        stack_trace: Vec<StackFrame>,
    },
    #[error("Unknown error: {0}")]
    Unknown(String),
    #[error("BlockingHostCallViolation: {0}")]
    BlockingHostCallViolation(String),
}

fn format_stack_trace(frames: &[StackFrame]) -> String {
    if frames.is_empty() {
        return String::new();
    }

    let mut output = String::from("Stack trace:\n");
    for (i, frame) in frames.iter().enumerate() {
        output.push_str(&format!(
            "  [{}] {} (line {})\n",
            i, frame.function, frame.line
        ));
    }

    output
}

pub type Result<T> = core::result::Result<T, LustError>;

impl LustError {
    /// JSON representation suitable for CLI diagnostic consumers.
    pub fn diagnostic_json(&self) -> String {
        self.clone().into_diagnostic(None, None).to_json()
    }
}

/// Severity level for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
    Note,
    Help,
}

/// Compilation/execution phase where diagnostic originated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticPhase {
    Lex,
    Parse,
    Resolve,
    Typecheck,
    EffectCheck,
    BorrowEscape,
    MirVerify,
    Compile,
    Runtime,
}

/// Source span with 1-based line/column and 0-based byte offsets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticSpan {
    pub file: String,
    pub line_start: usize,
    pub col_start: usize,
    pub line_end: usize,
    pub col_end: usize,
    pub byte_start: usize,
    pub byte_end: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl DiagnosticSpan {
    pub fn from_span(span: Span, file: &str, source: Option<&str>) -> Self {
        let (byte_start, byte_end) = match source {
            Some(src) => (
                Self::calculate_byte_offset(src, span.start_line, span.start_col),
                Self::calculate_byte_offset(src, span.end_line.max(span.start_line), span.end_col.max(span.start_col)),
            ),
            None => (0, 0),
        };

        Self {
            file: file.to_string(),
            line_start: span.start_line,
            col_start: span.start_col,
            line_end: if span.end_line >= span.start_line { span.end_line } else { span.start_line },
            col_end: if span.end_line == span.start_line && span.end_col < span.start_col { span.start_col } else { span.end_col },
            byte_start,
            byte_end: byte_end.max(byte_start),
            label: None,
        }
    }

    pub fn calculate_byte_offset(source: &str, target_line: usize, target_col: usize) -> usize {
        if target_line == 0 || target_col == 0 {
            return 0;
        }

        let mut current_line = 1;
        let mut byte_idx = 0;

        for line_str in source.split_inclusive('\n') {
            if current_line == target_line {
                let mut col_count = 1;
                for ch in line_str.chars() {
                    if col_count == target_col {
                        return byte_idx;
                    }
                    if ch == '\n' || ch == '\r' {
                        break;
                    }
                    byte_idx += ch.len_utf8();
                    col_count += 1;
                }
                return byte_idx;
            }
            byte_idx += line_str.len();
            current_line += 1;
        }

        byte_idx.min(source.len())
    }
}

/// Secondary note attached to a diagnostic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticNote {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<DiagnosticSpan>,
}

/// Machine-applicable fix suggestion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticSuggestion {
    pub message: String,
    pub replacement: String,
    pub span: DiagnosticSpan,
}

/// Structured diagnostic matching RFC normative specification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub schema_version: String,
    pub code: String,
    pub severity: Severity,
    pub phase: DiagnosticPhase,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_span: Option<DiagnosticSpan>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub secondary_spans: Vec<DiagnosticSpan>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<DiagnosticNote>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<DiagnosticSuggestion>,
}

impl Diagnostic {
    pub fn new(
        code: impl Into<String>,
        severity: Severity,
        phase: DiagnosticPhase,
        message: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: "1.0".to_string(),
            code: code.into(),
            severity,
            phase,
            message: message.into(),
            primary_span: None,
            secondary_spans: Vec::new(),
            notes: Vec::new(),
            suggestions: Vec::new(),
        }
    }

    pub fn to_json(&self) -> String {
        #[cfg(feature = "std")]
        {
            serde_json::to_string(self).unwrap_or_else(|_| self.fallback_json())
        }
        #[cfg(not(feature = "std"))]
        {
            self.fallback_json()
        }
    }

    fn fallback_json(&self) -> String {
        let msg = self.message.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
        let code = self.code.replace('\\', "\\\\").replace('"', "\\\"");
        let sev = match self.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
            Severity::Help => "help",
        };
        let ph = match self.phase {
            DiagnosticPhase::Lex => "lex",
            DiagnosticPhase::Parse => "parse",
            DiagnosticPhase::Resolve => "resolve",
            DiagnosticPhase::Typecheck => "typecheck",
            DiagnosticPhase::EffectCheck => "effect_check",
            DiagnosticPhase::BorrowEscape => "borrow_escape",
            DiagnosticPhase::MirVerify => "mir_verify",
            DiagnosticPhase::Compile => "compile",
            DiagnosticPhase::Runtime => "runtime",
        };
        format!(
            "{{\"schema_version\":\"1.0\",\"code\":\"{}\",\"severity\":\"{}\",\"phase\":\"{}\",\"message\":\"{}\"}}",
            code, sev, ph, msg
        )
    }
}

/// Accumulator for diagnostics emitted across compilation phases.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticBag {
    pub diagnostics: Vec<Diagnostic>,
}

impl DiagnosticBag {
    pub fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
        }
    }

    pub fn add(&mut self, diag: Diagnostic) {
        self.diagnostics.push(diag);
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(|d| d.severity == Severity::Error)
    }

    pub fn to_json_line(&self) -> String {
        #[cfg(feature = "std")]
        {
            serde_json::to_string(self).unwrap_or_else(|_| self.fallback_json_line())
        }
        #[cfg(not(feature = "std"))]
        {
            self.fallback_json_line()
        }
    }

    fn fallback_json_line(&self) -> String {
        let items: Vec<String> = self.diagnostics.iter().map(|d| d.to_json()).collect();
        format!("{{\"diagnostics\":[{}]}}", items.join(","))
    }
}

/// Converts an error type into a canonical Diagnostic.
pub trait IntoDiagnostic {
    fn into_diagnostic(self, default_file: Option<&str>, source: Option<&str>) -> Diagnostic;
}

fn extract_diagnostic_code(message: &str, default: &str) -> String {
    for word in message.split(|c: char| !c.is_alphanumeric() && c != '-') {
        let candidate = word.strip_prefix("LUST-").unwrap_or(word);
        if candidate.len() == 5
            && candidate.starts_with('E')
            && candidate[1..].chars().all(|c| c.is_ascii_digit())
        {
            return candidate.to_string();
        }
    }
    default.to_string()
}

impl IntoDiagnostic for LustError {
    fn into_diagnostic(self, default_file: Option<&str>, source: Option<&str>) -> Diagnostic {
        let file = default_file.unwrap_or("<unknown>");
        match self {
            LustError::LexerError {
                line,
                column,
                message,
                module,
            } => {
                let f = module.as_deref().unwrap_or(file);
                let span = Span::new(line, column, line, column);
                let diag_span = DiagnosticSpan::from_span(span, f, source);
                let mut diag = Diagnostic::new("E0001", Severity::Error, DiagnosticPhase::Lex, message);
                diag.primary_span = Some(diag_span);
                diag
            }
            LustError::ParserError {
                line,
                column,
                message,
                module,
            } => {
                let f = module.as_deref().unwrap_or(file);
                let span = Span::new(line, column, line, column);
                let diag_span = DiagnosticSpan::from_span(span, f, source);
                let mut diag = Diagnostic::new("E1001", Severity::Error, DiagnosticPhase::Parse, message);
                diag.primary_span = Some(diag_span);
                diag
            }
            LustError::TypeError { message } => {
                let code = extract_diagnostic_code(&message, "E2001");
                Diagnostic::new(code, Severity::Error, DiagnosticPhase::Typecheck, message)
            }
            LustError::TypeErrorWithSpan {
                message,
                line,
                column,
                module,
            } => {
                let f = module.as_deref().unwrap_or(file);
                let span = Span::new(line, column, line, column);
                let diag_span = DiagnosticSpan::from_span(span, f, source);
                let code = extract_diagnostic_code(&message, "E2001");
                let mut diag = Diagnostic::new(code, Severity::Error, DiagnosticPhase::Typecheck, message);
                diag.primary_span = Some(diag_span);
                diag
            }
            LustError::CompileError(msg) => {
                let code = extract_diagnostic_code(&msg, "E4001");
                Diagnostic::new(code, Severity::Error, DiagnosticPhase::Compile, msg)
            }
            LustError::CompileErrorWithSpan {
                message,
                line,
                column,
                module,
            } => {
                let f = module.as_deref().unwrap_or(file);
                let span = Span::new(line, column, line, column);
                let diag_span = DiagnosticSpan::from_span(span, f, source);
                let code = extract_diagnostic_code(&message, "E4001");
                let mut diag = Diagnostic::new(code, Severity::Error, DiagnosticPhase::Compile, message);
                diag.primary_span = Some(diag_span);
                diag
            }
            LustError::RuntimeError { message } => {
                Diagnostic::new("E9001", Severity::Error, DiagnosticPhase::Runtime, message)
            }
            LustError::RuntimeErrorWithTrace {
                message,
                function: _,
                line,
                stack_trace,
            } => {
                let mut diag = Diagnostic::new("E9001", Severity::Error, DiagnosticPhase::Runtime, message);
                if line > 0 {
                    let span = Span::new(line, 1, line, 1);
                    diag.primary_span = Some(DiagnosticSpan::from_span(span, file, source));
                }
                for frame in stack_trace {
                    diag.notes.push(DiagnosticNote {
                        message: format!("called from {} at line {}", frame.function, frame.line),
                        span: if frame.line > 0 {
                            Some(DiagnosticSpan::from_span(
                                Span::new(frame.line, 1, frame.line, 1),
                                file,
                                source,
                            ))
                        } else {
                            None
                        },
                    });
                }
                diag
            }
            LustError::Unknown(msg) => {
                Diagnostic::new("E9001", Severity::Error, DiagnosticPhase::Runtime, msg)
            }
            LustError::BlockingHostCallViolation(msg) => {
                Diagnostic::new(
                    "E9001",
                    Severity::Error,
                    DiagnosticPhase::Runtime,
                    format!("BlockingHostCallViolation: {}", msg),
                )
            }
        }
    }
}

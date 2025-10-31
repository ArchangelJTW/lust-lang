use alloc::{format, string::String, vec::Vec};
use thiserror::Error;
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

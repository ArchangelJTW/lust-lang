//! JSON Lines (JSONL / LustJsonLines1) streaming reader, writer, and validation.
//!
//! # Memory Bound Invariant (RFC Section 7.6.2 & Roadmap Section 5.4 Gate-06):
//!
//! ```text
//! M_live <= M_fixed + max_buffered_bytes + max_in_flight * max_record_memory_bytes + max_error_bytes
//! ```
//!
//! Where:
//! - `M_fixed`: Fixed base memory footprint of the reader/writer struct and decoder cursors.
//! - `max_buffered_bytes`: Maximum memory allocated for internal chunk and line staging buffers.
//! - `max_in_flight`: Maximum number of decoded records concurrently retained (1 for pull reader).
//! - `max_record_memory_bytes`: Maximum memory permitted for an individual decoded `DataValue`.
//! - `max_error_bytes`: Maximum diagnostic memory accumulated in collect-errors mode.
//!
//! # Hostile Input & Fail-Closed Guarantee:
//! An oversized line with no LF must fail closed without retaining the whole input.
//! As soon as accumulated line length exceeds `max_line_bytes`, buffering immediately halts,
//! the stream fails closed with a typed limit error, and remaining stream bytes are never retained.

use crate::data::budget::{BudgetError, BudgetTracker};
use crate::data::json::events::JsonError;
use crate::data::json::scalar::JsonParser;
use crate::data::owner::OwnerId;
use crate::data::value::DataValue;
use crate::data::view::DataView;
use crate::schema::builder::Schema;
use crate::schema::error::ValidationError;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Fixed base memory overhead for `JsonlReader` internal structures (M_fixed).
pub const M_FIXED_READER_BYTES: usize = 1024;

/// Fixed base memory overhead for `JsonlWriter` internal structures.
pub const M_FIXED_WRITER_BYTES: usize = 512;

/// Default line size limit (1 MiB).
pub const DEFAULT_MAX_LINE_BYTES: usize = 1024 * 1024;

/// Default per-record decoded memory cap (1 MiB).
pub const DEFAULT_MAX_RECORD_BYTES: usize = 1024 * 1024;

/// Default upper bound for retained buffer capacity (64 KiB).
pub const DEFAULT_MAX_BUFFERED_BYTES: usize = 64 * 1024;

/// Default chunk size for pull I/O reads (8 KiB).
pub const DEFAULT_CHUNK_SIZE: usize = 8192;

/// Default maximum errors collected before terminating stream in collect mode.
pub const DEFAULT_MAX_ERRORS: usize = 100;

/// Default maximum error diagnostic bytes retained in collect mode (64 KiB).
pub const DEFAULT_MAX_ERROR_BYTES: usize = 64 * 1024;

/// Blank line policy per RFC 7.6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlankLines {
    /// Blank lines (empty or whitespace-only) are rejected as errors (JSON Lines standard default).
    #[default]
    Reject,
    /// Blank lines are skipped without emitting a record, preserving physical line numbers.
    Skip,
}

/// Control flow for scoped view callbacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlFlow {
    Continue,
    Break,
}

impl From<bool> for ControlFlow {
    fn from(b: bool) -> Self {
        if b {
            ControlFlow::Continue
        } else {
            ControlFlow::Break
        }
    }
}

/// Thread-safe cooperative cancellation handle.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

/// Configuration options for `JsonlReader`.
#[derive(Debug, Clone)]
pub struct JsonlReaderOptions {
    pub blank_lines: BlankLines,
    pub max_line_bytes: usize,
    pub max_record_memory_bytes: usize,
    pub max_buffered_bytes: usize,
    pub max_in_flight: usize,
    pub max_errors: usize,
    pub max_error_bytes: usize,
    pub max_depth: usize,
}

impl Default for JsonlReaderOptions {
    fn default() -> Self {
        Self {
            blank_lines: BlankLines::Reject,
            max_line_bytes: DEFAULT_MAX_LINE_BYTES,
            max_record_memory_bytes: DEFAULT_MAX_RECORD_BYTES,
            max_buffered_bytes: DEFAULT_MAX_BUFFERED_BYTES,
            max_in_flight: 1,
            max_errors: DEFAULT_MAX_ERRORS,
            max_error_bytes: DEFAULT_MAX_ERROR_BYTES,
            max_depth: 256,
        }
    }
}

impl JsonlReaderOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_blank_lines(mut self, blank_lines: BlankLines) -> Self {
        self.blank_lines = blank_lines;
        self
    }

    pub fn with_max_line_bytes(mut self, max: usize) -> Self {
        self.max_line_bytes = max;
        if self.max_buffered_bytes < max {
            self.max_buffered_bytes = max.saturating_add(DEFAULT_CHUNK_SIZE);
        }
        self
    }

    pub fn with_max_buffered_bytes(mut self, max: usize) -> Self {
        self.max_buffered_bytes = max;
        self
    }

    pub fn with_max_record_memory_bytes(mut self, max: usize) -> Self {
        self.max_record_memory_bytes = max;
        self
    }

    pub fn with_max_errors(mut self, max: usize) -> Self {
        self.max_errors = max;
        self
    }

    pub fn with_max_error_bytes(mut self, max: usize) -> Self {
        self.max_error_bytes = max;
        self
    }

    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }
}

/// Category of line-level errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineErrorKind {
    Parse,
    Limit,
    Io,
    Interrupted,
    BlankLine,
}

/// Detailed error describing failure on a specific line of a JSONL stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineError {
    /// 1-based physical line index.
    pub line: usize,
    /// 1-based physical UTF-8 byte column.
    pub col: usize,
    /// 0-based absolute stream byte offset.
    pub byte_offset: u64,
    /// Classification of the error.
    pub kind: LineErrorKind,
    /// Stable machine-readable error code.
    pub code: String,
    /// Diagnostic message.
    pub message: String,
}

impl LineError {
    pub fn parse(
        line: usize,
        col: usize,
        byte_offset: u64,
        code: impl Into<String>,
        msg: impl Into<String>,
    ) -> Self {
        Self {
            line,
            col,
            byte_offset,
            kind: LineErrorKind::Parse,
            code: code.into(),
            message: msg.into(),
        }
    }

    pub fn limit(
        line: usize,
        col: usize,
        byte_offset: u64,
        code: impl Into<String>,
        msg: impl Into<String>,
    ) -> Self {
        Self {
            line,
            col,
            byte_offset,
            kind: LineErrorKind::Limit,
            code: code.into(),
            message: msg.into(),
        }
    }

    pub fn io(line: usize, col: usize, byte_offset: u64, msg: impl Into<String>) -> Self {
        Self {
            line,
            col,
            byte_offset,
            kind: LineErrorKind::Io,
            code: "IO_ERROR".into(),
            message: msg.into(),
        }
    }

    pub fn interrupted(line: usize, col: usize, byte_offset: u64) -> Self {
        Self {
            line,
            col,
            byte_offset,
            kind: LineErrorKind::Interrupted,
            code: "INTERRUPTED".into(),
            message: "Stream reading cancelled or interrupted".into(),
        }
    }

    pub fn blank_line(line: usize, col: usize, byte_offset: u64) -> Self {
        Self {
            line,
            col,
            byte_offset,
            kind: LineErrorKind::BlankLine,
            code: "BLANK_LINE_REJECTED".into(),
            message: format!("Blank line at physical line {} rejected in strict mode", line),
        }
    }
}

impl fmt::Display for LineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] line {}:{} (offset {}): {}",
            self.code, self.line, self.col, self.byte_offset, self.message
        )
    }
}

impl std::error::Error for LineError {}

/// Combined error for validated JSONL streams (RFC 7.6.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamValidationError {
    Line(LineError),
    Schema {
        line: usize,
        col: usize,
        byte_offset: u64,
        issue: ValidationError,
    },
}

impl fmt::Display for StreamValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamValidationError::Line(err) => write!(f, "{}", err),
            StreamValidationError::Schema {
                line,
                col,
                byte_offset,
                issue,
            } => {
                write!(
                    f,
                    "[SCHEMA_VALIDATION_ERROR] line {}:{} (offset {}): {}",
                    line, col, byte_offset, issue
                )
            }
        }
    }
}

impl std::error::Error for StreamValidationError {}

impl From<LineError> for StreamValidationError {
    fn from(e: LineError) -> Self {
        StreamValidationError::Line(e)
    }
}

/// Category of writer errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteErrorKind {
    Io(std::io::ErrorKind),
    LimitExceeded,
    Encoding,
    Interrupted,
}

/// Error encountered during JSONL stream writing (RFC 7.6.1).
#[derive(Debug)]
pub struct StreamWriteError {
    /// 1-based record index currently being written.
    pub record_index: usize,
    /// Total confirmed bytes successfully written to the sink prior to failure.
    pub bytes_written: u64,
    /// Failure classification.
    pub kind: WriteErrorKind,
    /// Diagnostic description.
    pub message: String,
}

impl fmt::Display for StreamWriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[WRITE_ERROR] record {} (confirmed bytes written: {}): {}",
            self.record_index, self.bytes_written, self.message
        )
    }
}

impl std::error::Error for StreamWriteError {}

/// Configuration for `JsonlWriter`.
#[derive(Debug, Clone)]
pub struct JsonlWriterOptions {
    pub max_record_bytes: usize,
    pub max_buffered_bytes: usize,
}

impl Default for JsonlWriterOptions {
    fn default() -> Self {
        Self {
            max_record_bytes: DEFAULT_MAX_RECORD_BYTES,
            max_buffered_bytes: DEFAULT_MAX_BUFFERED_BYTES,
        }
    }
}

/// Pull-driven streaming reader for JSON Lines (LustJsonLines1 profile).
pub struct JsonlReader<R: Read> {
    reader: R,
    options: JsonlReaderOptions,
    chunk_buf: Vec<u8>,
    chunk_pos: usize,
    chunk_len: usize,
    line_buf: Vec<u8>,
    current_line: usize,
    current_stream_offset: u64,
    is_eof: bool,
    is_terminated: bool,
    cancellation: Option<CancellationToken>,
    budget: Option<BudgetTracker>,
    owner_id: OwnerId,
    error_count: usize,
    current_buffered_bytes: usize,
    current_in_flight_bytes: usize,
    current_error_bytes: usize,
}

impl<R: Read> JsonlReader<R> {
    pub fn new(reader: R) -> Self {
        Self::with_options(reader, JsonlReaderOptions::default())
    }

    pub fn with_options(reader: R, mut options: JsonlReaderOptions) -> Self {
        // Ensure max_buffered_bytes can at least accommodate max_line_bytes + 1 chunk
        if options.max_buffered_bytes < options.max_line_bytes {
            options.max_buffered_bytes = options.max_line_bytes.saturating_add(DEFAULT_CHUNK_SIZE);
        }

        let chunk_buf = vec![0u8; DEFAULT_CHUNK_SIZE];
        let line_buf = Vec::with_capacity(1024);
        let current_buffered_bytes = chunk_buf.capacity() + line_buf.capacity();

        Self {
            reader,
            options,
            chunk_buf,
            chunk_pos: 0,
            chunk_len: 0,
            line_buf,
            current_line: 1,
            current_stream_offset: 0,
            is_eof: false,
            is_terminated: false,
            cancellation: None,
            budget: None,
            owner_id: OwnerId::dummy(),
            error_count: 0,
            current_buffered_bytes,
            current_in_flight_bytes: 0,
            current_error_bytes: 0,
        }
    }

    pub fn with_cancellation(mut self, token: CancellationToken) -> Self {
        self.cancellation = Some(token);
        self
    }

    pub fn with_budget(mut self, budget: BudgetTracker) -> Self {
        self.budget = Some(budget);
        self
    }

    pub fn with_owner_id(mut self, owner: OwnerId) -> Self {
        self.owner_id = owner;
        self
    }

    pub fn cancel(&self) {
        if let Some(token) = &self.cancellation {
            token.cancel();
        }
    }

    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation
            .clone()
            .unwrap_or_else(CancellationToken::new)
    }

    pub fn options(&self) -> &JsonlReaderOptions {
        &self.options
    }

    pub fn budget(&self) -> Option<&BudgetTracker> {
        self.budget.as_ref()
    }

    pub fn budget_mut(&mut self) -> Option<&mut BudgetTracker> {
        self.budget.as_mut()
    }

    /// Current live memory tracked in bytes.
    ///
    /// M_live = M_fixed + current_buffered_bytes + current_in_flight_bytes + current_error_bytes
    pub fn live_memory_bytes(&self) -> usize {
        M_FIXED_READER_BYTES
            .saturating_add(self.current_buffered_bytes)
            .saturating_add(self.current_in_flight_bytes)
            .saturating_add(self.current_error_bytes)
    }

    /// Theoretical live memory upper bound for this reader.
    pub fn max_live_memory_cap(&self) -> usize {
        M_FIXED_READER_BYTES
            .saturating_add(self.options.max_buffered_bytes)
            .saturating_add(
                self.options
                    .max_in_flight
                    .saturating_mul(self.options.max_record_memory_bytes),
            )
            .saturating_add(self.options.max_error_bytes)
    }

    /// Retained buffer capacity across chunk and line buffers.
    pub fn retained_buffer_bytes(&self) -> usize {
        self.chunk_buf.capacity() + self.line_buf.capacity()
    }

    fn check_cancellation(&self, line: usize, offset: u64) -> Result<(), LineError> {
        if let Some(token) = &self.cancellation {
            if token.is_cancelled() {
                return Err(LineError::interrupted(line, 1, offset));
            }
        }
        Ok(())
    }

    fn update_buffered_bytes(&mut self, line: usize, offset: u64) -> Result<(), LineError> {
        self.current_buffered_bytes = self.chunk_buf.capacity() + self.line_buf.capacity();
        if self.current_buffered_bytes > self.options.max_buffered_bytes {
            self.is_terminated = true;
            return Err(LineError::limit(
                line,
                1,
                offset,
                "BUFFER_LIMIT_EXCEEDED",
                format!(
                    "Retained buffer size {} bytes exceeds max_buffered_bytes {}",
                    self.current_buffered_bytes, self.options.max_buffered_bytes
                ),
            ));
        }
        Ok(())
    }

    /// Read next physical line bounded by `max_line_bytes`.
    ///
    /// Fails closed without retaining the entire input if line exceeds `max_line_bytes`.
    fn read_physical_line(&mut self) -> Result<Option<(usize, u64)>, LineError> {
        if self.is_terminated {
            return Ok(None);
        }

        self.check_cancellation(self.current_line, self.current_stream_offset)?;

        if let Some(budget) = &mut self.budget {
            if let Err(e) = budget.charge_work(1) {
                self.is_terminated = true;
                return Err(self.budget_err_to_line_err(e, self.current_line, self.current_stream_offset));
            }
        }

        self.line_buf.clear();
        let line_num = self.current_line;
        let line_offset = self.current_stream_offset;
        let max_line = self.options.max_line_bytes;

        loop {
            if self.chunk_pos >= self.chunk_len {
                if self.is_eof {
                    break;
                }

                self.chunk_pos = 0;
                match self.reader.read(&mut self.chunk_buf) {
                    Ok(0) => {
                        self.is_eof = true;
                        self.chunk_len = 0;
                        break;
                    }
                    Ok(n) => {
                        self.chunk_len = n;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(e) => {
                        self.is_terminated = true;
                        return Err(LineError::io(line_num, 1, self.current_stream_offset, e.to_string()));
                    }
                }
            }

            // Search for LF in active chunk slice
            let slice = &self.chunk_buf[self.chunk_pos..self.chunk_len];
            if let Some(rel_nl) = slice.iter().position(|&b| b == b'\n') {
                let seg_len = rel_nl;
                let total_line_len = self.line_buf.len() + seg_len;
                if total_line_len > max_line {
                    // FAIL CLOSED: do not append to line_buf, do not retain remaining stream
                    self.is_terminated = true;
                    return Err(LineError::limit(
                        line_num,
                        1,
                        line_offset,
                        "LINE_TOO_LONG",
                        format!("Line length {} exceeds max_line_bytes {}", total_line_len, max_line),
                    ));
                }

                self.line_buf.extend_from_slice(&slice[..seg_len]);
                self.chunk_pos += seg_len + 1; // Consume LF
                self.current_stream_offset += (seg_len + 1) as u64;
                self.current_line += 1;

                if self.line_buf.ends_with(b"\r") {
                    self.line_buf.pop();
                }

                self.update_buffered_bytes(line_num, line_offset)?;
                return Ok(Some((line_num, line_offset)));
            } else {
                let seg_len = slice.len();
                let total_line_len = self.line_buf.len() + seg_len;
                if total_line_len > max_line {
                    // FAIL CLOSED on oversized line with no LF
                    self.is_terminated = true;
                    return Err(LineError::limit(
                        line_num,
                        1,
                        line_offset,
                        "LINE_TOO_LONG",
                        format!(
                            "Oversized line ({} bytes) with no LF exceeds max_line_bytes {}",
                            total_line_len, max_line
                        ),
                    ));
                }

                self.line_buf.extend_from_slice(slice);
                self.chunk_pos = self.chunk_len;
                self.current_stream_offset += seg_len as u64;
                self.update_buffered_bytes(line_num, line_offset)?;
            }
        }

        // EOF reached
        if !self.line_buf.is_empty() {
            // Non-empty last line without trailing LF is valid in LustJsonLines1
            self.current_line += 1;
            if self.line_buf.ends_with(b"\r") {
                self.line_buf.pop();
            }
            self.update_buffered_bytes(line_num, line_offset)?;
            Ok(Some((line_num, line_offset)))
        } else {
            self.is_terminated = true;
            Ok(None)
        }
    }

    fn budget_err_to_line_err(&self, err: BudgetError, line: usize, offset: u64) -> LineError {
        match err {
            BudgetError::BytesExceeded { limit, requested } => LineError::limit(
                line,
                1,
                offset,
                "MEMORY_BUDGET_EXCEEDED",
                format!("Memory budget exceeded: limit {}, requested {}", limit, requested),
            ),
            BudgetError::WorkUnitsExceeded { limit, current } => LineError::limit(
                line,
                1,
                offset,
                "WORK_UNITS_EXCEEDED",
                format!("Work units budget exceeded: limit {}, consumed {}", limit, current),
            ),
            BudgetError::DepthExceeded { limit, current } => LineError::limit(
                line,
                1,
                offset,
                "DEPTH_EXCEEDED",
                format!("Nesting depth exceeded: limit {}, current {}", limit, current),
            ),
            BudgetError::Interrupted => LineError::interrupted(line, 1, offset),
        }
    }

    /// Parse raw line bytes into a `DataValue`, enforcing RFC 8259 strict scalar profile.
    fn parse_line_value(
        &mut self,
        line_num: usize,
        line_offset: u64,
    ) -> Result<DataValue, LineError> {
        // RS (0x1E) forbidden per RFC 7.6
        if self.line_buf.starts_with(&[0x1E]) || self.line_buf.contains(&0x1E) {
            return Err(LineError::parse(
                line_num,
                1,
                line_offset,
                "RFC7464_RS_FORBIDDEN",
                "RFC 7464 Record Separator (0x1E) is strictly forbidden in JSONL",
            ));
        }

        // Validate UTF-8
        let line_str = match std::str::from_utf8(&self.line_buf) {
            Ok(s) => s,
            Err(_) => {
                return Err(LineError::parse(
                    line_num,
                    1,
                    line_offset,
                    "INVALID_UTF8",
                    "Invalid UTF-8 sequence in JSONL record",
                ));
            }
        };

        // Check for blank line
        if line_str.trim().is_empty() {
            return Err(LineError::blank_line(line_num, 1, line_offset));
        }

        let mut parser = JsonParser::with_limits(
            self.line_buf.as_slice(),
            self.options.max_depth,
            self.options.max_line_bytes,
        );

        match parser.parse_all() {
            Ok(val) => {
                let rec_bytes = estimate_data_value_bytes(&val);
                if rec_bytes > self.options.max_record_memory_bytes {
                    return Err(LineError::limit(
                        line_num,
                        1,
                        line_offset,
                        "RECORD_MEMORY_EXCEEDED",
                        format!(
                            "Decoded record memory {} bytes exceeds max_record_memory_bytes {}",
                            rec_bytes, self.options.max_record_memory_bytes
                        ),
                    ));
                }

                if let Some(budget) = &mut self.budget {
                    if let Err(e) = budget.charge_bytes(rec_bytes) {
                        return Err(self.budget_err_to_line_err(e, line_num, line_offset));
                    }
                }

                self.current_in_flight_bytes = rec_bytes;
                Ok(val)
            }
            Err(json_err) => {
                let (col, code) = match &json_err {
                    JsonError::DuplicateKey(_, pos) => (pos + 1, "DUPLICATE_KEY"),
                    JsonError::TrailingComma(pos) => (pos + 1, "TRAILING_COMMA"),
                    JsonError::TrailingBytes(pos) => (pos + 1, "TRAILING_BYTES"),
                    JsonError::UnexpectedChar(_, pos) => (pos + 1, "UNEXPECTED_CHAR"),
                    JsonError::InvalidEscape(pos) => (pos + 1, "INVALID_ESCAPE"),
                    JsonError::InvalidUnicode(pos) => (pos + 1, "INVALID_UNICODE"),
                    JsonError::InvalidNumber(pos) => (pos + 1, "INVALID_NUMBER"),
                    JsonError::MaxDepthExceeded(_) => (1, "MAX_DEPTH_EXCEEDED"),
                    JsonError::MaxBytesExceeded(_) => (1, "MAX_BYTES_EXCEEDED"),
                    JsonError::UnexpectedEof => (self.line_buf.len() + 1, "UNEXPECTED_EOF"),
                    JsonError::Io(_) => (1, "IO_ERROR"),
                };
                let kind = match &json_err {
                    JsonError::MaxDepthExceeded(_) | JsonError::MaxBytesExceeded(_) => LineErrorKind::Limit,
                    JsonError::Io(_) => LineErrorKind::Io,
                    _ => LineErrorKind::Parse,
                };
                Err(LineError {
                    line: line_num,
                    col,
                    byte_offset: line_offset + (col.saturating_sub(1)) as u64,
                    kind,
                    code: code.into(),
                    message: json_err.to_string(),
                })
            }
        }
    }

    /// Pull reader: fetch next owned `DataValue` in fail-fast mode.
    ///
    /// Stops and terminates on the first parse, limit, or I/O error.
    pub fn next_value(&mut self) -> Result<Option<DataValue>, LineError> {
        self.release_in_flight();

        loop {
            let (line_num, line_offset) = match self.read_physical_line() {
                Ok(Some(info)) => info,
                Ok(None) => return Ok(None),
                Err(e) => {
                    self.is_terminated = true;
                    return Err(e);
                }
            };

            match self.parse_line_value(line_num, line_offset) {
                Ok(val) => return Ok(Some(val)),
                Err(err) if err.kind == LineErrorKind::BlankLine && self.options.blank_lines == BlankLines::Skip => {
                    // Skip blank lines per configured option
                    continue;
                }
                Err(err) => {
                    self.is_terminated = true;
                    return Err(err);
                }
            }
        }
    }

    /// Pull reader: fetch next `Result<DataValue, LineError>` in collect-errors mode.
    ///
    /// Non-fatal line parse errors are yielded as `Ok(Some(Err(err)))`, allowing subsequent
    /// lines to be processed. Terminal errors (I/O, line length, memory limits, cancellation)
    /// immediately fail the stream as `Err(err)`.
    pub fn next_result(&mut self) -> Result<Option<Result<DataValue, LineError>>, LineError> {
        self.release_in_flight();

        loop {
            let (line_num, line_offset) = match self.read_physical_line() {
                Ok(Some(info)) => info,
                Ok(None) => return Ok(None),
                Err(e) => {
                    self.is_terminated = true;
                    return Err(e);
                }
            };

            match self.parse_line_value(line_num, line_offset) {
                Ok(val) => return Ok(Some(Ok(val))),
                Err(err) if err.kind == LineErrorKind::BlankLine && self.options.blank_lines == BlankLines::Skip => {
                    continue;
                }
                Err(err) if err.kind == LineErrorKind::Parse || err.kind == LineErrorKind::BlankLine => {
                    // Recoverable syntax / blank line error: record diagnostics and bound memory
                    self.error_count += 1;
                    let est_err_bytes = err.message.len() + 64;
                    self.current_error_bytes = self.current_error_bytes.saturating_add(est_err_bytes);

                    if self.error_count > self.options.max_errors
                        || self.current_error_bytes > self.options.max_error_bytes
                    {
                        self.is_terminated = true;
                        return Err(LineError::limit(
                            err.line,
                            err.col,
                            err.byte_offset,
                            "ERROR_LIMIT_EXCEEDED",
                            format!(
                                "Diagnostic error limit exceeded (count {}, bytes {})",
                                self.error_count, self.current_error_bytes
                            ),
                        ));
                    }

                    return Ok(Some(Err(err)));
                }
                Err(terminal_err) => {
                    self.is_terminated = true;
                    return Err(terminal_err);
                }
            }
        }
    }

    /// Pull reader: fetch next record validated against `schema` in fail-fast mode.
    pub fn next_validated(
        &mut self,
        schema: &Schema,
    ) -> Result<Option<DataValue>, StreamValidationError> {
        let val_opt = self.next_value()?;
        match val_opt {
            None => Ok(None),
            Some(val) => match schema.parse(&val) {
                Ok(parsed) => Ok(Some(parsed)),
                Err(validation_errors) => {
                    self.is_terminated = true;
                    let issue = validation_errors
                        .errors
                        .into_iter()
                        .next()
                        .unwrap_or_else(|| ValidationError::new(vec![], "Schema validation failed", "SCHEMA_INVALID"));
                    Err(StreamValidationError::Schema {
                        line: self.current_line.saturating_sub(1),
                        col: 1,
                        byte_offset: self.current_stream_offset,
                        issue,
                    })
                }
            },
        }
    }

    /// Pull reader: fetch next result validated against `schema` in collect mode.
    pub fn next_validated_result(
        &mut self,
        schema: &Schema,
    ) -> Result<Option<Result<DataValue, StreamValidationError>>, StreamValidationError> {
        let res_opt = self.next_result()?;
        match res_opt {
            None => Ok(None),
            Some(Ok(val)) => match schema.parse(&val) {
                Ok(parsed) => Ok(Some(Ok(parsed))),
                Err(validation_errors) => {
                    let issue = validation_errors
                        .errors
                        .into_iter()
                        .next()
                        .unwrap_or_else(|| ValidationError::new(vec![], "Schema validation failed", "SCHEMA_INVALID"));
                    Ok(Some(Err(StreamValidationError::Schema {
                        line: self.current_line.saturating_sub(1),
                        col: 1,
                        byte_offset: self.current_stream_offset,
                        issue,
                    })))
                }
            },
            Some(Err(line_err)) => Ok(Some(Err(StreamValidationError::Line(line_err)))),
        }
    }

    /// Zero-copy scoped view callback over each line (RFC 7.6.2).
    ///
    /// Invokes `callback` with a borrowed `DataView` over the current physical line.
    /// On early break or cooperative cancellation, the scoped view is dropped immediately,
    /// buffers are reset, and live memory remains bounded.
    pub fn for_each_view<F>(&mut self, mut callback: F) -> Result<usize, LineError>
    where
        F: FnMut(DataView<'_>) -> Result<ControlFlow, LineError>,
    {
        let mut records_seen = 0;

        while !self.is_terminated {
            self.check_cancellation(self.current_line, self.current_stream_offset)?;

            let (line_num, line_offset) = match self.read_physical_line()? {
                Some(info) => info,
                None => break,
            };

            let line_str = match std::str::from_utf8(&self.line_buf) {
                Ok(s) => s,
                Err(_) => {
                    self.is_terminated = true;
                    return Err(LineError::parse(
                        line_num,
                        1,
                        line_offset,
                        "INVALID_UTF8",
                        "Invalid UTF-8 sequence in JSONL record",
                    ));
                }
            };

            if line_str.trim().is_empty() {
                if self.options.blank_lines == BlankLines::Reject {
                    self.is_terminated = true;
                    return Err(LineError::blank_line(line_num, 1, line_offset));
                } else {
                    continue;
                }
            }

            records_seen += 1;
            let view = DataView::from_str(line_str, self.owner_id);
            let flow = callback(view)?;
            // Scoped view borrow ends here before buffer clear
            let _ = view;

            if flow == ControlFlow::Break {
                self.line_buf.clear();
                break;
            }

            self.check_cancellation(self.current_line, self.current_stream_offset)?;
            self.line_buf.clear();
        }

        self.line_buf.clear();
        Ok(records_seen)
    }

    fn release_in_flight(&mut self) {
        if self.current_in_flight_bytes > 0 {
            if let Some(budget) = &mut self.budget {
                budget.release_bytes(self.current_in_flight_bytes);
            }
            self.current_in_flight_bytes = 0;
        }
    }
}

impl<R: Read> Iterator for JsonlReader<R> {
    type Item = Result<DataValue, LineError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_value() {
            Ok(Some(v)) => Some(Ok(v)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// JSON Lines writer adhering to compact RFC 8259 formatting, LF termination,
/// bounded output buffers, backpressure, and exact short-write tracking.
pub struct JsonlWriter<W: Write> {
    sink: W,
    options: JsonlWriterOptions,
    record_index: usize,
    bytes_written: u64,
    cancellation: Option<CancellationToken>,
    budget: Option<BudgetTracker>,
}

impl<W: Write> JsonlWriter<W> {
    pub fn new(sink: W) -> Self {
        Self::with_options(sink, JsonlWriterOptions::default())
    }

    pub fn with_options(sink: W, options: JsonlWriterOptions) -> Self {
        Self {
            sink,
            options,
            record_index: 0,
            bytes_written: 0,
            cancellation: None,
            budget: None,
        }
    }

    pub fn with_cancellation(mut self, token: CancellationToken) -> Self {
        self.cancellation = Some(token);
        self
    }

    pub fn with_budget(mut self, budget: BudgetTracker) -> Self {
        self.budget = Some(budget);
        self
    }

    pub fn cancel(&self) {
        if let Some(token) = &self.cancellation {
            token.cancel();
        }
    }

    /// 1-based index of records written.
    pub fn record_index(&self) -> usize {
        self.record_index
    }

    /// Total confirmed bytes successfully written to the sink.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Write a slice to the sink, handling short writes and updating confirmed `bytes_written`.
    fn write_all_short_aware(&mut self, data: &[u8]) -> Result<(), StreamWriteError> {
        let mut written = 0;
        while written < data.len() {
            if let Some(token) = &self.cancellation {
                if token.is_cancelled() {
                    return Err(StreamWriteError {
                        record_index: self.record_index + 1,
                        bytes_written: self.bytes_written,
                        kind: WriteErrorKind::Interrupted,
                        message: "Writer cancelled during write".to_string(),
                    });
                }
            }

            match self.sink.write(&data[written..]) {
                Ok(0) => {
                    return Err(StreamWriteError {
                        record_index: self.record_index + 1,
                        bytes_written: self.bytes_written,
                        kind: WriteErrorKind::Io(std::io::ErrorKind::WriteZero),
                        message: "Sink accepted 0 bytes (short write zero)".to_string(),
                    });
                }
                Ok(n) => {
                    written += n;
                    self.bytes_written += n as u64;
                    if let Some(budget) = &mut self.budget {
                        let _ = budget.charge_bytes(n);
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    return Err(StreamWriteError {
                        record_index: self.record_index + 1,
                        bytes_written: self.bytes_written,
                        kind: WriteErrorKind::Io(e.kind()),
                        message: e.to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Write an individual `DataValue` as a compact JSON record followed by LF.
    pub fn write_value(&mut self, value: &DataValue) -> Result<usize, StreamWriteError> {
        if let Some(token) = &self.cancellation {
            if token.is_cancelled() {
                return Err(StreamWriteError {
                    record_index: self.record_index + 1,
                    bytes_written: self.bytes_written,
                    kind: WriteErrorKind::Interrupted,
                    message: "Writer cancelled before record emission".to_string(),
                });
            }
        }

        let json_str = crate::data::json::to_string(value).map_err(|e| StreamWriteError {
            record_index: self.record_index + 1,
            bytes_written: self.bytes_written,
            kind: WriteErrorKind::Encoding,
            message: format!("JSON encoding failed: {}", e),
        })?;

        let total_len = json_str.len() + 1;
        if total_len > self.options.max_record_bytes {
            return Err(StreamWriteError {
                record_index: self.record_index + 1,
                bytes_written: self.bytes_written,
                kind: WriteErrorKind::LimitExceeded,
                message: format!(
                    "Record byte size {} exceeds max_record_bytes {}",
                    total_len, self.options.max_record_bytes
                ),
            });
        }

        self.write_all_short_aware(json_str.as_bytes())?;
        self.write_all_short_aware(b"\n")?;
        self.record_index += 1;

        if let Some(budget) = &mut self.budget {
            let _ = budget.charge_work(1);
        }

        Ok(total_len)
    }

    /// Write a pre-formatted single-line JSON string followed by LF.
    pub fn write_raw_record(&mut self, record_str: &str) -> Result<usize, StreamWriteError> {
        if record_str.contains('\n') || record_str.contains('\r') {
            return Err(StreamWriteError {
                record_index: self.record_index + 1,
                bytes_written: self.bytes_written,
                kind: WriteErrorKind::Encoding,
                message: "Multiline JSON strings are forbidden in JSONL framing".to_string(),
            });
        }

        let total_len = record_str.len() + 1;
        if total_len > self.options.max_record_bytes {
            return Err(StreamWriteError {
                record_index: self.record_index + 1,
                bytes_written: self.bytes_written,
                kind: WriteErrorKind::LimitExceeded,
                message: format!(
                    "Raw record byte size {} exceeds max_record_bytes {}",
                    total_len, self.options.max_record_bytes
                ),
            });
        }

        self.write_all_short_aware(record_str.as_bytes())?;
        self.write_all_short_aware(b"\n")?;
        self.record_index += 1;

        Ok(total_len)
    }

    /// Flush the underlying sink.
    pub fn flush(&mut self) -> Result<(), StreamWriteError> {
        self.sink.flush().map_err(|e| StreamWriteError {
            record_index: self.record_index,
            bytes_written: self.bytes_written,
            kind: WriteErrorKind::Io(e.kind()),
            message: e.to_string(),
        })
    }
}

/// Helper function to construct a `JsonlReader` in default fail-fast mode.
pub fn lines<R: Read>(stream: R) -> JsonlReader<R> {
    JsonlReader::new(stream)
}

/// Helper function to construct a `JsonlReader` with customized options.
pub fn lines_with_options<R: Read>(stream: R, options: JsonlReaderOptions) -> JsonlReader<R> {
    JsonlReader::with_options(stream, options)
}

/// Iterator yielding collect-mode `Result<DataValue, LineError>` results.
pub struct JsonlResults<R: Read> {
    reader: JsonlReader<R>,
}

impl<R: Read> JsonlResults<R> {
    pub fn new(reader: JsonlReader<R>) -> Self {
        Self { reader }
    }
}

impl<R: Read> Iterator for JsonlResults<R> {
    type Item = Result<DataValue, LineError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.reader.next_result() {
            Ok(Some(res)) => Some(res),
            Ok(None) => None,
            Err(terminal_err) => Some(Err(terminal_err)),
        }
    }
}

/// Helper function to construct a collect-mode reader iterator.
pub fn line_results<R: Read>(stream: R, options: JsonlReaderOptions) -> JsonlResults<R> {
    JsonlResults::new(JsonlReader::with_options(stream, options))
}

/// Helper function to write an iterable of `DataValue`s to a sink.
pub fn write_lines<W: Write, I: IntoIterator<Item = DataValue>>(
    sink: W,
    values: I,
    options: JsonlWriterOptions,
) -> Result<u64, StreamWriteError> {
    let mut writer = JsonlWriter::with_options(sink, options);
    for val in values {
        writer.write_value(&val)?;
    }
    writer.flush()?;
    Ok(writer.bytes_written())
}

/// Helper function to create a schema-validated reader stream in fail-fast mode.
pub fn validate_lines<R: Read>(
    stream: R,
    schema: Schema,
    options: JsonlReaderOptions,
) -> JsonlValidated<R> {
    JsonlValidated {
        reader: JsonlReader::with_options(stream, options),
        schema,
    }
}

/// Helper function to create a schema-validated reader in collect mode.
pub fn validate_line_results<R: Read>(
    stream: R,
    schema: Schema,
    options: JsonlReaderOptions,
) -> JsonlValidatedResults<R> {
    JsonlValidatedResults {
        reader: JsonlReader::with_options(stream, options),
        schema,
    }
}

pub struct JsonlValidated<R: Read> {
    reader: JsonlReader<R>,
    schema: Schema,
}

impl<R: Read> Iterator for JsonlValidated<R> {
    type Item = Result<DataValue, StreamValidationError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.reader.next_validated(&self.schema) {
            Ok(Some(val)) => Some(Ok(val)),
            Ok(None) => None,
            Err(err) => Some(Err(err)),
        }
    }
}

pub struct JsonlValidatedResults<R: Read> {
    reader: JsonlReader<R>,
    schema: Schema,
}

impl<R: Read> Iterator for JsonlValidatedResults<R> {
    type Item = Result<Result<DataValue, StreamValidationError>, StreamValidationError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.reader.next_validated_result(&self.schema).transpose()
    }
}

/// Estimate heap allocation bytes for a `DataValue` to bound live memory.
pub fn estimate_data_value_bytes(val: &DataValue) -> usize {
    let mut bytes = std::mem::size_of::<DataValue>();
    match val {
        DataValue::String(s) => bytes += s.capacity(),
        DataValue::Binary(b) => bytes += b.capacity(),
        DataValue::Array(arr) => {
            bytes += arr.capacity() * std::mem::size_of::<DataValue>();
            for item in arr {
                bytes += estimate_data_value_bytes(item);
            }
        }
        DataValue::Document(doc) => {
            bytes += doc.entries.capacity() * std::mem::size_of::<(String, DataValue)>();
            for (k, v) in doc.iter() {
                bytes += k.capacity() + estimate_data_value_bytes(v);
            }
        }
        _ => {}
    }
    bytes
}

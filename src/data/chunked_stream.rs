//! Chunked streaming decoders over `std::io::Read` for JSON, YAML, BSON, TOON, and CSV.
//!
//! Enforces bounded sliding-window memory buffers, incremental chunk pulling,
//! document boundary synchronization across chunk edges, and strict failure on malformed records.

use crate::data::bson::{self, BsonError};
use crate::data::csv::{self, CsvError, CsvOptions};
use crate::data::document::Document;
use crate::data::json::JsonError;
use crate::data::toon::ToonError;
use crate::data::value::DataValue;
use crate::data::yaml::{LustSafeYaml12, YamlError};
use std::fmt;
use std::io::{self, Read};

pub const DEFAULT_CHUNK_SIZE: usize = 8_192; // 8 KB
pub const DEFAULT_MAX_STREAM_BUFFER: usize = 16_777_216; // 16 MB
pub const DEFAULT_MAX_STREAM_DOCUMENTS: usize = 1_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamDecoderOptions {
    pub chunk_size: usize,
    pub max_buffer_capacity: usize,
    pub max_documents: usize,
}

impl Default for StreamDecoderOptions {
    fn default() -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE,
            max_buffer_capacity: DEFAULT_MAX_STREAM_BUFFER,
            max_documents: DEFAULT_MAX_STREAM_DOCUMENTS,
        }
    }
}

impl StreamDecoderOptions {
    pub fn new(chunk_size: usize, max_buffer_capacity: usize) -> Self {
        Self {
            chunk_size,
            max_buffer_capacity,
            max_documents: DEFAULT_MAX_STREAM_DOCUMENTS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamError {
    UnexpectedEof,
    BufferCapacityExceeded(usize),
    DocumentLimitExceeded(usize),
    InvalidDocument(String),
    Io(String),
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamError::UnexpectedEof => write!(f, "Unexpected end-of-file during chunked streaming"),
            StreamError::BufferCapacityExceeded(s) => {
                write!(f, "Stream buffer capacity {} exceeded maximum limit", s)
            }
            StreamError::DocumentLimitExceeded(d) => {
                write!(f, "Stream document count {} exceeded maximum limit", d)
            }
            StreamError::InvalidDocument(e) => write!(f, "Invalid document in stream: {}", e),
            StreamError::Io(e) => write!(f, "I/O error during chunked stream decode: {}", e),
        }
    }
}

impl std::error::Error for StreamError {}

impl From<io::Error> for StreamError {
    fn from(err: io::Error) -> Self {
        StreamError::Io(err.to_string())
    }
}

impl From<BsonError> for StreamError {
    fn from(err: BsonError) -> Self {
        StreamError::InvalidDocument(err.to_string())
    }
}

impl From<YamlError> for StreamError {
    fn from(err: YamlError) -> Self {
        StreamError::InvalidDocument(err.to_string())
    }
}

impl From<ToonError> for StreamError {
    fn from(err: ToonError) -> Self {
        StreamError::InvalidDocument(err.to_string())
    }
}

impl From<CsvError> for StreamError {
    fn from(err: CsvError) -> Self {
        StreamError::InvalidDocument(err.to_string())
    }
}

impl From<JsonError> for StreamError {
    fn from(err: JsonError) -> Self {
        StreamError::InvalidDocument(err.to_string())
    }
}

// ============================================================================
// 1. BSON Chunked Stream Decoder
// ============================================================================

/// Chunked stream reader for concatenated BSON documents.
/// Reads the 4-byte LE length header, buffers exactly that document,
/// and yields deserialized `Document`s.
pub struct BsonStreamDecoder<R> {
    reader: R,
    options: StreamDecoderOptions,
    doc_count: usize,
    eof_reached: bool,
}

impl<R: Read> BsonStreamDecoder<R> {
    pub fn new(reader: R, options: StreamDecoderOptions) -> Self {
        Self {
            reader,
            options,
            doc_count: 0,
            eof_reached: false,
        }
    }

    /// Pull the next BSON document from the stream.
    pub fn next_document(&mut self) -> Result<Option<Document>, StreamError> {
        if self.eof_reached {
            return Ok(None);
        }

        // Read 4-byte length prefix
        let mut len_bytes = [0u8; 4];
        let mut bytes_read = 0;
        while bytes_read < 4 {
            let n = self.reader.read(&mut len_bytes[bytes_read..])?;
            if n == 0 {
                if bytes_read == 0 {
                    self.eof_reached = true;
                    return Ok(None);
                } else {
                    return Err(StreamError::UnexpectedEof);
                }
            }
            bytes_read += n;
        }

        let doc_len = i32::from_le_bytes(len_bytes) as usize;
        if doc_len < 5 {
            return Err(StreamError::InvalidDocument(format!(
                "BSON document length {} is below minimum 5 bytes",
                doc_len
            )));
        }
        if doc_len > self.options.max_buffer_capacity {
            return Err(StreamError::BufferCapacityExceeded(doc_len));
        }

        let mut doc_buf = vec![0u8; doc_len];
        doc_buf[0..4].copy_from_slice(&len_bytes);

        let mut body_read = 4;
        while body_read < doc_len {
            let n = self.reader.read(&mut doc_buf[body_read..])?;
            if n == 0 {
                return Err(StreamError::UnexpectedEof);
            }
            body_read += n;
        }

        let doc = bson::from_slice(&doc_buf)?;
        self.doc_count += 1;
        if self.doc_count > self.options.max_documents {
            return Err(StreamError::DocumentLimitExceeded(self.doc_count));
        }

        Ok(Some(doc))
    }
}

// ============================================================================
// 2. YAML Chunked Stream Decoder
// ============================================================================

/// Chunked streaming decoder for multi-document YAML streams (separated by `---`).
pub struct YamlStreamDecoder<R> {
    reader: R,
    options: StreamDecoderOptions,
    buffer: String,
    doc_count: usize,
    eof_reached: bool,
}

impl<R: Read> YamlStreamDecoder<R> {
    pub fn new(reader: R, options: StreamDecoderOptions) -> Self {
        Self {
            reader,
            options,
            buffer: String::new(),
            doc_count: 0,
            eof_reached: false,
        }
    }

    pub fn next_document(&mut self) -> Result<Option<Document>, StreamError> {
        loop {
            // Check if buffer contains a document separator "---"
            if let Some(pos) = find_yaml_doc_boundary(&self.buffer) {
                let doc_str = self.buffer[..pos].trim().to_string();
                let remainder = self.buffer[pos..].trim_start_matches(|c| c == '-' || c == '\n' || c == '\r').to_string();
                self.buffer = remainder;

                if !doc_str.is_empty() {
                    let mut yaml_parser = LustSafeYaml12::new();
                    let val = yaml_parser.parse(&doc_str)?;
                    let doc = match val {
                        DataValue::Document(d) => d,
                        other => {
                            return Err(StreamError::InvalidDocument(format!(
                                "Expected Document in YAML stream, found {:?}",
                                other
                            )))
                        }
                    };
                    self.doc_count += 1;
                    if self.doc_count > self.options.max_documents {
                        return Err(StreamError::DocumentLimitExceeded(self.doc_count));
                    }
                    return Ok(Some(doc));
                }
            }

            if self.eof_reached {
                let remaining = self.buffer.trim();
                if remaining.is_empty() {
                    return Ok(None);
                }
                let doc_str = self.buffer.clone();
                self.buffer.clear();
                let mut yaml_parser = LustSafeYaml12::new();
                let val = yaml_parser.parse(&doc_str)?;
                let doc = match val {
                    DataValue::Document(d) => d,
                    other => {
                        return Err(StreamError::InvalidDocument(format!(
                            "Expected Document in YAML stream, found {:?}",
                            other
                        )))
                    }
                };
                self.doc_count += 1;
                return Ok(Some(doc));
            }

            let remaining_cap = self.options.max_buffer_capacity.saturating_sub(self.buffer.len());
            if remaining_cap == 0 {
                return Err(StreamError::BufferCapacityExceeded(
                    self.buffer.len().saturating_add(1),
                ));
            }
            let read_size = self.options.chunk_size.min(remaining_cap);
            let mut chunk_buf = vec![0u8; read_size];
            let n = self.reader.read(&mut chunk_buf)?;
            if n == 0 {
                self.eof_reached = true;
                continue;
            }

            if self.buffer.len().saturating_add(n) > self.options.max_buffer_capacity {
                return Err(StreamError::BufferCapacityExceeded(self.buffer.len().saturating_add(n)));
            }

            let chunk_str = std::str::from_utf8(&chunk_buf[..n])
                .map_err(|e| StreamError::InvalidDocument(format!("Invalid UTF-8 in YAML stream: {}", e)))?;
            self.buffer.push_str(chunk_str);
        }
    }
}

fn find_yaml_doc_boundary(s: &str) -> Option<usize> {
    // Look for "\n---" or start with "---"
    if s.starts_with("---\n") || s.starts_with("---\r\n") {
        return Some(3);
    }
    s.find("\n---").map(|idx| idx + 1)
}

// ============================================================================
// 3. JSON Chunked Stream Decoder
// ============================================================================

/// Chunked streaming decoder for concatenated or sequence JSON objects/values.
pub struct JsonStreamDecoder<R> {
    reader: R,
    options: StreamDecoderOptions,
    buffer: Vec<u8>,
    doc_count: usize,
    eof_reached: bool,
}

impl<R: Read> JsonStreamDecoder<R> {
    pub fn new(reader: R, options: StreamDecoderOptions) -> Self {
        Self {
            reader,
            options,
            buffer: Vec::new(),
            doc_count: 0,
            eof_reached: false,
        }
    }

    pub fn next_value(&mut self) -> Result<Option<DataValue>, StreamError> {
        loop {
            // Trim leading whitespace
            let start_idx = self.buffer.iter().position(|&b| !b.is_ascii_whitespace());
            if let Some(start) = start_idx {
                if start > 0 {
                    self.buffer.drain(..start);
                }
            } else {
                self.buffer.clear();
            }

            if !self.buffer.is_empty() {
                // Try scanning for a complete balanced JSON value
                if let Some(end_idx) = find_complete_json_value(&self.buffer) {
                    let json_slice = &self.buffer[..end_idx];
                    let val: serde_json::Value = serde_json::from_slice(json_slice)
                        .map_err(|e| StreamError::InvalidDocument(e.to_string()))?;
                    let data_val = json_value_to_data_value(val);

                    self.buffer.drain(..end_idx);
                    self.doc_count += 1;
                    if self.doc_count > self.options.max_documents {
                        return Err(StreamError::DocumentLimitExceeded(self.doc_count));
                    }
                    return Ok(Some(data_val));
                }
            }

            if self.eof_reached {
                if self.buffer.is_empty() {
                    return Ok(None);
                }
                // Try parsing whatever remains
                let val: serde_json::Value = serde_json::from_slice(&self.buffer)
                    .map_err(|e| StreamError::InvalidDocument(e.to_string()))?;
                let data_val = json_value_to_data_value(val);
                self.buffer.clear();
                self.doc_count += 1;
                return Ok(Some(data_val));
            }

            let remaining_cap = self.options.max_buffer_capacity.saturating_sub(self.buffer.len());
            if remaining_cap == 0 {
                return Err(StreamError::BufferCapacityExceeded(
                    self.buffer.len().saturating_add(1),
                ));
            }
            let read_size = self.options.chunk_size.min(remaining_cap);
            let mut chunk_buf = vec![0u8; read_size];
            let n = self.reader.read(&mut chunk_buf)?;
            if n == 0 {
                self.eof_reached = true;
                continue;
            }

            if self.buffer.len().saturating_add(n) > self.options.max_buffer_capacity {
                return Err(StreamError::BufferCapacityExceeded(self.buffer.len().saturating_add(n)));
            }

            self.buffer.extend_from_slice(&chunk_buf[..n]);
        }
    }

    pub fn next_document(&mut self) -> Result<Option<Document>, StreamError> {
        match self.next_value()? {
            Some(DataValue::Document(doc)) => Ok(Some(doc)),
            Some(other) => Err(StreamError::InvalidDocument(format!(
                "Expected Document in stream, got {:?}",
                other
            ))),
            None => Ok(None),
        }
    }
}

fn find_complete_json_value(buf: &[u8]) -> Option<usize> {
    if buf.is_empty() {
        return None;
    }
    let first = buf[0];
    if first == b'{' || first == b'[' {
        let mut depth = 0usize;
        let mut in_str = false;
        let mut escape = false;

        for (i, &b) in buf.iter().enumerate() {
            if in_str {
                if escape {
                    escape = false;
                } else if b == b'\\' {
                    escape = true;
                } else if b == b'"' {
                    in_str = false;
                }
            } else if b == b'"' {
                in_str = true;
            } else if b == b'{' || b == b'[' {
                depth += 1;
            } else if b == b'}' || b == b']' {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
        }
        None
    } else if first == b'"' {
        let mut escape = false;
        for (i, &b) in buf.iter().enumerate().skip(1) {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                return Some(i + 1);
            }
        }
        None
    } else {
        // Primitive (number, boolean, null): scan until whitespace, comma, or delimiter
        let end = buf.iter().position(|&b| b.is_ascii_whitespace() || b == b',');
        end
    }
}

fn json_value_to_data_value(v: serde_json::Value) -> DataValue {
    match v {
        serde_json::Value::Null => DataValue::Null,
        serde_json::Value::Bool(b) => DataValue::Bool(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                DataValue::Number(crate::data::ExactNumber::Int(i))
            } else if let Some(u) = n.as_u64() {
                DataValue::Number(crate::data::ExactNumber::UInt(u))
            } else if let Some(f) = n.as_f64() {
                DataValue::Number(crate::data::ExactNumber::Float(f))
            } else {
                DataValue::String(n.to_string())
            }
        }
        serde_json::Value::String(s) => DataValue::String(s),
        serde_json::Value::Array(arr) => {
            DataValue::Array(arr.into_iter().map(json_value_to_data_value).collect())
        }
        serde_json::Value::Object(map) => {
            let mut doc = Document::new();
            for (k, val) in map {
                let _ = doc.try_insert(k, json_value_to_data_value(val));
            }
            DataValue::Document(doc)
        }
    }
}

// ============================================================================
// 4. TOON Chunked Stream Decoder
// ============================================================================

/// Chunked streaming decoder for TOON table / stream rows.
pub struct ToonStreamDecoder<R> {
    reader: R,
    options: StreamDecoderOptions,
    headers: Option<Vec<String>>,
    buffer: String,
    eof_reached: bool,
    doc_count: usize,
}

impl<R: Read> ToonStreamDecoder<R> {
    pub fn new(reader: R, options: StreamDecoderOptions) -> Self {
        Self {
            reader,
            options,
            headers: None,
            buffer: String::new(),
            eof_reached: false,
            doc_count: 0,
        }
    }

    pub fn next_document(&mut self) -> Result<Option<Document>, StreamError> {
        loop {
            // Find newline in buffer
            if let Some(pos) = self.buffer.find('\n') {
                let line = self.buffer[..pos].trim().to_string();
                self.buffer.drain(..pos + 1);

                if line.is_empty() {
                    continue;
                }

                if self.headers.is_none() {
                    // First line is headers
                    let cols: Vec<String> = line.split('|').map(|s| s.trim().to_string()).collect();
                    if cols.is_empty() {
                        return Err(StreamError::InvalidDocument("Missing TOON headers".to_string()));
                    }
                    self.headers = Some(cols);
                    continue;
                }

                let headers = self.headers.as_ref().unwrap();
                let cells: Vec<&str> = line.split('|').map(|s| s.trim()).collect();
                if cells.len() != headers.len() {
                    return Err(StreamError::InvalidDocument(format!(
                        "TOON column count mismatch: expected {}, found {}",
                        headers.len(),
                        cells.len()
                    )));
                }

                let mut doc = Document::new();
                for (i, cell) in cells.iter().enumerate() {
                    let val = parse_toon_scalar_cell(cell);
                    doc.try_insert(headers[i].clone(), val)
                        .map_err(|e| StreamError::InvalidDocument(e.to_string()))?;
                }

                self.doc_count += 1;
                if self.doc_count > self.options.max_documents {
                    return Err(StreamError::DocumentLimitExceeded(self.doc_count));
                }
                return Ok(Some(doc));
            }

            if self.eof_reached {
                if self.buffer.trim().is_empty() {
                    return Ok(None);
                }
                let line = self.buffer.trim().to_string();
                self.buffer.clear();
                if let Some(headers) = &self.headers {
                    let cells: Vec<&str> = line.split('|').map(|s| s.trim()).collect();
                    if cells.len() != headers.len() {
                        return Err(StreamError::InvalidDocument("TOON column mismatch at EOF".to_string()));
                    }
                    let mut doc = Document::new();
                    for (i, cell) in cells.iter().enumerate() {
                        let val = parse_toon_scalar_cell(cell);
                        doc.try_insert(headers[i].clone(), val)
                            .map_err(|e| StreamError::InvalidDocument(e.to_string()))?;
                    }
                    self.doc_count += 1;
                    return Ok(Some(doc));
                } else {
                    return Ok(None);
                }
            }

            let remaining_cap = self.options.max_buffer_capacity.saturating_sub(self.buffer.len());
            if remaining_cap == 0 {
                return Err(StreamError::BufferCapacityExceeded(
                    self.buffer.len().saturating_add(1),
                ));
            }
            let read_size = self.options.chunk_size.min(remaining_cap);
            let mut chunk_buf = vec![0u8; read_size];
            let n = self.reader.read(&mut chunk_buf)?;
            if n == 0 {
                self.eof_reached = true;
                continue;
            }

            if self.buffer.len().saturating_add(n) > self.options.max_buffer_capacity {
                return Err(StreamError::BufferCapacityExceeded(self.buffer.len().saturating_add(n)));
            }

            let s = std::str::from_utf8(&chunk_buf[..n])
                .map_err(|e| StreamError::InvalidDocument(format!("Invalid UTF-8 in TOON stream: {}", e)))?;
            self.buffer.push_str(s);
        }
    }
}

fn parse_toon_scalar_cell(s: &str) -> DataValue {
    if s == "true" {
        DataValue::Bool(true)
    } else if s == "false" {
        DataValue::Bool(false)
    } else if s == "null" || s.is_empty() {
        DataValue::Null
    } else if let Ok(i) = s.parse::<i64>() {
        DataValue::Number(crate::data::ExactNumber::Int(i))
    } else if let Ok(f) = s.parse::<f64>() {
        if f.is_finite() && s.contains('.') {
            DataValue::Number(crate::data::ExactNumber::Float(f))
        } else {
            DataValue::String(s.to_string())
        }
    } else {
        DataValue::String(s.to_string())
    }
}

// ============================================================================
// 5. CSV Chunked Stream Decoder
// ============================================================================

/// Chunked streaming decoder for CSV / TSV records into `Document`s.
pub struct CsvStreamDecoder<R> {
    parser: csv::CsvParser<R>,
    headers: Option<Vec<String>>,
    doc_count: usize,
    max_documents: usize,
}

impl<R: Read> CsvStreamDecoder<R> {
    pub fn new(reader: R, options: CsvOptions) -> Result<Self, StreamError> {
        let max_documents = options.max_rows;
        let parser = csv::CsvParser::new(reader, options);
        // Header is read in first record if has_headers is enabled
        Ok(Self {
            parser,
            headers: None,
            doc_count: 0,
            max_documents,
        })
    }

    pub fn next_document(&mut self) -> Result<Option<Document>, StreamError> {
        if self.headers.is_none() {
            let first_row = match self.parser.next_record()? {
                Some(r) => r,
                None => return Ok(None),
            };
            let hdrs: Vec<String> = first_row
                .into_iter()
                .map(|v| match v {
                    DataValue::String(s) => s,
                    other => other.to_string(),
                })
                .collect();
            // Gate-SCHEMA-01 check
            for (i, h) in hdrs.iter().enumerate() {
                for other in &hdrs[i + 1..] {
                    if h == other {
                        return Err(StreamError::InvalidDocument(format!("Duplicate CSV header '{}'", h)));
                    }
                }
            }
            self.headers = Some(hdrs);
        }

        let headers = self.headers.as_ref().unwrap();
        let record = match self.parser.next_record()? {
            Some(r) => r,
            None => return Ok(None),
        };

        if record.len() != headers.len() {
            return Err(StreamError::InvalidDocument(format!(
                "CSV column mismatch: expected {}, found {}",
                headers.len(),
                record.len()
            )));
        }

        let mut doc = Document::new();
        for (i, val) in record.into_iter().enumerate() {
            doc.try_insert(headers[i].clone(), val)
                .map_err(|e| StreamError::InvalidDocument(e.to_string()))?;
        }

        self.doc_count += 1;
        if self.doc_count > self.max_documents {
            return Err(StreamError::DocumentLimitExceeded(self.doc_count));
        }

        Ok(Some(doc))
    }
}

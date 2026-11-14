//! High-performance CSV and TSV codec with configurable delimiters, quotes, headers, and strict budgets.
//!
//! Fully compliant with RFC 4180, supporting configurable delimiters (e.g. comma, tab, semicolon),
//! configurable quote characters, multiline quoted fields, strict column consistency,
//! duplicate header rejection (Gate-SCHEMA-01), scalar type inference, and streaming record iterators.

use crate::data::{DataValue, Document, ExactNumber};
use std::fmt;
use std::io::{self, Read, Write};

pub const DEFAULT_MAX_ROWS: usize = 1_000_000;
pub const DEFAULT_MAX_COLUMNS: usize = 1_024;
pub const DEFAULT_MAX_FIELD_BYTES: usize = 16_777_216; // 16 MB
pub const DEFAULT_MAX_TOTAL_BYTES: usize = 67_108_864; // 64 MB

/// Configuration options for CSV and TSV parsing and serialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsvOptions {
    pub delimiter: u8,
    pub quote: u8,
    pub has_headers: bool,
    pub trim: bool,
    pub flexible: bool,
    pub parse_scalars: bool,
    pub max_rows: usize,
    pub max_columns: usize,
    pub max_field_bytes: usize,
    pub max_total_bytes: usize,
}

impl Default for CsvOptions {
    fn default() -> Self {
        Self {
            delimiter: b',',
            quote: b'"',
            has_headers: true,
            trim: false,
            flexible: false,
            parse_scalars: true,
            max_rows: DEFAULT_MAX_ROWS,
            max_columns: DEFAULT_MAX_COLUMNS,
            max_field_bytes: DEFAULT_MAX_FIELD_BYTES,
            max_total_bytes: DEFAULT_MAX_TOTAL_BYTES,
        }
    }
}

impl CsvOptions {
    /// Standard comma-separated values options.
    pub fn csv() -> Self {
        Self::default()
    }

    /// Tab-separated values options.
    pub fn tsv() -> Self {
        Self {
            delimiter: b'\t',
            ..Self::default()
        }
    }

    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = delimiter;
        self
    }

    pub fn with_quote(mut self, quote: u8) -> Self {
        self.quote = quote;
        self
    }

    pub fn with_headers(mut self, has_headers: bool) -> Self {
        self.has_headers = has_headers;
        self
    }

    pub fn with_trim(mut self, trim: bool) -> Self {
        self.trim = trim;
        self
    }

    pub fn with_flexible(mut self, flexible: bool) -> Self {
        self.flexible = flexible;
        self
    }

    pub fn with_parse_scalars(mut self, parse_scalars: bool) -> Self {
        self.parse_scalars = parse_scalars;
        self
    }

    pub fn with_max_rows(mut self, max_rows: usize) -> Self {
        self.max_rows = max_rows;
        self
    }

    pub fn with_max_columns(mut self, max_columns: usize) -> Self {
        self.max_columns = max_columns;
        self
    }

    pub fn with_max_field_bytes(mut self, max_field_bytes: usize) -> Self {
        self.max_field_bytes = max_field_bytes;
        self
    }

    pub fn with_max_total_bytes(mut self, max_total_bytes: usize) -> Self {
        self.max_total_bytes = max_total_bytes;
        self
    }
}

/// Errors occurring during CSV or TSV parsing/serialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CsvError {
    EmptyInput,
    MissingHeader,
    DuplicateHeader(String),
    MaxColumnsExceeded(usize),
    MaxRowsExceeded(usize),
    MaxFieldBytesExceeded(usize),
    MaxTotalBytesExceeded(usize),
    ColumnMismatch { expected: usize, found: usize, row: usize },
    UnterminatedQuote { line: usize, col: usize },
    InvalidQuotePlacement { line: usize, col: usize },
    InvalidUtf8(String),
    IoError(String),
}

impl fmt::Display for CsvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CsvError::EmptyInput => write!(f, "CSV input is empty"),
            CsvError::MissingHeader => write!(f, "Missing CSV header row"),
            CsvError::DuplicateHeader(h) => write!(f, "Duplicate column header '{}' in CSV (Gate-SCHEMA-01)", h),
            CsvError::MaxColumnsExceeded(c) => write!(f, "Column count {} exceeds maximum {}", c, DEFAULT_MAX_COLUMNS),
            CsvError::MaxRowsExceeded(r) => write!(f, "Row count {} exceeds maximum {}", r, DEFAULT_MAX_ROWS),
            CsvError::MaxFieldBytesExceeded(b) => write!(f, "Field byte length {} exceeds maximum {}", b, DEFAULT_MAX_FIELD_BYTES),
            CsvError::MaxTotalBytesExceeded(b) => write!(f, "Total byte length {} exceeds maximum {}", b, DEFAULT_MAX_TOTAL_BYTES),
            CsvError::ColumnMismatch { expected, found, row } => {
                write!(f, "Row {} has {} columns, expected {}", row, found, expected)
            }
            CsvError::UnterminatedQuote { line, col } => {
                write!(f, "Unterminated quoted field at line {}, col {}", line, col)
            }
            CsvError::InvalidQuotePlacement { line, col } => {
                write!(f, "Invalid quote placement at line {}, col {}", line, col)
            }
            CsvError::InvalidUtf8(e) => write!(f, "Invalid UTF-8 in CSV input: {}", e),
            CsvError::IoError(e) => write!(f, "I/O error during CSV processing: {}", e),
        }
    }
}

impl std::error::Error for CsvError {}

impl From<io::Error> for CsvError {
    fn from(err: io::Error) -> Self {
        CsvError::IoError(err.to_string())
    }
}

/// Tabular representation of CSV data with optional headers and rows of DataValues.
#[derive(Debug, Clone, PartialEq)]
pub struct CsvTable {
    headers: Option<Vec<String>>,
    records: Vec<Vec<DataValue>>,
}

impl CsvTable {
    pub fn new(headers: Option<Vec<String>>, records: Vec<Vec<DataValue>>) -> Result<Self, CsvError> {
        if let Some(hdrs) = &headers {
            if hdrs.is_empty() {
                return Err(CsvError::MissingHeader);
            }
            // Gate-SCHEMA-01: verify unique headers
            for (i, h) in hdrs.iter().enumerate() {
                for other in &hdrs[i + 1..] {
                    if h == other {
                        return Err(CsvError::DuplicateHeader(h.clone()));
                    }
                }
            }
        }
        Ok(Self { headers, records })
    }

    pub fn headers(&self) -> Option<&[String]> {
        self.headers.as_deref()
    }

    pub fn records(&self) -> &[Vec<DataValue>] {
        &self.records
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Converts table rows into `Document`s using headers as keys.
    pub fn to_documents(&self) -> Result<Vec<Document>, CsvError> {
        let headers = match &self.headers {
            Some(h) => h,
            None => return Err(CsvError::MissingHeader),
        };

        let mut docs = Vec::with_capacity(self.records.len());
        for (row_idx, row) in self.records.iter().enumerate() {
            if row.len() != headers.len() {
                return Err(CsvError::ColumnMismatch {
                    expected: headers.len(),
                    found: row.len(),
                    row: row_idx + 1,
                });
            }
            let mut doc = Document::new();
            for (col_idx, val) in row.iter().enumerate() {
                doc.try_insert(headers[col_idx].clone(), val.clone())
                    .map_err(|_| CsvError::DuplicateHeader(headers[col_idx].clone()))?;
            }
            docs.push(doc);
        }
        Ok(docs)
    }

    /// Constructs a `CsvTable` from a slice of `Document`s.
    pub fn from_documents(docs: &[Document], options: &CsvOptions) -> Result<Self, CsvError> {
        if docs.is_empty() {
            return Ok(Self {
                headers: if options.has_headers { Some(Vec::new()) } else { None },
                records: Vec::new(),
            });
        }

        // Collect ordered headers from the first document or union of all documents
        let mut headers = Vec::new();
        for key in docs[0].keys() {
            headers.push(key.clone());
        }
        for doc in docs.iter().skip(1) {
            for key in doc.keys() {
                if !headers.contains(key) {
                    headers.push(key.clone());
                }
            }
        }

        if headers.len() > options.max_columns {
            return Err(CsvError::MaxColumnsExceeded(headers.len()));
        }

        let mut records = Vec::with_capacity(docs.len());
        for doc in docs {
            let mut row = Vec::with_capacity(headers.len());
            for h in &headers {
                let val = doc.get(h).cloned().unwrap_or(DataValue::Null);
                row.push(val);
            }
            records.push(row);
        }

        Ok(Self {
            headers: if options.has_headers { Some(headers) } else { None },
            records,
        })
    }

    /// Parse a CSV/TSV string into a `CsvTable`.
    pub fn from_str(input: &str, options: &CsvOptions) -> Result<Self, CsvError> {
        if input.len() > options.max_total_bytes {
            return Err(CsvError::MaxTotalBytesExceeded(input.len()));
        }
        let mut reader = CsvParser::new(input.as_bytes(), options.clone());
        reader.parse_all()
    }

    /// Parse a CSV/TSV from a reader into a `CsvTable`.
    pub fn from_reader<R: Read>(reader: R, options: &CsvOptions) -> Result<Self, CsvError> {
        let mut parser = CsvParser::new(reader, options.clone());
        parser.parse_all()
    }

    /// Serialize this table into a CSV/TSV string.
    pub fn to_string(&self, options: &CsvOptions) -> Result<String, CsvError> {
        let mut buf = Vec::new();
        self.write_to(&mut buf, options)?;
        String::from_utf8(buf).map_err(|e| CsvError::InvalidUtf8(e.to_string()))
    }

    /// Serialize this table into a writer.
    pub fn write_to<W: Write>(&self, writer: &mut W, options: &CsvOptions) -> Result<(), CsvError> {
        if options.has_headers {
            if let Some(headers) = &self.headers {
                write_csv_row(writer, headers.iter().map(|s| s.as_str()), options)?;
            }
        }

        for row in &self.records {
            let string_cells: Vec<String> = row.iter().map(value_to_csv_string).collect();
            write_csv_row(writer, string_cells.iter().map(|s| s.as_str()), options)?;
        }

        Ok(())
    }
}

fn value_to_csv_string(val: &DataValue) -> String {
    match val {
        DataValue::Null => String::new(),
        DataValue::Bool(b) => b.to_string(),
        DataValue::Number(num) => num.to_string(),
        DataValue::String(s) => s.clone(),
        DataValue::Binary(bin) => format!("<binary len={}>", bin.len()),
        DataValue::Array(_) | DataValue::Document(_) => val.to_string(),
        DataValue::DateTimeUtc(dt) => dt.to_string(),
        DataValue::ObjectId(oid) => oid.to_hex(),
    }
}

fn write_csv_row<'a, W: Write, I: Iterator<Item = &'a str>>(
    writer: &mut W,
    cells: I,
    options: &CsvOptions,
) -> Result<(), CsvError> {
    let delim = options.delimiter;
    let quote = options.quote;

    let mut first = true;
    for cell in cells {
        if !first {
            writer.write_all(&[delim])?;
        }
        first = false;

        let needs_quote = cell.as_bytes().iter().any(|&b| {
            b == delim || b == quote || b == b'\n' || b == b'\r'
        }) || (options.trim && (cell.starts_with(' ') || cell.ends_with(' ')));

        if needs_quote {
            writer.write_all(&[quote])?;
            for &b in cell.as_bytes() {
                if b == quote {
                    writer.write_all(&[quote, quote])?;
                } else {
                    writer.write_all(&[b])?;
                }
            }
            writer.write_all(&[quote])?;
        } else {
            writer.write_all(cell.as_bytes())?;
        }
    }
    writer.write_all(b"\r\n")?;
    Ok(())
}

/// Streaming parser for CSV / TSV inputs over any `Read`.
pub struct CsvParser<R> {
    reader: R,
    options: CsvOptions,
    line_number: usize,
    col_number: usize,
    total_bytes_read: usize,
    eof_reached: bool,
}

impl<R: Read> CsvParser<R> {
    pub fn new(reader: R, options: CsvOptions) -> Self {
        Self {
            reader,
            options,
            line_number: 1,
            col_number: 1,
            total_bytes_read: 0,
            eof_reached: false,
        }
    }

    /// Read the next record row as a vector of DataValues.
    pub fn next_record(&mut self) -> Result<Option<Vec<DataValue>>, CsvError> {
        let raw_fields = match self.read_raw_fields()? {
            Some(fields) => fields,
            None => return Ok(None),
        };

        if raw_fields.is_empty() {
            return Ok(Some(Vec::new()));
        }

        let mut values = Vec::with_capacity(raw_fields.len());
        for field in raw_fields {
            let processed_str = if self.options.trim {
                field.trim().to_string()
            } else {
                field
            };

            let val = if self.options.parse_scalars {
                parse_scalar_inference(&processed_str)
            } else {
                DataValue::String(processed_str)
            };
            values.push(val);
        }

        Ok(Some(values))
    }

    fn read_raw_fields(&mut self) -> Result<Option<Vec<String>>, CsvError> {
        if self.eof_reached {
            return Ok(None);
        }

        let mut fields = Vec::new();
        let mut current_field = Vec::new();
        let mut in_quotes = false;
        let mut row_has_data = false;

        let delim = self.options.delimiter;
        let quote = self.options.quote;

        let mut quote_start_line = self.line_number;
        let mut quote_start_col = self.col_number;

        let mut byte_buf = [0u8; 1];

        loop {
            let bytes_read = self.reader.read(&mut byte_buf)?;
            if bytes_read == 0 {
                self.eof_reached = true;
                if in_quotes {
                    return Err(CsvError::UnterminatedQuote {
                        line: quote_start_line,
                        col: quote_start_col,
                    });
                }
                if row_has_data || !current_field.is_empty() {
                    let field_str = String::from_utf8(current_field)
                        .map_err(|e| CsvError::InvalidUtf8(e.to_string()))?;
                    fields.push(field_str);
                    return Ok(Some(fields));
                }
                return Ok(None);
            }

            self.total_bytes_read = self.total_bytes_read.saturating_add(bytes_read);
            if self.total_bytes_read > self.options.max_total_bytes {
                return Err(CsvError::MaxTotalBytesExceeded(self.total_bytes_read));
            }

            let b = byte_buf[0];
            self.col_number += 1;

            if in_quotes {
                if b == quote {
                    // Peek ahead for escaped quote
                    let mut peek_buf = [0u8; 1];
                    let peek_read = self.reader.read(&mut peek_buf)?;
                    if peek_read > 0 {
                        self.total_bytes_read = self.total_bytes_read.saturating_add(peek_read);
                        if self.total_bytes_read > self.options.max_total_bytes {
                            return Err(CsvError::MaxTotalBytesExceeded(self.total_bytes_read));
                        }
                    }
                    if peek_read > 0 && peek_buf[0] == quote {
                        // Escaped quote ("")
                        self.col_number += 1;
                        current_field.push(quote);
                    } else {
                        // Closing quote
                        in_quotes = false;
                        if peek_read > 0 {
                            let next_b = peek_buf[0];
                            self.col_number += 1;
                            if next_b == delim {
                                let field_str = String::from_utf8(std::mem::take(&mut current_field))
                                    .map_err(|e| CsvError::InvalidUtf8(e.to_string()))?;
                                fields.push(field_str);
                                row_has_data = true;
                            } else if next_b == b'\n' {
                                self.line_number += 1;
                                self.col_number = 1;
                                let field_str = String::from_utf8(std::mem::take(&mut current_field))
                                    .map_err(|e| CsvError::InvalidUtf8(e.to_string()))?;
                                fields.push(field_str);
                                return Ok(Some(fields));
                            } else if next_b == b'\r' {
                                // Check for CRLF
                                let mut crlf_buf = [0u8; 1];
                                let crlf_read = self.reader.read(&mut crlf_buf)?;
                                if crlf_read > 0 {
                                    self.total_bytes_read = self.total_bytes_read.saturating_add(crlf_read);
                                    if self.total_bytes_read > self.options.max_total_bytes {
                                        return Err(CsvError::MaxTotalBytesExceeded(self.total_bytes_read));
                                    }
                                }
                                if crlf_read > 0 && crlf_buf[0] != b'\n' {
                                    current_field.push(crlf_buf[0]);
                                }
                                self.line_number += 1;
                                self.col_number = 1;
                                let field_str = String::from_utf8(std::mem::take(&mut current_field))
                                    .map_err(|e| CsvError::InvalidUtf8(e.to_string()))?;
                                fields.push(field_str);
                                return Ok(Some(fields));
                            } else {
                                return Err(CsvError::InvalidQuotePlacement {
                                    line: self.line_number,
                                    col: self.col_number,
                                });
                            }
                        }
                    }
                } else {
                    if b == b'\n' {
                        self.line_number += 1;
                        self.col_number = 1;
                    }
                    current_field.push(b);
                }
            } else {
                if b == quote && current_field.is_empty() {
                    in_quotes = true;
                    quote_start_line = self.line_number;
                    quote_start_col = self.col_number;
                    row_has_data = true;
                } else if b == delim {
                    let field_str = String::from_utf8(std::mem::take(&mut current_field))
                        .map_err(|e| CsvError::InvalidUtf8(e.to_string()))?;
                    fields.push(field_str);
                    row_has_data = true;
                } else if b == b'\n' {
                    self.line_number += 1;
                    self.col_number = 1;
                    if row_has_data || !current_field.is_empty() {
                        let field_str = String::from_utf8(std::mem::take(&mut current_field))
                            .map_err(|e| CsvError::InvalidUtf8(e.to_string()))?;
                        fields.push(field_str);
                        return Ok(Some(fields));
                    }
                } else if b == b'\r' {
                    // Check for trailing \n
                    let mut peek_buf = [0u8; 1];
                    let peek_read = self.reader.read(&mut peek_buf)?;
                    if peek_read > 0 {
                        self.total_bytes_read = self.total_bytes_read.saturating_add(peek_read);
                        if self.total_bytes_read > self.options.max_total_bytes {
                            return Err(CsvError::MaxTotalBytesExceeded(self.total_bytes_read));
                        }
                    }
                    if peek_read > 0 && peek_buf[0] != b'\n' {
                        current_field.push(peek_buf[0]);
                    }
                    self.line_number += 1;
                    self.col_number = 1;
                    if row_has_data || !current_field.is_empty() {
                        let field_str = String::from_utf8(std::mem::take(&mut current_field))
                            .map_err(|e| CsvError::InvalidUtf8(e.to_string()))?;
                        fields.push(field_str);
                        return Ok(Some(fields));
                    }
                } else {
                    if b == quote {
                        return Err(CsvError::InvalidQuotePlacement {
                            line: self.line_number,
                            col: self.col_number,
                        });
                    }
                    current_field.push(b);
                    row_has_data = true;
                }
            }

            if current_field.len() > self.options.max_field_bytes {
                return Err(CsvError::MaxFieldBytesExceeded(current_field.len()));
            }
            if fields.len() > self.options.max_columns {
                return Err(CsvError::MaxColumnsExceeded(fields.len()));
            }
        }
    }

    pub fn parse_all(&mut self) -> Result<CsvTable, CsvError> {
        let headers = if self.options.has_headers {
            match self.read_raw_fields()? {
                Some(hdrs) => {
                    let processed_headers: Vec<String> = if self.options.trim {
                        hdrs.into_iter().map(|h| h.trim().to_string()).collect()
                    } else {
                        hdrs
                    };
                    if processed_headers.is_empty() {
                        return Err(CsvError::MissingHeader);
                    }
                    // Gate-SCHEMA-01 check
                    for (i, h) in processed_headers.iter().enumerate() {
                        for other in &processed_headers[i + 1..] {
                            if h == other {
                                return Err(CsvError::DuplicateHeader(h.clone()));
                            }
                        }
                    }
                    Some(processed_headers)
                }
                None => return Err(CsvError::EmptyInput),
            }
        } else {
            None
        };

        let expected_cols = headers.as_ref().map(|h| h.len());
        let mut records: Vec<Vec<DataValue>> = Vec::new();
        let mut row_idx = 1;

        while let Some(row) = self.next_record()? {
            if let Some(expected) = expected_cols {
                if !self.options.flexible && row.len() != expected {
                    return Err(CsvError::ColumnMismatch {
                        expected,
                        found: row.len(),
                        row: row_idx,
                    });
                }
            } else if !self.options.flexible && !records.is_empty() {
                let expected = records[0].len();
                if row.len() != expected {
                    return Err(CsvError::ColumnMismatch {
                        expected,
                        found: row.len(),
                        row: row_idx,
                    });
                }
            }

            records.push(row);
            row_idx += 1;

            if records.len() > self.options.max_rows {
                return Err(CsvError::MaxRowsExceeded(records.len()));
            }
        }

        Ok(CsvTable { headers, records })
    }
}

fn parse_scalar_inference(s: &str) -> DataValue {
    if s.is_empty() {
        return DataValue::String(String::new());
    }
    if s.eq_ignore_ascii_case("true") {
        return DataValue::Bool(true);
    }
    if s.eq_ignore_ascii_case("false") {
        return DataValue::Bool(false);
    }
    if s.eq_ignore_ascii_case("null") {
        return DataValue::Null;
    }
    if let Ok(i) = s.parse::<i64>() {
        return DataValue::Number(ExactNumber::Int(i));
    }
    if let Ok(u) = s.parse::<u64>() {
        return DataValue::Number(ExactNumber::UInt(u));
    }
    if let Ok(f) = s.parse::<f64>() {
        if f.is_finite() && s.contains('.') {
            return DataValue::Number(ExactNumber::Float(f));
        }
    }
    DataValue::String(s.to_string())
}

/// Convenience functions
pub fn parse_csv(input: &str) -> Result<CsvTable, CsvError> {
    CsvTable::from_str(input, &CsvOptions::csv())
}

pub fn parse_tsv(input: &str) -> Result<CsvTable, CsvError> {
    CsvTable::from_str(input, &CsvOptions::tsv())
}

pub fn to_csv_string(table: &CsvTable) -> Result<String, CsvError> {
    table.to_string(&CsvOptions::csv())
}

pub fn to_tsv_string(table: &CsvTable) -> Result<String, CsvError> {
    table.to_string(&CsvOptions::tsv())
}

pub fn documents_to_csv(docs: &[Document]) -> Result<String, CsvError> {
    let table = CsvTable::from_documents(docs, &CsvOptions::csv())?;
    table.to_string(&CsvOptions::csv())
}

pub fn documents_to_tsv(docs: &[Document]) -> Result<String, CsvError> {
    let table = CsvTable::from_documents(docs, &CsvOptions::tsv())?;
    table.to_string(&CsvOptions::tsv())
}

pub fn csv_to_documents(input: &str) -> Result<Vec<Document>, CsvError> {
    let table = parse_csv(input)?;
    table.to_documents()
}

pub fn tsv_to_documents(input: &str) -> Result<Vec<Document>, CsvError> {
    let table = parse_tsv(input)?;
    table.to_documents()
}

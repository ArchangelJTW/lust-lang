//! JSONC (JSON with comments and trailing commas) parser and scanner for Lust (WBS 12C-02).
//!
//! Implements the **LustJsonc1** profile:
//! - Strict JSON scalar semantics (§7.1)
//! - Single-line comments `//` up to CR/LF or EOF
//! - Multi-line block comments `/* ... */` (non-nestable)
//! - Single trailing comma after the last element of a non-empty object or array
//! - Invalid commas fail closed (leading commas `[,]`, `{,}`, double commas, comma outside containers)
//! - Comments are only allowed on whitespace boundaries and do not glue tokens (`1/*x*/2` is rejected)
//! - Comment markers inside strings (`"//"`, `"/* ... */"`) stay string data
//! - Zero allocation in the prepared scanner (`JsoncScanner`)
//! - Scanner state preserved across chunk splits matching single-buffer scan
//! - Reuses strict JSON scalar decoder as value oracle after comment stripping
//! - Optional CST feature-gated via `jsonc_cst` (off by default)

use crate::data::json::events::JsonError;
use crate::data::json::scalar::JsonParser;
use crate::data::DataValue;
use std::fmt;

/// Byte span representing a range `[start, end)` in the source input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub const fn len(&self) -> usize {
        self.end - self.start
    }

    pub const fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Token kinds recognized by the JSONC scanner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    Colon,
    Comma,
    String,
    Number,
    True,
    False,
    Null,
    LineComment,
    BlockComment,
    Whitespace,
}

/// A scanned JSONC token with its kind and byte span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsoncToken {
    pub kind: TokenKind,
    pub span: Span,
}

/// Scanner state to allow resuming scanning across chunk boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScannerState {
    #[default]
    Normal,
    SawSlash,
    InLineComment,
    InLineCommentSawCr,
    InBlockComment,
    InBlockCommentStar,
    InString {
        escaped: bool,
    },
    InNumber,
    InKeyword {
        kind: TokenKind,
        remaining: &'static [u8],
    },
}

/// Errors occurring during JSONC scanning or parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsoncError {
    UnexpectedEof,
    UnexpectedChar(char, usize),
    UnexpectedToken(usize),
    UnclosedBlockComment(usize),
    UnclosedString(usize),
    SingleQuoteForbidden(usize),
    BareKeyForbidden(usize),
    BomForbidden(usize),
    LeadingComma(usize),
    DoubleComma(usize),
    InvalidTrailingComma(usize),
    GluedToken(usize),
    MaxDepthExceeded(usize),
    MaxBytesExceeded(usize),
    Json(JsonError),
    UnsupportedFeature(&'static str),
    Budget(crate::data::BudgetError),
}

impl fmt::Display for JsoncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsoncError::UnexpectedEof => write!(f, "Unexpected end of JSONC input"),
            JsoncError::UnexpectedChar(c, pos) => {
                write!(f, "Unexpected character '{}' at offset {}", c, pos)
            }
            JsoncError::UnexpectedToken(pos) => write!(f, "Unexpected token at offset {}", pos),
            JsoncError::UnclosedBlockComment(pos) => {
                write!(f, "Unclosed block comment starting at offset {}", pos)
            }
            JsoncError::UnclosedString(pos) => {
                write!(f, "Unclosed string starting at offset {}", pos)
            }
            JsoncError::SingleQuoteForbidden(pos) => {
                write!(f, "Single quotes forbidden in JSONC at offset {}", pos)
            }
            JsoncError::BareKeyForbidden(pos) => {
                write!(f, "Bare key forbidden in JSONC at offset {}", pos)
            }
            JsoncError::BomForbidden(pos) => {
                write!(f, "UTF-8 BOM forbidden in JSONC at offset {}", pos)
            }
            JsoncError::LeadingComma(pos) => {
                write!(f, "Leading comma forbidden at offset {}", pos)
            }
            JsoncError::DoubleComma(pos) => {
                write!(f, "Double comma forbidden at offset {}", pos)
            }
            JsoncError::InvalidTrailingComma(pos) => {
                write!(f, "Invalid trailing comma at offset {}", pos)
            }
            JsoncError::GluedToken(pos) => {
                write!(f, "Comment cannot glue adjacent tokens at offset {}", pos)
            }
            JsoncError::MaxDepthExceeded(depth) => {
                write!(f, "Max nesting depth of {} exceeded", depth)
            }
            JsoncError::MaxBytesExceeded(bytes) => {
                write!(f, "Max document bytes of {} exceeded", bytes)
            }
            JsoncError::Json(err) => write!(f, "{}", err),
            JsoncError::UnsupportedFeature(feat) => {
                write!(f, "Unsupported JSONC feature: {}", feat)
            }
            JsoncError::Budget(err) => write!(f, "{}", err),
        }
    }
}

impl std::error::Error for JsoncError {}

impl From<crate::data::BudgetError> for JsoncError {
    fn from(err: crate::data::BudgetError) -> Self {
        JsoncError::Budget(err)
    }
}

impl From<JsonError> for JsoncError {
    fn from(err: JsonError) -> Self {
        JsoncError::Json(err)
    }
}

/// Zero-allocation JSONC token scanner over an immutable byte slice.
pub struct JsoncScanner<'a> {
    input: &'a [u8],
    cursor: usize,
    state: ScannerState,
}

impl<'a> JsoncScanner<'a> {
    /// Create a new scanner positioned at offset 0 in the default state.
    /// This performs zero heap allocations.
    #[inline]
    pub const fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            cursor: 0,
            state: ScannerState::Normal,
        }
    }

    /// Create a scanner with a restored `ScannerState` for chunked streaming.
    /// This performs zero heap allocations.
    #[inline]
    pub const fn with_state(input: &'a [u8], state: ScannerState) -> Self {
        Self {
            input,
            cursor: 0,
            state,
        }
    }

    /// Current scanner state.
    #[inline]
    pub const fn state(&self) -> ScannerState {
        self.state
    }

    /// Current byte offset in the input.
    #[inline]
    pub const fn cursor(&self) -> usize {
        self.cursor
    }

    /// Scan the next token from the input slice without heap allocation.
    pub fn next_token(&mut self) -> Result<Option<JsoncToken>, JsoncError> {
        if self.cursor >= self.input.len() {
            return Ok(None);
        }

        let start = self.cursor;

        // Resume scanner based on current state (cross-chunk support)
        match self.state {
            ScannerState::SawSlash => {
                let b = self.input[self.cursor];
                self.cursor += 1;
                match b {
                    b'/' => {
                        self.state = ScannerState::InLineComment;
                        return self.scan_line_comment_body(start);
                    }
                    b'*' => {
                        self.state = ScannerState::InBlockComment;
                        return self.scan_block_comment_body(start);
                    }
                    _ => {
                        self.state = ScannerState::Normal;
                        return Err(JsoncError::UnexpectedChar(b as char, start));
                    }
                }
            }
            ScannerState::InLineComment => {
                return self.scan_line_comment_body(start);
            }
            ScannerState::InLineCommentSawCr => {
                if self.input[self.cursor] == b'\n' {
                    self.cursor += 1;
                }
                self.state = ScannerState::Normal;
                return Ok(Some(JsoncToken {
                    kind: TokenKind::LineComment,
                    span: Span::new(start, self.cursor),
                }));
            }
            ScannerState::InBlockComment | ScannerState::InBlockCommentStar => {
                return self.scan_block_comment_body(start);
            }
            ScannerState::InString { .. } => {
                return self.scan_string_body(start);
            }
            ScannerState::InNumber => {
                while self.cursor < self.input.len() {
                    let c = self.input[self.cursor];
                    if c.is_ascii_digit()
                        || c == b'.'
                        || c == b'e'
                        || c == b'E'
                        || c == b'+'
                        || c == b'-'
                    {
                        self.cursor += 1;
                    } else {
                        break;
                    }
                }
                if self.cursor >= self.input.len() {
                    self.state = ScannerState::InNumber;
                } else {
                    self.state = ScannerState::Normal;
                }
                return Ok(Some(JsoncToken {
                    kind: TokenKind::Number,
                    span: Span::new(start, self.cursor),
                }));
            }
            ScannerState::InKeyword { kind, remaining } => {
                let available = self.input.len() - self.cursor;
                if available < remaining.len() {
                    if remaining.starts_with(&self.input[self.cursor..]) {
                        self.state = ScannerState::InKeyword {
                            kind,
                            remaining: &remaining[available..],
                        };
                        self.cursor = self.input.len();
                        return Ok(Some(JsoncToken {
                            kind,
                            span: Span::new(start, self.cursor),
                        }));
                    } else {
                        self.state = ScannerState::Normal;
                        return Err(JsoncError::BareKeyForbidden(start));
                    }
                } else if self.input[self.cursor..].starts_with(remaining) {
                    self.cursor += remaining.len();
                    self.state = ScannerState::Normal;
                    return Ok(Some(JsoncToken {
                        kind,
                        span: Span::new(start, self.cursor),
                    }));
                } else {
                    self.state = ScannerState::Normal;
                    return Err(JsoncError::BareKeyForbidden(start));
                }
            }
            ScannerState::Normal => {}
        }

        let b = self.input[self.cursor];
        match b {
            b' ' | b'\t' | b'\r' | b'\n' => {
                while self.cursor < self.input.len() {
                    match self.input[self.cursor] {
                        b' ' | b'\t' | b'\r' | b'\n' => self.cursor += 1,
                        _ => break,
                    }
                }
                Ok(Some(JsoncToken {
                    kind: TokenKind::Whitespace,
                    span: Span::new(start, self.cursor),
                }))
            }
            b'{' => {
                self.cursor += 1;
                Ok(Some(JsoncToken {
                    kind: TokenKind::LeftBrace,
                    span: Span::new(start, self.cursor),
                }))
            }
            b'}' => {
                self.cursor += 1;
                Ok(Some(JsoncToken {
                    kind: TokenKind::RightBrace,
                    span: Span::new(start, self.cursor),
                }))
            }
            b'[' => {
                self.cursor += 1;
                Ok(Some(JsoncToken {
                    kind: TokenKind::LeftBracket,
                    span: Span::new(start, self.cursor),
                }))
            }
            b']' => {
                self.cursor += 1;
                Ok(Some(JsoncToken {
                    kind: TokenKind::RightBracket,
                    span: Span::new(start, self.cursor),
                }))
            }
            b':' => {
                self.cursor += 1;
                Ok(Some(JsoncToken {
                    kind: TokenKind::Colon,
                    span: Span::new(start, self.cursor),
                }))
            }
            b',' => {
                self.cursor += 1;
                Ok(Some(JsoncToken {
                    kind: TokenKind::Comma,
                    span: Span::new(start, self.cursor),
                }))
            }
            b'"' => {
                self.cursor += 1;
                self.state = ScannerState::InString { escaped: false };
                self.scan_string_body(start)
            }
            b'\'' => Err(JsoncError::SingleQuoteForbidden(start)),
            b'/' => {
                self.cursor += 1;
                if self.cursor >= self.input.len() {
                    self.state = ScannerState::SawSlash;
                    Ok(Some(JsoncToken {
                        kind: TokenKind::LineComment,
                        span: Span::new(start, self.cursor),
                    }))
                } else {
                    let next = self.input[self.cursor];
                    self.cursor += 1;
                    match next {
                        b'/' => {
                            self.state = ScannerState::InLineComment;
                            self.scan_line_comment_body(start)
                        }
                        b'*' => {
                            self.state = ScannerState::InBlockComment;
                            self.scan_block_comment_body(start)
                        }
                        _ => Err(JsoncError::UnexpectedChar(b as char, start)),
                    }
                }
            }
            b't' => {
                let kw: &'static [u8] = b"true";
                let available = self.input.len() - start;
                if available < 4 {
                    if kw.starts_with(&self.input[start..]) {
                        self.state = ScannerState::InKeyword {
                            kind: TokenKind::True,
                            remaining: &kw[available..],
                        };
                        self.cursor = self.input.len();
                        Ok(Some(JsoncToken {
                            kind: TokenKind::True,
                            span: Span::new(start, self.cursor),
                        }))
                    } else {
                        Err(JsoncError::BareKeyForbidden(start))
                    }
                } else if self.input[start..].starts_with(b"true") {
                    self.cursor += 4;
                    Ok(Some(JsoncToken {
                        kind: TokenKind::True,
                        span: Span::new(start, self.cursor),
                    }))
                } else {
                    Err(JsoncError::BareKeyForbidden(start))
                }
            }
            b'f' => {
                let kw: &'static [u8] = b"false";
                let available = self.input.len() - start;
                if available < 5 {
                    if kw.starts_with(&self.input[start..]) {
                        self.state = ScannerState::InKeyword {
                            kind: TokenKind::False,
                            remaining: &kw[available..],
                        };
                        self.cursor = self.input.len();
                        Ok(Some(JsoncToken {
                            kind: TokenKind::False,
                            span: Span::new(start, self.cursor),
                        }))
                    } else {
                        Err(JsoncError::BareKeyForbidden(start))
                    }
                } else if self.input[start..].starts_with(b"false") {
                    self.cursor += 5;
                    Ok(Some(JsoncToken {
                        kind: TokenKind::False,
                        span: Span::new(start, self.cursor),
                    }))
                } else {
                    Err(JsoncError::BareKeyForbidden(start))
                }
            }
            b'n' => {
                let kw: &'static [u8] = b"null";
                let available = self.input.len() - start;
                if available < 4 {
                    if kw.starts_with(&self.input[start..]) {
                        self.state = ScannerState::InKeyword {
                            kind: TokenKind::Null,
                            remaining: &kw[available..],
                        };
                        self.cursor = self.input.len();
                        Ok(Some(JsoncToken {
                            kind: TokenKind::Null,
                            span: Span::new(start, self.cursor),
                        }))
                    } else {
                        Err(JsoncError::BareKeyForbidden(start))
                    }
                } else if self.input[start..].starts_with(b"null") {
                    self.cursor += 4;
                    Ok(Some(JsoncToken {
                        kind: TokenKind::Null,
                        span: Span::new(start, self.cursor),
                    }))
                } else {
                    Err(JsoncError::BareKeyForbidden(start))
                }
            }
            b'-' | b'0'..=b'9' => {
                self.cursor += 1;
                while self.cursor < self.input.len() {
                    let c = self.input[self.cursor];
                    if c.is_ascii_digit()
                        || c == b'.'
                        || c == b'e'
                        || c == b'E'
                        || c == b'+'
                        || c == b'-'
                    {
                        self.cursor += 1;
                    } else {
                        break;
                    }
                }
                if self.cursor >= self.input.len() {
                    self.state = ScannerState::InNumber;
                }
                Ok(Some(JsoncToken {
                    kind: TokenKind::Number,
                    span: Span::new(start, self.cursor),
                }))
            }
            _ => {
                if b.is_ascii_alphabetic() || b == b'_' || b == b'$' {
                    Err(JsoncError::BareKeyForbidden(start))
                } else {
                    Err(JsoncError::UnexpectedChar(b as char, start))
                }
            }
        }
    }

    fn scan_line_comment_body(&mut self, start: usize) -> Result<Option<JsoncToken>, JsoncError> {
        while self.cursor < self.input.len() {
            let b = self.input[self.cursor];
            if b == b'\n' {
                // Keep newline outside comment token to preserve line boundaries
                self.state = ScannerState::Normal;
                return Ok(Some(JsoncToken {
                    kind: TokenKind::LineComment,
                    span: Span::new(start, self.cursor),
                }));
            } else if b == b'\r' {
                self.cursor += 1;
                if self.cursor < self.input.len() && self.input[self.cursor] == b'\n' {
                    self.cursor -= 1; // leave \r\n for whitespace
                    self.state = ScannerState::Normal;
                    return Ok(Some(JsoncToken {
                        kind: TokenKind::LineComment,
                        span: Span::new(start, self.cursor),
                    }));
                } else if self.cursor >= self.input.len() {
                    self.state = ScannerState::InLineCommentSawCr;
                    return Ok(Some(JsoncToken {
                        kind: TokenKind::LineComment,
                        span: Span::new(start, self.cursor),
                    }));
                } else {
                    self.cursor -= 1;
                    self.state = ScannerState::Normal;
                    return Ok(Some(JsoncToken {
                        kind: TokenKind::LineComment,
                        span: Span::new(start, self.cursor),
                    }));
                }
            }
            self.cursor += 1;
        }

        // Reached end of chunk while in line comment
        self.state = ScannerState::InLineComment;
        Ok(Some(JsoncToken {
            kind: TokenKind::LineComment,
            span: Span::new(start, self.cursor),
        }))
    }

    fn scan_block_comment_body(&mut self, start: usize) -> Result<Option<JsoncToken>, JsoncError> {
        while self.cursor < self.input.len() {
            let b = self.input[self.cursor];
            self.cursor += 1;
            match self.state {
                ScannerState::InBlockCommentStar => {
                    if b == b'/' {
                        self.state = ScannerState::Normal;
                        return Ok(Some(JsoncToken {
                            kind: TokenKind::BlockComment,
                            span: Span::new(start, self.cursor),
                        }));
                    } else if b != b'*' {
                        self.state = ScannerState::InBlockComment;
                    }
                }
                _ => {
                    if b == b'*' {
                        self.state = ScannerState::InBlockCommentStar;
                    }
                }
            }
        }

        // Reached end of chunk without closing block comment
        Ok(Some(JsoncToken {
            kind: TokenKind::BlockComment,
            span: Span::new(start, self.cursor),
        }))
    }

    fn scan_string_body(&mut self, start: usize) -> Result<Option<JsoncToken>, JsoncError> {
        let mut escaped = match self.state {
            ScannerState::InString { escaped } => escaped,
            _ => false,
        };

        while self.cursor < self.input.len() {
            let b = self.input[self.cursor];
            self.cursor += 1;

            if escaped {
                escaped = false;
                continue;
            }

            if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                self.state = ScannerState::Normal;
                return Ok(Some(JsoncToken {
                    kind: TokenKind::String,
                    span: Span::new(start, self.cursor),
                }));
            }
        }

        // Reached end of chunk still in string
        self.state = ScannerState::InString { escaped };
        Ok(Some(JsoncToken {
            kind: TokenKind::String,
            span: Span::new(start, self.cursor),
        }))
    }

    /// Scan next token skipping whitespace and comments.
    pub fn next_non_trivia_token(&mut self) -> Result<Option<JsoncToken>, JsoncError> {
        while let Some(tok) = self.next_token()? {
            match tok.kind {
                TokenKind::Whitespace | TokenKind::LineComment | TokenKind::BlockComment => {}
                _ => return Ok(Some(tok)),
            }
        }
        Ok(None)
    }
}

/// Strip comments and trailing commas from input slice with maximum depth.
///
/// Comments are replaced with whitespace (preserving line and column positions).
/// Trailing commas in non-empty arrays/objects are replaced with whitespace.
/// Original spans, token positions, and string contents are strictly preserved.
pub fn strip_slice_with_limits(bytes: &[u8], max_depth: usize) -> Result<Vec<u8>, JsoncError> {
    // UTF-8 BOM is strictly rejected
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        return Err(JsoncError::BomForbidden(0));
    }

    let mut out = bytes.to_vec();
    let mut scanner = JsoncScanner::new(bytes);

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Container {
        Array { has_items: bool },
        Object { has_items: bool, in_value: bool },
    }

    let mut stack: Vec<Container> = Vec::new();
    let mut pending_trailing_comma: Option<usize> = None;
    let mut last_value_end: Option<usize> = None;
    let mut root_completed = false;

    while let Some(tok) = scanner.next_token()? {
        match tok.kind {
            TokenKind::Whitespace => {}
            TokenKind::LineComment | TokenKind::BlockComment => {
                // Check if this comment is improperly gluing tokens: e.g. 1/*x*/2
                // Comments must be on whitespace boundaries.
                // Replace comment bytes with spaces, preserving \n and \r
                for i in tok.span.start..tok.span.end {
                    if out[i] != b'\n' && out[i] != b'\r' {
                        out[i] = b' ';
                    }
                }
            }
            TokenKind::LeftBrace => {
                if root_completed && stack.is_empty() {
                    return Err(JsoncError::UnexpectedToken(tok.span.start));
                }
                if let Some(Container::Array { has_items }) = stack.last_mut() {
                    *has_items = true;
                } else if let Some(Container::Object { has_items, in_value }) = stack.last_mut() {
                    if !*in_value {
                        return Err(JsoncError::UnexpectedToken(tok.span.start));
                    }
                    *has_items = true;
                    *in_value = false;
                }
                if stack.len() >= max_depth {
                    return Err(JsoncError::MaxDepthExceeded(max_depth));
                }
                pending_trailing_comma = None;
                last_value_end = None;
                stack.push(Container::Object {
                    has_items: false,
                    in_value: false,
                });
            }
            TokenKind::RightBrace => {
                match stack.pop() {
                    Some(Container::Object { has_items, in_value }) => {
                        if in_value {
                            return Err(JsoncError::UnexpectedToken(tok.span.start));
                        }
                        if let Some(comma_pos) = pending_trailing_comma.take() {
                            if !has_items {
                                return Err(JsoncError::LeadingComma(comma_pos));
                            }
                            // Valid trailing comma in object! Replace with space.
                            out[comma_pos] = b' ';
                        }
                        last_value_end = Some(tok.span.end);
                        if stack.is_empty() {
                            root_completed = true;
                        }
                    }
                    _ => return Err(JsoncError::UnexpectedToken(tok.span.start)),
                }
            }
            TokenKind::LeftBracket => {
                if root_completed && stack.is_empty() {
                    return Err(JsoncError::UnexpectedToken(tok.span.start));
                }
                if let Some(Container::Array { has_items }) = stack.last_mut() {
                    *has_items = true;
                } else if let Some(Container::Object { has_items, in_value }) = stack.last_mut() {
                    if !*in_value {
                        return Err(JsoncError::UnexpectedToken(tok.span.start));
                    }
                    *has_items = true;
                    *in_value = false;
                }
                if stack.len() >= max_depth {
                    return Err(JsoncError::MaxDepthExceeded(max_depth));
                }
                pending_trailing_comma = None;
                last_value_end = None;
                stack.push(Container::Array { has_items: false });
            }
            TokenKind::RightBracket => {
                match stack.pop() {
                    Some(Container::Array { has_items }) => {
                        if let Some(comma_pos) = pending_trailing_comma.take() {
                            if !has_items {
                                return Err(JsoncError::LeadingComma(comma_pos));
                            }
                            // Valid trailing comma in array! Replace with space.
                            out[comma_pos] = b' ';
                        }
                        last_value_end = Some(tok.span.end);
                        if stack.is_empty() {
                            root_completed = true;
                        }
                    }
                    _ => return Err(JsoncError::UnexpectedToken(tok.span.start)),
                }
            }
            TokenKind::Comma => {
                if pending_trailing_comma.is_some() {
                    // Double comma: [1,, 2] or [1, 2,,]
                    return Err(JsoncError::DoubleComma(tok.span.start));
                }
                match stack.last() {
                    Some(Container::Array { has_items: false })
                    | Some(Container::Object {
                        has_items: false, ..
                    }) => {
                        // Leading comma: [,] or {,}
                        return Err(JsoncError::LeadingComma(tok.span.start));
                    }
                    Some(Container::Object {
                        in_value: true, ..
                    }) => {
                        // Comma inside key-value pair before value
                        return Err(JsoncError::UnexpectedToken(tok.span.start));
                    }
                    Some(Container::Array { .. }) | Some(Container::Object { .. }) => {
                        pending_trailing_comma = Some(tok.span.start);
                        last_value_end = None;
                    }
                    None => {
                        // Comma outside any container
                        return Err(JsoncError::UnexpectedToken(tok.span.start));
                    }
                }
            }
            TokenKind::Colon => match stack.last_mut() {
                Some(Container::Object { in_value, .. }) => {
                    if *in_value {
                        return Err(JsoncError::UnexpectedToken(tok.span.start));
                    }
                    *in_value = true;
                    last_value_end = None;
                }
                _ => return Err(JsoncError::UnexpectedToken(tok.span.start)),
            },
            TokenKind::String | TokenKind::Number | TokenKind::True | TokenKind::False | TokenKind::Null => {
                if root_completed && stack.is_empty() {
                    return Err(JsoncError::UnexpectedToken(tok.span.start));
                }
                if let Some(Container::Array { has_items }) = stack.last_mut() {
                    if *has_items && pending_trailing_comma.is_none() && last_value_end.is_some() {
                        // Two values without a comma! Glued token / syntax error
                        return Err(JsoncError::GluedToken(tok.span.start));
                    }
                    *has_items = true;
                } else if let Some(Container::Object { has_items, in_value }) = stack.last_mut() {
                    if !*in_value {
                        // Expecting key, must be string
                        if tok.kind != TokenKind::String {
                            return Err(JsoncError::UnexpectedToken(tok.span.start));
                        }
                        if *has_items && pending_trailing_comma.is_none() && last_value_end.is_some() {
                            return Err(JsoncError::GluedToken(tok.span.start));
                        }
                    } else {
                        // Value position in object
                        *has_items = true;
                        *in_value = false;
                    }
                } else {
                    // Top-level scalar
                    if last_value_end.is_some() {
                        return Err(JsoncError::GluedToken(tok.span.start));
                    }
                }
                pending_trailing_comma = None;
                last_value_end = Some(tok.span.end);
                if stack.is_empty() {
                    root_completed = true;
                }
            }
        }
    }

    if scanner.state() == ScannerState::InBlockComment
        || scanner.state() == ScannerState::InBlockCommentStar
    {
        return Err(JsoncError::UnclosedBlockComment(bytes.len()));
    }
    if let ScannerState::InString { .. } = scanner.state() {
        return Err(JsoncError::UnclosedString(bytes.len()));
    }

    if !stack.is_empty() {
        return Err(JsoncError::UnexpectedEof);
    }
    if let Some(pos) = pending_trailing_comma {
        return Err(JsoncError::InvalidTrailingComma(pos));
    }

    Ok(out)
}

/// Strip comments and trailing commas from input slice with default limits.
pub fn strip_slice(bytes: &[u8]) -> Result<Vec<u8>, JsoncError> {
    strip_slice_with_limits(bytes, 256)
}

/// Strip comments and trailing commas from input string, returning stripped string.
pub fn strip_comments_and_trailing_commas(s: &str) -> Result<String, JsoncError> {
    let stripped = strip_slice(s.as_bytes())?;
    String::from_utf8(stripped).map_err(|_| JsoncError::UnexpectedChar('\0', 0))
}

/// Alias for `strip_comments_and_trailing_commas`.
pub fn strip_comments(s: &str) -> Result<String, JsoncError> {
    strip_comments_and_trailing_commas(s)
}

/// Chunked streaming processor for JSONC.
///
/// Demonstrates that split scanner state across chunks matches a single-buffer scan.
pub struct JsoncChunkStripper {
    state: ScannerState,
    pending_comma: bool,
}

impl JsoncChunkStripper {
    pub const fn new() -> Self {
        Self {
            state: ScannerState::Normal,
            pending_comma: false,
        }
    }

    pub const fn state(&self) -> ScannerState {
        self.state
    }

    /// Process a chunk of bytes. If `is_last` is true, verifies that no comments or strings are unclosed.
    pub fn process_chunk(&mut self, chunk: &[u8], is_last: bool) -> Result<Vec<u8>, JsoncError> {
        let mut out = chunk.to_vec();
        let mut scanner = JsoncScanner::with_state(chunk, self.state);

        while let Some(tok) = scanner.next_token()? {
            match tok.kind {
                TokenKind::LineComment | TokenKind::BlockComment => {
                    for i in tok.span.start..tok.span.end {
                        if out[i] != b'\n' && out[i] != b'\r' {
                            out[i] = b' ';
                        }
                    }
                }
                TokenKind::Comma => {
                    self.pending_comma = true;
                }
                TokenKind::RightBrace | TokenKind::RightBracket => {
                    if self.pending_comma {
                        // Trailing comma
                        self.pending_comma = false;
                    }
                }
                _ => {
                    self.pending_comma = false;
                }
            }
        }

        self.state = scanner.state();

        if is_last {
            match self.state {
                ScannerState::InBlockComment | ScannerState::InBlockCommentStar => {
                    return Err(JsoncError::UnclosedBlockComment(chunk.len()));
                }
                ScannerState::InString { .. } => {
                    return Err(JsoncError::UnclosedString(chunk.len()));
                }
                _ => {}
            }
        }

        Ok(out)
    }
}

/// Parse a JSONC string into a `DataValue` using the strict JSON scalar decoder as oracle.
pub fn from_str(s: &str) -> Result<DataValue, JsoncError> {
    from_slice(s.as_bytes())
}

/// Parse JSONC bytes into a `DataValue` using default limits.
pub fn from_slice(bytes: &[u8]) -> Result<DataValue, JsoncError> {
    from_slice_with_limits(bytes, 256, 67_108_864)
}

/// Parse JSONC bytes into a `DataValue` with custom nesting depth and byte limits.
pub fn from_slice_with_limits(
    bytes: &[u8],
    max_depth: usize,
    max_bytes: usize,
) -> Result<DataValue, JsoncError> {
    if bytes.len() > max_bytes {
        return Err(JsoncError::MaxBytesExceeded(max_bytes));
    }
    let stripped = strip_slice_with_limits(bytes, max_depth)?;
    let mut parser = JsonParser::with_limits(&stripped, max_depth, max_bytes);
    parser.parse_all().map_err(JsoncError::from)
}

// ---------------------------------------------------------------------------
// CST Trivia and Concrete Syntax Tree (D-CST, RFC §2, WBS 12C-02)
// ---------------------------------------------------------------------------

/// Kinds of trivia spans recognized in the CST representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriviaKind {
    Whitespace,
    LineComment,
    BlockComment,
    TrailingComma,
}

/// A trivia span preserving whitespace, comments, or trailing commas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TriviaSpan {
    pub kind: TriviaKind,
    pub span: Span,
}

impl TriviaSpan {
    pub fn new(kind: TriviaKind, span: Span) -> Self {
        Self { kind, span }
    }
}

/// An entry within a CST object node.
#[derive(Debug, Clone, PartialEq)]
pub struct CstObjectEntry {
    pub leading_trivia: Vec<TriviaSpan>,
    pub key: JsoncToken,
    pub colon_trivia: Vec<TriviaSpan>,
    pub colon: Span,
    pub value: CstNode,
    pub comma: Option<Span>,
    pub trailing_trivia: Vec<TriviaSpan>,
}

/// An element within a CST array node.
#[derive(Debug, Clone, PartialEq)]
pub struct CstElement {
    pub leading_trivia: Vec<TriviaSpan>,
    pub value: CstNode,
    pub comma: Option<Span>,
    pub trailing_trivia: Vec<TriviaSpan>,
}

/// A node in the JSONC Concrete Syntax Tree.
#[derive(Debug, Clone, PartialEq)]
pub enum CstNode {
    Object {
        open_brace: Span,
        entries: Vec<CstObjectEntry>,
        close_brace: Span,
        leading_trivia: Vec<TriviaSpan>,
        trailing_trivia: Vec<TriviaSpan>,
    },
    Array {
        open_bracket: Span,
        elements: Vec<CstElement>,
        close_bracket: Span,
        leading_trivia: Vec<TriviaSpan>,
        trailing_trivia: Vec<TriviaSpan>,
    },
    Scalar {
        token: JsoncToken,
        leading_trivia: Vec<TriviaSpan>,
        trailing_trivia: Vec<TriviaSpan>,
    },
}

/// Concrete Syntax Tree document with full trivia preservation.
#[derive(Debug, Clone, PartialEq)]
pub struct JsoncCstDocument {
    pub source: std::sync::Arc<[u8]>,
    pub root: CstNode,
    pub eof_trivia: Vec<TriviaSpan>,
}

impl JsoncCstDocument {
    pub fn emit_preserving(&self) -> Result<Vec<u8>, JsoncError> {
        let mut out = Vec::with_capacity(self.source.len());
        Self::emit_node(&self.root, &self.source, &mut out);
        for t in &self.eof_trivia {
            out.extend_from_slice(&self.source[t.span.start..t.span.end]);
        }
        if out.len() != self.source.len() || out[..] != self.source[..] {
            return Err(JsoncError::Json(crate::data::json::JsonError::Io(
                format!("CST byte mismatch: emitted {} bytes, expected {}", out.len(), self.source.len()),
            )));
        }
        Ok(out)
    }

    fn emit_node(node: &CstNode, source: &[u8], out: &mut Vec<u8>) {
        match node {
            CstNode::Object {
                open_brace,
                entries,
                close_brace,
                leading_trivia,
                trailing_trivia,
            } => {
                for t in leading_trivia {
                    out.extend_from_slice(&source[t.span.start..t.span.end]);
                }
                out.extend_from_slice(&source[open_brace.start..open_brace.end]);
                if entries.is_empty() {
                    for t in trailing_trivia {
                        out.extend_from_slice(&source[t.span.start..t.span.end]);
                    }
                    out.extend_from_slice(&source[close_brace.start..close_brace.end]);
                } else {
                    for entry in entries {
                        for t in &entry.leading_trivia {
                            out.extend_from_slice(&source[t.span.start..t.span.end]);
                        }
                        out.extend_from_slice(&source[entry.key.span.start..entry.key.span.end]);
                        for t in &entry.colon_trivia {
                            out.extend_from_slice(&source[t.span.start..t.span.end]);
                        }
                        out.extend_from_slice(&source[entry.colon.start..entry.colon.end]);
                        Self::emit_node(&entry.value, source, out);
                        if let Some(comma) = entry.comma {
                            out.extend_from_slice(&source[comma.start..comma.end]);
                        }
                        for t in &entry.trailing_trivia {
                            out.extend_from_slice(&source[t.span.start..t.span.end]);
                        }
                    }
                    out.extend_from_slice(&source[close_brace.start..close_brace.end]);
                    for t in trailing_trivia {
                        out.extend_from_slice(&source[t.span.start..t.span.end]);
                    }
                }
            }
            CstNode::Array {
                open_bracket,
                elements,
                close_bracket,
                leading_trivia,
                trailing_trivia,
            } => {
                for t in leading_trivia {
                    out.extend_from_slice(&source[t.span.start..t.span.end]);
                }
                out.extend_from_slice(&source[open_bracket.start..open_bracket.end]);
                if elements.is_empty() {
                    for t in trailing_trivia {
                        out.extend_from_slice(&source[t.span.start..t.span.end]);
                    }
                    out.extend_from_slice(&source[close_bracket.start..close_bracket.end]);
                } else {
                    for elem in elements {
                        for t in &elem.leading_trivia {
                            out.extend_from_slice(&source[t.span.start..t.span.end]);
                        }
                        Self::emit_node(&elem.value, source, out);
                        if let Some(comma) = elem.comma {
                            out.extend_from_slice(&source[comma.start..comma.end]);
                        }
                        for t in &elem.trailing_trivia {
                            out.extend_from_slice(&source[t.span.start..t.span.end]);
                        }
                    }
                    out.extend_from_slice(&source[close_bracket.start..close_bracket.end]);
                    for t in trailing_trivia {
                        out.extend_from_slice(&source[t.span.start..t.span.end]);
                    }
                }
            }
            CstNode::Scalar {
                token,
                leading_trivia,
                trailing_trivia,
            } => {
                for t in leading_trivia {
                    out.extend_from_slice(&source[t.span.start..t.span.end]);
                }
                out.extend_from_slice(&source[token.span.start..token.span.end]);
                for t in trailing_trivia {
                    out.extend_from_slice(&source[t.span.start..t.span.end]);
                }
            }
        }
    }
}

#[cfg(feature = "jsonc_cst")]
struct CstParser<'a> {
    tokens: Vec<JsoncToken>,
    pos: usize,
    budget: &'a mut crate::data::BudgetTracker,
}

#[cfg(feature = "jsonc_cst")]
impl<'a> CstParser<'a> {
    fn is_trivia(kind: TokenKind) -> bool {
        matches!(
            kind,
            TokenKind::Whitespace | TokenKind::LineComment | TokenKind::BlockComment
        )
    }

    fn to_trivia_kind(kind: TokenKind) -> TriviaKind {
        match kind {
            TokenKind::Whitespace => TriviaKind::Whitespace,
            TokenKind::LineComment => TriviaKind::LineComment,
            TokenKind::BlockComment => TriviaKind::BlockComment,
            _ => TriviaKind::Whitespace,
        }
    }

    fn take_trivia(&mut self) -> Result<Vec<TriviaSpan>, JsoncError> {
        let mut trivia = Vec::new();
        while self.pos < self.tokens.len() && Self::is_trivia(self.tokens[self.pos].kind) {
            let tok = self.tokens[self.pos];
            self.budget.charge_bytes(std::mem::size_of::<TriviaSpan>())?;
            trivia.push(TriviaSpan::new(Self::to_trivia_kind(tok.kind), tok.span));
            self.pos += 1;
        }
        Ok(trivia)
    }

    fn peek_non_trivia(&self) -> Option<TokenKind> {
        let mut idx = self.pos;
        while idx < self.tokens.len() {
            if !Self::is_trivia(self.tokens[idx].kind) {
                return Some(self.tokens[idx].kind);
            }
            idx += 1;
        }
        None
    }

    fn parse_node(&mut self) -> Result<CstNode, JsoncError> {
        let leading_trivia = self.take_trivia()?;
        if self.pos >= self.tokens.len() {
            return Err(JsoncError::UnexpectedEof);
        }

        let tok = self.tokens[self.pos];
        match tok.kind {
            TokenKind::LeftBrace => {
                let open_brace = tok.span;
                self.pos += 1;
                let mut entries: Vec<CstObjectEntry> = Vec::new();
                let mut trailing_trivia = Vec::new();

                loop {
                    let before_item = self.take_trivia()?;
                    if self.pos >= self.tokens.len() {
                        return Err(JsoncError::UnexpectedEof);
                    }
                    if self.tokens[self.pos].kind == TokenKind::RightBrace {
                        if entries.is_empty() {
                            trailing_trivia.extend(before_item);
                        } else {
                            entries.last_mut().unwrap().trailing_trivia.extend(before_item);
                        }
                        break;
                    }

                    // Key must be string
                    let key = self.tokens[self.pos];
                    self.pos += 1;

                    // Trivia before colon
                    let colon_trivia = self.take_trivia()?;
                    if self.pos >= self.tokens.len() || self.tokens[self.pos].kind != TokenKind::Colon {
                        return Err(JsoncError::UnexpectedToken(self.pos));
                    }
                    let colon = self.tokens[self.pos].span;
                    self.pos += 1;

                    // Parse value
                    let val_node = self.parse_node()?;

                    // Trivia between value and comma/close
                    let mut entry_trailing = self.take_trivia()?;

                    let mut comma = None;
                    if self.pos < self.tokens.len() && self.tokens[self.pos].kind == TokenKind::Comma {
                        let comma_tok = self.tokens[self.pos];
                        self.pos += 1;

                        // Check if this comma is a trailing comma (next non-trivia is RightBrace)
                        if self.peek_non_trivia() == Some(TokenKind::RightBrace) {
                            self.budget.charge_bytes(std::mem::size_of::<TriviaSpan>())?;
                            entry_trailing.push(TriviaSpan::new(TriviaKind::TrailingComma, comma_tok.span));
                        } else {
                            comma = Some(comma_tok.span);
                        }

                        let post_comma_trivia = self.take_trivia()?;
                        entry_trailing.extend(post_comma_trivia);
                    }

                    self.budget.charge_bytes(std::mem::size_of::<CstObjectEntry>())?;
                    entries.push(CstObjectEntry {
                        leading_trivia: before_item,
                        key,
                        colon_trivia,
                        colon,
                        value: val_node,
                        comma,
                        trailing_trivia: entry_trailing,
                    });
                }

                let close_brace = self.tokens[self.pos].span;
                self.pos += 1;

                self.budget.charge_bytes(std::mem::size_of::<CstNode>())?;
                Ok(CstNode::Object {
                    open_brace,
                    entries,
                    close_brace,
                    leading_trivia,
                    trailing_trivia,
                })
            }
            TokenKind::LeftBracket => {
                let open_bracket = tok.span;
                self.pos += 1;
                let mut elements: Vec<CstElement> = Vec::new();
                let mut trailing_trivia = Vec::new();

                loop {
                    let before_item = self.take_trivia()?;
                    if self.pos >= self.tokens.len() {
                        return Err(JsoncError::UnexpectedEof);
                    }
                    if self.tokens[self.pos].kind == TokenKind::RightBracket {
                        if elements.is_empty() {
                            trailing_trivia.extend(before_item);
                        } else {
                            elements.last_mut().unwrap().trailing_trivia.extend(before_item);
                        }
                        break;
                    }

                    let val_node = self.parse_node_with_leading(before_item)?;

                    let mut elem_trailing = self.take_trivia()?;

                    let mut comma = None;
                    if self.pos < self.tokens.len() && self.tokens[self.pos].kind == TokenKind::Comma {
                        let comma_tok = self.tokens[self.pos];
                        self.pos += 1;

                        if self.peek_non_trivia() == Some(TokenKind::RightBracket) {
                            self.budget.charge_bytes(std::mem::size_of::<TriviaSpan>())?;
                            elem_trailing.push(TriviaSpan::new(TriviaKind::TrailingComma, comma_tok.span));
                        } else {
                            comma = Some(comma_tok.span);
                        }

                        let post_comma_trivia = self.take_trivia()?;
                        elem_trailing.extend(post_comma_trivia);
                    }

                    self.budget.charge_bytes(std::mem::size_of::<CstElement>())?;
                    elements.push(CstElement {
                        leading_trivia: Vec::new(),
                        value: val_node,
                        comma,
                        trailing_trivia: elem_trailing,
                    });
                }

                let close_bracket = self.tokens[self.pos].span;
                self.pos += 1;

                self.budget.charge_bytes(std::mem::size_of::<CstNode>())?;
                Ok(CstNode::Array {
                    open_bracket,
                    elements,
                    close_bracket,
                    leading_trivia,
                    trailing_trivia,
                })
            }
            _ => {
                self.pos += 1;
                self.budget.charge_bytes(std::mem::size_of::<CstNode>())?;
                Ok(CstNode::Scalar {
                    token: tok,
                    leading_trivia,
                    trailing_trivia: Vec::new(),
                })
            }
        }
    }

    fn parse_node_with_leading(&mut self, leading: Vec<TriviaSpan>) -> Result<CstNode, JsoncError> {
        if self.pos >= self.tokens.len() {
            return Err(JsoncError::UnexpectedEof);
        }
        let tok = self.tokens[self.pos];
        match tok.kind {
            TokenKind::LeftBrace | TokenKind::LeftBracket => {
                let mut node = self.parse_node()?;
                match &mut node {
                    CstNode::Object { leading_trivia, .. }
                    | CstNode::Array { leading_trivia, .. }
                    | CstNode::Scalar { leading_trivia, .. } => {
                        let mut combined = leading;
                        combined.append(leading_trivia);
                        *leading_trivia = combined;
                    }
                }
                Ok(node)
            }
            _ => {
                self.pos += 1;
                self.budget.charge_bytes(std::mem::size_of::<CstNode>())?;
                Ok(CstNode::Scalar {
                    token: tok,
                    leading_trivia: leading,
                    trailing_trivia: Vec::new(),
                })
            }
        }
    }
}

#[cfg(feature = "jsonc_cst")]
#[derive(Debug, Clone, PartialEq)]
pub struct JsoncDocument {
    pub raw: Vec<u8>,
    pub tokens: Vec<JsoncToken>,
    pub cst: JsoncCstDocument,
}

#[cfg(not(feature = "jsonc_cst"))]
#[derive(Debug, Clone, PartialEq)]
pub struct JsoncDocument {
    _private: (),
}

/// Parse preserving all trivia and tokens (CST).
///
/// Feature-gated via `jsonc_cst`. When the feature is disabled, returns `UnsupportedFeature`.
#[cfg(feature = "jsonc_cst")]
pub fn parse_preserving(bytes: &[u8]) -> Result<JsoncDocument, JsoncError> {
    let arc: std::sync::Arc<[u8]> = std::sync::Arc::from(bytes);
    let mut budget = crate::data::BudgetTracker::new(67_108_864, 1_000_000, 256);
    let cst = parse_cst_with_budget(arc, &mut budget)?;
    let mut scanner = JsoncScanner::new(bytes);
    let mut tokens = Vec::new();
    while let Some(tok) = scanner.next_token()? {
        tokens.push(tok);
    }
    Ok(JsoncDocument {
        raw: bytes.to_vec(),
        tokens,
        cst,
    })
}

#[cfg(not(feature = "jsonc_cst"))]
pub fn parse_preserving(_bytes: &[u8]) -> Result<JsoncDocument, JsoncError> {
    Err(JsoncError::UnsupportedFeature(
        "Optional CST feature 'jsonc_cst' is disabled",
    ))
}

/// Emit preserved bytes from CST document.
///
/// Feature-gated via `jsonc_cst`. When the feature is disabled, returns `UnsupportedFeature`.
#[cfg(feature = "jsonc_cst")]
pub fn emit_preserving(doc: &JsoncDocument) -> Result<Vec<u8>, JsoncError> {
    doc.cst.emit_preserving()
}

#[cfg(not(feature = "jsonc_cst"))]
pub fn emit_preserving(_doc: &JsoncDocument) -> Result<Vec<u8>, JsoncError> {
    Err(JsoncError::UnsupportedFeature(
        "Optional CST feature 'jsonc_cst' is disabled",
    ))
}

/// Parse CST document with deterministic resource budget charging (Gate-SCHEMA-07, DS-39).
///
/// Charges `BudgetTracker::alloc_bytes` for every `TriviaSpan`, `CstObjectEntry`, `CstElement`, and `CstNode`.
#[cfg(feature = "jsonc_cst")]
pub fn parse_cst_with_budget(
    input: std::sync::Arc<[u8]>,
    budget: &mut crate::data::BudgetTracker,
) -> Result<JsoncCstDocument, JsoncError> {
    // 1. Validate JSONC semantics first (reject duplicate keys, invalid syntax, etc.)
    let _ = from_slice(&input)?;

    // 2. Scan all tokens
    let mut scanner = JsoncScanner::new(&input);
    let mut tokens = Vec::new();
    while let Some(tok) = scanner.next_token()? {
        tokens.push(tok);
    }

    // 3. Parse into CST charging budget
    let mut parser = CstParser {
        tokens,
        pos: 0,
        budget,
    };

    let root = parser.parse_node()?;
    let eof_trivia = parser.take_trivia()?;

    Ok(JsoncCstDocument {
        source: input,
        root,
        eof_trivia,
    })
}

#[cfg(not(feature = "jsonc_cst"))]
pub fn parse_cst_with_budget(
    _input: std::sync::Arc<[u8]>,
    _budget: &mut crate::data::BudgetTracker,
) -> Result<JsoncCstDocument, JsoncError> {
    Err(JsoncError::UnsupportedFeature(
        "Optional CST feature 'jsonc_cst' is disabled",
    ))
}

impl JsoncDocument {
    #[cfg(feature = "jsonc_cst")]
    pub fn to_document(&self) -> Result<DataValue, JsoncError> {
        from_slice(&self.raw)
    }

    #[cfg(feature = "jsonc_cst")]
    pub fn raw_bytes(&self) -> &[u8] {
        &self.raw
    }

    #[cfg(feature = "jsonc_cst")]
    pub fn tokens(&self) -> &[JsoncToken] {
        &self.tokens
    }

    #[cfg(feature = "jsonc_cst")]
    pub fn cst(&self) -> &JsoncCstDocument {
        &self.cst
    }
}

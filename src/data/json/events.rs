use crate::data::{DataValue, Document, ExactNumber};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum JsonEvent {
    StartObject,
    EndObject,
    StartArray,
    EndArray,
    Key(String),
    Scalar(DataValue),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsonError {
    UnexpectedEof,
    UnexpectedChar(char, usize),
    DuplicateKey(String, usize),
    MaxDepthExceeded(usize),
    MaxBytesExceeded(usize),
    TrailingBytes(usize),
    TrailingComma(usize),
    InvalidEscape(usize),
    InvalidUnicode(usize),
    InvalidNumber(usize),
    Io(String),
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsonError::UnexpectedEof => write!(f, "Unexpected end of JSON input"),
            JsonError::UnexpectedChar(c, pos) => write!(f, "Unexpected character '{}' at offset {}", c, pos),
            JsonError::DuplicateKey(k, pos) => write!(f, "Duplicate key '{}' rejected at offset {}", k, pos),
            JsonError::MaxDepthExceeded(depth) => write!(f, "Max nesting depth of {} exceeded", depth),
            JsonError::MaxBytesExceeded(bytes) => write!(f, "Max document bytes of {} exceeded", bytes),
            JsonError::TrailingBytes(pos) => write!(f, "Trailing unconsumed bytes at offset {}", pos),
            JsonError::TrailingComma(pos) => write!(f, "Trailing comma forbidden in strict mode at offset {}", pos),
            JsonError::InvalidEscape(pos) => write!(f, "Invalid escape sequence at offset {}", pos),
            JsonError::InvalidUnicode(pos) => write!(f, "Invalid unicode escape at offset {}", pos),
            JsonError::InvalidNumber(pos) => write!(f, "Invalid number syntax at offset {}", pos),
            JsonError::Io(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for JsonError {}

/// Convert a `DataValue` into a stream of canonical `JsonEvent`s.
pub fn to_events(value: &DataValue) -> Vec<JsonEvent> {
    let mut events = Vec::new();
    emit_events_recursive(value, &mut events);
    events
}

fn emit_events_recursive(value: &DataValue, out: &mut Vec<JsonEvent>) {
    match value {
        DataValue::Document(doc) => {
            out.push(JsonEvent::StartObject);
            for (k, v) in doc.iter() {
                out.push(JsonEvent::Key(k.clone()));
                emit_events_recursive(v, out);
            }
            out.push(JsonEvent::EndObject);
        }
        DataValue::Array(items) => {
            out.push(JsonEvent::StartArray);
            for item in items {
                emit_events_recursive(item, out);
            }
            out.push(JsonEvent::EndArray);
        }
        scalar => {
            out.push(JsonEvent::Scalar(scalar.clone()));
        }
    }
}

/// Reconstruct a `DataValue` from a sequence of `JsonEvent`s.
pub fn from_events<I: IntoIterator<Item = JsonEvent>>(events: I) -> Result<DataValue, JsonError> {
    from_events_with_limits(events, 256)
}

/// Reconstruct a `DataValue` from a sequence of `JsonEvent`s with max nesting depth limit.
pub fn from_events_with_limits<I: IntoIterator<Item = JsonEvent>>(
    events: I,
    max_depth: usize,
) -> Result<DataValue, JsonError> {
    let mut iter = events.into_iter();
    let first = iter.next().ok_or(JsonError::UnexpectedEof)?;
    let val = parse_event_node(first, &mut iter, 0, max_depth)?;
    if iter.next().is_some() {
        return Err(JsonError::TrailingBytes(0));
    }
    Ok(val)
}

fn parse_event_node<I: Iterator<Item = JsonEvent>>(
    current: JsonEvent,
    iter: &mut I,
    depth: usize,
    max_depth: usize,
) -> Result<DataValue, JsonError> {
    if depth > max_depth {
        return Err(JsonError::MaxDepthExceeded(max_depth));
    }

    match current {
        JsonEvent::Scalar(val) => Ok(val),
        JsonEvent::StartArray => {
            let mut arr = Vec::new();
            loop {
                let next = iter.next().ok_or(JsonError::UnexpectedEof)?;
                match next {
                    JsonEvent::EndArray => break,
                    other => {
                        let child = parse_event_node(other, iter, depth + 1, max_depth)?;
                        arr.push(child);
                    }
                }
            }
            Ok(DataValue::Array(arr))
        }
        JsonEvent::StartObject => {
            let mut doc = Document::new();
            loop {
                let next = iter.next().ok_or(JsonError::UnexpectedEof)?;
                match next {
                    JsonEvent::EndObject => break,
                    JsonEvent::Key(key) => {
                        let val_event = iter.next().ok_or(JsonError::UnexpectedEof)?;
                        let val = parse_event_node(val_event, iter, depth + 1, max_depth)?;
                        if doc.try_insert(key.clone(), val).is_err() {
                            return Err(JsonError::DuplicateKey(key, 0));
                        }
                    }
                    _ => return Err(JsonError::UnexpectedChar('?', 0)),
                }
            }
            Ok(DataValue::Document(doc))
        }
        JsonEvent::EndArray | JsonEvent::EndObject | JsonEvent::Key(_) => {
            Err(JsonError::UnexpectedChar('?', 0))
        }
    }
}

/// Tokenize raw JSON bytes directly into an event stream, verifying syntax and structure.
pub fn stream_events(input: &[u8]) -> Result<Vec<JsonEvent>, JsonError> {
    stream_events_with_limits(input, 256, 67_108_864)
}

/// Tokenize raw JSON bytes into an event stream with depth and byte limits.
pub fn stream_events_with_limits(
    input: &[u8],
    max_depth: usize,
    max_bytes: usize,
) -> Result<Vec<JsonEvent>, JsonError> {
    let mut streamer = EventStreamer::new(input, max_depth, max_bytes);
    streamer.stream_all()
}

struct EventStreamer<'a> {
    input: &'a [u8],
    pos: usize,
    depth: usize,
    max_depth: usize,
    max_bytes: usize,
}

impl<'a> EventStreamer<'a> {
    fn new(input: &'a [u8], max_depth: usize, max_bytes: usize) -> Self {
        Self {
            input,
            pos: 0,
            depth: 0,
            max_depth,
            max_bytes,
        }
    }

    fn stream_all(&mut self) -> Result<Vec<JsonEvent>, JsonError> {
        if self.input.len() > self.max_bytes {
            return Err(JsonError::MaxBytesExceeded(self.max_bytes));
        }

        self.skip_whitespace();
        if self.is_eof() {
            return Err(JsonError::UnexpectedEof);
        }

        let mut events = Vec::new();
        self.stream_value(&mut events)?;
        self.skip_whitespace();
        if !self.is_eof() {
            return Err(JsonError::TrailingBytes(self.pos));
        }
        Ok(events)
    }

    fn stream_value(&mut self, out: &mut Vec<JsonEvent>) -> Result<(), JsonError> {
        self.skip_whitespace();
        if self.is_eof() {
            return Err(JsonError::UnexpectedEof);
        }

        match self.peek_byte()? {
            b'{' => self.stream_object(out),
            b'[' => self.stream_array(out),
            b'"' => {
                let s = self.parse_string()?;
                out.push(JsonEvent::Scalar(DataValue::String(s)));
                Ok(())
            }
            b't' => {
                self.consume_slice(b"true")?;
                out.push(JsonEvent::Scalar(DataValue::Bool(true)));
                Ok(())
            }
            b'f' => {
                self.consume_slice(b"false")?;
                out.push(JsonEvent::Scalar(DataValue::Bool(false)));
                Ok(())
            }
            b'n' => {
                self.consume_slice(b"null")?;
                out.push(JsonEvent::Scalar(DataValue::Null));
                Ok(())
            }
            b'-' | b'0'..=b'9' => {
                let num = self.parse_number()?;
                out.push(JsonEvent::Scalar(num));
                Ok(())
            }
            b => Err(JsonError::UnexpectedChar(b as char, self.pos)),
        }
    }

    fn stream_object(&mut self, out: &mut Vec<JsonEvent>) -> Result<(), JsonError> {
        self.consume_byte(b'{')?;
        self.depth += 1;
        if self.depth > self.max_depth {
            return Err(JsonError::MaxDepthExceeded(self.max_depth));
        }

        out.push(JsonEvent::StartObject);
        self.skip_whitespace();

        if self.peek_byte()? == b'}' {
            self.consume_byte(b'}')?;
            self.depth -= 1;
            out.push(JsonEvent::EndObject);
            return Ok(());
        }

        loop {
            self.skip_whitespace();
            if self.peek_byte()? == b',' {
                return Err(JsonError::TrailingComma(self.pos));
            }

            let key = self.parse_string()?;
            out.push(JsonEvent::Key(key));

            self.skip_whitespace();
            self.consume_byte(b':')?;
            self.stream_value(out)?;

            self.skip_whitespace();
            match self.peek_byte()? {
                b',' => {
                    self.consume_byte(b',')?;
                    self.skip_whitespace();
                    if self.peek_byte()? == b'}' {
                        return Err(JsonError::TrailingComma(self.pos));
                    }
                }
                b'}' => {
                    self.consume_byte(b'}')?;
                    break;
                }
                b => return Err(JsonError::UnexpectedChar(b as char, self.pos)),
            }
        }

        self.depth -= 1;
        out.push(JsonEvent::EndObject);
        Ok(())
    }

    fn stream_array(&mut self, out: &mut Vec<JsonEvent>) -> Result<(), JsonError> {
        self.consume_byte(b'[')?;
        self.depth += 1;
        if self.depth > self.max_depth {
            return Err(JsonError::MaxDepthExceeded(self.max_depth));
        }

        out.push(JsonEvent::StartArray);
        self.skip_whitespace();

        if self.peek_byte()? == b']' {
            self.consume_byte(b']')?;
            self.depth -= 1;
            out.push(JsonEvent::EndArray);
            return Ok(());
        }

        loop {
            self.skip_whitespace();
            if self.peek_byte()? == b',' {
                return Err(JsonError::TrailingComma(self.pos));
            }

            self.stream_value(out)?;

            self.skip_whitespace();
            match self.peek_byte()? {
                b',' => {
                    self.consume_byte(b',')?;
                    self.skip_whitespace();
                    if self.peek_byte()? == b']' {
                        return Err(JsonError::TrailingComma(self.pos));
                    }
                }
                b']' => {
                    self.consume_byte(b']')?;
                    break;
                }
                b => return Err(JsonError::UnexpectedChar(b as char, self.pos)),
            }
        }

        self.depth -= 1;
        out.push(JsonEvent::EndArray);
        Ok(())
    }

    fn parse_string(&mut self) -> Result<String, JsonError> {
        self.consume_byte(b'"')?;
        let mut bytes = Vec::new();

        while !self.is_eof() {
            let b = self.next_byte()?;
            match b {
                b'"' => {
                    return String::from_utf8(bytes).map_err(|_| JsonError::InvalidUnicode(self.pos));
                }
                b'\\' => {
                    let esc = self.next_byte()?;
                    match esc {
                        b'"' => bytes.push(b'"'),
                        b'\\' => bytes.push(b'\\'),
                        b'/' => bytes.push(b'/'),
                        b'b' => bytes.push(b'\x08'),
                        b'f' => bytes.push(b'\x0c'),
                        b'n' => bytes.push(b'\n'),
                        b'r' => bytes.push(b'\r'),
                        b't' => bytes.push(b'\t'),
                        b'u' => {
                            let ch = self.parse_unicode_hex()?;
                            let mut buf = [0u8; 4];
                            let s = ch.encode_utf8(&mut buf);
                            bytes.extend_from_slice(s.as_bytes());
                        }
                        _ => return Err(JsonError::InvalidEscape(self.pos - 1)),
                    }
                }
                0x00..=0x1F => return Err(JsonError::UnexpectedChar(b as char, self.pos - 1)),
                _ => bytes.push(b),
            }
        }

        Err(JsonError::UnexpectedEof)
    }

    fn parse_unicode_hex(&mut self) -> Result<char, JsonError> {
        let code = self.parse_hex4()?;
        if (0xD800..=0xDBFF).contains(&code) {
            if self.next_byte()? != b'\\' || self.next_byte()? != b'u' {
                return Err(JsonError::InvalidUnicode(self.pos));
            }
            let low = self.parse_hex4()?;
            if !(0xDC00..=0xDFFF).contains(&low) {
                return Err(JsonError::InvalidUnicode(self.pos));
            }
            let codepoint = 0x10000 + (((code - 0xD800) << 10) | (low - 0xDC00));
            char::from_u32(codepoint).ok_or(JsonError::InvalidUnicode(self.pos))
        } else {
            char::from_u32(code).ok_or(JsonError::InvalidUnicode(self.pos))
        }
    }

    fn parse_hex4(&mut self) -> Result<u32, JsonError> {
        let mut val = 0u32;
        for _ in 0..4 {
            let b = self.next_byte()?;
            let digit = match b {
                b'0'..=b'9' => (b - b'0') as u32,
                b'a'..=b'f' => (b - b'a' + 10) as u32,
                b'A'..=b'F' => (b - b'A' + 10) as u32,
                _ => return Err(JsonError::InvalidUnicode(self.pos - 1)),
            };
            val = (val << 4) | digit;
        }
        Ok(val)
    }

    fn parse_number(&mut self) -> Result<DataValue, JsonError> {
        let start = self.pos;
        let mut is_float = false;

        if self.peek_byte()? == b'-' {
            self.consume_byte(b'-')?;
        }

        let first_digit_pos = self.pos;
        if self.is_eof() {
            return Err(JsonError::UnexpectedEof);
        }

        let first_digit = self.next_byte()?;
        if !first_digit.is_ascii_digit() {
            return Err(JsonError::InvalidNumber(first_digit_pos));
        }

        if first_digit == b'0' {
            if !self.is_eof() && self.peek_byte()?.is_ascii_digit() {
                return Err(JsonError::InvalidNumber(self.pos));
            }
        } else {
            while !self.is_eof() && self.peek_byte()?.is_ascii_digit() {
                self.next_byte()?;
            }
        }

        if !self.is_eof() && self.peek_byte()? == b'.' {
            is_float = true;
            self.consume_byte(b'.' )?;
            if self.is_eof() || !self.peek_byte()?.is_ascii_digit() {
                return Err(JsonError::InvalidNumber(self.pos));
            }
            while !self.is_eof() && self.peek_byte()?.is_ascii_digit() {
                self.next_byte()?;
            }
        }

        if !self.is_eof() && (self.peek_byte()? == b'e' || self.peek_byte()? == b'E') {
            is_float = true;
            self.next_byte()?;
            if !self.is_eof() && (self.peek_byte()? == b'+' || self.peek_byte()? == b'-') {
                self.next_byte()?;
            }
            if self.is_eof() || !self.peek_byte()?.is_ascii_digit() {
                return Err(JsonError::InvalidNumber(self.pos));
            }
            while !self.is_eof() && self.peek_byte()?.is_ascii_digit() {
                self.next_byte()?;
            }
        }

        let raw_str = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|_| JsonError::InvalidNumber(start))?;

        if is_float {
            let f: f64 = raw_str.parse().map_err(|_| JsonError::InvalidNumber(start))?;
            if !f.is_finite() {
                return Err(JsonError::InvalidNumber(start));
            }
            Ok(DataValue::Number(ExactNumber::Float(f)))
        } else {
            if let Ok(i) = raw_str.parse::<i64>() {
                Ok(DataValue::Number(ExactNumber::Int(i)))
            } else if let Ok(u) = raw_str.parse::<u64>() {
                Ok(DataValue::Number(ExactNumber::UInt(u)))
            } else {
                Err(JsonError::InvalidNumber(start))
            }
        }
    }

    fn skip_whitespace(&mut self) {
        while !self.is_eof() {
            match self.input[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.input.len()
    }

    fn peek_byte(&self) -> Result<u8, JsonError> {
        if self.is_eof() {
            Err(JsonError::UnexpectedEof)
        } else {
            Ok(self.input[self.pos])
        }
    }

    fn next_byte(&mut self) -> Result<u8, JsonError> {
        if self.is_eof() {
            Err(JsonError::UnexpectedEof)
        } else {
            let b = self.input[self.pos];
            self.pos += 1;
            Ok(b)
        }
    }

    fn consume_byte(&mut self, expected: u8) -> Result<(), JsonError> {
        let b = self.next_byte()?;
        if b != expected {
            Err(JsonError::UnexpectedChar(b as char, self.pos - 1))
        } else {
            Ok(())
        }
    }

    fn consume_slice(&mut self, expected: &[u8]) -> Result<(), JsonError> {
        for &b in expected {
            self.consume_byte(b)?;
        }
        Ok(())
    }
}

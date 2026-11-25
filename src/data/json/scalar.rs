use crate::data::json::events::JsonError;
use crate::data::{DataValue, Document, ExactNumber};

pub struct JsonParser<'a> {
    input: &'a [u8],
    pos: usize,
    depth: usize,
    max_depth: usize,
    max_bytes: usize,
}

impl<'a> JsonParser<'a> {
    pub fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            pos: 0,
            depth: 0,
            max_depth: 256,
            max_bytes: 67_108_864, // 64 MiB
        }
    }

    pub fn with_limits(input: &'a [u8], max_depth: usize, max_bytes: usize) -> Self {
        Self {
            input,
            pos: 0,
            depth: 0,
            max_depth,
            max_bytes,
        }
    }

    pub fn parse_all(&mut self) -> Result<DataValue, JsonError> {
        if self.input.len() > self.max_bytes {
            return Err(JsonError::MaxBytesExceeded(self.max_bytes));
        }

        self.skip_whitespace();
        if self.is_eof() {
            return Err(JsonError::UnexpectedEof);
        }

        let val = self.parse_value()?;
        self.skip_whitespace();
        if !self.is_eof() {
            return Err(JsonError::TrailingBytes(self.pos));
        }
        Ok(val)
    }

    pub fn parse_value(&mut self) -> Result<DataValue, JsonError> {
        self.skip_whitespace();
        if self.is_eof() {
            return Err(JsonError::UnexpectedEof);
        }

        match self.peek_byte()? {
            b'{' => self.parse_object(),
            b'[' => self.parse_array(),
            b'"' => self.parse_string().map(DataValue::String),
            b't' => self.parse_true(),
            b'f' => self.parse_false(),
            b'n' => self.parse_null(),
            b'-' | b'0'..=b'9' => self.parse_number(),
            b => Err(JsonError::UnexpectedChar(b as char, self.pos)),
        }
    }

    fn parse_object(&mut self) -> Result<DataValue, JsonError> {
        self.consume_byte(b'{')?;
        self.depth += 1;
        if self.depth > self.max_depth {
            return Err(JsonError::MaxDepthExceeded(self.max_depth));
        }

        let mut doc = Document::new();
        self.skip_whitespace();

        if self.peek_byte()? == b'}' {
            self.consume_byte(b'}')?;
            self.depth -= 1;
            return Ok(DataValue::Document(doc));
        }

        loop {
            self.skip_whitespace();
            if self.peek_byte()? == b',' {
                return Err(JsonError::TrailingComma(self.pos));
            }

            let key_pos = self.pos;
            let key = self.parse_string()?;
            self.skip_whitespace();
            self.consume_byte(b':')?;
            let val = self.parse_value()?;

            if let Err(_) = doc.try_insert(key.clone(), val) {
                return Err(JsonError::DuplicateKey(key, key_pos));
            }

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
        Ok(DataValue::Document(doc))
    }

    fn parse_array(&mut self) -> Result<DataValue, JsonError> {
        self.consume_byte(b'[')?;
        self.depth += 1;
        if self.depth > self.max_depth {
            return Err(JsonError::MaxDepthExceeded(self.max_depth));
        }

        let mut arr = Vec::new();
        self.skip_whitespace();

        if self.peek_byte()? == b']' {
            self.consume_byte(b']')?;
            self.depth -= 1;
            return Ok(DataValue::Array(arr));
        }

        loop {
            self.skip_whitespace();
            if self.peek_byte()? == b',' {
                return Err(JsonError::TrailingComma(self.pos));
            }

            let val = self.parse_value()?;
            arr.push(val);

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
        Ok(DataValue::Array(arr))
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
                0x00..=0x1F => {
                    // RFC 8259 unescaped control chars are forbidden
                    return Err(JsonError::UnexpectedChar(b as char, self.pos - 1));
                }
                _ => {
                    bytes.push(b);
                }
            }
        }

        Err(JsonError::UnexpectedEof)
    }

    fn parse_unicode_hex(&mut self) -> Result<char, JsonError> {
        let code = self.parse_hex4()?;
        if (0xD800..=0xDBFF).contains(&code) {
            // High surrogate, expect \uDC00..\uDFFF
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
            // leading zero: next char cannot be digit (RFC 8259: 012 is invalid)
            if !self.is_eof() && self.peek_byte()?.is_ascii_digit() {
                return Err(JsonError::InvalidNumber(self.pos));
            }
        } else {
            while !self.is_eof() && self.peek_byte()?.is_ascii_digit() {
                self.next_byte()?;
            }
        }

        // Fractional part
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

        // Exponent part
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

    fn parse_true(&mut self) -> Result<DataValue, JsonError> {
        self.consume_slice(b"true")?;
        Ok(DataValue::Bool(true))
    }

    fn parse_false(&mut self) -> Result<DataValue, JsonError> {
        self.consume_slice(b"false")?;
        Ok(DataValue::Bool(false))
    }

    fn parse_null(&mut self) -> Result<DataValue, JsonError> {
        self.consume_slice(b"null")?;
        Ok(DataValue::Null)
    }

    fn consume_slice(&mut self, s: &[u8]) -> Result<(), JsonError> {
        for &expected in s {
            let actual = self.next_byte()?;
            if actual != expected {
                return Err(JsonError::UnexpectedChar(actual as char, self.pos - 1));
            }
        }
        Ok(())
    }

    fn consume_byte(&mut self, expected: u8) -> Result<(), JsonError> {
        let b = self.next_byte()?;
        if b != expected {
            Err(JsonError::UnexpectedChar(b as char, self.pos - 1))
        } else {
            Ok(())
        }
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

    fn skip_whitespace(&mut self) {
        while !self.is_eof() {
            match self.input[self.pos] {
                b' ' | b'\t' | b'\r' | b'\n' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.input.len()
    }
}

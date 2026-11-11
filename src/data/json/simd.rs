use crate::data::json::events::JsonError;
use crate::data::{DataValue, Document, ExactNumber};

/// CPU SIMD dispatch targets supported for JSON parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuDispatch {
    /// Automatically select the highest-performing SIMD engine supported by the host CPU.
    Auto,
    /// x86_64 AVX2 256-bit SIMD vector engine (32-byte chunks).
    Avx2,
    /// AArch64 NEON 128-bit SIMD vector engine (16-byte chunks).
    Neon,
    /// Portable 64-bit SWAR (SIMD Within A Register) engine (8-byte chunks).
    Swar,
    /// Byte-by-byte baseline scalar engine.
    Scalar,
}

impl CpuDispatch {
    /// Detect the highest-capability SIMD instruction set on the host platform.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                return CpuDispatch::Avx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return CpuDispatch::Neon;
        }
        #[allow(unreachable_code)]
        CpuDispatch::Swar
    }

    /// Check if this dispatch target is supported on the current host.
    pub fn is_supported(&self) -> bool {
        match self {
            CpuDispatch::Auto | CpuDispatch::Swar | CpuDispatch::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            CpuDispatch::Avx2 => std::is_x86_feature_detected!("avx2"),
            #[cfg(not(target_arch = "x86_64"))]
            CpuDispatch::Avx2 => false,
            #[cfg(target_arch = "aarch64")]
            CpuDispatch::Neon => true,
            #[cfg(not(target_arch = "aarch64"))]
            CpuDispatch::Neon => false,
        }
    }

    /// Resolve `Auto` or unsupported targets to an available engine on this host.
    pub fn effective(&self) -> Self {
        match self {
            CpuDispatch::Auto => Self::detect(),
            other => {
                if other.is_supported() {
                    *other
                } else {
                    CpuDispatch::Swar
                }
            }
        }
    }

    /// Return the vector chunk size in bytes for this dispatch engine.
    pub fn chunk_size(&self) -> usize {
        match self {
            CpuDispatch::Avx2 => 32,
            CpuDispatch::Neon => 16,
            CpuDispatch::Swar => 8,
            CpuDispatch::Scalar => 1,
            CpuDispatch::Auto => self.effective().chunk_size(),
        }
    }
}

/// Strict RFC 8259 JSON parser accelerated by SIMD vector instructions and CPU dispatch.
///
/// Ensures exact behavioral and error parity with `scalar::JsonParser` (the oracle).
/// Never assumes trailing buffer padding and respects page/chunk/tail boundaries.
pub struct SimdJsonParser<'a> {
    input: &'a [u8],
    pos: usize,
    depth: usize,
    max_depth: usize,
    max_bytes: usize,
    dispatch: CpuDispatch,
}

impl<'a> SimdJsonParser<'a> {
    /// Create a new SIMD parser with default limits and automatic CPU dispatch.
    pub fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            pos: 0,
            depth: 0,
            max_depth: 256,
            max_bytes: 67_108_864, // 64 MiB
            dispatch: CpuDispatch::Auto.effective(),
        }
    }

    /// Create a SIMD parser with a specific CPU dispatch target.
    pub fn with_dispatch(input: &'a [u8], dispatch: CpuDispatch) -> Self {
        Self {
            input,
            pos: 0,
            depth: 0,
            max_depth: 256,
            max_bytes: 67_108_864,
            dispatch: dispatch.effective(),
        }
    }

    /// Create a SIMD parser with custom nesting depth and byte limits.
    pub fn with_limits(input: &'a [u8], max_depth: usize, max_bytes: usize) -> Self {
        Self {
            input,
            pos: 0,
            depth: 0,
            max_depth,
            max_bytes,
            dispatch: CpuDispatch::Auto.effective(),
        }
    }

    /// Create a SIMD parser with custom limits and specific CPU dispatch.
    pub fn with_limits_and_dispatch(
        input: &'a [u8],
        max_depth: usize,
        max_bytes: usize,
        dispatch: CpuDispatch,
    ) -> Self {
        Self {
            input,
            pos: 0,
            depth: 0,
            max_depth,
            max_bytes,
            dispatch: dispatch.effective(),
        }
    }

    /// Get the resolved CPU dispatch engine used by this parser.
    pub fn dispatch(&self) -> CpuDispatch {
        self.dispatch
    }

    /// Current byte offset into the input buffer.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Parse the complete JSON document into a `DataValue`.
    ///
    /// Validates strict limits, duplicate keys, and trailing unconsumed bytes.
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

    /// Parse a single JSON value from the current position.
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

    /// Parse a JSON string using SIMD chunk-scanning to fast-forward past clean runs of bytes.
    fn parse_string(&mut self) -> Result<String, JsonError> {
        self.consume_byte(b'"')?;
        let mut bytes = Vec::new();

        while self.pos < self.input.len() {
            // Check if we can run a vector chunk scan
            let (clean_len, found_special) = match self.dispatch {
                #[cfg(target_arch = "x86_64")]
                CpuDispatch::Avx2 if self.pos + 32 <= self.input.len() => {
                    unsafe { find_string_special_avx2(self.input, self.pos) }
                }
                #[cfg(target_arch = "aarch64")]
                CpuDispatch::Neon if self.pos + 16 <= self.input.len() => {
                    unsafe { find_string_special_neon(self.input, self.pos) }
                }
                CpuDispatch::Swar if self.pos + 8 <= self.input.len() => {
                    find_string_special_swar(self.input, self.pos)
                }
                _ => (0, false),
            };

            if clean_len > 0 {
                let slice = &self.input[self.pos..self.pos + clean_len];
                bytes.extend_from_slice(slice);
                self.pos += clean_len;
            }

            if !found_special {
                if self.is_eof() {
                    return Err(JsonError::UnexpectedEof);
                }
                // Process one byte via scalar logic (for tail < CHUNK_SIZE or pure scalar dispatch)
                let b = self.input[self.pos];
                self.pos += 1;
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
            } else {
                // A special delimiter or control byte was detected at self.pos
                let b = self.input[self.pos];
                self.pos += 1;
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
                    _ => unreachable!("Special character mask only triggers on \", \\, or < 0x20"),
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
            // Leading zero: next char cannot be digit (RFC 8259 forbids e.g. 012)
            if !self.is_eof() && self.peek_byte()?.is_ascii_digit() {
                return Err(JsonError::InvalidNumber(self.pos));
            }
        } else {
            self.consume_digits();
        }

        // Fractional part
        if !self.is_eof() && self.peek_byte()? == b'.' {
            is_float = true;
            self.consume_byte(b'.')?;
            if self.is_eof() || !self.peek_byte()?.is_ascii_digit() {
                return Err(JsonError::InvalidNumber(self.pos));
            }
            self.consume_digits();
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
            self.consume_digits();
        }

        let raw_str = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|_| JsonError::InvalidNumber(start))?;

        if is_float {
            let f: f64 = raw_str.parse().map_err(|_| JsonError::InvalidNumber(start))?;
            if !f.is_finite() {
                return Err(JsonError::InvalidNumber(start));
            }
            Ok(DataValue::Number(ExactNumber::Float(f)))
        } else if let Ok(i) = raw_str.parse::<i64>() {
            Ok(DataValue::Number(ExactNumber::Int(i)))
        } else if let Ok(u) = raw_str.parse::<u64>() {
            Ok(DataValue::Number(ExactNumber::UInt(u)))
        } else {
            Err(JsonError::InvalidNumber(start))
        }
    }

    #[inline]
    fn consume_digits(&mut self) {
        while !self.is_eof() && self.input[self.pos].is_ascii_digit() {
            self.pos += 1;
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

    #[inline]
    fn consume_slice(&mut self, s: &[u8]) -> Result<(), JsonError> {
        for &expected in s {
            let actual = self.next_byte()?;
            if actual != expected {
                return Err(JsonError::UnexpectedChar(actual as char, self.pos - 1));
            }
        }
        Ok(())
    }

    #[inline]
    fn consume_byte(&mut self, expected: u8) -> Result<(), JsonError> {
        let b = self.next_byte()?;
        if b != expected {
            Err(JsonError::UnexpectedChar(b as char, self.pos - 1))
        } else {
            Ok(())
        }
    }

    #[inline]
    fn peek_byte(&self) -> Result<u8, JsonError> {
        if self.is_eof() {
            Err(JsonError::UnexpectedEof)
        } else {
            Ok(self.input[self.pos])
        }
    }

    #[inline]
    fn next_byte(&mut self) -> Result<u8, JsonError> {
        if self.is_eof() {
            Err(JsonError::UnexpectedEof)
        } else {
            let b = self.input[self.pos];
            self.pos += 1;
            Ok(b)
        }
    }

    /// Fast-forward past JSON whitespace characters using SIMD vector scanning.
    ///
    /// Checks for chunks of ' ' (0x20), '\t' (0x09), '\n' (0x0A), and '\r' (0x0D).
    /// Tail bytes shorter than vector chunk are handled strictly within buffer bounds.
    pub fn skip_whitespace(&mut self) {
        match self.dispatch {
            #[cfg(target_arch = "x86_64")]
            CpuDispatch::Avx2 => {
                unsafe {
                    self.pos = skip_whitespace_avx2(self.input, self.pos);
                }
            }
            #[cfg(target_arch = "aarch64")]
            CpuDispatch::Neon => {
                unsafe {
                    self.pos = skip_whitespace_neon(self.input, self.pos);
                }
            }
            CpuDispatch::Swar => {
                self.pos = skip_whitespace_swar(self.input, self.pos);
            }
            _ => {}
        }

        // Tail bounds check: process any remaining bytes (or scalar dispatch) safely
        while self.pos < self.input.len() {
            match self.input[self.pos] {
                b' ' | b'\t' | b'\r' | b'\n' => self.pos += 1,
                _ => break,
            }
        }
    }

    #[inline]
    fn is_eof(&self) -> bool {
        self.pos >= self.input.len()
    }
}

// ============================================================================
// SIMD Vector Kernels
// ============================================================================

// --- AArch64 NEON (16-byte chunks) ---

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn skip_whitespace_neon(input: &[u8], mut pos: usize) -> usize {
    use core::arch::aarch64::*;

    unsafe {
        let space = vdupq_n_u8(b' ');
        let tab = vdupq_n_u8(b'\t');
        let newline = vdupq_n_u8(b'\n');
        let cr = vdupq_n_u8(b'\r');

        while pos + 16 <= input.len() {
            let chunk = vld1q_u8(input.as_ptr().add(pos));
            let m_space = vceqq_u8(chunk, space);
            let m_tab = vceqq_u8(chunk, tab);
            let m_newline = vceqq_u8(chunk, newline);
            let m_cr = vceqq_u8(chunk, cr);

            let is_ws = vorrq_u8(vorrq_u8(m_space, m_tab), vorrq_u8(m_newline, m_cr));
            let non_ws = vmvnq_u8(is_ws);

            if vmaxvq_u8(non_ws) == 0 {
                pos += 16;
            } else {
                let low = vgetq_lane_u64::<0>(vreinterpretq_u64_u8(non_ws));
                if low != 0 {
                    let idx = (low.trailing_zeros() / 8) as usize;
                    return pos + idx;
                }
                let high = vgetq_lane_u64::<1>(vreinterpretq_u64_u8(non_ws));
                let idx = 8 + (high.trailing_zeros() / 8) as usize;
                return pos + idx;
            }
        }
    }
    pos
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn find_string_special_neon(input: &[u8], pos: usize) -> (usize, bool) {
    use core::arch::aarch64::*;

    unsafe {
        let quote = vdupq_n_u8(b'"');
        let backslash = vdupq_n_u8(b'\\');
        let control_limit = vdupq_n_u8(0x20);

        let chunk = vld1q_u8(input.as_ptr().add(pos));
        let m_quote = vceqq_u8(chunk, quote);
        let m_slash = vceqq_u8(chunk, backslash);
        let m_ctrl = vcltq_u8(chunk, control_limit);

        let special = vorrq_u8(vorrq_u8(m_quote, m_slash), m_ctrl);

        if vmaxvq_u8(special) == 0 {
            (16, false)
        } else {
            let low = vgetq_lane_u64::<0>(vreinterpretq_u64_u8(special));
            let idx = if low != 0 {
                (low.trailing_zeros() / 8) as usize
            } else {
                let high = vgetq_lane_u64::<1>(vreinterpretq_u64_u8(special));
                8 + (high.trailing_zeros() / 8) as usize
            };
            (idx, true)
        }
    }
}

// --- x86_64 AVX2 (32-byte chunks) ---

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn skip_whitespace_avx2(input: &[u8], mut pos: usize) -> usize {
    use core::arch::x86_64::*;

    unsafe {
        let space = _mm256_set1_epi8(b' ' as i8);
        let tab = _mm256_set1_epi8(b'\t' as i8);
        let newline = _mm256_set1_epi8(b'\n' as i8);
        let cr = _mm256_set1_epi8(b'\r' as i8);

        while pos + 32 <= input.len() {
            let chunk = _mm256_loadu_si256(input.as_ptr().add(pos) as *const __m256i);
            let m_space = _mm256_cmpeq_epi8(chunk, space);
            let m_tab = _mm256_cmpeq_epi8(chunk, tab);
            let m_newline = _mm256_cmpeq_epi8(chunk, newline);
            let m_cr = _mm256_cmpeq_epi8(chunk, cr);

            let is_ws = _mm256_or_si256(_mm256_or_si256(m_space, m_tab), _mm256_or_si256(m_newline, m_cr));
            let mask = _mm256_movemask_epi8(is_ws) as u32;
            let non_ws_mask = !mask;

            if non_ws_mask == 0 {
                pos += 32;
            } else {
                let idx = non_ws_mask.trailing_zeros() as usize;
                return pos + idx;
            }
        }
    }
    pos
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_string_special_avx2(input: &[u8], pos: usize) -> (usize, bool) {
    use core::arch::x86_64::*;

    unsafe {
        let quote = _mm256_set1_epi8(b'"' as i8);
        let backslash = _mm256_set1_epi8(b'\\' as i8);
        let ctrl_sub = _mm256_set1_epi8(0x1F);
        let zero = _mm256_setzero_si256();

        let chunk = _mm256_loadu_si256(input.as_ptr().add(pos) as *const __m256i);
        let m_quote = _mm256_cmpeq_epi8(chunk, quote);
        let m_slash = _mm256_cmpeq_epi8(chunk, backslash);
        // Unsigned saturating subtract: x - 0x1F saturates to 0 iff x <= 0x1F (< 0x20)
        let m_ctrl = _mm256_cmpeq_epi8(_mm256_subs_epu8(chunk, ctrl_sub), zero);

        let special = _mm256_or_si256(_mm256_or_si256(m_quote, m_slash), m_ctrl);
        let mask = _mm256_movemask_epi8(special) as u32;

        if mask == 0 {
            (32, false)
        } else {
            (mask.trailing_zeros() as usize, true)
        }
    }
}

// --- Portable 64-bit SWAR (8-byte chunks) ---

#[inline]
fn swar_has_byte(chunk: u64, val: u8) -> u64 {
    let diff = chunk ^ (0x0101_0101_0101_0101u64 * (val as u64));
    diff.wrapping_sub(0x0101_0101_0101_0101u64) & !diff & 0x8080_8080_8080_8080u64
}

#[inline]
fn skip_whitespace_swar(input: &[u8], mut pos: usize) -> usize {
    while pos + 8 <= input.len() {
        let chunk = u64::from_le_bytes(input[pos..pos + 8].try_into().unwrap());
        let ws = swar_has_byte(chunk, b' ')
            | swar_has_byte(chunk, b'\t')
            | swar_has_byte(chunk, b'\n')
            | swar_has_byte(chunk, b'\r');
        let non_ws = (!ws) & 0x8080_8080_8080_8080u64;

        if non_ws == 0 {
            pos += 8;
        } else {
            let idx = (non_ws.trailing_zeros() / 8) as usize;
            return pos + idx;
        }
    }
    pos
}

#[inline]
fn find_string_special_swar(input: &[u8], pos: usize) -> (usize, bool) {
    let chunk = u64::from_le_bytes(input[pos..pos + 8].try_into().unwrap());
    let m_quote = swar_has_byte(chunk, b'"');
    let m_slash = swar_has_byte(chunk, b'\\');

    // For any byte x in 0..=255, x < 0x20 iff bits 7, 6, and 5 are all 0 ((x & 0xE0) == 0).
    // Bits 7, 6, 5 of each byte are isolated and combined into bit 7.
    let bits567 = chunk & 0xE0E0_E0E0_E0E0_E0E0u64;
    let any_set = (bits567 | (bits567 << 1) | (bits567 << 2)) & 0x8080_8080_8080_8080u64;
    let m_ctrl = (!any_set) & 0x8080_8080_8080_8080u64;

    let special = m_quote | m_slash | m_ctrl;
    if special == 0 {
        (8, false)
    } else {
        ((special.trailing_zeros() / 8) as usize, true)
    }
}

// ============================================================================
// Public Facade Functions
// ============================================================================

/// Parse a complete JSON string into a `DataValue` using SIMD acceleration.
pub fn from_str_simd(s: &str) -> Result<DataValue, JsonError> {
    let mut parser = SimdJsonParser::new(s.as_bytes());
    parser.parse_all()
}

/// Parse a complete JSON string with an explicitly specified CPU dispatch engine.
pub fn from_str_simd_with_dispatch(s: &str, dispatch: CpuDispatch) -> Result<DataValue, JsonError> {
    let mut parser = SimdJsonParser::with_dispatch(s.as_bytes(), dispatch);
    parser.parse_all()
}

/// Parse JSON bytes with custom limits using SIMD acceleration.
pub fn from_slice_simd_with_limits(
    bytes: &[u8],
    max_depth: usize,
    max_bytes: usize,
) -> Result<DataValue, JsonError> {
    let mut parser = SimdJsonParser::with_limits(bytes, max_depth, max_bytes);
    parser.parse_all()
}

/// Parse JSON bytes with custom limits and specific CPU dispatch.
pub fn from_slice_simd_with_limits_and_dispatch(
    bytes: &[u8],
    max_depth: usize,
    max_bytes: usize,
    dispatch: CpuDispatch,
) -> Result<DataValue, JsonError> {
    let mut parser = SimdJsonParser::with_limits_and_dispatch(bytes, max_depth, max_bytes, dispatch);
    parser.parse_all()
}

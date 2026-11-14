//! High-performance MessagePack codec (RFC §7.2 binary codec).
//!
//! Provides strict MessagePack serialization and deserialization for Lust `DataValue` and `Document`.
//! Features bounded decoding, integer width specialization, nominal BSON/Extended types
//! (ObjectId, DateTimeUtc, Decimal128) via standard and custom extension tags,
//! and Gate-SCHEMA-01 strict duplicate key rejection.

use crate::data::{DataValue, DateTimeUtc, Decimal128, Document, ExactNumber, ObjectId};
use std::fmt;
use std::io::{self, Read};

pub const DEFAULT_MAX_DEPTH: usize = 128;
pub const DEFAULT_MAX_ELEMENTS: usize = 1_000_000;
pub const DEFAULT_MAX_BYTES: usize = 67_108_864; // 64 MB

pub const EXT_TYPE_DECIMAL128: i8 = 0x01;
pub const EXT_TYPE_OBJECT_ID: i8 = 0x02;
pub const EXT_TYPE_TIMESTAMP: i8 = -1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MsgPackError {
    UnexpectedEof,
    InvalidMarker(u8),
    UnknownExtensionType(i8),
    MalformedSuffix,
    MaxDepthExceeded(usize),
    MaxElementsExceeded(usize),
    MaxBytesExceeded(usize),
    DuplicateKey(String),
    InvalidUtf8(String),
    ExpectedDocument,
    IoError(String),
}

impl fmt::Display for MsgPackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MsgPackError::UnexpectedEof => write!(f, "Unexpected end of MessagePack input"),
            MsgPackError::InvalidMarker(m) => write!(f, "Invalid MessagePack marker byte: 0x{:02x}", m),
            MsgPackError::UnknownExtensionType(t) => write!(f, "Unknown/unsupported MessagePack extension type: {}", t),
            MsgPackError::MalformedSuffix => write!(f, "Malformed trailing data after MessagePack payload"),
            MsgPackError::MaxDepthExceeded(d) => write!(f, "MessagePack recursion depth {} exceeded limit {}", d, DEFAULT_MAX_DEPTH),
            MsgPackError::MaxElementsExceeded(e) => write!(f, "MessagePack element count {} exceeded limit {}", e, DEFAULT_MAX_ELEMENTS),
            MsgPackError::MaxBytesExceeded(b) => write!(f, "MessagePack byte length {} exceeded limit {}", b, DEFAULT_MAX_BYTES),
            MsgPackError::DuplicateKey(k) => write!(f, "Duplicate key '{}' in MessagePack map (Gate-SCHEMA-01)", k),
            MsgPackError::InvalidUtf8(e) => write!(f, "Invalid UTF-8 string in MessagePack: {}", e),
            MsgPackError::ExpectedDocument => write!(f, "Expected MessagePack map/document at root"),
            MsgPackError::IoError(e) => write!(f, "MessagePack I/O error: {}", e),
        }
    }
}

impl std::error::Error for MsgPackError {}

impl From<io::Error> for MsgPackError {
    fn from(err: io::Error) -> Self {
        MsgPackError::IoError(err.to_string())
    }
}

/// Encode a `DataValue` into MessagePack bytes.
pub fn to_vec(value: &DataValue) -> Vec<u8> {
    let mut buf = Vec::new();
    encode_value(value, &mut buf);
    buf
}

/// Encode a `Document` into MessagePack bytes.
pub fn to_vec_document(doc: &Document) -> Vec<u8> {
    let mut buf = Vec::new();
    encode_document(doc, &mut buf);
    buf
}

/// Decode MessagePack bytes into a `DataValue`.
pub fn from_slice(bytes: &[u8]) -> Result<DataValue, MsgPackError> {
    if bytes.is_empty() {
        return Err(MsgPackError::UnexpectedEof);
    }
    if bytes.len() > DEFAULT_MAX_BYTES {
        return Err(MsgPackError::MaxBytesExceeded(bytes.len()));
    }
    let mut cursor = 0;
    let mut total_elements = 0;
    let val = decode_value(bytes, &mut cursor, 0, &mut total_elements)?;
    if cursor < bytes.len() {
        return Err(MsgPackError::MalformedSuffix);
    }
    Ok(val)
}

/// Decode MessagePack bytes into a `Document`.
pub fn from_slice_document(bytes: &[u8]) -> Result<Document, MsgPackError> {
    let val = from_slice(bytes)?;
    match val {
        DataValue::Document(doc) => Ok(doc),
        _ => Err(MsgPackError::ExpectedDocument),
    }
}

/// Decode MessagePack from any `Read` stream.
pub fn from_reader<R: Read>(reader: R) -> Result<DataValue, MsgPackError> {
    let mut bytes = Vec::new();
    reader.take(DEFAULT_MAX_BYTES as u64 + 1).read_to_end(&mut bytes)?;
    from_slice(&bytes)
}

/// Decode MessagePack from any `Read` stream into a `Document`.
pub fn from_reader_document<R: Read>(reader: R) -> Result<Document, MsgPackError> {
    let val = from_reader(reader)?;
    match val {
        DataValue::Document(doc) => Ok(doc),
        _ => Err(MsgPackError::ExpectedDocument),
    }
}

// ============================================================================
// Encoder Implementation
// ============================================================================

fn encode_value(val: &DataValue, buf: &mut Vec<u8>) {
    match val {
        DataValue::Null => buf.push(0xc0),
        DataValue::Bool(false) => buf.push(0xc2),
        DataValue::Bool(true) => buf.push(0xc3),
        DataValue::Number(num) => encode_number(num, buf),
        DataValue::String(s) => encode_str(s, buf),
        DataValue::Binary(bin) => encode_bin(bin, buf),
        DataValue::Array(arr) => {
            encode_array_header(arr.len(), buf);
            for elem in arr {
                encode_value(elem, buf);
            }
        }
        DataValue::Document(doc) => encode_document(doc, buf),
        DataValue::DateTimeUtc(dt) => {
            // Encode as timestamp extension (type -1) with 8-byte millis
            let millis = dt.millis();
            buf.push(0xd7); // fixext 8
            buf.push(EXT_TYPE_TIMESTAMP as u8);
            buf.extend_from_slice(&millis.to_be_bytes());
        }
        DataValue::ObjectId(oid) => {
            // Ext 8: 0xc7, len=12, type=0x02
            buf.push(0xc7);
            buf.push(12);
            buf.push(EXT_TYPE_OBJECT_ID as u8);
            buf.extend_from_slice(oid.bytes());
        }
    }
}

fn encode_document(doc: &Document, buf: &mut Vec<u8>) {
    encode_map_header(doc.len(), buf);
    for (k, v) in doc.iter() {
        encode_str(k, buf);
        encode_value(v, buf);
    }
}

fn encode_number(num: &ExactNumber, buf: &mut Vec<u8>) {
    match num {
        ExactNumber::Int(i) => encode_i64(*i, buf),
        ExactNumber::UInt(u) => encode_u64(*u, buf),
        ExactNumber::Float(f) => {
            buf.push(0xcb); // float 64
            buf.extend_from_slice(&f.to_be_bytes());
        }
        ExactNumber::Decimal128(d128) => {
            buf.push(0xd8); // fixext 16
            buf.push(EXT_TYPE_DECIMAL128 as u8);
            buf.extend_from_slice(d128.raw_bytes());
        }
    }
}

fn encode_i64(val: i64, buf: &mut Vec<u8>) {
    if (0..=127).contains(&val) {
        buf.push(val as u8);
    } else if (-32..= -1).contains(&val) {
        buf.push(val as i8 as u8);
    } else if val >= 0 {
        encode_u64(val as u64, buf);
    } else if val >= i8::MIN as i64 {
        buf.push(0xd0);
        buf.push(val as i8 as u8);
    } else if val >= i16::MIN as i64 {
        buf.push(0xd1);
        buf.extend_from_slice(&(val as i16).to_be_bytes());
    } else if val >= i32::MIN as i64 {
        buf.push(0xd2);
        buf.extend_from_slice(&(val as i32).to_be_bytes());
    } else {
        buf.push(0xd3);
        buf.extend_from_slice(&val.to_be_bytes());
    }
}

fn encode_u64(val: u64, buf: &mut Vec<u8>) {
    if val <= 127 {
        buf.push(val as u8);
    } else if val <= u8::MAX as u64 {
        buf.push(0xcc);
        buf.push(val as u8);
    } else if val <= u16::MAX as u64 {
        buf.push(0xcd);
        buf.extend_from_slice(&(val as u16).to_be_bytes());
    } else if val <= u32::MAX as u64 {
        buf.push(0xce);
        buf.extend_from_slice(&(val as u32).to_be_bytes());
    } else {
        buf.push(0xcf);
        buf.extend_from_slice(&val.to_be_bytes());
    }
}

fn encode_bin(bin: &[u8], buf: &mut Vec<u8>) {
    let len = bin.len();
    if len <= u8::MAX as usize {
        buf.push(0xc4);
        buf.push(len as u8);
    } else if len <= u16::MAX as usize {
        buf.push(0xc5);
        buf.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        buf.push(0xc6);
        buf.extend_from_slice(&(len as u32).to_be_bytes());
    }
    buf.extend_from_slice(bin);
}

fn encode_str(s: &str, buf: &mut Vec<u8>) {
    let len = s.len();
    if len <= 31 {
        buf.push(0xa0 | (len as u8));
    } else if len <= u8::MAX as usize {
        buf.push(0xd9);
        buf.push(len as u8);
    } else if len <= u16::MAX as usize {
        buf.push(0xda);
        buf.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        buf.push(0xdb);
        buf.extend_from_slice(&(len as u32).to_be_bytes());
    }
    buf.extend_from_slice(s.as_bytes());
}

fn encode_array_header(len: usize, buf: &mut Vec<u8>) {
    if len <= 15 {
        buf.push(0x90 | (len as u8));
    } else if len <= u16::MAX as usize {
        buf.push(0xdc);
        buf.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        buf.push(0xdd);
        buf.extend_from_slice(&(len as u32).to_be_bytes());
    }
}

fn encode_map_header(len: usize, buf: &mut Vec<u8>) {
    if len <= 15 {
        buf.push(0x80 | (len as u8));
    } else if len <= u16::MAX as usize {
        buf.push(0xde);
        buf.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        buf.push(0xdf);
        buf.extend_from_slice(&(len as u32).to_be_bytes());
    }
}

// ============================================================================
// Decoder Implementation
// ============================================================================

fn decode_value(
    bytes: &[u8],
    cursor: &mut usize,
    depth: usize,
    total_elements: &mut usize,
) -> Result<DataValue, MsgPackError> {
    if depth > DEFAULT_MAX_DEPTH {
        return Err(MsgPackError::MaxDepthExceeded(depth));
    }
    if *cursor >= bytes.len() {
        return Err(MsgPackError::UnexpectedEof);
    }

    let marker = bytes[*cursor];
    *cursor += 1;

    // Positive fixint: 0x00 ..= 0x7f
    if marker <= 0x7f {
        return Ok(DataValue::Number(ExactNumber::Int(marker as i64)));
    }

    // Fixmap: 0x80 ..= 0x8f
    if (0x80..=0x8f).contains(&marker) {
        let len = (marker & 0x0f) as usize;
        return decode_map(bytes, cursor, len, depth + 1, total_elements);
    }

    // Fixarray: 0x90 ..= 0x9f
    if (0x90..=0x9f).contains(&marker) {
        let len = (marker & 0x0f) as usize;
        return decode_array(bytes, cursor, len, depth + 1, total_elements);
    }

    // Fixstr: 0xa0 ..= 0xbf
    if (0xa0..=0xbf).contains(&marker) {
        let len = (marker & 0x1f) as usize;
        return decode_str_content(bytes, cursor, len);
    }

    // Negative fixint: 0xe0 ..= 0xff
    if marker >= 0xe0 {
        let val = (marker as i8) as i64;
        return Ok(DataValue::Number(ExactNumber::Int(val)));
    }

    match marker {
        0xc0 => Ok(DataValue::Null),
        0xc1 => Err(MsgPackError::InvalidMarker(0xc1)),
        0xc2 => Ok(DataValue::Bool(false)),
        0xc3 => Ok(DataValue::Bool(true)),
        0xc4 => {
            let len = read_u8(bytes, cursor)? as usize;
            let bin = read_bytes(bytes, cursor, len)?;
            // Binary string or byte array representation
            Ok(DataValue::String(String::from_utf8_lossy(bin).to_string()))
        }
        0xc5 => {
            let len = read_u16(bytes, cursor)? as usize;
            let bin = read_bytes(bytes, cursor, len)?;
            Ok(DataValue::String(String::from_utf8_lossy(bin).to_string()))
        }
        0xc6 => {
            let len = read_u32(bytes, cursor)? as usize;
            let bin = read_bytes(bytes, cursor, len)?;
            Ok(DataValue::String(String::from_utf8_lossy(bin).to_string()))
        }
        0xc7 => {
            let len = read_u8(bytes, cursor)? as usize;
            let ext_type = read_i8(bytes, cursor)?;
            decode_ext(bytes, cursor, ext_type, len)
        }
        0xc8 => {
            let len = read_u16(bytes, cursor)? as usize;
            let ext_type = read_i8(bytes, cursor)?;
            decode_ext(bytes, cursor, ext_type, len)
        }
        0xc9 => {
            let len = read_u32(bytes, cursor)? as usize;
            let ext_type = read_i8(bytes, cursor)?;
            decode_ext(bytes, cursor, ext_type, len)
        }
        0xca => {
            let f = read_f32(bytes, cursor)?;
            Ok(DataValue::Number(ExactNumber::Float(f as f64)))
        }
        0xcb => {
            let f = read_f64(bytes, cursor)?;
            Ok(DataValue::Number(ExactNumber::Float(f)))
        }
        0xcc => {
            let u = read_u8(bytes, cursor)?;
            Ok(DataValue::Number(ExactNumber::Int(u as i64)))
        }
        0xcd => {
            let u = read_u16(bytes, cursor)?;
            Ok(DataValue::Number(ExactNumber::Int(u as i64)))
        }
        0xce => {
            let u = read_u32(bytes, cursor)?;
            Ok(DataValue::Number(ExactNumber::Int(u as i64)))
        }
        0xcf => {
            let u = read_u64(bytes, cursor)?;
            if u <= i64::MAX as u64 {
                Ok(DataValue::Number(ExactNumber::Int(u as i64)))
            } else {
                Ok(DataValue::Number(ExactNumber::UInt(u)))
            }
        }
        0xd0 => {
            let i = read_i8(bytes, cursor)?;
            Ok(DataValue::Number(ExactNumber::Int(i as i64)))
        }
        0xd1 => {
            let i = read_i16(bytes, cursor)?;
            Ok(DataValue::Number(ExactNumber::Int(i as i64)))
        }
        0xd2 => {
            let i = read_i32(bytes, cursor)?;
            Ok(DataValue::Number(ExactNumber::Int(i as i64)))
        }
        0xd3 => {
            let i = read_i64(bytes, cursor)?;
            Ok(DataValue::Number(ExactNumber::Int(i)))
        }
        0xd4 => {
            let ext_type = read_i8(bytes, cursor)?;
            decode_ext(bytes, cursor, ext_type, 1)
        }
        0xd5 => {
            let ext_type = read_i8(bytes, cursor)?;
            decode_ext(bytes, cursor, ext_type, 2)
        }
        0xd6 => {
            let ext_type = read_i8(bytes, cursor)?;
            decode_ext(bytes, cursor, ext_type, 4)
        }
        0xd7 => {
            let ext_type = read_i8(bytes, cursor)?;
            decode_ext(bytes, cursor, ext_type, 8)
        }
        0xd8 => {
            let ext_type = read_i8(bytes, cursor)?;
            decode_ext(bytes, cursor, ext_type, 16)
        }
        0xd9 => {
            let len = read_u8(bytes, cursor)? as usize;
            decode_str_content(bytes, cursor, len)
        }
        0xda => {
            let len = read_u16(bytes, cursor)? as usize;
            decode_str_content(bytes, cursor, len)
        }
        0xdb => {
            let len = read_u32(bytes, cursor)? as usize;
            decode_str_content(bytes, cursor, len)
        }
        0xdc => {
            let len = read_u16(bytes, cursor)? as usize;
            decode_array(bytes, cursor, len, depth + 1, total_elements)
        }
        0xdd => {
            let len = read_u32(bytes, cursor)? as usize;
            decode_array(bytes, cursor, len, depth + 1, total_elements)
        }
        0xde => {
            let len = read_u16(bytes, cursor)? as usize;
            decode_map(bytes, cursor, len, depth + 1, total_elements)
        }
        0xdf => {
            let len = read_u32(bytes, cursor)? as usize;
            decode_map(bytes, cursor, len, depth + 1, total_elements)
        }
        _ => Err(MsgPackError::InvalidMarker(marker)),
    }
}

fn decode_str_content(bytes: &[u8], cursor: &mut usize, len: usize) -> Result<DataValue, MsgPackError> {
    let slice = read_bytes(bytes, cursor, len)?;
    let s = std::str::from_utf8(slice).map_err(|e| MsgPackError::InvalidUtf8(e.to_string()))?;
    Ok(DataValue::String(s.to_string()))
}

fn decode_str_raw(bytes: &[u8], cursor: &mut usize) -> Result<String, MsgPackError> {
    if *cursor >= bytes.len() {
        return Err(MsgPackError::UnexpectedEof);
    }
    let marker = bytes[*cursor];
    *cursor += 1;

    let len = if (0xa0..=0xbf).contains(&marker) {
        (marker & 0x1f) as usize
    } else if marker == 0xd9 {
        read_u8(bytes, cursor)? as usize
    } else if marker == 0xda {
        read_u16(bytes, cursor)? as usize
    } else if marker == 0xdb {
        read_u32(bytes, cursor)? as usize
    } else {
        return Err(MsgPackError::InvalidMarker(marker));
    };

    let slice = read_bytes(bytes, cursor, len)?;
    let s = std::str::from_utf8(slice).map_err(|e| MsgPackError::InvalidUtf8(e.to_string()))?;
    Ok(s.to_string())
}

fn decode_array(
    bytes: &[u8],
    cursor: &mut usize,
    len: usize,
    depth: usize,
    total_elements: &mut usize,
) -> Result<DataValue, MsgPackError> {
    let projected = total_elements
        .checked_add(len)
        .ok_or(MsgPackError::MaxElementsExceeded(DEFAULT_MAX_ELEMENTS + 1))?;
    if projected > DEFAULT_MAX_ELEMENTS {
        return Err(MsgPackError::MaxElementsExceeded(projected));
    }
    let remaining_bytes = bytes.len().saturating_sub(*cursor);
    if len > remaining_bytes {
        return Err(MsgPackError::UnexpectedEof);
    }
    let initial_cap = len.min(1024).min(remaining_bytes);
    let mut elements = Vec::with_capacity(initial_cap);
    for _ in 0..len {
        *total_elements = total_elements
            .checked_add(1)
            .ok_or(MsgPackError::MaxElementsExceeded(DEFAULT_MAX_ELEMENTS + 1))?;
        if *total_elements > DEFAULT_MAX_ELEMENTS {
            return Err(MsgPackError::MaxElementsExceeded(*total_elements));
        }
        let elem = decode_value(bytes, cursor, depth, total_elements)?;
        elements.push(elem);
    }
    Ok(DataValue::Array(elements))
}

fn decode_map(
    bytes: &[u8],
    cursor: &mut usize,
    len: usize,
    depth: usize,
    total_elements: &mut usize,
) -> Result<DataValue, MsgPackError> {
    let map_elements = len
        .checked_mul(2)
        .ok_or(MsgPackError::MaxElementsExceeded(DEFAULT_MAX_ELEMENTS + 1))?;
    let projected = total_elements
        .checked_add(map_elements)
        .ok_or(MsgPackError::MaxElementsExceeded(DEFAULT_MAX_ELEMENTS + 1))?;
    if projected > DEFAULT_MAX_ELEMENTS {
        return Err(MsgPackError::MaxElementsExceeded(projected));
    }
    let remaining_bytes = bytes.len().saturating_sub(*cursor);
    if len > remaining_bytes / 2 {
        return Err(MsgPackError::UnexpectedEof);
    }
    let mut doc = Document::new();
    for _ in 0..len {
        *total_elements = total_elements
            .checked_add(2)
            .ok_or(MsgPackError::MaxElementsExceeded(DEFAULT_MAX_ELEMENTS + 1))?;
        if *total_elements > DEFAULT_MAX_ELEMENTS {
            return Err(MsgPackError::MaxElementsExceeded(*total_elements));
        }
        let key = decode_str_raw(bytes, cursor)?;
        let val = decode_value(bytes, cursor, depth, total_elements)?;
        // Gate-SCHEMA-01: duplicate keys strictly fail closed
        doc.try_insert(key.clone(), val)
            .map_err(|_| MsgPackError::DuplicateKey(key))?;
    }
    Ok(DataValue::Document(doc))
}

fn decode_ext(bytes: &[u8], cursor: &mut usize, ext_type: i8, len: usize) -> Result<DataValue, MsgPackError> {
    let payload = read_bytes(bytes, cursor, len)?;
    match ext_type {
        EXT_TYPE_TIMESTAMP => {
            if len == 4 {
                let secs = u32::from_be_bytes(payload.try_into().unwrap());
                Ok(DataValue::DateTimeUtc(DateTimeUtc::from_millis(secs as i64 * 1000)))
            } else if len == 8 {
                let val = i64::from_be_bytes(payload.try_into().unwrap());
                Ok(DataValue::DateTimeUtc(DateTimeUtc::from_millis(val)))
            } else if len == 12 {
                let millis = i64::from_be_bytes(payload[4..12].try_into().unwrap());
                Ok(DataValue::DateTimeUtc(DateTimeUtc::from_millis(millis)))
            } else {
                Err(MsgPackError::InvalidMarker(ext_type as u8))
            }
        }
        EXT_TYPE_DECIMAL128 => {
            if len == 16 {
                let d128 = Decimal128::new(payload.try_into().unwrap());
                Ok(DataValue::Number(ExactNumber::Decimal128(d128)))
            } else {
                Err(MsgPackError::InvalidMarker(ext_type as u8))
            }
        }
        EXT_TYPE_OBJECT_ID => {
            if len == 12 {
                let oid = ObjectId::new(payload.try_into().unwrap());
                Ok(DataValue::ObjectId(oid))
            } else {
                Err(MsgPackError::InvalidMarker(ext_type as u8))
            }
        }
        _ => Err(MsgPackError::UnknownExtensionType(ext_type)),
    }
}

// Low-level byte helpers
fn read_bytes<'a>(bytes: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8], MsgPackError> {
    if *cursor + len > bytes.len() {
        return Err(MsgPackError::UnexpectedEof);
    }
    let slice = &bytes[*cursor..*cursor + len];
    *cursor += len;
    Ok(slice)
}

fn read_u8(bytes: &[u8], cursor: &mut usize) -> Result<u8, MsgPackError> {
    let b = read_bytes(bytes, cursor, 1)?;
    Ok(b[0])
}

fn read_i8(bytes: &[u8], cursor: &mut usize) -> Result<i8, MsgPackError> {
    let b = read_bytes(bytes, cursor, 1)?;
    Ok(b[0] as i8)
}

fn read_u16(bytes: &[u8], cursor: &mut usize) -> Result<u16, MsgPackError> {
    let b = read_bytes(bytes, cursor, 2)?;
    Ok(u16::from_be_bytes(b.try_into().unwrap()))
}

fn read_i16(bytes: &[u8], cursor: &mut usize) -> Result<i16, MsgPackError> {
    let b = read_bytes(bytes, cursor, 2)?;
    Ok(i16::from_be_bytes(b.try_into().unwrap()))
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, MsgPackError> {
    let b = read_bytes(bytes, cursor, 4)?;
    Ok(u32::from_be_bytes(b.try_into().unwrap()))
}

fn read_i32(bytes: &[u8], cursor: &mut usize) -> Result<i32, MsgPackError> {
    let b = read_bytes(bytes, cursor, 4)?;
    Ok(i32::from_be_bytes(b.try_into().unwrap()))
}

fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, MsgPackError> {
    let b = read_bytes(bytes, cursor, 8)?;
    Ok(u64::from_be_bytes(b.try_into().unwrap()))
}

fn read_i64(bytes: &[u8], cursor: &mut usize) -> Result<i64, MsgPackError> {
    let b = read_bytes(bytes, cursor, 8)?;
    Ok(i64::from_be_bytes(b.try_into().unwrap()))
}

fn read_f32(bytes: &[u8], cursor: &mut usize) -> Result<f32, MsgPackError> {
    let b = read_bytes(bytes, cursor, 4)?;
    Ok(f32::from_be_bytes(b.try_into().unwrap()))
}

fn read_f64(bytes: &[u8], cursor: &mut usize) -> Result<f64, MsgPackError> {
    let b = read_bytes(bytes, cursor, 8)?;
    Ok(f64::from_be_bytes(b.try_into().unwrap()))
}

#[allow(dead_code)]
mod hex {
    pub fn encode(data: &[u8]) -> String {
        let mut s = String::with_capacity(data.len() * 2);
        for &b in data {
            s.push_str(&format!("{:02x}", b));
        }
        s
    }
}

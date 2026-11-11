use crate::data::{DataValue, DateTimeUtc, Decimal128, Document, ExactNumber, ObjectId};
use std::fmt;

pub const MAX_BSON_BYTES: usize = 16_777_216; // 16 MiB

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BsonError {
    UnexpectedEof,
    InvalidDocumentLength(i32),
    MaxBytesExceeded(usize),
    MissingNullTerminator,
    InvalidUtf8,
    DuplicateKey(String),
    UnsupportedElementType(u8),
    MalformedSuffix,
    Io(String),
}

impl fmt::Display for BsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BsonError::UnexpectedEof => write!(f, "Unexpected end of BSON data"),
            BsonError::InvalidDocumentLength(len) => write!(f, "Invalid BSON document length header: {}", len),
            BsonError::MaxBytesExceeded(len) => write!(f, "BSON document length {} exceeds maximum {}", len, MAX_BSON_BYTES),
            BsonError::MissingNullTerminator => write!(f, "Missing BSON null terminator"),
            BsonError::InvalidUtf8 => write!(f, "Invalid UTF-8 in BSON string or key"),
            BsonError::DuplicateKey(k) => write!(f, "Duplicate key '{}' rejected in BSON document", k),
            BsonError::UnsupportedElementType(t) => write!(f, "Unsupported BSON element type byte: 0x{:02X}", t),
            BsonError::MalformedSuffix => write!(f, "Trailing unconsumed or malformed bytes in BSON buffer"),
            BsonError::Io(msg) => write!(f, "BSON I/O error: {}", msg),
        }
    }
}

impl std::error::Error for BsonError {}

pub struct BsonDecoder<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BsonDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn decode_document(&mut self) -> Result<Document, BsonError> {
        self.decode_document_bounded(self.data.len())
    }

    pub fn decode_document_bounded(&mut self, max_end: usize) -> Result<Document, BsonError> {
        if self.data.len() < 5 {
            return Err(BsonError::UnexpectedEof);
        }
        if self.data.len() > MAX_BSON_BYTES {
            return Err(BsonError::MaxBytesExceeded(self.data.len()));
        }

        let doc_start = self.pos;
        let doc_len = self.read_i32()? as usize;
        let doc_end = doc_start
            .checked_add(doc_len)
            .ok_or(BsonError::InvalidDocumentLength(doc_len as i32))?;

        if doc_len < 5 || doc_end > max_end {
            return Err(BsonError::InvalidDocumentLength(doc_len as i32));
        }

        if self.data[doc_end - 1] != 0x00 {
            return Err(BsonError::MissingNullTerminator);
        }

        let mut doc = Document::new();

        while self.pos < doc_end - 1 {
            let elem_type = self.read_u8()?;
            let key = self.read_cstring()?;

            let value = match elem_type {
                0x01 => {
                    // Float 64-bit
                    let f = self.read_f64()?;
                    DataValue::Number(ExactNumber::Float(f))
                }
                0x02 => {
                    // UTF-8 string: int32 length + bytes + 0x00
                    let str_len = self.read_i32()?;
                    if str_len < 0 {
                        return Err(BsonError::InvalidDocumentLength(str_len));
                    }
                    if str_len == 0 {
                        return Err(BsonError::UnexpectedEof);
                    }
                    let str_len = str_len as usize;
                    let s_bytes = self.read_slice(str_len - 1)?;
                    self.consume_byte(0x00)?;
                    let s = std::str::from_utf8(s_bytes).map_err(|_| BsonError::InvalidUtf8)?;
                    DataValue::String(s.to_string())
                }
                0x03 => {
                    // Embedded document strictly bounded by enclosing document
                    let sub_doc = self.decode_document_bounded(doc_end - 1)?;
                    DataValue::Document(sub_doc)
                }
                0x04 => {
                    // Array (sub-document with keys "0", "1", ...) strictly bounded
                    let sub_doc = self.decode_document_bounded(doc_end - 1)?;
                    let mut items = Vec::with_capacity(sub_doc.len());
                    for (_, v) in sub_doc.into_iter() {
                        items.push(v);
                    }
                    DataValue::Array(items)
                }
                0x05 => {
                    // Binary: int32 len + subtype (1 byte) + data
                    let bin_len = self.read_i32()?;
                    if bin_len < 0 {
                        return Err(BsonError::InvalidDocumentLength(bin_len));
                    }
                    let bin_len = bin_len as usize;
                    let _subtype = self.read_u8()?;
                    let bin_bytes = self.read_slice(bin_len)?;
                    DataValue::Binary(bin_bytes.to_vec())
                }
                0x07 => {
                    // ObjectId 12 bytes
                    let raw = self.read_slice(12)?;
                    let mut bytes = [0u8; 12];
                    bytes.copy_from_slice(raw);
                    DataValue::ObjectId(ObjectId(bytes))
                }
                0x08 => {
                    // Boolean 1 byte
                    let b = self.read_u8()?;
                    DataValue::Bool(b != 0)
                }
                0x09 => {
                    // UTC datetime int64
                    let ms = self.read_i64()?;
                    DataValue::DateTimeUtc(DateTimeUtc(ms))
                }
                0x0A => {
                    // Null
                    DataValue::Null
                }
                0x10 => {
                    // 32-bit integer
                    let i = self.read_i32()?;
                    DataValue::Number(ExactNumber::Int(i as i64))
                }
                0x11 => {
                    // 64-bit unsigned integer (timestamp / uint64)
                    let u = self.read_u64()?;
                    DataValue::Number(ExactNumber::UInt(u))
                }
                0x12 => {
                    // 64-bit integer
                    let i = self.read_i64()?;
                    DataValue::Number(ExactNumber::Int(i))
                }
                0x13 => {
                    // 128-bit decimal
                    let raw = self.read_slice(16)?;
                    let mut bytes = [0u8; 16];
                    bytes.copy_from_slice(raw);
                    DataValue::Number(ExactNumber::Decimal128(Decimal128(bytes)))
                }
                other => return Err(BsonError::UnsupportedElementType(other)),
            };

            if let Err(_) = doc.try_insert(key.clone(), value) {
                return Err(BsonError::DuplicateKey(key));
            }
        }

        // Consume doc terminator 0x00
        self.consume_byte(0x00)?;

        Ok(doc)
    }

    fn read_u8(&mut self) -> Result<u8, BsonError> {
        if self.pos >= self.data.len() {
            Err(BsonError::UnexpectedEof)
        } else {
            let b = self.data[self.pos];
            self.pos += 1;
            Ok(b)
        }
    }

    fn read_slice(&mut self, len: usize) -> Result<&'a [u8], BsonError> {
        if self.pos + len > self.data.len() {
            Err(BsonError::UnexpectedEof)
        } else {
            let slice = &self.data[self.pos..self.pos + len];
            self.pos += len;
            Ok(slice)
        }
    }

    fn consume_byte(&mut self, expected: u8) -> Result<(), BsonError> {
        let b = self.read_u8()?;
        if b != expected {
            Err(BsonError::MissingNullTerminator)
        } else {
            Ok(())
        }
    }

    fn read_i32(&mut self) -> Result<i32, BsonError> {
        let slice = self.read_slice(4)?;
        Ok(i32::from_le_bytes(slice.try_into().unwrap()))
    }

    fn read_i64(&mut self) -> Result<i64, BsonError> {
        let slice = self.read_slice(8)?;
        Ok(i64::from_le_bytes(slice.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64, BsonError> {
        let slice = self.read_slice(8)?;
        Ok(u64::from_le_bytes(slice.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64, BsonError> {
        let slice = self.read_slice(8)?;
        Ok(f64::from_le_bytes(slice.try_into().unwrap()))
    }

    fn read_cstring(&mut self) -> Result<String, BsonError> {
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos] != 0x00 {
            self.pos += 1;
        }
        if self.pos >= self.data.len() {
            return Err(BsonError::MissingNullTerminator);
        }
        let s_bytes = &self.data[start..self.pos];
        self.pos += 1; // skip null byte
        let s = std::str::from_utf8(s_bytes).map_err(|_| BsonError::InvalidUtf8)?;
        Ok(s.to_string())
    }
}

pub struct BsonEncoder {
    buf: Vec<u8>,
}

impl BsonEncoder {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn encode_document(&mut self, doc: &Document) -> Vec<u8> {
        let start_pos = self.buf.len();
        // Reserve 4 bytes for total document length
        self.buf.extend_from_slice(&[0u8; 4]);

        for (key, val) in doc.iter() {
            self.encode_element(key, val);
        }

        // Document terminator
        self.buf.push(0x00);

        let total_len = (self.buf.len() - start_pos) as i32;
        self.buf[start_pos..start_pos + 4].copy_from_slice(&total_len.to_le_bytes());

        self.buf.clone()
    }

    fn encode_element(&mut self, key: &str, val: &DataValue) {
        match val {
            DataValue::Null => {
                self.buf.push(0x0A);
                self.write_cstring(key);
            }
            DataValue::Bool(b) => {
                self.buf.push(0x08);
                self.write_cstring(key);
                self.buf.push(if *b { 0x01 } else { 0x00 });
            }
            DataValue::Number(ExactNumber::Int(i)) => {
                if *i >= i32::MIN as i64 && *i <= i32::MAX as i64 {
                    self.buf.push(0x10);
                    self.write_cstring(key);
                    self.buf.extend_from_slice(&(*i as i32).to_le_bytes());
                } else {
                    self.buf.push(0x12);
                    self.write_cstring(key);
                    self.buf.extend_from_slice(&i.to_le_bytes());
                }
            }
            DataValue::Number(ExactNumber::UInt(u)) => {
                if *u > i64::MAX as u64 {
                    self.buf.push(0x11);
                    self.write_cstring(key);
                    self.buf.extend_from_slice(&u.to_le_bytes());
                } else {
                    self.buf.push(0x12);
                    self.write_cstring(key);
                    self.buf.extend_from_slice(&(*u as i64).to_le_bytes());
                }
            }
            DataValue::Number(ExactNumber::Float(f)) => {
                self.buf.push(0x01);
                self.write_cstring(key);
                self.buf.extend_from_slice(&f.to_le_bytes());
            }
            DataValue::Number(ExactNumber::Decimal128(d)) => {
                self.buf.push(0x13);
                self.write_cstring(key);
                self.buf.extend_from_slice(&d.0);
            }
            DataValue::String(s) => {
                self.buf.push(0x02);
                self.write_cstring(key);
                let str_len = (s.len() + 1) as i32;
                self.buf.extend_from_slice(&str_len.to_le_bytes());
                self.buf.extend_from_slice(s.as_bytes());
                self.buf.push(0x00);
            }
            DataValue::Document(sub_doc) => {
                self.buf.push(0x03);
                self.write_cstring(key);
                let mut sub_encoder = BsonEncoder::new();
                let sub_bytes = sub_encoder.encode_document(sub_doc);
                self.buf.extend_from_slice(&sub_bytes);
            }
            DataValue::Array(items) => {
                self.buf.push(0x04);
                self.write_cstring(key);
                let mut arr_doc = Document::new();
                for (idx, item) in items.iter().enumerate() {
                    let _ = arr_doc.try_insert(idx.to_string(), item.clone());
                }
                let mut sub_encoder = BsonEncoder::new();
                let sub_bytes = sub_encoder.encode_document(&arr_doc);
                self.buf.extend_from_slice(&sub_bytes);
            }
            DataValue::Binary(bytes) => {
                self.buf.push(0x05);
                self.write_cstring(key);
                self.buf.extend_from_slice(&(bytes.len() as i32).to_le_bytes());
                self.buf.push(0x00); // generic binary subtype
                self.buf.extend_from_slice(bytes);
            }
            DataValue::ObjectId(oid) => {
                self.buf.push(0x07);
                self.write_cstring(key);
                self.buf.extend_from_slice(&oid.0);
            }
            DataValue::DateTimeUtc(dt) => {
                self.buf.push(0x09);
                self.write_cstring(key);
                self.buf.extend_from_slice(&dt.0.to_le_bytes());
            }
        }
    }

    fn write_cstring(&mut self, s: &str) {
        self.buf.extend_from_slice(s.as_bytes());
        self.buf.push(0x00);
    }
}

pub fn from_slice(bytes: &[u8]) -> Result<Document, BsonError> {
    let view = BsonDocumentView::from_slice(bytes)?;
    view.to_document()
}

pub fn to_vec(doc: &Document) -> Vec<u8> {
    let mut encoder = BsonEncoder::new();
    encoder.encode_document(doc)
}

/// Zero-copy borrowed view of a BSON document without full deserialization
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BsonDocumentView<'a> {
    data: &'a [u8],
}

/// Zero-copy borrowed view of a BSON array
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BsonArrayView<'a> {
    data: &'a [u8],
}

/// Zero-copy borrowed view of a BSON value
#[derive(Debug, Clone, PartialEq)]
pub enum BsonValueView<'a> {
    Double(f64),
    String(&'a str),
    Document(BsonDocumentView<'a>),
    Array(BsonArrayView<'a>),
    Binary(&'a [u8]),
    ObjectId(ObjectId),
    Bool(bool),
    DateTimeUtc(DateTimeUtc),
    Null,
    Int32(i32),
    Int64(i64),
    UInt64(u64),
    Decimal128(Decimal128),
}

impl<'a> BsonDocumentView<'a> {
    pub fn from_slice(bytes: &'a [u8]) -> Result<Self, BsonError> {
        validate_document_slice(bytes, true)?;
        Ok(Self { data: bytes })
    }

    pub fn from_slice_bounded(bytes: &'a [u8], max_len: usize) -> Result<Self, BsonError> {
        if bytes.len() > max_len {
            return Err(BsonError::MalformedSuffix);
        }
        Self::from_slice(bytes)
    }

    pub fn raw_bytes(&self) -> &'a [u8] {
        self.data
    }

    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    pub fn len(&self) -> usize {
        self.iter().count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    pub fn get(&self, target_key: &str) -> Option<BsonValueView<'a>> {
        for (key, val) in self.iter() {
            if key == target_key {
                return Some(val);
            }
        }
        None
    }

    pub fn get_str(&self, key: &str) -> Option<&'a str> {
        match self.get(key) {
            Some(BsonValueView::String(s)) => Some(s),
            _ => None,
        }
    }

    pub fn get_i32(&self, key: &str) -> Option<i32> {
        match self.get(key) {
            Some(BsonValueView::Int32(i)) => Some(i),
            _ => None,
        }
    }

    pub fn get_i64(&self, key: &str) -> Option<i64> {
        match self.get(key) {
            Some(BsonValueView::Int64(i)) => Some(i),
            Some(BsonValueView::Int32(i)) => Some(i as i64),
            _ => None,
        }
    }

    pub fn get_f64(&self, key: &str) -> Option<f64> {
        match self.get(key) {
            Some(BsonValueView::Double(f)) => Some(f),
            _ => None,
        }
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.get(key) {
            Some(BsonValueView::Bool(b)) => Some(b),
            _ => None,
        }
    }

    pub fn get_document(&self, key: &str) -> Option<BsonDocumentView<'a>> {
        match self.get(key) {
            Some(BsonValueView::Document(d)) => Some(d),
            _ => None,
        }
    }

    pub fn get_array(&self, key: &str) -> Option<BsonArrayView<'a>> {
        match self.get(key) {
            Some(BsonValueView::Array(a)) => Some(a),
            _ => None,
        }
    }

    pub fn get_object_id(&self, key: &str) -> Option<ObjectId> {
        match self.get(key) {
            Some(BsonValueView::ObjectId(oid)) => Some(oid),
            _ => None,
        }
    }

    pub fn get_datetime(&self, key: &str) -> Option<DateTimeUtc> {
        match self.get(key) {
            Some(BsonValueView::DateTimeUtc(dt)) => Some(dt),
            _ => None,
        }
    }

    pub fn keys(&self) -> Vec<&'a str> {
        self.iter().map(|(k, _)| k).collect()
    }

    pub fn iter(&self) -> BsonElementIter<'a> {
        BsonElementIter {
            data: self.data,
            pos: 4,
            limit: self.data.len() - 1,
        }
    }

    pub fn to_document(&self) -> Result<Document, BsonError> {
        let mut doc = Document::new();
        for (key, val_view) in self.iter() {
            let data_val = val_view.to_data_value()?;
            if let Err(_) = doc.try_insert(key.to_string(), data_val) {
                return Err(BsonError::DuplicateKey(key.to_string()));
            }
        }
        Ok(doc)
    }
}

impl<'a> BsonArrayView<'a> {
    pub fn from_slice(bytes: &'a [u8]) -> Result<Self, BsonError> {
        validate_document_slice(bytes, true)?;
        Ok(Self { data: bytes })
    }

    pub fn doc_view(&self) -> BsonDocumentView<'a> {
        BsonDocumentView { data: self.data }
    }

    pub fn len(&self) -> usize {
        self.doc_view().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<BsonValueView<'a>> {
        let key = index.to_string();
        self.doc_view().get(&key)
    }

    pub fn iter(&self) -> impl Iterator<Item = BsonValueView<'a>> {
        self.doc_view().iter().map(|(_, v)| v)
    }

    pub fn to_vec(&self) -> Result<Vec<DataValue>, BsonError> {
        let mut items = Vec::new();
        for (_, val_view) in self.doc_view().iter() {
            items.push(val_view.to_data_value()?);
        }
        Ok(items)
    }
}

impl<'a> BsonValueView<'a> {
    pub fn to_data_value(&self) -> Result<DataValue, BsonError> {
        match self {
            BsonValueView::Double(f) => Ok(DataValue::Number(ExactNumber::Float(*f))),
            BsonValueView::String(s) => Ok(DataValue::String((*s).to_string())),
            BsonValueView::Document(doc) => Ok(DataValue::Document(doc.to_document()?)),
            BsonValueView::Array(arr) => Ok(DataValue::Array(arr.to_vec()?)),
            BsonValueView::Binary(b) => Ok(DataValue::Binary(b.to_vec())),
            BsonValueView::ObjectId(oid) => Ok(DataValue::ObjectId(*oid)),
            BsonValueView::Bool(b) => Ok(DataValue::Bool(*b)),
            BsonValueView::DateTimeUtc(dt) => Ok(DataValue::DateTimeUtc(*dt)),
            BsonValueView::Null => Ok(DataValue::Null),
            BsonValueView::Int32(i) => Ok(DataValue::Number(ExactNumber::Int(*i as i64))),
            BsonValueView::Int64(i) => Ok(DataValue::Number(ExactNumber::Int(*i))),
            BsonValueView::UInt64(u) => Ok(DataValue::Number(ExactNumber::UInt(*u))),
            BsonValueView::Decimal128(d) => Ok(DataValue::Number(ExactNumber::Decimal128(*d))),
        }
    }

    pub fn as_str(&self) -> Option<&'a str> {
        match self {
            BsonValueView::String(s) => Some(*s),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            BsonValueView::Int64(i) => Some(*i),
            BsonValueView::Int32(i) => Some(*i as i64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            BsonValueView::Double(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            BsonValueView::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_document(&self) -> Option<BsonDocumentView<'a>> {
        match self {
            BsonValueView::Document(d) => Some(*d),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<BsonArrayView<'a>> {
        match self {
            BsonValueView::Array(a) => Some(*a),
            _ => None,
        }
    }

    pub fn as_object_id(&self) -> Option<ObjectId> {
        match self {
            BsonValueView::ObjectId(oid) => Some(*oid),
            _ => None,
        }
    }

    pub fn as_datetime(&self) -> Option<DateTimeUtc> {
        match self {
            BsonValueView::DateTimeUtc(dt) => Some(*dt),
            _ => None,
        }
    }
}

pub struct BsonElementIter<'a> {
    data: &'a [u8],
    pos: usize,
    limit: usize,
}

impl<'a> Iterator for BsonElementIter<'a> {
    type Item = (&'a str, BsonValueView<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.limit {
            return None;
        }
        let elem_type = self.data[self.pos];
        self.pos += 1;

        let key_start = self.pos;
        while self.pos < self.limit && self.data[self.pos] != 0x00 {
            self.pos += 1;
        }
        let key_bytes = &self.data[key_start..self.pos];
        self.pos += 1; // skip null byte
        let key = std::str::from_utf8(key_bytes).ok()?;

        let val_view = match elem_type {
            0x01 => {
                let bytes: [u8; 8] = self.data[self.pos..self.pos + 8].try_into().ok()?;
                self.pos += 8;
                BsonValueView::Double(f64::from_le_bytes(bytes))
            }
            0x02 => {
                let s_len = i32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().ok()?) as usize;
                self.pos += 4;
                let s_bytes = &self.data[self.pos..self.pos + s_len - 1];
                self.pos += s_len;
                let s = std::str::from_utf8(s_bytes).ok()?;
                BsonValueView::String(s)
            }
            0x03 => {
                let child_len = i32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().ok()?) as usize;
                let child_slice = &self.data[self.pos..self.pos + child_len];
                self.pos += child_len;
                BsonValueView::Document(BsonDocumentView { data: child_slice })
            }
            0x04 => {
                let child_len = i32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().ok()?) as usize;
                let child_slice = &self.data[self.pos..self.pos + child_len];
                self.pos += child_len;
                BsonValueView::Array(BsonArrayView { data: child_slice })
            }
            0x05 => {
                let bin_len = i32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().ok()?) as usize;
                self.pos += 4;
                let _subtype = self.data[self.pos];
                self.pos += 1;
                let bin_bytes = &self.data[self.pos..self.pos + bin_len];
                self.pos += bin_len;
                BsonValueView::Binary(bin_bytes)
            }
            0x07 => {
                let raw: [u8; 12] = self.data[self.pos..self.pos + 12].try_into().ok()?;
                self.pos += 12;
                BsonValueView::ObjectId(ObjectId(raw))
            }
            0x08 => {
                let b = self.data[self.pos] != 0;
                self.pos += 1;
                BsonValueView::Bool(b)
            }
            0x09 => {
                let ms = i64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().ok()?);
                self.pos += 8;
                BsonValueView::DateTimeUtc(DateTimeUtc(ms))
            }
            0x0A => BsonValueView::Null,
            0x10 => {
                let i = i32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().ok()?);
                self.pos += 4;
                BsonValueView::Int32(i)
            }
            0x11 => {
                let bytes: [u8; 8] = self.data[self.pos..self.pos + 8].try_into().ok()?;
                self.pos += 8;
                BsonValueView::UInt64(u64::from_le_bytes(bytes))
            }
            0x12 => {
                let i = i64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().ok()?);
                self.pos += 8;
                BsonValueView::Int64(i)
            }
            0x13 => {
                let raw: [u8; 16] = self.data[self.pos..self.pos + 16].try_into().ok()?;
                self.pos += 16;
                BsonValueView::Decimal128(Decimal128(raw))
            }
            _ => return None,
        };

        Some((key, val_view))
    }
}

fn validate_document_slice(bytes: &[u8], is_root: bool) -> Result<(), BsonError> {
    if bytes.len() < 5 {
        return Err(BsonError::UnexpectedEof);
    }
    if is_root && bytes.len() > MAX_BSON_BYTES {
        return Err(BsonError::MaxBytesExceeded(bytes.len()));
    }
    let doc_len = i32::from_le_bytes(bytes[0..4].try_into().unwrap());
    if doc_len < 5 {
        return Err(BsonError::InvalidDocumentLength(doc_len));
    }
    let doc_len_usize = doc_len as usize;
    if bytes.len() < doc_len_usize {
        return Err(BsonError::UnexpectedEof);
    }
    if is_root && bytes.len() > doc_len_usize {
        return Err(BsonError::MalformedSuffix);
    }
    if bytes[doc_len_usize - 1] != 0x00 {
        return Err(BsonError::MissingNullTerminator);
    }

    let limit = doc_len_usize - 1;
    let mut pos = 4;
    let mut seen_keys: Vec<&str> = Vec::new();

    while pos < limit {
        let elem_type = bytes[pos];
        pos += 1;

        let key_start = pos;
        while pos < limit && bytes[pos] != 0x00 {
            pos += 1;
        }
        if pos >= limit {
            return Err(BsonError::MissingNullTerminator);
        }
        let key_bytes = &bytes[key_start..pos];
        pos += 1; // skip null
        let key = std::str::from_utf8(key_bytes).map_err(|_| BsonError::InvalidUtf8)?;
        if seen_keys.contains(&key) {
            return Err(BsonError::DuplicateKey(key.to_string()));
        }
        seen_keys.push(key);

        match elem_type {
            0x01 => {
                if pos + 8 > limit {
                    return Err(BsonError::UnexpectedEof);
                }
                pos += 8;
            }
            0x02 => {
                if pos + 4 > limit {
                    return Err(BsonError::UnexpectedEof);
                }
                let str_len = i32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
                if str_len <= 0 {
                    return Err(BsonError::UnexpectedEof);
                }
                let str_len_usize = str_len as usize;
                pos += 4;
                if pos + str_len_usize > limit {
                    return Err(BsonError::UnexpectedEof);
                }
                if bytes[pos + str_len_usize - 1] != 0x00 {
                    return Err(BsonError::MissingNullTerminator);
                }
                let s_bytes = &bytes[pos..pos + str_len_usize - 1];
                std::str::from_utf8(s_bytes).map_err(|_| BsonError::InvalidUtf8)?;
                pos += str_len_usize;
            }
            0x03 | 0x04 => {
                if pos + 4 > limit {
                    return Err(BsonError::UnexpectedEof);
                }
                let child_len = i32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
                if child_len < 5 {
                    return Err(BsonError::InvalidDocumentLength(child_len));
                }
                let child_len_usize = child_len as usize;
                if pos + child_len_usize > limit {
                    return Err(BsonError::InvalidDocumentLength(child_len));
                }
                validate_document_slice(&bytes[pos..pos + child_len_usize], false)?;
                pos += child_len_usize;
            }
            0x05 => {
                if pos + 4 > limit {
                    return Err(BsonError::UnexpectedEof);
                }
                let bin_len = i32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
                if bin_len < 0 {
                    return Err(BsonError::UnexpectedEof);
                }
                let bin_len_usize = bin_len as usize;
                pos += 4;
                if pos + 1 + bin_len_usize > limit {
                    return Err(BsonError::UnexpectedEof);
                }
                pos += 1 + bin_len_usize;
            }
            0x07 => {
                if pos + 12 > limit {
                    return Err(BsonError::UnexpectedEof);
                }
                pos += 12;
            }
            0x08 => {
                if pos + 1 > limit {
                    return Err(BsonError::UnexpectedEof);
                }
                pos += 1;
            }
            0x09 => {
                if pos + 8 > limit {
                    return Err(BsonError::UnexpectedEof);
                }
                pos += 8;
            }
            0x0A => {}
            0x10 => {
                if pos + 4 > limit {
                    return Err(BsonError::UnexpectedEof);
                }
                pos += 4;
            }
            0x11 => {
                if pos + 8 > limit {
                    return Err(BsonError::UnexpectedEof);
                }
                pos += 8;
            }
            0x12 => {
                if pos + 8 > limit {
                    return Err(BsonError::UnexpectedEof);
                }
                pos += 8;
            }
            0x13 => {
                if pos + 16 > limit {
                    return Err(BsonError::UnexpectedEof);
                }
                pos += 16;
            }
            other => return Err(BsonError::UnsupportedElementType(other)),
        }
    }

    if pos != limit {
        return Err(BsonError::MalformedSuffix);
    }
    Ok(())
}

pub fn view(bytes: &[u8]) -> Result<BsonDocumentView<'_>, BsonError> {
    BsonDocumentView::from_slice(bytes)
}

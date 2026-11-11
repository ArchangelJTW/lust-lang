use super::document::Document;
use super::number::{DateTimeUtc, Decimal128, ExactNumber, ObjectId};
use alloc::{
    string::{String, ToString},
    vec::Vec,
};
use core::fmt;

/// First-class canonical data value across JSON, JSONC, JSONL, BSON, YAML, and TOON
#[derive(Debug, Clone, PartialEq)]
pub enum DataValue {
    Null,
    Bool(bool),
    Number(ExactNumber),
    String(String),
    Binary(Vec<u8>),
    Array(Vec<DataValue>),
    Document(Document),
    ObjectId(ObjectId),
    DateTimeUtc(DateTimeUtc),
}

impl DataValue {
    pub fn is_null(&self) -> bool {
        matches!(self, DataValue::Null)
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, DataValue::Bool(_))
    }

    pub fn is_number(&self) -> bool {
        matches!(self, DataValue::Number(_))
    }

    pub fn is_string(&self) -> bool {
        matches!(self, DataValue::String(_))
    }

    pub fn is_array(&self) -> bool {
        matches!(self, DataValue::Array(_))
    }

    pub fn is_document(&self) -> bool {
        matches!(self, DataValue::Document(_))
    }

    pub fn is_binary(&self) -> bool {
        matches!(self, DataValue::Binary(_))
    }

    pub fn is_object_id(&self) -> bool {
        matches!(self, DataValue::ObjectId(_))
    }

    pub fn is_datetime(&self) -> bool {
        matches!(self, DataValue::DateTimeUtc(_))
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            DataValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            DataValue::Number(n) => n.as_i64(),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            DataValue::Number(n) => n.as_u64(),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            DataValue::Number(n) => n.as_f64(),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            DataValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[DataValue]> {
        match self {
            DataValue::Array(arr) => Some(arr.as_slice()),
            _ => None,
        }
    }

    pub fn as_array_mut(&mut self) -> Option<&mut Vec<DataValue>> {
        match self {
            DataValue::Array(arr) => Some(arr),
            _ => None,
        }
    }

    pub fn as_document(&self) -> Option<&Document> {
        match self {
            DataValue::Document(doc) => Some(doc),
            _ => None,
        }
    }

    pub fn as_document_mut(&mut self) -> Option<&mut Document> {
        match self {
            DataValue::Document(doc) => Some(doc),
            _ => None,
        }
    }

    pub fn as_binary(&self) -> Option<&[u8]> {
        match self {
            DataValue::Binary(b) => Some(b.as_slice()),
            _ => None,
        }
    }

    pub fn as_object_id(&self) -> Option<&ObjectId> {
        match self {
            DataValue::ObjectId(oid) => Some(oid),
            _ => None,
        }
    }

    pub fn as_datetime(&self) -> Option<&DateTimeUtc> {
        match self {
            DataValue::DateTimeUtc(dt) => Some(dt),
            _ => None,
        }
    }

    pub fn as_decimal128(&self) -> Option<&Decimal128> {
        match self {
            DataValue::Number(ExactNumber::Decimal128(dec)) => Some(dec),
            _ => None,
        }
    }

    pub fn len(&self) -> Option<usize> {
        match self {
            DataValue::String(s) => Some(s.len()),
            DataValue::Binary(b) => Some(b.len()),
            DataValue::Array(arr) => Some(arr.len()),
            DataValue::Document(doc) => Some(doc.len()),
            _ => None,
        }
    }

    pub fn is_empty(&self) -> Option<bool> {
        self.len().map(|l| l == 0)
    }

    pub fn get(&self, key: &str) -> Option<&DataValue> {
        match self {
            DataValue::Document(doc) => doc.get(key),
            _ => None,
        }
    }

    pub fn get_index(&self, index: usize) -> Option<&DataValue> {
        match self {
            DataValue::Array(arr) => arr.get(index),
            _ => None,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            DataValue::Null => "null",
            DataValue::Bool(_) => "bool",
            DataValue::Number(n) => {
                if n.is_integer() {
                    "int"
                } else {
                    "float"
                }
            }
            DataValue::String(_) => "string",
            DataValue::Binary(_) => "binary",
            DataValue::Array(_) => "array",
            DataValue::Document(_) => "document",
            DataValue::ObjectId(_) => "objectId",
            DataValue::DateTimeUtc(_) => "dateTimeUtc",
        }
    }
}

impl From<bool> for DataValue {
    fn from(v: bool) -> Self {
        DataValue::Bool(v)
    }
}

impl From<i64> for DataValue {
    fn from(v: i64) -> Self {
        DataValue::Number(ExactNumber::Int(v))
    }
}

impl From<i32> for DataValue {
    fn from(v: i32) -> Self {
        DataValue::Number(ExactNumber::Int(v as i64))
    }
}

impl From<u64> for DataValue {
    fn from(v: u64) -> Self {
        DataValue::Number(ExactNumber::UInt(v))
    }
}

impl From<u32> for DataValue {
    fn from(v: u32) -> Self {
        DataValue::Number(ExactNumber::UInt(v as u64))
    }
}

impl From<f64> for DataValue {
    fn from(v: f64) -> Self {
        DataValue::Number(ExactNumber::Float(v))
    }
}

impl From<String> for DataValue {
    fn from(v: String) -> Self {
        DataValue::String(v)
    }
}

impl From<&str> for DataValue {
    fn from(v: &str) -> Self {
        DataValue::String(v.to_string())
    }
}

impl From<Vec<u8>> for DataValue {
    fn from(v: Vec<u8>) -> Self {
        DataValue::Binary(v)
    }
}

impl From<Vec<DataValue>> for DataValue {
    fn from(v: Vec<DataValue>) -> Self {
        DataValue::Array(v)
    }
}

impl From<Document> for DataValue {
    fn from(v: Document) -> Self {
        DataValue::Document(v)
    }
}

impl From<ObjectId> for DataValue {
    fn from(v: ObjectId) -> Self {
        DataValue::ObjectId(v)
    }
}

impl From<DateTimeUtc> for DataValue {
    fn from(v: DateTimeUtc) -> Self {
        DataValue::DateTimeUtc(v)
    }
}

impl From<Decimal128> for DataValue {
    fn from(v: Decimal128) -> Self {
        DataValue::Number(ExactNumber::Decimal128(v))
    }
}

impl fmt::Display for DataValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataValue::Null => write!(f, "null"),
            DataValue::Bool(b) => write!(f, "{}", b),
            DataValue::Number(n) => write!(f, "{}", n),
            DataValue::String(s) => write!(f, "\"{}\"", s),
            DataValue::Binary(b) => write!(f, "<binary len={}>", b.len()),
            DataValue::Array(arr) => {
                write!(f, "[")?;
                for (i, elem) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", elem)?;
                }
                write!(f, "]")
            }
            DataValue::Document(doc) => write!(f, "{}", doc),
            DataValue::ObjectId(oid) => write!(f, "{}", oid),
            DataValue::DateTimeUtc(dt) => write!(f, "{}", dt),
        }
    }
}

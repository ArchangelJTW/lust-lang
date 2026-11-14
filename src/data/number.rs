use alloc::string::String;
use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DataError(&'static str);

impl DataError {
    pub const fn new(msg: &'static str) -> Self {
        Self(msg)
    }

    pub fn message(&self) -> &'static str {
        self.0
    }
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// 12-byte BSON ObjectId
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObjectId(pub [u8; 12]);

impl ObjectId {
    pub const fn new(bytes: [u8; 12]) -> Self {
        Self(bytes)
    }

    pub fn from_hex(hex: &str) -> core::result::Result<Self, DataError> {
        if hex.len() != 24 {
            return Err(DataError::new("ObjectId hex string must be exactly 24 characters"));
        }
        let mut bytes = [0u8; 12];
        for i in 0..12 {
            let byte_hex = &hex[i * 2..i * 2 + 2];
            bytes[i] = u8::from_str_radix(byte_hex, 16)
                .map_err(|_| DataError::new("Invalid hex character in ObjectId"))?;
        }
        Ok(Self(bytes))
    }

    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(24);
        for b in &self.0 {
            use core::fmt::Write;
            let _ = write!(s, "{:02x}", b);
        }
        s
    }

    pub fn timestamp(&self) -> u32 {
        u32::from_be_bytes([self.0[0], self.0[1], self.0[2], self.0[3]])
    }

    pub const fn bytes(&self) -> &[u8; 12] {
        &self.0
    }
}

impl fmt::Display for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ObjectId(\"{}\")", self.to_hex())
    }
}

/// BSON UTC DateTime: 64-bit integer signed milliseconds since Unix epoch
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DateTimeUtc(pub i64);

impl DateTimeUtc {
    pub const fn from_millis(ms: i64) -> Self {
        Self(ms)
    }

    pub const fn millis(&self) -> i64 {
        self.0
    }
}

impl fmt::Display for DateTimeUtc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DateTimeUtc({} ms)", self.0)
    }
}

/// 128-bit IEEE 754-2008 decimal floating point raw representation
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Decimal128(pub [u8; 16]);

impl Decimal128 {
    pub const fn new(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub const fn raw_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

impl fmt::Display for Decimal128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Decimal128(0x")?;
        for b in &self.0 {
            write!(f, "{:02x}", b)?;
        }
        write!(f, ")")
    }
}

/// Canonical exact numerical representation distinguishing integers, unsigned, floats, and Decimal128
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExactNumber {
    Int(i64),
    UInt(u64),
    Float(f64),
    Decimal128(Decimal128),
}

impl ExactNumber {
    pub fn is_integer(&self) -> bool {
        matches!(self, ExactNumber::Int(_) | ExactNumber::UInt(_))
    }

    pub fn is_float(&self) -> bool {
        matches!(self, ExactNumber::Float(_))
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ExactNumber::Int(i) => Some(*i),
            ExactNumber::UInt(u) => i64::try_from(*u).ok(),
            ExactNumber::Float(f) => {
                if f.fract() == 0.0 && *f >= i64::MIN as f64 && *f < 9_223_372_036_854_775_808.0 {
                    Some(*f as i64)
                } else {
                    None
                }
            }
            ExactNumber::Decimal128(_) => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            ExactNumber::UInt(u) => Some(*u),
            ExactNumber::Int(i) => {
                if *i >= 0 {
                    Some(*i as u64)
                } else {
                    None
                }
            }
            ExactNumber::Float(f) => {
                if f.fract() == 0.0 && *f >= 0.0 && *f < 18_446_744_073_709_551_616.0 {
                    Some(*f as u64)
                } else {
                    None
                }
            }
            ExactNumber::Decimal128(_) => None,
        }
    }

    pub fn try_as_f64(&self) -> Option<f64> {
        const MAX_SAFE_INTEGER: i64 = 9_007_199_254_740_991; // 2^53 - 1
        const MIN_SAFE_INTEGER: i64 = -9_007_199_254_740_991;
        match self {
            ExactNumber::Int(i) => {
                if *i >= MIN_SAFE_INTEGER && *i <= MAX_SAFE_INTEGER {
                    Some(*i as f64)
                } else {
                    None
                }
            }
            ExactNumber::UInt(u) => {
                if *u <= MAX_SAFE_INTEGER as u64 {
                    Some(*u as f64)
                } else {
                    None
                }
            }
            ExactNumber::Float(f) => Some(*f),
            ExactNumber::Decimal128(_) => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        self.try_as_f64()
    }
}

impl From<i64> for ExactNumber {
    fn from(v: i64) -> Self {
        ExactNumber::Int(v)
    }
}

impl From<i32> for ExactNumber {
    fn from(v: i32) -> Self {
        ExactNumber::Int(v as i64)
    }
}

impl From<u64> for ExactNumber {
    fn from(v: u64) -> Self {
        ExactNumber::UInt(v)
    }
}

impl From<u32> for ExactNumber {
    fn from(v: u32) -> Self {
        ExactNumber::UInt(v as u64)
    }
}

impl From<f64> for ExactNumber {
    fn from(v: f64) -> Self {
        ExactNumber::Float(v)
    }
}

impl From<Decimal128> for ExactNumber {
    fn from(v: Decimal128) -> Self {
        ExactNumber::Decimal128(v)
    }
}

impl fmt::Display for ExactNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExactNumber::Int(i) => write!(f, "{}", i),
            ExactNumber::UInt(u) => write!(f, "{}", u),
            ExactNumber::Float(fl) => write!(f, "{}", fl),
            ExactNumber::Decimal128(d) => write!(f, "{}", d),
        }
    }
}

#![allow(dead_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericType {
    Int,
    Float,
}

#[cfg(feature = "std")]
pub type LustInt = i64;
#[cfg(not(feature = "std"))]
pub type LustInt = i32;

#[cfg(feature = "std")]
pub type LustFloat = f64;
#[cfg(not(feature = "std"))]
pub type LustFloat = f32;

#[cfg(feature = "std")]
pub type FloatBits = u64;
#[cfg(not(feature = "std"))]
pub type FloatBits = u32;

#[inline]
pub const fn int_zero() -> LustInt {
    0
}

#[inline]
pub fn int_from_usize(value: usize) -> LustInt {
    value as LustInt
}

#[inline]
pub fn float_from_int(value: LustInt) -> LustFloat {
    value as LustFloat
}

#[inline]
pub fn int_from_float(value: LustFloat) -> LustInt {
    value as LustInt
}

#[inline]
pub fn float_to_hash_bits(value: LustFloat) -> u64 {
    #[cfg(feature = "std")]
    {
        value.to_bits()
    }
    #[cfg(not(feature = "std"))]
    {
        value.to_bits() as u64
    }
}

#[inline]
pub fn float_is_nan(value: LustFloat) -> bool {
    value.is_nan()
}

#[inline]
pub fn parse_float(input: &str) -> Result<LustFloat, core::num::ParseFloatError> {
    input.parse::<LustFloat>()
}

#[inline]
pub fn float_abs(value: LustFloat) -> LustFloat {
    value.abs()
}

#[inline]
pub fn float_floor(value: LustFloat) -> LustFloat {
    #[cfg(feature = "std")]
    {
        value.floor()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::floorf(value)
    }
}

#[inline]
pub fn float_ceil(value: LustFloat) -> LustFloat {
    #[cfg(feature = "std")]
    {
        value.ceil()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::ceilf(value)
    }
}

#[inline]
pub fn float_round(value: LustFloat) -> LustFloat {
    #[cfg(feature = "std")]
    {
        value.round()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::roundf(value)
    }
}

#[inline]
pub fn float_sqrt(value: LustFloat) -> LustFloat {
    #[cfg(feature = "std")]
    {
        value.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrtf(value)
    }
}

#[inline]
pub fn float_sin(value: LustFloat) -> LustFloat {
    #[cfg(feature = "std")]
    {
        value.sin()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sinf(value)
    }
}

#[inline]
pub fn float_cos(value: LustFloat) -> LustFloat {
    #[cfg(feature = "std")]
    {
        value.cos()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::cosf(value)
    }
}

#[inline]
pub fn float_tan(value: LustFloat) -> LustFloat {
    #[cfg(feature = "std")]
    {
        value.tan()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::tanf(value)
    }
}

#[inline]
pub fn float_asin(value: LustFloat) -> LustFloat {
    #[cfg(feature = "std")]
    {
        value.asin()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::asinf(value)
    }
}

#[inline]
pub fn float_acos(value: LustFloat) -> LustFloat {
    #[cfg(feature = "std")]
    {
        value.acos()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::acosf(value)
    }
}

#[inline]
pub fn float_atan(value: LustFloat) -> LustFloat {
    #[cfg(feature = "std")]
    {
        value.atan()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::atanf(value)
    }
}

#[inline]
pub fn float_atan2(y: LustFloat, x: LustFloat) -> LustFloat {
    #[cfg(feature = "std")]
    {
        y.atan2(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::atan2f(y, x)
    }
}

#[inline]
pub fn float_clamp(value: LustFloat, min: LustFloat, max: LustFloat) -> LustFloat {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

#[inline]
pub fn float_is_finite(value: LustFloat) -> bool {
    value.is_finite()
}

#[inline]
pub fn float_is_infinite(value: LustFloat) -> bool {
    value.is_infinite()
}

#[inline]
pub fn float_is_sign_negative(value: LustFloat) -> bool {
    value.is_sign_negative()
}

#[inline]
pub fn float_is_sign_positive(value: LustFloat) -> bool {
    value.is_sign_positive()
}

#[inline]
pub fn float_min(a: LustFloat, b: LustFloat) -> LustFloat {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else if a < b {
        a
    } else {
        b
    }
}

#[inline]
pub fn float_max(a: LustFloat, b: LustFloat) -> LustFloat {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else if a > b {
        a
    } else {
        b
    }
}

#[inline]
pub fn float_pow(base: LustFloat, exp: LustFloat) -> LustFloat {
    #[cfg(feature = "std")]
    {
        base.powf(exp)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::powf(base, exp)
    }
}

#[inline]
pub fn float_exp(value: LustFloat) -> LustFloat {
    #[cfg(feature = "std")]
    {
        value.exp()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::expf(value)
    }
}

#[inline]
pub fn float_ln(value: LustFloat) -> LustFloat {
    #[cfg(feature = "std")]
    {
        value.ln()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::logf(value)
    }
}

pub const MAX_SAFE_INTEGER_F64: f64 = 9007199254740991.0; // 2^53 - 1
pub const MIN_SAFE_INTEGER_F64: f64 = -9007199254740991.0;

#[inline]
pub fn is_safe_integer_f64(value: f64) -> bool {
    value.is_finite()
        && value.fract() == 0.0
        && value >= MIN_SAFE_INTEGER_F64
        && value <= MAX_SAFE_INTEGER_F64
}

#[inline]
pub fn safe_float_to_int(value: LustFloat) -> Option<LustInt> {
    if !value.is_finite() {
        return None;
    }
    if value < LustInt::MIN as LustFloat || value > LustInt::MAX as LustFloat {
        return None;
    }
    Some(value as LustInt)
}

#[inline]
pub fn safe_int_to_float_exact(value: LustInt) -> Option<LustFloat> {
    const MAX_EXACT: LustInt = 9007199254740991;
    const MIN_EXACT: LustInt = -9007199254740991;
    if value >= MIN_EXACT && value <= MAX_EXACT {
        Some(value as LustFloat)
    } else {
        None
    }
}

#[inline]
pub fn int_clamp(value: LustInt, min: LustInt, max: LustInt) -> LustInt {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

#[inline]
pub fn int_abs(value: LustInt) -> Option<LustInt> {
    value.checked_abs()
}

use alloc::string::String;
use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumberParseError {
    InvalidDigit { char: char, radix: u32 },
    Overflow(&'static str),
    MalformedLiteral(&'static str),
}

impl fmt::Display for NumberParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDigit { char, radix } => write!(f, "Invalid digit '{}' for base {}", char, radix),
            Self::Overflow(msg) => write!(f, "Integer overflow: {}", msg),
            Self::MalformedLiteral(msg) => write!(f, "Malformed numeric literal: {}", msg),
        }
    }
}

impl core::error::Error for NumberParseError {}

/// Parse integer literal (decimal, 0x, 0b, 0o) with digit separators '_'
pub fn parse_integer_literal(input: &str) -> Result<LustInt, NumberParseError> {
    let clean: String = input.chars().filter(|&c| c != '_').collect();
    if clean.is_empty() {
        return Err(NumberParseError::MalformedLiteral("Empty literal"));
    }

    #[cfg(feature = "std")]
    {
        if clean.starts_with("0x") || clean.starts_with("0X") {
            let digits = &clean[2..];
            if digits.is_empty() {
                return Err(NumberParseError::MalformedLiteral("Expected hex digits after prefix"));
            }
            u64::from_str_radix(digits, 16)
                .map(|v| v as i64)
                .map_err(|_| NumberParseError::Overflow("exceeds u64::MAX"))
        } else if clean.starts_with("0b") || clean.starts_with("0B") {
            let digits = &clean[2..];
            if digits.is_empty() {
                return Err(NumberParseError::MalformedLiteral("Expected binary digits after prefix"));
            }
            u64::from_str_radix(digits, 2)
                .map(|v| v as i64)
                .map_err(|_| NumberParseError::Overflow("exceeds u64::MAX"))
        } else if clean.starts_with("0o") || clean.starts_with("0O") {
            let digits = &clean[2..];
            if digits.is_empty() {
                return Err(NumberParseError::MalformedLiteral("Expected octal digits after prefix"));
            }
            u64::from_str_radix(digits, 8)
                .map(|v| v as i64)
                .map_err(|_| NumberParseError::Overflow("exceeds u64::MAX"))
        } else {
            clean.parse::<i64>()
                .map_err(|_| NumberParseError::Overflow("exceeds i64::MAX"))
        }
    }
    #[cfg(not(feature = "std"))]
    {
        if clean.starts_with("0x") || clean.starts_with("0X") {
            let digits = &clean[2..];
            if digits.is_empty() {
                return Err(NumberParseError::MalformedLiteral("Expected hex digits after prefix"));
            }
            u32::from_str_radix(digits, 16)
                .map(|v| v as i32)
                .map_err(|_| NumberParseError::Overflow("exceeds u32::MAX"))
        } else if clean.starts_with("0b") || clean.starts_with("0B") {
            let digits = &clean[2..];
            if digits.is_empty() {
                return Err(NumberParseError::MalformedLiteral("Expected binary digits after prefix"));
            }
            u32::from_str_radix(digits, 2)
                .map(|v| v as i32)
                .map_err(|_| NumberParseError::Overflow("exceeds u32::MAX"))
        } else if clean.starts_with("0o") || clean.starts_with("0O") {
            let digits = &clean[2..];
            if digits.is_empty() {
                return Err(NumberParseError::MalformedLiteral("Expected octal digits after prefix"));
            }
            u32::from_str_radix(digits, 8)
                .map(|v| v as i32)
                .map_err(|_| NumberParseError::Overflow("exceeds u32::MAX"))
        } else {
            clean.parse::<i32>()
                .map_err(|_| NumberParseError::Overflow("exceeds i32::MAX"))
        }
    }
}

/// Parse floating point literal with digit separators '_'
pub fn parse_float_literal(input: &str) -> Result<LustFloat, NumberParseError> {
    let clean: String = input.chars().filter(|&c| c != '_').collect();
    clean.parse::<LustFloat>()
        .map_err(|_| NumberParseError::MalformedLiteral("Invalid floating point format"))
}

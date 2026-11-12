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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_radix_bases_and_prefixes() {
        // Decimal
        assert_eq!(parse_integer_literal("0").unwrap(), 0);
        assert_eq!(parse_integer_literal("42").unwrap(), 42);
        assert_eq!(parse_integer_literal("-42").unwrap(), -42);
        assert_eq!(parse_integer_literal("1234567890").unwrap(), 1234567890);

        // Hexadecimal (lower and upper prefix)
        assert_eq!(parse_integer_literal("0x0").unwrap(), 0);
        assert_eq!(parse_integer_literal("0x0F").unwrap(), 15);
        assert_eq!(parse_integer_literal("0Xff").unwrap(), 255);
        assert_eq!(parse_integer_literal("0xDead_Beef").unwrap(), 3735928559);
        assert_eq!(
            parse_integer_literal("0x1234_5678_9ABC_DEF0").unwrap(),
            0x1234_5678_9ABC_DEF0u64 as i64
        );

        // Binary (lower and upper prefix)
        assert_eq!(parse_integer_literal("0b0").unwrap(), 0);
        assert_eq!(parse_integer_literal("0b1").unwrap(), 1);
        assert_eq!(parse_integer_literal("0B1010").unwrap(), 10);
        assert_eq!(parse_integer_literal("0b1111_0000").unwrap(), 240);

        // Octal (lower and upper prefix)
        assert_eq!(parse_integer_literal("0o0").unwrap(), 0);
        assert_eq!(parse_integer_literal("0o7").unwrap(), 7);
        assert_eq!(parse_integer_literal("0O755").unwrap(), 493);
        assert_eq!(parse_integer_literal("0o644").unwrap(), 420);
        assert_eq!(parse_integer_literal("0o177777").unwrap(), 65535);
        assert_eq!(parse_integer_literal("0o0123_4567").unwrap(), 342391);
    }

    #[test]
    fn test_msb_boundary_vectors() {
        // Hexadecimal boundary bit patterns
        assert_eq!(parse_integer_literal("0x7FFF_FFFF_FFFF_FFFF").unwrap(), i64::MAX);
        assert_eq!(parse_integer_literal("0x8000_0000_0000_0000").unwrap(), i64::MIN);
        assert_eq!(parse_integer_literal("0xFFFF_FFFF_FFFF_FFFF").unwrap(), -1);

        // Binary boundary bit patterns
        // i64::MAX = 63 ones (0 followed by 63 ones)
        assert_eq!(
            parse_integer_literal("0b0111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111").unwrap(),
            i64::MAX
        );
        // i64::MIN = 1 followed by 63 zeros
        assert_eq!(
            parse_integer_literal("0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000").unwrap(),
            i64::MIN
        );
        // -1 = 64 ones
        assert_eq!(
            parse_integer_literal("0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111").unwrap(),
            -1
        );

        // Octal boundary bit patterns
        // 0o0777_7777_7777_7777_7777_77 (21 sevens preceded by 0) = 63 ones = i64::MAX
        assert_eq!(
            parse_integer_literal("0o0777_7777_7777_7777_7777_77").unwrap(),
            i64::MAX
        );
        // 0o1000_0000_0000_0000_0000_00 (1 followed by 21 zeros) = 1 << 63 = i64::MIN
        assert_eq!(
            parse_integer_literal("0o1000_0000_0000_0000_0000_00").unwrap(),
            i64::MIN
        );
        // 0o1777_7777_7777_7777_7777_77 (1 followed by 21 sevens) = 64 ones = -1
        assert_eq!(
            parse_integer_literal("0o1777_7777_7777_7777_7777_77").unwrap(),
            -1
        );
    }

    #[test]
    fn test_underscore_cleaning() {
        // Decimal integers
        assert_eq!(parse_integer_literal("1_000_000").unwrap(), 1_000_000);
        assert_eq!(parse_integer_literal("42_133_7").unwrap(), 421_337);
        assert_eq!(parse_integer_literal("1_0").unwrap(), 10);
        assert_eq!(parse_integer_literal("9_223_372_036_854_775_807").unwrap(), i64::MAX);

        // Radix integers
        assert_eq!(parse_integer_literal("0xDE_AD_BE_EF").unwrap(), 0xDEAD_BEEFu64 as i64);
        assert_eq!(parse_integer_literal("0b10_10_10_10").unwrap(), 170);
        assert_eq!(parse_integer_literal("0o7_5_5").unwrap(), 493);

        // Floats
        assert_eq!(parse_float_literal("3.1415_9265").unwrap(), 3.14159265);
        assert_eq!(parse_float_literal("1_000.5_0").unwrap(), 1000.5);
        assert_eq!(parse_float_literal("1_0.5").unwrap(), 10.5);
        assert_eq!(parse_float_literal("1e1_0").unwrap(), 1e10);
        assert_eq!(parse_float_literal("1_000.0e+0_2").unwrap(), 100000.0);
    }

    #[test]
    fn test_overflow_teeth() {
        // Radix overflow > u64::MAX
        assert_eq!(
            parse_integer_literal("0x1_0000_0000_0000_0000").unwrap_err(),
            NumberParseError::Overflow("exceeds u64::MAX")
        );
        assert_eq!(
            parse_integer_literal("0b1_00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000").unwrap_err(),
            NumberParseError::Overflow("exceeds u64::MAX")
        );
        assert_eq!(
            parse_integer_literal("0o2000_0000_0000_0000_0000_00").unwrap_err(),
            NumberParseError::Overflow("exceeds u64::MAX")
        );

        // Decimal overflow > i64::MAX
        assert_eq!(
            parse_integer_literal("9223372036854775808").unwrap_err(),
            NumberParseError::Overflow("exceeds i64::MAX")
        );
        assert_eq!(
            parse_integer_literal("99999999999999999999").unwrap_err(),
            NumberParseError::Overflow("exceeds i64::MAX")
        );
    }

    #[test]
    fn test_malformed_literals_teeth() {
        // Empty / only underscores
        assert_eq!(
            parse_integer_literal("").unwrap_err(),
            NumberParseError::MalformedLiteral("Empty literal")
        );
        assert_eq!(
            parse_integer_literal("___").unwrap_err(),
            NumberParseError::MalformedLiteral("Empty literal")
        );

        // Prefixes without digits
        assert_eq!(
            parse_integer_literal("0x").unwrap_err(),
            NumberParseError::MalformedLiteral("Expected hex digits after prefix")
        );
        assert_eq!(
            parse_integer_literal("0X").unwrap_err(),
            NumberParseError::MalformedLiteral("Expected hex digits after prefix")
        );
        assert_eq!(
            parse_integer_literal("0b").unwrap_err(),
            NumberParseError::MalformedLiteral("Expected binary digits after prefix")
        );
        assert_eq!(
            parse_integer_literal("0B").unwrap_err(),
            NumberParseError::MalformedLiteral("Expected binary digits after prefix")
        );
        assert_eq!(
            parse_integer_literal("0o").unwrap_err(),
            NumberParseError::MalformedLiteral("Expected octal digits after prefix")
        );
        assert_eq!(
            parse_integer_literal("0O").unwrap_err(),
            NumberParseError::MalformedLiteral("Expected octal digits after prefix")
        );

        // Malformed float
        assert_eq!(
            parse_float_literal("").unwrap_err(),
            NumberParseError::MalformedLiteral("Invalid floating point format")
        );
        assert_eq!(
            parse_float_literal("abc").unwrap_err(),
            NumberParseError::MalformedLiteral("Invalid floating point format")
        );
        assert_eq!(
            parse_float_literal("1.2.3").unwrap_err(),
            NumberParseError::MalformedLiteral("Invalid floating point format")
        );
    }
}


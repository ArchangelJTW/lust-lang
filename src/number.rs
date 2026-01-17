#![allow(dead_code)]

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

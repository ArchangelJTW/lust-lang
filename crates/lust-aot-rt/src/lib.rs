//! Lightweight `#![no_std] + alloc` runtime crate for Lust Rust-based AOT binaries.
//!
//! Zero external dependencies. Every runtime operation returns `Result<_, AotError>`
//! and never calls `panic!()` or `.unwrap()` (Hazard 9), guaranteeing deterministic
//! `defer`/`errdefer` cleanup via `core::mem::DropGuard` even under `-C panic=abort`.

#![no_std]

extern crate alloc as alloc_crate;

#[cfg(any(feature = "std", test))]
extern crate std;

pub mod alloc;
pub mod effects;
pub mod value;

pub use crate::alloc::{
    BudgetAllocError, DEFAULT_CHUNK_SIZE, LustBudgetAlloc, LustChunkArena, StaticSlabAllocator,
};
pub use crate::effects::{
    AotFiberHandle, AotFiberState, AotScope, AotScopePhase, EffectHandlerArm, EffectHandlerFrame,
    EffectRuntime, EffectYield, OneShotResume, active_scope_depth, default_zst_provider,
    get_provided_service, handle_non_tail_oneshot, handle_tail_resumptive,
    handle_tail_resumptive_typed, perform_effect, resume_effect, scope_register_finalizer,
    scope_spawn_child, scope_spawn_future, with_aot_scope, with_provided_service,
};
pub use crate::schema::{AotDecimal128, AotObjectId};
pub use crate::value::{AotClosure, AotEnum, AotKey, AotMap, AotStruct, AotValue};

use alloc_crate::string::String;
use core::convert::Infallible;
use core::fmt::{self, Write as _};
use core::mem::DropGuard;

/// Structured runtime error returned by Lust Rust AOT functions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AotError {
    OutOfGas { limit: u64, used: u64 },
    OutOfMemory { limit_bytes: usize, requested_bytes: usize },
    DivisionByZero,
    NegativeExponent(i64),
    IndexOutOfBounds { index: i64, len: usize },
    MissingMapKey,
    SchemaDecodeError(&'static str),
    CapabilityDenied(&'static str),
    UnhandledEffect(String),
    EffectAbort { handler_id: u64 },
    UserError(String),
}

impl AotError {
    /// Canonical error-class identifier used by differential oracles.
    #[inline]
    pub const fn error_class(&self) -> &'static str {
        match self {
            Self::OutOfGas { .. } => "OutOfGas",
            Self::OutOfMemory { .. } => "OutOfMemory",
            Self::DivisionByZero => "DivisionByZero",
            Self::NegativeExponent(_) => "NegativeExponent",
            Self::IndexOutOfBounds { .. } => "IndexOutOfBounds",
            Self::MissingMapKey => "MissingMapKey",
            Self::SchemaDecodeError(_) => "SchemaDecodeError",
            Self::CapabilityDenied(_) => "CapabilityDenied",
            Self::UnhandledEffect(_) => "UnhandledEffect",
            Self::EffectAbort { .. } => "EffectAbort",
            Self::UserError(_) => "UserError",
        }
    }

    /// Constructs an [`AotError::OutOfMemory`] snapshot from a [`LustBudgetAlloc`].
    #[inline]
    pub fn from_alloc_failure(alloc: &LustBudgetAlloc, requested_bytes: usize) -> Self {
        Self::OutOfMemory {
            limit_bytes: alloc.limit_bytes(),
            requested_bytes,
        }
    }
}

impl fmt::Display for AotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfGas { limit, used } => {
                write!(f, "Out of gas: limit {limit}, used {used}")
            }
            Self::OutOfMemory {
                limit_bytes,
                requested_bytes,
            } => {
                write!(
                    f,
                    "Out of memory: limit {limit_bytes} bytes, requested {requested_bytes} bytes"
                )
            }
            Self::DivisionByZero => write!(f, "Division by zero"),
            Self::NegativeExponent(exp) => {
                write!(f, "Negative exponent in integer power: {exp}")
            }
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "Index out of bounds: index {index}, len {len}")
            }
            Self::MissingMapKey => write!(f, "Missing map key"),
            Self::SchemaDecodeError(msg) => write!(f, "Schema decode error: {msg}"),
            Self::CapabilityDenied(cap) => write!(f, "Capability denied: {cap}"),
            Self::UnhandledEffect(name) => write!(f, "Unhandled effect: {name}"),
            Self::EffectAbort { handler_id } => {
                write!(f, "Unhandled effect abort (handler {handler_id})")
            }
            Self::UserError(msg) => write!(f, "{msg}"),
        }
    }
}

impl core::error::Error for AotError {}

impl From<Infallible> for AotError {
    #[inline(always)]
    fn from(never: Infallible) -> Self {
        match never {}
    }
}

impl From<BudgetAllocError> for AotError {
    #[inline]
    fn from(err: BudgetAllocError) -> Self {
        Self::OutOfMemory {
            limit_bytes: err.limit,
            requested_bytes: err.requested,
        }
    }
}

/// Execution context threaded through Lust Rust AOT functions.
#[derive(Debug)]
pub struct AotCtx {
    pub gas_remaining: i64,
    pub gas_limit: Option<u64>,
    pub budget_alloc: LustBudgetAlloc,
    pub deny_ffi: bool,
    pub pure_core: bool,
    pub effects: EffectRuntime,
}

impl AotCtx {
    /// Creates an unbounded AOT context (no gas limit, unbounded memory, FFI allowed).
    #[inline]
    pub const fn new() -> Self {
        Self {
            gas_remaining: i64::MAX,
            gas_limit: None,
            budget_alloc: LustBudgetAlloc::unbounded(),
            deny_ffi: false,
            pure_core: false,
            effects: EffectRuntime::new(),
        }
    }

    /// Creates an AOT context with explicit optional gas and memory limits.
    #[inline]
    pub const fn with_limits(gas_limit: Option<u64>, memory_limit_bytes: Option<usize>) -> Self {
        let gas_remaining = match gas_limit {
            Some(limit) => {
                if limit > i64::MAX as u64 {
                    i64::MAX
                } else {
                    limit as i64
                }
            }
            None => i64::MAX,
        };
        let budget_alloc = match memory_limit_bytes {
            Some(limit) => LustBudgetAlloc::with_limit(limit),
            None => LustBudgetAlloc::unbounded(),
        };
        Self {
            gas_remaining,
            gas_limit,
            budget_alloc,
            deny_ffi: false,
            pure_core: false,
            effects: EffectRuntime::new(),
        }
    }

    /// Creates an AOT context with a gas limit and unbounded memory.
    #[inline]
    pub const fn with_gas_limit(gas_limit: u64) -> Self {
        Self::with_limits(Some(gas_limit), None)
    }

    /// Creates an AOT context with a memory limit and unbounded gas.
    #[inline]
    pub const fn with_memory_limit(limit_bytes: usize) -> Self {
        Self::with_limits(None, Some(limit_bytes))
    }

    /// Creates a `PureCore` AOT context (`deny_ffi = true`, `pure_core = true`).
    #[inline]
    pub const fn pure_core() -> Self {
        Self {
            gas_remaining: i64::MAX,
            gas_limit: None,
            budget_alloc: LustBudgetAlloc::unbounded(),
            deny_ffi: true,
            pure_core: true,
            effects: EffectRuntime::new(),
        }
    }

    /// Returns total gas units consumed so far (0 when `gas_limit` is `None`).
    #[inline]
    pub fn gas_used(&self) -> u64 {
        match self.gas_limit {
            Some(limit) => {
                let initial = limit.min(i64::MAX as u64);
                if self.gas_remaining >= 0 {
                    initial.saturating_sub(self.gas_remaining as u64)
                } else {
                    initial.saturating_add(self.gas_remaining.unsigned_abs())
                }
            }
            None => 0,
        }
    }

    /// Charges `amt` gas units. Returns `Err(AotError::OutOfGas { .. })` when quota is exceeded.
    ///
    /// Non-positive `amt` is a no-op so callers cannot forge negative gas refunds.
    #[inline(always)]
    pub fn charge_gas(&mut self, amt: i64) -> Result<(), AotError> {
        if amt <= 0 {
            return Ok(());
        }
        if let Some(limit) = self.gas_limit {
            let next = self.gas_remaining.saturating_sub(amt);
            if next < 0 {
                return Err(self.trip_out_of_gas(limit, amt, next));
            }
            self.gas_remaining = next;
        }
        Ok(())
    }

    #[cold]
    #[inline(never)]
    fn trip_out_of_gas(&mut self, limit: u64, amt: i64, next: i64) -> AotError {
        let used = self.gas_used().saturating_add(amt as u64);
        self.gas_remaining = next;
        AotError::OutOfGas { limit, used }
    }

    /// Checks whether an FFI / host capability is permitted in this context.
    #[inline]
    pub fn check_capability(&self, capability: &'static str) -> Result<(), AotError> {
        if self.deny_ffi || self.pure_core {
            Err(AotError::CapabilityDenied(capability))
        } else {
            Ok(())
        }
    }

    /// Charges `bytes` against `self.budget_alloc`, converting failures into [`AotError::OutOfMemory`].
    #[inline]
    pub fn try_charge_memory(&self, bytes: usize) -> Result<(), AotError> {
        self.budget_alloc.try_charge(bytes).map_err(AotError::from)
    }
}

impl Default for AotCtx {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Signed 64-bit truncating division matching Lust VM and Cranelift JIT semantics.
///
/// - `b == 0` -> `Err(AotError::DivisionByZero)`
/// - `i64::MIN / -1` -> `Ok(i64::MIN)` (wrapping, never panics)
#[inline]
pub fn div_int(a: i64, b: i64) -> Result<i64, AotError> {
    if b == 0 {
        return Err(AotError::DivisionByZero);
    }
    Ok(a.wrapping_div(b))
}

/// Signed 64-bit remainder matching Lust VM and Cranelift JIT semantics.
///
/// - `b == 0` -> `Err(AotError::DivisionByZero)`
/// - `i64::MIN % -1` -> `Ok(0)` (wrapping, never panics)
#[inline]
pub fn rem_int(a: i64, b: i64) -> Result<i64, AotError> {
    if b == 0 {
        return Err(AotError::DivisionByZero);
    }
    Ok(a.wrapping_rem(b))
}

/// Signed 64-bit floor division (`//`) matching Lust VM and Cranelift JIT semantics.
///
/// - `b == 0` -> `Err(AotError::DivisionByZero)`
/// - `i64::MIN // -1` -> `Ok(i64::MIN)`
/// - `-7 // 2 == -4`, `7 // -2 == -4`, `-7 // -2 == 3`
#[inline]
pub fn floordiv_int(a: i64, b: i64) -> Result<i64, AotError> {
    if b == 0 {
        return Err(AotError::DivisionByZero);
    }
    if a == i64::MIN && b == -1 {
        return Ok(i64::MIN);
    }
    let d = a.wrapping_div(b);
    let rem = a.wrapping_rem(b);
    if rem != 0 && ((rem < 0) != (b < 0)) {
        Ok(d.wrapping_sub(1))
    } else {
        Ok(d)
    }
}

/// Signed 64-bit integer exponentiation matching Lust VM and Cranelift JIT semantics.
///
/// - `exp < 0` -> `Err(AotError::NegativeExponent(exp))`
/// - `pow_int(0, 0) == Ok(1)`
/// - Wrapping multiplication on overflow (never panics).
#[inline]
pub fn pow_int(mut base: i64, exp: i64) -> Result<i64, AotError> {
    if exp < 0 {
        return Err(AotError::NegativeExponent(exp));
    }
    let mut e = exp as u64;
    let mut res: i64 = 1;
    while e > 0 {
        if (e & 1) == 1 {
            res = res.wrapping_mul(base);
        }
        e >>= 1;
        if e > 0 {
            base = base.wrapping_mul(base);
        }
    }
    Ok(res)
}

/// Checked array index conversion for Lust `i64` indices.
#[inline]
pub fn array_index_check(index: i64, len: usize) -> Result<usize, AotError> {
    if index < 0 {
        return Err(AotError::IndexOutOfBounds { index, len });
    }
    let Ok(idx) = usize::try_from(index) else {
        return Err(AotError::IndexOutOfBounds { index, len });
    };
    if idx >= len {
        return Err(AotError::IndexOutOfBounds { index, len });
    }
    Ok(idx)
}

/// Fast decimal integer append into `buf` (zero-division fast path for `0..=9`, panic-free).
#[inline]
pub fn string_append_int(buf: &mut String, value: i64) {
    if (0..=9).contains(&value) {
        buf.push((b'0' + (value as u8)) as char);
        return;
    }
    string_append_int_slow(buf, value);
}

#[cold]
#[inline(never)]
fn string_append_int_slow(buf: &mut String, value: i64) {
    let mut digits = [0u8; 20];
    let mut pos = digits.len();
    let mut n = value.unsigned_abs();
    while n > 0 && pos > 0 {
        pos -= 1;
        digits[pos] = b'0' + ((n % 10) as u8);
        n /= 10;
    }
    if value < 0 {
        buf.push('-');
    }
    if let Ok(s) = core::str::from_utf8(&digits[pos..]) {
        buf.push_str(s);
    }
}

/// Budget-checked integer append into `buf` (returns `Err(AotError::OutOfMemory)` on limit breach).
pub fn try_string_append_int(
    buf: &mut String,
    value: i64,
    alloc: &LustBudgetAlloc,
) -> Result<(), AotError> {
    let mut digits = [0u8; 21];
    let mut pos = digits.len();
    if value == 0 {
        pos -= 1;
        digits[pos] = b'0';
    } else {
        let mut n = value.unsigned_abs();
        while n > 0 && pos > 0 {
            pos -= 1;
            digits[pos] = b'0' + ((n % 10) as u8);
            n /= 10;
        }
        if value < 0 && pos > 0 {
            pos -= 1;
            digits[pos] = b'-';
        }
    }
    let slice = &digits[pos..];
    let add_len = slice.len();
    let old_cap = buf.capacity();
    let needed_len = buf.len().saturating_add(add_len);
    if needed_len > old_cap {
        let extra = needed_len.saturating_sub(old_cap);
        alloc.try_charge(extra)?;
        if buf.try_reserve_exact(add_len).is_err() {
            alloc.refund(extra);
            return Err(AotError::from_alloc_failure(
                alloc,
                alloc.used_bytes().saturating_add(extra),
            ));
        }
        let actual_delta = buf.capacity().saturating_sub(old_cap);
        if actual_delta > extra && alloc.try_charge(actual_delta - extra).is_err() {
            alloc.refund(extra);
            return Err(AotError::from_alloc_failure(
                alloc,
                alloc.used_bytes().saturating_add(actual_delta),
            ));
        }
    }
    if let Ok(s) = core::str::from_utf8(slice) {
        buf.push_str(s);
    }
    Ok(())
}

/// Formats an `f64` using the exact same `core::fmt::Display` formatting as Lust VM (`Value::Float`).
#[inline]
pub fn format_float(value: f64) -> String {
    let mut out = String::new();
    let _ = write!(&mut out, "{value}");
    out
}

/// Appends a formatted `f64` directly into `buf` without intermediate allocation.
#[inline]
pub fn string_append_float(buf: &mut String, value: f64) {
    let _ = write!(buf, "{value}");
}

/// Registers an unconditional `defer` cleanup closure backed by [`core::mem::DropGuard`].
///
/// Runs automatically on scope exit, early `return`, and `?` propagation (even under `-C panic=abort`).
#[inline(always)]
pub fn defer<F: FnOnce()>(f: F) -> DropGuard<(), impl FnOnce(())> {
    DropGuard::new((), move |()| f())
}

/// Registers an `errdefer` cleanup closure backed by [`core::mem::DropGuard`].
///
/// Dismiss before a successful return via `core::mem::DropGuard::dismiss(guard)`;
/// otherwise executes automatically on `Err(...)` propagation via `?` (even under `-C panic=abort`).
#[inline(always)]
pub fn errdefer<F: FnOnce()>(f: F) -> DropGuard<(), impl FnOnce(())> {
    DropGuard::new((), move |()| f())
}

/// Panic-free shared borrow of a [`core::cell::RefCell`] for aliased Tier 4 containers/structs.
#[inline(always)]
pub fn ref_borrow<T>(cell: &core::cell::RefCell<T>) -> Result<core::cell::Ref<'_, T>, AotError> {
    cell.try_borrow()
        .map_err(|_| AotError::UserError(String::from("re-entrant borrow")))
}

/// Panic-free mutable borrow of a [`core::cell::RefCell`] for aliased Tier 4 containers/structs.
#[inline(always)]
pub fn ref_borrow_mut<T>(
    cell: &core::cell::RefCell<T>,
) -> Result<core::cell::RefMut<'_, T>, AotError> {
    cell.try_borrow_mut()
        .map_err(|_| AotError::UserError(String::from("re-entrant mutable borrow")))
}

/// Writes `text` to stdout without panicking on broken pipes (available under the `std` feature).
#[cfg(feature = "std")]
#[inline]
pub fn print_str(text: &str) {
    use std::io::Write as _;
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(text.as_bytes());
}

/// Writes `text` followed by a newline to stdout without panicking on broken pipes.
#[cfg(feature = "std")]
#[inline]
pub fn println_str(text: &str) {
    use std::io::Write as _;
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(text.as_bytes());
    let _ = out.write_all(b"\n");
}

/// Panic-free, `unsafe`-free, `#![no_std] + alloc` schema runtime primitives (`NG-W12-4C`).
///
/// Supports `#[repr(C)]` layout-verified structs, unaligned little-endian binary decoding
/// (without `transmute`), and zero-DOM streaming JSON decoding with memory budget enforcement.
pub mod schema {
    use crate::alloc_crate::string::String;
    use crate::alloc_crate::vec::Vec;
    use crate::{AotCtx, AotError};
    use core::fmt;

    /// 12-byte ObjectId with C-ABI 4-byte alignment matching `CanonicalTypeTag::ObjectId`.
    #[repr(C, align(4))]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct AotObjectId(pub [u8; 12]);

    const _: () = assert!(core::mem::size_of::<AotObjectId>() == 12);
    const _: () = assert!(core::mem::align_of::<AotObjectId>() == 4);

    impl AotObjectId {
        #[inline]
        pub const fn new(bytes: [u8; 12]) -> Self {
            Self(bytes)
        }

        #[inline]
        pub const fn as_bytes(&self) -> &[u8; 12] {
            &self.0
        }
    }

    impl fmt::Display for AotObjectId {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            let mut buf = [0u8; 24];
            let mut i = 0usize;
            while i < 12 {
                let b = self.0[i];
                buf[i * 2] = HEX[(b >> 4) as usize];
                buf[i * 2 + 1] = HEX[(b & 0x0f) as usize];
                i += 1;
            }
            if let Ok(s) = core::str::from_utf8(&buf) {
                f.write_str(s)
            } else {
                Ok(())
            }
        }
    }

    /// 16-byte fixed-width Decimal128 payload with C-ABI 16-byte alignment matching `CanonicalTypeTag::Decimal128`.
    #[repr(C, align(16))]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct AotDecimal128(pub i128);

    const _: () = assert!(core::mem::size_of::<AotDecimal128>() == 16);
    const _: () = assert!(core::mem::align_of::<AotDecimal128>() == 16);

    impl AotDecimal128 {
        #[inline]
        pub const fn new(raw: i128) -> Self {
            Self(raw)
        }

        #[inline]
        pub const fn raw(&self) -> i128 {
            self.0
        }
    }

    impl fmt::Display for AotDecimal128 {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    /// Verifies that `raw` contains at least `min_len` bytes.
    #[inline]
    pub fn check_binary_len(raw: &[u8], min_len: usize) -> Result<(), AotError> {
        if raw.len() < min_len {
            Err(AotError::SchemaDecodeError("truncated binary payload"))
        } else {
            Ok(())
        }
    }

    /// Checked offset addition for nested binary structs.
    #[inline]
    pub fn binary_offset(base: usize, rel: usize) -> Result<usize, AotError> {
        base.checked_add(rel)
            .ok_or(AotError::SchemaDecodeError("binary offset overflow"))
    }

    /// Reads a validated `bool` tag (`0` or `1`) at `offset` without alignment assumptions.
    #[inline]
    pub fn decode_bin_bool(raw: &[u8], offset: usize) -> Result<bool, AotError> {
        match raw.get(offset).copied() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            Some(_) => Err(AotError::SchemaDecodeError(
                "invalid bool tag in binary payload",
            )),
            None => Err(AotError::SchemaDecodeError("truncated binary payload")),
        }
    }

    /// Reads a validated `Option<bool>` tag (`0` = `Some(false)`, `1` = `Some(true)`, `2` = `None`).
    #[inline]
    pub fn decode_bin_opt_bool(raw: &[u8], offset: usize) -> Result<Option<bool>, AotError> {
        match raw.get(offset).copied() {
            Some(0) => Ok(Some(false)),
            Some(1) => Ok(Some(true)),
            Some(2) => Ok(None),
            Some(_) => Err(AotError::SchemaDecodeError(
                "invalid optional bool tag in binary payload",
            )),
            None => Err(AotError::SchemaDecodeError("truncated binary payload")),
        }
    }

    /// Reads a little-endian `i64` at `offset` using `from_le_bytes` (safe on any alignment).
    #[inline]
    pub fn decode_bin_i64(raw: &[u8], offset: usize) -> Result<i64, AotError> {
        let end = binary_offset(offset, 8)?;
        let slice = raw
            .get(offset..end)
            .ok_or(AotError::SchemaDecodeError("truncated binary payload"))?;
        let bytes: [u8; 8] = slice
            .try_into()
            .map_err(|_| AotError::SchemaDecodeError("truncated binary payload"))?;
        Ok(i64::from_le_bytes(bytes))
    }

    /// Reads a little-endian `u64` at `offset` using `from_le_bytes` (safe on any alignment).
    #[inline]
    pub fn decode_bin_u64(raw: &[u8], offset: usize) -> Result<u64, AotError> {
        let end = binary_offset(offset, 8)?;
        let slice = raw
            .get(offset..end)
            .ok_or(AotError::SchemaDecodeError("truncated binary payload"))?;
        let bytes: [u8; 8] = slice
            .try_into()
            .map_err(|_| AotError::SchemaDecodeError("truncated binary payload"))?;
        Ok(u64::from_le_bytes(bytes))
    }

    /// Reads a little-endian `f64` at `offset` using `from_le_bytes` (safe on any alignment).
    #[inline]
    pub fn decode_bin_f64(raw: &[u8], offset: usize) -> Result<f64, AotError> {
        let end = binary_offset(offset, 8)?;
        let slice = raw
            .get(offset..end)
            .ok_or(AotError::SchemaDecodeError("truncated binary payload"))?;
        let bytes: [u8; 8] = slice
            .try_into()
            .map_err(|_| AotError::SchemaDecodeError("truncated binary payload"))?;
        Ok(f64::from_le_bytes(bytes))
    }

    /// Reads a 12-byte [`AotObjectId`] at `offset` without alignment assumptions.
    #[inline]
    pub fn decode_bin_object_id(raw: &[u8], offset: usize) -> Result<AotObjectId, AotError> {
        let end = binary_offset(offset, 12)?;
        let slice = raw
            .get(offset..end)
            .ok_or(AotError::SchemaDecodeError("truncated binary payload"))?;
        let bytes: [u8; 12] = slice
            .try_into()
            .map_err(|_| AotError::SchemaDecodeError("truncated binary payload"))?;
        Ok(AotObjectId(bytes))
    }

    /// Reads a little-endian 16-byte [`AotDecimal128`] at `offset` using `i128::from_le_bytes`.
    #[inline]
    pub fn decode_bin_decimal128(raw: &[u8], offset: usize) -> Result<AotDecimal128, AotError> {
        let end = binary_offset(offset, 16)?;
        let slice = raw
            .get(offset..end)
            .ok_or(AotError::SchemaDecodeError("truncated binary payload"))?;
        let bytes: [u8; 16] = slice
            .try_into()
            .map_err(|_| AotError::SchemaDecodeError("truncated binary payload"))?;
        Ok(AotDecimal128(i128::from_le_bytes(bytes)))
    }

    #[inline]
    fn read_slice_header(raw: &[u8], offset: usize) -> Result<(usize, usize, u64), AotError> {
        let data_off_u64 = decode_bin_u64(raw, offset)?;
        let data_len_u64 = decode_bin_u64(raw, binary_offset(offset, 8)?)?;
        let tag = decode_bin_u64(raw, binary_offset(offset, 16)?)?;
        let data_off = usize::try_from(data_off_u64)
            .map_err(|_| AotError::SchemaDecodeError("binary slice offset overflow"))?;
        let data_len = usize::try_from(data_len_u64)
            .map_err(|_| AotError::SchemaDecodeError("binary slice length overflow"))?;
        Ok((data_off, data_len, tag))
    }

    #[inline]
    fn alloc_string_from_str(text: &str, ctx: &mut AotCtx) -> Result<String, AotError> {
        let len = text.len();
        if len == 0 {
            return Ok(String::new());
        }
        ctx.try_charge_memory(len)?;
        let mut out = String::new();
        if out.try_reserve_exact(len).is_err() {
            ctx.budget_alloc.refund(len);
            return Err(AotError::from_alloc_failure(&ctx.budget_alloc, len));
        }
        out.push_str(text);
        Ok(out)
    }

    #[inline]
    fn alloc_bytes_from_slice(bytes: &[u8], ctx: &mut AotCtx) -> Result<Vec<u8>, AotError> {
        let len = bytes.len();
        if len == 0 {
            return Ok(Vec::new());
        }
        ctx.try_charge_memory(len)?;
        let mut out = Vec::new();
        if out.try_reserve_exact(len).is_err() {
            ctx.budget_alloc.refund(len);
            return Err(AotError::from_alloc_failure(&ctx.budget_alloc, len));
        }
        out.extend_from_slice(bytes);
        Ok(out)
    }

    /// Reads a 24-byte `(data_offset: u64, data_len: u64, owner_tag: u64)` UTF-8 string descriptor
    /// at `offset`, validates bounds & UTF-8, and charges `ctx.budget_alloc`.
    pub fn decode_bin_string(
        raw: &[u8],
        offset: usize,
        ctx: &mut AotCtx,
    ) -> Result<String, AotError> {
        let (data_off, data_len, _tag) = read_slice_header(raw, offset)?;
        let end = binary_offset(data_off, data_len)?;
        let slice = raw
            .get(data_off..end)
            .ok_or(AotError::SchemaDecodeError("binary string slice out of bounds"))?;
        let text = core::str::from_utf8(slice).map_err(|_| {
            AotError::SchemaDecodeError("invalid utf-8 in binary string slice")
        })?;
        alloc_string_from_str(text, ctx)
    }

    /// Reads an optional 24-byte UTF-8 string descriptor (`tag == 0 => None`, `tag == 1 => Some`).
    pub fn decode_bin_opt_string(
        raw: &[u8],
        offset: usize,
        ctx: &mut AotCtx,
    ) -> Result<Option<String>, AotError> {
        let (data_off, data_len, tag) = read_slice_header(raw, offset)?;
        match tag {
            0 => {
                if data_off != 0 || data_len != 0 {
                    return Err(AotError::SchemaDecodeError(
                        "non-zero offset or length for null string slice",
                    ));
                }
                Ok(None)
            }
            1 => {
                let end = binary_offset(data_off, data_len)?;
                let slice = raw.get(data_off..end).ok_or(AotError::SchemaDecodeError(
                    "binary string slice out of bounds",
                ))?;
                let text = core::str::from_utf8(slice).map_err(|_| {
                    AotError::SchemaDecodeError("invalid utf-8 in binary string slice")
                })?;
                Ok(Some(alloc_string_from_str(text, ctx)?))
            }
            _ => Err(AotError::SchemaDecodeError(
                "invalid optional string slice tag in binary payload",
            )),
        }
    }

    /// Reads a 24-byte `(data_offset: u64, data_len: u64, owner_tag: u64)` binary slice descriptor
    /// at `offset`, validates bounds, and charges `ctx.budget_alloc`.
    pub fn decode_bin_bytes(
        raw: &[u8],
        offset: usize,
        ctx: &mut AotCtx,
    ) -> Result<Vec<u8>, AotError> {
        let (data_off, data_len, _tag) = read_slice_header(raw, offset)?;
        let end = binary_offset(data_off, data_len)?;
        let slice = raw
            .get(data_off..end)
            .ok_or(AotError::SchemaDecodeError("binary slice out of bounds"))?;
        alloc_bytes_from_slice(slice, ctx)
    }

    /// Reads an optional 24-byte binary slice descriptor (`tag == 0 => None`, `tag == 1 => Some`).
    pub fn decode_bin_opt_bytes(
        raw: &[u8],
        offset: usize,
        ctx: &mut AotCtx,
    ) -> Result<Option<Vec<u8>>, AotError> {
        let (data_off, data_len, tag) = read_slice_header(raw, offset)?;
        match tag {
            0 => {
                if data_off != 0 || data_len != 0 {
                    return Err(AotError::SchemaDecodeError(
                        "non-zero offset or length for null binary slice",
                    ));
                }
                Ok(None)
            }
            1 => {
                let end = binary_offset(data_off, data_len)?;
                let slice = raw
                    .get(data_off..end)
                    .ok_or(AotError::SchemaDecodeError("binary slice out of bounds"))?;
                Ok(Some(alloc_bytes_from_slice(slice, ctx)?))
            }
            _ => Err(AotError::SchemaDecodeError(
                "invalid optional binary slice tag in binary payload",
            )),
        }
    }

    /// Zero-allocation JSON object key wrapper (borrows unescaped keys directly from input `&str`).
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum JsonKey<'a> {
        Borrowed(&'a str),
        Owned(String),
    }

    impl<'a> JsonKey<'a> {
        #[inline]
        pub fn as_str(&self) -> &str {
            match self {
                Self::Borrowed(s) => s,
                Self::Owned(s) => s.as_str(),
            }
        }
    }

    #[inline]
    fn hex_nibble(b: u8) -> Option<u8> {
        match b {
            b'0'..=b'9' => Some(b - b'0'),
            b'a'..=b'f' => Some(b - b'a' + 10),
            b'A'..=b'F' => Some(b - b'A' + 10),
            _ => None,
        }
    }

    #[inline]
    fn parse_hex4(slice: &[u8]) -> Option<u16> {
        if slice.len() < 4 {
            return None;
        }
        let a = hex_nibble(slice[0])? as u16;
        let b = hex_nibble(slice[1])? as u16;
        let c = hex_nibble(slice[2])? as u16;
        let d = hex_nibble(slice[3])? as u16;
        Some((a << 12) | (b << 8) | (c << 4) | d)
    }

    #[inline]
    fn is_ident_continue(b: u8) -> bool {
        b.is_ascii_alphanumeric() || b == b'_'
    }

    /// Zero-DOM, panic-free, `#![no_std] + alloc` streaming JSON cursor for direct schema decoding.
    pub struct JsonCursor<'a> {
        src: &'a str,
        bytes: &'a [u8],
        pos: usize,
    }

    impl<'a> JsonCursor<'a> {
        #[inline]
        pub fn new(src: &'a str) -> Self {
            Self {
                src,
                bytes: src.as_bytes(),
                pos: 0,
            }
        }

        #[inline]
        pub fn skip_ws(&mut self) {
            while let Some(&b) = self.bytes.get(self.pos) {
                if matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
                    self.pos += 1;
                } else {
                    break;
                }
            }
        }

        #[inline]
        pub fn peek(&self) -> Option<u8> {
            self.bytes.get(self.pos).copied()
        }

        #[inline]
        pub fn consume_byte(&mut self, expected: u8) -> bool {
            self.skip_ws();
            if self.peek() == Some(expected) {
                self.pos += 1;
                true
            } else {
                false
            }
        }

        #[inline]
        pub fn expect_byte(&mut self, expected: u8) -> Result<(), AotError> {
            if self.consume_byte(expected) {
                Ok(())
            } else {
                Err(AotError::SchemaDecodeError(match expected {
                    b'{' => "expected '{' in json object",
                    b'}' => "expected '}' in json object",
                    b':' => "expected ':' after json key",
                    b',' => "expected ',' between json fields",
                    _ => "unexpected json token",
                }))
            }
        }

        #[inline]
        pub fn expect_eof(&mut self) -> Result<(), AotError> {
            self.skip_ws();
            if self.pos == self.bytes.len() {
                Ok(())
            } else {
                Err(AotError::SchemaDecodeError(
                    "trailing junk after json object",
                ))
            }
        }

        /// Consumes `null` if present at the current position (after whitespace).
        #[inline]
        pub fn consume_null(&mut self) -> bool {
            self.skip_ws();
            let rest = self.bytes.get(self.pos..).unwrap_or(&[]);
            if rest.starts_with(b"null")
                && !self
                    .bytes
                    .get(self.pos + 4)
                    .copied()
                    .is_some_and(is_ident_continue)
            {
                self.pos += 4;
                true
            } else {
                false
            }
        }

        /// Scans a JSON string literal, returning `(raw_inner_slice, has_escape)`.
        fn scan_raw_string(&mut self) -> Result<(&'a str, bool), AotError> {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(AotError::SchemaDecodeError("expected json string"));
            }
            self.pos += 1;
            let start = self.pos;
            let mut has_escape = false;
            while let Some(&b) = self.bytes.get(self.pos) {
                if b == b'"' {
                    let end = self.pos;
                    self.pos += 1;
                    let raw = self.src.get(start..end).ok_or(AotError::SchemaDecodeError(
                        "invalid utf-8 boundary in json string",
                    ))?;
                    return Ok((raw, has_escape));
                } else if b == b'\\' {
                    has_escape = true;
                    self.pos += 1;
                    if self.pos >= self.bytes.len() {
                        return Err(AotError::SchemaDecodeError("unterminated json escape"));
                    }
                    self.pos += 1;
                } else if b < 0x20 {
                    return Err(AotError::SchemaDecodeError(
                        "unescaped control character in json string",
                    ));
                } else {
                    self.pos += 1;
                }
            }
            Err(AotError::SchemaDecodeError("unterminated json string"))
        }

        fn unescape_json_str(raw: &str, ctx: &mut AotCtx) -> Result<String, AotError> {
            let bytes = raw.as_bytes();
            let cap = bytes.len();
            if cap > 0 {
                ctx.try_charge_memory(cap)?;
            }
            let mut out = Vec::new();
            if cap > 0 && out.try_reserve_exact(cap).is_err() {
                ctx.budget_alloc.refund(cap);
                return Err(AotError::from_alloc_failure(&ctx.budget_alloc, cap));
            }
            let mut i = 0usize;
            while let Some(&b) = bytes.get(i) {
                if b == b'\\' {
                    i += 1;
                    let esc = bytes
                        .get(i)
                        .copied()
                        .ok_or(AotError::SchemaDecodeError("incomplete json escape"))?;
                    match esc {
                        b'"' => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'/' => out.push(b'/'),
                        b'b' => out.push(0x08),
                        b'f' => out.push(0x0c),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'u' => {
                            let hex_slice = bytes
                                .get(i + 1..i + 5)
                                .ok_or(AotError::SchemaDecodeError("invalid \\u escape"))?;
                            let code = parse_hex4(hex_slice)
                                .ok_or(AotError::SchemaDecodeError("invalid \\u hex"))?;
                            i += 4;
                            let scalar: u32 = if (0xD800..=0xDBFF).contains(&code) {
                                // High surrogate: must be followed by `\uXXXX` low surrogate
                                let next_esc = bytes.get(i + 1..i + 3);
                                if next_esc != Some(b"\\u") {
                                    return Err(AotError::SchemaDecodeError(
                                        "unpaired utf-16 high surrogate",
                                    ));
                                }
                                let lo_slice = bytes.get(i + 3..i + 7).ok_or(
                                    AotError::SchemaDecodeError("invalid surrogate \\u escape"),
                                )?;
                                let lo = parse_hex4(lo_slice).ok_or(
                                    AotError::SchemaDecodeError("invalid surrogate \\u hex"),
                                )?;
                                if !(0xDC00..=0xDFFF).contains(&lo) {
                                    return Err(AotError::SchemaDecodeError(
                                        "invalid utf-16 low surrogate",
                                    ));
                                }
                                i += 6;
                                0x10000 + ((((code as u32) - 0xD800) << 10) | ((lo as u32) - 0xDC00))
                            } else if (0xDC00..=0xDFFF).contains(&code) {
                                return Err(AotError::SchemaDecodeError(
                                    "unexpected utf-16 low surrogate",
                                ));
                            } else {
                                code as u32
                            };
                            let ch = char::from_u32(scalar)
                                .ok_or(AotError::SchemaDecodeError("invalid unicode scalar"))?;
                            let mut utf8_buf = [0u8; 4];
                            let encoded = ch.encode_utf8(&mut utf8_buf).as_bytes();
                            out.extend_from_slice(encoded);
                        }
                        _ => {
                            return Err(AotError::SchemaDecodeError("unsupported json escape"));
                        }
                    }
                    i += 1;
                } else {
                    out.push(b);
                    i += 1;
                }
            }
            String::from_utf8(out)
                .map_err(|_| AotError::SchemaDecodeError("invalid utf-8 after json unescape"))
        }

        /// Parses a JSON object key with a zero-allocation fast path for unescaped keys.
        #[inline]
        pub fn parse_key(&mut self, ctx: &mut AotCtx) -> Result<JsonKey<'a>, AotError> {
            let (raw, has_escape) = self.scan_raw_string()?;
            if !has_escape {
                Ok(JsonKey::Borrowed(raw))
            } else {
                Ok(JsonKey::Owned(Self::unescape_json_str(raw, ctx)?))
            }
        }

        /// Parses an owned JSON string value with memory budget enforcement.
        pub fn parse_string(&mut self, ctx: &mut AotCtx) -> Result<String, AotError> {
            let (raw, has_escape) = self.scan_raw_string()?;
            if !has_escape {
                alloc_string_from_str(raw, ctx)
            } else {
                Self::unescape_json_str(raw, ctx)
            }
        }

        /// Parses a JSON string into owned bytes with memory budget enforcement.
        pub fn parse_bytes(&mut self, ctx: &mut AotCtx) -> Result<Vec<u8>, AotError> {
            Ok(self.parse_string(ctx)?.into_bytes())
        }

        /// Parses a JSON boolean (`true` or `false`) with strict delimiter validation.
        pub fn parse_bool(&mut self) -> Result<bool, AotError> {
            self.skip_ws();
            let rest = self.bytes.get(self.pos..).unwrap_or(&[]);
            if rest.starts_with(b"true")
                && !self
                    .bytes
                    .get(self.pos + 4)
                    .copied()
                    .is_some_and(is_ident_continue)
            {
                self.pos += 4;
                Ok(true)
            } else if rest.starts_with(b"false")
                && !self
                    .bytes
                    .get(self.pos + 5)
                    .copied()
                    .is_some_and(is_ident_continue)
            {
                self.pos += 5;
                Ok(false)
            } else {
                Err(AotError::SchemaDecodeError("expected json boolean"))
            }
        }

        fn scan_number_token(&mut self) -> Result<(&'a str, bool), AotError> {
            self.skip_ws();
            let start = self.pos;
            if self.peek() == Some(b'-') {
                self.pos += 1;
            }
            let int_start = self.pos;
            let mut has_digit = false;
            while let Some(b) = self.peek() {
                if b.is_ascii_digit() {
                    has_digit = true;
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if !has_digit {
                return Err(AotError::SchemaDecodeError("expected json number"));
            }
            // Reject RFC 8259 forbidden leading zeroes like `01` or `-01`
            if self.pos - int_start > 1 && self.bytes.get(int_start) == Some(&b'0') {
                return Err(AotError::SchemaDecodeError("leading zero in json number"));
            }
            let mut has_frac_or_exp = false;
            if self.peek() == Some(b'.') {
                has_frac_or_exp = true;
                self.pos += 1;
                let mut frac_digits = false;
                while let Some(b) = self.peek() {
                    if b.is_ascii_digit() {
                        frac_digits = true;
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                if !frac_digits {
                    return Err(AotError::SchemaDecodeError(
                        "missing fractional digits in json number",
                    ));
                }
            }
            if matches!(self.peek(), Some(b'e' | b'E')) {
                has_frac_or_exp = true;
                self.pos += 1;
                if matches!(self.peek(), Some(b'+' | b'-')) {
                    self.pos += 1;
                }
                let mut exp_digits = false;
                while let Some(b) = self.peek() {
                    if b.is_ascii_digit() {
                        exp_digits = true;
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                if !exp_digits {
                    return Err(AotError::SchemaDecodeError(
                        "missing exponent digits in json number",
                    ));
                }
            }
            let tok = self
                .src
                .get(start..self.pos)
                .ok_or(AotError::SchemaDecodeError("invalid json number slice"))?;
            Ok((tok, has_frac_or_exp))
        }

        /// Parses a strict JSON `i64` (rejects fractional/exponent syntax and 64-bit overflow).
        pub fn parse_i64(&mut self) -> Result<i64, AotError> {
            let (tok, has_frac_or_exp) = self.scan_number_token()?;
            if has_frac_or_exp {
                return Err(AotError::SchemaDecodeError(
                    "type mismatch: expected integer, got float",
                ));
            }
            tok.parse::<i64>()
                .map_err(|_| AotError::SchemaDecodeError("int64 width overflow"))
        }

        /// Parses a strict JSON `u64` (rejects negative, fractional/exponent syntax, and overflow).
        pub fn parse_u64(&mut self) -> Result<u64, AotError> {
            let (tok, has_frac_or_exp) = self.scan_number_token()?;
            if has_frac_or_exp {
                return Err(AotError::SchemaDecodeError(
                    "type mismatch: expected uint64, got float",
                ));
            }
            tok.parse::<u64>()
                .map_err(|_| AotError::SchemaDecodeError("uint64 width overflow"))
        }

        /// Parses a strict JSON finite `f64` (rejects overflow to infinity/NaN).
        pub fn parse_f64(&mut self) -> Result<f64, AotError> {
            let (tok, _) = self.scan_number_token()?;
            let val = tok
                .parse::<f64>()
                .map_err(|_| AotError::SchemaDecodeError("float64 parse error"))?;
            if !val.is_finite() {
                return Err(AotError::SchemaDecodeError("float64 width overflow"));
            }
            Ok(val)
        }

        /// Parses a 16-byte [`AotDecimal128`] integer token (or quoted integer string).
        pub fn parse_decimal128(&mut self) -> Result<AotDecimal128, AotError> {
            self.skip_ws();
            if self.peek() == Some(b'"') {
                let (raw, has_escape) = self.scan_raw_string()?;
                if has_escape {
                    return Err(AotError::SchemaDecodeError("invalid decimal128 string"));
                }
                let v = raw
                    .parse::<i128>()
                    .map_err(|_| AotError::SchemaDecodeError("decimal128 width overflow"))?;
                return Ok(AotDecimal128(v));
            }
            let (tok, has_frac_or_exp) = self.scan_number_token()?;
            if has_frac_or_exp {
                return Err(AotError::SchemaDecodeError(
                    "type mismatch: expected decimal128 integer",
                ));
            }
            let v = tok
                .parse::<i128>()
                .map_err(|_| AotError::SchemaDecodeError("decimal128 width overflow"))?;
            Ok(AotDecimal128(v))
        }

        /// Parses a 24-hex-char JSON string into a 12-byte [`AotObjectId`] with zero allocation.
        pub fn parse_object_id(&mut self) -> Result<AotObjectId, AotError> {
            let (raw, has_escape) = self.scan_raw_string()?;
            let bytes = raw.as_bytes();
            if has_escape || bytes.len() != 24 {
                return Err(AotError::SchemaDecodeError(
                    "expected 24-character hex string for ObjectId",
                ));
            }
            let mut out = [0u8; 12];
            let mut i = 0usize;
            while i < 12 {
                let hi = hex_nibble(bytes[i * 2]).ok_or(AotError::SchemaDecodeError(
                    "invalid hex digit in ObjectId",
                ))?;
                let lo = hex_nibble(bytes[i * 2 + 1]).ok_or(AotError::SchemaDecodeError(
                    "invalid hex digit in ObjectId",
                ))?;
                out[i] = (hi << 4) | lo;
                i += 1;
            }
            Ok(AotObjectId(out))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc_crate::boxed::Box;
    use crate::alloc_crate::vec;
    use crate::alloc_crate::vec::Vec;
    use core::cell::RefCell;

    #[test]
    fn test_aot_ctx_gas_charging_and_oog() {
        let mut ctx = AotCtx::with_gas_limit(100);
        assert_eq!(ctx.gas_remaining, 100);
        assert_eq!(ctx.gas_used(), 0);

        // Non-positive charges are no-ops
        assert!(ctx.charge_gas(0).is_ok());
        assert!(ctx.charge_gas(-50).is_ok());
        assert_eq!(ctx.gas_remaining, 100);

        assert!(ctx.charge_gas(60).is_ok());
        assert_eq!(ctx.gas_remaining, 40);
        assert_eq!(ctx.gas_used(), 60);

        // Exact boundary charge succeeds
        assert!(ctx.charge_gas(40).is_ok());
        assert_eq!(ctx.gas_remaining, 0);
        assert_eq!(ctx.gas_used(), 100);

        // Exceeding by 1 triggers OutOfGas
        let err = ctx.charge_gas(1).unwrap_err();
        assert_eq!(
            err,
            AotError::OutOfGas {
                limit: 100,
                used: 101
            }
        );
        assert_eq!(err.error_class(), "OutOfGas");
        assert_eq!(ctx.gas_used(), 101);

        // Subsequent charges also fail closed
        let err2 = ctx.charge_gas(5).unwrap_err();
        assert_eq!(
            err2,
            AotError::OutOfGas {
                limit: 100,
                used: 106
            }
        );

        // Unbounded context never trips OOG
        let mut unbounded = AotCtx::new();
        assert!(unbounded.charge_gas(i64::MAX).is_ok());
        assert_eq!(unbounded.gas_used(), 0);

        // PureCore denies FFI capabilities
        let pure = AotCtx::pure_core();
        assert!(pure.pure_core && pure.deny_ffi);
        assert_eq!(
            pure.check_capability("fs.read"),
            Err(AotError::CapabilityDenied("fs.read"))
        );
    }

    #[test]
    fn test_infallible_conversion_to_aot_error() {
        fn pure_fn(x: i64) -> Result<i64, Infallible> {
            Ok(x.wrapping_add(1))
        }
        fn caller(x: i64) -> Result<i64, AotError> {
            let v = pure_fn(x)?;
            Ok(v)
        }
        assert_eq!(caller(41), Ok(42));
    }

    #[test]
    fn test_integer_intrinsics_parity() {
        // div_int
        assert_eq!(div_int(10, 3), Ok(3));
        assert_eq!(div_int(-10, 3), Ok(-3));
        assert_eq!(div_int(i64::MIN, -1), Ok(i64::MIN));
        assert_eq!(div_int(10, 0), Err(AotError::DivisionByZero));

        // rem_int
        assert_eq!(rem_int(10, 3), Ok(1));
        assert_eq!(rem_int(-10, 3), Ok(-1));
        assert_eq!(rem_int(7, -2), Ok(1));
        assert_eq!(rem_int(i64::MIN, -1), Ok(0));
        assert_eq!(rem_int(10, 0), Err(AotError::DivisionByZero));

        // floordiv_int
        assert_eq!(floordiv_int(7, 2), Ok(3));
        assert_eq!(floordiv_int(-7, 2), Ok(-4));
        assert_eq!(floordiv_int(7, -2), Ok(-4));
        assert_eq!(floordiv_int(-7, -2), Ok(3));
        assert_eq!(floordiv_int(-6, 2), Ok(-3));
        assert_eq!(floordiv_int(i64::MIN, -1), Ok(i64::MIN));
        assert_eq!(floordiv_int(5, 0), Err(AotError::DivisionByZero));

        // pow_int
        assert_eq!(pow_int(0, 0), Ok(1));
        assert_eq!(pow_int(2, 10), Ok(1024));
        assert_eq!(pow_int(-2, 3), Ok(-8));
        assert_eq!(pow_int(-2, 4), Ok(16));
        assert_eq!(pow_int(2, 62), Ok(1i64 << 62));
        assert_eq!(pow_int(2, -1), Err(AotError::NegativeExponent(-1)));
        assert_eq!(
            pow_int(2, i64::MIN),
            Err(AotError::NegativeExponent(i64::MIN))
        );
    }

    #[test]
    fn test_string_append_int_and_format_float() {
        let mut s = String::new();
        for d in 0..=9 {
            string_append_int(&mut s, d);
        }
        assert_eq!(s, "0123456789");

        s.clear();
        string_append_int(&mut s, -42);
        s.push(',');
        string_append_int(&mut s, i64::MAX);
        s.push(',');
        string_append_int(&mut s, i64::MIN);
        assert_eq!(s, "-42,9223372036854775807,-9223372036854775808");

        assert_eq!(format_float(0.0), "0");
        assert_eq!(format_float(1.5), "1.5");
        assert_eq!(format_float(166666.66666658907), "166666.66666658907");
        assert_eq!(format_float(f64::NAN), "NaN");
        assert_eq!(format_float(f64::INFINITY), "inf");
        assert_eq!(format_float(f64::NEG_INFINITY), "-inf");

        // Budgeted string append
        let alloc = LustBudgetAlloc::with_limit(8);
        let mut bounded = String::new();
        assert!(try_string_append_int(&mut bounded, 1234, &alloc).is_ok());
        assert_eq!(bounded, "1234");
        assert!(matches!(
            try_string_append_int(&mut bounded, i64::MIN, &alloc),
            Err(AotError::OutOfMemory { .. })
        ));
    }

    #[test]
    fn test_defer_and_errdefer_lifo_and_dismiss() {
        let log: RefCell<Vec<&'static str>> = RefCell::new(Vec::new());

        let ok_path = || -> Result<i64, AotError> {
            let _d1 = defer(|| log.borrow_mut().push("defer_1"));
            let e1 = errdefer(|| log.borrow_mut().push("errdefer_1"));
            let _d2 = defer(|| log.borrow_mut().push("defer_2"));
            DropGuard::dismiss(e1);
            Ok(7)
        };
        assert_eq!(ok_path(), Ok(7));
        assert_eq!(*log.borrow(), vec!["defer_2", "defer_1"]);

        log.borrow_mut().clear();

        let err_path = || -> Result<i64, AotError> {
            let _d1 = defer(|| log.borrow_mut().push("defer_1"));
            let _e1 = errdefer(|| log.borrow_mut().push("errdefer_1"));
            let _d2 = defer(|| log.borrow_mut().push("defer_2"));
            let _ = div_int(1, 0)?;
            DropGuard::dismiss(_e1);
            Ok(99)
        };
        assert_eq!(err_path(), Err(AotError::DivisionByZero));
        assert_eq!(*log.borrow(), vec!["defer_2", "errdefer_1", "defer_1"]);
    }

    #[test]
    fn test_allocators_zst_oom_arena_and_slab() {
        // LustBudgetAlloc ZST + OOM rollback
        let alloc = LustBudgetAlloc::with_limit(1024);
        let mut zvec: Vec<(), &LustBudgetAlloc> = Vec::with_capacity_in(64, &alloc);
        for _ in 0..64 {
            zvec.push(());
        }
        assert_eq!(alloc.used_bytes(), 0);
        drop(zvec);

        {
            let mut v: Vec<u8, &LustBudgetAlloc> = Vec::new_in(&alloc);
            assert!(v.try_reserve_exact(512).is_ok());
            assert_eq!(alloc.used_bytes(), 512);
            for i in 0..512usize {
                v.push((i & 0xFF) as u8);
            }
            assert!(v.try_reserve_exact(513).is_err());
            assert_eq!(alloc.used_bytes(), 512);
            assert_eq!(v[511], 255);
        }
        assert_eq!(alloc.used_bytes(), 0);
        assert_eq!(alloc.peak_bytes(), 512);

        // LustChunkArena bulk free
        let parent = LustBudgetAlloc::with_limit(DEFAULT_CHUNK_SIZE * 2);
        {
            let arena = LustChunkArena::new(&parent);
            let b: Box<[u64; 8], &LustChunkArena<'_>> = Box::new_in([42u64; 8], &arena);
            assert_eq!(b[7], 42);
            assert_eq!(parent.used_bytes(), DEFAULT_CHUNK_SIZE);
            drop(b);
            assert_eq!(parent.used_bytes(), DEFAULT_CHUNK_SIZE);
        }
        assert_eq!(parent.used_bytes(), 0);

        // StaticSlabAllocator tests (including unsafe reset) live in alloc.rs to keep lib.rs 100% unsafe-free.
    }

    #[test]
    fn test_aot_value_containers_closures_and_cycles() {
        let mut ctx = AotCtx::with_memory_limit(1024);

        let arr = AotValue::empty_array();
        assert!(arr.array_push(AotValue::Int(10), &ctx).is_ok());
        assert!(arr.array_push(AotValue::str("hi"), &ctx).is_ok());
        assert_eq!(arr.array_get(0), Ok(AotValue::Int(10)));
        assert_eq!(arr.array_get(1), Ok(AotValue::str("hi")));
        assert_eq!(
            arr.array_get(2),
            Err(AotError::IndexOutOfBounds { index: 2, len: 2 })
        );
        assert_eq!(
            arr.array_get(-1),
            Err(AotError::IndexOutOfBounds { index: -1, len: 2 })
        );

        let map = AotValue::map();
        assert!(map.map_set(AotKey::from("k"), AotValue::Int(99), &ctx).is_ok());
        assert_eq!(map.map_get(&AotKey::from("k")), Ok(AotValue::Int(99)));
        assert_eq!(
            map.map_get(&AotKey::from("missing")),
            Err(AotError::MissingMapKey)
        );

        // Closure call charging gas
        let f = AotValue::closure(|c, args| {
            c.charge_gas(1)?;
            let x = args.first().and_then(AotValue::as_int).unwrap_or(0);
            Ok(AotValue::Int(x * 2))
        });
        assert_eq!(f.call(&mut ctx, &[AotValue::Int(21)]), Ok(AotValue::Int(42)));

        // Self-referential array does not panic in Display / Debug / PartialEq
        let cyclic = AotValue::empty_array();
        assert!(cyclic.array_push(cyclic.clone(), &ctx).is_ok());
        let displayed = crate::alloc_crate::format!("{cyclic}");
        assert!(displayed.contains("[...]"));
        assert_eq!(cyclic, cyclic);
    }

    #[test]
    fn test_aot_map_and_ref_borrow_helpers_and_dynamic_ops() {
        let alloc = LustBudgetAlloc::with_limit(256);
        let m1: AotMap<i64, i64> = AotMap::new();
        let m2 = m1.clone();
        assert!(m1.set(1, 100, &alloc).is_ok());
        assert!(m1.set(2, 200, &alloc).is_ok());
        assert_eq!(m2.get(&1), Ok(100));
        assert_eq!(m2.get_opt(&3), Ok(None));
        assert_eq!(m2.get(&3), Err(AotError::MissingMapKey));
        assert_eq!(m2.has(&2), Ok(true));
        assert_eq!(m2.len(), Ok(2));
        assert_eq!(m2.delete(&1, &alloc), Ok(Some(100)));
        assert_eq!(m1.len(), Ok(1));
        let keys = m1.keys_in(&alloc).expect("keys_in");
        assert_eq!(&keys[..], &[2]);
        let vals = m1.values_in(&alloc).expect("values_in");
        assert_eq!(&vals[..], &[200]);

        // Re-entrant RefCell borrow returns Err(AotError::UserError) instead of panicking
        let cell = RefCell::new(10_i64);
        let _guard = ref_borrow(&cell).expect("shared borrow");
        assert!(matches!(ref_borrow_mut(&cell), Err(AotError::UserError(_))));
        drop(_guard);
        *ref_borrow_mut(&cell).expect("mut borrow") = 20;
        assert_eq!(*ref_borrow(&cell).expect("shared borrow 2"), 20);

        // Dynamic AotValue arithmetic & enum accessors
        assert_eq!(
            AotValue::Int(10).add(&AotValue::Int(32)),
            Ok(AotValue::Int(42))
        );
        assert_eq!(
            AotValue::Int(10).div(&AotValue::Int(0)),
            Err(AotError::DivisionByZero)
        );
        let en = AotValue::from_enum(1, 2, vec![AotValue::Int(77)]);
        assert_eq!(en.enum_tag(), Ok(2));
        assert_eq!(en.enum_payload(0), Ok(AotValue::Int(77)));
        assert!(matches!(
            en.enum_payload(1),
            Err(AotError::IndexOutOfBounds { index: 1, len: 1 })
        ));
    }

    #[test]
    fn test_schema_primitives_binary_unaligned_and_json_cursor() {
        use crate::schema::{
            AotDecimal128, AotObjectId, JsonCursor, check_binary_len, decode_bin_bool,
            decode_bin_bytes, decode_bin_decimal128, decode_bin_f64, decode_bin_i64,
            decode_bin_object_id, decode_bin_opt_bool, decode_bin_opt_bytes, decode_bin_opt_string,
            decode_bin_string, decode_bin_u64,
        };

        assert_eq!(core::mem::size_of::<AotObjectId>(), 12);
        assert_eq!(core::mem::align_of::<AotObjectId>(), 4);
        assert_eq!(core::mem::size_of::<AotDecimal128>(), 16);
        assert_eq!(core::mem::align_of::<AotDecimal128>(), 16);

        let oid = AotObjectId([0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb]);
        assert_eq!(
            crate::alloc_crate::format!("{oid}"),
            "00112233445566778899aabb"
        );
        let dec = AotDecimal128(12345678901234567890_i128);
        assert_eq!(
            crate::alloc_crate::format!("{dec}"),
            "12345678901234567890"
        );

        // Unaligned binary buffer (starts at odd index 1)
        let mut backing = vec![0xAAu8; 1 + 128];
        let buf = &mut backing[1..];
        buf[0] = 1; // bool true
        buf[1] = 2; // opt bool None
        buf[8..16].copy_from_slice(&(-77_i64).to_le_bytes());
        buf[16..24].copy_from_slice(&999_u64.to_le_bytes());
        buf[24..32].copy_from_slice(&(42.5_f64).to_le_bytes());
        buf[32..44].copy_from_slice(&oid.0);
        buf[48..64].copy_from_slice(&dec.0.to_le_bytes());
        // String slice header at 64: offset=112, len=5, tag=1 ("hello")
        buf[64..72].copy_from_slice(&112_u64.to_le_bytes());
        buf[72..80].copy_from_slice(&5_u64.to_le_bytes());
        buf[80..88].copy_from_slice(&1_u64.to_le_bytes());
        buf[112..117].copy_from_slice(b"hello");
        // Optional binary slice header at 88: offset=120, len=4, tag=1 (non-UTF-8 bytes)
        buf[88..96].copy_from_slice(&120_u64.to_le_bytes());
        buf[96..104].copy_from_slice(&4_u64.to_le_bytes());
        buf[104..112].copy_from_slice(&1_u64.to_le_bytes());
        buf[120..124].copy_from_slice(&[0xFF, 0x00, 0xFE, 0x80]);

        let mut ctx = AotCtx::new();
        assert!(check_binary_len(buf, 112).is_ok());
        assert!(matches!(
            check_binary_len(&buf[..4], 16),
            Err(AotError::SchemaDecodeError(_))
        ));
        assert_eq!(decode_bin_bool(buf, 0), Ok(true));
        assert_eq!(decode_bin_opt_bool(buf, 1), Ok(None));
        assert_eq!(decode_bin_i64(buf, 8), Ok(-77));
        assert_eq!(decode_bin_u64(buf, 16), Ok(999));
        assert_eq!(decode_bin_f64(buf, 24), Ok(42.5));
        assert_eq!(decode_bin_object_id(buf, 32), Ok(oid));
        assert_eq!(decode_bin_decimal128(buf, 48), Ok(dec));
        assert_eq!(decode_bin_string(buf, 64, &mut ctx), Ok(String::from("hello")));
        assert_eq!(
            decode_bin_opt_string(buf, 64, &mut ctx),
            Ok(Some(String::from("hello")))
        );
        assert_eq!(
            decode_bin_bytes(buf, 88, &mut ctx),
            Ok(vec![0xFF, 0x00, 0xFE, 0x80])
        );
        assert_eq!(
            decode_bin_opt_bytes(buf, 88, &mut ctx),
            Ok(Some(vec![0xFF, 0x00, 0xFE, 0x80]))
        );

        // Invalid bool & option tags fail closed
        buf[0] = 2;
        assert!(matches!(
            decode_bin_bool(buf, 0),
            Err(AotError::SchemaDecodeError(_))
        ));
        buf[80..88].copy_from_slice(&7_u64.to_le_bytes());
        assert!(matches!(
            decode_bin_opt_string(buf, 64, &mut ctx),
            Err(AotError::SchemaDecodeError(_))
        ));

        // Tight memory limit enforces OOM on binary string decode without panicking
        let mut tiny_ctx = AotCtx::with_memory_limit(2);
        assert!(matches!(
            decode_bin_string(buf, 64, &mut tiny_ctx),
            Err(AotError::OutOfMemory { .. })
        ));

        // JsonCursor escapes, surrogate pairs, numbers, overflow, trailing junk
        let mut cur = JsonCursor::new(r#""a\nb\t\"\\\/\u0041\uD83D\uDE00""#);
        let parsed = cur.parse_string(&mut ctx).expect("parse escaped string");
        assert_eq!(parsed, "a\nb\t\"\\/A😀");
        assert!(cur.expect_eof().is_ok());

        let mut c_ovf = JsonCursor::new("9223372036854775808");
        assert!(matches!(
            c_ovf.parse_i64(),
            Err(AotError::SchemaDecodeError(_))
        ));

        let mut c_neg_u = JsonCursor::new("-1");
        assert!(matches!(
            c_neg_u.parse_u64(),
            Err(AotError::SchemaDecodeError(_))
        ));

        let mut c_inf = JsonCursor::new("1e309");
        assert!(matches!(
            c_inf.parse_f64(),
            Err(AotError::SchemaDecodeError(_))
        ));

        let mut c_oid = JsonCursor::new(r#""00112233445566778899aabb""#);
        assert_eq!(c_oid.parse_object_id(), Ok(oid));
    }

    #[test]
    fn test_effects_tail_resumptive_abort_nested_dropguard_and_zst_provider() {
        #[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
        struct TestZstProvider;

        assert_eq!(core::mem::size_of::<TestZstProvider>(), 0);
        let p: TestZstProvider = default_zst_provider();
        assert_eq!(p, TestZstProvider::default());

        let mut ctx = AotCtx::new();
        assert_eq!(ctx.effects.active_frame_depth(), 0);

        // 1. Tail-resumptive multiple sequential resumptions + forwarding across nested handlers
        let nested_res = handle_tail_resumptive(
            &mut ctx,
            vec![EffectHandlerArm::from_fn("Outer", |c, args| {
                let y = args.first().and_then(AotValue::as_int).unwrap_or(0);
                resume_effect(c, AotValue::Int(y + 1))
            })],
            |c| {
                handle_tail_resumptive(
                    c,
                    vec![EffectHandlerArm::from_fn("Inner", |c2, args| {
                        let x = args.first().and_then(AotValue::as_int).unwrap_or(0);
                        resume_effect(c2, AotValue::Int(x * 2))
                    })],
                    |c2| {
                        assert_eq!(c2.effects.active_frame_depth(), 2);
                        let a = perform_effect(c2, "Inner", &[AotValue::Int(5)])?
                            .as_int()
                            .unwrap_or(0);
                        let b = perform_effect(c2, "Outer", &[AotValue::Int(100)])?
                            .as_int()
                            .unwrap_or(0);
                        Ok(AotValue::Int(a + b))
                    },
                )
            },
        );
        assert_eq!(nested_res, Ok(AotValue::Int(111)));
        assert_eq!(ctx.effects.active_frame_depth(), 0);

        // 2. Abortive handler unwinds inner defer DropGuard and pops inner EffectHandlerFrame
        let defer_ran = RefCell::new(false);
        let abort_res = handle_tail_resumptive_typed(
            &mut ctx,
            vec![EffectHandlerArm::from_fn("AbortOuter", |_c, args| {
                let code = args.first().and_then(AotValue::as_int).unwrap_or(0);
                Ok(AotValue::Int(-code))
            })],
            |c| {
                handle_tail_resumptive_typed(
                    c,
                    vec![EffectHandlerArm::from_fn("Other", |c2, _args| {
                        resume_effect(c2, AotValue::Int(0))
                    })],
                    |c2| {
                        let _d = defer(|| *defer_ran.borrow_mut() = true);
                        let _ = perform_effect(c2, "AbortOuter", &[AotValue::Int(404)])?;
                        Ok(9999_i64)
                    },
                    |v| Ok(v.as_int().unwrap_or(0)),
                )
            },
            |v| Ok(v.as_int().unwrap_or(0)),
        );
        assert_eq!(abort_res, Ok(-404));
        assert!(*defer_ran.borrow(), "defer DropGuard must run on abort unwind");
        assert_eq!(
            ctx.effects.active_frame_depth(),
            0,
            "inner and outer frames must both be popped by DropGuard on abort"
        );

        // 3. Unhandled effect, double resume, and resume outside handler arm errors
        assert_eq!(
            perform_effect(&mut ctx, "Missing", &[]),
            Err(AotError::UnhandledEffect(String::from("Missing")))
        );
        assert!(matches!(
            resume_effect(&mut ctx, AotValue::Int(1)),
            Err(AotError::UserError(ref m)) if m.contains("Cannot resume outside")
        ));
        let dbl_res = handle_tail_resumptive(
            &mut ctx,
            vec![EffectHandlerArm::from_fn("Ask", |c, _| {
                resume_effect(c, AotValue::Int(1))?;
                resume_effect(c, AotValue::Int(2))
            })],
            |c| perform_effect(c, "Ask", &[]),
        );
        assert!(matches!(
            dbl_res,
            Err(AotError::UserError(ref m)) if m.contains("Effect already resumed")
        ));
        assert_eq!(ctx.effects.active_frame_depth(), 0);
        assert!(!ctx.effects.is_in_handler_arm());

        // 4. Scoped service provider restores prior binding on normal & error exit
        assert_eq!(get_provided_service(&ctx, "AppDb"), None);
        let svc_res: Result<i64, AotError> =
            with_provided_service(&mut ctx, "AppDb", AotValue::Int(10), |c| {
                assert_eq!(get_provided_service(c, "AppDb"), Some(AotValue::Int(10)));
                let inner_err: Result<i64, AotError> =
                    with_provided_service(c, "AppDb", AotValue::Int(20), |c2| {
                        assert_eq!(get_provided_service(c2, "AppDb"), Some(AotValue::Int(20)));
                        Err(AotError::DivisionByZero)
                    });
                assert_eq!(inner_err, Err(AotError::DivisionByZero));
                assert_eq!(get_provided_service(c, "AppDb"), Some(AotValue::Int(10)));
                Ok(42)
            });
        assert_eq!(svc_res, Ok(42));
        assert_eq!(get_provided_service(&ctx, "AppDb"), None);
    }

    #[test]
    fn test_effect_yield_parameterized_all_types_and_oneshot_resume() {
        use crate::alloc_crate::rc::Rc;
        let mut ctx = AotCtx::new();

        // 1. EffectYield<i64, i64, i64> with pre + resume + post
        let trace = Rc::new(RefCell::new(Vec::<&'static str>::new()));
        let t1 = Rc::clone(&trace);
        let t2 = Rc::clone(&trace);
        let r_int = handle_non_tail_oneshot::<i64, i64, i64, _, _>(
            &mut ctx,
            |_c| {
                t1.borrow_mut().push("body_pre");
                Ok(EffectYield::yielded(10_i64, move |_c2, v| {
                    t1.borrow_mut().push("body_post");
                    Ok(v * 2)
                }))
            },
            |c, arg, resume_fn| {
                t2.borrow_mut().push("handler_pre");
                let r = resume_fn(c, arg + 5)?;
                t2.borrow_mut().push("handler_post");
                Ok(r + 100)
            },
        );
        assert_eq!(r_int, Ok(130));
        assert_eq!(
            &*trace.borrow(),
            &["body_pre", "handler_pre", "body_post", "handler_post"]
        );

        // 2. EffectYield<f64, bool, String> (three distinct non-i64 types)
        let r_hetero = handle_non_tail_oneshot::<f64, bool, String, _, _>(
            &mut ctx,
            |_c| {
                Ok(EffectYield::yielded(3.5_f64, |_c2, ok| {
                    Ok(if ok {
                        String::from("yes")
                    } else {
                        String::from("no")
                    })
                }))
            },
            |c, arg, resume_fn| {
                let body_s = resume_fn(c, arg > 2.0)?;
                Ok(crate::alloc_crate::format!("wrapped:{body_s}"))
            },
        );
        assert_eq!(r_hetero, Ok(String::from("wrapped:yes")));

        // 3. EffectYield<String, (), ()> (String + unit)
        let log = Rc::new(RefCell::new(String::new()));
        let l1 = Rc::clone(&log);
        let l2 = Rc::clone(&log);
        let r_unit = handle_non_tail_oneshot::<String, (), (), _, _>(
            &mut ctx,
            |_c| {
                Ok(EffectYield::yielded(String::from("ping"), move |_c2, ()| {
                    l1.borrow_mut().push_str(":resumed");
                    Ok(())
                }))
            },
            |c, arg, resume_fn| {
                l2.borrow_mut().push_str(&arg);
                resume_fn(c, ())?;
                l2.borrow_mut().push_str(":done");
                Ok(())
            },
        );
        assert_eq!(r_unit, Ok(()));
        assert_eq!(log.borrow().as_str(), "ping:resumed:done");

        // 4. EffectYield<AotValue, AotValue, AotValue> + Completed fast path
        let r_completed = handle_non_tail_oneshot::<AotValue, AotValue, AotValue, _, _>(
            &mut ctx,
            |_c| Ok(EffectYield::completed(AotValue::Int(77))),
            |_c, _arg, _resume_fn| Ok(AotValue::Int(-1)),
        );
        assert_eq!(r_completed, Ok(AotValue::Int(77)));

        // 5. OneShotResume rejects double resume cleanly without panicking
        let r_dbl = handle_non_tail_oneshot::<(), i64, i64, _, _>(
            &mut ctx,
            |_c| Ok(EffectYield::yielded((), |_c2, v| Ok(v + 1))),
            |c, (), resume_fn| {
                let mut k = OneShotResume::new(resume_fn);
                assert!(!k.is_consumed());
                let first = k.resume(c, 10)?;
                assert_eq!(first, 11);
                assert!(k.is_consumed());
                k.resume(c, 20)
            },
        );
        assert!(matches!(
            r_dbl,
            Err(AotError::UserError(ref m)) if m.contains("Effect already resumed")
        ));
    }

    #[test]
    fn test_aot_scope_three_phase_lifo_and_first_defect_preservation() {
        use crate::alloc_crate::rc::Rc;
        use core::cell::Cell;
        let mut ctx = AotCtx::new();
        let events = Rc::new(RefCell::new(Vec::<&'static str>::new()));

        let mut scope = AotScope::new(1);
        assert!(scope.is_open());
        assert_eq!(scope.phase(), AotScopePhase::Open);

        // Spawn two cooperative children that take 2 polls to finish after cancellation
        let e_c1 = Rc::clone(&events);
        let e_j1 = Rc::clone(&events);
        let mut ticks1 = 0u8;
        let h1 = scope
            .spawn_child_with_cancel(
                move |_c| e_c1.borrow_mut().push("cancel:child1"),
                move |_c, cancelled| {
                    assert!(cancelled);
                    ticks1 += 1;
                    if ticks1 >= 2 {
                        e_j1.borrow_mut().push("join:child1");
                        Ok(true)
                    } else {
                        Ok(false)
                    }
                },
            )
            .expect("spawn child1");

        let e_c2 = Rc::clone(&events);
        let e_j2 = Rc::clone(&events);
        let h2 = scope
            .spawn_child_with_cancel(
                move |_c| e_c2.borrow_mut().push("cancel:child2"),
                move |_c, cancelled| {
                    assert!(cancelled);
                    e_j2.borrow_mut().push("join:child2");
                    Ok(true)
                },
            )
            .expect("spawn child2");

        // Register 3 finalizers; middle & last return errors -> first defect must win, all 3 run LIFO
        let e_f1 = Rc::clone(&events);
        scope
            .register_finalizer(move |_c| {
                e_f1.borrow_mut().push("finalize:1");
                Err(AotError::UserError(String::from("second_defect")))
            })
            .expect("fin1");
        let e_f2 = Rc::clone(&events);
        scope
            .register_finalizer(move |_c| {
                e_f2.borrow_mut().push("finalize:2");
                Err(AotError::UserError(String::from("first_defect")))
            })
            .expect("fin2");
        let e_f3 = Rc::clone(&events);
        scope
            .register_finalizer(move |_c| {
                e_f3.borrow_mut().push("finalize:3");
                Ok(())
            })
            .expect("fin3");

        let close_res = scope.close(&mut ctx);
        assert_eq!(
            close_res,
            Err(AotError::UserError(String::from("first_defect")))
        );
        assert_eq!(
            scope.suppressed_errors(),
            &[AotError::UserError(String::from("second_defect"))]
        );
        assert_eq!(
            scope.phase_transitions(),
            &[
                AotScopePhase::Open,
                AotScopePhase::Cancelling,
                AotScopePhase::Joining,
                AotScopePhase::Finalizing,
                AotScopePhase::Done,
            ]
        );
        assert!(h1.is_cancel_requested());
        assert!(h2.is_cancel_requested());
        assert_eq!(h1.state(), AotFiberState::Cancelled);
        assert_eq!(h2.state(), AotFiberState::Cancelled);
        assert_eq!(
            &*events.borrow(),
            &[
                "cancel:child1",
                "cancel:child2",
                "join:child2",
                "join:child1",
                "finalize:3",
                "finalize:2",
                "finalize:1",
            ]
        );

        // Admission refused after close & close is idempotent
        assert!(scope.register_finalizer(|_| Ok(())).is_err());
        assert!(scope.spawn_child(|_, _| Ok(true)).is_err());
        assert_eq!(scope.close(&mut ctx), Ok(()));
        assert_eq!(events.borrow().len(), 7);

        // Gate-NEG-02: masked child prevents finalization during try_close until unmasked
        let mut gated = AotScope::new(2);
        let child = gated.register_external_child().expect("ext child");
        child.enter_mask();
        let fin_ran = Rc::new(Cell::new(false));
        let fin_ran_c = Rc::clone(&fin_ran);
        gated
            .register_finalizer(move |_| {
                fin_ran_c.set(true);
                Ok(())
            })
            .expect("reg fin");

        assert_eq!(gated.try_close(&mut ctx), Ok(false));
        assert_eq!(gated.phase(), AotScopePhase::Joining);
        assert!(child.is_cancel_requested());
        assert!(!fin_ran.get(), "Gate-NEG-02: finalizer must NOT run while child is masked");

        assert!(child.exit_mask(), "deferred cancel must surface on unmask");
        child.mark_cancelled();
        assert_eq!(gated.try_close(&mut ctx), Ok(true));
        assert_eq!(gated.phase(), AotScopePhase::Done);
        assert!(fin_ran.get());

        // with_aot_scope + spawn_future: child future RAII DropGuard runs BEFORE parent finalizers
        let async_events = Rc::new(RefCell::new(Vec::<&'static str>::new()));
        let ae_body = Rc::clone(&async_events);
        let scope_res = with_aot_scope(&mut ctx, |c| {
            assert_eq!(active_scope_depth(c), 1);
            let ae_fin = Rc::clone(&ae_body);
            scope_register_finalizer(c, move |_| {
                ae_fin.borrow_mut().push("parent_fin");
                Ok(())
            })?;
            let ae_fut = Rc::clone(&ae_body);
            let _fh = scope_spawn_future(c, async move {
                let _g = defer(move || ae_fut.borrow_mut().push("child_future_drop"));
                core::future::pending::<Result<AotValue, AotError>>().await
            })?;
            // Poll once implicitly via scope teardown: future is pending (Wait: first poll starts future!)
            Ok(99_i64)
        });
        assert_eq!(scope_res, Ok(99));
        assert_eq!(active_scope_depth(&ctx), 0);
        // Child future RAII DropGuard ran during Joining, strictly BEFORE parent_fin in Finalizing!
        assert_eq!(
            &*async_events.borrow(),
            &["child_future_drop", "parent_fin"]
        );
    }
}

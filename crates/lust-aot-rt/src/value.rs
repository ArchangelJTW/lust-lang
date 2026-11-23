//! Dynamic fallback value representation (`AotValue`) for Lust Rust AOT.
//!
//! Used only when static MIR types remain `Unknown` / union / heterogeneous containers.
//! Fully `#![no_std] + alloc`, zero external dependencies, and panic-free (Hazard 9).

use crate::alloc::LustBudgetAlloc;
use crate::alloc_crate::collections::BTreeMap;
use crate::alloc_crate::rc::Rc;
use crate::alloc_crate::string::String;
use crate::alloc_crate::vec::Vec;
use crate::{AotCtx, AotError};
use core::cell::RefCell;
use core::fmt::{self, Write as _};

/// Maximum recursive traversal depth for `Debug`, `Display`, and `PartialEq` to prevent stack overflow on cycles.
const MAX_VALUE_DEPTH: usize = 64;

/// Deterministic ordered key type for dynamic [`AotValue::Map`] containers.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AotKey {
    Nil,
    Bool(bool),
    Int(i64),
    Str(String),
}

impl fmt::Display for AotKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AotKey::Nil => write!(f, "nil"),
            AotKey::Bool(b) => write!(f, "{b}"),
            AotKey::Int(i) => write!(f, "{i}"),
            AotKey::Str(s) => write!(f, "{s}"),
        }
    }
}

impl From<bool> for AotKey {
    #[inline]
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

impl From<i64> for AotKey {
    #[inline]
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<String> for AotKey {
    #[inline]
    fn from(v: String) -> Self {
        Self::Str(v)
    }
}

impl From<&str> for AotKey {
    #[inline]
    fn from(v: &str) -> Self {
        Self::Str(String::from(v))
    }
}

/// Dynamic closure type stored in [`AotValue::Closure`].
pub type AotClosure = Rc<dyn Fn(&mut AotCtx, &[AotValue]) -> Result<AotValue, AotError>>;

/// Dynamic struct payload stored in [`AotValue::Struct`].
#[derive(Debug, Clone, PartialEq)]
pub struct AotStruct {
    pub type_id: u32,
    pub fields: Vec<AotValue>,
}

/// Dynamic enum payload stored in [`AotValue::Enum`].
#[derive(Debug, Clone, PartialEq)]
pub struct AotEnum {
    pub enum_id: u32,
    pub variant: u32,
    pub payload: Vec<AotValue>,
}

/// Compact dynamic fallback value for untyped / heterogeneous AOT paths.
#[derive(Clone)]
pub enum AotValue {
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Rc<String>),
    Array(Rc<RefCell<Vec<AotValue>>>),
    Map(Rc<RefCell<BTreeMap<AotKey, AotValue>>>),
    Struct(Rc<RefCell<AotStruct>>),
    Enum(Rc<AotEnum>),
    Closure(AotClosure),
}

impl Default for AotValue {
    #[inline]
    fn default() -> Self {
        Self::Nil
    }
}

impl AotValue {
    #[inline]
    pub fn str(s: impl Into<String>) -> Self {
        Self::Str(Rc::new(s.into()))
    }

    #[inline]
    pub fn array(items: Vec<AotValue>) -> Self {
        Self::Array(Rc::new(RefCell::new(items)))
    }

    #[inline]
    pub fn empty_array() -> Self {
        Self::Array(Rc::new(RefCell::new(Vec::new())))
    }

    #[inline]
    pub fn map() -> Self {
        Self::Map(Rc::new(RefCell::new(BTreeMap::new())))
    }

    #[inline]
    pub fn from_struct(type_id: u32, fields: Vec<AotValue>) -> Self {
        Self::Struct(Rc::new(RefCell::new(AotStruct { type_id, fields })))
    }

    #[inline]
    pub fn from_enum(enum_id: u32, variant: u32, payload: Vec<AotValue>) -> Self {
        Self::Enum(Rc::new(AotEnum {
            enum_id,
            variant,
            payload,
        }))
    }

    #[inline]
    pub fn closure<F>(f: F) -> Self
    where
        F: Fn(&mut AotCtx, &[AotValue]) -> Result<AotValue, AotError> + 'static,
    {
        Self::Closure(Rc::new(f))
    }

    #[inline]
    pub fn is_nil(&self) -> bool {
        matches!(self, Self::Nil)
    }

    /// Lust truthiness: `nil` and `false` are falsy; all other values are truthy.
    #[inline]
    pub fn is_truthy(&self) -> bool {
        !matches!(self, Self::Nil | Self::Bool(false))
    }

    #[inline]
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Nil => "nil",
            Self::Bool(_) => "bool",
            Self::Int(_) => "int",
            Self::Float(_) => "float",
            Self::Str(_) => "string",
            Self::Array(_) => "array",
            Self::Map(_) => "map",
            Self::Struct(_) => "struct",
            Self::Enum(_) => "enum",
            Self::Closure(_) => "function",
        }
    }

    #[inline]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    #[inline]
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            _ => None,
        }
    }

    #[inline]
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(fl) => Some(*fl),
            _ => None,
        }
    }

    #[inline]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Converts scalar values to a map key without panicking.
    #[inline]
    pub fn to_key(&self) -> Result<AotKey, AotError> {
        match self {
            Self::Nil => Ok(AotKey::Nil),
            Self::Bool(b) => Ok(AotKey::Bool(*b)),
            Self::Int(i) => Ok(AotKey::Int(*i)),
            Self::Str(s) => Ok(AotKey::Str(s.as_ref().clone())),
            _ => Err(AotError::SchemaDecodeError(
                "unsupported key type for AotValue::Map",
            )),
        }
    }

    /// Reads an element from an [`AotValue::Array`] with bounds checking and zero panics.
    pub fn array_get(&self, index: i64) -> Result<AotValue, AotError> {
        let Self::Array(arr) = self else {
            return Err(AotError::UserError(String::from("expected array")));
        };
        let borrowed = arr
            .try_borrow()
            .map_err(|_| AotError::UserError(String::from("re-entrant array borrow")))?;
        let idx = crate::array_index_check(index, borrowed.len())?;
        borrowed
            .get(idx)
            .cloned()
            .ok_or(AotError::IndexOutOfBounds {
                index,
                len: borrowed.len(),
            })
    }

    /// Writes an element in an [`AotValue::Array`] with bounds checking and zero panics.
    pub fn array_set(&self, index: i64, value: AotValue) -> Result<(), AotError> {
        let Self::Array(arr) = self else {
            return Err(AotError::UserError(String::from("expected array")));
        };
        let mut borrowed = arr
            .try_borrow_mut()
            .map_err(|_| AotError::UserError(String::from("re-entrant mutable array borrow")))?;
        let idx = crate::array_index_check(index, borrowed.len())?;
        let len = borrowed.len();
        let Some(slot) = borrowed.get_mut(idx) else {
            return Err(AotError::IndexOutOfBounds { index, len });
        };
        *slot = value;
        Ok(())
    }

    /// Pushes an element to an [`AotValue::Array`], charging `ctx.budget_alloc` first (Hazard 8/9).
    pub fn array_push(&self, value: AotValue, ctx: &AotCtx) -> Result<(), AotError> {
        let Self::Array(arr) = self else {
            return Err(AotError::UserError(String::from("expected array")));
        };
        let mut borrowed = arr
            .try_borrow_mut()
            .map_err(|_| AotError::UserError(String::from("re-entrant mutable array borrow")))?;
        let charge = core::mem::size_of::<AotValue>();
        ctx.budget_alloc.try_charge(charge)?;
        if borrowed.try_reserve(1).is_err() {
            ctx.budget_alloc.refund(charge);
            return Err(AotError::OutOfMemory {
                limit_bytes: ctx.budget_alloc.limit_bytes(),
                requested_bytes: ctx.budget_alloc.used_bytes().saturating_add(charge),
            });
        }
        borrowed.push(value);
        Ok(())
    }

    /// Returns the length of an array or map (or string byte length).
    pub fn len(&self) -> Result<usize, AotError> {
        match self {
            Self::Str(s) => Ok(s.len()),
            Self::Array(arr) => arr
                .try_borrow()
                .map(|b| b.len())
                .map_err(|_| AotError::UserError(String::from("re-entrant array borrow"))),
            Self::Map(map) => map
                .try_borrow()
                .map(|b| b.len())
                .map_err(|_| AotError::UserError(String::from("re-entrant map borrow"))),
            _ => Err(AotError::UserError(String::from(
                "len() requires string, array, or map",
            ))),
        }
    }

    /// Looks up a key in an [`AotValue::Map`], returning [`AotError::MissingMapKey`] if absent.
    pub fn map_get(&self, key: &AotKey) -> Result<AotValue, AotError> {
        let Self::Map(map) = self else {
            return Err(AotError::UserError(String::from("expected map")));
        };
        let borrowed = map
            .try_borrow()
            .map_err(|_| AotError::UserError(String::from("re-entrant map borrow")))?;
        borrowed.get(key).cloned().ok_or(AotError::MissingMapKey)
    }

    /// Inserts or updates a key in an [`AotValue::Map`], charging `ctx.budget_alloc` on new entries (Hazard 8).
    pub fn map_set(&self, key: AotKey, value: AotValue, ctx: &AotCtx) -> Result<(), AotError> {
        let Self::Map(map) = self else {
            return Err(AotError::UserError(String::from("expected map")));
        };
        let mut borrowed = map
            .try_borrow_mut()
            .map_err(|_| AotError::UserError(String::from("re-entrant mutable map borrow")))?;
        if !borrowed.contains_key(&key) {
            let extra_str = match &key {
                AotKey::Str(s) => s.len(),
                _ => 0,
            };
            let entry_bytes = core::mem::size_of::<AotKey>()
                .saturating_add(core::mem::size_of::<AotValue>())
                .saturating_add(extra_str);
            ctx.budget_alloc.try_charge(entry_bytes)?;
        }
        borrowed.insert(key, value);
        Ok(())
    }

    /// Removes a key from an [`AotValue::Map`], refunding entry bytes when present.
    pub fn map_remove(&self, key: &AotKey, ctx: &AotCtx) -> Result<Option<AotValue>, AotError> {
        let Self::Map(map) = self else {
            return Err(AotError::UserError(String::from("expected map")));
        };
        let mut borrowed = map
            .try_borrow_mut()
            .map_err(|_| AotError::UserError(String::from("re-entrant mutable map borrow")))?;
        let removed = borrowed.remove(key);
        if removed.is_some() {
            let extra_str = match key {
                AotKey::Str(s) => s.len(),
                _ => 0,
            };
            let entry_bytes = core::mem::size_of::<AotKey>()
                .saturating_add(core::mem::size_of::<AotValue>())
                .saturating_add(extra_str);
            ctx.budget_alloc.refund(entry_bytes);
        }
        Ok(removed)
    }

    /// Invokes a dynamic [`AotValue::Closure`].
    pub fn call(&self, ctx: &mut AotCtx, args: &[AotValue]) -> Result<AotValue, AotError> {
        let Self::Closure(func) = self else {
            return Err(AotError::UserError(String::from(
                "attempted to call non-function value",
            )));
        };
        func(ctx, args)
    }

    /// Converts an `AotValue` to `f64`, accepting both `Float` and `Int`.
    #[inline]
    pub fn as_float_coerced(&self) -> Result<f64, AotError> {
        match self {
            Self::Float(fl) => Ok(*fl),
            Self::Int(i) => Ok(*i as f64),
            _ => Err(AotError::UserError(String::from("expected number"))),
        }
    }

    /// Formats this value using its canonical `Display` representation.
    pub fn to_display_string(&self) -> String {
        let mut out = String::new();
        let _ = write!(&mut out, "{self}");
        out
    }

    /// Looks up a key in an [`AotValue::Map`], returning `Ok(None)` when absent.
    pub fn map_get_opt(&self, key: &AotKey) -> Result<Option<AotValue>, AotError> {
        let Self::Map(map) = self else {
            return Err(AotError::UserError(String::from("expected map")));
        };
        let borrowed = map
            .try_borrow()
            .map_err(|_| AotError::UserError(String::from("re-entrant map borrow")))?;
        Ok(borrowed.get(key).cloned())
    }

    /// Returns whether `key` exists in an [`AotValue::Map`].
    pub fn map_has(&self, key: &AotKey) -> Result<bool, AotError> {
        let Self::Map(map) = self else {
            return Err(AotError::UserError(String::from("expected map")));
        };
        let borrowed = map
            .try_borrow()
            .map_err(|_| AotError::UserError(String::from("re-entrant map borrow")))?;
        Ok(borrowed.contains_key(key))
    }

    /// Returns whether an [`AotValue::Map`] is empty.
    pub fn map_is_empty(&self) -> Result<bool, AotError> {
        let Self::Map(map) = self else {
            return Err(AotError::UserError(String::from("expected map")));
        };
        let borrowed = map
            .try_borrow()
            .map_err(|_| AotError::UserError(String::from("re-entrant map borrow")))?;
        Ok(borrowed.is_empty())
    }

    /// Clears all entries from an [`AotValue::Map`].
    pub fn map_clear(&self) -> Result<(), AotError> {
        let Self::Map(map) = self else {
            return Err(AotError::UserError(String::from("expected map")));
        };
        let mut borrowed = map
            .try_borrow_mut()
            .map_err(|_| AotError::UserError(String::from("re-entrant mutable map borrow")))?;
        borrowed.clear();
        Ok(())
    }

    /// Returns an [`AotValue::Array`] of all keys in an [`AotValue::Map`].
    pub fn map_keys(&self, alloc: &LustBudgetAlloc) -> Result<AotValue, AotError> {
        let Self::Map(map) = self else {
            return Err(AotError::UserError(String::from("expected map")));
        };
        let borrowed = map
            .try_borrow()
            .map_err(|_| AotError::UserError(String::from("re-entrant map borrow")))?;
        let charge = borrowed
            .len()
            .saturating_mul(core::mem::size_of::<AotValue>());
        if charge > 0 {
            alloc.try_charge(charge)?;
        }
        let mut keys = Vec::new();
        if keys.try_reserve_exact(borrowed.len()).is_err() {
            if charge > 0 {
                alloc.refund(charge);
            }
            return Err(AotError::from_alloc_failure(alloc, charge));
        }
        for k in borrowed.keys() {
            let v = match k {
                AotKey::Nil => AotValue::Nil,
                AotKey::Bool(b) => AotValue::Bool(*b),
                AotKey::Int(i) => AotValue::Int(*i),
                AotKey::Str(s) => AotValue::str(s.clone()),
            };
            keys.push(v);
        }
        Ok(AotValue::array(keys))
    }

    /// Returns an [`AotValue::Array`] of all values in an [`AotValue::Map`].
    pub fn map_values(&self, alloc: &LustBudgetAlloc) -> Result<AotValue, AotError> {
        let Self::Map(map) = self else {
            return Err(AotError::UserError(String::from("expected map")));
        };
        let borrowed = map
            .try_borrow()
            .map_err(|_| AotError::UserError(String::from("re-entrant map borrow")))?;
        let charge = borrowed
            .len()
            .saturating_mul(core::mem::size_of::<AotValue>());
        if charge > 0 {
            alloc.try_charge(charge)?;
        }
        let mut vals = Vec::new();
        if vals.try_reserve_exact(borrowed.len()).is_err() {
            if charge > 0 {
                alloc.refund(charge);
            }
            return Err(AotError::from_alloc_failure(alloc, charge));
        }
        for v in borrowed.values() {
            vals.push(v.clone());
        }
        Ok(AotValue::array(vals))
    }

    /// Extracts the variant tag of an [`AotValue::Enum`].
    pub fn enum_tag(&self) -> Result<i64, AotError> {
        let Self::Enum(en) = self else {
            return Err(AotError::UserError(String::from("expected enum")));
        };
        Ok(i64::from(en.variant))
    }

    /// Extracts payload element `idx` from an [`AotValue::Enum`].
    pub fn enum_payload(&self, idx: usize) -> Result<AotValue, AotError> {
        let Self::Enum(en) = self else {
            return Err(AotError::UserError(String::from("expected enum")));
        };
        en.payload
            .get(idx)
            .cloned()
            .ok_or(AotError::IndexOutOfBounds {
                index: idx as i64,
                len: en.payload.len(),
            })
    }

    /// Dynamic addition matching Lust VM semantics (wrapping `i64`, IEEE 754 `f64`).
    pub fn add(&self, rhs: &Self) -> Result<Self, AotError> {
        match (self, rhs) {
            (Self::Int(a), Self::Int(b)) => Ok(Self::Int(a.wrapping_add(*b))),
            (Self::Float(a), Self::Float(b)) => Ok(Self::Float(*a + *b)),
            (Self::Int(a), Self::Float(b)) => Ok(Self::Float((*a as f64) + *b)),
            (Self::Float(a), Self::Int(b)) => Ok(Self::Float(*a + (*b as f64))),
            _ => Err(AotError::UserError(String::from(
                "invalid operands for '+'",
            ))),
        }
    }

    /// Dynamic subtraction matching Lust VM semantics.
    pub fn sub(&self, rhs: &Self) -> Result<Self, AotError> {
        match (self, rhs) {
            (Self::Int(a), Self::Int(b)) => Ok(Self::Int(a.wrapping_sub(*b))),
            (Self::Float(a), Self::Float(b)) => Ok(Self::Float(*a - *b)),
            (Self::Int(a), Self::Float(b)) => Ok(Self::Float((*a as f64) - *b)),
            (Self::Float(a), Self::Int(b)) => Ok(Self::Float(*a - (*b as f64))),
            _ => Err(AotError::UserError(String::from(
                "invalid operands for '-'",
            ))),
        }
    }

    /// Dynamic multiplication matching Lust VM semantics.
    pub fn mul(&self, rhs: &Self) -> Result<Self, AotError> {
        match (self, rhs) {
            (Self::Int(a), Self::Int(b)) => Ok(Self::Int(a.wrapping_mul(*b))),
            (Self::Float(a), Self::Float(b)) => Ok(Self::Float(*a * *b)),
            (Self::Int(a), Self::Float(b)) => Ok(Self::Float((*a as f64) * *b)),
            (Self::Float(a), Self::Int(b)) => Ok(Self::Float(*a * (*b as f64))),
            _ => Err(AotError::UserError(String::from(
                "invalid operands for '*'",
            ))),
        }
    }

    /// Dynamic division matching Lust VM semantics (panic-free).
    pub fn div(&self, rhs: &Self) -> Result<Self, AotError> {
        match (self, rhs) {
            (Self::Int(a), Self::Int(b)) => Ok(Self::Int(crate::div_int(*a, *b)?)),
            (Self::Float(a), Self::Float(b)) => Ok(Self::Float(*a / *b)),
            (Self::Int(a), Self::Float(b)) => Ok(Self::Float((*a as f64) / *b)),
            (Self::Float(a), Self::Int(b)) => Ok(Self::Float(*a / (*b as f64))),
            _ => Err(AotError::UserError(String::from(
                "invalid operands for '/'",
            ))),
        }
    }

    /// Dynamic remainder matching Lust VM semantics (panic-free).
    pub fn rem(&self, rhs: &Self) -> Result<Self, AotError> {
        match (self, rhs) {
            (Self::Int(a), Self::Int(b)) => Ok(Self::Int(crate::rem_int(*a, *b)?)),
            (Self::Float(a), Self::Float(b)) => Ok(Self::Float(*a % *b)),
            (Self::Int(a), Self::Float(b)) => Ok(Self::Float((*a as f64) % *b)),
            (Self::Float(a), Self::Int(b)) => Ok(Self::Float(*a % (*b as f64))),
            _ => Err(AotError::UserError(String::from(
                "invalid operands for '%'",
            ))),
        }
    }

    /// Dynamic floor division matching Lust VM semantics (panic-free).
    pub fn floordiv(&self, rhs: &Self) -> Result<Self, AotError> {
        match (self, rhs) {
            (Self::Int(a), Self::Int(b)) => Ok(Self::Int(crate::floordiv_int(*a, *b)?)),
            (Self::Float(a), Self::Float(b)) => Ok(Self::Float(floor_f64(*a / *b))),
            (Self::Int(a), Self::Float(b)) => Ok(Self::Float(floor_f64((*a as f64) / *b))),
            (Self::Float(a), Self::Int(b)) => Ok(Self::Float(floor_f64(*a / (*b as f64)))),
            _ => Err(AotError::UserError(String::from(
                "invalid operands for '//'",
            ))),
        }
    }

    /// Dynamic exponentiation matching Lust VM semantics (panic-free).
    pub fn pow(&self, rhs: &Self) -> Result<Self, AotError> {
        match (self, rhs) {
            (Self::Int(a), Self::Int(b)) => Ok(Self::Int(crate::pow_int(*a, *b)?)),
            (Self::Float(a), Self::Float(b)) => Ok(Self::Float(powf_f64(*a, *b))),
            (Self::Int(a), Self::Float(b)) => Ok(Self::Float(powf_f64(*a as f64, *b))),
            (Self::Float(a), Self::Int(b)) => Ok(Self::Float(powf_f64(*a, *b as f64))),
            _ => Err(AotError::UserError(String::from(
                "invalid operands for '^'",
            ))),
        }
    }

    /// Dynamic negation matching Lust VM semantics.
    pub fn neg(&self) -> Result<Self, AotError> {
        match self {
            Self::Int(a) => Ok(Self::Int(a.wrapping_neg())),
            Self::Float(a) => Ok(Self::Float(-*a)),
            _ => Err(AotError::UserError(String::from(
                "invalid operand for unary '-'",
            ))),
        }
    }

    /// Dynamic string concatenation (`..`).
    pub fn concat(&self, rhs: &Self, alloc: &LustBudgetAlloc) -> Result<Self, AotError> {
        let mut out = self.to_display_string();
        let rhs_s = rhs.to_display_string();
        let add_len = rhs_s.len();
        if add_len > 0 {
            alloc.try_charge(add_len)?;
        }
        out.push_str(&rhs_s);
        Ok(Self::Str(Rc::new(out)))
    }

    /// Dynamic `<` comparison.
    pub fn cmp_lt(&self, rhs: &Self) -> Result<bool, AotError> {
        match (self, rhs) {
            (Self::Int(a), Self::Int(b)) => Ok(*a < *b),
            (Self::Float(a), Self::Float(b)) => Ok(*a < *b),
            (Self::Int(a), Self::Float(b)) => Ok((*a as f64) < *b),
            (Self::Float(a), Self::Int(b)) => Ok(*a < (*b as f64)),
            (Self::Str(a), Self::Str(b)) => Ok(a < b),
            _ => Err(AotError::UserError(String::from(
                "invalid operands for '<'",
            ))),
        }
    }

    /// Dynamic `<=` comparison.
    pub fn cmp_le(&self, rhs: &Self) -> Result<bool, AotError> {
        match (self, rhs) {
            (Self::Int(a), Self::Int(b)) => Ok(*a <= *b),
            (Self::Float(a), Self::Float(b)) => Ok(*a <= *b),
            (Self::Int(a), Self::Float(b)) => Ok((*a as f64) <= *b),
            (Self::Float(a), Self::Int(b)) => Ok(*a <= (*b as f64)),
            (Self::Str(a), Self::Str(b)) => Ok(a <= b),
            _ => Err(AotError::UserError(String::from(
                "invalid operands for '<='",
            ))),
        }
    }

    /// Dynamic `>` comparison.
    pub fn cmp_gt(&self, rhs: &Self) -> Result<bool, AotError> {
        rhs.cmp_lt(self)
    }

    /// Dynamic `>=` comparison.
    pub fn cmp_ge(&self, rhs: &Self) -> Result<bool, AotError> {
        rhs.cmp_le(self)
    }

    fn eq_depth(&self, other: &Self, depth: usize) -> bool {
        if depth >= MAX_VALUE_DEPTH {
            return false;
        }
        match (self, other) {
            (Self::Nil, Self::Nil) => true,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Int(a), Self::Int(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => a == b,
            (Self::Str(a), Self::Str(b)) => a == b,
            (Self::Array(a), Self::Array(b)) => {
                if Rc::ptr_eq(a, b) {
                    return true;
                }
                let (Ok(va), Ok(vb)) = (a.try_borrow(), b.try_borrow()) else {
                    return false;
                };
                if va.len() != vb.len() {
                    return false;
                }
                va.iter()
                    .zip(vb.iter())
                    .all(|(x, y)| x.eq_depth(y, depth + 1))
            }
            (Self::Map(a), Self::Map(b)) => {
                if Rc::ptr_eq(a, b) {
                    return true;
                }
                let (Ok(ma), Ok(mb)) = (a.try_borrow(), b.try_borrow()) else {
                    return false;
                };
                if ma.len() != mb.len() {
                    return false;
                }
                ma.iter()
                    .zip(mb.iter())
                    .all(|((ka, va), (kb, vb))| ka == kb && va.eq_depth(vb, depth + 1))
            }
            (Self::Struct(a), Self::Struct(b)) => {
                if Rc::ptr_eq(a, b) {
                    return true;
                }
                let (Ok(sa), Ok(sb)) = (a.try_borrow(), b.try_borrow()) else {
                    return false;
                };
                sa.type_id == sb.type_id
                    && sa.fields.len() == sb.fields.len()
                    && sa
                        .fields
                        .iter()
                        .zip(sb.fields.iter())
                        .all(|(x, y)| x.eq_depth(y, depth + 1))
            }
            (Self::Enum(a), Self::Enum(b)) => {
                if Rc::ptr_eq(a, b) {
                    return true;
                }
                a.enum_id == b.enum_id
                    && a.variant == b.variant
                    && a.payload.len() == b.payload.len()
                    && a.payload
                        .iter()
                        .zip(b.payload.iter())
                        .all(|(x, y)| x.eq_depth(y, depth + 1))
            }
            (Self::Closure(a), Self::Closure(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }

    fn fmt_display_depth(&self, f: &mut fmt::Formatter<'_>, depth: usize) -> fmt::Result {
        match self {
            Self::Nil => write!(f, "nil"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::Int(i) => write!(f, "{i}"),
            Self::Float(fl) => write!(f, "{fl}"),
            Self::Str(s) => write!(f, "{s}"),
            Self::Array(arr) => {
                if depth >= MAX_VALUE_DEPTH {
                    return write!(f, "[...]");
                }
                let Ok(borrowed) = arr.try_borrow() else {
                    return write!(f, "[...]");
                };
                write!(f, "[")?;
                for (i, item) in borrowed.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    item.fmt_display_depth(f, depth + 1)?;
                }
                write!(f, "]")
            }
            Self::Map(map) => {
                if depth >= MAX_VALUE_DEPTH {
                    return write!(f, "{{...}}");
                }
                let Ok(borrowed) = map.try_borrow() else {
                    return write!(f, "{{...}}");
                };
                write!(f, "{{")?;
                for (i, (k, v)) in borrowed.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k}: ")?;
                    v.fmt_display_depth(f, depth + 1)?;
                }
                write!(f, "}}")
            }
            Self::Struct(st) => {
                if depth >= MAX_VALUE_DEPTH {
                    return write!(f, "struct(...)");
                }
                let Ok(borrowed) = st.try_borrow() else {
                    return write!(f, "struct(...)");
                };
                write!(f, "struct#{}(", borrowed.type_id)?;
                for (i, field) in borrowed.fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    field.fmt_display_depth(f, depth + 1)?;
                }
                write!(f, ")")
            }
            Self::Enum(en) => {
                if depth >= MAX_VALUE_DEPTH {
                    return write!(f, "enum(...)");
                }
                write!(f, "enum#{}::{}", en.enum_id, en.variant)?;
                if !en.payload.is_empty() {
                    write!(f, "(")?;
                    for (i, item) in en.payload.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        item.fmt_display_depth(f, depth + 1)?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }
            Self::Closure(_) => write!(f, "<function>"),
        }
    }

    fn fmt_debug_depth(&self, f: &mut fmt::Formatter<'_>, depth: usize) -> fmt::Result {
        match self {
            Self::Nil => write!(f, "Nil"),
            Self::Bool(b) => write!(f, "Bool({b})"),
            Self::Int(i) => write!(f, "Int({i})"),
            Self::Float(fl) => write!(f, "Float({fl})"),
            Self::Str(s) => write!(f, "Str({s:?})"),
            Self::Array(arr) => {
                if depth >= MAX_VALUE_DEPTH {
                    return write!(f, "Array([...])");
                }
                let Ok(borrowed) = arr.try_borrow() else {
                    return write!(f, "Array([...])");
                };
                write!(f, "Array([")?;
                for (i, item) in borrowed.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    item.fmt_debug_depth(f, depth + 1)?;
                }
                write!(f, "])")
            }
            Self::Map(map) => {
                if depth >= MAX_VALUE_DEPTH {
                    return write!(f, "Map({{...}})");
                }
                let Ok(borrowed) = map.try_borrow() else {
                    return write!(f, "Map({{...}})");
                };
                write!(f, "Map({{")?;
                for (i, (k, v)) in borrowed.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k:?}: ")?;
                    v.fmt_debug_depth(f, depth + 1)?;
                }
                write!(f, "}})")
            }
            Self::Struct(st) => {
                if depth >= MAX_VALUE_DEPTH {
                    return write!(f, "Struct(...)");
                }
                let Ok(borrowed) = st.try_borrow() else {
                    return write!(f, "Struct(...)");
                };
                write!(f, "Struct {{ type_id: {}, fields: [", borrowed.type_id)?;
                for (i, item) in borrowed.fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    item.fmt_debug_depth(f, depth + 1)?;
                }
                write!(f, "] }}")
            }
            Self::Enum(en) => {
                if depth >= MAX_VALUE_DEPTH {
                    return write!(f, "Enum(...)");
                }
                write!(
                    f,
                    "Enum {{ enum_id: {}, variant: {}, payload: [",
                    en.enum_id, en.variant
                )?;
                for (i, item) in en.payload.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    item.fmt_debug_depth(f, depth + 1)?;
                }
                write!(f, "] }}")
            }
            Self::Closure(_) => write!(f, "Closure(<fn>)"),
        }
    }
}

impl PartialEq for AotValue {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.eq_depth(other, 0)
    }
}

impl fmt::Display for AotValue {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_display_depth(f, 0)
    }
}

impl fmt::Debug for AotValue {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_debug_depth(f, 0)
    }
}

impl From<()> for AotValue {
    #[inline]
    fn from(_: ()) -> Self {
        Self::Nil
    }
}

impl From<bool> for AotValue {
    #[inline]
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

impl From<i64> for AotValue {
    #[inline]
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<f64> for AotValue {
    #[inline]
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}

impl From<String> for AotValue {
    #[inline]
    fn from(v: String) -> Self {
        Self::Str(Rc::new(v))
    }
}

impl From<&str> for AotValue {
    #[inline]
    fn from(v: &str) -> Self {
        Self::Str(Rc::new(String::from(v)))
    }
}

#[inline]
fn floor_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.floor()
    }
    #[cfg(not(feature = "std"))]
    {
        if x.is_nan() || x.is_infinite() {
            return x;
        }
        let trunc = (x as i64) as f64;
        if x < trunc { trunc - 1.0 } else { trunc }
    }
}

#[inline]
fn powf_f64(base: f64, exp: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        base.powf(exp)
    }
    #[cfg(not(feature = "std"))]
    {
        if exp == 0.0 {
            return 1.0;
        }
        if base.is_nan() || exp.is_nan() {
            return f64::NAN;
        }
        let int_exp = exp as i64;
        if (int_exp as f64) == exp {
            let mut acc = 1.0_f64;
            let mut b = if int_exp < 0 { 1.0 / base } else { base };
            let mut e = int_exp.unsigned_abs();
            while e > 0 {
                if (e & 1) == 1 {
                    acc *= b;
                }
                e >>= 1;
                if e > 0 {
                    b *= b;
                }
            }
            return acc;
        }
        f64::NAN
    }
}

/// Typed, reference-counted, panic-free ordered map container for Tier 4 Rust AOT (`Map<K, V>`).
///
/// Wraps `Rc<RefCell<BTreeMap<K, V>>>` so `Clone` preserves Lust reference aliasing semantics
/// across local aliases, closures, and function calls while enforcing `LustBudgetAlloc` limits
/// with zero `unsafe` and zero panics (Hazard 9 / Hazard 12).
#[derive(Debug)]
pub struct AotMap<K, V>(pub Rc<RefCell<BTreeMap<K, V>>>);

impl<K, V> Clone for AotMap<K, V> {
    #[inline]
    fn clone(&self) -> Self {
        Self(Rc::clone(&self.0))
    }
}

impl<K: Ord, V> Default for AotMap<K, V> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord, V: PartialEq> PartialEq for AotMap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        if Rc::ptr_eq(&self.0, &other.0) {
            return true;
        }
        let (Ok(a), Ok(b)) = (self.0.try_borrow(), other.0.try_borrow()) else {
            return false;
        };
        *a == *b
    }
}

impl<K: Ord, V> AotMap<K, V> {
    #[inline]
    pub fn new() -> Self {
        Self(Rc::new(RefCell::new(BTreeMap::new())))
    }

    #[inline]
    pub fn len(&self) -> Result<i64, AotError> {
        let borrowed = self
            .0
            .try_borrow()
            .map_err(|_| AotError::UserError(String::from("re-entrant map borrow")))?;
        Ok(borrowed.len() as i64)
    }

    #[inline]
    pub fn is_empty(&self) -> Result<bool, AotError> {
        let borrowed = self
            .0
            .try_borrow()
            .map_err(|_| AotError::UserError(String::from("re-entrant map borrow")))?;
        Ok(borrowed.is_empty())
    }

    #[inline]
    pub fn has(&self, key: &K) -> Result<bool, AotError> {
        let borrowed = self
            .0
            .try_borrow()
            .map_err(|_| AotError::UserError(String::from("re-entrant map borrow")))?;
        Ok(borrowed.contains_key(key))
    }

    #[inline]
    pub fn get(&self, key: &K) -> Result<V, AotError>
    where
        V: Clone,
    {
        let borrowed = self
            .0
            .try_borrow()
            .map_err(|_| AotError::UserError(String::from("re-entrant map borrow")))?;
        borrowed.get(key).cloned().ok_or(AotError::MissingMapKey)
    }

    #[inline]
    pub fn get_opt(&self, key: &K) -> Result<Option<V>, AotError>
    where
        V: Clone,
    {
        let borrowed = self
            .0
            .try_borrow()
            .map_err(|_| AotError::UserError(String::from("re-entrant map borrow")))?;
        Ok(borrowed.get(key).cloned())
    }

    #[inline]
    pub fn set(&self, key: K, value: V, alloc: &LustBudgetAlloc) -> Result<(), AotError> {
        let mut borrowed = self
            .0
            .try_borrow_mut()
            .map_err(|_| AotError::UserError(String::from("re-entrant mutable map borrow")))?;
        if !borrowed.contains_key(&key) {
            let entry_bytes = core::mem::size_of::<K>()
                .saturating_add(core::mem::size_of::<V>())
                .max(16);
            alloc.try_charge(entry_bytes)?;
        }
        borrowed.insert(key, value);
        Ok(())
    }

    #[inline]
    pub fn delete(&self, key: &K, alloc: &LustBudgetAlloc) -> Result<Option<V>, AotError> {
        let mut borrowed = self
            .0
            .try_borrow_mut()
            .map_err(|_| AotError::UserError(String::from("re-entrant mutable map borrow")))?;
        let removed = borrowed.remove(key);
        if removed.is_some() {
            let entry_bytes = core::mem::size_of::<K>()
                .saturating_add(core::mem::size_of::<V>())
                .max(16);
            alloc.refund(entry_bytes);
        }
        Ok(removed)
    }

    #[inline]
    pub fn clear(&self) -> Result<(), AotError> {
        let mut borrowed = self
            .0
            .try_borrow_mut()
            .map_err(|_| AotError::UserError(String::from("re-entrant mutable map borrow")))?;
        borrowed.clear();
        Ok(())
    }

    pub fn keys_in<'a>(
        &self,
        alloc: &'a LustBudgetAlloc,
    ) -> Result<Vec<K, &'a LustBudgetAlloc>, AotError>
    where
        K: Clone,
    {
        let borrowed = self
            .0
            .try_borrow()
            .map_err(|_| AotError::UserError(String::from("re-entrant map borrow")))?;
        let mut out = Vec::new_in(alloc);
        if out.try_reserve_exact(borrowed.len()).is_err() {
            return Err(AotError::from_alloc_failure(
                alloc,
                borrowed.len().saturating_mul(core::mem::size_of::<K>()),
            ));
        }
        for k in borrowed.keys() {
            out.push(k.clone());
        }
        Ok(out)
    }

    pub fn values_in<'a>(
        &self,
        alloc: &'a LustBudgetAlloc,
    ) -> Result<Vec<V, &'a LustBudgetAlloc>, AotError>
    where
        V: Clone,
    {
        let borrowed = self
            .0
            .try_borrow()
            .map_err(|_| AotError::UserError(String::from("re-entrant map borrow")))?;
        let mut out = Vec::new_in(alloc);
        if out.try_reserve_exact(borrowed.len()).is_err() {
            return Err(AotError::from_alloc_failure(
                alloc,
                borrowed.len().saturating_mul(core::mem::size_of::<V>()),
            ));
        }
        for v in borrowed.values() {
            out.push(v.clone());
        }
        Ok(out)
    }
}

impl<K: fmt::Display, V: fmt::Display> fmt::Display for AotMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Ok(borrowed) = self.0.try_borrow() else {
            return write!(f, "{{...}}");
        };
        write!(f, "{{")?;
        for (i, (k, v)) in borrowed.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{k}: {v}")?;
        }
        write!(f, "}}")
    }
}

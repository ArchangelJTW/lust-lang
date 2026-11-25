//! Dynamic fallback value representation (`AotValue`) for Lust Rust AOT.
//!
//! Used only when static MIR types remain `Unknown` / union / heterogeneous containers.
//! Fully `#![no_std] + alloc`, zero external dependencies, and panic-free (Hazard 9).

use crate::alloc::LustBudgetAlloc;
use crate::alloc_crate::borrow::ToOwned;
use crate::alloc_crate::boxed::Box;
use crate::alloc_crate::collections::BTreeMap;
use crate::alloc_crate::collections::btree_map::Entry as BTreeEntry;
use crate::alloc_crate::rc::Rc;
use crate::alloc_crate::string::String;
use crate::alloc_crate::vec::Vec;
use crate::{AotCtx, AotError};
use core::borrow::Borrow;
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
    ///
    /// Probes the underlying map once via [`BTreeEntry`]: existing keys overwrite in-place with
    /// zero budget charge, while vacant keys charge once and insert at the probed entry.
    pub fn map_set(&self, key: AotKey, value: AotValue, ctx: &AotCtx) -> Result<(), AotError> {
        let Self::Map(map) = self else {
            return Err(AotError::UserError(String::from("expected map")));
        };
        let mut borrowed = map
            .try_borrow_mut()
            .map_err(|_| AotError::UserError(String::from("re-entrant mutable map borrow")))?;
        match borrowed.entry(key) {
            BTreeEntry::Occupied(mut occ) => {
                occ.insert(value);
            }
            BTreeEntry::Vacant(vac) => {
                let extra_str = match vac.key() {
                    AotKey::Str(s) => s.len(),
                    _ => 0,
                };
                let entry_bytes = core::mem::size_of::<AotKey>()
                    .saturating_add(core::mem::size_of::<AotValue>())
                    .saturating_add(extra_str);
                ctx.budget_alloc.try_charge(entry_bytes)?;
                vac.insert(value);
            }
        }
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

const K_FX: u64 = 0x517c_c1b7_2722_0a95;
const K_WY1: u64 = 0xa076_1d64_78bd_642f;
const K_WY2: u64 = 0xe703_7ed1_a0b4_28db;

/// Fast 128-bit wyhash-style folding mixer with full 64-bit avalanche.
#[inline(always)]
pub const fn aot_mix64(a: u64, b: u64) -> u64 {
    let m = (a as u128).wrapping_mul(b as u128);
    (m as u64) ^ ((m >> 64) as u64)
}

/// Fast deterministic 64-bit integer hash with avalanche on both low index bits and high H2 tag bits.
#[inline(always)]
pub const fn aot_hash_u64(x: u64) -> u64 {
    aot_mix64(x ^ K_WY1, K_WY2)
}

/// Fast deterministic 64-bit byte-slice hash (FxHash 8-byte word folding + wyhash finalizer).
#[inline]
pub fn aot_hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash = K_WY1 ^ (bytes.len() as u64);
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        if let Ok(arr) = <[u8; 8]>::try_from(chunk) {
            let word = u64::from_le_bytes(arr);
            hash = (hash.rotate_left(5) ^ word).wrapping_mul(K_FX);
        }
    }
    let rem = chunks.remainder();
    let mut tail = 0u64;
    let mut shift = 0u32;
    for &b in rem {
        tail |= (b as u64) << shift;
        shift += 8;
    }
    if !rem.is_empty() {
        hash = (hash.rotate_left(5) ^ tail).wrapping_mul(K_FX);
    }
    aot_mix64(hash ^ (bytes.len() as u64), K_WY2)
}

/// Zero-dependency `#![no_std]` deterministic hashing trait for [`AotRawMap`] and [`AotMap`].
///
/// Implementations MUST preserve the `Borrow<Q>` hash consistency invariant:
/// whenever `K: Borrow<Q>`, `k.aot_hash() == k.borrow().aot_hash()`.
pub trait AotHash {
    fn aot_hash(&self) -> u64;
}

impl<T: AotHash + ?Sized> AotHash for &T {
    #[inline(always)]
    fn aot_hash(&self) -> u64 {
        (**self).aot_hash()
    }
}

impl AotHash for () {
    #[inline(always)]
    fn aot_hash(&self) -> u64 {
        aot_hash_u64(0)
    }
}

impl AotHash for bool {
    #[inline(always)]
    fn aot_hash(&self) -> u64 {
        aot_hash_u64(if *self { 1 } else { 0 })
    }
}

impl AotHash for i8 {
    #[inline(always)]
    fn aot_hash(&self) -> u64 {
        aot_hash_u64(*self as u64)
    }
}

impl AotHash for u8 {
    #[inline(always)]
    fn aot_hash(&self) -> u64 {
        aot_hash_u64(*self as u64)
    }
}

impl AotHash for i16 {
    #[inline(always)]
    fn aot_hash(&self) -> u64 {
        aot_hash_u64(*self as u64)
    }
}

impl AotHash for u16 {
    #[inline(always)]
    fn aot_hash(&self) -> u64 {
        aot_hash_u64(*self as u64)
    }
}

impl AotHash for i32 {
    #[inline(always)]
    fn aot_hash(&self) -> u64 {
        aot_hash_u64(*self as u64)
    }
}

impl AotHash for u32 {
    #[inline(always)]
    fn aot_hash(&self) -> u64 {
        aot_hash_u64(*self as u64)
    }
}

impl AotHash for i64 {
    #[inline(always)]
    fn aot_hash(&self) -> u64 {
        aot_hash_u64(*self as u64)
    }
}

impl AotHash for u64 {
    #[inline(always)]
    fn aot_hash(&self) -> u64 {
        aot_hash_u64(*self)
    }
}

impl AotHash for isize {
    #[inline(always)]
    fn aot_hash(&self) -> u64 {
        aot_hash_u64(*self as u64)
    }
}

impl AotHash for usize {
    #[inline(always)]
    fn aot_hash(&self) -> u64 {
        aot_hash_u64(*self as u64)
    }
}

impl AotHash for char {
    #[inline(always)]
    fn aot_hash(&self) -> u64 {
        aot_hash_u64(*self as u64)
    }
}

impl AotHash for str {
    #[inline]
    fn aot_hash(&self) -> u64 {
        aot_hash_bytes(self.as_bytes())
    }
}

impl AotHash for String {
    #[inline]
    fn aot_hash(&self) -> u64 {
        self.as_str().aot_hash()
    }
}

impl AotHash for Box<str> {
    #[inline]
    fn aot_hash(&self) -> u64 {
        self.as_ref().aot_hash()
    }
}

impl AotHash for [u8] {
    #[inline]
    fn aot_hash(&self) -> u64 {
        aot_hash_bytes(self)
    }
}

impl AotHash for Vec<u8> {
    #[inline]
    fn aot_hash(&self) -> u64 {
        aot_hash_bytes(self.as_slice())
    }
}

impl AotHash for AotKey {
    #[inline]
    fn aot_hash(&self) -> u64 {
        match self {
            Self::Nil => aot_hash_u64(0),
            Self::Bool(b) => aot_hash_u64(if *b { 2 } else { 1 }),
            Self::Int(i) => aot_mix64((*i as u64) ^ 3, K_WY2),
            Self::Str(s) => aot_mix64(s.as_str().aot_hash() ^ 4, K_WY2),
        }
    }
}

impl AotHash for AotValue {
    #[inline]
    fn aot_hash(&self) -> u64 {
        match self {
            Self::Nil => aot_hash_u64(0x10),
            Self::Bool(b) => aot_hash_u64(if *b { 0x12 } else { 0x11 }),
            Self::Int(i) => aot_mix64((*i as u64) ^ 0x13, K_WY2),
            Self::Float(f) => {
                let bits = if *f == 0.0 {
                    0u64
                } else if f.is_nan() {
                    0x7ff8_0000_0000_0000
                } else {
                    f.to_bits()
                };
                aot_mix64(bits ^ 0x14, K_WY2)
            }
            Self::Str(s) => aot_mix64(s.as_str().aot_hash() ^ 0x15, K_WY2),
            Self::Array(rc) => aot_mix64((Rc::as_ptr(rc) as usize as u64) ^ 0x16, K_WY2),
            Self::Map(rc) => aot_mix64((Rc::as_ptr(rc) as usize as u64) ^ 0x17, K_WY2),
            Self::Struct(rc) => aot_mix64((Rc::as_ptr(rc) as usize as u64) ^ 0x18, K_WY2),
            Self::Enum(rc) => aot_mix64((Rc::as_ptr(rc) as usize as u64) ^ 0x19, K_WY2),
            Self::Closure(rc) => {
                let ptr = Rc::as_ptr(rc) as *const () as usize as u64;
                aot_mix64(ptr ^ 0x1a, K_WY2)
            }
        }
    }
}

impl AotHash for crate::schema::AotObjectId {
    #[inline]
    fn aot_hash(&self) -> u64 {
        aot_hash_bytes(&self.0)
    }
}

impl AotHash for crate::schema::AotDecimal128 {
    #[inline]
    fn aot_hash(&self) -> u64 {
        let lo = self.0 as u64;
        let hi = (self.0 >> 64) as u64;
        aot_mix64(lo ^ K_WY1, hi ^ K_WY2)
    }
}

const CTRL_EMPTY: u8 = 0x00;
const INITIAL_MAP_CAP: usize = 16;

#[inline(always)]
const fn h2_tag(hash: u64) -> u8 {
    0x80 | ((hash >> 57) as u8)
}

/// Zero-dependency `#![no_std] + alloc`, 100% safe open-addressing flat hash map with
/// 7-bit H2 control bytes, backward-shift deletion (zero tombstones), single-probe `set` /
/// `set_borrowed`, and `Borrow<Q>` zero-copy lookups.
///
/// Suitable for direct unboxed storage (`LocalStorage::Direct`) of non-aliasing `Map<K, V>` locals
/// as well as the backing table inside [`AotMap<K, V>`].
pub struct AotRawMap<K, V> {
    ctrl: Vec<u8>,
    entries: Vec<Option<(K, V)>>,
    len: usize,
}

impl<K, V> AotRawMap<K, V> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            ctrl: Vec::new(),
            entries: Vec::new(),
            len: 0,
        }
    }

    #[inline]
    pub fn with_capacity(min_cap: usize) -> Self {
        if min_cap == 0 {
            return Self::new();
        }
        let target = min_cap
            .saturating_mul(4)
            .div_ceil(3)
            .max(INITIAL_MAP_CAP)
            .checked_next_power_of_two()
            .unwrap_or(INITIAL_MAP_CAP);
        let mut ctrl = Vec::new();
        let mut entries = Vec::new();
        if ctrl.try_reserve_exact(target).is_err() || entries.try_reserve_exact(target).is_err() {
            return Self::new();
        }
        ctrl.resize(target, CTRL_EMPTY);
        entries.resize_with(target, || None);
        Self {
            ctrl,
            entries,
            len: 0,
        }
    }

    #[inline(always)]
    const fn entry_bytes() -> usize {
        let sz = core::mem::size_of::<K>().saturating_add(core::mem::size_of::<V>());
        if sz < 16 { 16 } else { sz }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.ctrl.len()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.ctrl.fill(CTRL_EMPTY);
        for slot in &mut self.entries {
            *slot = None;
        }
        self.len = 0;
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.entries
            .iter()
            .filter_map(|slot| slot.as_ref().map(|(k, v)| (k, v)))
    }

    #[inline]
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.entries
            .iter()
            .filter_map(|slot| slot.as_ref().map(|(k, _)| k))
    }

    #[inline]
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.entries
            .iter()
            .filter_map(|slot| slot.as_ref().map(|(_, v)| v))
    }

    pub fn keys_in<'a>(
        &self,
        alloc: &'a LustBudgetAlloc,
    ) -> Result<Vec<K, &'a LustBudgetAlloc>, AotError>
    where
        K: Clone,
    {
        let mut out = Vec::new_in(alloc);
        if out.try_reserve_exact(self.len).is_err() {
            return Err(AotError::from_alloc_failure(
                alloc,
                self.len.saturating_mul(core::mem::size_of::<K>()),
            ));
        }
        for k in self.keys() {
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
        let mut out = Vec::new_in(alloc);
        if out.try_reserve_exact(self.len).is_err() {
            return Err(AotError::from_alloc_failure(
                alloc,
                self.len.saturating_mul(core::mem::size_of::<V>()),
            ));
        }
        for v in self.values() {
            out.push(v.clone());
        }
        Ok(out)
    }

    #[inline(always)]
    fn needs_grow_for_one_more(&self) -> bool {
        let cap = self.ctrl.len();
        cap == 0 || (self.len.saturating_add(1)).saturating_mul(4) > cap.saturating_mul(3)
    }
}

impl<K: AotHash + Eq, V> AotRawMap<K, V> {
    /// Single-probe lookup returning `Ok(occupied_idx)` if found or `Err(first_empty_idx)` if absent
    /// (`Err(usize::MAX)` when the table is unallocated or full).
    #[inline]
    fn find_slot<Q: ?Sized>(&self, key: &Q, hash: u64) -> Result<usize, usize>
    where
        K: Borrow<Q>,
        Q: Eq,
    {
        let cap = self.ctrl.len();
        if cap == 0 {
            return Err(usize::MAX);
        }
        let mask = cap - 1;
        let tag = h2_tag(hash);
        let mut idx = (hash as usize) & mask;
        let mut probes = 0usize;
        while probes < cap {
            let Some(&c) = self.ctrl.get(idx) else {
                return Err(usize::MAX);
            };
            if c == tag {
                if let Some(Some((k, _))) = self.entries.get(idx) {
                    if k.borrow() == key {
                        return Ok(idx);
                    }
                }
            } else if c == CTRL_EMPTY {
                return Err(idx);
            }
            idx = (idx + 1) & mask;
            probes += 1;
        }
        Err(usize::MAX)
    }

    #[cold]
    fn grow_table(&mut self, alloc: &LustBudgetAlloc) -> Result<(), AotError> {
        let old_cap = self.ctrl.len();
        let new_cap = if old_cap == 0 {
            INITIAL_MAP_CAP
        } else {
            old_cap
                .checked_mul(2)
                .ok_or_else(|| AotError::from_alloc_failure(alloc, usize::MAX))?
        };
        let mut new_ctrl = Vec::new();
        if new_ctrl.try_reserve_exact(new_cap).is_err() {
            return Err(AotError::from_alloc_failure(alloc, new_cap));
        }
        new_ctrl.resize(new_cap, CTRL_EMPTY);

        let mut new_entries: Vec<Option<(K, V)>> = Vec::new();
        let entries_bytes = new_cap.saturating_mul(core::mem::size_of::<Option<(K, V)>>());
        if new_entries.try_reserve_exact(new_cap).is_err() {
            return Err(AotError::from_alloc_failure(alloc, entries_bytes));
        }
        new_entries.resize_with(new_cap, || None);

        let mask = new_cap - 1;
        for slot in self.entries.drain(..) {
            if let Some((k, v)) = slot {
                let hash = k.aot_hash();
                let tag = h2_tag(hash);
                let mut idx = (hash as usize) & mask;
                let mut probes = 0usize;
                while probes < new_cap {
                    if let Some(c) = new_ctrl.get_mut(idx) {
                        if *c == CTRL_EMPTY {
                            *c = tag;
                            if let Some(dst) = new_entries.get_mut(idx) {
                                *dst = Some((k, v));
                            }
                            break;
                        }
                    }
                    idx = (idx + 1) & mask;
                    probes += 1;
                }
            }
        }
        self.ctrl = new_ctrl;
        self.entries = new_entries;
        Ok(())
    }

    #[inline]
    fn insert_at_vacant_after_grow(&mut self, hash: u64, key: K, value: V) {
        let cap = self.ctrl.len();
        if cap == 0 {
            return;
        }
        let mask = cap - 1;
        let tag = h2_tag(hash);
        let mut idx = (hash as usize) & mask;
        let mut probes = 0usize;
        while probes < cap {
            if let Some(c) = self.ctrl.get_mut(idx) {
                if *c == CTRL_EMPTY {
                    *c = tag;
                    if let Some(dst) = self.entries.get_mut(idx) {
                        *dst = Some((key, value));
                    }
                    self.len = self.len.saturating_add(1);
                    return;
                }
            }
            idx = (idx + 1) & mask;
            probes += 1;
        }
    }

    #[inline]
    pub fn has<Q: ?Sized>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: AotHash + Eq,
    {
        let hash = key.aot_hash();
        self.find_slot(key, hash).is_ok()
    }

    #[inline]
    pub fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: AotHash + Eq,
    {
        self.has(key)
    }

    #[inline]
    pub fn get<Q: ?Sized>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: AotHash + Eq,
    {
        let hash = key.aot_hash();
        let idx = self.find_slot(key, hash).ok()?;
        match self.entries.get(idx) {
            Some(Some((_, v))) => Some(v),
            _ => None,
        }
    }

    #[inline]
    pub fn get_mut<Q: ?Sized>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: AotHash + Eq,
    {
        let hash = key.aot_hash();
        let idx = self.find_slot(key, hash).ok()?;
        match self.entries.get_mut(idx) {
            Some(Some((_, v))) => Some(v),
            _ => None,
        }
    }

    #[inline]
    pub fn get_opt<Q: ?Sized>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: AotHash + Eq,
        V: Clone,
    {
        self.get(key).cloned()
    }

    #[inline]
    pub fn get_or_err<Q: ?Sized>(&self, key: &Q) -> Result<V, AotError>
    where
        K: Borrow<Q>,
        Q: AotHash + Eq,
        V: Clone,
    {
        self.get(key).cloned().ok_or(AotError::MissingMapKey)
    }

    /// Single-probe insert or in-place overwrite with budget enforcement on vacant inserts only.
    #[inline]
    pub fn set(&mut self, key: K, value: V, alloc: &LustBudgetAlloc) -> Result<(), AotError> {
        let hash = key.aot_hash();
        match self.find_slot(&key, hash) {
            Ok(idx) => {
                if let Some(Some((_, v_slot))) = self.entries.get_mut(idx) {
                    *v_slot = value;
                }
                Ok(())
            }
            Err(vacant_idx) => {
                let entry_bytes = Self::entry_bytes();
                alloc.try_charge(entry_bytes)?;
                if self.needs_grow_for_one_more() || vacant_idx == usize::MAX {
                    if let Err(e) = self.grow_table(alloc) {
                        alloc.refund(entry_bytes);
                        return Err(e);
                    }
                    self.insert_at_vacant_after_grow(hash, key, value);
                } else {
                    if let Some(c) = self.ctrl.get_mut(vacant_idx) {
                        *c = h2_tag(hash);
                    }
                    if let Some(dst) = self.entries.get_mut(vacant_idx) {
                        *dst = Some((key, value));
                    }
                    self.len = self.len.saturating_add(1);
                }
                Ok(())
            }
        }
    }

    /// Single-probe borrowed insert or zero-allocation in-place overwrite.
    ///
    /// Probes with `&Q` first; when the key already exists, updates its value in-place with
    /// zero budget charges and zero `make_key` / heap allocations. Calls `make_key(key)` only
    /// when inserting a brand-new key after budget enforcement succeeds.
    #[inline]
    pub fn set_with<Q: ?Sized, F>(
        &mut self,
        key: &Q,
        value: V,
        make_key: F,
        alloc: &LustBudgetAlloc,
    ) -> Result<(), AotError>
    where
        K: Borrow<Q>,
        Q: AotHash + Eq,
        F: FnOnce(&Q) -> K,
    {
        let hash = key.aot_hash();
        match self.find_slot(key, hash) {
            Ok(idx) => {
                if let Some(Some((_, v_slot))) = self.entries.get_mut(idx) {
                    *v_slot = value;
                }
                Ok(())
            }
            Err(vacant_idx) => {
                let entry_bytes = Self::entry_bytes();
                alloc.try_charge(entry_bytes)?;
                if self.needs_grow_for_one_more() || vacant_idx == usize::MAX {
                    if let Err(e) = self.grow_table(alloc) {
                        alloc.refund(entry_bytes);
                        return Err(e);
                    }
                    let owned_key = make_key(key);
                    self.insert_at_vacant_after_grow(hash, owned_key, value);
                } else {
                    let owned_key = make_key(key);
                    if let Some(c) = self.ctrl.get_mut(vacant_idx) {
                        *c = h2_tag(hash);
                    }
                    if let Some(dst) = self.entries.get_mut(vacant_idx) {
                        *dst = Some((owned_key, value));
                    }
                    self.len = self.len.saturating_add(1);
                }
                Ok(())
            }
        }
    }

    /// Single-probe borrowed insert or zero-allocation in-place overwrite via [`ToOwned`].
    #[inline]
    pub fn set_borrowed<Q: ?Sized>(
        &mut self,
        key: &Q,
        value: V,
        alloc: &LustBudgetAlloc,
    ) -> Result<(), AotError>
    where
        K: Borrow<Q>,
        Q: AotHash + Eq + ToOwned<Owned = K>,
    {
        self.set_with(key, value, ToOwned::to_owned, alloc)
    }

    /// Removes `key` using backward-shift deletion (zero tombstones) and refunds entry bytes.
    #[inline]
    pub fn delete<Q: ?Sized>(&mut self, key: &Q, alloc: &LustBudgetAlloc) -> Option<V>
    where
        K: Borrow<Q>,
        Q: AotHash + Eq,
    {
        let hash = key.aot_hash();
        let hole = self.find_slot(key, hash).ok()?;
        let cap = self.ctrl.len();
        if cap == 0 {
            return None;
        }
        let mask = cap - 1;
        let removed = self.entries.get_mut(hole)?.take()?;
        if let Some(c) = self.ctrl.get_mut(hole) {
            *c = CTRL_EMPTY;
        }
        self.len = self.len.saturating_sub(1);

        // Backward-shift deletion preserves linear-probing chains with zero tombstones.
        let mut empty = hole;
        let mut cur = (empty + 1) & mask;
        let mut steps = 0usize;
        while steps < cap {
            let Some(&cur_ctrl) = self.ctrl.get(cur) else {
                break;
            };
            if cur_ctrl == CTRL_EMPTY {
                break;
            }
            let should_shift = if let Some(Some((k, _))) = self.entries.get(cur) {
                let home = (k.aot_hash() as usize) & mask;
                let dist_empty = empty.wrapping_sub(home) & mask;
                let dist_cur = cur.wrapping_sub(home) & mask;
                dist_empty < dist_cur
            } else {
                false
            };
            if should_shift {
                if let Some(c_empty) = self.ctrl.get_mut(empty) {
                    *c_empty = cur_ctrl;
                }
                if let Some(c_cur) = self.ctrl.get_mut(cur) {
                    *c_cur = CTRL_EMPTY;
                }
                if empty < self.entries.len() && cur < self.entries.len() {
                    self.entries.swap(empty, cur);
                }
                empty = cur;
            }
            cur = (cur + 1) & mask;
            steps += 1;
        }

        alloc.refund(Self::entry_bytes());
        Some(removed.1)
    }
}

impl<K: Clone, V: Clone> Clone for AotRawMap<K, V> {
    fn clone(&self) -> Self {
        Self {
            ctrl: self.ctrl.clone(),
            entries: self.entries.clone(),
            len: self.len,
        }
    }
}

impl<K, V> Default for AotRawMap<K, V> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<K: AotHash + Eq, V: PartialEq> PartialEq for AotRawMap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        for (k, v) in self.iter() {
            match other.get(k) {
                Some(ov) if v == ov => {}
                _ => return false,
            }
        }
        true
    }
}

impl<K: AotHash + Eq, V: Eq> Eq for AotRawMap<K, V> {}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for AotRawMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K: fmt::Display, V: fmt::Display> fmt::Display for AotRawMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        if self.len <= 1 {
            for (i, (k, v)) in self.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{k}: {v}")?;
            }
        } else {
            let mut items: Vec<(String, &V)> = Vec::new();
            if items.try_reserve_exact(self.len).is_ok() {
                for (k, v) in self.iter() {
                    let mut ks = String::new();
                    let _ = write!(&mut ks, "{k}");
                    items.push((ks, v));
                }
                items.sort_by(|a, b| a.0.cmp(&b.0));
                for (i, (ks, v)) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{ks}: {v}")?;
                }
            } else {
                for (i, (k, v)) in self.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k}: {v}")?;
                }
            }
        }
        write!(f, "}}")
    }
}

/// Typed, reference-counted, panic-free flat open-addressing hash map for Tier 4 Rust AOT (`Map<K, V>`).
///
/// Wraps `Rc<RefCell<AotRawMap<K, V>>>` so `Clone` preserves Lust reference aliasing semantics
/// across local aliases, closures, and function calls while providing O(1) lookups, single-probe
/// `set`/`set_borrowed`, `Borrow<Q>` zero-copy queries, and `LustBudgetAlloc` enforcement with
/// zero `unsafe` and zero panics (Hazard 9 / Hazard 12).
#[derive(Debug)]
pub struct AotMap<K, V>(pub Rc<RefCell<AotRawMap<K, V>>>);

/// Alias for [`AotMap<K, V>`] emphasizing its flat open-addressing hash-table implementation.
pub type AotHashMap<K, V> = AotMap<K, V>;

impl<K, V> Clone for AotMap<K, V> {
    #[inline]
    fn clone(&self) -> Self {
        Self(Rc::clone(&self.0))
    }
}

impl<K, V> Default for AotMap<K, V> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<K: AotHash + Eq, V: PartialEq> PartialEq for AotMap<K, V> {
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

impl<K: AotHash + Eq, V: Eq> Eq for AotMap<K, V> {}

impl<K, V> AotMap<K, V> {
    #[inline]
    pub fn new() -> Self {
        Self(Rc::new(RefCell::new(AotRawMap::new())))
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self(Rc::new(RefCell::new(AotRawMap::with_capacity(cap))))
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
        borrowed.keys_in(alloc)
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
        borrowed.values_in(alloc)
    }
}

impl<K: AotHash + Eq, V> AotMap<K, V> {
    #[inline]
    pub fn has<Q: ?Sized>(&self, key: &Q) -> Result<bool, AotError>
    where
        K: Borrow<Q>,
        Q: AotHash + Eq,
    {
        let borrowed = self
            .0
            .try_borrow()
            .map_err(|_| AotError::UserError(String::from("re-entrant map borrow")))?;
        Ok(borrowed.has(key))
    }

    #[inline]
    pub fn get<Q: ?Sized>(&self, key: &Q) -> Result<V, AotError>
    where
        K: Borrow<Q>,
        Q: AotHash + Eq,
        V: Clone,
    {
        let borrowed = self
            .0
            .try_borrow()
            .map_err(|_| AotError::UserError(String::from("re-entrant map borrow")))?;
        borrowed.get_or_err(key)
    }

    #[inline]
    pub fn get_opt<Q: ?Sized>(&self, key: &Q) -> Result<Option<V>, AotError>
    where
        K: Borrow<Q>,
        Q: AotHash + Eq,
        V: Clone,
    {
        let borrowed = self
            .0
            .try_borrow()
            .map_err(|_| AotError::UserError(String::from("re-entrant map borrow")))?;
        Ok(borrowed.get_opt(key))
    }

    #[inline]
    pub fn set(&self, key: K, value: V, alloc: &LustBudgetAlloc) -> Result<(), AotError> {
        let mut borrowed = self
            .0
            .try_borrow_mut()
            .map_err(|_| AotError::UserError(String::from("re-entrant mutable map borrow")))?;
        borrowed.set(key, value, alloc)
    }

    #[inline]
    pub fn set_borrowed<Q: ?Sized>(
        &self,
        key: &Q,
        value: V,
        alloc: &LustBudgetAlloc,
    ) -> Result<(), AotError>
    where
        K: Borrow<Q>,
        Q: AotHash + Eq + ToOwned<Owned = K>,
    {
        let mut borrowed = self
            .0
            .try_borrow_mut()
            .map_err(|_| AotError::UserError(String::from("re-entrant mutable map borrow")))?;
        borrowed.set_borrowed(key, value, alloc)
    }

    #[inline]
    pub fn set_with<Q: ?Sized, F>(
        &self,
        key: &Q,
        value: V,
        make_key: F,
        alloc: &LustBudgetAlloc,
    ) -> Result<(), AotError>
    where
        K: Borrow<Q>,
        Q: AotHash + Eq,
        F: FnOnce(&Q) -> K,
    {
        let mut borrowed = self
            .0
            .try_borrow_mut()
            .map_err(|_| AotError::UserError(String::from("re-entrant mutable map borrow")))?;
        borrowed.set_with(key, value, make_key, alloc)
    }

    #[inline]
    pub fn delete<Q: ?Sized>(&self, key: &Q, alloc: &LustBudgetAlloc) -> Result<Option<V>, AotError>
    where
        K: Borrow<Q>,
        Q: AotHash + Eq,
    {
        let mut borrowed = self
            .0
            .try_borrow_mut()
            .map_err(|_| AotError::UserError(String::from("re-entrant mutable map borrow")))?;
        Ok(borrowed.delete(key, alloc))
    }
}

impl<K: fmt::Display, V: fmt::Display> fmt::Display for AotMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Ok(borrowed) = self.0.try_borrow() else {
            return write!(f, "{{...}}");
        };
        fmt::Display::fmt(&*borrowed, f)
    }
}

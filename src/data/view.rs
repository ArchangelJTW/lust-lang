use super::number::{DataError, ExactNumber};
use super::owner::{LeaseError, OwnerId, OwnerRegistry};
use super::value::DataValue;
use alloc::{string::ToString, vec::Vec};
use core::fmt;

/// Zero-copy borrowed view over contiguous buffer memory stamped with an OwnerId
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataView<'s> {
    Null,
    Bool(bool),
    Number(ExactNumber),
    Str(&'s str, OwnerId),
    Binary(&'s [u8], OwnerId),
}

impl<'s> DataView<'s> {
    pub const fn null() -> Self {
        DataView::Null
    }

    pub const fn from_bool(b: bool) -> Self {
        DataView::Bool(b)
    }

    pub const fn from_number(n: ExactNumber) -> Self {
        DataView::Number(n)
    }

    pub const fn from_str(s: &'s str, owner: OwnerId) -> Self {
        DataView::Str(s, owner)
    }

    pub const fn from_binary(bytes: &'s [u8], owner: OwnerId) -> Self {
        DataView::Binary(bytes, owner)
    }

    pub fn owner_id(&self) -> Option<OwnerId> {
        match self {
            DataView::Str(_, owner) | DataView::Binary(_, owner) => Some(*owner),
            _ => None,
        }
    }

    /// Reborrowing creates a sub-slice preserving the parent region token and OwnerId
    pub fn slice(&self, start: usize, end: usize) -> Result<DataView<'s>, DataError> {
        match self {
            DataView::Str(s, owner) => {
                if start <= end && end <= s.len() && s.is_char_boundary(start) && s.is_char_boundary(end) {
                    Ok(DataView::Str(&s[start..end], *owner))
                } else {
                    Err(DataError::new("Slice index out of bounds"))
                }
            }
            DataView::Binary(b, owner) => {
                if start <= end && end <= b.len() {
                    Ok(DataView::Binary(&b[start..end], *owner))
                } else {
                    Err(DataError::new("Slice index out of bounds"))
                }
            }
            _ => Err(DataError::new("Cannot slice non-buffer view")),
        }
    }

    /// Pre-dereference access validation using the OwnerRegistry
    pub fn validate_access(&self, registry: &OwnerRegistry, current_tenant: u32) -> Result<(), LeaseError> {
        if let Some(owner) = self.owner_id() {
            registry.validate_access(owner, current_tenant)
        } else {
            Ok(())
        }
    }

    /// Mandatory materialization when crossing into asynchronous tasks or detached worker queues
    pub fn to_owned(&self) -> DataValue {
        match self {
            DataView::Null => DataValue::Null,
            DataView::Bool(b) => DataValue::Bool(*b),
            DataView::Number(n) => DataValue::Number(*n),
            DataView::Str(s, _) => DataValue::String(s.to_string()),
            DataView::Binary(b, _) => DataValue::Binary(Vec::from(*b)),
        }
    }

    /// Checked dereference: strictly validates tenant and generation before returning borrowed &str
    pub fn as_str_checked(&self, registry: &OwnerRegistry, current_tenant: u32) -> Result<&'s str, LeaseError> {
        self.validate_access(registry, current_tenant)?;
        self.as_str().ok_or(LeaseError::InvalidState("View is not a string"))
    }

    /// Checked dereference: strictly validates tenant and generation before returning borrowed &[u8]
    pub fn as_bytes_checked(&self, registry: &OwnerRegistry, current_tenant: u32) -> Result<&'s [u8], LeaseError> {
        self.validate_access(registry, current_tenant)?;
        self.as_bytes().ok_or(LeaseError::InvalidState("View is not a binary or string buffer"))
    }

    pub fn as_str(&self) -> Option<&'s str> {
        match self {
            DataView::Str(s, _) => Some(s),
            _ => None,
        }
    }

    pub fn as_bytes(&self) -> Option<&'s [u8]> {
        match self {
            DataView::Binary(b, _) => Some(b),
            DataView::Str(s, _) => Some(s.as_bytes()),
            _ => None,
        }
    }
}

impl<'s> fmt::Display for DataView<'s> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataView::Null => write!(f, "null"),
            DataView::Bool(b) => write!(f, "{}", b),
            DataView::Number(n) => write!(f, "{}", n),
            DataView::Str(s, owner) => write!(f, "DataView::Str(\"{}\", {})", s, owner),
            DataView::Binary(b, owner) => write!(f, "DataView::Binary(len={}, {})", b.len(), owner),
        }
    }
}

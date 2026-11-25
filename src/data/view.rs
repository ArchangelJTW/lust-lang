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

    pub fn is_null(&self) -> bool {
        matches!(self, DataView::Null)
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, DataView::Bool(_))
    }

    pub fn is_number(&self) -> bool {
        matches!(self, DataView::Number(_))
    }

    pub fn is_str(&self) -> bool {
        matches!(self, DataView::Str(_, _))
    }

    pub fn is_binary(&self) -> bool {
        matches!(self, DataView::Binary(_, _))
    }

    pub fn len(&self) -> usize {
        match self {
            DataView::Str(s, _) => s.len(),
            DataView::Binary(b, _) => b.len(),
            _ => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            DataView::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_number(&self) -> Option<ExactNumber> {
        match self {
            DataView::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Checked slice: validates tenant access through registry before slicing
    pub fn slice_checked(
        &self,
        start: usize,
        end: usize,
        registry: &OwnerRegistry,
        current_tenant: u32,
    ) -> Result<DataView<'s>, DataError> {
        self.validate_access(registry, current_tenant)
            .map_err(|_| DataError::new("Lease check failed during slice"))?;
        self.slice(start, end)
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

    /// Checked materialization: refuses to clone if owner lease has been cancelled, released, or mismatched
    pub fn to_owned_checked(
        &self,
        registry: &OwnerRegistry,
        current_tenant: u32,
    ) -> Result<DataValue, LeaseError> {
        self.validate_access(registry, current_tenant)?;
        Ok(self.to_owned())
    }

    /// Budget-enforced materialization: charges allocated bytes against the budget tracker
    pub fn to_owned_with_budget(
        &self,
        budget: &mut crate::data::budget::BudgetTracker,
    ) -> Result<DataValue, crate::data::budget::BudgetError> {
        let bytes_needed = self.len();
        if bytes_needed > 0 {
            budget.charge_bytes(bytes_needed)?;
        }
        Ok(self.to_owned())
    }

    /// Combined checked and budgeted materialization: ensures valid lease and non-exceeded budget
    pub fn to_owned_checked_with_budget(
        &self,
        registry: &OwnerRegistry,
        current_tenant: u32,
        budget: &mut crate::data::budget::BudgetTracker,
    ) -> Result<DataValue, DataError> {
        self.validate_access(registry, current_tenant)
            .map_err(|_| DataError::new("Lease check failed"))?;
        self.to_owned_with_budget(budget)
            .map_err(|_| DataError::new("Budget check failed"))
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

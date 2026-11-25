use super::value::DataValue;
use alloc::{string::String, vec::Vec};
use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DuplicateKeyError;

impl fmt::Display for DuplicateKeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Duplicate key in document")
    }
}

/// Ordered map of unique string keys to DataValue entries
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Document {
    pub(crate) entries: Vec<(String, DataValue)>,
}

impl Document {
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.iter().any(|(k, _)| k == key)
    }

    pub fn get(&self, key: &str) -> Option<&DataValue> {
        self.entries
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut DataValue> {
        self.entries
            .iter_mut()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    /// Inserts or replaces a key-value pair, preserving insertion order on update
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<DataValue>) -> Option<DataValue> {
        let key_str = key.into();
        let val = value.into();
        if let Some(pos) = self.entries.iter().position(|(k, _)| k == &key_str) {
            let old = core::mem::replace(&mut self.entries[pos].1, val);
            Some(old)
        } else {
            self.entries.push((key_str, val));
            None
        }
    }

    /// Fails closed if the key already exists (used to enforce Gate-SCHEMA-01 duplicate key rejection)
    pub fn try_insert(
        &mut self,
        key: impl Into<String>,
        value: impl Into<DataValue>,
    ) -> Result<(), DuplicateKeyError> {
        let key_str = key.into();
        if self.contains_key(&key_str) {
            Err(DuplicateKeyError)
        } else {
            self.entries.push((key_str, value.into()));
            Ok(())
        }
    }

    pub fn remove(&mut self, key: &str) -> Option<DataValue> {
        if let Some(pos) = self.entries.iter().position(|(k, _)| k == key) {
            Some(self.entries.remove(pos).1)
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &DataValue)> {
        self.entries.iter().map(|(k, v)| (k, v))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&String, &mut DataValue)> {
        self.entries.iter_mut().map(|(k, v)| (&*k, v))
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.entries.iter().map(|(k, _)| k)
    }

    pub fn values(&self) -> impl Iterator<Item = &DataValue> {
        self.entries.iter().map(|(_, v)| v)
    }

    pub fn entries(&self) -> &[(String, DataValue)] {
        &self.entries
    }
}

impl IntoIterator for Document {
    type Item = (String, DataValue);
    type IntoIter = std::vec::IntoIter<(String, DataValue)>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl<K: Into<String>, V: Into<DataValue>> FromIterator<(K, V)> for Document {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut doc = Document::new();
        for (k, v) in iter {
            doc.insert(k, v);
        }
        doc
    }
}

impl fmt::Display for Document {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        for (i, (k, v)) in self.entries.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "\"{}\": {}", k, v)?;
        }
        write!(f, "}}")
    }
}

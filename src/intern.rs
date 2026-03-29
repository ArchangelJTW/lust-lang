//! String interning for memory-efficient compilation.
//!
//! Stores each unique string once and returns a cheap-to-copy handle.
//! Dramatically reduces memory usage for repeated identifiers and keywords.

use alloc::{
    string::{String, ToString},
    vec::Vec,
};
use hashbrown::HashMap;

/// A cheap-to-copy handle to an interned string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol(usize);

/// String interner that deduplicates strings.
#[derive(Debug, Default)]
pub struct Interner {
    /// Storage for all interned strings
    strings: Vec<String>,
    /// Map from string content to index for O(1) lookup
    indices: HashMap<String, usize>,
}

impl Interner {
    /// Create a new empty interner
    pub fn new() -> Self {
        Self {
            strings: Vec::new(),
            indices: HashMap::new(),
        }
    }

    /// Create an interner with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            strings: Vec::with_capacity(capacity),
            indices: HashMap::with_capacity(capacity),
        }
    }

    /// Intern a string, returning a Symbol handle.
    /// If the string already exists, returns the existing Symbol.
    pub fn intern(&mut self, s: &str) -> Symbol {
        if let Some(&idx) = self.indices.get(s) {
            return Symbol(idx);
        }

        let idx = self.strings.len();
        self.strings.push(s.to_string());
        self.indices.insert(s.to_string(), idx);
        Symbol(idx)
    }

    /// Get the string for a Symbol
    pub fn get(&self, symbol: Symbol) -> &str {
        &self.strings[symbol.0]
    }

    /// Returns number of unique strings
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Returns true if empty
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }

    /// Returns total bytes used by all strings (not including overhead)
    pub fn total_string_bytes(&self) -> usize {
        self.strings.iter().map(|s| s.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interning() {
        let mut interner = Interner::new();

        let s1 = interner.intern("hello");
        let s2 = interner.intern("world");
        let s3 = interner.intern("hello");

        assert_eq!(s1, s3); // Same string returns same symbol
        assert_ne!(s1, s2); // Different strings return different symbols

        assert_eq!(interner.get(s1), "hello");
        assert_eq!(interner.get(s2), "world");
        assert_eq!(interner.len(), 2); // Only 2 unique strings
    }
}

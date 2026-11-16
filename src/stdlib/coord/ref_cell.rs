//! Coordination Ref: deterministic, shared mutable state cell for fibers.
//!
//! Complies with `RFC-STDLIB-ASYNC-COORDINATION` (§3.4):
//! - Cooperative, zero-OS-thread reference cell (`Rc<RefCell<T>>`).
//! - Pure functional transformations via `update` and `modify`.
//! - Safe sharing across fibers within a cooperative VM context.

use alloc::rc::Rc;
use core::cell::RefCell;

/// Deterministic, shared mutable cell for fiber coordination.
#[derive(Clone, Debug)]
pub struct Ref<T> {
    inner: Rc<RefCell<T>>,
}

impl<T: Clone> Ref<T> {
    /// Creates a new `Ref` with the given initial value.
    pub fn new(initial: T) -> Self {
        Self {
            inner: Rc::new(RefCell::new(initial)),
        }
    }

    /// Alias for `new`.
    pub fn make(initial: T) -> Self {
        Self::new(initial)
    }

    /// Reads the current value of the cell.
    pub fn get(&self) -> T {
        self.inner.borrow().clone()
    }

    /// Overwrites the cell with a new value.
    pub fn set(&self, val: T) {
        *self.inner.borrow_mut() = val;
    }

    /// Atomically updates the cell by applying `f` to the current value.
    /// Returns the newly updated value.
    pub fn update<F: FnOnce(T) -> T>(&self, f: F) -> T {
        let mut b = self.inner.borrow_mut();
        let new_val = f(b.clone());
        *b = new_val.clone();
        new_val
    }

    /// Atomically modifies the cell by applying `f` which returns `(result, new_val)`.
    /// Sets the cell to `new_val` and returns `result`.
    pub fn modify<B, F: FnOnce(T) -> (B, T)>(&self, f: F) -> (B, T) {
        let mut b = self.inner.borrow_mut();
        let (res, new_val) = f(b.clone());
        *b = new_val.clone();
        (res, new_val)
    }
}

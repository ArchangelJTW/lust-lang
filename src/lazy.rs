#![allow(dead_code)]

#[cfg(feature = "std")]
use std::sync::OnceLock;

#[cfg(not(feature = "std"))]
use core::cell::OnceCell;

pub struct StaticOnceCell<T> {
    #[cfg(feature = "std")]
    inner: OnceLock<T>,
    #[cfg(not(feature = "std"))]
    inner: OnceCell<T>,
}

impl<T> StaticOnceCell<T> {
    pub const fn new() -> Self {
        Self {
            #[cfg(feature = "std")]
            inner: OnceLock::new(),
            #[cfg(not(feature = "std"))]
            inner: OnceCell::new(),
        }
    }

    pub fn get_or_init<F>(&self, f: F) -> &T
    where
        F: FnOnce() -> T,
    {
        self.inner.get_or_init(f)
    }
}

#[cfg(not(feature = "std"))]
unsafe impl<T: Sync> Sync for StaticOnceCell<T> {}
#[cfg(not(feature = "std"))]
unsafe impl<T: Send> Send for StaticOnceCell<T> {}

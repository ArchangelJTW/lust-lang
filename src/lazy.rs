#![allow(dead_code)]

use core::cell::OnceCell;

pub struct StaticOnceCell<T> {
    inner: OnceCell<T>,
}

impl<T> StaticOnceCell<T> {
    pub const fn new() -> Self {
        Self {
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

unsafe impl<T: Sync> Sync for StaticOnceCell<T> {}
unsafe impl<T: Send> Send for StaticOnceCell<T> {}

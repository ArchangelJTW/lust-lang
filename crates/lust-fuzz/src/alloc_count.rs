//! A global allocator that counts the live allocations made on each thread,
//! so a worker can tell whether running a program left anything behind. A
//! leaked reference count leaks its allocation, which the outputs never
//! show; this does.
//!
//! The count is per thread: allocations are attributed to the thread that
//! made them, frees to the thread freeing. A VM and everything it owns live
//! and die on one worker thread, so around one run the count comes back to
//! where it was — apart from lazily initialized statics (the name interner,
//! `OnceLock`s), which is why a leak is only reported when it repeats.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

pub struct Counting;

thread_local! {
    static LIVE: Cell<isize> = const { Cell::new(0) };
}

/// Live allocations attributed to this thread.
pub fn live() -> isize {
    LIVE.with(Cell::get)
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            // A thread-local access can allocate while the thread's storage
            // is being set up or torn down; `try_with` skips those.
            let _ = LIVE.try_with(|c| c.set(c.get() + 1));
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        let _ = LIVE.try_with(|c| c.set(c.get() - 1));
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // One allocation before, one after.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

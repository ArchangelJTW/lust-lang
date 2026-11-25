//! Single-threaded (`!Send + !Sync`) budget-enforcing allocators implementing `core::alloc::Allocator`.
//!
//! Provides zero-atomic memory accounting (`SEC-02`) for `#![no_std] + alloc` Lust AOT binaries:
//! - [`LustBudgetAlloc`]: `Cell<usize>`-backed wrapper over `alloc::alloc::Global`
//! - [`LustChunkArena`]: 256 KiB chunked bump arena backed by `&LustBudgetAlloc`
//! - [`StaticSlabAllocator`]: fixed-size `UnsafeCell<[u8; N]>` bump slab for `no_std` embedded targets

use crate::alloc_crate::alloc::Global;
use crate::alloc_crate::rc::Rc;
use crate::alloc_crate::vec::Vec;
use core::alloc::{AllocError, Allocator, Layout};
use core::cell::{Cell, RefCell, UnsafeCell};
use core::fmt;
use core::marker::PhantomData;
use core::ptr::NonNull;

/// Default chunk size (256 KiB) for [`LustChunkArena`].
pub const DEFAULT_CHUNK_SIZE: usize = 256 * 1024;

/// Error returned when a memory charge would exceed a [`LustBudgetAlloc`] limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetAllocError {
    pub limit: usize,
    pub requested: usize,
}

impl fmt::Display for BudgetAllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Memory budget exceeded: limit {} bytes, requested {}",
            self.limit, self.requested
        )
    }
}

impl core::error::Error for BudgetAllocError {}

/// Shared single-threaded cumulative budget ledger backing [`LustBudgetAlloc::share`].
#[derive(Debug)]
struct BudgetLedger {
    used_bytes: Cell<usize>,
    limit_bytes: Cell<usize>,
    peak_bytes: Cell<usize>,
}

/// Single-threaded (`!Send + !Sync`) budgeting allocator backed by [`Global`].
///
/// Unshared instances track counters inline in `Cell<usize>` with zero heap overhead
/// (supporting `const fn` construction in `AotCtx`). Calling [`LustBudgetAlloc::share`]
/// lazily promotes the counters into a shared `Rc<BudgetLedger>` so clones across effect
/// handlers, continuations, scope finalizers, spawned tasks, and closures charge and
/// refund the exact same cumulative budget ledger.
pub struct LustBudgetAlloc {
    used_bytes: Cell<usize>,
    limit_bytes: Cell<usize>,
    peak_bytes: Cell<usize>,
    is_shared: Cell<bool>,
    shared: RefCell<Option<Rc<BudgetLedger>>>,
    _not_send_sync: PhantomData<*const ()>,
}

impl LustBudgetAlloc {
    #[inline]
    pub const fn unbounded() -> Self {
        Self::with_limit(usize::MAX)
    }

    #[inline]
    pub const fn with_limit(limit_bytes: usize) -> Self {
        Self {
            used_bytes: Cell::new(0),
            limit_bytes: Cell::new(limit_bytes),
            peak_bytes: Cell::new(0),
            is_shared: Cell::new(false),
            shared: RefCell::new(None),
            _not_send_sync: PhantomData,
        }
    }

    /// Creates a new [`LustBudgetAlloc`] handle sharing the exact same cumulative
    /// `used_bytes`, `peak_bytes`, and `limit_bytes` ledger as `self`.
    #[inline]
    pub fn share(&self) -> Self {
        if self.is_shared.get() {
            if let Ok(guard) = self.shared.try_borrow() {
                if let Some(ledger) = guard.as_ref() {
                    return Self::from_ledger(Rc::clone(ledger));
                }
            }
        }
        self.promote_and_share()
    }

    #[cold]
    fn promote_and_share(&self) -> Self {
        if let Ok(mut slot) = self.shared.try_borrow_mut() {
            let ledger = match slot.as_ref() {
                Some(existing) => Rc::clone(existing),
                None => {
                    let created = Rc::new(BudgetLedger {
                        used_bytes: Cell::new(self.used_bytes.get()),
                        limit_bytes: Cell::new(self.limit_bytes.get()),
                        peak_bytes: Cell::new(self.peak_bytes.get()),
                    });
                    *slot = Some(Rc::clone(&created));
                    self.is_shared.set(true);
                    created
                }
            };
            return Self::from_ledger(ledger);
        }
        // Fail-closed fallback if re-entrantly borrowed (unreachable in normal flow;
        // preserves panic-freedom without ever handing out uncounted budget).
        let current_limit = self.limit_bytes.get();
        Self {
            used_bytes: Cell::new(current_limit),
            limit_bytes: Cell::new(0),
            peak_bytes: Cell::new(self.peak_bytes.get()),
            is_shared: Cell::new(false),
            shared: RefCell::new(None),
            _not_send_sync: PhantomData,
        }
    }

    #[inline]
    fn from_ledger(ledger: Rc<BudgetLedger>) -> Self {
        Self {
            used_bytes: Cell::new(0),
            limit_bytes: Cell::new(0),
            peak_bytes: Cell::new(0),
            is_shared: Cell::new(true),
            shared: RefCell::new(Some(ledger)),
            _not_send_sync: PhantomData,
        }
    }

    #[inline(always)]
    fn charge_cells(
        used_cell: &Cell<usize>,
        limit_cell: &Cell<usize>,
        peak_cell: &Cell<usize>,
        delta: usize,
    ) -> Result<(), BudgetAllocError> {
        let cur = used_cell.get();
        let limit = limit_cell.get();
        let next = cur.checked_add(delta).ok_or(BudgetAllocError {
            limit,
            requested: usize::MAX,
        })?;
        if next > limit {
            return Err(BudgetAllocError {
                limit,
                requested: next,
            });
        }
        used_cell.set(next);
        if next > peak_cell.get() {
            peak_cell.set(next);
        }
        Ok(())
    }

    #[inline(always)]
    pub fn used_bytes(&self) -> usize {
        if !self.is_shared.get() {
            return self.used_bytes.get();
        }
        if let Ok(guard) = self.shared.try_borrow() {
            if let Some(ledger) = guard.as_ref() {
                return ledger.used_bytes.get();
            }
        }
        self.used_bytes.get()
    }

    #[inline(always)]
    pub fn peak_bytes(&self) -> usize {
        if !self.is_shared.get() {
            return self.peak_bytes.get();
        }
        if let Ok(guard) = self.shared.try_borrow() {
            if let Some(ledger) = guard.as_ref() {
                return ledger.peak_bytes.get();
            }
        }
        self.peak_bytes.get()
    }

    #[inline(always)]
    pub fn limit_bytes(&self) -> usize {
        if !self.is_shared.get() {
            return self.limit_bytes.get();
        }
        if let Ok(guard) = self.shared.try_borrow() {
            if let Some(ledger) = guard.as_ref() {
                return ledger.limit_bytes.get();
            }
        }
        self.limit_bytes.get()
    }

    #[inline(always)]
    pub fn set_limit_bytes(&self, limit: usize) {
        if !self.is_shared.get() {
            self.limit_bytes.set(limit);
            return;
        }
        if let Ok(guard) = self.shared.try_borrow() {
            if let Some(ledger) = guard.as_ref() {
                ledger.limit_bytes.set(limit);
                return;
            }
        }
        self.limit_bytes.set(limit);
    }

    #[inline(always)]
    pub fn try_charge(&self, delta: usize) -> Result<(), BudgetAllocError> {
        if delta == 0 {
            return Ok(());
        }
        if !self.is_shared.get() {
            return Self::charge_cells(&self.used_bytes, &self.limit_bytes, &self.peak_bytes, delta);
        }
        if let Ok(guard) = self.shared.try_borrow() {
            if let Some(ledger) = guard.as_ref() {
                return Self::charge_cells(
                    &ledger.used_bytes,
                    &ledger.limit_bytes,
                    &ledger.peak_bytes,
                    delta,
                );
            }
        }
        Self::charge_cells(&self.used_bytes, &self.limit_bytes, &self.peak_bytes, delta)
    }

    #[inline(always)]
    pub fn refund(&self, delta: usize) {
        if !self.is_shared.get() {
            self.used_bytes
                .set(self.used_bytes.get().saturating_sub(delta));
            return;
        }
        if let Ok(guard) = self.shared.try_borrow() {
            if let Some(ledger) = guard.as_ref() {
                ledger
                    .used_bytes
                    .set(ledger.used_bytes.get().saturating_sub(delta));
                return;
            }
        }
        self.used_bytes
            .set(self.used_bytes.get().saturating_sub(delta));
    }
}

impl Clone for LustBudgetAlloc {
    #[inline]
    fn clone(&self) -> Self {
        self.share()
    }
}

impl fmt::Debug for LustBudgetAlloc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LustBudgetAlloc")
            .field("used_bytes", &self.used_bytes())
            .field("limit_bytes", &self.limit_bytes())
            .field("peak_bytes", &self.peak_bytes())
            .field("shared", &self.is_shared.get())
            .finish()
    }
}

impl Default for LustBudgetAlloc {
    #[inline]
    fn default() -> Self {
        Self::unbounded()
    }
}

// SAFETY:
// 1. ZST rule: if layout.size() == 0, return NonNull::slice_from_raw_parts(layout.dangling_ptr(), 0)
//    and NEVER call Global.deallocate on a zero-sized layout.
// 2. Rollback rule: if Global.allocate / Global.grow fails, immediately call self.refund(delta)
//    so used_bytes never leaks on OOM.
// 3. Grow rule: when new_layout.size() > old_layout.size(), charge only delta = new - old;
//    Global.grow / Global.grow_zeroed preserves old_layout.size() bytes.
unsafe impl Allocator for LustBudgetAlloc {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let size = layout.size();
        if size == 0 {
            return Ok(NonNull::slice_from_raw_parts(layout.dangling_ptr(), 0));
        }
        self.try_charge(size).map_err(|_| AllocError)?;
        match Global.allocate(layout) {
            Ok(ptr) => Ok(ptr),
            Err(e) => {
                self.refund(size);
                Err(e)
            }
        }
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let size = layout.size();
        if size == 0 {
            return Ok(NonNull::slice_from_raw_parts(layout.dangling_ptr(), 0));
        }
        self.try_charge(size).map_err(|_| AllocError)?;
        match Global.allocate_zeroed(layout) {
            Ok(ptr) => Ok(ptr),
            Err(e) => {
                self.refund(size);
                Err(e)
            }
        }
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let size = layout.size();
        if size != 0 {
            // SAFETY: Caller guarantees `ptr` was allocated by this allocator with `layout`
            // and `size != 0`, so it came directly from `Global`.
            unsafe { Global.deallocate(ptr, layout) };
            self.refund(size);
        }
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let old_size = old_layout.size();
        let new_size = new_layout.size();
        let Some(delta) = new_size.checked_sub(old_size) else {
            return Err(AllocError);
        };
        if new_size == 0 {
            return Ok(NonNull::slice_from_raw_parts(new_layout.dangling_ptr(), 0));
        }
        self.try_charge(delta).map_err(|_| AllocError)?;
        let res = if old_size == 0 {
            Global.allocate(new_layout)
        } else {
            // SAFETY: `old_size != 0`, so `ptr` was allocated by `Global` with `old_layout`.
            unsafe { Global.grow(ptr, old_layout, new_layout) }
        };
        match res {
            Ok(p) => Ok(p),
            Err(e) => {
                self.refund(delta);
                Err(e)
            }
        }
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let old_size = old_layout.size();
        let new_size = new_layout.size();
        let Some(delta) = new_size.checked_sub(old_size) else {
            return Err(AllocError);
        };
        if new_size == 0 {
            return Ok(NonNull::slice_from_raw_parts(new_layout.dangling_ptr(), 0));
        }
        self.try_charge(delta).map_err(|_| AllocError)?;
        let res = if old_size == 0 {
            Global.allocate_zeroed(new_layout)
        } else {
            // SAFETY: `old_size != 0`, so `ptr` was allocated by `Global` with `old_layout`.
            unsafe { Global.grow_zeroed(ptr, old_layout, new_layout) }
        };
        match res {
            Ok(p) => Ok(p),
            Err(e) => {
                self.refund(delta);
                Err(e)
            }
        }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let old_size = old_layout.size();
        let new_size = new_layout.size();
        let Some(delta) = old_size.checked_sub(new_size) else {
            return Err(AllocError);
        };
        if new_size == 0 {
            if old_size != 0 {
                // SAFETY: `old_size != 0`, so `ptr` was allocated by `Global` with `old_layout`.
                unsafe { Global.deallocate(ptr, old_layout) };
                self.refund(old_size);
            }
            return Ok(NonNull::slice_from_raw_parts(new_layout.dangling_ptr(), 0));
        }
        // SAFETY: `new_size > 0` implies `old_size > 0`, so `ptr` came from `Global`.
        let res = unsafe { Global.shrink(ptr, old_layout, new_layout) };
        if res.is_ok() {
            self.refund(delta);
        }
        res
    }
}

/// Chunked single-threaded bump arena backed by `&'a LustBudgetAlloc`.
///
/// Allocates 256 KiB chunks (by default) from `parent`. Individual [`Allocator::deallocate`]
/// calls are no-ops; every chunk is bulk-freed via `parent.deallocate` when the arena drops.
pub struct LustChunkArena<'a> {
    parent: &'a LustBudgetAlloc,
    chunk_size: usize,
    cursor: Cell<*mut u8>,
    end: Cell<*mut u8>,
    chunks: RefCell<Vec<(NonNull<u8>, Layout)>>,
    _not_send_sync: PhantomData<*const ()>,
}

impl<'a> LustChunkArena<'a> {
    #[inline]
    pub fn new(parent: &'a LustBudgetAlloc) -> Self {
        Self::with_chunk_size(parent, DEFAULT_CHUNK_SIZE)
    }

    #[inline]
    pub fn with_chunk_size(parent: &'a LustBudgetAlloc, chunk_size: usize) -> Self {
        Self {
            parent,
            chunk_size: chunk_size.max(1),
            cursor: Cell::new(core::ptr::null_mut()),
            end: Cell::new(core::ptr::null_mut()),
            chunks: RefCell::new(Vec::new()),
            _not_send_sync: PhantomData,
        }
    }

    #[inline(always)]
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    #[inline]
    pub fn chunk_count(&self) -> usize {
        self.chunks.try_borrow().map(|c| c.len()).unwrap_or(0)
    }

    #[inline(always)]
    pub fn parent(&self) -> &'a LustBudgetAlloc {
        self.parent
    }

    #[cold]
    fn allocate_from_new_chunk(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let size = layout.size();
        let chunk_align = layout.align().max(16);
        let is_oversized = size > self.chunk_size;
        let alloc_size = if is_oversized { size } else { self.chunk_size };
        let chunk_layout =
            Layout::from_size_align(alloc_size, chunk_align).map_err(|_| AllocError)?;

        let Ok(mut chunks) = self.chunks.try_borrow_mut() else {
            return Err(AllocError);
        };
        // Reserve tracking slot BEFORE allocating chunk so push cannot fail or leak.
        chunks.try_reserve(1).map_err(|_| AllocError)?;
        let raw_slice = self.parent.allocate(chunk_layout)?;
        let chunk_ptr = raw_slice.cast::<u8>();
        chunks.push((chunk_ptr, chunk_layout));

        if !is_oversized {
            let base = chunk_ptr.as_ptr();
            // SAFETY: `size <= alloc_size == chunk_layout.size()`, so `base.add(size)` and
            // `base.add(alloc_size)` lie within (or one byte past) the allocated chunk.
            unsafe {
                self.cursor.set(base.add(size));
                self.end.set(base.add(alloc_size));
            }
        }

        // `chunk_ptr` is aligned to `chunk_align >= layout.align()`.
        Ok(NonNull::slice_from_raw_parts(chunk_ptr, size))
    }
}

impl fmt::Debug for LustChunkArena<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LustChunkArena")
            .field("chunk_size", &self.chunk_size)
            .field("chunk_count", &self.chunk_count())
            .field("parent_used_bytes", &self.parent.used_bytes())
            .finish()
    }
}

impl Drop for LustChunkArena<'_> {
    fn drop(&mut self) {
        self.cursor.set(core::ptr::null_mut());
        self.end.set(core::ptr::null_mut());
        for (chunk_ptr, chunk_layout) in self.chunks.get_mut().drain(..) {
            // SAFETY: Each `(chunk_ptr, chunk_layout)` was allocated via
            // `self.parent.allocate(chunk_layout)` with `chunk_layout.size() > 0`.
            unsafe { self.parent.deallocate(chunk_ptr, chunk_layout) };
        }
    }
}

// SAFETY:
// - ZST allocations return `layout.dangling_ptr()` with length 0 and never allocate a chunk.
// - Non-ZST allocations return disjoint, properly aligned sub-slices within chunks owned by
//   `self.chunks`, which remain live until `self` is dropped.
// - `LustChunkArena` is `!Clone` and `!Copy`; callers using `&LustChunkArena<'a>` cannot
//   outlive `self` due to Rust's borrow checker.
unsafe impl Allocator for LustChunkArena<'_> {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let size = layout.size();
        if size == 0 {
            return Ok(NonNull::slice_from_raw_parts(layout.dangling_ptr(), 0));
        }

        let cur = self.cursor.get();
        if !cur.is_null() {
            let cur_addr = cur as usize;
            let end_addr = self.end.get() as usize;
            let mask = layout.align() - 1;
            if let Some(aligned_addr) = cur_addr.checked_add(mask).map(|a| a & !mask) {
                if let Some(next_addr) = aligned_addr.checked_add(size) {
                    if next_addr <= end_addr {
                        let pad = aligned_addr - cur_addr;
                        // SAFETY: `cur` points within the current chunk and
                        // `pad + size <= end - cur`, preserving chunk provenance.
                        let aligned_ptr = unsafe { cur.add(pad) };
                        let next_ptr = unsafe { aligned_ptr.add(size) };
                        self.cursor.set(next_ptr);
                        // SAFETY: `cur` is derived from a `NonNull<u8>` chunk base.
                        let nn = unsafe { NonNull::new_unchecked(aligned_ptr) };
                        return Ok(NonNull::slice_from_raw_parts(nn, size));
                    }
                }
            }
        }

        self.allocate_from_new_chunk(layout)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let ptr = self.allocate(layout)?;
        let size = layout.size();
        if size != 0 {
            // SAFETY: `ptr` is uniquely owned and valid for `size` writable bytes.
            unsafe { ptr.cast::<u8>().as_ptr().write_bytes(0, size) };
        }
        Ok(ptr)
    }

    #[inline]
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        // Arena bulk-frees all chunks in Drop; individual deallocate is a no-op.
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let old_size = old_layout.size();
        let new_size = new_layout.size();
        if new_size < old_size {
            return Err(AllocError);
        }
        if new_size == 0 {
            return Ok(NonNull::slice_from_raw_parts(new_layout.dangling_ptr(), 0));
        }
        let new_slice = self.allocate(new_layout)?;
        if old_size != 0 {
            // SAFETY: `ptr` is valid for `old_size` bytes and disjoint from `new_slice`.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    ptr.as_ptr(),
                    new_slice.cast::<u8>().as_ptr(),
                    old_size,
                );
            }
        }
        Ok(new_slice)
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let old_size = old_layout.size();
        let new_size = new_layout.size();
        let Some(zero_len) = new_size.checked_sub(old_size) else {
            return Err(AllocError);
        };
        if new_size == 0 {
            return Ok(NonNull::slice_from_raw_parts(new_layout.dangling_ptr(), 0));
        }
        let new_slice = self.allocate(new_layout)?;
        let dst = new_slice.cast::<u8>().as_ptr();
        // SAFETY: `dst` is valid for `new_size` writable bytes and disjoint from `ptr`.
        unsafe {
            if old_size != 0 {
                core::ptr::copy_nonoverlapping(ptr.as_ptr(), dst, old_size);
            }
            dst.add(old_size).write_bytes(0, zero_len);
        }
        Ok(new_slice)
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let new_size = new_layout.size();
        if new_size > old_layout.size() {
            return Err(AllocError);
        }
        if new_size == 0 {
            return Ok(NonNull::slice_from_raw_parts(new_layout.dangling_ptr(), 0));
        }
        if (ptr.as_ptr() as usize) % new_layout.align() == 0 {
            return Ok(NonNull::slice_from_raw_parts(ptr, new_size));
        }
        let new_slice = self.allocate(new_layout)?;
        // SAFETY: `ptr` is valid for `old_layout.size() >= new_size` bytes and disjoint from `new_slice`.
        unsafe {
            core::ptr::copy_nonoverlapping(ptr.as_ptr(), new_slice.cast::<u8>().as_ptr(), new_size);
        }
        Ok(new_slice)
    }
}

/// Fixed-size single-threaded bump slab allocator backed by `UnsafeCell<[u8; N]>`.
///
/// Computes the base pointer directly via `self.buf.get() as *mut u8` without ever
/// creating `&mut [u8; N]`, preserving Stacked/Tree Borrows soundness across multiple
/// concurrent live allocations from the slab.
#[repr(align(16))]
pub struct StaticSlabAllocator<const N: usize> {
    buf: UnsafeCell<[u8; N]>,
    offset: Cell<usize>,
    _not_send_sync: PhantomData<*const ()>,
}

impl<const N: usize> StaticSlabAllocator<N> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            buf: UnsafeCell::new([0u8; N]),
            offset: Cell::new(0),
            _not_send_sync: PhantomData,
        }
    }

    #[inline(always)]
    pub fn used_bytes(&self) -> usize {
        self.offset.get()
    }

    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        N
    }

    #[inline(always)]
    pub fn remaining_bytes(&self) -> usize {
        N.saturating_sub(self.offset.get())
    }

    /// Resets the bump cursor to 0, reclaiming all slab space.
    ///
    /// # Safety
    /// Caller must ensure no live allocations handed out by this slab are accessed after reset.
    #[inline]
    pub unsafe fn reset(&self) {
        self.offset.set(0);
    }
}

impl<const N: usize> Default for StaticSlabAllocator<N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> fmt::Debug for StaticSlabAllocator<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StaticSlabAllocator")
            .field("capacity", &N)
            .field("used_bytes", &self.offset.get())
            .finish()
    }
}

// SAFETY:
// - ZST allocations return `layout.dangling_ptr()` with length 0 and never touch `self.buf`.
// - Base pointer is derived via `self.buf.get() as *mut u8` (raw `UnsafeCell` provenance),
//   NEVER creating `&mut [u8; N]`, so earlier allocations are never invalidated by later ones.
// - Each non-ZST allocation advances `self.offset` monotonically within `0..=N`, handing out
//   strictly disjoint, properly aligned byte ranges inside `self.buf`.
unsafe impl<const N: usize> Allocator for StaticSlabAllocator<N> {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let size = layout.size();
        if size == 0 {
            return Ok(NonNull::slice_from_raw_parts(layout.dangling_ptr(), 0));
        }

        // Compute base pointer directly from UnsafeCell without creating `&mut [u8; N]`.
        let base = self.buf.get() as *mut u8;
        let cur_offset = self.offset.get();
        let cur_addr = (base as usize).checked_add(cur_offset).ok_or(AllocError)?;
        let mask = layout.align() - 1;
        let aligned_addr = cur_addr.checked_add(mask).ok_or(AllocError)? & !mask;
        let pad = aligned_addr.checked_sub(cur_addr).ok_or(AllocError)?;
        let aligned_offset = cur_offset.checked_add(pad).ok_or(AllocError)?;
        let next_offset = aligned_offset.checked_add(size).ok_or(AllocError)?;
        if next_offset > N {
            return Err(AllocError);
        }

        self.offset.set(next_offset);
        // SAFETY: `aligned_offset + size <= N`, so `base.add(aligned_offset)` is in-bounds
        // of `self.buf`, aligned to `layout.align()`, non-null, and disjoint from all other
        // live allocations.
        let ptr = unsafe { base.add(aligned_offset) };
        let nn = NonNull::new(ptr).ok_or(AllocError)?;
        Ok(NonNull::slice_from_raw_parts(nn, size))
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let ptr = self.allocate(layout)?;
        let size = layout.size();
        if size != 0 {
            // SAFETY: `ptr` is uniquely owned and valid for `size` writable bytes.
            unsafe { ptr.cast::<u8>().as_ptr().write_bytes(0, size) };
        }
        Ok(ptr)
    }

    #[inline]
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        // Bump slab allocator: individual deallocate is a no-op.
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let old_size = old_layout.size();
        let new_size = new_layout.size();
        if new_size < old_size {
            return Err(AllocError);
        }
        if new_size == 0 {
            return Ok(NonNull::slice_from_raw_parts(new_layout.dangling_ptr(), 0));
        }
        let new_slice = self.allocate(new_layout)?;
        if old_size != 0 {
            // SAFETY: `ptr` is valid for `old_size` bytes and disjoint from `new_slice`.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    ptr.as_ptr(),
                    new_slice.cast::<u8>().as_ptr(),
                    old_size,
                );
            }
        }
        Ok(new_slice)
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let old_size = old_layout.size();
        let new_size = new_layout.size();
        let Some(zero_len) = new_size.checked_sub(old_size) else {
            return Err(AllocError);
        };
        if new_size == 0 {
            return Ok(NonNull::slice_from_raw_parts(new_layout.dangling_ptr(), 0));
        }
        let new_slice = self.allocate(new_layout)?;
        let dst = new_slice.cast::<u8>().as_ptr();
        // SAFETY: `dst` is valid for `new_size` writable bytes and disjoint from `ptr`.
        unsafe {
            if old_size != 0 {
                core::ptr::copy_nonoverlapping(ptr.as_ptr(), dst, old_size);
            }
            dst.add(old_size).write_bytes(0, zero_len);
        }
        Ok(new_slice)
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let new_size = new_layout.size();
        if new_size > old_layout.size() {
            return Err(AllocError);
        }
        if new_size == 0 {
            return Ok(NonNull::slice_from_raw_parts(new_layout.dangling_ptr(), 0));
        }
        if (ptr.as_ptr() as usize) % new_layout.align() == 0 {
            return Ok(NonNull::slice_from_raw_parts(ptr, new_size));
        }
        let new_slice = self.allocate(new_layout)?;
        // SAFETY: `ptr` is valid for `old_layout.size() >= new_size` bytes and disjoint from `new_slice`.
        unsafe {
            core::ptr::copy_nonoverlapping(ptr.as_ptr(), new_slice.cast::<u8>().as_ptr(), new_size);
        }
        Ok(new_slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc as alloc_crate;
    use alloc_crate::boxed::Box;

    #[test]
    fn test_static_slab_allocator_concurrent_and_reset() {
        let slab = StaticSlabAllocator::<2048>::new();
        {
            let mut a: Box<[u64; 8], &StaticSlabAllocator<2048>> = Box::new_in([1u64; 8], &slab);
            let mut b: Box<[u64; 8], &StaticSlabAllocator<2048>> = Box::new_in([2u64; 8], &slab);
            a[0] = 10;
            b[7] = 20;
            assert_eq!(a[0], 10);
            assert_eq!(b[7], 20);
            let huge = Layout::from_size_align(4096, 8).unwrap();
            assert!(matches!(slab.allocate(huge), Err(AllocError)));
        }
        unsafe { slab.reset() };
        assert_eq!(slab.used_bytes(), 0);
    }
}

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::bytecode::{Value, ValueKey};
use crate::vm::stdlib::{
    normalize_range, string_byte_values, string_find_values, string_gsub_fast, string_split_values,
    string_sub_slice,
};
use alloc::boxed::Box;
use hashbrown::hash_map::Entry;
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use core::sync::atomic::{AtomicI64, AtomicUsize};
use core::sync::atomic::Ordering;

/// Zero-cost single-threaded replacement for atomic integer fields in thread-confined `MirContext`.
/// Preserves `.load(Ordering)` / `.store(val, Ordering)` / `.fetch_add` / `.fetch_sub` API compatibility
/// without hardware atomic RMW (`lock xadd` / `ldadd`) instructions.
#[repr(transparent)]
#[derive(Debug, Default)]
pub struct LocalCell<T: Copy>(core::cell::Cell<T>);

impl<T: Copy> LocalCell<T> {
    pub const fn new(val: T) -> Self {
        Self(core::cell::Cell::new(val))
    }
    #[inline(always)]
    pub fn get(&self) -> T {
        self.0.get()
    }
    #[inline(always)]
    pub fn set(&self, val: T) {
        self.0.set(val);
    }
    #[inline(always)]
    pub fn load(&self, _order: core::sync::atomic::Ordering) -> T {
        self.0.get()
    }
    #[inline(always)]
    pub fn store(&self, val: T, _order: core::sync::atomic::Ordering) {
        self.0.set(val);
    }
}

impl LocalCell<i64> {
    #[inline(always)]
    pub fn fetch_add(&self, val: i64, _order: core::sync::atomic::Ordering) -> i64 {
        let old = self.0.get();
        self.0.set(old.wrapping_add(val));
        old
    }
    #[inline(always)]
    pub fn fetch_sub(&self, val: i64, _order: core::sync::atomic::Ordering) -> i64 {
        let old = self.0.get();
        self.0.set(old.wrapping_sub(val));
        old
    }
}

impl LocalCell<usize> {
    #[inline(always)]
    pub fn fetch_add(&self, val: usize, _order: core::sync::atomic::Ordering) -> usize {
        let old = self.0.get();
        self.0.set(old.wrapping_add(val));
        old
    }
    #[inline(always)]
    pub fn fetch_sub(&self, val: usize, _order: core::sync::atomic::Ordering) -> usize {
        let old = self.0.get();
        self.0.set(old.wrapping_sub(val));
        old
    }
}

/// Zero-cost single-threaded replacement for `std::sync::Mutex` fields in thread-confined `MirContext`.
/// Preserves `.lock().unwrap()` compatibility while also offering direct `.borrow()` / `.borrow_mut()`.
#[repr(transparent)]
#[derive(Debug, Default)]
pub struct LocalLock<T>(core::cell::RefCell<T>);

impl<T> LocalLock<T> {
    pub const fn new(val: T) -> Self {
        Self(core::cell::RefCell::new(val))
    }
    #[inline(always)]
    pub fn borrow(&self) -> core::cell::Ref<'_, T> {
        self.0.borrow()
    }
    #[inline(always)]
    pub fn borrow_mut(&self) -> core::cell::RefMut<'_, T> {
        self.0.borrow_mut()
    }
    #[inline(always)]
    pub fn lock(&self) -> Result<core::cell::RefMut<'_, T>, core::convert::Infallible> {
        Ok(self.0.borrow_mut())
    }
}

pub const MIR_STATUS_OK: i64 = crate::effect_abi::LUST_STATUS_SUCCESS as i64;
pub const MIR_STATUS_ERR_OUT_OF_GAS: i64 = crate::effect_abi::LUST_STATUS_ERR_OUT_OF_GAS as i64;
pub const MIR_STATUS_ERR_OOM: i64 = crate::effect_abi::LUST_STATUS_ERR_OUT_OF_MEMORY as i64;
pub const MIR_STATUS_ERR_DIV_BY_ZERO: i64 = 5;
pub const MIR_STATUS_ERR_NEG_EXPONENT: i64 = 6;
pub const MIR_STATUS_ERR_OUT_OF_BOUNDS: i64 = 7;

/// Chunk size replenished by `lust_mir_gas_slow()` during unbounded VM execution.
pub const UNBOUNDED_GAS_CHUNK: i64 = i64::MAX / 2;

/// Thread-confined inline gas fast-path state (`PERF-T4-1`, Hazard 7 item 1).
///
/// `Cell<i64>` / `Cell<u64>` are `#[repr(transparent)]` over `UnsafeCell` and NEVER
/// expose `&mut i64` or `&i64` references, making same-thread JIT raw-pointer loads/stores
/// and Rust `Cell::get`/`Cell::set` structurally free of Stacked-Borrows aliasing conflicts.
#[repr(C)]
#[derive(Debug)]
pub struct GasFastState {
    /// Single authoritative gas counter for this thread-confined `MirContext`.
    pub gas_remaining: core::cell::Cell<i64>,
    /// `0` = finite budget (OOG on `< 0`); `1` = unbounded VM execution (replenish on `< 0`).
    pub gas_unbounded: core::cell::Cell<u64>,
}

impl GasFastState {
    pub const fn new(gas: i64) -> Self {
        Self {
            gas_remaining: core::cell::Cell::new(gas),
            gas_unbounded: core::cell::Cell::new(0),
        }
    }
}

#[cfg(feature = "std")]
thread_local! {
    static UNBOUNDED_GAS_CHUNK_OVERRIDE: core::cell::Cell<Option<i64>> =
        const { core::cell::Cell::new(None) };
    static UNBOUNDED_GAS_REPLENISH_COUNT: core::cell::Cell<u64> =
        const { core::cell::Cell::new(0) };
}

/// Returns the active unbounded replenishment chunk size for the current thread.
#[cfg(feature = "std")]
#[inline]
pub fn unbounded_gas_chunk() -> i64 {
    UNBOUNDED_GAS_CHUNK_OVERRIDE
        .with(|c| c.get())
        .unwrap_or(UNBOUNDED_GAS_CHUNK)
}

#[cfg(not(feature = "std"))]
#[inline]
pub const fn unbounded_gas_chunk() -> i64 {
    UNBOUNDED_GAS_CHUNK
}

#[cfg(feature = "std")]
fn restore_unbounded_gas_chunk(prev: Option<i64>) {
    UNBOUNDED_GAS_CHUNK_OVERRIDE.with(|c| c.set(prev));
}

/// RAII guard restoring the previous thread-local unbounded gas chunk override on drop.
#[cfg(feature = "std")]
pub struct UnboundedGasChunkGuard {
    _guard: core::mem::DropGuard<Option<i64>, fn(Option<i64>)>,
}

/// Sets a thread-local override for the unbounded gas replenishment chunk size (clamped to `>= 1`).
/// Used by integration tests to deterministically exercise `lust_mir_gas_slow()` replenishment.
#[cfg(feature = "std")]
pub fn set_unbounded_gas_chunk_override(chunk: Option<i64>) -> UnboundedGasChunkGuard {
    let normalized = chunk.map(|v| v.max(1));
    let prev = UNBOUNDED_GAS_CHUNK_OVERRIDE.with(|c| {
        let old = c.get();
        c.set(normalized);
        old
    });
    UnboundedGasChunkGuard {
        _guard: core::mem::DropGuard::new(prev, restore_unbounded_gas_chunk),
    }
}

/// Returns the number of unbounded gas replenishments performed on the current thread.
#[cfg(feature = "std")]
pub fn unbounded_gas_replenish_count() -> u64 {
    UNBOUNDED_GAS_REPLENISH_COUNT.with(|c| c.get())
}

/// Resets the thread-local unbounded gas replenishment counter to `0`.
#[cfg(feature = "std")]
pub fn reset_unbounded_gas_replenish_count() {
    UNBOUNDED_GAS_REPLENISH_COUNT.with(|c| c.set(0));
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MirArrayHeader {
    pub len: i64,
    pub cap: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MirAllocationKind {
    Array(crate::mir::ir::MirType),
    ArrayBacking,
    Tuple(Vec<crate::mir::ir::MirType>),
    Struct(alloc::rc::Rc<str>),
    CFlatStruct(alloc::string::String),
    RawWords,
}

#[cfg(feature = "std")]
const MAX_INTERNED_STRUCT_TYPE_NAMES: usize = 4096;

#[cfg(feature = "std")]
const SMALL_BUF_POOL_MAX_WORDS: usize = 6;
#[cfg(feature = "std")]
const SMALL_BUF_POOL_BUCKET_CAP: usize = 512;

/// Reserved negative handle range for static 1-byte ASCII strings (`0x00..=0x7F`) and `""`.
/// Handles `ASCII_HANDLE_BASE - (b as i64)` (`-0x4000_0000` down to `-0x4000_007F`) and
/// `EMPTY_STR_HANDLE` (`-0x4000_0080`) never collide with `interned_list` handles (`-1, -2, ...`)
/// in either scoped or default `MirContext`, require zero heap allocation, and work across contexts.
pub(crate) const ASCII_HANDLE_BASE: i64 = -(1i64 << 30);
pub(crate) const EMPTY_STR_HANDLE: i64 = ASCII_HANDLE_BASE - 128;

/// Reserved negative handle range for per-`MirContext` bounded dynamic short-string interning
/// (`-0x2000_0000` down to `-0x2000_0FFF`). Cleared on top-level `clear_to_baselines(0, 0)`.
pub(crate) const DYN_INTERN_BASE: i64 = -(1i64 << 29);
pub(crate) const DYN_INTERN_CAP: usize = 4096;
pub(crate) const DYN_INTERN_MAX_BYTES: usize = 64;

#[cfg(feature = "std")]
const BUMP_SLAB_WORDS: usize = 8192;
#[cfg(feature = "std")]
const BUMP_MAX_WORDS: usize = 64;

#[cfg(feature = "std")]
#[derive(Debug, Clone)]
struct BumpAllocEntry {
    word_off: u16,
    rec: MirAllocationRecord,
}

#[cfg(feature = "std")]
#[derive(Debug)]
struct BumpSlabMeta {
    base_addr: usize,
    end_addr: usize,
    entries: Vec<BumpAllocEntry>,
}

#[inline(always)]
pub(crate) fn is_dyn_intern_handle(handle: i64) -> bool {
    handle <= DYN_INTERN_BASE && handle > DYN_INTERN_BASE - (DYN_INTERN_CAP as i64)
}

const ASCII_BYTES: [u8; 128] = {
    let mut arr = [0u8; 128];
    let mut i = 0;
    while i < 128 {
        arr[i] = i as u8;
        i += 1;
    }
    arr
};

#[inline(always)]
pub fn ascii_byte_handle(b: u8) -> i64 {
    debug_assert!(b < 128);
    ASCII_HANDLE_BASE - (b as i64)
}

#[inline(always)]
pub(crate) fn is_ascii_handle(handle: i64) -> bool {
    handle <= ASCII_HANDLE_BASE && handle > ASCII_HANDLE_BASE - 128
}

#[inline(always)]
pub(crate) fn is_static_str_handle(handle: i64) -> bool {
    handle == EMPTY_STR_HANDLE || is_ascii_handle(handle)
}

#[inline(always)]
pub(crate) fn static_str_from_handle(handle: i64) -> Option<&'static str> {
    if handle == EMPTY_STR_HANDLE {
        return Some("");
    }
    if is_ascii_handle(handle) {
        let b = (ASCII_HANDLE_BASE - handle) as usize;
        return core::str::from_utf8(&ASCII_BYTES[b..b + 1]).ok();
    }
    None
}

#[cfg(feature = "std")]
#[derive(Debug)]
pub struct MirIntQueue {
    pub cap: usize,
    pub items: alloc::collections::VecDeque<i64>,
}

#[derive(Debug, Clone)]
pub struct MirAllocationRecord {
    pub base_ptr: *mut i64,
    pub word_capacity: usize,
    pub kind: MirAllocationKind,
    pub buffer_index: usize,
    pub handle_epoch: usize,
    pub alloc_seq: usize,
    pub is_ctx_owned: bool,
    pub read_only_borrowed: bool,
    pub registered_scope_depth: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MirContextBaseline {
    pub handles: usize,
    pub buffers: usize,
}

#[cfg_attr(feature = "std", repr(C))]
#[derive(Debug)]
pub struct MirContext {
    #[cfg(feature = "std")]
    pub(crate) gas_fast: GasFastState,
    #[cfg(feature = "std")]
    pub last_status: LocalCell<i64>,
    #[cfg(feature = "std")]
    pub(crate) gas_slow_result: core::cell::Cell<i64>,
    #[cfg(feature = "std")]
    nested_vm_gas_charged: core::cell::Cell<u64>,
    #[cfg(not(feature = "std"))]
    gas_remaining: AtomicI64,
    #[cfg(feature = "std")]
    pub memory_used: LocalCell<i64>,
    #[cfg(not(feature = "std"))]
    pub memory_used: AtomicI64,
    #[cfg(not(feature = "std"))]
    pub last_status: AtomicI64,
    #[cfg(feature = "std")]
    pub max_memory_limit: LocalCell<i64>,
    #[cfg(not(feature = "std"))]
    pub max_memory_limit: AtomicI64,
    #[cfg(feature = "std")]
    pub next_alloc_seq: LocalCell<usize>,
    #[cfg(not(feature = "std"))]
    pub next_alloc_seq: AtomicUsize,
    #[cfg(feature = "std")]
    pub heap_values: LocalLock<Vec<Option<crate::bytecode::Value>>>,
    #[cfg(feature = "std")]
    pub allocated_buffers: LocalLock<Vec<(*mut i64, usize)>>,
    #[cfg(feature = "std")]
    pub allocation_records: LocalLock<hashbrown::HashMap<usize, MirAllocationRecord>>,
    #[cfg(feature = "std")]
    pub scope_baselines: LocalLock<Vec<MirContextBaseline>>,
    #[cfg(feature = "std")]
    pub scope_alloc_floors: LocalLock<Vec<usize>>,
    /// Scope-bound undo log for borrowed read-only flat array provenance (`PERF-T4-3`).
    /// Never frees the underlying `FlatArrayBuffer` (owned by `Rc<FlatArrayBuffer>`).
    #[cfg(feature = "std")]
    pub borrowed_readonly_stack: LocalLock<Vec<(usize, Option<MirAllocationRecord>)>>,
    #[cfg(feature = "std")]
    pub borrowed_scope_floors: LocalLock<Vec<usize>>,
    #[cfg(feature = "std")]
    pub interned_strings: LocalLock<hashbrown::HashMap<alloc::string::String, i64>>,
    #[cfg(feature = "std")]
    pub interned_list: LocalLock<Vec<alloc::string::String>>,
    #[cfg(feature = "std")]
    interned_rc_list: LocalLock<Vec<alloc::rc::Rc<alloc::string::String>>>,
    #[cfg(feature = "std")]
    pub(crate) dyn_interned_map: LocalLock<hashbrown::HashMap<alloc::string::String, i64>>,
    #[cfg(feature = "std")]
    pub(crate) dyn_interned_list: LocalLock<Vec<alloc::rc::Rc<alloc::string::String>>>,
    #[cfg(feature = "std")]
    pub small_digit_handles: [LocalCell<i64>; SMALL_INT_STRING_COUNT],
    #[cfg(feature = "std")]
    pub struct_fields: LocalLock<hashbrown::HashMap<alloc::string::String, Vec<crate::mir::ir::MirType>>>,
    #[cfg(feature = "std")]
    struct_type_names: LocalLock<hashbrown::HashSet<alloc::rc::Rc<str>>>,
    #[cfg(feature = "std")]
    last_struct_type_name: core::cell::RefCell<Option<alloc::rc::Rc<str>>>,
    #[cfg(feature = "std")]
    empty_struct_type_name: alloc::rc::Rc<str>,
    #[cfg(feature = "std")]
    small_buf_pool: LocalLock<[Vec<Box<[i64]>>; SMALL_BUF_POOL_MAX_WORDS + 1]>,
    #[cfg(feature = "std")]
    bump_slabs: LocalLock<Vec<Box<[core::cell::Cell<i64>]>>>,
    #[cfg(feature = "std")]
    bump_slab_meta: LocalLock<Vec<BumpSlabMeta>>,
    #[cfg(feature = "std")]
    bump_cursor: core::cell::Cell<(usize, usize)>,
    #[cfg(feature = "std")]
    native_queues: LocalLock<Vec<MirIntQueue>>,
    #[cfg(feature = "std")]
    dirty_allocations: core::cell::Cell<bool>,
    #[cfg(feature = "std")]
    dirty_allocations_count: core::cell::Cell<u64>,
    #[cfg(feature = "std")]
    push_grow_slow_calls: core::cell::Cell<u64>,
    /// Compiler-enforced thread confinement under `std`: makes both `MirContext`
    /// and `Arc<MirContext>` statically `!Send + !Sync` (`PERF-T4-1`, Hazard 7 item 1).
    #[cfg(feature = "std")]
    _not_send_sync: core::marker::PhantomData<*const ()>,
}

#[cfg(feature = "std")]
const _: () = {
    assert!(core::mem::offset_of!(MirContext, gas_fast) == 0);
    assert!(core::mem::offset_of!(MirContext, last_status) == 16);
    assert!(core::mem::offset_of!(MirContext, gas_slow_result) == 24);
};

impl Default for MirContext {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MirContext {
    fn drop(&mut self) {
        #[cfg(feature = "std")]
        self.clear_to_baselines(0, 0);
    }
}

impl MirContext {
    #[cfg(feature = "std")]
    pub fn new() -> Self {
        Self {
            gas_fast: GasFastState::new(1_000_000_000),
            last_status: LocalCell::new(0),
            gas_slow_result: core::cell::Cell::new(0),
            nested_vm_gas_charged: core::cell::Cell::new(0),
            memory_used: LocalCell::new(0),
            max_memory_limit: LocalCell::new(i64::MAX),
            next_alloc_seq: LocalCell::new(1),
            heap_values: LocalLock::new(Vec::new()),
            allocated_buffers: LocalLock::new(Vec::new()),
            allocation_records: LocalLock::new(hashbrown::HashMap::new()),
            scope_baselines: LocalLock::new(Vec::new()),
            scope_alloc_floors: LocalLock::new(Vec::new()),
            borrowed_readonly_stack: LocalLock::new(Vec::new()),
            borrowed_scope_floors: LocalLock::new(Vec::new()),
            interned_strings: LocalLock::new(hashbrown::HashMap::new()),
            interned_list: LocalLock::new(Vec::new()),
            interned_rc_list: LocalLock::new(Vec::new()),
            dyn_interned_map: LocalLock::new(hashbrown::HashMap::new()),
            dyn_interned_list: LocalLock::new(Vec::new()),
            small_digit_handles: [const { LocalCell::new(0) }; SMALL_INT_STRING_COUNT],
            struct_fields: LocalLock::new(hashbrown::HashMap::new()),
            struct_type_names: LocalLock::new(hashbrown::HashSet::new()),
            last_struct_type_name: core::cell::RefCell::new(None),
            empty_struct_type_name: alloc::rc::Rc::from(""),
            small_buf_pool: LocalLock::new([const { Vec::new() }; SMALL_BUF_POOL_MAX_WORDS + 1]),
            bump_slabs: LocalLock::new(Vec::new()),
            bump_slab_meta: LocalLock::new(Vec::new()),
            bump_cursor: core::cell::Cell::new((0, 0)),
            native_queues: LocalLock::new(Vec::new()),
            dirty_allocations: core::cell::Cell::new(false),
            dirty_allocations_count: core::cell::Cell::new(0),
            push_grow_slow_calls: core::cell::Cell::new(0),
            _not_send_sync: core::marker::PhantomData,
        }
    }

    #[cfg(not(feature = "std"))]
    pub const fn new() -> Self {
        Self {
            gas_remaining: AtomicI64::new(1_000_000_000),
            memory_used: AtomicI64::new(0),
            last_status: AtomicI64::new(0),
            max_memory_limit: AtomicI64::new(i64::MAX),
            next_alloc_seq: AtomicUsize::new(1),
        }
    }

    #[inline]
    pub fn dirty_allocations(&self) -> bool {
        #[cfg(feature = "std")]
        {
            self.dirty_allocations.get()
        }
        #[cfg(not(feature = "std"))]
        {
            false
        }
    }

    #[inline]
    pub fn dirty_allocations_count(&self) -> u64 {
        #[cfg(feature = "std")]
        {
            self.dirty_allocations_count.get()
        }
        #[cfg(not(feature = "std"))]
        {
            0
        }
    }

    #[inline]
    pub fn push_grow_slow_calls(&self) -> u64 {
        #[cfg(feature = "std")]
        {
            self.push_grow_slow_calls.get()
        }
        #[cfg(not(feature = "std"))]
        {
            0
        }
    }

    #[inline]
    pub fn reset_push_grow_slow_calls(&self) {
        #[cfg(feature = "std")]
        {
            self.push_grow_slow_calls.set(0);
        }
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub(crate) fn mark_dirty_allocation(&self) {
        self.dirty_allocations.set(true);
        self.dirty_allocations_count
            .set(self.dirty_allocations_count.get().saturating_add(1));
    }

    pub fn with_gas(gas: i64) -> Self {
        let ctx = Self::new();
        ctx.set_gas_unbounded(false);
        ctx.set_gas(gas);
        ctx
    }

    pub fn enter(&self) -> MirContextGuard {
        enter_mir_context(self)
    }

    #[inline]
    pub fn gas_remaining(&self) -> i64 {
        #[cfg(feature = "std")]
        {
            self.gas_fast.gas_remaining.get()
        }
        #[cfg(not(feature = "std"))]
        {
            self.gas_remaining.load(Ordering::SeqCst)
        }
    }

    #[inline]
    pub fn gas_unbounded(&self) -> bool {
        #[cfg(feature = "std")]
        {
            self.gas_fast.gas_unbounded.get() != 0
        }
        #[cfg(not(feature = "std"))]
        {
            false
        }
    }

    pub fn memory_used(&self) -> i64 {
        self.memory_used.load(Ordering::Relaxed)
    }

    pub fn last_status(&self) -> i64 {
        self.last_status.load(Ordering::Relaxed)
    }

    pub fn max_memory_limit(&self) -> i64 {
        self.max_memory_limit.load(Ordering::Relaxed)
    }

    pub fn set_status(&self, status: i64) {
        self.last_status.store(status, Ordering::Relaxed);
    }

    #[inline]
    pub fn set_gas(&self, gas: i64) {
        #[cfg(feature = "std")]
        {
            self.gas_fast.gas_remaining.set(gas);
        }
        #[cfg(not(feature = "std"))]
        {
            self.gas_remaining.store(gas, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn set_gas_unbounded(&self, unbounded: bool) {
        #[cfg(feature = "std")]
        {
            self.gas_fast.gas_unbounded.set(u64::from(unbounded));
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = unbounded;
        }
    }

    /// Overflow-safe gas debit matching the inline Cranelift fast/slow-path contract.
    #[inline]
    pub fn charge_gas(&self, amt: i64) -> i64 {
        #[cfg(feature = "std")]
        {
            let cur = self.gas_fast.gas_remaining.get();
            let (rem, overflowed) = cur.overflowing_sub(amt);
            let rem_st = if overflowed { i64::MIN } else { rem };
            self.gas_fast.gas_remaining.set(rem_st);
            if overflowed || rem_st < 0 {
                if self.gas_fast.gas_unbounded.get() != 0 {
                    let chunk = unbounded_gas_chunk();
                    self.gas_fast.gas_remaining.set(chunk);
                    UNBOUNDED_GAS_REPLENISH_COUNT.with(|c| c.set(c.get().saturating_add(1)));
                    return chunk;
                }
                self.last_status
                    .store(MIR_STATUS_ERR_OUT_OF_GAS, Ordering::Relaxed);
            }
            rem_st
        }
        #[cfg(not(feature = "std"))]
        {
            loop {
                let cur = self.gas_remaining.load(Ordering::Relaxed);
                let (rem, overflowed) = cur.overflowing_sub(amt);
                let rem_st = if overflowed { i64::MIN } else { rem };
                if self
                    .gas_remaining
                    .compare_exchange(cur, rem_st, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
                {
                    if overflowed || rem_st < 0 {
                        self.last_status
                            .store(MIR_STATUS_ERR_OUT_OF_GAS, Ordering::Relaxed);
                    }
                    return rem_st;
                }
            }
        }
    }

    #[cfg(feature = "std")]
    #[inline]
    pub(crate) fn gas_fast_ptr(&self) -> *mut GasFastState {
        // SAFETY: `MirContext` is `#[repr(C)]` with `gas_fast` at offset 0 and
        // `last_status` at offset 16. Both are `Cell`-backed (`UnsafeCell`) and
        // Rust never forms `&mut` references to those fields, keeping JIT loads/stores
        // structurally free of Stacked-Borrows aliasing conflicts.
        core::ptr::from_ref(self) as *mut GasFastState
    }

    #[cfg(feature = "std")]
    #[inline]
    pub(crate) fn nested_vm_gas_charged(&self) -> u64 {
        self.nested_vm_gas_charged.get()
    }

    #[cfg(feature = "std")]
    #[inline]
    pub(crate) fn add_nested_vm_gas_charged(&self, delta: u64) {
        self.nested_vm_gas_charged
            .set(self.nested_vm_gas_charged.get().saturating_add(delta));
    }

    pub fn set_memory_used(&self, bytes: i64) {
        self.memory_used.store(bytes, Ordering::Relaxed);
    }

    pub fn set_max_memory_limit(&self, bytes: i64) {
        self.max_memory_limit.store(bytes, Ordering::Relaxed);
    }

    pub fn handle_count(&self) -> usize {
        #[cfg(feature = "std")]
        {
            self.heap_values.borrow().len()
        }
        #[cfg(not(feature = "std"))]
        {
            0
        }
    }

    pub fn register_handle(&self, val: crate::bytecode::Value) -> i64 {
        #[cfg(feature = "std")]
        {
            let handle = {
                let mut hv = self.heap_values.borrow_mut();
                hv.push(Some(val));
                hv.len() as i64
            };
            self.mark_dirty_allocation();
            handle
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = val;
            0
        }
    }

    #[inline]
    pub fn intern_or_register_str(&self, s: &str) -> i64 {
        if s.is_empty() {
            return EMPTY_STR_HANDLE;
        }
        let bytes = s.as_bytes();
        if bytes.len() == 1 && bytes[0] < 128 {
            return ascii_byte_handle(bytes[0]);
        }
        self.register_handle(crate::bytecode::Value::string(s))
    }

    #[inline]
    pub fn intern_or_register_string(&self, s: alloc::string::String) -> i64 {
        if s.is_empty() {
            return EMPTY_STR_HANDLE;
        }
        let bytes = s.as_bytes();
        if bytes.len() == 1 && bytes[0] < 128 {
            return ascii_byte_handle(bytes[0]);
        }
        self.register_handle(crate::bytecode::Value::string(s))
    }

    /// Interns short strings (`<= DYN_INTERN_MAX_BYTES`) into the per-context bounded dynamic
    /// intern table (or reuses an existing compile-time interned literal handle in `self`),
    /// eliminating repeated `String` + `Rc` heap allocations in hot split/sub pipelines.
    #[inline]
    pub fn intern_short_str_or_register(&self, s: &str) -> i64 {
        if s.is_empty() {
            return EMPTY_STR_HANDLE;
        }
        let bytes = s.as_bytes();
        if bytes.len() == 1 && bytes[0] < 128 {
            return ascii_byte_handle(bytes[0]);
        }
        #[cfg(feature = "std")]
        {
            if bytes.len() <= DYN_INTERN_MAX_BYTES {
                if let Some(&h) = self.interned_strings.borrow().get(s) {
                    return h;
                }
                if let Some(&h) = self.dyn_interned_map.borrow().get(s) {
                    return h;
                }
                let mut list = self.dyn_interned_list.borrow_mut();
                if list.len() < DYN_INTERN_CAP {
                    let owned = alloc::string::ToString::to_string(s);
                    let idx = list.len();
                    let h = DYN_INTERN_BASE - (idx as i64);
                    list.push(alloc::rc::Rc::new(owned.clone()));
                    self.dyn_interned_map.borrow_mut().insert(owned, h);
                    self.mark_dirty_allocation();
                    return h;
                }
            }
        }
        self.register_handle(crate::bytecode::Value::string(s))
    }

    #[inline]
    pub fn tostring_small_int_handle(&self, v: i64) -> Option<i64> {
        #[cfg(feature = "std")]
        {
            if (0..=9).contains(&v) {
                return Some(ascii_byte_handle(b'0' + (v as u8)));
            }
            if (10..SMALL_INT_STRING_COUNT as i64).contains(&v) {
                let cached = &self.small_digit_handles[v as usize];
                let existing = cached.get();
                if existing != 0 {
                    return Some(existing);
                }
                let h = self.intern_string(format_i64_stack(v, &mut [0; 24]));
                cached.set(h);
                return Some(h);
            }
            None
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = v;
            None
        }
    }

    /// Borrows the `&str` backing `handle` directly in place with zero `String::clone`
    /// and zero `Rc::new` allocations. Callers must NOT mutate `heap_values` or
    /// `interned_list` while inside `f`.
    #[inline]
    pub fn with_str_handle<R>(&self, handle: i64, f: impl FnOnce(&str) -> R) -> Option<R> {
        #[cfg(feature = "std")]
        {
            if handle < 0 {
                if let Some(s) = static_str_from_handle(handle) {
                    return Some(f(s));
                }
                if is_dyn_intern_handle(handle) {
                    let idx = (DYN_INTERN_BASE - handle) as usize;
                    let list = self.dyn_interned_list.borrow();
                    return list.get(idx).map(|rc| f(rc.as_str()));
                }
                let idx = usize::try_from(handle.unsigned_abs() - 1).ok()?;
                {
                    let list = self.interned_list.borrow();
                    if let Some(s) = list.get(idx) {
                        return Some(f(s.as_str()));
                    }
                }
                return DEFAULT_MIR_CONTEXT.with(|d| {
                    let d_list = d.interned_list.borrow();
                    d_list.get(idx).map(|s| f(s.as_str()))
                });
            }
            if handle <= 0 {
                return None;
            }
            let idx = (handle - 1) as usize;
            {
                let hv = self.heap_values.borrow();
                if let Some(slot) = hv.get(idx) {
                    return match slot {
                        Some(crate::bytecode::Value::String(s)) => Some(f(s.as_str())),
                        Some(_) => None,
                        None => {
                            drop(hv);
                            DEFAULT_MIR_CONTEXT.with(|d| {
                                let d_hv = d.heap_values.borrow();
                                match d_hv.get(idx) {
                                    Some(Some(crate::bytecode::Value::String(s))) => Some(f(s.as_str())),
                                    _ => None,
                                }
                            })
                        }
                    };
                }
            }
            DEFAULT_MIR_CONTEXT.with(|d| {
                let d_hv = d.heap_values.borrow();
                match d_hv.get(idx) {
                    Some(Some(crate::bytecode::Value::String(s))) => Some(f(s.as_str())),
                    _ => None,
                }
            })
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (handle, f);
            None
        }
    }

    /// Borrows two `&str` handles simultaneously with zero clones or allocations.
    #[inline]
    pub fn with_str_pair<R>(&self, a: i64, b: i64, f: impl FnOnce(&str, &str) -> R) -> Option<R> {
        self.with_str_handle(a, |sa| self.with_str_handle(b, |sb| f(sa, sb)))
            .flatten()
    }

    pub fn get_handle(&self, handle: i64) -> Option<crate::bytecode::Value> {
        #[cfg(feature = "std")]
        {
            if handle < 0 {
                if let Some(s) = static_str_from_handle(handle) {
                    return Some(crate::bytecode::Value::string(s));
                }
                if is_dyn_intern_handle(handle) {
                    let idx = (DYN_INTERN_BASE - handle) as usize;
                    let list = self.dyn_interned_list.borrow();
                    return list
                        .get(idx)
                        .map(|rc| crate::bytecode::Value::String(alloc::rc::Rc::clone(rc)));
                }
                let idx = usize::try_from(handle.unsigned_abs() - 1).ok()?;
                {
                    let rc_list = self.interned_rc_list.borrow();
                    if let Some(rc) = rc_list.get(idx) {
                        return Some(crate::bytecode::Value::String(alloc::rc::Rc::clone(rc)));
                    }
                }
                {
                    let list = self.interned_list.borrow();
                    if let Some(s) = list.get(idx) {
                        return Some(crate::bytecode::Value::string(s.clone()));
                    }
                }
                return DEFAULT_MIR_CONTEXT.with(|d| {
                    let d_rc = d.interned_rc_list.borrow();
                    if let Some(rc) = d_rc.get(idx) {
                        return Some(crate::bytecode::Value::String(alloc::rc::Rc::clone(rc)));
                    }
                    drop(d_rc);
                    let d_list = d.interned_list.borrow();
                    d_list.get(idx).map(|s| crate::bytecode::Value::string(s.clone()))
                });
            }
            if handle <= 0 {
                return None;
            }
            let hv = self.heap_values.borrow();
            let idx = (handle - 1) as usize;
            if let Some(v) = hv.get(idx).and_then(|opt| opt.clone()) {
                return Some(v);
            }
            drop(hv);
            DEFAULT_MIR_CONTEXT.with(|d| {
                let d_hv = d.heap_values.borrow();
                d_hv.get(idx).and_then(|opt| opt.clone())
            })
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = handle;
            None
        }
    }

    pub fn intern_string(&self, s: &str) -> i64 {
        #[cfg(feature = "std")]
        {
            if s.is_empty() {
                return EMPTY_STR_HANDLE;
            }
            let bytes = s.as_bytes();
            if bytes.len() == 1 && bytes[0] < 128 {
                return ascii_byte_handle(bytes[0]);
            }
            let mut map = self.interned_strings.borrow_mut();
            if let Some(&h) = map.get(s) {
                return h;
            }
            let owned = alloc::string::ToString::to_string(s);
            let mut list = self.interned_list.borrow_mut();
            let mut rc_list = self.interned_rc_list.borrow_mut();
            list.push(owned.clone());
            rc_list.push(alloc::rc::Rc::new(owned.clone()));
            let h = -(list.len() as i64);
            map.insert(owned, h);
            h
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = s;
            0
        }
    }

    pub fn register_struct_field_types(&self, name: &str, field_types: Vec<crate::mir::ir::MirType>) {
        #[cfg(feature = "std")]
        {
            self.intern_struct_type_name(name);
            let mut sf = self.struct_fields.borrow_mut();
            sf.insert(alloc::string::ToString::to_string(name), field_types);
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (name, field_types);
        }
    }

    /// Interns a struct type name into a shared `Rc<str>` within this thread-confined `MirContext`.
    /// Uses a content-verified 1-entry MRU cache backed by a bounded `HashSet<Rc<str>>` so hot
    /// struct allocation loops avoid per-allocation heap `String` allocations and hashing.
    #[inline]
    pub fn intern_struct_type_name(&self, name: &str) -> alloc::rc::Rc<str> {
        #[cfg(feature = "std")]
        {
            if name.is_empty() {
                return alloc::rc::Rc::clone(&self.empty_struct_type_name);
            }
            {
                let last = self.last_struct_type_name.borrow();
                if let Some(cached) = last.as_ref()
                    && cached.as_ref() == name
                {
                    return alloc::rc::Rc::clone(cached);
                }
            }
            let rc = {
                let mut set = self.struct_type_names.borrow_mut();
                if let Some(existing) = set.get(name) {
                    alloc::rc::Rc::clone(existing)
                } else {
                    let fresh: alloc::rc::Rc<str> = alloc::rc::Rc::from(name);
                    if set.len() < MAX_INTERNED_STRUCT_TYPE_NAMES {
                        set.insert(alloc::rc::Rc::clone(&fresh));
                    }
                    fresh
                }
            };
            *self.last_struct_type_name.borrow_mut() = Some(alloc::rc::Rc::clone(&rc));
            rc
        }
        #[cfg(not(feature = "std"))]
        {
            alloc::rc::Rc::from(name)
        }
    }

    /// Fast byte-slice interner for `lust_mir_alloc_struct_named`.
    /// Because the MRU cached `Rc<str>` is guaranteed valid UTF-8, a byte-slice equality match
    /// proves validity and returns a shared `Rc<str>` without UTF-8 rescanning or hashing.
    #[inline]
    fn intern_struct_type_name_bytes(&self, bytes: &[u8]) -> Option<alloc::rc::Rc<str>> {
        #[cfg(feature = "std")]
        {
            {
                let last = self.last_struct_type_name.borrow();
                if let Some(cached) = last.as_ref()
                    && cached.as_bytes() == bytes
                {
                    return Some(alloc::rc::Rc::clone(cached));
                }
            }
            let name_str = core::str::from_utf8(bytes).ok()?;
            Some(self.intern_struct_type_name(name_str))
        }
        #[cfg(not(feature = "std"))]
        {
            let name_str = core::str::from_utf8(bytes).ok()?;
            Some(alloc::rc::Rc::from(name_str))
        }
    }

    #[cfg(feature = "std")]
    #[inline]
    fn take_or_alloc_buffer(&self, size: usize) -> Option<Box<[i64]>> {
        if size <= SMALL_BUF_POOL_MAX_WORDS {
            let mut pool = self.small_buf_pool.borrow_mut();
            if let Some(mut recycled) = pool[size].pop() {
                recycled.fill(0);
                return Some(recycled);
            }
        }
        let mut vec = Vec::<i64>::new();
        if vec.try_reserve_exact(size).is_err() {
            return None;
        }
        vec.resize(size, 0);
        Some(vec.into_boxed_slice())
    }

    #[cfg(feature = "std")]
    #[inline]
    fn try_bump_alloc(
        &self,
        size: usize,
        header0: i64,
        header1: i64,
        payload: Option<&[i64]>,
        kind: MirAllocationKind,
    ) -> Result<*mut i64, MirAllocationKind> {
        if size == 0 || size > BUMP_MAX_WORDS {
            return Err(kind);
        }
        let mut slabs = self.bump_slabs.borrow_mut();
        let mut metas = self.bump_slab_meta.borrow_mut();
        let (mut slab_idx, mut word_off) = self.bump_cursor.get();
        if slabs.is_empty() || word_off + size > BUMP_SLAB_WORDS {
            if !slabs.is_empty() && slab_idx + 1 < slabs.len() {
                slab_idx += 1;
                word_off = 0;
            } else {
                let mut v = Vec::<core::cell::Cell<i64>>::new();
                if v.try_reserve_exact(BUMP_SLAB_WORDS).is_err() {
                    return Err(kind);
                }
                v.resize_with(BUMP_SLAB_WORDS, || core::cell::Cell::new(0));
                let slab = v.into_boxed_slice();
                let base_addr = slab[0].as_ptr() as usize;
                let end_addr = base_addr + BUMP_SLAB_WORDS * core::mem::size_of::<i64>();
                slabs.push(slab);
                metas.push(BumpSlabMeta {
                    base_addr,
                    end_addr,
                    entries: Vec::with_capacity(1024),
                });
                slab_idx = slabs.len() - 1;
                word_off = 0;
            }
        }
        let mut bufs = self.allocated_buffers.borrow_mut();
        if bufs.try_reserve(1).is_err() || metas[slab_idx].entries.try_reserve(1).is_err() {
            return Err(kind);
        }
        let slab = &slabs[slab_idx];
        slab[word_off].set(header0);
        if size > 1 {
            slab[word_off + 1].set(header1);
        }
        match payload {
            Some(elems) => {
                for (i, &w) in elems.iter().enumerate() {
                    slab[word_off + 2 + i].set(w);
                }
                for i in elems.len()..(size - 2) {
                    slab[word_off + 2 + i].set(0);
                }
            }
            None => {
                for i in 2..size {
                    slab[word_off + i].set(0);
                }
            }
        }
        let ptr = slab[word_off].as_ptr();
        let handle_epoch = self.handle_count();
        let scope_depth = self.scope_alloc_floors.borrow().len();
        let seq = self.next_alloc_seq.fetch_add(1, Ordering::Relaxed);
        let buffer_index = bufs.len();
        bufs.push((ptr, 0));
        metas[slab_idx].entries.push(BumpAllocEntry {
            word_off: word_off as u16,
            rec: MirAllocationRecord {
                base_ptr: ptr,
                word_capacity: size,
                kind,
                buffer_index,
                handle_epoch,
                alloc_seq: seq,
                is_ctx_owned: true,
                read_only_borrowed: false,
                registered_scope_depth: scope_depth,
            },
        });
        self.bump_cursor.set((slab_idx, word_off + size));
        drop(bufs);
        drop(metas);
        drop(slabs);
        self.mark_dirty_allocation();
        Ok(ptr)
    }

    #[cfg(feature = "std")]
    #[inline]
    fn lookup_bump_record(&self, ptr_usize: usize) -> Option<MirAllocationRecord> {
        let metas = self.bump_slab_meta.borrow();
        for meta in metas.iter().rev() {
            if ptr_usize >= meta.base_addr && ptr_usize < meta.end_addr {
                let byte_off = ptr_usize - meta.base_addr;
                if byte_off % core::mem::size_of::<i64>() != 0 {
                    return None;
                }
                let word_off = (byte_off / core::mem::size_of::<i64>()) as u16;
                return meta
                    .entries
                    .binary_search_by_key(&word_off, |e| e.word_off)
                    .ok()
                    .map(|idx| meta.entries[idx].rec.clone());
            }
        }
        None
    }

    /// Allocates an owned typed MIR array buffer directly populated with `elems` in one pass
    /// (zero intermediate `Value::Array`, zero per-element `lust_mir_array_set` HashMap lookups),
    /// enforcing exact `SEC-02` memory limits and `allocation_records` provenance.
    #[inline]
    pub fn alloc_array_with_words(&self, elem_ty: crate::mir::ir::MirType, elems: &[i64]) -> *mut i64 {
        let count = elems.len();
        let payload_slots = count.max(1);
        let Some(size) = payload_slots.checked_add(2) else {
            self.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
            return core::ptr::null_mut();
        };
        let Some(bytes_u) = size.checked_mul(core::mem::size_of::<i64>()) else {
            self.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
            return core::ptr::null_mut();
        };
        let Ok(bytes_needed) = i64::try_from(bytes_u) else {
            self.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
            return core::ptr::null_mut();
        };
        let limit = self.max_memory_limit.load(Ordering::Relaxed);
        let prev = self.memory_used.fetch_add(bytes_needed, Ordering::Relaxed);
        if prev.saturating_add(bytes_needed) > limit {
            self.memory_used.fetch_sub(bytes_needed, Ordering::Relaxed);
            self.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
            return core::ptr::null_mut();
        }
        let mem_rollback = core::mem::DropGuard::new(bytes_needed, |bytes| {
            self.memory_used.fetch_sub(bytes, Ordering::Relaxed);
        });
        let kind = MirAllocationKind::Array(elem_ty);
        #[cfg(feature = "std")]
        let kind = match self.try_bump_alloc(size, count as i64, count as i64, Some(elems), kind) {
            Ok(ptr) => {
                core::mem::DropGuard::dismiss(mem_rollback);
                return ptr;
            }
            Err(k) => k,
        };
        #[cfg(feature = "std")]
        let Some(mut slice) = self.take_or_alloc_buffer(size)
        else {
            self.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
            return core::ptr::null_mut();
        };
        #[cfg(not(feature = "std"))]
        let mut slice = {
            let mut vec = Vec::<i64>::new();
            if vec.try_reserve_exact(size).is_err() {
                self.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
                return core::ptr::null_mut();
            }
            vec.resize(size, 0);
            vec.into_boxed_slice()
        };
        slice[0] = count as i64;
        slice[1] = count as i64;
        if count > 0 {
            slice[2..2 + count].copy_from_slice(elems);
        }
        let ptr = {
            #[cfg(feature = "std")]
            {
                let handle_epoch = self.handle_count();
                let scope_depth = self.scope_alloc_floors.borrow().len();
                let mut bufs = self.allocated_buffers.borrow_mut();
                let mut records = self.allocation_records.borrow_mut();
                if bufs.try_reserve(1).is_err() || records.try_reserve(1).is_err() {
                    self.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
                    return core::ptr::null_mut();
                }
                let seq = self.next_alloc_seq.fetch_add(1, Ordering::Relaxed);
                let ptr = Box::into_raw(slice) as *mut i64;
                let buffer_index = bufs.len();
                bufs.push((ptr, size));
                records.insert(
                    ptr as usize,
                    MirAllocationRecord {
                        base_ptr: ptr,
                        word_capacity: size,
                        kind,
                        buffer_index,
                        handle_epoch,
                        alloc_seq: seq,
                        is_ctx_owned: true,
                        read_only_borrowed: false,
                        registered_scope_depth: scope_depth,
                    },
                );
                drop(records);
                drop(bufs);
                self.mark_dirty_allocation();
                ptr
            }
            #[cfg(not(feature = "std"))]
            {
                let _ = kind;
                Box::into_raw(slice) as *mut i64
            }
        };
        core::mem::DropGuard::dismiss(mem_rollback);
        ptr
    }

    /// Allocates an owned struct buffer populated directly with `fields`, interning `name`
    /// and enforcing the exact `SEC-02` memory budget, `try_reserve_exact` + `DropGuard`
    /// OOM rollback, and `allocation_records` provenance contract.
    #[inline]
    pub fn alloc_struct_with_fields(&self, name: &str, fields: &[i64]) -> *mut i64 {
        if name.is_empty() {
            self.last_status.store(
                crate::effect_abi::LUST_STATUS_ERR_GENERIC as i64,
                Ordering::Relaxed,
            );
            return core::ptr::null_mut();
        }
        let count = fields.len();
        let Some(size) = count.checked_add(2) else {
            self.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
            return core::ptr::null_mut();
        };
        let Some(bytes_u) = size.checked_mul(core::mem::size_of::<i64>()) else {
            self.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
            return core::ptr::null_mut();
        };
        let Ok(bytes_needed) = i64::try_from(bytes_u) else {
            self.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
            return core::ptr::null_mut();
        };
        let limit = self.max_memory_limit.load(Ordering::Relaxed);
        let prev = self.memory_used.fetch_add(bytes_needed, Ordering::Relaxed);
        if prev.saturating_add(bytes_needed) > limit {
            self.memory_used.fetch_sub(bytes_needed, Ordering::Relaxed);
            self.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
            return core::ptr::null_mut();
        }
        let mem_rollback = core::mem::DropGuard::new(bytes_needed, |bytes| {
            self.memory_used.fetch_sub(bytes, Ordering::Relaxed);
        });
        let kind = MirAllocationKind::Struct(self.intern_struct_type_name(name));
        #[cfg(feature = "std")]
        let kind = match self.try_bump_alloc(size, count as i64, count as i64, Some(fields), kind) {
            Ok(ptr) => {
                core::mem::DropGuard::dismiss(mem_rollback);
                return ptr;
            }
            Err(k) => k,
        };
        #[cfg(feature = "std")]
        let Some(mut slice) = self.take_or_alloc_buffer(size)
        else {
            self.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
            return core::ptr::null_mut();
        };
        #[cfg(not(feature = "std"))]
        let mut slice = {
            let mut vec = Vec::<i64>::new();
            if vec.try_reserve_exact(size).is_err() {
                self.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
                return core::ptr::null_mut();
            }
            vec.resize(size, 0);
            vec.into_boxed_slice()
        };
        slice[0] = count as i64;
        slice[1] = count as i64;
        if count > 0 {
            slice[2..2 + count].copy_from_slice(fields);
        }
        let ptr = {
            #[cfg(feature = "std")]
            {
                let handle_epoch = self.handle_count();
                let scope_depth = self.scope_alloc_floors.borrow().len();
                let mut bufs = self.allocated_buffers.borrow_mut();
                let mut records = self.allocation_records.borrow_mut();
                if bufs.try_reserve(1).is_err() || records.try_reserve(1).is_err() {
                    self.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
                    return core::ptr::null_mut();
                }
                let seq = self.next_alloc_seq.fetch_add(1, Ordering::Relaxed);
                let ptr = Box::into_raw(slice) as *mut i64;
                let buffer_index = bufs.len();
                bufs.push((ptr, size));
                records.insert(
                    ptr as usize,
                    MirAllocationRecord {
                        base_ptr: ptr,
                        word_capacity: size,
                        kind,
                        buffer_index,
                        handle_epoch,
                        alloc_seq: seq,
                        is_ctx_owned: true,
                        read_only_borrowed: false,
                        registered_scope_depth: scope_depth,
                    },
                );
                drop(records);
                drop(bufs);
                self.mark_dirty_allocation();
                ptr
            }
            #[cfg(not(feature = "std"))]
            {
                let _ = kind;
                Box::into_raw(slice) as *mut i64
            }
        };
        core::mem::DropGuard::dismiss(mem_rollback);
        ptr
    }

    /// Read-only inspection helper returning the count of interned struct type names in this context.
    #[inline]
    pub fn interned_struct_type_name_count(&self) -> usize {
        #[cfg(feature = "std")]
        {
            self.struct_type_names.borrow().len()
        }
        #[cfg(not(feature = "std"))]
        {
            0
        }
    }

    pub fn get_struct_field_type(&self, name: &str, idx: usize) -> Option<crate::mir::ir::MirType> {
        #[cfg(feature = "std")]
        {
            let sf = self.struct_fields.borrow();
            if let Some(types) = sf.get(name) {
                return types.get(idx).cloned();
            }
            drop(sf);
            DEFAULT_MIR_CONTEXT.with(|d| {
                let d_sf = d.struct_fields.borrow();
                d_sf.get(name).and_then(|t| t.get(idx).cloned())
            })
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (name, idx);
            None
        }
    }

    pub fn buffer_count(&self) -> usize {
        #[cfg(feature = "std")]
        {
            self.allocated_buffers.borrow().len()
        }
        #[cfg(not(feature = "std"))]
        {
            0
        }
    }

    pub fn capture_baseline(&self) -> MirContextBaseline {
        #[cfg(feature = "std")]
        if !self.dirty_allocations.get() {
            return MirContextBaseline {
                handles: 0,
                buffers: 0,
            };
        }
        MirContextBaseline {
            handles: self.handle_count(),
            buffers: self.buffer_count(),
        }
    }

    pub fn active_buffer_baseline(&self) -> usize {
        #[cfg(feature = "std")]
        {
            self.scope_baselines
                .borrow()
                .last()
                .map_or(0, |b| b.buffers)
        }
        #[cfg(not(feature = "std"))]
        {
            0
        }
    }

    pub fn active_handle_baseline(&self) -> usize {
        #[cfg(feature = "std")]
        {
            self.scope_baselines
                .borrow()
                .last()
                .map_or(0, |b| b.handles)
        }
        #[cfg(not(feature = "std"))]
        {
            0
        }
    }

    pub fn push_owned_buffer(&self, ptr: *mut i64, words: usize, kind: MirAllocationKind) {
        #[cfg(feature = "std")]
        {
            let handle_epoch = self.handle_count();
            let seq = self.next_alloc_seq.fetch_add(1, Ordering::SeqCst);
            let scope_depth = self.scope_alloc_floors.borrow().len();
            {
                let mut bufs = self.allocated_buffers.borrow_mut();
                let mut records = self.allocation_records.borrow_mut();
                let buffer_index = bufs.len();
                bufs.push((ptr, words));
                records.insert(
                    ptr as usize,
                    MirAllocationRecord {
                        base_ptr: ptr,
                        word_capacity: words,
                        kind,
                        buffer_index,
                        handle_epoch,
                        alloc_seq: seq,
                        is_ctx_owned: true,
                        read_only_borrowed: false,
                        registered_scope_depth: scope_depth,
                    },
                );
            }
            self.mark_dirty_allocation();
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (ptr, words, kind);
        }
    }

    pub fn register_allocation(&self, ptr: *mut i64, words: usize, kind: MirAllocationKind) {
        #[cfg(feature = "std")]
        {
            let handle_epoch = self.handle_count();
            let seq = self.next_alloc_seq.fetch_add(1, Ordering::SeqCst);
            let scope_depth = self.scope_alloc_floors.borrow().len();
            let bufs_match = self
                .allocated_buffers
                .borrow()
                .iter()
                .rposition(|(p, _)| *p == ptr);
            let updated = {
                let mut records = self.allocation_records.borrow_mut();
                if let Some(existing) = records.get_mut(&(ptr as usize)) {
                    if existing.read_only_borrowed {
                        false
                    } else {
                        existing.base_ptr = ptr;
                        existing.word_capacity = words;
                        existing.kind = kind;
                        existing.alloc_seq = seq;
                        existing.registered_scope_depth = scope_depth;
                        existing.handle_epoch = handle_epoch;
                        if let Some(idx) = bufs_match {
                            existing.is_ctx_owned = true;
                            existing.buffer_index = idx;
                        }
                        true
                    }
                } else {
                    let is_ctx_owned = bufs_match.is_some();
                    let buffer_index = bufs_match.unwrap_or(usize::MAX);
                    records.insert(
                        ptr as usize,
                        MirAllocationRecord {
                            base_ptr: ptr,
                            word_capacity: words,
                            kind,
                            buffer_index,
                            handle_epoch,
                            alloc_seq: seq,
                            is_ctx_owned,
                            read_only_borrowed: false,
                            registered_scope_depth: scope_depth,
                        },
                    );
                    true
                }
            };
            if updated {
                self.mark_dirty_allocation();
            }
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (ptr, words, kind);
        }
    }

    pub fn get_allocation_record(&self, ptr_usize: usize) -> Option<MirAllocationRecord> {
        #[cfg(feature = "std")]
        {
            if let Some(rec) = self.allocation_records.borrow().get(&ptr_usize).cloned() {
                return Some(rec);
            }
            self.lookup_bump_record(ptr_usize)
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = ptr_usize;
            None
        }
    }

    pub fn get_active_allocation_record(&self, ptr_usize: usize) -> Option<MirAllocationRecord> {
        #[cfg(feature = "std")]
        {
            let seq_floor = self
                .scope_alloc_floors
                .borrow()
                .last()
                .copied()
                .unwrap_or(0);
            let buf_floor = self.active_buffer_baseline();
            let rec_opt = self
                .allocation_records
                .borrow()
                .get(&ptr_usize)
                .cloned()
                .or_else(|| self.lookup_bump_record(ptr_usize));
            rec_opt.filter(|rec| {
                rec.base_ptr as usize == ptr_usize
                    && rec.alloc_seq >= seq_floor
                    && (!rec.is_ctx_owned || rec.buffer_index >= buf_floor)
            })
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = ptr_usize;
            None
        }
    }

    /// Registers a scope-bound borrowed read-only flat array provenance record (`PERF-T4-3`).
    ///
    /// The underlying memory is owned by `Rc<FlatArrayBuffer>` pinned by the caller and is NEVER
    /// pushed to `allocated_buffers` or freed by `clear_to_baselines`. Unregistered automatically
    /// in reverse order when `MirContextGuard::drop` exits the current scope.
    pub fn register_borrowed_readonly_array(
        &self,
        ptr: *const i64,
        word_capacity: usize,
        elem_type: crate::mir::ir::MirType,
    ) -> core::result::Result<(), alloc::string::String> {
        if ptr.is_null() {
            return Err("Cannot register null pointer as borrowed read-only array".into());
        }
        if word_capacity < 2 {
            return Err(alloc::format!(
                "Cannot register borrowed read-only array with word_capacity ({word_capacity}) < 2"
            ));
        }
        #[cfg(feature = "std")]
        {
            let handle_epoch = self.handle_count();
            let seq = self.next_alloc_seq.fetch_add(1, Ordering::SeqCst);
            let scope_depth = self.scope_alloc_floors.borrow().len();
            let addr = ptr as usize;
            {
                let mut records = self.allocation_records.borrow_mut();
                if let Some(existing) = records.get(&addr)
                    && existing.is_ctx_owned
                {
                    return Err(alloc::format!(
                        "Cannot register borrowed read-only array over context-owned allocation at {addr:#x}"
                    ));
                }
                let prev = records.get(&addr).cloned();
                records.insert(
                    addr,
                    MirAllocationRecord {
                        base_ptr: ptr as *mut i64,
                        word_capacity,
                        kind: MirAllocationKind::Array(elem_type),
                        buffer_index: usize::MAX,
                        handle_epoch,
                        alloc_seq: seq,
                        is_ctx_owned: false,
                        read_only_borrowed: true,
                        registered_scope_depth: scope_depth,
                    },
                );
                self.borrowed_readonly_stack.borrow_mut().push((addr, prev));
            }
            self.mark_dirty_allocation();
            Ok(())
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = elem_type;
            Err("Borrowed read-only arrays require std feature".into())
        }
    }

    pub fn set_allocation_kind(&self, ptr_usize: usize, kind: MirAllocationKind) -> bool {
        #[cfg(feature = "std")]
        {
            let updated = {
                let mut records = self.allocation_records.borrow_mut();
                if let Some(rec) = records.get_mut(&ptr_usize) {
                    if rec.read_only_borrowed {
                        false
                    } else {
                        rec.kind = kind;
                        true
                    }
                } else {
                    drop(records);
                    let mut metas = self.bump_slab_meta.borrow_mut();
                    let mut found = false;
                    for meta in metas.iter_mut().rev() {
                        if ptr_usize >= meta.base_addr && ptr_usize < meta.end_addr {
                            let byte_off = ptr_usize - meta.base_addr;
                            if byte_off % core::mem::size_of::<i64>() == 0 {
                                let word_off = (byte_off / core::mem::size_of::<i64>()) as u16;
                                if let Ok(idx) =
                                    meta.entries.binary_search_by_key(&word_off, |e| e.word_off)
                                {
                                    if !meta.entries[idx].rec.read_only_borrowed {
                                        meta.entries[idx].rec.kind = kind;
                                        found = true;
                                    }
                                }
                            }
                            break;
                        }
                    }
                    found
                }
            };
            if updated {
                self.mark_dirty_allocation();
            }
            updated
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (ptr_usize, kind);
            false
        }
    }

    pub fn clear_to_baselines(&self, baseline_handles: usize, baseline_buffers: usize) {
        #[cfg(feature = "std")]
        {
            if !self.dirty_allocations.get() && baseline_handles == 0 && baseline_buffers == 0 {
                return;
            }
            let (scope_depth, seq_floor) = {
                let floors = self.scope_alloc_floors.borrow();
                (floors.len(), floors.last().copied().unwrap_or(0))
            };
            let (handle_floor, buf_floor) = {
                let scopes = self.scope_baselines.borrow();
                scopes
                    .last()
                    .map_or((0, 0), |b| (b.handles, b.buffers))
            };
            let effective_handles = if scope_depth > 0 {
                baseline_handles.max(handle_floor)
            } else {
                baseline_handles
            };
            let effective_buffers = if scope_depth > 0 {
                baseline_buffers.max(buf_floor)
            } else {
                baseline_buffers
            };
            let hv_empty = {
                let mut hv = self.heap_values.borrow_mut();
                hv.truncate(effective_handles);
                hv.is_empty()
            };
            let drained: Vec<(*mut i64, usize)> = {
                let mut bufs = self.allocated_buffers.borrow_mut();
                let mut records = self.allocation_records.borrow_mut();
                let start = effective_buffers.min(bufs.len());
                if effective_handles == 0 && start == 0 && scope_depth == 0 {
                    records.clear();
                    self.borrowed_readonly_stack.borrow_mut().clear();
                    self.native_queues.borrow_mut().clear();
                    self.dyn_interned_map.borrow_mut().clear();
                    self.dyn_interned_list.borrow_mut().clear();
                    {
                        let mut slabs = self.bump_slabs.borrow_mut();
                        let mut metas = self.bump_slab_meta.borrow_mut();
                        if !metas.is_empty() {
                            metas[0].entries.clear();
                            metas.truncate(1);
                            slabs.truncate(1);
                        }
                        self.bump_cursor.set((0, 0));
                    }
                    self.dirty_allocations.set(false);
                    core::mem::take(&mut *bufs)
                } else {
                    let items: Vec<(*mut i64, usize)> = bufs.drain(start..).collect();
                    let mut had_slab = false;
                    for &(ptr, cap) in &items {
                        if !ptr.is_null() {
                            if cap == 0 {
                                had_slab = true;
                            } else {
                                records.remove(&(ptr as usize));
                            }
                        }
                    }
                    if had_slab {
                        let mut metas = self.bump_slab_meta.borrow_mut();
                        let mut slabs = self.bump_slabs.borrow_mut();
                        while let Some(last_meta) = metas.last_mut() {
                            while let Some(last_entry) = last_meta.entries.last() {
                                if last_entry.rec.buffer_index >= start {
                                    last_meta.entries.pop();
                                } else {
                                    break;
                                }
                            }
                            if last_meta.entries.is_empty() && metas.len() > 1 {
                                metas.pop();
                                slabs.pop();
                            } else {
                                break;
                            }
                        }
                        let new_cursor = if let Some(last_meta) = metas.last() {
                            let slab_idx = metas.len() - 1;
                            let next_off = last_meta
                                .entries
                                .last()
                                .map_or(0, |e| (e.word_off as usize) + e.rec.word_capacity);
                            (slab_idx, next_off)
                        } else {
                            (0, 0)
                        };
                        self.bump_cursor.set(new_cursor);
                    }
                    if scope_depth > 0
                        && baseline_handles <= handle_floor
                        && baseline_buffers <= buf_floor
                    {
                        records.retain(|_, rec| {
                            rec.is_ctx_owned
                                || rec.read_only_borrowed
                                || !(rec.registered_scope_depth >= scope_depth
                                    && rec.alloc_seq >= seq_floor)
                        });
                    }
                    if hv_empty
                        && bufs.is_empty()
                        && records.is_empty()
                        && self.borrowed_readonly_stack.borrow().is_empty()
                        && self.dyn_interned_list.borrow().is_empty()
                    {
                        self.dirty_allocations.set(false);
                    }
                    items
                }
            };
            let mut pool = self.small_buf_pool.borrow_mut();
            for (ptr, cap) in drained {
                if !ptr.is_null() && cap > 0 {
                    let boxed = unsafe {
                        let slice = core::ptr::slice_from_raw_parts_mut(ptr, cap);
                        Box::from_raw(slice)
                    };
                    if cap <= SMALL_BUF_POOL_MAX_WORDS
                        && pool[cap].len() < SMALL_BUF_POOL_BUCKET_CAP
                    {
                        pool[cap].push(boxed);
                    } else {
                        drop(boxed);
                    }
                }
            }
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (baseline_handles, baseline_buffers);
        }
    }

    /// Cleans up heap values above `baseline_handles` and any context-owned buffers
    /// allocated at or after `handle_epoch >= baseline_handles` within the active
    /// `ctx.enter()` scope (`buffer_index >= active_buffer_baseline()`).
    ///
    /// **Shared `handle_epoch` invariant:** Inside a nested `ctx.enter()` scope
    /// (`active_buffer_baseline() > 0`), outer buffers at index `< active_buffer_baseline()`
    /// are strictly protected even if they share the exact same `handle_epoch == baseline_handles`.
    /// Within a single flat scope (without `ctx.enter()`), callers needing sub-scope cleanup
    /// inside the same `handle_epoch` must use `capture_baseline()` +
    /// `clear_to_baselines(b.handles, b.buffers)`.
    pub fn clear_to_baseline(&self, baseline_handles: usize) {
        #[cfg(feature = "std")]
        {
            if !self.dirty_allocations.get() && baseline_handles == 0 {
                return;
            }
            let scope_buf_floor = self.active_buffer_baseline();
            let buf_baseline = {
                let bufs = self.allocated_buffers.borrow();
                let floor = scope_buf_floor.min(bufs.len());
                if baseline_handles == 0 && floor == 0 {
                    0
                } else {
                    let records = self.allocation_records.borrow();
                    let first_after = (floor..bufs.len()).find(|&idx| {
                        let ptr = bufs[idx].0;
                        if ptr.is_null() {
                            return false;
                        }
                        let ptr_addr = ptr as usize;
                        match records
                            .get(&ptr_addr)
                            .cloned()
                            .or_else(|| self.lookup_bump_record(ptr_addr))
                        {
                            Some(rec) => rec.handle_epoch >= baseline_handles,
                            None => true,
                        }
                    });
                    first_after.unwrap_or(bufs.len())
                }
            };
            self.clear_to_baselines(baseline_handles, buf_baseline);
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = baseline_handles;
        }
    }

    /// Validates an active array header pointer (both flat `cap >= 0` and redirected `cap < 0`)
    /// without mutating `last_status`. Returns `(len, cap, data_ptr)` on success.
    pub fn resolve_validated_array_data(&self, ptr: *const i64) -> Option<(usize, usize, *const i64)> {
        validate_active_array_inner(self, ptr).map(|v| (v.len, v.cap, v.data_ptr as *const i64))
    }
}

pub struct MirContextGuard(*const MirContext);

impl Drop for MirContextGuard {
    fn drop(&mut self) {
        #[cfg(feature = "std")]
        {
            if !self.0.is_null() {
                let ctx = unsafe { &*self.0 };
                let borrowed_floor = ctx.borrowed_scope_floors.borrow_mut().pop().unwrap_or(0);
                if ctx.dirty_allocations.get() {
                    let mut stack = ctx.borrowed_readonly_stack.borrow_mut();
                    let start = borrowed_floor.min(stack.len());
                    if start < stack.len() {
                        let mut records = ctx.allocation_records.borrow_mut();
                        for (addr, prev_opt) in stack.drain(start..).rev() {
                            match prev_opt {
                                Some(prev) => {
                                    records.insert(addr, prev);
                                }
                                None => {
                                    records.remove(&addr);
                                }
                            }
                        }
                    }
                }
                ctx.scope_baselines.borrow_mut().pop();
                ctx.scope_alloc_floors.borrow_mut().pop();
            }
            CURRENT_MIR_STACK.with(|stack| {
                let mut stack = stack.borrow_mut();
                if let Some(top) = stack.last() {
                    if *top == self.0 {
                        stack.pop();
                        return;
                    }
                }
                if let Some(pos) = stack.iter().rposition(|&p| p == self.0) {
                    stack.remove(pos);
                }
            });
        }
    }
}

pub fn enter_mir_context(ctx: &MirContext) -> MirContextGuard {
    let ptr = ctx as *const MirContext;
    #[cfg(feature = "std")]
    {
        let (baseline, borrowed_floor) = if !ctx.dirty_allocations.get() {
            (
                MirContextBaseline {
                    handles: 0,
                    buffers: 0,
                },
                0,
            )
        } else {
            (
                ctx.capture_baseline(),
                ctx.borrowed_readonly_stack.borrow().len(),
            )
        };
        let seq_floor = ctx.next_alloc_seq.load(Ordering::SeqCst);
        ctx.scope_baselines.borrow_mut().push(baseline);
        ctx.scope_alloc_floors.borrow_mut().push(seq_floor);
        ctx.borrowed_scope_floors.borrow_mut().push(borrowed_floor);
        CURRENT_MIR_STACK.with(|stack| {
            stack.borrow_mut().push(ptr);
        });
    }
    MirContextGuard(ptr)
}

#[cfg(feature = "std")]
thread_local! {
    static CURRENT_MIR_STACK: core::cell::RefCell<Vec<*const MirContext>> = const { core::cell::RefCell::new(Vec::new()) };
    static DEFAULT_MIR_CONTEXT: MirContext = MirContext::new();
}

#[cfg(feature = "std")]
pub fn default_mir_context_allocated_buffer_count() -> usize {
    DEFAULT_MIR_CONTEXT.with(|ctx| ctx.allocated_buffers.borrow().len())
}

pub fn with_current_mir_context<F, R>(f: F) -> R
where
    F: FnOnce(&MirContext) -> R,
{
    #[cfg(feature = "std")]
    {
        let top = CURRENT_MIR_STACK.with(|stack| stack.borrow().last().copied());
        if let Some(top) = top {
            let ctx = unsafe { &*top };
            f(ctx)
        } else {
            DEFAULT_MIR_CONTEXT.with(|default_ctx| f(default_ctx))
        }
    }
    #[cfg(not(feature = "std"))]
    {
        static FALLBACK: MirContext = MirContext::new();
        f(&FALLBACK)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_gas_fast_ptr() -> *mut GasFastState {
    #[cfg(feature = "std")]
    {
        with_current_mir_context(|ctx| ctx.gas_fast_ptr())
    }
    #[cfg(not(feature = "std"))]
    {
        core::ptr::null_mut()
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_gas_slow() -> i64 {
    with_current_mir_context(|ctx| {
        #[cfg(feature = "std")]
        {
            if ctx.gas_fast.gas_unbounded.get() != 0 {
                ctx.gas_fast.gas_remaining.set(unbounded_gas_chunk());
                UNBOUNDED_GAS_REPLENISH_COUNT.with(|c| c.set(c.get().saturating_add(1)));
                return MIR_STATUS_OK;
            }
        }
        ctx.last_status
            .store(MIR_STATUS_ERR_OUT_OF_GAS, Ordering::Relaxed);
        MIR_STATUS_ERR_OUT_OF_GAS
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_set_gas(gas: i64) {
    with_current_mir_context(|ctx| {
        ctx.set_gas_unbounded(false);
        ctx.set_gas(gas);
        if gas >= 0 {
            ctx.last_status.store(0, Ordering::Relaxed);
        }
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_get_gas() -> i64 {
    with_current_mir_context(|ctx| ctx.gas_remaining())
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_charge_gas(amt: i64) -> i64 {
    with_current_mir_context(|ctx| ctx.charge_gas(amt))
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_get_status() -> i64 {
    with_current_mir_context(|ctx| ctx.last_status.load(Ordering::Relaxed))
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_set_status(status: i64) {
    with_current_mir_context(|ctx| ctx.last_status.store(status, Ordering::Relaxed));
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_set_max_memory(bytes: i64) {
    with_current_mir_context(|ctx| ctx.max_memory_limit.store(bytes, Ordering::Relaxed));
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_set_memory_limit(bytes: i64) {
    lust_mir_set_max_memory(bytes);
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_get_memory_used() -> i64 {
    with_current_mir_context(|ctx| ctx.memory_used.load(Ordering::Relaxed))
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_track_memory_allocated(bytes: i64) -> i64 {
    with_current_mir_context(|ctx| {
        ctx.memory_used.fetch_add(bytes, Ordering::Relaxed) + bytes
    })
}

pub const MIR_ARRAY_ELEM_TAG_INT: i64 = 1;
pub const MIR_ARRAY_ELEM_TAG_BOOL: i64 = 2;
pub const MIR_ARRAY_ELEM_TAG_FLOAT: i64 = 3;
pub const MIR_ARRAY_ELEM_TAG_STRING: i64 = 4;

pub fn mir_array_elem_tag(ty: &crate::mir::ir::MirType) -> Option<i64> {
    match ty {
        crate::mir::ir::MirType::Int => Some(MIR_ARRAY_ELEM_TAG_INT),
        crate::mir::ir::MirType::Bool => Some(MIR_ARRAY_ELEM_TAG_BOOL),
        crate::mir::ir::MirType::Float => Some(MIR_ARRAY_ELEM_TAG_FLOAT),
        crate::mir::ir::MirType::String => Some(MIR_ARRAY_ELEM_TAG_STRING),
        _ => None,
    }
}

pub fn mir_array_elem_type_from_tag(tag: i64) -> Option<crate::mir::ir::MirType> {
    match tag {
        MIR_ARRAY_ELEM_TAG_INT => Some(crate::mir::ir::MirType::Int),
        MIR_ARRAY_ELEM_TAG_BOOL => Some(crate::mir::ir::MirType::Bool),
        MIR_ARRAY_ELEM_TAG_FLOAT => Some(crate::mir::ir::MirType::Float),
        MIR_ARRAY_ELEM_TAG_STRING => Some(crate::mir::ir::MirType::String),
        _ => None,
    }
}

#[inline]
fn is_bool_array_kind(kind: &MirAllocationKind) -> bool {
    matches!(kind, MirAllocationKind::Array(crate::mir::ir::MirType::Bool))
}

/// Minimum number of 64-bit (`i64`) payload words required to store `elem_cap` elements.
/// `Array(Bool)` uses 1-byte (`u8`) element stride rounded up to 8-byte words (`NG-W10-5`).
#[inline]
fn array_payload_words_for_cap(elem_cap: usize, is_bool: bool) -> Option<usize> {
    if is_bool {
        elem_cap.checked_add(7).map(|n| n / 8)
    } else {
        Some(elem_cap)
    }
}

#[inline]
fn alloc_mir_buffer_in_ctx(ctx: &MirContext, len: i64, kind: MirAllocationKind) -> *mut i64 {
    let count = if len < 0 { 0 } else { len as usize };
    let is_bool_arr = is_bool_array_kind(&kind);
    // Arrays always allocate at least 3 words (`max(elem_words, 1) + 2`) so word[2] can hold
    // a redirected backing-buffer pointer when growing via `lust_mir_array_push`.
    let payload_slots = if matches!(
        kind,
        MirAllocationKind::Array(_) | MirAllocationKind::RawWords
    ) {
        let Some(elem_words) = array_payload_words_for_cap(count, is_bool_arr) else {
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
            return core::ptr::null_mut();
        };
        elem_words.max(1)
    } else {
        count
    };
    let Some(size) = payload_slots.checked_add(2) else {
        ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
        return core::ptr::null_mut();
    };
    let Some(bytes_u) = size.checked_mul(core::mem::size_of::<i64>()) else {
        ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
        return core::ptr::null_mut();
    };
    let Ok(bytes_needed) = i64::try_from(bytes_u) else {
        ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
        return core::ptr::null_mut();
    };
    let limit = ctx.max_memory_limit.load(Ordering::Relaxed);
    let prev = ctx.memory_used.fetch_add(bytes_needed, Ordering::Relaxed);
    if prev.saturating_add(bytes_needed) > limit {
        ctx.memory_used.fetch_sub(bytes_needed, Ordering::Relaxed);
        ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
        return core::ptr::null_mut();
    }
    let mem_rollback = core::mem::DropGuard::new(bytes_needed, |bytes| {
        ctx.memory_used.fetch_sub(bytes, Ordering::Relaxed);
    });
    #[cfg(feature = "std")]
    let kind = match ctx.try_bump_alloc(size, count as i64, count as i64, None, kind) {
        Ok(ptr) => {
            core::mem::DropGuard::dismiss(mem_rollback);
            return ptr;
        }
        Err(k) => k,
    };
    #[cfg(feature = "std")]
    let Some(mut slice) = ctx.take_or_alloc_buffer(size)
    else {
        ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
        return core::ptr::null_mut();
    };
    #[cfg(not(feature = "std"))]
    let mut slice = {
        let mut vec = Vec::<i64>::new();
        if vec.try_reserve_exact(size).is_err() {
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
            return core::ptr::null_mut();
        }
        vec.resize(size, 0);
        vec.into_boxed_slice()
    };
    slice[0] = count as i64;
    slice[1] = count as i64;
    let ptr = {
        #[cfg(feature = "std")]
        {
            let handle_epoch = ctx.handle_count();
            let scope_depth = ctx.scope_alloc_floors.borrow().len();
            let mut bufs = ctx.allocated_buffers.borrow_mut();
            let mut records = ctx.allocation_records.borrow_mut();
            if bufs.try_reserve(1).is_err() || records.try_reserve(1).is_err() {
                ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
                return core::ptr::null_mut();
            }
            let seq = ctx.next_alloc_seq.fetch_add(1, Ordering::Relaxed);
            let ptr = Box::into_raw(slice) as *mut i64;
            let buffer_index = bufs.len();
            bufs.push((ptr, size));
            records.insert(
                ptr as usize,
                MirAllocationRecord {
                    base_ptr: ptr,
                    word_capacity: size,
                    kind,
                    buffer_index,
                    handle_epoch,
                    alloc_seq: seq,
                    is_ctx_owned: true,
                    read_only_borrowed: false,
                    registered_scope_depth: scope_depth,
                },
            );
            drop(records);
            drop(bufs);
            ctx.mark_dirty_allocation();
            ptr
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = kind;
            Box::into_raw(slice) as *mut i64
        }
    };
    core::mem::DropGuard::dismiss(mem_rollback);
    ptr
}

fn alloc_mir_buffer_with_kind(len: i64, kind: MirAllocationKind) -> *mut i64 {
    with_current_mir_context(|ctx| alloc_mir_buffer_in_ctx(ctx, len, kind))
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_alloc_array(len: i64) -> *mut i64 {
    alloc_mir_buffer_with_kind(len, MirAllocationKind::RawWords)
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_alloc_array_typed(len: i64, elem_tag: i64) -> *mut i64 {
    let Some(elem_ty) = mir_array_elem_type_from_tag(elem_tag) else {
        with_current_mir_context(|ctx| {
            ctx.last_status.store(crate::effect_abi::LUST_STATUS_ERR_GENERIC as i64, Ordering::SeqCst);
        });
        return core::ptr::null_mut();
    };
    alloc_mir_buffer_with_kind(len, MirAllocationKind::Array(elem_ty))
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_alloc_tuple(len: i64) -> *mut i64 {
    alloc_mir_buffer_with_kind(len, MirAllocationKind::Tuple(Vec::new()))
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_alloc_tuple_typed(len: i64, tags_ptr: *const i64) -> *mut i64 {
    if len < 0 || (len > 0 && tags_ptr.is_null()) {
        with_current_mir_context(|ctx| {
            ctx.last_status
                .store(crate::effect_abi::LUST_STATUS_ERR_GENERIC as i64, Ordering::SeqCst);
        });
        return core::ptr::null_mut();
    }
    let count = len as usize;
    let mut elem_types = Vec::with_capacity(count);
    for i in 0..count {
        let tag = unsafe { *tags_ptr.add(i) };
        let Some(ty) = mir_array_elem_type_from_tag(tag) else {
            with_current_mir_context(|ctx| {
                ctx.last_status
                    .store(crate::effect_abi::LUST_STATUS_ERR_GENERIC as i64, Ordering::SeqCst);
            });
            return core::ptr::null_mut();
        };
        elem_types.push(ty);
    }
    alloc_mir_buffer_with_kind(len, MirAllocationKind::Tuple(elem_types))
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_alloc_struct(len: i64) -> *mut i64 {
    with_current_mir_context(|ctx| {
        let name = ctx.intern_struct_type_name("");
        alloc_mir_buffer_in_ctx(ctx, len, MirAllocationKind::Struct(name))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_alloc_struct_named(
    len: i64,
    name_ptr: *const u8,
    name_len: i64,
) -> *mut i64 {
    if len < 0 || name_len <= 0 || name_ptr.is_null() {
        with_current_mir_context(|ctx| {
            ctx.last_status
                .store(crate::effect_abi::LUST_STATUS_ERR_GENERIC as i64, Ordering::SeqCst);
        });
        return core::ptr::null_mut();
    }
    let slice = unsafe { core::slice::from_raw_parts(name_ptr, name_len as usize) };
    with_current_mir_context(|ctx| {
        let Some(interned) = ctx.intern_struct_type_name_bytes(slice) else {
            ctx.last_status
                .store(crate::effect_abi::LUST_STATUS_ERR_GENERIC as i64, Ordering::SeqCst);
            return core::ptr::null_mut();
        };
        alloc_mir_buffer_in_ctx(ctx, len, MirAllocationKind::Struct(interned))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_alloc_cflat_struct(
    size_bytes: i64,
    name_ptr: *const u8,
    name_len: i64,
) -> *mut i64 {
    if name_len <= 0 || name_ptr.is_null() {
        with_current_mir_context(|ctx| {
            ctx.last_status
                .store(crate::effect_abi::LUST_STATUS_ERR_GENERIC as i64, Ordering::SeqCst);
        });
        return core::ptr::null_mut();
    }
    let slice = unsafe { core::slice::from_raw_parts(name_ptr, name_len as usize) };
    let Ok(name_str) = core::str::from_utf8(slice) else {
        with_current_mir_context(|ctx| {
            ctx.last_status
                .store(crate::effect_abi::LUST_STATUS_ERR_GENERIC as i64, Ordering::SeqCst);
        });
        return core::ptr::null_mut();
    };
    let bytes_req = usize::try_from(size_bytes.max(1)).unwrap_or(1);
    let words = bytes_req.saturating_add(7) / 8;
    let words = words.max(1);
    let Some(bytes_u) = words.checked_mul(core::mem::size_of::<i64>()) else {
        with_current_mir_context(|ctx| {
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
        });
        return core::ptr::null_mut();
    };
    let Ok(bytes_needed) = i64::try_from(bytes_u) else {
        with_current_mir_context(|ctx| {
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
        });
        return core::ptr::null_mut();
    };
    let oom = with_current_mir_context(|ctx| {
        let limit = ctx.max_memory_limit.load(Ordering::SeqCst);
        let prev = ctx.memory_used.fetch_add(bytes_needed, Ordering::SeqCst);
        if prev.saturating_add(bytes_needed) > limit {
            ctx.memory_used.fetch_sub(bytes_needed, Ordering::SeqCst);
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
            return true;
        }
        false
    });
    if oom {
        return core::ptr::null_mut();
    }
    let boxed = alloc::vec![0i64; words].into_boxed_slice();
    let ptr = Box::into_raw(boxed) as *mut i64;
    let name = alloc::string::ToString::to_string(name_str);
    with_current_mir_context(|ctx| {
        ctx.push_owned_buffer(ptr, words, MirAllocationKind::CFlatStruct(name));
    });
    ptr
}

#[cfg(feature = "std")]
struct ValidatedArray {
    base_ptr: *mut i64,
    len: usize,
    cap: usize,
    data_ptr: *mut i64,
    header_rec: MirAllocationRecord,
    backing_rec: Option<MirAllocationRecord>,
}

#[cfg(not(feature = "std"))]
struct ValidatedArray {
    base_ptr: *mut i64,
    len: usize,
    cap: usize,
    data_ptr: *mut i64,
}

fn validate_active_array_inner(ctx: &MirContext, ptr: *const i64) -> Option<ValidatedArray> {
    if ptr.is_null() {
        return None;
    }
    #[cfg(feature = "std")]
    {
        let rec = ctx.get_active_allocation_record(ptr as usize)?;
        if !matches!(
            rec.kind,
            MirAllocationKind::Array(_) | MirAllocationKind::RawWords
        ) {
            return None;
        }
        if rec.word_capacity < 2 {
            return None;
        }
        let len_raw = unsafe { *rec.base_ptr };
        let cap_raw = unsafe { *rec.base_ptr.add(1) };
        if len_raw < 0 {
            return None;
        }
        let len = len_raw as usize;
        let is_bool = is_bool_array_kind(&rec.kind);
        if cap_raw >= 0 {
            let cap = cap_raw as usize;
            let needed_payload_words = array_payload_words_for_cap(cap, is_bool)?;
            if len > cap
                || needed_payload_words
                    .checked_add(2)
                    .is_none_or(|n| n > rec.word_capacity)
            {
                return None;
            }
            let data_ptr = unsafe { rec.base_ptr.add(2) };
            Some(ValidatedArray {
                base_ptr: rec.base_ptr,
                len,
                cap,
                data_ptr,
                header_rec: rec,
                backing_rec: None,
            })
        } else {
            // Borrowed read-only flat buffers are strictly flat (`cap >= 0`), never redirected.
            if rec.read_only_borrowed {
                return None;
            }
            // Bit 63 set: redirected backing buffer stored at word[2]
            if rec.word_capacity < 3 {
                return None;
            }
            let cap = (cap_raw & i64::MAX) as usize;
            let needed_backing_words = array_payload_words_for_cap(cap, is_bool)?;
            if cap == 0 || len > cap {
                return None;
            }
            let data_raw = unsafe { *rec.base_ptr.add(2) };
            if data_raw == 0 {
                return None;
            }
            let backing_rec = ctx.get_active_allocation_record(data_raw as usize)?;
            if backing_rec.kind != MirAllocationKind::ArrayBacking
                || !backing_rec.is_ctx_owned
                || backing_rec.word_capacity < needed_backing_words
                || backing_rec.base_ptr == rec.base_ptr
            {
                return None;
            }
            let data_ptr = backing_rec.base_ptr;
            Some(ValidatedArray {
                base_ptr: rec.base_ptr,
                len,
                cap,
                data_ptr,
                header_rec: rec,
                backing_rec: Some(backing_rec),
            })
        }
    }
    #[cfg(not(feature = "std"))]
    {
        let _ = (ctx, ptr);
        None
    }
}

fn validate_active_array_ptr(ctx: &MirContext, ptr: *const i64) -> Option<ValidatedArray> {
    match validate_active_array_inner(ctx, ptr) {
        Some(v) => Some(v),
        None => {
            ctx.last_status
                .store(MIR_STATUS_ERR_OUT_OF_BOUNDS, Ordering::SeqCst);
            None
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_array_len(ptr: *const i64) -> i64 {
    with_current_mir_context(|ctx| {
        validate_active_array_ptr(ctx, ptr).map_or(0, |v| v.len as i64)
    })
}

/// Checked array-length helper for compiled `MirUnaryOp::Len`: returns `-1`
/// (and sets `MIR_STATUS_ERR_OUT_OF_BOUNDS`) on invalid/unowned pointers so
/// callers can detect failure without tripping on stale `last_status` from an
/// earlier independent native invocation.
#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_array_len_checked(ptr: *const i64) -> i64 {
    with_current_mir_context(|ctx| {
        validate_active_array_ptr(ctx, ptr).map_or(-1, |v| v.len as i64)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_array_get(ptr: *const i64, idx: i64) -> i64 {
    with_current_mir_context(|ctx| {
        let Some(v) = validate_active_array_ptr(ctx, ptr) else {
            return 0;
        };
        if idx < 0 || (idx as usize) >= v.len {
            ctx.last_status
                .store(MIR_STATUS_ERR_OUT_OF_BOUNDS, Ordering::SeqCst);
            return 0;
        }
        #[cfg(feature = "std")]
        if is_bool_array_kind(&v.header_rec.kind) {
            return unsafe { u64::from(*(v.data_ptr as *const u8).add(idx as usize)) as i64 };
        }
        unsafe { *v.data_ptr.add(idx as usize) }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_array_set(ptr: *mut i64, idx: i64, val: i64) {
    with_current_mir_context(|ctx| {
        let Some(v) = validate_active_array_ptr(ctx, ptr) else {
            return;
        };
        #[cfg(feature = "std")]
        if v.header_rec.read_only_borrowed {
            ctx.last_status.store(
                crate::effect_abi::LUST_STATUS_ERR_GENERIC as i64,
                Ordering::SeqCst,
            );
            crate::mir::cranelift_backend::set_current_sticky_error(
                "Attempted to mutate borrowed read-only array".into(),
            );
            return;
        }
        if idx < 0 || (idx as usize) >= v.len {
            ctx.last_status
                .store(MIR_STATUS_ERR_OUT_OF_BOUNDS, Ordering::SeqCst);
            return;
        }
        #[cfg(feature = "std")]
        if is_bool_array_kind(&v.header_rec.kind) {
            unsafe {
                *(v.data_ptr as *mut u8).add(idx as usize) = u8::from(val != 0);
            }
            return;
        }
        unsafe {
            *v.data_ptr.add(idx as usize) = val;
        }
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_array_push(ptr: *mut i64, val: i64) -> i64 {
    with_current_mir_context(|ctx| {
        #[cfg(feature = "std")]
        {
            ctx.push_grow_slow_calls
                .set(ctx.push_grow_slow_calls.get().saturating_add(1));
            let Some(v) = validate_active_array_ptr(ctx, ptr) else {
                return MIR_STATUS_ERR_OUT_OF_BOUNDS;
            };
            if v.header_rec.read_only_borrowed {
                ctx.last_status.store(
                    crate::effect_abi::LUST_STATUS_ERR_GENERIC as i64,
                    Ordering::SeqCst,
                );
                crate::mir::cranelift_backend::set_current_sticky_error(
                    "Attempted to mutate borrowed read-only array".into(),
                );
                return crate::effect_abi::LUST_STATUS_ERR_GENERIC as i64;
            }
            if v.header_rec.word_capacity < 3 {
                ctx.last_status
                    .store(MIR_STATUS_ERR_OUT_OF_BOUNDS, Ordering::SeqCst);
                return MIR_STATUS_ERR_OUT_OF_BOUNDS;
            }
            let is_bool = is_bool_array_kind(&v.header_rec.kind);
            if v.len < v.cap {
                unsafe {
                    if is_bool {
                        *(v.data_ptr as *mut u8).add(v.len) = u8::from(val != 0);
                    } else {
                        *v.data_ptr.add(v.len) = val;
                    }
                    *v.base_ptr = (v.len + 1) as i64;
                }
                return MIR_STATUS_OK;
            }
            let Some(new_len) = v.len.checked_add(1) else {
                ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
                return MIR_STATUS_ERR_OOM;
            };
            let new_cap = if is_bool {
                v.cap.saturating_mul(2).max(8).max(new_len)
            } else {
                v.cap.saturating_mul(2).max(4).max(new_len)
            };
            if new_cap > (i64::MAX as usize) {
                ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
                return MIR_STATUS_ERR_OOM;
            }
            let Some(new_word_cap) = array_payload_words_for_cap(new_cap, is_bool) else {
                ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
                return MIR_STATUS_ERR_OOM;
            };
            let Some(bytes_u) = new_word_cap.checked_mul(core::mem::size_of::<i64>()) else {
                ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
                return MIR_STATUS_ERR_OOM;
            };
            let Ok(bytes_needed) = i64::try_from(bytes_u) else {
                ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
                return MIR_STATUS_ERR_OOM;
            };
            // Check memory limit BEFORE allocating or mutating header
            let limit = ctx.max_memory_limit.load(Ordering::SeqCst);
            let prev = ctx.memory_used.fetch_add(bytes_needed, Ordering::SeqCst);
            if prev.saturating_add(bytes_needed) > limit {
                ctx.memory_used.fetch_sub(bytes_needed, Ordering::SeqCst);
                ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
                return MIR_STATUS_ERR_OOM;
            }

            let mut new_vec = Vec::<i64>::new();
            if new_vec.try_reserve_exact(new_word_cap).is_err() {
                ctx.memory_used.fetch_sub(bytes_needed, Ordering::SeqCst);
                ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
                return MIR_STATUS_ERR_OOM;
            }
            if is_bool {
                new_vec.resize(new_word_cap, 0);
                let dst_bytes = new_vec.as_mut_ptr() as *mut u8;
                if v.len > 0 {
                    unsafe {
                        core::ptr::copy_nonoverlapping(v.data_ptr as *const u8, dst_bytes, v.len);
                    }
                }
                unsafe {
                    *dst_bytes.add(v.len) = u8::from(val != 0);
                }
            } else {
                if v.len > 0 {
                    let src = unsafe { core::slice::from_raw_parts(v.data_ptr, v.len) };
                    new_vec.extend_from_slice(src);
                }
                new_vec.push(val);
                new_vec.resize(new_word_cap, 0);
            }
            let new_slice = new_vec.into_boxed_slice();
            let new_data_ptr = Box::into_raw(new_slice) as *mut i64;

            let old_to_free: Option<(*mut i64, usize)> = {
                let seq = ctx.next_alloc_seq.fetch_add(1, Ordering::SeqCst);
                let scope_depth = ctx.scope_alloc_floors.borrow().len();
                let res = {
                    let mut bufs = ctx.allocated_buffers.borrow_mut();
                    let mut records = ctx.allocation_records.borrow_mut();
                    if let Some(old_backing) = v.backing_rec {
                        let old_idx = old_backing.buffer_index;
                        let old_words = old_backing.word_capacity;
                        records.remove(&(old_backing.base_ptr as usize));
                        if old_idx < bufs.len() && bufs[old_idx].0 == old_backing.base_ptr {
                            bufs[old_idx] = (new_data_ptr, new_word_cap);
                            records.insert(
                                new_data_ptr as usize,
                                MirAllocationRecord {
                                    base_ptr: new_data_ptr,
                                    word_capacity: new_word_cap,
                                    kind: MirAllocationKind::ArrayBacking,
                                    buffer_index: old_idx,
                                    handle_epoch: v.header_rec.handle_epoch,
                                    alloc_seq: seq,
                                    is_ctx_owned: true,
                                    read_only_borrowed: false,
                                    registered_scope_depth: scope_depth,
                                },
                            );
                            Some((old_backing.base_ptr, old_words))
                        } else {
                            let buffer_index = bufs.len();
                            bufs.push((new_data_ptr, new_word_cap));
                            records.insert(
                                new_data_ptr as usize,
                                MirAllocationRecord {
                                    base_ptr: new_data_ptr,
                                    word_capacity: new_word_cap,
                                    kind: MirAllocationKind::ArrayBacking,
                                    buffer_index,
                                    handle_epoch: v.header_rec.handle_epoch,
                                    alloc_seq: seq,
                                    is_ctx_owned: true,
                                    read_only_borrowed: false,
                                    registered_scope_depth: scope_depth,
                                },
                            );
                            None
                        }
                    } else {
                        let buffer_index = bufs.len();
                        bufs.push((new_data_ptr, new_word_cap));
                        records.insert(
                            new_data_ptr as usize,
                            MirAllocationRecord {
                                base_ptr: new_data_ptr,
                                word_capacity: new_word_cap,
                                kind: MirAllocationKind::ArrayBacking,
                                buffer_index,
                                handle_epoch: v.header_rec.handle_epoch,
                                alloc_seq: seq,
                                is_ctx_owned: true,
                                read_only_borrowed: false,
                                registered_scope_depth: scope_depth,
                            },
                        );
                        None
                    }
                };
                ctx.mark_dirty_allocation();
                res
            };

            // Commit header update: base_ptr never moves, keeping all aliases valid.
            unsafe {
                *v.base_ptr.add(2) = new_data_ptr as i64;
                *v.base_ptr.add(1) = (new_cap as i64) | i64::MIN;
                *v.base_ptr = new_len as i64;
            }

            if let Some((old_ptr, old_words)) = old_to_free {
                unsafe {
                    let slice = core::ptr::slice_from_raw_parts_mut(old_ptr, old_words);
                    drop(Box::from_raw(slice));
                }
                if let Some(old_bytes_u) = old_words.checked_mul(core::mem::size_of::<i64>()) {
                    if let Ok(old_bytes) = i64::try_from(old_bytes_u) {
                        ctx.memory_used.fetch_sub(old_bytes, Ordering::SeqCst);
                    }
                }
            }

            MIR_STATUS_OK
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (ptr, val);
            ctx.last_status
                .store(MIR_STATUS_ERR_OUT_OF_BOUNDS, Ordering::SeqCst);
            MIR_STATUS_ERR_OUT_OF_BOUNDS
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_div_int(a: i64, b: i64) -> i64 {
    if b == 0 {
        with_current_mir_context(|ctx| {
            ctx.last_status.store(MIR_STATUS_ERR_DIV_BY_ZERO, Ordering::SeqCst);
        });
        return 0;
    }
    if a == i64::MIN && b == -1 {
        i64::MIN
    } else {
        a / b
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_rem_int(a: i64, b: i64) -> i64 {
    if b == 0 {
        with_current_mir_context(|ctx| {
            ctx.last_status.store(MIR_STATUS_ERR_DIV_BY_ZERO, Ordering::SeqCst);
        });
        return 0;
    }
    if a == i64::MIN && b == -1 {
        0
    } else {
        a % b
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_floordiv_int(a: i64, b: i64) -> i64 {
    if b == 0 {
        with_current_mir_context(|ctx| {
            ctx.last_status.store(MIR_STATUS_ERR_DIV_BY_ZERO, Ordering::SeqCst);
        });
        return 0;
    }
    if a == i64::MIN && b == -1 {
        i64::MIN
    } else {
        let d = a / b;
        let rem = a % b;
        if (rem != 0) && ((rem < 0) != (b < 0)) {
            d - 1
        } else {
            d
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_pow_int(mut base: i64, exp: i64) -> i64 {
    if exp < 0 {
        with_current_mir_context(|ctx| {
            ctx.last_status.store(MIR_STATUS_ERR_NEG_EXPONENT, Ordering::SeqCst);
        });
        return 0;
    }
    let mut e = exp as u64;
    let mut res: i64 = 1;
    while e > 0 {
        if e & 1 == 1 {
            res = res.wrapping_mul(base);
        }
        base = base.wrapping_mul(base);
        e >>= 1;
    }
    res
}

fn write_mir_stdout(text: &str) {
    if let Some(vm_ptr) = crate::vm::current_vm_ptr() {
        unsafe {
            (*vm_ptr).console.write_stdout(text.as_bytes());
        }
    } else {
        print!("{text}");
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_print_int(x: i64) {
    write_mir_stdout(&alloc::format!("{}\n", crate::bytecode::Value::Int(x)));
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_print_int_part(x: i64) {
    write_mir_stdout(&alloc::format!("{}", crate::bytecode::Value::Int(x)));
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_print_float_part(x: f64) {
    write_mir_stdout(&alloc::format!("{}", crate::bytecode::Value::Float(x)));
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_print_bool_part(x: i64) {
    write_mir_stdout(&alloc::format!("{}", crate::bytecode::Value::Bool(x != 0)));
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_print_string_part(handle: i64) {
    with_current_mir_context(|ctx| {
        ctx.with_str_handle(handle, write_mir_stdout);
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_print_sep() {
    write_mir_stdout("\t");
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_print_nl() {
    write_mir_stdout("\n");
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_abs_int(x: i64) -> i64 {
    x.saturating_abs()
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_min_int(a: i64, b: i64) -> i64 {
    a.min(b)
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_max_int(a: i64, b: i64) -> i64 {
    a.max(b)
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_clamp_int(x: i64, lo: i64, hi: i64) -> i64 {
    if lo <= hi {
        x.clamp(lo, hi)
    } else {
        x.clamp(hi, lo)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_floordiv_float(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        with_current_mir_context(|ctx| {
            ctx.last_status.store(MIR_STATUS_ERR_DIV_BY_ZERO, Ordering::SeqCst);
        });
        return f64::NAN;
    }
    (a / b).floor()
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_pow_float(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_rem_float(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        with_current_mir_context(|ctx| {
            ctx.last_status.store(MIR_STATUS_ERR_DIV_BY_ZERO, Ordering::SeqCst);
        });
        return f64::NAN;
    }
    a % b
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_string_concat(a_handle: i64, b_handle: i64) -> i64 {
    with_current_mir_context(|ctx| {
        let joined: Option<Result<alloc::string::String, ()>> = ctx.with_str_pair(a_handle, b_handle, |s1, s2| {
            let Some(total_len) = s1.len().checked_add(s2.len()) else {
                ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
                return Err(());
            };
            let Ok(bytes) = i64::try_from(total_len) else {
                ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
                return Err(());
            };
            let limit = ctx.max_memory_limit.load(Ordering::Relaxed);
            let prev = ctx.memory_used.fetch_add(bytes, Ordering::Relaxed);
            if prev.saturating_add(bytes) > limit {
                ctx.memory_used.fetch_sub(bytes, Ordering::Relaxed);
                ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
                return Err(());
            }
            let mut res = alloc::string::String::new();
            if res.try_reserve_exact(total_len).is_err() {
                ctx.memory_used.fetch_sub(bytes, Ordering::Relaxed);
                ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
                return Err(());
            }
            res.push_str(s1);
            res.push_str(s2);
            Ok(res)
        });
        match joined {
            Some(Ok(res)) => ctx.register_handle(crate::bytecode::Value::string(res)),
            _ => 0,
        }
    })
}

/// Appends `rhs_handle` onto `lhs_handle` in place via `Rc::get_mut` when `lhs_handle > 0`
/// (non-interned heap handle in the active `MirContext`), `lhs_handle != rhs_handle`, and
/// `Rc::strong_count == 1`. Enforces `max_memory_limit` BEFORE mutating `lhs` (rolling back
/// `memory_used` and leaving `lhs` unmodified on `MIR_STATUS_ERR_OOM`). Falls back to COW
/// `lust_mir_string_concat` whenever `lhs` is interned, aliased (`strong_count > 1`), or
/// `lhs_handle == rhs_handle`.
#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_string_append_inplace(lhs_handle: i64, rhs_handle: i64) -> i64 {
    if lhs_handle <= 0 || lhs_handle == rhs_handle {
        return lust_mir_string_concat(lhs_handle, rhs_handle);
    }
    #[cfg(feature = "std")]
    {
        enum RhsBuf {
            Static(&'static str),
            Inline([u8; 64], usize),
            Heap(alloc::rc::Rc<alloc::string::String>),
            Owned(alloc::string::String),
        }
        impl RhsBuf {
            #[inline(always)]
            fn as_str(&self) -> &str {
                match self {
                    RhsBuf::Static(s) => s,
                    RhsBuf::Inline(buf, len) => unsafe {
                        core::str::from_utf8_unchecked(&buf[..*len])
                    },
                    RhsBuf::Heap(rc) => rc.as_str(),
                    RhsBuf::Owned(s) => s.as_str(),
                }
            }
        }

        let need_cow_fallback = with_current_mir_context(|ctx| {
            let rhs_buf = if rhs_handle < 0 {
                if let Some(s) = static_str_from_handle(rhs_handle) {
                    RhsBuf::Static(s)
                } else if is_dyn_intern_handle(rhs_handle) {
                    let idx = (DYN_INTERN_BASE - rhs_handle) as usize;
                    let list = ctx.dyn_interned_list.borrow();
                    match list.get(idx) {
                        Some(rc) => RhsBuf::Heap(alloc::rc::Rc::clone(rc)),
                        None => return Some(0),
                    }
                } else {
                    let idx = ((-rhs_handle) - 1) as usize;
                    let mut found: Option<RhsBuf> = None;
                    {
                        let list = ctx.interned_list.borrow();
                        if let Some(s) = list.get(idx) {
                            let b = s.as_bytes();
                            if b.len() <= 64 {
                                let mut arr = [0u8; 64];
                                arr[..b.len()].copy_from_slice(b);
                                found = Some(RhsBuf::Inline(arr, b.len()));
                            } else {
                                found = Some(RhsBuf::Owned(s.clone()));
                            }
                        }
                    }
                    if found.is_none() {
                        DEFAULT_MIR_CONTEXT.with(|d| {
                            let d_list = d.interned_list.borrow();
                            if let Some(s) = d_list.get(idx) {
                                let b = s.as_bytes();
                                if b.len() <= 64 {
                                    let mut arr = [0u8; 64];
                                    arr[..b.len()].copy_from_slice(b);
                                    found = Some(RhsBuf::Inline(arr, b.len()));
                                } else {
                                    found = Some(RhsBuf::Owned(s.clone()));
                                }
                            }
                        });
                    }
                    match found {
                        Some(rb) => rb,
                        None => return Some(0),
                    }
                }
            } else {
                match ctx.get_handle(rhs_handle) {
                    Some(crate::bytecode::Value::String(s)) => RhsBuf::Heap(s),
                    _ => return Some(0),
                }
            };

            let rhs_str = rhs_buf.as_str();
            let lhs_idx = (lhs_handle - 1) as usize;
            let mut hv = ctx.heap_values.borrow_mut();
            if let Some(Some(crate::bytecode::Value::String(lhs_rc))) = hv.get_mut(lhs_idx) {
                if alloc::rc::Rc::strong_count(lhs_rc) == 1 {
                    if let Some(lhs_str) = alloc::rc::Rc::get_mut(lhs_rc) {
                        let Ok(rhs_bytes) = i64::try_from(rhs_str.len()) else {
                            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
                            return Some(0);
                        };
                        let limit = ctx.max_memory_limit.load(Ordering::Relaxed);
                        let prev = ctx.memory_used.fetch_add(rhs_bytes, Ordering::Relaxed);
                        if prev.saturating_add(rhs_bytes) > limit {
                            ctx.memory_used.fetch_sub(rhs_bytes, Ordering::Relaxed);
                            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
                            return Some(0);
                        }
                        if lhs_str.try_reserve(rhs_str.len()).is_err() {
                            ctx.memory_used.fetch_sub(rhs_bytes, Ordering::Relaxed);
                            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
                            return Some(0);
                        }
                        lhs_str.push_str(rhs_str);
                        ctx.mark_dirty_allocation();
                        return Some(lhs_handle);
                    }
                }
            }
            None
        });

        match need_cow_fallback {
            Some(handle) => handle,
            None => lust_mir_string_concat(lhs_handle, rhs_handle),
        }
    }
    #[cfg(not(feature = "std"))]
    {
        lust_mir_string_concat(lhs_handle, rhs_handle)
    }
}

pub(crate) const SMALL_INT_STRING_COUNT: usize = 1024;

pub(crate) fn format_i64_stack(value: i64, buf: &mut [u8; 24]) -> &str {
    let mut magnitude = value.unsigned_abs();
    let mut start = buf.len();
    loop {
        start -= 1;
        buf[start] = b'0' + (magnitude % 10) as u8;
        magnitude /= 10;
        if magnitude == 0 {
            break;
        }
    }
    if value < 0 {
        start -= 1;
        buf[start] = b'-';
    }
    core::str::from_utf8(&buf[start..]).expect("integer formatting produces ASCII")
}

#[cfg(feature = "std")]
pub(crate) enum MirStrSource {
    Static(&'static str),
    Heap(alloc::rc::Rc<alloc::string::String>),
}

#[cfg(feature = "std")]
impl MirStrSource {
    #[inline(always)]
    pub(crate) fn as_str(&self) -> &str {
        match self {
            MirStrSource::Static(s) => s,
            MirStrSource::Heap(rc) => rc.as_str(),
        }
    }
}

#[cfg(feature = "std")]
impl MirContext {
    /// Resolves a string handle into an owned zero-copy handle (`&'static str` or `Rc<String>`)
    /// without holding any `RefCell` borrow afterward, allowing callers to safely mutate
    /// `heap_values` while iterating over the string contents.
    #[inline]
    pub(crate) fn resolve_str_source(&self, handle: i64) -> Option<MirStrSource> {
        if handle < 0 {
            if let Some(s) = static_str_from_handle(handle) {
                return Some(MirStrSource::Static(s));
            }
            if is_dyn_intern_handle(handle) {
                let idx = (DYN_INTERN_BASE - handle) as usize;
                let list = self.dyn_interned_list.borrow();
                return list
                    .get(idx)
                    .map(|rc| MirStrSource::Heap(alloc::rc::Rc::clone(rc)));
            }
            let idx = usize::try_from(handle.unsigned_abs() - 1).ok()?;
            {
                let rc_list = self.interned_rc_list.borrow();
                if let Some(rc) = rc_list.get(idx) {
                    return Some(MirStrSource::Heap(alloc::rc::Rc::clone(rc)));
                }
            }
            return DEFAULT_MIR_CONTEXT.with(|d| {
                let d_rc = d.interned_rc_list.borrow();
                d_rc.get(idx)
                    .map(|rc| MirStrSource::Heap(alloc::rc::Rc::clone(rc)))
            });
        }
        if handle <= 0 {
            return None;
        }
        let idx = (handle - 1) as usize;
        {
            let hv = self.heap_values.borrow();
            if let Some(slot) = hv.get(idx) {
                return match slot {
                    Some(crate::bytecode::Value::String(rc)) => {
                        Some(MirStrSource::Heap(alloc::rc::Rc::clone(rc)))
                    }
                    Some(_) => None,
                    None => {
                        drop(hv);
                        DEFAULT_MIR_CONTEXT.with(|d| {
                            let d_hv = d.heap_values.borrow();
                            match d_hv.get(idx) {
                                Some(Some(crate::bytecode::Value::String(rc))) => {
                                    Some(MirStrSource::Heap(alloc::rc::Rc::clone(rc)))
                                }
                                _ => None,
                            }
                        })
                    }
                };
            }
        }
        DEFAULT_MIR_CONTEXT.with(|d| {
            let d_hv = d.heap_values.borrow();
            match d_hv.get(idx) {
                Some(Some(crate::bytecode::Value::String(rc))) => {
                    Some(MirStrSource::Heap(alloc::rc::Rc::clone(rc)))
                }
                _ => None,
            }
        })
    }
}

pub(crate) fn charge_handle_bytes(ctx: &MirContext, bytes: usize) -> bool {
    if let Ok(bytes) = i64::try_from(bytes)
        && let Some(total) = ctx.memory_used().checked_add(bytes)
        && total <= ctx.max_memory_limit()
    {
        ctx.set_memory_used(total);
        return true;
    }
    ctx.set_status(MIR_STATUS_ERR_OOM);
    false
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_tostring_int(v: i64) -> i64 {
    with_current_mir_context(|ctx| {
        if (0..SMALL_INT_STRING_COUNT as i64).contains(&v) {
            let digits = match v {
                0..=9 => 1,
                10..=99 => 2,
                100..=999 => 3,
                _ => 4,
            };
            if !charge_handle_bytes(ctx, digits) {
                return 0;
            }
            if let Some(h) = ctx.tostring_small_int_handle(v) {
                return h;
            }
        }
        let mut buffer = [0; 24];
        let text = format_i64_stack(v, &mut buffer);
        if !charge_handle_bytes(ctx, text.len()) {
            return 0;
        }
        ctx.register_handle(Value::string(text))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_tostring_float(v: f64) -> i64 {
    with_current_mir_context(|ctx| {
        let s = alloc::format!("{}", crate::bytecode::Value::Float(v));
        let Ok(bytes) = i64::try_from(s.len()) else {
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
            return 0;
        };
        let limit = ctx.max_memory_limit.load(Ordering::Relaxed);
        let prev = ctx.memory_used.fetch_add(bytes, Ordering::Relaxed);
        if prev.saturating_add(bytes) > limit {
            ctx.memory_used.fetch_sub(bytes, Ordering::Relaxed);
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
            return 0;
        }
        ctx.register_handle(crate::bytecode::Value::string(s))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_tostring_bool(v: i64) -> i64 {
    with_current_mir_context(|ctx| {
        let s = if v != 0 { "true" } else { "false" };
        let bytes = s.len() as i64;
        let limit = ctx.max_memory_limit.load(Ordering::Relaxed);
        let prev = ctx.memory_used.fetch_add(bytes, Ordering::Relaxed);
        if prev.saturating_add(bytes) > limit {
            ctx.memory_used.fetch_sub(bytes, Ordering::Relaxed);
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
            return 0;
        }
        ctx.intern_string(s)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_string_eq(a_handle: i64, b_handle: i64) -> i64 {
    // Fast path for static 1-byte ASCII & empty string handles: zero TLS, zero RefCell, zero allocs.
    if is_static_str_handle(a_handle) && is_static_str_handle(b_handle) {
        return i64::from(a_handle == b_handle);
    }
    with_current_mir_context(|ctx| {
        ctx.with_str_pair(a_handle, b_handle, |s1, s2| i64::from(s1 == s2))
            .unwrap_or(0)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_string_len(handle: i64) -> i64 {
    if handle == EMPTY_STR_HANDLE {
        return 0;
    }
    if is_ascii_handle(handle) {
        return 1;
    }
    with_current_mir_context(|ctx| {
        ctx.with_str_handle(handle, |s| s.len() as i64).unwrap_or(0)
    })
}

fn register_string_results(ctx: &MirContext, values: Vec<Value>) -> i64 {
    let bytes = values.iter().fold(
        values.len().saturating_mul(core::mem::size_of::<Value>()),
        |bytes, value| bytes.saturating_add(value.as_string().map_or(0, str::len)),
    );
    if !charge_handle_bytes(ctx, bytes) {
        return 0;
    }
    ctx.register_handle(Value::array(values))
}

// Native helper calls use the active MIR context guard. The leading context
// operand is reserved by the call ABI and is never dereferenced.
pub extern "C" fn lust_mir_string_sub(_vm_ctx: i64, handle: i64, start: i64, end: i64) -> i64 {
    enum SubPiece {
        Static(i64, usize),
        ShortInline([u8; DYN_INTERN_MAX_BYTES], usize),
        Owned(alloc::string::String),
    }
    with_current_mir_context(|ctx| {
        let piece = ctx.with_str_handle(handle, |text| {
            let tbytes = text.as_bytes();
            if start == end && start >= 1 {
                let idx = (start - 1) as usize;
                if idx < tbytes.len() && tbytes[idx] < 128 {
                    return SubPiece::Static(ascii_byte_handle(tbytes[idx]), 1);
                }
            }
            let slice = string_sub_slice(text, start, Some(end));
            let bytes = slice.as_bytes();
            if bytes.is_empty() {
                SubPiece::Static(EMPTY_STR_HANDLE, 0)
            } else if bytes.len() == 1 && bytes[0] < 128 {
                SubPiece::Static(ascii_byte_handle(bytes[0]), 1)
            } else if bytes.len() <= DYN_INTERN_MAX_BYTES {
                let mut buf = [0u8; DYN_INTERN_MAX_BYTES];
                buf[..bytes.len()].copy_from_slice(bytes);
                SubPiece::ShortInline(buf, bytes.len())
            } else {
                SubPiece::Owned(slice.into_owned())
            }
        });
        let Some(piece) = piece else {
            ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
            return 0;
        };
        match piece {
            SubPiece::Static(static_h, len) => {
                if !charge_handle_bytes(ctx, len) {
                    return 0;
                }
                static_h
            }
            SubPiece::ShortInline(buf, len) => {
                if !charge_handle_bytes(ctx, len) {
                    return 0;
                }
                let Ok(s) = core::str::from_utf8(&buf[..len]) else {
                    ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                    return 0;
                };
                ctx.intern_short_str_or_register(s)
            }
            SubPiece::Owned(owned) => {
                if !charge_handle_bytes(ctx, owned.len()) {
                    return 0;
                }
                ctx.register_handle(Value::string(owned))
            }
        }
    })
}

/// Returns an array of byte values. An empty range returns a single nil.
pub extern "C" fn lust_mir_string_byte(_vm_ctx: i64, handle: i64, start: i64, end: i64) -> i64 {
    with_current_mir_context(|ctx| {
        let Some(values) = ctx.with_str_handle(handle, |text| string_byte_values(text, start, Some(end))) else {
            ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
            return 0;
        };
        register_string_results(ctx, values)
    })
}

/// Scalar byte lookup for a call that requires an integer result.
pub extern "C" fn lust_mir_string_byte_at(_vm_ctx: i64, handle: i64, index: i64) -> i64 {
    with_current_mir_context(|ctx| {
        let res = ctx.with_str_handle(handle, |text| {
            let (start, end) = normalize_range(index, Some(index), text.len());
            if start == end {
                None
            } else {
                Some(i64::from(text.as_bytes()[start]))
            }
        });
        match res {
            Some(Some(b)) => b,
            _ => {
                ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                0
            }
        }
    })
}

pub extern "C" fn lust_mir_string_gsub(
    _vm_ctx: i64,
    handle: i64,
    pattern: i64,
    replacement: i64,
) -> i64 {
    with_current_mir_context(|ctx| {
        let res = ctx.with_str_pair(handle, pattern, |text, pat| {
            ctx.with_str_handle(replacement, |repl| string_gsub_fast(text, pat, repl))
        });
        match res.flatten() {
            Some(Ok((output, count))) => {
                register_string_results(ctx, alloc::vec![Value::string(output), Value::Int(count)])
            }
            _ => {
                ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                0
            }
        }
    })
}

/// Returns an array of 1-based endpoints and captures, or a single nil on no match.
pub extern "C" fn lust_mir_string_find(
    _vm_ctx: i64,
    handle: i64,
    pattern: i64,
    start: i64,
    plain: i64,
) -> i64 {
    with_current_mir_context(|ctx| {
        let res = ctx.with_str_pair(handle, pattern, |text, pat| {
            string_find_values(text, pat, start, plain != 0)
        });
        match res {
            Some(Ok(values)) => register_string_results(ctx, values),
            _ => {
                ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                0
            }
        }
    })
}

pub extern "C" fn lust_mir_string_split(_vm_ctx: i64, handle: i64, separator: i64) -> i64 {
    with_current_mir_context(|ctx| {
        let Some(values) = ctx.with_str_pair(handle, separator, string_split_values) else {
            ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
            return 0;
        };
        register_string_results(ctx, values)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_alloc_map() -> i64 {
    with_current_mir_context(|ctx| {
        let bytes = 64;
        let limit = ctx.max_memory_limit.load(Ordering::Relaxed);
        let prev = ctx.memory_used.fetch_add(bytes, Ordering::Relaxed);
        if prev.saturating_add(bytes) > limit {
            ctx.memory_used.fetch_sub(bytes, Ordering::Relaxed);
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::Relaxed);
            return 0;
        }
        ctx.register_handle(crate::bytecode::Value::map(crate::bytecode::LustMap::default()))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_map_get(map_handle: i64, key_handle: i64) -> i64 {
    with_current_mir_context(|ctx| {
        let map_val = match ctx.get_handle(map_handle) {
            Some(crate::bytecode::Value::Map(m)) => m,
            _ => return 0,
        };
        let key_val = match ctx.get_handle(key_handle) {
            Some(v) => v,
            None => crate::bytecode::Value::Int(key_handle),
        };
        let key = crate::bytecode::ValueKey::from(key_val);
        let m = map_val.borrow();
        if let Some(val) = m.get(&key) {
            ctx.register_handle(crate::bytecode::Value::some(val.clone()))
        } else {
            0
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_map_len(map_handle: i64) -> i64 {
    with_current_mir_context(|ctx| {
        match ctx.get_handle(map_handle) {
            Some(crate::bytecode::Value::Map(m)) => m.borrow().len() as i64,
            _ => 0,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_map_set(map_handle: i64, key_handle: i64, val_handle: i64) {
    with_current_mir_context(|ctx| {
        let map_val = match ctx.get_handle(map_handle) {
            Some(crate::bytecode::Value::Map(m)) => m,
            _ => return,
        };
        let key_val = match ctx.get_handle(key_handle) {
            Some(v) => v,
            None => crate::bytecode::Value::Int(key_handle),
        };
        let val = match ctx.get_handle(val_handle) {
            Some(v) => v,
            None => crate::bytecode::Value::Int(val_handle),
        };
        set_map_entry(ctx, &mut map_val.borrow_mut(), ValueKey::from(key_val), val);
    })
}

fn set_map_entry(ctx: &MirContext, map: &mut crate::bytecode::LustMap, key: ValueKey, value: Value) {
    match map.entry(key) {
        Entry::Occupied(mut entry) => {
            entry.insert(value);
        }
        Entry::Vacant(entry) => {
            if charge_handle_bytes(ctx, 32) {
                entry.insert(value);
            }
        }
    }
}

/// Borrowed `&str` key equivalent to `ValueKey(Value::String(_))` for zero-clone `LustMap` lookups.
struct BorrowedStrKey<'a>(&'a str);

impl core::hash::Hash for BorrowedStrKey<'_> {
    #[inline(always)]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        4u8.hash(state);
        self.0.hash(state);
    }
}

impl hashbrown::Equivalent<ValueKey> for BorrowedStrKey<'_> {
    #[inline(always)]
    fn equivalent(&self, key: &ValueKey) -> bool {
        match key.owned_values().1 {
            Value::String(s) => s.as_str() == self.0,
            Value::Enum(obj) if obj.enum_name == "LuaValue" && obj.variant == "String" => {
                obj.values
                    .as_ref()
                    .and_then(|v| v.first())
                    .and_then(|v| v.as_string())
                    == Some(self.0)
            }
            _ => false,
        }
    }
}

#[cfg(feature = "std")]
#[inline]
fn with_map_ref<R>(
    ctx: &MirContext,
    map_handle: i64,
    f: impl FnOnce(&core::cell::RefCell<crate::bytecode::LustMap>) -> R,
) -> Option<R> {
    if map_handle <= 0 {
        return None;
    }
    let idx = (map_handle - 1) as usize;
    {
        let hv = ctx.heap_values.borrow();
        if let Some(slot) = hv.get(idx) {
            return match slot {
                Some(Value::Map(m)) => Some(f(m)),
                Some(_) => None,
                None => {
                    drop(hv);
                    DEFAULT_MIR_CONTEXT.with(|d| {
                        let d_hv = d.heap_values.borrow();
                        match d_hv.get(idx) {
                            Some(Some(Value::Map(m))) => Some(f(m)),
                            _ => None,
                        }
                    })
                }
            };
        }
    }
    DEFAULT_MIR_CONTEXT.with(|d| {
        let d_hv = d.heap_values.borrow();
        match d_hv.get(idx) {
            Some(Some(Value::Map(m))) => Some(f(m)),
            _ => None,
        }
    })
}

fn decode_map_operand(ctx: &MirContext, raw: i64, tag: i64) -> Option<Value> {
    let value = match tag {
        0 => Some(Value::Int(raw)),
        1 => Some(Value::Bool(raw != 0)),
        2 => Some(Value::Float(f64::from_bits(raw as u64))),
        3 => ctx.get_handle(raw),
        _ => None,
    };
    if value.is_none() {
        ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
    }
    value
}

fn with_typed_map<R>(
    map_handle: i64,
    key_raw: i64,
    key_tag: i64,
    access: impl FnOnce(&MirContext, &core::cell::RefCell<crate::bytecode::LustMap>, ValueKey) -> R,
) -> Option<R> {
    with_current_mir_context(|ctx| {
        let Some(Value::Map(map)) = ctx.get_handle(map_handle) else {
            ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
            return None;
        };
        let key = ValueKey::from(decode_map_operand(ctx, key_raw, key_tag)?);
        Some(access(ctx, &map, key))
    })
}

/// Returns MIR status. Operand tags are 0 = int, 1 = bool, 2 = float bits, 3 = handle.
pub extern "C" fn lust_mir_map_set_typed(
    _vm_ctx: i64,
    map_handle: i64,
    key_raw: i64,
    val_raw: i64,
    key_tag: i64,
    val_tag: i64,
) -> i64 {
    #[cfg(feature = "std")]
    {
        with_current_mir_context(|ctx| {
            // Fast path for occupied string-key updates (zero ValueKey/String/Rc clone)
            if key_tag == 3 && val_tag <= 2 {
                let value = match val_tag {
                    0 => Value::Int(val_raw),
                    1 => Value::Bool(val_raw != 0),
                    _ => Value::Float(f64::from_bits(val_raw as u64)),
                };
                if let Some(updated) = with_map_ref(ctx, map_handle, |map| {
                    ctx.with_str_handle(key_raw, |key_str| {
                        let mut m = map.borrow_mut();
                        if let Some(slot) = m.get_mut(&BorrowedStrKey(key_str)) {
                            *slot = value.clone();
                            true
                        } else {
                            false
                        }
                    })
                }) {
                    if updated == Some(true) {
                        return ctx.last_status();
                    }
                }
            }
            // Fast path for int/bool/float keys without cloning Rc<RefCell<LustMap>>
            if key_tag <= 2 && val_tag <= 2 {
                let key = match key_tag {
                    0 => ValueKey::from(Value::Int(key_raw)),
                    1 => ValueKey::from(Value::Bool(key_raw != 0)),
                    _ => ValueKey::from(Value::Float(f64::from_bits(key_raw as u64))),
                };
                let value = match val_tag {
                    0 => Value::Int(val_raw),
                    1 => Value::Bool(val_raw != 0),
                    _ => Value::Float(f64::from_bits(val_raw as u64)),
                };
                if with_map_ref(ctx, map_handle, |map| {
                    set_map_entry(ctx, &mut map.borrow_mut(), key, value);
                })
                .is_some()
                {
                    return ctx.last_status();
                }
                ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                return MIR_STATUS_ERR_OUT_OF_BOUNDS;
            }
            let Some(Value::Map(map)) = ctx.get_handle(map_handle) else {
                ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                return MIR_STATUS_ERR_OUT_OF_BOUNDS;
            };
            let Some(key_val) = decode_map_operand(ctx, key_raw, key_tag) else {
                return MIR_STATUS_ERR_OUT_OF_BOUNDS;
            };
            if let Some(value) = decode_map_operand(ctx, val_raw, val_tag) {
                set_map_entry(ctx, &mut map.borrow_mut(), ValueKey::from(key_val), value);
            }
            ctx.last_status()
        })
    }
    #[cfg(not(feature = "std"))]
    {
        with_typed_map(map_handle, key_raw, key_tag, |ctx, map, key| {
            if let Some(value) = decode_map_operand(ctx, val_raw, val_tag) {
                set_map_entry(ctx, &mut map.borrow_mut(), key, value);
            }
            ctx.last_status()
        })
        .unwrap_or(MIR_STATUS_ERR_OUT_OF_BOUNDS)
    }
}

pub extern "C" fn lust_mir_map_get_typed(
    _vm_ctx: i64,
    map_handle: i64,
    key_raw: i64,
    key_tag: i64,
) -> i64 {
    #[cfg(feature = "std")]
    if key_tag == 3 {
        let found: Option<Option<Value>> = with_current_mir_context(|ctx| {
            with_map_ref(ctx, map_handle, |map| {
                ctx.with_str_handle(key_raw, |key_str| {
                    map.borrow().get(&BorrowedStrKey(key_str)).cloned()
                })
            })
            .flatten()
        });
        if let Some(val_opt) = found {
            return match val_opt {
                Some(v) => with_current_mir_context(|ctx| ctx.register_handle(Value::some(v))),
                None => 0,
            };
        }
    }
    with_typed_map(map_handle, key_raw, key_tag, |ctx, map, key| {
        map.borrow()
            .get(&key)
            .map_or(0, |value| ctx.register_handle(Value::some(value.clone())))
    })
    .unwrap_or(0)
}

pub extern "C" fn lust_mir_map_has_typed(
    _vm_ctx: i64,
    map_handle: i64,
    key_raw: i64,
    key_tag: i64,
) -> i64 {
    #[cfg(feature = "std")]
    {
        if key_tag == 3 {
            let found = with_current_mir_context(|ctx| {
                with_map_ref(ctx, map_handle, |map| {
                    ctx.with_str_handle(key_raw, |key_str| {
                        i64::from(map.borrow().contains_key(&BorrowedStrKey(key_str)))
                    })
                })
                .flatten()
            });
            if let Some(v) = found {
                return v;
            }
        } else if key_tag <= 2 {
            let key = match key_tag {
                0 => ValueKey::from(Value::Int(key_raw)),
                1 => ValueKey::from(Value::Bool(key_raw != 0)),
                _ => ValueKey::from(Value::Float(f64::from_bits(key_raw as u64))),
            };
            return with_current_mir_context(|ctx| {
                match with_map_ref(ctx, map_handle, |map| {
                    i64::from(map.borrow().contains_key(&key))
                }) {
                    Some(v) => v,
                    None => {
                        ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                        0
                    }
                }
            });
        }
    }
    with_typed_map(map_handle, key_raw, key_tag, |_, map, key| {
        i64::from(map.borrow().contains_key(&key))
    })
    .unwrap_or(0)
}

pub extern "C" fn lust_mir_map_get_int_unwrap(
    _vm_ctx: i64,
    map_handle: i64,
    key_raw: i64,
    key_tag: i64,
    default_or_trap: i64,
) -> i64 {
    #[cfg(feature = "std")]
    {
        if key_tag == 3 {
            let fast = with_current_mir_context(|ctx| {
                with_map_ref(ctx, map_handle, |map| {
                    ctx.with_str_handle(key_raw, |key_str| match map.borrow().get(&BorrowedStrKey(key_str)) {
                        Some(Value::Int(value)) => *value,
                        None if default_or_trap != i64::MIN => default_or_trap,
                        _ => {
                            ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                            0
                        }
                    })
                })
                .flatten()
            });
            if let Some(v) = fast {
                return v;
            }
        } else if key_tag <= 2 {
            let key = match key_tag {
                0 => ValueKey::from(Value::Int(key_raw)),
                1 => ValueKey::from(Value::Bool(key_raw != 0)),
                _ => ValueKey::from(Value::Float(f64::from_bits(key_raw as u64))),
            };
            return with_current_mir_context(|ctx| {
                match with_map_ref(ctx, map_handle, |map| match map.borrow().get(&key) {
                    Some(Value::Int(value)) => *value,
                    None if default_or_trap != i64::MIN => default_or_trap,
                    _ => {
                        ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                        0
                    }
                }) {
                    Some(v) => v,
                    None => {
                        ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                        0
                    }
                }
            });
        }
    }
    with_typed_map(map_handle, key_raw, key_tag, |ctx, map, key| {
        match map.borrow().get(&key) {
            Some(Value::Int(value)) => *value,
            None if default_or_trap != i64::MIN => default_or_trap,
            _ => {
                ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                0
            }
        }
    })
    .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_alloc_enum(tag: i64, payload_handle: i64) -> i64 {
    with_current_mir_context(|ctx| {
        let bytes = 48;
        let limit = ctx.max_memory_limit.load(Ordering::SeqCst);
        let prev = ctx.memory_used.fetch_add(bytes, Ordering::SeqCst);
        if prev.saturating_add(bytes) > limit {
            ctx.memory_used.fetch_sub(bytes, Ordering::SeqCst);
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
            return 0;
        }
        let payload = if payload_handle != 0 {
            ctx.get_handle(payload_handle).map(|v| alloc::vec![v])
        } else {
            None
        };
        let enum_obj = crate::bytecode::EnumObject::new("Enum", alloc::format!("Variant{tag}"), payload);
        ctx.register_handle(crate::bytecode::Value::Enum(enum_obj))
    })
}

/// Allocates a numeric-tag enum with up to two raw integer payloads.
pub extern "C" fn lust_mir_alloc_enum2(
    _vm_ctx: i64,
    tag: i64,
    p0: i64,
    p1: i64,
    arity: i64,
) -> i64 {
    with_current_mir_context(|ctx| {
        if !(0..=2).contains(&arity) {
            ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
            return 0;
        }
        let arity = arity as usize;
        if !charge_handle_bytes(ctx, 48 + arity * core::mem::size_of::<Value>()) {
            return 0;
        }
        let raw = [p0, p1];
        let payload = (arity != 0).then(|| raw[..arity].iter().copied().map(Value::Int).collect());
        let object = crate::bytecode::EnumObject::new("Enum", alloc::format!("Variant{tag}"), payload);
        ctx.register_handle(Value::Enum(object))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_enum_tag(enum_handle: i64) -> i64 {
    with_current_mir_context(|ctx| {
        match ctx.get_handle(enum_handle) {
            Some(crate::bytecode::Value::Enum(e)) => {
                if let Some(num_str) = e.variant.as_str().strip_prefix("Variant") {
                    num_str.parse::<i64>().unwrap_or(0)
                } else {
                    match e.variant.as_str() {
                        "None" => 0,
                        "Some" => 1,
                        "Ok" => 0,
                        "Err" => 1,
                        _ => 0,
                    }
                }
            }
            _ => 0,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_enum_payload(enum_handle: i64) -> i64 {
    lust_mir_enum_payload_at(0, enum_handle, 0)
}

/// Reads a zero-based payload field, unboxing primitive values.
pub extern "C" fn lust_mir_enum_payload_at(_vm_ctx: i64, enum_handle: i64, field_idx: i64) -> i64 {
    with_current_mir_context(|ctx| {
        let Some(Value::Enum(object)) = ctx.get_handle(enum_handle) else {
            return 0;
        };
        let Ok(index) = usize::try_from(field_idx) else {
            return 0;
        };
        match object.values.as_ref().and_then(|values| values.get(index)) {
            Some(Value::Int(n)) => *n,
            Some(Value::Bool(b)) => i64::from(*b),
            Some(Value::Float(f)) => f.to_bits() as i64,
            Some(value) => ctx.register_handle(value.clone()),
            None => 0,
        }
    })
}

/// Creates a bounded FIFO integer queue in the active thread-confined `MirContext`.
/// Returns a positive 1-based queue handle, or `0` on invalid capacity / OOM.
pub extern "C" fn lust_mir_queue_new(cap: i64) -> i64 {
    #[cfg(feature = "std")]
    {
        with_current_mir_context(|ctx| {
            if cap < 0 {
                ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                return 0;
            }
            let cap_usize = cap as usize;
            let bytes_est = cap_usize
                .saturating_mul(core::mem::size_of::<i64>())
                .max(64);
            if !charge_handle_bytes(ctx, bytes_est) {
                return 0;
            }
            let mut items = alloc::collections::VecDeque::new();
            if items.try_reserve(cap_usize).is_err() {
                ctx.set_status(MIR_STATUS_ERR_OOM);
                return 0;
            }
            let handle = {
                let mut queues = ctx.native_queues.borrow_mut();
                if queues.try_reserve(1).is_err() {
                    ctx.set_status(MIR_STATUS_ERR_OOM);
                    return 0;
                }
                queues.push(MirIntQueue {
                    cap: cap_usize,
                    items,
                });
                queues.len() as i64
            };
            ctx.mark_dirty_allocation();
            handle
        })
    }
    #[cfg(not(feature = "std"))]
    {
        let _ = cap;
        0
    }
}

/// Non-blocking offer into a bounded FIFO integer queue.
/// Returns `1` if enqueued, `0` if full (or on invalid handle, which sets OOB status).
pub extern "C" fn lust_mir_queue_offer_int(q_handle: i64, val: i64) -> i64 {
    #[cfg(feature = "std")]
    {
        with_current_mir_context(|ctx| {
            if q_handle <= 0 {
                ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                return 0;
            }
            let idx = (q_handle - 1) as usize;
            let mut queues = ctx.native_queues.borrow_mut();
            let Some(q) = queues.get_mut(idx) else {
                ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                return 0;
            };
            if q.cap == 0 || q.items.len() >= q.cap {
                return 0;
            }
            q.items.push_back(val);
            1
        })
    }
    #[cfg(not(feature = "std"))]
    {
        let _ = (q_handle, val);
        0
    }
}

/// Non-blocking take from a bounded FIFO integer queue.
/// Returns the dequeued integer, or sets `MIR_STATUS_ERR_OUT_OF_BOUNDS` and returns `0` if empty/invalid.
pub extern "C" fn lust_mir_queue_take_int(q_handle: i64) -> i64 {
    #[cfg(feature = "std")]
    {
        with_current_mir_context(|ctx| {
            if q_handle <= 0 {
                ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                return 0;
            }
            let idx = (q_handle - 1) as usize;
            let mut queues = ctx.native_queues.borrow_mut();
            let Some(q) = queues.get_mut(idx) else {
                ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                return 0;
            };
            match q.items.pop_front() {
                Some(v) => v,
                None => {
                    ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                    0
                }
            }
        })
    }
    #[cfg(not(feature = "std"))]
    {
        let _ = q_handle;
        0
    }
}

/// Returns `1` iff `handle` references a `Value::Int` or `LuaValue::Int` in the active `MirContext`.
pub extern "C" fn lust_mir_unknown_is_int(handle: i64) -> i64 {
    #[cfg(feature = "std")]
    {
        if handle <= 0 {
            return 0;
        }
        let idx = (handle - 1) as usize;
        with_current_mir_context(|ctx| {
            let check = |val: &Value| -> bool {
                match val {
                    Value::Int(_) => true,
                    Value::Enum(obj) => obj.enum_name == "LuaValue" && obj.variant == "Int",
                    _ => false,
                }
            };
            {
                let hv = ctx.heap_values.borrow();
                if let Some(Some(v)) = hv.get(idx) {
                    return i64::from(check(v));
                }
            }
            DEFAULT_MIR_CONTEXT.with(|d| {
                let d_hv = d.heap_values.borrow();
                match d_hv.get(idx) {
                    Some(Some(v)) => i64::from(check(v)),
                    _ => 0,
                }
            })
        })
    }
    #[cfg(not(feature = "std"))]
    {
        let _ = handle;
        0
    }
}

/// Unwraps the integer payload from a `Value::Int` or `LuaValue::Int` handle in the active `MirContext`.
/// Sets `MIR_STATUS_ERR_OUT_OF_BOUNDS` and returns `0` if `handle` does not hold an integer.
pub extern "C" fn lust_mir_unknown_unwrap_int(handle: i64) -> i64 {
    #[cfg(feature = "std")]
    {
        if handle <= 0 {
            with_current_mir_context(|ctx| ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS));
            return 0;
        }
        let idx = (handle - 1) as usize;
        with_current_mir_context(|ctx| {
            let extract = |val: &Value| -> Option<i64> {
                match val {
                    Value::Int(n) => Some(*n),
                    Value::Enum(obj) if obj.enum_name == "LuaValue" && obj.variant == "Int" => {
                        match obj.values.as_ref().and_then(|vals| vals.first()) {
                            Some(Value::Int(n)) => Some(*n),
                            _ => None,
                        }
                    }
                    _ => None,
                }
            };
            {
                let hv = ctx.heap_values.borrow();
                if let Some(Some(v)) = hv.get(idx) {
                    if let Some(n) = extract(v) {
                        return n;
                    }
                    ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                    return 0;
                }
            }
            let fallback = DEFAULT_MIR_CONTEXT.with(|d| {
                let d_hv = d.heap_values.borrow();
                d_hv.get(idx).and_then(|opt| opt.as_ref()).and_then(extract)
            });
            match fallback {
                Some(n) => n,
                None => {
                    ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                    0
                }
            }
        })
    }
    #[cfg(not(feature = "std"))]
    {
        let _ = handle;
        0
    }
}

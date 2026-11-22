#![allow(clippy::not_unsafe_ptr_arg_deref)]

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

#[cfg(feature = "std")]
use std::sync::Mutex;

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

/// RAII guard restoring the previous thread-local unbounded gas chunk override on drop.
#[cfg(feature = "std")]
pub struct UnboundedGasChunkGuard {
    prev: Option<i64>,
}

#[cfg(feature = "std")]
impl Drop for UnboundedGasChunkGuard {
    fn drop(&mut self) {
        UNBOUNDED_GAS_CHUNK_OVERRIDE.with(|c| c.set(self.prev));
    }
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
    UnboundedGasChunkGuard { prev }
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
    Struct(alloc::string::String),
    CFlatStruct(alloc::string::String),
    RawWords,
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

#[derive(Debug)]
pub struct MirContext {
    #[cfg(feature = "std")]
    gas_fast: GasFastState,
    #[cfg(feature = "std")]
    nested_vm_gas_charged: core::cell::Cell<u64>,
    #[cfg(not(feature = "std"))]
    gas_remaining: AtomicI64,
    pub memory_used: AtomicI64,
    pub last_status: AtomicI64,
    pub max_memory_limit: AtomicI64,
    pub next_alloc_seq: AtomicUsize,
    #[cfg(feature = "std")]
    pub heap_values: Mutex<Vec<Option<crate::bytecode::Value>>>,
    #[cfg(feature = "std")]
    pub allocated_buffers: Mutex<Vec<(*mut i64, usize)>>,
    #[cfg(feature = "std")]
    pub allocation_records: Mutex<hashbrown::HashMap<usize, MirAllocationRecord>>,
    #[cfg(feature = "std")]
    pub scope_baselines: Mutex<Vec<MirContextBaseline>>,
    #[cfg(feature = "std")]
    pub scope_alloc_floors: Mutex<Vec<usize>>,
    /// Scope-bound undo log for borrowed read-only flat array provenance (`PERF-T4-3`).
    /// Never frees the underlying `FlatArrayBuffer` (owned by `Rc<FlatArrayBuffer>`).
    #[cfg(feature = "std")]
    pub borrowed_readonly_stack: Mutex<Vec<(usize, Option<MirAllocationRecord>)>>,
    #[cfg(feature = "std")]
    pub borrowed_scope_floors: Mutex<Vec<usize>>,
    #[cfg(feature = "std")]
    pub interned_strings: Mutex<hashbrown::HashMap<alloc::string::String, i64>>,
    #[cfg(feature = "std")]
    pub interned_list: Mutex<Vec<alloc::string::String>>,
    #[cfg(feature = "std")]
    pub small_digit_handles: [AtomicI64; 10],
    #[cfg(feature = "std")]
    pub struct_fields: Mutex<hashbrown::HashMap<alloc::string::String, Vec<crate::mir::ir::MirType>>>,
    /// Compiler-enforced thread confinement under `std`: makes both `MirContext`
    /// and `Arc<MirContext>` statically `!Send + !Sync` (`PERF-T4-1`, Hazard 7 item 1).
    #[cfg(feature = "std")]
    _not_send_sync: core::marker::PhantomData<*const ()>,
}

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
            nested_vm_gas_charged: core::cell::Cell::new(0),
            memory_used: AtomicI64::new(0),
            last_status: AtomicI64::new(0),
            max_memory_limit: AtomicI64::new(i64::MAX),
            next_alloc_seq: AtomicUsize::new(1),
            heap_values: Mutex::new(Vec::new()),
            allocated_buffers: Mutex::new(Vec::new()),
            allocation_records: Mutex::new(hashbrown::HashMap::new()),
            scope_baselines: Mutex::new(Vec::new()),
            scope_alloc_floors: Mutex::new(Vec::new()),
            borrowed_readonly_stack: Mutex::new(Vec::new()),
            borrowed_scope_floors: Mutex::new(Vec::new()),
            interned_strings: Mutex::new(hashbrown::HashMap::new()),
            interned_list: Mutex::new(Vec::new()),
            small_digit_handles: core::array::from_fn(|_| AtomicI64::new(0)),
            struct_fields: Mutex::new(hashbrown::HashMap::new()),
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
        self.memory_used.load(Ordering::SeqCst)
    }

    pub fn last_status(&self) -> i64 {
        self.last_status.load(Ordering::SeqCst)
    }

    pub fn max_memory_limit(&self) -> i64 {
        self.max_memory_limit.load(Ordering::SeqCst)
    }

    pub fn set_status(&self, status: i64) {
        self.last_status.store(status, Ordering::SeqCst);
    }

    #[inline]
    pub fn set_gas(&self, gas: i64) {
        #[cfg(feature = "std")]
        {
            self.gas_fast.gas_remaining.set(gas);
        }
        #[cfg(not(feature = "std"))]
        {
            self.gas_remaining.store(gas, Ordering::SeqCst);
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
                    .store(MIR_STATUS_ERR_OUT_OF_GAS, Ordering::SeqCst);
            }
            rem_st
        }
        #[cfg(not(feature = "std"))]
        {
            loop {
                let cur = self.gas_remaining.load(Ordering::SeqCst);
                let (rem, overflowed) = cur.overflowing_sub(amt);
                let rem_st = if overflowed { i64::MIN } else { rem };
                if self
                    .gas_remaining
                    .compare_exchange(cur, rem_st, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    if overflowed || rem_st < 0 {
                        self.last_status
                            .store(MIR_STATUS_ERR_OUT_OF_GAS, Ordering::SeqCst);
                    }
                    return rem_st;
                }
            }
        }
    }

    #[cfg(feature = "std")]
    #[inline]
    pub(crate) fn gas_fast_ptr(&self) -> *mut GasFastState {
        // SAFETY: Every field of `GasFastState` is wrapped in `Cell` (`UnsafeCell`),
        // and Rust never forms `&mut GasFastState` or `&mut i64` references.
        core::ptr::addr_of!(self.gas_fast) as *mut GasFastState
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
        self.memory_used.store(bytes, Ordering::SeqCst);
    }

    pub fn set_max_memory_limit(&self, bytes: i64) {
        self.max_memory_limit.store(bytes, Ordering::SeqCst);
    }

    pub fn handle_count(&self) -> usize {
        #[cfg(feature = "std")]
        {
            self.heap_values.lock().unwrap().len()
        }
        #[cfg(not(feature = "std"))]
        {
            0
        }
    }

    pub fn register_handle(&self, val: crate::bytecode::Value) -> i64 {
        #[cfg(feature = "std")]
        {
            let mut hv = self.heap_values.lock().unwrap();
            hv.push(Some(val));
            hv.len() as i64
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = val;
            0
        }
    }

    pub fn get_handle(&self, handle: i64) -> Option<crate::bytecode::Value> {
        #[cfg(feature = "std")]
        {
            if handle < 0 {
                let idx = ((-handle) - 1) as usize;
                let list = self.interned_list.lock().unwrap();
                if let Some(s) = list.get(idx) {
                    return Some(crate::bytecode::Value::string(s.clone()));
                }
                drop(list);
                return DEFAULT_MIR_CONTEXT.with(|d| {
                    let d_list = d.interned_list.lock().unwrap();
                    d_list.get(idx).map(|s| crate::bytecode::Value::string(s.clone()))
                });
            }
            if handle <= 0 {
                return None;
            }
            let hv = self.heap_values.lock().unwrap();
            let idx = (handle - 1) as usize;
            if let Some(v) = hv.get(idx).and_then(|opt| opt.clone()) {
                return Some(v);
            }
            drop(hv);
            DEFAULT_MIR_CONTEXT.with(|d| {
                let d_hv = d.heap_values.lock().unwrap();
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
            let mut map = self.interned_strings.lock().unwrap();
            if let Some(&h) = map.get(s) {
                return h;
            }
            let mut list = self.interned_list.lock().unwrap();
            list.push(alloc::string::ToString::to_string(s));
            let h = -(list.len() as i64);
            map.insert(alloc::string::ToString::to_string(s), h);
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
            let mut sf = self.struct_fields.lock().unwrap();
            sf.insert(alloc::string::ToString::to_string(name), field_types);
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (name, field_types);
        }
    }

    pub fn get_struct_field_type(&self, name: &str, idx: usize) -> Option<crate::mir::ir::MirType> {
        #[cfg(feature = "std")]
        {
            let sf = self.struct_fields.lock().unwrap();
            if let Some(types) = sf.get(name) {
                return types.get(idx).cloned();
            }
            drop(sf);
            DEFAULT_MIR_CONTEXT.with(|d| {
                let d_sf = d.struct_fields.lock().unwrap();
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
            self.allocated_buffers.lock().unwrap().len()
        }
        #[cfg(not(feature = "std"))]
        {
            0
        }
    }

    pub fn capture_baseline(&self) -> MirContextBaseline {
        MirContextBaseline {
            handles: self.handle_count(),
            buffers: self.buffer_count(),
        }
    }

    pub fn active_buffer_baseline(&self) -> usize {
        #[cfg(feature = "std")]
        {
            self.scope_baselines
                .lock()
                .unwrap()
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
                .lock()
                .unwrap()
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
            let scope_depth = self.scope_alloc_floors.lock().unwrap().len();
            let mut bufs = self.allocated_buffers.lock().unwrap();
            let mut records = self.allocation_records.lock().unwrap();
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
            let scope_depth = self.scope_alloc_floors.lock().unwrap().len();
            let bufs = self.allocated_buffers.lock().unwrap();
            let bufs_match = bufs.iter().rposition(|(p, _)| *p == ptr);
            let mut records = self.allocation_records.lock().unwrap();
            if let Some(existing) = records.get_mut(&(ptr as usize)) {
                if existing.read_only_borrowed {
                    return;
                }
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
            let records = self.allocation_records.lock().unwrap();
            records.get(&ptr_usize).cloned()
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
                .lock()
                .unwrap()
                .last()
                .copied()
                .unwrap_or(0);
            let buf_floor = self.active_buffer_baseline();
            let records = self.allocation_records.lock().unwrap();
            records
                .get(&ptr_usize)
                .filter(|rec| {
                    rec.base_ptr as usize == ptr_usize
                        && rec.alloc_seq >= seq_floor
                        && (!rec.is_ctx_owned || rec.buffer_index >= buf_floor)
                })
                .cloned()
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
            let scope_depth = self.scope_alloc_floors.lock().unwrap().len();
            let addr = ptr as usize;
            let mut records = self.allocation_records.lock().unwrap();
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
            self.borrowed_readonly_stack
                .lock()
                .unwrap()
                .push((addr, prev));
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
            let mut records = self.allocation_records.lock().unwrap();
            if let Some(rec) = records.get_mut(&ptr_usize) {
                if rec.read_only_borrowed {
                    return false;
                }
                rec.kind = kind;
                true
            } else {
                false
            }
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
            let (scope_depth, seq_floor) = {
                let floors = self.scope_alloc_floors.lock().unwrap();
                (floors.len(), floors.last().copied().unwrap_or(0))
            };
            let (handle_floor, buf_floor) = {
                let scopes = self.scope_baselines.lock().unwrap();
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
            {
                let mut hv = self.heap_values.lock().unwrap();
                hv.truncate(effective_handles);
            }
            let drained: Vec<(*mut i64, usize)> = {
                let mut bufs = self.allocated_buffers.lock().unwrap();
                let mut records = self.allocation_records.lock().unwrap();
                let start = effective_buffers.min(bufs.len());
                let items: Vec<(*mut i64, usize)> = bufs.drain(start..).collect();
                for &(ptr, _) in &items {
                    if !ptr.is_null() {
                        records.remove(&(ptr as usize));
                    }
                }
                if effective_handles == 0 && start == 0 && scope_depth == 0 {
                    records.clear();
                    self.borrowed_readonly_stack.lock().unwrap().clear();
                } else if scope_depth > 0
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
                items
            };
            for (ptr, cap) in drained {
                if !ptr.is_null() {
                    unsafe {
                        let slice = core::ptr::slice_from_raw_parts_mut(ptr, cap);
                        drop(Box::from_raw(slice));
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
            let scope_buf_floor = self.active_buffer_baseline();
            let buf_baseline = {
                let bufs = self.allocated_buffers.lock().unwrap();
                let floor = scope_buf_floor.min(bufs.len());
                if baseline_handles == 0 && floor == 0 {
                    0
                } else {
                    let records = self.allocation_records.lock().unwrap();
                    let first_after = (floor..bufs.len()).find(|&idx| {
                        let ptr = bufs[idx].0;
                        if ptr.is_null() {
                            return false;
                        }
                        let ptr_addr = ptr as usize;
                        match records.get(&ptr_addr) {
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
                let borrowed_floor = ctx.borrowed_scope_floors.lock().unwrap().pop().unwrap_or(0);
                {
                    let mut stack = ctx.borrowed_readonly_stack.lock().unwrap();
                    let start = borrowed_floor.min(stack.len());
                    let mut records = ctx.allocation_records.lock().unwrap();
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
                ctx.scope_baselines.lock().unwrap().pop();
                ctx.scope_alloc_floors.lock().unwrap().pop();
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
        let baseline = ctx.capture_baseline();
        let seq_floor = ctx.next_alloc_seq.load(Ordering::SeqCst);
        let borrowed_floor = ctx.borrowed_readonly_stack.lock().unwrap().len();
        ctx.scope_baselines.lock().unwrap().push(baseline);
        ctx.scope_alloc_floors.lock().unwrap().push(seq_floor);
        ctx.borrowed_scope_floors.lock().unwrap().push(borrowed_floor);
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
    DEFAULT_MIR_CONTEXT.with(|ctx| ctx.allocated_buffers.lock().unwrap().len())
}

pub fn with_current_mir_context<F, R>(f: F) -> R
where
    F: FnOnce(&MirContext) -> R,
{
    #[cfg(feature = "std")]
    {
        CURRENT_MIR_STACK.with(|stack| {
            if let Some(&top) = stack.borrow().last() {
                let ctx = unsafe { &*top };
                f(ctx)
            } else {
                DEFAULT_MIR_CONTEXT.with(|default_ctx| f(default_ctx))
            }
        })
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
            .store(MIR_STATUS_ERR_OUT_OF_GAS, Ordering::SeqCst);
        MIR_STATUS_ERR_OUT_OF_GAS
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_set_gas(gas: i64) {
    with_current_mir_context(|ctx| {
        ctx.set_gas_unbounded(false);
        ctx.set_gas(gas);
        if gas >= 0 {
            ctx.last_status.store(0, Ordering::SeqCst);
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
    with_current_mir_context(|ctx| ctx.last_status.load(Ordering::SeqCst))
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_set_status(status: i64) {
    with_current_mir_context(|ctx| ctx.last_status.store(status, Ordering::SeqCst));
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_set_max_memory(bytes: i64) {
    with_current_mir_context(|ctx| ctx.max_memory_limit.store(bytes, Ordering::SeqCst));
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_set_memory_limit(bytes: i64) {
    lust_mir_set_max_memory(bytes);
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_get_memory_used() -> i64 {
    with_current_mir_context(|ctx| ctx.memory_used.load(Ordering::SeqCst))
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_track_memory_allocated(bytes: i64) -> i64 {
    with_current_mir_context(|ctx| {
        ctx.memory_used.fetch_add(bytes, Ordering::SeqCst) + bytes
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

fn alloc_mir_buffer_with_kind(len: i64, kind: MirAllocationKind) -> *mut i64 {
    let count = if len < 0 { 0 } else { len as usize };
    // Arrays always allocate at least 3 words (`max(count, 1) + 2`) so word[2] can hold
    // a redirected backing-buffer pointer when growing via `lust_mir_array_push`.
    let payload_slots = if matches!(
        kind,
        MirAllocationKind::Array(_) | MirAllocationKind::RawWords
    ) {
        count.max(1)
    } else {
        count
    };
    let Some(size) = payload_slots.checked_add(2) else {
        with_current_mir_context(|ctx| {
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
        });
        return core::ptr::null_mut();
    };
    let Some(bytes_u) = size.checked_mul(core::mem::size_of::<i64>()) else {
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
    let mut vec = Vec::<i64>::with_capacity(size);
    vec.push(count as i64);
    vec.push(count as i64);
    vec.resize(size, 0);
    let slice = vec.into_boxed_slice();
    let ptr = Box::into_raw(slice) as *mut i64;
    with_current_mir_context(|ctx| {
        #[cfg(feature = "std")]
        {
            let handle_epoch = ctx.handle_count();
            let seq = ctx.next_alloc_seq.fetch_add(1, Ordering::SeqCst);
            let scope_depth = ctx.scope_alloc_floors.lock().unwrap().len();
            let mut bufs = ctx.allocated_buffers.lock().unwrap();
            let mut records = ctx.allocation_records.lock().unwrap();
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
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (ctx, kind);
        }
    });
    ptr
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
    alloc_mir_buffer_with_kind(len, MirAllocationKind::Struct(alloc::string::String::new()))
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
    let Ok(name_str) = core::str::from_utf8(slice) else {
        with_current_mir_context(|ctx| {
            ctx.last_status
                .store(crate::effect_abi::LUST_STATUS_ERR_GENERIC as i64, Ordering::SeqCst);
        });
        return core::ptr::null_mut();
    };
    alloc_mir_buffer_with_kind(
        len,
        MirAllocationKind::Struct(alloc::string::ToString::to_string(name_str)),
    )
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
        if cap_raw >= 0 {
            let cap = cap_raw as usize;
            if len > cap || cap.checked_add(2).is_none_or(|n| n > rec.word_capacity) {
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
                || backing_rec.word_capacity < cap
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
            if v.len < v.cap {
                unsafe {
                    *v.data_ptr.add(v.len) = val;
                    *v.base_ptr = (v.len + 1) as i64;
                }
                return MIR_STATUS_OK;
            }
            let Some(new_len) = v.len.checked_add(1) else {
                ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
                return MIR_STATUS_ERR_OOM;
            };
            let new_cap = v.cap.saturating_mul(2).max(4).max(new_len);
            if new_cap > (i64::MAX as usize) {
                ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
                return MIR_STATUS_ERR_OOM;
            }
            let Some(bytes_u) = new_cap.checked_mul(core::mem::size_of::<i64>()) else {
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
            if new_vec.try_reserve_exact(new_cap).is_err() {
                ctx.memory_used.fetch_sub(bytes_needed, Ordering::SeqCst);
                ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
                return MIR_STATUS_ERR_OOM;
            }
            if v.len > 0 {
                let src = unsafe { core::slice::from_raw_parts(v.data_ptr, v.len) };
                new_vec.extend_from_slice(src);
            }
            new_vec.push(val);
            new_vec.resize(new_cap, 0);
            let new_slice = new_vec.into_boxed_slice();
            let new_data_ptr = Box::into_raw(new_slice) as *mut i64;

            let old_to_free: Option<(*mut i64, usize)> = {
                let seq = ctx.next_alloc_seq.fetch_add(1, Ordering::SeqCst);
                let scope_depth = ctx.scope_alloc_floors.lock().unwrap().len();
                let mut bufs = ctx.allocated_buffers.lock().unwrap();
                let mut records = ctx.allocation_records.lock().unwrap();
                if let Some(old_backing) = v.backing_rec {
                    let old_idx = old_backing.buffer_index;
                    let old_words = old_backing.word_capacity;
                    records.remove(&(old_backing.base_ptr as usize));
                    if old_idx < bufs.len() && bufs[old_idx].0 == old_backing.base_ptr {
                        bufs[old_idx] = (new_data_ptr, new_cap);
                        records.insert(
                            new_data_ptr as usize,
                            MirAllocationRecord {
                                base_ptr: new_data_ptr,
                                word_capacity: new_cap,
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
                        bufs.push((new_data_ptr, new_cap));
                        records.insert(
                            new_data_ptr as usize,
                            MirAllocationRecord {
                                base_ptr: new_data_ptr,
                                word_capacity: new_cap,
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
                    bufs.push((new_data_ptr, new_cap));
                    records.insert(
                        new_data_ptr as usize,
                        MirAllocationRecord {
                            base_ptr: new_data_ptr,
                            word_capacity: new_cap,
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
        if let Some(crate::bytecode::Value::String(s)) = ctx.get_handle(handle) {
            write_mir_stdout(s.as_str());
        }
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
        let s1 = match ctx.get_handle(a_handle) {
            Some(crate::bytecode::Value::String(s)) => s,
            _ => return 0,
        };
        let s2 = match ctx.get_handle(b_handle) {
            Some(crate::bytecode::Value::String(s)) => s,
            _ => return 0,
        };
        let Some(total_len) = s1.len().checked_add(s2.len()) else {
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
            return 0;
        };
        let Ok(bytes) = i64::try_from(total_len) else {
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
            return 0;
        };
        let limit = ctx.max_memory_limit.load(Ordering::SeqCst);
        let prev = ctx.memory_used.fetch_add(bytes, Ordering::SeqCst);
        if prev.saturating_add(bytes) > limit {
            ctx.memory_used.fetch_sub(bytes, Ordering::SeqCst);
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
            return 0;
        }
        let mut res = alloc::string::String::new();
        if res.try_reserve_exact(total_len).is_err() {
            ctx.memory_used.fetch_sub(bytes, Ordering::SeqCst);
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
            return 0;
        }
        res.push_str(s1.as_str());
        res.push_str(s2.as_str());
        ctx.register_handle(crate::bytecode::Value::string(res))
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
            Inline([u8; 64], usize),
            Heap(alloc::rc::Rc<alloc::string::String>),
            Owned(alloc::string::String),
        }
        impl RhsBuf {
            #[inline(always)]
            fn as_str(&self) -> &str {
                match self {
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
                let idx = ((-rhs_handle) - 1) as usize;
                let mut found: Option<RhsBuf> = None;
                {
                    let list = ctx.interned_list.lock().unwrap();
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
                        let d_list = d.interned_list.lock().unwrap();
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
            } else {
                match ctx.get_handle(rhs_handle) {
                    Some(crate::bytecode::Value::String(s)) => RhsBuf::Heap(s),
                    _ => return Some(0),
                }
            };

            let rhs_str = rhs_buf.as_str();
            let lhs_idx = (lhs_handle - 1) as usize;
            let mut hv = ctx.heap_values.lock().unwrap();
            if let Some(Some(crate::bytecode::Value::String(lhs_rc))) = hv.get_mut(lhs_idx) {
                if alloc::rc::Rc::strong_count(lhs_rc) == 1 {
                    if let Some(lhs_str) = alloc::rc::Rc::get_mut(lhs_rc) {
                        let Ok(rhs_bytes) = i64::try_from(rhs_str.len()) else {
                            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
                            return Some(0);
                        };
                        let limit = ctx.max_memory_limit.load(Ordering::SeqCst);
                        let prev = ctx.memory_used.fetch_add(rhs_bytes, Ordering::SeqCst);
                        if prev.saturating_add(rhs_bytes) > limit {
                            ctx.memory_used.fetch_sub(rhs_bytes, Ordering::SeqCst);
                            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
                            return Some(0);
                        }
                        if lhs_str.try_reserve(rhs_str.len()).is_err() {
                            ctx.memory_used.fetch_sub(rhs_bytes, Ordering::SeqCst);
                            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
                            return Some(0);
                        }
                        lhs_str.push_str(rhs_str);
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

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_tostring_int(v: i64) -> i64 {
    with_current_mir_context(|ctx| {
        #[cfg(feature = "std")]
        if (0..10).contains(&v) {
            const DIGITS: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
            let bytes: i64 = 1;
            let limit = ctx.max_memory_limit.load(Ordering::SeqCst);
            let prev = ctx.memory_used.fetch_add(bytes, Ordering::SeqCst);
            if prev.saturating_add(bytes) > limit {
                ctx.memory_used.fetch_sub(bytes, Ordering::SeqCst);
                ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
                return 0;
            }
            let idx = v as usize;
            let cached = ctx.small_digit_handles[idx].load(Ordering::Relaxed);
            if cached < 0 {
                return cached;
            }
            let h = ctx.intern_string(DIGITS[idx]);
            ctx.small_digit_handles[idx].store(h, Ordering::Relaxed);
            return h;
        }
        let s = alloc::format!("{}", crate::bytecode::Value::Int(v));
        let Ok(bytes) = i64::try_from(s.len()) else {
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
            return 0;
        };
        let limit = ctx.max_memory_limit.load(Ordering::SeqCst);
        let prev = ctx.memory_used.fetch_add(bytes, Ordering::SeqCst);
        if prev.saturating_add(bytes) > limit {
            ctx.memory_used.fetch_sub(bytes, Ordering::SeqCst);
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
            return 0;
        }
        ctx.register_handle(crate::bytecode::Value::string(s))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_tostring_float(v: f64) -> i64 {
    with_current_mir_context(|ctx| {
        let s = alloc::format!("{}", crate::bytecode::Value::Float(v));
        let Ok(bytes) = i64::try_from(s.len()) else {
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
            return 0;
        };
        let limit = ctx.max_memory_limit.load(Ordering::SeqCst);
        let prev = ctx.memory_used.fetch_add(bytes, Ordering::SeqCst);
        if prev.saturating_add(bytes) > limit {
            ctx.memory_used.fetch_sub(bytes, Ordering::SeqCst);
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
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
        let limit = ctx.max_memory_limit.load(Ordering::SeqCst);
        let prev = ctx.memory_used.fetch_add(bytes, Ordering::SeqCst);
        if prev.saturating_add(bytes) > limit {
            ctx.memory_used.fetch_sub(bytes, Ordering::SeqCst);
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
            return 0;
        }
        ctx.intern_string(s)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_string_eq(a_handle: i64, b_handle: i64) -> i64 {
    with_current_mir_context(|ctx| {
        let s1 = match ctx.get_handle(a_handle) {
            Some(crate::bytecode::Value::String(s)) => s,
            _ => return 0,
        };
        let s2 = match ctx.get_handle(b_handle) {
            Some(crate::bytecode::Value::String(s)) => s,
            _ => return 0,
        };
        if *s1 == *s2 { 1 } else { 0 }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_string_len(handle: i64) -> i64 {
    with_current_mir_context(|ctx| {
        match ctx.get_handle(handle) {
            Some(crate::bytecode::Value::String(s)) => s.len() as i64,
            _ => 0,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_alloc_map() -> i64 {
    with_current_mir_context(|ctx| {
        let bytes = 64;
        let limit = ctx.max_memory_limit.load(Ordering::SeqCst);
        let prev = ctx.memory_used.fetch_add(bytes, Ordering::SeqCst);
        if prev.saturating_add(bytes) > limit {
            ctx.memory_used.fetch_sub(bytes, Ordering::SeqCst);
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
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
        let bytes = 32;
        let limit = ctx.max_memory_limit.load(Ordering::SeqCst);
        let prev = ctx.memory_used.fetch_add(bytes, Ordering::SeqCst);
        if prev.saturating_add(bytes) > limit {
            ctx.memory_used.fetch_sub(bytes, Ordering::SeqCst);
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
            return;
        }
        let key = crate::bytecode::ValueKey::from(key_val);
        map_val.borrow_mut().insert(key, val);
    })
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
    with_current_mir_context(|ctx| {
        match ctx.get_handle(enum_handle) {
            Some(crate::bytecode::Value::Enum(e)) => {
                if let Some(first) = e.values.as_ref().and_then(|v| v.first()) {
                    match first {
                        crate::bytecode::Value::Int(n) => *n,
                        crate::bytecode::Value::Bool(b) => i64::from(*b),
                        crate::bytecode::Value::Float(f) => f.to_bits() as i64,
                        _ => ctx.register_handle(first.clone()),
                    }
                } else {
                    0
                }
            }
            _ => 0,
        }
    })
}

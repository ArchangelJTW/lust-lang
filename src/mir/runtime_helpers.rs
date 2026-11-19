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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MirAllocationKind {
    Array(crate::mir::ir::MirType),
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
    pub registered_scope_depth: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MirContextBaseline {
    pub handles: usize,
    pub buffers: usize,
}

#[derive(Debug)]
pub struct MirContext {
    pub gas_remaining: AtomicI64,
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
    #[cfg(feature = "std")]
    pub interned_strings: Mutex<hashbrown::HashMap<alloc::string::String, i64>>,
    #[cfg(feature = "std")]
    pub interned_list: Mutex<Vec<alloc::string::String>>,
    #[cfg(feature = "std")]
    pub struct_fields: Mutex<hashbrown::HashMap<alloc::string::String, Vec<crate::mir::ir::MirType>>>,
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
    pub fn new() -> Self {
        Self {
            gas_remaining: AtomicI64::new(1_000_000_000),
            memory_used: AtomicI64::new(0),
            last_status: AtomicI64::new(0),
            max_memory_limit: AtomicI64::new(i64::MAX),
            next_alloc_seq: AtomicUsize::new(1),
            #[cfg(feature = "std")]
            heap_values: Mutex::new(Vec::new()),
            #[cfg(feature = "std")]
            allocated_buffers: Mutex::new(Vec::new()),
            #[cfg(feature = "std")]
            allocation_records: Mutex::new(hashbrown::HashMap::new()),
            #[cfg(feature = "std")]
            scope_baselines: Mutex::new(Vec::new()),
            #[cfg(feature = "std")]
            scope_alloc_floors: Mutex::new(Vec::new()),
            #[cfg(feature = "std")]
            interned_strings: Mutex::new(hashbrown::HashMap::new()),
            #[cfg(feature = "std")]
            interned_list: Mutex::new(Vec::new()),
            #[cfg(feature = "std")]
            struct_fields: Mutex::new(hashbrown::HashMap::new()),
        }
    }

    pub fn with_gas(gas: i64) -> Self {
        let ctx = Self::new();
        ctx.gas_remaining.store(gas, Ordering::SeqCst);
        ctx
    }

    pub fn enter(&self) -> MirContextGuard {
        enter_mir_context(self)
    }

    pub fn gas_remaining(&self) -> i64 {
        self.gas_remaining.load(Ordering::SeqCst)
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

    pub fn set_gas(&self, gas: i64) {
        self.gas_remaining.store(gas, Ordering::SeqCst);
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

    pub fn set_allocation_kind(&self, ptr_usize: usize, kind: MirAllocationKind) -> bool {
        #[cfg(feature = "std")]
        {
            let mut records = self.allocation_records.lock().unwrap();
            if let Some(rec) = records.get_mut(&ptr_usize) {
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
                } else if scope_depth > 0
                    && baseline_handles <= handle_floor
                    && baseline_buffers <= buf_floor
                {
                    records.retain(|_, rec| {
                        rec.is_ctx_owned
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
                        let ptr_addr = bufs[idx].0 as usize;
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
}

pub struct MirContextGuard(*const MirContext);

impl Drop for MirContextGuard {
    fn drop(&mut self) {
        #[cfg(feature = "std")]
        {
            if !self.0.is_null() {
                let ctx = unsafe { &*self.0 };
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
        ctx.scope_baselines.lock().unwrap().push(baseline);
        ctx.scope_alloc_floors.lock().unwrap().push(seq_floor);
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
pub extern "C" fn lust_mir_set_gas(gas: i64) {
    with_current_mir_context(|ctx| {
        ctx.gas_remaining.store(gas, Ordering::SeqCst);
        if gas >= 0 {
            ctx.last_status.store(0, Ordering::SeqCst);
        }
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_get_gas() -> i64 {
    with_current_mir_context(|ctx| ctx.gas_remaining.load(Ordering::SeqCst))
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_charge_gas(amt: i64) -> i64 {
    with_current_mir_context(|ctx| {
        let cur = ctx.gas_remaining.fetch_sub(amt, Ordering::SeqCst);
        let rem = cur - amt;
        if rem < 0 {
            ctx.last_status.store(crate::effect_abi::LUST_STATUS_ERR_OUT_OF_GAS as i64, Ordering::SeqCst);
        }
        rem
    })
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
    let Some(size) = count.checked_add(2) else {
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

fn validate_active_array_ptr(ctx: &MirContext, ptr: *const i64) -> Option<(*mut i64, usize)> {
    if ptr.is_null() {
        ctx.last_status
            .store(MIR_STATUS_ERR_OUT_OF_BOUNDS, Ordering::SeqCst);
        return None;
    }
    #[cfg(feature = "std")]
    {
        let Some(rec) = ctx.get_active_allocation_record(ptr as usize) else {
            ctx.last_status
                .store(MIR_STATUS_ERR_OUT_OF_BOUNDS, Ordering::SeqCst);
            return None;
        };
        if !matches!(
            rec.kind,
            MirAllocationKind::Array(_) | MirAllocationKind::RawWords
        ) {
            ctx.last_status
                .store(MIR_STATUS_ERR_OUT_OF_BOUNDS, Ordering::SeqCst);
            return None;
        }
        if rec.word_capacity < 2 {
            ctx.last_status
                .store(MIR_STATUS_ERR_OUT_OF_BOUNDS, Ordering::SeqCst);
            return None;
        }
        let len_raw = unsafe { *rec.base_ptr };
        let cap_raw = unsafe { *rec.base_ptr.add(1) };
        if len_raw < 0 || cap_raw < 0 {
            ctx.last_status
                .store(MIR_STATUS_ERR_OUT_OF_BOUNDS, Ordering::SeqCst);
            return None;
        }
        let len = len_raw as usize;
        let cap = cap_raw as usize;
        if len > cap || cap.checked_add(2).is_none_or(|n| n > rec.word_capacity) {
            ctx.last_status
                .store(MIR_STATUS_ERR_OUT_OF_BOUNDS, Ordering::SeqCst);
            return None;
        }
        Some((rec.base_ptr, len))
    }
    #[cfg(not(feature = "std"))]
    {
        let _ = ptr;
        ctx.last_status
            .store(MIR_STATUS_ERR_OUT_OF_BOUNDS, Ordering::SeqCst);
        None
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_array_len(ptr: *const i64) -> i64 {
    with_current_mir_context(|ctx| {
        validate_active_array_ptr(ctx, ptr).map_or(0, |(_, len)| len as i64)
    })
}

/// Checked array-length helper for compiled `MirUnaryOp::Len`: returns `-1`
/// (and sets `MIR_STATUS_ERR_OUT_OF_BOUNDS`) on invalid/unowned pointers so
/// callers can detect failure without tripping on stale `last_status` from an
/// earlier independent native invocation.
#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_array_len_checked(ptr: *const i64) -> i64 {
    with_current_mir_context(|ctx| {
        validate_active_array_ptr(ctx, ptr).map_or(-1, |(_, len)| len as i64)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_array_get(ptr: *const i64, idx: i64) -> i64 {
    with_current_mir_context(|ctx| {
        let Some((base, len)) = validate_active_array_ptr(ctx, ptr) else {
            return 0;
        };
        if idx < 0 || (idx as usize) >= len {
            ctx.last_status
                .store(MIR_STATUS_ERR_OUT_OF_BOUNDS, Ordering::SeqCst);
            return 0;
        }
        unsafe { *base.add(2 + idx as usize) }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_array_set(ptr: *mut i64, idx: i64, val: i64) {
    with_current_mir_context(|ctx| {
        let Some((base, len)) = validate_active_array_ptr(ctx, ptr) else {
            return;
        };
        if idx < 0 || (idx as usize) >= len {
            ctx.last_status
                .store(MIR_STATUS_ERR_OUT_OF_BOUNDS, Ordering::SeqCst);
            return;
        }
        unsafe {
            *base.add(2 + idx as usize) = val;
        }
    });
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
        let res = alloc::format!("{}{}", *s1, *s2);
        let bytes = res.len() as i64;
        let limit = ctx.max_memory_limit.load(Ordering::SeqCst);
        let prev = ctx.memory_used.fetch_add(bytes, Ordering::SeqCst);
        if prev.saturating_add(bytes) > limit {
            ctx.memory_used.fetch_sub(bytes, Ordering::SeqCst);
            ctx.last_status.store(MIR_STATUS_ERR_OOM, Ordering::SeqCst);
            return 0;
        }
        ctx.register_handle(crate::bytecode::Value::string(res))
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

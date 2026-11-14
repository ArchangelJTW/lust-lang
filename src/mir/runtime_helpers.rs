#![allow(clippy::not_unsafe_ptr_arg_deref)]

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicI64, Ordering};

#[cfg(feature = "std")]
use std::sync::Mutex;

pub const MIR_STATUS_OK: i64 = crate::effect_abi::LUST_STATUS_SUCCESS as i64;
pub const MIR_STATUS_ERR_OUT_OF_GAS: i64 = crate::effect_abi::LUST_STATUS_ERR_OUT_OF_GAS as i64;
pub const MIR_STATUS_ERR_OOM: i64 = crate::effect_abi::LUST_STATUS_ERR_OUT_OF_MEMORY as i64;
pub const MIR_STATUS_ERR_DIV_BY_ZERO: i64 = 5;
pub const MIR_STATUS_ERR_NEG_EXPONENT: i64 = 6;

#[derive(Debug)]
pub struct MirContext {
    pub gas_remaining: AtomicI64,
    pub memory_used: AtomicI64,
    pub last_status: AtomicI64,
    pub max_memory_limit: AtomicI64,
    #[cfg(feature = "std")]
    pub heap_values: Mutex<Vec<Option<crate::bytecode::Value>>>,
    #[cfg(feature = "std")]
    pub allocated_buffers: Mutex<Vec<(*mut i64, usize)>>,
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

impl MirContext {
    pub fn new() -> Self {
        Self {
            gas_remaining: AtomicI64::new(1_000_000_000),
            memory_used: AtomicI64::new(0),
            last_status: AtomicI64::new(0),
            max_memory_limit: AtomicI64::new(i64::MAX),
            #[cfg(feature = "std")]
            heap_values: Mutex::new(Vec::new()),
            #[cfg(feature = "std")]
            allocated_buffers: Mutex::new(Vec::new()),
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

    pub fn clear_to_baseline(&self, baseline: usize) {
        #[cfg(feature = "std")]
        {
            let mut hv = self.heap_values.lock().unwrap();
            hv.truncate(baseline);
            let mut bufs = self.allocated_buffers.lock().unwrap();
            for (ptr, cap) in bufs.drain(..) {
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
            let _ = baseline;
        }
    }
}

pub struct MirContextGuard(*const MirContext);

impl Drop for MirContextGuard {
    fn drop(&mut self) {
        #[cfg(feature = "std")]
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

pub fn enter_mir_context(ctx: &MirContext) -> MirContextGuard {
    let ptr = ctx as *const MirContext;
    #[cfg(feature = "std")]
    CURRENT_MIR_STACK.with(|stack| {
        stack.borrow_mut().push(ptr);
    });
    MirContextGuard(ptr)
}

#[cfg(feature = "std")]
thread_local! {
    static CURRENT_MIR_STACK: core::cell::RefCell<Vec<*const MirContext>> = const { core::cell::RefCell::new(Vec::new()) };
    static DEFAULT_MIR_CONTEXT: MirContext = MirContext::new();
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

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_alloc_array(len: i64) -> *mut i64 {
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
        ctx.allocated_buffers.lock().unwrap().push((ptr, size));
    });
    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_array_len(ptr: *const i64) -> i64 {
    if ptr.is_null() {
        return 0;
    }
    unsafe { *ptr }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_array_get(ptr: *const i64, idx: i64) -> i64 {
    if ptr.is_null() {
        return 0;
    }
    let len = unsafe { *ptr };
    if idx < 0 || idx >= len {
        return 0;
    }
    unsafe { *ptr.offset(2 + idx as isize) }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_array_set(ptr: *mut i64, idx: i64, val: i64) {
    if ptr.is_null() {
        return;
    }
    let len = unsafe { *ptr };
    if idx < 0 || idx >= len {
        return;
    }
    unsafe {
        *ptr.offset(2 + idx as isize) = val;
    }
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

#[unsafe(no_mangle)]
pub extern "C" fn lust_mir_print_int(x: i64) {
    println!("{x}");
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

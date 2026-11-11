#![allow(clippy::not_unsafe_ptr_arg_deref)]

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicI64, Ordering};

#[derive(Debug)]
pub struct MirContext {
    pub gas_remaining: AtomicI64,
    pub memory_used: AtomicI64,
    pub last_status: AtomicI64,
    pub max_memory_limit: AtomicI64,
}

impl Default for MirContext {
    fn default() -> Self {
        Self::new()
    }
}

impl MirContext {
    pub const fn new() -> Self {
        Self {
            gas_remaining: AtomicI64::new(1_000_000_000),
            memory_used: AtomicI64::new(0),
            last_status: AtomicI64::new(0),
            max_memory_limit: AtomicI64::new(i64::MAX),
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
    let size = (count + 2).max(2);
    let bytes_needed = (size * core::mem::size_of::<i64>()) as i64;
    with_current_mir_context(|ctx| {
        let limit = ctx.max_memory_limit.load(Ordering::SeqCst);
        let prev = ctx.memory_used.fetch_add(bytes_needed, Ordering::SeqCst);
        if prev + bytes_needed > limit {
            ctx.memory_used.fetch_sub(bytes_needed, Ordering::SeqCst);
            ctx.last_status.store(2, Ordering::SeqCst);
            panic!("Out of memory");
        }
    });
    let mut vec = Vec::<i64>::with_capacity(size);
    vec.push(count as i64);
    vec.push(count as i64);
    vec.resize(size, 0);
    let slice = vec.into_boxed_slice();
    Box::into_raw(slice) as *mut i64
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
        panic!("Division by zero");
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
        panic!("Modulo by zero");
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
        panic!("Division by zero");
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
        panic!("Negative exponent in integer power");
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

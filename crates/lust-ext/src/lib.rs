#[path = "../../lust-ext-abi/src/lib.rs"]
#[cfg(test)]
pub mod lust_ext_abi;

#[cfg(not(test))]
use lust_ext_abi;

pub use lust_ext_abi::*;
pub use lust_macros::{lust_fn, lust_module, LustStructView};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LustExtError {
    OutOfGas,
    OutOfMemory,
    Interrupted(u32),
    InvalidHandle,
    StaleHandle,
    CapabilityDenied,
    PanicDefect,
    Generic(i32),
}

impl core::fmt::Display for LustExtError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfGas => write!(f, "Out of gas"),
            Self::OutOfMemory => write!(f, "Out of memory"),
            Self::Interrupted(r) => write!(f, "Interrupted (reason: {})", r),
            Self::InvalidHandle => write!(f, "Invalid handle"),
            Self::StaleHandle => write!(f, "Stale handle"),
            Self::CapabilityDenied => write!(f, "Capability denied"),
            Self::PanicDefect => write!(f, "Panic defect"),
            Self::Generic(c) => write!(f, "Extension error ({})", c),
        }
    }
}

impl std::error::Error for LustExtError {}

pub struct HostCtx<'a> {
    host: *const LustExtHostV1,
    ctx_ptr: *mut core::ffi::c_void,
    _marker: core::marker::PhantomData<&'a ()>,
}

impl<'a> HostCtx<'a> {
    /// Constructs a HostCtx from raw host pointers.
    ///
    /// # Safety
    /// `host` must be a valid pointer to `LustExtHostV1` and `ctx_ptr` must be valid for the duration of `'a`.
    pub unsafe fn from_raw(host: *const LustExtHostV1, ctx_ptr: *mut core::ffi::c_void) -> Self {
        Self {
            host,
            ctx_ptr,
            _marker: core::marker::PhantomData,
        }
    }

    pub fn charge_gas(&mut self, units: u64) -> Result<(), LustExtError> {
        if self.host.is_null() {
            return Ok(());
        }
        let rc = unsafe { ((*self.host).charge_gas)(self.ctx_ptr, units) };
        if rc == LUST_EXT_STATUS_OK {
            Ok(())
        } else if rc == LUST_EXT_STATUS_ERR_OUT_OF_GAS {
            Err(LustExtError::OutOfGas)
        } else {
            Err(LustExtError::Generic(rc))
        }
    }

    pub fn check_interrupt(&self) -> Result<(), LustExtError> {
        if self.host.is_null() {
            return Ok(());
        }
        let mut reason: u32 = 0;
        let rc = unsafe { ((*self.host).check_interrupt)(self.ctx_ptr, &mut reason) };
        if rc == LUST_EXT_STATUS_OK {
            Ok(())
        } else {
            Err(LustExtError::Interrupted(reason))
        }
    }

    pub fn try_reserve_memory(&mut self, bytes: usize) -> Result<MemoryReservation<'_>, LustExtError> {
        if self.host.is_null() {
            return Ok(MemoryReservation {
                host: core::ptr::null(),
                ctx_ptr: core::ptr::null_mut(),
                reservation_id: 0,
                reserved_bytes: bytes,
                committed: false,
                _marker: core::marker::PhantomData,
            });
        }
        let mut res_id: u64 = 0;
        let rc = unsafe { ((*self.host).try_reserve_memory)(self.ctx_ptr, bytes, &mut res_id) };
        if rc == LUST_EXT_STATUS_OK {
            Ok(MemoryReservation {
                host: self.host,
                ctx_ptr: self.ctx_ptr,
                reservation_id: res_id,
                reserved_bytes: bytes,
                committed: false,
                _marker: core::marker::PhantomData,
            })
        } else {
            Err(LustExtError::OutOfMemory)
        }
    }

    pub fn acquire_resource<T: 'static>(
        &mut self,
        scope_id: Option<u64>,
        resource: T,
    ) -> Result<OpaqueHandle<T>, LustExtError> {
        let boxed = Box::new(resource);
        let raw_ptr = Box::into_raw(boxed) as *mut core::ffi::c_void;

        unsafe extern "C" fn cleanup_box<T>(ptr: *mut core::ffi::c_void) {
            if !ptr.is_null() {
                unsafe {
                    drop(Box::from_raw(ptr as *mut T));
                }
            }
        }

        if self.host.is_null() {
            let token = raw_ptr as usize as u64;
            return Ok(OpaqueHandle {
                token,
                _marker: core::marker::PhantomData,
            });
        }

        let mut token: u64 = 0;
        let rc = unsafe {
            ((*self.host).acquire_resource)(
                self.ctx_ptr,
                scope_id.unwrap_or(0),
                raw_ptr,
                cleanup_box::<T>,
                &mut token,
            )
        };

        if rc == LUST_EXT_STATUS_OK {
            Ok(OpaqueHandle {
                token,
                _marker: core::marker::PhantomData,
            })
        } else {
            unsafe {
                drop(Box::from_raw(raw_ptr as *mut T));
            }
            Err(LustExtError::Generic(rc))
        }
    }

    pub fn release_resource(&mut self, token: u64) -> Result<(), LustExtError> {
        if self.host.is_null() {
            return Ok(());
        }
        let rc = unsafe { ((*self.host).release_resource)(self.ctx_ptr, token) };
        if rc == LUST_EXT_STATUS_OK {
            Ok(())
        } else {
            Err(LustExtError::InvalidHandle)
        }
    }
}

pub struct MemoryReservation<'a> {
    host: *const LustExtHostV1,
    ctx_ptr: *mut core::ffi::c_void,
    reservation_id: u64,
    #[allow(dead_code)]
    reserved_bytes: usize,
    committed: bool,
    _marker: core::marker::PhantomData<&'a ()>,
}

impl<'a> MemoryReservation<'a> {
    pub fn commit(mut self, actual_bytes: usize) -> Result<(), LustExtError> {
        self.committed = true;
        if self.host.is_null() {
            return Ok(());
        }
        let rc = unsafe {
            ((*self.host).commit_memory)(self.ctx_ptr, self.reservation_id, actual_bytes)
        };
        if rc == LUST_EXT_STATUS_OK {
            Ok(())
        } else {
            Err(LustExtError::OutOfMemory)
        }
    }
}

impl<'a> Drop for MemoryReservation<'a> {
    fn drop(&mut self) {
        if !self.committed && !self.host.is_null() {
            unsafe {
                let _ = ((*self.host).rollback_memory)(self.ctx_ptr, self.reservation_id);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OpaqueHandle<T> {
    token: u64,
    _marker: core::marker::PhantomData<T>,
}

impl<T> OpaqueHandle<T> {
    pub fn token(&self) -> u64 {
        self.token
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LustBytes<'a> {
    slice: &'a [u8],
}

impl<'a> LustBytes<'a> {
    pub fn from_slice(slice: &'a [u8]) -> Self {
        Self { slice }
    }

    pub fn as_slice(&self) -> &'a [u8] {
        self.slice
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LustStr<'a> {
    s: &'a str,
}

impl<'a> LustStr<'a> {
    pub fn from_str(s: &'a str) -> Self {
        Self { s }
    }

    pub fn as_str(&self) -> &'a str {
        self.s
    }
}

pub mod prelude {
    pub use super::{
        HostCtx, LustBytes, LustExtError, LustStr, MemoryReservation, OpaqueHandle,
    };
    pub use super::lust_ext_abi::*;
    pub use lust_macros::{lust_fn, lust_module, LustStructView};
}

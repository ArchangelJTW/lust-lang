#![cfg_attr(not(test), no_std)]

pub const LUST_EXT_ABI_VERSION: u32 = 1;
pub const LUST_HANDLE_MAGIC: u32 = 0x4C555354; // 'LUST'

// Status codes
pub const LUST_EXT_STATUS_OK: i32 = 0;
pub const LUST_EXT_STATUS_ERR_GENERIC: i32 = 1;
pub const LUST_EXT_STATUS_ERR_OUT_OF_GAS: i32 = 2;
pub const LUST_EXT_STATUS_ERR_OUT_OF_MEMORY: i32 = 3;
pub const LUST_EXT_STATUS_ERR_INTERRUPTED: i32 = 4;
pub const LUST_EXT_STATUS_ERR_INVALID_HANDLE: i32 = 5;
pub const LUST_EXT_STATUS_ERR_STALE_HANDLE: i32 = 6;
pub const LUST_EXT_STATUS_ERR_CAPABILITY_DENIED: i32 = 7;
pub const LUST_EXT_STATUS_PANIC_DEFECT: i32 = 8;

pub type LustExtCleanFn = unsafe extern "C" fn(*mut core::ffi::c_void);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LustExtHostV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub vm_or_runtime_ptr: *mut core::ffi::c_void,
    pub charge_gas: unsafe extern "C" fn(ctx: *mut core::ffi::c_void, units: u64) -> i32,
    pub check_interrupt: unsafe extern "C" fn(ctx: *mut core::ffi::c_void, out_reason: *mut u32) -> i32,
    pub try_reserve_memory: unsafe extern "C" fn(
        ctx: *mut core::ffi::c_void,
        bytes: usize,
        out_reservation_id: *mut u64,
    ) -> i32,
    pub commit_memory: unsafe extern "C" fn(
        ctx: *mut core::ffi::c_void,
        reservation_id: u64,
        actual_bytes: usize,
    ) -> i32,
    pub rollback_memory: unsafe extern "C" fn(ctx: *mut core::ffi::c_void, reservation_id: u64) -> i32,
    pub acquire_resource: unsafe extern "C" fn(
        ctx: *mut core::ffi::c_void,
        scope_id: u64,
        handle: *mut core::ffi::c_void,
        cleanup: LustExtCleanFn,
        out_token: *mut u64,
    ) -> i32,
    pub release_resource: unsafe extern "C" fn(ctx: *mut core::ffi::c_void, token: u64) -> i32,
    pub host_alloc_value_bytes: unsafe extern "C" fn(
        ctx: *mut core::ffi::c_void,
        len: usize,
        out_ptr: *mut *mut u8,
    ) -> i32,
    pub host_free_value_bytes: unsafe extern "C" fn(ctx: *mut core::ffi::c_void, ptr: *mut u8, len: usize),
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LustExtValTag {
    Nil = 0,
    Bool = 1,
    Int = 2,
    Float = 3,
    String = 4,
    Handle = 5,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union LustExtValPayload {
    pub bool_val: bool,
    pub int_val: i64,
    pub float_val: f64,
    pub str_val: *const core::ffi::c_char,
    pub handle_val: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LustExtVal {
    pub tag: LustExtValTag,
    pub payload: LustExtValPayload,
}

impl Default for LustExtVal {
    fn default() -> Self {
        Self {
            tag: LustExtValTag::Nil,
            payload: LustExtValPayload { int_val: 0 },
        }
    }
}

pub type LustExtNativeFn = unsafe extern "C" fn(
    host: *const LustExtHostV1,
    args: *const LustExtVal,
    args_len: usize,
    out_val: *mut LustExtVal,
) -> i32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LustExtExportMeta {
    pub name: *const core::ffi::c_char,
    pub params: *const *const core::ffi::c_char,
    pub params_len: usize,
    pub return_type: *const core::ffi::c_char,
    pub doc: *const core::ffi::c_char,
    pub purity: u8, // 0 = Pure, 1 = Effectful
    pub required_capability: *const core::ffi::c_char,
    pub deterministic: bool,
    pub fn_ptr: Option<LustExtNativeFn>,
}

unsafe impl Sync for LustExtExportMeta {}
unsafe impl Send for LustExtExportMeta {}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LustExtManifestV1 {
    pub abi_version: u32,
    pub module_name: *const core::ffi::c_char,
    pub exports: *const LustExtExportMeta,
    pub exports_len: usize,
}

unsafe impl Sync for LustExtManifestV1 {}
unsafe impl Send for LustExtManifestV1 {}

#[inline]
pub const fn make_token(generation: u32, slot: u32) -> u64 {
    ((generation as u64) << 32) | (slot as u64)
}

#[inline]
pub const fn token_generation(token: u64) -> u32 {
    (token >> 32) as u32
}

#[inline]
pub const fn token_slot(token: u64) -> u32 {
    token as u32
}

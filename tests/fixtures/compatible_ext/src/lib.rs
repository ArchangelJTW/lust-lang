//! Compatible extension fixture for positive control testing using narrow stable C ABI.

use std::fs::OpenOptions;
use std::io::Write;

pub static LUST_EXT_ABI_VERSION: u32 = 1;

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

pub type LustExtNativeFn = unsafe extern "C" fn(
    host: *const LustExtHostV1,
    args: *const LustExtVal,
    args_len: usize,
    out_val: *mut LustExtVal,
) -> i32;

#[repr(C)]
pub struct LustExtExportMeta {
    pub name: *const core::ffi::c_char,
    pub params: *const *const core::ffi::c_char,
    pub params_len: usize,
    pub return_type: *const core::ffi::c_char,
    pub doc: *const core::ffi::c_char,
    pub purity: u8,
    pub required_capability: *const core::ffi::c_char,
    pub deterministic: bool,
    pub fn_ptr: Option<LustExtNativeFn>,
}

unsafe impl Sync for LustExtExportMeta {}
unsafe impl Send for LustExtExportMeta {}

#[repr(C)]
pub struct LustExtManifestV1 {
    pub abi_version: u32,
    pub module_name: *const core::ffi::c_char,
    pub exports: *const LustExtExportMeta,
    pub exports_len: usize,
}

unsafe impl Sync for LustExtManifestV1 {}
unsafe impl Send for LustExtManifestV1 {}

pub unsafe extern "C" fn compatible_double_export(
    host: *const LustExtHostV1,
    args: *const LustExtVal,
    args_len: usize,
    out_val: *mut LustExtVal,
) -> i32 {
    if host.is_null() || args.is_null() || out_val.is_null() {
        return 1;
    }
    let host_ref = &*host;
    if host_ref.abi_version != 1 {
        return 1;
    }

    // Call host callback table: charge_gas(10)
    let rc = (host_ref.charge_gas)(host_ref.vm_or_runtime_ptr, 10);
    if rc != 0 {
        return rc;
    }

    if args_len < 1 {
        return 1;
    }
    let arg0 = &*args;
    let input = match arg0.tag {
        LustExtValTag::Int => arg0.payload.int_val,
        _ => return 1,
    };

    let result = input * 2;
    (*out_val).tag = LustExtValTag::Int;
    (*out_val).payload.int_val = result;

    if let Ok(path) = std::env::var("LUST_FIXTURE_EVENT_LOG") {
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(f, "EVENT: COMPATIBLE_EXPORT_INVOKED input={input} result={result} gas_charged=10");
        }
    }

    0 // OK
}

pub unsafe extern "C" fn compatible_net_export(
    host: *const LustExtHostV1,
    _args: *const LustExtVal,
    _args_len: usize,
    out_val: *mut LustExtVal,
) -> i32 {
    if host.is_null() || out_val.is_null() {
        return 1;
    }
    if let Ok(path) = std::env::var("LUST_FIXTURE_EVENT_LOG") {
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(f, "EVENT: COMPATIBLE_NET_EXPORT_INVOKED");
        }
    }
    (*out_val).tag = LustExtValTag::Int;
    (*out_val).payload.int_val = 99;
    0
}

static EXPORT_DOUBLE_NAME: &[u8] = b"compatible_double\0";
static RETURN_TYPE_INT: &[u8] = b"int\0";
static PARAM_INT: &[u8] = b"int\0";

struct ParamList([*const core::ffi::c_char; 1]);
unsafe impl Sync for ParamList {}
static DOUBLE_PARAMS: ParamList = ParamList([PARAM_INT.as_ptr() as *const core::ffi::c_char]);

static EXPORT_NET_NAME: &[u8] = b"compatible_require_net\0";
static CAP_NET: &[u8] = b"net\0";

static EXPORTS: [LustExtExportMeta; 2] = [
    LustExtExportMeta {
        name: EXPORT_DOUBLE_NAME.as_ptr() as *const core::ffi::c_char,
        params: DOUBLE_PARAMS.0.as_ptr(),
        params_len: 1,
        return_type: RETURN_TYPE_INT.as_ptr() as *const core::ffi::c_char,
        doc: core::ptr::null(),
        purity: 0, // Pure
        required_capability: core::ptr::null(),
        deterministic: true,
        fn_ptr: Some(compatible_double_export),
    },
    LustExtExportMeta {
        name: EXPORT_NET_NAME.as_ptr() as *const core::ffi::c_char,
        params: core::ptr::null(),
        params_len: 0,
        return_type: RETURN_TYPE_INT.as_ptr() as *const core::ffi::c_char,
        doc: core::ptr::null(),
        purity: 1, // Effectful
        required_capability: CAP_NET.as_ptr() as *const core::ffi::c_char,
        deterministic: false,
        fn_ptr: Some(compatible_net_export),
    },
];

static MANIFEST: LustExtManifestV1 = LustExtManifestV1 {
    abi_version: 1,
    module_name: b"compatible_ext\0".as_ptr() as *const core::ffi::c_char,
    exports: EXPORTS.as_ptr(),
    exports_len: 2,
};

#[no_mangle]
pub extern "C" fn lust_ext_manifest_v1() -> *const LustExtManifestV1 {
    &MANIFEST
}

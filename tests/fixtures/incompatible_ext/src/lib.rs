//! Deliberately incompatible native extension fixture for testing ABI compatibility rejection.

use std::fs::OpenOptions;
use std::io::Write;

pub static LUST_EXT_ABI_VERSION: u32 = 999;

fn log_fixture_event(event: &str) {
    if let Ok(path) = std::env::var("LUST_FIXTURE_EVENT_LOG") {
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(f, "{event}");
        }
    }
}

pub type LustExtNativeFn = unsafe extern "C" fn(
    host: *const core::ffi::c_void,
    args: *const core::ffi::c_void,
    args_len: usize,
    out_val: *mut core::ffi::c_void,
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

#[cfg(feature = "hanging_constructor")]
#[used]
#[cfg_attr(target_os = "macos", link_section = "__DATA,__mod_init_func")]
#[cfg_attr(not(target_os = "macos"), link_section = ".init_array")]
static HANGING_CTOR: unsafe extern "C" fn() = {
    unsafe extern "C" fn hang_init() {
        log_fixture_event("EVENT: HANGING_CONSTRUCTOR_ENTERED");
        loop {
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    }
    hang_init
};

#[cfg(feature = "hanging_callback")]
pub unsafe extern "C" fn hanging_export_fn(
    _host: *const core::ffi::c_void,
    _args: *const core::ffi::c_void,
    _args_len: usize,
    _out_val: *mut core::ffi::c_void,
) -> i32 {
    log_fixture_event("EVENT: HANGING_CALLBACK_ENTERED");
    loop {
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

#[cfg(feature = "hanging_constructor")]
#[no_mangle]
pub extern "C" fn lust_ext_manifest_v1() -> *const LustExtManifestV1 {
    static MANIFEST: LustExtManifestV1 = LustExtManifestV1 {
        abi_version: 1,
        module_name: b"hanging_ctor_ext\0".as_ptr() as *const core::ffi::c_char,
        exports: core::ptr::null(),
        exports_len: 0,
    };
    &MANIFEST
}

#[cfg(feature = "hanging_callback")]
#[no_mangle]
pub extern "C" fn lust_ext_manifest_v1() -> *const LustExtManifestV1 {
    static EXPORT_NAME: &[u8] = b"hang_callback_fn\0";
    static HANG_EXPORT: LustExtExportMeta = LustExtExportMeta {
        name: EXPORT_NAME.as_ptr() as *const core::ffi::c_char,
        params: core::ptr::null(),
        params_len: 0,
        return_type: b"int\0".as_ptr() as *const core::ffi::c_char,
        doc: core::ptr::null(),
        purity: 0,
        required_capability: core::ptr::null(),
        deterministic: true,
        fn_ptr: Some(hanging_export_fn),
    };
    static MANIFEST: LustExtManifestV1 = LustExtManifestV1 {
        abi_version: 1,
        module_name: b"hanging_callback_ext\0".as_ptr() as *const core::ffi::c_char,
        exports: &HANG_EXPORT,
        exports_len: 1,
    };
    &MANIFEST
}

#[cfg(feature = "duplicate_export")]
#[no_mangle]
pub extern "C" fn lust_ext_manifest_v1() -> *const LustExtManifestV1 {
    static EXPORT_NAME: &[u8] = b"duplicate_fn\0";
    static DUPLICATE_EXPORTS: [LustExtExportMeta; 2] = [
        LustExtExportMeta {
            name: EXPORT_NAME.as_ptr() as *const core::ffi::c_char,
            params: core::ptr::null(),
            params_len: 0,
            return_type: b"int\0".as_ptr() as *const core::ffi::c_char,
            doc: core::ptr::null(),
            purity: 0,
            required_capability: core::ptr::null(),
            deterministic: true,
            fn_ptr: None,
        },
        LustExtExportMeta {
            name: EXPORT_NAME.as_ptr() as *const core::ffi::c_char,
            params: core::ptr::null(),
            params_len: 0,
            return_type: b"int\0".as_ptr() as *const core::ffi::c_char,
            doc: core::ptr::null(),
            purity: 0,
            required_capability: core::ptr::null(),
            deterministic: true,
            fn_ptr: None,
        },
    ];
    static MANIFEST: LustExtManifestV1 = LustExtManifestV1 {
        abi_version: 1,
        module_name: b"duplicate_ext\0".as_ptr() as *const core::ffi::c_char,
        exports: DUPLICATE_EXPORTS.as_ptr(),
        exports_len: 2,
    };
    &MANIFEST
}

#[cfg(feature = "same_version_legacy_raw_vm")]
#[no_mangle]
pub unsafe extern "C" fn lust_extension_register(_vm_ptr: *mut core::ffi::c_void) -> bool {
    // If called, record that registration was attempted (which proves failure of fail-closed policy)
    log_fixture_event("EVENT: LEGACY_RAW_VM_REGISTER_CALLED");
    false
}

#[cfg(feature = "version_mismatch")]
#[no_mangle]
pub extern "C" fn lust_ext_manifest_v1() -> *const LustExtManifestV1 {
    static MANIFEST: LustExtManifestV1 = LustExtManifestV1 {
        abi_version: 999,
        module_name: b"incompatible_ext\0".as_ptr() as *const core::ffi::c_char,
        exports: core::ptr::null(),
        exports_len: 0,
    };
    &MANIFEST
}

#[cfg(feature = "corrupt_manifest")]
#[no_mangle]
pub extern "C" fn lust_ext_manifest_v1() -> *const LustExtManifestV1 {
    static MANIFEST: LustExtManifestV1 = LustExtManifestV1 {
        abi_version: 1,
        module_name: b"corrupt_ext\0".as_ptr() as *const core::ffi::c_char,
        exports: core::ptr::null(),
        exports_len: 42,
    };
    &MANIFEST
}

#[cfg(feature = "null_manifest")]
#[no_mangle]
pub extern "C" fn lust_ext_manifest_v1() -> *const LustExtManifestV1 {
    core::ptr::null()
}

#[cfg(feature = "same_version_legacy_raw_vm")]
#[no_mangle]
pub extern "C" fn lust_ext_manifest_v1() -> *const LustExtManifestV1 {
    // Declares abi_version 1, but retains legacy raw *mut VM entry lust_extension_register
    static MANIFEST: LustExtManifestV1 = LustExtManifestV1 {
        abi_version: 1,
        module_name: b"legacy_raw_vm_ext\0".as_ptr() as *const core::ffi::c_char,
        exports: core::ptr::null(),
        exports_len: 0,
    };
    &MANIFEST
}

#[cfg(feature = "malformed_scalar_metadata")]
#[no_mangle]
pub extern "C" fn lust_ext_manifest_v1() -> *const LustExtManifestV1 {
    // Declares abi_version 1, but exports[0].purity has invalid scalar value 255 (valid: 0 or 1)
    static EXPORT_NAME: &[u8] = b"malformed_scalar_fn\0";
    static MALFORMED_EXPORT: LustExtExportMeta = LustExtExportMeta {
        name: EXPORT_NAME.as_ptr() as *const core::ffi::c_char,
        params: core::ptr::null(),
        params_len: 0,
        return_type: core::ptr::null(),
        doc: core::ptr::null(),
        purity: 255, // INVALID SCALAR: neither Pure (0) nor Effectful (1)
        required_capability: core::ptr::null(),
        deterministic: true,
        fn_ptr: None,
    };
    static MANIFEST: LustExtManifestV1 = LustExtManifestV1 {
        abi_version: 1,
        module_name: b"malformed_scalar_ext\0".as_ptr() as *const core::ffi::c_char,
        exports: &MALFORMED_EXPORT,
        exports_len: 1,
    };
    &MANIFEST
}

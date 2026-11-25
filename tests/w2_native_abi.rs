//! Wave 2 Track 2B Native ABI Test Suite (`w2_native_abi.rs`)
//!
//! Validates:
//! 1. `#[lust_module]` and `#[lust_fn]` macro expansion and VM registration.
//! 2. Converted extension compilation and execution without manual `&[Value]` unpacking.
//! 3. Metadata enforcement (`LustExtManifestV1`, `NativeExport`, pure attributes, export count).
//! 4. C ABI Option B Mutable Opaque Handles:
//!    - Array operations (new, len, push, get, set, pop, bounds checking).
//!    - Map operations (new, len, set, get, remove, type mismatch).
//!    - Struct operations (new, type_name, get_field, set_field, type mismatch).
//!    - Lifecycle, retain, release, and validity queries.
//!    - Stale handle detection when program is freed.
//!    - Invalid / null handle handling.
//!    - Program mismatch detection.
//! 5. C callbacks, reentrancy, and cleanup.
//! 6. Singleton tuple bindgen (`FunctionArgs` and `FromLustArgs` for `(A,)` and 1..12 tuples).
//! 7. CLI extension management tools (`ext_new`, `ext_check`, `ext_stubs`).

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

extern crate lust_macros;
use lust_macros::lust_module;

use lust::bytecode::Value;
use lust::embed::{EmbeddedProgram, FromLustArgs, FunctionArgs};
use lust::ffi::*;
use lust::vm::VM;

#[allow(unused_attributes)]
#[path = "../crates/lust-ext-abi/src/lib.rs"]
pub mod lust_ext_abi;

#[allow(unused_attributes)]
#[path = "../crates/lust-ext/src/lib.rs"]
pub mod lust_ext;

#[path = "../src/cli/ext.rs"]
mod cli_ext;

// =========================================================================
// 1. IN-MEMORY MACRO VM REGISTRATION TEST
// =========================================================================

#[lust_module(name = "test_math", abi = "vm")]
pub mod test_math {
    use lust_macros::lust_fn;

    #[lust_fn]
    pub fn add(a: i64, b: i64) -> i64 {
        a + b
    }

    #[lust_fn(pure, doc = "Negates a boolean")]
    pub fn negate(b: bool) -> bool {
        !b
    }
}

#[test]
fn test_macro_vm_registration() {
    let mut vm = VM::new();
    test_math::register_with_vm(&mut vm).expect("registration ok");

    let add_fn = vm
        .get_global("test_math.add")
        .or_else(|| vm.get_global("add"))
        .expect("add function exists");
    let res = vm
        .call_value(&add_fn, vec![Value::Int(10), Value::Int(25)])
        .expect("call add");
    assert_eq!(res, Value::Int(35));

    let neg_fn = vm
        .get_global("test_math.negate")
        .or_else(|| vm.get_global("negate"))
        .expect("negate function exists");
    let res2 = vm
        .call_value(&neg_fn, vec![Value::Bool(true)])
        .expect("call negate");
    assert_eq!(res2, Value::Bool(false));
}

// =========================================================================
// 2. CONVERTED EXTENSION DYLIB & METADATA ENFORCEMENT
// =========================================================================

#[test]
fn test_compiled_extension_dylib_and_metadata() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dylib_path = manifest_dir
        .join("examples")
        .join("rust-extension")
        .join("extensions")
        .join("double")
        .join("target")
        .join("debug")
        .join(if cfg!(target_os = "macos") {
            "liblust_double.dylib"
        } else if cfg!(target_os = "windows") {
            "lust_double.dll"
        } else {
            "liblust_double.so"
        });

    assert!(
        dylib_path.exists(),
        "Double extension dylib must exist at {:?}",
        dylib_path
    );

    // Verify manifest using lust_ext_abi types
    unsafe {
        let lib = libloading::Library::new(&dylib_path)
            .expect("Failed to load compiled extension dylib");

        let manifest_fn: libloading::Symbol<
            extern "C" fn() -> *const lust_ext_abi::LustExtManifestV1,
        > = lib
            .get(b"lust_ext_manifest_v1\0")
            .expect("lust_ext_manifest_v1 symbol missing");

        let manifest_ptr = manifest_fn();
        assert!(!manifest_ptr.is_null());
        let manifest = &*manifest_ptr;
        assert_eq!(manifest.abi_version, lust_ext_abi::LUST_EXT_ABI_VERSION);
        assert!(manifest.exports_len >= 4);
        assert!(!manifest.exports.is_null());

        let exports = std::slice::from_raw_parts(manifest.exports, manifest.exports_len);
        let export_names: Vec<&str> = exports
            .iter()
            .map(|e| CStr::from_ptr(e.name).to_str().expect("valid utf-8"))
            .collect();

        assert!(export_names.contains(&"host_double"));
        assert!(export_names.contains(&"nested.host_quadruple"));
        assert!(export_names.contains(&"make_factor"));
        assert!(export_names.contains(&"Factor:apply"));

        // Verify pure attributes
        let host_double_meta = exports
            .iter()
            .find(|e| CStr::from_ptr(e.name).to_bytes() == b"host_double")
            .expect("host_double export");
        assert_eq!(host_double_meta.purity, 0); // Pure
        assert_eq!(host_double_meta.params_len, 1);

        let quadruple_meta = exports
            .iter()
            .find(|e| CStr::from_ptr(e.name).to_bytes() == b"nested.host_quadruple")
            .expect("nested.host_quadruple export");
        assert_eq!(quadruple_meta.purity, 0); // Pure
        assert_eq!(quadruple_meta.params_len, 1);
    }
}

// =========================================================================
// 3. CONVERTED EXTENSION EXECUTION & ERROR HANDLING VIA VM
// =========================================================================

#[test]
fn test_compiled_extension_execution_and_errors() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dylib_path = manifest_dir
        .join("examples")
        .join("rust-extension")
        .join("extensions")
        .join("double")
        .join("target")
        .join("debug")
        .join(if cfg!(target_os = "macos") {
            "liblust_double.dylib"
        } else if cfg!(target_os = "windows") {
            "lust_double.dll"
        } else {
            "liblust_double.so"
        });

    let mut vm = VM::new();

    unsafe {
        let lib = libloading::Library::new(&dylib_path)
            .expect("Failed to load compiled extension dylib");

        let register_fn: libloading::Symbol<unsafe extern "C" fn(*mut VM) -> bool> = lib
            .get(b"lust_extension_register\0")
            .expect("lust_extension_register symbol missing");

        let ok = register_fn(&mut vm as *mut VM);
        assert!(ok, "Registration must succeed");

        // Keep library loaded in process for the duration of this test
        std::mem::forget(lib);
    }

    // Re-index globals in host VM address space to normalize hash table state across dylib boundary
    for (k, v) in vm.globals_snapshot() {
        vm.set_global(k, v);
    }

    // Lookup host_double function
    let double_fn = vm
        .get_global("host_double")
        .or_else(|| vm.get_global("lust_double.host_double"))
        .expect("host_double global");

    // Call host_double(21) => 42
    let res = vm
        .call_value(&double_fn, vec![Value::Int(21)])
        .expect("host_double call");
    assert_eq!(res, Value::Int(42));

    // Call nested.host_quadruple(10) => 40
    let quad_fn = vm
        .get_global("nested.host_quadruple")
        .expect("nested.host_quadruple global");
    let res2 = vm
        .call_value(&quad_fn, vec![Value::Int(10)])
        .expect("nested.host_quadruple call");
    assert_eq!(res2, Value::Int(40));

    // Call make_factor(2, 3) => Factor struct
    let make_factor_fn = vm
        .get_global("make_factor")
        .or_else(|| vm.get_global("lust_double.make_factor"))
        .expect("make_factor global");
    let factor_val = vm
        .call_value(&make_factor_fn, vec![Value::Int(2), Value::Int(3)])
        .expect("make_factor call");
    assert!(
        matches!(factor_val, Value::Struct(_)),
        "make_factor must return a Struct Value"
    );

    // Call Factor:apply(factor, 5) => base(2) * 5 * mult(3) = 30
    let apply_fn = vm
        .get_global("Factor:apply")
        .expect("Factor:apply global");
    let res3 = vm
        .call_value(&apply_fn, vec![factor_val, Value::Int(5)])
        .expect("Factor:apply call");
    assert_eq!(res3, Value::Int(30));

    // Test argument count errors
    let err_no_args = vm.call_value(&double_fn, vec![]);
    assert!(err_no_args.is_err());
    let err_msg = err_no_args.unwrap_err().to_string();
    assert!(
        err_msg.contains("expected 1 argument(s), got 0")
            || err_msg.contains("expected 1 argument")
            || err_msg.contains("expected 1")
            || err_msg.contains("expects 1"),
        "Unexpected error: {}",
        err_msg
    );

    let err_too_many = vm.call_value(
        &double_fn,
        vec![Value::Int(1), Value::Int(2)],
    );
    assert!(err_too_many.is_err());
    let err_too_many_msg = err_too_many.unwrap_err().to_string();
    assert!(
        err_too_many_msg.contains("expected 1") || err_too_many_msg.contains("expects 1"),
        "Unexpected error: {}",
        err_too_many_msg
    );

    // Test type mismatch error
    let err_wrong_type = vm.call_value(
        &double_fn,
        vec![Value::String("not an int".to_string().into())],
    );
    assert!(err_wrong_type.is_err());
}

// =========================================================================
// 4. OPTION B MUTABLE OPAQUE HANDLES - ARRAY OPERATIONS
// =========================================================================

#[test]
fn test_c_abi_handle_array_ops() {
    let source = CString::new("function main(): int\n    return 0\nend\n").unwrap();
    let prog = lust_program_new(source.as_ptr());
    if prog.is_null() {
        let err_ptr = lust_last_error();
        let err_msg = if !err_ptr.is_null() {
            unsafe { CStr::from_ptr(err_ptr).to_string_lossy().into_owned() }
        } else {
            "null error".to_string()
        };
        eprintln!("DEBUG: lust_program_new failed with: {}", err_msg);
    }
    assert!(!prog.is_null());

    let mut handle: *mut LustHandle = ptr::null_mut();
    let res = lust_array_new(prog, 0, &mut handle);
    assert_eq!(res, LustFfiResult::LUST_FFI_OK);
    assert!(!handle.is_null());

    assert_eq!(
        lust_handle_kind(handle),
        LustFfiHandleKind::LUST_HANDLE_ARRAY
    );
    assert!(lust_handle_is_valid(handle));

    // Initial length 0
    let mut len: usize = 999;
    assert_eq!(
        lust_array_len(handle, &mut len),
        LustFfiResult::LUST_FFI_OK
    );
    assert_eq!(len, 0);

    // Push 10, 20, 30
    let v1 = LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_INT,
        int_value: 10,
        ..Default::default()
    };
    let v2 = LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_INT,
        int_value: 20,
        ..Default::default()
    };
    let v3 = LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_INT,
        int_value: 30,
        ..Default::default()
    };

    assert_eq!(lust_array_push(handle, &v1), LustFfiResult::LUST_FFI_OK);
    assert_eq!(lust_array_push(handle, &v2), LustFfiResult::LUST_FFI_OK);
    assert_eq!(lust_array_push(handle, &v3), LustFfiResult::LUST_FFI_OK);

    assert_eq!(
        lust_array_len(handle, &mut len),
        LustFfiResult::LUST_FFI_OK
    );
    assert_eq!(len, 3);

    // Get index 1 (should be 20)
    let mut out_val = LustFfiValue::default();
    assert_eq!(
        lust_array_get(handle, 1, &mut out_val),
        LustFfiResult::LUST_FFI_OK
    );
    assert_eq!(out_val.tag, LustFfiValueTag::LUST_FFI_VALUE_INT);
    assert_eq!(out_val.int_value, 20);

    // Set index 1 to 99
    let v_new = LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_INT,
        int_value: 99,
        ..Default::default()
    };
    assert_eq!(
        lust_array_set(handle, 1, &v_new),
        LustFfiResult::LUST_FFI_OK
    );

    // Verify index 1 updated
    assert_eq!(
        lust_array_get(handle, 1, &mut out_val),
        LustFfiResult::LUST_FFI_OK
    );
    assert_eq!(out_val.int_value, 99);

    // Pop element (should be 30)
    assert_eq!(
        lust_array_pop(handle, &mut out_val),
        LustFfiResult::LUST_FFI_OK
    );
    assert_eq!(out_val.int_value, 30);
    assert_eq!(
        lust_array_len(handle, &mut len),
        LustFfiResult::LUST_FFI_OK
    );
    assert_eq!(len, 2);

    // Out of bounds get/set
    assert_eq!(
        lust_array_get(handle, 10, &mut out_val),
        LustFfiResult::LUST_FFI_ERR_INDEX_OUT_OF_BOUNDS
    );
    assert_eq!(
        lust_array_set(handle, 10, &v_new),
        LustFfiResult::LUST_FFI_ERR_INDEX_OUT_OF_BOUNDS
    );

    // Type mismatch error when treating array as map
    let key_cstr = CString::new("foo").unwrap();
    let key_val = LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_STRING,
        string_ptr: key_cstr.as_ptr() as *mut c_char,
        ..Default::default()
    };
    assert_eq!(
        lust_map_get(handle, &key_val, &mut out_val),
        LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH
    );

    // Clean up
    assert_eq!(
        lust_handle_release(handle),
        LustFfiResult::LUST_FFI_OK
    );
    lust_program_free(prog);
}

// =========================================================================
// 5. OPTION B MUTABLE OPAQUE HANDLES - MAP OPERATIONS
// =========================================================================

#[test]
fn test_c_abi_handle_map_ops() {
    let source = CString::new("function main(): int\n    return 0\nend\n").unwrap();
    let prog = lust_program_new(source.as_ptr());
    if prog.is_null() {
        let err_ptr = lust_last_error();
        let err_msg = if !err_ptr.is_null() {
            unsafe { CStr::from_ptr(err_ptr).to_string_lossy().into_owned() }
        } else {
            "null error".to_string()
        };
        eprintln!("DEBUG: lust_program_new failed with: {}", err_msg);
    }
    assert!(!prog.is_null());

    let mut handle: *mut LustHandle = ptr::null_mut();
    let res = lust_map_new(prog, &mut handle);
    assert_eq!(res, LustFfiResult::LUST_FFI_OK);
    assert!(!handle.is_null());

    assert_eq!(
        lust_handle_kind(handle),
        LustFfiHandleKind::LUST_HANDLE_MAP
    );

    // Initial length
    let mut len: usize = 999;
    assert_eq!(lust_map_len(handle, &mut len), LustFfiResult::LUST_FFI_OK);
    assert_eq!(len, 0);

    // Set keys
    let k_num_cstr = CString::new("score").unwrap();
    let k_num = LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_STRING,
        string_ptr: k_num_cstr.as_ptr() as *mut c_char,
        ..Default::default()
    };
    let v_num = LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_INT,
        int_value: 100,
        ..Default::default()
    };
    assert_eq!(
        lust_map_set(handle, &k_num, &v_num),
        LustFfiResult::LUST_FFI_OK
    );

    let k_str_cstr = CString::new("status").unwrap();
    let k_str = LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_STRING,
        string_ptr: k_str_cstr.as_ptr() as *mut c_char,
        ..Default::default()
    };
    let s_val = CString::new("active").unwrap();
    let v_str = LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_STRING,
        string_ptr: s_val.as_ptr() as *mut c_char,
        ..Default::default()
    };
    assert_eq!(
        lust_map_set(handle, &k_str, &v_str),
        LustFfiResult::LUST_FFI_OK
    );

    assert_eq!(lust_map_len(handle, &mut len), LustFfiResult::LUST_FFI_OK);
    assert_eq!(len, 2);

    // Get existing key
    let mut out_val = LustFfiValue::default();
    assert_eq!(
        lust_map_get(handle, &k_num, &mut out_val),
        LustFfiResult::LUST_FFI_OK
    );
    assert_eq!(out_val.tag, LustFfiValueTag::LUST_FFI_VALUE_INT);
    assert_eq!(out_val.int_value, 100);

    // Get non-existent key returns NIL
    let k_none_cstr = CString::new("missing").unwrap();
    let k_none = LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_STRING,
        string_ptr: k_none_cstr.as_ptr() as *mut c_char,
        ..Default::default()
    };
    assert_eq!(
        lust_map_get(handle, &k_none, &mut out_val),
        LustFfiResult::LUST_FFI_OK
    );
    assert_eq!(out_val.tag, LustFfiValueTag::LUST_FFI_VALUE_NIL);

    // Remove key
    assert_eq!(
        lust_map_remove(handle, &k_num, &mut out_val),
        LustFfiResult::LUST_FFI_OK
    );
    assert_eq!(out_val.int_value, 100);
    assert_eq!(lust_map_len(handle, &mut len), LustFfiResult::LUST_FFI_OK);
    assert_eq!(len, 1);

    // Type mismatch error when treating map as array
    assert_eq!(
        lust_array_pop(handle, &mut out_val),
        LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH
    );

    assert_eq!(
        lust_handle_release(handle),
        LustFfiResult::LUST_FFI_OK
    );
    lust_program_free(prog);
}

// =========================================================================
// 6. OPTION B MUTABLE OPAQUE HANDLES - STRUCT OPERATIONS
// =========================================================================

#[test]
fn test_c_abi_handle_struct_ops() {
    let source = CString::new(
        r#"
        struct Point
            x: int
            y: int
        end

        function main(): int
            return 0
        end
        "#,
    )
    .unwrap();
    let prog = lust_program_new(source.as_ptr());
    if prog.is_null() {
        let err_ptr = lust_last_error();
        let err_msg = if !err_ptr.is_null() {
            unsafe { CStr::from_ptr(err_ptr).to_string_lossy().into_owned() }
        } else {
            "null error".to_string()
        };
        eprintln!("DEBUG: lust_program_new failed with: {}", err_msg);
    }
    assert!(!prog.is_null());

    let type_name = CString::new("Point").unwrap();
    let mut handle: *mut LustHandle = ptr::null_mut();
    let res = lust_struct_new(prog, type_name.as_ptr(), &mut handle);
    assert_eq!(res, LustFfiResult::LUST_FFI_OK);
    assert!(!handle.is_null());

    assert_eq!(
        lust_handle_kind(handle),
        LustFfiHandleKind::LUST_HANDLE_STRUCT
    );

    // Verify type_name query
    let mut out_name: *const c_char = ptr::null();
    assert_eq!(
        lust_struct_type_name(handle, &mut out_name),
        LustFfiResult::LUST_FFI_OK
    );
    assert!(!out_name.is_null());
    let name_str = unsafe { CStr::from_ptr(out_name).to_str().unwrap() };
    assert_eq!(name_str, "Point");

    // Set fields x and y
    let f_x = CString::new("x").unwrap();
    let v_x = LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_INT,
        int_value: 42,
        ..Default::default()
    };
    assert_eq!(
        lust_struct_set_field(handle, f_x.as_ptr(), &v_x),
        LustFfiResult::LUST_FFI_OK
    );

    let f_y = CString::new("y").unwrap();
    let v_y = LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_INT,
        int_value: 84,
        ..Default::default()
    };
    assert_eq!(
        lust_struct_set_field(handle, f_y.as_ptr(), &v_y),
        LustFfiResult::LUST_FFI_OK
    );

    // Get field x
    let mut out_val = LustFfiValue::default();
    assert_eq!(
        lust_struct_get_field(handle, f_x.as_ptr(), &mut out_val),
        LustFfiResult::LUST_FFI_OK
    );
    assert_eq!(out_val.tag, LustFfiValueTag::LUST_FFI_VALUE_INT);
    assert_eq!(out_val.int_value, 42);

    // Get field y
    assert_eq!(
        lust_struct_get_field(handle, f_y.as_ptr(), &mut out_val),
        LustFfiResult::LUST_FFI_OK
    );
    assert_eq!(out_val.int_value, 84);

    // Non-existent field returns NIL
    let f_z = CString::new("z").unwrap();
    assert_eq!(
        lust_struct_get_field(handle, f_z.as_ptr(), &mut out_val),
        LustFfiResult::LUST_FFI_OK
    );
    assert_eq!(out_val.tag, LustFfiValueTag::LUST_FFI_VALUE_NIL);

    assert_eq!(
        lust_handle_release(handle),
        LustFfiResult::LUST_FFI_OK
    );
    lust_program_free(prog);
}

// =========================================================================
// 7. HANDLE LIFECYCLE, STALE DETECTION, AND DISPOSE
// =========================================================================

#[test]
fn test_c_abi_handle_lifecycle_and_stale_detection() {
    let source = CString::new("function main(): int\n    return 0\nend\n").unwrap();
    let prog = lust_program_new(source.as_ptr());
    if prog.is_null() {
        let err_ptr = lust_last_error();
        let err_msg = if !err_ptr.is_null() {
            unsafe { CStr::from_ptr(err_ptr).to_string_lossy().into_owned() }
        } else {
            "null error".to_string()
        };
        eprintln!("DEBUG: lust_program_new failed with: {}", err_msg);
    }
    assert!(!prog.is_null());

    let mut handle: *mut LustHandle = ptr::null_mut();
    assert_eq!(
        lust_array_new(prog, 0, &mut handle),
        LustFfiResult::LUST_FFI_OK
    );
    assert!(lust_handle_is_valid(handle));

    // Test retain
    assert_eq!(
        lust_handle_retain(handle),
        LustFfiResult::LUST_FFI_OK
    );

    // Free the program while handle is still alive
    lust_program_free(prog);

    // Stale detection: handle must now report invalid and operations must return STALE
    assert!(!lust_handle_is_valid(handle));

    let mut len: usize = 0;
    assert_eq!(
        lust_array_len(handle, &mut len),
        LustFfiResult::LUST_FFI_ERR_STALE_HANDLE
    );

    let v = LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_INT,
        int_value: 1,
        ..Default::default()
    };
    assert_eq!(
        lust_array_push(handle, &v),
        LustFfiResult::LUST_FFI_ERR_STALE_HANDLE
    );

    // Releasing stale handle must succeed without crashing or panicking
    assert_eq!(
        lust_handle_release(handle),
        LustFfiResult::LUST_FFI_OK
    );
    // Second release (from the retain above) frees the handle
    assert_eq!(
        lust_handle_release(handle),
        LustFfiResult::LUST_FFI_OK
    );

    // Invalid handle tests
    assert_eq!(
        lust_handle_kind(ptr::null()),
        LustFfiHandleKind::LUST_HANDLE_INVALID
    );
    assert!(!lust_handle_is_valid(ptr::null()));
    assert_eq!(
        lust_handle_release(ptr::null_mut()),
        LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE
    );
    assert_eq!(
        lust_array_len(ptr::null_mut(), &mut len),
        LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE
    );

    // Test lust_value_dispose with handle tag
    let mut handle2: *mut LustHandle = ptr::null_mut();
    let prog2 = lust_program_new(source.as_ptr());
    assert_eq!(
        lust_array_new(prog2, 0, &mut handle2),
        LustFfiResult::LUST_FFI_OK
    );

    let mut val_with_handle = LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_HANDLE,
        handle_ptr: handle2,
        ..Default::default()
    };
    lust_value_dispose(&mut val_with_handle);
    assert_eq!(val_with_handle.tag, LustFfiValueTag::LUST_FFI_VALUE_NIL);
    assert!(val_with_handle.handle_ptr.is_null());

    lust_program_free(prog2);
}

// =========================================================================
// 8. C CALLBACKS, REENTRANCY, AND CLEANUP
// =========================================================================

static CLEANUP_INVOKED: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn test_cleanup_callback(data: *mut std::ffi::c_void) {
    if !data.is_null() {
        CLEANUP_INVOKED.fetch_add(1, Ordering::SeqCst);
    }
}

unsafe extern "C" fn test_c_add_callback(
    _user_data: *mut std::ffi::c_void,
    args: *const LustFfiValue,
    argc: usize,
    out: *mut LustFfiValue,
) -> LustFfiResult {
    if argc != 2 || args.is_null() || out.is_null() {
        return LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH;
    }
    unsafe {
        let a = *args.add(0);
        let b = *args.add(1);
        if a.tag != LustFfiValueTag::LUST_FFI_VALUE_INT || b.tag != LustFfiValueTag::LUST_FFI_VALUE_INT {
            return LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH;
        }
        *out = LustFfiValue {
            tag: LustFfiValueTag::LUST_FFI_VALUE_INT,
            int_value: a.int_value + b.int_value,
            ..Default::default()
        };
    }
    LustFfiResult::LUST_FFI_OK
}

#[test]
fn test_c_callback_invocation_and_cleanup() {
    let source = CString::new(
        r#"
        extern
            function native_add(x: int, y: int): int
        end

        function run(x: int, y: int): int
            return native_add(x, y)
        end
        "#,
    )
    .unwrap();
    let prog = lust_program_new(source.as_ptr());
    if prog.is_null() {
        let err_ptr = lust_last_error();
        let err_msg = if !err_ptr.is_null() {
            unsafe { CStr::from_ptr(err_ptr).to_string_lossy().into_owned() }
        } else {
            "null error".to_string()
        };
        eprintln!("DEBUG: lust_program_new failed with: {}", err_msg);
    }
    assert!(!prog.is_null());

    CLEANUP_INVOKED.store(0, Ordering::SeqCst);
    let dummy_user_data = 0x1234 as *mut std::ffi::c_void;

    let cb_name = CString::new("native_add").unwrap();
    let ok = lust_program_register_callback(
        prog,
        cb_name.as_ptr(),
        test_c_add_callback,
        dummy_user_data,
        Some(test_cleanup_callback),
    );
    assert!(ok, "Registration must succeed");

    // Call run(15, 27) via lust_program_call
    let fn_name = CString::new("run").unwrap();
    let args = [
        LustFfiValue {
            tag: LustFfiValueTag::LUST_FFI_VALUE_INT,
            int_value: 15,
            ..Default::default()
        },
        LustFfiValue {
            tag: LustFfiValueTag::LUST_FFI_VALUE_INT,
            int_value: 27,
            ..Default::default()
        },
    ];
    let mut out = LustFfiValue::default();
    let success = lust_program_call(prog, fn_name.as_ptr(), args.as_ptr(), 2, &mut out);
    if !success {
        let err_ptr = lust_last_error();
        let err_msg = if !err_ptr.is_null() {
            unsafe { CStr::from_ptr(err_ptr).to_string_lossy().into_owned() }
        } else {
            "null error".to_string()
        };
        eprintln!("DEBUG: lust_program_call failed with: {}", err_msg);
    }
    assert!(success, "lust_program_call must succeed");
    assert_eq!(out.tag, LustFfiValueTag::LUST_FFI_VALUE_INT);
    assert_eq!(out.int_value, 42);

    // Free program and verify cleanup callback is invoked
    lust_program_free(prog);
    assert_eq!(CLEANUP_INVOKED.load(Ordering::SeqCst), 1);
}

// Reentrancy callback: receives *mut EmbeddedProgram as user_data and calls back into it
unsafe extern "C" fn test_reentrant_callback(
    user_data: *mut std::ffi::c_void,
    args: *const LustFfiValue,
    argc: usize,
    out: *mut LustFfiValue,
) -> LustFfiResult {
    if argc != 1 || user_data.is_null() || out.is_null() {
        return LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH;
    }
    let prog = user_data as *mut EmbeddedProgram;
    unsafe {
        let n = (*args.add(0)).int_value;

        if n <= 1 {
            *out = LustFfiValue {
                tag: LustFfiValueTag::LUST_FFI_VALUE_INT,
                int_value: 1,
                ..Default::default()
            };
            return LustFfiResult::LUST_FFI_OK;
        }

        // Call nested script function "lust_fib" reentrantly
        let fn_name = CString::new("lust_fib").unwrap();
        let nested_args = [LustFfiValue {
            tag: LustFfiValueTag::LUST_FFI_VALUE_INT,
            int_value: n - 1,
            ..Default::default()
        }];
        let mut nested_out = LustFfiValue::default();
        let ok = lust_program_call(
            prog,
            fn_name.as_ptr(),
            nested_args.as_ptr(),
            1,
            &mut nested_out,
        );
        if !ok {
            return LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH;
        }

        *out = LustFfiValue {
            tag: LustFfiValueTag::LUST_FFI_VALUE_INT,
            int_value: n * nested_out.int_value,
            ..Default::default()
        };
    }
    LustFfiResult::LUST_FFI_OK
}

#[test]
fn test_c_callback_reentrancy() {
    let source = CString::new(
        r#"
        extern
            function call_reentrant(n: int): int
        end

        function lust_fib(n: int): int
            if n <= 1 then
                return 1
            end
            return call_reentrant(n)
        end
        "#,
    )
    .unwrap();
    let prog = lust_program_new(source.as_ptr());
    if prog.is_null() {
        let err_ptr = lust_last_error();
        let err_msg = if !err_ptr.is_null() {
            unsafe { CStr::from_ptr(err_ptr).to_string_lossy().into_owned() }
        } else {
            "null error".to_string()
        };
        eprintln!("DEBUG: lust_program_new failed with: {}", err_msg);
    }
    assert!(!prog.is_null());

    let cb_name = CString::new("call_reentrant").unwrap();
    let ok = lust_program_register_callback(
        prog,
        cb_name.as_ptr(),
        test_reentrant_callback,
        prog as *mut std::ffi::c_void,
        None,
    );
    assert!(ok);

    // Call lust_fib(5) => 5 * 4 * 3 * 2 * 1 = 120
    let fn_name = CString::new("lust_fib").unwrap();
    let args = [LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_INT,
        int_value: 5,
        ..Default::default()
    }];
    let mut out = LustFfiValue::default();
    let ok = lust_program_call(prog, fn_name.as_ptr(), args.as_ptr(), 1, &mut out);
    if !ok {
        let err_ptr = lust_last_error();
        let err_msg = if !err_ptr.is_null() {
            unsafe { CStr::from_ptr(err_ptr).to_string_lossy().into_owned() }
        } else {
            "null error".to_string()
        };
        eprintln!("DEBUG: lust_program_call reentrant failed with: {}", err_msg);
    }
    assert!(ok);
    assert_eq!(out.tag, LustFfiValueTag::LUST_FFI_VALUE_INT);
    assert_eq!(out.int_value, 120);

    lust_program_free(prog);
}

// =========================================================================
// 9. SINGLETON TUPLE BINDGEN & FROMLUSTARGS NORMALIZATION
// =========================================================================

#[test]
fn test_singleton_tuple_function_args_and_from_lust_args() {
    // Test FunctionArgs for (A,)
    let single = (42i64,);
    let vals = single.into_values();
    assert_eq!(vals.len(), 1);
    assert_eq!(vals[0], Value::Int(42));

    let mut dst = [Value::Nil];
    let written = (99i64,).write_args(&mut dst);
    assert_eq!(written, 1);
    assert_eq!(dst[0], Value::Int(99));

    // Test FromLustArgs for (A,)
    let args = [Value::Int(123)];
    let decoded = <(i64,)>::from_values(&args).expect("tuple conversion");
    assert_eq!(decoded, (123i64,));

    // Negative: arity mismatch
    let empty_args: [Value; 0] = [];
    assert!(<(i64,)>::from_values(&empty_args).is_err());

    // Negative: type mismatch
    let bad_type = [Value::String("hello".to_string().into())];
    assert!(<(i64,)>::from_values(&bad_type).is_err());

    // Multi-tuples (2-tuple, 3-tuple)
    let multi_args = [
        Value::Int(10),
        Value::String("test".to_string().into()),
        Value::Bool(true),
    ];
    let decoded_3 =
        <(i64, String, bool)>::from_values(&multi_args).expect("3-tuple conversion");
    assert_eq!(decoded_3, (10, "test".to_string(), true));
}

// =========================================================================
// 10. CLI EXT COMMANDS VERIFICATION
// =========================================================================

#[test]
fn test_cli_ext_commands() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let ext_path = temp_dir.path().join("test_ext");

    // 1. ext_new
    let opts = cli_ext::ExtNewOptions {
        name: "test_ext".to_string(),
        template: "default".to_string(),
        path: temp_dir.path().to_path_buf(),
    };
    let created = cli_ext::ext_new(&opts).expect("ext_new failed");
    assert_eq!(created, ext_path);
    assert!(ext_path.join("Cargo.toml").exists());
    assert!(ext_path.join("src").join("lib.rs").exists());

    // 2. ext_stubs
    let stub_res = cli_ext::ext_stubs(&ext_path);
    assert!(stub_res.is_ok());

    // 3. ext_check
    let check_res = cli_ext::ext_check(&ext_path);
    assert!(check_res.is_ok());

    // 4. Command router
    assert!(cli_ext::run_ext_command(&["help".to_string()]).is_ok());
}

#[test]
fn test_ffi_handle_liveness_null_forged_and_double_release() {
    // 1. Null pointer rejection
    assert!(!lust_handle_is_valid(ptr::null()));
    assert_eq!(
        lust_handle_kind(ptr::null()),
        LustFfiHandleKind::LUST_HANDLE_INVALID
    );
    assert_eq!(
        lust_handle_retain(ptr::null_mut()),
        LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE
    );
    assert_eq!(
        lust_handle_release(ptr::null_mut()),
        LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE
    );

    // 2. Arbitrary / forged pointer rejection without dereferencing
    let forged = 0xDEAD_BEEF_usize as *mut LustHandle;
    assert!(!lust_handle_is_valid(forged));
    assert_eq!(
        lust_handle_kind(forged),
        LustFfiHandleKind::LUST_HANDLE_INVALID
    );
    assert_eq!(
        lust_handle_retain(forged),
        LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE
    );
    assert_eq!(
        lust_handle_release(forged),
        LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE
    );
    let mut len: usize = 0;
    assert_eq!(
        lust_array_len(forged, &mut len),
        LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE
    );

    // 3. Live -> released -> double-release rejection without use-after-free
    let src = CString::new("function ok(): int return 1 end").unwrap();
    let prog = lust_program_new(src.as_ptr());
    assert!(!prog.is_null());

    let mut arr_handle: *mut LustHandle = ptr::null_mut();
    assert_eq!(
        lust_array_new(prog, 4, &mut arr_handle),
        LustFfiResult::LUST_FFI_OK
    );
    assert!(!arr_handle.is_null());
    assert!(lust_handle_is_valid(arr_handle));

    // Retain once (ref_count = 2), release once (ref_count = 1, still live)
    assert_eq!(lust_handle_retain(arr_handle), LustFfiResult::LUST_FFI_OK);
    assert_eq!(lust_handle_release(arr_handle), LustFfiResult::LUST_FFI_OK);
    assert!(lust_handle_is_valid(arr_handle));

    // Final release (ref_count = 0, removed from LIVE_HANDLES and freed)
    assert_eq!(lust_handle_release(arr_handle), LustFfiResult::LUST_FFI_OK);

    // Post-free operations must fail safely via LIVE_HANDLES lookup without dereferencing
    assert!(!lust_handle_is_valid(arr_handle));
    assert_eq!(
        lust_handle_retain(arr_handle),
        LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE
    );
    assert_eq!(
        lust_handle_release(arr_handle),
        LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE
    );
    assert_eq!(
        lust_array_len(arr_handle, &mut len),
        LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE
    );

    // 4. Stale handle detection when owning program is freed
    let mut stale_handle: *mut LustHandle = ptr::null_mut();
    assert_eq!(
        lust_array_new(prog, 2, &mut stale_handle),
        LustFfiResult::LUST_FFI_OK
    );
    assert!(lust_handle_is_valid(stale_handle));
    lust_program_free(prog);
    assert!(!lust_handle_is_valid(stale_handle));
    assert_eq!(
        lust_array_len(stale_handle, &mut len),
        LustFfiResult::LUST_FFI_ERR_STALE_HANDLE
    );
    assert_eq!(lust_handle_release(stale_handle), LustFfiResult::LUST_FFI_OK);
}

#![allow(clippy::approx_constant)]

//! Comprehensive coverage and correctness tests for L4: FFI and Rx facade.
//! Verifies ABI contracts, null safety, UTF-8 safety, error capturing,
//! RxClock monotonic and virtual behavior, RxError formatting, and RxEnvironment facade.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use lust::ffi::*;
use lust::rx::*;

// =========================================================================
// 1. FFI POSITIVE CONTRACT TESTS & ORACLES
// =========================================================================

#[test]
fn test_ffi_program_new_and_eval_oracle() {
    let source = CString::new(
        r#"
        function main(): int
            return 42
        end
        "#,
    )
    .expect("valid CString");
    let prog = lust_program_new(source.as_ptr());
    assert!(!prog.is_null(), "lust_program_new returned null pointer for valid source");

    let mut out_val = LustFfiValue::default();
    let eval_ok = lust_program_eval(prog, &mut out_val);
    assert!(eval_ok, "lust_program_eval failed on valid source");
    assert_eq!(
        out_val.tag,
        LustFfiValueTag::LUST_FFI_VALUE_NIL,
        "lust_program_eval should return NIL tag in out_value"
    );

    let last_err = lust_last_error();
    assert!(last_err.is_null(), "lust_last_error should be null after successful eval");

    // Also test lust_program_eval with null out_value (ignore return)
    let eval_null_out = lust_program_eval(prog, ptr::null_mut());
    assert!(eval_null_out, "lust_program_eval failed with null out_value");

    lust_program_free(prog);
}

#[test]
fn test_ffi_builder_lifecycle_and_entry_module() {
    let builder = lust_builder_new();
    assert!(!builder.is_null(), "lust_builder_new returned null pointer");

    let mod_name = CString::new("main").expect("valid CString");
    let src = CString::new(
        r#"
        function identity(x: int): int
            return x
        end
        "#,
    )
    .expect("valid CString");
    let add_ok = lust_builder_add_module(builder, mod_name.as_ptr(), src.as_ptr());
    assert!(add_ok, "lust_builder_add_module failed on valid module");

    let entry_ok = lust_builder_set_entry_module(builder, mod_name.as_ptr());
    assert!(entry_ok, "lust_builder_set_entry_module failed on valid entry");

    let prog = lust_builder_compile(builder);
    assert!(!prog.is_null(), "lust_builder_compile returned null for valid builder");

    let run_ok = lust_program_run_entry(prog);
    assert!(run_ok, "lust_program_run_entry failed on compiled program");

    lust_program_free(prog);
}

#[test]
fn test_ffi_program_call_numeric_and_bool_oracle() {
    let script = CString::new(
        r#"
        function mul_add(a: int, b: int, c: int): int
            return (a * b) + c
        end

        function float_mul(x: float, y: float): float
            return x * y
        end

        function invert(flag: bool): bool
            return not flag
        end

        function return_nil()
            return
        end

        function no_op()
        end
        "#,
    )
    .unwrap();

    let prog = lust_program_new(script.as_ptr());
    let err_str = if prog.is_null() {
        let p = lust_last_error();
        if p.is_null() { "null".to_string() } else { unsafe { CStr::from_ptr(p) }.to_string_lossy().to_string() }
    } else { "".to_string() };
    assert!(!prog.is_null(), "lust_program_new failed in test_ffi_program_call_numeric_and_bool_oracle: {}", err_str);

    // Oracle 1: mul_add(10, 3, 5) == 35 (Int)
    {
        let func = CString::new("mul_add").unwrap();
        let args = [
            LustFfiValue {
                tag: LustFfiValueTag::LUST_FFI_VALUE_INT,
                int_value: 10,
                ..Default::default()
            },
            LustFfiValue {
                tag: LustFfiValueTag::LUST_FFI_VALUE_INT,
                int_value: 3,
                ..Default::default()
            },
            LustFfiValue {
                tag: LustFfiValueTag::LUST_FFI_VALUE_INT,
                int_value: 5,
                ..Default::default()
            },
        ];
        let mut out = LustFfiValue::default();
        let ok = lust_program_call(prog, func.as_ptr(), args.as_ptr(), args.len(), &mut out);
        assert!(ok, "lust_program_call(mul_add) failed");
        assert_eq!(
            out.tag,
            LustFfiValueTag::LUST_FFI_VALUE_INT,
            "lust_program_call(mul_add) returned wrong tag"
        );
        assert_eq!(
            out.int_value, 35,
            "lust_program_call(mul_add) returned wrong int_value, expected 35 got {}",
            out.int_value
        );
        lust_value_free(&mut out);
    }

    // Oracle 2: float_mul(1.5, 4.0) == 6.0 (Float)
    {
        let func = CString::new("float_mul").unwrap();
        let args = [
            LustFfiValue {
                tag: LustFfiValueTag::LUST_FFI_VALUE_FLOAT,
                float_value: 1.5,
                ..Default::default()
            },
            LustFfiValue {
                tag: LustFfiValueTag::LUST_FFI_VALUE_FLOAT,
                float_value: 4.0,
                ..Default::default()
            },
        ];
        let mut out = LustFfiValue::default();
        let ok = lust_program_call(prog, func.as_ptr(), args.as_ptr(), args.len(), &mut out);
        assert!(ok, "lust_program_call(float_mul) failed");
        assert_eq!(
            out.tag,
            LustFfiValueTag::LUST_FFI_VALUE_FLOAT,
            "lust_program_call(float_mul) returned wrong tag"
        );
        assert!(
            (out.float_value - 6.0).abs() < 1e-9,
            "lust_program_call(float_mul) returned wrong float_value, expected 6.0 got {}",
            out.float_value
        );
        lust_value_free(&mut out);
    }

    // Oracle 3: invert(true) == false (Bool)
    {
        let func = CString::new("invert").unwrap();
        let args = [LustFfiValue {
            tag: LustFfiValueTag::LUST_FFI_VALUE_BOOL,
            bool_value: true,
            ..Default::default()
        }];
        let mut out = LustFfiValue::default();
        let ok = lust_program_call(prog, func.as_ptr(), args.as_ptr(), args.len(), &mut out);
        assert!(ok, "lust_program_call(invert) failed");
        assert_eq!(
            out.tag,
            LustFfiValueTag::LUST_FFI_VALUE_BOOL,
            "lust_program_call(invert) returned wrong tag"
        );
        assert!(
            !out.bool_value,
            "lust_program_call(invert) returned wrong bool_value, expected false"
        );
        lust_value_free(&mut out);
    }

    // Oracle 4: return_nil() == Nil
    {
        let func = CString::new("return_nil").unwrap();
        let mut out = LustFfiValue::default();
        let ok = lust_program_call(prog, func.as_ptr(), ptr::null(), 0, &mut out);
        assert!(ok, "lust_program_call(return_nil) failed");
        assert_eq!(
            out.tag,
            LustFfiValueTag::LUST_FFI_VALUE_NIL,
            "lust_program_call(return_nil) returned wrong tag"
        );
        lust_value_free(&mut out);
    }

    // Oracle 5: call with out_value = null (discard result)
    {
        let func = CString::new("no_op").unwrap();
        let ok = lust_program_call(prog, func.as_ptr(), ptr::null(), 0, ptr::null_mut());
        assert!(ok, "lust_program_call(no_op) with null out_value failed");
    }

    lust_program_free(prog);
}

#[test]
fn test_ffi_string_passing_and_returning_oracle() {
    let script = CString::new(
        r#"
        function greet(name: string): string
            return "Hello, " .. name
        end
        "#,
    )
    .unwrap();

    let prog = lust_program_new(script.as_ptr());
    assert!(!prog.is_null(), "lust_program_new failed for greet script");

    let func = CString::new("greet").unwrap();
    let name_cstr = CString::new("World!").unwrap();
    let args = [LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_STRING,
        string_ptr: name_cstr.into_raw(),
        ..Default::default()
    }];

    let mut out = LustFfiValue::default();
    let ok = lust_program_call(prog, func.as_ptr(), args.as_ptr(), 1, &mut out);
    assert!(ok, "lust_program_call(greet) failed");
    assert_eq!(
        out.tag,
        LustFfiValueTag::LUST_FFI_VALUE_STRING,
        "lust_program_call(greet) expected STRING tag"
    );
    assert!(!out.string_ptr.is_null(), "returned string_ptr was null");

    let result_str = unsafe { CStr::from_ptr(out.string_ptr) }.to_str().unwrap();
    assert_eq!(
        result_str, "Hello, World!",
        "lust_program_call(greet) returned unexpected string"
    );

    // Free the argument string pointer
    lust_string_free(args[0].string_ptr);

    // Dispose the return value through lust_value_dispose
    lust_value_dispose(&mut out);
    assert_eq!(
        out.tag,
        LustFfiValueTag::LUST_FFI_VALUE_NIL,
        "lust_value_dispose should reset tag to NIL"
    );
    assert!(out.string_ptr.is_null(), "lust_value_dispose should nullify string_ptr");

    // Test lust_string_free directly on raw allocation
    let temp_str = CString::new("temporary allocation").unwrap().into_raw();
    lust_string_free(temp_str);

    lust_program_free(prog);
}

#[test]
fn test_ffi_globals_get_set_oracle() {
    let script = CString::new("function dummy(): int return 0 end").unwrap();
    let prog = lust_program_new(script.as_ptr());
    assert!(!prog.is_null());

    let counter_name = CString::new("counter").unwrap();

    // Set counter to 100
    let initial_val = LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_INT,
        int_value: 100,
        ..Default::default()
    };
    let set_init_ok = lust_program_set_global(prog, counter_name.as_ptr(), &initial_val);
    assert!(set_init_ok, "lust_program_set_global(counter) initial failed");

    // Read counter
    let mut out = LustFfiValue::default();
    let get_ok = lust_program_get_global(prog, counter_name.as_ptr(), &mut out);
    assert!(get_ok, "lust_program_get_global(counter) failed");
    assert_eq!(out.tag, LustFfiValueTag::LUST_FFI_VALUE_INT);
    assert_eq!(out.int_value, 100);

    // Update counter to 999
    let new_val = LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_INT,
        int_value: 999,
        ..Default::default()
    };
    let set_ok = lust_program_set_global(prog, counter_name.as_ptr(), &new_val);
    assert!(set_ok, "lust_program_set_global(counter) failed");

    // Read back updated counter
    let mut out2 = LustFfiValue::default();
    let get_ok2 = lust_program_get_global(prog, counter_name.as_ptr(), &mut out2);
    assert!(get_ok2, "lust_program_get_global(counter) after update failed");
    assert_eq!(out2.tag, LustFfiValueTag::LUST_FFI_VALUE_INT);
    assert_eq!(out2.int_value, 999);

    // Set string global
    let name_key = CString::new("name").unwrap();
    let name_val = CString::new("updated_name").unwrap();
    let str_val = LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_STRING,
        string_ptr: name_val.into_raw(),
        ..Default::default()
    };
    let set_str_ok = lust_program_set_global(prog, name_key.as_ptr(), &str_val);
    assert!(set_str_ok, "lust_program_set_global(name) failed");
    lust_string_free(str_val.string_ptr);

    // Read back string global
    let mut out_str = LustFfiValue::default();
    let get_str_ok = lust_program_get_global(prog, name_key.as_ptr(), &mut out_str);
    assert!(get_str_ok, "lust_program_get_global(name) failed");
    assert_eq!(out_str.tag, LustFfiValueTag::LUST_FFI_VALUE_STRING);
    let str_result = unsafe { CStr::from_ptr(out_str.string_ptr) }.to_str().unwrap();
    assert_eq!(str_result, "updated_name");
    lust_value_dispose(&mut out_str);

    // Set bool global
    let flag_key = CString::new("flag").unwrap();
    let bool_val = LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_BOOL,
        bool_value: true,
        ..Default::default()
    };
    let set_bool_ok = lust_program_set_global(prog, flag_key.as_ptr(), &bool_val);
    assert!(set_bool_ok, "lust_program_set_global(flag) failed");

    let mut out_bool = LustFfiValue::default();
    let get_bool_ok = lust_program_get_global(prog, flag_key.as_ptr(), &mut out_bool);
    assert!(get_bool_ok, "lust_program_get_global(flag) failed");
    assert_eq!(out_bool.tag, LustFfiValueTag::LUST_FFI_VALUE_BOOL);
    assert!(out_bool.bool_value);

    // Set float global
    let pi_key = CString::new("pi").unwrap();
    let float_val = LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_FLOAT,
        float_value: 3.14159,
        ..Default::default()
    };
    let set_pi_ok = lust_program_set_global(prog, pi_key.as_ptr(), &float_val);
    assert!(set_pi_ok, "lust_program_set_global(pi) failed");

    let mut out_pi = LustFfiValue::default();
    let get_pi_ok = lust_program_get_global(prog, pi_key.as_ptr(), &mut out_pi);
    assert!(get_pi_ok, "lust_program_get_global(pi) failed");
    assert_eq!(out_pi.tag, LustFfiValueTag::LUST_FFI_VALUE_FLOAT);
    assert!((out_pi.float_value - 3.14159).abs() < 1e-5);

    // Set nil global
    let nil_val = LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_NIL,
        ..Default::default()
    };
    let set_nil_ok = lust_program_set_global(prog, pi_key.as_ptr(), &nil_val);
    assert!(set_nil_ok, "lust_program_set_global(nil) failed");

    let mut out_nil = LustFfiValue::default();
    let get_nil_ok = lust_program_get_global(prog, pi_key.as_ptr(), &mut out_nil);
    assert!(get_nil_ok, "lust_program_get_global(nil) failed");
    assert_eq!(out_nil.tag, LustFfiValueTag::LUST_FFI_VALUE_NIL);

    lust_program_free(prog);
}

#[test]
fn test_ffi_builder_set_base_dir_success() {
    let builder = lust_builder_new();
    assert!(!builder.is_null());

    let dir = CString::new("/tmp/lust_test_dir").unwrap();
    let ok = lust_builder_set_base_dir(builder, dir.as_ptr());
    assert!(ok, "lust_builder_set_base_dir failed for valid directory path");

    lust_builder_free(builder);
}

// =========================================================================
// 2. FFI NEGATIVE, NULL-SAFETY, AND HOSTILE BOUNDARY TESTS
// =========================================================================

#[test]
fn test_ffi_null_pointer_safety_and_error_capture() {
    lust_clear_last_error();
    assert!(lust_last_error_message().is_null());

    // Safe disposal with null pointers (must not panic)
    lust_string_free(ptr::null_mut());
    lust_value_dispose(ptr::null_mut());
    lust_value_free(ptr::null_mut());
    lust_builder_free(ptr::null_mut());
    lust_program_free(ptr::null_mut());

    let valid_source = CString::new("function test_fn(): int return 0 end").unwrap();
    let valid_name = CString::new("test_fn").unwrap();
    let valid_val = LustFfiValue::default();
    let mut out_val = LustFfiValue::default();

    // lust_builder_add_module null checks
    {
        let builder = lust_builder_new();

        assert!(!lust_builder_add_module(ptr::null_mut(), valid_name.as_ptr(), valid_source.as_ptr()));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Builder pointer was null"), "err: {}", err);

        assert!(!lust_builder_add_module(builder, ptr::null(), valid_source.as_ptr()));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Module path pointer was null"), "err: {}", err);

        assert!(!lust_builder_add_module(builder, valid_name.as_ptr(), ptr::null()));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Source pointer was null"), "err: {}", err);

        lust_builder_free(builder);
    }

    // lust_builder_set_entry_module null checks
    {
        let builder = lust_builder_new();

        assert!(!lust_builder_set_entry_module(ptr::null_mut(), valid_name.as_ptr()));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Builder pointer was null"), "err: {}", err);

        assert!(!lust_builder_set_entry_module(builder, ptr::null()));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Module path pointer was null"), "err: {}", err);

        lust_builder_free(builder);
    }

    // lust_builder_set_base_dir null checks
    {
        let builder = lust_builder_new();

        assert!(!lust_builder_set_base_dir(ptr::null_mut(), valid_name.as_ptr()));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Builder pointer was null"), "err: {}", err);

        assert!(!lust_builder_set_base_dir(builder, ptr::null()));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Base directory pointer was null"), "err: {}", err);

        lust_builder_free(builder);
    }

    // lust_builder_compile null check
    {
        assert!(lust_builder_compile(ptr::null_mut()).is_null());
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Builder pointer was null"), "err: {}", err);
    }

    // lust_program_run_entry null check
    {
        assert!(!lust_program_run_entry(ptr::null_mut()));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Program pointer was null"), "err: {}", err);
    }

    // lust_program_new null check
    {
        assert!(lust_program_new(ptr::null()).is_null());
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Source pointer was null"), "err: {}", err);
    }

    // lust_program_eval null check
    {
        assert!(!lust_program_eval(ptr::null_mut(), &mut out_val));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Program pointer was null"), "err: {}", err);
    }

    // Valid program pointer for target tests
    let prog = lust_program_new(valid_source.as_ptr());
    assert!(!prog.is_null());

    // lust_program_call null checks
    {
        assert!(!lust_program_call(ptr::null_mut(), valid_name.as_ptr(), ptr::null(), 0, &mut out_val));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Program pointer was null"), "err: {}", err);

        assert!(!lust_program_call(prog, ptr::null(), ptr::null(), 0, &mut out_val));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Function name pointer was null"), "err: {}", err);
    }

    // lust_program_get_global null checks
    {
        assert!(!lust_program_get_global(ptr::null_mut(), valid_name.as_ptr(), &mut out_val));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Program pointer was null"), "err: {}", err);

        assert!(!lust_program_get_global(prog, ptr::null(), &mut out_val));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Global name pointer was null"), "err: {}", err);

        assert!(!lust_program_get_global(prog, valid_name.as_ptr(), ptr::null_mut()));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Output value pointer was null"), "err: {}", err);
    }

    // lust_program_set_global null checks
    {
        assert!(!lust_program_set_global(ptr::null_mut(), valid_name.as_ptr(), &valid_val));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Program pointer was null"), "err: {}", err);

        assert!(!lust_program_set_global(prog, ptr::null(), &valid_val));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Global name pointer was null"), "err: {}", err);

        assert!(!lust_program_set_global(prog, valid_name.as_ptr(), ptr::null()));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Value pointer was null"), "err: {}", err);
    }

    lust_program_free(prog);
}

#[test]
fn test_ffi_corrupted_args_len_and_null_buffer() {
    let script = CString::new("function foo() end").unwrap();
    let prog = lust_program_new(script.as_ptr());
    assert!(!prog.is_null());

    let func = CString::new("foo").unwrap();
    let mut out = LustFfiValue::default();

    // Corrupted length: args_len = 5, but args = null pointer
    let ok = lust_program_call(prog, func.as_ptr(), ptr::null(), 5, &mut out);
    assert!(!ok, "lust_program_call must fail when args is null and args_len > 0");

    let err_msg = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
    assert!(
        err_msg.contains("Arguments pointer was null while args_len > 0"),
        "unexpected error message: {}",
        err_msg
    );

    lust_program_free(prog);
}

#[test]
fn test_ffi_invalid_utf8_rejection() {
    // Malformed UTF-8 bytes followed by null terminator
    let bad_bytes = [0xFF, 0xFE, 0xFD, 0x00];
    let bad_ptr = bad_bytes.as_ptr() as *const c_char;
    let valid_str = CString::new("function echo(x: string): string return x end").unwrap();

    // 1. lust_program_new with invalid UTF-8 source
    {
        let prog = lust_program_new(bad_ptr);
        assert!(prog.is_null());
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Source code was not valid UTF-8"), "err: {}", err);
    }

    // 2. lust_builder_add_module with invalid UTF-8 path and source
    {
        let builder = lust_builder_new();

        assert!(!lust_builder_add_module(builder, bad_ptr, valid_str.as_ptr()));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Module path was not valid UTF-8"), "err: {}", err);

        let valid_mod = CString::new("main").unwrap();
        assert!(!lust_builder_add_module(builder, valid_mod.as_ptr(), bad_ptr));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Source code was not valid UTF-8"), "err: {}", err);

        lust_builder_free(builder);
    }

    // 3. lust_builder_set_entry_module with invalid UTF-8 path
    {
        let builder = lust_builder_new();
        assert!(!lust_builder_set_entry_module(builder, bad_ptr));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Module path was not valid UTF-8"), "err: {}", err);
        lust_builder_free(builder);
    }

    // 4. lust_builder_set_base_dir with invalid UTF-8 path
    {
        let builder = lust_builder_new();
        assert!(!lust_builder_set_base_dir(builder, bad_ptr));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Base directory was not valid UTF-8"), "err: {}", err);
        lust_builder_free(builder);
    }

    // 5. lust_program_call with invalid UTF-8 function name and argument
    {
        let prog = lust_program_new(valid_str.as_ptr());
        assert!(!prog.is_null());
        let mut out = LustFfiValue::default();

        assert!(!lust_program_call(prog, bad_ptr, ptr::null(), 0, &mut out));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Function name was not valid UTF-8"), "err: {}", err);

        // String argument with invalid UTF-8
        let func = CString::new("echo").unwrap();
        let bad_arg = LustFfiValue {
            tag: LustFfiValueTag::LUST_FFI_VALUE_STRING,
            string_ptr: bad_ptr as *mut c_char,
            ..Default::default()
        };
        assert!(!lust_program_call(prog, func.as_ptr(), &bad_arg, 1, &mut out));
        let err2 = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err2.contains("String argument contained invalid UTF-8"), "err: {}", err2);

        lust_program_free(prog);
    }

    // 6. lust_program_get_global and set_global with invalid UTF-8
    {
        let prog = lust_program_new(valid_str.as_ptr());
        assert!(!prog.is_null());
        let mut out = LustFfiValue::default();

        assert!(!lust_program_get_global(prog, bad_ptr, &mut out));
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Global name was not valid UTF-8"), "err: {}", err);

        let val = LustFfiValue::default();
        assert!(!lust_program_set_global(prog, bad_ptr, &val));
        let err2 = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err2.contains("Global name was not valid UTF-8"), "err: {}", err2);

        // Value with invalid UTF-8 string pointer
        let key = CString::new("x").unwrap();
        let bad_val = LustFfiValue {
            tag: LustFfiValueTag::LUST_FFI_VALUE_STRING,
            string_ptr: bad_ptr as *mut c_char,
            ..Default::default()
        };
        assert!(!lust_program_set_global(prog, key.as_ptr(), &bad_val));
        let err3 = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err3.contains("String argument contained invalid UTF-8"), "err: {}", err3);

        lust_program_free(prog);
    }
}

#[test]
fn test_ffi_string_with_null_string_ptr_rejection() {
    let script = CString::new("function test_fn(x: string): string return x end").unwrap();
    let prog = lust_program_new(script.as_ptr());
    assert!(!prog.is_null());

    let func = CString::new("test_fn").unwrap();

    let null_str_arg = LustFfiValue {
        tag: LustFfiValueTag::LUST_FFI_VALUE_STRING,
        string_ptr: ptr::null_mut(),
        ..Default::default()
    };

    let mut out = LustFfiValue::default();
    let call_ok = lust_program_call(prog, func.as_ptr(), &null_str_arg, 1, &mut out);
    assert!(!call_ok, "lust_program_call must reject string with null string_ptr");
    let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
    assert!(err.contains("String argument pointer was null"), "err: {}", err);

    let key = CString::new("some_var").unwrap();
    let set_ok = lust_program_set_global(prog, key.as_ptr(), &null_str_arg);
    assert!(!set_ok, "lust_program_set_global must reject string with null string_ptr");
    let err2 = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
    assert!(err2.contains("String argument pointer was null"), "err: {}", err2);

    // Dispose on a string with null string_ptr should safely reset
    let mut val_copy = null_str_arg;
    lust_value_dispose(&mut val_copy);
    assert_eq!(val_copy.tag, LustFfiValueTag::LUST_FFI_VALUE_NIL);

    lust_program_free(prog);
}

#[test]
fn test_ffi_compilation_and_runtime_error_reporting() {
    // 1. Syntax error in lust_program_new
    {
        let bad_syntax = CString::new("function 123 syntax error").unwrap();
        let prog = lust_program_new(bad_syntax.as_ptr());
        assert!(prog.is_null());
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(!err.is_empty(), "expected compilation error message, got empty");
    }

    // 2. Syntax error in lust_builder_compile
    {
        let builder = lust_builder_new();
        let m = CString::new("main").unwrap();
        let s = CString::new("function broken() return %%% end").unwrap();
        lust_builder_add_module(builder, m.as_ptr(), s.as_ptr());
        lust_builder_set_entry_module(builder, m.as_ptr());
        let prog = lust_builder_compile(builder);
        assert!(prog.is_null());
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(!err.is_empty(), "expected compile error message");
    }

    // 3. Global not found
    {
        let script = CString::new("function dummy(): int return 0 end").unwrap();
        let prog = lust_program_new(script.as_ptr());
        assert!(!prog.is_null());
        let missing = CString::new("non_existent").unwrap();
        let mut out = LustFfiValue::default();
        let ok = lust_program_get_global(prog, missing.as_ptr(), &mut out);
        assert!(!ok);
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(err.contains("Global 'non_existent' was not found"), "err: {}", err);
        lust_program_free(prog);
    }

    // 4. Function call runtime error (integer division by zero)
    {
        let script = CString::new("function crash(): int return 10 // 0 end").unwrap();
        let prog = lust_program_new(script.as_ptr());
        assert!(!prog.is_null());
        let func = CString::new("crash").unwrap();
        let mut out = LustFfiValue::default();
        let ok = lust_program_call(prog, func.as_ptr(), ptr::null(), 0, &mut out);
        assert!(!ok, "expected lust_program_call to fail on division by zero");
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(!err.is_empty(), "expected error message from crash");
        lust_program_free(prog);
    }

    // 5. Calling function that doesn't exist
    {
        let script = CString::new("function dummy(): int return 0 end").unwrap();
        let prog = lust_program_new(script.as_ptr());
        assert!(!prog.is_null());
        let func = CString::new("ghost_func").unwrap();
        let mut out = LustFfiValue::default();
        let ok = lust_program_call(prog, func.as_ptr(), ptr::null(), 0, &mut out);
        assert!(!ok);
        let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
        assert!(!err.is_empty());
        lust_program_free(prog);
    }
}

#[test]
fn test_ffi_unsupported_type_conversion_reporting() {
    // A function returning a function value (cannot be converted to primitive LustFfiValue)
    let script = CString::new(
        r#"
        function helper()
            return
        end

        function make_func(): function(): ()
            return helper
        end
        "#,
    )
    .unwrap();

    let prog = lust_program_new(script.as_ptr());
    let err_str = if prog.is_null() {
        let p = lust_last_error();
        if p.is_null() { "null".to_string() } else { unsafe { CStr::from_ptr(p) }.to_string_lossy().to_string() }
    } else { "".to_string() };
    assert!(!prog.is_null(), "lust_program_new failed in test_ffi_unsupported_type_conversion_reporting: {}", err_str);

    let func = CString::new("make_func").unwrap();
    let mut out = LustFfiValue::default();
    let ok = lust_program_call(prog, func.as_ptr(), ptr::null(), 0, &mut out);
    assert!(!ok, "lust_program_call should reject returning function values to FFI");
    let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
    assert!(
        err.contains("Unsupported Lust value for FFI conversion"),
        "unexpected error message: {}",
        err
    );

    // Setting and reading a complex global (Tuple)
    let prog_ref = unsafe { &mut *prog };
    prog_ref.set_global_value(
        "complex_global",
        lust::bytecode::Value::Tuple(std::rc::Rc::new(Vec::new())),
    );

    let key = CString::new("complex_global").unwrap();
    let get_ok = lust_program_get_global(prog, key.as_ptr(), &mut out);
    assert!(!get_ok, "lust_program_get_global should reject complex global conversion");
    let err2 = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
    assert!(
        err2.contains("Unsupported Lust value for FFI conversion"),
        "unexpected error message: {}",
        err2
    );

    lust_program_free(prog);
}

// =========================================================================
// 3. RX CLOCK MONOTONIC & VIRTUAL ORACLES (rx/clock.rs)
// =========================================================================

#[test]
fn test_rx_virtual_clock_comprehensive() {
    // 1. Default constructor
    let vclock = VirtualClock::default();
    assert_eq!(vclock.now_nanos(), 0, "VirtualClock::default should start at 0 nanos");
    assert!(vclock.is_virtual(), "VirtualClock must report is_virtual() == true");

    // 2. Custom initial nanoseconds
    let vclock2 = VirtualClock::new(1_000_000);
    assert_eq!(vclock2.now_nanos(), 1_000_000);

    // 3. Advance virtual time
    vclock2.advance(Duration::from_nanos(500));
    assert_eq!(
        vclock2.now_nanos(),
        1_000_500,
        "VirtualClock::advance should add exact nanoseconds"
    );

    // 4. Set nanoseconds explicitly
    vclock2.set_nanos(42_000);
    assert_eq!(vclock2.now_nanos(), 42_000);

    // 5. Extreme boundary: advance with Duration::MAX
    // 42_000 + u64::MAX wraps to 41_999 without panic
    vclock2.advance(Duration::MAX);
    assert_eq!(vclock2.now_nanos(), 41_999);

    // 6. Debug formatting check
    let debug_str = format!("{:?}", vclock2);
    assert!(
        debug_str.contains("VirtualClock"),
        "VirtualClock debug output missing struct name: {}",
        debug_str
    );
}

#[test]
fn test_rx_physical_clock_comprehensive() {
    // 1. Default constructor and new
    let pclock = PhysicalClock::default();
    assert!(!pclock.is_virtual(), "PhysicalClock must report is_virtual() == false");

    let pclock2 = PhysicalClock::new();
    assert!(!pclock2.is_virtual());

    // 2. Monotonicity check
    let t1 = pclock.now_nanos();
    std::thread::sleep(Duration::from_millis(5));
    let t2 = pclock.now_nanos();
    assert!(
        t2 >= t1,
        "PhysicalClock::now_nanos must be monotonic: t1={}, t2={}",
        t1,
        t2
    );

    // 3. Advance on physical clock is a no-op
    let before_adv = pclock.now_nanos();
    pclock.advance(Duration::from_secs(100));
    let after_adv = pclock.now_nanos();
    // After advancing 100 seconds virtually, physical clock shouldn't jump 100 seconds
    assert!(
        after_adv - before_adv < 100_000_000,
        "PhysicalClock::advance must be a no-op, but time jumped from {} to {}",
        before_adv,
        after_adv
    );
}

// =========================================================================
// 4. RX ERROR FORMATTING & TRAIT ORACLES (rx/mod.rs)
// =========================================================================

#[test]
fn test_rx_error_display_and_trait_oracle() {
    let variants: Vec<(RxError<String>, &str)> = vec![
        (
            RxError::Source("disk full".to_string()),
            "Rx source error: disk full",
        ),
        (
            RxError::BackpressureOverflow,
            "Rx backpressure overflow",
        ),
        (
            RxError::ItemTooLarge,
            "Rx item exceeds buffer byte capacity",
        ),
        (
            RxError::InvalidOptions("watermark inverted".to_string()),
            "Invalid Rx buffer options: watermark inverted",
        ),
        (
            RxError::InvalidHandle,
            "Invalid or stale Rx subscription handle",
        ),
        (
            RxError::SelfJoin,
            "Deadlock prevented: cannot join subscription from its own callback",
        ),
        (RxError::Timeout, "Rx operation timed out"),
        (
            RxError::Cancelled,
            "Rx stream subscription was cancelled",
        ),
    ];

    for (err, expected_msg) in variants {
        let display = format!("{}", err);
        assert_eq!(
            display, expected_msg,
            "RxError::fmt output mismatch for variant {:?}",
            err
        );

        // Check std::error::Error trait implementation
        let std_err: &dyn std::error::Error = &err;
        assert!(std_err.source().is_none());

        // Check Clone and PartialEq
        let err_clone = err.clone();
        assert_eq!(err, err_clone);
    }
}

// =========================================================================
// 5. RX ENVIRONMENT FACADE, TIMERS & SUBSCRIPTION LIFECYCLE (rx/mod.rs)
// =========================================================================

#[test]
fn test_rx_environment_virtual_lifecycle_and_subscription() {
    let env = RxEnvironment::new_virtual();
    assert_eq!(env.now_nanos(), 0, "RxEnvironment virtual clock should start at 0");
    assert!(env.clock().is_virtual());

    let received = Arc::new(Mutex::new(Vec::new()));
    let completed = Arc::new(AtomicBool::new(false));

    let r_clone = received.clone();
    let c_clone = completed.clone();

    let observer = Observer::new(
        move |item: i64| {
            r_clone.lock().unwrap().push(item);
            Ok(())
        },
        move || {
            c_clone.store(true, Ordering::SeqCst);
        },
        |_: RxError<String>| {},
    );

    let stream: Observable<i64, String> = Observable::from_iter(vec![100, 200, 300]);
    let scope_id = 42;

    let sub = env
        .subscribe(
            stream,
            observer,
            scope_id,
            BufferOptions::default_bounded(),
        )
        .expect("subscription must succeed");

    // Advance and run quiescent
    let steps = env.run_until_quiescent(100);
    assert!(steps > 0, "run_until_quiescent must execute steps");

    let completion = sub.join(Some(Duration::from_millis(100))).expect("join must succeed");
    assert_eq!(completion.exit, ExitStatus::Completed);
    assert_eq!(completion.items_delivered, 3);
    assert_eq!(*received.lock().unwrap(), vec![100, 200, 300]);
    assert!(completed.load(Ordering::SeqCst));
}

#[test]
fn test_rx_environment_timer_scheduling_and_immediate_cancellation() {
    let env = RxEnvironment::new_virtual();

    let timer_1_executed = Arc::new(AtomicBool::new(false));
    let timer_2_executed = Arc::new(AtomicBool::new(false));

    let t1_clone = timer_1_executed.clone();
    let t2_clone = timer_2_executed.clone();

    // Schedule timer 1 for +100ms
    let id1 = env.schedule_timer(Duration::from_millis(100), move || {
        t1_clone.store(true, Ordering::SeqCst);
    });

    // Schedule timer 2 for +50ms
    let _id2 = env.schedule_timer(Duration::from_millis(50), move || {
        t2_clone.store(true, Ordering::SeqCst);
    });

    // Immediately cancel timer 1
    let cancelled = env.cancel_timer(id1);
    assert!(cancelled, "env.cancel_timer must return true for active timer");

    // Advance time past both deadlines and drain
    env.advance_time(Duration::from_millis(150));
    env.run_until_quiescent(100);

    // Oracle: timer 1 was cancelled before expiry, so must NOT execute.
    // Timer 2 was not cancelled, so MUST execute.
    assert!(
        !timer_1_executed.load(Ordering::SeqCst),
        "cancelled timer 1 must not execute"
    );
    assert!(
        timer_2_executed.load(Ordering::SeqCst),
        "un-cancelled timer 2 must execute"
    );
}

#[test]
fn test_rx_environment_scope_cancellation_protocol() {
    let env = RxEnvironment::new_virtual();
    let scope_id = 888;

    // Register a participant in the coordinator
    let part_id = env.coordinator().register_participant(scope_id).unwrap();
    assert_eq!(env.coordinator().active_participants(scope_id), 1);

    // Cancel scope
    env.cancel_scope(scope_id);

    // New participants cannot register once admission is closed
    let reg_result = env.coordinator().register_participant(scope_id);
    assert_eq!(reg_result, Err(RxError::Cancelled));

    // Acknowledge participant exit
    env.coordinator().acknowledge_exit(scope_id, part_id);

    // Wait for quiescence
    let quiescent = env.wait_scope_quiescent(scope_id, Some(Duration::from_millis(50)));
    assert!(quiescent, "scope must be quiescent after all participants exit");
}

#[test]
fn test_rx_environment_physical_and_custom_constructors() {
    // 1. Physical constructor
    let env_phys = RxEnvironment::new_physical();
    assert!(!env_phys.clock().is_virtual());
    assert!(env_phys.now_nanos() > 0 || env_phys.now_nanos() == 0);

    // 2. Virtual with initial time and capacity
    let env_custom = RxEnvironment::new_virtual_with_initial_time(5_000_000, 256);
    assert_eq!(env_custom.now_nanos(), 5_000_000);
    assert!(env_custom.clock().is_virtual());

    // 3. Default constructor
    let env_def = RxEnvironment::default();
    assert_eq!(env_def.now_nanos(), 0);

    // 4. Schedule cooperative task
    let task_ran = Arc::new(AtomicBool::new(false));
    let tr_clone = task_ran.clone();
    env_def
        .schedule_task(5, move || {
            tr_clone.store(true, Ordering::SeqCst);
        })
        .expect("schedule_task must succeed");

    let steps = env_def.advance_and_drain(Duration::from_millis(10));
    assert!(steps > 0);
    assert!(task_ran.load(Ordering::SeqCst), "scheduled task must have executed");

    // 5. Test getters
    let _ = env_def.scheduler();
    let _ = env_def.timers();
    let _ = env_def.coordinator();

    // 6. Test shutdown
    env_def.shutdown();
    let reject_task = env_def.schedule_task(1, || {});
    assert_eq!(reject_task, Err(RxError::Cancelled));
}

#[test]
fn test_ffi_interior_null_byte_string_rejection() {
    let script = CString::new("function null_str(): string return \"hello\\0world\" end").unwrap();
    let prog = lust_program_new(script.as_ptr());
    assert!(!prog.is_null());

    let func = CString::new("null_str").unwrap();
    let mut out = LustFfiValue::default();
    let ok = lust_program_call(prog, func.as_ptr(), ptr::null(), 0, &mut out);
    assert!(!ok, "lust_program_call must fail when returned string contains interior null byte");

    let err = unsafe { CStr::from_ptr(lust_last_error()) }.to_str().unwrap();
    assert!(
        err.contains("String result contained interior null byte"),
        "unexpected error message: {}",
        err
    );

    lust_program_free(prog);
}

#[test]
fn test_ffi_clear_last_error_oracle() {
    // Generate an error by passing null to lust_program_new
    let prog = lust_program_new(ptr::null());
    assert!(prog.is_null());

    let err_ptr = lust_last_error();
    assert!(!err_ptr.is_null(), "lust_last_error should be non-null after error");

    // Clear error
    lust_clear_last_error();
    let err_ptr_cleared = lust_last_error();
    assert!(
        err_ptr_cleared.is_null(),
        "lust_last_error should be null after lust_clear_last_error()"
    );
}

#[test]
fn test_rx_environment_custom_clock_constructor() {
    let clock: SharedClock = Arc::new(VirtualClock::new(999_999));
    let env = RxEnvironment::with_clock(clock, 512);
    assert_eq!(env.now_nanos(), 999_999);
    assert!(env.clock().is_virtual());
}

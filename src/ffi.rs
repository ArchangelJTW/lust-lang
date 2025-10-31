use crate::bytecode::Value;
use crate::embed::{EmbeddedBuilder, EmbeddedProgram};
use crate::number::{LustFloat, LustInt};
use crate::{LustError, Result};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::slice;
thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<CString>> = std::cell::RefCell::new(None);
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| {
        slot.borrow_mut().take();
    });
}

fn set_last_error(message: impl Into<String>) {
    let msg = message.into();
    LAST_ERROR.with(|slot| {
        let cstring = CString::new(msg.clone()).unwrap_or_else(|_| {
            CString::new("FFI error message contained null byte").expect("static string")
        });
        *slot.borrow_mut() = Some(cstring);
    });
}

fn handle_error(err: LustError) {
    set_last_error(err.to_string());
}

fn handle_result<T>(result: Result<T>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(err) => {
            handle_error(err);
            None
        }
    }
}

#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LustFfiValueTag {
    LUST_FFI_VALUE_NIL = 0,
    LUST_FFI_VALUE_BOOL = 1,
    LUST_FFI_VALUE_INT = 2,
    LUST_FFI_VALUE_FLOAT = 3,
    LUST_FFI_VALUE_STRING = 4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LustFfiValue {
    pub tag: LustFfiValueTag,
    pub bool_value: bool,
    pub int_value: LustInt,
    pub float_value: LustFloat,
    pub string_ptr: *mut c_char,
}

impl Default for LustFfiValue {
    fn default() -> Self {
        Self {
            tag: LustFfiValueTag::LUST_FFI_VALUE_NIL,
            bool_value: false,
            int_value: 0,
            float_value: 0.0,
            string_ptr: ptr::null_mut(),
        }
    }
}

fn value_from_ffi(value: &LustFfiValue) -> Result<Value> {
    match value.tag {
        LustFfiValueTag::LUST_FFI_VALUE_NIL => Ok(Value::Nil),
        LustFfiValueTag::LUST_FFI_VALUE_BOOL => Ok(Value::Bool(value.bool_value)),
        LustFfiValueTag::LUST_FFI_VALUE_INT => Ok(Value::Int(value.int_value)),
        LustFfiValueTag::LUST_FFI_VALUE_FLOAT => Ok(Value::Float(value.float_value)),
        LustFfiValueTag::LUST_FFI_VALUE_STRING => {
            if value.string_ptr.is_null() {
                return Err(LustError::RuntimeError {
                    message: "String argument pointer was null".into(),
                });
            }

            let c_str = unsafe { CStr::from_ptr(value.string_ptr) };
            let string = c_str.to_str().map_err(|_| LustError::RuntimeError {
                message: "String argument contained invalid UTF-8".into(),
            })?;
            Ok(Value::String(std::rc::Rc::new(string.to_owned())))
        }
    }
}

fn value_to_ffi(value: Value) -> Result<LustFfiValue> {
    match value {
        Value::Nil => Ok(LustFfiValue::default()),
        Value::Bool(flag) => Ok(LustFfiValue {
            tag: LustFfiValueTag::LUST_FFI_VALUE_BOOL,
            bool_value: flag,
            ..Default::default()
        }),
        Value::Int(i) => Ok(LustFfiValue {
            tag: LustFfiValueTag::LUST_FFI_VALUE_INT,
            int_value: i,
            ..Default::default()
        }),
        Value::Float(f) => Ok(LustFfiValue {
            tag: LustFfiValueTag::LUST_FFI_VALUE_FLOAT,
            float_value: f,
            ..Default::default()
        }),
        Value::String(s) => {
            let cstring = CString::new(s.as_str()).map_err(|_| LustError::RuntimeError {
                message: "String result contained interior null byte".into(),
            })?;
            let ptr = cstring.into_raw();
            Ok(LustFfiValue {
                tag: LustFfiValueTag::LUST_FFI_VALUE_STRING,
                string_ptr: ptr,
                ..Default::default()
            })
        }

        other => Err(LustError::RuntimeError {
            message: format!("Unsupported Lust value for FFI conversion: {:?}", other),
        }),
    }
}

#[no_mangle]
pub extern "C" fn lust_clear_last_error() {
    clear_last_error();
}

#[no_mangle]
pub extern "C" fn lust_last_error_message() -> *const c_char {
    LAST_ERROR.with(|slot| {
        if let Some(err) = slot.borrow().as_ref() {
            err.as_ptr()
        } else {
            ptr::null()
        }
    })
}

#[no_mangle]
pub extern "C" fn lust_string_free(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }

    unsafe {
        drop(CString::from_raw(ptr));
    }
}

#[no_mangle]
pub extern "C" fn lust_value_dispose(value: *mut LustFfiValue) {
    if value.is_null() {
        return;
    }

    unsafe {
        if (*value).tag == LustFfiValueTag::LUST_FFI_VALUE_STRING && !(*value).string_ptr.is_null()
        {
            drop(CString::from_raw((*value).string_ptr));
        }

        *value = LustFfiValue::default();
    }
}

#[no_mangle]
pub extern "C" fn lust_builder_new() -> *mut EmbeddedBuilder {
    clear_last_error();
    Box::into_raw(Box::new(EmbeddedBuilder::new()))
}

#[no_mangle]
pub extern "C" fn lust_builder_free(builder: *mut EmbeddedBuilder) {
    if builder.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(builder));
    }
}

#[no_mangle]
pub extern "C" fn lust_builder_add_module(
    builder: *mut EmbeddedBuilder,
    module_path: *const c_char,
    source: *const c_char,
) -> bool {
    clear_last_error();
    if builder.is_null() {
        set_last_error("Builder pointer was null");
        return false;
    }

    if module_path.is_null() {
        set_last_error("Module path pointer was null");
        return false;
    }

    if source.is_null() {
        set_last_error("Source pointer was null");
        return false;
    }

    let path = unsafe { CStr::from_ptr(module_path) };
    let source_str = unsafe { CStr::from_ptr(source) };
    let path_str = match path.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("Module path was not valid UTF-8");
            return false;
        }
    };
    let source_str = match source_str.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("Source code was not valid UTF-8");
            return false;
        }
    };
    let builder_ref = unsafe { &mut *builder };
    builder_ref.add_module(path_str, source_str);
    true
}

#[no_mangle]
pub extern "C" fn lust_builder_set_entry_module(
    builder: *mut EmbeddedBuilder,
    module_path: *const c_char,
) -> bool {
    clear_last_error();
    if builder.is_null() {
        set_last_error("Builder pointer was null");
        return false;
    }

    if module_path.is_null() {
        set_last_error("Module path pointer was null");
        return false;
    }

    let path = unsafe { CStr::from_ptr(module_path) };
    let path_str = match path.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("Module path was not valid UTF-8");
            return false;
        }
    };
    let builder_ref = unsafe { &mut *builder };
    builder_ref.set_entry_module(path_str);
    true
}

#[no_mangle]
pub extern "C" fn lust_builder_set_base_dir(
    builder: *mut EmbeddedBuilder,
    base_dir: *const c_char,
) -> bool {
    clear_last_error();
    if builder.is_null() {
        set_last_error("Builder pointer was null");
        return false;
    }

    if base_dir.is_null() {
        set_last_error("Base directory pointer was null");
        return false;
    }

    let base_dir = unsafe { CStr::from_ptr(base_dir) };
    let base_dir_str = match base_dir.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("Base directory was not valid UTF-8");
            return false;
        }
    };
    let builder_ref = unsafe { &mut *builder };
    builder_ref.set_entry_module(base_dir_str);
    true
}

#[no_mangle]
pub extern "C" fn lust_builder_compile(builder: *mut EmbeddedBuilder) -> *mut EmbeddedProgram {
    clear_last_error();
    if builder.is_null() {
        set_last_error("Builder pointer was null");
        return ptr::null_mut();
    }

    let builder_box = unsafe { Box::from_raw(builder) };
    match builder_box.compile() {
        Ok(program) => Box::into_raw(Box::new(program)),
        Err(err) => {
            handle_error(err);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn lust_program_free(program: *mut EmbeddedProgram) {
    if program.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(program));
    }
}

#[no_mangle]
pub extern "C" fn lust_program_run_entry(program: *mut EmbeddedProgram) -> bool {
    clear_last_error();
    if program.is_null() {
        set_last_error("Program pointer was null");
        return false;
    }

    let program_ref = unsafe { &mut *program };
    handle_result(program_ref.run_entry_script()).is_some()
}

#[no_mangle]
pub extern "C" fn lust_program_call(
    program: *mut EmbeddedProgram,
    function_name: *const c_char,
    args: *const LustFfiValue,
    args_len: usize,
    out_value: *mut LustFfiValue,
) -> bool {
    clear_last_error();
    if program.is_null() {
        set_last_error("Program pointer was null");
        return false;
    }

    if function_name.is_null() {
        set_last_error("Function name pointer was null");
        return false;
    }

    if args_len > 0 && args.is_null() {
        set_last_error("Arguments pointer was null while args_len > 0");
        return false;
    }

    let func_name = unsafe { CStr::from_ptr(function_name) };
    let func_name = match func_name.to_str() {
        Ok(name) => name.to_owned(),
        Err(_) => {
            set_last_error("Function name was not valid UTF-8");
            return false;
        }
    };
    let arg_slice = if args_len == 0 {
        &[][..]
    } else {
        unsafe { slice::from_raw_parts(args, args_len) }
    };
    let mut converted_args = Vec::with_capacity(arg_slice.len());
    for arg in arg_slice {
        match value_from_ffi(arg) {
            Ok(value) => converted_args.push(value),
            Err(err) => {
                handle_error(err);
                return false;
            }
        }
    }

    let program_ref = unsafe { &mut *program };
    let result = match program_ref.call_raw(&func_name, converted_args) {
        Ok(value) => value,
        Err(err) => {
            handle_error(err);
            return false;
        }
    };
    if out_value.is_null() {
        return true;
    }

    match value_to_ffi(result) {
        Ok(converted) => {
            unsafe {
                *out_value = converted;
            }

            true
        }

        Err(err) => {
            handle_error(err);
            false
        }
    }
}

#[no_mangle]
pub extern "C" fn lust_program_get_global(
    program: *mut EmbeddedProgram,
    name: *const c_char,
    out_value: *mut LustFfiValue,
) -> bool {
    clear_last_error();
    if program.is_null() {
        set_last_error("Program pointer was null");
        return false;
    }

    if name.is_null() {
        set_last_error("Global name pointer was null");
        return false;
    }

    if out_value.is_null() {
        set_last_error("Output value pointer was null");
        return false;
    }

    let name_cstr = unsafe { CStr::from_ptr(name) };
    let name_str = match name_cstr.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("Global name was not valid UTF-8");
            return false;
        }
    };

    let program_ref = unsafe { &mut *program };
    let value = match program_ref.get_global_value(name_str) {
        Some(value) => value,
        None => {
            set_last_error(format!("Global '{}' was not found", name_str));
            return false;
        }
    };

    match value_to_ffi(value) {
        Ok(converted) => {
            unsafe {
                *out_value = converted;
            }

            true
        }
        Err(err) => {
            handle_error(err);
            false
        }
    }
}

#[no_mangle]
pub extern "C" fn lust_program_set_global(
    program: *mut EmbeddedProgram,
    name: *const c_char,
    value: *const LustFfiValue,
) -> bool {
    clear_last_error();
    if program.is_null() {
        set_last_error("Program pointer was null");
        return false;
    }

    if name.is_null() {
        set_last_error("Global name pointer was null");
        return false;
    }

    if value.is_null() {
        set_last_error("Value pointer was null");
        return false;
    }

    let name_cstr = unsafe { CStr::from_ptr(name) };
    let name_str = match name_cstr.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("Global name was not valid UTF-8");
            return false;
        }
    };

    let value_ref = unsafe { &*value };
    let converted = match value_from_ffi(value_ref) {
        Ok(val) => val,
        Err(err) => {
            handle_error(err);
            return false;
        }
    };

    let program_ref = unsafe { &mut *program };
    program_ref.set_global_value(name_str, converted);
    true
}
// #![cfg(feature = "std")]

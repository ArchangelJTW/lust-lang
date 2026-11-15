#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::bytecode::{LustMap, NativeCallResult, Value, ValueKey};
use crate::embed::{EmbeddedBuilder, EmbeddedProgram};
use crate::number::{LustFloat, LustInt};
use crate::{LustError, Result};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::rc::{Rc, Weak};
use std::slice;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
    static PROGRAM_TOKENS: RefCell<HashMap<usize, (u64, Rc<()>)>> =
        RefCell::new(HashMap::new());
    static NEXT_PROGRAM_ID: Cell<u64> = const { Cell::new(1) };
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

pub fn register_program_token(program_ptr: *mut EmbeddedProgram) -> (u64, Rc<()>) {
    let id = NEXT_PROGRAM_ID.with(|cell| {
        let cur = cell.get();
        cell.set(cur.wrapping_add(1));
        cur
    });
    let token = Rc::new(());
    let pair = (id, token.clone());
    PROGRAM_TOKENS.with(|map| {
        map.borrow_mut().insert(program_ptr as usize, pair);
    });
    (id, token)
}

pub fn unregister_program_token(program_ptr: *mut EmbeddedProgram) {
    PROGRAM_TOKENS.with(|map| {
        map.borrow_mut().remove(&(program_ptr as usize));
    });
}

pub fn get_program_token(program_ptr: *mut EmbeddedProgram) -> Option<(u64, Rc<()>)> {
    PROGRAM_TOKENS.with(|map| map.borrow().get(&(program_ptr as usize)).cloned())
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
    LUST_FFI_VALUE_HANDLE = 5,
}

#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LustFfiResult {
    LUST_FFI_OK = 0,
    LUST_FFI_ERR_INVALID_HANDLE = 1,
    LUST_FFI_ERR_STALE_HANDLE = 2,
    LUST_FFI_ERR_TYPE_MISMATCH = 3,
    LUST_FFI_ERR_INDEX_OUT_OF_BOUNDS = 4,
    LUST_FFI_ERR_OUT_OF_MEMORY = 5,
    LUST_FFI_ERR_PROGRAM_MISMATCH = 6,
}

#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LustFfiHandleKind {
    LUST_HANDLE_INVALID = 0,
    LUST_HANDLE_ARRAY = 1,
    LUST_HANDLE_MAP = 2,
    LUST_HANDLE_STRUCT = 3,
}

pub const LUST_HANDLE_MAGIC: u32 = 0x4C555354; // 'LUST'

#[repr(C)]
pub struct LustHandle {
    pub magic: u32,
    pub program_id: u64,
    pub weak_token: Weak<()>,
    pub kind: LustFfiHandleKind,
    pub value: Value,
    pub ref_count: usize,
    pub cached_name: Option<CString>,
    pub charged_capacity_bytes: usize,
}

impl LustHandle {
    pub fn is_valid(&self) -> bool {
        self.magic == LUST_HANDLE_MAGIC && self.weak_token.upgrade().is_some()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LustFfiValue {
    pub tag: LustFfiValueTag,
    pub bool_value: bool,
    pub int_value: LustInt,
    pub float_value: LustFloat,
    pub string_ptr: *mut c_char,
    pub handle_ptr: *mut LustHandle,
}

impl Default for LustFfiValue {
    fn default() -> Self {
        Self {
            tag: LustFfiValueTag::LUST_FFI_VALUE_NIL,
            bool_value: false,
            int_value: 0,
            float_value: 0.0,
            string_ptr: ptr::null_mut(),
            handle_ptr: ptr::null_mut(),
        }
    }
}

fn live_handles() -> &'static std::sync::Mutex<std::collections::HashSet<usize>> {
    static LIVE_HANDLES: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<usize>>> =
        std::sync::OnceLock::new();
    LIVE_HANDLES.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()))
}

fn register_live_handle(handle: Box<LustHandle>) -> *mut LustHandle {
    let ptr = Box::into_raw(handle);
    let mut guard = live_handles().lock().unwrap();
    guard.insert(ptr as usize);
    ptr
}

fn with_live_handle<R>(
    handle: *mut LustHandle,
    f: impl FnOnce(&mut LustHandle) -> std::result::Result<R, LustFfiResult>,
) -> std::result::Result<R, LustFfiResult> {
    if handle.is_null() {
        return Err(LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE);
    }
    let guard = live_handles().lock().unwrap();
    if !guard.contains(&(handle as usize)) {
        return Err(LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE);
    }
    let h = unsafe { &mut *handle };
    if h.magic != LUST_HANDLE_MAGIC || h.ref_count == 0 {
        return Err(LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE);
    }
    if !h.is_valid() {
        return Err(LustFfiResult::LUST_FFI_ERR_STALE_HANDLE);
    }
    f(h)
}

fn with_live_const_handle<R>(
    handle: *const LustHandle,
    f: impl FnOnce(&LustHandle) -> std::result::Result<R, LustFfiResult>,
) -> std::result::Result<R, LustFfiResult> {
    if handle.is_null() {
        return Err(LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE);
    }
    let guard = live_handles().lock().unwrap();
    if !guard.contains(&(handle as usize)) {
        return Err(LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE);
    }
    let h = unsafe { &*handle };
    if h.magic != LUST_HANDLE_MAGIC || h.ref_count == 0 {
        return Err(LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE);
    }
    if !h.is_valid() {
        return Err(LustFfiResult::LUST_FFI_ERR_STALE_HANDLE);
    }
    f(h)
}

pub fn value_from_ffi(value: &LustFfiValue) -> Result<Value> {
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
            Ok(Value::String(Rc::new(string.to_owned())))
        }
        LustFfiValueTag::LUST_FFI_VALUE_HANDLE => {
            if value.handle_ptr.is_null() {
                return Err(LustError::RuntimeError {
                    message: "Handle argument pointer was null".into(),
                });
            }
            let val = with_live_const_handle(value.handle_ptr, |h| Ok(h.value.clone())).map_err(|err| LustError::RuntimeError {
                message: match err {
                    LustFfiResult::LUST_FFI_ERR_STALE_HANDLE => "Stale handle".to_string(),
                    _ => "Invalid handle".to_string(),
                },
            })?;
            Ok(val)
        }
    }
}

pub fn value_to_ffi(value: Value) -> Result<LustFfiValue> {
    value_to_ffi_with_program(ptr::null_mut(), value)
}

pub fn value_to_ffi_with_program(program: *mut EmbeddedProgram, value: Value) -> Result<LustFfiValue> {
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
            let p = cstring.into_raw();
            Ok(LustFfiValue {
                tag: LustFfiValueTag::LUST_FFI_VALUE_STRING,
                string_ptr: p,
                ..Default::default()
            })
        }
        val @ (Value::Array(_) | Value::Map(_) | Value::Struct(_)) => {
            let (kind, cached_name) = match &val {
                Value::Array(_) => (LustFfiHandleKind::LUST_HANDLE_ARRAY, None),
                Value::Map(_) => (LustFfiHandleKind::LUST_HANDLE_MAP, None),
                Value::Struct(obj) => {
                    let c_name = CString::new(obj.name.as_str()).ok();
                    (LustFfiHandleKind::LUST_HANDLE_STRUCT, c_name)
                }
                _ => unreachable!(),
            };

            let (program_id, weak_token) = if !program.is_null() {
                let (id, tok) = get_program_token(program)
                    .unwrap_or_else(|| register_program_token(program));
                (id, Rc::downgrade(&tok))
            } else {
                (0, Weak::new())
            };

            let handle = Box::new(LustHandle {
                magic: LUST_HANDLE_MAGIC,
                program_id,
                weak_token,
                kind,
                value: val,
                ref_count: 1,
                cached_name,
                charged_capacity_bytes: 0,
            });

            Ok(LustFfiValue {
                tag: LustFfiValueTag::LUST_FFI_VALUE_HANDLE,
                handle_ptr: register_live_handle(handle),
                ..Default::default()
            })
        }
        other => Err(LustError::RuntimeError {
            message: format!("Unsupported Lust value for FFI conversion: {:?}", other),
        }),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_clear_last_error() {
    clear_last_error();
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_last_error_message() -> *const c_char {
    LAST_ERROR.with(|slot| {
        if let Some(err) = slot.borrow().as_ref() {
            err.as_ptr()
        } else {
            ptr::null()
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_string_free(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }

    unsafe {
        drop(CString::from_raw(ptr));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_value_dispose(value: *mut LustFfiValue) {
    if value.is_null() {
        return;
    }

    unsafe {
        match (*value).tag {
            LustFfiValueTag::LUST_FFI_VALUE_STRING => {
                if !(*value).string_ptr.is_null() {
                    drop(CString::from_raw((*value).string_ptr));
                    (*value).string_ptr = ptr::null_mut();
                }
            }
            LustFfiValueTag::LUST_FFI_VALUE_HANDLE => {
                if !(*value).handle_ptr.is_null() {
                    lust_handle_release((*value).handle_ptr);
                    (*value).handle_ptr = ptr::null_mut();
                }
            }
            _ => {}
        }
        *value = LustFfiValue::default();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_handle_retain(handle: *mut LustHandle) -> LustFfiResult {
    match with_live_handle(handle, |h| {
        h.ref_count = h.ref_count.saturating_add(1);
        Ok(())
    }) {
        Ok(()) => LustFfiResult::LUST_FFI_OK,
        Err(e) => e,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_handle_release(handle: *mut LustHandle) -> LustFfiResult {
    if handle.is_null() {
        return LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE;
    }
    let mut guard = live_handles().lock().unwrap();
    if !guard.contains(&(handle as usize)) {
        return LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE;
    }
    let h = unsafe { &mut *handle };
    if h.magic != LUST_HANDLE_MAGIC || h.ref_count == 0 {
        return LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE;
    }
    h.ref_count -= 1;
    if h.ref_count == 0 {
        h.magic = 0;
        guard.remove(&(handle as usize));
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
    LustFfiResult::LUST_FFI_OK
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_handle_kind(handle: *const LustHandle) -> LustFfiHandleKind {
    match with_live_const_handle(handle, |h| Ok(h.kind)) {
        Ok(kind) => kind,
        Err(_) => LustFfiHandleKind::LUST_HANDLE_INVALID,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_handle_is_valid(handle: *const LustHandle) -> bool {
    with_live_const_handle(handle, |_| Ok(())).is_ok()
}

// Array handle operations
#[unsafe(no_mangle)]
pub extern "C" fn lust_array_new(
    program: *mut EmbeddedProgram,
    initial_capacity: usize,
    out_handle: *mut *mut LustHandle,
) -> LustFfiResult {
    if out_handle.is_null() {
        return LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE;
    }

    let (program_id, weak_token) = if !program.is_null() {
        let (id, tok) = get_program_token(program).unwrap_or_else(|| register_program_token(program));
        (id, Rc::downgrade(&tok))
    } else {
        (0, Weak::new())
    };

    let arr = Value::Array(Rc::new(RefCell::new(Vec::with_capacity(initial_capacity))));
    let handle = Box::new(LustHandle {
        magic: LUST_HANDLE_MAGIC,
        program_id,
        weak_token,
        kind: LustFfiHandleKind::LUST_HANDLE_ARRAY,
        value: arr,
        ref_count: 1,
        cached_name: None,
        charged_capacity_bytes: 0,
    });

    unsafe {
        *out_handle = register_live_handle(handle);
    }
    LustFfiResult::LUST_FFI_OK
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_array_len(handle: *const LustHandle, out_len: *mut usize) -> LustFfiResult {
    if out_len.is_null() {
        return LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE;
    }
    match with_live_const_handle(handle, |h| {
        if h.kind != LustFfiHandleKind::LUST_HANDLE_ARRAY {
            return Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH);
        }
        match &h.value {
            Value::Array(rc) => {
                unsafe {
                    *out_len = rc.borrow().len();
                }
                Ok(())
            }
            _ => Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH),
        }
    }) {
        Ok(()) => LustFfiResult::LUST_FFI_OK,
        Err(e) => e,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_array_get(
    handle: *const LustHandle,
    index: usize,
    out_value: *mut LustFfiValue,
) -> LustFfiResult {
    if out_value.is_null() {
        return LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE;
    }
    let val = match with_live_const_handle(handle, |h| {
        if h.kind != LustFfiHandleKind::LUST_HANDLE_ARRAY {
            return Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH);
        }
        match &h.value {
            Value::Array(rc) => {
                let vec = rc.borrow();
                if index >= vec.len() {
                    return Err(LustFfiResult::LUST_FFI_ERR_INDEX_OUT_OF_BOUNDS);
                }
                Ok(vec[index].clone())
            }
            _ => Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH),
        }
    }) {
        Ok(v) => v,
        Err(e) => return e,
    };
    match value_to_ffi(val) {
        Ok(converted) => {
            unsafe {
                *out_value = converted;
            }
            LustFfiResult::LUST_FFI_OK
        }
        Err(_) => LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_array_set(
    handle: *mut LustHandle,
    index: usize,
    value: *const LustFfiValue,
) -> LustFfiResult {
    if value.is_null() {
        return LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE;
    }
    let ffi_val = unsafe { &*value };
    let val = match value_from_ffi(ffi_val) {
        Ok(v) => v,
        Err(_) => return LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH,
    };
    match with_live_handle(handle, |h| {
        if h.kind != LustFfiHandleKind::LUST_HANDLE_ARRAY {
            return Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH);
        }
        match &h.value {
            Value::Array(rc) => {
                let mut vec = rc.borrow_mut();
                if index >= vec.len() {
                    return Err(LustFfiResult::LUST_FFI_ERR_INDEX_OUT_OF_BOUNDS);
                }
                vec[index] = val;
                Ok(())
            }
            _ => Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH),
        }
    }) {
        Ok(()) => LustFfiResult::LUST_FFI_OK,
        Err(e) => e,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_array_push(
    handle: *mut LustHandle,
    value: *const LustFfiValue,
) -> LustFfiResult {
    if value.is_null() {
        return LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE;
    }
    let ffi_val = unsafe { &*value };
    let val = match value_from_ffi(ffi_val) {
        Ok(v) => v,
        Err(_) => return LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH,
    };
    match with_live_handle(handle, |h| {
        if h.kind != LustFfiHandleKind::LUST_HANDLE_ARRAY {
            return Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH);
        }
        match &h.value {
            Value::Array(rc) => {
                let mut vec = rc.borrow_mut();
                vec.push(val);
                Ok(())
            }
            _ => Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH),
        }
    }) {
        Ok(()) => LustFfiResult::LUST_FFI_OK,
        Err(e) => e,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_array_pop(
    handle: *mut LustHandle,
    out_value: *mut LustFfiValue,
) -> LustFfiResult {
    let popped = match with_live_handle(handle, |h| {
        if h.kind != LustFfiHandleKind::LUST_HANDLE_ARRAY {
            return Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH);
        }
        match &h.value {
            Value::Array(rc) => rc
                .borrow_mut()
                .pop()
                .ok_or(LustFfiResult::LUST_FFI_ERR_INDEX_OUT_OF_BOUNDS),
            _ => Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH),
        }
    }) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if !out_value.is_null() {
        match value_to_ffi(popped) {
            Ok(converted) => unsafe {
                *out_value = converted;
            },
            Err(_) => return LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH,
        }
    }
    LustFfiResult::LUST_FFI_OK
}

// Map handle operations
#[unsafe(no_mangle)]
pub extern "C" fn lust_map_new(
    program: *mut EmbeddedProgram,
    out_handle: *mut *mut LustHandle,
) -> LustFfiResult {
    if out_handle.is_null() {
        return LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE;
    }

    let (program_id, weak_token) = if !program.is_null() {
        let (id, tok) = get_program_token(program).unwrap_or_else(|| register_program_token(program));
        (id, Rc::downgrade(&tok))
    } else {
        (0, Weak::new())
    };

    let map_val = Value::map(LustMap::default());
    let handle = Box::new(LustHandle {
        magic: LUST_HANDLE_MAGIC,
        program_id,
        weak_token,
        kind: LustFfiHandleKind::LUST_HANDLE_MAP,
        value: map_val,
        ref_count: 1,
        cached_name: None,
        charged_capacity_bytes: 0,
    });

    unsafe {
        *out_handle = register_live_handle(handle);
    }
    LustFfiResult::LUST_FFI_OK
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_map_len(handle: *const LustHandle, out_len: *mut usize) -> LustFfiResult {
    if out_len.is_null() {
        return LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE;
    }
    match with_live_const_handle(handle, |h| {
        if h.kind != LustFfiHandleKind::LUST_HANDLE_MAP {
            return Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH);
        }
        match &h.value {
            Value::Map(rc) => {
                unsafe {
                    *out_len = rc.borrow().len();
                }
                Ok(())
            }
            _ => Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH),
        }
    }) {
        Ok(()) => LustFfiResult::LUST_FFI_OK,
        Err(e) => e,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_map_get(
    handle: *const LustHandle,
    key: *const LustFfiValue,
    out_value: *mut LustFfiValue,
) -> LustFfiResult {
    if key.is_null() || out_value.is_null() {
        return LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE;
    }
    let key_val = match value_from_ffi(unsafe { &*key }) {
        Ok(v) => v,
        Err(_) => return LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH,
    };
    let value_key = ValueKey::from_value(&key_val);

    let val = match with_live_const_handle(handle, |h| {
        if h.kind != LustFfiHandleKind::LUST_HANDLE_MAP {
            return Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH);
        }
        match &h.value {
            Value::Map(rc) => Ok(rc.borrow().get(&value_key).cloned().unwrap_or(Value::Nil)),
            _ => Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH),
        }
    }) {
        Ok(v) => v,
        Err(e) => return e,
    };
    match value_to_ffi(val) {
        Ok(converted) => {
            unsafe {
                *out_value = converted;
            }
            LustFfiResult::LUST_FFI_OK
        }
        Err(_) => LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_map_set(
    handle: *mut LustHandle,
    key: *const LustFfiValue,
    value: *const LustFfiValue,
) -> LustFfiResult {
    if key.is_null() || value.is_null() {
        return LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE;
    }
    let key_val = match value_from_ffi(unsafe { &*key }) {
        Ok(v) => v,
        Err(_) => return LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH,
    };
    let val = match value_from_ffi(unsafe { &*value }) {
        Ok(v) => v,
        Err(_) => return LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH,
    };
    let value_key = ValueKey::from_value(&key_val);

    match with_live_handle(handle, |h| {
        if h.kind != LustFfiHandleKind::LUST_HANDLE_MAP {
            return Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH);
        }
        match &h.value {
            Value::Map(rc) => {
                rc.borrow_mut().insert(value_key, val);
                Ok(())
            }
            _ => Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH),
        }
    }) {
        Ok(()) => LustFfiResult::LUST_FFI_OK,
        Err(e) => e,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_map_remove(
    handle: *mut LustHandle,
    key: *const LustFfiValue,
    out_value: *mut LustFfiValue,
) -> LustFfiResult {
    if key.is_null() {
        return LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE;
    }
    let key_val = match value_from_ffi(unsafe { &*key }) {
        Ok(v) => v,
        Err(_) => return LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH,
    };
    let value_key = ValueKey::from_value(&key_val);

    let removed = match with_live_handle(handle, |h| {
        if h.kind != LustFfiHandleKind::LUST_HANDLE_MAP {
            return Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH);
        }
        match &h.value {
            Value::Map(rc) => Ok(rc.borrow_mut().remove(&value_key)),
            _ => Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH),
        }
    }) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if !out_value.is_null() {
        let val = removed.unwrap_or(Value::Nil);
        match value_to_ffi(val) {
            Ok(converted) => unsafe {
                *out_value = converted;
            },
            Err(_) => return LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH,
        }
    }
    LustFfiResult::LUST_FFI_OK
}

// Struct handle operations
#[unsafe(no_mangle)]
pub extern "C" fn lust_struct_new(
    program: *mut EmbeddedProgram,
    type_name: *const c_char,
    out_handle: *mut *mut LustHandle,
) -> LustFfiResult {
    if program.is_null() || type_name.is_null() || out_handle.is_null() {
        return LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE;
    }

    let c_str = unsafe { CStr::from_ptr(type_name) };
    let type_name_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH,
    };

    let prog = unsafe { &mut *program };
    let struct_val = match prog.vm_mut().instantiate_struct(type_name_str, Vec::new()) {
        Ok(v) => v,
        Err(_) => {
            // Try ordered empty instantiation or fallback
            match prog.vm_mut().instantiate_struct_ordered(type_name_str, Vec::new()) {
                Ok(v) => v,
                Err(_) => return LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH,
            }
        }
    };
    if let Value::Struct(ref obj) = struct_val {
        let count = obj.layout.field_names().len();
        let mut fields = obj.fields.borrow_mut();
        if fields.len() < count {
            fields.resize(count, Value::Nil);
        }
    }

    let (program_id, weak_token) = {
        let (id, tok) = get_program_token(program).unwrap_or_else(|| register_program_token(program));
        (id, Rc::downgrade(&tok))
    };

    let cached_name = CString::new(type_name_str).ok();
    let handle = Box::new(LustHandle {
        magic: LUST_HANDLE_MAGIC,
        program_id,
        weak_token,
        kind: LustFfiHandleKind::LUST_HANDLE_STRUCT,
        value: struct_val,
        ref_count: 1,
        cached_name,
        charged_capacity_bytes: 0,
    });

    unsafe {
        *out_handle = register_live_handle(handle);
    }
    LustFfiResult::LUST_FFI_OK
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_struct_type_name(
    handle: *const LustHandle,
    out_name: *mut *const c_char,
) -> LustFfiResult {
    if out_name.is_null() {
        return LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE;
    }
    match with_live_const_handle(handle, |h| {
        if h.kind != LustFfiHandleKind::LUST_HANDLE_STRUCT {
            return Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH);
        }
        if let Some(c_name) = &h.cached_name {
            unsafe {
                *out_name = c_name.as_ptr();
            }
            Ok(())
        } else {
            match &h.value {
                Value::Struct(obj) => {
                    let name = obj.name.as_str();
                    unsafe {
                        *out_name = concat!("", "\0").as_ptr() as *const c_char;
                    }
                    let _ = name;
                    Ok(())
                }
                _ => Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH),
            }
        }
    }) {
        Ok(()) => LustFfiResult::LUST_FFI_OK,
        Err(e) => e,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_struct_get_field(
    handle: *const LustHandle,
    field_name: *const c_char,
    out_value: *mut LustFfiValue,
) -> LustFfiResult {
    if field_name.is_null() || out_value.is_null() {
        return LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE;
    }
    let field_cstr = unsafe { CStr::from_ptr(field_name) };
    let field_str = match field_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH,
    };

    let val = match with_live_const_handle(handle, |h| {
        if h.kind != LustFfiHandleKind::LUST_HANDLE_STRUCT {
            return Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH);
        }
        match &h.value {
            Value::Struct(obj) => {
                let v = match obj.layout.index_of_str(field_str) {
                    Some(i) => {
                        let fields = obj.fields.borrow();
                        fields.get(i).cloned().unwrap_or(Value::Nil)
                    }
                    None => Value::Nil,
                };
                Ok(v)
            }
            _ => Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH),
        }
    }) {
        Ok(v) => v,
        Err(e) => return e,
    };
    match value_to_ffi(val) {
        Ok(converted) => {
            unsafe {
                *out_value = converted;
            }
            LustFfiResult::LUST_FFI_OK
        }
        Err(_) => LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_struct_set_field(
    handle: *mut LustHandle,
    field_name: *const c_char,
    value: *const LustFfiValue,
) -> LustFfiResult {
    if field_name.is_null() || value.is_null() {
        return LustFfiResult::LUST_FFI_ERR_INVALID_HANDLE;
    }
    let field_cstr = unsafe { CStr::from_ptr(field_name) };
    let field_str = match field_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH,
    };

    let ffi_val = unsafe { &*value };
    let val = match value_from_ffi(ffi_val) {
        Ok(v) => v,
        Err(_) => return LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH,
    };

    match with_live_handle(handle, |h| {
        if h.kind != LustFfiHandleKind::LUST_HANDLE_STRUCT {
            return Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH);
        }
        match &h.value {
            Value::Struct(obj) => {
                let idx = match obj.layout.index_of_str(field_str) {
                    Some(i) => i,
                    None => return Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH),
                };
                let mut fields = obj.fields.borrow_mut();
                if fields.len() < obj.layout.field_names().len() {
                    fields.resize(obj.layout.field_names().len(), Value::Nil);
                }
                if idx < fields.len() {
                    fields[idx] = val;
                    Ok(())
                } else {
                    Err(LustFfiResult::LUST_FFI_ERR_INDEX_OUT_OF_BOUNDS)
                }
            }
            _ => Err(LustFfiResult::LUST_FFI_ERR_TYPE_MISMATCH),
        }
    }) {
        Ok(()) => LustFfiResult::LUST_FFI_OK,
        Err(e) => e,
    }
}

// Callback support
pub type LustFfiCallbackFn = unsafe extern "C" fn(
    user_data: *mut std::ffi::c_void,
    args: *const LustFfiValue,
    args_len: usize,
    out_result: *mut LustFfiValue,
) -> LustFfiResult;

struct CallbackState {
    cb: LustFfiCallbackFn,
    user_data: *mut std::ffi::c_void,
    cleanup: Option<unsafe extern "C" fn(*mut std::ffi::c_void)>,
}

unsafe impl Send for CallbackState {}
unsafe impl Sync for CallbackState {}

impl Drop for CallbackState {
    fn drop(&mut self) {
        if let Some(cleanup_fn) = self.cleanup {
            unsafe {
                cleanup_fn(self.user_data);
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_program_register_callback(
    program: *mut EmbeddedProgram,
    name: *const c_char,
    callback: LustFfiCallbackFn,
    user_data: *mut std::ffi::c_void,
    cleanup: Option<unsafe extern "C" fn(*mut std::ffi::c_void)>,
) -> bool {
    clear_last_error();
    if program.is_null() || name.is_null() {
        set_last_error("Null pointer passed to lust_program_register_callback");
        return false;
    }

    let name_cstr = unsafe { CStr::from_ptr(name) };
    let name_str = match name_cstr.to_str() {
        Ok(s) => s.to_owned(),
        Err(_) => {
            set_last_error("Invalid UTF-8 in callback name");
            return false;
        }
    };

    let state = Rc::new(CallbackState {
        cb: callback,
        user_data,
        cleanup,
    });

    let prog = unsafe { &mut *program };
    let cb_name = name_str.clone();

    let native_func = Value::NativeFunction(crate::bytecode::native_fn(move |values: &[Value]| {
        let mut ffi_args: Vec<LustFfiValue> = Vec::with_capacity(values.len());
        for v in values {
            match value_to_ffi(v.clone()) {
                Ok(fv) => ffi_args.push(fv),
                Err(err) => return Err(format!("FFI argument conversion failed: {err}")),
            }
        }

        let mut out_result = LustFfiValue::default();
        let rc = unsafe {
            (state.cb)(
                state.user_data,
                ffi_args.as_ptr(),
                ffi_args.len(),
                &mut out_result,
            )
        };

        for mut fv in ffi_args {
            lust_value_dispose(&mut fv);
        }

        if rc != LustFfiResult::LUST_FFI_OK {
            return Err(format!("Callback '{}' failed with code {:?}", cb_name, rc));
        }

        let res_val = match value_from_ffi(&out_result) {
            Ok(v) => v,
            Err(err) => {
                lust_value_dispose(&mut out_result);
                return Err(format!("Callback return conversion failed: {err}"));
            }
        };
        lust_value_dispose(&mut out_result);
        Ok(NativeCallResult::Return(res_val))
    }));

    prog.vm_mut().register_native(&name_str, native_func.clone());
    if !name_str.contains('.') {
        prog.vm_mut().register_native(format!("main.{}", name_str), native_func);
    }

    true
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_builder_new() -> *mut EmbeddedBuilder {
    clear_last_error();
    Box::into_raw(Box::new(EmbeddedBuilder::new()))
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_builder_free(builder: *mut EmbeddedBuilder) {
    if builder.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(builder));
    }
}

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
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
    let old = unsafe { ptr::read(builder) };
    let updated = old.set_base_dir(base_dir_str);
    unsafe { ptr::write(builder, updated) };
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_builder_compile(builder: *mut EmbeddedBuilder) -> *mut EmbeddedProgram {
    clear_last_error();
    if builder.is_null() {
        set_last_error("Builder pointer was null");
        return ptr::null_mut();
    }

    let builder_box = unsafe { Box::from_raw(builder) };
    match builder_box.compile() {
        Ok(program) => {
            let prog_ptr = Box::into_raw(Box::new(program));
            register_program_token(prog_ptr);
            prog_ptr
        }
        Err(err) => {
            handle_error(err);
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_program_free(program: *mut EmbeddedProgram) {
    if program.is_null() {
        return;
    }

    unregister_program_token(program);
    unsafe {
        drop(Box::from_raw(program));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_program_run_entry(program: *mut EmbeddedProgram) -> bool {
    clear_last_error();
    if program.is_null() {
        set_last_error("Program pointer was null");
        return false;
    }

    let program_ref = unsafe { &mut *program };
    handle_result(program_ref.run_entry_script()).is_some()
}

#[unsafe(no_mangle)]
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
    let mut call_res = program_ref.call_raw(&func_name, converted_args.clone());
    if call_res.is_err() && !func_name.contains('.') {
        let qualified = format!("main.{}", func_name);
        if let Ok(value) = program_ref.call_raw(&qualified, converted_args) {
            call_res = Ok(value);
        }
    }
    let result = match call_res {
        Ok(value) => value,
        Err(err) => {
            handle_error(err);
            return false;
        }
    };
    if out_value.is_null() {
        return true;
    }

    match value_to_ffi_with_program(program, result) {
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

#[unsafe(no_mangle)]
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

    match value_to_ffi_with_program(program, value) {
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

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
pub extern "C" fn lust_last_error() -> *const c_char {
    lust_last_error_message()
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_value_free(value: *mut LustFfiValue) {
    lust_value_dispose(value);
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_program_new(source: *const c_char) -> *mut EmbeddedProgram {
    clear_last_error();
    if source.is_null() {
        set_last_error("Source pointer was null");
        return ptr::null_mut();
    }

    let source_cstr = unsafe { CStr::from_ptr(source) };
    let source_str = match source_cstr.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("Source code was not valid UTF-8");
            return ptr::null_mut();
        }
    };

    let mut builder = EmbeddedBuilder::new();
    builder.add_module("main", source_str);
    builder.set_entry_module("main");
    match builder.compile() {
        Ok(prog) => {
            let prog_ptr = Box::into_raw(Box::new(prog));
            register_program_token(prog_ptr);
            prog_ptr
        }
        Err(err) => {
            handle_error(err);
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lust_program_eval(
    program: *mut EmbeddedProgram,
    out_value: *mut LustFfiValue,
) -> bool {
    clear_last_error();
    if program.is_null() {
        set_last_error("Program pointer was null");
        return false;
    }

    let program_ref = unsafe { &mut *program };
    if let Err(err) = program_ref.run_entry_script() {
        handle_error(err);
        return false;
    }

    if !out_value.is_null() {
        unsafe {
            *out_value = LustFfiValue::default();
        }
    }
    true
}

#![allow(non_camel_case_types, dead_code)]

use core::ffi::{c_char, c_void};

#[repr(C)]
pub struct napi_env__ {
    _unused: [u8; 0],
}
#[repr(C)]
pub struct napi_value__ {
    _unused: [u8; 0],
}
#[repr(C)]
pub struct napi_deferred__ {
    _unused: [u8; 0],
}
#[repr(C)]
pub struct napi_async_work__ {
    _unused: [u8; 0],
}
#[repr(C)]
pub struct napi_callback_info__ {
    _unused: [u8; 0],
}

pub type napi_env = *mut napi_env__;
pub type node_api_basic_env = napi_env;
pub type napi_value = *mut napi_value__;
pub type napi_deferred = *mut napi_deferred__;
pub type napi_async_work = *mut napi_async_work__;
pub type napi_callback_info = *mut napi_callback_info__;

pub type napi_status = i32;
pub const NAPI_OK: napi_status = 0;
pub const NAPI_AUTO_LENGTH: usize = usize::MAX;

pub type napi_callback =
    Option<unsafe extern "C" fn(env: napi_env, info: napi_callback_info) -> napi_value>;
pub type napi_async_execute_callback = Option<unsafe extern "C" fn(env: napi_env, data: *mut c_void)>;
pub type napi_async_complete_callback =
    Option<unsafe extern "C" fn(env: napi_env, status: napi_status, data: *mut c_void)>;
pub type napi_cleanup_hook = Option<unsafe extern "C" fn(arg: *mut c_void)>;

unsafe extern "C" {
    pub fn napi_get_cb_info(
        env: napi_env,
        cbinfo: napi_callback_info,
        argc: *mut usize,
        argv: *mut napi_value,
        this_arg: *mut napi_value,
        data: *mut *mut c_void,
    ) -> napi_status;

    pub fn napi_create_int64(env: napi_env, value: i64, result: *mut napi_value) -> napi_status;
    pub fn napi_get_value_int64(env: napi_env, value: napi_value, result: *mut i64) -> napi_status;
    pub fn napi_get_boolean(env: napi_env, value: bool, result: *mut napi_value) -> napi_status;
    pub fn napi_get_undefined(env: napi_env, result: *mut napi_value) -> napi_status;

    pub fn napi_create_string_utf8(
        env: napi_env,
        str_ptr: *const c_char,
        length: usize,
        result: *mut napi_value,
    ) -> napi_status;

    pub fn napi_get_value_string_utf8(
        env: napi_env,
        value: napi_value,
        buf: *mut c_char,
        bufsize: usize,
        result: *mut usize,
    ) -> napi_status;

    pub fn napi_create_object(env: napi_env, result: *mut napi_value) -> napi_status;
    pub fn napi_set_named_property(
        env: napi_env,
        object: napi_value,
        utf8name: *const c_char,
        value: napi_value,
    ) -> napi_status;

    pub fn napi_create_function(
        env: napi_env,
        utf8name: *const c_char,
        length: usize,
        cb: napi_callback,
        data: *mut c_void,
        result: *mut napi_value,
    ) -> napi_status;

    pub fn napi_create_error(
        env: napi_env,
        code: napi_value,
        msg: napi_value,
        result: *mut napi_value,
    ) -> napi_status;
    pub fn napi_throw_error(
        env: napi_env,
        code: *const c_char,
        msg: *const c_char,
    ) -> napi_status;

    pub fn napi_create_promise(
        env: napi_env,
        deferred: *mut napi_deferred,
        promise: *mut napi_value,
    ) -> napi_status;
    pub fn napi_resolve_deferred(
        env: napi_env,
        deferred: napi_deferred,
        resolution: napi_value,
    ) -> napi_status;
    pub fn napi_reject_deferred(
        env: napi_env,
        deferred: napi_deferred,
        rejection: napi_value,
    ) -> napi_status;

    pub fn napi_create_async_work(
        env: napi_env,
        async_resource: napi_value,
        async_resource_name: napi_value,
        execute: napi_async_execute_callback,
        complete: napi_async_complete_callback,
        data: *mut c_void,
        result: *mut napi_async_work,
    ) -> napi_status;
    pub fn napi_queue_async_work(env: node_api_basic_env, work: napi_async_work) -> napi_status;
    pub fn napi_delete_async_work(env: napi_env, work: napi_async_work) -> napi_status;

    pub fn napi_add_env_cleanup_hook(
        env: node_api_basic_env,
        fun: napi_cleanup_hook,
        arg: *mut c_void,
    ) -> napi_status;
}

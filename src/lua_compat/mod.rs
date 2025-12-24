#![allow(non_snake_case, non_camel_case_types)]
//! Lua 5.1 C API compatibility scaffolding.
//! This module will host the runtime bridge and tracing that drive extern stub generation.

use crate::bytecode::{Value, ValueKey};
use crate::number::{LustFloat, LustInt};
use crate::vm::{NativeCallResult, VM};
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::ffi::{c_char, c_int, c_void};
use core::hash::{Hash, Hasher};
use core::ptr;
use core::sync::atomic::{AtomicUsize, Ordering};
use hashbrown::{hash_map::Entry, HashMap};
#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
use libloading::Library;
use std::cell::RefCell as StdRefCell;
use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::sync::OnceLock;

thread_local! {
    static LUST_FN_REGISTRY: StdRefCell<HashMap<usize, Value>> = StdRefCell::new(HashMap::new());
}

pub mod transpile;

static NEXT_LUA_FUNCTION_ID: AtomicUsize = AtomicUsize::new(1);

#[derive(Clone, Debug)]
struct LuaTraceConfig {
    enabled: bool,
    stack: bool,
    cfunc: bool,
    filter: Option<String>,
}

static LUA_TRACE_CONFIG: OnceLock<LuaTraceConfig> = OnceLock::new();

fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => {
            let value = value.trim();
            !(value.is_empty() || value == "0" || value.eq_ignore_ascii_case("false"))
        }
        Err(_) => false,
    }
}

fn lua_trace_config() -> &'static LuaTraceConfig {
    LUA_TRACE_CONFIG.get_or_init(|| LuaTraceConfig {
        enabled: env_flag("LUST_LUA_TRACE"),
        stack: env_flag("LUST_LUA_TRACE_STACK"),
        cfunc: env_flag("LUST_LUA_TRACE_CFUNC"),
        filter: std::env::var("LUST_LUA_TRACE_FILTER")
            .ok()
            .and_then(|v| (!v.trim().is_empty()).then_some(v)),
    })
}

fn format_lua_value_brief(value: &LuaValue) -> String {
    match value {
        LuaValue::Nil => "nil".to_string(),
        LuaValue::Bool(b) => format!("bool({})", b),
        LuaValue::Int(i) => format!("int({})", i),
        LuaValue::Float(f) => format!("num({})", f),
        LuaValue::String(s) => {
            const MAX: usize = 40;
            if s.len() > MAX {
                format!("str({:?}…)", &s[..MAX])
            } else {
                format!("str({:?})", s)
            }
        }
        LuaValue::Table(handle) => format!("table({:p})", Rc::as_ptr(handle)),
        LuaValue::Function(f) => match &f.name {
            Some(name) => format!("func#{}({})", f.id, name),
            None => format!("func#{}", f.id),
        },
        LuaValue::Userdata(u) => format!("ud#{}", u.id),
        LuaValue::Thread(t) => format!("thread#{}", t.id),
        LuaValue::LightUserdata(ptr) => format!("lud({:#x})", ptr),
    }
}

fn format_lua_stack_brief(stack: &[LuaValue]) -> String {
    const MAX: usize = 8;
    if stack.is_empty() {
        return "[]".to_string();
    }
    let mut parts = Vec::new();
    let start = stack.len().saturating_sub(MAX);
    for value in &stack[start..] {
        parts.push(format_lua_value_brief(value));
    }
    if start > 0 {
        format!("[… {}]", parts.join(", "))
    } else {
        format!("[{}]", parts.join(", "))
    }
}

fn next_lua_function_id() -> usize {
    NEXT_LUA_FUNCTION_ID.fetch_add(1, Ordering::SeqCst)
}

pub(crate) fn register_lust_function(value: Value) -> usize {
    let id = next_lua_function_id();
    LUST_FN_REGISTRY.with(|registry| {
        registry.borrow_mut().insert(id, value);
    });
    id
}

pub(crate) fn lookup_lust_function(id: usize) -> Option<Value> {
    LUST_FN_REGISTRY.with(|registry| registry.borrow().get(&id).cloned())
}

/// Lua 5.1 type codes.
pub const LUA_TNONE: c_int = -1;
pub const LUA_TNIL: c_int = 0;
pub const LUA_TBOOLEAN: c_int = 1;
pub const LUA_TLIGHTUSERDATA: c_int = 2;
pub const LUA_TNUMBER: c_int = 3;
pub const LUA_TSTRING: c_int = 4;
pub const LUA_TTABLE: c_int = 5;
pub const LUA_TFUNCTION: c_int = 6;
pub const LUA_TUSERDATA: c_int = 7;
pub const LUA_TTHREAD: c_int = 8;
pub const LUA_REGISTRYINDEX: c_int = -10000;
pub const LUA_ENVIRONINDEX: c_int = -10001;
pub const LUA_GLOBALSINDEX: c_int = -10002;

/// Lua 5.1 error codes.
pub const LUA_ERRRUN: c_int = 2;
pub const LUA_ERRSYNTAX: c_int = 3;
pub const LUA_ERRMEM: c_int = 4;
pub const LUA_ERRERR: c_int = 5;

pub type lua_Number = LustFloat;
pub type lua_Integer = LustInt;
pub type lua_CFunction = Option<unsafe extern "C" fn(*mut lua_State) -> c_int>;

#[repr(C)]
pub struct lua_State {
    pub state: LuaState,
}

#[repr(C)]
pub struct luaL_Reg {
    pub name: *const c_char,
    pub func: lua_CFunction,
}

/// Simplified mirror of the Lua 5.1 buffer helper type.
pub const LUAL_BUFFERSIZE: usize = 8192;

#[repr(C)]
pub struct luaL_Buffer {
    pub p: *mut c_char,
    pub lvl: c_int,
    pub L: *mut lua_State,
    pub buffer: [c_char; LUAL_BUFFERSIZE],
}

/// Metadata about a Lua C library that should be loaded through the compatibility layer.
#[derive(Debug, Clone)]
pub struct LuaModuleSpec {
    pub library_path: PathBuf,
    pub entrypoints: Vec<String>,
}

impl LuaModuleSpec {
    pub fn new(library_path: PathBuf, entrypoints: Vec<String>) -> Self {
        Self {
            library_path,
            entrypoints,
        }
    }
}

/// Describes a traced call into the Lua 5.1 API while evaluating `luaopen_*`.
#[derive(Debug, Clone)]
pub struct LuaApiCall {
    pub function: String,
    pub args: Vec<String>,
}

/// Placeholder for a traced module export that can later be turned into a Lust extern stub.
#[derive(Debug, Clone)]
pub struct LuaModuleTrace {
    pub module: String,
    pub api_calls: Vec<LuaApiCall>,
}

/// Result of running a `luaopen_*` entrypoint through the compat shim.
#[derive(Clone)]
pub struct LuaOpenResult {
    pub module: String,
    pub trace: Vec<LuaApiCall>,
    pub returns: Vec<LuaValue>,
    pub state: Option<Rc<RefCell<Box<lua_State>>>>,
}

impl core::fmt::Debug for LuaOpenResult {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LuaOpenResult")
            .field("module", &self.module)
            .field("trace", &self.trace)
            .field("returns", &self.returns)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug)]
pub enum LuaValue {
    Nil,
    Bool(bool),
    Int(LustInt),
    Float(LustFloat),
    String(String),
    Table(LuaTableHandle),
    Function(LuaFunction),
    Userdata(LuaUserdata),
    Thread(LuaThread),
    LightUserdata(usize),
}

#[derive(Clone, Debug)]
pub struct LuaFunction {
    pub id: usize,
    pub name: Option<String>,
    pub cfunc: lua_CFunction,
    pub lust_handle: Option<usize>,
    pub upvalues: Vec<LuaValue>,
}

#[derive(Clone, Debug)]
pub struct LuaUserdata {
    pub id: usize,
    pub data: *mut c_void,
    pub state: *mut lua_State,
}

#[derive(Clone, Debug)]
pub struct LuaThread {
    pub id: usize,
}

#[derive(Clone, Debug, Default)]
pub struct LuaTable {
    pub entries: HashMap<LuaValue, LuaValue>,
    pub metamethods: HashMap<String, LuaValue>,
    pub metatable: Option<LuaTableHandle>,
}

pub type LuaTableHandle = Rc<RefCell<LuaTable>>;

impl PartialEq for LuaValue {
    fn eq(&self, other: &Self) -> bool {
        use LuaValue::*;
        match (self, other) {
            (Nil, Nil) => true,
            (Bool(a), Bool(b)) => a == b,
            (Int(a), Int(b)) => a == b,
            (Float(a), Float(b)) => a == b,
            (String(a), String(b)) => a == b,
            (LightUserdata(a), LightUserdata(b)) => a == b,
            (Table(a), Table(b)) => Rc::ptr_eq(a, b),
            (Function(a), Function(b)) => a.id == b.id,
            (Userdata(a), Userdata(b)) => a.id == b.id,
            (Thread(a), Thread(b)) => a.id == b.id,
            _ => false,
        }
    }
}

impl Eq for LuaValue {}

impl Hash for LuaValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            LuaValue::Nil => {}
            LuaValue::Bool(b) => b.hash(state),
            LuaValue::Int(i) => i.hash(state),
            LuaValue::Float(f) => f.to_bits().hash(state),
            LuaValue::String(s) => s.hash(state),
            LuaValue::LightUserdata(ptr) => ptr.hash(state),
            LuaValue::Table(handle) => (Rc::as_ptr(handle) as usize).hash(state),
            LuaValue::Function(f) => f.id.hash(state),
            LuaValue::Userdata(u) => u.id.hash(state),
            LuaValue::Thread(t) => t.id.hash(state),
        }
    }
}

// Arithmetic operators for LuaValue
impl std::ops::Add for LuaValue {
    type Output = LuaValue;
    fn add(self, other: Self) -> Self::Output {
        use LuaValue::*;
        match (self, other) {
            (Int(a), Int(b)) => Int(a + b),
            (Float(a), Float(b)) => Float(a + b),
            (Int(a), Float(b)) => Float(a as LustFloat + b),
            (Float(a), Int(b)) => Float(a + b as LustFloat),
            _ => Nil, // Lua would throw error, but return Nil for now
        }
    }
}

impl std::ops::Sub for LuaValue {
    type Output = LuaValue;
    fn sub(self, other: Self) -> Self::Output {
        use LuaValue::*;
        match (self, other) {
            (Int(a), Int(b)) => Int(a - b),
            (Float(a), Float(b)) => Float(a - b),
            (Int(a), Float(b)) => Float(a as LustFloat - b),
            (Float(a), Int(b)) => Float(a - b as LustFloat),
            _ => Nil,
        }
    }
}

impl std::ops::Mul for LuaValue {
    type Output = LuaValue;
    fn mul(self, other: Self) -> Self::Output {
        use LuaValue::*;
        match (self, other) {
            (Int(a), Int(b)) => Int(a * b),
            (Float(a), Float(b)) => Float(a * b),
            (Int(a), Float(b)) => Float(a as LustFloat * b),
            (Float(a), Int(b)) => Float(a * b as LustFloat),
            _ => Nil,
        }
    }
}

impl std::ops::Div for LuaValue {
    type Output = LuaValue;
    fn div(self, other: Self) -> Self::Output {
        use LuaValue::*;
        match (self, other) {
            (Int(a), Int(b)) => Float(a as LustFloat / b as LustFloat),
            (Float(a), Float(b)) => Float(a / b),
            (Int(a), Float(b)) => Float(a as LustFloat / b),
            (Float(a), Int(b)) => Float(a / b as LustFloat),
            _ => Nil,
        }
    }
}

impl std::ops::Rem for LuaValue {
    type Output = LuaValue;
    fn rem(self, other: Self) -> Self::Output {
        use LuaValue::*;
        match (self, other) {
            (Int(a), Int(b)) => Int(a % b),
            (Float(a), Float(b)) => Float(a % b),
            (Int(a), Float(b)) => Float(a as LustFloat % b),
            (Float(a), Int(b)) => Float(a % b as LustFloat),
            _ => Nil,
        }
    }
}

impl std::ops::Neg for LuaValue {
    type Output = LuaValue;
    fn neg(self) -> Self::Output {
        use LuaValue::*;
        match self {
            Int(a) => Int(-a),
            Float(a) => Float(-a),
            _ => Nil,
        }
    }
}

impl PartialOrd for LuaValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use LuaValue::*;
        match (self, other) {
            (Int(a), Int(b)) => a.partial_cmp(b),
            (Float(a), Float(b)) => a.partial_cmp(b),
            (Int(a), Float(b)) => (*a as LustFloat).partial_cmp(b),
            (Float(a), Int(b)) => a.partial_cmp(&(*b as LustFloat)),
            (String(a), String(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

impl LuaTable {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Convert a LuaValue into a Lust `Value` that matches the builtin LuaValue enum/structs.
fn register_c_function(
    func: &LuaFunction,
    state: &Rc<RefCell<Box<lua_State>>>,
    vm: &VM,
) -> Result<Value, String> {
    let cfunc = func
        .cfunc
        .ok_or_else(|| "missing cfunc pointer".to_string())?;
    let shared_state = state.clone();
    let cfunc_name = func.name.clone();
    let cfunc_upvalues = func.upvalues.clone();
    let native = Value::NativeFunction(Rc::new(move |args: &[Value]| {
        let state_cell = shared_state.clone();
        VM::with_current(|vm| {
            let cfg = lua_trace_config();
            let saved_upvalues: Vec<LuaValue>;
            let mut saved_stack: Vec<LuaValue>;
            let mut saved_pending_error: Option<LuaValue>;
            let ptr: *mut lua_State = {
                let mut guard = state_cell.borrow_mut();
                // Simulate a fresh Lua call frame without clobbering an existing frame.
                // This matters for nested calls (Lua -> Lust -> Lua) where the outer C function
                // still expects its own stack arguments to be intact.
                saved_stack = core::mem::take(&mut guard.state.stack);
                saved_pending_error = guard.state.pending_error.take();
                saved_upvalues =
                    core::mem::replace(&mut guard.state.current_upvalues, cfunc_upvalues.clone());

                // Lua multiple-return semantics: if the last argument is a multi-return list, expand it.
                // Our Lua-compat layer represents multi-returns as `Value::Array` (typically `Array<LuaValue>`).
                // - For the last arg: expand all elements onto the Lua stack.
                // - For non-last args: only pass the first element (Lua would truncate).
                for (idx, arg) in args.iter().enumerate() {
                    let is_last = idx + 1 == args.len();
                    if let Value::Array(arr) = arg {
                        let arr = arr.borrow();
                        if is_last {
                            for elem in arr.iter() {
                                guard.state.push(value_to_lua(elem, vm));
                            }
                        } else if let Some(first) = arr.first() {
                            guard.state.push(value_to_lua(first, vm));
                        } else {
                            guard.state.push(LuaValue::Nil);
                        }
                    } else {
                        let lua_arg = value_to_lua(arg, vm);
                        guard.state.push(lua_arg);
                    }
                }

                if cfg.enabled || cfg.cfunc {
                    let label = cfunc_name.as_deref().unwrap_or("<anonymous>");
                    eprintln!(
                        "[lua-cfunc] -> {} {:#x} nargs={} top={} stack={}",
                        label,
                        cfunc as usize,
                        args.len(),
                        guard.state.len(),
                        format_lua_stack_brief(&guard.state.stack),
                    );
                }

                &mut **guard as *mut lua_State
            };

            // Drop the RefCell borrow while the C function runs so nested cfunc calls can borrow too.
            let ret_count = unsafe { cfunc(ptr) };

            let results = {
                let mut guard = state_cell.borrow_mut();
                if cfg.enabled || cfg.cfunc {
                    let label = cfunc_name.as_deref().unwrap_or("<anonymous>");
                    eprintln!(
                        "[lua-cfunc] <- {} {:#x} nret={} top={} stack={}",
                        label,
                        cfunc as usize,
                        ret_count,
                        guard.state.len(),
                        format_lua_stack_brief(&guard.state.stack),
                    );
                }
                let pending_error = guard.state.pending_error.take();
                guard.state.current_upvalues = saved_upvalues;
                if let Some(err) = pending_error {
                    // Restore the outer call frame before returning the error.
                    guard.state.stack = core::mem::take(&mut saved_stack);
                    guard.state.pending_error = saved_pending_error.take();
                    return Err(lua_error_message(&err));
                }
                if ret_count < 0 {
                    guard.state.stack = core::mem::take(&mut saved_stack);
                    guard.state.pending_error = saved_pending_error.take();
                    return Err("lua_error".to_string());
                }

                let mut results = Vec::new();
                if ret_count > 0 {
                    for _ in 0..ret_count {
                        results.push(guard.state.pop().unwrap_or(LuaValue::Nil));
                    }
                    results.reverse();
                }

                // Discard the callee call frame and restore the outer one.
                guard.state.stack = core::mem::take(&mut saved_stack);
                guard.state.pending_error = saved_pending_error.take();
                results
            };

            let mut converted = Vec::new();
            for value in &results {
                converted.push(lua_to_lust(value, vm, Some(state_cell.clone()))?);
            }
            let return_value = Value::array(converted);
            Ok(NativeCallResult::Return(return_value))
        })
    }));
    let handle = register_lust_function(native.clone());
    let instance = vm
        .instantiate_struct(
            "LuaFunction",
            vec![(Rc::new("handle".to_string()), Value::Int(handle as i64))],
        )
        .map_err(|e| e.to_string())?;
    Ok(Value::enum_variant("LuaValue", "Function", vec![instance]))
}

/// Register a Lua closure (non-C function) so it can be called from Lust code
fn register_lua_closure(
    func: &LuaFunction,
    state: &Rc<RefCell<Box<lua_State>>>,
    vm: &VM,
) -> Result<Value, String> {
    // Store the entire function value so we can call it later
    let lua_func = LuaValue::Function(func.clone());
    let shared_state = state.clone();

    let native = Value::NativeFunction(Rc::new(move |args: &[Value]| {
        VM::with_current(|vm| {
            let state_cell = shared_state.clone();

            // Build argument list as Lua values
            let mut lua_args = Vec::new();
            for arg in args {
                lua_args.push(value_to_lua(arg, vm));
            }

            // Try to call the Lua function
            // If it has a lust_handle, it's a Lust function wrapped as Lua
            if let LuaValue::Function(f) = &lua_func {
                if let Some(handle) = f.lust_handle {
                    // Call through the Lust function registry
                    if let Some(lust_func) = lookup_lust_function(handle) {
                        let lust_args: Result<Vec<Value>, String> = lua_args
                            .iter()
                            .map(|v| lua_to_lust(v, vm, Some(state_cell.clone())))
                            .collect();

                        return match lust_args {
                            Ok(converted_args) => {
                                match vm.call_value(&lust_func, converted_args) {
                                    Ok(ret_val) => {
                                        let results = if let Value::Tuple(vals) = ret_val {
                                            vals.iter().map(|v| value_to_lua(v, vm)).collect()
                                        } else {
                                            vec![value_to_lua(&ret_val, vm)]
                                        };

                                        // Convert back to Lust values
                                        let lust_results: Result<Vec<Value>, String> = results
                                            .iter()
                                            .map(|v| lua_to_lust(v, vm, Some(state_cell.clone())))
                                            .collect();

                                        match lust_results {
                                            Ok(vals) if vals.len() == 1 => Ok(
                                                crate::bytecode::value::NativeCallResult::Return(
                                                    vals[0].clone(),
                                                ),
                                            ),
                                            Ok(vals) if vals.is_empty() => Ok(
                                                crate::bytecode::value::NativeCallResult::Return(
                                                    Value::Nil,
                                                ),
                                            ),
                                            Ok(vals) => {
                                                let arr = Rc::new(RefCell::new(vals));
                                                Ok(crate::bytecode::value::NativeCallResult::Return(Value::Array(arr)))
                                            }
                                            Err(e) => Err(e),
                                        }
                                    }
                                    Err(e) => Err(e.to_string()),
                                }
                            }
                            Err(e) => Err(e),
                        };
                    }
                }
            }

            // Pure Lua closures without bytecode interpreter support
            Err("Cannot call pure Lua closures (no Lua bytecode interpreter available)".to_string())
        })
    }));

    let handle = register_lust_function(native.clone());
    let instance = vm
        .instantiate_struct(
            "LuaFunction",
            vec![(Rc::new("handle".to_string()), Value::Int(handle as i64))],
        )
        .map_err(|e| e.to_string())?;
    Ok(Value::enum_variant("LuaValue", "Function", vec![instance]))
}

fn lua_error_message(err: &LuaValue) -> String {
    match err {
        LuaValue::String(s) => s.clone(),
        LuaValue::Table(handle) => {
            let table = handle.borrow();
            for key in [
                LuaValue::Int(2),
                LuaValue::String("message".to_string()),
                LuaValue::String("msg".to_string()),
                LuaValue::Int(1),
            ] {
                if let Some(LuaValue::String(s)) = table.entries.get(&key) {
                    return s.clone();
                }
            }
            format_lua_value_brief(err)
        }
        _ => format_lua_value_brief(err),
    }
}

pub fn lua_to_lust(
    value: &LuaValue,
    vm: &VM,
    state: Option<Rc<RefCell<Box<lua_State>>>>,
) -> Result<Value, String> {
    let mut table_cache: HashMap<usize, Value> = HashMap::new();
    lua_to_lust_cached(value, vm, state, &mut table_cache)
}

fn lua_to_lust_cached(
    value: &LuaValue,
    vm: &VM,
    state: Option<Rc<RefCell<Box<lua_State>>>>,
    table_cache: &mut HashMap<usize, Value>,
) -> Result<Value, String> {
    match value {
        LuaValue::Nil => Ok(Value::enum_unit("LuaValue", "Nil")),
        LuaValue::Bool(b) => Ok(Value::enum_variant(
            "LuaValue",
            "Bool",
            vec![Value::Bool(*b)],
        )),
        LuaValue::Int(i) => Ok(Value::enum_variant("LuaValue", "Int", vec![Value::Int(*i)])),
        LuaValue::Float(f) => Ok(Value::enum_variant(
            "LuaValue",
            "Float",
            vec![Value::Float(*f)],
        )),
        LuaValue::String(s) => Ok(Value::enum_variant(
            "LuaValue",
            "String",
            vec![Value::string(s.clone())],
        )),
        LuaValue::LightUserdata(ptr) => Ok(Value::enum_variant(
            "LuaValue",
            "LightUserdata",
            vec![Value::Int(*ptr as i64)],
        )),
        LuaValue::Function(func) => {
            if let Some(handle) = func.lust_handle {
                if let Some(inner) = lookup_lust_function(handle) {
                    return Ok(inner);
                }
            }
            if func.cfunc.is_some() {
                if let Some(state) = &state {
                    return register_c_function(func, state, vm);
                }
            }
            // For Lua closures without cfunc, create a wrapper that can call them
            if let Some(state_rc) = &state {
                return register_lua_closure(func, state_rc, vm);
            }
            let instance = vm
                .instantiate_struct(
                    "LuaFunction",
                    vec![(Rc::new("handle".to_string()), Value::Int(func.id as i64))],
                )
                .map_err(|e| e.to_string())?;
            Ok(Value::enum_variant("LuaValue", "Function", vec![instance]))
        }
        LuaValue::Userdata(data) => {
            let metamethods_value = vm.new_map_value();
            let userdata_meta = state.as_ref().and_then(|cell| {
                cell.borrow()
                    .state
                    .userdata_metamethods
                    .get(&data.id)
                    .cloned()
            });
            if let Some(meta) = userdata_meta {
                for (name, meta_value) in meta {
                    let converted =
                        lua_to_lust_cached(&meta_value, vm, state.clone(), table_cache)?;
                    metamethods_value
                        .map_set(ValueKey::string(name), converted)
                        .map_err(|e| e.to_string())?;
                }
            }

            let instance = vm
                .instantiate_struct(
                    "LuaUserdata",
                    vec![
                        (Rc::new("handle".to_string()), Value::Int(data.id as i64)),
                        (
                            Rc::new("ptr".to_string()),
                            Value::Int(data.data as usize as i64),
                        ),
                        (
                            Rc::new("state".to_string()),
                            Value::Int(data.state as usize as i64),
                        ),
                        (Rc::new("metamethods".to_string()), metamethods_value),
                    ],
                )
                .map_err(|e| e.to_string())?;
            Ok(Value::enum_variant("LuaValue", "Userdata", vec![instance]))
        }
        LuaValue::Thread(thread) => {
            let instance = vm
                .instantiate_struct(
                    "LuaThread",
                    vec![(Rc::new("handle".to_string()), Value::Int(thread.id as i64))],
                )
                .map_err(|e| e.to_string())?;
            Ok(Value::enum_variant("LuaValue", "Thread", vec![instance]))
        }
        LuaValue::Table(handle) => lua_table_to_struct_cached(handle, vm, state, table_cache),
    }
}

fn unwrap_lua_value_for_key(value: Value) -> Value {
    if let Value::Enum {
        enum_name,
        variant,
        values,
    } = &value
    {
        if enum_name == "LuaValue" {
            return match variant.as_str() {
                "Nil" => Value::Nil,
                "Bool" | "Int" | "Float" | "String" | "Table" | "Function" | "LightUserdata"
                | "Userdata" | "Thread" => values
                    .as_ref()
                    .and_then(|v| v.get(0))
                    .cloned()
                    .unwrap_or(Value::Nil),
                _ => value,
            };
        }
    }
    value
}

fn lua_table_to_struct_cached(
    handle: &LuaTableHandle,
    vm: &VM,
    state: Option<Rc<RefCell<Box<lua_State>>>>,
    table_cache: &mut HashMap<usize, Value>,
) -> Result<Value, String> {
    let cache_key = Rc::as_ptr(handle) as usize;
    if let Some(existing) = table_cache.get(&cache_key) {
        return Ok(existing.clone());
    }

    let table_value = vm.new_map_value();
    let metamethods_value = vm.new_map_value();
    let struct_value = vm
        .instantiate_struct(
            "LuaTable",
            vec![
                (Rc::new("table".to_string()), table_value.clone()),
                (
                    Rc::new("metamethods".to_string()),
                    metamethods_value.clone(),
                ),
            ],
        )
        .map_err(|e| e.to_string())?;
    let wrapped = Value::enum_variant("LuaValue", "Table", vec![struct_value]);
    table_cache.insert(cache_key, wrapped.clone());

    {
        let borrowed = handle.borrow();
        for (key, value) in &borrowed.entries {
            // Table keys should behave like Lua keys (string/number/bool, etc.). Using the LuaValue
            // wrapper enum directly as a key is problematic because enum keys currently compare by
            // payload-pointer identity. Unwrap keys into their raw underlying values instead.
            let key_val =
                unwrap_lua_value_for_key(lua_to_lust_cached(key, vm, state.clone(), table_cache)?);
            let value_val = lua_to_lust_cached(value, vm, state.clone(), table_cache)?;
            table_value
                .map_set(ValueKey::from(key_val), value_val)
                .map_err(|e| e.to_string())?;
        }
        for (name, meta) in &borrowed.metamethods {
            let meta_val = lua_to_lust_cached(meta, vm, state.clone(), table_cache)?;
            metamethods_value
                .map_set(ValueKey::string(name.clone()), meta_val)
                .map_err(|e| e.to_string())?;
        }
    }

    Ok(wrapped)
}

fn lua_function_from_handle(handle: usize) -> LuaFunction {
    LuaFunction {
        id: handle,
        name: None,
        cfunc: None,
        lust_handle: Some(handle),
        upvalues: Vec::new(),
    }
}

pub(crate) fn value_to_lua(value: &Value, vm: &VM) -> LuaValue {
    match value {
        Value::Nil => LuaValue::Nil,
        Value::Bool(b) => LuaValue::Bool(*b),
        Value::Int(i) => LuaValue::Int(*i),
        Value::Float(f) => LuaValue::Float(*f),
        Value::String(s) => LuaValue::String((**s).clone()),
        Value::Enum {
            enum_name,
            variant,
            values,
        } if enum_name == "LuaValue" => match variant.as_str() {
            "Nil" => LuaValue::Nil,
            "Bool" => values
                .as_ref()
                .and_then(|v| v.get(0))
                .and_then(|v| {
                    if let Value::Bool(b) = v {
                        Some(*b)
                    } else {
                        None
                    }
                })
                .map(LuaValue::Bool)
                .unwrap_or(LuaValue::Nil),
            "Int" => values
                .as_ref()
                .and_then(|v| v.get(0))
                .and_then(|v| v.as_int())
                .map(LuaValue::Int)
                .unwrap_or(LuaValue::Nil),
            "Float" => values
                .as_ref()
                .and_then(|v| v.get(0))
                .and_then(|v| v.as_float())
                .map(LuaValue::Float)
                .unwrap_or(LuaValue::Nil),
            "String" => values
                .as_ref()
                .and_then(|v| v.get(0))
                .and_then(|v| v.as_string_rc())
                .map(|s| LuaValue::String((*s).clone()))
                .unwrap_or(LuaValue::Nil),
            "LightUserdata" => values
                .as_ref()
                .and_then(|v| v.get(0))
                .and_then(|v| v.as_int())
                .map(|i| LuaValue::LightUserdata(i as usize))
                .unwrap_or(LuaValue::LightUserdata(0)),
            "Function" => {
                let handle = values
                    .as_ref()
                    .and_then(|vals| vals.get(0))
                    .and_then(|v| v.struct_get_field("handle"))
                    .and_then(|v| v.as_int())
                    .unwrap_or(0) as usize;
                LuaValue::Function(lua_function_from_handle(handle))
            }
            "Userdata" => {
                let handle = values
                    .as_ref()
                    .and_then(|vals| vals.get(0))
                    .and_then(|v| v.struct_get_field("handle"))
                    .and_then(|v| v.as_int())
                    .unwrap_or(0) as usize;
                let ptr = values
                    .as_ref()
                    .and_then(|vals| vals.get(0))
                    .and_then(|v| v.struct_get_field("ptr"))
                    .and_then(|v| v.as_int())
                    .unwrap_or(0) as usize;
                let state_ptr = values
                    .as_ref()
                    .and_then(|vals| vals.get(0))
                    .and_then(|v| v.struct_get_field("state"))
                    .and_then(|v| v.as_int())
                    .unwrap_or(0) as usize;
                LuaValue::Userdata(LuaUserdata {
                    id: handle,
                    data: ptr as *mut c_void,
                    state: state_ptr as *mut lua_State,
                })
            }
            "Thread" => {
                let handle = values
                    .as_ref()
                    .and_then(|vals| vals.get(0))
                    .and_then(|v| v.struct_get_field("handle"))
                    .and_then(|v| v.as_int())
                    .unwrap_or(0) as usize;
                LuaValue::Thread(LuaThread { id: handle })
            }
            "Table" => {
                if let Some(values) = values {
                    if let Some(table_struct) = values.get(0) {
                        let table_field = table_struct.struct_get_field("table");
                        let meta_field = table_struct.struct_get_field("metamethods");
                        let mut lua_table = LuaTable::new();
                        if let Some(Value::Map(map)) = table_field {
                            for (k, v) in map.borrow().iter() {
                                lua_table
                                    .entries
                                    .insert(value_to_lua(&k.to_value(), vm), value_to_lua(v, vm));
                            }
                        }
                        if let Some(Value::Map(meta)) = meta_field {
                            for (k, v) in meta.borrow().iter() {
                                if let Some(key) = k.to_value().as_string() {
                                    lua_table
                                        .metamethods
                                        .insert(key.to_string(), value_to_lua(v, vm));
                                }
                            }
                        }
                        return LuaValue::Table(Rc::new(RefCell::new(lua_table)));
                    }
                }
                LuaValue::Nil
            }
            _ => LuaValue::Nil,
        },
        Value::Map(map) => {
            let mut table = LuaTable::new();
            for (k, v) in map.borrow().iter() {
                table
                    .entries
                    .insert(value_to_lua(&k.to_value(), vm), value_to_lua(v, vm));
            }
            LuaValue::Table(Rc::new(RefCell::new(table)))
        }
        Value::Function(_) | Value::Closure { .. } | Value::NativeFunction(_) => {
            let handle = register_lust_function(value.clone());
            LuaValue::Function(lua_function_from_handle(handle))
        }
        _ => LuaValue::LightUserdata(value.type_of() as usize),
    }
}

/// Lightweight state used while running `luaopen_*` entrypoints via the compat layer.
/// This is not a full Lua VM; it is just enough structure to drive API tracing and
/// convert between Rust-side LuaValues and Lust-visible LuaValue enums.
pub struct LuaState {
    pub stack: Vec<LuaValue>,
    trace: Vec<LuaApiCall>,
    globals: HashMap<String, LuaValue>,
    pending_error: Option<LuaValue>,
    next_userdata_id: usize,
    next_reference: i32,
    references: HashMap<i32, LuaValue>,
    userdata_storage: HashMap<usize, (Box<[usize]>, usize)>,
    userdata_metamethods: HashMap<usize, HashMap<String, LuaValue>>,
    pub userdata_metatables: HashMap<usize, LuaTableHandle>,
    string_cache: Vec<std::ffi::CString>,
    pub current_upvalues: Vec<LuaValue>,
    #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
    libraries: Vec<Rc<Library>>,
    registry: LuaTableHandle,
    globals_table: LuaTableHandle,
}

impl Default for LuaState {
    fn default() -> Self {
        let globals_table = Rc::new(RefCell::new(LuaTable::new()));
        let registry = Rc::new(RefCell::new(LuaTable::new()));
        let loaded = Rc::new(RefCell::new(LuaTable::new()));

        registry.borrow_mut().entries.insert(
            LuaValue::String("_LOADED".to_string()),
            LuaValue::Table(loaded.clone()),
        );

        {
            let mut globals_guard = globals_table.borrow_mut();
            globals_guard.entries.insert(
                LuaValue::String("_G".to_string()),
                LuaValue::Table(globals_table.clone()),
            );
            let package = Rc::new(RefCell::new(LuaTable::new()));
            package.borrow_mut().entries.insert(
                LuaValue::String("loaded".to_string()),
                LuaValue::Table(loaded),
            );
            globals_guard.entries.insert(
                LuaValue::String("package".to_string()),
                LuaValue::Table(package),
            );
        }

        Self {
            stack: Vec::new(),
            trace: Vec::new(),
            globals: HashMap::new(),
            pending_error: None,
            next_userdata_id: 0,
            next_reference: 0,
            references: HashMap::new(),
            userdata_storage: HashMap::new(),
            userdata_metamethods: HashMap::new(),
            userdata_metatables: HashMap::new(),
            string_cache: Vec::new(),
            current_upvalues: Vec::new(),
            #[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
            libraries: Vec::new(),
            registry,
            globals_table,
        }
    }
}

impl LuaState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, value: LuaValue) {
        self.stack.push(value);
    }

    pub fn pop(&mut self) -> Option<LuaValue> {
        self.stack.pop()
    }

    pub fn len(&self) -> usize {
        self.stack.len()
    }

    pub fn record_call(&mut self, function: impl Into<String>, args: Vec<String>) {
        let function = function.into();
        let cfg = lua_trace_config();
        if cfg.enabled
            && cfg
                .filter
                .as_ref()
                .map(|filter| function.contains(filter))
                .unwrap_or(true)
        {
            if cfg.stack {
                eprintln!(
                    "[lua-api] {}({}) top={} stack={}",
                    function,
                    args.join(", "),
                    self.len(),
                    format_lua_stack_brief(&self.stack)
                );
            } else {
                eprintln!(
                    "[lua-api] {}({}) top={}",
                    function,
                    args.join(", "),
                    self.len()
                );
            }
        }
        self.trace.push(LuaApiCall { function, args });
    }

    pub fn take_trace(&mut self) -> Vec<LuaApiCall> {
        core::mem::take(&mut self.trace)
    }

    pub fn stack_snapshot(&self) -> Vec<LuaValue> {
        self.stack.clone()
    }

    fn next_func_id(&mut self) -> usize {
        next_lua_function_id()
    }

    fn next_userdata_id(&mut self) -> usize {
        let id = self.next_userdata_id;
        self.next_userdata_id += 1;
        id
    }

    fn next_ref(&mut self) -> i32 {
        self.next_reference += 1;
        self.next_reference
    }
}

fn state_from_ptr<'a>(ptr: *mut lua_State) -> Option<&'a mut LuaState> {
    unsafe { ptr.as_mut().map(|raw| &mut raw.state) }
}

fn translate_index(len: usize, idx: c_int) -> Option<usize> {
    if idx == 0 {
        return None;
    }
    if idx > 0 {
        let idx = idx as usize;
        if idx == 0 || idx > len {
            None
        } else {
            Some(idx - 1)
        }
    } else {
        let adj = len as isize + idx as isize;
        if adj < 0 {
            None
        } else {
            Some(adj as usize)
        }
    }
}

fn pseudo_table(state: &mut LuaState, idx: c_int) -> Option<LuaTableHandle> {
    match idx {
        LUA_REGISTRYINDEX => Some(state.registry.clone()),
        LUA_ENVIRONINDEX | LUA_GLOBALSINDEX => Some(state.globals_table.clone()),
        _ => None,
    }
}

fn value_at(state: &mut LuaState, idx: c_int) -> Option<LuaValue> {
    if let Some(table) = pseudo_table(state, idx) {
        return Some(LuaValue::Table(table));
    }
    // Lua 5.1 C-closure upvalues are addressed via `lua_upvalueindex(i)`, which expands to
    // `LUA_GLOBALSINDEX - i`.
    if idx < LUA_GLOBALSINDEX {
        let upvalue = (LUA_GLOBALSINDEX - idx) as isize;
        if upvalue > 0 {
            let slot = (upvalue - 1) as usize;
            return state.current_upvalues.get(slot).cloned();
        }
    }
    translate_index(state.stack.len(), idx).and_then(|slot| state.stack.get(slot).cloned())
}

fn ensure_table_at(state: &mut LuaState, idx: c_int) -> Option<LuaTableHandle> {
    if let Some(handle) = pseudo_table(state, idx) {
        return Some(handle);
    }
    let slot = translate_index(state.stack.len(), idx)?;
    match state.stack.get(slot) {
        Some(LuaValue::Table(handle)) => Some(handle.clone()),
        _ => None,
    }
}

fn ensure_child_table(parent: &LuaTableHandle, key: &str) -> LuaTableHandle {
    let mut guard = parent.borrow_mut();
    match guard.entries.entry(LuaValue::String(key.to_string())) {
        Entry::Occupied(mut entry) => match entry.get() {
            LuaValue::Table(handle) => handle.clone(),
            _ => {
                let table = Rc::new(RefCell::new(LuaTable::new()));
                entry.insert(LuaValue::Table(table.clone()));
                table
            }
        },
        Entry::Vacant(entry) => {
            let table = Rc::new(RefCell::new(LuaTable::new()));
            entry.insert(LuaValue::Table(table.clone()));
            table
        }
    }
}

fn cache_cstring<'a>(state: &'a mut LuaState, s: String) -> *const c_char {
    let owned = CString::new(s).unwrap_or_else(|_| CString::new("").unwrap());
    state.string_cache.push(owned);
    state
        .string_cache
        .last()
        .map(|c| c.as_ptr())
        .unwrap_or(core::ptr::null())
}

fn value_typecode(value: &LuaValue) -> c_int {
    match value {
        LuaValue::Nil => LUA_TNIL,
        LuaValue::Bool(_) => LUA_TBOOLEAN,
        LuaValue::LightUserdata(_) => LUA_TLIGHTUSERDATA,
        LuaValue::Int(_) | LuaValue::Float(_) => LUA_TNUMBER,
        LuaValue::String(_) => LUA_TSTRING,
        LuaValue::Table(_) => LUA_TTABLE,
        LuaValue::Function(_) => LUA_TFUNCTION,
        LuaValue::Userdata(_) => LUA_TUSERDATA,
        LuaValue::Thread(_) => LUA_TTHREAD,
    }
}

fn buffer_len(buf: &luaL_Buffer) -> usize {
    if buf.p.is_null() {
        return 0;
    }
    let start = buf.buffer.as_ptr();
    let diff = unsafe { buf.p.offset_from(start) };
    if diff < 0 {
        0
    } else {
        diff as usize
    }
}

/// Render a Lust extern stub for a Lua table value returned by a `luaopen_*` call.
pub fn render_table_stub(module_name: &str, handle: &LuaTableHandle) -> String {
    let mut functions = Vec::new();
    let mut values = Vec::new();
    for (key, value) in handle.borrow().entries.iter() {
        if let LuaValue::String(name) = key {
            match value {
                LuaValue::Function(_) => functions.push(name.clone()),
                _ => values.push(name.clone()),
            }
        }
    }
    if std::env::var("LUST_DEBUG_LUAOPEN").is_ok() {
        let mut keys: Vec<String> = handle
            .borrow()
            .entries
            .keys()
            .map(|k| format!("{:?}", k))
            .collect();
        keys.sort();
        eprintln!("luaopen '{}' table keys: {}", module_name, keys.join(", "));
    }
    functions.sort();
    values.sort();

    let mut out = String::new();
    out.push_str(&format!(
        "-- Auto-generated stub for Lua module '{}'\n\npub extern\n",
        module_name
    ));
    for func in functions {
        out.push_str(&format!(
            "    function {}.{}(LuaValue): LuaValue\n",
            module_name, func
        ));
    }
    for value in values {
        out.push_str(&format!("    const {}.{}: LuaValue\n", module_name, value));
    }
    out.push_str("end\n");
    out
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn luaopen_symbol_name(name: &str) -> String {
    if name.ends_with('\0') {
        name.to_string()
    } else {
        format!("{name}\0")
    }
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
fn luaopen_module_name(name: &str) -> String {
    name.strip_prefix("luaopen_")
        .unwrap_or(name)
        .replace('_', ".")
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
pub fn trace_luaopen(spec: &LuaModuleSpec) -> Result<Vec<LuaOpenResult>, String> {
    let library = Rc::new(unsafe { Library::new(&spec.library_path) }.map_err(|e| {
        format!(
            "failed to load Lua library '{}': {e}",
            spec.library_path.display()
        )
    })?);
    let mut results = Vec::new();

    for entry in &spec.entrypoints {
        let symbol_name = luaopen_symbol_name(entry);
        unsafe {
            let func = library
                .get::<unsafe extern "C" fn(*mut lua_State) -> c_int>(symbol_name.as_bytes())
                .map_err(|e| {
                    format!(
                        "missing symbol '{}' in {}: {e}",
                        entry,
                        spec.library_path.display()
                    )
                })?;
            let state_ptr = luaL_newstate();
            let ret_count = func(state_ptr);
            let shared_state: Rc<RefCell<Box<lua_State>>> =
                Rc::new(RefCell::new(Box::from_raw(state_ptr)));
            let mut boxed = shared_state.borrow_mut();
            boxed.state.libraries.push(library.clone());
            let mut returns = Vec::new();
            let count = if ret_count < 0 { 0 } else { ret_count as usize };
            for _ in 0..count {
                returns.push(boxed.state.pop().unwrap_or(LuaValue::Nil));
            }
            returns.reverse();
            if returns
                .iter()
                .all(|value| !matches!(value, LuaValue::Table(_)))
            {
                if let Some(table) =
                    boxed
                        .state
                        .stack_snapshot()
                        .into_iter()
                        .find_map(|value| match value {
                            LuaValue::Table(handle) => Some(handle),
                            _ => None,
                        })
                {
                    returns.insert(0, LuaValue::Table(table));
                }
            }
            if returns.is_empty() {
                if let Some(LuaValue::Table(handle)) = boxed.state.stack.last().cloned() {
                    returns.push(LuaValue::Table(handle));
                }
            }
            if returns.is_empty() {
                let module_name = luaopen_module_name(entry);
                let mut keys = vec![module_name.clone()];
                if let Some(first) = module_name.split('.').next() {
                    keys.push(first.to_string());
                }
                if let Some(loaded) = boxed
                    .state
                    .registry
                    .borrow()
                    .entries
                    .get(&LuaValue::String("_LOADED".to_string()))
                {
                    if let LuaValue::Table(tbl) = loaded {
                        for key in &keys {
                            let lookup = LuaValue::String(key.clone());
                            if let Some(val) = tbl.borrow().entries.get(&lookup).cloned() {
                                returns.push(val);
                                break;
                            }
                        }
                    }
                }
            }
            if returns.is_empty() {
                let module_name = luaopen_module_name(entry);
                let mut keys = vec![module_name.clone()];
                if let Some(first) = module_name.split('.').next() {
                    keys.push(first.to_string());
                }
                for key in keys {
                    let lookup = LuaValue::String(key);
                    if let Some(value) = boxed
                        .state
                        .globals_table
                        .borrow()
                        .entries
                        .get(&lookup)
                        .cloned()
                    {
                        returns.push(value);
                        break;
                    }
                }
            }
            if returns.is_empty() && std::env::var("LUST_DEBUG_LUAOPEN").is_ok() {
                eprintln!(
                    "luaopen '{}' returned no table; stack snapshot: {:?}, trace: {:?}",
                    entry,
                    boxed.state.stack_snapshot(),
                    boxed.state.trace
                );
            }
            if std::env::var("LUST_DEBUG_LUAOPEN").is_ok() {
                eprintln!("luaopen '{}' final returns: {:?}", entry, returns);
                eprintln!("luaopen '{}' trace: {:?}", entry, boxed.state.trace);
            }
            let trace = boxed.state.take_trace();
            drop(boxed);
            if std::env::var("LUST_DEBUG_LUAOPEN").is_ok() {
                let set_calls: Vec<_> = trace
                    .iter()
                    .filter(|c| c.function == "lua_setfield")
                    .collect();
                eprintln!("luaopen '{}' setfield calls: {:?}", entry, set_calls);
            }
            results.push(LuaOpenResult {
                module: luaopen_module_name(entry),
                trace,
                returns,
                state: Some(shared_state.clone()),
            });
        }
    }

    Ok(results)
}

/// --- C ABI shims ---

#[no_mangle]
pub unsafe extern "C" fn luaL_newstate() -> *mut lua_State {
    Box::into_raw(Box::new(lua_State {
        state: LuaState::new(),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn lua_newstate(
    _alloc: Option<unsafe extern "C" fn(*mut c_void, *mut c_void, usize, usize) -> *mut c_void>,
    _ud: *mut c_void,
) -> *mut lua_State {
    luaL_newstate()
}

#[no_mangle]
pub unsafe extern "C" fn lua_close(L: *mut lua_State) {
    if !L.is_null() {
        drop(Box::from_raw(L));
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_gettop(L: *mut lua_State) -> c_int {
    if let Some(state) = state_from_ptr(L) {
        let top = state.len() as c_int;
        state.record_call("lua_gettop", vec![]);
        top
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_settop(L: *mut lua_State, idx: c_int) {
    if let Some(state) = state_from_ptr(L) {
        let new_len = if idx >= 0 {
            idx as usize
        } else {
            let base = state.len() as isize + idx as isize + 1;
            if base < 0 {
                0
            } else {
                base as usize
            }
        };
        if new_len < state.stack.len() {
            state.stack.truncate(new_len);
        } else {
            while state.stack.len() < new_len {
                state.stack.push(LuaValue::Nil);
            }
        }
        state.record_call("lua_settop", vec![idx.to_string()]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_pushvalue(L: *mut lua_State, idx: c_int) {
    if let Some(state) = state_from_ptr(L) {
        if let Some(val) = value_at(state, idx) {
            state.push(val);
        }
        state.record_call("lua_pushvalue", vec![idx.to_string()]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_remove(L: *mut lua_State, idx: c_int) {
    if let Some(state) = state_from_ptr(L) {
        if pseudo_table(state, idx).is_some() {
            state.record_call("lua_remove", vec![idx.to_string()]);
            return;
        }
        if let Some(slot) = translate_index(state.stack.len(), idx) {
            state.stack.remove(slot);
        }
        state.record_call("lua_remove", vec![idx.to_string()]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_insert(L: *mut lua_State, idx: c_int) {
    if let Some(state) = state_from_ptr(L) {
        if let Some(slot) = translate_index(state.stack.len(), idx) {
            if let Some(val) = state.pop() {
                state.stack.insert(slot, val);
            }
        }
        state.record_call("lua_insert", vec![idx.to_string()]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_replace(L: *mut lua_State, idx: c_int) {
    if let Some(state) = state_from_ptr(L) {
        // Lua semantics: `lua_replace(L, idx)` is equivalent to `lua_copy(L, -1, idx); lua_pop(L, 1);`
        // The index is relative to the stack *before* the value is popped.
        let len_before_pop = state.stack.len();
        let slot = translate_index(len_before_pop, idx);
        let value = state.pop();
        if let (Some(slot), Some(value)) = (slot, value) {
            // After popping, the stack is 1 shorter. If `idx` referred to the old top (-1),
            // it no longer exists, and the operation is effectively just a pop.
            if slot < state.stack.len() {
                state.stack[slot] = value;
            }
        }
        state.record_call("lua_replace", vec![idx.to_string()]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_checkstack(L: *mut lua_State, _sz: c_int) -> c_int {
    if let Some(state) = state_from_ptr(L) {
        state.record_call("lua_checkstack", vec![_sz.to_string()]);
    }
    1
}

#[no_mangle]
pub unsafe extern "C" fn lua_type(L: *mut lua_State, idx: c_int) -> c_int {
    if let Some(state) = state_from_ptr(L) {
        let code = value_at(state, idx)
            .map(|value| value_typecode(&value))
            .unwrap_or(LUA_TNONE);
        state.record_call("lua_type", vec![idx.to_string()]);
        code
    } else {
        LUA_TNONE
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_typename(L: *mut lua_State, tp: c_int) -> *const c_char {
    const TYPE_NAMES: [&[u8]; 10] = [
        b"no value\0",
        b"nil\0",
        b"boolean\0",
        b"light userdata\0",
        b"number\0",
        b"string\0",
        b"table\0",
        b"function\0",
        b"userdata\0",
        b"thread\0",
    ];
    let raw = match tp {
        LUA_TNONE => TYPE_NAMES[0],
        LUA_TNIL => TYPE_NAMES[1],
        LUA_TBOOLEAN => TYPE_NAMES[2],
        LUA_TLIGHTUSERDATA => TYPE_NAMES[3],
        LUA_TNUMBER => TYPE_NAMES[4],
        LUA_TSTRING => TYPE_NAMES[5],
        LUA_TTABLE => TYPE_NAMES[6],
        LUA_TFUNCTION => TYPE_NAMES[7],
        LUA_TUSERDATA => TYPE_NAMES[8],
        LUA_TTHREAD => TYPE_NAMES[9],
        _ => b"<invalid>\0",
    };
    if let Some(state) = state_from_ptr(L) {
        state.record_call("lua_typename", vec![tp.to_string()]);
        return cache_cstring(state, String::from_utf8_lossy(raw).to_string());
    }
    raw.as_ptr() as *const c_char
}

#[no_mangle]
pub unsafe extern "C" fn lua_isstring(L: *mut lua_State, idx: c_int) -> c_int {
    matches!(lua_type(L, idx), LUA_TSTRING | LUA_TNUMBER) as c_int
}

fn parse_lua_number(text: &str) -> Option<lua_Number> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    let (sign, rest) = match trimmed.as_bytes().first() {
        Some(b'+') => (1.0, &trimmed[1..]),
        Some(b'-') => (-1.0, &trimmed[1..]),
        _ => (1.0, trimmed),
    };

    if let Some(hex) = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
        let hex = hex.trim();
        if hex.is_empty() {
            return None;
        }
        if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        let parsed = u64::from_str_radix(hex, 16).ok()? as lua_Number;
        return Some(parsed * sign);
    }

    rest.parse::<lua_Number>().ok().map(|v| v * sign)
}

#[no_mangle]
pub unsafe extern "C" fn lua_isnumber(L: *mut lua_State, idx: c_int) -> c_int {
    if let Some(state) = state_from_ptr(L) {
        state.record_call("lua_isnumber", vec![idx.to_string()]);
        match value_at(state, idx) {
            Some(LuaValue::Int(_)) | Some(LuaValue::Float(_)) => return 1,
            Some(LuaValue::String(s)) => return parse_lua_number(&s).is_some() as c_int,
            _ => return 0,
        }
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn lua_iscfunction(L: *mut lua_State, idx: c_int) -> c_int {
    matches!(lua_type(L, idx), LUA_TFUNCTION) as c_int
}

#[no_mangle]
pub unsafe extern "C" fn lua_istable(L: *mut lua_State, idx: c_int) -> c_int {
    (lua_type(L, idx) == LUA_TTABLE) as c_int
}

#[no_mangle]
pub unsafe extern "C" fn lua_isuserdata(L: *mut lua_State, idx: c_int) -> c_int {
    matches!(lua_type(L, idx), LUA_TUSERDATA | LUA_TLIGHTUSERDATA) as c_int
}

#[no_mangle]
pub unsafe extern "C" fn lua_toboolean(L: *mut lua_State, idx: c_int) -> c_int {
    if let Some(state) = state_from_ptr(L) {
        state.record_call("lua_toboolean", vec![idx.to_string()]);
    }
    match lua_type(L, idx) {
        LUA_TNIL | LUA_TNONE => 0,
        LUA_TBOOLEAN => {
            if let Some(state) = state_from_ptr(L) {
                if let Some(LuaValue::Bool(b)) = value_at(state, idx) {
                    return b as c_int;
                }
            }
            0
        }
        _ => 1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_tonumber(L: *mut lua_State, idx: c_int) -> lua_Number {
    if let Some(state) = state_from_ptr(L) {
        state.record_call("lua_tonumber", vec![idx.to_string()]);
        match value_at(state, idx) {
            Some(LuaValue::Float(f)) => f,
            Some(LuaValue::Int(i)) => i as lua_Number,
            Some(LuaValue::String(s)) => parse_lua_number(&s).unwrap_or(0.0),
            _ => 0.0,
        }
    } else {
        0.0
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_tointeger(L: *mut lua_State, idx: c_int) -> lua_Integer {
    if let Some(state) = state_from_ptr(L) {
        state.record_call("lua_tointeger", vec![idx.to_string()]);
        match value_at(state, idx) {
            Some(LuaValue::Int(i)) => i,
            Some(LuaValue::Float(f)) => f as lua_Integer,
            Some(LuaValue::String(s)) => {
                parse_lua_number(&s).map(|v| v as lua_Integer).unwrap_or(0)
            }
            _ => 0,
        }
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_tolstring(
    L: *mut lua_State,
    idx: c_int,
    len: *mut usize,
) -> *const c_char {
    let mut ptr = core::ptr::null();
    if let Some(state) = state_from_ptr(L) {
        state.record_call("lua_tolstring", vec![idx.to_string()]);
        if let Some(value) = value_at(state, idx) {
            let s = match value {
                LuaValue::String(s) => s,
                LuaValue::Int(i) => i.to_string(),
                LuaValue::Float(f) => f.to_string(),
                LuaValue::Bool(b) => b.to_string(),
                _ => String::new(),
            };
            ptr = cache_cstring(state, s.clone());
            if !len.is_null() {
                *len = s.len();
            }
        }
    }
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn lua_objlen(L: *mut lua_State, idx: c_int) -> usize {
    let mut length = 0usize;
    if let Some(state) = state_from_ptr(L) {
        state.record_call("lua_objlen", vec![idx.to_string()]);
        if let Some(value) = value_at(state, idx) {
            match value {
                LuaValue::String(s) => length = s.len(),
                LuaValue::Table(handle) => {
                    // Lua 5.1 length operator is sequence-like for tables.
                    let table = handle.borrow();
                    let mut i: lua_Integer = 1;
                    loop {
                        if table.entries.contains_key(&LuaValue::Int(i)) {
                            i += 1;
                        } else {
                            break;
                        }
                    }
                    length = (i - 1).max(0) as usize;
                }
                LuaValue::Userdata(userdata) => {
                    if let Some((_blob, size)) = state.userdata_storage.get(&userdata.id) {
                        length = *size;
                    }
                }
                _ => {}
            }
        }
    }
    length
}

#[no_mangle]
pub unsafe extern "C" fn lua_equal(L: *mut lua_State, idx1: c_int, idx2: c_int) -> c_int {
    let t1 = lua_type(L, idx1);
    let t2 = lua_type(L, idx2);
    if t1 == LUA_TNONE || t2 == LUA_TNONE {
        return 0;
    }
    if let Some(state) = state_from_ptr(L) {
        if let (Some(v1), Some(v2)) = (value_at(state, idx1), value_at(state, idx2)) {
            return (v1 == v2) as c_int;
        }
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn lua_rawequal(L: *mut lua_State, idx1: c_int, idx2: c_int) -> c_int {
    lua_equal(L, idx1, idx2)
}

#[no_mangle]
pub unsafe extern "C" fn lua_lessthan(L: *mut lua_State, _idx1: c_int, _idx2: c_int) -> c_int {
    // Minimal stub: ordering not tracked.
    if let Some(state) = state_from_ptr(L) {
        state.record_call("lua_lessthan", vec![]);
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn lua_pushnil(L: *mut lua_State) {
    if let Some(state) = state_from_ptr(L) {
        state.push(LuaValue::Nil);
        state.record_call("lua_pushnil", vec![]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_pushnumber(L: *mut lua_State, n: lua_Number) {
    if let Some(state) = state_from_ptr(L) {
        state.push(LuaValue::Float(n));
        state.record_call("lua_pushnumber", vec![n.to_string()]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_pushinteger(L: *mut lua_State, n: lua_Integer) {
    if let Some(state) = state_from_ptr(L) {
        state.push(LuaValue::Int(n));
        state.record_call("lua_pushinteger", vec![n.to_string()]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_pushlstring(L: *mut lua_State, s: *const c_char, len: usize) {
    if let Some(state) = state_from_ptr(L) {
        let string = if s.is_null() || len == 0 {
            String::new()
        } else {
            let slice = core::slice::from_raw_parts(s as *const u8, len);
            String::from_utf8_lossy(slice).to_string()
        };
        state.push(LuaValue::String(string.clone()));
        state.record_call(
            "lua_pushlstring",
            vec![format!("len={}", len), format!("text={:?}", string)],
        );
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_pushstring(L: *mut lua_State, s: *const c_char) {
    if let Some(state) = state_from_ptr(L) {
        let text = if s.is_null() {
            String::new()
        } else {
            CStr::from_ptr(s).to_string_lossy().to_string()
        };
        state.push(LuaValue::String(text.clone()));
        state.record_call("lua_pushstring", vec![format!("text={:?}", text)]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_pushfstring(L: *mut lua_State, fmt: *const c_char) -> *const c_char {
    let mut text = String::new();
    if !fmt.is_null() {
        text = CStr::from_ptr(fmt).to_string_lossy().to_string();
    }
    if let Some(state) = state_from_ptr(L) {
        state.push(LuaValue::String(text.clone()));
        let ptr = cache_cstring(state, text);
        state.record_call("lua_pushfstring", vec![]);
        return ptr;
    }
    core::ptr::null()
}

#[no_mangle]
pub unsafe extern "C" fn lua_pushboolean(L: *mut lua_State, b: c_int) {
    if let Some(state) = state_from_ptr(L) {
        state.push(LuaValue::Bool(b != 0));
        state.record_call("lua_pushboolean", vec![b.to_string()]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_pushlightuserdata(L: *mut lua_State, p: *mut c_void) {
    if let Some(state) = state_from_ptr(L) {
        state.push(LuaValue::LightUserdata(p as usize));
        state.record_call("lua_pushlightuserdata", vec![format!("{p:p}")]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_pushcclosure(L: *mut lua_State, f: lua_CFunction, _n: c_int) {
    if let Some(state) = state_from_ptr(L) {
        let n = _n.max(0) as usize;
        let len = state.stack.len();
        let upvalues = if n > 0 && n <= len {
            state.stack.split_off(len - n)
        } else {
            Vec::new()
        };
        let id = state.next_func_id();
        state.push(LuaValue::Function(LuaFunction {
            id,
            name: None,
            cfunc: f,
            lust_handle: None,
            upvalues,
        }));
        let addr = f.map(|func| func as usize).unwrap_or(0);
        state.record_call(
            "lua_pushcclosure",
            vec![format!("n={}", _n), format!("f={:#x}", addr)],
        );
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_pushcfunction(L: *mut lua_State, f: lua_CFunction) {
    lua_pushcclosure(L, f, 0);
}

#[no_mangle]
pub unsafe extern "C" fn lua_newtable(L: *mut lua_State) {
    if let Some(state) = state_from_ptr(L) {
        state.push(LuaValue::Table(Rc::new(RefCell::new(LuaTable::new()))));
        state.record_call("lua_newtable", vec![]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_createtable(L: *mut lua_State, _narr: c_int, _nrec: c_int) {
    lua_newtable(L);
}

#[no_mangle]
pub unsafe extern "C" fn lua_gettable(L: *mut lua_State, idx: c_int) {
    if let Some(state) = state_from_ptr(L) {
        // If `idx` is negative, it is relative to the stack *before* the key is popped.
        let handle = ensure_table_at(state, idx);
        if let Some(key) = state.pop() {
            if let Some(handle) = handle {
                let value = handle
                    .borrow()
                    .entries
                    .get(&key)
                    .cloned()
                    .unwrap_or(LuaValue::Nil);
                state.push(value);
            } else {
                state.push(LuaValue::Nil);
            }
        }
        state.record_call("lua_gettable", vec![idx.to_string()]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_settable(L: *mut lua_State, idx: c_int) {
    if let Some(state) = state_from_ptr(L) {
        // If `idx` is negative, it is relative to the stack *before* the key/value are popped.
        let handle = ensure_table_at(state, idx);
        let value = state.pop();
        let key = state.pop();
        if let (Some(k), Some(v)) = (key, value) {
            if let Some(handle) = handle {
                let v = match (&k, v) {
                    (LuaValue::String(name), LuaValue::Function(mut func)) => {
                        if func.name.is_none() {
                            func.name = Some(name.clone());
                        }
                        LuaValue::Function(func)
                    }
                    (_, other) => other,
                };
                handle.borrow_mut().entries.insert(k, v);
            }
        }
        state.record_call("lua_settable", vec![idx.to_string()]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_getfield(L: *mut lua_State, idx: c_int, k: *const c_char) {
    if let Some(state) = state_from_ptr(L) {
        let key = if k.is_null() {
            String::new()
        } else {
            CStr::from_ptr(k).to_string_lossy().to_string()
        };
        if let Some(handle) = ensure_table_at(state, idx) {
            let value = handle
                .borrow()
                .entries
                .get(&LuaValue::String(key.clone()))
                .cloned()
                .unwrap_or(LuaValue::Nil);
            state.push(value);
        } else {
            state.push(LuaValue::Nil);
        }
        state.record_call("lua_getfield", vec![idx.to_string(), key]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_setfield(L: *mut lua_State, idx: c_int, k: *const c_char) {
    if let Some(state) = state_from_ptr(L) {
        // If `idx` is negative, it is relative to the stack *before* the value is popped.
        let handle = ensure_table_at(state, idx);
        let value = state.pop();
        let key = if k.is_null() {
            String::new()
        } else {
            CStr::from_ptr(k).to_string_lossy().to_string()
        };
        if let Some(v) = value {
            if let Some(handle) = handle {
                let v = match v {
                    LuaValue::Function(mut func) => {
                        if func.name.is_none() && !key.is_empty() {
                            func.name = Some(key.clone());
                        }
                        LuaValue::Function(func)
                    }
                    other => other,
                };
                handle
                    .borrow_mut()
                    .entries
                    .insert(LuaValue::String(key.clone()), v);
            }
        }
        state.record_call("lua_setfield", vec![idx.to_string(), key]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_next(L: *mut lua_State, _idx: c_int) -> c_int {
    if let Some(state) = state_from_ptr(L) {
        let handle = ensure_table_at(state, _idx);
        let key = state.pop().unwrap_or(LuaValue::Nil);
        if let Some(handle) = handle {
            let table = handle.borrow();
            let keys: Vec<LuaValue> = table.entries.keys().cloned().collect();
            let mut next_key: Option<LuaValue> = None;

            if matches!(key, LuaValue::Nil) {
                next_key = keys.get(0).cloned();
            } else {
                let mut seen_current = false;
                for k in &keys {
                    if !seen_current {
                        if *k == key {
                            seen_current = true;
                        }
                        continue;
                    }
                    next_key = Some(k.clone());
                    break;
                }
            }

            if let Some(k) = next_key {
                let v = table.entries.get(&k).cloned().unwrap_or(LuaValue::Nil);
                state.push(k);
                state.push(v);
                state.record_call("lua_next", vec![_idx.to_string()]);
                return 1;
            }
        }

        state.record_call("lua_next", vec![_idx.to_string()]);
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn lua_rawget(L: *mut lua_State, idx: c_int) {
    lua_gettable(L, idx)
}

#[no_mangle]
pub unsafe extern "C" fn lua_rawgeti(L: *mut lua_State, idx: c_int, n: c_int) {
    if let Some(state) = state_from_ptr(L) {
        if let Some(handle) = ensure_table_at(state, idx) {
            let value = handle
                .borrow()
                .entries
                .get(&LuaValue::Int(n as lua_Integer))
                .cloned()
                .unwrap_or(LuaValue::Nil);
            state.push(value);
        } else {
            state.push(LuaValue::Nil);
        }
        state.record_call("lua_rawgeti", vec![idx.to_string(), n.to_string()]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_rawset(L: *mut lua_State, idx: c_int) {
    lua_settable(L, idx)
}

#[no_mangle]
pub unsafe extern "C" fn lua_rawseti(L: *mut lua_State, idx: c_int, n: c_int) {
    if let Some(state) = state_from_ptr(L) {
        // If `idx` is negative, it is relative to the stack *before* the value is popped.
        let handle = ensure_table_at(state, idx);
        let value = state.pop().unwrap_or(LuaValue::Nil);
        if let Some(handle) = handle {
            handle
                .borrow_mut()
                .entries
                .insert(LuaValue::Int(n as lua_Integer), value);
        }
        state.record_call("lua_rawseti", vec![idx.to_string(), n.to_string()]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_concat(L: *mut lua_State, n: c_int) {
    if let Some(state) = state_from_ptr(L) {
        let mut parts: Vec<String> = Vec::new();
        for _ in 0..n {
            if let Some(val) = state.pop() {
                match val {
                    LuaValue::String(s) => parts.push(s),
                    LuaValue::Int(i) => parts.push(i.to_string()),
                    LuaValue::Float(f) => parts.push(f.to_string()),
                    LuaValue::Bool(b) => parts.push(b.to_string()),
                    _ => parts.push(String::new()),
                }
            }
        }
        parts.reverse();
        state.push(LuaValue::String(parts.join("")));
        state.record_call("lua_concat", vec![n.to_string()]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_setmetatable(L: *mut lua_State, objindex: c_int) -> c_int {
    if let Some(state) = state_from_ptr(L) {
        let len_before_pop = state.stack.len();
        let meta = state.pop();
        if let Some(LuaValue::Table(meta_handle)) = meta {
            let index_value = meta_handle
                .borrow()
                .entries
                .get(&LuaValue::String("__index".to_string()))
                .cloned()
                .unwrap_or_else(|| LuaValue::Table(meta_handle.clone()));

            if let Some(handle) = pseudo_table(state, objindex) {
                handle
                    .borrow_mut()
                    .metamethods
                    .insert("__index".to_string(), index_value.clone());
                state.record_call("lua_setmetatable", vec![objindex.to_string()]);
                return 1;
            }

            if let Some(slot) = translate_index(len_before_pop, objindex) {
                match state.stack.get(slot) {
                    Some(LuaValue::Table(handle)) => {
                        handle.borrow_mut().metatable = Some(meta_handle.clone());
                        handle
                            .borrow_mut()
                            .metamethods
                            .insert("__index".to_string(), index_value.clone());
                        state.record_call("lua_setmetatable", vec![objindex.to_string()]);
                        return 1;
                    }
                    Some(LuaValue::Userdata(userdata)) => {
                        state
                            .userdata_metatables
                            .insert(userdata.id, meta_handle.clone());
                        state
                            .userdata_metamethods
                            .entry(userdata.id)
                            .or_default()
                            .insert("__index".to_string(), index_value.clone());
                        state.record_call("lua_setmetatable", vec![objindex.to_string()]);
                        return 1;
                    }
                    _ => {}
                }
            }
        } else if matches!(meta, None | Some(LuaValue::Nil)) {
            if let Some(slot) = translate_index(len_before_pop, objindex) {
                match state.stack.get(slot) {
                    Some(LuaValue::Table(handle)) => {
                        handle.borrow_mut().metatable = None;
                        handle.borrow_mut().metamethods.remove("__index");
                        state.record_call("lua_setmetatable", vec![objindex.to_string()]);
                        return 1;
                    }
                    Some(LuaValue::Userdata(userdata)) => {
                        state.userdata_metatables.remove(&userdata.id);
                        state.userdata_metamethods.remove(&userdata.id);
                        state.record_call("lua_setmetatable", vec![objindex.to_string()]);
                        return 1;
                    }
                    _ => {}
                }
            }
        }
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn lua_getmetatable(L: *mut lua_State, objindex: c_int) -> c_int {
    if let Some(state) = state_from_ptr(L) {
        if let Some(handle) = ensure_table_at(state, objindex) {
            if let Some(meta) = handle.borrow().metatable.clone() {
                state.push(LuaValue::Table(meta));
                state.record_call("lua_getmetatable", vec![objindex.to_string()]);
                return 1;
            }
        }
        if let Some(LuaValue::Userdata(userdata)) = value_at(state, objindex) {
            if let Some(meta) = state.userdata_metatables.get(&userdata.id).cloned() {
                state.push(LuaValue::Table(meta));
                state.record_call("lua_getmetatable", vec![objindex.to_string()]);
                return 1;
            }
        }
    }
    0
}

fn push_lua_results(state: &mut LuaState, mut results: Vec<LuaValue>, nresults: c_int) {
    if nresults >= 0 {
        let target = nresults as usize;
        if results.len() > target {
            results.truncate(target);
        } else {
            while results.len() < target {
                results.push(LuaValue::Nil);
            }
        }
    }
    for value in results {
        state.push(value);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_call(L: *mut lua_State, nargs: c_int, nresults: c_int) {
    if let Some(state) = state_from_ptr(L) {
        let mut args: Vec<LuaValue> = Vec::new();
        for _ in 0..nargs {
            args.push(state.pop().unwrap_or(LuaValue::Nil));
        }
        let func = state.pop();
        let base_len = state.stack.len();
        let mut results: Vec<LuaValue> = Vec::new();
        args.reverse();
        if let Some(LuaValue::Function(f)) = func {
            if let Some(handle) = f.lust_handle {
                match VM::with_current(|vm| {
                    let mut converted = Vec::new();
                    for arg in &args {
                        converted.push(lua_to_lust(arg, vm, None)?);
                    }
                    let func_value = lookup_lust_function(handle).ok_or_else(|| {
                        format!("Missing Lust function for LuaValue handle {}", handle)
                    })?;
                    vm.call_value(&func_value, converted)
                        .map_err(|e| e.to_string())
                }) {
                    Ok(ret) => {
                        let tuple_values: Vec<Value> = if let Value::Tuple(values) = &ret {
                            values.iter().cloned().collect()
                        } else {
                            vec![ret]
                        };
                        for value in tuple_values {
                            if let Ok(lua_ret) = VM::with_current(|vm| Ok(value_to_lua(&value, vm)))
                            {
                                results.push(lua_ret);
                            } else {
                                results.push(LuaValue::Nil);
                            }
                        }
                    }
                    Err(err) => {
                        eprintln!("lua_call Lust dispatch failed: {}", err);
                        results.push(LuaValue::Nil);
                    }
                }
            } else if let Some(cfunc) = f.cfunc {
                // C functions assume a fresh call frame where their arguments start at stack index 1.
                // Our shim stores the stack as a flat Vec, so we temporarily swap in an isolated frame.
                let caller_stack = core::mem::take(&mut state.stack);
                for arg in args {
                    state.push(arg);
                }
                let cfg = lua_trace_config();
                if cfg.enabled || cfg.cfunc {
                    let label = f.name.as_deref().unwrap_or("<anonymous>");
                    eprintln!(
                        "[lua-cfunc] lua_call -> {} {:#x} nargs={} top={} stack={}",
                        label,
                        cfunc as usize,
                        nargs,
                        state.len(),
                        format_lua_stack_brief(&state.stack)
                    );
                }
                let ret_count = cfunc(L);
                if cfg.enabled || cfg.cfunc {
                    let label = f.name.as_deref().unwrap_or("<anonymous>");
                    eprintln!(
                        "[lua-cfunc] lua_call <- {} {:#x} nret={} top={} stack={}",
                        label,
                        cfunc as usize,
                        ret_count,
                        state.len(),
                        format_lua_stack_brief(&state.stack)
                    );
                }
                if state.pending_error.is_none() && ret_count > 0 {
                    for _ in 0..ret_count {
                        results.push(state.pop().unwrap_or(LuaValue::Nil));
                    }
                    results.reverse();
                }
                state.stack = caller_stack;
            }
        }
        state.stack.truncate(base_len);
        state.record_call("lua_call", vec![nargs.to_string(), nresults.to_string()]);
        push_lua_results(state, results, nresults);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lua_pcall(
    L: *mut lua_State,
    nargs: c_int,
    nresults: c_int,
    _errfunc: c_int,
) -> c_int {
    if let Some(state) = state_from_ptr(L) {
        let mut args: Vec<LuaValue> = Vec::new();
        for _ in 0..nargs {
            args.push(state.pop().unwrap_or(LuaValue::Nil));
        }
        let func = state.pop();
        let base_len = state.stack.len();
        args.reverse();

        let mut results: Vec<LuaValue> = Vec::new();
        let mut error_obj: Option<LuaValue> = None;

        match func {
            Some(LuaValue::Function(f)) => {
                if let Some(handle) = f.lust_handle {
                    match VM::with_current(|vm| {
                        let mut converted = Vec::new();
                        for arg in &args {
                            converted.push(lua_to_lust(arg, vm, None)?);
                        }
                        let func_value = lookup_lust_function(handle).ok_or_else(|| {
                            format!("Missing Lust function for LuaValue handle {}", handle)
                        })?;
                        vm.call_value(&func_value, converted)
                            .map_err(|e| e.to_string())
                    }) {
                        Ok(ret) => {
                            let tuple_values: Vec<Value> = if let Value::Tuple(values) = &ret {
                                values.iter().cloned().collect()
                            } else {
                                vec![ret]
                            };
                            for value in tuple_values {
                                if let Ok(lua_ret) =
                                    VM::with_current(|vm| Ok(value_to_lua(&value, vm)))
                                {
                                    results.push(lua_ret);
                                } else {
                                    results.push(LuaValue::Nil);
                                }
                            }
                        }
                        Err(err) => {
                            error_obj = Some(LuaValue::String(err));
                        }
                    }
                } else if let Some(cfunc) = f.cfunc {
                    // C functions assume a fresh call frame where their arguments start at stack index 1.
                    let caller_stack = core::mem::take(&mut state.stack);
                    for arg in args {
                        state.push(arg);
                    }
                    let ret_count = cfunc(L);
                    if let Some(err) = state.pending_error.take() {
                        error_obj = Some(err);
                    } else if ret_count < 0 {
                        error_obj = Some(LuaValue::String("lua_error".to_string()));
                    } else if ret_count > 0 {
                        for _ in 0..ret_count {
                            results.push(state.pop().unwrap_or(LuaValue::Nil));
                        }
                        results.reverse();
                    }
                    state.stack = caller_stack;
                } else {
                    error_obj = Some(LuaValue::String(
                        "attempt to call a non-callable function value".to_string(),
                    ));
                }
            }
            Some(other) => {
                error_obj = Some(LuaValue::String(format!(
                    "attempt to call {}",
                    format_lua_value_brief(&other)
                )));
            }
            None => {
                error_obj = Some(LuaValue::String("attempt to call nil".to_string()));
            }
        }

        state.stack.truncate(base_len);
        state.record_call("lua_pcall", vec![nargs.to_string(), nresults.to_string()]);

        if let Some(err) = error_obj {
            state.push(err);
            return LUA_ERRRUN;
        }

        push_lua_results(state, results, nresults);
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn lua_error(L: *mut lua_State) -> c_int {
    if let Some(state) = state_from_ptr(L) {
        let err = state.pop().unwrap_or(LuaValue::Nil);
        state.pending_error = Some(err);
        state.record_call("lua_error", vec![]);
    }
    -1
}

#[no_mangle]
pub unsafe extern "C" fn luaL_argerror(
    L: *mut lua_State,
    narg: c_int,
    extramsg: *const c_char,
) -> c_int {
    if let Some(state) = state_from_ptr(L) {
        let msg = if extramsg.is_null() {
            "<unknown>".to_string()
        } else {
            CStr::from_ptr(extramsg).to_string_lossy().to_string()
        };
        state.pending_error = Some(LuaValue::String(msg.clone()));
        state.record_call("luaL_argerror", vec![narg.to_string(), msg]);
    }
    -1
}

#[no_mangle]
pub unsafe extern "C" fn luaL_checkstack(
    L: *mut lua_State,
    sz: c_int,
    msg: *const c_char,
) -> c_int {
    if let Some(state) = state_from_ptr(L) {
        let text = if msg.is_null() {
            String::new()
        } else {
            CStr::from_ptr(msg).to_string_lossy().to_string()
        };
        state.record_call("luaL_checkstack", vec![sz.to_string(), text]);
    }
    lua_checkstack(L, sz)
}

#[no_mangle]
pub unsafe extern "C" fn luaL_checktype(L: *mut lua_State, narg: c_int, t: c_int) {
    if let Some(state) = state_from_ptr(L) {
        state.record_call("luaL_checktype", vec![narg.to_string(), t.to_string()]);
    }
    let current = lua_type(L, narg);
    if current == t || t == LUA_TNONE {
        return;
    }
    let _ = luaL_argerror(L, narg, core::ptr::null());
}

#[no_mangle]
pub unsafe extern "C" fn luaL_checkoption(
    L: *mut lua_State,
    idx: c_int,
    def: *const c_char,
    lst: *const *const c_char,
) -> c_int {
    let chosen = if lua_type(L, idx) == LUA_TNONE && !def.is_null() {
        CStr::from_ptr(def).to_string_lossy().to_string()
    } else {
        let mut len: usize = 0;
        let ptr = lua_tolstring(L, idx, &mut len as *mut usize);
        if ptr.is_null() {
            String::new()
        } else {
            let slice = core::slice::from_raw_parts(ptr as *const u8, len);
            String::from_utf8_lossy(slice).to_string()
        }
    };

    let mut i: c_int = 0;
    let mut iter = lst;
    while !iter.is_null() && !(*iter).is_null() {
        let option = CStr::from_ptr(*iter).to_string_lossy().to_string();
        if option == chosen {
            if let Some(state) = state_from_ptr(L) {
                state.record_call("luaL_checkoption", vec![idx.to_string(), option]);
            }
            return i;
        }
        i += 1;
        iter = iter.add(1);
    }

    luaL_argerror(L, idx, core::ptr::null());
    0
}

#[no_mangle]
pub unsafe extern "C" fn luaL_openlib(
    L: *mut lua_State,
    libname: *const c_char,
    regs: *const luaL_Reg,
    _nup: c_int,
) {
    luaL_register(L, libname, regs);
}

#[no_mangle]
pub unsafe extern "C" fn luaL_register(
    L: *mut lua_State,
    libname: *const c_char,
    regs: *const luaL_Reg,
) {
    if let Some(state) = state_from_ptr(L) {
        let name =
            (!libname.is_null()).then(|| CStr::from_ptr(libname).to_string_lossy().to_string());

        // Ensure table is on the stack, creating nested globals for dotted names.
        let target_handle = if let Some(module) = &name {
            let segments: Vec<&str> = module.split('.').collect();
            if segments.is_empty() {
                lua_newtable(L);
                ensure_table_at(state, -1)
            } else {
                let mut current = state.globals_table.clone();
                if segments.len() > 1 {
                    for seg in &segments[..segments.len() - 1] {
                        current = ensure_child_table(&current, seg);
                    }
                }
                let leaf = segments.last().unwrap().to_string();
                let leaf_handle = ensure_child_table(&current, &leaf);
                state.push(LuaValue::Table(leaf_handle.clone()));
                Some(leaf_handle)
            }
        } else {
            if !matches!(state.stack.last(), Some(LuaValue::Table(_))) {
                lua_newtable(L);
            }
            ensure_table_at(state, -1)
        };

        if let Some(handle) = target_handle {
            let mut iter = regs;
            while !iter.is_null() && !(*iter).name.is_null() {
                let entry = &*iter;
                let key = CStr::from_ptr(entry.name).to_string_lossy().to_string();
                let id = state.next_func_id();
                handle.borrow_mut().entries.insert(
                    LuaValue::String(key.clone()),
                    LuaValue::Function(LuaFunction {
                        id,
                        name: Some(key.clone()),
                        cfunc: entry.func,
                        lust_handle: None,
                        upvalues: Vec::new(),
                    }),
                );
                iter = iter.add(1);
            }
            if let Some(module) = name {
                state
                    .globals
                    .insert(module.clone(), LuaValue::Table(handle.clone()));
                state.globals_table.borrow_mut().entries.insert(
                    LuaValue::String(module.clone()),
                    LuaValue::Table(handle.clone()),
                );
                if let Some(loaded) = state
                    .registry
                    .borrow_mut()
                    .entries
                    .get_mut(&LuaValue::String("_LOADED".to_string()))
                {
                    if let LuaValue::Table(tbl) = loaded {
                        tbl.borrow_mut().entries.insert(
                            LuaValue::String(module.clone()),
                            LuaValue::Table(handle.clone()),
                        );
                    }
                }
                state.record_call("luaL_register", vec![module]);
            } else {
                state.record_call("luaL_register", vec!["<anonymous>".to_string()]);
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn luaL_newmetatable(L: *mut lua_State, tname: *const c_char) -> c_int {
    let mut created = 0;
    if let Some(state) = state_from_ptr(L) {
        let name = if tname.is_null() {
            "<unknown>".to_string()
        } else {
            CStr::from_ptr(tname).to_string_lossy().to_string()
        };
        let key = LuaValue::String(name.clone());
        if !state.registry.borrow().entries.contains_key(&key) {
            lua_newtable(L);
            if let Some(meta) = state.stack.last().cloned() {
                state
                    .registry
                    .borrow_mut()
                    .entries
                    .insert(key.clone(), meta);
            }
            created = 1;
        } else {
            let meta = state
                .registry
                .borrow()
                .entries
                .get(&key)
                .cloned()
                .unwrap_or_else(|| LuaValue::Table(Rc::new(RefCell::new(LuaTable::new()))));
            state.push(meta);
        }
        state.record_call("luaL_newmetatable", vec![name]);
    }
    created
}

#[no_mangle]
pub unsafe extern "C" fn luaL_getmetatable(L: *mut lua_State, tname: *const c_char) {
    if let Some(state) = state_from_ptr(L) {
        let name = if tname.is_null() {
            "<unknown>".to_string()
        } else {
            CStr::from_ptr(tname).to_string_lossy().to_string()
        };
        let key = LuaValue::String(name.clone());
        let value = state
            .registry
            .borrow()
            .entries
            .get(&key)
            .cloned()
            .unwrap_or(LuaValue::Nil);
        state.push(value);
        state.record_call("luaL_getmetatable", vec![name]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn luaL_checklstring(
    L: *mut lua_State,
    idx: c_int,
    len: *mut usize,
) -> *const c_char {
    let ptr = lua_tolstring(L, idx, len);
    if let Some(state) = state_from_ptr(L) {
        state.record_call("luaL_checklstring", vec![idx.to_string()]);
    }
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn luaL_checkstring(L: *mut lua_State, idx: c_int) -> *const c_char {
    luaL_checklstring(L, idx, core::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn luaL_optlstring(
    L: *mut lua_State,
    idx: c_int,
    def: *const c_char,
    len: *mut usize,
) -> *const c_char {
    let ty = lua_type(L, idx);
    if ty == LUA_TNONE || ty == LUA_TNIL {
        if let Some(state) = state_from_ptr(L) {
            if def.is_null() {
                if !len.is_null() {
                    *len = 0;
                }
                state.record_call("luaL_optlstring", vec![idx.to_string()]);
                return core::ptr::null();
            }
            let default = CStr::from_ptr(def).to_string_lossy().to_string();
            let ptr = cache_cstring(state, default.clone());
            if !len.is_null() {
                *len = default.len();
            }
            state.record_call("luaL_optlstring", vec![idx.to_string()]);
            return ptr;
        }
    }
    lua_tolstring(L, idx, len)
}

#[no_mangle]
pub unsafe extern "C" fn luaL_checknumber(L: *mut lua_State, idx: c_int) -> lua_Number {
    let n = lua_tonumber(L, idx);
    if let Some(state) = state_from_ptr(L) {
        state.record_call("luaL_checknumber", vec![idx.to_string()]);
    }
    n
}

#[no_mangle]
pub unsafe extern "C" fn luaL_checkinteger(L: *mut lua_State, idx: c_int) -> lua_Integer {
    let n = lua_tointeger(L, idx);
    if let Some(state) = state_from_ptr(L) {
        state.record_call("luaL_checkinteger", vec![idx.to_string()]);
    }
    n
}

#[no_mangle]
pub unsafe extern "C" fn luaL_optnumber(
    L: *mut lua_State,
    idx: c_int,
    def: lua_Number,
) -> lua_Number {
    if lua_type(L, idx) == LUA_TNONE {
        def
    } else {
        lua_tonumber(L, idx)
    }
}

#[no_mangle]
pub unsafe extern "C" fn luaL_optinteger(
    L: *mut lua_State,
    idx: c_int,
    def: lua_Integer,
) -> lua_Integer {
    if lua_type(L, idx) == LUA_TNONE {
        def
    } else {
        lua_tointeger(L, idx)
    }
}

#[no_mangle]
pub unsafe extern "C" fn luaL_ref(L: *mut lua_State, _t: c_int) -> c_int {
    if let Some(state) = state_from_ptr(L) {
        let value = state.pop().unwrap_or(LuaValue::Nil);
        let reference = state.next_ref();
        state.references.insert(reference, value);
        state.record_call("luaL_ref", vec![reference.to_string()]);
        return reference;
    }
    -1
}

#[no_mangle]
pub unsafe extern "C" fn luaL_unref(L: *mut lua_State, _t: c_int, r: c_int) {
    if let Some(state) = state_from_ptr(L) {
        state.references.remove(&r);
        state.record_call("luaL_unref", vec![r.to_string()]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn luaL_error(L: *mut lua_State, s: *const c_char) -> c_int {
    if let Some(state) = state_from_ptr(L) {
        let msg = if s.is_null() {
            "<unknown>".to_string()
        } else {
            CStr::from_ptr(s).to_string_lossy().to_string()
        };
        state.pending_error = Some(LuaValue::String(msg.clone()));
        state.record_call("luaL_error", vec![msg]);
    }
    -1
}

#[no_mangle]
pub unsafe extern "C" fn luaL_loadbuffer(
    L: *mut lua_State,
    _buff: *const c_char,
    _size: usize,
    _name: *const c_char,
) -> c_int {
    if let Some(state) = state_from_ptr(L) {
        state.record_call("luaL_loadbuffer", vec![]);
    }
    // Stub: push a dummy function.
    lua_pushcfunction(L, None);
    0
}

#[no_mangle]
pub unsafe extern "C" fn luaL_loadstring(L: *mut lua_State, s: *const c_char) -> c_int {
    if let Some(state) = state_from_ptr(L) {
        let name = if s.is_null() {
            "<null>".to_string()
        } else {
            CStr::from_ptr(s).to_string_lossy().to_string()
        };
        state.record_call("luaL_loadstring", vec![name]);
    }
    lua_pushcfunction(L, None);
    0
}

#[no_mangle]
pub unsafe extern "C" fn luaL_checkudata(
    L: *mut lua_State,
    idx: c_int,
    _tname: *const c_char,
) -> *mut c_void {
    lua_touserdata(L, idx)
}

#[no_mangle]
pub unsafe extern "C" fn luaL_buffinit(L: *mut lua_State, B: *mut luaL_Buffer) {
    if B.is_null() {
        return;
    }
    (*B).L = L;
    (*B).p = (*B).buffer.as_mut_ptr();
    (*B).lvl = 0;
    if let Some(state) = state_from_ptr(L) {
        state.record_call("luaL_buffinit", vec![]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn luaL_prepbuffer(B: *mut luaL_Buffer) -> *mut c_char {
    if B.is_null() {
        return core::ptr::null_mut();
    }
    let buf = &mut *B;
    let start = buf.buffer.as_mut_ptr();
    let current_len = buffer_len(buf);
    if current_len > 0 {
        // Lua 5.1 semantics: flush pending data onto the Lua stack and reset `p`.
        let l = buf.L;
        if !l.is_null() {
            lua_pushlstring(l, buf.buffer.as_ptr(), current_len);
            buf.lvl += 1;
        }
    }
    buf.p = start;
    if let Some(state) = state_from_ptr(buf.L) {
        state.record_call("luaL_prepbuffer", vec![]);
    }
    buf.buffer.as_mut_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn luaL_addlstring(B: *mut luaL_Buffer, s: *const c_char, len: usize) {
    if B.is_null() {
        return;
    }
    let buf = &mut *B;
    let start = buf.buffer.as_mut_ptr();
    let mut remaining = len;
    let mut src = s as *const u8;

    while remaining > 0 {
        let current_len = buffer_len(buf);
        if current_len >= LUAL_BUFFERSIZE {
            // Flush and continue.
            let L = buf.L;
            if !L.is_null() {
                lua_pushlstring(L, buf.buffer.as_ptr(), current_len);
                buf.lvl += 1;
            }
            buf.p = start;
            continue;
        }

        let available = LUAL_BUFFERSIZE - current_len;
        let copy_len = remaining.min(available);
        if copy_len > 0 && !src.is_null() {
            ptr::copy_nonoverlapping(src, start.add(current_len) as *mut u8, copy_len);
            buf.p = start.add(current_len + copy_len);
            src = src.add(copy_len);
            remaining -= copy_len;
        } else {
            break;
        }
    }

    if let Some(state) = state_from_ptr(buf.L) {
        state.record_call("luaL_addlstring", vec![len.to_string()]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn luaL_pushresult(B: *mut luaL_Buffer) {
    if B.is_null() {
        return;
    }
    let buf = &mut *B;
    let L = buf.L;
    if !L.is_null() {
        let len = buffer_len(buf);
        if len > 0 {
            // Equivalent to Lua 5.1 `emptybuffer(B)`.
            lua_pushlstring(L, buf.buffer.as_ptr(), len);
            buf.p = buf.buffer.as_mut_ptr();
            buf.lvl += 1;
        }
        if buf.lvl > 1 {
            lua_concat(L, buf.lvl);
        } else if buf.lvl == 0 {
            // `lua_concat(L, 0)` pushes the empty string.
            lua_concat(L, 0);
        }
        if let Some(state) = state_from_ptr(L) {
            state.record_call("luaL_pushresult", vec![len.to_string()]);
        }
    }
    buf.lvl = 0;
    buf.p = buf.buffer.as_mut_ptr();
}

#[no_mangle]
pub unsafe extern "C" fn luaL_addstring(B: *mut luaL_Buffer, s: *const c_char) {
    if B.is_null() || s.is_null() {
        return;
    }
    let len = CStr::from_ptr(s).to_bytes().len();
    luaL_addlstring(B, s, len);
    if let Some(buf) = B.as_ref() {
        if let Some(state) = state_from_ptr(buf.L) {
            state.record_call("luaL_addstring", vec![len.to_string()]);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn luaL_addvalue(B: *mut luaL_Buffer) {
    if B.is_null() {
        return;
    }
    let buf = &mut *B;
    let L = buf.L;
    if L.is_null() {
        return;
    }
    let mut len: usize = 0;
    let ptr = lua_tolstring(L, -1, &mut len as *mut usize);
    if !ptr.is_null() && len > 0 {
        luaL_addlstring(B, ptr, len);
    }
    // Pop the value we just consumed.
    lua_settop(L, -2);
    if let Some(state) = state_from_ptr(L) {
        state.record_call("luaL_addvalue", vec![len.to_string()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lua_replace_negative_one_pops() {
        let L = unsafe { luaL_newstate() };
        assert!(!L.is_null());
        unsafe {
            lua_pushinteger(L, 1);
            lua_pushinteger(L, 2);
            lua_pushinteger(L, 3);
            assert_eq!(lua_gettop(L), 3);
            lua_replace(L, -1);
            assert_eq!(lua_gettop(L), 2);
            let state = state_from_ptr(L).unwrap();
            assert!(matches!(
                state.stack.as_slice(),
                [LuaValue::Int(1), LuaValue::Int(2)]
            ));
            lua_close(L);
        }
    }

    #[test]
    fn lua_replace_replaces_and_pops() {
        let L = unsafe { luaL_newstate() };
        assert!(!L.is_null());
        unsafe {
            lua_pushinteger(L, 1);
            lua_pushinteger(L, 2);
            lua_pushinteger(L, 3);
            lua_replace(L, 1);
            assert_eq!(lua_gettop(L), 2);
            let state = state_from_ptr(L).unwrap();
            assert!(matches!(
                state.stack.as_slice(),
                [LuaValue::Int(3), LuaValue::Int(2)]
            ));
            lua_close(L);
        }
    }

    #[test]
    fn lualib_buffer_concatenates_pieces() {
        let L = unsafe { luaL_newstate() };
        assert!(!L.is_null());
        unsafe {
            let mut b: luaL_Buffer = core::mem::zeroed();
            luaL_buffinit(L, &mut b as *mut luaL_Buffer);
            let prefix = b"HTTP/\0";
            luaL_addlstring(
                &mut b as *mut luaL_Buffer,
                prefix.as_ptr() as *const c_char,
                5,
            );

            let p = luaL_prepbuffer(&mut b as *mut luaL_Buffer) as *mut u8;
            assert!(!p.is_null());
            let rest = b"1.1 200 OK\r";
            core::ptr::copy_nonoverlapping(rest.as_ptr(), p, rest.len());
            // Simulate `luaL_addsize`.
            b.p = (p as *mut c_char).add(rest.len());

            luaL_pushresult(&mut b as *mut luaL_Buffer);
            let state = state_from_ptr(L).unwrap();
            assert!(matches!(
                state.stack.last(),
                Some(LuaValue::String(s)) if s == "HTTP/1.1 200 OK\r"
            ));
            lua_close(L);
        }
    }

    #[test]
    fn lua_isnumber_only_accepts_numeric_strings() {
        let L = unsafe { luaL_newstate() };
        assert!(!L.is_null());
        unsafe {
            lua_pushstring(L, b"*l\0".as_ptr() as *const c_char);
            assert_eq!(lua_isnumber(L, -1), 0);
            lua_settop(L, 0);
            lua_pushstring(L, b"123\0".as_ptr() as *const c_char);
            assert_eq!(lua_isnumber(L, -1), 1);
            lua_close(L);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn luaL_openlibs(_L: *mut lua_State) {
    // No-op stub for compatibility; libraries will be installed manually as needed.
}

#[no_mangle]
pub unsafe extern "C" fn lua_newuserdata(L: *mut lua_State, sz: usize) -> *mut c_void {
    if let Some(state) = state_from_ptr(L) {
        let id = state.next_userdata_id();
        let word_size = core::mem::size_of::<usize>();
        let words = (sz + word_size - 1) / word_size;
        let mut blob: Box<[usize]> = vec![0usize; words].into_boxed_slice();
        let ptr = blob.as_mut_ptr() as *mut c_void;
        state.userdata_storage.insert(id, (blob, sz));
        state.push(LuaValue::Userdata(LuaUserdata {
            id,
            data: ptr,
            state: L,
        }));
        state.record_call("lua_newuserdata", vec![sz.to_string()]);
        return ptr;
    }
    core::ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn lua_touserdata(L: *mut lua_State, idx: c_int) -> *mut c_void {
    if let Some(state) = state_from_ptr(L) {
        state.record_call("lua_touserdata", vec![idx.to_string()]);
        if let Some(LuaValue::Userdata(data)) = value_at(state, idx) {
            return data.data;
        }
    }
    core::ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn lua_tocfunction(L: *mut lua_State, idx: c_int) -> lua_CFunction {
    if let Some(state) = state_from_ptr(L) {
        state.record_call("lua_tocfunction", vec![idx.to_string()]);
        if let Some(LuaValue::Function(func)) = value_at(state, idx) {
            return func.cfunc;
        }
    }
    None
}

#[no_mangle]
pub unsafe extern "C" fn lua_topointer(L: *mut lua_State, idx: c_int) -> *const c_void {
    if let Some(state) = state_from_ptr(L) {
        state.record_call("lua_topointer", vec![idx.to_string()]);
        if let Some(value) = value_at(state, idx) {
            let ptr_val = match value {
                LuaValue::Table(handle) => Rc::as_ptr(&handle) as *const c_void,
                LuaValue::Function(f) => &f as *const _ as *const c_void,
                LuaValue::Userdata(u) => u.data,
                LuaValue::Thread(t) => &t as *const _ as *const c_void,
                LuaValue::LightUserdata(p) => p as *const c_void,
                _ => core::ptr::null(),
            };
            return ptr_val;
        }
    }
    core::ptr::null()
}

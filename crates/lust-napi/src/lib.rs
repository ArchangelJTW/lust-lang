mod napi_raw;

use core::ffi::{c_char, c_void};
use std::collections::HashMap;
use std::ffi::CString;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::thread::JoinHandle;

use lust::{EmbeddedProgram, Value};
use napi_raw::*;

/// Thread-safe Rust actor pool owning `!Send`/`!Sync` `EmbeddedProgram`/`VM` instances
/// strictly inside dedicated OS worker threads.
struct PoolRequest {
    source: String,
    fn_name: String,
    arg: i64,
    reply: mpsc::SyncSender<Result<i64, String>>,
}

pub struct VmActorPool {
    workers: Vec<mpsc::Sender<PoolRequest>>,
    handles: Mutex<Vec<JoinHandle<()>>>,
    next_worker: AtomicUsize,
    shut_down: AtomicBool,
}

impl VmActorPool {
    pub fn new(worker_count: usize) -> Result<Self, String> {
        let count = worker_count.max(1);
        let mut workers = Vec::with_capacity(count);
        let mut handles = Vec::with_capacity(count);
        for idx in 0..count {
            let (tx, rx) = mpsc::channel::<PoolRequest>();
            let handle = std::thread::Builder::new()
                .name(format!("lust-napi-vm-{idx}"))
                .spawn(move || {
                    while let Ok(req) = rx.recv() {
                        let res = execute_on_current_thread(&req.source, &req.fn_name, req.arg);
                        let _ = req.reply.send(res);
                    }
                })
                .map_err(|e| format!("failed spawning worker thread: {e}"))?;
            workers.push(tx);
            handles.push(handle);
        }
        Ok(Self {
            workers,
            handles: Mutex::new(handles),
            next_worker: AtomicUsize::new(0),
            shut_down: AtomicBool::new(false),
        })
    }

    pub fn eval_on_worker(&self, source: String, fn_name: String, arg: i64) -> Result<i64, String> {
        if self.shut_down.load(Ordering::SeqCst) {
            return Err("Cancelled: VM actor pool already shut down".to_string());
        }
        let idx = self.next_worker.fetch_add(1, Ordering::Relaxed) % self.workers.len();
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        self.workers[idx]
            .send(PoolRequest {
                source,
                fn_name,
                arg,
                reply: reply_tx,
            })
            .map_err(|_| "Cancelled: VM worker channel closed".to_string())?;
        reply_rx
            .recv()
            .map_err(|_| "Cancelled: VM worker exited before replying".to_string())?
    }

    pub fn shutdown(&self) {
        if self.shut_down.swap(true, Ordering::SeqCst) {
            return;
        }
    }
}

impl Drop for VmActorPool {
    fn drop(&mut self) {
        self.workers.clear();
        if let Ok(mut guard) = self.handles.lock() {
            for h in guard.drain(..) {
                let _ = h.join();
            }
        }
    }
}

fn execute_on_current_thread(source: &str, fn_name: &str, arg: i64) -> Result<i64, String> {
    let mut program = EmbeddedProgram::builder()
        .module("main", source)
        .entry_module("main")
        .compile()
        .map_err(|e| format!("compile error: {e}"))?;
    let target_fn = if fn_name.contains('.') {
        fn_name.to_string()
    } else {
        format!("main.{fn_name}")
    };
    let val = program
        .vm_mut()
        .call(&target_fn, vec![Value::Int(arg)])
        .map_err(|e| format!("runtime error: {e}"))?;
    match val {
        Value::Int(i) => Ok(i),
        other => Err(format!("expected int return value, got {other:?}")),
    }
}

static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);
static POOLS: OnceLock<Mutex<HashMap<u64, Arc<VmActorPool>>>> = OnceLock::new();

fn pools() -> &'static Mutex<HashMap<u64, Arc<VmActorPool>>> {
    POOLS.get_or_init(|| Mutex::new(HashMap::new()))
}

struct EnvState {
    alive: Arc<AtomicBool>,
    owned_pools: Mutex<Vec<u64>>,
}

unsafe extern "C" fn on_env_cleanup(arg: *mut c_void) {
    if arg.is_null() {
        return;
    }
    let state = unsafe { Box::from_raw(arg as *mut EnvState) };
    state.alive.store(false, Ordering::SeqCst);
    if let Ok(mut ids) = state.owned_pools.lock() {
        if let Ok(mut map) = pools().lock() {
            for id in ids.drain(..) {
                if let Some(pool) = map.remove(&id) {
                    pool.shutdown();
                }
            }
        }
    }
}

struct AsyncJob {
    env_alive: Arc<AtomicBool>,
    pool: Arc<VmActorPool>,
    source: String,
    fn_name: String,
    arg: i64,
    deferred: napi_deferred,
    work: napi_async_work,
    result: Option<Result<i64, String>>,
}

unsafe extern "C" fn async_execute(_env: napi_env, data: *mut c_void) {
    let job = unsafe { &mut *(data as *mut AsyncJob) };
    if !job.env_alive.load(Ordering::SeqCst) {
        job.result = Some(Err("Cancelled: JS environment torn down before execution".to_string()));
        return;
    }
    job.result = Some(
        job.pool
            .eval_on_worker(job.source.clone(), job.fn_name.clone(), job.arg),
    );
}

unsafe extern "C" fn async_complete(env: napi_env, status: napi_status, data: *mut c_void) {
    let mut job = unsafe { Box::from_raw(data as *mut AsyncJob) };
    let alive = job.env_alive.load(Ordering::SeqCst);
    if !alive {
        // Environment already tore down cleanly; never touch napi_env again.
        return;
    }
    if status == NAPI_OK {
        match job.result.take().unwrap_or_else(|| Err("Cancelled: missing result".to_string())) {
            Ok(val) => unsafe {
                let mut js_val: napi_value = ptr::null_mut();
                if napi_create_int64(env, val, &mut js_val) == NAPI_OK {
                    let _ = napi_resolve_deferred(env, job.deferred, js_val);
                }
            },
            Err(err_msg) => unsafe {
                let mut err_obj: napi_value = ptr::null_mut();
                let c_msg = CString::new(err_msg).unwrap_or_else(|_| CString::new("error").unwrap());
                let mut msg_val: napi_value = ptr::null_mut();
                let _ = napi_create_string_utf8(env, c_msg.as_ptr(), NAPI_AUTO_LENGTH, &mut msg_val);
                if napi_create_error(env, ptr::null_mut(), msg_val, &mut err_obj) == NAPI_OK {
                    let _ = napi_reject_deferred(env, job.deferred, err_obj);
                }
            },
        }
    }
    unsafe {
        let _ = napi_delete_async_work(env, job.work);
    }
}

unsafe fn get_str_arg(env: napi_env, val: napi_value) -> Result<String, &'static str> {
    let mut len: usize = 0;
    if unsafe { napi_get_value_string_utf8(env, val, ptr::null_mut(), 0, &mut len) } != NAPI_OK {
        return Err("expected UTF-8 string argument");
    }
    let mut buf = vec![0u8; len + 1];
    let mut copied: usize = 0;
    if unsafe {
        napi_get_value_string_utf8(
            env,
            val,
            buf.as_mut_ptr() as *mut c_char,
            buf.len(),
            &mut copied,
        )
    } != NAPI_OK
    {
        return Err("failed reading UTF-8 string argument");
    }
    buf.truncate(copied);
    String::from_utf8(buf).map_err(|_| "invalid UTF-8 argument")
}

unsafe fn throw_js_err(env: napi_env, msg: &str) -> napi_value {
    let c = CString::new(msg).unwrap_or_else(|_| CString::new("lust-napi error").unwrap());
    unsafe {
        let _ = napi_throw_error(env, ptr::null(), c.as_ptr());
        let mut undef: napi_value = ptr::null_mut();
        let _ = napi_get_undefined(env, &mut undef);
        undef
    }
}

unsafe extern "C" fn js_create_pool(env: napi_env, info: napi_callback_info) -> napi_value {
    let mut argc: usize = 1;
    let mut argv = [ptr::null_mut(); 1];
    let mut data: *mut c_void = ptr::null_mut();
    unsafe {
        let _ = napi_get_cb_info(env, info, &mut argc, argv.as_mut_ptr(), ptr::null_mut(), &mut data);
    }
    let mut size: i64 = 2;
    if argc >= 1 {
        let _ = unsafe { napi_get_value_int64(env, argv[0], &mut size) };
    }
    let pool = match VmActorPool::new((size.max(1)) as usize) {
        Ok(p) => Arc::new(p),
        Err(e) => return unsafe { throw_js_err(env, &e) },
    };
    let id = NEXT_POOL_ID.fetch_add(1, Ordering::SeqCst);
    pools().lock().unwrap().insert(id, pool);
    if !data.is_null() {
        let state = unsafe { &*(data as *const EnvState) };
        if let Ok(mut owned) = state.owned_pools.lock() {
            owned.push(id);
        }
    }
    let mut out: napi_value = ptr::null_mut();
    unsafe {
        let _ = napi_create_int64(env, id as i64, &mut out);
    }
    out
}

unsafe extern "C" fn js_shutdown_pool(env: napi_env, info: napi_callback_info) -> napi_value {
    let mut argc: usize = 1;
    let mut argv = [ptr::null_mut(); 1];
    unsafe {
        let _ = napi_get_cb_info(env, info, &mut argc, argv.as_mut_ptr(), ptr::null_mut(), ptr::null_mut());
    }
    let mut id: i64 = 0;
    if argc < 1 || unsafe { napi_get_value_int64(env, argv[0], &mut id) } != NAPI_OK {
        return unsafe { throw_js_err(env, "shutdownPool(poolId) requires integer poolId") };
    }
    let removed = pools().lock().unwrap().remove(&(id as u64));
    let ok = if let Some(pool) = removed {
        pool.shutdown();
        true
    } else {
        false
    };
    let mut out: napi_value = ptr::null_mut();
    unsafe {
        let _ = napi_get_boolean(env, ok, &mut out);
    }
    out
}

unsafe extern "C" fn js_eval_sync(env: napi_env, info: napi_callback_info) -> napi_value {
    let mut argc: usize = 3;
    let mut argv = [ptr::null_mut(); 3];
    unsafe {
        let _ = napi_get_cb_info(env, info, &mut argc, argv.as_mut_ptr(), ptr::null_mut(), ptr::null_mut());
    }
    if argc < 3 {
        return unsafe { throw_js_err(env, "evalSync(source, fnName, argInt) requires 3 args") };
    }
    let source = match unsafe { get_str_arg(env, argv[0]) } {
        Ok(s) => s,
        Err(e) => return unsafe { throw_js_err(env, e) },
    };
    let fn_name = match unsafe { get_str_arg(env, argv[1]) } {
        Ok(s) => s,
        Err(e) => return unsafe { throw_js_err(env, e) },
    };
    let mut arg_int: i64 = 0;
    if unsafe { napi_get_value_int64(env, argv[2], &mut arg_int) } != NAPI_OK {
        return unsafe { throw_js_err(env, "argInt must be int64") };
    }
    match execute_on_current_thread(&source, &fn_name, arg_int) {
        Ok(res) => {
            let mut out: napi_value = ptr::null_mut();
            unsafe {
                let _ = napi_create_int64(env, res, &mut out);
            }
            out
        }
        Err(e) => unsafe { throw_js_err(env, &e) },
    }
}

unsafe extern "C" fn js_eval_async_pool(env: napi_env, info: napi_callback_info) -> napi_value {
    let mut argc: usize = 4;
    let mut argv = [ptr::null_mut(); 4];
    let mut data: *mut c_void = ptr::null_mut();
    unsafe {
        let _ = napi_get_cb_info(env, info, &mut argc, argv.as_mut_ptr(), ptr::null_mut(), &mut data);
    }
    if argc < 4 || data.is_null() {
        return unsafe { throw_js_err(env, "evalAsync(poolId, source, fnName, argInt) requires 4 args") };
    }
    let mut pool_id: i64 = 0;
    if unsafe { napi_get_value_int64(env, argv[0], &mut pool_id) } != NAPI_OK {
        return unsafe { throw_js_err(env, "poolId must be int64") };
    }
    let source = match unsafe { get_str_arg(env, argv[1]) } {
        Ok(s) => s,
        Err(e) => return unsafe { throw_js_err(env, e) },
    };
    let fn_name = match unsafe { get_str_arg(env, argv[2]) } {
        Ok(s) => s,
        Err(e) => return unsafe { throw_js_err(env, e) },
    };
    let mut arg_int: i64 = 0;
    if unsafe { napi_get_value_int64(env, argv[3], &mut arg_int) } != NAPI_OK {
        return unsafe { throw_js_err(env, "argInt must be int64") };
    }

    let pool_opt = pools().lock().unwrap().get(&(pool_id as u64)).cloned();
    let pool = match pool_opt {
        Some(p) => p,
        None => {
            // Return cleanly rejected Promise for closed/unknown pool
            let mut deferred: napi_deferred = ptr::null_mut();
            let mut promise: napi_value = ptr::null_mut();
            unsafe {
                let _ = napi_create_promise(env, &mut deferred, &mut promise);
                let msg = CString::new("Cancelled: unknown or closed VM pool").unwrap();
                let mut msg_val: napi_value = ptr::null_mut();
                let mut err_obj: napi_value = ptr::null_mut();
                let _ = napi_create_string_utf8(env, msg.as_ptr(), NAPI_AUTO_LENGTH, &mut msg_val);
                let _ = napi_create_error(env, ptr::null_mut(), msg_val, &mut err_obj);
                let _ = napi_reject_deferred(env, deferred, err_obj);
            }
            return promise;
        }
    };

    let state = unsafe { &*(data as *const EnvState) };
    let mut deferred: napi_deferred = ptr::null_mut();
    let mut promise: napi_value = ptr::null_mut();
    if unsafe { napi_create_promise(env, &mut deferred, &mut promise) } != NAPI_OK {
        return unsafe { throw_js_err(env, "failed creating Promise") };
    }

    let job = Box::new(AsyncJob {
        env_alive: Arc::clone(&state.alive),
        pool,
        source,
        fn_name,
        arg: arg_int,
        deferred,
        work: ptr::null_mut(),
        result: None,
    });
    let job_ptr = Box::into_raw(job);
    let work_name = CString::new("lust_napi_eval").unwrap();
    let mut name_val: napi_value = ptr::null_mut();
    let mut work: napi_async_work = ptr::null_mut();
    unsafe {
        let _ = napi_create_string_utf8(env, work_name.as_ptr(), NAPI_AUTO_LENGTH, &mut name_val);
        if napi_create_async_work(
            env,
            ptr::null_mut(),
            name_val,
            Some(async_execute),
            Some(async_complete),
            job_ptr as *mut c_void,
            &mut work,
        ) != NAPI_OK
        {
            drop(Box::from_raw(job_ptr));
            return throw_js_err(env, "napi_create_async_work failed");
        }
        (*job_ptr).work = work;
        let _ = napi_queue_async_work(env, work);
    }
    promise
}

#[unsafe(no_mangle)]
pub extern "C" fn node_api_module_get_api_version_v1() -> i32 {
    8
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn napi_register_module_v1(
    env: napi_env,
    exports: napi_value,
) -> napi_value {
    let state = Box::new(EnvState {
        alive: Arc::new(AtomicBool::new(true)),
        owned_pools: Mutex::new(Vec::new()),
    });
    let state_ptr = Box::into_raw(state) as *mut c_void;
    unsafe {
        let _ = napi_add_env_cleanup_hook(env, Some(on_env_cleanup), state_ptr);
    }

    unsafe {
        register_fn(env, exports, "createPool", Some(js_create_pool), state_ptr);
        register_fn(env, exports, "shutdownPool", Some(js_shutdown_pool), ptr::null_mut());
        register_fn(env, exports, "evalSync", Some(js_eval_sync), ptr::null_mut());
        register_fn(env, exports, "evalAsync", Some(js_eval_async_pool), state_ptr);

        let mut meta_obj: napi_value = ptr::null_mut();
        if napi_create_object(env, &mut meta_obj) == NAPI_OK {
            set_str_prop(env, meta_obj, "abi", "c-node-api-v8");
            set_str_prop(env, meta_obj, "napiRsStatus", "BLOCKED: absent offline");
            set_str_prop(env, meta_obj, "vmSafetyModel", "dedicated-worker-actor-pool");
            let prop_name = CString::new("capabilities").unwrap();
            let _ = napi_set_named_property(env, exports, prop_name.as_ptr(), meta_obj);
        }
    }

    exports
}

unsafe fn register_fn(
    env: napi_env,
    exports: napi_value,
    name: &str,
    cb: napi_callback,
    data: *mut c_void,
) {
    let c_name = CString::new(name).unwrap();
    let mut fn_val: napi_value = ptr::null_mut();
    unsafe {
        if napi_create_function(env, c_name.as_ptr(), NAPI_AUTO_LENGTH, cb, data, &mut fn_val)
            == NAPI_OK
        {
            let _ = napi_set_named_property(env, exports, c_name.as_ptr(), fn_val);
        }
    }
}

unsafe fn set_str_prop(env: napi_env, obj: napi_value, key: &str, val: &str) {
    let ck = CString::new(key).unwrap();
    let cv = CString::new(val).unwrap();
    let mut v: napi_value = ptr::null_mut();
    unsafe {
        if napi_create_string_utf8(env, cv.as_ptr(), NAPI_AUTO_LENGTH, &mut v) == NAPI_OK {
            let _ = napi_set_named_property(env, obj, ck.as_ptr(), v);
        }
    }
}

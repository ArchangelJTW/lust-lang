//! Coordination Builtins: script bridge exposing `Stream`, `Sink`, `Queue`,
//! `Deferred`, `Semaphore`, `Ref`, and `Schedule` to Lust programs.
//!
//! Complies with `RFC-STDLIB-ASYNC-COORDINATION` (§3.4):
//! - Modules: `@std/coord`, `@std/sync`, `@std/stream`, `@std/schedule`.
//! - Native constructors and methods for all 7 primitives.
//! - Thread-local registries for handle-based coordination across fibers.

use alloc::{
    format,
    rc::Rc,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bytecode::{
    native_fn, LustMap, NativeCallResult, NativeFn, Value, ValueKey,
};
use crate::effect_abi::Exit;
use crate::rx::buffer::OverflowStrategy;
use super::{
    deferred::Deferred,
    queue::Queue,
    ref_cell::Ref,
    schedule::Schedule,
    semaphore::Semaphore,
};
use crate::vm::VM;

static NEXT_HANDLE_ID: AtomicU64 = AtomicU64::new(100_000);

thread_local! {
    static THREAD_QUEUES: RefCell<HashMap<u64, Queue<Value>>> = RefCell::new(HashMap::new());
    static THREAD_SEMAPHORES: RefCell<HashMap<u64, Semaphore>> = RefCell::new(HashMap::new());
    static THREAD_DEFERREDS: RefCell<HashMap<u64, Deferred<Value, String>>> = RefCell::new(HashMap::new());
}

pub fn reset_global_coord_registries() {
    THREAD_QUEUES.with(|reg| reg.borrow_mut().clear());
    THREAD_SEMAPHORES.with(|reg| reg.borrow_mut().clear());
    THREAD_DEFERREDS.with(|reg| reg.borrow_mut().clear());
}

fn string_key(s: &str) -> ValueKey {
    ValueKey::string(s.to_string())
}

fn invoke_val(func: &Value, args: &[Value]) -> Result<Value, String> {
    if let Value::NativeFunction(nf) = func {
        match nf(args) {
            Ok(NativeCallResult::Return(v)) => Ok(v),
            Ok(_) => Ok(Value::Nil),
            Err(e) => Err(e),
        }
    } else {
        VM::with_current(|vm| vm.call_value(func, args.to_vec()).map_err(|e| e.to_string()))
    }
}

// ============================================================================
// 1. Ref Native Functions
// ============================================================================

pub fn builtin_ref_make(args: &[Value]) -> Result<NativeCallResult, String> {
    let initial = args.first().cloned().unwrap_or(Value::Nil);
    let r = Ref::new(initial);
    let mut map = LustMap::default();
    map.insert(string_key("__type"), Value::string("Ref"));
    map.insert(string_key("value"), r.get());

    let r_get = r.clone();
    map.insert(
        string_key("get"),
        Value::NativeFunction(native_fn(move |_| {
            Ok(NativeCallResult::Return(r_get.get()))
        })),
    );

    let r_set = r.clone();
    map.insert(
        string_key("set"),
        Value::NativeFunction(native_fn(move |args| {
            let val = args.first().cloned().unwrap_or(Value::Nil);
            r_set.set(val);
            Ok(NativeCallResult::Return(Value::Nil))
        })),
    );

    let r_update = r.clone();
    map.insert(
        string_key("update"),
        Value::NativeFunction(native_fn(move |args| {
            if let Some(func) = args.first() {
                let old = r_update.get();
                let new_val = invoke_val(func, &[old])?;
                r_update.set(new_val.clone());
                Ok(NativeCallResult::Return(new_val))
            } else {
                Err("Ref.update expects a function argument".into())
            }
        })),
    );

    let r_modify = r.clone();
    map.insert(
        string_key("modify"),
        Value::NativeFunction(native_fn(move |args| {
            if let Some(func) = args.first() {
                let old = r_modify.get();
                match invoke_val(func, &[old])? {
                    Value::Array(arr) => {
                        let borrowed = arr.borrow();
                        let res = borrowed.first().cloned().unwrap_or(Value::Nil);
                        let new_val = borrowed.get(1).cloned().unwrap_or(Value::Nil);
                        r_modify.set(new_val);
                        Ok(NativeCallResult::Return(res))
                    }
                    other => {
                        r_modify.set(other.clone());
                        Ok(NativeCallResult::Return(other))
                    }
                }
            } else {
                Err("Ref.modify expects a function argument".into())
            }
        })),
    );

    Ok(NativeCallResult::Return(Value::map(map)))
}

// ============================================================================
// 2. Deferred Native Functions
// ============================================================================

pub fn builtin_deferred_make(_args: &[Value]) -> Result<NativeCallResult, String> {
    let id = NEXT_HANDLE_ID.fetch_add(1, Ordering::SeqCst);
    let d = Deferred::<Value, String>::new();
    THREAD_DEFERREDS.with(|reg| reg.borrow_mut().insert(id, d));

    let mut map = LustMap::default();
    map.insert(string_key("__type"), Value::string("Deferred"));
    map.insert(string_key("id"), Value::Int(id as i64));

    map.insert(
        string_key("complete"),
        Value::NativeFunction(native_fn(move |args| {
            let val = args.first().cloned().unwrap_or(Value::Nil);
            THREAD_DEFERREDS.with(|reg| {
                if let Some(def) = reg.borrow_mut().get_mut(&id) {
                    let ok = def.complete(val);
                    Ok(NativeCallResult::Return(Value::Bool(ok)))
                } else {
                    Err("Invalid Deferred handle".into())
                }
            })
        })),
    );

    map.insert(
        string_key("fail"),
        Value::NativeFunction(native_fn(move |args| {
            let err_msg = args
                .first()
                .and_then(|v| v.as_string())
                .unwrap_or_else(|| "Deferred failure".into());
            THREAD_DEFERREDS.with(|reg| {
                if let Some(def) = reg.borrow_mut().get_mut(&id) {
                    let ok = def.fail(err_msg.to_string());
                    Ok(NativeCallResult::Return(Value::Bool(ok)))
                } else {
                    Err("Invalid Deferred handle".into())
                }
            })
        })),
    );

    map.insert(
        string_key("is_completed"),
        Value::NativeFunction(native_fn(move |_| {
            THREAD_DEFERREDS.with(|reg| {
                if let Some(def) = reg.borrow().get(&id) {
                    Ok(NativeCallResult::Return(Value::Bool(def.is_completed())))
                } else {
                    Err("Invalid Deferred handle".into())
                }
            })
        })),
    );

    map.insert(
        string_key("poll"),
        Value::NativeFunction(native_fn(move |_| {
            THREAD_DEFERREDS.with(|reg| {
                if let Some(def) = reg.borrow().get(&id) {
                    match def.poll() {
                        Some(Exit::Success(v)) => Ok(NativeCallResult::Return(v.clone())),
                        Some(Exit::Failure(c)) => {
                            Err(format!("Deferred failed: {:?}", c))
                        }
                        None => Ok(NativeCallResult::Return(Value::Nil)),
                    }
                } else {
                    Err("Invalid Deferred handle".into())
                }
            })
        })),
    );

    map.insert(
        string_key("await"),
        Value::NativeFunction(native_fn(move |_| {
            THREAD_DEFERREDS.with(|reg| {
                if let Some(def) = reg.borrow().get(&id) {
                    match def.poll() {
                        Some(Exit::Success(v)) => Ok(NativeCallResult::Return(v.clone())),
                        Some(Exit::Failure(c)) => {
                            Err(format!("Deferred failed: {:?}", c))
                        }
                        None => Ok(NativeCallResult::Return(Value::Nil)),
                    }
                } else {
                    Err("Invalid Deferred handle".into())
                }
            })
        })),
    );

    Ok(NativeCallResult::Return(Value::map(map)))
}

// ============================================================================
// 3. Semaphore Native Functions
// ============================================================================

pub fn builtin_semaphore_make(args: &[Value]) -> Result<NativeCallResult, String> {
    let permits = match args.first() {
        Some(Value::Int(n)) if *n >= 0 => *n as usize,
        Some(Value::Int(n)) => {
            return Err(format!("Semaphore permits cannot be negative: {}", n));
        }
        Some(other) => {
            return Err(format!("Semaphore.make expects integer permits, got {:?}", other));
        }
        None => 1,
    };

    let id = NEXT_HANDLE_ID.fetch_add(1, Ordering::SeqCst);
    let sem = Semaphore::make(permits);
    THREAD_SEMAPHORES.with(|reg| reg.borrow_mut().insert(id, sem));

    let mut map = LustMap::default();
    map.insert(string_key("__type"), Value::string("Semaphore"));
    map.insert(string_key("id"), Value::Int(id as i64));

    map.insert(
        string_key("available"),
        Value::NativeFunction(native_fn(move |_| {
            THREAD_SEMAPHORES.with(|reg| {
                if let Some(s) = reg.borrow().get(&id) {
                    Ok(NativeCallResult::Return(Value::Int(s.available_permits() as i64)))
                } else {
                    Err("Invalid Semaphore handle".into())
                }
            })
        })),
    );

    map.insert(
        string_key("try_acquire"),
        Value::NativeFunction(native_fn(move |args| {
            let n = match args.first() {
                Some(Value::Int(n)) if *n >= 0 => *n as usize,
                Some(Value::Nil) | None => 1,
                _ => return Err("try_acquire expects non-negative integer permits".into()),
            };
            THREAD_SEMAPHORES.with(|reg| {
                if let Some(s) = reg.borrow_mut().get_mut(&id) {
                    let ok = s.try_acquire(n);
                    Ok(NativeCallResult::Return(Value::Bool(ok)))
                } else {
                    Err("Invalid Semaphore handle".into())
                }
            })
        })),
    );

    map.insert(
        string_key("release"),
        Value::NativeFunction(native_fn(move |args| {
            let n = match args.first() {
                Some(Value::Int(n)) if *n >= 0 => *n as usize,
                Some(Value::Nil) | None => 1,
                _ => return Err("release expects non-negative integer permits".into()),
            };
            THREAD_SEMAPHORES.with(|reg| {
                if let Some(s) = reg.borrow_mut().get_mut(&id) {
                    s.add_permits(n);
                    Ok(NativeCallResult::Return(Value::Int(s.available_permits() as i64)))
                } else {
                    Err("Invalid Semaphore handle".into())
                }
            })
        })),
    );

    map.insert(
        string_key("with_permits"),
        Value::NativeFunction(native_fn(move |args| {
            let n = match args.first() {
                Some(Value::Int(n)) if *n >= 0 => *n as usize,
                _ => 1,
            };
            let func = match args.get(1) {
                Some(f) => f.clone(),
                _ => return Err("Semaphore.with_permits expects (n, function)".into()),
            };

            // Acquire
            let acquired = THREAD_SEMAPHORES.with(|reg| {
                if let Some(s) = reg.borrow_mut().get_mut(&id) {
                    s.try_acquire(n)
                } else {
                    false
                }
            });

            if !acquired {
                return Err("Semaphore permits unavailable for with_permits".into());
            }

            // Execute function under permit guard
            let res = invoke_val(&func, &[]);

            // Guarantee release
            THREAD_SEMAPHORES.with(|reg| {
                if let Some(s) = reg.borrow_mut().get_mut(&id) {
                    s.add_permits(n);
                }
            });

            res.map(NativeCallResult::Return)
        })),
    );

    Ok(NativeCallResult::Return(Value::map(map)))
}

// ============================================================================
// 4. Queue Native Functions
// ============================================================================

pub fn builtin_queue_make(args: &[Value]) -> Result<NativeCallResult, String> {
    let capacity = match args.first() {
        Some(Value::Int(n)) if *n >= 0 => *n as usize,
        Some(Value::Int(n)) => {
            return Err(format!("Queue capacity cannot be negative: {}", n));
        }
        Some(Value::Nil) | None => 64,
        Some(other) => {
            return Err(format!("Queue.make expects integer capacity, got {:?}", other));
        }
    };

    let strategy = match args.get(1) {
        Some(Value::String(s)) => match s.as_str() {
            "drop_oldest" | "sliding" => OverflowStrategy::DropOldest,
            "drop_newest" | "dropping" => OverflowStrategy::DropNewest,
            "fail" => OverflowStrategy::Fail,
            _ => OverflowStrategy::Suspend,
        },
        _ => OverflowStrategy::Suspend,
    };

    let id = NEXT_HANDLE_ID.fetch_add(1, Ordering::SeqCst);
    let queue = Queue::<Value>::with_overflow(capacity, strategy);
    THREAD_QUEUES.with(|reg| reg.borrow_mut().insert(id, queue));

    let mut map = LustMap::default();
    map.insert(string_key("__type"), Value::string("Queue"));
    map.insert(string_key("id"), Value::Int(id as i64));

    map.insert(
        string_key("offer"),
        Value::NativeFunction(native_fn(move |args| {
            let item = args.first().cloned().unwrap_or(Value::Nil);
            THREAD_QUEUES.with(|reg| {
                if let Some(q) = reg.borrow_mut().get_mut(&id) {
                    match q.try_offer(item) {
                        Ok(b) => Ok(NativeCallResult::Return(Value::Bool(b))),
                        Err(e) => Err(format!("Queue offer failed: {:?}", e)),
                    }
                } else {
                    Err("Invalid Queue handle".into())
                }
            })
        })),
    );

    map.insert(
        string_key("take"),
        Value::NativeFunction(native_fn(move |_| {
            THREAD_QUEUES.with(|reg| {
                if let Some(q) = reg.borrow_mut().get_mut(&id) {
                    match q.try_take() {
                        Ok(Some(v)) => Ok(NativeCallResult::Return(v)),
                        Ok(None) => Ok(NativeCallResult::Return(Value::Nil)),
                        Err(e) => Err(format!("Queue take failed: {:?}", e)),
                    }
                } else {
                    Err("Invalid Queue handle".into())
                }
            })
        })),
    );

    map.insert(
        string_key("len"),
        Value::NativeFunction(native_fn(move |_| {
            THREAD_QUEUES.with(|reg| {
                if let Some(q) = reg.borrow().get(&id) {
                    Ok(NativeCallResult::Return(Value::Int(q.len() as i64)))
                } else {
                    Err("Invalid Queue handle".into())
                }
            })
        })),
    );

    map.insert(
        string_key("is_empty"),
        Value::NativeFunction(native_fn(move |_| {
            THREAD_QUEUES.with(|reg| {
                if let Some(q) = reg.borrow().get(&id) {
                    Ok(NativeCallResult::Return(Value::Bool(q.is_empty())))
                } else {
                    Err("Invalid Queue handle".into())
                }
            })
        })),
    );

    map.insert(
        string_key("is_full"),
        Value::NativeFunction(native_fn(move |_| {
            THREAD_QUEUES.with(|reg| {
                if let Some(q) = reg.borrow().get(&id) {
                    Ok(NativeCallResult::Return(Value::Bool(q.is_full())))
                } else {
                    Err("Invalid Queue handle".into())
                }
            })
        })),
    );

    map.insert(
        string_key("close"),
        Value::NativeFunction(native_fn(move |_| {
            THREAD_QUEUES.with(|reg| {
                if let Some(q) = reg.borrow_mut().get_mut(&id) {
                    q.close_tm(&mut VM::new().task_manager);
                    Ok(NativeCallResult::Return(Value::Bool(true)))
                } else {
                    Err("Invalid Queue handle".into())
                }
            })
        })),
    );

    Ok(NativeCallResult::Return(Value::map(map)))
}

// ============================================================================
// 5. Schedule Native Functions
// ============================================================================

pub fn builtin_schedule_recurs(args: &[Value]) -> Result<NativeCallResult, String> {
    let n = match args.first() {
        Some(Value::Int(n)) if *n >= 0 => *n as u64,
        _ => 1,
    };
    let mut map = LustMap::default();
    map.insert(string_key("__type"), Value::string("Schedule"));
    map.insert(string_key("kind"), Value::string("recurs"));
    map.insert(string_key("n"), Value::Int(n as i64));
    map.insert(
        string_key("get_n"),
        Value::NativeFunction(native_fn(move |_| {
            Ok(NativeCallResult::Return(Value::Int(n as i64)))
        })),
    );
    Ok(NativeCallResult::Return(Value::map(map)))
}

pub fn builtin_schedule_spaced(args: &[Value]) -> Result<NativeCallResult, String> {
    let ms = match args.first() {
        Some(Value::Int(n)) if *n >= 0 => *n as u64,
        _ => 10,
    };
    let mut map = LustMap::default();
    map.insert(string_key("__type"), Value::string("Schedule"));
    map.insert(string_key("kind"), Value::string("spaced"));
    map.insert(string_key("ms"), Value::Int(ms as i64));
    map.insert(
        string_key("get_ms"),
        Value::NativeFunction(native_fn(move |_| {
            Ok(NativeCallResult::Return(Value::Int(ms as i64)))
        })),
    );
    Ok(NativeCallResult::Return(Value::map(map)))
}

pub fn builtin_schedule_exponential(args: &[Value]) -> Result<NativeCallResult, String> {
    let base = match args.first() {
        Some(Value::Int(n)) if *n >= 0 => *n as u64,
        _ => 10,
    };
    let factor = match args.get(1) {
        Some(Value::Float(f)) => *f,
        Some(Value::Int(i)) => *i as f64,
        _ => 2.0,
    };
    let max = match args.get(2) {
        Some(Value::Int(n)) if *n >= 0 => *n as u64,
        _ => 1000,
    };
    let mut map = LustMap::default();
    map.insert(string_key("__type"), Value::string("Schedule"));
    map.insert(string_key("kind"), Value::string("exponential"));
    map.insert(string_key("base"), Value::Int(base as i64));
    map.insert(string_key("factor"), Value::Float(factor));
    map.insert(string_key("max"), Value::Int(max as i64));
    map.insert(
        string_key("get_base"),
        Value::NativeFunction(native_fn(move |_| {
            Ok(NativeCallResult::Return(Value::Int(base as i64)))
        })),
    );
    Ok(NativeCallResult::Return(Value::map(map)))
}

pub fn builtin_schedule_retry(args: &[Value]) -> Result<NativeCallResult, String> {
    let retries = match args.first() {
        Some(Value::Map(m)) => {
            let b = m.borrow();
            b.get(&string_key("n"))
                .and_then(|v| match v {
                    Value::Int(i) => Some(*i as u64),
                    _ => None,
                })
                .unwrap_or(3)
        }
        Some(Value::Int(n)) => *n as u64,
        _ => 3,
    };

    let func = match args.get(1) {
        Some(f) => f.clone(),
        _ => return Err("Schedule.retry expects (schedule, function)".into()),
    };

    let sched = Schedule::recurs(retries);
    let mut attempts = 0;
    sched
        .retry(|| {
            attempts += 1;
            invoke_val(&func, &[Value::Int(attempts)])
        })
        .map(NativeCallResult::Return)
}

pub fn builtin_schedule_timeout(args: &[Value]) -> Result<NativeCallResult, String> {
    let ms = match args.first() {
        Some(Value::Int(n)) if *n >= 0 => *n as u64,
        _ => 1000,
    };
    let func = match args.get(1) {
        Some(f) => f.clone(),
        _ => return Err("Schedule.timeout expects (ms, function)".into()),
    };

    Schedule::timeout(ms, || invoke_val(&func, &[]))
        .map(NativeCallResult::Return)
}

// ============================================================================
// 6. Stream Native Functions
// ============================================================================

pub fn builtin_stream_from_array(args: &[Value]) -> Result<NativeCallResult, String> {
    let items = match args.first() {
        Some(Value::Array(arr)) => arr.borrow().clone(),
        Some(other) if !matches!(other, Value::Nil) => vec![other.clone()],
        _ => Vec::new(),
    };

    let mut map = LustMap::default();
    map.insert(string_key("__type"), Value::string("Stream"));
    map.insert(string_key("items"), Value::array(items.clone()));

    let st_items = items.clone();
    map.insert(
        string_key("collect"),
        Value::NativeFunction(native_fn(move |_| {
            Ok(NativeCallResult::Return(Value::array(st_items.clone())))
        })),
    );

    let st_count = items.len();
    map.insert(
        string_key("count"),
        Value::NativeFunction(native_fn(move |_| {
            Ok(NativeCallResult::Return(Value::Int(st_count as i64)))
        })),
    );

    Ok(NativeCallResult::Return(Value::map(map)))
}

pub fn builtin_stream_collect(args: &[Value]) -> Result<NativeCallResult, String> {
    match args.first() {
        Some(Value::Map(m)) => {
            let b = m.borrow();
            if let Some(items) = b.get(&string_key("items")) {
                Ok(NativeCallResult::Return(items.clone()))
            } else {
                Ok(NativeCallResult::Return(Value::array(Vec::new())))
            }
        }
        Some(Value::Array(a)) => Ok(NativeCallResult::Return(Value::Array(a.clone()))),
        _ => Ok(NativeCallResult::Return(Value::array(Vec::new()))),
    }
}

pub fn builtin_stream_range(args: &[Value]) -> Result<NativeCallResult, String> {
    let start = match args.first() {
        Some(Value::Int(n)) => *n,
        _ => 0,
    };
    let end = match args.get(1) {
        Some(Value::Int(n)) => *n,
        _ => 10,
    };
    let step = match args.get(2) {
        Some(Value::Int(n)) if *n != 0 => *n,
        _ => 1,
    };

    let mut items = Vec::new();
    let mut curr = start;
    while (step > 0 && curr < end) || (step < 0 && curr > end) {
        items.push(Value::Int(curr));
        curr += step;
    }

    builtin_stream_from_array(&[Value::array(items)])
}

// ============================================================================
// 7. Sink Native Functions
// ============================================================================

pub fn builtin_sink_collect(_args: &[Value]) -> Result<NativeCallResult, String> {
    let mut map = LustMap::default();
    map.insert(string_key("__type"), Value::string("Sink"));
    map.insert(string_key("kind"), Value::string("collect"));
    Ok(NativeCallResult::Return(Value::map(map)))
}

pub fn builtin_sink_drain(_args: &[Value]) -> Result<NativeCallResult, String> {
    let mut map = LustMap::default();
    map.insert(string_key("__type"), Value::string("Sink"));
    map.insert(string_key("kind"), Value::string("drain"));
    Ok(NativeCallResult::Return(Value::map(map)))
}

// ============================================================================
// VM Registration
// ============================================================================

/// Registers all coordination builtins and module maps into the given `VM`.
pub fn register_builtins(vm: &mut VM) {
    let funcs: Vec<(&str, Rc<NativeFn>)> = vec![
        // Ref
        ("Ref", native_fn(builtin_ref_make)),
        ("Ref.make", native_fn(builtin_ref_make)),
        ("ref_make", native_fn(builtin_ref_make)),
        // Deferred
        ("Deferred", native_fn(builtin_deferred_make)),
        ("Deferred.make", native_fn(builtin_deferred_make)),
        ("deferred_make", native_fn(builtin_deferred_make)),
        // Semaphore
        ("Semaphore", native_fn(builtin_semaphore_make)),
        ("Semaphore.make", native_fn(builtin_semaphore_make)),
        ("semaphore_make", native_fn(builtin_semaphore_make)),
        // Queue
        ("Queue", native_fn(builtin_queue_make)),
        ("Queue.make", native_fn(builtin_queue_make)),
        ("queue_make", native_fn(builtin_queue_make)),
        // Schedule
        ("Schedule", native_fn(builtin_schedule_recurs)),
        ("Schedule.recurs", native_fn(builtin_schedule_recurs)),
        ("Schedule.spaced", native_fn(builtin_schedule_spaced)),
        ("Schedule.exponential", native_fn(builtin_schedule_exponential)),
        ("Schedule.retry", native_fn(builtin_schedule_retry)),
        ("Schedule.timeout", native_fn(builtin_schedule_timeout)),
        ("schedule_recurs", native_fn(builtin_schedule_recurs)),
        ("schedule_retry", native_fn(builtin_schedule_retry)),
        ("schedule_timeout", native_fn(builtin_schedule_timeout)),
        // Stream
        ("Stream", native_fn(builtin_stream_from_array)),
        ("Stream.from_array", native_fn(builtin_stream_from_array)),
        ("Stream.range", native_fn(builtin_stream_range)),
        ("Stream.collect", native_fn(builtin_stream_collect)),
        ("stream_from_array", native_fn(builtin_stream_from_array)),
        // Sink
        ("Sink", native_fn(builtin_sink_collect)),
        ("Sink.collect", native_fn(builtin_sink_collect)),
        ("Sink.drain", native_fn(builtin_sink_drain)),
        ("sink_collect", native_fn(builtin_sink_collect)),
    ];

    let mut coord_map = LustMap::default();
    let mut sync_map = LustMap::default();
    let mut stream_map = LustMap::default();
    let mut sched_map = LustMap::default();

    for (name, func) in &funcs {
        let val = Value::NativeFunction(func.clone());

        // Global and prefixed registrations
        vm.register_native(*name, val.clone());
        vm.register_native(format!("coord.{}", name), val.clone());
        vm.register_native(format!("std.coord.{}", name), val.clone());
        vm.register_native(format!("sync.{}", name), val.clone());
        vm.register_native(format!("std.sync.{}", name), val.clone());
        vm.register_native(format!("stream.{}", name), val.clone());
        vm.register_native(format!("std.stream.{}", name), val.clone());
        vm.register_native(format!("schedule.{}", name), val.clone());
        vm.register_native(format!("std.schedule.{}", name), val.clone());

        // Module maps
        coord_map.insert(string_key(name), val.clone());
        if name.starts_with("Queue") || name.starts_with("Deferred") || name.starts_with("Semaphore") || name.starts_with("Ref") {
            sync_map.insert(string_key(name), val.clone());
        }
        if name.starts_with("Stream") || name.starts_with("Sink") {
            stream_map.insert(string_key(name), val.clone());
        }
        if name.starts_with("Schedule") {
            sched_map.insert(string_key(name), val.clone());
        }
    }

    let coord_val = Value::map(coord_map);
    let sync_val = Value::map(sync_map);
    let stream_val = Value::map(stream_map);
    let sched_val = Value::map(sched_map);

    vm.register_native("coord", coord_val.clone());
    vm.register_native("std.coord", coord_val.clone());
    vm.register_native("@std/coord", coord_val);

    vm.register_native("sync", sync_val.clone());
    vm.register_native("std.sync", sync_val.clone());
    vm.register_native("@std/sync", sync_val);

    vm.register_native("stream", stream_val.clone());
    vm.register_native("std.stream", stream_val.clone());
    vm.register_native("@std/stream", stream_val);

    vm.register_native("schedule", sched_val.clone());
    vm.register_native("std.schedule", sched_val.clone());
    vm.register_native("@std/schedule", sched_val);
}

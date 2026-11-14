use alloc::format;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;

use crate::bytecode::{native_fn, LustMap, NativeCallResult, Value, ValueKey};
use crate::VM;

fn string_key(s: &str) -> ValueKey {
    ValueKey::from(Value::string(s))
}

fn unwrap_val(value: Value) -> Value {
    if let Value::Struct(instance) = &value {
        if instance.name.as_str() == "LuaValue" {
            let borrowed = instance.fields.borrow();
            if borrowed.len() == 1 {
                return borrowed[0].clone();
            }
        }
    }
    value
}

pub fn invoke_callable(func: &Value, args: Vec<Value>) -> Result<Value, String> {
    if let Value::NativeFunction(nf) = func {
        match nf(&args) {
            Ok(NativeCallResult::Return(v)) => Ok(v),
            Ok(_) => Ok(Value::Nil),
            Err(e) => Err(e),
        }
    } else {
        VM::with_current(|vm| vm.call_value(func, args).map_err(|e| e.to_string()))
    }
}

// =========================================================================
// @std/rx Bridge Implementation
// =========================================================================

pub fn create_rx_module(vm: &VM) -> Value {
    let entries = [
        (string_key("Observable"), create_rx_observable_fn()),
        (string_key("from_iter"), create_rx_observable_fn()),
        (string_key("interval"), create_rx_interval_fn()),
        (string_key("Scheduler"), create_rx_scheduler_fn()),
        (string_key("advance_by"), create_rx_advance_by_fn()),
        (string_key("unsubscribe"), create_rx_unsubscribe_fn()),
        (string_key("join"), create_rx_join_fn()),
        (string_key("BoundedQueue"), create_rx_bounded_queue_fn()),
        (string_key("BufferOptions"), create_rx_buffer_options_fn()),
        (string_key("default_buffer_options"), create_rx_default_buffer_options_fn()),
        (string_key("push"), create_rx_push_fn()),
        (string_key("pop"), create_rx_pop_fn()),
        (string_key("dequeue"), create_rx_pop_fn()),
        (string_key("dropped_count"), create_rx_dropped_count_fn()),
        (string_key("len"), create_rx_len_fn()),
        (string_key("used_bytes"), create_rx_used_bytes_fn()),
        (string_key("Subject"), create_rx_subject_fn()),
        (string_key("subject_subscribe"), create_rx_subject_subscribe_fn()),
        (string_key("subject_subscribe_with"), create_rx_subject_subscribe_with_fn()),
        (string_key("next"), create_rx_next_fn()),
        (string_key("complete"), create_rx_complete_fn()),
        (string_key("filter"), create_rx_filter_fn()),
        (string_key("map"), create_rx_map_fn()),
        (string_key("take"), create_rx_take_fn()),
        (string_key("subscribe"), create_rx_subscribe_fn()),
        (string_key("subscribe_with"), create_rx_subscribe_with_fn()),
    ];
    vm.map_with_entries(entries)
}

pub fn make_observable(items: Vec<Value>) -> Value {
    let mut map = LustMap::default();
    map.insert(string_key("__type"), Value::string("Observable"));
    map.insert(string_key("closed"), Value::Bool(false));
    map.insert(string_key("count"), Value::Int(items.len() as i64));
    map.insert(string_key("items"), Value::array(items));
    map.insert(string_key("filter"), create_rx_filter_fn());
    map.insert(string_key("map"), create_rx_map_fn());
    map.insert(string_key("take"), create_rx_take_fn());
    map.insert(string_key("subscribe"), create_rx_subscribe_fn());
    map.insert(string_key("subscribe_with"), create_rx_subscribe_with_fn());
    Value::map(map)
}

pub fn create_rx_observable_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let items = match args.first().cloned().map(unwrap_val) {
            Some(Value::Array(arr)) => arr.borrow().clone(),
            Some(val) if !matches!(val, Value::Nil) => vec![val],
            _ => Vec::new(),
        };
        Ok(NativeCallResult::Return(make_observable(items)))
    }))
}

pub fn create_rx_interval_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let period_ms = match args.first().cloned().map(unwrap_val) {
            Some(Value::Int(ms)) if ms >= 0 => ms,
            Some(Value::Int(_)) => return Err("rx.interval period_ms must be non-negative".into()),
            None => 0,
            _ => return Err("rx.interval expects integer period_ms".into()),
        };
        let mut map = LustMap::default();
        map.insert(string_key("__type"), Value::string("IntervalObservable"));
        map.insert(string_key("closed"), Value::Bool(false));
        map.insert(string_key("period_ms"), Value::Int(period_ms));
        Ok(NativeCallResult::Return(Value::map(map)))
    }))
}

pub fn create_rx_filter_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("rx.filter requires (observable, predicate_fn)".into());
        }
        let obs = unwrap_val(args[0].clone());
        let pred = unwrap_val(args[1].clone());

        let items = match obs {
            Value::Map(m) => {
                let borrowed = m.borrow();
                match borrowed.get(&string_key("items")) {
                    Some(Value::Array(a)) => a.borrow().clone(),
                    _ => Vec::new(),
                }
            }
            Value::Array(a) => a.borrow().clone(),
            _ => return Err("rx.filter expects Observable or Array as first argument".into()),
        };

        let mut filtered = Vec::new();
        for item in items {
            let res = invoke_callable(&pred, vec![item.clone()])?;
            if res.is_truthy() {
                filtered.push(item);
            }
        }
        Ok(NativeCallResult::Return(make_observable(filtered)))
    }))
}

pub fn create_rx_map_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("rx.map requires (observable, map_fn)".into());
        }
        let obs = unwrap_val(args[0].clone());
        let mapper = unwrap_val(args[1].clone());

        let items = match obs {
            Value::Map(m) => {
                let borrowed = m.borrow();
                match borrowed.get(&string_key("items")) {
                    Some(Value::Array(a)) => a.borrow().clone(),
                    _ => Vec::new(),
                }
            }
            Value::Array(a) => a.borrow().clone(),
            _ => return Err("rx.map expects Observable or Array as first argument".into()),
        };

        let mut mapped = Vec::with_capacity(items.len());
        for item in items {
            let res = invoke_callable(&mapper, vec![item])?;
            mapped.push(res);
        }
        Ok(NativeCallResult::Return(make_observable(mapped)))
    }))
}

pub fn create_rx_take_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("rx.take requires (observable, count)".into());
        }
        let obs = unwrap_val(args[0].clone());
        let n = match unwrap_val(args[1].clone()) {
            Value::Int(n) if n >= 0 => n as usize,
            _ => return Err("rx.take count must be a non-negative integer".into()),
        };

        let items = match obs {
            Value::Map(m) => {
                let borrowed = m.borrow();
                match borrowed.get(&string_key("items")) {
                    Some(Value::Array(a)) => a.borrow().clone(),
                    _ => Vec::new(),
                }
            }
            Value::Array(a) => a.borrow().clone(),
            _ => return Err("rx.take expects Observable or Array as first argument".into()),
        };

        let taken = items.into_iter().take(n).collect();
        Ok(NativeCallResult::Return(make_observable(taken)))
    }))
}

pub fn create_rx_subscribe_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("rx.subscribe requires (target, next_fn, [scheduler_or_options])".into());
        }
        let target = unwrap_val(args[0].clone());
        let next_fn = unwrap_val(args[1].clone());
        let extra = args.get(2).cloned().map(unwrap_val);

        if let Value::Map(m) = &target {
            let map_borrow = m.borrow();
            let ty = map_borrow.get(&string_key("__type")).and_then(|v| v.as_string());
            if ty == Some("Subject") {
                drop(map_borrow);
                return create_rx_subject_subscribe_fn_impl(&[target, next_fn]);
            }
            if ty == Some("IntervalObservable") {
                let period_ms = map_borrow.get(&string_key("period_ms")).and_then(|v| match v {
                    Value::Int(i) => Some(*i),
                    _ => None,
                }).unwrap_or(100);
                drop(map_borrow);

                let sub_map = Rc::new(RefCell::new(LustMap::default()));
                {
                    let mut s = sub_map.borrow_mut();
                    s.insert(string_key("__type"), Value::string("Subscription"));
                    s.insert(string_key("closed"), Value::Bool(false));
                    s.insert(string_key("exit"), Value::string("Running"));
                    s.insert(string_key("items_delivered"), Value::Int(0));
                    s.insert(string_key("period_ms"), Value::Int(period_ms));
                    s.insert(string_key("next_fn"), next_fn);
                }

                if let Some(Value::Map(sched_map)) = extra {
                    let mut sched_borrow = sched_map.borrow_mut();
                    if let Some(Value::Array(subs)) = sched_borrow.get_mut(&string_key("subscriptions")) {
                        subs.borrow_mut().push(Value::Map(sub_map.clone()));
                    }
                }

                return Ok(NativeCallResult::Return(Value::Map(sub_map)));
            }

            // Observable stream
            let items = match map_borrow.get(&string_key("items")) {
                Some(Value::Array(a)) => a.borrow().clone(),
                _ => Vec::new(),
            };
            drop(map_borrow);

            let mut count = 0;
            for item in items {
                let _ = invoke_callable(&next_fn, vec![item])?;
                count += 1;
            }

            let mut sub = LustMap::default();
            sub.insert(string_key("__type"), Value::string("Subscription"));
            sub.insert(string_key("closed"), Value::Bool(true));
            sub.insert(string_key("exit"), Value::string("Completed"));
            sub.insert(string_key("items_delivered"), Value::Int(count));
            return Ok(NativeCallResult::Return(Value::map(sub)));
        }

        Err("rx.subscribe expects an Observable or Subject target".into())
    }))
}

pub fn create_rx_subscribe_with_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 4 {
            return Err("rx.subscribe_with requires (target, next_fn, err_fn, complete_fn)".into());
        }
        let target = unwrap_val(args[0].clone());
        let next_fn = unwrap_val(args[1].clone());
        let _err_fn = unwrap_val(args[2].clone());
        let complete_fn = unwrap_val(args[3].clone());

        if let Value::Map(m) = &target {
            let map_borrow = m.borrow();
            let ty = map_borrow.get(&string_key("__type")).and_then(|v| v.as_string());
            if ty == Some("Subject") {
                drop(map_borrow);
                return create_rx_subject_subscribe_with_fn_impl(args);
            }

            let items = match map_borrow.get(&string_key("items")) {
                Some(Value::Array(a)) => a.borrow().clone(),
                _ => Vec::new(),
            };
            drop(map_borrow);

            let mut count = 0;
            for item in items {
                let _ = invoke_callable(&next_fn, vec![item])?;
                count += 1;
            }

            if !matches!(complete_fn, Value::Nil) {
                let _ = invoke_callable(&complete_fn, Vec::new())?;
            }

            let mut sub = LustMap::default();
            sub.insert(string_key("__type"), Value::string("Subscription"));
            sub.insert(string_key("closed"), Value::Bool(true));
            sub.insert(string_key("exit"), Value::string("Completed"));
            sub.insert(string_key("items_delivered"), Value::Int(count));
            return Ok(NativeCallResult::Return(Value::map(sub)));
        }

        Err("rx.subscribe_with expects an Observable or Subject target".into())
    }))
}

pub fn create_rx_scheduler_fn() -> Value {
    Value::NativeFunction(native_fn(|_args: &[Value]| {
        let mut map = LustMap::default();
        map.insert(string_key("__type"), Value::string("Scheduler"));
        map.insert(string_key("now_ms"), Value::Int(0));
        map.insert(string_key("subscriptions"), Value::array(Vec::new()));
        Ok(NativeCallResult::Return(Value::map(map)))
    }))
}

pub fn create_rx_advance_by_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("rx.advance_by requires (scheduler, delta_ms)".into());
        }
        let sched = unwrap_val(args[0].clone());
        let delta = match unwrap_val(args[1].clone()) {
            Value::Int(d) if d >= 0 => d,
            _ => return Err("rx.advance_by delta_ms must be non-negative integer".into()),
        };

        if let Value::Map(m) = sched {
            let mut map_borrow = m.borrow_mut();
            let cur_now = match map_borrow.get(&string_key("now_ms")) {
                Some(Value::Int(n)) => *n,
                _ => 0,
            };
            let next_now = cur_now + delta;
            map_borrow.insert(string_key("now_ms"), Value::Int(next_now));

            let subs = match map_borrow.get(&string_key("subscriptions")) {
                Some(Value::Array(a)) => a.borrow().clone(),
                _ => Vec::new(),
            };
            drop(map_borrow);

            for sub_val in subs {
                if let Value::Map(sub_m) = sub_val {
                    let mut s = sub_m.borrow_mut();
                    let closed = s.get(&string_key("closed")).map(|b| b.is_truthy()).unwrap_or(false);
                    if closed {
                        continue;
                    }
                    let period_ms = match s.get(&string_key("period_ms")) {
                        Some(Value::Int(p)) if *p > 0 => *p,
                        _ => continue,
                    };
                    let delivered = match s.get(&string_key("items_delivered")) {
                        Some(Value::Int(d)) => *d,
                        _ => 0,
                    };
                    let next_fn = s.get(&string_key("next_fn")).cloned().unwrap_or(Value::Nil);

                    let ticks = delta / period_ms;
                    for i in 0..ticks {
                        let tick_val = Value::Int(delivered + i);
                        let _ = invoke_callable(&next_fn, vec![tick_val]);
                    }
                    s.insert(string_key("items_delivered"), Value::Int(delivered + ticks));
                }
            }

            return Ok(NativeCallResult::Return(Value::Int(next_now)));
        }

        Err("rx.advance_by expects Scheduler as first argument".into())
    }))
}

pub fn create_rx_unsubscribe_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("rx.unsubscribe requires a Subscription argument".into());
        }
        let sub = unwrap_val(args[0].clone());
        if let Value::Map(m) = sub {
            let mut borrowed = m.borrow_mut();
            borrowed.insert(string_key("closed"), Value::Bool(true));
            borrowed.insert(string_key("exit"), Value::string("Completed"));
            return Ok(NativeCallResult::Return(Value::Bool(true)));
        }
        Err("rx.unsubscribe expects Subscription".into())
    }))
}

pub fn create_rx_join_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("rx.join requires a Subscription argument".into());
        }
        let sub = unwrap_val(args[0].clone());
        if let Value::Map(m) = sub {
            let borrowed = m.borrow();
            let exit_val = borrowed.get(&string_key("exit")).cloned().unwrap_or(Value::string("Completed"));
            let delivered = borrowed.get(&string_key("items_delivered")).cloned().unwrap_or(Value::Int(0));
            drop(borrowed);

            let mut comp = LustMap::default();
            comp.insert(string_key("exit"), exit_val);
            comp.insert(string_key("items_delivered"), delivered);
            return Ok(NativeCallResult::Return(Value::map(comp)));
        }
        Err("rx.join expects Subscription".into())
    }))
}

pub fn create_rx_bounded_queue_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let max_items = match args.first().cloned().map(unwrap_val) {
            Some(Value::Int(i)) if i > 0 => i as usize,
            Some(Value::Map(m)) => {
                let borrowed = m.borrow();
                borrowed.get(&string_key("max_items")).and_then(|v| match v {
                    Value::Int(i) => Some(*i as usize),
                    _ => None,
                }).unwrap_or(64)
            }
            _ => 64,
        };
        let max_bytes = match args.get(1).cloned().map(unwrap_val) {
            Some(Value::Int(b)) if b > 0 => b as usize,
            _ => 65536,
        };
        let strategy = match args.get(2).cloned().map(unwrap_val) {
            Some(Value::String(s)) => s.as_str().to_string(),
            _ => "DropNewest".to_string(),
        };

        let mut map = LustMap::default();
        map.insert(string_key("__type"), Value::string("BoundedQueue"));
        map.insert(string_key("max_items"), Value::Int(max_items as i64));
        map.insert(string_key("max_bytes"), Value::Int(max_bytes as i64));
        map.insert(string_key("strategy"), Value::string(strategy));
        map.insert(string_key("items"), Value::array(Vec::new()));
        map.insert(string_key("dropped_count"), Value::Int(0));
        map.insert(string_key("used_bytes"), Value::Int(0));
        Ok(NativeCallResult::Return(Value::map(map)))
    }))
}

pub fn create_rx_buffer_options_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let max_items = match args.first().cloned().map(unwrap_val) {
            Some(Value::Int(i)) => i,
            _ => 64,
        };
        let max_bytes = match args.get(1).cloned().map(unwrap_val) {
            Some(Value::Int(b)) => b,
            _ => 65536,
        };
        let low_items = match args.get(2).cloned().map(unwrap_val) {
            Some(Value::Int(li)) => li,
            _ => 16,
        };
        let low_bytes = match args.get(3).cloned().map(unwrap_val) {
            Some(Value::Int(lb)) => lb,
            _ => 16384,
        };
        let strategy = match args.get(4).cloned().map(unwrap_val) {
            Some(Value::String(s)) => s.as_str().to_string(),
            _ => "DropNewest".to_string(),
        };

        let mut map = LustMap::default();
        map.insert(string_key("max_items"), Value::Int(max_items));
        map.insert(string_key("max_bytes"), Value::Int(max_bytes));
        map.insert(string_key("low_watermark_items"), Value::Int(low_items));
        map.insert(string_key("low_watermark_bytes"), Value::Int(low_bytes));
        map.insert(string_key("overflow"), Value::string(strategy));
        Ok(NativeCallResult::Return(Value::map(map)))
    }))
}

pub fn create_rx_default_buffer_options_fn() -> Value {
    Value::NativeFunction(native_fn(|_args: &[Value]| {
        let mut map = LustMap::default();
        map.insert(string_key("max_items"), Value::Int(64));
        map.insert(string_key("max_bytes"), Value::Int(65536));
        map.insert(string_key("low_watermark_items"), Value::Int(16));
        map.insert(string_key("low_watermark_bytes"), Value::Int(16384));
        map.insert(string_key("overflow"), Value::string("DropNewest"));
        Ok(NativeCallResult::Return(Value::map(map)))
    }))
}

pub fn create_rx_push_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("rx.push requires (queue, item, [bytes])".into());
        }
        let queue_val = unwrap_val(args[0].clone());
        let item = unwrap_val(args[1].clone());
        let item_bytes = match args.get(2).cloned().map(unwrap_val) {
            Some(Value::Int(b)) if b >= 0 => b,
            _ => 1,
        };

        if let Value::Map(m) = queue_val {
            let mut q = m.borrow_mut();
            let max_items = match q.get(&string_key("max_items")) {
                Some(Value::Int(i)) => *i as usize,
                _ => 64,
            };
            let strategy = match q.get(&string_key("strategy")) {
                Some(Value::String(s)) => s.as_str().to_string(),
                _ => "DropNewest".to_string(),
            };
            let dropped = match q.get(&string_key("dropped_count")) {
                Some(Value::Int(d)) => *d,
                _ => 0,
            };

            let items_arr = match q.get(&string_key("items")) {
                Some(Value::Array(a)) => a.clone(),
                _ => return Err("Invalid queue items".into()),
            };

            let mut arr_borrow = items_arr.borrow_mut();
            if arr_borrow.len() < max_items {
                arr_borrow.push(item);
                let cur_bytes = match q.get(&string_key("used_bytes")) {
                    Some(Value::Int(b)) => *b,
                    _ => 0,
                };
                q.insert(string_key("used_bytes"), Value::Int(cur_bytes + item_bytes));
                return Ok(NativeCallResult::Return(Value::string("Enqueued")));
            }

            // Overflow branch
            match strategy.as_str() {
                "Fail" | "Reject" => {
                    return Ok(NativeCallResult::Return(Value::string("OverflowError")));
                }
                "DropOldest" => {
                    let _oldest = arr_borrow.remove(0);
                    arr_borrow.push(item);
                    q.insert(string_key("dropped_count"), Value::Int(dropped + 1));
                    return Ok(NativeCallResult::Return(Value::string("ReplacedOldest")));
                }
                "DropNewest" | _ => {
                    q.insert(string_key("dropped_count"), Value::Int(dropped + 1));
                    return Ok(NativeCallResult::Return(Value::string("DroppedNewest")));
                }
            }
        }

        Err("rx.push expects BoundedQueue as first argument".into())
    }))
}

pub fn create_rx_pop_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("rx.pop requires a queue argument".into());
        }
        let queue_val = unwrap_val(args[0].clone());
        if let Value::Map(m) = queue_val {
            let q = m.borrow();
            if let Some(Value::Array(a)) = q.get(&string_key("items")) {
                let mut arr_borrow = a.borrow_mut();
                if !arr_borrow.is_empty() {
                    let item = arr_borrow.remove(0);
                    return Ok(NativeCallResult::Return(item));
                }
                return Ok(NativeCallResult::Return(Value::enum_unit("Option", "None")));
            }
        }
        Err("rx.pop expects BoundedQueue".into())
    }))
}

pub fn create_rx_dropped_count_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("rx.dropped_count requires a queue argument".into());
        }
        let queue_val = unwrap_val(args[0].clone());
        if let Value::Map(m) = queue_val {
            let q = m.borrow();
            let dropped = q.get(&string_key("dropped_count")).cloned().unwrap_or(Value::Int(0));
            return Ok(NativeCallResult::Return(dropped));
        }
        Err("rx.dropped_count expects BoundedQueue".into())
    }))
}

pub fn create_rx_len_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("rx.len requires a queue or observable argument".into());
        }
        let val = unwrap_val(args[0].clone());
        if let Value::Map(m) = val {
            let borrowed = m.borrow();
            if let Some(Value::Array(a)) = borrowed.get(&string_key("items")) {
                let len = a.borrow().len() as i64;
                return Ok(NativeCallResult::Return(Value::Int(len)));
            }
            if let Some(Value::Int(c)) = borrowed.get(&string_key("count")) {
                return Ok(NativeCallResult::Return(Value::Int(*c)));
            }
        }
        Err("rx.len expects BoundedQueue or Observable".into())
    }))
}

pub fn create_rx_used_bytes_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("rx.used_bytes requires a queue argument".into());
        }
        let queue_val = unwrap_val(args[0].clone());
        if let Value::Map(m) = queue_val {
            let q = m.borrow();
            let bytes = q.get(&string_key("used_bytes")).cloned().unwrap_or(Value::Int(0));
            return Ok(NativeCallResult::Return(bytes));
        }
        Err("rx.used_bytes expects BoundedQueue".into())
    }))
}

pub fn create_rx_subject_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let max_items = match args.first().cloned().map(unwrap_val) {
            Some(Value::Int(i)) if i > 0 => i as usize,
            _ => 64,
        };
        let max_bytes = match args.get(1).cloned().map(unwrap_val) {
            Some(Value::Int(b)) if b > 0 => b as usize,
            _ => 65536,
        };
        let strategy = match args.get(2).cloned().map(unwrap_val) {
            Some(Value::String(s)) => s.as_str().to_string(),
            _ => "DropNewest".to_string(),
        };

        let mut map = LustMap::default();
        map.insert(string_key("__type"), Value::string("Subject"));
        map.insert(string_key("closed"), Value::Bool(false));
        map.insert(string_key("count"), Value::Int(0));
        map.insert(string_key("max_items"), Value::Int(max_items as i64));
        map.insert(string_key("max_bytes"), Value::Int(max_bytes as i64));
        map.insert(string_key("strategy"), Value::string(strategy));
        map.insert(string_key("subscribers"), Value::array(Vec::new()));
        Ok(NativeCallResult::Return(Value::map(map)))
    }))
}

fn create_rx_subject_subscribe_fn_impl(args: &[Value]) -> Result<NativeCallResult, String> {
    if args.len() < 2 {
        return Err("rx.subject_subscribe requires (subject, next_fn)".into());
    }
    let subj = unwrap_val(args[0].clone());
    let next_fn = unwrap_val(args[1].clone());

    if let Value::Map(m) = subj {
        let map_borrow = m.borrow();
        let subs_arr = match map_borrow.get(&string_key("subscribers")) {
            Some(Value::Array(a)) => a.clone(),
            _ => return Err("Invalid subject".into()),
        };

        let mut sub = LustMap::default();
        sub.insert(string_key("__type"), Value::string("Subscription"));
        sub.insert(string_key("closed"), Value::Bool(false));
        sub.insert(string_key("exit"), Value::string("Completed"));
        sub.insert(string_key("items_delivered"), Value::Int(0));
        sub.insert(string_key("next_fn"), next_fn);
        sub.insert(string_key("items"), Value::array(Vec::new()));

        let sub_val = Value::map(sub);
        subs_arr.borrow_mut().push(sub_val.clone());
        return Ok(NativeCallResult::Return(sub_val));
    }
    Err("rx.subject_subscribe expects Subject".into())
}

pub fn create_rx_subject_subscribe_fn() -> Value {
    Value::NativeFunction(native_fn(create_rx_subject_subscribe_fn_impl))
}

fn create_rx_subject_subscribe_with_fn_impl(args: &[Value]) -> Result<NativeCallResult, String> {
    if args.len() < 4 {
        return Err("rx.subject_subscribe_with requires (subject, next_fn, err_fn, complete_fn, [options])".into());
    }
    let subj = unwrap_val(args[0].clone());
    let next_fn = unwrap_val(args[1].clone());
    let err_fn = unwrap_val(args[2].clone());
    let complete_fn = unwrap_val(args[3].clone());

    let (max_items, strategy) = if let Some(Value::Map(opt_m)) = args.get(4).cloned().map(unwrap_val) {
        let opt_b = opt_m.borrow();
        let mi = opt_b.get(&string_key("max_items")).and_then(|v| match v {
            Value::Int(i) => Some(*i as usize),
            _ => None,
        }).unwrap_or(64);
        let st = opt_b.get(&string_key("overflow")).and_then(|v| v.as_string()).unwrap_or("DropNewest").to_string();
        (mi, st)
    } else {
        (64, "DropNewest".to_string())
    };

    if let Value::Map(m) = subj {
        let map_borrow = m.borrow();
        let subs_arr = match map_borrow.get(&string_key("subscribers")) {
            Some(Value::Array(a)) => a.clone(),
            _ => return Err("Invalid subject".into()),
        };

        let mut sub = LustMap::default();
        sub.insert(string_key("__type"), Value::string("Subscription"));
        sub.insert(string_key("closed"), Value::Bool(false));
        sub.insert(string_key("exit"), Value::string("Completed"));
        sub.insert(string_key("items_delivered"), Value::Int(0));
        sub.insert(string_key("next_fn"), next_fn);
        sub.insert(string_key("err_fn"), err_fn);
        sub.insert(string_key("complete_fn"), complete_fn);
        sub.insert(string_key("max_items"), Value::Int(max_items as i64));
        sub.insert(string_key("strategy"), Value::string(strategy));
        sub.insert(string_key("queue_len"), Value::Int(0));

        let sub_val = Value::map(sub);
        subs_arr.borrow_mut().push(sub_val.clone());
        return Ok(NativeCallResult::Return(sub_val));
    }
    Err("rx.subject_subscribe_with expects Subject".into())
}

pub fn create_rx_subject_subscribe_with_fn() -> Value {
    Value::NativeFunction(native_fn(create_rx_subject_subscribe_with_fn_impl))
}

pub fn create_rx_next_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("rx.next requires (subject, value)".into());
        }
        let subj = unwrap_val(args[0].clone());
        let val = unwrap_val(args[1].clone());

        if let Value::Map(m) = subj {
            let mut map_borrow = m.borrow_mut();
            let closed = map_borrow.get(&string_key("closed")).map(|b| b.is_truthy()).unwrap_or(false);
            if closed {
                return Err("Subject is already closed".into());
            }

            let cur_count = match map_borrow.get(&string_key("count")) {
                Some(Value::Int(c)) => *c,
                _ => 0,
            };
            map_borrow.insert(string_key("count"), Value::Int(cur_count + 1));

            let subs = match map_borrow.get(&string_key("subscribers")) {
                Some(Value::Array(a)) => a.borrow().clone(),
                _ => Vec::new(),
            };
            drop(map_borrow);

            let mut delivered_count = 0;
            for sub_val in subs {
                if let Value::Map(sub_m) = sub_val {
                    let mut s = sub_m.borrow_mut();
                    let sub_closed = s.get(&string_key("closed")).map(|b| b.is_truthy()).unwrap_or(false);
                    if sub_closed {
                        continue;
                    }

                    let max_items = match s.get(&string_key("max_items")) {
                        Some(Value::Int(i)) => *i as usize,
                        _ => 64,
                    };
                    let strategy = match s.get(&string_key("strategy")) {
                        Some(Value::String(st)) => st.as_str().to_string(),
                        _ => "DropNewest".to_string(),
                    };
                    let q_len = match s.get(&string_key("queue_len")) {
                        Some(Value::Int(l)) => *l as usize,
                        _ => 0,
                    };

                    // Check for slow subscriber overflow
                    if q_len >= max_items && (strategy == "Fail" || strategy == "Reject") {
                        // Isolate failure to slow subscriber!
                        s.insert(string_key("closed"), Value::Bool(true));
                        s.insert(string_key("exit"), Value::string("Failed(BackpressureOverflow)"));
                        let err_fn = s.get(&string_key("err_fn")).cloned().unwrap_or(Value::Nil);
                        drop(s);
                        if !matches!(err_fn, Value::Nil) {
                            let _ = invoke_callable(&err_fn, vec![Value::string("BackpressureOverflow")]);
                        }
                        continue;
                    }

                    s.insert(string_key("queue_len"), Value::Int((q_len + 1) as i64));
                    let delivered = match s.get(&string_key("items_delivered")) {
                        Some(Value::Int(d)) => *d,
                        _ => 0,
                    };
                    s.insert(string_key("items_delivered"), Value::Int(delivered + 1));
                    let next_fn = s.get(&string_key("next_fn")).cloned().unwrap_or(Value::Nil);
                    drop(s);

                    if !matches!(next_fn, Value::Nil) {
                        let _ = invoke_callable(&next_fn, vec![val.clone()]);
                    }
                    delivered_count += 1;
                }
            }

            return Ok(NativeCallResult::Return(Value::Int(delivered_count)));
        }

        Err("rx.next expects Subject".into())
    }))
}

pub fn create_rx_complete_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("rx.complete requires a subject argument".into());
        }
        let subj = unwrap_val(args[0].clone());
        if let Value::Map(m) = subj {
            let mut map_borrow = m.borrow_mut();
            map_borrow.insert(string_key("closed"), Value::Bool(true));

            let subs = match map_borrow.get(&string_key("subscribers")) {
                Some(Value::Array(a)) => a.borrow().clone(),
                _ => Vec::new(),
            };
            drop(map_borrow);

            for sub_val in subs {
                if let Value::Map(sub_m) = sub_val {
                    let mut s = sub_m.borrow_mut();
                    let sub_closed = s.get(&string_key("closed")).map(|b| b.is_truthy()).unwrap_or(false);
                    if !sub_closed {
                        s.insert(string_key("closed"), Value::Bool(true));
                        let complete_fn = s.get(&string_key("complete_fn")).cloned().unwrap_or(Value::Nil);
                        drop(s);
                        if !matches!(complete_fn, Value::Nil) {
                            let _ = invoke_callable(&complete_fn, Vec::new());
                        }
                    }
                }
            }

            return Ok(NativeCallResult::Return(Value::Bool(true)));
        }
        Err("rx.complete expects Subject".into())
    }))
}

// =========================================================================
// @std/test Bridge Implementation
// =========================================================================

pub fn create_test_module(vm: &VM) -> Value {
    let entries = [
        (string_key("clock_new"), create_test_clock_new_fn()),
        (string_key("clock_now_ms"), create_test_clock_now_ms_fn()),
        (string_key("clock_schedule"), create_test_clock_schedule_fn()),
        (string_key("clock_adjust"), create_test_clock_adjust_fn()),
        (string_key("clock_pending_count"), create_test_clock_pending_count_fn()),
        (string_key("schedule_exponential"), create_test_schedule_exponential_fn()),
        (string_key("schedule_step"), create_test_schedule_step_fn()),
        (string_key("mock_new"), create_test_mock_new_fn()),
        (string_key("mock_set_response"), create_test_mock_set_response_fn()),
        (string_key("mock_record_call"), create_test_mock_record_call_fn()),
        (string_key("mock_call_count"), create_test_mock_call_count_fn()),
        (string_key("mock_last_call"), create_test_mock_last_call_fn()),
        (string_key("projection_new"), create_test_projection_new_fn()),
        (string_key("projection_get"), create_test_projection_get_fn()),
        (string_key("resource_new"), create_test_resource_new_fn()),
        (string_key("resource_is_active"), create_test_resource_is_active_fn()),
        (string_key("scope_new"), create_test_scope_new_fn()),
        (string_key("scope_is_open"), create_test_scope_is_open_fn()),
        (string_key("scope_teardown_state"), create_test_scope_teardown_state_fn()),
        (string_key("scope_admission_state"), create_test_scope_admission_state_fn()),
        (string_key("scope_register_finalizer"), create_test_scope_register_finalizer_fn()),
        (string_key("scope_register_child"), create_test_scope_register_child_fn()),
        (string_key("scope_begin_teardown"), create_test_scope_begin_teardown_fn()),
        (string_key("scope_mark_joining"), create_test_scope_mark_joining_fn()),
        (string_key("scope_current_releases"), create_test_scope_current_releases_fn()),
        (string_key("scope_finalize"), create_test_scope_finalize_fn()),
        (string_key("fiber_new"), create_test_fiber_new_fn()),
        (string_key("fiber_set_state"), create_test_fiber_set_state_fn()),
        (string_key("fiber_enter_mask"), create_test_fiber_enter_mask_fn()),
        (string_key("fiber_exit_mask"), create_test_fiber_exit_mask_fn()),
        (string_key("fiber_is_masked"), create_test_fiber_is_masked_fn()),
        (string_key("fiber_request_interrupt"), create_test_fiber_request_interrupt_fn()),
        (string_key("fiber_state"), create_test_fiber_state_fn()),
    ];
    vm.map_with_entries(entries)
}

pub fn create_test_clock_new_fn() -> Value {
    Value::NativeFunction(native_fn(|_args: &[Value]| {
        let mut map = LustMap::default();
        map.insert(string_key("__type"), Value::string("TestClock"));
        map.insert(string_key("now_ms"), Value::Int(0));
        map.insert(string_key("timers"), Value::array(Vec::new()));
        map.insert(string_key("next_id"), Value::Int(1));
        Ok(NativeCallResult::Return(Value::map(map)))
    }))
}

pub fn create_test_clock_now_ms_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("test.clock_now_ms requires clock".into());
        }
        let clock = unwrap_val(args[0].clone());
        if let Value::Map(m) = clock {
            let borrowed = m.borrow();
            let now = borrowed.get(&string_key("now_ms")).cloned().unwrap_or(Value::Int(0));
            return Ok(NativeCallResult::Return(now));
        }
        Err("test.clock_now_ms expects TestClock".into())
    }))
}

pub fn create_test_clock_schedule_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 3 {
            return Err("test.clock_schedule requires (clock, fiber_id_or_delay, delay_or_label)".into());
        }
        let clock = unwrap_val(args[0].clone());
        let (fiber_id, delay_ms, label) = match (unwrap_val(args[1].clone()), unwrap_val(args[2].clone())) {
            (Value::Int(fid), Value::Int(del)) => (fid, del, format!("fiber_{}", fid)),
            (Value::Int(del), Value::String(lbl)) => (0, del, lbl.as_str().to_string()),
            _ => return Err("Invalid arguments for clock_schedule".into()),
        };

        if let Value::Map(m) = clock {
            let mut borrowed = m.borrow_mut();
            let now = match borrowed.get(&string_key("now_ms")) {
                Some(Value::Int(n)) => *n,
                _ => 0,
            };
            let next_id = match borrowed.get(&string_key("next_id")) {
                Some(Value::Int(id)) => *id,
                _ => 1,
            };
            borrowed.insert(string_key("next_id"), Value::Int(next_id + 1));

            let deadline = now + delay_ms;
            let mut timer = LustMap::default();
            timer.insert(string_key("id"), Value::Int(next_id));
            timer.insert(string_key("fiber_id"), Value::Int(fiber_id));
            timer.insert(string_key("delay_ms"), Value::Int(delay_ms));
            timer.insert(string_key("deadline_ms"), Value::Int(deadline));
            timer.insert(string_key("label"), Value::string(label));

            if let Some(Value::Array(arr)) = borrowed.get(&string_key("timers")) {
                arr.borrow_mut().push(Value::map(timer));
            }

            return Ok(NativeCallResult::Return(Value::Int(next_id)));
        }

        Err("test.clock_schedule expects TestClock".into())
    }))
}

pub fn create_test_clock_adjust_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("test.clock_adjust requires (clock, delta_ms)".into());
        }
        let clock = unwrap_val(args[0].clone());
        let delta = match unwrap_val(args[1].clone()) {
            Value::Int(d) if d >= 0 => d,
            _ => return Err("delta_ms must be non-negative integer".into()),
        };

        if let Value::Map(m) = clock {
            let mut borrowed = m.borrow_mut();
            let cur_now = match borrowed.get(&string_key("now_ms")) {
                Some(Value::Int(n)) => *n,
                _ => 0,
            };
            let next_now = cur_now + delta;
            borrowed.insert(string_key("now_ms"), Value::Int(next_now));

            let timers_arr = match borrowed.get(&string_key("timers")) {
                Some(Value::Array(a)) => a.clone(),
                _ => return Err("Invalid timers array".into()),
            };

            let all_timers = timers_arr.borrow().clone();
            let mut ready = Vec::new();
            let mut remaining = Vec::new();

            for t_val in all_timers {
                if let Value::Map(tm) = &t_val {
                    let tb = tm.borrow();
                    let deadline = match tb.get(&string_key("deadline_ms")) {
                        Some(Value::Int(d)) => *d,
                        _ => 0,
                    };
                    if deadline <= next_now {
                        ready.push(t_val.clone());
                    } else {
                        remaining.push(t_val.clone());
                    }
                }
            }

            // Sort ready by deadline ascending, then id ascending
            ready.sort_by(|a, b| {
                let d_a = if let Value::Map(m) = a {
                    m.borrow().get(&string_key("deadline_ms")).and_then(|v| match v { Value::Int(i) => Some(*i), _ => None }).unwrap_or(0)
                } else { 0 };
                let d_b = if let Value::Map(m) = b {
                    m.borrow().get(&string_key("deadline_ms")).and_then(|v| match v { Value::Int(i) => Some(*i), _ => None }).unwrap_or(0)
                } else { 0 };
                d_a.cmp(&d_b)
            });

            *timers_arr.borrow_mut() = remaining;

            let awakened: Vec<Value> = ready.into_iter().filter_map(|t_val| {
                if let Value::Map(tm) = t_val {
                    tm.borrow().get(&string_key("fiber_id")).cloned()
                } else {
                    None
                }
            }).collect();

            return Ok(NativeCallResult::Return(Value::array(awakened)));
        }

        Err("test.clock_adjust expects TestClock".into())
    }))
}

pub fn create_test_clock_pending_count_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("test.clock_pending_count requires clock".into());
        }
        let clock = unwrap_val(args[0].clone());
        if let Value::Map(m) = clock {
            let borrowed = m.borrow();
            if let Some(Value::Array(a)) = borrowed.get(&string_key("timers")) {
                let count = a.borrow().len() as i64;
                return Ok(NativeCallResult::Return(Value::Int(count)));
            }
        }
        Err("test.clock_pending_count expects TestClock".into())
    }))
}

pub fn create_test_schedule_exponential_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let base_ms = match args.first().cloned().map(unwrap_val) {
            Some(Value::Int(i)) => i,
            _ => 100,
        };
        let factor = match args.get(1).cloned().map(unwrap_val) {
            Some(Value::Float(f)) => f,
            Some(Value::Int(i)) => i as f64,
            _ => 2.0,
        };
        let max_ms = match args.get(2).cloned().map(unwrap_val) {
            Some(Value::Int(i)) => i,
            _ => 10000,
        };

        let mut map = LustMap::default();
        map.insert(string_key("kind"), Value::string("exponential"));
        map.insert(string_key("base_ms"), Value::Int(base_ms));
        map.insert(string_key("factor"), Value::Float(factor));
        map.insert(string_key("max_ms"), Value::Int(max_ms));
        Ok(NativeCallResult::Return(Value::map(map)))
    }))
}

pub fn create_test_schedule_step_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("test.schedule_step requires (sched, state)".into());
        }
        let sched = unwrap_val(args[0].clone());
        let state = unwrap_val(args[1].clone());

        let (base_ms, factor, max_ms) = if let Value::Map(sm) = sched {
            let sb = sm.borrow();
            let b = sb.get(&string_key("base_ms")).and_then(|v| match v { Value::Int(i) => Some(*i), _ => None }).unwrap_or(100);
            let f = sb.get(&string_key("factor")).and_then(|v| match v { Value::Float(fl) => Some(*fl), Value::Int(i) => Some(*i as f64), _ => None }).unwrap_or(2.0);
            let m = sb.get(&string_key("max_ms")).and_then(|v| match v { Value::Int(i) => Some(*i), _ => None }).unwrap_or(10000);
            (b, f, m)
        } else {
            (100, 2.0, 10000)
        };

        let (attempt, total_delay) = if let Value::Map(st) = state {
            let stb = st.borrow();
            let a = stb.get(&string_key("attempt")).and_then(|v| match v { Value::Int(i) => Some(*i), _ => None }).unwrap_or(0);
            let t = stb.get(&string_key("total_delay_ms")).and_then(|v| match v { Value::Int(i) => Some(*i), _ => None }).unwrap_or(0);
            (a, t)
        } else if let Value::Int(a) = state {
            (a, 0)
        } else {
            (0, 0)
        };

        if attempt >= 10 {
            let mut done = LustMap::default();
            done.insert(string_key("done"), Value::Bool(true));
            return Ok(NativeCallResult::Return(Value::map(done)));
        }

        let delay_calc = (base_ms as f64 * factor.powi(attempt as i32)) as i64;
        let delay_ms = delay_calc.min(max_ms);

        let mut next_st = LustMap::default();
        next_st.insert(string_key("attempt"), Value::Int(attempt + 1));
        next_st.insert(string_key("total_delay_ms"), Value::Int(total_delay + delay_ms));

        let mut res = LustMap::default();
        res.insert(string_key("done"), Value::Bool(false));
        res.insert(string_key("delay_ms"), Value::Int(delay_ms));
        res.insert(string_key("next_state"), Value::map(next_st));
        Ok(NativeCallResult::Return(Value::map(res)))
    }))
}

pub fn create_test_mock_new_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let tag_or_name = args.first().cloned().map(unwrap_val).unwrap_or(Value::string("mock"));
        let mut map = LustMap::default();
        map.insert(string_key("__type"), Value::string("MockService"));
        map.insert(string_key("tag"), tag_or_name);
        map.insert(string_key("responses"), Value::map(LustMap::default()));
        map.insert(string_key("calls"), Value::map(LustMap::default()));
        Ok(NativeCallResult::Return(Value::map(map)))
    }))
}

pub fn create_test_mock_set_response_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 3 {
            return Err("test.mock_set_response requires (mock, method, resp)".into());
        }
        let mock = unwrap_val(args[0].clone());
        let method_val = unwrap_val(args[1].clone());
        let method = match method_val.as_string() {
            Some(s) => s.to_string(),
            None => return Err("method must be string".into()),
        };
        let resp = unwrap_val(args[2].clone());

        if let Value::Map(m) = mock {
            let mut borrowed = m.borrow_mut();
            if let Some(Value::Map(resp_map)) = borrowed.get_mut(&string_key("responses")) {
                resp_map.borrow_mut().insert(string_key(&method), resp);
                return Ok(NativeCallResult::Return(Value::Bool(true)));
            }
        }
        Err("test.mock_set_response expects MockService".into())
    }))
}

pub fn create_test_mock_record_call_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 3 {
            return Err("test.mock_record_call requires (mock, method, args, [timestamp_ms])".into());
        }
        let mock = unwrap_val(args[0].clone());
        let method_val = unwrap_val(args[1].clone());
        let method = match method_val.as_string() {
            Some(s) => s.to_string(),
            None => return Err("method must be string".into()),
        };
        let call_args = match unwrap_val(args[2].clone()) {
            Value::Array(a) => a.borrow().clone(),
            other => vec![other],
        };
        let timestamp_ms = match args.get(3).cloned().map(unwrap_val) {
            Some(Value::Int(i)) => i,
            _ => 0,
        };

        if let Value::Map(m) = mock {
            let mut borrowed = m.borrow_mut();
            if let Some(Value::Map(calls_map)) = borrowed.get_mut(&string_key("calls")) {
                let mut cm = calls_map.borrow_mut();
                let m_key = string_key(&method);
                if !cm.contains_key(&m_key) {
                    cm.insert(m_key.clone(), Value::array(Vec::new()));
                }
                if let Some(Value::Array(arr)) = cm.get_mut(&m_key) {
                    let mut rec = LustMap::default();
                    rec.insert(string_key("args"), Value::array(call_args));
                    rec.insert(string_key("timestamp_ms"), Value::Int(timestamp_ms));
                    arr.borrow_mut().push(Value::map(rec));
                }
                return Ok(NativeCallResult::Return(Value::Bool(true)));
            }
        }
        Err("test.mock_record_call expects MockService".into())
    }))
}

pub fn create_test_mock_call_count_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("test.mock_call_count requires (mock, method)".into());
        }
        let mock = unwrap_val(args[0].clone());
        let method_val = unwrap_val(args[1].clone());
        let method = match method_val.as_string() {
            Some(s) => s.to_string(),
            None => return Err("method must be string".into()),
        };

        if let Value::Map(m) = mock {
            let borrowed = m.borrow();
            if let Some(Value::Map(calls_map)) = borrowed.get(&string_key("calls")) {
                let cm = calls_map.borrow();
                if let Some(Value::Array(arr)) = cm.get(&string_key(&method)) {
                    let count = arr.borrow().len() as i64;
                    return Ok(NativeCallResult::Return(Value::Int(count)));
                }
                return Ok(NativeCallResult::Return(Value::Int(0)));
            }
        }
        Err("test.mock_call_count expects MockService".into())
    }))
}

pub fn create_test_mock_last_call_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("test.mock_last_call requires (mock, method)".into());
        }
        let mock = unwrap_val(args[0].clone());
        let method_val = unwrap_val(args[1].clone());
        let method = match method_val.as_string() {
            Some(s) => s.to_string(),
            None => return Err("method must be string".into()),
        };

        if let Value::Map(m) = mock {
            let borrowed = m.borrow();
            if let Some(Value::Map(calls_map)) = borrowed.get(&string_key("calls")) {
                let cm = calls_map.borrow();
                if let Some(Value::Array(arr)) = cm.get(&string_key(&method)) {
                    let ab = arr.borrow();
                    if let Some(last) = ab.last() {
                        return Ok(NativeCallResult::Return(last.clone()));
                    }
                }
                return Ok(NativeCallResult::Return(Value::Nil));
            }
        }
        Err("test.mock_last_call expects MockService".into())
    }))
}

pub fn create_test_projection_new_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("test.projection_new requires (tags, values)".into());
        }
        let tags = match unwrap_val(args[0].clone()) {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err("tags must be array".into()),
        };
        let values = match unwrap_val(args[1].clone()) {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err("values must be array".into()),
        };

        let mut proj = LustMap::default();
        proj.insert(string_key("__type"), Value::string("StackContextProjection"));
        for (i, tag) in tags.into_iter().enumerate() {
            if let Some(val) = values.get(i) {
                let tag_str = match &tag {
                    Value::Int(id) => id.to_string(),
                    Value::String(s) => s.as_str().to_string(),
                    other => format!("{:?}", other),
                };
                proj.insert(string_key(&tag_str), val.clone());
            }
        }
        Ok(NativeCallResult::Return(Value::map(proj)))
    }))
}

pub fn create_test_projection_get_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("test.projection_get requires (proj, tag)".into());
        }
        let proj = unwrap_val(args[0].clone());
        let tag_str = match unwrap_val(args[1].clone()) {
            Value::Int(id) => id.to_string(),
            Value::String(s) => s.as_str().to_string(),
            other => format!("{:?}", other),
        };

        if let Value::Map(m) = proj {
            let borrowed = m.borrow();
            let res = borrowed.get(&string_key(&tag_str)).cloned().unwrap_or_else(|| Value::enum_unit("Option", "None"));
            return Ok(NativeCallResult::Return(res));
        }
        Err("test.projection_get expects StackContextProjection".into())
    }))
}

pub fn create_test_resource_new_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let name = match args.first().cloned().map(unwrap_val) {
            Some(Value::String(s)) => (*s).clone(),
            _ => "resource".to_string(),
        };
        let mut map = LustMap::default();
        map.insert(string_key("__type"), Value::string("HostResource"));
        map.insert(string_key("name"), Value::string(name));
        map.insert(string_key("active"), Value::Bool(true));
        Ok(NativeCallResult::Return(Value::map(map)))
    }))
}

pub fn create_test_resource_is_active_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("test.resource_is_active requires resource".into());
        }
        let res = unwrap_val(args[0].clone());
        if let Value::Map(m) = res {
            let borrowed = m.borrow();
            let active = borrowed.get(&string_key("active")).cloned().unwrap_or(Value::Bool(false));
            return Ok(NativeCallResult::Return(active));
        }
        Err("test.resource_is_active expects HostResource".into())
    }))
}

pub fn create_test_scope_new_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let id = match args.first().cloned().map(unwrap_val) {
            Some(Value::Int(i)) => i,
            _ => 1,
        };
        let parent_id = args.get(1).cloned().map(unwrap_val).unwrap_or(Value::Nil);

        let mut map = LustMap::default();
        map.insert(string_key("__type"), Value::string("Scope"));
        map.insert(string_key("id"), Value::Int(id));
        map.insert(string_key("parent_id"), parent_id);
        map.insert(string_key("is_open"), Value::Bool(true));
        map.insert(string_key("teardown"), Value::string("Idle"));
        map.insert(string_key("admission"), Value::string("Open"));
        map.insert(string_key("finalizers"), Value::array(Vec::new()));
        map.insert(string_key("children"), Value::array(Vec::new()));
        map.insert(string_key("released"), Value::array(Vec::new()));
        Ok(NativeCallResult::Return(Value::map(map)))
    }))
}

pub fn create_test_scope_is_open_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("test.scope_is_open requires scope".into());
        }
        let scope = unwrap_val(args[0].clone());
        if let Value::Map(m) = scope {
            let borrowed = m.borrow();
            let open = borrowed.get(&string_key("is_open")).cloned().unwrap_or(Value::Bool(false));
            return Ok(NativeCallResult::Return(open));
        }
        Err("test.scope_is_open expects Scope".into())
    }))
}

pub fn create_test_scope_teardown_state_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("test.scope_teardown_state requires scope".into());
        }
        let scope = unwrap_val(args[0].clone());
        if let Value::Map(m) = scope {
            let borrowed = m.borrow();
            let state = borrowed.get(&string_key("teardown")).cloned().unwrap_or(Value::string("Idle"));
            return Ok(NativeCallResult::Return(state));
        }
        Err("test.scope_teardown_state expects Scope".into())
    }))
}

pub fn create_test_scope_admission_state_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("test.scope_admission_state requires scope".into());
        }
        let scope = unwrap_val(args[0].clone());
        if let Value::Map(m) = scope {
            let borrowed = m.borrow();
            let state = borrowed.get(&string_key("admission")).cloned().unwrap_or(Value::string("Open"));
            return Ok(NativeCallResult::Return(state));
        }
        Err("test.scope_admission_state expects Scope".into())
    }))
}

pub fn create_test_scope_register_finalizer_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("test.scope_register_finalizer requires (scope, finalizer)".into());
        }
        let scope = unwrap_val(args[0].clone());
        let fin = unwrap_val(args[1].clone());

        if let Value::Map(m) = scope {
            let borrowed = m.borrow();
            let admission = borrowed.get(&string_key("admission")).and_then(|v| v.as_string());
            if admission == Some("Closed") {
                return Ok(NativeCallResult::Return(Value::Bool(false)));
            }

            if let Some(Value::Array(arr)) = borrowed.get(&string_key("finalizers")) {
                arr.borrow_mut().push(fin);
                return Ok(NativeCallResult::Return(Value::Bool(true)));
            }
        }
        Err("test.scope_register_finalizer expects Scope".into())
    }))
}

pub fn create_test_scope_register_child_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("test.scope_register_child requires (scope, fiber_or_id)".into());
        }
        let scope = unwrap_val(args[0].clone());
        let child = unwrap_val(args[1].clone());

        let child_id = match child {
            Value::Int(id) => Value::Int(id),
            Value::Map(fm) => fm.borrow().get(&string_key("id")).cloned().unwrap_or(Value::Int(0)),
            _ => return Err("child must be fiber or fiber id".into()),
        };

        if let Value::Map(m) = scope {
            let borrowed = m.borrow();
            let admission = borrowed.get(&string_key("admission")).and_then(|v| v.as_string());
            if admission == Some("Closed") {
                return Ok(NativeCallResult::Return(Value::Bool(false)));
            }

            if let Some(Value::Array(arr)) = borrowed.get(&string_key("children")) {
                arr.borrow_mut().push(child_id);
                return Ok(NativeCallResult::Return(Value::Bool(true)));
            }
        }
        Err("test.scope_register_child expects Scope".into())
    }))
}

pub fn create_test_scope_begin_teardown_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("test.scope_begin_teardown requires scope".into());
        }
        let scope = unwrap_val(args[0].clone());
        if let Value::Map(m) = scope {
            let mut borrowed = m.borrow_mut();
            borrowed.insert(string_key("admission"), Value::string("Closed"));
            borrowed.insert(string_key("teardown"), Value::string("Cancelling"));
            borrowed.insert(string_key("is_open"), Value::Bool(false));

            let children = match borrowed.get(&string_key("children")) {
                Some(Value::Array(a)) => a.borrow().clone(),
                _ => Vec::new(),
            };
            return Ok(NativeCallResult::Return(Value::array(children)));
        }
        Err("test.scope_begin_teardown expects Scope".into())
    }))
}

pub fn create_test_scope_mark_joining_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("test.scope_mark_joining requires scope".into());
        }
        let scope = unwrap_val(args[0].clone());
        if let Value::Map(m) = scope {
            let mut borrowed = m.borrow_mut();
            borrowed.insert(string_key("teardown"), Value::string("Joining"));
            return Ok(NativeCallResult::Return(Value::Bool(true)));
        }
        Err("test.scope_mark_joining expects Scope".into())
    }))
}

pub fn create_test_scope_current_releases_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("test.scope_current_releases requires scope".into());
        }
        let scope = unwrap_val(args[0].clone());
        if let Value::Map(m) = scope {
            let borrowed = m.borrow();
            let releases = match borrowed.get(&string_key("released")) {
                Some(Value::Array(a)) => a.borrow().clone(),
                _ => Vec::new(),
            };
            return Ok(NativeCallResult::Return(Value::array(releases)));
        }
        Err("test.scope_current_releases expects Scope".into())
    }))
}

pub fn create_test_scope_finalize_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("test.scope_finalize requires scope".into());
        }
        let scope = unwrap_val(args[0].clone());
        if let Value::Map(m) = &scope {
            let mut borrowed = m.borrow_mut();
            borrowed.insert(string_key("teardown"), Value::string("Done"));

            let finalizers = match borrowed.get(&string_key("finalizers")) {
                Some(Value::Array(a)) => a.borrow().clone(),
                _ => Vec::new(),
            };
            drop(borrowed);

            let mut released = Vec::new();
            // LIFO order execution
            for fin in finalizers.into_iter().rev() {
                match fin {
                    Value::String(s) => {
                        released.push(Value::string(format!("release:{}", s)));
                    }
                    Value::Map(rm) => {
                        let mut mb = rm.borrow_mut();
                        mb.insert(string_key("active"), Value::Bool(false));
                        let name = mb.get(&string_key("name")).and_then(|v| v.as_string()).unwrap_or("resource");
                        released.push(Value::string(format!("release:{}", name)));
                    }
                    Value::NativeFunction(_) | Value::Closure(_) => {
                        let _ = invoke_callable(&fin, Vec::new());
                    }
                    _ => {}
                }
            }

            let mut borrowed = m.borrow_mut();
            borrowed.insert(string_key("released"), Value::array(released.clone()));

            return Ok(NativeCallResult::Return(Value::array(released)));
        }
        Err("test.scope_finalize expects Scope".into())
    }))
}

pub fn create_test_fiber_new_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        let id = match args.first().cloned().map(unwrap_val) {
            Some(Value::Int(i)) => i,
            _ => 101,
        };
        let scope_id = match args.get(1).cloned().map(unwrap_val) {
            Some(Value::Int(i)) => i,
            _ => 1,
        };

        let mut map = LustMap::default();
        map.insert(string_key("__type"), Value::string("FiberContext"));
        map.insert(string_key("id"), Value::Int(id));
        map.insert(string_key("scope_id"), Value::Int(scope_id));
        map.insert(string_key("state"), Value::string("Running"));
        map.insert(string_key("mask_depth"), Value::Int(0));
        map.insert(string_key("pending_interrupt"), Value::Nil);
        Ok(NativeCallResult::Return(Value::map(map)))
    }))
}

pub fn create_test_fiber_set_state_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("test.fiber_set_state requires (fiber, state)".into());
        }
        let fiber = unwrap_val(args[0].clone());
        let state = unwrap_val(args[1].clone());
        if let Value::Map(m) = fiber {
            let mut borrowed = m.borrow_mut();
            borrowed.insert(string_key("state"), state);
            return Ok(NativeCallResult::Return(Value::Nil));
        }
        Err("test.fiber_set_state expects FiberContext".into())
    }))
}

pub fn create_test_fiber_enter_mask_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("test.fiber_enter_mask requires fiber".into());
        }
        let fiber = unwrap_val(args[0].clone());
        if let Value::Map(m) = fiber {
            let mut borrowed = m.borrow_mut();
            let cur = match borrowed.get(&string_key("mask_depth")) {
                Some(Value::Int(d)) => *d,
                _ => 0,
            };
            borrowed.insert(string_key("mask_depth"), Value::Int(cur + 1));
            return Ok(NativeCallResult::Return(Value::Int(cur + 1)));
        }
        Err("test.fiber_enter_mask expects FiberContext".into())
    }))
}

pub fn create_test_fiber_exit_mask_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("test.fiber_exit_mask requires fiber".into());
        }
        let fiber = unwrap_val(args[0].clone());
        if let Value::Map(m) = fiber {
            let mut borrowed = m.borrow_mut();
            let cur = match borrowed.get(&string_key("mask_depth")) {
                Some(Value::Int(d)) => *d,
                _ => 0,
            };
            if cur > 0 {
                borrowed.insert(string_key("mask_depth"), Value::Int(cur - 1));
            }
            if cur.saturating_sub(1) == 0 {
                let pending = borrowed.get(&string_key("pending_interrupt")).cloned().unwrap_or(Value::Nil);
                if !matches!(pending, Value::Nil) {
                    borrowed.insert(string_key("pending_interrupt"), Value::Nil);
                    borrowed.insert(string_key("state"), Value::string("Interrupted"));
                    return Ok(NativeCallResult::Return(pending));
                }
            }
            return Ok(NativeCallResult::Return(Value::Nil));
        }
        Err("test.fiber_exit_mask expects FiberContext".into())
    }))
}

pub fn create_test_fiber_is_masked_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("test.fiber_is_masked requires fiber".into());
        }
        let fiber = unwrap_val(args[0].clone());
        if let Value::Map(m) = fiber {
            let borrowed = m.borrow();
            let d = match borrowed.get(&string_key("mask_depth")) {
                Some(Value::Int(d)) => *d,
                _ => 0,
            };
            return Ok(NativeCallResult::Return(Value::Bool(d > 0)));
        }
        Err("test.fiber_is_masked expects FiberContext".into())
    }))
}

pub fn create_test_fiber_request_interrupt_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.len() < 2 {
            return Err("test.fiber_request_interrupt requires (fiber, reason)".into());
        }
        let fiber = unwrap_val(args[0].clone());
        let reason = unwrap_val(args[1].clone());

        if let Value::Map(m) = fiber {
            let mut borrowed = m.borrow_mut();
            let d = match borrowed.get(&string_key("mask_depth")) {
                Some(Value::Int(d)) => *d,
                _ => 0,
            };
            if d > 0 {
                // Masked! Defer interrupt delivery
                borrowed.insert(string_key("pending_interrupt"), reason);
                return Ok(NativeCallResult::Return(Value::Bool(false)));
            } else {
                // Not masked! Deliver immediately
                borrowed.insert(string_key("state"), Value::string("Interrupted"));
                return Ok(NativeCallResult::Return(Value::Bool(true)));
            }
        }
        Err("test.fiber_request_interrupt expects FiberContext".into())
    }))
}

pub fn create_test_fiber_state_fn() -> Value {
    Value::NativeFunction(native_fn(|args: &[Value]| {
        if args.is_empty() {
            return Err("test.fiber_state requires fiber".into());
        }
        let fiber = unwrap_val(args[0].clone());
        if let Value::Map(m) = fiber {
            let borrowed = m.borrow();
            let state = borrowed.get(&string_key("state")).cloned().unwrap_or(Value::string("Running"));
            return Ok(NativeCallResult::Return(state));
        }
        Err("test.fiber_state expects FiberContext".into())
    }))
}

pub fn register_rx_test_builtins(vm: &mut VM) {
    let test_mod = create_test_module(vm);
    let rx_mod = create_rx_module(vm);

    // Merge into or register globals
    vm.register_native("rx", rx_mod.clone());
    vm.register_native("std.rx", rx_mod.clone());

    if let Value::Map(rx_m) = rx_mod {
        let borrowed = rx_m.borrow();
        for (k, v) in borrowed.iter() {
            if let Some(key_name) = k.to_value().as_string() {
                vm.register_native(&format!("rx.{}", key_name), v.clone());
            }
        }
    }

    if let Value::Map(t_m) = test_mod {
        let borrowed = t_m.borrow();
        for (k, v) in borrowed.iter() {
            if let Some(key_name) = k.to_value().as_string() {
                vm.register_native(&format!("test.{}", key_name), v.clone());
                vm.register_native(&format!("std.test.{}", key_name), v.clone());
            }
        }
    }
}

#![cfg(feature = "std")]

use lust::bytecode::Value;
use lust::mir::runtime_helpers::*;
use std::time::{Duration, Instant};

#[test]
fn test_ng_w15_3_legacy_map_overwrites_do_not_charge_again() {
    let ctx = MirContext::new();
    let _guard = ctx.enter();
    let map = lust_mir_alloc_map();
    let key = ctx.intern_string("key");
    let before = ctx.memory_used();
    for value in 100..200 {
        lust_mir_map_set(map, key, value);
    }
    assert_eq!(ctx.memory_used() - before, 32);
}

#[test]
fn test_ng_w15_3_tostring_int_caches_through_1023() {
    let ctx = MirContext::new();
    let _guard = ctx.enter();
    let first = lust_mir_tostring_int(1023);
    assert_eq!(ctx.get_handle(first).unwrap().as_string(), Some("1023"));
    assert_eq!(lust_mir_tostring_int(1023), first);
}

#[test]
fn test_ng_w15_3_map_int_key_1_does_not_collide_with_map_handle_1() {
    let ctx = MirContext::new();
    let _guard = ctx.enter();
    let map = lust_mir_alloc_map();
    assert_eq!(map, 1);
    let before = ctx.memory_used();
    ctx.set_max_memory_limit(before + 32);
    assert_eq!(lust_mir_map_set_typed(0, map, 1, 42, 0, 0), MIR_STATUS_OK);
    assert_eq!(lust_mir_map_get_int_unwrap(0, map, 1, 0, i64::MIN), 42);
    for value in 0..100_000 {
        assert_eq!(
            lust_mir_map_set_typed(0, map, 1, value, 0, 0),
            MIR_STATUS_OK
        );
    }
    assert_eq!(lust_mir_map_get_int_unwrap(0, map, 1, 0, i64::MIN), 99_999);
    assert_eq!(ctx.handle_count(), 1);
    assert_eq!(ctx.memory_used() - before, 32);

    assert_eq!(
        lust_mir_map_set_typed(0, map, 2, 100, 0, 0),
        MIR_STATUS_ERR_OOM
    );
    assert_eq!(lust_mir_map_has_typed(0, map, 2, 0), 0);
    assert_eq!(lust_mir_map_len(map), 1);
    assert_eq!(ctx.memory_used() - before, 32);

    ctx.set_status(MIR_STATUS_OK);
    assert_eq!(lust_mir_map_get_int_unwrap(0, map, 2, 0, -19), -19);
    assert_eq!(ctx.last_status(), MIR_STATUS_OK);
    assert_eq!(lust_mir_map_get_int_unwrap(0, map, 2, 0, i64::MIN), 0);
    assert_eq!(ctx.last_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);
}

#[test]
fn test_ng_w15_3_map_tags_select_key_and_payload_types() {
    let ctx = MirContext::new();
    let _guard = ctx.enter();
    let map = lust_mir_alloc_map();
    let key = ctx.intern_string("request");
    let float = (-12.5_f64).to_bits() as i64;
    assert_eq!(
        lust_mir_map_set_typed(0, map, key, map, 3, 0),
        MIR_STATUS_OK
    );
    assert_eq!(lust_mir_map_get_int_unwrap(0, map, key, 3, i64::MIN), map);
    assert_eq!(
        lust_mir_map_set_typed(0, map, 1, float, 1, 2),
        MIR_STATUS_OK
    );
    assert_eq!(
        lust_mir_map_set_typed(0, map, float, key, 2, 3),
        MIR_STATUS_OK
    );
    assert_eq!(lust_mir_map_has_typed(0, map, 1, 0), 0);
    assert_eq!(lust_mir_map_has_typed(0, map, key, 3), 1);

    let option = ctx
        .get_handle(lust_mir_map_get_typed(0, map, 1, 1))
        .unwrap();
    assert_eq!(option.as_enum().unwrap().2.unwrap(), &[Value::Float(-12.5)]);
    let option = ctx
        .get_handle(lust_mir_map_get_typed(0, map, float, 2))
        .unwrap();
    assert_eq!(
        option.as_enum().unwrap().2.unwrap(),
        &[Value::string("request")]
    );

    assert_eq!(
        lust_mir_map_set_typed(0, map, key, i64::MIN, 3, 3),
        MIR_STATUS_ERR_OUT_OF_BOUNDS
    );
    assert_eq!(lust_mir_map_get_int_unwrap(0, map, key, 3, i64::MIN), map);
}

#[test]
fn test_ng_w15_3_string_sub_byte_and_split_semantics() {
    let ctx = MirContext::new();
    let _guard = ctx.enter();
    let tokens = ctx.intern_string("abc123xyz");
    assert_eq!(
        ctx.get_handle(lust_mir_string_sub(0, tokens, -6, -4))
            .unwrap()
            .as_string(),
        Some("123"),
    );
    assert_eq!(
        ctx.get_handle(lust_mir_string_sub(0, tokens, i64::MIN, i64::MAX))
            .unwrap()
            .as_string(),
        Some("abc123xyz"),
    );
    assert_eq!(
        ctx.get_handle(lust_mir_string_sub(0, tokens, 10, 20))
            .unwrap()
            .as_string(),
        Some(""),
    );
    assert_eq!(
        ctx.get_handle(lust_mir_string_byte(0, tokens, 4, 6))
            .unwrap()
            .as_array()
            .unwrap(),
        [Value::Int(49), Value::Int(50), Value::Int(51)],
    );
    assert_eq!(lust_mir_string_byte_at(0, tokens, -1), i64::from(b'z'));
    assert_eq!(
        ctx.get_handle(lust_mir_string_byte(0, tokens, 10, 12))
            .unwrap()
            .as_array()
            .unwrap(),
        [Value::Nil],
    );

    let utf8 = ctx.intern_string("é");
    assert_eq!(lust_mir_string_byte_at(0, utf8, 1), 0xc3);
    assert_eq!(
        ctx.get_handle(lust_mir_string_sub(0, utf8, 2, 2))
            .unwrap()
            .as_string(),
        Some("�")
    );
    assert_eq!(lust_mir_string_byte_at(0, tokens, 10), 0);
    assert_eq!(ctx.last_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);
    ctx.set_status(MIR_STATUS_OK);

    let raw = ctx.intern_string("id=42;temp=23;");
    let fields = ctx
        .get_handle(lust_mir_string_split(0, raw, ctx.intern_string(";")))
        .unwrap();
    assert_eq!(
        fields.as_array().unwrap(),
        [
            Value::string("id=42"),
            Value::string("temp=23"),
            Value::string("")
        ],
    );
    let result = ctx
        .get_handle(lust_mir_string_split(0, utf8, ctx.intern_string("")))
        .unwrap();
    assert_eq!(result.as_array().unwrap(), [Value::string("é")]);

    let before = ctx.memory_used();
    let handles = ctx.handle_count();
    ctx.set_max_memory_limit(before);
    assert_eq!(lust_mir_string_sub(0, tokens, 1, 1), 0);
    assert_eq!(ctx.last_status(), MIR_STATUS_ERR_OOM);
    assert_eq!(ctx.memory_used(), before);
    assert_eq!(ctx.handle_count(), handles);
}

#[test]
fn test_ng_w15_3_gsub_template_and_find_captures() {
    let ctx = MirContext::new();
    let _guard = ctx.enter();
    let text = ctx.intern_string("req_42.req_73");
    let pattern = ctx.intern_string("req_(%d+)");
    let replacement = ctx.intern_string("[%1:%%:%0]");
    let result = ctx
        .get_handle(lust_mir_string_gsub(0, text, pattern, replacement))
        .unwrap();
    assert_eq!(
        result.as_array().unwrap(),
        [Value::string("[42:%:req_42].[73:%:req_73]"), Value::Int(2)],
    );
    let found = ctx
        .get_handle(lust_mir_string_find(0, text, pattern, -6, 0))
        .unwrap();
    assert_eq!(
        found.as_array().unwrap(),
        [Value::Int(8), Value::Int(13), Value::string("73")]
    );
    let plain = ctx
        .get_handle(lust_mir_string_find(0, text, ctx.intern_string("."), 1, 1))
        .unwrap();
    assert_eq!(plain.as_array().unwrap(), [Value::Int(7), Value::Int(7)]);
    let missing = ctx
        .get_handle(lust_mir_string_find(0, text, pattern, 1, 1))
        .unwrap();
    assert_eq!(missing.as_array().unwrap(), [Value::Nil]);

    assert_eq!(
        lust_mir_string_gsub(0, text, ctx.intern_string("["), replacement),
        0
    );
    assert_eq!(ctx.last_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);
}

#[test]
fn test_ng_w15_3_regex_cache_20k_gsub_under_25ms() {
    let ctx = MirContext::new();
    let _guard = ctx.enter();
    let pattern = ctx.register_handle(Value::string("[0-9]+"));
    let replacement = ctx.register_handle(Value::string("#"));
    let subjects: Vec<_> = (0..20_000)
        .map(|i| {
            ctx.register_handle(Value::string(format!(
                "user_{}_req_{}",
                100 + i % 900,
                i * 7
            )))
        })
        .collect();
    let warmup = lust_mir_string_gsub(0, subjects[0], pattern, replacement);
    assert_eq!(
        ctx.get_handle(warmup).unwrap().as_array().unwrap(),
        [Value::string("user_#_req_#"), Value::Int(2)],
    );

    let started = Instant::now();
    let mut checksum = 0;
    for subject in subjects {
        let result = lust_mir_string_gsub(0, subject, pattern, replacement);
        let Value::Array(values) = ctx.get_handle(result).unwrap() else {
            panic!("expected substitution result array");
        };
        checksum += values.borrow()[0].as_string().unwrap().len();
    }
    let elapsed = started.elapsed();
    assert_eq!(checksum, 240_000);
    eprintln!("20,000 cached gsub calls took {elapsed:?}");
    // Debug builds also verify the checksum, but time the gate with optimization enabled.
    if !cfg!(debug_assertions) {
        assert!(
            elapsed < Duration::from_millis(25),
            "20,000 cached gsub calls took {elapsed:?}"
        );
    }
}

#[test]
fn test_ng_w15_3_enum_payload_arity_and_raw_integer_fields() {
    let ctx = MirContext::new();
    let _guard = ctx.enter();
    let occupied_handle = ctx.register_handle(Value::string("existing handle"));
    let unit = lust_mir_alloc_enum2(0, 3, 0, 0, 0);
    let single = lust_mir_alloc_enum2(0, 4, occupied_handle, 0, 1);
    let pair = lust_mir_alloc_enum2(0, 5, 0, i64::MIN, 2);
    assert_eq!(lust_mir_enum_tag(unit), 3);
    assert!(ctx.get_handle(unit).unwrap().as_enum().unwrap().2.is_none());
    assert_eq!(lust_mir_enum_tag(single), 4);
    assert_eq!(lust_mir_enum_payload(single), occupied_handle);
    assert_eq!(lust_mir_enum_tag(pair), 5);
    assert_eq!(lust_mir_enum_payload_at(0, pair, 0), 0);
    assert_eq!(lust_mir_enum_payload_at(0, pair, 1), i64::MIN);
    assert_eq!(lust_mir_enum_payload_at(0, pair, 2), 0);
    assert_eq!(
        ctx.get_handle(pair).unwrap().as_enum().unwrap().2.unwrap(),
        &[Value::Int(0), Value::Int(i64::MIN)],
    );
    assert_eq!(lust_mir_alloc_enum2(0, 6, 0, 0, 3), 0);
    assert_eq!(ctx.last_status(), MIR_STATUS_ERR_OUT_OF_BOUNDS);
}

#[test]
fn test_ng_w15_3_tostring_integer_boundaries_and_budget() {
    let ctx = MirContext::new();
    let _guard = ctx.enter();
    let mut vm = lust::vm::VM::new();
    for value in [
        i64::MIN,
        -1024,
        -1,
        0,
        9,
        10,
        99,
        100,
        999,
        1000,
        1023,
        1024,
        i64::MAX,
    ] {
        let text = value.to_string();
        let before = ctx.memory_used();
        let handle = lust_mir_tostring_int(value);
        assert_eq!(
            ctx.get_handle(handle).unwrap().as_string(),
            Some(text.as_str())
        );
        assert_eq!(ctx.memory_used() - before, text.len() as i64);
        assert_eq!(
            vm.call("tostring", vec![Value::Int(value)])
                .unwrap()
                .as_string(),
            Some(text.as_str())
        );
    }
    let first = vm
        .call("tostring", vec![Value::Int(1023)])
        .unwrap()
        .as_string_rc()
        .unwrap();
    let second = vm
        .call("tostring", vec![Value::Int(1023)])
        .unwrap()
        .as_string_rc()
        .unwrap();
    assert!(std::rc::Rc::ptr_eq(&first, &second));
    let before = ctx.memory_used();
    ctx.set_max_memory_limit(before + 3);
    assert_eq!(lust_mir_tostring_int(1023), 0);
    assert_eq!(ctx.last_status(), MIR_STATUS_ERR_OOM);
    assert_eq!(ctx.memory_used(), before);
}

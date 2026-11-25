use lust::data::budget::BudgetTracker;
use lust::data::json::encode::{to_string_with_limits, to_string_pretty_with_limits};
use lust::data::json::events::{
    from_events, from_events_with_limits, stream_events, stream_events_with_limits, to_events,
    JsonError, JsonEvent,
};
use lust::data::json::{self, to_string, to_string_pretty};
use lust::data::number::{DataError, DateTimeUtc, Decimal128, ExactNumber, ObjectId};
use lust::data::owner::{LeaseError, OwnerId, OwnerRegistry};
use lust::data::{DataValue, DataView, Document};
use lust::number::{
    float_abs, float_acos, float_asin, float_atan, float_atan2, float_ceil, float_clamp,
    float_cos, float_exp, float_floor, float_from_int, float_is_finite, float_is_infinite,
    float_is_nan, float_is_sign_negative, float_is_sign_positive, float_ln, float_max, float_min,
    float_pow, float_round, float_sin, float_sqrt, float_tan, float_to_hash_bits, int_abs,
    int_clamp, int_from_float, int_from_usize, int_zero, is_safe_integer_f64, parse_float,
    safe_float_to_int, safe_int_to_float_exact, LustFloat, NumericType,
    MAX_SAFE_INTEGER_F64, MIN_SAFE_INTEGER_F64,
};

// =========================================================================
// 1. JSON ENCODING & SCALAR ROUNDTRIP ORACLE TESTS
// =========================================================================

#[test]
fn test_json_scalar_roundtrip_primitives() {
    let test_cases = vec![
        ("null", DataValue::Null),
        ("true", DataValue::Bool(true)),
        ("false", DataValue::Bool(false)),
        ("0", DataValue::Number(ExactNumber::Int(0))),
        ("42", DataValue::Number(ExactNumber::Int(42))),
        ("-42", DataValue::Number(ExactNumber::Int(-42))),
        ("9223372036854775807", DataValue::Number(ExactNumber::Int(i64::MAX))),
        ("-9223372036854775808", DataValue::Number(ExactNumber::Int(i64::MIN))),
        ("18446744073709551615", DataValue::Number(ExactNumber::UInt(u64::MAX))),
        ("1.0", DataValue::Number(ExactNumber::Float(1.0))),
        ("-1.5", DataValue::Number(ExactNumber::Float(-1.5))),
        ("3.141592653589793", DataValue::Number(ExactNumber::Float(3.141592653589793))),
        ("\"\"", DataValue::String(String::new())),
        ("\"hello world\"", DataValue::String("hello world".to_string())),
        (
            "\"escapes: \\\" \\\\ \\b \\f \\n \\r \\t\"",
            DataValue::String("escapes: \" \\ \x08 \x0c \n \r \t".to_string()),
        ),
        (
            "\"unicode: \u{041f}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442} \u{4e16}\u{754c} \u{1f600}\"",
            DataValue::String("unicode: Привет 世界 😀".to_string()),
        ),
    ];

    for (name, expected_val) in test_cases {
        let encoded = to_string(&expected_val)
            .unwrap_or_else(|e| panic!("Encoding failed for input '{}': {:?}", name, e));
        let decoded = json::from_str(&encoded)
            .unwrap_or_else(|e| panic!("Decoding failed for input '{}', json '{}': {:?}", name, encoded, e));

        assert_eq!(
            decoded, expected_val,
            "Roundtrip mismatch for input '{}': encoded='{}', decoded='{:?}', expected='{:?}'",
            name, encoded, decoded, expected_val
        );
    }
}

#[test]
fn test_json_encode_control_characters_escape() {
    let raw = "\x00\x01\x1f";
    let val = DataValue::String(raw.to_string());
    let encoded = to_string(&val).expect("Failed to encode control chars");
    assert_eq!(encoded, "\"\\u0000\\u0001\\u001f\"", "Escaped control chars mismatch for input");

    let decoded = json::from_str(&encoded).expect("Failed to decode escaped control chars");
    assert_eq!(decoded, val, "Roundtrip control characters mismatch for input");
}

#[test]
fn test_json_roundtrip_empty_and_nested_structures() {
    // Empty array
    let empty_arr = DataValue::Array(Vec::new());
    let enc_arr = to_string(&empty_arr).expect("Failed to encode empty array");
    assert_eq!(enc_arr, "[]", "Empty array encoding mismatch");
    assert_eq!(json::from_str(&enc_arr).expect("decode empty array"), empty_arr);

    // Empty document
    let empty_doc = DataValue::Document(Document::new());
    let enc_doc = to_string(&empty_doc).expect("Failed to encode empty doc");
    assert_eq!(enc_doc, "{}", "Empty doc encoding mismatch");
    assert_eq!(json::from_str(&enc_doc).expect("decode empty doc"), empty_doc);

    // Complex nested structure
    let mut inner_doc = Document::new();
    inner_doc.try_insert("count".to_string(), DataValue::Number(ExactNumber::Int(100))).unwrap();
    inner_doc.try_insert("active".to_string(), DataValue::Bool(true)).unwrap();

    let mut root_doc = Document::new();
    root_doc.try_insert("items".to_string(), DataValue::Array(vec![
        DataValue::Number(ExactNumber::Int(1)),
        DataValue::Number(ExactNumber::Int(2)),
        DataValue::String("three".to_string()),
    ])).unwrap();
    root_doc.try_insert("meta".to_string(), DataValue::Document(inner_doc)).unwrap();

    let complex = DataValue::Document(root_doc);
    let encoded = to_string(&complex).expect("Failed to encode complex document");
    let decoded = json::from_str(&encoded).expect("Failed to decode complex document");
    assert_eq!(decoded, complex, "Complex document roundtrip mismatch for input");
}

#[test]
fn test_json_encode_pretty_multiline_structure() {
    let mut doc = Document::new();
    doc.try_insert("key1".to_string(), DataValue::String("val1".to_string())).unwrap();
    doc.try_insert("nested".to_string(), DataValue::Array(vec![
        DataValue::Number(ExactNumber::Int(1)),
        DataValue::Number(ExactNumber::Int(2)),
    ])).unwrap();

    let val = DataValue::Document(doc);
    let pretty = to_string_pretty(&val).expect("to_string_pretty failed");
    assert!(pretty.contains('\n'), "Pretty json must contain newlines: {}", pretty);
    assert!(pretty.contains("  \"key1\": \"val1\""), "Pretty json must have indented fields: {}", pretty);

    // Oracle: scalar decode of pretty JSON must exactly equal original DataValue
    let decoded = json::from_str(&pretty).expect("Failed to decode pretty JSON");
    assert_eq!(decoded, val, "Pretty JSON decode did not match original DataValue");

    // Test to_string_pretty_with_limits with custom indent
    let pretty_4 = to_string_pretty_with_limits(&val, 10, 4).expect("to_string_pretty_with_limits failed");
    assert!(pretty_4.contains("    \"key1\": \"val1\""), "Custom indent step 4 missing: {}", pretty_4);
    assert_eq!(
        to_string_pretty_with_limits(&val, 1, 2),
        Err(JsonError::MaxDepthExceeded(1)),
        "Pretty with max depth 1 must fail on nested value"
    );
}

#[test]
fn test_json_encode_bson_types_interop() {
    // ObjectId
    let oid = ObjectId::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let val_oid = DataValue::ObjectId(oid);
    let enc_oid = to_string(&val_oid).expect("encode ObjectId");
    assert_eq!(enc_oid, "{\"$oid\":\"0102030405060708090a0b0c\"}", "ObjectId JSON mismatch");

    // DateTimeUtc
    let dt = DateTimeUtc(1609459200000);
    let val_dt = DataValue::DateTimeUtc(dt);
    let enc_dt = to_string(&val_dt).expect("encode DateTimeUtc");
    assert_eq!(enc_dt, "{\"$date\":1609459200000}", "DateTimeUtc JSON mismatch");

    // Binary encoded as numeric array
    let bytes = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let val_bin = DataValue::Binary(bytes);
    let enc_bin = to_string(&val_bin).expect("encode Binary");
    assert_eq!(enc_bin, "[222,173,190,239]", "Binary JSON mismatch");

    // Decimal128
    let dec = Decimal128([0x12, 0x34, 0x56, 0x78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let val_dec = DataValue::Number(ExactNumber::Decimal128(dec));
    let enc_dec = to_string(&val_dec).expect("encode Decimal128");
    assert!(enc_dec.starts_with('"') && enc_dec.ends_with('"'), "Decimal128 must be quoted: {}", enc_dec);
}

// =========================================================================
// 2. HOSTILE & BOUNDARY REJECTIONS IN ENCODER
// =========================================================================

#[test]
fn test_json_encode_rejects_non_finite_floats() {
    let hostile_inputs: Vec<(&'static str, f64)> = vec![
        ("NaN", f64::NAN),
        ("Infinity", f64::INFINITY),
        ("-Infinity", f64::NEG_INFINITY),
    ];

    for (name, val) in hostile_inputs {
        let dv = DataValue::Number(ExactNumber::Float(val));
        let res = to_string(&dv);
        assert_eq!(
            res,
            Err(JsonError::InvalidNumber(0)),
            "Non-finite float '{}' must be rejected with InvalidNumber(0)",
            name
        );

        // Hostile float nested inside array
        let arr = DataValue::Array(vec![DataValue::Number(ExactNumber::Int(1)), dv.clone()]);
        assert_eq!(
            to_string(&arr),
            Err(JsonError::InvalidNumber(0)),
            "Nested non-finite float '{}' inside array must be rejected",
            name
        );

        // Hostile float nested inside document
        let mut doc = Document::new();
        doc.try_insert("bad".to_string(), dv).unwrap();
        assert_eq!(
            to_string(&DataValue::Document(doc)),
            Err(JsonError::InvalidNumber(0)),
            "Nested non-finite float '{}' inside doc must be rejected",
            name
        );
    }
}

#[test]
fn test_json_encode_depth_limit_boundary() {
    // Build nested array of depth 6: [[[[[[1]]]]]]
    let mut val = DataValue::Number(ExactNumber::Int(1));
    for _ in 0..6 {
        val = DataValue::Array(vec![val]);
    }

    // Limit 10 -> OK
    let ok_res = to_string_with_limits(&val, 10);
    assert!(ok_res.is_ok(), "Encoding with limit 10 should succeed, input depth 6");

    // Limit 4 -> Must fail with MaxDepthExceeded(4)
    let err_res = to_string_with_limits(&val, 4);
    assert_eq!(
        err_res,
        Err(JsonError::MaxDepthExceeded(4)),
        "Encoding with limit 4 must fail with MaxDepthExceeded(4), input depth 6"
    );
}

// =========================================================================
// 3. JSON EVENT STREAMING ORACLE TESTS
// =========================================================================

#[test]
fn test_json_event_streaming_token_stream_parity() {
    let json_inputs = vec![
        "null",
        "true",
        "false",
        "12345",
        "\"a string\"",
        "[]",
        "{}",
        "[1, \"two\", true, null]",
        "{\"alpha\": 1, \"beta\": [true, false], \"nested\": {\"inner\": 42}}",
    ];

    for json_str in json_inputs {
        // Independent Oracle: Parse using scalar parser
        let scalar_val = json::from_str(json_str)
            .unwrap_or_else(|e| panic!("Scalar parse failed on '{}': {:?}", json_str, e));

        // Stream events directly from raw bytes
        let streamed_events = stream_events(json_str.as_bytes())
            .unwrap_or_else(|e| panic!("Event stream failed on '{}': {:?}", json_str, e));

        // Generate events from the DataValue
        let val_events = to_events(&scalar_val);

        // Oracle 1: Direct streaming matches DataValue to_events
        assert_eq!(
            streamed_events, val_events,
            "Streamed events do not match value events for input '{}'",
            json_str
        );

        // Oracle 2: Reconstructing DataValue from events yields exact scalar_val
        let reconstructed = from_events(streamed_events)
            .unwrap_or_else(|e| panic!("from_events failed on '{}': {:?}", json_str, e));
        assert_eq!(
            reconstructed, scalar_val,
            "Reconstructed DataValue does not match scalar_val for input '{}'",
            json_str
        );
    }
}

#[test]
fn test_json_event_streaming_hostile_syntax_errors() {
    // Unexpected EOF
    assert_eq!(
        stream_events(b"{\"key\": "),
        Err(JsonError::UnexpectedEof),
        "Incomplete JSON must return UnexpectedEof"
    );

    // Trailing comma in array
    assert!(
        matches!(stream_events(b"[1, 2,]"), Err(JsonError::TrailingComma(_))),
        "Trailing comma in array must return TrailingComma"
    );

    // Trailing comma in object
    assert!(
        matches!(stream_events(b"{\"k\": 1,}"), Err(JsonError::TrailingComma(_))),
        "Trailing comma in object must return TrailingComma"
    );

    // Trailing bytes
    assert!(
        matches!(stream_events(b"{\"k\": 1} extra"), Err(JsonError::TrailingBytes(_))),
        "Trailing bytes after root must return TrailingBytes"
    );

    // Depth limit exceeded in stream
    assert_eq!(
        stream_events_with_limits(b"[[[[ 1 ]]]]", 2, 1024),
        Err(JsonError::MaxDepthExceeded(2)),
        "Depth limit in stream must return MaxDepthExceeded"
    );

    // Max bytes exceeded in stream
    assert_eq!(
        stream_events_with_limits(b"[1, 2, 3]", 10, 3),
        Err(JsonError::MaxBytesExceeded(3)),
        "Max bytes in stream must return MaxBytesExceeded"
    );

    // Duplicate key in from_events
    let dup_events = vec![
        JsonEvent::StartObject,
        JsonEvent::Key("dup".to_string()),
        JsonEvent::Scalar(DataValue::Number(ExactNumber::Int(1))),
        JsonEvent::Key("dup".to_string()),
        JsonEvent::Scalar(DataValue::Number(ExactNumber::Int(2))),
        JsonEvent::EndObject,
    ];
    assert_eq!(
        from_events(dup_events),
        Err(JsonError::DuplicateKey("dup".to_string(), 0)),
        "Duplicate key in event stream must return DuplicateKey"
    );

    // Depth limit exceeded in from_events
    let deep_events = vec![
        JsonEvent::StartArray,
        JsonEvent::StartArray,
        JsonEvent::StartArray,
        JsonEvent::Scalar(DataValue::Bool(true)),
        JsonEvent::EndArray,
        JsonEvent::EndArray,
        JsonEvent::EndArray,
    ];
    assert_eq!(
        from_events_with_limits(deep_events, 2),
        Err(JsonError::MaxDepthExceeded(2)),
        "Depth limit in from_events must return MaxDepthExceeded"
    );
}

#[test]
fn test_json_event_streaming_utf16_surrogates() {
    // 1. Valid surrogate pair: \uD83D\uDE00 -> 😀
    let valid_surrogate = br#""\uD83D\uDE00""#;
    let events = stream_events(valid_surrogate).expect("valid surrogate pair must decode to emoji");
    assert_eq!(
        events,
        vec![JsonEvent::Scalar(DataValue::String("😀".to_string()))],
        "Surrogate pair must stream to emoji string"
    );

    // 2. Lone high surrogate followed by quote
    let lone_high = br#""\uD800""#;
    assert!(
        matches!(stream_events(lone_high), Err(JsonError::InvalidUnicode(_))),
        "Lone high surrogate must return InvalidUnicode"
    );

    // 3. Lone high surrogate followed by letters
    let high_letters = br#""\uD800abc""#;
    assert!(
        matches!(stream_events(high_letters), Err(JsonError::InvalidUnicode(_))),
        "High surrogate followed by letters must return InvalidUnicode"
    );

    // 4. High surrogate followed by non-low surrogate \u1234
    let high_non_low = br#""\uD800\u1234""#;
    assert!(
        matches!(stream_events(high_non_low), Err(JsonError::InvalidUnicode(_))),
        "High surrogate followed by non-low surrogate must return InvalidUnicode"
    );

    // 5. Lone low surrogate \uDC00
    let lone_low = br#""\uDC00""#;
    assert!(
        matches!(stream_events(lone_low), Err(JsonError::InvalidUnicode(_))),
        "Lone low surrogate must return InvalidUnicode"
    );

    // 6. High surrogate followed by premature EOF
    let high_eof = br#""\uD800"#;
    assert!(
        matches!(
            stream_events(high_eof),
            Err(JsonError::UnexpectedEof) | Err(JsonError::InvalidUnicode(_))
        ),
        "High surrogate at premature EOF must error"
    );
}

#[test]
fn test_json_error_display_coverage() {
    let errors: Vec<(&'static str, JsonError)> = vec![
        ("Unexpected end of JSON input", JsonError::UnexpectedEof),
        ("Unexpected character 'x' at offset 5", JsonError::UnexpectedChar('x', 5)),
        ("Duplicate key 'id' rejected at offset 12", JsonError::DuplicateKey("id".to_string(), 12)),
        ("Max nesting depth of 128 exceeded", JsonError::MaxDepthExceeded(128)),
        ("Max document bytes of 4096 exceeded", JsonError::MaxBytesExceeded(4096)),
        ("Trailing unconsumed bytes at offset 18", JsonError::TrailingBytes(18)),
        ("Trailing comma forbidden in strict mode at offset 24", JsonError::TrailingComma(24)),
        ("Invalid escape sequence at offset 7", JsonError::InvalidEscape(7)),
        ("Invalid unicode escape at offset 9", JsonError::InvalidUnicode(9)),
        ("Invalid number syntax at offset 3", JsonError::InvalidNumber(3)),
        ("I/O error: disk full", JsonError::Io("disk full".to_string())),
    ];

    for (expected_msg, err) in errors {
        let msg = format!("{}", err);
        assert_eq!(msg, expected_msg, "Error display message mismatch for {:?}", err);
    }
}

// =========================================================================
// 4. DATAVIEW LIFECYCLE, TENANT ISOLATION & QUOTA TESTS
// =========================================================================

#[test]
fn test_dataview_access_validation_and_lifecycle_oracle() {
    let mut registry = OwnerRegistry::new(1);
    let tenant_id = 100;
    let owner = registry.allocate(tenant_id);

    let sample_str = "secure confidential payload";
    let view = DataView::from_str(sample_str, owner);

    // Oracle 1: Valid tenant on active lease succeeds
    let borrowed = view.as_str_checked(&registry, tenant_id);
    assert_eq!(borrowed, Ok(sample_str), "Valid tenant read must succeed");
    assert_eq!(view.as_bytes_checked(&registry, tenant_id), Ok(sample_str.as_bytes()));

    // Finding 1: slice_checked on valid lease
    let sliced = view.slice_checked(0, 6, &registry, tenant_id).expect("valid slice_checked");
    assert_eq!(sliced.as_str(), Some("secure"));

    // Finding 1: slice_checked on tenant mismatch
    let bad_tenant = 999;
    assert_eq!(
        view.slice_checked(0, 6, &registry, bad_tenant),
        Err(DataError::new("Lease check failed during slice")),
        "slice_checked must reject mismatched tenant"
    );

    // Finding 1: to_owned_checked_with_budget on valid lease + sufficient budget
    let mut ok_budget = BudgetTracker::new(100, 1000, 10);
    let res = view.to_owned_checked_with_budget(&registry, tenant_id, &mut ok_budget);
    assert_eq!(res, Ok(DataValue::String(sample_str.to_string())), "Valid to_owned_checked_with_budget");

    // Finding 1: to_owned_checked_with_budget on lease failure (tenant mismatch)
    let mut ok_budget2 = BudgetTracker::new(100, 1000, 10);
    let res_lease_fail = view.to_owned_checked_with_budget(&registry, bad_tenant, &mut ok_budget2);
    assert_eq!(
        res_lease_fail,
        Err(DataError::new("Lease check failed")),
        "to_owned_checked_with_budget must fail on tenant mismatch"
    );

    // Finding 1: to_owned_checked_with_budget on budget exhaustion
    let mut tiny_budget = BudgetTracker::new(5, 1000, 10);
    let res_budget_fail = view.to_owned_checked_with_budget(&registry, tenant_id, &mut tiny_budget);
    assert_eq!(
        res_budget_fail,
        Err(DataError::new("Budget check failed")),
        "to_owned_checked_with_budget must fail on budget exhaustion"
    );

    // Finding 1: Type mismatches on checked dereferences
    let bin_view = DataView::from_binary(b"raw data", owner);
    assert_eq!(
        bin_view.as_str_checked(&registry, tenant_id),
        Err(LeaseError::InvalidState("View is not a string")),
        "as_str_checked on Binary must return 'View is not a string'"
    );

    let num_view = DataView::from_number(ExactNumber::Int(42));
    assert_eq!(
        num_view.as_bytes_checked(&registry, tenant_id),
        Err(LeaseError::InvalidState("View is not a binary or string buffer")),
        "as_bytes_checked on Number must return 'View is not a binary or string buffer'"
    );

    let null_view = DataView::null();
    assert_eq!(
        null_view.as_bytes_checked(&registry, tenant_id),
        Err(LeaseError::InvalidState("View is not a binary or string buffer")),
        "as_bytes_checked on Null must return 'View is not a binary or string buffer'"
    );

    let bool_view = DataView::from_bool(true);
    assert_eq!(
        bool_view.as_str_checked(&registry, tenant_id),
        Err(LeaseError::InvalidState("View is not a string")),
        "as_str_checked on Bool must return 'View is not a string'"
    );

    // Oracle 2: Tenant mismatch fails on as_str_checked
    assert_eq!(
        view.as_str_checked(&registry, bad_tenant),
        Err(LeaseError::TenantMismatch { expected: bad_tenant, actual: tenant_id }),
        "Foreign tenant access must be rejected"
    );

    // Oracle 3: Step 1 of lifecycle: cancel lease
    registry.cancel(owner).expect("cancel lease");
    assert_eq!(
        view.as_str_checked(&registry, tenant_id),
        Err(LeaseError::InvalidState("Slot is not active")),
        "Access to cancelled lease must be rejected"
    );
    assert_eq!(
        view.to_owned_checked(&registry, tenant_id),
        Err(LeaseError::InvalidState("Slot is not active")),
        "Materialization of cancelled lease must be rejected"
    );
    assert_eq!(
        view.slice_checked(0, 6, &registry, tenant_id),
        Err(DataError::new("Lease check failed during slice")),
        "slice_checked must reject cancelled lease"
    );

    // Oracle 4: Step 2: acknowledge child join
    registry.acknowledge_join(owner).expect("acknowledge join");
    assert_eq!(
        view.as_str_checked(&registry, tenant_id),
        Err(LeaseError::InvalidState("Slot is not active")),
        "Access to joined lease must be rejected"
    );

    // Oracle 5: Step 3: release lease
    registry.release(owner).expect("release lease");
    assert_eq!(
        view.as_str_checked(&registry, tenant_id),
        Err(LeaseError::InvalidState("Slot is not active")),
        "Access to released lease must be rejected"
    );

    // Oracle 6: Reallocated slot has incremented generation -> StaleGeneration
    let new_owner = registry.allocate(tenant_id);
    assert_eq!(new_owner.slot_id, owner.slot_id, "Slot index should be reused");
    assert_ne!(new_owner.generation, owner.generation, "Generation must increment on reuse");

    assert_eq!(
        view.as_str_checked(&registry, tenant_id),
        Err(LeaseError::StaleGeneration {
            expected: new_owner.generation,
            actual: owner.generation,
        }),
        "Old view accessing reused slot must fail with StaleGeneration"
    );
    assert_eq!(
        view.slice_checked(0, 6, &registry, tenant_id),
        Err(DataError::new("Lease check failed during slice")),
        "slice_checked must reject stale generation"
    );
}

#[test]
fn test_dataview_to_owned_under_budget_oracle() {
    let buffer = "12345678901234567890"; // 20 bytes
    let owner = OwnerId::new(1, 10, 0, 1);
    let view = DataView::from_str(buffer, owner);

    // Budget with 15 bytes limit (insufficient for 20 bytes buffer)
    let mut strict_budget = BudgetTracker::new(15, 1000, 10);
    let budget_res = view.to_owned_with_budget(&mut strict_budget);
    assert!(
        matches!(budget_res, Err(lust::data::budget::BudgetError::BytesExceeded { limit: 15, requested: 20 })),
        "Materialization exceeding budget must fail: {:?}",
        budget_res
    );

    // Budget with 50 bytes limit (sufficient)
    let mut generous_budget = BudgetTracker::new(50, 1000, 10);
    let owned = view.to_owned_with_budget(&mut generous_budget).expect("budget materialization");
    assert_eq!(owned, DataValue::String(buffer.to_string()), "Materialized value mismatch");

    // Second materialization of 40 bytes on the same budget (now 20 + 40 = 60 > 50)
    let large_buf = "a".repeat(40);
    let view_large = DataView::from_str(&large_buf, owner);
    assert!(
        matches!(
            view_large.to_owned_with_budget(&mut generous_budget),
            Err(lust::data::budget::BudgetError::BytesExceeded { limit: 50, requested: 60 })
        ),
        "Subsequent materialization exceeding remaining budget must fail"
    );
}

#[test]
fn test_dataview_slicing_and_queries() {
    let owner = OwnerId::new(1, 1, 0, 1);
    let str_view = DataView::from_str("Hello World", owner);

    assert_eq!(str_view.len(), 11);
    assert!(!str_view.is_empty());
    assert!(str_view.is_str());
    assert!(!str_view.is_binary());
    assert_eq!(str_view.as_str(), Some("Hello World"));

    // Slice valid
    let sub = str_view.slice(0, 5).expect("valid slice");
    assert_eq!(sub.as_str(), Some("Hello"));

    // Slice out of bounds
    assert!(str_view.slice(0, 50).is_err(), "Out of bounds slice must fail");

    // Binary view
    let bin_data: &[u8] = &[1, 2, 3, 4, 5];
    let bin_view = DataView::from_binary(bin_data, owner);
    assert_eq!(bin_view.len(), 5);
    assert!(bin_view.is_binary());
    assert_eq!(bin_view.as_bytes(), Some(bin_data));

    let bin_sub = bin_view.slice(1, 4).expect("valid binary slice");
    assert_eq!(bin_sub.as_bytes(), Some(&[2, 3, 4][..]));

    // Primitive views
    let null_view = DataView::null();
    assert!(null_view.is_null());
    assert_eq!(null_view.len(), 0);
    assert!(null_view.is_empty());

    let bool_view = DataView::from_bool(true);
    assert!(bool_view.is_bool());
    assert_eq!(bool_view.as_bool(), Some(true));

    let num_view = DataView::from_number(ExactNumber::Int(42));
    assert!(num_view.is_number());
    assert_eq!(num_view.as_number(), Some(ExactNumber::Int(42)));

    // Display
    assert_eq!(format!("{}", null_view), "null");
    assert_eq!(format!("{}", bool_view), "true");
    assert_eq!(format!("{}", num_view), "42");
    assert!(format!("{}", str_view).contains("Hello World"));
    assert!(format!("{}", bin_view).contains("len=5"));
}

#[test]
fn test_dataview_utf8_char_boundary_and_non_buffer_slicing() {
    let owner = OwnerId::new(1, 1, 0, 1);

    // Finding 1: Non-buffer slicing: Null, Bool, Number
    assert_eq!(
        DataView::null().slice(0, 0),
        Err(DataError::new("Cannot slice non-buffer view")),
        "Null slice must return 'Cannot slice non-buffer view'"
    );
    assert_eq!(
        DataView::from_bool(true).slice(0, 0),
        Err(DataError::new("Cannot slice non-buffer view")),
        "Bool slice must return 'Cannot slice non-buffer view'"
    );
    assert_eq!(
        DataView::from_number(ExactNumber::Int(123)).slice(0, 0),
        Err(DataError::new("Cannot slice non-buffer view")),
        "Number slice must return 'Cannot slice non-buffer view'"
    );

    // Finding 3: Non-char boundary slicing on multi-byte UTF-8: Cyrillic "Привет"
    // Each Cyrillic character is 2 UTF-8 bytes:
    // 'П' = [0xD0, 0x9F], 'р' = [0xD1, 0x80], 'и' = [0xD0, 0xB8], ...
    let cyrillic_view = DataView::from_str("Привет", owner);
    assert_eq!(cyrillic_view.slice(0, 2).unwrap().as_str(), Some("П"));
    assert_eq!(cyrillic_view.slice(2, 6).unwrap().as_str(), Some("ри"));

    assert_eq!(
        cyrillic_view.slice(0, 1),
        Err(DataError::new("Slice index out of bounds")),
        "Interior byte 1 in Cyrillic 'П' must return Slice index out of bounds"
    );
    assert_eq!(
        cyrillic_view.slice(1, 2),
        Err(DataError::new("Slice index out of bounds")),
        "Interior start 1 in Cyrillic 'П' must return Slice index out of bounds"
    );
    assert_eq!(
        cyrillic_view.slice(1, 3),
        Err(DataError::new("Slice index out of bounds")),
        "Interior start and end in Cyrillic must return Slice index out of bounds"
    );
    assert_eq!(
        cyrillic_view.slice(0, 3),
        Err(DataError::new("Slice index out of bounds")),
        "Interior end 3 in Cyrillic must return Slice index out of bounds"
    );

    // Finding 3: Non-char boundary slicing on 4-byte UTF-8 Emoji: "😀" [0xF0, 0x9F, 0x98, 0x80]
    let emoji_view = DataView::from_str("😀", owner);
    assert_eq!(emoji_view.slice(0, 4).unwrap().as_str(), Some("😀"));
    assert_eq!(
        emoji_view.slice(0, 1),
        Err(DataError::new("Slice index out of bounds")),
        "Emoji interior byte 1 must return Slice index out of bounds"
    );
    assert_eq!(
        emoji_view.slice(0, 2),
        Err(DataError::new("Slice index out of bounds")),
        "Emoji interior byte 2 must return Slice index out of bounds"
    );
    assert_eq!(
        emoji_view.slice(0, 3),
        Err(DataError::new("Slice index out of bounds")),
        "Emoji interior byte 3 must return Slice index out of bounds"
    );
    assert_eq!(
        emoji_view.slice(1, 4),
        Err(DataError::new("Slice index out of bounds")),
        "Emoji interior start 1 must return Slice index out of bounds"
    );
    assert_eq!(
        emoji_view.slice(2, 4),
        Err(DataError::new("Slice index out of bounds")),
        "Emoji interior start 2 must return Slice index out of bounds"
    );

    // Out-of-bounds start > end and end > len
    assert_eq!(
        cyrillic_view.slice(5, 3),
        Err(DataError::new("Slice index out of bounds")),
        "start > end must return Slice index out of bounds"
    );
    assert_eq!(
        cyrillic_view.slice(0, 100),
        Err(DataError::new("Slice index out of bounds")),
        "end > len must return Slice index out of bounds"
    );

    // Binary slicing bounds: start > end and end > len
    let bin_view = DataView::from_binary(&[10, 20, 30], owner);
    assert_eq!(
        bin_view.slice(2, 1),
        Err(DataError::new("Slice index out of bounds")),
        "Binary start > end must return Slice index out of bounds"
    );
    assert_eq!(
        bin_view.slice(0, 10),
        Err(DataError::new("Slice index out of bounds")),
        "Binary end > len must return Slice index out of bounds"
    );

    // Complete DataView query/accessor negative branches
    let null_v = DataView::null();
    assert_eq!(null_v.owner_id(), None);
    assert_eq!(null_v.as_bool(), None);
    assert_eq!(null_v.as_number(), None);
    assert_eq!(null_v.as_str(), None);
    assert_eq!(null_v.as_bytes(), None);
    assert_eq!(null_v.to_owned(), DataValue::Null);

    let bool_v = DataView::from_bool(false);
    assert_eq!(bool_v.owner_id(), None);
    assert_eq!(bool_v.as_number(), None);
    assert_eq!(bool_v.as_str(), None);
    assert_eq!(bool_v.as_bytes(), None);
    assert_eq!(bool_v.to_owned(), DataValue::Bool(false));

    let num_v = DataView::from_number(ExactNumber::UInt(99));
    assert_eq!(num_v.owner_id(), None);
    assert_eq!(num_v.as_bool(), None);
    assert_eq!(num_v.as_str(), None);
    assert_eq!(num_v.as_bytes(), None);
    assert_eq!(num_v.to_owned(), DataValue::Number(ExactNumber::UInt(99)));

    // to_owned_with_budget on len == 0 views
    let mut tracker = BudgetTracker::new(0, 100, 10);
    assert_eq!(null_v.to_owned_with_budget(&mut tracker), Ok(DataValue::Null));
}

// =========================================================================
// 5. DATAVALUE COMPLETENESS TESTS
// =========================================================================

#[test]
fn test_datavalue_conversions_queries_and_accessors() {
    // bool
    let v_bool: DataValue = true.into();
    assert!(v_bool.is_bool());
    assert_eq!(v_bool.as_bool(), Some(true));
    assert_eq!(v_bool.type_name(), "bool");

    // i64 and i32
    let v_i64: DataValue = 42i64.into();
    let v_i32: DataValue = 42i32.into();
    assert_eq!(v_i64, v_i32);
    assert!(v_i64.is_number());
    assert_eq!(v_i64.as_i64(), Some(42));
    assert_eq!(v_i64.type_name(), "int");

    // u64 and u32
    let v_u64: DataValue = 100u64.into();
    let v_u32: DataValue = 100u32.into();
    assert_eq!(v_u64, v_u32);
    assert_eq!(v_u64.as_u64(), Some(100));

    // f64
    let v_f64: DataValue = 3.5f64.into();
    assert_eq!(v_f64.as_f64(), Some(3.5));
    assert_eq!(v_f64.type_name(), "float");

    // String and &str
    let v_str: DataValue = "test string".into();
    let v_string: DataValue = String::from("test string").into();
    assert_eq!(v_str, v_string);
    assert!(v_str.is_string());
    assert_eq!(v_str.as_str(), Some("test string"));
    assert_eq!(v_str.len(), Some(11));
    assert_eq!(v_str.type_name(), "string");

    // Binary
    let bytes = vec![1, 2, 3];
    let v_bin: DataValue = bytes.clone().into();
    assert!(v_bin.is_binary());
    assert_eq!(v_bin.as_binary(), Some(&bytes[..]));
    assert_eq!(v_bin.len(), Some(3));
    assert_eq!(v_bin.type_name(), "binary");

    // Array
    let items = vec![DataValue::Null, DataValue::Bool(false)];
    let mut v_arr: DataValue = items.clone().into();
    assert!(v_arr.is_array());
    assert_eq!(v_arr.as_array(), Some(&items[..]));
    assert_eq!(v_arr.len(), Some(2));
    assert_eq!(v_arr.get_index(0), Some(&DataValue::Null));
    assert_eq!(v_arr.get_index(5), None);
    assert_eq!(v_arr.type_name(), "array");
    v_arr.as_array_mut().unwrap().push(DataValue::Bool(true));
    assert_eq!(v_arr.len(), Some(3));

    // Document
    let mut doc = Document::new();
    doc.try_insert("k1".to_string(), DataValue::Bool(true)).unwrap();
    let mut v_doc: DataValue = doc.clone().into();
    assert!(v_doc.is_document());
    assert_eq!(v_doc.len(), Some(1));
    assert_eq!(v_doc.get("k1"), Some(&DataValue::Bool(true)));
    assert_eq!(v_doc.get("nonexistent"), None);
    assert_eq!(v_doc.type_name(), "document");
    v_doc.as_document_mut().unwrap().try_insert("k2".to_string(), DataValue::Null).unwrap();
    assert_eq!(v_doc.len(), Some(2));

    // ObjectId
    let oid = ObjectId::new([0; 12]);
    let v_oid: DataValue = oid.into();
    assert!(v_oid.is_object_id());
    assert_eq!(v_oid.as_object_id(), Some(&oid));
    assert_eq!(v_oid.type_name(), "objectId");

    // DateTimeUtc
    let dt = DateTimeUtc(1000);
    let v_dt: DataValue = dt.into();
    assert!(v_dt.is_datetime());
    assert_eq!(v_dt.as_datetime(), Some(&dt));
    assert_eq!(v_dt.type_name(), "dateTimeUtc");

    // Decimal128
    let dec = Decimal128([0; 16]);
    let v_dec: DataValue = dec.into();
    assert!(v_dec.is_number());
    assert_eq!(v_dec.as_decimal128(), Some(&dec));

    // Negative queries
    let v_null = DataValue::Null;
    assert!(v_null.is_null());
    assert_eq!(v_null.as_bool(), None);
    assert_eq!(v_null.as_i64(), None);
    assert_eq!(v_null.as_u64(), None);
    assert_eq!(v_null.as_f64(), None);
    assert_eq!(v_null.as_str(), None);
    assert_eq!(v_null.as_array(), None);
    assert_eq!(v_null.as_document(), None);
    assert_eq!(v_null.as_binary(), None);
    assert_eq!(v_null.as_object_id(), None);
    assert_eq!(v_null.as_datetime(), None);
    assert_eq!(v_null.as_decimal128(), None);
    assert_eq!(v_null.len(), None);
    assert_eq!(v_null.is_empty(), None);

    // Display coverage
    assert_eq!(format!("{}", v_null), "null");
    assert_eq!(format!("{}", v_bool), "true");
    assert_eq!(format!("{}", v_i64), "42");
    assert_eq!(format!("{}", v_str), "\"test string\"");
    assert_eq!(format!("{}", v_bin), "<binary len=3>");
    assert!(format!("{}", v_arr).starts_with('['));
    assert!(format!("{}", v_doc).starts_with('{'));
    assert_eq!(format!("{}", v_oid), "ObjectId(\"000000000000000000000000\")");
    assert_eq!(format!("{}", v_dt), "DateTimeUtc(1000 ms)");
}

// =========================================================================
// 6. NUMERIC OPERATIONS & MANTISSA BOUNDARY TESTS (number.rs)
// =========================================================================

#[test]
fn test_numeric_primitives_and_mantissa_boundaries() {
    assert_eq!(int_zero(), 0);
    assert_eq!(int_from_usize(42), 42);
    assert_eq!(float_from_int(42), 42.0);
    assert_eq!(int_from_float(42.7), 42);

    // float_to_hash_bits must distinguish +0.0 and -0.0
    let bits_pos = float_to_hash_bits(0.0);
    let bits_neg = float_to_hash_bits(-0.0);
    assert_ne!(bits_pos, bits_neg, "+0.0 and -0.0 must have different hash bits");

    // NaN and Infinity checks
    assert!(float_is_nan(LustFloat::NAN));
    assert!(!float_is_nan(0.0));
    assert!(float_is_finite(100.5));
    assert!(!float_is_finite(LustFloat::NAN));
    assert!(!float_is_finite(LustFloat::INFINITY));
    assert!(float_is_infinite(LustFloat::INFINITY));
    assert!(float_is_infinite(LustFloat::NEG_INFINITY));
    assert!(!float_is_infinite(1.0));

    // Sign checks
    assert!(float_is_sign_negative(-5.0));
    assert!(float_is_sign_negative(-0.0));
    assert!(!float_is_sign_negative(5.0));
    assert!(float_is_sign_positive(5.0));
    assert!(float_is_sign_positive(0.0));
    assert!(!float_is_sign_positive(-5.0));

    // parse_float
    assert_eq!(parse_float("3.14").unwrap(), 3.14);
    assert!(parse_float("not_a_number").is_err());

    // Basic math
    assert_eq!(float_abs(-10.5), 10.5);
    assert_eq!(float_floor(3.7), 3.0);
    assert_eq!(float_ceil(3.2), 4.0);
    assert_eq!(float_round(3.5), 4.0);
    assert_eq!(float_round(3.4), 3.0);
    assert_eq!(float_sqrt(16.0), 4.0);
    assert!(float_sqrt(-1.0).is_nan());

    // Trigonometry
    assert_eq!(float_sin(0.0), 0.0);
    assert_eq!(float_cos(0.0), 1.0);
    assert_eq!(float_tan(0.0), 0.0);
    assert_eq!(float_asin(0.0), 0.0);
    assert_eq!(float_acos(1.0), 0.0);
    assert_eq!(float_atan(0.0), 0.0);
    assert_eq!(float_atan2(0.0, 1.0), 0.0);

    // Min, max, clamp
    assert_eq!(float_clamp(5.0, 0.0, 10.0), 5.0);
    assert_eq!(float_clamp(-5.0, 0.0, 10.0), 0.0);
    assert_eq!(float_clamp(15.0, 0.0, 10.0), 10.0);
    assert_eq!(float_min(3.0, 7.0), 3.0);
    assert_eq!(float_max(3.0, 7.0), 7.0);

    // Pow, exp, ln
    assert_eq!(float_pow(2.0, 3.0), 8.0);
    assert_eq!(float_ln(1.0), 0.0);
    assert!((float_exp(0.0) - 1.0).abs() < 1e-10);

    // IEEE-754 Safe Integer Mantissa boundaries
    assert!(is_safe_integer_f64(MAX_SAFE_INTEGER_F64));
    assert!(is_safe_integer_f64(MIN_SAFE_INTEGER_F64));
    assert!(is_safe_integer_f64(0.0));
    assert!(!is_safe_integer_f64(MAX_SAFE_INTEGER_F64 + 1.0));
    assert!(!is_safe_integer_f64(MIN_SAFE_INTEGER_F64 - 1.0));
    assert!(!is_safe_integer_f64(1.5));
    assert!(!is_safe_integer_f64(f64::NAN));
    assert!(!is_safe_integer_f64(f64::INFINITY));

    // safe_float_to_int
    assert_eq!(safe_float_to_int(42.0), Some(42));
    assert_eq!(safe_float_to_int(f64::NAN), None);
    assert_eq!(safe_float_to_int(f64::INFINITY), None);

    // safe_int_to_float_exact
    assert_eq!(safe_int_to_float_exact(9007199254740991), Some(9007199254740991.0));
    assert_eq!(safe_int_to_float_exact(i64::MAX), None); // Precision lost beyond 2^53

    // int_clamp & int_abs
    assert_eq!(int_clamp(5, 0, 10), 5);
    assert_eq!(int_clamp(-5, 0, 10), 0);
    assert_eq!(int_clamp(15, 0, 10), 10);
    assert_eq!(int_abs(-42), Some(42));
    assert_eq!(int_abs(i64::MIN), None); // Overflow prevented

    // NumericType enum
    let nt_int = NumericType::Int;
    let nt_float = NumericType::Float;
    assert_eq!(nt_int, NumericType::Int);
    assert_ne!(nt_int, nt_float);
    assert_eq!(format!("{:?}", nt_int), "Int");
}

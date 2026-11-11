//! Desired-behavior regressions: failures are intentional until production fixes land.
use lust::{DataView, DataValue, OwnerRegistry, NativeExport};
use lust::data::{Document, jsonl::estimate_data_value_bytes};

#[test]
fn data1_every_utf8_boundary_is_fallible_and_preserves_owner() {
    let mut registry = OwnerRegistry::new(1);
    let owner = registry.allocate(1);
    let mut violations = Vec::new();
    for text in ["", "abc", "é", "a€🦀z"] {
        let view = DataView::from_str(text, owner);
        for start in 0..=text.len()+1 {
            for end in 0..=text.len()+1 {
                let expected = text.get(start..end);
                match std::panic::catch_unwind(|| view.slice(start, end)) {
                    Ok(Ok(actual)) => {
                        assert_eq!(actual.as_str(), expected, "{text:?}[{start}..{end}]");
                        assert_eq!(actual.owner_id(), Some(owner));
                    }
                    Ok(Err(_)) => assert!(expected.is_none(), "valid slice rejected"),
                    Err(_) => violations.push(format!("{text:?}[{start}..{end}] panicked")),
                }
                let binary = DataView::from_binary(text.as_bytes(), owner);
                match binary.slice(start, end) {
                    Ok(actual) => {
                        assert_eq!(actual.as_bytes(), text.as_bytes().get(start..end));
                        assert_eq!(actual.owner_id(), Some(owner));
                    }
                    Err(_) => assert!(text.as_bytes().get(start..end).is_none()),
                }
            }
        }
    }
    assert!(violations.is_empty(), "{}", violations.join("\n"));
}

#[test]
fn data2_reserved_document_allocation_is_never_undercounted() {
    // Vec::with_capacity guarantees AT LEAST requested capacity. No private-field
    // access or allocator estimate is needed for this conservative lower bound.
    for capacity in [64, 1024, 4096] {
        let value = DataValue::Document(Document::with_capacity(capacity));
        let lower_bound = capacity * std::mem::size_of::<(String, DataValue)>();
        assert!(estimate_data_value_bytes(&value) >= lower_bound,
            "reserved {capacity} entries: estimate {}, minimum heap {lower_bound}",
            estimate_data_value_bytes(&value));
    }
}

#[test]
fn pkg1_stub_paths_match_module_segments() {
    let mut failures = Vec::new();
    for (module, expected) in [("a.b", "a/b.lust"), ("math", "math.lust"),
        ("my-math", "my_math.lust"), ("externs.math", "math.lust")] {
        let exports = [NativeExport::new(format!("{module}.f"), vec![], "int")];
        let files = lust::packages::stub_files_from_exports(&exports, &[]);
        assert_eq!(files.len(), 1);
        assert!(files[0].contents.contains("function f("));
        if files[0].relative_path != std::path::Path::new(expected) {
            failures.push(format!("{module}: {:?}, expected {expected}", files[0].relative_path));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn data3_target_abi_governs_layout_calculation() {
    use lust::data::layout::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1, TargetAbi};
    let fields = vec![
        LogicalFieldDesc::new(0, "tag", CanonicalTypeTag::Int64),
        LogicalFieldDesc::new(1, "s", CanonicalTypeTag::StringSlice),
    ];
    let x86_layout = LogicalLayoutV1::new_with_abi(1, "Test", fields.clone(), TargetAbi::x86_64());
    let rv32_layout = LogicalLayoutV1::new_with_abi(1, "Test", fields, TargetAbi::rv32());

    // On 64-bit architectures, StringSlice is 24 bytes (ptr: 8, len: 8, owner: 8).
    // On 32-bit architectures (rv32), pointers are 4 bytes, so layout size must reflect target ABI.
    assert_ne!(
        x86_layout.size_bytes, rv32_layout.size_bytes,
        "TargetAbi::rv32 must produce distinct struct layout size from x86_64 for pointer-containing slices"
    );
}

#[test]
fn omission01_materialize_from_document_rejects_extra_fields() {
    use lust::data::layout::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1};
    use lust::data::materialize::{materialize_from_document, MaterializeError};
    use lust::data::{ExactNumber};

    let fields = vec![
        LogicalFieldDesc::new(0, "id", CanonicalTypeTag::Int64),
    ];
    let layout = LogicalLayoutV1::new(1, "User", fields);

    let mut doc = Document::new();
    doc.insert("id".to_string(), DataValue::Number(ExactNumber::Int(42)));
    doc.insert("unregistered_field".to_string(), DataValue::String("extra".to_string()));

    let res = materialize_from_document(&layout, &doc);
    match res {
        Err(MaterializeError::ExtraFieldInStrictLayout(field)) => {
            assert_eq!(field, "unregistered_field");
        }
        Ok(_) => panic!("materialize_from_document must fail closed on extra fields in strict layout, but succeeded"),
        Err(e) => panic!("expected ExtraFieldInStrictLayout, got {:?}", e),
    }
}

#[test]
fn fs2_vm_remove_dir_fails_on_non_empty_directory() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let config_path = temp_dir.path().join("lust-config.toml");
    std::fs::write(&config_path, "[settings]\nstdlib_modules = [\"io\", \"os\"]\n").expect("write config");

    let target_dir = temp_dir.path().join("target_dir");
    std::fs::create_dir(&target_dir).expect("create dir");
    let sub_file = target_dir.join("file.txt");
    std::fs::write(&sub_file, "data").expect("write");

    let script_path = temp_dir.path().join("probe.lust");
    std::fs::write(
        &script_path,
        format!(
            "local res = os.remove_dir(\"{}\")\nprintln(res)\n",
            target_dir.display()
        ),
    )
    .expect("write script");

    let bin = env!("CARGO_BIN_EXE_lust");
    let output = std::process::Command::new(bin)
        .args(["run", script_path.to_str().unwrap(), "--backend", "vm"])
        .current_dir(temp_dir.path())
        .output()
        .expect("execute lust run");

    // In POSIX rmdir / Zig runtime, removing a non-empty directory is forbidden and fails.
    // VM must NOT recursively remove all contents under remove_dir!
    assert!(
        target_dir.exists(),
        "os.remove_dir must not remove non-empty directory tree (recursive deletion defect in VM stdlib.rs)"
    );
    assert!(
        sub_file.exists(),
        "files inside directory must not be recursively deleted by os.remove_dir"
    );
    assert_ne!(
        output.stdout, b"Result.Ok(nil)\n",
        "os.remove_dir must not return Ok when attempting to remove a non-empty directory"
    );
}

#[test]
fn data4_json_utf8_multibyte_mojibake_corruption() {
    use lust::data::json::scalar::JsonParser;
    use lust::data::DataValue;

    let input = r#"{"name": "München", "msg": "Привет"}"#.as_bytes();
    let mut parser = JsonParser::new(input);
    let val = parser.parse_all().expect("valid JSON with UTF-8 characters");
    if let DataValue::Document(doc) = val {
        assert_eq!(
            doc.get("name").unwrap().as_str().unwrap(),
            "München",
            "Non-ASCII UTF-8 string was corrupted into mojibake"
        );
        assert_eq!(
            doc.get("msg").unwrap().as_str().unwrap(),
            "Привет",
            "Cyrillic UTF-8 was corrupted into mojibake"
        );
    } else {
        panic!("expected Document");
    }
}

#[test]
fn data5_bson_decoder_negative_bin_len_causes_overflow_panic() {
    use lust::data::bson::BsonDecoder;

    let data = vec![
        16, 0, 0, 0,            // doc_len = 16
        0x05,                   // type: Binary
        b'a', 0x00,             // key "a\0"
        0xFF, 0xFF, 0xFF, 0xFF, // bin_len = -1 (as i32)
        0x00,                   // subtype
        0x00, 0x00, 0x00,       // remaining bytes
        0x00,                   // doc terminator
    ];
    let mut decoder = BsonDecoder::new(&data);
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        decoder.decode_document()
    }));
    assert!(res.is_ok(), "BsonDecoder must not panic on negative length");
    assert!(res.unwrap().is_err(), "negative length must fail gracefully with Err");
}

#[test]
fn data6_bson_uint64_sign_corruption() {
    use lust::data::{bson, DataValue, Document, ExactNumber};

    let mut doc = Document::new();
    let original = u64::MAX;
    doc.try_insert("large_u64", DataValue::Number(ExactNumber::UInt(original))).unwrap();

    let bytes = bson::to_vec(&doc);
    let decoded = bson::from_slice(&bytes).unwrap();
    let val = decoded.get("large_u64").unwrap();

    assert_eq!(val, &DataValue::Number(ExactNumber::UInt(original)), "BSON must preserve unsigned integer values > i64::MAX");
}

#[test]
fn data7_exact_number_boundary_precision_loss() {
    use lust::data::ExactNumber;

    let f_i64_overflow = ExactNumber::Float(9_223_372_036_854_775_808.0);
    assert_eq!(f_i64_overflow.as_i64(), None, "2^63 must not be convertible to i64 (exceeds i64::MAX)");

    let f_u64_overflow = ExactNumber::Float(18_446_744_073_709_551_616.0);
    assert_eq!(f_u64_overflow.as_u64(), None, "2^64 must not be convertible to u64 (exceeds u64::MAX)");
}

#[test]
fn data8_schema_optional_nullable_chain_omitted_field() {
    use lust::schema::s;
    use lust::data::{DataValue, Document};

    let schema = s.object()
        .field("id", s.int())
        .field("metadata", s.string().build().optional().nullable())
        .build();

    let mut doc = Document::new();
    doc.try_insert("id", DataValue::from(1i64)).unwrap();

    let res = schema.parse(&DataValue::Document(doc));
    assert!(res.is_ok(), "Optional field with nullable() must not be required when omitted: {:?}", res);
}

#[test]
fn data9_schema_float_nan_bypasses_min_max_bounds() {
    use lust::schema::s;
    use lust::data::{DataValue, ExactNumber};

    let schema = s.float().min(0.0).max(100.0).build();
    let nan_val = DataValue::Number(ExactNumber::Float(f64::NAN));

    let res = schema.parse(&nan_val);
    assert!(res.is_err(), "NaN must not satisfy 0.0 <= f <= 100.0 range bounds");
}

#[test]
fn tc01_validate_is_pattern_err_uses_ok_type() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let script_path = temp_dir.path().join("probe.lust");
    std::fs::write(
        &script_path,
        r#"
        local r: Result<string, Option<int>> = Result.Err(Option.Some(42))
        local matched = false
        if r is Err(Some(x)) then
            matched = true
        end
        println(matched)
        "#,
    ).expect("write");

    let bin = env!("CARGO_BIN_EXE_lust");
    let output = std::process::Command::new(bin)
        .args(["run", script_path.to_str().unwrap(), "--backend", "vm"])
        .env("LUST_JIT", "0")
        .output()
        .expect("run");

    assert!(
        output.status.success(),
        "Compiler must validate Err pattern using Err type parameter, but failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}


#[test]
fn vm01_errdefer_is_emitted_and_executed() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let script_path = temp_dir.path().join("probe.lust");
    std::fs::write(
        &script_path,
        r#"
        function test_err(): Result<int, string>
            errdefer println("ERRDEFER_RAN")
            return Result.Err("oops")
        end
        test_err()
        "#,
    ).expect("write");

    let bin = env!("CARGO_BIN_EXE_lust");
    let output = std::process::Command::new(bin)
        .args(["run", script_path.to_str().unwrap(), "--backend", "vm"])
        .env("LUST_JIT", "0")
        .output()
        .expect("run");

    assert!(
        String::from_utf8_lossy(&output.stdout).contains("ERRDEFER_RAN"),
        "errdefer block must be emitted and executed on error return, but was swallowed"
    );
}

#[test]
fn vm03_upvalue_mutations_persist_across_closure_calls() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let script_path = temp_dir.path().join("probe.lust");
    std::fs::write(
        &script_path,
        r#"
        function make_counter(): function(): int
            local count = 0
            return function(): int
                count = count + 1
                return count
            end
        end
        local c = make_counter()
        println(c())
        println(c())
        "#,
    ).expect("write");

    let bin = env!("CARGO_BIN_EXE_lust");
    let output = std::process::Command::new(bin)
        .args(["run", script_path.to_str().unwrap(), "--backend", "vm"])
        .env("LUST_JIT", "0")
        .output()
        .expect("run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout, "1\n2\n", "Upvalue mutation must persist across closure calls (StoreUpvalue)");
}

#[test]
fn vm04_task_stop_does_not_clear_parent_stack() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let script_path = temp_dir.path().join("probe.lust");
    std::fs::write(
        &script_path,
        r#"
        function worker(): string
            task.yield("working")
            return "done"
        end
        function caller()
            local t = task.create(worker)
            task.resume(t)
            task.stop(t)
            println("CALLER_ALIVE")
        end
        caller()
        "#,
    ).expect("write");


    let bin = env!("CARGO_BIN_EXE_lust");
    let output = std::process::Command::new(bin)
        .args(["run", script_path.to_str().unwrap(), "--backend", "vm"])
        .env("LUST_JIT", "0")
        .output()
        .expect("run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("CALLER_ALIVE"), "task.stop must not clear parent call stack; stderr: {}", String::from_utf8_lossy(&output.stderr));
}





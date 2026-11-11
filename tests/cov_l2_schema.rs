use std::sync::Arc;
use lust::ast::{FieldOwnership, Item, ItemKind, Span, StructDef, StructField, Type, TypeKind, Visibility};
use lust::data::layout::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1, TargetAbi};
use lust::data::{DataValue, DateTimeUtc, Decimal128, Document, DuplicateKeyError, ExactNumber, ObjectId};
use lust::schema::binding::{
    derive_schema, derive_schema_from_item, derive_schema_from_struct_def,
    DeriveSchemaTarget, StructSchemaBinding,
};
use lust::schema::builder::{s, unpack_optional, ArrayBuilder, BinaryBuilder, FloatBuilder, IntBuilder, ObjectBuilder, Schema, StringBuilder};
use lust::schema::error::{ValidationError, ValidationErrors};
use lust::schema::ir::{ObjectFieldIR, RefinePredicate, SchemaIR, TransformFn};
use lust::schema::validate::validate_ir;
use lust::schema::Effect;

// ============================================================================
// 1. Table of Pairs Oracle: (Document, Schema) -> Ok or Exact Error Code + Path
// ============================================================================

struct TableTestCase {
    name: &'static str,
    schema: Schema,
    input: DataValue,
    expected_ok: Option<DataValue>,
    expected_err: Option<(&'static str, &'static str)>, // (code, path)
}

#[test]
fn test_table_of_pairs_oracle() {
    let cases = vec![
        // String primitives
        TableTestCase {
            name: "string_valid",
            schema: s.string().min(2).max(10).build(),
            input: DataValue::String("valid".to_string()),
            expected_ok: Some(DataValue::String("valid".to_string())),
            expected_err: None,
        },
        TableTestCase {
            name: "string_too_small",
            schema: s.string().min(5).build(),
            input: DataValue::String("abc".to_string()),
            expected_ok: None,
            expected_err: Some(("too_small", "$")),
        },
        TableTestCase {
            name: "string_too_big",
            schema: s.string().max(3).build(),
            input: DataValue::String("toolong".to_string()),
            expected_ok: None,
            expected_err: Some(("too_big", "$")),
        },
        TableTestCase {
            name: "string_regex_match",
            schema: s.string().regex("^[a-z]+$").build(),
            input: DataValue::String("lowercase".to_string()),
            expected_ok: Some(DataValue::String("lowercase".to_string())),
            expected_err: None,
        },
        TableTestCase {
            name: "string_regex_mismatch",
            schema: s.string().regex("^[a-z]+$").build(),
            input: DataValue::String("UPPER123".to_string()),
            expected_ok: None,
            expected_err: Some(("regex_mismatch", "$")),
        },
        TableTestCase {
            name: "string_invalid_type",
            schema: s.string().build(),
            input: DataValue::Number(ExactNumber::Int(42)),
            expected_ok: None,
            expected_err: Some(("invalid_type", "$")),
        },

        // Int primitives
        TableTestCase {
            name: "int_valid",
            schema: s.int().min(10).max(100).build(),
            input: DataValue::Number(ExactNumber::Int(50)),
            expected_ok: Some(DataValue::Number(ExactNumber::Int(50))),
            expected_err: None,
        },
        TableTestCase {
            name: "int_uint_valid_conversion",
            schema: s.int().min(0).max(100).build(),
            input: DataValue::Number(ExactNumber::UInt(50)),
            expected_ok: Some(DataValue::Number(ExactNumber::Int(50))),
            expected_err: None,
        },
        TableTestCase {
            name: "int_too_small",
            schema: s.int().min(10).build(),
            input: DataValue::Number(ExactNumber::Int(5)),
            expected_ok: None,
            expected_err: Some(("too_small", "$")),
        },
        TableTestCase {
            name: "int_too_big",
            schema: s.int().max(100).build(),
            input: DataValue::Number(ExactNumber::Int(150)),
            expected_ok: None,
            expected_err: Some(("too_big", "$")),
        },
        TableTestCase {
            name: "int_invalid_type_string",
            schema: s.int().build(),
            input: DataValue::String("not_an_int".to_string()),
            expected_ok: None,
            expected_err: Some(("invalid_type", "$")),
        },
        TableTestCase {
            name: "int_uint_overflow_rejected",
            schema: s.int().build(),
            input: DataValue::Number(ExactNumber::UInt(u64::MAX)),
            expected_ok: None,
            expected_err: Some(("invalid_type", "$")),
        },

        // Float primitives
        TableTestCase {
            name: "float_valid",
            schema: s.float().min(0.0).max(100.0).build(),
            input: DataValue::Number(ExactNumber::Float(45.5)),
            expected_ok: Some(DataValue::Number(ExactNumber::Float(45.5))),
            expected_err: None,
        },
        TableTestCase {
            name: "float_int_conversion",
            schema: s.float().min(0.0).max(100.0).build(),
            input: DataValue::Number(ExactNumber::Int(42)),
            expected_ok: Some(DataValue::Number(ExactNumber::Float(42.0))),
            expected_err: None,
        },
        TableTestCase {
            name: "float_uint_conversion",
            schema: s.float().min(0.0).max(100.0).build(),
            input: DataValue::Number(ExactNumber::UInt(42)),
            expected_ok: Some(DataValue::Number(ExactNumber::Float(42.0))),
            expected_err: None,
        },
        TableTestCase {
            name: "float_too_small",
            schema: s.float().min(10.0).build(),
            input: DataValue::Number(ExactNumber::Float(2.5)),
            expected_ok: None,
            expected_err: Some(("too_small", "$")),
        },
        TableTestCase {
            name: "float_too_big",
            schema: s.float().max(50.0).build(),
            input: DataValue::Number(ExactNumber::Float(75.5)),
            expected_ok: None,
            expected_err: Some(("too_big", "$")),
        },
        TableTestCase {
            name: "float_nan_with_bounds_fails",
            schema: s.float().min(0.0).build(),
            input: DataValue::Number(ExactNumber::Float(f64::NAN)),
            expected_ok: None,
            expected_err: Some(("invalid_range", "$")),
        },
        TableTestCase {
            name: "float_nan_without_bounds_succeeds",
            schema: s.float().build(),
            input: DataValue::Number(ExactNumber::Float(f64::NAN)),
            expected_ok: Some(DataValue::Number(ExactNumber::Float(f64::NAN))),
            expected_err: None,
        },
        TableTestCase {
            name: "float_invalid_type",
            schema: s.float().build(),
            input: DataValue::Bool(true),
            expected_ok: None,
            expected_err: Some(("invalid_type", "$")),
        },

        // Bool primitive
        TableTestCase {
            name: "bool_valid_true",
            schema: s.bool(),
            input: DataValue::Bool(true),
            expected_ok: Some(DataValue::Bool(true)),
            expected_err: None,
        },
        TableTestCase {
            name: "bool_valid_false",
            schema: s.bool(),
            input: DataValue::Bool(false),
            expected_ok: Some(DataValue::Bool(false)),
            expected_err: None,
        },
        TableTestCase {
            name: "bool_invalid_type",
            schema: s.bool(),
            input: DataValue::String("true".to_string()),
            expected_ok: None,
            expected_err: Some(("invalid_type", "$")),
        },

        // Any primitive
        TableTestCase {
            name: "any_accepts_anything",
            schema: s.any(),
            input: DataValue::Null,
            expected_ok: Some(DataValue::Null),
            expected_err: None,
        },

        // Literal primitive
        TableTestCase {
            name: "literal_match",
            schema: s.literal(DataValue::String("alpha".to_string())),
            input: DataValue::String("alpha".to_string()),
            expected_ok: Some(DataValue::String("alpha".to_string())),
            expected_err: None,
        },
        TableTestCase {
            name: "literal_mismatch",
            schema: s.literal(DataValue::String("alpha".to_string())),
            input: DataValue::String("beta".to_string()),
            expected_ok: None,
            expected_err: Some(("invalid_literal", "$")),
        },

        // Binary primitive
        TableTestCase {
            name: "binary_valid",
            schema: s.binary().min(2).max(5).build(),
            input: DataValue::Binary(vec![1, 2, 3]),
            expected_ok: Some(DataValue::Binary(vec![1, 2, 3])),
            expected_err: None,
        },
        TableTestCase {
            name: "binary_too_small",
            schema: s.binary().min(4).build(),
            input: DataValue::Binary(vec![1, 2]),
            expected_ok: None,
            expected_err: Some(("too_small", "$")),
        },
        TableTestCase {
            name: "binary_too_big",
            schema: s.binary().max(2).build(),
            input: DataValue::Binary(vec![1, 2, 3, 4]),
            expected_ok: None,
            expected_err: Some(("too_big", "$")),
        },
        TableTestCase {
            name: "binary_invalid_type",
            schema: s.binary().build(),
            input: DataValue::String("bytes".to_string()),
            expected_ok: None,
            expected_err: Some(("invalid_type", "$")),
        },

        // ObjectId primitive
        TableTestCase {
            name: "object_id_valid",
            schema: s.object_id(),
            input: DataValue::ObjectId(ObjectId::new([7; 12])),
            expected_ok: Some(DataValue::ObjectId(ObjectId::new([7; 12]))),
            expected_err: None,
        },
        TableTestCase {
            name: "object_id_invalid_type",
            schema: s.object_id(),
            input: DataValue::String("not_oid".to_string()),
            expected_ok: None,
            expected_err: Some(("invalid_type", "$")),
        },

        // DateTimeUtc primitive
        TableTestCase {
            name: "datetime_utc_valid",
            schema: s.datetime_utc(),
            input: DataValue::DateTimeUtc(DateTimeUtc::from_millis(1700000000000)),
            expected_ok: Some(DataValue::DateTimeUtc(DateTimeUtc::from_millis(1700000000000))),
            expected_err: None,
        },
        TableTestCase {
            name: "datetime_utc_invalid_type",
            schema: s.datetime_utc(),
            input: DataValue::Number(ExactNumber::Int(1700000000000)),
            expected_ok: None,
            expected_err: Some(("invalid_type", "$")),
        },

        // Decimal128 primitive
        TableTestCase {
            name: "decimal128_valid",
            schema: s.decimal128(),
            input: DataValue::Number(ExactNumber::Decimal128(Decimal128::new([1; 16]))),
            expected_ok: Some(DataValue::Number(ExactNumber::Decimal128(Decimal128::new([1; 16])))),
            expected_err: None,
        },
        TableTestCase {
            name: "decimal128_invalid_type",
            schema: s.decimal128(),
            input: DataValue::Number(ExactNumber::Float(1.23)),
            expected_ok: None,
            expected_err: Some(("invalid_type", "$")),
        },

        // Array
        TableTestCase {
            name: "array_valid",
            schema: s.array(s.int()).min(1).max(3).build(),
            input: DataValue::Array(vec![
                DataValue::Number(ExactNumber::Int(10)),
                DataValue::Number(ExactNumber::Int(20)),
            ]),
            expected_ok: Some(DataValue::Array(vec![
                DataValue::Number(ExactNumber::Int(10)),
                DataValue::Number(ExactNumber::Int(20)),
            ])),
            expected_err: None,
        },
        TableTestCase {
            name: "array_too_few_items",
            schema: s.array(s.int()).min(2).build(),
            input: DataValue::Array(vec![DataValue::Number(ExactNumber::Int(1))]),
            expected_ok: None,
            expected_err: Some(("too_few_items", "$")),
        },
        TableTestCase {
            name: "array_too_many_items",
            schema: s.array(s.int()).max(2).build(),
            input: DataValue::Array(vec![
                DataValue::Number(ExactNumber::Int(1)),
                DataValue::Number(ExactNumber::Int(2)),
                DataValue::Number(ExactNumber::Int(3)),
            ]),
            expected_ok: None,
            expected_err: Some(("too_many_items", "$")),
        },
        TableTestCase {
            name: "array_item_invalid_type_index_0",
            schema: s.array(s.int()).build(),
            input: DataValue::Array(vec![DataValue::String("bad".to_string())]),
            expected_ok: None,
            expected_err: Some(("invalid_type", "0")),
        },
        TableTestCase {
            name: "array_item_invalid_type_index_2",
            schema: s.array(s.int()).build(),
            input: DataValue::Array(vec![
                DataValue::Number(ExactNumber::Int(1)),
                DataValue::Number(ExactNumber::Int(2)),
                DataValue::String("bad".to_string()),
            ]),
            expected_ok: None,
            expected_err: Some(("invalid_type", "2")),
        },
        TableTestCase {
            name: "array_invalid_type",
            schema: s.array(s.int()).build(),
            input: DataValue::String("not_an_array".to_string()),
            expected_ok: None,
            expected_err: Some(("invalid_type", "$")),
        },

        // Union
        TableTestCase {
            name: "union_match_first",
            schema: s.union(vec![s.int().build(), s.string().build()]),
            input: DataValue::Number(ExactNumber::Int(10)),
            expected_ok: Some(DataValue::Number(ExactNumber::Int(10))),
            expected_err: None,
        },
        TableTestCase {
            name: "union_match_second",
            schema: s.union(vec![s.int().build(), s.string().build()]),
            input: DataValue::String("matched".to_string()),
            expected_ok: Some(DataValue::String("matched".to_string())),
            expected_err: None,
        },
        TableTestCase {
            name: "union_mismatch_all",
            schema: s.union(vec![s.int().build(), s.string().build()]),
            input: DataValue::Bool(true),
            expected_ok: None,
            expected_err: Some(("invalid_union", "$")),
        },
    ];

    for case in &cases {
        let result = case.schema.parse(&case.input);
        if let Some(ref exp_ok) = case.expected_ok {
            assert!(
                result.is_ok(),
                "Case '{}' failed: input {:?} against schema was expected to succeed but got err: {:?}",
                case.name, case.input, result.err()
            );
            let val = result.unwrap();
            if exp_ok.is_nan_float() {
                assert!(val.is_nan_float(), "Case '{}' expected NaN float", case.name);
            } else {
                assert_eq!(
                    &val, exp_ok,
                    "Case '{}': input {:?} returned {:?}, expected {:?}",
                    case.name, case.input, val, exp_ok
                );
            }
        } else if let Some((exp_code, exp_path)) = case.expected_err {
            assert!(
                result.is_err(),
                "Case '{}' failed: input {:?} against schema was expected to fail with ({}, {}) but succeeded",
                case.name, case.input, exp_code, exp_path
            );
            let errors = result.unwrap_err();
            assert!(
                !errors.is_empty(),
                "Case '{}': errors list must not be empty",
                case.name
            );
            let err = &errors.errors[0];
            assert_eq!(
                err.code, exp_code,
                "Case '{}': input {:?} expected error code '{}', got '{}'",
                case.name, case.input, exp_code, err.code
            );
            assert_eq!(
                err.path_string(), exp_path,
                "Case '{}': input {:?} expected error path '{}', got '{}'",
                case.name, case.input, exp_path, err.path_string()
            );
        }
    }
}

trait NanCheck {
    fn is_nan_float(&self) -> bool;
}

impl NanCheck for DataValue {
    fn is_nan_float(&self) -> bool {
        match self {
            DataValue::Number(ExactNumber::Float(f)) => f.is_nan(),
            _ => false,
        }
    }
}

// ============================================================================
// 2. Composition of Optional / Nullable
// ============================================================================

#[test]
fn test_composition_optional_and_nullable() {
    // 1. Optional Nullable: s.string().nullable().optional()
    // Permitted: omitted, null, valid string.
    // Rejected: wrong type (e.g. integer)
    let schema_opt_null = s.object().field("field", s.string().nullable().optional()).build();

    // 1a. Omitted field -> OK
    let doc_omitted = DataValue::Document(Document::new());
    assert!(
        schema_opt_null.parse(&doc_omitted).is_ok(),
        "Optional nullable field when omitted must be accepted"
    );

    // 1b. Explicit null -> OK
    let mut doc_null = Document::new();
    doc_null.try_insert("field", DataValue::Null).unwrap();
    let res = schema_opt_null.parse(&DataValue::Document(doc_null.clone()));
    assert!(
        res.is_ok(),
        "Optional nullable field when explicit null must be accepted, got: {:?}", res.err()
    );

    // 1c. Valid string -> OK
    let mut doc_str = Document::new();
    doc_str.try_insert("field", DataValue::String("hello".to_string())).unwrap();
    assert!(schema_opt_null.parse(&DataValue::Document(doc_str)).is_ok());

    // 1d. Wrong type -> Rejected with invalid_type at path "field"
    let mut doc_bad = Document::new();
    doc_bad.try_insert("field", DataValue::Number(ExactNumber::Int(123))).unwrap();
    let err = schema_opt_null.parse(&DataValue::Document(doc_bad)).unwrap_err();
    assert_eq!(err.errors[0].code, "invalid_type", "Optional nullable with wrong type must reject with invalid_type");
    assert_eq!(err.errors[0].path_string(), "field");

    // 2. Required Nullable: s.string().nullable()
    // Permitted: explicit null, valid string.
    // Rejected: omitted (missing_field), wrong type (invalid_type)
    let schema_req_null = s.object().field("field", s.string().nullable()).build();

    // 2a. Omitted -> Rejected with missing_field at path "field"
    let err = schema_req_null.parse(&doc_omitted).unwrap_err();
    assert_eq!(err.errors[0].code, "missing_field", "Required nullable field when omitted must be rejected");
    assert_eq!(err.errors[0].path_string(), "field");

    // 2b. Explicit null -> OK
    assert!(schema_req_null.parse(&DataValue::Document(doc_null)).is_ok());

    // 2c. Valid string -> OK
    let mut doc_str2 = Document::new();
    doc_str2.try_insert("field", DataValue::String("valid".to_string())).unwrap();
    assert!(schema_req_null.parse(&DataValue::Document(doc_str2)).is_ok());

    // 3. Optional Non-Nullable: s.string().optional()
    // Permitted: omitted, valid string.
    // Rejected: explicit null (invalid_type), wrong type (invalid_type)
    let schema_opt_non_null = s.object().field("field", s.string().optional()).build();

    // 3a. Omitted -> OK
    assert!(schema_opt_non_null.parse(&doc_omitted).is_ok());

    // 3b. Explicit null -> Rejected with invalid_type at path "field"
    let mut doc_null2 = Document::new();
    doc_null2.try_insert("field", DataValue::Null).unwrap();
    let err = schema_opt_non_null.parse(&DataValue::Document(doc_null2)).unwrap_err();
    assert_eq!(err.errors[0].code, "invalid_type", "Optional non-nullable field with null must be rejected with invalid_type");
    assert_eq!(err.errors[0].path_string(), "field");

    // 3c. Valid string -> OK
    let mut doc_str3 = Document::new();
    doc_str3.try_insert("field", DataValue::String("ok".to_string())).unwrap();
    assert!(schema_opt_non_null.parse(&DataValue::Document(doc_str3)).is_ok());

    // 4. Required Non-Nullable: s.string()
    // Permitted: valid string.
    // Rejected: omitted (missing_field), explicit null (invalid_type), wrong type (invalid_type)
    let schema_req_non_null = s.object().field("field", s.string()).build();

    let err = schema_req_non_null.parse(&doc_omitted).unwrap_err();
    assert_eq!(err.errors[0].code, "missing_field");
    assert_eq!(err.errors[0].path_string(), "field");

    let mut doc_null3 = Document::new();
    doc_null3.try_insert("field", DataValue::Null).unwrap();
    let err = schema_req_non_null.parse(&DataValue::Document(doc_null3)).unwrap_err();
    assert_eq!(err.errors[0].code, "invalid_type");
    assert_eq!(err.errors[0].path_string(), "field");

    // Also verify reversed order: s.string().optional().nullable() is also optional nullable
    let schema_reversed = s.object().field("field", s.string().optional().nullable()).build();
    assert!(schema_reversed.parse(&doc_omitted).is_ok());
    let mut doc_null4 = Document::new();
    doc_null4.try_insert("field", DataValue::Null).unwrap();
    assert!(schema_reversed.parse(&DataValue::Document(doc_null4)).is_ok());
}

// ============================================================================
// 3. Extra Keys in Strict Mode Rejected with Exact Path
// ============================================================================

#[test]
fn test_strict_mode_rejects_extra_keys_with_exact_path() {
    let schema = s
        .object()
        .field("id", s.int())
        .strict()
        .build();

    let mut doc = Document::new();
    doc.try_insert("id", DataValue::Number(ExactNumber::Int(1))).unwrap();
    doc.try_insert("unexpected_extra", DataValue::String("bad".to_string())).unwrap();

    let err = schema.parse(&DataValue::Document(doc)).unwrap_err();
    assert_eq!(err.errors.len(), 1);
    assert_eq!(err.errors[0].code, "unrecognized_key", "Strict mode must reject extra keys with unrecognized_key");
    assert_eq!(err.errors[0].path_string(), "unexpected_extra", "Error path must be exact key name");

    // Nested strict object extra key path
    let nested_schema = s
        .object()
        .field(
            "user",
            s.object()
                .field("name", s.string())
                .strict()
                .build(),
        )
        .strict()
        .build();

    let mut inner_doc = Document::new();
    inner_doc.try_insert("name", DataValue::String("alice".to_string())).unwrap();
    inner_doc.try_insert("extra_nested", DataValue::Bool(true)).unwrap();

    let mut outer_doc = Document::new();
    outer_doc.try_insert("user", DataValue::Document(inner_doc)).unwrap();

    let err = nested_schema.parse(&DataValue::Document(outer_doc)).unwrap_err();
    assert_eq!(err.errors.len(), 1);
    assert_eq!(err.errors[0].code, "unrecognized_key");
    assert_eq!(
        err.errors[0].path_string(), "user.extra_nested",
        "Nested extra key path must include parent object path"
    );

    // Non-strict object preserves extra keys without error
    let non_strict = s.object().field("id", s.int()).build();
    let mut non_strict_doc = Document::new();
    non_strict_doc.try_insert("id", DataValue::Number(ExactNumber::Int(1))).unwrap();
    non_strict_doc.try_insert("extra", DataValue::String("kept".to_string())).unwrap();

    let validated = non_strict.parse(&DataValue::Document(non_strict_doc)).unwrap();
    if let DataValue::Document(d) = validated {
        assert_eq!(d.get("extra"), Some(&DataValue::String("kept".to_string())));
    } else {
        panic!("Expected Document");
    }
}

// ============================================================================
// 4. Missing Required Field Rejected with Exact Path
// ============================================================================

#[test]
fn test_missing_required_field_rejected_with_exact_path() {
    let deep_schema = s
        .object()
        .field(
            "level1",
            s.object()
                .field(
                    "level2",
                    s.object()
                        .field("required_leaf", s.string())
                        .build(),
                )
                .build(),
        )
        .build();

    // Leaf missing
    let mut doc_missing_leaf = Document::new();
    let mut l1 = Document::new();
    let l2 = Document::new(); // empty level2
    l1.try_insert("level2", DataValue::Document(l2)).unwrap();
    doc_missing_leaf.try_insert("level1", DataValue::Document(l1)).unwrap();

    let err = deep_schema.parse(&DataValue::Document(doc_missing_leaf)).unwrap_err();
    assert_eq!(err.errors[0].code, "missing_field");
    assert_eq!(
        err.errors[0].path_string(), "level1.level2.required_leaf",
        "Missing required field path must pinpoint deep leaf location"
    );

    // Intermediate level missing
    let mut doc_missing_l1 = Document::new();
    let l1_empty = Document::new();
    doc_missing_l1.try_insert("level1", DataValue::Document(l1_empty)).unwrap();

    let err = deep_schema.parse(&DataValue::Document(doc_missing_l1)).unwrap_err();
    assert_eq!(err.errors[0].code, "missing_field");
    assert_eq!(err.errors[0].path_string(), "level1.level2");
}

// ============================================================================
// 5. Duplicate Keys Rejected
// ============================================================================

#[test]
fn test_duplicate_keys_rejected() {
    // 5a. Document::try_insert duplicate rejection oracle
    let mut doc = Document::new();
    assert!(doc.try_insert("key", DataValue::Number(ExactNumber::Int(1))).is_ok());
    let dup_res = doc.try_insert("key", DataValue::Number(ExactNumber::Int(2)));
    assert_eq!(dup_res, Err(DuplicateKeyError), "try_insert must return DuplicateKeyError on duplicate key");

    // 5b. Schema with duplicate field definitions fails closed with duplicate_key
    let dup_schema = s
        .object()
        .field("tag", s.string())
        .field("tag", s.string())
        .build();
    let mut valid_doc = Document::new();
    valid_doc.try_insert("tag", DataValue::String("val".to_string())).unwrap();
    let res = dup_schema.parse(&DataValue::Document(valid_doc));
    assert!(res.is_err(), "Duplicate field schema must fail closed on validation");
    let err = res.unwrap_err();
    assert_eq!(err.errors[0].code, "duplicate_key");
    assert_eq!(err.errors[0].path_string(), "tag");

    // 5c. Schema with duplicate field via default
    let dup_default_schema = s
        .object()
        .field_with_default("dup", s.int(), DataValue::Number(ExactNumber::Int(1)))
        .field_with_default("dup", s.int(), DataValue::Number(ExactNumber::Int(2)))
        .build();
    let empty_doc = Document::new();
    let res = dup_default_schema.parse(&DataValue::Document(empty_doc));
    assert!(res.is_err(), "Duplicate default field schema must fail closed");
    let err = res.unwrap_err();
    assert_eq!(err.errors[0].code, "duplicate_key");
    assert_eq!(err.errors[0].path_string(), "dup");
}

// ============================================================================
// 6. Malformed Regex at Builder Compile Time Fails Closed
// ============================================================================

#[test]
fn test_malformed_regex_fails_closed() {
    let invalid_patterns = vec![
        "[a-",     // unclosed bracket
        "(abc",    // unclosed parenthesis
        "*bad",    // repeat without operand
        "\\",      // trailing backslash
    ];

    for pattern in invalid_patterns {
        let builder = s.string().regex(pattern);
        let schema = builder.build();

        // 6a. Builder compile produces InvalidRegex IR
        assert!(
            matches!(schema.to_ir(), SchemaIR::InvalidRegex { pattern: pat, .. } if pat == pattern),
            "Builder with invalid pattern '{}' must compile to SchemaIR::InvalidRegex",
            pattern
        );

        // 6b. Validation fails closed with invalid_regex error
        let input = DataValue::String("some_test_string".to_string());
        let res = schema.parse(&input);
        assert!(
            res.is_err(),
            "Invalid regex pattern '{}' must fail closed on validation",
            pattern
        );
        let err = res.unwrap_err();
        assert_eq!(
            err.errors[0].code, "invalid_regex",
            "Error code for malformed regex must be 'invalid_regex', got: {}", err.errors[0].code
        );

        // 6c. Direct validate_ir on SchemaIR::String with invalid pattern also fails closed
        let direct_ir = SchemaIR::String {
            min_len: None,
            max_len: None,
            regex: Some(pattern.to_string()),
        };
        let mut path = Vec::new();
        let mut errors = Vec::new();
        let out = validate_ir(&direct_ir, &input, &mut path, &mut errors);
        assert!(out.is_none());
        assert_eq!(errors[0].code, "invalid_regex");
    }
}

// ============================================================================
// 7. Range Limits (Min, Max, Float NaN)
// ============================================================================

#[test]
fn test_range_limits_and_float_nan() {
    // 7a. Float boundaries: min=10.0, max=20.0
    let f_schema = s.float().min(10.0).max(20.0).build();

    // Boundary - epsilon
    let err = f_schema.parse(&DataValue::Number(ExactNumber::Float(9.99999))).unwrap_err();
    assert_eq!(err.errors[0].code, "too_small");
    // Boundary exact min
    assert!(f_schema.parse(&DataValue::Number(ExactNumber::Float(10.0))).is_ok());
    // Boundary midpoint
    assert!(f_schema.parse(&DataValue::Number(ExactNumber::Float(15.0))).is_ok());
    // Boundary exact max
    assert!(f_schema.parse(&DataValue::Number(ExactNumber::Float(20.0))).is_ok());
    // Boundary + epsilon
    let err = f_schema.parse(&DataValue::Number(ExactNumber::Float(20.00001))).unwrap_err();
    assert_eq!(err.errors[0].code, "too_big");

    // Float Infinity and -Infinity
    let err = f_schema.parse(&DataValue::Number(ExactNumber::Float(f64::INFINITY))).unwrap_err();
    assert_eq!(err.errors[0].code, "too_big");
    let err = f_schema.parse(&DataValue::Number(ExactNumber::Float(f64::NEG_INFINITY))).unwrap_err();
    assert_eq!(err.errors[0].code, "too_small");

    // Float NaN with bounds specified MUST fail closed with invalid_range
    let err = f_schema.parse(&DataValue::Number(ExactNumber::Float(f64::NAN))).unwrap_err();
    assert_eq!(err.errors[0].code, "invalid_range");
    assert_eq!(err.errors[0].received, Some("NaN".to_string()));

    // 7b. Int boundaries: min=-5, max=10
    let i_schema = s.int().min(-5).max(10).build();
    let err = i_schema.parse(&DataValue::Number(ExactNumber::Int(-6))).unwrap_err();
    assert_eq!(err.errors[0].code, "too_small");
    assert!(i_schema.parse(&DataValue::Number(ExactNumber::Int(-5))).is_ok());
    assert!(i_schema.parse(&DataValue::Number(ExactNumber::Int(10))).is_ok());
    let err = i_schema.parse(&DataValue::Number(ExactNumber::Int(11))).unwrap_err();
    assert_eq!(err.errors[0].code, "too_big");

    // 7c. String multi-byte UTF-8 char counting (not bytes)
    // "Привет" is 6 characters but 12 UTF-8 bytes.
    let str_schema = s.string().min(5).max(7).build();
    let utf8_val = DataValue::String("Привет".to_string());
    assert!(
        str_schema.parse(&utf8_val).is_ok(),
        "6 Cyrillic characters must satisfy min=5, max=7 (was counted in chars, not bytes)"
    );

    let too_short_utf8 = DataValue::String("Пр".to_string()); // 2 chars, 4 bytes
    let err = str_schema.parse(&too_short_utf8).unwrap_err();
    assert_eq!(err.errors[0].code, "too_small");
    assert_eq!(err.errors[0].received, Some("2".to_string()));

    // 7d. Binary byte length limits
    let bin_schema = s.binary().min(3).max(5).build();
    assert!(bin_schema.parse(&DataValue::Binary(vec![1, 2, 3])).is_ok());
    assert!(bin_schema.parse(&DataValue::Binary(vec![1, 2, 3, 4, 5])).is_ok());
    let err = bin_schema.parse(&DataValue::Binary(vec![1, 2])).unwrap_err();
    assert_eq!(err.errors[0].code, "too_small");
    let err = bin_schema.parse(&DataValue::Binary(vec![1, 2, 3, 4, 5, 6])).unwrap_err();
    assert_eq!(err.errors[0].code, "too_big");
}

// ============================================================================
// 8. Array Item Validation with Element Index Path
// ============================================================================

#[test]
fn test_array_item_validation_element_index_path() {
    // Array of objects
    let item_schema = s
        .object()
        .field("id", s.int())
        .field("email", s.string().regex(r"^[a-z0-9]+@[a-z0-9]+\.[a-z]+$"))
        .build();

    let array_schema = s.array(item_schema).build();

    let mut item0 = Document::new();
    item0.try_insert("id", DataValue::Number(ExactNumber::Int(1))).unwrap();
    item0.try_insert("email", DataValue::String("user1@example.com".to_string())).unwrap();

    let mut item1 = Document::new();
    item1.try_insert("id", DataValue::Number(ExactNumber::Int(2))).unwrap();
    item1.try_insert("email", DataValue::String("invalid-email-address".to_string())).unwrap();

    let input = DataValue::Array(vec![DataValue::Document(item0), DataValue::Document(item1)]);

    let err = array_schema.parse(&input).unwrap_err();
    assert_eq!(err.errors.len(), 1);
    assert_eq!(err.errors[0].code, "regex_mismatch");
    assert_eq!(
        err.errors[0].path_string(), "1.email",
        "Array item error path must include element index '1' followed by property 'email'"
    );

    // Deep nested array of arrays: matrix
    let matrix_schema = s.array(s.array(s.int())).build();
    let matrix_input = DataValue::Array(vec![
        DataValue::Array(vec![DataValue::Number(ExactNumber::Int(1)), DataValue::Number(ExactNumber::Int(2))]),
        DataValue::Array(vec![DataValue::Number(ExactNumber::Int(3)), DataValue::String("not_an_int".to_string())]),
    ]);

    let err = matrix_schema.parse(&matrix_input).unwrap_err();
    assert_eq!(err.errors[0].code, "invalid_type");
    assert_eq!(
        err.errors[0].path_string(), "1.1",
        "Matrix error path must be row.col index '1.1'"
    );
}

// ============================================================================
// 9. Discriminated Union, Record, Refine, Transform
// ============================================================================

#[test]
fn test_discriminated_union() {
    let circle_schema = s
        .object()
        .field("kind", s.literal(DataValue::String("circle".to_string())))
        .field("radius", s.float().min(0.0))
        .build();

    let square_schema = s
        .object()
        .field("kind", s.literal(DataValue::String("square".to_string())))
        .field("side", s.float().min(0.0))
        .build();

    let shape_schema = s.discriminated_union("kind", vec![circle_schema, square_schema]);

    // Valid circle
    let mut circle_doc = Document::new();
    circle_doc.try_insert("kind", DataValue::String("circle".to_string())).unwrap();
    circle_doc.try_insert("radius", DataValue::Number(ExactNumber::Float(5.0))).unwrap();
    assert!(shape_schema.parse(&DataValue::Document(circle_doc)).is_ok());

    // Valid square
    let mut square_doc = Document::new();
    square_doc.try_insert("kind", DataValue::String("square".to_string())).unwrap();
    square_doc.try_insert("side", DataValue::Number(ExactNumber::Float(10.0))).unwrap();
    assert!(shape_schema.parse(&DataValue::Document(square_doc)).is_ok());

    // Unknown discriminator value
    let mut triangle_doc = Document::new();
    triangle_doc.try_insert("kind", DataValue::String("triangle".to_string())).unwrap();
    let err = shape_schema.parse(&DataValue::Document(triangle_doc)).unwrap_err();
    assert_eq!(err.errors[0].code, "invalid_discriminator");

    // Missing discriminator property
    let empty_doc = Document::new();
    let err = shape_schema.parse(&DataValue::Document(empty_doc)).unwrap_err();
    assert_eq!(err.errors[0].code, "missing_discriminator");
    assert_eq!(err.errors[0].path_string(), "kind");

    // Non-document input to discriminated union
    let err = shape_schema.parse(&DataValue::String("not_a_doc".to_string())).unwrap_err();
    assert_eq!(err.errors[0].code, "invalid_type");
}

#[test]
fn test_record_validation() {
    let record_schema = s.record(s.string().min(2), s.int().min(0));

    let mut doc = Document::new();
    doc.try_insert("alpha", DataValue::Number(ExactNumber::Int(10))).unwrap();
    doc.try_insert("beta", DataValue::Number(ExactNumber::Int(20))).unwrap();
    assert!(record_schema.parse(&DataValue::Document(doc)).is_ok());

    // Invalid value
    let mut doc_bad_val = Document::new();
    doc_bad_val.try_insert("score", DataValue::Number(ExactNumber::Int(-5))).unwrap();
    let err = record_schema.parse(&DataValue::Document(doc_bad_val)).unwrap_err();
    assert_eq!(err.errors[0].code, "too_small");
    assert_eq!(err.errors[0].path_string(), "score");

    // Invalid key (key length < 2)
    let mut doc_bad_key = Document::new();
    doc_bad_key.try_insert("k", DataValue::Number(ExactNumber::Int(1))).unwrap();
    let err = record_schema.parse(&DataValue::Document(doc_bad_key)).unwrap_err();
    assert_eq!(err.errors[0].code, "too_small");
    assert_eq!(err.errors[0].path_string(), "k");

    // Non-document
    let err = record_schema.parse(&DataValue::Bool(false)).unwrap_err();
    assert_eq!(err.errors[0].code, "invalid_type");
}

#[test]
fn test_refine_and_transform() {
    // Refine: positive even integer
    let even_schema = s.int().refine(|val| match val {
        DataValue::Number(ExactNumber::Int(i)) => *i % 2 == 0,
        _ => false,
    });

    assert!(even_schema.parse(&DataValue::Number(ExactNumber::Int(4))).is_ok());
    let err = even_schema.parse(&DataValue::Number(ExactNumber::Int(5))).unwrap_err();
    assert_eq!(err.errors[0].code, "custom_refinement");
    assert_eq!(err.errors[0].message, "Value failed refinement predicate");

    // Refine with custom message
    let custom_msg_schema = s.int().refine_with_message(
        |val| match val {
            DataValue::Number(ExactNumber::Int(i)) => *i > 0,
            _ => false,
        },
        "Must be strictly positive",
    );
    let err = custom_msg_schema.parse(&DataValue::Number(ExactNumber::Int(-1))).unwrap_err();
    assert_eq!(err.errors[0].message, "Must be strictly positive");

    // Refine on invalid inner type: fails inner validation, does not invoke predicate
    let err = even_schema.parse(&DataValue::String("not_int".to_string())).unwrap_err();
    assert_eq!(err.errors[0].code, "invalid_type");

    // Transform: uppercase string
    let upper_schema = s.string().transform(|val| match val {
        DataValue::String(val_str) => DataValue::String(val_str.to_uppercase()),
        other => other,
    });
    let transformed = upper_schema.parse(&DataValue::String("hello world".to_string())).unwrap();
    assert_eq!(transformed, DataValue::String("HELLO WORLD".to_string()));

    // Transform on invalid inner type: fails inner, transform fn not run
    let err = upper_schema.parse(&DataValue::Number(ExactNumber::Int(1))).unwrap_err();
    assert_eq!(err.errors[0].code, "invalid_type");
}

// ============================================================================
// 10. Struct Schema Binding & Derivation Coverage
// ============================================================================

#[test]
fn test_struct_schema_binding_full_coverage() {
    // 10a. Derive from layout with all canonical type tags
    let fields = vec![
        LogicalFieldDesc::new(0, "f_bool", CanonicalTypeTag::Bool),
        LogicalFieldDesc::new(1, "f_int", CanonicalTypeTag::Int64),
        LogicalFieldDesc::new(2, "f_uint", CanonicalTypeTag::UInt64),
        LogicalFieldDesc::new(3, "f_float", CanonicalTypeTag::Float64),
        LogicalFieldDesc::new(4, "f_string", CanonicalTypeTag::StringSlice),
        LogicalFieldDesc::new(5, "f_binary", CanonicalTypeTag::BinarySlice),
        LogicalFieldDesc::new(6, "f_oid", CanonicalTypeTag::ObjectId),
        LogicalFieldDesc::new(7, "f_datetime", CanonicalTypeTag::DateTimeUtc),
        LogicalFieldDesc::new(8, "f_decimal", CanonicalTypeTag::Decimal128),
        LogicalFieldDesc::new(9, "f_nested", CanonicalTypeTag::NestedStruct(999)).with_nullability(true, true),
    ];

    let layout = LogicalLayoutV1::new_with_abi(5001, "AllTypesStruct", fields, TargetAbi::x86_64());
    let binding = StructSchemaBinding::from_layout(layout.clone());

    assert_eq!(binding.get_field_by_ordinal(0).unwrap().name, "f_bool");
    assert_eq!(binding.get_field_by_ordinal(99), None);
    assert_eq!(binding.get_field_by_name("f_binary").unwrap().ordinal, 5);
    assert_eq!(binding.get_field_by_name("unknown"), None);

    // Test Schema Deref on binding
    assert!(matches!(binding.to_ir(), SchemaIR::Object { .. }));
    let schema_from_binding: Schema = binding.clone().into();
    assert_eq!(schema_from_binding.to_ir(), binding.to_ir());

    // 10b. DeriveSchemaTarget implementations
    let s1 = layout.derive_schema();
    let s2 = (&layout).derive_schema();
    assert_eq!(s1.to_ir(), s2.to_ir());
    let s3 = derive_schema(&layout);
    assert_eq!(s1.to_ir(), s3.to_ir());

    // 10c. Derive from StructDef
    let def = StructDef {
        name: "Simple".to_string(),
        type_params: Vec::new(),
        trait_bounds: Vec::new(),
        fields: vec![
            StructField {
                name: "field1".to_string(),
                ty: Type::new(TypeKind::Int, Span::dummy()),
                visibility: Visibility::Public,
                ownership: FieldOwnership::Strong,
                weak_target: None,
            },
        ],
        visibility: Visibility::Public,
        doc: None,
    };
    let binding_def = StructSchemaBinding::from_struct_def(&def);
    assert_eq!(binding_def.layout.type_name, "Simple");
    let s_def1 = def.derive_schema();
    let s_def2 = (&def).derive_schema();
    assert_eq!(s_def1.to_ir(), s_def2.to_ir());
    assert_eq!(derive_schema_from_struct_def(&def).to_ir(), s_def1.to_ir());

    // 10d. Derive from Item
    let item_struct = Item::new(ItemKind::Struct(def.clone()), Span::dummy());
    let binding_item = StructSchemaBinding::from_item(&item_struct).unwrap();
    assert_eq!(binding_item.layout.type_name, "Simple");
    let s_item1 = item_struct.derive_schema();
    let s_item2 = (&item_struct).derive_schema();
    assert_eq!(s_item1.to_ir(), s_item2.to_ir());
    assert_eq!(derive_schema_from_item(&item_struct).unwrap().to_ir(), s_item1.to_ir());

    // Item error branch: non-struct Item
    let item_non_struct = Item::new(ItemKind::Script(Vec::new()), Span::dummy());
    assert!(StructSchemaBinding::from_item(&item_non_struct).is_err());
    assert!(derive_schema_from_item(&item_non_struct).is_err());

    // 10e. StructSchemaBuilder fluent methods
    let builder_binding: StructSchemaBinding = s
        .r#struct("FluentStruct")
        .field("id", s.int())
        .field("opt_str", s.string().optional())
        .field("opt_null_str", s.string().nullable().optional())
        .nullable_field("null_only", s.string())
        .optional_field("opt_only", s.bool())
        .field_with_default("default_num", s.int(), DataValue::Number(ExactNumber::Int(99)))
        .strict(true)
        .build();

    assert_eq!(builder_binding.layout.type_name, "FluentStruct");
    assert_eq!(builder_binding.layout.fields.len(), 6);

    // compile() on StructSchemaBuilder
    let ir = s.r#struct("Compiled").field("a", s.int()).compile();
    assert!(matches!(ir, SchemaIR::Object { .. }));

    // From<StructSchemaBuilder> for Schema
    let builder2 = s.r#struct("FromTest").field("b", s.bool());
    let sch: Schema = builder2.into();
    assert!(matches!(sch.to_ir(), SchemaIR::Object { .. }));
}

// ============================================================================
// 11. Schema IR, Error, Effect and Builders Coverage
// ============================================================================

#[test]
fn test_schema_error_display_and_conversions() {
    // ValidationError
    let err1 = ValidationError::new(vec!["root".to_string(), "leaf".to_string()], "is invalid", "err_code");
    assert_eq!(err1.path_string(), "root.leaf");
    assert_eq!(format!("{}", err1), "[err_code] root.leaf: is invalid");

    let err2 = ValidationError::new(vec![], "global error", "global_code")
        .with_expected("integer")
        .with_received("string");
    assert_eq!(err2.path_string(), "$");
    assert_eq!(format!("{}", err2), "[global_code] $: global error (expected integer, received string)");

    // ValidationErrors
    let errs_empty = ValidationErrors::new(vec![]);
    assert!(errs_empty.is_empty());
    assert_eq!(errs_empty.len(), 0);

    let errs_multi = ValidationErrors::new(vec![err1.clone(), err2.clone()]);
    assert!(!errs_multi.is_empty());
    assert_eq!(errs_multi.len(), 2);
    let multi_str = format!("{}", errs_multi);
    assert!(multi_str.contains("[err_code] root.leaf: is invalid"));
    assert!(multi_str.contains("[global_code] $: global error"));

    // From conversions
    let from_single: ValidationErrors = err1.clone().into();
    assert_eq!(from_single.len(), 1);

    let back_to_single: ValidationError = from_single.into();
    assert_eq!(back_to_single.code, "err_code");

    let empty_back: ValidationError = errs_empty.into();
    assert_eq!(empty_back.code, "validation_failed");
}

#[test]
fn test_effect_representation() {
    let succ: Effect<i32, String> = Effect::success(42);
    assert!(succ.is_success());
    assert!(!succ.is_failure());
    assert_eq!(succ.into_result().unwrap(), 42);

    let fail: Effect<i32, String> = Effect::failure("failed".to_string());
    assert!(!fail.is_success());
    assert!(fail.is_failure());
    assert_eq!(fail.clone().into_result().unwrap_err(), "failed");

    // into_exit and From<Exit>
    let exit = fail.into_exit();
    let reconstructed: Effect<i32, String> = Effect::from(exit);
    assert!(reconstructed.is_failure());

    // Schema::validate and Schema::validate_single
    let sch = s.int().min(10).build();
    let eff_ok = sch.validate(&DataValue::Number(ExactNumber::Int(15)));
    assert!(eff_ok.is_success());

    let eff_err = sch.validate(&DataValue::Number(ExactNumber::Int(5)));
    assert!(eff_err.is_failure());

    let eff_single_ok = sch.validate_single(&DataValue::Number(ExactNumber::Int(15)));
    assert!(eff_single_ok.is_success());

    let eff_single_err = sch.validate_single(&DataValue::Number(ExactNumber::Int(5)));
    assert!(eff_single_err.is_failure());
}

#[test]
fn test_schema_ir_variants_debug_and_eq() {
    // RefinePredicate debug and eq
    let pred1 = RefinePredicate { f: Arc::new(|_| true) };
    let pred2 = pred1.clone();
    let pred3 = RefinePredicate { f: Arc::new(|_| true) };
    assert_eq!(format!("{:?}", pred1), "RefinePredicate(<fn>)");
    assert_eq!(pred1, pred2);
    assert_ne!(pred1, pred3);

    // TransformFn debug and eq
    let tf1 = TransformFn { f: Arc::new(|v| v) };
    let tf2 = tf1.clone();
    let tf3 = TransformFn { f: Arc::new(|v| v) };
    assert_eq!(format!("{:?}", tf1), "TransformFn(<fn>)");
    assert_eq!(tf1, tf2);
    assert_ne!(tf1, tf3);

    // ObjectFieldIR debug and eq
    let of1 = ObjectFieldIR {
        name: "foo".to_string(),
        schema: SchemaIR::Int { min: None, max: None },
        optional: false,
        default: None,
    };
    let of2 = of1.clone();
    assert_eq!(of1, of2);
    assert!(format!("{:?}", of1).contains("foo"));

    // All SchemaIR variants debug, clone and eq
    let variants = vec![
        SchemaIR::Any,
        SchemaIR::String { min_len: Some(1), max_len: Some(10), regex: Some("abc".to_string()) },
        SchemaIR::Int { min: Some(1), max: Some(10) },
        SchemaIR::Float { min: Some(1.0), max: Some(10.0) },
        SchemaIR::Bool,
        SchemaIR::Binary { min_len: Some(1), max_len: Some(10) },
        SchemaIR::ObjectId,
        SchemaIR::DateTimeUtc,
        SchemaIR::Decimal128,
        SchemaIR::Literal(DataValue::Bool(true)),
        SchemaIR::Array { item: Box::new(SchemaIR::Int { min: None, max: None }), min_items: Some(1), max_items: Some(5) },
        SchemaIR::Object { fields: vec![of1], strict: true },
        SchemaIR::Record { key: Box::new(SchemaIR::String { min_len: None, max_len: None, regex: None }), value: Box::new(SchemaIR::Int { min: None, max: None }) },
        SchemaIR::Union(vec![SchemaIR::Int { min: None, max: None }, SchemaIR::Bool]),
        SchemaIR::DiscriminatedUnion { discriminator: "kind".to_string(), variants: vec![] },
        SchemaIR::Optional(Box::new(SchemaIR::Bool)),
        SchemaIR::Nullable(Box::new(SchemaIR::Bool)),
        SchemaIR::Refine { inner: Box::new(SchemaIR::Bool), predicate: pred1, message: Some("err".to_string()) },
        SchemaIR::Transform { inner: Box::new(SchemaIR::Bool), transform: tf1 },
        SchemaIR::InvalidRegex { pattern: "[a-".to_string(), error: "unclosed".to_string() },
    ];

    for var in variants {
        let cloned = var.clone();
        assert_eq!(var, cloned);
        let dbg = format!("{:?}", var);
        assert!(!dbg.is_empty());
    }
}

#[test]
fn test_all_builders_fluent_api() {
    // Schema::min and Schema::max on various types
    let sch_str = Schema::new(SchemaIR::String { min_len: None, max_len: None, regex: None }).min(3).max(10);
    assert_eq!(sch_str.to_ir(), &SchemaIR::String { min_len: Some(3), max_len: Some(10), regex: None });

    let sch_int = Schema::new(SchemaIR::Int { min: None, max: None }).min(3).max(10);
    assert_eq!(sch_int.to_ir(), &SchemaIR::Int { min: Some(3), max: Some(10) });

    let sch_float = Schema::new(SchemaIR::Float { min: None, max: None }).min(3).max(10);
    assert_eq!(sch_float.to_ir(), &SchemaIR::Float { min: Some(3.0), max: Some(10.0) });

    let sch_arr = Schema::new(SchemaIR::Array { item: Box::new(SchemaIR::Bool), min_items: None, max_items: None }).min(3).max(10);
    assert_eq!(sch_arr.to_ir(), &SchemaIR::Array { item: Box::new(SchemaIR::Bool), min_items: Some(3), max_items: Some(10) });

    let sch_bin = Schema::new(SchemaIR::Binary { min_len: None, max_len: None }).min(3).max(10);
    assert_eq!(sch_bin.to_ir(), &SchemaIR::Binary { min_len: Some(3), max_len: Some(10) });

    // compile() on Schema
    assert_eq!(sch_str.compile(), SchemaIR::String { min_len: Some(3), max_len: Some(10), regex: None });

    // StringBuilder
    let sb = StringBuilder::new().min(1).max(5).regex("^[0-9]+$");
    let compiled_sb = sb.compile();
    assert!(matches!(compiled_sb, SchemaIR::String { min_len: Some(1), max_len: Some(5), .. }));

    let sb_refine = StringBuilder::new().refine(|_| true);
    assert!(matches!(sb_refine.to_ir(), SchemaIR::Refine { .. }));

    let sb_refine_msg = StringBuilder::new().refine_with_message(|_| true, "msg");
    assert!(matches!(sb_refine_msg.to_ir(), SchemaIR::Refine { .. }));

    let sb_transform = StringBuilder::new().transform(|v| v);
    assert!(matches!(sb_transform.to_ir(), SchemaIR::Transform { .. }));

    let sb_into: Schema = StringBuilder::new().into();
    assert!(matches!(sb_into.to_ir(), SchemaIR::String { .. }));

    // IntBuilder
    let ib = IntBuilder::new().min(1).max(5);
    let compiled_ib = ib.compile();
    assert!(matches!(compiled_ib, SchemaIR::Int { min: Some(1), max: Some(5) }));

    let ib_refine = IntBuilder::new().refine(|_| true);
    assert!(matches!(ib_refine.to_ir(), SchemaIR::Refine { .. }));

    let ib_refine_msg = IntBuilder::new().refine_with_message(|_| true, "msg");
    assert!(matches!(ib_refine_msg.to_ir(), SchemaIR::Refine { .. }));

    let ib_transform = IntBuilder::new().transform(|v| v);
    assert!(matches!(ib_transform.to_ir(), SchemaIR::Transform { .. }));

    let ib_into: Schema = IntBuilder::new().into();
    assert!(matches!(ib_into.to_ir(), SchemaIR::Int { .. }));

    // FloatBuilder
    let fb = FloatBuilder::new().min(1.0).max(5.0);
    let compiled_fb = fb.compile();
    assert!(matches!(compiled_fb, SchemaIR::Float { min: Some(1.0), max: Some(5.0) }));

    let fb_refine = FloatBuilder::new().refine(|_| true);
    assert!(matches!(fb_refine.to_ir(), SchemaIR::Refine { .. }));

    let fb_refine_msg = FloatBuilder::new().refine_with_message(|_| true, "msg");
    assert!(matches!(fb_refine_msg.to_ir(), SchemaIR::Refine { .. }));

    let fb_transform = FloatBuilder::new().transform(|v| v);
    assert!(matches!(fb_transform.to_ir(), SchemaIR::Transform { .. }));

    let fb_into: Schema = FloatBuilder::new().into();
    assert!(matches!(fb_into.to_ir(), SchemaIR::Float { .. }));

    // ArrayBuilder
    let ab = ArrayBuilder::new(s.bool()).min(1).max(5);
    let compiled_ab = ab.compile();
    assert!(matches!(compiled_ab, SchemaIR::Array { min_items: Some(1), max_items: Some(5), .. }));

    let ab_refine = ArrayBuilder::new(s.bool()).refine(|_| true);
    assert!(matches!(ab_refine.to_ir(), SchemaIR::Refine { .. }));

    let ab_refine_msg = ArrayBuilder::new(s.bool()).refine_with_message(|_| true, "msg");
    assert!(matches!(ab_refine_msg.to_ir(), SchemaIR::Refine { .. }));

    let ab_transform = ArrayBuilder::new(s.bool()).transform(|v| v);
    assert!(matches!(ab_transform.to_ir(), SchemaIR::Transform { .. }));

    let ab_into: Schema = ArrayBuilder::new(s.bool()).into();
    assert!(matches!(ab_into.to_ir(), SchemaIR::Array { .. }));

    // BinaryBuilder
    let bb = BinaryBuilder::new().min(1).max(5);
    let compiled_bb = bb.compile();
    assert!(matches!(compiled_bb, SchemaIR::Binary { min_len: Some(1), max_len: Some(5) }));

    let bb_refine = BinaryBuilder::new().refine(|_| true);
    assert!(matches!(bb_refine.to_ir(), SchemaIR::Refine { .. }));

    let bb_refine_msg = BinaryBuilder::new().refine_with_message(|_| true, "msg");
    assert!(matches!(bb_refine_msg.to_ir(), SchemaIR::Refine { .. }));

    let bb_transform = BinaryBuilder::new().transform(|v| v);
    assert!(matches!(bb_transform.to_ir(), SchemaIR::Transform { .. }));

    let bb_into: Schema = BinaryBuilder::new().into();
    assert!(matches!(bb_into.to_ir(), SchemaIR::Binary { .. }));

    // ObjectBuilder
    let ob = ObjectBuilder::new().field("a", s.int()).field_with_default("b", s.bool(), DataValue::Bool(true)).strict();
    let compiled_ob = ob.compile();
    assert!(matches!(compiled_ob, SchemaIR::Object { strict: true, .. }));

    let ob_refine = ObjectBuilder::new().refine(|_| true);
    assert!(matches!(ob_refine.to_ir(), SchemaIR::Refine { .. }));

    let ob_refine_msg = ObjectBuilder::new().refine_with_message(|_| true, "msg");
    assert!(matches!(ob_refine_msg.to_ir(), SchemaIR::Refine { .. }));

    let ob_transform = ObjectBuilder::new().transform(|v| v);
    assert!(matches!(ob_transform.to_ir(), SchemaIR::Transform { .. }));

    let ob_into: Schema = ObjectBuilder::new().into();
    assert!(matches!(ob_into.to_ir(), SchemaIR::Object { .. }));

    // unpack_optional recursive variants
    let (u1, opt1) = unpack_optional(SchemaIR::Optional(Box::new(SchemaIR::Bool)));
    assert_eq!((u1, opt1), (SchemaIR::Bool, true));

    let (u2, opt2) = unpack_optional(SchemaIR::Nullable(Box::new(SchemaIR::Optional(Box::new(SchemaIR::Bool)))));
    assert_eq!((u2, opt2), (SchemaIR::Nullable(Box::new(SchemaIR::Bool)), true));

    let (u3, opt3) = unpack_optional(SchemaIR::Refine {
        inner: Box::new(SchemaIR::Optional(Box::new(SchemaIR::Bool))),
        predicate: RefinePredicate { f: Arc::new(|_| true) },
        message: None,
    });
    assert!(opt3);
    assert!(matches!(u3, SchemaIR::Refine { .. }));

    let (u4, opt4) = unpack_optional(SchemaIR::Transform {
        inner: Box::new(SchemaIR::Optional(Box::new(SchemaIR::Bool))),
        transform: TransformFn { f: Arc::new(|v| v) },
    });
    assert!(opt4);
    assert!(matches!(u4, SchemaIR::Transform { .. }));
}

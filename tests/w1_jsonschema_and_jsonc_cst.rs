//! Wave 1 Track 1C Integration Test: D-JSONSCHEMA + D-CST (TASK-W1-1C-JSONSCHEMA-CST).
//!
//! Verifies:
//! - DS-90: JSON Schema 2020-12 & OpenAPI 3.1 dialect import/export, local $defs/$ref, cycles, remote ref rejection
//! - DS-91: Pattern translation: Lust full-match vs JSON Schema search-mode ("a" rejects vs accepts "ba")
//! - DS-92: Strict fail-closed export (refine, transform, BSON) vs explicit input projections (equivalent: false, omissions)
//! - DS-93: Multi-format byte facades (JSON, JSONC, BSON, YAML, TOON) preserving tagged Parse vs Constraint errors
//! - D-CST: CST trivia preservation, byte-for-byte roundtrip via emit_preserving, budget charging & BudgetError failure

use lust::data::{DataValue, Document, ExactNumber};
use lust::schema::{
    s, ExportOptions, JsonSchemaDialect, ProjectionMode, SchemaExportError, SchemaImportError,
};
use std::sync::Arc;

// ===========================================================================
// DS-90 & DS-91: JSON Schema Subset & Pattern Translation Tests
// ===========================================================================

#[test]
fn test_jsonschema_roundtrip_basic_types() {
    let schema = s
        .object()
        .field("active", s.bool())
        .field("count", s.int().min(1).max(100).build())
        .field("score", s.float().min(0.0).max(10.0).build())
        .field("name", s.string().min(2).max(20).build())
        .build();

    let doc = schema.to_json_schema().expect("Export should succeed");
    assert_eq!(
        doc.get("$schema"),
        Some(&DataValue::String("https://json-schema.org/draft/2020-12/schema".to_string()))
    );
    assert_eq!(doc.get("type"), Some(&DataValue::String("object".to_string())));

    let imported = s.schema_from_json_schema(&doc, None).expect("Import should succeed");

    let mut valid_doc = Document::new();
    valid_doc.try_insert("active", DataValue::Bool(true)).unwrap();
    valid_doc.try_insert("count", DataValue::Number(ExactNumber::Int(5))).unwrap();
    valid_doc.try_insert("score", DataValue::Number(ExactNumber::Float(8.5))).unwrap();
    valid_doc.try_insert("name", DataValue::String("Alice".to_string())).unwrap();

    let val = DataValue::Document(valid_doc);
    assert!(imported.validate(&val).is_success());

    // Invalid score (> 10.0) should fail
    let mut invalid_doc = Document::new();
    invalid_doc.try_insert("active", DataValue::Bool(true)).unwrap();
    invalid_doc.try_insert("count", DataValue::Number(ExactNumber::Int(5))).unwrap();
    invalid_doc.try_insert("score", DataValue::Number(ExactNumber::Float(15.0))).unwrap();
    invalid_doc.try_insert("name", DataValue::String("Alice".to_string())).unwrap();

    assert!(imported.validate(&DataValue::Document(invalid_doc)).is_failure());
}

#[test]
fn test_pattern_mode_distinction_lust_vs_jsonschema() {
    // DA/LV Criterion:
    // Verify JSON Schema import of pattern "a" accepts "ba" (search mode),
    // whereas Lust .regex("a") rejects "ba" (full match).

    // 1. Lust .regex("a") - FullMatch semantics
    let lust_schema = s.string().regex("a").build();
    let res_full = lust_schema.validate(&DataValue::String("ba".to_string()));
    assert!(
        res_full.is_failure(),
        "Lust full-match .regex('a') must reject 'ba'"
    );

    let res_full_exact = lust_schema.validate(&DataValue::String("a".to_string()));
    assert!(
        res_full_exact.is_success(),
        "Lust full-match .regex('a') must accept 'a'"
    );

    // 2. JSON Schema pattern "a" - Search mode semantics
    let mut js_doc = Document::new();
    js_doc.try_insert("type", DataValue::String("string".to_string())).unwrap();
    js_doc.try_insert("pattern", DataValue::String("a".to_string())).unwrap();

    let imported_schema = s
        .schema_from_json_schema(&js_doc, None)
        .expect("Import of pattern 'a' should succeed");

    let res_search = imported_schema.validate(&DataValue::String("ba".to_string()));
    assert!(
        res_search.is_success(),
        "JSON Schema import of pattern 'a' must accept 'ba' in search mode"
    );

    let res_search_none = imported_schema.validate(&DataValue::String("xyz".to_string()));
    assert!(
        res_search_none.is_failure(),
        "JSON Schema import of pattern 'a' must reject string without 'a'"
    );
}

#[test]
fn test_pattern_export_wrapping_and_inline_flags_rejection() {
    // Exporting Lust full-match regex wraps in ^(?:...)$
    let schema = s.string().regex("[0-9]+").build();
    let doc = schema.to_json_schema().expect("Export should succeed");
    assert_eq!(
        doc.get("pattern"),
        Some(&DataValue::String("^(?:[0-9]+)$".to_string()))
    );

    // Exporting pattern with non-portable inline flags fails closed
    let bad_schema = s.string().regex("(?i)[a-z]+").build();
    let err = bad_schema.to_json_schema().unwrap_err();
    assert!(matches!(err, SchemaExportError::NonPortablePattern(_)));
}

#[test]
fn test_openapi_3_1_dialect_and_discriminator() {
    let mut v1 = Document::new();
    v1.try_insert("type", DataValue::String("object".to_string())).unwrap();
    let mut v1_props = Document::new();
    let mut v1_tag = Document::new();
    v1_tag.try_insert("const", DataValue::String("cat".to_string())).unwrap();
    v1_props.try_insert("kind", DataValue::Document(v1_tag)).unwrap();
    v1.try_insert("properties", DataValue::Document(v1_props)).unwrap();

    let mut v2 = Document::new();
    v2.try_insert("type", DataValue::String("object".to_string())).unwrap();
    let mut v2_props = Document::new();
    let mut v2_tag = Document::new();
    v2_tag.try_insert("const", DataValue::String("dog".to_string())).unwrap();
    v2_props.try_insert("kind", DataValue::Document(v2_tag)).unwrap();
    v2.try_insert("properties", DataValue::Document(v2_props)).unwrap();

    let mut root = Document::new();
    root.try_insert("oneOf", DataValue::Array(vec![DataValue::Document(v1), DataValue::Document(v2)])).unwrap();
    let mut disc = Document::new();
    disc.try_insert("propertyName", DataValue::String("kind".to_string())).unwrap();
    root.try_insert("discriminator", DataValue::Document(disc)).unwrap();

    let imported = s.schema_from_json_schema(&root, None).expect("Import of discriminated union should succeed");

    // Export with OpenAPI 3.1 dialect
    let opts = ExportOptions {
        dialect: JsonSchemaDialect::OpenApi3_1,
    };
    let exported = imported.to_json_schema_with_options(&opts).expect("OpenAPI export should succeed");
    assert!(exported.get("$schema").is_none());
    assert!(exported.get("discriminator").is_some());
}

#[test]
fn test_local_defs_and_ref_resolution() {
    let mut defs = Document::new();
    let mut user_def = Document::new();
    user_def.try_insert("type", DataValue::String("string".to_string())).unwrap();
    user_def.try_insert("minLength", DataValue::Number(ExactNumber::Int(3))).unwrap();
    defs.try_insert("User", DataValue::Document(user_def)).unwrap();

    let mut root = Document::new();
    root.try_insert("$defs", DataValue::Document(defs)).unwrap();
    root.try_insert("$ref", DataValue::String("#/$defs/User".to_string())).unwrap();

    let imported = s.schema_from_json_schema(&root, None).expect("Import with $ref should succeed");
    assert!(imported.validate(&DataValue::String("Alice".to_string())).is_success());
    assert!(imported.validate(&DataValue::String("Al".to_string())).is_failure());
}

#[test]
fn test_remote_ref_and_cyclic_ref_rejection() {
    // Remote ref rejected fail-closed
    let mut remote_root = Document::new();
    remote_root.try_insert("$ref", DataValue::String("https://example.com/schema.json".to_string())).unwrap();
    let err = s.schema_from_json_schema(&remote_root, None).unwrap_err();
    assert!(matches!(err, SchemaImportError::UnsupportedRemoteRef(_)));

    // Cyclic ref detected and rejected
    let mut defs = Document::new();
    let mut loop_a = Document::new();
    loop_a.try_insert("$ref", DataValue::String("#/$defs/LoopA".to_string())).unwrap();
    defs.try_insert("LoopA", DataValue::Document(loop_a)).unwrap();

    let mut cyclic_root = Document::new();
    cyclic_root.try_insert("$defs", DataValue::Document(defs)).unwrap();
    cyclic_root.try_insert("$ref", DataValue::String("#/$defs/LoopA".to_string())).unwrap();

    let err2 = s.schema_from_json_schema(&cyclic_root, None).unwrap_err();
    assert!(matches!(err2, SchemaImportError::CyclicRef(_)));
}

#[test]
fn test_additional_properties_import_vs_export_semantics() {
    // On import: absent additionalProperties means passthrough (!strict)
    let mut obj_doc = Document::new();
    obj_doc.try_insert("type", DataValue::String("object".to_string())).unwrap();
    let props = Document::new();
    obj_doc.try_insert("properties", DataValue::Document(props)).unwrap();

    let imported = s.schema_from_json_schema(&obj_doc, None).unwrap();
    let mut extra = Document::new();
    extra.try_insert("unknown_field", DataValue::String("allowed".to_string())).unwrap();
    assert!(
        imported.validate(&DataValue::Document(extra)).is_success(),
        "Absent additionalProperties on import should allow unknown fields"
    );

    // .strict() exports additionalProperties: false
    let strict_schema = s.object().strict().build();
    let exported = strict_schema.to_json_schema().unwrap();
    assert_eq!(exported.get("additionalProperties"), Some(&DataValue::Bool(false)));
}

// ===========================================================================
// DS-92: Strict Fail-Closed Export & Explicit Projections
// ===========================================================================

#[test]
fn test_strict_export_fail_closed_on_refine_and_transform() {
    // DA/LV Criterion:
    // Verify strict to_json_schema on a schema with .refine(...) or .transform(...)
    // fails closed with NonPortableConstraint / NonPortableTransform, while
    // export_json_schema_projection(ProjectionMode::Input) succeeds with
    // equivalent: false and non-empty omissions.

    // 1. Refine callback
    let schema_refine = s
        .string()
        .min(2)
        .refine_with_message(|v| match v {
            DataValue::String(value) => value.starts_with('A'),
            _ => false,
        }, "Starts with A");
        

    let err_refine = schema_refine.to_json_schema().unwrap_err();
    assert!(matches!(err_refine, SchemaExportError::NonPortableConstraint(_)));

    let proj_refine = schema_refine
        .export_json_schema_projection(ProjectionMode::Input)
        .expect("Input projection should succeed");
    assert!(!proj_refine.equivalent, "Equivalent must be false when omissions present");
    assert_eq!(proj_refine.omissions.len(), 1);
    assert_eq!(proj_refine.omissions[0].constraint, "refine");
    assert_eq!(proj_refine.projection, ProjectionMode::Input);
    assert_eq!(proj_refine.document.get("type"), Some(&DataValue::String("string".to_string())));

    // 2. Transform stage
    let schema_transform = s
        .string()
        .min(1)
        .transform(|v| match v {
            DataValue::String(value) => DataValue::String(value.to_uppercase()),
            other => other,
        });
        

    let err_transform = schema_transform.to_json_schema().unwrap_err();
    assert!(matches!(err_transform, SchemaExportError::NonPortableTransform(_)));

    let proj_transform = schema_transform
        .export_json_schema_projection(ProjectionMode::Input)
        .expect("Input projection should succeed");
    assert!(!proj_transform.equivalent);
    assert_eq!(proj_transform.omissions.len(), 1);
    assert_eq!(proj_transform.omissions[0].constraint, "transform");

    // Output projection on uninvertible transform fails closed
    let err_out_proj = schema_transform
        .export_json_schema_projection(ProjectionMode::Output)
        .unwrap_err();
    assert!(matches!(err_out_proj, SchemaExportError::NonPortableTransform(_)));
}

#[test]
fn test_strict_export_fail_closed_on_extended_bson() {
    let schema_oid = s.object_id();
    let err = schema_oid.to_json_schema().unwrap_err();
    assert!(matches!(err, SchemaExportError::UnsupportedRepresentation(_)));

    let proj = schema_oid
        .export_json_schema_projection(ProjectionMode::Input)
        .expect("Projection should succeed with omissions");
    assert!(!proj.equivalent);
    assert_eq!(proj.omissions[0].constraint, "ObjectId");
}

// ===========================================================================
// DS-93: Multi-Format Byte Facades
// ===========================================================================

#[test]
fn test_multiformat_byte_facades_parse_vs_constraint_errors() {
    let schema = s
        .object()
        .field("title", s.string().min(3).build())
        .field("views", s.int().min(0).build())
        .build();

    // 1. JSON
    let valid_json = br#"{"title":"Rust","views":100}"#;
    assert!(schema.validate_json(valid_json).is_success());

    let malformed_json = br#"{"title": "Rust", "views": }"#;
    let err_json_parse = schema.validate_json(malformed_json).into_result().unwrap_err();
    assert!(err_json_parse.has_parse_error());
    assert!(!err_json_parse.has_constraint_error());

    let constraint_fail_json = br#"{"title":"R","views":100}"#;
    let err_json_const = schema.validate_json(constraint_fail_json).into_result().unwrap_err();
    assert!(!err_json_const.has_parse_error());
    assert!(err_json_const.has_constraint_error());

    // 2. JSONC (comments and trailing comma allowed)
    let valid_jsonc = b"{\n  // comment\n  \"title\": \"Rust\",\n  \"views\": 100,\n}";
    assert!(schema.validate_jsonc(valid_jsonc).is_success());

    let malformed_jsonc = b"{ \"title\": /* unclosed comment }";
    let err_jsonc_parse = schema.validate_jsonc(malformed_jsonc).into_result().unwrap_err();
    assert!(err_jsonc_parse.has_parse_error());

    // 3. YAML
    let valid_yaml = b"title: Rust\nviews: 100\n";
    assert!(schema.validate_yaml(valid_yaml).is_success());

    let malformed_yaml = b"title: Rust\nviews: !unsupported value\n";
    let err_yaml_parse = schema.validate_yaml(malformed_yaml).into_result().unwrap_err();
    assert!(err_yaml_parse.has_parse_error());

    // 4. TOON
    let malformed_toon = b"";
    let err_toon_parse = schema.validate_toon(malformed_toon).into_result().unwrap_err();
    assert!(err_toon_parse.has_parse_error());

    // 5. BSON
    let mut doc = Document::new();
    doc.try_insert("title", DataValue::String("Rust".to_string())).unwrap();
    doc.try_insert("views", DataValue::Number(ExactNumber::Int(100))).unwrap();
    let mut encoder = lust::data::bson::BsonEncoder::new();
    let valid_bson = encoder.encode_document(&doc);
    assert!(schema.validate_bson(&valid_bson).is_success());

    let truncated_bson = &valid_bson[..valid_bson.len() - 3];
    let err_bson_parse = schema.validate_bson(truncated_bson).into_result().unwrap_err();
    assert!(err_bson_parse.has_parse_error());
}

// ===========================================================================
// D-CST: Concrete Syntax Tree Trivia Preservation & Budget Charging
// ===========================================================================

#[cfg(feature = "jsonc_cst")]
#[test]
fn test_cst_trivia_emit_preserving_byte_for_byte_identity() {
    use lust::data::jsonc::{emit_preserving, parse_preserving};

    // DA/LV Criterion:
    // Verify emit_preserving(parse_preserving(bytes)) reconstructs bytes
    // from CST spans + TriviaSpans and matches input byte-for-byte.

    let input = b"// Config header\n{\n  /* settings */\n  \"name\": \"test\", // inline\n  \"items\": [ 1, 2, ], // trailing comma\n}\n// EOF comment\n";
    let doc = parse_preserving(input).expect("parse_preserving should succeed with jsonc_cst feature");
    let emitted = emit_preserving(&doc).expect("emit_preserving should succeed");

    assert_eq!(emitted.len(), input.len());
    assert_eq!(&emitted[..], &input[..]);
}

#[cfg(feature = "jsonc_cst")]
#[test]
fn test_cst_parse_with_budget_and_failure_on_exceeded() {
    use lust::data::budget::{BudgetError, BudgetTracker};
    use lust::data::jsonc::{parse_cst_with_budget, JsoncError};

    let input: Arc<[u8]> = Arc::from(b"{\n  \"a\": 1,\n  \"b\": [2, 3]\n}".as_slice());

    // 1. Adequate budget succeeds
    let mut good_budget = BudgetTracker::new(65536, 1000, 32);
    let cst = parse_cst_with_budget(input.clone(), &mut good_budget).expect("Parse should succeed with adequate budget");
    let emitted = cst.emit_preserving().expect("Emit should match");
    assert_eq!(&emitted[..], &input[..]);

    // 2. DA/LV Criterion:
    // parse_cst_with_budget fails with BudgetError when BudgetTracker limit is exceeded.
    let mut tiny_budget = BudgetTracker::new(10, 1000, 32); // 10 bytes limit
    let err = parse_cst_with_budget(input, &mut tiny_budget).unwrap_err();
    assert!(
        matches!(err, JsoncError::Budget(BudgetError::BytesExceeded { .. })),
        "Must fail with BudgetError::BytesExceeded when memory budget is exceeded"
    );
}

#[cfg(not(feature = "jsonc_cst"))]
#[test]
fn test_cst_feature_gate_disabled_by_default() {
    use lust::data::budget::BudgetTracker;
    use lust::data::jsonc::{parse_cst_with_budget, parse_preserving, JsoncError};

    let input = b"{\n  \"a\": 1\n}";
    let res = parse_preserving(input);
    assert!(
        matches!(res, Err(JsoncError::UnsupportedFeature(_))),
        "parse_preserving must return UnsupportedFeature when jsonc_cst is disabled"
    );

    let mut budget = BudgetTracker::new(65536, 1000, 32);
    let res_cst = parse_cst_with_budget(Arc::from(input.as_slice()), &mut budget);
    assert!(
        matches!(res_cst, Err(JsoncError::UnsupportedFeature(_))),
        "parse_cst_with_budget must return UnsupportedFeature when jsonc_cst is disabled"
    );
}

//! Example: Schema Definition and Validation via the `s` builder API.
//! Demonstrates constructing structured schemas with integer/string constraints,
//! parsing DataValue documents, optional fields, strict object checking, and fail-closed validation.

use lust::data::{DataValue, Document, ExactNumber};
use lust::schema::s;

fn main() {
    println!("--- [Schema Validation Example] ---");

    // 1. Build a user profile schema with fluent constraints
    let user_schema = s
        .object()
        .field("id", s.int().min(1).max(1_000_000).build())
        .field("username", s.string().min(3).max(32).build())
        .field("age", s.int().min(0).max(150).build().optional())
        .strict()
        .build();

    // 2. Validate valid user document
    let mut valid_doc = Document::new();
    valid_doc
        .try_insert("id", DataValue::Number(ExactNumber::Int(1042)))
        .unwrap();
    valid_doc
        .try_insert("username", DataValue::String("alex_dev".to_string()))
        .unwrap();
    valid_doc
        .try_insert("age", DataValue::Number(ExactNumber::Int(28)))
        .unwrap();

    let valid_val = DataValue::Document(valid_doc);
    let parse_res = user_schema.parse(&valid_val);
    assert!(parse_res.is_ok(), "Valid document must pass schema validation");
    println!("✓ Valid document passed validation");

    // 3. Test invalid document: id out of range (min 1)
    let mut out_of_bounds_doc = Document::new();
    out_of_bounds_doc
        .try_insert("id", DataValue::Number(ExactNumber::Int(0)))
        .unwrap();
    out_of_bounds_doc
        .try_insert("username", DataValue::String("alex_dev".to_string()))
        .unwrap();

    let invalid_res = user_schema.parse(&DataValue::Document(out_of_bounds_doc));
    assert!(invalid_res.is_err(), "Out-of-bounds id must be rejected");
    println!("✓ Out-of-bounds integer correctly rejected: {:?}", invalid_res.err().unwrap());

    // 4. Test invalid document: extra field in strict mode
    let mut extra_field_doc = Document::new();
    extra_field_doc
        .try_insert("id", DataValue::Number(ExactNumber::Int(1042)))
        .unwrap();
    extra_field_doc
        .try_insert("username", DataValue::String("alex_dev".to_string()))
        .unwrap();
    extra_field_doc
        .try_insert("unregistered_extra", DataValue::Bool(true))
        .unwrap();

    let strict_res = user_schema.parse(&DataValue::Document(extra_field_doc));
    assert!(strict_res.is_err(), "Strict schema must reject unrecognized fields");
    println!("✓ Strict field violation correctly rejected: {:?}", strict_res.err().unwrap());

    println!("All schema validation assertions passed!");
}

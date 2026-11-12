//! Example: Strict RFC 8259 JSON Deserialization and Fail-Closed Validation.
//! Demonstrates parsing valid JSON into DataValue documents and fail-closed
//! duplicate key rejection (Gate-SCHEMA-01) via JsonError::DuplicateKey.

use lust::data::json::{self, JsonError};
use lust::data::DataValue;

fn main() {
    println!("--- [JSON Deserialization Example] ---");

    // 1. Parse valid JSON document
    let valid_json = r#"{
        "service": "order-gateway",
        "port": 8080,
        "tls_enabled": true,
        "tags": ["web", "rpc", "production"]
    }"#;

    let parsed = json::from_str(valid_json).expect("Valid JSON must parse successfully");
    println!("✓ Successfully parsed valid JSON document");

    if let DataValue::Document(ref doc) = parsed {
        assert_eq!(doc.len(), 4);
        println!("  - service: {:?}", doc.get("service").unwrap());
        println!("  - port: {:?}", doc.get("port").unwrap());
    } else {
        panic!("Expected DataValue::Document");
    }

    // 2. Gate-SCHEMA-01: Rejection of duplicate object keys (fail-closed)
    let duplicate_key_json = r#"{
        "account_id": "ACC-1001",
        "balance": 500,
        "balance": 999999
    }"#;

    let res = json::from_str(duplicate_key_json);
    match res {
        Err(JsonError::DuplicateKey(key, offset)) => {
            println!(
                "✓ Correctly rejected duplicate key '{}' at offset {} (Gate-SCHEMA-01)",
                key, offset
            );
            assert_eq!(key, "balance");
        }
        other => panic!("Expected JsonError::DuplicateKey, got: {:?}", other),
    }

    // 3. Round-trip serialization
    let json_string = json::to_string(&parsed).expect("JSON serialization succeeds");
    println!("✓ Re-serialized JSON output length: {} bytes", json_string.len());

    println!("All JSON deserialization assertions passed!");
}

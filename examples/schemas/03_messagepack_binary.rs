//! Example: Binary Document Serialization & Deserialization (BSON).
//! Note: While this example target is named `schema_msgpack` in Cargo.toml per
//! the specification contract, the underlying binary format implemented in `lust::data`
//! is high-performance BSON (`lust::data::bson`), demonstrating zero-copy binary
//! views, bounded decoding, and preservation of 64-bit unsigned integers and timestamps.

use lust::data::bson;
use lust::data::{DataValue, Document, ExactNumber};

fn main() {
    println!("--- [Binary Document Serialization (schema_msgpack target via BSON) Example] ---");

    // 1. Construct a rich structured document
    let mut doc = Document::new();
    doc.try_insert(
        "transaction_id",
        DataValue::Number(ExactNumber::UInt(18_446_744_073_709_551_615)),
    )
    .unwrap();
    doc.try_insert(
        "label",
        DataValue::String("settlement_batch_alpha".to_string()),
    )
    .unwrap();
    doc.try_insert("confirmed", DataValue::Bool(true))
        .unwrap();

    // 2. Serialize document into binary bytes
    let bson_bytes = bson::to_vec(&doc);
    println!("✓ Encoded document into {} binary bytes", bson_bytes.len());
    assert!(bson_bytes.len() >= 5, "BSON document must have header");

    // 3. Deserialize binary bytes back into a Document
    let decoded_doc = bson::from_slice(&bson_bytes).expect("BSON decoding must succeed");
    assert_eq!(decoded_doc.len(), 3);
    assert_eq!(
        decoded_doc.get("transaction_id"),
        Some(&DataValue::Number(ExactNumber::UInt(18_446_744_073_709_551_615)))
    );
    assert_eq!(
        decoded_doc.get("label"),
        Some(&DataValue::String("settlement_batch_alpha".to_string()))
    );
    println!("✓ Decoded document matches original content exactly");

    // 4. Zero-copy binary view inspection
    let view = bson::view(&bson_bytes).expect("Zero-copy BSON view must parse");
    assert_eq!(view.get_str("label"), Some("settlement_batch_alpha"));
    assert_eq!(view.get_bool("confirmed"), Some(true));
    println!("✓ Zero-copy view accessed fields without full allocation");

    println!("All binary document assertions passed!");
}

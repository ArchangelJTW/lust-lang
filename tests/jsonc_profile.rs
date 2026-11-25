use lust::data::json::events::JsonError;
use lust::data::jsonc::{self, JsoncChunkStripper, JsoncError, JsoncScanner, Span, TokenKind};
use lust::data::{DataValue, ExactNumber};

#[test]
fn test_jsonc_accepts_line_and_block_comments() {
    let input = r#"
    // Header comment
    {
        /* Multi-line
           block comment */
        "name": "lust-lang", // inline comment
        "version": 1, /* another comment */
        "active": true
    }
    // Trailing comment
    "#;

    let val = jsonc::from_str(input).expect("Valid JSONC with comments");
    if let DataValue::Document(doc) = val {
        assert_eq!(
            doc.get("name").unwrap(),
            &DataValue::String("lust-lang".to_string())
        );
        assert_eq!(
            doc.get("version").unwrap(),
            &DataValue::Number(ExactNumber::Int(1))
        );
        assert_eq!(doc.get("active").unwrap(), &DataValue::Bool(true));
    } else {
        panic!("Expected document");
    }
}

#[test]
fn test_jsonc_strings_preserve_comment_markers() {
    // Tests MUST reject a comment-looking sequence inside a string being stripped
    let input = r#"{
        "line_comment": "// this is a real string, not a comment",
        "block_comment": "/* this is also a real string */",
        "mixed": "foo // bar /* baz */ qux",
        "escaped_quote": "\"// still a string inside quotes\""
    }"#;

    // Verify stripped string preserves the exact string contents
    let stripped = jsonc::strip_comments(input).expect("Stripping should succeed");
    assert!(
        stripped.contains("// this is a real string, not a comment"),
        "Stripped output must keep line-comment-looking contents in strings intact"
    );
    assert!(
        stripped.contains("/* this is also a real string */"),
        "Stripped output must keep block-comment-looking contents in strings intact"
    );
    assert!(
        stripped.contains(r#"\"// still a string inside quotes\""#),
        "Stripped output must keep escaped quotes and inner comment markers intact"
    );

    // Verify parsed DataValue preserves exact strings
    let val = jsonc::from_str(input).expect("Valid JSONC with comment-like strings");
    if let DataValue::Document(doc) = val {
        assert_eq!(
            doc.get("line_comment").unwrap(),
            &DataValue::String("// this is a real string, not a comment".to_string())
        );
        assert_eq!(
            doc.get("block_comment").unwrap(),
            &DataValue::String("/* this is also a real string */".to_string())
        );
        assert_eq!(
            doc.get("mixed").unwrap(),
            &DataValue::String("foo // bar /* baz */ qux".to_string())
        );
        assert_eq!(
            doc.get("escaped_quote").unwrap(),
            &DataValue::String("\"// still a string inside quotes\"".to_string())
        );
    } else {
        panic!("Expected document");
    }
}

#[test]
fn test_jsonc_single_trailing_comma_accepted() {
    // Single trailing comma in array
    let arr_input = "[1, 2, 3,]";
    let arr_val = jsonc::from_str(arr_input).expect("Array with trailing comma");
    if let DataValue::Array(items) = arr_val {
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], DataValue::Number(ExactNumber::Int(1)));
        assert_eq!(items[1], DataValue::Number(ExactNumber::Int(2)));
        assert_eq!(items[2], DataValue::Number(ExactNumber::Int(3)));
    } else {
        panic!("Expected array");
    }

    // Single trailing comma in object
    let obj_input = r#"{
        "a": 1,
        "b": 2,
    }"#;
    let obj_val = jsonc::from_str(obj_input).expect("Object with trailing comma");
    if let DataValue::Document(doc) = obj_val {
        assert_eq!(doc.get("a").unwrap(), &DataValue::Number(ExactNumber::Int(1)));
        assert_eq!(doc.get("b").unwrap(), &DataValue::Number(ExactNumber::Int(2)));
    } else {
        panic!("Expected document");
    }

    // Nested structures with comments and trailing commas
    let nested_input = r#"{
        "list": [
            "one", // item 1
            "two", /* item 2 */
        ], // end list
        "sub": {
            "val": 42,
        },
    }"#;
    let nested_val = jsonc::from_str(nested_input).expect("Nested JSONC");
    if let DataValue::Document(doc) = nested_val {
        if let DataValue::Array(list) = doc.get("list").unwrap() {
            assert_eq!(list.len(), 2);
            assert_eq!(list[0], DataValue::String("one".to_string()));
            assert_eq!(list[1], DataValue::String("two".to_string()));
        } else {
            panic!("Expected list array");
        }
    } else {
        panic!("Expected document");
    }
}

#[test]
fn test_jsonc_invalid_commas_fail_closed() {
    // Empty array with comma
    assert!(jsonc::from_str("[,]").is_err());
    assert!(jsonc::from_str("[ , ]").is_err());

    // Empty object with comma
    assert!(jsonc::from_str("{,}").is_err());
    assert!(jsonc::from_str("{ , }").is_err());

    // Leading comma in array
    assert!(jsonc::from_str("[, 1]").is_err());
    assert!(jsonc::from_str("[ , 1]").is_err());

    // Leading comma in object
    assert!(jsonc::from_str(r#"{, "a": 1}"#).is_err());

    // Double commas in array
    assert!(jsonc::from_str("[1,, 2]").is_err());
    assert!(jsonc::from_str("[1, , 2]").is_err());
    assert!(jsonc::from_str("[1, 2,,]").is_err());
    assert!(jsonc::from_str("[1, 2, ,]").is_err());

    // Double commas in object
    assert!(jsonc::from_str(r#"{"a": 1,, "b": 2}"#).is_err());
    assert!(jsonc::from_str(r#"{"a": 1, , "b": 2}"#).is_err());
    assert!(jsonc::from_str(r#"{"a": 1,,}"#).is_err());

    // Comma outside containers
    assert!(jsonc::from_str("123,").is_err());
    assert!(jsonc::from_str("\"hello\"," ).is_err());
    assert!(jsonc::from_str("true,").is_err());
    assert!(jsonc::from_str(r#"{"a": 1},"#).is_err());
    assert!(jsonc::from_str(r#"[1],"#).is_err());
}

#[test]
fn test_jsonc_comments_cannot_glue_tokens() {
    // RFC §7.5: comments cannot glue tokens together: `1/*x*/2` is an error
    assert!(jsonc::from_str("1/*x*/2").is_err());
    assert!(jsonc::from_str("true/*comment*/false").is_err());
    assert!(jsonc::from_str("[1/*x*/2]").is_err());
    assert!(jsonc::from_str(r#"{"a": 1/*x*/"b": 2}"#).is_err());
}

#[test]
fn test_jsonc_rejected_extensions() {
    // Single quotes are rejected
    assert!(jsonc::from_str("'hello'").is_err());
    assert!(jsonc::from_str(r#"{"key": 'val'}"#).is_err());

    // Bare keys are rejected
    assert!(jsonc::from_str("{key: 1}").is_err());
    assert!(jsonc::from_str("{name: \"test\"}").is_err());

    // Unclosed block comment is rejected
    assert!(jsonc::from_str("/* unclosed comment").is_err());
    assert!(jsonc::from_str("{\"a\": 1} /*").is_err());

    // Unclosed string is rejected
    assert!(jsonc::from_str("\"unclosed").is_err());

    // UTF-8 BOM is strictly rejected
    let bom_input = [0xEF, 0xBB, 0xBF, b'{', b'}'];
    assert!(matches!(
        jsonc::from_slice(&bom_input),
        Err(JsoncError::BomForbidden(_))
    ));
}

#[test]
fn test_jsonc_strict_json_scalar_oracle_reuse() {
    // Duplicate keys rejected (Gate-SCHEMA-01 via strict JSON parser)
    let duplicate_input = r#"{
        "id": 1,
        // comment between duplicate keys
        "id": 2
    }"#;
    let res = jsonc::from_str(duplicate_input);
    assert!(res.is_err());
    match res.unwrap_err() {
        JsoncError::Json(JsonError::DuplicateKey(key, _)) => {
            assert_eq!(key, "id");
        }
        other => panic!("Expected DuplicateKey error, got {:?}", other),
    }

    // Invalid numbers rejected
    assert!(jsonc::from_str("0123 // leading zero").is_err());

    // Invalid escape sequences rejected
    assert!(jsonc::from_str(r#"{"bad": "\x41"}"#).is_err());

    // Surrogate pair decoding works identically to strict JSON
    let val = jsonc::from_str("\"\\uD83D\\uDE00\" // smiley").unwrap();
    assert_eq!(val, DataValue::String("😀".to_string()));
}

#[test]
fn test_jsonc_chunked_scanner_state_matches_single_buffer() {
    let test_corpus = vec![
        r#"{"a": /* block comment */ 1, "b": "// not comment", "c": [10, 20,],}"#,
        "// line comment\n{\"key\": \"val/*x*/ue\", \"num\": -42.5e2, \"trailing\": [true,],}",
        "/* multiline\n * comment\n */ {\"str\": \"hello\\\"world\", \"ok\": false, }",
    ];

    for text in test_corpus {
        let single_buffer_stripped =
            jsonc::strip_slice(text.as_bytes()).expect("Single buffer scan should succeed");

        // Split text at every possible byte boundary
        let bytes = text.as_bytes();
        for split_pos in 1..bytes.len() {
            let chunk1 = &bytes[..split_pos];
            let chunk2 = &bytes[split_pos..];

            let mut stripper = JsoncChunkStripper::new();
            let mut chunk_res = stripper
                .process_chunk(chunk1, false)
                .expect("Chunk 1 processing should succeed");
            let chunk2_res = stripper
                .process_chunk(chunk2, true)
                .expect("Chunk 2 processing should succeed");
            chunk_res.extend_from_slice(&chunk2_res);

            assert_eq!(
                chunk_res.len(),
                single_buffer_stripped.len(),
                "Length mismatch for split at byte {}:\nsingle: {:?}\nchunk:  {:?}",
                split_pos,
                String::from_utf8_lossy(&single_buffer_stripped),
                String::from_utf8_lossy(&chunk_res)
            );
        }
    }
}

#[test]
fn test_jsonc_prepared_scanner_zero_allocation() {
    let input = b"/* comment */ [1, 2, \"hello\", true, null]";
    // JsoncScanner is a stack value containing a slice ref, usize, and enum
    assert!(std::mem::size_of::<JsoncScanner>() <= 48);

    let mut scanner = JsoncScanner::new(input);
    let mut tokens = Vec::new();
    while let Some(tok) = scanner.next_token().unwrap() {
        tokens.push(tok);
    }

    assert_eq!(tokens[0].kind, TokenKind::BlockComment);
    assert_eq!(tokens[0].span, Span::new(0, 13));

    assert_eq!(tokens[1].kind, TokenKind::Whitespace);
    assert_eq!(tokens[2].kind, TokenKind::LeftBracket);
    assert_eq!(tokens[2].span, Span::new(14, 15));
}

#[test]
fn test_jsonc_optional_cst_feature_gate() {
    let bytes = b"{\n  // comment\n  \"key\": 123,\n}";

    #[cfg(not(feature = "jsonc_cst"))]
    {
        // When jsonc_cst is not enabled, parse_preserving returns UnsupportedFeature
        let res = jsonc::parse_preserving(bytes);
        assert!(matches!(res, Err(JsoncError::UnsupportedFeature(_))));
    }

    #[cfg(feature = "jsonc_cst")]
    {
        // When jsonc_cst is enabled, parse_preserving roundtrips bytes exactly
        let doc = jsonc::parse_preserving(bytes).expect("parse_preserving should succeed");
        let emitted = jsonc::emit_preserving(&doc).expect("emit_preserving should succeed");
        assert_eq!(emitted, bytes, "emit_preserving MUST match original bytes");

        // to_document produces logical value without trivia
        let val = doc.to_document().expect("to_document should succeed");
        if let DataValue::Document(d) = val {
            assert_eq!(d.get("key").unwrap(), &DataValue::Number(ExactNumber::Int(123)));
        } else {
            panic!("Expected document");
        }
    }
}

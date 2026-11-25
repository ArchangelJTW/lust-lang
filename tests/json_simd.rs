use lust::data::json::{self, CpuDispatch, JsonError};

fn test_dispatches() -> Vec<CpuDispatch> {
    let mut dispatches = vec![CpuDispatch::Auto, CpuDispatch::Swar, CpuDispatch::Scalar];
    #[cfg(target_arch = "aarch64")]
    {
        dispatches.push(CpuDispatch::Neon);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            dispatches.push(CpuDispatch::Avx2);
        }
    }
    dispatches
}

/// Verification helper: asserts that SIMD and scalar produce 100% identical outputs or errors.
fn assert_simd_scalar_parity(json: &str) {
    let scalar_res = json::from_str(json);

    for dispatch in test_dispatches() {
        let simd_res = json::from_str_simd_with_dispatch(json, dispatch);
        assert_eq!(
            simd_res, scalar_res,
            "SIMD dispatch {:?} diverged from scalar oracle on input: {:?}",
            dispatch, json
        );
    }
}

/// Verification helper with limits: asserts that SIMD and scalar produce identical results.
fn assert_simd_scalar_parity_with_limits(bytes: &[u8], max_depth: usize, max_bytes: usize) {
    let scalar_res = json::from_slice_with_limits(bytes, max_depth, max_bytes);

    for dispatch in test_dispatches() {
        let simd_res = json::from_slice_simd_with_limits_and_dispatch(bytes, max_depth, max_bytes, dispatch);
        assert_eq!(
            simd_res, scalar_res,
            "SIMD dispatch {:?} diverged from scalar oracle with limits (depth={}, bytes={}) on input: {:?}",
            dispatch, max_depth, max_bytes, bytes
        );
    }
}

// ============================================================================
// 1. CPU Dispatch & Detection
// ============================================================================

#[test]
fn test_simd_cpu_dispatch_detection() {
    let detected = CpuDispatch::detect();
    assert!(detected.is_supported());

    let auto_effective = CpuDispatch::Auto.effective();
    assert_eq!(auto_effective, detected);

    #[cfg(target_arch = "aarch64")]
    {
        assert_eq!(detected, CpuDispatch::Neon);
        assert_eq!(CpuDispatch::Neon.chunk_size(), 16);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            assert_eq!(detected, CpuDispatch::Avx2);
            assert_eq!(CpuDispatch::Avx2.chunk_size(), 32);
        }
    }

    assert_eq!(CpuDispatch::Swar.chunk_size(), 8);
    assert_eq!(CpuDispatch::Scalar.chunk_size(), 1);
}

// ============================================================================
// 2. Primitives & Scalar Equivalence
// ============================================================================

#[test]
fn test_simd_scalar_primitives_equivalence() {
    let primitives = [
        "null",
        "true",
        "false",
        "0",
        "-0",
        "42",
        "-100",
        "123456789012345",
        "-98765432109876",
        "3.1415926535",
        "-0.00001",
        "1e6",
        "2.5e-3",
        "-1.25E+4",
        "\"\"",
        "\"hello world\"",
        "\"simd accelerated json parser for lust language\"",
    ];

    for s in primitives {
        assert_simd_scalar_parity(s);
    }
}

#[test]
fn test_simd_unicode_and_escapes_equivalence() {
    let cases = [
        r#""hello \n world \t with \"escapes\" and \\ backslashes""#,
        r#""\u0041\u0042\u0043""#,
        r#""\uD83D\uDE00 surrogate emoji""#,
        r#""multi-byte utf-8: привет мир, 你好世界, 🚀🦀""#,
        r#""\/ solidus escape""#,
        r#""\b\f\r\t\n controls""#,
    ];

    for c in cases {
        assert_simd_scalar_parity(c);
    }
}

// ============================================================================
// 3. Short Inputs (< Vector Chunk Size)
// ============================================================================

#[test]
fn test_simd_short_inputs() {
    // Inputs spanning 0, 1, 2, 3, 4, 5, 6, 7 bytes (all strictly < SWAR (8), NEON (16), AVX2 (32))
    let short_cases = [
        "",
        " ",
        "  \t\r\n",
        "1",
        "-1",
        "{}",
        "[]",
        "\"\"",
        "\"a\"",
        "[1]",
        "[1,2]",
        r#"{"":1}"#,
        "true",
        "null",
    ];

    for sc in short_cases {
        assert_simd_scalar_parity(sc);
    }
}

// ============================================================================
// 4. Chunk Boundaries (Exact Multiple of Vector Sizes: 8, 16, 32 bytes)
// ============================================================================

#[test]
fn test_simd_chunk_boundaries() {
    // Exact 8-byte input
    let eight_bytes = "[1,2,34]";
    assert_eq!(eight_bytes.len(), 8);
    assert_simd_scalar_parity(eight_bytes);

    // Exact 16-byte input
    let sixteen_bytes = r#"{"key":"val123"}"#;
    assert_eq!(sixteen_bytes.len(), 16);
    assert_simd_scalar_parity(sixteen_bytes);

    // Exact 32-byte input
    let thirty_two_bytes = r#"{"name":"lust-lang","simd":true}"#;
    assert_eq!(thirty_two_bytes.len(), 32);
    assert_simd_scalar_parity(thirty_two_bytes);

    // Exact 64-byte input
    let sixty_four_bytes = format!(r#"{{"header":"{}"}}"#, "a".repeat(51));
    assert_eq!(sixty_four_bytes.len(), 64);
    assert_simd_scalar_parity(&sixty_four_bytes);

    // Whitespace runs of exact chunk lengths
    let ws_8 = "        42";
    assert_eq!(&ws_8[..8], "        ");
    assert_simd_scalar_parity(ws_8);

    let ws_16 = "                42";
    assert_eq!(&ws_16[..16], "                ");
    assert_simd_scalar_parity(ws_16);

    let ws_32 = "                                42";
    assert_eq!(&ws_32[..32], "                                ");
    assert_simd_scalar_parity(ws_32);
}

// ============================================================================
// 5. Tail Shorter than a Vector Chunk (Chunk + Tail Combinations)
// ============================================================================

#[test]
fn test_simd_tail_shorter_than_vector() {
    // 9 bytes = 8-byte chunk + 1-byte tail
    let nine_bytes = "[1,2,3,4]";
    assert_eq!(nine_bytes.len(), 9);
    assert_simd_scalar_parity(nine_bytes);

    // 17 bytes = 16-byte chunk + 1-byte tail
    let seventeen_bytes = r#"{"key":"val1234"}"#;
    assert_eq!(seventeen_bytes.len(), 17);
    assert_simd_scalar_parity(seventeen_bytes);

    // 33 bytes = 32-byte chunk + 1-byte tail
    let thirty_three_bytes = r#"{"name":"lust-lang","simd":true} "#;
    assert_eq!(thirty_three_bytes.len(), 33);
    assert_simd_scalar_parity(thirty_three_bytes);

    // 49 bytes = 32-byte chunk + 16-byte chunk + 1-byte tail
    let forty_nine_bytes = format!(r#"{{"title":"{}"}}"#, "a".repeat(37));
    assert_eq!(forty_nine_bytes.len(), 49);
    assert_simd_scalar_parity(&forty_nine_bytes);

    // Varied tail lengths in strings
    for tail_len in 1..=35 {
        let text = format!("\"{}\"", "x".repeat(tail_len));
        assert_simd_scalar_parity(&text);
    }

    // Varied tail lengths in whitespace
    for ws_len in 1..=35 {
        let text = format!("{}[1,2,3]", " ".repeat(ws_len));
        assert_simd_scalar_parity(&text);
    }
}

// ============================================================================
// 6. Fail-Closed Semantics: Duplicate Keys, Non-Finite Numbers, Trailing Bytes
// ============================================================================

#[test]
fn test_simd_rejects_non_finite_overflow() {
    // Gate-01 / RFC 8259: Non-finite numbers must fail closed with InvalidNumber
    let non_finite_cases = [
        "1e999",
        "-1e999",
        "1e1000",
        "-1e1000",
        "1e309",
        "-1e309",
    ];

    for nf in non_finite_cases {
        assert_simd_scalar_parity(nf);
        let res = json::from_str_simd(nf);
        assert!(matches!(res, Err(JsonError::InvalidNumber(_))));
    }
}

#[test]
fn test_simd_rejects_duplicate_keys() {
    // Duplicate keys must fail closed with DuplicateKey(key, pos) matching scalar oracle
    let dup_cases = [
        r#"{"a": 1, "a": 2}"#,
        r#"{"title": "test", "score": 10, "title": "duplicate"}"#,
        r#"{"nested": {"dup": true, "dup": false}}"#,
        r#"{"a": 1, "b": 2, "c": 3, "a": 4}"#,
    ];

    for dc in dup_cases {
        assert_simd_scalar_parity(dc);
        let res = json::from_str_simd(dc);
        assert!(matches!(res, Err(JsonError::DuplicateKey(..))));
    }
}

#[test]
fn test_simd_rejects_trailing_bytes() {
    let trailing_cases = [
        r#"{"a": 1} extra"#,
        "123 456",
        "true false",
        "\"hello\" world",
        "[1, 2] [3, 4]",
        "null trailing_data",
        "true   trailing",
    ];

    for tc in trailing_cases {
        assert_simd_scalar_parity(tc);
        let res = json::from_str_simd(tc);
        assert!(matches!(res, Err(JsonError::TrailingBytes(_))));
    }
}

#[test]
fn test_simd_rejects_trailing_commas() {
    let trailing_comma_cases = [
        "[1, 2, 3,]",
        "[,]",
        r#"{"a": 1,}"#,
        r#"{,}"#,
        r#"{"a": 1, "b": 2,}"#,
    ];

    for tcc in trailing_comma_cases {
        assert_simd_scalar_parity(tcc);
        let res = json::from_str_simd(tcc);
        assert!(matches!(res, Err(JsonError::TrailingComma(_))));
    }
}

#[test]
fn test_simd_rejects_leading_zeros() {
    let leading_zeros = [
        "0123",
        "-01",
        "00",
        "01.5",
    ];

    for lz in leading_zeros {
        assert_simd_scalar_parity(lz);
        let res = json::from_str_simd(lz);
        assert!(matches!(res, Err(JsonError::InvalidNumber(_))));
    }
}

#[test]
fn test_simd_rejects_control_characters_in_strings() {
    // Unescaped 0x00..=0x1F inside string literals is strictly forbidden by RFC 8259
    let bad_ctrl_cases = [
        "\"hello\x00world\"",
        "\"test\x01me\"",
        "\"line1\nline2\"",
        "\"tab\there\"",
        "\"carriage\rreturn\"",
        "\"form\x0cfeed\"",
    ];

    for bc in bad_ctrl_cases {
        assert_simd_scalar_parity(bc);
        let res = json::from_str_simd(bc);
        assert!(matches!(res, Err(JsonError::UnexpectedChar(..))));
    }
}

#[test]
fn test_simd_rejects_invalid_escapes_and_unicode() {
    let invalid_escapes = [
        r#""\a""#,
        r#""\1""#,
        r#""\x20""#,
        r#""\u12""#,
        r#""\uZZZZ""#,
        r#""\uD800""#,       // Lone high surrogate without low
        r#""\uD800\u0000""#, // High surrogate followed by non-surrogate
    ];

    for ie in invalid_escapes {
        assert_simd_scalar_parity(ie);
    }
}

// ============================================================================
// 7. Limits: Max Depth and Max Document Bytes
// ============================================================================

#[test]
fn test_simd_limits_parity() {
    // Max depth
    let deeply_nested = "[[[[[1]]]]]";
    assert_simd_scalar_parity_with_limits(deeply_nested.as_bytes(), 3, 1024);
    assert_simd_scalar_parity_with_limits(deeply_nested.as_bytes(), 6, 1024);

    // Max bytes
    let large = "{\"a\": 12345}";
    assert_simd_scalar_parity_with_limits(large.as_bytes(), 10, 5);
    assert_simd_scalar_parity_with_limits(large.as_bytes(), 10, 50);
}

// ============================================================================
// 8. Page / Chunk / Tail Bounds (No Padding Assumption)
// ============================================================================

#[test]
fn test_simd_unpadded_slice_bounds_safety() {
    // Construct a buffer without any trailing zero/padding bytes
    let text = r#"{"name":"lust-simd","items":[1,2,3,4],"verified":true}"#;
    let bytes = text.as_bytes();

    // Slices ending at every sub-length from 1 to bytes.len()
    for end in 1..=bytes.len() {
        let sub_slice = &bytes[..end];
        let sub_str = std::str::from_utf8(sub_slice).unwrap();
        assert_simd_scalar_parity(sub_str);
    }

    // Slices starting from various offsets to the end
    for start in 0..bytes.len() {
        let sub_slice = &bytes[start..];
        let sub_str = std::str::from_utf8(sub_slice).unwrap();
        assert_simd_scalar_parity(sub_str);
    }
}

// ============================================================================
// 9. Complex & Nested Documents
// ============================================================================

#[test]
fn test_simd_complex_nested_document() {
    let complex_json = r#"{
        "project": "Lust Language",
        "version": 3,
        "active": true,
        "features": [
            "fast",
            "safe",
            "embeddable",
            "simd-json",
            "zero-copy"
        ],
        "stats": {
            "score": 99.85,
            "passes": 1420,
            "failures": 0,
            "ratio": 1.0e-4
        },
        "description": "High performance \t scripting engine with \"JIT\" and strict RFC 8259 JSON parser.",
        "payload": {
            "nested_array": [
                {"id": 1, "valid": true},
                {"id": 2, "valid": false},
                {"id": 3, "valid": null}
            ]
        }
    }"#;

    assert_simd_scalar_parity(complex_json);

    let parsed_simd = json::from_str_simd(complex_json).unwrap();
    let parsed_scalar = json::from_str(complex_json).unwrap();
    assert_eq!(parsed_simd, parsed_scalar);

    // Verify roundtrip encoding
    let encoded = json::to_string(&parsed_simd).unwrap();
    let reparsed_simd = json::from_str_simd(&encoded).unwrap();
    assert_eq!(parsed_simd, reparsed_simd);
}

// ============================================================================
// 10. Multi-byte UTF-8 Chunk Alignment and Vector Boundary Crossing
// ============================================================================

#[test]
fn test_simd_multibyte_utf8_boundaries_and_alignment() {
    // 10.1 Aligned exactly on and straddling chunk boundaries (8 bytes SWAR, 16 bytes NEON, 32 bytes AVX2, 64 bytes)
    for &boundary in &[8usize, 16, 32, 64] {
        // --- Cyrillic (2-byte UTF-8, e.g. 'ж' = 0xD0 0xB6) ---
        // Aligned at boundary: string starts with '"', (boundary - 1) ASCII chars, then 'ж'
        let cyr_aligned = format!("\"{}ж\"", "a".repeat(boundary - 1));
        assert_simd_scalar_parity(&cyr_aligned);

        // Straddling boundary: index boundary - 1 is 0xD0, index boundary is 0xB6
        if boundary >= 2 {
            let cyr_straddle = format!("\"{}ж\"", "a".repeat(boundary - 2));
            assert_simd_scalar_parity(&cyr_straddle);
        }

        // --- CJK (3-byte UTF-8, e.g. '你' = 0xE4 0xBD 0xA0) ---
        // Aligned at boundary
        let cjk_aligned = format!("\"{}你\"", "a".repeat(boundary - 1));
        assert_simd_scalar_parity(&cjk_aligned);

        // Straddling boundary with 1 byte in chunk N-1 and 2 bytes in chunk N
        if boundary >= 2 {
            let cjk_straddle1 = format!("\"{}你\"", "a".repeat(boundary - 2));
            assert_simd_scalar_parity(&cjk_straddle1);
        }

        // Straddling boundary with 2 bytes in chunk N-1 and 1 byte in chunk N
        if boundary >= 3 {
            let cjk_straddle2 = format!("\"{}你\"", "a".repeat(boundary - 3));
            assert_simd_scalar_parity(&cjk_straddle2);
        }

        // --- Emoji (4-byte UTF-8, e.g. '🦀' = 0xF0 0x9F 0xA6 0x80) ---
        // Aligned at boundary
        let emoji_aligned = format!("\"{}🦀\"", "a".repeat(boundary - 1));
        assert_simd_scalar_parity(&emoji_aligned);

        // Straddling boundary with 1 byte in chunk N-1 and 3 bytes in chunk N
        if boundary >= 2 {
            let emoji_straddle1 = format!("\"{}🦀\"", "a".repeat(boundary - 2));
            assert_simd_scalar_parity(&emoji_straddle1);
        }

        // Straddling boundary with 2 bytes in chunk N-1 and 2 bytes in chunk N
        if boundary >= 3 {
            let emoji_straddle2 = format!("\"{}🦀\"", "a".repeat(boundary - 3));
            assert_simd_scalar_parity(&emoji_straddle2);
        }

        // Straddling boundary with 3 bytes in chunk N-1 and 1 byte in chunk N
        if boundary >= 4 {
            let emoji_straddle3 = format!("\"{}🦀\"", "a".repeat(boundary - 4));
            assert_simd_scalar_parity(&emoji_straddle3);
        }
    }

    // 10.2 Sweeping alignment: test all offsets 0..=48 for Cyrillic, CJK, Emoji, and mixed text
    for pad in 0..=48 {
        let p = "x".repeat(pad);

        // Cyrillic pangram with varying alignments
        let cyrillic_doc = format!("\"{p}Съешь же ещё этих мягких французских булок, да выпей чаю.\"");
        assert_simd_scalar_parity(&cyrillic_doc);

        // CJK text with varying alignments
        let cjk_doc = format!("\"{p}天地玄黄，宇宙洪荒。日月盈昃，辰宿列张。\"");
        assert_simd_scalar_parity(&cjk_doc);

        // Multi-emoji runs with varying alignments
        let emoji_doc = format!("\"{p}🦀🚀🔥🎉✨📦💡🎯⚡️🦄🌈\"");
        assert_simd_scalar_parity(&emoji_doc);

        // Mixed multi-byte text interleaved with JSON escapes
        let mixed_doc = format!("\"{p}Prefix_\\n_Привет_\\t_世界_\\\"_🦀_\\\\_End\"");
        assert_simd_scalar_parity(&mixed_doc);
    }

    // 10.3 Exact vector chunk multiples of pure non-ASCII UTF-8
    // 8 bytes (4 Cyrillic chars)
    assert_simd_scalar_parity("\"тест\"");
    // 16 bytes (8 Cyrillic chars)
    assert_simd_scalar_parity("\"тестиров\"");
    // 32 bytes (16 Cyrillic chars)
    assert_simd_scalar_parity("\"тестирование_sim\"");
    // 64 bytes (32 Cyrillic chars)
    assert_simd_scalar_parity("\"тестирование_simd_парсер_lust_\"");

    // Pure Emoji: 8 bytes (2 emoji), 16 bytes (4 emoji), 32 bytes (8 emoji)
    assert_simd_scalar_parity("\"🦀🚀\"");
    assert_simd_scalar_parity("\"🦀🚀🎉🔥\"");
    assert_simd_scalar_parity("\"🦀🚀🎉🔥✨📦💡🎯\"");

    // Document structures with multi-byte keys and values
    let doc = r#"{
        "ключ_на_русском": "значение_кириллица",
        "中文键名": "这是中文值",
        "emoji_key_🦀": ["🚀", "🔥", {"вложенный": "объект"}]
    }"#;
    assert_simd_scalar_parity(doc);
}

// ============================================================================
// 11. Fail-Closed: Invalid UTF-8 Byte Rejection across All CPU Dispatches
// ============================================================================

/// Verification helper for invalid UTF-8 bytes: asserts that both scalar and all SIMD dispatches
/// reject the input with JsonError::InvalidUnicode and have exact parity.
fn assert_invalid_utf8_rejected(bytes: &[u8]) {
    let scalar_res = json::from_slice_with_limits(bytes, 256, 1024 * 1024);
    assert!(
        matches!(scalar_res, Err(JsonError::InvalidUnicode(_))),
        "Scalar oracle did not reject invalid UTF-8 with InvalidUnicode on input {:?}: got {:?}",
        bytes,
        scalar_res
    );

    for dispatch in test_dispatches() {
        let simd_res = json::from_slice_simd_with_limits_and_dispatch(bytes, 256, 1024 * 1024, dispatch);
        assert_eq!(
            simd_res, scalar_res,
            "SIMD dispatch {:?} diverged from scalar oracle on invalid UTF-8 input: {:?}",
            dispatch, bytes
        );
        assert!(
            matches!(simd_res, Err(JsonError::InvalidUnicode(_))),
            "SIMD dispatch {:?} did not return InvalidUnicode on input {:?}: got {:?}",
            dispatch, bytes, simd_res
        );
    }
}

#[test]
fn test_simd_rejects_invalid_utf8_across_all_dispatches() {
    // 11.1 Lone invalid bytes (never valid anywhere in UTF-8)
    let invalid_bytes: &[u8] = &[0xC0, 0xC1, 0xF5, 0xF6, 0xF7, 0xF8, 0xFC, 0xFE, 0xFF];
    for &bad in invalid_bytes {
        let input = [b'"', bad, b'"'];
        assert_invalid_utf8_rejected(&input);

        // Preceded and followed by clean ASCII bytes
        let with_surrounding = [b'"', b'a', b'b', bad, b'c', b'd', b'"'];
        assert_invalid_utf8_rejected(&with_surrounding);
    }

    // 11.2 Lone continuation bytes (0x80..=0xBF without a leading byte)
    let lone_continuations: &[u8] = &[0x80, 0x85, 0x90, 0x9F, 0xA0, 0xBF];
    for &cont in lone_continuations {
        let input = [b'"', cont, b'"'];
        assert_invalid_utf8_rejected(&input);

        let input_mid = [b'"', b'h', b'e', b'l', b'l', b'o', cont, b'w', b'o', b'r', b'l', b'd', b'"'];
        assert_invalid_utf8_rejected(&input_mid);
    }

    // 11.3 Overlong encodings (security vulnerability vector rejected by RFC 3629)
    let overlong_cases: &[&[u8]] = &[
        // Overlong 2-byte sequences
        &[b'"', 0xC0, 0xAF, b'"'], // Overlong '/' (0x2F)
        &[b'"', 0xC1, 0xBF, b'"'], // Overlong 0x7F
        // Overlong 3-byte sequences
        &[b'"', 0xE0, 0x80, 0xAF, b'"'],
        &[b'"', 0xE0, 0x9F, 0xBF, b'"'],
        // Overlong 4-byte sequences
        &[b'"', 0xF0, 0x80, 0x80, 0xAF, b'"'],
        &[b'"', 0xF0, 0x8F, 0xBF, 0xBF, b'"'],
    ];
    for &case in overlong_cases {
        assert_invalid_utf8_rejected(case);
    }

    // 11.4 Incomplete / truncated sequences cut off by closing quote
    let incomplete_cases: &[&[u8]] = &[
        // Incomplete 2-byte sequence (lead only)
        &[b'"', 0xD0, b'"'],
        // Incomplete 3-byte sequence (lead only)
        &[b'"', 0xE4, b'"'],
        // Incomplete 3-byte sequence (lead + 1 continuation byte)
        &[b'"', 0xE4, 0xBD, b'"'],
        // Incomplete 4-byte sequence (lead only)
        &[b'"', 0xF0, b'"'],
        // Incomplete 4-byte sequence (lead + 1 continuation byte)
        &[b'"', 0xF0, 0x9F, b'"'],
        // Incomplete 4-byte sequence (lead + 2 continuation bytes)
        &[b'"', 0xF0, 0x9F, 0xA6, b'"'],
    ];
    for &case in incomplete_cases {
        assert_invalid_utf8_rejected(case);
    }

    // 11.5 Invalid continuation bytes (expected continuation byte 0x80..0xBF, got something else)
    let bad_continuation_cases: &[&[u8]] = &[
        // 2-byte lead followed by ASCII letter
        &[b'"', 0xD0, b'a', b'"'],
        // 2-byte lead followed by another lead byte
        &[b'"', 0xD0, 0xD0, b'"'],
        // 3-byte lead followed by ASCII space and continuation
        &[b'"', 0xE4, 0x20, 0xA0, b'"'],
        // 3-byte lead + valid continuation followed by ASCII letter
        &[b'"', 0xE4, 0xBD, b'z', b'"'],
        // 4-byte lead followed by ASCII char
        &[b'"', 0xF0, b'x', 0x9F, 0xA6, b'"'],
        // 4-byte lead + valid 2 bytes followed by ASCII char
        &[b'"', 0xF0, 0x9F, 0xA6, b'!', b'"'],
    ];
    for &case in bad_continuation_cases {
        assert_invalid_utf8_rejected(case);
    }

    // 11.6 UTF-16 surrogate halves in UTF-8 (U+D800..=U+DFFF strictly forbidden by Unicode standard)
    let surrogate_cases: &[&[u8]] = &[
        // U+D800
        &[b'"', 0xED, 0xA0, 0x80, b'"'],
        // U+DBFF
        &[b'"', 0xED, 0xAF, 0xBF, b'"'],
        // U+DC00
        &[b'"', 0xED, 0xB0, 0x80, b'"'],
        // U+DFFF
        &[b'"', 0xED, 0xBF, 0xBF, b'"'],
    ];
    for &case in surrogate_cases {
        assert_invalid_utf8_rejected(case);
    }

    // 11.7 Out-of-range codepoints (> U+10FFFF)
    let out_of_range_cases: &[&[u8]] = &[
        // U+110000
        &[b'"', 0xF4, 0x90, 0x80, 0x80, b'"'],
        // 0xF5.. starts codepoint >= U+140000
        &[b'"', 0xF5, 0x80, 0x80, 0x80, b'"'],
    ];
    for &case in out_of_range_cases {
        assert_invalid_utf8_rejected(case);
    }

    // 11.8 Invalid UTF-8 positioned at and crossing vector chunk boundaries (8, 16, 32 bytes)
    for &boundary in &[8usize, 16, 32] {
        // 0xFF placed right at chunk boundary
        let mut buf_at_boundary = vec![b'"'];
        buf_at_boundary.extend(vec![b'a'; boundary - 1]);
        buf_at_boundary.push(0xFF);
        buf_at_boundary.push(b'"');
        assert_invalid_utf8_rejected(&buf_at_boundary);

        // Incomplete 2-byte sequence straddling chunk boundary
        if boundary >= 2 {
            let mut buf_straddle_2 = vec![b'"'];
            buf_straddle_2.extend(vec![b'a'; boundary - 2]);
            buf_straddle_2.push(0xD0); // lead byte at end of chunk 0
            buf_straddle_2.push(b'"'); // closing quote at start of chunk 1
            assert_invalid_utf8_rejected(&buf_straddle_2);

            // Lead byte at end of chunk 0, invalid continuation byte at start of chunk 1
            let mut buf_straddle_bad = vec![b'"'];
            buf_straddle_bad.extend(vec![b'a'; boundary - 2]);
            buf_straddle_bad.push(0xD0);
            buf_straddle_bad.push(b'X');
            buf_straddle_bad.push(b'"');
            assert_invalid_utf8_rejected(&buf_straddle_bad);
        }

        // Incomplete 3-byte sequence straddling chunk boundary
        if boundary >= 3 {
            let mut buf_straddle_3 = vec![b'"'];
            buf_straddle_3.extend(vec![b'a'; boundary - 3]);
            buf_straddle_3.push(0xE4); // byte boundary - 2
            buf_straddle_3.push(0xBD); // byte boundary - 1
            buf_straddle_3.push(b'"'); // closing quote at boundary
            assert_invalid_utf8_rejected(&buf_straddle_3);
        }

        // Incomplete 4-byte sequence straddling chunk boundary
        if boundary >= 4 {
            let mut buf_straddle_4 = vec![b'"'];
            buf_straddle_4.extend(vec![b'a'; boundary - 4]);
            buf_straddle_4.push(0xF0); // byte boundary - 3
            buf_straddle_4.push(0x9F); // byte boundary - 2
            buf_straddle_4.push(0xA6); // byte boundary - 1
            buf_straddle_4.push(b'"'); // closing quote at boundary
            assert_invalid_utf8_rejected(&buf_straddle_4);
        }
    }
}

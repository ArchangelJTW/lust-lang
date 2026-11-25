#![cfg(feature = "std")]

use lust::data::chunked_stream::{
    BsonStreamDecoder, CsvStreamDecoder, JsonStreamDecoder, StreamDecoderOptions,
    ToonStreamDecoder, YamlStreamDecoder, StreamError,
};
use lust::data::csv::{
    parse_csv, parse_tsv, to_csv_string, to_tsv_string, CsvError, CsvOptions, CsvTable,
};
use lust::data::document::Document;
use lust::data::msgpack::{
    from_slice as from_msgpack_slice,
    from_slice_document as from_msgpack_slice_document,
    to_vec_document as to_msgpack_vec_document, MsgPackError,
};
use lust::data::number::{DateTimeUtc, Decimal128, ExactNumber, ObjectId};
use lust::data::value::DataValue;
use std::io::Cursor;

// ============================================================================
// 1. D-CSV: CSV / TSV Configurable Delimiters, Quotes, Headers, Gate-01
// ============================================================================

#[test]
fn test_csv_standard_rfc4180_roundtrip() {
    let csv_text = "id,name,age,active\r\n1,Alice,30,true\r\n2,\"Bob, Jr.\",25,false\r\n3,\"Charlie\nMultiline\",40,true\r\n";
    let table = parse_csv(csv_text).expect("RFC 4180 CSV parsing succeeds");

    assert_eq!(table.headers(), Some(&["id".to_string(), "name".to_string(), "age".to_string(), "active".to_string()][..]));
    assert_eq!(table.len(), 3);

    let docs = table.to_documents().expect("conversion to documents succeeds");
    assert_eq!(docs.len(), 3);
    assert_eq!(docs[0].get("id"), Some(&DataValue::Number(ExactNumber::Int(1))));
    assert_eq!(docs[0].get("name"), Some(&DataValue::String("Alice".to_string())));
    assert_eq!(docs[0].get("active"), Some(&DataValue::Bool(true)));

    // Quoted comma in Bob, Jr.
    assert_eq!(docs[1].get("name"), Some(&DataValue::String("Bob, Jr.".to_string())));
    // Multiline field in Charlie
    assert_eq!(docs[2].get("name"), Some(&DataValue::String("Charlie\nMultiline".to_string())));

    // Roundtrip back to string
    let serialized = to_csv_string(&table).expect("serialization succeeds");
    let reparsed = parse_csv(&serialized).expect("reparsing succeeds");
    assert_eq!(reparsed.to_documents().unwrap(), docs);
}

#[test]
fn test_tsv_configurable_delimiter_tabs() {
    let tsv_text = "sku\tprice\tquantity\r\nITEM-001\t19.99\t100\r\nITEM-002\t5.5\t42\r\n";
    let table = parse_tsv(tsv_text).expect("TSV parsing succeeds");

    assert_eq!(table.headers(), Some(&["sku".to_string(), "price".to_string(), "quantity".to_string()][..]));
    assert_eq!(table.len(), 2);

    let docs = table.to_documents().expect("TSV documents valid");
    assert_eq!(docs[0].get("sku"), Some(&DataValue::String("ITEM-001".to_string())));
    assert_eq!(docs[0].get("price"), Some(&DataValue::Number(ExactNumber::Float(19.99))));
    assert_eq!(docs[0].get("quantity"), Some(&DataValue::Number(ExactNumber::Int(100))));

    let serialized = to_tsv_string(&table).expect("TSV serialization succeeds");
    assert!(serialized.contains('\t'), "TSV output must contain tabs");
    let reparsed = parse_tsv(&serialized).expect("TSV roundtrip valid");
    assert_eq!(reparsed.to_documents().unwrap(), docs);
}

#[test]
fn test_csv_custom_delimiter_semicolon_and_custom_quote() {
    let options = CsvOptions::default()
        .with_delimiter(b';')
        .with_quote(b'\'');

    let text = "col1;col2;col3\r\n'val1;extra';42;'hello ''world'''\r\n";
    let table = CsvTable::from_str(text, &options).expect("custom delimiter/quote succeeds");

    let docs = table.to_documents().unwrap();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].get("col1"), Some(&DataValue::String("val1;extra".to_string())));
    assert_eq!(docs[0].get("col2"), Some(&DataValue::Number(ExactNumber::Int(42))));
    assert_eq!(docs[0].get("col3"), Some(&DataValue::String("hello 'world'".to_string())));
}

#[test]
fn test_csv_gate_schema_01_duplicate_headers_strictly_rejected() {
    let csv_duplicate = "user,email,user\r\n1,alice@example.com,duplicate_val\r\n";
    let res = parse_csv(csv_duplicate);
    assert!(res.is_err(), "Duplicate header names must fail closed under Gate-SCHEMA-01");
    match res.unwrap_err() {
        CsvError::DuplicateHeader(h) => assert_eq!(h, "user"),
        other => panic!("Expected DuplicateHeader error, got {:?}", other),
    }
}

#[test]
fn test_csv_column_mismatch_strict_rejection() {
    let csv_jagged = "a,b,c\r\n1,2,3\r\n4,5\r\n";
    let res = parse_csv(csv_jagged);
    assert!(res.is_err(), "Jagged rows must fail closed when flexible is false");
    match res.unwrap_err() {
        CsvError::ColumnMismatch { expected, found, row } => {
            assert_eq!(expected, 3);
            assert_eq!(found, 2);
            assert_eq!(row, 2);
        }
        other => panic!("Expected ColumnMismatch error, got {:?}", other),
    }

    // Flexible mode accepts variable column rows
    let options = CsvOptions::csv().with_flexible(true);
    let table_flex = CsvTable::from_str(csv_jagged, &options).expect("flexible mode succeeds");
    assert_eq!(table_flex.records().len(), 2);
}

#[test]
fn test_csv_unterminated_quote_fails_closed() {
    let unclosed = "id,name\r\n1,\"Alice unclosed quote at eof";
    let res = parse_csv(unclosed);
    assert!(res.is_err(), "Unclosed quote at EOF must fail closed");
    assert!(matches!(res.unwrap_err(), CsvError::UnterminatedQuote { .. }));
}

// ============================================================================
// 2. D-MSGPACK: MessagePack Codec, Extended Tags, Gate-01
// ============================================================================

#[test]
fn test_msgpack_scalars_and_structures_roundtrip() {
    let mut doc = Document::new();
    doc.try_insert("int_val".to_string(), DataValue::Number(ExactNumber::Int(-12345))).unwrap();
    doc.try_insert("uint_val".to_string(), DataValue::Number(ExactNumber::UInt(18_446_744_073_709_551_615))).unwrap();
    doc.try_insert("float_val".to_string(), DataValue::Number(ExactNumber::Float(3.1415926535))).unwrap();
    doc.try_insert("bool_val".to_string(), DataValue::Bool(true)).unwrap();
    doc.try_insert("null_val".to_string(), DataValue::Null).unwrap();
    doc.try_insert("str_val".to_string(), DataValue::String("Lust MessagePack".to_string())).unwrap();

    let mut inner = Document::new();
    inner.try_insert("nested_k".to_string(), DataValue::String("nested_v".to_string())).unwrap();
    doc.try_insert("sub_doc".to_string(), DataValue::Document(inner.clone())).unwrap();

    let arr = vec![
        DataValue::Number(ExactNumber::Int(1)),
        DataValue::Number(ExactNumber::Int(2)),
        DataValue::Number(ExactNumber::Int(3)),
    ];
    doc.try_insert("items".to_string(), DataValue::Array(arr)).unwrap();

    // Serialize to MessagePack bytes
    let bytes = to_msgpack_vec_document(&doc);
    assert!(!bytes.is_empty());

    // Deserialize back
    let decoded = from_msgpack_slice_document(&bytes).expect("MessagePack decoding succeeds");
    assert_eq!(decoded, doc);
}

#[test]
fn test_msgpack_nominal_extensions_roundtrip() {
    let hex_id = "507f1f77bcf86cd799439011";
    let oid = ObjectId::from_hex(hex_id).unwrap();
    let dt = DateTimeUtc::from_millis(1700000000000);
    let d128 = Decimal128::new([42u8; 16]);

    let mut doc = Document::new();
    doc.try_insert("oid".to_string(), DataValue::ObjectId(oid)).unwrap();
    doc.try_insert("dt".to_string(), DataValue::DateTimeUtc(dt)).unwrap();
    doc.try_insert("d128".to_string(), DataValue::Number(ExactNumber::Decimal128(d128))).unwrap();

    let bytes = to_msgpack_vec_document(&doc);
    let decoded = from_msgpack_slice_document(&bytes).expect("MessagePack extension decoding succeeds");

    assert_eq!(decoded.get("oid"), Some(&DataValue::ObjectId(oid)));
    assert_eq!(decoded.get("dt"), Some(&DataValue::DateTimeUtc(dt)));
    assert_eq!(decoded.get("d128"), Some(&DataValue::Number(ExactNumber::Decimal128(d128))));
}

#[test]
fn test_msgpack_gate_schema_01_duplicate_keys_strictly_rejected() {
    // Manually craft MessagePack map with 2 keys: fixmap with 2 entries: {"key": 1, "key": 2}
    let mut bytes = Vec::new();
    bytes.push(0x82); // fixmap 2 pairs

    // Pair 1: "key": 1
    bytes.push(0xa3); // fixstr 3
    bytes.extend_from_slice(b"key");
    bytes.push(1); // fixint 1

    // Pair 2: "key": 2
    bytes.push(0xa3); // fixstr 3
    bytes.extend_from_slice(b"key");
    bytes.push(2); // fixint 2

    let res = from_msgpack_slice_document(&bytes);
    assert!(res.is_err(), "Duplicate keys in MessagePack map must fail closed under Gate-SCHEMA-01");
    match res.unwrap_err() {
        MsgPackError::DuplicateKey(k) => assert_eq!(k, "key"),
        other => panic!("Expected DuplicateKey error, got {:?}", other),
    }
}

#[test]
fn test_msgpack_malformed_suffix_rejected() {
    let mut doc = Document::new();
    doc.try_insert("a".to_string(), DataValue::Number(ExactNumber::Int(1))).unwrap();
    let mut bytes = to_msgpack_vec_document(&doc);

    // Append extra unexpected byte
    bytes.push(0xff);

    let res = from_msgpack_slice_document(&bytes);
    assert_eq!(res.unwrap_err(), MsgPackError::MalformedSuffix);
}

// ============================================================================
// 3. D-STREAM: Chunked Streaming Decoders (JSON, YAML, BSON, TOON, CSV)
// ============================================================================

#[test]
fn test_chunked_stream_bson_decoder() {
    let mut doc1 = Document::new();
    doc1.try_insert("doc".to_string(), DataValue::Number(ExactNumber::Int(1))).unwrap();

    let mut doc2 = Document::new();
    doc2.try_insert("doc".to_string(), DataValue::Number(ExactNumber::Int(2))).unwrap();

    let mut stream_bytes = lust::data::bson::to_vec(&doc1);
    stream_bytes.extend_from_slice(&lust::data::bson::to_vec(&doc2));

    // Stream with small 16-byte chunk buffer
    let options = StreamDecoderOptions::new(16, 1024);
    let mut decoder = BsonStreamDecoder::new(Cursor::new(stream_bytes), options);

    let first = decoder.next_document().expect("pull doc1 succeeds");
    assert_eq!(first, Some(doc1));

    let second = decoder.next_document().expect("pull doc2 succeeds");
    assert_eq!(second, Some(doc2));

    let third = decoder.next_document().expect("pull eof succeeds");
    assert_eq!(third, None);
}

#[test]
fn test_chunked_stream_json_decoder() {
    let json_stream_text = r#"{"id": 1, "name": "alpha"} {"id": 2, "name": "beta"} [1, 2, 3] "plain string""#;

    // Stream with small 8-byte chunk buffer across value boundaries
    let options = StreamDecoderOptions::new(8, 4096);
    let mut decoder = JsonStreamDecoder::new(Cursor::new(json_stream_text.as_bytes()), options);

    let v1 = decoder.next_value().expect("read v1 succeeds").unwrap();
    assert!(matches!(v1, DataValue::Document(_)));

    let v2 = decoder.next_value().expect("read v2 succeeds").unwrap();
    assert!(matches!(v2, DataValue::Document(_)));

    let v3 = decoder.next_value().expect("read v3 succeeds").unwrap();
    assert!(matches!(v3, DataValue::Array(_)));

    let v4 = decoder.next_value().expect("read v4 succeeds").unwrap();
    assert_eq!(v4, DataValue::String("plain string".to_string()));

    let eof = decoder.next_value().expect("read eof succeeds");
    assert_eq!(eof, None);
}

#[test]
fn test_chunked_stream_yaml_decoder() {
    let yaml_stream = "title: Doc 1\r\ncount: 10\r\n---\r\ntitle: Doc 2\r\ncount: 20\r\n";

    // 16-byte chunk buffer
    let options = StreamDecoderOptions::new(16, 2048);
    let mut decoder = YamlStreamDecoder::new(Cursor::new(yaml_stream.as_bytes()), options);

    let d1 = decoder.next_document().expect("pull d1 succeeds").unwrap();
    assert_eq!(d1.get("title"), Some(&DataValue::String("Doc 1".to_string())));
    assert_eq!(d1.get("count"), Some(&DataValue::Number(ExactNumber::Int(10))));

    let d2 = decoder.next_document().expect("pull d2 succeeds").unwrap();
    assert_eq!(d2.get("title"), Some(&DataValue::String("Doc 2".to_string())));
    assert_eq!(d2.get("count"), Some(&DataValue::Number(ExactNumber::Int(20))));

    let eof = decoder.next_document().expect("pull eof succeeds");
    assert_eq!(eof, None);
}

#[test]
fn test_chunked_stream_toon_decoder() {
    let toon_stream = "item | qty | in_stock\nwidget | 100 | true\ngadget | 42 | false\n";

    let options = StreamDecoderOptions::new(16, 2048);
    let mut decoder = ToonStreamDecoder::new(Cursor::new(toon_stream.as_bytes()), options);

    let r1 = decoder.next_document().expect("pull r1 succeeds").unwrap();
    assert_eq!(r1.get("item"), Some(&DataValue::String("widget".to_string())));
    assert_eq!(r1.get("qty"), Some(&DataValue::Number(ExactNumber::Int(100))));
    assert_eq!(r1.get("in_stock"), Some(&DataValue::Bool(true)));

    let r2 = decoder.next_document().expect("pull r2 succeeds").unwrap();
    assert_eq!(r2.get("item"), Some(&DataValue::String("gadget".to_string())));
    assert_eq!(r2.get("qty"), Some(&DataValue::Number(ExactNumber::Int(42))));
    assert_eq!(r2.get("in_stock"), Some(&DataValue::Bool(false)));

    let eof = decoder.next_document().expect("pull eof succeeds");
    assert_eq!(eof, None);
}

#[test]
fn test_chunked_stream_csv_decoder() {
    let csv_stream = "service,tier,rate\r\nauth,core,1000\r\npayments,critical,500\r\n";

    let options = CsvOptions::csv();
    let mut decoder = CsvStreamDecoder::new(Cursor::new(csv_stream.as_bytes()), options).unwrap();

    let d1 = decoder.next_document().expect("pull d1 succeeds").unwrap();
    assert_eq!(d1.get("service"), Some(&DataValue::String("auth".to_string())));
    assert_eq!(d1.get("tier"), Some(&DataValue::String("core".to_string())));
    assert_eq!(d1.get("rate"), Some(&DataValue::Number(ExactNumber::Int(1000))));

    let d2 = decoder.next_document().expect("pull d2 succeeds").unwrap();
    assert_eq!(d2.get("service"), Some(&DataValue::String("payments".to_string())));
    assert_eq!(d2.get("tier"), Some(&DataValue::String("critical".to_string())));
    assert_eq!(d2.get("rate"), Some(&DataValue::Number(ExactNumber::Int(500))));

    let eof = decoder.next_document().expect("pull eof succeeds");
    assert_eq!(eof, None);
}

// ============================================================================
// 4. Red Tests for DA/LV Fix Plan (Wave 0 Track 0C)
// ============================================================================

#[test]
fn test_msgpack_declared_size_allocation_guard() {
    // Array32 declaring 1,000,000 elements (0x00, 0x0f, 0x42, 0x40) with only 1 trailing byte
    let hostile_array = vec![0xdd, 0x00, 0x0f, 0x42, 0x40, 0x01];
    let res = from_msgpack_slice(&hostile_array);
    assert!(res.is_err(), "Tiny payload with 1M declared elements must fail closed before proportional allocation");
    assert_eq!(res.unwrap_err(), MsgPackError::UnexpectedEof);

    // Map32 declaring 500,000 pairs = 1,000,000 elements (0x00, 0x07, 0xa1, 0x20) with only 2 trailing bytes
    let hostile_map = vec![0xdf, 0x00, 0x07, 0xa1, 0x20, 0xa1, b'a'];
    let res_map = from_msgpack_slice(&hostile_map);
    assert!(res_map.is_err(), "Tiny payload with 1M declared map pairs must fail closed before proportional allocation");
    assert_eq!(res_map.unwrap_err(), MsgPackError::UnexpectedEof);
}

#[test]
fn test_msgpack_unknown_extension_type_rejected() {
    // fixext 1 with unknown extension type 99 and 1 byte payload 0x42
    let unknown_ext = vec![0xd4, 99, 0x42];
    let res = from_msgpack_slice(&unknown_ext);
    assert!(res.is_err(), "Unknown extension type must be rejected fail-closed, not parsed as generic string");
    assert_eq!(res.unwrap_err(), MsgPackError::UnknownExtensionType(99));
}

#[test]
fn test_msgpack_malformed_marker_truncation_matrix() {
    // 0xc1: never used byte
    assert_eq!(from_msgpack_slice(&[0xc1]).unwrap_err(), MsgPackError::InvalidMarker(0xc1));

    // str8/16/32 with truncated length or payload
    assert_eq!(from_msgpack_slice(&[0xd9]).unwrap_err(), MsgPackError::UnexpectedEof);
    assert_eq!(from_msgpack_slice(&[0xda, 0x00]).unwrap_err(), MsgPackError::UnexpectedEof);
    assert_eq!(from_msgpack_slice(&[0xdb, 0x00, 0x00]).unwrap_err(), MsgPackError::UnexpectedEof);
    assert_eq!(from_msgpack_slice(&[0xd9, 0x05, b'a', b'b']).unwrap_err(), MsgPackError::UnexpectedEof);

    // bin8/16/32 truncated
    assert_eq!(from_msgpack_slice(&[0xc4]).unwrap_err(), MsgPackError::UnexpectedEof);
    assert_eq!(from_msgpack_slice(&[0xc5, 0x01]).unwrap_err(), MsgPackError::UnexpectedEof);
    assert_eq!(from_msgpack_slice(&[0xc6, 0x00, 0x00]).unwrap_err(), MsgPackError::UnexpectedEof);

    // array16/32 and map16/32 truncated headers
    assert_eq!(from_msgpack_slice(&[0xdc]).unwrap_err(), MsgPackError::UnexpectedEof);
    assert_eq!(from_msgpack_slice(&[0xdd, 0x00, 0x01]).unwrap_err(), MsgPackError::UnexpectedEof);
    assert_eq!(from_msgpack_slice(&[0xde]).unwrap_err(), MsgPackError::UnexpectedEof);
    assert_eq!(from_msgpack_slice(&[0xdf, 0x00]).unwrap_err(), MsgPackError::UnexpectedEof);

    // ext8/16/32 truncated headers
    assert_eq!(from_msgpack_slice(&[0xc7]).unwrap_err(), MsgPackError::UnexpectedEof);
    assert_eq!(from_msgpack_slice(&[0xc8, 0x00]).unwrap_err(), MsgPackError::UnexpectedEof);
    assert_eq!(from_msgpack_slice(&[0xc9, 0x00, 0x01]).unwrap_err(), MsgPackError::UnexpectedEof);
}

#[test]
fn test_csv_total_byte_cap_enforced() {
    let options = CsvOptions::csv().with_max_total_bytes(30);
    // 25 bytes: headers + row within 30 bytes
    let valid_csv = "col1,col2\r\nval1,val2\r\n";
    let res_ok = CsvTable::from_str(valid_csv, &options);
    assert!(res_ok.is_ok(), "CSV within total byte cap must succeed: {:?}", res_ok);

    // 45 bytes: exceeds 30 bytes
    let oversized_csv = "col1,col2\r\nval1,val2\r\nval3,val4\r\nval5,val6\r\n";
    let res_err = CsvTable::from_str(oversized_csv, &options);
    assert!(res_err.is_err(), "CSV exceeding total byte cap must fail closed");
    assert!(matches!(res_err.unwrap_err(), CsvError::MaxTotalBytesExceeded(_)));
}

#[test]
fn test_csv_malformed_quote_placement() {
    let options = CsvOptions::csv();
    // Quote inside unquoted field
    let mid_quote = "col1,col2\r\nval\"1,val2\r\n";
    let res = CsvTable::from_str(mid_quote, &options);
    assert!(res.is_err(), "Quote inside unquoted field must fail closed with InvalidQuotePlacement");
    assert!(matches!(res.unwrap_err(), CsvError::InvalidQuotePlacement { .. }));

    // Character immediately after closing quote before delimiter
    let bad_close = "col1,col2\r\n\"val1\"extra,val2\r\n";
    let res2 = CsvTable::from_str(bad_close, &options);
    assert!(res2.is_err(), "Characters following closing quote before delimiter must fail closed");
    assert!(matches!(res2.unwrap_err(), CsvError::InvalidQuotePlacement { .. }));
}

#[test]
fn test_chunked_stream_yaml_pre_growth_cap() {
    // Buffer limit 32 bytes, chunk size 16
    let options = StreamDecoderOptions::new(16, 32);
    // 80 bytes of continuous YAML without document separator ---
    let long_yaml = "field1: this is a very long line that exceeds the capacity of thirty-two bytes\n";
    let mut decoder = YamlStreamDecoder::new(Cursor::new(long_yaml.as_bytes()), options);

    let res = decoder.next_document();
    assert!(res.is_err(), "YAML stream exceeding max_buffer_capacity must fail closed before unbounded growth");
    match res.unwrap_err() {
        StreamError::BufferCapacityExceeded(cap) => {
            assert!(cap >= 32);
        }
        other => panic!("Expected BufferCapacityExceeded, got {:?}", other),
    }
}

#[test]
fn test_chunked_stream_json_pre_growth_cap() {
    // Buffer limit 32 bytes, chunk size 16
    let options = StreamDecoderOptions::new(16, 32);
    // 60 bytes of continuous unterminated JSON object without closing }
    let long_json = r#"{"id": 12345, "unclosed_string_field": "long continuous value"#;
    let mut decoder = JsonStreamDecoder::new(Cursor::new(long_json.as_bytes()), options);

    let res = decoder.next_value();
    assert!(res.is_err(), "JSON stream exceeding max_buffer_capacity must fail closed before unbounded growth");
    match res.unwrap_err() {
        StreamError::BufferCapacityExceeded(cap) => {
            assert!(cap >= 32);
        }
        other => panic!("Expected BufferCapacityExceeded, got {:?}", other),
    }
}

#[test]
fn test_chunked_stream_toon_pre_growth_cap() {
    // Buffer limit 32 bytes, chunk size 16
    let options = StreamDecoderOptions::new(16, 32);
    // 70 bytes of continuous TOON without newline
    let long_toon = "col1 | col2 | col3 | this_is_a_very_long_unbroken_line_without_any_newline";
    let mut decoder = ToonStreamDecoder::new(Cursor::new(long_toon.as_bytes()), options);

    let res = decoder.next_document();
    assert!(res.is_err(), "TOON stream exceeding max_buffer_capacity must fail closed before unbounded growth");
    match res.unwrap_err() {
        StreamError::BufferCapacityExceeded(cap) => {
            assert!(cap >= 32);
        }
        other => panic!("Expected BufferCapacityExceeded, got {:?}", other),
    }
}

// ============================================================================
// 5. Red Tests for Remediation Re-Review Findings (Track 0C)
// ============================================================================

struct CountingReader<R> {
    inner: R,
    bytes_read: usize,
}

impl<R: std::io::Read> CountingReader<R> {
    fn new(inner: R) -> Self {
        Self { inner, bytes_read: 0 }
    }
}

impl<R: std::io::Read> std::io::Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.bytes_read += n;
        Ok(n)
    }
}

/// Red Test: CSV total-byte cap must constrain underlying reader reads before BufReader read-ahead.
/// Uses CountingReader to verify that reads on underlying reader never exceed max_total_bytes + 1 (detection byte).
#[test]
fn test_csv_counting_reader_exact_cap_and_detection_byte() {
    let large_csv = "col1,col2,col3\r\n".to_string() + &"val1,val2,val3\r\n".repeat(1000);
    assert!(large_csv.len() > 10_000);

    let max_cap = 50;
    let options = CsvOptions::csv().with_max_total_bytes(max_cap);

    // 1. Direct CsvParser
    let mut direct_counter = CountingReader::new(Cursor::new(large_csv.as_bytes()));
    let mut parser = lust::data::csv::CsvParser::new(&mut direct_counter, options.clone());
    let res_parser = parser.parse_all();
    assert!(res_parser.is_err(), "Exceeding CSV cap must fail closed");
    assert!(matches!(res_parser.unwrap_err(), CsvError::MaxTotalBytesExceeded(_)));
    assert_eq!(
        direct_counter.bytes_read,
        max_cap + 1,
        "Direct CsvParser underlying reader read {} bytes, expected exact cap + 1 detection byte ({})",
        direct_counter.bytes_read,
        max_cap + 1
    );

    // 2. Public CsvTable::from_reader API
    let mut table_counter = CountingReader::new(Cursor::new(large_csv.as_bytes()));
    let res_table = CsvTable::from_reader(&mut table_counter, &options);
    assert!(res_table.is_err(), "CsvTable::from_reader exceeding cap must fail closed");
    assert!(matches!(res_table.unwrap_err(), CsvError::MaxTotalBytesExceeded(_)));
    assert_eq!(
        table_counter.bytes_read,
        max_cap + 1,
        "CsvTable::from_reader underlying reader read {} bytes, expected exact cap + 1 detection byte ({})",
        table_counter.bytes_read,
        max_cap + 1
    );

    // 3. CsvStreamDecoder public API delegation
    let mut stream_counter = CountingReader::new(Cursor::new(large_csv.as_bytes()));
    let mut stream_decoder = CsvStreamDecoder::new(&mut stream_counter, options.clone()).unwrap();
    let res_stream: Result<Option<Document>, StreamError> = loop {
        match stream_decoder.next_document() {
            Ok(Some(_)) => continue,
            Ok(None) => break Ok(None),
            Err(e) => break Err(e),
        }
    };
    assert!(res_stream.is_err(), "CsvStreamDecoder exceeding cap must fail closed");
    assert_eq!(
        res_stream.unwrap_err(),
        StreamError::InvalidDocument(CsvError::MaxTotalBytesExceeded(max_cap + 1).to_string())
    );
    assert_eq!(
        stream_counter.bytes_read,
        max_cap + 1,
        "CsvStreamDecoder underlying reader read {} bytes, expected exact cap + 1 detection byte ({})",
        stream_counter.bytes_read,
        max_cap + 1
    );
}

/// Red Test: MessagePack map keys and values must be counted separately in aggregate budget with preflight.
#[test]
fn test_msgpack_map_keys_and_values_counted_separately_preflight() {
    // Craft MessagePack map32 with 600,000 declared pairs:
    // 600,000 pairs = 1,200,000 elements, exceeding DEFAULT_MAX_ELEMENTS (1,000,000).
    // Preflight must reject this before proportional allocation or loop iteration.
    let count: u32 = 600_000;
    let mut bytes = Vec::new();
    bytes.push(0xdf); // map32
    bytes.extend_from_slice(&count.to_be_bytes());
    // Provide a few trailing bytes
    bytes.extend_from_slice(&[0xa1, b'k', 0x01]);

    let res = from_msgpack_slice(&bytes);
    assert!(res.is_err(), "Map declaring 600k pairs (1.2M elements) must fail closed under aggregate preflight");
    match res.unwrap_err() {
        MsgPackError::MaxElementsExceeded(cnt) => {
            assert!(cnt > 1_000_000, "Element count {} must exceed 1M", cnt);
        }
        MsgPackError::UnexpectedEof => {
            // Also acceptable if minimum-bytes-per-element truncation preflight triggers first (600,000 * 2 > remaining)
        }
        other => panic!("Expected MaxElementsExceeded or UnexpectedEof, got {:?}", other),
    }
}

/// Red Test: MessagePack map with 500,001 pairs must fail closed because keys and values are counted separately (1,000,002 > 1,000,000).
#[test]
fn test_msgpack_map_keys_and_values_counted_separately_exceeds_budget() {
    // 500,001 pairs = 1,000,002 elements.
    // If only pairs were counted (500,001 <= 1,000,000), it would erroneously pass or not trigger MaxElementsExceeded.
    // But counting keys and values separately (500,001 * 2 = 1,000,002) MUST trigger MaxElementsExceeded.
    let count: u32 = 500_001;
    let mut bytes = Vec::new();
    bytes.push(0xdf); // map32
    bytes.extend_from_slice(&count.to_be_bytes());
    // Minimum valid payload for 500,001 pairs would be large, but aggregate preflight must reject it immediately:
    let res = from_msgpack_slice(&bytes);
    assert!(res.is_err());
    match res.unwrap_err() {
        MsgPackError::MaxElementsExceeded(cnt) => {
            assert_eq!(cnt, 1_000_002, "Preflight must detect 500,001 * 2 = 1,000,002 elements");
        }
        other => panic!("Expected MaxElementsExceeded(1000002) from aggregate preflight, got {:?}", other),
    }
}


/// Red Test: Streams chunk_size > capacity must compute bounded read size before allocation and Read::read.
#[test]
fn test_chunked_stream_chunk_size_larger_than_capacity_bounded_reads() {
    let hostile_payload = "x".repeat(5000);
    let cap = 32;
    let huge_chunk = 65_536; // chunk_size significantly larger than capacity

    // 1. JSON stream decoder
    let mut json_counter = CountingReader::new(Cursor::new(hostile_payload.as_bytes()));
    let json_options = StreamDecoderOptions::new(huge_chunk, cap);
    let mut json_dec = JsonStreamDecoder::new(&mut json_counter, json_options);
    let res_json = json_dec.next_value();
    assert!(res_json.is_err(), "JSON stream exceeding capacity must fail closed");
    assert!(
        json_counter.bytes_read <= cap,
        "JSON stream read {} bytes from reader, which exceeds capacity {}",
        json_counter.bytes_read,
        cap
    );

    // 2. YAML stream decoder
    let mut yaml_counter = CountingReader::new(Cursor::new(hostile_payload.as_bytes()));
    let yaml_options = StreamDecoderOptions::new(huge_chunk, cap);
    let mut yaml_dec = YamlStreamDecoder::new(&mut yaml_counter, yaml_options);
    let res_yaml = yaml_dec.next_document();
    assert!(res_yaml.is_err(), "YAML stream exceeding capacity must fail closed");
    assert!(
        yaml_counter.bytes_read <= cap,
        "YAML stream read {} bytes from reader, which exceeds capacity {}",
        yaml_counter.bytes_read,
        cap
    );

    // 3. TOON stream decoder
    let mut toon_counter = CountingReader::new(Cursor::new(hostile_payload.as_bytes()));
    let toon_options = StreamDecoderOptions::new(huge_chunk, cap);
    let mut toon_dec = ToonStreamDecoder::new(&mut toon_counter, toon_options);
    let res_toon = toon_dec.next_document();
    assert!(res_toon.is_err(), "TOON stream exceeding capacity must fail closed");
    assert!(
        toon_counter.bytes_read <= cap,
        "TOON stream read {} bytes from reader, which exceeds capacity {}",
        toon_counter.bytes_read,
        cap
    );
}

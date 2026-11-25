//! Integration, compile-time layout verification, and parity test suite for `NG-W12-4C`:
//! Rust AOT `#[repr(C)]` schema generation (`src/aot/rust_emitter.rs`, `src/data/layout.rs`,
//! `crates/lust-aot-rt/src/lib.rs`):
//! 1. Compile-time `const _: () = assert!(core::mem::offset_of!(...) == ...);`, `size_of`, and
//!    `align_of` checks verified directly against `LogicalLayoutV1` in pure `#![no_std] + alloc`
//!    with `-C panic=abort`.
//! 2. Negative compile-time mutants proving tampered `offset_of!`, `size_of`, or `align_of`
//!    fail `rustc` compilation.
//! 3. Zero-lookup, unaligned-safe `decode_binary` (using `from_le_bytes`, zero `unsafe`,
//!    zero `transmute`) and zero-DOM streaming `decode_json` with OOM budget enforcement.
//! 4. End-to-end Lust `User.decode(raw)` MIR -> Rust AOT parity against Lust VM (`w2_layout` oracle).

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::cell::RefCell;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::rc::Rc;
use tempfile::tempdir;

use lust::bytecode::Value;
use lust::codegen_zig::aot::{
    AotCompileOptions, AotCompiler, AotTargetKind, RustCompileOptions, RustEmitterOptions,
    emit_rust_from_mir, ensure_cached_aot_rt_rlib,
};
use lust::data::layout::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1, TargetAbi};
use lust::data::materialize::LayoutPerfCounters;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::ir::{MirModule, MirStructLayout};
use lust::mir::lowering::AstLowering;
use lust::mir::vm_backend::{
    MirToBytecodeCompiler, register_struct_decode_natives_with_counters,
};
use lust::parser::Parser;
use lust::vm::VM;

fn find_rustc() -> String {
    std::env::var("RUSTC")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "rustc".to_string())
}

fn assert_safe_and_panic_free(code: &str, label: &str) {
    assert!(
        !code.contains("unsafe"),
        "[{label}] emitted Rust must contain ZERO `unsafe` blocks:\n{code}"
    );
    assert!(
        !code.contains("transmute"),
        "[{label}] emitted Rust must NEVER use `transmute`:\n{code}"
    );
    assert!(
        !code.contains(".unwrap("),
        "[{label}] emitted Rust must contain ZERO `.unwrap()` calls (Hazard 9):\n{code}"
    );
    assert!(
        !code.contains("panic!("),
        "[{label}] emitted Rust must contain ZERO `panic!()` calls (Hazard 9):\n{code}"
    );
}

fn build_comprehensive_schema_module() -> (
    MirModule,
    LogicalLayoutV1,
    LogicalLayoutV1,
    LogicalLayoutV1,
    LogicalLayoutV1,
    LogicalLayoutV1,
) {
    let abi = TargetAbi::host();

    let record_layout = LogicalLayoutV1::new_with_abi(
        9001,
        "Record",
        vec![
            LogicalFieldDesc::new(0, "flag", CanonicalTypeTag::Bool),
            LogicalFieldDesc::new(1, "id", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(2, "oid", CanonicalTypeTag::ObjectId),
            LogicalFieldDesc::new(3, "amount", CanonicalTypeTag::Decimal128),
            LogicalFieldDesc::new(4, "score", CanonicalTypeTag::Float64),
            LogicalFieldDesc::new(5, "name", CanonicalTypeTag::StringSlice),
            LogicalFieldDesc::new(6, "opt_note", CanonicalTypeTag::StringSlice)
                .with_nullability(true, true),
        ],
        abi,
    );

    let account_layout = LogicalLayoutV1::new_with_abi(
        9002,
        "Account",
        vec![
            LogicalFieldDesc::new(0, "id", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(1, "balance", CanonicalTypeTag::UInt64),
            LogicalFieldDesc::new(2, "rate", CanonicalTypeTag::Float64),
            LogicalFieldDesc::new(3, "owner", CanonicalTypeTag::StringSlice),
        ],
        abi,
    );

    let packet_layout = LogicalLayoutV1::new_with_abi(
        9003,
        "Packet",
        vec![
            LogicalFieldDesc::new(0, "valid", CanonicalTypeTag::Bool),
            LogicalFieldDesc::new(1, "opt_flag", CanonicalTypeTag::Bool)
                .with_nullability(true, true),
            LogicalFieldDesc::new(2, "ts", CanonicalTypeTag::DateTimeUtc),
            LogicalFieldDesc::new(3, "payload", CanonicalTypeTag::BinarySlice),
            LogicalFieldDesc::new(4, "opt_payload", CanonicalTypeTag::BinarySlice)
                .with_nullability(true, true),
        ],
        abi,
    );

    let inner_layout = LogicalLayoutV1::new_with_abi(
        9004,
        "Inner",
        vec![
            LogicalFieldDesc::new(0, "x", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(1, "y", CanonicalTypeTag::Float64),
            LogicalFieldDesc::new(2, "tag", CanonicalTypeTag::Bool),
        ],
        abi,
    );

    let outer_layout = LogicalLayoutV1::new_with_abi(
        9005,
        "Outer",
        vec![
            LogicalFieldDesc::new(0, "lead", CanonicalTypeTag::Bool),
            LogicalFieldDesc::new(1, "inner", CanonicalTypeTag::NestedStruct(9004)),
            LogicalFieldDesc::new(2, "tail", CanonicalTypeTag::Int64),
        ],
        abi,
    );

    let empty_layout = LogicalLayoutV1::new_with_abi(9006, "Empty", vec![], abi);

    let wide_fields: Vec<LogicalFieldDesc> = (0..70u16)
        .map(|i| LogicalFieldDesc::new(i, format!("f{i}"), CanonicalTypeTag::Int64))
        .collect();
    let wide_layout = LogicalLayoutV1::new_with_abi(9007, "Wide70", wide_fields, abi);

    let mut module = MirModule::new("schema_suite".to_string());
    module.register_struct_layout(MirStructLayout::from(&record_layout));
    module.register_struct_layout(MirStructLayout::from(&account_layout));
    module.register_struct_layout(MirStructLayout::from(&packet_layout));
    module.register_struct_layout(MirStructLayout::from(&inner_layout));
    module.register_struct_layout(MirStructLayout::from(&outer_layout));
    module.register_struct_layout(MirStructLayout::from(&empty_layout));
    module.register_struct_layout(MirStructLayout::from(&wide_layout));

    (
        module,
        record_layout,
        account_layout,
        packet_layout,
        inner_layout,
        outer_layout,
    )
}

fn compile_nostd_schema_rlib(
    emitted_src: &str,
    out_dir: &Path,
    rt_rlib: &Path,
) -> Result<std::path::PathBuf, String> {
    let rs_path = out_dir.join("gen_schemas.rs");
    let rlib_path = out_dir.join("libgen_schemas.rlib");
    fs::write(&rs_path, emitted_src).map_err(|e| format!("write gen_schemas.rs: {e}"))?;

    let out = Command::new(find_rustc())
        .arg("--edition=2024")
        .arg("--crate-name")
        .arg("gen_schemas")
        .arg("--crate-type=rlib")
        .arg("-C")
        .arg("opt-level=2")
        .arg("-C")
        .arg("panic=abort")
        .arg("--extern")
        .arg(format!("lust_aot_rt={}", rt_rlib.display()))
        .arg(&rs_path)
        .arg("-o")
        .arg(&rlib_path)
        .output()
        .map_err(|e| format!("spawn rustc: {e}"))?;

    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    Ok(rlib_path)
}

// ---------------------------------------------------------------------------
// Test 1: Pure `#![no_std] + alloc` compilation, static `offset_of!` / `size_of` /
//         `align_of` assertions, unaligned `decode_binary`, and strict `decode_json`.
// ---------------------------------------------------------------------------

#[test]
fn test_ng_w12_4c_nostd_repr_c_layout_assertions_and_binary_json_decoders() {
    let (module, record_layout, account_layout, packet_layout, inner_layout, outer_layout) =
        build_comprehensive_schema_module();

    // Verify `LogicalLayoutV1::resolve_nested_layouts` independently
    let resolved_outer = outer_layout
        .resolve_nested_layouts(&|id| {
            if id == inner_layout.type_id {
                Some((inner_layout.size_bytes, inner_layout.align_bytes))
            } else {
                None
            }
        })
        .expect("resolve nested Outer layout");
    assert_eq!(inner_layout.size_bytes, 24);
    assert_eq!(inner_layout.align_bytes, 8);
    assert_eq!(resolved_outer.fields[0].byte_offset, 0);
    assert_eq!(resolved_outer.fields[1].byte_offset, 8);
    assert_eq!(resolved_outer.fields[2].byte_offset, 32);
    assert_eq!(resolved_outer.size_bytes, 40);
    assert_eq!(resolved_outer.align_bytes, 8);

    // Emit pure `#![no_std]` library Rust source
    let emitter_opts = RustEmitterOptions {
        is_library: true,
        no_std: true,
        ..RustEmitterOptions::default()
    };
    let emitted = emit_rust_from_mir(&module, &emitter_opts).expect("emit schema rust");
    assert_safe_and_panic_free(&emitted, "schema_suite");

    assert!(emitted.contains("#![no_std]"), "must emit #![no_std]");
    assert!(emitted.contains("#[repr(C)]"), "must emit #[repr(C)]");
    assert!(
        emitted.contains("core::mem::offset_of!(LustStruct_Record, field_3) == 32"),
        "missing Record field_3 offset_of! assertion:\n{emitted}"
    );
    assert!(
        emitted.contains(&format!(
            "core::mem::size_of::<LustStruct_Record>() == {}",
            record_layout.size_bytes
        )),
        "missing Record size_of assertion"
    );
    assert!(
        emitted.contains(&format!(
            "core::mem::align_of::<LustStruct_Record>() == {}",
            record_layout.align_bytes
        )),
        "missing Record align_of assertion"
    );
    assert!(
        emitted.contains("core::mem::offset_of!(LustStruct_Outer, field_1) == 8")
            && emitted.contains("core::mem::offset_of!(LustStruct_Outer, field_2) == 32")
            && emitted.contains("core::mem::size_of::<LustStruct_Outer>() == 40"),
        "missing resolved nested Outer layout assertions:\n{emitted}"
    );
    assert!(
        emitted.contains("pub fn decode_binary(") && emitted.contains("pub fn decode_json("),
        "must emit decode_binary and decode_json"
    );

    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");

    // 1. Compile generated schemas against pure `#![no_std]` `liblust_aot_rt.rlib` under `-C panic=abort`
    let nostd_rt_opts = RustCompileOptions {
        aot: AotCompileOptions {
            target_kind: AotTargetKind::StaticLibrary,
            release_mode: "ReleaseFast".to_string(),
            ..AotCompileOptions::default()
        },
        no_std: true,
        cache_dir: Some(cache_dir.clone()),
        ..RustCompileOptions::default()
    };
    let nostd_rt_rlib =
        ensure_cached_aot_rt_rlib(&nostd_rt_opts).expect("build nostd lust-aot-rt rlib");
    let gen_rlib = compile_nostd_schema_rlib(&emitted, tmp.path(), &nostd_rt_rlib)
        .expect("pure #![no_std] schema crate with static offset_of!/size_of/align_of assertions must compile");

    // 2. Compile and run a panic=abort test driver exercising every decoder path at runtime
    let driver_rs = tmp.path().join("driver.rs");
    let driver_bin = tmp.path().join("driver_bin");
    let rec_size = record_layout.size_bytes;
    let rec_off_flag = record_layout.fields[0].byte_offset;
    let rec_off_id = record_layout.fields[1].byte_offset;
    let rec_off_oid = record_layout.fields[2].byte_offset;
    let rec_off_amt = record_layout.fields[3].byte_offset;
    let rec_off_score = record_layout.fields[4].byte_offset;
    let rec_off_name = record_layout.fields[5].byte_offset;
    let rec_off_note = record_layout.fields[6].byte_offset;
    let pkt_size = packet_layout.size_bytes;
    let pkt_off_valid = packet_layout.fields[0].byte_offset;
    let pkt_off_opt_flag = packet_layout.fields[1].byte_offset;
    let pkt_off_ts = packet_layout.fields[2].byte_offset;
    let pkt_off_payload = packet_layout.fields[3].byte_offset;
    let pkt_off_opt_payload = packet_layout.fields[4].byte_offset;
    let acc_size = account_layout.size_bytes;

    let driver_code = format!(
        r##"
extern crate gen_schemas;
extern crate lust_aot_rt;

use gen_schemas::{{
    LustStruct_Account, LustStruct_Empty, LustStruct_Inner, LustStruct_Outer,
    LustStruct_Packet, LustStruct_Record, LustStruct_Wide70,
}};
use lust_aot_rt::{{AotCtx, AotDecimal128, AotError, AotObjectId}};

fn write_slice_hdr(buf: &mut [u8], off: usize, data_off: u64, data_len: u64, tag: u64) {{
    buf[off..off + 8].copy_from_slice(&data_off.to_le_bytes());
    buf[off + 8..off + 16].copy_from_slice(&data_len.to_le_bytes());
    buf[off + 16..off + 24].copy_from_slice(&tag.to_le_bytes());
}}

fn main() {{
    let mut ctx = AotCtx::new();

    // --- A. Unaligned binary decode of Record (all canonical types + slices) ---
    let oid = AotObjectId([0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb]);
    let amt = AotDecimal128(12345678901234567890_i128);
    let name_bytes = b"alice";
    let note_bytes = b"vip-note";

    // Start payload at odd byte index 1 to prove unaligned decode has zero UB
    let total_len = {rec_size} + name_bytes.len() + note_bytes.len();
    let mut backing = vec![0x55u8; 1 + total_len];
    {{
        let raw = &mut backing[1..];
        raw.fill(0);
        raw[{rec_off_flag}] = 1;
        raw[{rec_off_id}..{rec_off_id} + 8].copy_from_slice(&(-77_i64).to_le_bytes());
        raw[{rec_off_oid}..{rec_off_oid} + 12].copy_from_slice(&oid.0);
        raw[{rec_off_amt}..{rec_off_amt} + 16].copy_from_slice(&amt.0.to_le_bytes());
        raw[{rec_off_score}..{rec_off_score} + 8].copy_from_slice(&(42.5_f64).to_le_bytes());
        let name_off = {rec_size};
        let note_off = name_off + name_bytes.len();
        write_slice_hdr(raw, {rec_off_name}, name_off as u64, name_bytes.len() as u64, 1);
        write_slice_hdr(raw, {rec_off_note}, note_off as u64, note_bytes.len() as u64, 1);
        raw[name_off..name_off + name_bytes.len()].copy_from_slice(name_bytes);
        raw[note_off..note_off + note_bytes.len()].copy_from_slice(note_bytes);
    }}
    let unaligned = &backing[1..];
    assert_eq!((unaligned.as_ptr() as usize) % 2, 1, "buffer must be genuinely unaligned");
    let rec = LustStruct_Record::decode_binary(unaligned, &mut ctx).expect("unaligned Record binary decode");
    assert_eq!(rec.field_0, true);
    assert_eq!(rec.field_1, -77);
    assert_eq!(rec.field_2, oid);
    assert_eq!(rec.field_3, amt);
    assert_eq!(rec.field_4, 42.5);
    assert_eq!(rec.field_5, "alice");
    assert_eq!(rec.field_6.as_deref(), Some("vip-note"));

    // Optional string = None in binary
    let mut backing_none = backing.clone();
    write_slice_hdr(&mut backing_none[1..], {rec_off_note}, 0, 0, 0);
    let rec_none = LustStruct_Record::decode_binary(&backing_none[1..], &mut ctx).expect("opt_note None");
    assert_eq!(rec_none.field_6, None);

    // Binary error cases: truncated, invalid bool tag, invalid opt slice tag, OOB slice, bad UTF-8, OOM
    assert!(matches!(
        LustStruct_Record::decode_binary(&unaligned[..{rec_size} - 1], &mut ctx),
        Err(AotError::SchemaDecodeError(_))
    ));
    let mut bad_bool = backing.clone();
    bad_bool[1 + {rec_off_flag}] = 2;
    assert!(matches!(
        LustStruct_Record::decode_binary(&bad_bool[1..], &mut ctx),
        Err(AotError::SchemaDecodeError(_))
    ));
    let mut bad_opt_tag = backing.clone();
    write_slice_hdr(&mut bad_opt_tag[1..], {rec_off_note}, 0, 0, 7);
    assert!(matches!(
        LustStruct_Record::decode_binary(&bad_opt_tag[1..], &mut ctx),
        Err(AotError::SchemaDecodeError(_))
    ));
    let mut oob_slice = backing.clone();
    write_slice_hdr(&mut oob_slice[1..], {rec_off_name}, 9999, 10, 1);
    assert!(matches!(
        LustStruct_Record::decode_binary(&oob_slice[1..], &mut ctx),
        Err(AotError::SchemaDecodeError(_))
    ));
    let mut bad_utf8 = backing.clone();
    bad_utf8[1 + {rec_size}] = 0xFF;
    assert!(matches!(
        LustStruct_Record::decode_binary(&bad_utf8[1..], &mut ctx),
        Err(AotError::SchemaDecodeError(_))
    ));
    let mut tiny_ctx = AotCtx::with_memory_limit(2);
    assert!(matches!(
        LustStruct_Record::decode_binary(unaligned, &mut tiny_ctx),
        Err(AotError::OutOfMemory {{ .. }})
    ));

    // --- B. Packet binary decode (arbitrary non-UTF-8 BinarySlice + Option<bool>) ---
    let raw_bin = [0xFFu8, 0x00, 0xFE, 0x80, 0x01];
    let mut pkt_backing = vec![0u8; 1 + {pkt_size} + raw_bin.len()];
    {{
        let p = &mut pkt_backing[1..];
        p[{pkt_off_valid}] = 1;
        p[{pkt_off_opt_flag}] = 2; // None
        p[{pkt_off_ts}..{pkt_off_ts} + 8].copy_from_slice(&1700000000000_i64.to_le_bytes());
        write_slice_hdr(p, {pkt_off_payload}, {pkt_size} as u64, raw_bin.len() as u64, 1);
        write_slice_hdr(p, {pkt_off_opt_payload}, 0, 0, 0);
        p[{pkt_size}..{pkt_size} + raw_bin.len()].copy_from_slice(&raw_bin);
    }}
    let pkt = LustStruct_Packet::decode_binary(&pkt_backing[1..], &mut ctx).expect("Packet binary decode");
    assert!(pkt.field_0);
    assert_eq!(pkt.field_1, None);
    assert_eq!(pkt.field_2, 1700000000000);
    assert_eq!(pkt.field_3, raw_bin);
    assert_eq!(pkt.field_4, None);

    let mut bad_opt_bool = pkt_backing.clone();
    bad_opt_bool[1 + {pkt_off_opt_flag}] = 3;
    assert!(matches!(
        LustStruct_Packet::decode_binary(&bad_opt_bool[1..], &mut ctx),
        Err(AotError::SchemaDecodeError(_))
    ));

    // --- C. Nested Outer {{ Inner }} binary + JSON decode ---
    let mut outer_backing = vec![0u8; 1 + 40];
    {{
        let o = &mut outer_backing[1..];
        o[0] = 1;
        o[8..16].copy_from_slice(&123_i64.to_le_bytes());
        o[16..24].copy_from_slice(&(3.25_f64).to_le_bytes());
        o[24] = 1;
        o[32..40].copy_from_slice(&(-999_i64).to_le_bytes());
    }}
    let outer_bin = LustStruct_Outer::decode_binary(&outer_backing[1..], &mut ctx).expect("Outer binary");
    let outer_json = LustStruct_Outer::decode_json(
        r#"{{"tail": -999, "inner": {{"tag": true, "y": 3.25, "x": 123}}, "lead": true}}"#,
        &mut ctx,
    ).expect("Outer json");
    assert_eq!(outer_bin, outer_json);
    assert_eq!(outer_bin.field_1, LustStruct_Inner {{ field_0: 123, field_1: 3.25, field_2: true }});

    // --- D. Empty struct binary + JSON ---
    assert_eq!(LustStruct_Empty::decode_binary(&[], &mut ctx), Ok(LustStruct_Empty {{}}));
    assert_eq!(LustStruct_Empty::decode_json("{{}}", &mut ctx), Ok(LustStruct_Empty {{}}));
    assert_eq!(
        LustStruct_Empty::decode_json(r#"{{"_type": "Empty"}}"#, &mut ctx),
        Ok(LustStruct_Empty {{}})
    );

    // --- E. Record JSON decode (shuffled, nominal _type, null optional, escapes, UTF-8) ---
    let rec_json = LustStruct_Record::decode_json(
        r#"{{
            "_type": "Record",
            "opt_note": null,
            "name": "alice\n\u0041\uD83D\uDE00",
            "score": 42.5,
            "amount": 12345678901234567890,
            "oid": "00112233445566778899aabb",
            "id": -77,
            "flag": true
        }}"#,
        &mut ctx,
    ).expect("Record JSON decode");
    assert_eq!(rec_json.field_0, true);
    assert_eq!(rec_json.field_1, -77);
    assert_eq!(rec_json.field_2, oid);
    assert_eq!(rec_json.field_3, amt);
    assert_eq!(rec_json.field_4, 42.5);
    assert_eq!(rec_json.field_5, "alice\nA😀");
    assert_eq!(rec_json.field_6, None);

    // Omitted optional field succeeds
    let rec_omit = LustStruct_Record::decode_json(
        r#"{{"flag":false,"id":1,"oid":"00112233445566778899aabb","amount":0,"score":0.0,"name":"ok"}}"#,
        &mut ctx,
    ).expect("Record JSON omitted optional");
    assert_eq!(rec_omit.field_6, None);

    // --- F. Account JSON strict error matrix (matches w2_layout.rs) ---
    assert_eq!(LustStruct_Account::LOGICAL_SIZE_BYTES, {acc_size});
    let acc_ord = LustStruct_Account::decode_json(
        r#"{{"id": 10, "balance": 500, "rate": 1.25, "owner": "bob"}}"#,
        &mut ctx,
    ).expect("ordered account");
    let acc_shuf = LustStruct_Account::decode_json(
        r#"{{"owner": "bob", "rate": 1.25, "balance": 500, "id": 10}}"#,
        &mut ctx,
    ).expect("shuffled account");
    assert_eq!(acc_ord, acc_shuf);
    assert_eq!(acc_ord.field_1, 500_u64);

    let bad_accounts = [
        // duplicate key
        r#"{{"id": 10, "id": 10, "balance": 500, "rate": 1.25, "owner": "bob"}}"#,
        // i64 overflow
        r#"{{"id": 9223372036854775808, "balance": 500, "rate": 1.25, "owner": "bob"}}"#,
        // i64 given float
        r#"{{"id": 1.5, "balance": 500, "rate": 1.25, "owner": "bob"}}"#,
        // u64 negative
        r#"{{"id": 10, "balance": -1, "rate": 1.25, "owner": "bob"}}"#,
        // f64 overflow to inf
        r#"{{"id": 10, "balance": 500, "rate": 1e309, "owner": "bob"}}"#,
        // nominal type mismatch
        r#"{{"_type": "Other", "id": 10, "balance": 500, "rate": 1.25, "owner": "bob"}}"#,
        // duplicate nominal tag
        r#"{{"_type": "Account", "$type": "Account", "id": 10, "balance": 500, "rate": 1.25, "owner": "bob"}}"#,
        // extra unknown field
        r#"{{"id": 10, "balance": 500, "rate": 1.25, "owner": "bob", "extra": 1}}"#,
        // null for non-nullable field
        r#"{{"id": 10, "balance": 500, "rate": 1.25, "owner": null}}"#,
        // missing required field
        r#"{{"id": 10, "balance": 500, "rate": 1.25}}"#,
        // trailing comma
        r#"{{"id": 10, "balance": 500, "rate": 1.25, "owner": "bob",}}"#,
        // trailing junk
        r#"{{"id": 10, "balance": 500, "rate": 1.25, "owner": "bob"}} trailing"#,
    ];
    for (idx, payload) in bad_accounts.iter().enumerate() {{
        assert!(
            matches!(
                LustStruct_Account::decode_json(payload, &mut ctx),
                Err(AotError::SchemaDecodeError(_))
            ),
            "bad account case #{{idx}} must fail with SchemaDecodeError: {{payload}}"
        );
    }}

    // Tight memory limit enforces OOM on JSON string decode without panicking
    let mut json_oom_ctx = AotCtx::with_memory_limit(2);
    assert!(matches!(
        LustStruct_Account::decode_json(
            r#"{{"id": 10, "balance": 500, "rate": 1.25, "owner": "this string exceeds two bytes"}}"#,
            &mut json_oom_ctx
        ),
        Err(AotError::OutOfMemory {{ .. }})
    ));

    // --- G. Wide70 (>64 fields) shuffled JSON decode + duplicate/missing field 69 ---
    let mut wide_json = String::from("{{");
    for i in (0..70usize).rev() {{
        if i < 69 {{ wide_json.push(','); }}
        wide_json.push_str(&format!("\"f{{i}}\":{{}}", (i as i64) * 3 + 1));
    }}
    wide_json.push('}}');
    let wide = LustStruct_Wide70::decode_json(&wide_json, &mut ctx).expect("Wide70 JSON");
    assert_eq!(wide.field_0, 1);
    assert_eq!(wide.field_69, 208);

    let mut wide_missing = String::from("{{");
    for i in 0..69usize {{
        if i > 0 {{ wide_missing.push(','); }}
        wide_missing.push_str(&format!("\"f{{i}}\":{{i}}"));
    }}
    wide_missing.push('}}');
    assert!(matches!(
        LustStruct_Wide70::decode_json(&wide_missing, &mut ctx),
        Err(AotError::SchemaDecodeError(_))
    ));

    println!("ALL_SCHEMA_CHECKS_PASSED");
}}
"##
    );

    fs::write(&driver_rs, driver_code).expect("write driver.rs");
    let rt_dir = nostd_rt_rlib
        .parent()
        .expect("nostd_rt_rlib parent dir");
    let rustc_out = Command::new(find_rustc())
        .arg("--edition=2024")
        .arg("-C")
        .arg("opt-level=2")
        .arg("-C")
        .arg("panic=abort")
        .arg("-L")
        .arg(format!("dependency={}", rt_dir.display()))
        .arg("--extern")
        .arg(format!("lust_aot_rt={}", nostd_rt_rlib.display()))
        .arg("--extern")
        .arg(format!("gen_schemas={}", gen_rlib.display()))
        .arg(&driver_rs)
        .arg("-o")
        .arg(&driver_bin)
        .output()
        .expect("compile schema driver");
    assert!(
        rustc_out.status.success(),
        "schema driver failed to compile:\n{}",
        String::from_utf8_lossy(&rustc_out.stderr)
    );

    let run = Command::new(&driver_bin).output().expect("run schema driver");
    assert!(
        run.status.success(),
        "schema driver execution failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).trim(),
        "ALL_SCHEMA_CHECKS_PASSED"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Negative compile-time layout mutants MUST fail rustc compilation
// ---------------------------------------------------------------------------

#[test]
fn test_ng_w12_4c_compile_time_offset_and_size_mutants_must_fail_rustc() {
    let (module, record_layout, _, _, _, _) = build_comprehensive_schema_module();
    let emitter_opts = RustEmitterOptions {
        is_library: true,
        no_std: true,
        ..RustEmitterOptions::default()
    };
    let emitted = emit_rust_from_mir(&module, &emitter_opts).expect("emit schema rust");

    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");
    let nostd_rt_opts = RustCompileOptions {
        aot: AotCompileOptions {
            target_kind: AotTargetKind::StaticLibrary,
            release_mode: "ReleaseFast".to_string(),
            ..AotCompileOptions::default()
        },
        no_std: true,
        cache_dir: Some(cache_dir),
        ..RustCompileOptions::default()
    };
    let rt_rlib = ensure_cached_aot_rt_rlib(&nostd_rt_opts).expect("build nostd rt rlib");

    // Mutant 1: tampered `offset_of!`
    let bad_offset = emitted.replacen(
        "core::mem::offset_of!(LustStruct_Record, field_3) == 32",
        "core::mem::offset_of!(LustStruct_Record, field_3) == 33",
        1,
    );
    assert_ne!(bad_offset, emitted);
    let err1 = compile_nostd_schema_rlib(&bad_offset, tmp.path(), &rt_rlib)
        .expect_err("tampered offset_of! mutant must fail rustc compile-time assertion");
    assert!(
        err1.contains("assertion failed") || err1.contains("E0080"),
        "expected const-eval assertion error, got:\n{err1}"
    );

    // Mutant 2: tampered `size_of`
    let good_size = format!(
        "core::mem::size_of::<LustStruct_Record>() == {}",
        record_layout.size_bytes
    );
    let bad_size_stmt = format!(
        "core::mem::size_of::<LustStruct_Record>() == {}",
        record_layout.size_bytes + 8
    );
    let bad_size = emitted.replacen(&good_size, &bad_size_stmt, 1);
    assert_ne!(bad_size, emitted);
    let err2 = compile_nostd_schema_rlib(&bad_size, tmp.path(), &rt_rlib)
        .expect_err("tampered size_of mutant must fail rustc compile-time assertion");
    assert!(
        err2.contains("assertion failed") || err2.contains("E0080"),
        "expected const-eval assertion error, got:\n{err2}"
    );

    // Mutant 3: tampered `align_of`
    let good_align = format!(
        "core::mem::align_of::<LustStruct_Record>() == {}",
        record_layout.align_bytes
    );
    let bad_align_stmt = format!(
        "core::mem::align_of::<LustStruct_Record>() == {}",
        record_layout.align_bytes / 2
    );
    let bad_align = emitted.replacen(&good_align, &bad_align_stmt, 1);
    assert_ne!(bad_align, emitted);
    let err3 = compile_nostd_schema_rlib(&bad_align, tmp.path(), &rt_rlib)
        .expect_err("tampered align_of mutant must fail rustc compile-time assertion");
    assert!(
        err3.contains("assertion failed") || err3.contains("E0080"),
        "expected const-eval assertion error, got:\n{err3}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: End-to-end Lust `User.decode(raw)` MIR -> Rust AOT parity with VM
// ---------------------------------------------------------------------------

#[test]
fn test_ng_w12_4c_lust_user_decode_rust_aot_vm_parity_and_fail_closed() {
    let src = r#"
struct User
    id: int,
    score: float,
    active: bool,
    name: string
end

function eval_user(raw: string): int
    local u: User = User.decode(raw)
    local s: int = u.id
    if u.active then
        s = s + 100
    end
    return s
end

function main(): int
    local a: int = eval_user("{\"name\":\"a\",\"active\":true,\"score\":1.5,\"id\":7}")
    local b: int = eval_user("{\"id\":42,\"score\":2.5,\"active\":false,\"name\":\"b\"}")
    local c: int = eval_user("{\"score\":9.0,\"id\":-5,\"name\":\"c\",\"active\":true}")
    println(a)
    println(b)
    println(c)
    return 0
end
"#;

    let mut interner = Interner::new();
    let tokens = Lexer::new(src, &mut interner).tokenize().expect("lex");
    let items = Parser::new(tokens).parse().expect("parse");
    let module = AstLowering::lower_module_items("w12_4c_user", &items).expect("lower MIR");

    // 1. VM reference oracle (identical setup to w2_layout.rs)
    let bc_fns = MirToBytecodeCompiler::compile_module(&module).expect("compile vm bytecode");
    let counters = Rc::new(RefCell::new(LayoutPerfCounters::default()));
    let mut vm = VM::new();
    vm.load_functions(bc_fns);
    register_struct_decode_natives_with_counters(&mut vm, &module.struct_layouts, counters.clone());

    let cases = [
        (r#"{"name":"a","active":true,"score":1.5,"id":7}"#, 107i64),
        (r#"{"id":42,"score":2.5,"active":false,"name":"b"}"#, 42i64),
        (r#"{"score":9.0,"id":-5,"name":"c","active":true}"#, 95i64),
    ];
    let mut vm_expected = String::new();
    for (json, expected) in cases {
        let res = vm
            .call("eval_user", vec![Value::string(json.to_string())])
            .expect("VM eval_user");
        assert_eq!(res, Value::Int(expected));
        vm_expected.push_str(&format!("{expected}\n"));
    }

    // 2. Rust AOT compilation & execution
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");
    let opts = RustCompileOptions {
        aot: AotCompileOptions {
            target_kind: AotTargetKind::Executable,
            release_mode: "ReleaseFast".to_string(),
            ..AotCompileOptions::default()
        },
        cache_dir: Some(cache_dir.clone()),
        ..RustCompileOptions::default()
    };

    let emitted =
        emit_rust_from_mir(&module, &opts.to_emitter_options()).expect("emit User Rust AOT");
    assert_safe_and_panic_free(&emitted, "user_decode");
    assert!(emitted.contains("core::mem::offset_of!(LustStruct_User, field_0) == 0"));
    assert!(emitted.contains("core::mem::offset_of!(LustStruct_User, field_1) == 8"));
    assert!(emitted.contains("core::mem::offset_of!(LustStruct_User, field_2) == 16"));
    assert!(emitted.contains("core::mem::offset_of!(LustStruct_User, field_3) == 24"));
    assert!(emitted.contains("core::mem::size_of::<LustStruct_User>() == 48"));
    assert!(emitted.contains("core::mem::align_of::<LustStruct_User>() == 8"));
    assert!(emitted.contains("LustStruct_User::decode_json("));

    let bin = tmp.path().join("user_decode_bin");
    AotCompiler::compile_module_rust(&module, &bin, &opts).expect("compile User AOT bin");
    let run = Command::new(&bin).output().expect("run User AOT bin");
    assert!(
        run.status.success(),
        "User AOT bin failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&run.stdout), vm_expected);

    // 3. Fail-closed negative AOT binary cases (duplicate key, i64 overflow, missing field)
    let bad_inputs = [
        r#"{\"id\":1,\"id\":2,\"score\":1.0,\"active\":true,\"name\":\"x\"}"#,
        r#"{\"id\":9223372036854775808,\"score\":1.0,\"active\":true,\"name\":\"x\"}"#,
        r#"{\"id\":1,\"score\":1.0,\"active\":true}"#,
    ];
    for (idx, bad_json) in bad_inputs.iter().enumerate() {
        let neg_src = format!(
            r#"
struct User
    id: int,
    score: float,
    active: bool,
    name: string
end

function main(): int
    local u: User = User.decode("{bad_json}")
    return u.id
end
"#
        );
        let mut neg_interner = Interner::new();
        let neg_tokens = Lexer::new(&neg_src, &mut neg_interner)
            .tokenize()
            .expect("lex neg");
        let neg_items = Parser::new(neg_tokens).parse().expect("parse neg");
        let neg_module = AstLowering::lower_module_items("neg_user", &neg_items).expect("lower neg");
        let neg_bin = tmp.path().join(format!("neg_user_{idx}"));
        AotCompiler::compile_module_rust(&neg_module, &neg_bin, &opts).expect("compile neg bin");
        let neg_run = Command::new(&neg_bin).output().expect("run neg bin");
        assert!(
            !neg_run.status.success(),
            "negative case #{idx} must fail closed"
        );
        assert_eq!(
            neg_run.status.code(),
            Some(1),
            "negative case #{idx} must exit cleanly with status 1 (not abort signal)"
        );
        let stderr = String::from_utf8_lossy(&neg_run.stderr);
        assert!(
            stderr.contains("Schema decode error"),
            "negative case #{idx} must report Schema decode error, got: {stderr}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 4: Nullable primitive structs & cyclic structs coexist cleanly
// ---------------------------------------------------------------------------

#[test]
fn test_ng_w12_4c_nullable_primitive_and_cyclic_structs_compile_cleanly() {
    let abi = TargetAbi::host();
    let opt_prim_layout = LogicalLayoutV1::new_with_abi(
        9101,
        "OptPrim",
        vec![
            LogicalFieldDesc::new(0, "id", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(1, "maybe_count", CanonicalTypeTag::Int64)
                .with_nullability(true, true),
        ],
        abi,
    );
    let cyclic_layout = LogicalLayoutV1::new_with_abi(
        9102,
        "Node",
        vec![
            LogicalFieldDesc::new(0, "val", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(1, "next", CanonicalTypeTag::NestedStruct(9102))
                .with_nullability(true, true),
        ],
        abi,
    );

    let mut module = MirModule::new("mixed_structs".to_string());
    module.register_struct_layout(MirStructLayout::from(&opt_prim_layout));
    module.register_struct_layout(MirStructLayout::from(&cyclic_layout));

    let emitted = emit_rust_from_mir(
        &module,
        &RustEmitterOptions {
            is_library: true,
            no_std: true,
            ..RustEmitterOptions::default()
        },
    )
    .expect("emit mixed structs");
    assert_safe_and_panic_free(&emitted, "mixed_structs");

    // OptPrim has Option<i64> (16B vs LogicalLayoutV1 8B) -> no false size_of assertion, still has decode_json!
    assert!(!emitted.contains("size_of::<LustStruct_OptPrim>()"));
    assert!(emitted.contains("impl LustStruct_OptPrim"));
    assert!(emitted.contains("pub fn decode_json("));

    // Cyclic Node has no false LogicalLayoutV1 assertions and compiles cleanly in `#![no_std]`
    assert!(!emitted.contains("size_of::<LustStruct_Node"));

    let tmp = tempdir().expect("tempdir");
    let nostd_rt_opts = RustCompileOptions {
        aot: AotCompileOptions {
            target_kind: AotTargetKind::StaticLibrary,
            release_mode: "ReleaseFast".to_string(),
            ..AotCompileOptions::default()
        },
        no_std: true,
        cache_dir: Some(tmp.path().join("rlib_cache")),
        ..RustCompileOptions::default()
    };
    let rt_rlib = ensure_cached_aot_rt_rlib(&nostd_rt_opts).expect("build nostd rt");
    compile_nostd_schema_rlib(&emitted, tmp.path(), &rt_rlib)
        .expect("mixed nullable primitive + cyclic struct crate must compile in #![no_std]");
}

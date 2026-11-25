//! Wave 17 Track NG-W17-3 Integration & Verification Suite:
//! Layout / C FFI — D-LAYOUT Gate-SCHEMA-09 & Gate-SCHEMA-13.
//!
//! Strictly safe Rust (`#![forbid(unsafe_code)]`, `TASK_ALLOWS_UNSAFE=0`).
//! All C fixtures compile via host `cc` with `-fsanitize=address,undefined` (when supported)
//! and execute in isolated child processes guarded by an external safety watchdog.

#![cfg(feature = "std")]
#![forbid(unsafe_code)]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

// Re-export library modules at crate root so `src/w17_layout/` compiles identically
// both standalone here (before 17C registration) and inside `src/lib.rs` (after 17C).
#[allow(unused_imports)]
use lust::{
    alloc_budget, ast, bytecode, codegen_zig, config, data, effect_abi, embed, intern, lexer, mir,
    model, parser, schema, vm,
};

#[path = "../src/w17_layout/mod.rs"]
pub mod w17_layout;

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::rc::Rc;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use tempfile::tempdir;

use lust::alloc_budget::VmAllocHandle;
use lust::bytecode::{Chunk, Instruction, Value};
use lust::codegen_zig::aot::{
    AotCompileOptions, AotCompiler, AotTargetKind, RustCompileOptions, emit_rust_from_mir,
};
use lust::config::CapabilityMask;
use lust::data::layout::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1, TargetAbi};
use lust::data::materialize::{
    AlignedStorage, ContiguousDecodeOptions, ContiguousStruct, LayoutPerfCounters,
    MaterializeError, decode_json_into_contiguous,
};
use lust::data::owner::LeaseError;
use lust::data::{DataValue, Document, ExactNumber};
use lust::effect_abi::{Cause, Exit, InterruptReason};
use lust::embed::EmbeddedProgram;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{
    CRANELIFT_NATIVE_ENTRY_COUNT, reset_native_instrumentation_counters, safepoint_reset,
};
use lust::mir::ir::{MirStructOp, mir_struct_decode_callee, parse_mir_struct_decode_callee};
use lust::mir::lowering::AstLowering;
use lust::mir::vm_backend::{
    MirToBytecodeCompiler, audit_chunk_field_access, register_struct_decode_natives_with_counters,
};
use lust::parser::Parser;
use lust::vm::VM;
use lust::LustConfig;

use w17_layout::c_abi::{
    CBuildEvidence, audit_c_accessor_disassembly, build_c_fixture_with_sanitizers,
    compare_layout_with_c_entry, compile_c_negative_layout_mutant, run_c_layout_report,
    verify_zig_layout_probe,
};
use w17_layout::ffi_boundary::{
    FfiAbiDescriptor, FfiBoundaryRuntime, FfiExecutionMode, FfiManifest, FfiManifestError,
    FfiMutationMode, FfiOwnership, FfiStatusKind, FfiStatusRule, NativeCallError,
};
use w17_layout::model_contract::{
    BindingOrigin, DbOperation, GeneratedIdPolicy, InsertInput, ModelContract, ModelContractError,
    PatchField, PatchInput, RecordingDbAdapter, StructBindingPlan, WireAliasMap,
    decode_json_with_plan,
};
use w17_layout::watchdog::{WatchdogOutcome, run_child_with_watchdog};

static GLOBAL_CRANELIFT_LOCK: Mutex<()> = Mutex::new(());
static C_LAYOUT_PROBE_CACHE: OnceLock<CBuildEvidence> = OnceLock::new();
static C_FFI_FIXTURE_CACHE: OnceLock<CBuildEvidence> = OnceLock::new();

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("w17_layout")
}

fn c_build_cache_dir() -> PathBuf {
    let dir = PathBuf::from("/tmp/lust_eval_target/w17-3/c_fixtures");
    fs::create_dir_all(&dir).expect("create /tmp/lust_eval_target/w17-3/c_fixtures");
    dir
}

fn built_c_layout_probe() -> &'static CBuildEvidence {
    C_LAYOUT_PROBE_CACHE.get_or_init(|| {
        let dir = fixture_dir();
        let src = dir.join("c_layout_probe.c");
        let bin = c_build_cache_dir().join("c_layout_probe");
        build_c_fixture_with_sanitizers(&src, &dir, &bin, "-O2")
            .expect("compile c_layout_probe.c with host cc")
    })
}

fn built_c_ffi_fixture() -> &'static CBuildEvidence {
    C_FFI_FIXTURE_CACHE.get_or_init(|| {
        let dir = fixture_dir();
        let src = dir.join("c_ffi_boundary_fixture.c");
        let bin = c_build_cache_dir().join("c_ffi_boundary_fixture");
        build_c_fixture_with_sanitizers(&src, &dir, &bin, "-O1")
            .expect("compile c_ffi_boundary_fixture.c with host cc")
    })
}

fn build_standard_layouts() -> (
    LogicalLayoutV1,
    LogicalLayoutV1,
    LogicalLayoutV1,
    LogicalLayoutV1,
    LogicalLayoutV1,
    LogicalLayoutV1,
) {
    let abi = TargetAbi::host();
    let record = LogicalLayoutV1::new_with_abi(
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
    let account = LogicalLayoutV1::new_with_abi(
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
    let packet = LogicalLayoutV1::new_with_abi(
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
    let inner = LogicalLayoutV1::new_with_abi(
        9004,
        "Inner",
        vec![
            LogicalFieldDesc::new(0, "x", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(1, "y", CanonicalTypeTag::Float64),
            LogicalFieldDesc::new(2, "tag", CanonicalTypeTag::Bool),
        ],
        abi,
    );
    let outer_raw = LogicalLayoutV1::new_with_abi(
        9005,
        "Outer",
        vec![
            LogicalFieldDesc::new(0, "lead", CanonicalTypeTag::Bool),
            LogicalFieldDesc::new(1, "inner", CanonicalTypeTag::NestedStruct(9004)),
            LogicalFieldDesc::new(2, "tail", CanonicalTypeTag::Int64),
        ],
        abi,
    );
    let outer = outer_raw
        .resolve_nested_layouts(&|id| {
            if id == 9004 {
                Some((inner.size_bytes, inner.align_bytes))
            } else {
                None
            }
        })
        .expect("resolve nested Outer layout");
    let user = LogicalLayoutV1::new_with_abi(
        9006,
        "User",
        vec![
            LogicalFieldDesc::new(0, "id", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(1, "score", CanonicalTypeTag::Float64),
            LogicalFieldDesc::new(2, "active", CanonicalTypeTag::Bool),
            LogicalFieldDesc::new(3, "name", CanonicalTypeTag::StringSlice),
        ],
        abi,
    );
    (record, account, packet, inner, outer, user)
}

fn permitted_store_caps() -> CapabilityMask {
    let mut caps = CapabilityMask::none();
    caps.allow_ffi = true;
    caps.custom_capabilities.insert("Store.read".to_string());
    caps
}

// =========================================================================
// 1. C ABI sizeof / alignof / offsetof, Raw-Byte Parity & Binding Checks
// =========================================================================

#[test]
fn test_w17_c_abi_sizeof_alignof_offsetof_and_raw_bytes_parity() {
    let build = built_c_layout_probe();
    assert!(
        build.sanitizers_enabled,
        "Expected macOS host cc to support -fsanitize=address,undefined; cmd={}",
        build.command_line
    );
    eprintln!(
        "[W17-3 C BUILD] compiler={} version={} cmd={}",
        build.compiler, build.compiler_version, build.command_line
    );

    let (report, run) =
        run_c_layout_report(&build.output_binary, Duration::from_secs(10)).expect("c layout report");
    assert!(
        !run.outcome.is_watchdog_kill(),
        "c_layout_probe must not trigger watchdog"
    );
    assert_eq!(report.pointer_width, std::mem::size_of::<usize>());

    // Verify RFC Section 10 & 10.2 ABI descriptor sizes and alignments on host C compiler.
    assert_eq!(report.descriptors["LustOwnerHandle"].size, 16);
    assert_eq!(report.descriptors["LustOwnerHandle"].align, 8);
    assert_eq!(report.descriptors["LustDataView"].size, 48);
    assert_eq!(report.descriptors["LustDataView"].align, 8);
    assert_eq!(report.descriptors["LustSchemaDescriptor"].size, 64);
    assert_eq!(report.descriptors["LustStructBindingInfoV1"].size, 64);
    assert_eq!(report.descriptors["LustStructFieldInfoV1"].size, 56);

    // Verify all 6 runtime LogicalLayoutV1 definitions match host C compiler exactly.
    let (record, account, packet, inner, outer, user) = build_standard_layouts();
    for layout in [&record, &account, &packet, &inner, &outer, &user] {
        let c_entry = report
            .structs
            .get(&layout.type_name)
            .unwrap_or_else(|| panic!("missing C entry for {}", layout.type_name));
        compare_layout_with_c_entry(layout, c_entry)
            .unwrap_or_else(|e| panic!("C ABI mismatch: {e}"));
    }

    // Decode Record JSON in Rust ContiguousStruct, write raw memory image, and verify host C
    // reads identical values at exact C offsetof + runs N=10,000 scalar reads with 0 allocs.
    let payload = br#"{
        "_type": "Record",
        "opt_note": null,
        "name": "alice_w17",
        "score": 42.5,
        "amount": 12345678901234567890,
        "oid": "00112233445566778899aabb",
        "id": -77,
        "flag": true
    }"#;
    let mut counters = LayoutPerfCounters::default();
    let cs = decode_json_into_contiguous(
        &record,
        payload,
        &mut counters,
        &ContiguousDecodeOptions {
            require_nominal_tag: true,
            ..Default::default()
        },
    )
    .expect("direct contiguous decode");
    assert_eq!(counters.dom_node_constructions, 0);
    assert_eq!(counters.intermediate_map_allocations, 0);
    assert_eq!(cs.raw_bytes().len(), record.size_bytes);

    let tmp = tempdir().expect("tempdir for raw bytes");
    let raw_path = tmp.path().join("record.bin");
    fs::write(&raw_path, cs.raw_bytes()).expect("write raw contiguous bytes");

    let raw_arg = raw_path.display().to_string();
    let verify_run = run_child_with_watchdog(
        &build.output_binary,
        &["--verify-record-bytes", &raw_arg],
        &[],
        Duration::from_secs(10),
    )
    .expect("run c verify-record-bytes");
    assert!(
        verify_run.outcome.is_clean_success(),
        "C raw-bytes verification failed: outcome={:?} stderr={}",
        verify_run.outcome,
        verify_run.stderr
    );

    let c_val: serde_json::Value =
        serde_json::from_str(verify_run.stdout.trim()).expect("parse C verify json");
    assert_eq!(c_val["flag"].as_u64(), Some(1));
    assert_eq!(c_val["id"].as_i64(), Some(-77));
    assert_eq!(
        c_val["oid_hex"].as_str(),
        Some("00112233445566778899aabb")
    );
    let expected_dec = 12345678901234567890i128.to_ne_bytes();
    let exp_lo = u64::from_ne_bytes(expected_dec[0..8].try_into().unwrap());
    let exp_hi = u64::from_ne_bytes(expected_dec[8..16].try_into().unwrap());
    assert_eq!(c_val["dec_lo"].as_u64(), Some(exp_lo));
    assert_eq!(c_val["dec_hi"].as_u64(), Some(exp_hi));
    assert!((c_val["score"].as_f64().unwrap() - 42.5).abs() < 1e-9);
    assert_eq!(c_val["name_len"].as_u64(), Some(9));
    assert_eq!(c_val["name_owner_id"].as_u64(), Some(6)); // ordinal 5 + 1
    assert_eq!(c_val["opt_note_len"].as_u64(), Some(0));
    assert_eq!(c_val["opt_note_owner_id"].as_u64(), Some(0));
    assert_eq!(c_val["eval_once"].as_i64(), Some(23)); // -77 + 100
    assert_eq!(c_val["eval_10k_sum"].as_i64(), Some(230_000));
    assert_eq!(
        c_val["scalar_read_allocs"].as_u64(),
        Some(0),
        "10,000 prepared C scalar reads must perform 0 heap allocations"
    );

    // Run C binding info positive/negative/partial-init cleanup checks under ASan+UBSan.
    let bind_run = run_child_with_watchdog(
        &build.output_binary,
        &["--binding-info-checks"],
        &[],
        Duration::from_secs(10),
    )
    .expect("run c binding-info-checks");
    assert!(
        bind_run.outcome.is_clean_success(),
        "C binding-info-checks failed: outcome={:?} stderr={}",
        bind_run.outcome,
        bind_run.stderr
    );
    let bind_val: serde_json::Value =
        serde_json::from_str(bind_run.stdout.trim()).expect("parse binding-info json");
    assert_eq!(bind_val["binding_checks"].as_str(), Some("PASS"));
    assert_eq!(bind_val["allocs"].as_u64(), Some(2));
    assert_eq!(bind_val["frees"].as_u64(), Some(2));
    assert_eq!(bind_val["cleanup_order_checksum"].as_u64(), Some(63));
}

// =========================================================================
// 2. C ABI Negative Layout Mutants (Compile-Fail Proofs)
// =========================================================================

#[test]
fn test_w17_c_abi_negative_layout_mutants_fail_cc() {
    let inc = fixture_dir();

    let err_size =
        compile_c_negative_layout_mutant(&inc, "tampered_sizeof", "sizeof(LustCRecord) == 120")
            .expect("tampered sizeof mutant must fail cc");
    assert!(
        err_size.contains("static_assert") || err_size.contains("static assertion"),
        "expected static_assert error, got: {err_size}"
    );

    let err_align =
        compile_c_negative_layout_mutant(&inc, "tampered_alignof", "alignof(LustCRecord) == 8")
            .expect("tampered alignof mutant must fail cc");
    assert!(
        err_align.contains("static_assert") || err_align.contains("static assertion"),
        "expected static_assert error, got: {err_align}"
    );

    let err_offset = compile_c_negative_layout_mutant(
        &inc,
        "tampered_offsetof",
        "offsetof(LustCRecord, amount) == 28",
    )
    .expect("tampered offsetof mutant must fail cc");
    assert!(
        err_offset.contains("static_assert") || err_offset.contains("static assertion"),
        "expected static_assert error, got: {err_offset}"
    );
}

// =========================================================================
// 3. Disassembly, Bytecode & MIR Zero-Lookup Proof (Gate-SCHEMA-09)
// =========================================================================

#[test]
fn test_w17_c_disassembly_and_vm_bytecode_zero_lookup() {
    // 1. Host C binary disassembly proof for constant-offset scalar field loads.
    let build = built_c_layout_probe();
    let disasm = audit_c_accessor_disassembly(&build.output_binary, "lust_c_record_read_id_and_flag")
        .expect("audit disassembly of lust_c_record_read_id_and_flag");
    assert!(
        disasm.is_zero_lookup_direct(),
        "C accessor disassembly must use direct constant-offset loads with zero lookup calls:\n{}",
        disasm.raw_excerpt
    );

    let disasm_user = audit_c_accessor_disassembly(&build.output_binary, "lust_c_user_eval")
        .expect("audit disassembly of lust_c_user_eval");
    assert!(
        disasm_user.is_zero_lookup_direct(),
        "lust_c_user_eval must use direct constant-offset loads:\n{}",
        disasm_user.raw_excerpt
    );

    // 2. MIR & VM bytecode zero-lookup proof.
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
    "#;
    let mut interner = Interner::new();
    let tokens = Lexer::new(src, &mut interner).tokenize().expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");
    let module = AstLowering::lower_module_items("w17_zero_lookup", &items).expect("lower MIR");

    assert!(module.struct_ops.iter().any(|op| matches!(
        op,
        MirStructOp::StructDecodeInto { struct_name, .. } if struct_name == "User"
    )));
    assert!(module.struct_ops.iter().any(|op| matches!(
        op,
        MirStructOp::StructLoadField { struct_name, ordinal: 0, byte_offset: 0, .. } if struct_name == "User"
    )));
    assert!(module.struct_ops.iter().any(|op| matches!(
        op,
        MirStructOp::StructLoadField { struct_name, ordinal: 2, byte_offset: 16, .. } if struct_name == "User"
    )));
    assert_eq!(
        parse_mir_struct_decode_callee(&mir_struct_decode_callee("User")),
        Some("User")
    );

    let bc_fns = MirToBytecodeCompiler::compile_module(&module).expect("compile MIR bytecode");
    for f in &bc_fns {
        let audit = audit_chunk_field_access(&f.chunk);
        assert!(
            audit.has_zero_name_lookup(),
            "Expected zero GetField/SetField in MIR VM chunk, got {:?}",
            audit
        );
        assert!(audit.get_index_count >= 2);
    }
}

// =========================================================================
// 4. Gate-SCHEMA-09 Multi-Backend Parity: VM / Cranelift / Rust AOT / C / Zig
// =========================================================================

#[test]
fn test_w17_gate09_multi_backend_parity_and_native_entry() {
    let _lock = GLOBAL_CRANELIFT_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    safepoint_reset();
    reset_native_instrumentation_counters();

    // Varying records with shuffled keys (prevents constant folding) + independent oracle.
    let cases: [(&str, i64, f64, bool, &str, i64); 4] = [
        (
            r#"{"name":"alpha","active":true,"score":1.5,"id":7}"#,
            7,
            1.5,
            true,
            "alpha",
            107,
        ),
        (
            r#"{"id":42,"score":2.5,"active":false,"name":"beta"}"#,
            42,
            2.5,
            false,
            "beta",
            42,
        ),
        (
            r#"{"score":9.0,"id":-5,"name":"gamma","active":true}"#,
            -5,
            9.0,
            true,
            "gamma",
            95,
        ),
        (
            r#"{"active":false,"id":-100,"name":"delta","score":0.25}"#,
            -100,
            0.25,
            false,
            "delta",
            -100,
        ),
    ];

    let lust_src = r#"
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
    local a: int = eval_user("{\"name\":\"alpha\",\"active\":true,\"score\":1.5,\"id\":7}")
    local b: int = eval_user("{\"id\":42,\"score\":2.5,\"active\":false,\"name\":\"beta\"}")
    local c: int = eval_user("{\"score\":9.0,\"id\":-5,\"name\":\"gamma\",\"active\":true}")
    local d: int = eval_user("{\"active\":false,\"id\":-100,\"name\":\"delta\",\"score\":0.25}")
    println(a)
    println(b)
    println(c)
    println(d)
    return 0
end
"#;

    let mut interner = Interner::new();
    let tokens = Lexer::new(lust_src, &mut interner).tokenize().expect("lex");
    let items = Parser::new(tokens).parse().expect("parse");
    let module = AstLowering::lower_module_items("w17_parity", &items).expect("lower MIR");

    // --- Backend 1: Bytecode VM ---
    let bc_fns = MirToBytecodeCompiler::compile_module(&module).expect("compile VM bytecode");
    let vm_counters = Rc::new(RefCell::new(LayoutPerfCounters::default()));
    let mut vm = VM::new();
    vm.load_functions(bc_fns);
    register_struct_decode_natives_with_counters(
        &mut vm,
        &module.struct_layouts,
        vm_counters.clone(),
    );

    let mut expected_stdout = String::new();
    for (json, _, _, _, _, expected) in cases {
        let vm_res = vm
            .call("eval_user", vec![Value::string(json.to_string())])
            .expect("VM eval_user");
        assert_eq!(vm_res, Value::Int(expected), "VM oracle mismatch for {json}");
        expected_stdout.push_str(&format!("{expected}\n"));
    }
    {
        let c = vm_counters.borrow();
        assert_eq!(c.dom_node_constructions, 0);
        assert_eq!(c.intermediate_map_allocations, 0);
        assert_eq!(c.field_read_name_lookups, 0);
        assert_eq!(c.field_read_hash_lookups, 0);
        assert_eq!(c.field_read_allocations, 0);
        assert_eq!(c.decode_key_dispatches, 16);
    }

    // --- Backend 2: Cranelift JIT (Safe EmbeddedProgram + verified native-entry counter) ---
    let cranelift_src = r#"
function eval_user_fields(id: int, active: int): int
    local s: int = id
    if active != 0 then
        s = s + 100
    end
    return s
end
"#;
    let mut cfg = LustConfig::default().with_cranelift_jit(true);
    cfg.set_use_mir(false);
    let mut program = EmbeddedProgram::builder()
        .with_config(cfg)
        .module("main", cranelift_src)
        .entry_module("main")
        .compile()
        .expect("compile Cranelift EmbeddedProgram");
    let handle = program
        .get_typed_handle::<(i64, i64), i64>("main.eval_user_fields")
        .expect("get typed Cranelift handle");

    let user_layout = build_standard_layouts().5;
    let global_before = CRANELIFT_NATIVE_ENTRY_COUNT.load(std::sync::atomic::Ordering::SeqCst);
    for (json, exp_id, exp_score, exp_active, exp_name, expected) in cases {
        let mut c = LayoutPerfCounters::default();
        let cs = decode_json_into_contiguous(
            &user_layout,
            json.as_bytes(),
            &mut c,
            &ContiguousDecodeOptions::default(),
        )
        .expect("direct contiguous decode for Cranelift");
        assert_eq!(c.dom_node_constructions, 0);
        assert_eq!(c.intermediate_map_allocations, 0);
        let id = cs
            .read_i64_at(0, user_layout.fields[0].byte_offset, Some(&mut c))
            .unwrap()
            .unwrap();
        let score = cs
            .read_f64_at(1, user_layout.fields[1].byte_offset, Some(&mut c))
            .unwrap()
            .unwrap();
        let active = cs
            .read_bool_at(2, user_layout.fields[2].byte_offset, Some(&mut c))
            .unwrap()
            .unwrap();
        let name = cs
            .read_str_at(3, user_layout.fields[3].byte_offset, Some(&mut c))
            .unwrap()
            .unwrap();
        assert_eq!((id, active, name), (exp_id, exp_active, exp_name));
        assert!((score - exp_score).abs() < 1e-9);
        assert_eq!(c.field_read_name_lookups, 0);
        assert_eq!(c.field_read_hash_lookups, 0);
        assert_eq!(c.field_read_allocations, 0);

        let cl_res = program
            .call_handle(&handle, (id, i64::from(active)))
            .expect("Cranelift native call");
        assert_eq!(cl_res, expected, "Cranelift parity mismatch for {json}");
    }
    let global_after = CRANELIFT_NATIVE_ENTRY_COUNT.load(std::sync::atomic::Ordering::SeqCst);
    assert!(
        program.cranelift_native_entry_count() >= 4,
        "Cranelift must record >= 4 native entries, got {}",
        program.cranelift_native_entry_count()
    );
    assert!(
        global_after.saturating_sub(global_before) >= 4,
        "Global CRANELIFT_NATIVE_ENTRY_COUNT must advance by >= 4"
    );
    assert_eq!(
        program.vm().mir_context.dirty_allocations_count(),
        0,
        "Cranelift scalar evaluation must produce 0 dirty allocations"
    );

    // --- Backend 3: Rust AOT (compiled executable + static repr(C) offset_of/size_of/align_of) ---
    let tmp = tempdir().expect("tempdir for Rust AOT");
    let cache_dir = PathBuf::from("/tmp/lust_eval_target/w17-3/aot_rlib_cache");
    fs::create_dir_all(&cache_dir).expect("create aot rlib cache dir");
    let aot_opts = RustCompileOptions {
        aot: AotCompileOptions {
            target_kind: AotTargetKind::Executable,
            release_mode: "ReleaseFast".to_string(),
            ..AotCompileOptions::default()
        },
        cache_dir: Some(cache_dir),
        ..RustCompileOptions::default()
    };
    let emitted =
        emit_rust_from_mir(&module, &aot_opts.to_emitter_options()).expect("emit Rust AOT");
    assert!(!emitted.contains("unsafe"), "emitted Rust AOT must be safe");
    assert!(emitted.contains("core::mem::offset_of!(LustStruct_User, field_0) == 0"));
    assert!(emitted.contains("core::mem::offset_of!(LustStruct_User, field_1) == 8"));
    assert!(emitted.contains("core::mem::offset_of!(LustStruct_User, field_2) == 16"));
    assert!(emitted.contains("core::mem::offset_of!(LustStruct_User, field_3) == 24"));
    assert!(emitted.contains("core::mem::size_of::<LustStruct_User>() == 48"));
    assert!(emitted.contains("core::mem::align_of::<LustStruct_User>() == 8"));

    let aot_bin = tmp.path().join("w17_user_aot_bin");
    AotCompiler::compile_module_rust(&module, &aot_bin, &aot_opts).expect("compile Rust AOT bin");
    let aot_run = Command::new(&aot_bin).output().expect("execute Rust AOT bin");
    assert!(
        aot_run.status.success(),
        "Rust AOT bin failed: stderr={}",
        String::from_utf8_lossy(&aot_run.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&aot_run.stdout),
        expected_stdout,
        "Rust AOT stdout must match VM & Cranelift oracle exactly"
    );

    // --- Backend 4: Host C11 fixture verifying raw ContiguousStruct bytes ---
    let c_build = built_c_layout_probe();
    for (idx, (json, _, _, _, _, expected)) in cases.iter().enumerate() {
        let mut c = LayoutPerfCounters::default();
        let cs = decode_json_into_contiguous(
            &user_layout,
            json.as_bytes(),
            &mut c,
            &ContiguousDecodeOptions::default(),
        )
        .expect("decode User for C parity");
        let bin_path = tmp.path().join(format!("user_{idx}.bin"));
        fs::write(&bin_path, cs.raw_bytes()).expect("write user raw bytes");
        let bin_arg = bin_path.display().to_string();
        let run = run_child_with_watchdog(
            &c_build.output_binary,
            &["--verify-user-bytes", &bin_arg],
            &[],
            Duration::from_secs(10),
        )
        .expect("run C user parity");
        assert!(run.outcome.is_clean_success());
        let parsed: serde_json::Value =
            serde_json::from_str(run.stdout.trim()).expect("parse C user json");
        assert_eq!(
            parsed["eval"].as_i64(),
            Some(*expected),
            "C11 parity mismatch for {json}"
        );
    }

    // --- Backend 5: Host Zig extern struct comptime layout & eval probe ---
    let zig_src = fixture_dir().join("zig_layout_probe.zig");
    let zig_ev = verify_zig_layout_probe(&zig_src, Duration::from_secs(15)).expect("zig probe");
    eprintln!(
        "[W17-3 ZIG] available={} version={} passed={} cmd={}",
        zig_ev.available, zig_ev.version, zig_ev.passed, zig_ev.command_line
    );
    if zig_ev.available {
        assert!(
            zig_ev.passed,
            "Zig layout probe failed (version={}): stderr={}",
            zig_ev.version, zig_ev.stderr
        );
    }
    // Note: Dynasm backend (`J-DYNASM-RET`) is explicitly excluded in Wave 17 per plan.
}

// =========================================================================
// 5. Gate-SCHEMA-09 Zero-Allocation Prepared Scalar Reads & N/2N/4N Scaling
// =========================================================================

#[test]
fn test_w17_gate09_zero_alloc_scalar_and_n_2n_4n_scaling() {
    // 1. Prepared fixed-scalar ContiguousStruct reads over N=10,000 iterations:
    //    0 heap allocations (verified via VmAllocHandle budget delta & LayoutPerfCounters).
    let scalar_layout = LogicalLayoutV1::new_with_abi(
        9100,
        "ScalarMetrics",
        vec![
            LogicalFieldDesc::new(0, "a", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(1, "b", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(2, "c", CanonicalTypeTag::Float64),
            LogicalFieldDesc::new(3, "ok", CanonicalTypeTag::Bool),
        ],
        TargetAbi::host(),
    );
    assert!(scalar_layout.is_c_flat);

    // Verify AlignedStorage::new_zeroed_in budget charge/refund accounting.
    let alloc_handle = VmAllocHandle::with_limit(4096);
    {
        let storage =
            AlignedStorage::new_zeroed_in(scalar_layout.size_bytes, scalar_layout.align_bytes, alloc_handle.clone())
                .expect("budgeted aligned storage");
        assert_eq!(alloc_handle.0.used_bytes(), scalar_layout.size_bytes);
        assert_eq!((storage.as_ptr() as usize) % scalar_layout.align_bytes, 0);
    }
    assert_eq!(
        alloc_handle.0.used_bytes(),
        0,
        "AlignedStorage drop must refund 100% of charged bytes"
    );

    let mut counters = LayoutPerfCounters::default();
    let cs = decode_json_into_contiguous(
        &scalar_layout,
        br#"{"ok":true,"c":3.5,"b":19,"a":23}"#,
        &mut counters,
        &ContiguousDecodeOptions::default(),
    )
    .expect("decode scalar struct");
    assert_eq!(counters.dom_node_constructions, 0);
    assert_eq!(counters.intermediate_map_allocations, 0);
    assert_eq!(counters.owned_slice_allocations, 0);

    let used_before = alloc_handle.0.used_bytes();
    let peak_before = alloc_handle.0.peak_bytes();
    let mut checksum = 0i64;
    for iter in 0..10_000i64 {
        // Arbitrary read order on every iteration.
        let b = cs
            .read_i64_at(1, scalar_layout.fields[1].byte_offset, Some(&mut counters))
            .unwrap()
            .unwrap();
        let ok = cs
            .read_bool_at(3, scalar_layout.fields[3].byte_offset, Some(&mut counters))
            .unwrap()
            .unwrap();
        let a = cs
            .read_i64_at(0, scalar_layout.fields[0].byte_offset, Some(&mut counters))
            .unwrap()
            .unwrap();
        let c = cs
            .read_f64_at(2, scalar_layout.fields[2].byte_offset, Some(&mut counters))
            .unwrap()
            .unwrap();
        checksum += (a + b) + if ok { c as i64 } else { 0 } + (iter & 1);
    }
    assert_eq!(checksum, 10_000 * (23 + 19 + 3) + 5_000);
    assert_eq!(alloc_handle.0.used_bytes() - used_before, 0);
    assert_eq!(alloc_handle.0.peak_bytes() - peak_before, 0);
    assert_eq!(counters.field_read_name_lookups, 0);
    assert_eq!(counters.field_read_hash_lookups, 0);
    assert_eq!(counters.field_read_allocations, 0);
    assert_eq!(counters.dom_node_constructions, 0);
    assert_eq!(counters.intermediate_map_allocations, 0);

    // 2. Variable-length string ownership measured separately (exact alloc == free on cleanup).
    let var_layout = build_standard_layouts().1; // Account with StringSlice `owner`
    let mut var_counters = LayoutPerfCounters::default();
    let mut acc = decode_json_into_contiguous(
        &var_layout,
        br#"{"owner":"carol","rate":1.75,"balance":900,"id":12}"#,
        &mut var_counters,
        &ContiguousDecodeOptions::default(),
    )
    .expect("decode Account");
    assert_eq!(var_counters.owned_slice_allocations, 1);
    assert_eq!(var_counters.owned_slice_frees, 0);
    let cleaned = acc.cleanup_partial(Some(&mut var_counters));
    assert_eq!(cleaned, 4);
    assert_eq!(var_counters.owned_slice_frees, 1);
    assert_eq!(
        var_counters.owned_slice_allocations,
        var_counters.owned_slice_frees
    );

    // 3. N / 2N / 4N scaling with fixed read count and varying checksums.
    for n in [4usize, 8, 16, 32] {
        let fields: Vec<_> = (0..n)
            .map(|i| LogicalFieldDesc::new(i as u16, format!("f{i}"), CanonicalTypeTag::Int64))
            .collect();
        let layout =
            LogicalLayoutV1::new_with_abi(9200 + n as u64, format!("Scale{n}"), fields, TargetAbi::host());
        let mut json = String::from("{");
        let mut expected = 0i64;
        for i in (0..n).rev() {
            let v = (i as i64 + 1) * 13 - 7;
            if i < n - 1 {
                json.push(',');
            }
            json.push_str(&format!("\"f{i}\":{v}"));
            if i == 0 || i == n / 2 || i == n - 1 {
                expected += v;
            }
        }
        json.push('}');
        let mut c = LayoutPerfCounters::default();
        let s = decode_json_into_contiguous(&layout, json.as_bytes(), &mut c, &ContiguousDecodeOptions::default())
            .expect("decode scale struct");
        let read_indices = [n - 1, 0, n / 2]; // arbitrary read order, fixed count = 3
        let actual: i64 = read_indices
            .iter()
            .map(|&idx| {
                s.read_i64_at(idx as u16, layout.fields[idx].byte_offset, Some(&mut c))
                    .unwrap()
                    .unwrap()
            })
            .sum();
        assert_eq!(actual, expected);
        assert_eq!(c.field_read_name_lookups, 0);
        assert_eq!(c.field_read_hash_lookups, 0);
        assert_eq!(c.field_read_allocations, 0);
    }
}

// =========================================================================
// 6. Gate-SCHEMA-09 Decoding, Wire Aliases, Mutate-Then-Encode & DB Model
// =========================================================================

#[test]
fn test_w17_gate09_decoding_aliases_mutate_encode_and_db_model() {
    let user_layout = LogicalLayoutV1::new_with_abi(
        9300,
        "ModelUser",
        vec![
            LogicalFieldDesc::new(0, "id", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(1, "username", CanonicalTypeTag::StringSlice),
            LogicalFieldDesc::new(2, "age", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(3, "bio", CanonicalTypeTag::StringSlice)
                .with_nullability(true, true),
        ],
        TargetAbi::host(),
    );

    let mut aliases = WireAliasMap::new();
    aliases.add_alias("id", "_id").expect("alias _id");
    aliases
        .add_alias("username", "user_name")
        .expect("alias user_name");

    // Verify all 3 binding origins (Derived, Overridden, ImportBound) construct valid plans.
    let plan_derived =
        StructBindingPlan::new(BindingOrigin::Derived, user_layout.clone(), 0xAA01, WireAliasMap::new())
            .expect("derived plan");
    let plan_overridden =
        StructBindingPlan::new(BindingOrigin::Overridden, user_layout.clone(), 0xAA02, aliases.clone())
            .expect("overridden plan")
            .require_non_negative_int(2);
    let plan_import =
        StructBindingPlan::new(BindingOrigin::ImportBound, user_layout.clone(), 0xAA03, aliases)
            .expect("import-bound plan")
            .require_non_negative_int(2);
    assert_ne!(plan_derived.layout_fingerprint, plan_overridden.layout_fingerprint);
    assert_ne!(plan_overridden.layout_fingerprint, plan_import.layout_fingerprint);

    // 1. Shuffled wire fields with aliases decode directly into ContiguousStruct (0 DOM nodes).
    let mut c = LayoutPerfCounters::default();
    let aliased_json = br#"{"_type":"ModelUser","bio":"hi","age":29,"user_name":"vedanta","_id":101}"#;
    let cs = decode_json_with_plan(
        &plan_overridden,
        aliased_json,
        &mut c,
        &ContiguousDecodeOptions {
            require_nominal_tag: true,
            ..Default::default()
        },
    )
    .expect("aliased decode succeeds");
    plan_overridden
        .validate_record_constraints(&cs)
        .expect("record constraints pass");
    assert_eq!(c.dom_node_constructions, 0);
    assert_eq!(c.intermediate_map_allocations, 0);
    assert_eq!(cs.read_i64_at(0, user_layout.fields[0].byte_offset, None).unwrap(), Some(101));
    assert_eq!(
        cs.read_str_at(1, user_layout.fields[1].byte_offset, None).unwrap(),
        Some("vedanta")
    );

    // 2. Duplicate alias (both `username` and `user_name`) MUST fail closed as DuplicateField.
    let dup_alias_json = br#"{"_id":101,"username":"a","user_name":"b","age":20}"#;
    let dup_err = decode_json_with_plan(
        &plan_overridden,
        dup_alias_json,
        &mut c,
        &ContiguousDecodeOptions::default(),
    )
    .unwrap_err();
    assert!(
        matches!(dup_err, MaterializeError::DuplicateField(ref k) if k == "username"),
        "expected DuplicateField(username), got {dup_err:?}"
    );

    // 3. Mutate-then-encode: valid mutation passes revalidation & round-trips JSON + BSON;
    //    invalid mutation (`age = -1`) rejected before encode.
    let mut doc = cs.to_materialized().to_document();
    // Valid mutation: update age to 35
    let mut valid_mut = Document::new();
    for (k, v) in doc.entries() {
        let new_v = if k == "age" {
            DataValue::Number(ExactNumber::Int(35))
        } else {
            v.clone()
        };
        valid_mut.try_insert(k.clone(), new_v).unwrap();
    }
    let reencoded_json = plan_overridden
        .encode_validated_json(&valid_mut)
        .expect("valid mutated encode json");
    assert!(reencoded_json.contains("\"_id\":101"));
    assert!(reencoded_json.contains("\"user_name\":\"vedanta\""));
    assert!(reencoded_json.contains("\"age\":35"));
    let reencoded_bson = plan_overridden
        .encode_validated_bson(&valid_mut)
        .expect("valid mutated encode bson");
    assert!(!reencoded_bson.is_empty());

    // Round-trip the re-encoded JSON back through direct decoder!
    let mut rt_c = LayoutPerfCounters::default();
    let rt_cs = decode_json_with_plan(
        &plan_overridden,
        reencoded_json.as_bytes(),
        &mut rt_c,
        &ContiguousDecodeOptions::default(),
    )
    .expect("round-trip decode");
    assert_eq!(rt_cs.read_i64_at(2, user_layout.fields[2].byte_offset, None).unwrap(), Some(35));

    // Invalid mutation (`age = -1`) rejected on revalidation before encode.
    let mut invalid_mut = Document::new();
    for (k, v) in doc.entries() {
        let new_v = if k == "age" {
            DataValue::Number(ExactNumber::Int(-1))
        } else {
            v.clone()
        };
        invalid_mut.try_insert(k.clone(), new_v).unwrap();
    }
    let enc_err = plan_overridden
        .encode_validated_json(&invalid_mut)
        .unwrap_err();
    assert!(matches!(
        enc_err,
        ModelContractError::ConstraintViolation { ref field, .. } if field == "age"
    ));
    let _ = &mut doc;

    // 4. ModelContract + DbWritePlan + RecordingDbAdapter (generated ID, Patch NULL,
    //    stale DbWritePlan, lifecycle hooks in FIFO order, zero I/O on pre-write failure).
    let mut generated = HashMap::new();
    generated.insert("id".to_string(), GeneratedIdPolicy::DatabaseIdentity);
    let mut columns = HashMap::new();
    columns.insert("username".to_string(), "user_name".to_string());

    let mut contract = ModelContract::new(
        "users",
        vec!["id".to_string()],
        generated.clone(),
        vec![],
        columns.clone(),
        plan_overridden.clone(),
    );
    contract.hooks.validate_record.push(Box::new(|d| {
        if let Some(DataValue::Number(ExactNumber::Int(age))) = d.get("age") {
            if *age < 18 {
                return Err("underage user rejected by validate_record".to_string());
            }
        }
        Ok(())
    }));
    contract.hooks.before_save.push(Box::new(|d| {
        if let Some(DataValue::String(name)) = d.get("username") {
            if name == "cancel_me" {
                return Ok(false);
            }
            if name == "fail_hook" {
                return Err("before_save hook rejected".to_string());
            }
        }
        Ok(true)
    }));
    contract.hooks.after_load.push(Box::new(|_d| Ok(())));

    let insert_plan = contract.compile_insert_plan();
    // Generated `id` must be excluded from INSERT columns/placeholders.
    assert_eq!(insert_plan.ordered_fields, vec!["username", "age", "bio"]);
    assert_eq!(insert_plan.ordered_columns, vec!["user_name", "age", "bio"]);
    assert_eq!(insert_plan.placeholders, vec!["$1", "$2", "$3"]);

    let mut adapter = RecordingDbAdapter::new(5001);
    let mut trace = Vec::new();
    let ins_ok = InsertInput::new()
        .with_field("username", DataValue::String("arjuna".to_string()))
        .with_field("age", DataValue::Number(ExactNumber::Int(28)))
        .with_field("bio", DataValue::Null);
    let persisted = adapter
        .execute_insert(&contract, &ins_ok, &insert_plan, &mut trace, false)
        .expect("valid insert succeeds");
    assert_eq!(adapter.io_calls, 1);
    assert_eq!(
        persisted.get("id"),
        Some(&DataValue::Number(ExactNumber::Int(5001)))
    );
    assert_eq!(
        trace,
        vec![
            "validate_record[0]",
            "before_save[0]",
            "revalidate_record[0]",
            "after_load[0]",
            "post_load_validate[0]",
        ]
    );
    assert_eq!(adapter.writes[0].operation, DbOperation::Insert);
    assert_eq!(
        adapter.writes[0].params,
        vec![
            ("user_name".to_string(), DataValue::String("arjuna".to_string())),
            ("age".to_string(), DataValue::Number(ExactNumber::Int(28))),
            ("bio".to_string(), DataValue::Null),
        ]
    );

    // Hook failure -> zero additional DB I/O.
    let io_before = adapter.io_calls;
    let ins_fail_hook = InsertInput::new()
        .with_field("username", DataValue::String("fail_hook".to_string()))
        .with_field("age", DataValue::Number(ExactNumber::Int(25)));
    let err_hook = adapter
        .execute_insert(&contract, &ins_fail_hook, &insert_plan, &mut Vec::new(), false)
        .unwrap_err();
    assert!(matches!(err_hook, ModelContractError::HookFailed(_)));
    assert_eq!(adapter.io_calls, io_before, "hook failure must perform 0 DB I/O");

    // Hook cancel -> zero additional DB I/O.
    let ins_cancel = InsertInput::new()
        .with_field("username", DataValue::String("cancel_me".to_string()))
        .with_field("age", DataValue::Number(ExactNumber::Int(25)));
    let err_cancel = adapter
        .execute_insert(&contract, &ins_cancel, &insert_plan, &mut Vec::new(), false)
        .unwrap_err();
    assert!(matches!(err_cancel, ModelContractError::HookCancelled(_)));
    assert_eq!(adapter.io_calls, io_before, "hook cancel must perform 0 DB I/O");

    // Sparse PatchInput: Missing omits field, Null on nullable `bio` binds SQL NULL,
    // Null on non-nullable `age` rejected pre-IO!
    let patch_ok = PatchInput::new(DataValue::Number(ExactNumber::Int(5001)))
        .with_patch("username", PatchField::Missing)
        .with_patch("age", PatchField::Value(DataValue::Number(ExactNumber::Int(30))))
        .with_patch("bio", PatchField::Null);
    let patch_plan = contract.compile_patch_plan(&patch_ok).expect("compile patch plan");
    assert_eq!(patch_plan.ordered_columns, vec!["age", "bio", "id"]);
    adapter
        .execute_patch(&contract, &patch_ok, &patch_plan)
        .expect("execute valid sparse patch");
    assert_eq!(adapter.io_calls, io_before + 1);
    assert_eq!(
        adapter.writes.last().unwrap().params,
        vec![
            ("age".to_string(), DataValue::Number(ExactNumber::Int(30))),
            ("bio".to_string(), DataValue::Null),
            ("id".to_string(), DataValue::Number(ExactNumber::Int(5001))),
        ]
    );

    let patch_bad_null = PatchInput::new(DataValue::Number(ExactNumber::Int(5001)))
        .with_patch("age", PatchField::Null);
    assert!(matches!(
        contract.compile_patch_plan(&patch_bad_null),
        Err(ModelContractError::NullOnNonNullableColumn(ref f)) if f == "age"
    ));

    // Stale DbWritePlan (compiled for a different/older model fingerprint) rejected pre-IO!
    let newer_contract = ModelContract::new(
        "users_v2",
        vec!["id".to_string()],
        generated,
        vec![],
        columns,
        plan_import,
    );
    let io_before_stale = adapter.io_calls;
    let stale_err = adapter
        .execute_insert(&newer_contract, &ins_ok, &insert_plan, &mut Vec::new(), false)
        .unwrap_err();
    assert!(matches!(stale_err, ModelContractError::StaleWritePlan { .. }));
    assert_eq!(
        adapter.io_calls, io_before_stale,
        "stale DbWritePlan must perform 0 DB I/O"
    );
}

// =========================================================================
// 7. Gate-SCHEMA-09 All Four Required Mutants (MUST FAIL)
// =========================================================================

#[test]
fn test_w17_gate09_four_mutants_must_fail() {
    // Mutant 1: Name-lookup mutant (GetField emitted instead of GetIndex).
    let mut chunk = Chunk::new();
    let k = chunk.add_constant(Value::string("id".to_string()));
    chunk.emit(Instruction::GetField(1, 0, k), 1);
    let audit = audit_chunk_field_access(&chunk);
    assert!(
        !audit.has_zero_name_lookup(),
        "Gate-09 Mutant 1 (name-lookup) MUST fail zero-lookup audit"
    );

    // Mutant 2: DOM fallback mutant (constructs intermediate Document/Map).
    let mut dom_c = LayoutPerfCounters::default();
    let mut doc = Document::new();
    doc.try_insert("id", DataValue::Number(ExactNumber::Int(1)))
        .unwrap();
    dom_c.dom_node_constructions += 1;
    dom_c.intermediate_map_allocations += 1;
    assert!(
        dom_c.dom_node_constructions > 0 && dom_c.intermediate_map_allocations > 0,
        "Gate-09 Mutant 2 (DOM fallback) MUST fail zero-DOM invariant"
    );

    // Mutant 3: Omitted cleanup mutant on partial-init failure.
    let layout = LogicalLayoutV1::new_with_abi(
        9401,
        "CleanupMutant",
        vec![
            LogicalFieldDesc::new(0, "s", CanonicalTypeTag::StringSlice),
            LogicalFieldDesc::new(1, "x", CanonicalTypeTag::Int64),
        ],
        TargetAbi::host(),
    );
    let mut cs = ContiguousStruct::new_uninit(layout);
    let mut leak_c = LayoutPerfCounters::default();
    leak_c.owned_slice_allocations += 1;
    // Deliberately omit `cleanup_partial`: leak invariant MUST fail!
    assert_ne!(
        leak_c.owned_slice_allocations, leak_c.owned_slice_frees,
        "Gate-09 Mutant 3 (omitted cleanup) MUST fail alloc==free invariant"
    );
    let _ = cs.cleanup_partial(Some(&mut leak_c));

    // Mutant 4: Swapped DB write-plan params mutant MUST fail independent oracle!
    let swap_layout = LogicalLayoutV1::new_with_abi(
        9402,
        "SwapRec",
        vec![
            LogicalFieldDesc::new(0, "id", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(1, "a", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(2, "b", CanonicalTypeTag::Int64),
        ],
        TargetAbi::host(),
    );
    let plan = StructBindingPlan::new(
        BindingOrigin::Derived,
        swap_layout,
        0xBB01,
        WireAliasMap::new(),
    )
    .unwrap();
    let mut generated_map = HashMap::new();
    generated_map.insert("id".to_string(), GeneratedIdPolicy::DatabaseIdentity);
    let contract = ModelContract::new(
        "swaps",
        vec!["id".to_string()],
        generated_map,
        vec![],
        HashMap::new(),
        plan,
    );
    let write_plan = contract.compile_insert_plan();
    let input = InsertInput::new()
        .with_field("a", DataValue::Number(ExactNumber::Int(10)))
        .with_field("b", DataValue::Number(ExactNumber::Int(20)));

    let good_params = contract
        .prepare_insert_params(&input, &write_plan, &mut Vec::new(), false)
        .unwrap();
    let swapped_params = contract
        .prepare_insert_params(&input, &write_plan, &mut Vec::new(), true)
        .unwrap();
    let oracle_params = vec![
        ("a".to_string(), DataValue::Number(ExactNumber::Int(10))),
        ("b".to_string(), DataValue::Number(ExactNumber::Int(20))),
    ];
    assert_eq!(good_params, oracle_params);
    assert_ne!(
        swapped_params, oracle_params,
        "Gate-09 Mutant 4 (swapped params) MUST fail independent oracle"
    );
}

// =========================================================================
// 8. Gate-SCHEMA-13 Manifest Validation & All Status/Errno/Null/Cancel Cases
// =========================================================================

#[test]
fn test_w17_gate13_ffi_semantic_boundary_statuses_and_sanitizers() {
    // 1. Fail-closed manifest validator checks (RFC Section 16.4).
    let valid_manifest = FfiManifest::store_read_v1();
    valid_manifest.validate().expect("canonical manifest valid");

    let mut dup_status = valid_manifest.clone();
    dup_status.rules.push(FfiStatusRule {
        raw_status: 10,
        kind: FfiStatusKind::RecoverableNotFound,
        uses_errno: false,
    });
    assert_eq!(
        dup_status.validate(),
        Err(FfiManifestError::OverlappingStatusCode(10))
    );

    let mut bad_owner = valid_manifest.clone();
    bad_owner.output_ownership = FfiOwnership::IncompleteUnspecified;
    assert_eq!(
        bad_owner.validate(),
        Err(FfiManifestError::IncompleteOwnership("output"))
    );

    let mut bad_errno = valid_manifest.clone();
    bad_errno.rules[0].uses_errno = true;
    assert_eq!(
        bad_errno.validate(),
        Err(FfiManifestError::ErrnoDisallowedForStatus(0))
    );

    let mut bad_varargs = valid_manifest.clone();
    bad_varargs.allows_varargs = true;
    assert_eq!(
        bad_varargs.validate(),
        Err(FfiManifestError::VarargsRejected)
    );

    let mut bad_cb = valid_manifest.clone();
    bad_cb.execution = FfiExecutionMode::UnsupportedDirectCallback;
    assert_eq!(
        bad_cb.validate(),
        Err(FfiManifestError::UnregisteredCallbackRejected)
    );

    let mut bad_iso = valid_manifest.clone();
    bad_iso.execution = FfiExecutionMode::NonblockingTrustedInProcess;
    bad_iso.untrusted_or_blocking = true;
    assert_eq!(
        bad_iso.validate(),
        Err(FfiManifestError::MissingIsolationForUntrustedWorker)
    );

    // 2. Compile C FFI fixture with ASan + UBSan and exercise every Gate-13 case in isolated children.
    let build = built_c_ffi_fixture();
    assert!(
        build.sanitizers_enabled,
        "Expected host cc to enable ASan+UBSan for c_ffi_boundary_fixture"
    );
    let mut rt = FfiBoundaryRuntime::new(1, valid_manifest, build.output_binary.clone())
        .expect("init FfiBoundaryRuntime");
    let owner = rt.allocate_owner(100);
    let desc = FfiAbiDescriptor::valid(100, owner);
    let caps = permitted_store_caps();
    let timeout = Duration::from_secs(10);

    // Case A: Success positive payload
    let ev_pos = rt
        .invoke_isolated("success_positive", false, &caps, &desc, 100, timeout, FfiMutationMode::None)
        .unwrap();
    assert!(!ev_pos.watchdog_killed);
    assert_eq!(ev_pos.exit, Exit::Success(42001));
    assert_eq!(ev_pos.owner_leaks(), 0);
    assert!(ev_pos.constructor_runs_observed >= 1);

    // Case B: Success negative payloads (-42 and i64::MIN — never confused with error sentinel!)
    let ev_neg = rt
        .invoke_isolated("success_negative", false, &caps, &desc, 100, timeout, FfiMutationMode::None)
        .unwrap();
    assert_eq!(ev_neg.exit, Exit::Success(-42));
    assert_eq!(ev_neg.owner_leaks(), 0);

    let ev_min = rt
        .invoke_isolated("success_i64_min", false, &caps, &desc, 100, timeout, FfiMutationMode::None)
        .unwrap();
    assert_eq!(ev_min.exit, Exit::Success(i64::MIN));
    assert_eq!(ev_min.owner_leaks(), 0);

    // Case C: Declared status STORE_MISSING -> Cause::Fail(NotFound)
    let ev_miss = rt
        .invoke_isolated("status_missing", false, &caps, &desc, 100, timeout, FfiMutationMode::None)
        .unwrap();
    assert_eq!(ev_miss.exit, Exit::Failure(Cause::Fail(NativeCallError::NotFound)));
    assert_eq!(ev_miss.captured_errno, 0);

    // Case D: Declared status STORE_ERROR with immediate errno capture (EACCES=13 on macOS, not ENOENT=2!)
    let ev_io = rt
        .invoke_isolated("status_io_errno", false, &caps, &desc, 100, timeout, FfiMutationMode::None)
        .unwrap();
    assert!(
        matches!(ev_io.exit, Exit::Failure(Cause::Fail(NativeCallError::NativeIo { errno })) if errno != 0 && errno != 2),
        "expected immediate EACCES errno capture (not post-cleanup ENOENT=2), got {:?}",
        ev_io.exit
    );

    // Case E: Stale errno on success & on non-errno status MUST be ignored (captured_errno == 0)
    let ev_stale_ok = rt
        .invoke_isolated("stale_errno_on_success", false, &caps, &desc, 100, timeout, FfiMutationMode::None)
        .unwrap();
    assert_eq!(ev_stale_ok.exit, Exit::Success(77));
    assert_eq!(ev_stale_ok.captured_errno, 0);

    let ev_stale_miss = rt
        .invoke_isolated("stale_errno_on_missing", false, &caps, &desc, 100, timeout, FfiMutationMode::None)
        .unwrap();
    assert_eq!(ev_stale_miss.exit, Exit::Failure(Cause::Fail(NativeCallError::NotFound)));
    assert_eq!(ev_stale_miss.captured_errno, 0);

    // Case F: Native EINTR without guest cancel token -> Cause::Fail (NOT Cause::Interrupted!)
    let ev_eintr = rt
        .invoke_isolated("status_eintr_uncancelled", false, &caps, &desc, 100, timeout, FfiMutationMode::None)
        .unwrap();
    assert!(
        !ev_eintr.exit.is_interrupt(),
        "uncancelled EINTR must NOT become Cause::Interrupted"
    );
    assert!(matches!(
        ev_eintr.exit,
        Exit::Failure(Cause::Fail(NativeCallError::InterruptedSysCall { errno })) if errno != 0
    ));

    // Case G: Declared cancel with token -> terminal Cause::Interrupted(Cancelled)
    let ev_cancel_ok = rt
        .invoke_isolated("status_cancelled_with_token", true, &caps, &desc, 100, timeout, FfiMutationMode::None)
        .unwrap();
    assert_eq!(
        ev_cancel_ok.exit,
        Exit::Failure(Cause::Interrupted(InterruptReason::Cancelled))
    );

    // Case H: Declared cancel WITHOUT guest token -> defect Cause::Die
    let ev_cancel_bad = rt
        .invoke_isolated("status_cancelled_without_token", false, &caps, &desc, 100, timeout, FfiMutationMode::None)
        .unwrap();
    assert!(matches!(ev_cancel_bad.exit, Exit::Failure(Cause::Die(_))));

    // Case I: Invalid-success / NULL output -> defect Cause::Die (NEVER Exit::Success!)
    let ev_null_ok = rt
        .invoke_isolated("invalid_success_null", false, &caps, &desc, 100, timeout, FfiMutationMode::None)
        .unwrap();
    assert!(matches!(ev_null_ok.exit, Exit::Failure(Cause::Die(_))));

    // Case J: Unknown status (777) -> defect Cause::Die
    let ev_unk = rt
        .invoke_isolated("unknown_status", false, &caps, &desc, 100, timeout, FfiMutationMode::None)
        .unwrap();
    assert!(matches!(ev_unk.exit, Exit::Failure(Cause::Die(ref m)) if m.contains("777")));

    // Case K: Partial-init cleanup in C (allocs == frees == 2, reverse order checksum == 63, ASan clean)
    let ev_partial = rt
        .invoke_isolated("partial_init_cleanup", false, &caps, &desc, 100, timeout, FfiMutationMode::None)
        .unwrap();
    assert_eq!(
        ev_partial.exit,
        Exit::Failure(Cause::Fail(NativeCallError::PartialInitFailed {
            cleaned_owners: 2
        }))
    );
    assert_eq!(ev_partial.owner_allocs, 2);
    assert_eq!(ev_partial.owner_frees, 2);
    assert_eq!(ev_partial.owner_leaks(), 0);
    assert_eq!(ev_partial.cleanup_order_checksum, 63);

    // Case L: Cooperative cancel race with reverse-order partial cleanup
    let ev_race = rt
        .invoke_isolated("cancel_race", true, &caps, &desc, 100, timeout, FfiMutationMode::None)
        .unwrap();
    assert!(!ev_race.watchdog_killed);
    assert_eq!(
        ev_race.exit,
        Exit::Failure(Cause::Interrupted(InterruptReason::Cancelled))
    );
    assert_eq!(ev_race.owner_allocs, 2);
    assert_eq!(ev_race.owner_frees, 2);
    assert_eq!(ev_race.owner_leaks(), 0);
}

// =========================================================================
// 9. Gate-SCHEMA-13 Capabilities, Stale Tenant/ABI Handles & Worker Watchdog
// =========================================================================

#[test]
fn test_w17_gate13_capabilities_handles_and_isolated_workers() {
    // Explicit RFC Section 16.4 honesty check: in-process FFI is NEVER claimed sandboxed.
    assert!(!FfiBoundaryRuntime::in_process_ffi_is_sandboxed());

    let build = built_c_ffi_fixture();
    let mut rt = FfiBoundaryRuntime::new(7, FfiManifest::store_read_v1(), build.output_binary.clone())
        .expect("init runtime");
    let owner_t1 = rt.allocate_owner(1);
    let desc_t1 = FfiAbiDescriptor::valid(1, owner_t1);
    let timeout = Duration::from_secs(10);

    // 1. Denied capability (CapabilityMask::none()) -> constructor counter == 0, native entry == 0!
    let denied_none = CapabilityMask::none();
    let ev_denied_1 = rt
        .invoke_isolated("success_positive", false, &denied_none, &desc_t1, 1, timeout, FfiMutationMode::None)
        .unwrap();
    assert!(matches!(
        ev_denied_1.exit,
        Exit::Failure(Cause::Fail(NativeCallError::CapabilityDenied(_)))
    ));
    assert_eq!(ev_denied_1.constructor_runs_observed, 0);
    assert_eq!(ev_denied_1.native_entries_observed, 0);

    // Denied capability (`allow_ffi=true` but missing `"Store.read"`) -> constructor == 0!
    let mut denied_missing_scope = CapabilityMask::none();
    denied_missing_scope.allow_ffi = true;
    let ev_denied_2 = rt
        .invoke_isolated("success_positive", false, &denied_missing_scope, &desc_t1, 1, timeout, FfiMutationMode::None)
        .unwrap();
    assert!(matches!(
        ev_denied_2.exit,
        Exit::Failure(Cause::Fail(NativeCallError::CapabilityDenied(_)))
    ));
    assert_eq!(ev_denied_2.constructor_runs_observed, 0);
    assert_eq!(ev_denied_2.native_entries_observed, 0);

    // Permitted control -> constructor counter > 0 AND native entry > 0!
    let permitted = permitted_store_caps();
    let ev_perm = rt
        .invoke_isolated("success_positive", false, &permitted, &desc_t1, 1, timeout, FfiMutationMode::None)
        .unwrap();
    assert_eq!(ev_perm.exit, Exit::Success(42001));
    assert!(ev_perm.constructor_runs_observed > 0);
    assert!(ev_perm.native_entries_observed > 0);

    // 2. Revoked / stale / cross-tenant / foreign-runtime / wrong-ABI / short-size rejection!
    // Cross-tenant access (tenant 2 using tenant 1's handle):
    let ev_xtenant = rt
        .invoke_isolated("success_positive", false, &permitted, &desc_t1, 2, timeout, FfiMutationMode::None)
        .unwrap();
    assert!(matches!(
        ev_xtenant.exit,
        Exit::Failure(Cause::Fail(NativeCallError::InvalidOwnerLease(
            LeaseError::TenantMismatch { expected: 2, actual: 1 }
        )))
    ));
    assert_eq!(ev_xtenant.constructor_runs_observed, 0);

    // Wrong ABI version (999):
    let mut bad_abi = desc_t1;
    bad_abi.abi_version = 999;
    let ev_bad_abi = rt
        .invoke_isolated("success_positive", false, &permitted, &bad_abi, 1, timeout, FfiMutationMode::None)
        .unwrap();
    assert!(matches!(
        ev_bad_abi.exit,
        Exit::Failure(Cause::Fail(NativeCallError::InvalidAbiVersion { expected: 1, actual: 999 }))
    ));
    assert_eq!(ev_bad_abi.constructor_runs_observed, 0);

    // Short descriptor size (16 < 64):
    let mut short_desc = desc_t1;
    short_desc.struct_size = 16;
    let ev_short = rt
        .invoke_isolated("success_positive", false, &permitted, &short_desc, 1, timeout, FfiMutationMode::None)
        .unwrap();
    assert!(matches!(
        ev_short.exit,
        Exit::Failure(Cause::Fail(NativeCallError::ShortDescriptorSize { required: 64, actual: 16 }))
    ));
    assert_eq!(ev_short.constructor_runs_observed, 0);

    // Release-before-join protocol violation rejected by OwnerRegistry:
    assert_eq!(rt.owners.release(owner_t1), Err(LeaseError::ReleaseBeforeJoin));

    // Revoke owner (cancel -> acknowledge_join -> release) and reallocate slot -> generation bumps!
    rt.revoke_owner(owner_t1).expect("revoke owner_t1");
    // Double release rejected:
    assert!(rt.owners.release(owner_t1).is_err());

    let owner_t1_new = rt.allocate_owner(1);
    assert_eq!(owner_t1_new.slot_id, owner_t1.slot_id);
    assert_ne!(owner_t1_new.generation, owner_t1.generation);

    // Old stale handle rejected with StaleGeneration before spawning native worker!
    let ev_stale = rt
        .invoke_isolated("success_positive", false, &permitted, &desc_t1, 1, timeout, FfiMutationMode::None)
        .unwrap();
    assert!(matches!(
        ev_stale.exit,
        Exit::Failure(Cause::Fail(NativeCallError::InvalidOwnerLease(
            LeaseError::StaleGeneration { .. }
        )))
    ));
    assert_eq!(ev_stale.constructor_runs_observed, 0);

    // New generation handle succeeds!
    let desc_new = FfiAbiDescriptor::valid(1, owner_t1_new);
    let ev_new_ok = rt
        .invoke_isolated("success_positive", false, &permitted, &desc_new, 1, timeout, FfiMutationMode::None)
        .unwrap();
    assert_eq!(ev_new_ok.exit, Exit::Success(42001));

    // 3. Isolated C callback fixture with scoped registration + unregister acknowledgement.
    let ev_cb = rt
        .invoke_isolated("callback_bounded", false, &permitted, &desc_new, 1, timeout, FfiMutationMode::None)
        .unwrap();
    assert!(!ev_cb.watchdog_killed);
    assert_eq!(ev_cb.exit, Exit::Success(42)); // 10 + 20 + 12
    assert_eq!(ev_cb.callbacks_ran, 3);
    assert_eq!(ev_cb.post_unreg_callbacks, 0);
    assert_eq!(ev_cb.owner_leaks(), 0);

    // 4. Unbounded C worker containment: manager terminates + reaps BEFORE outer watchdog!
    let containment = rt
        .run_unbounded_worker_containment(
            &permitted,
            &desc_new,
            1,
            Duration::from_millis(120),
            Duration::from_millis(800),
            Duration::from_secs(5),
        )
        .expect("run unbounded worker containment");
    assert!(
        containment.readiness_line.starts_with("WORKER_READY"),
        "worker must prove it entered native loop: got `{}`",
        containment.readiness_line
    );
    assert!(containment.terminated_by_manager);
    assert!(containment.acknowledged);
    assert!(containment.reaped);
    assert!(
        !containment.killed_by_outer_watchdog,
        "expected worker containment must complete before outer safety watchdog"
    );

    // 5. Negative watchdog control: prove outer safety watchdog kill is classified as FAIL!
    let runaway = run_child_with_watchdog(
        &build.output_binary,
        &["runaway_watchdog_probe"],
        &[],
        Duration::from_millis(150),
    )
    .expect("run runaway watchdog probe");
    assert!(
        matches!(runaway.outcome, WatchdogOutcome::WatchdogKilledFail { reaped: true, .. }),
        "outer safety watchdog kill MUST be classified as WatchdogKilledFail (FAIL), got {:?}",
        runaway.outcome
    );
    assert!(runaway.outcome.is_watchdog_kill());
    assert!(!runaway.outcome.is_clean_success());
}

// =========================================================================
// 10. Gate-SCHEMA-13 All Three Required Mutants (MUST FAIL)
// =========================================================================

#[test]
fn test_w17_gate13_three_mutants_must_fail() {
    let build = built_c_ffi_fixture();
    let mut rt = FfiBoundaryRuntime::new(9, FfiManifest::store_read_v1(), build.output_binary.clone())
        .expect("init runtime");
    let owner = rt.allocate_owner(1);
    let desc = FfiAbiDescriptor::valid(1, owner);
    let denied = CapabilityMask::none();
    let permitted = permitted_store_caps();
    let timeout = Duration::from_secs(10);

    // Mutant 1: Skipped capability check -> constructor counter becomes > 0 under denied context!
    let m1 = rt
        .invoke_isolated(
            "success_positive",
            false,
            &denied,
            &desc,
            1,
            timeout,
            FfiMutationMode::SkipCapabilityCheck,
        )
        .unwrap();
    assert!(
        m1.constructor_runs_observed > 0 && m1.native_entries_observed > 0,
        "Gate-13 Mutant 1 (skipped capability) MUST violate denied constructor_runs==0 invariant"
    );

    // Mutant 2: Misclassified EINTR -> becomes Cause::Interrupted without guest cancel token!
    let m2 = rt
        .invoke_isolated(
            "status_eintr_uncancelled",
            false,
            &permitted,
            &desc,
            1,
            timeout,
            FfiMutationMode::MisclassifyEintrAsInterrupt,
        )
        .unwrap();
    assert!(
        m2.exit.is_interrupt(),
        "Gate-13 Mutant 2 (misclassified EINTR) MUST violate !exit.is_interrupt() invariant"
    );

    // Mutant 3: Leaked owner on failure -> allocs != frees (owner_leaks > 0)!
    let m3 = rt
        .invoke_isolated(
            "partial_init_cleanup",
            false,
            &permitted,
            &desc,
            1,
            timeout,
            FfiMutationMode::LeakOwnerOnFailure,
        )
        .unwrap();
    assert!(
        m3.owner_leaks() > 0 && m3.owner_allocs != m3.owner_frees,
        "Gate-13 Mutant 3 (leaked owner) MUST violate owner_leaks==0 invariant (allocs={}, frees={})",
        m3.owner_allocs,
        m3.owner_frees
    );
}

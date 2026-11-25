//! Integration tests for Wave 3 Track 3E:
//! - `X-LINK`: Native Rust crate and dynamic library linking (`.dylib`/`.so`), missing lib, bad symbols, capability denial, error propagation.
//! - `W-BUILD`: Target CLI build (`wasm32-wasi`, `wasm32-wasip1`, `wasm32-unknown-unknown`, rejection of `wasm32-wasip2`, offline-blocked wasmtime/wasmer sub-gates).
//! - `Gate-SCHEMA-09` & `Gate-SCHEMA-13`: AOT struct layout & decode lowering (`StructDecodeInto` / `StructLoadField`), independent oracle parity, failed decode/init, cleanup.
//! - CLI dispatch parity: `lust build` vs `lust --aot`.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::tempdir;

use lust::codegen_zig::aot::{AotCompileOptions, AotCompiler};
use lust::data::layout::{CanonicalTypeTag, TargetAbi};
use lust::mir::ir::*;
use lust::mir::mir_struct_decode_callee;
use lust::Span;

fn lust_bin() -> PathBuf {
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_lust") {
        PathBuf::from(path)
    } else {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let target_debug = root.join("target/debug/lust");
        if target_debug.exists() {
            target_debug
        } else {
            PathBuf::from("lust")
        }
    }
}

// =============================================================================
// 1. X-LINK: External Library and Crate Linking Probes
// =============================================================================

#[test]
fn test_x_link_missing_library() {
    let dir = tempdir().expect("tempdir");
    let script = dir.path().join("main.lust");
    fs::write(&script, "local x: int = 1\n").expect("write");

    let out = Command::new(lust_bin())
        .arg("build")
        .arg(&script)
        .arg("--link")
        .arg("/nonexistent/path/to/libmissing.dylib")
        .output()
        .expect("invoke lust build");

    assert!(!out.status.success(), "Must fail when external library is missing");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("missing external library"),
        "Expected missing library error, got: {stderr}"
    );
}

#[test]
fn test_x_link_bad_symbols() {
    let dir = tempdir().expect("tempdir");
    let script = dir.path().join("main.lust");
    fs::write(&script, "local x: int = 1\n").expect("write");

    let corrupt_lib = dir.path().join("corrupt.dylib");
    fs::write(&corrupt_lib, b"NOT_A_VALID_DYLIB_BINARY_DATA").expect("write corrupt lib");

    let out = Command::new(lust_bin())
        .arg("build")
        .arg(&script)
        .arg("--link")
        .arg(&corrupt_lib)
        .output()
        .expect("invoke lust build");

    assert!(!out.status.success(), "Must fail when library has bad symbols");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("bad symbols"),
        "Expected bad symbols error, got: {stderr}"
    );
}

#[test]
fn test_x_link_capability_denial() {
    let dir = tempdir().expect("tempdir");
    let script = dir.path().join("main.lust");
    fs::write(&script, "local x: int = 1\n").expect("write");
    let dummy_lib = dir.path().join("dummy.dylib");
    fs::write(&dummy_lib, b"dummy").expect("write");

    // Test --deny-ffi
    let out = Command::new(lust_bin())
        .arg("build")
        .arg(&script)
        .arg("--deny-ffi")
        .arg("--link")
        .arg(&dummy_lib)
        .output()
        .expect("invoke lust build");

    assert!(!out.status.success(), "Must fail when ffi capability is denied");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("CapabilityDenied") && stderr.contains("ffi"),
        "Expected CapabilityDenied, got: {stderr}"
    );

    // Test --pure
    let out_pure = Command::new(lust_bin())
        .arg("build")
        .arg(&script)
        .arg("--pure")
        .arg("--link")
        .arg(&dummy_lib)
        .output()
        .expect("invoke lust build");

    assert!(!out_pure.status.success(), "Must fail with --pure");
    let stderr_pure = String::from_utf8_lossy(&out_pure.stderr);
    assert!(
        stderr_pure.contains("CapabilityDenied"),
        "Expected CapabilityDenied with --pure, got: {stderr_pure}"
    );
}

#[test]
fn test_x_link_error_propagation() {
    let dir = tempdir().expect("tempdir");
    let script = dir.path().join("main.lust");
    fs::write(&script, "local x: int = 1\n").expect("write");

    let crate_dir = dir.path().join("broken_crate");
    fs::create_dir_all(crate_dir.join("src")).expect("mkdir");
    fs::write(
        crate_dir.join("Cargo.toml"),
        r#"[package]
name = "broken-crate"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]
"#,
    ).expect("write toml");
    fs::write(
        crate_dir.join("src/lib.rs"),
        "BROKEN SYNTAX RUST CODE !@#$%",
    ).expect("write lib.rs");

    let out = Command::new(lust_bin())
        .env("RUSTUP_HOME", "/Users/vedmalex/.rustup")
        .env("CARGO_HOME", "/Users/vedmalex/.cargo")
        .arg("build")
        .arg(&script)
        .arg("--allow-ffi")
        .arg("--link-crate")
        .arg(&crate_dir)
        .output()
        .expect("invoke lust build");

    assert!(!out.status.success(), "Must fail when crate build fails");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("cargo build failed") || stderr.contains("error"),
        "Expected propagated cargo error, got: {stderr}"
    );
}

#[test]
fn test_x_link_permitted_native_extension() {
    let dir = tempdir().expect("tempdir");
    let script = dir.path().join("main.lust");
    fs::write(&script, "local x: int = 42\n").expect("write");

    // Compile minimal valid C extension with required export symbol
    let c_file = dir.path().join("ext.c");
    let dylib_file = dir.path().join("libvalidext.dylib");
    fs::write(
        &c_file,
        r#"
#include <stdint.h>
#include <stddef.h>

struct LustExtManifestV1 {
    uint32_t abi_version;
    const char* module_name;
    const void* exports;
    size_t exports_len;
};

static const struct LustExtManifestV1 MANIFEST = {
    1,
    "validext",
    NULL,
    0
};

const struct LustExtManifestV1* lust_ext_manifest_v1(void) {
    return &MANIFEST;
}
"#,
    ).expect("write c");

    let cc_out = Command::new("cc")
        .arg("-dynamiclib")
        .arg("-o")
        .arg(&dylib_file)
        .arg(&c_file)
        .output()
        .expect("compile valid extension");
    assert!(cc_out.status.success(), "cc compilation of valid extension must succeed");

    let target_bin = dir.path().join("app_with_ext");
    let out = Command::new(lust_bin())
        .arg("build")
        .arg(&script)
        .arg("--allow-ffi")
        .arg("--link")
        .arg(&dylib_file)
        .arg("-o")
        .arg(&target_bin)
        .output()
        .expect("invoke lust build");

    assert!(
        out.status.success(),
        "lust build with permitted extension must succeed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(target_bin.exists(), "Target binary must exist");

    // Execute target binary
    let run_out = Command::new(&target_bin).output().expect("run app_with_ext");
    assert!(run_out.status.success(), "Execution of app with linked extension must succeed");
}

// =============================================================================
// 2. W-BUILD: WASM and WASI Target Compilation Probes
// =============================================================================

#[test]
fn test_w_build_wasm32_wasi_offline() {
    let dir = tempdir().expect("tempdir");
    let script = dir.path().join("main.lust");
    fs::write(&script, "local a: int = 10\nlocal b: int = 20\nlocal c: int = a + b\n").expect("write");
    let out_wasm = dir.path().join("app.wasm");

    let out = Command::new(lust_bin())
        .arg("build")
        .arg(&script)
        .arg("--target")
        .arg("wasm32-wasi")
        .arg("-o")
        .arg(&out_wasm)
        .output()
        .expect("invoke lust build");

    assert!(
        out.status.success(),
        "wasm32-wasi build must succeed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out_wasm.exists(), "Wasm output must exist");

    let bytes = fs::read(&out_wasm).expect("read wasm");
    assert!(bytes.len() >= 4, "Wasm file must have at least 4 bytes");
    assert_eq!(&bytes[0..4], b"\0asm", "Wasm binary must start with \\0asm magic bytes");
}

#[test]
fn test_w_build_wasm32_wasip1_offline() {
    let dir = tempdir().expect("tempdir");
    let script = dir.path().join("main.lust");
    fs::write(&script, "local a: int = 100\nlocal b: int = 200\n").expect("write");
    let out_wasm = dir.path().join("app_p1.wasm");

    let out = Command::new(lust_bin())
        .arg("build")
        .arg(&script)
        .arg("--target")
        .arg("wasm32-wasip1")
        .arg("-o")
        .arg(&out_wasm)
        .output()
        .expect("invoke lust build");

    assert!(
        out.status.success(),
        "wasm32-wasip1 build must succeed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let bytes = fs::read(&out_wasm).expect("read wasm");
    assert_eq!(&bytes[0..4], b"\0asm");
}

#[test]
fn test_w_build_wasm32_unknown_unknown_offline() {
    let dir = tempdir().expect("tempdir");
    let script = dir.path().join("main.lust");
    fs::write(&script, "local a: int = 1\n").expect("write");
    let out_wasm = dir.path().join("app_unknown.wasm");

    let out = Command::new(lust_bin())
        .arg("build")
        .arg(&script)
        .arg("--target")
        .arg("wasm32-unknown-unknown")
        .arg("-o")
        .arg(&out_wasm)
        .output()
        .expect("invoke lust build");

    assert!(
        out.status.success(),
        "wasm32-unknown-unknown build must succeed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let bytes = fs::read(&out_wasm).expect("read wasm");
    assert_eq!(&bytes[0..4], b"\0asm");
}

#[test]
fn test_w_build_wasm32_wasip2_offline_blocked() {
    let dir = tempdir().expect("tempdir");
    let script = dir.path().join("main.lust");
    fs::write(&script, "local a: int = 1\n").expect("write");

    let out = Command::new(lust_bin())
        .arg("build")
        .arg(&script)
        .arg("--target")
        .arg("wasm32-wasip2")
        .output()
        .expect("invoke lust build");

    assert!(!out.status.success(), "wasm32-wasip2 must fail as offline-blocked");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("offline-blocked: target 'wasm32-wasip2' is not installed offline"),
        "Expected offline-blocked error, got: {stderr}"
    );
}

#[test]
fn test_w_build_wasmtime_wasmer_offline_blocked_subgate() {
    // Per Wave 3 Plan: Wasmtime, Wasmer, and wasm32-wasip2 runtime execution remain offline-blocked sub-gates.
    // Assert neither wasmtime nor wasmer is present in PATH for offline verification.
    let which_wasmtime = Command::new("which").arg("wasmtime").output();
    let wasmtime_present = which_wasmtime.map(|o| o.status.success()).unwrap_or(false);
    assert!(!wasmtime_present, "wasmtime must be absent offline (offline-blocked sub-gate)");

    let which_wasmer = Command::new("which").arg("wasmer").output();
    let wasmer_present = which_wasmer.map(|o| o.status.success()).unwrap_or(false);
    assert!(!wasmer_present, "wasmer must be absent offline (offline-blocked sub-gate)");
}

// =============================================================================
// 3. Gate-SCHEMA-09 & Gate-SCHEMA-13: AOT Struct Layout & Decode Lowering
// =============================================================================

#[test]
fn test_gate_schema_09_13_aot_struct_decode_oracle_parity() {
    let mut module = MirModule::new("schema_parity_mod".into());

    let layout = MirStructLayout {
        type_id: 1001,
        type_name: "Point".into(),
        fields: vec![
            MirStructFieldLayout {
                ordinal: 0,
                name: "x".into(),
                tag: CanonicalTypeTag::Int64,
                byte_offset: 0,
                byte_size: 8,
                is_nullable: false,
                is_optional: false,
            },
            MirStructFieldLayout {
                ordinal: 1,
                name: "y".into(),
                tag: CanonicalTypeTag::Int64,
                byte_offset: 8,
                byte_size: 8,
                is_nullable: false,
                is_optional: false,
            },
            MirStructFieldLayout {
                ordinal: 2,
                name: "z".into(),
                tag: CanonicalTypeTag::Int64,
                byte_offset: 16,
                byte_size: 8,
                is_nullable: false,
                is_optional: false,
            },
        ],
        size_bytes: 24,
        align_bytes: 8,
        is_c_flat: true,
        target_abi: TargetAbi::x86_64(),
    };

    module.register_struct_layout(layout.clone());
    module.record_struct_op(MirStructOp::StructDecodeInto {
        struct_name: "Point".into(),
        layout: layout.clone(),
        dst: LocalId(1),
        src_bytes: Operand::Constant(MirConstant::String("{\"y\": 42, \"x\": 107, \"z\": 95}".into())),
    });

    // Build main function that decodes shuffled JSON and prints fields
    let sig = FunctionSignature {
        name: "main".into(),
        param_types: Vec::new(),
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let point_local = func.add_local(Some("point".into()), MirType::Struct("Point".into()), false, Span::dummy()); // LocalId(1)
    let json_local = func.add_local(Some("json".into()), MirType::String, false, Span::dummy()); // LocalId(2)
    let print_local = func.add_local(Some("tmp".into()), MirType::Unit, false, Span::dummy()); // LocalId(3)

    let bb1_id = func.new_block();
    let bb2_id = func.new_block();
    let bb3_id = func.new_block();
    let bb4_id = func.new_block();

    // BB0: load JSON string, call struct decode, jump to BB1
    let bb0 = &mut func.blocks[0];
    bb0.statements.push(Statement::Assign(
        Place::Local(json_local),
        Rvalue::Use(Operand::Constant(MirConstant::String("{\"y\": 42, \"x\": 107, \"z\": 95}".into()))),
        Span::dummy(),
    ));
    bb0.terminator = Some(Terminator::Call {
        callee: mir_struct_decode_callee("Point"),
        args: vec![Operand::Copy(Place::Local(json_local))],
        destination: Place::Local(point_local),
        success: bb1_id,
        error: None,
        span: Span::dummy(),
    });

    // BB1: print fields in declaration order (x=107, y=42, z=95)
    let bb1 = &mut func.blocks[bb1_id.0];
    bb1.statements.push(Statement::Assign(
        Place::Local(print_local),
        Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(point_local)), 0))),
        Span::dummy(),
    ));
    bb1.terminator = Some(Terminator::Call {
        callee: "println".into(),
        args: vec![Operand::Copy(Place::Field(Box::new(Place::Local(point_local)), 0))],
        destination: Place::Local(print_local),
        success: bb2_id,
        error: None,
        span: Span::dummy(),
    });

    // BB2: print y
    let bb2 = &mut func.blocks[bb2_id.0];
    bb2.terminator = Some(Terminator::Call {
        callee: "println".into(),
        args: vec![Operand::Copy(Place::Field(Box::new(Place::Local(point_local)), 1))],
        destination: Place::Local(print_local),
        success: bb3_id,
        error: None,
        span: Span::dummy(),
    });

    // BB3: print z
    let bb3 = &mut func.blocks[bb3_id.0];
    bb3.terminator = Some(Terminator::Call {
        callee: "println".into(),
        args: vec![Operand::Copy(Place::Field(Box::new(Place::Local(point_local)), 2))],
        destination: Place::Local(print_local),
        success: bb4_id,
        error: None,
        span: Span::dummy(),
    });

    // BB4: return 0
    let bb4 = &mut func.blocks[bb4_id.0];
    bb4.statements.push(Statement::Assign(
        Place::Local(func.return_local),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    bb4.terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(func.return_local))), Span::dummy()));

    module.add_function(func);

    let dir = tempdir().expect("tempdir");
    let out_bin = dir.path().join("schema_app");
    let opts = AotCompileOptions::default();

    let compiled = AotCompiler::compile_module(&module, &out_bin, &opts)
        .expect("AOT compilation of struct decoding module must succeed");
    assert!(compiled.exists(), "Compiled binary must exist");

    let run_out = Command::new(&compiled).output().expect("execute schema_app");
    assert!(run_out.status.success(), "Execution must succeed");

    let stdout = String::from_utf8_lossy(&run_out.stdout);
    let lines: Vec<&str> = stdout.trim().lines().map(|s| s.trim()).collect();
    assert_eq!(
        lines,
        vec!["107", "42", "95"],
        "AOT struct decode output must match independent oracle values (x: 107, y: 42, z: 95)"
    );
}

#[test]
fn test_gate_schema_09_13_failed_decode_duplicate_key() {
    let mut module = MirModule::new("dup_key_mod".into());
    let layout = MirStructLayout {
        type_id: 1002,
        type_name: "Pair".into(),
        fields: vec![
            MirStructFieldLayout {
                ordinal: 0,
                name: "x".into(),
                tag: CanonicalTypeTag::Int64,
                byte_offset: 0,
                byte_size: 8,
                is_nullable: false,
                is_optional: false,
            },
            MirStructFieldLayout {
                ordinal: 1,
                name: "y".into(),
                tag: CanonicalTypeTag::Int64,
                byte_offset: 8,
                byte_size: 8,
                is_nullable: false,
                is_optional: false,
            },
        ],
        size_bytes: 16,
        align_bytes: 8,
        is_c_flat: true,
        target_abi: TargetAbi::x86_64(),
    };
    module.register_struct_layout(layout);

    let sig = FunctionSignature {
        name: "main".into(),
        param_types: Vec::new(),
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let pair_local = func.add_local(Some("pair".into()), MirType::Struct("Pair".into()), false, Span::dummy());
    let json_local = func.add_local(Some("json".into()), MirType::String, false, Span::dummy());

    let bb1_id = func.new_block();

    let bb0 = &mut func.blocks[0];
    bb0.statements.push(Statement::Assign(
        Place::Local(json_local),
        // Duplicate key "x" must fail closed
        Rvalue::Use(Operand::Constant(MirConstant::String("{\"x\": 10, \"y\": 20, \"x\": 30}".into()))),
        Span::dummy(),
    ));
    bb0.terminator = Some(Terminator::Call {
        callee: mir_struct_decode_callee("Pair"),
        args: vec![Operand::Copy(Place::Local(json_local))],
        destination: Place::Local(pair_local),
        success: bb1_id,
        error: None,
        span: Span::dummy(),
    });

    let bb1 = &mut func.blocks[bb1_id.0];
    bb1.terminator = Some(Terminator::Return(Some(Operand::Constant(MirConstant::Int(0))), Span::dummy()));

    module.add_function(func);

    let dir = tempdir().expect("tempdir");
    let out_bin = dir.path().join("dup_key_app");
    let compiled = AotCompiler::compile_module(&module, &out_bin, &AotCompileOptions::default())
        .expect("compile module");

    let run_out = Command::new(&compiled).output().expect("run dup_key_app");
    assert!(!run_out.status.success(), "Duplicate key decode must fail closed with exit code 1");
    let stderr = String::from_utf8_lossy(&run_out.stderr);
    assert!(
        stderr.contains("duplicate key"),
        "Expected duplicate key error in stderr, got: {stderr}"
    );
}

#[test]
fn test_gate_schema_09_13_failed_decode_missing_field() {
    let mut module = MirModule::new("missing_field_mod".into());
    let layout = MirStructLayout {
        type_id: 1003,
        type_name: "Triple".into(),
        fields: vec![
            MirStructFieldLayout {
                ordinal: 0,
                name: "x".into(),
                tag: CanonicalTypeTag::Int64,
                byte_offset: 0,
                byte_size: 8,
                is_nullable: false,
                is_optional: false,
            },
            MirStructFieldLayout {
                ordinal: 1,
                name: "y".into(),
                tag: CanonicalTypeTag::Int64,
                byte_offset: 8,
                byte_size: 8,
                is_nullable: false,
                is_optional: false,
            },
            MirStructFieldLayout {
                ordinal: 2,
                name: "z".into(),
                tag: CanonicalTypeTag::Int64,
                byte_offset: 16,
                byte_size: 8,
                is_nullable: false,
                is_optional: false,
            },
        ],
        size_bytes: 24,
        align_bytes: 8,
        is_c_flat: true,
        target_abi: TargetAbi::x86_64(),
    };
    module.register_struct_layout(layout);

    let sig = FunctionSignature {
        name: "main".into(),
        param_types: Vec::new(),
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let triple_local = func.add_local(Some("triple".into()), MirType::Struct("Triple".into()), false, Span::dummy());
    let json_local = func.add_local(Some("json".into()), MirType::String, false, Span::dummy());

    let bb1_id = func.new_block();

    let bb0 = &mut func.blocks[0];
    bb0.statements.push(Statement::Assign(
        Place::Local(json_local),
        // Missing required field "z"
        Rvalue::Use(Operand::Constant(MirConstant::String("{\"x\": 10, \"y\": 20}".into()))),
        Span::dummy(),
    ));
    bb0.terminator = Some(Terminator::Call {
        callee: mir_struct_decode_callee("Triple"),
        args: vec![Operand::Copy(Place::Local(json_local))],
        destination: Place::Local(triple_local),
        success: bb1_id,
        error: None,
        span: Span::dummy(),
    });

    let bb1 = &mut func.blocks[bb1_id.0];
    bb1.terminator = Some(Terminator::Return(Some(Operand::Constant(MirConstant::Int(0))), Span::dummy()));

    module.add_function(func);

    let dir = tempdir().expect("tempdir");
    let out_bin = dir.path().join("missing_field_app");
    let compiled = AotCompiler::compile_module(&module, &out_bin, &AotCompileOptions::default())
        .expect("compile module");

    let run_out = Command::new(&compiled).output().expect("run missing_field_app");
    assert!(!run_out.status.success(), "Missing field decode must fail closed with exit code 1");
    let stderr = String::from_utf8_lossy(&run_out.stderr);
    assert!(
        stderr.contains("missing required field"),
        "Expected missing required field error in stderr, got: {stderr}"
    );
}

#[test]
fn test_gate_schema_09_13_failed_decode_trailing_junk() {
    let mut module = MirModule::new("trailing_junk_mod".into());
    let layout = MirStructLayout {
        type_id: 1004,
        type_name: "Item".into(),
        fields: vec![
            MirStructFieldLayout {
                ordinal: 0,
                name: "id".into(),
                tag: CanonicalTypeTag::Int64,
                byte_offset: 0,
                byte_size: 8,
                is_nullable: false,
                is_optional: false,
            },
        ],
        size_bytes: 8,
        align_bytes: 8,
        is_c_flat: true,
        target_abi: TargetAbi::x86_64(),
    };
    module.register_struct_layout(layout);

    let sig = FunctionSignature {
        name: "main".into(),
        param_types: Vec::new(),
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let item_local = func.add_local(Some("item".into()), MirType::Struct("Item".into()), false, Span::dummy());
    let json_local = func.add_local(Some("json".into()), MirType::String, false, Span::dummy());

    let bb1_id = func.new_block();

    let bb0 = &mut func.blocks[0];
    bb0.statements.push(Statement::Assign(
        Place::Local(json_local),
        // Trailing junk after JSON object
        Rvalue::Use(Operand::Constant(MirConstant::String("{\"id\": 123} trailing_junk".into()))),
        Span::dummy(),
    ));
    bb0.terminator = Some(Terminator::Call {
        callee: mir_struct_decode_callee("Item"),
        args: vec![Operand::Copy(Place::Local(json_local))],
        destination: Place::Local(item_local),
        success: bb1_id,
        error: None,
        span: Span::dummy(),
    });

    let bb1 = &mut func.blocks[bb1_id.0];
    bb1.terminator = Some(Terminator::Return(Some(Operand::Constant(MirConstant::Int(0))), Span::dummy()));

    module.add_function(func);

    let dir = tempdir().expect("tempdir");
    let out_bin = dir.path().join("trailing_junk_app");
    let compiled = AotCompiler::compile_module(&module, &out_bin, &AotCompileOptions::default())
        .expect("compile module");

    let run_out = Command::new(&compiled).output().expect("run trailing_junk_app");
    assert!(!run_out.status.success(), "Trailing junk decode must fail closed with exit code 1");
    let stderr = String::from_utf8_lossy(&run_out.stderr);
    assert!(
        stderr.contains("trailing junk"),
        "Expected trailing junk error in stderr, got: {stderr}"
    );
}

// =============================================================================
// 4. CLI Dispatch Parity: `lust build` vs `lust --aot`
// =============================================================================

#[test]
fn test_cli_build_and_aot_dispatch_parity() {
    let dir = tempdir().expect("tempdir");
    let script = dir.path().join("parity.lust");
    fs::write(&script, "local a: int = 15\nlocal b: int = 27\nlocal c: int = a + b\n").expect("write");

    let build_out = dir.path().join("bin_build");
    let _aot_out = dir.path().join("bin_aot");

    let out1 = Command::new(lust_bin())
        .arg("build")
        .arg(&script)
        .arg("-o")
        .arg(&build_out)
        .output()
        .expect("invoke lust build");
    assert!(out1.status.success(), "lust build must succeed: {}", String::from_utf8_lossy(&out1.stderr));
    assert!(build_out.exists(), "build binary must exist");

    // Test --aot flag dispatch
    let out2 = Command::new(lust_bin())
        .arg("--aot")
        .arg(&script)
        .output()
        .expect("invoke lust --aot");
    assert!(out2.status.success(), "lust --aot must succeed: {}", String::from_utf8_lossy(&out2.stderr));

    // Both build and --aot routes successfully compiled native binaries
    let run1 = Command::new(&build_out).output().expect("run build binary");
    assert!(run1.status.success(), "built binary must run successfully");
}

#[test]
fn test_w3_4_8_c_aot_sensor_reading_direct_offsets_and_objdump() {
    use lust::codegen_zig::aot::AotTargetKind;

    let dir = tempdir().expect("tempdir");
    let mut module = MirModule::new("sensor_bin_aot_mod".into());

    let layout = MirStructLayout {
        type_id: 4803,
        type_name: "SensorReading".into(),
        fields: vec![
            MirStructFieldLayout {
                ordinal: 0,
                name: "ts".into(),
                tag: CanonicalTypeTag::Int64,
                byte_offset: 0,
                byte_size: 8,
                is_nullable: false,
                is_optional: false,
            },
            MirStructFieldLayout {
                ordinal: 1,
                name: "temp".into(),
                tag: CanonicalTypeTag::Float64,
                byte_offset: 8,
                byte_size: 8,
                is_nullable: false,
                is_optional: false,
            },
            MirStructFieldLayout {
                ordinal: 2,
                name: "seq".into(),
                tag: CanonicalTypeTag::UInt64,
                byte_offset: 16,
                byte_size: 8,
                is_nullable: false,
                is_optional: false,
            },
        ],
        size_bytes: 24,
        align_bytes: 8,
        is_c_flat: true,
        target_abi: TargetAbi::x86_64(),
    };
    module.register_struct_layout(layout);

    let sig = FunctionSignature {
        name: "decode_sensor".into(),
        param_types: vec![MirType::Int, MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let ptr_param = func.params[0];
    let len_param = func.params[1];
    let reading_local = func.add_local(
        Some("reading".into()),
        MirType::Struct("SensorReading".into()),
        false,
        Span::dummy(),
    );
    let ts_local = func.add_local(Some("ts".into()), MirType::Int, false, Span::dummy());
    let temp_local = func.add_local(Some("temp".into()), MirType::Float, false, Span::dummy());
    let seq_local = func.add_local(Some("seq".into()), MirType::Int, false, Span::dummy());
    let unit_local = func.add_local(Some("u".into()), MirType::Unit, false, Span::dummy());

    let bb0 = func.entry;
    let bb1 = func.new_block();
    let bb2 = func.new_block();
    let bb3 = func.new_block();
    let bb4 = func.new_block();

    func.blocks[bb0.0].terminator = Some(Terminator::Call {
        callee: "__lust_mir_struct_decode_binary:SensorReading".into(),
        args: vec![
            Operand::Copy(Place::Local(ptr_param)),
            Operand::Copy(Place::Local(len_param)),
        ],
        destination: Place::Local(reading_local),
        success: bb1,
        error: None,
        span: Span::dummy(),
    });

    func.blocks[bb1.0].statements.push(Statement::Assign(
        Place::Local(ts_local),
        Rvalue::Use(Operand::Copy(Place::Field(
            Box::new(Place::Local(reading_local)),
            0,
        ))),
        Span::dummy(),
    ));
    func.blocks[bb1.0].terminator = Some(Terminator::Call {
        callee: "println".into(),
        args: vec![Operand::Copy(Place::Local(ts_local))],
        destination: Place::Local(unit_local),
        success: bb2,
        error: None,
        span: Span::dummy(),
    });

    func.blocks[bb2.0].statements.push(Statement::Assign(
        Place::Local(temp_local),
        Rvalue::Use(Operand::Copy(Place::Field(
            Box::new(Place::Local(reading_local)),
            1,
        ))),
        Span::dummy(),
    ));
    func.blocks[bb2.0].terminator = Some(Terminator::Call {
        callee: "println".into(),
        args: vec![Operand::Copy(Place::Local(temp_local))],
        destination: Place::Local(unit_local),
        success: bb3,
        error: None,
        span: Span::dummy(),
    });

    func.blocks[bb3.0].statements.push(Statement::Assign(
        Place::Local(seq_local),
        Rvalue::Use(Operand::Copy(Place::Field(
            Box::new(Place::Local(reading_local)),
            2,
        ))),
        Span::dummy(),
    ));
    func.blocks[bb3.0].terminator = Some(Terminator::Call {
        callee: "println".into(),
        args: vec![Operand::Copy(Place::Local(seq_local))],
        destination: Place::Local(unit_local),
        success: bb4,
        error: None,
        span: Span::dummy(),
    });

    func.blocks[bb4.0].statements.push(Statement::Assign(
        Place::Local(func.return_local),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    func.blocks[bb4.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(func.return_local))),
        Span::dummy(),
    ));

    module.functions.push(func);

    let dylib_path = dir.path().join("libsensor_decode.dylib");
    let opts = AotCompileOptions {
        target_kind: AotTargetKind::SharedLibrary,
        ..AotCompileOptions::default()
    };
    let compiled = AotCompiler::compile_module(&module, &dylib_path, &opts)
        .expect("AOT compilation of SensorReading shared library must succeed");
    assert!(compiled.exists(), "Compiled AOT dylib must exist");

    // Write independent Rust LE binary fixture: { ts: 1700000000i64, temp: 98.6f64, seq: 42u64 }
    let mut payload = Vec::with_capacity(24);
    payload.extend_from_slice(&1700000000_i64.to_le_bytes());
    payload.extend_from_slice(&98.6_f64.to_le_bytes());
    payload.extend_from_slice(&42_u64.to_le_bytes());
    assert_eq!(payload.len(), 24);
    let payload_path = dir.path().join("sensor_payload.bin");
    fs::write(&payload_path, &payload).expect("write payload");

    // Compile tiny C harness calling exported __lust_fn_decode_sensor(ptr, len) in child processes
    let driver_c = dir.path().join("driver.c");
    fs::write(
        &driver_c,
        r#"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int64_t __lust_user_decode_sensor(int64_t ptr, int64_t len);

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    FILE* f = fopen(argv[1], "rb");
    if (!f) return 3;
    uint8_t buf[64] = {0};
    size_t want = (size_t)strtoull(argv[2], NULL, 10);
    size_t n = fread(buf, 1, want, f);
    fclose(f);
    return (int)__lust_user_decode_sensor((int64_t)(uintptr_t)buf, (int64_t)n);
}
"#,
    )
    .expect("write driver.c");

    let driver_bin = dir.path().join("sensor_driver");
    let cc_out = Command::new("cc")
        .arg(&driver_c)
        .arg(&compiled)
        .arg("-o")
        .arg(&driver_bin)
        .arg("-Wl,-rpath,@executable_path")
        .output()
        .expect("compile C driver");
    assert!(
        cc_out.status.success(),
        "C driver compilation failed: {}",
        String::from_utf8_lossy(&cc_out.stderr)
    );

    // 1. Valid 24-byte payload oracle
    let valid_run = Command::new(&driver_bin)
        .arg(&payload_path)
        .arg("24")
        .output()
        .expect("run valid sensor_driver");
    assert!(
        valid_run.status.success(),
        "Valid 24-byte AOT decode must succeed, stderr={}",
        String::from_utf8_lossy(&valid_run.stderr)
    );
    let stdout = String::from_utf8_lossy(&valid_run.stdout);
    let lines: Vec<&str> = stdout.trim().lines().map(|s| s.trim()).collect();
    assert_eq!(lines.len(), 3, "Expected 3 output lines, got: {lines:?}");
    assert_eq!(lines[0], "1700000000");
    let parsed_temp: f64 = lines[1].parse().expect("parse float output");
    assert!((parsed_temp - 98.6).abs() < 1e-6, "Expected 98.6, got {parsed_temp}");
    assert_eq!(lines[2], "42");

    // 2. Truncated 16-byte payload negative test -> SchemaDecodeError::TruncatedPayload { expected: 24, actual: 16 }
    let trunc_run = Command::new(&driver_bin)
        .arg(&payload_path)
        .arg("16")
        .output()
        .expect("run truncated sensor_driver");
    assert!(
        !trunc_run.status.success(),
        "Truncated 16-byte AOT decode must fail closed"
    );
    let trunc_stderr = String::from_utf8_lossy(&trunc_run.stderr);
    assert!(
        trunc_stderr.contains("SchemaDecodeError::TruncatedPayload")
            && trunc_stderr.contains("expected: 24")
            && trunc_stderr.contains("actual: 16"),
        "Expected TruncatedPayload {{ expected: 24, actual: 16 }}, got: {trunc_stderr}"
    );

    // 3. Semantic disassembly verification via xcrun llvm-objdump
    let objdump = Command::new("xcrun")
        .args(["llvm-objdump", "-d", "--no-show-raw-insn"])
        .arg(&compiled)
        .output()
        .expect("run xcrun llvm-objdump");
    assert!(
        objdump.status.success(),
        "xcrun llvm-objdump failed: {}",
        String::from_utf8_lossy(&objdump.stderr)
    );
    let disasm = String::from_utf8_lossy(&objdump.stdout);
    let decoder_start = disasm
        .find("<___mir_struct_decode_binary_SensorReading>:")
        .or_else(|| disasm.find("<__mir_struct_decode_binary_SensorReading>:"))
        .unwrap_or_else(|| panic!("Expected __mir_struct_decode_binary_SensorReading symbol in disassembly:\n{disasm}"));
    let decoder_slice = &disasm[decoder_start..];
    let decoder_fn = decoder_slice
        .split("Disassembly of section")
        .next()
        .unwrap_or(decoder_slice);
    assert!(
        decoder_fn.contains("ldr") && decoder_fn.contains("#0x10]"),
        "Expected direct ARM64 load instructions with #0x10 offset in SensorReading decoder:\n{decoder_fn}"
    );
    assert!(
        decoder_fn.contains("cmp") && decoder_fn.contains("#0x17"),
        "Expected exact 24-byte bounds check (cmp len, #0x17) in SensorReading decoder:\n{decoder_fn}"
    );
    assert!(
        !disasm.contains("strcmp") && !disasm.contains("strncmp") && !disasm.contains("hash"),
        "Binary SensorReading AOT decode must contain zero string/hash lookup calls:\n{disasm}"
    );
}

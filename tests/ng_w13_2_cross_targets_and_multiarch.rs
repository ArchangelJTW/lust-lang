//! Integration & multi-architecture verification matrix for `NG-W13-2` (`J-MULTIARCH-VERIFY`):
//!
//! 1. Cross-architecture `LogicalLayoutV1` / `TargetAbi` matrix using ONLY the canonical
//!    constructors `TargetAbi::rv32()`, `TargetAbi::aarch64()`, and `TargetAbi::x86_64()` (`F-14`).
//! 2. Honest offline sysroot detection (`check_target_core_installed`) separating installed targets
//!    (`aarch64-apple-darwin`, `wasm32-wasip1`, `wasm32-unknown-unknown`) from uninstalled targets
//!    (`wasm32-wasip2`, `thumbv7em-none-eabihf`, `riscv32imac-unknown-none-elf`) (`F-11`).
//! 3. Real offline `rustc` compilation for `aarch64-apple-darwin` (Mach-O execution),
//!    `wasm32-wasip1` (`.wasm` WASI binary), and `wasm32-unknown-unknown` (`#![no_std]` `.wasm` `cdylib`).
//! 4. `#![no_std]` `.rs` generation with `StaticSlabAllocator<65536>` preamble and deterministic
//!    `TargetStdlibNotInstalledOffline { target, generated_rs_path }` status for uninstalled targets
//!    (`wasm32-wasip2`, `thumbv7em-none-eabihf`, `riscv32imac-unknown-none-elf`), plus `--no-std`.
//! 5. Conditional/blocked `--backend cranelift-aot` adapter (`cranelift_object_backend.rs`):
//!    validates 100% CLIF compilability via `CraneliftJitCompiler`, then returns the explicit
//!    offline-blocked diagnostic without hand-rolling any Mach-O/ELF encoder (`F-11`).
//! 6. Local tech-debt cleanup in `benchmarks/numeric.rs:87` (`F-10`).

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::tempdir;

use lust::codegen_zig::aot::compiler::RustLinkOptions;
use lust::codegen_zig::aot::{
    AotCompileOptions, AotCompiler, AotTargetKind, CRANELIFT_OBJECT_BACKEND_STATUS,
    CRANELIFT_OBJECT_BLOCKED_REASON, RustAotBuildOutcome, RustCompileOptions,
    check_target_core_installed, compile_mir_to_object_file, is_embedded_mcu_target,
    is_nostd_target, prepare_items_for_mir, validate_mir_module_clif,
};
use lust::data::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1, TargetAbi};
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::lowering::AstLowering;
use lust::parser::Parser;

const WASM_MAGIC_V1: &[u8; 8] = b"\0asm\x01\x00\x00\x00";

fn lust_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_lust"))
}

fn assert_nostd_safe_preamble(code: &str, label: &str) {
    assert!(
        code.contains("#![no_std]"),
        "[{label}] emitted Rust must start with `#![no_std]`:\n{code}"
    );
    assert!(
        code.contains("StaticSlabAllocator<65536>"),
        "[{label}] emitted `#![no_std]` Rust must include `StaticSlabAllocator<65536>` preamble:\n{code}"
    );
    assert!(
        !code.contains("std::process::") && !code.contains("std::io::"),
        "[{label}] emitted `#![no_std]` Rust must not reference `std::process` or `std::io`:\n{code}"
    );
    assert!(
        !code.contains("unsafe"),
        "[{label}] emitted Rust must contain zero `unsafe` (TASK_ALLOWS_UNSAFE=0):\n{code}"
    );
    assert!(
        !code.contains(".unwrap("),
        "[{label}] emitted Rust must contain zero `.unwrap(` calls (Hazard 9):\n{code}"
    );
    assert!(
        !code.contains("panic!("),
        "[{label}] emitted Rust must contain zero `panic!(` calls (Hazard 9):\n{code}"
    );
}

fn sample_compute_script() -> &'static str {
    r#"
struct Vec2
    x: int,
    y: int
end

function dot_sum(n: int): int
    local acc: int = 0
    local i: int = 1
    while i <= n do
        local v: Vec2 = Vec2 { x = i, y = i + 1 }
        acc = acc + v.x * v.y
        i = i + 1
    end
    return acc
end

function main(): int
    local total: int = dot_sum(10)
    print(total)
    return 0
end
"#
}

fn sample_schema_with_string_script() -> &'static str {
    r#"
struct SensorPacket
    active: bool,
    reading: int,
    label: string
end

function compute_checksum(base: int, delta: int): int
    local acc: int = base
    local i: int = 0
    while i < 8 do
        acc = acc + delta + i
        i = i + 1
    end
    return acc
end

function main(): int
    local p: SensorPacket = SensorPacket { active = true, reading = 42, label = "ok" }
    if p.active then
        return compute_checksum(p.reading, 3) - 94
    end
    return 1
end
"#
}

// =============================================================================
// 1. J-MULTIARCH-VERIFY: TargetAbi & LogicalLayoutV1 Matrix (F-14)
// =============================================================================

#[test]
fn test_j_multiarch_verify_target_abi_layout_matrix() {
    let rv32 = TargetAbi::rv32();
    let aarch64 = TargetAbi::aarch64();
    let x86_64 = TargetAbi::x86_64();

    assert_eq!((rv32.pointer_width, rv32.max_alignment), (4, 8));
    assert_eq!((aarch64.pointer_width, aarch64.max_alignment), (8, 16));
    assert_eq!((x86_64.pointer_width, x86_64.max_alignment), (8, 16));

    // A. Pure C-flat scalar struct: identical offsets/size/align across rv32, aarch64, x86_64
    let scalar_fields = || {
        vec![
            LogicalFieldDesc::new(0, "flag", CanonicalTypeTag::Bool),
            LogicalFieldDesc::new(1, "counter", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(2, "score", CanonicalTypeTag::Float64),
            LogicalFieldDesc::new(3, "oid", CanonicalTypeTag::ObjectId),
        ]
    };

    let l_rv32 = LogicalLayoutV1::new_with_abi(101, "TelemetryScalar", scalar_fields(), rv32);
    let l_arm64 = LogicalLayoutV1::new_with_abi(101, "TelemetryScalar", scalar_fields(), aarch64);
    let l_x64 = LogicalLayoutV1::new_with_abi(101, "TelemetryScalar", scalar_fields(), x86_64);

    for (label, layout) in [("rv32", &l_rv32), ("aarch64", &l_arm64), ("x86_64", &l_x64)] {
        assert!(layout.is_c_flat, "{label} scalar struct must be C-flat");
        assert!(
            layout.is_rust_repr_c_exact(),
            "{label} scalar struct must match Rust #[repr(C)] exactly"
        );
        assert_eq!(layout.fields[0].byte_offset, 0, "{label} flag offset");
        assert_eq!(layout.fields[1].byte_offset, 8, "{label} counter offset");
        assert_eq!(layout.fields[2].byte_offset, 16, "{label} score offset");
        assert_eq!(layout.fields[3].byte_offset, 24, "{label} oid offset");
        assert_eq!(layout.size_bytes, 40, "{label} total padded size");
        assert_eq!(layout.align_bytes, 8, "{label} struct align");
    }

    // B. Managed slices (`StringSlice`, `BinarySlice`): 24B/align 8 on 64-bit vs 16B/align 4 on rv32
    let slice_fields = || {
        vec![
            LogicalFieldDesc::new(0, "tag", CanonicalTypeTag::Bool),
            LogicalFieldDesc::new(1, "name", CanonicalTypeTag::StringSlice),
            LogicalFieldDesc::new(2, "blob", CanonicalTypeTag::BinarySlice),
        ]
    };

    let s_rv32 = LogicalLayoutV1::new_with_abi(102, "PacketSlices", slice_fields(), rv32);
    let s_arm64 = LogicalLayoutV1::new_with_abi(102, "PacketSlices", slice_fields(), aarch64);
    let s_x64 = LogicalLayoutV1::new_with_abi(102, "PacketSlices", slice_fields(), x86_64);

    assert_eq!(s_arm64.size_bytes, s_x64.size_bytes);
    assert_eq!(s_arm64.align_bytes, 8);
    assert_eq!(s_arm64.fields[1].byte_offset, 8);
    assert_eq!(s_arm64.fields[1].byte_size, 24);
    assert_eq!(s_arm64.fields[2].byte_offset, 32);
    assert_eq!(s_arm64.size_bytes, 56);
    assert!(s_arm64.is_rust_repr_c_exact());
    assert!(s_x64.is_rust_repr_c_exact());

    assert_eq!(s_rv32.align_bytes, 4);
    assert_eq!(s_rv32.fields[1].byte_offset, 4);
    assert_eq!(s_rv32.fields[1].byte_size, 16);
    assert_eq!(s_rv32.fields[2].byte_offset, 20);
    assert_eq!(s_rv32.size_bytes, 36);
    assert!(
        !s_rv32.is_rust_repr_c_exact(),
        "rv32 managed slices must not claim 64-bit Rust repr(C) exactness"
    );

    // C. Decimal128 alignment clamping on rv32 (max_alignment = 8) vs 64-bit (align = 16)
    let dec_fields = || {
        vec![
            LogicalFieldDesc::new(0, "active", CanonicalTypeTag::Bool),
            LogicalFieldDesc::new(1, "amount", CanonicalTypeTag::Decimal128),
        ]
    };
    let d_rv32 = LogicalLayoutV1::new_with_abi(103, "LedgerAmount", dec_fields(), rv32);
    let d_arm64 = LogicalLayoutV1::new_with_abi(103, "LedgerAmount", dec_fields(), aarch64);
    let d_x64 = LogicalLayoutV1::new_with_abi(103, "LedgerAmount", dec_fields(), x86_64);

    assert_eq!(d_arm64.align_bytes, 16);
    assert_eq!(d_arm64.fields[1].byte_offset, 16);
    assert_eq!(d_arm64.size_bytes, 32);
    assert_eq!(d_arm64, LogicalLayoutV1 { target_abi: aarch64, ..d_x64.clone() });
    assert!(d_arm64.is_rust_repr_c_exact());
    assert!(d_x64.is_rust_repr_c_exact());

    assert_eq!(d_rv32.align_bytes, 8);
    assert_eq!(d_rv32.fields[1].byte_offset, 8);
    assert_eq!(d_rv32.size_bytes, 24);
    assert!(!d_rv32.is_rust_repr_c_exact());

    // D. Nested struct resolution across all three TargetAbi values
    for abi in [rv32, aarch64, x86_64] {
        let inner = LogicalLayoutV1::new_with_abi(201, "Inner", scalar_fields(), abi);
        let outer_raw = LogicalLayoutV1::new_with_abi(
            202,
            "Outer",
            vec![
                LogicalFieldDesc::new(0, "seq", CanonicalTypeTag::Int64),
                LogicalFieldDesc::new(1, "inner", CanonicalTypeTag::NestedStruct(201)),
            ],
            abi,
        );
        let outer = outer_raw
            .resolve_nested_layouts(&|id| {
                if id == 201 {
                    Some((inner.size_bytes, inner.align_bytes))
                } else {
                    None
                }
            })
            .expect("nested layout must resolve across all ABIs");
        assert_eq!(outer.fields[0].byte_offset, 0);
        assert_eq!(outer.fields[1].byte_offset, 8);
        assert_eq!(outer.fields[1].byte_size, 40);
        assert_eq!(outer.size_bytes, 48);
        assert_eq!(outer.align_bytes, 8);
        assert!(outer.is_rust_repr_c_exact());
    }
}

// =============================================================================
// 2. Honest Offline Target Sysroot Detection (F-11)
// =============================================================================

#[test]
fn test_offline_target_sysroot_detection_matrix() {
    // Target classification helpers
    assert!(is_nostd_target("wasm32-unknown-unknown"));
    assert!(is_nostd_target("thumbv7em-none-eabihf"));
    assert!(is_nostd_target("riscv32imac-unknown-none-elf"));
    assert!(!is_nostd_target("wasm32-wasip1"));
    assert!(!is_nostd_target("aarch64-apple-darwin"));

    assert!(is_embedded_mcu_target("thumbv7em-none-eabihf"));
    assert!(is_embedded_mcu_target("riscv32imac-unknown-none-elf"));
    assert!(!is_embedded_mcu_target("wasm32-unknown-unknown"));
    assert!(!is_embedded_mcu_target("wasm32-wasip1"));

    // Installed offline targets on this host (`rustup target list --installed`)
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    assert!(
        check_target_core_installed("aarch64-apple-darwin").expect("check host target"),
        "aarch64-apple-darwin must have libcore installed offline"
    );
    assert!(
        check_target_core_installed("wasm32-wasip1").expect("check wasm32-wasip1"),
        "wasm32-wasip1 must have libcore installed offline"
    );
    assert!(
        check_target_core_installed("wasm32-unknown-unknown")
            .expect("check wasm32-unknown-unknown"),
        "wasm32-unknown-unknown must have libcore installed offline"
    );

    // Uninstalled offline targets (honest BLOCKED / fallback boundary)
    for uninstalled in [
        "wasm32-wasip2",
        "thumbv7em-none-eabihf",
        "riscv32imac-unknown-none-elf",
    ] {
        assert!(
            !check_target_core_installed(uninstalled).expect("check uninstalled target"),
            "expected {uninstalled} libcore to be absent offline"
        );
    }
}

// =============================================================================
// 3. Real Offline Compilation: Native Host (`aarch64-apple-darwin`)
// =============================================================================

#[test]
fn test_offline_compile_native_host_rust_aot_and_execute() {
    let tmp = tempdir().expect("tempdir");
    let script = tmp.path().join("native_compute.lust");
    fs::write(&script, sample_compute_script()).expect("write script");
    let out_bin = tmp.path().join("native_compute_bin");

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    let target_triple = Some("aarch64-apple-darwin");
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    let target_triple: Option<&str> = None;

    let mut cmd = Command::new(lust_bin());
    cmd.arg("build")
        .arg(&script)
        .arg("--backend")
        .arg("rust")
        .arg("-o")
        .arg(&out_bin);
    if let Some(triple) = target_triple {
        cmd.arg("--target").arg(triple);
    }
    let build_out = cmd.output().expect("run lust build native");
    assert!(
        build_out.status.success(),
        "native Rust AOT build failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&build_out.stdout),
        String::from_utf8_lossy(&build_out.stderr)
    );
    assert!(out_bin.exists(), "native binary must exist");

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        let bytes = fs::read(&out_bin).expect("read mach-o");
        assert!(bytes.len() > 1024, "mach-o binary must be non-empty");
        // Mach-O 64-bit little-endian magic (MH_MAGIC_64 = 0xfeedfacf -> cf fa ed fe)
        assert_eq!(
            &bytes[0..4],
            &[0xcf, 0xfa, 0xed, 0xfe],
            "aarch64-apple-darwin artifact must have Mach-O 64-bit magic"
        );
    }

    let run_out = Command::new(&out_bin)
        .output()
        .expect("execute compiled native binary");
    assert!(
        run_out.status.success(),
        "native binary exited non-zero: {}",
        String::from_utf8_lossy(&run_out.stderr)
    );
    // sum_{i=1..10} i*(i+1) = 440
    assert_eq!(String::from_utf8_lossy(&run_out.stdout).trim(), "440");
}

// =============================================================================
// 4. Real Offline Compilation: `wasm32-wasip1` (`.wasm` binary)
// =============================================================================

#[test]
fn test_offline_compile_wasm32_wasip1_rust_aot_api_and_cli() {
    let tmp = tempdir().expect("tempdir");

    // A. Direct API (`AotCompiler::compile_source_rust_cross`)
    let api_wasm = tmp.path().join("compute_api_wasip1.wasm");
    let opts = RustCompileOptions {
        aot: AotCompileOptions {
            target_kind: AotTargetKind::Executable,
            release_mode: "ReleaseFast".to_string(),
            strip: false,
            max_memory_bytes: None,
            target_triple: Some("wasm32-wasip1".to_string()),
        },
        ..RustCompileOptions::default()
    };
    let outcome = AotCompiler::compile_source_rust_cross(
        sample_compute_script(),
        &api_wasm,
        &opts,
        &RustLinkOptions::default(),
    )
    .expect("wasm32-wasip1 API cross-compile must succeed offline");
    assert_eq!(
        outcome,
        RustAotBuildOutcome::Compiled {
            target: "wasm32-wasip1".to_string(),
            artifact_path: api_wasm.clone(),
        }
    );
    let api_bytes = fs::read(&api_wasm).expect("read api wasm");
    assert!(
        api_bytes.len() > 64 && &api_bytes[0..8] == WASM_MAGIC_V1,
        "wasm32-wasip1 API artifact must have valid WebAssembly v1 header"
    );

    // B. CLI (`lust build --backend rust --target wasm32-wasip1`)
    let script = tmp.path().join("compute_cli.lust");
    fs::write(&script, sample_compute_script()).expect("write script");
    let cli_wasm = tmp.path().join("compute_cli_wasip1.wasm");
    let cli_out = Command::new(lust_bin())
        .arg("build")
        .arg(&script)
        .arg("--backend")
        .arg("rust")
        .arg("--target")
        .arg("wasm32-wasip1")
        .arg("-o")
        .arg(&cli_wasm)
        .output()
        .expect("run lust build wasm32-wasip1");
    assert!(
        cli_out.status.success(),
        "CLI wasm32-wasip1 Rust AOT build failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&cli_out.stdout),
        String::from_utf8_lossy(&cli_out.stderr)
    );
    let cli_bytes = fs::read(&cli_wasm).expect("read cli wasm");
    assert!(
        cli_bytes.len() > 64 && &cli_bytes[0..8] == WASM_MAGIC_V1,
        "wasm32-wasip1 CLI artifact must have valid WebAssembly v1 header"
    );
}

// =============================================================================
// 5. Real Offline Compilation: `wasm32-unknown-unknown` (`#![no_std]` `cdylib`)
// =============================================================================

#[test]
fn test_offline_compile_wasm32_unknown_unknown_nostd_cdylib_rust_aot() {
    let tmp = tempdir().expect("tempdir");

    // A. Direct API on program with scalar struct + string-bearing struct
    //    (verifies 32-bit target does not fail on 64-bit-only StringSlice size_of assertions!)
    let api_wasm = tmp.path().join("sensor_unk.wasm");
    let opts = RustCompileOptions {
        aot: AotCompileOptions {
            target_kind: AotTargetKind::Executable, // auto-promoted to SharedLibrary (cdylib)
            release_mode: "ReleaseFast".to_string(),
            strip: false,
            max_memory_bytes: None,
            target_triple: Some("wasm32-unknown-unknown".to_string()),
        },
        no_std: true,
        ..RustCompileOptions::default()
    };
    let outcome = AotCompiler::compile_source_rust_cross(
        sample_schema_with_string_script(),
        &api_wasm,
        &opts,
        &RustLinkOptions::default(),
    )
    .expect("wasm32-unknown-unknown #![no_std] cdylib build must succeed offline");
    assert_eq!(
        outcome,
        RustAotBuildOutcome::Compiled {
            target: "wasm32-unknown-unknown".to_string(),
            artifact_path: api_wasm.clone(),
        }
    );
    let api_bytes = fs::read(&api_wasm).expect("read api wasm32-unknown-unknown");
    assert!(
        api_bytes.len() > 32 && &api_bytes[0..8] == WASM_MAGIC_V1,
        "wasm32-unknown-unknown API artifact must have valid WebAssembly v1 header"
    );

    // B. CLI (`lust build --backend rust --target wasm32-unknown-unknown --no-std`)
    let script = tmp.path().join("compute_unk.lust");
    fs::write(&script, sample_compute_script()).expect("write script");
    let cli_wasm = tmp.path().join("compute_unk.wasm");
    let cli_out = Command::new(lust_bin())
        .arg("build")
        .arg(&script)
        .arg("--backend")
        .arg("rust")
        .arg("--no-std")
        .arg("--target=wasm32-unknown-unknown")
        .arg("-o")
        .arg(&cli_wasm)
        .output()
        .expect("run lust build wasm32-unknown-unknown");
    assert!(
        cli_out.status.success(),
        "CLI wasm32-unknown-unknown Rust AOT build failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&cli_out.stdout),
        String::from_utf8_lossy(&cli_out.stderr)
    );
    let cli_bytes = fs::read(&cli_wasm).expect("read cli wasm32-unknown-unknown");
    assert!(
        cli_bytes.len() > 32 && &cli_bytes[0..8] == WASM_MAGIC_V1,
        "wasm32-unknown-unknown CLI artifact must have valid WebAssembly v1 header"
    );
}

// =============================================================================
// 6. Uninstalled Offline Targets: `#![no_std]` `.rs` Generation & Diagnostic
// =============================================================================

#[test]
fn test_uninstalled_targets_generate_nostd_rs_with_static_slab_and_deterministic_status() {
    let tmp = tempdir().expect("tempdir");
    let script = tmp.path().join("mcu_firmware.lust");
    fs::write(&script, sample_compute_script()).expect("write script");

    for target in [
        "wasm32-wasip2",
        "thumbv7em-none-eabihf",
        "riscv32imac-unknown-none-elf",
    ] {
        // A. Direct API (`compile_source_rust_cross`)
        let out_bin = tmp
            .path()
            .join(format!("api_{}.bin", target.replace('-', "_")));
        let opts = RustCompileOptions {
            aot: AotCompileOptions {
                target_kind: AotTargetKind::Executable,
                release_mode: "ReleaseSmall".to_string(),
                strip: false,
                max_memory_bytes: Some(65536),
                target_triple: Some(target.to_string()),
            },
            no_std: true,
            ..RustCompileOptions::default()
        };
        let outcome = AotCompiler::compile_source_rust_cross(
            sample_compute_script(),
            &out_bin,
            &opts,
            &RustLinkOptions::default(),
        )
        .expect("uninstalled target must return Ok(TargetStdlibNotInstalledOffline)");

        let expected_rs = out_bin.with_extension("rs");
        assert_eq!(
            outcome,
            RustAotBuildOutcome::TargetStdlibNotInstalledOffline {
                target: target.to_string(),
                generated_rs_path: expected_rs.clone(),
            }
        );
        assert!(
            expected_rs.exists(),
            "generated #![no_std] .rs must exist for {target}"
        );
        let generated_src = fs::read_to_string(&expected_rs).expect("read generated .rs");
        assert_nostd_safe_preamble(&generated_src, target);
        assert!(
            generated_src.contains("#[panic_handler]"),
            "[{target}] generated #![no_std] source must include bare-metal #[panic_handler]"
        );

        // B. CLI (`lust build --backend rust --target <target> -o <out>`)
        let cli_out_bin = tmp
            .path()
            .join(format!("cli_{}.elf", target.replace('-', "_")));
        let cli_res = Command::new(lust_bin())
            .arg("build")
            .arg(&script)
            .arg("--backend")
            .arg("rust")
            .arg("--target")
            .arg(target)
            .arg("-o")
            .arg(&cli_out_bin)
            .output()
            .expect("invoke lust build uninstalled target");
        assert!(
            !cli_res.status.success(),
            "CLI build for uninstalled target {target} must exit non-zero (honest offline-blocked)"
        );
        let stderr = String::from_utf8_lossy(&cli_res.stderr);
        assert!(
            stderr.contains("TargetStdlibNotInstalledOffline") && stderr.contains(target),
            "[{target}] expected TargetStdlibNotInstalledOffline diagnostic, got: {stderr}"
        );
        // Verify old hardcoded wasm32-wasip2 ban was lifted for `--backend rust`
        assert!(
            !stderr.contains("offline-blocked: target 'wasm32-wasip2' is not installed offline"),
            "[{target}] `--backend rust` must use real sysroot check, not legacy Zig hardcoded ban"
        );
        let cli_rs = cli_out_bin.with_extension("rs");
        assert!(
            cli_rs.exists(),
            "CLI must materialize generated #![no_std] .rs at {} for {target}",
            cli_rs.display()
        );
        let cli_src = fs::read_to_string(&cli_rs).expect("read CLI generated .rs");
        assert_nostd_safe_preamble(&cli_src, &format!("cli:{target}"));
    }

    // C. Explicit `--emit=rs --no-std` exits 0 and writes `#![no_std]` `.rs` with `StaticSlabAllocator<65536>`
    let emit_only_rs = tmp.path().join("explicit_nostd.rs");
    let emit_res = Command::new(lust_bin())
        .arg("build")
        .arg(&script)
        .arg("--backend")
        .arg("rust")
        .arg("--no-std")
        .arg("--emit=rs")
        .arg("-o")
        .arg(&emit_only_rs)
        .output()
        .expect("invoke lust build --emit=rs --no-std");
    assert!(
        emit_res.status.success(),
        "--emit=rs --no-std failed:\n{}",
        String::from_utf8_lossy(&emit_res.stderr)
    );
    let emitted = fs::read_to_string(&emit_only_rs).expect("read explicit_nostd.rs");
    assert_nostd_safe_preamble(&emitted, "cli:--emit=rs --no-std");
}

// =============================================================================
// 7. Conditional / Blocked `--backend cranelift-aot` Adapter (F-11)
// =============================================================================

#[test]
fn test_cranelift_object_backend_validates_clif_and_fails_closed_offline() {
    let source = r#"
function add_mul(a: int, b: int): int
    return a * 3 + b
end

function main(): int
    return add_mul(10, 12)
end
"#;

    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");
    let prepared = prepare_items_for_mir(&items);
    let mir_module = AstLowering::lower_module_items("clif_check_mod", &prepared).expect("lower");

    // 1. 100% CLIF validation via CraneliftJitCompiler succeeds
    let report = validate_mir_module_clif(&mir_module, Some("aarch64-apple-darwin"))
        .expect("CLIF validation must succeed for valid MIR module");
    assert_eq!(report.module_name, "clif_check_mod");
    assert_eq!(report.object_emission_status, CRANELIFT_OBJECT_BACKEND_STATUS);
    assert!(
        report.validated_functions.iter().any(|f| f == "add_mul"),
        "add_mul must be CLIF-validated: {:?}",
        report.validated_functions
    );
    assert!(
        report.validated_functions.iter().any(|f| f == "main"),
        "main must be CLIF-validated: {:?}",
        report.validated_functions
    );

    // 2. Object emission fails closed with exact offline-blocked diagnostic (NO hand-rolled Mach-O/ELF!)
    let tmp = tempdir().expect("tempdir");
    let fake_obj = tmp.path().join("should_not_exist.o");
    let err = compile_mir_to_object_file(&mir_module, &fake_obj, Some("aarch64-apple-darwin"))
        .expect_err("cranelift-aot object emission must be BLOCKED offline");
    assert!(
        err.to_string().contains(CRANELIFT_OBJECT_BLOCKED_REASON),
        "unexpected cranelift-aot error: {err}"
    );
    assert!(
        !fake_obj.exists(),
        "blocked cranelift-aot backend must never fabricate an object file on disk"
    );

    // 3. CLI `--backend cranelift-aot` surfaces the exact blocked diagnostic
    let script = tmp.path().join("clif_cli.lust");
    fs::write(&script, source).expect("write clif_cli.lust");
    let cli_out = Command::new(lust_bin())
        .arg("build")
        .arg(&script)
        .arg("--backend")
        .arg("cranelift-aot")
        .arg("-o")
        .arg(&fake_obj)
        .output()
        .expect("invoke lust build --backend cranelift-aot");
    assert!(
        !cli_out.status.success(),
        "CLI --backend cranelift-aot must fail closed offline"
    );
    let stderr = String::from_utf8_lossy(&cli_out.stderr);
    assert!(
        stderr.contains(CRANELIFT_OBJECT_BLOCKED_REASON),
        "CLI stderr must report exact blocked diagnostic, got: {stderr}"
    );
    assert!(
        !fake_obj.exists(),
        "CLI --backend cranelift-aot must never write a bogus object file"
    );
}

// =============================================================================
// 8. Local Tech-Debt Cleanup Verification (`benchmarks/numeric.rs:87`, F-10)
// =============================================================================

#[test]
fn test_benchmarks_numeric_cleaned_of_removed_dynasm_counter() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let numeric_rs = fs::read_to_string(manifest_dir.join("benchmarks/numeric.rs"))
        .expect("read benchmarks/numeric.rs");
    assert!(
        !numeric_rs.contains("root_traces_compiled"),
        "benchmarks/numeric.rs must no longer assert removed Dynasm counter `root_traces_compiled`"
    );
    assert!(
        numeric_rs.contains("cranelift_native_entry_count() >= 1"),
        "benchmarks/numeric.rs must assert `cranelift_native_entry_count() >= 1` (F-10)"
    );
}

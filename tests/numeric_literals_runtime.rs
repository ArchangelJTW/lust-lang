#![cfg(feature = "std")]
//! Step 4 (P4): End-to-end runtime backend verification for numeric literals.
//!
//! Verifies exact character-by-character stdout parity across all 4 backends:
//! 1. Bytecode VM: `lust <file>` (or `run ... --backend vm`)
//! 2. Strict MIR: `lust --strict-mir <file>`
//! 3. Cranelift JIT: CLI `lust run <file> --backend cranelift` (note: in current CLI `run`,
//!    `--backend cranelift` delegates execution to the MIR Bytecode VM interpreter;
//!    true native Cranelift JIT is verified via direct library API `CraneliftJitCompiler`).
//! 4. Zig AOT: `lust build <file> --backend zig -o <bin>` && `<bin>`
//!
//! Golden fixture status:
//! `tests/fixtures/bitwise_golden.lust` uses `println(<expr>)` as the minimal common
//! denominator across all 4 backends to verify identical character-for-character stdout.
//!
//! Also verifies shift arithmetic masking (`& 63`) and two's-complement
//! bitwise assertions inside Lust scripts across all 4 backends.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn lust_bin() -> PathBuf {
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_lust") {
        return PathBuf::from(path);
    }
    let target_debug = PathBuf::from("./target/debug/lust");
    if target_debug.exists() {
        return target_debug;
    }
    let target_release = PathBuf::from("./target/release/lust");
    if target_release.exists() {
        return target_release;
    }
    PathBuf::from("lust")
}

fn run_cmd(bin: &Path, args: &[&str]) -> Output {
    let output = Command::new(bin)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("Failed to execute {:?} with args {:?}: {}", bin, args, e));
    assert!(
        output.status.success(),
        "Command {:?} {:?} failed with status {}.\nStdout: {}\nStderr: {}",
        bin,
        args,
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

#[test]
fn test_runtime_bitwise_golden_parity_all_four_backends() {
    let bin = lust_bin();
    let fixture_path = "tests/fixtures/bitwise_golden.lust";
    let stdout_path = "tests/fixtures/bitwise_golden.stdout";

    let expected = fs::read_to_string(stdout_path)
        .expect("tests/fixtures/bitwise_golden.stdout must exist");
    let expected_trimmed = expected.trim();

    // 1. VM (Bytecode VM)
    let out_vm = run_cmd(&bin, &[fixture_path]);
    let vm_stdout = String::from_utf8_lossy(&out_vm.stdout);
    assert_eq!(
        vm_stdout.trim(),
        expected_trimmed,
        "VM (Bytecode VM) stdout mismatch against golden stdout"
    );

    // 2. Strict MIR (--strict-mir)
    let out_mir = run_cmd(&bin, &["--strict-mir", fixture_path]);
    let mir_stdout = String::from_utf8_lossy(&out_mir.stdout);
    assert_eq!(
        mir_stdout.trim(),
        expected_trimmed,
        "Strict MIR (--strict-mir) stdout mismatch against golden stdout"
    );

    // 3. Cranelift JIT (run ... --backend cranelift)
    let out_jit = run_cmd(&bin, &["run", fixture_path, "--backend", "cranelift"]);
    let jit_stdout = String::from_utf8_lossy(&out_jit.stdout);
    assert_eq!(
        jit_stdout.trim(),
        expected_trimmed,
        "Cranelift JIT (--backend cranelift) stdout mismatch against golden stdout"
    );

    // 4. Zig AOT (build ... --backend zig -o <out>)
    let temp_dir = tempfile::tempdir().expect("Failed to create temporary directory for Zig AOT");
    let aot_bin = temp_dir.path().join("bitwise_golden_aot");
    let aot_bin_str = aot_bin.to_string_lossy().to_string();

    let build_out = run_cmd(
        &bin,
        &["build", fixture_path, "--backend", "zig", "-o", &aot_bin_str],
    );
    assert!(
        build_out.status.success(),
        "Zig AOT compilation must succeed"
    );
    assert!(aot_bin.exists(), "Zig AOT executable binary must exist");

    let out_aot = run_cmd(&aot_bin, &[]);
    let aot_stdout = String::from_utf8_lossy(&out_aot.stdout);
    assert_eq!(
        aot_stdout.trim(),
        expected_trimmed,
        "Zig AOT binary stdout mismatch against golden stdout"
    );

    // Cross-backend identity assertion: all 4 backends produce byte-for-byte identical output
    assert_eq!(vm_stdout.trim(), mir_stdout.trim(), "VM vs Strict MIR mismatch");
    assert_eq!(mir_stdout.trim(), jit_stdout.trim(), "Strict MIR vs Cranelift JIT mismatch");
    assert_eq!(jit_stdout.trim(), aot_stdout.trim(), "Cranelift JIT vs Zig AOT mismatch");
}

#[test]
fn test_runtime_all_ten_vectors_script_assertions_all_backends() {
    let bin = lust_bin();
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let script_file = temp_dir.path().join("all_ten_vectors.lust");

    let lust_source = r#"
assert(0xFFFF_FFFF_FFFF_FFFF >> 4 == -1)
assert(0x8000_0000_0000_0000 >> 63 == -1)
assert(1 << 64 == 1)
assert(0x5555_5555_5555_5555 ^^ 0xAAAA_AAAA_AAAA_AAAA == -1)
assert(0xDEAD_BEEF & 0xFFFF == 48879)
assert(0b1010_1010 | 0b0101_0101 == 255)
assert(0o755 & 0o777 == 493)
assert(-16 >> 2 == -4)
assert(40 >> 2 == 10)
assert((0x0F << 4) | 0x05 == 245)
println(1)
"#;
    fs::write(&script_file, lust_source).expect("write script");
    let script_str = script_file.to_string_lossy().to_string();

    // 1. VM
    let out_vm = run_cmd(&bin, &[&script_str]);
    assert_eq!(String::from_utf8_lossy(&out_vm.stdout).trim(), "1");

    // 2. Strict MIR
    let out_mir = run_cmd(&bin, &["--strict-mir", &script_str]);
    assert_eq!(String::from_utf8_lossy(&out_mir.stdout).trim(), "1");

    // 3. Cranelift JIT
    let out_jit = run_cmd(&bin, &["run", &script_str, "--backend", "cranelift"]);
    assert_eq!(String::from_utf8_lossy(&out_jit.stdout).trim(), "1");

    // 4. Zig AOT
    let aot_bin = temp_dir.path().join("asserts_aot");
    let aot_bin_str = aot_bin.to_string_lossy().to_string();
    run_cmd(&bin, &["build", &script_str, "--backend", "zig", "-o", &aot_bin_str]);
    let out_aot = run_cmd(&aot_bin, &[]);
    assert_eq!(String::from_utf8_lossy(&out_aot.stdout).trim(), "1");
}

#[test]
fn test_runtime_shift_mask_63_modulo_semantics_across_backends() {
    let bin = lust_bin();
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let script_file = temp_dir.path().join("shift_mask_63.lust");

    // Test that shift counts are strictly masked with & 63 on all 4 backends:
    // (1 << 63) == -9223372036854775808 (i64::MIN)
    // (3 << 62) == -4611686018427387904
    // (1 << 127) == (1 << (127 & 63)) == 1 << 63 == -9223372036854775808 (i64::MIN)
    // (1 << 64) == (1 << (64 & 63)) == 1 << 0 == 1
    // (1 << 65) == (1 << (65 & 63)) == 1 << 1 == 2
    // (-1 >> 64) == -1
    // (-16 >> 66) == (-16 >> 2) == -4
    let lust_source = r#"
assert((1 << 63) == -9223372036854775808)
assert((3 << 62) == -4611686018427387904)
assert((1 << 127) == -9223372036854775808)
assert((1 << 64) == 1)
assert((1 << 65) == 2)
assert((-1 >> 64) == -1)
assert((-16 >> 66) == -4)
println(1)
"#;
    fs::write(&script_file, lust_source).expect("write script");
    let script_str = script_file.to_string_lossy().to_string();

    // Verify VM
    let out_vm = run_cmd(&bin, &[&script_str]);
    assert_eq!(String::from_utf8_lossy(&out_vm.stdout).trim(), "1");

    // Verify Strict MIR
    let out_mir = run_cmd(&bin, &["--strict-mir", &script_str]);
    assert_eq!(String::from_utf8_lossy(&out_mir.stdout).trim(), "1");

    // Verify Cranelift JIT
    let out_jit = run_cmd(&bin, &["run", &script_str, "--backend", "cranelift"]);
    assert_eq!(String::from_utf8_lossy(&out_jit.stdout).trim(), "1");

    // Verify Zig AOT
    let aot_bin = temp_dir.path().join("shift_mask_aot");
    let aot_bin_str = aot_bin.to_string_lossy().to_string();
    run_cmd(&bin, &["build", &script_str, "--backend", "zig", "-o", &aot_bin_str]);
    let out_aot = run_cmd(&aot_bin, &[]);
    assert_eq!(String::from_utf8_lossy(&out_aot.stdout).trim(), "1");
}

#[test]
fn test_runtime_zig_aot_multi_arg_print_parity() {
    let bin = lust_bin();
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let script_file = temp_dir.path().join("multi_arg_print.lust");

    let lust_source = r#"
print("A", 1, 0x0F)
println("B", 2, 0b1010)
"#;
    fs::write(&script_file, lust_source).expect("write script");
    let script_str = script_file.to_string_lossy().to_string();

    let out_vm = run_cmd(&bin, &[&script_str]);
    let vm_output = String::from_utf8_lossy(&out_vm.stdout);

    let aot_bin = temp_dir.path().join("multi_arg_aot");
    let aot_bin_str = aot_bin.to_string_lossy().to_string();
    run_cmd(&bin, &["build", &script_str, "--backend", "zig", "-o", &aot_bin_str]);
    let out_aot = run_cmd(&aot_bin, &[]);
    let aot_output = String::from_utf8_lossy(&out_aot.stdout);

    assert_eq!(
        vm_output,
        aot_output,
        "Zig AOT multi-argument print/println must match VM behavior exactly"
    );
}

#[test]
fn test_runtime_zig_aot_nested_print_no_label_collision() {
    let bin = lust_bin();
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let script_file = temp_dir.path().join("nested_print.lust");

    let lust_source = r#"
println("A", println("B", "C"))
"#;
    fs::write(&script_file, lust_source).expect("write script");
    let script_str = script_file.to_string_lossy().to_string();

    let out_vm = run_cmd(&bin, &[&script_str]);
    let vm_output = String::from_utf8_lossy(&out_vm.stdout);

    let aot_bin = temp_dir.path().join("nested_print_aot");
    let aot_bin_str = aot_bin.to_string_lossy().to_string();
    run_cmd(&bin, &["build", &script_str, "--backend", "zig", "-o", &aot_bin_str]);
    let out_aot = run_cmd(&aot_bin, &[]);
    let aot_output = String::from_utf8_lossy(&out_aot.stdout);

    assert_eq!(
        vm_output,
        aot_output,
        "Zig AOT nested println must compile without label collision and match VM output"
    );
}

#[test]
fn test_runtime_zig_aot_string_pattern_matching() {
    let bin = lust_bin();
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let script_file = temp_dir.path().join("string_pattern.lust");

    let lust_source = r#"
function check(s: string): string
    if s is "hello" then
        return "matched_hello"
    elseif s is "world" then
        return "matched_world"
    else
        return "unmatched"
    end
end

println(check("hello"), check("world"), check("other"))
"#;
    fs::write(&script_file, lust_source).expect("write script");
    let script_str = script_file.to_string_lossy().to_string();

    let out_vm = run_cmd(&bin, &[&script_str]);
    let vm_output = String::from_utf8_lossy(&out_vm.stdout);

    let aot_bin = temp_dir.path().join("string_pattern_aot");
    let aot_bin_str = aot_bin.to_string_lossy().to_string();
    run_cmd(&bin, &["build", &script_str, "--backend", "zig", "-o", &aot_bin_str]);
    let out_aot = run_cmd(&aot_bin, &[]);
    let aot_output = String::from_utf8_lossy(&out_aot.stdout);

    assert_eq!(
        vm_output,
        aot_output,
        "Zig AOT string literal pattern matching (std.mem.eql) must match VM behavior"
    );
}

#[test]
fn test_runtime_print_side_effect_evaluation_order() {
    let bin = lust_bin();
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let script_file = temp_dir.path().join("side_effects.lust");

    let lust_source = r#"
function first(): string
    print("[first]")
    return "1"
end

function second(): string
    print("[second]")
    return "2"
end

println("RES:", first(), second())
"#;
    fs::write(&script_file, lust_source).expect("write script");
    let script_str = script_file.to_string_lossy().to_string();

    let out_vm = run_cmd(&bin, &[&script_str]);
    let vm_output = String::from_utf8_lossy(&out_vm.stdout);

    let aot_bin = temp_dir.path().join("side_effects_aot");
    let aot_bin_str = aot_bin.to_string_lossy().to_string();
    run_cmd(&bin, &["build", &script_str, "--backend", "zig", "-o", &aot_bin_str]);
    let out_aot = run_cmd(&aot_bin, &[]);
    let aot_output = String::from_utf8_lossy(&out_aot.stdout);

    assert_eq!(
        vm_output,
        aot_output,
        "Arguments to print/println must be evaluated strictly before printing outputs (no interleaving)"
    );
}

#[test]
fn test_cranelift_direct_jit_shifts_and_msb() {
    // Note on CLI vs Library API status:
    // In CLI runs (`lust run <file> --backend cranelift`), the `--backend cranelift` flag
    // delegates to the MIR Bytecode VM interpreter.
    // Pure Cranelift JIT compilation and native execution is tested directly via
    // the Rust library API (`CraneliftJitCompiler`).
    //
    // Golden fixture status:
    // `tests/fixtures/bitwise_golden.lust` uses `println(<expr>)` for each of the 10
    // canonical bitwise expressions, serving as the minimal common denominator for all 4 backends.
    use lust::{
        ast::ItemKind,
        intern::Interner,
        lexer::Lexer,
        mir::{
            MirModule,
            cranelift_backend::jit_engine::CraneliftJitCompiler,
            lowering::AstLowering,
        },
        parser::Parser,
    };

    let source = r#"
function test_shl_63(): int
    return 1 << 63
end

function test_shl_64(): int
    return 1 << 64
end

function test_shr_63(): int
    return 0x8000_0000_0000_0000 >> 63
end

function test_shl_127(): int
    return 1 << 127
end
"#;

    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("tokenization should succeed");
    let items = Parser::new(tokens)
        .parse()
        .expect("parsing should succeed");

    let mut module = MirModule::new("test_cranelift_bitwise".into());
    for item in items {
        if let ItemKind::Function(func_def) = item.kind {
            let mir_fn = AstLowering::lower_function(&func_def).expect("lower function to MIR");
            module.add_function(mir_fn);
        }
    }

    let mut jit = CraneliftJitCompiler::new().expect("Cranelift JIT compiler initialization");
    let func_ptrs = jit.compile_module(&module).expect("Cranelift JIT compilation");

    let fn_shl_63 = *func_ptrs.get("test_shl_63").expect("test_shl_63 found");
    let native_shl_63: extern "C" fn() -> i64 = unsafe { core::mem::transmute(fn_shl_63) };
    assert_eq!(
        native_shl_63(),
        -9223372036854775808,
        "1 << 63 must equal i64::MIN (-9223372036854775808) in Cranelift JIT"
    );

    let fn_shl_64 = *func_ptrs.get("test_shl_64").expect("test_shl_64 found");
    let native_shl_64: extern "C" fn() -> i64 = unsafe { core::mem::transmute(fn_shl_64) };
    assert_eq!(
        native_shl_64(),
        1,
        "1 << 64 must equal 1 (masked with & 63) in Cranelift JIT"
    );

    let fn_shr_63 = *func_ptrs.get("test_shr_63").expect("test_shr_63 found");
    let native_shr_63: extern "C" fn() -> i64 = unsafe { core::mem::transmute(fn_shr_63) };
    assert_eq!(
        native_shr_63(),
        -1,
        "0x8000_0000_0000_0000 >> 63 must equal -1 (arithmetic shift right) in Cranelift JIT"
    );

    let fn_shl_127 = *func_ptrs.get("test_shl_127").expect("test_shl_127 found");
    let native_shl_127: extern "C" fn() -> i64 = unsafe { core::mem::transmute(fn_shl_127) };
    assert_eq!(
        native_shl_127(),
        -9223372036854775808,
        "1 << 127 must equal i64::MIN (127 & 63 == 63) in Cranelift JIT"
    );
}

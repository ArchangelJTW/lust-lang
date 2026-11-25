//! Live RV32 (`riscv32imafc-unknown-none-elf` + `qemu-system-riscv32`)
//! execution & deterministic replay parity test.
//!
//! Verifies:
//! 1. Real `#![no_std]` `.rs` kernel emission from Lust source (`AotCompiler` + `RustCompileOptions`)
//!    with `StaticSlabAllocator<65536>` preamble, zero `unsafe`, zero `panic!`, zero `.unwrap(`.
//! 2. Offline `cargo build --offline --target riscv32imafc-unknown-none-elf` linking via `rust-lld`.
//! 3. Live QEMU RV32 (`-machine virt -bios none -nographic -kernel`) execution via 16550 UART
//!    (`0x1000_0000`) and SiFive test finisher (`0x10_0000` `0x5555`):
//!    a) Deterministic Lust `#![no_std]` computation (`dot_sum(10) == 440`) + parity vs host Lust VM.
//!    b) `LustBudgetAlloc` and `StaticSlabAllocator` OOM budget enforcement without UB.
//!    c) Gas exhaustion (`AotError::OutOfGas`) enforcement on `riscv32imafc-unknown-none-elf`.
//!    d) Deterministic `CanonicalTraceCodec` hash & `final_state` parity vs host
//!       and across two independent `qemu-system-riscv32` runs.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use tempfile::tempdir;

use lust::bytecode::Compiler;
use lust::codegen_zig::aot::rust_emitter::{GasMetering, RustInteropOptions};
use lust::codegen_zig::aot::{
    AotCompileOptions, AotCompiler, AotTargetKind, RustCompileOptions, prepare_items_for_mir,
};
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::lowering::AstLowering;
use lust::no_std::{
    CanonicalTraceCodec, DeterministicScheduler, ExternalInput, FinalOutcome, SimTaskSpec,
    TargetInfo,
};
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::VM;
use lust::Value;

const QEMU_RV32_BIN: &str = "/opt/homebrew/bin/qemu-system-riscv32";
const QEMU_TIMEOUT: Duration = Duration::from_secs(15);

fn rv32_lust_source() -> &'static str {
    r#"
function dot_sum(n: int): int
    local acc: int = 0
    local i: int = 1
    while i <= n do
        acc = acc + i * (i + 1)
        i = i + 1
    end
    return acc
end

function main(): int
    return dot_sum(10)
end
"#
}

fn eval_host_lust_vm(source: &str) -> i64 {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("tokenize host VM source");
    let items = Parser::new(tokens).parse().expect("parse host VM source");
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    let fns = compiler
        .compile_module(&items)
        .expect("compile host VM bytecode");

    let mut vm = VM::new();
    vm.load_functions(fns);
    let res = vm
        .call("main", vec![])
        .expect("execute main() in host Lust VM");
    match res {
        Value::Int(v) => v,
        other => panic!("expected Value::Int from host Lust VM, got {other:?}"),
    }
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01B3);
    }
    h
}

fn run_qemu_with_watchdog(elf_path: &Path) -> (i32, String, String) {
    let mut child = Command::new(QEMU_RV32_BIN)
        .arg("-machine")
        .arg("virt")
        .arg("-bios")
        .arg("none")
        .arg("-nographic")
        .arg("-kernel")
        .arg(elf_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn qemu-system-riscv32");

    let start = Instant::now();
    let status = loop {
        match child.try_wait().expect("poll qemu-system-riscv32") {
            Some(st) => break st,
            None => {
                if start.elapsed() > QEMU_TIMEOUT {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!(
                        "qemu-system-riscv32 timed out after {:?} for {}",
                        QEMU_TIMEOUT,
                        elf_path.display()
                    );
                }
                std::thread::sleep(Duration::from_millis(25));
            }
        }
    };

    let mut stdout = String::new();
    if let Some(mut out) = child.stdout.take() {
        let _ = out.read_to_string(&mut stdout);
    }
    let mut stderr = String::new();
    if let Some(mut err) = child.stderr.take() {
        let _ = err.read_to_string(&mut stderr);
    }

    let code = status.code().unwrap_or(-1);
    assert!(
        status.success(),
        "qemu-system-riscv32 exited with {code}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    (code, stdout, stderr)
}

#[test]
fn test_w18_rv32_live_qemu_execution_and_replay_parity() {
    assert!(
        Path::new(QEMU_RV32_BIN).exists(),
        "qemu-system-riscv32 not found at {QEMU_RV32_BIN}"
    );

    let source = rv32_lust_source();

    // 1. Host Lust VM baseline
    let host_vm_val = eval_host_lust_vm(source);
    assert_eq!(host_vm_val, 440, "host Lust VM dot_sum(10) must equal 440");

    // 2. Emit `#![no_std]` Rust AOT kernel from Lust source
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize().expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");
    let prepared = prepare_items_for_mir(&items);
    let mir_module = AstLowering::lower_module_items("w18_rv32_kernel", &prepared)
        .expect("lower MIR");

    let opts = RustCompileOptions {
        aot: AotCompileOptions {
            target_kind: AotTargetKind::StaticLibrary,
            release_mode: "ReleaseSmall".to_string(),
            strip: false,
            max_memory_bytes: Some(4096),
            target_triple: Some("riscv32imafc-unknown-none-elf".to_string()),
        },
        gas_metering: GasMetering::Exact,
        gas_limit: Some(10_000),
        pure_core: false,
        no_std: true,
        cache_dir: None,
    };

    // 2a. Verify standalone `#![no_std]` emission has StaticSlabAllocator<65536> & 0 unsafe/panic
    let standalone_nostd = AotCompiler::emit_rust_module_with_interop(
        &mir_module,
        &items,
        &opts,
        &RustInteropOptions {
            linked_crates: Vec::new(),
            emit_as_include_module: false,
            export_public_api: true,
        },
    )
    .expect("emit standalone #![no_std] .rs");
    assert!(standalone_nostd.contains("#![no_std]"));
    assert!(standalone_nostd.contains("StaticSlabAllocator<65536>"));
    assert!(!standalone_nostd.contains("unsafe"));
    assert!(!standalone_nostd.contains("panic!("));
    assert!(!standalone_nostd.contains(".unwrap("));

    // 2b. Emit include!() module for the RV32 firmware crate
    let include_kernel = AotCompiler::emit_rust_include_module_from_items(
        &mir_module,
        &items,
        &opts,
        &[],
    )
    .expect("emit include #![no_std] kernel");
    assert!(include_kernel.contains("pub fn dot_sum_with_ctx"));
    assert!(!include_kernel.contains("unsafe"));
    assert!(!include_kernel.contains("panic!("));
    assert!(!include_kernel.contains(".unwrap("));

    let tmp = tempdir().expect("tempdir");
    let kernel_path = tmp.path().join("lust_kernel.rs");
    fs::write(&kernel_path, &include_kernel).expect("write emitted kernel");

    // 3. Build `riscv32imafc-unknown-none-elf` ELF binary offline
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture_dir = manifest_dir.join("tests/fixtures/w18_rv32_qemu");
    let rv32_target_dir = tmp.path().join("rv32_target");

    let build_out = Command::new("cargo")
        .current_dir(&fixture_dir)
        .env("LUST_RV32_KERNEL_RS", &kernel_path)
        .arg("build")
        .arg("--offline")
        .arg("--target")
        .arg("riscv32imafc-unknown-none-elf")
        .arg("--target-dir")
        .arg(&rv32_target_dir)
        .output()
        .expect("run cargo build riscv32imafc-unknown-none-elf");
    assert!(
        build_out.status.success(),
        "riscv32imafc-unknown-none-elf firmware build failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&build_out.stdout),
        String::from_utf8_lossy(&build_out.stderr)
    );

    let elf_path = rv32_target_dir
        .join("riscv32imafc-unknown-none-elf")
        .join("debug")
        .join("w18_rv32_qemu");
    assert!(elf_path.exists(), "RV32 ELF missing at {}", elf_path.display());

    // 4. Execute twice independently in QEMU RV32 (`virt`)
    let (code1, stdout1, stderr1) = run_qemu_with_watchdog(&elf_path);
    let (code2, stdout2, stderr2) = run_qemu_with_watchdog(&elf_path);
    assert_eq!(code1, 0);
    assert_eq!(code2, 0);
    assert_eq!(
        stdout1, stdout2,
        "two independent QEMU RV32 runs must produce byte-exact identical stdout"
    );
    assert_eq!(
        stderr1, stderr2,
        "two independent QEMU RV32 runs must produce byte-exact identical stderr"
    );
    let uart_out = format!("{stdout1}{stderr1}");

    // 5. Host DST trace parity check
    let tasks = vec![
        SimTaskSpec {
            task_id: 1,
            steps: 3,
            delta_per_step: 10,
        },
        SimTaskSpec {
            task_id: 2,
            steps: 3,
            delta_per_step: 5,
        },
        SimTaskSpec {
            task_id: 3,
            steps: 2,
            delta_per_step: 7,
        },
    ];
    let sched = DeterministicScheduler {
        seed: 42,
        run_id: "rv32-dst-t3".to_string(),
        target: TargetInfo {
            triple: "riscv32imafc-unknown-none-elf".to_string(),
            endianness: "little".to_string(),
            toolchain: "rustc-1.100.0".to_string(),
            candidate_digest: "w18-rv32".to_string(),
            fixture_hash: "rv32-sample-v1".to_string(),
        },
        external_inputs: vec![ExternalInput {
            name: "sim_tasks".to_string(),
            content_hash: "sha256:tasks-v1".to_string(),
            payload_len: 3,
        }],
        fault_at_event: None,
    };
    let host_trace = sched.run(&tasks);
    let host_bytes = CanonicalTraceCodec::encode_canonical(&host_trace);
    let host_fnv64 = fnv1a_64(&host_bytes);
    let host_final_val = match host_trace.final_result {
        FinalOutcome::OkValue(v) => v,
        FinalOutcome::StableError(ref e) => panic!("unexpected error: {e}"),
    };

    let expected_line = format!(
        "RV32-DST:OK|dot_sum={host_vm_val}|budget_oom=1|slab_oom=1|gas_oom=1|trace_len={}|trace_fnv64={host_fnv64:016x}|final_val={host_final_val}|final_state={}",
        host_bytes.len(),
        host_trace.final_state
    );
    assert_eq!(
        uart_out.trim(),
        expected_line,
        "QEMU RV32 UART output must match host VM + CanonicalTraceCodec parity"
    );
    println!("QEMU_RV32_VERIFIED: {}", uart_out.trim());
}

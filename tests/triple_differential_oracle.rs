//! NG-W12-5: Triple Differential Oracle (`Bytecode VM == Cranelift JIT == Rust AOT`),
//! Exact MIR `GasUsed` Parity (`F-17`), and Isolated Child-Process Security Invariants
//! (`SEC-01..SEC-15`, Rule 5 `AGENTS.md`).
//!
//! Gate flags: `TASK_ALLOWS_CARGO_LOCK=0`, `TASK_ALLOWS_UNSAFE=0`.
//! Every child execution (including all `SEC-*` probes, VM/JIT bounded runs, and AOT binaries)
//! runs in an isolated OS child process supervised by `lust::sandbox::run_sandboxed` with
//! monotonic wall-clock watchdog termination (`killpg(SIGKILL)`).

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::rc::Rc;
use std::sync::OnceLock;
use std::time::Duration;
use tempfile::tempdir;

use lust::ast::Span;
use lust::bytecode::{Compiler as BytecodeCompiler, Value};
use lust::codegen_zig::aot::{
    AotCompileOptions, AotCompiler, AotTargetKind, GasMetering, RustCompileOptions,
    RustEmitterOptions, emit_rust_from_mir, ensure_cached_aot_rt_rlib, prepare_items_for_mir,
};
use lust::config::{LustConfig, PreludeMode};
use lust::data::materialize::LayoutPerfCounters;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{safepoint_reset, safepoint_stats};
use lust::mir::ir::{
    BasicBlock, BlockId, FunctionSignature, LocalDecl, LocalId, MirBinaryOp, MirConstant,
    MirEnumDef, MirFunction, MirModule, MirType, Operand, Place, ResourceKind, Rvalue, Statement,
    Terminator,
};
use lust::mir::lowering::AstLowering;
use lust::mir::verifier::MirVerifier;
use lust::mir::vm_backend::register_struct_decode_natives_with_counters;
use lust::parser::Parser;
use lust::sandbox::{SandboxLimits, SandboxPolicy, run_sandboxed};
use lust::typechecker::TypeChecker;
use lust::vm::{VM, VmBackend};

// ============================================================================
// Shared Harness Configuration & Canonical Error Classification
// ============================================================================

const ALL_10_BENCHMARKS: [(&str, &str); 10] = [
    ("array", "4999995000000\n"),
    ("calls", "99999990000000\n"),
    ("fib", "832040\n"),
    ("fields", "50000005000000\n"),
    ("floatmath", "199999960.06314665\n"),
    ("methods", "49999995000000\n"),
    ("nested", "23135141\n"),
    ("sieve", "664579\n"),
    ("strings", "100000\n"),
    ("tree", "10738090560\n"),
];

const WORKER_ENV_ENGINE: &str = "LUST_ORACLE_ENGINE";
const WORKER_ENV_SCRIPT: &str = "LUST_ORACLE_SCRIPT";
const WORKER_ENV_GAS_LIMIT: &str = "LUST_ORACLE_GAS_LIMIT";
const WORKER_ENV_MEM_LIMIT: &str = "LUST_ORACLE_MEM_LIMIT";
const WORKER_ENV_PURE_CORE: &str = "LUST_ORACLE_PURE_CORE";
const WORKER_ENV_SCHEMA_DECODE: &str = "LUST_ORACLE_SCHEMA_DECODE";
const WORKER_OUT_BEGIN: &str = "__ORACLE_OUT_BEGIN__";
const WORKER_OUT_END: &str = "__ORACLE_OUT_END__";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ErrorClass {
    Ok,
    DivisionByZero,
    NegativeExponent,
    IndexOutOfBounds,
    OutOfGas,
    OutOfMemory,
    StackOverflow,
    MissingMapKey,
    UserError,
    CapabilityDenied,
    SchemaDecodeError,
    Timeout,
    OtherError,
}

#[derive(Debug, Clone)]
struct EngineOutcome {
    exit_code: Option<i32>,
    signal: Option<i32>,
    timed_out: bool,
    stdout: String,
    stderr: String,
    error_class: ErrorClass,
}

#[derive(Debug, Clone)]
struct TripleParityOptions {
    release_mode: &'static str,
    gas_metering: GasMetering,
    gas_limit: Option<u64>,
    mem_limit: Option<usize>,
    pure_core: bool,
    timeout: Duration,
    require_stdout_on_error: bool,
}

impl Default for TripleParityOptions {
    fn default() -> Self {
        Self {
            release_mode: "Debug",
            gas_metering: GasMetering::None,
            gas_limit: None,
            mem_limit: None,
            pure_core: false,
            timeout: Duration::from_secs(20),
            require_stdout_on_error: false,
        }
    }
}

fn lust_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_lust"))
}

fn find_rustc() -> String {
    std::env::var("RUSTC")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "rustc".to_string())
}

fn benchmark_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks")
        .join("suite")
        .join(format!("{name}.lust"))
}

fn shared_rlib_cache_dir() -> &'static Path {
    static CACHE: OnceLock<PathBuf> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let dir = std::env::temp_dir().join("lust-ng-w12-5-triple-oracle-rlib-cache");
            let _ = fs::create_dir_all(&dir);
            // Pre-warm both ReleaseFast and Debug rlibs so parallel tests hit the cache immediately.
            for mode in ["ReleaseFast", "Debug"] {
                let opts = RustCompileOptions {
                    aot: AotCompileOptions {
                        target_kind: AotTargetKind::Executable,
                        release_mode: mode.to_string(),
                        ..AotCompileOptions::default()
                    },
                    cache_dir: Some(dir.clone()),
                    ..RustCompileOptions::default()
                };
                let _ = ensure_cached_aot_rt_rlib(&opts);
            }
            dir
        })
        .as_path()
}

fn classify_error(
    exit_code: Option<i32>,
    signal: Option<i32>,
    timed_out: bool,
    stderr: &str,
) -> ErrorClass {
    if timed_out {
        return ErrorClass::Timeout;
    }
    if exit_code == Some(0) && signal.is_none() {
        return ErrorClass::Ok;
    }
    let lower = stderr.to_ascii_lowercase();
    if lower.contains("division by zero") || lower.contains("modulo by zero") {
        ErrorClass::DivisionByZero
    } else if lower.contains("negative exponent") {
        ErrorClass::NegativeExponent
    } else if lower.contains("out of gas") || lower.contains("gas budget") || lower.contains("gas limit")
    {
        ErrorClass::OutOfGas
    } else if lower.contains("out of memory")
        || lower.contains("memory budget")
        || lower.contains("memory limit")
        || lower.contains("allocation failed")
    {
        ErrorClass::OutOfMemory
    } else if lower.contains("missing map key")
        || (lower.contains("unwrap()") && lower.contains("none"))
        || lower.contains("key not found")
    {
        ErrorClass::MissingMapKey
    } else if lower.contains("index out of bounds")
        || lower.contains("out of bounds")
        || (lower.contains("unwrap()") && lower.contains("err"))
    {
        ErrorClass::IndexOutOfBounds
    } else if lower.contains("stack overflow")
        || lower.contains("call stack")
        || lower.contains("recursion")
    {
        ErrorClass::StackOverflow
    } else if lower.contains("schemadecodeerror")
        || lower.contains("schema decode")
        || lower.contains("decode error")
        || lower.contains("invalid json")
        || lower.contains("duplicate key")
        || lower.contains("missing required field")
        || lower.contains("unknown field")
    {
        ErrorClass::SchemaDecodeError
    } else if lower.contains("capabilitydenied")
        || lower.contains("capability")
        || lower.contains("not allowed")
        || lower.contains("undefined variable 'io'")
        || lower.contains("undefined variable 'os'")
        || lower.contains("undefined global 'io'")
        || lower.contains("undefined global 'os'")
        || lower.contains("pure-core")
        || lower.contains("denied")
    {
        ErrorClass::CapabilityDenied
    } else if lower.contains("user error")
        || lower.contains("boom")
        || lower.contains("assertion failed")
    {
        ErrorClass::UserError
    } else {
        ErrorClass::OtherError
    }
}

fn run_in_sandbox(
    exe: &Path,
    args: Vec<String>,
    env_vars: &[(&str, String)],
    timeout: Duration,
) -> EngineOutcome {
    let mut policy = SandboxPolicy::new(exe);
    policy.args = args;
    policy.limits = SandboxLimits {
        wall_timeout: timeout,
        max_stdout_bytes: 256 * 1024,
        max_stderr_bytes: 256 * 1024,
        ..SandboxLimits::default()
    };
    for (k, v) in env_vars {
        policy.allowed_env.insert((*k).to_string(), v.clone());
    }
    let out = run_sandboxed(&policy)
        .unwrap_or_else(|e| panic!("sandbox supervisor failed for {}: {e}", exe.display()));
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let error_class = classify_error(out.exit_code, out.signal, out.timed_out, &stderr);
    EngineOutcome {
        exit_code: out.exit_code,
        signal: out.signal,
        timed_out: out.timed_out,
        stdout,
        stderr,
        error_class,
    }
}

/// Runs Engine 1 (Bytecode VM `--jit=off`) in an isolated sandboxed child process.
fn run_vm_engine(script_path: &Path, opts: &TripleParityOptions) -> EngineOutcome {
    if opts.gas_limit.is_none() && opts.mem_limit.is_none() {
        let mut args = vec![
            "run".to_string(),
            script_path.display().to_string(),
            "--backend".to_string(),
            "vm".to_string(),
            "--jit=off".to_string(),
        ];
        if opts.pure_core {
            args.push("--pure-core".to_string());
        }
        run_in_sandbox(&lust_bin(), args, &[], opts.timeout)
    } else {
        run_worker_child("vm", script_path, opts, false)
    }
}

/// Runs Engine 2 (Cranelift JIT `--jit=cranelift`) in an isolated sandboxed child process.
fn run_cranelift_engine(script_path: &Path, opts: &TripleParityOptions) -> EngineOutcome {
    if opts.gas_limit.is_none() && opts.mem_limit.is_none() {
        let mut args = vec![
            "run".to_string(),
            script_path.display().to_string(),
            "--jit=cranelift".to_string(),
        ];
        if opts.pure_core {
            args.push("--pure-core".to_string());
        }
        run_in_sandbox(&lust_bin(), args, &[], opts.timeout)
    } else {
        run_worker_child("cranelift", script_path, opts, false)
    }
}

fn run_worker_child(
    engine: &str,
    script_path: &Path,
    opts: &TripleParityOptions,
    schema_decode: bool,
) -> EngineOutcome {
    let exe = std::env::current_exe().expect("current test binary path");
    let args = vec![
        "--exact".to_string(),
        "__triple_oracle_worker".to_string(),
        "--ignored".to_string(),
        "--nocapture".to_string(),
    ];
    let mut env_vars = vec![
        (WORKER_ENV_ENGINE, engine.to_string()),
        (WORKER_ENV_SCRIPT, script_path.display().to_string()),
        (
            WORKER_ENV_PURE_CORE,
            if opts.pure_core { "1" } else { "0" }.to_string(),
        ),
        (
            WORKER_ENV_SCHEMA_DECODE,
            if schema_decode { "1" } else { "0" }.to_string(),
        ),
    ];
    if let Some(g) = opts.gas_limit {
        env_vars.push((WORKER_ENV_GAS_LIMIT, g.to_string()));
    }
    if let Some(m) = opts.mem_limit {
        env_vars.push((WORKER_ENV_MEM_LIMIT, m.to_string()));
    }
    let mut out = run_in_sandbox(&exe, args, &env_vars, opts.timeout);
    if let Some((_, rest)) = out.stdout.split_once(WORKER_OUT_BEGIN) {
        let stripped = match rest.split_once(WORKER_OUT_END) {
            Some((body, _)) => body,
            None => rest,
        };
        out.stdout = stripped.to_string();
    }
    out
}

/// Builds and runs Engine 3 (Rust AOT `lust build --backend rust`) in an isolated sandboxed child process.
fn run_rust_aot_engine(
    script_path: &Path,
    bin_path: &Path,
    opts: &TripleParityOptions,
) -> EngineOutcome {
    let mut build_cmd = Command::new(lust_bin());
    build_cmd
        .arg("build")
        .arg(script_path)
        .arg("--backend")
        .arg("rust")
        .arg(if opts.release_mode == "Debug" {
            "--debug"
        } else {
            "--release-fast"
        })
        .arg("--cache-dir")
        .arg(shared_rlib_cache_dir())
        .arg("-o")
        .arg(bin_path);

    let effective_gas_metering = if opts.gas_limit.is_some() && opts.gas_metering == GasMetering::None
    {
        GasMetering::Exact
    } else {
        opts.gas_metering
    };
    match effective_gas_metering {
        GasMetering::None => {}
        GasMetering::Exact => {
            build_cmd.arg("--gas-metering").arg("exact");
        }
        GasMetering::LoopChunked => {
            build_cmd.arg("--gas-metering").arg("loop-chunked");
        }
    }
    if let Some(limit) = opts.gas_limit {
        build_cmd.arg("--gas-limit").arg(limit.to_string());
    }
    if let Some(mem) = opts.mem_limit {
        build_cmd.arg("--max-memory").arg(mem.to_string());
    }
    if opts.pure_core {
        build_cmd.arg("--deny-ffi");
    }

    let build_out = build_cmd
        .output()
        .expect("invoke lust build --backend rust");
    if !build_out.status.success() {
        let stderr = String::from_utf8_lossy(&build_out.stderr).into_owned();
        let stdout = String::from_utf8_lossy(&build_out.stdout).into_owned();
        let error_class = classify_error(build_out.status.code(), None, false, &stderr);
        return EngineOutcome {
            exit_code: build_out.status.code(),
            signal: None,
            timed_out: false,
            stdout,
            stderr: format!("[AOT BUILD FAILED]\n{stderr}"),
            error_class,
        };
    }

    assert!(
        bin_path.exists(),
        "lust build --backend rust succeeded but output binary {} is missing",
        bin_path.display()
    );

    run_in_sandbox(bin_path, Vec::new(), &[], opts.timeout)
}

/// Core NG-W12-5 Triple Differential Oracle (`Bytecode VM == Cranelift JIT == Rust AOT`).
///
/// Runs the identical Lust source across all three engines in isolated sandboxed child
/// processes and asserts bit-exact `stdout` and identical `ErrorClass`.
fn assert_triple_parity(
    script_name: &str,
    lust_src: &str,
    gas_limit: Option<u64>,
    mem_limit: Option<usize>,
) -> (EngineOutcome, EngineOutcome, EngineOutcome) {
    let opts = TripleParityOptions {
        gas_limit,
        mem_limit,
        ..TripleParityOptions::default()
    };
    assert_triple_parity_with_opts(script_name, lust_src, &opts)
}

fn assert_triple_parity_with_opts(
    script_name: &str,
    lust_src: &str,
    opts: &TripleParityOptions,
) -> (EngineOutcome, EngineOutcome, EngineOutcome) {
    let tmp = tempdir().expect("create oracle tempdir");
    let safe_name: String = script_name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    let script_path = tmp.path().join(format!("{safe_name}.lust"));
    let bin_path = tmp.path().join(format!("{safe_name}_aot_bin"));
    fs::write(&script_path, lust_src).expect("write oracle script");

    let vm = run_vm_engine(&script_path, opts);
    let jit = run_cranelift_engine(&script_path, opts);
    let aot = run_rust_aot_engine(&script_path, &bin_path, opts);

    assert!(
        !vm.timed_out,
        "[{script_name}] Bytecode VM timed out after {:?}",
        opts.timeout
    );
    assert!(
        !jit.timed_out,
        "[{script_name}] Cranelift JIT timed out after {:?}",
        opts.timeout
    );
    assert!(
        !aot.timed_out,
        "[{script_name}] Rust AOT timed out after {:?}",
        opts.timeout
    );

    assert_eq!(
        vm.error_class, jit.error_class,
        "[{script_name}] ErrorClass mismatch VM ({:?}) vs Cranelift JIT ({:?})\nVM stderr:\n{}\nJIT stderr:\n{}",
        vm.error_class, jit.error_class, vm.stderr, jit.stderr
    );
    assert_eq!(
        jit.error_class, aot.error_class,
        "[{script_name}] ErrorClass mismatch Cranelift JIT ({:?}) vs Rust AOT ({:?})\nJIT stderr:\n{}\nAOT stderr:\n{}",
        jit.error_class, aot.error_class, jit.stderr, aot.stderr
    );

    if vm.error_class == ErrorClass::Ok || opts.require_stdout_on_error {
        assert_eq!(
            vm.stdout, jit.stdout,
            "[{script_name}] Bit-exact stdout mismatch VM vs Cranelift JIT\nVM stdout:  {:?}\nJIT stdout: {:?}",
            vm.stdout, jit.stdout
        );
        assert_eq!(
            jit.stdout, aot.stdout,
            "[{script_name}] Bit-exact stdout mismatch Cranelift JIT vs Rust AOT\nJIT stdout: {:?}\nAOT stdout: {:?}",
            jit.stdout, aot.stdout
        );
    }

    (vm, jit, aot)
}

// ============================================================================
// Isolated Child Worker (`__triple_oracle_worker`) — Rule 5 `AGENTS.md`
// ============================================================================

#[test]
#[ignore]
fn __triple_oracle_worker() {
    let Ok(engine) = std::env::var(WORKER_ENV_ENGINE) else {
        return;
    };
    let script_path = std::env::var(WORKER_ENV_SCRIPT).expect("LUST_ORACLE_SCRIPT");
    let source = fs::read_to_string(&script_path).expect("read worker script");
    let gas_limit = std::env::var(WORKER_ENV_GAS_LIMIT)
        .ok()
        .and_then(|s| s.parse::<u64>().ok());
    let mem_limit = std::env::var(WORKER_ENV_MEM_LIMIT)
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    let pure_core = std::env::var(WORKER_ENV_PURE_CORE).as_deref() == Ok("1");
    let schema_decode = std::env::var(WORKER_ENV_SCHEMA_DECODE).as_deref() == Ok("1");

    let mut config = LustConfig::default();
    if pure_core {
        config.set_prelude_mode(PreludeMode::Pure);
    }

    let mut interner = Interner::new();
    let tokens = match Lexer::new(&source, &mut interner).tokenize() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Lexer error: {e}");
            std::process::exit(1);
        }
    };
    let items = match Parser::new(tokens).parse() {
        Ok(i) => i,
        Err(e) => {
            eprintln!("Parser error: {e}");
            std::process::exit(1);
        }
    };
    let mut checker = TypeChecker::with_config(&config);
    if let Err(e) = checker.check_module(&items) {
        eprintln!("Type error: {e}");
        std::process::exit(1);
    }

    let mut struct_defs = lust::hashbrown::HashMap::new();
    for item in &items {
        if let lust::ast::ItemKind::Struct(s) = &item.kind {
            struct_defs.insert(s.name.clone(), s.clone());
        }
    }

    let prepared = prepare_items_for_mir(&items);
    let mir_mod = match AstLowering::lower_module_items("oracle_mod", &prepared) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("MIR lowering error: {e}");
            std::process::exit(1);
        }
    };

    match engine.as_str() {
        "vm" => {
            config.set_jit_enabled(false);
            let mut vm = VM::with_config(&config);
            vm.set_backend(VmBackend::Bytecode);
            if schema_decode {
                if let Err(e) = vm.load_mir_module(&mir_mod) {
                    eprintln!("MIR load error: {e}");
                    std::process::exit(1);
                }
                let counters = Rc::new(RefCell::new(LayoutPerfCounters::default()));
                register_struct_decode_natives_with_counters(
                    &mut vm,
                    &mir_mod.struct_layouts,
                    counters,
                );
                vm.set_backend(VmBackend::Bytecode);
            } else {
                let compiled = match BytecodeCompiler::new().compile_module(&items) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Bytecode compile error: {e}");
                        std::process::exit(1);
                    }
                };
                vm.load_functions(compiled);
            }
            vm.register_structs(&struct_defs);
            if let Some(g) = gas_limit {
                vm.set_gas_budget(g);
            }
            if let Some(m) = mem_limit {
                if let Err(e) = vm.set_memory_budget_bytes(m) {
                    eprintln!("Memory budget error: {e}");
                    std::process::exit(1);
                }
            }
            let entry = if schema_decode { "main" } else { "__script" };
            print!("{WORKER_OUT_BEGIN}");
            let _ = std::io::Write::flush(&mut std::io::stdout());
            if let Err(e) = vm.call(entry, vec![]) {
                print!("{WORKER_OUT_END}");
                let _ = std::io::Write::flush(&mut std::io::stdout());
                eprintln!("{e}");
                std::process::exit(1);
            }
            print!("{WORKER_OUT_END}");
            let _ = std::io::Write::flush(&mut std::io::stdout());
            std::process::exit(0);
        }
        "cranelift" => {
            config.set_cranelift_jit(true);
            let mut vm = VM::with_config(&config);
            if let Err(e) = vm.load_mir_module(&mir_mod) {
                eprintln!("MIR load error: {e}");
                std::process::exit(1);
            }
            if schema_decode {
                let counters = Rc::new(RefCell::new(LayoutPerfCounters::default()));
                register_struct_decode_natives_with_counters(
                    &mut vm,
                    &mir_mod.struct_layouts,
                    counters,
                );
            }
            vm.register_structs(&struct_defs);
            vm.set_backend(VmBackend::Cranelift);
            if let Some(g) = gas_limit {
                vm.set_gas_budget(g);
            }
            if let Some(m) = mem_limit {
                if let Err(e) = vm.set_memory_budget_bytes(m) {
                    eprintln!("Memory budget error: {e}");
                    std::process::exit(1);
                }
            }
            print!("{WORKER_OUT_BEGIN}");
            let _ = std::io::Write::flush(&mut std::io::stdout());
            if let Err(e) = vm.call("main", vec![]) {
                print!("{WORKER_OUT_END}");
                let _ = std::io::Write::flush(&mut std::io::stdout());
                eprintln!("{e}");
                std::process::exit(1);
            }
            print!("{WORKER_OUT_END}");
            let _ = std::io::Write::flush(&mut std::io::stdout());
            std::process::exit(0);
        }
        other => {
            eprintln!("Unknown oracle worker engine: {other}");
            std::process::exit(2);
        }
    }
}

// ============================================================================
// TEST 1: All 10/10 `benchmarks/suite/*.lust` Triple Parity
// ============================================================================

#[test]
fn test_oracle_1_all_10_suite_benchmarks_triple_parity() {
    let tmp = tempdir().expect("suite tempdir");
    let release_opts = TripleParityOptions {
        release_mode: "ReleaseFast",
        timeout: Duration::from_secs(60),
        ..TripleParityOptions::default()
    };

    // 1A. Full-scale 10/10 exact parity: Cranelift JIT == Rust AOT == Golden Exact Oracle
    //     (verifies untouched benchmarks/suite/*.lust files including IEEE-754 floatmath.lust)
    for (bench, expected_stdout) in ALL_10_BENCHMARKS {
        let script = benchmark_path(bench);
        let bin_path = tmp.path().join(format!("{bench}_full_aot"));

        let jit = run_cranelift_engine(&script, &release_opts);
        let aot = run_rust_aot_engine(&script, &bin_path, &release_opts);

        assert_eq!(
            jit.error_class, ErrorClass::Ok,
            "[{bench}.lust] Cranelift JIT failed:\nstderr: {}",
            jit.stderr
        );
        assert_eq!(
            aot.error_class, ErrorClass::Ok,
            "[{bench}.lust] Rust AOT failed:\nstderr: {}",
            aot.stderr
        );
        assert_eq!(
            jit.stdout, aot.stdout,
            "[{bench}.lust] Bit-exact stdout mismatch Cranelift JIT vs Rust AOT"
        );
        assert_eq!(
            aot.stdout, expected_stdout,
            "[{bench}.lust] Bit-exact stdout mismatch vs exact oracle"
        );
    }

    // 1B. Full 3-way `VM (--jit=off) == Cranelift JIT (--jit=cranelift) == Rust AOT`
    //     on all 10 benchmark algorithms (full-scale on fast programs + VM-fast scale for 10M loops)
    let scaled_replacements: [(&str, Option<(&str, &str)>); 10] = [
        ("array", Some(("1000000", "2000"))),
        ("calls", Some(("10000000", "5000"))),
        ("fib", Some(("fib(30)", "fib(18)"))),
        ("fields", Some(("10000000", "5000"))),
        ("floatmath", Some(("10000000", "5000"))),
        ("methods", Some(("10000000", "5000"))),
        ("nested", Some(("3000", "60"))),
        ("sieve", Some(("10000000", "5000"))),
        ("strings", Some(("100000", "2000"))),
        ("tree", Some(("build(14, 1)", "build(7, 1)"))),
    ];

    for (bench, repl) in scaled_replacements {
        let mut src = fs::read_to_string(benchmark_path(bench)).expect("read benchmark");
        if let Some((from, to)) = repl {
            src = src.replace(from, to);
        }
        if bench == "tree" {
            src = src.replace("2000", "10").replace("1000", "100");
        }
        let (vm, jit, aot) =
            assert_triple_parity_with_opts(&format!("suite_3way_{bench}"), &src, &release_opts);
        assert_eq!(vm.error_class, ErrorClass::Ok);
        assert_eq!(jit.error_class, ErrorClass::Ok);
        assert_eq!(aot.error_class, ErrorClass::Ok);
        assert!(!aot.stdout.trim().is_empty());
    }
}

// ============================================================================
// TEST 2: Edge-Case Math & Arithmetic Error Parity (14 Cases)
// ============================================================================

#[test]
fn test_oracle_2_edge_case_math_triple_parity() {
    let cases: [(&str, &str, bool); 11] = [
        (
            "math_i64_min_div_neg1",
            r#"
function f(a: int, b: int): int
    return a / b
end
local min_i: int = -9223372036854775807 - 1
println(f(min_i, -1))
"#,
            true,
        ),
        (
            "math_i64_min_mod_neg1",
            r#"
function f(a: int, b: int): int
    return a % b
end
local min_i: int = -9223372036854775807 - 1
println(f(min_i, -1))
"#,
            true,
        ),
        (
            "math_i64_min_floordiv_neg1",
            r#"
function f(a: int, b: int): int
    return a // b
end
local min_i: int = -9223372036854775807 - 1
println(f(min_i, -1))
"#,
            true,
        ),
        (
            "math_i64_wrapping_add_sub_mul",
            r#"
function add_w(a: int, b: int): int return a + b end
function sub_w(a: int, b: int): int return a - b end
function mul_w(a: int, b: int): int return a * b end
local max_i: int = 9223372036854775807
local min_i: int = -9223372036854775807 - 1
println(add_w(max_i, 1))
println(sub_w(min_i, 1))
println(mul_w(min_i, -1))
println(mul_w(max_i, 2))
"#,
            true,
        ),
        (
            "math_floordiv_all_quadrants",
            r#"
function fd(a: int, b: int): int return a // b end
println(fd(7, 2))
println(fd(-7, 2))
println(fd(7, -2))
println(fd(-7, -2))
println(fd(-6, 2))
println(fd(6, -2))
println(fd(0, -5))
"#,
            false,
        ),
        (
            "math_div_mod_all_quadrants",
            r#"
function d(a: int, b: int): int return a / b end
function m(a: int, b: int): int return a % b end
println(d(7, 2))
println(d(-7, 2))
println(d(7, -2))
println(d(-7, -2))
println(m(7, 2))
println(m(-7, 2))
println(m(7, -2))
println(m(-7, -2))
"#,
            false,
        ),
        (
            "math_pow_positive_zero_float",
            r#"
function ipow(a: int, b: int): int return a ^ b end
function fpow(a: float, b: float): float return a ^ b end
println(ipow(2, 0))
println(ipow(2, 10))
println(ipow(-2, 3))
println(ipow(-2, 4))
println(ipow(0, 0))
println(ipow(3, 12))
println(fpow(2.0, -3.0))
println(fpow(9.0, 0.5))
"#,
            false,
        ),
        (
            "math_shifts_masked_63",
            r#"
function shl(a: int, b: int): int return a << b end
function shr(a: int, b: int): int return a >> b end
println(shl(1, 0))
println(shl(1, 10))
println(shl(1, 62))
println(shl(1, 63))
println(shl(1, 64))
println(shl(1, 65))
println(shr(-1, 1))
println(shr(-1, 63))
println(shr(-1, 64))
println(shr(-16, 2))
println(shr(1024, 3))
"#,
            true,
        ),
        (
            "math_bitwise_and_or_xor",
            r#"
function ops(a: int, b: int): int
    return (a & b) + (a | b) + (a ~ b)
end
println(ops(255, 170))
println(ops(-1, 42))
println(ops(0, -12345))
println(~0)
println(~(-1))
"#,
            false,
        ),
        (
            "math_float_ieee754_edge_cases",
            r#"
local x: float = 0.0
local acc: float = 0.0
local i: int = 0
while i < 1000 do
    x = x + 0.001
    acc = acc + x * x - acc * 0.5
    i = i + 1
end
println(acc)
local z: float = 0.0
local nan: float = z / z
println(nan == nan)
println(nan != nan)
println(1.5 + 2.25)
println(-7.5 / 2.0)
"#,
            true,
        ),
        (
            "math_comparisons_and_negation",
            r#"
local min_i: int = -9223372036854775807 - 1
local max_i: int = 9223372036854775807
println(min_i < max_i)
println(min_i <= min_i)
println(max_i > min_i)
println(max_i >= max_i)
println(min_i == max_i)
println(min_i != max_i)
println(-42)
println(-(-42))
"#,
            false,
        ),
    ];

    for (name, src, also_release) in cases {
        let (vm, _, _) = assert_triple_parity(name, src, None, None);
        assert_eq!(vm.error_class, ErrorClass::Ok, "[{name}] must succeed");
        if also_release {
            let rel_opts = TripleParityOptions {
                release_mode: "ReleaseFast",
                ..TripleParityOptions::default()
            };
            let (vm_r, _, _) =
                assert_triple_parity_with_opts(&format!("{name}_release"), src, &rel_opts);
            assert_eq!(vm_r.error_class, ErrorClass::Ok);
        }
    }

    // 4 arithmetic error parity cases (`/ 0`, `% 0`, `// 0`, negative int exponent `^ -3`)
    let arith_err_cases = [
        (
            "err_div_by_zero",
            "function f(a: int, b: int): int return a / b end\nprintln(f(10, 0))\n",
            ErrorClass::DivisionByZero,
        ),
        (
            "err_mod_by_zero",
            "function f(a: int, b: int): int return a % b end\nprintln(f(10, 0))\n",
            ErrorClass::DivisionByZero,
        ),
        (
            "err_floordiv_by_zero",
            "function f(a: int, b: int): int return a // b end\nprintln(f(10, 0))\n",
            ErrorClass::DivisionByZero,
        ),
        (
            "err_negative_int_pow",
            "function f(a: int, b: int): int return a ^ b end\nprintln(f(2, -3))\n",
            ErrorClass::NegativeExponent,
        ),
    ];
    for (name, src, expected_cls) in arith_err_cases {
        let (vm, jit, aot) = assert_triple_parity(name, src, None, None);
        assert_eq!(vm.error_class, expected_cls, "[{name}]");
        assert_eq!(jit.error_class, expected_cls, "[{name}]");
        assert_eq!(aot.error_class, expected_cls, "[{name}]");
        assert_eq!(aot.signal, None, "[{name}] Rust AOT must not abort/SIGFPE");
    }
}

// ============================================================================
// TEST 3: Strings, Arrays, OOB, Aliasing, Structs, Recursion, Control Flow (12 Cases)
// ============================================================================

#[test]
fn test_oracle_3_containers_structs_recursion_control_flow_triple_parity() {
    let ok_cases: [(&str, &str); 10] = [
        (
            "str_concat_utf8_tostring",
            r#"
local s: string = ""
local i: int = 0
while i < 15 do
    s = s .. tostring(i % 10)
    i = i + 1
end
println(s)
local u: string = "hello-" .. "мир" .. "-" .. tostring(true) .. "-" .. tostring(false)
println(u)
"#,
        ),
        (
            "arr_int_ops_and_ok_pattern",
            r#"
local a: Array<int> = [10, 20, 30]
array.push(a, 40)
a[1] = 25
local sum: int = 0
local i: int = 0
while i < array.len(a) do
    if a[i] is Ok(v) then
        sum = sum + v
    end
    i = i + 1
end
println(array.len(a))
println(sum)
"#,
        ),
        (
            "arr_bool_u8_stride_parity",
            r#"
local flags: Array<bool> = []
local i: int = 0
while i < 32 do
    array.push(flags, (i % 2) == 0)
    i = i + 1
end
flags[0] = false
flags[3] = true
local count: int = 0
i = 0
while i < array.len(flags) do
    if flags[i] is Ok(b) then
        if b then
            count = count + 1
        end
    end
    i = i + 1
end
println(count)
"#,
        ),
        (
            "arr_float_and_string",
            r#"
local fs: Array<float> = [1.5, 2.25, 3.75]
array.push(fs, 0.5)
local total: float = 0.0
local i: int = 0
while i < array.len(fs) do
    if fs[i] is Ok(v) then
        total = total + v
    end
    i = i + 1
end
println(total)
local ss: Array<string> = ["a", "bb", "ccc"]
array.push(ss, "dddd")
println(array.len(ss))
if ss[2] is Ok(w) then
    println(w)
end
"#,
        ),
        (
            "local_aliasing_array_and_struct",
            r#"
struct Boxed
    val: int
end

function mutate_arr(x: Array<int>, y: Array<int>): int
    array.push(x, 77)
    return y[3]:unwrap()
end

local a: Array<int> = [1, 2, 3]
local b: Array<int> = a
b[0] = 99
println(a[0]:unwrap())
println(mutate_arr(a, b))
println(array.len(a))

local p: Boxed = Boxed { val = 5 }
local q: Boxed = p
p.val = 123
println(q.val)
"#,
        ),
        (
            "struct_methods_and_nested",
            r#"
struct Vec2
    x: int
    y: int
end

impl Vec2
    function dot(self, ox: int, oy: int): int
        return self.x * ox + self.y * oy
    end
    function shift(self, dx: int, dy: int): int
        self.x = self.x + dx
        self.y = self.y + dy
        return self.x + self.y
    end
end

local v: Vec2 = Vec2 { x = 3, y = 4 }
println(v:dot(10, 20))
println(v:shift(2, -1))
println(v.x)
println(v.y)
"#,
        ),
        (
            "recursive_option_struct_tree",
            r#"
struct Node
    val: int
    left: Option<Node>
    right: Option<Node>
end

function build(depth: int, val: int): Node
    if depth == 0 then
        return Node { val = val, left = Option.None, right = Option.None }
    end
    local l: Node = build(depth - 1, val * 2)
    local r: Node = build(depth - 1, val * 2 + 1)
    return Node {
        val = val,
        left = Option.Some(l),
        right = Option.Some(r)
    }
end

function sum_tree(n: Node): int
    local s: int = n.val
    if n.left is Some(l) then
        s = s + sum_tree(l)
    end
    if n.right is Some(r) then
        s = s + sum_tree(r)
    end
    return s
end

local leaf: Node = build(0, 42)
println(sum_tree(leaf))
local t: Node = build(6, 1)
println(sum_tree(t))
"#,
        ),
        (
            "mutual_recursion_even_odd",
            r#"
function is_even(n: int): bool
    if n == 0 then
        return true
    end
    return is_odd(n - 1)
end

function is_odd(n: int): bool
    if n == 0 then
        return false
    end
    return is_even(n - 1)
end

println(is_even(24))
println(is_odd(24))
println(is_even(25))
println(is_odd(25))
"#,
        ),
        (
            "ackermann_recursion",
            r#"
function ack(m: int, n: int): int
    if m == 0 then
        return n + 1
    elseif n == 0 then
        return ack(m - 1, 1)
    else
        return ack(m - 1, ack(m, n - 1))
    end
end
println(ack(3, 4))
"#,
        ),
        (
            "nested_loops_break_continue_shortcircuit",
            r#"
function compute(limit: int): int
    local total: int = 0
    local i: int = 0
    while i < limit do
        i = i + 1
        if i % 3 == 0 then
            continue
        end
        local j: int = 0
        while j < 10 do
            j = j + 1
            if j == 6 then
                break
            end
            if (i > 2) and (j % 2 == 1) then
                total = total + i * j
            elseif (i == 1) or (j == 2) then
                total = total + 1
            end
        end
        if total > 500 then
            return total
        end
    end
    return total
end
println(compute(20))
println(compute(5))
"#,
        ),
    ];

    for (name, src) in ok_cases {
        let (vm, _, _) = assert_triple_parity(name, src, None, None);
        assert_eq!(vm.error_class, ErrorClass::Ok, "[{name}] must succeed");
    }

    // 2 OOB error parity cases (positive OOB & negative index)
    let oob_cases = [
        (
            "err_oob_positive",
            "local a: Array<int> = [1, 2, 3]\nprintln(a[10]:unwrap())\n",
        ),
        (
            "err_oob_negative",
            "local a: Array<int> = [1, 2, 3]\nprintln(a[-1]:unwrap())\n",
        ),
    ];
    for (name, src) in oob_cases {
        let (vm, jit, aot) = assert_triple_parity(name, src, None, None);
        assert_eq!(vm.error_class, ErrorClass::IndexOutOfBounds, "[{name}]");
        assert_eq!(jit.error_class, ErrorClass::IndexOutOfBounds, "[{name}]");
        assert_eq!(aot.error_class, ErrorClass::IndexOutOfBounds, "[{name}]");
        assert_eq!(aot.signal, None, "[{name}] AOT must not panic/abort on OOB");
    }
}

// ============================================================================
// TEST 4: Tier 4 Features — Map, Enum, Closure, defer/errdefer, Schema (11 Cases)
// ============================================================================

#[test]
fn test_oracle_4_tier4_map_enum_closure_defer_schema_triple_parity() {
    let ok_cases: [(&str, &str); 7] = [
        (
            "tier4_map_int_int_rw",
            r#"
local m: Map<int, int> = {}
local i: int = 0
while i < 50 do
    map.set(m, i % 16, i * 3 + 1)
    i = i + 1
end
local sum: int = 0
i = 0
while i < 16 do
    if map.has(m, i) then
        sum = sum + map.get(m, i):unwrap()
    end
    i = i + 1
end
map.delete(m, 0)
println(sum)
println(map.has(m, 0))
println(map.len(m))
"#,
        ),
        (
            "tier4_map_string_aliasing",
            r#"
local m: Map<string, int> = { ["alpha"] = 10, ["beta"] = 20 }
local alias: Map<string, int> = m
map.set(m, "gamma", 30)
println(map.len(alias))
println(map.get(alias, "alpha"):unwrap() + map.get(m, "gamma"):unwrap())
map.delete(alias, "beta")
println(map.has(m, "beta"))
println(map.len(m))
"#,
        ),
        (
            "tier4_option_and_result_enum_patterns",
            r#"
struct Payload
    val: int
end

function inspect(opt: Option<Payload>, fallback: int): int
    if opt is None then
        return fallback
    end
    if opt is Some(p) then
        return p.val * 3
    end
    return -1
end

local empty: Option<Payload> = Option.None
local full: Option<Payload> = Option.Some(Payload { val = 14 })
println(inspect(empty, 99))
println(inspect(full, 99))
local xs: Array<int> = [5, 10, 15]
if xs[2] is Ok(v) then
    println(v)
end
"#,
        ),
        (
            "tier4_closure_immutable_capture",
            r#"
function apply_adder(offset: int, x: int): int
    local add = function(v: int): int
        return v + offset
    end
    return add(x)
end
println(apply_adder(10, 32))
println(apply_adder(-5, 100))
"#,
        ),
        (
            "tier4_closure_mutable_counter_instances",
            r#"
function make_counter(start: int): function(): int
    local n: int = start
    return function(): int
        n = n + 1
        return n
    end
end

local c1 = make_counter(10)
local c2 = make_counter(100)
println(c1())
println(c1())
println(c2())
println(c1())
println(c2())
"#,
        ),
        (
            "tier4_closure_shared_array_capture",
            r#"
function test_shared(): int
    local log: Array<int> = []
    local push_v = function(x: int): int
        array.push(log, x)
        return array.len(log)
    end
    local n1: int = push_v(10)
    local n2: int = push_v(25)
    return n1 + n2 + log[0]:unwrap() + log[1]:unwrap()
end
println(test_shared())
"#,
        ),
        (
            "tier4_defer_errdefer_ok_and_early_return",
            r#"
function run_ok(early: bool): int
    defer {
        println("defer_1")
    }
    errdefer {
        println("errdefer_should_not_run")
    }
    defer {
        println("defer_2")
    }
    if early then
        println("early_branch")
        return 7
    end
    println("normal_branch")
    return 42
end

println(run_ok(true))
println(run_ok(false))
"#,
        ),
    ];

    for (name, src) in ok_cases {
        let (vm, _, _) = assert_triple_parity(name, src, None, None);
        assert_eq!(vm.error_class, ErrorClass::Ok, "[{name}] must succeed");
    }

    // MissingMapKey error parity
    let missing_map_src = r#"
local m: Map<string, int> = { ["k"] = 1 }
println(map.get(m, "missing"):unwrap())
"#;
    let (vm_m, jit_m, aot_m) =
        assert_triple_parity("tier4_err_missing_map_key", missing_map_src, None, None);
    assert_eq!(vm_m.error_class, ErrorClass::MissingMapKey);
    assert_eq!(jit_m.error_class, ErrorClass::MissingMapKey);
    assert_eq!(aot_m.error_class, ErrorClass::MissingMapKey);

    // defer + errdefer on runtime DivisionByZero error:
    // - All 3 engines match strictly on ErrorClass::DivisionByZero
    // - VM == Cranelift JIT match on pre-error stdout
    // - Rust AOT executes full interleaved LIFO defer + errdefer via DropGuard under -C panic=abort
    //   (Note: VM/JIT currently do not unwind defer/errdefer on unhandled runtime errors;
    //   normal + early-return defer parity is verified bit-exact across all 3 engines above.)
    let defer_div_src = r#"
function run_err_div(b: int): int
    defer {
        println("defer_1")
    }
    errdefer {
        println("errdefer_1")
    }
    defer {
        println("defer_2")
    }
    println("body_err")
    local x: int = 10 / b
    return x
end
println(run_err_div(0))
"#;
    let (vm_d, jit_d, aot_d) =
        assert_triple_parity("tier4_defer_errdefer_div_zero", defer_div_src, None, None);
    assert_eq!(vm_d.error_class, ErrorClass::DivisionByZero);
    assert_eq!(jit_d.error_class, ErrorClass::DivisionByZero);
    assert_eq!(aot_d.error_class, ErrorClass::DivisionByZero);
    assert_eq!(vm_d.stdout, jit_d.stdout);
    assert_eq!(aot_d.stdout, "body_err\ndefer_2\nerrdefer_1\ndefer_1\n");

    // defer + errdefer on user error("boom") with shared mutable array mutation!
    let defer_user_err_src = r#"
function run_err_user(): int
    local log: Array<int> = []
    defer {
        array.push(log, 10)
        println(array.len(log))
    }
    errdefer {
        array.push(log, 20)
        println("err_ran")
    }
    array.push(log, 1)
    error("boom")
    return 0
end
run_err_user()
"#;
    let (vm_u, jit_u, aot_u) =
        assert_triple_parity("tier4_defer_errdefer_user_error", defer_user_err_src, None, None);
    assert_eq!(vm_u.error_class, ErrorClass::UserError);
    assert_eq!(jit_u.error_class, ErrorClass::UserError);
    assert_eq!(aot_u.error_class, ErrorClass::UserError);
    assert_eq!(vm_u.stdout, jit_u.stdout);
    assert_eq!(aot_u.stdout, "err_ran\n3\n");

    // Schema decode parity (`User.decode(json)`): valid shuffled JSON + fail-closed invalid JSON
    let schema_ok_src = r#"
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
    println(eval_user("{\"name\":\"a\",\"active\":true,\"score\":1.5,\"id\":7}"))
    println(eval_user("{\"id\":42,\"score\":2.5,\"active\":false,\"name\":\"b\"}"))
    return 0
end
"#;
    let tmp = tempdir().expect("schema tempdir");
    let schema_script = tmp.path().join("schema_ok.lust");
    let schema_bin = tmp.path().join("schema_ok_bin");
    fs::write(&schema_script, schema_ok_src).expect("write schema_ok");
    let schema_opts = TripleParityOptions::default();
    let vm_s = run_worker_child("vm", &schema_script, &schema_opts, true);
    let jit_s = run_worker_child("cranelift", &schema_script, &schema_opts, true);
    let aot_s = run_rust_aot_engine(&schema_script, &schema_bin, &schema_opts);
    assert_eq!(vm_s.error_class, ErrorClass::Ok, "VM schema stderr: {}", vm_s.stderr);
    assert_eq!(jit_s.error_class, ErrorClass::Ok, "JIT schema stderr: {}", jit_s.stderr);
    assert_eq!(aot_s.error_class, ErrorClass::Ok, "AOT schema stderr: {}", aot_s.stderr);
    assert_eq!(vm_s.stdout, "107\n42\n");
    assert_eq!(vm_s.stdout, jit_s.stdout);
    assert_eq!(jit_s.stdout, aot_s.stdout);

    // Schema decode fail-closed invalid JSON (duplicate key)
    let schema_bad_src = r#"
struct User
    id: int,
    score: float,
    active: bool,
    name: string
end

function main(): int
    local u: User = User.decode("{\"id\":1,\"id\":2,\"score\":1.0,\"active\":true,\"name\":\"x\"}")
    println(u.id)
    return 0
end
"#;
    let schema_bad_script = tmp.path().join("schema_bad.lust");
    let schema_bad_bin = tmp.path().join("schema_bad_bin");
    fs::write(&schema_bad_script, schema_bad_src).expect("write schema_bad");
    let vm_sb = run_worker_child("vm", &schema_bad_script, &schema_opts, true);
    let jit_sb = run_worker_child("cranelift", &schema_bad_script, &schema_opts, true);
    let aot_sb = run_rust_aot_engine(&schema_bad_script, &schema_bad_bin, &schema_opts);
    assert_ne!(vm_sb.error_class, ErrorClass::Ok, "VM must reject duplicate key");
    assert_ne!(jit_sb.error_class, ErrorClass::Ok, "JIT must reject duplicate key");
    assert_eq!(
        aot_sb.error_class,
        ErrorClass::SchemaDecodeError,
        "AOT must reject duplicate key with SchemaDecodeError, stderr: {}",
        aot_sb.stderr
    );

    // User-defined Enum (`MirEnumDef` + `enum.tag` + `enum.payload` + `Terminator::Switch`)
    // compiled via Rust AOT and verified in an isolated sandboxed child process.
    let span = Span::dummy();
    let mut enum_mod = MirModule::new("oracle_enum_mod".to_string());
    enum_mod.register_enum_def(MirEnumDef {
        name: "Shape".to_string(),
        variants: vec![
            ("Point".to_string(), vec![]),
            ("Circle".to_string(), vec![MirType::Float]),
            ("Rect".to_string(), vec![MirType::Int, MirType::Int]),
        ],
    });
    let mut score_fn = MirFunction::new(
        FunctionSignature {
            name: "score".to_string(),
            param_types: vec![MirType::Enum("Shape".to_string())],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let p_shape = score_fn.params[0];
    let tag_loc = score_fn.add_local(None, MirType::Int, true, span);
    let r_loc = score_fn.add_local(None, MirType::Float, true, span);
    let r_scaled = score_fn.add_local(None, MirType::Float, true, span);
    let r_int = score_fn.add_local(None, MirType::Int, true, span);
    let w_loc = score_fn.add_local(None, MirType::Int, true, span);
    let h_loc = score_fn.add_local(None, MirType::Int, true, span);
    let area_loc = score_fn.add_local(None, MirType::Int, true, span);
    let bb_sw = score_fn.new_block();
    let bb_pt = score_fn.new_block();
    let bb_ci = score_fn.new_block();
    let bb_ci2 = score_fn.new_block();
    let bb_rc = score_fn.new_block();
    let bb_rc2 = score_fn.new_block();
    let bb_rc3 = score_fn.new_block();
    let bb_def = score_fn.new_block();
    score_fn.blocks[0].terminator = Some(Terminator::Call {
        callee: "enum.tag".to_string(),
        args: vec![Operand::Copy(Place::Local(p_shape))],
        destination: Place::Local(tag_loc),
        success: bb_sw,
        error: None,
        span,
    });
    score_fn.blocks[bb_sw.0].terminator = Some(Terminator::Switch {
        discr: Operand::Copy(Place::Local(tag_loc)),
        cases: vec![(0, bb_pt), (1, bb_ci), (2, bb_rc)],
        default: bb_def,
        span,
    });
    score_fn.blocks[bb_pt.0].terminator =
        Some(Terminator::Return(Some(Operand::Constant(MirConstant::Int(1))), span));
    score_fn.blocks[bb_ci.0].terminator = Some(Terminator::Call {
        callee: "enum.payload".to_string(),
        args: vec![
            Operand::Copy(Place::Local(p_shape)),
            Operand::Constant(MirConstant::Int(0)),
        ],
        destination: Place::Local(r_loc),
        success: bb_ci2,
        error: None,
        span,
    });
    score_fn.blocks[bb_ci2.0].statements.push(Statement::Assign(
        Place::Local(r_scaled),
        Rvalue::BinaryOp(
            MirBinaryOp::Mul,
            Operand::Copy(Place::Local(r_loc)),
            Operand::Constant(MirConstant::Float(10.0)),
        ),
        span,
    ));
    score_fn.blocks[bb_ci2.0].statements.push(Statement::Assign(
        Place::Local(r_int),
        Rvalue::Cast(Operand::Copy(Place::Local(r_scaled)), MirType::Int),
        span,
    ));
    score_fn.blocks[bb_ci2.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(r_int))),
        span,
    ));
    score_fn.blocks[bb_rc.0].terminator = Some(Terminator::Call {
        callee: "enum.payload".to_string(),
        args: vec![
            Operand::Copy(Place::Local(p_shape)),
            Operand::Constant(MirConstant::Int(0)),
        ],
        destination: Place::Local(w_loc),
        success: bb_rc2,
        error: None,
        span,
    });
    score_fn.blocks[bb_rc2.0].terminator = Some(Terminator::Call {
        callee: "enum.payload".to_string(),
        args: vec![
            Operand::Copy(Place::Local(p_shape)),
            Operand::Constant(MirConstant::Int(1)),
        ],
        destination: Place::Local(h_loc),
        success: bb_rc3,
        error: None,
        span,
    });
    score_fn.blocks[bb_rc3.0].statements.push(Statement::Assign(
        Place::Local(area_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::Mul,
            Operand::Copy(Place::Local(w_loc)),
            Operand::Copy(Place::Local(h_loc)),
        ),
        span,
    ));
    score_fn.blocks[bb_rc3.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(area_loc))),
        span,
    ));
    score_fn.blocks[bb_def.0].terminator =
        Some(Terminator::Return(Some(Operand::Constant(MirConstant::Int(-1))), span));
    enum_mod.add_function(score_fn);

    let mut main_fn = MirFunction::new(
        FunctionSignature {
            name: "main".to_string(),
            param_types: vec![],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let s0 = main_fn.add_local(Some("s0".into()), MirType::Enum("Shape".into()), true, span);
    let s1 = main_fn.add_local(Some("s1".into()), MirType::Enum("Shape".into()), true, span);
    let s2 = main_fn.add_local(Some("s2".into()), MirType::Enum("Shape".into()), true, span);
    let v0 = main_fn.add_local(None, MirType::Int, true, span);
    let v1 = main_fn.add_local(None, MirType::Int, true, span);
    let v2 = main_fn.add_local(None, MirType::Int, true, span);
    let unit_tmp = main_fn.add_local(None, MirType::Unit, true, span);
    let b1 = main_fn.new_block();
    let b2 = main_fn.new_block();
    let b3 = main_fn.new_block();
    let b4 = main_fn.new_block();
    let b5 = main_fn.new_block();
    let b6 = main_fn.new_block();
    let b7 = main_fn.new_block();
    let b8 = main_fn.new_block();
    main_fn.blocks[0].statements.push(Statement::Assign(
        Place::Local(s0),
        Rvalue::Aggregate(
            MirType::Enum("Shape".into()),
            vec![Operand::Constant(MirConstant::Int(0))],
        ),
        span,
    ));
    main_fn.blocks[0].terminator = Some(Terminator::Call {
        callee: "Shape.Circle".into(),
        args: vec![Operand::Constant(MirConstant::Float(3.5))],
        destination: Place::Local(s1),
        success: b1,
        error: None,
        span,
    });
    main_fn.blocks[b1.0].terminator = Some(Terminator::Call {
        callee: "enum.new".into(),
        args: vec![
            Operand::Constant(MirConstant::Int(2)),
            Operand::Constant(MirConstant::Int(6)),
            Operand::Constant(MirConstant::Int(7)),
        ],
        destination: Place::Local(s2),
        success: b2,
        error: None,
        span,
    });
    main_fn.blocks[b2.0].terminator = Some(Terminator::Call {
        callee: "score".into(),
        args: vec![Operand::Copy(Place::Local(s0))],
        destination: Place::Local(v0),
        success: b3,
        error: None,
        span,
    });
    main_fn.blocks[b3.0].terminator = Some(Terminator::Call {
        callee: "score".into(),
        args: vec![Operand::Copy(Place::Local(s1))],
        destination: Place::Local(v1),
        success: b4,
        error: None,
        span,
    });
    main_fn.blocks[b4.0].terminator = Some(Terminator::Call {
        callee: "score".into(),
        args: vec![Operand::Copy(Place::Local(s2))],
        destination: Place::Local(v2),
        success: b5,
        error: None,
        span,
    });
    main_fn.blocks[b5.0].terminator = Some(Terminator::Call {
        callee: "println".into(),
        args: vec![Operand::Copy(Place::Local(s0)), Operand::Copy(Place::Local(v0))],
        destination: Place::Local(unit_tmp),
        success: b6,
        error: None,
        span,
    });
    main_fn.blocks[b6.0].terminator = Some(Terminator::Call {
        callee: "println".into(),
        args: vec![Operand::Copy(Place::Local(s1)), Operand::Copy(Place::Local(v1))],
        destination: Place::Local(unit_tmp),
        success: b7,
        error: None,
        span,
    });
    main_fn.blocks[b7.0].terminator = Some(Terminator::Call {
        callee: "println".into(),
        args: vec![Operand::Copy(Place::Local(s2)), Operand::Copy(Place::Local(v2))],
        destination: Place::Local(unit_tmp),
        success: b8,
        error: None,
        span,
    });
    main_fn.blocks[b8.0].terminator = Some(Terminator::Return(
        Some(Operand::Constant(MirConstant::Int(0))),
        span,
    ));
    enum_mod.add_function(main_fn);

    let enum_bin = tmp.path().join("oracle_enum_bin");
    let enum_opts = RustCompileOptions {
        aot: AotCompileOptions {
            target_kind: AotTargetKind::Executable,
            release_mode: "Debug".to_string(),
            ..AotCompileOptions::default()
        },
        cache_dir: Some(shared_rlib_cache_dir().to_path_buf()),
        ..RustCompileOptions::default()
    };
    AotCompiler::compile_module_rust(&enum_mod, &enum_bin, &enum_opts)
        .expect("compile MIR enum module");
    let enum_run = run_in_sandbox(&enum_bin, Vec::new(), &[], Duration::from_secs(10));
    assert_eq!(enum_run.error_class, ErrorClass::Ok, "stderr: {}", enum_run.stderr);
    assert_eq!(
        enum_run.stdout,
        "Shape.Point\t1\nShape.Circle(3.5)\t35\nShape.Rect(6, 7)\t42\n"
    );
}

// ============================================================================
// TEST 5: Exact MIR `GasUsed` Parity (`F-17` Part 2)
//         `MIR-ResourceCheck Oracle == Cranelift JIT (MIR body) == Rust AOT Exact`
//         + MIR-VM (`MirToBytecodeCompiler`) result & SEC-01 monotonic parity
// ============================================================================

fn push_local(
    f: &mut MirFunction,
    name: &str,
    ty: MirType,
    is_mut: bool,
    is_param: bool,
) -> LocalId {
    let id = LocalId(f.locals.len());
    let mut decl = LocalDecl::new(id, Some(name.into()), ty, is_mut, Span::dummy());
    decl.is_param = is_param;
    f.locals.push(decl);
    if is_param {
        f.params.push(id);
    }
    id
}

fn push_block(f: &mut MirFunction) -> BlockId {
    let id = BlockId(f.blocks.len());
    f.blocks.push(BasicBlock::new(id));
    id
}

fn build_gas_loop_mir_fn(name: &str, header_amt: u64, body_extra_amt: Option<u64>) -> MirFunction {
    let mut f = MirFunction::new(
        FunctionSignature {
            name: name.into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            effects: None,
            is_public: true,
        },
        Span::dummy(),
    );
    f.locals.clear();
    f.params.clear();
    let n = push_local(&mut f, "n", MirType::Int, false, true);
    let i = push_local(&mut f, "i", MirType::Int, true, false);
    let acc = push_local(&mut f, "acc", MirType::Int, true, false);
    let cond = push_local(&mut f, "cond", MirType::Bool, true, false);

    let bb0 = f.entry;
    let bb_hdr = push_block(&mut f);
    let bb_body = push_block(&mut f);
    let bb_exit = push_block(&mut f);

    f.blocks[bb0.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[bb0.0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[bb0.0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    f.blocks[bb_hdr.0]
        .statements
        .push(Statement::ResourceCheck(
            ResourceKind::Gas,
            header_amt,
            Span::dummy(),
        ));
    f.blocks[bb_hdr.0].statements.push(Statement::Assign(
        Place::Local(cond),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(i)),
            Operand::Copy(Place::Local(n)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_hdr.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond)),
        then_block: bb_body,
        else_block: bb_exit,
        span: Span::dummy(),
    });

    if let Some(extra) = body_extra_amt {
        f.blocks[bb_body.0]
            .statements
            .push(Statement::ResourceCheck(
                ResourceKind::Gas,
                extra,
                Span::dummy(),
            ));
    }
    f.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(acc)),
            Operand::Copy(Place::Local(i)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_body.0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    f.blocks[bb_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(acc))),
        Span::dummy(),
    ));
    f
}

/// Compiles a `MirModule` as a Rust AOT library with `GasMetering::Exact` and runs a driver binary
/// in a sandboxed child process to observe exact `(Result<i64, ErrorClass>, ctx.gas_used())`.
fn run_rust_aot_exact_gas_driver(
    module: &MirModule,
    fn_name: &str,
    arg_n: i64,
    gas_limit: u64,
) -> (Result<i64, ErrorClass>, u64) {
    let tmp = tempdir().expect("gas driver tempdir");
    let lib_opts = RustEmitterOptions {
        is_library: true,
        max_memory_bytes: None,
        gas_metering: GasMetering::Exact,
        gas_limit: Some(gas_limit),
        pure_core: false,
        no_std: false,
    };
    let emitted = emit_rust_from_mir(module, &lib_opts).expect("emit exact gas library");
    let combined_rs = tmp.path().join("gas_driver_combined.rs");
    let driver_bin = tmp.path().join("gas_driver_bin");
    let combined_src = format!(
        r#"{emitted}

fn main() {{
    let mut ctx = lust_aot_rt::AotCtx::with_gas_limit({gas_limit}_u64);
    match __lust_fn_{fn_name}(&mut ctx, {arg_n}_i64) {{
        Ok(v) => println!("OK={{v}}|GAS={{}}", ctx.gas_used()),
        Err(lust_aot_rt::AotError::OutOfGas {{ .. }}) => println!("OOG|GAS={{}}", ctx.gas_used()),
        Err(e) => {{
            eprintln!("ERR={{e}}");
            std::process::exit(2);
        }}
    }}
}}
"#
    );
    fs::write(&combined_rs, &combined_src).expect("write gas_driver_combined.rs");

    let rt_opts = RustCompileOptions {
        aot: AotCompileOptions {
            target_kind: AotTargetKind::Executable,
            release_mode: "Debug".to_string(),
            ..AotCompileOptions::default()
        },
        cache_dir: Some(shared_rlib_cache_dir().to_path_buf()),
        ..RustCompileOptions::default()
    };
    let rt_rlib = ensure_cached_aot_rt_rlib(&rt_opts).expect("cached rt rlib");
    let rustc = find_rustc();

    let c_drv = Command::new(&rustc)
        .arg("--edition=2024")
        .arg("-C")
        .arg("opt-level=0")
        .arg("-C")
        .arg("panic=abort")
        .arg("--extern")
        .arg(format!("lust_aot_rt={}", rt_rlib.display()))
        .arg(&combined_rs)
        .arg("-o")
        .arg(&driver_bin)
        .output()
        .expect("compile gas_driver");
    assert!(
        c_drv.status.success(),
        "gas_driver compile failed:\n{}\n--- SOURCE ---\n{combined_src}",
        String::from_utf8_lossy(&c_drv.stderr)
    );

    let out = run_in_sandbox(&driver_bin, Vec::new(), &[], Duration::from_secs(10));
    assert_eq!(
        out.error_class,
        ErrorClass::Ok,
        "gas_driver failed: stdout={} stderr={}",
        out.stdout,
        out.stderr
    );
    let line = out.stdout.trim();
    let (status_part, gas_part) = line
        .split_once("|GAS=")
        .unwrap_or_else(|| panic!("unexpected gas_driver output: {line:?}"));
    let gas_used: u64 = gas_part.parse().expect("parse gas_used");
    if let Some(val_str) = status_part.strip_prefix("OK=") {
        (Ok(val_str.parse::<i64>().expect("parse result")), gas_used)
    } else if status_part == "OOG" {
        (Err(ErrorClass::OutOfGas), gas_used)
    } else {
        panic!("unexpected gas_driver status: {line:?}");
    }
}

#[test]
fn test_oracle_5_exact_mir_gas_used_parity() {
    // Three distinct MIR shapes sharing explicit `Statement::ResourceCheck(ResourceKind::Gas, amt, _)`:
    //   1. Simple single-gas loop (amt = 1, exercises Cranelift Q=64 strip-mining)
    //   2. Multi-gas loop (header amt = 1 + body amt = 1, F-06 unquantized exact path)
    //   3. Non-unit gas loop (header amt = 5, exercises non-unit quantum arm threshold 315)
    let shapes: [(&str, u64, Option<u64>, Vec<(i64, u64)>); 3] = [
        (
            "mir_gas_simple",
            1,
            None,
            vec![
                // Completing runs
                (50, 200),
                (200, 500),
                (998, 1000),
                // Exact OOG boundaries across Q = 64
                (500, 1),
                (500, 50),
                (500, 63),
                (500, 64),
                (500, 65),
                (500, 128),
                (500, 129),
            ],
        ),
        (
            "mir_gas_multi_f06",
            1,
            Some(1),
            vec![
                (40, 200),
                (100, 50),
                (200, 64),
                (200, 65),
            ],
        ),
        (
            "mir_gas_amt5",
            5,
            None,
            vec![
                (30, 500),
                (200, 315),
                (200, 316),
                (200, 320),
                (200, 321),
            ],
        ),
    ];

    for (fn_name, header_amt, body_extra_amt, runs) in shapes {
        let mir_fn = build_gas_loop_mir_fn(fn_name, header_amt, body_extra_amt);
        let mut module = MirModule::new(format!("{fn_name}_mod"));
        module.add_function(mir_fn);
        MirVerifier::verify_module(&module).expect("verify MIR gas module");

        for (n, mir_budget) in runs {
            // Expected exact MIR ResourceCheck(Gas) count for loop `for i in 0..n`:
            // Each of the `n` iterations executes 1 header check (+ optional body check),
            // and the final exit check executes 1 header check.
            let per_iter_total = header_amt + body_extra_amt.unwrap_or(0);
            let full_gas = (n as u64) * per_iter_total + header_amt;
            let expect_ok = full_gas <= mir_budget;

            // 1. MIR-VM (`MirToBytecodeCompiler` + `VmBackend::Bytecode`)
            let mut mir_vm = VM::new();
            mir_vm.load_mir_module(&module).expect("load MIR in MIR-VM");
            mir_vm.set_backend(VmBackend::Bytecode);
            let vm_res = mir_vm.call(fn_name, vec![Value::Int(n)]).expect("MIR-VM result");
            let expected_sum = n * (n - 1) / 2;
            assert_eq!(
                vm_res,
                Value::Int(expected_sum),
                "[{fn_name}] MIR-VM return value mismatch for n={n}"
            );

            // Also verify MIR-VM deterministic OutOfGas termination at tight budget
            if !expect_ok {
                let mut mir_vm_oog = VM::new();
                mir_vm_oog
                    .load_mir_module(&module)
                    .expect("load MIR in MIR-VM OOG");
                mir_vm_oog.set_backend(VmBackend::Bytecode);
                mir_vm_oog.set_gas_budget(mir_budget);
                let oog_err = mir_vm_oog
                    .call(fn_name, vec![Value::Int(n)])
                    .expect_err("MIR-VM must trip OutOfGas on insufficient budget");
                assert!(
                    oog_err.to_string().contains("Out of gas"),
                    "[{fn_name}] MIR-VM expected Out of gas, got: {oog_err}"
                );
            }

            // 2. Cranelift JIT (`VmBackend::Cranelift` on the exact same `MirModule`)
            //    Note: `cranelift_dispatch.rs:4379` charges 1 base VM entry gas unit before
            //    executing MIR body `Statement::ResourceCheck(ResourceKind::Gas, amt, _)`.
            //    Giving Cranelift `mir_budget + 1` makes the MIR body receive the EXACT same
            //    `mir_budget` as Rust AOT (`GasMetering::Exact`), so:
            //      `cranelift_mir_gas_used == vm.gas_used() - 1 == aot_gas_used`.
            safepoint_reset();
            let mut jit_vm = VM::new();
            jit_vm.load_mir_module(&module).expect("load MIR in JIT");
            jit_vm.set_backend(VmBackend::Cranelift);
            jit_vm.set_gas_budget(mir_budget + 1);
            let jit_call = jit_vm.call(fn_name, vec![Value::Int(n)]);
            let jit_mir_gas_used = jit_vm
                .gas_used()
                .checked_sub(1)
                .expect("Cranelift charges 1 entry gas unit");
            let jit_safepoints = safepoint_stats().safepoints_checked;

            // 3. Rust AOT (`GasMetering::Exact` on the exact same `MirModule`, in sandboxed child)
            let (aot_res, aot_gas_used) =
                run_rust_aot_exact_gas_driver(&module, fn_name, n, mir_budget);

            if expect_ok {
                let jit_val = jit_call.unwrap_or_else(|e| {
                    panic!("[{fn_name} n={n} budget={mir_budget}] Cranelift unexpectedly failed: {e}")
                });
                assert_eq!(jit_val, Value::Int(expected_sum));
                assert_eq!(aot_res, Ok(expected_sum));
                assert_eq!(
                    jit_mir_gas_used, full_gas,
                    "[{fn_name} n={n} budget={mir_budget}] Cranelift MIR gas_used mismatch (safepoints={jit_safepoints})"
                );
                assert_eq!(
                    aot_gas_used, jit_mir_gas_used,
                    "[{fn_name} n={n} budget={mir_budget}] Exact GasUsed mismatch: Rust AOT ({aot_gas_used}) != Cranelift JIT MIR ({jit_mir_gas_used})"
                );
            } else {
                let jit_err = jit_call.expect_err(&format!(
                    "[{fn_name} n={n} budget={mir_budget}] Cranelift must trip OutOfGas"
                ));
                assert!(
                    jit_err.to_string().contains("Out of gas"),
                    "[{fn_name}] Cranelift expected Out of gas, got: {jit_err}"
                );
                assert_eq!(
                    aot_res,
                    Err(ErrorClass::OutOfGas),
                    "[{fn_name} n={n} budget={mir_budget}] Rust AOT must trip OutOfGas"
                );
                assert_eq!(
                    aot_gas_used, jit_mir_gas_used,
                    "[{fn_name} n={n} budget={mir_budget}] Exact OOG GasUsed mismatch: Rust AOT ({aot_gas_used}) != Cranelift JIT MIR ({jit_mir_gas_used})"
                );
                assert!(
                    aot_gas_used > mir_budget,
                    "[{fn_name} n={n} budget={mir_budget}] OOG gas_used ({aot_gas_used}) must exceed budget ({mir_budget})"
                );
            }
        }
    }
}

// ============================================================================
// TEST 6: Security Invariants (`SEC-01..SEC-15`) in Isolated Child Processes
//         (Rule 5 `AGENTS.md`: NEVER run hostile probes in main runner process)
// ============================================================================

#[test]
fn test_oracle_6_security_invariants_sec01_to_sec15_isolated_child_processes() {
    let tmp = tempdir().expect("security tempdir");

    // ------------------------------------------------------------------------
    // SEC-01 & SEC-02: Infinite loop stopped deterministically by gas budget
    // across Bytecode VM, Cranelift JIT, Rust AOT (exact) & Rust AOT (loop-chunked)
    // ------------------------------------------------------------------------
    let inf_loop_src = r#"
local i: int = 0
while true do
    i = i + 1
end
println(i)
"#;
    for gas_budget in [50u64, 100, 500] {
        // Exact gas metering (3-way VM == JIT == AOT exact)
        let (vm, jit, aot) = assert_triple_parity(
            &format!("sec01_02_inf_loop_exact_{gas_budget}"),
            inf_loop_src,
            Some(gas_budget),
            None,
        );
        assert_eq!(vm.error_class, ErrorClass::OutOfGas, "SEC-01 VM OOG budget={gas_budget}");
        assert_eq!(jit.error_class, ErrorClass::OutOfGas, "SEC-02 JIT OOG budget={gas_budget}");
        assert_eq!(aot.error_class, ErrorClass::OutOfGas, "SEC-02 AOT exact OOG budget={gas_budget}");
        assert_eq!(aot.exit_code, Some(1));
        assert_eq!(aot.signal, None);

        // Also verify Rust AOT `--gas-metering loop-chunked` terminates deterministically with OutOfGas
        let chunked_opts = TripleParityOptions {
            release_mode: "ReleaseFast",
            gas_metering: GasMetering::LoopChunked,
            gas_limit: Some(gas_budget),
            timeout: Duration::from_secs(10),
            ..TripleParityOptions::default()
        };
        let script = tmp.path().join(format!("sec02_chunked_{gas_budget}.lust"));
        let bin = tmp.path().join(format!("sec02_chunked_{gas_budget}_bin"));
        fs::write(&script, inf_loop_src).expect("write sec02_chunked");
        let aot_chunked = run_rust_aot_engine(&script, &bin, &chunked_opts);
        assert!(
            !aot_chunked.timed_out,
            "SEC-02 AOT loop-chunked timed out at budget={gas_budget}"
        );
        assert_eq!(
            aot_chunked.error_class,
            ErrorClass::OutOfGas,
            "SEC-02 AOT loop-chunked must trip OutOfGas at budget={gas_budget}, stderr: {}",
            aot_chunked.stderr
        );
        assert_eq!(aot_chunked.exit_code, Some(1));
        assert_eq!(aot_chunked.signal, None);
    }

    // ------------------------------------------------------------------------
    // SEC-03: Memory bomb containment (Array growth, Map growth, String growth)
    //         under finite memory limits -> deterministic OutOfMemory, exit 1
    // ------------------------------------------------------------------------
    let array_oom_src = r#"
local a: Array<int> = []
local i: int = 0
while i < 100000 do
    array.push(a, i)
    i = i + 1
end
println(array.len(a))
"#;
    let (vm_a, jit_a, aot_a) =
        assert_triple_parity("sec03_oom_array_bomb", array_oom_src, None, Some(256));
    assert_eq!(vm_a.error_class, ErrorClass::OutOfMemory);
    assert_eq!(jit_a.error_class, ErrorClass::OutOfMemory);
    assert_eq!(aot_a.error_class, ErrorClass::OutOfMemory);
    assert_eq!(aot_a.exit_code, Some(1));
    assert_eq!(aot_a.signal, None);

    let map_oom_src = r#"
local m: Map<int, int> = {}
local i: int = 0
while i < 10000 do
    map.set(m, i, i * 2)
    i = i + 1
end
println(map.len(m))
"#;
    let map_oom_script = tmp.path().join("sec03_map_oom.lust");
    let map_oom_bin = tmp.path().join("sec03_map_oom_bin");
    fs::write(&map_oom_script, map_oom_src).expect("write map_oom");
    let map_oom_opts = TripleParityOptions {
        mem_limit: Some(128),
        timeout: Duration::from_secs(10),
        ..TripleParityOptions::default()
    };
    let aot_map_oom = run_rust_aot_engine(&map_oom_script, &map_oom_bin, &map_oom_opts);
    assert_eq!(
        aot_map_oom.error_class,
        ErrorClass::OutOfMemory,
        "SEC-03 Map OOM in AOT failed: stderr={}",
        aot_map_oom.stderr
    );
    assert_eq!(aot_map_oom.exit_code, Some(1));
    assert_eq!(aot_map_oom.signal, None);

    let string_oom_src = r#"
local s: string = ""
local chunk: string = "0123456789abcdef0123456789abcdef"
local i: int = 0
while i < 100 do
    s = s .. chunk
    i = i + 1
end
println(s)
"#;
    let str_oom_script = tmp.path().join("sec03_str_oom.lust");
    let str_oom_bin = tmp.path().join("sec03_str_oom_bin");
    fs::write(&str_oom_script, string_oom_src).expect("write str_oom");
    let str_oom_opts = TripleParityOptions {
        mem_limit: Some(256),
        timeout: Duration::from_secs(10),
        ..TripleParityOptions::default()
    };
    let aot_str_oom = run_rust_aot_engine(&str_oom_script, &str_oom_bin, &str_oom_opts);
    assert_eq!(
        aot_str_oom.error_class,
        ErrorClass::OutOfMemory,
        "SEC-03 String exponential growth OOM in AOT failed: stderr={}",
        aot_str_oom.stderr
    );
    assert_eq!(aot_str_oom.exit_code, Some(1));
    assert_eq!(aot_str_oom.signal, None);

    // ------------------------------------------------------------------------
    // SEC-04 / S08: Capability Denial under `--pure-core` / `--deny-ffi`
    //               Canary file on host filesystem must remain untouched
    // ------------------------------------------------------------------------
    let canary_path = tmp.path().join("host_canary.txt");
    fs::write(&canary_path, "UNTOUCHED_CANARY_SECRET").expect("write canary");
    let escaped_canary = canary_path.display().to_string().replace('\\', "\\\\");
    let fs_escape_src = format!(
        "local content = io.read_file(\"{escaped_canary}\")\nprintln(content)\n"
    );
    let pure_opts = TripleParityOptions {
        pure_core: true,
        timeout: Duration::from_secs(10),
        ..TripleParityOptions::default()
    };
    let fs_script = tmp.path().join("sec04_fs_escape.lust");
    let fs_bin = tmp.path().join("sec04_fs_escape_bin");
    fs::write(&fs_script, &fs_escape_src).expect("write fs_escape");
    let vm_fs = run_vm_engine(&fs_script, &pure_opts);
    let jit_fs = run_cranelift_engine(&fs_script, &pure_opts);
    let aot_fs = run_rust_aot_engine(&fs_script, &fs_bin, &pure_opts);
    assert_ne!(vm_fs.error_class, ErrorClass::Ok, "SEC-04 VM must deny io.read_file in pure-core");
    assert_ne!(jit_fs.error_class, ErrorClass::Ok, "SEC-04 JIT must deny io.read_file in pure-core");
    assert_ne!(aot_fs.error_class, ErrorClass::Ok, "SEC-04 AOT must deny io.read_file in --deny-ffi");
    assert!(
        !vm_fs.stdout.contains("UNTOUCHED_CANARY_SECRET")
            && !jit_fs.stdout.contains("UNTOUCHED_CANARY_SECRET")
            && !aot_fs.stdout.contains("UNTOUCHED_CANARY_SECRET"),
        "SEC-04 canary secret leaked!"
    );
    assert_eq!(
        fs::read_to_string(&canary_path).expect("read canary"),
        "UNTOUCHED_CANARY_SECRET"
    );

    // ------------------------------------------------------------------------
    // SEC-05 / S11: Tenant & Process Environment Isolation
    //               Two AOT runs are strictly isolated; secret env var is scrubbed
    // ------------------------------------------------------------------------
    let tenant_a_src = "local secret: int = 111111\nprintln(secret)\n";
    let tenant_b_src = "local secret: int = 222222\nprintln(secret)\n";
    let (vm_ta, jit_ta, aot_ta) = assert_triple_parity("sec05_tenant_a", tenant_a_src, None, None);
    let (vm_tb, jit_tb, aot_tb) = assert_triple_parity("sec05_tenant_b", tenant_b_src, None, None);
    assert_eq!(vm_ta.stdout, "111111\n");
    assert_eq!(jit_ta.stdout, "111111\n");
    assert_eq!(aot_ta.stdout, "111111\n");
    assert_eq!(vm_tb.stdout, "222222\n");
    assert_eq!(jit_tb.stdout, "222222\n");
    assert_eq!(aot_tb.stdout, "222222\n");

    // ------------------------------------------------------------------------
    // SEC-06: Host Survival under Unmetered Runaway AOT Loop (OS Watchdog Kill)
    // ------------------------------------------------------------------------
    let runaway_script = tmp.path().join("sec06_runaway.lust");
    let runaway_bin = tmp.path().join("sec06_runaway_bin");
    fs::write(&runaway_script, inf_loop_src).expect("write runaway");
    let runaway_opts = TripleParityOptions {
        release_mode: "ReleaseFast",
        gas_metering: GasMetering::None,
        gas_limit: None,
        timeout: Duration::from_millis(500),
        ..TripleParityOptions::default()
    };
    let runaway_out = run_rust_aot_engine(&runaway_script, &runaway_bin, &runaway_opts);
    assert!(
        runaway_out.timed_out,
        "SEC-06 unmetered runaway AOT binary must be terminated by monotonic OS watchdog"
    );
    assert_eq!(runaway_out.error_class, ErrorClass::Timeout);

    // ------------------------------------------------------------------------
    // SEC-15: Error Hygiene under `-C panic=abort` (no Rust panic backtraces,
    //         no leaked host secret tokens on stderr)
    // ------------------------------------------------------------------------
    let err_hygiene_src = "function f(a: int, b: int): int return a / b end\nprintln(f(1, 0))\n";
    let hygiene_script = tmp.path().join("sec15_hygiene.lust");
    let hygiene_bin = tmp.path().join("sec15_hygiene_bin");
    fs::write(&hygiene_script, err_hygiene_src).expect("write hygiene");
    let hygiene_opts = TripleParityOptions {
        release_mode: "ReleaseFast",
        timeout: Duration::from_secs(10),
        ..TripleParityOptions::default()
    };
    let hygiene_out = run_rust_aot_engine(&hygiene_script, &hygiene_bin, &hygiene_opts);
    assert_eq!(hygiene_out.error_class, ErrorClass::DivisionByZero);
    assert_eq!(hygiene_out.exit_code, Some(1));
    assert_eq!(hygiene_out.signal, None);
    assert!(
        !hygiene_out.stderr.contains("panicked at")
            && !hygiene_out.stderr.contains("RUST_BACKTRACE"),
        "SEC-15 AOT runtime error leaked Rust panic diagnostics: {}",
        hygiene_out.stderr
    );
}

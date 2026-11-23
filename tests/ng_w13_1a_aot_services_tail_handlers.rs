//! Integration & parity test suite for `NG-W13-1A`:
//! - Direct AST forwarding (`ast_items: &[Item]`) into `emit_rust_from_mir_with_ast` (`F-19`)
//! - Nominal services (`service` / `provide`) with safe `Default::default()` ZST providers (`F-11`)
//! - Tail-resumptive & abortive `handle..with` / `perform` / `resume` compiled to stack
//!   `EffectHandlerFrame` + `core::mem::DropGuard` in `lust-aot-rt::effects` (`TASK_ALLOWS_UNSAFE=0`).

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::tempdir;

use lust::ast::{ExprKind, Item, ItemKind, StmtKind};
use lust::bytecode::Value;
use lust::codegen_zig::aot::rust_emitter::{
    HandlerArmTailKind, classify_handler_arm_expr, emit_rust_from_mir_with_ast,
};
use lust::codegen_zig::aot::{
    AotCompileOptions, AotCompiler, AotTargetKind, RustCompileOptions, ensure_cached_aot_rt_rlib,
    prepare_items_for_mir,
};
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::lowering::AstLowering;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{VM, reset_effect_handler_state};
use lust::LustConfig;

fn parse_items(source: &str) -> Vec<Item> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("tokenize");
    Parser::new(tokens).parse().expect("parse")
}

fn run_vm_fn(source: &str, fn_name: &str) -> Value {
    reset_effect_handler_state();
    let items = parse_items(source);
    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let mut compiler = lust::Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    let funcs = compiler.compile_module(&items).expect("compile vm module");
    let mut vm = VM::new();
    vm.load_functions(funcs);
    vm.call(fn_name, vec![]).expect("execute vm function")
}

fn compile_and_run_rust_aot(
    source: &str,
    cache_dir: &Path,
) -> (String, std::process::Output) {
    let tmp = tempdir().expect("tempdir");
    let opts = RustCompileOptions {
        cache_dir: Some(cache_dir.to_path_buf()),
        ..RustCompileOptions::default()
    };
    let emitted = AotCompiler::emit_rust_source(source, &opts).expect("emit_rust_source");
    assert_zero_unsafe_and_panic_free(&emitted);

    let bin = tmp.path().join("aot_bin");
    AotCompiler::compile_source_rust(source, &bin, &opts).expect("compile_source_rust");
    let run = Command::new(&bin).output().expect("execute aot binary");
    (emitted, run)
}

fn assert_zero_unsafe_and_panic_free(code: &str) {
    assert!(
        !code.contains("unsafe"),
        "emitted Rust must contain ZERO `unsafe` blocks (TASK_ALLOWS_UNSAFE=0):\n{code}"
    );
    assert!(
        !code.contains("conjure_zst"),
        "emitted Rust must NEVER call `conjure_zst` (F-11):\n{code}"
    );
    assert!(
        !code.contains(".unwrap("),
        "emitted Rust must contain ZERO `.unwrap(` calls (Hazard 9):\n{code}"
    );
    assert!(
        !code.contains("panic!("),
        "emitted Rust must contain ZERO `panic!(` calls (Hazard 9):\n{code}"
    );
}

fn rustc_bin() -> String {
    std::env::var("RUSTC")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "rustc".to_string())
}

// ---------------------------------------------------------------------------
// Test 1: Nominal Services (`service` / `trait` / `impl` / `~ Service`) via
//         direct AST forwarding and safe `Default::default()` ZST providers.
// ---------------------------------------------------------------------------

#[test]
fn test_nominal_services_safe_zst_default_providers_and_host_execution() {
    let src = r#"
trait Storage
    function fetch(self, key: int): int
    function store(self, key: int, val: int): int
end

struct ZeroStorage
end

impl Storage for ZeroStorage
    function fetch(self, key: int): int
        return key * 10 + 7
    end
    function store(self, key: int, val: int): int
        return key + val
    end
end

struct Database
    shard: int
end

impl Database
    function read_row(self, id: int): int
        return self.shard * 100 + id
    end
end

service AppStorage: Storage
service PrimaryDb: Database
service ReplicaDb: Database
service PureClock

function sync_pair(x: int): int ~ (PrimaryDb + ReplicaDb)
    return x + 1
end
"#;

    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");

    // 1A. Verify direct `emit_rust_from_mir_with_ast` API without modifying `src/mir/ir.rs`
    let items = parse_items(src);
    let prepared = prepare_items_for_mir(&items);
    let mir_module = AstLowering::lower_module_items("services_mod", &prepared).expect("lower");
    let lib_opts = RustCompileOptions {
        aot: AotCompileOptions {
            target_kind: AotTargetKind::StaticLibrary,
            ..AotCompileOptions::default()
        },
        cache_dir: Some(cache_dir.clone()),
        ..RustCompileOptions::default()
    };
    let emitted =
        emit_rust_from_mir_with_ast(&mir_module, &items, &lib_opts.to_emitter_options())
            .expect("emit_rust_from_mir_with_ast");
    assert_zero_unsafe_and_panic_free(&emitted);

    for expected in [
        "pub trait LustService_AppStorage",
        "pub struct LustDefaultProvider_AppStorage;",
        "pub trait LustService_PrimaryDb",
        "pub struct LustDefaultProvider_PrimaryDb;",
        "pub trait LustService_ReplicaDb",
        "pub struct LustDefaultProvider_ReplicaDb;",
        "pub trait LustService_PureClock",
        "pub struct LustDefaultProvider_PureClock;",
        "#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]",
        "core::mem::size_of::<LustDefaultProvider_AppStorage>() == 0",
        "core::mem::size_of::<LustDefaultProvider_PureClock>() == 0",
        "lust_aot_rt::default_zst_provider::<Self>()",
        "Nominal service requirements (NG-W13-1A): LustService_PrimaryDb + LustService_ReplicaDb",
    ] {
        assert!(
            emitted.contains(expected),
            "expected `{expected}` in emitted Rust:\n{emitted}"
        );
    }

    // 1B. Compile emitted library + host driver calling `Default::default()` ZST providers
    let rlib = ensure_cached_aot_rt_rlib(&lib_opts).expect("build lust-aot-rt rlib");
    let mod_rs = tmp.path().join("services_generated.rs");
    // Strip crate-level `#![allow(...)]` / `extern crate` when embedding as module inside host harness
    let driver_rs = tmp.path().join("driver.rs");
    fs::write(&mod_rs, &emitted).expect("write generated rlib src");

    let gen_rlib = tmp.path().join("libservices_generated.rlib");
    let out_lib = Command::new(rustc_bin())
        .arg("--edition=2024")
        .arg("--crate-name")
        .arg("services_generated")
        .arg("--crate-type=rlib")
        .arg("-C")
        .arg("panic=abort")
        .arg("--extern")
        .arg(format!("lust_aot_rt={}", rlib.display()))
        .arg(&mod_rs)
        .arg("-o")
        .arg(&gen_rlib)
        .output()
        .expect("compile services_generated rlib");
    assert!(
        out_lib.status.success(),
        "failed to compile emitted nominal services rlib:\n{}\n---\n{emitted}",
        String::from_utf8_lossy(&out_lib.stderr)
    );

    let driver_code = r#"
extern crate lust_aot_rt;
extern crate services_generated;

use lust_aot_rt::AotCtx;
use services_generated::{
    LustDefaultProvider_AppStorage, LustDefaultProvider_PrimaryDb, LustDefaultProvider_PureClock,
    LustDefaultProvider_ReplicaDb, LustService_AppStorage, LustService_PrimaryDb,
    LustService_PureClock, LustService_ReplicaDb, LustStruct_Database,
};

fn main() {
    assert_eq!(core::mem::size_of::<LustDefaultProvider_AppStorage>(), 0);
    assert_eq!(core::mem::size_of::<LustDefaultProvider_PrimaryDb>(), 0);
    assert_eq!(core::mem::size_of::<LustDefaultProvider_ReplicaDb>(), 0);
    assert_eq!(core::mem::size_of::<LustDefaultProvider_PureClock>(), 0);

    let mut ctx = AotCtx::new();

    // Safe Default::default() ZST provider instantiation (zero unsafe!)
    let mut storage = LustDefaultProvider_AppStorage::default();
    assert_eq!(storage.fetch(&mut ctx, 5).expect("fetch"), 57);
    assert_eq!(storage.store(&mut ctx, 3, 4).expect("store"), 7);
    assert_eq!(storage.service_id(&mut ctx).expect("id"), "AppStorage");

    // Nominal distinctness between PrimaryDb and ReplicaDb targeting same struct Database
    let mut primary = LustStruct_Database { field_0: 2 };
    let mut replica = LustStruct_Database { field_0: 9 };
    assert_eq!(<LustStruct_Database as LustService_PrimaryDb>::service_id(&primary, &mut ctx).expect("p id"), "PrimaryDb");
    assert_eq!(<LustStruct_Database as LustService_ReplicaDb>::service_id(&replica, &mut ctx).expect("r id"), "ReplicaDb");
    assert_eq!(<LustStruct_Database as LustService_PrimaryDb>::read_row(&mut primary, &mut ctx, 15).expect("p read"), 215);
    assert_eq!(<LustStruct_Database as LustService_ReplicaDb>::read_row(&mut replica, &mut ctx, 15).expect("r read"), 915);

    // Pure stateless ZST service
    let clock = LustDefaultProvider_PureClock::instantiate();
    let clock_copy = clock;
    assert_eq!(clock, clock_copy);
    assert_eq!(clock.service_id(&mut ctx).expect("clock id"), "PureClock");

    let mut def_primary = LustDefaultProvider_PrimaryDb::default();
    assert_eq!(def_primary.read_row(&mut ctx, 42).expect("def read"), 42);

    println!("SERVICES_OK");
}
"#;
    fs::write(&driver_rs, driver_code).expect("write driver");
    let driver_bin = tmp.path().join("services_driver");
    let out_drv = Command::new(rustc_bin())
        .arg("--edition=2024")
        .arg("-C")
        .arg("panic=abort")
        .arg("--extern")
        .arg(format!("lust_aot_rt={}", rlib.display()))
        .arg("--extern")
        .arg(format!("services_generated={}", gen_rlib.display()))
        .arg(&driver_rs)
        .arg("-o")
        .arg(&driver_bin)
        .output()
        .expect("compile services driver");
    assert!(
        out_drv.status.success(),
        "failed to compile services driver:\n{}",
        String::from_utf8_lossy(&out_drv.stderr)
    );

    let run = Command::new(&driver_bin)
        .output()
        .expect("run services driver");
    assert!(
        run.status.success(),
        "services driver failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(String::from_utf8(run.stdout).expect("utf8"), "SERVICES_OK\n");
}

// ---------------------------------------------------------------------------
// Test 2: `provide` + `run_sync` end-to-end execution parity with VM
// ---------------------------------------------------------------------------

#[test]
fn test_provide_and_run_sync_end_to_end_parity() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");

    let src = r#"
struct Env
    seed: int
end

service AppEnv: Env

function run_with_env(e: Effect<int, never>, env: Env): int
    return e:provide(env):run_sync()
end

function main(): int
    local env: Env = Env { seed = 40 }
    local direct: int = (effect do
        return env.seed + 2
    end):provide(env):run_sync()
    local dot_res: int = (effect do
        return env.seed + 10
    end).provide(env).run_sync()
    local e = effect do
        return env.seed + 100
    end
    local via_param: int = run_with_env(e, env)
    println(direct)
    println(dot_res)
    println(via_param)
    return 0
end
"#;

    let (emitted, run) = compile_and_run_rust_aot(src, &cache_dir);
    assert!(
        run.status.success(),
        "AOT provide/run_sync failed:\nstderr: {}\n---\n{emitted}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(
        emitted.contains("lust_aot_rt::with_provided_service"),
        "expected `with_provided_service` in emitted Rust:\n{emitted}"
    );
    let aot_stdout = String::from_utf8(run.stdout).expect("utf8");
    assert_eq!(aot_stdout, "42\n50\n140\n");

    // Cross-check against Bytecode VM CLI (`lust run --backend vm --jit=off`) on the same colon syntax
    let vm_script = tmp.path().join("provide_vm_oracle.lust");
    fs::write(
        &vm_script,
        r#"
struct Env
    seed: int
end
local env: Env = Env { seed = 40 }
local direct: int = (effect do
    return env.seed + 2
end):provide(env):run_sync()
println(direct)
"#,
    )
    .expect("write provide_vm_oracle");
    let vm_out = Command::new(env!("CARGO_BIN_EXE_lust"))
        .arg("run")
        .arg(&vm_script)
        .arg("--backend")
        .arg("vm")
        .arg("--jit=off")
        .output()
        .expect("run lust vm");
    assert!(
        vm_out.status.success(),
        "VM oracle failed: {}",
        String::from_utf8_lossy(&vm_out.stderr)
    );
    assert_eq!(String::from_utf8(vm_out.stdout).expect("utf8"), "42\n");
}

// ---------------------------------------------------------------------------
// Test 3: Tail-resumptive `handle..with` (single, sequential, stateful, string)
//         parity with Bytecode VM
// ---------------------------------------------------------------------------

#[test]
fn test_tail_resumptive_handlers_vm_parity_and_stateful_captures() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");

    let src = r#"
function test_basic(): int
    handle {
        let x: int = perform("GetVal", []);
        x + 10
    } with {
        GetVal() => resume(42)
    }
end

function test_sequential(): int
    handle {
        let a = perform("Step", [1]);
        let b = perform("Step", [2]);
        let c = perform("Step", [3]);
        a + b + c
    } with {
        Step(x) => resume(x * 10)
    }
end

function test_stateful(): int
    local state: int = 10
    return handle {
        let a = perform("Get", []);
        perform("Put", [a + 5]);
        let b = perform("Get", []);
        perform("Put", [b * 2]);
        perform("Get", [])
    } with {
        Get() => resume(state),
        Put(v) => {
            state = v;
            resume(0)
        }
    }
end

function test_string_effect(): string
    return handle {
        let msg: string = perform("AskMsg", []);
        msg .. "!"
    } with {
        AskMsg() => resume("hello")
    }
end

function main(): int
    println(test_basic())
    println(test_sequential())
    println(test_stateful())
    println(test_string_effect())
    return 0
end
"#;

    let (emitted, run) = compile_and_run_rust_aot(src, &cache_dir);
    assert!(
        run.status.success(),
        "tail-resumptive AOT binary failed:\nstderr: {}\n---\n{emitted}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(
        emitted.contains("lust_aot_rt::handle_tail_resumptive_typed"),
        "expected stack handle_tail_resumptive_typed:\n{emitted}"
    );
    assert!(
        emitted.contains("lust_aot_rt::perform_effect"),
        "expected perform_effect:\n{emitted}"
    );
    assert!(
        emitted.contains("lust_aot_rt::resume_effect"),
        "expected resume_effect:\n{emitted}"
    );

    let aot_stdout = String::from_utf8(run.stdout).expect("utf8");
    assert_eq!(aot_stdout, "52\n60\n30\nhello!\n");

    // Verify exact parity with Bytecode VM on basic & sequential cases
    assert_eq!(run_vm_fn(src, "test_basic"), Value::Int(52));
    assert_eq!(run_vm_fn(src, "test_sequential"), Value::Int(60));
}

// ---------------------------------------------------------------------------
// Test 4: Nested handlers, forwarding, abortive handlers, and `defer` unwinding
// ---------------------------------------------------------------------------

#[test]
fn test_nested_forwarding_abortive_handlers_and_defer_unwind() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");

    let src = r#"
function helper_abort(): int
    defer println("helper_defer_ran")
    let x: int = perform("AbortOuter", [404]);
    return x + 9999
end

function test_nested_forward(): int
    handle {
        handle {
            let inner_val = perform("Inner", [5]);
            let outer_val = perform("Outer", [100]);
            inner_val + outer_val
        } with {
            Inner(x) => resume(x * 2)
        }
    } with {
        Outer(y) => resume(y + 1)
    }
end

function test_inner_abort(): int
    let res = handle {
        perform("AbortInner", []);
        10
    } with {
        AbortInner() => 99
    };
    return res + 1
end

function test_outer_abort_through_inner_handler(): int
    return handle {
        handle {
            helper_abort()
        } with {
            Unrelated() => resume(1)
        }
    } with {
        AbortOuter(code) => -code
    }
end

function main(): int
    println(test_nested_forward())
    println(test_inner_abort())
    println(test_outer_abort_through_inner_handler())
    return 0
end
"#;

    let (emitted, run) = compile_and_run_rust_aot(src, &cache_dir);
    assert!(
        run.status.success(),
        "nested/abort AOT failed:\nstderr: {}\n---\n{emitted}",
        String::from_utf8_lossy(&run.stderr)
    );
    let aot_stdout = String::from_utf8(run.stdout).expect("utf8");
    assert_eq!(aot_stdout, "111\n100\nhelper_defer_ran\n-404\n");

    // Cross-check nested forwarding + inner abort against Bytecode VM
    assert_eq!(run_vm_fn(src, "test_nested_forward"), Value::Int(111));
    assert_eq!(run_vm_fn(src, "test_inner_abort"), Value::Int(100));
}

// ---------------------------------------------------------------------------
// Test 5: Unhandled effect & double-resume error parity (isolated child procs)
// ---------------------------------------------------------------------------

#[test]
fn test_effect_runtime_error_parity_in_child_processes() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");

    // 5A. Unhandled effect returns exit code 1 + structured error
    let unhandled_src = r#"
function main(): int
    return handle {
        perform("NonExistent", []);
        10
    } with {
        Other() => resume(1)
    }
end
"#;
    let (_emitted_u, run_u) = compile_and_run_rust_aot(unhandled_src, &cache_dir);
    assert!(!run_u.status.success());
    assert_eq!(run_u.status.code(), Some(1));
    let err_u = String::from_utf8_lossy(&run_u.stderr);
    assert!(
        err_u.contains("Unhandled effect: NonExistent"),
        "expected Unhandled effect error, got: {err_u}"
    );

    // 5B. Double resumption returns exit code 1 + structured error
    let dbl_src = r#"
function main(): int
    return handle {
        perform("Ask", []);
        10
    } with {
        Ask() => {
            resume(1);
            resume(2)
        }
    }
end
"#;
    let (_emitted_d, run_d) = compile_and_run_rust_aot(dbl_src, &cache_dir);
    assert!(!run_d.status.success());
    assert_eq!(run_d.status.code(), Some(1));
    let err_d = String::from_utf8_lossy(&run_d.stderr);
    assert!(
        err_d.contains("Effect already resumed"),
        "expected Effect already resumed error, got: {err_d}"
    );

    // 5C. Genuine TypeChecker errors still fail closed even when effects/services are used
    let opts = RustCompileOptions {
        cache_dir: Some(cache_dir.to_path_buf()),
        ..RustCompileOptions::default()
    };
    let bad_resume = "function main(): int\n    resume(42);\n    return 0\nend\n";
    let err_resume = AotCompiler::emit_rust_source(bad_resume, &opts)
        .expect_err("resume outside arm must fail typechecking");
    assert!(err_resume.to_string().contains("outside of an active effect handler arm"));

    let bad_arm = "function main(): int\n    handle 100 with {\n        Abort() => \"bad\"\n    }\nend\n";
    let err_arm = AotCompiler::emit_rust_source(bad_arm, &opts)
        .expect_err("mismatched arm return type must fail typechecking");
    assert!(err_arm.to_string().contains("incompatible with handle body return type"));

    let bad_svc = r#"
struct Fs end
service FileSystem: Fs
function read_file(p: string): string ! string ~ FileSystem
    return p
end
function pure_caller(): string
    return read_file("x")
end
"#;
    let err_svc = AotCompiler::emit_rust_source(bad_svc, &opts)
        .expect_err("unsatisfied service requirement must fail typechecking");
    assert!(err_svc.to_string().contains("Unsatisfied requirement: 'FileSystem'"));
}

// ---------------------------------------------------------------------------
// Test 6: AST tail-resumptive arm classifier & `#![no_std]` library emission
// ---------------------------------------------------------------------------

#[test]
fn test_ast_tail_resumptive_classifier_and_nostd_emission() {
    let src = r#"
function main(): int
    return handle {
        let a = perform("TailOp", [1]);
        let b = perform("AbortOp", [2]);
        let c = perform("NonTailOp", [3]);
        a + b + c
    } with {
        TailOp(x) => resume(x + 1),
        AbortOp(y) => -y,
        NonTailOp(z) => {
            let r = resume(z);
            r + 100
        }
    }
end
"#;
    let items = parse_items(src);
    let ItemKind::Function(f) = &items[0].kind else {
        panic!("expected function");
    };
    let StmtKind::Return(vals) = &f.body[0].kind else {
        panic!("expected return");
    };
    let ExprKind::Call { args, .. } = &vals[0].kind else {
        panic!("expected __lust_handle call");
    };
    let ExprKind::Array(tuples) = &args[1].kind else {
        panic!("expected arms array");
    };
    assert_eq!(tuples.len(), 3);

    let mut kinds = Vec::new();
    for tup in tuples {
        let ExprKind::Tuple(elems) = &tup.kind else {
            panic!("expected arm tuple");
        };
        let ExprKind::Lambda { body, .. } = &elems[1].kind else {
            panic!("expected arm lambda");
        };
        kinds.push(classify_handler_arm_expr(body));
    }
    assert_eq!(
        kinds,
        vec![
            HandlerArmTailKind::TailResumptive,
            HandlerArmTailKind::Abortive,
            HandlerArmTailKind::NonTailOrMultiShot,
        ]
    );

    // Verify `#![no_std]` emission with nominal services + tail-resumptive handle compiles via rustc
    let nostd_src = r#"
service Telemetry

function compute(): int ~ Telemetry
    return handle {
        let x: int = perform("Metric", [21]);
        x * 2
    } with {
        Metric(v) => resume(v)
    }
end
"#;
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache_nostd");
    let nostd_opts = RustCompileOptions {
        aot: AotCompileOptions {
            target_kind: AotTargetKind::StaticLibrary,
            ..AotCompileOptions::default()
        },
        no_std: true,
        cache_dir: Some(cache_dir),
        ..RustCompileOptions::default()
    };
    let nostd_code =
        AotCompiler::emit_rust_source(nostd_src, &nostd_opts).expect("emit nostd rust");
    assert_zero_unsafe_and_panic_free(&nostd_code);
    assert!(nostd_code.contains("#![no_std]"));
    assert!(nostd_code.contains("pub struct LustDefaultProvider_Telemetry;"));

    let nostd_rlib = ensure_cached_aot_rt_rlib(&nostd_opts).expect("build nostd lust-aot-rt rlib");
    let nostd_rs = tmp.path().join("nostd_effects.rs");
    let nostd_out_rlib = tmp.path().join("libnostd_effects.rlib");
    fs::write(&nostd_rs, &nostd_code).expect("write nostd_effects.rs");
    let check_out = Command::new(rustc_bin())
        .arg("--edition=2024")
        .arg("--crate-name")
        .arg("nostd_effects")
        .arg("--crate-type=rlib")
        .arg("-C")
        .arg("panic=abort")
        .arg("--extern")
        .arg(format!("lust_aot_rt={}", nostd_rlib.display()))
        .arg(&nostd_rs)
        .arg("-o")
        .arg(&nostd_out_rlib)
        .output()
        .expect("compile nostd_effects rlib");
    assert!(
        check_out.status.success(),
        "no_std Rust AOT effects/services library failed to compile:\n{}\n---\n{nostd_code}",
        String::from_utf8_lossy(&check_out.stderr)
    );
}

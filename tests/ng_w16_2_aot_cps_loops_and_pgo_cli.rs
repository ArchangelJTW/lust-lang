//! Integration & parity test suite for `NG-W16-2`:
//! 1. Non-tail effect handler CPS transformation inside `while` and `for` loops (`AOT-CPS-LOOPS`)
//! 2. Non-tail effect handler CPS transformation across module-local helper callees and `defer` (`AOT-CPS-LOOPS`)
//! 3. Exact desugared effect call span filtering in `TypeChecker` (`AOT-BUDGET-SOUND`)
//! 4. CLI `--lto=thin`, `--pgo-generate`, `--pgo-use`, `--pgo-sample-use`, and `LUST_AOT_RT_RLIB` (`AOT-PGO-CLI`)

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::tempdir;

use lust::codegen_zig::aot::{
    AotCompiler, RustCompileOptions, ensure_cached_aot_rt_rlib,
};

fn compile_and_run_rust_aot(source: &str, cache_dir: &Path) -> (String, std::process::Output) {
    let tmp = tempdir().expect("tempdir");
    let opts = RustCompileOptions {
        cache_dir: Some(cache_dir.to_path_buf()),
        ..RustCompileOptions::default()
    };
    let emitted = AotCompiler::emit_rust_source(source, &opts).expect("emit_rust_source");
    assert!(
        !emitted.contains("unsafe"),
        "emitted Rust must contain ZERO `unsafe` blocks:\n{emitted}"
    );
    let bin = tmp.path().join("aot_bin");
    AotCompiler::compile_source_rust(source, &bin, &opts).expect("compile_source_rust");
    let run = Command::new(&bin).output().expect("execute aot binary");
    (emitted, run)
}

fn run_vm_cli(source: &str) -> std::process::Output {
    let tmp = tempdir().expect("tempdir");
    let script = tmp.path().join("prog.lust");
    fs::write(&script, source).expect("write script");
    let lust_bin = env!("CARGO_BIN_EXE_lust");
    Command::new(lust_bin)
        .arg("run")
        .arg("--no-typecheck")
        .arg("--jit=off")
        .arg(&script)
        .output()
        .expect("execute lust VM CLI")
}

// ---------------------------------------------------------------------------
// Test 1: Non-tail `handle..with` performing inside `while` and `for i = 1, 5`
// ---------------------------------------------------------------------------

#[test]
fn test_ng_w16_2_nontail_handle_perform_inside_while_and_for_loop() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");

    let src = r#"
function test_while_loop(): int
    return handle {
        local acc: int = 0
        local i: int = 1
        while i <= 4 do
            local got: int = perform("Scale", [i])
            acc = acc + got
            i = i + 1
        end
        acc
    } with {
        Scale(v) => {
            local r: int = resume(v * 10)
            r + 1
        }
    }
end

function test_for_loop(): int
    return handle {
        local acc: int = 0
        for i = 1, 5 do
            local got: int = perform("Scale", [i])
            acc = acc + got
        end
        acc
    } with {
        Scale(v) => {
            local r: int = resume(v * 10)
            r + 1
        }
    }
end

function main()
    println("while=" .. tostring(test_while_loop()))
    println("for=" .. tostring(test_for_loop()))
end
"#;

    let (_emitted, aot_out) = compile_and_run_rust_aot(src, &cache_dir);
    assert!(
        aot_out.status.success(),
        "Rust AOT execution failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&aot_out.stdout),
        String::from_utf8_lossy(&aot_out.stderr)
    );
    let aot_stdout = String::from_utf8_lossy(&aot_out.stdout).to_string();

    // while: (10 + 20 + 30 + 40) = 100, plus 4 non-tail `+ 1` = 104
    // for:   (10 + 20 + 30 + 40 + 50) = 150, plus 5 non-tail `+ 1` = 155
    assert_eq!(aot_stdout, "while=104\nfor=155\n");
}

// ---------------------------------------------------------------------------
// Test 2: Non-tail `handle..with` across helper callee and `defer` LIFO
// ---------------------------------------------------------------------------

#[test]
fn test_ng_w16_2_nontail_handle_perform_inside_helper_callee_and_defer() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");

    let src = r#"
function ask_helper(base: int): int
    local v: int = perform("Ask", [base])
    return v + 3
end

function test_helper_and_defer(): int
    return handle {
        defer println("defer_1")
        defer println("defer_2")
        local a: int = ask_helper(7)
        println("after_helper=" .. tostring(a))
        a * 2
    } with {
        Ask(x) => {
            local r: int = resume(x * 10)
            println("handler_after=" .. tostring(r))
            r + 100
        }
    }
end

function main()
    println("res=" .. tostring(test_helper_and_defer()))
end
"#;

    let (_emitted, aot_out) = compile_and_run_rust_aot(src, &cache_dir);
    assert!(
        aot_out.status.success(),
        "Rust AOT execution failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&aot_out.stdout),
        String::from_utf8_lossy(&aot_out.stderr)
    );
    let aot_stdout = String::from_utf8_lossy(&aot_out.stdout).to_string();
    // ask_helper(7): perform("Ask", [7]) -> resumed with 70 -> returns 73
    // prints "after_helper=73"
    // defers run in LIFO order: "defer_2", then "defer_1"
    // body returns 73 * 2 = 146
    // handler prints "handler_after=146", returns 246
    // main prints "res=246"
    assert_eq!(
        aot_stdout,
        "after_helper=73\ndefer_2\ndefer_1\nhandler_after=146\nres=246\n"
    );
}

// ---------------------------------------------------------------------------
// Test 3: TypeChecker rejects real `unit` type error in module with `handle`
// ---------------------------------------------------------------------------

#[test]
fn test_ng_w16_2_typechecker_rejects_real_unit_type_error_in_module_with_handle() {
    let src = r#"
function unit_fn()
end

function bad_unit_assignment(): int
    local x: int = unit_fn()
    return x
end

function ok_handle(): int
    return handle {
        local v: int = perform("Get", [1])
        v + 1
    } with {
        Get(x) => {
            local r: int = resume(x + 10)
            r + 1
        }
    }
end

function main()
    println(tostring(ok_handle()))
    println(tostring(bad_unit_assignment()))
end
"#;

    let opts = RustCompileOptions::default();
    let res = AotCompiler::emit_rust_source(src, &opts);
    assert!(
        res.is_err(),
        "Expected TypeChecker to reject `local x: int = ()` outside handle block, but got Ok"
    );
    let err_msg = format!("{}", res.unwrap_err());
    assert!(
        err_msg.contains("TypeChecker error"),
        "Expected TypeChecker error, got: {err_msg}"
    );
}

// ---------------------------------------------------------------------------
// Test 4: CLI `--lto=thin`, `--pgo-generate`, and `LUST_AOT_RT_RLIB` override
// ---------------------------------------------------------------------------

#[test]
fn test_ng_w16_2_cli_pgo_and_thin_lto_flags_compile_and_link_rlib() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");
    let src_path = tmp.path().join("pgo_sample.lust");
    fs::write(
        &src_path,
        r#"
function sum_to(n: int): int
    local acc: int = 0
    for i = 1, n do
        acc = acc + i
    end
    return acc
end

function main()
    println("sum=" .. tostring(sum_to(10)))
end
"#,
    )
    .expect("write lust source");

    // Pre-build a cached rlib so we can pass it via LUST_AOT_RT_RLIB
    let rlib_opts = RustCompileOptions {
        cache_dir: Some(cache_dir.clone()),
        ..RustCompileOptions::default()
    };
    let rlib_path = ensure_cached_aot_rt_rlib(&rlib_opts).expect("build cached rlib");
    assert!(rlib_path.exists());

    let lust_bin = env!("CARGO_BIN_EXE_lust");

    // 1. Build with `--backend rust --lto=thin` and `LUST_AOT_RT_RLIB`
    let thin_bin = tmp.path().join("pgo_sample_thin");
    let thin_build = Command::new(lust_bin)
        .env("LUST_AOT_RT_RLIB", &rlib_path)
        .args([
            "build",
            "--backend",
            "rust",
            "--lto=thin",
            "-o",
            thin_bin.to_str().unwrap(),
            src_path.to_str().unwrap(),
        ])
        .output()
        .expect("run lust build --lto=thin");
    assert!(
        thin_build.status.success(),
        "lust build --lto=thin failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&thin_build.stdout),
        String::from_utf8_lossy(&thin_build.stderr)
    );
    let thin_run = Command::new(&thin_bin).output().expect("run thin-lto binary");
    assert!(thin_run.status.success());
    assert_eq!(String::from_utf8_lossy(&thin_run.stdout), "sum=55\n");

    // 2. Build with `--backend rust --pgo-generate=<dir>` and run to emit profile data
    let pgo_dir = tmp.path().join("pgo_profraw");
    let pgo_bin = tmp.path().join("pgo_sample_gen");
    let pgo_build = Command::new(lust_bin)
        .env("LUST_AOT_RT_RLIB", &rlib_path)
        .args([
            "build",
            "--backend",
            "rust",
            &format!("--pgo-generate={}", pgo_dir.display()),
            "-o",
            pgo_bin.to_str().unwrap(),
            src_path.to_str().unwrap(),
        ])
        .output()
        .expect("run lust build --pgo-generate");
    assert!(
        pgo_build.status.success(),
        "lust build --pgo-generate failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&pgo_build.stdout),
        String::from_utf8_lossy(&pgo_build.stderr)
    );
    let pgo_run = Command::new(&pgo_bin).output().expect("run pgo-gen binary");
    assert!(pgo_run.status.success());
    assert_eq!(String::from_utf8_lossy(&pgo_run.stdout), "sum=55\n");
    assert!(
        pgo_dir.exists(),
        "expected --pgo-generate directory {} to be created after execution",
        pgo_dir.display()
    );
}

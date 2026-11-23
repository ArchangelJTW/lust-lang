//! Integration tests for Task NG-W12-1: `crates/lust-aot-rt` (`#![no_std] + alloc`,
//! zero external dependencies, direct `rustc --crate-type=rlib` + `--extern lust_aot_rt`
//! compilation under `-C panic=abort`, and Hazard 9 panic-free source audit).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn crate_root() -> PathBuf {
    repo_root().join("crates").join("lust-aot-rt")
}

/// Strips everything starting from `#[cfg(test)]` so Hazard 9 checks audit only runtime code.
fn strip_test_module(src: &str) -> &str {
    src.split("#[cfg(test)]").next().unwrap_or(src)
}

#[test]
fn test_crate_manifest_zero_deps_and_workspace_membership() {
    let root_toml = fs::read_to_string(repo_root().join("Cargo.toml"))
        .expect("read workspace Cargo.toml");
    assert!(
        root_toml.contains("\"crates/lust-aot-rt\""),
        "workspace Cargo.toml must include crates/lust-aot-rt in members"
    );

    let rt_toml = fs::read_to_string(crate_root().join("Cargo.toml"))
        .expect("read crates/lust-aot-rt/Cargo.toml");
    assert!(rt_toml.contains("name = \"lust-aot-rt\""));
    assert!(rt_toml.contains("edition = \"2024\""));
    assert!(rt_toml.contains("default = [\"std\"]"));
    assert!(rt_toml.contains("std = []"));

    // Verify `[dependencies]` section has zero external crates
    let deps_section = rt_toml
        .split("[dependencies]")
        .nth(1)
        .expect("[dependencies] section present")
        .split('[')
        .next()
        .unwrap_or("");
    let non_empty_dep_lines: Vec<&str> = deps_section
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    assert!(
        non_empty_dep_lines.is_empty(),
        "lust-aot-rt must have 0 external dependencies, found: {non_empty_dep_lines:?}"
    );
}

#[test]
fn test_hazard_9_no_panic_or_unwrap_and_confined_unsafe() {
    let files = [
        ("lib.rs", false),
        ("alloc.rs", true),
        ("value.rs", false),
    ];

    for (name, allows_unsafe) in files {
        let path = crate_root().join("src").join(name);
        let full_src = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        let runtime_src = strip_test_module(&full_src);

        // Strip `//` line comments before scanning code tokens
        let code_only: String = runtime_src
            .lines()
            .map(|line| line.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");

        for forbidden in ["panic!(", ".unwrap(", ".expect(", "todo!(", "unimplemented!("] {
            assert!(
                !code_only.contains(forbidden),
                "Hazard 9 violation in crates/lust-aot-rt/src/{name}: found forbidden `{forbidden}`"
            );
        }

        if !allows_unsafe {
            assert!(
                !code_only.contains("unsafe "),
                "Gate violation: unexpected `unsafe` outside alloc.rs in crates/lust-aot-rt/src/{name}"
            );
        }
    }

    let lib_src = fs::read_to_string(crate_root().join("src").join("lib.rs"))
        .expect("read lib.rs");
    assert!(
        lib_src.contains("#![no_std]"),
        "crates/lust-aot-rt/src/lib.rs must declare `#![no_std]`"
    );
}

fn compile_rlib(out_dir: &Path, enable_std: bool) -> PathBuf {
    let rlib_name = if enable_std {
        "liblust_aot_rt_std.rlib"
    } else {
        "liblust_aot_rt_nostd.rlib"
    };
    let rlib_path = out_dir.join(rlib_name);
    let mut cmd = Command::new("rustc");
    cmd.args([
        "--edition=2024",
        "--crate-type=rlib",
        "--crate-name=lust_aot_rt",
        "-C",
        "opt-level=2",
        "-C",
        "panic=abort",
    ]);
    if enable_std {
        cmd.args(["--cfg", "feature=\"std\""]);
    }
    cmd.arg(crate_root().join("src").join("lib.rs"))
        .arg("-o")
        .arg(&rlib_path);

    let output = cmd.output().expect("invoke rustc for lust-aot-rt rlib");
    assert!(
        output.status.success(),
        "rustc failed to build lust-aot-rt (std={enable_std}):\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(rlib_path.exists());
    rlib_path
}

#[test]
fn test_direct_rustc_rlib_nostd_and_panic_abort_linked_binary() {
    let tmp = tempfile::tempdir().expect("create tempdir");

    // 1. Build pure `#![no_std] + alloc` rlib (without feature="std")
    let nostd_rlib = compile_rlib(tmp.path(), false);

    // 2. Build default `feature="std"` rlib
    let std_rlib = compile_rlib(tmp.path(), true);

    // 3. Compile & execute a standalone AOT harness against the pure no_std rlib under `-C panic=abort`
    let driver_rs = tmp.path().join("aot_driver.rs");
    let driver_bin = tmp.path().join("aot_driver");
    let driver_src = r#"
use core::alloc::{Allocator, Layout};
use core::cell::RefCell;
use core::convert::Infallible;
use core::mem::DropGuard;
use lust_aot_rt::{
    AotCtx, AotError, AotKey, AotValue, LustBudgetAlloc, LustChunkArena, StaticSlabAllocator,
    DEFAULT_CHUNK_SIZE, defer, div_int, errdefer, floordiv_int, format_float, pow_int, rem_int,
    string_append_int,
};

fn infallible_add(a: i64, b: i64) -> Result<i64, Infallible> {
    Ok(a.wrapping_add(b))
}

fn run_scenario() -> Result<(), AotError> {
    // Gas metering & boundary
    let mut ctx = AotCtx::with_limits(Some(10), Some(1024));
    ctx.charge_gas(4)?;
    ctx.charge_gas(6)?;
    assert_eq!(ctx.gas_remaining, 0);
    assert_eq!(ctx.gas_used(), 10);
    assert_eq!(
        ctx.charge_gas(1),
        Err(AotError::OutOfGas { limit: 10, used: 11 })
    );

    // Infallible -> AotError via `?`
    let sum = infallible_add(20, 22)?;
    assert_eq!(sum, 42);

    // Integer intrinsics parity
    assert_eq!(div_int(i64::MIN, -1)?, i64::MIN);
    assert_eq!(rem_int(i64::MIN, -1)?, 0);
    assert_eq!(floordiv_int(-7, 2)?, -4);
    assert_eq!(floordiv_int(7, -2)?, -4);
    assert_eq!(pow_int(3, 5)?, 243);
    assert_eq!(div_int(1, 0), Err(AotError::DivisionByZero));
    assert_eq!(pow_int(2, -3), Err(AotError::NegativeExponent(-3)));

    // Fast string_append_int & format_float
    let mut buf = String::new();
    for i in 0..10 {
        string_append_int(&mut buf, i);
    }
    string_append_int(&mut buf, -98765);
    assert_eq!(buf, "0123456789-98765");
    assert_eq!(format_float(166666.66666658907), "166666.66666658907");

    // Allocators under panic=abort: LustBudgetAlloc, LustChunkArena, StaticSlabAllocator
    let alloc = LustBudgetAlloc::with_limit(512);
    {
        let mut v: Vec<u8, &LustBudgetAlloc> = Vec::new_in(&alloc);
        v.try_reserve_exact(256).map_err(|_| AotError::from_alloc_failure(&alloc, 256))?;
        for i in 0..256usize {
            v.push((i & 0xFF) as u8);
        }
        assert_eq!(alloc.used_bytes(), 256);
        assert!(v.try_reserve_exact(257).is_err());
        assert_eq!(alloc.used_bytes(), 256);
        assert_eq!(v[255], 255);
    }
    assert_eq!(alloc.used_bytes(), 0);

    let parent = LustBudgetAlloc::with_limit(DEFAULT_CHUNK_SIZE);
    {
        let arena = LustChunkArena::new(&parent);
        let node: Box<i64, &LustChunkArena<'_>> = Box::new_in(12345, &arena);
        assert_eq!(*node, 12345);
        assert_eq!(parent.used_bytes(), DEFAULT_CHUNK_SIZE);
    }
    assert_eq!(parent.used_bytes(), 0);

    let slab = StaticSlabAllocator::<1024>::new();
    let l16 = Layout::from_size_align(64, 16).ok().ok_or(AotError::SchemaDecodeError("layout"))?;
    let p = slab.allocate(l16).map_err(|_| AotError::OutOfMemory { limit_bytes: 1024, requested_bytes: 64 })?;
    assert_eq!(p.len(), 64);

    // defer / errdefer via core::mem::DropGuard under `-C panic=abort`
    let events: RefCell<Vec<&'static str>> = RefCell::new(Vec::new());
    let ok_fn = || -> Result<i64, AotError> {
        let _d = defer(|| events.borrow_mut().push("ok_defer"));
        let e = errdefer(|| events.borrow_mut().push("ok_errdefer_must_not_run"));
        DropGuard::dismiss(e);
        Ok(1)
    };
    assert_eq!(ok_fn()?, 1);
    assert_eq!(*events.borrow(), vec!["ok_defer"]);

    events.borrow_mut().clear();
    let err_fn = || -> Result<i64, AotError> {
        let _d = defer(|| events.borrow_mut().push("err_defer"));
        let e = errdefer(|| events.borrow_mut().push("err_errdefer"));
        let _ = div_int(1, 0)?;
        DropGuard::dismiss(e);
        Ok(2)
    };
    assert_eq!(err_fn(), Err(AotError::DivisionByZero));
    assert_eq!(*events.borrow(), vec!["err_errdefer", "err_defer"]);

    // AotValue dynamic operations
    let dyn_ctx = AotCtx::with_memory_limit(4096);
    let arr = AotValue::empty_array();
    arr.array_push(AotValue::Int(7), &dyn_ctx)?;
    assert_eq!(arr.array_get(0)?, AotValue::Int(7));

    let map = AotValue::map();
    map.map_set(AotKey::from("alpha"), AotValue::Bool(true), &dyn_ctx)?;
    assert_eq!(map.map_get(&AotKey::from("alpha"))?, AotValue::Bool(true));

    Ok(())
}

fn main() {
    if let Err(e) = run_scenario() {
        eprintln!("AOT runtime scenario failed: {e}");
        std::process::exit(1);
    }
    println!("LUST_AOT_RT_OK");
}
"#;
    fs::write(&driver_rs, driver_src).expect("write driver.rs");

    for (label, rlib) in [("nostd", &nostd_rlib), ("std", &std_rlib)] {
        let bin_path = driver_bin.with_extension(label);
        let compile = Command::new("rustc")
            .args([
                "--edition=2024",
                "-C",
                "opt-level=2",
                "-C",
                "panic=abort",
                "--extern",
            ])
            .arg(format!("lust_aot_rt={}", rlib.display()))
            .arg(&driver_rs)
            .arg("-o")
            .arg(&bin_path)
            .output()
            .expect("compile driver binary");
        assert!(
            compile.status.success(),
            "failed to link driver against {label} rlib:\nstderr:\n{}",
            String::from_utf8_lossy(&compile.stderr)
        );

        let run = Command::new(&bin_path)
            .output()
            .expect("execute linked AOT driver");
        assert!(
            run.status.success(),
            "AOT driver ({label}) exited with failure:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&run.stdout).trim(),
            "LUST_AOT_RT_OK"
        );
    }
}

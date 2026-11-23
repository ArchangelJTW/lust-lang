//! Integration & parity test suite for `NG-W13-1B`:
//! - Parameterized one-shot non-tail `handle..with` state-machine via
//!   `EffectYield<OpArg, ResumeVal, BodyRet>` (`i64`, `f64`, `bool`, `String`, `()`, `AotValue`)
//! - 3-phase LIFO structured concurrency `AotScope` (`Cancelling -> Joining -> Finalizing`)
//!   for `StmtKind::Scoped` and `ExprKind::EffectBlock`
//! - `#![no_std] + alloc`, `TASK_ALLOWS_UNSAFE=0`, panic-free (`-C panic=abort`).

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::tempdir;

use lust::codegen_zig::aot::{
    AotCompileOptions, AotCompiler, AotTargetKind, RustCompileOptions, ensure_cached_aot_rt_rlib,
};

fn rustc_bin() -> String {
    std::env::var("RUSTC")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "rustc".to_string())
}

fn assert_zero_unsafe_and_panic_free(code: &str) {
    assert!(
        !code.contains("unsafe"),
        "emitted Rust must contain ZERO `unsafe` blocks (TASK_ALLOWS_UNSAFE=0):\n{code}"
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

fn compile_and_run_rust_aot(source: &str, cache_dir: &Path) -> (String, std::process::Output) {
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

// ---------------------------------------------------------------------------
// Test 1: Parameterized `EffectYield<OpArg, ResumeVal, BodyRet>` across
//         i64, f64, bool, String, () with pre -> resume -> post ordering
// ---------------------------------------------------------------------------

#[test]
fn test_nontail_oneshot_pre_resume_post_and_all_types() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");

    let src = r#"
function test_int_pre_post(): int
    return handle {
        let _u0: int = 0;
        println("body_pre");
        let x: int = perform("AskInt", [10]);
        println("body_post:" .. tostring(x));
        x * 2
    } with {
        AskInt(v) => {
            let _u1: int = 0;
            println("handler_pre:" .. tostring(v));
            let r: int = resume(v + 5);
            println("handler_post:" .. tostring(r));
            r + 100
        }
    }
end

function test_hetero_f64_bool_string(): string
    return handle {
        let ok: bool = perform("CheckThreshold", [3.5]);
        (if ok then "yes" else "no")
    } with {
        CheckThreshold(x) => {
            let r: string = resume(x > 2.0);
            "wrapped:" .. r
        }
    }
end

function test_bool_f64_f64(): float
    return handle {
        let factor: float = perform("Scale", [true]);
        factor * 4.0
    } with {
        Scale(flag) => {
            let v: float = (if flag then 2.5 else 1.0);
            let r: float = resume(v);
            r + 0.5
        }
    }
end

function unit_val()
end

function test_string_unit_unit()
    handle {
        let _u: int = 0;
        perform("Log", ["ping"]);
        println("body_after_unit");
        unit_val()
    } with {
        Log(msg) => {
            let _u2: int = 0;
            println("pre:" .. msg);
            resume(unit_val());
            println("post:" .. msg);
            unit_val()
        }
    };
end

function main(): int
    println(test_int_pre_post())
    println(test_hetero_f64_bool_string())
    println(test_bool_f64_f64())
    test_string_unit_unit()
    return 0
end
"#;

    let (emitted, run) = compile_and_run_rust_aot(src, &cache_dir);
    assert!(
        run.status.success(),
        "non-tail all-types AOT binary failed:\nstderr: {}\n---\n{emitted}",
        String::from_utf8_lossy(&run.stderr)
    );

    // Verify parameterized EffectYield instantiation across distinct non-i64 types
    for expected in [
        "lust_aot_rt::handle_non_tail_oneshot::<i64, i64, i64",
        "lust_aot_rt::handle_non_tail_oneshot::<f64, bool, alloc::string::String",
        "lust_aot_rt::handle_non_tail_oneshot::<bool, f64, f64",
        "lust_aot_rt::handle_non_tail_oneshot::<alloc::string::String, (), ()",
        "lust_aot_rt::EffectYield::Yielded",
        "lust_aot_rt::OneShotResume::new(__resume_fn)",
    ] {
        assert!(
            emitted.contains(expected),
            "expected `{expected}` in emitted Rust:\n{emitted}"
        );
    }

    let stdout = String::from_utf8(run.stdout).expect("utf8");
    assert_eq!(
        stdout,
        concat!(
            "body_pre\n",
            "handler_pre:10\n",
            "body_post:15\n",
            "handler_post:30\n",
            "130\n",
            "wrapped:yes\n",
            "10.5\n",
            "pre:ping\n",
            "body_after_unit\n",
            "post:ping\n",
        )
    );
}

// ---------------------------------------------------------------------------
// Test 2: Sequential non-tail performs (deep handler), conditional Completed vs
//         Yielded, multi-arm non-tail + abortive, stateful capture, nested
// ---------------------------------------------------------------------------

#[test]
fn test_nontail_sequential_conditional_multiarm_abort_and_stateful() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");

    let src = r#"
function test_sequential_deep(): int
    return handle {
        let a: int = perform("AddOne", [1]);
        let b: int = perform("AddOne", [a + 10]);
        b * 2
    } with {
        AddOne(x) => {
            let r: int = resume(x + 1);
            r + 100
        }
    }
end

function test_conditional(do_yield: bool): int
    return handle {
        (if do_yield then perform("Bump", [7]) * 3 else 99)
    } with {
        Bump(v) => {
            let r: int = resume(v + 3);
            r + 1000
        }
    }
end

function test_multiarm(should_abort: bool): int
    return handle {
        (if should_abort then perform("Bail", [42]) + 9999 else perform("Wrap", [5]) + 1)
    } with {
        Wrap(x) => {
            let r: int = resume(x * 2);
            r * 10
        },
        Bail(code) => -code
    }
end

function test_stateful_nontail(): int
    local acc: int = 1
    let res: int = handle {
        let x: int = perform("Trace", [10]);
        acc = acc + x;
        acc
    } with {
        Trace(v) => {
            let _u: int = 0;
            acc = acc + v;
            let r: int = resume(5);
            acc = acc * 2;
            r + acc
        }
    };
    return res
end

function test_nested_outer_forward(): int
    return handle {
        let _u: int = 0;
        handle {
            let inner: int = perform("InnerNT", [4]);
            let outer: int = perform("OuterTail", [20]);
            inner + outer
        } with {
            InnerNT(x) => {
                let r: int = resume(x + 1);
                r * 2
            }
        }
    } with {
        OuterTail(y) => resume(y + 5)
    }
end

function main(): int
    println(test_sequential_deep())
    println(test_conditional(true))
    println(test_conditional(false))
    println(test_multiarm(false))
    println(test_multiarm(true))
    println(test_stateful_nontail())
    println(test_nested_outer_forward())
    return 0
end
"#;

    let (emitted, run) = compile_and_run_rust_aot(src, &cache_dir);
    assert!(
        run.status.success(),
        "sequential/conditional/multiarm AOT failed:\nstderr: {}\n---\n{emitted}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(
        emitted.contains("lust_aot_rt::EffectYield::Completed"),
        "expected EffectYield::Completed for non-yielding conditional branch:\n{emitted}"
    );
    let stdout = String::from_utf8(run.stdout).expect("utf8");
    // 1. sequential deep: a = 2, b = 13 -> b*2 = 26 -> +100 = 126 -> +100 = 226
    // 2. conditional(true): x = 10 -> x*3 = 30 -> +1000 = 1030
    // 3. conditional(false): Completed(99) -> 99
    // 4. multiarm(false): v = 10 -> v+1 = 11 -> *10 = 110
    // 5. multiarm(true): Bail(42) -> -42
    // 6. stateful_nontail: acc=1 -> pre: acc=11 -> body: acc=16, r=16 -> post: acc=32 -> 16+32=48
    // 7. nested_outer_forward: inner=5, outer=25 -> 30 -> InnerNT post: 30*2 = 60
    assert_eq!(stdout, "226\n1030\n99\n110\n-42\n48\n60\n");
}

// ---------------------------------------------------------------------------
// Test 3: One-shot invariant — double resume in non-tail arm fails cleanly
//         in an isolated child process (Rule 5)
// ---------------------------------------------------------------------------

#[test]
fn test_nontail_double_resume_rejected_in_child_process() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");

    let src = r#"
function main(): int
    return handle {
        let x: int = perform("Ask", [1]);
        x + 10
    } with {
        Ask(v) => {
            let r1: int = resume(v);
            let r2: int = resume(v + 1);
            r1 + r2
        }
    }
end
"#;

    let (_emitted, run) = compile_and_run_rust_aot(src, &cache_dir);
    assert!(!run.status.success());
    assert_eq!(run.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains("Effect already resumed"),
        "expected double-resume error in stderr, got: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// Test 4: 3-Phase LIFO `AotScope` (`Cancelling -> Joining -> Finalizing`),
//         Gate-NEG-02, async `Future` child cancellation, first-defect
//         preservation, and `EffectYield<AotValue, ...>` against `liblust_aot_rt.rlib`
// ---------------------------------------------------------------------------

#[test]
fn test_aot_scope_three_phase_lifo_async_future_and_effect_yield_rlib() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");
    let opts = RustCompileOptions {
        cache_dir: Some(cache_dir),
        ..RustCompileOptions::default()
    };
    let rlib = ensure_cached_aot_rt_rlib(&opts).expect("build lust-aot-rt rlib");

    let driver_rs = tmp.path().join("scope_yield_driver.rs");
    let driver_bin = tmp.path().join("scope_yield_driver");
    let driver_code = r#"
extern crate lust_aot_rt;

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use lust_aot_rt::{
    AotCtx, AotError, AotFiberState, AotScope, AotScopePhase, AotValue, EffectYield,
    active_scope_depth, defer, handle_non_tail_oneshot, scope_register_finalizer,
    scope_spawn_future, with_aot_scope,
};

fn main() {
    let mut ctx = AotCtx::new();

    // 1. Strict 3-phase Cancelling -> Joining -> Finalizing (LIFO) + first defect preservation
    let trace = Rc::new(RefCell::new(Vec::<&'static str>::new()));
    let mut scope = AotScope::new(10);

    let t_c1 = Rc::clone(&trace);
    let t_j1 = Rc::clone(&trace);
    let mut polls = 0u8;
    let f1 = scope
        .spawn_child_with_cancel(
            move |_| t_c1.borrow_mut().push("cancel:1"),
            move |_, cancelled| {
                assert!(cancelled);
                polls += 1;
                if polls >= 2 {
                    t_j1.borrow_mut().push("join:1");
                    Ok(true)
                } else {
                    Ok(false)
                }
            },
        )
        .expect("spawn f1");

    let t_c2 = Rc::clone(&trace);
    let t_j2 = Rc::clone(&trace);
    let f2 = scope
        .spawn_child_with_cancel(
            move |_| t_c2.borrow_mut().push("cancel:2"),
            move |_, cancelled| {
                assert!(cancelled);
                t_j2.borrow_mut().push("join:2");
                Err(AotError::UserError(String::from("defect_child_join")))
            },
        )
        .expect("spawn f2");

    for (name, fail_msg) in [
        ("fin:1", Some("defect_fin_1")),
        ("fin:2", Some("defect_fin_2")),
        ("fin:3", None),
    ] {
        let t_f = Rc::clone(&trace);
        scope
            .register_finalizer(move |_| {
                t_f.borrow_mut().push(name);
                match fail_msg {
                    Some(m) => Err(AotError::UserError(String::from(m))),
                    None => Ok(()),
                }
            })
            .expect("register finalizer");
    }

    let close_err = scope.close(&mut ctx);
    assert_eq!(
        close_err,
        Err(AotError::UserError(String::from("defect_child_join"))),
        "first defect from Joining must be preserved"
    );
    assert_eq!(scope.suppressed_errors().len(), 2);
    assert_eq!(f1.state(), AotFiberState::Cancelled);
    assert_eq!(f2.state(), AotFiberState::Failed);
    assert_eq!(
        scope.phase_transitions(),
        &[
            AotScopePhase::Open,
            AotScopePhase::Cancelling,
            AotScopePhase::Joining,
            AotScopePhase::Finalizing,
            AotScopePhase::Done,
        ]
    );
    assert_eq!(
        &*trace.borrow(),
        &[
            "cancel:1", "cancel:2", "join:2", "join:1", "fin:3", "fin:2", "fin:1"
        ]
    );

    // 2. Gate-NEG-02: masked child prevents Phase 3 Finalizing until unmasked
    let mut neg02 = AotScope::new(20);
    let masked = neg02.register_external_child().expect("ext");
    masked.enter_mask();
    let released = Rc::new(Cell::new(false));
    let rel_c = Rc::clone(&released);
    neg02
        .register_finalizer(move |_| {
            rel_c.set(true);
            Ok(())
        })
        .expect("fin");
    assert_eq!(neg02.try_close(&mut ctx), Ok(false));
    assert_eq!(neg02.phase(), AotScopePhase::Joining);
    assert!(!released.get());
    assert!(masked.exit_mask());
    masked.mark_cancelled();
    assert_eq!(neg02.try_close(&mut ctx), Ok(true));
    assert_eq!(neg02.phase(), AotScopePhase::Done);
    assert!(released.get());

    // 3. Async core::future::Future child fiber RAII DropGuard unwinds BEFORE parent finalizers
    let ftrace = Rc::new(RefCell::new(Vec::<&'static str>::new()));
    let ft1 = Rc::clone(&ftrace);
    let res = with_aot_scope(&mut ctx, |c| {
        assert_eq!(active_scope_depth(c), 1);
        let ft_fin = Rc::clone(&ft1);
        scope_register_finalizer(c, move |_| {
            ft_fin.borrow_mut().push("parent_fin");
            Ok(())
        })?;
        let ft_fut = Rc::clone(&ft1);
        let _fh = scope_spawn_future(c, async move {
            let _g1 = defer({
                let f = Rc::clone(&ft_fut);
                move || f.borrow_mut().push("child_guard_1")
            });
            let _g2 = defer({
                let f = Rc::clone(&ft_fut);
                move || f.borrow_mut().push("child_guard_2")
            });
            core::future::pending::<Result<AotValue, AotError>>().await
        })?;
        Ok(7_i64)
    });
    assert_eq!(res, Ok(7));
    assert_eq!(active_scope_depth(&ctx), 0);
    assert_eq!(
        &*ftrace.borrow(),
        &["child_guard_2", "child_guard_1", "parent_fin"]
    );

    // 4. EffectYield<AotValue, AotValue, AotValue> dynamic generic instantiation
    let dyn_res = handle_non_tail_oneshot::<AotValue, AotValue, AotValue, _, _>(
        &mut ctx,
        |_c| {
            Ok(EffectYield::yielded(AotValue::str("hello"), |_c2, v: AotValue| {
                Ok(AotValue::str(format!("{}:body", v.to_display_string())))
            }))
        },
        |c, arg, resume_fn| {
            let r = resume_fn(c, AotValue::str(format!("pre:{}", arg.to_display_string())))?;
            Ok(AotValue::str(format!("{}:post", r.to_display_string())))
        },
    );
    assert_eq!(dyn_res, Ok(AotValue::str("pre:hello:body:post")));

    println!("AOT_SCOPE_AND_YIELD_OK");
}
"#;
    fs::write(&driver_rs, driver_code).expect("write driver");
    let out = Command::new(rustc_bin())
        .arg("--edition=2024")
        .arg("-C")
        .arg("panic=abort")
        .arg("--extern")
        .arg(format!("lust_aot_rt={}", rlib.display()))
        .arg(&driver_rs)
        .arg("-o")
        .arg(&driver_bin)
        .output()
        .expect("compile scope_yield_driver");
    assert!(
        out.status.success(),
        "failed to compile scope_yield_driver:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run = Command::new(&driver_bin)
        .output()
        .expect("run scope_yield_driver");
    assert!(
        run.status.success(),
        "scope_yield_driver failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(
        String::from_utf8(run.stdout).expect("utf8"),
        "AOT_SCOPE_AND_YIELD_OK\n"
    );
}

// ---------------------------------------------------------------------------
// Test 5: End-to-end Lust `scoped do ... end` (StmtKind::Scoped) &
//         `effect do ... end` (ExprKind::EffectBlock) -> `AotScope` LIFO
// ---------------------------------------------------------------------------

#[test]
fn test_lust_scoped_block_and_effect_block_end_to_end_aot() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");

    let src = r#"
function test_scoped_lifo_boundary(): int
    local log: int = 1
    scoped do
        defer log = log * 2
        defer log = log + 100
        log = log + 10
    end
    -- Scoped LIFO defers MUST have already run at scope exit:
    -- log = 1 + 10 = 11 -> +100 = 111 -> *2 = 222
    return log + 1
end

function test_nested_scoped_order()
    scoped do
        defer println("outer_fin_1")
        defer println("outer_fin_2")
        println("outer_body_start")
        scoped do
            defer println("inner_fin_1")
            defer println("inner_fin_2")
            println("inner_body")
        end
        println("outer_body_after_inner")
    end
end

function test_effect_block_with_scope(): int
    local base: int = 40
    local res: int = (effect do
        scoped do
            defer println("effect_scoped_fin")
            println("effect_scoped_body")
        end
        return base + 2
    end):run_sync()
    return res
end

function main(): int
    println(test_scoped_lifo_boundary())
    test_nested_scoped_order()
    println(test_effect_block_with_scope())
    return 0
end
"#;

    let (emitted, run) = compile_and_run_rust_aot(src, &cache_dir);
    assert!(
        run.status.success(),
        "scoped/effect block AOT failed:\nstderr: {}\n---\n{emitted}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(
        emitted.contains("lust_aot_rt::with_aot_scope"),
        "expected `with_aot_scope` in emitted Rust:\n{emitted}"
    );
    assert!(
        emitted.contains("lust_aot_rt::scope_register_finalizer"),
        "expected `scope_register_finalizer` in emitted Rust:\n{emitted}"
    );

    let stdout = String::from_utf8(run.stdout).expect("utf8");
    assert_eq!(
        stdout,
        concat!(
            "223\n",
            "outer_body_start\n",
            "inner_body\n",
            "inner_fin_2\n",
            "inner_fin_1\n",
            "outer_body_after_inner\n",
            "outer_fin_2\n",
            "outer_fin_1\n",
            "effect_scoped_body\n",
            "effect_scoped_fin\n",
            "42\n",
        )
    );
}

// ---------------------------------------------------------------------------
// Test 6: `#![no_std]` library emission combining non-tail `handle..with`
//         and `scoped do ... end` compiles cleanly under `-C panic=abort`
// ---------------------------------------------------------------------------

#[test]
fn test_nontail_and_scoped_nostd_rlib_compilation() {
    let nostd_src = r#"
function compute(v: float): int
    local acc: int = 0
    scoped do
        defer acc = acc + 10
        acc = acc + 1
    end
    let h: int = handle {
        let ok: bool = perform("Check", [v]);
        (if ok then 100 else 200)
    } with {
        Check(x) => {
            let r: int = resume(x > 0.0);
            r + acc
        }
    };
    return h
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
    assert!(nostd_code.contains("lust_aot_rt::with_aot_scope"));
    assert!(nostd_code.contains("lust_aot_rt::handle_non_tail_oneshot"));

    let nostd_rlib = ensure_cached_aot_rt_rlib(&nostd_opts).expect("build nostd lust-aot-rt rlib");
    let nostd_rs = tmp.path().join("nostd_w13_1b.rs");
    let nostd_out_rlib = tmp.path().join("libnostd_w13_1b.rlib");
    fs::write(&nostd_rs, &nostd_code).expect("write nostd_w13_1b.rs");

    let check_out = Command::new(rustc_bin())
        .arg("--edition=2024")
        .arg("--crate-name")
        .arg("nostd_w13_1b")
        .arg("--crate-type=rlib")
        .arg("-C")
        .arg("panic=abort")
        .arg("--extern")
        .arg(format!("lust_aot_rt={}", nostd_rlib.display()))
        .arg(&nostd_rs)
        .arg("-o")
        .arg(&nostd_out_rlib)
        .output()
        .expect("compile nostd_w13_1b rlib");
    assert!(
        check_out.status.success(),
        "no_std Rust AOT non-tail/scoped library failed to compile:\n{}\n---\n{nostd_code}",
        String::from_utf8_lossy(&check_out.stderr)
    );
}

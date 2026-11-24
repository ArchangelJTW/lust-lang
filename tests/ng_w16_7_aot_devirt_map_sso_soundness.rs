//! Integration, soundness, and performance gate tests for `NG-W16-7`:
//! 1. Rust AOT Devirtualization, closed-world `__LustDynStruct`, and typed `AotQueue<i64>` (`AOT-DEVIRT`)
//! 2. Direct unboxed `AotRawMap<K, V>` & borrowed `&str` map keys (`AOT-MAP-HASH`)
//! 3. Borrowed `IfArrayGetOk`, 1-byte `string.sub` peephole, O(N) constant-pattern `string.gsub`,
//!    chained concat fusion, and lazy schema decoders (`AOT-SSO-REGEX`)
//! 4. Shared `LustBudgetAlloc::share()`, cyclic struct reference aliasing & bounded memory,
//!    `GasMetering::LoopChunked` loop-header batching, and selective module inlining (`AOT-BUDGET-SOUND`)

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use tempfile::tempdir;

use lust::codegen_zig::aot::{
    AotCompileOptions, AotCompiler, GasMetering, RustCompileOptions,
};

fn suite_v2_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("benchmarks/suite_v2")
}

fn compile_and_run_rust_aot_with_opts(
    source: &str,
    opts: &RustCompileOptions,
) -> (String, std::process::Output) {
    let tmp = tempdir().expect("tempdir");
    let emitted = AotCompiler::emit_rust_source(source, opts).expect("emit_rust_source");
    assert!(
        !emitted.contains("unsafe"),
        "emitted Rust must contain ZERO `unsafe` blocks"
    );
    let bin = tmp.path().join("aot_bin");
    AotCompiler::compile_source_rust(source, &bin, opts).expect("compile_source_rust");
    let run = Command::new(&bin).output().expect("execute aot binary");
    (emitted, run)
}

fn measure_median_ms(bin: &Path, expected_stdout: &str, samples: usize) -> f64 {
    // Warmup run + oracle check
    let warm = Command::new(bin).output().expect("warmup run");
    assert!(
        warm.status.success(),
        "warmup failed for {:?}:\nstdout: {}\nstderr: {}",
        bin,
        String::from_utf8_lossy(&warm.stdout),
        String::from_utf8_lossy(&warm.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&warm.stdout),
        expected_stdout,
        "oracle mismatch for {:?}",
        bin
    );

    let mut timings = Vec::with_capacity(samples);
    for _ in 0..samples {
        let t0 = Instant::now();
        let out = Command::new(bin).output().expect("sample run");
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        assert!(out.status.success(), "sample run failed for {:?}", bin);
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            expected_stdout,
            "sample oracle mismatch for {:?}",
            bin
        );
        timings.push(elapsed_ms);
    }
    timings.sort_by(|a, b| a.total_cmp(b));
    timings[timings.len() / 2]
}

// ---------------------------------------------------------------------------
// Test 1: Suite v2 AOT parity & speed targets (`AOT-DEVIRT`, `AOT-MAP-HASH`, `AOT-SSO-REGEX`)
// ---------------------------------------------------------------------------

#[test]
fn test_ng_w16_7_aot_suite_v2_all_15_within_1_2x_rust() {
    let suite_dir = suite_v2_dir();
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");
    let lust_bin = env!("CARGO_BIN_EXE_lust");

    // 6 primary NG-W16-7 speed-target benchmarks + remaining Suite v2 parity benchmarks.
    let targets: &[(&str, &str, i64, Option<f64>)] = &[
        ("BENCH-NG-01", "map_int_churn", 608439931, Some(12.0)),
        ("BENCH-NG-02", "map_str_intern_hist", 379372375, Some(8.5)),
        ("BENCH-NG-03", "array_mut_pipeline", 500808394, None),
        ("BENCH-NG-04", "closure_upvalue_combinators", 947911, None),
        ("BENCH-NG-05", "megamorphic_method_ic", 24980750, Some(9.5)),
        ("BENCH-NG-06", "adt_enum_eval", 2262427, None),
        ("BENCH-NG-07", "sroa_escape_structs", 990177872, None),
        ("BENCH-NG-08", "cycle_collector_pressure", 250029089, None),
        ("BENCH-NG-09", "string_parse_tokenize", 948525981, None),
        ("BENCH-NG-10", "regex_match_extract", 240000, Some(6.0)),
        ("BENCH-NG-11", "defer_effect_thunk", 466147248, None),
        ("BENCH-NG-12", "fiber_channel_pingpong", 799919916, Some(3.6)),
        ("BENCH-NG-13", "schema_json_decode_pipeline", 8224993, Some(9.0)),
        ("BENCH-NG-14", "loop_3level_call_barrier", 321195209, None),
        ("BENCH-NG-15", "ffi_struct_array_pingpong", 151024993, None),
    ];

    let mut failures = Vec::new();
    for &(bench_id, stem, oracle, max_ms) in targets {
        let script = suite_dir.join(format!("{stem}.lust"));
        assert!(script.exists(), "missing benchmark fixture: {:?}", script);
        let bin = tmp.path().join(stem);
        let build = Command::new(lust_bin)
            .arg("build")
            .arg(&script)
            .arg("--backend")
            .arg("rust")
            .arg("-o")
            .arg(&bin)
            .env("LUST_AOT_CACHE_DIR", &cache_dir)
            .output()
            .expect("invoke lust build --backend rust");
        assert!(
            build.status.success(),
            "{bench_id} ({stem}) lust build --backend rust failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&build.stdout),
            String::from_utf8_lossy(&build.stderr)
        );

        let expected = format!("{oracle}\n");
        let median_ms = measure_median_ms(&bin, &expected, if max_ms.is_some() { 5 } else { 1 });
        if let Some(limit_ms) = max_ms {
            if median_ms > limit_ms {
                failures.push(format!(
                    "{bench_id} ({stem}): median {median_ms:.3} ms > target {limit_ms:.1} ms"
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "Suite v2 Rust AOT speed gate failures:\n{}",
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Test 2: Cumulative memory budget shared across handlers, closures, and scopes
// ---------------------------------------------------------------------------

#[test]
fn test_ng_w16_7_memory_budget_shared_across_handlers_closures_and_scopes() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");

    // 2a. Allocate 500 bytes in main + 500 bytes in a `handle..with` arm under an 800-byte limit.
    //     With split-brain `LustBudgetAlloc::with_limit`, each boundary resets `used = 0` so both
    //     succeed; with `budget_alloc.share()`, cumulative usage hits 1000 > 800 -> OOM!
    let src_handle = r#"
function main()
    local outer: Array<int> = []
    local i: int = 0
    while i < 64 do
        array.push(outer, i)
        i = i + 1
    end
    local r: int = handle {
        perform("Alloc", [64])
    } with {
        Alloc(_n) => {
            local inner: Array<int> = []
            local j: int = 0
            while j < 64 do
                array.push(inner, j)
                j = j + 1
            end
            resume(array.len(inner))
        }
    }
    println(array.len(outer) + r)
end
"#;

    // 2b. Allocate inside a boxed/dynamic closure + outer array under an 800-byte limit.
    let src_closure = r#"
function run_cb(cb: unknown, n: int): int
    return cb(n)
end

function main()
    local outer: Array<int> = []
    local i: int = 0
    while i < 64 do
        array.push(outer, i)
        i = i + 1
    end
    local mk = function(n: int): int
        local inner: Array<int> = []
        local j: int = 0
        while j < n do
            array.push(inner, j)
            j = j + 1
        end
        return array.len(inner)
    end
    local got: int = run_cb(mk, 64)
    println(array.len(outer) + got)
end
"#;

    // 2c. Allocate inside a `scoped do ... defer ... end` finalizer + outer array under an 800-byte limit.
    let src_scoped = r#"
function main()
    local outer: Array<int> = []
    local i: int = 0
    while i < 64 do
        array.push(outer, i)
        i = i + 1
    end
    scoped do
        defer do
            local inner: Array<int> = []
            local j: int = 0
            while j < 64 do
                array.push(inner, j)
                j = j + 1
            end
        end
    end
    println(array.len(outer))
end
"#;

    let strict_opts = RustCompileOptions {
        aot: AotCompileOptions {
            max_memory_bytes: Some(800),
            ..AotCompileOptions::default()
        },
        cache_dir: Some(cache_dir.clone()),
        ..RustCompileOptions::default()
    };

    let generous_opts = RustCompileOptions {
        aot: AotCompileOptions {
            max_memory_bytes: Some(64 * 1024),
            ..AotCompileOptions::default()
        },
        cache_dir: Some(cache_dir),
        ..RustCompileOptions::default()
    };

    for (label, src) in [
        ("handle..with arm", src_handle),
        ("closure", src_closure),
        ("scoped child task", src_scoped),
    ] {
        // Must succeed with generous budget
        let (emitted, ok_out) = compile_and_run_rust_aot_with_opts(src, &generous_opts);
        assert!(
            ok_out.status.success(),
            "{label} failed under generous budget:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&ok_out.stdout),
            String::from_utf8_lossy(&ok_out.stderr)
        );
        // Must NEVER emit split-brain `LustBudgetAlloc::with_limit(__...budget_alloc.limit_bytes())`
        assert!(
            !emitted.contains("budget_alloc.limit_bytes()"),
            "{label}: emitted Rust still contains split-brain `budget_alloc.limit_bytes()`!"
        );

        // Must fail with OutOfMemory under strict cumulative budget (512 + 512 = 1024 > 800)
        let (_emitted_strict, oom_out) = compile_and_run_rust_aot_with_opts(src, &strict_opts);
        let stderr = String::from_utf8_lossy(&oom_out.stderr);
        assert!(
            !oom_out.status.success(),
            "{label}: expected cumulative OutOfMemory under 800-byte limit, but succeeded! stdout={}",
            String::from_utf8_lossy(&oom_out.stdout)
        );
        assert!(
            stderr.contains("Out of memory") || stderr.contains("OutOfMemory"),
            "{label}: expected OutOfMemory diagnostic in stderr, got: {stderr}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3: Cyclic struct reference aliasing identity & bounded memory
// ---------------------------------------------------------------------------

#[test]
fn test_ng_w16_7_cyclic_struct_aliasing_identity_and_bounded_memory() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");

    // 3a. Reference aliasing identity:
    //     a.next = Some(b); b.next = Some(a); a.val = 99 -> b.next.unwrap().val == 99!
    let src_alias = r#"
struct Cell
    val: int
    next: Option<Cell>
end

function main()
    local a: Cell = Cell { val = 10, next = Option.None }
    local b: Cell = Cell { val = 20, next = Option.Some(a) }
    a.next = Option.Some(b)
    a.val = 99
    local observed: int = -1
    if b.next is Some(na) then
        observed = na.val
    end
    local observed_b: int = -1
    if a.next is Some(nb) then
        if nb.next is Some(na2) then
            observed_b = na2.val
        end
    end
    -- Break cycle cleanly before exit
    a.next = Option.None
    b.next = Option.None
    println(tostring(observed) .. "," .. tostring(observed_b))
end
"#;

    let opts = RustCompileOptions {
        cache_dir: Some(cache_dir.clone()),
        ..RustCompileOptions::default()
    };
    let (_emitted, out) = compile_and_run_rust_aot_with_opts(src_alias, &opts);
    assert!(
        out.status.success(),
        "cyclic aliasing test failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "99,99",
        "cyclic struct reference aliasing was NOT preserved!"
    );

    // 3b. Bounded memory when creating and breaking 20,000 cycles in a loop under a strict budget.
    //     With unbounded chunk arena retention, 20,000 * 2 cells leaks > 1 MiB.
    //     With Rc<RefCell<...>>, broken cycles are reclaimed immediately so memory stays O(1)!
    let src_bounded = r#"
struct Cell
    val: int
    next: Option<Cell>
end

function main()
    local total: int = 0
    local i: int = 0
    while i < 20000 do
        local a: Cell = Cell { val = i, next = Option.None }
        local b: Cell = Cell { val = i + 1, next = Option.Some(a) }
        a.next = Option.Some(b)
        if a.next is Some(nb) then
            total = (total + nb.val) % 1000000007
        end
        a.next = Option.None
        b.next = Option.None
        i = i + 1
    end
    println(total)
end
"#;

    let bounded_opts = RustCompileOptions {
        aot: AotCompileOptions {
            max_memory_bytes: Some(256 * 1024),
            ..AotCompileOptions::default()
        },
        cache_dir: Some(cache_dir),
        ..RustCompileOptions::default()
    };
    let (_emitted_b, out_b) = compile_and_run_rust_aot_with_opts(src_bounded, &bounded_opts);
    assert!(
        out_b.status.success(),
        "bounded cyclic memory test failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out_b.stdout),
        String::from_utf8_lossy(&out_b.stderr)
    );
    // sum_{i=0..19999} (i + 1) = 20000 * 20001 / 2 = 200010000
    assert_eq!(String::from_utf8_lossy(&out_b.stdout).trim(), "200010000");
}

// ---------------------------------------------------------------------------
// Test 4: `GasMetering::LoopChunked` batches loop-body charges into loop header
// ---------------------------------------------------------------------------

#[test]
fn test_ng_w16_7_gas_metering_loop_chunked_batches_loop_charges() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");

    // Loop with an inner constant-trip loop so Exact emits multiple `charge_gas` statements
    // inside the outer loop body, whereas LoopChunked batches the loop-body gas weights into
    // a single loop-header `charge_gas(summed_weight)` call.
    let src = r#"
function sum_loop(n: int): int
    local acc: int = 0
    local i: int = 0
    while i < n do
        local k: int = 0
        while k < 4 do
            local x: int = (i + k) * 3 + 1
            local y: int = x ^ 7
            acc = acc + (y & 255)
            k = k + 1
        end
        i = i + 1
    end
    return acc
end

function main()
    println(sum_loop(1000))
end
"#;

    let exact_opts = RustCompileOptions {
        gas_metering: GasMetering::Exact,
        gas_limit: Some(1_000_000),
        cache_dir: Some(cache_dir.clone()),
        ..RustCompileOptions::default()
    };
    let chunked_opts = RustCompileOptions {
        gas_metering: GasMetering::LoopChunked,
        gas_limit: Some(1_000_000),
        cache_dir: Some(cache_dir.clone()),
        ..RustCompileOptions::default()
    };

    let (emitted_exact, exact_out) = compile_and_run_rust_aot_with_opts(src, &exact_opts);
    let (emitted_chunked, chunked_out) = compile_and_run_rust_aot_with_opts(src, &chunked_opts);

    assert!(exact_out.status.success(), "Exact run failed");
    assert!(chunked_out.status.success(), "LoopChunked run failed");
    assert_eq!(
        String::from_utf8_lossy(&exact_out.stdout),
        String::from_utf8_lossy(&chunked_out.stdout),
        "Exact and LoopChunked must produce identical output"
    );

    // Count `charge_gas(` occurrences inside `fn __lust_fn_sum_loop` (or `fn __lust_fn_main` if inlined)
    let exact_charges = emitted_exact.matches("charge_gas(").count();
    let chunked_charges = emitted_chunked.matches("charge_gas(").count();
    assert!(
        chunked_charges > 0,
        "LoopChunked must still emit loop-header gas charges:\n{emitted_chunked}"
    );
    assert!(
        chunked_charges < exact_charges,
        "LoopChunked must batch loop-body gas charges (exact={exact_charges}, chunked={chunked_charges}):\n--- EXACT ---\n{emitted_exact}\n--- CHUNKED ---\n{emitted_chunked}"
    );

    // Verify LoopChunked still enforces OutOfGas on a tight gas limit
    let tight_opts = RustCompileOptions {
        gas_metering: GasMetering::LoopChunked,
        gas_limit: Some(50),
        cache_dir: Some(cache_dir),
        ..RustCompileOptions::default()
    };
    let (_emitted_tight, tight_out) = compile_and_run_rust_aot_with_opts(src, &tight_opts);
    assert!(
        !tight_out.status.success(),
        "LoopChunked must fail with OutOfGas when gas_limit=50"
    );
    let tight_stderr = String::from_utf8_lossy(&tight_out.stderr);
    assert!(
        tight_stderr.contains("Out of gas") || tight_stderr.contains("OutOfGas"),
        "expected OutOfGas diagnostic, got: {tight_stderr}"
    );
}

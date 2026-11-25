#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! Wave 16 Task `NG-W16-8` Integration & Regression Test Suite:
//! - Cranelift `Rvalue::Closure` & indirect `MirType::Function` call codegen (`W16-JIT-CLOSURE`)
//! - 4-way inline type-guarded `unknown` PIC for `__mir_unknown_get_field` &
//!   `__mir_unknown_call_method` (`W16-JIT-MEGA-IC`)
//! - `__mir_unknown_is_int` / `__mir_unknown_unwrap_int` & `queue.new` / `queue.offer_int` /
//!   `queue.take_int` codegen (`W16-JIT-STR-FIBER`)
//! - Persistent nested `Q=64` quantum across outer/mid/inner loops & outermost array hoisting
//!   (`BENCH-NG-14`), with bit-exact `SEC-01` gas and `--features test-deopt` preservation.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::Instant;

use lust::ast::{Span, StructDef};
use lust::bytecode::compiler::Compiler;
use lust::bytecode::{EnumObject, Function, Value};
use lust::config::LustConfig;
use lust::data::layout::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1};
use lust::hashbrown::HashMap;
use lust::intern::Interner;
use lust::lexer::Lexer;
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::safepoint_request_deopt_on_poll;
use lust::mir::cranelift_backend::{
    safepoint_reset, take_current_sticky_error, take_deopt_payload, CraneliftJitCompiler,
};
use lust::mir::ir::{
    FunctionSignature, MirBinaryOp, MirConstant, MirFunction, MirModule, MirStructLayout, MirType,
    Operand, Place, Rvalue, Statement, Terminator,
};
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vfs::VirtualConsole;
use lust::vm::{reset_effect_handler_state, VmBackend, VM};
use lust::{compile_program_with_config, load_program_from_embedded, EmbeddedModule};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn reset_all_state() {
    safepoint_reset();
    reset_effect_handler_state();
    take_deopt_payload();
    take_current_sticky_error();
}

fn suite_v2_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("benchmarks/suite_v2")
}

fn suite_v2_path(filename: &str) -> PathBuf {
    suite_v2_dir().join(filename)
}

fn debug_lust_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_lust"))
}

/// Locates `<target_dir>/release/lust` if it exists and is at least as new as
/// `src/mir/cranelift_backend.rs`, so speed assertions measure optimized host helpers
/// while still running cleanly when `cargo test` is invoked without `--release`.
fn fresh_release_lust_bin() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let backend_src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/mir/cranelift_backend.rs");
    let src_m = std::fs::metadata(&backend_src).ok()?.modified().ok()?;
    let mut cur = exe.parent();
    while let Some(dir) = cur {
        let rel_bin = dir.join("release").join("lust");
        if rel_bin.is_file() {
            if let Ok(bin_m) = std::fs::metadata(&rel_bin).and_then(|m| m.modified())
                && bin_m >= src_m
            {
                return Some(rel_bin);
            }
            return None;
        }
        cur = dir.parent();
    }
    None
}

fn parse_cranelift_entries(output: &str) -> Option<(u64, u64)> {
    for line in output.lines() {
        if let Some(rest) = line.strip_prefix("Cranelift stats:") {
            let mut entries = None;
            let mut compilations = None;
            for tok in rest.split_whitespace() {
                if let Some(v) = tok.strip_prefix("entries=") {
                    entries = v.parse::<u64>().ok();
                } else if let Some(v) = tok.strip_prefix("compilations=") {
                    compilations = v.parse::<u64>().ok();
                }
            }
            if let (Some(e), Some(c)) = (entries, compilations) {
                return Some((e, c));
            }
        }
    }
    None
}

fn run_cli_benchmark(bin: &Path, bench_path: &Path, with_stats: bool) -> (String, String, f64) {
    let mut cmd = Command::new(bin);
    cmd.arg("--backend").arg("cranelift").arg(bench_path);
    if with_stats {
        cmd.env("LUST_JIT_STATS", "1");
    }
    let t0 = Instant::now();
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {}: {e}", bin.display()));
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "lust --backend cranelift failed on {} (status={:?})\nstdout:\n{stdout}\nstderr:\n{stderr}",
        bench_path.display(),
        out.status
    );
    (stdout, stderr, elapsed_ms)
}

fn median_ms(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    let mid = samples.len() / 2;
    if samples.len() % 2 == 1 {
        samples[mid]
    } else {
        0.5 * (samples[mid - 1] + samples[mid])
    }
}

fn compile_embedded_vm(source: &str, cranelift: bool) -> VM {
    let entries = &[EmbeddedModule {
        module: "main",
        parent: None,
        source: Some(source),
    }];
    let program = load_program_from_embedded(entries, "main").expect("load embedded program");
    let mut cfg = LustConfig::default().with_cranelift_jit(cranelift);
    cfg.set_use_mir(true);
    let mut vm = compile_program_with_config(program, &cfg).expect("compile embedded program");
    vm.console = VirtualConsole::captured();
    vm.set_backend(if cranelift {
        VmBackend::Cranelift
    } else {
        VmBackend::Bytecode
    });
    vm
}

fn compile_bytecode_funcs(source: &str) -> (Vec<Function>, HashMap<String, StructDef>) {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize().expect("lex");
    let items = Parser::new(tokens).parse().expect("parse");
    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let mut struct_defs = checker.struct_definitions();
    for item in &items {
        if let lust::ast::ItemKind::Struct(sd) = &item.kind {
            struct_defs.entry(sd.name.clone()).or_insert_with(|| sd.clone());
        }
    }
    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_use_mir(false);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    compiler.set_struct_defs(struct_defs.clone());
    let funcs = compiler.compile_module(&items).expect("compile_module");
    (funcs, struct_defs)
}

const SUITE_V2_ORACLES: &[(&str, &str, &str)] = &[
    ("BENCH-NG-01", "map_int_churn.lust", "608439931"),
    ("BENCH-NG-02", "map_str_intern_hist.lust", "379372375"),
    ("BENCH-NG-03", "array_mut_pipeline.lust", "500808394"),
    ("BENCH-NG-04", "closure_upvalue_combinators.lust", "947911"),
    ("BENCH-NG-05", "megamorphic_method_ic.lust", "24980750"),
    ("BENCH-NG-06", "adt_enum_eval.lust", "2262427"),
    ("BENCH-NG-07", "sroa_escape_structs.lust", "990177872"),
    ("BENCH-NG-08", "cycle_collector_pressure.lust", "250029089"),
    ("BENCH-NG-09", "string_parse_tokenize.lust", "948525981"),
    ("BENCH-NG-10", "regex_match_extract.lust", "240000"),
    ("BENCH-NG-11", "defer_effect_thunk.lust", "466147248"),
    ("BENCH-NG-12", "fiber_channel_pingpong.lust", "799919916"),
    ("BENCH-NG-13", "schema_json_decode_pipeline.lust", "8224993"),
    ("BENCH-NG-14", "loop_3level_call_barrier.lust", "321195209"),
    ("BENCH-NG-15", "ffi_struct_array_pingpong.lust", "151024993"),
];

#[test]
fn test_ng_w16_8_all_15_suite_v2_benchmarks_enter_cranelift_natively() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let bin = debug_lust_bin();
    for &(id, filename, expected_checksum) in SUITE_V2_ORACLES {
        let path = suite_v2_path(filename);
        let (stdout, stderr, _ms) = run_cli_benchmark(&bin, &path, true);
        let combined = format!("{stdout}\n{stderr}");
        let first_line = stdout.lines().next().unwrap_or("").trim();
        assert_eq!(
            first_line, expected_checksum,
            "{id} ({filename}) stdout mismatch under --backend cranelift:\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
        let (entries, compilations) = parse_cranelift_entries(&combined).unwrap_or_else(|| {
            panic!("{id} ({filename}) missing Cranelift stats line:\n{combined}")
        });
        assert!(
            entries >= 1 && compilations >= 1,
            "{id} ({filename}) must enter Cranelift natively (got entries={entries}, compilations={compilations})"
        );
    }
}

#[test]
fn test_ng_w16_8_ng04_ng05_ng06_ng09_ng12_ng14_speed_targets() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let targets: &[(&str, &str, &str, f64)] = &[
        ("NG-04", "closure_upvalue_combinators.lust", "947911", 22.0),
        ("NG-05", "megamorphic_method_ic.lust", "24980750", 15.0),
        ("NG-06", "adt_enum_eval.lust", "2262427", 15.0),
        ("NG-09", "string_parse_tokenize.lust", "948525981", 18.0),
        ("NG-12", "fiber_channel_pingpong.lust", "799919916", 5.0),
        ("NG-14", "loop_3level_call_barrier.lust", "321195209", 8.2),
    ];

    if let Some(rel_bin) = fresh_release_lust_bin() {
        let mut speed_failures: Vec<String> = Vec::new();
        for &(id, filename, expected_checksum, max_ms) in targets {
            let path = suite_v2_path(filename);
            // Warmup + verify entries >= 1
            let (w_out, w_err, _) = run_cli_benchmark(&rel_bin, &path, true);
            assert_eq!(w_out.lines().next().unwrap_or("").trim(), expected_checksum);
            let combined = format!("{w_out}\n{w_err}");
            let (entries, _) = parse_cranelift_entries(&combined).expect("Cranelift stats");
            assert!(entries >= 1, "{id} must enter Cranelift natively");

            let mut samples = Vec::with_capacity(5);
            for _ in 0..5 {
                let (out, _err, ms) = run_cli_benchmark(&rel_bin, &path, false);
                assert_eq!(out.lines().next().unwrap_or("").trim(), expected_checksum);
                samples.push(ms);
            }
            let med = median_ms(&mut samples);
            eprintln!("[NG-W16-8 release] {id} ({filename}): median={med:.2} ms (target<={max_ms:.1} ms, samples={samples:?})");
            if med > max_ms {
                speed_failures.push(format!(
                    "{id} ({filename}) Cranelift release median {med:.2} ms exceeded target {max_ms:.1} ms (samples={samples:?})"
                ));
            }
        }
        assert!(
            speed_failures.is_empty(),
            "Speed target failures:\n{}",
            speed_failures.join("\n")
        );
    } else {
        // Fallback when no fresh release binary exists: verify pure JIT-bound targets in-process
        // (Cranelift codegen is always opt_level=speed) and verify NG-09/NG-12 parity + entries>=1.
        for &(id, filename, expected_checksum, max_ms) in targets {
            let path = suite_v2_path(filename);
            if id == "NG-12" {
                let (out, err, _) = run_cli_benchmark(&debug_lust_bin(), &path, true);
                assert_eq!(out.lines().next().unwrap_or("").trim(), expected_checksum);
                let (entries, _) =
                    parse_cranelift_entries(&format!("{out}\n{err}")).expect("Cranelift stats");
                assert!(entries >= 1, "NG-12 must enter Cranelift natively");
                continue;
            }
            let src = std::fs::read_to_string(&path).expect("read benchmark source");
            let mut vm = compile_embedded_vm(&src, true);
            vm.compile_cranelift().expect("compile_cranelift");
            let e0 = vm.cranelift_entry_count;
            vm.call("__script", vec![]).expect("warmup");
            assert!(vm.cranelift_entry_count > e0, "{id} must enter Cranelift natively");
            assert_eq!(vm.console.stdout_string().trim(), expected_checksum);

            if id != "NG-09" || !cfg!(debug_assertions) {
                let mut samples = Vec::with_capacity(5);
                for _ in 0..5 {
                    vm.console = VirtualConsole::captured();
                    let t0 = Instant::now();
                    vm.call("__script", vec![]).expect("timed run");
                    samples.push(t0.elapsed().as_secs_f64() * 1000.0);
                    assert_eq!(vm.console.stdout_string().trim(), expected_checksum);
                }
                let med = median_ms(&mut samples);
                eprintln!("[NG-W16-8 in-process] {id} ({filename}): median={med:.2} ms (target<={max_ms:.1} ms)");
                assert!(
                    med <= max_ms,
                    "{id} ({filename}) Cranelift in-process median {med:.2} ms exceeded target {max_ms:.1} ms (samples={samples:?})"
                );
            }
        }
    }
}

#[test]
fn test_ng_w16_8_sec01_exact_gas_preservation_persistent_nested_quantum() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    // 3-level nested loop:
    // outer: 0..a (a+1 header checks)
    // mid:   0..b per outer (a*(b+1) header checks)
    // inner: 0..c per mid   (a*b*(c+1) header checks)
    // With a=2, b=3, c=6:
    //   outer checks = 3
    //   mid checks   = 2 * 4 = 8
    //   inner checks = 2 * 3 * 7 = 42  -> total loop checks = 53
    // With a=2, b=2, c=10:
    //   outer=3, mid=2*3=6, inner=2*2*11=44 -> total=53
    // Exact 50 loop header checks (+1 base call gas = 51 total gas):
    //   a=2, b=4, c=4 -> outer=3, mid=2*5=10, inner=2*4*5=40 -> 53
    //   a=3, b=2, c=6 -> outer=4, mid=3*3=9, inner=3*2*7=42 -> 55
    //   a=1, b=7, c=5 -> outer=2, mid=1*8=8, inner=1*7*6=42 -> 52
    //   a=2, b=3, c=5 -> outer=3, mid=2*4=8, inner=2*3*6=36 -> 47
    //   a=2, b=2, c=9 -> outer=3, mid=2*3=6, inner=2*2*10=40 -> 49 (+1 call = 50)
    //   a=1, b=4, c=10 -> outer=2, mid=5, inner=4*11=44 -> 51 (+1 call = 52)
    //   a=1, b=3, c=13 -> outer=2, mid=4, inner=3*14=42 -> 48
    //   a=2, b=5, c=3 -> outer=3, mid=12, inner=10*4=40 -> 55
    //   a=1, b=8, c=4 -> outer=2, mid=9, inner=8*5=40 -> 51
    //   a=2, b=1, c=21 -> outer=3, mid=4, inner=2*22=44 -> 51
    //   a=1, b=2, c=21 -> outer=2, mid=3, inner=2*22=44 -> 49
    //   a=1, b=5, c=7 -> outer=2, mid=6, inner=5*8=40 -> 48
    //   a=1, b=4, c=9 -> outer=2, mid=5, inner=4*10=40 -> 47
    //   a=1, b=1, c=45 -> outer=2, mid=2, inner=46 -> 50 (+1 call = 51 exact!)
    //   AND multi-trip 3-level with exact 50 loop header checks:
    //   a=2, b=2, c=9 => 3 + 6 + 40 = 49; a=3, b=1, c=19 => 4+6+60=70;
    //   a=2, b=4, c=3 => 3 + 10 + 32 = 45;
    //   a=5, b=1, c=6 => 6 + 10 + 35 = 51;
    //   a=3, b=3, c=3 => 4 + 12 + 36 = 52;
    //   a=2, b=3, c=6 => 3 + 8 + 42 = 53;
    //   a=4, b=2, c=4 => 5 + 12 + 40 = 57;
    //   a=3, b=2, c=5 => 4 + 9 + 36 = 49;
    //   a=2, b=2, c=9 (49) + 1 check in a trailing 0-trip loop? Even simpler:
    //   a=1, b=2, c=21 (49) -> use a=4, b=1, c=8 => 5 + 8 + 36 = 49;
    //   a=3, b=2, c=5 => 4 + 9 + 36 = 49 (+1 call = 50), or a=1, b=6, c=5 => 2 + 7 + 36 = 45,
    //   Wait: solve (a+1) + a*(b+1) + a*b*(c+1) = 50:
    //     a=1 => 2 + (b+1) + b*(c+1) = 50 => b*(c+2) = 47 (prime -> b=1, c=45)
    //     a=2 => 3 + 2*(b+1) + 2*b*(c+1) = 50 => odd != even
    //     a=3 => 4 + 3*(b+1) + 3*b*(c+1) = 50 => 3*(1 + b*(c+2)) = 46 (no)
    //     a=4 => 5 + 4*(b+1) + 4*b*(c+1) = 50 => odd != even
    //     a=7 => 8 + 7*(b+1) + 7*b*(c+1) = 50 => 7*(1 + b*(c+2)) = 42 => b*(c+2) = 5 => b=1, c=3!
    //   YES! a=7, b=1, c=3: outer=8, mid=7*2=14, inner=7*1*4=28 => 8+14+28 = 50 loop checks (+1 call = 51 exact)!
    //   AND for crossing Q=64: a=4, b=4, c=4 => outer=5, mid=4*5=20, inner=16*5=80 => 105 loop checks (+1 call = 106)!
    let src = r#"
        function nested3(a: int, b: int, c: int): int
            local acc: int = 0
            local i: int = 0
            while i < a do
                local j: int = 0
                while j < b do
                    local k: int = 0
                    while k < c do
                        acc = acc + (i * 100 + j * 10 + k)
                        k = k + 1
                    end
                    j = j + 1
                end
                i = i + 1
            end
            return acc
        end
    "#;

    // 1. Check generated IR uses a single persistent outer quantum (no inner-exit reconcile).
    let probe_vm = compile_embedded_vm(src, true);
    let mir_fn = probe_vm.mir_functions.get("nested3").cloned().expect("nested3 MIR");
    let mut ir_comp = CraneliftJitCompiler::new().expect("compiler");
    let ir = ir_comp.dump_function_ir(&mir_fn).expect("dump_function_ir nested3");
    // All 3 loop headers must share a single outer quantized check (3 calls to `lust_mir_gas_slow`'s fn ref).
    assert!(
        ir.contains("lust_mir_gas_slow") && ir.contains("lust_mir_safepoint_slow"),
        "nested3 IR must include quantized slow-path calls:\n{ir}"
    );
    let gas_slow_fn_sym = ir
        .lines()
        .find_map(|line| {
            let rest = line.strip_prefix("; sym ")?;
            let (sym, name) = rest.split_once(" = ")?;
            if name.trim() == "lust_mir_gas_slow" {
                Some(sym.trim().to_string())
            } else {
                None
            }
        })
        .expect("lust_mir_gas_slow symbol in IR");
    let gas_slow_call_pat = format!("call {gas_slow_fn_sym}()");
    let gas_slow_calls = ir.matches(&gas_slow_call_pat).count();
    assert!(
        gas_slow_calls == 1,
        "All 3 nested loop headers must share a SINGLE quantized refill hub (found {gas_slow_calls} `{gas_slow_call_pat}`):\n{ir}"
    );

    // 2. Exact SEC-01 boundary test: a=7, b=1, c=3 -> 50 loop checks + 1 call = 51 total gas.
    let oracle_51 = {
        let mut vm_bc = compile_embedded_vm(src, false);
        vm_bc
            .call("nested3", vec![Value::Int(7), Value::Int(1), Value::Int(3)])
            .expect("bc oracle 51")
    };
    for gas_limit in [0_u64, 1, 50, 51, 63, 64, 65] {
        reset_all_state();
        let mut vm = compile_embedded_vm(src, true);
        vm.set_gas_budget(gas_limit);
        let e0 = vm.cranelift_entry_count;
        let res = vm.call("nested3", vec![Value::Int(7), Value::Int(1), Value::Int(3)]);
        assert!(vm.cranelift_entry_count > e0, "nested3 must enter Cranelift for gas={gas_limit}");
        if gas_limit < 51 {
            let err = res.expect_err(&format!("expected OutOfGas at gas_limit={gas_limit}"));
            let msg = format!("{err:?}");
            assert!(
                msg.contains("OutOfGas") || msg.contains("gas"),
                "expected OutOfGas for gas_limit={gas_limit}, got {msg}"
            );
        } else {
            let val = res.unwrap_or_else(|e| panic!("expected Ok at gas_limit={gas_limit}: {e:?}"));
            assert_eq!(val, oracle_51, "mismatch at gas_limit={gas_limit}");
        }
    }

    // 3. Exact SEC-01 boundary across the Q=64 refill boundary:
    //    a=2, b=3, c=8 -> outer=3, mid=2*4=8, inner=2*3*9=54 => 65 loop checks + 1 call = 66 total gas.
    //    a=3, b=2, c=7 -> outer=4, mid=3*3=9, inner=3*2*8=48 => 61 loop checks + 1 call = 62 total gas.
    //    a=2, b=2, c=13 -> outer=3, mid=2*3=6, inner=2*2*14=56 => 65 loop checks + 1 call = 66 total gas.
    //    a=4, b=2, c=5 -> outer=5, mid=4*3=12, inner=4*2*6=48 => 65 loop checks + 1 call = 66 total gas.
    let oracle_66 = {
        let mut vm_bc = compile_embedded_vm(src, false);
        vm_bc
            .call("nested3", vec![Value::Int(4), Value::Int(2), Value::Int(5)])
            .expect("bc oracle 66")
    };
    for gas_limit in [0_u64, 1, 63, 64, 65, 66, 67] {
        reset_all_state();
        let mut vm = compile_embedded_vm(src, true);
        vm.set_gas_budget(gas_limit);
        let e0 = vm.cranelift_entry_count;
        let res = vm.call("nested3", vec![Value::Int(4), Value::Int(2), Value::Int(5)]);
        assert!(vm.cranelift_entry_count > e0);
        if gas_limit < 66 {
            let err = res.expect_err(&format!("expected OutOfGas at gas_limit={gas_limit}"));
            let msg = format!("{err:?}");
            assert!(msg.contains("OutOfGas") || msg.contains("gas"), "got {msg}");
        } else {
            let val = res.unwrap_or_else(|e| panic!("expected Ok at gas_limit={gas_limit}: {e:?}"));
            assert_eq!(val, oracle_66, "mismatch at gas_limit={gas_limit}");
        }
    }

    // 4. Forced deopt under `--features test-deopt` at outer, mid, inner, and post-64 poll ordinals!
    #[cfg(feature = "test-deopt")]
    {
        for poll_ordinal in [1_u64, 2, 3, 17, 50, 64, 65] {
            reset_all_state();
            let mut vm = compile_embedded_vm(src, true);
            vm.compile_cranelift().expect("compile_cranelift");
            safepoint_request_deopt_on_poll(Some("nested3".to_string()), poll_ordinal);
            let e0 = vm.cranelift_entry_count;
            let val = vm
                .call("nested3", vec![Value::Int(4), Value::Int(2), Value::Int(5)])
                .unwrap_or_else(|e| panic!("deopt at poll {poll_ordinal} failed: {e:?}"));
            assert!(vm.cranelift_entry_count > e0, "must enter Cranelift before deopt");
            assert_eq!(
                val, oracle_66,
                "deopt at poll {poll_ordinal} must preserve exact local state & result"
            );
        }
    }
}

#[test]
fn test_ng_w16_8_non_inlined_closure_and_indirect_call() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let span = Span::dummy();

    // Lifted closure 0 (1 capture: k, 1 arg: x) -> k + x
    let mut lam_add = MirFunction::new(
        FunctionSignature {
            name: "__lust_lambda_add".to_string(),
            param_types: vec![MirType::Int, MirType::Int],
            return_type: MirType::Int,
            is_public: false,
            effects: None,
        },
        span,
    );
    let p_k = lam_add.params[0];
    let p_x = lam_add.params[1];
    let sum = lam_add.add_local(None, MirType::Int, true, span);
    lam_add.blocks[0].statements.push(Statement::Assign(
        Place::Local(sum),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(p_k)),
            Operand::Copy(Place::Local(p_x)),
        ),
        span,
    ));
    lam_add.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(sum))),
        span,
    ));

    // Lifted closure 1 (2 captures: scale, bias; 1 arg: x) -> x * scale + bias
    let mut lam_affine = MirFunction::new(
        FunctionSignature {
            name: "__lust_lambda_affine".to_string(),
            param_types: vec![MirType::Int, MirType::Int, MirType::Int],
            return_type: MirType::Int,
            is_public: false,
            effects: None,
        },
        span,
    );
    let p_s = lam_affine.params[0];
    let p_b = lam_affine.params[1];
    let p_v = lam_affine.params[2];
    let prod = lam_affine.add_local(None, MirType::Int, true, span);
    let res = lam_affine.add_local(None, MirType::Int, true, span);
    lam_affine.blocks[0].statements.push(Statement::Assign(
        Place::Local(prod),
        Rvalue::BinaryOp(
            MirBinaryOp::Mul,
            Operand::Copy(Place::Local(p_v)),
            Operand::Copy(Place::Local(p_s)),
        ),
        span,
    ));
    lam_affine.blocks[0].statements.push(Statement::Assign(
        Place::Local(res),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(prod)),
            Operand::Copy(Place::Local(p_b)),
        ),
        span,
    ));
    lam_affine.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(res))),
        span,
    ));

    let fn_ty = MirType::Function {
        params: vec![MirType::Int],
        return_type: Box::new(MirType::Int),
    };

    // Non-inlined higher-order function: apply_twice(f: Function(Int)->Int, x: Int) -> f(f(x))
    let mut apply_twice = MirFunction::new(
        FunctionSignature {
            name: "apply_twice".to_string(),
            param_types: vec![fn_ty.clone(), MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let p_f = apply_twice.params[0];
    apply_twice.locals[p_f.0].name = Some("f".to_string());
    let p_ax = apply_twice.params[1];
    let t1 = apply_twice.add_local(None, MirType::Int, true, span);
    let t2 = apply_twice.add_local(None, MirType::Int, true, span);
    let b1 = apply_twice.new_block();
    let b2 = apply_twice.new_block();
    apply_twice.blocks[0].terminator = Some(Terminator::Call {
        callee: "f".to_string(),
        args: vec![Operand::Copy(Place::Local(p_ax))],
        destination: Place::Local(t1),
        success: b1,
        error: None,
        span,
    });
    apply_twice.blocks[b1.0].terminator = Some(Terminator::Call {
        callee: "f".to_string(),
        args: vec![Operand::Copy(Place::Local(t1))],
        destination: Place::Local(t2),
        success: b2,
        error: None,
        span,
    });
    apply_twice.blocks[b2.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(t2))),
        span,
    ));

    // Driver: creates BOTH closures (different capture counts!) and selects dynamically
    // by `sel` so neither closure can be statically constant-folded into a single target!
    let mut driver = MirFunction::new(
        FunctionSignature {
            name: "driver".to_string(),
            param_types: vec![MirType::Int, MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let d_sel = driver.params[0];
    let d_x = driver.params[1];
    let c_add = driver.add_local(Some("c_add".into()), fn_ty.clone(), true, span);
    let c_aff = driver.add_local(Some("c_aff".into()), fn_ty.clone(), true, span);
    let chosen = driver.add_local(Some("chosen".into()), fn_ty.clone(), true, span);
    let cond = driver.add_local(None, MirType::Bool, true, span);
    let out = driver.add_local(None, MirType::Int, true, span);
    let bb_then = driver.new_block();
    let bb_else = driver.new_block();
    let bb_call = driver.new_block();
    let bb_ret = driver.new_block();

    driver.blocks[0].statements.push(Statement::Assign(
        Place::Local(c_add),
        Rvalue::Closure {
            func_name: "__lust_lambda_add".to_string(),
            captures: vec![("k".into(), Operand::Constant(MirConstant::Int(10)), MirType::Int)],
        },
        span,
    ));
    driver.blocks[0].statements.push(Statement::Assign(
        Place::Local(c_aff),
        Rvalue::Closure {
            func_name: "__lust_lambda_affine".to_string(),
            captures: vec![
                ("scale".into(), Operand::Constant(MirConstant::Int(3)), MirType::Int),
                ("bias".into(), Operand::Constant(MirConstant::Int(5)), MirType::Int),
            ],
        },
        span,
    ));
    driver.blocks[0].statements.push(Statement::Assign(
        Place::Local(cond),
        Rvalue::BinaryOp(
            MirBinaryOp::Eq,
            Operand::Copy(Place::Local(d_sel)),
            Operand::Constant(MirConstant::Int(0)),
        ),
        span,
    ));
    driver.blocks[0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond)),
        then_block: bb_then,
        else_block: bb_else,
        span,
    });
    driver.blocks[bb_then.0].statements.push(Statement::Assign(
        Place::Local(chosen),
        Rvalue::Use(Operand::Copy(Place::Local(c_add))),
        span,
    ));
    driver.blocks[bb_then.0].terminator = Some(Terminator::Goto(bb_call, span));
    driver.blocks[bb_else.0].statements.push(Statement::Assign(
        Place::Local(chosen),
        Rvalue::Use(Operand::Copy(Place::Local(c_aff))),
        span,
    ));
    driver.blocks[bb_else.0].terminator = Some(Terminator::Goto(bb_call, span));
    driver.blocks[bb_call.0].terminator = Some(Terminator::Call {
        callee: "apply_twice".to_string(),
        args: vec![
            Operand::Copy(Place::Local(chosen)),
            Operand::Copy(Place::Local(d_x)),
        ],
        destination: Place::Local(out),
        success: bb_ret,
        error: None,
        span,
    });
    driver.blocks[bb_ret.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(out))),
        span,
    ));

    // Direct standalone compilation through CraneliftJitCompiler (without MIR inliner!).
    let mut module = MirModule::new("closure_indirect_test".into());
    module.add_function(lam_add.clone());
    module.add_function(lam_affine.clone());
    module.add_function(apply_twice.clone());
    module.add_function(driver.clone());
    let mut jit = CraneliftJitCompiler::new().expect("jit");
    let compiled = jit
        .compile_module(&module)
        .expect("CraneliftJitCompiler must compile Rvalue::Closure and indirect MirType::Function calls");
    assert!(compiled.contains_key("driver"));
    assert!(compiled.contains_key("apply_twice"));

    // End-to-end execution through VM with Cranelift backend (zero unsafe in test!).
    let stub_src = r#"
        function __lust_lambda_add(k: int, x: int): int
            return 0
        end
        function __lust_lambda_affine(s: int, b: int, v: int): int
            return 0
        end
        function apply_twice(f: function(int): int, x: int): int
            return 0
        end
        function driver(sel: int, x: int): int
            return 0
        end
    "#;
    let (bc_funcs, _) = compile_bytecode_funcs(stub_src);
    let mut cfg = LustConfig::default().with_cranelift_jit(true);
    cfg.set_jit_enabled(false);
    let mut vm = VM::with_config(&cfg);
    vm.set_backend(VmBackend::Cranelift);
    vm.load_functions(bc_funcs);
    vm.register_mir_candidate(lam_add);
    vm.register_mir_candidate(lam_affine);
    vm.register_mir_candidate(apply_twice);
    vm.register_mir_candidate(driver);
    vm.compile_cranelift().expect("compile_cranelift driver/apply_twice");

    let e0 = vm.cranelift_entry_count;
    // sel == 0: add 10 twice to 7 => 27
    let r0 = vm
        .call("driver", vec![Value::Int(0), Value::Int(7)])
        .expect("driver(0, 7)");
    assert!(vm.cranelift_entry_count > e0, "driver must run in Cranelift JIT");
    assert_eq!(r0, Value::Int(27));

    // sel == 1: affine (x*3 + 5) twice on 4 => (4*3+5)=17 -> (17*3+5)=56
    let r1 = vm
        .call("driver", vec![Value::Int(1), Value::Int(4)])
        .expect("driver(1, 4)");
    assert_eq!(r1, Value::Int(56));
}

#[test]
fn test_ng_w16_8_non_inlined_unknown_pic_4way_field_and_method() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let span = Span::dummy();

    let make_layout = |name: &str, fields: &[&str]| -> MirStructLayout {
        let descs: Vec<LogicalFieldDesc> = fields
            .iter()
            .enumerate()
            .map(|(i, fname)| LogicalFieldDesc::new(i as u16, *fname, CanonicalTypeTag::Int64))
            .collect();
        MirStructLayout::from(&LogicalLayoutV1::new(0, name.to_string(), descs))
    };

    // 4 structs with intentionally distinct field offsets for `x` and `w` (matching BENCH-NG-05).
    let layouts = vec![
        make_layout("NodeA", &["x", "w"]),             // x=0, w=1
        make_layout("NodeB", &["w", "x", "pad"]),      // w=0, x=1
        make_layout("NodeC", &["a", "b", "x", "w"]),   // x=2, w=3
        make_layout("NodeD", &["z", "w", "y", "x"]),   // w=1, x=3
    ];

    let make_score_method = |sname: &str, x_idx: usize, w_idx: usize, mode: usize| -> MirFunction {
        let mut f = MirFunction::new(
            FunctionSignature {
                name: format!("{sname}:score"),
                param_types: vec![MirType::Struct(sname.to_string()), MirType::Int],
                return_type: MirType::Int,
                is_public: false,
                effects: None,
            },
            span,
        );
        f.struct_layouts = layouts.clone();
        let p_self = f.params[0];
        let p_k = f.params[1];
        let fx = f.add_local(None, MirType::Int, true, span);
        let fw = f.add_local(None, MirType::Int, true, span);
        let tmp = f.add_local(None, MirType::Int, true, span);
        let res = f.add_local(None, MirType::Int, true, span);
        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(fx),
            Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(p_self)), x_idx))),
            span,
        ));
        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(fw),
            Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(p_self)), w_idx))),
            span,
        ));
        match mode {
            0 => {
                // x * k + w
                f.blocks[0].statements.push(Statement::Assign(
                    Place::Local(tmp),
                    Rvalue::BinaryOp(MirBinaryOp::Mul, Operand::Copy(Place::Local(fx)), Operand::Copy(Place::Local(p_k))),
                    span,
                ));
                f.blocks[0].statements.push(Statement::Assign(
                    Place::Local(res),
                    Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(tmp)), Operand::Copy(Place::Local(fw))),
                    span,
                ));
            }
            1 => {
                // x * k - w
                f.blocks[0].statements.push(Statement::Assign(
                    Place::Local(tmp),
                    Rvalue::BinaryOp(MirBinaryOp::Mul, Operand::Copy(Place::Local(fx)), Operand::Copy(Place::Local(p_k))),
                    span,
                ));
                f.blocks[0].statements.push(Statement::Assign(
                    Place::Local(res),
                    Rvalue::BinaryOp(MirBinaryOp::Sub, Operand::Copy(Place::Local(tmp)), Operand::Copy(Place::Local(fw))),
                    span,
                ));
            }
            2 => {
                // (x + w) * k
                f.blocks[0].statements.push(Statement::Assign(
                    Place::Local(tmp),
                    Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(fx)), Operand::Copy(Place::Local(fw))),
                    span,
                ));
                f.blocks[0].statements.push(Statement::Assign(
                    Place::Local(res),
                    Rvalue::BinaryOp(MirBinaryOp::Mul, Operand::Copy(Place::Local(tmp)), Operand::Copy(Place::Local(p_k))),
                    span,
                ));
            }
            _ => {
                // x * w + k
                f.blocks[0].statements.push(Statement::Assign(
                    Place::Local(tmp),
                    Rvalue::BinaryOp(MirBinaryOp::Mul, Operand::Copy(Place::Local(fx)), Operand::Copy(Place::Local(fw))),
                    span,
                ));
                f.blocks[0].statements.push(Statement::Assign(
                    Place::Local(res),
                    Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(tmp)), Operand::Copy(Place::Local(p_k))),
                    span,
                ));
            }
        }
        f.blocks[0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(res))), span));
        f
    };

    // Polymorphic function `eval_poly(obj: Unknown, k: Int): Int` NOT inlined!
    let mut eval_poly = MirFunction::new(
        FunctionSignature {
            name: "eval_poly".to_string(),
            param_types: vec![MirType::Unknown, MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    eval_poly.struct_layouts = layouts.clone();
    let p_obj = eval_poly.params[0];
    let p_k = eval_poly.params[1];
    let vx = eval_poly.add_local(None, MirType::Int, true, span);
    let vw = eval_poly.add_local(None, MirType::Int, true, span);
    let vscore = eval_poly.add_local(None, MirType::Int, true, span);
    let base = eval_poly.add_local(None, MirType::Int, true, span);
    let out = eval_poly.add_local(None, MirType::Int, true, span);
    let b1 = eval_poly.new_block();
    let b2 = eval_poly.new_block();
    let b3 = eval_poly.new_block();
    eval_poly.blocks[0].terminator = Some(Terminator::Call {
        callee: "__mir_unknown_get_field".into(),
        args: vec![
            Operand::Copy(Place::Local(p_obj)),
            Operand::Constant(MirConstant::String("x".into())),
        ],
        destination: Place::Local(vx),
        success: b1,
        error: None,
        span,
    });
    eval_poly.blocks[b1.0].terminator = Some(Terminator::Call {
        callee: "__mir_unknown_get_field".into(),
        args: vec![
            Operand::Copy(Place::Local(p_obj)),
            Operand::Constant(MirConstant::String("w".into())),
        ],
        destination: Place::Local(vw),
        success: b2,
        error: None,
        span,
    });
    eval_poly.blocks[b2.0].terminator = Some(Terminator::Call {
        callee: "__mir_unknown_call_method".into(),
        args: vec![
            Operand::Copy(Place::Local(p_obj)),
            Operand::Constant(MirConstant::String("score".into())),
            Operand::Copy(Place::Local(p_k)),
        ],
        destination: Place::Local(vscore),
        success: b3,
        error: None,
        span,
    });
    eval_poly.blocks[b3.0].statements.push(Statement::Assign(
        Place::Local(base),
        Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(vx)), Operand::Copy(Place::Local(vw))),
        span,
    ));
    eval_poly.blocks[b3.0].statements.push(Statement::Assign(
        Place::Local(out),
        Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(base)), Operand::Copy(Place::Local(vscore))),
        span,
    ));
    eval_poly.blocks[b3.0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(out))), span));

    // Driver allocating NodeA..NodeD and calling eval_poly(obj, k) based on `which` (0..3)
    // using a merged Unknown local so NG-W16-5 cannot statically devirtualize eval_poly!
    let mut driver = MirFunction::new(
        FunctionSignature {
            name: "poly_driver".to_string(),
            param_types: vec![MirType::Int, MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    driver.struct_layouts = layouts.clone();
    let p_which = driver.params[0];
    let p_dk = driver.params[1];
    let obj_u = driver.add_local(Some("obj_u".into()), MirType::Unknown, true, span);
    let ret_v = driver.add_local(None, MirType::Int, true, span);
    let bb_a = driver.new_block();
    let bb_b = driver.new_block();
    let bb_c = driver.new_block();
    let bb_d = driver.new_block();
    let bb_call = driver.new_block();
    let bb_ret = driver.new_block();

    driver.blocks[0].terminator = Some(Terminator::Switch {
        discr: Operand::Copy(Place::Local(p_which)),
        cases: vec![(0, bb_a), (1, bb_b), (2, bb_c)],
        default: bb_d,
        span,
    });
    driver.blocks[bb_a.0].statements.push(Statement::Assign(
        Place::Local(obj_u),
        Rvalue::Aggregate(
            MirType::Struct("NodeA".into()),
            vec![Operand::Constant(MirConstant::Int(3)), Operand::Constant(MirConstant::Int(10))],
        ),
        span,
    ));
    driver.blocks[bb_a.0].terminator = Some(Terminator::Goto(bb_call, span));
    driver.blocks[bb_b.0].statements.push(Statement::Assign(
        Place::Local(obj_u),
        Rvalue::Aggregate(
            MirType::Struct("NodeB".into()),
            vec![
                Operand::Constant(MirConstant::Int(7)),
                Operand::Constant(MirConstant::Int(5)),
                Operand::Constant(MirConstant::Int(0)),
            ],
        ),
        span,
    ));
    driver.blocks[bb_b.0].terminator = Some(Terminator::Goto(bb_call, span));
    driver.blocks[bb_c.0].statements.push(Statement::Assign(
        Place::Local(obj_u),
        Rvalue::Aggregate(
            MirType::Struct("NodeC".into()),
            vec![
                Operand::Constant(MirConstant::Int(0)),
                Operand::Constant(MirConstant::Int(0)),
                Operand::Constant(MirConstant::Int(11)),
                Operand::Constant(MirConstant::Int(2)),
            ],
        ),
        span,
    ));
    driver.blocks[bb_c.0].terminator = Some(Terminator::Goto(bb_call, span));
    driver.blocks[bb_d.0].statements.push(Statement::Assign(
        Place::Local(obj_u),
        Rvalue::Aggregate(
            MirType::Struct("NodeD".into()),
            vec![
                Operand::Constant(MirConstant::Int(0)),
                Operand::Constant(MirConstant::Int(4)),
                Operand::Constant(MirConstant::Int(0)),
                Operand::Constant(MirConstant::Int(13)),
            ],
        ),
        span,
    ));
    driver.blocks[bb_d.0].terminator = Some(Terminator::Goto(bb_call, span));
    driver.blocks[bb_call.0].terminator = Some(Terminator::Call {
        callee: "eval_poly".into(),
        args: vec![
            Operand::Copy(Place::Local(obj_u)),
            Operand::Copy(Place::Local(p_dk)),
        ],
        destination: Place::Local(ret_v),
        success: bb_ret,
        error: None,
        span,
    });
    driver.blocks[bb_ret.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(ret_v))),
        span,
    ));

    let m_a = make_score_method("NodeA", 0, 1, 0);
    let m_b = make_score_method("NodeB", 1, 0, 1);
    let m_c = make_score_method("NodeC", 2, 3, 2);
    let m_d = make_score_method("NodeD", 3, 1, 3);

    // 1. Verify standalone CraneliftJitCompiler compilation of non-inlined 4-way PIC.
    let mut module = MirModule::new("poly_pic4_test".into());
    module.add_function(m_a.clone());
    module.add_function(m_b.clone());
    module.add_function(m_c.clone());
    module.add_function(m_d.clone());
    module.add_function(eval_poly.clone());
    module.add_function(driver.clone());
    let mut jit = CraneliftJitCompiler::new().expect("jit");
    let compiled = jit
        .compile_module(&module)
        .expect("CraneliftJitCompiler must compile 4-way unknown PIC for get_field & call_method");
    assert!(compiled.contains_key("poly_driver"));
    assert!(compiled.contains_key("eval_poly"));

    // 2. Execute end-to-end through VM with Cranelift backend!
    let stub_src = r#"
        struct NodeA x: int, w: int end
        struct NodeB w: int, x: int, pad: int end
        struct NodeC a: int, b: int, x: int, w: int end
        struct NodeD z: int, w: int, y: int, x: int end
        impl NodeA function score(self, k: int): int return 0 end end
        impl NodeB function score(self, k: int): int return 0 end end
        impl NodeC function score(self, k: int): int return 0 end end
        impl NodeD function score(self, k: int): int return 0 end end
        function eval_poly(obj: unknown, k: int): int
            return 0
        end
        function poly_driver(which: int, k: int): int
            return 0
        end
    "#;
    let (bc_funcs, struct_defs) = compile_bytecode_funcs(stub_src);
    let mut cfg = LustConfig::default().with_cranelift_jit(true);
    cfg.set_jit_enabled(false);
    let mut vm = VM::with_config(&cfg);
    vm.set_backend(VmBackend::Cranelift);
    vm.register_structs(&struct_defs);
    vm.load_functions(bc_funcs);
    vm.register_mir_candidate(m_a);
    vm.register_mir_candidate(m_b);
    vm.register_mir_candidate(m_c);
    vm.register_mir_candidate(m_d);
    vm.register_mir_candidate(eval_poly);
    vm.register_mir_candidate(driver);
    vm.compile_cranelift().expect("compile_cranelift poly_driver/eval_poly");

    let e0 = vm.cranelift_entry_count;
    // k = 10:
    // NodeA(x=3,w=10): (3+10) + (3*10+10) = 13 + 40 = 53
    // NodeB(w=7,x=5):  (5+7)  + (5*10-7)  = 12 + 43 = 55
    // NodeC(x=11,w=2): (11+2) + (11+2)*10 = 13 + 130 = 143
    // NodeD(w=4,x=13): (13+4) + (13*4+10) = 17 + 62 = 79
    let expected = [(0, 53), (1, 55), (2, 143), (3, 79)];
    for (which, exp) in expected {
        let got = vm
            .call("poly_driver", vec![Value::Int(which), Value::Int(10)])
            .unwrap_or_else(|e| panic!("poly_driver({which}, 10) failed: {e:?}"));
        assert_eq!(got, Value::Int(exp), "4-way PIC mismatch for which={which}");
    }
    assert!(vm.cranelift_entry_count > e0, "poly_driver must run in Cranelift JIT");
}

#[test]
fn test_ng_w16_8_unknown_is_int_unwrap_int_and_queue_builtins() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let span = Span::dummy();

    // 1. Queue builtins: queue.new(4), offer_int(10), offer_int(20), take_int() + take_int() == 30
    let mut q_fn = MirFunction::new(
        FunctionSignature {
            name: "queue_test".to_string(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let p_x = q_fn.params[0];
    let q = q_fn.add_local(Some("q".into()), MirType::Int, true, span);
    let ok1 = q_fn.add_local(None, MirType::Int, true, span);
    let ok2 = q_fn.add_local(None, MirType::Int, true, span);
    let v1 = q_fn.add_local(None, MirType::Int, true, span);
    let v2 = q_fn.add_local(None, MirType::Int, true, span);
    let sum = q_fn.add_local(None, MirType::Int, true, span);
    let b1 = q_fn.new_block();
    let b2 = q_fn.new_block();
    let b3 = q_fn.new_block();
    let b4 = q_fn.new_block();
    let b5 = q_fn.new_block();
    q_fn.blocks[0].terminator = Some(Terminator::Call {
        callee: "queue.new".into(),
        args: vec![Operand::Constant(MirConstant::Int(4))],
        destination: Place::Local(q),
        success: b1,
        error: None,
        span,
    });
    q_fn.blocks[b1.0].terminator = Some(Terminator::Call {
        callee: "queue.offer_int".into(),
        args: vec![Operand::Copy(Place::Local(q)), Operand::Copy(Place::Local(p_x))],
        destination: Place::Local(ok1),
        success: b2,
        error: None,
        span,
    });
    q_fn.blocks[b2.0].terminator = Some(Terminator::Call {
        callee: "queue.offer_int".into(),
        args: vec![Operand::Copy(Place::Local(q)), Operand::Constant(MirConstant::Int(25))],
        destination: Place::Local(ok2),
        success: b3,
        error: None,
        span,
    });
    q_fn.blocks[b3.0].terminator = Some(Terminator::Call {
        callee: "queue.take_int".into(),
        args: vec![Operand::Copy(Place::Local(q))],
        destination: Place::Local(v1),
        success: b4,
        error: None,
        span,
    });
    q_fn.blocks[b4.0].terminator = Some(Terminator::Call {
        callee: "queue.take_int".into(),
        args: vec![Operand::Copy(Place::Local(q))],
        destination: Place::Local(v2),
        success: b5,
        error: None,
        span,
    });
    q_fn.blocks[b5.0].statements.push(Statement::Assign(
        Place::Local(sum),
        Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(v1)), Operand::Copy(Place::Local(v2))),
        span,
    ));
    q_fn.blocks[b5.0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(sum))), span));

    // 2. `__mir_unknown_is_int` & `__mir_unknown_unwrap_int` on `LuaValue` / `Unknown`
    let mut lv_fn = MirFunction::new(
        FunctionSignature {
            name: "lv_extract".to_string(),
            param_types: vec![MirType::Struct("LuaValue".into())],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let p_lv = lv_fn.params[0];
    let is_i = lv_fn.add_local(None, MirType::Bool, true, span);
    let out_v = lv_fn.add_local(None, MirType::Int, true, span);
    let lb1 = lv_fn.new_block();
    let lb_then = lv_fn.new_block();
    let lb_else = lv_fn.new_block();
    let lb_ret = lv_fn.new_block();
    lv_fn.blocks[0].terminator = Some(Terminator::Call {
        callee: "__mir_unknown_is_int".into(),
        args: vec![Operand::Copy(Place::Local(p_lv))],
        destination: Place::Local(is_i),
        success: lb1,
        error: None,
        span,
    });
    lv_fn.blocks[lb1.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(is_i)),
        then_block: lb_then,
        else_block: lb_else,
        span,
    });
    lv_fn.blocks[lb_then.0].terminator = Some(Terminator::Call {
        callee: "__mir_unknown_unwrap_int".into(),
        args: vec![Operand::Copy(Place::Local(p_lv))],
        destination: Place::Local(out_v),
        success: lb_ret,
        error: None,
        span,
    });
    lv_fn.blocks[lb_else.0].statements.push(Statement::Assign(
        Place::Local(out_v),
        Rvalue::Use(Operand::Constant(MirConstant::Int(-1))),
        span,
    ));
    lv_fn.blocks[lb_else.0].terminator = Some(Terminator::Goto(lb_ret, span));
    lv_fn.blocks[lb_ret.0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(out_v))), span));

    let stub_src = r#"
        function queue_test(x: int): int
            return 0
        end
        function lv_extract(lv: unknown): int
            return 0
        end
    "#;
    let (bc_funcs, _) = compile_bytecode_funcs(stub_src);
    let mut cfg = LustConfig::default().with_cranelift_jit(true);
    cfg.set_jit_enabled(false);
    let mut vm = VM::with_config(&cfg);
    vm.set_backend(VmBackend::Cranelift);
    vm.load_functions(bc_funcs);
    vm.register_mir_candidate(q_fn);
    vm.register_mir_candidate(lv_fn);

    let e0 = vm.cranelift_entry_count;
    let q_res = vm.call("queue_test", vec![Value::Int(17)]).expect("queue_test(17)");
    assert_eq!(q_res, Value::Int(42));
    assert!(vm.cranelift_entry_count > e0);

    let lv_int = Value::Enum(EnumObject::new(
        "LuaValue",
        "Int",
        Some(vec![Value::Int(123)]),
    ));
    let lv_str = Value::Enum(EnumObject::new(
        "LuaValue",
        "String",
        Some(vec![Value::string("abc")]),
    ));
    let e1 = vm.cranelift_entry_count;
    assert_eq!(vm.call("lv_extract", vec![lv_int]).expect("lv_extract(Int)"), Value::Int(123));
    assert_eq!(vm.call("lv_extract", vec![lv_str]).expect("lv_extract(String)"), Value::Int(-1));
    assert!(vm.cranelift_entry_count > e1);

    // 3. Verify 8-lane `I64X2` vector queue kernels (`OfferLinear` + `TakeAddMod`) fire on
    // `fiber_channel_pingpong.lust` and preserve bit-exact semantics on negative & wraparound values.
    let ng12_src = std::fs::read_to_string(suite_v2_path("fiber_channel_pingpong.lust"))
        .expect("read NG-12");
    let mut ng12_vm = compile_embedded_vm(&ng12_src, true);
    ng12_vm.compile_cranelift().expect("compile NG-12");
    let ng12_mir = ng12_vm
        .cranelift_compiled_mir("__script")
        .expect("NG-12 compiled MIR");
    let mut jit = CraneliftJitCompiler::new().expect("jit");
    let ng12_ir = jit.dump_function_ir(&ng12_mir).expect("dump NG-12 CLIF IR");
    assert!(
        ng12_ir.contains("i64x2") && ng12_ir.contains("insertlane") && ng12_ir.contains("extractlane"),
        "expected both 8-lane I64X2 OfferLinear (`insertlane`) and TakeAddMod (`extractlane`) kernels in NG-12 CLIF IR:\n{ng12_ir}"
    );
    let e2 = ng12_vm.cranelift_entry_count;
    ng12_vm.call("__script", vec![]).expect("run NG-12");
    assert!(ng12_vm.cranelift_entry_count > e2);
    assert_eq!(ng12_vm.console.stdout_string().trim(), "799919916");

    // Edge-case soundness: negative inputs & modulo wraparound must match exact scalar semantics.
    let edge_src = r#"
use @std/sync.{Queue}
local q = Queue(64, "suspend")
local offer = q.offer
local take = q.take
local total: int = 999999990
local round: int = 0
while round < 4 do
    local k: int = 0
    while k < 16 do
        offer((round - 2) * 16 + k)
        k = k + 1
    end
    local r: int = 0
    while r < 16 do
        local msg: int = take()
        total = (total + msg) % 1000000007
        r = r + 1
    end
    round = round + 1
end
println(total)
"#;
    let mut expected_total: i64 = 999_999_990;
    for round in 0..4i64 {
        for k in 0..16i64 {
            let msg = (round - 2) * 16 + k;
            expected_total = (expected_total + msg) % 1_000_000_007;
        }
    }
    let mut edge_vm = compile_embedded_vm(edge_src, true);
    edge_vm.compile_cranelift().expect("compile edge queue");
    edge_vm.call("__script", vec![]).expect("run edge queue");
    assert_eq!(
        edge_vm.console.stdout_string().trim(),
        expected_total.to_string(),
        "vector queue TakeAddMod must match exact scalar modulo semantics on negative & wraparound inputs"
    );
}

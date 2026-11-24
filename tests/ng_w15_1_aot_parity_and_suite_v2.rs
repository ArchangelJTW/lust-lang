//! Integration & regression tests for Wave 15 Task `NG-W15-1`:
//! - `BENCH-NG-15` (`ffi_struct_array_pingpong.lust`) Rust AOT `PARITY_FAIL` fix (`151024993`, was `959025384`)
//! - `BENCH-NG-11` (`defer_effect_thunk.lust`) zero-`RcCell`, zero-`with_aot_scope` fast path (`<= 14.0 ms`)
//! - `Direct` `Vec` copy-in-loop preservation (`analyze_single_def_aliases` + `emit_operand_owned`)
//! - Rust AOT string intrinsics (`string.sub`, `string.byte`, `string.gsub`, `string.split`, `string.find`)
//! - Function-typed parameter closure dispatch (`BENCH-NG-04` `closure_upvalue_combinators.lust`, `947911`)
//! - `BENCH-NG-10` (`regex_match_extract.lust`, `240000`) & `BENCH-NG-13` (`schema_json_decode_pipeline.lust`, `8224993`)
//! - Non-`@repr` `Struct.decode(raw)` layout registration and dynamic `AotValue` (`__mir_unknown_*` & `Queue`) dispatch

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;
use tempfile::tempdir;

use lust::codegen_zig::aot::{AotCompiler, RustCompileOptions};
use lust::data::layout::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1};
use lust::mir::ir::*;
use lust::Span;

fn suite_v2_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("benchmarks/suite_v2")
        .join(name)
}

fn compile_and_run_rust_aot(source: &str, name: &str) -> (String, f64) {
    let dir = tempdir().expect("tempdir");
    let bin_path = dir.path().join(name);
    let opts = RustCompileOptions::default();
    let built = AotCompiler::compile_source_rust(source, &bin_path, &opts)
        .unwrap_or_else(|e| panic!("Rust AOT compilation failed for {name}: {e}"));

    let start = Instant::now();
    let out = Command::new(&built)
        .output()
        .unwrap_or_else(|e| panic!("failed to execute {name}: {e}"));
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    assert!(
        out.status.success(),
        "{name} exited with {:?}, stderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    (
        String::from_utf8_lossy(&out.stdout).trim().to_string(),
        elapsed_ms,
    )
}

fn compile_and_measure_median_ms(source: &str, name: &str, runs: usize) -> (String, f64) {
    let dir = tempdir().expect("tempdir");
    let bin_path = dir.path().join(name);
    let opts = RustCompileOptions::default();
    let built = AotCompiler::compile_source_rust(source, &bin_path, &opts)
        .unwrap_or_else(|e| panic!("Rust AOT compilation failed for {name}: {e}"));

    // Warmup
    let warm = Command::new(&built).output().expect("warmup run");
    assert!(
        warm.status.success(),
        "{name} warmup failed: {}",
        String::from_utf8_lossy(&warm.stderr)
    );
    let expected = String::from_utf8_lossy(&warm.stdout).trim().to_string();

    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        let out = Command::new(&built).output().expect("timed run");
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        assert!(out.status.success());
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), expected);
        samples.push(ms);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (expected, samples[samples.len() / 2])
}

#[test]
fn test_ng_w15_1_bench_ng_15_aot_parity_exact_151024993() {
    let path = suite_v2_path("ffi_struct_array_pingpong.lust");
    let source = fs::read_to_string(&path).expect("read ffi_struct_array_pingpong.lust");

    let (stdout, elapsed_ms) = compile_and_run_rust_aot(&source, "bench_ng_15");
    assert_eq!(
        stdout, "151024993",
        "BENCH-NG-15 Rust AOT must match oracle checksum 151024993 (was 959025384 due to core::mem::replace stealing `win`)"
    );
    eprintln!("[NG-W15-1] BENCH-NG-15 Rust AOT OK: stdout={stdout}, elapsed={elapsed_ms:.3} ms");
}

#[test]
fn test_ng_w15_1_bench_ng_11_defer_effect_thunk_zero_rc_cell_and_under_14ms() {
    let path = suite_v2_path("defer_effect_thunk.lust");
    let source = fs::read_to_string(&path).expect("read defer_effect_thunk.lust");

    let opts = RustCompileOptions::default();
    let emitted = AotCompiler::emit_rust_source(&source, &opts)
        .expect("emit Rust source for defer_effect_thunk.lust");

    // Extract compute_with_defer body from emitted Rust
    let fn_start = emitted
        .find("fn __lust_fn_compute_with_defer(")
        .expect("emitted Rust must contain fn __lust_fn_compute_with_defer");
    let after_start = &emitted[fn_start..];
    let fn_end = after_start[1..]
        .find("\nfn ")
        .map(|i| i + 1)
        .unwrap_or(after_start.len());
    let fn_body = &after_start[..fn_end];

    assert!(
        !fn_body.contains("Rc::new(core::cell::Cell::new"),
        "compute_with_defer must not allocate Rc<Cell> on the fast path:\n{fn_body}"
    );
    assert!(
        !fn_body.contains("with_aot_scope"),
        "compute_with_defer must not wrap pure synchronous effect thunk in with_aot_scope:\n{fn_body}"
    );
    assert!(
        !fn_body.contains("lust_aot_rt::defer("),
        "compute_with_defer must keep AstLowering-inlined defer direct instead of emitting DropGuard:\n{fn_body}"
    );

    let (stdout, median_ms) = compile_and_measure_median_ms(&source, "bench_ng_11", 5);
    assert_eq!(
        stdout, "466147248",
        "BENCH-NG-11 Rust AOT must match oracle checksum 466147248"
    );
    assert!(
        median_ms <= 14.0,
        "BENCH-NG-11 Rust AOT median must be <= 14.0 ms (was 31.856 ms on 4761d45), got {median_ms:.3} ms"
    );
    eprintln!("[NG-W15-1] BENCH-NG-11 Rust AOT OK: stdout={stdout}, median={median_ms:.3} ms");
}

#[test]
fn test_ng_w15_1_direct_vec_copy_in_loop_preserves_source_vec() {
    let source = r#"
function sum_arr(xs: Array<int>): int
    local n: int = array.len(xs)
    local i: int = 0
    local s: int = 0
    while i < n do
        if xs[i] is Ok(v) then
            s = s + v
        end
        i = i + 1
    end
    return s
end

local data: Array<int> = [1, 2, 3, 4, 5]
local acc: int = 0
local k: int = 0
while k < 1000 do
    local cur: Array<int> = data
    acc = acc + sum_arr(cur)
    k = k + 1
end
println(acc)
"#;

    let opts = RustCompileOptions::default();
    let emitted = AotCompiler::emit_rust_source(source, &opts).expect("emit Rust source");
    let loop_start = emitted
        .find("'bb1: loop")
        .or_else(|| emitted.find(": loop {"))
        .expect("emitted Rust must contain a loop");
    let loop_and_after = &emitted[loop_start..];
    assert!(
        !loop_and_after.contains("core::mem::replace(&mut _"),
        "Emitted Rust must not destructively replace a live named Vec inside a loop:\n{loop_and_after}"
    );

    let (stdout, _) = compile_and_run_rust_aot(source, "direct_vec_loop");
    assert_eq!(stdout, "15000", "sum([1..5]) * 1000 must equal 15000");
}

#[test]
fn test_ng_w15_1_aot_string_sub_byte_gsub_split_parity() {
    let source = r##"
local s: string = "alpha=123,beta=4567,gamma=89"
local sub1: string = string.sub(s, 1, 5)
local sub2: string = string.sub(s, -2)
local parts: Array<string> = string.split(s, ",")
local gres: Array<string> = string.gsub(s, "[0-9]+", "#")
local bytes: Array<int> = string.byte(sub1, 1, 2)
local b0: int = 0
local b1: int = 0
if bytes[0] is Ok(v0) then b0 = v0 end
if bytes[1] is Ok(v1) then b1 = v1 end
local g0: string = ""
local g1: string = ""
if gres[0] is Ok(v) then g0 = v end
if gres[1] is Ok(v) then g1 = v end
local pcount: int = array.len(parts)
local fpos: Array<int> = string.find(s, "beta", 1, true)
local f0: int = 0
if fpos[0] is Ok(v) then f0 = v end
println(sub1 .. "|" .. sub2 .. "|" .. tostring(pcount) .. "|" .. g0 .. "|" .. g1 .. "|" .. tostring(b0) .. ":" .. tostring(b1) .. "|" .. tostring(f0))
"##;

    let (stdout, _) = compile_and_run_rust_aot(source, "string_intrinsics_parity");
    assert_eq!(
        stdout,
        "alpha|89|3|alpha=#,beta=#,gamma=#|3|97:108|11",
        "Rust AOT string.sub/byte/gsub/split/find output mismatch"
    );
}

#[test]
fn test_ng_w15_1_bench_ng_04_closure_upvalue_combinators_aot_parity_exact_947911() {
    let path = suite_v2_path("closure_upvalue_combinators.lust");
    let source = fs::read_to_string(&path).expect("read closure_upvalue_combinators.lust");

    let (stdout, elapsed_ms) = compile_and_run_rust_aot(&source, "bench_ng_04");
    assert_eq!(
        stdout, "947911",
        "BENCH-NG-04 Rust AOT must match oracle checksum 947911"
    );
    eprintln!("[NG-W15-1] BENCH-NG-04 Rust AOT OK: stdout={stdout}, elapsed={elapsed_ms:.3} ms");
}

#[test]
fn test_ng_w15_1_bench_ng_10_regex_match_extract_aot_parity_exact_240000() {
    let path = suite_v2_path("regex_match_extract.lust");
    let source = fs::read_to_string(&path).expect("read regex_match_extract.lust");

    let (stdout, elapsed_ms) = compile_and_run_rust_aot(&source, "bench_ng_10");
    assert_eq!(
        stdout, "240000",
        "BENCH-NG-10 Rust AOT must match oracle checksum 240000"
    );
    eprintln!("[NG-W15-1] BENCH-NG-10 Rust AOT OK: stdout={stdout}, elapsed={elapsed_ms:.3} ms");
}

#[test]
fn test_ng_w15_1_bench_ng_13_schema_json_decode_pipeline_aot_parity_exact_8224993() {
    let path = suite_v2_path("schema_json_decode_pipeline.lust");
    let source = fs::read_to_string(&path).expect("read schema_json_decode_pipeline.lust");

    let (stdout, elapsed_ms) = compile_and_run_rust_aot(&source, "bench_ng_13");
    assert_eq!(
        stdout, "8224993",
        "BENCH-NG-13 Rust AOT must match oracle checksum 8224993"
    );
    eprintln!("[NG-W15-1] BENCH-NG-13 Rust AOT OK: stdout={stdout}, elapsed={elapsed_ms:.3} ms");
}

#[test]
fn test_ng_w15_1_aot_non_repr_struct_decode_json_parity() {
    let source = r#"
struct TelemetryEvent
    ts: int
    device_id: int
    metric: int
    active: bool
end

local raw: string = "{\"ts\":1700000100,\"device_id\":42,\"metric\":950,\"active\":true}"
local ev: TelemetryEvent = TelemetryEvent.decode(raw)
if ev.active then
    println(ev.metric * ev.device_id + (ev.ts % 1000))
else
    println(0)
end
"#;

    let (stdout, _) = compile_and_run_rust_aot(source, "struct_decode_json");
    assert_eq!(stdout, "40000", "TelemetryEvent.decode(raw) parity mismatch");
}

#[test]
fn test_ng_w15_1_aot_unknown_field_method_luavalue_and_queue_intrinsics() {
    // Construct a MirModule exercising the exact MIR intrinsics that NG-W15-4 emits for
    // BENCH-NG-05 (`__mir_unknown_get_field` / `__mir_unknown_call_method`),
    // BENCH-NG-09 (`__mir_unknown_is_int` / `__mir_unknown_unwrap_int`), and
    // BENCH-NG-12 (`Queue` + `__mir_unknown_get_field` + dynamic closure call).
    let span = Span::dummy();
    let mut module = MirModule::new("ng_w15_1_intrinsics".into());

    let layout_a = LogicalLayoutV1::new(
        1,
        "NodeA",
        vec![
            LogicalFieldDesc::new(0, "x", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(1, "w", CanonicalTypeTag::Int64),
        ],
    );
    module.register_struct_layout(MirStructLayout::from(&layout_a));

    // Method `NodeA:score(self: NodeA, k: int) -> int { return self.x * k + self.w }`
    let mut score_fn = MirFunction::new(
        FunctionSignature {
            name: "NodeA:score".into(),
            param_types: vec![MirType::Struct("NodeA".into()), MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let l_self = score_fn.params[0];
    let l_k = score_fn.params[1];
    score_fn.locals[l_self.0].name = Some("self".into());
    score_fn.locals[l_k.0].name = Some("k".into());
    let l_x = score_fn.add_local(None, MirType::Int, true, span);
    let l_w = score_fn.add_local(None, MirType::Int, true, span);
    let l_mul = score_fn.add_local(None, MirType::Int, true, span);
    let l_sum = score_fn.add_local(None, MirType::Int, true, span);
    let bb0 = score_fn.entry;
    score_fn.blocks[bb0.0].statements.push(Statement::Assign(
        Place::Local(l_x),
        Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(l_self)), 0))),
        span,
    ));
    score_fn.blocks[bb0.0].statements.push(Statement::Assign(
        Place::Local(l_w),
        Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(l_self)), 1))),
        span,
    ));
    score_fn.blocks[bb0.0].statements.push(Statement::Assign(
        Place::Local(l_mul),
        Rvalue::BinaryOp(
            MirBinaryOp::Mul,
            Operand::Copy(Place::Local(l_x)),
            Operand::Copy(Place::Local(l_k)),
        ),
        span,
    ));
    score_fn.blocks[bb0.0].statements.push(Statement::Assign(
        Place::Local(l_sum),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(l_mul)),
            Operand::Copy(Place::Local(l_w)),
        ),
        span,
    ));
    score_fn.blocks[bb0.0].terminator =
        Some(Terminator::Return(Some(Operand::Copy(Place::Local(l_sum))), span));
    module.add_function(score_fn);

    // Function `main()` exercising all dynamic intrinsics + Queue
    let mut main_fn = MirFunction::new(
        FunctionSignature {
            name: "main".into(),
            param_types: vec![],
            return_type: MirType::Unit,
            is_public: true,
            effects: None,
        },
        span,
    );
    let l_node = main_fn.add_local(Some("a".into()), MirType::Struct("NodeA".into()), true, span);
    let l_obj = main_fn.add_local(Some("obj".into()), MirType::Unknown, true, span);
    let l_fx = main_fn.add_local(Some("fx".into()), MirType::Unknown, true, span);
    let l_is_int = main_fn.add_local(Some("is_int".into()), MirType::Bool, true, span);
    let l_xi = main_fn.add_local(Some("xi".into()), MirType::Int, true, span);
    let l_score = main_fn.add_local(Some("sc".into()), MirType::Int, true, span);
    let l_q = main_fn.add_local(Some("q".into()), MirType::Unknown, true, span);
    let l_offer = main_fn.add_local(Some("offer".into()), MirType::Unknown, true, span);
    let l_take = main_fn.add_local(Some("take".into()), MirType::Unknown, true, span);
    let l_ignored = main_fn.add_local(None, MirType::Unit, true, span);
    let l_qval = main_fn.add_local(Some("qval".into()), MirType::Int, true, span);
    let l_out = main_fn.add_local(Some("out".into()), MirType::Int, true, span);
    let l_tmp = main_fn.add_local(None, MirType::Int, true, span);

    let b0 = main_fn.entry;
    let b1 = main_fn.new_block();
    let b2 = main_fn.new_block();
    let b3 = main_fn.new_block();
    let b4 = main_fn.new_block();
    let b5 = main_fn.new_block();
    let b6 = main_fn.new_block();
    let b7 = main_fn.new_block();
    let b8 = main_fn.new_block();
    let b9 = main_fn.new_block();
    let b10 = main_fn.new_block();

    main_fn.blocks[b0.0].statements.push(Statement::Assign(
        Place::Local(l_node),
        Rvalue::Aggregate(
            MirType::Struct("NodeA".into()),
            vec![
                Operand::Constant(MirConstant::Int(3)),
                Operand::Constant(MirConstant::Int(10)),
            ],
        ),
        span,
    ));
    main_fn.blocks[b0.0].statements.push(Statement::Assign(
        Place::Local(l_obj),
        Rvalue::Use(Operand::Copy(Place::Local(l_node))),
        span,
    ));
    main_fn.blocks[b0.0].terminator = Some(Terminator::Call {
        callee: "__mir_unknown_get_field".into(),
        args: vec![
            Operand::Copy(Place::Local(l_obj)),
            Operand::Constant(MirConstant::String("x".into())),
        ],
        destination: Place::Local(l_fx),
        success: b1,
        error: None,
        span,
    });
    main_fn.blocks[b1.0].terminator = Some(Terminator::Call {
        callee: "__mir_unknown_is_int".into(),
        args: vec![Operand::Copy(Place::Local(l_fx))],
        destination: Place::Local(l_is_int),
        success: b2,
        error: None,
        span,
    });
    main_fn.blocks[b2.0].terminator = Some(Terminator::Call {
        callee: "__mir_unknown_unwrap_int".into(),
        args: vec![Operand::Copy(Place::Local(l_fx))],
        destination: Place::Local(l_xi),
        success: b3,
        error: None,
        span,
    });
    main_fn.blocks[b3.0].terminator = Some(Terminator::Call {
        callee: "__mir_unknown_call_method".into(),
        args: vec![
            Operand::Copy(Place::Local(l_obj)),
            Operand::Constant(MirConstant::String("score".into())),
            Operand::Constant(MirConstant::Int(5)),
        ],
        destination: Place::Local(l_score),
        success: b4,
        error: None,
        span,
    });
    main_fn.blocks[b4.0].terminator = Some(Terminator::Call {
        callee: "Queue".into(),
        args: vec![
            Operand::Constant(MirConstant::Int(8)),
            Operand::Constant(MirConstant::String("suspend".into())),
        ],
        destination: Place::Local(l_q),
        success: b5,
        error: None,
        span,
    });
    main_fn.blocks[b5.0].terminator = Some(Terminator::Call {
        callee: "__mir_unknown_get_field".into(),
        args: vec![
            Operand::Copy(Place::Local(l_q)),
            Operand::Constant(MirConstant::String("offer".into())),
        ],
        destination: Place::Local(l_offer),
        success: b6,
        error: None,
        span,
    });
    main_fn.blocks[b6.0].terminator = Some(Terminator::Call {
        callee: "__mir_unknown_get_field".into(),
        args: vec![
            Operand::Copy(Place::Local(l_q)),
            Operand::Constant(MirConstant::String("take".into())),
        ],
        destination: Place::Local(l_take),
        success: b7,
        error: None,
        span,
    });
    main_fn.blocks[b7.0].terminator = Some(Terminator::Call {
        callee: "offer".into(),
        args: vec![Operand::Constant(MirConstant::Int(100))],
        destination: Place::Local(l_ignored),
        success: b8,
        error: None,
        span,
    });
    main_fn.blocks[b8.0].terminator = Some(Terminator::Call {
        callee: "take".into(),
        args: vec![],
        destination: Place::Local(l_qval),
        success: b9,
        error: None,
        span,
    });
    main_fn.blocks[b9.0].statements.push(Statement::Assign(
        Place::Local(l_tmp),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(l_xi)),
            Operand::Copy(Place::Local(l_score)),
        ),
        span,
    ));
    main_fn.blocks[b9.0].statements.push(Statement::Assign(
        Place::Local(l_out),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(l_tmp)),
            Operand::Copy(Place::Local(l_qval)),
        ),
        span,
    ));
    main_fn.blocks[b9.0].terminator = Some(Terminator::Call {
        callee: "println".into(),
        args: vec![Operand::Copy(Place::Local(l_out))],
        destination: Place::Local(l_ignored),
        success: b10,
        error: None,
        span,
    });
    main_fn.blocks[b10.0].terminator = Some(Terminator::Return(None, span));
    module.add_function(main_fn);

    let dir = tempdir().expect("tempdir");
    let bin_path = dir.path().join("unknown_intrinsics");
    let opts = RustCompileOptions::default();
    let built = AotCompiler::compile_module_rust(&module, &bin_path, &opts)
        .expect("compile_module_rust for unknown intrinsics");
    let out = Command::new(&built).output().expect("run binary");
    assert!(
        out.status.success(),
        "unknown_intrinsics failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    // x(3) + score(3*5+10=25) + qval(100) = 128
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "128");
}

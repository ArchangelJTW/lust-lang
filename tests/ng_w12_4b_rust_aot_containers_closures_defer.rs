//! Integration, parity & invariant test suite for `NG-W12-4B`:
//! Rust AOT Tier 4 (`src/aot/rust_emitter.rs` + `crates/lust-aot-rt`):
//! 1. Local aliasing (`Rc<RefCell<...>>` / `Rc<Cell<...>>`, Hazard 12)
//! 2. Typed & dynamic `Map<K, V>` (`lust_aot_rt::AotMap`)
//! 3. User `Enum` (`module.enum_defs` -> `pub enum LustEnum_<Name>`) + `match`/`Switch`
//! 4. `Closure` (`Rvalue::Closure` + lifted `__lust_lambda_N`) with immutable & mutable upvalues
//! 5. Dynamic `AotValue` fallback (`MirType::Unknown`)
//! 6. `defer` & `errdefer` via `core::mem::DropGuard` under `-C panic=abort` (Hazard 9)

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::tempdir;

use lust::ast::Span;
use lust::codegen_zig::aot::{
    AotCompileOptions, AotCompiler, AotTargetKind, RustCompileOptions,
};
use lust::mir::ir::{
    FunctionSignature, MirBinaryOp, MirCleanupEntry, MirCleanupKind, MirConstant, MirEnumDef,
    MirFunction, MirModule, MirType, Operand, Place, Rvalue, Statement, Terminator,
};

fn lust_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_lust"))
}

fn run_vm_stdout(script_path: &Path) -> String {
    let out = Command::new(lust_bin())
        .arg("run")
        .arg(script_path)
        .arg("--backend")
        .arg("vm")
        .arg("--jit=off")
        .output()
        .expect("invoke lust vm");
    assert!(
        out.status.success(),
        "lust vm failed on {}:\nstdout: {}\nstderr: {}",
        script_path.display(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf8 stdout")
}

fn rust_opts(cache_dir: &Path, max_memory_bytes: Option<usize>) -> RustCompileOptions {
    RustCompileOptions {
        aot: AotCompileOptions {
            target_kind: AotTargetKind::Executable,
            release_mode: "ReleaseFast".to_string(),
            strip: false,
            max_memory_bytes,
            target_triple: None,
        },
        cache_dir: Some(cache_dir.to_path_buf()),
        ..RustCompileOptions::default()
    }
}

fn assert_panic_and_unsafe_free(code: &str, label: &str) {
    assert!(
        !code.contains("unsafe"),
        "[{label}] emitted Rust must contain ZERO `unsafe` blocks:\n{code}"
    );
    assert!(
        !code.contains(".unwrap("),
        "[{label}] emitted Rust must contain ZERO `.unwrap()` calls (Hazard 9):\n{code}"
    );
    assert!(
        !code.contains("panic!("),
        "[{label}] emitted Rust must contain ZERO `panic!()` calls (Hazard 9):\n{code}"
    );
}

// ---------------------------------------------------------------------------
// Test 1: Local Aliasing via Rc<RefCell<...>> (Hazard 12: Arrays & Structs)
// ---------------------------------------------------------------------------

#[test]
fn test_tier4_local_aliasing_arrays_and_mutable_structs_vm_parity() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");
    let opts = rust_opts(&cache_dir, None);

    let src = r#"
struct Boxed
    val: int
end

function mutate_and_read(x: Array<int>, y: Array<int>): int
    array.push(x, 4)
    return y[3]:unwrap()
end

local a: Array<int> = [1, 2]
local b: Array<int> = a
array.push(a, 3)
assert(b[2]:unwrap() == 3)
println(b[2]:unwrap())
println(array.len(b))

b[0] = 99
assert(a[0]:unwrap() == 99)
println(a[0]:unwrap())

local from_call: int = mutate_and_read(a, b)
assert(from_call == 4)
println(from_call)

local p: Boxed = Boxed { val = 10 }
local q: Boxed = p
p.val = 42
assert(q.val == 42)
println(q.val)
"#;

    let script = tmp.path().join("aliasing.lust");
    fs::write(&script, src).expect("write aliasing.lust");
    let vm_stdout = run_vm_stdout(&script);

    let emitted = AotCompiler::emit_rust_source(src, &opts).expect("emit aliasing rust");
    assert_panic_and_unsafe_free(&emitted, "aliasing");
    assert!(
        emitted.contains("alloc::rc::Rc<core::cell::RefCell<"),
        "aliased array/struct must emit Rc<RefCell<...>> (Hazard 12):\n{emitted}"
    );
    assert!(
        emitted.contains("alloc::rc::Rc::clone("),
        "aliased assignment `local b = a` must clone Rc handle:\n{emitted}"
    );

    let bin = tmp.path().join("aliasing_bin");
    AotCompiler::compile_source_rust(src, &bin, &opts).expect("compile aliasing bin");
    let run = Command::new(&bin).output().expect("run aliasing bin");
    assert!(
        run.status.success(),
        "aliasing bin failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let aot_stdout = String::from_utf8(run.stdout).expect("utf8");
    assert_eq!(aot_stdout, "3\n3\n99\n4\n42\n");
    assert_eq!(
        aot_stdout, vm_stdout,
        "Rust AOT must match VM bit-exact on local aliasing"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Map<K, V> (AotMap<K, V>) — VM parity, aliasing, MissingMapKey, OOM
// ---------------------------------------------------------------------------

#[test]
fn test_tier4_map_operations_aliasing_errors_and_vm_parity() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");
    let opts = rust_opts(&cache_dir, None);

    let src = r#"
function b08_map_rw(n: int): int
    local m: Map<int, int> = {}
    local i: int = 0
    while i < n do
        map.set(m, i % 64, (i * 7 + 3) % 1009)
        i = i + 1
    end
    local sum: int = 0
    i = 0
    while i < n do
        local k: int = (i * 13) % 64
        local v: int = map.get(m, k):unwrap()
        map.set(m, k, v + 1)
        sum = sum + v
        i = i + 1
    end
    return sum
end

local sm: Map<string, int> = { ["alpha"] = 10, ["beta"] = 20 }
local sm_alias: Map<string, int> = sm
map.set(sm, "gamma", 30)
assert(map.has(sm_alias, "gamma"))
assert(map.get(sm_alias, "gamma"):unwrap() == 30)
println(map.len(sm_alias))
println(map.get(sm_alias, "alpha"):unwrap() + map.get(sm, "beta"):unwrap())
map.delete(sm, "beta")
println(map.has(sm_alias, "beta"))
println(map.len(sm))
println(b08_map_rw(200))
"#;

    let script = tmp.path().join("maps.lust");
    fs::write(&script, src).expect("write maps.lust");
    let vm_stdout = run_vm_stdout(&script);

    let emitted = AotCompiler::emit_rust_source(src, &opts).expect("emit maps rust");
    assert_panic_and_unsafe_free(&emitted, "maps");
    assert!(
        emitted.contains("lust_aot_rt::AotMap<"),
        "Map<K, V> must emit lust_aot_rt::AotMap:\n{emitted}"
    );

    let bin = tmp.path().join("maps_bin");
    AotCompiler::compile_source_rust(src, &bin, &opts).expect("compile maps bin");
    let run = Command::new(&bin).output().expect("run maps bin");
    assert!(
        run.status.success(),
        "maps bin failed:\nstderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let aot_stdout = String::from_utf8(run.stdout).expect("utf8");
    assert_eq!(aot_stdout, vm_stdout, "Map<K, V> AOT must match VM bit-exact");

    // MissingMapKey returns structured error (exit 1, never panics)
    let missing_src = r#"
local m: Map<string, int> = {}
local x: int = map.get(m, "nope"):unwrap()
println(x)
"#;
    let missing_bin = tmp.path().join("map_missing_bin");
    AotCompiler::compile_source_rust(missing_src, &missing_bin, &opts)
        .expect("compile missing map key");
    let r_miss = Command::new(&missing_bin).output().expect("run missing");
    assert_eq!(r_miss.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&r_miss.stderr);
    assert!(
        stderr.contains("Missing map key"),
        "expected Missing map key error, got: {stderr}"
    );

    // Budget enforcement (--max-memory 128) trips OutOfMemory on Map growth
    let oom_map_src = r#"
local m: Map<int, int> = {}
local i: int = 0
while i < 1000 do
    map.set(m, i, i * 2)
    i = i + 1
end
println(map.len(m))
"#;
    let oom_bin = tmp.path().join("map_oom_bin");
    let oom_opts = rust_opts(&cache_dir, Some(128));
    AotCompiler::compile_source_rust(oom_map_src, &oom_bin, &oom_opts)
        .expect("compile map oom");
    let r_oom = Command::new(&oom_bin).output().expect("run map oom");
    assert_eq!(r_oom.status.code(), Some(1));
    let oom_err = String::from_utf8_lossy(&r_oom.stderr);
    assert!(
        oom_err.contains("Out of memory"),
        "expected Out of memory on map growth, got: {oom_err}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: User Enum (`module.enum_defs` -> `pub enum LustEnum_Shape`) + match/Switch
// ---------------------------------------------------------------------------

#[test]
fn test_tier4_user_enum_definition_construction_and_match_switch() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");
    let opts = rust_opts(&cache_dir, None);
    let span = Span::dummy();

    let mut module = MirModule::new("enum_test_mod".to_string());
    module.register_enum_def(MirEnumDef {
        name: "Shape".to_string(),
        variants: vec![
            ("Point".to_string(), vec![]),
            ("Circle".to_string(), vec![MirType::Float]),
            ("Rect".to_string(), vec![MirType::Int, MirType::Int]),
        ],
    });

    // Helper: score(s: Shape) -> int
    //   Point => 1
    //   Circle(r) => (r * 10.0) as int
    //   Rect(w, h) => w * h
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

    let bb_sw = score_fn.new_block(); // 1
    let bb_pt = score_fn.new_block(); // 2
    let bb_ci = score_fn.new_block(); // 3
    let bb_ci2 = score_fn.new_block(); // 4
    let bb_rc = score_fn.new_block(); // 5
    let bb_rc2 = score_fn.new_block(); // 6
    let bb_rc3 = score_fn.new_block(); // 7
    let bb_def = score_fn.new_block(); // 8

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

    module.add_function(score_fn);

    // main(): constructs Point (Aggregate), Circle(3.5) (Shape.Circle call), Rect(6, 7) (enum.new),
    // prints Display + score(s) for each.
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
    let b9 = main_fn.new_block();

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
    main_fn.blocks[b8.0].terminator = Some(Terminator::Goto(b9, span));
    main_fn.blocks[b9.0].terminator = Some(Terminator::Return(
        Some(Operand::Constant(MirConstant::Int(0))),
        span,
    ));

    module.add_function(main_fn);

    let emitted = AotCompiler::emit_rust_module(&module, &opts).expect("emit enum rust");
    assert_panic_and_unsafe_free(&emitted, "user_enum");
    assert!(
        emitted.contains("pub enum LustEnum_Shape"),
        "must emit native pub enum LustEnum_Shape:\n{emitted}"
    );
    assert!(emitted.contains("Point,"), "missing Point variant:\n{emitted}");
    assert!(
        emitted.contains("Circle(f64),"),
        "missing Circle(f64) variant:\n{emitted}"
    );
    assert!(
        emitted.contains("Rect(i64, i64),"),
        "missing Rect(i64, i64) variant:\n{emitted}"
    );

    let bin = tmp.path().join("enum_bin");
    AotCompiler::compile_module_rust(&module, &bin, &opts).expect("compile enum bin");
    let run = Command::new(&bin).output().expect("run enum bin");
    assert!(
        run.status.success(),
        "enum bin failed:\nstderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout = String::from_utf8(run.stdout).expect("utf8");
    assert_eq!(stdout, "Shape.Point\t1\nShape.Circle(3.5)\t35\nShape.Rect(6, 7)\t42\n");
}

// ---------------------------------------------------------------------------
// Test 4: Closures (`Rvalue::Closure`) with Immutable & Mutable Upvalues
// ---------------------------------------------------------------------------

#[test]
fn test_tier4_closures_with_immutable_and_mutable_upvalues_vm_parity() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");
    let opts = rust_opts(&cache_dir, None);

    let src = r#"
function run_adder(offset: int): int
    local add = function(x: int): int
        return x + offset
    end
    return add(32)
end

function run_counter(): int
    local count: int = 0
    local inc = function(): int
        count = count + 1
        return count
    end
    local a: int = inc()
    local b: int = inc()
    local c: int = inc()
    return a * 100 + b * 10 + c + count
end

function make_counter(start: int): function(): int
    local n: int = start
    return function(): int
        n = n + 1
        return n
    end
end

function test_shared_array_capture(): int
    local log: Array<int> = []
    local push_val = function(x: int): int
        array.push(log, x)
        return array.len(log)
    end
    local n1: int = push_val(10)
    local n2: int = push_val(20)
    return n1 + n2 + log[0]:unwrap() + log[1]:unwrap()
end

println(run_adder(10))
println(run_counter())
local c1 = make_counter(10)
println(c1())
println(c1())
local c2 = make_counter(100)
println(c2())
println(c1())
println(test_shared_array_capture())
"#;

    let script = tmp.path().join("closures.lust");
    fs::write(&script, src).expect("write closures.lust");
    let vm_stdout = run_vm_stdout(&script);

    let emitted = AotCompiler::emit_rust_source(src, &opts).expect("emit closures rust");
    assert_panic_and_unsafe_free(&emitted, "closures");
    assert!(
        emitted.contains("__lust_fn___lust_lambda_"),
        "closures must emit lifted __lust_lambda_N functions:\n{emitted}"
    );
    assert!(
        emitted.contains("core::cell::Cell<i64>"),
        "mutable int upvalue must use Rc<Cell<i64>>:\n{emitted}"
    );
    assert!(
        emitted.contains("core::cell::RefCell<"),
        "captured container upvalue must use Rc<RefCell<...>>:\n{emitted}"
    );

    let bin = tmp.path().join("closures_bin");
    AotCompiler::compile_source_rust(src, &bin, &opts).expect("compile closures bin");
    let run = Command::new(&bin).output().expect("run closures bin");
    assert!(
        run.status.success(),
        "closures bin failed:\nstderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let aot_stdout = String::from_utf8(run.stdout).expect("utf8");
    assert_eq!(aot_stdout, "42\n123\n11\n12\n101\n13\n33\n");
    assert_eq!(
        aot_stdout, vm_stdout,
        "Closures with upvalues must match VM bit-exact"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Dynamic AotValue Fallback (`MirType::Unknown`)
// ---------------------------------------------------------------------------

#[test]
fn test_tier4_dynamic_aot_value_fallback() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");
    let opts = rust_opts(&cache_dir, None);
    let span = Span::dummy();

    let mut module = MirModule::new("dyn_val_mod".to_string());
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

    // Heterogeneous dynamic local `v: Unknown` assigned both Int and String (conflicting types -> AotValue)
    // plus heterogeneous Map<string, Unknown> holding both Int(42) and String("ok").
    let v = main_fn.add_local(Some("v".into()), MirType::Unknown, true, span);
    let b = main_fn.add_local(Some("b".into()), MirType::Unknown, true, span);
    let sum = main_fn.add_local(Some("sum".into()), MirType::Unknown, true, span);
    let m = main_fn.add_local(
        Some("m".into()),
        MirType::Map(Box::new(MirType::String), Box::new(MirType::Unknown)),
        true,
        span,
    );
    let got_num = main_fn.add_local(Some("got_num".into()), MirType::Unknown, true, span);
    let got_tag = main_fn.add_local(Some("got_tag".into()), MirType::Unknown, true, span);
    let unit_tmp = main_fn.add_local(None, MirType::Unit, true, span);

    let bb1 = main_fn.new_block();
    let bb2 = main_fn.new_block();
    let bb3 = main_fn.new_block();
    let bb4 = main_fn.new_block();
    let bb5 = main_fn.new_block();
    let bb6 = main_fn.new_block();
    let bb7 = main_fn.new_block();
    let bb8 = main_fn.new_block();

    main_fn.blocks[0].statements.push(Statement::Assign(
        Place::Local(v),
        Rvalue::Use(Operand::Constant(MirConstant::Int(19))),
        span,
    ));
    main_fn.blocks[0].statements.push(Statement::Assign(
        Place::Local(b),
        Rvalue::Use(Operand::Constant(MirConstant::Int(23))),
        span,
    ));
    main_fn.blocks[0].statements.push(Statement::Assign(
        Place::Local(sum),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(v)),
            Operand::Copy(Place::Local(b)),
        ),
        span,
    ));
    // Reassign `v` to a String so `v` is genuinely heterogeneous (stays MirType::Unknown -> AotValue!)
    main_fn.blocks[0].statements.push(Statement::Assign(
        Place::Local(v),
        Rvalue::Use(Operand::Constant(MirConstant::String("ok".into()))),
        span,
    ));
    main_fn.blocks[0].terminator = Some(Terminator::Call {
        callee: "map.new".into(),
        args: vec![],
        destination: Place::Local(m),
        success: bb1,
        error: None,
        span,
    });
    main_fn.blocks[bb1.0].terminator = Some(Terminator::Call {
        callee: "map.set".into(),
        args: vec![
            Operand::Copy(Place::Local(m)),
            Operand::Constant(MirConstant::String("ans".into())),
            Operand::Copy(Place::Local(sum)),
        ],
        destination: Place::Local(unit_tmp),
        success: bb2,
        error: None,
        span,
    });
    main_fn.blocks[bb2.0].terminator = Some(Terminator::Call {
        callee: "map.set".into(),
        args: vec![
            Operand::Copy(Place::Local(m)),
            Operand::Constant(MirConstant::String("tag".into())),
            Operand::Copy(Place::Local(v)),
        ],
        destination: Place::Local(unit_tmp),
        success: bb3,
        error: None,
        span,
    });
    main_fn.blocks[bb3.0].terminator = Some(Terminator::Call {
        callee: "map.get".into(),
        args: vec![
            Operand::Copy(Place::Local(m)),
            Operand::Constant(MirConstant::String("ans".into())),
        ],
        destination: Place::Local(got_num),
        success: bb4,
        error: None,
        span,
    });
    main_fn.blocks[bb4.0].terminator = Some(Terminator::Call {
        callee: "map.get".into(),
        args: vec![
            Operand::Copy(Place::Local(m)),
            Operand::Constant(MirConstant::String("tag".into())),
        ],
        destination: Place::Local(got_tag),
        success: bb5,
        error: None,
        span,
    });
    main_fn.blocks[bb5.0].terminator = Some(Terminator::Call {
        callee: "println".into(),
        args: vec![
            Operand::Copy(Place::Local(got_num)),
            Operand::Copy(Place::Local(got_tag)),
        ],
        destination: Place::Local(unit_tmp),
        success: bb6,
        error: None,
        span,
    });
    main_fn.blocks[bb6.0].terminator = Some(Terminator::Goto(bb7, span));
    main_fn.blocks[bb7.0].terminator = Some(Terminator::Goto(bb8, span));
    main_fn.blocks[bb8.0].terminator = Some(Terminator::Return(
        Some(Operand::Constant(MirConstant::Int(0))),
        span,
    ));

    module.add_function(main_fn);

    let emitted = AotCompiler::emit_rust_module(&module, &opts).expect("emit dyn rust");
    assert_panic_and_unsafe_free(&emitted, "dynamic_aot_value");
    assert!(
        emitted.contains("lust_aot_rt::AotValue"),
        "heterogeneous MirType::Unknown must lower to lust_aot_rt::AotValue:\n{emitted}"
    );

    let bin = tmp.path().join("dyn_bin");
    AotCompiler::compile_module_rust(&module, &bin, &opts).expect("compile dyn bin");
    let run = Command::new(&bin).output().expect("run dyn bin");
    assert!(
        run.status.success(),
        "dyn bin failed:\nstderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(String::from_utf8(run.stdout).expect("utf8"), "42\tok\n");
}

// ---------------------------------------------------------------------------
// Test 6: `defer` & `errdefer` via `core::mem::DropGuard` under `-C panic=abort`
// ---------------------------------------------------------------------------

#[test]
fn test_tier4_defer_and_errdefer_dropguard_ok_and_error_lifo_under_panic_abort() {
    let tmp = tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("rlib_cache");
    let opts = rust_opts(&cache_dir, None);

    // 6A. Success path: errdefer is dismissed via core::mem::DropGuard::dismiss,
    //     defers execute ONCE each in LIFO order (bit-exact VM parity!).
    let ok_src = r#"
function run_ok(): int
    defer {
        println("defer_1")
    }
    errdefer {
        println("errdefer_1")
    }
    defer {
        println("defer_2")
    }
    println("body_ok")
    return 42
end

local r: int = run_ok()
println(r)
"#;

    let ok_script = tmp.path().join("defer_ok.lust");
    fs::write(&ok_script, ok_src).expect("write defer_ok.lust");
    let vm_ok_stdout = run_vm_stdout(&ok_script);

    let emitted_ok = AotCompiler::emit_rust_source(ok_src, &opts).expect("emit defer_ok");
    assert_panic_and_unsafe_free(&emitted_ok, "defer_ok");
    assert!(
        emitted_ok.contains("lust_aot_rt::defer("),
        "must emit lust_aot_rt::defer:\n{emitted_ok}"
    );
    assert!(
        emitted_ok.contains("lust_aot_rt::errdefer("),
        "must emit lust_aot_rt::errdefer:\n{emitted_ok}"
    );
    assert!(
        emitted_ok.contains("core::mem::DropGuard::dismiss("),
        "must emit core::mem::DropGuard::dismiss before Ok return:\n{emitted_ok}"
    );

    let ok_bin = tmp.path().join("defer_ok_bin");
    AotCompiler::compile_source_rust(ok_src, &ok_bin, &opts).expect("compile defer_ok_bin");
    let r_ok = Command::new(&ok_bin).output().expect("run defer_ok_bin");
    assert!(
        r_ok.status.success(),
        "defer_ok_bin failed:\nstderr: {}",
        String::from_utf8_lossy(&r_ok.stderr)
    );
    let aot_ok_stdout = String::from_utf8(r_ok.stdout).expect("utf8");
    assert_eq!(aot_ok_stdout, "body_ok\ndefer_2\ndefer_1\n42\n");
    assert_eq!(
        aot_ok_stdout, vm_ok_stdout,
        "defer/errdefer ok path must match VM bit-exact"
    );

    // 6B. Error path (division by zero via `?`): under `-C panic=abort`,
    //     BOTH errdefer and defer execute in exact interleaved LIFO order!
    let err_div_src = r#"
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
    let err_div_bin = tmp.path().join("defer_err_div_bin");
    AotCompiler::compile_source_rust(err_div_src, &err_div_bin, &opts)
        .expect("compile defer_err_div_bin");
    let r_err = Command::new(&err_div_bin)
        .output()
        .expect("run defer_err_div_bin");
    assert_eq!(r_err.status.code(), Some(1));
    let err_stdout = String::from_utf8(r_err.stdout).expect("utf8");
    let err_stderr = String::from_utf8_lossy(&r_err.stderr);
    assert_eq!(
        err_stdout, "body_err\ndefer_2\nerrdefer_1\ndefer_1\n",
        "on error exit under -C panic=abort, defer + errdefer must run in exact LIFO order"
    );
    assert!(
        err_stderr.contains("Division by zero"),
        "expected Division by zero stderr, got: {err_stderr}"
    );

    // 6C. Error path via `error("boom")` + captured state mutation inside defer/errdefer
    let err_user_src = r#"
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
    let err_user_bin = tmp.path().join("defer_err_user_bin");
    AotCompiler::compile_source_rust(err_user_src, &err_user_bin, &opts)
        .expect("compile defer_err_user_bin");
    let r_user = Command::new(&err_user_bin)
        .output()
        .expect("run defer_err_user_bin");
    assert_eq!(r_user.status.code(), Some(1));
    let user_stdout = String::from_utf8(r_user.stdout).expect("utf8");
    let user_stderr = String::from_utf8_lossy(&r_user.stderr);
    assert_eq!(
        user_stdout, "err_ran\n3\n",
        "errdefer must mutate shared log before defer reads updated length=3"
    );
    assert!(
        user_stderr.contains("boom"),
        "expected user error 'boom' on stderr, got: {user_stderr}"
    );

    // 6D. Pure MIR cleanup_blocks test (without AstLowering inline defer duplication)
    let span = Span::dummy();
    let mut mir_mod = MirModule::new("pure_cleanup_mod".to_string());
    let mut f = MirFunction::new(
        FunctionSignature {
            name: "main".to_string(),
            param_types: vec![],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let unit_tmp = f.add_local(None, MirType::Unit, true, span);
    f.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Constant(MirConstant::Int(0))),
        span,
    ));

    let bb_d = f.new_block();
    let bb_d_end = f.new_block();
    f.blocks[bb_d.0].is_cleanup = true;
    f.blocks[bb_d_end.0].is_cleanup = true;
    f.blocks[bb_d.0].terminator = Some(Terminator::Call {
        callee: "println".into(),
        args: vec![Operand::Constant(MirConstant::String("mir_defer".into()))],
        destination: Place::Local(unit_tmp),
        success: bb_d_end,
        error: None,
        span,
    });
    f.blocks[bb_d_end.0].terminator = Some(Terminator::Return(None, span));

    let bb_e = f.new_block();
    let bb_e_end = f.new_block();
    f.blocks[bb_e.0].is_cleanup = true;
    f.blocks[bb_e_end.0].is_cleanup = true;
    f.blocks[bb_e.0].terminator = Some(Terminator::Call {
        callee: "println".into(),
        args: vec![Operand::Constant(MirConstant::String("mir_errdefer".into()))],
        destination: Place::Local(unit_tmp),
        success: bb_e_end,
        error: None,
        span,
    });
    f.blocks[bb_e_end.0].terminator = Some(Terminator::Return(None, span));

    f.cleanup_blocks.push(MirCleanupEntry {
        kind: MirCleanupKind::Defer,
        block: bb_d,
    });
    f.cleanup_blocks.push(MirCleanupEntry {
        kind: MirCleanupKind::ErrDefer,
        block: bb_e,
    });
    mir_mod.add_function(f);

    let pure_bin = tmp.path().join("pure_cleanup_bin");
    AotCompiler::compile_module_rust(&mir_mod, &pure_bin, &opts).expect("compile pure cleanup");
    let r_pure = Command::new(&pure_bin).output().expect("run pure cleanup");
    assert!(r_pure.status.success());
    assert_eq!(String::from_utf8(r_pure.stdout).expect("utf8"), "mir_defer\n");
}

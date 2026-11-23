#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::sync::Mutex;

use lust::ast::Span;
use lust::bytecode::Value;
use lust::mir::cranelift_backend::{
    CraneliftJitCompiler, DeoptSiteKind, compute_default_module_roots,
    compute_reachable_native_functions, get_current_sticky_error, safepoint_request_deopt,
    safepoint_reset, take_current_sticky_error, take_deopt_payload,
};
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::safepoint_request_deopt_on_poll;
use lust::mir::ir::{
    FunctionSignature, MirBinaryOp, MirConstant, MirFunction, MirModule, MirType, Operand, Place,
    Rvalue, Statement, Terminator,
};
use lust::mir::opt::{JitInlineConfig, MirOptimizer};
use lust::vfs::VirtualConsole;
#[cfg(feature = "test-deopt")]
use lust::vm::DeoptPayloadSource;
use lust::vm::{VM, reset_effect_handler_state};
use lust::{
    EmbeddedModule, EmbeddedProgram, LustConfig, compile_program_with_config,
    load_program_from_embedded,
};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn reset_all_state() {
    safepoint_reset();
    reset_effect_handler_state();
    take_deopt_payload();
    take_current_sticky_error();
}

fn build_vm(source: &str, cranelift: bool) -> VM {
    let entries = &[EmbeddedModule {
        module: "main",
        parent: None,
        source: Some(source),
    }];
    let program = load_program_from_embedded(entries, "main").expect("load embedded program");
    let mut cfg = LustConfig::default().with_cranelift_jit(cranelift);
    cfg.set_use_mir(false);
    let mut vm = compile_program_with_config(program, &cfg).expect("compile embedded program");
    vm.console = VirtualConsole::captured();
    vm
}

fn build_embedded_program(source: &str) -> EmbeddedProgram {
    let mut cfg = LustConfig::default().with_cranelift_jit(true);
    cfg.set_use_mir(false);
    EmbeddedProgram::builder()
        .with_config(cfg)
        .module("main", source)
        .entry_module("main")
        .compile()
        .expect("compile EmbeddedProgram")
}

fn make_add_const_fn(name: &str, delta: i64, is_public: bool) -> MirFunction {
    let span = Span::dummy();
    let mut f = MirFunction::new(
        FunctionSignature {
            name: name.into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public,
            effects: None,
        },
        span,
    );
    let p0 = f.params[0];
    let out = f.add_local(Some("out".into()), MirType::Int, false, span);
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(out),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(p0)),
            Operand::Constant(MirConstant::Int(delta)),
        ),
        span,
    ));
    f.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(out))),
        span,
    ));
    f
}

fn make_caller_fn(name: &str, callee: &str, is_public: bool) -> MirFunction {
    let span = Span::dummy();
    let mut f = MirFunction::new(
        FunctionSignature {
            name: name.into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public,
            effects: None,
        },
        span,
    );
    let p0 = f.params[0];
    let tmp = f.add_local(Some("tmp".into()), MirType::Int, false, span);
    let bb1 = f.new_block();
    f.blocks[0].terminator = Some(Terminator::Call {
        callee: callee.into(),
        args: vec![Operand::Copy(Place::Local(p0))],
        destination: Place::Local(tmp),
        success: bb1,
        error: None,
        span,
    });
    f.blocks[bb1.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(tmp))),
        span,
    ));
    f
}

// =========================================================================
// Test 1: Dead-Helper Cranelift DCE — Inlined-only helpers skipped,
//         exported / indirectly-called / closure / call_handle helpers kept.
// =========================================================================

#[test]
fn test_ng_w14_7_dead_inlined_helper_skipped_indirect_kept() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    // ---------------------------------------------------------------------
    // Part A.1: `calls.lust`-style script with 100%-inlined helper `step`
    //           skips Cranelift codegen for `step` when executing `__script`.
    // ---------------------------------------------------------------------
    let calls_src = r#"
function step(a: int, b: int): int
    return a + b * 2
end

local acc: int = 0
local i: int = 0
while i < 100 do
    acc = step(acc, i)
    i = i + 1
end
println(acc)
"#;

    let mut vm_bc = build_vm(calls_src, false);
    let mut vm_jit = build_vm(calls_src, true);

    vm_bc.call("__script", vec![]).expect("bc __script");
    vm_jit.call("__script", vec![]).expect("jit __script");

    assert_eq!(vm_jit.console.stdout_bytes(), vm_bc.console.stdout_bytes());
    let step_name = vm_jit
        .mir_functions
        .keys()
        .find(|k| k.ends_with("step"))
        .cloned()
        .expect("step in mir_functions");

    assert!(
        vm_jit.cranelift_compiled.contains_key("__script"),
        "__script must compile to Cranelift"
    );
    assert!(
        !vm_jit.cranelift_compiled.contains_key(&step_name),
        "100%-inlined helper `{step_name}` with 0 remaining callers must be skipped by Cranelift DCE"
    );
    assert_eq!(
        vm_jit.cranelift_compilation_count, 1,
        "only __script should be compiled in Cranelift (got {})",
        vm_jit.cranelift_compilation_count
    );
    assert!(
        vm_jit.cranelift_skipped_dead_helper_count() >= 1,
        "expected >= 1 skipped dead helper, got {}",
        vm_jit.cranelift_skipped_dead_helper_count()
    );
    assert!(vm_jit.cranelift_is_deferred_dead_helper(&step_name));

    // On-demand fallback: directly invoking previously-skipped helper compiles & runs it natively.
    let entries_before = vm_jit.cranelift_entry_count;
    let direct_step = vm_jit
        .call(&step_name, vec![Value::Int(10), Value::Int(5)])
        .expect("direct call to previously-pruned helper must compile on demand");
    assert_eq!(direct_step, Value::Int(20));
    assert!(
        vm_jit.cranelift_compiled.contains_key(&step_name),
        "directly invoked helper must now be compiled in Cranelift"
    );
    assert_eq!(vm_jit.cranelift_entry_count, entries_before + 1);

    // ---------------------------------------------------------------------
    // Part A.2: `methods.lust`-style script with 100%-inlined method `Counter:bump`
    //           skips Cranelift codegen for `Counter:bump` when executing `__script`.
    // ---------------------------------------------------------------------
    let methods_src = r#"
struct Counter
    value: int
end

impl Counter
    function bump(self, d: int): int
        self.value = self.value + d
        return self.value
    end
end

local c: Counter = Counter { value = 0 }
local last: int = 0
local i: int = 0
while i < 100 do
    last = c:bump(i)
    i = i + 1
end
println(last)
"#;

    let mut vm_m_bc = build_vm(methods_src, false);
    let mut vm_m_jit = build_vm(methods_src, true);

    vm_m_bc.call("__script", vec![]).expect("bc methods __script");
    vm_m_jit.call("__script", vec![]).expect("jit methods __script");
    assert_eq!(
        vm_m_jit.console.stdout_bytes(),
        vm_m_bc.console.stdout_bytes()
    );

    let bump_name = vm_m_jit
        .mir_functions
        .keys()
        .find(|k| k.ends_with("Counter:bump"))
        .cloned()
        .expect("Counter:bump in mir_functions");
    assert!(vm_m_jit.cranelift_compiled.contains_key("__script"));
    assert!(
        !vm_m_jit.cranelift_compiled.contains_key(&bump_name),
        "100%-inlined method `{bump_name}` with 0 remaining callers must be skipped by Cranelift DCE"
    );
    assert_eq!(vm_m_jit.cranelift_compilation_count, 1);
    assert!(vm_m_jit.cranelift_skipped_dead_helper_count() >= 1);
    assert!(vm_m_jit.cranelift_is_deferred_dead_helper(&bump_name));

    // ---------------------------------------------------------------------
    // Part B: Direct `CraneliftJitCompiler::compile_module` prunes private
    //         100%-inlined helpers while keeping exported & reachable helpers.
    // ---------------------------------------------------------------------
    let mut mir_mod = MirModule::new("dce_direct_mod".into());
    mir_mod.add_function(make_add_const_fn("dead_private_helper", 7, false));
    mir_mod.add_function(make_add_const_fn("exported_helper", 100, true));
    mir_mod.add_function(make_caller_fn("entry_fn", "dead_private_helper", true));

    let inlined = MirOptimizer::optimize_module_for_jit(&mut mir_mod, &JitInlineConfig::default());
    assert!(
        inlined.contains("entry_fn"),
        "entry_fn must inline dead_private_helper before DCE"
    );
    let default_roots = compute_default_module_roots(&mir_mod);
    let reachable = compute_reachable_native_functions(&mir_mod, &default_roots);
    assert!(reachable.contains("entry_fn"));
    assert!(reachable.contains("exported_helper"));
    assert!(
        !reachable.contains("dead_private_helper"),
        "dead private helper with 0 remaining callers must not be reachable"
    );

    let mut compiler = CraneliftJitCompiler::new().expect("CraneliftJitCompiler::new");
    let compiled = compiler
        .compile_module(&mir_mod)
        .expect("compile_module with dead-helper DCE");
    assert!(compiled.contains_key("entry_fn"));
    assert!(compiled.contains_key("exported_helper"));
    assert!(
        !compiled.contains_key("dead_private_helper"),
        "dead_private_helper must be skipped from Cranelift codegen"
    );
    assert_eq!(compiler.skipped_dead_helper_count(), 1);

    // ---------------------------------------------------------------------
    // Part C: Indirect calls / closures / interpreted callers MUST keep
    //         referenced helpers reachable and compiled.
    // ---------------------------------------------------------------------
    let indirect_src = r#"
function inlined_only(x: int): int
    return x + 1
end

function indirect_target(x: int): int
    return x * 3 + 2
end

function call_via_closure(x: int): int
    local cb = function(v: int): int
        return indirect_target(v)
    end
    return cb(x)
end

local acc: int = 0
local i: int = 0
while i < 50 do
    acc = acc + inlined_only(i)
    i = i + 1
end
println(acc)
"#;

    let mut vm_ind_bc = build_vm(indirect_src, false);
    let mut vm_ind_jit = build_vm(indirect_src, true);

    let inlined_only_name = vm_ind_jit
        .mir_functions
        .keys()
        .find(|k| k.ends_with("inlined_only"))
        .cloned()
        .expect("inlined_only in mir_functions");
    let indirect_target_name = vm_ind_jit
        .mir_functions
        .keys()
        .find(|k| k.ends_with("indirect_target"))
        .cloned()
        .expect("indirect_target in mir_functions");
    let closure_driver_name = if vm_ind_jit
        .function_by_name("main.call_via_closure")
        .is_some()
    {
        "main.call_via_closure".to_string()
    } else {
        "call_via_closure".to_string()
    };

    vm_ind_bc.call("__script", vec![]).expect("bc script");
    vm_ind_jit.call("__script", vec![]).expect("jit script");
    assert_eq!(
        vm_ind_jit.console.stdout_bytes(),
        vm_ind_bc.console.stdout_bytes()
    );

    assert!(
        !vm_ind_jit.cranelift_compiled.contains_key(&inlined_only_name),
        "inlined-only helper `{inlined_only_name}` must be skipped"
    );
    assert!(
        vm_ind_jit
            .cranelift_compiled
            .contains_key(&indirect_target_name),
        "indirectly-referenced / closure-called helper `{indirect_target_name}` MUST stay reachable and compiled"
    );

    let expected_cb = vm_ind_bc
        .call(&closure_driver_name, vec![Value::Int(11)])
        .expect("bc closure call");
    let entries_before_cb = vm_ind_jit.cranelift_entry_count;
    let actual_cb = vm_ind_jit
        .call(&closure_driver_name, vec![Value::Int(11)])
        .expect("jit closure call");
    assert_eq!(actual_cb, expected_cb);
    assert_eq!(actual_cb, Value::Int(35));
    assert!(
        vm_ind_jit.cranelift_entry_count > entries_before_cb,
        "indirect_target called from closure must execute natively in Cranelift"
    );

    // ---------------------------------------------------------------------
    // Part D: Host-invoked `EmbeddedProgram::call_handle` on a helper that
    //         was initially pruned during `__script` execution MUST work 100%.
    // ---------------------------------------------------------------------
    let handle_src = r#"
function step(a: int, b: int): int
    return a + b * 2
end

local acc: int = 0
local i: int = 0
while i < 20 do
    acc = step(acc, i)
    i = i + 1
end
"#;
    let mut program = build_embedded_program(handle_src);
    let step_handle = program
        .get_typed_handle::<(i64, i64), i64>("main.step")
        .expect("get_typed_handle main.step");

    // Run __script first so `main.step` is pruned as a dead inlined helper.
    program
        .vm_mut()
        .call("__script", vec![])
        .expect("run __script");
    assert!(
        !program.vm().cranelift_compiled.contains_key("main.step"),
        "main.step should be pruned while only __script has executed"
    );

    // Now invoke via EmbeddedProgram::call_handle: call #1 compiles on demand, call #2..N hit direct ABI!
    for i in 0..10i64 {
        let r = program
            .call_handle(&step_handle, (i, 3))
            .expect("call_handle main.step");
        assert_eq!(r, i + 6);
    }
    assert!(
        program.vm().cranelift_compiled.contains_key("main.step"),
        "main.step must be compiled after call_handle"
    );
    assert_eq!(
        program.vm().cranelift_is_pure_scalar_abi("main.step"),
        Some(true)
    );
}

// =========================================================================
// Test 2: Lazy Deopt Preparation — Defers continuation bytecode & deopt
//         recipes on happy path, materializes on demand on first deopt,
//         reuses cache on subsequent deopts, and preserves rollback.
// =========================================================================

#[test]
fn test_ng_w14_7_lazy_deopt_happy_path_and_safepoint_materialization() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
struct Point
    x: int
    y: int
end

function bump_x(v: int, d: int): int
    return v + d
end

local p: Point = Point { x = 0, y = 0 }
local i: int = 1
while i <= 50 do
    p.x = bump_x(p.x, i)
    p.y = p.y + i * 2
    i = i + 1
end
println(p.x + p.y)
"#;

    let mut vm_bc = build_vm(src, false);
    let mut vm_jit = build_vm(src, true);

    vm_bc.call("__script", vec![]).expect("bc __script");
    let expected_stdout = String::from_utf8(vm_bc.console.stdout_bytes()).unwrap();
    assert_eq!(expected_stdout.trim(), "3825");

    // 1. Happy-path execution: zero deopt recipes & zero continuation bytecode materialized!
    vm_jit.call("__script", vec![]).expect("jit __script happy");
    assert_eq!(
        String::from_utf8(vm_jit.console.stdout_bytes()).unwrap(),
        expected_stdout
    );
    assert_eq!(vm_jit.cranelift_entry_count, 1);
    assert_eq!(
        vm_jit.cranelift_materialized_recipe_count(),
        0,
        "happy path must NOT eagerly materialize CompiledDeoptRecipe entries"
    );
    assert_eq!(
        vm_jit.cranelift_lazy_deopt_materialization_count(),
        0,
        "happy path must trigger 0 lazy deopt materializations"
    );
    assert!(
        vm_jit.cranelift_has_deferred_continuation("__script"),
        "__script continuation bytecode must be deferred until first actual deopt"
    );
    assert!(
        vm_jit
            .function_by_name("__script")
            .expect("__script fn")
            .block_entry_pcs
            .is_empty(),
        "__script must remain untouched AST bytecode before any deopt occurs"
    );

    // Read-only recipe inspection still works without mutating VM state.
    let inspected_sites = vm_jit.cranelift_deopt_recipe_sites("__script");
    assert!(
        inspected_sites
            .iter()
            .any(|&(_, kind, _)| kind == DeoptSiteKind::BlockEntry),
        "read-only inspection must report loop-header BlockEntry site"
    );
    assert_eq!(
        vm_jit.cranelift_materialized_recipe_count(),
        0,
        "read-only inspection must not mutate cranelift_recipes"
    );

    // 2. First actual deopt: materializes continuation bytecode & recipes on demand!
    vm_jit.console = VirtualConsole::captured();
    reset_all_state();
    safepoint_request_deopt(true);
    vm_jit
        .call("__script", vec![])
        .expect("first lazy deopt on __script must succeed");
    reset_all_state();

    assert_eq!(
        String::from_utf8(vm_jit.console.stdout_bytes()).unwrap(),
        expected_stdout,
        "post-deopt lazy continuation must produce exact VM parity"
    );
    assert_eq!(
        vm_jit.cranelift_lazy_deopt_materialization_count(),
        1,
        "first actual deopt must trigger exactly 1 lazy materialization"
    );
    assert!(
        vm_jit.cranelift_materialized_recipe_count() >= 1,
        "recipes must now be materialized and cached"
    );
    assert!(
        !vm_jit.cranelift_has_deferred_continuation("__script"),
        "__script continuation must no longer be marked deferred after materialization"
    );
    assert!(
        !vm_jit
            .function_by_name("__script")
            .expect("__script fn")
            .block_entry_pcs
            .is_empty(),
        "__script must have materialized MIR continuation bytecode installed after first deopt"
    );

    // 3. Second deopt reuses cached continuation & recipes (0 additional materializations).
    vm_jit.console = VirtualConsole::captured();
    reset_all_state();
    safepoint_request_deopt(true);
    vm_jit
        .call("__script", vec![])
        .expect("second deopt on __script must reuse cached continuation");
    reset_all_state();

    assert_eq!(
        String::from_utf8(vm_jit.console.stdout_bytes()).unwrap(),
        expected_stdout
    );
    assert_eq!(
        vm_jit.cranelift_lazy_deopt_materialization_count(),
        1,
        "second deopt must reuse cached recipes without re-materializing"
    );
    assert!(take_deopt_payload().is_none());
    assert!(get_current_sticky_error().is_none());
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_ng_w14_7_lazy_deopt_materializes_on_demand() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    // 1. Cold-path mid-loop deopt (poll ordinal 17) inside `__script` with inlined helper + SROA struct!
    let script_src = r#"
struct Acc
    total: int
    count: int
end

function add_step(a: int, b: int): int
    return a + b * 3 + 1
end

local st: Acc = Acc { total = 0, count = 0 }
local i: int = 0
while i < 40 do
    st.total = add_step(st.total, i)
    st.count = st.count + 1
    i = i + 1
end
println(st.total + st.count)
"#;

    let mut vm_bc = build_vm(script_src, false);
    let mut vm_jit = build_vm(script_src, true);

    vm_bc.call("__script", vec![]).expect("bc script");
    let expected_stdout = String::from_utf8(vm_bc.console.stdout_bytes()).unwrap();

    // Warm normal run: 0 deopts -> 0 recipes & continuation deferred!
    vm_jit.call("__script", vec![]).expect("jit warm run");
    assert_eq!(
        String::from_utf8(vm_jit.console.stdout_bytes()).unwrap(),
        expected_stdout
    );
    assert_eq!(vm_jit.cranelift_materialized_recipe_count(), 0);
    assert_eq!(vm_jit.cranelift_lazy_deopt_materialization_count(), 0);
    assert!(vm_jit.cranelift_has_deferred_continuation("__script"));
    let helper_name = vm_jit
        .mir_functions
        .keys()
        .find(|k| k.ends_with("add_step"))
        .cloned()
        .expect("add_step");
    assert!(
        !vm_jit.cranelift_compiled.contains_key(&helper_name),
        "inlined helper `{helper_name}` must be skipped by DCE"
    );

    // Force mid-loop deopt at poll ordinal 17 (midway through loop after inlining & SROA).
    vm_jit.console = VirtualConsole::captured();
    reset_all_state();
    safepoint_request_deopt_on_poll(Some("__script".into()), 17);
    vm_jit
        .call("__script", vec![])
        .expect("poll-17 lazy deopt on __script must resume cleanly");
    reset_all_state();

    assert_eq!(
        String::from_utf8(vm_jit.console.stdout_bytes()).unwrap(),
        expected_stdout,
        "mid-loop poll-17 lazy deopt must match VM oracle identically"
    );
    assert_eq!(
        vm_jit.last_deopt_source,
        Some(DeoptPayloadSource::NativeExecution)
    );
    assert_eq!(vm_jit.cranelift_lazy_deopt_materialization_count(), 1);
    assert!(vm_jit.cranelift_materialized_recipe_count() >= 1);
    assert!(!vm_jit.cranelift_has_deferred_continuation("__script"));

    // 2. Named function cold-path deopt: recipes stay deferred until first deopt!
    let named_src = r#"
function helper(x: int): int
    return x * 2 + 1
end

function sum_named(n: int): int
    local acc: int = 0
    local i: int = 0
    while i < n do
        acc = acc + helper(i)
        i = i + 1
    end
    return acc
end
"#;
    let mut vm_named_bc = build_vm(named_src, false);
    let mut vm_named_jit = build_vm(named_src, true);
    let sum_name = vm_named_jit
        .mir_functions
        .keys()
        .find(|k| k.ends_with("sum_named"))
        .cloned()
        .expect("sum_named");
    let helper_named = vm_named_jit
        .mir_functions
        .keys()
        .find(|k| k.ends_with("helper"))
        .cloned()
        .expect("helper");

    let expected_named = vm_named_bc
        .call(&sum_name, vec![Value::Int(30)])
        .expect("bc sum_named");
    let warm_named = vm_named_jit
        .call(&sum_name, vec![Value::Int(30)])
        .expect("jit warm sum_named");
    assert_eq!(warm_named, expected_named);
    assert_eq!(
        vm_named_jit.cranelift_materialized_recipe_count(),
        0,
        "named function happy path must defer CompiledDeoptRecipe materialization"
    );
    assert!(
        !vm_named_jit.cranelift_compiled.contains_key(&helper_named),
        "inlined helper `{helper_named}` must be skipped when entering `{sum_name}`"
    );

    reset_all_state();
    safepoint_request_deopt_on_poll(Some(sum_name.clone()), 12);
    let deopt_named = vm_named_jit
        .call(&sum_name, vec![Value::Int(30)])
        .expect("poll-12 lazy deopt on sum_named");
    reset_all_state();

    assert_eq!(deopt_named, expected_named);
    assert_eq!(
        vm_named_jit.last_deopt_source,
        Some(DeoptPayloadSource::NativeExecution)
    );
    assert_eq!(vm_named_jit.cranelift_lazy_deopt_materialization_count(), 1);
    assert!(vm_named_jit.cranelift_materialized_recipe_count() >= 1);
    assert!(take_deopt_payload().is_none());
    assert!(get_current_sticky_error().is_none());
}

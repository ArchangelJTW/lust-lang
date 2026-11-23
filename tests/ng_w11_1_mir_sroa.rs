#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

use lust::ast::Span;
use lust::bytecode::Value;
use lust::data::layout::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1};
use lust::mir::cranelift_backend::{
    CraneliftJitCompiler, safepoint_reset, take_current_sticky_error, take_deopt_payload,
};
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::{get_current_sticky_error, safepoint_request_deopt_on_poll};
use lust::mir::ir::{
    FunctionSignature, MirBinaryOp, MirConstant, MirFunction, MirStructLayout, MirType, Operand,
    Place, Rvalue, Statement, Terminator,
};
use lust::mir::opt::ScalarReplacementOfAggregates;
use lust::mir::verifier::MirVerifier;
use lust::vfs::VirtualConsole;
#[cfg(feature = "test-deopt")]
use lust::vm::DeoptPayloadSource;
#[cfg(feature = "test-deopt")]
use lust::vm::cranelift_dispatch::take_last_deopt_sroa_materializations;
use lust::vm::{VM, reset_effect_handler_state};
use lust::{EmbeddedModule, LustConfig, compile_program_with_config, load_program_from_embedded};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn lust_bin() -> &'static str {
    env!("CARGO_BIN_EXE_lust")
}

fn bench_path(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks")
        .join("suite")
        .join(file)
}

fn reset_all_state() {
    safepoint_reset();
    reset_effect_handler_state();
    take_deopt_payload();
    take_current_sticky_error();
    #[cfg(feature = "test-deopt")]
    {
        let _ = take_last_deopt_sroa_materializations();
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
    cfg.set_use_mir(false);
    let mut vm = compile_program_with_config(program, &cfg).expect("compile embedded program");
    vm.console = VirtualConsole::captured();
    vm
}

/// Parses `; sym fnN = <symbol>` annotations from `CraneliftJitCompiler::dump_function_ir`
/// and returns every invoked symbol name referenced by a `call fnN(...)` instruction in the IR body.
fn called_symbols_in_clif_ir(ir: &str) -> Vec<String> {
    let mut sym_map: HashMap<&str, &str> = HashMap::new();
    for line in ir.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("; sym ") {
            if let Some((fn_ref, sym_name)) = rest.split_once(" = ") {
                sym_map.insert(fn_ref.trim(), sym_name.trim());
            }
        }
    }

    let mut called = Vec::new();
    for line in ir.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(';') {
            continue;
        }
        if let Some(pos) = trimmed.find("call ") {
            let after_call = &trimmed[pos + 5..];
            let fn_ref = after_call
                .split(['(', ' ', ','])
                .next()
                .unwrap_or("")
                .trim();
            if let Some(&sym_name) = sym_map.get(fn_ref) {
                called.push(sym_name.to_string());
            }
        }
    }
    called
}

fn struct_name_matches(actual: &str, expected_simple: &str) -> bool {
    actual == expected_simple || actual.ends_with(&format!(".{expected_simple}"))
}

fn make_point_mir_layout() -> lust::hashbrown::HashMap<String, MirStructLayout> {
    let logical = LogicalLayoutV1::new(
        1,
        "Point",
        vec![
            LogicalFieldDesc::new(0, "x", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(1, "y", CanonicalTypeTag::Int64),
        ],
    );
    let mut map = lust::hashbrown::HashMap::new();
    map.insert("Point".to_string(), MirStructLayout::from(&logical));
    map
}

// =========================================================================
// 1. fields.lust & methods.lust: zero `lust_mir_alloc_struct` in Cranelift IR
// =========================================================================

#[test]
fn test_ng_w11_1_fields_and_methods_ir_eliminate_struct_allocations() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    for (bench_file, expected_struct, expected_fields, expected_stdout) in [
        (
            "fields.lust",
            "Point",
            vec!["x".to_string(), "y".to_string()],
            "50000005000000\n",
        ),
        (
            "methods.lust",
            "Counter",
            vec!["value".to_string()],
            "49999995000000\n",
        ),
    ] {
        let path = bench_path(bench_file);
        let src = std::fs::read_to_string(&path).expect("read benchmark source");

        let mut vm = compile_embedded_vm(&src, true);
        vm.compile_cranelift().expect("compile_cranelift");

        let mir_fn = vm
            .cranelift_compiled_mir("__script")
            .unwrap_or_else(|| panic!("expected compiled MIR for __script in {bench_file}"));
        MirVerifier::verify_fn(&mir_fn)
            .unwrap_or_else(|e| panic!("post-SROA MIR failed verify_fn in {bench_file}: {e:?}"));

        // 1a. Verify SroaStructRecipe was recorded both in MirFunction and CompiledFnEntry
        assert_eq!(
            mir_fn.sroa_decomposed_structs.len(),
            1,
            "expected exactly 1 SROA struct recipe in {bench_file}, got {:?}",
            mir_fn.sroa_decomposed_structs
        );
        let recipe = &mir_fn.sroa_decomposed_structs[0];
        assert!(
            struct_name_matches(&recipe.struct_name, expected_struct),
            "expected struct {expected_struct}, got {}",
            recipe.struct_name
        );
        assert_eq!(recipe.field_names, expected_fields);
        assert_eq!(recipe.field_locals.len(), expected_fields.len());
        for &fl in &recipe.field_locals {
            assert!(
                !mir_fn.locals[fl.0].is_stack_promoted,
                "SROA field local {fl} in {bench_file} must have is_stack_promoted == false (SSA scalar!)"
            );
        }

        let cached_recipes = vm
            .cranelift_sroa_recipes("__script")
            .expect("cached sroa_recipes for __script");
        assert_eq!(cached_recipes, mir_fn.sroa_decomposed_structs);

        // 1b. Verify zero struct Aggregate allocations remain in post-SROA MIR
        let has_struct_agg = mir_fn.blocks.iter().any(|b| {
            b.statements.iter().any(|s| {
                matches!(
                    s,
                    Statement::Assign(_, Rvalue::Aggregate(MirType::Struct(_), _), _)
                )
            })
        });
        assert!(
            !has_struct_agg,
            "post-SROA MIR for {bench_file} must not retain any Rvalue::Aggregate(Struct)"
        );

        // 1c. Verify Cranelift IR dump has zero calls to any `lust_mir_alloc_struct*` helper
        let mut compiler = CraneliftJitCompiler::new().expect("create CraneliftJitCompiler");
        let clif_ir = compiler
            .dump_function_ir(&mir_fn)
            .unwrap_or_else(|e| panic!("dump_function_ir failed for {bench_file}: {e:?}"));
        let called_syms = called_symbols_in_clif_ir(&clif_ir);
        let alloc_struct_calls: Vec<&String> = called_syms
            .iter()
            .filter(|sym| sym.contains("lust_mir_alloc_struct"))
            .collect();
        assert!(
            alloc_struct_calls.is_empty(),
            "Cranelift IR for {bench_file} must contain ZERO calls to lust_mir_alloc_struct*, found {alloc_struct_calls:?}\nIR:\n{clif_ir}"
        );
        // Also verify body text (excluding `; sym ...` declaration footer) never mentions lust_mir_alloc_struct
        let ir_body: String = clif_ir
            .lines()
            .filter(|line| !line.trim_start().starts_with("; sym "))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !ir_body.contains("lust_mir_alloc_struct"),
            "Cranelift IR body for {bench_file} must not contain \"lust_mir_alloc_struct\"\n{ir_body}"
        );

        // 1d. CLI end-to-end verification of exact output
        let out = Command::new(lust_bin())
            .args(["--jit=cranelift"])
            .arg(&path)
            .output()
            .expect("run lust --jit=cranelift");
        assert!(
            out.status.success(),
            "lust --jit=cranelift {bench_file} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            expected_stdout,
            "exact stdout parity failed for {bench_file}"
        );
    }
}

// =========================================================================
// 2. Deoptimization cold materialization on iteration 10 (`--features test-deopt`)
// =========================================================================

#[cfg(feature = "test-deopt")]
#[test]
fn test_ng_w11_1_deopt_iter_10_materializes_point_and_counter_and_resumes_exact() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    // A. Full `benchmarks/suite/fields.lust` deopt on loop poll 10
    {
        reset_all_state();
        let src = std::fs::read_to_string(bench_path("fields.lust")).expect("read fields.lust");
        let mut vm = compile_embedded_vm(&src, true);
        vm.compile_cranelift().expect("compile_cranelift fields.lust");

        let recipes = vm
            .cranelift_sroa_recipes("__script")
            .expect("fields.lust sroa_recipes");
        assert_eq!(recipes.len(), 1);
        let point_orig_slot = recipes[0].orig_local.0;

        safepoint_request_deopt_on_poll(Some("__script".to_string()), 10);
        let entries_before = vm.cranelift_entry_count;
        vm.call("__script", vec![])
            .expect("fields.lust poll-10 deopt must resume cleanly in interpreter");
        safepoint_reset();

        assert_eq!(vm.cranelift_entry_count, entries_before + 1);
        assert_eq!(vm.last_deopt_source, Some(DeoptPayloadSource::NativeExecution));
        assert!(take_deopt_payload().is_none());
        assert!(get_current_sticky_error().is_none());

        let mat = take_last_deopt_sroa_materializations();
        assert_eq!(
            mat.len(),
            1,
            "expected Point to be cold-materialized during poll-10 deopt"
        );
        assert_eq!(mat[0].0, point_orig_slot);
        match &mat[0].1 {
            Value::Struct(rc) => {
                assert!(
                    struct_name_matches(&rc.name, "Point"),
                    "expected Point, got {}",
                    rc.name
                );
                let fields = rc.fields.borrow();
                assert_eq!(fields.len(), 2);
                // At poll 10 of `while i <= 10000000` (polls happen at loop header before iteration 1..10):
                // 9 iterations (i = 1..=9) have completed => point.y == sum(1..=9) == 45!
                assert_eq!(fields[1], Value::Int(45));
            }
            other => panic!("expected materialized Value::Struct(Point), got {other:?}"),
        }

        let stdout = String::from_utf8(vm.console.stdout_bytes()).expect("utf8 stdout");
        assert_eq!(stdout, "50000005000000\n");
    }

    // B. Full `benchmarks/suite/methods.lust` deopt on loop poll 10
    {
        reset_all_state();
        let src = std::fs::read_to_string(bench_path("methods.lust")).expect("read methods.lust");
        let mut vm = compile_embedded_vm(&src, true);
        vm.compile_cranelift().expect("compile_cranelift methods.lust");

        let recipes = vm
            .cranelift_sroa_recipes("__script")
            .expect("methods.lust sroa_recipes");
        assert_eq!(recipes.len(), 1);
        let counter_orig_slot = recipes[0].orig_local.0;

        safepoint_request_deopt_on_poll(Some("__script".to_string()), 10);
        let entries_before = vm.cranelift_entry_count;
        vm.call("__script", vec![])
            .expect("methods.lust poll-10 deopt must resume cleanly in interpreter");
        safepoint_reset();

        assert_eq!(vm.cranelift_entry_count, entries_before + 1);
        assert_eq!(vm.last_deopt_source, Some(DeoptPayloadSource::NativeExecution));
        assert!(take_deopt_payload().is_none());
        assert!(get_current_sticky_error().is_none());

        let mat = take_last_deopt_sroa_materializations();
        assert_eq!(
            mat.len(),
            1,
            "expected Counter to be cold-materialized during poll-10 deopt"
        );
        assert_eq!(mat[0].0, counter_orig_slot);
        match &mat[0].1 {
            Value::Struct(rc) => {
                assert!(
                    struct_name_matches(&rc.name, "Counter"),
                    "expected Counter, got {}",
                    rc.name
                );
                let fields = rc.fields.borrow();
                assert_eq!(fields.len(), 1);
                // At poll 10 of `while i < 10000000` (i = 0..=8 completed => sum(0..=8) == 36)
                assert_eq!(fields[0], Value::Int(36));
            }
            other => panic!("expected materialized Value::Struct(Counter), got {other:?}"),
        }

        let stdout = String::from_utf8(vm.console.stdout_bytes()).expect("utf8 stdout");
        assert_eq!(stdout, "49999995000000\n");
    }

    // C. Both fields live at loop header (`p.x` and `p.y` read after loop) -> both exact in materialized Point!
    {
        reset_all_state();
        let src = r#"
struct Point
    x: int
    y: int
end

function sum_both_fields(n: int): int
    local p: Point = Point { x = 0, y = 0 }
    local i: int = 1
    while i <= n do
        p.x = i
        p.y = p.y + i
        i = i + 1
    end
    return p.y * 1000 + p.x
end
"#;
        let mut vm_oracle = compile_embedded_vm(src, false);
        let mut vm_clif = compile_embedded_vm(src, true);
        let fn_name = vm_clif
            .mir_functions
            .keys()
            .find(|k| k.ends_with("sum_both_fields"))
            .cloned()
            .expect("sum_both_fields in mir_functions");

        let expected = vm_oracle.call(&fn_name, vec![Value::Int(50)]).expect("oracle");
        vm_clif.compile_cranelift().expect("compile_cranelift");
        let recipes = vm_clif
            .cranelift_sroa_recipes(&fn_name)
            .expect("sum_both_fields sroa_recipes");
        assert_eq!(recipes.len(), 1);

        safepoint_request_deopt_on_poll(Some(fn_name.clone()), 10);
        let actual = vm_clif
            .call(&fn_name, vec![Value::Int(50)])
            .expect("poll-10 deopt on sum_both_fields");
        safepoint_reset();

        assert_eq!(actual, expected);
        let mat = take_last_deopt_sroa_materializations();
        assert_eq!(mat.len(), 1);
        assert_eq!(mat[0].0, recipes[0].orig_local.0);
        match &mat[0].1 {
            Value::Struct(rc) => {
                assert!(
                    struct_name_matches(&rc.name, "Point"),
                    "expected Point, got {}",
                    rc.name
                );
                let fields = rc.fields.borrow();
                assert_eq!(*fields, vec![Value::Int(9), Value::Int(45)]);
            }
            other => panic!("expected Point struct, got {other:?}"),
        }
    }
}

// =========================================================================
// 3. Hard Gates, Preconditions, Inlined Constructor, and Negative Safety Tests
// =========================================================================

#[test]
fn test_ng_w11_1_sroa_hard_250_local_cap_and_ssa_scalar_flag() {
    let span = Span::dummy();
    let layouts = make_point_mir_layout(); // 2 fields

    let build_fn_with_total_locals = |target_locals: usize| -> MirFunction {
        let mut f = MirFunction::new(
            FunctionSignature {
                name: format!("cap_test_{target_locals}"),
                param_types: vec![],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            span,
        );
        let p = f.add_local(Some("p".into()), MirType::Struct("Point".into()), true, span);
        while f.locals.len() < target_locals {
            f.add_local(None, MirType::Int, true, span);
        }
        assert_eq!(f.locals.len(), target_locals);

        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(p),
            Rvalue::Aggregate(
                MirType::Struct("Point".into()),
                vec![
                    Operand::Constant(MirConstant::Int(10)),
                    Operand::Constant(MirConstant::Int(32)),
                ],
            ),
            span,
        ));
        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(f.return_local),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Field(Box::new(Place::Local(p)), 0)),
                Operand::Copy(Place::Field(Box::new(Place::Local(p)), 1)),
            ),
            span,
        ));
        f.blocks[0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(f.return_local))),
            span,
        ));
        f
    };

    // 249 + 2 = 251 > 250 -> MUST reject SROA!
    let mut over_cap = build_fn_with_total_locals(249);
    assert!(
        !ScalarReplacementOfAggregates::run_on_function(&mut over_cap, &layouts),
        "SROA MUST NOT apply when func.locals.len() + layout.fields.len() > 250"
    );
    assert!(over_cap.sroa_decomposed_structs.is_empty());
    assert_eq!(over_cap.locals.len(), 249);

    // 248 + 2 = 250 <= 250 -> MUST apply SROA, and field locals MUST keep is_stack_promoted == false!
    let mut at_cap = build_fn_with_total_locals(248);
    assert!(
        ScalarReplacementOfAggregates::run_on_function(&mut at_cap, &layouts),
        "SROA should apply at exact 250 boundary (248 + 2 == 250)"
    );
    assert_eq!(at_cap.sroa_decomposed_structs.len(), 1);
    assert_eq!(at_cap.locals.len(), 250);
    for &fl in &at_cap.sroa_decomposed_structs[0].field_locals {
        assert!(
            !at_cap.locals[fl.0].is_stack_promoted,
            "field_local {fl} MUST have is_stack_promoted == false"
        );
    }
    MirVerifier::verify_fn(&at_cap).expect("post-SROA function at 250 cap must verify");
}

#[test]
fn test_ng_w11_1_sroa_rejects_escaping_aliased_and_non_scalar_structs() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    // 1. User-aliased struct (`local b: Point = a`), returned struct, and non-scalar struct
    let src = r#"
struct Point
    x: int
    y: int
end

function aliased_struct(n: int): int
    local a: Point = Point { x = 1, y = 2 }
    local b: Point = a
    local i: int = 0
    while i < n do
        a.x = a.x + 1
        b.y = b.y + a.x
        i = i + 1
    end
    return b.x + a.y
end

function returns_struct(v: int): Point
    local p: Point = Point { x = v, y = v + 1 }
    return p
end

function inlined_ctor_and_multi_scalar(n: int): int
    local c: Counter = make_counter(0)
    local i: int = 0
    while i < n do
        c.value = c.value + i
        i = i + 1
    end
    return c.value
end

struct Counter
    value: int
end

function make_counter(init: int): Counter
    return Counter { value = init }
end
"#;
    let mut vm_oracle = compile_embedded_vm(src, false);
    let mut vm_clif = compile_embedded_vm(src, true);
    vm_clif.compile_cranelift().expect("compile_cranelift");

    let aliased_fn = vm_clif
        .mir_functions
        .keys()
        .find(|k| k.ends_with("aliased_struct"))
        .cloned()
        .expect("aliased_struct");
    let returns_fn = vm_clif
        .mir_functions
        .keys()
        .find(|k| k.ends_with("returns_struct"))
        .cloned()
        .expect("returns_struct");
    let ctor_fn = vm_clif
        .mir_functions
        .keys()
        .find(|k| k.ends_with("inlined_ctor_and_multi_scalar"))
        .cloned()
        .expect("inlined_ctor_and_multi_scalar");

    // Aliased user struct MUST be rejected from SROA
    assert!(
        vm_clif
            .cranelift_sroa_recipes(&aliased_fn)
            .expect("aliased_struct compiled")
            .is_empty(),
        "aliased user struct `local b = a` MUST be rejected from SROA"
    );
    // Returned struct MUST be rejected from SROA
    assert!(
        vm_clif
            .cranelift_sroa_recipes(&returns_fn)
            .expect("returns_struct compiled")
            .is_empty(),
        "escaping returned struct MUST be rejected from SROA"
    );
    // Inlined constructor `local c = make_counter(0)` MUST be decomposed by SROA!
    let ctor_recipes = vm_clif
        .cranelift_sroa_recipes(&ctor_fn)
        .expect("inlined_ctor_and_multi_scalar compiled");
    let ctor_mir = vm_clif
        .cranelift_compiled_mir(&ctor_fn)
        .expect("compiled MIR for inlined_ctor_and_multi_scalar");
    assert_eq!(
        ctor_recipes.len(),
        1,
        "inlined constructor temporary chain MUST fold and trigger SROA"
    );
    let mut compiler = CraneliftJitCompiler::new().expect("compiler");
    let ctor_ir = compiler.dump_function_ir(&ctor_mir).expect("dump_function_ir");
    assert!(
        !called_symbols_in_clif_ir(&ctor_ir)
            .iter()
            .any(|s| s.contains("lust_mir_alloc_struct")),
        "inlined constructor SROA must eliminate lust_mir_alloc_struct* calls:\n{ctor_ir}"
    );

    // Exact oracle parity on all three functions
    assert_eq!(
        vm_clif.call(&aliased_fn, vec![Value::Int(10)]).unwrap(),
        vm_oracle.call(&aliased_fn, vec![Value::Int(10)]).unwrap()
    );
    assert_eq!(
        vm_clif.call(&ctor_fn, vec![Value::Int(100)]).unwrap(),
        vm_oracle.call(&ctor_fn, vec![Value::Int(100)]).unwrap()
    );

    // 2. Non-scalar (nullable/optional or nested struct) layout MUST be rejected directly by SROA
    let span = Span::dummy();
    let mut f = MirFunction::new(
        FunctionSignature {
            name: "non_scalar_field".into(),
            param_types: vec![],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let p = f.add_local(Some("n".into()), MirType::Struct("Node".into()), true, span);
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(p),
        Rvalue::Aggregate(
            MirType::Struct("Node".into()),
            vec![
                Operand::Constant(MirConstant::Int(1)),
                Operand::Constant(MirConstant::Int(0)),
            ],
        ),
        span,
    ));
    f.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Field(Box::new(Place::Local(p)), 0))),
        span,
    ));

    let non_scalar_logical = LogicalLayoutV1::new(
        2,
        "Node",
        vec![
            LogicalFieldDesc::new(0, "val", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(1, "next", CanonicalTypeTag::NestedStruct(2))
                .with_nullability(true, true),
        ],
    );
    let mut bad_layouts = lust::hashbrown::HashMap::new();
    bad_layouts.insert("Node".to_string(), MirStructLayout::from(&non_scalar_logical));
    assert!(
        !ScalarReplacementOfAggregates::run_on_function(&mut f, &bad_layouts),
        "struct with NestedStruct / optional field MUST be rejected by SROA"
    );
    assert!(f.sroa_decomposed_structs.is_empty());
}

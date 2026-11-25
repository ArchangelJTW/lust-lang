#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! Integration & regression tests for Wave 16 Task `NG-W16-6`:
//! 1. Negative-cache population in `resolve_cranelift_fn_entry_slow` when `compilations == 0`
//!    (plus O(1) cached `is_trivial_leaf_method_for_vm_subframe` without per-call rescans and
//!    soundness when mixing ineligible and eligible functions in the same VM).
//! 2. Cranelift VM eligibility & builtin allowlist for all 5 `entries=0` Next-Gen Suite v2
//!    workloads (`BENCH-NG-04`, `BENCH-NG-05`, `BENCH-NG-06`, `BENCH-NG-09`, `BENCH-NG-12`),
//!    covering `MirType::Function`, `Rvalue::Closure`, `MirType::Unknown`, synthetic `__Enum_*`,
//!    `"LuaValue"`, `__mir_unknown_*`, and `queue.*`.
//! 3. Deopt recipe acceptance and fail-closed handle materialization for `MirType::Function`
//!    (closures/functions) and `MirType::Unknown` (and `LuaValue`) under `--features test-deopt`.

use std::fs;
use std::path::PathBuf;
#[cfg(feature = "test-deopt")]
use std::rc::Rc;
use std::sync::Mutex;

use lust::ast::{ItemKind, Span};
use lust::bytecode::compiler::Compiler;
#[cfg(feature = "test-deopt")]
use lust::bytecode::{NativeCallResult, native_fn};
use lust::bytecode::{Function, Value};
use lust::codegen_zig::aot::prepare_items_for_mir;
use lust::config::LustConfig;
use lust::hashbrown::HashMap;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{
    safepoint_reset, take_current_sticky_error, take_deopt_payload,
};
use lust::mir::ir::{
    FunctionSignature, MirBinaryOp, MirConstant, MirFunction, MirType, Operand, Place, Rvalue,
    Statement, Terminator,
};
use lust::mir::lowering::AstLowering;
use lust::modules::ModuleLoader;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{VM, VmBackend, reset_effect_handler_state};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn reset_all_state() {
    safepoint_reset();
    reset_effect_handler_state();
    take_deopt_payload();
    take_current_sticky_error();
}

fn suite_v2_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("benchmarks/suite_v2")
        .join(filename)
}

fn compile_source_via_compiler(
    source: &str,
    collect_mir: bool,
) -> (
    Vec<Function>,
    HashMap<String, lust::ast::StructDef>,
    Vec<MirFunction>,
) {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");

    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    config.set_cranelift_jit(collect_mir);
    config.set_use_mir(false);

    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let mut struct_defs = checker.struct_definitions();
    for item in &items {
        if let ItemKind::Struct(def) = &item.kind {
            struct_defs
                .entry(def.name.clone())
                .or_insert_with(|| def.clone());
        }
    }

    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_collect_mir(collect_mir);
    compiler.set_option_coercions(checker.take_option_coercions());
    compiler.set_checked_array_indices(checker.take_checked_array_indices());
    compiler.set_lambda_return_types(checker.take_lambda_return_types());
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    compiler.set_struct_defs(struct_defs.clone());

    let functions = compiler.compile_module(&items).expect("compile_module");
    let mir_functions = compiler.take_mir_functions();
    (functions, struct_defs, mir_functions)
}

fn lower_suite_v2_program_mir(
    filename: &str,
) -> (
    lust::mir::ir::MirModule,
    HashMap<String, lust::ast::StructDef>,
    Vec<MirFunction>,
) {
    let path = suite_v2_path(filename);
    let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let mut loader = ModuleLoader::new(parent);
    let program = loader
        .load_program_from_entry(path.to_str().expect("utf8 path"))
        .unwrap_or_else(|e| panic!("ModuleLoader failed for {filename}: {e}"));

    let mut all_items = Vec::new();
    for m in &program.modules {
        all_items.extend(m.items.clone());
    }
    let mut struct_defs: HashMap<String, lust::ast::StructDef> = HashMap::new();
    for item in &all_items {
        if let ItemKind::Struct(def) = &item.kind {
            struct_defs
                .entry(def.name.clone())
                .or_insert_with(|| def.clone());
        }
    }

    // 1. Direct AST -> MIR module lowering (full program including enums/closures/imports).
    let prepared = prepare_items_for_mir(&all_items);
    let module = AstLowering::lower_module_items(filename, &prepared)
        .unwrap_or_else(|e| panic!("AstLowering::lower_module_items failed for {filename}: {e}"));

    // 2. Also run `Compiler::compile_module` on the raw entry source when self-contained.
    let raw_src = fs::read_to_string(&path).expect("read benchmark source");
    let compiler_mirs = if !raw_src.contains("use @std/") {
        let (_, _, mirs) = compile_source_via_compiler(&raw_src, true);
        mirs
    } else {
        // Multi-module entry (`use @std/sync.{Queue}`): mirror CLI `compile_program` module wiring.
        let mut config = LustConfig::default();
        config.set_jit_enabled(false);
        config.set_cranelift_jit(true);
        config.set_use_mir(false);

        let mut imports_map: HashMap<String, lust::modules::ModuleImports> = HashMap::new();
        let mut wrapped_items: Vec<lust::ast::Item> = Vec::new();
        for m in &program.modules {
            imports_map.insert(m.path.clone(), m.imports.clone());
            wrapped_items.push(lust::ast::Item::new(
                ItemKind::Module {
                    name: m.path.clone(),
                    items: m.items.clone(),
                },
                Span::dummy(),
            ));
        }

        let mut checker = TypeChecker::with_config(&config);
        checker.set_imports_by_module(imports_map.clone());
        let _ = checker.check_program(&program.modules);
        let mut compiler = Compiler::new();
        compiler.configure(&config);
        compiler.configure_stdlib(&config);
        compiler.set_imports_by_module(imports_map);
        compiler.set_entry_module(program.entry_module.clone());
        compiler.set_collect_mir(true);
        compiler.set_option_coercions(checker.take_option_coercions());
        compiler.set_checked_array_indices(checker.take_checked_array_indices());
        compiler.set_lambda_return_types(checker.take_lambda_return_types());
        compiler.set_numeric_types(checker.take_numeric_types());
        compiler.set_function_signatures(checker.take_function_signatures());
        compiler.set_struct_defs(struct_defs.clone());
        compiler
            .compile_module(&wrapped_items)
            .unwrap_or_else(|e| panic!("Compiler::compile_module failed for {filename}: {e}"));
        compiler.take_mir_functions()
    };

    (module, struct_defs, compiler_mirs)
}

#[test]
fn test_ng_w16_6_negative_cache_populated_when_compilations_zero() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    // Part A: Purely ineligible module (compilations == 0) called 100,000 times.
    // `ineligible_callee` has `Terminator::Raise` in MIR so it is unconditionally ineligible.
    let src = r#"
function ineligible_callee(x: int): int
    return x + 1
end

function hot_driver(n: int): int
    local i: int = 0
    local acc: int = 0
    while i < n do
        acc = acc + ineligible_callee(i)
        i = i + 1
    end
    return acc
end
"#;
    let (funcs, struct_defs, _) = compile_source_via_compiler(src, false);

    let mut cfg = LustConfig::default();
    cfg.set_jit_enabled(false);
    cfg.set_cranelift_jit(true);
    let mut vm = VM::with_config(&cfg);
    vm.set_backend(VmBackend::Cranelift);
    vm.register_structs(&struct_defs);
    vm.load_functions(funcs);

    // Register an explicitly ineligible MIR candidate (Terminator::Raise) for `ineligible_callee`
    // and leave `hot_driver` without any MIR candidate so `compilations == 0`.
    let span = Span::dummy();
    let mut bad_mir = MirFunction::new(
        FunctionSignature {
            name: "ineligible_callee".to_string(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let p0 = bad_mir.params[0];
    bad_mir.blocks[0].terminator = Some(Terminator::Raise(Operand::Copy(Place::Local(p0)), span));
    assert!(
        !vm.is_cranelift_eligible_in_vm(&bad_mir),
        "Terminator::Raise MIR function must be ineligible"
    );
    vm.register_mir_candidate(bad_mir);

    let slow_before = vm.cranelift_slow_resolve_count();
    let refine_before = vm.cranelift_dispatch_refine_count();

    // Direct 100,000-call loop via `vm.call("ineligible_callee", ...)` — exercises
    // `try_run_cranelift_top_frame` on every single call when `compilations == 0`!
    let mut sum = 0i64;
    for i in 0..100_000i64 {
        let res = vm
            .call("ineligible_callee", vec![Value::Int(i)])
            .expect("ineligible_callee call");
        let Value::Int(v) = res else {
            panic!("expected Int, got {res:?}");
        };
        sum += v;
    }
    let expected_sum: i64 = (1..=100_000i64).sum();
    assert_eq!(sum, expected_sum);

    let slow_calls = vm.cranelift_slow_resolve_count() - slow_before;
    let refine_calls = vm.cranelift_dispatch_refine_count() - refine_before;

    assert_eq!(
        vm.cranelift_compilation_count, 0,
        "compilations must stay 0 for ineligible function"
    );
    assert!(
        vm.cranelift_fn_cache_len() >= 2,
        "cranelift_fn_cache must be populated with negative None entries (len={})",
        vm.cranelift_fn_cache_len()
    );
    assert!(
        slow_calls <= 1,
        "resolve_cranelift_fn_entry_slow must be called at most once across 100,000 calls, got {slow_calls}"
    );
    assert!(
        refine_calls <= 1,
        "cranelift_dispatch_refine_count must be <= 1 across 100,000 calls, got {refine_calls}"
    );

    // Part B: Calling the driver with 100,000 in-VM calls also populates negative cache at most once.
    let slow_before_driver = vm.cranelift_slow_resolve_count();
    let drv_res = vm
        .call("hot_driver", vec![Value::Int(100_000)])
        .expect("hot_driver");
    assert_eq!(drv_res, Value::Int(expected_sum));
    let slow_driver_delta = vm.cranelift_slow_resolve_count() - slow_before_driver;
    assert!(
        slow_driver_delta <= 1,
        "hot_driver itself must hit slow resolve at most once (ineligible_callee already cached), got {slow_driver_delta}"
    );

    // Part C: Soundness — mixing an ineligible function (called first) with an eligible function
    // (called second) MUST NOT prevent the eligible function from compiling & executing natively!
    let mut good_mir = MirFunction::new(
        FunctionSignature {
            name: "eligible_fn".to_string(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let gp0 = good_mir.params[0];
    let gdst = good_mir.add_local(Some("res".into()), MirType::Int, false, span);
    good_mir.blocks[0].statements.push(Statement::Assign(
        Place::Local(gdst),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(gp0)),
            Operand::Constant(MirConstant::Int(10)),
        ),
        span,
    ));
    good_mir.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(gdst))),
        span,
    ));

    let mixed_src = r#"
function bad_fn(x: int): int
    return x + 1
end

function eligible_fn(x: int): int
    return x + 10
end
"#;
    let (mixed_funcs, _, _) = compile_source_via_compiler(mixed_src, false);
    let mut vm_mixed = VM::with_config(&cfg);
    vm_mixed.set_backend(VmBackend::Cranelift);
    vm_mixed.load_functions(mixed_funcs);
    let mut bad_mir_2 = MirFunction::new(
        FunctionSignature {
            name: "bad_fn".to_string(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let bp0 = bad_mir_2.params[0];
    bad_mir_2.blocks[0].terminator = Some(Terminator::Raise(Operand::Copy(Place::Local(bp0)), span));
    vm_mixed.register_mir_candidate(bad_mir_2);
    vm_mixed.register_mir_candidate(good_mir);

    let bad_slow_0 = vm_mixed.cranelift_slow_resolve_count();
    for i in 0..1000i64 {
        assert_eq!(
            vm_mixed.call("bad_fn", vec![Value::Int(i)]).expect("bad_fn"),
            Value::Int(i + 1)
        );
    }
    assert!(
        vm_mixed.cranelift_slow_resolve_count() - bad_slow_0 <= 1,
        "bad_fn must negative-cache after first probe"
    );

    let entries_before = vm_mixed.cranelift_entry_count;
    let good_res = vm_mixed
        .call("eligible_fn", vec![Value::Int(32)])
        .expect("eligible_fn");
    assert_eq!(good_res, Value::Int(42));
    assert!(
        vm_mixed.cranelift_entry_count > entries_before,
        "eligible_fn must still compile and enter Cranelift natively after bad_fn was negative-cached"
    );

    // Part D: `is_trivial_leaf_method_for_vm_subframe` must be cached in O(1) (<= 1 scan across 10,000 calls).
    let leaf_src = r#"
struct Counter
    x: int
end

impl Counter
    function bump(self, d: int): int
        return self.x + d
    end
end

function run_leaf(n: int): int
    local c: Counter = Counter { x = 7 }
    local i: int = 0
    local acc: int = 0
    while i < n do
        acc = acc + c:bump(i)
        i = i + 1
    end
    return acc
end
"#;
    let (leaf_funcs, leaf_structs, _) = compile_source_via_compiler(leaf_src, false);
    let mut vm_leaf = VM::with_config(&cfg);
    vm_leaf.set_backend(VmBackend::Cranelift);
    vm_leaf.register_structs(&leaf_structs);
    vm_leaf.load_functions(leaf_funcs);

    let scans_before = vm_leaf.cranelift_trivial_leaf_scan_count();
    let leaf_res = vm_leaf
        .call("run_leaf", vec![Value::Int(10_000)])
        .expect("run_leaf");
    let expected_leaf: i64 = (0..10_000i64).map(|i| 7 + i).sum();
    assert_eq!(leaf_res, Value::Int(expected_leaf));
    let scans_delta = vm_leaf.cranelift_trivial_leaf_scan_count() - scans_before;
    assert!(
        scans_delta <= 1,
        "is_trivial_leaf_method_for_vm_subframe must scan instructions at most once per function, got {scans_delta}"
    );
}

#[test]
fn test_ng_w16_6_all_5_entries_zero_suite_v2_workloads_pass_eligibility() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    // 1. Builtin allowlist checks for all 7 newly-unblocked builtins.
    for callee in [
        "__mir_unknown_is_int",
        "__mir_unknown_unwrap_int",
        "__mir_unknown_get_field",
        "__mir_unknown_call_method",
        "queue.new",
        "queue.offer_int",
        "queue.take_int",
    ] {
        assert!(
            VM::is_cranelift_builtin_callee(callee),
            "builtin '{callee}' must be accepted by is_cranelift_builtin_callee"
        );
    }

    // 2. Struct auto-acceptance for `"__Enum_*"` and `"LuaValue"`, while keeping fail-closed
    // rejection for unregistered arbitrary structs and empty names.
    {
        let vm = VM::new();
        assert!(
            vm.is_cranelift_supported_struct("__Enum_Op"),
            "synthetic scalar enum __Enum_Op must be auto-accepted"
        );
        assert!(
            vm.is_cranelift_supported_struct("LuaValue"),
            "LuaValue must be auto-accepted"
        );
        assert!(
            !vm.is_cranelift_supported_struct(""),
            "empty struct name must remain rejected"
        );
        assert!(
            !vm.is_cranelift_supported_struct("__Enum_"),
            "bare __Enum_ prefix without name must remain rejected"
        );
        assert!(
            !vm.is_cranelift_supported_struct("UnregisteredRandomStruct"),
            "unregistered arbitrary struct must remain rejected"
        );
    }

    // 3. Direct / synthetic MIR pattern checks for all 5 patterns (`MirType::Function`,
    // `Rvalue::Closure`, `MirType::Unknown`, `__Enum_*`, `LuaValue`, `__mir_unknown_*`, `queue.*`).
    {
        let vm = VM::new();
        let span = Span::dummy();

        // Pattern A (BENCH-NG-04): MirType::Function + Rvalue::Closure
        let mut closure_fn = MirFunction::new(
            FunctionSignature {
                name: "synth_closure_pattern".into(),
                param_types: vec![MirType::Int],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            span,
        );
        let p0 = closure_fn.params[0];
        let fn_ty = MirType::Function {
            params: vec![MirType::Int],
            return_type: Box::new(MirType::Int),
        };
        let cl = closure_fn.add_local(Some("cl".into()), fn_ty, false, span);
        let out = closure_fn.add_local(Some("out".into()), MirType::Int, false, span);
        let bb1 = closure_fn.new_block();
        closure_fn.blocks[0].statements.push(Statement::Assign(
            Place::Local(cl),
            Rvalue::Closure {
                func_name: "__lambda_0".into(),
                captures: vec![("bias".into(), Operand::Copy(Place::Local(p0)), MirType::Int)],
            },
            span,
        ));
        closure_fn.blocks[0].terminator = Some(Terminator::Call {
            callee: "apply_twice".into(),
            args: vec![Operand::Copy(Place::Local(cl)), Operand::Copy(Place::Local(p0))],
            destination: Place::Local(out),
            success: bb1,
            error: None,
            span,
        });
        closure_fn.blocks[bb1.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(out))),
            span,
        ));
        assert!(
            vm.is_cranelift_eligible_in_vm(&closure_fn),
            "MirType::Function + Rvalue::Closure pattern must pass is_cranelift_eligible_in_vm"
        );

        // Pattern B (BENCH-NG-05): MirType::Unknown + __mir_unknown_get_field + __mir_unknown_call_method
        let mut poly_fn = MirFunction::new(
            FunctionSignature {
                name: "synth_poly_unknown".into(),
                param_types: vec![MirType::Unknown, MirType::Int],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            span,
        );
        let obj = poly_fn.params[0];
        let k = poly_fn.params[1];
        let fx = poly_fn.add_local(Some("fx".into()), MirType::Unknown, false, span);
        let sc = poly_fn.add_local(Some("sc".into()), MirType::Int, false, span);
        let bb1 = poly_fn.new_block();
        let bb2 = poly_fn.new_block();
        poly_fn.blocks[0].terminator = Some(Terminator::Call {
            callee: "__mir_unknown_get_field".into(),
            args: vec![
                Operand::Copy(Place::Local(obj)),
                Operand::Constant(MirConstant::String("x".into())),
            ],
            destination: Place::Local(fx),
            success: bb1,
            error: None,
            span,
        });
        poly_fn.blocks[bb1.0].terminator = Some(Terminator::Call {
            callee: "__mir_unknown_call_method".into(),
            args: vec![
                Operand::Copy(Place::Local(obj)),
                Operand::Constant(MirConstant::String("score".into())),
                Operand::Copy(Place::Local(k)),
            ],
            destination: Place::Local(sc),
            success: bb2,
            error: None,
            span,
        });
        poly_fn.blocks[bb2.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(sc))),
            span,
        ));
        assert!(
            vm.is_cranelift_eligible_in_vm(&poly_fn),
            "MirType::Unknown + __mir_unknown_get_field/call_method must pass is_cranelift_eligible_in_vm"
        );

        // Pattern C (BENCH-NG-06): synthetic `__Enum_Op` struct without pre-registration
        let mut enum_fn = MirFunction::new(
            FunctionSignature {
                name: "synth_enum_op".into(),
                param_types: vec![MirType::Int],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            span,
        );
        let step = enum_fn.params[0];
        let op = enum_fn.add_local(
            Some("op".into()),
            MirType::Struct("__Enum_Op".into()),
            false,
            span,
        );
        let tag = enum_fn.add_local(Some("tag".into()), MirType::Int, false, span);
        let p_val = enum_fn.add_local(Some("p0".into()), MirType::Int, false, span);
        let out = enum_fn.add_local(Some("out".into()), MirType::Int, false, span);
        enum_fn.blocks[0].statements.push(Statement::Assign(
            Place::Local(op),
            Rvalue::Aggregate(
                MirType::Struct("__Enum_Op".into()),
                vec![
                    Operand::Constant(MirConstant::Int(1)),
                    Operand::Copy(Place::Local(step)),
                    Operand::Constant(MirConstant::Int(0)),
                ],
            ),
            span,
        ));
        enum_fn.blocks[0].statements.push(Statement::Assign(
            Place::Local(tag),
            Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(op)), 0))),
            span,
        ));
        enum_fn.blocks[0].statements.push(Statement::Assign(
            Place::Local(p_val),
            Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(op)), 1))),
            span,
        ));
        enum_fn.blocks[0].statements.push(Statement::Assign(
            Place::Local(out),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(tag)),
                Operand::Copy(Place::Local(p_val)),
            ),
            span,
        ));
        enum_fn.blocks[0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(out))),
            span,
        ));
        assert!(
            vm.is_cranelift_eligible_in_vm(&enum_fn),
            "Synthetic __Enum_Op struct with 3 slots must pass is_cranelift_eligible_in_vm"
        );

        // Pattern D (BENCH-NG-09): Array<Struct("LuaValue")> + __mir_unknown_is_int + __mir_unknown_unwrap_int
        let mut lv_fn = MirFunction::new(
            FunctionSignature {
                name: "synth_luavalue_byte".into(),
                param_types: vec![MirType::String],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            span,
        );
        let s = lv_fn.params[0];
        let arr = lv_fn.add_local(
            Some("b0_arr".into()),
            MirType::Array(Box::new(MirType::Struct("LuaValue".into()))),
            false,
            span,
        );
        let idx0 = lv_fn.add_local(Some("idx0".into()), MirType::Int, false, span);
        let lv = lv_fn.add_local(
            Some("lv".into()),
            MirType::Struct("LuaValue".into()),
            false,
            span,
        );
        let is_i = lv_fn.add_local(Some("is_i".into()), MirType::Bool, false, span);
        let v0 = lv_fn.add_local(Some("v0".into()), MirType::Int, false, span);
        let bb1 = lv_fn.new_block();
        let bb2 = lv_fn.new_block();
        let bb3 = lv_fn.new_block();
        lv_fn.blocks[0].statements.push(Statement::Assign(
            Place::Local(idx0),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            span,
        ));
        lv_fn.blocks[0].terminator = Some(Terminator::Call {
            callee: "string.byte".into(),
            args: vec![
                Operand::Copy(Place::Local(s)),
                Operand::Constant(MirConstant::Int(1)),
                Operand::Constant(MirConstant::Int(1)),
            ],
            destination: Place::Local(arr),
            success: bb1,
            error: None,
            span,
        });
        lv_fn.blocks[bb1.0].statements.push(Statement::Assign(
            Place::Local(lv),
            Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(arr)), idx0))),
            span,
        ));
        lv_fn.blocks[bb1.0].terminator = Some(Terminator::Call {
            callee: "__mir_unknown_is_int".into(),
            args: vec![Operand::Copy(Place::Local(lv))],
            destination: Place::Local(is_i),
            success: bb2,
            error: None,
            span,
        });
        lv_fn.blocks[bb2.0].terminator = Some(Terminator::Call {
            callee: "__mir_unknown_unwrap_int".into(),
            args: vec![Operand::Copy(Place::Local(lv))],
            destination: Place::Local(v0),
            success: bb3,
            error: None,
            span,
        });
        lv_fn.blocks[bb3.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(v0))),
            span,
        ));
        assert!(
            vm.is_cranelift_eligible_in_vm(&lv_fn),
            "Array<LuaValue> + __mir_unknown_is_int/unwrap_int must pass is_cranelift_eligible_in_vm"
        );

        // Pattern E (BENCH-NG-12): queue.new + queue.offer_int + queue.take_int
        let mut q_fn = MirFunction::new(
            FunctionSignature {
                name: "synth_queue_pingpong".into(),
                param_types: vec![MirType::Int],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            span,
        );
        let x = q_fn.params[0];
        let q = q_fn.add_local(Some("q".into()), MirType::Int, false, span);
        let offered = q_fn.add_local(Some("offered".into()), MirType::Int, false, span);
        let taken = q_fn.add_local(Some("taken".into()), MirType::Int, false, span);
        let bb1 = q_fn.new_block();
        let bb2 = q_fn.new_block();
        let bb3 = q_fn.new_block();
        q_fn.blocks[0].terminator = Some(Terminator::Call {
            callee: "queue.new".into(),
            args: vec![Operand::Constant(MirConstant::Int(64))],
            destination: Place::Local(q),
            success: bb1,
            error: None,
            span,
        });
        q_fn.blocks[bb1.0].terminator = Some(Terminator::Call {
            callee: "queue.offer_int".into(),
            args: vec![Operand::Copy(Place::Local(q)), Operand::Copy(Place::Local(x))],
            destination: Place::Local(offered),
            success: bb2,
            error: None,
            span,
        });
        q_fn.blocks[bb2.0].terminator = Some(Terminator::Call {
            callee: "queue.take_int".into(),
            args: vec![Operand::Copy(Place::Local(q))],
            destination: Place::Local(taken),
            success: bb3,
            error: None,
            span,
        });
        q_fn.blocks[bb3.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(taken))),
            span,
        ));
        assert!(
            vm.is_cranelift_eligible_in_vm(&q_fn),
            "queue.new/offer_int/take_int pattern must pass is_cranelift_eligible_in_vm"
        );
    }

    // 4. All 5 real `entries=0` Next-Gen Suite v2 benchmarks (`BENCH-NG-04`, `BENCH-NG-05`,
    // `BENCH-NG-06`, `BENCH-NG-09`, `BENCH-NG-12`): verify `is_cranelift_eligible_in_vm` returns
    // `true` for `__script` from BOTH direct MIR lowering and `Compiler::compile_module`!
    let benchmarks = [
        ("BENCH-NG-04", "closure_upvalue_combinators.lust"),
        ("BENCH-NG-05", "megamorphic_method_ic.lust"),
        ("BENCH-NG-06", "adt_enum_eval.lust"),
        ("BENCH-NG-09", "string_parse_tokenize.lust"),
        ("BENCH-NG-12", "fiber_channel_pingpong.lust"),
    ];

    for (bench_id, filename) in benchmarks {
        let (module, struct_defs, compiler_mirs) = lower_suite_v2_program_mir(filename);

        let mut vm = VM::new();
        vm.set_backend(VmBackend::Cranelift);
        vm.register_structs(&struct_defs);
        for f in &module.functions {
            vm.mir_functions.insert(f.signature.name.clone(), f.clone());
        }

        let script_fn = module
            .functions
            .iter()
            .find(|f| f.signature.name == "__script" || f.signature.name == "main")
            .unwrap_or_else(|| {
                panic!("{bench_id} ({filename}) missing __script/main in lowered MIR")
            });
        assert!(
            vm.is_cranelift_eligible_in_vm(script_fn),
            "{bench_id} ({filename}) lowered entry '{}' MUST pass is_cranelift_eligible_in_vm",
            script_fn.signature.name
        );

        // Also verify every MIR `__script` emitted by `Compiler::compile_module` passes!
        let comp_script = compiler_mirs
            .iter()
            .find(|f| f.signature.name == "__script")
            .unwrap_or_else(|| {
                panic!("{bench_id} ({filename}) missing __script in Compiler::compile_module MIR")
            });
        let mut vm_comp = VM::new();
        vm_comp.set_backend(VmBackend::Cranelift);
        vm_comp.register_structs(&struct_defs);
        for f in &compiler_mirs {
            vm_comp
                .mir_functions
                .insert(f.signature.name.clone(), f.clone());
        }
        assert!(
            vm_comp.is_cranelift_eligible_in_vm(comp_script),
            "{bench_id} ({filename}) Compiler::compile_module __script MUST pass is_cranelift_eligible_in_vm"
        );
    }
}

#[test]
fn test_ng_w16_6_deopt_materialization_for_closure_and_unknown_handles() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    // 1. Deopt recipe type acceptance for `MirType::Function`, `MirType::Unknown`, `LuaValue`,
    // and synthetic `__Enum_*` (always verified under both default and `test-deopt` builds).
    let vm = VM::new();
    let fn_ty = MirType::Function {
        params: vec![MirType::Int],
        return_type: Box::new(MirType::Int),
    };
    assert!(
        vm.is_supported_deopt_recipe_type(&fn_ty),
        "MirType::Function must be accepted in is_supported_deopt_recipe_type"
    );
    assert!(
        vm.is_supported_deopt_recipe_type(&MirType::Unknown),
        "MirType::Unknown must be accepted in is_supported_deopt_recipe_type"
    );
    assert!(
        vm.is_supported_deopt_recipe_type(&MirType::Struct("LuaValue".into())),
        "MirType::Struct(LuaValue) must be accepted in is_supported_deopt_recipe_type"
    );
    assert!(
        vm.is_supported_deopt_recipe_type(&MirType::Struct("__Enum_Op".into())),
        "synthetic MirType::Struct(__Enum_Op) must be accepted in is_supported_deopt_recipe_type"
    );
    assert!(
        !vm.is_supported_deopt_recipe_type(&MirType::Tuple(vec![MirType::Int])),
        "MirType::Tuple must remain rejected in is_supported_deopt_recipe_type (test_b2 contract)"
    );
    assert!(
        !vm.is_supported_deopt_recipe_type(&MirType::Struct("UnregisteredStruct".into())),
        "unregistered arbitrary struct must remain rejected in is_supported_deopt_recipe_type"
    );

    // 2. Full handle-backed deopt slot restoration under `--features test-deopt`.
    #[cfg(feature = "test-deopt")]
    {
        use lust::bytecode::{ClosureObject, EnumObject, Upvalue};
        use lust::mir::cranelift_backend::{
            DeoptPayload, DeoptSiteKind, push_native_invocation_with_generation,
            set_current_deopt_payload,
        };

        let mut vm_deopt = VM::new();
        vm_deopt.set_backend(VmBackend::Cranelift);

        // Register a dummy target function so `Value::Closure` / `Value::Function` have valid idx 0.
        let (bc_funcs, _, _) =
            compile_source_via_compiler("function target_fn(x: int): int return x + 1 end", false);
        vm_deopt.load_functions(bc_funcs);

        let local_types = vec![
            fn_ty.clone(),                        // slot 0: MirType::Function (Value::Closure)
            fn_ty.clone(),                        // slot 1: MirType::Function (Value::Function)
            MirType::Unknown,                     // slot 2: MirType::Unknown (LuaValue::Int(42))
            MirType::Unknown,                     // slot 3: MirType::Unknown (Value::String)
            MirType::Struct("LuaValue".into()),   // slot 4: MirType::Struct("LuaValue")
            MirType::Int,                         // slot 5: plain Int scalar
        ];

        let closure_rc = Rc::new(ClosureObject {
            function_idx: 0,
            upvalues: vec![Upvalue::new(Value::Int(7)), Upvalue::new(Value::Int(3))],
        });
        let closure_val = Value::Closure(Rc::clone(&closure_rc));
        let func_val = Value::Function(0);
        let native_val = Value::NativeFunction(native_fn(|_| Ok(NativeCallResult::Return(Value::Int(1)))));
        let lua_int_42 = Value::Enum(EnumObject::new(
            "LuaValue",
            "Int",
            Some(vec![Value::Int(42)]),
        ));
        let dyn_str = Value::string("hello_unknown");
        let lua_int_99 = Value::Enum(EnumObject::new(
            "LuaValue",
            "Int",
            Some(vec![Value::Int(99)]),
        ));

        let _ctx_guard = vm_deopt.mir_context.enter();
        let h_closure = vm_deopt.mir_context.register_handle(closure_val.clone());
        let h_func = vm_deopt.mir_context.register_handle(func_val.clone());
        let h_lua42 = vm_deopt.mir_context.register_handle(lua_int_42.clone());
        let h_str = vm_deopt.mir_context.register_handle(dyn_str.clone());
        let h_lua99 = vm_deopt.mir_context.register_handle(lua_int_99.clone());
        let h_native = vm_deopt.mir_context.register_handle(native_val.clone());

        let payload = {
            let _inv = push_native_invocation_with_generation(
                "deopt_closure_unknown".to_string(),
                1,
                1,
                11,
            );
            DeoptPayload::new(
                2,
                vec![
                    (0, h_closure),
                    (1, h_func),
                    (2, h_lua42),
                    (3, h_str),
                    (4, h_lua99),
                    (5, 1234),
                ],
            )
        };
        set_current_deopt_payload(payload);

        let (restored_ip, restored_regs) = vm_deopt
            .test_restore_deopt_top_frame_for_handles(
                "deopt_closure_unknown",
                11,
                1,
                2,
                DeoptSiteKind::BlockEntry,
                77,
                &local_types,
            )
            .expect("deopt restoration for Function + Unknown + LuaValue handles must succeed");

        assert_eq!(restored_ip, 77);
        match &restored_regs[0] {
            Value::Closure(rc) => assert!(
                Rc::ptr_eq(rc, &closure_rc),
                "restored Closure must preserve exact Rc identity and upvalues"
            ),
            other => panic!("expected Value::Closure in slot 0, got {other:?}"),
        }
        assert_eq!(restored_regs[1], Value::Function(0));
        assert_eq!(restored_regs[2], lua_int_42);
        assert_eq!(restored_regs[3], dyn_str);
        assert_eq!(restored_regs[4], lua_int_99);
        assert_eq!(restored_regs[5], Value::Int(1234));

        // NativeFunction handle is also valid for `MirType::Function`.
        let payload_native = {
            let _inv = push_native_invocation_with_generation(
                "deopt_native_fn".to_string(),
                1,
                2,
                11,
            );
            DeoptPayload::new(2, vec![(0, h_native)])
        };
        set_current_deopt_payload(payload_native);
        let (_, restored_native) = vm_deopt
            .test_restore_deopt_top_frame_for_handles(
                "deopt_native_fn",
                11,
                2,
                2,
                DeoptSiteKind::BlockEntry,
                88,
                std::slice::from_ref(&fn_ty),
            )
            .expect("NativeFunction handle must restore into MirType::Function slot");
        assert!(matches!(restored_native[0], Value::NativeFunction(_)));

        // Fail-closed negative checks: missing handle (0 / 999999) and wrong handle kind for
        // `MirType::Function` (e.g. String handle in a Function slot) MUST fail cleanly.
        for bad_handle in [0i64, 999_999i64, h_str] {
            let bad_payload = {
                let _inv = push_native_invocation_with_generation(
                    "deopt_bad_fn".to_string(),
                    1,
                    3,
                    11,
                );
                DeoptPayload::new(2, vec![(0, bad_handle)])
            };
            set_current_deopt_payload(bad_payload);
            let err = vm_deopt
                .test_restore_deopt_top_frame_for_handles(
                    "deopt_bad_fn",
                    11,
                    3,
                    2,
                    DeoptSiteKind::BlockEntry,
                    99,
                    std::slice::from_ref(&fn_ty),
                )
                .expect_err("forged/missing/wrong-typed Function handle must fail closed");
            assert!(
                err.to_string().contains("Function")
                    || err.to_string().contains("handle"),
                "unexpected error for bad Function handle {bad_handle}: {err}"
            );
        }

        for bad_unknown in [0i64, 999_999i64] {
            let bad_payload = {
                let _inv = push_native_invocation_with_generation(
                    "deopt_bad_unknown".to_string(),
                    1,
                    4,
                    11,
                );
                DeoptPayload::new(2, vec![(0, bad_unknown)])
            };
            set_current_deopt_payload(bad_payload);
            let err = vm_deopt
                .test_restore_deopt_top_frame_for_handles(
                    "deopt_bad_unknown",
                    11,
                    4,
                    2,
                    DeoptSiteKind::BlockEntry,
                    99,
                    &[MirType::Unknown],
                )
                .expect_err("missing Unknown handle must fail closed");
            assert!(
                err.to_string().contains("Unknown")
                    || err.to_string().contains("handle"),
                "unexpected error for bad Unknown handle {bad_unknown}: {err}"
            );
        }
    }
}

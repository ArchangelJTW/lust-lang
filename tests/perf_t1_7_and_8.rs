#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::fs::File;
use std::io::Write;
use std::process::Command;

use lust::bytecode::Instruction;
use lust::{
    Compiler, EmbeddedModule, LustConfig, TypeChecker, compile_program_with_config,
    load_program_from_embedded,
};
use tempfile::tempdir;

fn lust_bin() -> &'static str {
    env!("CARGO_BIN_EXE_lust")
}

const STRUCT_LOOP_SCRIPT: &str = r#"
struct Point
    x: int
    y: int
end

function sum_points(n: int): int
    local p: Point = Point { x = 1, y = 2 }
    local acc: int = 0
    local i: int = 0
    while i < n do
        p.x = p.x + 1
        acc = acc + p.x + p.y
        i = i + 1
    end
    return acc
end
"#;

const UNLOWERABLE_MIR_SCRIPT: &str = r#"
function make_closure(x: int): int
    local f = function(y: int): int
        return x + y
    end
    return f(10)
end
"#;

fn compile_embedded_to_functions(
    source: &str,
    config: &LustConfig,
) -> lust::Result<(Vec<lust::bytecode::Function>, Vec<lust::mir::ir::MirFunction>)> {
    let entries = &[EmbeddedModule {
        module: "main",
        parent: None,
        source: Some(source),
    }];
    let program = load_program_from_embedded(entries, "main")?;
    let mut imports_map = lust::hashbrown::HashMap::new();
    for m in &program.modules {
        imports_map.insert(m.path.clone(), m.imports.clone());
    }
    let mut typechecker = TypeChecker::with_config(config);
    typechecker.set_imports_by_module(imports_map.clone());
    typechecker.check_program(&program.modules)?;

    let mut wrapped_items = Vec::new();
    for m in program.modules {
        wrapped_items.push(lust::ast::Item::new(
            lust::ast::ItemKind::Module {
                name: m.path,
                items: m.items,
            },
            lust::ast::Span::new(0, 0, 0, 0),
        ));
    }

    let mut compiler = Compiler::new();
    compiler.set_option_coercions(typechecker.take_option_coercions());
    compiler.set_checked_array_indices(typechecker.take_checked_array_indices());
    compiler.set_lambda_return_types(typechecker.take_lambda_return_types());
    compiler.set_numeric_types(typechecker.take_numeric_types());
    compiler.set_imports_by_module(imports_map);
    compiler.set_entry_module(program.entry_module);
    compiler.set_function_signatures(typechecker.take_function_signatures());
    compiler.configure(config);
    let functions = compiler.compile_module(&wrapped_items)?;
    let mir_functions = compiler.take_mir_functions();
    Ok((functions, mir_functions))
}

#[test]
fn test_perf_t1_7_cranelift_ineligible_fn_keeps_ast_bytecode() {
    // (a.1) CLI `--jit=cranelift --disassemble` must keep AST->Bytecode (GetField/SetField),
    // NOT MIR-VM bytecode (NewArray/GetIndex/SetIndex), for a Cranelift-ineligible struct loop.
    let dir = tempdir().expect("create tempdir");
    let script_path = dir.path().join("struct_loop.lust");
    let mut f = File::create(&script_path).expect("create script");
    f.write_all(STRUCT_LOOP_SCRIPT.as_bytes())
        .expect("write script");

    let output = Command::new(lust_bin())
        .args(["--jit=cranelift", "--disassemble"])
        .arg(&script_path)
        .env_remove("LUST_JIT")
        .env_remove("LUST_JIT_BACKEND")
        .env_remove("LUST_BACKEND")
        .output()
        .expect("run lust --jit=cranelift --disassemble");
    assert!(
        output.status.success(),
        "lust --jit=cranelift --disassemble failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let disasm = String::from_utf8_lossy(&output.stdout);
    assert!(
        disasm.contains("GetField") && disasm.contains("SetField"),
        "expected AST->Bytecode GetField/SetField under --jit=cranelift, got:\n{disasm}"
    );
    assert!(
        !disasm.contains("NewArray")
            && !disasm.contains("GetIndex")
            && !disasm.contains("SetIndex"),
        "expected NO MIR-VM NewArray/GetIndex/SetIndex under --jit=cranelift, got:\n{disasm}"
    );

    // (a.2) Programmatic config (`cranelift_jit=true, use_mir=false`) must BOTH:
    //   - keep AST->Bytecode (GetField/SetField, no NewArray/GetIndex/SetIndex) in Function.chunk
    //   - still collect MIR candidates for Cranelift in `mir_functions`.
    let mut clif_cfg = LustConfig::default().with_cranelift_jit(true);
    clif_cfg.set_use_mir(false);
    let (functions, mir_functions) =
        compile_embedded_to_functions(STRUCT_LOOP_SCRIPT, &clif_cfg)
            .expect("compile with cranelift_jit=true, use_mir=false");
    let sum_fn = functions
        .iter()
        .find(|func| func.name.ends_with("sum_points"))
        .expect("sum_points function present");
    let has_get_field = sum_fn
        .chunk
        .instructions
        .iter()
        .any(|i| matches!(i, Instruction::GetField(..)));
    let has_set_field = sum_fn
        .chunk
        .instructions
        .iter()
        .any(|i| matches!(i, Instruction::SetField(..)));
    let has_mir_struct_array_ops = sum_fn.chunk.instructions.iter().any(|i| {
        matches!(
            i,
            Instruction::NewArray(..) | Instruction::GetIndex(..) | Instruction::SetIndex(..)
        )
    });
    assert!(
        has_get_field && has_set_field && !has_mir_struct_array_ops,
        "expected direct AST->Bytecode (GetField/SetField, no NewArray/GetIndex/SetIndex), got: {:?}",
        sum_fn.chunk.instructions
    );
    assert!(
        mir_functions
            .iter()
            .any(|m| m.signature.name.ends_with("sum_points")),
        "expected MIR candidate collected for Cranelift even when use_mir=false"
    );

    // (b) `--strict-mir` (`with_strict_mir(true)`) still emits MIR-VM bytecode and rejects
    // unlowerable functions.
    let strict_cfg = LustConfig::default().with_strict_mir(true);
    let (strict_funcs, strict_mirs) =
        compile_embedded_to_functions(STRUCT_LOOP_SCRIPT, &strict_cfg)
            .expect("strict-mir compile of struct loop");
    let strict_sum_fn = strict_funcs
        .iter()
        .find(|func| func.name.ends_with("sum_points"))
        .expect("strict sum_points present");
    assert!(
        !strict_sum_fn.block_entry_pcs.is_empty(),
        "strict-mir must still emit MIR-VM bytecode with block_entry_pcs"
    );
    assert!(
        strict_mirs
            .iter()
            .any(|m| m.signature.name.ends_with("sum_points")),
        "strict-mir must retain MIR function"
    );

    let entries = &[EmbeddedModule {
        module: "main",
        parent: None,
        source: Some(UNLOWERABLE_MIR_SCRIPT),
    }];
    let bad_program = load_program_from_embedded(entries, "main").expect("load unlowerable");
    let strict_err = compile_program_with_config(bad_program, &strict_cfg);
    assert!(
        strict_err.is_err(),
        "with_strict_mir(true) must reject unlowerable function"
    );
}

#[test]
fn test_perf_t1_7_deopt_resumes_in_continuation_and_rollback_restores_ast() {
    use lust::ast::Span;
    use lust::bytecode::Value;
    use lust::mir::cranelift_backend::{safepoint_request_deopt, safepoint_reset};
    use lust::mir::ir::{
        FunctionSignature, MirFunction, MirType, Operand, Place, Terminator,
    };

    let mixed_source = r#"
struct Point
    x: int
    y: int
end

function sum_points(n: int): int
    local p: Point = Point { x = 1, y = 2 }
    local acc: int = 0
    local i: int = 0
    while i < n do
        p.x = p.x + 1
        acc = acc + p.x + p.y
        i = i + 1
    end
    return acc
end

function sum_loop(n: int): int
    local acc: int = 0
    local i: int = 0
    while i < n do
        acc = acc + i
        i = i + 1
    end
    return acc
end
"#;

    let entries = &[EmbeddedModule {
        module: "main",
        parent: None,
        source: Some(mixed_source),
    }];
    let program = load_program_from_embedded(entries, "main").expect("load mixed");
    let mut cfg = LustConfig::default().with_cranelift_jit(true);
    cfg.set_use_mir(false);
    let mut vm = compile_program_with_config(program, &cfg).expect("compile mixed");

    let sum_points_name = vm
        .mir_functions
        .keys()
        .find(|k| k.ends_with("sum_points"))
        .cloned()
        .expect("sum_points in mir_functions");
    let sum_loop_name = vm
        .mir_functions
        .keys()
        .find(|k| k.ends_with("sum_loop"))
        .cloned()
        .expect("sum_loop in mir_functions");

    // Before compilation: BOTH functions hold direct AST->Bytecode (empty MIR block_entry_pcs).
    let orig_loop_insns = vm
        .function_by_name(&sum_loop_name)
        .expect("sum_loop bc")
        .chunk
        .instructions
        .clone();
    assert!(
        vm.function_by_name(&sum_loop_name)
            .unwrap()
            .block_entry_pcs
            .is_empty(),
        "sum_loop must start as AST->Bytecode before Cranelift compilation"
    );
    assert!(
        vm.function_by_name(&sum_points_name)
            .unwrap()
            .chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::GetField(..))),
        "sum_points must start as AST->Bytecode with GetField"
    );

    // Executing ineligible sum_points keeps AST->Bytecode and computes exact result (4+5+6+7+8 = 30).
    let pts_res = vm
        .call(&sum_points_name, vec![Value::Int(5)])
        .expect("call sum_points");
    assert_eq!(pts_res, Value::Int(30));
    assert_eq!(vm.cranelift_entry_count, 0);

    // Executing eligible sum_loop triggers lazy Cranelift compilation + installs continuation bc.
    let loop_res = vm
        .call(&sum_loop_name, vec![Value::Int(10)])
        .expect("call sum_loop");
    assert_eq!(loop_res, Value::Int(45));
    assert!(
        vm.cranelift_entry_count >= 1,
        "sum_loop must execute in Cranelift"
    );
    assert!(
        !vm.function_by_name(&sum_loop_name)
            .unwrap()
            .block_entry_pcs
            .is_empty(),
        "compiled sum_loop must have MIR continuation bytecode installed"
    );
    // Ineligible sum_points must remain untouched AST->Bytecode.
    assert!(
        vm.function_by_name(&sum_points_name)
            .unwrap()
            .block_entry_pcs
            .is_empty(),
        "ineligible sum_points must remain AST->Bytecode after compile_cranelift"
    );

    // Forced safepoint deopt in Cranelift resumes cleanly in installed MIR continuation bytecode!
    safepoint_reset();
    safepoint_request_deopt(true);
    let deopt_res = vm
        .call(&sum_loop_name, vec![Value::Int(10)])
        .expect("deopted sum_loop must resume cleanly in continuation bytecode");
    safepoint_reset();
    assert_eq!(deopt_res, Value::Int(45));

    // Recompilation rollback: replace sum_loop MIR with an ineligible function (Terminator::Raise),
    // recompile, and verify original AST bytecode is restored from ast_bytecode_backup!
    let mut ineligible_mir = MirFunction::new(
        FunctionSignature {
            name: sum_loop_name.clone(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        Span::dummy(),
    );
    let p0 = ineligible_mir.params[0];
    ineligible_mir.blocks[0].terminator =
        Some(Terminator::Raise(Operand::Copy(Place::Local(p0)), Span::dummy()));
    vm.register_mir_candidate(ineligible_mir);
    vm.compile_cranelift().expect("recompile after drop");

    let restored_fn = vm
        .function_by_name(&sum_loop_name)
        .expect("sum_loop restored");
    assert!(
        restored_fn.block_entry_pcs.is_empty(),
        "dropped function must restore AST bytecode (empty block_entry_pcs)"
    );
    assert_eq!(
        restored_fn.chunk.instructions, orig_loop_insns,
        "dropped function must restore exact original AST bytecode instructions"
    );
    let fallback_res = vm
        .call(&sum_loop_name, vec![Value::Int(10)])
        .expect("restored AST bytecode execution");
    assert_eq!(fallback_res, Value::Int(45));
}

#[test]
fn test_perf_t1_8_metadata_cache_hit_and_struct_invalidation() {
    use lust::bytecode::Value;

    let src = r#"
function add_one(x: int): int
    return x + 1
end
"#;
    let entries = &[EmbeddedModule {
        module: "main",
        parent: None,
        source: Some(src),
    }];
    let program = load_program_from_embedded(entries, "main").expect("load add_one");
    let mut cfg = LustConfig::default().with_cranelift_jit(true);
    cfg.set_use_mir(false);
    let mut vm = compile_program_with_config(program, &cfg).expect("compile add_one");
    let fn_name = vm
        .mir_functions
        .keys()
        .find(|k| k.ends_with("add_one"))
        .cloned()
        .expect("add_one in mir_functions");

    vm.compile_cranelift().expect("compile_cranelift");

    let refines_before = vm.cranelift_dispatch_refine_count();
    let syncs_before = vm.cranelift_struct_sync_count();
    for i in 0..100i64 {
        let res = vm.call(&fn_name, vec![Value::Int(i)]).expect("call add_one");
        assert_eq!(res, Value::Int(i + 1));
    }
    let refines_after = vm.cranelift_dispatch_refine_count();
    let syncs_after = vm.cranelift_struct_sync_count();
    assert_eq!(vm.cranelift_entry_count, 100);
    assert_eq!(
        refines_after - refines_before,
        0,
        "100 cached Cranelift calls must trigger 0 dispatch refines, got {}",
        refines_after - refines_before
    );
    assert_eq!(
        syncs_after - syncs_before,
        0,
        "100 cached Cranelift calls must trigger 0 struct metadata syncs, got {}",
        syncs_after - syncs_before
    );
}

#[test]
fn test_perf_t1_8_fresh_and_builtins_cache_restored_vm_always_sync_struct_metadata() {
    use lust::ast::Span;
    use lust::bytecode::Value;
    use lust::mir::ir::{
        FunctionSignature, MirConstant, MirFunction, MirType, Operand, Terminator,
    };
    use lust::VM;
    use lust::vm::VmBackend;

    let make_const_fn = |name: &str, c: i64| {
        let mut f = MirFunction::new(
            FunctionSignature {
                name: name.to_string(),
                param_types: vec![],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            Span::dummy(),
        );
        f.blocks[0].terminator = Some(Terminator::Return(
            Some(Operand::Constant(MirConstant::Int(c))),
            Span::dummy(),
        ));
        f
    };

    // VM 1 (fresh / populates BUILTINS_CACHE) and VM 2 (restored from BUILTINS_CACHE at api.rs:323).
    let cfg = LustConfig::default().with_cranelift_jit(true);
    for (idx, mut vm) in [VM::with_config(&cfg), VM::with_config(&cfg)]
        .into_iter()
        .enumerate()
    {
        assert_eq!(vm.backend(), VmBackend::Cranelift);
        assert_eq!(
            vm.mir_context_struct_gen(),
            0,
            "VM #{idx}: mir_context_struct_gen must initialize to 0"
        );
        assert!(
            vm.struct_metadata_generation() >= 1,
            "VM #{idx}: struct_metadata_generation must always be >= 1 (never 0)"
        );
        assert!(
            vm.mir_context.get_struct_field_type("IndexError", 0).is_none(),
            "VM #{idx}: MirContext must not yet have IndexError synced before first compile/call"
        );

        vm.load_mir_function(make_const_fn("ping", 7))
            .expect("load ping");
        let res = vm.call("ping", vec![]).expect("call ping");
        assert_eq!(res, Value::Int(7));
        assert_eq!(vm.cranelift_entry_count, 1);
        assert_eq!(
            vm.cranelift_struct_sync_count(),
            1,
            "VM #{idx}: first Cranelift compile/call must perform exactly 1 struct metadata sync"
        );
        assert_eq!(
            vm.mir_context_struct_gen(),
            vm.struct_metadata_generation(),
            "VM #{idx}: mir_context_struct_gen must match struct_metadata_generation after sync"
        );
        assert_eq!(
            vm.mir_context.get_struct_field_type("IndexError", 0),
            Some(MirType::Int),
            "VM #{idx}: MirContext must contain synced builtin IndexError layout"
        );

        // Subsequent calls must NOT re-sync struct metadata.
        let syncs_before = vm.cranelift_struct_sync_count();
        let refines_before = vm.cranelift_dispatch_refine_count();
        for _ in 0..20 {
            assert_eq!(vm.call("ping", vec![]).expect("cached ping"), Value::Int(7));
        }
        assert_eq!(vm.cranelift_struct_sync_count(), syncs_before);
        assert_eq!(vm.cranelift_dispatch_refine_count(), refines_before);
    }
}

#[test]
fn test_perf_t1_8_struct_mutation_and_hot_reload_invalidate_fn_cache() {
    use lust::ast::{FieldOwnership, Span, StructDef, StructField, Type, TypeKind, Visibility};
    use lust::bytecode::Value;
    use lust::mir::ir::{
        FunctionSignature, MirBinaryOp, MirConstant, MirFunction, MirType, Operand, Place, Rvalue,
        Statement, Terminator,
    };
    use lust::hashbrown::HashMap;
    use lust::VM;

    let make_adder = |delta: i64| {
        let mut f = MirFunction::new(
            FunctionSignature {
                name: "hot_fn".to_string(),
                param_types: vec![MirType::Int],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            Span::dummy(),
        );
        let p0 = f.params[0];
        let out = f.add_local(Some("out".to_string()), MirType::Int, false, Span::dummy());
        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(out),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(p0)),
                Operand::Constant(MirConstant::Int(delta)),
            ),
            Span::dummy(),
        ));
        f.blocks[0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(out))),
            Span::dummy(),
        ));
        f
    };

    let cfg = LustConfig::default().with_cranelift_jit(true);
    let mut vm = VM::with_config(&cfg);
    vm.load_mir_function(make_adder(10)).expect("load v1");
    vm.compile_cranelift().expect("compile v1");
    assert_eq!(
        vm.call("hot_fn", vec![Value::Int(5)]).expect("call v1"),
        Value::Int(15)
    );
    let gen_before_struct = vm.struct_metadata_generation();
    let syncs_before_struct = vm.cranelift_struct_sync_count();
    let comps_before_struct = vm.cranelift_compilation_count;

    // 1. Mutate struct_metadata via register_structs -> bumps generation, invalidates cache,
    //    and forces re-sync + recompilation on next call.
    let mut defs = HashMap::new();
    defs.insert(
        "DynamicBox".to_string(),
        StructDef {
            name: "DynamicBox".to_string(),
            type_params: vec![],
            trait_bounds: vec![],
            fields: vec![StructField {
                name: "v".to_string(),
                ty: Type::new(TypeKind::Int, Span::dummy()),
                visibility: Visibility::Public,
                ownership: FieldOwnership::Strong,
                weak_target: None,
            }],
            visibility: Visibility::Public,
            doc: None,
        },
    );
    vm.register_structs(&defs);
    assert!(
        vm.struct_metadata_generation() > gen_before_struct,
        "register_structs must bump struct_metadata_generation"
    );
    assert_eq!(
        vm.call("hot_fn", vec![Value::Int(5)])
            .expect("call after struct mutation"),
        Value::Int(15)
    );
    assert!(
        vm.cranelift_struct_sync_count() > syncs_before_struct,
        "struct_metadata mutation must trigger MirContext re-sync"
    );
    assert!(
        vm.cranelift_compilation_count > comps_before_struct,
        "struct_metadata mutation must invalidate cranelift_fn_cache and recompile"
    );
    assert_eq!(
        vm.mir_context.get_struct_field_type("DynamicBox", 0),
        Some(MirType::Int),
        "MirContext must reflect newly registered DynamicBox struct layout"
    );

    // 2. Hot-reload MIR function (B14-style swap): must invalidate cranelift_fn_cache and
    //    execute v2 (+100) rather than stale v1 (+10) code_ptr!
    let comps_before_swap = vm.cranelift_compilation_count;
    vm.load_mir_function(make_adder(100)).expect("load v2");
    let swapped = vm
        .call("hot_fn", vec![Value::Int(5)])
        .expect("call after hot reload");
    assert_eq!(
        swapped,
        Value::Int(105),
        "hot-reloaded MIR function must execute new code_ptr, not stale cached entry"
    );
    assert!(vm.cranelift_compilation_count > comps_before_swap);

    // Subsequent calls to v2 must hit cache with 0 refines and 0 struct syncs!
    let refines_before = vm.cranelift_dispatch_refine_count();
    let syncs_before = vm.cranelift_struct_sync_count();
    for i in 0..50i64 {
        assert_eq!(
            vm.call("hot_fn", vec![Value::Int(i)]).expect("cached v2"),
            Value::Int(i + 100)
        );
    }
    assert_eq!(vm.cranelift_dispatch_refine_count(), refines_before);
    assert_eq!(vm.cranelift_struct_sync_count(), syncs_before);
}

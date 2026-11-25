//! NG-W16-5: CLI & MIR Frontend Wiring (`enum_defs`, Lambda Lifting, `self` Type),
//! Call-Site Polymorphic/Closure Specialization, `Queue`/`string.byte` Lowering & Affine Bounds.

use lust::ast::{FieldOwnership, Item, ItemKind, Span, StructDef, StructField, Type, TypeKind, Visibility};
use lust::bytecode::{Compiler, Instruction};
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{
    safepoint_reset, safepoint_stats, take_current_sticky_error, take_deopt_payload,
};
use lust::mir::ir::{MirFunction, MirModule, MirType, Rvalue, Statement, Terminator};
use lust::mir::opt::{JitInlineConfig, MirInliner, MirOptimizer};
use lust::mir::verifier::MirVerifier;
use lust::mir::AstLowering;
use lust::modules::ModuleLoader;
use lust::parser::Parser;
use lust::vfs::VirtualConsole;
use lust::vm::{reset_effect_handler_state, VmBackend, VM};
use lust::compile_program_with_config;
use std::path::PathBuf;
use std::sync::Mutex;

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn reset_all() {
    safepoint_reset();
    reset_effect_handler_state();
    take_deopt_payload();
    take_current_sticky_error();
}

fn suite_v2_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/suite_v2")
        .join(filename)
}

fn parse_items(source: &str) -> Vec<Item> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("lex source");
    Parser::new(tokens).parse().expect("parse source")
}

fn lower_module_with_script(source: &str, name: &str) -> MirModule {
    let items = parse_items(source);
    let mut non_script: Vec<Item> = Vec::new();
    let mut script_stmts = Vec::new();
    for item in items {
        match item.kind {
            ItemKind::Script(stmts) => script_stmts.extend(stmts),
            _ => non_script.push(item),
        }
    }
    if !script_stmts.is_empty() {
        non_script.push(Item {
            kind: ItemKind::Function(lust::ast::FunctionDef {
                name: "__script".to_string(),
                type_params: vec![],
                trait_bounds: vec![],
                params: vec![],
                return_type: None,
                body: script_stmts,
                is_method: false,
                visibility: lust::ast::Visibility::Private,
                doc: None,
                effects: None,
            }),
            span: lust::ast::Span::dummy(),
        });
    }
    AstLowering::lower_module_items(name, &non_script).expect("lower_module_items")
}

fn compile_strict_mir(source: &str) -> (Vec<lust::bytecode::Function>, Vec<MirFunction>) {
    let items = parse_items(source);
    let mut compiler = Compiler::new();
    compiler.set_strict_mir(true);
    let funcs = compiler
        .compile_module(&items)
        .expect("Compiler::compile_module with strict_mir = true must succeed");
    let mir_fns = compiler.take_mir_functions();
    (funcs, mir_fns)
}

fn compile_collect_mir(source: &str) -> (Vec<lust::bytecode::Function>, Vec<MirFunction>) {
    let items = parse_items(source);
    let mut compiler = Compiler::new();
    compiler.set_use_mir(false);
    compiler.set_collect_mir(true);
    let funcs = compiler
        .compile_module(&items)
        .expect("Compiler::compile_module with collect_mir = true must succeed");
    let mir_fns = compiler.take_mir_functions();
    (funcs, mir_fns)
}

fn collect_vm_struct_defs(items: &[Item]) -> lust::hashbrown::HashMap<String, StructDef> {
    let mut struct_defs = lust::hashbrown::HashMap::new();
    for item in items {
        match &item.kind {
            ItemKind::Struct(sd) => {
                struct_defs.insert(sd.name.clone(), sd.clone());
            }
            ItemKind::Enum(ed) => {
                let mut max_arity = 0usize;
                for v in &ed.variants {
                    if let Some(fs) = &v.fields {
                        max_arity = max_arity.max(fs.len());
                    }
                }
                let mut fields = Vec::with_capacity(1 + max_arity);
                fields.push(StructField {
                    name: "__tag".to_string(),
                    ty: Type::new(TypeKind::Int, Span::dummy()),
                    visibility: Visibility::Public,
                    ownership: FieldOwnership::Strong,
                    weak_target: None,
                });
                for i in 0..max_arity {
                    fields.push(StructField {
                        name: format!("__f{i}"),
                        ty: Type::new(TypeKind::Int, Span::dummy()),
                        visibility: Visibility::Public,
                        ownership: FieldOwnership::Strong,
                        weak_target: None,
                    });
                }
                let synth_name = format!("__Enum_{}", ed.name);
                struct_defs.insert(
                    synth_name.clone(),
                    StructDef {
                        name: synth_name,
                        type_params: vec![],
                        trait_bounds: vec![],
                        fields,
                        visibility: Visibility::Public,
                        doc: None,
                    },
                );
            }
            _ => {}
        }
    }
    struct_defs
}

fn run_vm_script_stdout(
    funcs: Vec<lust::bytecode::Function>,
    struct_defs: &lust::hashbrown::HashMap<String, StructDef>,
) -> String {
    let mut cfg = LustConfig::default();
    cfg.set_jit_enabled(false);
    let mut vm = VM::with_config(&cfg);
    vm.console = VirtualConsole::captured();
    vm.register_structs(struct_defs);
    vm.load_functions(funcs);
    vm.call("__script", vec![]).expect("run __script");
    String::from_utf8(vm.console.stdout_bytes()).expect("utf8 stdout")
}

#[test]
fn test_ng_w16_5_cli_compile_module_lowers_adt_enum_eval_under_strict_mir() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    let src = std::fs::read_to_string(suite_v2_path("adt_enum_eval.lust"))
        .expect("read adt_enum_eval.lust");
    let (funcs, mir_fns) = compile_strict_mir(&src);

    let names: Vec<&str> = mir_fns.iter().map(|f| f.signature.name.as_str()).collect();
    assert!(
        names.contains(&"make_op") && names.contains(&"apply_op") && names.contains(&"__script"),
        "expected make_op, apply_op, __script in MIR functions, got: {names:?}"
    );

    for f in &mir_fns {
        MirVerifier::verify_fn(f)
            .unwrap_or_else(|e| panic!("MIR verification failed for {}: {e:?}", f.signature.name));
        assert!(
            f.struct_layouts.iter().any(|l| l.type_name == "__Enum_Op"),
            "synthetic __Enum_Op layout must be populated in {}.struct_layouts",
            f.signature.name
        );
    }

    // Verify exact runtime parity on a small-trip copy against AST VM.
    let small_src = src.replace("1000000", "2000");
    let small_items = parse_items(&small_src);
    let small_structs = collect_vm_struct_defs(&small_items);
    let (small_funcs, _) = compile_strict_mir(&small_src);
    let small_mir_out = run_vm_script_stdout(small_funcs, &small_structs);

    let mut ast_compiler = Compiler::new();
    let ast_funcs = ast_compiler.compile_module(&small_items).expect("ast compile");
    let small_ast_out = run_vm_script_stdout(ast_funcs, &small_structs);
    assert_eq!(
        small_mir_out.trim(),
        small_ast_out.trim(),
        "strict_mir output must match AST VM output on adt_enum_eval"
    );

    assert!(funcs.iter().any(|f| f.name == "__script"));
}

#[test]
fn test_ng_w16_5_cli_compile_module_lowers_closure_upvalue_combinators_under_strict_mir() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    let src = std::fs::read_to_string(suite_v2_path("closure_upvalue_combinators.lust"))
        .expect("read closure_upvalue_combinators.lust");
    let (_funcs, mir_fns) = compile_strict_mir(&src);

    let script_mir = mir_fns
        .iter()
        .find(|f| f.signature.name == "__script")
        .expect("__script MIR function must exist");
    MirVerifier::verify_fn(script_mir).expect("__script MIR must verify cleanly");

    // Verify MirInliner specialized `apply_twice(step_fn, x)` and eliminated closures in `__script`.
    let has_closure = script_mir.blocks.iter().any(|b| {
        b.statements
            .iter()
            .any(|s| matches!(s, Statement::Assign(_, Rvalue::Closure { .. }, _)))
    });
    assert!(
        !has_closure,
        "MirInliner must specialize apply_twice(step_fn, x) and eliminate dead Rvalue::Closure in __script"
    );
    let has_apply_twice_call = script_mir.blocks.iter().any(|b| {
        matches!(
            &b.terminator,
            Some(Terminator::Call { callee, .. }) if callee == "apply_twice" || callee == "step_fn"
        )
    });
    assert!(
        !has_apply_twice_call,
        "__script must not retain unspecialized calls to apply_twice or step_fn"
    );

    // Verify bit-exact runtime parity (including persistent mutable `bias` upvalue!) vs AST VM.
    let small_src = src.replace("1000000", "2000");
    let empty_structs = lust::hashbrown::HashMap::new();
    let (small_mir_funcs, _) = compile_strict_mir(&small_src);
    let mir_out = run_vm_script_stdout(small_mir_funcs, &empty_structs);

    let items = parse_items(&small_src);
    let mut ast_compiler = Compiler::new();
    let ast_funcs = ast_compiler.compile_module(&items).expect("ast compile");
    let ast_out = run_vm_script_stdout(ast_funcs, &empty_structs);
    assert_eq!(
        mir_out.trim(),
        ast_out.trim(),
        "strict_mir specialized closure output must match AST VM output bit-for-bit"
    );
}

#[test]
fn test_ng_w16_5_inliner_devirtualizes_eval_poly_on_concrete_node_structs() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    let src = std::fs::read_to_string(suite_v2_path("megamorphic_method_ic.lust"))
        .expect("read megamorphic_method_ic.lust");

    // 1. Test via `AstLowering::lower_module_items` + `MirOptimizer::optimize_module_for_jit`
    let mut module = lower_module_with_script(&src, "megamorphic_method_ic");
    let _modified = MirOptimizer::optimize_module_for_jit(&mut module, &JitInlineConfig::default());
    let script_fn = module
        .functions
        .iter()
        .find(|f| f.signature.name == "__script")
        .expect("__script function");
    MirVerifier::verify_fn(script_fn).expect("__script must verify after devirtualization");

    let remaining_unknown_calls: Vec<String> = script_fn
        .blocks
        .iter()
        .filter_map(|b| match &b.terminator {
            Some(Terminator::Call { callee, .. })
                if callee == "__mir_unknown_get_field" || callee == "__mir_unknown_call_method" =>
            {
                Some(callee.clone())
            }
            _ => None,
        })
        .collect();
    assert!(
        remaining_unknown_calls.is_empty(),
        "__script must have 0 remaining __mir_unknown_get_field/__mir_unknown_call_method calls, got: {remaining_unknown_calls:?}"
    );

    // 2. Also test via CLI `Compiler::compile_module` (where MirModule::struct_layouts is empty
    //    and layouts come from `MirFunction::struct_layouts`!).
    let (ast_funcs, cli_mir_fns) = compile_collect_mir(&src);
    let mut cli_mod = MirModule::new("vm_cranelift_module".into());
    for f in cli_mir_fns {
        cli_mod.add_function(f);
    }
    let _ = MirOptimizer::optimize_module_for_jit(&mut cli_mod, &JitInlineConfig::default());
    let cli_script = cli_mod
        .functions
        .iter()
        .find(|f| f.signature.name == "__script")
        .expect("cli __script");
    let cli_unknown_calls = cli_script
        .blocks
        .iter()
        .filter(|b| {
            matches!(
                &b.terminator,
                Some(Terminator::Call { callee, .. })
                    if callee == "__mir_unknown_get_field" || callee == "__mir_unknown_call_method"
            )
        })
        .count();
    assert_eq!(
        cli_unknown_calls, 0,
        "CLI-lowered __script must also devirtualize all eval_poly sites to 0 unknown calls"
    );

    // 3. Verify AST bytecode `self` type registration unlocks `GetFieldSlot` in NodeA..D:score!
    for method_name in ["NodeA:score", "NodeB:score", "NodeC:score", "NodeD:score"] {
        let f = ast_funcs
            .iter()
            .find(|f| f.name == method_name)
            .unwrap_or_else(|| panic!("missing compiled method {method_name}"));
        let has_slot = f
            .chunk
            .instructions
            .iter()
            .any(|ins| matches!(ins, Instruction::GetFieldSlot(..)));
        let has_generic = f
            .chunk
            .instructions
            .iter()
            .any(|ins| matches!(ins, Instruction::GetField(..)));
        assert!(
            has_slot && !has_generic,
            "{method_name} must use GetFieldSlot (not generic GetField) after self type registration"
        );
    }
}

#[test]
fn test_ng_w16_5_sec01_exact_gas_preservation_after_poly_and_closure_inlining() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    // 1. Polymorphic struct dispatch (`eval_poly`) — 49 iterations -> exact gas cost = 51 (1 entry + 50 while-header checks)
    let poly_src_51 = r#"
struct NodeA x: int, w: int end
struct NodeB w: int, x: int, pad: int end

impl NodeA function score(self, k: int): int return self.x * k + self.w end end
impl NodeB function score(self, k: int): int return self.x * k - self.w end end

function eval_poly(obj: unknown, k: int): int
    local base: int = obj.x + obj.w
    return base + obj:score(k)
end

local a: NodeA = NodeA { x = 3, w = 10 }
local b: NodeB = NodeB { w = 7, x = 5, pad = 0 }

local total: int = 0
local i: int = 0
while i < 49 do
    if (i & 1) == 0 then
        total = (total + eval_poly(a, i)) % 1000000007
    else
        total = (total + eval_poly(b, i)) % 1000000007
    end
    i = i + 1
end
println(total)
"#;

    // 2. Closure upvalue combinator (`apply_twice` + `step_fn`) — 49 iterations -> exact gas cost = 51
    let closure_src_51 = r#"
local apply_twice = function(f: function(int): int, x: int): int
    return f(f(x))
end

local bias: int = 7
local scale: int = 3
local step_fn = function(v: int): int
    bias = (bias + 1) % 997
    return (v * scale + bias) % 1000003
end

local acc: int = 1
local i: int = 0
while i < 49 do
    acc = apply_twice(step_fn, acc)
    i = i + 1
end
println(acc + step_fn(0))
"#;

    // 3. Exact Q=64 boundary (62 iterations -> 64 gas; 63 iterations -> 65 gas)
    let poly_src_64 = poly_src_51.replace("i < 49", "i < 62");
    let closure_src_65 = closure_src_51.replace("i < 49", "i < 63");

    let cases: [(&str, &str, u64); 4] = [
        ("poly_51", poly_src_51, 51),
        ("closure_51", closure_src_51, 51),
        ("poly_64", &poly_src_64, 64),
        ("closure_65", &closure_src_65, 65),
    ];

    for (label, src, exact_cost) in cases {
        let items = parse_items(src);
        let mut struct_defs = lust::hashbrown::HashMap::new();
        for item in &items {
            if let ItemKind::Struct(sd) = &item.kind {
                struct_defs.insert(sd.name.clone(), sd.clone());
            }
        }
        // Reference output from AST VM with generous gas budget.
        let mut ast_compiler = Compiler::new();
        let ast_funcs = ast_compiler.compile_module(&items).expect("ast compile");
        let expected_stdout = run_vm_script_stdout(ast_funcs, &struct_defs);

        // Compile via `Compiler::compile_module` with `collect_mir = true`, specialize via
        // `MirOptimizer::optimize_module_for_jit`, and execute the specialized `__script` in Cranelift JIT.
        let (funcs, raw_mir_fns) = compile_collect_mir(src);
        let mut mir_mod = MirModule::new(format!("sec01_{label}"));
        for f in raw_mir_fns {
            mir_mod.add_function(f);
        }
        let _ = MirOptimizer::optimize_module_for_jit(&mut mir_mod, &JitInlineConfig::default());
        let specialized_script = mir_mod
            .functions
            .into_iter()
            .find(|f| f.signature.name == "__script")
            .expect("specialized __script");
        MirVerifier::verify_fn(&specialized_script).expect("specialized __script must verify");

        let limits: [u64; 7] = [0, 1, 50, 51, 63, 64, 65];
        for gas_limit in limits {
            reset_all();
            let mut cfg = LustConfig::default().with_cranelift_jit(true);
            cfg.set_use_mir(false);
            let mut vm_jit = VM::with_config(&cfg);
            vm_jit.console = VirtualConsole::captured();
            vm_jit.set_backend(VmBackend::Cranelift);
            vm_jit.register_structs(&struct_defs);
            vm_jit.load_functions(funcs.clone());
            vm_jit.register_mir_candidate(specialized_script.clone());
            vm_jit.compile_cranelift().expect("compile_cranelift");
            vm_jit.set_gas_budget(gas_limit);

            let res = vm_jit.call("__script", vec![]);
            if gas_limit < exact_cost {
                let err = res.expect_err(&format!(
                    "{label}: expected OutOfGas at gas_limit={gas_limit} (< {exact_cost})"
                ));
                assert!(
                    err.to_string().contains("Out of gas"),
                    "{label} gas_limit={gas_limit}: unexpected error {err}"
                );
                assert_eq!(
                    vm_jit.gas_remaining(),
                    Some(0),
                    "{label} gas_limit={gas_limit}: gas_remaining must be 0"
                );
                assert_eq!(
                    vm_jit.cranelift_entry_count, 1,
                    "{label} gas_limit={gas_limit}: must enter Cranelift JIT natively (entries=1)"
                );
                assert_eq!(
                    vm_jit.gas_used(),
                    if gas_limit == 0 { 2 } else { gas_limit + 1 },
                    "{label} gas_limit={gas_limit}: exact gas_used overshoot check"
                );
                let sp = safepoint_stats().safepoints_checked;
                assert!(
                    sp <= gas_limit + 1,
                    "{label} gas_limit={gas_limit}: safepoints_checked={sp} must not overshoot"
                );
            } else {
                res.unwrap_or_else(|e| {
                    panic!("{label}: must succeed at gas_limit={gas_limit} (>= {exact_cost}), got {e}")
                });
                let got_stdout = String::from_utf8(vm_jit.console.stdout_bytes()).unwrap();
                assert_eq!(
                    got_stdout.trim(),
                    expected_stdout.trim(),
                    "{label} gas_limit={gas_limit}: stdout parity mismatch"
                );
                assert_eq!(
                    vm_jit.gas_used(),
                    exact_cost,
                    "{label} gas_limit={gas_limit}: exact gas_used mismatch"
                );
                assert_eq!(
                    vm_jit.gas_remaining(),
                    Some(gas_limit - exact_cost),
                    "{label} gas_limit={gas_limit}: exact gas_remaining mismatch"
                );
                assert_eq!(
                    vm_jit.cranelift_entry_count, 1,
                    "{label} gas_limit={gas_limit}: must execute natively in Cranelift JIT"
                );
            }
        }
    }
}

#[test]
fn test_ng_w16_5_queue_string_byte_ng13_inline_and_ng14_affine_bounds() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    // 1. BENCH-NG-12 (`fiber_channel_pingpong.lust`): Queue/offer/take native MIR lowering
    {
        let path = suite_v2_path("fiber_channel_pingpong.lust");
        let parent = path.parent().unwrap();
        let mut loader = ModuleLoader::new(parent);
        let program = loader
            .load_program_from_entry(path.to_str().unwrap())
            .expect("load fiber_channel_pingpong.lust");
        let mut cfg = LustConfig::default().with_cranelift_jit(true);
        cfg.set_use_mir(false);
        let vm = compile_program_with_config(program, &cfg)
            .expect("compile fiber_channel_pingpong.lust");
        let script = vm
            .mir_functions
            .get("__script")
            .expect("NG-12 __script MIR");
        MirVerifier::verify_fn(script).expect("NG-12 __script verifies");
        let callees: Vec<&str> = script
            .blocks
            .iter()
            .filter_map(|b| match &b.terminator {
                Some(Terminator::Call { callee, .. }) => Some(callee.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            callees.contains(&"queue.new")
                && callees.contains(&"queue.offer_int")
                && callees.contains(&"queue.take_int"),
            "NG-12 must lower Queue/offer/take to queue.new/queue.offer_int/queue.take_int, got: {callees:?}"
        );
        assert!(
            !callees.contains(&"__mir_unknown_get_field"),
            "NG-12 must not emit __mir_unknown_get_field: {callees:?}"
        );
    }

    // 2. BENCH-NG-09 (`string_parse_tokenize.lust`): string.byte Array<LuaValue> -> Array<int>
    {
        let src = std::fs::read_to_string(suite_v2_path("string_parse_tokenize.lust"))
            .expect("read string_parse_tokenize.lust");
        let (_funcs, mir_fns) = compile_collect_mir(&src);
        let script = mir_fns
            .iter()
            .find(|f| f.signature.name == "__script")
            .expect("NG-09 __script MIR");
        MirVerifier::verify_fn(script).expect("NG-09 __script verifies");
        let has_lua_value_ty = script.locals.iter().any(|l| {
            matches!(&l.ty, MirType::Struct(s) if s == "LuaValue")
                || matches!(&l.ty, MirType::Array(inner) if matches!(inner.as_ref(), MirType::Struct(s) if s == "LuaValue"))
        });
        assert!(
            !has_lua_value_ty,
            "NG-09 string.byte int-only locals must lower as Array<int>/int without Struct(LuaValue)"
        );
    }

    // 3. BENCH-NG-13 (`schema_json_decode_pipeline.lust`): parse_int & decode_telemetry_event inline into __script
    {
        let src = std::fs::read_to_string(suite_v2_path("schema_json_decode_pipeline.lust"))
            .expect("read schema_json_decode_pipeline.lust");
        let (_funcs, cli_mir_fns) = compile_collect_mir(&src);
        let mut module = MirModule::new("ng13".into());
        for f in cli_mir_fns {
            module.add_function(f);
        }
        let parse_int_fn = module
            .functions
            .iter()
            .find(|f| f.signature.name == "parse_int")
            .expect("parse_int");
        assert!(
            MirInliner::is_eligible_small_loop_leaf_callee(parse_int_fn),
            "parse_int (blocks={}, stmts={}) must qualify as small loop leaf callee",
            parse_int_fn.blocks.len(),
            parse_int_fn
                .blocks
                .iter()
                .map(|b| b.statements.len())
                .sum::<usize>()
        );
        let _ = MirOptimizer::optimize_module_for_jit(&mut module, &JitInlineConfig::default());
        let script = module
            .functions
            .iter()
            .find(|f| f.signature.name == "__script")
            .expect("NG-13 __script");
        let remaining_decode_calls: Vec<&str> = script
            .blocks
            .iter()
            .filter_map(|b| match &b.terminator {
                Some(Terminator::Call { callee, .. })
                    if callee == "parse_int" || callee == "decode_telemetry_event" =>
                {
                    Some(callee.as_str())
                }
                _ => None,
            })
            .collect();
        assert!(
            remaining_decode_calls.is_empty(),
            "NG-13 must inline decode_telemetry_event and parse_int into __script, remaining: {remaining_decode_calls:?}"
        );
    }

    // 4. BENCH-NG-14 (`loop_3level_call_barrier.lust`): affine `idx = offset + k` proven in bounds
    {
        let src = std::fs::read_to_string(suite_v2_path("loop_3level_call_barrier.lust"))
            .expect("read loop_3level_call_barrier.lust");
        let mut module = lower_module_with_script(&src, "ng14");
        let _ = MirOptimizer::optimize_module_for_jit(&mut module, &JitInlineConfig::default());
        let script = module
            .functions
            .iter()
            .find(|f| f.signature.name == "__script")
            .expect("NG-14 __script");
        let excluded = lust::hashbrown::HashSet::new();
        let analysis = script.compute_loop_induction_bounds_analysis(script.entry.0, &excluded, true);
        let has_affine_proof = analysis
            .proven_in_bounds
            .values()
            .chain(analysis.proven_in_bounds_at_stmt.values())
            .flat_map(|pairs| pairs.iter())
            .any(|&(arr_loc, idx_loc)| {
                let arr_is_data = script.locals[arr_loc.0]
                    .name
                    .as_deref()
                    .is_some_and(|n| n.contains("data") || n.contains("arr"));
                let idx_is_shifted = script.locals[idx_loc.0]
                    .name
                    .as_deref()
                    .is_some_and(|n| n.contains("idx"));
                arr_is_data && idx_is_shifted
            });
        assert!(
            has_affine_proof,
            "NG-14 affine index `idx = offset + k` must be proven in bounds by compute_loop_induction_bounds_analysis"
        );
    }
}

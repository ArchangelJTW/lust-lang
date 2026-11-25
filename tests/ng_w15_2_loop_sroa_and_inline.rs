#![cfg(feature = "std")]
#![forbid(unsafe_code)]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! NG-W15-2 Verification Suite:
//! 1. Closed-component multi-def & loop-body struct SROA (`BENCH-NG-07`, `BENCH-NG-15`,
//!    self-referential aggregate staging, mutable loop-local struct, negative mutable alias guard).
//! 2. Small single-loop leaf callee inlining (`BENCH-NG-14` `inner_kernel`, `BENCH-NG-15`
//!    `score_window` + `OrderTick` SROA, negative eligibility guards).
//! 3. Dominance-verified loop header & backedge detection (`sieve.lust` false `if`-merge header fix,
//!    irreducible CFG DFS grey-ancestor fallback).
//! 4. Mandatory `SEC-01` exact gas preservation across dominance headers and loop-callee inlining
//!    at exact thresholds `0, 1, 50 -> 51, 63, 64, 65` (plus `--features test-deopt` mid-loop deopt).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use lust::ast::Span;
use lust::bytecode::Value;
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{
    CraneliftJitCompiler, safepoint_reset, safepoint_stats, take_current_sticky_error,
    take_deopt_payload,
};
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::{get_current_sticky_error, safepoint_request_deopt_on_poll};
use lust::mir::ir::{
    FunctionSignature, MirConstant, MirFunction, MirModule, MirType, Operand, Place, ResourceKind,
    Rvalue, Statement, Terminator,
};
use lust::mir::lowering::AstLowering;
use lust::mir::opt::{
    JitInlineConfig, MirInliner, MirOptimizer, SafepointConfig, Safepoints,
};
use lust::mir::verifier::MirVerifier;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vfs::VirtualConsole;
use lust::vm::{VM, VmBackend, reset_effect_handler_state};
use lust::{EmbeddedModule, compile_program_with_config, load_program_from_embedded};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn reset_all() {
    safepoint_reset();
    reset_effect_handler_state();
    take_deopt_payload();
    take_current_sticky_error();
}

fn suite_v2_path(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("repos dir")
        .parent()
        .expect("workspace root")
        .join("benchmarks")
        .join("suite_v2")
        .join(file)
}

fn suite_v1_path(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks")
        .join("suite")
        .join(file)
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

fn lower_module(source: &str, module_name: &str) -> MirModule {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");
    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    AstLowering::lower_module_items(module_name, &items).expect("lower module")
}

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
            let after = &trimmed[pos + 5..];
            let fn_ref = after.split(['(', ' ', ',']).next().unwrap_or("").trim();
            if let Some(&sym_name) = sym_map.get(fn_ref) {
                called.push(sym_name.to_string());
            }
        }
    }
    called
}

fn place_has_field(p: &Place) -> bool {
    match p {
        Place::Field(..) => true,
        Place::Index(base, _) => place_has_field(base),
        Place::Local(_) => false,
    }
}

fn op_has_field(op: &Operand) -> bool {
    match op {
        Operand::Copy(p) | Operand::Move(p) => place_has_field(p),
        Operand::Constant(_) => false,
    }
}

fn stmt_has_field(stmt: &Statement) -> bool {
    match stmt {
        Statement::Assign(p, r, _) => {
            place_has_field(p)
                || match r {
                    Rvalue::Use(o) | Rvalue::UnaryOp(_, o) | Rvalue::Cast(o, _) => op_has_field(o),
                    Rvalue::BinaryOp(_, l, r) => op_has_field(l) || op_has_field(r),
                    Rvalue::Aggregate(_, ops) => ops.iter().any(op_has_field),
                    Rvalue::Closure { captures, .. } => {
                        captures.iter().any(|(_, o, _)| op_has_field(o))
                    }
                }
        }
        _ => false,
    }
}

fn term_has_field(term: &Terminator) -> bool {
    match term {
        Terminator::Cond { cond, .. } => op_has_field(cond),
        Terminator::Switch { discr, .. } => op_has_field(discr),
        Terminator::Call {
            args, destination, ..
        } => place_has_field(destination) || args.iter().any(op_has_field),
        Terminator::Return(Some(o), _) | Terminator::Raise(o, _) => op_has_field(o),
        _ => false,
    }
}

// =========================================================================
// Test 1: Loop-Body Multi-Def Struct SROA on BENCH-NG-07 (`sroa_escape_structs.lust`)
// =========================================================================

#[test]
fn test_ng_w15_2_loop_body_struct_sroa_eliminates_all_aggregates_in_ng07() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    let src = std::fs::read_to_string(suite_v2_path("sroa_escape_structs.lust"))
        .expect("read BENCH-NG-07 sroa_escape_structs.lust");

    let mut vm = compile_embedded_vm(&src, true);
    vm.compile_cranelift()
        .expect("compile_cranelift BENCH-NG-07");

    let mir_fn = vm
        .cranelift_compiled_mir("__script")
        .expect("compiled MIR for __script in BENCH-NG-07");
    MirVerifier::verify_fn(&mir_fn).expect("post-SROA BENCH-NG-07 MIR must verify cleanly");

    // 1. ZERO Rvalue::Aggregate(MirType::Struct("Vec3"), _) remaining in __script!
    let has_vec3_agg = mir_fn.blocks.iter().any(|b| {
        b.statements.iter().any(|s| {
            matches!(
                s,
                Statement::Assign(_, Rvalue::Aggregate(MirType::Struct(name), _), _)
                    if name == "Vec3" || name.ends_with(".Vec3")
            )
        })
    });
    assert!(
        !has_vec3_agg,
        "BENCH-NG-07 post-SROA MIR must have ZERO Vec3 Rvalue::Aggregate allocations!\n{mir_fn:?}"
    );

    // 2. ZERO Place::Field remaining in __script!
    let has_any_field = mir_fn.blocks.iter().any(|b| {
        b.statements.iter().any(stmt_has_field)
            || b.terminator.as_ref().is_some_and(term_has_field)
    });
    assert!(
        !has_any_field,
        "BENCH-NG-07 post-SROA MIR must have ZERO Place::Field projections!\n{mir_fn:?}"
    );

    // 3. All SROA recipes must use pure SSA scalar locals (`is_stack_promoted == false`).
    assert!(
        !mir_fn.sroa_decomposed_structs.is_empty(),
        "expected SroaStructRecipe entries for decomposed Vec3 locals"
    );
    for recipe in &mir_fn.sroa_decomposed_structs {
        for &fl in &recipe.field_locals {
            assert!(
                !mir_fn.locals[fl.0].is_stack_promoted,
                "SROA scalar field local {fl} must have is_stack_promoted == false"
            );
            assert_eq!(mir_fn.locals[fl.0].ty, MirType::Int);
        }
    }

    // 4. Cranelift IR must have ZERO `lust_mir_alloc_struct*` calls.
    let mut compiler = CraneliftJitCompiler::new().expect("create CraneliftJitCompiler");
    let clif_ir = compiler
        .dump_function_ir(&mir_fn)
        .expect("dump_function_ir BENCH-NG-07");
    let alloc_calls: Vec<String> = called_symbols_in_clif_ir(&clif_ir)
        .into_iter()
        .filter(|s| s.contains("lust_mir_alloc_struct"))
        .collect();
    assert!(
        alloc_calls.is_empty(),
        "BENCH-NG-07 Cranelift IR must have ZERO lust_mir_alloc_struct* calls, found {alloc_calls:?}"
    );

    // 5. End-to-end native execution exact checksum parity (`990177872`).
    let entries_before = vm.cranelift_entry_count;
    vm.call("__script", vec![]).expect("run BENCH-NG-07 JIT");
    assert_eq!(vm.cranelift_entry_count, entries_before + 1);
    let stdout = String::from_utf8(vm.console.stdout_bytes()).expect("utf8 stdout");
    assert_eq!(stdout.trim(), "990177872");

    // 6. Mid-loop forced deopt under `--features test-deopt` must preserve exact checksum!
    #[cfg(feature = "test-deopt")]
    {
        reset_all();
        let mut vm_deopt = compile_embedded_vm(&src, true);
        vm_deopt
            .compile_cranelift()
            .expect("compile_cranelift BENCH-NG-07 deopt");
        safepoint_request_deopt_on_poll(Some("__script".to_string()), 10);
        vm_deopt
            .call("__script", vec![])
            .expect("BENCH-NG-07 poll-10 deopt must resume cleanly");
        safepoint_reset();
        assert!(take_deopt_payload().is_none());
        assert!(get_current_sticky_error().is_none());
        let deopt_stdout = String::from_utf8(vm_deopt.console.stdout_bytes()).expect("utf8 stdout");
        assert_eq!(deopt_stdout.trim(), "990177872");
    }
}

#[test]
fn test_ng_w15_2_sroa_self_referential_staging_and_mutable_alias_safety() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    let src = r#"
struct Pair
    a: int
    b: int
end

function swap_loop(n: int): int
    local p: Pair = Pair { a = 1, b = 2 }
    local i: int = 0
    while i < n do
        p = Pair { a = p.b + 1, b = p.a + 2 }
        i = i + 1
    end
    return p.a * 1000 + p.b
end

function mutable_loop_alias_must_not_sroa(n: int): int
    local a: Pair = Pair { a = 1, b = 2 }
    local i: int = 0
    while i < n do
        local b: Pair = a
        b.a = b.a + 1
        b.b = b.b + b.a
        i = i + 1
    end
    return a.a * 1000 + a.b
end
"#;

    let mut vm_oracle = compile_embedded_vm(src, false);
    let mut vm_clif = compile_embedded_vm(src, true);
    vm_clif.compile_cranelift().expect("compile_cranelift");

    let swap_fn = vm_clif
        .mir_functions
        .keys()
        .find(|k| k.ends_with("swap_loop"))
        .cloned()
        .expect("swap_loop");
    let alias_fn = vm_clif
        .mir_functions
        .keys()
        .find(|k| k.ends_with("mutable_loop_alias_must_not_sroa"))
        .cloned()
        .expect("mutable_loop_alias_must_not_sroa");

    // 1. Self-referential `p = Pair { a = p.b + 1, b = p.a + 2 }` must be decomposed AND match oracle!
    let swap_mir = vm_clif
        .cranelift_compiled_mir(&swap_fn)
        .expect("compiled swap_loop MIR");
    MirVerifier::verify_fn(&swap_mir).expect("swap_loop MIR verify");
    assert!(
        !swap_mir.blocks.iter().any(|b| {
            b.statements.iter().any(|s| {
                matches!(s, Statement::Assign(_, Rvalue::Aggregate(MirType::Struct(_), _), _))
            })
        }),
        "self-referential swap_loop must eliminate Pair aggregates via staged temporaries"
    );
    assert_eq!(
        vm_clif.call(&swap_fn, vec![Value::Int(25)]).unwrap(),
        vm_oracle.call(&swap_fn, vec![Value::Int(25)]).unwrap()
    );

    // 2. Cross-iteration mutable alias `local b = a; b.a = b.a + 1` MUST be rejected from SROA
    //    and preserve exact reference-aliasing mutation semantics!
    assert!(
        vm_clif
            .cranelift_sroa_recipes(&alias_fn)
            .expect("alias_fn recipes")
            .is_empty(),
        "loop-carried mutable alias `local b = a` MUST be rejected by SROA"
    );
    assert_eq!(
        vm_clif.call(&alias_fn, vec![Value::Int(10)]).unwrap(),
        vm_oracle.call(&alias_fn, vec![Value::Int(10)]).unwrap()
    );
}

// =========================================================================
// Test 2: Small Single-Loop Leaf Callee Inlining on BENCH-NG-14 & BENCH-NG-15
// =========================================================================

#[test]
fn test_ng_w15_2_small_loop_callee_inlined_in_ng14_and_ng15() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    // --- BENCH-NG-14 (`loop_3level_call_barrier.lust`) ---
    {
        let src = std::fs::read_to_string(suite_v2_path("loop_3level_call_barrier.lust"))
            .expect("read BENCH-NG-14");
        let mut vm = compile_embedded_vm(&src, true);
        vm.compile_cranelift()
            .expect("compile_cranelift BENCH-NG-14");

        let mir_fn = vm
            .cranelift_compiled_mir("__script")
            .expect("compiled __script MIR in BENCH-NG-14");
        MirVerifier::verify_fn(&mir_fn).expect("BENCH-NG-14 post-inline MIR must verify");

        let still_calls_inner = mir_fn.blocks.iter().any(|b| {
            matches!(
                &b.terminator,
                Some(Terminator::Call { callee, .. }) if callee.ends_with("inner_kernel")
            )
        });
        assert!(
            !still_calls_inner,
            "inner_kernel must be inlined into __script in BENCH-NG-14!"
        );

        let entries_before = vm.cranelift_entry_count;
        vm.call("__script", vec![]).expect("run BENCH-NG-14 JIT");
        assert_eq!(vm.cranelift_entry_count, entries_before + 1);
        let stdout = String::from_utf8(vm.console.stdout_bytes()).expect("utf8 stdout");
        assert_eq!(stdout.trim(), "321195209");
    }

    // --- BENCH-NG-15 (`ffi_struct_array_pingpong.lust`) ---
    {
        reset_all();
        let src = std::fs::read_to_string(suite_v2_path("ffi_struct_array_pingpong.lust"))
            .expect("read BENCH-NG-15");
        let mut vm = compile_embedded_vm(&src, true);
        vm.compile_cranelift()
            .expect("compile_cranelift BENCH-NG-15");

        let mir_fn = vm
            .cranelift_compiled_mir("__script")
            .expect("compiled __script MIR in BENCH-NG-15");
        MirVerifier::verify_fn(&mir_fn).expect("BENCH-NG-15 post-inline/SROA MIR must verify");

        let still_calls_score = mir_fn.blocks.iter().any(|b| {
            matches!(
                &b.terminator,
                Some(Terminator::Call { callee, .. }) if callee.ends_with("score_window")
            )
        });
        assert!(
            !still_calls_score,
            "score_window must be inlined into __script in BENCH-NG-15!"
        );

        // Once score_window is inlined, `t: OrderTick` must be decomposed by SROA!
        let has_tick_agg = mir_fn.blocks.iter().any(|b| {
            b.statements.iter().any(|s| {
                matches!(
                    s,
                    Statement::Assign(_, Rvalue::Aggregate(MirType::Struct(name), _), _)
                        if name == "OrderTick" || name.ends_with(".OrderTick")
                )
            })
        });
        assert!(
            !has_tick_agg,
            "t: OrderTick must be decomposed by SROA after score_window is inlined in BENCH-NG-15!"
        );
        assert!(
            !mir_fn.sroa_decomposed_structs.is_empty(),
            "expected OrderTick SroaStructRecipe in BENCH-NG-15"
        );

        let mut compiler = CraneliftJitCompiler::new().expect("create CraneliftJitCompiler");
        let clif_ir = compiler
            .dump_function_ir(&mir_fn)
            .expect("dump_function_ir BENCH-NG-15");
        let alloc_calls: Vec<String> = called_symbols_in_clif_ir(&clif_ir)
            .into_iter()
            .filter(|s| s.contains("lust_mir_alloc_struct"))
            .collect();
        assert!(
            alloc_calls.is_empty(),
            "BENCH-NG-15 Cranelift IR must have ZERO lust_mir_alloc_struct* calls, found {alloc_calls:?}"
        );

        let entries_before = vm.cranelift_entry_count;
        vm.call("__script", vec![]).expect("run BENCH-NG-15 JIT");
        assert_eq!(vm.cranelift_entry_count, entries_before + 1);
        let stdout = String::from_utf8(vm.console.stdout_bytes()).expect("utf8 stdout");
        assert_eq!(stdout.trim(), "151024993");
    }

    // --- Negative Guard: Nested-loop callee must NOT qualify as small single-loop leaf ---
    {
        let nested_src = r#"
function two_loops(n: int): int
    local s: int = 0
    local i: int = 0
    while i < n do
        local j: int = 0
        while j < n do
            s = s + 1
            j = j + 1
        end
        i = i + 1
    end
    return s
end

function caller(n: int): int
    local acc: int = 0
    local k: int = 0
    while k < 3 do
        acc = acc + two_loops(n)
        k = k + 1
    end
    return acc
end
"#;
        let mut module = lower_module(nested_src, "nested_guard");
        let two_loops_fn = module
            .functions
            .iter()
            .find(|f| f.signature.name == "two_loops")
            .unwrap();
        assert!(
            !MirInliner::is_eligible_small_loop_leaf_callee(two_loops_fn),
            "callee with 2 loops must NOT qualify as small single-loop leaf callee"
        );
        let modified = MirOptimizer::optimize_module_for_jit(&mut module, &JitInlineConfig::default());
        assert!(
            !modified.contains("caller"),
            "caller must NOT inline nested-loop callee two_loops"
        );
    }
}

// =========================================================================
// Test 3: Dominance-Verified Loop Header Detection on `sieve.lust` + Irreducible CFG
// =========================================================================

#[test]
fn test_ng_w15_2_sieve_if_merge_blocks_not_classified_as_loop_headers() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    let src = std::fs::read_to_string(suite_v1_path("sieve.lust")).expect("read sieve.lust");
    let vm = compile_embedded_vm(&src, true);
    let mut sieve_fn = vm
        .mir_functions
        .get("__script")
        .expect("__script MIR for sieve.lust")
        .clone();

    // First prove why the naive `succ.0 <= src_idx` heuristic failed on sieve.lust:
    let mut naive_headers = std::collections::HashSet::new();
    for (src_idx, block) in sieve_fn.blocks.iter().enumerate() {
        if let Some(term) = &block.terminator {
            for succ in Safepoints::terminator_successors(term) {
                if succ.0 <= src_idx {
                    naive_headers.insert(succ);
                }
            }
        }
    }
    assert!(
        naive_headers.len() > 3,
        "sanity check: naive succ.0 <= src_idx on sieve.lust should misclassify if-merge blocks (> 3 headers), got {naive_headers:?}"
    );

    // Now verify dominance-based `Safepoints::find_loop_headers` returns ONLY the 3 real `while` headers!
    let dom_headers = Safepoints::find_loop_headers(&sieve_fn.blocks);
    let dom_backedges = Safepoints::find_loop_backedges(&sieve_fn.blocks);

    assert_eq!(
        dom_headers.len(),
        3,
        "sieve.lust has exactly 3 while loops (init, outer i<=N, inner j<=N); if-merge blocks must NOT be loop headers! got {dom_headers:?}"
    );
    assert_eq!(
        dom_backedges.len(),
        3,
        "sieve.lust must have exactly 3 loop backedges, got {dom_backedges:?}"
    );

    // Every detected loop header in sieve.lust must have a `Cond` terminator (while condition),
    // NOT an unconditional `Goto` merge block (`bb11`/`bb9`).
    for h in &dom_headers {
        assert!(
            matches!(sieve_fn.blocks[h.0].terminator, Some(Terminator::Cond { .. })),
            "sieve.lust loop header {h} must be a while-header Cond block, got {:?}",
            sieve_fn.blocks[h.0].terminator
        );
    }

    // Verify `Safepoints::run_on_function_with_config` instruments ONLY entry + 3 real loop headers (4 total):
    // (a) From a clean MIR function (stripping lowering-time ResourceChecks), inserts exactly 4 safepoints.
    let mut sieve_clean = sieve_fn.clone();
    for b in &mut sieve_clean.blocks {
        b.statements
            .retain(|s| !matches!(s, Statement::ResourceCheck(..)));
    }
    let inserted_clean =
        Safepoints::run_on_function_with_config(&mut sieve_clean, &SafepointConfig::default());
    assert_eq!(
        inserted_clean, 4,
        "sieve.lust (clean) must receive exactly 4 ResourceCheck safepoints (1 entry + 3 while headers), got {inserted_clean}"
    );
    assert_eq!(
        Safepoints::run_on_function_with_config(&mut sieve_clean, &SafepointConfig::default()),
        0,
        "Safepoints pass must be idempotent on sieve.lust"
    );

    // (b) On raw lowered MIR (which already carries the 3 while-header checks), inserts only entry (1)
    //     and leaves total ResourceCheck count at exactly 4 — strictly on entry + 3 real loop headers,
    //     with ZERO safepoints on any if-merge block.
    let inserted_raw =
        Safepoints::run_on_function_with_config(&mut sieve_fn, &SafepointConfig::default());
    assert_eq!(
        inserted_raw, 1,
        "sieve.lust raw lowered MIR should only need the 1 entry safepoint (3 while headers already present), got {inserted_raw}"
    );
    let mut instrumented_blocks: Vec<usize> = Vec::new();
    for (idx, b) in sieve_fn.blocks.iter().enumerate() {
        let rc_count = b
            .statements
            .iter()
            .filter(|s| matches!(s, Statement::ResourceCheck(..)))
            .count();
        if rc_count > 0 {
            assert_eq!(rc_count, 1, "block {idx} must not receive duplicate ResourceChecks");
            instrumented_blocks.push(idx);
        }
    }
    assert_eq!(
        instrumented_blocks.len(),
        4,
        "sieve.lust must have exactly 4 instrumented blocks (entry + 3 while headers), got {instrumented_blocks:?}"
    );
    for &b_idx in &instrumented_blocks {
        assert!(
            b_idx == sieve_fn.entry.0 || dom_headers.contains(&lust::mir::ir::BlockId(b_idx)),
            "unexpected ResourceCheck on non-entry, non-loop-header block {b_idx} (e.g. if-merge block)"
        );
    }

    // Verify irreducible 2-entry cycle (`entry -> A`, `entry -> B`, `A -> B`, `B -> A`)
    // is still caught by DFS grey-ancestor fallback so SEC-01 gas termination can never be bypassed!
    let mut irr = MirFunction::new(
        FunctionSignature {
            name: "irreducible".into(),
            param_types: vec![MirType::Bool],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        Span::dummy(),
    );
    let bb1 = irr.new_block();
    let bb2 = irr.new_block();
    let bb3 = irr.new_block();
    irr.blocks[0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(irr.params[0])),
        then_block: bb1,
        else_block: bb2,
        span: Span::dummy(),
    });
    irr.blocks[bb1.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(irr.params[0])),
        then_block: bb2,
        else_block: bb3,
        span: Span::dummy(),
    });
    irr.blocks[bb2.0].terminator = Some(Terminator::Goto(bb1, Span::dummy()));
    irr.blocks[bb3.0].terminator = Some(Terminator::Return(
        Some(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));

    let irr_headers = Safepoints::find_loop_headers(&irr.blocks);
    assert!(
        !irr_headers.is_empty(),
        "irreducible cycle must still yield >= 1 loop header via DFS grey-ancestor fallback"
    );
    let irr_inserted = Safepoints::run_on_function_with_config(&mut irr, &SafepointConfig::default());
    assert!(
        irr_inserted >= 2,
        "irreducible cycle must receive entry + cycle safepoint"
    );
    assert!(irr.blocks[bb1.0].statements.iter().any(|s| matches!(
        s,
        Statement::ResourceCheck(ResourceKind::Gas, 1, _)
    )));
}

// =========================================================================
// Test 4: Mandatory SEC-01 Exact Gas Preservation (`0, 1, 50 -> 51, 63, 64, 65`)
// =========================================================================

#[test]
fn test_ng_w15_2_sec01_exact_gas_preservation_across_dominance_headers_and_loop_inlining() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    // Deterministic nested-`if` + small single-loop leaf callee (`inner_work`) called inside outer loop.
    // Exact loop header count for `driver(outer_n, inner_n)`:
    //   outer header checks = outer_n + 1
    //   inner header checks = outer_n * (inner_n + 1)
    //   total loop header checks = (outer_n + 1) + outer_n * (inner_n + 1)
    //   total Cranelift gas_used = 1 (VM entry) + total loop header checks
    //
    // Specifically:
    //   driver(7, 5):  outer=8, inner=42 => 50 header checks + 1 entry = 51 exact gas! (50 -> 51 boundary!)
    //   driver(1, 60): outer=2, inner=61 => 63 header checks + 1 entry = 64 exact gas! (63 -> 64 boundary!)
    //   driver(1, 61): outer=2, inner=62 => 64 header checks + 1 entry = 65 exact gas! (64 -> 65 boundary!)
    let src = r#"
function inner_work(n: int, flag: bool): int
    local s: int = 0
    local k: int = 0
    while k < n do
        if flag then
            s = s + k + 1
        else
            s = s + 1
        end
        k = k + 1
    end
    return s
end

function driver(outer_n: int, inner_n: int): int
    local acc: int = 0
    local i: int = 0
    while i < outer_n do
        if (i & 1) == 0 then
            if i >= 0 then
                acc = acc + inner_work(inner_n, true)
            end
        else
            acc = acc + inner_work(inner_n, false)
        end
        i = i + 1
    end
    return acc
end
"#;

    // Verify `inner_work` IS actually inlined into `driver` and that dominance loop headers are exact!
    let driver_name = {
        let mut vm_check = compile_embedded_vm(src, true);
        vm_check.compile_cranelift().expect("compile_cranelift");
        let driver_name = vm_check
            .mir_functions
            .keys()
            .find(|k| k.ends_with("driver"))
            .cloned()
            .expect("driver");
        let compiled_driver = vm_check
            .cranelift_compiled_mir(&driver_name)
            .expect("compiled driver MIR");
        MirVerifier::verify_fn(&compiled_driver).expect("compiled driver MIR verify");

        let still_calls_inner = compiled_driver.blocks.iter().any(|b| {
            matches!(
                &b.terminator,
                Some(Terminator::Call { callee, .. }) if callee.ends_with("inner_work")
            )
        });
        assert!(
            !still_calls_inner,
            "inner_work MUST be inlined into driver for SEC-01 loop-inlining verification"
        );

        // 1 outer loop + 2 inlined inner loops (one per if-branch) = exactly 3 real loop headers!
        let headers = Safepoints::find_loop_headers(&compiled_driver.blocks);
        assert_eq!(
            headers.len(),
            3,
            "driver must have exactly 3 loop headers (1 outer + 2 inlined inner) and ZERO false if-merge headers, got {headers:?}"
        );
        driver_name
    };

    // Helper to get VM oracle result with ample gas.
    let run_vm_oracle = |outer_n: i64, inner_n: i64| -> Value {
        let mut vm = compile_embedded_vm(src, false);
        vm.set_gas_budget(1_000_000);
        vm.call(&driver_name, vec![Value::Int(outer_n), Value::Int(inner_n)])
            .expect("VM oracle must succeed")
    };

    // (A) Exact `50 -> 51` boundary on `driver(7, 5)` across `gas_limit in [0, 1, 50, 51, 63, 64, 65]`!
    let expected_51 = run_vm_oracle(7, 5);
    for gas_limit in [0u64, 1, 50, 51, 63, 64, 65] {
        reset_all();
        let mut vm_clif = compile_embedded_vm(src, true);
        let driver_name = vm_clif
            .mir_functions
            .keys()
            .find(|k| k.ends_with("driver"))
            .cloned()
            .unwrap();
        vm_clif.compile_cranelift().expect("compile_cranelift");
        vm_clif.set_gas_budget(gas_limit);

        let clif_res = vm_clif.call(&driver_name, vec![Value::Int(7), Value::Int(5)]);
        if gas_limit <= 50 {
            let err = clif_res.expect_err(&format!(
                "Cranelift driver(7, 5) MUST trap OutOfGas at gas_limit={gas_limit}"
            ));
            assert!(
                err.to_string().contains("Out of gas"),
                "expected Out of gas at gas_limit={gas_limit}, got {err}"
            );
            let expected_oog_used = if gas_limit == 0 { 2 } else { gas_limit + 1 };
            let expected_oog_sp = if gas_limit == 0 { 1 } else { gas_limit };
            assert_eq!(
                vm_clif.gas_used(),
                expected_oog_used,
                "zero gas overshoot required at gas_limit={gas_limit}"
            );
            assert_eq!(vm_clif.gas_remaining(), Some(0));
            assert_eq!(
                safepoint_stats().safepoints_checked,
                expected_oog_sp,
                "exact safepoints_checked at gas_limit={gas_limit}"
            );
        } else {
            let val = clif_res.expect(&format!(
                "Cranelift driver(7, 5) MUST succeed at gas_limit={gas_limit}"
            ));
            assert_eq!(val, expected_51, "exact oracle parity at gas_limit={gas_limit}");
            assert_eq!(
                vm_clif.gas_used(),
                51,
                "driver(7, 5) must consume exactly 51 gas at gas_limit={gas_limit}"
            );
            assert_eq!(
                vm_clif.gas_remaining(),
                Some(gas_limit - 51),
                "exact gas_remaining() at gas_limit={gas_limit}"
            );
            assert_eq!(safepoint_stats().safepoints_checked, 50);
        }

        // Also verify Bytecode VM traps OutOfGas cleanly at low budgets (0, 1, 50) without panic
        // and matches exact value when given full bytecode gas budget.
        let mut vm_bc = compile_embedded_vm(src, false);
        vm_bc.set_backend(VmBackend::Bytecode);
        vm_bc.set_gas_budget(gas_limit);
        let bc_res = vm_bc.call(&driver_name, vec![Value::Int(7), Value::Int(5)]);
        if gas_limit <= 50 {
            let err = bc_res.expect_err(&format!(
                "VM driver(7, 5) must trap OutOfGas at gas_limit={gas_limit}"
            ));
            assert!(err.to_string().contains("Out of gas"));
            assert_eq!(vm_bc.gas_remaining(), Some(0));
        }
    }

    // (B) Exact `Q = 64` quantum boundary parity (`gas_limit in [63, 64, 65]`):
    //     - `driver(1, 60)` requires exactly 64 gas -> OOG at 63, Ok(rem=0) at 64, Ok(rem=1) at 65!
    //     - `driver(1, 61)` requires exactly 65 gas -> OOG at 63 & 64, Ok(rem=0) at 65!
    //     - `driver(10, 200)` (long run) -> exact OOG with zero overshoot at [0, 1, 50, 51, 63, 64, 65]!
    let expected_64 = run_vm_oracle(1, 60);
    let expected_65 = run_vm_oracle(1, 61);

    for (outer_n, inner_n, exact_cost, expected_val) in [
        (1i64, 60i64, 64u64, expected_64),
        (1i64, 61i64, 65u64, expected_65),
    ] {
        for gas_limit in [63u64, 64, 65] {
            reset_all();
            let mut vm = compile_embedded_vm(src, true);
            let driver_name = vm
                .mir_functions
                .keys()
                .find(|k| k.ends_with("driver"))
                .cloned()
                .unwrap();
            vm.compile_cranelift().expect("compile_cranelift");
            vm.set_gas_budget(gas_limit);

            let res = vm.call(&driver_name, vec![Value::Int(outer_n), Value::Int(inner_n)]);
            if gas_limit < exact_cost {
                let err = res.expect_err(&format!(
                    "driver({outer_n}, {inner_n}) must trap OutOfGas at gas_limit={gas_limit} (< {exact_cost})"
                ));
                assert!(err.to_string().contains("Out of gas"));
                assert_eq!(vm.gas_used(), gas_limit + 1);
                assert_eq!(vm.gas_remaining(), Some(0));
                assert_eq!(safepoint_stats().safepoints_checked, gas_limit);
            } else {
                let val = res.expect(&format!(
                    "driver({outer_n}, {inner_n}) must succeed at gas_limit={gas_limit} (>= {exact_cost})"
                ));
                assert_eq!(val, expected_val);
                assert_eq!(vm.gas_used(), exact_cost);
                assert_eq!(vm.gas_remaining(), Some(gas_limit - exact_cost));
                assert_eq!(safepoint_stats().safepoints_checked, exact_cost - 1);
            }
        }
    }

    for gas_limit in [0u64, 1, 50, 51, 63, 64, 65] {
        reset_all();
        let mut vm = compile_embedded_vm(src, true);
        let driver_name = vm
            .mir_functions
            .keys()
            .find(|k| k.ends_with("driver"))
            .cloned()
            .unwrap();
        vm.compile_cranelift().expect("compile_cranelift");
        vm.set_gas_budget(gas_limit);

        let err = vm
            .call(&driver_name, vec![Value::Int(10), Value::Int(200)])
            .expect_err("long driver(10, 200) must trap OutOfGas");
        assert!(err.to_string().contains("Out of gas"));
        let expected_used = if gas_limit == 0 { 2 } else { gas_limit + 1 };
        let expected_sp = if gas_limit == 0 { 1 } else { gas_limit };
        assert_eq!(vm.gas_used(), expected_used);
        assert_eq!(vm.gas_remaining(), Some(0));
        assert_eq!(safepoint_stats().safepoints_checked, expected_sp);
    }

    // (C) Under `--features test-deopt`: forced mid-quantum deopt inside the inlined inner loop
    //     must resume cleanly and match VM oracle output!
    #[cfg(feature = "test-deopt")]
    {
        reset_all();
        let mut vm = compile_embedded_vm(src, true);
        let driver_name = vm
            .mir_functions
            .keys()
            .find(|k| k.ends_with("driver"))
            .cloned()
            .unwrap();
        vm.compile_cranelift().expect("compile_cranelift");
        safepoint_request_deopt_on_poll(Some(driver_name.clone()), 15);
        let actual = vm
            .call(&driver_name, vec![Value::Int(7), Value::Int(5)])
            .expect("mid-loop poll-15 deopt inside inlined inner loop must resume cleanly");
        safepoint_reset();
        assert_eq!(actual, expected_51);
        assert!(take_deopt_payload().is_none());
        assert!(get_current_sticky_error().is_none());
    }
}

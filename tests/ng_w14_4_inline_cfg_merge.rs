#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! NG-W14-4 Verification Suite:
//! MIR Inliner Basic-Block Merging (`sub-echelon 14-E1`).
//!
//! Verified Live Anchors:
//! - `src/mir/opt/inline.rs:283` (`inline_module_for_jit`)
//! - `src/mir/opt/mod.rs:146` (`optimize_module_for_jit_with_layouts`)
//!
//! Invariants:
//! - Trivial unconditional `Goto` chains (`pre_call_bb -> cloned_entry_bb -> continuation_bb`)
//!   are merged when predecessor/successor counts are 1.
//! - Guardrails: Do NOT merge across exception/cleanup/effect handler boundaries or deopt-point headers.
//! - Preserves and remaps `proven_in_bounds` and deopt/cleanup metadata.
//! - `0` new unsafe code.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Mutex;

use lust::ast::Span;
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{
    safepoint_reset, take_current_sticky_error, take_deopt_payload,
};
use lust::mir::ir::{
    BlockId, FunctionSignature, LocalId, MirBinaryOp, MirCleanupEntry, MirCleanupKind,
    MirConstant, MirFunction, MirModule, MirType, Operand, Place, Rvalue, Statement, Terminator,
};
use lust::mir::lowering::AstLowering;
use lust::mir::opt::{JitInlineConfig, MirInliner, MirOptimizer};
use lust::mir::verifier::MirVerifier;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vfs::VirtualConsole;
use lust::vm::{reset_effect_handler_state, VM};
use lust::{compile_program_with_config, load_program_from_embedded, EmbeddedModule};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn reset_all() {
    safepoint_reset();
    reset_effect_handler_state();
    take_deopt_payload();
    take_current_sticky_error();
}

fn lower_module(source: &str) -> MirModule {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");
    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    AstLowering::lower_module_items("ng_w14_4_mod", &items).expect("lower module")
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
    if cranelift {
        vm.compile_cranelift().expect("compile cranelift");
    }
    vm
}

/// Identifies the basic blocks belonging to the while loop body.
/// Returns the list of block IDs reachable between the loop header's `then_block`
/// and the back-edge jumping to the loop header.
fn collect_loop_body_blocks(func: &MirFunction) -> Vec<BlockId> {
    let mut loop_headers: Vec<BlockId> =
        lust::mir::opt::Safepoints::find_loop_headers(&func.blocks)
            .into_iter()
            .collect();
    assert!(!loop_headers.is_empty(), "Function must have at least one natural loop");
    loop_headers.sort_by_key(|b| b.0);
    let header_id = loop_headers[0];
    let header_block = &func.blocks[header_id.0];

    let body_entry = match header_block.terminator.as_ref().unwrap() {
        Terminator::Cond { then_block, .. } => *then_block,
        other => panic!("Expected Cond terminator on loop header, got {:?}", other),
    };

    let mut body_blocks = Vec::new();
    let mut visited = HashSet::new();
    let mut stack = vec![body_entry];

    while let Some(bb) = stack.pop() {
        if bb == header_id || !visited.insert(bb) {
            continue;
        }
        body_blocks.push(bb);
        if let Some(term) = &func.blocks[bb.0].terminator {
            match term {
                Terminator::Goto(target, _) => {
                    if *target != header_id {
                        stack.push(*target);
                    }
                }
                Terminator::Cond { then_block, else_block, .. } => {
                    if *then_block != header_id {
                        stack.push(*then_block);
                    }
                    if *else_block != header_id {
                        stack.push(*else_block);
                    }
                }
                Terminator::Call { success, error, .. } => {
                    if *success != header_id {
                        stack.push(*success);
                    }
                    if let Some(err_bb) = error {
                        if *err_bb != header_id {
                            stack.push(*err_bb);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    body_blocks.sort_by_key(|b| b.0);
    body_blocks
}

// =========================================================================
// Test 1: Core NG-W14-4 RED -> GREEN Verification
// =========================================================================

#[test]
fn test_ng_w14_4_inlined_helper_merges_trivial_blocks() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    let src = r#"
function step(a: int, b: int): int
    return a + b * 2
end

function run_calls(n: int): int
    local acc: int = 0
    local i: int = 0
    while i < n do
        acc = step(acc, i)
        i = i + 1
    end
    return acc
end
"#;

    // 1. Lower to clean MIR
    let module = lower_module(src);
    let step_fn = module
        .functions
        .iter()
        .find(|f| f.signature.name == "step")
        .expect("step function must exist")
        .clone();
    let run_calls_orig = module
        .functions
        .iter()
        .find(|f| f.signature.name == "run_calls")
        .expect("run_calls function must exist")
        .clone();

    // 2. RED Check: Splicing without basic-block merging generates exactly 3 split blocks
    // in the loop body (`pre_call_bb -> cloned_entry_bb -> continuation_bb`).
    let inliner = MirInliner::new();
    let mut run_calls_red = run_calls_orig.clone();

    // Find the call site in run_calls
    let call_block_idx = run_calls_red
        .blocks
        .iter()
        .position(|b| {
            matches!(
                &b.terminator,
                Some(Terminator::Call { callee, .. }) if callee == "step"
            )
        })
        .expect("run_calls must contain call to step");

    let (call_args, destination, succ_bb, err_bb, call_span) = match run_calls_red.blocks[call_block_idx].terminator.clone().unwrap() {
        Terminator::Call { args, destination, success, error, span, .. } => (args, destination, success, error, span),
        _ => unreachable!(),
    };

    inliner.inline_call_site_unmerged(
        &mut run_calls_red,
        call_block_idx,
        &step_fn,
        &call_args,
        destination.clone(),
        succ_bb,
        err_bb,
        call_span,
    );

    let red_loop_body = collect_loop_body_blocks(&run_calls_red);
    assert_eq!(
        red_loop_body.len(),
        3,
        "RED: Unmerged inline call site must produce exactly 3 split blocks in the loop body (`pre_call_bb -> cloned_entry_bb -> continuation_bb`), got {:?}",
        red_loop_body
    );

    // Verify the unmerged chain: pre_call_bb -> cloned_entry_bb -> continuation_bb -> loop header
    let loop_hdr_id = run_calls_red.blocks[0].terminator.as_ref().unwrap().successors()[0];
    let pre_call_id = run_calls_red.blocks[loop_hdr_id]
        .terminator
        .as_ref()
        .unwrap()
        .successors()
        .into_iter()
        .find(|s| red_loop_body.iter().any(|b| b.0 == *s))
        .expect("loop header must branch to pre_call_bb");
    let pre_call_term = run_calls_red.blocks[pre_call_id].terminator.as_ref().unwrap();
    let cloned_id = match pre_call_term {
        Terminator::Goto(target, _) if red_loop_body.contains(target) && target.0 != pre_call_id => target.0,
        other => panic!("pre_call_bb must jump to cloned_entry_bb, got {other:?}"),
    };
    let cloned_term = run_calls_red.blocks[cloned_id].terminator.as_ref().unwrap();
    let cont_id = match cloned_term {
        Terminator::Goto(target, _) if red_loop_body.contains(target) && target.0 != pre_call_id && target.0 != cloned_id => target.0,
        other => panic!("cloned_entry_bb must jump to continuation_bb, got {other:?}"),
    };
    let cont_term = run_calls_red.blocks[cont_id].terminator.as_ref().unwrap();
    assert!(
        matches!(cont_term, Terminator::Goto(target, _) if target.0 == loop_hdr_id),
        "continuation_bb must jump back to loop header"
    );

    // 3. GREEN Check: `inline_call_site` or `merge_linear_blocks` collapses the 3 split blocks
    // into a single merged loop body block!
    let mut run_calls_green = run_calls_red.clone();
    let merged = MirInliner::merge_linear_blocks(&mut run_calls_green);
    assert!(merged, "merge_linear_blocks must report modifications");

    MirVerifier::verify_fn(&run_calls_green)
        .expect("Merged green MIR must pass MirVerifier verification");

    let green_loop_body = collect_loop_body_blocks(&run_calls_green);
    assert_eq!(
        green_loop_body.len(),
        1,
        "GREEN: Merged inline call site must collapse to a single basic block in the loop body, got {:?}",
        green_loop_body
    );

    // The single merged block must jump directly to the loop header
    let single_body_term = run_calls_green.blocks[green_loop_body[0].0].terminator.as_ref().unwrap();
    assert!(
        matches!(single_body_term, Terminator::Goto(target, _) if target.0 == run_calls_green.blocks[0].terminator.as_ref().unwrap().successors()[0]),
        "Single merged loop body block must jump directly to the loop header"
    );

    // 4. Also test via full `MirInliner::inline_module_for_jit` / `MirOptimizer::optimize_module_for_jit_with_layouts`
    let mut opt_module = lower_module(src);
    let cfg = JitInlineConfig::default();
    let modified = MirOptimizer::optimize_module_for_jit(&mut opt_module, &cfg);
    assert!(modified.contains("run_calls"), "run_calls must be modified by inliner");

    let opt_run_calls = opt_module
        .functions
        .iter()
        .find(|f| f.signature.name == "run_calls")
        .unwrap();
    MirVerifier::verify_fn(opt_run_calls).expect("Optimized run_calls must verify");
    let opt_loop_body = collect_loop_body_blocks(opt_run_calls);
    assert_eq!(
        opt_loop_body.len(),
        1,
        "Full optimize_module_for_jit must produce single-block loop body, got {:?}",
        opt_loop_body
    );

    // 5. Differential JIT vs VM Parity Verification
    for test_n in [0, 1, 5, 20, 1000] {
        let runner_src = format!(
            r#"
function step(a: int, b: int): int
    return a + b * 2
end

function run_calls(n: int): int
    local acc: int = 0
    local i: int = 0
    while i < n do
        acc = step(acc, i)
        i = i + 1
    end
    return acc
end

println(run_calls({test_n}))
"#
        );

        let mut vm_jit = compile_embedded_vm(&runner_src, true);
        vm_jit.call("__script", vec![]).expect("JIT call __script");
        let jit_stdout = String::from_utf8(vm_jit.console.stdout_bytes()).expect("utf8 jit stdout");

        let mut vm_interp = compile_embedded_vm(&runner_src, false);
        vm_interp.call("__script", vec![]).expect("VM call __script");
        let vm_stdout = String::from_utf8(vm_interp.console.stdout_bytes()).expect("utf8 vm stdout");

        assert_eq!(
            jit_stdout, vm_stdout,
            "JIT vs VM stdout mismatch for n={test_n}: JIT='{jit_stdout}', VM='{vm_stdout}'"
        );
        assert!(
            vm_jit.cranelift_entry_count >= 1,
            "Expected native Cranelift entry for n={test_n}"
        );
    }

    // 6. Verification with real calls.lust benchmark
    let calls_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks")
        .join("suite")
        .join("calls.lust");
    let calls_src = std::fs::read_to_string(&calls_path).expect("read calls.lust");
    let vm_calls_jit = compile_embedded_vm(&calls_src, true);
    let compiled_script = vm_calls_jit
        .cranelift_compiled_mir("__script")
        .expect("compiled MIR for __script");
    let calls_loop_body = collect_loop_body_blocks(&compiled_script);
    assert_eq!(
        calls_loop_body.len(),
        1,
        "calls.lust inlined loop body must be collapsed into 1 basic block, got {:?}",
        calls_loop_body
    );
}

// =========================================================================
// Test 2: Multi-block Straight-Line Callee Inlining & Merging
// =========================================================================

#[test]
fn test_ng_w14_4_straight_line_multi_block_callee_merges_completely() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    // Construct a straight-line callee with 3 basic blocks:
    // block 0: t0 = a + b; Goto(1)
    // block 1: t1 = t0 * 3; Goto(2)
    // block 2: Return(t1)
    let mut callee = MirFunction::new(
        FunctionSignature {
            name: "three_block_step".into(),
            param_types: vec![MirType::Int, MirType::Int],
            return_type: MirType::Int,
            is_public: false,
            effects: None,
        },
        Span::dummy(),
    );
    let b1 = callee.new_block();
    let b2 = callee.new_block();

    let t0 = callee.add_local(Some("t0".into()), MirType::Int, false, Span::dummy());
    let t1 = callee.add_local(Some("t1".into()), MirType::Int, false, Span::dummy());

    callee.blocks[0].statements.push(Statement::Assign(
        Place::Local(t0),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(callee.params[0])),
            Operand::Copy(Place::Local(callee.params[1])),
        ),
        Span::dummy(),
    ));
    callee.blocks[0].terminator = Some(Terminator::Goto(b1, Span::dummy()));

    callee.blocks[b1.0].statements.push(Statement::Assign(
        Place::Local(t1),
        Rvalue::BinaryOp(
            MirBinaryOp::Mul,
            Operand::Copy(Place::Local(t0)),
            Operand::Constant(MirConstant::Int(3)),
        ),
        Span::dummy(),
    ));
    callee.blocks[b1.0].terminator = Some(Terminator::Goto(b2, Span::dummy()));

    callee.blocks[b2.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(t1))),
        Span::dummy(),
    ));

    assert!(
        MirInliner::is_straight_line(&callee),
        "three_block_step must be recognized as straight-line"
    );

    let src = r#"
function run_loop(): int
    local acc: int = 1
    local i: int = 1
    while i < 10 do
        acc = three_block_step(acc, i)
        i = i + 1
    end
    return acc
end
"#;
    let mut module = lower_module(src);
    module.functions.push(callee);

    let cfg = JitInlineConfig::default();
    let modified = MirInliner::inline_module_for_jit(&mut module, &cfg);
    assert!(modified.contains("run_loop"));

    let post_fn = module.functions.iter().find(|f| f.signature.name == "run_loop").unwrap();
    MirVerifier::verify_fn(post_fn).expect("Merged run_loop must verify");

    let loop_body = collect_loop_body_blocks(post_fn);
    assert_eq!(
        loop_body.len(),
        1,
        "3-block straight-line callee inlined into loop body must collapse into exactly 1 basic block, got {:?}",
        loop_body
    );
}

// =========================================================================
// Test 3: Safety Guardrails — Exception & Cleanup Boundaries Preserved
// =========================================================================

#[test]
fn test_ng_w14_4_guardrail_exception_and_cleanup_boundaries_preserved() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    // Construct a MIR function with:
    // Block 0: normal statements, Goto(1)
    // Block 1: cleanup block (is_cleanup = true), Goto(2)
    // Block 2: normal block, Return
    let mut func = MirFunction::new(
        FunctionSignature {
            name: "test_guard_cleanup".into(),
            param_types: vec![],
            return_type: MirType::Unit,
            is_public: false,
            effects: None,
        },
        Span::dummy(),
    );
    let b1 = func.new_block();
    let b2 = func.new_block();

    func.blocks[0].terminator = Some(Terminator::Goto(b1, Span::dummy()));
    func.blocks[b1.0].is_cleanup = true;
    func.blocks[b1.0].terminator = Some(Terminator::Goto(b2, Span::dummy()));
    func.blocks[b2.0].terminator = Some(Terminator::Return(None, Span::dummy()));

    func.cleanup_blocks.push(MirCleanupEntry {
        kind: MirCleanupKind::Defer,
        block: b1,
    });

    let before_blocks = func.blocks.len();
    let merged = MirInliner::merge_linear_blocks(&mut func);
    assert!(!merged, "Must NOT merge across cleanup boundary or cleanup entry block");
    assert_eq!(func.blocks.len(), before_blocks);
    assert_eq!(func.cleanup_blocks.len(), 1);
    assert_eq!(func.cleanup_blocks[0].block, b1);
    assert!(func.blocks[b1.0].is_cleanup);

    // Also test exception / error target of Terminator::Call
    let mut func2 = MirFunction::new(
        FunctionSignature {
            name: "test_guard_unwind".into(),
            param_types: vec![],
            return_type: MirType::Unit,
            is_public: false,
            effects: None,
        },
        Span::dummy(),
    );
    let b_err = func2.new_block();
    let b_succ = func2.new_block();

    func2.blocks[0].terminator = Some(Terminator::Call {
        callee: "some_fn".into(),
        args: vec![],
        destination: Place::Local(LocalId(0)),
        success: b_succ,
        error: Some(b_err),
        span: Span::dummy(),
    });
    // b_err has Goto to b_succ
    func2.blocks[b_err.0].terminator = Some(Terminator::Goto(b_succ, Span::dummy()));
    func2.blocks[b_succ.0].terminator = Some(Terminator::Return(None, Span::dummy()));

    // b_err is an error target — it must NOT be merged into anything, nor should anything merge into it
    let merged2 = MirInliner::merge_linear_blocks(&mut func2);
    assert!(!merged2, "Must NOT merge across call unwind/error target");
}

// =========================================================================
// Test 4: Safety Guardrails — Loop Headers & Deopt Headers Preserved
// =========================================================================

#[test]
fn test_ng_w14_4_guardrail_loop_headers_and_deopt_headers_preserved() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    let src = r#"
function run_nested(): int
    local sum: int = 0
    local i: int = 0
    while i < 10 do
        local j: int = 0
        while j < 10 do
            sum = sum + i * j
            j = j + 1
        end
        i = i + 1
    end
    return sum
end
"#;
    let mut module = lower_module(src);
    let func = module.functions.iter_mut().find(|f| f.signature.name == "run_nested").unwrap();

    let loop_headers_before = lust::mir::opt::Safepoints::find_loop_headers(&func.blocks);
    assert_eq!(loop_headers_before.len(), 2, "Expected 2 nested loop headers");

    // Pre-loop block jumps unconditionally to outer loop header.
    // Inner loop pre-header jumps unconditionally to inner loop header.
    // merge_linear_blocks MUST NOT merge loop headers into their pre-headers!
    MirInliner::merge_linear_blocks(func);

    let loop_headers_after = lust::mir::opt::Safepoints::find_loop_headers(&func.blocks);
    assert_eq!(
        loop_headers_after.len(),
        2,
        "Loop headers must not be merged away or destroyed"
    );

    MirVerifier::verify_fn(func).expect("Function must verify after merge attempt");

    // Verify deopt metadata derivation at entry of both loop headers succeeds
    for hdr in loop_headers_after {
        let meta = func.derive_proven_site_metadata_at_entry(hdr.0);
        assert!(
            meta.is_ok(),
            "Deopt site metadata derivation must succeed at loop header {}: {:?}",
            hdr.0,
            meta.err()
        );
    }
}

// =========================================================================
// Test 5: Proven In Bounds Preservation and Remapping
// =========================================================================

#[test]
fn test_ng_w14_4_proven_in_bounds_preserved_and_remapped() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all();

    // Construct a function with proven_in_bounds on block 1, jumping to block 2:
    let mut func = MirFunction::new(
        FunctionSignature {
            name: "test_pib".into(),
            param_types: vec![MirType::Array(Box::new(MirType::Int)), MirType::Int],
            return_type: MirType::Int,
            is_public: false,
            effects: None,
        },
        Span::dummy(),
    );
    let b1 = func.new_block();
    let b2 = func.new_block();
    let res = func.add_local(Some("r".into()), MirType::Int, false, Span::dummy());

    func.blocks[0].terminator = Some(Terminator::Goto(b1, Span::dummy()));

    // b1 has proven_in_bounds for (param0, param1)
    func.proven_in_bounds
        .insert(b1.0, vec![(func.params[0], func.params[1])]);
    func.blocks[b1.0].terminator = Some(Terminator::Goto(b2, Span::dummy()));

    func.blocks[b2.0].statements.push(Statement::Assign(
        Place::Local(res),
        Rvalue::Use(Operand::Copy(Place::Index(
            Box::new(Place::Local(func.params[0])),
            func.params[1],
        ))),
        Span::dummy(),
    ));
    func.blocks[b2.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(res))),
        Span::dummy(),
    ));

    // When b2 is merged into b1, proven_in_bounds MUST be preserved on b1!
    // And when b1 is merged into b0, proven_in_bounds MUST be preserved on b0!
    let changed = MirInliner::merge_linear_blocks(&mut func);
    assert!(changed, "linear blocks must merge");

    MirVerifier::verify_fn(&func).expect("verify");
    assert_eq!(func.blocks.len(), 1, "All 3 linear blocks must collapse into block 0");

    assert!(
        func.proven_in_bounds.contains_key(&0),
        "proven_in_bounds must be preserved on block 0 after full merge: {:?}",
        func.proven_in_bounds
    );
    assert_eq!(
        func.proven_in_bounds[&0],
        vec![(func.params[0], func.params[1])],
        "proven_in_bounds pairs must match exactly"
    );
}

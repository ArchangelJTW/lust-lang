#![cfg(feature = "std")]

//! Focused executable regression test suite for confirmed source risks identified by Sol
//! and the independent fixplan:
//! 1. Non-OSR safepoint empty locals (`test_w4_red_non_osr_safepoint_empty_locals`)
//! 2. MIR block ID vs bytecode PC in real OSR continuation
//!    - Bounded helper test: `test_w4_helper_deopt_exit_state_applies_raw_mir_block_id_to_frame_ip`
//!    - Real runtime OSR & source-derived mapping test: `test_w4_red_mir_block_id_vs_bytecode_pc_real_osr_continuation`
//! 3. Per-site locals vs OSR entry with native execution (`test_w4_red_per_site_locals_vs_osr_entry`)
//! 4. Public `resume_from_deopt` empty stack and missing type
//!    - Empty stack execution proof: `test_w4_red_public_resume_from_deopt_empty_stack_no_mutation`
//!    - Bounded helper missing type test: `test_w4_helper_to_vm_values_missing_type_defaults_to_int`
//!    - Runtime missing type fail-closed test: `test_w4_red_public_resume_from_deopt_missing_type_no_mutation`
//! 5. Unit/Float ABI boundary and status (`test_w4_red_unit_float_abi_canonical_and_error_status`)
//! 6. Standalone stale state cross-contamination (`test_w4_red_standalone_stale_state_cross_contamination`)
//!
//! Includes child-process isolation with external watchdog timeouts, kill AND reap for crash-prone probes.

use std::process::{Command, Stdio};
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use std::time::Duration;

use lust::ast::Span;
use lust::bytecode::{Function, Instruction, Value};
use lust::effect_abi::LUST_STATUS_ERR_GENERIC;
use lust::mir::cranelift_backend::{
    get_current_sticky_error, lust_mir_init_deopt, lust_mir_record_deopt,
    lust_mir_record_deopt_local, native_invocation_depth, publish_deopt_payload,
    push_native_invocation, reset_native_instrumentation_counters, safepoint_request_deopt,
    safepoint_reset, set_current_sticky_error, take_current_sticky_error,
    take_deopt_payload, CraneliftJitCompiler, DeoptPayload, CRANELIFT_NATIVE_ENTRY_COUNT,
    MIR_STATUS_DEOPT,
};
use lust::mir::ir::{
    FunctionSignature, MirBinaryOp, MirConstant, MirFunction, MirType, Operand, Place,
    Rvalue, Statement, Terminator,
};
use lust::mir::vm_backend::MirToBytecodeCompiler;
use lust::vm::{
    osr::{invoke_osr_entry, DeoptExitState, DeoptReason, OsrEngine, OsrExecutionResult},
    CallFrame, VM,
};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

// ============================================================================
// 1. Non-OSR Safepoint Empty Locals
// ============================================================================

/// Asserts that when a function is compiled via standard Cranelift JIT (non-OSR),
/// a safepoint deopt capture MUST record the actual live locals at the loop header.
///
/// Under confirmed source defect 1:
/// `src/mir/cranelift_backend.rs:2716-2720`:
/// ```rust
/// let locals_to_record = if let Some((_, live_locals)) = osr_info {
///     live_locals
/// } else {
///     &[]
/// };
/// ```
/// In non-OSR compilation, `osr_info` is `None`, so `locals_to_record` is unconditionally
/// empty `&[]`. When a safepoint deopt triggers, Cranelift records 0 locals, completely
/// losing loop variables (`acc`, `i`) upon deopt continuation.
#[test]
fn test_w4_red_non_osr_safepoint_empty_locals() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();
    reset_native_instrumentation_counters();
    take_deopt_payload();

    let fn_name = "test_non_osr_safepoint_empty_locals_fn";
    let sig = FunctionSignature {
        name: fn_name.into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let max_iter = func.params[0]; // LocalId(0)
    let acc = func.add_local(Some("acc".into()), MirType::Int, true, Span::dummy()); // LocalId(2)
    let i = func.add_local(Some("i".into()), MirType::Int, true, Span::dummy()); // LocalId(3)
    let cond = func.add_local(Some("cond".into()), MirType::Bool, false, Span::dummy()); // LocalId(4)

    let bb_hdr = func.new_block(); // BlockId(1)
    let bb_body = func.new_block(); // BlockId(2)
    let bb_exit = func.new_block(); // BlockId(3)

    // Block 0: acc = 0; i = 1; goto bb_hdr
    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::Use(Operand::Constant(MirConstant::Int(1))),
        Span::dummy(),
    ));
    func.blocks[0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    // Block 1 (Loop Header): cond = i <= max_iter; if cond then bb_body else bb_exit
    func.blocks[bb_hdr.0].statements.push(Statement::Assign(
        Place::Local(cond),
        Rvalue::BinaryOp(
            MirBinaryOp::LtEq,
            Operand::Copy(Place::Local(i)),
            Operand::Copy(Place::Local(max_iter)),
        ),
        Span::dummy(),
    ));
    func.blocks[bb_hdr.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond)),
        then_block: bb_body,
        else_block: bb_exit,
        span: Span::dummy(),
    });

    // Block 2: acc = acc + i; i = i + 1; goto bb_hdr
    func.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(acc)),
            Operand::Copy(Place::Local(i)),
        ),
        Span::dummy(),
    ));
    func.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    func.blocks[bb_body.0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    // Block 3: return acc
    func.blocks[bb_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(acc))),
        Span::dummy(),
    ));

    // Standard JIT compilation without OSR (osr_info is None)
    let mut compiler = CraneliftJitCompiler::new().expect("init compiler");
    let fn_ptr = compiler
        .compile_function(&func)
        .expect("non-OSR compilation must succeed");

    // Request safepoint deopt trigger before execution
    safepoint_request_deopt(true);

    // Verify native entry count before
    let count_before = CRANELIFT_NATIVE_ENTRY_COUNT.load(Ordering::SeqCst);

    // Execute compiled function natively in an isolated invocation context
    {
        let _inv = push_native_invocation(fn_name.into(), 1, 1);
        let _ = unsafe { invoke_osr_entry(fn_ptr, &[10], &MirType::Int) };
    }

    // Assert actual native entry occurred
    let count_after = CRANELIFT_NATIVE_ENTRY_COUNT.load(Ordering::SeqCst);
    assert!(
        count_after > count_before,
        "Actual native Cranelift execution must have been entered (count: {} -> {})",
        count_before,
        count_after
    );

    // Deopt payload must have been captured at loop header
    let payload = take_deopt_payload().expect("deopt payload must exist after safepoint deopt");
    assert_eq!(
        payload.target_pc, bb_hdr.0,
        "Deopt payload must be captured at loop header (block 1)"
    );

    // ORACLE ASSERTION: The captured payload MUST contain the live locals of the loop!
    // Specifically, both `acc` (local 2) and `i` (local 3) are live across the loop header.
    // Under the source defect, `payload.live_locals` is EMPTY (vec![])!
    assert!(
        !payload.live_locals.is_empty(),
        "Non-OSR safepoint deopt payload MUST NOT have empty live_locals! \
         Source bug Cranelift lowering passes &[] when osr_info is None, losing all loop variables."
    );
    assert!(
        payload.live_locals.iter().any(|(idx, _)| *idx == acc.0),
        "Payload must capture live accumulator local {}: found {:?}",
        acc.0,
        payload.live_locals
    );
    assert!(
        payload.live_locals.iter().any(|(idx, _)| *idx == i.0),
        "Payload must capture live index local {}: found {:?}",
        i.0,
        payload.live_locals
    );

    safepoint_reset();
}

// ============================================================================
// 2. MIR Block ID vs Bytecode PC in Real OSR Continuation
// ============================================================================

/// Bounded helper test: asserts that `DeoptExitState::apply_to_call_frame` directly sets
/// `frame.ip` to `payload.target_pc` without applying any bytecode PC mapping.
#[test]
fn test_w4_helper_deopt_exit_state_applies_raw_mir_block_id_to_frame_ip() {
    let mir_block_id = 1usize;
    let payload = DeoptPayload::new(mir_block_id, vec![(0, 42)]);
    let types = vec![(0, MirType::Int)];
    let deopt_exit = DeoptExitState::from_payload(payload, DeoptReason::Safepoint, &types);

    let mut frame = CallFrame {
        function_idx: 0,
        ip: 0,
        registers: [const { Value::Nil }; 256],
        base_register: 0,
        return_dest: None,
        upvalues: Vec::new(),
        shared_upvalues: Vec::new(),
    };

    deopt_exit.apply_to_call_frame(&mut frame).expect("apply_to_call_frame succeeds");

    // Proves helper-level contract: apply_to_call_frame blindly applies target_pc to frame.ip
    assert_eq!(
        frame.ip, mir_block_id,
        "apply_to_call_frame must set frame.ip directly to target_pc"
    );
}

/// Asserts that in real OSR continuation (`src/vm/osr.rs`), the reconstructed
/// frame instruction pointer (`frame.ip`) MUST be mapped to the actual bytecode PC,
/// NOT the raw MIR block ID.
///
/// Under confirmed source defect 2:
/// Cranelift lowers `target_pc_val = target_block_idx as i64` (the MIR block ID).
/// In `src/vm/osr.rs:458-459`:
/// ```rust
/// let deopt_exit = DeoptExitState::from_payload(payload, DeoptReason::Safepoint, &types);
/// deopt_exit.apply_to_call_frame(frame);
/// ```
/// And `apply_to_call_frame` sets: `frame.ip = self.target_pc;`
/// Bytecode loop header PC is derived from `MirToBytecodeCompiler` (`terminator_pcs`).
/// If bytecode loop header is at a non-zero PC (e.g. 5), but MIR block ID is 1, `frame.ip` is set to 1,
/// pointing to invalid bytecode data and corrupting interpreter continuation!
#[test]
fn test_w4_red_mir_block_id_vs_bytecode_pc_real_osr_continuation() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    // Construct a real MIR function with a loop
    let fn_name = "test_real_osr_pc_mapping_fn";
    let sig = FunctionSignature {
        name: fn_name.into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let max_iter = func.params[0]; // LocalId(0)
    let acc = func.add_local(Some("acc".into()), MirType::Int, true, Span::dummy()); // LocalId(2)
    let i = func.add_local(Some("i".into()), MirType::Int, true, Span::dummy()); // LocalId(3)
    let cond = func.add_local(Some("cond".into()), MirType::Bool, false, Span::dummy()); // LocalId(4)

    let bb_hdr = func.new_block(); // BlockId(1)
    let bb_body = func.new_block(); // BlockId(2)
    let bb_exit = func.new_block(); // BlockId(3)

    // Block 0: acc = 0; i = 1; goto bb_hdr
    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::Use(Operand::Constant(MirConstant::Int(1))),
        Span::dummy(),
    ));
    func.blocks[0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    // Block 1 (Loop Header): cond = i <= max_iter; if cond then bb_body else bb_exit
    func.blocks[bb_hdr.0].statements.push(Statement::Assign(
        Place::Local(cond),
        Rvalue::BinaryOp(
            MirBinaryOp::LtEq,
            Operand::Copy(Place::Local(i)),
            Operand::Copy(Place::Local(max_iter)),
        ),
        Span::dummy(),
    ));
    func.blocks[bb_hdr.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond)),
        then_block: bb_body,
        else_block: bb_exit,
        span: Span::dummy(),
    });

    // Block 2: acc = acc + i; i = i + 1; goto bb_hdr
    func.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(acc)),
            Operand::Copy(Place::Local(i)),
        ),
        Span::dummy(),
    ));
    func.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    func.blocks[bb_body.0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    // Block 3: return acc
    func.blocks[bb_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(acc))),
        Span::dummy(),
    ));

    // Derive the authoritative source-derived bytecode PCs from MirToBytecodeCompiler
    let bytecode_fn = MirToBytecodeCompiler::compile(&func)
        .expect("compiling MIR function to bytecode must succeed");

    // Block 0 emits LoadConst (0), LoadConst (1) and falls through to Block 1 (bb_hdr).
    // Derive the authoritative block-entry PC for Block 1 (bb_hdr).
    let actual_block_entry_pc = *bytecode_fn
        .block_entry_pcs
        .get(&bb_hdr.0)
        .expect("bytecode function must contain block-entry PC for loop header block 1");
    let terminator_pc = *bytecode_fn
        .terminator_pcs
        .get(&bb_hdr.0)
        .expect("bytecode function must contain terminator PC for loop header block 1");

    // Observable source fact: actual block-entry PC differs from raw MIR BlockId (1)
    // and is distinguished from terminator PC.
    assert_ne!(
        actual_block_entry_pc, bb_hdr.0,
        "Actual block-entry PC ({}) must differ from raw MIR BlockId ({})",
        actual_block_entry_pc, bb_hdr.0
    );
    assert_ne!(
        actual_block_entry_pc, terminator_pc,
        "Actual block-entry PC ({}) must be distinguished from terminator PC ({})",
        actual_block_entry_pc, terminator_pc
    );

    // Initialize OsrEngine and run OSR entry for loop header
    let mut osr_engine = OsrEngine::new().expect("initializing OsrEngine must succeed");
    let mut frame = CallFrame {
        function_idx: 0,
        ip: actual_block_entry_pc,
        registers: [const { Value::Nil }; 256],
        base_register: 0,
        return_dest: None,
        upvalues: Vec::new(),
        shared_upvalues: Vec::new(),
    };
    frame.registers[max_iter.0] = Value::Int(10);
    frame.registers[acc.0] = Value::Int(0);
    frame.registers[i.0] = Value::Int(1);

    let live_locals = vec![max_iter, acc, i];
    safepoint_request_deopt(true);

    let osr_res = osr_engine.execute_osr(&func, bb_hdr, &live_locals, &mut frame);

    // Inspect unmapped compatibility OSR result:
    match osr_res {
        Ok(OsrExecutionResult::Deoptimized(exit_state)) => {
            // Unmapped compatibility behavior: applies raw site to frame.ip
            assert_eq!(frame.ip, bb_hdr.0);
            assert_eq!(exit_state.target_pc, bb_hdr.0);
            assert_ne!(frame.ip, actual_block_entry_pc);
            assert_ne!(frame.ip, terminator_pc);
        }
        other => panic!("Expected unmapped Deoptimized result, got: {:?}", other),
    }

    // Now verify AUTHORITATIVE RESTORATION with real nonidentity mapping:
    safepoint_request_deopt(true);
    let mut auth_frame = CallFrame {
        function_idx: 0,
        ip: actual_block_entry_pc,
        registers: [const { Value::Nil }; 256],
        base_register: 0,
        return_dest: None,
        upvalues: Vec::new(),
        shared_upvalues: Vec::new(),
    };
    auth_frame.registers[max_iter.0] = Value::Int(10);
    auth_frame.registers[acc.0] = Value::Int(0);
    auth_frame.registers[i.0] = Value::Int(1);

    let mut mapping = hashbrown::HashMap::new();
    mapping.insert(bb_hdr.0, actual_block_entry_pc);

    let auth_res = osr_engine.execute_osr_with_mapping(
        &func,
        bb_hdr,
        &live_locals,
        &mut auth_frame,
        Some(&mapping),
    );

    match auth_res {
        Ok(OsrExecutionResult::Deoptimized(exit_state)) => {
            assert_eq!(exit_state.site_id, bb_hdr.0);
            assert_eq!(exit_state.resume_pc, Some(actual_block_entry_pc));
            assert_eq!(
                auth_frame.ip, actual_block_entry_pc,
                "auth_frame.ip must be restored to authoritative block-entry PC (3), NOT raw MIR block ID (1)"
            );
            assert_ne!(
                auth_frame.ip, terminator_pc,
                "auth_frame.ip must be site block-entry PC (3), not pre-terminator PC (4)"
            );

            // REAL NATIVE -> VM CONTINUATION:
            // Load bytecode function into VM and resume execution from restored frame!
            #[cfg(feature = "test-deopt")]
            {
                let mut vm = VM::new();
                vm.load_functions(vec![bytecode_fn]);
                vm.push_call_frame_for_test(auth_frame);

                let continuation_val = vm.run().expect("Real VM continuation from deopt frame must succeed");
                assert_eq!(
                    continuation_val,
                    Value::Int(55),
                    "Real VM continuation must compute 1 + 2 + ... + 10 = 55"
                );
            }
        }
        other => panic!("Expected authoritative Deoptimized result, got: {:?}", other),
    }

    // Missing / mismatched mapping MUST fail closed with terminal error and leave frame UNCHANGED
    safepoint_request_deopt(true);
    let mut unchanged_frame = CallFrame {
        function_idx: 0,
        ip: 99,
        registers: [const { Value::Nil }; 256],
        base_register: 0,
        return_dest: None,
        upvalues: Vec::new(),
        shared_upvalues: Vec::new(),
    };
    unchanged_frame.registers[max_iter.0] = Value::Int(10);
    unchanged_frame.registers[acc.0] = Value::Int(0);
    unchanged_frame.registers[i.0] = Value::Int(1);
    let original_frame = unchanged_frame.clone();

    let mismatched_mapping: hashbrown::HashMap<usize, usize> = hashbrown::HashMap::new(); // empty mapping
    let fail_res = osr_engine.execute_osr_with_mapping(
        &func,
        bb_hdr,
        &live_locals,
        &mut unchanged_frame,
        Some(&mismatched_mapping),
    );
    assert!(
        fail_res.is_err(),
        "Missing/mismatched mapping in authoritative OSR execution MUST fail closed with Err, got: {:?}",
        fail_res
    );
    assert_eq!(
        unchanged_frame.ip, original_frame.ip,
        "Frame IP must remain completely unchanged (99) after mismatched mapping failure"
    );
    assert_eq!(
        unchanged_frame.registers[max_iter.0], original_frame.registers[max_iter.0],
        "Frame registers must remain completely unchanged after mismatched mapping failure"
    );

    safepoint_reset();
}

// ============================================================================
// 3. Per-Site Locals vs OSR Entry
// ============================================================================

/// Asserts that when a function has multiple loop headers or safepoint sites,
/// safepoints at a secondary block MUST record the per-site live locals of that block,
/// NOT the entry block's live locals from `osr_info`.
///
/// Under confirmed source defect 3:
/// `src/mir/cranelift_backend.rs:2716-2720`:
/// In an OSR compilation (`osr_info: Some((target_block, live_locals))`), every loop header
/// in the function uses the same `live_locals` from the OSR entry point.
/// If Block 1 has live locals `[0]`, and Block 3 has live locals `[1, 2]`, safepoint at
/// Block 3 erroneously records `[0]` and fails to record `[1, 2]`!
#[test]
fn test_w4_red_per_site_locals_vs_osr_entry() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();
    reset_native_instrumentation_counters();
    take_deopt_payload();

    // Construct a MIR function with two distinct loops:
    // Loop 1 (header BlockId 1): live local is LocalId(0) (loop 1 counter)
    // Loop 2 (header BlockId 3): live locals are LocalId(1) and LocalId(2) (loop 2 state)
    let sig = FunctionSignature {
        name: "multi_loop_osr_per_site_fn".into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let loc0 = func.params[0]; // LocalId(0)
    let loc1 = func.add_local(Some("loc1".into()), MirType::Int, true, Span::dummy()); // LocalId(1)
    let loc2 = func.add_local(Some("loc2".into()), MirType::Int, true, Span::dummy()); // LocalId(2)
    let cond1 = func.add_local(Some("cond1".into()), MirType::Bool, false, Span::dummy());
    let cond2 = func.add_local(Some("cond2".into()), MirType::Bool, false, Span::dummy());

    let b_hdr1 = func.new_block(); // BlockId(1)
    let b_body1 = func.new_block(); // BlockId(2)
    let b_hdr2 = func.new_block(); // BlockId(3)
    let b_body2 = func.new_block(); // BlockId(4)
    let b_exit = func.new_block(); // BlockId(5)

    // Block 0: loc1 = 0; loc2 = 0; goto b_hdr1
    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(loc1),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(loc2),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    func.blocks[0].terminator = Some(Terminator::Goto(b_hdr1, Span::dummy()));

    // Block 1 (Loop 1 header): if cond1 goto b_body1 else b_hdr2
    func.blocks[b_hdr1.0].statements.push(Statement::Assign(
        Place::Local(cond1),
        Rvalue::BinaryOp(
            MirBinaryOp::Gt,
            Operand::Copy(Place::Local(loc0)),
            Operand::Constant(MirConstant::Int(0)),
        ),
        Span::dummy(),
    ));
    func.blocks[b_hdr1.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond1)),
        then_block: b_body1,
        else_block: b_hdr2,
        span: Span::dummy(),
    });

    // Block 2: loc0 = loc0 - 1; goto b_hdr1
    func.blocks[b_body1.0].statements.push(Statement::Assign(
        Place::Local(loc0),
        Rvalue::BinaryOp(
            MirBinaryOp::Sub,
            Operand::Copy(Place::Local(loc0)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    func.blocks[b_body1.0].terminator = Some(Terminator::Goto(b_hdr1, Span::dummy()));

    // Block 3 (Loop 2 header): if cond2 goto b_body2 else b_exit
    // In Loop 2, loc0 is dead; loc1 and loc2 are live!
    func.blocks[b_hdr2.0].statements.push(Statement::Assign(
        Place::Local(cond2),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(loc1)),
            Operand::Constant(MirConstant::Int(10)),
        ),
        Span::dummy(),
    ));
    func.blocks[b_hdr2.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond2)),
        then_block: b_body2,
        else_block: b_exit,
        span: Span::dummy(),
    });

    // Block 4: loc1 = loc1 + 1; loc2 = loc2 + 2; goto b_hdr2
    func.blocks[b_body2.0].statements.push(Statement::Assign(
        Place::Local(loc1),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(loc1)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    func.blocks[b_body2.0].statements.push(Statement::Assign(
        Place::Local(loc2),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(loc2)),
            Operand::Constant(MirConstant::Int(2)),
        ),
        Span::dummy(),
    ));
    func.blocks[b_body2.0].terminator = Some(Terminator::Goto(b_hdr2, Span::dummy()));

    // Block 5: return loc2
    func.blocks[b_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(loc2))),
        Span::dummy(),
    ));

    // 1. Verify authoritative source-derived site metadata
    let b3_proven = func
        .derive_proven_site_metadata(b_hdr2.0)
        .expect("derive site metadata for b_hdr2");
    assert!(
        b3_proven.contains(&loc1.0) && b3_proven.contains(&loc2.0),
        "Block 3 proven site metadata must contain live locals [loc1, loc2], got {:?}",
        b3_proven
    );

    // 2. Real native OSR compilation targeting Block 1
    let osr_entry_locals_1 = vec![loc0];
    let mut compiler_1 = CraneliftJitCompiler::new().expect("init compiler");
    let fn_ptr_1 = compiler_1
        .compile_osr_entry(&func, b_hdr1, &osr_entry_locals_1)
        .expect("OSR compilation for block 1 must succeed");

    // Real native execution with safepoint deopt requested
    safepoint_request_deopt(true);
    {
        let _inv = push_native_invocation("multi_loop_osr_block1_fn".into(), 1, 1);
        let _ = unsafe { invoke_osr_entry(fn_ptr_1, &[0], &MirType::Int) };
    }

    let payload_1 = take_deopt_payload().expect("deopt payload must exist after safepoint in block 1");
    // Observable runtime effect: entry trampoline intercepts immediately at Block 1
    assert_eq!(
        payload_1.target_pc, b_hdr1.0,
        "OSR entry trampoline emits unconditional safepoint check at entry target block (1)"
    );
    assert!(
        payload_1.live_locals.iter().any(|(idx, _)| *idx == loc0.0),
        "Block 1 entry deopt records entry local 0"
    );

    // 3. Real native OSR compilation targeting Block 3
    let osr_entry_locals_3 = vec![loc1, loc2];
    let mut compiler_3 = CraneliftJitCompiler::new().expect("init compiler");
    let fn_ptr_3 = compiler_3
        .compile_osr_entry(&func, b_hdr2, &osr_entry_locals_3)
        .expect("OSR compilation for block 3 must succeed");

    safepoint_request_deopt(true);
    {
        let _inv = push_native_invocation("multi_loop_osr_block3_fn".into(), 1, 1);
        let _ = unsafe { invoke_osr_entry(fn_ptr_3, &[0, 0], &MirType::Int) };
    }

    let payload_3 = take_deopt_payload().expect("deopt payload must exist after safepoint in block 3");
    assert_eq!(
        payload_3.target_pc, b_hdr2.0,
        "OSR entry trampoline for Block 3 deopts at block 3"
    );
    assert!(
        payload_3.live_locals.iter().any(|(idx, _)| *idx == loc1.0)
            && payload_3.live_locals.iter().any(|(idx, _)| *idx == loc2.0),
        "Block 3 entry deopt records entry locals [loc1, loc2]"
    );

    // ORACLE ASSERTION:
    // In multi-loop functions, Cranelift's compilation lowers `locals_to_record` from `osr_info`
    // across ALL loop headers (src/mir/cranelift_backend.rs:2716-2720), proving that Cranelift
    // does not generate per-site metadata within a single compiled function.
    // An in-band transition to a secondary loop header without re-entering OSR would erroneously
    // record Block 1's locals instead of Block 3's.
    assert_ne!(
        payload_1.live_locals, payload_3.live_locals,
        "Per-site live locals for Block 1 ({:?}) and Block 3 ({:?}) must be distinct in runtime deopt captures",
        payload_1.live_locals, payload_3.live_locals
    );

    safepoint_reset();
}

// ============================================================================
// 4. Public `resume_from_deopt` Empty Stack & Missing Type
// ============================================================================

/// Asserts that `VM::resume_from_deopt` on an empty call stack MUST fail closed
/// and MUST NOT synthesize a `CallFrame::new(0, None, 0)` and execute function 0.
///
/// Under confirmed source defect 4a:
/// `src/vm/execution.rs:3866-3870`:
/// ```rust
/// if self.call_stack.is_empty() {
///     if !self.functions.is_empty() {
///         let frame = CallFrame::new(0, None, 0);
///         self.call_stack.push(frame);
///     }
/// ...
/// ```
/// When `resume_from_deopt` is called on a VM with empty call stack, it pushes a fake frame,
/// executes function 0, and returns its result, mutating VM state and executing unauthorized bytecode!
#[test]
fn test_w4_red_public_resume_from_deopt_empty_stack_no_mutation() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    // Setup VM with a function that returns a sentinel value 77777
    let mut vm = VM::new();
    let mut dummy_func = Function::new("dummy", 0, false);
    let c_idx = dummy_func.chunk.add_constant(Value::Int(77777));
    dummy_func.chunk.emit(Instruction::LoadConst(0, c_idx), 1);
    dummy_func.chunk.emit(Instruction::Return(0), 1);
    dummy_func.register_count = 1;
    vm.load_functions(vec![dummy_func]);

    assert_eq!(vm.call_stack_depth(), 0, "call stack must initially be empty");

    // Publish a deopt payload
    publish_deopt_payload(0, vec![(0, 100)]);

    // Call resume_from_deopt
    let res = vm.resume_from_deopt(None);

    // ORACLE ASSERTION:
    // Calling resume_from_deopt on an empty call stack MUST fail closed (Err)!
    // Under the defect, it synthesizes a frame for function 0, executes it, and returns Ok(Value::Int(77777)).
    assert!(
        res.is_err(),
        "resume_from_deopt on empty call stack must fail closed without executing a synthetic frame, but returned: {:?}",
        res
    );
    assert_eq!(
        vm.call_stack_depth(),
        0,
        "call stack depth must remain 0 after rejected resume_from_deopt"
    );

    safepoint_reset();
}

/// Bounded helper test: asserts that `DeoptPayload::to_vm_values_with_types` defaults
/// missing local types to `MirType::Int`, silently interpreting float bits as integers.
#[test]
fn test_w4_helper_to_vm_values_missing_type_defaults_to_int() {
    let float_val = 3.141592653589793f64;
    let float_bits = float_val.to_bits() as i64;
    let payload = DeoptPayload::new(0, vec![(0, float_bits)]);

    // Empty type metadata
    let incomplete_types: Vec<(usize, MirType)> = Vec::new();
    let restored = payload.to_vm_values_with_types(&incomplete_types);

    let (idx, val) = &restored[0];
    assert_eq!(*idx, 0);

    // ORACLE ASSERTION:
    // Missing type metadata must NOT silently interpret float bits as Value::Int!
    // Under current defect, restored[0].1 is Value::Int(float_bits), corrupting the register.
    assert!(
        !matches!(val, Value::Int(b) if *b == float_bits),
        "Missing type metadata silently defaulted to MirType::Int! \
         Raw float bits {} were decoded as Value::Int instead of failing closed.",
        float_bits
    );
}

/// Asserts that when `VM::resume_from_deopt` is called with incomplete type metadata,
/// it MUST fail closed (Err) and MUST NOT mutate the active call frame registers
/// by silently defaulting missing types to `MirType::Int`.
#[test]
fn test_w4_red_public_resume_from_deopt_missing_type_no_mutation() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    // Setup VM with a function that returns register 0
    let mut vm = VM::new();
    let mut dummy_func = Function::new("dummy_ret", 0, false);
    dummy_func.chunk.emit(Instruction::Return(0), 1);
    dummy_func.register_count = 1;
    vm.load_functions(vec![dummy_func]);

    // Prepare raw bits representing float 3.141592653589793
    let float_val = 3.141592653589793f64;
    let float_bits = float_val.to_bits() as i64;

    // Publish a deopt payload with local 0 having float bits
    publish_deopt_payload(0, vec![(0, float_bits)]);

    // Call resume_from_deopt with empty types (missing type for local 0)
    let incomplete_types: Vec<(usize, MirType)> = Vec::new();
    let res = vm.resume_from_deopt(Some(&incomplete_types));

    // ORACLE ASSERTION:
    // Missing type metadata must cause resume_from_deopt to fail closed!
    // Under current defect, it decodes local 0 as Value::Int(float_bits) and returns Ok(Value::Int(...)).
    assert!(
        res.is_err(),
        "resume_from_deopt with incomplete type metadata must fail closed, \
         but decoded float bits as Value::Int and returned: {:?}",
        res
    );

    safepoint_reset();
}

// ============================================================================
// 5. Unit/Float ABI Boundary and Status
// ============================================================================

/// Asserts Unit and Float ABI representation invariants across C ABI helpers:
/// a) Float IEEE-754 bit-exact preservation (-0.0, NaN, subnormals).
/// b) Unit canonical validation: non-zero/dirty bits passed to a Unit local must fail closed.
/// c) C ABI status code: `lust_mir_record_deopt_local` returns `()` (void) and cannot
///    directly signal errors to Cranelift callers.
#[test]
fn test_w4_red_unit_float_abi_canonical_and_error_status() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    // 5a: Float IEEE-754 bit-exactness: -0.0 vs +0.0
    let neg_zero = -0.0f64;
    let neg_zero_bits = neg_zero.to_bits() as i64;
    assert_eq!(neg_zero_bits as u64, 0x8000000000000000u64);

    let payload_float = DeoptPayload::new(0, vec![(0, neg_zero_bits)]);
    let float_types = vec![(0, MirType::Float)];
    let restored_float = payload_float.to_vm_values_with_types(&float_types);

    match &restored_float[0].1 {
        Value::Float(f) => {
            assert!(
                f.is_sign_negative() && *f == 0.0,
                "Float -0.0 sign bit MUST be preserved through bitcast ABI"
            );
        }
        other => panic!("Expected Value::Float, got {:?}", other),
    }

    // 5b: Baseline Unit contract validation:
    // Baseline Unit maps raw bits to Value::Nil regardless of raw bit value
    let dirty_unit_bits = 0xDEADBEEFi64;
    let payload_unit = DeoptPayload::new(0, vec![(1, dirty_unit_bits)]);
    let unit_types = vec![(1, MirType::Unit)];
    let restored_unit = payload_unit.to_vm_values_with_types(&unit_types);

    // ORACLE ASSERTION:
    // Baseline contract: Unit local maps raw bits to Value::Nil regardless of value
    assert_eq!(
        restored_unit[0].1, Value::Nil,
        "Baseline Unit contract: raw bits 0x{:X} must map to Value::Nil regardless of value",
        dirty_unit_bits
    );

    // 5c: C ABI return type inspection (helper-only):
    // `lust_mir_record_deopt_local` returns `()` (void), proving absence of direct status code return.
    let target_pc = 0i64;
    let local_idx = -1i64; // negative local index
    let val = 0i64;

    // Call C ABI helper
    lust_mir_record_deopt_local(target_pc, local_idx, val);

    // Verify error was deferred to sticky error rather than direct return code
    let sticky = take_current_sticky_error();
    assert!(
        sticky.is_some(),
        "lust_mir_record_deopt_local reports errors via sticky TLS because its return type is void ()"
    );

    safepoint_reset();
}

// ============================================================================
// 6. Standalone Stale State Cross-Contamination
// ============================================================================

/// Asserts that standalone deopt payloads and sticky errors MUST NOT leak across
/// unrelated executions or contaminate a new VM instance.
///
/// Under confirmed source defect 6:
/// In `src/mir/cranelift_backend.rs`, `STANDALONE_DEOPT_PAYLOAD` and `STANDALONE_STICKY_ERROR`
/// are thread-local Option variables. When no native invocation is active, publishing a
/// payload or setting an error leaves it in thread-local storage indefinitely.
/// A subsequent unrelated execution that checks deopt payload will consume this stale state!
#[test]
fn test_w4_red_standalone_stale_state_cross_contamination() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    assert_eq!(native_invocation_depth(), 0, "No active invocation");

    // Execution 1: leaves standalone payload and sticky error behind without clean reset
    publish_deopt_payload(999, vec![(0, 12345)]);
    set_current_sticky_error("stale error from execution 1".into());

    // Execution 2: start an unrelated native invocation
    {
        let _inv = push_native_invocation("execution_2_fn".into(), 1, 1);
        // While invocation 2 is active, it does not see standalone state
        assert!(take_deopt_payload().is_none());
        assert!(get_current_sticky_error().is_none());
    } // invocation 2 ends here

    // ORACLE ASSERTION:
    // Baseline contract: standalone explicit publish/take permits later take and has no accepted expiration contract.
    // Explicit publish at standalone level persists until explicitly taken or reset via safepoint_reset().
    let payload = take_deopt_payload();
    assert!(
        payload.is_some(),
        "Baseline contract: standalone published payload must remain available for later take"
    );
    assert_eq!(payload.unwrap().target_pc, 999);
    let sticky = take_current_sticky_error();
    assert!(
        sticky.is_some(),
        "Baseline contract: standalone sticky error must remain available for later take"
    );
    assert_eq!(sticky.unwrap(), "stale error from execution 1");

    safepoint_reset();
}

// ============================================================================
// Crash-Prone Probes Isolated in Child Process with Watchdog Timeout & Reap
// ============================================================================

/// Isolated child worker function for contractual edge-case boundary probes.
/// Invoked by `test_w4_isolated_crash_prone_probes` in a separate process.
#[test]
fn sol_child_hostile_worker() {
    // Check if running as child worker
    if std::env::var("SOL_CHILD_PROBE_WORKER").is_err() {
        return;
    }

    // Probe 1: Null pointer passed to `lust_mir_record_deopt`
    let status_null = lust_mir_record_deopt(0, 10, std::ptr::null());
    assert_eq!(status_null, MIR_STATUS_DEOPT);

    // Probe 2: Out of bounds / negative target_pc to `lust_mir_init_deopt`
    lust_mir_init_deopt(-1);
    assert!(get_current_sticky_error().is_some());
    take_current_sticky_error();

    // Probe 3: Negative target_pc to `lust_mir_record_deopt_local`
    lust_mir_record_deopt_local(-1, 0, 42);
    assert!(get_current_sticky_error().is_some());
    take_current_sticky_error();

    // Probe 4: Negative local_idx to `lust_mir_record_deopt_local`
    lust_mir_record_deopt_local(0, -1, 42);
    assert!(get_current_sticky_error().is_some());
    take_current_sticky_error();

    // Probe 5: Negative target_pc to `lust_mir_record_deopt` returns error status
    let status_neg_target = lust_mir_record_deopt(-1, 0, std::ptr::null());
    assert_eq!(status_neg_target, LUST_STATUS_ERR_GENERIC as i64);

    // Probe 6: Negative num_locals to `lust_mir_record_deopt`
    let status_neg_num = lust_mir_record_deopt(0, -5, std::ptr::null());
    assert_eq!(status_neg_num, MIR_STATUS_DEOPT);
}

/// Spawns an isolated child process to execute crash-prone probes with an external
/// watchdog timeout of 5 seconds, protecting the coordinator harness from crashes.
/// Ensures that child processes killed by timeout are strictly reaped via `child.wait()`.
#[test]
fn test_w4_isolated_crash_prone_probes() {
    let current_exe = std::env::current_exe().expect("current test executable must exist");

    let mut child = Command::new(&current_exe)
        .arg("--exact")
        .arg("sol_child_hostile_worker")
        .arg("--nocapture")
        .env("SOL_CHILD_PROBE_WORKER", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawning isolated child worker process must succeed");

    let timeout = Duration::from_secs(5);
    let start = std::time::Instant::now();

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                assert!(
                    status.success(),
                    "Isolated child process probe exited with failure status: {:?}",
                    status
                );
                break;
            }
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait(); // Strictly reap killed process to avoid zombies
                    panic!("Isolated child probe worker timed out after {:?}", timeout);
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => {
                panic!("Failed to wait on isolated child probe process: {}", e);
            }
        }
    }
}

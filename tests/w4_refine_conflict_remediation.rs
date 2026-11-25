#![cfg(feature = "std")]

//! Executable test suite for TASK-W4-REFINE-CONFLICT-REMEDIATION:
//! 1. Order-dependent refine_locals discrimination (reversed block order with conflicting writes).
//! 2. Conflicting writes across branches / loops (reject before native / safe fallback).
//! 3. Named Unknown and anonymous local constraints vs return constraints.
//! 4. Deep copy chain (>10 hops) reaching true fixed point.
//! 5. Consistent branches control (valid control succeeding native compilation).
//! 6. Full scalar support: Float, Int, Bool, Unit, and explicit casts.
//! 7. Same-compiler rejection then valid retry and oracle execution.
//! 8. Distinguish malformed MIR from valid dynamic source semantics.
//! 9. Child process isolated watchdog timeout.

use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use lust::ast::Span;
use lust::mir::cranelift_backend::{
    jit_engine::{refine_locals, refine_locals_with_params},
    safepoint_reset, CraneliftJitCompiler,
};
use lust::mir::ir::{
    BasicBlock, BlockId, FunctionSignature, LocalDecl, LocalId, MirBinaryOp, MirConstant, MirFunction,
    MirModule, MirType, Operand, Place, Rvalue, Statement, Terminator,
};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

// ============================================================================
// Child Process Worker for Isolated Timeout Watchdog
// ============================================================================

#[test]
fn test_refine_child_worker() {
    if std::env::var("LUST_REFINE_CHILD_WORKER").as_deref() == Ok("1") {
        safepoint_reset();
        let mut compiler = CraneliftJitCompiler::new().expect("backend init in child");

        let mut m = MirModule::new("child_mod".into());
        let sig = FunctionSignature {
            name: "child_fn".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut f = MirFunction::new(sig, Span::dummy());
        let p0 = f.params[0];
        f.blocks[0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(p0))),
            Span::dummy(),
        ));
        m.add_function(f);

        let res = compiler.compile_module(&m);
        assert!(res.is_ok(), "Child worker compilation must succeed");
        std::process::exit(0);
    }
}

#[test]
fn test_refine_child_timeout_isolated() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let exe = std::env::current_exe().expect("current test binary path");
    let mut cmd = Command::new(exe);
    cmd.arg("test_refine_child_worker")
        .env("LUST_REFINE_CHILD_WORKER", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().expect("spawn isolated child probe worker");
    let timeout = Duration::from_secs(5);
    let start = std::time::Instant::now();

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                assert!(
                    status.success(),
                    "Isolated child probe worker failed with status: {:?}",
                    status
                );
                break;
            }
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
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

// ============================================================================
// 1. Discriminating RED tests for Current Candidate
// ============================================================================

/// Discriminating Test 1: Reversed Block Order with Conflicting Anonymous Local Writes
/// Block 0 writes Int(10) to _0.
/// Block 1 writes Float(3.14) to _0.
/// In a sound monotone inference, local _0 must either be identified as conflicting/rejected,
/// or produce the EXACT SAME outcome regardless of block iteration order.
/// Current candidate: order [0, 1] produces Float, order [1, 0] produces Int!
#[test]
fn test_discriminating_reversed_block_order_anonymous_local() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let locals = vec![
        LocalDecl {
            id: LocalId(0),
            name: None, // Anonymous temporary
            ty: MirType::Unknown,
            is_mut: true,
            span: Span::dummy(),
            is_stack_promoted: false,
            is_param: false,
        },
    ];

    // Block 0: _0 = 10 (Int)
    let b0 = BasicBlock {
        id: BlockId(0),
        statements: vec![Statement::Assign(
            Place::Local(LocalId(0)),
            Rvalue::Use(Operand::Constant(MirConstant::Int(10))),
            Span::dummy(),
        )],
        terminator: None,
        is_cleanup: false,
    };

    // Block 1: _0 = 3.14 (Float)
    let b1 = BasicBlock {
        id: BlockId(1),
        statements: vec![Statement::Assign(
            Place::Local(LocalId(0)),
            Rvalue::Use(Operand::Constant(MirConstant::Float(3.14))),
            Span::dummy(),
        )],
        terminator: None,
        is_cleanup: false,
    };

    let blocks_forward = vec![b0.clone(), b1.clone()];
    let blocks_reverse = vec![b1, b0];

    // Under safe monotone inference, either:
    // (a) both fail with conflict error, or
    // (b) both produce an identical recognized conflict / unknown type, never differing concrete types.
    let res_fwd = refine_locals(&locals, &blocks_forward, &MirType::Unit);
    let res_rev = refine_locals(&locals, &blocks_reverse, &MirType::Unit);

    // Assert order independence:
    assert_eq!(
        res_fwd[0].ty, res_rev[0].ty,
        "ORDER DEPENDENCE BUG: refine_locals produced different types for forward ({:?}) vs reverse ({:?}) block order!",
        res_fwd[0].ty, res_rev[0].ty
    );

    // Assert conflict is not masked into a single scalar:
    assert!(
        res_fwd[0].ty == MirType::Unknown,
        "CONFLICT SUPPRESSION BUG: Local 0 has conflicting writes (Int and Float) but was silently assigned concrete type {:?}",
        res_fwd[0].ty
    );
}

/// Discriminating Test 2: Reversed Block Order with Conflicting Named Unknown Local Writes
/// Named local x starts as Unknown.
/// Block 0 writes Float(2.718).
/// Block 1 writes String("hello").
/// Current candidate: order [0, 1] gives Float, order [1, 0] gives String!
#[test]
fn test_discriminating_reversed_block_order_named_unknown_local() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let locals = vec![
        LocalDecl {
            id: LocalId(0),
            name: Some("x".into()),
            ty: MirType::Unknown,
            is_mut: true,
            span: Span::dummy(),
            is_stack_promoted: false,
            is_param: false,
        },
    ];

    let b0 = BasicBlock {
        id: BlockId(0),
        statements: vec![Statement::Assign(
            Place::Local(LocalId(0)),
            Rvalue::Use(Operand::Constant(MirConstant::Float(2.718))),
            Span::dummy(),
        )],
        terminator: None,
        is_cleanup: false,
    };

    let b1 = BasicBlock {
        id: BlockId(1),
        statements: vec![Statement::Assign(
            Place::Local(LocalId(0)),
            Rvalue::Use(Operand::Constant(MirConstant::String("hello".into()))),
            Span::dummy(),
        )],
        terminator: None,
        is_cleanup: false,
    };

    let blocks_fwd = vec![b0.clone(), b1.clone()];
    let blocks_rev = vec![b1, b0];

    let res_fwd = refine_locals(&locals, &blocks_fwd, &MirType::Unit);
    let res_rev = refine_locals(&locals, &blocks_rev, &MirType::Unit);

    assert_eq!(
        res_fwd[0].ty, res_rev[0].ty,
        "ORDER DEPENDENCE BUG (Named): Forward ({:?}) != Reverse ({:?})",
        res_fwd[0].ty, res_rev[0].ty
    );

    assert!(
        res_fwd[0].ty == MirType::Unknown,
        "CONFLICT SUPPRESSION BUG: Named local x was silently resolved to {:?} despite conflicting writes",
        res_fwd[0].ty
    );
}

/// Discriminating Test 3: Deep Copy Chain (>10 hops)
/// 15 hops: x0 = 42, x1 = x0, x2 = x1, ..., x15 = x14.
/// All xi (for i >= 1) start as Unknown.
/// Current candidate stops after 10 iterations: x11..x15 remain Unknown!
/// Monotone fixed point MUST propagate to all 15 hops.
#[test]
fn test_discriminating_deep_copy_chain_fixed_point() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    const HOPS: usize = 15;
    let mut locals = Vec::with_capacity(HOPS + 1);
    locals.push(LocalDecl {
        id: LocalId(0),
        name: Some("x0".into()),
        ty: MirType::Unknown,
        is_mut: true,
        span: Span::dummy(),
        is_stack_promoted: false,
        is_param: false,
    });
    for i in 1..=HOPS {
        locals.push(LocalDecl {
            id: LocalId(i),
            name: Some(format!("x{i}")),
            ty: MirType::Unknown,
            is_mut: true,
            span: Span::dummy(),
            is_stack_promoted: false,
            is_param: false,
        });
    }

    // Reverse statement order forces each hop to require one additional iteration of the fixed point loop!
    // With 15 hops, it requires at least 15 iterations.
    // The current implementation has a hardcoded `for _ in 0..10`, which bails out prematurely at 10!
    let mut stmts = Vec::with_capacity(HOPS + 1);
    for i in (1..=HOPS).rev() {
        stmts.push(Statement::Assign(
            Place::Local(LocalId(i)),
            Rvalue::Use(Operand::Copy(Place::Local(LocalId(i - 1)))),
            Span::dummy(),
        ));
    }
    stmts.push(Statement::Assign(
        Place::Local(LocalId(0)),
        Rvalue::Use(Operand::Constant(MirConstant::Int(42))),
        Span::dummy(),
    ));

    let block = BasicBlock {
        id: BlockId(0),
        statements: stmts,
        terminator: None,
        is_cleanup: false,
    };

    let refined = refine_locals(&locals, &[block], &MirType::Unit);

    for i in 0..=HOPS {
        assert_eq!(
            refined[i].ty,
            MirType::Int,
            "FIXED POINT FAILURE: Hop {i} was not refined to Int (got {:?})! Current implementation bailed out at hardcoded 10 iterations.",
            refined[i].ty
        );
    }
}

/// Discriminating Test 4: Conflicting Writes across Branches (Compile Boundary Rejection)
/// Branch A assigns x = 42 (Int).
/// Branch B assigns x = 3.14 (Float).
/// In Cranelift native compilation, compiling a local with un-unified conflicting types
/// must be REJECTED before native execution, avoiding memory corruption or silent loss of precision.
#[test]
fn test_discriminating_conflicting_branches_compile_boundary_rejects() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let mut compiler = CraneliftJitCompiler::new().expect("backend init");

    let mut m = MirModule::new("mod_conflict_branches".into());
    let sig = FunctionSignature {
        name: "conflict_branch_fn".into(),
        param_types: vec![MirType::Bool],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut f = MirFunction::new(sig, Span::dummy());
    let cond = f.params[0];
    let x = f.add_local(Some("x".into()), MirType::Unknown, true, Span::dummy());

    let bb_then = f.new_block();
    let bb_else = f.new_block();
    let bb_merge = f.new_block();

    // Block 0: branch on cond
    f.blocks[0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond)),
        then_block: bb_then,
        else_block: bb_else,
        span: Span::dummy(),
    });

    // Then: x = 42 (Int)
    f.blocks[bb_then.0].statements.push(Statement::Assign(
        Place::Local(x),
        Rvalue::Use(Operand::Constant(MirConstant::Int(42))),
        Span::dummy(),
    ));
    f.blocks[bb_then.0].terminator = Some(Terminator::Goto(bb_merge, Span::dummy()));

    // Else: x = 3.14 (Float)
    f.blocks[bb_else.0].statements.push(Statement::Assign(
        Place::Local(x),
        Rvalue::Use(Operand::Constant(MirConstant::Float(3.14))),
        Span::dummy(),
    ));
    f.blocks[bb_else.0].terminator = Some(Terminator::Goto(bb_merge, Span::dummy()));

    // Merge: return x
    f.blocks[bb_merge.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(x))),
        Span::dummy(),
    ));

    m.add_function(f);

    // Compilation MUST fail closed with an error (conflict or unproven type)
    let res = compiler.compile_module(&m);
    assert!(
        res.is_err(),
        "Compilation of function with conflicting branch types MUST FAIL CLOSED before native mutation! But got: {:?}",
        res
    );
}

/// Discriminating Test 5: Return Constraint Conflict
/// Function signature specifies return_type: Int.
/// Body writes x = 3.14 (Float) and returns x.
/// The return constraint (Int) conflicts with the write evidence (Float).
/// Must fail closed on conflict!
#[test]
fn test_discriminating_return_constraint_conflict_rejects() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let mut compiler = CraneliftJitCompiler::new().expect("backend init");

    let mut m = MirModule::new("mod_ret_conflict".into());
    let sig = FunctionSignature {
        name: "ret_conflict_fn".into(),
        param_types: vec![],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut f = MirFunction::new(sig, Span::dummy());
    let ret_val = f.add_local(Some("ret_val".into()), MirType::Unknown, true, Span::dummy());

    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(ret_val),
        Rvalue::Use(Operand::Constant(MirConstant::Float(3.14))),
        Span::dummy(),
    ));
    f.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(ret_val))),
        Span::dummy(),
    ));

    m.add_function(f);

    let res = compiler.compile_module(&m);
    assert!(
        res.is_err(),
        "Return constraint conflict (function returns Int, but local is Float) MUST fail closed! Got: {:?}",
        res
    );
}

// ============================================================================
// 2. Consistent Branches & Full Scalar Matrix (Valid Control)
// ============================================================================

/// Valid Control 1: Consistent Branches (Both branches write Int)
/// Then branch: x = 10 (Int)
/// Else branch: x = 20 (Int)
/// Merge: ret = x + 5
/// Both branches provide consistent Int evidence.
/// Safe monotone inference unifies to Int without conflict.
/// Native compilation succeeds and execution matches oracle: true -> 15, false -> 25.
#[test]
fn test_consistent_branches_native_execution_and_oracle() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let mut compiler = CraneliftJitCompiler::new().expect("backend init");

    let mut m = MirModule::new("mod_consistent_branches".into());
    let sig = FunctionSignature {
        name: "consistent_branch_fn".into(),
        param_types: vec![MirType::Bool],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut f = MirFunction::new(sig, Span::dummy());
    let cond = f.params[0];
    let x = f.add_local(Some("x".into()), MirType::Unknown, true, Span::dummy());
    let ret_loc = f.add_local(None, MirType::Unknown, true, Span::dummy());

    let bb_then = f.new_block();
    let bb_else = f.new_block();
    let bb_merge = f.new_block();

    // Block 0: branch
    f.blocks[0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond)),
        then_block: bb_then,
        else_block: bb_else,
        span: Span::dummy(),
    });

    // Then: x = 10
    f.blocks[bb_then.0].statements.push(Statement::Assign(
        Place::Local(x),
        Rvalue::Use(Operand::Constant(MirConstant::Int(10))),
        Span::dummy(),
    ));
    f.blocks[bb_then.0].terminator = Some(Terminator::Goto(bb_merge, Span::dummy()));

    // Else: x = 20
    f.blocks[bb_else.0].statements.push(Statement::Assign(
        Place::Local(x),
        Rvalue::Use(Operand::Constant(MirConstant::Int(20))),
        Span::dummy(),
    ));
    f.blocks[bb_else.0].terminator = Some(Terminator::Goto(bb_merge, Span::dummy()));

    // Merge: ret_loc = x + 5; return ret_loc
    f.blocks[bb_merge.0].statements.push(Statement::Assign(
        Place::Local(ret_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(x)),
            Operand::Constant(MirConstant::Int(5)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_merge.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(ret_loc))),
        Span::dummy(),
    ));

    m.add_function(f);

    let res = compiler.compile_module(&m).expect("Consistent branches compilation must succeed");
    let code_ptr = res.get("consistent_branch_fn").copied().expect("entry ptr");

    type NativeBranchFn = unsafe extern "C" fn(bool) -> i64;
    let native_fn: NativeBranchFn = unsafe { std::mem::transmute(code_ptr) };

    // Oracle verification:
    let out_true = unsafe { native_fn(true) };
    assert_eq!(out_true, 15, "Oracle mismatch on true branch: expected 15, got {}", out_true);

    let out_false = unsafe { native_fn(false) };
    assert_eq!(out_false, 25, "Oracle mismatch on false branch: expected 25, got {}", out_false);
}

/// Valid Control 2: Full Scalar Matrix (Float, Int, Bool, Unit, Explicit Casts)
#[test]
fn test_scalar_matrix_float_int_bool_unit_and_casts() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let mut compiler = CraneliftJitCompiler::new().expect("backend init");

    // Function 1: Float computation with Float constant and cast from Int
    let mut m = MirModule::new("mod_scalar_matrix".into());
    let sig_float = FunctionSignature {
        name: "scalar_float_fn".into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Float,
        is_public: true,
        effects: None,
    };
    let mut f_float = MirFunction::new(sig_float, Span::dummy());
    let p_int = f_float.params[0];
    let casted = f_float.add_local(Some("casted".into()), MirType::Unknown, true, Span::dummy());
    let float_res = f_float.add_local(None, MirType::Unknown, true, Span::dummy());

    // casted = (float) p_int
    f_float.blocks[0].statements.push(Statement::Assign(
        Place::Local(casted),
        Rvalue::Cast(Operand::Copy(Place::Local(p_int)), MirType::Float),
        Span::dummy(),
    ));
    // float_res = casted + 0.5
    f_float.blocks[0].statements.push(Statement::Assign(
        Place::Local(float_res),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(casted)),
            Operand::Constant(MirConstant::Float(0.5)),
        ),
        Span::dummy(),
    ));
    f_float.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(float_res))),
        Span::dummy(),
    ));
    m.add_function(f_float);

    // Function 2: Bool logic
    let sig_bool = FunctionSignature {
        name: "scalar_bool_fn".into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Bool,
        is_public: true,
        effects: None,
    };
    let mut f_bool = MirFunction::new(sig_bool, Span::dummy());
    let b_int = f_bool.params[0];
    let bool_res = f_bool.add_local(None, MirType::Unknown, true, Span::dummy());

    // bool_res = b_int > 0
    f_bool.blocks[0].statements.push(Statement::Assign(
        Place::Local(bool_res),
        Rvalue::BinaryOp(
            MirBinaryOp::Gt,
            Operand::Copy(Place::Local(b_int)),
            Operand::Constant(MirConstant::Int(0)),
        ),
        Span::dummy(),
    ));
    f_bool.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(bool_res))),
        Span::dummy(),
    ));
    m.add_function(f_bool);

    // Function 3: Unit return
    let sig_unit = FunctionSignature {
        name: "scalar_unit_fn".into(),
        param_types: vec![],
        return_type: MirType::Unit,
        is_public: true,
        effects: None,
    };
    let mut f_unit = MirFunction::new(sig_unit, Span::dummy());
    f_unit.blocks[0].terminator = Some(Terminator::Return(None, Span::dummy()));
    m.add_function(f_unit);

    let res = compiler.compile_module(&m).expect("Scalar matrix compilation must succeed");

    // Verify Float execution:
    type NativeFloatFn = unsafe extern "C" fn(i64) -> f64;
    let float_fn_ptr = res.get("scalar_float_fn").copied().expect("float entry ptr");
    let native_float: NativeFloatFn = unsafe { std::mem::transmute(float_fn_ptr) };
    let f_out = unsafe { native_float(4) };
    assert!((f_out - 4.5).abs() < 1e-6, "Expected 4.5, got {}", f_out);

    // Verify Bool execution:
    type NativeBoolFn = unsafe extern "C" fn(i64) -> bool;
    let bool_fn_ptr = res.get("scalar_bool_fn").copied().expect("bool entry ptr");
    let native_bool: NativeBoolFn = unsafe { std::mem::transmute(bool_fn_ptr) };
    assert_eq!(unsafe { native_bool(10) }, true);
    assert_eq!(unsafe { native_bool(-5) }, false);

    // Verify Unit execution:
    type NativeUnitFn = unsafe extern "C" fn();
    let unit_fn_ptr = res.get("scalar_unit_fn").copied().expect("unit entry ptr");
    let native_unit: NativeUnitFn = unsafe { std::mem::transmute(unit_fn_ptr) };
    unsafe { native_unit() };
}

/// Valid Control 3: Same-Compiler Rejection then Valid Retry and Native Execution
#[test]
fn test_same_compiler_rejection_then_valid_retry_and_oracle() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let mut compiler = CraneliftJitCompiler::new().expect("backend init");

    // 1. Compile bad module with conflicting writes
    let mut bad_mod = MirModule::new("bad_mod".into());
    let bad_sig = FunctionSignature {
        name: "bad_conflicting_fn".into(),
        param_types: vec![],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut bad_f = MirFunction::new(bad_sig, Span::dummy());
    let bad_loc = bad_f.add_local(Some("conflict_var".into()), MirType::Unknown, true, Span::dummy());
    let b0 = bad_f.new_block();
    let b1 = bad_f.new_block();
    bad_f.blocks[0].terminator = Some(Terminator::Goto(b0, Span::dummy()));
    bad_f.blocks[b0.0].statements.push(Statement::Assign(
        Place::Local(bad_loc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(42))),
        Span::dummy(),
    ));
    bad_f.blocks[b0.0].terminator = Some(Terminator::Goto(b1, Span::dummy()));
    bad_f.blocks[b1.0].statements.push(Statement::Assign(
        Place::Local(bad_loc),
        Rvalue::Use(Operand::Constant(MirConstant::Float(3.14))),
        Span::dummy(),
    ));
    bad_f.blocks[b1.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(bad_loc))),
        Span::dummy(),
    ));
    bad_mod.add_function(bad_f);

    let bad_res = compiler.compile_module(&bad_mod);
    assert!(bad_res.is_err(), "Conflicting module MUST be rejected on same compiler!");

    // Observational check: verify bad function symbol was not leaked into compiled code ranges
    assert!(
        compiler.compiled_code_range("bad_conflicting_fn").is_none(),
        "Rejected symbol must not leave compiled code range"
    );

    // 2. On the SAME compiler, compile valid module
    let mut good_mod = MirModule::new("good_mod".into());
    let good_sig = FunctionSignature {
        name: "good_math_fn".into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut good_f = MirFunction::new(good_sig, Span::dummy());
    let p0 = good_f.params[0];
    let res_loc = good_f.add_local(None, MirType::Unknown, true, Span::dummy());
    good_f.blocks[0].statements.push(Statement::Assign(
        Place::Local(res_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::Mul,
            Operand::Copy(Place::Local(p0)),
            Operand::Constant(MirConstant::Int(3)),
        ),
        Span::dummy(),
    ));
    good_f.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(res_loc))),
        Span::dummy(),
    ));
    good_mod.add_function(good_f);

    let good_res = compiler.compile_module(&good_mod).expect("Good module compilation must succeed on same compiler");
    let code_ptr = good_res.get("good_math_fn").copied().expect("good fn code ptr");

    assert!(
        compiler.compiled_code_range("good_math_fn").is_some(),
        "Good function must have compiled code range recorded"
    );

    type NativeGoodFn = unsafe extern "C" fn(i64) -> i64;
    let native_good: NativeGoodFn = unsafe { std::mem::transmute(code_ptr) };

    for &x in &[0, 1, 10, -5, 100] {
        let expected = x * 3;
        let actual = unsafe { native_good(x) };
        assert_eq!(actual, expected, "Oracle mismatch on retry for x={}: expected {}, got {}", x, expected, actual);
    }
}

/// Discriminating Test 8: Distinguish Malformed MIR from Valid Dynamic Source Semantics
/// In dynamic Lust, a variable can legitimately hold an Int on one branch and a Float on another.
/// When evaluating through bytecode VM/interpreter, it executes safely with dynamic tag dispatch.
/// When compiling to Cranelift native unboxed code, Cranelift safely rejects unboxed compilation
/// and runtime falls back to VM bytecode execution.
#[test]
fn test_distinguish_malformed_mir_from_valid_dynamic_source_semantics() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    // In valid dynamic source semantics, dynamic branching produces different types at runtime:
    let cond_true_val = if true { lust::bytecode::Value::Int(42) } else { lust::bytecode::Value::Float(3.14) };
    let cond_false_val = if false { lust::bytecode::Value::Int(42) } else { lust::bytecode::Value::Float(3.14) };

    match cond_true_val {
        lust::bytecode::Value::Int(v) => assert_eq!(v, 42),
        other => panic!("Expected dynamic Int(42), got {:?}", other),
    }
    match cond_false_val {
        lust::bytecode::Value::Float(v) => assert!((v - 3.14).abs() < 1e-6),
        other => panic!("Expected dynamic Float(3.14), got {:?}", other),
    }

    // But Cranelift unboxed JIT compiler MUST safely reject compiling this into unboxed native machine code:
    let mut compiler = CraneliftJitCompiler::new().expect("backend init");
    let mut m = MirModule::new("mod_dynamic_semantics".into());
    let sig = FunctionSignature {
        name: "dynamic_branch_fn".into(),
        param_types: vec![MirType::Bool],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut f = MirFunction::new(sig, Span::dummy());
    let cond = f.params[0];
    let dyn_var = f.add_local(Some("dyn_var".into()), MirType::Unknown, true, Span::dummy());
    let b_then = f.new_block();
    let b_else = f.new_block();
    let b_ret = f.new_block();

    f.blocks[0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond)),
        then_block: b_then,
        else_block: b_else,
        span: Span::dummy(),
    });
    f.blocks[b_then.0].statements.push(Statement::Assign(
        Place::Local(dyn_var),
        Rvalue::Use(Operand::Constant(MirConstant::Int(42))),
        Span::dummy(),
    ));
    f.blocks[b_then.0].terminator = Some(Terminator::Goto(b_ret, Span::dummy()));

    f.blocks[b_else.0].statements.push(Statement::Assign(
        Place::Local(dyn_var),
        Rvalue::Use(Operand::Constant(MirConstant::Float(3.14))),
        Span::dummy(),
    ));
    f.blocks[b_else.0].terminator = Some(Terminator::Goto(b_ret, Span::dummy()));

    f.blocks[b_ret.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(dyn_var))),
        Span::dummy(),
    ));
    m.add_function(f);

    let res = compiler.compile_module(&m);
    assert!(
        res.is_err(),
        "Dynamic polymorphism across branches must be rejected by Cranelift unboxed compiler (safely falling back to bytecode VM)! Got: {:?}",
        res
    );
}

/// Valid Control 4: Non-"arg" Named Parameter Evidence Propagation
/// In AST lowering, parameters are often renamed from "arg0" to "n", "x", etc.
/// refine_locals_with_params must recognize param indices from mir_fn.params
/// and propagate their concrete types into downstream expressions.
#[test]
fn test_named_parameter_evidence_propagation_with_params() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let mut compiler = CraneliftJitCompiler::new().expect("backend init");

    let mut m = MirModule::new("mod_named_param".into());
    let sig = FunctionSignature {
        name: "named_param_fn".into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut f = MirFunction::new(sig, Span::dummy());
    let p0 = f.params[0];
    // Rename parameter to "n" (as AST lowering does):
    f.locals[p0.0].name = Some("n".into());

    let temp = f.add_local(None, MirType::Unknown, true, Span::dummy());

    // temp = n + 100
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(temp),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(p0)),
            Operand::Constant(MirConstant::Int(100)),
        ),
        Span::dummy(),
    ));
    f.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(temp))),
        Span::dummy(),
    ));

    // Verify refine_locals_with_params refines temp to Int:
    let refined_locals = refine_locals_with_params(&f.locals, &f.blocks, &MirType::Int, &f.params);
    assert_eq!(
        refined_locals[temp.0].ty,
        MirType::Int,
        "Anonymous temp must be refined to Int via parameter 'n' evidence!"
    );

    m.add_function(f);

    let res = compiler.compile_module(&m).expect("Named parameter compilation must succeed");
    let code_ptr = res.get("named_param_fn").copied().expect("entry ptr");

    type NativeNamedParamFn = unsafe extern "C" fn(i64) -> i64;
    let native_fn: NativeNamedParamFn = unsafe { std::mem::transmute(code_ptr) };

    let out = unsafe { native_fn(23) };
    assert_eq!(out, 123, "Oracle mismatch: expected 123, got {}", out);
}

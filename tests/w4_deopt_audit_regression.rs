#![cfg(feature = "test-deopt")]

use lust::ast::Span;
use lust::mir::cranelift_backend::{
    get_current_sticky_error, native_invocation_depth, push_native_invocation, safepoint_reset,
    set_current_deopt_payload, set_current_sticky_error, take_current_sticky_error,
    take_deopt_payload, CraneliftJitCompiler, DeoptPayload,
};
use lust::mir::ir::{
    BlockId, FunctionSignature, MirConstant, MirFunction, MirType, Operand, Place, Rvalue,
    Statement, Terminator,
};

#[test]
fn invocation_must_not_consume_standalone_payload_from_an_older_execution() {
    safepoint_reset();
    set_current_deopt_payload(DeoptPayload::new(7, vec![(0, 123)]));

    let _invocation = push_native_invocation("new_call".into(), 1, 2);
    assert!(
        take_deopt_payload().is_none(),
        "a new invocation must not fall through to a standalone payload from an earlier execution"
    );
    drop(_invocation);
    safepoint_reset();
}

#[test]
fn invocation_must_not_consume_standalone_sticky_error_from_an_older_execution() {
    safepoint_reset();
    set_current_sticky_error("standalone_sticky_err".into());
    assert_eq!(
        get_current_sticky_error(),
        Some("standalone_sticky_err".into())
    );

    let _invocation = push_native_invocation("isolated_err_call".into(), 1, 3);
    assert!(
        get_current_sticky_error().is_none(),
        "active invocation must not see standalone sticky error from older execution"
    );
    assert!(
        take_current_sticky_error().is_none(),
        "active invocation must not consume standalone sticky error from older execution"
    );
    drop(_invocation);
    safepoint_reset();
}

#[test]
fn nested_invocation_isolation_and_no_leakage() {
    safepoint_reset();
    let _outer = push_native_invocation("outer_fn".into(), 1, 10);
    set_current_deopt_payload(DeoptPayload::new(10, vec![(0, 42)]));
    set_current_sticky_error("outer error".into());

    {
        let _inner = push_native_invocation("inner_fn".into(), 2, 20);
        // Inner invocation has no payload or error of its own:
        assert!(
            take_deopt_payload().is_none(),
            "inner invocation cannot consume outer payload"
        );
        assert!(
            get_current_sticky_error().is_none(),
            "inner invocation cannot see outer sticky error"
        );
        assert!(
            take_current_sticky_error().is_none(),
            "inner invocation cannot consume outer sticky error"
        );

        // Inner sets its own payload and error:
        set_current_deopt_payload(DeoptPayload::new(20, vec![(1, 99)]));
        set_current_sticky_error("inner error".into());

        assert_eq!(get_current_sticky_error(), Some("inner error".into()));
        let inner_p = take_deopt_payload().expect("inner payload must exist");
        assert_eq!(inner_p.target_pc, 20);
        assert_eq!(take_current_sticky_error(), Some("inner error".into()));
    } // inner drops here

    // Outer invocation payload and error must remain intact:
    assert_eq!(get_current_sticky_error(), Some("outer error".into()));
    let outer_p = take_deopt_payload().expect("outer payload must remain intact");
    assert_eq!(outer_p.target_pc, 10);
    assert_eq!(take_current_sticky_error(), Some("outer error".into()));

    drop(_outer);
    safepoint_reset();
}

#[test]
fn explicit_legacy_behavior_without_invocation() {
    safepoint_reset();
    assert_eq!(native_invocation_depth(), 0);

    // Direct standalone deopt payload
    set_current_deopt_payload(DeoptPayload::new(99, vec![(0, 1)]));
    let p = take_deopt_payload().expect("standalone consumer must receive payload");
    assert_eq!(p.target_pc, 99);
    assert!(take_deopt_payload().is_none());

    // Direct standalone sticky error
    set_current_sticky_error("standalone err".into());
    assert_eq!(get_current_sticky_error(), Some("standalone err".into()));
    assert_eq!(take_current_sticky_error(), Some("standalone err".into()));
    assert!(take_current_sticky_error().is_none());
}

#[test]
fn metadata_derivation_error_rejects_compile_no_all_locals_fallback() {
    let sig = FunctionSignature {
        name: "bad_metadata_fn".into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    // Add local with unsupported type Array(Int)
    let bad_loc = func.add_local(
        Some("bad_arr".into()),
        MirType::Array(Box::new(MirType::Int)),
        true,
        Span::dummy(),
    );
    let dest_loc = func.add_local(Some("dest".into()), MirType::Int, false, Span::dummy());

    // Block 0 calls an effect/handler passing bad_loc:
    func.blocks[0].terminator = Some(Terminator::Call {
        callee: "effect.perform".into(),
        args: vec![Operand::Copy(Place::Local(bad_loc))],
        destination: Place::Local(dest_loc),
        success: BlockId(0),
        error: None,
        span: Span::dummy(),
    });

    // Verification: derive_proven_site_metadata MUST fail closed:
    assert!(
        func.derive_proven_site_metadata(0).is_err(),
        "derive_proven_site_metadata must fail on unsupported non-scalar type"
    );

    // JIT compilation MUST reject compilation before native entry/capture:
    let mut compiler = CraneliftJitCompiler::new().expect("compiler init");
    let compile_res = compiler.compile_function(&func);
    assert!(
        compile_res.is_err(),
        "Cranelift compilation must reject deopt site when metadata derivation fails (no all_locals fallback!)"
    );
}

#[test]
fn definite_assignment_order_independent_fixed_point_loops_and_joins() {
    let sig = FunctionSignature {
        name: "loop_join_fn".into(),
        param_types: vec![],
        return_type: MirType::Unit,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let cond_loc = func.add_local(Some("cond".into()), MirType::Bool, true, Span::dummy());
    let x_loc = func.add_local(Some("x".into()), MirType::Int, true, Span::dummy());
    let dest_loc = func.add_local(Some("dest".into()), MirType::Unit, false, Span::dummy());

    let b0 = BlockId(0);
    let b1 = func.new_block(); // BlockId(1)
    let b2 = func.new_block(); // BlockId(2)
    let b3 = func.new_block(); // BlockId(3)

    // Block 0: cond = true; goto b1
    func.blocks[b0.0].statements.push(Statement::Assign(
        Place::Local(cond_loc),
        Rvalue::Use(Operand::Constant(MirConstant::Bool(true))),
        Span::dummy(),
    ));
    func.blocks[b0.0].terminator = Some(Terminator::Goto(b1, Span::dummy()));

    // Block 1: if cond then b2 else b3
    func.blocks[b1.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond_loc)),
        then_block: b2,
        else_block: b3,
        span: Span::dummy(),
    });

    // Block 2: x = 42; goto b1
    func.blocks[b2.0].statements.push(Statement::Assign(
        Place::Local(x_loc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(42))),
        Span::dummy(),
    ));
    func.blocks[b2.0].terminator = Some(Terminator::Goto(b1, Span::dummy()));

    // Block 3: effect.perform()
    func.blocks[b3.0].terminator = Some(Terminator::Call {
        callee: "effect.perform".into(),
        args: vec![],
        destination: Place::Local(dest_loc),
        success: b3,
        error: None,
        span: Span::dummy(),
    });

    let proven = func
        .derive_proven_site_metadata(b3.0)
        .expect("metadata derivation must succeed for valid block");

    assert!(
        !proven.contains(&x_loc.0),
        "x is NOT definitely assigned at join Block 3 because path 0->1->3 does not assign x"
    );
}

#[test]
fn liveness_at_pre_terminator_excludes_dead_locals() {
    let sig = FunctionSignature {
        name: "liveness_test_fn".into(),
        param_types: vec![],
        return_type: MirType::Unit,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let live_loc = func.add_local(Some("live".into()), MirType::Int, true, Span::dummy());
    let dead_loc = func.add_local(Some("dead".into()), MirType::Int, true, Span::dummy());
    let dest_loc = func.add_local(Some("dest".into()), MirType::Unit, false, Span::dummy());

    // Both live and dead locals are assigned in Block 0:
    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(live_loc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(10))),
        Span::dummy(),
    ));
    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(dead_loc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(20))),
        Span::dummy(),
    ));

    // Terminator calls effect with ONLY live_loc. dead_loc is NEVER used anywhere:
    func.blocks[0].terminator = Some(Terminator::Call {
        callee: "effect.perform".into(),
        args: vec![Operand::Copy(Place::Local(live_loc))],
        destination: Place::Local(dest_loc),
        success: BlockId(0),
        error: None,
        span: Span::dummy(),
    });

    let proven = func
        .derive_proven_site_metadata(0)
        .expect("metadata derivation must succeed");

    assert!(
        proven.contains(&live_loc.0),
        "live local used in call must be in authoritative metadata"
    );
    assert!(
        !proven.contains(&dead_loc.0),
        "dead local must NOT be in authoritative metadata (never assume all locals initialized are live)"
    );
}

#[test]
fn invalid_cfg_fails_closed() {
    let sig = FunctionSignature {
        name: "invalid_cfg_fn".into(),
        param_types: vec![],
        return_type: MirType::Unit,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    // Block 0 jumps to out-of-bounds BlockId(99)
    func.blocks[0].terminator = Some(Terminator::Goto(BlockId(99), Span::dummy()));

    let res = func.derive_proven_site_metadata(0);
    assert!(
        res.is_err(),
        "derive_proven_site_metadata must fail closed on invalid CFG jumping out of bounds"
    );
}

#[test]
fn live_uninitialized_local_fails_closed() {
    let sig = FunctionSignature {
        name: "uninit_read_fn".into(),
        param_types: vec![],
        return_type: MirType::Unit,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let uninit_loc = func.add_local(Some("uninit".into()), MirType::Int, true, Span::dummy());
    let dest_loc = func.add_local(Some("dest".into()), MirType::Unit, false, Span::dummy());

    // Block 0 does NOT initialize uninit_loc, but passes it to effect.perform:
    func.blocks[0].terminator = Some(Terminator::Call {
        callee: "effect.perform".into(),
        args: vec![Operand::Copy(Place::Local(uninit_loc))],
        destination: Place::Local(dest_loc),
        success: BlockId(0),
        error: None,
        span: Span::dummy(),
    });

    let res = func.derive_proven_site_metadata(0);
    assert!(
        res.is_err(),
        "derive_proven_site_metadata must fail closed when a live local is uninitialized"
    );
}

#![cfg(feature = "std")]

//! RED Test Suite for TASK-W4-REFINE-BOUNDARY:
//! Investigating and remediating DA/LV concerns:
//! 1. `refine_locals` lacks explicit params and relies on unsound `argN` name heuristic;
//!    backward propagation exclusion fails when params slice is empty.
//! 2. Non-param locals named `arg0` misidentified as parameters by name heuristic.
//! 3. Projected writes (`Place::Field`, `Place::Index`) omitted from `def_count`,
//!    allowing unsound backward copy propagation on mutated root locals.
//! 4. Nonconvergence fail-closed branch tested via explicit test seams (`refine_locals_with_seam`,
//!    `refine_mir_function_with_seam`) and scoped compiler configuration (`set_refine_max_iterations_override_for_test`),
//!    completely avoiding process-global environment variables.
//! 5. Valid native execution with struct projections and parameters verified against oracles.

use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use lust::ast::Span;
use lust::data::layout::{CanonicalTypeTag, TargetAbi};
use lust::mir::cranelift_backend::{
    jit_engine::refine_locals,
    safepoint_reset, CraneliftJitCompiler,
};
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::jit_engine::{refine_locals_with_seam, refine_mir_function_with_seam};
use lust::mir::ir::{
    BasicBlock, BlockId, FunctionSignature, LocalDecl, LocalId, MirBinaryOp, MirConstant,
    MirFunction, MirModule, MirStructFieldLayout, MirStructLayout, MirType, Operand, Place,
    Rvalue, Statement, Terminator,
};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

// ============================================================================
// RED 1: Non-"arg" Named Parameter Must Be Preserved and Excluded from Unsound Backward Copy
// ============================================================================

/// When a parameter is named "n" (as produced by AST lowering from function `fn f(n: int)`),
/// calling `refine_locals` with empty params must NOT treat it as an unconstrained temporary
/// and must NOT allow backward propagation to overwrite its declared type.
#[test]
fn test_red_non_arg_param_preserved_in_refine_locals() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    // Local 0: param "n" of type Int
    // Local 1: temp "y"
    let locals = vec![
        LocalDecl {
            id: LocalId(0),
            name: Some("n".into()),
            ty: MirType::Int,
            is_mut: false,
            span: Span::dummy(),
            is_stack_promoted: false,
            is_param: true,
        },
        LocalDecl {
            id: LocalId(1),
            name: Some("y".into()),
            ty: MirType::Unknown,
            is_mut: true,
            span: Span::dummy(),
            is_stack_promoted: false,
            is_param: false,
        },
    ];

    // Block 0:
    // y = n
    // return y (with return_type Float)
    // Under the buggy heuristic:
    // 1. "n" does not start with "arg", so is_param is FALSE in refine_locals.
    // 2. Local "n" (ty Int) is treated as Unconstrained.
    // 3. Return constraint forces "y" to Float.
    // 4. Backward copy from "y" into "n" overwrites "n" to Float!
    let block = BasicBlock {
        id: BlockId(0),
        statements: vec![Statement::Assign(
            Place::Local(LocalId(1)),
            Rvalue::Use(Operand::Copy(Place::Local(LocalId(0)))),
            Span::dummy(),
        )],
        terminator: Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(LocalId(1)))),
            Span::dummy(),
        )),
        is_cleanup: false,
    };

    let refined = refine_locals(&locals, &[block], &MirType::Float);

    // Param "n" must remain Int (or conflict), it must NEVER be rewritten to Float!
    assert_ne!(
        refined[0].ty,
        MirType::Float,
        "Parameter 'n' was overwritten to Float by backward copy because name did not start with 'arg'!"
    );
    assert_eq!(
        refined[0].ty,
        MirType::Int,
        "Parameter 'n' must retain its declared Int type via parameter metadata!"
    );
}

// ============================================================================
// RED 2: Non-Param Local Named "arg0" Must NOT Be Misidentified as a Parameter
// ============================================================================

/// In user source code, a developer may write `let arg0 = 10; arg0 = 3.14;` (e.g. CLI argument parsing).
/// A local variable named "arg0" that is NOT a parameter must be treated as a normal local:
/// its conflicting assignments must cause conflict / fallback, and it must not be pinned as an immutable parameter.
#[test]
fn test_red_non_param_named_arg0_not_misidentified_as_param() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    // Local 0: user local variable named "arg0", declared with default ty Int (or Unknown),
    // but it is NOT a formal parameter.
    let locals = vec![
        LocalDecl {
            id: LocalId(0),
            name: Some("arg0".into()),
            ty: MirType::Int,
            is_mut: true,
            span: Span::dummy(),
            is_stack_promoted: false,
            is_param: false,
        },
        LocalDecl {
            id: LocalId(1),
            name: Some("res".into()),
            ty: MirType::Unknown,
            is_mut: true,
            span: Span::dummy(),
            is_stack_promoted: false,
            is_param: false,
        },
    ];

    // Block 0: arg0 is assigned Float 2.718
    let block = BasicBlock {
        id: BlockId(0),
        statements: vec![
            Statement::Assign(
                Place::Local(LocalId(0)),
                Rvalue::Use(Operand::Constant(MirConstant::Float(2.718))),
                Span::dummy(),
            ),
            Statement::Assign(
                Place::Local(LocalId(1)),
                Rvalue::Use(Operand::Copy(Place::Local(LocalId(0)))),
                Span::dummy(),
            ),
        ],
        terminator: None,
        is_cleanup: false,
    };

    let refined = refine_locals(&locals, &[block], &MirType::Unit);

    // Since arg0 is a normal local assigned Float, it must be refined to Float!
    // Under the buggy heuristic, because its name is "arg0", it was locked into Concrete(Int),
    // causing an erroneous conflict with Float(2.718)!
    assert_eq!(
        refined[0].ty,
        MirType::Float,
        "Local 'arg0' was misidentified as parameter and locked to declared Int instead of refining to assigned Float!"
    );
}

// ============================================================================
// RED 3: Projected Writes Must Increment def_count and Prevent Unsound Backward Copy
// ============================================================================

/// When a struct/tuple local `s` is initialized from `src` (s = src), and subsequently
/// modified via a projected write (`s.0 = 99`), `s` is mutated and is NOT singly defined.
/// Backward copy propagation from `s` into `src` must be suppressed because `s` was modified!
#[test]
fn test_red_projected_write_increments_def_count_suppressing_backward_copy() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let mut layouts = hashbrown::HashMap::new();
    layouts.insert(
        "Pair".to_string(),
        MirStructLayout {
            type_id: 1,
            type_name: "Pair".to_string(),
            size_bytes: 16,
            align_bytes: 8,
            is_c_flat: true,
            target_abi: TargetAbi::x86_64(),
            fields: vec![
                MirStructFieldLayout {
                    ordinal: 0,
                    name: "first".to_string(),
                    tag: CanonicalTypeTag::Int64,
                    byte_offset: 0,
                    byte_size: 8,
                    is_nullable: false,
                    is_optional: false,
                },
                MirStructFieldLayout {
                    ordinal: 1,
                    name: "second".to_string(),
                    tag: CanonicalTypeTag::Int64,
                    byte_offset: 8,
                    byte_size: 8,
                    is_nullable: false,
                    is_optional: false,
                },
            ],
        },
    );

    // Local 0: src of type Unknown (unconstrained)
    // Local 1: dest of type Unknown
    let locals = vec![
        LocalDecl {
            id: LocalId(0),
            name: Some("src".into()),
            ty: MirType::Unknown,
            is_mut: true,
            span: Span::dummy(),
            is_stack_promoted: false,
            is_param: false,
        },
        LocalDecl {
            id: LocalId(1),
            name: Some("dest".into()),
            ty: MirType::Unknown,
            is_mut: true,
            span: Span::dummy(),
            is_stack_promoted: false,
            is_param: false,
        },
    ];

    // Block 0:
    // dest = src;           (def 1 of dest)
    // dest.first = 99;      (def 2 of dest via projected write!)
    // return dest;          (return constraint: Struct("Pair"))
    //
    // Under buggy code:
    // 1. def_count ONLY counts Place::Local, so def_count[dest] == 1!
    // 2. Return terminator constraints dest to Struct("Pair").
    // 3. Because def_count[dest] == 1, backward copy treats dest as singly-defined
    //    and propagates Struct("Pair") into src!
    // BUT dest was mutated via dest.first = 99! If projected write is counted,
    // dest has def_count == 2, so backward copy MUST NOT treat dest as singly-defined!
    let block = BasicBlock {
        id: BlockId(0),
        statements: vec![
            Statement::Assign(
                Place::Local(LocalId(1)),
                Rvalue::Use(Operand::Copy(Place::Local(LocalId(0)))),
                Span::dummy(),
            ),
            Statement::Assign(
                Place::Field(Box::new(Place::Local(LocalId(1))), 0),
                Rvalue::Use(Operand::Constant(MirConstant::Int(99))),
                Span::dummy(),
            ),
        ],
        terminator: Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(LocalId(1)))),
            Span::dummy(),
        )),
        is_cleanup: false,
    };

    use lust::mir::cranelift_backend::jit_engine::refine_locals_with_context;
    let refined = refine_locals_with_context(&locals, &[block], &MirType::Struct("Pair".into()), &[], &layouts);

    // Because dest had a projected write, dest is multiply-defined (def_count == 2).
    // Unsound backward copy must NOT propagate into src; src must remain Unknown!
    assert_eq!(
        refined[0].ty,
        MirType::Unknown,
        "src was unsoundly refined via backward copy because projected write dest.first was omitted from def_count!"
    );
}

/// A projected write storing a conflicting type (e.g. Float into Int field) must fail closed.
#[test]
fn test_red_projected_write_type_conflict_fails_closed() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let mut layouts = hashbrown::HashMap::new();
    layouts.insert(
        "Pair".to_string(),
        MirStructLayout {
            type_id: 1,
            type_name: "Pair".to_string(),
            size_bytes: 16,
            align_bytes: 8,
            is_c_flat: true,
            target_abi: TargetAbi::x86_64(),
            fields: vec![
                MirStructFieldLayout {
                    ordinal: 0,
                    name: "first".to_string(),
                    tag: CanonicalTypeTag::Int64,
                    byte_offset: 0,
                    byte_size: 8,
                    is_nullable: false,
                    is_optional: false,
                },
            ],
        },
    );

    let locals = vec![
        LocalDecl {
            id: LocalId(0),
            name: Some("pt".into()),
            ty: MirType::Struct("Pair".into()),
            is_mut: true,
            span: Span::dummy(),
            is_stack_promoted: false,
            is_param: false,
        },
    ];

    // Assign Float 3.14 to pt.first (which is CanonicalTypeTag::Int64):
    let block = BasicBlock {
        id: BlockId(0),
        statements: vec![
            Statement::Assign(
                Place::Field(Box::new(Place::Local(LocalId(0))), 0),
                Rvalue::Use(Operand::Constant(MirConstant::Float(3.14))),
                Span::dummy(),
            ),
        ],
        terminator: None,
        is_cleanup: false,
    };

    use lust::mir::cranelift_backend::jit_engine::refine_locals_with_context;
    let refined = refine_locals_with_context(&locals, &[block], &MirType::Unit, &[], &layouts);

    // pt must fail closed to Unknown due to field write type conflict!
    assert_eq!(
        refined[0].ty,
        MirType::Unknown,
        "pt must fail closed to Unknown when conflicting type is assigned to its field!"
    );
}

// ============================================================================
// RED 4: Forced Convergence Guard Test Seam Must Fail Closed
// ============================================================================

/// When the convergence guard bound is exceeded without reaching a fixed point,
/// all locals must transition to Conflict / Unknown and Cranelift compilation must
/// fail closed, safely refusing native code generation.
#[cfg(feature = "test-deopt")]
#[test]
fn test_red_forced_convergence_guard_fails_closed() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    // 10-hop copy chain requiring 10 iterations to converge:
    // x10 = 42
    // x9 = x10
    // ...
    // x0 = x1
    const HOPS: usize = 10;
    let mut locals = Vec::with_capacity(HOPS + 1);
    for i in 0..=HOPS {
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

    let mut stmts = Vec::new();
    for i in 0..HOPS {
        stmts.push(Statement::Assign(
            Place::Local(LocalId(i)),
            Rvalue::Use(Operand::Copy(Place::Local(LocalId(i + 1)))),
            Span::dummy(),
        ));
    }
    stmts.push(Statement::Assign(
        Place::Local(LocalId(HOPS)),
        Rvalue::Use(Operand::Constant(MirConstant::Int(42))),
        Span::dummy(),
    ));

    let block = BasicBlock {
        id: BlockId(0),
        statements: stmts,
        terminator: None,
        is_cleanup: false,
    };

    // When forced max_iterations is set to 2 (via explicit test seam refine_locals_with_seam with Some(2)),
    // the 10-hop chain CANNOT converge in 2 iterations.
    // The fail-closed branch MUST trigger, setting all locals to Conflict -> Unknown!
    let dummy_layouts = hashbrown::HashMap::new();
    let refined = refine_locals_with_seam(
        &locals,
        &[block.clone()],
        &MirType::Unit,
        &[],
        &dummy_layouts,
        Some(2),
    );

    // All locals must be Unknown because fixed point was not reached!
    for i in 0..=HOPS {
        assert_eq!(
            refined[i].ty,
            MirType::Unknown,
            "Local x{i} was NOT failed-closed to Unknown when convergence bound was exceeded!"
        );
    }
}

/// Direct test seam call via `refine_locals_with_seam` with `max_iterations_override`.
#[cfg(feature = "test-deopt")]
#[test]
fn test_refine_locals_with_seam_direct_override() {
    use lust::mir::cranelift_backend::jit_engine::refine_locals_with_seam;

    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    const HOPS: usize = 6;
    let mut locals = Vec::with_capacity(HOPS + 1);
    for i in 0..=HOPS {
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

    let mut stmts = Vec::new();
    for i in 0..HOPS {
        stmts.push(Statement::Assign(
            Place::Local(LocalId(i)),
            Rvalue::Use(Operand::Copy(Place::Local(LocalId(i + 1)))),
            Span::dummy(),
        ));
    }
    stmts.push(Statement::Assign(
        Place::Local(LocalId(HOPS)),
        Rvalue::Use(Operand::Constant(MirConstant::Int(99))),
        Span::dummy(),
    ));

    let block = BasicBlock {
        id: BlockId(0),
        statements: stmts,
        terminator: None,
        is_cleanup: false,
    };

    let dummy_layouts = hashbrown::HashMap::new();

    // 1. With seam override limit = 1: fails closed (Unknown)
    let refined_limited = refine_locals_with_seam(
        &locals,
        &[block.clone()],
        &MirType::Unit,
        &[],
        &dummy_layouts,
        Some(1),
    );
    assert_eq!(
        refined_limited[0].ty,
        MirType::Unknown,
        "x0 must fail closed when limit=1 is exceeded"
    );

    // 2. With seam override limit = None (normal dynamic budget): converges to Int!
    let refined_normal = refine_locals_with_seam(
        &locals,
        &[block],
        &MirType::Unit,
        &[],
        &dummy_layouts,
        None,
    );
    assert_eq!(
        refined_normal[0].ty,
        MirType::Int,
        "x0 must converge to Int under normal budget"
    );
}

// ============================================================================
// 5. Valid Native Execution & Cranelift JIT Compilation with Independent Oracle
// ============================================================================

/// Valid native execution: function with explicit parameter named "n" (not "arg0"),
/// performing arithmetic, compiled through Cranelift JIT and verified against native oracle.
#[test]
fn test_valid_native_explicit_param_execution() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let mut compiler = CraneliftJitCompiler::new().expect("backend init");

    let mut m = MirModule::new("mod_valid_native".into());
    let sig = FunctionSignature {
        name: "explicit_param_fn".into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut f = MirFunction::new(sig, Span::dummy());
    let p0 = f.params[0];
    // Rename parameter to "n" (explicit parameter metadata must be retained):
    f.locals[p0.0].name = Some("n".into());

    let temp = f.add_local(Some("local_temp".into()), MirType::Unknown, true, Span::dummy());

    // temp = n * 3 + 7
    let mul_res = f.add_local(None, MirType::Unknown, true, Span::dummy());
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(mul_res),
        Rvalue::BinaryOp(
            MirBinaryOp::Mul,
            Operand::Copy(Place::Local(p0)),
            Operand::Constant(MirConstant::Int(3)),
        ),
        Span::dummy(),
    ));
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(temp),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(mul_res)),
            Operand::Constant(MirConstant::Int(7)),
        ),
        Span::dummy(),
    ));
    f.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(temp))),
        Span::dummy(),
    ));

    m.add_function(f);

    let res = compiler.compile_module(&m).expect("Native compilation must succeed");
    let code_ptr = res.get("explicit_param_fn").copied().expect("entry ptr");

    type NativeFn = unsafe extern "C" fn(i64) -> i64;
    let native_fn: NativeFn = unsafe { std::mem::transmute(code_ptr) };

    for input in [0, 5, -10, 42, 100] {
        let expected = input * 3 + 7;
        let actual = unsafe { native_fn(input) };
        assert_eq!(
            actual, expected,
            "Oracle mismatch for input {}: expected {}, got {}",
            input, expected, actual
        );
    }
}

/// Valid native execution with struct allocation, field mutation (projected write),
/// and field read matching independent oracle.
#[test]
fn test_valid_native_struct_projection_execution() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let mut compiler = CraneliftJitCompiler::new().expect("backend init");

    compiler.register_struct_layouts(&[MirStructLayout {
        type_id: 42,
        type_name: "Vec2".to_string(),
        size_bytes: 16,
        align_bytes: 8,
        is_c_flat: true,
        target_abi: TargetAbi::x86_64(),
        fields: vec![
            MirStructFieldLayout {
                ordinal: 0,
                name: "x".to_string(),
                tag: CanonicalTypeTag::Int64,
                byte_offset: 0,
                byte_size: 8,
                is_nullable: false,
                is_optional: false,
            },
            MirStructFieldLayout {
                ordinal: 1,
                name: "y".to_string(),
                tag: CanonicalTypeTag::Int64,
                byte_offset: 8,
                byte_size: 8,
                is_nullable: false,
                is_optional: false,
            },
        ],
    }]);

    let mut m = MirModule::new("mod_valid_struct".into());
    let sig = FunctionSignature {
        name: "struct_proj_fn".into(),
        param_types: vec![MirType::Int, MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut f = MirFunction::new(sig, Span::dummy());
    let p_x = f.params[0];
    let p_y = f.params[1];
    f.locals[p_x.0].name = Some("init_x".into());
    f.locals[p_y.0].name = Some("init_y".into());

    let pt = f.add_local(
        Some("pt".into()),
        MirType::Struct("Vec2".into()),
        true,
        Span::dummy(),
    );

    // pt = Aggregate(Vec2, [init_x, init_y])
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(pt),
        Rvalue::Aggregate(
            MirType::Struct("Vec2".into()),
            vec![
                Operand::Copy(Place::Local(p_x)),
                Operand::Copy(Place::Local(p_y)),
            ],
        ),
        Span::dummy(),
    ));

    // pt.x = pt.x + 10 (projected read + projected write!)
    let read_x = f.add_local(None, MirType::Unknown, true, Span::dummy());
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(read_x),
        Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(pt)), 0))),
        Span::dummy(),
    ));
    let new_x = f.add_local(None, MirType::Unknown, true, Span::dummy());
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(new_x),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(read_x)),
            Operand::Constant(MirConstant::Int(10)),
        ),
        Span::dummy(),
    ));
    f.blocks[0].statements.push(Statement::Assign(
        Place::Field(Box::new(Place::Local(pt)), 0),
        Rvalue::Use(Operand::Copy(Place::Local(new_x))),
        Span::dummy(),
    ));

    // Return pt.x + pt.y
    let fin_x = f.add_local(None, MirType::Unknown, true, Span::dummy());
    let fin_y = f.add_local(None, MirType::Unknown, true, Span::dummy());
    let sum = f.add_local(None, MirType::Unknown, true, Span::dummy());
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(fin_x),
        Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(pt)), 0))),
        Span::dummy(),
    ));
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(fin_y),
        Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(pt)), 1))),
        Span::dummy(),
    ));
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(sum),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(fin_x)),
            Operand::Copy(Place::Local(fin_y)),
        ),
        Span::dummy(),
    ));
    f.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(sum))),
        Span::dummy(),
    ));

    m.add_function(f);

    let res = compiler.compile_module(&m).expect("Struct projection compilation must succeed");
    let code_ptr = res.get("struct_proj_fn").copied().expect("entry ptr");

    type NativeVecFn = unsafe extern "C" fn(i64, i64) -> i64;
    let native_fn: NativeVecFn = unsafe { std::mem::transmute(code_ptr) };

    let out = unsafe { native_fn(5, 20) };
    // expected: (5 + 10) + 20 = 35
    assert_eq!(out, 35, "Oracle mismatch for Vec2: expected 35, got {}", out);
}

/// Cranelift native compilation MUST fail closed (Err) when the convergence guard triggers.
#[cfg(feature = "test-deopt")]
#[test]
fn test_native_compile_refusal_when_convergence_guard_triggers() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let mut compiler = CraneliftJitCompiler::new().expect("backend init");

    let mut m = MirModule::new("mod_convergence_guard".into());
    let sig = FunctionSignature {
        name: "guard_fn".into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut f = MirFunction::new(sig, Span::dummy());

    // 5-hop copy chain
    let mut prev = f.params[0];
    for i in 0..5 {
        let nxt = f.add_local(Some(format!("v{i}")), MirType::Unknown, true, Span::dummy());
        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(nxt),
            Rvalue::Use(Operand::Copy(Place::Local(prev))),
            Span::dummy(),
        ));
        prev = nxt;
    }
    f.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(prev))),
        Span::dummy(),
    ));
    m.add_function(f);

    // Force max iterations = 1 via scoped compiler test configuration (default-off, no process-global env manipulation)
    compiler.set_refine_max_iterations_override_for_test(Some(1));
    let res = compiler.compile_module(&m);

    assert!(
        res.is_err(),
        "Cranelift JIT compilation MUST refuse native code generation when convergence guard triggers! Got: {:?}",
        res
    );

    // Verify direct MirFunction test seam also fails closed to Unknown when iterations are constrained
    let dummy_layouts = hashbrown::HashMap::new();
    let f_fail = refine_mir_function_with_seam(&m.functions[0], &dummy_layouts, Some(1));
    for local in &f_fail.locals {
        assert_eq!(
            local.ty,
            MirType::Unknown,
            "Direct seam refine_mir_function_with_seam must fail closed to Unknown"
        );
    }

    // Verify compiler without override succeeds on same module pattern
    safepoint_reset();
    let mut normal_compiler = CraneliftJitCompiler::new().expect("backend init normal");
    let normal_res = normal_compiler.compile_module(&m);
    assert!(
        normal_res.is_ok(),
        "Cranelift JIT compilation without test override MUST succeed on normal copy chain: {:?}",
        normal_res
    );
}

/// Cranelift native compilation succeeds on normal multi-hop copy chain in standard production mode without override.
#[test]
fn test_normal_compiler_copy_chain_succeeds_without_override() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let mut compiler = CraneliftJitCompiler::new().expect("backend init");

    let mut m = MirModule::new("mod_normal_convergence".into());
    let sig = FunctionSignature {
        name: "normal_fn".into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut f = MirFunction::new(sig, Span::dummy());

    // 5-hop copy chain
    let mut prev = f.params[0];
    for i in 0..5 {
        let nxt = f.add_local(Some(format!("v{i}")), MirType::Unknown, true, Span::dummy());
        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(nxt),
            Rvalue::Use(Operand::Copy(Place::Local(prev))),
            Span::dummy(),
        ));
        prev = nxt;
    }
    f.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(prev))),
        Span::dummy(),
    ));
    m.add_function(f);

    let res = compiler.compile_module(&m);
    assert!(
        res.is_ok(),
        "Normal Cranelift JIT compilation without test override MUST succeed on normal copy chain: {:?}",
        res
    );
}

// ============================================================================
// 6. Child Process Worker for Isolated Timeout Watchdog
// ============================================================================

#[test]
fn test_boundary_child_worker() {
    if std::env::var("LUST_BOUNDARY_CHILD_WORKER").as_deref() == Ok("1") {
        safepoint_reset();
        let mut compiler = CraneliftJitCompiler::new().expect("backend init in child");

        let mut m = MirModule::new("child_boundary_mod".into());
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
fn test_boundary_child_timeout_isolated() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let exe = std::env::current_exe().expect("current test binary path");
    let mut cmd = Command::new(exe);
    cmd.arg("test_boundary_child_worker")
        .env("LUST_BOUNDARY_CHILD_WORKER", "1")
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
// 7. Process-Isolated Test: Normal Compiler Behavior Unaffected by External Environment Override
// ============================================================================

/// Worker process entry point: executes in an isolated child process with
/// LUST_FORCE_REFINE_MAX_ITERATIONS set in its external environment.
/// In a sound compiler, normal compilation and native execution MUST succeed and match the oracle,
/// completely unaffected by any external LUST_FORCE_REFINE_MAX_ITERATIONS environment variable.
#[test]
fn test_unaffected_by_env_worker() {
    if std::env::var("LUST_UNAFFECTED_BY_ENV_WORKER").as_deref() == Ok("1") {
        safepoint_reset();
        let mut compiler = CraneliftJitCompiler::new().expect("backend init in child");

        let mut m = MirModule::new("unaffected_mod".into());
        let sig = FunctionSignature {
            name: "unaffected_fn".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut f = MirFunction::new(sig, Span::dummy());

        // 5-hop copy chain where type inference requires multiple iterations to converge:
        // v0 = copy(param0)
        // v1 = copy(v0)
        // v2 = copy(v1)
        // v3 = copy(v2)
        // v4 = copy(v3)
        // return copy(v4)
        let mut prev = f.params[0];
        for i in 0..5 {
            let nxt = f.add_local(Some(format!("v{i}")), MirType::Unknown, true, Span::dummy());
            f.blocks[0].statements.push(Statement::Assign(
                Place::Local(nxt),
                Rvalue::Use(Operand::Copy(Place::Local(prev))),
                Span::dummy(),
            ));
            prev = nxt;
        }
        f.blocks[0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(prev))),
            Span::dummy(),
        ));
        m.add_function(f);

        let res = compiler.compile_module(&m);
        let funcs = match res {
            Ok(funcs) => funcs,
            Err(e) => {
                eprintln!(
                    "CRITICAL ERROR: Normal compilation failed due to external env override: {:?}",
                    e
                );
                std::process::exit(42);
            }
        };

        let fn_ptr = funcs.get("unaffected_fn").copied().expect("entry pointer");
        let jitted: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(fn_ptr) };

        // Test with several inputs and compare against independent oracle (identity function)
        for input in [0i64, 1, -7, 42, 999_999] {
            let actual = jitted(input);
            let oracle = input; // independent oracle: exact identity calculation
            assert_eq!(
                actual, oracle,
                "JIT output {actual} must match independent oracle {oracle}"
            );
        }

        std::process::exit(0);
    }
}

/// Spawns the worker in an isolated child process with LUST_FORCE_REFINE_MAX_ITERATIONS=1.
/// Prior to remediation, the production compiler reads LUST_FORCE_REFINE_MAX_ITERATIONS=1,
/// which truncates type refinement to 1 iteration, causing compilation to fail (RED).
/// After remediation, production compiler ignores this environment variable,
/// succeeding and matching the independent oracle (GREEN).
#[test]
fn test_normal_compiler_behavior_unaffected_by_env_override() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let exe = std::env::current_exe().expect("current test binary path");
    let mut cmd = Command::new(exe);
    cmd.arg("test_unaffected_by_env_worker")
        .arg("--nocapture")
        .env("LUST_UNAFFECTED_BY_ENV_WORKER", "1")
        .env("LUST_FORCE_REFINE_MAX_ITERATIONS", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().expect("spawn isolated child probe worker");
    let timeout = Duration::from_secs(5);
    let start = std::time::Instant::now();

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let output = child.wait_with_output().expect("read child output");
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                assert!(
                    status.success(),
                    "Normal compiler behavior MUST be unaffected by external LUST_FORCE_REFINE_MAX_ITERATIONS env var!\nChild exited with {:?}\nStdout: {}\nStderr: {}",
                    status.code(),
                    stdout,
                    stderr
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

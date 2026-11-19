#![cfg(feature = "std")]

//! Executable test suite for TASK-W4-REFINE-SOUNDNESS:
//! Discriminating RED tests for in-path defects in refine lattice:
//! 1. Unsound backward copy on multiply-defined mutable local poisoning independent locals.
//! 2. Real Option/Result constructors with Float/non-Int payloads rejected due to hardcoded Int compatibility.
//! 3. Hardcoded "enum.new" / "enum.alloc" returning Option, causing conflict when Result is expected.
//! 4. Deep reverse copy chain (>32 hops) reaching true fixed point without silent finalization while changing.

use std::sync::Mutex;
use std::process::{Command, Stdio};
use std::time::Duration;

use lust::ast::Span;
use lust::bytecode::{compiler::Compiler, Value};
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{
    jit_engine::refine_locals,
    safepoint_reset, CraneliftJitCompiler,
};
use lust::data::layout::{CanonicalTypeTag, TargetAbi};
use lust::mir::ir::{
    BasicBlock, BlockId, FunctionSignature, LocalDecl, LocalId, MirBinaryOp,
    MirConstant, MirFunction, MirModule, MirStructFieldLayout, MirStructLayout, MirType, Operand,
    Place, Rvalue, Statement, Terminator,
};
use lust::mir::lowering::AstLowering;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{VM, VmBackend};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn compile_module_from_source(src: &str) -> (MirModule, Vec<lust::ast::Item>) {
    let mut interner = Interner::new();
    let tokens = Lexer::new(src, &mut interner)
        .tokenize()
        .expect("tokenization should succeed");
    let items = Parser::new(tokens).parse().expect("parsing should succeed");
    let mut module = MirModule::new("soundness_suite".into());
    for item in &items {
        if let lust::ast::ItemKind::Function(func) = &item.kind {
            let mir_fn = AstLowering::lower_function(func)
                .expect("MIR lowering should succeed");
            module.add_function(mir_fn);
        }
    }
    (module, items)
}

fn create_vms(src: &str) -> (VM, VM) {
    let (module, items) = compile_module_from_source(src);

    let mut vm_clif = VM::new();
    vm_clif.load_mir_module(&module).unwrap();
    vm_clif.set_backend(VmBackend::Cranelift);

    let mut vm_bc = VM::new();
    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let struct_defs = checker.struct_definitions();
    vm_clif.register_structs(&struct_defs);
    vm_bc.register_structs(&struct_defs);
    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    if let Ok(bc_functions) = compiler.compile_module(&items) {
        vm_clif.load_functions(bc_functions.clone());
        vm_bc.load_functions(bc_functions);
    } else {
        vm_bc.load_mir_module(&module).unwrap();
    }
    vm_bc.set_backend(VmBackend::Bytecode);

    (vm_clif, vm_bc)
}

// ============================================================================
// Child Process Worker for Isolated Timeout Watchdog
// ============================================================================

#[test]
fn test_soundness_child_worker() {
    if std::env::var("LUST_SOUNDNESS_CHILD_WORKER").as_deref() == Ok("1") {
        safepoint_reset();
        let mut compiler = CraneliftJitCompiler::new().expect("backend init in child");

        let mut m = MirModule::new("child_soundness_mod".into());
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
fn test_soundness_child_timeout_isolated() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let exe = std::env::current_exe().expect("current test binary path");
    let mut cmd = Command::new(exe);
    cmd.arg("test_soundness_child_worker")
        .env("LUST_SOUNDNESS_CHILD_WORKER", "1")
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
// Discriminating RED Test 1: Unsound Backward Copy on Multiply-Defined Mutable Local
// ============================================================================

/// Defect: A mutable local `m` is assigned from `x` (Int).
/// Later, in another basic block, `m` is assigned 3.14 (Float).
/// `m` has conflicting writes (Int and Float).
/// But `x` is an independent local that only ever held Int(42).
/// Current refine_locals has an unsound bidirectional copy:
/// `lattice[src.0] = TypeLattice::Conflict(...)`
/// which propagates `m`'s conflict backwards into `x`!
/// This causes `x` to be marked Unknown/Conflict and rejected from compilation!
#[test]
fn test_red_unsound_backward_copy_on_multiply_defined_mutable_local() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    // Locals:
    // _0: x (starts Unknown, assigned 42)
    // _1: m (mutable temporary, assigned x, then reassigned 3.14)
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
        LocalDecl {
            id: LocalId(1),
            name: Some("m".into()),
            ty: MirType::Unknown,
            is_mut: true,
            span: Span::dummy(),
            is_stack_promoted: false,
            is_param: false,
        },
    ];

    // Block 0:
    // x = 42 (Int)
    // m = 3.14 (Float) -- m starts as Float
    // goto Block 1
    let b0 = BasicBlock {
        id: BlockId(0),
        statements: vec![
            Statement::Assign(
                Place::Local(LocalId(0)),
                Rvalue::Use(Operand::Constant(MirConstant::Int(42))),
                Span::dummy(),
            ),
            Statement::Assign(
                Place::Local(LocalId(1)),
                Rvalue::Use(Operand::Constant(MirConstant::Float(3.14))),
                Span::dummy(),
            ),
        ],
        terminator: Some(Terminator::Goto(BlockId(1), Span::dummy())),
        is_cleanup: false,
    };

    // Block 1:
    // m = x (copy) -- m is reassigned from x!
    // Because m previously had Float, unsound backward copy propagates Float back to x!
    // goto Block 2
    let b1 = BasicBlock {
        id: BlockId(1),
        statements: vec![
            Statement::Assign(
                Place::Local(LocalId(1)),
                Rvalue::Use(Operand::Copy(Place::Local(LocalId(0)))),
                Span::dummy(),
            ),
        ],
        terminator: Some(Terminator::Goto(BlockId(2), Span::dummy())),
        is_cleanup: false,
    };

    // Block 2:
    // return x
    let b2 = BasicBlock {
        id: BlockId(2),
        statements: vec![],
        terminator: Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(LocalId(0)))),
            Span::dummy(),
        )),
        is_cleanup: false,
    };

    let blocks = vec![b0, b1, b2];
    let refined = refine_locals(&locals, &blocks, &MirType::Int);

    // Independent expected truth:
    // Local `m` (LocalId(1)) has conflicting writes, so it is Unknown.
    // BUT local `x` (LocalId(0)) is purely Int(42), never modified, returned as Int.
    // It MUST BE refined to MirType::Int!
    assert_eq!(
        refined[0].ty,
        MirType::Int,
        "DEFECT DETECTED: Local 'x' was poisoned by backward propagation of conflict from multiply-defined mutable local 'm'! Got {:?}",
        refined[0].ty
    );
}

// ============================================================================
// Discriminating RED Test 2: Deep Reverse Copy Chain (>32 hops)
// ============================================================================

/// Defect: Verify that a reverse copy chain of 35 hops (> 32 iterations)
/// reaches fixed point without silently finalizing partial states while changing.
#[test]
fn test_red_deep_reverse_copy_chain_over_32_hops() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    const HOPS: usize = 35;
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

    // In reverse order: x35 = x34; x34 = x33; ... x1 = x0; x0 = 999.
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
        Rvalue::Use(Operand::Constant(MirConstant::Int(999))),
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
            "FIXED POINT FAILURE: Hop {i} was not refined to Int (got {:?}) in >32 reverse copy chain!",
            refined[i].ty
        );
    }
}

// ============================================================================
// Discriminating RED Test 3: Option<float> Source-to-Native Execution
// ============================================================================

/// Defect: Source program constructing Option<float> via `Some(x)` where `x: float`.
/// In current refine_locals:
/// `*existing == MirType::Int && incoming == Enum("Option")` only allows Int!
/// Merging Float with Enum("Option") triggers a Conflict, failing Cranelift compilation!
#[test]
fn test_source_option_native_entry_vs_fallback() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let src = r#"
        function make_some_int(x: int): Option<int>
            return Some(x)
        end

        function make_some_float(x: float): Option<float>
            return Some(x)
        end
    "#;

    let (mut vm_clif, mut vm_bc) = create_vms(src);

    // 1. Proven unboxed Option<int>: executes via native Cranelift entry
    let entries_before = vm_clif.cranelift_entry_count;
    let int_res_clif = vm_clif.call("make_some_int", vec![Value::Int(42)]);
    let int_res_bc = vm_bc.call("make_some_int", vec![Value::Int(42)]);
    assert!(int_res_clif.is_ok(), "make_some_int must succeed natively: {:?}", int_res_clif.err());
    assert_eq!(int_res_clif.as_ref().unwrap(), &Value::some(Value::Int(42)));
    assert_eq!(int_res_clif.unwrap(), int_res_bc.unwrap());
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "make_some_int must execute via native Cranelift entry"
    );

    // 2. Unproven unboxed Option<float>: safely falls back to VM without panic/error
    let entries_before_float = vm_clif.cranelift_entry_count;
    let float_res_clif = vm_clif.call("make_some_float", vec![Value::Float(3.14159)]);
    let float_res_bc = vm_bc.call("make_some_float", vec![Value::Float(3.14159)]);
    assert!(
        float_res_clif.is_ok(),
        "make_some_float safe fallback must succeed! Got: {:?}",
        float_res_clif.err()
    );
    assert_eq!(float_res_clif.as_ref().unwrap(), &Value::some(Value::Float(3.14159)));
    assert_eq!(float_res_clif.unwrap(), float_res_bc.unwrap());
    assert_eq!(
        vm_clif.cranelift_entry_count, entries_before_float,
        "make_some_float must safely fallback to Bytecode VM rather than corrupted native float truncation"
    );
}

// ============================================================================
// Discriminating RED Test 4: Hardcoded "enum.new" Returning Option Conflict with Result
// ============================================================================

/// Defect: A MIR function calls builtin "enum.new" expecting Result<int, string>.
/// Current refine_locals hardcodes:
/// `"enum.new" | "enum.alloc" => Some(MirType::Enum("Option".to_string()))`
/// This assigns `Enum("Option")` to the destination local!
/// When returned from a function returning `MirType::Enum("Result")`,
/// `Option` vs `Result` causes a hard conflict, failing compilation!
#[test]
fn test_red_enum_new_result_not_option() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let mut compiler = CraneliftJitCompiler::new().expect("compiler init");
    let mut m = MirModule::new("mod_enum_new_result".into());

    let sig = FunctionSignature {
        name: "alloc_result_fn".into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Enum("Result".into()),
        is_public: true,
        effects: None,
    };
    let mut f = MirFunction::new(sig, Span::dummy());
    let arg = f.params[0];
    let ret = f.return_local;
    let b1 = f.new_block();

    f.blocks[0].terminator = Some(Terminator::Call {
        callee: "enum.new".into(),
        args: vec![
            Operand::Constant(MirConstant::Int(0)), // tag 0 for Ok
            Operand::Copy(Place::Local(arg)),       // payload
        ],
        destination: Place::Local(ret),
        success: b1,
        error: None,
        span: Span::dummy(),
    });

    f.blocks[b1.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(ret))),
        Span::dummy(),
    ));

    m.add_function(f);

    let comp_res = compiler.compile_module(&m);
    assert!(
        comp_res.is_ok(),
        "DEFECT DETECTED: 'enum.new' was hardcoded to Enum('Option'), conflicting with return type Enum('Result')! Got: {:?}",
        comp_res.err()
    );
}

// ============================================================================
// 5. Unreachable Blocks Audit: Dead Code Does Not Poison Reachable Inference
// ============================================================================

#[test]
fn test_unreachable_block_does_not_poison_reachable_locals() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let mut compiler = CraneliftJitCompiler::new().expect("compiler init");
    let mut m = MirModule::new("mod_unreachable_block".into());

    let sig = FunctionSignature {
        name: "unreachable_block_fn".into(),
        param_types: vec![],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut f = MirFunction::new(sig, Span::dummy());
    let x = f.add_local(Some("x".into()), MirType::Unknown, true, Span::dummy());

    // Block 0 (Entry):
    // x = 42 (Int)
    // return x
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(x),
        Rvalue::Use(Operand::Constant(MirConstant::Int(42))),
        Span::dummy(),
    ));
    f.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(x))),
        Span::dummy(),
    ));

    // Block 1 (Completely unreachable dead code!):
    // x = 3.14 (Float)
    let b_dead = f.new_block();
    f.blocks[b_dead.0].statements.push(Statement::Assign(
        Place::Local(x),
        Rvalue::Use(Operand::Constant(MirConstant::Float(3.14))),
        Span::dummy(),
    ));
    f.blocks[b_dead.0].terminator = Some(Terminator::Unreachable(Span::dummy()));

    m.add_function(f);

    // Because Block 1 is unreachable, its conflicting float assignment must NOT poison `x`.
    // The function must compile natively!
    let comp_res = compiler.compile_module(&m);
    assert!(
        comp_res.is_ok(),
        "Unreachable block poisoned reachable local inference: {:?}",
        comp_res.err()
    );
}

// ============================================================================
// 6. Arg Prefix Audit: Local Variables Named `argc`, `arg_count` are Not Params
// ============================================================================

#[test]
fn test_arg_prefix_audit_local_not_param() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let mut compiler = CraneliftJitCompiler::new().expect("compiler init");
    let mut m = MirModule::new("mod_arg_prefix".into());

    let sig = FunctionSignature {
        name: "arg_prefix_fn".into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut f = MirFunction::new(sig, Span::dummy());
    let p0 = f.params[0]; // Real param
    // Temporary locals named with "arg" prefix that are NOT parameters:
    let argc = f.add_local(Some("argc".into()), MirType::Unknown, true, Span::dummy());
    let arg_max = f.add_local(Some("arg_max".into()), MirType::Unknown, true, Span::dummy());

    // argc = p0 + 10
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(argc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(p0)),
            Operand::Constant(MirConstant::Int(10)),
        ),
        Span::dummy(),
    ));

    // arg_max = argc * 2
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(arg_max),
        Rvalue::BinaryOp(
            MirBinaryOp::Mul,
            Operand::Copy(Place::Local(argc)),
            Operand::Constant(MirConstant::Int(2)),
        ),
        Span::dummy(),
    ));

    f.blocks[0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(arg_max))),
        Span::dummy(),
    ));

    m.add_function(f);

    let comp_res = compiler.compile_module(&m);
    assert!(
        comp_res.is_ok(),
        "Locals with 'arg' prefix failed compilation: {:?}",
        comp_res.err()
    );
}

// ============================================================================
// 7. Loop Consistent & Conflicting Writes
// ============================================================================

#[test]
fn test_loop_consistent_writes_native_execution() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    // Loop computing sum from 0 to 5
    let src = r#"
        function compute_loop_sum(n: int): int
            local sum = 0
            local i = 0
            while i < n do
                sum = sum + i
                i = i + 1
            end
            return sum
        end
    "#;

    let (mut vm_clif, mut vm_bc) = create_vms(src);

    let entries_before = vm_clif.cranelift_entry_count;
    let res_clif = vm_clif.call("compute_loop_sum", vec![Value::Int(5)]).unwrap();
    let res_bc = vm_bc.call("compute_loop_sum", vec![Value::Int(5)]).unwrap();

    // Sum 0+1+2+3+4 = 10
    assert_eq!(res_clif, Value::Int(10));
    assert_eq!(res_clif, res_bc);
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "compute_loop_sum must execute via native Cranelift entry"
    );
}

#[test]
fn test_loop_conflicting_writes_fails_closed() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let mut compiler = CraneliftJitCompiler::new().expect("compiler init");
    let mut m = MirModule::new("mod_loop_conflict".into());

    let sig = FunctionSignature {
        name: "loop_conflict_fn".into(),
        param_types: vec![MirType::Bool],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut f = MirFunction::new(sig, Span::dummy());
    let cond = f.params[0];
    let acc = f.add_local(Some("acc".into()), MirType::Unknown, true, Span::dummy());

    let b_header = f.new_block();
    let b_body = f.new_block();
    let b_exit = f.new_block();

    // Block 0: acc = 0 (Int), goto b_header
    f.blocks[0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[0].terminator = Some(Terminator::Goto(b_header, Span::dummy()));

    // Header: if cond goto b_body else b_exit
    f.blocks[b_header.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond)),
        then_block: b_body,
        else_block: b_exit,
        span: Span::dummy(),
    });

    // Body: acc = 3.14 (Float conflict in loop!), goto b_header
    f.blocks[b_body.0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::Use(Operand::Constant(MirConstant::Float(3.14))),
        Span::dummy(),
    ));
    f.blocks[b_body.0].terminator = Some(Terminator::Goto(b_header, Span::dummy()));

    // Exit: return acc
    f.blocks[b_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(acc))),
        Span::dummy(),
    ));

    m.add_function(f);

    // Compilation MUST fail closed
    let comp_res = compiler.compile_module(&m);
    assert!(
        comp_res.is_err(),
        "Loop with conflicting writes MUST fail closed before native emission!"
    );
}

// ============================================================================
// 8. Branch Consistent & Conflicting Writes
// ============================================================================

#[test]
fn test_branch_consistent_writes_native_execution() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let src = r#"
        function pick_value(flag: bool): int
            local res = 0
            if flag then
                res = 100
            else
                res = 200
            end
            return res + 5
        end
    "#;

    let (mut vm_clif, mut vm_bc) = create_vms(src);

    let entries_before = vm_clif.cranelift_entry_count;
    let true_clif = vm_clif.call("pick_value", vec![Value::Bool(true)]).unwrap();
    let true_bc = vm_bc.call("pick_value", vec![Value::Bool(true)]).unwrap();
    let false_clif = vm_clif.call("pick_value", vec![Value::Bool(false)]).unwrap();
    let false_bc = vm_bc.call("pick_value", vec![Value::Bool(false)]).unwrap();

    assert_eq!(true_clif, Value::Int(105));
    assert_eq!(true_clif, true_bc);
    assert_eq!(false_clif, Value::Int(205));
    assert_eq!(false_clif, false_bc);
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "pick_value must execute via native Cranelift entry"
    );
}

// ============================================================================
// 9. Real Source Option & Result Cases (Nested, Heap String, Ok/Err)
// ============================================================================

#[test]
fn test_source_option_result_nested_and_heap_cases() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    let src = r#"
        function make_ok_int(x: int): Result<int, string>
            return Ok(x)
        end

        function make_some_str(s: string): Option<string>
            return Some(s)
        end

        function make_none_int(): Option<int>
            return Option.None()
        end
    "#;

    let (mut vm_clif, mut vm_bc) = create_vms(src);

    // 1. Proven unboxed Result<int, string> with Ok(int) -> native Cranelift entry
    let entries_before = vm_clif.cranelift_entry_count;
    let ok_clif = vm_clif.call("make_ok_int", vec![Value::Int(777)]).unwrap();
    let ok_bc = vm_bc.call("make_ok_int", vec![Value::Int(777)]).unwrap();
    assert_eq!(ok_clif, Value::ok(Value::Int(777)));
    assert_eq!(ok_clif, ok_bc);
    assert!(
        vm_clif.cranelift_entry_count > entries_before,
        "make_ok_int must execute natively"
    );

    // 2. Proven unboxed Option.None() -> native Cranelift entry
    let entries_before_none = vm_clif.cranelift_entry_count;
    let none_clif = vm_clif.call("make_none_int", vec![]).unwrap();
    let none_bc = vm_bc.call("make_none_int", vec![]).unwrap();
    assert_eq!(none_clif, Value::none());
    assert_eq!(none_clif, none_bc);
    assert!(
        vm_clif.cranelift_entry_count > entries_before_none,
        "make_none_int must execute natively"
    );

    // 3. Option<string> carrying heap string -> safe fallback to VM
    let entries_before_str = vm_clif.cranelift_entry_count;
    let str_clif = vm_clif.call("make_some_str", vec![Value::string("lust-native")]).unwrap();
    let str_bc = vm_bc.call("make_some_str", vec![Value::string("lust-native")]).unwrap();
    assert_eq!(str_clif, Value::some(Value::string("lust-native")));
    assert_eq!(str_clif, str_bc);
    assert_eq!(
        vm_clif.cranelift_entry_count, entries_before_str,
        "Option<string> must safely fallback without corruption"
    );
}

// ============================================================================
// 10. Projections Audit: Struct Field Type Refinement via Struct Layouts
// ============================================================================

#[test]
fn test_projections_struct_field_refinement() {
    use lust::mir::cranelift_backend::jit_engine::refine_locals_with_context;

    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let mut layouts = hashbrown::HashMap::new();
    layouts.insert(
        "Point".to_string(),
        MirStructLayout {
            type_id: 1,
            type_name: "Point".to_string(),
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
                    tag: CanonicalTypeTag::Float64,
                    byte_offset: 8,
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
            ty: MirType::Struct("Point".into()),
            is_mut: false,
            span: Span::dummy(),
            is_stack_promoted: false,
            is_param: false,
        },
        LocalDecl {
            id: LocalId(1),
            name: Some("field_x".into()),
            ty: MirType::Unknown,
            is_mut: true,
            span: Span::dummy(),
            is_stack_promoted: false,
            is_param: false,
        },
        LocalDecl {
            id: LocalId(2),
            name: Some("field_y".into()),
            ty: MirType::Unknown,
            is_mut: true,
            span: Span::dummy(),
            is_stack_promoted: false,
            is_param: false,
        },
    ];

    let b0 = BasicBlock {
        id: BlockId(0),
        statements: vec![
            Statement::Assign(
                Place::Local(LocalId(1)),
                Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(LocalId(0))), 0))),
                Span::dummy(),
            ),
            Statement::Assign(
                Place::Local(LocalId(2)),
                Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(LocalId(0))), 1))),
                Span::dummy(),
            ),
        ],
        terminator: None,
        is_cleanup: false,
    };

    let refined = refine_locals_with_context(&locals, &[b0], &MirType::Unit, &[], &layouts);

    assert_eq!(
        refined[1].ty,
        MirType::Int,
        "Field 0 projection should refine to Int"
    );
    assert_eq!(
        refined[2].ty,
        MirType::Float,
        "Field 1 projection should refine to Float"
    );
}

// ============================================================================
// 11. Deep Copy Chain: 45 Hops Fixed Point Convergence
// ============================================================================

#[test]
fn test_reverse_copy_chain_over_40_hops() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    const HOPS: usize = 45;
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
        Rvalue::Use(Operand::Constant(MirConstant::Int(12345))),
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
            "45-hop chain: hop {i} failed to refine to Int"
        );
    }
}


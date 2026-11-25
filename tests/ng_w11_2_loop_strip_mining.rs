#![cfg(feature = "std")]

//! NG-W11-2 Verification Suite:
//! 1. 2-instruction promoted inline gas check (`iadd_imm` + `icmp_imm`, zero `ssub_overflow`).
//! 2. Strictly verified atomic loop header quantization (`emit_quantized_loop_header_check`, Q = 64, SEC-01 sound).
//! 3. F-06 / F-20 safety: loops with >1 `Statement::ResourceCheck(ResourceKind::Gas, ..)` or
//!    nested loop headers disable quantization fail-closed and never overrun gas or violate `rem0 >= 0`.
//! 4. Exact `test-deopt` ordinal precision across quantum boundaries (`1, 2, 63, 64, 65, 128, 129`).

use std::sync::Mutex;

use lust::ast::Span;
use lust::bytecode::Value;
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{
    reset_unbounded_gas_replenish_count, safepoint_reset, safepoint_stats,
    set_unbounded_gas_chunk_override, unbounded_gas_replenish_count, CraneliftJitCompiler,
};
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::safepoint_request_deopt_on_poll;
use lust::mir::ir::{
    BasicBlock, BlockId, FunctionSignature, LocalDecl, LocalId, MirBinaryOp, MirConstant,
    MirFunction, MirModule, MirType, Operand, Place, ResourceKind, Rvalue, Statement, Terminator,
};
use lust::mir::lowering::AstLowering;
use lust::mir::verifier::MirVerifier;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{VmBackend, VM};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn parse_and_lower_module(source: &str, module_name: &str) -> MirModule {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("Tokenization must succeed");
    let items = Parser::new(tokens).parse().expect("Parsing must succeed");

    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);

    AstLowering::lower_module_items(module_name, &items).expect("MIR module lowering must succeed")
}

fn push_local(
    f: &mut MirFunction,
    name: &str,
    ty: MirType,
    is_mut: bool,
    is_param: bool,
) -> LocalId {
    let id = LocalId(f.locals.len());
    let mut decl = LocalDecl::new(id, Some(name.into()), ty, is_mut, Span::dummy());
    decl.is_param = is_param;
    f.locals.push(decl);
    if is_param {
        f.params.push(id);
    }
    id
}

fn push_block(f: &mut MirFunction) -> BlockId {
    let id = BlockId(f.blocks.len());
    f.blocks.push(BasicBlock::new(id));
    id
}

fn build_empty_fn(
    name: &str,
    param_types: Vec<MirType>,
    return_type: MirType,
) -> (MirFunction, Vec<LocalId>) {
    let mut f = MirFunction::new(
        FunctionSignature {
            name: name.into(),
            param_types: param_types.clone(),
            return_type,
            effects: None,
            is_public: true,
        },
        Span::dummy(),
    );
    f.locals.clear();
    f.params.clear();
    let mut param_ids = Vec::with_capacity(param_types.len());
    for (idx, ty) in param_types.into_iter().enumerate() {
        let id = push_local(&mut f, &format!("p{idx}"), ty, false, true);
        param_ids.push(id);
    }
    (f, param_ids)
}

/// Builds a simple loop with `Statement::ResourceCheck(ResourceKind::Gas, amt, _)` at `hdr.statements[0]`
/// and optional second `Statement::ResourceCheck(ResourceKind::Gas, second_amt, _)` in the loop body
/// (modeling an inlined callee that carries its own gas check, F-06).
fn build_loop_fn(name: &str, header_amt: u64, body_extra_amt: Option<u64>) -> MirFunction {
    let (mut f, params) = build_empty_fn(name, vec![MirType::Int], MirType::Int);
    let n = params[0];
    let i = push_local(&mut f, "i", MirType::Int, true, false);
    let acc = push_local(&mut f, "acc", MirType::Int, true, false);
    let cond = push_local(&mut f, "cond", MirType::Bool, true, false);

    let bb0 = f.entry;
    let bb_hdr = push_block(&mut f);
    let bb_body = push_block(&mut f);
    let bb_exit = push_block(&mut f);

    f.blocks[bb0.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[bb0.0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    f.blocks[bb0.0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    f.blocks[bb_hdr.0]
        .statements
        .push(Statement::ResourceCheck(
            ResourceKind::Gas,
            header_amt,
            Span::dummy(),
        ));
    f.blocks[bb_hdr.0].statements.push(Statement::Assign(
        Place::Local(cond),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(i)),
            Operand::Copy(Place::Local(n)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_hdr.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond)),
        then_block: bb_body,
        else_block: bb_exit,
        span: Span::dummy(),
    });

    if let Some(extra_amt) = body_extra_amt {
        f.blocks[bb_body.0]
            .statements
            .push(Statement::ResourceCheck(
                ResourceKind::Gas,
                extra_amt,
                Span::dummy(),
            ));
    }
    f.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(acc)),
            Operand::Copy(Place::Local(i)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_body.0].statements.push(Statement::Assign(
        Place::Local(i),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    f.blocks[bb_body.0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

    f.blocks[bb_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(acc))),
        Span::dummy(),
    ));
    f
}

// ============================================================================
// 1. IR CONTRACT: Zero `ssub_overflow` + Quantum Strip-Mining Gating
// ============================================================================

#[test]
fn test_ng_w11_2_ir_no_ssub_overflow_and_quantum_gating() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    // (a) Simple single-gas loop: NO ssub_overflow AND arms Q=64 (63 fast iterations).
    let simple_src = r#"
        function sum_loop(n: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                acc = acc + i
                i = i + 1
            end
            return acc
        end
    "#;
    let simple_mod = parse_and_lower_module(simple_src, "ng_w11_2_simple_ir");
    let mut compiler = CraneliftJitCompiler::new().expect("create compiler");
    let simple_ir = compiler
        .dump_function_ir(&simple_mod.functions[0])
        .expect("dump sum_loop IR");
    assert!(
        !simple_ir.contains("ssub_overflow"),
        "Promoted/quantized loop must not emit ssub_overflow:\n{simple_ir}"
    );
    assert!(
        simple_ir.contains("iconst.i64 63"),
        "Simple single-gas loop must arm Q=64 quantum (iconst.i64 63):\n{simple_ir}"
    );

    // (b) Multi-gas-check loop (F-06): NO ssub_overflow (Stage 1 2-inst check),
    //     and MUST NOT arm Q=64 quantum (`quantized_header_gas_amt == None`).
    let multi_gas_fn = build_loop_fn("multi_gas_loop", 1, Some(1));
    let mut multi_mod = MirModule::new("ng_w11_2_multi_gas_ir".into());
    multi_mod.add_function(multi_gas_fn.clone());
    MirVerifier::verify_module(&multi_mod).expect("verify multi_gas_loop");
    let mut multi_compiler = CraneliftJitCompiler::new().expect("create compiler");
    let multi_ir = multi_compiler
        .dump_function_ir(&multi_gas_fn)
        .expect("dump multi_gas_loop IR");
    assert!(
        !multi_ir.contains("ssub_overflow"),
        "Promoted multi-gas loop must use 2-instruction iadd_imm, never ssub_overflow:\n{multi_ir}"
    );
    assert!(
        !multi_ir.contains("iconst.i64 63"),
        "Loop with >1 ResourceCheck(Gas) MUST disable quantization (F-06):\n{multi_ir}"
    );

    // (c) Nested loop (F-20): NO ssub_overflow, and MUST NOT arm Q=64 quantum.
    let nested_src = r#"
        function nested_sum(outer: int, inner: int): int
            local i: int = 0
            local total: int = 0
            while i < outer do
                local j: int = 0
                while j < inner do
                    total = total + 1
                    j = j + 1
                end
                i = i + 1
            end
            return total
        end
    "#;
    let nested_mod = parse_and_lower_module(nested_src, "ng_w11_2_nested_ir");
    let mut nested_compiler = CraneliftJitCompiler::new().expect("create compiler");
    let nested_ir = nested_compiler
        .dump_function_ir(&nested_mod.functions[0])
        .expect("dump nested_sum IR");
    assert!(
        !nested_ir.contains("ssub_overflow"),
        "Promoted nested loop must not emit ssub_overflow:\n{nested_ir}"
    );
    assert!(
        !nested_ir.contains("iconst.i64 63"),
        "Nested loop containing >1 loop header MUST disable quantization (F-20):\n{nested_ir}"
    );

    // (d) Peeled-header hoisted array loop: NO ssub_overflow AND arms Q=64 quantum.
    let arr_src = r#"
        function sum_arr(arr: Array<int>, n: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                acc = acc + arr[i]
                i = i + 1
            end
            return acc
        end
    "#;
    let arr_mod = parse_and_lower_module(arr_src, "ng_w11_2_arr_ir");
    let mut arr_compiler = CraneliftJitCompiler::new().expect("create compiler");
    let arr_ir = arr_compiler
        .dump_function_ir(&arr_mod.functions[0])
        .expect("dump sum_arr IR");
    assert!(
        !arr_ir.contains("ssub_overflow"),
        "Peeled-header quantized loop must not emit ssub_overflow:\n{arr_ir}"
    );
    assert!(
        arr_ir.contains("iconst.i64 63"),
        "Peeled-header single-gas loop must arm Q=64 quantum:\n{arr_ir}"
    );
}

// ============================================================================
// 2. REGRESSION TEST F-06: Multi-Gas-Check Loop Safety (No Overrun, rem0 >= 0)
// ============================================================================

#[test]
fn test_ng_w11_2_f06_multi_gas_check_loop_safety() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    safepoint_reset();

    // Two ResourceCheck(Gas, 1) per iteration (1 at header + 1 in body, simulating inlined call).
    let multi_fn = build_loop_fn("multi_gas_inlined_loop", 1, Some(1));
    let mut module = MirModule::new("ng_w11_2_f06_mod".into());
    module.add_function(multi_fn);
    MirVerifier::verify_module(&module).expect("verify multi_gas_inlined_loop");

    // Verify exact OutOfGas at gas_limit = 50 -> gas_used == 51 (strictly zero overrun).
    {
        let mut vm = VM::new();
        vm.load_mir_module(&module).expect("load module");
        vm.set_backend(VmBackend::Cranelift);
        vm.set_gas_budget(50);

        let err = vm
            .call("multi_gas_inlined_loop", vec![Value::Int(100)])
            .expect_err("Multi-gas loop must fail OutOfGas at budget=50");
        assert!(
            err.to_string().contains("Out of gas"),
            "Expected Out of gas error, got: {err}"
        );
        assert_eq!(
            vm.gas_used(),
            51,
            "F-06: multi-gas loop must stop at gas_used == 51 with zero overrun"
        );
        assert_eq!(vm.gas_remaining(), Some(0));
        assert!(
            vm.mir_context.gas_remaining() >= -1,
            "rem0 >= 0 invariant must keep post-OOG gas_remaining >= -1, got {}",
            vm.mir_context.gas_remaining()
        );
    }

    // Sweep budgets across Q=64 boundaries (odd & even limits): exact zero-overrun parity.
    for budget in [1u64, 2, 3, 50, 63, 64, 65, 127, 128, 129, 130] {
        safepoint_reset();
        let mut vm = VM::new();
        vm.load_mir_module(&module).expect("load module");
        vm.set_backend(VmBackend::Cranelift);
        vm.set_gas_budget(budget);

        let err = vm
            .call("multi_gas_inlined_loop", vec![Value::Int(500)])
            .expect_err("Multi-gas loop must hit OutOfGas");
        assert!(err.to_string().contains("Out of gas"), "budget={budget}: {err}");
        assert_eq!(
            vm.gas_used(),
            budget + 1,
            "F-06 overrun at budget={budget}: expected gas_used={}, got {}",
            budget + 1,
            vm.gas_used()
        );
        assert_eq!(vm.gas_remaining(), Some(0));
        assert!(
            vm.mir_context.gas_remaining() >= -1,
            "F-06 rem0 >= 0 violation at budget={budget}: {}",
            vm.mir_context.gas_remaining()
        );
    }
}

// ============================================================================
// 3. SIMPLE QUANTIZED LOOP: Exact SEC-01 Parity, Quantum Boundaries & Unbounded
// ============================================================================

#[test]
fn test_ng_w11_2_simple_loop_sec01_exact_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let src = r#"
        function simple_sum(n: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                acc = acc + i
                i = i + 1
            end
            return acc
        end
    "#;
    let module = parse_and_lower_module(src, "ng_w11_2_simple_parity");

    // (a) gas_limit = 50 -> OutOfGas at exactly 51 units consumed.
    {
        safepoint_reset();
        let mut vm = VM::new();
        vm.load_mir_module(&module).expect("load module");
        vm.set_backend(VmBackend::Cranelift);
        vm.set_gas_budget(50);

        let err = vm
            .call("simple_sum", vec![Value::Int(100)])
            .expect_err("budget=50 with n=100 must fail OutOfGas");
        assert!(err.to_string().contains("Out of gas"), "unexpected err: {err}");
        assert_eq!(vm.gas_used(), 51);
        assert_eq!(vm.gas_remaining(), Some(0));
        assert_eq!(safepoint_stats().safepoints_checked, 50);
    }

    // (b) Exact quantum boundaries (around Q = 64, 128, 192): zero overrun!
    for budget in [1u64, 2, 50, 62, 63, 64, 65, 66, 127, 128, 129, 130, 192, 193] {
        safepoint_reset();
        let mut vm = VM::new();
        vm.load_mir_module(&module).expect("load module");
        vm.set_backend(VmBackend::Cranelift);
        vm.set_gas_budget(budget);

        let err = vm
            .call("simple_sum", vec![Value::Int(500)])
            .expect_err("must hit OutOfGas");
        assert!(err.to_string().contains("Out of gas"), "budget={budget}: {err}");
        assert_eq!(
            vm.gas_used(),
            budget + 1,
            "SEC-01 exact OOG boundary failed at budget={budget}"
        );
        assert_eq!(vm.gas_remaining(), Some(0));
        // 1 VM entry charge + `budget` header checks = `budget + 1` total gas -> `budget` safepoints.
        assert_eq!(
            safepoint_stats().safepoints_checked,
            budget,
            "Exact safepoints_checked mismatch at budget={budget}"
        );
    }

    // (c) Large quantized loop completion: exact gas_remaining and safepoints_checked.
    //     n = 998 executes 999 loop headers (+ 1 VM entry charge = 1000 total VM gas).
    //     With budget = 1001 (1000 budget remaining on loop entry): gas_remaining == 1, safepoints == 999.
    {
        safepoint_reset();
        let mut vm = VM::new();
        vm.load_mir_module(&module).expect("load module");
        vm.set_backend(VmBackend::Cranelift);
        vm.set_gas_budget(1001);

        let res = vm
            .call("simple_sum", vec![Value::Int(998)])
            .expect("998 iterations must succeed within budget 1001");
        assert_eq!(res, Value::Int(998 * 997 / 2));
        assert_eq!(vm.gas_remaining(), Some(1));
        assert_eq!(vm.gas_used(), 1000);
        assert_eq!(safepoint_stats().safepoints_checked, 999);
    }

    //     With budget = 1000 and n = 997: gas_remaining == 1, safepoints_checked == 998.
    {
        safepoint_reset();
        let mut vm = VM::new();
        vm.load_mir_module(&module).expect("load module");
        vm.set_backend(VmBackend::Cranelift);
        vm.set_gas_budget(1000);

        let res = vm
            .call("simple_sum", vec![Value::Int(997)])
            .expect("997 iterations must succeed within budget 1000");
        assert_eq!(res, Value::Int(997 * 996 / 2));
        assert_eq!(vm.gas_remaining(), Some(1));
        assert_eq!(vm.gas_used(), 999);
        assert_eq!(safepoint_stats().safepoints_checked, 998);
    }

    // (d) Non-unit quantized header gas amount (amt = 5, arm threshold = 63 * 5 = 315):
    //     verify exact OOG boundary across threshold without overrun.
    {
        let amt5_fn = build_loop_fn("quantized_amt5_loop", 5, None);
        let mut amt5_mod = MirModule::new("ng_w11_2_amt5_mod".into());
        amt5_mod.add_function(amt5_fn);
        MirVerifier::verify_module(&amt5_mod).expect("verify quantized_amt5_loop");

        for budget in [315u64, 316, 320, 321, 325, 326, 640, 641] {
            safepoint_reset();
            let mut vm = VM::new();
            vm.load_mir_module(&amt5_mod).expect("load amt5_mod");
            vm.set_backend(VmBackend::Cranelift);
            vm.set_gas_budget(budget);

            let err = vm
                .call("quantized_amt5_loop", vec![Value::Int(500)])
                .expect_err("amt5 loop must fail OutOfGas");
            assert!(err.to_string().contains("Out of gas"), "budget={budget}: {err}");
            // 1 entry charge leaves (budget - 1) for loop; each header charges 5.
            let loop_budget = budget - 1;
            let expected_headers = (loop_budget / 5) + 1;
            assert_eq!(
                safepoint_stats().safepoints_checked,
                expected_headers,
                "amt=5 header count mismatch at budget={budget}"
            );
            assert_eq!(
                vm.gas_used(),
                1 + expected_headers * 5,
                "amt=5 exact gas_used mismatch at budget={budget}"
            );
            assert_eq!(vm.gas_remaining(), Some(0));
        }
    }

    // (e) Unbounded mode with small replenishment chunk (chunk = 64) over 500 iterations.
    {
        safepoint_reset();
        let mut vm = VM::new();
        vm.load_mir_module(&module).expect("load module");
        vm.set_backend(VmBackend::Cranelift);

        reset_unbounded_gas_replenish_count();
        let _chunk_guard = set_unbounded_gas_chunk_override(Some(64));
        let res = vm
            .call("simple_sum", vec![Value::Int(500)])
            .expect("Unbounded quantized loop with chunk=64 must succeed");
        assert_eq!(res, Value::Int(500 * 499 / 2));
        assert!(
            unbounded_gas_replenish_count() >= 7,
            "Expected >= 7 replenishments, got {}",
            unbounded_gas_replenish_count()
        );
        assert_eq!(safepoint_stats().safepoints_checked, 501);
    }
}

// ============================================================================
// 4. TEST-DEOPT: Exact Deopt at Ordinals 1, 2, 63, 64, 65, 128, 129
// ============================================================================

#[cfg(feature = "test-deopt")]
#[test]
fn test_ng_w11_2_test_deopt_exact_ordinals_across_quantum_boundaries() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let src = r#"
        function simple_sum(n: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                acc = acc + i
                i = i + 1
            end
            return acc
        end

        function arr_sum(arr: Array<int>, n: int): int
            local i: int = 0
            local acc: int = 0
            while i < n do
                acc = acc + arr[i]
                i = i + 1
            end
            return acc
        end
    "#;
    let module = parse_and_lower_module(src, "ng_w11_2_deopt_mod");

    let ordinals = [1u64, 2, 63, 64, 65, 128, 129];

    // (a) Non-peeled quantized loop (`simple_sum`)
    //     Use `n = ordinal + 10` so the post-deopt interpreter tail (11 iterations) stays below
    //     the OSR re-entry threshold, letting us assert exact `safepoints_checked == ordinal`.
    for ordinal in ordinals {
        let n = (ordinal + 10) as i64;
        let expected_sum = n * (n - 1) / 2;
        safepoint_reset();
        let mut vm = VM::new();
        vm.load_mir_module(&module).expect("load deopt_mod");
        vm.set_backend(VmBackend::Cranelift);
        vm.set_gas_budget(10_000);
        safepoint_request_deopt_on_poll(None, ordinal);

        let res = vm
            .call("simple_sum", vec![Value::Int(n)])
            .expect("deopt + interpreter resume must succeed");
        assert_eq!(
            res,
            Value::Int(expected_sum),
            "simple_sum result mismatch after deopt at ordinal {ordinal}"
        );
        assert!(vm.cranelift_entry_count > 0);
        assert_eq!(
            safepoint_stats().safepoints_checked,
            ordinal,
            "simple_sum must deopt at EXACT safepoint ordinal {ordinal}"
        );
    }

    // (b) Peeled-header hoisted quantized loop (`arr_sum`)
    for ordinal in ordinals {
        let n = (ordinal + 10) as i64;
        let expected_sum = n * (n - 1) / 2;
        safepoint_reset();
        let mut vm = VM::new();
        vm.load_mir_module(&module).expect("load deopt_mod");
        vm.set_backend(VmBackend::Cranelift);
        vm.set_gas_budget(10_000);
        safepoint_request_deopt_on_poll(None, ordinal);

        let arr = Value::array((0..n).map(Value::Int).collect());
        let res = vm
            .call("arr_sum", vec![arr, Value::Int(n)])
            .expect("peeled-header deopt + interpreter resume must succeed");
        assert_eq!(
            res,
            Value::Int(expected_sum),
            "arr_sum result mismatch after deopt at ordinal {ordinal}"
        );
        assert!(vm.cranelift_entry_count > 0);
        assert_eq!(
            safepoint_stats().safepoints_checked,
            ordinal,
            "arr_sum must deopt at EXACT safepoint ordinal {ordinal}"
        );
    }

    // (c) Long-tail deopt + OSR re-entry parity across all quantum boundary ordinals (`n = 200`)
    for ordinal in ordinals {
        let n = 200i64;
        let expected_sum = n * (n - 1) / 2;
        safepoint_reset();
        let mut vm = VM::new();
        vm.load_mir_module(&module).expect("load deopt_mod");
        vm.set_backend(VmBackend::Cranelift);
        vm.set_gas_budget(10_000);
        safepoint_request_deopt_on_poll(None, ordinal);

        let res = vm
            .call("simple_sum", vec![Value::Int(n)])
            .expect("deopt + interpreter + OSR resume must succeed");
        assert_eq!(
            res,
            Value::Int(expected_sum),
            "simple_sum(200) result mismatch after deopt at ordinal {ordinal}"
        );
        assert!(vm.cranelift_entry_count > 0);
    }

    safepoint_reset();
}

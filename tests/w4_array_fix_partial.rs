//! Regression suite for TASK-W4-ARRAY-FIX-PARTIAL (F1, F2, F4, F5, F6, F7).
//! Intentionally separate from `tests/w4_jit_simd.rs` to avoid concurrent edit conflicts.

use std::rc::Rc;
use std::sync::Mutex;

use lust::ast::Span;
use lust::bytecode::{LustArray, Value};
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::*;
use lust::mir::ir::*;
use lust::mir::lowering::AstLowering;
use lust::mir::runtime_helpers::{
    enter_mir_context, lust_mir_alloc_array, lust_mir_alloc_array_typed, MirContext,
    MIR_ARRAY_ELEM_TAG_INT,
};
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{VmBackend, VM};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn setup_vm_from_source(source: &str, module_name: &str) -> VM {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("Tokenization must succeed");
    let items = Parser::new(tokens).parse().expect("Parsing must succeed");

    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let struct_defs = checker.struct_definitions();

    let mut module = MirModule::new(module_name.into());
    for item in &items {
        if let lust::ast::ItemKind::Function(func_def) = &item.kind {
            let mir_fn = AstLowering::lower_function(func_def).expect("MIR lowering must succeed");
            module.add_function(mir_fn);
        }
    }

    let mut vm = VM::new();
    vm.set_backend(VmBackend::Cranelift);
    vm.load_mir_module(&module).expect("Load MIR module");
    vm.register_structs(&struct_defs);
    vm
}

// =========================================================================
// F1: Terminal exits (Gas & Memory budget exhaustion) preserve prefix writes
// =========================================================================

#[test]
fn test_w4_partial_f1_terminal_gas_and_oom_preserve_prefix_writes() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    // Case 1: Finite Gas Exhaustion after array write
    {
        let fn_name = "write_then_oog";
        let sig = FunctionSignature {
            name: fn_name.into(),
            param_types: vec![MirType::Array(Box::new(MirType::Int))],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut func = MirFunction::new(sig, Span::dummy());
        let a = func.params[0];
        let zero = func.add_local(Some("zero".into()), MirType::Int, false, Span::dummy());
        let seven = func.add_local(Some("seven".into()), MirType::Int, false, Span::dummy());
        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(zero),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));
        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(seven),
            Rvalue::Use(Operand::Constant(MirConstant::Int(7))),
            Span::dummy(),
        ));
        func.blocks[0].statements.push(Statement::Assign(
            Place::Index(Box::new(Place::Local(a)), zero),
            Rvalue::Use(Operand::Copy(Place::Local(seven))),
            Span::dummy(),
        ));
        func.blocks[0].statements.push(Statement::ResourceCheck(
            ResourceKind::Gas,
            100,
            Span::dummy(),
        ));
        func.blocks[0].terminator = Some(Terminator::Return(
            Some(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));

        let mut vm = VM::new();
        vm.set_backend(VmBackend::Cranelift);
        vm.load_mir_function(func).expect("Load write_then_oog");
        vm.set_gas_budget(10);

        let arr = Rc::new(LustArray::new(vec![Value::Int(3)]));
        let entry_before = vm.cranelift_entry_count;
        let res = vm.call(fn_name, vec![Value::Array(arr.clone())]);

        assert_eq!(vm.cranelift_entry_count, entry_before + 1);
        let err = res.expect_err("Expected Out of gas error");
        assert!(
            err.to_string().contains("Out of gas"),
            "Expected Out of gas error, got: {err}"
        );
        assert_eq!(
            arr.borrow()[0],
            Value::Int(7),
            "Completed prefix write a[0]=7 must survive OutOfGas terminal exit"
        );
        assert_eq!(vm.mir_context.allocated_buffers.lock().unwrap().len(), 0);
        assert_eq!(vm.mir_context.allocation_records.lock().unwrap().len(), 0);
        assert_eq!(vm.call_stack_depth(), 0);
    }

    // Case 2: Finite Memory Budget Exhaustion (native Aggregate allocation OOM) after array write
    {
        let fn_name = "write_then_oom";
        let sig = FunctionSignature {
            name: fn_name.into(),
            param_types: vec![MirType::Array(Box::new(MirType::Int))],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut func = MirFunction::new(sig, Span::dummy());
        let a = func.params[0];
        let zero = func.add_local(Some("zero".into()), MirType::Int, false, Span::dummy());
        let seven = func.add_local(Some("seven".into()), MirType::Int, false, Span::dummy());
        let tmp_arr = func.add_local(
            Some("tmp_arr".into()),
            MirType::Array(Box::new(MirType::Int)),
            false,
            Span::dummy(),
        );
        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(zero),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));
        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(seven),
            Rvalue::Use(Operand::Constant(MirConstant::Int(7))),
            Span::dummy(),
        ));
        func.blocks[0].statements.push(Statement::Assign(
            Place::Index(Box::new(Place::Local(a)), zero),
            Rvalue::Use(Operand::Copy(Place::Local(seven))),
            Span::dummy(),
        ));
        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(tmp_arr),
            Rvalue::Aggregate(
                MirType::Array(Box::new(MirType::Int)),
                vec![
                    Operand::Constant(MirConstant::Int(1)),
                    Operand::Constant(MirConstant::Int(2)),
                    Operand::Constant(MirConstant::Int(3)),
                    Operand::Constant(MirConstant::Int(4)),
                ],
            ),
            Span::dummy(),
        ));
        func.blocks[0].terminator = Some(Terminator::Return(
            Some(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));

        let mut vm = VM::new();
        vm.set_backend(VmBackend::Cranelift);
        vm.load_mir_function(func).expect("Load write_then_oom");
        vm.set_memory_budget_bytes(16).expect("Set memory budget");

        let arr = Rc::new(LustArray::new(vec![Value::Int(3)]));
        let entry_before = vm.cranelift_entry_count;
        let res = vm.call(fn_name, vec![Value::Array(arr.clone())]);

        assert_eq!(vm.cranelift_entry_count, entry_before + 1);
        let err = res.expect_err("Expected Out of memory error");
        assert!(
            err.to_string().contains("Out of memory") || err.to_string().contains("memory"),
            "Expected OOM error, got: {err}"
        );
        assert_eq!(
            arr.borrow()[0],
            Value::Int(7),
            "Completed prefix write a[0]=7 must survive OOM terminal exit"
        );
        assert_eq!(vm.mir_context.allocated_buffers.lock().unwrap().len(), 0);
        assert_eq!(vm.mir_context.allocation_records.lock().unwrap().len(), 0);
        assert_eq!(vm.call_stack_depth(), 0);
    }

    // Case 3: Terminal writeback failure (aliased borrow active) must NOT be silently ignored
    {
        let fn_name = "write_then_oog_borrow";
        let sig = FunctionSignature {
            name: fn_name.into(),
            param_types: vec![MirType::Array(Box::new(MirType::Int))],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut func = MirFunction::new(sig, Span::dummy());
        let a = func.params[0];
        let zero = func.add_local(Some("zero".into()), MirType::Int, false, Span::dummy());
        let seven = func.add_local(Some("seven".into()), MirType::Int, false, Span::dummy());
        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(zero),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));
        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(seven),
            Rvalue::Use(Operand::Constant(MirConstant::Int(7))),
            Span::dummy(),
        ));
        func.blocks[0].statements.push(Statement::Assign(
            Place::Index(Box::new(Place::Local(a)), zero),
            Rvalue::Use(Operand::Copy(Place::Local(seven))),
            Span::dummy(),
        ));
        func.blocks[0].statements.push(Statement::ResourceCheck(
            ResourceKind::Gas,
            100,
            Span::dummy(),
        ));
        func.blocks[0].terminator = Some(Terminator::Return(
            Some(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));

        let mut vm = VM::new();
        vm.set_backend(VmBackend::Cranelift);
        vm.load_mir_function(func).expect("Load");
        vm.set_gas_budget(10);

        let arr = Rc::new(LustArray::new(vec![Value::Int(3)]));
        let _shared = arr.borrow();
        let err = vm
            .call(fn_name, vec![Value::Array(arr.clone())])
            .expect_err("Expected error when terminal writeback fails due to active borrow");
        let msg = err.to_string();
        assert!(
            msg.contains("aliased borrow active") || msg.contains("writeback failed"),
            "Terminal writeback error must NOT be silently ignored, got: {msg}"
        );
        assert_eq!(vm.mir_context.allocated_buffers.lock().unwrap().len(), 0);
        assert_eq!(vm.call_stack_depth(), 0);
    }
}

// =========================================================================
// F2: Deopt preserves alias identity for non-input fresh native arrays
// =========================================================================

#[test]
#[cfg(feature = "test-deopt")]
fn test_w4_partial_f2_deopt_fresh_and_input_array_alias_preservation() {
    use lust::vm::DeoptPayloadSource;

    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    safepoint_reset();
    reset_native_instrumentation_counters();
    take_deopt_payload();

    // Case 1: Fresh native array allocated in bb0, aliased `b = a`, deopt at loop header BlockEntry
    {
        let fn_name = "deopt_fresh_array_alias_loop";
        let sig = FunctionSignature {
            name: fn_name.into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut func = MirFunction::new(sig, Span::dummy());
        let n = func.params[0];
        let a = func.add_local(
            Some("a".into()),
            MirType::Array(Box::new(MirType::Int)),
            false,
            Span::dummy(),
        );
        let b = func.add_local(
            Some("b".into()),
            MirType::Array(Box::new(MirType::Int)),
            false,
            Span::dummy(),
        );
        let i = func.add_local(Some("i".into()), MirType::Int, false, Span::dummy());
        let zero = func.add_local(Some("zero".into()), MirType::Int, false, Span::dummy());
        let seven = func.add_local(Some("seven".into()), MirType::Int, false, Span::dummy());
        let cond = func.add_local(Some("cond".into()), MirType::Bool, false, Span::dummy());
        let ret = func.add_local(Some("ret".into()), MirType::Int, false, Span::dummy());

        let bb_hdr = func.new_block();
        let bb_body = func.new_block();
        let bb_exit = func.new_block();

        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(a),
            Rvalue::Aggregate(
                MirType::Array(Box::new(MirType::Int)),
                vec![Operand::Constant(MirConstant::Int(3))],
            ),
            Span::dummy(),
        ));
        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(b),
            Rvalue::Use(Operand::Copy(Place::Local(a))),
            Span::dummy(),
        ));
        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(i),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));
        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(zero),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));
        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(seven),
            Rvalue::Use(Operand::Constant(MirConstant::Int(7))),
            Span::dummy(),
        ));
        func.blocks[0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

        func.blocks[bb_hdr.0].statements.push(Statement::Assign(
            Place::Local(cond),
            Rvalue::BinaryOp(
                MirBinaryOp::Lt,
                Operand::Copy(Place::Local(i)),
                Operand::Copy(Place::Local(n)),
            ),
            Span::dummy(),
        ));
        func.blocks[bb_hdr.0].terminator = Some(Terminator::Cond {
            cond: Operand::Copy(Place::Local(cond)),
            then_block: bb_body,
            else_block: bb_exit,
            span: Span::dummy(),
        });

        func.blocks[bb_body.0].statements.push(Statement::Assign(
            Place::Index(Box::new(Place::Local(a)), zero),
            Rvalue::Use(Operand::Copy(Place::Local(seven))),
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

        func.blocks[bb_exit.0].statements.push(Statement::Assign(
            Place::Local(ret),
            Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(b)), zero))),
            Span::dummy(),
        ));
        func.blocks[bb_exit.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(ret))),
            Span::dummy(),
        ));

        let mut vm = VM::new();
        vm.set_jit_backend(VmBackend::Cranelift);
        vm.load_mir_function(func).expect("Load deopt_fresh_array_alias_loop");
        vm.test_deopt_spike_hook = Some(lust::vm::TestDeoptSpikeHook::new(fn_name, bb_hdr.0));
        safepoint_request_deopt_on_poll(Some(fn_name.into()), 1);

        let entry_before = vm.cranelift_entry_count;
        let res = vm
            .call(fn_name, vec![Value::Int(1)])
            .expect("VM::call deopt continuation");

        assert_eq!(
            vm.cranelift_entry_count,
            entry_before + 1,
            "No post-native replay allowed"
        );
        assert_eq!(
            vm.last_deopt_source,
            Some(DeoptPayloadSource::NativeExecution),
            "Must use genuine NativeExecution deopt"
        );
        assert_eq!(
            res,
            Value::Int(7),
            "Fresh native array alias b=a must preserve identity across BlockEntry deopt"
        );
        assert_eq!(vm.mir_context.allocated_buffers.lock().unwrap().len(), 0);
        assert_eq!(vm.call_stack_depth(), 0);
    }

    // Case 2: Input array parameter `a` copied to local `b = a` across BlockEntry deopt
    {
        safepoint_reset();
        take_deopt_payload();

        let fn_name = "deopt_input_array_alias_loop";
        let sig = FunctionSignature {
            name: fn_name.into(),
            param_types: vec![MirType::Array(Box::new(MirType::Int)), MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut func = MirFunction::new(sig, Span::dummy());
        let a = func.params[0];
        let n = func.params[1];
        let b = func.add_local(
            Some("b".into()),
            MirType::Array(Box::new(MirType::Int)),
            false,
            Span::dummy(),
        );
        let i = func.add_local(Some("i".into()), MirType::Int, false, Span::dummy());
        let zero = func.add_local(Some("zero".into()), MirType::Int, false, Span::dummy());
        let seven = func.add_local(Some("seven".into()), MirType::Int, false, Span::dummy());
        let cond = func.add_local(Some("cond".into()), MirType::Bool, false, Span::dummy());
        let ret = func.add_local(Some("ret".into()), MirType::Int, false, Span::dummy());

        let bb_hdr = func.new_block();
        let bb_body = func.new_block();
        let bb_exit = func.new_block();

        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(b),
            Rvalue::Use(Operand::Copy(Place::Local(a))),
            Span::dummy(),
        ));
        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(i),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));
        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(zero),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));
        func.blocks[0].statements.push(Statement::Assign(
            Place::Local(seven),
            Rvalue::Use(Operand::Constant(MirConstant::Int(7))),
            Span::dummy(),
        ));
        func.blocks[0].terminator = Some(Terminator::Goto(bb_hdr, Span::dummy()));

        func.blocks[bb_hdr.0].statements.push(Statement::Assign(
            Place::Local(cond),
            Rvalue::BinaryOp(
                MirBinaryOp::Lt,
                Operand::Copy(Place::Local(i)),
                Operand::Copy(Place::Local(n)),
            ),
            Span::dummy(),
        ));
        func.blocks[bb_hdr.0].terminator = Some(Terminator::Cond {
            cond: Operand::Copy(Place::Local(cond)),
            then_block: bb_body,
            else_block: bb_exit,
            span: Span::dummy(),
        });

        func.blocks[bb_body.0].statements.push(Statement::Assign(
            Place::Index(Box::new(Place::Local(a)), zero),
            Rvalue::Use(Operand::Copy(Place::Local(seven))),
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

        func.blocks[bb_exit.0].statements.push(Statement::Assign(
            Place::Local(ret),
            Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(b)), zero))),
            Span::dummy(),
        ));
        func.blocks[bb_exit.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(ret))),
            Span::dummy(),
        ));

        let mut vm = VM::new();
        vm.set_jit_backend(VmBackend::Cranelift);
        vm.load_mir_function(func).expect("Load deopt_input_array_alias_loop");
        vm.test_deopt_spike_hook = Some(lust::vm::TestDeoptSpikeHook::new(fn_name, bb_hdr.0));
        safepoint_request_deopt_on_poll(Some(fn_name.into()), 1);

        let arr = Rc::new(LustArray::new(vec![Value::Int(3)]));
        let entry_before = vm.cranelift_entry_count;
        let res = vm
            .call(fn_name, vec![Value::Array(arr.clone()), Value::Int(1)])
            .expect("VM::call deopt continuation");

        assert_eq!(vm.cranelift_entry_count, entry_before + 1);
        assert_eq!(vm.last_deopt_source, Some(DeoptPayloadSource::NativeExecution));
        assert_eq!(res, Value::Int(7));
        assert_eq!(arr.borrow()[0], Value::Int(7));
    }
}

// =========================================================================
// F4: Nested MirContext scoped ownership & isolation
// =========================================================================

#[test]
fn test_w4_partial_f4_nested_mir_context_scoped_ownership_and_isolation() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    // Case 1: Unit probe with handles (F4_NESTED_CLEAR_TO_BASELINE pattern)
    {
        let ctx = MirContext::new();
        let _outer_guard = enter_mir_context(&ctx);
        let outer_buf = lust_mir_alloc_array(2);
        assert!(!outer_buf.is_null());
        let _outer_handle = ctx.register_handle(Value::string("outer"));
        let baseline_handles = ctx.handle_count();

        let _inner_handle = ctx.register_handle(Value::string("inner"));
        let inner_buf = lust_mir_alloc_array(2);
        assert!(!inner_buf.is_null());

        ctx.clear_to_baseline(baseline_handles);

        assert!(
            ctx.get_allocation_record(outer_buf as usize).is_some(),
            "Outer buffer must remain registered after inner clear_to_baseline"
        );
        assert!(
            ctx.get_allocation_record(inner_buf as usize).is_none(),
            "Inner buffer must be freed and unregistered"
        );
        assert_eq!(ctx.allocated_buffers.lock().unwrap().len(), 1);
        ctx.clear_to_baseline(0);
    }

    // Case 2: Zero-handle nested `ctx.enter()` scopes + active scope isolation
    {
        let ctx = MirContext::new();
        let _outer_guard = ctx.enter();
        let outer_buf = lust_mir_alloc_array_typed(2, MIR_ARRAY_ELEM_TAG_INT);
        assert!(!outer_buf.is_null());
        unsafe {
            *outer_buf.add(2) = 111;
            *outer_buf.add(3) = 222;
        }
        assert_eq!(ctx.handle_count(), 0);

        {
            let _inner_guard = ctx.enter();
            let inner_buf = lust_mir_alloc_array_typed(2, MIR_ARRAY_ELEM_TAG_INT);
            assert!(!inner_buf.is_null());
            unsafe {
                *inner_buf.add(2) = 333;
                *inner_buf.add(3) = 444;
            }

            // While inner scope is active, outer buffer must be rejected as out-of-scope
            let outer_in_inner = lust::vm::cranelift_dispatch::validate_and_decode_array(
                &ctx,
                &[],
                outer_buf as i64,
                &MirType::Int,
            );
            assert!(
                outer_in_inner.is_err(),
                "Inner scope must reject outer-scope buffer pointer"
            );

            // Inner buffer must succeed
            let inner_ok = lust::vm::cranelift_dispatch::validate_and_decode_array(
                &ctx,
                &[],
                inner_buf as i64,
                &MirType::Int,
            )
            .expect("Inner buffer must decode in inner scope");
            assert_eq!(inner_ok.elements, vec![Value::Int(333), Value::Int(444)]);

            // Clear inner scope with zero handles
            ctx.clear_to_baseline(0);
            assert!(ctx.get_allocation_record(inner_buf as usize).is_none());
            assert!(
                ctx.get_allocation_record(outer_buf as usize).is_some(),
                "Zero-handle inner clear_to_baseline must preserve outer buffer"
            );
        }

        // Outer scope is active again and outer_buf decodes cleanly
        let outer_decoded = lust::vm::cranelift_dispatch::validate_and_decode_array(
            &ctx,
            &[],
            outer_buf as i64,
            &MirType::Int,
        )
        .expect("Outer buffer must remain valid and intact after inner scope exits");
        assert_eq!(outer_decoded.elements, vec![Value::Int(111), Value::Int(222)]);
        ctx.clear_to_baseline(0);
        assert_eq!(ctx.allocated_buffers.lock().unwrap().len(), 0);
    }

    // Case 3: Outer MirContext::enter() survives pre-entry fallback in VM::call
    {
        let source = r#"
            function nested_bail_noop(a: Array<Array<int>>): int
                return 1
            end
        "#;
        let mut vm = setup_vm_from_source(source, "f4_nested_vm_bail");
        let _outer_guard = vm.mir_context.enter();
        let outer_buf = lust_mir_alloc_array(2);
        assert!(!outer_buf.is_null());

        let inner = Rc::new(LustArray::new(vec![Value::Int(42)]));
        let outer = Rc::new(LustArray::new(vec![Value::Array(inner)]));
        let _ = vm.call("nested_bail_noop", vec![Value::Array(outer)]);

        assert!(
            vm.mir_context.get_allocation_record(outer_buf as usize).is_some(),
            "Outer buffer on vm.mir_context must survive nested VM::call pre-entry bailout"
        );
        drop(_outer_guard);
        vm.mir_context.clear_to_baseline(0);
    }
}

// =========================================================================
// F5: Atomic array & struct rollback and zero-leak cleanup on commit error
// =========================================================================

#[test]
fn test_w4_partial_f5_atomic_array_and_struct_rollback_and_no_leak_on_commit_error() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    let source = r#"
        struct BoxedInt
            x: int
        end

        function mutate_array_and_struct(a: Array<int>, b: Array<int>, s: BoxedInt): int
            local dummy = BoxedInt { x = 0 }
            a[0] = 77
            b[0] = 88
            s.x = 99
            return 1
        end

        function mutate_aliased_structs(s1: BoxedInt, s2: BoxedInt): int
            local dummy = BoxedInt { x = 0 }
            s1.x = 42
            s2.x = s1.x + 1
            return s2.x
        end
    "#;
    let mut vm = setup_vm_from_source(source, "f5_atomic");

    let arr_a = Rc::new(LustArray::new(vec![Value::Int(1)]));
    let arr_b = Rc::new(LustArray::new(vec![Value::Int(2)]));
    let struct_s_val = vm
        .instantiate_struct_ordered("BoxedInt", vec![Value::Int(10)])
        .expect("Instantiate BoxedInt");
    let Value::Struct(ref struct_s) = struct_s_val else {
        panic!("Expected Value::Struct");
    };

    // Case 1: Active shared borrow on arr_b causes commit failure -> ZERO partial mutations & ZERO leaks
    {
        let _shared = arr_b.borrow();
        let entry_before = vm.cranelift_entry_count;
        let res = vm.call(
            "mutate_array_and_struct",
            vec![
                Value::Array(arr_a.clone()),
                Value::Array(arr_b.clone()),
                struct_s_val.clone(),
            ],
        );

        assert_eq!(vm.cranelift_entry_count, entry_before + 1);
        let err = res.expect_err("Expected aliased borrow commit error");
        assert!(
            err.to_string().contains("aliased borrow active"),
            "Expected aliased borrow error, got: {err}"
        );
        assert_eq!(
            arr_a.borrow()[0],
            Value::Int(1),
            "Unborrowed sibling array must NOT be partially mutated on commit failure"
        );
        assert_eq!(
            struct_s.fields.borrow()[0],
            Value::Int(10),
            "Struct argument must NOT be mutated before array commit succeeds"
        );
        assert_eq!(
            vm.mir_context.allocated_buffers.lock().unwrap().len(),
            0,
            "MirContext buffers must not leak on normal-return commit error"
        );
        assert_eq!(
            vm.mir_context.allocation_records.lock().unwrap().len(),
            0,
            "MirContext allocation records must not leak on normal-return commit error"
        );
        assert_eq!(vm.call_stack_depth(), 0, "Frame must be popped on commit error");
    }
    assert_eq!(arr_b.borrow()[0], Value::Int(2));

    // Case 2: Aliased struct arguments s1, s2 share one native buffer and update cleanly
    {
        let res = vm
            .call(
                "mutate_aliased_structs",
                vec![struct_s_val.clone(), struct_s_val.clone()],
            )
            .expect("Aliased structs call must succeed");
        assert_eq!(res, Value::Int(43));
        assert_eq!(struct_s.fields.borrow()[0], Value::Int(43));
        assert_eq!(vm.mir_context.allocated_buffers.lock().unwrap().len(), 0);
        assert_eq!(vm.mir_context.allocation_records.lock().unwrap().len(), 0);
    }
}

// =========================================================================
// F6: Unsupported non-primitive & mismatched arrays fall back pre-entry
// =========================================================================

#[test]
fn test_w4_partial_f6_unsupported_nested_and_mismatched_arrays_fallback_pre_entry() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    // Case 1: Static MIR signature Array<Array<int>> falls back before native entry
    {
        let fn_name = "nested_array_noop";
        let sig = FunctionSignature {
            name: fn_name.into(),
            param_types: vec![MirType::Array(Box::new(MirType::Array(Box::new(MirType::Int))))],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut func = MirFunction::new(sig, Span::dummy());
        func.blocks[0].terminator = Some(Terminator::Return(
            Some(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));

        let mut vm = VM::new();
        vm.set_backend(VmBackend::Cranelift);
        vm.load_mir_function(func).expect("Load nested_array_noop");

        let inner = Rc::new(LustArray::new(vec![Value::Int(42)]));
        let outer = Rc::new(LustArray::new(vec![Value::Array(inner.clone())]));
        let entry_before = vm.cranelift_entry_count;

        let _ = vm.call(fn_name, vec![Value::Array(outer.clone())]);

        assert_eq!(
            vm.cranelift_entry_count, entry_before,
            "Array<Array<int>> must fall back before Cranelift native entry"
        );
        assert!(
            matches!(&outer.borrow()[0], Value::Array(rc) if Rc::ptr_eq(rc, &inner) && rc.borrow()[0] == Value::Int(42)),
            "Nested array element must NOT be clobbered to Value::Int(0)"
        );
        assert_eq!(vm.mir_context.allocated_buffers.lock().unwrap().len(), 0);
    }

    // Case 2: Runtime mismatched element Value::Array([Value::String("bad")]) passed to Array<int>
    {
        let source = r#"
            function int_array_noop(a: Array<int>): int
                return 0
            end
        "#;
        let mut vm = setup_vm_from_source(source, "f6_mismatch");

        let bad_arr = Rc::new(LustArray::new(vec![Value::string("bad")]));
        let entry_before = vm.cranelift_entry_count;

        let _ = vm.call("int_array_noop", vec![Value::Array(bad_arr.clone())]);

        assert_eq!(
            vm.cranelift_entry_count, entry_before,
            "Runtime mismatched array element must fall back before Cranelift native entry"
        );
        assert_eq!(bad_arr.borrow()[0], Value::string("bad"));
        assert_eq!(vm.mir_context.allocated_buffers.lock().unwrap().len(), 0);
    }

    // Case 3: validate_and_decode_array rejects non-primitive expected_elem_ty
    {
        let ctx = MirContext::new();
        let buf = Box::into_raw(Box::new([1i64, 1i64, 0i64])) as *mut i64;
        ctx.register_allocation(
            buf,
            3,
            lust::mir::runtime_helpers::MirAllocationKind::Array(MirType::Array(Box::new(
                MirType::Int,
            ))),
        );
        let err = lust::vm::cranelift_dispatch::validate_and_decode_array(
            &ctx,
            &[],
            buf as i64,
            &MirType::Array(Box::new(MirType::Int)),
        )
        .expect_err("Non-primitive array element type must be rejected");
        assert!(err.to_string().contains("Unsupported non-primitive"));
        unsafe {
            drop(Box::from_raw(buf as *mut [i64; 3]));
        }
    }
}

// =========================================================================
// F7: Specialized array cache coherence across CurrentVmGuard & VmId(0)
// =========================================================================

#[test]
fn test_w4_partial_f7_specialized_cache_coherence_across_vm_guard_boundaries() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_native_instrumentation_counters();

    let source = r#"
        function read_first_int(a: Array<int>): int
            return a[0]
        end

        function write_first_int(a: Array<int>, v: int): int
            a[0] = v
            return a[0]
        end
    "#;
    let mut vm = setup_vm_from_source(source, "f7_cache");

    // Case 1: Warmed outside CurrentVmGuard (VmId(0)) -> read inside VM::call (VmId(vm.id))
    {
        let arr = Rc::new(LustArray::new(vec![Value::Int(10)]));
        let val = Value::Array(arr.clone());
        let mut slot = lust::bytecode::value::JitVecSlot {
            ptr: core::ptr::null_mut(),
            len: 0,
            cap: 0,
            array: core::ptr::null(),
        };
        unsafe {
            assert_eq!(lust::bytecode::value::jit_unbox_array_int(&val, &mut slot), 1);
            *slot.ptr = 20;
            assert_eq!(lust::bytecode::value::jit_rebox_array_int(&mut slot), 1);
        }
        assert_eq!(arr.borrow()[0], Value::Int(10));

        let res = vm
            .call("read_first_int", vec![Value::Array(arr.clone())])
            .expect("VM::call read_first_int");
        assert_eq!(
            res,
            Value::Int(20),
            "VM::call must observe specialized cache warmed outside CurrentVmGuard"
        );
        assert_eq!(arr.borrow()[0], Value::Int(20));
    }

    // Case 2: Warmed outside CurrentVmGuard -> mutated inside VM::call -> no stale VmId(0) left
    {
        let arr = Rc::new(LustArray::new(vec![Value::Int(10)]));
        let val = Value::Array(arr.clone());
        let mut slot = lust::bytecode::value::JitVecSlot {
            ptr: core::ptr::null_mut(),
            len: 0,
            cap: 0,
            array: core::ptr::null(),
        };
        unsafe {
            assert_eq!(lust::bytecode::value::jit_unbox_array_int(&val, &mut slot), 1);
            *slot.ptr = 20;
            assert_eq!(lust::bytecode::value::jit_rebox_array_int(&mut slot), 1);
        }

        let res = vm
            .call("write_first_int", vec![Value::Array(arr.clone()), Value::Int(99)])
            .expect("VM::call write_first_int");
        assert_eq!(res, Value::Int(99));
        assert_eq!(arr.borrow()[0], Value::Int(99));
        assert_eq!(
            lust::bytecode::value::get_specialized_cached_element(&arr, 0),
            None,
            "No stale VmId(0) cache entry may remain after VM::call writeback invalidation"
        );
    }

    // Case 3: Warmed inside CurrentVmGuard -> read & invalidated outside CurrentVmGuard
    {
        let arr = Rc::new(LustArray::new(vec![Value::Int(10)]));
        let val = Value::Array(arr.clone());
        let mut slot = lust::bytecode::value::JitVecSlot {
            ptr: core::ptr::null_mut(),
            len: 0,
            cap: 0,
            array: core::ptr::null(),
        };
        {
            let _guard = lust::vm::CurrentVmGuard::from_vm(&mut vm);
            unsafe {
                assert_eq!(lust::bytecode::value::jit_unbox_array_int(&val, &mut slot), 1);
                *slot.ptr = 55;
                assert_eq!(lust::bytecode::value::jit_rebox_array_int(&mut slot), 1);
            }
        }
        assert_eq!(
            lust::bytecode::value::get_specialized_cached_element(&arr, 0),
            Some(Value::Int(55)),
            "Reader outside CurrentVmGuard must see cache dirtied inside CurrentVmGuard"
        );
        lust::bytecode::value::invalidate_specialized_array_cache(&arr);
        assert_eq!(
            lust::bytecode::value::get_specialized_cached_element(&arr, 0),
            None,
            "Invalidation outside CurrentVmGuard must clear VM-scoped cache entry"
        );
    }

    // Case 4: Dirty specialized cache + active shared borrow must return Err on VM::call
    // (must not silently ignore sync error or cause unsafe RefCell aliasing UB)
    {
        let arr = Rc::new(LustArray::new(vec![Value::Int(10)]));
        let val = Value::Array(arr.clone());
        let mut slot = lust::bytecode::value::JitVecSlot {
            ptr: core::ptr::null_mut(),
            len: 0,
            cap: 0,
            array: core::ptr::null(),
        };
        unsafe {
            assert_eq!(lust::bytecode::value::jit_unbox_array_int(&val, &mut slot), 1);
            *slot.ptr = 77;
            assert_eq!(lust::bytecode::value::jit_rebox_array_int(&mut slot), 1);
        }
        let _shared = arr.borrow();
        let err = vm
            .call("read_first_int", vec![Value::Array(arr.clone())])
            .expect_err("Dirty cache sync with active borrow must fail cleanly instead of reading stale value");
        assert!(
            err.to_string().contains("specialized int array sync"),
            "Expected sync borrow error, got: {err}"
        );
        assert_eq!(vm.mir_context.allocated_buffers.lock().unwrap().len(), 0);
        assert_eq!(vm.mir_context.allocation_records.lock().unwrap().len(), 0);
        drop(_shared);
        lust::bytecode::value::invalidate_specialized_array_cache(&arr);
    }
}

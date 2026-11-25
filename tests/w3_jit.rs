use std::sync::atomic::Ordering;
use lust::ast::Span;
use lust::bytecode::Value;
use lust::data::layout::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1, TargetAbi};
use lust::mir::cranelift_backend::*;
use lust::mir::ir::*;
use lust::mir::opt::escape_analysis::EscapeAnalysis;
use lust::mir::runtime_helpers::{enter_mir_context, MirContext};

static TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn create_user_layout() -> MirStructLayout {
    let logical = LogicalLayoutV1::new_with_abi(
        101,
        "User",
        vec![
            LogicalFieldDesc::new(0, "id", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(1, "score", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(2, "is_active", CanonicalTypeTag::Bool),
        ],
        TargetAbi::x86_64(),
    );
    MirStructLayout::from(&logical)
}

// =========================================================================
// SECTION 1: Native compiled entry & arithmetic parity
// =========================================================================

#[test]
fn test_native_compiled_entry_and_execution() {
    let _lock = TEST_MUTEX.lock().unwrap();
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);

    let sig = FunctionSignature {
        name: "compute_sum_w3".into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let arg0 = func.params[0];
    let multiplier = func.add_local(Some("multiplier".into()), MirType::Int, false, Span::dummy());

    let bb0 = &mut func.blocks[0];
    bb0.statements.push(Statement::Assign(
        Place::Local(multiplier),
        Rvalue::Use(Operand::Constant(MirConstant::Int(10))),
        Span::dummy(),
    ));
    bb0.statements.push(Statement::Assign(
        Place::Local(func.return_local),
        Rvalue::BinaryOp(
            MirBinaryOp::Mul,
            Operand::Copy(Place::Local(arg0)),
            Operand::Copy(Place::Local(multiplier)),
        ),
        Span::dummy(),
    ));
    bb0.terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(func.return_local))),
        Span::dummy(),
    ));

    let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift compiler");
    let code_ptr = compiler.compile_function(&func).expect("Compile compute_sum_w3");
    assert!(!code_ptr.is_null());

    let native_fn: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code_ptr) };
    let result = native_fn(42);
    assert_eq!(result, 420, "42 * 10 must equal 420");
}

// =========================================================================
// SECTION 2: J-IC (Inline Caches: feedback, hit/miss, invalidation, fallback)
// =========================================================================

#[test]
fn test_inline_cache_monomorphic_feedback_and_hits() {
    let _lock = TEST_MUTEX.lock().unwrap();
    ic_reset();

    let site_id = 1u32;
    ic_register_site(site_id);

    // Initial state: Uninitialized
    let stats = ic_site_stats(site_id).expect("Site must exist");
    assert_eq!(stats.state, IcStateKind::Uninitialized);
    assert_eq!(stats.entry_count, 0);

    // 1. First probe on unseen tag -> Miss
    let probe1 = ic_probe(site_id, 1001);
    assert_eq!(probe1, None);

    // 2. Dispatch with tag 1001 and target 0x1111 -> Records Monomorphic
    let target1 = ic_dispatch(site_id, 1001, 0x1111);
    assert_eq!(target1, 0x1111);

    let stats = ic_site_stats(site_id).unwrap();
    assert_eq!(stats.state, IcStateKind::Monomorphic);
    assert_eq!(stats.entry_count, 1);

    // 3. Subsequent probes and dispatches -> Cache Hit!
    let probe2 = ic_probe(site_id, 1001);
    assert_eq!(probe2, Some(0x1111));

    let target_hit = ic_dispatch(site_id, 1001, 0x9999);
    assert_eq!(target_hit, 0x1111, "Must return cached target, not default");

    let stats_after = ic_site_stats(site_id).unwrap();
    assert!(stats_after.hits >= 2, "Must record cache hits");
}

#[test]
fn test_inline_cache_polymorphic_feedback_fallback_and_invalidation() {
    let _lock = TEST_MUTEX.lock().unwrap();
    ic_reset();

    let site_id = 2u32;
    ic_register_site(site_id);

    // Add 4 distinct types up to capacity
    ic_dispatch(site_id, 101, 0xA1); // transitions to Monomorphic
    assert_eq!(ic_site_stats(site_id).unwrap().state, IcStateKind::Monomorphic);

    ic_dispatch(site_id, 102, 0xA2); // transitions to Polymorphic (count=2)
    assert_eq!(ic_site_stats(site_id).unwrap().state, IcStateKind::Polymorphic);
    assert_eq!(ic_site_stats(site_id).unwrap().entry_count, 2);

    ic_dispatch(site_id, 103, 0xA3); // Polymorphic (count=3)
    ic_dispatch(site_id, 104, 0xA4); // Polymorphic (count=4)
    assert_eq!(ic_site_stats(site_id).unwrap().entry_count, 4);

    // Verify all 4 are hits
    assert_eq!(ic_probe(site_id, 101), Some(0xA1));
    assert_eq!(ic_probe(site_id, 102), Some(0xA2));
    assert_eq!(ic_probe(site_id, 103), Some(0xA3));
    assert_eq!(ic_probe(site_id, 104), Some(0xA4));

    // 5th distinct type exceeds IC capacity -> MegamorphicFallback
    let fallback_res = ic_dispatch(site_id, 105, 0xA5);
    assert_eq!(fallback_res, 0xA5);

    let stats = ic_site_stats(site_id).unwrap();
    assert_eq!(stats.state, IcStateKind::MegamorphicFallback);
    assert!(stats.fallbacks >= 1);

    // Invalidation
    ic_invalidate(site_id);
    let stats_inv = ic_site_stats(site_id).unwrap();
    assert_eq!(stats_inv.state, IcStateKind::Uninitialized);
    assert_eq!(stats_inv.entry_count, 0);
    assert_eq!(stats_inv.invalidations, 1);

    // Post-invalidation probe misses
    assert_eq!(ic_probe(site_id, 101), None);
}

#[test]
fn test_inline_cache_native_compiled_execution() {
    let _lock = TEST_MUTEX.lock().unwrap();
    ic_reset();

    let site_id = 42u32;
    let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift compiler");
    let code_ptr = compiler
        .compile_ic_dispatch_tester(site_id)
        .expect("Compile native IC dispatch tester");
    assert!(!code_ptr.is_null());

    let native_ic_fn: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(code_ptr) };

    // 1. Initial call with tag 1 -> registers target 500
    let res1 = native_ic_fn(1, 500);
    assert_eq!(res1, 500);

    // 2. Second call with tag 1 -> hit returns cached 500 (ignores new default 999)
    let res2 = native_ic_fn(1, 999);
    assert_eq!(res2, 500);

    // 3. Invalidate site -> next call takes new default 777
    ic_invalidate(site_id);
    let res3 = native_ic_fn(1, 777);
    assert_eq!(res3, 777);
}

// =========================================================================
// SECTION 3: J-ESC (Conservative stack promotion vs escaping locals)
// =========================================================================

#[test]
fn test_escape_analysis_stack_promotion_and_escaping_safety() {
    let _lock = TEST_MUTEX.lock().unwrap();

    let sig = FunctionSignature {
        name: "test_esc_fn".into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let arg0 = func.params[0];
    let pure_temp = func.add_local(Some("pure_temp".into()), MirType::Int, true, Span::dummy());
    let alias_arg = func.add_local(Some("alias_arg".into()), MirType::Int, true, Span::dummy());
    let escaping_call = func.add_local(Some("escaping_call".into()), MirType::Int, true, Span::dummy());

    let bb1 = func.new_block();
    let bb0 = &mut func.blocks[0];
    // pure_temp = 100 (non-escaping local computation)
    bb0.statements.push(Statement::Assign(
        Place::Local(pure_temp),
        Rvalue::Use(Operand::Constant(MirConstant::Int(100))),
        Span::dummy(),
    ));
    // alias_arg = arg0 (alias of parameter -> escapes call)
    bb0.statements.push(Statement::Assign(
        Place::Local(alias_arg),
        Rvalue::Use(Operand::Copy(Place::Local(arg0))),
        Span::dummy(),
    ));
    // escaping_call passed to external call
    bb0.statements.push(Statement::Assign(
        Place::Local(escaping_call),
        Rvalue::Use(Operand::Constant(MirConstant::Int(5))),
        Span::dummy(),
    ));

    bb0.terminator = Some(Terminator::Call {
        callee: "external_sink".into(),
        args: vec![Operand::Copy(Place::Local(escaping_call))],
        destination: Place::Local(func.return_local),
        success: bb1,
        error: None,
        span: Span::dummy(),
    });

    func.blocks[bb1.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(func.return_local))),
        Span::dummy(),
    ));

    // Run escape analysis
    let changed = EscapeAnalysis::run_on_function(&mut func);
    assert!(changed);

    // Verify conservative escape classification:
    assert!(func.locals[pure_temp.0].is_stack_promoted, "Non-escaping local must be stack promoted");
    assert!(!func.locals[arg0.0].is_stack_promoted, "Parameter must NOT be stack promoted");
    assert!(!func.locals[alias_arg.0].is_stack_promoted, "Alias of parameter must NOT be stack promoted");
    assert!(!func.locals[escaping_call.0].is_stack_promoted, "Argument to call must NOT be stack promoted");
    assert!(!func.locals[func.return_local.0].is_stack_promoted, "Return local must NOT be stack promoted");
}

#[test]
fn test_stack_promoted_locals_lowered_to_stack_slots() {
    let _lock = TEST_MUTEX.lock().unwrap();
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);

    let sig = FunctionSignature {
        name: "test_stack_slot_exec".into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let arg0 = func.params[0];
    let mut local1 = func.add_local(Some("local1".into()), MirType::Int, true, Span::dummy());
    let mut local2 = func.add_local(Some("local2".into()), MirType::Int, true, Span::dummy());

    // Explicitly mark locals as stack promoted
    func.locals[local1.0].is_stack_promoted = true;
    func.locals[local2.0].is_stack_promoted = true;

    let bb0 = &mut func.blocks[0];
    // local1 = arg0 + 10
    bb0.statements.push(Statement::Assign(
        Place::Local(local1),
        Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(arg0)), Operand::Constant(MirConstant::Int(10))),
        Span::dummy(),
    ));
    // local2 = local1 * 2
    bb0.statements.push(Statement::Assign(
        Place::Local(local2),
        Rvalue::BinaryOp(MirBinaryOp::Mul, Operand::Copy(Place::Local(local1)), Operand::Constant(MirConstant::Int(2))),
        Span::dummy(),
    ));
    // return = local2 + 5
    bb0.statements.push(Statement::Assign(
        Place::Local(func.return_local),
        Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(local2)), Operand::Constant(MirConstant::Int(5))),
        Span::dummy(),
    ));
    bb0.terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(func.return_local))),
        Span::dummy(),
    ));

    let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift compiler");
    let code_ptr = compiler.compile_function(&func).expect("Compile stack slot function");
    let native_fn: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code_ptr) };

    // Computation: (15 + 10) * 2 + 5 = 50 + 5 = 55
    let result = native_fn(15);
    assert_eq!(result, 55, "Native stack slot execution must yield correct result 55");
}

// =========================================================================
// SECTION 4: Gate-SCHEMA-09/13 Cranelift Schema Emission (Decode & LoadField)
// =========================================================================

#[test]
fn test_cranelift_struct_decode_into_and_constant_offset_loads() {
    let _lock = TEST_MUTEX.lock().unwrap();
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    clear_registered_struct_layouts();

    let layout = create_user_layout();
    let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift compiler");

    // 1. Compile native struct decoder and field loaders
    let decode_ptr = compiler.compile_struct_decoder(&layout).expect("Compile decoder");
    let load_id_ptr = compiler.compile_struct_field_loader(&layout, 0).expect("Compile field 0 loader");
    let load_score_ptr = compiler.compile_struct_field_loader(&layout, 1).expect("Compile field 1 loader");
    let load_active_ptr = compiler.compile_struct_field_loader(&layout, 2).expect("Compile field 2 loader");

    let native_decode: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(decode_ptr) };
    let native_load_id: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(load_id_ptr) };
    let native_load_score: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(load_score_ptr) };
    let native_load_active: extern "C" fn(i64) -> u8 = unsafe { std::mem::transmute(load_active_ptr) };

    // 2. Prepare JSON input in MirContext
    let json_input = r#"{"id": 42001, "score": 9850, "is_active": true}"#;
    let json_handle = ctx.intern_string(json_input);

    // 3. Execute native StructDecodeInto
    let struct_mem_ptr = native_decode(json_handle);
    assert_ne!(struct_mem_ptr, 0, "Native decode must return valid non-null memory pointer");

    // 4. Execute StructLoadField via native constant byte-offset dereference
    let id_val = native_load_id(struct_mem_ptr);
    let score_val = native_load_score(struct_mem_ptr);
    let active_val = native_load_active(struct_mem_ptr);

    // 5. Compare against oracle values
    assert_eq!(id_val, 42001, "Field 'id' at offset 0 must match oracle 42001");
    assert_eq!(score_val, 9850, "Field 'score' at offset 8 must match oracle 9850");
    assert_eq!(active_val, 1, "Field 'is_active' at offset 16 must match oracle true (1)");
}

#[test]
fn test_cranelift_struct_decode_negative_cases_and_cleanup() {
    let _lock = TEST_MUTEX.lock().unwrap();
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    clear_registered_struct_layouts();

    let layout = create_user_layout();
    let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift compiler");
    let decode_ptr = compiler.compile_struct_decoder(&layout).expect("Compile decoder");
    let native_decode: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(decode_ptr) };

    // Negative Case 1: Trailing junk
    let bad_json_1 = r#"{"id": 1, "score": 2, "is_active": true} trailing_junk"#;
    let h1 = ctx.intern_string(bad_json_1);
    let res1 = native_decode(h1);
    assert_eq!(res1, 0, "Decoder must reject trailing junk and return 0 (null)");
    assert_ne!(ctx.last_status.load(Ordering::SeqCst), 0, "Error status must be recorded");

    // Negative Case 2: Invalid JSON syntax
    let bad_json_2 = r#"{"id": 1, "score": invalid_token}"#;
    let h2 = ctx.intern_string(bad_json_2);
    let res2 = native_decode(h2);
    assert_eq!(res2, 0, "Decoder must reject invalid syntax and return 0 (null)");

    // Negative Case 3: Invalid string handle
    let res3 = native_decode(999999);
    assert_eq!(res3, 0, "Decoder must reject invalid handle");
}

// =========================================================================
// SECTION 5: 3A->3D Deopt Seam (publish DeoptPayload on unsupported effects)
// =========================================================================

#[test]
fn test_cranelift_deopt_seam_on_unsupported_effect_op() {
    let _lock = TEST_MUTEX.lock().unwrap();
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    safepoint_reset();

    let sig = FunctionSignature {
        name: "test_effect_deopt_fn".into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let arg0 = func.params[0];
    let local_accum = func.add_local(Some("accum".into()), MirType::Int, true, Span::dummy());

    let bb1 = func.new_block();
    let bb0 = &mut func.blocks[0];
    bb0.statements.push(Statement::Assign(
        Place::Local(local_accum),
        Rvalue::BinaryOp(MirBinaryOp::Add, Operand::Copy(Place::Local(arg0)), Operand::Constant(MirConstant::Int(100))),
        Span::dummy(),
    ));

    // Encounter unsupported effect call: callee starts with "effect."
    bb0.terminator = Some(Terminator::Call {
        callee: "effect.perform".into(),
        args: vec![Operand::Copy(Place::Local(local_accum))],
        destination: Place::Local(func.return_local),
        success: bb1,
        error: None,
        span: Span::dummy(),
    });

    func.blocks[bb1.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(func.return_local))),
        Span::dummy(),
    ));

    let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift compiler");
    let code_ptr = compiler.compile_function(&func).expect("Compile deopting effect function");
    let native_fn: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code_ptr) };

    // Execute compiled code: encounters effect.perform -> emits MIR_STATUS_DEOPT and publishes DeoptPayload!
    let ret_code = native_fn(25);
    assert_eq!(ret_code, MIR_STATUS_DEOPT, "Native execution must return MIR_STATUS_DEOPT");

    // Take published DeoptPayload
    let payload = take_deopt_payload().expect("DeoptPayload must be published into thread-local state");
    assert_eq!(payload.target_pc, 0, "Target PC must match the basic block where effect was encountered");

    // Live locals must be captured
    let captured_accum = payload.live_locals.iter().find(|(idx, _)| *idx == local_accum.0);
    assert!(captured_accum.is_some(), "local_accum must be present in live_locals");
    assert_eq!(captured_accum.unwrap().1, 125, "local_accum value must be 25 + 100 = 125");

    // Lossless VM value reconstruction
    let vm_values = payload.to_vm_values(&func);
    let accum_vm = vm_values.iter().find(|(idx, _)| *idx == local_accum.0).map(|(_, v)| v);
    assert_eq!(accum_vm, Some(&Value::Int(125)), "Reconstructed VM Value must match exactly");
}

// =========================================================================
// SECTION 6: End-to-End MIR Module Native Execution with Schema & IC
// =========================================================================

#[test]
fn test_end_to_end_module_native_execution() {
    let _lock = TEST_MUTEX.lock().unwrap();
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    clear_registered_struct_layouts();
    ic_reset();

    let layout = create_user_layout();
    let mut module = MirModule::new("test_w3_module".into());
    module.register_struct_layout(layout.clone());

    // Function: fn run_ic(tag: int, default_val: int) -> int
    let sig = FunctionSignature {
        name: "run_ic".into(),
        param_types: vec![MirType::Int, MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let p0 = func.params[0];
    let p1 = func.params[1];
    let bb1 = func.new_block();
    let bb0 = &mut func.blocks[0];

    bb0.terminator = Some(Terminator::Call {
        callee: "ic:99".into(),
        args: vec![Operand::Copy(Place::Local(p0)), Operand::Copy(Place::Local(p1))],
        destination: Place::Local(func.return_local),
        success: bb1,
        error: None,
        span: Span::dummy(),
    });
    func.blocks[bb1.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(func.return_local))),
        Span::dummy(),
    ));
    module.add_function(func);

    let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift compiler");
    let func_map = compiler.compile_module(&module).expect("Compile module");

    let run_ic_ptr = func_map["run_ic"];
    let native_run_ic: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(run_ic_ptr) };

    // Initial dispatch registers tag 7 with target 777
    let out1 = native_run_ic(7, 777);
    assert_eq!(out1, 777);

    // Hit dispatch returns cached 777
    let out2 = native_run_ic(7, 9999);
    assert_eq!(out2, 777);
}

// =========================================================================
// SECTION 7: Section 4.8.A & 4.8.C Independent Oracles & Counters
// =========================================================================

#[test]
fn test_w3_4_8_a_hot_loop_deopt_state_reconstruction_oracle() {
    let _lock = TEST_MUTEX.lock().unwrap();
    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    safepoint_reset();
    reset_native_instrumentation_counters();
    take_deopt_payload();

    // Section 4.8.A: compute_with_effect(100)
    // acc = 0; for i in 0..iterations { acc += i; if i == 50 { acc += perform Yield(acc); } }
    let sig = FunctionSignature {
        name: "compute_with_effect".into(),
        param_types: vec![MirType::Int],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut func = MirFunction::new(sig, Span::dummy());
    let iterations_loc = func.params[0]; // LocalId(1)
    let acc_loc = func.add_local(Some("acc".into()), MirType::Int, true, Span::dummy()); // LocalId(2)
    let i_loc = func.add_local(Some("i".into()), MirType::Int, true, Span::dummy()); // LocalId(3)
    let cond_loop = func.add_local(Some("cond_loop".into()), MirType::Bool, false, Span::dummy());
    let cond_yield = func.add_local(Some("cond_yield".into()), MirType::Bool, false, Span::dummy());
    let yield_res = func.add_local(Some("yield_res".into()), MirType::Int, false, Span::dummy());

    let bb_loop_hdr = func.new_block(); // 1
    let bb_loop_body = func.new_block(); // 2
    let bb_do_yield = func.new_block(); // 3
    let bb_after_yield = func.new_block(); // 4
    let bb_loop_next = func.new_block(); // 5
    let bb_exit = func.new_block(); // 6

    // bb0: acc = 0; i = 0; goto bb_loop_hdr
    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(acc_loc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    func.blocks[0].statements.push(Statement::Assign(
        Place::Local(i_loc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        Span::dummy(),
    ));
    func.blocks[0].terminator = Some(Terminator::Goto(bb_loop_hdr, Span::dummy()));

    // bb_loop_hdr: if i < iterations -> bb_loop_body else bb_exit
    func.blocks[bb_loop_hdr.0].statements.push(Statement::Assign(
        Place::Local(cond_loop),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(i_loc)),
            Operand::Copy(Place::Local(iterations_loc)),
        ),
        Span::dummy(),
    ));
    func.blocks[bb_loop_hdr.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond_loop)),
        then_block: bb_loop_body,
        else_block: bb_exit,
        span: Span::dummy(),
    });

    // bb_loop_body: acc = acc + i; if i == 50 -> bb_do_yield else bb_loop_next
    func.blocks[bb_loop_body.0].statements.push(Statement::Assign(
        Place::Local(acc_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(acc_loc)),
            Operand::Copy(Place::Local(i_loc)),
        ),
        Span::dummy(),
    ));
    func.blocks[bb_loop_body.0].statements.push(Statement::Assign(
        Place::Local(cond_yield),
        Rvalue::BinaryOp(
            MirBinaryOp::Eq,
            Operand::Copy(Place::Local(i_loc)),
            Operand::Constant(MirConstant::Int(50)),
        ),
        Span::dummy(),
    ));
    func.blocks[bb_loop_body.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond_yield)),
        then_block: bb_do_yield,
        else_block: bb_loop_next,
        span: Span::dummy(),
    });

    // bb_do_yield: effect.perform(acc) -> MIR_STATUS_DEOPT
    func.blocks[bb_do_yield.0].terminator = Some(Terminator::Call {
        callee: "effect.perform".into(),
        args: vec![Operand::Copy(Place::Local(acc_loc))],
        destination: Place::Local(yield_res),
        success: bb_after_yield,
        error: None,
        span: Span::dummy(),
    });

    // bb_after_yield: acc = acc + yield_res; goto bb_loop_next
    func.blocks[bb_after_yield.0].statements.push(Statement::Assign(
        Place::Local(acc_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(acc_loc)),
            Operand::Copy(Place::Local(yield_res)),
        ),
        Span::dummy(),
    ));
    func.blocks[bb_after_yield.0].terminator = Some(Terminator::Goto(bb_loop_next, Span::dummy()));

    // bb_loop_next: i = i + 1; goto bb_loop_hdr
    func.blocks[bb_loop_next.0].statements.push(Statement::Assign(
        Place::Local(i_loc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(i_loc)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        Span::dummy(),
    ));
    func.blocks[bb_loop_next.0].terminator = Some(Terminator::Goto(bb_loop_hdr, Span::dummy()));

    // bb_exit: return acc
    func.blocks[bb_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(acc_loc))),
        Span::dummy(),
    ));

    let mut module = MirModule::new("deopt_4_8_a".into());
    module.add_function(func.clone());

    let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift compiler");
    let func_map = compiler.compile_module(&module).expect("Compile compute_with_effect");
    let code_ptr = func_map["compute_with_effect"];
    let (jit_start, jit_end) = compiler
        .compiled_code_range("compute_with_effect")
        .expect("Compiled code range must be recorded");
    let entry_addr = code_ptr as usize;
    assert!(
        entry_addr >= jit_start && entry_addr < jit_end,
        "Entry ptr 0x{entry_addr:x} must lie inside [0x{jit_start:x}, 0x{jit_end:x})"
    );

    assert_eq!(CRANELIFT_NATIVE_ENTRY_COUNT.load(Ordering::SeqCst), 0);
    let native_fn: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code_ptr) };
    let ret_code = native_fn(100);

    // Proof of native entry BEFORE deopt check
    assert!(
        CRANELIFT_NATIVE_ENTRY_COUNT.load(Ordering::SeqCst) >= 1,
        "CRANELIFT_NATIVE_ENTRY_COUNT must be >= 1"
    );
    assert_eq!(ret_code, MIR_STATUS_DEOPT, "Must deopt at i == 50 on Yield");

    let payload = take_deopt_payload().expect("DeoptPayload must be published");
    assert!(take_deopt_payload().is_none(), "Second take_deopt_payload must be None");
    assert_eq!(payload.target_pc, bb_do_yield.0);

    let find_local = |loc: LocalId| -> i64 {
        payload
            .live_locals
            .iter()
            .find(|(idx, _)| *idx == loc.0)
            .map(|(_, v)| *v)
            .expect("live local present")
    };
    assert_eq!(find_local(acc_loc), 1275, "Sum(0..50) = 1275 at deopt");
    assert_eq!(find_local(i_loc), 50, "i must be 50 at deopt");
    assert_eq!(find_local(iterations_loc), 100, "iterations must be 100 at deopt");
}

#[test]
fn test_w3_4_8_c_sensor_reading_binary_decode_and_truncated_error() {
    let _lock = TEST_MUTEX.lock().unwrap();
    clear_registered_struct_layouts();
    reset_native_instrumentation_counters();
    take_last_schema_decode_error();

    let logical = LogicalLayoutV1::new_with_abi(
        202,
        "SensorReading",
        vec![
            LogicalFieldDesc::new(0, "timestamp", CanonicalTypeTag::Int64),
            LogicalFieldDesc::new(1, "temperature", CanonicalTypeTag::Float64),
            LogicalFieldDesc::new(2, "sensor_id", CanonicalTypeTag::Int64),
        ],
        TargetAbi::x86_64(),
    );
    let layout = MirStructLayout::from(&logical);
    assert_eq!(layout.size_bytes, 24);
    assert_eq!(layout.fields[0].byte_offset, 0);
    assert_eq!(layout.fields[1].byte_offset, 8);
    assert_eq!(layout.fields[2].byte_offset, 16);

    let payload: [u8; 24] = [
        // timestamp: 1700000000i64 (0x000000006553f100)
        0x00, 0xf1, 0x53, 0x65, 0x00, 0x00, 0x00, 0x00,
        // temperature: 98.6f64 (0x4058a66666666666)
        0x66, 0x66, 0x66, 0x66, 0x66, 0xa6, 0x58, 0x40,
        // sensor_id: 42i64 (0x000000000000002a)
        0x2a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    let mut compiler = CraneliftJitCompiler::new().expect("Init Cranelift compiler");
    let decode_ptr = compiler
        .compile_binary_struct_decoder(&layout)
        .expect("Compile binary decoder");
    let load_ts_ptr = compiler
        .compile_struct_field_loader(&layout, 0)
        .expect("Compile timestamp loader");
    let load_temp_ptr = compiler
        .compile_struct_field_loader(&layout, 1)
        .expect("Compile temperature loader");
    let load_id_ptr = compiler
        .compile_struct_field_loader(&layout, 2)
        .expect("Compile sensor_id loader");

    let native_decode: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(decode_ptr) };
    let native_load_ts: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(load_ts_ptr) };
    let native_load_temp: extern "C" fn(i64) -> f64 = unsafe { std::mem::transmute(load_temp_ptr) };
    let native_load_id: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(load_id_ptr) };

    let ctx = MirContext::new();
    {
        let _guard = enter_mir_context(&ctx);

        let struct_ptr = native_decode(payload.as_ptr() as i64, payload.len() as i64);
        assert_ne!(struct_ptr, 0, "Valid 24-byte payload must decode");

        let ts = native_load_ts(struct_ptr);
        let temp = native_load_temp(struct_ptr);
        let id = native_load_id(struct_ptr);

        assert_eq!(ts, 1_700_000_000);
        assert!((temp - 98.6_f64).abs() < 1e-12, "Expected 98.6, got {temp}");
        assert_eq!(id, 42);

        assert!(
            CRANELIFT_NATIVE_ENTRY_COUNT.load(Ordering::SeqCst) >= 1,
            "CRANELIFT_NATIVE_ENTRY_COUNT must be >= 1"
        );
        assert!(
            DIRECT_DECODE_COUNT.load(Ordering::SeqCst) >= 1,
            "DIRECT_DECODE_COUNT must be >= 1"
        );
        assert_eq!(
            DYNAMIC_LOOKUP_COUNT.load(Ordering::SeqCst),
            0,
            "DYNAMIC_LOOKUP_COUNT must be 0"
        );

        // Malformed truncated 16-byte payload
        let buf_count_before = ctx.allocated_buffers.lock().unwrap().len();
        let err_ptr = native_decode(payload[..16].as_ptr() as i64, 16);
        assert_eq!(err_ptr, 0, "Truncated payload must fail closed with null");
        assert_eq!(
            ctx.allocated_buffers.lock().unwrap().len(),
            buf_count_before,
            "Failed decode must allocate zero buffers"
        );
        assert_eq!(
            take_last_schema_decode_error(),
            Some(SchemaDecodeError::TruncatedPayload {
                expected: 24,
                actual: 16,
            })
        );
    }
    drop(ctx);
}

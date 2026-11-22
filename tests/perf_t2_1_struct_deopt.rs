#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::path::PathBuf;
use std::process::Command;
use std::rc::Rc;
use std::sync::Mutex;

use lust::ast::{Span, Type, TypeKind};
use lust::bytecode::{FieldStorage, StructLayout};
#[cfg(feature = "test-deopt")]
use lust::bytecode::{StructObject, Value};
use lust::mir::cranelift_backend::{
    safepoint_reset, take_current_sticky_error, take_deopt_payload,
};
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::{
    DeoptPayload, DeoptSiteKind, get_current_sticky_error, lust_mir_alloc_struct_named,
    push_native_invocation_with_generation, safepoint_request_deopt_on_poll,
};
#[cfg(feature = "test-deopt")]
use lust::mir::ir::{
    FunctionSignature, MirBinaryOp, MirConstant, MirFunction, MirType, Operand, Place, Rvalue,
    Statement, Terminator,
};
use lust::vfs::VirtualConsole;
use lust::vm::cranelift_dispatch::validate_and_decode_struct;
#[cfg(feature = "test-deopt")]
use lust::vm::{DeoptPayloadSource, TestDeoptSpikeHook, VmBackend};
use lust::vm::{VM, reset_effect_handler_state};
use lust::{EmbeddedModule, LustConfig, compile_program_with_config, load_program_from_embedded};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn lust_bin() -> &'static str {
    env!("CARGO_BIN_EXE_lust")
}

fn fields_lust_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks")
        .join("suite")
        .join("fields.lust")
}

fn reset_all_state() {
    safepoint_reset();
    reset_effect_handler_state();
    take_deopt_payload();
    take_current_sticky_error();
}

fn parse_cranelift_stats(stderr: &str) -> (u64, u64) {
    for line in stderr.lines() {
        if let Some(rest) = line.strip_prefix("Cranelift stats:") {
            let mut entries = None;
            let mut compilations = None;
            for tok in rest.split_whitespace() {
                if let Some(v) = tok.strip_prefix("entries=") {
                    entries = v.parse::<u64>().ok();
                } else if let Some(v) = tok.strip_prefix("compilations=") {
                    compilations = v.parse::<u64>().ok();
                }
            }
            if let (Some(e), Some(c)) = (entries, compilations) {
                return (e, c);
            }
        }
    }
    panic!("Missing `Cranelift stats: entries=... compilations=...` in stderr:\n{stderr}");
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

#[cfg(feature = "test-deopt")]
fn make_point_struct_defs() -> lust::hashbrown::HashMap<String, lust::ast::StructDef> {
    let span = Span::dummy();
    let mut defs = lust::hashbrown::HashMap::new();
    defs.insert(
        "Point".to_string(),
        lust::ast::StructDef {
            name: "Point".to_string(),
            type_params: vec![],
            trait_bounds: vec![],
            fields: vec![
                lust::ast::StructField {
                    name: "x".to_string(),
                    ty: Type::new(TypeKind::Int, span),
                    ownership: lust::ast::FieldOwnership::Strong,
                    weak_target: None,
                    visibility: lust::ast::Visibility::Public,
                },
                lust::ast::StructField {
                    name: "y".to_string(),
                    ty: Type::new(TypeKind::Int, span),
                    ownership: lust::ast::FieldOwnership::Strong,
                    weak_target: None,
                    visibility: lust::ast::Visibility::Public,
                },
            ],
            visibility: lust::ast::Visibility::Public,
            doc: None,
        },
    );
    defs
}

fn make_point_layout() -> Rc<StructLayout> {
    let span = Span::dummy();
    Rc::new(StructLayout::new(
        "Point".to_string(),
        vec![Rc::new("x".to_string()), Rc::new("y".to_string())],
        vec![FieldStorage::Strong, FieldStorage::Strong],
        vec![Type::new(TypeKind::Int, span), Type::new(TypeKind::Int, span)],
        vec![None, None],
    ))
}

#[test]
fn test_perf_t2_1_struct_live_in_loop_compiles_and_deopts() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let fields_path = fields_lust_path();
    let fields_src = std::fs::read_to_string(&fields_path).expect("read benchmarks/suite/fields.lust");

    // 1. CLI execution: `lust --jit=cranelift` with `LUST_JIT_STATS=1` on benchmarks/suite/fields.lust
    let output = Command::new(lust_bin())
        .args(["--jit=cranelift"])
        .arg(&fields_path)
        .env("LUST_JIT_STATS", "1")
        .output()
        .expect("run lust --jit=cranelift fields.lust");
    assert!(
        output.status.success(),
        "lust --jit=cranelift fields.lust failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(stdout, "50000005000000\n", "fields.lust stdout mismatch");
    let (cli_entries, cli_compilations) = parse_cranelift_stats(&stderr);
    assert!(
        cli_entries >= 1 && cli_compilations >= 1,
        "Expected fields.lust to compile and enter Cranelift (entries >= 1, compilations >= 1), got entries={cli_entries}, compilations={cli_compilations}\nstderr:\n{stderr}"
    );

    // 2. VM API execution of fields.lust
    let mut vm = compile_embedded_vm(&fields_src, true);
    vm.call("__script", vec![]).expect("VM::call __script for fields.lust");
    let vm_stdout = String::from_utf8(vm.console.stdout_bytes()).expect("utf8 stdout");
    assert_eq!(vm_stdout, "50000005000000\n");
    assert!(
        vm.cranelift_entry_count >= 1 && vm.cranelift_compilation_count >= 1,
        "VM API fields.lust must compile & enter Cranelift (entries={}, compilations={})",
        vm.cranelift_entry_count,
        vm.cranelift_compilation_count
    );

    // 3. Under `test-deopt`, poll-1 deopt inside a Point struct loop must reconstruct
    //    the live Point local and resume in interpreter to produce exact oracle result.
    #[cfg(feature = "test-deopt")]
    {
        reset_all_state();
        let src = r#"
struct Point
    x: int
    y: int
end

function sum_point_loop(n: int): int
    local p: Point = Point { x = 0, y = 0 }
    local i: int = 1
    while i <= n do
        p.x = i
        p.y = p.y + i
        i = i + 1
    end
    return p.y + p.x
end
"#;
        let mut vm_oracle = compile_embedded_vm(src, false);
        let mut vm_clif = compile_embedded_vm(src, true);
        let fn_name = vm_clif
            .mir_functions
            .keys()
            .find(|k| k.ends_with("sum_point_loop"))
            .cloned()
            .expect("sum_point_loop in mir_functions");

        let expected = vm_oracle
            .call(&fn_name, vec![Value::Int(20)])
            .expect("oracle call");
        assert_eq!(expected, Value::Int(230)); // sum(1..=20) + 20 = 210 + 20

        vm_clif.compile_cranelift().expect("compile_cranelift");
        assert!(
            vm_clif.cranelift_compiled.contains_key(&fn_name),
            "sum_point_loop with live Point struct must compile to Cranelift"
        );

        // Request deopt at loop poll ordinal 1 (after initial Point creation, mid-loop)
        safepoint_request_deopt_on_poll(Some(fn_name.clone()), 1);
        let entries_before = vm_clif.cranelift_entry_count;
        let deopt_res = vm_clif
            .call(&fn_name, vec![Value::Int(20)])
            .expect("poll-1 deopt on Point struct loop must resume cleanly");
        safepoint_reset();

        assert_eq!(
            vm_clif.cranelift_entry_count,
            entries_before + 1,
            "sum_point_loop must enter Cranelift natively"
        );
        assert_eq!(
            vm_clif.last_deopt_source,
            Some(DeoptPayloadSource::NativeExecution),
            "deopt must come from native safepoint"
        );
        assert_eq!(deopt_res, expected, "post-deopt result must match oracle");
        assert!(take_deopt_payload().is_none());
        assert!(get_current_sticky_error().is_none());
    }
}

#[test]
fn test_perf_t2_1_validate_and_decode_struct_field_count_mismatch_direct() {
    let ctx = lust::mir::runtime_helpers::MirContext::new();
    let _guard = ctx.enter();
    let ptr = lust::mir::cranelift_backend::lust_mir_alloc_struct_named(
        1,
        "Point".as_ptr(),
        "Point".len() as i64,
    );
    assert!(!ptr.is_null());
    let layout = make_point_layout(); // expects 2 fields (x, y)
    let err = validate_and_decode_struct(&ctx, &[], ptr as i64, "Point", Some(&layout))
        .expect_err("struct header len=1 vs layout 2 fields must fail closed");
    assert!(
        err.to_string().contains("field count") || err.to_string().contains("Malformed struct"),
        "unexpected error: {err}"
    );
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_perf_t2_1_adv_corrupted_unowned_struct_ptr_rejected_fail_closed() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let fn_name = "struct_deopt_corrupt_ptr";
    let span = Span::dummy();
    let mut f = MirFunction::new(
        FunctionSignature {
            name: fn_name.into(),
            param_types: vec![MirType::Struct("Point".into())],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let p = f.params[0];
    let d = f.add_local(Some("d".into()), MirType::Int, false, span);
    let x_val = f.add_local(Some("x_val".into()), MirType::Int, false, span);
    let out = f.add_local(Some("out".into()), MirType::Int, false, span);
    let bb1 = f.new_block();
    let bb2 = f.new_block();

    // Mutate p.x = 777 before deopt seam so we can prove zero writeback on rejected payload!
    f.blocks[0].statements.push(Statement::Assign(
        Place::Field(Box::new(Place::Local(p)), 0),
        Rvalue::Use(Operand::Constant(MirConstant::Int(777))),
        span,
    ));
    f.blocks[0].terminator = Some(Terminator::Goto(bb1, span));
    f.blocks[bb1.0].terminator = Some(Terminator::Call {
        callee: "effect.perform".into(),
        args: vec![],
        destination: Place::Local(d),
        success: bb2,
        error: None,
        span,
    });
    f.blocks[bb2.0].statements.push(Statement::Assign(
        Place::Local(x_val),
        Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(p)), 0))),
        span,
    ));
    f.blocks[bb2.0].statements.push(Statement::Assign(
        Place::Local(out),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(d)),
            Operand::Copy(Place::Local(x_val)),
        ),
        span,
    ));
    f.blocks[bb2.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(out))),
        span,
    ));

    let layout = make_point_layout();
    for bad_ptr in [0_i64, 0x0000_dead_beef_i64] {
        reset_all_state();
        let mut vm = VM::new();
        vm.set_backend(VmBackend::Cranelift);
        vm.register_native(
            "effect.perform",
            Value::NativeFunction(lust::bytecode::native_fn(|_| {
                Ok(lust::bytecode::NativeCallResult::Return(Value::Int(1)))
            })),
        );
        vm.register_structs(&make_point_struct_defs());
        vm.load_mir_function(f.clone()).expect("load_mir_function");

        // Install authorized spike hook BEFORE compile so `effect.perform` is authorized
        vm.test_deopt_spike_hook = Some(TestDeoptSpikeHook::new(fn_name, bb1.0));
        vm.compile_cranelift().expect("compile_cranelift");
        assert!(
            vm.cranelift_compiled.contains_key(fn_name),
            "struct_deopt_corrupt_ptr must compile to Cranelift"
        );

        let compiled_gen = vm.cranelift_compiled_generation();
        let next_inv_id = vm.cranelift_entry_count + 1;
        let bad_payload = {
            let _inv = push_native_invocation_with_generation(
                fn_name.into(),
                1,
                next_inv_id,
                compiled_gen,
            );
            DeoptPayload::at_pre_terminator(bb1.0, vec![(p.0, bad_ptr)])
        };
        vm.test_deopt_spike_hook.as_mut().unwrap().injected_payload = Some(bad_payload);

        let point_obj = StructObject::new(
            "Point".to_string(),
            layout.clone(),
            vec![Value::Int(11), Value::Int(22)],
        );
        let err = vm
            .call(fn_name, vec![Value::Struct(point_obj.clone())])
            .expect_err("corrupted/unowned struct pointer in deopt payload must fail closed");
        let msg = err.to_string();
        assert!(
            msg.contains("provenance") || msg.contains("null pointer") || msg.contains("Invalid struct"),
            "expected provenance error for bad_ptr={bad_ptr:#x}, got: {msg}"
        );
        // Zero mutation on caller's Point struct!
        assert_eq!(
            *point_obj.fields.borrow(),
            vec![Value::Int(11), Value::Int(22)],
            "caller Point must remain unmutated after rejected deopt payload"
        );
    }
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_perf_t2_1_adv_struct_field_count_mismatch_rejected_fail_closed() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let fn_name = "struct_deopt_field_count_mismatch";
    let span = Span::dummy();
    let mut f = MirFunction::new(
        FunctionSignature {
            name: fn_name.into(),
            param_types: vec![MirType::Struct("Point".into())],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let p = f.params[0];
    let d = f.add_local(Some("d".into()), MirType::Int, false, span);
    let x_val = f.add_local(Some("x_val".into()), MirType::Int, false, span);
    let bb1 = f.new_block();
    let bb2 = f.new_block();

    f.blocks[0].statements.push(Statement::Assign(
        Place::Field(Box::new(Place::Local(p)), 0),
        Rvalue::Use(Operand::Constant(MirConstant::Int(888))),
        span,
    ));
    f.blocks[0].terminator = Some(Terminator::Goto(bb1, span));
    f.blocks[bb1.0].terminator = Some(Terminator::Call {
        callee: "effect.perform".into(),
        args: vec![],
        destination: Place::Local(d),
        success: bb2,
        error: None,
        span,
    });
    f.blocks[bb2.0].statements.push(Statement::Assign(
        Place::Local(x_val),
        Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(p)), 0))),
        span,
    ));
    f.blocks[bb2.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(x_val))),
        span,
    ));

    let mut vm = VM::new();
    vm.set_backend(VmBackend::Cranelift);
    vm.register_native(
        "effect.perform",
        Value::NativeFunction(lust::bytecode::native_fn(|_| {
            Ok(lust::bytecode::NativeCallResult::Return(Value::Int(0)))
        })),
    );
    vm.register_structs(&make_point_struct_defs());
    vm.load_mir_function(f).expect("load_mir_function");
    vm.test_deopt_spike_hook = Some(TestDeoptSpikeHook::new(fn_name, bb1.0));
    vm.compile_cranelift().expect("compile_cranelift");
    assert!(vm.cranelift_compiled.contains_key(fn_name));

    // Allocate an owned `Point` buffer in the VM's MirContext with len=1 (mismatched vs layout len=2).
    // Note: clear_to_baselines only clears allocations above baseline, so allocate before capture_baseline
    // or verify mismatched header fails closed (provenance or field count mismatch).
    let wrong_len_ptr = {
        let _g = vm.mir_context.enter();
        lust_mir_alloc_struct_named(1, "Point".as_ptr(), "Point".len() as i64)
    };
    assert!(!wrong_len_ptr.is_null());

    let compiled_gen = vm.cranelift_compiled_generation();
    let next_inv_id = vm.cranelift_entry_count + 1;
    let bad_payload = {
        let _inv = push_native_invocation_with_generation(
            fn_name.into(),
            1,
            next_inv_id,
            compiled_gen,
        );
        DeoptPayload::at_pre_terminator(bb1.0, vec![(p.0, wrong_len_ptr as i64)])
            .with_site_kind(DeoptSiteKind::PreTerminator)
    };
    vm.test_deopt_spike_hook.as_mut().unwrap().injected_payload = Some(bad_payload);

    let layout = make_point_layout();
    let point_obj = StructObject::new(
        "Point".to_string(),
        layout,
        vec![Value::Int(33), Value::Int(44)],
    );
    let err = vm
        .call(fn_name, vec![Value::Struct(point_obj.clone())])
        .expect_err("mismatched struct field count / scope in deopt payload must fail closed");
    let msg = err.to_string();
    assert!(
        msg.contains("field count")
            || msg.contains("Malformed struct")
            || msg.contains("provenance"),
        "unexpected error: {msg}"
    );
    assert_eq!(
        *point_obj.fields.borrow(),
        vec![Value::Int(33), Value::Int(44)],
        "caller Point must remain unmutated after rejected deopt payload"
    );
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_perf_t2_1_adv_aliased_fresh_struct_preserves_rc_identity_post_deopt() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    // Two live locals `a` and `b` alias the SAME freshly allocated Point inside a loop.
    // Poll-1 deopt resumes in interpreter; mutating `a.x` MUST be observed through `b.x`!
    let src = r#"
struct Point
    x: int
    y: int
end

function aliased_point_loop(n: int): int
    local a: Point = Point { x = 1, y = 10 }
    local b: Point = a
    local acc: int = 0
    local i: int = 0
    while i < n do
        a.x = a.x + 1
        acc = acc + b.x
        b.y = b.y + 2
        acc = acc + a.y
        i = i + 1
    end
    return acc
end
"#;

    let mut vm_oracle = compile_embedded_vm(src, false);
    let mut vm_clif = compile_embedded_vm(src, true);
    let fn_name = vm_clif
        .mir_functions
        .keys()
        .find(|k| k.ends_with("aliased_point_loop"))
        .cloned()
        .expect("aliased_point_loop in mir_functions");

    let expected = vm_oracle
        .call(&fn_name, vec![Value::Int(10)])
        .expect("oracle call");

    vm_clif.compile_cranelift().expect("compile_cranelift");
    assert!(
        vm_clif.cranelift_compiled.contains_key(&fn_name),
        "aliased_point_loop must compile to Cranelift"
    );

    safepoint_request_deopt_on_poll(Some(fn_name.clone()), 1);
    let entries_before = vm_clif.cranelift_entry_count;
    let deopt_res = vm_clif
        .call(&fn_name, vec![Value::Int(10)])
        .expect("poll-1 deopt on aliased fresh struct must succeed");
    safepoint_reset();

    assert_eq!(vm_clif.cranelift_entry_count, entries_before + 1);
    assert_eq!(
        vm_clif.last_deopt_source,
        Some(DeoptPayloadSource::NativeExecution)
    );
    assert_eq!(
        deopt_res, expected,
        "mutating a.x / b.y after deopt MUST be observed through alias b.x / a.y (Rc::ptr_eq identity)"
    );
}

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::path::PathBuf;
use std::sync::Mutex;

use lust::ast::Span;
#[cfg(feature = "test-deopt")]
use lust::bytecode::Value;
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::{safepoint_request_deopt_on_poll, safepoint_stats};
use lust::mir::cranelift_backend::{
    safepoint_reset, take_current_sticky_error, take_deopt_payload, CraneliftJitCompiler,
};
use lust::mir::ir::{LocalDecl, LocalId, MirType};
use lust::mir::opt::escape_analysis::{requires_explicit_stack_slot, EscapeAnalysis};
use lust::mir::runtime_helpers::{enter_mir_context, MirContext};
use lust::vfs::VirtualConsole;
use lust::vm::{reset_effect_handler_state, VmBackend, VM};
use lust::{compile_program_with_config, load_program_from_embedded, EmbeddedModule, LustConfig};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn suite_source(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks")
        .join("suite")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

fn compile_embedded_vm(source: &str, cranelift: bool) -> VM {
    let entries = &[EmbeddedModule {
        module: "main",
        parent: None,
        source: Some(source),
    }];
    let program = load_program_from_embedded(entries, "main").expect("load embedded program");
    let mut cfg = LustConfig::default().with_cranelift_jit(cranelift);
    cfg.set_use_mir(true);
    let mut vm = compile_program_with_config(program, &cfg).expect("compile embedded program");
    vm.console = VirtualConsole::captured();
    vm.set_backend(if cranelift {
        VmBackend::Cranelift
    } else {
        VmBackend::Bytecode
    });
    vm
}

fn reset_all_state() {
    safepoint_reset();
    reset_effect_handler_state();
    take_deopt_payload();
    take_current_sticky_error();
}

#[test]
fn test_ng_w9_3_requires_explicit_stack_slot_classification() {
    let make_local = |ty: MirType, promoted: bool| {
        let mut decl = LocalDecl::new(LocalId(0), None, ty, true, Span::dummy());
        decl.is_stack_promoted = promoted;
        decl
    };

    for ty in [
        MirType::Int,
        MirType::Float,
        MirType::Bool,
        MirType::Unit,
        MirType::String,
        MirType::Option(Box::new(MirType::Struct("Node".into()))),
        MirType::Struct("Node".into()),
        MirType::Array(Box::new(MirType::Int)),
    ] {
        assert!(
            !requires_explicit_stack_slot(&make_local(ty.clone(), true)),
            "promoted {ty:?} must stay in Cranelift SSA register without explicit stack slot"
        );
        assert!(!requires_explicit_stack_slot(&make_local(ty, false)));
    }

    let tuple_ty = MirType::Tuple(vec![MirType::Int, MirType::Int]);
    assert!(requires_explicit_stack_slot(&make_local(tuple_ty.clone(), true)));
    assert!(!requires_explicit_stack_slot(&make_local(tuple_ty, false)));
}

#[test]
fn test_ng_w9_3_floatmath_cranelift_ir_has_no_explicit_stack_slots_for_scalars() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = suite_source("floatmath.lust");
    let vm = compile_embedded_vm(&src, true);
    let mut floatmath_mir_fn = vm
        .mir_functions
        .get("__script")
        .expect("__script MIR function must be present")
        .clone();

    // EscapeAnalysis must still mark non-escaping scalar locals as `is_stack_promoted = true`
    EscapeAnalysis::run_on_function(&mut floatmath_mir_fn);
    assert!(
        floatmath_mir_fn.locals.iter().any(|l| l.is_stack_promoted),
        "EscapeAnalysis must continue setting is_stack_promoted on non-escaping locals"
    );

    let ctx = MirContext::new();
    let _guard = enter_mir_context(&ctx);
    let mut compiler = CraneliftJitCompiler::new().expect("init Cranelift compiler");
    let ir = compiler
        .dump_function_ir(&floatmath_mir_fn)
        .expect("dump_function_ir must succeed");

    assert!(
        !ir.contains("explicit_slot"),
        "floatmath IR must not allocate explicit_slot for scalar locals:\n{ir}"
    );
    assert!(
        !ir.contains("stack_store"),
        "floatmath IR must not emit stack_store for scalar locals:\n{ir}"
    );
    assert!(
        !ir.contains("stack_load"),
        "floatmath IR must not emit stack_load for scalar locals:\n{ir}"
    );
}

#[test]
fn test_ng_w9_3_floatmath_and_nested_vm_cranelift_bitwise_parity() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    // 1. floatmath.lust IEEE-754 bitwise parity across Rust oracle, Bytecode VM, and Cranelift JIT
    let mut x = 0.0_f64;
    let mut acc = 0.0_f64;
    for _ in 0..10_000_000 {
        x = x + 0.001;
        acc = acc + x * x - acc * 0.5;
    }
    let floatmath_src = suite_source("floatmath.lust");
    let mut vm_bc_float = compile_embedded_vm(&floatmath_src, false);
    vm_bc_float
        .call("__script", vec![])
        .expect("floatmath bytecode VM execution");
    let bc_float_out = vm_bc_float.console.stdout_string();

    let mut vm_clif_float = compile_embedded_vm(&floatmath_src, true);
    vm_clif_float
        .call("__script", vec![])
        .expect("floatmath Cranelift JIT execution");
    let clif_float_out = vm_clif_float.console.stdout_string();

    assert!(
        vm_clif_float.cranelift_entry_count >= 1,
        "floatmath must execute in Cranelift JIT"
    );
    let bc_val: f64 = bc_float_out
        .trim()
        .parse()
        .expect("floatmath VM output must parse as f64");
    let clif_val: f64 = clif_float_out
        .trim()
        .parse()
        .expect("floatmath JIT output must parse as f64");
    assert_eq!(
        bc_val.to_bits(),
        clif_val.to_bits(),
        "floatmath VM vs Cranelift must match bit-for-bit: vm={bc_val:?}, jit={clif_val:?}"
    );
    assert_eq!(
        clif_val.to_bits(),
        acc.to_bits(),
        "floatmath Cranelift output must match IEEE-754 oracle bit-for-bit: jit={clif_val:?}, oracle={acc:?}"
    );

    // 2. nested.lust exact oracle parity across Rust, Bytecode VM, and Cranelift JIT
    let mut nested_oracle: i64 = 0;
    for i in 0..3000_i64 {
        for j in 0..3000_i64 {
            nested_oracle += (i * j) % 7;
        }
    }
    let nested_src = suite_source("nested.lust");
    let mut vm_bc_nested = compile_embedded_vm(&nested_src, false);
    vm_bc_nested
        .call("__script", vec![])
        .expect("nested bytecode VM execution");
    let bc_nested_out = vm_bc_nested.console.stdout_string();

    let mut vm_clif_nested = compile_embedded_vm(&nested_src, true);
    vm_clif_nested
        .call("__script", vec![])
        .expect("nested Cranelift JIT execution");
    let clif_nested_out = vm_clif_nested.console.stdout_string();

    assert!(
        vm_clif_nested.cranelift_entry_count >= 1,
        "nested must execute in Cranelift JIT"
    );
    let clif_nested_val: i64 = clif_nested_out
        .trim()
        .parse()
        .expect("nested JIT output must parse as i64");
    assert_eq!(clif_nested_val, nested_oracle);
    assert_eq!(
        bc_nested_out.trim(),
        clif_nested_out.trim(),
        "nested VM vs Cranelift must match identically"
    );
}

#[cfg(feature = "test-deopt")]
#[test]
fn test_ng_w9_3_deopt_on_5th_iteration_restores_int_float_bool_from_ssa() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
        function deopt_ssa_scalars(n: int): float
            local i: int = 0
            local acc_int: int = 10
            local acc_flt: float = 1.25
            local flag: bool = false
            while i < n do
                acc_int = acc_int + i * 3 + 1
                if flag then
                    acc_flt = acc_flt + 0.5
                else
                    acc_flt = acc_flt + 1.5
                end
                flag = not flag
                i = i + 1
            end
            local k: int = acc_int % 50
            while k > 0 do
                acc_flt = acc_flt + 0.01
                k = k - 1
            end
            if flag then
                return acc_flt + 100.0
            end
            return acc_flt
        end
    "#;

    let mut vm_bc = compile_embedded_vm(src, false);
    let bc_res = vm_bc
        .call("deopt_ssa_scalars", vec![Value::Int(12)])
        .expect("bytecode oracle must succeed");

    let mut vm_clif = compile_embedded_vm(src, true);
    // Verify the compiled function uses SSA registers (no stack slots for scalars)
    let mir_fn = vm_clif
        .mir_functions
        .get("deopt_ssa_scalars")
        .expect("deopt_ssa_scalars MIR function present")
        .clone();
    {
        let ctx = MirContext::new();
        let _guard = enter_mir_context(&ctx);
        let mut compiler = CraneliftJitCompiler::new().expect("init Cranelift compiler");
        let ir = compiler
            .dump_function_ir(&mir_fn)
            .expect("dump_function_ir must succeed");
        assert!(!ir.contains("explicit_slot"));
        assert!(!ir.contains("stack_store"));
        assert!(!ir.contains("stack_load"));
    }

    safepoint_request_deopt_on_poll(Some("deopt_ssa_scalars".to_string()), 5);
    let clif_res = vm_clif
        .call("deopt_ssa_scalars", vec![Value::Int(12)])
        .expect("deopt on 5th iteration must resume cleanly in interpreter");
    let stats = safepoint_stats();
    safepoint_reset();

    assert!(
        vm_clif.cranelift_entry_count >= 1,
        "function must enter Cranelift JIT natively before deopting"
    );
    assert_eq!(
        stats.safepoints_checked, 5,
        "deopt must fire on the 5th loop poll"
    );

    match (bc_res, clif_res) {
        (Value::Float(b), Value::Float(c)) => {
            assert_eq!(
                c.to_bits(),
                b.to_bits(),
                "post-deopt result must match oracle bit-for-bit: bc={b}, clif={c}"
            );
        }
        (b, c) => panic!("expected Float values, got bc={b:?}, clif={c:?}"),
    }
}

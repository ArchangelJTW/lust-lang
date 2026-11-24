//! Integration & regression tests for Wave 15 Task `NG-W15-7`:
//! - Cranelift VM dispatch eligibility & `MirType::Map` / `MirType::Enum` / `MirType::Option` /
//!   `MirType::Array(String | Struct)` deopt recipe support and live deopt roundtrips
//! - Sub-frame trivial leaf method JIT marshaling guard (`BENCH-NG-05` `megamorphic_method_ic`)
//! - 5- and 6-argument direct scalar FFI fast path (`BENCH-NG-15` `evaluate_tick_5arg`) preserving
//!   `SEC-01` gas enforcement

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Mutex;

use lust::ast::Span;
use lust::bytecode::{EnumObject, LustMap, Value, ValueKey};
use lust::mir::cranelift_backend::{
    reset_native_instrumentation_counters, safepoint_reset, take_current_sticky_error,
    take_deopt_payload, CRANELIFT_NATIVE_ENTRY_COUNT,
};
#[cfg(not(feature = "test-deopt"))]
use lust::mir::cranelift_backend::safepoint_request_deopt;
#[cfg(feature = "test-deopt")]
use lust::mir::cranelift_backend::safepoint_request_deopt_on_poll;
use lust::mir::ir::{
    FunctionSignature, MirBinaryOp, MirConstant, MirFunction, MirType, Operand, Place, Rvalue,
    Statement, Terminator,
};
use lust::mir::vm_backend::MirToBytecodeCompiler;
use lust::vfs::VirtualConsole;
use lust::vm::{reset_effect_handler_state, VmBackend, VM};
use lust::{
    compile_program_with_config, load_program_from_embedded, EmbeddedModule, EmbeddedProgram,
    LustConfig,
};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn reset_all_state() {
    safepoint_reset();
    reset_effect_handler_state();
    reset_native_instrumentation_counters();
    take_deopt_payload();
    take_current_sticky_error();
}

fn suite_v2_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("benchmarks/suite_v2")
        .join(name)
}

fn build_vm(source: &str, cranelift: bool) -> VM {
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

fn compile_cranelift_program(src: &str) -> EmbeddedProgram {
    let mut cfg = LustConfig::default().with_cranelift_jit(true);
    cfg.set_use_mir(false);
    EmbeddedProgram::builder()
        .with_config(cfg)
        .module("main", src)
        .entry_module("main")
        .compile()
        .expect("compile Cranelift EmbeddedProgram")
}

fn resolve_fn_name(vm: &VM, suffix: &str) -> String {
    let qualified = format!("main.{suffix}");
    if vm.function_by_name(&qualified).is_some() {
        qualified
    } else if vm.function_by_name(suffix).is_some() {
        suffix.to_string()
    } else {
        vm.mir_functions
            .keys()
            .find(|k| k.ends_with(suffix))
            .cloned()
            .unwrap_or_else(|| panic!("function matching '{suffix}' not found in VM"))
    }
}

// =========================================================================
// Test 1: BENCH-NG-05 Sub-Frame Trivial Leaf Method JIT Marshaling Guard
// =========================================================================

#[test]
fn test_ng_w15_7_megamorphic_method_ic_cranelift_backend_eliminates_1m_ffi_entries() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = std::fs::read_to_string(suite_v2_path("megamorphic_method_ic.lust"))
        .expect("read benchmarks/suite_v2/megamorphic_method_ic.lust");

    let mut vm_jit = build_vm(&src, true);
    vm_jit
        .call("__script", vec![])
        .expect("run megamorphic_method_ic under Cranelift");

    let stdout = String::from_utf8(vm_jit.console.stdout_bytes().to_vec())
        .expect("utf8 stdout")
        .trim()
        .to_string();
    assert_eq!(
        stdout, "24980750",
        "BENCH-NG-05 oracle checksum must match verbatim"
    );
    assert_eq!(
        vm_jit.cranelift_entry_count(),
        0,
        "sub-frame trivial leaf method guard must eliminate all 1,000,000 VM<->Cranelift transitions"
    );

    // Positive control: a top-level call (`call_stack.len() == 1`) to a trivial method OR a
    // sub-frame call to a looping method MUST still enter Cranelift JIT.
    let control_src = r#"
struct Counter
    x: int
end

impl Counter
    function leaf(self, k: int): int
        return self.x + k
    end

    function looped(self, n: int): int
        local acc: int = self.x
        local i: int = 0
        while i < n do
            acc = acc + i
            i = i + 1
        end
        return acc
    end
end

function make_counter(x: int): Counter
    return Counter { x = x }
end

function run_looped_subframe(obj: unknown, n: int): int
    return obj:looped(n)
end
"#;
    let mut vm_ctrl = build_vm(control_src, true);
    let make_name = resolve_fn_name(&vm_ctrl, "make_counter");
    let c = vm_ctrl
        .call(&make_name, vec![Value::Int(10)])
        .expect("make_counter(10)");

    let leaf_name = resolve_fn_name(&vm_ctrl, "Counter:leaf");
    let entries_before_top = vm_ctrl.cranelift_entry_count();
    let top_res = vm_ctrl
        .call(&leaf_name, vec![c.clone(), Value::Int(5)])
        .expect("top-level Counter:leaf call");
    assert_eq!(top_res, Value::Int(15));
    assert_eq!(
        vm_ctrl.cranelift_entry_count(),
        entries_before_top + 1,
        "top-level call to leaf method must still execute natively in Cranelift"
    );

    let driver_name = resolve_fn_name(&vm_ctrl, "run_looped_subframe");
    let entries_before_loop = vm_ctrl.cranelift_entry_count();
    let loop_res = vm_ctrl
        .call(&driver_name, vec![c, Value::Int(10)])
        .expect("sub-frame call to looping method");
    assert_eq!(loop_res, Value::Int(55));
    assert_eq!(
        vm_ctrl.cranelift_entry_count(),
        entries_before_loop + 1,
        "sub-frame call to method with a loop must still execute natively in Cranelift"
    );
}

// =========================================================================
// Test 2: Map / Enum / Option / Array<String|Struct> Deopt Recipe & Roundtrip
// =========================================================================

#[test]
fn test_ng_w15_7_map_enum_and_array_struct_pass_cranelift_eligibility_and_deopt_roundtrip() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    // Part A: `is_supported_deopt_recipe_type` positive and negative checks
    let cell_src = r#"
struct Cell
    val: int
    next: Option<Cell>
end

function touch(c: Cell): int
    return c.val
end
"#;
    let vm = build_vm(cell_src, true);
    let cell_name = if vm.is_supported_deopt_recipe_type(&MirType::Struct("main.Cell".into())) {
        "main.Cell"
    } else {
        "Cell"
    };

    // Positive: newly supported deopt recipe types
    assert!(vm.is_supported_deopt_recipe_type(&MirType::Map(
        Box::new(MirType::Int),
        Box::new(MirType::Int)
    )));
    assert!(vm.is_supported_deopt_recipe_type(&MirType::Map(
        Box::new(MirType::String),
        Box::new(MirType::Int)
    )));
    assert!(vm.is_supported_deopt_recipe_type(&MirType::Enum("Op".into())));
    assert!(vm.is_supported_deopt_recipe_type(&MirType::Enum("Option".into())));
    assert!(vm.is_supported_deopt_recipe_type(&MirType::Option(Box::new(MirType::Int))));
    assert!(vm.is_supported_deopt_recipe_type(&MirType::Option(Box::new(MirType::Bool))));
    assert!(vm.is_supported_deopt_recipe_type(&MirType::Option(Box::new(MirType::Float))));
    assert!(vm.is_supported_deopt_recipe_type(&MirType::Option(Box::new(MirType::String))));
    assert!(vm.is_supported_deopt_recipe_type(&MirType::Option(Box::new(MirType::Struct(
        cell_name.into()
    )))));
    assert!(vm.is_supported_deopt_recipe_type(&MirType::Array(Box::new(MirType::String))));
    assert!(vm.is_supported_deopt_recipe_type(&MirType::Array(Box::new(MirType::Struct(
        cell_name.into()
    )))));

    // Negative fail-closed checks
    assert!(!vm.is_supported_deopt_recipe_type(&MirType::Unknown));
    assert!(!vm.is_supported_deopt_recipe_type(&MirType::Enum(String::new())));
    assert!(!vm.is_supported_deopt_recipe_type(&MirType::Map(
        Box::new(MirType::Unknown),
        Box::new(MirType::Int)
    )));
    assert!(!vm.is_supported_deopt_recipe_type(&MirType::Array(Box::new(MirType::Struct(
        "Unregistered".into()
    )))));
    assert!(!vm.is_supported_deopt_recipe_type(&MirType::Option(Box::new(MirType::Struct(
        "Unregistered".into()
    )))));

    // Part B: Live Cranelift JIT -> VM deopt roundtrip with a live `MirType::Map(Int, Int)` local
    let map_src = r#"
function map_loop_sum(m: Map<int, int>, n: int): int
    local acc: int = 0
    local i: int = 0
    while i < n do
        acc = acc + map.len(m) + i
        i = i + 1
    end
    return acc
end
"#;
    let mut vm_bc = build_vm(map_src, false);
    let mut vm_jit = build_vm(map_src, true);

    let mut raw_map = LustMap::default();
    raw_map.insert(ValueKey::from(Value::Int(1)), Value::Int(100));
    raw_map.insert(ValueKey::from(Value::Int(2)), Value::Int(200));
    raw_map.insert(ValueKey::from(Value::Int(3)), Value::Int(300));
    let map_val_bc = Value::map(raw_map.clone());
    let map_val_jit = Value::map(raw_map);

    let fn_name = resolve_fn_name(&vm_jit, "map_loop_sum");

    let expected = vm_bc
        .call(&fn_name, vec![map_val_bc, Value::Int(12)])
        .expect("vm map_loop_sum");
    assert_eq!(expected, Value::Int(3 * 12 + 66));

    // Trigger safepoint deopt while `m: Map<int, int>` is live at the loop header!
    #[cfg(feature = "test-deopt")]
    safepoint_request_deopt_on_poll(Some(fn_name.clone()), 2);
    #[cfg(not(feature = "test-deopt"))]
    safepoint_request_deopt(true);

    let actual = vm_jit
        .call(&fn_name, vec![map_val_jit.clone(), Value::Int(12)])
        .expect("jit map_loop_sum deopt roundtrip must succeed");
    safepoint_reset();

    assert_eq!(
        actual, expected,
        "deopt roundtrip with live Map<int, int> must preserve exact return value"
    );
    assert_eq!(
        vm_jit.cranelift_entry_count(),
        1,
        "map_loop_sum must have entered Cranelift JIT before deopting"
    );
    assert!(
        vm_jit.cranelift_lazy_deopt_materialization_count() >= 1,
        "deopt must materialize continuation & recipe with live Map local"
    );
    if let Value::Map(m_rc) = &map_val_jit {
        assert_eq!(m_rc.borrow().len(), 3, "live Map handle identity preserved");
    } else {
        panic!("expected Value::Map");
    }

    // Part C: Live Cranelift JIT -> VM deopt roundtrip with a live `MirType::Enum("Op")` local
    let span = Span::dummy();
    let mut vm_enum = VM::new();
    vm_enum.set_backend(VmBackend::Cranelift);

    let mut enum_fn = MirFunction::new(
        FunctionSignature {
            name: "enum_loop_deopt".into(),
            param_types: vec![MirType::Enum("Op".into()), MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        span,
    );
    let e_param = enum_fn.params[0];
    let n_param = enum_fn.params[1];
    let acc = enum_fn.add_local(Some("acc".into()), MirType::Int, true, span);
    let idx = enum_fn.add_local(Some("i".into()), MirType::Int, true, span);
    let cond = enum_fn.add_local(Some("cond".into()), MirType::Bool, true, span);
    let tag_loc = enum_fn.add_local(Some("tag".into()), MirType::Int, true, span);

    let bb_hdr = enum_fn.new_block();
    let bb_body = enum_fn.new_block();
    let bb_after_tag = enum_fn.new_block();
    let bb_exit = enum_fn.new_block();

    enum_fn.blocks[0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        span,
    ));
    enum_fn.blocks[0].statements.push(Statement::Assign(
        Place::Local(idx),
        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
        span,
    ));
    enum_fn.blocks[0].terminator = Some(Terminator::Goto(bb_hdr, span));

    enum_fn.blocks[bb_hdr.0].statements.push(Statement::Assign(
        Place::Local(cond),
        Rvalue::BinaryOp(
            MirBinaryOp::Lt,
            Operand::Copy(Place::Local(idx)),
            Operand::Copy(Place::Local(n_param)),
        ),
        span,
    ));
    enum_fn.blocks[bb_hdr.0].terminator = Some(Terminator::Cond {
        cond: Operand::Copy(Place::Local(cond)),
        then_block: bb_body,
        else_block: bb_exit,
        span,
    });

    enum_fn.blocks[bb_body.0].terminator = Some(Terminator::Call {
        callee: "enum.tag".into(),
        args: vec![Operand::Copy(Place::Local(e_param))],
        destination: Place::Local(tag_loc),
        success: bb_after_tag,
        error: None,
        span,
    });

    enum_fn.blocks[bb_after_tag.0].statements.push(Statement::Assign(
        Place::Local(acc),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(acc)),
            Operand::Copy(Place::Local(tag_loc)),
        ),
        span,
    ));
    enum_fn.blocks[bb_after_tag.0].statements.push(Statement::Assign(
        Place::Local(idx),
        Rvalue::BinaryOp(
            MirBinaryOp::Add,
            Operand::Copy(Place::Local(idx)),
            Operand::Constant(MirConstant::Int(1)),
        ),
        span,
    ));
    enum_fn.blocks[bb_after_tag.0].terminator = Some(Terminator::Goto(bb_hdr, span));

    enum_fn.blocks[bb_exit.0].terminator = Some(Terminator::Return(
        Some(Operand::Copy(Place::Local(acc))),
        span,
    ));

    let enum_bc = MirToBytecodeCompiler::compile(&enum_fn).expect("compile enum_loop_deopt bc");
    vm_enum.load_functions(vec![enum_bc]);
    vm_enum.register_mir_candidate(enum_fn);

    let op_val = Value::Enum(EnumObject::new("Op", "Variant3", Some(vec![Value::Int(42)])));

    // Register `enum.tag` global fallback for post-deopt interpreter continuation
    vm_enum.register_native(
        "enum.tag".to_string(),
        Value::NativeFunction(lust::bytecode::native_fn(|args: &[Value]| {
            let tag = match args.first() {
                Some(Value::Enum(e)) => e
                    .variant
                    .as_str()
                    .strip_prefix("Variant")
                    .and_then(|s| s.parse::<i64>().ok())
                    .unwrap_or(0),
                _ => 0,
            };
            Ok(lust::bytecode::NativeCallResult::Return(Value::Int(tag)))
        })),
    );

    #[cfg(feature = "test-deopt")]
    safepoint_request_deopt_on_poll(Some("enum_loop_deopt".into()), 2);
    #[cfg(not(feature = "test-deopt"))]
    safepoint_request_deopt(true);

    let enum_res = vm_enum
        .call("enum_loop_deopt", vec![op_val, Value::Int(10)])
        .expect("jit enum_loop_deopt deopt roundtrip must succeed");
    safepoint_reset();

    assert_eq!(
        enum_res,
        Value::Int(30),
        "deopt roundtrip with live MirType::Enum local must compute 10 * 3 = 30"
    );
    assert_eq!(vm_enum.cranelift_entry_count(), 1);
    assert!(vm_enum.cranelift_lazy_deopt_materialization_count() >= 1);
}

// =========================================================================
// Test 3: 5-Arg & 6-Arg Direct Scalar FFI Fast Path + SEC-01 Gas Gate
// =========================================================================

#[test]
fn test_ng_w15_7_direct_ffi_5arg_and_6arg_fast_dispatch() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_all_state();

    let src = r#"
function evaluate_tick_5arg(price: int, qty: int, side: int, fee_bps: int, limit_p: int): int
    local notional: int = price * qty
    local fee: int = (notional * fee_bps) / 10000
    if side == 1 then
        if price <= limit_p then
            return notional - fee
        else
            return 0
        end
    else
        if price >= limit_p then
            return notional + fee
        else
            return 0
        end
    end
end

function sum6_int(a: int, b: int, c: int, d: int, e: int, f: int): int
    return a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f
end

function sum6_float(a: float, b: float, c: float, d: float, e: float, f: float): float
    return a + b + c + d + e + f
end

function loop5_gas(a: int, b: int, c: int, d: int, n: int): int
    local acc: int = a + b + c + d
    local i: int = 0
    while i < n do
        acc = acc + i
        i = i + 1
    end
    return acc
end

function loop6_gas(a: int, b: int, c: int, d: int, e: int, n: int): int
    local acc: int = a + b + c + d + e
    local i: int = 0
    while i < n do
        acc = acc + i
        i = i + 1
    end
    return acc
end
"#;

    let mut program = compile_cranelift_program(src);

    // 1. 5-arg `evaluate_tick_5arg` via `call_typed` (cold compile + warm fast path)
    let g0 = CRANELIFT_NATIVE_ENTRY_COUNT.load(Ordering::SeqCst);
    let tick1: i64 = program
        .call_typed("main.evaluate_tick_5arg", (105i64, 12i64, 1i64, 15i64, 110i64))
        .expect("call_typed evaluate_tick_5arg #1");
    let g1 = CRANELIFT_NATIVE_ENTRY_COUNT.load(Ordering::SeqCst);
    // notional = 1260, fee = (1260 * 15) / 10000 = 1, side==1 & 105<=110 => 1259
    assert_eq!(tick1, 1259);
    assert_eq!(g1 - g0, 1, "5-arg call_typed must invoke JIT prologue once");
    assert_eq!(program.cranelift_native_entry_count(), 1);
    assert_eq!(
        program
            .vm()
            .cranelift_is_pure_scalar_abi_6("main.evaluate_tick_5arg"),
        Some(true),
        "5-arg scalar int function must be marked cranelift_is_pure_scalar_abi_6"
    );
    assert_eq!(program.vm().mir_context.dirty_allocations_count(), 0);

    // 2. 5-arg `evaluate_tick_5arg` via `TypedHandle` (`call_handle`)
    let h_tick = program
        .get_typed_handle::<(i64, i64, i64, i64, i64), i64>("main.evaluate_tick_5arg")
        .expect("get_typed_handle evaluate_tick_5arg");
    let g2 = CRANELIFT_NATIVE_ENTRY_COUNT.load(Ordering::SeqCst);
    let tick2 = program
        .call_handle(&h_tick, (115i64, 10i64, 0i64, 20i64, 110i64))
        .expect("call_handle evaluate_tick_5arg");
    let g3 = CRANELIFT_NATIVE_ENTRY_COUNT.load(Ordering::SeqCst);
    // notional = 1150, fee = (1150 * 20) / 10000 = 2, side==0 & 115>=110 => 1152
    assert_eq!(tick2, 1152);
    assert_eq!(g3 - g2, 1);
    assert_eq!(program.cranelift_native_entry_count(), 2);
    assert_eq!(program.vm().mir_context.dirty_allocations_count(), 0);

    // 3. 6-arg `sum6_int` via `call_typed` and `call_handle`
    let g4 = CRANELIFT_NATIVE_ENTRY_COUNT.load(Ordering::SeqCst);
    let s6: i64 = program
        .call_typed("main.sum6_int", (1i64, 2i64, 3i64, 4i64, 5i64, 6i64))
        .expect("call_typed sum6_int");
    let g5 = CRANELIFT_NATIVE_ENTRY_COUNT.load(Ordering::SeqCst);
    assert_eq!(s6, 123456);
    assert_eq!(g5 - g4, 1);
    assert_eq!(
        program.vm().cranelift_is_pure_scalar_abi_6("main.sum6_int"),
        Some(true)
    );
    assert_eq!(program.vm().mir_context.dirty_allocations_count(), 0);

    let h_s6 = program
        .get_typed_handle::<(i64, i64, i64, i64, i64, i64), i64>("main.sum6_int")
        .expect("get_typed_handle sum6_int");
    let s6_h = program
        .call_handle(&h_s6, (6i64, 5i64, 4i64, 3i64, 2i64, 1i64))
        .expect("call_handle sum6_int");
    assert_eq!(s6_h, 654321);
    assert_eq!(program.cranelift_native_entry_count(), 4);
    assert_eq!(program.vm().mir_context.dirty_allocations_count(), 0);

    // 4. 6-arg `sum6_float` via `call_typed`
    let sf6: f64 = program
        .call_typed(
            "main.sum6_float",
            (1.5f64, 2.5f64, 3.0f64, 4.0f64, 5.25f64, 6.75f64),
        )
        .expect("call_typed sum6_float");
    assert!((sf6 - 23.0).abs() < 1e-12);
    assert_eq!(
        program.vm().cranelift_is_pure_scalar_abi_6("main.sum6_float"),
        Some(true)
    );
    assert_eq!(program.cranelift_native_entry_count(), 5);
    assert_eq!(program.vm().mir_context.dirty_allocations_count(), 0);

    // 5. SEC-01 gas enforcement on 5-arg and 6-arg direct FFI fast paths
    let h_loop5 = program
        .get_typed_handle::<(i64, i64, i64, i64, i64), i64>("main.loop5_gas")
        .expect("get_typed_handle loop5_gas");
    let h_loop6 = program
        .get_typed_handle::<(i64, i64, i64, i64, i64, i64), i64>("main.loop6_gas")
        .expect("get_typed_handle loop6_gas");

    // Warm up
    assert_eq!(
        program
            .call_handle(&h_loop5, (1, 2, 3, 4, 3))
            .expect("warm loop5"),
        13
    );
    assert_eq!(
        program
            .call_handle(&h_loop6, (1, 2, 3, 4, 5, 3))
            .expect("warm loop6"),
        18
    );

    // Exact gas exhaustion at gas_limit = 0, 1, 5
    for limit in [0u64, 1u64, 5u64] {
        program.reset_gas_counter();
        program.set_gas_budget(limit);
        let err5 = program
            .call_handle(&h_loop5, (1, 2, 3, 4, 20))
            .expect_err("loop5_gas(20) must trap OutOfGas under tight budget");
        assert!(
            err5.to_string().contains("Out of gas"),
            "expected Out of gas for loop5 limit={limit}, got: {err5}"
        );

        program.reset_gas_counter();
        program.set_gas_budget(limit);
        let err6 = program
            .call_handle(&h_loop6, (1, 2, 3, 4, 5, 20))
            .expect_err("loop6_gas(20) must trap OutOfGas under tight budget");
        assert!(
            err6.to_string().contains("Out of gas"),
            "expected Out of gas for loop6 limit={limit}, got: {err6}"
        );
    }

    // Sufficient budget succeeds and charges exact gas > 0
    program.reset_gas_counter();
    program.set_gas_budget(100);
    let ok6 = program
        .call_handle(&h_loop6, (1, 2, 3, 4, 5, 10))
        .expect("loop6_gas(10) with gas=100 must succeed");
    assert_eq!(ok6, 15 + 45);
    assert!(
        program.gas_used() > 0 && program.gas_used() <= 100,
        "gas_used must be charged on 6-arg direct FFI fast path (got {})",
        program.gas_used()
    );
    program.clear_gas_budget();
}

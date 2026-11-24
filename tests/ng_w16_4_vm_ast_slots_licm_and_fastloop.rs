#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! NG-W16-4 Acceptance & Verification Suite:
//! 1. AST bytecode compiler emits `GetFieldSlot`, `SetFieldSlot`, and fused `GetFieldAddSetInt`
//!    for statically typed structs (`fields.lust`, `methods.lust`), and hoists `LoadBool`
//!    (`sieve.lust`) and invariant `<function@N>` (`calls.lust`) out of `while` loops.
//! 2. All-leaf structs (`Vec3` in `sroa_escape_structs.lust`) skip `CycleCollector` registration
//!    (`tracked_count() == 0`), while mutating a leaf struct to hold a container transitions it
//!    out of leaf state and registers it for cycle collection.
//! 3. In-loop `Value::NativeFunction` dispatch (`string.sub`, `string.len`, `string.split`) and
//!    `ChunkIcSlot::ModuleMapNative` IC keep hot native-calling loops inside `'frame_fast` with
//!    `0` `FrameExit::Slow` ejections (and invalidate soundly on map mutation).
//! 4. `--backend cranelift` fallback (`allow_inloop_calls = true` + `osr_exhausted_funcs`) never
//!    degrades below `--jit=off` (`<= 1.05x` wall-clock time on ineligible nested-call + loop
//!    workloads, with zero steady-state `FrameExit::Slow` / `FrameExit::TryOsr` thrashing).

use lust::ast::ItemKind;
use lust::bytecode::chunk::ChunkIcSlot;
use lust::bytecode::compiler::Compiler;
use lust::bytecode::{Function, Instruction, NativeCallResult, Value, native_fn};
use lust::config::LustConfig;
use lust::hashbrown::HashMap;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{VM, VmBackend};
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

fn compile_ast_module(
    source: &str,
    collect_mir: bool,
) -> (
    Vec<Function>,
    HashMap<String, lust::ast::StructDef>,
    Vec<lust::mir::ir::MirFunction>,
) {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");

    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    // Note: `set_cranelift_jit(true)` implicitly enables `use_mir`, so clear
    // `use_mir` afterwards to keep AST bytecode identical between `--jit=off`
    // and `--backend cranelift` while still collecting MIR candidates.
    config.set_cranelift_jit(collect_mir);
    config.set_use_mir(false);

    let mut checker = TypeChecker::with_config(&config);
    checker.check_module(&items).expect("typecheck");
    let struct_defs = checker.struct_definitions();

    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_collect_mir(collect_mir);
    compiler.set_option_coercions(checker.take_option_coercions());
    compiler.set_checked_array_indices(checker.take_checked_array_indices());
    compiler.set_lambda_return_types(checker.take_lambda_return_types());
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    compiler.set_struct_defs(struct_defs.clone());

    let functions = compiler.compile_module(&items).expect("compile_module");
    let mir_functions = compiler.take_mir_functions();
    let mut all_structs = struct_defs;
    for item in &items {
        if let ItemKind::Struct(def) = &item.kind {
            all_structs
                .entry(def.name.clone())
                .or_insert_with(|| def.clone());
        }
    }
    (functions, all_structs, mir_functions)
}

fn build_vm(
    source: &str,
    backend: VmBackend,
) -> (VM, Rc<RefCell<Vec<Value>>>) {
    let collect_mir = backend == VmBackend::Cranelift;
    let (functions, struct_defs, mir_functions) = compile_ast_module(source, collect_mir);
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    config.set_cranelift_jit(collect_mir);
    let mut vm = VM::with_config(&config);
    vm.set_backend(backend);
    vm.register_structs(&struct_defs);
    vm.load_functions(functions);
    if collect_mir {
        for mir_fn in mir_functions {
            vm.register_mir_candidate(mir_fn);
        }
    }
    let output = Rc::new(RefCell::new(Vec::new()));
    let captured = Rc::clone(&output);
    vm.register_native(
        "println",
        Value::NativeFunction(native_fn(move |args| {
            captured.borrow_mut().extend_from_slice(args);
            Ok(NativeCallResult::Return(Value::Nil))
        })),
    );
    (vm, output)
}

fn find_fn<'a>(functions: &'a [Function], name: &str) -> &'a Function {
    functions
        .iter()
        .find(|f| f.name == name)
        .unwrap_or_else(|| panic!("function '{name}' not found"))
}

#[test]
fn test_ng_w16_4_ast_compiler_emits_field_slots_and_hoists_loadbool() {
    // 1. fields.lust: must emit GetFieldSlot, SetFieldSlot, and fused GetFieldAddSetInt
    let fields_src = include_str!("../../../benchmarks/suite/fields.lust");
    let (fields_fns, _, _) = compile_ast_module(fields_src, false);
    let fields_script = find_fn(&fields_fns, "__script");
    let fields_insns = &fields_script.chunk.instructions;

    assert!(
        fields_insns
            .iter()
            .any(|i| matches!(i, Instruction::GetFieldSlot(..))),
        "AST-compiled fields.lust must emit GetFieldSlot:\n{}",
        fields_script.disassemble()
    );
    assert!(
        fields_insns
            .iter()
            .any(|i| matches!(i, Instruction::SetFieldSlot(..))),
        "AST-compiled fields.lust must emit SetFieldSlot:\n{}",
        fields_script.disassemble()
    );
    assert!(
        fields_insns
            .iter()
            .any(|i| matches!(i, Instruction::GetFieldAddSetInt(..))),
        "AST-compiled fields.lust must fuse `point.y = point.y + i` into GetFieldAddSetInt:\n{}",
        fields_script.disassemble()
    );
    assert!(
        !fields_insns
            .iter()
            .any(|i| matches!(i, Instruction::GetField(..) | Instruction::SetField(..))),
        "AST-compiled fields.lust should not fall back to generic GetField/SetField:\n{}",
        fields_script.disassemble()
    );

    // 2. methods.lust: Counter:bump must also emit slot opcodes on `self`
    let methods_src = include_str!("../../../benchmarks/suite/methods.lust");
    let (methods_fns, _, _) = compile_ast_module(methods_src, false);
    let bump_fn = find_fn(&methods_fns, "Counter:bump");
    assert!(
        bump_fn
            .chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::GetFieldAddSetInt(..))),
        "Counter:bump must emit GetFieldAddSetInt for `self.value = self.value + d`:\n{}",
        bump_fn.disassemble()
    );
    assert!(
        bump_fn
            .chunk
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::GetFieldSlot(..))),
        "Counter:bump must emit GetFieldSlot for `return self.value`:\n{}",
        bump_fn.disassemble()
    );

    // 3. sieve.lust: LoadBool(_, false) must be hoisted BEFORE the inner `while j <= N` loop header
    let sieve_src = include_str!("../../../benchmarks/suite/sieve.lust");
    let (sieve_fns, _, _) = compile_ast_module(sieve_src, false);
    let sieve_script = find_fn(&sieve_fns, "__script");
    let sieve_insns = &sieve_script.chunk.instructions;

    // Identify the inner while loop: a backward Jump whose body contains SetIndex.
    let mut inner_loop_range: Option<(usize, usize)> = None;
    for (pc, inst) in sieve_insns.iter().enumerate() {
        if let Instruction::Jump(off) = *inst
            && off < 0
        {
            let target = (pc as isize + 1 + off as isize) as usize;
            let body = &sieve_insns[target..=pc];
            let has_set_index = body.iter().any(|i| matches!(i, Instruction::SetIndex(..)));
            if has_set_index {
                // Pick the innermost loop containing SetIndex (smallest span).
                if inner_loop_range.is_none_or(|(s, e)| (pc - target) < (e - s)) {
                    inner_loop_range = Some((target, pc));
                }
            }
        }
    }
    let (inner_header, inner_backedge) = inner_loop_range.unwrap_or_else(|| {
        panic!(
            "Failed to locate inner while loop with SetIndex in sieve.lust:\n{}",
            sieve_script.disassemble()
        )
    });
    assert!(
        sieve_insns[..inner_header]
            .iter()
            .any(|i| matches!(i, Instruction::LoadBool(_, false))),
        "LoadBool(_, false) must be hoisted before inner loop header pc={inner_header}:\n{}",
        sieve_script.disassemble()
    );
    assert!(
        !sieve_insns[inner_header..=inner_backedge]
            .iter()
            .any(|i| matches!(i, Instruction::LoadBool(..))),
        "Inner while loop [{inner_header}..={inner_backedge}] must contain ZERO LoadBool instructions:\n{}",
        sieve_script.disassemble()
    );

    // 4. calls.lust: direct function constant `<function@0>` hoisted before the while loop header
    let calls_src = include_str!("../../../benchmarks/suite/calls.lust");
    let (calls_fns, _, _) = compile_ast_module(calls_src, false);
    let calls_script = find_fn(&calls_fns, "__script");
    let calls_insns = &calls_script.chunk.instructions;
    let calls_backedge = calls_insns
        .iter()
        .enumerate()
        .find_map(|(pc, inst)| match *inst {
            Instruction::Jump(off) if off < 0 => {
                Some(((pc as isize + 1 + off as isize) as usize, pc))
            }
            _ => None,
        })
        .expect("while loop in calls.lust");
    let is_fn_load_const = |inst: &Instruction| match *inst {
        Instruction::LoadConst(_, idx) => matches!(
            calls_script.chunk.constants.get(idx as usize),
            Some(Value::Function(_))
        ),
        _ => false,
    };
    assert!(
        calls_insns[..calls_backedge.0].iter().any(is_fn_load_const),
        "calls.lust must hoist LoadConst <function> before loop header pc={}:\n{}",
        calls_backedge.0,
        calls_script.disassemble()
    );
    assert!(
        !calls_insns[calls_backedge.0..=calls_backedge.1]
            .iter()
            .any(is_fn_load_const),
        "calls.lust loop body must not reload LoadConst <function>:\n{}",
        calls_script.disassemble()
    );

    // 5. Runtime parity check on compound field assignment + small sieve
    let parity_src = r#"
struct Counter
    x: int
    y: int
end
function run_test(): int
    local c: Counter = Counter { x = 1, y = 2 }
    local i: int = 1
    while i <= 100 do
        c.x = i
        c.y += i
        c.y = c.y + i
        i = i + 1
    end
    local flags: Array<bool> = []
    local k: int = 0
    while k <= 100 do
        array.push(flags, true)
        k = k + 1
    end
    local primes: int = 0
    k = 2
    while k <= 100 do
        if flags[k] is Ok(f) then
            if f then
                primes = primes + 1
                local j: int = k * k
                while j <= 100 do
                    flags[j] = false
                    j = j + k
                end
            end
        end
        k = k + 1
    end
    return c.x + c.y + primes
end
"#;
    let (mut vm, _) = build_vm(parity_src, VmBackend::Bytecode);
    // c.x = 100, c.y = 2 + 2*(5050) = 10102, primes(<=100) = 25 => 10227
    assert_eq!(vm.call("run_test", vec![]).unwrap(), Value::Int(10_227));
}

#[test]
fn test_ng_w16_4_leaf_struct_vec3_skips_cycle_collector_registration() {
    let raw_src = include_str!("../../../benchmarks/suite_v2/sroa_escape_structs.lust");
    // Use full 1M iterations in release, scaled 10k in debug so unit runs stay sub-second.
    let (src, expected_checksum): (String, i64) = if cfg!(debug_assertions) {
        (raw_src.replace("1000000", "10000"), 149_867_321)
    } else {
        (raw_src.to_string(), 990_177_872)
    };
    let (mut vm, output) = build_vm(&src, VmBackend::Bytecode);
    // Stdlib module maps are registered during VM initialization before script execution.
    let tracked_before = vm.cycle_collector_tracked_count();
    vm.call("__script", vec![]).expect("run sroa_escape_structs");

    // Verify exact checksum in release (and non-zero deterministic value in debug)
    if !cfg!(debug_assertions) {
        assert_eq!(&*output.borrow(), &[Value::Int(expected_checksum)]);
    } else {
        assert_eq!(output.borrow().len(), 1);
        assert!(matches!(output.borrow()[0], Value::Int(n) if n > 0));
    }
    assert_eq!(
        vm.cycle_collector_tracked_count() - tracked_before,
        0,
        "All-leaf Vec3 structs in sroa_escape_structs.lust must add 0 CycleCollector registrations"
    );

    // Also run the exact Vec3 transform loop in a PreludeMode::Pure VM (no stdlib maps)
    // and assert literal `vm.cycle_collector_tracked_count() == 0`.
    let pure_src = r#"
struct Vec3
    x: int
    y: int
    z: int
end
function transform(v: Vec3, k: int): Vec3
    return Vec3 {
        x = (v.x + k) % 10007,
        y = (v.y * 3 + v.x) % 10007,
        z = (v.z + v.y + k) % 10007
    }
end
function run_vec3(n: int): int
    local cur: Vec3 = Vec3 { x = 1, y = 2, z = 3 }
    local i: int = 0
    local acc: int = 0
    while i < n do
        cur = transform(cur, i)
        acc = (acc + cur.x + cur.y + cur.z) % 1000000007
        i = i + 1
    end
    return acc
end
"#;
    let (pure_fns, pure_structs, _) = compile_ast_module(pure_src, false);
    let mut pure_cfg = LustConfig::default().with_prelude_mode(lust::config::PreludeMode::Pure);
    pure_cfg.set_jit_enabled(false);
    let mut pure_vm = VM::with_config(&pure_cfg);
    pure_vm.set_backend(VmBackend::Bytecode);
    pure_vm.register_structs(&pure_structs);
    pure_vm.load_functions(pure_fns);
    let pure_before = pure_vm.cycle_collector_tracked_count();
    let pure_res = pure_vm
        .call("run_vec3", vec![Value::Int(2_000)])
        .expect("run_vec3");
    assert!(matches!(pure_res, Value::Int(n) if n > 0));
    assert_eq!(
        pure_vm.cycle_collector_tracked_count() - pure_before,
        0,
        "Running 2,000 Vec3 allocations must register 0 Vec3 structs in CycleCollector"
    );
    // Also directly verify `is_leaf_struct` flag on a constructed Vec3 instance.
    let vec3_val = pure_vm
        .instantiate_struct_ordered("Vec3", vec![Value::Int(1), Value::Int(2), Value::Int(3)])
        .expect("instantiate Vec3");
    match &vec3_val {
        Value::Struct(s) => assert!(
            s.is_leaf_struct.get(),
            "All-int Vec3 struct must have is_leaf_struct == true"
        ),
        other => panic!("Expected Value::Struct(Vec3), got {:?}", other),
    }

    // Soundness check: a struct starting with all-leaf fields that is later mutated to hold a
    // container / struct reference MUST clear `is_leaf_struct` and register in CycleCollector.
    let cycle_src = r#"
struct Node
    val: int
    next: Option<Node>
end
function make_cycle(): int
    local a: Node = Node { val = 1, next = Option.None }
    local b: Node = Node { val = 2, next = Option.None }
    a.next = Option.Some(b)
    b.next = Option.Some(a)
    return a.val + b.val
end
"#;
    let (mut cycle_vm, _) = build_vm(cycle_src, VmBackend::Bytecode);
    let cycle_before = cycle_vm.cycle_collector_tracked_count();
    assert_eq!(cycle_vm.call("make_cycle", vec![]).unwrap(), Value::Int(3));
    assert!(
        cycle_vm.cycle_collector_tracked_count() > cycle_before,
        "Mutating a leaf struct to hold a non-leaf Option<Node> must register it in CycleCollector"
    );
}

#[test]
fn test_ng_w16_4_inloop_native_function_and_module_map_ic_zero_frame_exits() {
    let src = r#"
function string_hot_loop(n: int): int
    local s: string = "alpha,beta,gamma,delta"
    local i: int = 0
    local acc: int = 0
    while i < n do
        local part: string = string.sub(s, 1, 5)
        acc = acc + string.len(part)
        i = i + 1
    end
    return acc
end
"#;
    let (mut vm, _) = build_vm(src, VmBackend::Bytecode);
    // Warm up once to populate global/module-map IC slots if needed, then verify 0 FrameExit::Slow
    // even across the cold + hot call!
    VM::reset_frame_exit_slow_count();
    let res = vm
        .call("string_hot_loop", vec![Value::Int(2000)])
        .expect("run string_hot_loop");
    assert_eq!(res, Value::Int(10_000));
    assert_eq!(
        VM::frame_exit_slow_count(),
        0,
        "Loop calling string.sub and string.len under --jit=off must have 0 FrameExit::Slow ejections"
    );

    // Verify ModuleMapNative IC slots were populated for string.sub and string.len
    let func = vm.function_by_name("string_hot_loop").unwrap();
    let module_map_ic_count = func
        .chunk
        .ic_slots
        .borrow()
        .iter()
        .filter(|slot| matches!(slot, ChunkIcSlot::ModuleMapNative { .. }))
        .count();
    assert!(
        module_map_ic_count >= 2,
        "Expected >= 2 ChunkIcSlot::ModuleMapNative slots (string.sub & string.len), got {module_map_ic_count}"
    );

    // Soundness check: mutating a map entry must invalidate ModuleMapNative IC immediately.
    let mutation_src = r#"
function test_map_ic_invalidation(): int
    local m = {}
    m.f = string.len
    local a: int = m.f("abc")
    m.f = string.sub
    local b: string = m.f("abcdef", 1, 2)
    return a * 100 + string.len(b)
end
"#;
    let (mut mut_vm, _) = build_vm(mutation_src, VmBackend::Bytecode);
    assert_eq!(
        mut_vm.call("test_map_ic_invalidation", vec![]).unwrap(),
        Value::Int(302)
    );
}

#[test]
fn test_ng_w16_4_cranelift_fallback_never_slower_than_jit_off() {
    // An ineligible function (uses `unknown` parameters/values so Cranelift rejects it and falls
    // back to VM bytecode) containing nested calls and an inner loop.
    let src = r#"
function helper_ineligible(a: unknown, b: unknown): unknown
    local x: int = 0
    local j: int = 0
    while j < 4 do
        x = x + j
        j = j + 1
    end
    return a + b + x
end

function driver_ineligible(n: int): unknown
    local acc: unknown = 0
    local i: int = 0
    while i < n do
        acc = helper_ineligible(acc, i)
        i = i + 1
    end
    return acc
end
"#;
    let iters: i64 = if cfg!(debug_assertions) { 5_000 } else { 200_000 };

    // Build both VMs
    let (mut vm_off, _) = build_vm(src, VmBackend::Bytecode);
    let (mut vm_cl, _) = build_vm(src, VmBackend::Cranelift);

    // Warmup + verify parity and Cranelift fallback state (`entries == 0`)
    let res_off = vm_off
        .call("driver_ineligible", vec![Value::Int(iters)])
        .expect("jit=off run");
    VM::reset_frame_exit_slow_count();
    let res_cl = vm_cl
        .call("driver_ineligible", vec![Value::Int(iters)])
        .expect("cranelift fallback run");
    let cl_slow_exits_first = VM::frame_exit_slow_count();

    assert_eq!(res_cl, res_off, "Cranelift fallback and --jit=off must produce identical output");
    assert_eq!(
        vm_cl.cranelift_entry_count, 0,
        "Test function must exercise the Cranelift fallback path (entries=0)"
    );
    assert!(
        cl_slow_exits_first <= 8,
        "Cranelift fallback must negative-cache ineligible callees and stay in 'frame_fast (got {cl_slow_exits_first} slow exits across {iters} calls)"
    );

    // Second run after negative cache warmup must have 0 slow exits!
    VM::reset_frame_exit_slow_count();
    VM::reset_frame_exit_osr_count();
    let res_cl2 = vm_cl
        .call("driver_ineligible", vec![Value::Int(iters)])
        .expect("cranelift fallback run 2");
    assert_eq!(res_cl2, res_off);
    assert_eq!(
        VM::frame_exit_slow_count(),
        0,
        "After negative-cache warmup, Cranelift fallback must have 0 FrameExit::Slow ejections"
    );
    assert_eq!(
        VM::frame_exit_osr_count(),
        0,
        "Ineligible function without OSR entries must have 0 FrameExit::TryOsr ejections"
    );

    // Also verify `osr_exhausted_funcs`: give a function non-empty `block_entry_pcs` whose loop
    // backedge is cold/ineligible for OSR, run once to mark the site cold, and verify the next run
    // has ZERO `FrameExit::TryOsr` ejections.
    let osr_src = r#"
function osr_ineligible_loop(n: int): int
    local i: int = 0
    local acc: int = 0
    while i < n do
        acc = acc + i
        i = i + 1
    end
    return acc
end
"#;
    let (mut osr_fns, osr_structs, _) = compile_ast_module(osr_src, false);
    for f in &mut osr_fns {
        if f.name == "osr_ineligible_loop" {
            // Synthetic non-empty block_entry_pcs (without registering MIR) forces OSR attempt ->
            // cold site insertion -> `osr_exhausted_funcs[func_idx] = true`.
            f.block_entry_pcs.insert(0, 0);
        }
    }
    let mut osr_cfg = LustConfig::default();
    osr_cfg.set_jit_enabled(false);
    osr_cfg.set_cranelift_jit(true);
    let mut osr_vm = VM::with_config(&osr_cfg);
    osr_vm.set_backend(VmBackend::Cranelift);
    osr_vm.register_structs(&osr_structs);
    osr_vm.load_functions(osr_fns);

    // Run once with enough iterations to hit OSR_HOT_BACKEDGE_THRESHOLD and mark site cold.
    let osr_r1 = osr_vm
        .call("osr_ineligible_loop", vec![Value::Int(5_000)])
        .expect("osr warmup run");
    assert_eq!(osr_r1, Value::Int(12_497_500));
    assert!(
        osr_vm.osr_cold_site_count() > 0,
        "Ineligible backedge must be recorded in osr_cold_sites"
    );

    // Second run must have 0 FrameExit::TryOsr ejections because `osr_exhausted_funcs` is true!
    VM::reset_frame_exit_osr_count();
    let osr_r2 = osr_vm
        .call("osr_ineligible_loop", vec![Value::Int(5_000)])
        .expect("osr exhausted run");
    assert_eq!(osr_r2, Value::Int(12_497_500));
    assert_eq!(
        VM::frame_exit_osr_count(),
        0,
        "Once all backedges are in osr_cold_sites, osr_exhausted_funcs must eliminate all FrameExit::TryOsr ejections"
    );

    // Interleaved median timing benchmark: `--backend cranelift` <= 1.05x `--jit=off`
    let samples_count = if cfg!(debug_assertions) { 3 } else { 7 };
    let mut off_times = Vec::with_capacity(samples_count);
    let mut cl_times = Vec::with_capacity(samples_count);
    for _ in 0..samples_count {
        let t0 = Instant::now();
        let r0 = vm_off
            .call("driver_ineligible", vec![Value::Int(iters)])
            .unwrap();
        off_times.push(t0.elapsed().as_secs_f64());
        assert_eq!(r0, res_off);

        let t1 = Instant::now();
        let r1 = vm_cl
            .call("driver_ineligible", vec![Value::Int(iters)])
            .unwrap();
        cl_times.push(t1.elapsed().as_secs_f64());
        assert_eq!(r1, res_off);
    }
    off_times.sort_by(|a, b| a.total_cmp(b));
    cl_times.sort_by(|a, b| a.total_cmp(b));
    let off_med = off_times[off_times.len() / 2];
    let cl_med = cl_times[cl_times.len() / 2];
    let ratio = cl_med / off_med.max(1e-9);
    eprintln!(
        "NG-W16-4 Cranelift fallback vs --jit=off: cl={:.3} ms, off={:.3} ms, ratio={:.3}x",
        cl_med * 1000.0,
        off_med * 1000.0,
        ratio
    );
    assert!(
        ratio <= 1.05,
        "Cranelift fallback must match --jit=off within <= 1.05x (got {ratio:.3}x: cl={:.3}ms vs off={:.3}ms)",
        cl_med * 1000.0,
        off_med * 1000.0,
    );
}

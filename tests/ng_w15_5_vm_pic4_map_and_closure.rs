#![cfg(feature = "std")]

use lust::ast::ItemKind;
use lust::bytecode::chunk::ChunkIcSlot;
use lust::bytecode::compiler::Compiler;
use lust::bytecode::instruction::PackedInstruction;
use lust::bytecode::{Function, Instruction, NativeCallResult, Value, ValueKey, native_fn};
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::ir::{MirModule, MirStructLayout, mir_struct_decode_callee};
use lust::mir::vm_backend::register_module_struct_decoders;
use lust::number::LustInt;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{VM, VmBackend};
use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant};

fn parse(source: &str) -> Vec<lust::ast::Item> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("tokenize");
    Parser::new(tokens).parse().expect("parse")
}

fn compile(source: &str) -> (VM, Rc<RefCell<Vec<Value>>>) {
    let items = parse(source);
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    let mut checker = TypeChecker::with_config(&config);
    checker.check_module(&items).expect("typecheck");
    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    let functions = compiler.compile_module(&items).expect("compile");
    let structs = items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Struct(def) => Some((def.name.clone(), def.clone())),
            _ => None,
        })
        .collect();
    let mut vm = VM::with_config(&config);
    vm.set_backend(VmBackend::Bytecode);
    vm.register_structs(&structs);
    vm.load_functions(functions);
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

fn benchmark(name: &str, source: &str, checksum: LustInt) -> (VM, Duration) {
    let mut samples = Vec::new();
    let mut final_vm = None;
    for _ in 0..if cfg!(debug_assertions) { 1 } else { 5 } {
        let (mut vm, output) = compile(source);
        let started = Instant::now();
        vm.call("__script", vec![]).expect("run benchmark");
        samples.push(started.elapsed());
        assert_eq!(&*output.borrow(), &[Value::Int(checksum)]);
        final_vm = Some(vm);
    }
    samples.sort_unstable();
    let elapsed = samples[samples.len() / 2];
    eprintln!(
        "{name} VM execution median: {:.3} ms",
        elapsed.as_secs_f64() * 1000.0
    );
    (final_vm.unwrap(), elapsed)
}

fn assert_release_time(name: &str, elapsed: Duration, max_ms: u64) {
    if !cfg!(debug_assertions) {
        assert!(
            elapsed <= Duration::from_millis(max_ms),
            "{name} took {:.3} ms, target {max_ms} ms",
            elapsed.as_secs_f64() * 1000.0,
        );
    }
}

#[test]
fn test_ng_w15_5_pic4_achieves_100_percent_hit_rate_on_4_way_polymorphic_dispatch() {
    let (vm, elapsed) = benchmark(
        "BENCH-NG-05",
        include_str!("../../../benchmarks/suite_v2/megamorphic_method_ic.lust"),
        24_980_750,
    );
    let mut slots = vm
        .function_by_name("eval_poly")
        .unwrap()
        .chunk
        .ic_slots
        .borrow_mut();
    let layouts: Vec<_> = [("NodeA", 2), ("NodeB", 3), ("NodeC", 4), ("NodeD", 4)]
        .into_iter()
        .map(|(name, count)| {
            let Value::Struct(object) = vm
                .instantiate_struct_ordered(name, vec![Value::Int(0); count])
                .unwrap()
            else {
                panic!("expected a struct for {name}");
            };
            Rc::clone(&object.layout)
        })
        .collect();
    for (layout, (x, w)) in layouts.iter().zip([(0, 1), (1, 0), (2, 3), (3, 1)]) {
        let field_slots: Vec<_> = slots
            .iter_mut()
            .filter_map(|slot| slot.probe_field(layout))
            .collect();
        assert_eq!(field_slots, [x, w], "both field sites must hit");
        assert_eq!(
            slots
                .iter_mut()
                .filter_map(|slot| slot.probe_method(layout))
                .count(),
            1,
            "the method site must hit",
        );
    }
    assert_release_time("BENCH-NG-05", elapsed, 65);
}

#[test]
fn test_ng_w15_5_enum_variant_tags_cache_positive_and_negative_checks() {
    let (mut vm, _) = benchmark(
        "BENCH-NG-06",
        include_str!("../../../benchmarks/suite_v2/adt_enum_eval.lust"),
        2_262_427,
    );
    assert_eq!(
        vm.function_by_name("apply_op")
            .unwrap()
            .chunk
            .ic_slots
            .borrow()
            .iter()
            .filter(|slot| matches!(slot, ChunkIcSlot::EnumVariantTag { .. }))
            .count(),
        4,
    );
    // The same variant spelling can belong to a different enum. A qualified
    // test must reject it while an explicitly unqualified test accepts it.
    let mut functions = Vec::new();
    for (name, enum_name) in [("matches_op", "Op"), ("matches_any_lit", "")] {
        let mut function = Function::new(name, 1, false);
        let enum_idx = function.chunk.add_constant(Value::string(enum_name));
        let variant_idx = function.chunk.add_constant(Value::string("Lit"));
        function
            .chunk
            .emit(Instruction::IsEnumVariant(1, 0, enum_idx, variant_idx), 1);
        function.chunk.emit(Instruction::Return(1), 1);
        function.set_register_count(2);
        functions.push(function);
    }
    let mut enum_vm = VM::new();
    enum_vm.set_backend(VmBackend::Bytecode);
    enum_vm.load_functions(functions);
    let valid = Value::enum_variant("Op", "Lit", vec![Value::Int(1)]);
    let wrong_enum = Value::enum_variant("Other", "Lit", vec![Value::Int(1)]);
    assert_eq!(
        enum_vm.call("matches_op", vec![valid]).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        enum_vm
            .call("matches_op", vec![wrong_enum.clone()])
            .unwrap(),
        Value::Bool(false)
    );
    assert_eq!(
        enum_vm.call("matches_any_lit", vec![wrong_enum]).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        vm.call(
            "apply_op",
            vec![Value::Int(8), Value::enum_unit("Op", "Reset")]
        )
        .unwrap(),
        Value::Int(4)
    );
}

#[test]
fn test_ng_w15_5_in_loop_closure_call_zero_vec_allocations() {
    #[cfg(debug_assertions)]
    let before = (
        VM::frame_hoist_reload_count(),
        VM::closure_upvalue_buffer_growths(),
    );
    let (_, elapsed) = benchmark(
        "BENCH-NG-04",
        include_str!("../../../benchmarks/suite_v2/closure_upvalue_combinators.lust"),
        947_911,
    );
    #[cfg(debug_assertions)]
    {
        assert!(VM::frame_hoist_reload_count() - before.0 < 20);
        // Wave 16 NG-W16-4 eliminated the duplicate frame.upvalues Vec<Value> buffer
        // (keeping only frame.shared_upvalues), so warmup buffer growth count is 1.
        assert_eq!(VM::closure_upvalue_buffer_growths() - before.1, 1);
    }
    assert_release_time("BENCH-NG-04", elapsed, 95);
}

#[test]
fn test_ng_w15_5_bytecode_struct_decode_and_map_get_unwrap_fast_opcodes() {
    let mut timings = Vec::new();
    for (name, source, checksum, target_ms) in [
        (
            "BENCH-NG-01",
            include_str!("../../../benchmarks/suite_v2/map_int_churn.lust"),
            608_439_931,
            140,
        ),
        (
            "BENCH-NG-02",
            include_str!("../../../benchmarks/suite_v2/map_str_intern_hist.lust"),
            379_372_375,
            125,
        ),
    ] {
        let (vm, elapsed) = benchmark(name, source, checksum);
        timings.push((name, elapsed, target_ms));
        let chunk = &vm.function_by_name("__script").unwrap().chunk;
        assert!(
            chunk
                .instructions
                .iter()
                .any(|inst| matches!(inst, Instruction::MapSet(..)))
        );
        assert!(
            chunk
                .instructions
                .iter()
                .any(|inst| matches!(inst, Instruction::MapGetUnwrap(..)))
        );
        assert!(
            !chunk
                .instructions
                .iter()
                .any(|inst| matches!(inst, Instruction::MapGet(..)))
        );
    }

    for instruction in [
        Instruction::MapGet(14, 212, 255),
        Instruction::MapSet(144, 21, 220),
        Instruction::MapGetUnwrap(1, 3, 248),
    ] {
        let mut extra = Vec::new();
        let packed = PackedInstruction::from_instruction(instruction, &mut extra).unwrap();
        assert!(extra.is_empty());
        assert_eq!(packed.to_instruction(&extra).unwrap(), instruction);
        assert_eq!(
            Instruction::decode(&instruction.encode()).unwrap(),
            instruction
        );
    }

    let source = r#"
        struct TelemetryEvent ts: int, device_id: int, metric: int, active: bool end
        println(TelemetryEvent.decode("{\"ts\":1700000100,\"device_id\":42,\"metric\":950,\"active\":true}"))
    "#;
    let (mut vm, output) = compile(source);
    let mut module = MirModule::new("test".into());
    for item in parse(source) {
        if let ItemKind::Struct(def) = item.kind {
            let layout = lust::data::struct_def_to_layout(&def);
            module.register_struct_layout(MirStructLayout::from(&layout));
        }
    }
    register_module_struct_decoders(&mut vm, &module);
    // Bind the public bytecode name to the existing MIR decoder so this
    // exercises compiler dispatch independently of CLI registration.
    let decoder = vm
        .get_global(&mir_struct_decode_callee("TelemetryEvent"))
        .unwrap();
    vm.register_native("TelemetryEvent.decode", decoder);
    let chunk = &vm.function_by_name("__script").unwrap().chunk;
    assert!(
        chunk
            .instructions
            .iter()
            .any(|inst| matches!(inst, Instruction::LoadGlobal(_, index)
        if chunk.constants[*index as usize].as_string() == Some("TelemetryEvent.decode")))
    );
    vm.call("__script", vec![]).expect("run decoder");
    let result = output.borrow();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].array_get(0), Some(Value::Int(1_700_000_100)));
    assert_eq!(result[0].array_get(1), Some(Value::Int(42)));
    assert_eq!(result[0].array_get(2), Some(Value::Int(950)));
    assert_eq!(result[0].array_get(3), Some(Value::Bool(true)));

    for (name, elapsed, target_ms) in timings {
        assert_release_time(name, elapsed, target_ms);
    }
}

#[test]
fn test_ng_w15_5_map_slow_dispatch_gas_and_memory_budget_match_packed() {
    let (mut vm, _) = compile(
        r#"
        function set(m: Map<int, int>, key: int, value: int)
            map.set(m, key, value)
        end
        function get(m: Map<int, int>, key: int): Option<int>
            return map.get(m, key)
        end
        function read_required(m: Map<int, int>, key: int): int
            return map.get(m, key):unwrap()
        end
    "#,
    );
    let map = Value::map(Default::default());
    let read = |vm: &mut VM, name: &str, key| vm.call(name, vec![map.clone(), Value::Int(key)]);
    let write = |vm: &mut VM, key, value| {
        vm.call("set", vec![map.clone(), Value::Int(key), Value::Int(value)])
    };
    // Grow the register window before measuring the map entry's budget cost.
    assert_eq!(read(&mut vm, "get", 1).unwrap(), Value::none());
    assert_eq!(write(&mut vm, 1, 10).unwrap(), Value::Nil);
    vm.set_memory_budget_bytes(64).unwrap();
    for slow in [false, true] {
        if slow {
            vm.set_instruction_hook(Some(Box::new(|_, _, _| {})));
        }
        let before = vm.memory_used_bytes();
        assert_eq!(write(&mut vm, 1, 20).unwrap(), Value::Nil);
        assert_eq!(vm.memory_used_bytes(), before);
        assert_eq!(
            read(&mut vm, "get", 1).unwrap(),
            Value::some(Value::Int(20))
        );
        vm.reset_gas_counter();
        assert_eq!(read(&mut vm, "read_required", 1).unwrap(), Value::Int(20));
        assert_eq!(vm.gas_used(), 2);
        assert!(
            read(&mut vm, "read_required", 99)
                .unwrap_err()
                .to_string()
                .contains("Called unwrap() on Option::None")
        );
        vm.clear_trace_hooks();
    }
    assert_eq!(write(&mut vm, 2, 30).unwrap(), Value::Nil);
    assert_eq!(vm.memory_used_bytes(), 64);
    assert!(
        write(&mut vm, 3, 40)
            .unwrap_err()
            .to_string()
            .contains("Out of memory budget")
    );
    assert_eq!(map.map_get(&ValueKey::from_value(&Value::Int(3))), None);
}

#[test]
fn test_ng_w15_5_map_fusion_keeps_live_options_and_lexical_shadows() {
    let (mut vm, output) = compile(
        r#"
        local m: Map<int, int> = {}
        map.set(m, 1, 42)
        if true then
            local found: Option<int> = map.get(m, 1)
            local n: int = found:unwrap()
            println(n, found)
        end

        struct Shadow get: function(Map<int, int>, int): Option<int> end
        function shadowed(map: Shadow, m: Map<int, int>): Option<int>
            return map.get(m, 1)
        end
        local shadow: Shadow = Shadow { get = function(m: Map<int, int>, k: int): Option<int> return Some(73) end }
        println(shadowed(shadow, m))
    "#,
    );
    let script = &vm.function_by_name("__script").unwrap().chunk.instructions;
    assert!(
        script
            .iter()
            .any(|inst| matches!(inst, Instruction::MapGet(..)))
    );
    assert!(
        !script
            .iter()
            .any(|inst| matches!(inst, Instruction::MapGetUnwrap(..)))
    );
    assert!(
        !vm.function_by_name("shadowed")
            .unwrap()
            .chunk
            .instructions
            .iter()
            .any(|inst| matches!(inst, Instruction::MapGet(..)))
    );
    vm.call("__script", vec![]).unwrap();
    assert_eq!(
        &*output.borrow(),
        &[
            Value::Int(42),
            Value::some(Value::Int(42)),
            Value::some(Value::Int(73)),
        ]
    );
}

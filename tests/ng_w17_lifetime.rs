//! Wave 17 Track NG-W17-2 Integration & Verification Suite:
//! D-LIFE — Static Borrow / Lifetime Checker, Runtime Weak-VM / Generation Invalidation,
//! Destruction & Cross-Thread Transfer Rules, and ADR-003 N=10,000 Allocation Oracle.
//!
//! Strictly safe Rust (`#![forbid(unsafe_code)]`, `TASK_ALLOWS_UNSAFE=0`).
//! All script tests exercise the REAL lexer, parser, typechecker, static lifetime checker,
//! bytecode compiler, and VM — never manually constructed ASTs or synthetic registries.

#![cfg(feature = "std")]
#![forbid(unsafe_code)]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

// Re-export library modules at crate root so `src/w17_lifetime/` compiles identically
// both standalone here (before 17C registration) and inside `src/lib.rs` (after 17C).
#[allow(unused_imports)]
use lust::{
    alloc_budget, ast, bytecode, data, embed, error, intern, lexer, parser, typechecker, vm,
};

#[path = "../src/w17_lifetime/mod.rs"]
pub mod w17_lifetime;

use std::env;
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use lust::alloc_budget::VmAllocHandle;
use lust::bytecode::Value;
use lust::data::layout::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1, TargetAbi};
use lust::data::materialize::{
    AlignedStorage, ContiguousDecodeOptions, LayoutPerfCounters, decode_json_into_contiguous,
};
use lust::data::owner::{LeaseError, LeaseLifecycleState};
use lust::data::value::DataValue;
use lust::embed::{
    ArrayHandle, EmbeddedProgram, FromLustValue, LustStructView, MapHandle, ValueRef, struct_field,
};
use lust::error::LustError;
use lust::typechecker::lifetime::LifetimeDiagCode;

use w17_lifetime::{
    LifecycleError, LifetimeVm, OwnedTransferPayload, TransferTaskHandle,
    deep_materialize_to_data_value, emit_alloc_interval_begin, emit_alloc_interval_end,
    materialize_array_for_transfer, materialize_map_for_transfer, materialize_struct_for_transfer,
    run_isolated_alloc_child, verify_compile_fail_not_send,
};

const CHILD_ALLOC_MODE_ENV: &str = "LUST_W17_LIFETIME_ALLOC_CHILD";

fn active_target_dir() -> String {
    if let Ok(exe) = env::current_exe() {
        for ancestor in exe.ancestors().skip(1) {
            if ancestor.join("debug").is_dir() || ancestor.join("release").is_dir() {
                return ancestor.to_string_lossy().into_owned();
            }
        }
    }
    env::var("CARGO_TARGET_DIR")
        .unwrap_or_else(|_| "/tmp/lust_eval_target/w17-2".to_string())
}

fn assert_script_lifetime_err(source: &str, code: LifetimeDiagCode) {
    let err = match LifetimeVm::compile_script(source) {
        Ok(_) => panic!(
            "expected static lifetime rejection {} from real compiler pipeline, got Ok",
            code.as_str()
        ),
        Err(e) => e,
    };
    match err {
        LifecycleError::Lust(LustError::TypeErrorWithSpan {
            message,
            line,
            column,
            ..
        }) => {
            assert!(
                line > 0 && column > 0,
                "expected nonzero offending span for {}, got {}:{}",
                code.as_str(),
                line,
                column
            );
            let expected_prefix = format!("{}: {}", code.as_str(), code.primary_message());
            assert!(
                message.starts_with(&expected_prefix),
                "expected exact prefix {:?}, got {:?}",
                expected_prefix,
                message
            );
        }
        other => panic!(
            "expected typed LustError::TypeErrorWithSpan for {}, got {:?}",
            code.as_str(),
            other
        ),
    }
}

fn assert_script_type_err(source: &str, expected_substr: &str) {
    let err = match LifetimeVm::compile_script(source) {
        Ok(_) => panic!("expected static type error containing {:?}, got Ok", expected_substr),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains(expected_substr),
        "expected error containing {:?}, got {:?}",
        expected_substr,
        msg
    );
}

// ---------------------------------------------------------------------------
// 1. Real Script Borrow Producer, Type Propagation & End-to-End Execution
// ---------------------------------------------------------------------------

#[test]
fn w17_dlife_script_real_borrow_producer_reads_and_type_propagation() {
    let src = r#"
struct FixedBox
    value: int
    label: string
end

struct Holder
    inner: FixedBox
end

function read_box_ref(b: &FixedBox): int
    return view.read(b.value)
end

function bump_box_mut(b: &mut FixedBox): int
    b.value = view.read(b.value) + 1
    return view.read(b.value)
end

function run_valid_borrows(): int
    local box_owner: FixedBox = FixedBox { value = 41, label = "orig" }

    -- 1. Shared borrow &owner inside nested scoped regions + helper propagation + reborrow
    local read_res: int = 0
    scoped do
        local v: &FixedBox = &box_owner
        scoped do
            local reborrowed: &FixedBox = &v
            read_res = read_box_ref(reborrowed)
        end
        local via_ns: &FixedBox = view.borrow(box_owner)
        local direct_field: int = view.read(via_ns.value)
        assert(direct_field == 41)
    end
    assert(read_res == 41)

    -- 2. Sequential mutation after shared borrow region closes
    box_owner.value = 42

    -- 3. Mutable borrow &mut owner + mutation through &mut + read
    local after_mut: int = 0
    scoped do
        local m: &mut FixedBox = &mut box_owner
        after_mut = bump_box_mut(m)
    end
    assert(after_mut == 43)

    -- 4. Nested struct field borrow type propagation (&Holder -> &FixedBox -> scalar read)
    local h: Holder = Holder { inner = FixedBox { value = 100, label = "nested" } }
    local nested_sum: int = 0
    scoped do
        local hv: &Holder = &h
        local inner_v: &FixedBox = hv.inner
        local label_v: &string = inner_v.label
        local owned_label: string = view.materialize(label_v)
        assert(owned_label == "nested")
        nested_sum = view.read(inner_v.value)
    end

    -- 5. Immediate synchronous closure reading a live borrow
    local imm_val: int = 0
    scoped do
        local v2: &FixedBox = &box_owner
        imm_val = (function(): int
            return view.read(v2.value)
        end)()
    end

    return read_res + after_mut + nested_sum + imm_val
end
"#;

    let mut vm = LifetimeVm::compile_script(src).expect("valid script borrows must compile");
    let got = vm
        .call("main.run_valid_borrows", vec![])
        .expect("valid script borrows must execute");
    assert_eq!(got, Value::Int(41 + 43 + 100 + 43));
}

#[test]
fn w17_dlife_script_deep_materialization_independence() {
    let src = r#"
struct Payload
    nums: Array<int>
    tag: string
end

function test_deep_copy(): int
    local owner: Payload = Payload { nums = [41, 42], tag = "v1" }
    local owned_copy: Payload = Payload { nums = [], tag = "" }
    scoped do
        local v: &Payload = &owner
        owned_copy = view.materialize(v)
    end
    -- Mutate original owner's nested array and scalar field after borrow ends
    owner.nums[0] = 999
    owner.tag = "mutated"
    -- Materialized copy MUST retain independent nested backing storage!
    local orig_0: int = owner.nums[0]:unwrap()
    local copy_0: int = owned_copy.nums[0]:unwrap()
    local copy_1: int = owned_copy.nums[1]:unwrap()
    assert(orig_0 == 999)
    assert(copy_0 == 41)
    assert(copy_1 == 42)
    assert(owned_copy.tag == "v1")
    return copy_0 + copy_1
end
"#;

    let mut vm = LifetimeVm::compile_script(src).expect("materialization script must compile");
    let got = vm
        .call("main.test_deep_copy", vec![])
        .expect("materialization script must execute");
    assert_eq!(got, Value::Int(83));
}

#[test]
fn w17_dlife_script_type_propagation_rejects_mismatched_ref_and_mut_ref() {
    // &int cannot unify with int
    let ref_into_owned = r#"
function bad(): int
    local owner: int = 42
    local x: int = &owner
    return x
end
"#;
    assert_script_type_err(ref_into_owned, "Type mismatch");

    // &mut cannot be created from immutable &T
    let mut_from_shared = r#"
function bad(): int
    local owner: int = 42
    local s: &int = &owner
    local m: &mut int = &mut s
    return view.read(m)
end
"#;
    assert_script_type_err(mut_from_shared, "mutable borrow");

    // &int cannot unify with &mut int
    let shared_into_mut_ann = r#"
function bad(): int
    local owner: int = 42
    local m: &mut int = &owner
    return view.read(m)
end
"#;
    assert_script_type_err(shared_into_mut_ann, "Type mismatch");
}

// ---------------------------------------------------------------------------
// 2. Real Script Negative Controls (E4G001–E4G005) Without Synthetic Registry
// ---------------------------------------------------------------------------

#[test]
fn w17_dlife_script_rejects_returns_globals_and_aggregate_escapes_e4g001() {
    // Positive control first: compiles & executes cleanly
    let pos = r#"
struct FixedBox
    value: int
end
function ok_read(): int
    local owner: FixedBox = FixedBox { value = 42 }
    local v: &FixedBox = &owner
    return view.read(v.value)
end
"#;
    let mut vm = LifetimeVm::compile_script(pos).expect("positive control must compile");
    assert_eq!(vm.call("main.ok_read", vec![]).unwrap(), Value::Int(42));

    // Return &FixedBox directly
    assert_script_lifetime_err(
        r#"
struct FixedBox
    value: int
end
function bad_return(): &FixedBox
    local owner: FixedBox = FixedBox { value = 42 }
    local v: &FixedBox = &owner
    return v
end
"#,
        LifetimeDiagCode::E4G001,
    );

    // Return borrowed field without view.read/materialize
    assert_script_lifetime_err(
        r#"
struct FixedBox
    label: string
end
function bad_return_field(): &string
    local owner: FixedBox = FixedBox { label = "secret" }
    local v: &FixedBox = &owner
    return v.label
end
"#,
        LifetimeDiagCode::E4G001,
    );

    // Store borrow into module-level global
    assert_script_lifetime_err(
        r#"
local GLOBAL_SLOT: unknown = 0
function bad_global(): int
    local owner: int = 42
    local v: &int = &owner
    GLOBAL_SLOT = v
    return 0
end
"#,
        LifetimeDiagCode::E4G001,
    );

    // Return borrow wrapped in aggregate array
    assert_script_lifetime_err(
        r#"
function bad_agg(): Array<&int>
    local owner: int = 42
    local v: &int = &owner
    local arr: Array<&int> = [v]
    return arr
end
"#,
        LifetimeDiagCode::E4G001,
    );

    // Escape inner-scoped owner borrow to outer-scoped variable
    assert_script_lifetime_err(
        r#"
function bad_outer_assign(): int
    local outer_dummy: int = 1
    local outer_ref: &int = &outer_dummy
    scoped do
        local inner_owner: int = 42
        outer_ref = &inner_owner
    end
    return view.read(outer_ref)
end
"#,
        LifetimeDiagCode::E4G001,
    );
}

#[test]
fn w17_dlife_script_rejects_closure_captures_e4g002() {
    // Positive immediate closure control
    let pos = r#"
function ok_immediate(): int
    local owner: int = 42
    local v: &int = &owner
    return (function(): int
        return view.read(v)
    end)()
end
"#;
    let mut vm = LifetimeVm::compile_script(pos).expect("immediate closure must compile");
    assert_eq!(vm.call("main.ok_immediate", vec![]).unwrap(), Value::Int(42));

    // Escaping closure capturing borrow directly
    assert_script_lifetime_err(
        r#"
function bad_closure(): int
    local owner: int = 42
    local v: &int = &owner
    local cl = function(): int
        return view.read(v)
    end
    return 0
end
"#,
        LifetimeDiagCode::E4G002,
    );

    // Escaping closure capturing aliased borrow in aggregate
    assert_script_lifetime_err(
        r#"
function bad_closure_agg(): int
    local owner: int = 42
    local v: &int = &owner
    local box_arr: Array<&int> = [v]
    local cl = function(): int
        return 0 + (box_arr[0]:is_ok() and 1 or 0)
    end
    return 0
end
"#,
        LifetimeDiagCode::E4G002,
    );
}

#[test]
fn w17_dlife_script_rejects_fiber_actor_async_and_thread_transfer_e4g003() {
    let helper_stubs = r#"
function spawn(x: unknown): int return 0 end
function yield(x: unknown): int return 0 end
"#;
    // Positive control: borrow ends in scoped block before suspension
    let pos = format!(
        "{helper_stubs}
function ok_before_suspend(): int
    local owner: int = 42
    local copied: int = 0
    scoped do
        local v: &int = &owner
        copied = view.read(v)
    end
    local _s: int = yield(copied)
    return copied
end
"
    );
    let mut vm = LifetimeVm::compile_script(&pos).expect("ended borrow before suspend must pass");
    assert_eq!(
        vm.call("main.ok_before_suspend", vec![]).unwrap(),
        Value::Int(42)
    );

    // Live borrow passed to spawn (fiber/task transfer)
    assert_script_lifetime_err(
        &format!(
            "{helper_stubs}
function bad_spawn(): int
    local owner: int = 42
    local v: &int = &owner
    local _t: int = spawn(v)
    return 0
end
"
        ),
        LifetimeDiagCode::E4G003,
    );

    // Live borrow held across suspension point
    assert_script_lifetime_err(
        &format!(
            "{helper_stubs}
function bad_across_yield(): int
    local owner: int = 42
    local v: &int = &owner
    local _y: int = yield(0)
    return view.read(v)
end
"
        ),
        LifetimeDiagCode::E4G003,
    );

    // Live borrow passed to thread.spawn / actor.send
    assert_script_lifetime_err(
        r#"
local thread: unknown = 0
function bad_thread_spawn(): int
    local owner: int = 42
    local v: &int = &owner
    local _t: unknown = thread.spawn(v)
    return 0
end
"#,
        LifetimeDiagCode::E4G003,
    );

    assert_script_lifetime_err(
        r#"
local actor: unknown = 0
function bad_actor_send(): int
    local owner: int = 42
    local v: &int = &owner
    local _t: unknown = actor.send(v)
    return 0
end
"#,
        LifetimeDiagCode::E4G003,
    );
}

#[test]
fn w17_dlife_script_rejects_backing_mutation_drop_conflicts_and_defer_e4g004() {
    // Positive controls: distinct owners + shadowed binding + sequential write
    let pos = r#"
struct Boxed
    value: int
end
function ok_controls(): int
    local a: Boxed = Boxed { value = 10 }
    local b: Boxed = Boxed { value = 20 }
    local r: int = 0
    scoped do
        local va: &Boxed = &a
        b.value = 30
        local a: Boxed = Boxed { value = 99 }
        a.value = 100
        r = view.read(va.value) + a.value
    end
    a.value = 42
    return r + b.value + a.value
end
"#;
    let mut vm = LifetimeVm::compile_script(pos).expect("positive mutation controls must compile");
    assert_eq!(
        vm.call("main.ok_controls", vec![]).unwrap(),
        Value::Int(10 + 100 + 30 + 42)
    );

    // Direct owner field mutation while shared borrow is live
    assert_script_lifetime_err(
        r#"
struct Boxed
    value: int
end
function bad_field_mutate(): int
    local owner: Boxed = Boxed { value = 41 }
    local v: &Boxed = &owner
    owner.value = 42
    return view.read(v.value)
end
"#,
        LifetimeDiagCode::E4G004,
    );

    // Mutate through shared borrow (&Boxed)
    assert_script_lifetime_err(
        r#"
struct Boxed
    value: int
end
function bad_write_through_shared(): int
    local owner: Boxed = Boxed { value = 41 }
    local v: &Boxed = &owner
    v.value = 42
    return view.read(v.value)
end
"#,
        LifetimeDiagCode::E4G004,
    );

    // Overlapping &mut and & borrows
    assert_script_lifetime_err(
        r#"
function bad_mut_overlap(): int
    local owner: int = 41
    local s: &int = &owner
    local m: &mut int = &mut owner
    return view.read(s) + view.read(m)
end
"#,
        LifetimeDiagCode::E4G004,
    );

    // Explicit drop while borrowed
    assert_script_lifetime_err(
        r#"
function drop(x: unknown): int return 0 end
function bad_drop(): int
    local owner: int = 41
    local v: &int = &owner
    drop(owner)
    return view.read(v)
end
"#,
        LifetimeDiagCode::E4G004,
    );

    // Loop back-edge mutation while borrow carried across iterations
    assert_script_lifetime_err(
        r#"
function bad_loop(n: int): int
    local owner: int = 41
    local dummy: int = 0
    local v: &int = &dummy
    local i: int = 0
    while i < n do
        owner = 42
        v = &owner
        i = i + 1
    end
    return view.read(v)
end
"#,
        LifetimeDiagCode::E4G004,
    );

    // Deferred read keeps borrow alive across owner mutation
    assert_script_lifetime_err(
        r#"
function bad_defer(): int
    local owner: int = 41
    local v: &int = &owner
    defer view.read(v)
    owner = 42
    return 0
end
"#,
        LifetimeDiagCode::E4G004,
    );
}

#[test]
fn w17_dlife_script_rejects_unaudited_indirect_and_recursive_calls_e4g005() {
    // Positive: finite safe indirect target set
    let pos = r#"
function reader_a(x: &int): int
    return view.read(x)
end
function reader_b(x: &int): int
    return view.read(x) + 1
end
function ok_indirect(flag: bool): int
    local owner: int = 41
    local v: &int = &owner
    local f = reader_a
    if flag then
        f = reader_b
    end
    return f(v)
end
"#;
    let mut vm = LifetimeVm::compile_script(pos).expect("safe indirect targets must compile");
    assert_eq!(
        vm.call("main.ok_indirect", vec![Value::Bool(true)]).unwrap(),
        Value::Int(42)
    );

    // Negative: indirect target set includes unaudited sink
    assert_script_lifetime_err(
        r#"
function reader_a(x: unknown): int
    return view.read(x)
end
function unsafe_sink(x: unknown): int
    local _n: unknown = math.abs(x)
    return 0
end
function bad_indirect(flag: bool): int
    local owner: int = 41
    local v: &int = &owner
    local f = reader_a
    if flag then
        f = unsafe_sink
    end
    return f(v)
end
"#,
        LifetimeDiagCode::E4G005,
    );

    // Negative: unknown cast cannot erase borrow provenance
    assert_script_lifetime_err(
        r#"
function opaque_sink(x: unknown): int
    local _z: unknown = math.abs(x)
    return 0
end
function bad_erasure(): int
    local owner: int = 42
    local v: &int = &owner
    local erased: unknown = v
    return opaque_sink(erased)
end
"#,
        LifetimeDiagCode::E4G005,
    );

    // Negative: recursive function cannot accept borrowed argument pessimistically
    assert_script_lifetime_err(
        r#"
function rec_reader(x: &int, d: int): int
    if d > 0 then
        return rec_reader(x, d - 1)
    end
    return 0
end
function bad_rec(): int
    local owner: int = 42
    local v: &int = &owner
    return rec_reader(v, 2)
end
"#,
        LifetimeDiagCode::E4G005,
    );
}

// ---------------------------------------------------------------------------
// 3. D-LIFE Oracle #1: Weak VM / Generation Invalidation (Live -> Drop -> Recreate)
// ---------------------------------------------------------------------------

#[derive(lust::LustStructView)]
#[lust(type = "main.FixedBox")]
struct FixedBoxView<'a> {
    #[lust(field = "value")]
    value: ValueRef<'a>,
}

#[test]
fn w17_dlife_oracle_weak_vm_generation_invalidation_drop_and_recreate() {
    let source = r#"
struct FixedBox
    value: int
end

function make_arr(): Array<int>
    return [42]
end

function make_map(): Map<string, int>
    return { k = 42 }
end

function add_one(x: int): int
    return x + 1
end
"#;

    let bytecode = EmbeddedProgram::builder()
        .module("main", source)
        .entry_module("main")
        .precompile()
        .expect("precompile lifecycle fixture");

    // 1. Create VM #1, retain generation-tagged handles, verify valid live reads
    let mut vm1 = LifetimeVm::from_compiled(&bytecode).expect("instantiate vm1");
    let gen1 = vm1.generation();

    let struct_h = vm1
        .create_struct("main.FixedBox", [struct_field("value", 42_i64)])
        .expect("create struct on vm1");
    assert_eq!(struct_h.vm_generation(), gen1);
    assert_eq!(struct_h.read_int_field("value").expect("live read"), 42);
    {
        let view = struct_h.read_view::<FixedBoxView<'_>>().expect("live view");
        assert_eq!(view.value.as_int(), Some(42));
    }

    let arr_val = vm1.call("main.make_arr", vec![]).expect("make_arr");
    let arr_h = vm1.wrap_array(ArrayHandle::from_value(arr_val).expect("ArrayHandle"));
    assert_eq!(
        arr_h.borrow_get(0).expect("live arr get").unwrap().as_int(),
        Some(42)
    );

    let map_val = vm1.call("main.make_map", vec![]).expect("make_map");
    let map_h = vm1.wrap_map(MapHandle::from_value(map_val).expect("MapHandle"));
    assert_eq!(
        map_h.borrow_get("k").expect("live map get").unwrap().as_int(),
        Some(42)
    );

    let fn_h = vm1.function_handle("main.add_one").expect("fn handle");
    let call_val: i64 = fn_h
        .call_typed_on(&mut vm1, (41_i64,))
        .expect("live fn call");
    assert_eq!(call_val, 42);

    // 2. Drop VM #1 while handles are still retained!
    drop(vm1);

    // 3. Assert ALL post-drop accesses deterministically return Err(VmDropped)
    match struct_h.read_int_field("value") {
        Err(LifecycleError::VmDropped { handle_generation }) => {
            assert_eq!(handle_generation, gen1);
        }
        other => panic!("expected Err(VmDropped) on stale struct read, got {:?}", other),
    }
    assert!(matches!(
        struct_h.read_view::<FixedBoxView<'_>>(),
        Err(LifecycleError::VmDropped { handle_generation }) if handle_generation == gen1
    ));
    assert!(matches!(
        struct_h.borrow_field("value"),
        Err(LifecycleError::VmDropped { handle_generation }) if handle_generation == gen1
    ));
    assert!(matches!(
        struct_h.try_set_field("value", 99_i64),
        Err(LifecycleError::VmDropped { handle_generation }) if handle_generation == gen1
    ));
    assert!(matches!(
        arr_h.borrow_get(0),
        Err(LifecycleError::VmDropped { handle_generation }) if handle_generation == gen1
    ));
    assert!(matches!(
        arr_h.try_push(Value::Int(7)),
        Err(LifecycleError::VmDropped { handle_generation }) if handle_generation == gen1
    ));
    assert!(matches!(
        map_h.borrow_get("k"),
        Err(LifecycleError::VmDropped { handle_generation }) if handle_generation == gen1
    ));
    assert!(matches!(
        fn_h.validate_live(),
        Err(LifecycleError::VmDropped { handle_generation }) if handle_generation == gen1
    ));
    assert!(matches!(
        materialize_struct_for_transfer(&struct_h),
        Err(LifecycleError::VmDropped { handle_generation }) if handle_generation == gen1
    ));

    // 4. Recreate VM #2 from the exact same bytecode -> strictly new generation!
    let mut vm2 = LifetimeVm::from_compiled(&bytecode).expect("recreate vm2");
    let gen2 = vm2.generation();
    assert!(
        gen2 > gen1,
        "recreated VM generation must strictly increase (gen1={gen1}, gen2={gen2})"
    );

    // Old generation handle MUST remain invalid both directly and against recreated vm2!
    assert!(matches!(
        struct_h.read_int_field("value"),
        Err(LifecycleError::VmDropped { handle_generation }) if handle_generation == gen1
    ));
    assert!(matches!(
        vm2.validate_struct_handle(&struct_h),
        Err(LifecycleError::GenerationMismatch {
            expected_vm_generation,
            handle_generation,
        }) if expected_vm_generation == gen2 && handle_generation == gen1
    ));
    assert!(matches!(
        fn_h.call_typed_on::<_, i64>(&mut vm2, (41_i64,)),
        Err(LifecycleError::GenerationMismatch {
            expected_vm_generation,
            handle_generation,
        }) if expected_vm_generation == gen2 && handle_generation == gen1
    ));

    // Fresh handle created on vm2 works cleanly!
    let fresh_h = vm2
        .create_struct("main.FixedBox", [struct_field("value", 100_i64)])
        .expect("create fresh struct on vm2");
    assert_eq!(fresh_h.vm_generation(), gen2);
    assert_eq!(fresh_h.read_int_field("value").expect("vm2 read"), 100);
    vm2.validate_struct_handle(&fresh_h)
        .expect("vm2 validates its own fresh handle");
}

// ---------------------------------------------------------------------------
// 4. Owner Destruction, Slot-Reuse Generation Bump, Tenant & Borrow Conflict
// ---------------------------------------------------------------------------

#[test]
fn w17_dlife_owner_destruction_borrow_conflict_slot_reuse_and_tenant_isolation() {
    let source = r#"
struct FixedBox
    value: int
end
"#;
    let vm = LifetimeVm::compile_script(source).expect("compile");

    let h1 = vm
        .create_struct_for_tenant("main.FixedBox", [struct_field("value", 41_i64)], 7)
        .expect("create h1");
    let h1_clone = h1.clone();
    let owner1 = h1.owner_id();

    // Active ScopedViewGuard rejects conflicting mutation safely with Err(BorrowConflict)
    {
        let guard = h1.borrow_field("value").expect("borrow_field");
        assert_eq!(guard.as_int(), Some(41));
        assert!(matches!(
            h1_clone.try_set_field("value", 42_i64),
            Err(LifecycleError::BorrowConflict { .. })
        ));
    }
    // After guard drop, mutation succeeds cleanly
    h1_clone.try_set_field("value", 42_i64).expect("mutate after guard drop");
    assert_eq!(h1.read_int_field("value").expect("read updated"), 42);

    // Tenant isolation check
    assert!(matches!(
        h1.validate_for_tenant(999),
        Err(LifecycleError::OwnerInvalidated(LeaseError::TenantMismatch {
            expected: 999,
            actual: 7,
        }))
    ));

    // Protocol enforcement: release before cancel/join MUST fail with ReleaseBeforeJoin
    assert!(matches!(
        vm.release_owner(owner1),
        Err(LifecycleError::OwnerInvalidated(LeaseError::ReleaseBeforeJoin))
    ));

    // Step 1: cancel -> immediately blocks further handle reads
    vm.cancel_owner(owner1).expect("cancel owner1");
    assert_eq!(vm.owner_slot_state(owner1), Some(LeaseLifecycleState::Cancelled));
    assert!(matches!(
        h1.read_int_field("value"),
        Err(LifecycleError::OwnerInvalidated(LeaseError::InvalidState(_)))
    ));
    assert!(matches!(
        h1_clone.read_int_field("value"),
        Err(LifecycleError::OwnerInvalidated(LeaseError::InvalidState(_)))
    ));

    // Release still fails before join acknowledgement
    assert!(matches!(
        vm.release_owner(owner1),
        Err(LifecycleError::OwnerInvalidated(LeaseError::ReleaseBeforeJoin))
    ));

    // Step 2 & 3: acknowledge_join -> release
    vm.acknowledge_owner_join(owner1).expect("acknowledge join");
    vm.release_owner(owner1).expect("release owner1");
    assert_eq!(vm.owner_slot_state(owner1), Some(LeaseLifecycleState::Released));

    // Reallocating a new struct reuses slot 0 with bumped generation (gen + 1)
    let h2 = vm
        .create_struct_for_tenant("main.FixedBox", [struct_field("value", 77_i64)], 7)
        .expect("create h2");
    let owner2 = h2.owner_id();
    assert_eq!(owner2.slot_id, owner1.slot_id, "slot should be reused");
    assert_eq!(
        owner2.generation,
        owner1.generation + 1,
        "slot generation must monotonically advance on reuse"
    );

    // Old handle targeting reused slot fails with StaleGeneration!
    assert!(matches!(
        h1.read_int_field("value"),
        Err(LifecycleError::OwnerInvalidated(LeaseError::StaleGeneration {
            expected,
            actual,
        })) if expected == owner2.generation && actual == owner1.generation
    ));
    // New handle succeeds!
    assert_eq!(h2.read_int_field("value").expect("h2 read"), 77);
}

// ---------------------------------------------------------------------------
// 5. Cross-Thread Transfer: Rejected `!Send` Compile-Fail + Positive `Send` Lifecycle
// ---------------------------------------------------------------------------

#[test]
fn w17_dlife_cross_thread_not_send_compile_fail_probes() {
    let target_dir = active_target_dir();

    // Probe 1: Attempting to send ScopedViewGuard across std::thread::spawn MUST fail to compile
    let p1 = verify_compile_fail_not_send(
        "not_send_scoped_view_guard",
        r#"
    let vm = w17_lifetime::LifetimeVm::compile_script("struct FixedBox\n value: int\nend\n").unwrap();
    let h = vm.create_struct("main.FixedBox", [lust::embed::struct_field("value", 42_i64)]).unwrap();
    let guard = h.borrow_field("value").unwrap();
    let _ = std::thread::spawn(move || {
        let _ = guard.as_int();
    });
"#,
        &target_dir,
    )
    .expect("run rustc probe 1");
    assert!(
        p1.rejected_as_not_send,
        "ScopedViewGuard must fail compile with !Send error; stderr:\n{}",
        p1.stderr
    );

    // Probe 2: Attempting to send LustStructView (ValueRef) across std::thread::spawn MUST fail
    let p2 = verify_compile_fail_not_send(
        "not_send_value_ref",
        r#"
    fn force_send<T: Send>(_: T) {}
    let vm = w17_lifetime::LifetimeVm::compile_script("struct FixedBox\n value: int\nend\n").unwrap();
    let h = vm.create_struct("main.FixedBox", [lust::embed::struct_field("value", 42_i64)]).unwrap();
    let raw = h.checked_raw_handle().unwrap();
    let vr: lust::embed::ValueRef<'_> = raw.borrow_field("value").unwrap();
    force_send(vr);
"#,
        &target_dir,
    )
    .expect("run rustc probe 2");
    assert!(
        p2.rejected_as_not_send,
        "ValueRef must fail compile with !Send error; stderr:\n{}",
        p2.stderr
    );

    // Probe 3: Attempting to send GenStructHandle across std::thread::spawn MUST fail
    let p3 = verify_compile_fail_not_send(
        "not_send_gen_struct_handle",
        r#"
    let vm = w17_lifetime::LifetimeVm::compile_script("struct FixedBox\n value: int\nend\n").unwrap();
    let h = vm.create_struct("main.FixedBox", [lust::embed::struct_field("value", 42_i64)]).unwrap();
    let _ = std::thread::spawn(move || {
        let _ = h.read_int_field("value");
    });
"#,
        &target_dir,
    )
    .expect("run rustc probe 3");
    assert!(
        p3.rejected_as_not_send,
        "GenStructHandle must fail compile with !Send error; stderr:\n{}",
        p3.stderr
    );

    // Probe 4: Attempting to send LifetimeVm across std::thread::spawn MUST fail
    let p4 = verify_compile_fail_not_send(
        "not_send_lifetime_vm",
        r#"
    let vm = w17_lifetime::LifetimeVm::compile_script("function f(): int return 1 end").unwrap();
    let _ = std::thread::spawn(move || {
        let _ = vm.generation();
    });
"#,
        &target_dir,
    )
    .expect("run rustc probe 4");
    assert!(
        p4.rejected_as_not_send,
        "LifetimeVm must fail compile with !Send error; stderr:\n{}",
        p4.stderr
    );
}

#[test]
fn w17_dlife_cross_thread_owned_transfer_and_cancellation_lifecycle() {
    let source = r#"
struct TransferRecord
    id: int
    label: string
end

function make_arr(): Array<int>
    return [10, 20, 30]
end

function make_map(): Map<string, int>
    return { score = 42 }
end

function dummy_fn(): int
    return 1
end
"#;
    let mut vm = LifetimeVm::compile_script(source).expect("compile transfer fixture");
    let vm_gen = vm.generation();

    let rec = vm
        .create_struct(
            "main.TransferRecord",
            [
                struct_field("id", 42_i64),
                struct_field("label", "independent"),
            ],
        )
        .expect("create record");

    let arr_val = vm.call("main.make_arr", vec![]).expect("make_arr");
    let arr_h = vm.wrap_array(ArrayHandle::from_value(arr_val).expect("ArrayHandle"));
    let map_val = vm.call("main.make_map", vec![]).expect("make_map");
    let map_h = vm.wrap_map(MapHandle::from_value(map_val).expect("MapHandle"));

    // Negative: non-Send runtime values (functions) fail closed on materialization
    let fn_val = vm.program().function_handle("main.dummy_fn").unwrap().into_value();
    assert!(matches!(
        deep_materialize_to_data_value(&fn_val),
        Err(LifecycleError::UnsupportedTransfer(_))
    ));

    // Positive: deep-materialize into Send+Sync OwnedTransferPayload
    let payload: OwnedTransferPayload =
        materialize_struct_for_transfer(&rec).expect("materialize struct");
    let arr_payload = materialize_array_for_transfer(&arr_h).expect("materialize array");
    let map_payload = materialize_map_for_transfer(&map_h).expect("materialize map");
    assert_eq!(payload.source_vm_generation(), vm_gen);

    let worker_started = Arc::new(AtomicBool::new(false));
    let allow_finish = Arc::new(AtomicBool::new(false));
    let ws = Arc::clone(&worker_started);
    let af = Arc::clone(&allow_finish);

    let task = TransferTaskHandle::spawn(payload, move |owned, _cancel| {
        ws.store(true, Ordering::Release);
        while !af.load(Ordering::Acquire) {
            thread::sleep(Duration::from_millis(2));
        }
        let doc = match owned.value() {
            DataValue::Document(d) => d,
            other => panic!("expected Document payload, got {:?}", other),
        };
        let id = doc.get("id").and_then(|v| v.as_i64()).unwrap();
        let label = doc.get("label").and_then(|v| v.as_str()).unwrap().to_string();
        (id, label)
    });

    while !worker_started.load(Ordering::Acquire) {
        thread::sleep(Duration::from_millis(1));
    }

    // Mutate source AND drop the entire VM while the worker thread is alive!
    rec.try_set_field("id", 999_i64).expect("mutate source id");
    rec.try_set_field("label", "mutated").expect("mutate source label");
    arr_h.try_push(Value::Int(999)).expect("mutate source arr");
    map_h.try_insert("score", Value::Int(999)).expect("mutate source map");
    drop(vm);

    // Release worker and verify it observes exact original independent values!
    allow_finish.store(true, Ordering::Release);
    let (obs_id, obs_label) = task.join().expect("worker join");
    assert_eq!(obs_id, 42);
    assert_eq!(obs_label, "independent");

    // Verify array and map payloads also survived source mutation + VM drop across threads
    let (arr_sum, map_score) = thread::spawn(move || {
        let sum: i64 = arr_payload
            .value()
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .sum();
        let score = match map_payload.value() {
            DataValue::Document(d) => d.get("score").and_then(|v| v.as_i64()).unwrap(),
            _ => panic!("expected document"),
        };
        (sum, score)
    })
    .join()
    .expect("coll worker join");
    assert_eq!(arr_sum, 60);
    assert_eq!(map_score, 42);

    // Cancellable pending task + cancel -> join -> release protocol on a fresh VM
    let vm2 = LifetimeVm::compile_script(source).expect("compile vm2");
    let rec2 = vm2
        .create_struct(
            "main.TransferRecord",
            [
                struct_field("id", 7_i64),
                struct_field("label", "cancellable"),
            ],
        )
        .expect("create rec2");
    let owner2 = rec2.owner_id();
    let payload2 = materialize_struct_for_transfer(&rec2).expect("materialize rec2");

    let pending_task = TransferTaskHandle::spawn(payload2, |owned, cancel| {
        let mut spins = 0_u64;
        while !cancel.load(Ordering::Acquire) {
            spins += 1;
            thread::sleep(Duration::from_millis(1));
            if spins > 5_000 {
                panic!("cancellation did not arrive within deadline");
            }
        }
        owned.source_owner_id()
    });

    // Step 1: cancel owner on VM and signal worker cancellation
    vm2.cancel_owner(owner2).expect("cancel owner2");
    assert!(matches!(
        rec2.read_int_field("id"),
        Err(LifecycleError::OwnerInvalidated(LeaseError::InvalidState(_)))
    ));
    // Step 2: join worker thread and acknowledge join
    let returned_owner = pending_task.cancel_and_join().expect("cancel and join");
    assert_eq!(returned_owner, owner2);
    vm2.acknowledge_owner_join(owner2)
        .expect("acknowledge join after thread exit");
    // Step 3: release owner slot; confirm post-release access fails
    vm2.release_owner(owner2).expect("release owner2");
    assert!(matches!(
        rec2.read_int_field("id"),
        Err(LifecycleError::OwnerInvalidated(LeaseError::InvalidState(_)))
    ));
}

// ---------------------------------------------------------------------------
// 6. D-LIFE Oracle #2: Isolated Allocation Counter Over N=10,000 Fixed-Scalar Reads
// ---------------------------------------------------------------------------

#[test]
fn w17_dlife_oracle_10000_fixed_scalar_reads_zero_alloc_adr003() {
    if env::var(CHILD_ALLOC_MODE_ENV).as_deref() == Ok("1") {
        run_10k_fixed_scalar_reads_child();
        return;
    }

    let target_dir = active_target_dir();
    let (reports, child_stdout, child_stderr) = run_isolated_alloc_child(
        &target_dir,
        "w17_dlife_oracle_10000_fixed_scalar_reads_zero_alloc_adr003",
        CHILD_ALLOC_MODE_ENV,
    )
    .expect("run isolated DYLD_INTERPOSE alloc oracle child");

    assert_eq!(
        reports.len(),
        5,
        "expected exactly 5 intervals (null, canary, struct_view_10k, gen_handle_10k, contiguous_10k), got {:?}\nstdout:\n{}\nstderr:\n{}",
        reports,
        child_stdout,
        child_stderr
    );

    let null_ctrl = reports[0];
    let canary_ctrl = reports[1];
    let struct_view_10k = reports[2];
    let gen_handle_10k = reports[3];
    let contiguous_10k = reports[4];

    eprintln!(
        "W17_DLIFE_ADR003_ALLOC_ORACLE methodology=DYLD_INTERPOSE(malloc,calloc,realloc,posix_memalign)+VmAllocHandle reads=10000 \
         null_ctrl_total={} canary_ctrl_total={} \
         struct_view_10k(malloc={},calloc={},realloc={},memalign={},total={}) \
         gen_handle_10k(malloc={},calloc={},realloc={},memalign={},total={}) \
         contiguous_10k(malloc={},calloc={},realloc={},memalign={},total={})",
        null_ctrl.total(),
        canary_ctrl.total(),
        struct_view_10k.malloc_calls,
        struct_view_10k.calloc_calls,
        struct_view_10k.realloc_calls,
        struct_view_10k.posix_memalign_calls,
        struct_view_10k.total(),
        gen_handle_10k.malloc_calls,
        gen_handle_10k.calloc_calls,
        gen_handle_10k.realloc_calls,
        gen_handle_10k.posix_memalign_calls,
        gen_handle_10k.total(),
        contiguous_10k.malloc_calls,
        contiguous_10k.calloc_calls,
        contiguous_10k.realloc_calls,
        contiguous_10k.posix_memalign_calls,
        contiguous_10k.total(),
    );

    assert_eq!(
        null_ctrl.total(),
        0,
        "null-interval control must record 0 allocations, got {:?}",
        null_ctrl
    );
    assert!(
        canary_ctrl.total() >= 1,
        "canary allocation calibration failed (expected >= 1, got {:?})",
        canary_ctrl
    );
    assert_eq!(
        struct_view_10k.total(),
        0,
        "10,000 prepared LustStructView fixed-scalar reads must allocate 0 (< 1), got {:?}",
        struct_view_10k
    );
    assert_eq!(
        gen_handle_10k.total(),
        0,
        "10,000 prepared generation-tagged weak-VM fixed-scalar reads must allocate 0 (< 1), got {:?}",
        gen_handle_10k
    );
    assert_eq!(
        contiguous_10k.total(),
        0,
        "10,000 prepared ContiguousStruct fixed-scalar reads must allocate 0 (< 1), got {:?}",
        contiguous_10k
    );
    assert!(
        child_stderr.contains("CHILD_ORACLE_SUMS struct_view=420000 gen_handle=420000 contiguous=420000 vm_alloc_delta=0"),
        "child sum / VmAllocHandle verification marker missing:\n{}",
        child_stderr
    );
}

fn run_10k_fixed_scalar_reads_child() {
    const N: usize = 10_000;
    const EXPECTED_SUM: i64 = 42 * (N as i64);

    // Setup (explicitly excluded from measurement interval)
    let source = r#"
struct FixedBox
    value: int
end
"#;
    let vm = LifetimeVm::compile_script(source).expect("compile FixedBox");
    let gen_handle = vm
        .create_struct("main.FixedBox", [struct_field("value", 42_i64)])
        .expect("create FixedBox");
    let prepared = gen_handle
        .clone()
        .prepare_scalar_field("value")
        .expect("prepare scalar field");
    let raw_handle = gen_handle
        .checked_raw_handle()
        .expect("raw handle")
        .clone();

    // ContiguousStruct setup with VmAllocHandle budget tracker
    let alloc_handle = VmAllocHandle::with_limit(4096);
    let layout = LogicalLayoutV1::new_with_abi(
        9200,
        "FixedBox",
        vec![LogicalFieldDesc::new(0, "value", CanonicalTypeTag::Int64)],
        TargetAbi::host(),
    );
    let _budget_storage = AlignedStorage::new_zeroed_in(
        layout.size_bytes,
        layout.align_bytes,
        alloc_handle.clone(),
    )
    .expect("aligned storage");
    let mut counters = LayoutPerfCounters::default();
    let contiguous = decode_json_into_contiguous(
        &layout,
        br#"{"value": 42}"#,
        &mut counters,
        &ContiguousDecodeOptions::default(),
    )
    .expect("decode contiguous");
    let field_offset = layout.fields[0].byte_offset;

    // Warmup before intervals
    {
        let w1 = FixedBoxView::from_handle(&raw_handle).expect("warm view");
        assert_eq!(w1.value.as_int(), Some(42));
        assert_eq!(prepared.read_i64().expect("warm gen read"), 42);
        assert_eq!(
            contiguous
                .read_i64_at(0, field_offset, Some(&mut counters))
                .expect("warm contiguous read"),
            Some(42)
        );
    }

    // Interval 0: Null control (verifies measurement hooks allocate 0 bytes)
    emit_alloc_interval_begin();
    emit_alloc_interval_end();

    // Interval 1: Positive canary control (verifies counter detects real Rust heap allocations)
    emit_alloc_interval_begin();
    let canary: Vec<u8> = black_box(Vec::with_capacity(64));
    emit_alloc_interval_end();
    drop(canary);

    // Interval 2: N=10,000 prepared LustStructView fixed-scalar reads
    let mut sum_view: i64 = 0;
    emit_alloc_interval_begin();
    for _ in 0..N {
        let view = FixedBoxView::from_handle(black_box(&raw_handle)).expect("view");
        sum_view += black_box(view.value.as_int().expect("i64"));
    }
    emit_alloc_interval_end();
    assert_eq!(sum_view, EXPECTED_SUM);

    // Interval 3: N=10,000 prepared generation-tagged weak-VM fixed-scalar reads
    let mut sum_gen: i64 = 0;
    emit_alloc_interval_begin();
    for _ in 0..N {
        sum_gen += black_box(prepared.read_i64().expect("gen scalar read"));
    }
    emit_alloc_interval_end();
    assert_eq!(sum_gen, EXPECTED_SUM);

    // Interval 4: N=10,000 prepared ContiguousStruct fixed-scalar reads (+ VmAllocHandle delta)
    let used_before = alloc_handle.0.used_bytes();
    let peak_before = alloc_handle.0.peak_bytes();
    let mut sum_contig: i64 = 0;
    emit_alloc_interval_begin();
    for _ in 0..N {
        sum_contig += black_box(
            contiguous
                .read_i64_at(0, field_offset, None)
                .expect("contig scalar read")
                .expect("present i64"),
        );
    }
    emit_alloc_interval_end();
    let vm_alloc_delta = (alloc_handle.0.used_bytes() - used_before)
        + (alloc_handle.0.peak_bytes() - peak_before);
    assert_eq!(sum_contig, EXPECTED_SUM);
    assert_eq!(vm_alloc_delta, 0);

    eprintln!(
        "CHILD_ORACLE_SUMS struct_view={} gen_handle={} contiguous={} vm_alloc_delta={}",
        sum_view, sum_gen, sum_contig, vm_alloc_delta
    );
}

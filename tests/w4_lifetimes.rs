#![cfg(feature = "std")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use lust::{
    EmbeddedProgram, Value,
    ast::Item,
    embed::{
        ArrayHandle, FromLustValue, FunctionHandle, LustStructView, MapHandle, StructHandle,
        StructInstance, ValueRef, struct_field,
    },
    intern::Interner,
    lexer::Lexer,
    parser::Parser,
    typechecker::TypeChecker,
    typechecker::lifetime::{
        LifetimeDiagCode, LifetimeRegistry, check_module_lifetimes,
        check_module_lifetimes_with_registry,
    },
};

// ---------------------------------------------------------------------------
// Thread-isolated counting allocator for Matrix Row 6 (fixed-scalar view)
// ---------------------------------------------------------------------------

struct CountingAlloc;

static TRACKING_ACTIVE: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_ZEROED_COUNT: AtomicUsize = AtomicUsize::new(0);
static REALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_TEST_MUTEX: Mutex<()> = Mutex::new(());

thread_local! {
    static IS_MEASURED_THREAD: Cell<bool> = const { Cell::new(false) };
}

#[inline]
fn on_measured_thread() -> bool {
    TRACKING_ACTIVE.load(Ordering::Acquire)
        && IS_MEASURED_THREAD.try_with(|c| c.get()).unwrap_or(false)
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if on_measured_thread() {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if on_measured_thread() {
            ALLOC_ZEROED_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if on_measured_thread() {
            REALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOC: CountingAlloc = CountingAlloc;

// ---------------------------------------------------------------------------
// Real parser / typechecker / lifetime helpers
// ---------------------------------------------------------------------------

fn parse_items(source: &str) -> Vec<Item> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("lexer should succeed");
    Parser::new(tokens).parse().expect("parser should succeed")
}

fn typecheck_only(source: &str) -> lust::Result<()> {
    let items = parse_items(source);
    let mut tc = TypeChecker::new();
    tc.check_module(&items)
}

fn synthetic_registry() -> LifetimeRegistry {
    let mut reg = LifetimeRegistry::default();
    // Producer functions taking owner as arg 0
    reg.register_borrow_producer("borrow_view", 0, false);
    reg.register_borrow_producer("borrow_mut_view", 0, true);
    // Audited deep materializer (returns independent owned copy)
    reg.register_deep_materializer("materialize_owned");
    // Audited synchronous non-escaping read-only helpers
    reg.sync_read_safe_callees.insert("safe_read_helper".to_string());
    reg.sync_read_safe_callees.insert("safe_read_helper_b".to_string());
    // Audited suspending helper
    reg.register_suspending("suspend_point");
    reg.register_suspending("yield_now");
    reg
}

fn check_lifetime_with_synthetic(source: &str) -> lust::Result<()> {
    let items = parse_items(source);
    // Base typechecker parity check (with default registry, plain code passes)
    check_module_lifetimes_with_registry(&items, &None, &synthetic_registry())
}

fn assert_lifetime_err(source: &str, code: LifetimeDiagCode, expected_snippet: &str) {
    let err = check_lifetime_with_synthetic(source)
        .expect_err("expected lifetime diagnostic error, got Ok");
    match &err {
        lust::error::LustError::TypeErrorWithSpan {
            message,
            line,
            column,
            ..
        } => {
            assert!(*line > 0, "expected nonzero offending line span, got line {line}");
            assert!(
                *column > 0,
                "expected nonzero offending column span, got column {column}"
            );
            let expected_prefix = format!("{}: {}", code.as_str(), code.primary_message());
            assert!(
                message.starts_with(&expected_prefix),
                "expected exact prefix {:?}, got message {:?}",
                expected_prefix,
                message
            );
            if !expected_snippet.is_empty() {
                assert!(
                    message.contains(expected_snippet),
                    "expected span/context snippet {:?}, got: {}",
                    expected_snippet,
                    message
                );
            }
        }
        other => panic!(
            "expected typed LustError::TypeErrorWithSpan for {}, got {:?}",
            code.as_str(),
            other
        ),
    }
}

fn assert_positive_pair(positive_src: &str, negative_src: &str, code: LifetimeDiagCode) {
    assert!(
        typecheck_only(positive_src).is_ok(),
        "positive control failed base TypeChecker: {:?}",
        typecheck_only(positive_src)
    );
    assert!(
        check_lifetime_with_synthetic(positive_src).is_ok(),
        "positive control failed lifetime checker: {:?}",
        check_lifetime_with_synthetic(positive_src)
    );
    assert!(
        typecheck_only(negative_src).is_ok(),
        "negative control must parse and pass base TypeChecker before lifetime rejection: {:?}",
        typecheck_only(negative_src)
    );
    assert_lifetime_err(negative_src, code, "");
}

// Stubs helper header so real TypeChecker accepts calls cleanly
const STUB_HEADER: &str = r#"
function borrow_view(owner: int): int
    return owner
end
function borrow_mut_view(owner: int): int
    return owner
end
function materialize_owned(v: int): int
    return v
end
function safe_read_helper(v: int): int
    return v
end
function safe_read_helper_b(v: int): int
    return v + 1
end
function unknown_external_sink(v: int): int
    local _x: int = math.abs(v)
    return 0
end
function suspend_point(v: int): int
    return v
end
function drop(owner: int): int
    return 0
end
function spawn(task_arg: int): int
    return 0
end
"#;

#[test]
fn w4_4g_non_borrow_existing_semantics_unaffected() {
    let src = r#"
function add_pair(a: int, b: int): int
    local sum: int = a + b
    return sum
end
"#;
    assert!(typecheck_only(src).is_ok());
    let items = parse_items(src);
    assert!(check_module_lifetimes(&items, &None).is_ok());
}

#[test]
fn w4_4g_scoped_read_and_sequential_write_controls() {
    let pos = format!(
        "{STUB_HEADER}
function test_seq(): int
    local owner: int = 41
    scoped do
        local v: int = borrow_view(owner)
        local r: int = safe_read_helper(v)
        local _unused: int = r
    end
    owner = 42
    return owner
end
function test_nested(): int
    local owner: int = 42
    scoped do
        scoped do
            local v: int = borrow_view(owner)
            local _r: int = safe_read_helper(v)
        end
    end
    return owner
end
"
    );
    let items = parse_items(&pos);
    assert!(typecheck_only(&pos).is_ok());
    assert!(check_module_lifetimes_with_registry(&items, &None, &synthetic_registry()).is_ok());

    let mut prog = EmbeddedProgram::builder()
        .module("main", &pos)
        .entry_module("main")
        .compile()
        .expect("compile runtime seq");
    let got = prog.vm_mut().call("main.test_seq", vec![]).expect("run test_seq");
    assert_eq!(got, Value::Int(42));
}

#[test]
fn w4_4g_e4g001_return_local_store_global_and_aggregate_escape() {
    let pos = format!(
        "{STUB_HEADER}
local GLOBAL_SLOT: int = 0
function ok_return(): int
    local owner: int = 42
    local v: int = borrow_view(owner)
    local owned: int = materialize_owned(v)
    return owned
end
"
    );

    let neg_return = format!(
        "{STUB_HEADER}
function bad_return(): int
    local owner: int = 42
    local v: int = borrow_view(owner)
    return v
end
"
    );
    assert_positive_pair(&pos, &neg_return, LifetimeDiagCode::E4G001);

    let neg_global = format!(
        "{STUB_HEADER}
local GLOBAL_SLOT: int = 0
function bad_global(): int
    local owner: int = 42
    local v: int = borrow_view(owner)
    GLOBAL_SLOT = v
    return 0
end
"
    );
    assert_positive_pair(&pos, &neg_global, LifetimeDiagCode::E4G001);

    let neg_aggregate_escape = format!(
        "{STUB_HEADER}
function bad_agg(): Array<int>
    local owner: int = 42
    local v: int = borrow_view(owner)
    local arr: Array<int> = [v]
    return arr
end
"
    );
    assert_positive_pair(&pos, &neg_aggregate_escape, LifetimeDiagCode::E4G001);
}

#[test]
fn w4_4g_e4g002_closure_capture_immediate_vs_escaping() {
    let pos = format!(
        "{STUB_HEADER}
function ok_closure(): int
    local owner: int = 42
    local v: int = borrow_view(owner)
    local r: int = (function(): int
        return safe_read_helper(v)
    end)()
    return r
end
"
    );

    let neg_escape = format!(
        "{STUB_HEADER}
function bad_closure(): int
    local owner: int = 42
    local v: int = borrow_view(owner)
    local cl = function(): int
        return v
    end
    return 0
end
"
    );
    assert_positive_pair(&pos, &neg_escape, LifetimeDiagCode::E4G002);

    let neg_alias_in_closure = format!(
        "{STUB_HEADER}
function bad_closure_agg(): int
    local owner: int = 42
    local v: int = borrow_view(owner)
    local alias_v: int = v
    local cl = function(): int
        return alias_v
    end
    return 0
end
"
    );
    assert_positive_pair(&pos, &neg_alias_in_closure, LifetimeDiagCode::E4G002);
}

#[test]
fn w4_4g_e4g003_suspension_and_spawn_boundary() {
    let pos = format!(
        "{STUB_HEADER}
function ok_suspend(): int
    local owner: int = 42
    scoped do
        local v: int = borrow_view(owner)
        local _r: int = safe_read_helper(v)
    end
    local after: int = suspend_point(owner)
    return after
end
"
    );

    let neg_suspend = format!(
        "{STUB_HEADER}
function bad_suspend(): int
    local owner: int = 42
    local v: int = borrow_view(owner)
    local _s: int = suspend_point(0)
    local r: int = safe_read_helper(v)
    return r
end
"
    );
    assert_positive_pair(&pos, &neg_suspend, LifetimeDiagCode::E4G003);

    let neg_spawn = format!(
        "{STUB_HEADER}
function bad_spawn(): int
    local owner: int = 42
    local v: int = borrow_view(owner)
    local _t: int = spawn(v)
    return 0
end
"
    );
    assert_positive_pair(&pos, &neg_spawn, LifetimeDiagCode::E4G003);
}

#[test]
fn w4_4g_e4g004_mutate_drop_alias_branch_loop_defer_and_controls() {
    let pos_controls = format!(
        "{STUB_HEADER}
function ok_shadow_and_distinct(): int
    local owner_a: int = 10
    local owner_b: int = 20
    local v_a: int = borrow_view(owner_a)
    -- distinct owner write is allowed
    owner_b = 30
    -- shadowed owner name introduces an independent binding identity
    local owner_a: int = 99
    owner_a = 100
    local _r: int = safe_read_helper(v_a)
    return owner_b + owner_a
end
"
    );

    let neg_write_while_shared = format!(
        "{STUB_HEADER}
function bad_write(): int
    local owner: int = 41
    local v: int = borrow_view(owner)
    owner = 42
    return safe_read_helper(v)
end
"
    );
    assert_positive_pair(&pos_controls, &neg_write_while_shared, LifetimeDiagCode::E4G004);

    let neg_drop_while_shared = format!(
        "{STUB_HEADER}
function bad_drop(): int
    local owner: int = 41
    local v: int = borrow_view(owner)
    drop(owner)
    return safe_read_helper(v)
end
"
    );
    assert_positive_pair(&pos_controls, &neg_drop_while_shared, LifetimeDiagCode::E4G004);

    let neg_mut_shared_conflict = format!(
        "{STUB_HEADER}
function bad_mut_shared(): int
    local owner: int = 41
    local s: int = borrow_view(owner)
    local m: int = borrow_mut_view(owner)
    return safe_read_helper(s) + safe_read_helper(m)
end
"
    );
    assert_positive_pair(&pos_controls, &neg_mut_shared_conflict, LifetimeDiagCode::E4G004);

    let neg_branch_join = format!(
        "{STUB_HEADER}
function bad_branch(cond: bool): int
    local owner: int = 41
    local v: int = owner
    if cond then
        v = borrow_view(owner)
    end
    owner = 42
    return safe_read_helper(v)
end
"
    );
    assert_positive_pair(&pos_controls, &neg_branch_join, LifetimeDiagCode::E4G004);

    let neg_loop_backedge = format!(
        "{STUB_HEADER}
function bad_loop(n: int): int
    local owner: int = 41
    local i: int = 0
    local v: int = owner
    while i < n do
        owner = 42
        v = borrow_view(owner)
        i = i + 1
    end
    return safe_read_helper(v)
end
"
    );
    assert_positive_pair(&pos_controls, &neg_loop_backedge, LifetimeDiagCode::E4G004);

    let neg_defer = format!(
        "{STUB_HEADER}
function bad_defer(): int
    local owner: int = 41
    local v: int = borrow_view(owner)
    defer safe_read_helper(v)
    owner = 42
    return 0
end
"
    );
    assert_positive_pair(&pos_controls, &neg_defer, LifetimeDiagCode::E4G004);
}

#[test]
fn w4_4g_e4g005_indirect_calls_summaries_and_unknown_casts() {
    let pos_indirect_finite = format!(
        "{STUB_HEADER}
function ok_indirect(flag: bool): int
    local owner: int = 42
    local v: int = borrow_view(owner)
    local f = safe_read_helper
    if flag then
        f = safe_read_helper_b
    end
    return f(v)
end
"
    );

    let neg_unknown_call = format!(
        "{STUB_HEADER}
function bad_call(): int
    local owner: int = 42
    local v: int = borrow_view(owner)
    return unknown_external_sink(v)
end
"
    );
    assert_positive_pair(&pos_indirect_finite, &neg_unknown_call, LifetimeDiagCode::E4G005);

    let neg_indirect_unsafe_target = format!(
        "{STUB_HEADER}
function bad_indirect(flag: bool): int
    local owner: int = 42
    local v: int = borrow_view(owner)
    local f = safe_read_helper
    if flag then
        f = unknown_external_sink
    end
    return f(v)
end
"
    );
    assert_positive_pair(&pos_indirect_finite, &neg_indirect_unsafe_target, LifetimeDiagCode::E4G005);

    // Unknown provenance / cast cannot erase borrow taint
    let neg_cast_erasure = format!(
        "{STUB_HEADER}
function bad_cast(): int
    local owner: int = 42
    local v: int = borrow_view(owner)
    local erased: unknown = v
    return unknown_external_sink(erased)
end
"
    );
    assert_positive_pair(&pos_indirect_finite, &neg_cast_erasure, LifetimeDiagCode::E4G005);
}

#[test]
fn w4_4g_precedence_rule_e4g003_over_e4g002_over_e4g001() {
    let suspend_vs_return = format!(
        "{STUB_HEADER}
function prec(): int
    local owner: int = 42
    local v: int = borrow_view(owner)
    local _s: int = suspend_point(v)
    return v
end
"
    );
    assert_lifetime_err(&suspend_vs_return, LifetimeDiagCode::E4G003, "");
}

#[test]
fn w4_4g_recursive_cycle_pessimism() {
    let pos_non_cyclic = format!(
        "{STUB_HEADER}
function helper_leaf(x: int): int
    return safe_read_helper(x)
end
function ok_call(): int
    local owner: int = 42
    local v: int = borrow_view(owner)
    return helper_leaf(v)
end
"
    );

    let neg_direct_rec = format!(
        "{STUB_HEADER}
function rec_self(x: int, depth: int): int
    if depth > 0 then
        return rec_self(x, depth - 1)
    end
    return 0
end
function bad_rec(): int
    local owner: int = 42
    local v: int = borrow_view(owner)
    return rec_self(v, 2)
end
"
    );
    assert_positive_pair(&pos_non_cyclic, &neg_direct_rec, LifetimeDiagCode::E4G005);

    let neg_mutual_rec = format!(
        "{STUB_HEADER}
function mut_a(x: int, depth: int): int
    if depth > 0 then
        return mut_b(x, depth - 1)
    end
    return 0
end
function mut_b(x: int, depth: int): int
    if depth > 0 then
        return mut_a(x, depth - 1)
    end
    return 0
end
function bad_mutual(): int
    local owner: int = 42
    local v: int = borrow_view(owner)
    return mut_a(v, 2)
end
"
    );
    assert_positive_pair(&pos_non_cyclic, &neg_mutual_rec, LifetimeDiagCode::E4G005);
}

// ---------------------------------------------------------------------------
// Real Embed Handles & Fixed-Scalar LustStructView Allocation Oracle (Row 5-6)
// ---------------------------------------------------------------------------

#[derive(lust::LustStructView)]
#[lust(type = "main.FixedBox")]
struct FixedBoxView<'a> {
    #[lust(field = "value")]
    value: ValueRef<'a>,
}

#[test]
fn w4_4g_embed_handles_real_api_inventory_and_borrow_guards() {
    let source = r#"
struct FixedBox
    value: int
end

function add_one(x: int): int
    return x + 1
end
"#;
    let mut program = EmbeddedProgram::builder()
        .module("main", source)
        .entry_module("main")
        .compile()
        .expect("compile embed handle test program");

    // 1. StructInstance + StructHandle + ValueRef + clone aliasing
    let inst: StructInstance = program
        .struct_instance("main.FixedBox", [struct_field("value", 42_i64)])
        .expect("struct instance");
    let inst_clone = inst.clone();
    assert_eq!(inst.field::<i64>("value").unwrap(), 42);
    inst_clone.set_field("value", 43_i64).unwrap();
    assert_eq!(inst.field::<i64>("value").unwrap(), 43);

    // Live ValueRef holds RefCell dynamic borrow: conflicting mutation fails deterministically
    {
        let live_ref: ValueRef<'_> = inst.borrow_field("value").unwrap();
        assert_eq!(live_ref.as_int(), Some(43));
        let mutate_during_borrow =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| inst.set_field("value", 44_i64)));
        assert!(
            mutate_during_borrow.is_err(),
            "RefCell borrow must deterministically reject conflicting mutation while ValueRef is alive"
        );
    }
    inst.set_field("value", 42_i64).unwrap();

    // 2. LustStructView
    let handle: StructHandle = inst.to_handle();
    let view = FixedBoxView::from_handle(&handle).expect("view construct");
    assert_eq!(view.value.as_int(), Some(42));
    drop(view);

    // 3. ArrayHandle + MapHandle via real VM values and live ValueRef borrow guard
    let map_src = r#"
function make_arr(): Array<int>
    return [42]
end
function make_map(): Map<string, int>
    return { k = 42 }
end
"#;
    let mut coll_prog = EmbeddedProgram::builder()
        .module("main", map_src)
        .entry_module("main")
        .compile()
        .expect("compile collection program");
    let vm_arr_val = coll_prog.vm_mut().call("main.make_arr", vec![]).expect("make_arr");
    let vm_arr = ArrayHandle::from_value(vm_arr_val).expect("vm ArrayHandle");
    {
        let elem = vm_arr.get(0).unwrap();
        assert_eq!(elem.as_int(), Some(42));
        let push_while_borrowed =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| vm_arr.push(Value::Int(99))));
        assert!(push_while_borrowed.is_err());
    }
    let vm_map_val = coll_prog.vm_mut().call("main.make_map", vec![]).expect("make_map");
    let map_handle = MapHandle::from_value(vm_map_val).expect("vm MapHandle");
    assert_eq!(map_handle.get("k").unwrap().as_int(), Some(42));

    // 4. FunctionHandle
    let fn_handle: FunctionHandle = program.function_handle("main.add_one").expect("function_handle");
    let call_res: i64 = fn_handle.call_typed(&mut program, (41_i64,)).expect("call_typed");
    assert_eq!(call_res, 42);
}

#[test]
fn w4_4g_row6_fixed_scalar_view_1000_reads_zero_alloc() {
    let _guard = ALLOC_TEST_MUTEX.lock().expect("alloc test mutex");
    IS_MEASURED_THREAD.with(|c| c.set(true));

    let source = r#"
struct FixedBox
    value: int
end
"#;
    let program = EmbeddedProgram::builder()
        .module("main", source)
        .entry_module("main")
        .compile()
        .expect("compile fixed box");
    let inst = program
        .struct_instance("main.FixedBox", [struct_field("value", 42_i64)])
        .expect("struct instance");
    let handle = inst.to_handle();

    // Warmup before interval
    let warm = FixedBoxView::from_handle(&handle).expect("warm view");
    assert_eq!(warm.value.as_int(), Some(42));
    drop(warm);

    // Validate counter with deliberate allocation outside interval
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    ALLOC_ZEROED_COUNT.store(0, Ordering::SeqCst);
    REALLOC_COUNT.store(0, Ordering::SeqCst);
    TRACKING_ACTIVE.store(true, Ordering::Release);
    let canary: Vec<u8> = black_box(Vec::with_capacity(64));
    TRACKING_ACTIVE.store(false, Ordering::Release);
    drop(canary);
    let canary_allocs = ALLOC_COUNT.load(Ordering::Acquire);
    assert!(canary_allocs >= 1, "counting allocator calibration failed");

    // Reset counters and execute 1000 prepared fixed-scalar view reads
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    ALLOC_ZEROED_COUNT.store(0, Ordering::SeqCst);
    REALLOC_COUNT.store(0, Ordering::SeqCst);
    TRACKING_ACTIVE.store(true, Ordering::Release);

    let mut sum: i64 = 0;
    for _ in 0..1000 {
        let view = FixedBoxView::from_handle(black_box(&handle)).expect("view");
        sum += black_box(view.value.as_int().expect("i64"));
    }

    TRACKING_ACTIVE.store(false, Ordering::Release);
    IS_MEASURED_THREAD.with(|c| c.set(false));

    let allocs = ALLOC_COUNT.load(Ordering::Acquire);
    let zeroed = ALLOC_ZEROED_COUNT.load(Ordering::Acquire);
    let reallocs = REALLOC_COUNT.load(Ordering::Acquire);

    eprintln!(
        "4G_ROW6_ALLOC_ORACLE api=LustStructView(FixedBoxView::from_handle+ValueRef::as_int) reads=1000 sum={} alloc={} alloc_zeroed={} realloc={}",
        sum, allocs, zeroed, reallocs
    );
    assert_eq!(sum, 42000);
    assert_eq!(allocs, 0, "expected 0 heap allocs during 1000 fixed-scalar reads");
    assert_eq!(zeroed, 0, "expected 0 alloc_zeroed calls during 1000 fixed-scalar reads");
    assert_eq!(reallocs, 0, "expected 0 realloc calls during 1000 fixed-scalar reads");
}

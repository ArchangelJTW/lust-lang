#![cfg(feature = "std")]

use lust::{EmbeddedProgram, LustConfig, Value};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

// Isolate allocation accounting to this integration-test binary and to the
// current thread. Compilation, warming the frame pool/name interner, and
// constructing arguments happen before tracking is enabled.
struct CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

thread_local! {
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}

fn record_allocation(pointer: *mut u8) {
    if !pointer.is_null() {
        let _ = ALLOCATIONS.try_with(|count| {
            if let Some(current) = count.get() {
                count.set(Some(current + 1));
            }
        });
    }
}

// SAFETY: allocation and deallocation are delegated to System with the
// original pointer and layout. Accounting does not allocate or dereference
// the pointer, and uses thread-local state to avoid cross-test interference.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        record_allocation(pointer);
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        record_allocation(pointer);
        pointer
    }
}

fn call_counted(
    program: &mut EmbeddedProgram,
    value: &Value,
    iterations: i64,
) -> lust::Result<usize> {
    let args = vec![value.clone(), Value::Int(iterations)];
    ALLOCATIONS.with(|count| count.set(Some(0)));
    let result = program.vm_mut().call("alloc.copy_value", args);
    let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    // Drop the returned value only after accounting has stopped.
    result?;
    Ok(allocations)
}

#[test]
fn value_copies_do_not_allocate_cycle_discovery_bookkeeping() -> lust::Result<()> {
    for (ty, initializer) in [
        ("Option<int>", "Option.Some(1)"),
        ("Result<int, string>", "Result.Ok(1)"),
        ("(int, int)", "(1, 2)"),
        ("Array<int>", "[1]"),
        ("Map<string, int>", "{ item = 1 }"),
        ("Point", "Point { x = 1 }"),
    ] {
        let source = format!(
            r#"
                struct Point
                    x: int
                end
                function make_value(): {ty}
                    return {initializer}
                end
                function copy_value(original: {ty}, n: int): {ty}
                    local copied: {ty} = original
                    local i: int = 0
                    while i < n do
                        copied = original
                        i = i + 1
                    end
                    return copied
                end
            "#
        );
        let mut config = LustConfig::default();
        config.set_jit_enabled(false);
        let mut program = EmbeddedProgram::builder()
            .with_config(config)
            .module("alloc", source)
            .entry_module("alloc")
            .compile()?;
        let value = program.vm_mut().call("alloc.make_value", Vec::new())?;
        program
            .vm_mut()
            .call("alloc.copy_value", vec![value.clone(), Value::Int(1_000)])?;
        let short = call_counted(&mut program, &value, 1)?;
        let long = call_counted(&mut program, &value, 1_000)?;
        // Entering a host call may allocate, but copying the same value
        // into a register must not allocate once per iteration.
        // Both calls stay below the periodic whole-heap collection interval.
        assert_eq!(short, long, "{ty}: allocation count grew with copies");
    }
    Ok(())
}

#[test]
fn disabled_gc_tracks_zero_containers_and_skips_cycles() -> lust::Result<()> {
    let source = r#"
        struct Node
            value: int
            next: Option<Node>
        end

        function create_chain(n: int): int
            local head = Node { value = 0, next = Option.None }
            local i = 1
            while i < n do
                head = Node { value = i, next = Option.Some(head) }
                i = i + 1
            end
            return head.value
        end
    "#;

    let mut config = LustConfig::default();
    config.set_gc_enabled(false);
    assert!(!config.gc_enabled());

    let mut program = EmbeddedProgram::builder()
        .with_config(config)
        .module("chain", source)
        .entry_module("chain")
        .compile()?;

    assert!(!program.vm_mut().is_gc_enabled());
    let res = program.vm_mut().call("chain.create_chain", vec![Value::Int(1_000)])?;
    assert_eq!(res, Value::Int(999));
    assert_eq!(program.vm_mut().cycle_collector_tracked_count(), 0);

    // Forcing collection with disabled GC must be a safe no-op
    program.vm_mut().force_collect_cycles();
    assert_eq!(program.vm_mut().cycle_collector_tracked_count(), 0);

    Ok(())
}


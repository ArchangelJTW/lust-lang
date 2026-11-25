# Lust Language - Examples & Guides

Runnable examples (Effect-TS / ZIO style) for the **Effect System v2.0** and **Testing Framework (`@std/test`)** in Lust.

---

## Example Catalog

All examples live under [`examples/effects/`](./effects/).

| # | Example | File | Run Command | Description |
|---|---|---|---|---|
| **01** | **Basic Effects & Exit** | [`01_basic_effects.rs`](./effects/01_basic_effects.rs) | `cargo run --example effects_basic` | Core `Exit<A, E>` abstractions, error handling with `Cause::Fail`, fatal defects `Cause::Die`, interruptions `Cause::Interrupted`, and monadic `and_then`. |
| **02** | **Retry & Repeat Policies** | [`02_retry_and_repeat.rs`](./effects/02_retry_and_repeat.rs) | `cargo run --example effects_retry_repeat` | `ScheduleKind` policy algebra (recurs, exponential backoff with jitter). Verifies the fail-closed invariant: `OutOfMemory`/`OutOfGas` interrupts are never retried. |
| **03** | **Safe Resource Scopes** | [`03_acquire_release_and_scopes.rs`](./effects/03_acquire_release_and_scopes.rs) | `cargo run --example effects_acquire_release` | `acquire_release` idiom with interrupt masking during acquisition (`enter_mask`/`exit_mask`) and finalizer registration in a `Scope`. |
| **04** | **Virtual Clock Testing** | [`04_testing_with_virtual_clock.rs`](./effects/04_testing_with_virtual_clock.rs) | `cargo run --example 04_testing_with_virtual_clock` | Unit testing effects with virtual time (`TestClockState`). Simulates exponential backoff (>120 seconds of virtual time) in **< 1 ms** of real time without `std::thread::sleep`. |
| **05** | **Mocking & Stack Projection** | [`05_dependency_injection_mocking.rs`](./effects/05_dependency_injection_mocking.rs) | `cargo run --example 05_dependency_injection_mocking` | Dependency substitution via `MockService`, call spies, stubbed responses, and zero-allocation stack environment projection (`StackContextProjection<N>`). |
| **06** | **Structured Concurrency** | [`06_concurrency_and_supervision.rs`](./effects/06_concurrency_and_supervision.rs) | `cargo run --example 06_concurrency_and_supervision` | Structured concurrency: fiber trees in a `Scope`, automatic cascading cancellation on failure, and **Gate-NEG-02** (preventing host resource release until all child tasks complete, followed by LIFO finalization). |
| **07** | **Lust Test Syntax Demo** | [`07_test_syntax_demo.lust`](./effects/07_test_syntax_demo.lust) | Lust source file | Native `@std/test` / `std.test` syntax: `suite`, `describe`, `test`, `test.skip(...)`, `test.timeout(...)`, `before_each`/`after_each`/`before_all`/`after_all` hooks, and effect assertions. |

---

## Running & Verifying

Each Rust example compiles and runs standalone:

```bash
# 1. Virtual Clock Testing (zero-delay simulation)
cargo run --example 04_testing_with_virtual_clock

# 2. Dependency Injection & Mocking (StackContextProjection)
cargo run --example 05_dependency_injection_mocking

# 3. Structured Concurrency, Fiber Supervision & Gate-NEG-02
cargo run --example 06_concurrency_and_supervision
```

To run the test framework integration suite:

```bash
cargo test --test std_test_framework
```

---

## Architectural Notes

### 1. Virtual Time (`TestClockState`)
Enables deterministic testing of retry loops, exponential delays, and timeouts. The timer queue is ordered in a min-heap (`BinaryHeap<Reverse<ScheduledTimer>>`), and `adjust(delta_ms)` immediately wakes fibers whose timers have elapsed.

### 2. Zero-Heap `StackContextProjection`
Represents the environment context (`StackContextProjection<N>`) as a flat stack array of service pointers and tags (`[*const c_void; N]`, `[u32; N]`), avoiding heap allocations when resolving services on hot effect paths.

### 3. Gate-NEG-02 Invariant (Structured Concurrency Teardown)
On completion or failure of a `Scope`:
1. **Admission Closed**: New child tasks and resources are immediately rejected (`ScopeAdmission::Closed`).
2. **Cascading Cancellation**: All active fibers receive `InterruptReason::Cancelled`.
3. **Joining Barrier**: The scope enters `ScopeTeardown::Joining` and waits for all child tasks to stop. No host resource is finalized while fibers are still active or inside masked critical sections.
4. **LIFO Finalization**: Once all tasks have joined, finalizers run in strict reverse (LIFO) order.

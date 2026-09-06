# Runtime Performance

Run the numeric VM/x86_64 JIT microbenchmarks with:

```sh
cargo run --release --example numeric
```

The harness compiles typed Lust functions, warms each function with 1,000
iterations, and reports the median of seven calls. Parsing, typechecking, and
initial trace compilation are outside the timer. Each result is checked against
the expected sum; on x86_64 the harness also checks that native traces compiled
and executed. The VM runs 250,000 iterations per call and the JIT runs
100,000,000. The example target uses the ordinary release profile, including
its `panic = "abort"` setting.

## Measurements

Measured against `8628164` with the same harness, on a Ryzen 7 7800X3D with
Rust 1.96.0 and the existing release settings (`opt-level = "z"`, fat LTO).
The two binaries were run sequentially, pinned to logical CPU 2 with
`taskset -c 2`. Times below are milliseconds per call, not startup times.

| Mode | Workload | Before | After |
|---|---|---:|---:|
| VM | Integer sum | 98.471 | 61.073 |
| VM | Ascending float sum | 98.484 | 60.593 |
| VM | Descending float sum | 114.731 | 70.258 |
| x86_64 JIT | Integer sum | 55.342 | 56.701 |
| x86_64 JIT | Ascending float sum | 434.576 | 272.564 |
| x86_64 JIT | Descending float sum | 271.246 | 267.936 |

These are microbenchmarks, not a general language speedup claim. Across repeated
runs the VM improvement varied from approximately 25% to 40% less elapsed time.
The ascending float JIT improvement was consistently around 37%-40%. There was
no clear improvement in the already-fast integer JIT loop or the descending
float JIT loop; the small differences in those rows should not be treated as
reliable gains.

## Implemented Changes

- Cycle discovery returns immediately for leaf values, avoiding a heap-allocated
  traversal stack and cloning on every scalar register write and scalar VM root.
  Owning tuples, enums, closures, arrays, maps, structs, and iterators still get
  traversed. Register replacement and periodic collection are unchanged.
- VM integer ordering compares integers directly instead of converting to
  floats. This also fixes incorrect comparisons of large adjacent integers and
  brings them into agreement with the x86_64 JIT.
- x86_64 codegen fuses known numeric comparisons with branch guards, including
  floating-point and mixed numeric operands. It avoids materializing a dead
  boolean on the successful path, but restores the condition on bailout.
  NaNs, infinities, signed zero, live conditions, and both branch directions are
  covered by generated-code tests.
- Conditions already proven to be booleans use a direct payload test instead
  of generic truthiness dispatch. Unknown conditions retain the existing path.

The RISC-V JIT is unchanged. These changes use existing runtime/codegen type
facts; they do not yet introduce a new static typed-bytecode pipeline.

## Further Opportunities

1. **Carry expression types into bytecode lowering.** `TypeChecker` collects
   expression types, but `Compiler` receives function signatures and selected
   lowering hints, not general expression types. The current
   `Function::register_types` map covers annotated locals/parameters and loop
   variables, not every value at every instruction. Reused registers and erased
   generics make that map unsuitable as a blanket guarantee. Program-point type
   facts could select typed arithmetic and direct struct-field access. Raw VM
   and native boundaries still need validation or guards.
2. **Keep JIT scalars in machine registers.** Numeric codegen already selects
   integer/SSE instructions, but normally loads operands from and stores results
   to the boxed VM register array for every operation. Register allocation across
   operations and loop iterations could remove this traffic. It needs explicit
   spill/materialization rules for helper calls and every side exit.
3. **Offer a speed-oriented build profile.** The current release profile favors
   minimum binary size with `opt-level = "z"`. Benchmarking `opt-level = 3` in a
   separate desktop/server profile is worthwhile, especially for interpreter
   dispatch and runtime helpers. The embedded-size default was not changed.

## Verification

- `cargo test --workspace --quiet`: 144 tests passed; one doctest ignored.
- `cargo rustc --no-default-features --lib --crate-type rlib`: passed. A standalone
  `no_std` cdylib still requires the embedding target's allocator/panic runtime.
- The 1,255-case JIT differential corpus produced byte-for-byte identical JSON
  reports before and after: 1,148 `MATCH_OK`, 107 existing `FRONTEND` failures.
  Those frontend failures are not counted as passing tests; dedicated native
  code tests and the typed benchmark cover floating-point behavior here.

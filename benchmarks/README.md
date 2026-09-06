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

## Typed Numeric Lowering

The typechecker now passes compact per-expression `Int`/`Float` facts to the
bytecode compiler, including in low-memory mode. This is separate from editor
type information and from the function-wide `register_types` map. CLI,
embedding, embedded-module, and analyzer compilation paths all pass these facts.
They are compilation metadata, not additional per-register runtime storage.

Homogeneous numeric arithmetic, negation, and comparisons lower to typed
instructions such as `AddInt`, `AddFloat`, and `LtInt`. Mixed numeric operands,
unknown types, and erased generic operands retain generic instructions. The
interpreter checks the required input kinds rather than dispatching among all
numeric combinations. Trace lowering preserves that contract and retains type
guards before native payload access, including after host mutation.

Typed arithmetic results assigned to locals are written directly to the local
when the temporary is dead and the expression is straight-line. For example,
`sum = sum + i` now needs one arithmetic instruction rather than arithmetic plus
`Move`. Its input kinds remain recoverable by the post-execution trace recorder
even when the destination aliases an input. Generic aliased arithmetic and
aliased comparisons still require pre-execution operands and are not enabled.
The old JIT arithmetic/move peephole was removed because it could discard a live
local update; elimination now happens in the compiler with lifetime information.

Two correctness prerequisites accompany the lowering:

- Composite expressions have distinct source spans instead of overwriting their
  left operand's type information.
- Compound assignments and numeric loop steps cannot silently change a statically
  integer binding to float. A fractional loop step requires a float start value.

### Measurements

Compared with `542c1fd`, using the unchanged harness and the current release
profile (`opt-level = 3`, fat LTO), on the same Ryzen 7 7800X3D and Rust 1.96.0.
Both binaries were run sequentially with `taskset -c 2`. These times are
milliseconds per call; each is the median of seven samples.

| Mode | Workload | Before | After |
|---|---|---:|---:|
| VM | Integer sum | 46.175 | 31.502 |
| VM | Ascending float sum | 48.025 | 31.571 |
| VM | Descending float sum | 54.091 | 38.780 |
| x86_64 JIT | Integer sum | 55.017 | 56.525 |
| x86_64 JIT | Ascending float sum | 270.495 | 270.687 |
| x86_64 JIT | Descending float sum | 266.622 | 266.330 |

Typed opcode dispatch alone was approximately flat in this optimized build.
Removing redundant VM instructions produced the material improvement: roughly
25%-35% less VM time across repeated runs. Native float timings were unchanged;
the integer JIT loop was about 3% slower in this pair, with small variation
between runs. No JIT speedup is claimed. These are microbenchmarks, not a general
language speedup claim, and generic/mixed paths are not represented by this table.

## Earlier Measurements

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

## Earlier Changes

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

Those earlier optimizations used runtime/codegen type facts rather than the new
static lowering described above. RISC-V codegen has not been modified.

## Further Opportunities

1. **Extend static lowering to fields and mixed numeric operations.** Resolved
   receiver/layout information could replace field-name lookup in the VM.
   Strong-field mutation currently accepts arbitrary host `Value`s, so declared
   field types alone do not justify unchecked payload access. The function-wide
   `register_types` map remains unsuitable as a substitute for program-point facts.
2. **Keep JIT scalars in machine registers.** Numeric codegen already selects
   integer/SSE instructions, but normally loads operands from and stores results
   to the boxed VM register array for every operation. Register allocation across
   operations and loop iterations could remove this traffic. It needs explicit
   spill/materialization rules for helper calls and every side exit.
3. **Establish verified boundaries before removing input checks.** Static type
   facts select operations today, but raw bytecode, host mutation, and missing or
   erased signatures still need validation. Unchecked scalar storage requires a
   stronger invariant than source annotations alone. The release profile was
   already changed to `opt-level = 3` before this lowering work and is unchanged
   by it.

## Verification

- `cargo test --workspace --quiet`: 153 tests passed; one doctest ignored.
- `cargo test --workspace --features lua_transpile --locked --quiet`: the same
  153 tests passed with optional Lua transpilation enabled.
- `cargo rustc --no-default-features --lib --crate-type rlib`: passed. A standalone
  `no_std` cdylib still requires the embedding target's allocator/panic runtime.
  The same three unused-import/dead-code warnings are present in the baseline.
- The 1,255-case JIT differential corpus produced byte-for-byte identical JSON
  reports before and after: 1,148 `MATCH_OK`, 107 existing `FRONTEND` failures.
  Those frontend failures are not counted as passing tests; dedicated native
  code tests and the typed benchmark cover floating-point behavior here.

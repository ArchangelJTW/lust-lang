# JIT differential tester

Runs a corpus of small Lust programs twice — once with `jit = true`, once with
`jit = false` — and compares both against an expected value computed
independently in Python. Every case prints exactly one integer, so comparison is
exact and there is no float formatting to argue about.

The point is not "did it pass" but **where the two modes disagree**, which
separates JIT bugs from front-end bugs:

| verdict | meaning |
|---|---|
| `MATCH_OK` | both modes correct |
| `JIT_WRONG` | interpreter correct, JIT silently produced a different value |
| `JIT_HANG` | interpreter finished, JIT never terminated |
| `JIT_CRASH` | interpreter finished, JIT panicked or aborted the process |
| `JIT_ERROR` | JIT raised a runtime error, or produced no output, where the interpreter did not |
| `INTERP_WRONG` | the interpreter itself disagrees with the expected value |
| `BOTH_WRONG` | both modes agree with each other and disagree with expected |
| `FRONTEND` | identical parse/type/compile error in both modes |

## Usage

```sh
cargo build --release

# everything (~1200 cases, a few minutes; hangs cost the full timeout)
python3 tools/jit-diff/jitdiff.py

# just the named regressions
python3 tools/jit-diff/jitdiff.py --filter regression/

# one operation across all shapes
python3 tools/jit-diff/jitdiff.py --filter op/struct_method

# the single-loop trip-count sweep only
python3 tools/jit-diff/jitdiff.py --filter hot/ --jobs 12
```

Exits non-zero if any `JIT_*` verdict is present, so it works as a CI gate.
Writes `report.md` (human) and `report.json` (diffable baseline) next to itself.

Useful flags: `--tag array`, `--timeout 8`, `--jobs N`, `--lust path/to/binary`.

## Case corpus

* `regression/*` — named minimal reproductions, kept at the top of the report.
* `shape/*` — loop machinery only; the body is a single `acc = acc + 1`. Sweeps
  nesting depth 1–3, `for` vs `while` vs mixed nesting, trip counts around the
  JIT's own thresholds (`HOT_THRESHOLD = 5`, `LOOP_UNROLL_FACTOR = 32`,
  `MAX_TRACE_LENGTH = 2000`), step values, descending and zero-trip loops.
* `op/<name>/{d1,d2,d2r}` — 48 body operations, each at three shapes that all
  perform exactly 100 inner iterations (`100`, `20x5`, `5x20`). Same iteration
  count means the expected value is identical across shapes, so a difference
  between `d1` and `d2` isolates nesting as the trigger.
* `hot/<name>/n<N>` — every operation in a *single* loop, swept across trip
  counts, dense around the hot threshold. Divergence here needs no nesting, and
  the trip count where it first appears points at the responsible threshold.

Adding a case means adding an `Op` to `OPS()` in `cases.py` with a Python
`expected` function of the iteration count; it is then automatically run at
every shape and every trip count.

## Status

**The JIT now agrees with the interpreter on every case in the corpus.**

| | baseline | now |
|---|---|---|
| `MATCH_OK` | 850 | 1255 |
| `JIT_WRONG` | 121 | 0 |
| `JIT_HANG` | 235 | 0 |
| `JIT_CRASH` | 9 | 0 |
| `JIT_ERROR` | 17 | 0 |
| `BOTH_WRONG` | 3 | 0 |
| `FRONTEND` | 1 | 0 |

No case that passed at baseline regressed, and `cargo test` stayed green
throughout. The former single-letter nominal-type frontend failure is now a
passing scoped-generics regression.

Runtime `unknown` checks are native-trace operations as well: `value is T`
emits a type test, while `value as T` produces `Option<T>`. An immediately
matched `value as T is Some(x)` is fused to a type test and direct binding, so
the hot path never materializes the intermediate option.

Array source reads now produce `Result<T, IndexError>` instead of trapping on
bad bounds. The bytecode keeps internal bounds-proven `GetIndex` operations
separate from source `TryGetIndex` operations. An immediate
`array[index] is Ok(value)` pattern is fused to a bounds test and direct payload
load, avoiding Result materialization. Checked reads also disable an unused
eager array specialization rather than emitting an ownership-transferring
`Rebox` into an unrolled loop body.

Root causes were found by dumping recorded traces with `LUST_TRACE_DEBUG=1` and
bisecting each cluster to a minimal reproduction. In rough order of how many
cases each accounted for:

| # | fix | site |
|---|---|---|
| 1 | `hoist_constants` lifted registers that were not loop-invariant | `jit/optimizer.rs` |
| 2 | `unroll_loop` deleted user comparisons from the duplicated bodies | `jit/optimizer.rs` |
| 3 | Outer loop's partial trace installed under an inner loop's key — all 235 hangs | `vm/execution.rs` |
| 4 | Array specialization scanned all 256 registers, unboxing stale out-of-frame values | `jit/trace.rs` |
| 5 | Failed unbox left stack garbage that the rebox read as `len` — all 9 crashes | `jit/codegen/specialization.rs` |
| 6 | Cross-function instructions were silently *skipped* instead of aborting the trace | `jit/trace.rs` |
| 7 | A trace failing mid-body re-ran the iteration in the interpreter | `vm/execution.rs`, `jit/trace.rs` |
| 8 | `jit_unbox_array_int` left the source array empty for the span of the trace | `bytecode/value.rs` |
| 9 | Numeric `for` ignored the step's sign — a front-end bug | `bytecode/compiler/statements.rs` |

Details on the two most subtle ones (7 and 8) below, then the two optimizer bugs
that started it all.

### A trace that fails mid-body cannot simply restart the loop

`call_builtin_method_simple` handles arrays, iterators and enums; it has no arms
for ints or floats, and `jit_call_method_safe` rejects structs outright. So a
traced `acc = acc + neg:abs()` failed at runtime, and the trace returned `-1`
with `acc` and the loop counter **already mutated**. Two bugs stacked on that:

* the `-1` branch reset `frame.ip` to the loop header and then *fell through*, so
  the interpreter also executed the backward `Jump` and applied its offset on top,
  driving `ip` out of range: the frame was popped and the program exited 0 with no
  output at all. That was the `JIT_ERROR` cluster.
* with that fixed the program ran, but the re-executed iteration was counted
  twice: `55` came out as `57`.

There is no way to undo the partial execution, so the recorder now refuses to
trace a method call whose receiver the runtime helper cannot execute. Lifting that
restriction means giving `call_builtin_method_simple` real Int/Float arms, not
relaxing the check.

### Specialization must not outlive the register it describes

Two distinct failures, same underlying assumption — that a specialized array stays
private to compiled code.

`jit_unbox_array_int` took the buffer with `mem::replace(vec_ref, Vec::new())`,
leaving the source array **empty** until the postamble rebox refilled it. Anything
that re-entered the runtime mid-trace saw a zero-length array: a nested
`for elem in arr` built its iterator from the hollowed-out array and died with
`Array index 2 out of bounds (length: 0)`. It now copies instead of moving, which
is sound because no trace op overwrites the boxed array in place — `VecPush` only
appends, and the rebox publishes the result.

Separately, a specialization describes what a register held *at trace entry*. For
`grid[ctr % 4][ctr % 2]`, the temp holding the inner array was specialized at
entry and then reassigned by `GetIndex { dest: <temp> }` on every iteration, so
the postamble rebox wrote the entry-time row back over a different row of `grid`.
Invalidation existed but was an explicit call at eight recording sites, and
`GetIndex` was not one of them; it is now done centrally in `push_op` so no op can
forget.

### The two optimizer bugs

### `hoist_constants` lifted registers that were not loop-invariant

It decided hoistability by walking ops in order and consulting only what came
*earlier*, so it missed a later op in the same iteration clobbering the register.
For `while i < 10`, the recorded body is:

```
[0] LoadConst  r2 <- 10      ; the loop bound
[2] Lt         r3 <- r1, r2  ; i < 10
[7] LoadConst  r2 <- 1       ; body reuses r2 as a scratch slot
```

Hoisting op 0 out of the loop leaves `r2 == 1` from iteration 2 onward, so the
loop silently starts testing `i < 1`. Now a whole-body scan runs first and only
registers that no other op writes — and that no conflicting `LoadConst` targets
— are hoisted.

### `unroll_loop` deleted user comparisons from the duplicated bodies

It picked the *first* `Lt/Le/Gt/Ge` in the trace as "the loop condition", which
is just as likely to be a user comparison such as `if ctr > 10`. It then
synthesized its own cmp + guard per copy and **stripped every comparison** out of
the copied bodies while keeping the guards that consume their results, so copies
2..32 branched on a stale register — the `and_or` / `bool_local` /
`or_short_circuit` failures.

All of that was redundant: the recorder already records the back-edge's
`JumpIf`/`JumpIfNot` as a `GuardLoopContinue`, so the body *already contains* the
loop's exit test. An unrolled iteration is just the body again, verbatim, which
is what it now emits.

These two interacted, which is why neither showed up alone: the broken unroller's
stale condition made the guard bail straight to the interpreter, masking the
hoisting bug. Fixing the unroller alone made results *worse* (121 -> 159 wrong);
both together fixed 51 cases.

## Still open

### Runtime-erased generics

Generic parameters are resolved from declaration scopes rather than spelling.
The differential corpus covers multi-letter function parameters, recursive
inference through containers, explicit call arguments, generic structs and
instance methods, generic enums and patterns, bounded generic calls, bare trait
values, and single-letter nominal types such as `K`.

Generic functions and nominal types compile to one erased runtime body/type.
Specialized and conditional impls are therefore rejected until Lust has a
reified witness or monomorphization model for those forms.

### JIT activation limits

Root recordings now distinguish completion from abort, retry with bounded
backoff, avoid native trace entry while another trace is recording, and abort
an inner recording when it reaches an enclosing backedge. Ordinary loop traces
remain cached across normal exits. Loop hierarchies deliberately retain the
older conservative warmup and one-shot root behavior; the current nested-loop
guard is removed after its first unlinked exit, so it cannot yet accumulate the
failures needed to attach a reusable side trace.

Pure direct or mutual recursion is counted as function/recursive call activity
but has no bytecode backedge and therefore does not start the cyclic trace
compiler. A hot loop can contain an opaque recursive call and still compile;
the recursive callee remains interpreted. General recursive native compilation
needs a finite-function JIT mode with its own return/deoptimization ABI.

Also unaddressed, and costing throughput rather than correctness: the recorder
unrolls by `LOOP_UNROLL_COUNT` and the optimizer unrolls again by `UNROLL_FACTOR`,
so the two compound — a 21-op body was observed expanding to 544 ops.

### Front-end bugs found on the way, not yet fixed

* `string.byte(s, i)` returns the whole tail as an array rather than one byte.
* `float:floor()` returns a float, not an int.

## Historical: findings as of the first full run

Kept as a record of what the tester surfaced before any fixes. All of the JIT
items below are now fixed; the numbers are the original ones.


### 1. Any loop nest hangs once the outer loop runs 7+ times

Governed *entirely* by the outer trip count, independent of loop kind and of the
inner trip count. Outer ≤ 6 terminates, outer ≥ 7 never does. 79 of 108 depth-3
nests fail. This affects every nested loop in the language.

```lust
local acc: int = 0
for a = 1, 7 do
  for b = 1, 3 do
    acc = acc + 1
  end
end
println(acc)          -- jit=false: 21    jit=true: hangs forever
```

### 2. Numeric `for` loops with 3 or 6 trips lose iterations when traced

Deficit is `inner - 2`, occurs exactly once no matter how many times the outer
loop runs, and only for inner trip counts 3 and 6 — counts 1, 2, 4, 5, 7, 8, 31,
32 and 33 are all correct. `while` loops in the same position never lose
iterations, so this is attached to the numeric `for`.

```lust
local acc: int = 0
for a = 1, 3 do
  for b = 1, 3 do
    acc = acc + 1
  end
end
println(acc)          -- jit=false: 9     jit=true: 8
```

### 3. Fifteen operations are miscompiled in a *plain, single* loop

No nesting required. First failing trip count in brackets.

| operation | first bad n | symptom |
|---|---:|---|
| `arr_for_in` (for-in over array) | 7 | constant deficit of 33 |
| `arr_read_float` (`facc + arr[i]`) | 8 | `capacity overflow` panic, core dump |
| `arr_read_int` (`acc + arr[i]`) | 9 | wrong, error grows with n |
| `arr_read_int_const_index` (`arr[3]`) | 9 | **nondeterministic**, see below |
| `float_div`, `int_neg_abs` | 9 | exits 0 having printed nothing |
| `float_floor`, `float_sqrt` | 9 | 8 extra applications of the method |
| `struct_method` | 9 | 32-bit unsigned wraparound |
| `struct_arr_read`, `struct_write` | 10 | wrong, error grows with n |
| `and_or`, `bool_local` | 11 | condition becomes permanently false |
| `arr_nested_index` | 12 | `runtime error: Cannot index Int` |
| `or_short_circuit` | 64–100 | wrong |

Notably clean in a single loop: plain int/float arithmetic, `if`/`else`,
`continue`, function and closure calls, recursion, `arr_push`/`pop`/`len`/`get`,
array *writes*, struct reads, struct allocation, map operations, string
operations, `Option` construction and matching.

### 4. `and` / `or` compile to a permanently false condition

```lust
local acc: int = 0
local ctr: int = 0
for a = 1, 16 do
  ctr = ctr + 1
  if ctr > 10 and ctr < 90 then
    acc = acc + 1
  end
end
println(acc)          -- jit=false: 6     jit=true: 0
```

Hoisting the condition into a `local hit: bool = ...` behaves identically, so it
is the compiled `and` itself, not the branch. `or` fails the same way at a
higher trip count.

### 5. Method calls in a traced loop execute 8 extra times

The loop trip count is respected; the *method* is applied more often than the
body runs:

```lust
local facc: float = 0.0
local two: float = 2.0
local cnt: int = 0
for a = 1, 16 do
  cnt = cnt + 1
  facc = facc + two:sqrt()
end
println(cnt)                        -- 16, correct
println((facc * 1000.0):to_int())   -- 33941, expected 22627
```

`33941 - 22627 = 11314 = 8 x sqrt(2) x 1000`. Exactly 8 surplus applications,
and the surplus is constant for every trip count from 9 upward. `float_floor`
shows the same thing arithmetically: constant surplus of 24 = 8 x 3. Replacing
`two:sqrt()` with the literal `1.4142135623730951` in the identical loop gives
the correct answer, so the duplication is specific to the method-call op inside
the unrolled trace body — the unroller appears to be replicating some ops a
different number of times than others.

### 6. Integers round-trip through unsigned 32 bits

```lust
struct Counter
  n: int
end
impl Counter
  function bump(self): int
    self.n = self.n + 1
    return self.n
  end
end
local c = Counter { n = 0 }
local acc: int = 0
for a = 1, 100 do
  acc = acc + c:bump()
end
println(acc)          -- jit=false: 5050    jit=true: 4294967100
```

`4294967100 = 2^32 - 196`. The accumulator goes negative (itself wrong) and is
then widened as *unsigned* 32-bit rather than sign-extended from 64-bit. Two
separate defects stacked in one case.

### 7. Nondeterminism: the JIT sometimes sees a live array as length 0

Roughly half of runs of the identical program, no input, no concurrency:

```lust
local arr: Array<int> = [1, 2, 3, 4, 5, 6, 7, 8]
local acc: int = 0
for a = 1, 10 do
  acc = acc + arr[3]
end
println(acc)
```
```
40
runtime error: Array index 3 out of bounds (length: 0)
runtime error: Array index 3 out of bounds (length: 0)
40
runtime error: Array index 3 out of bounds (length: 0)
40
```

Nondeterministic output on a deterministic program means the compiled trace is
reading a stale or uninitialised pointer to the array buffer — most likely the
refcount/GC reclaiming or moving storage the trace still holds. This is memory
unsafety in generated code and is the one to fix first, ahead of the
correctness bugs, because it makes every other result unreliable.

### Not a JIT bug, but found on the way

* **Descending `for` loops run zero iterations**, in both modes:
  `for i = 5, 1, -1 do ... end` never executes its body. Lua counts down here.
  Caught as `BOTH_WRONG`, which is exactly what that verdict is for.
  *(Fixed — see fix 9.)*
* A struct named with a single uppercase letter collides with a generic type
  variable: `struct K` used as `Array<K>` fails with
  `type error: Cannot access field on type 'K'`. Renaming to `Point` fixes it.
  The error points at the field access rather than the name collision.
  *(Still open, and the tip of a larger generics problem — see Still open.)*
* `string.byte(s, i)` ignores the index and returns the whole tail as an array
  of `LuaValue`: `string.byte("abc", 2)` gives
  `[LuaValue.Int(98), LuaValue.Int(99)]` instead of `98`.
* `float:floor()` returns `float`, not `int`, so `acc + f:floor()` is a type
  error against an `int` accumulator. Worth confirming this is intended.

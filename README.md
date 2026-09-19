# Lust

[lust-lang.dev](https://lust-lang.dev) · [Docs](https://lust-lang.dev/docs) · Embeddable, strongly typed Lua-style scripting

Lust is a strongly typed, Lua-inspired scripting language implemented in Rust. It targets embedding scenarios while staying fast with a hybrid collector and a trace-based JIT.

## Features
- Strong static type system with ergonomic enum pattern matching via the `is` operator.
- High-performance runtime that pairs reference counting with a fallback mark-and-sweep pass for long-lived cycles.
- Trace-based JIT powered by `dynasm-rs`, emitting x64 machine code similar in function to LuaJIT.
- Friendly embedding surface for Rust and C, including typed value conversions and module loaders.
- Batteries-included tooling: bytecode compiler, VM, CLI runner, and optional WebAssembly build.

## Quick Start

Add the crate (renamed for ergonomic imports):

```bash
cargo add lust-rs --rename lust
```

Install the CLI:

```bash
cargo install lust-rs
lust --help
lust pkg add example-package
lust pkg remove example-package
lust pkg login
lust pkg publish
lust pkg logout
```

## Embedding in Rust

```rust
use lust::EmbeddedProgram;

fn main() -> lust::Result<()> {
    let mut program = EmbeddedProgram::builder()
        .module("main", r#"
            function greet(name: string): string
                return "hi, " .. name
            end
        "#)
        .entry_module("main")
        .compile()?;

    let greeting: String = program.call_typed("main.greet", "Lust")?;
    println!("{greeting}");
    Ok(())
}
```

If you register native APIs with export metadata (via `VM::register_exported_native` / `VM::record_exported_native`, or the embedding helpers like `EmbeddedProgram::register_typed_native`),
you can write Lust-readable extern stubs to disk from your embedder:

```rust
let _ = program.dump_externs_to_dir("externs");
```

The `is` operator tests and binds patterns:

```lust
if status is Complete(value) then
    print("done(" .. value .. ")")
end
```

Fallible extraction from `unknown` uses `as` and returns `Option<T>`. Casts and
patterns associate left-to-right, so no parentheses are required:

```lust
if value as int is Some(x) and x > 0 then
    print(x)
end
```

Array bracket reads are non-trapping and return `Result<T, IndexError>`.
`IndexError` exposes the attempted `index` and current `length`:

```lust
local values: Array<string> = ["first", "second"]

if values[2] is Ok(value) then
    print(value)
elseif values[2] is Err(error) then
    print("index " .. error.index .. " exceeds length " .. error.length)
end
```

Use `values:get(index)` for `Option<T>` when the bounds details do not matter.
Use `values[index]:unwrap()` only when the index is known to be valid and a
runtime error is intentional if that invariant is broken. Indexed assignment
still requires an existing element and raises a runtime error when out of bounds.

Lust's casing convention distinguishes language roles:

- Lowercase names are primitives and keywords: `int`, `string`, `unknown`.
- PascalCase names are nominal types and enum variants: `Option`, `Result`,
  `Array`, `Some`, `None`, and user-defined types.
- snake_case names are variables, functions, methods, fields, and modules.

This keeps `Option` visibly distinct from primitives; `OPTION` is reserved by
convention for constant-like names rather than types.

## Generics and traits

Lust generics are statically checked and runtime-erased. A generic function or
method has one bytecode body, while the typechecker infers and substitutes a
fresh set of type arguments at each call. The tracing JIT can still specialize
that body for the concrete values observed at runtime.

Type parameters are declaration-scoped identifiers, not a capitalization
heuristic. `T`, `Item`, and `value_type` are generic only inside a declaration
that introduces them; a nominal type named `K` remains a normal type everywhere
else. Multi-letter PascalCase parameter names are recommended for readability.

```lust
function identity<Item>(value: Item): Item
    return value
end

function first<Item>(values: Array<Item>): Item
    return values[0]:unwrap()
end

local number: int = identity(42)
local text: string = identity<string>("hello")
local first_number: int = first([1, 2, 3])
```

Inference is recursive and consistent. For example, `Array<Item>` infers from
`Array<int>`, while two parameters declared as the same `Item` must receive the
same type. Explicit arguments use `function_name<Type>(...)` or
`value:method<Type>(...)` and are required when a parameter cannot be inferred.
The `<` must directly follow the function name; spaced `a < b` remains a
comparison.

Generic structs, enums, functions, instance methods, and universal
impls are supported:

```lust
struct Box<Item>
    value: Item
end

impl<Item> Box<Item>
    function get(self): Item
        return self.value
    end
end
```

Traits are nominal contracts and keep the Rust-inspired `trait`, `impl Trait for
Type`, and `Item: Trait` spelling. A type conforms only through an explicit
`impl`. A bare trait name is also the dynamic constraint type, so no `dyn` or
`interface` keyword is needed:

```lust
trait Drawable
    function draw(self): string
end

function render<Item: Drawable>(item: Item): string
    return item:draw()
end

function render_dynamic(item: Drawable): string
    return item:draw()
end
```

Bounds are enforced at call and construction sites. Trait methods currently
require one unannotated leading `self`, and method names form one namespace for
each erased runtime type; conflicting implementations are rejected.

Runtime erasure deliberately imposes several current restrictions:

- Generic traits, generic trait methods, and default trait methods are rejected.
- Conditional impls such as `impl<Item: Trait> Box<Item>` are rejected.
- Specialized impls such as `impl Box<int>` are rejected; use one universal
  `impl<Item> Box<Item>`.
- `is` and `as` cannot target an erased type parameter.
- Generic arguments cannot be revalidated at a raw VM/native boundary because
  values retain their erased nominal runtime type. Validate them in typed Lust
  code or in the embedding conversion layer.

These forms fail with explicit type errors rather than being accepted and
silently ignored. Generic traits and richer trait dispatch remain future
language work, not implied current behavior.

## JIT activation

The tracing JIT (x86_64 and aarch64 backends; an rv32 backend exists for
riscv32 targets) profiles backward bytecode jumps and starts recording an
ordinary hot loop after five observed backedges. A compiled trace stays cached
across its normal exits (the loop condition failing, a nested loop being
entered); only a guard that fails on something the trace assumed evicts it,
after which the site is retried with bounded exponential backoff. A nested
loop gets its own root trace, which the outer loop's trace calls directly, so
loop nests run natively end to end.

Straight-line Lust functions and struct methods are inlined into the trace;
other calls are opaque operations inside it, so a hot loop stays native while a
branch-heavy callee executes through the interpreter.

Loop-free functions are also compiled whole after thirty calls: their bytecode
is translated statically, every branch included, and the code calls other
compiled functions natively — frames on the machine stack, arguments and
results copied directly — so direct and mutual recursion run native end to
end. A function using something the function compiler does not handle yet
(loops, field access, arrays, strings, closures, native calls) keeps running
in the interpreter, with its loops traced as before; a compiled caller hands
such a call, or one that would exhaust the native stack, back to the
interpreter at the call instruction. Exits from any depth of native calls
turn the native frames into interpreter frames first, so errors and stack
traces look the same either way.

Environment switches, read once at VM creation:

- `LUST_JIT=0` disables the JIT entirely (the interpreter is the reference
  semantics; use this to check a result against it).
- `LUST_JIT_NOPIN=1` keeps the JIT but disables register pinning on aarch64,
  for bisecting and benchmarking.
- `LUST_JIT_NOFN=1` keeps loop traces but disables whole-function
  compilation.

Two tools check the JIT against the interpreter, and should be run on every
backend change:

- `examples/jit_diff.py` runs every program under `examples/` with `LUST_JIT=0`
  and with the JIT and diffs the outputs.
- `cargo build --release -p lust-fuzz` builds a seeded random-program
  differential fuzzer (a workspace member, so it is not built by a plain
  `cargo build`). `lust-fuzz run --cases 20000 --size 4 --jobs 6 --keep-going`
  generates programs, runs each both ways in-process and reports every
  disagreement with a shrunk reproducer; `lust-fuzz one --seed S` prints a
  program and `lust-fuzz replay --seed S` reruns it with timings. `--size`
  scales program length, `--fg` runs the workers at normal priority (they
  default to background QoS), and a watchdog kills cases over 120 s or 2 GB.

Embedders can inspect cumulative activation counters after calling Lust code:

```rust
let stats = program.jit_stats();
println!("compiled roots: {}", stats.root_traces_compiled);
println!("native entries: {}", stats.native_trace_entries);
println!("recursive calls: {}", stats.recursive_calls);
```

`VM::jit_stats()` exposes the same `JitStats` value. Loading a replacement
function table invalidates compiled traces, profiles, retries, and counters so
machine code cannot retain stale function indexes.

## Embedding in C (WIP)

The crate ships with a C header at `include/lust_ffi.h` exposing a minimal ABI so native hosts
can compile and call Lust code. Build the shared library with
`cargo build --release --lib` and link against `liblust`:

```c
#include "lust_ffi.h"

int main(void) {
    EmbeddedBuilder *builder = lust_builder_new();
    lust_builder_add_module(builder, "main", "function answer(): int\n    return 42\nend\n");
    lust_builder_set_entry_module(builder, "main");
    EmbeddedProgram *program = lust_builder_compile(builder);

    LustFfiValue result = {0};
    lust_program_call(program, "main.answer", NULL, 0, &result);
    /* ... */
}
```

A complete example lives in `examples/c-ffi`.


## Things considered more in the stable territory:
 - std interpreter
 - std JIT (Many optimizations are WIP, error cases are rare under normal use)
 - no_std interpreter

## Other things WIP:
 - tree-sitter (Missing a few highlight scenarios)
 - vsc-extension (Haven't touched this in a while)
 - lust-analyzer (Still useful, but missing a lot of useful errors)
 - Package system

The language is still heavily WIP in general, absolute stability is not guaranteed.

# License

Lust is dual-licensed under either:

* [MIT License](LICENSE-MIT)
* [Apache License, Version 2.0](LICENSE-APACHE)

at your option.

# Lust

[lust-lang.dev](https://lust-lang.dev) · [Docs](https://lust-lang.dev/docs) · Embeddable, strongly typed Lua-style scripting

Lust is a strongly typed, Lua-inspired scripting language implemented in Rust. It targets embedding scenarios while staying fast with a hybrid collector and a trace-based JIT.

## Features
- Strong static type system with ergonomic enum pattern matching via the `is` helper.
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
            pub function greet(name: string): string
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

The `is` helper works intuitively:

```lust
if status is Complete(value) then
    print("done(" .. value .. ")")
end
```

## Embedding in C (WIP)

The crate ships with a C header at `include/lust_ffi.h` exposing a minimal ABI so native hosts
can compile and call Lust code. Build the shared library with
`cargo build --release --lib` and link against `liblust`:

```c
#include "lust_ffi.h"

int main(void) {
    EmbeddedBuilder *builder = lust_builder_new();
    lust_builder_add_module(builder, "main", "pub function answer(): int\n    return 42\nend\n");
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
License for the language/interpreter is currently PolyForm Noncommercial License 1.0.0.  Once I am comfortable with the state of the language core it will be moved to MIT/GPLv3 (Undecided).

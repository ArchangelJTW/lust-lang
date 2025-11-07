# Rust Extension Example

This example shows how to pair a Lust script with local Rust extension crates that
register native functions **and** Lust-visible types at runtime. The CLI inspects those
bindings and generates extern stubs automatically.

```
examples/rust-extension/
├── extensions/
│   ├── double/
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   └── triple/
│       ├── Cargo.toml
│       └── src/lib.rs
├── lust-config.toml
└── main.lust
```

The `lust-config.toml` declares the Rust extension crates via the new dependency table:

```toml
[dependencies]
lust-double = { path = "extensions/double", kind = "rust" }
lust-triple = { path = "extensions/triple", kind = "rust" }
```

## Running the example

1. From the project root, ask the CLI to build the extension and emit extern stubs:
   ```bash
   lust --dump-externs examples/rust-extension/main.lust
   ```
   The CLI compiles each extension crate, executes its `lust_extension_register` hook in a
   temporary VM, and captures every struct/enum/function it exposes through `ExternRegistry`
   and `register_exported_native`. Generated stubs land under
   `examples/rust-extension/externs/<crate_name>/`.

2. Execute the Lust program:
   ```bash
   lust examples/rust-extension/main.lust
   ```
The runtime loads the compiled shared libraries, invokes their register hooks, and the
script can construct `Factor` structs, invoke `Factor:apply`, and work with the
`Operation` enum just like native Lust definitions. The Lust source references the
bindings via the sanitized crate prefix (for example `use lust_double.*`), regardless of
whether the stubs come from the working tree, the package cache, or the generated
`externs/` folder. Resolution checks the project sources first, then the package cache,
and finally the `externs/` directory—so local edits automatically override cached
artifacts.

## Extension crate

- Each crate exposes the required `#[no_mangle] extern "C" fn lust_extension_register`
  entrypoint. Inside that hook it builds an `ExternRegistry`, declares Rust-driven
  structs/enums/functions, and then registers native implementations via
  `vm.register_exported_native`.
- The crates are built as `cdylib`s and depend on the root `lust-lang` crate (with the
  `packages` feature) to access runtime types and helpers.

### `lust-double`

- Declares a `Factor` struct (with `base` and `multiplier` fields) and a helper function
  `make_factor` that constructs instances from Rust.
- Provides a method-like native `Factor:apply` plus the classic `host_double` /
  `host_quadruple` helpers.

### `lust-triple`

- Declares an `Operation` enum with `Double`, `Triple`, and `Scale(int)` variants, along
  with helpers to build and apply operations.
- Demonstrates how native code can emit enum payloads for Lust to consume.

The updated `main.lust` script shows how to consume all of these bindings after running
`lust --dump-externs ...`.

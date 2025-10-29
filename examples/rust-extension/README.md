# Rust Extension Example

This example shows how to pair a Lust script with a local Rust extension crate that
registers native functions at runtime and lets the CLI generate extern stubs automatically.

```
examples/rust-extension/
├── extensions/
│   └── double/
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs
├── lust-config.toml
└── main.lust
```

## Running the example

1. From the project root, ask the CLI to build the extension and emit extern stubs:
   ```bash
   lust --dump-externs examples/rust-extension/main.lust
   ```
   The CLI compiles `extensions/double` with Cargo, runs its register hook in a temporary VM,
   and synthesises extern blocks from the metadata it records. Generated stubs land under
   `examples/rust-extension/externs/`.

2. Execute the Lust program:
   ```bash
   lust examples/rust-extension/main.lust
   ```
   The runtime loads the compiled shared library, invokes its
   `lust_extension_register` hook, and the script can call `host_double`
   (via `use externs.lust_double.*`) as if it were a regular Lust function.

## Extension crate

- `src/lib.rs` exposes the required `#[no_mangle] extern "C" fn lust_extension_register`
  entrypoint. It receives the VM pointer and uses `vm.register_exported_native` to bind
  `host_double`, which the loader automatically maps to
  `externs.lust_double.host_double` while recording its signature for stub generation.
- The crate is built as a `cdylib` and depends on the root `lust-lang` crate with the
  `packages` feature enabled so it can access the runtime types.

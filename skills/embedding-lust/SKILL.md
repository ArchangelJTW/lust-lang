---
name: embedding-lust
description: Use this skill when embedding the Lust scripting language into a Rust host program, covering in-memory compilation with EmbeddedProgram, calling Lust from Rust, exposing Rust functions and types to Lust, value conversion, async tasks, gas/memory budgets, JIT inspection, and no_std or C embedding paths.
---

# Embedding Lust into Rust

Lust is a strongly typed, Lua-inspired scripting language implemented in Rust. It is designed to be embedded: a Rust host compiles Lust source in memory, runs it on a VM, exchanges typed values in both directions, and can expose Rust functions and types to scripts. This skill maps the embedding architecture and gives working recipes.

For the Lust language syntax and standard library itself, see the `coding-lust` skill. Reference examples live in the repository: `examples/embed.rs` (the canonical tour), `examples/rust-extension/`, `examples/lua_c_api/`, and `examples/c-ffi/`.

---

## 1. Architecture: what happens when you embed

The embedding pipeline is: **source → ModuleLoader → TypeChecker → Compiler → VM**.

Two public surfaces exist:

| Surface | Feature gate | Use when |
|---|---|---|
| `EmbeddedProgram` (+ `EmbeddedBuilder`) | `std` | The normal case. In-memory modules, typed calls, natives, async, budgets. Wraps a `VM` and keeps type signatures/struct/enum metadata. |
| `VM` directly | always (incl. `no_std`) | You drive compilation yourself (e.g. `compile_program_with_config`) or need raw bytecode-level control. |

Key runtime facts:

- All Lust values are `lust::bytecode::Value` (`Nil`, `Bool`, `Int`, `Float`, `String`, `Array`, `Tuple`, `Map`, `Struct`, `Enum`, `Function`, `NativeFunction`, `Closure`, `Iterator`, `Task`).
- Functions are named by dotted module path: a function `greet` in module `main` is `main.greet`. Top-level statements of the **entry** module compile into a `__script` function; top-level statements of **non-entry** modules become `__init@<module>` functions that run automatically at compile time.
- The VM uses `Rc`/`RefCell` internally, so `EmbeddedProgram` is **not `Send`/`Sync`**. Keep a program on one thread.
- Number types: `lust::LustInt` is `i64` under `std` (`i32` in `no_std`), `lust::LustFloat` is `f64`/`f32`. Prefer the aliases over hardcoded `i64`/`f64`.

---

## 2. Cargo setup

```bash
cargo add lust-rs --rename lust
```

Features (from the crate's `Cargo.toml`):

- `default = ["std", "packages"]` — what most embedders want.
- `std` — required for the `embed` module (`EmbeddedProgram`), JIT, module loader from disk.
- `packages` — package manager plus dynamic loading of Rust extension crates (`libloading`, `dirs`, `serde_json`, `object`).
- `lua_transpile` — Lua 5.1 transpilation via `full_moon`.
- `rv32` — JIT for riscv32 targets (implies `std`).

For `no_std` targets use `default-features = false`; the interpreter, `VM`, `LustConfig`, `EmbeddedModule`, `load_program_from_embedded`, and `compile_program_with_config` remain available (see section 12).

---

## 3. Quick start

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

Rules of thumb:

- Every module registered via `.module(path, source)` is an in-memory source override; `use` statements between modules resolve against these paths.
- `.entry_module("main")` is required by `compile()` and must be one of the registered modules.
- `call_typed` takes `&mut self` and needs the fully qualified dotted name (`"main.greet"`).

---

## 4. Building a program: `EmbeddedBuilder`

All builder methods take `self` and return `Self` (chainable); `add_module`/`set_entry_module`/`set_config` are `&mut self` variants.

| Method | Purpose |
|---|---|
| `.module(path, source)` / `.add_module(&mut ...)` | Register an in-memory module (dotted path, e.g. `"lib.math"`). |
| `.entry_module(path)` / `.set_entry_module(&mut ...)` | Set the entry module (required before `compile()`). |
| `.with_base_dir(dir)` / `.set_base_dir(dir)` | Virtual base directory used for module path → file mapping (default `__embedded__`). |
| `.enable_stdlib_module("io")` | Enable an optional stdlib module (`io`, `os`), like `stdlib_modules` in `lust-config.toml`. |
| `.with_config(LustConfig)` / `.set_config(&mut ...)` | Supply a full config (JIT on/off, enabled modules, low-memory flags). |
| `.low_memory_mode()` | Skip storing expression/variable type info during compilation (ESP32-class targets). |
| `.minimal_runtime_types()` | Strip register type info from compiled functions. |
| `.declare_struct/enum/trait/impl/function(def)` | Register Rust-declared Lust types/functions (section 9). |
| `.with_extern_registry(ExternRegistry)` / `.extern_registry_mut()` | Bulk-register or mutate an `ExternRegistry`. |
| `.compile() -> Result<EmbeddedProgram>` | Typecheck + compile + build the VM. Non-entry module initializers run here. |

A multi-module example:

```rust
let mut program = EmbeddedProgram::builder()
    .module("main", r#"
        use lib.math as math

        function main(): int
            return math.add(1, 2)
        end
    "#)
    .module("lib.math", r#"
        function add(a: int, b: int): int
            return a + b
        end
    "#)
    .enable_stdlib_module("io")
    .entry_module("main")
    .compile()?;

program.run_entry_script()?; // runs the entry module's top-level statements
```

`run_entry_script()` executes `__script` (the entry module's top-level statements). It errors if the script returns a non-unit value. If the entry module has no top-level statements, calling it is unnecessary.

---

## 5. Calling Lust from Rust

### Typed calls

`call_typed<Args, R>(name, args)` validates the call against the typechecker's signature before running:

```rust
let sum: i64 = program.call_typed("main.add", (2_i64, 3_i64))?;   // tuple = multiple args
let hi: String = program.call_typed("main.greet", "Lust")?;       // single arg, passed directly
let nothing: () = program.call_typed("main.tick", ())?;           // zero args
```

- `Args` is either a single value implementing `IntoLustValue`, a tuple of up to 5 values, or `()` for no arguments.
- `R` implements `FromLustValue`. A `TypeError` is returned if argument or return types don't match the Lust signature.
- Names must match the signature key exactly (dotted form, e.g. `"main.add"`). Use `program.typed_functions()` to list available names and signatures.

### Raw calls

```rust
use lust::bytecode::Value;

let result: Value = program.call_raw("main.add", vec![Value::Int(2), Value::Int(3)])?;
assert_eq!(result.as_int(), Some(5));
```

`call_raw` skips signature validation (useful for `unknown`-typed or dynamic functions). `Value` accessors: `as_int`, `as_float`, `as_string`, `as_bool`, `as_enum` (→ `Some((enum_name, variant, payload))`), `struct_get_field`, `type_of`.

### Function handles

Hold a callable and invoke it later, including Lust closures:

```rust
use lust::{FunctionHandle, StructInstance};

let handle: FunctionHandle = program.function_handle("main.translate")?;
let result: StructInstance = handle.call_typed(&mut program, (point, 2_i64, 5_i64))?;
let raw = handle.call_raw(&mut program, vec![Value::Int(1)])?;

// Signature introspection / validation against the typechecker metadata:
if let Some((name, signature)) = handle.signature(&program) { /* ... */ }
```

### Globals

```rust
// Read (returns Option; None if unset). Types must line up with the declared global.
let scale: Option<i64> = program.get_typed_global::<i64>("main.SCALE_FACTOR")?;

// Write (accepts anything implementing IntoTypedValue; overwrites the global).
program.set_global_value("main.SCALE_FACTOR", 5_i64);

// Raw + enumeration.
let value = program.get_global_value("main.SCALE_FACTOR");
let names = program.global_names();
let snapshot = program.globals(); // Vec<(String, Value)>
```

Global names accept either `"main.SCALE_FACTOR"` or `"main::SCALE_FACTOR"` (normalized internally).

### Creating Lust values from Rust

```rust
use lust::{enum_variant, enum_variant_with, struct_field, struct_instance, StructInstance};

// struct: field order doesn't matter, but every declared field must be supplied
let point: StructInstance = program.struct_instance(
    "main.Point",
    [
        struct_field("x", 3_i64),
        struct_field("y", 4_i64),
        struct_field("name", "FirstPoint"),
    ],
)?;

// enum: unit variant, then variant with payload (types checked against the definition)
let pending = program.enum_variant("main.Status", "Pending")?;
let complete = program.enum_variant_with("main.Status", "Complete", vec![4_i64])?;
```

Struct/enum names must be known to the program (defined in Lust source or declared via the extern registry).

---

## 6. Value conversion

### The conversion traits

| Trait | Role |
|---|---|
| `IntoLustValue` | Rust → Lust (`into_value()`, `matches_lust_type(&Type)`, `type_description()`). |
| `FromLustValue` | Lust → Rust (`from_value(Value)`, same introspection methods). |
| `FunctionArgs` | Argument bundles for `call_typed` (single value, `()`, tuples ≤ 5). |
| `FromLustArgs` | Argument bundles for natives (single value, `()`, tuples ≤ 5). |
| `IntoTypedValue` | Rust → a type-checked value for globals/struct fields. |
| `FromStructField<'a>` | Field extraction for `LustStructView` derives. |

Built-in mappings:

| Lust | → Rust (`FromLustValue`) | ← Rust (`IntoLustValue`) |
|---|---|---|
| `int` | `LustInt` (`i64` under std) | `LustInt` |
| `float` | `LustFloat` (`f64` under std) | `LustFloat` |
| `string` | `String`, `Rc<String>` | `&str`, `&String`, `String`, `Rc<String>` |
| `bool` | `bool` | `bool` |
| unit `()` | `()` | `()` |
| `Array<T>` | `Vec<T>` (converts each element), `ArrayHandle` | `Vec<T>`, `ArrayHandle` |
| `Map<K, V>` | `MapHandle` | `MapHandle` |
| struct | `StructInstance`, `StructHandle` | `StructInstance`, `StructHandle` |
| enum | `EnumInstance` | `EnumInstance` |
| function | `FunctionHandle` | `FunctionHandle` |
| `unknown` / anything | `Value` (always matches) | `Value` |

Notes:

- There is **no direct `HashMap` conversion**; convert through `MapHandle` (section 7).
- `Value`/`ValueRef` always convert (they match any Lust type), which is how you handle `unknown`.
- Implement the traits on your own types to erase the boundary (example below).

### Custom conversion example

```rust
use lust::{FromLustValue, IntoLustValue};
use lust::ast::Type;
use lust::bytecode::Value;

struct UserId(i64);

impl FromLustValue for UserId {
    fn from_value(value: Value) -> lust::Result<Self> {
        Ok(UserId(i64::from_value(value)?))
    }
    fn matches_lust_type(ty: &Type) -> bool { i64::matches_lust_type(ty) }
    fn type_description() -> &'static str { "int" }
}

impl IntoLustValue for UserId {
    fn into_value(self) -> Value { self.0.into_value() }
    fn matches_lust_type(ty: &Type) -> bool { i64::matches_lust_type(ty) }
    fn type_description() -> &'static str { "int" }
}
```

---

## 7. Handles and zero-copy borrows

Handles share the VM's underlying storage (`Rc<RefCell<…>>`), so mutation through a handle is visible to scripts and vice versa. Dropping/clone is cheap; there is no copy of element data.

### `StructInstance` / `StructHandle`

```rust
use lust::bytecode::Value;
use lust::{StructHandle, StructInstance};

let point = program.struct_instance("main.Point", [ /* fields */ ])?;

let x: i64 = point.field::<i64>("x")?;              // typed read (clones the value)
let name_ref = point.borrow_field("name")?;          // zero-copy borrow -> ValueRef
println!("{}", name_ref.as_string().unwrap_or(""));

point.set_field("x", 8_i64)?;                        // type-checked write
point.update_field("y", |value| match value {        // read-modify-write
    Value::Int(current) => Ok(current + 5),
    other => Err(lust::LustError::RuntimeError {
        message: format!("expected int but saw {other:?}"),
    }),
})?;

let handle: StructHandle = point.to_handle();        // shareable view of the same instance
handle.set_field("x", 1_i64)?;                       // mutation visible through `point` too
handle.ensure_type("main.Point")?;                   // nominal type check
```

Pass `StructInstance`/`StructHandle` **into** calls as arguments (they implement `IntoLustValue`) and receive them as return values.

### `ArrayHandle`

```rust
use lust::{ArrayHandle, Value};

if let Some(array) = program.get_typed_global::<ArrayHandle>("main.arr_global")? {
    array.push(Value::Int(4));                        // mutates the script's array
    let snapshot = array.with_ref(|values| {          // closure over &[Value]
        values.iter().map(|v| v.as_int().unwrap()).collect::<Vec<_>>()
    });
    let _ = array.with_mut(|values: &mut Vec<Value>| values.insert(0, Value::Int(0)));
    let _len = array.len();
    let _first = array.get(0);                        // Option<ValueRef<'_>>
}
```

### `MapHandle`

```rust
use lust::{MapHandle, Value};

if let Some(map) = program.get_typed_global::<MapHandle>("main.map_global")? {
    map.insert("three", Value::Int(3));
    if map.contains_key("one") {
        let _v = map.get("one");                      // Option<ValueRef<'_>>
    }
    let _removed = map.remove("two");
    let _snapshot = map.with_ref(|view| view.len());
}
```

### `EnumInstance`

```rust
use lust::EnumInstance;

fn describe(status: EnumInstance) -> lust::Result<String> {
    match status.variant() {
        "Pending" => Ok("pending".into()),
        "Complete" => Ok(format!("done({})", status.payload::<i64>(0)?)),
        other => Err(lust::LustError::RuntimeError {
            message: format!("unexpected variant {other}"),
        }),
    }
}
```

### `ValueRef` and `StringRef`

`borrow_field`/`ArrayHandle::get`/`MapHandle::get` return `ValueRef<'a>` — either a borrow into the VM storage or an owned value (weak `ref` fields are materialized). Convert with `as_int()`, `as_string()`, `as_bool()`, `as_float()`, `as_rc_string()`, `as_array_handle()`, `as_map_handle()`, `as_struct_handle()`, `to_owned()`, `into_owned()`.

`StringRef<'a>` derefs to `str` (zero-copy) and can produce an `Rc<String>` via `as_rc()`.

### `LustStructView` derive (zero-copy struct views)

Derive a typed, borrowed view of a Lust struct — ideal for hot paths where cloning fields would be wasteful:

```rust
use lust::embed::LustStructView as _;      // trait (from_handle)
use lust::{LustStructView, StringRef, StructHandle}; // derive macro is at the crate root

#[derive(LustStructView)]
#[lust(type = "main.Point")]               // fully qualified Lust struct name
struct PointView<'a> {                     // must declare the lifetime (default 'a)
    #[lust(field = "x")]
    x: lust::LustInt,
    #[lust(field = "y")]
    y: lust::LustInt,
    #[lust(field = "name")]
    name: StringRef<'a>,
}

let handle: StructHandle = program.get_typed_global::<StructHandle>("main.lust_point")?.unwrap();
let view = PointView::from_handle(&handle)?;   // Err(TypeError) if the type name mismatches
println!("{}, {}, {}", view.x, view.y, view.name.as_str());
```

Field types must implement `FromStructField`: `ValueRef<'a>`, `StringRef<'a>`, `LustInt`, `LustFloat`, `bool`, `Rc<String>`, `StructHandle`, `StructInstance`, `ArrayHandle`, `MapHandle`, `FunctionHandle`, `EnumInstance`, `Value`, and `Option<T>` of any of these (`None` for `Nil`). Derive attributes: `#[lust(type = "...")]` (or `struct =`) on the container, `#[lust(field = "...")]` on fields, plus optional `#[lust(crate = "::lust")]` and `#[lust(lifetime = "'a")]`.

---

## 8. Exposing Rust functions to Lust

Declare the native in Lust so the typechecker knows its signature, then register an implementation:

```rust
// In the Lust module source:
//
//   extern
//       function host_scale(int): int
//   end

program.register_typed_native("host_scale", |value: i64| -> std::result::Result<i64, String> {
    Ok(value * 10)
})?;
```

- The name may be short (`"host_scale"`) if unambiguous, or fully qualified (`"main.host_scale"`).
- The closure's argument type is a single `FromLustValue` type, a tuple (≤ 5), or `()`; the return is `Result<R, String>` where `R: IntoLustValue + FromLustValue`. A `String` error becomes a Lust runtime error.
- Argument and return types are checked against the declared Lust signature **at registration**; the returned value is re-checked at each call.
- `register_typed_native` also records a `NativeExport` (see `dump_externs_to_dir` below).

Low-level alternative (no signature validation, no required declaration):

```rust
use lust::bytecode::{NativeCallResult, Value};

program.register_native_fn("host_raw", |values: &[Value]| {
    let n = values.first().and_then(|v| v.as_int()).unwrap_or(0);
    Ok(NativeCallResult::Return(Value::Int(n * 2)))
});
```

`NativeCallResult` variants: `Return(Value)` (normal result), `Yield(Value)` (cooperative yield), `Stop(Value)`.

### Export metadata and stub generation

If you register natives with export metadata (done automatically by `register_typed_native`, or manually via `program.vm_mut().record_exported_native(NativeExport::new(name, params, return_type))` with `NativeExportParam::new(name, "int")` entries), you can emit Lust-readable `extern` stubs for tooling:

```rust
// Returns io::Result<Vec<PathBuf>> with the written stub files.
let written = program
    .dump_externs_to_dir("externs")
    .map_err(|err| lust::LustError::Unknown(format!("dump externs: {err}")))?;
```

This is how `lust --dump-externs` (used by the package system and the `examples/rust-extension` workflow) generates the `externs/` directory.

---

## 9. Declaring Lust types from Rust

Rust can define structs, enums, traits, impls, and function signatures that Lust code then uses like native definitions. This is the mechanism behind Rust extension crates.

```rust
use lust::ast::{Span, Type, TypeKind};
use lust::embed::native_types::EnumBuilder;   // note: EnumBuilder lives in embed::native_types
use lust::embed::{
    enum_variant, enum_variant_with, function_param, struct_field_decl, ExternRegistry,
    FunctionBuilder, ImplBuilder, StructBuilder, self_param, type_named,
};

fn ty_int() -> Type { Type::new(TypeKind::Int, Span::dummy()) }
fn ty_point() -> Type { type_named("main.Point") }

let mut registry = ExternRegistry::new();

registry.add_struct(
    StructBuilder::new("main.Point")
        .field(struct_field_decl("x", ty_int()))
        .field(struct_field_decl("y", ty_int()))
        .finish(),
);

registry.add_function(
    FunctionBuilder::new("main.make_point")
        .param(function_param("x", ty_int()))
        .return_type(ty_point())
        .finish(),
);

// Method-bearing impl (methods get registered as `Type:method`).
registry.add_impl(
    ImplBuilder::new(type_named("main.Point"))
        .method(
            FunctionBuilder::new("apply")
                .param(self_param(None))
                .param(function_param("value", ty_int()))
                .return_type(ty_int())
                .finish(),
        )
        .finish(),
);

registry.add_enum(
    EnumBuilder::new("main.Operation")
        .variant(enum_variant("Double"))
        .variant(enum_variant_with("Scale", [ty_int()]))
        .finish(),
);
```

Then either pass the registry to the builder:

```rust
let program = EmbeddedProgram::builder()
    .module("main", module_source)
    .with_extern_registry(registry)
    .entry_module("main")
    .compile()?;
```

…or register directly on a bare `VM` (extension-crate style):

```rust
registry.register_with_vm(&mut vm);
// or finer-grained:
registry.register_with_typechecker(&mut typechecker)?;
registry.register_struct_layouts(&mut vm);
registry.register_type_stubs(&mut vm);
registry.register_trait_impls(&mut vm);
```

Also available: `weak_struct_field_decl` (weak `ref` field, stored as `Option<T>`), `private_struct_field_decl`, `trait_bound`, `type_unit()`, `type_unknown()`, `TraitBuilder` + `TraitMethodBuilder`, and `registry.add_const(name, ty)`. Names are module-qualified (`"main.Point"`); inside extension crates they use the crate prefix (section 11). To construct these values from Rust, use `program.struct_instance` / `program.enum_variant_with`, or on a raw `VM`, `vm.instantiate_struct(name, fields)` and `Value::enum_variant(name, variant, payload)`.

---

## 10. Async natives and cooperative tasks

Lust has cooperative tasks (`task` module). Rust can plug asynchronous operations into them in four ways:

| API | Behavior in Lust | Rust side |
|---|---|---|
| `register_async_native(name, \|args: Vec<Value>\| -> Future<Output = Result<Value, String>>)` | Call **suspends the calling task**; resumes with the future's value. | Future polled by `AsyncDriver`. |
| `register_async_typed_native::<Args, R, _, _>(name, \|args\| -> Future<...>)` | Same, but args/result are type-checked against the declared signature. | Same. |
| `register_async_task_native::<Args, R, _, _>(name, ...)` | Call returns immediately with a `Task` handle the script can poll (`task.info`, `task.yield`, …). | Future runs in the background; result lands in the task. |
| `register_async_task_queue::<Args, R>(name, queue)` | Call returns a `Task`; the work item is pushed to a queue the host owns. | Host pops from `AsyncTaskQueue`, does the work, completes the job. |

Queue-driven example (host performs the "async" work):

```rust
use lust::{AsyncDriver, AsyncTaskQueue, FunctionHandle, LustInt};

let queue = AsyncTaskQueue::<FunctionHandle, LustInt>::new();
program.register_async_task_queue::<FunctionHandle, LustInt>("fetch_value", queue.clone())?;

// Lust: `extern function fetch_value(function(int)): Task` and a function that calls it.
let task_value = program.call_raw("main.get_async_value", Vec::new())?;

let pending = queue.pop().expect("a pending job");        // or pop_blocking()
let callback: FunctionHandle = pending.args().clone();    // the Lust callback
callback.call_typed::<_, ()>(&mut program, 77_i64)?;      // invoke back into Lust
pending.complete_ok(77_i64);                              // or complete_err("reason")

AsyncDriver::new(&mut program).pump_until_idle()?;        // poll futures, resume tasks
// AsyncDriver also has `.poll()` and `.has_pending()`;
// EmbeddedProgram exposes `.poll_async_tasks()` / `.has_pending_async_tasks()` directly.
```

Inspect the resulting task through the VM:

```rust
let vm = program.vm_mut();
let task = vm.get_task_instance(task_handle)?;   // .state, .last_result, .last_yield, .error
```

Constraints: only one pending async native per task at a time; async natives must be invoked from a running VM (a Lust call), not out of thin air.

---

## 11. Runtime controls: budgets, JIT, tooling

### Gas and memory budgets

```rust
program.set_gas_budget(100_000);          // per-run instruction budget
// exhausted -> LustError::RuntimeError "Out of gas (limit: …, used: …)"
let _ = program.gas_used();
let _ = program.gas_remaining();          // Option<u64>
program.reset_gas_counter();
program.clear_gas_budget();

program.set_memory_budget_bytes(1 << 20); // allocation accounting (also set_memory_budget_kb)
let _ = program.memory_used_bytes();
let _ = program.memory_remaining_bytes(); // Option<usize>
program.reset_memory_counter();
program.clear_memory_budget();
```

Budgets are ideal for sandboxing untrusted scripts.

### JIT inspection

The tracing JIT activates after five hot backedges. After calling Lust code:

```rust
let stats = program.jit_stats();          // also vm.jit_stats()
println!("roots: {}", stats.root_traces_compiled);
println!("native entries: {}", stats.native_trace_entries);
println!("recursive calls: {}", stats.recursive_calls);
```

Note: loading a replacement function table (e.g. re-registering natives wholesale) invalidates compiled traces.

### Introspection

```rust
for (name, sig) in program.typed_functions() { /* name like "main.add" */ }
let sig = program.signature("main.add");
let struct_def = program.struct_definition("main.Point");   // ast::StructDef
let enum_def = program.enum_definition("main.Status");      // ast::EnumDef
let _ = program.vm_mut();                                    // full raw VM access
```

---

## 12. File-based and `no_std` embedding

### Loading modules from disk

```rust
use lust::{ModuleLoader, LustConfig, VM};

let mut loader = ModuleLoader::new("examples/modules");
let program = loader.load_program_from_entry("examples/modules/main.lust")?; // Program
// Returns Result<Self, ConfigError>; fall back to defaults if you like.
let config = LustConfig::load_for_entry("examples/modules/main.lust")
    .unwrap_or_else(|_| LustConfig::default());
let mut vm = lust::compile_program_with_config(program, &config)?;
vm.call("__script", Vec::new())?;   // run entry module top-level statements
```

With this path you work directly against the `VM` (`call`, `get_global`, `set_global`, `register_native`, `instantiate_struct`, …) — the typed conveniences of `EmbeddedProgram` are not available. `ModuleLoader` also supports `set_source_override(path, source)` and `add_module_root(prefix, dir, root_module)` for custom module resolution.

### `no_std` (and constrained targets)

With `default-features = false`, build the module graph from in-memory sources (no filesystem):

```rust
use lust::{EmbeddedModule, LustConfig, compile_program_with_config, load_program_from_embedded};

let entries = [EmbeddedModule { module: "main", parent: None, source: Some(source) }];
let program = load_program_from_embedded(&entries, "main")?;

let mut config = LustConfig::default();
config.set_low_memory_mode(true);
config.set_minimal_runtime_types(true);
config.set_jit_enabled(false);          // JIT requires std

let mut vm = compile_program_with_config(program, &config)?;
vm.call("__script", Vec::new())?;
```

Under `no_std`, `LustInt` is `i32` and `LustFloat` is `f32` — use the aliases.

---

## 13. Rust extension crates and C embedding

### Dynamic Rust extensions (cdylib)

A crate can ship natives + types to a running Lust program (the host or the `lust` CLI loads it at runtime; requires the `packages` feature):

1. Build a `cdylib` crate depending on `lust-rs` (with `packages`).
2. Export the registration hook:

```rust
#[no_mangle]
pub extern "C" fn lust_extension_register(vm_ptr: *mut lust::VM) -> bool {
    if vm_ptr.is_null() { return false; }
    let vm = unsafe { &mut *vm_ptr };
    // ExternRegistry::new() + StructBuilder/EnumBuilder/FunctionBuilder …,
    // then registry.register_with_vm(vm), and:
    // vm.register_exported_native(NativeExport::new(...), |values: &[Value]| { ... })
    true
}
```

3. Declare it in `lust-config.toml` and generate stubs:

```toml
[dependencies]
lust-double = { path = "extensions/double", kind = "rust" }
```

```bash
lust --dump-externs examples/rust-extension/main.lust   # writes externs/<crate>/
lust examples/rust-extension/main.lust                  # runs with extensions loaded
```

See `examples/rust-extension/` for complete `double`/`triple` crates. Inside extension crates, names are prefixed with the sanitized crate name (e.g. `lust_double.Factor`).

### C hosts (FFI)

Build the shared library and link against it using the header at `include/lust_ffi.h`:

```bash
cargo build --release --lib
```

```c
EmbeddedBuilder *builder = lust_builder_new();
lust_builder_add_module(builder, "main", "function answer(): int\n    return 42\nend\n");
lust_builder_set_entry_module(builder, "main");
EmbeddedProgram *program = lust_builder_compile(builder);

LustFfiValue result = {0};
lust_program_call(program, "main.answer", NULL, 0, &result);
```

Supported C surface: builder create/add module/set entry/compile, `lust_program_run_entry`, `lust_program_call`, `lust_program_get_global`/`set_global`, error access via `lust_last_error_message`, and `LustFfiValue` (nil/bool/int/float/string). Full example in `examples/c-ffi/`.

---

## 14. Common pitfalls

1. **Unqualified names in `call_typed`** — signatures are keyed by dotted module paths (`"main.add"`), not `"add"`. List candidates with `program.typed_functions()`.
2. **Registering a native for an undeclared function** — `register_typed_native` must be able to resolve a Lust signature; declare the function in Lust (usually in an `extern` block) first. Use `register_native_fn` when you deliberately don't want signature checking.
3. **`()` for zero-arg natives and calls** — a no-arg call is `call_typed("main.tick", ())`, and a zero-arg native closure takes `|_: ()|`. Tuples only go up to 5 elements.
4. **Thread sharing** — the VM is not `Send`/`Sync`; move the whole `EmbeddedProgram` between threads instead of sharing it.
5. **Type mismatches surface as `LustError::TypeError`/`RuntimeError`** — both carry a human-readable `message`; match on the enum for structured handling.
6. **Weak `ref` fields** — a struct field declared `ref T` is stored as `Option<T>`; `set_field` accepts the inner type directly, and `borrow_field` materializes an owned `ValueRef` for it.
7. **Generics are runtime-erased** — raw VM/native boundaries cannot revalidate generic arguments; validate in typed Lust code or in your conversion layer.
8. **`call_typed` on an `unknown`-typed function fails** — the program has no signature for it; use `call_raw` or a `FunctionHandle`.
9. **Entry module must be registered** — `compile()` errors if `.entry_module(...)` names a module never passed to `.module(...)`.
10. **Async natives need a running VM** — register before running, but they only execute (and create tasks) when invoked from Lust code; completion requires pumping `AsyncDriver`.
11. **Budget errors are ordinary runtime errors** — "Out of gas …" / "Out of memory budget …" propagate out of `call`/`call_typed`; catch and reset counters between runs.
12. **`LustInt` width changes without `std`** — hardcoding `i64` breaks `no_std` builds; prefer `lust::LustInt`/`lust::LustFloat` aliases.

# RFC: Bounded JSON Calls Through the C FFI (Option A)

- **Feature Name:** `c_ffi_json_helper`
- **Start Date:** 2026-11-03
- **Target Components:** `include/lust_ffi.h`, `src/ffi.rs`
- **Status:** Proposal / RFC (Design proposal only; no code or runtime changes; Option B handles deferred)
- **Upstream Issue:** [ArchangelJTW/lust-lang#11](https://github.com/ArchangelJTW/lust-lang/issues/11)
- **Pinned Base Commit:** `f2f81fe71e3961af6d5d4b5c9658f9ce184094e3` (Lust 3.5.8)

---

## 1. Summary

This RFC proposes an additive, bounded convenience helper for the C foreign function interface (FFI): **Option A: JSON text decoded into a bounded subset of Lust values, one synchronous function call, and JSON encoding of its return value**.

All declarations and semantics specified here are **proposals for future implementation** and are **not present or available** in the pinned upstream release (`f2f81fe71e3961af6d5d4b5c9658f9ce184094e3`).

Key boundaries of this proposal:
1. **Preserves Existing Scalar ABI:** The existing scalar tags (`NIL`, `BOOL`, `INT`, `FLOAT`, `STRING`), structs, and `lust_program_call` symbols remain untouched and binary-compatible.
2. **Option A Scope:** Transports and converts structured data (nested arrays and string-keyed maps) across the C boundary via bounded JSON parsing and serialization, without exposing internal bytecode or runtime layout details.
3. **Option B Deferred:** Mutable opaque object handles (e.g. `LustFfiArray`, `LustFfiMap`) are explicitly **deferred to a separate future RFC**. Handles require independent lifecycle, ownership, mutation accounting, and stale-pointer policies.
4. **No Runtime or Evaluation Changes:** This proposal does not modify the current evaluator runtime, standard library, or existing evaluation harnesses.

---

## 2. Motivation and Current Source Facts

### 2.1 Current ABI State (`f2f81fe71e3961af6d5d4b5c9658f9ce184094e3`)

Inspection of the pinned upstream source confirms the following facts:
- **Scalar-Only ABI:** In `include/lust_ffi.h:12–28` and `src/ffi.rs:78–138`, `LustFfiValueTag` is strictly limited to primitive scalar kinds (`LUST_FFI_VALUE_NIL`, `BOOL`, `INT`, `FLOAT`, `STRING`).
- **Structured Values Rejected:** If a called Lust function returns an `Array`, `Map`, or `Struct`, `value_to_ffi` fails with `"Unsupported Lust value for FFI conversion"`.
- **No Bytecode Constructors in C:** The C ABI does not expose public functions to construct Lust bytecode containers (`Array`, `Map`, `Struct`) directly.
- **No Standard-Library JSON Parser:** Searching `src/vm/stdlib.rs` confirms that the registered standard library modules do not include a built-in JSON parser. While a caller can pass JSON text as an ordinary `LUST_FFI_VALUE_STRING`, the script runtime does not automatically parse it into structured values without host-provided glue.

### 2.2 Polyglot Embedding Use Case

Hosts embedding Lust from languages with C FFI bindings (such as Python, Go, C++, or Zig) frequently need to pass structured configuration or tabular records into a Lust function and receive structured results without authoring manual serialization wrappers in both host code and Lust script.

---

## 3. Proposed Public C Interface

The declarations below are proposed additions to `include/lust_ffi.h`, guarded by a caller compile-time macro `LUST_ENABLE_FFI_JSON_V1` and a new optional Cargo feature `ffi-json = ["std", "dep:serde_json"]`.

```c
#ifdef LUST_ENABLE_FFI_JSON_V1

typedef int32_t LustJsonStatusV1;

#define LUST_JSON_OK_V1               0
#define LUST_JSON_INVALID_ARGUMENT_V1  1
#define LUST_JSON_INVALID_JSON_V1      2
#define LUST_JSON_NUMBER_RANGE_V1      3
#define LUST_JSON_LIMIT_V1             4
#define LUST_JSON_TYPE_V1              5
#define LUST_JSON_CALL_V1              6
#define LUST_JSON_UNSUPPORTED_V1       7
#define LUST_JSON_CYCLE_V1             8
#define LUST_JSON_BORROW_V1            9
#define LUST_JSON_INTERNAL_V1         10

typedef struct LustJsonLimitsV1 {
    uint32_t struct_size;          /* Must be sizeof(LustJsonLimitsV1) */
    uint32_t reserved;             /* Must be 0 */
    uint64_t max_input_bytes;      /* Total input JSON byte limit */
    uint64_t max_output_bytes;     /* Total output JSON byte limit */
    uint64_t max_depth;            /* Container nesting depth ceiling */
    uint64_t max_nodes;            /* Max total nodes traversed */
    uint64_t max_args;             /* Max positional argument count */
    uint64_t max_bridge_bytes;     /* Max staging & bridge allocation bytes */
} LustJsonLimitsV1;

/**
 * Call a Lust function with a positional argument list encoded as a JSON array,
 * returning a detached JSON-encoded result.
 *
 * Parameters:
 *   program:              Non-null pointer to live EmbeddedProgram.
 *   function_name:        UTF-8 function symbol name (e.g. "module.fn").
 *   function_name_len:    Length of function_name in bytes (excluding NUL).
 *   json_args:            Length-delimited UTF-8 JSON array of positional arguments.
 *   json_args_len:        Length of json_args in bytes.
 *   limits:               Optional resource quotas (NULL selects documented defaults).
 *   out_json_result:      Output slot receiving newly allocated UTF-8 JSON buffer.
 *   out_json_result_len:  Output slot receiving length of out_json_result in bytes.
 *
 * Returns:
 *   LUST_JSON_OK_V1 (0) on success, or non-zero status code on failure.
 *   On success, *out_json_result must be freed exactly once using lust_string_free().
 *   On failure, output slots are set to NULL / 0.
 */
LustJsonStatusV1 lust_program_call_json_v1(
    EmbeddedProgram *program,
    const char *function_name,
    size_t function_name_len,
    const char *json_args,
    size_t json_args_len,
    const LustJsonLimitsV1 *limits,
    char **out_json_result,
    size_t *out_json_result_len
);

#endif /* LUST_ENABLE_FFI_JSON_V1 */
```

### 3.1 Parameter and Memory Invariants

1. **Length-Delimited UTF-8:** `function_name` and `json_args` are byte-length delimited. Neither requires a trailing NUL byte; raw interior NUL bytes are rejected as invalid JSON.
2. **Positional Arguments:** The outer JSON document in `json_args` must be an array (`[...]`). Its elements map positionally to the target function's parameters:
   - `[]` passes 0 arguments.
   - `[1, 2]` passes 2 integer arguments.
   - `[[1, 2]]` passes 1 argument (an Array).
3. **Explicit Limits:** If `limits` is non-NULL, `struct_size` must equal `sizeof(LustJsonLimitsV1)` and `reserved` must be 0. Unknown layouts or zero values are rejected with `LUST_JSON_INVALID_ARGUMENT_V1`.
4. **Ownership of Result:** The output buffer is owned by the caller upon `LUST_JSON_OK_V1` and must be released using the matching library's `lust_string_free(char *ptr)`.

---

## 4. JSON ↔ Lust Mapping (Normative Specification)

| JSON Input Token | Converted Lust Value | Lust Return Value → JSON Output |
| :--- | :--- | :--- |
| `null` | `Value::Nil` | `Value::Nil` → `null` |
| `true` / `false` | `Value::Bool` | `Value::Bool` → `true` / `false` |
| Integer token (no decimal point or exponent) | `Value::Int` (exact signed 64-bit) | `Value::Int` → exact decimal integer |
| Decimal / exponent token (`1.0`, `1e3`) | `Value::Float` (finite binary64 `f64`) | `Value::Float` → round-trip `f64` JSON number with decimal/exponent marker |
| String token | `Value::String` (valid UTF-8) | `Value::String` → escaped JSON string |
| Array element within positional list | `Value::Array` (recursively converted) | `Value::Array` → JSON array (`[...]`) |
| Object (`{...}`) | `Value::Map` (keys must be Strings) | `Value::Map` → JSON object (`{...}`) |

### 4.1 Boundary Constraints and Type Invariants

- **Numeric Boundaries:**
  - Integer tokens outside `[i64::MIN, i64::MAX]` return `LUST_JSON_NUMBER_RANGE_V1`. No silent conversion to floating-point is permitted.
  - Floating-point tokens must be finite binary64. Non-finite values (`NaN`, `Infinity`, `-Infinity`) return `LUST_JSON_NUMBER_RANGE_V1`. Overflow and underflow-to-zero are rejected.
  - Floating zero (`-0.0`) preserves its negative sign in float formatting; integer `-0` normalizes to `Int(0)`.
- **Object and Key Semantics:**
  - JSON objects map strictly to `Value::Map`. **No automatic or inferred named struct creation** is performed.
  - Duplicate keys in input JSON objects (including escape-equivalent keys) are strictly rejected with `LUST_JSON_INVALID_JSON_V1`.
  - Returned Maps with non-string keys produce `LUST_JSON_UNSUPPORTED_V1`.
- **Unsupported Types:**
  - `Struct`, `WeakStruct`, `Tuple`, `Enum`, `Function`, `NativeFunction`, `Closure`, `Iterator`, and `Task` are unsupported on both input and return boundaries. Attempting to pass or return these types returns `LUST_JSON_TYPE_V1` (for input signatures) or `LUST_JSON_UNSUPPORTED_V1` (for return values).
  - No heuristic `$type` envelope or automatic struct field deduction is allowed in v1.

---

## 5. Execution, Status Codes, and State Management

### 5.1 Step-by-Step Execution Pipeline

To prevent side-effects during malformed requests, execution follows a strict sequential pipeline:
1. **Pointer & Limits Preflight:** Validate non-null pointers, valid non-zero lengths, and matching `limits->struct_size`.
2. **Streaming Parse & Input Validation:** Parse `json_args` under memory/depth/node quotas, rejecting duplicate keys and non-array roots.
3. **Signature & Arity Verification:** Resolve the canonical function symbol and verify parameter count and parameter types against the supported JSON subset before calling into the VM.
4. **Argument Conversion:** Convert parsed JSON items into `Value` instances within bridge memory accounting.
5. **Synchronous Invocation:** Execute the function once (`vm.call_raw`).
6. **Return Traversal & Serialization:** Traverse returned `Value`, checking for cyclic structures, borrow conflicts, and supported kinds under output size and depth limits.
7. **Publication:** Allocate NUL-terminated C string, assign output pointers, and return `LUST_JSON_OK_V1`.

### 5.2 Status Code Meanings

| Status Constant | Code | Meaning and Trigger Conditions |
| :--- | :---: | :--- |
| `LUST_JSON_OK_V1` | `0` | Call completed successfully; detached JSON string published. |
| `LUST_JSON_INVALID_ARGUMENT_V1` | `1` | Null program, empty name/input, invalid name UTF-8, incompatible limits layout. |
| `LUST_JSON_INVALID_JSON_V1` | `2` | Syntax error, non-array root, invalid UTF-8, duplicate object key. |
| `LUST_JSON_NUMBER_RANGE_V1` | `3` | Integer overflow beyond `i64`, float overflow/underflow, `NaN`/`Infinity`. |
| `LUST_JSON_LIMIT_V1` | `4` | Exceeded `max_input_bytes`, `max_output_bytes`, `max_depth`, `max_nodes`, `max_args`, or `max_bridge_bytes`. |
| `LUST_JSON_TYPE_V1` | `5` | Wrong parameter arity, incompatible parameter type, or unsupported input schema. |
| `LUST_JSON_CALL_V1` | `6` | Missing function symbol, unhandled VM runtime error, gas/instruction exhaustion. |
| `LUST_JSON_UNSUPPORTED_V1` | `7` | Return value is an unsupported Lust kind (e.g. Struct, Tuple, Closure) or Map with non-string keys. |
| `LUST_JSON_CYCLE_V1` | `8` | Cyclic container reference detected during return serialization. |
| `LUST_JSON_BORROW_V1` | `9` | `try_borrow` failed during container traversal. |
| `LUST_JSON_INTERNAL_V1` | `10` | Caught Rust panic or unexpected internal bridge error. |

### 5.3 Diagnostic Messages and Side-Effect Rules

- **Thread-Local Error Message:** On non-zero status, a concise diagnostic string (at most 1024 bytes, NUL-terminated) is stored in thread-local storage, accessible via `lust_last_error_message()`.
- **Pre-execution Failures:** Any failure occurring in Steps 1–4 guarantees **zero VM side-effects** (globals and callbacks remain unaltered).
- **Post-invocation Failures:** Failures in Step 5 (VM runtime error) or Step 6 (serialization/cycle/limit error) are **non-transactional**: function side effects (global mutations, host I/O) may have already taken effect. No automatic rollback or retry is attempted.

---

## 6. Ownership, Thread Safety, and Resource Quotas

### 6.1 Ownership and Thread Constraints

- **Borrowed Inputs:** `json_args` and `limits` are borrowed synchronously for the duration of the call and never retained across call boundaries.
- **Detached Output:** The published result is an independent heap-allocated C string. It remains valid even if `EmbeddedProgram` is destroyed, and must be freed using `lust_string_free()`.
- **Single-Threaded Model:** Lust's `EmbeddedProgram` and internal values use `Rc<RefCell<...>>` and are strictly **not thread-safe (`!Send`, `!Sync`)**. All invocations for a single program instance must occur on its owning OS thread.

### 6.2 Default Resource Ceilings

When `limits == NULL`, the following hard ceilings apply:

| Limit Parameter | Default Value | Scope / Description |
| :--- | :--- | :--- |
| `max_input_bytes` | `1,048,576` (1 MiB) | Maximum total byte length of `json_args`. |
| `max_output_bytes` | `1,048,576` (1 MiB) | Maximum serialized output length excluding trailing NUL. |
| `max_depth` | `64` | Maximum container nesting depth for inputs and outputs. |
| `max_nodes` | `65,536` | Maximum total values and object keys traversed. |
| `max_args` | `256` | Maximum number of positional arguments in outer list. |
| `max_bridge_bytes` | `16,777,216` (16 MiB) | Peak bridge memory allocation requested during parsing and serialization. |

- **Cycle Detection:** Container serialization tracks active ancestor identities to detect cyclic structures and reject them with `LUST_JSON_CYCLE_V1`.
- **Not an RSS Ceiling:** `max_bridge_bytes` bounds bridge scratch memory only; it does not replace operating system process memory limits or VM heap quotas.

---

## 7. Future Upstream Implementation Areas

Future implementation work (outside the scope of this RFC) will target the following boundaries:

1. `include/lust_ffi.h`: Add feature-guarded `LustJsonStatusV1`, `LustJsonLimitsV1`, and `lust_program_call_json_v1` declarations.
2. `src/ffi.rs`: Add FFI bridge entrypoint, TLS error message recording, and CString publication.
3. `src/ffi/json.rs` (new private module): Implement bounded streaming JSON parser, token mapping, cycle-checking serializer, and memory ledger.
4. `Cargo.toml`: Add optional Cargo feature `ffi-json = ["std", "dep:serde_json"]`.
5. `examples/c-ffi-json/`: Add self-contained runnable C embedding example exercising structured calls and status checks.

---

## 8. Planned Acceptance Test Matrix (Requirements for Future PR)

| Test ID | Test Focus | Planned Acceptance Criteria |
| :--- | :--- | :--- |
| **A01** | Real C/C++ Embedding | C and C++ programs compile against `include/lust_ffi.h` and link against `liblust`. |
| **A02** | Backward Compatibility | Existing scalar FFI functions and `lust_program_call` work identically. |
| **A03** | Round-Trip Types | Correct mapping for null, bool, int, float, string, nested arrays, and string-keyed maps. |
| **A04** | Integer Boundary | Exact `i64` preserved; values outside `[i64::MIN, i64::MAX]` return `LUST_JSON_NUMBER_RANGE_V1`. |
| **A05** | Float Boundary | Finite `f64` values round-trip; non-finite floats, overflow, and underflow-to-zero rejected. |
| **A06** | Unicode & NUL Handling | Valid UTF-8 and escaped `\u0000` preserved; raw NUL or invalid UTF-8 rejected. |
| **A07** | Duplicate Keys & Syntax | Non-array root and duplicate keys rejected with `LUST_JSON_INVALID_JSON_V1`. |
| **A08** | Signature & Preflight | Wrong arity or mismatched parameter types fail before VM execution. |
| **A09** | Unsupported Types | Returning Struct, Tuple, or Enum returns `LUST_JSON_UNSUPPORTED_V1`. |
| **A10** | Quota Limits | Exceeding byte, depth, node, or bridge limits returns `LUST_JSON_LIMIT_V1`. |
| **A11** | Cycle & Borrow Safety | Cyclic containers return `LUST_JSON_CYCLE_V1`; conflicting borrows return `LUST_JSON_BORROW_V1`. |
| **A12** | Input Validation | Null pointers, empty names, or invalid struct sizes return `LUST_JSON_INVALID_ARGUMENT_V1`. |
| **A13** | Leak & Lifecycle Check | Output buffers freed with `lust_string_free` show no leaks under AddressSanitizer/Valgrind. |
| **A14** | Non-Transactional Errors | Side-effects documented for post-invocation errors; no duplicate function execution. |
| **A15** | Independent Programs | Separate programs on independent threads operate without crosstalk. |
| **A16** | VM Accounting Transfer | Converted inputs retained by VM are properly accounted in VM memory tracking. |
| **A17** | Panic Boundary | Catchable Rust panics converted to `LUST_JSON_INTERNAL_V1`; abort behavior documented. |
| **A18** | Reusability After Errors | Program remains valid for subsequent calls after recoverable parsing/type errors. |

---

## 9. Open Questions and Maintainer Approval Gates

1. **Map-Only Representation:** Does the maintainer group agree that JSON objects should strictly map to `Value::Map`, deferring typed struct/enum synthesis to a separate schema RFC?
2. **Ceiling Tuning:** Are the proposed default limits (1 MiB input/output, 64 depth, 65,536 nodes) appropriate for standard embedding workloads?
3. **Symbol Naming:** Is `lust_program_call_json_v1` the preferred naming convention for versioned FFI additions?
4. **Option B Deferral:** Confirm that mutable handle APIs (`LustFfiArray`, `LustFfiMap`) remain out-of-scope until Option A is evaluated and stabilized.

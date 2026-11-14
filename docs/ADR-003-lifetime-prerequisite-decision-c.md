# ADR-003: RFC §15 Decision C — Static Lifetime Model ('s) and Borrow Prerequisite

**Status:** ACCEPTED / DECIDED  
**Date:** 2026-11-14  
**Target:** `lust-rs` 3.5.8 (`repos/lust-lang-fork`, Wave 0 Track 0C `D-LIFE-ADR`)  
**Deciders:** Project Coordinator / Architecture Review  
**Consulted:** `RFC_FIRST_CLASS_DATA_FORMATS_AND_SCHEMAS.md` (§4, §15 Decision C), `DEFERRED_DEPENDENCY_TREE.md` (`D-LIFE-ADR`, `D-LIFE`), `AGENTS.md`  

---

## 1. Context and Problem Statement

`RFC_FIRST_CLASS_DATA_FORMATS_AND_SCHEMAS.md` (§4 and §15) defines the architecture for first-class data formats (JSON, JSONC, JSONL, BSON, YAML, TOON, CSV, MessagePack) and high-performance zero-allocation views (`DataView<'s>`).

Section 15 (line 942) explicitly mandates:
> *"Координатор должен формально принять или вернуть на доработку: ... (C) lifetime prerequisite и narrow zero-allocation claim; ..."*

Under RFC §4 and §15 line 935:
- **Concept:** Explicit scoped borrowing (`DataView<'s>`) allows zero-copy field reads directly over raw buffer slices without intermediate heap allocations.
- **Risk:** Without a static borrow/escape analysis pass in the Lust compiler typechecker, user-level code could leak a `DataView<'s>` past the lifetime of the underlying buffer (e.g. returning a view from a scope, storing it in a global/fiber/actor, or mutating the backing document while a view is held), resulting in use-after-free or memory corruption.
- **Prerequisite:** Static borrow and escape analysis in the typechecker (`D-LIFE`, scheduled for Wave 4 Track 4G) must precede any claim of general zero-allocation user-facing views.

---

## 2. Evaluation Options for Decision C

| Option | Architecture | Safety Guarantee | Performance / Complexity Trade-off |
|---|---|---|---|
| **Option C.1: Eager Owned Values (`DataValue`)** | All decoders deserialize directly into owned heap values (`Document`, `DataValue`). No borrowing views exposed to user script scopes. | 100% memory safe. No use-after-free risk. Zero compiler complexity. | Allocates on every decode. Cannot achieve zero-allocation parsing claims. |
| **Option C.2: Runtime Lease Tokens (`OwnerRegistry` / `SlotEntry`)** | Views hold dynamic generation/lease handles registered in `OwnerRegistry`. Runtime checks prevent access if buffer is dropped/mutated. | Safe at runtime (panics or returns `LeaseError`). | Dynamic check overhead on every view access. Memory overhead for registry. |
| **Option C.3: Compiler Static Lifetime Analysis (`'s` & Escape Analysis)** | The typechecker statically tracks the lifetime `'s` of `DataView<'s>`. Escaping views past the enclosing `Scope` trigger compile-time errors. | Zero runtime cost, compile-time verified memory safety. | High compiler complexity. Requires full Wave 4 Track 4G (`D-LIFE`) implementation. |
| **Option C.4: Scoped Callback Closure (`for_each_view`)** | Views are provided exclusively as borrowed arguments to higher-order callbacks and invalidated upon callback return. | Safe by structural nesting in Rust/host code; difficult to enforce in dynamically dispatched scripting without static escape analysis. | Ergonomic friction; callbacks cannot yield or be stored across async yields. |

---

## 3. Formal Coordinator Decision

- **Current Formal Status:** **ACCEPTED / DECIDED (2026-11-14)**
- **Selected Architecture:** **Phased Hybrid of Option C.3 (Target Wave 4 Contract) + Option C.1 (Mandatory Safe Enforcement in Waves 0–3)**
- **Normative Clauses:**
  1. **Waves 0–3 Enforcement (Option C.1):** All user-facing Lust script decoders and schema bindings (`L-SCHEMA-DSL`, `D-JSONSCHEMA`, `D-CSV`, `D-MSGPACK`, `D-STREAM`) MUST materialize owned heap values (`DataValue`, `Document`). No unverified borrowed views (`DataView<'s>`) may escape to user script scopes.
  2. **Narrow Zero-Allocation Claim:** Zero-allocation view access in Waves 0–3 is strictly scoped to verified internal Rust-side bounded views (`BsonDocumentView`, `DataView<'s>` within verified host `Scope` boundaries). General zero-allocation claims for user-facing script views MUST NOT be advertised until `D-LIFE` is completed and verified.
  3. **Wave 4 Track 4G Prerequisite (Option C.3 / `D-LIFE`):** Exposing `DataView<'s>` to user Lust scripts requires full implementation and verification of static lifetime & escape analysis (`'s`) in `src/typechecker/*` in Wave 4 Track 4G (`D-LIFE`), including compile-fail escape tests and borrow invalidation oracles.

---

## 4. Downstream Unblocking & Verification Criteria

- **Unblocked Tracks:**
  - Wave 0 Track 0C (`D-LIFE-ADR`) is **CLOSED / DECIDED**.
  - Wave 1 Track 1B (`L-SCHEMA-DSL`) may proceed immediately under Clause 1 (owned `DataValue` / `Document` user script bindings).
- **Acceptance Criteria for Future Wave 4 `D-LIFE` Implementation:**
  1. Static borrow/escape checker rejects returning `DataView<'s>` past enclosing `Scope`, capturing it in closures/fibers/globals, or mutating the backing owner while a view is live.
  2. Verified allocation-counter proof for the prepared fixed-scalar zero-allocation case (`RFC_FIRST_CLASS_DATA_FORMATS_AND_SCHEMAS.md` §4) alongside `D-LAYOUT` (`Wave 2 Track 2F`).
  3. Independent DA/LV verification (`PASS`) on compile-fail escape vectors and runtime boundary controls.

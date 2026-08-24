# Lust JIT differential report

- binary: `/home/joshw/Desktop/Code/rs/lust-lang/target/release/lust`
- cases: 1249
- per-run timeout: 4.0s

## Verdict summary

| verdict | count | meaning |
|---|---:|---|
| FRONTEND | 1 | parse/type/compile error in both modes |
| MATCH_OK | 1248 | both modes correct |

## Failures by tag

| tag | total | ok | jit_wrong | jit_hang | jit_crash | other |
|---|---:|---:|---:|---:|---:|---:|
| `regression` | 21 | 20 | 0 | 0 | 0 | 1 |
| `frontend` | 4 | 3 | 0 | 0 | 0 | 1 |

## Smallest failing case per verdict

### FRONTEND -- `regression/single_letter_struct_name`

expected `7`, jit=off `error`, jit=on `error`

> type error: Cannot access field on type 'K'

```lust
struct K
  x: int
end
local a: Array<K> = []
a:push(K { x = 7 })
println(a[0]:unwrap().x)
```

## All failing cases

| case | verdict | expected | jit=off | jit=on |
|---|---|---:|---|---|
| `regression/single_letter_struct_name` | FRONTEND | 7 | error | error |

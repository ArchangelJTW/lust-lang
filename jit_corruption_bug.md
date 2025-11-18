# JIT Memory Corruption Bug: Direct vs Intermediate Variable

## Symptom

When calling `println(arr:len())` directly on a specialized array in a JIT-compiled loop with 41+ iterations, the program intermittently crashes with:

```
malloc_consolidate(): unaligned fastbin chunk detected
Aborted (core dumped)
```

However, using an intermediate variable works reliably:
```lust
local len = arr:len()
println(len)
```

## Crash Timing

The crash occurs **after** the program completes successfully and prints "done", during final cleanup when the `CompiledTrace` structures are being destroyed. Specifically, it happens when dropping the `guards: Vec<Guard>` field.

## Bytecode Differences

### Direct Call (CRASHES intermittently)
```
0008 [L010] Move R3, R5              # arr = make_array()
0009 [L011] CallMethod R3, K3, <no args>, R6    # R6 = arr:len()
0010 [L011] Move R5, R6              # prepare for println
0011 [L011] LoadGlobal R6, K4        # R6 = println (REUSES R6!)
0012 [L011] Call R6, R5..R5, R7      # println(len)
```

### Intermediate Variable (NEVER CRASHES)
```
0008 [L010] Move R3, R5              # arr = make_array()
0009 [L011] CallMethod R3, K3, <no args>, R5    # R5 = arr:len()
0010 [L011] Move R4, R5              # len = R5
0011 [L012] Move R5, R4              # prepare for println
0012 [L012] LoadGlobal R6, K4        # R6 = println (DIFFERENT REG!)
0013 [L012] Call R6, R5..R5, R7      # println(len)
```

**Key Difference**: In the crashing case, R6 is used as both the CallMethod destination AND immediately reused by LoadGlobal. In the working case, different registers are used (R5 for len, R6 for println).

## What We've Fixed So Far

### 1. Missing `remove_specialization_tracking()` calls
**Problem**: When registers were overwritten, specialization tracking wasn't cleaned up, leaving stale entries pointing to invalid JIT stack offsets.

**Fix**: Added `remove_specialization_tracking(dest)` to all instructions that write to destination registers:
- `LoadConst`
- `LoadGlobal` ← **This was particularly important**
- `Move`
- `Add`, `Sub`, etc.
- `CallMethod`
- `Call`
- `NewArray`

**Files**:
- `src/jit/trace.rs`: Lines 728, 742, 757, 780, 800, 977, 1263, 1405

### 2. Specialized `len()` Operation
**Problem**: When `arr:len()` was called on a specialized array, it was calling the method on the empty `Rc<RefCell<Vec<Value>>>` wrapper (because the data was unboxed to the JIT stack), returning 0 instead of the actual length.

**Fix**: Implemented `SpecializedOpKind::VecLen` that reads the length directly from the JIT stack.

**Files**:
- `src/jit/trace.rs`: Lines 1024-1043 (detecting and emitting specialized len)
- `src/jit/codegen/specialization.rs`: Lines 92-108 (matching VecLen), 295-334 (compile_vec_int_len)

### 3. Wrong Value Tag
**Problem**: Used tag `1` (Bool) instead of `2` (Int) when creating the length Value, causing it to be interpreted as boolean.

**Fix**: Changed tag from 1 to 2.

**File**: `src/jit/codegen/specialization.rs`: Line 329

### 4. Uninitialized Memory in Value Structure
**Problem**: Only set tag (byte 0) and value (bytes 8-15) when creating the length Value, leaving bytes 1-7 and 16-63 containing garbage. When this malformed Value was copied/cleaned up, garbage bytes could be interpreted as pointers or reference counts.

**Fix**: Zero out entire 64-byte Value structure before setting tag and value.

**File**: `src/jit/codegen/specialization.rs`: Lines 314-323

## Why the Bug Persists

Despite all these fixes, the crash still occurs intermittently with the direct call but never with the intermediate variable. This suggests:

### Hypothesis 1: Register-Specific Issue
The specific register numbers (R6 vs R5) might affect memory layout, code generation, or optimization in a way that triggers latent corruption. With 32x loop unrolling, the register usage pattern gets replicated 32 times, potentially amplifying the effect.

### Hypothesis 2: Timing/Layout Sensitivity
The extra Move instruction changes:
- The total number of TraceOps generated (511 vs 479 ops after optimization)
- The memory layout of the compiled code
- The timing of register reads/writes

This might mask an underlying heap corruption bug that only manifests under specific memory states.

### Hypothesis 3: Move Validation Side Effect
The Move operation copies the entire 64-byte Value structure and has special handling for reference-counted types (calls `jit_move_safe` for non-primitive values). This extra copying might:
- Force values into a well-formed state
- Trigger reference count operations that mask corruption
- Change memory access patterns in a way that avoids triggering the allocator's corruption detection

## The Mystery

The core mystery is: **What aspect of register R6 being reused immediately (in the direct call case) causes heap corruption that only manifests during `Vec<Guard>` cleanup?**

Possibilities:
1. The JIT code generator or optimizer has a bug specific to this register reuse pattern
2. Loop unrolling with this pattern creates out-of-bounds memory writes
3. The guards Vec allocation happens to be adjacent to memory corrupted by the JIT stack or register operations
4. Reference counting or Rc<RefCell<>> operations on the len() result interact badly with immediate register reuse

## Relevant Code Locations

### Specialization Tracking
- `src/jit/trace.rs`:
  - `specialized_registers: HashMap<Register, (usize, SpecializedLayout)>` (line 360)
  - `remove_specialization_tracking()` (lines 624-635)
  - Specialization at trace entry (lines 420-488)

### Specialized Operations
- `src/jit/trace.rs`:
  - `SpecializedOpKind` enum (lines 310-317)
  - Specialized len detection (lines 1024-1043)
- `src/jit/codegen/specialization.rs`:
  - `compile_vec_int_len()` (lines 295-334)
  - `compile_specialized_op()` (lines 92-108)

### Value Structure
- `src/bytecode/value.rs`:
  - `ValueTag` enum (lines 162-180): Int tag = 2
  - `Value` enum with `#[repr(C, u8)]` (lines 399+)
  - Each Value is 64 bytes

### Move Operation
- `src/jit/codegen/memory.rs`:
  - `compile_move()` (lines 61-101): Copies all 64 bytes of Value

## Test Cases

### Crashing Test
```lust
for i = 1, 41 do
  local arr = make_array()
  println(arr:len())  -- Direct call
end
```

### Working Test
```lust
for i = 1, 41 do
  local arr = make_array()
  local len = arr:len()  -- Intermediate variable
  println(len)
end
```

### Also Working
```lust
for i = 1, 40 do  -- 40 instead of 41
  local arr = make_array()
  println(arr:len())
end
```

The 40 vs 41 threshold is significant but unclear why - possibly related to JIT stack size limits, optimization thresholds, or memory allocator bucket sizes.

## Next Steps for Investigation

1. **Examine the generated assembly**: Compare the actual x86_64 code generated for both cases to see if there's an obvious buffer overflow or incorrect offset calculation

2. **Run under Valgrind**: Use memory error detection tools to catch the exact point of corruption

3. **Check JIT stack boundaries**: Verify that JIT stack accesses with 32x unrolling stay within the 504-byte limit

4. **Audit all register offset calculations**: Look for places where `dest_offset = reg * 64` might overflow or access past r12 + (255 * 64)

5. **Test with different register assignments**: Manually modify the bytecode to use different registers and see if the crash follows R6 specifically or the reuse pattern

6. **Add heap corruption checks**: Insert checks before/after JIT execution to narrow down when corruption occurs

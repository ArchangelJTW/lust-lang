#![cfg(feature = "std")]
//! Stage 4 - L14: JIT Cold Paths & Differential Parity Verification.
//! Compares VM interpreter vs JIT native execution on arithmetic, logic, and memory workloads.
//! Explicitly asserts JIT native trace compilation and execution for supported paths,
//! and verifies graceful interpreter fallback without state corruption when trace recording aborts.

use lust::bytecode::Value;
use lust::config::LustConfig;
use lust::jit::JitStats;
use lust::EmbeddedProgram;

fn run_differential(source: &str, func_name: &str, args: Vec<Value>) -> (Value, Value, JitStats) {
    // 1. Interpreter execution (JIT strictly disabled)
    let mut vm_cfg = LustConfig::default();
    vm_cfg.set_jit_enabled(false);
    let mut vm_prog = EmbeddedProgram::builder()
        .with_config(vm_cfg)
        .module("main", source)
        .entry_module("main")
        .compile()
        .expect("VM compile failed");

    let vm_res = vm_prog
        .call_raw(&format!("main.{func_name}"), args.clone())
        .expect("VM call failed");

    // 2. JIT execution (JIT enabled, trace recorder active)
    let mut jit_cfg = LustConfig::default();
    jit_cfg.set_jit_enabled(true);
    let mut jit_prog = EmbeddedProgram::builder()
        .with_config(jit_cfg)
        .module("main", source)
        .entry_module("main")
        .compile()
        .expect("JIT compile failed");

    let jit_res = jit_prog
        .call_raw(&format!("main.{func_name}"), args)
        .expect("JIT call failed");

    let stats = jit_prog.jit_stats();
    (vm_res, jit_res, stats)
}

// ---------------------------------------------------------------------------
// 1. Arithmetic: Integer Division & Modulo (Must NOT Silently Convert to Float)
// ---------------------------------------------------------------------------

#[test]
fn test_l14_arithmetic_integer_division_and_modulo_parity() {
    let source = r#"
        function int_div_mod_loop(n: int): int
            local acc: int = 1000000
            local i: int = 1
            while i <= n do
                local d: int = acc / (i % 7 + 2)
                local m: int = acc % (i % 5 + 2)
                acc = acc - d + m + 1
                i = i + 1
            end
            return acc
        end
    "#;

    let (vm_res, jit_res, stats) = run_differential(source, "int_div_mod_loop", vec![Value::Int(200)]);
    assert_eq!(vm_res, jit_res, "VM and JIT outputs must match exactly");
    assert!(
        stats.root_traces_compiled >= 1,
        "JIT must compile root traces for hot int div/mod loop, got: {stats:?}"
    );
    assert!(
        stats.native_trace_entries > 0,
        "JIT must execute native traces for hot int div/mod loop, got: {stats:?}"
    );
    match vm_res {
        Value::Int(val) => {
            assert!(val > 0, "Accumulator must remain positive integer");
        }
        other => panic!("Expected Value::Int, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 2. Logic: Short-circuit Boolean Chains
// ---------------------------------------------------------------------------

#[test]
fn test_l14_logic_short_circuit_chains_parity() {
    let source = r#"
        function logic_chains(n: int): int
            local true_count: int = 0
            local i: int = 0
            while i < n do
                local cond1 = (i % 2 == 0) and (i % 3 == 0)
                local cond2 = (i % 5 == 0) or (i % 7 == 0)
                local cond3 = not (i % 11 == 0)
                if (cond1 or cond2) and cond3 then
                    true_count = true_count + 1
                end
                i = i + 1
            end
            return true_count
        end
    "#;

    let (vm_res, jit_res, stats) = run_differential(source, "logic_chains", vec![Value::Int(500)]);
    assert_eq!(vm_res, jit_res, "Logic chains parity mismatch between VM and JIT");
    assert_eq!(vm_res, Value::Int(195));
    assert!(
        stats.root_traces_compiled >= 1,
        "JIT must compile root traces for logic chains loop, got: {stats:?}"
    );
    assert!(
        stats.native_trace_entries > 0,
        "JIT must execute native traces for logic chains loop, got: {stats:?}"
    );
}

// ---------------------------------------------------------------------------
// 3. Memory: Struct Field Accesses at Boundary
// ---------------------------------------------------------------------------

#[test]
fn test_l14_memory_struct_field_access_parity() {
    let source = r#"
        struct Accumulator
            val: int,
            hits: int
        end

        function struct_loop(n: int): int
            local acc = Accumulator { val = 0, hits = 0 }
            local i: int = 0
            while i < n do
                acc.val = acc.val + i
                if acc.val % 2 == 0 then
                    acc.hits = acc.hits + 1
                end
                i = i + 1
            end
            return acc.val + acc.hits
        end
    "#;

    let (vm_res, jit_res, stats) = run_differential(source, "struct_loop", vec![Value::Int(250)]);
    assert_eq!(vm_res, jit_res, "Struct field access parity mismatch between VM and JIT");
    assert!(matches!(vm_res, Value::Int(_)));
    assert!(
        stats.root_traces_compiled >= 1,
        "JIT must compile root traces for struct loop, got: {stats:?}"
    );
    assert!(
        stats.native_trace_entries > 0,
        "JIT must execute native traces for struct loop, got: {stats:?}"
    );
}

// ---------------------------------------------------------------------------
// 4. Memory: Array Indexing at Boundaries
// ---------------------------------------------------------------------------

#[test]
fn test_l14_memory_array_bounds_access_parity() {
    let source = r#"
        function array_boundary_loop(n: int): int
            local arr: Array<int> = []
            local i: int = 0
            while i < n do
                array.push(arr, i * 2)
                i = i + 1
            end

            local sum: int = 0
            i = 0
            while i < n do
                if arr[i] is Ok(v) then
                    sum = sum + v
                end
                i = i + 1
            end
            return sum
        end
    "#;

    let (vm_res, jit_res, stats) = run_differential(source, "array_boundary_loop", vec![Value::Int(200)]);
    assert_eq!(vm_res, jit_res, "Array bounds access parity mismatch");
    assert_eq!(vm_res, Value::Int(39800));
    assert!(
        stats.root_traces_compiled >= 1,
        "JIT must compile root traces for array loop, got: {stats:?}"
    );
    assert!(
        stats.native_trace_entries > 0,
        "JIT must execute native traces for array loop, got: {stats:?}"
    );
}

// ---------------------------------------------------------------------------
// 5. Arithmetic: Float Mixed Arithmetic Parity (Add, Sub, Mul, Div)
// ---------------------------------------------------------------------------

#[test]
fn test_l14_arithmetic_float_mixed_operations_parity() {
    let source = r#"
        function float_arith_loop(n: int): float
            local total: float = 100.0
            local i: int = 1
            while i <= n do
                total = (total + 2.5) * 1.01 - (total / 50.0)
                i = i + 1
            end
            return total
        end
    "#;

    let (vm_res, jit_res, stats) = run_differential(source, "float_arith_loop", vec![Value::Int(100)]);
    assert!(
        stats.root_traces_compiled >= 1,
        "JIT must compile root traces for float loop, got: {stats:?}"
    );
    assert!(
        stats.native_trace_entries > 0,
        "JIT must execute native traces for float loop, got: {stats:?}"
    );
    match (vm_res, jit_res) {
        (Value::Float(vf), Value::Float(jf)) => {
            let diff = (vf - jf).abs();
            assert!(diff < 1e-9, "Float divergence: VM={vf}, JIT={jf}, diff={diff}");
        }
        (v, j) => panic!("Expected floats from VM and JIT, got: VM={v:?}, JIT={j:?}"),
    }
}

// ---------------------------------------------------------------------------
// 6. Cold Path / Fallback: Unsupported Shl Aborts Trace, Preserves VM Parity
// ---------------------------------------------------------------------------

#[test]
fn test_l14_cold_fallback_unsupported_shl_trace_abort_parity() {
    let source = r#"
        function bitwise_loop(n: int): int
            local acc: int = 1
            local i: int = 1
            while i <= n do
                acc = (acc << 1) | (i & 1)
                if acc > 1000000 then
                    acc = acc & 65535
                end
                i = i + 1
            end
            return acc
        end
    "#;

    let (vm_res, jit_res, stats) = run_differential(source, "bitwise_loop", vec![Value::Int(300)]);
    assert_eq!(vm_res, jit_res, "VM fallback execution must match interpreter result exactly");
    assert!(
        stats.recordings_aborted > 0,
        "JIT trace recorder must abort on unsupported Shl opcode, got stats: {stats:?}"
    );
    assert_eq!(
        stats.root_traces_compiled, 0,
        "No root trace should be compiled for unsupported Shl opcode"
    );
    assert!(matches!(vm_res, Value::Int(_)));
}

// ---------------------------------------------------------------------------
// 7. Cold Path / Fallback: Unsupported Pow Aborts Trace, Preserves VM Parity
// ---------------------------------------------------------------------------

#[test]
fn test_l14_cold_fallback_unsupported_pow_trace_abort_parity() {
    let source = r#"
        function float_pow_loop(n: int): float
            local total: float = 1.0
            local i: int = 1
            while i <= n do
                total = total + (1.05 ^ i)
                i = i + 1
            end
            return total
        end
    "#;

    let (vm_res, jit_res, stats) = run_differential(source, "float_pow_loop", vec![Value::Int(50)]);
    assert_eq!(vm_res, jit_res, "VM fallback execution must match interpreter result exactly");
    assert!(
        stats.recordings_aborted > 0,
        "JIT trace recorder must abort on unsupported Pow opcode, got stats: {stats:?}"
    );
    assert_eq!(
        stats.root_traces_compiled, 0,
        "No root trace should be compiled for unsupported Pow opcode"
    );
}

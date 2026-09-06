use lust::{EmbeddedProgram, LustConfig, Value};
use std::hint::black_box;
use std::time::Instant;

const SOURCE: &str = r#"
function sum_int(n: int): int
    local sum: int = 0
    local i: int = 1
    while i <= n do
        sum = sum + i
        i = i + 1
    end
    return sum
end

function sum_float(n: float): float
    local sum: float = 0.0
    local i: float = 1.0
    while i <= n do
        sum = sum + i
        i = i + 1.0
    end
    return sum
end

function sum_down(n: float): float
    local sum: float = 0.0
    local i: float = n
    while i >= 1.0 do
        sum = sum + i
        i = i - 1.0
    end
    return sum
end
"#;

fn main() -> lust::Result<()> {
    assert!(
        cfg!(target_arch = "x86_64"),
        "this benchmark targets x86_64"
    );
    for jit in [false, true] {
        let mut config = LustConfig::default();
        config.set_jit_enabled(jit);
        let mut program = EmbeddedProgram::builder()
            .with_config(config)
            .module("bench", SOURCE)
            .entry_module("bench")
            .compile()?;
        let iterations: i64 = if jit { 100_000_000 } else { 250_000 };
        let expected = iterations * (iterations + 1) / 2;

        for name in ["sum_int", "sum_float", "sum_down"] {
            let function = format!("bench.{name}");
            let argument = |n: i64| {
                if name != "sum_int" {
                    Value::Float(n as _)
                } else {
                    Value::Int(n as _)
                }
            };
            // Exclude parsing, typechecking, and trace compilation from timing.
            program.vm_mut().call(&function, vec![argument(1_000)])?;
            let mut samples = Vec::new();
            for _ in 0..7 {
                let start = Instant::now();
                let result = program
                    .vm_mut()
                    .call(&function, vec![black_box(argument(iterations))])?;
                samples.push(start.elapsed());
                if name == "sum_int" {
                    assert_eq!(result.as_int(), Some(expected as _));
                } else {
                    assert_eq!(result.as_float(), Some(expected as _));
                }
                black_box(result);
            }
            samples.sort();
            println!(
                "{} {name:9}: {:8.3} ms, {:7.2} ns/iteration ({iterations} iterations)",
                if jit { "JIT" } else { "VM " },
                samples[3].as_secs_f64() * 1_000.0,
                samples[3].as_secs_f64() * 1_000_000_000.0 / iterations as f64,
            );
        }
        if jit {
            let stats = program.jit_stats();
            assert!(stats.root_traces_compiled >= 3, "{stats:?}");
            assert!(stats.native_trace_entries >= 21, "{stats:?}");
        }
    }
    Ok(())
}

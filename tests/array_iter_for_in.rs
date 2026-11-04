use lust::{EmbeddedProgram, LustConfig};

const SOURCE: &str = r#"
function array_iterator_values(): string
    local values = [10, 20, 30]
    local output = ""
    for value in array.iter(values) do
        output = output .. tostring(value) .. ","
    end
    return output
end

function direct_array_values(): string
    local values = [10, 20, 30]
    local output = ""
    for value in values do
        output = output .. tostring(value) .. ","
    end
    return output
end

function map_iterator_sum(): int
    local values: Map<string, int> = { a = 10, b = 20, c = 30 }
    local total: int = 0
    for key, value in map.iter(values) do
        total = total + value
    end
    return total
end
"#;

fn compile(jit_enabled: bool) -> EmbeddedProgram {
    let mut config = LustConfig::default();
    config.set_jit_enabled(jit_enabled);
    EmbeddedProgram::builder()
        .with_config(config)
        .module("main", SOURCE)
        .entry_module("main")
        .compile()
        .expect("compile iterator test program")
}

#[test]
fn for_in_consumes_explicit_array_iterators() {
    for jit_enabled in [false, true] {
        let mut program = compile(jit_enabled);

        for _ in 0..40 {
            let values: String = program
                .call_typed("main.array_iterator_values", ())
                .expect("iterate array.iter result");
            assert_eq!(values, "10,20,30,");

            let direct_values: String = program
                .call_typed("main.direct_array_values", ())
                .expect("iterate array directly");
            assert_eq!(direct_values, "10,20,30,");

            let sum: i64 = program
                .call_typed("main.map_iterator_sum", ())
                .expect("iterate map.iter key/value pairs");
            assert_eq!(sum, 60);
        }
    }
}

use lust::{EmbeddedProgram, LustConfig};

const LOGICAL_SOURCE: &str = r#"
function logical_results(): string
    local s: string = "x"
    local and_value = s and "YES"
    local or_value = s or "NO"
    local false_or_value = false or "fallback"
    local n: int = 7
    local number_and = n and 8
    local number_or = n or 9
    return s .. "|" .. and_value .. "|" .. or_value .. "|" .. tostring(number_and) .. "|" .. tostring(number_or) .. "|" .. false_or_value
end
"#;

fn compile(source: &str, jit_enabled: bool) -> EmbeddedProgram {
    let mut config = LustConfig::default();
    config.set_jit_enabled(jit_enabled);
    EmbeddedProgram::builder()
        .with_config(config)
        .module("main", source)
        .entry_module("main")
        .compile()
        .expect("compile test program")
}

#[test]
fn logical_operators_preserve_the_left_operand() {
    for jit_enabled in [false, true] {
        let mut program = compile(LOGICAL_SOURCE, jit_enabled);
        let mut result = String::new();
        for _ in 0..100 {
            result = program
                .call_typed("main.logical_results", ())
                .expect("call logical_results");
        }

        assert_eq!(result, "x|YES|x|8|7|fallback");
    }
}

#[test]
fn native_type_returns_a_string() {
    let mut program = compile(
        r#"
            function native_type_names(): string
                return type(1) .. "|" .. type("a") .. "|" .. type([1])
            end
        "#,
        false,
    );

    let names: String = program
        .call_typed("main.native_type_names", ())
        .expect("call native_type_names");
    assert_eq!(names, "int|string|array");
}

#[test]
fn map_field_access_cannot_hide_a_missing_key() {
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    let err = EmbeddedProgram::builder()
        .with_config(config)
        .module(
            "main",
            r#"
            function missing(): int
                local values: Map<string, int> = {["present"] = 1}
                return values.missing
            end
        "#,
        )
        .entry_module("main")
        .compile()
        .err()
        .expect("typed Map field access must be rejected");
    assert!(err.to_string().contains("map.get"), "{err}");

    for jit_enabled in [false, true] {
        let mut program = compile(
            r#"
                function lookup(key: string): Option<int>
                    local values: Map<string, int> = {["present"] = 1}
                    return map.get(values, key)
                end
            "#,
            jit_enabled,
        );
        let present: Option<i64> = program
            .call_typed("main.lookup", "present".to_string())
            .expect("lookup present map key");
        let missing: Option<i64> = program
            .call_typed("main.lookup", "missing".to_string())
            .expect("lookup missing map key");
        assert_eq!(present, Some(1));
        assert_eq!(missing, None);
    }
}

use lust::{EmbeddedProgram, LustConfig};

const SOURCE: &str = r#"
function draw(): string
    return tostring(math.random()) .. "|" .. tostring(math.random(-10, 10))
end

function reseed(seed: int): int
    math.randomseed(seed)
    return 0
end
"#;

fn make_program(jit_enabled: bool) -> EmbeddedProgram {
    let mut config = LustConfig::default();
    config.set_jit_enabled(jit_enabled);
    EmbeddedProgram::builder()
        .with_config(config)
        .module("main", SOURCE)
        .entry_module("main")
        .compile()
        .expect("compile RNG test program")
}

fn draw(program: &mut EmbeddedProgram) -> String {
    program
        .call_typed("main.draw", ())
        .expect("draw random values")
}

#[test]
fn embedded_rngs_are_isolated_and_script_reseed_is_local() {
    for jit_enabled in [false, true] {
        let mut first = make_program(jit_enabled);
        let mut second = make_program(jit_enabled);
        let mut first_control = make_program(jit_enabled);
        let mut second_control = make_program(jit_enabled);

        for program in [
            &mut first,
            &mut second,
            &mut first_control,
            &mut second_control,
        ] {
            program.set_random_seed(42);
        }

        for _ in 0..64 {
            let expected_first = draw(&mut first_control);
            let expected_second = draw(&mut second_control);
            assert_eq!(draw(&mut first), expected_first);
            assert_eq!(draw(&mut second), expected_second);
        }

        first
            .call_typed::<_, i64>("main.reseed", 99_i64)
            .expect("reseed first VM from Lust");
        first_control.set_random_seed(99);
        assert_eq!(draw(&mut first), draw(&mut first_control));

        // Reseeding the first VM did not change the second VM's next value.
        assert_eq!(draw(&mut second), draw(&mut second_control));
    }
}

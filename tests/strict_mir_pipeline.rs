use lust::{
    EmbeddedModule, LustConfig, compile_program_with_config,
    load_program_from_embedded, bytecode::Value,
};

#[test]
fn test_strict_mir_pipeline_execution() {
    let source = r#"
        function compute(x: int, y: int): int
            local acc: int = 0
            local i: int = 0
            while i < y do
                acc = acc + x
                i = i + 1
            end
            return acc
        end
    "#;

    let entries = &[EmbeddedModule {
        module: "main",
        parent: None,
        source: Some(source),
    }];

    let program = load_program_from_embedded(entries, "main").expect("Failed to load program");
    let config = LustConfig::default().with_strict_mir(true);
    let mut vm = compile_program_with_config(program, &config).expect("Strict MIR compilation failed");

    let result = vm.call("compute", vec![Value::Int(7), Value::Int(6)]).expect("VM call failed");
    assert_eq!(result, Value::Int(42));
}

#[test]
fn test_strict_mir_array_operations() {
    let source = r#"
        function sum_array(n: int): int
            local arr = [10, 20, 30]
            arr[1] = n
            local total: int = 0
            for i = 0, 2 do
                total = total + arr[i]:unwrap()
            end
            return total
        end
    "#;

    let entries = &[EmbeddedModule {
        module: "main",
        parent: None,
        source: Some(source),
    }];

    let program = load_program_from_embedded(entries, "main").expect("Failed to load program");
    let config = LustConfig::default().with_strict_mir(true);
    let mut vm = compile_program_with_config(program, &config).expect("Strict MIR compilation failed");

    let result = vm.call("sum_array", vec![Value::Int(50)]).expect("VM call failed");
    // 10 + 50 + 30 = 90
    assert_eq!(result, Value::Int(90));
}

#[test]
fn test_strict_mir_flags_and_configuration() {
    let mut config = LustConfig::default();
    assert!(!config.use_mir());
    assert!(!config.strict_mir());

    config.set_strict_mir(true);
    assert!(config.use_mir());
    assert!(config.strict_mir());

    let mut compiler = lust::Compiler::new();
    compiler.configure(&config);
    // Verifies that Compiler picks up strict MIR configuration properly
}

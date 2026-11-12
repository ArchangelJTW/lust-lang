use lust::intern::Interner;
use lust::number::{parse_float_literal, parse_integer_literal, NumberParseError};
use lust::{Compiler, Function, Lexer, LustConfig, Parser, TypeChecker, Value, VM};
use lust::sim::{MinimalSimScheduler, StepOutcome};

fn compile_source(source: &str) -> lust::Result<Vec<Function>> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize()?;
    let items = Parser::new(tokens).parse()?;
    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    compiler.compile_module(&items)
}

#[test]
fn test_numeric_literals_contract_parsing() {
    // 1. Hexadecimal bit patterns and MSB
    assert_eq!(parse_integer_literal("0x0").unwrap(), 0);
    assert_eq!(parse_integer_literal("0x0F").unwrap(), 15);
    assert_eq!(parse_integer_literal("0Xff").unwrap(), 255);
    assert_eq!(parse_integer_literal("0xDEAD_BEEF").unwrap(), 3735928559);
    assert_eq!(parse_integer_literal("0x7FFF_FFFF_FFFF_FFFF").unwrap(), i64::MAX);
    assert_eq!(parse_integer_literal("0x8000_0000_0000_0000").unwrap(), i64::MIN);
    assert_eq!(parse_integer_literal("0xFFFF_FFFF_FFFF_FFFF").unwrap(), -1);

    // 2. Binary
    assert_eq!(parse_integer_literal("0b0").unwrap(), 0);
    assert_eq!(parse_integer_literal("0b1").unwrap(), 1);
    assert_eq!(parse_integer_literal("0B1010").unwrap(), 10);
    assert_eq!(parse_integer_literal("0b1111_0000").unwrap(), 240);
    assert_eq!(parse_integer_literal("0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111").unwrap(), -1);

    // 3. Octal
    assert_eq!(parse_integer_literal("0o0").unwrap(), 0);
    assert_eq!(parse_integer_literal("0o7").unwrap(), 7);
    assert_eq!(parse_integer_literal("0O755").unwrap(), 493);
    assert_eq!(parse_integer_literal("0o644").unwrap(), 420);

    // 4. Decimal with '_'
    assert_eq!(parse_integer_literal("1_000_000").unwrap(), 1000000);
    assert_eq!(parse_integer_literal("42_133_7").unwrap(), 421337);
    assert_eq!(parse_integer_literal("1_0").unwrap(), 10);
    assert_eq!(parse_integer_literal("9_223_372_036_854_775_807").unwrap(), i64::MAX);

    // 5. Overflow teeth
    assert_eq!(
        parse_integer_literal("0x1_0000_0000_0000_0000").unwrap_err(),
        NumberParseError::Overflow("exceeds u64::MAX")
    );
    assert_eq!(
        parse_integer_literal("9223372036854775808").unwrap_err(),
        NumberParseError::Overflow("exceeds i64::MAX")
    );

    // 6. Malformed teeth
    assert_eq!(
        parse_integer_literal("0x").unwrap_err(),
        NumberParseError::MalformedLiteral("Expected hex digits after prefix")
    );
    assert_eq!(
        parse_integer_literal("0b").unwrap_err(),
        NumberParseError::MalformedLiteral("Expected binary digits after prefix")
    );
    assert_eq!(
        parse_integer_literal("0o").unwrap_err(),
        NumberParseError::MalformedLiteral("Expected octal digits after prefix")
    );

    // 7. Float with '_'
    assert_eq!(parse_float_literal("3.1415_9265").unwrap(), 3.14159265);
    assert_eq!(parse_float_literal("1_000.5_0").unwrap(), 1000.5);
    assert_eq!(parse_float_literal("1_0.5").unwrap(), 10.5);
    assert_eq!(parse_float_literal("1e1_0").unwrap(), 1e10);
}

#[test]
fn test_numeric_literals_bytecode_identity_and_dst_pipeline() {
    // 1. Bytecode Identity: 0x0FFF & 0b1010_0101 vs 4095 & 165
    let src_radix = "function worker(): int\nlocal mask = 0x0FFF\nlocal flags = 0b1010_0101\nreturn mask & flags\nend";
    let src_dec = "function worker(): int\nlocal mask = 4095\nlocal flags = 165\nreturn mask & flags\nend";

    let f_radix = compile_source(src_radix).expect("compile radix");
    let f_dec = compile_source(src_dec).expect("compile decimal");

    assert_eq!(f_radix[0].chunk.instructions, f_dec[0].chunk.instructions, "Instructions must be bit-identical");
    assert_eq!(f_radix[0].chunk.constants, f_dec[0].chunk.constants, "Constants must be bit-identical");

    // 2. Verify MinimalSimScheduler execution via spawn_fiber / step_fiber
    let mut vm = VM::with_seed(100);
    vm.load_functions(f_radix);
    let mut sched = MinimalSimScheduler::new(vm);
    let handle = sched.spawn_fiber(Value::Function(0), vec![]).expect("spawn fiber");
    let outcome = sched.step_fiber(handle, None).expect("step fiber");

    assert_eq!(outcome, StepOutcome::Completed(Some(Value::Int(165))));
    assert!(sched.vm.gas_used() > 0);
}

#[test]
fn test_numeric_literals_i64_min_and_overflow_contract() {
    // Negation of 9223372036854775808 produces i64::MIN literal directly
    let src_min = "function test_min(): int\nreturn -9223372036854775808\nend";
    let f_min = compile_source(src_min).expect("compile min");
    assert_eq!(f_min[0].chunk.constants[0], Value::Int(i64::MIN));

    // Standalone 9223372036854775808 exceeds i64::MAX and must fail compilation
    let src_overflow = "function test_overflow(): int\nreturn 9223372036854775808\nend";
    assert!(compile_source(src_overflow).is_err(), "Standalone 9223372036854775808 must fail");

    // Invalid literal suffixes must fail compilation
    assert!(compile_source("function f(): int\nreturn 0x1_G\nend").is_err(), "0x1_G must fail");
    assert!(compile_source("function f(): int\nreturn 0b102\nend").is_err(), "0b102 must fail");
    assert!(compile_source("function f(): int\nreturn 1__0\nend").is_err(), "1__0 must fail");
}

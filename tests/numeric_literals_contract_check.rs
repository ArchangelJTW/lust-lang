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
    assert_eq!(parse_integer_literal("0o177777").unwrap(), 65535);
    assert_eq!(parse_integer_literal("0o0123_4567").unwrap(), 342391);
    assert_eq!(parse_integer_literal("0o0777_7777_7777_7777_7777_77").unwrap(), i64::MAX);
    assert_eq!(parse_integer_literal("0o1000_0000_0000_0000_0000_00").unwrap(), i64::MIN);
    assert_eq!(parse_integer_literal("0o1777_7777_7777_7777_7777_77").unwrap(), -1);

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

    // 7. Float with '_' and exponents
    assert_eq!(parse_float_literal("3.1415_9265").unwrap(), 3.14159265);
    assert_eq!(parse_float_literal("1_000.5_0").unwrap(), 1000.5);
    assert_eq!(parse_float_literal("1_0.5").unwrap(), 10.5);
    assert_eq!(parse_float_literal("1e1_0").unwrap(), 1e10);
    assert_eq!(parse_float_literal("1e3").unwrap(), 1000.0);
    assert_eq!(parse_float_literal("1e-3").unwrap(), 0.001);
    assert_eq!(parse_float_literal("1.0e+6").unwrap(), 1000000.0);
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

    // 2. Octal Bytecode Identity: 0o755 vs 493
    let src_oct = "function worker(): int\nlocal mode = 0o755\nreturn mode\nend";
    let src_oct_dec = "function worker(): int\nlocal mode = 493\nreturn mode\nend";
    let f_oct = compile_source(src_oct).expect("compile octal");
    let f_oct_dec = compile_source(src_oct_dec).expect("compile octal decimal");
    assert_eq!(f_oct[0].chunk.instructions, f_oct_dec[0].chunk.instructions, "Octal instructions must be bit-identical");
    assert_eq!(f_oct[0].chunk.constants, f_oct_dec[0].chunk.constants, "Octal constants must be bit-identical");

    // 3. Float with '_' Bytecode Identity: 3.1415_9265 vs 3.14159265
    let src_flt = "function worker(): float\nlocal pi = 3.1415_9265\nreturn pi\nend";
    let src_flt_clean = "function worker(): float\nlocal pi = 3.14159265\nreturn pi\nend";
    let f_flt = compile_source(src_flt).expect("compile float with _");
    let f_flt_clean = compile_source(src_flt_clean).expect("compile float without _");
    assert_eq!(f_flt[0].chunk.instructions, f_flt_clean[0].chunk.instructions, "Float instructions must be bit-identical");
    assert_eq!(f_flt[0].chunk.constants, f_flt_clean[0].chunk.constants, "Float constants must be bit-identical");

    // 4. MSB Boundary Bytecode Identity: 0x7FFF_FFFF_FFFF_FFFF vs 9223372036854775807 (i64::MAX)
    let src_max_radix = "function worker(): int\nlocal val = 0x7FFF_FFFF_FFFF_FFFF\nreturn val\nend";
    let src_max_dec = "function worker(): int\nlocal val = 9223372036854775807\nreturn val\nend";
    let f_max_radix = compile_source(src_max_radix).expect("compile max radix");
    let f_max_dec = compile_source(src_max_dec).expect("compile max decimal");
    assert_eq!(f_max_radix[0].chunk.instructions, f_max_dec[0].chunk.instructions);
    assert_eq!(f_max_radix[0].chunk.constants, f_max_dec[0].chunk.constants);

    // 5. MSB Boundary Bytecode Identity: 0x8000_0000_0000_0000 vs -9223372036854775808 (i64::MIN)
    let src_min_radix = "function worker(): int\nlocal val = 0x8000_0000_0000_0000\nreturn val\nend";
    let src_min_dec = "function worker(): int\nlocal val = -9223372036854775808\nreturn val\nend";
    let f_min_radix = compile_source(src_min_radix).expect("compile min radix");
    let f_min_dec = compile_source(src_min_dec).expect("compile min decimal");
    assert_eq!(f_min_radix[0].chunk.instructions, f_min_dec[0].chunk.instructions);
    assert_eq!(f_min_radix[0].chunk.constants, f_min_dec[0].chunk.constants);

    // 6. MSB Boundary Bytecode Identity: 0xFFFF_FFFF_FFFF_FFFF (hex -1), binary -1, octal -1
    let src_neg1_hex = "function worker(): int\nlocal val = 0xFFFF_FFFF_FFFF_FFFF\nreturn val\nend";
    let src_neg1_bin = "function worker(): int\nlocal val = 0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111\nreturn val\nend";
    let src_neg1_oct = "function worker(): int\nlocal val = 0o1777_7777_7777_7777_7777_77\nreturn val\nend";

    let f_neg1_hex = compile_source(src_neg1_hex).expect("compile -1 hex");
    let f_neg1_bin = compile_source(src_neg1_bin).expect("compile -1 bin");
    let f_neg1_oct = compile_source(src_neg1_oct).expect("compile -1 oct");

    assert_eq!(f_neg1_hex[0].chunk.instructions, f_neg1_bin[0].chunk.instructions, "Hex and Bin -1 instructions must match");
    assert_eq!(f_neg1_hex[0].chunk.constants, f_neg1_bin[0].chunk.constants, "Hex and Bin -1 constants must match");
    assert_eq!(f_neg1_hex[0].chunk.instructions, f_neg1_oct[0].chunk.instructions, "Hex and Oct -1 instructions must match");
    assert_eq!(f_neg1_hex[0].chunk.constants, f_neg1_oct[0].chunk.constants, "Hex and Oct -1 constants must match");
    assert_eq!(f_neg1_hex[0].chunk.constants[0], Value::Int(-1));

    // 7. Verify MinimalSimScheduler execution via spawn_fiber / step_fiber
    let mut vm = VM::with_seed(100);
    vm.load_functions(f_radix);
    let mut sched = MinimalSimScheduler::new(vm);
    let handle = sched.spawn_fiber(Value::Function(0), vec![]).expect("spawn fiber");
    let outcome = sched.step_fiber(handle, None).expect("step fiber");

    assert_eq!(outcome, StepOutcome::Completed(Some(Value::Int(165))));
    assert!(sched.vm.gas_used() > 0);

    // 10. Verify MinimalSimScheduler execution for octal function
    let mut vm_oct = VM::with_seed(100);
    vm_oct.load_functions(f_oct);
    let mut sched_oct = MinimalSimScheduler::new(vm_oct);
    let handle_oct = sched_oct.spawn_fiber(Value::Function(0), vec![]).expect("spawn oct fiber");
    let outcome_oct = sched_oct.step_fiber(handle_oct, None).expect("step oct fiber");
    assert_eq!(outcome_oct, StepOutcome::Completed(Some(Value::Int(493))));
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

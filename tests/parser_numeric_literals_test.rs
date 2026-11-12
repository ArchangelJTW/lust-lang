use lust::ast::{ExprKind, ItemKind, Literal, Pattern, StmtKind};
use lust::intern::Interner;
use lust::{Compiler, Function, Item, Lexer, LustConfig, LustError, Parser, TypeChecker, Value, VM};

fn parse_to_items(source: &str) -> Result<Vec<Item>, LustError> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize()?;
    Parser::new(tokens).parse()
}

fn compile_source(source: &str) -> Result<Vec<Function>, LustError> {
    let items = parse_to_items(source)?;
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
fn test_radix_literals_in_expressions() {
    // 1. Arithmetic with mixed bases and underscores:
    // 0x10 (16) + 0b1010 (10) - 0o12 (10) + 1_000 (1000) = 1016
    let src_arith = "function calc_arith(): int\nreturn 0x10 + 0b1010 - 0o12 + 1_000\nend";
    let funcs = compile_source(src_arith).expect("compile arith");
    let mut vm = VM::new();
    vm.load_functions(funcs);
    let res = vm.call("calc_arith", vec![]).expect("run calc_arith");
    assert_eq!(res, Value::Int(1016));

    // 2. Bitwise operations:
    // 0xDEAD & 0xFFFF = 0xDEAD (57005)
    // 0b1100_0011 | 0b0011_1100 = 0b1111_1111 (255)
    // 0o755 ^ 0o055 = 0o700 (448)
    let src_bitwise = r#"
    function calc_bitwise(): int
        local a = 0xDEAD & 0xFFFF
        local b = 0b1100_0011 | 0b0011_1100
        local c = 0o755 ^^ 0o055
        if a == 57005 and b == 255 and c == 448 then
            return 1
        else
            return 0
        end
    end
    "#;
    let funcs = compile_source(src_bitwise).expect("compile bitwise");
    let mut vm = VM::new();
    vm.load_functions(funcs);
    let res = vm.call("calc_bitwise", vec![]).expect("run calc_bitwise");
    assert_eq!(res, Value::Int(1));

    // 3. Bit shifts:
    // (0x01 << 0b10) >> 1 = (1 << 2) >> 1 = 4 >> 1 = 2
    let src_shifts = "function calc_shifts(): int\nreturn (0x01 << 0b10) >> 1\nend";
    let funcs = compile_source(src_shifts).expect("compile shifts");
    let mut vm = VM::new();
    vm.load_functions(funcs);
    let res = vm.call("calc_shifts", vec![]).expect("run calc_shifts");
    assert_eq!(res, Value::Int(2));

    // 4. Function arguments with radix literals:
    let src_args = r#"
    function sum4(a: int, b: int, c: int, d: int): int
        return a + b + c + d
    end
    function call_sum4(): int
        return sum4(0x20, 0b1000, 0o10, 1_000)
    end
    "#;
    let funcs = compile_source(src_args).expect("compile args");
    let mut vm = VM::new();
    vm.load_functions(funcs);
    let res = vm.call("call_sum4", vec![]).expect("run call_sum4");
    // 32 + 8 + 8 + 1000 = 1048
    assert_eq!(res, Value::Int(1048));

    // 5. Constants & local bindings with underscores:
    let src_consts = r#"
    function check_consts(): int
        local a = 0xABCD
        local b = 0b1010_1010
        local c = 0o644
        local d = 1_234_567
        return a + b + c + d
    end
    "#;
    // 43981 + 170 + 420 + 1234567 = 1279138
    let funcs = compile_source(src_consts).expect("compile consts");
    let mut vm = VM::new();
    vm.load_functions(funcs);
    let res = vm.call("check_consts", vec![]).expect("run check_consts");
    assert_eq!(res, Value::Int(1279138));
}

#[test]
fn test_i64_min_boundary_and_overflow() {
    // 1. Direct AST inspection of -9223372036854775808
    let src_min = "function f(): int\nreturn -9223372036854775808\nend";
    let items = parse_to_items(src_min).expect("parse min");
    if let ItemKind::Function(func) = &items[0].kind {
        if let StmtKind::Return(vals) = &func.body[0].kind {
            match &vals[0].kind {
                ExprKind::Literal(Literal::Integer(val)) => {
                    assert_eq!(*val, i64::MIN);
                }
                other => panic!("Expected Literal::Integer(i64::MIN), got {:?}", other),
            }
        } else {
            panic!("Expected StmtKind::Return");
        }
    } else {
        panic!("Expected Function");
    }

    // 2. Direct AST inspection of -9_223_372_036_854_775_808 with underscores
    let src_min_underscore = "function f(): int\nreturn -9_223_372_036_854_775_808\nend";
    let items_u = parse_to_items(src_min_underscore).expect("parse min with _");
    if let ItemKind::Function(func) = &items_u[0].kind {
        if let StmtKind::Return(vals) = &func.body[0].kind {
            match &vals[0].kind {
                ExprKind::Literal(Literal::Integer(val)) => {
                    assert_eq!(*val, i64::MIN);
                }
                other => panic!("Expected Literal::Integer(i64::MIN), got {:?}", other),
            }
        } else {
            panic!("Expected StmtKind::Return");
        }
    }

    // 3. Bytecode and VM execution of -9223372036854775808
    let funcs = compile_source(src_min).expect("compile min");
    assert_eq!(funcs[0].chunk.constants[0], Value::Int(i64::MIN));
    let mut vm = VM::new();
    vm.load_functions(funcs);
    let res = vm.call("f", vec![]).expect("run min");
    assert_eq!(res, Value::Int(i64::MIN));

    // 4. Positive standalone 9223372036854775808 must fail with parser overflow error
    let src_pos_overflow = "function f(): int\nreturn 9223372036854775808\nend";
    let err = parse_to_items(src_pos_overflow).unwrap_err();
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("Integer overflow: exceeds i64::MAX")
            || err_msg.contains("exceeds i64::MAX"),
        "Error message should mention overflow: {}",
        err_msg
    );

    // 5. Positive standalone with underscores 9_223_372_036_854_775_808 must also fail
    let src_pos_u_overflow = "function f(): int\nreturn 9_223_372_036_854_775_808\nend";
    let err_u = parse_to_items(src_pos_u_overflow).unwrap_err();
    let err_u_msg = err_u.to_string();
    assert!(
        err_u_msg.contains("Integer overflow: exceeds i64::MAX")
            || err_u_msg.contains("exceeds i64::MAX"),
        "Error message should mention overflow: {}",
        err_u_msg
    );

    // 6. Parenthesized -(9223372036854775808) fails because the inner operand is standalone 9223372036854775808
    let src_paren_overflow = "function f(): int\nreturn -(9223372036854775808)\nend";
    assert!(
        parse_to_items(src_paren_overflow).is_err(),
        "Parenthesized -(9223372036854775808) must fail because inner primary literal overflows i64::MAX"
    );
}

#[test]
fn test_is_literal_patterns() {
    let source = r#"
    function check_hex(x: int): bool
        return x is 0x0F
    end

    function check_bin(x: int): bool
        return x is 0b1010
    end

    function check_oct(x: int): bool
        return x is 0o755
    end

    function check_dec_sep(x: int): bool
        return x is 1_000
    end

    function check_neg1(x: int): bool
        return x is -1
    end

    function check_min(x: int): bool
        return x is -9223372036854775808
    end

    function check_min_sep(x: int): bool
        return x is -9_223_372_036_854_775_808
    end
    "#;

    // Verify AST patterns
    let items = parse_to_items(source).expect("parse is patterns");
    assert_eq!(items.len(), 7);

    // check_hex -> Pattern::Literal(Literal::Integer(15))
    if let ItemKind::Function(f) = &items[0].kind {
        if let StmtKind::Return(vals) = &f.body[0].kind {
            if let ExprKind::IsPattern { pattern, .. } = &vals[0].kind {
                assert_eq!(*pattern, Pattern::Literal(Literal::Integer(15)));
            } else {
                panic!("Expected IsPattern");
            }
        } else {
            panic!("Expected StmtKind::Return");
        }
    }

    // check_bin -> Pattern::Literal(Literal::Integer(10))
    if let ItemKind::Function(f) = &items[1].kind {
        if let StmtKind::Return(vals) = &f.body[0].kind {
            if let ExprKind::IsPattern { pattern, .. } = &vals[0].kind {
                assert_eq!(*pattern, Pattern::Literal(Literal::Integer(10)));
            } else {
                panic!("Expected IsPattern");
            }
        } else {
            panic!("Expected StmtKind::Return");
        }
    }

    // check_oct -> Pattern::Literal(Literal::Integer(493))
    if let ItemKind::Function(f) = &items[2].kind {
        if let StmtKind::Return(vals) = &f.body[0].kind {
            if let ExprKind::IsPattern { pattern, .. } = &vals[0].kind {
                assert_eq!(*pattern, Pattern::Literal(Literal::Integer(493)));
            } else {
                panic!("Expected IsPattern");
            }
        } else {
            panic!("Expected StmtKind::Return");
        }
    }

    // check_dec_sep -> Pattern::Literal(Literal::Integer(1000))
    if let ItemKind::Function(f) = &items[3].kind {
        if let StmtKind::Return(vals) = &f.body[0].kind {
            if let ExprKind::IsPattern { pattern, .. } = &vals[0].kind {
                assert_eq!(*pattern, Pattern::Literal(Literal::Integer(1000)));
            } else {
                panic!("Expected IsPattern");
            }
        } else {
            panic!("Expected StmtKind::Return");
        }
    }

    // check_neg1 -> Pattern::Literal(Literal::Integer(-1))
    if let ItemKind::Function(f) = &items[4].kind {
        if let StmtKind::Return(vals) = &f.body[0].kind {
            if let ExprKind::IsPattern { pattern, .. } = &vals[0].kind {
                assert_eq!(*pattern, Pattern::Literal(Literal::Integer(-1)));
            } else {
                panic!("Expected IsPattern");
            }
        } else {
            panic!("Expected StmtKind::Return");
        }
    }

    // check_min -> Pattern::Literal(Literal::Integer(i64::MIN))
    if let ItemKind::Function(f) = &items[5].kind {
        if let StmtKind::Return(vals) = &f.body[0].kind {
            if let ExprKind::IsPattern { pattern, .. } = &vals[0].kind {
                assert_eq!(*pattern, Pattern::Literal(Literal::Integer(i64::MIN)));
            } else {
                panic!("Expected IsPattern");
            }
        } else {
            panic!("Expected StmtKind::Return");
        }
    }

    // check_min_sep -> Pattern::Literal(Literal::Integer(i64::MIN))
    if let ItemKind::Function(f) = &items[6].kind {
        if let StmtKind::Return(vals) = &f.body[0].kind {
            if let ExprKind::IsPattern { pattern, .. } = &vals[0].kind {
                assert_eq!(*pattern, Pattern::Literal(Literal::Integer(i64::MIN)));
            } else {
                panic!("Expected IsPattern");
            }
        } else {
            panic!("Expected StmtKind::Return");
        }
    }

    // Compile and execute in VM
    let funcs = compile_source(source).expect("compile is functions");
    let mut vm = VM::new();
    vm.load_functions(funcs);

    // check_hex: 0x0F (15)
    assert_eq!(vm.call("check_hex", vec![Value::Int(15)]).unwrap(), Value::Bool(true));
    assert_eq!(vm.call("check_hex", vec![Value::Int(16)]).unwrap(), Value::Bool(false));

    // check_bin: 0b1010 (10)
    assert_eq!(vm.call("check_bin", vec![Value::Int(10)]).unwrap(), Value::Bool(true));
    assert_eq!(vm.call("check_bin", vec![Value::Int(11)]).unwrap(), Value::Bool(false));

    // check_oct: 0o755 (493)
    assert_eq!(vm.call("check_oct", vec![Value::Int(493)]).unwrap(), Value::Bool(true));
    assert_eq!(vm.call("check_oct", vec![Value::Int(492)]).unwrap(), Value::Bool(false));

    // check_dec_sep: 1_000 (1000)
    assert_eq!(vm.call("check_dec_sep", vec![Value::Int(1000)]).unwrap(), Value::Bool(true));
    assert_eq!(vm.call("check_dec_sep", vec![Value::Int(999)]).unwrap(), Value::Bool(false));

    // check_neg1: -1
    assert_eq!(vm.call("check_neg1", vec![Value::Int(-1)]).unwrap(), Value::Bool(true));
    assert_eq!(vm.call("check_neg1", vec![Value::Int(1)]).unwrap(), Value::Bool(false));

    // check_min: -9223372036854775808 (i64::MIN)
    assert_eq!(vm.call("check_min", vec![Value::Int(i64::MIN)]).unwrap(), Value::Bool(true));
    assert_eq!(vm.call("check_min", vec![Value::Int(0)]).unwrap(), Value::Bool(false));

    // check_min_sep: -9_223_372_036_854_775_808 (i64::MIN)
    assert_eq!(vm.call("check_min_sep", vec![Value::Int(i64::MIN)]).unwrap(), Value::Bool(true));
    assert_eq!(vm.call("check_min_sep", vec![Value::Int(-1)]).unwrap(), Value::Bool(false));
}

#[test]
fn test_nested_patterns_in_enum() {
    let source = r#"
    function check_some_hex(opt: Option<int>): bool
        return opt is Some(0x0F)
    end

    function check_some_bin(opt: Option<int>): bool
        return opt is Some(0b1010)
    end

    function check_some_oct(opt: Option<int>): bool
        return opt is Some(0o755)
    end

    function check_some_min(opt: Option<int>): bool
        return opt is Some(-9223372036854775808)
    end

    function check_some_neg1(opt: Option<int>): bool
        return opt is Some(-1)
    end

    function check_some_dec(opt: Option<int>): bool
        return opt is Some(1_000)
    end
    "#;

    // Verify AST
    let items = parse_to_items(source).expect("parse nested patterns");
    assert_eq!(items.len(), 6);

    // check_some_hex -> Pattern::Enum { variant: "Some", bindings: [Pattern::Literal(Integer(15))] }
    if let ItemKind::Function(f) = &items[0].kind {
        if let StmtKind::Return(vals) = &f.body[0].kind {
            if let ExprKind::IsPattern { pattern, .. } = &vals[0].kind {
                match pattern {
                    Pattern::Enum { variant, bindings, .. } => {
                        assert_eq!(variant, "Some");
                        assert_eq!(bindings.len(), 1);
                        assert_eq!(bindings[0], Pattern::Literal(Literal::Integer(15)));
                    }
                    other => panic!("Expected Pattern::Enum, got {:?}", other),
                }
            } else {
                panic!("Expected IsPattern");
            }
        } else {
            panic!("Expected StmtKind::Return");
        }
    }

    // check_some_min -> Pattern::Enum { variant: "Some", bindings: [Pattern::Literal(Integer(i64::MIN))] }
    if let ItemKind::Function(f) = &items[3].kind {
        if let StmtKind::Return(vals) = &f.body[0].kind {
            if let ExprKind::IsPattern { pattern, .. } = &vals[0].kind {
                match pattern {
                    Pattern::Enum { variant, bindings, .. } => {
                        assert_eq!(variant, "Some");
                        assert_eq!(bindings.len(), 1);
                        assert_eq!(bindings[0], Pattern::Literal(Literal::Integer(i64::MIN)));
                    }
                    other => panic!("Expected Pattern::Enum, got {:?}", other),
                }
            } else {
                panic!("Expected IsPattern");
            }
        } else {
            panic!("Expected StmtKind::Return");
        }
    }

    // Compile and run in VM
    let funcs = compile_source(source).expect("compile enum functions");
    let mut vm = VM::new();
    vm.load_functions(funcs);

    // check_some_hex (0x0F = 15)
    assert_eq!(
        vm.call("check_some_hex", vec![Value::some(Value::Int(15))]).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        vm.call("check_some_hex", vec![Value::some(Value::Int(16))]).unwrap(),
        Value::Bool(false)
    );
    assert_eq!(
        vm.call("check_some_hex", vec![Value::none()]).unwrap(),
        Value::Bool(false)
    );

    // check_some_bin (0b1010 = 10)
    assert_eq!(
        vm.call("check_some_bin", vec![Value::some(Value::Int(10))]).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        vm.call("check_some_bin", vec![Value::some(Value::Int(9))]).unwrap(),
        Value::Bool(false)
    );
    assert_eq!(
        vm.call("check_some_bin", vec![Value::none()]).unwrap(),
        Value::Bool(false)
    );

    // check_some_oct (0o755 = 493)
    assert_eq!(
        vm.call("check_some_oct", vec![Value::some(Value::Int(493))]).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        vm.call("check_some_oct", vec![Value::some(Value::Int(400))]).unwrap(),
        Value::Bool(false)
    );

    // check_some_min (-9223372036854775808 = i64::MIN)
    assert_eq!(
        vm.call("check_some_min", vec![Value::some(Value::Int(i64::MIN))]).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        vm.call("check_some_min", vec![Value::some(Value::Int(0))]).unwrap(),
        Value::Bool(false)
    );

    // check_some_neg1 (-1)
    assert_eq!(
        vm.call("check_some_neg1", vec![Value::some(Value::Int(-1))]).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        vm.call("check_some_neg1", vec![Value::some(Value::Int(1))]).unwrap(),
        Value::Bool(false)
    );

    // check_some_dec (1_000 = 1000)
    assert_eq!(
        vm.call("check_some_dec", vec![Value::some(Value::Int(1000))]).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        vm.call("check_some_dec", vec![Value::some(Value::Int(999))]).unwrap(),
        Value::Bool(false)
    );
}

#[test]
fn test_vm_execution_branching_accuracy() {
    // 1. Branching with primitive integer patterns
    let src_branch = r#"
    function classify_val(x: int): int
        if x is 0x0F then
            return 1
        elseif x is 0b1010 then
            return 2
        elseif x is 0o755 then
            return 3
        elseif x is 1_000 then
            return 4
        elseif x is -1 then
            return 5
        elseif x is -9223372036854775808 then
            return 6
        else
            return 0
        end
    end
    "#;

    let funcs = compile_source(src_branch).expect("compile classify_val");
    let mut vm = VM::new();
    vm.load_functions(funcs);

    assert_eq!(vm.call("classify_val", vec![Value::Int(15)]).unwrap(), Value::Int(1));
    assert_eq!(vm.call("classify_val", vec![Value::Int(10)]).unwrap(), Value::Int(2));
    assert_eq!(vm.call("classify_val", vec![Value::Int(493)]).unwrap(), Value::Int(3));
    assert_eq!(vm.call("classify_val", vec![Value::Int(1000)]).unwrap(), Value::Int(4));
    assert_eq!(vm.call("classify_val", vec![Value::Int(-1)]).unwrap(), Value::Int(5));
    assert_eq!(vm.call("classify_val", vec![Value::Int(i64::MIN)]).unwrap(), Value::Int(6));
    assert_eq!(vm.call("classify_val", vec![Value::Int(42)]).unwrap(), Value::Int(0));

    // 2. Branching with Option<int> nested patterns
    let src_opt_branch = r#"
    function classify_opt(opt: Option<int>): int
        if opt is Some(0x0F) then
            return 10
        elseif opt is Some(0b1010) then
            return 20
        elseif opt is Some(0o755) then
            return 30
        elseif opt is Some(1_000) then
            return 40
        elseif opt is Some(-1) then
            return 50
        elseif opt is Some(-9223372036854775808) then
            return 60
        elseif opt is None then
            return 99
        else
            return 0
        end
    end
    "#;

    let funcs_opt = compile_source(src_opt_branch).expect("compile classify_opt");
    let mut vm_opt = VM::new();
    vm_opt.load_functions(funcs_opt);

    assert_eq!(vm_opt.call("classify_opt", vec![Value::some(Value::Int(15))]).unwrap(), Value::Int(10));
    assert_eq!(vm_opt.call("classify_opt", vec![Value::some(Value::Int(10))]).unwrap(), Value::Int(20));
    assert_eq!(vm_opt.call("classify_opt", vec![Value::some(Value::Int(493))]).unwrap(), Value::Int(30));
    assert_eq!(vm_opt.call("classify_opt", vec![Value::some(Value::Int(1000))]).unwrap(), Value::Int(40));
    assert_eq!(vm_opt.call("classify_opt", vec![Value::some(Value::Int(-1))]).unwrap(), Value::Int(50));
    assert_eq!(vm_opt.call("classify_opt", vec![Value::some(Value::Int(i64::MIN))]).unwrap(), Value::Int(60));
    assert_eq!(vm_opt.call("classify_opt", vec![Value::none()]).unwrap(), Value::Int(99));
    assert_eq!(vm_opt.call("classify_opt", vec![Value::some(Value::Int(777))]).unwrap(), Value::Int(0));
}

#[test]
fn test_adversarial_counterexamples_and_teeth() {
    // 1. Negative radix patterns: -0x0F, -0b1010, -0o755
    let src_neg_radix = r#"
    function check_neg_hex(x: int): bool
        return x is -0x0F
    end
    function check_neg_bin(x: int): bool
        return x is -0b1010
    end
    function check_neg_oct(x: int): bool
        return x is -0o755
    end
    function check_neg_zero(x: int): bool
        return x is -0
    end
    "#;
    let funcs = compile_source(src_neg_radix).expect("compile negative radix patterns");
    let mut vm = VM::new();
    vm.load_functions(funcs);

    assert_eq!(vm.call("check_neg_hex", vec![Value::Int(-15)]).unwrap(), Value::Bool(true));
    assert_eq!(vm.call("check_neg_hex", vec![Value::Int(15)]).unwrap(), Value::Bool(false));

    assert_eq!(vm.call("check_neg_bin", vec![Value::Int(-10)]).unwrap(), Value::Bool(true));
    assert_eq!(vm.call("check_neg_bin", vec![Value::Int(10)]).unwrap(), Value::Bool(false));

    assert_eq!(vm.call("check_neg_oct", vec![Value::Int(-493)]).unwrap(), Value::Bool(true));
    assert_eq!(vm.call("check_neg_oct", vec![Value::Int(493)]).unwrap(), Value::Bool(false));

    assert_eq!(vm.call("check_neg_zero", vec![Value::Int(0)]).unwrap(), Value::Bool(true));
    assert_eq!(vm.call("check_neg_zero", vec![Value::Int(1)]).unwrap(), Value::Bool(false));

    // 2. Nested enum with negative radix pattern: Some(-0x0F)
    let src_nested_neg = r#"
    function check_some_neg_hex(opt: Option<int>): bool
        return opt is Some(-0x0F)
    end
    "#;
    let funcs_nested = compile_source(src_nested_neg).expect("compile nested neg hex");
    let mut vm_nested = VM::new();
    vm_nested.load_functions(funcs_nested);

    assert_eq!(vm_nested.call("check_some_neg_hex", vec![Value::some(Value::Int(-15))]).unwrap(), Value::Bool(true));
    assert_eq!(vm_nested.call("check_some_neg_hex", vec![Value::some(Value::Int(15))]).unwrap(), Value::Bool(false));
    assert_eq!(vm_nested.call("check_some_neg_hex", vec![Value::none()]).unwrap(), Value::Bool(false));

    // 3. Two's complement radix boundary patterns:
    // 0xFFFF_FFFF_FFFF_FFFF parses to -1
    // 0x8000_0000_0000_0000 parses to i64::MIN
    // 0x7FFF_FFFF_FFFF_FFFF parses to i64::MAX
    let src_twos_comp = r#"
    function check_hex_neg1(x: int): bool
        return x is 0xFFFF_FFFF_FFFF_FFFF
    end
    function check_hex_min(x: int): bool
        return x is 0x8000_0000_0000_0000
    end
    function check_hex_max(x: int): bool
        return x is 0x7FFF_FFFF_FFFF_FFFF
    end
    "#;
    let funcs_tc = compile_source(src_twos_comp).expect("compile two's comp radix patterns");
    let mut vm_tc = VM::new();
    vm_tc.load_functions(funcs_tc);

    assert_eq!(vm_tc.call("check_hex_neg1", vec![Value::Int(-1)]).unwrap(), Value::Bool(true));
    assert_eq!(vm_tc.call("check_hex_neg1", vec![Value::Int(0)]).unwrap(), Value::Bool(false));

    assert_eq!(vm_tc.call("check_hex_min", vec![Value::Int(i64::MIN)]).unwrap(), Value::Bool(true));
    assert_eq!(vm_tc.call("check_hex_min", vec![Value::Int(0)]).unwrap(), Value::Bool(false));

    assert_eq!(vm_tc.call("check_hex_max", vec![Value::Int(i64::MAX)]).unwrap(), Value::Bool(true));
    assert_eq!(vm_tc.call("check_hex_max", vec![Value::Int(-1)]).unwrap(), Value::Bool(false));

    // 4. Floating point patterns: 3.14, -3.14, Some(-2.5)
    let src_float_pat = r#"
    function check_float_pos(f: float): bool
        return f is 3.14
    end
    function check_float_neg(f: float): bool
        return f is -3.14
    end
    function check_some_float_neg(opt: Option<float>): bool
        return opt is Some(-2.5)
    end
    "#;
    let funcs_flt = compile_source(src_float_pat).expect("compile float patterns");
    let mut vm_flt = VM::new();
    vm_flt.load_functions(funcs_flt);

    assert_eq!(vm_flt.call("check_float_pos", vec![Value::Float(3.14)]).unwrap(), Value::Bool(true));
    assert_eq!(vm_flt.call("check_float_pos", vec![Value::Float(3.15)]).unwrap(), Value::Bool(false));

    assert_eq!(vm_flt.call("check_float_neg", vec![Value::Float(-3.14)]).unwrap(), Value::Bool(true));
    assert_eq!(vm_flt.call("check_float_neg", vec![Value::Float(3.14)]).unwrap(), Value::Bool(false));

    assert_eq!(vm_flt.call("check_some_float_neg", vec![Value::some(Value::Float(-2.5))]).unwrap(), Value::Bool(true));
    assert_eq!(vm_flt.call("check_some_float_neg", vec![Value::some(Value::Float(2.5))]).unwrap(), Value::Bool(false));
    assert_eq!(vm_flt.call("check_some_float_neg", vec![Value::none()]).unwrap(), Value::Bool(false));

    // 5. Non-regression of TypeCheck with primitive types: x is int, x is float, x is string
    let src_type_check = r#"
    function check_types(x: int): int
        if x is int then
            return 1
        elseif x is float then
            return 2
        elseif x is string then
            return 3
        else
            return 0
        end
    end
    "#;
    let funcs_types = compile_source(src_type_check).expect("compile type check");
    let mut vm_types = VM::new();
    vm_types.load_functions(funcs_types);
    assert_eq!(vm_types.call("check_types", vec![Value::Int(42)]).unwrap(), Value::Int(1));

    // 6. Teeth: syntax errors on invalid pattern constructions
    // 6a. Dangling minus in pattern
    let err_dangling = parse_to_items("function f(x: int): bool return x is - end").unwrap_err();
    assert!(format!("{}", err_dangling).contains("Expected integer or float literal after '-' in pattern")
        || format!("{}", err_dangling).contains("syntax")
        || format!("{}", err_dangling).contains("Expected"),
        "Dangling minus must error: {}", err_dangling);

    // 6b. Invalid operand after minus in pattern (e.g. identifier)
    let err_invalid_op = parse_to_items("function f(x: int): bool return x is -abc end").unwrap_err();
    assert!(format!("{}", err_invalid_op).contains("Expected integer or float literal after '-' in pattern"),
        "Expected error on -abc: {}", err_invalid_op);

    // 6c. Standalone positive overflow in pattern: x is 9223372036854775808
    let err_pat_overflow = parse_to_items("function f(x: int): bool return x is 9223372036854775808 end").unwrap_err();
    assert!(format!("{}", err_pat_overflow).contains("exceeds i64::MAX"),
        "Pattern overflow must error: {}", err_pat_overflow);

    // 6d. Pattern negation overflow: x is -0x8000_0000_0000_0000
    // (0x8000_0000_0000_0000 is i64::MIN, so -i64::MIN overflows positive i64)
    let err_neg_min_overflow = parse_to_items("function f(x: int): bool return x is -0x8000_0000_0000_0000 end").unwrap_err();
    assert!(format!("{}", err_neg_min_overflow).contains("Integer overflow in pattern negation"),
        "Negation of i64::MIN in pattern must overflow: {}", err_neg_min_overflow);
}


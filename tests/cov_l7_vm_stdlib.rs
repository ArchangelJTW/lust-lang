#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use lust::bytecode::{NativeCallResult, Value, ValueKey, native_fn};
use lust::config::LustConfig;
use lust::number::LustInt;
use lust::vm::VM;
use std::f64::consts::PI;
use std::fs;
use std::rc::Rc;

// =========================================================================
// TEST HARNESS & HELPERS
// =========================================================================

fn make_vm() -> VM {
    VM::new()
}

fn make_vm_with_io_os() -> VM {
    let mut config = LustConfig::default();
    config.enable_module("io");
    config.enable_module("os");
    VM::with_config(&config)
}

fn call_global(vm: &mut VM, name: &str, args: Vec<Value>) -> lust::Result<Value> {
    let func = vm
        .get_global(name)
        .ok_or_else(|| lust::LustError::RuntimeError {
            message: format!("Global '{}' not found", name),
        })?;
    vm.call_value(&func, args)
}

fn call_map_entry(
    vm: &mut VM,
    module_name: &str,
    func_name: &str,
    args: Vec<Value>,
) -> lust::Result<Value> {
    let module = vm
        .get_global(module_name)
        .ok_or_else(|| lust::LustError::RuntimeError {
            message: format!("Module '{}' not found", module_name),
        })?;
    let key = ValueKey::string(func_name.to_string());
    let func = match &module {
        Value::Map(m) => m
            .borrow()
            .get(&key)
            .cloned()
            .ok_or_else(|| lust::LustError::RuntimeError {
                message: format!("Function '{}.{}' not found in map", module_name, func_name),
            })?,
        _ => {
            return Err(lust::LustError::RuntimeError {
                message: format!("Module '{}' is not a map", module_name),
            });
        }
    };
    vm.call_value(&func, args)
}

fn assert_float_close(actual: f64, expected: f64, eps: f64, func_name: &str, input: &str) {
    if actual.is_nan() && expected.is_nan() {
        return;
    }
    if actual.is_infinite() && expected.is_infinite() {
        assert_eq!(
            actual.is_sign_positive(),
            expected.is_sign_positive(),
            "Sign mismatch in {}({}): actual={}, expected={}",
            func_name,
            input,
            actual,
            expected
        );
        return;
    }
    let diff = (actual - expected).abs();
    assert!(
        diff <= eps,
        "Float mismatch in {}({}): actual={}, expected={}, diff={} > eps={}",
        func_name,
        input,
        actual,
        expected,
        diff,
        eps
    );
}

// =========================================================================
// 1. MATH LIBRARY & BOUNDARIES
// =========================================================================

#[test]
fn test_math_constants_and_properties() {
    let vm = make_vm();
    let math_mod = vm.get_global("math").expect("math module found");
    match math_mod {
        Value::Map(m) => {
            let map = m.borrow();
            let pi_val = map.get(&ValueKey::string("pi".to_string())).expect("math.pi");
            assert_eq!(pi_val, &Value::Float(PI), "math.pi must equal std::f64::consts::PI");

            let huge_val = map.get(&ValueKey::string("huge".to_string())).expect("math.huge");
            assert_eq!(huge_val, &Value::Float(f64::INFINITY), "math.huge must be f64::INFINITY");

            let max_int = map.get(&ValueKey::string("maxinteger".to_string())).expect("math.maxinteger");
            assert_eq!(max_int, &Value::Int(LustInt::MAX), "math.maxinteger must be LustInt::MAX");

            let min_int = map.get(&ValueKey::string("mininteger".to_string())).expect("math.mininteger");
            assert_eq!(min_int, &Value::Int(LustInt::MIN), "math.mininteger must be LustInt::MIN");
        }
        other => panic!("math module must be a Map, got {:?}", other),
    }
}

#[test]
fn test_math_abs_positive_negative_and_boundaries() {
    let mut vm = make_vm();

    // Integer cases
    for i in [-100_i64, -1, 0, 1, 42, LustInt::MAX] {
        let res = call_global(&mut vm, "math.abs", vec![Value::Int(i)]).expect("math.abs(int)");
        let expected = Value::Int(i.wrapping_abs());
        assert_eq!(res, expected, "math.abs({}) must be {}", i, expected);
    }
    // LustInt::MIN wrapping
    let min_res = call_global(&mut vm, "math.abs", vec![Value::Int(LustInt::MIN)]).expect("math.abs(MIN)");
    assert_eq!(min_res, Value::Int(LustInt::MIN), "math.abs(LustInt::MIN) wrapping_abs");

    // Float cases
    for f in [-100.5_f64, -0.0, 0.0, 3.14159, 1e20] {
        let res = call_global(&mut vm, "math.abs", vec![Value::Float(f)]).expect("math.abs(float)");
        let expected = Value::Float(f.abs());
        assert_eq!(res, expected, "math.abs({}) must be {}", f, expected);
    }

    // Negative case: type mismatch
    let err = call_global(&mut vm, "math.abs", vec![Value::string("invalid")]);
    assert!(err.is_err(), "math.abs('invalid') must return Err");
}

#[test]
fn test_math_floor_ceil_round() {
    let mut vm = make_vm();
    let cases = vec![
        (3.7_f64, 3_i64, 4_i64, 4.0_f64),
        (3.2_f64, 3_i64, 4_i64, 3.0_f64),
        (-3.7_f64, -4_i64, -3_i64, -4.0_f64),
        (-3.2_f64, -4_i64, -3_i64, -3.0_f64),
        (0.0_f64, 0_i64, 0_i64, 0.0_f64),
        (2.5_f64, 2_i64, 3_i64, 3.0_f64),
    ];

    for (input, exp_floor, exp_ceil, exp_round) in cases {
        let fl = call_global(&mut vm, "math.floor", vec![Value::Float(input)]).expect("math.floor");
        assert_eq!(fl, Value::Int(exp_floor), "math.floor({}) expected {}", input, exp_floor);

        let ce = call_global(&mut vm, "math.ceil", vec![Value::Float(input)]).expect("math.ceil");
        assert_eq!(ce, Value::Int(exp_ceil), "math.ceil({}) expected {}", input, exp_ceil);

        let rd = call_global(&mut vm, "math.round", vec![Value::Float(input)]).expect("math.round");
        assert_eq!(rd, Value::Float(exp_round), "math.round({}) expected {}", input, exp_round);
    }

    // Integer passthrough
    for i in [-10_i64, 0, 10] {
        assert_eq!(call_global(&mut vm, "math.floor", vec![Value::Int(i)]).unwrap(), Value::Int(i));
        assert_eq!(call_global(&mut vm, "math.ceil", vec![Value::Int(i)]).unwrap(), Value::Int(i));
        assert_eq!(call_global(&mut vm, "math.round", vec![Value::Int(i)]).unwrap(), Value::Int(i));
    }

    // Negative case
    assert!(call_global(&mut vm, "math.floor", vec![Value::Bool(true)]).is_err());
    assert!(call_global(&mut vm, "math.ceil", vec![Value::Bool(false)]).is_err());
    assert!(call_global(&mut vm, "math.round", vec![Value::Nil]).is_err());
}

#[test]
fn test_math_trigonometry_and_sqrt() {
    let mut vm = make_vm();

    // Sqrt
    let sqrt_cases = vec![(0.0_f64, 0.0_f64), (4.0, 2.0), (9.0, 3.0), (2.0, std::f64::consts::SQRT_2)];
    for (val, expected) in sqrt_cases {
        let res = call_global(&mut vm, "math.sqrt", vec![Value::Float(val)]).expect("math.sqrt");
        if let Value::Float(f) = res {
            assert_float_close(f, expected, 1e-12, "math.sqrt", &val.to_string());
        } else {
            panic!("math.sqrt must return Float");
        }
    }
    // Sqrt of int
    let sqrt_int = call_global(&mut vm, "math.sqrt", vec![Value::Int(16)]).expect("math.sqrt(int)");
    assert_eq!(sqrt_int, Value::Float(4.0));

    // Trigonometry
    let angles = [0.0_f64, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0, PI];
    for &angle in &angles {
        let sin_val = call_global(&mut vm, "math.sin", vec![Value::Float(angle)]).unwrap().as_float().unwrap();
        assert_float_close(sin_val, angle.sin(), 1e-12, "math.sin", &angle.to_string());

        let cos_val = call_global(&mut vm, "math.cos", vec![Value::Float(angle)]).unwrap().as_float().unwrap();
        assert_float_close(cos_val, angle.cos(), 1e-12, "math.cos", &angle.to_string());

        if (angle - PI / 2.0).abs() > 1e-6 {
            let tan_val = call_global(&mut vm, "math.tan", vec![Value::Float(angle)]).unwrap().as_float().unwrap();
            assert_float_close(tan_val, angle.tan(), 1e-12, "math.tan", &angle.to_string());
        }
    }

    // Inverse trig
    let asin_val = call_global(&mut vm, "math.asin", vec![Value::Float(1.0)]).unwrap().as_float().unwrap();
    assert_float_close(asin_val, 1.0_f64.asin(), 1e-12, "math.asin", "1.0");

    let acos_val = call_global(&mut vm, "math.acos", vec![Value::Float(1.0)]).unwrap().as_float().unwrap();
    assert_float_close(acos_val, 1.0_f64.acos(), 1e-12, "math.acos", "1.0");

    let atan_val = call_global(&mut vm, "math.atan", vec![Value::Float(1.0)]).unwrap().as_float().unwrap();
    assert_float_close(atan_val, 1.0_f64.atan(), 1e-12, "math.atan", "1.0");

    let atan2_val = call_global(&mut vm, "math.atan2", vec![Value::Float(1.0), Value::Float(1.0)]).unwrap().as_float().unwrap();
    assert_float_close(atan2_val, 1.0_f64.atan2(1.0), 1e-12, "math.atan2", "1.0, 1.0");

    // Negative cases
    assert!(call_global(&mut vm, "math.sqrt", vec![Value::string("xyz")]).is_err());
    assert!(call_global(&mut vm, "math.sin", vec![Value::Nil]).is_err());
    assert!(call_global(&mut vm, "math.atan2", vec![Value::Float(1.0)]).is_err(), "atan2 requires 2 args");
}

#[test]
fn test_math_min_max_clamp() {
    let mut vm = make_vm();

    // Min / Max with ints
    let min_int = call_global(&mut vm, "math.min", vec![Value::Int(10), Value::Int(3), Value::Int(20)]).unwrap();
    assert_eq!(min_int, Value::Int(3));

    let max_int = call_global(&mut vm, "math.max", vec![Value::Int(10), Value::Int(3), Value::Int(20)]).unwrap();
    assert_eq!(max_int, Value::Int(20));

    // Min / Max with floats
    let min_fl = call_global(&mut vm, "math.min", vec![Value::Float(10.5), Value::Float(3.2), Value::Float(20.1)]).unwrap();
    assert_eq!(min_fl, Value::Float(3.2));

    let max_fl = call_global(&mut vm, "math.max", vec![Value::Float(10.5), Value::Float(3.2), Value::Float(20.1)]).unwrap();
    assert_eq!(max_fl, Value::Float(20.1));

    // Mixed min / max
    let min_mixed = call_global(&mut vm, "math.min", vec![Value::Int(5), Value::Float(2.5)]).unwrap();
    assert_eq!(min_mixed, Value::Float(2.5));

    let max_mixed = call_global(&mut vm, "math.max", vec![Value::Int(5), Value::Float(2.5)]).unwrap();
    assert_eq!(max_mixed, Value::Int(5));

    // Clamp
    let clamp_in = call_global(&mut vm, "math.clamp", vec![Value::Int(5), Value::Int(0), Value::Int(10)]).unwrap();
    assert_eq!(clamp_in, Value::Int(5));

    let clamp_low = call_global(&mut vm, "math.clamp", vec![Value::Int(-5), Value::Int(0), Value::Int(10)]).unwrap();
    assert_eq!(clamp_low, Value::Int(0));

    let clamp_high = call_global(&mut vm, "math.clamp", vec![Value::Int(15), Value::Int(0), Value::Int(10)]).unwrap();
    assert_eq!(clamp_high, Value::Int(10));

    let clamp_fl = call_global(&mut vm, "math.clamp", vec![Value::Float(15.5), Value::Float(0.0), Value::Float(10.0)]).unwrap();
    assert_eq!(clamp_fl, Value::Float(10.0));

    // Negative cases
    assert!(call_global(&mut vm, "math.min", vec![]).is_err(), "min requires at least 1 arg");
    assert!(call_global(&mut vm, "math.max", vec![]).is_err(), "max requires at least 1 arg");
    assert!(call_global(&mut vm, "math.clamp", vec![Value::Int(1), Value::Int(2)]).is_err(), "clamp requires 3 args");
    assert!(call_global(&mut vm, "math.min", vec![Value::Int(1), Value::string("bad")]).is_err());
}

#[test]
fn test_math_exp_log_pow_mod_deg_rad() {
    let mut vm = make_vm();

    // Exp & Log
    let exp_res = call_global(&mut vm, "math.exp", vec![Value::Float(1.0)]).unwrap().as_float().unwrap();
    assert_float_close(exp_res, std::f64::consts::E, 1e-12, "math.exp", "1.0");

    let log_e = call_global(&mut vm, "math.log", vec![Value::Float(std::f64::consts::E)]).unwrap().as_float().unwrap();
    assert_float_close(log_e, 1.0, 1e-12, "math.log", "E");

    let log_10 = call_global(&mut vm, "math.log", vec![Value::Float(100.0), Value::Float(10.0)]).unwrap().as_float().unwrap();
    assert_float_close(log_10, 2.0, 1e-12, "math.log", "100, base 10");

    // Pow
    let pow_res = call_global(&mut vm, "math.pow", vec![Value::Float(2.0), Value::Float(8.0)]).unwrap().as_float().unwrap();
    assert_float_close(pow_res, 256.0, 1e-12, "math.pow", "2, 8");

    let pow_int = call_global(&mut vm, "math.pow", vec![Value::Int(3), Value::Int(3)]).unwrap().as_float().unwrap();
    assert_float_close(pow_int, 27.0, 1e-12, "math.pow", "3, 3");

    // Mod
    let mod_pos = call_global(&mut vm, "math.mod", vec![Value::Int(7), Value::Int(3)]).unwrap();
    assert_eq!(mod_pos, Value::Int(1));

    let mod_neg1 = call_global(&mut vm, "math.mod", vec![Value::Int(-7), Value::Int(3)]).unwrap();
    assert_eq!(mod_neg1, Value::Int(2));

    let mod_neg2 = call_global(&mut vm, "math.mod", vec![Value::Int(7), Value::Int(-3)]).unwrap();
    assert_eq!(mod_neg2, Value::Int(-2));

    let mod_div_neg1 = call_global(&mut vm, "math.mod", vec![Value::Int(10), Value::Int(-1)]).unwrap();
    assert_eq!(mod_div_neg1, Value::Int(0));

    // Division by zero in math.mod
    let mod_zero = call_global(&mut vm, "math.mod", vec![Value::Int(7), Value::Int(0)]);
    assert!(mod_zero.is_err(), "math.mod by zero must fail");

    // Deg and Rad
    let deg_res = call_global(&mut vm, "math.deg", vec![Value::Float(PI)]).unwrap().as_float().unwrap();
    assert_float_close(deg_res, 180.0, 1e-12, "math.deg", "PI");

    let rad_res = call_global(&mut vm, "math.rad", vec![Value::Float(180.0)]).unwrap().as_float().unwrap();
    assert_float_close(rad_res, PI, 1e-12, "math.rad", "180.0");
}

#[test]
fn test_math_tointeger_and_tofloat() {
    let mut vm = make_vm();

    // tointeger
    let to_int_i = call_global(&mut vm, "math.tointeger", vec![Value::Int(42)]).unwrap();
    assert_eq!(to_int_i, Value::some(Value::Int(42)));

    let to_int_f_whole = call_global(&mut vm, "math.tointeger", vec![Value::Float(42.0)]).unwrap();
    assert_eq!(to_int_f_whole, Value::some(Value::Int(42)));

    let to_int_f_frac = call_global(&mut vm, "math.tointeger", vec![Value::Float(42.5)]).unwrap();
    assert_eq!(to_int_f_frac, Value::none());

    let to_int_s = call_global(&mut vm, "math.tointeger", vec![Value::string(" 123 ")]).unwrap();
    assert_eq!(to_int_s, Value::some(Value::Int(123)));

    let to_int_s_bad = call_global(&mut vm, "math.tointeger", vec![Value::string("not_int")]).unwrap();
    assert_eq!(to_int_s_bad, Value::none());

    let to_int_nil = call_global(&mut vm, "math.tointeger", vec![Value::Nil]).unwrap();
    assert_eq!(to_int_nil, Value::none());

    // tofloat
    let to_fl_f = call_global(&mut vm, "math.tofloat", vec![Value::Float(3.14)]).unwrap();
    assert_eq!(to_fl_f, Value::some(Value::Float(3.14)));

    let to_fl_i = call_global(&mut vm, "math.tofloat", vec![Value::Int(42)]).unwrap();
    assert_eq!(to_fl_i, Value::some(Value::Float(42.0)));

    let to_fl_s = call_global(&mut vm, "math.tofloat", vec![Value::string(" 2.718 ")]).unwrap();
    assert_eq!(to_fl_s, Value::some(Value::Float(2.718)));

    let to_fl_bad = call_global(&mut vm, "math.tofloat", vec![Value::string("not_fl")]).unwrap();
    assert_eq!(to_fl_bad, Value::none());
}

#[test]
fn test_math_random_and_seed() {
    let mut vm = make_vm();

    // Seed
    let seed_res = call_global(&mut vm, "math.randomseed", vec![Value::Int(12345)]).unwrap();
    assert_eq!(seed_res, Value::Nil);

    // Random float [0, 1)
    let r1 = call_global(&mut vm, "math.random", vec![]).unwrap().as_float().unwrap();
    assert!((0.0..1.0).contains(&r1), "math.random() must be in [0, 1), got {}", r1);

    // Random int 1..=m
    let r2 = call_global(&mut vm, "math.random", vec![Value::Int(10)]).unwrap().as_int().unwrap();
    assert!((1..=10).contains(&r2), "math.random(10) must be in [1, 10], got {}", r2);

    // Random int lo..=hi
    let r3 = call_global(&mut vm, "math.random", vec![Value::Int(20), Value::Int(30)]).unwrap().as_int().unwrap();
    assert!((20..=30).contains(&r3), "math.random(20, 30) must be in [20, 30], got {}", r3);

    // Reversed bounds
    let r4 = call_global(&mut vm, "math.random", vec![Value::Int(30), Value::Int(20)]).unwrap().as_int().unwrap();
    assert!((20..=30).contains(&r4), "math.random(30, 20) must be in [20, 30], got {}", r4);
}

// =========================================================================
// 2. BITWISE LIBRARY
// =========================================================================

#[test]
fn test_bitwise_operations() {
    let mut vm = make_vm();

    // band
    let band_res = call_global(&mut vm, "bit.band", vec![Value::Int(0xFF), Value::Int(0x0F)]).unwrap();
    assert_eq!(band_res, Value::Int(0x0F), "bit.band(0xFF, 0x0F) must be 0x0F");

    let band_multi = call_global(&mut vm, "bit.band", vec![Value::Int(0xFF), Value::Int(0xF0), Value::Int(0x30)]).unwrap();
    assert_eq!(band_multi, Value::Int(0x30), "bit.band multi-operand");

    // bor
    let bor_res = call_global(&mut vm, "bit.bor", vec![Value::Int(0xF0), Value::Int(0x0F)]).unwrap();
    assert_eq!(bor_res, Value::Int(0xFF), "bit.bor(0xF0, 0x0F) must be 0xFF");

    // bxor
    let bxor_res = call_global(&mut vm, "bit.bxor", vec![Value::Int(0xFF), Value::Int(0x0F)]).unwrap();
    assert_eq!(bxor_res, Value::Int(0xF0), "bit.bxor(0xFF, 0x0F) must be 0xF0");

    // bnot
    let bnot_res = call_global(&mut vm, "bit.bnot", vec![Value::Int(0)]).unwrap();
    assert_eq!(bnot_res, Value::Int(-1), "bit.bnot(0) must be -1");

    let bnot_one = call_global(&mut vm, "bit.bnot", vec![Value::Int(1)]).unwrap();
    assert_eq!(bnot_one, Value::Int(-2), "bit.bnot(1) must be -2");

    // lshift
    let lshift_res = call_global(&mut vm, "bit.lshift", vec![Value::Int(1), Value::Int(4)]).unwrap();
    assert_eq!(lshift_res, Value::Int(16), "bit.lshift(1, 4) must be 16");

    // rshift (logical unsigned)
    let rshift_res = call_global(&mut vm, "bit.rshift", vec![Value::Int(16), Value::Int(2)]).unwrap();
    assert_eq!(rshift_res, Value::Int(4), "bit.rshift(16, 2) must be 4");

    let rshift_neg = call_global(&mut vm, "bit.rshift", vec![Value::Int(-1), Value::Int(1)]).unwrap();
    assert_eq!(rshift_neg, Value::Int(i64::MAX), "bit.rshift(-1, 1) logical right shift");

    // arshift (arithmetic signed)
    let arshift_res = call_global(&mut vm, "bit.arshift", vec![Value::Int(16), Value::Int(2)]).unwrap();
    assert_eq!(arshift_res, Value::Int(4), "bit.arshift(16, 2) must be 4");

    let arshift_neg = call_global(&mut vm, "bit.arshift", vec![Value::Int(-16), Value::Int(2)]).unwrap();
    assert_eq!(arshift_neg, Value::Int(-4), "bit.arshift(-16, 2) must be -4");

    // Negative cases
    assert!(call_global(&mut vm, "bit.band", vec![]).is_err(), "band requires >= 1 arg");
    assert!(call_global(&mut vm, "bit.bor", vec![]).is_err(), "bor requires >= 1 arg");
    assert!(call_global(&mut vm, "bit.lshift", vec![Value::Int(1)]).is_err(), "lshift requires 2 args");
    assert!(call_global(&mut vm, "bit.band", vec![Value::string("bad")]).is_err(), "band non-int error");
}

// =========================================================================
// 3. STRING LIBRARY & UTF-8
// =========================================================================

#[test]
fn test_string_len_lower_upper_reverse() {
    let mut vm = make_vm();

    // string.len
    assert_eq!(call_global(&mut vm, "string.len", vec![Value::string("")]).unwrap(), Value::Int(0));
    assert_eq!(call_global(&mut vm, "string.len", vec![Value::string("hello")]).unwrap(), Value::Int(5));
    assert_eq!(call_global(&mut vm, "string.len", vec![Value::string("Привет")]).unwrap(), Value::Int(12)); // 6 cyrillic chars = 12 UTF-8 bytes
    assert_eq!(call_global(&mut vm, "string.len", vec![Value::Nil]).unwrap(), Value::Int(0));

    // string.lower & upper
    assert_eq!(call_global(&mut vm, "string.lower", vec![Value::string("HeLLo WoRLD")]).unwrap(), Value::string("hello world"));
    assert_eq!(call_global(&mut vm, "string.upper", vec![Value::string("hello world")]).unwrap(), Value::string("HELLO WORLD"));
    assert_eq!(call_global(&mut vm, "string.lower", vec![Value::string("ПРИВЕТ")]).unwrap(), Value::string("привет"));
    assert_eq!(call_global(&mut vm, "string.upper", vec![Value::string("привет")]).unwrap(), Value::string("ПРИВЕТ"));

    // string.reverse
    assert_eq!(call_global(&mut vm, "string.reverse", vec![Value::string("abcdef")]).unwrap(), Value::string("fedcba"));

    // Negative cases
    assert!(call_global(&mut vm, "string.len", vec![Value::Int(123)]).is_err());
    assert!(call_global(&mut vm, "string.lower", vec![Value::Int(123)]).is_err());
    assert!(call_global(&mut vm, "string.upper", vec![Value::Bool(true)]).is_err());
    assert!(call_global(&mut vm, "string.reverse", vec![Value::Nil]).is_err());
}

#[test]
fn test_string_sub_and_slice() {
    let mut vm = make_vm();
    let s = Value::string("hello world");

    // 1-based sub
    assert_eq!(call_global(&mut vm, "string.sub", vec![s.clone(), Value::Int(1), Value::Int(5)]).unwrap(), Value::string("hello"));
    assert_eq!(call_global(&mut vm, "string.sub", vec![s.clone(), Value::Int(7)]).unwrap(), Value::string("world"));
    assert_eq!(call_global(&mut vm, "string.sub", vec![s.clone(), Value::Int(-5)]).unwrap(), Value::string("world"));
    assert_eq!(call_global(&mut vm, "string.sub", vec![s.clone(), Value::Int(10), Value::Int(2)]).unwrap(), Value::string(""));

    // 0-based slice
    let slice_fn = vm.get_global("string.slice").unwrap();
    assert_eq!(vm.call_value(&slice_fn, vec![s.clone(), Value::Int(0), Value::Int(5)]).unwrap(), Value::string("hello"));
    assert_eq!(vm.call_value(&slice_fn, vec![s.clone(), Value::Int(6), Value::Int(11)]).unwrap(), Value::string("world"));
    assert_eq!(vm.call_value(&slice_fn, vec![s.clone(), Value::Int(0), Value::Int(11)]).unwrap(), Value::string("hello world"));
}

#[test]
fn test_string_byte_and_char() {
    let mut vm = make_vm();

    // char
    let char_res = call_global(&mut vm, "string.char", vec![Value::Int(65), Value::Int(66), Value::Int(67)]).unwrap();
    assert_eq!(char_res, Value::string("ABC"));

    // byte
    let byte_res = call_global(&mut vm, "string.byte", vec![Value::string("ABC"), Value::Int(1), Value::Int(3)]).unwrap();
    // byte returns lua values packed in array
    if let Value::Array(arr) = byte_res {
        let b = arr.borrow();
        assert_eq!(b.len(), 3);
    } else {
        panic!("string.byte must return array of lua values");
    }

    // Negative cases
    assert!(call_global(&mut vm, "string.char", vec![Value::Int(300)]).is_err(), "codepoint > 255 must fail");
    assert!(call_global(&mut vm, "string.char", vec![Value::Int(-1)]).is_err(), "negative codepoint must fail");
}

#[test]
fn test_string_format_and_specifiers() {
    let mut vm = make_vm();

    let fmt_s = call_global(&mut vm, "string.format", vec![Value::string("hello %s"), Value::string("world")]).unwrap();
    assert_eq!(fmt_s, Value::string("hello world"));

    let fmt_d = call_global(&mut vm, "string.format", vec![Value::string("num: %05d"), Value::Int(42)]).unwrap();
    assert_eq!(fmt_d, Value::string("num: 00042"));

    let fmt_x = call_global(&mut vm, "string.format", vec![Value::string("hex: 0x%04x"), Value::Int(255)]).unwrap();
    assert_eq!(fmt_x, Value::string("hex: 0x00ff"));

    let fmt_upper_x = call_global(&mut vm, "string.format", vec![Value::string("HEX: 0x%X"), Value::Int(255)]).unwrap();
    assert_eq!(fmt_upper_x, Value::string("HEX: 0xFF"));

    let fmt_f = call_global(&mut vm, "string.format", vec![Value::string("float: %.2f"), Value::Float(3.14159)]).unwrap();
    assert_eq!(fmt_f, Value::string("float: 3.14"));

    let fmt_percent = call_global(&mut vm, "string.format", vec![Value::string("100%% %s"), Value::string("pure")]).unwrap();
    assert_eq!(fmt_percent, Value::string("100% pure"));

    let fmt_c = call_global(&mut vm, "string.format", vec![Value::string("char: %c"), Value::Int(65)]).unwrap();
    assert_eq!(fmt_c, Value::string("char: A"));

    // Incomplete format specifier error
    assert!(call_global(&mut vm, "string.format", vec![Value::string("incomplete %")]).is_err());
}

#[test]
fn test_string_find_match_gsub() {
    let mut vm = make_vm();

    // Plain find
    let find_plain = call_global(&mut vm, "string.find", vec![Value::string("abcdef"), Value::string("cd"), Value::Int(1), Value::Bool(true)]).unwrap();
    if let Value::Array(arr) = find_plain {
        let b = arr.borrow();
        assert!(!b.is_empty(), "find should find substring");
    }

    // Pattern find
    let find_pat = call_global(&mut vm, "string.find", vec![Value::string("foo 123 bar"), Value::string("%d+")]).unwrap();
    assert!(matches!(find_pat, Value::Array(_)));

    // Match full
    let m1 = call_global(&mut vm, "string.match", vec![Value::string("foo 123 bar"), Value::string("%d+")]).unwrap();
    assert_eq!(m1, Value::string("123"));

    // Match with capture
    let m2 = call_global(&mut vm, "string.match", vec![Value::string("key=value"), Value::string("key=(%a+)")]).unwrap();
    assert_eq!(m2, Value::string("value"));

    // Match not found
    let m_none = call_global(&mut vm, "string.match", vec![Value::string("hello"), Value::string("%d+")]).unwrap();
    assert_eq!(m_none, Value::Nil);

    // gsub with string replacement
    let gsub_res = call_global(&mut vm, "string.gsub", vec![Value::string("foo 123 bar 456"), Value::string("%d+"), Value::string("NUM")]).unwrap();
    if let Value::Array(arr) = gsub_res {
        let b = arr.borrow();
        // First element is replaced string
        assert!(!b.is_empty());
    }
}

#[test]
fn test_string_utilities() {
    let mut vm = make_vm();

    // rep
    let rep_res = call_global(&mut vm, "string.rep", vec![Value::string("ha"), Value::Int(3)]).unwrap();
    assert_eq!(rep_res, Value::string("hahaha"));

    let rep_sep = call_global(&mut vm, "string.rep", vec![Value::string("a"), Value::Int(3), Value::string("-")]).unwrap();
    assert_eq!(rep_sep, Value::string("a-a-a"));

    assert_eq!(call_global(&mut vm, "string.rep", vec![Value::string("a"), Value::Int(0)]).unwrap(), Value::string(""));

    // split
    let split_res = call_global(&mut vm, "string.split", vec![Value::string("a,b,c"), Value::string(",")]).unwrap();
    assert_eq!(split_res, Value::array(vec![Value::string("a"), Value::string("b"), Value::string("c")]));

    // trim
    assert_eq!(call_global(&mut vm, "string.trim", vec![Value::string("  hello  ")]).unwrap(), Value::string("hello"));
    assert_eq!(call_global(&mut vm, "string.trim_start", vec![Value::string("  hello  ")]).unwrap(), Value::string("hello  "));
    assert_eq!(call_global(&mut vm, "string.trim_end", vec![Value::string("  hello  ")]).unwrap(), Value::string("  hello"));

    // replace
    assert_eq!(call_global(&mut vm, "string.replace", vec![Value::string("banana"), Value::string("a"), Value::string("o")]).unwrap(), Value::string("bonono"));

    // starts_with, ends_with, contains, is_empty
    assert_eq!(call_global(&mut vm, "string.starts_with", vec![Value::string("hello world"), Value::string("hello")]).unwrap(), Value::Bool(true));
    assert_eq!(call_global(&mut vm, "string.starts_with", vec![Value::string("hello world"), Value::string("world")]).unwrap(), Value::Bool(false));
    assert_eq!(call_global(&mut vm, "string.ends_with", vec![Value::string("hello world"), Value::string("world")]).unwrap(), Value::Bool(true));
    assert_eq!(call_global(&mut vm, "string.contains", vec![Value::string("hello world"), Value::string("lo wo")]).unwrap(), Value::Bool(true));
    assert_eq!(call_global(&mut vm, "string.is_empty", vec![Value::string("")]).unwrap(), Value::Bool(true));
    assert_eq!(call_global(&mut vm, "string.is_empty", vec![Value::string("a")]).unwrap(), Value::Bool(false));

    // chars, lines, index_of
    assert_eq!(call_global(&mut vm, "string.chars", vec![Value::string("abc")]).unwrap(), Value::array(vec![Value::string("a"), Value::string("b"), Value::string("c")]));
    assert_eq!(call_global(&mut vm, "string.lines", vec![Value::string("a\nb\nc")]).unwrap(), Value::array(vec![Value::string("a"), Value::string("b"), Value::string("c")]));
    assert_eq!(call_global(&mut vm, "string.index_of", vec![Value::string("hello"), Value::string("ll")]).unwrap(), Value::some(Value::Int(2)));
    assert_eq!(call_global(&mut vm, "string.index_of", vec![Value::string("hello"), Value::string("xyz")]).unwrap(), Value::none());
}

// =========================================================================
// 4. ARRAY & TABLE LIBRARY
// =========================================================================

#[test]
fn test_array_basic_operations() {
    let mut vm = make_vm();
    let arr = Value::array(vec![Value::Int(10), Value::Int(20), Value::Int(30)]);

    // len & is_empty
    assert_eq!(call_global(&mut vm, "array.len", vec![arr.clone()]).unwrap(), Value::Int(3));
    assert_eq!(call_global(&mut vm, "array.is_empty", vec![arr.clone()]).unwrap(), Value::Bool(false));
    assert_eq!(call_global(&mut vm, "array.is_empty", vec![Value::array(vec![])]).unwrap(), Value::Bool(true));

    // get
    assert_eq!(call_global(&mut vm, "array.get", vec![arr.clone(), Value::Int(0)]).unwrap(), Value::some(Value::Int(10)));
    assert_eq!(call_global(&mut vm, "array.get", vec![arr.clone(), Value::Int(2)]).unwrap(), Value::some(Value::Int(30)));
    assert_eq!(call_global(&mut vm, "array.get", vec![arr.clone(), Value::Int(5)]).unwrap(), Value::none());
    assert_eq!(call_global(&mut vm, "array.get", vec![arr.clone(), Value::Int(-1)]).unwrap(), Value::none());

    // first & last
    assert_eq!(call_global(&mut vm, "array.first", vec![arr.clone()]).unwrap(), Value::some(Value::Int(10)));
    assert_eq!(call_global(&mut vm, "array.last", vec![arr.clone()]).unwrap(), Value::some(Value::Int(30)));
    assert_eq!(call_global(&mut vm, "array.first", vec![Value::array(vec![])]).unwrap(), Value::none());
    assert_eq!(call_global(&mut vm, "array.last", vec![Value::array(vec![])]).unwrap(), Value::none());

    // push & pop
    let mut_arr = Value::array(vec![Value::Int(1)]);
    call_global(&mut vm, "array.push", vec![mut_arr.clone(), Value::Int(2)]).unwrap();
    assert_eq!(call_global(&mut vm, "array.len", vec![mut_arr.clone()]).unwrap(), Value::Int(2));
    let popped = call_global(&mut vm, "array.pop", vec![mut_arr.clone()]).unwrap();
    assert_eq!(popped, Value::some(Value::Int(2)));

    // insert & remove
    call_global(&mut vm, "array.insert", vec![mut_arr.clone(), Value::Int(0), Value::Int(99)]).unwrap();
    assert_eq!(call_global(&mut vm, "array.get", vec![mut_arr.clone(), Value::Int(0)]).unwrap(), Value::some(Value::Int(99)));
    let removed = call_global(&mut vm, "array.remove", vec![mut_arr.clone(), Value::Int(0)]).unwrap();
    assert_eq!(removed, Value::some(Value::Int(99)));

    // clear
    call_global(&mut vm, "array.clear", vec![mut_arr.clone()]).unwrap();
    assert_eq!(call_global(&mut vm, "array.len", vec![mut_arr.clone()]).unwrap(), Value::Int(0));

    // slice & concat
    let full_arr = Value::array(vec![Value::string("a"), Value::string("b"), Value::string("c")]);
    let sliced = call_global(&mut vm, "array.slice", vec![full_arr.clone(), Value::Int(1), Value::Int(3)]).unwrap();
    assert_eq!(sliced, Value::array(vec![Value::string("b"), Value::string("c")]));

    let concatted = call_global(&mut vm, "array.concat", vec![full_arr.clone(), Value::string("-")]).unwrap();
    assert_eq!(concatted, Value::string("a-b-c"));

    // reverse
    let rev_arr = Value::array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    call_global(&mut vm, "array.reverse", vec![rev_arr.clone()]).unwrap();
    assert_eq!(rev_arr, Value::array(vec![Value::Int(3), Value::Int(2), Value::Int(1)]));

    // contains
    assert_eq!(call_global(&mut vm, "array.contains", vec![rev_arr.clone(), Value::Int(2)]).unwrap(), Value::Bool(true));
    assert_eq!(call_global(&mut vm, "array.contains", vec![rev_arr.clone(), Value::Int(99)]).unwrap(), Value::Bool(false));
}

#[test]
fn test_array_sort_and_higher_order() {
    let mut vm = make_vm();

    // Natural sort
    let to_sort = Value::array(vec![Value::Int(30), Value::Int(10), Value::Int(20)]);
    call_global(&mut vm, "array.sort", vec![to_sort.clone()]).unwrap();
    assert_eq!(to_sort, Value::array(vec![Value::Int(10), Value::Int(20), Value::Int(30)]));

    // Custom comparator sort (descending)
    let desc_comparator = Value::NativeFunction(native_fn(|args: &[Value]| {
        let a = args[0].as_int().unwrap();
        let b = args[1].as_int().unwrap();
        Ok(NativeCallResult::Return(Value::Bool(a > b)))
    }));
    call_global(&mut vm, "array.sort", vec![to_sort.clone(), desc_comparator]).unwrap();
    assert_eq!(to_sort, Value::array(vec![Value::Int(30), Value::Int(20), Value::Int(10)]));

    // Map
    let double_fn = Value::NativeFunction(native_fn(|args: &[Value]| {
        let n = args[0].as_int().unwrap();
        Ok(NativeCallResult::Return(Value::Int(n * 2)))
    }));
    let mapped = call_global(&mut vm, "array.map", vec![Value::array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]), double_fn]).unwrap();
    assert_eq!(mapped, Value::array(vec![Value::Int(2), Value::Int(4), Value::Int(6)]));

    // Filter
    let even_fn = Value::NativeFunction(native_fn(|args: &[Value]| {
        let n = args[0].as_int().unwrap();
        Ok(NativeCallResult::Return(Value::Bool(n % 2 == 0)))
    }));
    let filtered = call_global(&mut vm, "array.filter", vec![Value::array(vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4)]), even_fn]).unwrap();
    assert_eq!(filtered, Value::array(vec![Value::Int(2), Value::Int(4)]));

    // Reduce
    let sum_fn = Value::NativeFunction(native_fn(|args: &[Value]| {
        let acc = args[0].as_int().unwrap();
        let item = args[1].as_int().unwrap();
        Ok(NativeCallResult::Return(Value::Int(acc + item)))
    }));
    let sum = call_global(&mut vm, "array.reduce", vec![Value::array(vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4)]), Value::Int(0), sum_fn]).unwrap();
    assert_eq!(sum, Value::Int(10));

    // Iter
    let iter = call_global(&mut vm, "array.iter", vec![Value::array(vec![Value::Int(1)])]).unwrap();
    assert!(matches!(iter, Value::Iterator(_)));
}

#[test]
fn test_table_insert_remove_unpack() {
    let mut vm = make_vm();

    // 1-based table.insert and table.remove
    let tbl = Value::array(vec![Value::Int(10), Value::Int(30)]);

    // 3 args: table.insert(tbl, pos=1, value=20)
    call_global(&mut vm, "table.insert", vec![tbl.clone(), Value::Int(1), Value::Int(20)]).unwrap();
    assert_eq!(tbl, Value::array(vec![Value::Int(10), Value::Int(20), Value::Int(30)]));

    // 2 args: table.insert(tbl, value=40) -> push
    call_global(&mut vm, "table.insert", vec![tbl.clone(), Value::Int(40)]).unwrap();
    assert_eq!(tbl, Value::array(vec![Value::Int(10), Value::Int(20), Value::Int(30), Value::Int(40)]));

    // 2 args: table.remove(tbl, pos=1)
    let rem = call_global(&mut vm, "table.remove", vec![tbl.clone(), Value::Int(1)]).unwrap();
    assert_eq!(rem, Value::some(Value::Int(20)));

    // 1 arg: table.remove(tbl) -> pop
    let popped = call_global(&mut vm, "table.remove", vec![tbl.clone()]).unwrap();
    assert_eq!(popped, Value::some(Value::Int(40)));

    // table.unpack
    let unpack_res = call_global(&mut vm, "table.unpack", vec![tbl.clone(), Value::Int(1), Value::Int(2)]).unwrap();
    assert!(matches!(unpack_res, Value::Array(_)));

    // Negative cases
    assert!(call_global(&mut vm, "table.insert", vec![tbl.clone()]).is_err(), "insert requires >= 2 args");
    assert!(call_global(&mut vm, "table.insert", vec![tbl.clone(), Value::string("bad_idx"), Value::Int(1)]).is_err(), "insert with non-int index must fail");
    assert!(call_global(&mut vm, "table.insert", vec![Value::Int(123), Value::Int(1)]).is_err(), "insert on non-array must fail");
}

// =========================================================================
// 5. MAP LIBRARY
// =========================================================================

#[test]
fn test_map_crud_and_iteration() {
    let mut vm = make_vm();
    let m = Value::map(vec![].into_iter().collect());

    // len & is_empty
    assert_eq!(call_global(&mut vm, "map.len", vec![m.clone()]).unwrap(), Value::Int(0));
    assert_eq!(call_global(&mut vm, "map.is_empty", vec![m.clone()]).unwrap(), Value::Bool(true));

    // set
    call_global(&mut vm, "map.set", vec![m.clone(), Value::string("name"), Value::string("Lust")]).unwrap();
    call_global(&mut vm, "map.set", vec![m.clone(), Value::string("version"), Value::Int(3)]).unwrap();
    assert_eq!(call_global(&mut vm, "map.len", vec![m.clone()]).unwrap(), Value::Int(2));
    assert_eq!(call_global(&mut vm, "map.is_empty", vec![m.clone()]).unwrap(), Value::Bool(false));

    // get
    assert_eq!(call_global(&mut vm, "map.get", vec![m.clone(), Value::string("name")]).unwrap(), Value::some(Value::string("Lust")));
    assert_eq!(call_global(&mut vm, "map.get", vec![m.clone(), Value::string("missing")]).unwrap(), Value::none());

    // has
    assert_eq!(call_global(&mut vm, "map.has", vec![m.clone(), Value::string("name")]).unwrap(), Value::Bool(true));
    assert_eq!(call_global(&mut vm, "map.has", vec![m.clone(), Value::string("missing")]).unwrap(), Value::Bool(false));

    // keys & values
    let keys = call_global(&mut vm, "map.keys", vec![m.clone()]).unwrap();
    let values = call_global(&mut vm, "map.values", vec![m.clone()]).unwrap();
    if let (Value::Array(k), Value::Array(v)) = (keys, values) {
        assert_eq!(k.borrow().len(), 2);
        assert_eq!(v.borrow().len(), 2);
    } else {
        panic!("map.keys and map.values must return arrays");
    }

    // delete
    let del = call_global(&mut vm, "map.delete", vec![m.clone(), Value::string("name")]).unwrap();
    assert_eq!(del, Value::some(Value::string("Lust")));
    assert_eq!(call_global(&mut vm, "map.delete", vec![m.clone(), Value::string("name")]).unwrap(), Value::none());

    // iter
    let iter = call_global(&mut vm, "map.iter", vec![m.clone()]).unwrap();
    assert!(matches!(iter, Value::Iterator(_)));

    // clear
    call_global(&mut vm, "map.clear", vec![m.clone()]).unwrap();
    assert_eq!(call_global(&mut vm, "map.len", vec![m.clone()]).unwrap(), Value::Int(0));

    // Negative cases
    assert!(call_global(&mut vm, "map.get", vec![m.clone()]).is_err(), "get requires map and key");
    assert!(call_global(&mut vm, "map.set", vec![m.clone(), Value::string("k")]).is_err(), "set requires 3 args");
    assert!(call_global(&mut vm, "map.len", vec![Value::Int(123)]).is_err(), "len non-map error");
}

// =========================================================================
// 6. EFFECT LIBRARY
// =========================================================================

#[test]
fn test_effect_constructors_and_run_sync() {
    let mut vm = make_vm();

    // succeed
    let eff_ok = call_global(&mut vm, "effect.succeed", vec![Value::Int(42)]).unwrap();
    let res_ok = call_global(&mut vm, "effect.run_sync", vec![eff_ok]).unwrap();
    assert_eq!(res_ok, Value::enum_variant("Exit", "Success", vec![Value::Int(42)]));

    // fail
    let eff_fail = call_global(&mut vm, "effect.fail", vec![Value::string("boom")]).unwrap();
    let res_fail = call_global(&mut vm, "effect.run_sync", vec![eff_fail]).unwrap();
    assert_eq!(res_fail, Value::enum_variant("Exit", "Fail", vec![Value::string("boom")]));

    // die
    let eff_die = call_global(&mut vm, "effect.die", vec![Value::string("defect")]).unwrap();
    let res_die = call_global(&mut vm, "effect.run_sync", vec![eff_die]).unwrap();
    assert_eq!(res_die, Value::enum_variant("Exit", "Die", vec![Value::string("defect")]));

    // interrupt
    let eff_intr = call_global(&mut vm, "effect.interrupt", vec![]).unwrap();
    let res_intr = call_global(&mut vm, "effect.run_sync", vec![eff_intr]).unwrap();
    assert_eq!(res_intr, Value::enum_variant("Exit", "Interrupt", vec![]));

    // from_result
    let from_ok = call_global(&mut vm, "effect.from_result", vec![Value::ok(Value::Int(100))]).unwrap();
    let res_from_ok = call_global(&mut vm, "effect.run_sync", vec![from_ok]).unwrap();
    assert_eq!(res_from_ok, Value::enum_variant("Exit", "Success", vec![Value::Int(100)]));

    let from_err = call_global(&mut vm, "effect.from_result", vec![Value::err(Value::string("fail"))]).unwrap();
    let res_from_err = call_global(&mut vm, "effect.run_sync", vec![from_err]).unwrap();
    assert_eq!(res_from_err, Value::enum_variant("Exit", "Fail", vec![Value::string("fail")]));

    // suspend
    let thunk = Value::NativeFunction(native_fn(|_| Ok(NativeCallResult::Return(Value::Int(777)))));
    let eff_susp = call_global(&mut vm, "effect.suspend", vec![thunk]).unwrap();
    let res_susp = call_global(&mut vm, "effect.run_sync", vec![eff_susp]).unwrap();
    assert_eq!(res_susp, Value::enum_variant("Exit", "Success", vec![Value::Int(777)]));

    // Negative cases
    assert!(call_global(&mut vm, "effect.suspend", vec![]).is_err(), "suspend requires 1 arg");
    assert!(call_global(&mut vm, "effect.run_sync", vec![]).is_err(), "run_sync requires 1 arg");
}

// =========================================================================
// 7. CORELIB BUILTINS
// =========================================================================

#[test]
fn test_core_assert_and_error() {
    let mut vm = make_vm();

    // assert truthy
    for truthy in [Value::Bool(true), Value::Int(1), Value::Int(0), Value::string("hi"), Value::string("")] {
        let res = call_global(&mut vm, "assert", vec![truthy.clone()]).expect("assert(truthy)");
        assert_eq!(res, truthy, "assert({:?}) should return the value", truthy);
    }

    // assert falsy
    let err_false = call_global(&mut vm, "assert", vec![Value::Bool(false)]);
    assert!(err_false.is_err());
    assert_eq!(err_false.unwrap_err().to_string(), "Runtime error: assertion failed");

    let err_custom = call_global(&mut vm, "assert", vec![Value::Bool(false), Value::string("custom error message")]);
    assert!(err_custom.is_err());
    assert_eq!(err_custom.unwrap_err().to_string(), "Runtime error: custom error message");

    let err_nil = call_global(&mut vm, "assert", vec![Value::Nil]);
    assert!(err_nil.is_err());

    // error
    let err1 = call_global(&mut vm, "error", vec![Value::string("fatal mistake")]);
    assert_eq!(err1.unwrap_err().to_string(), "Runtime error: fatal mistake");

    let err2 = call_global(&mut vm, "error", vec![Value::Int(500)]);
    assert_eq!(err2.unwrap_err().to_string(), "Runtime error: 500");

    let err_empty = call_global(&mut vm, "error", vec![]);
    assert_eq!(err_empty.unwrap_err().to_string(), "Runtime error: error");
}

#[test]
fn test_core_tonumber_and_tostring() {
    let mut vm = make_vm();

    // tonumber identity & bool
    assert_eq!(call_global(&mut vm, "tonumber", vec![Value::Int(42)]).unwrap(), Value::Int(42));
    assert_eq!(call_global(&mut vm, "tonumber", vec![Value::Float(3.5)]).unwrap(), Value::Float(3.5));
    assert_eq!(call_global(&mut vm, "tonumber", vec![Value::Bool(true)]).unwrap(), Value::Int(1));
    assert_eq!(call_global(&mut vm, "tonumber", vec![Value::Bool(false)]).unwrap(), Value::Int(0));

    // tonumber string parsing
    assert_eq!(call_global(&mut vm, "tonumber", vec![Value::string("123")]).unwrap(), Value::Int(123));
    assert_eq!(call_global(&mut vm, "tonumber", vec![Value::string("3.14")]).unwrap(), Value::Float(3.14));

    // tonumber with radix
    assert_eq!(call_global(&mut vm, "tonumber", vec![Value::string("1010"), Value::Int(2)]).unwrap(), Value::Int(10));
    assert_eq!(call_global(&mut vm, "tonumber", vec![Value::string("77"), Value::Int(8)]).unwrap(), Value::Int(63));
    assert_eq!(call_global(&mut vm, "tonumber", vec![Value::string("ff"), Value::Int(16)]).unwrap(), Value::Int(255));
    assert_eq!(call_global(&mut vm, "tonumber", vec![Value::string("invalid"), Value::Int(16)]).unwrap(), Value::Nil);
    assert_eq!(call_global(&mut vm, "tonumber", vec![Value::string("not a num")]).unwrap(), Value::Nil);

    // tostring
    // Small int cache (0..255)
    assert_eq!(call_global(&mut vm, "tostring", vec![Value::Int(0)]).unwrap(), Value::string("0"));
    assert_eq!(call_global(&mut vm, "tostring", vec![Value::Int(42)]).unwrap(), Value::string("42"));
    assert_eq!(call_global(&mut vm, "tostring", vec![Value::Int(255)]).unwrap(), Value::string("255"));
    // Other ints
    assert_eq!(call_global(&mut vm, "tostring", vec![Value::Int(-1)]).unwrap(), Value::string("-1"));
    assert_eq!(call_global(&mut vm, "tostring", vec![Value::Int(1000)]).unwrap(), Value::string("1000"));
    // Booleans
    assert_eq!(call_global(&mut vm, "tostring", vec![Value::Bool(true)]).unwrap(), Value::string("true"));
    assert_eq!(call_global(&mut vm, "tostring", vec![Value::Bool(false)]).unwrap(), Value::string("false"));
    // String identity
    assert_eq!(call_global(&mut vm, "tostring", vec![Value::string("hello")]).unwrap(), Value::string("hello"));

    // tostring error on empty args
    assert!(call_global(&mut vm, "tostring", vec![]).is_err());
}

#[test]
fn test_core_pairs_and_ipairs() {
    let mut vm = make_vm();

    // Map pairs
    let m = Value::map(vec![(ValueKey::string("a".to_string()), Value::Int(1))].into_iter().collect());
    let pairs_res = call_global(&mut vm, "pairs", vec![m]).unwrap();
    assert!(matches!(pairs_res, Value::Iterator(_)));

    // Array ipairs
    let arr = Value::array(vec![Value::Int(10), Value::Int(20)]);
    let ipairs_res = call_global(&mut vm, "ipairs", vec![arr]).unwrap();
    assert!(matches!(ipairs_res, Value::Iterator(_)));
}

#[test]
fn test_core_lua_module_and_operators() {
    let mut vm = make_vm();

    // lua.nil
    let lua_mod = vm.get_global("lua").unwrap();
    if let Value::Map(m) = &lua_mod {
        let nil_val = m.borrow().get(&ValueKey::string("nil".to_string())).cloned().unwrap();
        assert_eq!(nil_val, Value::enum_unit("LuaValue", "Nil"));
    } else {
        panic!("lua module must be a Map");
    }

    // lua.table()
    let lua_tbl = call_map_entry(&mut vm, "lua", "table", vec![]).unwrap();
    assert!(matches!(lua_tbl, Value::Struct(_)));

    // lua.to_value and unwrap
    let wrapped_i = call_map_entry(&mut vm, "lua", "to_value", vec![Value::Int(42)]).unwrap();
    assert_eq!(wrapped_i, Value::enum_variant("LuaValue", "Int", vec![Value::Int(42)]));
    let unwrapped_i = call_map_entry(&mut vm, "lua", "unwrap", vec![wrapped_i]).unwrap();
    assert_eq!(unwrapped_i, Value::Int(42));

    // lua.is_truthy
    assert_eq!(call_map_entry(&mut vm, "lua", "is_truthy", vec![Value::Bool(true)]).unwrap(), Value::Bool(true));
    assert_eq!(call_map_entry(&mut vm, "lua", "is_truthy", vec![Value::Int(0)]).unwrap(), Value::Bool(true)); // 0 is truthy in Lua!
    assert_eq!(call_map_entry(&mut vm, "lua", "is_truthy", vec![Value::Bool(false)]).unwrap(), Value::Bool(false));
    assert_eq!(call_map_entry(&mut vm, "lua", "is_truthy", vec![Value::Nil]).unwrap(), Value::Bool(false));

    // lua.op_* arithmetic
    let op_add = call_map_entry(&mut vm, "lua", "op_add", vec![Value::Int(10), Value::Int(20)]).unwrap();
    assert_eq!(call_map_entry(&mut vm, "lua", "unwrap", vec![op_add]).unwrap(), Value::Int(30));

    let op_sub = call_map_entry(&mut vm, "lua", "op_sub", vec![Value::Int(30), Value::Int(10)]).unwrap();
    assert_eq!(call_map_entry(&mut vm, "lua", "unwrap", vec![op_sub]).unwrap(), Value::Int(20));

    let op_mul = call_map_entry(&mut vm, "lua", "op_mul", vec![Value::Int(5), Value::Int(6)]).unwrap();
    assert_eq!(call_map_entry(&mut vm, "lua", "unwrap", vec![op_mul]).unwrap(), Value::Int(30));

    let op_div = call_map_entry(&mut vm, "lua", "op_div", vec![Value::Int(20), Value::Int(4)]).unwrap();
    assert_eq!(call_map_entry(&mut vm, "lua", "unwrap", vec![op_div]).unwrap(), Value::Float(5.0));

    let op_mod = call_map_entry(&mut vm, "lua", "op_mod", vec![Value::Int(22), Value::Int(5)]).unwrap();
    assert_eq!(call_map_entry(&mut vm, "lua", "unwrap", vec![op_mod]).unwrap(), Value::Int(2));

    let op_neg = call_map_entry(&mut vm, "lua", "op_neg", vec![Value::Int(42)]).unwrap();
    assert_eq!(call_map_entry(&mut vm, "lua", "unwrap", vec![op_neg]).unwrap(), Value::Int(-42));

    let op_concat = call_map_entry(&mut vm, "lua", "op_concat", vec![Value::string("hello "), Value::string("world")]).unwrap();
    assert_eq!(call_map_entry(&mut vm, "lua", "unwrap", vec![op_concat]).unwrap(), Value::string("hello world"));

    // lua.op_* comparisons
    assert_eq!(call_map_entry(&mut vm, "lua", "op_eq", vec![Value::Int(5), Value::Int(5)]).unwrap(), Value::Bool(true));
    assert_eq!(call_map_entry(&mut vm, "lua", "op_eq", vec![Value::Int(5), Value::Int(6)]).unwrap(), Value::Bool(false));
    assert_eq!(call_map_entry(&mut vm, "lua", "op_ne", vec![Value::Int(5), Value::Int(6)]).unwrap(), Value::Bool(true));
    assert_eq!(call_map_entry(&mut vm, "lua", "op_lt", vec![Value::Int(5), Value::Int(10)]).unwrap(), Value::Bool(true));
    assert_eq!(call_map_entry(&mut vm, "lua", "op_le", vec![Value::Int(5), Value::Int(5)]).unwrap(), Value::Bool(true));
    assert_eq!(call_map_entry(&mut vm, "lua", "op_gt", vec![Value::Int(10), Value::Int(5)]).unwrap(), Value::Bool(true));
    assert_eq!(call_map_entry(&mut vm, "lua", "op_ge", vec![Value::Int(10), Value::Int(10)]).unwrap(), Value::Bool(true));

    // lua.table_from_entries
    let tuple1 = Value::Tuple(Rc::new(vec![Value::string("k1"), Value::Int(100)]));
    let tuple2 = Value::Tuple(Rc::new(vec![Value::string("k2"), Value::Int(200)]));
    let tbl_entries = call_map_entry(&mut vm, "lua", "table_from_entries", vec![Value::array(vec![tuple1, tuple2])]).unwrap();
    assert!(matches!(tbl_entries, Value::Struct(_)));

    // lua.require
    let req_math = call_map_entry(&mut vm, "lua", "require", vec![Value::string("math")]).unwrap();
    assert!(matches!(req_math, Value::Map(_)));
    let req_none = call_map_entry(&mut vm, "lua", "require", vec![Value::string("non_existent_mod")]).unwrap();
    assert_eq!(req_none, Value::enum_unit("LuaValue", "Nil"));
}

#[test]
fn test_core_lua_socket_helpers() {
    let mut vm = make_vm();

    // socket_protect with successful function
    let ok_fn = Value::NativeFunction(native_fn(|_| Ok(NativeCallResult::Return(Value::Int(123)))));
    let protected = call_map_entry(&mut vm, "lua", "socket_protect", vec![ok_fn]).unwrap();
    let res = vm.call_value(&protected, vec![]).unwrap();
    assert!(matches!(res, Value::Array(_)));

    // socket_protect with failing function
    let err_fn = Value::NativeFunction(native_fn(|_| Err("socket timeout".to_string())));
    let protected_err = call_map_entry(&mut vm, "lua", "socket_protect", vec![err_fn]).unwrap();
    let res_err = vm.call_value(&protected_err, vec![]).unwrap();
    assert!(matches!(res_err, Value::Array(_)));

    // socket_skip
    let skip_res = call_map_entry(&mut vm, "lua", "socket_skip", vec![Value::Int(1), Value::Int(10), Value::Int(20), Value::Int(30)]).unwrap();
    assert!(matches!(skip_res, Value::Array(_)));

    // socket_newtry & socket_try
    let finalizer_called = Rc::new(std::cell::Cell::new(false));
    let fin_clone = finalizer_called.clone();
    let finalizer = Value::NativeFunction(native_fn(move |_| {
        fin_clone.set(true);
        Ok(NativeCallResult::Return(Value::Nil))
    }));
    let newtry_fn = call_map_entry(&mut vm, "lua", "socket_newtry", vec![finalizer]).unwrap();

    // newtry with truthy -> succeeds
    let try_ok = vm.call_value(&newtry_fn, vec![Value::Bool(true), Value::string("success")]).unwrap();
    assert!(matches!(try_ok, Value::Array(_)));
    assert!(!finalizer_called.get(), "finalizer should not be called on success");

    // newtry with falsy -> fails and calls finalizer
    let try_err = vm.call_value(&newtry_fn, vec![Value::Bool(false), Value::string("network failed")]);
    assert!(try_err.is_err());
    assert!(finalizer_called.get(), "finalizer must be called on failure");

    // socket_try truthy
    let st_ok = call_map_entry(&mut vm, "lua", "socket_try", vec![Value::Bool(true), Value::Int(42)]).unwrap();
    assert!(matches!(st_ok, Value::Array(_)));

    // socket_try falsy -> fails
    let st_err = call_map_entry(&mut vm, "lua", "socket_try", vec![Value::Bool(false), Value::string("bad connection")]);
    assert!(st_err.is_err());
}

#[test]
fn test_core_task_module_negative_and_current() {
    let mut vm = make_vm();

    // task.current takes 0 args and returns None outside running task
    let cur = call_map_entry(&mut vm, "task", "current", vec![]).unwrap();
    assert_eq!(cur, Value::none());
    assert!(call_map_entry(&mut vm, "task", "current", vec![Value::Int(1)]).is_err(), "current takes 0 args");

    // task.run & task.create require a function
    assert!(call_map_entry(&mut vm, "task", "run", vec![]).is_err(), "task.run requires function");
    assert!(call_map_entry(&mut vm, "task", "create", vec![]).is_err(), "task.create requires function");

    // task.status, task.info, task.stop, task.restart require task handle
    assert!(call_map_entry(&mut vm, "task", "status", vec![]).is_err());
    assert!(call_map_entry(&mut vm, "task", "status", vec![Value::Int(123)]).is_err(), "requires task handle");
    assert!(call_map_entry(&mut vm, "task", "info", vec![Value::Nil]).is_err());
    assert!(call_map_entry(&mut vm, "task", "stop", vec![Value::string("bad")]).is_err());
    assert!(call_map_entry(&mut vm, "task", "restart", vec![Value::Bool(true)]).is_err());
}

// =========================================================================
// 8. IO & OS MODULES (CONFIGURED MODULES)
// =========================================================================

#[test]
fn test_io_module_operations() {
    let mut vm = make_vm_with_io_os();
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let test_file = temp_dir.path().join("test_io.txt");
    let test_file_str = test_file.to_str().unwrap();

    // write_file
    let write_res = call_map_entry(&mut vm, "io", "write_file", vec![Value::string(test_file_str), Value::string("Hello Lust IO!")]).unwrap();
    assert_eq!(write_res, Value::ok(Value::Nil));

    // read_file
    let read_res = call_map_entry(&mut vm, "io", "read_file", vec![Value::string(test_file_str)]).unwrap();
    assert_eq!(read_res, Value::ok(Value::string("Hello Lust IO!")));

    // read_file_bytes
    let bytes_res = call_map_entry(&mut vm, "io", "read_file_bytes", vec![Value::string(test_file_str)]).unwrap();
    assert!(matches!(bytes_res, Value::Enum(_)));

    // read_dir
    let dir_res = call_map_entry(&mut vm, "io", "read_dir", vec![Value::string(temp_dir.path().to_str().unwrap())]).unwrap();
    assert!(matches!(dir_res, Value::Enum(_)));

    // write_stdout
    let stdout_res = call_map_entry(&mut vm, "io", "write_stdout", vec![Value::string("test stdout output\n")]).unwrap();
    assert_eq!(stdout_res, Value::ok(Value::Nil));

    // read_stdin and read_line error on arguments (so we test them without blocking on stdin!)
    assert_eq!(call_map_entry(&mut vm, "io", "read_stdin", vec![Value::Int(1)]).unwrap(), Value::err(Value::string("io.read_stdin() takes no arguments")));
    assert_eq!(call_map_entry(&mut vm, "io", "read_line", vec![Value::Int(1)]).unwrap(), Value::err(Value::string("io.read_line() takes no arguments")));

    // Negative cases
    assert_eq!(call_map_entry(&mut vm, "io", "read_file", vec![]).unwrap(), Value::err(Value::string("io.read_file(path) requires a single string path")));
    assert_eq!(call_map_entry(&mut vm, "io", "read_file", vec![Value::Int(123)]).unwrap(), Value::err(Value::string("io.read_file(path) requires a string path")));
    assert_eq!(call_map_entry(&mut vm, "io", "write_file", vec![Value::string(test_file_str)]).unwrap(), Value::err(Value::string("io.write_file(path, contents) requires a path and value")));
}

#[test]
fn test_os_module_operations() {
    let mut vm = make_vm_with_io_os();
    let temp_dir = tempfile::tempdir().expect("tempdir");

    // os.time
    let time_res = call_map_entry(&mut vm, "os", "time", vec![]).unwrap();
    if let Value::Float(t) = time_res {
        assert!(t > 0.0, "os.time() must be positive timestamp");
    } else {
        panic!("os.time must return Float");
    }
    assert_eq!(call_map_entry(&mut vm, "os", "time", vec![Value::Int(1)]).unwrap(), Value::err(Value::string("os.time() takes no arguments")));

    // os.sleep (0 duration)
    let sleep_res = call_map_entry(&mut vm, "os", "sleep", vec![Value::Float(0.0)]).unwrap();
    assert_eq!(sleep_res, Value::ok(Value::Nil));
    // Negative sleep duration
    assert_eq!(call_map_entry(&mut vm, "os", "sleep", vec![Value::Float(-1.0)]).unwrap(), Value::err(Value::string("os.sleep(seconds) requires a finite, non-negative duration")));
    assert_eq!(call_map_entry(&mut vm, "os", "sleep", vec![Value::string("bad")]).unwrap(), Value::err(Value::string("os.sleep(seconds) requires a float duration")));

    // os.create_file & os.remove_file
    let os_file = temp_dir.path().join("os_temp.txt");
    let os_file_str = os_file.to_str().unwrap();
    let cr_file = call_map_entry(&mut vm, "os", "create_file", vec![Value::string(os_file_str)]).unwrap();
    assert_eq!(cr_file, Value::ok(Value::Nil));
    assert!(os_file.exists());

    let rm_file = call_map_entry(&mut vm, "os", "remove_file", vec![Value::string(os_file_str)]).unwrap();
    assert_eq!(rm_file, Value::ok(Value::Nil));
    assert!(!os_file.exists());

    // os.create_dir & os.remove_dir
    let os_subdir = temp_dir.path().join("subdir");
    let os_subdir_str = os_subdir.to_str().unwrap();
    let cr_dir = call_map_entry(&mut vm, "os", "create_dir", vec![Value::string(os_subdir_str)]).unwrap();
    assert_eq!(cr_dir, Value::ok(Value::Nil));
    assert!(os_subdir.exists());

    let rm_dir = call_map_entry(&mut vm, "os", "remove_dir", vec![Value::string(os_subdir_str)]).unwrap();
    assert_eq!(rm_dir, Value::ok(Value::Nil));
    assert!(!os_subdir.exists());

    // os.rename
    let f1 = temp_dir.path().join("orig.txt");
    let f2 = temp_dir.path().join("renamed.txt");
    fs::write(&f1, "data").unwrap();
    let ren = call_map_entry(&mut vm, "os", "rename", vec![Value::string(f1.to_str().unwrap()), Value::string(f2.to_str().unwrap())]).unwrap();
    assert_eq!(ren, Value::ok(Value::Nil));
    assert!(f2.exists());

    // os.get_env
    let env_res = call_map_entry(&mut vm, "os", "get_env", vec![Value::string("PATH")]).unwrap();
    assert!(matches!(env_res, Value::Enum(_)));

    let env_missing = call_map_entry(&mut vm, "os", "get_env", vec![Value::string("DEFINITELY_NON_EXISTENT_ENV_VAR_12345")]).unwrap();
    assert_eq!(env_missing, Value::ok(Value::none()));
}

// =========================================================================
// 9. ROOT STDLIB FUNCTIONS: print, println, type, select
// =========================================================================

#[test]
fn test_root_print_println_type_select() {
    let mut vm = make_vm();

    // print & println
    assert_eq!(call_global(&mut vm, "print", vec![]).unwrap(), Value::Nil);
    assert_eq!(call_global(&mut vm, "print", vec![Value::Int(1), Value::string("test")]).unwrap(), Value::Nil);
    assert_eq!(call_global(&mut vm, "println", vec![]).unwrap(), Value::Nil);
    assert_eq!(call_global(&mut vm, "println", vec![Value::Bool(true)]).unwrap(), Value::Nil);

    // type
    let lua_str = |s: &str| Value::enum_variant("LuaValue", "String", vec![Value::string(s)]);
    assert_eq!(call_global(&mut vm, "type", vec![Value::Nil]).unwrap(), lua_str("nil"));
    assert_eq!(call_global(&mut vm, "type", vec![Value::Bool(true)]).unwrap(), lua_str("bool"));
    assert_eq!(call_global(&mut vm, "type", vec![Value::Int(42)]).unwrap(), lua_str("int"));
    assert_eq!(call_global(&mut vm, "type", vec![Value::Float(3.14)]).unwrap(), lua_str("float"));
    assert_eq!(call_global(&mut vm, "type", vec![Value::string("hi")]).unwrap(), lua_str("string"));
    assert_eq!(call_global(&mut vm, "type", vec![Value::array(vec![])]).unwrap(), lua_str("array"));
    assert_eq!(call_global(&mut vm, "type", vec![Value::map(vec![].into_iter().collect())]).unwrap(), lua_str("map"));

    // select
    // '#' returns argument count
    assert_eq!(call_global(&mut vm, "select", vec![Value::string("#"), Value::Int(10), Value::Int(20), Value::Int(30)]).unwrap(), Value::Int(3));
    assert_eq!(call_global(&mut vm, "select", vec![Value::string("#")]).unwrap(), Value::Int(0));

    // positive index
    let sel_pos = call_global(&mut vm, "select", vec![Value::Int(2), Value::string("a"), Value::string("b"), Value::string("c")]).unwrap();
    assert!(matches!(sel_pos, Value::Array(_)));

    // negative index
    let sel_neg = call_global(&mut vm, "select", vec![Value::Int(-1), Value::string("a"), Value::string("b"), Value::string("c")]).unwrap();
    assert!(matches!(sel_neg, Value::Array(_)));

    // out of bounds
    let sel_oob = call_global(&mut vm, "select", vec![Value::Int(10), Value::string("a")]).unwrap();
    assert!(matches!(sel_oob, Value::Array(_)));

    // bad string index error
    assert!(call_global(&mut vm, "select", vec![Value::string("bad_selector")]).is_err());
}

// =========================================================================
// 10. ADVANCED METATABLE, OPERATOR, ERROR & HOST BOUNDARY CASES
// =========================================================================

#[test]
fn test_metatable_lifecycle_and_protection() {
    let mut vm = make_vm();

    // 1. Create a Lua table
    let tbl = call_map_entry(&mut vm, "lua", "table", vec![]).unwrap();

    // getmetatable on table without metatable returns Nil
    let meta_none = call_map_entry(&mut vm, "lua", "getmetatable", vec![tbl.clone()]).unwrap();
    assert_eq!(meta_none, Value::enum_unit("LuaValue", "Nil"), "getmetatable without meta must return Nil");

    // 2. Set metatable with __metatable protection
    let meta_table = call_map_entry(&mut vm, "lua", "table", vec![]).unwrap();
    // Insert "__metatable" = "protected!" into meta_table
    if let Some(Value::Map(m)) = meta_table.struct_get_field("table") {
        m.borrow_mut().insert(
            ValueKey::string("__metatable".to_string()),
            Value::string("protected!"),
        );
        m.borrow_mut().insert(
            ValueKey::string("__tostring".to_string()),
            Value::string("custom_tostring"),
        );
    }

    let set_res = call_map_entry(&mut vm, "lua", "setmetatable", vec![tbl.clone(), meta_table.clone()]).unwrap();
    assert!(matches!(set_res, Value::Struct(_)));

    // getmetatable should return the protected value ("protected!")
    let protected_meta = call_map_entry(&mut vm, "lua", "getmetatable", vec![tbl.clone()]).unwrap();
    assert_eq!(protected_meta, Value::string("protected!"), "getmetatable should return __metatable value");

    // 3. Clear metatable with Nil
    let cleared = call_map_entry(&mut vm, "lua", "setmetatable", vec![tbl.clone(), Value::Nil]).unwrap();
    assert!(matches!(cleared, Value::Struct(_)));

    // Negative cases for setmetatable / getmetatable
    assert!(call_map_entry(&mut vm, "lua", "setmetatable", vec![tbl.clone()]).is_err(), "setmetatable requires 2 args");
    assert!(call_map_entry(&mut vm, "lua", "getmetatable", vec![]).is_err(), "getmetatable requires 1 arg");
    assert!(call_global(&mut vm, "setmetatable", vec![Value::Int(123), Value::Nil]).is_err(), "setmetatable on non-LuaTable must fail");
}

#[test]
fn test_call_method_direct_and_metamethod_index() {
    let mut vm = make_vm();

    // 1. Direct method in LuaTable 'table' map
    let tbl = call_map_entry(&mut vm, "lua", "table", vec![]).unwrap();
    let greet_fn = Value::NativeFunction(native_fn(|args: &[Value]| {
        let receiver = args.first().cloned().unwrap_or(Value::Nil);
        Ok(NativeCallResult::Return(Value::string(format!("greeted {:?}", receiver.type_of()))))
    }));

    if let Some(Value::Map(m)) = tbl.struct_get_field("table") {
        m.borrow_mut().insert(
            ValueKey::string("greet".to_string()),
            greet_fn.clone(),
        );
    }

    let method_res = call_map_entry(&mut vm, "lua", "call_method", vec![tbl.clone(), Value::string("greet")]).unwrap();
    assert_eq!(method_res, Value::string("greeted Struct"));

    // 2. Method in metatable __index table
    let empty_map = || Value::map(vec![].into_iter().collect());
    let tbl2 = call_map_entry(&mut vm, "lua", "table", vec![]).unwrap();
    let index_map = empty_map();
    if let Value::Map(m) = &index_map {
        m.borrow_mut().insert(
            ValueKey::string("calc".to_string()),
            Value::NativeFunction(native_fn(|args: &[Value]| {
                let x = args.get(1).and_then(|v| v.as_int()).unwrap_or(0);
                Ok(NativeCallResult::Return(Value::Int(x * 10)))
            })),
        );
    }
    let meta_map = empty_map();
    if let Value::Map(m) = &meta_map {
        m.borrow_mut().insert(
            ValueKey::string("__index".to_string()),
            index_map,
        );
    }
    let meta_tbl = vm.instantiate_struct(
        "LuaTable",
        vec![
            (Rc::new("table".to_string()), empty_map()),
            (Rc::new("metamethods".to_string()), meta_map),
        ],
    ).unwrap();

    call_map_entry(&mut vm, "lua", "setmetatable", vec![tbl2.clone(), meta_tbl]).unwrap();
    let calc_res = call_map_entry(&mut vm, "lua", "call_method", vec![tbl2.clone(), Value::string("calc"), Value::Int(5)]).unwrap();
    assert_eq!(calc_res, Value::Int(50));

    // Negative cases for call_method
    assert!(call_map_entry(&mut vm, "lua", "call_method", vec![tbl.clone(), Value::string("non_existent")]).is_err(), "missing method must fail");
    assert!(call_map_entry(&mut vm, "lua", "call_method", vec![tbl.clone(), Value::Int(123)]).is_err(), "non-string method name must fail");
}

#[test]
fn test_gsub_advanced_and_functional_replacement() {
    let mut vm = make_vm();

    // Function replacer in gsub
    let replacer_fn = Value::NativeFunction(native_fn(|args: &[Value]| {
        let arg0 = args.first().cloned().unwrap_or(Value::Nil);
        let match_str = match &arg0 {
            Value::String(s) => s.to_string(),
            Value::Enum(obj) if obj.enum_name == "LuaValue" => {
                obj.values
                    .as_ref()
                    .and_then(|vals| vals.first())
                    .and_then(|v| v.as_string())
                    .unwrap_or("")
                    .to_string()
            }
            _ => String::new(),
        };
        let doubled = format!("{}_{}", match_str, match_str);
        Ok(NativeCallResult::Return(Value::string(doubled)))
    }));

    let gsub_fn_res = call_global(&mut vm, "string.gsub", vec![
        Value::string("hello world"),
        Value::string("%a+"),
        replacer_fn,
        Value::Int(1), // limit 1 replacement
    ]).unwrap();

    if let Value::Array(arr) = gsub_fn_res {
        let b = arr.borrow();
        assert_eq!(b.first().unwrap(), &Value::enum_variant("LuaValue", "String", vec![Value::string("hello_hello world")]));
    } else {
        panic!("gsub must return array of results");
    }

    // Pattern capture replacement with %1 and %2
    let swap_res = call_global(&mut vm, "string.gsub", vec![
        Value::string("first second"),
        Value::string("(%a+)%s+(%a+)"),
        Value::string("%2 %1"),
    ]).unwrap();

    if let Value::Array(arr) = swap_res {
        let b = arr.borrow();
        assert_eq!(b.first().unwrap(), &Value::enum_variant("LuaValue", "String", vec![Value::string("second first")]));
    }
}

#[test]
fn test_table_unpack_map_and_ranges() {
    let mut vm = make_vm();

    // Map sequence unpacking
    let map_seq = Value::map(vec![].into_iter().collect());
    if let Value::Map(m) = &map_seq {
        let mut borrowed = m.borrow_mut();
        borrowed.insert(ValueKey::from(Value::Int(1)), Value::string("one"));
        borrowed.insert(ValueKey::from(Value::Int(2)), Value::string("two"));
        borrowed.insert(ValueKey::from(Value::Int(3)), Value::string("three"));
    }

    let unpack_map = call_global(&mut vm, "table.unpack", vec![map_seq, Value::Int(1), Value::Int(2)]).unwrap();
    if let Value::Array(arr) = unpack_map {
        let b = arr.borrow();
        assert_eq!(b.len(), 2);
    } else {
        panic!("table.unpack must return packed array of Lua values");
    }

    // Array start > end returns empty
    let arr = Value::array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    let unpack_empty = call_global(&mut vm, "table.unpack", vec![arr, Value::Int(3), Value::Int(1)]).unwrap();
    if let Value::Array(arr) = unpack_empty {
        assert_eq!(arr.borrow().len(), 0);
    }
}

#[test]
fn test_os_and_io_advanced_error_branches() {
    let mut vm = make_vm_with_io_os();

    // os.sleep duration too large
    let sleep_huge = call_map_entry(&mut vm, "os", "sleep", vec![Value::Float(f64::INFINITY)]).unwrap();
    assert_eq!(sleep_huge, Value::err(Value::string("os.sleep(seconds) requires a finite, non-negative duration")));

    let sleep_nan = call_map_entry(&mut vm, "os", "sleep", vec![Value::Float(f64::NAN)]).unwrap();
    assert_eq!(sleep_nan, Value::err(Value::string("os.sleep(seconds) requires a finite, non-negative duration")));

    // io.read_file non-existent file
    let read_missing = call_map_entry(&mut vm, "io", "read_file", vec![Value::string("/non_existent_path_48291/file.txt")]).unwrap();
    assert!(matches!(read_missing, Value::Enum(ref e) if e.variant == "Err"));

    // io.write_file invalid directory
    let write_bad = call_map_entry(&mut vm, "io", "write_file", vec![Value::string("/non_existent_dir_83912/file.txt"), Value::string("content")]).unwrap();
    assert!(matches!(write_bad, Value::Enum(ref e) if e.variant == "Err"));

    // io.read_dir non-existent directory
    let dir_bad = call_map_entry(&mut vm, "io", "read_dir", vec![Value::string("/non_existent_dir_92812")]).unwrap();
    assert!(matches!(dir_bad, Value::Enum(ref e) if e.variant == "Err"));

    // os.remove_file non-existent file
    let rm_bad = call_map_entry(&mut vm, "os", "remove_file", vec![Value::string("/non_existent_file_82912")]).unwrap();
    assert!(matches!(rm_bad, Value::Enum(ref e) if e.variant == "Err"));

    // os.rename missing source
    let ren_bad = call_map_entry(&mut vm, "os", "rename", vec![Value::string("/missing_source_92812"), Value::string("/target_92812")]).unwrap();
    assert!(matches!(ren_bad, Value::Enum(ref e) if e.variant == "Err"));

    // os.get_env non-string argument
    let env_bad = call_map_entry(&mut vm, "os", "get_env", vec![Value::Int(123)]).unwrap();
    assert_eq!(env_bad, Value::err(Value::string("os.get_env(name) requires a string name")));
}

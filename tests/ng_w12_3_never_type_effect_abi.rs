use lust::effect_abi::{
    Cause, DefectOrInterrupt, Exit, InterruptReason, LUST_STATUS_ERR_GENERIC,
    LUST_STATUS_ERR_INTERRUPTED, LUST_STATUS_ERR_OUT_OF_GAS, LUST_STATUS_ERR_OUT_OF_MEMORY,
    LUST_STATUS_SUCCESS, PureCause, PureExit,
};
use lust::embed::program::EmbeddedBuilder;
use std::convert::Infallible;

#[test]
fn test_pure_exit_and_never_type_success() {
    let exit: PureExit<i64> = Exit::Success(42);
    assert!(exit.is_success());
    assert!(!exit.is_failure());
    assert!(!exit.is_interrupt());
    assert_eq!(exit.status_code(), LUST_STATUS_SUCCESS);

    let res = exit.into_ok_or_defect();
    assert_eq!(res, Ok(42));
}

#[test]
fn test_pure_exit_and_never_type_defects_and_interrupts() {
    // 1. Die defect
    let die_exit: PureExit<i64> = Exit::Failure(Cause::Die("fatal panic".to_string()));
    assert!(!die_exit.is_success());
    assert!(die_exit.is_failure());
    assert!(!die_exit.is_interrupt());
    assert_eq!(die_exit.status_code(), LUST_STATUS_ERR_GENERIC);
    match die_exit.clone().into_ok_or_defect() {
        Err(DefectOrInterrupt::Die(msg)) => assert_eq!(msg, "fatal panic"),
        other => panic!("expected DefectOrInterrupt::Die, got {other:?}"),
    }

    // 2. Generic interrupt
    let int_exit: PureExit<i64> = Exit::Failure(Cause::Interrupt);
    assert!(int_exit.is_interrupt());
    assert_eq!(int_exit.status_code(), LUST_STATUS_ERR_INTERRUPTED);
    assert_eq!(
        int_exit.interrupt_reason(),
        Some(InterruptReason::Cancelled)
    );
    match int_exit.into_ok_or_defect() {
        Err(DefectOrInterrupt::Interrupt) => {}
        other => panic!("expected DefectOrInterrupt::Interrupt, got {other:?}"),
    }

    // 3. Typed interrupt reason
    let oom_exit: PureExit<i64> =
        Exit::Failure(Cause::Interrupted(InterruptReason::OutOfMemory));
    assert!(oom_exit.is_interrupt());
    assert_eq!(oom_exit.status_code(), LUST_STATUS_ERR_OUT_OF_MEMORY);
    assert_eq!(
        oom_exit.interrupt_reason(),
        Some(InterruptReason::OutOfMemory)
    );
    match oom_exit.into_ok_or_defect() {
        Err(DefectOrInterrupt::Interrupted(InterruptReason::OutOfMemory)) => {}
        other => panic!("expected OutOfMemory, got {other:?}"),
    }

    let oog_exit: PureExit<i64> =
        Exit::Failure(Cause::Interrupted(InterruptReason::OutOfGas));
    assert_eq!(oog_exit.status_code(), LUST_STATUS_ERR_OUT_OF_GAS);
}

#[test]
fn test_defect_or_interrupt_helpers_and_conversions() {
    let die = DefectOrInterrupt::Die("crash".to_string());
    assert!(!die.is_interrupt());
    assert_eq!(die.status_code(), LUST_STATUS_ERR_GENERIC);
    assert_eq!(die.to_string(), "defect: crash");

    let cause: Cause<String> = die.into_cause();
    assert_eq!(cause, Cause::Die("crash".to_string()));

    let int = DefectOrInterrupt::Interrupt;
    assert!(int.is_interrupt());
    assert_eq!(int.interrupt_reason(), Some(InterruptReason::Cancelled));
    assert_eq!(int.status_code(), LUST_STATUS_ERR_INTERRUPTED);

    let typed_int = DefectOrInterrupt::Interrupted(InterruptReason::Timeout);
    assert!(typed_int.is_interrupt());
    assert_eq!(typed_int.interrupt_reason(), Some(InterruptReason::Timeout));
    let cause_from: Cause<Infallible> = typed_int.into();
    assert_eq!(
        cause_from,
        Cause::Interrupted(InterruptReason::Timeout)
    );
}

#[test]
fn test_pure_exit_upcast_infallible() {
    let pure_ok: PureExit<i64> = Exit::Success(100);
    let upcasted_ok: Exit<i64, String> = pure_ok.upcast_infallible();
    assert_eq!(upcasted_ok, Exit::Success(100));

    let pure_err: PureExit<i64> =
        Exit::Failure(Cause::Die("die in pure".to_string()));
    let upcasted_err: Exit<i64, String> = pure_err.upcast_infallible();
    assert_eq!(
        upcasted_err,
        Exit::Failure(Cause::Die("die in pure".to_string()))
    );

    let pure_cause: PureCause = Cause::Interrupt;
    let upcasted_cause: Cause<String> = pure_cause.upcast_infallible();
    assert_eq!(upcasted_cause, Cause::Interrupt);
}

#[test]
fn test_embedded_program_infallible_callbacks_and_into_embed_result() {
    let source = r#"
        extern
            function add_ten(int): int
            function fallible_double(int): int
            function pure_square(int): int
            function exit_triple(int): int
        end

        function run_callbacks(v: int): int
            let a = add_ten(v)
            let b = fallible_double(a)
            let c = pure_square(b)
            return exit_triple(c)
        end
    "#;

    let mut program = EmbeddedBuilder::new()
        .module("main", source)
        .entry_module("main")
        .compile()
        .expect("compile embedded program with externs");

    // 1. Infallible callback returning Result<i64, Infallible>
    program
        .register_typed_native("add_ten", |x: i64| -> Result<i64, Infallible> {
            Ok(x + 10)
        })
        .expect("register add_ten");

    // 2. Standard fallible callback returning Result<i64, String>
    program
        .register_typed_native("fallible_double", |x: i64| -> Result<i64, String> {
            Ok(x * 2)
        })
        .expect("register fallible_double");

    // 3. Infallible callback directly returning i64 via register_infallible_typed_native
    program
        .register_infallible_typed_native("pure_square", |x: i64| -> i64 {
            x * x
        })
        .expect("register pure_square");

    // 4. Infallible effect callback returning PureExit<i64>
    program
        .register_typed_native("exit_triple", |x: i64| -> PureExit<i64> {
            Exit::Success(x * 3)
        })
        .expect("register exit_triple");

    let result: i64 = program
        .call_typed("main.run_callbacks", 2i64)
        .expect("call run_callbacks");

    // Calculation:
    // v = 2
    // a = 2 + 10 = 12
    // b = 12 * 2 = 24
    // c = 24 * 24 = 576
    // result = 576 * 3 = 1728
    assert_eq!(result, 1728);
}

#[test]
fn test_pure_exit_defect_in_embedded_callback_returns_error() {
    let source = r#"
        extern
            function bad_callback(int): int
        end

        function test_call(x: int): int
            return bad_callback(x)
        end
    "#;

    let mut program = EmbeddedBuilder::new()
        .module("main", source)
        .entry_module("main")
        .compile()
        .expect("compile embedded program");

    program
        .register_typed_native("bad_callback", |_x: i64| -> PureExit<i64> {
            Exit::Failure(Cause::Die("defect triggered in pure callback".to_string()))
        })
        .expect("register bad_callback");

    let err = program.call_typed::<_, i64>("main.test_call", 10i64);
    assert!(err.is_err(), "expected error from defect in callback");
    let err_str = format!("{:?}", err.unwrap_err());
    assert!(
        err_str.contains("defect triggered in pure callback"),
        "error must contain defect message, got: {err_str}"
    );
}

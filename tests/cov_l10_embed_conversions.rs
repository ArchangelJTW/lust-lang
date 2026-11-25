#![cfg(feature = "std")]
//! Stage 3 - L10: Coverage and correctness tests for Embedding, Conversions, Native Types, and Async Runtime.

use lust::ast::{
    items::{
        EnumDef, EnumVariant, FieldOwnership, StructDef, StructField, TraitDef, TraitMethod,
        Visibility,
    },
    Span, Type, TypeKind,
};
use lust::bytecode::{TaskHandle, Value};
use lust::embed::{
    async_runtime::{AsyncTaskQueue, PendingAsyncTask, signal_pair},
    conversions::{FromLustArgs, FromLustValue, FunctionArgs, IntoLustValue},
    native_types::ExternRegistry,
    program::{CompiledBytecode, EmbeddedBuilder, EmbeddedProgram},
};
use lust::number::{LustFloat, LustInt};
use lust::typechecker::TypeChecker;
use lust::vm::VM;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

fn noop_waker() -> Waker {
    fn noop_clone(_: *const ()) -> RawWaker {
        RawWaker::new(std::ptr::null(), &NOOP_VTABLE)
    }
    fn noop(_: *const ()) {}
    static NOOP_VTABLE: RawWakerVTable = RawWakerVTable::new(noop_clone, noop, noop, noop);
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &NOOP_VTABLE)) }
}

// ---------------------------------------------------------------------------
// 1. Roundtrip Conversions: Rust -> Lust -> Rust & Error Rejections
// ---------------------------------------------------------------------------

#[test]
fn test_l10_primitive_value_roundtrips() {
    // 1. Unit ()
    let unit_val = ().into_value();
    assert_eq!(unit_val, Value::Nil);
    assert!(<()>::is_unit());
    assert_eq!(<()>::from_value(Value::Nil).unwrap(), ());

    // 2. Integers (LustInt / i64)
    let int_val: LustInt = 42;
    let lust_int = int_val.into_value();
    assert_eq!(lust_int, Value::Int(42));
    assert_eq!(LustInt::from_value(lust_int).unwrap(), 42);

    let int_neg: LustInt = -9999;
    let lust_neg = int_neg.into_value();
    assert_eq!(lust_neg, Value::Int(-9999));
    assert_eq!(LustInt::from_value(lust_neg).unwrap(), -9999);

    // 3. Floats (LustFloat / f64)
    let float_val: LustFloat = 3.14159;
    let lust_float = float_val.into_value();
    assert_eq!(lust_float, Value::Float(3.14159));
    assert_eq!(LustFloat::from_value(lust_float).unwrap(), 3.14159);

    // 4. Booleans
    assert_eq!(true.into_value(), Value::Bool(true));
    assert_eq!(bool::from_value(Value::Bool(true)).unwrap(), true);
    assert_eq!(bool::from_value(Value::Bool(false)).unwrap(), false);

    // 5. Strings (String, &str, Rc<String>)
    let s = "hello world".to_string();
    let lust_str = s.clone().into_value();
    assert_eq!(String::from_value(lust_str.clone()).unwrap(), "hello world");
    let rc_s: Rc<String> = Rc::from_value(lust_str).unwrap();
    assert_eq!(&*rc_s, "hello world");

    // 6. Option<T>
    let opt_some: Option<i64> = Some(777);
    let lust_some = opt_some.into_value();
    assert_eq!(Option::<i64>::from_value(lust_some).unwrap(), Some(777));

    let opt_none: Option<i64> = None;
    let lust_none = opt_none.into_value();
    assert_eq!(Option::<i64>::from_value(lust_none).unwrap(), None);

    // 7. Vec<T>
    let vec_data = vec![10i64, 20i64, 30i64];
    let lust_vec = vec_data.clone().into_value();
    let roundtrip_vec = Vec::<i64>::from_value(lust_vec).unwrap();
    assert_eq!(roundtrip_vec, vec_data);
}

#[test]
fn test_l10_tuple_conversions_up_to_large_arity() {
    // Pair (A, B)
    let tuple2 = (42i64, "answer".to_string());
    let vals2 = tuple2.clone().into_values();
    let roundtrip2: (i64, String) = FromLustArgs::from_values(&vals2).unwrap();
    assert_eq!(roundtrip2, tuple2);

    // Triple (A, B, C)
    let tuple3 = (1i64, 2.5f64, true);
    let vals3 = tuple3.into_values();
    let roundtrip3: (i64, f64, bool) = FromLustArgs::from_values(&vals3).unwrap();
    assert_eq!(roundtrip3, tuple3);

    // 4-tuple (A, B, C, D)
    let tuple4 = (10i64, 20i64, "thirty".to_string(), false);
    let vals4 = tuple4.clone().into_values();
    let roundtrip4: (i64, i64, String, bool) = FromLustArgs::from_values(&vals4).unwrap();
    assert_eq!(roundtrip4, tuple4);
}

#[test]
fn test_l10_conversion_error_rejection_oracles() {
    // 1. Expecting int, got string
    let err = LustInt::from_value(Value::string("not an int")).unwrap_err();
    assert!(
        err.to_string().contains("Expected Lust value 'int'"),
        "Error mismatch: {err}"
    );

    // 2. Expecting float, got bool
    let err = LustFloat::from_value(Value::Bool(true)).unwrap_err();
    assert!(
        err.to_string().contains("Expected Lust value 'float'"),
        "Error mismatch: {err}"
    );

    // 3. Expecting bool, got int
    let err = bool::from_value(Value::Int(0)).unwrap_err();
    assert!(
        err.to_string().contains("Expected Lust value 'bool'"),
        "Error mismatch: {err}"
    );

    // 4. Expecting string, got nil
    let err = String::from_value(Value::Nil).unwrap_err();
    assert!(
        err.to_string().contains("Expected Lust value 'string'"),
        "Error mismatch: {err}"
    );
}

#[test]
fn test_l10_function_args_and_opt07_write_args() {
    // Single argument
    let arg = 42i64;
    assert_eq!(arg.into_values(), vec![Value::Int(42)]);

    let mut buf = [const { Value::Nil }; 4];
    let written = arg.write_args(&mut buf);
    assert_eq!(written, 1);
    assert_eq!(buf[0], Value::Int(42));
    assert_eq!(buf[1], Value::Nil);

    // Pair tuple argument
    let tuple_args = (100i64, true);
    let values = tuple_args.into_values();
    assert_eq!(values, vec![Value::Int(100), Value::Bool(true)]);

    let mut buf2 = [const { Value::Nil }; 4];
    let written2 = tuple_args.write_args(&mut buf2);
    assert_eq!(written2, 2);
    assert_eq!(buf2[0], Value::Int(100));
    assert_eq!(buf2[1], Value::Bool(true));
}

// ---------------------------------------------------------------------------
// 2. Native Types & ExternRegistry (native_types.rs)
// ---------------------------------------------------------------------------

#[test]
fn test_l10_extern_registry_lifecycle_and_registration() {
    let span = Span::dummy();
    let mut reg = ExternRegistry::new();
    assert!(reg.is_empty());

    // Register Struct
    let s_def = StructDef {
        name: "User".to_string(),
        fields: vec![
            StructField {
                name: "id".to_string(),
                ty: Type::new(TypeKind::Int, span),
                visibility: Visibility::Public,
                ownership: FieldOwnership::Strong,
                weak_target: None,
            },
            StructField {
                name: "active".to_string(),
                ty: Type::new(TypeKind::Bool, span),
                visibility: Visibility::Public,
                ownership: FieldOwnership::Strong,
                weak_target: None,
            },
        ],
        visibility: Visibility::Public,
        type_params: vec![],
        trait_bounds: vec![],
        doc: None,
    };
    reg.add_struct(s_def);
    assert!(!reg.is_empty());

    // Register Enum
    let e_def = EnumDef {
        name: "Role".to_string(),
        variants: vec![
            EnumVariant {
                name: "Admin".to_string(),
                fields: None,
            },
            EnumVariant {
                name: "Guest".to_string(),
                fields: None,
            },
        ],
        visibility: Visibility::Public,
        type_params: vec![],
        trait_bounds: vec![],
        doc: None,
    };
    reg.add_enum(e_def);

    // Register Trait
    let t_def = TraitDef {
        name: "Describable".to_string(),
        type_params: vec![],
        methods: vec![TraitMethod {
            name: "describe".to_string(),
            type_params: vec![],
            params: vec![],
            return_type: Some(Type::new(TypeKind::String, span)),
            default_impl: None,
            doc: None,
        }],
        visibility: Visibility::Public,
        doc: None,
    };
    reg.add_trait(t_def);

    // Register Constant
    reg.add_const("MAX_CONNECTIONS", Type::new(TypeKind::Int, span));

    // Register with TypeChecker
    let mut checker = TypeChecker::new();
    reg.register_with_typechecker(&mut checker)
        .expect("ExternRegistry should register cleanly with TypeChecker");

    // Register with VM
    let mut vm = VM::new();
    reg.register_with_vm(&mut vm);

    // Extend into second registry
    let mut reg2 = ExternRegistry::new();
    reg2.extend(&reg);
    assert!(!reg2.is_empty());
}

// ---------------------------------------------------------------------------
// 3. Async Runtime: Queues, Completers, Signal Futures, and Cancellations
// ---------------------------------------------------------------------------

#[test]
fn test_l10_async_queue_push_pop_len() {
    let queue: AsyncTaskQueue<i64, String> = AsyncTaskQueue::new();
    assert!(queue.is_empty());
    assert_eq!(queue.len(), 0);

    let (completer, _) = signal_pair::<String>();
    let task = PendingAsyncTask::new(TaskHandle(42), 100i64, completer);
    assert_eq!(task.task(), TaskHandle(42));
    assert_eq!(*task.args(), 100i64);

    queue.push(task);
    assert!(!queue.is_empty());
    assert_eq!(queue.len(), 1);

    let popped = queue.pop().expect("Should pop task");
    assert_eq!(popped.task(), TaskHandle(42));
    assert_eq!(*popped.args(), 100i64);
    assert!(queue.is_empty());
}

#[test]
fn test_l10_async_signal_pair_ok_and_err() {
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);

    // 1. Success path
    let (completer_ok, mut future_ok) = signal_pair::<i64>();
    // First poll: pending
    let poll1 = Pin::new(&mut future_ok).poll(&mut cx);
    assert!(matches!(poll1, Poll::Pending));

    // Complete OK
    completer_ok.complete_ok(888);
    // Second poll: ready Ok
    let poll2 = Pin::new(&mut future_ok).poll(&mut cx);
    match poll2 {
        Poll::Ready(Ok(val)) => assert_eq!(val, 888),
        other => panic!("Expected Poll::Ready(Ok(888)), got {other:?}"),
    }

    // 2. Error path
    let (completer_err, mut future_err) = signal_pair::<i64>();
    completer_err.complete_err("failed to compute");
    let poll_err = Pin::new(&mut future_err).poll(&mut cx);
    match poll_err {
        Poll::Ready(Err(msg)) => assert_eq!(msg, "failed to compute"),
        other => panic!("Expected Poll::Ready(Err), got {other:?}"),
    }
}

#[test]
fn test_l10_async_cancellation_oracle_no_wall_clock() {
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);

    // Case A: Cancellation before poll (future dropped immediately without poll)
    {
        let (completer, future) = signal_pair::<String>();
        // Dropping future before poll
        drop(future);
        // Completer completing after future dropped must not panic or fault
        completer.complete_ok("abandoned result".to_string());
    }

    // Case B: Cancellation after completion/ready (future dropped after ready)
    {
        let (completer, mut future) = signal_pair::<i32>();
        completer.complete_ok(12345);
        let poll = Pin::new(&mut future).poll(&mut cx);
        assert!(matches!(poll, Poll::Ready(Ok(12345))));
        // Dropping future after ready
        drop(future);
    }

    // Case C: Completer dropped before completion (future abandoned)
    {
        let (completer, mut future) = signal_pair::<i32>();
        drop(completer); // Drop without complete_ok / complete_err
        let poll = Pin::new(&mut future).poll(&mut cx);
        // When completer is dropped without result, future remains pending
        assert!(matches!(poll, Poll::Pending));
        drop(future);
    }
}

// ---------------------------------------------------------------------------
// 4. EmbeddedProgram: Builder, Eval, Call, Gas Budget, Memory Budget
// ---------------------------------------------------------------------------

#[test]
fn test_l10_embedded_program_build_eval_call() {
    let src = r#"
        function square(x: int): int
            return x * x
        end

        function add(a: int, b: int): int
            return a + b
        end
    "#;

    let mut program = EmbeddedProgram::builder()
        .module("math_mod", src)
        .entry_module("math_mod")
        .compile()
        .expect("Program should compile");

    // Call typed function
    let res: i64 = program
        .call_typed("math_mod.square", 9i64)
        .expect("call square should succeed");
    assert_eq!(res, 81);

    let sum: i64 = program
        .call_typed("math_mod.add", (15i64, 25i64))
        .expect("call add should succeed");
    assert_eq!(sum, 40);
}

#[test]
fn test_l10_embedded_program_gas_budget() {
    let src = r#"
        function loop_gas(n: int): int
            local sum = 0
            for i = 1, n do
                sum = sum + i
            end
            return sum
        end
    "#;

    let mut program = EmbeddedProgram::builder()
        .module("gas_mod", src)
        .entry_module("gas_mod")
        .compile()
        .unwrap();

    // 1. With plenty of gas budget
    program.set_gas_budget(100_000);
    assert_eq!(program.gas_used(), 0);
    let res: i64 = program.call_typed("gas_mod.loop_gas", 50i64).unwrap();
    assert_eq!(res, 1275);
    assert!(program.gas_used() > 0);
    assert!(program.gas_remaining().unwrap() > 0);

    // Reset and clear gas
    program.reset_gas_counter();
    assert_eq!(program.gas_used(), 0);
    program.clear_gas_budget();
    assert_eq!(program.gas_remaining(), None);

    // 2. Exhaust gas budget
    program.set_gas_budget(10); // Very low gas budget
    let err = program
        .call_typed::<i64, i64>("gas_mod.loop_gas", 1000i64)
        .unwrap_err();
    assert!(
        err.to_string().contains("gas") || err.to_string().contains("exhausted"),
        "Expected gas exhaustion, got: {err}"
    );
}

#[test]
fn test_l10_embedded_program_memory_budget() {
    let mut program = EmbeddedProgram::builder()
        .module("empty", "function f(): int return 42 end")
        .entry_module("empty")
        .compile()
        .unwrap();

    // Set memory budget in bytes
    program
        .set_memory_budget_bytes(1024 * 1024)
        .expect("Should set 1MB memory budget");

    // Set memory budget in KB
    program
        .set_memory_budget_kb(512)
        .expect("Should set 512KB memory budget");

    // Clear memory budget
    program
        .clear_memory_budget()
        .expect("Should clear memory budget");

    // Zero memory budget immediately traps on heap allocation
    let mut alloc_prog = EmbeddedProgram::builder()
        .module(
            "mem_test",
            r#"
            function make_array(): Array<int>
                local a: Array<int> = [1, 2, 3]
                return a
            end
        "#,
        )
        .entry_module("mem_test")
        .compile()
        .unwrap();

    alloc_prog.set_memory_budget_bytes(0).unwrap();
    let err = alloc_prog
        .call_typed::<(), Value>("mem_test.make_array", ())
        .unwrap_err();
    assert!(
        err.to_string().to_lowercase().contains("memory budget"),
        "Expected memory budget error, got: {err}"
    );
}

#[test]
fn test_l10_precompiled_bytecode_roundtrip() {
    let builder = EmbeddedBuilder::new()
        .module("test", "function get_const(): int return 100 end")
        .entry_module("test");

    let precompiled: CompiledBytecode = EmbeddedProgram::precompile(builder)
        .expect("Precompile should succeed");

    assert_eq!(precompiled.entry_module(), "test");
    assert!(precompiled.function_count() > 0);

    let mut prog = EmbeddedProgram::from_compiled(&precompiled)
        .expect("Instantiate from precompiled bytecode should succeed");

    let val: i64 = prog.call_typed("test.get_const", ()).unwrap();
    assert_eq!(val, 100);
}

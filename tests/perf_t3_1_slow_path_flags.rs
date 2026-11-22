//! PERF-T3-1: VM Interpreter `slow_path_flags: u8` + Fast `HAS_ANY_TRACE_HOOK`.
//!
//! Verifies:
//! 1. `vm.slow_path_flags() == 0` in plain top-level `vm.call()` script mode (even though
//!    `execute_call_frame` sets `call_until_depth = Some(0)`).
//! 2. Accurate flag transitions on `set_gas_budget`, `clear_gas_budget`, `spawn_task_value`,
//!    `resume_task_handle`, `resume_task_handle_with_quantum`, and nested `call_until_depth` /
//!    `pending_return_value` reentrant calls.
//! 3. Accurate `HAS_ANY_TRACE_HOOK` transitions on `set_instruction_hook`, `set_call_hook`,
//!    `set_return_hook`, `set_error_hook`, and `clear_trace_hooks`.
//! 4. Exact `vm.gas_used()` increment by `1` per instruction even when `gas_limit.is_none()`
//!    and `slow_path_flags == 0`.
//! 5. Adversarial test 1: trace hook installed on VM1 while executing VM2 on the same thread
//!    triggers `HAS_ANY_TRACE_HOOK` (with panic-safe RAII cleanup).
//! 6. Adversarial test 2: partial hook clear keeps `HAS_ANY_TRACE_HOOK == true` until the final
//!    hook is cleared.
//! 7. Adversarial test 3: native host callback mutating `set_gas_budget` mid-execution syncs
//!    `slow_path_flags` immediately and triggers OutOfGas cleanly.

use std::cell::RefCell;
use std::rc::Rc;

use lust::bytecode::compiler::Compiler;
use lust::bytecode::{Function, Instruction, NativeCallResult, Value, native_fn};
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{
    SLOW_PATH_CURRENT_TASK, SLOW_PATH_GAS_LIMIT, SLOW_PATH_QUANTUM_FUEL, TaskState, VM,
};
use lust::LustError;

/// Panic-safe RAII guard ensuring thread-local trace hooks are cleared even on test panic.
struct TraceHookCleanupGuard;

impl TraceHookCleanupGuard {
    fn new() -> Self {
        let mut vm = VM::new();
        vm.clear_trace_hooks();
        Self
    }
}

impl Drop for TraceHookCleanupGuard {
    fn drop(&mut self) {
        let mut vm = VM::new();
        vm.clear_trace_hooks();
    }
}

fn compile_source(source: &str) -> lust::Result<Vec<Function>> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize()?;
    let items = Parser::new(tokens).parse()?;
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    compiler.compile_module(&items)
}

fn plain_vm() -> VM {
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    VM::with_config(&config)
}

#[test]
fn test_perf_t3_1_slow_path_flags_sync() {
    let _hook_guard = TraceHookCleanupGuard::new();

    // 1. Fresh VM starts with slow_path_flags == 0 and has_trace_hook() == false.
    let mut vm = plain_vm();
    assert_eq!(vm.slow_path_flags(), 0);
    assert!(!vm.has_trace_hook());

    // 2. Exact gas_used increment by 1 per instruction in plain script mode (slow_path_flags == 0).
    // Build a deterministic 5-instruction function:
    //   0: LoadConst R0, 10
    //   1: LoadConst R1, 32
    //   2: AddInt    R2, R0, R1
    //   3: Move      R3, R2
    //   4: Return    R3
    let mut exact_fn = Function::new("exact_five".to_string(), 0, false);
    exact_fn.register_count = 4;
    let c10 = exact_fn.chunk.add_constant(Value::Int(10));
    let c32 = exact_fn.chunk.add_constant(Value::Int(32));
    exact_fn.chunk.emit(Instruction::LoadConst(0, c10), 1);
    exact_fn.chunk.emit(Instruction::LoadConst(1, c32), 1);
    exact_fn.chunk.emit(Instruction::AddInt(2, 0, 1), 1);
    exact_fn.chunk.emit(Instruction::Move(3, 2), 1);
    exact_fn.chunk.emit(Instruction::Return(3), 1);
    vm.load_functions(vec![exact_fn]);
    vm.reset_gas_counter();
    let res = vm.call("exact_five", vec![]).expect("exact_five must succeed");
    assert_eq!(res, Value::Int(42));
    assert_eq!(
        vm.gas_used(),
        5,
        "plain script mode (slow_path_flags == 0) must still charge 1 gas per instruction"
    );
    assert_eq!(vm.slow_path_flags(), 0);

    // 3. Top-level vm.call() hot loop MUST observe slow_path_flags == 0 even though
    //    VM::execute_call_frame sets call_until_depth = Some(0), AND nested vm.call()
    //    (exercising call_until_depth + pending_return_value) must return accurately.
    let observed_outer_flags = Rc::new(RefCell::new(Vec::<u8>::new()));
    let observed_inner_flags = Rc::new(RefCell::new(Vec::<u8>::new()));

    let src = r#"
        extern
            function record_inner_flags(): int
            function call_inner_from_host(x: int): int
        end

        function inner(x: int): int
            local _ = record_inner_flags()
            return x * 3 + 1
        end

        function outer(n: int): int
            local total = 0
            local i = 0
            while i < n do
                if i == 2 then
                    total = total + call_inner_from_host(i)
                else
                    total = total + i
                end
                i = i + 1
            end
            return total
        end
    "#;
    let funcs = compile_source(src).expect("compile must succeed");
    let mut vm = plain_vm();
    vm.load_functions(funcs);

    let outer_clone = observed_outer_flags.clone();
    vm.register_native(
        "call_inner_from_host",
        Value::NativeFunction(native_fn(move |args| {
            let arg = args.first().cloned().unwrap_or(Value::Int(0));
            VM::with_current(|cur_vm| {
                // Inside top-level vm.call(), call_until_depth == Some(0) but slow_path_flags MUST be 0!
                outer_clone.borrow_mut().push(cur_vm.slow_path_flags());
                let nested_ret = cur_vm
                    .call("inner", vec![arg])
                    .map_err(|e| e.to_string())?;
                // After nested return finishes, slow_path_flags MUST be back to 0!
                outer_clone.borrow_mut().push(cur_vm.slow_path_flags());
                Ok(NativeCallResult::Return(nested_ret))
            })
        })),
    );

    let inner_clone = observed_inner_flags.clone();
    vm.register_native(
        "record_inner_flags",
        Value::NativeFunction(native_fn(move |_args| {
            VM::with_current(|cur_vm| {
                inner_clone.borrow_mut().push(cur_vm.slow_path_flags());
                Ok(NativeCallResult::Return(Value::Int(0)))
            })
        })),
    );

    // Expected: i=0 (0) + i=1 (1) + i=2 (inner(2) = 7) + i=3 (3) + i=4 (4) = 15
    let out = vm.call("outer", vec![Value::Int(5)]).expect("outer must succeed");
    assert_eq!(out, Value::Int(15));
    assert_eq!(
        *observed_outer_flags.borrow(),
        vec![0, 0],
        "top-level vm.call() must have slow_path_flags == 0 before and after nested vm.call()"
    );
    assert_eq!(
        *observed_inner_flags.borrow(),
        vec![0],
        "nested vm.call() body must also have slow_path_flags == 0 until Return sets pending_return_value"
    );
    assert_eq!(vm.slow_path_flags(), 0);

    // 4. set_gas_budget / clear_gas_budget transitions.
    vm.set_gas_budget(10_000);
    assert_ne!(vm.slow_path_flags() & SLOW_PATH_GAS_LIMIT, 0);
    vm.clear_gas_budget();
    assert_eq!(vm.slow_path_flags() & SLOW_PATH_GAS_LIMIT, 0);
    assert_eq!(vm.slow_path_flags(), 0);

    // 5. Trace hook transitions on all 4 setters + clear_trace_hooks.
    let instr_hits = Rc::new(RefCell::new(0usize));
    let call_hits = Rc::new(RefCell::new(0usize));
    let return_hits = Rc::new(RefCell::new(0usize));
    let error_hits = Rc::new(RefCell::new(0usize));

    let ih = instr_hits.clone();
    vm.set_instruction_hook(Some(Box::new(move |_, _, _| {
        *ih.borrow_mut() += 1;
    })));
    assert!(vm.has_trace_hook());

    let ch = call_hits.clone();
    vm.set_call_hook(Some(Box::new(move |_, _, _| {
        *ch.borrow_mut() += 1;
    })));
    assert!(vm.has_trace_hook());

    let rh = return_hits.clone();
    vm.set_return_hook(Some(Box::new(move |_, _, _| {
        *rh.borrow_mut() += 1;
    })));
    assert!(vm.has_trace_hook());

    let eh = error_hits.clone();
    vm.set_error_hook(Some(Box::new(move |_, _, _| {
        *eh.borrow_mut() += 1;
    })));
    assert!(vm.has_trace_hook());

    let _ = vm.call("outer", vec![Value::Int(3)]).expect("traced call must succeed");
    assert!(*instr_hits.borrow() > 0, "instruction_hook must fire");
    assert!(*return_hits.borrow() > 0, "return_hook must fire");

    vm.clear_trace_hooks();
    assert!(!vm.has_trace_hook());
    let instr_before = *instr_hits.borrow();
    let _ = vm.call("outer", vec![Value::Int(3)]).expect("untraced call must succeed");
    assert_eq!(
        *instr_hits.borrow(),
        instr_before,
        "cleared hooks must not fire"
    );

    // 6. Task execution & quantum fuel transitions (spawn_task_value + resume_task_handle_with_quantum).
    let task_flags = Rc::new(RefCell::new(Vec::<u8>::new()));
    let task_src = r#"
        extern
            function record_task_flags(): int
        end

        function worker(): int
            local _ = record_task_flags()
            local s = 0
            local i = 0
            while i < 100 do
                s = s + i
                i = i + 1
            end
            return s
        end
    "#;
    let task_funcs = compile_source(task_src).expect("compile worker must succeed");
    let mut task_vm = plain_vm();
    task_vm.load_functions(task_funcs);
    let tf = task_flags.clone();
    task_vm.register_native(
        "record_task_flags",
        Value::NativeFunction(native_fn(move |_args| {
            VM::with_current(|cur_vm| {
                tf.borrow_mut().push(cur_vm.slow_path_flags());
                Ok(NativeCallResult::Return(Value::Int(0)))
            })
        })),
    );

    // 6a. spawn_task_value (runs to completion with current_task set, no quantum_fuel).
    assert_eq!(task_vm.slow_path_flags(), 0);
    let h1 = task_vm
        .spawn_task_value(Value::Function(0), vec![])
        .expect("spawn_task_value must succeed");
    assert_eq!(
        task_vm.get_task_instance(h1).unwrap().state,
        TaskState::Completed
    );
    assert_eq!(
        task_vm.slow_path_flags(),
        0,
        "slow_path_flags must return to 0 after spawn_task_value finishes"
    );
    assert_eq!(
        *task_flags.borrow(),
        vec![SLOW_PATH_CURRENT_TASK],
        "inside spawn_task_value, SLOW_PATH_CURRENT_TASK must be set"
    );

    // 6b. create_task_value + resume_task_handle_with_quantum (sets CURRENT_TASK | QUANTUM_FUEL).
    task_flags.borrow_mut().clear();
    let h2 = task_vm
        .create_task_value(Value::Function(0), vec![])
        .expect("create_task_value must succeed");
    assert_eq!(task_vm.slow_path_flags(), 0);
    let state = task_vm
        .resume_task_handle_with_quantum(h2, 10, None)
        .expect("quantum resume must succeed");
    assert!(
        matches!(state, TaskState::Yielded(_)),
        "10-instruction quantum must preempt worker loop, got {:?}",
        state
    );
    assert_eq!(
        task_vm.slow_path_flags(),
        0,
        "slow_path_flags must return to 0 after quantum preemption"
    );
    assert_eq!(
        *task_flags.borrow(),
        vec![SLOW_PATH_CURRENT_TASK | SLOW_PATH_QUANTUM_FUEL],
        "inside quantum-bounded task, both CURRENT_TASK and QUANTUM_FUEL bits must be set"
    );
}

#[test]
fn test_perf_t3_1_cross_vm_trace_hook_same_thread() {
    let _hook_guard = TraceHookCleanupGuard::new();

    let mut vm1 = plain_vm();
    let mut vm2 = plain_vm();

    let funcs = compile_source(
        r#"
        function work(n: int): int
            local acc = 0
            local i = 0
            while i < n do
                acc = acc + i
                i = i + 1
            end
            return acc
        end
    "#,
    )
    .expect("compile must succeed");
    vm2.load_functions(funcs);

    let hits = Rc::new(RefCell::new(0usize));
    let hits_clone = hits.clone();
    // Install hook on VM1 only!
    vm1.set_instruction_hook(Some(Box::new(move |func, _, _| {
        if func == "work" {
            *hits_clone.borrow_mut() += 1;
        }
    })));

    // VM2 on the same thread must see HAS_ANY_TRACE_HOOK == true and dispatch the hook.
    assert!(vm1.has_trace_hook());
    assert!(vm2.has_trace_hook());
    let res = vm2.call("work", vec![Value::Int(10)]).expect("vm2 work must succeed");
    assert_eq!(res, Value::Int(45));
    assert!(
        *hits.borrow() > 0,
        "trace hook installed on VM1 must fire during VM2 execution on the same thread"
    );

    // Clearing on VM1 clears thread-local HAS_ANY_TRACE_HOOK for VM2 as well.
    vm1.clear_trace_hooks();
    assert!(!vm1.has_trace_hook());
    assert!(!vm2.has_trace_hook());
}

#[test]
fn test_perf_t3_1_partial_trace_hook_clear() {
    let _hook_guard = TraceHookCleanupGuard::new();

    let mut vm = plain_vm();
    let funcs = compile_source(
        r#"
        function helper(x: int): int
            return x + 1
        end
        function main(): int
            return helper(41)
        end
    "#,
    )
    .expect("compile must succeed");
    vm.load_functions(funcs);

    let instr_count = Rc::new(RefCell::new(0usize));
    let call_count = Rc::new(RefCell::new(0usize));
    let ret_count = Rc::new(RefCell::new(0usize));
    let err_count = Rc::new(RefCell::new(0usize));

    let ic = instr_count.clone();
    vm.set_instruction_hook(Some(Box::new(move |_, _, _| *ic.borrow_mut() += 1)));
    let cc = call_count.clone();
    vm.set_call_hook(Some(Box::new(move |_, _, _| *cc.borrow_mut() += 1)));
    let rc = ret_count.clone();
    vm.set_return_hook(Some(Box::new(move |_, _, _| *rc.borrow_mut() += 1)));
    let ec = err_count.clone();
    vm.set_error_hook(Some(Box::new(move |_, _, _| *ec.borrow_mut() += 1)));
    assert!(vm.has_trace_hook());

    // Clear instruction_hook: 3 hooks remain -> HAS_ANY_TRACE_HOOK MUST stay true.
    vm.set_instruction_hook(None);
    assert!(vm.has_trace_hook());
    let res = vm.call("main", vec![]).expect("main must succeed");
    assert_eq!(res, Value::Int(42));
    assert_eq!(*instr_count.borrow(), 0, "cleared instruction_hook must not fire");
    assert!(*call_count.borrow() > 0, "remaining call_hook must still fire");
    assert!(*ret_count.borrow() > 0, "remaining return_hook must still fire");

    // Clear call_hook: 2 hooks remain -> HAS_ANY_TRACE_HOOK MUST stay true.
    vm.set_call_hook(None);
    assert!(vm.has_trace_hook());

    // Clear return_hook: 1 hook (error_hook) remains -> HAS_ANY_TRACE_HOOK MUST stay true.
    vm.set_return_hook(None);
    assert!(vm.has_trace_hook());

    // Clear final hook (error_hook): 0 hooks remain -> HAS_ANY_TRACE_HOOK MUST become false!
    vm.set_error_hook(None);
    assert!(
        !vm.has_trace_hook(),
        "clearing the final remaining trace hook must reset HAS_ANY_TRACE_HOOK to false"
    );
}

#[test]
fn test_perf_t3_1_native_callback_sets_gas_budget_mid_execution() {
    let _hook_guard = TraceHookCleanupGuard::new();

    let mut vm = plain_vm();
    let funcs = compile_source(
        r#"
        extern
            function arm_tight_gas_budget(): int
        end

        function spin_after_arm(): int
            local i = 0
            while i < 100000 do
                if i == 1 then
                    local _ = arm_tight_gas_budget()
                end
                i = i + 1
            end
            return i
        end
    "#,
    )
    .expect("compile must succeed");
    vm.load_functions(funcs);

    let flags_before_set = Rc::new(RefCell::new(None::<u8>));
    let flags_after_set = Rc::new(RefCell::new(None::<u8>));
    let fb = flags_before_set.clone();
    let fa = flags_after_set.clone();

    vm.register_native(
        "arm_tight_gas_budget",
        Value::NativeFunction(native_fn(move |_args| {
            VM::with_current(|cur_vm| {
                *fb.borrow_mut() = Some(cur_vm.slow_path_flags());
                let used = cur_vm.gas_used();
                // Arm a tight gas budget mid-execution: only 20 more instructions allowed.
                cur_vm.set_gas_budget(used + 20);
                *fa.borrow_mut() = Some(cur_vm.slow_path_flags());
                Ok(NativeCallResult::Return(Value::Int(1)))
            })
        })),
    );

    assert_eq!(vm.slow_path_flags(), 0);
    vm.reset_gas_counter();
    let err = vm
        .call("spin_after_arm", vec![])
        .expect_err("mid-execution set_gas_budget must trigger OutOfGas");

    assert_eq!(
        *flags_before_set.borrow(),
        Some(0),
        "before set_gas_budget in callback, slow_path_flags must be 0"
    );
    assert_eq!(
        *flags_after_set.borrow(),
        Some(SLOW_PATH_GAS_LIMIT),
        "set_gas_budget inside native callback must immediately set SLOW_PATH_GAS_LIMIT"
    );

    match err {
        LustError::RuntimeError { message } | LustError::RuntimeErrorWithTrace { message, .. } => {
            assert!(
                message.to_lowercase().contains("out of gas"),
                "expected Out of gas error, got: {message}"
            );
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

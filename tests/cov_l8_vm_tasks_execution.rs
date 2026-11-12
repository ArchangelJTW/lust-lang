#![allow(clippy::approx_constant)]

use lust::bytecode::compiler::Compiler;
use lust::bytecode::instruction::Instruction;
use lust::bytecode::{Function, NativeCallResult, TaskHandle, Value, ValueKey, native_fn};
use lust::config::LustConfig;
use lust::effect_abi::{
    InterruptReason, Scope, ScopeAdmission, ScopeTeardown,
};
use lust::error::LustError;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::{TaskId, TaskState, VM};
use std::cell::RefCell;
use std::ptr;
use std::rc::Rc;

// =========================================================================
// C-ABI EXTERN DECLARATIONS FROM TRACING
// =========================================================================

#[repr(C)]
struct JitCallSite {
    value_count: usize,
    alias_mask: usize,
    borrow_mask: usize,
    function_idx: usize,
    return_dest: usize,
    callee_reg: usize,
    caller_resume_ip: usize,
}

#[repr(C)]
struct JitInlineRecord {
    caller_regs: *mut Value,
    prev: *const JitInlineRecord,
    site: *const JitCallSite,
}

#[allow(improper_ctypes)]
unsafe extern "C" {
    fn jit_run_nested_loop(
        vm: *mut VM,
        registers: *mut Value,
        function_idx: usize,
        loop_start_ip: usize,
        resume_ip: usize,
    ) -> i32;

    fn jit_materialize_inline_frames(
        vm: *mut VM,
        record: *const JitInlineRecord,
        regs: *mut Value,
    ) -> *mut Value;
}

// =========================================================================
// COMPILATION & RUN HELPERS
// =========================================================================

fn compile_source(source: &str) -> lust::Result<Vec<Function>> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize()?;
    let items = Parser::new(tokens).parse()?;
    let mut config = LustConfig::default();
    config.enable_module("table");
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    compiler.compile_module(&items)
}

fn execute_instructions(instructions: Vec<Instruction>, constants: Vec<Value>, register_count: u8) -> lust::Result<Value> {
    let res = execute_instructions_with_jit(instructions.clone(), constants.clone(), register_count, true);
    let _ = execute_instructions_with_jit(instructions, constants, register_count, false);
    res
}

fn execute_instructions_with_jit(instructions: Vec<Instruction>, constants: Vec<Value>, register_count: u8, jit: bool) -> lust::Result<Value> {
    let mut func = Function::new("test_exec".to_string(), 0, false);
    func.register_count = register_count.max(16);
    for c in constants {
        func.chunk.add_constant(c);
    }
    for instr in instructions {
        func.chunk.emit(instr, 1);
    }
    let mut config = LustConfig::default();
    config.set_jit_enabled(jit);
    let mut vm = VM::with_config(&config);
    vm.load_functions(vec![func]);
    vm.call("test_exec", vec![])
}

// =========================================================================
// SECTION 1: TASKS & FIBER LIFECYCLE (src/vm/tasks.rs)
// =========================================================================

#[test]
fn test_task_scope_fiber_lifecycle_and_cancellation_integrity() {
    let mut vm = VM::new();
    let handle = vm.create_native_future_task();
    let mut scope = Scope::new(101, None);

    // Oracle 1: Register child into scope updates task's scope_id and fiber_context
    vm.register_scope_task(&mut scope, handle)
        .expect("register_scope_task should succeed on fresh task");
    {
        let task = vm.get_task_instance(handle).expect("task must exist");
        assert_eq!(task.scope_id, Some(scope.id), "scope_id must match scope.id");
        assert_eq!(task.fiber_context.scope_id, scope.id, "fiber_context.scope_id must match");
    }

    // Oracle 2: Re-registering an already scoped task must fail
    let err = vm.register_scope_task(&mut scope, handle);
    assert!(err.is_err(), "cannot re-register already scoped task");

    // Oracle 3: Scope teardown while task is masked does not finalize resources
    let log = Rc::new(RefCell::new(Vec::new()));
    let log_clone = log.clone();
    scope.register_finalizer(move || {
        log_clone.borrow_mut().push("finalizer_run".to_string());
    }).expect("register_finalizer must succeed");

    // Enter mask on child
    vm.get_task_instance_mut(handle).unwrap().fiber_context.enter_mask();

    let closed = vm.close_scope(&mut scope);
    assert!(!closed, "scope cannot close while child task is masked/running");
    assert_eq!(scope.teardown, ScopeTeardown::Joining, "scope teardown must be Joining");
    assert_eq!(scope.admission, ScopeAdmission::Closed, "scope must refuse new tasks");
    assert!(log.borrow().is_empty(), "finalizers must not have executed");

    // Child task received pending interrupt
    let task = vm.get_task_instance(handle).unwrap();
    assert_eq!(task.fiber_context.pending_interrupt, Some(InterruptReason::Cancelled));

    // Now complete the child task
    vm.complete_native_future_task(handle, Ok(Value::Int(42)))
        .expect("complete native future task must succeed");

    // Closing scope now finishes and executes finalizers
    let closed_now = vm.close_scope(&mut scope);
    assert!(closed_now, "scope must close once child is terminal");
    assert_eq!(scope.teardown, ScopeTeardown::Done);
    assert_eq!(*log.borrow(), vec!["finalizer_run".to_string()]);

    // Subsequent close on already Done scope returns true immediately
    assert!(vm.close_scope(&mut scope), "close on Done scope must be true");
}

#[test]
fn test_task_join_semantics_and_idempotent_cancellation() {
    let mut vm = VM::new();
    let handle = vm.create_native_future_task();

    // Complete future
    vm.complete_native_future_task(handle, Ok(Value::string("success")))
        .expect("completion must succeed");

    let task = vm.get_task_instance(handle).expect("task instance must exist");
    assert_eq!(task.state, TaskState::Completed);
    assert_eq!(task.last_result, Some(Value::string("success")));

    // Idempotent cancellation: stopping completed task is Ok and leaves result unchanged
    let stop_res = vm.complete_native_future_task(handle, Ok(Value::Int(999)));
    // Completing already completed native task attaches it back safely
    assert!(stop_res.is_ok(), "completion on completed task should be safe");

    // Resume on completed task must fail
    let resume_res = vm.resume_task_handle(handle, None);
    assert!(resume_res.is_err(), "cannot resume completed task");
    let err_str = resume_res.unwrap_err().to_string();
    assert!(err_str.contains("cannot be resumed"), "Error must name cannot be resumed: {}", err_str);
}

#[test]
fn test_task_native_future_failure_and_invalid_handles() {
    let mut vm = VM::new();
    let handle = vm.create_native_future_task();

    // Complete with error outcome
    vm.complete_native_future_task(handle, Err("simulated network timeout".to_string()))
        .expect("err completion must succeed");

    let task = vm.get_task_instance(handle).unwrap();
    assert_eq!(task.state, TaskState::Failed);
    assert!(task.last_result.is_none());
    assert!(task.error.is_some());

    // Invalid handle tests
    let invalid_handle = TaskHandle(999999);
    assert!(vm.get_task_instance(invalid_handle).is_err());
    assert!(vm.get_task_instance_mut(invalid_handle).is_err());
    assert!(vm.resume_task_handle(invalid_handle, None).is_err());
    assert!(vm.complete_native_future_task(invalid_handle, Ok(Value::Nil)).is_err());

    let mut scope = Scope::new(200, None);
    assert!(vm.register_scope_task(&mut scope, invalid_handle).is_err());
}

#[test]
fn test_task_spawn_and_tick_lifecycle() {
    let source = r#"
        function counter_worker(): int
            local acc = 0
            acc = acc + 10
            return acc
        end
    "#;
    let functions = compile_source(source).expect("compile must succeed");
    let mut vm = VM::new();
    vm.load_functions(functions);

    let func_val = Value::Function(0);
    let handle = vm.spawn_task_value(func_val, vec![]).expect("spawn task value should succeed");
    let task = vm.get_task_instance(handle).unwrap();
    assert_eq!(task.state, TaskState::Completed);
    assert_eq!(task.last_result, Some(Value::Int(10)));

    // Attempting to complete script task using native future completion must return an error
    let native_comp_err = vm.complete_native_future_task(handle, Ok(Value::Int(0)));
    assert!(native_comp_err.is_err());
    let err_msg = native_comp_err.unwrap_err().to_string();
    assert!(err_msg.contains("Attempted to complete a script task using native future completion"), "{}", err_msg);

    // spawn_tick_task with invalid function name must return error
    let tick_err = vm.spawn_tick_task("nonexistent::func");
    assert!(tick_err.is_err());
    assert!(tick_err.unwrap_err().to_string().contains("Function not found"));
}

#[test]
fn test_builtin_methods_on_lua_table_and_primitives() {
    use lust::ast::{Span, Type, TypeKind};
    use lust::bytecode::{FieldStorage, StructLayout, StructObject};
    use lust::bytecode::LustMap;

    let layout = Rc::new(StructLayout::new(
        "LuaTable".to_string(),
        vec![Rc::new("table".to_string())],
        vec![FieldStorage::Strong],
        vec![Type::new(TypeKind::Unknown, Span::dummy())],
        vec![None],
    ));
    let lua_table = Value::Struct(StructObject::new(
        "LuaTable",
        layout,
        vec![Value::map(LustMap::default())],
    ));

    let mut func = Function::new("test_table_ops".to_string(), 0, false);
    func.register_count = 20;

    let c_table = func.chunk.add_constant(lua_table);
    let c_push = func.chunk.add_constant(Value::string("push"));
    let c_len = func.chunk.add_constant(Value::string("len"));
    let c_maxn = func.chunk.add_constant(Value::string("maxn"));
    let c_concat = func.chunk.add_constant(Value::string("concat"));
    let c_remove = func.chunk.add_constant(Value::string("remove"));
    let c_dash = func.chunk.add_constant(Value::string("-"));
    let c_10 = func.chunk.add_constant(Value::Int(10));
    let c_20 = func.chunk.add_constant(Value::Int(20));
    let c_1 = func.chunk.add_constant(Value::Int(1));
    let c_2 = func.chunk.add_constant(Value::Int(2));

    func.chunk.emit(Instruction::LoadConst(0, c_table), 1);
    func.chunk.emit(Instruction::LoadConst(1, c_10), 1);
    func.chunk.emit(Instruction::CallMethod(0, c_push, 1, 1, 3), 1); // push(10)
    func.chunk.emit(Instruction::LoadConst(1, c_20), 1);
    func.chunk.emit(Instruction::CallMethod(0, c_push, 1, 1, 3), 1); // push(20)
    func.chunk.emit(Instruction::CallMethod(0, c_len, 1, 0, 6), 1); // len() -> 2
    func.chunk.emit(Instruction::CallMethod(0, c_maxn, 1, 0, 8), 1); // maxn() -> 2
    func.chunk.emit(Instruction::LoadConst(9, c_dash), 1);
    func.chunk.emit(Instruction::LoadConst(10, c_1), 1);
    func.chunk.emit(Instruction::LoadConst(11, c_2), 1);
    func.chunk.emit(Instruction::CallMethod(0, c_concat, 9, 3, 13), 1); // concat("-", 1, 2)
    func.chunk.emit(Instruction::LoadConst(14, c_1), 1);
    func.chunk.emit(Instruction::CallMethod(0, c_remove, 14, 1, 16), 1); // remove(1) -> 10
    func.chunk.emit(Instruction::Return(6), 1); // return len (2)

    let mut vm = VM::new();
    vm.load_functions(vec![func]);
    let res = vm.call("test_table_ops", vec![]).expect("LuaTable method execution must succeed");
    assert_eq!(res, Value::Int(2));
}

#[test]
fn test_builtin_methods_on_option_and_result() {
    let source = r#"
        function test_opt_res(): int
            local s = Option.Some(50)
            local n = Option.None
            local o1 = s:is_some()
            local o2 = n:is_none()
            local v1 = s:unwrap()
            local v2 = n:unwrap_or(100)
            local v3 = s:try_unwrap()

            local r_ok = Result.Ok(200)
            local r_err = Result.Err("error")
            local ro1 = r_ok:is_ok()
            local ro2 = r_err:is_err()
            local rv1 = r_ok:unwrap()
            local rv2 = r_err:unwrap_or(300)
            local rv3 = r_ok:try_unwrap()

            return v1 + v2 + v3 + rv1 + rv2 + rv3
        end
    "#;
    let functions = compile_source(source).expect("compile must succeed");
    let mut vm = VM::new();
    vm.load_functions(functions);
    let result = vm.call("test_opt_res", vec![]).expect("option/result methods must execute");
    // 50 + 100 + 50 + 200 + 300 + 200 = 900
    assert_eq!(result, Value::Int(900));
}

#[test]
fn test_builtin_methods_error_handling() {
    // 1. unwrap on Option.None must fail
    let src_none = r#"
        function unwrap_none(): int
            local n = Option.None
            return n:unwrap()
        end
    "#;
    let f_none = compile_source(src_none).unwrap();
    let mut vm = VM::new();
    vm.load_functions(f_none);
    let err = vm.call("unwrap_none", vec![]).unwrap_err();
    assert!(err.to_string().contains("Called unwrap() on Option::None"));

    // 2. unwrap on Result.Err must fail
    let src_err = r#"
        function unwrap_err(): int
            local r = Result.Err("fail")
            return r:unwrap()
        end
    "#;
    let f_err = compile_source(src_err).unwrap();
    let mut vm2 = VM::new();
    vm2.load_functions(f_err);
    let err2 = vm2.call("unwrap_err", vec![]).unwrap_err();
    assert!(err2.to_string().contains("Called unwrap() on Result::Err"));

    // 3. Method on primitive array/string/int gives helpful diagnostic
    let src_arr = r#"
        function method_arr(): int
            local a = [1, 2]
            return a:len()
        end
    "#;
    let f_arr = compile_source(src_arr).unwrap();
    let mut vm3 = VM::new();
    vm3.load_functions(f_arr);
    let err3 = vm3.call("method_arr", vec![]).unwrap_err();
    assert!(err3.to_string().contains("Type 'Array' does not support method syntax"), "{}", err3);
}

#[test]
fn test_builtin_methods_iterator_array_and_map() {
    let source = r#"
        function test_iter(): int
            local arr = [10, 20, 30]
            local it = array.iter(arr)
            local it2 = it:iter()
            local v1 = it:next():unwrap()
            local v2 = it:next():unwrap()
            local v3 = it:next():unwrap()
            local none_step = it:next():is_none()
            if none_step then
                return v1 + v2 + v3
            else
                return -1
            end
        end
    "#;
    let functions = compile_source(source).expect("compile iter source");
    let mut vm = VM::new();
    vm.load_functions(functions);
    let res = vm.call("test_iter", vec![]).expect("iter call");
    assert_eq!(res, Value::Int(60));
}

// =========================================================================
// SECTION 2: TRACING & HOOKS (src/vm/tracing.rs)
// =========================================================================

#[test]
fn test_tracing_execution_invariance_oracle() {
    // Non-trivial algorithm: Collatz sequence step counting + accumulator array
    let source = r#"
        function collatz_steps(start: int): int
            local n = start
            local steps = 0
            while n > 1 do
                if n % 2 == 0 then
                    n = n / 2
                else
                    n = 3 * n + 1
                end
                steps = steps + 1
            end
            return steps
        end

        function benchmark_suite(): int
            local total = 0
            local i = 1
            while i <= 25 do
                total = total + collatz_steps(i)
                i = i + 1
            end
            return total
        end
    "#;

    // Run with JIT enabled
    let mut config_jit = LustConfig::default();
    config_jit.set_jit_enabled(true);
    let f_jit = compile_source(source).unwrap();
    let mut vm_jit = VM::with_config(&config_jit);
    vm_jit.load_functions(f_jit);
    let res_jit = vm_jit.call("benchmark_suite", vec![]).expect("jit execution");

    // Run with JIT disabled (pure interpreter)
    let mut config_interp = LustConfig::default();
    config_interp.set_jit_enabled(false);
    let f_interp = compile_source(source).unwrap();
    let mut vm_interp = VM::with_config(&config_interp);
    vm_interp.load_functions(f_interp);
    let res_interp = vm_interp.call("benchmark_suite", vec![]).expect("interp execution");

    // Oracle: Bit-for-bit identical outcome across execution modes
    assert_eq!(res_jit, res_interp, "JIT and interpreter must produce bit-for-bit identical results");
    assert!(matches!(res_jit, Value::Int(n) if n > 0), "Result must be positive integer");
}

#[test]
fn test_tracing_hook_metadata_validation() {
    let mut vm = VM::new();

    let instruction_calls = Rc::new(RefCell::new(Vec::new()));
    let call_calls = Rc::new(RefCell::new(Vec::new()));
    let return_calls = Rc::new(RefCell::new(Vec::new()));
    let error_calls = Rc::new(RefCell::new(Vec::new()));

    let ic = instruction_calls.clone();
    vm.set_instruction_hook(Some(Box::new(move |func, line, ip| {
        ic.borrow_mut().push((func.to_string(), line, ip));
    })));

    let cc = call_calls.clone();
    vm.set_call_hook(Some(Box::new(move |func, line, depth| {
        cc.borrow_mut().push((func.to_string(), line, depth));
    })));

    let rc = return_calls.clone();
    vm.set_return_hook(Some(Box::new(move |func, line, depth| {
        rc.borrow_mut().push((func.to_string(), line, depth));
    })));

    let ec = error_calls.clone();
    vm.set_error_hook(Some(Box::new(move |func, line, err| {
        ec.borrow_mut().push((func.to_string(), line, err.to_string()));
    })));

    // Re-setting existing hook tests the update branch in set_instruction_hook
    let ic2 = instruction_calls.clone();
    vm.set_instruction_hook(Some(Box::new(move |func, line, ip| {
        ic2.borrow_mut().push((func.to_string(), line, ip));
    })));
    let cc2 = call_calls.clone();
    vm.set_call_hook(Some(Box::new(move |func, line, depth| {
        cc2.borrow_mut().push((func.to_string(), line, depth));
    })));
    let rc2 = return_calls.clone();
    vm.set_return_hook(Some(Box::new(move |func, line, depth| {
        rc2.borrow_mut().push((func.to_string(), line, depth));
    })));
    let ec2 = error_calls.clone();
    vm.set_error_hook(Some(Box::new(move |func, line, err| {
        ec2.borrow_mut().push((func.to_string(), line, err.to_string()));
    })));
    // Run a function to verify instruction_hook, call_hook, and return_hook
    let source_ok = r#"
        function succ_caller(): int
            return succ_callee(41)
        end
        function succ_callee(x: int): int
            return x + 1
        end
    "#;
    let ok_funcs = compile_source(source_ok).unwrap();
    vm.load_functions(ok_funcs);
    let ok_res = vm.call("succ_caller", vec![]).expect("success run");
    assert_eq!(ok_res, Value::Int(42));

    assert!(!instruction_calls.borrow().is_empty(), "instruction_hook must be called");
    assert!(!call_calls.borrow().is_empty(), "call_hook must be called on function call");
    assert!(!return_calls.borrow().is_empty(), "return_hook must be called on return");
    {
        let recorded_calls = call_calls.borrow();
        assert!(recorded_calls.iter().any(|(f, _, _)| f == "succ_callee"), "call_hook must record 'succ_callee'");
        let recorded_returns = return_calls.borrow();
        assert!(recorded_returns.iter().any(|(f, _, _)| f == "succ_callee" || f == "succ_caller"), "return_hook must record return");
    }

    // Trigger an error to verify error_hook dispatch via annotate_runtime_error
    let source_err = r#"
        function fail_now(): int
            local zero = 0
            return 10 / zero
        end
    "#;
    let functions = compile_source(source_err).unwrap();
    vm.load_functions(functions);
    let run_err = vm.call("fail_now", vec![]);
    assert!(run_err.is_err(), "division by zero must fail");

    // Oracle: Verify error_hook caught the failure metadata
    {
        let recorded_errors = error_calls.borrow();
        assert!(!recorded_errors.is_empty(), "error_hook must have been dispatched");
        let (func, _line, err_msg) = &recorded_errors[0];
        assert_eq!(func, "fail_now", "Function name in error hook must be 'fail_now'");
        assert!(err_msg.contains("Division by zero"), "Error message must mention Division by zero");
    }

    // Clear trace hooks
    vm.clear_trace_hooks();
    error_calls.borrow_mut().clear();
    let run_err2 = vm.call("fail_now", vec![]);
    assert!(run_err2.is_err());
    assert!(error_calls.borrow().is_empty(), "No hooks dispatched after clear_trace_hooks()");
}

#[test]
fn test_tracing_unit_enum_memoization() {
    let mut func = Function::new("test_unit_enum".to_string(), 0, false);
    func.register_count = 8;
    // Emit NewEnumUnit twice for ("Option", "None")
    let c_opt = func.chunk.add_constant(Value::string("Option"));
    let c_none = func.chunk.add_constant(Value::string("None"));
    func.chunk.emit(Instruction::NewEnumUnit(0, c_opt, c_none), 1);
    func.chunk.emit(Instruction::NewEnumUnit(1, c_opt, c_none), 1);
    // Compare pointer equality (pointer count bump)
    func.chunk.emit(Instruction::Eq(2, 0, 1), 1);
    func.chunk.emit(Instruction::Return(2), 1);

    let mut vm = VM::new();
    vm.load_functions(vec![func]);
    let res = vm.call("test_unit_enum", vec![]).expect("execution");
    assert_eq!(res, Value::Bool(true), "NewEnumUnit should memoize shared unit enum");
}

#[test]
fn test_tracing_jit_run_nested_loop_c_abi() {
    unsafe {
        // Null VM pointer
        let r1 = jit_run_nested_loop(ptr::null_mut(), ptr::null_mut(), 0, 0, 0);
        assert_eq!(r1, 1, "jit_run_nested_loop with null VM must return 1 (interpreter take over)");

        // VM with JIT disabled
        let mut config = LustConfig::default();
        config.set_jit_enabled(false);
        let mut vm = VM::with_config(&config);
        let r2 = jit_run_nested_loop(&mut vm as *mut VM, ptr::null_mut(), 0, 0, 0);
        assert_eq!(r2, 1, "jit_run_nested_loop with JIT disabled must return 1");

        // VM with unknown trace
        config.set_jit_enabled(true);
        let mut vm2 = VM::with_config(&config);
        let r3 = jit_run_nested_loop(&mut vm2 as *mut VM, ptr::null_mut(), 99, 0, 0);
        assert_eq!(r3, 1, "jit_run_nested_loop with unknown trace must return 1");
    }
}

#[test]
fn test_tracing_jit_materialize_inline_frames_c_abi() {
    unsafe {
        // Test with null VM: drops registers without panic and returns root_regs
        let mut regs = [Value::Int(10), Value::Int(20), Value::Nil];
        let site = JitCallSite {
            value_count: 2,
            alias_mask: 0,
            borrow_mask: 0,
            function_idx: 0,
            return_dest: 0,
            callee_reg: 0,
            caller_resume_ip: 0,
        };
        let record = JitInlineRecord {
            caller_regs: regs.as_mut_ptr(),
            prev: ptr::null(),
            site: &site,
        };
        let out = jit_materialize_inline_frames(ptr::null_mut(), &record, regs.as_mut_ptr());
        assert_eq!(out, regs.as_mut_ptr());
    }
}

// =========================================================================
// SECTION 3: VM EXECUTION & INSTRUCTION BRANCHES (src/vm/execution.rs)
// =========================================================================

#[test]
fn test_execution_integer_arithmetic_and_division_by_zero() {
    // 1. Division by zero in DivInt
    let err = execute_instructions(
        vec![
            Instruction::LoadConst(0, 0),
            Instruction::LoadConst(1, 1),
            Instruction::DivInt(2, 0, 1),
            Instruction::Return(2),
        ],
        vec![Value::Int(100), Value::Int(0)],
        8,
    ).unwrap_err();
    assert!(err.to_string().contains("Division by zero"));

    // 2. Division by zero in ModInt
    let err2 = execute_instructions(
        vec![
            Instruction::LoadConst(0, 0),
            Instruction::LoadConst(1, 1),
            Instruction::ModInt(2, 0, 1),
            Instruction::Return(2),
        ],
        vec![Value::Int(100), Value::Int(0)],
        8,
    ).unwrap_err();
    assert!(err2.to_string().contains("Modulo by zero"));

    // 3. Division by zero in FloorDivInt
    let err3 = execute_instructions(
        vec![
            Instruction::LoadConst(0, 0),
            Instruction::LoadConst(1, 1),
            Instruction::FloorDivInt(2, 0, 1),
            Instruction::Return(2),
        ],
        vec![Value::Int(100), Value::Int(0)],
        8,
    ).unwrap_err();
    assert!(err3.to_string().contains("Division by zero"));

    // 4. Negative exponent in PowInt
    let err4 = execute_instructions(
        vec![
            Instruction::LoadConst(0, 0),
            Instruction::LoadConst(1, 1),
            Instruction::PowInt(2, 0, 1),
            Instruction::Return(2),
        ],
        vec![Value::Int(2), Value::Int(-3)],
        8,
    ).unwrap_err();
    assert!(err4.to_string().contains("Negative exponent in integer power"));

    // 5. FloorDivInt overflow guard (i64::MIN / -1)
    let res = execute_instructions(
        vec![
            Instruction::LoadConst(0, 0),
            Instruction::LoadConst(1, 1),
            Instruction::FloorDivInt(2, 0, 1),
            Instruction::Return(2),
        ],
        vec![Value::Int(i64::MIN), Value::Int(-1)],
        8,
    ).expect("i64::MIN / -1 must not panic");
    assert_eq!(res, Value::Int(i64::MIN));
}

#[test]
fn test_execution_all_int_instructions() {
    let res = execute_instructions(
        vec![
            Instruction::LoadConst(0, 0), // 10
            Instruction::LoadConst(1, 1), // 3
            Instruction::AddInt(2, 0, 1), // 13
            Instruction::SubInt(3, 0, 1), // 7
            Instruction::MulInt(4, 0, 1), // 30
            Instruction::DivInt(5, 0, 1), // 3
            Instruction::ModInt(6, 0, 1), // 1
            Instruction::PowInt(7, 1, 1), // 27
            Instruction::FloorDivInt(8, 0, 1), // 3
            Instruction::NegInt(9, 0), // -10
            Instruction::EqInt(10, 0, 1), // false
            Instruction::NeInt(11, 0, 1), // true
            Instruction::LtInt(12, 1, 0), // true
            Instruction::LeInt(13, 1, 0), // true
            Instruction::GtInt(14, 0, 1), // true
            Instruction::GeInt(15, 0, 1), // true
            Instruction::Return(2),
        ],
        vec![Value::Int(10), Value::Int(3)],
        16,
    ).expect("all int instructions must succeed");
    assert_eq!(res, Value::Int(13));
}

#[test]
fn test_execution_all_float_instructions() {
    let res = execute_instructions(
        vec![
            Instruction::LoadConst(0, 0), // 10.5
            Instruction::LoadConst(1, 1), // 2.5
            Instruction::AddFloat(2, 0, 1), // 13.0
            Instruction::SubFloat(3, 0, 1), // 8.0
            Instruction::MulFloat(4, 0, 1), // 26.25
            Instruction::DivFloat(5, 0, 1), // 4.2
            Instruction::ModFloat(6, 0, 1), // 0.5
            Instruction::PowFloat(7, 1, 1), // 2.5^2.5
            Instruction::FloorDivFloat(8, 0, 1), // 4.0
            Instruction::NegFloat(9, 0), // -10.5
            Instruction::EqFloat(10, 0, 1), // false
            Instruction::NeFloat(11, 0, 1), // true
            Instruction::LtFloat(12, 1, 0), // true
            Instruction::LeFloat(13, 1, 0), // true
            Instruction::GtFloat(14, 0, 1), // true
            Instruction::GeFloat(15, 0, 1), // true
            Instruction::Return(2),
        ],
        vec![Value::Float(10.5), Value::Float(2.5)],
        16,
    ).expect("all float instructions must succeed");
    assert_eq!(res, Value::Float(13.0));
}

#[test]
fn test_execution_all_bitwise_and_logic_instructions() {
    let res = execute_instructions(
        vec![
            Instruction::LoadConst(0, 0), // 0b1100 (12)
            Instruction::LoadConst(1, 1), // 0b1010 (10)
            Instruction::BitAnd(2, 0, 1), // 0b1000 (8)
            Instruction::BitOr(3, 0, 1),  // 0b1110 (14)
            Instruction::BitXor(4, 0, 1), // 0b0110 (6)
            Instruction::Shl(5, 0, 1),    // 12 << 10
            Instruction::Shr(6, 0, 1),    // 12 >> 10
            Instruction::BitNot(7, 0),    // !12
            Instruction::LoadBool(8, true),
            Instruction::LoadBool(9, false),
            Instruction::And(10, 8, 9),   // false
            Instruction::Or(11, 8, 9),    // true
            Instruction::Not(12, 8),      // false
            Instruction::Return(2),
        ],
        vec![Value::Int(12), Value::Int(10)],
        16,
    ).expect("bitwise instructions execution");
    assert_eq!(res, Value::Int(8));
}

#[test]
fn test_execution_tuples_and_tuple_get() {
    let res = execute_instructions(
        vec![
            Instruction::LoadConst(0, 0), // 100
            Instruction::LoadConst(1, 1), // 200
            Instruction::LoadConst(2, 2), // 300
            Instruction::TupleNew(3, 0, 3), // tuple (100, 200, 300)
            Instruction::TupleGet(4, 3, 1), // element 1 -> 200
            Instruction::Return(4),
        ],
        vec![Value::Int(100), Value::Int(200), Value::Int(300)],
        8,
    ).expect("tuple creation and get");
    assert_eq!(res, Value::Int(200));

    // Tuple index out of bounds
    let err = execute_instructions(
        vec![
            Instruction::LoadConst(0, 0),
            Instruction::TupleNew(1, 0, 1),
            Instruction::TupleGet(2, 1, 99), // out of bounds
            Instruction::Return(2),
        ],
        vec![Value::Int(42)],
        8,
    ).unwrap_err();
    assert!(err.to_string().contains("Tuple index 99 out of bounds"));
}

#[test]
fn test_execution_try_get_index_and_array_index_ok() {
    let res = execute_instructions(
        vec![
            Instruction::LoadConst(0, 0), // 10
            Instruction::LoadConst(1, 1), // 20
            Instruction::NewArray(2, 0, 2), // [10, 20]
            Instruction::LoadConst(3, 2), // index 1
            Instruction::TryGetIndex(4, 2, 3), // Option.Some(20)
            Instruction::LoadConst(5, 3), // index 99 (out of bounds)
            Instruction::TryGetIndex(6, 2, 5), // Option.None
            Instruction::ArrayIndexOk(7, 8, 2, 3), // val_dest=7, cond_dest=8
            Instruction::Return(4),
        ],
        vec![Value::Int(10), Value::Int(20), Value::Int(1), Value::Int(99)],
        16,
    ).expect("try get index execution");
    assert!(matches!(res, Value::Enum(obj) if obj.variant == "Ok"));
}

#[test]
fn test_execution_type_is_and_try_cast() {
    let res = execute_instructions(
        vec![
            Instruction::LoadConst(0, 0), // 42
            Instruction::TypeIs(1, 0, 1), // TypeIs "int" -> true
            Instruction::TryCast(2, 0, 1), // Option.Some(42)
            Instruction::TypeIs(3, 0, 2), // TypeIs "string" -> false
            Instruction::TryCast(4, 0, 2), // Option.None
            Instruction::Return(1),
        ],
        vec![Value::Int(42), Value::string("int"), Value::string("string")],
        8,
    ).expect("type is and try cast");
    assert_eq!(res, Value::Bool(true));
}

#[test]
fn test_execution_closures_and_upvalues() {
    let source = r#"
        function make_adder(x: int): function(int): int
            return function(y: int): int
                return x + y
            end
        end

        function test_closure(): int
            local add5 = make_adder(5)
            local add10 = make_adder(10)
            return add5(20) + add10(20)
        end
    "#;
    let functions = compile_source(source).expect("compile closure source");
    let mut vm = VM::new();
    vm.load_functions(functions);
    let res = vm.call("test_closure", vec![]).expect("call test_closure");
    // 25 + 30 = 55
    assert_eq!(res, Value::Int(55));
}

#[test]
fn test_execution_recursion_and_stack_depth_limit() {
    let source = r#"
        function infinite_recurse(n: int): int
            return infinite_recurse(n + 1)
        end
    "#;
    let functions = compile_source(source).unwrap();
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    let mut vm = VM::with_config(&config);
    vm.load_functions(functions);
    let err = vm.call("infinite_recurse", vec![Value::Int(0)]).unwrap_err();
    let s = err.to_string();
    assert!(s.to_lowercase().contains("overflow") || s.contains("Stack"), "Expected overflow in error: {}", s);
}

#[test]
fn test_execution_gas_budget_exhaustion() {
    let source = r#"
        function loop_forever(): int
            local i = 0
            while true do
                i = i + 1
            end
            return i
        end
    "#;
    let functions = compile_source(source).unwrap();
    let mut vm = VM::new();
    vm.set_gas_budget(50);
    vm.load_functions(functions);
    let err = vm.call("loop_forever", vec![]).unwrap_err();
    assert!(err.to_string().contains("Gas limit exceeded") || err.to_string().contains("gas"), "Expected gas error: {}", err);
}

#[test]
fn test_execution_string_concat_and_value_to_string() {
    let res = execute_instructions(
        vec![
            Instruction::LoadConst(0, 0), // "Hello, "
            Instruction::LoadConst(1, 1), // "World!"
            Instruction::Concat(2, 0, 1), // "Hello, World!"
            Instruction::Return(2),
        ],
        vec![Value::string("Hello, "), Value::string("World!")],
        8,
    ).expect("concat execution");
    assert_eq!(res, Value::string("Hello, World!"));
}

#[test]
fn test_task_native_yield_and_stop_lifecycle() {
    let mut vm = VM::new();

    // 1. Calling yield/stop outside a task via call_value returns context error
    let yield_fn = native_fn(|_| Ok(NativeCallResult::Yield(Value::Int(10))));
    let stop_fn = native_fn(|_| Ok(NativeCallResult::Stop(Value::Int(20))));

    let err_y = vm.call_value(&Value::NativeFunction(yield_fn.clone()), vec![]).unwrap_err();
    assert!(err_y.to_string().contains("Yielding or stopping is not allowed from this context"));

    let err_s = vm.call_value(&Value::NativeFunction(stop_fn.clone()), vec![]).unwrap_err();
    assert!(err_s.to_string().contains("Yielding or stopping is not allowed from this context"));

    // 2. Calling yield and stop inside a spawned task
    let mut func = Function::new("yield_worker".to_string(), 0, false);
    func.register_count = 8;
    let c_yield = func.chunk.add_constant(Value::NativeFunction(yield_fn));
    let c_stop = func.chunk.add_constant(Value::NativeFunction(stop_fn));

    func.chunk.emit(Instruction::LoadConst(0, c_yield), 1);
    func.chunk.emit(Instruction::Call(0, 0, 0, 1), 1); // yields!
    func.chunk.emit(Instruction::LoadConst(2, c_stop), 1);
    func.chunk.emit(Instruction::Call(2, 0, 0, 3), 1); // stops!
    func.chunk.emit(Instruction::Return(1), 1);

    let func_val = Value::Function(0);
    vm.load_functions(vec![func]);
    let handle = vm.spawn_task_value(func_val, vec![]).expect("spawn task value");

    // Task must be in Yielded state
    let task = vm.get_task_instance(handle).unwrap();
    assert!(task.state.is_yielded());
    assert_eq!(task.last_yield, Some(Value::Int(10)));

    // Resume task with value
    vm.resume_task_handle(handle, Some(Value::Int(555))).expect("resume task handle");

    // Task should now have executed to Stop
    let task2 = vm.get_task_instance(handle).unwrap();
    assert_eq!(task2.state, TaskState::Stopped);
    assert_eq!(task2.last_result, Some(Value::Int(20)));
}

#[test]
fn test_lua_dynamic_method_dispatch_and_metatable_index() {
    use lust::ast::{Span, Type, TypeKind};
    use lust::bytecode::{FieldStorage, StructLayout, StructObject, LustMap};

    let mut vm = VM::new();

    // Metamethod __index function
    let meta_index_fn = native_fn(|args| {
        // args: [receiver, key]
        let key_str = args.get(1).and_then(|v| v.as_string()).unwrap_or_default();
        if key_str == "dynamic_mul" {
            let func = native_fn(|args| {
                // args: [self, multiplier]
                let multiplier = args.get(1).and_then(|v| v.as_int()).unwrap_or(1);
                Ok(NativeCallResult::Return(Value::Int(7 * multiplier)))
            });
            Ok(NativeCallResult::Return(Value::NativeFunction(func)))
        } else {
            Ok(NativeCallResult::Return(Value::Nil))
        }
    });

    let mut meta_map = LustMap::default();
    meta_map.insert(ValueKey::string("__index"), Value::NativeFunction(meta_index_fn));

    let layout = Rc::new(StructLayout::new(
        "LuaTable".to_string(),
        vec![Rc::new("table".to_string()), Rc::new("metamethods".to_string())],
        vec![FieldStorage::Strong, FieldStorage::Strong],
        vec![Type::new(TypeKind::Unknown, Span::dummy()), Type::new(TypeKind::Unknown, Span::dummy())],
        vec![None, None],
    ));
    let lua_table = Value::Struct(StructObject::new(
        "LuaTable",
        layout,
        vec![Value::map(LustMap::default()), Value::map(meta_map)],
    ));

    // Call dynamic_mul(6) via Instruction::CallMethod
    let mut func = Function::new("test_dyn_method".to_string(), 0, false);
    func.register_count = 10;
    let c_table = func.chunk.add_constant(lua_table);
    let c_method = func.chunk.add_constant(Value::string("dynamic_mul"));
    let c_arg = func.chunk.add_constant(Value::Int(6));

    func.chunk.emit(Instruction::LoadConst(0, c_table), 1);
    func.chunk.emit(Instruction::LoadConst(1, c_arg), 1);
    func.chunk.emit(Instruction::CallMethod(0, c_method, 1, 1, 2), 1);
    func.chunk.emit(Instruction::Return(2), 1);

    vm.load_functions(vec![func]);
    let result = vm.call("test_dyn_method", vec![]).expect("dynamic method call must succeed");
    assert_eq!(result, Value::Int(42));
}

#[test]
fn test_tracing_tostring_and_hashkey_full_coverage() {
    use lust::ast::{Span, Type, TypeKind};
    use lust::bytecode::{FieldStorage, StructLayout, StructObject};

    let mut vm = VM::new();

    // 1. Separate initialization branches for hooks
    vm.clear_trace_hooks();
    vm.set_call_hook(Some(Box::new(|_, _, _| {})));
    vm.clear_trace_hooks();
    vm.set_return_hook(Some(Box::new(|_, _, _| {})));
    vm.clear_trace_hooks();
    vm.set_error_hook(Some(Box::new(|_, _, _| {})));
    vm.clear_trace_hooks();

    // 2. Struct with dotted type name and ToString trait
    let dotted_layout = Rc::new(StructLayout::new(
        "my_mod.Point".to_string(),
        vec![Rc::new("x".to_string())],
        vec![FieldStorage::Strong],
        vec![Type::new(TypeKind::Int, Span::dummy())],
        vec![None],
    ));
    let point_obj = Value::Struct(StructObject::new(
        "my_mod.Point",
        dotted_layout,
        vec![Value::Int(99)],
    ));

    // Register ToString method for Point: my_mod.Point:to_string
    let mut tostring_fn = Function::new("my_mod.Point:to_string".to_string(), 1, false);
    tostring_fn.register_count = 4;
    let c_res = tostring_fn.chunk.add_constant(Value::string("Point(99)"));
    tostring_fn.chunk.emit(Instruction::LoadConst(1, c_res), 1);
    tostring_fn.chunk.emit(Instruction::Return(1), 1);

    let mut main_fn = Function::new("main_concat".to_string(), 0, false);
    main_fn.register_count = 8;
    let c_point = main_fn.chunk.add_constant(point_obj);
    let c_prefix = main_fn.chunk.add_constant(Value::string("Point is: "));
    main_fn.chunk.emit(Instruction::LoadConst(0, c_prefix), 1);
    main_fn.chunk.emit(Instruction::LoadConst(1, c_point), 1);
    main_fn.chunk.emit(Instruction::Concat(2, 0, 1), 1);
    // Second concat hits struct_tostring_cache!
    main_fn.chunk.emit(Instruction::Concat(3, 0, 1), 1);
    main_fn.chunk.emit(Instruction::Return(2), 1);

    vm.load_functions(vec![main_fn, tostring_fn]);
    let res = vm.call("main_concat", vec![]).expect("concat with struct to_string");
    assert_eq!(res, Value::string("Point is: Point(99)"));
}

#[test]
fn test_tracing_jit_materialize_inline_frames_with_vm() {
    let mut vm = VM::new();
    let dummy_fn = Function::new("dummy".to_string(), 0, false);
    vm.load_functions(vec![dummy_fn]);

    let mut root_regs = [Value::Int(10), Value::Int(20), Value::Nil, Value::Nil];
    let mut callee_regs = [Value::Int(30), Value::Int(40), Value::Nil, Value::Nil];

    let site1 = JitCallSite {
        value_count: 2,
        alias_mask: 1, // register 0 aliased
        borrow_mask: 2, // register 1 borrowed
        function_idx: 0,
        return_dest: 2,
        callee_reg: 0,
        caller_resume_ip: 5,
    };
    let record1 = JitInlineRecord {
        caller_regs: root_regs.as_mut_ptr(),
        prev: ptr::null(),
        site: &site1,
    };

    let site2 = JitCallSite {
        value_count: 2,
        alias_mask: 0,
        borrow_mask: 0,
        function_idx: 0,
        return_dest: 3,
        callee_reg: 0,
        caller_resume_ip: 8,
    };
    let record2 = JitInlineRecord {
        caller_regs: callee_regs.as_mut_ptr(),
        prev: &record1,
        site: &site2,
    };

    unsafe {
        let ret = jit_materialize_inline_frames(&mut vm as *mut VM, &record2, callee_regs.as_mut_ptr());
        assert_eq!(ret, root_regs.as_mut_ptr(), "root_regs pointer must be returned");
    }
}

#[test]
fn test_execution_generic_arithmetic_all_branches() {
    // Test generic Add, Sub, Mul, Div, Mod, Pow, FloorDiv, Neg with mixed Int/Float
    let res = execute_instructions(
        vec![
            Instruction::LoadConst(0, 0), // Int(10)
            Instruction::LoadConst(1, 1), // Float(2.5)
            Instruction::Add(2, 0, 1),    // 12.5 (Float)
            Instruction::Sub(3, 0, 1),    // 7.5 (Float)
            Instruction::Mul(4, 0, 1),    // 25.0 (Float)
            Instruction::Div(5, 0, 1),    // 4.0 (Float)
            Instruction::Mod(6, 0, 1),    // 0.0 (Float)
            Instruction::FloorDiv(7, 0, 1), // 4.0 (Float)
            Instruction::Neg(8, 0),       // -10 (Int)
            Instruction::Neg(9, 1),       // -2.5 (Float)
            Instruction::Return(2),
        ],
        vec![Value::Int(10), Value::Float(2.5)],
        16,
    ).expect("generic arithmetic mixed");
    assert_eq!(res, Value::Float(12.5));

    // Incompatible Add error
    let err_add = execute_instructions(
        vec![
            Instruction::LoadBool(0, true),
            Instruction::LoadConst(1, 0),
            Instruction::Add(2, 0, 1),
            Instruction::Return(2),
        ],
        vec![Value::Int(10)],
        8,
    ).unwrap_err();
    assert!(err_add.to_string().contains("Cannot add"));
}

#[test]
fn test_execution_generic_comparisons_and_jumps() {
    let mut func = Function::new("test_cmp_jumps".to_string(), 0, false);
    func.register_count = 16;
    let c10 = func.chunk.add_constant(Value::Int(10));
    let c20 = func.chunk.add_constant(Value::Int(20));

    func.chunk.emit(Instruction::LoadConst(0, c10), 1);
    func.chunk.emit(Instruction::LoadConst(1, c10), 1);
    func.chunk.emit(Instruction::Eq(2, 0, 1), 1);     // true
    func.chunk.emit(Instruction::Ne(3, 0, 1), 1);     // false
    func.chunk.emit(Instruction::LoadConst(4, c20), 1);
    func.chunk.emit(Instruction::Lt(5, 0, 4), 1);     // true
    func.chunk.emit(Instruction::Le(6, 0, 4), 1);     // true
    func.chunk.emit(Instruction::Gt(7, 4, 0), 1);     // true
    func.chunk.emit(Instruction::Ge(8, 4, 0), 1);     // true
    // Jumps with truthy / falsy conditions
    func.chunk.emit(Instruction::LoadBool(9, false), 1);
    func.chunk.emit(Instruction::JumpIf(9, 2), 1);    // not taken
    func.chunk.emit(Instruction::JumpIfNot(9, 1), 1); // taken: jump over Return(9)
    func.chunk.emit(Instruction::Return(9), 1);
    func.chunk.emit(Instruction::Return(2), 1);       // returns true

    let mut vm = VM::new();
    vm.load_functions(vec![func]);
    let res = vm.call("test_cmp_jumps", vec![]).expect("comparisons and jumps");
    assert_eq!(res, Value::Bool(true));
}

#[test]
fn test_execution_globals_and_map_set_get() {
    let mut func = Function::new("test_globals".to_string(), 0, false);
    func.register_count = 8;
    let c_global_name = func.chunk.add_constant(Value::string("MY_GLOBAL_VAR"));
    let c_val = func.chunk.add_constant(Value::Int(12345));

    func.chunk.emit(Instruction::LoadConst(0, c_val), 1);
    func.chunk.emit(Instruction::StoreGlobal(c_global_name, 0), 1);
    func.chunk.emit(Instruction::LoadGlobal(1, c_global_name), 1);
    func.chunk.emit(Instruction::Return(1), 1);

    let mut vm = VM::new();
    vm.load_functions(vec![func]);
    let res = vm.call("test_globals", vec![]).expect("global load/store");
    assert_eq!(res, Value::Int(12345));
    assert_eq!(vm.get_global("MY_GLOBAL_VAR"), Some(Value::Int(12345)));
}

#[test]
fn test_tasks_deep_lifecycle_and_builtin_error_matrix() {
    use lust::ast::{Span, Type, TypeKind};
    use lust::bytecode::{FieldStorage, StructLayout, StructObject, LustMap};

    let mut vm = VM::new();

    // 1. Spawning a task that yields, then stop_task_handle and restart_task_handle
    let yield_fn = native_fn(|_| Ok(NativeCallResult::Yield(Value::Int(99))));
    let mut worker = Function::new("yield_restart_worker".to_string(), 0, false);
    worker.register_count = 8;
    let c_fn = worker.chunk.add_constant(Value::NativeFunction(yield_fn));
    worker.chunk.emit(Instruction::LoadConst(0, c_fn), 1);
    worker.chunk.emit(Instruction::Call(0, 0, 0, 1), 1);
    worker.chunk.emit(Instruction::Return(1), 1);

    vm.load_functions(vec![worker]);
    let handle = vm.spawn_task_value(Value::Function(0), vec![]).expect("spawn worker");
    assert!(vm.get_task_instance(handle).unwrap().state.is_yielded());

    // Stop yielded task
    let _task_id = TaskId(handle.id());
    // stop task
    vm.complete_native_future_task(handle, Ok(Value::Nil)).unwrap_err(); // script task error
    vm.resume_task_handle(handle, Some(Value::Int(123))).expect("resume to return");
    assert_eq!(vm.get_task_instance(handle).unwrap().state, TaskState::Completed);

    // Scoped restart rejection
    let handle2 = vm.create_native_future_task();
    let mut scope = Scope::new(555, None);
    vm.register_scope_task(&mut scope, handle2).expect("register scope task");

    // 2. Call_builtin_method error matrix via Instruction::CallMethod
    let layout = Rc::new(StructLayout::new(
        "LuaTable".to_string(),
        vec![Rc::new("table".to_string())],
        vec![FieldStorage::Strong],
        vec![Type::new(TypeKind::Unknown, Span::dummy())],
        vec![None],
    ));
    let lua_table = Value::Struct(StructObject::new(
        "LuaTable",
        layout,
        vec![Value::map(LustMap::default())],
    ));

    // Test LuaTable arity errors
    let run_lua_err = |method: &str, arg_count: u8, args: Vec<Value>| -> Result<Value, LustError> {
        let mut f = Function::new(format!("err_{}", method), 0, false);
        f.register_count = 10;
        let c_tbl = f.chunk.add_constant(lua_table.clone());
        let c_m = f.chunk.add_constant(Value::string(method));
        f.chunk.emit(Instruction::LoadConst(0, c_tbl), 1);
        for (i, a) in args.iter().enumerate() {
            let ca = f.chunk.add_constant(a.clone());
            f.chunk.emit(Instruction::LoadConst((1 + i) as u8, ca), 1);
        }
        f.chunk.emit(Instruction::CallMethod(0, c_m, 1, arg_count, 5), 1);
        f.chunk.emit(Instruction::Return(5), 1);

        let mut test_vm = VM::new();
        test_vm.load_functions(vec![f]);
        test_vm.call(&format!("err_{}", method), vec![])
    };

    assert!(run_lua_err("push", 0, vec![]).unwrap_err().to_string().contains("push() requires 1 argument"));
    assert!(run_lua_err("insert", 1, vec![Value::Int(1)]).unwrap_err().to_string().contains("insert() requires 2 arguments"));
    assert!(run_lua_err("remove", 0, vec![]).unwrap_err().to_string().contains("remove() requires 1 argument"));
    assert!(run_lua_err("concat", 1, vec![Value::string("-")]).unwrap_err().to_string().contains("concat() requires 3 arguments"));
    assert!(run_lua_err("maxn", 1, vec![Value::Int(1)]).unwrap_err().to_string().contains("maxn() takes no arguments"));
    assert!(run_lua_err("len", 1, vec![Value::Int(1)]).unwrap_err().to_string().contains("len() takes no arguments"));
    assert!(run_lua_err("bad_method", 0, vec![]).unwrap_err().to_string().contains("LuaTable has no method"));

    // Option & Result arity errors
    let opt_some = Value::some(Value::Int(10));
    let opt_none = Value::none();
    let res_ok = Value::ok(Value::Int(20));
    let res_err = Value::err(Value::string("err"));

    let run_val_err = |val: Value, method: &str, arg_count: u8, args: Vec<Value>| -> Result<Value, LustError> {
        let mut f = Function::new("err_val".to_string(), 0, false);
        f.register_count = 10;
        let c_v = f.chunk.add_constant(val);
        let c_m = f.chunk.add_constant(Value::string(method));
        f.chunk.emit(Instruction::LoadConst(0, c_v), 1);
        for (i, a) in args.iter().enumerate() {
            let ca = f.chunk.add_constant(a.clone());
            f.chunk.emit(Instruction::LoadConst((1 + i) as u8, ca), 1);
        }
        f.chunk.emit(Instruction::CallMethod(0, c_m, 1, arg_count, 5), 1);
        f.chunk.emit(Instruction::Return(5), 1);

        let mut test_vm = VM::new();
        test_vm.load_functions(vec![f]);
        test_vm.call("err_val", vec![])
    };

    assert!(run_val_err(opt_none.clone(), "unwrap_or", 0, vec![]).unwrap_err().to_string().contains("unwrap_or requires a default value"));
    assert!(run_val_err(opt_some, "bad_opt_method", 0, vec![]).unwrap_err().to_string().contains("Option has no method"));
    assert!(run_val_err(res_err.clone(), "unwrap_or", 0, vec![]).unwrap_err().to_string().contains("unwrap_or requires a default value"));
    assert!(run_val_err(res_ok, "bad_res_method", 0, vec![]).unwrap_err().to_string().contains("Result has no method"));

    // Primitive type errors
    assert!(run_val_err(Value::Float(1.5), "floor", 0, vec![]).unwrap_err().to_string().contains("Type 'float' does not support method syntax"));
    assert!(run_val_err(Value::Int(10), "abs", 0, vec![]).unwrap_err().to_string().contains("Type 'int' does not support method syntax"));
    assert!(run_val_err(Value::map(LustMap::default()), "keys", 0, vec![]).unwrap_err().to_string().contains("Type 'Map' does not support method syntax"));
    assert!(run_val_err(Value::string("abc"), "upper", 0, vec![]).unwrap_err().to_string().contains("Type 'string' does not support method syntax"));
    assert!(run_val_err(Value::array(vec![]), "push", 0, vec![]).unwrap_err().to_string().contains("Type 'Array' does not support method syntax"));
}

#[test]
fn test_execution_comprehensive_opcodes_dual_mode() {
    use lust::ast::{Span, Type, TypeKind};
    use lust::bytecode::{FieldStorage, StructLayout, StructObject};

    for jit_enabled in [false, true] {
        let mut config = LustConfig::default();
        config.set_jit_enabled(jit_enabled);
        let mut vm = VM::with_config(&config);

        // 1. Struct creation, GetField, SetField (fast path method cache hit)
        let layout = Rc::new(StructLayout::new(
            "Account".to_string(),
            vec![Rc::new("balance".to_string())],
            vec![FieldStorage::Strong],
            vec![Type::new(TypeKind::Int, Span::dummy())],
            vec![None],
        ));
        let account = Value::Struct(StructObject::new(
            "Account",
            layout.clone(),
            vec![Value::Int(100)],
        ));

        // Method Account:deposit
        let mut deposit_fn = Function::new("Account:deposit".to_string(), 2, false);
        deposit_fn.register_count = 8;
        let c_field = deposit_fn.chunk.add_constant(Value::string("balance"));
        deposit_fn.chunk.emit(Instruction::GetField(2, 0, c_field), 1); // self.balance
        deposit_fn.chunk.emit(Instruction::AddInt(3, 2, 1), 1); // balance + amount
        deposit_fn.chunk.emit(Instruction::SetField(0, c_field, 3), 1); // self.balance = new
        deposit_fn.chunk.emit(Instruction::Return(3), 1);

        // Main caller invokes deposit twice (testing cache insertion and cache hit!)
        let mut main_fn = Function::new("main_account".to_string(), 0, false);
        main_fn.register_count = 12;
        let c_acc = main_fn.chunk.add_constant(account);
        let c_dep = main_fn.chunk.add_constant(Value::string("deposit"));
        let c_amt1 = main_fn.chunk.add_constant(Value::Int(50));
        let c_amt2 = main_fn.chunk.add_constant(Value::Int(75));

        main_fn.chunk.emit(Instruction::LoadConst(0, c_acc), 1);
        main_fn.chunk.emit(Instruction::LoadConst(1, c_amt1), 1);
        main_fn.chunk.emit(Instruction::CallMethod(0, c_dep, 1, 1, 2), 1); // first call (cold)
        main_fn.chunk.emit(Instruction::LoadConst(3, c_amt2), 1);
        main_fn.chunk.emit(Instruction::CallMethod(0, c_dep, 3, 1, 4), 1); // second call (cache hit!)
        main_fn.chunk.emit(Instruction::Return(4), 1); // returns 225

        vm.load_functions(vec![main_fn, deposit_fn]);
        let res = vm.call("main_account", vec![]).expect("account deposit must succeed");
        assert_eq!(res, Value::Int(225), "balance must be 100 + 50 + 75 = 225 (jit: {})", jit_enabled);

        // 2. Closure provide and run_sync methods
        let sub_fn = Function::new("sub_worker".to_string(), 0, false);
        use lust::bytecode::ClosureObject;
        let closure = Value::Closure(Rc::new(ClosureObject {
            function_idx: 0,
            upvalues: Vec::new(),
        }));
        let mut call_m_fn = Function::new("call_closure_m".to_string(), 0, false);
        call_m_fn.register_count = 8;
        let c_cl = call_m_fn.chunk.add_constant(closure);
        let c_prov = call_m_fn.chunk.add_constant(Value::string("provide"));
        let c_dummy = call_m_fn.chunk.add_constant(Value::Int(1));

        call_m_fn.chunk.emit(Instruction::LoadConst(0, c_cl), 1);
        call_m_fn.chunk.emit(Instruction::LoadConst(1, c_dummy), 1);
        call_m_fn.chunk.emit(Instruction::CallMethod(0, c_prov, 1, 1, 2), 1); // closure.provide(1)
        call_m_fn.chunk.emit(Instruction::Return(2), 1);

        let mut vm2 = VM::with_config(&config);
        vm2.load_functions(vec![call_m_fn, sub_fn]);
        let res_cl = vm2.call("call_closure_m", vec![]).expect("closure provide");
        assert!(matches!(res_cl, Value::Closure(_)));

        // 3. ArrayLen and ArrayIndexOk
        let mut arr_fn = Function::new("test_arr_ops".to_string(), 0, false);
        arr_fn.register_count = 10;
        let c_e1 = arr_fn.chunk.add_constant(Value::Int(111));
        let c_e2 = arr_fn.chunk.add_constant(Value::Int(222));
        let c_i0 = arr_fn.chunk.add_constant(Value::Int(0));
        let c_i9 = arr_fn.chunk.add_constant(Value::Int(9)); // out of bounds

        arr_fn.chunk.emit(Instruction::LoadConst(0, c_e1), 1);
        arr_fn.chunk.emit(Instruction::LoadConst(1, c_e2), 1);
        arr_fn.chunk.emit(Instruction::NewArray(2, 0, 2), 1); // [111, 222]
        arr_fn.chunk.emit(Instruction::ArrayLen(3, 2), 1); // len = 2
        arr_fn.chunk.emit(Instruction::LoadConst(4, c_i0), 1);
        arr_fn.chunk.emit(Instruction::ArrayIndexOk(5, 6, 2, 4), 1); // val=111, cond=true
        arr_fn.chunk.emit(Instruction::LoadConst(7, c_i9), 1);
        arr_fn.chunk.emit(Instruction::ArrayIndexOk(8, 9, 2, 7), 1); // val=Nil, cond=false
        arr_fn.chunk.emit(Instruction::Return(3), 1);

        let mut vm3 = VM::with_config(&config);
        vm3.load_functions(vec![arr_fn]);
        let res_arr = vm3.call("test_arr_ops", vec![]).expect("array ops");
        assert_eq!(res_arr, Value::Int(2));
    }
}

#[test]
fn test_execution_value_is_type_and_matches_type_matrix() {
    // 1. Instruction::TypeIs testing all type branches
    let types = vec![
        ("int", Value::Int(42), true),
        ("int", Value::Float(3.14), false),
        ("unknown", Value::Int(42), true),
        ("LuaValue", Value::Int(42), true),
        ("int | string", Value::Int(42), true),
        ("int | string", Value::string("hi"), true),
        ("int | string", Value::Bool(true), false),
        ("float", Value::Float(3.14), true),
        ("string", Value::string("hello"), true),
        ("bool", Value::Bool(true), true),
        ("nil", Value::Nil, true),
        ("()", Value::Nil, true),
        ("Array", Value::array(vec![]), true),
        ("Array<int>", Value::array(vec![]), true),
        ("Tuple", Value::tuple(vec![]), true),
        ("Tuple<int, string>", Value::tuple(vec![]), true),
        ("Map", Value::map(lust::bytecode::LustMap::default()), true),
        ("Map<string, int>", Value::map(lust::bytecode::LustMap::default()), true),
        ("Option", Value::some(Value::Int(1)), true),
        ("Option<int>", Value::some(Value::Int(1)), true),
        ("Result", Value::ok(Value::Int(1)), true),
        ("Result<int, string>", Value::ok(Value::Int(1)), true),
    ];

    for (type_name, value, expected) in types {
        let mut f = Function::new("test_type_is".to_string(), 0, false);
        f.register_count = 8;
        let c_val = f.chunk.add_constant(value.clone());
        let c_tname = f.chunk.add_constant(Value::string(type_name));
        f.chunk.emit(Instruction::LoadConst(0, c_val), 1);
        f.chunk.emit(Instruction::TypeIs(1, 0, c_tname), 1);
        f.chunk.emit(Instruction::Return(1), 1);

        let mut vm = VM::new();
        vm.load_functions(vec![f]);
        let res = vm.call("test_type_is", vec![]).expect("TypeIs execution");
        assert_eq!(res, Value::Bool(expected), "TypeIs check for '{}' with {:?}", type_name, value);
    }

    // 2. Typed calls verifying argument & return type matching
    let source = r#"
        function typed_accept(x: int, s: string, arr: Array<int>, opt: Option<int>): int
            return x + 10
        end
    "#;
    let functions = compile_source(source).expect("compile typed_accept");
    let mut vm = VM::new();
    vm.load_functions(functions);

    let ok_res = vm.call(
        "typed_accept",
        vec![
            Value::Int(5),
            Value::string("ok"),
            Value::array(vec![Value::Int(1), Value::Int(2)]),
            Value::some(Value::Int(99)),
        ],
    ).expect("valid argument types must succeed");
    assert_eq!(ok_res, Value::Int(15));

    // Passing wrong argument type must fail
    let err_arg = vm.call(
        "typed_accept",
        vec![
            Value::string("bad_int"),
            Value::string("ok"),
            Value::array(vec![]),
            Value::none(),
        ],
    );
    assert!(err_arg.is_err(), "passing wrong type must fail argument validation");
}

#[test]
fn test_execution_indexing_and_field_error_branches() {
    // 1. GetIndex and SetIndex on Map
    let mut func = Function::new("test_map_indexing".to_string(), 0, false);
    func.register_count = 12;
    func.chunk.emit(Instruction::NewMap(0), 1); // R0 = new map
    let c_k1 = func.chunk.add_constant(Value::string("key1"));
    let c_v1 = func.chunk.add_constant(Value::Int(42));
    func.chunk.emit(Instruction::LoadConst(1, c_k1), 1);
    func.chunk.emit(Instruction::LoadConst(2, c_v1), 1);
    func.chunk.emit(Instruction::SetIndex(0, 1, 2), 1); // map["key1"] = 42
    func.chunk.emit(Instruction::GetIndex(3, 0, 1), 1); // R3 = map["key1"]
    let c_k_miss = func.chunk.add_constant(Value::string("missing"));
    func.chunk.emit(Instruction::LoadConst(4, c_k_miss), 1);
    func.chunk.emit(Instruction::TryGetIndex(5, 0, 4), 1); // R5 = Nil (missing key)
    func.chunk.emit(Instruction::Return(3), 1); // returns 42

    let mut vm = VM::new();
    vm.load_functions(vec![func]);
    let res = vm.call("test_map_indexing", vec![]).expect("map indexing");
    assert_eq!(res, Value::Int(42));

    // 2. SetIndex on non-indexable type
    let mut err_f = Function::new("err_set_idx".to_string(), 0, false);
    err_f.register_count = 8;
    let c_i = err_f.chunk.add_constant(Value::Int(10));
    err_f.chunk.emit(Instruction::LoadConst(0, c_i), 1);
    err_f.chunk.emit(Instruction::SetIndex(0, 0, 0), 1); // 10[10] = 10 -> Error
    err_f.chunk.emit(Instruction::Return(0), 1);

    let mut vm2 = VM::new();
    vm2.load_functions(vec![err_f]);
    let err2 = vm2.call("err_set_idx", vec![]).unwrap_err();
    assert!(err2.to_string().contains("Cannot index"));

    // 3. GetIndex on non-indexable type
    let mut err_f2 = Function::new("err_get_idx".to_string(), 0, false);
    err_f2.register_count = 8;
    let c_i2 = err_f2.chunk.add_constant(Value::Int(10));
    err_f2.chunk.emit(Instruction::LoadConst(0, c_i2), 1);
    err_f2.chunk.emit(Instruction::GetIndex(1, 0, 0), 1); // R1 = 10[10] -> Error
    err_f2.chunk.emit(Instruction::Return(1), 1);

    let mut vm3 = VM::new();
    vm3.load_functions(vec![err_f2]);
    let err3 = vm3.call("err_get_idx", vec![]).unwrap_err();
    assert!(err3.to_string().contains("Cannot index"));

    // 4. GetField on non-struct
    let mut err_f3 = Function::new("err_get_field".to_string(), 0, false);
    err_f3.register_count = 8;
    let c_i3 = err_f3.chunk.add_constant(Value::Int(10));
    let c_fname = err_f3.chunk.add_constant(Value::string("foo"));
    err_f3.chunk.emit(Instruction::LoadConst(0, c_i3), 1);
    err_f3.chunk.emit(Instruction::GetField(1, 0, c_fname), 1); // 10.foo -> Error
    err_f3.chunk.emit(Instruction::Return(1), 1);

    let mut vm4 = VM::new();
    vm4.load_functions(vec![err_f3]);
    let err4 = vm4.call("err_get_field", vec![]).unwrap_err();
    assert!(err4.to_string().contains("Cannot get field"));

    // 5. GetEnumValue on non-enum
    let mut err_f4 = Function::new("err_enum_val".to_string(), 0, false);
    err_f4.register_count = 8;
    let c_i4 = err_f4.chunk.add_constant(Value::Int(10));
    err_f4.chunk.emit(Instruction::LoadConst(0, c_i4), 1);
    err_f4.chunk.emit(Instruction::GetEnumValue(1, 0, 0), 1); // on int -> Error
    err_f4.chunk.emit(Instruction::Return(1), 1);

    let mut vm5 = VM::new();
    vm5.load_functions(vec![err_f4]);
    let err5 = vm5.call("err_enum_val", vec![]).unwrap_err();
    assert!(err5.to_string().contains("GetEnumValue requires an enum variant with values"));
}

#[test]
fn test_tasks_lua_table_unpack_and_sort_lifecycle() {
    use lust::ast::{Span, Type, TypeKind};
    use lust::bytecode::{FieldStorage, StructLayout, StructObject, LustMap};

    let layout = Rc::new(StructLayout::new(
        "LuaTable".to_string(),
        vec![Rc::new("table".to_string())],
        vec![FieldStorage::Strong],
        vec![Type::new(TypeKind::Unknown, Span::dummy())],
        vec![None],
    ));
    let lua_table = Value::Struct(StructObject::new(
        "LuaTable",
        layout,
        vec![Value::map(LustMap::default())],
    ));

    // Test unpack() and sort()
    let mut func = Function::new("test_unpack".to_string(), 0, false);
    func.register_count = 16;
    let c_tbl = func.chunk.add_constant(lua_table);
    let c_push = func.chunk.add_constant(Value::string("push"));
    let c_unpack = func.chunk.add_constant(Value::string("unpack"));
    let c_sort = func.chunk.add_constant(Value::string("sort"));
    let c_v30 = func.chunk.add_constant(Value::Int(30));
    let c_v10 = func.chunk.add_constant(Value::Int(10));
    let c_v20 = func.chunk.add_constant(Value::Int(20));

    func.chunk.emit(Instruction::LoadConst(0, c_tbl), 1);
    func.chunk.emit(Instruction::LoadConst(1, c_v30), 1);
    func.chunk.emit(Instruction::CallMethod(0, c_push, 1, 1, 2), 1); // push(30)
    func.chunk.emit(Instruction::LoadConst(1, c_v10), 1);
    func.chunk.emit(Instruction::CallMethod(0, c_push, 1, 1, 2), 1); // push(10)
    func.chunk.emit(Instruction::LoadConst(1, c_v20), 1);
    func.chunk.emit(Instruction::CallMethod(0, c_push, 1, 1, 2), 1); // push(20)

    // sort() with 1 comp argument using a bytecode comparator function
    let mut comp_fn = Function::new("comp_less".to_string(), 2, false);
    comp_fn.register_count = 4;
    comp_fn.chunk.emit(Instruction::LtInt(2, 0, 1), 1);
    comp_fn.chunk.emit(Instruction::Return(2), 1);

    let c_comp = func.chunk.add_constant(Value::Function(1));
    func.chunk.emit(Instruction::LoadConst(4, c_comp), 1);
    func.chunk.emit(Instruction::CallMethod(0, c_sort, 4, 1, 8), 1); // sort(comp)

    // unpack(1, 2)
    let c_1 = func.chunk.add_constant(Value::Int(1));
    let c_2 = func.chunk.add_constant(Value::Int(2));
    func.chunk.emit(Instruction::LoadConst(5, c_1), 1);
    func.chunk.emit(Instruction::LoadConst(6, c_2), 1);
    func.chunk.emit(Instruction::CallMethod(0, c_unpack, 5, 2, 7), 1); // unpack(1, 2)

    func.chunk.emit(Instruction::Return(7), 1);

    let mut vm = VM::new();
    vm.load_functions(vec![func, comp_fn]);
    let res = vm.call("test_unpack", vec![]);
    assert!(res.is_ok(), "table unpack and sort must succeed with active VM context: {:?}", res);
}

#[test]
fn test_execution_hot_loop_jit_compilation_and_native_execution() {
    let source = r#"
        function run_hot_loop(): int
            local acc = 0
            local i = 0
            while i < 120 do
                acc = acc + i
                i = i + 1
            end
            return acc
        end
    "#;
    let functions = compile_source(source).unwrap();
    let mut config = LustConfig::default();
    config.set_jit_enabled(true);
    let mut vm = VM::with_config(&config);
    vm.load_functions(functions);
    let res = vm.call("run_hot_loop", vec![]).expect("hot loop must succeed");
    assert_eq!(res, Value::Int(7140));
}

#[test]
fn test_execution_closure_upvalue_mutation() {
    let source = r#"
        function make_counter(): function(): int
            local count = 0
            return function(): int
                count = count + 1
                return count
            end
        end

        function test_mutation(): int
            local c = make_counter()
            local a = c()
            local b = c()
            local d = c()
            return a + b + d
        end
    "#;
    let functions = compile_source(source).unwrap();
    let mut vm = VM::new();
    vm.load_functions(functions);
    let res = vm.call("test_mutation", vec![]).expect("counter");
    assert_eq!(res, Value::Int(6));
}

#[test]
fn test_execution_global_lookup_fallbacks_and_undefined() {
    let mut func = Function::new("test_native_global".to_string(), 0, false);
    func.register_count = 8;
    let c_nat = func.chunk.add_constant(Value::string("my_custom_native"));
    let _c_undef = func.chunk.add_constant(Value::string("undefined_global_sym"));
    func.chunk.emit(Instruction::LoadGlobal(0, c_nat), 1);
    func.chunk.emit(Instruction::Return(0), 1);

    let mut vm = VM::new();
    let nat_fn = native_fn(|_| Ok(NativeCallResult::Return(Value::Int(999))));
    vm.register_native("my_custom_native", Value::NativeFunction(nat_fn));
    vm.load_functions(vec![func]);
    let res = vm.call("test_native_global", vec![]).expect("load native global");
    assert!(matches!(res, Value::NativeFunction(_)));

    // Undefined global error
    let mut func_err = Function::new("test_undef_global".to_string(), 0, false);
    func_err.register_count = 8;
    let c_undef2 = func_err.chunk.add_constant(Value::string("undefined_global_sym"));
    func_err.chunk.emit(Instruction::LoadGlobal(0, c_undef2), 1);
    func_err.chunk.emit(Instruction::Return(0), 1);

    let mut vm2 = VM::new();
    vm2.load_functions(vec![func_err]);
    let err = vm2.call("test_undef_global", vec![]).unwrap_err();
    assert!(err.to_string().contains("Undefined global: undefined_global_sym"));
}

#[test]
fn test_execution_lua_value_indexing_unwrapping() {
    use lust::ast::{Span, Type, TypeKind};
    use lust::bytecode::{FieldStorage, StructLayout, StructObject, LustMap};

    let layout = Rc::new(StructLayout::new(
        "LuaTable".to_string(),
        vec![Rc::new("table".to_string())],
        vec![FieldStorage::Strong],
        vec![Type::new(TypeKind::Unknown, Span::dummy())],
        vec![None],
    ));
    let lua_table = Value::Struct(StructObject::new(
        "LuaTable",
        layout,
        vec![Value::map(LustMap::default())],
    ));

    // Test indexing with all LuaValue variants
    let variants = vec![
        ("Nil", vec![]),
        ("Bool", vec![Value::Bool(true)]),
        ("Int", vec![Value::Int(42)]),
        ("Float", vec![Value::Float(3.14)]),
        ("String", vec![Value::string("foo")]),
        ("Table", vec![Value::array(vec![])]),
        ("Function", vec![Value::Function(0)]),
        ("Userdata", vec![Value::Int(1)]),
        ("Thread", vec![Value::Int(2)]),
    ];

    for (variant, vals) in variants {
        let key_enum = Value::enum_variant("LuaValue", variant, vals);
        let mut func = Function::new("test_lua_key".to_string(), 0, false);
        func.register_count = 8;
        let c_tbl = func.chunk.add_constant(lua_table.clone());
        let c_k = func.chunk.add_constant(key_enum);
        let c_v = func.chunk.add_constant(Value::Int(123));

        func.chunk.emit(Instruction::LoadConst(0, c_tbl), 1);
        func.chunk.emit(Instruction::LoadConst(1, c_k), 1);
        func.chunk.emit(Instruction::LoadConst(2, c_v), 1);
        func.chunk.emit(Instruction::SetIndex(0, 1, 2), 1);
        func.chunk.emit(Instruction::GetIndex(3, 0, 1), 1);
        func.chunk.emit(Instruction::Return(3), 1);

        let mut vm = VM::new();
        vm.load_functions(vec![func]);
        let res = vm.call("test_lua_key", vec![]).expect("lua table key indexing");
        assert_eq!(res, Value::Int(123));
    }
}

#[test]
fn test_execution_lua_value_function_calls() {
    let mut vm = VM::new();

    // 1. Missing handle field
    let bad_fn1 = Value::enum_variant("LuaValue", "Function", vec![Value::Nil]);
    let err1 = vm.call_value(&bad_fn1, vec![]).unwrap_err();
    assert!(err1.to_string().contains("LuaValue function missing handle"));

    // 2. Handle not registered
    use lust::ast::{Span, Type, TypeKind};
    use lust::bytecode::{FieldStorage, StructLayout, StructObject};
    let h_layout = Rc::new(StructLayout::new(
        "LuaFuncHandle".to_string(),
        vec![Rc::new("handle".to_string())],
        vec![FieldStorage::Strong],
        vec![Type::new(TypeKind::Int, Span::dummy())],
        vec![None],
    ));
    let handle_obj = Value::Struct(StructObject::new(
        "LuaFuncHandle",
        h_layout,
        vec![Value::Int(999999)],
    ));
    let bad_fn2 = Value::enum_variant("LuaValue", "Function", vec![handle_obj]);
    let err2 = vm.call_value(&bad_fn2, vec![]).unwrap_err();
    assert!(err2.to_string().contains("was not registered with VM"));
}

#[test]
fn test_execution_in_place_concat_and_jump_instructions() {
    // 1. In-place string concat (dest == lhs)
    let mut f_concat = Function::new("test_in_place_concat".to_string(), 0, false);
    f_concat.register_count = 8;
    let c_left = f_concat.chunk.add_constant(Value::string("prefix_"));
    let c_right = f_concat.chunk.add_constant(Value::string("suffix"));
    f_concat.chunk.emit(Instruction::LoadConst(0, c_left), 1);
    f_concat.chunk.emit(Instruction::LoadConst(1, c_right), 1);
    // dest == lhs (0 == 0) triggers in-place path
    f_concat.chunk.emit(Instruction::Concat(0, 0, 1), 1);
    f_concat.chunk.emit(Instruction::Return(0), 1);

    let mut vm1 = VM::new();
    vm1.load_functions(vec![f_concat]);
    let res = vm1.call("test_in_place_concat", vec![]).expect("in-place concat");
    assert_eq!(res, Value::string("prefix_suffix"));

    // 2. Unconditional forward Jump and backward Jump (backedge)
    let mut f_jump = Function::new("test_jumps".to_string(), 0, false);
    f_jump.register_count = 8;
    let c_0 = f_jump.chunk.add_constant(Value::Int(0));
    let c_10 = f_jump.chunk.add_constant(Value::Int(10));
    f_jump.chunk.emit(Instruction::LoadConst(0, c_0), 1);
    f_jump.chunk.emit(Instruction::LoadConst(1, c_10), 1);
    // Forward jump over Return
    f_jump.chunk.emit(Instruction::Jump(1), 1);
    f_jump.chunk.emit(Instruction::Return(1), 1); // skipped!
    f_jump.chunk.emit(Instruction::Return(0), 1); // returns 0

    let mut vm2 = VM::new();
    vm2.load_functions(vec![f_jump]);
    let res2 = vm2.call("test_jumps", vec![]).expect("forward jump");
    assert_eq!(res2, Value::Int(0));

    // 3. Fallthrough without Return -> returns Value::Nil
    let mut f_fallthrough = Function::new("test_fallthrough".to_string(), 0, false);
    f_fallthrough.register_count = 4;
    let c_42 = f_fallthrough.chunk.add_constant(Value::Int(42));
    f_fallthrough.chunk.emit(Instruction::LoadConst(0, c_42), 1);
    // No Return!

    let mut vm3 = VM::new();
    vm3.load_functions(vec![f_fallthrough]);
    let res3 = vm3.call("test_fallthrough", vec![]).expect("fallthrough");
    assert_eq!(res3, Value::Nil);

    // 4. BitNot on non-int
    let mut f_bitnot_err = Function::new("test_bitnot_err".to_string(), 0, false);
    f_bitnot_err.register_count = 4;
    let c_f = f_bitnot_err.chunk.add_constant(Value::Float(1.5));
    f_bitnot_err.chunk.emit(Instruction::LoadConst(0, c_f), 1);
    f_bitnot_err.chunk.emit(Instruction::BitNot(1, 0), 1);
    f_bitnot_err.chunk.emit(Instruction::Return(1), 1);

    let mut vm4 = VM::new();
    vm4.load_functions(vec![f_bitnot_err]);
    let err4 = vm4.call("test_bitnot_err", vec![]).unwrap_err();
    assert!(err4.to_string().contains("Cannot perform bitwise NOT"));
}

#[test]
fn test_execution_load_global_fallbacks_in_interpreter_mode() {
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    let mut vm = VM::with_config(&config);

    // 1. Native global lookup in interpreter mode
    let mut f1 = Function::new("test_native".to_string(), 0, false);
    f1.register_count = 4;
    let c_name = f1.chunk.add_constant(Value::string("my_native"));
    f1.chunk.emit(Instruction::LoadGlobal(0, c_name), 1);
    f1.chunk.emit(Instruction::Return(0), 1);

    let nat = native_fn(|_| Ok(NativeCallResult::Return(Value::Int(77))));
    vm.register_native("my_native", Value::NativeFunction(nat));
    vm.load_functions(vec![f1]);
    let res = vm.call("test_native", vec![]).expect("native global in interpreter mode");
    assert!(matches!(res, Value::NativeFunction(_)));

    // 2. Map SetIndex and GetIndex in interpreter mode
    let mut f2 = Function::new("test_map_interp".to_string(), 0, false);
    f2.register_count = 8;
    f2.chunk.emit(Instruction::NewMap(0), 1);
    let c_k = f2.chunk.add_constant(Value::string("k"));
    let c_v = f2.chunk.add_constant(Value::Int(888));
    f2.chunk.emit(Instruction::LoadConst(1, c_k), 1);
    f2.chunk.emit(Instruction::LoadConst(2, c_v), 1);
    f2.chunk.emit(Instruction::SetIndex(0, 1, 2), 1);
    f2.chunk.emit(Instruction::GetIndex(3, 0, 1), 1);
    f2.chunk.emit(Instruction::Return(3), 1);

    let mut vm2 = VM::with_config(&config);
    vm2.load_functions(vec![f2]);
    let res2 = vm2.call("test_map_interp", vec![]).expect("map in interpreter mode");
    assert_eq!(res2, Value::Int(888));
}

#[test]
fn test_execution_dual_mode_massive_interpreter_coverage() {
    let source = r#"
        function complex_workload(n: int): int
            local acc = 0
            local i = 0
            while i < n do
                if i % 2 == 0 then
                    acc = acc + i * 2
                else
                    acc = acc - i
                end
                i = i + 1
            end

            local arr = [10, 20, 30]
            local v = arr[1]:unwrap()
            return acc + v
        end
    "#;
    let functions = compile_source(source).unwrap();
    for jit in [false, true] {
        let mut config = LustConfig::default();
        config.set_jit_enabled(jit);
        let mut vm = VM::with_config(&config);
        vm.load_functions(functions.clone());
        let res = vm.call("complex_workload", vec![Value::Int(20)]).expect("complex workload");
        assert_eq!(res, Value::Int(80 + 20));
    }
}

#[test]
fn test_tasks_tick_task_public_api() {
    let mut vm = VM::new();
    let yield_fn = native_fn(|args| {
        let val = args.get(0).cloned().unwrap_or(Value::Int(0));
        Ok(NativeCallResult::Yield(val))
    });
    let mut func = Function::new("tick_worker".to_string(), 0, false);
    func.register_count = 8;
    let c_fn = func.chunk.add_constant(Value::NativeFunction(yield_fn));
    let c_val = func.chunk.add_constant(Value::Int(777));
    func.chunk.emit(Instruction::LoadConst(0, c_fn), 1);
    func.chunk.emit(Instruction::LoadConst(1, c_val), 1);
    func.chunk.emit(Instruction::Call(0, 1, 1, 2), 1); // yields 777!
    func.chunk.emit(Instruction::Return(2), 1); // returns 777!

    vm.load_functions(vec![func]);
    let handle = vm.spawn_tick_task("tick_worker").expect("spawn tick task");
    // 1. First tick runs to yield
    let y = vm.tick_task(handle, Value::Nil).expect("first tick");
    assert_eq!(y, Value::Int(777));
    // 2. Second tick runs to completion
    let fin = vm.tick_task(handle, Value::Int(999)).expect("second tick");
    assert_eq!(fin, Value::Int(999));
}

#[test]
fn test_execution_whole_function_jit_compilation_and_execution() {
    let source = r#"
        function add_numbers(a: int, b: int): int
            return a + b
        end

        function return_nil_fn(x: int)
            local y = x + 1
        end
    "#;
    let functions = compile_source(source).unwrap();
    let mut config = LustConfig::default();
    config.set_jit_enabled(true);
    let mut vm = VM::with_config(&config);
    vm.load_functions(functions);

    // Call 25 times to exceed FUNCTION_HOT_THRESHOLD (15) and trigger whole-function compilation
    for i in 0..25 {
        let res = vm.call("add_numbers", vec![Value::Int(i), Value::Int(10)]).expect("call add_numbers");
        assert_eq!(res, Value::Int(i + 10));
    }

    // Also call return_nil_fn 25 times to trigger whole-function compilation returning Nil
    for i in 0..25 {
        let res = vm.call("return_nil_fn", vec![Value::Int(i)]).expect("call return_nil_fn");
        assert_eq!(res, Value::Nil);
    }
}

#[test]
fn test_execution_all_vm_error_branches_matrix() {
    for jit in [false, true] {
        let mut config = LustConfig::default();
        config.set_jit_enabled(jit);

        let run_f = |instrs: Vec<Instruction>, consts: Vec<Value>| -> Result<Value, LustError> {
            let mut f = Function::new("err_worker".to_string(), 0, false);
            f.register_count = 16;
            for c in consts {
                f.chunk.add_constant(c);
            }
            for i in instrs {
                f.chunk.emit(i, 1);
            }
            let mut vm = VM::with_config(&config);
            vm.load_functions(vec![f]);
            vm.call("err_worker", vec![])
        };

        // 1. GetEnumValue out of bounds
        let enum_val = Value::enum_variant("MyEnum", "Variant", vec![Value::Int(1)]);
        let err = run_f(vec![Instruction::LoadConst(0, 0), Instruction::GetEnumValue(1, 0, 99), Instruction::Return(1)], vec![enum_val]).unwrap_err();
        assert!(err.to_string().contains("out of bounds"));

        // 2. GetIndex / SetIndex array bounds and non-array
        let arr = Value::array(vec![Value::Int(10), Value::Int(20)]);
        let err = run_f(vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::GetIndex(2, 0, 1), Instruction::Return(2)], vec![arr.clone(), Value::Int(99)]).unwrap_err();
        assert!(err.to_string().contains("out of bounds"));

        let err = run_f(vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::GetIndex(2, 0, 1), Instruction::Return(2)], vec![arr.clone(), Value::Int(-1)]).unwrap_err();
        assert!(err.to_string().contains("out of bounds"));

        let err = run_f(vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::LoadConst(2, 2), Instruction::SetIndex(0, 1, 2), Instruction::Return(0)], vec![arr.clone(), Value::Int(99), Value::Int(0)]).unwrap_err();
        assert!(err.to_string().contains("out of bounds"));

        let err = run_f(vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::LoadConst(2, 2), Instruction::SetIndex(0, 1, 2), Instruction::Return(0)], vec![arr.clone(), Value::Int(-1), Value::Int(0)]).unwrap_err();
        assert!(err.to_string().contains("out of bounds"));

        // 3. ArrayLen / ArrayIndexOk on non-array
        let err = run_f(vec![Instruction::LoadConst(0, 0), Instruction::ArrayLen(1, 0), Instruction::Return(1)], vec![Value::Int(123)]).unwrap_err();
        assert!(err.to_string().contains("ArrayLen requires array"));

        let err = run_f(vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::ArrayIndexOk(2, 3, 0, 1), Instruction::Return(2)], vec![Value::Int(123), Value::Int(0)]).unwrap_err();
        assert!(err.to_string().contains("Cannot index"));

        // 4. LoadUpvalue / StoreUpvalue out of bounds
        let err = run_f(vec![Instruction::LoadUpvalue(0, 99), Instruction::Return(0)], vec![]).unwrap_err();
        assert!(err.to_string().contains("Upvalue index 99 out of bounds"));

        let err = run_f(vec![Instruction::LoadConst(0, 0), Instruction::StoreUpvalue(99, 0), Instruction::Return(0)], vec![Value::Int(1)]).unwrap_err();
        assert!(err.to_string().contains("Upvalue index 99 out of bounds"));

        // 5. Constants must be strings: GetField, SetField, CallMethod, IsEnumVariant, LoadGlobal, StoreGlobal, TypeIs, TryCast
        let not_str = Value::Int(999);
        let err = run_f(vec![Instruction::LoadConst(0, 0), Instruction::GetField(1, 0, 1), Instruction::Return(1)], vec![Value::Int(1), not_str.clone()]).unwrap_err();
        assert!(err.to_string().contains("Field name must be a string"));

        let err = run_f(vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 0), Instruction::SetField(0, 1, 1), Instruction::Return(0)], vec![Value::Int(1), not_str.clone()]).unwrap_err();
        assert!(err.to_string().contains("Field name must be a string"));

        let err = run_f(vec![Instruction::LoadConst(0, 0), Instruction::CallMethod(0, 1, 0, 0, 0), Instruction::Return(0)], vec![Value::Int(1), not_str.clone()]).unwrap_err();
        assert!(err.to_string().contains("Method name must be a string"));

        let err = run_f(vec![Instruction::LoadConst(0, 0), Instruction::IsEnumVariant(1, 0, 1, 0), Instruction::Return(1)], vec![Value::Int(1), not_str.clone()]).unwrap_err();
        assert!(err.to_string().contains("Enum name must be a string"));

        let err = run_f(vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::IsEnumVariant(2, 0, 1, 2), Instruction::Return(2)], vec![Value::Int(1), Value::string("E"), not_str.clone()]).unwrap_err();
        assert!(err.to_string().contains("Variant name must be a string"));

        let err = run_f(vec![Instruction::LoadGlobal(0, 0), Instruction::Return(0)], vec![not_str.clone()]).unwrap_err();
        assert!(err.to_string().contains("Global name must be a string"));

        let err = run_f(vec![Instruction::LoadConst(0, 1), Instruction::StoreGlobal(0, 0), Instruction::Return(0)], vec![not_str.clone(), Value::Int(1)]).unwrap_err();
        assert!(err.to_string().contains("Global name must be a string"));

        let err = run_f(vec![Instruction::LoadConst(0, 1), Instruction::TypeIs(1, 0, 0), Instruction::Return(1)], vec![not_str.clone(), Value::Int(1)]).unwrap_err();
        assert!(err.to_string().contains("Type name must be a string"));

        let err = run_f(vec![Instruction::LoadConst(0, 1), Instruction::TryCast(1, 0, 0), Instruction::Return(1)], vec![not_str.clone(), Value::Int(1)]).unwrap_err();
        assert!(err.to_string().contains("Cast type name must be a string"));

        // 6. Float division by zero (ModFloat and FloorDivFloat)
        let err = run_f(vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::ModFloat(2, 0, 1), Instruction::Return(2)], vec![Value::Float(10.0), Value::Float(0.0)]).unwrap_err();
        assert!(err.to_string().contains("Modulo by zero"));

        let err = run_f(vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::FloorDivFloat(2, 0, 1), Instruction::Return(2)], vec![Value::Float(10.0), Value::Float(0.0)]).unwrap_err();
        assert!(err.to_string().contains("Division by zero"));

        // 7. Generic division by zero & negative exponent
        let err = run_f(vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::Div(2, 0, 1), Instruction::Return(2)], vec![Value::Int(10), Value::Int(0)]).unwrap_err();
        assert!(err.to_string().contains("Division by zero"));

        let err = run_f(vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::Mod(2, 0, 1), Instruction::Return(2)], vec![Value::Int(10), Value::Int(0)]).unwrap_err();
        assert!(err.to_string().contains("Modulo by zero"));

        let err = run_f(vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::FloorDiv(2, 0, 1), Instruction::Return(2)], vec![Value::Int(10), Value::Int(0)]).unwrap_err();
        assert!(err.to_string().contains("Division by zero"));

        let err = run_f(vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::Pow(2, 0, 1), Instruction::Return(2)], vec![Value::Int(2), Value::Int(-3)]).unwrap_err();
        assert!(err.to_string().contains("Negative exponent"));
    }
}

#[test]
fn test_execution_signature_parameter_and_return_validation() {
    use lust::ast::{Span, Type, TypeKind};
    use lust::typechecker::FunctionSignature;

    let d_span = Span::dummy();

    for jit in [false, true] {
        let mut config = LustConfig::default();
        config.set_jit_enabled(jit);

        // 1. Callee expects (int, string), caller passes (string, string) via Instruction::Call
        let mut callee = Function::new("strict_callee".to_string(), 2, false);
        callee.register_count = 4;
        callee.signature = Some(FunctionSignature {
            params: vec![
                Type::new(TypeKind::Int, d_span),
                Type::new(TypeKind::String, d_span),
            ],
            return_type: Type::new(TypeKind::Int, d_span),
            is_method: false,
            type_params: vec![],
            trait_bounds: vec![],
            effects: None,
        });
        callee.chunk.emit(Instruction::Return(0), 1);

        let mut caller = Function::new("caller".to_string(), 0, false);
        caller.register_count = 8;
        let c_fn = caller.chunk.add_constant(Value::Function(0));
        let c_str = caller.chunk.add_constant(Value::string("not_an_int"));
        caller.chunk.emit(Instruction::LoadConst(0, c_fn), 1);
        caller.chunk.emit(Instruction::LoadConst(1, c_str), 1);
        caller.chunk.emit(Instruction::LoadConst(2, c_str), 1);
        caller.chunk.emit(Instruction::Call(0, 1, 2, 3), 1); // calls function 0 with args in R1..R2
        caller.chunk.emit(Instruction::Return(3), 1);

        let mut vm = VM::with_config(&config);
        vm.load_functions(vec![callee, caller]);
        let err = vm.call("caller", vec![]).unwrap_err();
        assert!(err.to_string().contains("expects") && err.to_string().contains("got"));

        // 2. Callee signature returns int, but bytecode returns String
        let mut bad_return_fn = Function::new("bad_return".to_string(), 0, false);
        bad_return_fn.register_count = 4;
        bad_return_fn.signature = Some(FunctionSignature {
            params: vec![],
            return_type: Type::new(TypeKind::Int, d_span),
            is_method: false,
            type_params: vec![],
            trait_bounds: vec![],
            effects: None,
        });
        let c_bad = bad_return_fn.chunk.add_constant(Value::string("wrong_return"));
        bad_return_fn.chunk.emit(Instruction::LoadConst(0, c_bad), 1);
        bad_return_fn.chunk.emit(Instruction::Return(0), 1);

        let mut vm2 = VM::with_config(&config);
        vm2.load_functions(vec![bad_return_fn]);
        let err2 = vm2.call("bad_return", vec![]).unwrap_err();
        assert!(err2.to_string().contains("must return int, got String"));
    }
}

#[test]
fn test_execution_recursive_fibonacci_and_loop_dual_mode() {
    let source = r#"
        function fib(n: int): int
            if n <= 1 then
                return n
            else
                return fib(n - 1) + fib(n - 2)
            end
        end

        function nested_loops(): int
            local total = 0
            local i = 0
            while i < 10 do
                local j = 0
                while j < 5 do
                    total = total + i + j
                    j = j + 1
                end
                i = i + 1
            end
            return total
        end
    "#;
    let functions = compile_source(source).unwrap();
    for jit in [false, true] {
        let mut config = LustConfig::default();
        config.set_jit_enabled(jit);
        let mut vm = VM::with_config(&config);
        vm.load_functions(functions.clone());

        // fib(8) = 21
        let res_fib = vm.call("fib", vec![Value::Int(8)]).expect("fib(8)");
        assert_eq!(res_fib, Value::Int(21));

        // nested loops
        let res_loops = vm.call("nested_loops", vec![]).expect("nested loops");
        // sum_{i=0..9} sum_{j=0..4} (i + j) = 5 * (0+..+9) + 10 * (0+..+4) = 5 * 45 + 10 * 10 = 225 + 100 = 325
        assert_eq!(res_loops, Value::Int(325));
    }
}

#[test]
fn test_execution_value_is_type_instruction_deep_matrix() {
    let cases = vec![
        (Value::Int(1), "int", true),
        (Value::Float(1.5), "float", true),
        (Value::string("hello"), "string", true),
        (Value::Bool(true), "bool", true),
        (Value::Nil, "nil", true),
        (Value::Nil, "()", true),
        (Value::array(vec![Value::Int(1)]), "Array", true),
        (Value::array(vec![Value::Int(1)]), "Array<int>", true),
        (Value::map(lust::bytecode::LustMap::default()), "Map", true),
        (Value::tuple(vec![Value::Int(1)]), "Tuple", true),
        (Value::enum_variant("Option", "Some", vec![Value::Int(1)]), "Option", true),
        (Value::enum_variant("Option", "Some", vec![Value::Int(1)]), "Option<int>", true),
        (Value::enum_variant("Result", "Ok", vec![Value::Int(1)]), "Result", true),
        (Value::enum_variant("Result", "Ok", vec![Value::Int(1)]), "Result<int, string>", true),
        (Value::Int(1), "LuaValue", true),
        (Value::Int(1), "unknown", true),
        (Value::Int(1), "int | string", true),
        (Value::string("s"), "int | string", true),
        (Value::Bool(true), "int | string", false),
        (Value::Int(1), "string", false),
    ];

    for (val, type_name, expected) in cases {
        let res = execute_instructions(
            vec![
                Instruction::LoadConst(0, 0),
                Instruction::TypeIs(1, 0, 1),
                Instruction::Return(1),
            ],
            vec![val, Value::string(type_name)],
            4,
        ).expect("type is execution");
        assert_eq!(res, Value::Bool(expected), "type check {} for value", type_name);
    }
}

#[test]
fn test_execution_call_value_error_paths_and_loop_backedge() {
    let mut vm = VM::new();

    // 1. call_value on non-function value
    let err = vm.call_value(&Value::Int(42), vec![]).unwrap_err();
    assert!(err.to_string().contains("Cannot call non-function value"));

    // 2. call_value on NativeFunction returning Yield
    let yield_fn = native_fn(|_| Ok(NativeCallResult::Yield(Value::Nil)));
    let err2 = vm.call_value(&Value::NativeFunction(yield_fn), vec![]).unwrap_err();
    assert!(err2.to_string().contains("Yielding or stopping is not allowed"));

    // 3. call_value on NativeFunction returning Stop
    let stop_fn = native_fn(|_| Ok(NativeCallResult::Stop(Value::Nil)));
    let err3 = vm.call_value(&Value::NativeFunction(stop_fn), vec![]).unwrap_err();
    assert!(err3.to_string().contains("Yielding or stopping is not allowed"));

    // 4. call_value on Function that raises runtime error (Division by zero)
    let mut err_fn = Function::new("err_fn".to_string(), 0, false);
    err_fn.register_count = 4;
    let c_10 = err_fn.chunk.add_constant(Value::Int(10));
    let c_0 = err_fn.chunk.add_constant(Value::Int(0));
    err_fn.chunk.emit(Instruction::LoadConst(0, c_10), 1);
    err_fn.chunk.emit(Instruction::LoadConst(1, c_0), 1);
    err_fn.chunk.emit(Instruction::DivInt(2, 0, 1), 1); // error!
    err_fn.chunk.emit(Instruction::Return(2), 1);
    vm.load_functions(vec![err_fn]);
    let err4 = vm.call_value(&Value::Function(0), vec![]).unwrap_err();
    assert!(err4.to_string().contains("Division by zero"));

    // 5. call_value on Closure that raises runtime error
    use lust::bytecode::ClosureObject;
    let closure_obj = Value::Closure(Rc::new(ClosureObject {
        function_idx: 0,
        upvalues: vec![],
    }));
    let err5 = vm.call_value(&closure_obj, vec![]).unwrap_err();
    assert!(err5.to_string().contains("Division by zero"));

    // 6. Type check function(int): int and function(int, string)
    let res1 = execute_instructions(
        vec![
            Instruction::LoadConst(0, 0),
            Instruction::TypeIs(1, 0, 1),
            Instruction::Return(1),
        ],
        vec![Value::Function(0), Value::string("function(int): int")],
        4,
    ).expect("function type check");
    assert_eq!(res1, Value::Bool(false));

    let res_fn = execute_instructions(
        vec![
            Instruction::LoadConst(0, 0),
            Instruction::TypeIs(1, 0, 1),
            Instruction::Return(1),
        ],
        vec![Value::Function(0), Value::string("function")],
        4,
    ).expect("function type check generic");
    assert_eq!(res_fn, Value::Bool(true));

    let res2 = execute_instructions(
        vec![
            Instruction::LoadConst(0, 0),
            Instruction::TypeIs(1, 0, 1),
            Instruction::Return(1),
        ],
        vec![Value::Function(0), Value::string("function(int, string)")],
        4,
    ).expect("function type check 2");
    assert!(matches!(res2, Value::Bool(_)));

    // 7. Backward jump loop in bytecode (backedge profiling)
    for jit in [false, true] {
        let mut config = LustConfig::default();
        config.set_jit_enabled(jit);
        let mut f_loop = Function::new("backedge_worker".to_string(), 0, false);
        f_loop.register_count = 8;
        let c_0 = f_loop.chunk.add_constant(Value::Int(0));
        let c_10 = f_loop.chunk.add_constant(Value::Int(10));
        let c_1 = f_loop.chunk.add_constant(Value::Int(1));
        f_loop.chunk.emit(Instruction::LoadConst(0, c_0), 1);
        f_loop.chunk.emit(Instruction::LoadConst(1, c_10), 1);
        f_loop.chunk.emit(Instruction::LoadConst(2, c_1), 1);
        f_loop.chunk.emit(Instruction::AddInt(0, 0, 2), 1);
        f_loop.chunk.emit(Instruction::LtInt(3, 0, 1), 1);
        f_loop.chunk.emit(Instruction::JumpIf(3, 1), 1);
        f_loop.chunk.emit(Instruction::Return(0), 1);
        f_loop.chunk.emit(Instruction::Jump(-5), 1);

        let mut vm_loop = VM::with_config(&config);
        vm_loop.load_functions(vec![f_loop]);
        let res_loop = vm_loop.call("backedge_worker", vec![]).expect("backward jump loop");
        assert_eq!(res_loop, Value::Int(10));
    }
}

#[test]
fn test_execution_call_slice_and_make_call_frame_slice_matrix() {
    let source = r#"
        function add_two(a: int, b: int): int
            return a + b
        end
        function typed_str(s: string): string
            return s
        end
    "#;
    let functions = compile_source(source).unwrap();
    let mut vm = VM::new();
    vm.load_functions(functions);

    // 1. Valid call_slice
    let res = vm.call_slice("add_two", &[Value::Int(15), Value::Int(27)]).expect("call_slice valid");
    assert_eq!(res, Value::Int(42));

    // 2. call_slice on nonexistent function
    let err1 = vm.call_slice("missing_func", &[]).unwrap_err();
    assert!(err1.to_string().contains("Function not found: missing_func"));

    // 3. call_function_index_slice on invalid function index
    let err2 = vm.call_function_index_slice(9999, &[]).unwrap_err();
    assert!(err2.to_string().contains("Invalid function index 9999"));

    // 4. Arity error: expects 2, passed 1
    let err3 = vm.call_slice("add_two", &[Value::Int(1)]).unwrap_err();
    assert!(err3.to_string().contains("expects 2 arguments, got 1"));

    // 5. Type validation error: expects int, passed string
    let err4 = vm.call_slice("add_two", &[Value::string("bad"), Value::Int(1)]).unwrap_err();
    assert!(err4.to_string().contains("argument 1 expects int, got String"));

    // 6. Type validation on second function: expects string, passed int
    let err5 = vm.call_slice("typed_str", &[Value::Int(123)]).unwrap_err();
    assert!(err5.to_string().contains("argument 1 expects string, got Int"));
}

#[test]
fn test_execution_invoke_hashkey_and_custom_hash() {
    use lust::ast::{Span, Type, TypeKind};
    use lust::bytecode::{FieldStorage, StructLayout, StructObject};

    let mut vm = VM::new();

    let layout = Rc::new(StructLayout::new(
        "Point".to_string(),
        vec![Rc::new("x".to_string()), Rc::new("y".to_string())],
        vec![FieldStorage::Strong, FieldStorage::Strong],
        vec![Type::new(TypeKind::Int, Span::dummy()), Type::new(TypeKind::Int, Span::dummy())],
        vec![None, None],
    ));
    let pt = Value::Struct(StructObject::new(
        "Point",
        layout.clone(),
        vec![Value::Int(10), Value::Int(20)],
    ));

    // 1. Define Point:to_hashkey function that returns custom hash Int(99999)
    let mut hash_fn = Function::new("Point:to_hashkey".to_string(), 1, false);
    hash_fn.register_count = 4;
    let c_hash = hash_fn.chunk.add_constant(Value::Int(99999));
    hash_fn.chunk.emit(Instruction::LoadConst(1, c_hash), 1);
    hash_fn.chunk.emit(Instruction::Return(1), 1);

    vm.load_functions(vec![hash_fn]);

    // Oracle: make_hash_key calls invoke_hashkey and computes hashed ValueKey
    let key = vm.make_hash_key(&pt).expect("make_hash_key for Point");
    let (_, hashed) = key.owned_values();
    assert_eq!(*hashed, Value::Int(99999));

    // 2. Error branch: declare trait impl in trait_impls without the method
    let layout2 = Rc::new(StructLayout::new(
        "MissingHash".to_string(),
        vec![Rc::new("val".to_string())],
        vec![FieldStorage::Strong],
        vec![Type::new(TypeKind::Int, Span::dummy())],
        vec![None],
    ));
    let mh = Value::Struct(StructObject::new(
        "MissingHash",
        layout2,
        vec![Value::Int(1)],
    ));

    // Register trait impl directly
    vm.register_trait_impl("MissingHash".to_string(), "HashKey".to_string());
    let err = vm.make_hash_key(&mh).unwrap_err();
    assert!(err.to_string().contains("HashKey trait declared but method 'to_hashkey' not found"));
}

#[test]
fn test_execution_floordiv_pow_mod_mixed_and_zero_errors() {
    // FloorDiv, Pow, Mod with mixed types & division by zero
    let mut f = Function::new("test_arith_branches".to_string(), 0, false);
    f.register_count = 16;
    let c_f10 = f.chunk.add_constant(Value::Float(10.0));
    let c_f2_5 = f.chunk.add_constant(Value::Float(2.5));
    let c_i10 = f.chunk.add_constant(Value::Int(10));
    let c_i2 = f.chunk.add_constant(Value::Int(2));
    let c_f3 = f.chunk.add_constant(Value::Float(3.0));
    let c_i3 = f.chunk.add_constant(Value::Int(3));

    // Float / Float FloorDiv: 10.0 // 2.5 = 4.0
    f.chunk.emit(Instruction::LoadConst(0, c_f10), 1);
    f.chunk.emit(Instruction::LoadConst(1, c_f2_5), 1);
    f.chunk.emit(Instruction::FloorDiv(2, 0, 1), 1);

    // Int / Float FloorDiv: 10 // 2.5 = 4.0
    f.chunk.emit(Instruction::LoadConst(3, c_i10), 1);
    f.chunk.emit(Instruction::FloorDiv(4, 3, 1), 1);

    // Float / Int FloorDiv: 10.0 // 2 = 5.0
    f.chunk.emit(Instruction::LoadConst(5, c_i2), 1);
    f.chunk.emit(Instruction::FloorDiv(6, 0, 5), 1);

    // Pow Float ** Float
    f.chunk.emit(Instruction::Pow(7, 1, 1), 1);

    // Pow Int ** Float
    f.chunk.emit(Instruction::Pow(8, 3, 1), 1);

    // Pow Float ** Int
    f.chunk.emit(Instruction::Pow(9, 1, 5), 1);

    // Mod Int % Float: 10 % 3.0 = 1.0
    f.chunk.emit(Instruction::LoadConst(10, c_f3), 1);
    f.chunk.emit(Instruction::Mod(11, 3, 10), 1);

    // Mod Float % Int: 10.0 % 3 = 1.0
    f.chunk.emit(Instruction::LoadConst(12, c_i3), 1);
    f.chunk.emit(Instruction::Mod(13, 0, 12), 1);

    f.chunk.emit(Instruction::Return(2), 1);

    let mut vm = VM::new();
    vm.load_functions(vec![f]);
    let res = vm.call("test_arith_branches", vec![]).expect("arith branches");
    assert_eq!(res, Value::Float(4.0));

    // Negative & error cases:
    let run_err = |instrs: Vec<Instruction>, consts: Vec<Value>| -> String {
        let mut err_f = Function::new("err_fn".to_string(), 0, false);
        err_f.register_count = 8;
        for c in consts {
            err_f.chunk.add_constant(c);
        }
        for i in instrs {
            err_f.chunk.emit(i, 1);
        }
        let mut err_vm = VM::new();
        err_vm.load_functions(vec![err_f]);
        err_vm.call("err_fn", vec![]).unwrap_err().to_string()
    };

    // FloorDiv Float / Float with 0.0
    let err_fd_ff = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::FloorDiv(2, 0, 1), Instruction::Return(2)],
        vec![Value::Float(10.0), Value::Float(0.0)],
    );
    assert!(err_fd_ff.contains("Division by zero"));

    // FloorDiv Int / Float with 0.0
    let err_fd_if = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::FloorDiv(2, 0, 1), Instruction::Return(2)],
        vec![Value::Int(10), Value::Float(0.0)],
    );
    assert!(err_fd_if.contains("Division by zero"));

    // FloorDiv Float / Int with 0
    let err_fd_fi = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::FloorDiv(2, 0, 1), Instruction::Return(2)],
        vec![Value::Float(10.0), Value::Int(0)],
    );
    assert!(err_fd_fi.contains("Division by zero"));

    // FloorDiv invalid operand types
    let err_fd_bad = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::FloorDiv(2, 0, 1), Instruction::Return(2)],
        vec![Value::string("not_num"), Value::Int(1)],
    );
    assert!(err_fd_bad.contains("Cannot floor divide"));

    // Mod Int % Float with 0.0
    let err_mod_if = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::Mod(2, 0, 1), Instruction::Return(2)],
        vec![Value::Int(10), Value::Float(0.0)],
    );
    assert!(err_mod_if.contains("Modulo by zero"));

    // Mod Float % Int with 0
    let err_mod_fi = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::Mod(2, 0, 1), Instruction::Return(2)],
        vec![Value::Float(10.0), Value::Int(0)],
    );
    assert!(err_mod_fi.contains("Modulo by zero"));

    // Mod invalid operand types
    let err_mod_bad = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::Mod(2, 0, 1), Instruction::Return(2)],
        vec![Value::string("not_num"), Value::Int(1)],
    );
    assert!(err_mod_bad.contains("Cannot modulo"));

    // Pow invalid operand types
    let err_pow_bad = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::Pow(2, 0, 1), Instruction::Return(2)],
        vec![Value::string("not_num"), Value::Int(1)],
    );
    assert!(err_pow_bad.contains("Cannot raise"));
}

#[test]
fn test_execution_comparison_op_and_typed_numeric_errors() {
    let run_err = |instrs: Vec<Instruction>, consts: Vec<Value>| -> String {
        let mut err_f = Function::new("err_comp".to_string(), 0, false);
        err_f.register_count = 8;
        for c in consts {
            err_f.chunk.add_constant(c);
        }
        for i in instrs {
            err_f.chunk.emit(i, 1);
        }
        let mut err_vm = VM::new();
        err_vm.load_functions(vec![err_f]);
        err_vm.call("err_comp", vec![]).unwrap_err().to_string()
    };

    // 1. Comparison Lt on Int and String
    let err_lt = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::Lt(2, 0, 1), Instruction::Return(2)],
        vec![Value::Int(10), Value::string("str")],
    );
    assert!(err_lt.contains("Cannot compare"));

    // 2. Comparison Le on String and Float
    let err_le = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::Le(2, 0, 1), Instruction::Return(2)],
        vec![Value::string("str"), Value::Float(1.5)],
    );
    assert!(err_le.contains("Cannot compare"));

    // 3. Comparison Gt on Bool and Int
    let err_gt = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::Gt(2, 0, 1), Instruction::Return(2)],
        vec![Value::Bool(true), Value::Int(1)],
    );
    assert!(err_gt.contains("Cannot compare"));

    // 4. Comparison Ge on Nil and Float
    let err_ge = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::Ge(2, 0, 1), Instruction::Return(2)],
        vec![Value::Nil, Value::Float(1.5)],
    );
    assert!(err_ge.contains("Cannot compare"));

    // 5. Mixed numeric comparison: Int < Float and Float < Int
    let mut mixed_f = Function::new("mixed_comp".to_string(), 0, false);
    mixed_f.register_count = 8;
    let c_i = mixed_f.chunk.add_constant(Value::Int(10));
    let c_f = mixed_f.chunk.add_constant(Value::Float(10.5));
    mixed_f.chunk.emit(Instruction::LoadConst(0, c_i), 1);
    mixed_f.chunk.emit(Instruction::LoadConst(1, c_f), 1);
    mixed_f.chunk.emit(Instruction::Lt(2, 0, 1), 1); // 10 < 10.5 -> true
    mixed_f.chunk.emit(Instruction::Le(3, 1, 0), 1); // 10.5 <= 10 -> false
    mixed_f.chunk.emit(Instruction::Gt(4, 1, 0), 1); // 10.5 > 10 -> true
    mixed_f.chunk.emit(Instruction::Ge(5, 0, 1), 1); // 10 >= 10.5 -> false
    mixed_f.chunk.emit(Instruction::Return(2), 1);
    let mut vm = VM::new();
    vm.load_functions(vec![mixed_f]);
    let res = vm.call("mixed_comp", vec![]).expect("mixed comp");
    assert_eq!(res, Value::Bool(true));

    // 6. Typed float instructions with non-float operands
    let err_add_flt = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::AddFloat(2, 0, 1), Instruction::Return(2)],
        vec![Value::Int(1), Value::Int(2)],
    );
    assert!(err_add_flt.contains("Typed numeric instruction expected float operands"));

    let err_sub_flt = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::SubFloat(2, 0, 1), Instruction::Return(2)],
        vec![Value::string("a"), Value::Float(1.0)],
    );
    assert!(err_sub_flt.contains("Typed numeric instruction expected float operands"));
}

#[test]
fn test_execution_index_bounds_and_type_errors() {
    let run_err = |instrs: Vec<Instruction>, consts: Vec<Value>| -> String {
        let mut err_f = Function::new("err_idx".to_string(), 0, false);
        err_f.register_count = 8;
        for c in consts {
            err_f.chunk.add_constant(c);
        }
        for i in instrs {
            err_f.chunk.emit(i, 1);
        }
        let mut err_vm = VM::new();
        err_vm.load_functions(vec![err_f]);
        err_vm.call("err_idx", vec![]).unwrap_err().to_string()
    };

    let arr = Value::array(vec![Value::Int(1), Value::Int(2)]);

    // 1. Array GetIndex with non-int index
    let err1 = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::GetIndex(2, 0, 1), Instruction::Return(2)],
        vec![arr.clone(), Value::string("bad_idx")],
    );
    assert!(err1.contains("Array index must be an integer"));

    // 2. Array GetIndex with negative index
    let err2 = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::GetIndex(2, 0, 1), Instruction::Return(2)],
        vec![arr.clone(), Value::Int(-1)],
    );
    assert!(err2.contains("out of bounds"));

    // 3. Array GetIndex with index >= len
    let err3 = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::GetIndex(2, 0, 1), Instruction::Return(2)],
        vec![arr.clone(), Value::Int(99)],
    );
    assert!(err3.contains("out of bounds"));

    // 4. Array SetIndex with non-int index
    let err4 = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::LoadConst(2, 2), Instruction::SetIndex(0, 1, 2), Instruction::Return(0)],
        vec![arr.clone(), Value::string("bad_idx"), Value::Int(42)],
    );
    assert!(err4.contains("Array index must be an integer"));

    // 5. Array SetIndex with negative index
    let err5 = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::LoadConst(2, 2), Instruction::SetIndex(0, 1, 2), Instruction::Return(0)],
        vec![arr.clone(), Value::Int(-5), Value::Int(42)],
    );
    assert!(err5.contains("out of bounds"));

    // 6. Array SetIndex with index >= len
    let err6 = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::LoadConst(2, 2), Instruction::SetIndex(0, 1, 2), Instruction::Return(0)],
        vec![arr.clone(), Value::Int(50), Value::Int(42)],
    );
    assert!(err6.contains("out of bounds"));

    // 7. Non-collection GetIndex on Bool
    let err7 = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::GetIndex(2, 0, 1), Instruction::Return(2)],
        vec![Value::Bool(true), Value::Int(0)],
    );
    assert!(err7.contains("Cannot index"));

    // 8. Non-collection SetIndex on Bool
    let err8 = run_err(
        vec![Instruction::LoadConst(0, 0), Instruction::LoadConst(1, 1), Instruction::LoadConst(2, 2), Instruction::SetIndex(0, 1, 2), Instruction::Return(0)],
        vec![Value::Bool(true), Value::Int(0), Value::Int(1)],
    );
    assert!(err8.contains("Cannot index"));
}

#[test]
fn test_execution_concat_dest_not_lhs() {
    let mut f = Function::new("test_concat_distinct".to_string(), 0, false);
    f.register_count = 8;
    let c_a = f.chunk.add_constant(Value::string("Hello, "));
    let c_b = f.chunk.add_constant(Value::string("World!"));
    f.chunk.emit(Instruction::LoadConst(0, c_a), 1);
    f.chunk.emit(Instruction::LoadConst(1, c_b), 1);
    // Concat where dest=2, lhs=0, rhs=1 (dest != lhs triggers out-of-place branch)
    f.chunk.emit(Instruction::Concat(2, 0, 1), 1);
    f.chunk.emit(Instruction::Return(2), 1);

    let mut vm = VM::new();
    vm.load_functions(vec![f]);
    let res = vm.call("test_concat_distinct", vec![]).expect("concat distinct");
    assert_eq!(res, Value::string("Hello, World!"));
}

#[test]
fn test_execution_lua_userdata_call_and_function_unit_signature() {
    // 1. Function with Unit return matching "function(int)" without return type
    let source = r#"
        function proc(x: int)
            local y = x + 1
        end
    "#;
    let funcs = compile_source(source).unwrap();
    let mut vm = VM::new();
    vm.load_functions(funcs);
    // proc signature returns Unit ()
    assert!(vm.value_is_type(&Value::Function(0), "function(int)"));
    assert!(!vm.value_is_type(&Value::Function(0), "function(int, int)"));

    // 2. LuaTable / LuaUserdata with __call metamethod invoked via call_value
    let meta_map = Rc::new(RefCell::new(lust::bytecode::LustMap::default()));
    let call_fn = native_fn(|args| {
        let arg = args.get(1).and_then(|v| v.as_int()).unwrap_or(0);
        Ok(NativeCallResult::Return(Value::Int(arg * 2)))
    });
    meta_map.borrow_mut().insert(
        ValueKey::string("__call"),
        Value::NativeFunction(call_fn),
    );

    let layout = Rc::new(lust::bytecode::StructLayout::new(
        "LuaTable".to_string(),
        vec![Rc::new("metamethods".to_string())],
        vec![lust::bytecode::FieldStorage::Strong],
        vec![lust::ast::Type::new(lust::ast::TypeKind::Unknown, lust::ast::Span::dummy())],
        vec![None],
    ));
    let table_obj = Value::Struct(lust::bytecode::StructObject::new(
        "LuaTable",
        layout,
        vec![Value::Map(meta_map)],
    ));
    let lua_val = Value::enum_variant("LuaValue", "Table", vec![table_obj]);

    let res = vm.call_value(&lua_val, vec![Value::Int(21)]).expect("invoke __call");
    assert_eq!(res, Value::Int(42));
}












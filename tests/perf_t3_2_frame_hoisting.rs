#![cfg(feature = "std")]

//! PERF-T3-2: Hoist Frame/Chunk Slice Accesses Outside `'dispatch: loop` with Strict Reload Barriers.
//!
//! Verifies:
//! 1. `VM::frame_cache_hoisting_enabled()` is true and pure intra-frame loops execute with O(1)
//!    frame-cache reloads (`VM::frame_hoist_reload_count()`).
//! 2. Deep recursive calls (`depth > 64`, forcing `Vec<Box<CallFrame>>` reallocation), native host
//!    callbacks that re-enter `vm.call(...)`, and `CallMethod`/`Return` transitions produce exact
//!    results with zero stale frame references (mechanically verified by debug-mode invariant
//!    assertions on every instruction).
//! 3. Adversarial test 1: Host callback that mutates `self.call_stack` via deep nested `vm.call`,
//!    reallocates `self.functions` via `load_mir_function`, and a host callback returning an error
//!    mid-dispatch leave the VM cleanly recoverable.
//! 4. Adversarial test 2: Stack overflow (`max_stack_depth = 1000`) error path unwinds cleanly and
//!    leaves the VM completely recoverable for subsequent calls.
//! 5. Adversarial test 3: Cranelift + production OSR + forced safepoint deopt (and mid-loop poll
//!    deopt under `--features test-deopt`) modify the top frame during execution and resume cleanly
//!    with exact oracle results and zero debug_assert trips.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::sync::Mutex;

use lust::bytecode::compiler::Compiler;
use lust::bytecode::{Function, NativeCallResult, Value, native_fn};
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::cranelift_backend::{safepoint_request_deopt, safepoint_reset};
use lust::mir::ir::MirModule;
use lust::mir::lowering::AstLowering;
use lust::parser::Parser;
use lust::typechecker::TypeChecker;
use lust::vm::osr::GLOBAL_OSR_COUNTERS;
use lust::vm::{VmBackend, VM};

static OSR_TEST_MUTEX: Mutex<()> = Mutex::new(());

fn compile_source(source: &str) -> lust::Result<(Vec<Function>, hashbrown::HashMap<String, lust::ast::StructDef>)> {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize()?;
    let items = Parser::new(tokens).parse()?;
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    let mut checker = TypeChecker::with_config(&config);
    let _ = checker.check_module(&items);
    let mut struct_defs = hashbrown::HashMap::new();
    for item in &items {
        if let lust::ast::ItemKind::Struct(s) = &item.kind {
            struct_defs.insert(s.name.clone(), s.clone());
        }
    }
    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    let funcs = compiler.compile_module(&items)?;
    Ok((funcs, struct_defs))
}

fn plain_vm() -> VM {
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    VM::with_config(&config)
}

fn parse_and_lower_module(source: &str, module_name: &str) -> MirModule {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("Tokenization must succeed");
    let items = Parser::new(tokens).parse().expect("Parsing must succeed");

    let mut module = MirModule::new(module_name.into());
    for item in items {
        if let lust::ast::ItemKind::Function(func_def) = item.kind {
            let mir_fn = AstLowering::lower_function(&func_def).expect("MIR lowering must succeed");
            module.add_function(mir_fn);
        }
    }
    module
}

#[test]
fn test_perf_t3_2_frame_cache_reloads_across_reallocation() {
    // Contract gate (RED before PERF-T3-2 implementation):
    assert!(
        VM::frame_cache_hoisting_enabled(),
        "PERF-T3-2 frame cache hoisting must be enabled"
    );

    let src = r#"
        extern
            function host_reenter(depth: int): int
        end

        struct Counter
            value: int
        end

        impl Counter
            function bump(self, delta: int): int
                self.value = self.value + delta
                return self.value
            end

            function bump_recursive(self, n: int): int
                if n <= 0 then
                    return self.value
                end
                self.value = self.value + n
                return self:bump_recursive(n - 1)
            end
        end

        function pure_hot_loop(n: int): int
            local acc = 0
            local i = 0
            while i < n do
                acc = acc + (i * 3 - 1)
                i = i + 1
            end
            return acc
        end

        function deep_rec(n: int): int
            if n <= 0 then
                return 0
            end
            return n + deep_rec(n - 1)
        end

        function run_reentrant_host_loop(k: int): int
            local total = 0
            local i = 1
            while i <= k do
                total = total + host_reenter(i)
                i = i + 1
            end
            return total
        end

        function run_method_transitions(n: int): int
            local c = Counter { value = 0 }
            local i = 1
            while i <= n do
                local _ = c:bump(i)
                i = i + 1
            end
            return c:bump_recursive(10)
        end
    "#;

    let (funcs, structs) = compile_source(src).expect("compile must succeed");
    let mut vm = plain_vm();
    vm.register_structs(&structs);
    vm.load_functions(funcs);

    let host_invocations = Rc::new(RefCell::new(0usize));
    let hi_clone = host_invocations.clone();
    vm.register_native(
        "host_reenter",
        Value::NativeFunction(native_fn(move |args| {
            let depth = args.first().and_then(|v| v.as_int()).unwrap_or(0);
            *hi_clone.borrow_mut() += 1;
            VM::with_current(|cur_vm| {
                // Re-enter VM recursively with depth > 64 to force call_stack reallocation
                // while the outer caller frame is suspended on the stack.
                let nested = cur_vm
                    .call("deep_rec", vec![Value::Int(70 + depth)])
                    .map_err(|e| e.to_string())?;
                Ok(NativeCallResult::Return(nested))
            })
        })),
    );

    // 1. Pure intra-frame tight loop: executes tens of thousands of instructions with O(1) reloads.
    vm.reset_gas_counter();
    let reloads_before = VM::frame_hoist_reload_count();
    let loop_res = vm
        .call("pure_hot_loop", vec![Value::Int(5_000)])
        .expect("pure_hot_loop must succeed");
    let reloads_after = VM::frame_hoist_reload_count();
    let expected_loop: i64 = (0..5_000i64).map(|i| i * 3 - 1).sum();
    assert_eq!(loop_res, Value::Int(expected_loop));
    assert!(
        vm.gas_used() > 20_000,
        "expected > 20_000 instructions executed, got {}",
        vm.gas_used()
    );
    #[cfg(debug_assertions)]
    {
        let reload_delta = reloads_after - reloads_before;
        assert!(
            reload_delta >= 1 && reload_delta <= 4,
            "pure intra-frame loop of 5,000 iterations must do O(1) frame reloads, got {}",
            reload_delta
        );
    }

    // 2. Deep recursion (depth = 80 > 64) forcing Vec<Box<CallFrame>> reallocations across doublings.
    let rec_reloads_before = VM::frame_hoist_reload_count();
    let rec_res = vm
        .call("deep_rec", vec![Value::Int(80)])
        .expect("deep_rec(80) must succeed");
    let rec_reloads_after = VM::frame_hoist_reload_count();
    assert_eq!(rec_res, Value::Int(80 * 81 / 2));
    #[cfg(debug_assertions)]
    {
        let rec_delta = rec_reloads_after - rec_reloads_before;
        assert!(
            rec_delta >= 160,
            "depth-80 recursion must reload on every call push and return pop, got {}",
            rec_delta
        );
    }

    // 3. Re-entrant native host callbacks mutating call_stack via nested deep vm.call.
    let reenter_res = vm
        .call("run_reentrant_host_loop", vec![Value::Int(5)])
        .expect("run_reentrant_host_loop must succeed");
    let expected_reenter: i64 = (1..=5i64).map(|i| (70 + i) * (71 + i) / 2).sum();
    assert_eq!(reenter_res, Value::Int(expected_reenter));
    assert_eq!(*host_invocations.borrow(), 5);

    // 4. CallMethod / Return transitions (cached & uncached method calls + recursive method calls).
    let method_res = vm
        .call("run_method_transitions", vec![Value::Int(100)])
        .expect("run_method_transitions must succeed");
    let expected_methods = (100 * 101 / 2) + (10 * 11 / 2);
    assert_eq!(method_res, Value::Int(expected_methods));
}

#[test]
fn test_perf_t3_2_host_callback_reentrancy_functions_realloc_and_error() {
    let src = r#"
        extern
            function host_grow_functions_and_reenter(x: int): int
            function host_fail_mid_dispatch(x: int): int
        end

        function helper_rec(n: int): int
            if n <= 0 then
                return 1
            end
            return 1 + helper_rec(n - 1)
        end

        function caller_with_realloc(n: int): int
            local sum = 0
            local i = 0
            while i < n do
                sum = sum + host_grow_functions_and_reenter(i)
                i = i + 1
            end
            return sum
        end

        function caller_with_error(trigger: int): int
            local acc = 10
            if trigger > 0 then
                acc = acc + host_fail_mid_dispatch(trigger)
            end
            return acc + 5
        end
    "#;

    let (funcs, _) = compile_source(src).expect("compile must succeed");
    let mut vm = plain_vm();
    vm.load_functions(funcs);

    let counter = Rc::new(RefCell::new(0usize));
    let counter_clone = counter.clone();
    vm.register_native(
        "host_grow_functions_and_reenter",
        Value::NativeFunction(native_fn(move |args| {
            let x = args.first().and_then(|v| v.as_int()).unwrap_or(0);
            let idx = {
                let mut c = counter_clone.borrow_mut();
                *c += 1;
                *c
            };
            VM::with_current(|cur_vm| {
                // Dynamically add brand-new MIR functions mid-dispatch to force `self.functions`
                // Vec reallocation while the caller frame is active on the call stack!
                let dyn_src = format!(
                    "function __dyn_added_{idx}(v: int): int\n    return v + {idx}\nend"
                );
                let dyn_mod = parse_and_lower_module(&dyn_src, "dyn_realloc");
                for mir_fn in dyn_mod.functions {
                    cur_vm
                        .load_mir_function(mir_fn)
                        .map_err(|e| e.to_string())?;
                }

                // Also force call_stack reallocation via deep nested vm.call.
                let rec_val = cur_vm
                    .call("helper_rec", vec![Value::Int(75)])
                    .map_err(|e| e.to_string())?
                    .as_int()
                    .unwrap_or(0);

                // Call the newly appended function too!
                let dyn_val = cur_vm
                    .call(&format!("__dyn_added_{idx}"), vec![Value::Int(x)])
                    .map_err(|e| e.to_string())?
                    .as_int()
                    .unwrap_or(0);

                Ok(NativeCallResult::Return(Value::Int(rec_val + dyn_val)))
            })
        })),
    );

    vm.register_native(
        "host_fail_mid_dispatch",
        Value::NativeFunction(native_fn(move |args| {
            let x = args.first().and_then(|v| v.as_int()).unwrap_or(0);
            Err(format!("host boom on {x}"))
        })),
    );

    // 1. Reallocate both `self.functions` and `self.call_stack` repeatedly during a hot loop.
    let res = vm
        .call("caller_with_realloc", vec![Value::Int(20)])
        .expect("caller_with_realloc must survive Vec<Function> & Vec<Box<CallFrame>> reallocations");
    // Each iteration i in 0..20 adds: helper_rec(75) = 76, plus __dyn_added_{i+1}(i) = i + (i + 1)
    let expected: i64 = (0..20i64).map(|i| 76 + i + (i + 1)).sum();
    assert_eq!(res, Value::Int(expected));

    // 2. Host callback returning an error mid-dispatch propagates cleanly.
    let err = vm
        .call("caller_with_error", vec![Value::Int(42)])
        .expect_err("host_fail_mid_dispatch must return an error");
    let err_msg = format!("{err}");
    assert!(
        err_msg.contains("host boom on 42"),
        "expected host error message, got: {err_msg}"
    );

    // 3. VM must remain cleanly recoverable after mid-dispatch host callback error!
    let recovered = vm
        .call("caller_with_error", vec![Value::Int(0)])
        .expect("VM must recover cleanly after host error");
    assert_eq!(recovered, Value::Int(15));
}

#[test]
fn test_perf_t3_2_stack_overflow_leaves_vm_recoverable() {
    let src = r#"
        function infinite_rec(n: int): int
            return 1 + infinite_rec(n + 1)
        end

        function healthy_sum(n: int): int
            local acc = 0
            local i = 1
            while i <= n do
                acc = acc + i
                i = i + 1
            end
            return acc
        end
    "#;

    let (funcs, _) = compile_source(src).expect("compile must succeed");
    let mut vm = plain_vm();
    vm.load_functions(funcs);

    // Trigger stack overflow (default max_stack_depth = 1000).
    let err = vm
        .call("infinite_rec", vec![Value::Int(0)])
        .expect_err("infinite_rec must hit Stack overflow");
    let msg = format!("{err}");
    assert!(
        msg.contains("Stack overflow"),
        "expected Stack overflow error, got: {msg}"
    );

    // Subsequent calls on the same VM must execute cleanly with fresh frame state.
    let ok = vm
        .call("healthy_sum", vec![Value::Int(200)])
        .expect("VM must be cleanly recoverable after Stack overflow");
    assert_eq!(ok, Value::Int(200 * 201 / 2));
}

#[test]
fn test_perf_t3_2_cranelift_osr_and_deopt_frame_reloads() {
    let _lock = OSR_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    GLOBAL_OSR_COUNTERS.reset();
    safepoint_reset();

    let source = r#"
        function setup_then_osr_loop(n: int): int
            local bias: int = host_bias(9)
            local sum: int = bias
            local i: int = 0
            while i < n do
                sum = sum + i
                i = i + 1
            end
            return sum
        end
    "#;

    let module = parse_and_lower_module(source, "perf_t3_2_osr_deopt");
    let n = 2_000i64;
    let expected = 109i64 + (n * (n - 1)) / 2;

    let register_bias = |vm: &mut VM| {
        vm.register_native(
            "host_bias",
            Value::NativeFunction(native_fn(|args| {
                let x = args.first().and_then(|v| v.as_int()).unwrap_or(0);
                Ok(NativeCallResult::Return(Value::Int(x + 100)))
            })),
        );
    };

    // 1. Production OSR entry under Cranelift backend with hoisted frame cache active.
    let mut vm_osr = VM::with_config(&LustConfig::default().with_cranelift_jit(true));
    register_bias(&mut vm_osr);
    vm_osr
        .load_mir_module(&module)
        .expect("load_mir_module must succeed");
    vm_osr.set_backend(VmBackend::Cranelift);

    let osr_before = GLOBAL_OSR_COUNTERS.entry_successes.load(Ordering::SeqCst);
    let osr_res = vm_osr
        .call("setup_then_osr_loop", vec![Value::Int(n)])
        .expect("Cranelift OSR loop must succeed");
    let osr_after = GLOBAL_OSR_COUNTERS.entry_successes.load(Ordering::SeqCst);
    assert_eq!(osr_res, Value::Int(expected));
    assert!(
        osr_after > osr_before,
        "expected production OSR entry_successes to increase (delta = {})",
        osr_after - osr_before
    );

    // 2. Forced safepoint deopt inside OSR-entered loop modifies top frame registers & ip;
    //    interpreter must reload frame state cleanly and complete with exact oracle result.
    let mut vm_deopt = VM::with_config(&LustConfig::default().with_cranelift_jit(true));
    register_bias(&mut vm_deopt);
    vm_deopt
        .load_mir_module(&module)
        .expect("load_mir_module must succeed");
    vm_deopt.set_backend(VmBackend::Cranelift);

    safepoint_request_deopt(true);
    let deopt_before = GLOBAL_OSR_COUNTERS.deopt_count.load(Ordering::SeqCst);
    let deopt_res = vm_deopt
        .call("setup_then_osr_loop", vec![Value::Int(n)])
        .expect("Forced OSR safepoint deopt must resume cleanly with hoisted frame cache");
    let deopt_after = GLOBAL_OSR_COUNTERS.deopt_count.load(Ordering::SeqCst);
    safepoint_reset();

    assert_eq!(deopt_res, Value::Int(expected));
    assert_eq!(deopt_after - deopt_before, 1);

    // 3. Under `test-deopt`: trigger mid-loop deopt at poll ordinal 30 so mutated native locals
    //    are reconstructed into the interpreter frame mid-loop.
    #[cfg(feature = "test-deopt")]
    {
        let mut vm_mid = VM::with_config(&LustConfig::default().with_cranelift_jit(true));
        register_bias(&mut vm_mid);
        vm_mid
            .load_mir_module(&module)
            .expect("load_mir_module must succeed");
        vm_mid.set_backend(VmBackend::Cranelift);

        lust::mir::cranelift_backend::safepoint_request_deopt_on_poll(
            Some("setup_then_osr_loop".to_string()),
            30,
        );
        let mid_before = GLOBAL_OSR_COUNTERS.deopt_count.load(Ordering::SeqCst);
        let mid_res = vm_mid
            .call("setup_then_osr_loop", vec![Value::Int(n)])
            .expect("Mid-loop poll-30 OSR deopt must resume cleanly");
        let mid_after = GLOBAL_OSR_COUNTERS.deopt_count.load(Ordering::SeqCst);
        safepoint_reset();

        assert_eq!(mid_res, Value::Int(expected));
        assert_eq!(mid_after - mid_before, 1);
    }
}

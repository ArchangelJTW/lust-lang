#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! NG-W11-4B Acceptance & Regression Suite:
//! 1. `Function::trusted_signature` & `Function::compute_trusted_signature()` across AST->Bytecode,
//!    MIR->Bytecode (`MirToBytecodeCompiler`), and VM loading (`CallMeta`).
//! 2. Zero `'frame_fast: loop` exits (`VM::frame_hoist_reload_count()` delta == 1) on
//!    `fib(20)`, `calls.lust` (`N = 10_000`), and `methods.lust` (`N = 10_000`) in `--jit=off` mode.
//! 3. Preserved `ArgCheck::Deep` validation on host/dynamic calls (`vm.call` / `vm.call_value`)
//!    and return validation on outermost returns to host.
//! 4. Deterministic `SEC-04` `max_stack_depth` stack overflow enforcement & VM recoverability.
//! 5. Deterministic `SEC-01` exact gas limit enforcement inside recursive in-loop calls.
//! 6. Exact `non_plain_mask` Rc cleanup and polymorphic `CallMethod` IC updates inside `'frame_fast`.

use std::cell::RefCell;
use std::rc::Rc;

use lust::ast::{Span, Type, TypeKind};
use lust::bytecode::chunk::ChunkIcSlot;
use lust::bytecode::compiler::Compiler;
use lust::bytecode::{Function, Instruction, NativeCallResult, Value, native_fn};
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::ir::{
    BasicBlock, BlockId, FunctionSignature as MirFnSig, MirConstant, MirFunction, MirType, Operand,
    Terminator,
};
use lust::mir::vm_backend::MirToBytecodeCompiler;
use lust::parser::Parser;
use lust::typechecker::{FunctionSignature, TypeChecker};
use lust::vm::{VmBackend, VM};

fn compile_typed_vm(source: &str) -> VM {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner).tokenize().expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");
    let mut config = LustConfig::default();
    config.set_jit_enabled(false);
    let mut checker = TypeChecker::with_config(&config);
    checker.check_module(&items).expect("typecheck");
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
    let funcs = compiler.compile_module(&items).expect("compile_module");
    let mut vm = VM::with_config(&config);
    vm.set_backend(VmBackend::Bytecode);
    vm.register_structs(&struct_defs);
    vm.load_functions(funcs);
    vm
}

fn ty(kind: TypeKind) -> Type {
    Type::new(kind, Span::dummy())
}

#[test]
fn ng_w11_4b_trusted_signature_classification_ast_and_mir() {
    let source = r#"
        struct Counter
            value: int
        end

        impl Counter
            function bump(self, d: int): int
                self.value = self.value + d
                return self.value
            end
        end

        function static_fn(a: int, b: float): bool
            return a > 0
        end

        function dynamic_unknown(x: unknown): int
            return 1
        end

        function generic_fn<T>(x: T): T
            return x
        end
    "#;

    let vm = compile_typed_vm(source);
    let static_fn = vm.function_by_name("static_fn").expect("static_fn");
    assert!(static_fn.trusted_signature);
    assert!(static_fn.compute_trusted_signature());

    let bump_fn = vm.function_by_name("Counter:bump").expect("Counter:bump");
    assert!(bump_fn.trusted_signature);
    assert!(bump_fn.compute_trusted_signature());

    let dyn_fn = vm
        .function_by_name("dynamic_unknown")
        .expect("dynamic_unknown");
    assert!(!dyn_fn.trusted_signature);
    assert!(!dyn_fn.compute_trusted_signature());

    let gen_fn = vm.function_by_name("generic_fn").expect("generic_fn");
    assert!(!gen_fn.trusted_signature);
    assert!(!gen_fn.compute_trusted_signature());

    // Direct Function::compute_trusted_signature checks for Schema & LuaValue contracts
    let mut f_schema = Function::new("schema_fn", 1, false);
    f_schema.set_signature(FunctionSignature {
        params: vec![ty(TypeKind::Schema {
            input: Box::new(ty(TypeKind::String)),
            output: Box::new(ty(TypeKind::Int)),
            fields: None,
        })],
        return_type: ty(TypeKind::Int),
        is_method: false,
        type_params: Vec::new(),
        trait_bounds: Vec::new(),
        effects: None,
    });
    assert!(!f_schema.trusted_signature);
    assert!(!f_schema.compute_trusted_signature());

    let mut f_lua = Function::new("lua_fn", 1, false);
    f_lua.set_signature(FunctionSignature {
        params: vec![ty(TypeKind::Named("LuaValue".to_string()))],
        return_type: ty(TypeKind::Int),
        is_method: false,
        type_params: Vec::new(),
        trait_bounds: Vec::new(),
        effects: None,
    });
    assert!(!f_lua.trusted_signature);
    assert!(!f_lua.compute_trusted_signature());

    let mut f_lua_multi = Function::new("lua_multi_fn", 0, false);
    f_lua_multi.set_signature(FunctionSignature {
        params: Vec::new(),
        return_type: ty(TypeKind::Array(Box::new(ty(TypeKind::Named(
            "LuaValue".to_string(),
        ))))),
        is_method: false,
        type_params: Vec::new(),
        trait_bounds: Vec::new(),
        effects: None,
    });
    assert!(!f_lua_multi.trusted_signature);
    assert!(!f_lua_multi.compute_trusted_signature());

    // Untyped function (signature: None) must default to false
    let f_none = Function::new("untyped", 1, false);
    assert!(!f_none.trusted_signature);
    assert!(!f_none.compute_trusted_signature());

    // MIR backend: static MIR signature -> trusted_signature == true; Unknown/LuaValue -> false
    let mut mir_static = MirFunction::new(
        MirFnSig {
            name: "mir_static".to_string(),
            param_types: vec![MirType::Int, MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        Span::dummy(),
    );
    mir_static.blocks.push(BasicBlock {
        id: BlockId(0),
        statements: Vec::new(),
        terminator: Some(Terminator::Return(
            Some(Operand::Constant(MirConstant::Int(1))),
            Span::dummy(),
        )),
        is_cleanup: false,
    });
    let bc_static = MirToBytecodeCompiler::compile(&mir_static).expect("compile mir_static");
    assert!(bc_static.trusted_signature);

    let mut mir_unknown = MirFunction::new(
        MirFnSig {
            name: "mir_unknown".to_string(),
            param_types: vec![MirType::Unknown],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        Span::dummy(),
    );
    mir_unknown.blocks.push(BasicBlock {
        id: BlockId(0),
        statements: Vec::new(),
        terminator: Some(Terminator::Return(
            Some(Operand::Constant(MirConstant::Int(1))),
            Span::dummy(),
        )),
        is_cleanup: false,
    });
    let bc_unknown = MirToBytecodeCompiler::compile(&mir_unknown).expect("compile mir_unknown");
    assert!(!bc_unknown.trusted_signature);
}

#[test]
fn ng_w11_4b_zero_frame_reloads_on_fib20_calls_and_methods() {
    let source = r#"
        struct Counter
            value: int
        end

        impl Counter
            function bump(self, d: int): int
                self.value = self.value + d
                return self.value
            end
        end

        function fib(n: int): int
            if n < 2 then
                return n
            end
            return fib(n - 1) + fib(n - 2)
        end

        function step(a: int, b: int): int
            return a + b * 2
        end

        function run_calls(n: int): int
            local acc: int = 0
            local i: int = 0
            while i < n do
                acc = step(acc, i)
                i = i + 1
            end
            return acc
        end

        function run_methods(n: int): int
            local c: Counter = Counter { value = 0 }
            local last: int = 0
            local i: int = 0
            while i < n do
                last = c:bump(i)
                i = i + 1
            end
            return last
        end
    "#;

    let mut vm = compile_typed_vm(source);
    vm.set_gc_enabled(false);

    // 1. fib(20): 21,891 total calls, 0 frame-reload exits from 'frame_fast: loop!
    let r0 = VM::frame_hoist_reload_count();
    let fib_res = vm.call("fib", vec![Value::Int(20)]).expect("fib(20)");
    let r1 = VM::frame_hoist_reload_count();
    assert_eq!(fib_res, Value::Int(6765));
    assert_eq!(vm.call_stack_depth(), 0);
    assert_eq!(vm.last_frame_pop_cleanup_iters(), 0);
    #[cfg(debug_assertions)]
    assert_eq!(
        r1 - r0,
        1,
        "fib(20) must execute with 0 exits from 'frame_fast: loop (reload_count == 1), got {}",
        r1 - r0
    );

    // 2. calls.lust workload (N = 10_000): 10,000 internal calls, 0 exits!
    let r2 = VM::frame_hoist_reload_count();
    let calls_res = vm
        .call("run_calls", vec![Value::Int(10_000)])
        .expect("run_calls(10_000)");
    let r3 = VM::frame_hoist_reload_count();
    let expected_calls: i64 = (0..10_000i64).map(|i| i * 2).sum();
    assert_eq!(calls_res, Value::Int(expected_calls));
    assert_eq!(vm.call_stack_depth(), 0);
    #[cfg(debug_assertions)]
    assert_eq!(
        r3 - r2,
        1,
        "calls(10_000) must execute with 0 exits from 'frame_fast: loop (reload_count == 1), got {}",
        r3 - r2
    );

    // 3. methods.lust workload (N = 10_000): 10,000 struct method calls, 0 exits!
    let r4 = VM::frame_hoist_reload_count();
    let methods_res = vm
        .call("run_methods", vec![Value::Int(10_000)])
        .expect("run_methods(10_000)");
    let r5 = VM::frame_hoist_reload_count();
    let expected_methods: i64 = (0..10_000i64).sum();
    assert_eq!(methods_res, Value::Int(expected_methods));
    assert_eq!(vm.call_stack_depth(), 0);
    #[cfg(debug_assertions)]
    assert_eq!(
        r5 - r4,
        1,
        "methods(10_000) must execute with 0 exits from 'frame_fast: loop (reload_count == 1), got {}",
        r5 - r4
    );

    // Verify CallMethod IC slot was populated in run_methods
    let run_methods_fn = vm.function_by_name("run_methods").expect("run_methods");
    let has_method_ic = run_methods_fn
        .chunk
        .ic_slots
        .borrow()
        .iter()
        .any(|slot| matches!(slot, ChunkIcSlot::MethodCall { .. }));
    assert!(
        has_method_ic,
        "run_methods must populate ChunkIcSlot::MethodCall IC in-loop"
    );
}

#[test]
fn ng_w11_4b_sec04_stack_overflow_and_sec01_gas_enforcement_in_loop() {
    let source = r#"
        function inf_rec(n: int): int
            return 1 + inf_rec(n + 1)
        end

        function fib(n: int): int
            if n < 2 then
                return n
            end
            return fib(n - 1) + fib(n - 2)
        end
    "#;

    let mut vm = compile_typed_vm(source);

    // SEC-04: infinite recursion must hit Stack overflow deterministically
    let err = vm
        .call("inf_rec", vec![Value::Int(0)])
        .expect_err("infinite recursion must fail with Stack overflow");
    assert!(
        err.to_string().contains("Stack overflow"),
        "expected Stack overflow error, got: {err}"
    );
    assert_eq!(vm.call_stack_depth(), 0);

    // VM remains healthy after stack overflow
    vm.reset_gas_counter();
    let fib10 = vm
        .call("fib", vec![Value::Int(10)])
        .expect("VM must remain usable after Stack overflow");
    assert_eq!(fib10, Value::Int(55));
    let exact_gas = vm.gas_used();
    assert!(exact_gas > 100, "fib(10) gas must be > 100, got {exact_gas}");

    // SEC-01: gas limit strictly enforced inside in-loop recursive Call/Return
    vm.reset_gas_counter();
    vm.set_gas_budget(exact_gas);
    let fib10_exact = vm
        .call("fib", vec![Value::Int(10)])
        .expect("fib(10) must succeed with exact gas limit");
    assert_eq!(fib10_exact, Value::Int(55));
    assert_eq!(vm.gas_used(), exact_gas);

    vm.reset_gas_counter();
    vm.set_gas_budget(exact_gas - 1);
    let gas_err = vm
        .call("fib", vec![Value::Int(10)])
        .expect_err("fib(10) must fail with exact_gas - 1");
    assert!(
        gas_err.to_string().contains("Out of gas"),
        "expected Out of gas error, got: {gas_err}"
    );
    assert_eq!(vm.call_stack_depth(), 0);
}

#[test]
fn ng_w11_4b_host_and_dynamic_calls_keep_deep_validation() {
    let source = r#"
        function sum_arr(xs: Array<int>): int
            local acc: int = 0
            local i: int = 0
            local n: int = array.len(xs)
            while i < n do
                acc = acc + xs[i].unwrap()
                i = i + 1
            end
            return acc
        end
    "#;

    let mut vm = compile_typed_vm(source);
    let func_idx = vm
        .function_by_name("sum_arr")
        .expect("sum_arr")
        .trusted_signature;
    assert!(func_idx, "sum_arr must have trusted_signature == true");

    // Host `vm.call` must still run ArgCheck::Deep and reject bad container elements!
    let err_call = vm
        .call(
            "sum_arr",
            vec![Value::array(vec![Value::Int(1), Value::Bool(false)])],
        )
        .expect_err("host vm.call must reject Array<bool> for Array<int>");
    assert!(
        err_call.to_string().contains("expects Array<int>"),
        "unexpected error: {err_call}"
    );

    // Host `vm.call_value` must also run ArgCheck::Deep!
    let err_val = vm
        .call_value(
            &Value::Function(0),
            vec![Value::array(vec![Value::Bool(true)])],
        )
        .expect_err("host vm.call_value must reject Array<bool> for Array<int>");
    assert!(
        err_val.to_string().contains("expects Array<int>"),
        "unexpected error: {err_val}"
    );

    // Hand-built function returning wrong type to host still triggers validate_function_return
    let mut bad_ret = Function::new("bad_ret", 0, false);
    bad_ret.set_register_count(1);
    let k_bool = bad_ret.chunk.add_constant(Value::Bool(true));
    bad_ret.chunk.emit(Instruction::LoadConst(0, k_bool), 1);
    bad_ret.chunk.emit(Instruction::Return(0), 1);
    bad_ret.set_signature(FunctionSignature {
        params: Vec::new(),
        return_type: ty(TypeKind::Int),
        is_method: false,
        type_params: Vec::new(),
        trait_bounds: Vec::new(),
        effects: None,
    });
    assert!(bad_ret.trusted_signature);
    let mut vm2 = VM::new();
    vm2.load_functions(vec![bad_ret]);
    let ret_err = vm2
        .call("bad_ret", Vec::new())
        .expect_err("host return must still validate return type");
    assert!(
        ret_err.to_string().contains("must return int"),
        "unexpected return error: {ret_err}"
    );
}

#[test]
fn ng_w11_4b_non_plain_cleanup_polymorphic_ic_and_reentrant_host() {
    let source = r#"
        extern
            function host_cb(x: int): int
        end

        struct BoxA
            v: int
        end

        struct BoxB
            pad: int
            v: int
        end

        impl BoxA
            function read(self, k: int): int
                return self.v + k
            end
        end

        impl BoxB
            function read(self, k: int): int
                return self.v * 2 + k
            end
        end

        function join_str(a: string, b: string): string
            return a .. ":" .. b
        end

        function call_join(s: string): string
            local t: string = join_str(s, "ok")
            return t
        end

        function poly_driver(): int
            local a: BoxA = BoxA { v = 10 }
            local b: BoxB = BoxB { pad = 99, v = 20 }
            local x: int = a:read(1)
            local y: int = b:read(2)
            local z: int = a:read(3)
            return x + y + z
        end

        function helper(n: int): int
            if n <= 0 then
                return 0
            end
            return n + helper(n - 1)
        end

        function reentrant_driver(k: int): int
            local total: int = 0
            local i: int = 1
            while i <= k do
                total = total + host_cb(i)
                i = i + 1
            end
            return total
        end
    "#;

    let mut vm = compile_typed_vm(source);
    vm.set_gc_enabled(false);

    // 1. Non-plain Rc cleanup across in-loop Call/Return
    let rc_s = Rc::new("hello".to_string());
    let joined = vm
        .call("call_join", vec![Value::String(Rc::clone(&rc_s))])
        .expect("call_join");
    assert_eq!(joined, Value::string("hello:ok"));
    assert_eq!(
        Rc::strong_count(&rc_s),
        1,
        "callee non-plain argument Rc must be released on in-loop Return"
    );

    // 2. Polymorphic CallMethod IC in-loop with 0 exits
    let r0 = VM::frame_hoist_reload_count();
    let poly_res = vm.call("poly_driver", Vec::new()).expect("poly_driver");
    let r1 = VM::frame_hoist_reload_count();
    // (10 + 1) + (20 * 2 + 2) + (10 + 3) = 11 + 42 + 13 = 66
    assert_eq!(poly_res, Value::Int(66));
    #[cfg(debug_assertions)]
    assert_eq!(r1 - r0, 1);

    // 3. Re-entrant native callback calling back into VM (tests call_until_depth boundary!)
    let calls_seen = Rc::new(RefCell::new(0usize));
    let calls_seen_cl = Rc::clone(&calls_seen);
    vm.register_native(
        "host_cb",
        Value::NativeFunction(native_fn(move |args| {
            let n = args.first().and_then(|v| v.as_int()).unwrap_or(0);
            *calls_seen_cl.borrow_mut() += 1;
            VM::with_current(|cur_vm| {
                let inner = cur_vm
                    .call("helper", vec![Value::Int(n + 5)])
                    .map_err(|e| e.to_string())?;
                Ok(NativeCallResult::Return(inner))
            })
        })),
    );
    let reent = vm
        .call("reentrant_driver", vec![Value::Int(4)])
        .expect("reentrant_driver");
    let expected_reent: i64 = (1..=4i64).map(|i| (i + 5) * (i + 6) / 2).sum();
    assert_eq!(reent, Value::Int(expected_reent));
    assert_eq!(*calls_seen.borrow(), 4);
}

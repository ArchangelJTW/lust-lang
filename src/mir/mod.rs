pub mod ir;
pub mod lowering;
pub mod opt;
pub mod verifier;
pub mod vm_backend;
#[cfg(feature = "std")]
pub mod cranelift_backend;
#[cfg(feature = "std")]
pub mod zig_backend;

pub use ir::*;
pub use lowering::*;
pub use opt::*;
pub use verifier::*;
pub use vm_backend::*;
#[cfg(feature = "std")]
pub use cranelift_backend::*;
#[cfg(feature = "std")]
pub use zig_backend::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Span;

    #[test]
    fn test_mir_verifier_accepts_valid_function() {
        let sig = FunctionSignature {
            name: "add_one".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
        };

        let mut func = MirFunction::new(sig, Span::dummy());
        let arg0 = func.params[0];
        let temp = func.add_local(Some("temp".into()), MirType::Int, true, Span::dummy());

        // bb0:
        //   StorageLive(_2);
        //   _2 = copy _1 + const 1;
        //   _0 = copy _2;
        //   return copy _0;
        let bb0 = &mut func.blocks[0];
        bb0.statements.push(Statement::StorageLive(temp, Span::dummy()));
        bb0.statements.push(Statement::Assign(
            Place::Local(temp),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(arg0)),
                Operand::Constant(MirConstant::Int(1)),
            ),
            Span::dummy(),
        ));
        bb0.statements.push(Statement::Assign(
            Place::Local(func.return_local),
            Rvalue::Use(Operand::Copy(Place::Local(temp))),
            Span::dummy(),
        ));
        bb0.terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(func.return_local))),
            Span::dummy(),
        ));

        let mut module = MirModule::new("test_mod".into());
        module.add_function(func);

        let result = MirVerifier::verify_module(&module);
        assert!(result.is_ok(), "Verification failed: {:?}", result.err());
    }

    #[test]
    fn test_mir_verifier_rejects_missing_terminator() {
        let sig = FunctionSignature {
            name: "broken_fn".into(),
            param_types: vec![],
            return_type: MirType::Unit,
            is_public: true,
        };

        let mut func = MirFunction::new(sig, Span::dummy());
        func.blocks[0].terminator = None; // Missing terminator

        let mut module = MirModule::new("test_mod".into());
        module.add_function(func);

        let result = MirVerifier::verify_module(&module);
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(errs.iter().any(|e| e.message.contains("missing terminator")));
    }

    #[test]
    fn test_mir_verifier_rejects_out_of_bounds_target() {
        let sig = FunctionSignature {
            name: "oob_fn".into(),
            param_types: vec![],
            return_type: MirType::Unit,
            is_public: true,
        };

        let mut func = MirFunction::new(sig, Span::dummy());
        func.blocks[0].terminator = Some(Terminator::Goto(BlockId(999), Span::dummy()));

        let mut module = MirModule::new("test_mod".into());
        module.add_function(func);

        let result = MirVerifier::verify_module(&module);
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(errs.iter().any(|e| e.message.contains("out of bounds")));
    }

    #[test]
    fn test_mir_verifier_rejects_uninitialized_local_read() {
        let sig = FunctionSignature {
            name: "uninit_fn".into(),
            param_types: vec![],
            return_type: MirType::Int,
            is_public: true,
        };

        let mut func = MirFunction::new(sig, Span::dummy());
        let uninit = func.add_local(Some("uninit".into()), MirType::Int, true, Span::dummy());

        // bb0:
        //   _0 = copy _1;  // Error: _1 uninitialized!
        //   return copy _0;
        let bb0 = &mut func.blocks[0];
        bb0.statements.push(Statement::Assign(
            Place::Local(func.return_local),
            Rvalue::Use(Operand::Copy(Place::Local(uninit))),
            Span::dummy(),
        ));
        bb0.terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(func.return_local))),
            Span::dummy(),
        ));

        let mut module = MirModule::new("test_mod".into());
        module.add_function(func);

        let result = MirVerifier::verify_module(&module);
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(errs.iter().any(|e| e.message.contains("uninitialized local")));
    }

    #[test]
    fn test_ast_to_mir_lowering_and_verification() {
        use crate::ast::ItemKind;
        use crate::lexer::Lexer;
        use crate::parser::Parser;

        let src = r#"
            function compute(x: int, y: int): int
                local acc: int = x
                while y > 0 do
                    acc = acc + 10
                    y = y - 1
                end
                return acc
            end
        "#;

        use crate::intern::Interner;
        let mut interner = Interner::new();
        let mut lexer = Lexer::new(src, &mut interner);
        let mut parser = Parser::from_lexer(&mut lexer).expect("lexer");
        let items = parser.parse().expect("parse");

        let mut module = MirModule::new("test_compute".into());
        for item in items {
            if let ItemKind::Function(func_def) = item.kind {
                let mir_fn = AstLowering::lower_function(&func_def);
                module.add_function(mir_fn);
            }
        }

        assert_eq!(module.functions.len(), 1);
        let mir_fn = &module.functions[0];
        assert_eq!(mir_fn.signature.name, "compute");
        assert_eq!(mir_fn.params.len(), 2);
        assert!(mir_fn.blocks.len() >= 3); // entry, header, body, exit

        let verify_res = MirVerifier::verify_module(&module);
        assert!(verify_res.is_ok(), "Verification failed: {:?}", verify_res.err());
    }

    #[test]
    fn test_mir_to_vm_bytecode_execution() {
        use crate::ast::ItemKind;
        use crate::bytecode::Value;
        use crate::intern::Interner;
        use crate::lexer::Lexer;
        use crate::parser::Parser;
        use crate::vm::VM;

        let src = r#"
            function compute(x: int, y: int): int
                local acc: int = x
                while y > 0 do
                    acc = acc + 10
                    y = y - 1
                end
                return acc
            end
        "#;

        let mut interner = Interner::new();
        let mut lexer = Lexer::new(src, &mut interner);
        let mut parser = Parser::from_lexer(&mut lexer).expect("lexer");
        let items = parser.parse().expect("parse");

        let mut mir_fn = None;
        for item in items {
            if let ItemKind::Function(func_def) = item.kind {
                mir_fn = Some(AstLowering::lower_function(&func_def));
                break;
            }
        }

        let mir_fn = mir_fn.expect("Function parsed and lowered");
        let mut module = MirModule::new("test_mod".into());
        module.add_function(mir_fn.clone());
        assert!(MirVerifier::verify_module(&module).is_ok());

        // Compile MIR to Bytecode Function
        let bytecode_fn = MirToBytecodeCompiler::compile(&mir_fn);

        // Execute via Lust VM
        let mut vm = VM::new();
        vm.load_functions(vec![bytecode_fn]);

        let result = vm.call("compute", vec![Value::Int(5), Value::Int(3)]).expect("VM call succeeds");
        assert_eq!(result, Value::Int(35));
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_mir_to_cranelift_jit_execution() {
        use crate::ast::ItemKind;
        use crate::intern::Interner;
        use crate::lexer::Lexer;
        use crate::mir::cranelift_backend::jit_engine::CraneliftJitCompiler;
        use crate::parser::Parser;

        let src = r#"
            function compute(x: int, y: int): int
                local acc: int = x
                while y > 0 do
                    acc = acc + 10
                    y = y - 1
                end
                return acc
            end
        "#;

        let mut interner = Interner::new();
        let mut lexer = Lexer::new(src, &mut interner);
        let mut parser = Parser::from_lexer(&mut lexer).expect("lexer");
        let items = parser.parse().expect("parse");

        let mut mir_fn = None;
        for item in items {
            if let ItemKind::Function(func_def) = item.kind {
                mir_fn = Some(AstLowering::lower_function(&func_def));
                break;
            }
        }

        let mir_fn = mir_fn.expect("Function parsed and lowered");
        let mut module = MirModule::new("test_jit_mod".into());
        module.add_function(mir_fn.clone());
        assert!(MirVerifier::verify_module(&module).is_ok());

        // Compile MIR to native code using Cranelift JIT
        let mut jit = CraneliftJitCompiler::new().expect("JIT initialization");
        let code_ptr = jit.compile_function(&mir_fn).expect("Cranelift compilation");

        // Cast to native function pointer and execute directly on CPU
        let native_compute: extern "C" fn(i64, i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
        let result = native_compute(5, 3);
        assert_eq!(result, 35);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_mir_to_zig_aot_execution() {
        use crate::ast::ItemKind;
        use crate::intern::Interner;
        use crate::lexer::Lexer;
        use crate::mir::zig_backend::zig_engine::MirToZigCompiler;
        use crate::parser::Parser;

        let src = r#"
            function compute(x: int, y: int): int
                local acc: int = x
                while y > 0 do
                    acc = acc + 10
                    y = y - 1
                end
                return acc
            end
        "#;

        let mut interner = Interner::new();
        let mut lexer = Lexer::new(src, &mut interner);
        let mut parser = Parser::from_lexer(&mut lexer).expect("lexer");
        let items = parser.parse().expect("parse");

        let mut mir_fn = None;
        for item in items {
            if let ItemKind::Function(func_def) = item.kind {
                mir_fn = Some(AstLowering::lower_function(&func_def));
                break;
            }
        }

        let mir_fn = mir_fn.expect("Function parsed and lowered");
        let mut module = MirModule::new("test_aot_mod".into());
        module.add_function(mir_fn.clone());
        assert!(MirVerifier::verify_module(&module).is_ok());

        // Compile MIR to native dylib using Zig 0.17.1
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let dylib_path = MirToZigCompiler::compile_to_dylib(&mir_fn, temp_dir.path()).expect("Zig compilation");

        // Load dylib via C ABI and execute
        unsafe {
            let lib = libloading::Library::new(&dylib_path).expect("Load dylib");
            let func: libloading::Symbol<extern "C" fn(i64, i64) -> i64> =
                lib.get(b"compute\0").expect("Find compute symbol");
            let result = func(5, 3);
            assert_eq!(result, 35);
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_gate2_for_numeric_break_continue_parity_across_all_backends() {
        use crate::ast::ItemKind;
        use crate::bytecode::Value;
        use crate::intern::Interner;
        use crate::lexer::Lexer;
        use crate::mir::cranelift_backend::jit_engine::CraneliftJitCompiler;
        use crate::mir::zig_backend::zig_engine::MirToZigCompiler;
        use crate::parser::Parser;
        use crate::vm::VM;

        let src = r#"
            function loop_test(limit: int): int
                local total: int = 0
                for i = 1, limit do
                    if i == 5 then
                        continue
                    end
                    if i > 8 then
                        break
                    end
                    total = total + i
                end
                return total
            end
        "#;

        let mut interner = Interner::new();
        let mut lexer = Lexer::new(src, &mut interner);
        let mut parser = Parser::from_lexer(&mut lexer).expect("lexer");
        let items = parser.parse().expect("parse");

        let mut mir_fn = None;
        for item in items {
            if let ItemKind::Function(func_def) = item.kind {
                mir_fn = Some(AstLowering::lower_function(&func_def));
                break;
            }
        }

        let mir_fn = mir_fn.expect("Function parsed and lowered");
        let mut module = MirModule::new("test_gate2_mod".into());
        module.add_function(mir_fn.clone());
        let v_res = MirVerifier::verify_module(&module);
        assert!(v_res.is_ok(), "Verification failed: {:?}", v_res.err());

        // 1. VM Bytecode Execution
        let bytecode_fn = MirToBytecodeCompiler::compile(&mir_fn);
        let mut vm = VM::new();
        vm.load_functions(vec![bytecode_fn]);
        let vm_res = vm.call("loop_test", vec![Value::Int(10)]).expect("VM execution");
        assert_eq!(vm_res, Value::Int(31));

        // 2. Cranelift JIT Execution
        let mut jit = CraneliftJitCompiler::new().expect("JIT initialization");
        let code_ptr = jit.compile_function(&mir_fn).expect("JIT compile");
        let jit_func: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
        let jit_res = jit_func(10);
        assert_eq!(jit_res, 31);

        // 3. Zig AOT Execution
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let dylib_path = MirToZigCompiler::compile_to_dylib(&mir_fn, temp_dir.path()).expect("Zig compilation");
        unsafe {
            let lib = libloading::Library::new(&dylib_path).expect("Load dylib");
            let aot_func: libloading::Symbol<extern "C" fn(i64) -> i64> =
                lib.get(b"loop_test\0").expect("Find symbol");
            let aot_res = aot_func(10);
            assert_eq!(aot_res, 31);
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_gate2_constant_folding_dce_and_execution() {
        use crate::ast::ItemKind;
        use crate::bytecode::Value;
        use crate::intern::Interner;
        use crate::lexer::Lexer;
        use crate::mir::cranelift_backend::jit_engine::CraneliftJitCompiler;
        use crate::mir::zig_backend::zig_engine::MirToZigCompiler;
        use crate::parser::Parser;
        use crate::vm::VM;

        let src = r#"
            function const_test(x: int): int
                local a: int = 2 + 3 * 4
                if 10 > 5 then
                    return a + x
                else
                    return 0
                end
            end
        "#;

        let mut interner = Interner::new();
        let mut lexer = Lexer::new(src, &mut interner);
        let mut parser = Parser::from_lexer(&mut lexer).expect("lexer");
        let items = parser.parse().expect("parse");

        let mut mir_fn = None;
        for item in items {
            if let ItemKind::Function(func_def) = item.kind {
                mir_fn = Some(AstLowering::lower_function(&func_def));
                break;
            }
        }

        let mut mir_fn = mir_fn.expect("Function parsed and lowered");
        let initial_blocks = mir_fn.blocks.len();

        // Run optimizations: Constant Folding + Dead Code Elimination
        MirOptimizer::optimize_function(&mut mir_fn);

        // DCE should have eliminated the dead `else` block
        assert!(mir_fn.blocks.len() < initial_blocks, "DCE should reduce block count");

        let mut module = MirModule::new("test_opt_mod".into());
        module.add_function(mir_fn.clone());
        assert!(MirVerifier::verify_module(&module).is_ok());

        // Differential verification with x = 6 -> a (14) + 6 = 20
        // VM
        let bytecode_fn = MirToBytecodeCompiler::compile(&mir_fn);
        let mut vm = VM::new();
        vm.load_functions(vec![bytecode_fn]);
        let vm_res = vm.call("const_test", vec![Value::Int(6)]).expect("VM run");
        assert_eq!(vm_res, Value::Int(20));

        // Cranelift JIT
        let mut jit = CraneliftJitCompiler::new().expect("JIT init");
        let code_ptr = jit.compile_function(&mir_fn).expect("JIT compile");
        let jit_func: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
        let jit_res = jit_func(6);
        assert_eq!(jit_res, 20);

        // Zig AOT
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let dylib_path = MirToZigCompiler::compile_to_dylib(&mir_fn, temp_dir.path()).expect("Zig compile");
        unsafe {
            let lib = libloading::Library::new(&dylib_path).expect("Load dylib");
            let aot_func: libloading::Symbol<extern "C" fn(i64) -> i64> =
                lib.get(b"const_test\0").expect("Find symbol");
            let aot_res = aot_func(6);
            assert_eq!(aot_res, 20);
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_gate2_short_circuit_protection_across_all_backends() {
        use crate::ast::ItemKind;
        use crate::bytecode::Value;
        use crate::intern::Interner;
        use crate::lexer::Lexer;
        use crate::mir::cranelift_backend::jit_engine::CraneliftJitCompiler;
        use crate::mir::zig_backend::zig_engine::MirToZigCompiler;
        use crate::parser::Parser;
        use crate::vm::VM;

        let src = r#"
            function safe_check(d: int): int
                if d != 0 and 100 / d > 5 then
                    return 1
                else
                    return 0
                end
            end
        "#;

        let mut interner = Interner::new();
        let mut lexer = Lexer::new(src, &mut interner);
        let mut parser = Parser::from_lexer(&mut lexer).expect("lexer");
        let items = parser.parse().expect("parse");

        let mut mir_fn = None;
        for item in items {
            if let ItemKind::Function(func_def) = item.kind {
                mir_fn = Some(AstLowering::lower_function(&func_def));
                break;
            }
        }

        let mir_fn = mir_fn.expect("Function parsed and lowered");
        let mut module = MirModule::new("test_sc_mod".into());
        module.add_function(mir_fn.clone());
        assert!(MirVerifier::verify_module(&module).is_ok());

        // Test with d = 0 (must NOT divide by zero, returns 0)
        // 1. VM Bytecode Execution
        let bytecode_fn = MirToBytecodeCompiler::compile(&mir_fn);
        let mut vm = VM::new();
        vm.load_functions(vec![bytecode_fn.clone()]);
        let vm_res = vm.call("safe_check", vec![Value::Int(0)]).expect("VM execution");
        assert_eq!(vm_res, Value::Int(0));

        // 2. Cranelift JIT Execution
        let mut jit = CraneliftJitCompiler::new().expect("JIT initialization");
        let code_ptr = jit.compile_function(&mir_fn).expect("JIT compile");
        let jit_func: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
        let jit_res = jit_func(0);
        assert_eq!(jit_res, 0);

        // 3. Zig AOT Execution
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let dylib_path = MirToZigCompiler::compile_to_dylib(&mir_fn, temp_dir.path()).expect("Zig compilation");
        unsafe {
            let lib = libloading::Library::new(&dylib_path).expect("Load dylib");
            let aot_func: libloading::Symbol<extern "C" fn(i64) -> i64> =
                lib.get(b"safe_check\0").expect("Find symbol");
            let aot_res = aot_func(0);
            assert_eq!(aot_res, 0);
        }

        // Also test with d = 10 (100 / 10 = 10 > 5 -> returns 1)
        let vm_res_10 = vm.call("safe_check", vec![Value::Int(10)]).expect("VM execution");
        assert_eq!(vm_res_10, Value::Int(1));
        assert_eq!(jit_func(10), 1);
        unsafe {
            let lib = libloading::Library::new(&dylib_path).expect("Load dylib");
            let aot_func: libloading::Symbol<extern "C" fn(i64) -> i64> =
                lib.get(b"safe_check\0").expect("Find symbol");
            assert_eq!(aot_func(10), 1);
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_gate3_arrays_and_aggregates_parity_across_all_backends() {
        use crate::ast::ItemKind;
        use crate::bytecode::Value;
        use crate::intern::Interner;
        use crate::lexer::Lexer;
        use crate::mir::cranelift_backend::jit_engine::CraneliftJitCompiler;
        use crate::mir::zig_backend::zig_engine::MirToZigCompiler;
        use crate::parser::Parser;
        use crate::vm::VM;

        let src = r#"
            function array_ops(n: int): int
                local arr = [10, 20, 30]
                arr[1] = n
                local len = arr.len()
                local sum: int = 0
                for i = 0, len - 1 do
                    sum = sum + arr[i]
                end
                return sum
            end
        "#;

        let mut interner = Interner::new();
        let mut lexer = Lexer::new(src, &mut interner);
        let mut parser = Parser::from_lexer(&mut lexer).expect("lexer");
        let items = parser.parse().expect("parse");

        let mut mir_fn = None;
        for item in items {
            if let ItemKind::Function(func_def) = item.kind {
                mir_fn = Some(AstLowering::lower_function(&func_def));
                break;
            }
        }

        let mir_fn = mir_fn.expect("Function parsed and lowered");
        let mut module = MirModule::new("test_array_mod".into());
        module.add_function(mir_fn.clone());
        assert!(MirVerifier::verify_module(&module).is_ok());

        // Array [10, 20, 30], arr[1] = 42 -> [10, 42, 30], sum is 10 + 42 + 30 = 82
        // 1. VM Bytecode Execution
        let bytecode_fn = MirToBytecodeCompiler::compile(&mir_fn);
        let mut vm = VM::new();
        vm.load_functions(vec![bytecode_fn]);
        let vm_res = vm.call("array_ops", vec![Value::Int(42)]).expect("VM execution");
        assert_eq!(vm_res, Value::Int(82));

        // 2. Cranelift JIT Execution
        let mut jit = CraneliftJitCompiler::new().expect("JIT initialization");
        let code_ptr = jit.compile_function(&mir_fn).expect("JIT compile");
        let jit_func: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
        let jit_res = jit_func(42);
        assert_eq!(jit_res, 82);

        // 3. Zig AOT Execution
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let dylib_path = MirToZigCompiler::compile_to_dylib(&mir_fn, temp_dir.path()).expect("Zig compilation");
        unsafe {
            let lib = libloading::Library::new(&dylib_path).expect("Load dylib");
            let aot_func: libloading::Symbol<extern "C" fn(i64) -> i64> =
                lib.get(b"array_ops\0").expect("Find symbol");
            let aot_res = aot_func(42);
            assert_eq!(aot_res, 82);
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_gate4_gas_resource_limits_parity_across_all_backends() {
        use crate::ast::ItemKind;
        use crate::bytecode::Value;
        use crate::intern::Interner;
        use crate::lexer::Lexer;
        use crate::mir::cranelift_backend::jit_engine::{CraneliftJitCompiler, lust_mir_set_gas};
        use crate::mir::zig_backend::zig_engine::MirToZigCompiler;
        use crate::parser::Parser;
        use crate::vm::VM;

        let src = r#"
            function bounded_loop(n: int): int
                local count: int = 0
                while count < n do
                    count = count + 1
                end
                return count
            end
        "#;

        let mut interner = Interner::new();
        let mut lexer = Lexer::new(src, &mut interner);
        let mut parser = Parser::from_lexer(&mut lexer).expect("lexer");
        let items = parser.parse().expect("parse");

        let mut mir_fn = None;
        for item in items {
            if let ItemKind::Function(func_def) = item.kind {
                mir_fn = Some(AstLowering::lower_function(&func_def));
                break;
            }
        }

        let mir_fn = mir_fn.expect("Function parsed and lowered");
        let mut module = MirModule::new("test_gas_mod".into());
        module.add_function(mir_fn.clone());
        assert!(MirVerifier::verify_module(&module).is_ok());

        // 1. VM Execution
        let bytecode_fn = MirToBytecodeCompiler::compile(&mir_fn);
        let mut vm = VM::new();
        vm.load_functions(vec![bytecode_fn]);
        let vm_res = vm.call("bounded_loop", vec![Value::Int(10)]).expect("VM execution");
        assert_eq!(vm_res, Value::Int(10));

        // 2. Cranelift JIT Execution
        let mut jit = CraneliftJitCompiler::new().expect("JIT initialization");
        let code_ptr = jit.compile_function(&mir_fn).expect("JIT compile");
        let jit_func: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
        lust_mir_set_gas(1_000_000);
        let jit_res = jit_func(10);
        assert_eq!(jit_res, 10);
        // The while loop checked gas 11 times (10 iterations + 1 terminating condition)
        assert_eq!(crate::mir::cranelift_backend::jit_engine::lust_mir_get_gas(), 1_000_000 - 11);

        // 3. Zig AOT Execution
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let dylib_path = MirToZigCompiler::compile_to_dylib(&mir_fn, temp_dir.path()).expect("Zig compilation");
        unsafe {
            let lib = libloading::Library::new(&dylib_path).expect("Load dylib");
            let aot_func: libloading::Symbol<extern "C" fn(i64) -> i64> =
                lib.get(b"bounded_loop\0").expect("Find symbol");
            let set_gas: libloading::Symbol<extern "C" fn(i64)> =
                lib.get(b"lust_set_gas_limit\0").expect("Find gas symbol");
            let get_gas: libloading::Symbol<extern "C" fn() -> i64> =
                lib.get(b"lust_get_gas_remaining\0").expect("Find get_gas symbol");

            set_gas(1_000_000);
            let aot_res = aot_func(10);
            assert_eq!(aot_res, 10);
            assert_eq!(get_gas(), 1_000_000 - 11);
        }
    }
}

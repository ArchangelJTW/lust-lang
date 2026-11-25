pub mod ir;
pub mod lowering;
pub mod opt;
pub mod verifier;
pub mod vm_backend;
#[cfg(feature = "std")]
pub mod c_backend;
#[cfg(feature = "std")]
pub mod c_header_gen;
#[cfg(feature = "cranelift")]
pub mod cranelift_backend;
#[cfg(feature = "std")]
pub mod runtime_helpers;
#[cfg(feature = "std")]
pub mod zig_backend;

pub use ir::*;
pub use lowering::*;
pub use opt::*;
pub use verifier::*;
pub use vm_backend::*;
#[cfg(feature = "std")]
pub use c_backend::*;
#[cfg(feature = "std")]
pub use c_header_gen::*;
#[cfg(feature = "cranelift")]
pub use cranelift_backend::*;
#[cfg(feature = "std")]
pub use runtime_helpers::*;
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
            effects: None,
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
            effects: None,
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
            effects: None,
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
            effects: None,
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
                let mir_fn = AstLowering::lower_function(&func_def).expect("lower function");
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
                mir_fn = Some(AstLowering::lower_function(&func_def).expect("lower function"));
                break;
            }
        }

        let mir_fn = mir_fn.expect("Function parsed and lowered");
        let mut module = MirModule::new("test_mod".into());
        module.add_function(mir_fn.clone());
        assert!(MirVerifier::verify_module(&module).is_ok());

        // Compile MIR to Bytecode Function
        let bytecode_fn = MirToBytecodeCompiler::compile(&mir_fn).expect("compile bytecode");

        // Execute via Lust VM
        let mut vm = VM::new();
        vm.load_functions(vec![bytecode_fn]);

        let result = vm.call("compute", vec![Value::Int(5), Value::Int(3)]).expect("VM call succeeds");
        assert_eq!(result, Value::Int(35));
    }

    #[test]
    #[cfg(feature = "cranelift")]
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
                mir_fn = Some(AstLowering::lower_function(&func_def).expect("lower function"));
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
                mir_fn = Some(AstLowering::lower_function(&func_def).expect("lower function"));
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
    #[cfg(feature = "cranelift")]
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
                mir_fn = Some(AstLowering::lower_function(&func_def).expect("lower function"));
                break;
            }
        }

        let mir_fn = mir_fn.expect("Function parsed and lowered");
        let mut module = MirModule::new("test_gate2_mod".into());
        module.add_function(mir_fn.clone());
        let v_res = MirVerifier::verify_module(&module);
        assert!(v_res.is_ok(), "Verification failed: {:?}", v_res.err());

        // 1. VM Bytecode Execution
        let bytecode_fn = MirToBytecodeCompiler::compile(&mir_fn).expect("compile bytecode");
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
    #[cfg(feature = "cranelift")]
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
                mir_fn = Some(AstLowering::lower_function(&func_def).expect("lower function"));
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
        let bytecode_fn = MirToBytecodeCompiler::compile(&mir_fn).expect("compile bytecode");
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
    #[cfg(feature = "cranelift")]
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
                mir_fn = Some(AstLowering::lower_function(&func_def).expect("lower function"));
                break;
            }
        }

        let mir_fn = mir_fn.expect("Function parsed and lowered");
        let mut module = MirModule::new("test_sc_mod".into());
        module.add_function(mir_fn.clone());
        assert!(MirVerifier::verify_module(&module).is_ok());

        // Test with d = 0 (must NOT divide by zero, returns 0)
        // 1. VM Bytecode Execution
        let bytecode_fn = MirToBytecodeCompiler::compile(&mir_fn).expect("compile bytecode");
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
    #[cfg(feature = "cranelift")]
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
                mir_fn = Some(AstLowering::lower_function(&func_def).expect("lower function"));
                break;
            }
        }

        let mir_fn = mir_fn.expect("Function parsed and lowered");
        let mut module = MirModule::new("test_array_mod".into());
        module.add_function(mir_fn.clone());
        assert!(MirVerifier::verify_module(&module).is_ok());

        // Array [10, 20, 30], arr[1] = 42 -> [10, 42, 30], sum is 10 + 42 + 30 = 82
        // 1. VM Bytecode Execution
        let bytecode_fn = MirToBytecodeCompiler::compile(&mir_fn).expect("compile bytecode");
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
    #[cfg(feature = "cranelift")]
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
                mir_fn = Some(AstLowering::lower_function(&func_def).expect("lower function"));
                break;
            }
        }

        let mir_fn = mir_fn.expect("Function parsed and lowered");
        let mut module = MirModule::new("test_gas_mod".into());
        module.add_function(mir_fn.clone());
        assert!(MirVerifier::verify_module(&module).is_ok());

        // 1. VM Execution
        let bytecode_fn = MirToBytecodeCompiler::compile(&mir_fn).expect("compile bytecode");
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

    #[test]
    fn test_unsupported_ast_binop_returns_error_in_lowering() {
        use crate::ast::BinaryOp;
        let res = mir_binop_from_ast(BinaryOp::Concat);
        assert!(matches!(res, Err(MirLowerError::UnsupportedBinaryOp(BinaryOp::Concat))));
    }

    #[test]
    fn test_mir_math_and_bitwise_in_vm_backend() {
        use crate::bytecode::Value;
        use crate::vm::VM;

        let ops = [
            (MirBinaryOp::Pow, 3, 4, 81),
            (MirBinaryOp::FloorDiv, -7, 2, -4),
            (MirBinaryOp::BitAnd, 13, 11, 9),
            (MirBinaryOp::BitOr, 12, 3, 15),
            (MirBinaryOp::BitXor, 12, 10, 6),
            (MirBinaryOp::Shl, 5, 3, 40),
            (MirBinaryOp::Shr, -16, 2, -4),
        ];

        for (op, a, b, expected) in ops {
            let fn_name = format!("vm_op_{op:?}").to_lowercase();
            let sig = FunctionSignature {
                name: fn_name.clone(),
                param_types: vec![MirType::Int, MirType::Int],
                return_type: MirType::Int,
                is_public: true,
            effects: None,
            };
            let mut func = MirFunction::new(sig, Span::dummy());
            let arg0 = func.params[0];
            let arg1 = func.params[1];
            let temp = func.add_local(Some("temp".into()), MirType::Int, true, Span::dummy());
            let bb0 = &mut func.blocks[0];
            bb0.statements.push(Statement::Assign(
                Place::Local(temp),
                Rvalue::BinaryOp(
                    op,
                    Operand::Copy(Place::Local(arg0)),
                    Operand::Copy(Place::Local(arg1)),
                ),
                Span::dummy(),
            ));
            bb0.terminator = Some(Terminator::Return(
                Some(Operand::Copy(Place::Local(temp))),
                Span::dummy(),
            ));

            let bc_fn = MirToBytecodeCompiler::compile(&func)
                .unwrap_or_else(|e| panic!("VM MIR compilation failed for {op:?}: {e}"));
            let mut vm = VM::new();
            vm.load_functions(vec![bc_fn]);
            let res = vm
                .call(&fn_name, vec![Value::Int(a), Value::Int(b)])
                .expect("VM call");
            assert_eq!(res, Value::Int(expected), "Mismatch for VM MIR op {op:?}");
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_unsupported_mir_binop_returns_error_in_zig_backend() {
        use crate::mir::zig_backend::zig_engine::MirToZigCompiler;

        // Verify that Pow, FloorDiv, BitAnd, BitOr, BitXor, Shl, Shr, BitNot now succeed in Zig backend
        let ops = [
            (MirBinaryOp::Pow, 3, 4, 81),
            (MirBinaryOp::FloorDiv, 27, 4, 6),
            (MirBinaryOp::BitAnd, 13, 11, 9),
            (MirBinaryOp::BitOr, 12, 3, 15),
            (MirBinaryOp::BitXor, 12, 10, 6),
            (MirBinaryOp::Shl, 5, 3, 40),
            (MirBinaryOp::Shr, 40, 2, 10),
        ];

        let temp_dir = tempfile::tempdir().expect("tempdir");

        for (op, a, b, expected) in ops {
            let sig = FunctionSignature {
                name: format!("zig_op_{op:?}").to_lowercase(),
                param_types: vec![MirType::Int, MirType::Int],
                return_type: MirType::Int,
                is_public: true,
            effects: None,
            };
            let mut func = MirFunction::new(sig, Span::dummy());
            let arg0 = func.params[0];
            let arg1 = func.params[1];
            let temp = func.add_local(Some("temp".into()), MirType::Int, true, Span::dummy());
            let bb0 = &mut func.blocks[0];
            bb0.statements.push(Statement::Assign(
                Place::Local(temp),
                Rvalue::BinaryOp(
                    op,
                    Operand::Copy(Place::Local(arg0)),
                    Operand::Copy(Place::Local(arg1)),
                ),
                Span::dummy(),
            ));
            bb0.terminator = Some(Terminator::Return(
                Some(Operand::Copy(Place::Local(temp))),
                Span::dummy(),
            ));

            let dylib_path = MirToZigCompiler::compile_to_dylib(&func, temp_dir.path())
                .unwrap_or_else(|e| panic!("Zig compilation failed for {op:?}: {e}"));
            unsafe {
                let lib = libloading::Library::new(&dylib_path).expect("Load dylib");
                let sym_name = format!("zig_op_{op:?}\0").to_lowercase();
                let f: libloading::Symbol<extern "C" fn(i64, i64) -> i64> =
                    lib.get(sym_name.as_bytes()).expect("Find symbol");
                assert_eq!(f(a, b), expected, "Mismatch for Zig op {op:?}");
            }
        }

        // Test BitNot
        let sig = FunctionSignature {
            name: "zig_op_bitnot".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut func = MirFunction::new(sig, Span::dummy());
        let arg0 = func.params[0];
        let temp = func.add_local(Some("temp".into()), MirType::Int, true, Span::dummy());
        let bb0 = &mut func.blocks[0];
        bb0.statements.push(Statement::Assign(
            Place::Local(temp),
            Rvalue::UnaryOp(MirUnaryOp::BitNot, Operand::Copy(Place::Local(arg0))),
            Span::dummy(),
        ));
        bb0.terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(temp))),
            Span::dummy(),
        ));
        let dylib_path = MirToZigCompiler::compile_to_dylib(&func, temp_dir.path()).expect("Zig compile BitNot");
        unsafe {
            let lib = libloading::Library::new(&dylib_path).expect("Load dylib");
            let f: libloading::Symbol<extern "C" fn(i64) -> i64> = lib.get(b"zig_op_bitnot\0").expect("Find symbol");
            assert_eq!(f(42), !42);
        }
    }

    #[test]
    #[cfg(feature = "cranelift")]
    fn test_unsupported_mir_binop_returns_error_in_cranelift_backend() {
        use crate::mir::cranelift_backend::jit_engine::CraneliftJitCompiler;

        let ops = [
            (MirBinaryOp::Pow, 3, 4, 81),
            (MirBinaryOp::FloorDiv, 27, 4, 6),
            (MirBinaryOp::BitAnd, 13, 11, 9),
            (MirBinaryOp::BitOr, 12, 3, 15),
            (MirBinaryOp::BitXor, 12, 10, 6),
            (MirBinaryOp::Shl, 5, 3, 40),
            (MirBinaryOp::Shr, 40, 2, 10),
        ];

        let mut jit = CraneliftJitCompiler::new().expect("JIT initialization");

        for (op, a, b, expected) in ops {
            let sig = FunctionSignature {
                name: format!("clif_op_{op:?}").to_lowercase(),
                param_types: vec![MirType::Int, MirType::Int],
                return_type: MirType::Int,
                is_public: true,
            effects: None,
            };
            let mut func = MirFunction::new(sig, Span::dummy());
            let arg0 = func.params[0];
            let arg1 = func.params[1];
            let temp = func.add_local(Some("temp".into()), MirType::Int, true, Span::dummy());
            let bb0 = &mut func.blocks[0];
            bb0.statements.push(Statement::Assign(
                Place::Local(temp),
                Rvalue::BinaryOp(
                    op,
                    Operand::Copy(Place::Local(arg0)),
                    Operand::Copy(Place::Local(arg1)),
                ),
                Span::dummy(),
            ));
            bb0.terminator = Some(Terminator::Return(
                Some(Operand::Copy(Place::Local(temp))),
                Span::dummy(),
            ));

            let code_ptr = jit.compile_function(&func)
                .unwrap_or_else(|e| panic!("Cranelift compile failed for {op:?}: {e}"));
            let f: extern "C" fn(i64, i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
            assert_eq!(f(a, b), expected, "Mismatch for Cranelift op {op:?}");
        }

        // Test BitNot
        let sig = FunctionSignature {
            name: "clif_op_bitnot".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut func = MirFunction::new(sig, Span::dummy());
        let arg0 = func.params[0];
        let temp = func.add_local(Some("temp".into()), MirType::Int, true, Span::dummy());
        let bb0 = &mut func.blocks[0];
        bb0.statements.push(Statement::Assign(
            Place::Local(temp),
            Rvalue::UnaryOp(MirUnaryOp::BitNot, Operand::Copy(Place::Local(arg0))),
            Span::dummy(),
        ));
        bb0.terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(temp))),
            Span::dummy(),
        ));
        let code_ptr = jit.compile_function(&func).expect("Cranelift compile BitNot");
        let f: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
        assert_eq!(f(42), !42);
    }

    #[test]
    #[cfg(feature = "cranelift")]
    fn test_sprint1b_mir_math_and_bitwise_parity_clif_and_zig() {
        use crate::ast::ItemKind;
        use crate::intern::Interner;
        use crate::lexer::Lexer;
        use crate::mir::cranelift_backend::jit_engine::CraneliftJitCompiler;
        use crate::mir::opt::constant_folding::ConstantFolding;
        use crate::mir::zig_backend::zig_engine::MirToZigCompiler;
        use crate::parser::Parser;

        let src = r#"
            function parity_ops(a: int, b: int): int
                local fd: int = a // b
                local rem: int = a % b
                local p: int = b ^ 3
                local band: int = a & b
                local bor: int = a | b
                local bxor: int = a ^^ b
                local shl: int = a << 2
                local shr: int = a >> 1
                local bnot: int = ~a
                return fd + rem + p + band + bor + bxor + shl + shr + bnot
            end
        "#;

        let mut interner = Interner::new();
        let mut lexer = Lexer::new(src, &mut interner);
        let mut parser = Parser::from_lexer(&mut lexer).expect("lexer");
        let items = parser.parse().expect("parse");

        let mut mir_fn = None;
        for item in items {
            if let ItemKind::Function(func_def) = item.kind {
                mir_fn = Some(AstLowering::lower_function(&func_def).expect("lower function"));
                break;
            }
        }

        let mir_fn = mir_fn.expect("Function parsed and lowered");
        let mut module = MirModule::new("test_parity_mod".into());
        module.add_function(mir_fn.clone());
        assert!(MirVerifier::verify_module(&module).is_ok());

        // Constant folding verification
        let const_src = r#"
            function const_folded(): int
                local fd = 27 // 4
                local rem = 27 % 4
                local p = 4 ^ 3
                local band = 27 & 4
                local bor = 27 | 4
                local bxor = 27 ^^ 4
                local shl = 27 << 2
                local shr = 27 >> 1
                local bnot = ~27
                return fd + rem + p + band + bor + bxor + shl + shr + bnot
            end
        "#;
        let mut const_lexer = Lexer::new(const_src, &mut interner);
        let mut const_parser = Parser::from_lexer(&mut const_lexer).expect("lexer");
        let const_items = const_parser.parse().expect("parse");
        let const_fn_def = match &const_items[0].kind {
            ItemKind::Function(f) => f.clone(),
            _ => panic!("Expected function"),
        };
        let mut const_mir = AstLowering::lower_function(&const_fn_def).expect("lower const");
        let folded = ConstantFolding::run_on_function(&mut const_mir);
        assert!(folded, "Constant folding should optimize the constant expressions");

        // 1. Cranelift JIT Execution
        let mut jit = CraneliftJitCompiler::new().expect("JIT initialization");
        let clif_code = jit.compile_function(&mir_fn).expect("Cranelift compile");
        let clif_func: extern "C" fn(i64, i64) -> i64 = unsafe { core::mem::transmute(clif_code) };
        let clif_res = clif_func(27, 4);

        // 2. Zig AOT Execution
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let dylib_path = MirToZigCompiler::compile_to_dylib(&mir_fn, temp_dir.path()).expect("Zig compile");
        let aot_res = unsafe {
            let lib = libloading::Library::new(&dylib_path).expect("Load dylib");
            let f: libloading::Symbol<extern "C" fn(i64, i64) -> i64> =
                lib.get(b"parity_ops\0").expect("Find symbol");
            f(27, 4)
        };

        // Expected calculation:
        // fd = 27 // 4 = 6
        // rem = 27 % 4 = 3
        // p = 4 ^ 3 = 64
        // band = 27 & 4 = 0
        // bor = 27 | 4 = 31
        // bxor = 27 ^^ 4 = 31
        // shl = 27 << 2 = 108
        // shr = 27 >> 1 = 13
        // bnot = ~27 = -28
        // total = 6 + 3 + 64 + 0 + 31 + 31 + 108 + 13 + (-28) = 228
        assert_eq!(clif_res, 228);
        assert_eq!(aot_res, 228);
        assert_eq!(clif_res, aot_res);

        // Verify constant folded version produces identical result
        let clif_folded_code = jit.compile_function(&const_mir).expect("Cranelift folded compile");
        let clif_folded_func: extern "C" fn() -> i64 = unsafe { core::mem::transmute(clif_folded_code) };
        assert_eq!(clif_folded_func(), 228);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_sprint4_zig_aot_emit_c_header_and_staticlib() {
        use crate::mir::zig_backend::zig_engine::{
            MirToZigCompiler, ZigArtifactKind, ZigCompileOptions,
        };
        use std::fs;

        let sig = FunctionSignature {
            name: "math_static_test".into(),
            param_types: vec![MirType::Int, MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut func = MirFunction::new(sig, Span::dummy());
        let a = func.params[0];
        let b = func.params[1];
        let temp = func.add_local(Some("temp".into()), MirType::Int, true, Span::dummy());
        let bb0 = &mut func.blocks[0];
        bb0.statements.push(Statement::Assign(
            Place::Local(temp),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(a)),
                Operand::Copy(Place::Local(b)),
            ),
            Span::dummy(),
        ));
        bb0.terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(temp))),
            Span::dummy(),
        ));

        let temp_dir = tempfile::tempdir().expect("tempdir");
        let opts = ZigCompileOptions {
            artifact_kind: ZigArtifactKind::StaticLib,
            emit_c_header: true,
            max_memory_bytes: None,
        };

        let static_lib_path = MirToZigCompiler::compile_with_options(&func, temp_dir.path(), &opts)
            .expect("Compile static lib");

        assert!(static_lib_path.exists(), "Static lib file must exist");
        let meta = fs::metadata(&static_lib_path).expect("Static lib metadata");
        assert!(meta.len() > 0, "Static lib must not be empty");

        let header_path = temp_dir.path().join("math_static_test.h");
        assert!(header_path.exists(), "C header file must exist");
        let header_content = fs::read_to_string(&header_path).expect("Read C header");

        assert!(header_content.contains("#ifndef LUST_GENERATED_MATH_STATIC_TEST_H"));
        assert!(header_content.contains("#define LUST_GENERATED_MATH_STATIC_TEST_H"));
        assert!(header_content.contains("extern \"C\""));
        assert!(header_content.contains("typedef enum {"));
        assert!(header_content.contains("LUST_STATUS_OK = 0"));
        assert!(header_content.contains("void lust_set_gas_limit(int64_t gas);"));
        assert!(header_content.contains("void lust_set_memory_limit(size_t bytes);"));
        assert!(header_content.contains("int64_t math_static_test(int64_t"));
        assert!(header_content.contains("#endif /* LUST_GENERATED_MATH_STATIC_TEST_H */"));
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_sprint4_zig_aot_max_memory_limit_enforcement() {
        use crate::ast::ItemKind;
        use crate::intern::Interner;
        use crate::lexer::Lexer;
        use crate::mir::zig_backend::zig_engine::MirToZigCompiler;
        use crate::parser::Parser;

        let src = r#"
            function alloc_ten(): int
                local arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                return arr.len()
            end
        "#;

        let mut interner = Interner::new();
        let mut lexer = Lexer::new(src, &mut interner);
        let mut parser = Parser::from_lexer(&mut lexer).expect("lexer");
        let items = parser.parse().expect("parse");
        let func_def = match &items[0].kind {
            ItemKind::Function(f) => f.clone(),
            _ => panic!("Expected function"),
        };
        let mir_fn = AstLowering::lower_function(&func_def).expect("lower function");

        let temp_dir = tempfile::tempdir().expect("tempdir");
        let dylib_path = MirToZigCompiler::compile_to_dylib(&mir_fn, temp_dir.path()).expect("Zig compile");

        unsafe {
            let lib = libloading::Library::new(&dylib_path).expect("Load dylib");
            let set_mem: libloading::Symbol<extern "C" fn(usize)> =
                lib.get(b"lust_set_memory_limit\0").expect("Find set_memory_limit");
            let get_status: libloading::Symbol<extern "C" fn() -> i32> =
                lib.get(b"lust_get_last_status\0").expect("Find get_last_status");
            let reset_arena: libloading::Symbol<extern "C" fn()> =
                lib.get(b"lust_reset_arena\0").expect("Find reset_arena");
            let alloc_fn: libloading::Symbol<extern "C" fn() -> i64> =
                lib.get(b"alloc_ten\0").expect("Find alloc_ten");

            // 1. Set limit to 16 bytes (too small for 10 * 8 = 80 bytes)
            set_mem(16);
            let res = alloc_fn();
            assert_eq!(res, 0, "When out of memory, returned slice is empty, len is 0");
            assert_eq!(get_status(), 2, "Status must be 2 (LUST_STATUS_ERR_OUT_OF_MEMORY)");

            // 2. Reset arena and increase limit to 1024 bytes -> success
            reset_arena();
            assert_eq!(get_status(), 0, "Status must be reset to 0");
            set_mem(1024);
            let res_ok = alloc_fn();
            assert_eq!(res_ok, 10, "With adequate memory, len must be 10");
            assert_eq!(get_status(), 0, "Status must remain 0 (LUST_STATUS_OK)");
        }
    }

    #[test]
    fn test_mir_to_bytecode_index_assignment_rvalue_binary_op() {
        use crate::bytecode::Value;
        use crate::vm::VM;

        let sig = FunctionSignature {
            name: "set_index_binary_op".into(),
            param_types: vec![MirType::Int, MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut func = MirFunction::new(sig, Span::dummy());
        let a = func.params[0];
        let b = func.params[1];
        let arr = func.add_local(Some("arr".into()), MirType::Array(Box::new(MirType::Int)), true, Span::dummy());
        let zero = func.add_local(Some("zero".into()), MirType::Int, true, Span::dummy());
        let result = func.add_local(Some("result".into()), MirType::Int, true, Span::dummy());

        let bb0 = &mut func.blocks[0];
        // arr = [10]
        bb0.statements.push(Statement::Assign(
            Place::Local(arr),
            Rvalue::Aggregate(MirType::Array(Box::new(MirType::Int)), vec![Operand::Constant(MirConstant::Int(10))]),
            Span::dummy(),
        ));
        // zero = 0
        bb0.statements.push(Statement::Assign(
            Place::Local(zero),
            Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));
        // arr[zero] = a + b
        bb0.statements.push(Statement::Assign(
            Place::Index(Box::new(Place::Local(arr)), zero),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(a)),
                Operand::Copy(Place::Local(b)),
            ),
            Span::dummy(),
        ));
        // result = arr[zero]
        bb0.statements.push(Statement::Assign(
            Place::Local(result),
            Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(arr)), zero))),
            Span::dummy(),
        ));
        bb0.terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(result))),
            Span::dummy(),
        ));

        let bytecode_fn = MirToBytecodeCompiler::compile(&func).expect("compile bytecode");
        let mut vm = VM::new();
        vm.load_functions(vec![bytecode_fn]);
        let res = vm.call("set_index_binary_op", vec![Value::Int(15), Value::Int(27)]).expect("VM run");
        assert_eq!(res, Value::Int(42));
    }

    #[test]
    #[cfg(feature = "cranelift")]
    fn test_unsupported_mir_constant_returns_error_in_zig_and_cranelift_backends() {
        use crate::mir::cranelift_backend::jit_engine::CraneliftJitCompiler;
        use crate::mir::zig_backend::zig_engine::MirToZigCompiler;

        let sig = FunctionSignature {
            name: "unsupported_str_const".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut func = MirFunction::new(sig, Span::dummy());
        let temp = func.add_local(Some("temp".into()), MirType::String, true, Span::dummy());
        let bb0 = &mut func.blocks[0];
        bb0.statements.push(Statement::Assign(
            Place::Local(temp),
            Rvalue::Use(Operand::Constant(MirConstant::String("hello".into()))),
            Span::dummy(),
        ));
        bb0.terminator = Some(Terminator::Return(
            Some(Operand::Constant(MirConstant::Int(0))),
            Span::dummy(),
        ));

        let zig_res = MirToZigCompiler::generate_source(&func);
        assert!(matches!(zig_res, Err(MirZigBackendError::UnsupportedConstant(MirConstant::String(_)))));

        let mut jit = CraneliftJitCompiler::new().expect("JIT init");
        let clif_res = jit.compile_function(&func);
        assert!(matches!(clif_res, Err(MirCraneliftBackendError::UnsupportedConstant(MirConstant::String(_)))));
    }

    #[test]
    fn test_unsupported_lvalue_field_access_returns_error_in_mir_lowering() {
        use crate::ast::ItemKind;
        use crate::intern::Interner;
        use crate::lexer::Lexer;
        use crate::parser::Parser;

        let src = r#"
            function field_assign(x: int): int
                local obj = [1, 2]
                obj.field = x
                return x
            end
        "#;
        let mut interner = Interner::new();
        let mut lexer = Lexer::new(src, &mut interner);
        let mut parser = Parser::from_lexer(&mut lexer).expect("lexer");
        let items = parser.parse().expect("parse");
        let func_def = match &items[0].kind {
            ItemKind::Function(f) => f.clone(),
            _ => panic!("Expected function"),
        };
        let res = AstLowering::lower_function(&func_def);
        assert!(matches!(res, Err(MirLowerError::UnsupportedLvalue(_))));
    }

    #[test]
    #[cfg(feature = "cranelift")]
    fn test_mir_for_in_array_and_ipairs_parity() {
        use crate::ast::ItemKind;
        use crate::bytecode::Value;
        use crate::intern::Interner;
        use crate::lexer::Lexer;
        use crate::mir::cranelift_backend::jit_engine::CraneliftJitCompiler;
        use crate::mir::zig_backend::zig_engine::MirToZigCompiler;
        use crate::parser::Parser;
        use crate::vm::VM;

        let src = r#"
            function sum_for_in(n: int): int
                local arr = [10, 20, 30]
                local s1: int = 0
                for x in arr do
                    s1 = s1 + x
                end
                local s2: int = 0
                for i, v in ipairs(arr) do
                    s2 = s2 + i * 100 + v
                end
                return s1 + s2 + n
            end
        "#;

        let mut interner = Interner::new();
        let mut lexer = Lexer::new(src, &mut interner);
        let mut parser = Parser::from_lexer(&mut lexer).expect("lexer");
        let items = parser.parse().expect("parse");

        let mut mir_fn = None;
        for item in items {
            if let ItemKind::Function(func_def) = item.kind {
                mir_fn = Some(AstLowering::lower_function(&func_def).expect("lower function"));
                break;
            }
        }

        let mir_fn = mir_fn.expect("Function parsed and lowered");
        let mut module = MirModule::new("test_for_in_mod".into());
        module.add_function(mir_fn.clone());
        assert!(MirVerifier::verify_module(&module).is_ok());

        // For n = 5: s1 = 10+20+30 = 60; s2 = (0*100+10) + (1*100+20) + (2*100+30) = 10+120+230 = 360; 60 + 360 + 5 = 425
        // 1. VM Bytecode Execution
        let bytecode_fn = MirToBytecodeCompiler::compile(&mir_fn).expect("compile bytecode");
        let mut vm = VM::new();
        vm.load_functions(vec![bytecode_fn]);
        let vm_res = vm.call("sum_for_in", vec![Value::Int(5)]).expect("VM execution");
        assert_eq!(vm_res, Value::Int(425));

        // 2. Cranelift JIT Execution
        let mut jit = CraneliftJitCompiler::new().expect("JIT initialization");
        let code_ptr = jit.compile_function(&mir_fn).expect("JIT compile");
        let jit_func: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
        let jit_res = jit_func(5);
        assert_eq!(jit_res, 425);

        // 3. Zig AOT Execution
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let dylib_path = MirToZigCompiler::compile_to_dylib(&mir_fn, temp_dir.path()).expect("Zig compilation");
        unsafe {
            let lib = libloading::Library::new(&dylib_path).expect("Load dylib");
            let aot_func: libloading::Symbol<extern "C" fn(i64) -> i64> =
                lib.get(b"sum_for_in\0").expect("Find symbol");
            let aot_res = aot_func(5);
            assert_eq!(aot_res, 425);
        }
    }

    #[test]
    #[cfg(feature = "cranelift")]
    fn test_mir_struct_and_tuple_field_access_and_mutation_parity() {
        use crate::ast::ItemKind;
        use crate::bytecode::Value;
        use crate::intern::Interner;
        use crate::lexer::Lexer;
        use crate::mir::cranelift_backend::jit_engine::CraneliftJitCompiler;
        use crate::mir::zig_backend::zig_engine::MirToZigCompiler;
        use crate::parser::Parser;
        use crate::vm::VM;

        let src = r#"
            function struct_tuple_ops(n: int): int
                local p = Point { x = 10, y = 20 }
                local t = (5, 15)
                p.y = 30
                p.x += 5
                return p.x + p.y + t._0 + t._1 + n
            end
        "#;

        let mut interner = Interner::new();
        let mut lexer = Lexer::new(src, &mut interner);
        let mut parser = Parser::from_lexer(&mut lexer).expect("lexer");
        let items = parser.parse().expect("parse");

        let mut mir_fn = None;
        for item in items {
            if let ItemKind::Function(func_def) = item.kind {
                mir_fn = Some(AstLowering::lower_function(&func_def).expect("lower function"));
                break;
            }
        }

        let mir_fn = mir_fn.expect("Function parsed and lowered");
        let mut module = MirModule::new("test_struct_tuple_mod".into());
        module.add_function(mir_fn.clone());
        assert!(MirVerifier::verify_module(&module).is_ok());

        // p.x (15) + p.y (30) + t._0 (5) + t._1 (15) + n (7) = 72
        // 1. VM Bytecode Execution
        let bytecode_fn = MirToBytecodeCompiler::compile(&mir_fn).expect("compile bytecode");
        let mut vm = VM::new();
        vm.load_functions(vec![bytecode_fn]);
        let vm_res = vm.call("struct_tuple_ops", vec![Value::Int(7)]).expect("VM execution");
        assert_eq!(vm_res, Value::Int(72));

        // 2. Cranelift JIT Execution
        let mut jit = CraneliftJitCompiler::new().expect("JIT initialization");
        let code_ptr = jit.compile_function(&mir_fn).expect("JIT compile");
        let jit_func: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
        let jit_res = jit_func(7);
        assert_eq!(jit_res, 72);

        // 3. Zig AOT Execution
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let dylib_path = MirToZigCompiler::compile_to_dylib(&mir_fn, temp_dir.path()).expect("Zig compilation");
        unsafe {
            let lib = libloading::Library::new(&dylib_path).expect("Load dylib");
            let aot_func: libloading::Symbol<extern "C" fn(i64) -> i64> =
                lib.get(b"struct_tuple_ops\0").expect("Find symbol");
            let aot_res = aot_func(7);
            assert_eq!(aot_res, 72);
        }
    }

    #[test]
    #[cfg(feature = "cranelift")]
    fn test_mir_switch_terminator_and_try_operator_parity() {
        use crate::ast::ItemKind;
        use crate::bytecode::Value;
        use crate::intern::Interner;
        use crate::lexer::Lexer;
        use crate::mir::cranelift_backend::jit_engine::CraneliftJitCompiler;
        use crate::mir::zig_backend::zig_engine::MirToZigCompiler;
        use crate::parser::Parser;
        use crate::vm::VM;

        // Part 1: Test Terminator::Switch across all 3 backends
        let sig = FunctionSignature {
            name: "switch_dispatch".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };
        let mut func = MirFunction::new(sig, Span::dummy());
        let arg0 = func.params[0];
        let ret_loc = func.return_local;

        // bb0: Switch
        //   case 1 -> bb1 (return 100)
        //   case 2 -> bb2 (return 200)
        //   case 3 -> bb3 (return 300)
        //   default -> bb4 (return -1)
        let bb1 = func.new_block();
        let bb2 = func.new_block();
        let bb3 = func.new_block();
        let bb4 = func.new_block();

        func.blocks[0].terminator = Some(Terminator::Switch {
            discr: Operand::Copy(Place::Local(arg0)),
            cases: vec![(1, bb1), (2, bb2), (3, bb3)],
            default: bb4,
            span: Span::dummy(),
        });

        for (b_id, val) in [(bb1, 100), (bb2, 200), (bb3, 300), (bb4, -1)] {
            func.blocks[b_id.0].statements.push(Statement::Assign(
                Place::Local(ret_loc),
                Rvalue::Use(Operand::Constant(MirConstant::Int(val))),
                Span::dummy(),
            ));
            func.blocks[b_id.0].terminator = Some(Terminator::Return(
                Some(Operand::Copy(Place::Local(ret_loc))),
                Span::dummy(),
            ));
        }

        let mut module = MirModule::new("test_switch_mod".into());
        module.add_function(func.clone());
        assert!(MirVerifier::verify_module(&module).is_ok());

        // 1. VM Backend Switch
        let bytecode_fn = MirToBytecodeCompiler::compile(&func).expect("compile bytecode");
        let mut vm = VM::new();
        vm.load_functions(vec![bytecode_fn]);
        assert_eq!(vm.call("switch_dispatch", vec![Value::Int(1)]).unwrap(), Value::Int(100));
        assert_eq!(vm.call("switch_dispatch", vec![Value::Int(2)]).unwrap(), Value::Int(200));
        assert_eq!(vm.call("switch_dispatch", vec![Value::Int(3)]).unwrap(), Value::Int(300));
        assert_eq!(vm.call("switch_dispatch", vec![Value::Int(99)]).unwrap(), Value::Int(-1));

        // 2. Cranelift JIT Switch
        let mut jit = CraneliftJitCompiler::new().expect("JIT init");
        let code_ptr = jit.compile_function(&func).expect("JIT compile");
        let jit_func: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
        assert_eq!(jit_func(1), 100);
        assert_eq!(jit_func(2), 200);
        assert_eq!(jit_func(3), 300);
        assert_eq!(jit_func(99), -1);

        // 3. Zig AOT Switch
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let dylib_path = MirToZigCompiler::compile_to_dylib(&func, temp_dir.path()).expect("Zig compilation");
        unsafe {
            let lib = libloading::Library::new(&dylib_path).expect("Load dylib");
            let aot_func: libloading::Symbol<extern "C" fn(i64) -> i64> =
                lib.get(b"switch_dispatch\0").expect("Find symbol");
            assert_eq!(aot_func(1), 100);
            assert_eq!(aot_func(2), 200);
            assert_eq!(aot_func(3), 300);
            assert_eq!(aot_func(99), -1);
        }

        // Part 2: Test try operator (arr[idx]?) lowering and execution
        let src = r#"
            function safe_lookup(idx: int): int
                local arr = [100, 200, 300]
                local val = arr[idx]?
                return val + 1
            end
        "#;
        let mut interner = Interner::new();
        let mut lexer = Lexer::new(src, &mut interner);
        let mut parser = Parser::from_lexer(&mut lexer).expect("lexer");
        let items = parser.parse().expect("parse");
        let func_def = match &items[0].kind {
            ItemKind::Function(f) => f.clone(),
            _ => panic!("Expected function"),
        };
        let try_fn = AstLowering::lower_function(&func_def).expect("lower try_fn");

        // VM
        let try_bytecode = MirToBytecodeCompiler::compile(&try_fn).expect("compile try bytecode");
        let mut try_vm = VM::new();
        try_vm.load_functions(vec![try_bytecode]);
        assert_eq!(try_vm.call("safe_lookup", vec![Value::Int(1)]).unwrap(), Value::Int(201));
        assert_eq!(try_vm.call("safe_lookup", vec![Value::Int(5)]).unwrap(), Value::Int(-1));

        // Cranelift JIT
        let mut try_jit = CraneliftJitCompiler::new().expect("JIT init");
        let try_code_ptr = try_jit.compile_function(&try_fn).expect("JIT compile");
        let try_jit_func: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(try_code_ptr) };
        assert_eq!(try_jit_func(1), 201);
        assert_eq!(try_jit_func(5), -1);

        // Zig AOT
        let try_temp_dir = tempfile::tempdir().expect("tempdir");
        let try_dylib_path = MirToZigCompiler::compile_to_dylib(&try_fn, try_temp_dir.path()).expect("Zig compilation");
        unsafe {
            let lib = libloading::Library::new(&try_dylib_path).expect("Load dylib");
            let try_aot_func: libloading::Symbol<extern "C" fn(i64) -> i64> =
                lib.get(b"safe_lookup\0").expect("Find symbol");
            assert_eq!(try_aot_func(1), 201);
            assert_eq!(try_aot_func(5), -1);
        }
    }

    #[test]
    #[cfg(feature = "cranelift")]
    fn test_mir_multi_function_call_and_recursion_parity() {
        use crate::ast::ItemKind;
        use crate::bytecode::Value;
        use crate::intern::Interner;
        use crate::lexer::Lexer;
        use crate::mir::cranelift_backend::jit_engine::CraneliftJitCompiler;
        use crate::mir::zig_backend::zig_engine::MirToZigCompiler;
        use crate::parser::Parser;
        use crate::vm::VM;

        let src = r#"
            function add_mul(a: int, b: int): int
                return (a + b) * 2
            end

            function fib(n: int): int
                if n <= 1 then
                    return n
                else
                    return fib(n - 1) + fib(n - 2)
                end
            end

            function main_entry(n: int): int
                return fib(n) + add_mul(n, 3)
            end
        "#;

        let mut interner = Interner::new();
        let mut lexer = Lexer::new(src, &mut interner);
        let mut parser = Parser::from_lexer(&mut lexer).expect("lexer");
        let items = parser.parse().expect("parse");

        let mut module = MirModule::new("test_multi_fn_mod".into());
        let mut bytecode_fns = Vec::new();

        for item in items {
            if let ItemKind::Function(func_def) = item.kind {
                let mir_fn = AstLowering::lower_function(&func_def).expect("lower function");
                let bc_fn = MirToBytecodeCompiler::compile(&mir_fn).expect("compile bytecode");
                bytecode_fns.push(bc_fn);
                module.add_function(mir_fn);
            }
        }

        assert_eq!(module.functions.len(), 3);
        assert!(MirVerifier::verify_module(&module).is_ok());

        // For n = 7:
        // fib(7) = 13
        // add_mul(7, 3) = (7 + 3) * 2 = 20
        // total = 13 + 20 = 33

        // 1. VM Execution
        let mut vm = VM::new();
        vm.load_functions(bytecode_fns.clone());
        for (i, f) in bytecode_fns.iter().enumerate() {
            vm.set_global(&f.name, Value::Function(i));
        }
        let vm_res = vm.call("main_entry", vec![Value::Int(7)]).expect("VM execution");
        assert_eq!(vm_res, Value::Int(33));

        // 2. Cranelift JIT Execution
        let mut jit = CraneliftJitCompiler::new().expect("JIT initialization");
        let func_ptrs = jit.compile_module(&module).expect("compile module");
        let main_ptr = *func_ptrs.get("main_entry").expect("main_entry found");
        let jit_func: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(main_ptr) };
        let jit_res = jit_func(7);
        assert_eq!(jit_res, 33);

        // 3. Zig AOT Execution
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let dylib_path = MirToZigCompiler::compile_module_to_dylib(&module, temp_dir.path()).expect("Zig compilation");
        unsafe {
            let lib = libloading::Library::new(&dylib_path).expect("Load dylib");
            let aot_func: libloading::Symbol<extern "C" fn(i64) -> i64> =
                lib.get(b"main_entry\0").expect("Find symbol");
            let aot_res = aot_func(7);
            assert_eq!(aot_res, 33);
        }
    }
}

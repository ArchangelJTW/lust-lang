#![cfg(all(feature = "std", feature = "packages"))]

use lust::ast::ItemKind;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::c_backend::{CArtifactKind, CCompileOptions, MirToCCompiler};
use lust::mir::ir::MirModule;
use lust::mir::lowering::AstLowering;
use lust::mir::opt::MirOptimizer;
use lust::mir::verifier::MirVerifier;
use lust::parser::Parser;
use std::process::Command;

fn lower_module(name: &str, source: &str) -> MirModule {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");

    let mut module = MirModule::new(name.to_string());
    for item in &items {
        if let ItemKind::Function(f) = &item.kind {
            let mut mir_fn = AstLowering::lower_function(f).expect("MIR lowering");
            MirOptimizer::optimize_function(&mut mir_fn);
            MirVerifier::verify_fn(&mir_fn).expect("MIR verification");
            module.add_function(mir_fn);
        }
    }
    module
}

#[test]
fn mir_c_backend_math_bitwise_and_loops_dylib() {
    let source = r#"
        function compute(n: int): int
            local acc: int = 0
            for i = 1, n do
                local v: int = (i * 3) // 2 + (i % 5) + ((i & 7) ~ (i | 1)) + (2 ^ (i % 4))
                acc = acc + v
            end
            return acc
        end
    "#;
    let module = lower_module("compute_mod", source);
    let tmp = tempfile::tempdir().expect("tempdir");
    let dylib_path = MirToCCompiler::compile_module_to_dylib(&module, tmp.path())
        .expect("C dylib compilation must succeed");

    unsafe {
        let lib = libloading::Library::new(&dylib_path).expect("load compiled C dylib");
        let func: libloading::Symbol<unsafe extern "C" fn(i64) -> i64> =
            lib.get(b"compute").expect("load symbol compute");
        let mut expected: i64 = 0;
        for i in 1..=20i64 {
            let v = i.wrapping_mul(3).div_euclid(2)
                + (i % 5)
                + ((i & 7) ^ (i | 1))
                + 2i64.pow((i % 4) as u32);
            expected = expected.wrapping_add(v);
        }
        assert_eq!(func(20), expected);
    }
}

#[test]
fn mir_c_backend_recursion_and_multi_function_dylib() {
    let source = r#"
        function fib(n: int): int
            if n <= 1 then
                return n
            end
            return fib(n - 1) + fib(n - 2)
        end

        function run_fib(n: int): int
            return fib(n)
        end
    "#;
    let module = lower_module("fib_mod", source);
    let tmp = tempfile::tempdir().expect("tempdir");
    let dylib_path = MirToCCompiler::compile_module_to_dylib(&module, tmp.path())
        .expect("C dylib compilation must succeed");

    unsafe {
        let lib = libloading::Library::new(&dylib_path).expect("load compiled C dylib");
        let run_fib: libloading::Symbol<unsafe extern "C" fn(i64) -> i64> =
            lib.get(b"run_fib").expect("load symbol run_fib");
        assert_eq!(run_fib(10), 55);
        assert_eq!(run_fib(20), 6765);
    }
}

#[test]
fn mir_c_backend_arrays_structs_and_memory_limit() {
    let source = r#"
        function sum_array(): int
            local arr = [10, 20, 30, 40]
            arr[1] = 25
            local s: int = 0
            for v in arr do
                s = s + v
            end
            return s
        end
    "#;
    let module = lower_module("arr_mod", source);
    let tmp = tempfile::tempdir().expect("tempdir");
    let dylib_path = MirToCCompiler::compile_module_to_dylib(&module, tmp.path())
        .expect("C dylib compilation must succeed");

    unsafe {
        let lib = libloading::Library::new(&dylib_path).expect("load compiled C dylib");
        let sum_array: libloading::Symbol<unsafe extern "C" fn() -> i64> =
            lib.get(b"sum_array").expect("load symbol sum_array");
        let get_mem: libloading::Symbol<unsafe extern "C" fn() -> usize> =
            lib.get(b"lust_get_memory_used").expect("load lust_get_memory_used");
        let set_limit: libloading::Symbol<unsafe extern "C" fn(usize)> =
            lib.get(b"lust_set_memory_limit").expect("load lust_set_memory_limit");
        let get_status: libloading::Symbol<unsafe extern "C" fn() -> i32> =
            lib.get(b"lust_get_last_status").expect("load lust_get_last_status");
        let reset: libloading::Symbol<unsafe extern "C" fn()> =
            lib.get(b"lust_reset_arena").expect("load lust_reset_arena");

        reset();
        assert_eq!(sum_array(), 10 + 25 + 30 + 40);
        assert_eq!(get_status(), 0);
        assert!(get_mem() >= 32);

        // Enforce a tight memory limit (16 bytes < 32 bytes needed for 4 i64s)
        reset();
        set_limit(16);
        let _ = sum_array();
        assert_eq!(get_status(), 2, "exceeding memory limit must set status=2");
    }
}

#[test]
fn mir_c_backend_gas_limit_enforcement() {
    let source = r#"
        function spin(n: int): int
            local acc: int = 0
            for i = 0, n do
                acc = acc + i
            end
            return acc
        end
    "#;
    let module = lower_module("gas_mod", source);
    let tmp = tempfile::tempdir().expect("tempdir");
    let dylib_path = MirToCCompiler::compile_module_to_dylib(&module, tmp.path())
        .expect("C dylib compilation must succeed");

    unsafe {
        let lib = libloading::Library::new(&dylib_path).expect("load compiled C dylib");
        let spin: libloading::Symbol<unsafe extern "C" fn(i64) -> i64> =
            lib.get(b"spin").expect("load symbol spin");
        let set_gas: libloading::Symbol<unsafe extern "C" fn(i64)> =
            lib.get(b"lust_set_gas_limit").expect("load lust_set_gas_limit");
        let get_status: libloading::Symbol<unsafe extern "C" fn() -> i32> =
            lib.get(b"lust_get_last_status").expect("load lust_get_last_status");
        let reset: libloading::Symbol<unsafe extern "C" fn()> =
            lib.get(b"lust_reset_arena").expect("load lust_reset_arena");

        reset();
        set_gas(5);
        let _ = spin(1000);
        assert_eq!(get_status(), 1, "exhausting gas must set status=1");
    }
}

#[test]
fn mir_c_backend_staticlib_and_c_header() {
    let source = r#"
        function add_nums(a: int, b: int): int
            return a + b
        end
    "#;
    let module = lower_module("static_mod", source);
    let tmp = tempfile::tempdir().expect("tempdir");
    let opts = CCompileOptions {
        artifact_kind: CArtifactKind::StaticLib,
        emit_c_header: true,
        ..Default::default()
    };
    let lib_path = MirToCCompiler::compile_module_with_options(&module, tmp.path(), &opts)
        .expect("C staticlib compilation must succeed");
    assert!(lib_path.exists(), "static lib artifact must exist");
    let header_path = tmp.path().join("static_mod.h");
    assert!(header_path.exists(), "C header must be emitted");
}

#[test]
fn mir_c_backend_standalone_executable() {
    let source = r#"
        function fib(n: int): int
            if n <= 1 then
                return n
            end
            return fib(n - 1) + fib(n - 2)
        end

        function main(): int
            println(fib(12))
            return 0
        end
    "#;
    let module = lower_module("exec_mod", source);
    let tmp = tempfile::tempdir().expect("tempdir");
    let opts = CCompileOptions {
        artifact_kind: CArtifactKind::Executable,
        ..Default::default()
    };
    let exe_path = MirToCCompiler::compile_module_with_options(&module, tmp.path(), &opts)
        .expect("C executable compilation must succeed");

    let out = Command::new(&exe_path)
        .output()
        .expect("execute compiled C binary");
    assert!(out.status.success(), "compiled binary must exit 0");
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "144");
}

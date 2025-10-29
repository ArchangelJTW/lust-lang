#![allow(improper_ctypes)]
pub mod ast;
pub mod builtins;
pub mod bytecode;
pub mod config;
pub mod embed;
pub mod error;
pub mod ffi;
pub mod jit;
pub mod lexer;
pub mod modules;
#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
pub mod packages;
pub mod parser;
pub mod typechecker;
pub mod vm;
#[cfg(target_arch = "wasm32")]
pub mod wasm;
pub use ast::{Expr, Item, Span, Stmt, Type};
pub use bytecode::{Chunk, Compiler, Function, Instruction, Value};
pub use config::{ConfigError, LustConfig};
pub use embed::{
    EmbeddedBuilder, EmbeddedProgram, EnumInstance, FromLustValue, FunctionArgs, IntoLustValue,
    StructInstance,
};
pub use error::{LustError, Result};
pub use jit::{JitCompiler, JitState};
pub use lexer::{Lexer, Token, TokenKind};
pub use modules::{LoadedModule, ModuleImports, ModuleLoader, Program};
#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
pub use packages::{
    build_local_module, collect_stub_files, load_local_module, stub_files_from_exports,
    write_stub_files, LoadedRustModule, LocalBuildOutput, LocalModuleError, PackageKind,
    PackageManager, PackageSpecifier, StubFile,
};
pub use parser::Parser;
pub use typechecker::{FunctionSignature, TypeChecker, TypeCollection};
pub use vm::{NativeExport, NativeExportParam, VM};

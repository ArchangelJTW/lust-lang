#![allow(improper_ctypes)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod ast;
pub mod builtins;
pub mod bytecode;
pub mod config;
#[cfg(feature = "std")]
pub mod embed;
pub mod error;
#[cfg(feature = "std")]
pub mod ffi;
pub mod jit;
mod lazy;
pub mod lexer;
pub mod modules;
pub mod number;
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
#[cfg(feature = "std")]
pub use embed::{
    struct_field, ArrayHandle, EmbeddedBuilder, EmbeddedProgram, EnumInstance, FromLustValue,
    FunctionArgs, IntoLustValue, MapHandle, StructField, StructInstance, ValueRef,
};
pub use error::{LustError, Result};
pub use jit::{JitCompiler, JitState};
pub use lexer::{Lexer, Token, TokenKind};
pub use modules::{LoadedModule, ModuleImports, ModuleLoader, Program};
pub use number::{LustFloat, LustInt};
#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
pub use packages::{
    build_local_module, collect_stub_files, load_local_module, stub_files_from_exports,
    write_stub_files, LoadedRustModule, LocalBuildOutput, LocalModuleError, PackageKind,
    PackageManager, PackageSpecifier, StubFile,
};
pub use parser::Parser;
pub use typechecker::{FunctionSignature, TypeChecker, TypeCollection};
pub use vm::{NativeExport, NativeExportParam, VM};

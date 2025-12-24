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
#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
pub mod lua_compat;
pub mod lexer;
#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
pub mod lua_compat;
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
pub use config::{ConfigError, DependencyKind, DependencySpec, LustConfig};
#[cfg(feature = "std")]
pub use embed::{
    enum_variant, enum_variant_with, function_param, private_struct_field_decl, self_param,
    struct_field, struct_field_decl, trait_bound, type_named, type_unit, type_unknown,
    weak_struct_field_decl, ArrayHandle, AsyncDriver, AsyncTaskQueue, EmbeddedBuilder,
    EmbeddedProgram, EnumInstance, ExternRegistry, FromLustValue, FromStructField, FunctionArgs,
    FunctionBuilder, FunctionHandle, ImplBuilder, IntoLustValue, MapHandle, ModuleStub, StringRef,
    StructBuilder, StructField, StructHandle, StructInstance, TraitBuilder, TraitMethodBuilder,
    ValueRef,
};
pub use error::{LustError, Result};
pub use jit::{JitCompiler, JitState};
pub use lexer::{Lexer, Token, TokenKind};
#[cfg(feature = "std")]
pub use lust_macros::LustStructView;
pub use modules::{LoadedModule, ModuleImports, ModuleLoader, Program};
pub use number::{LustFloat, LustInt};
#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
pub use packages::{
    build_local_module, build_package_archive, clear_credentials, collect_stub_files,
    credentials_file, load_credentials, load_local_module, resolve_dependencies, save_credentials,
    stub_files_from_exports, write_stub_files, ArchiveError, BuildOptions, Credentials,
    CredentialsError, DependencyResolution, DependencyResolutionError, DownloadedArchive,
    LoadedRustModule, LocalBuildOutput, LocalModuleError, ManifestError, PackageArchive,
    PackageDetails, PackageKind, PackageManager, PackageManifest, PackageSpecifier, PackageSummary,
    PackageVersionInfo, PublishResponse, RegistryClient, RegistryError, ResolvedLuaDependency,
    ResolvedLustDependency, ResolvedRustDependency, SearchParameters, StubFile, DEFAULT_BASE_URL,
};
pub use parser::Parser;
pub use typechecker::{FunctionSignature, TypeChecker, TypeCollection};
pub use vm::{NativeExport, NativeExportParam, VM};

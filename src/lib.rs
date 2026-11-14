#![allow(improper_ctypes)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod ast;
#[cfg(feature = "bindgen")]
pub mod bindgen;
pub mod builtins;
pub mod bytecode;
pub mod config;
pub mod data;
pub mod model;
pub mod schema;
#[cfg(feature = "std")]
pub mod regex;
#[cfg(feature = "std")]
pub mod rx;
#[cfg(feature = "std")]
pub mod codegen_zig;
pub mod effect_abi;
pub mod embed;
pub mod error;
#[cfg(feature = "std")]
pub mod externs;
#[cfg(feature = "std")]
pub mod ffi;
pub mod intern;
pub mod jit;
pub mod mir;
mod lazy;
pub mod lexer;
#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
pub mod lua_compat;
pub mod modules;
pub mod number;
#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
pub mod packages;
pub mod parser;
pub mod test_framework;
pub mod typechecker;
pub mod vm;
pub mod sim;
#[cfg(feature = "std")]
pub mod stdlib;
#[cfg(feature = "std")]
pub mod vfs;
#[cfg(target_arch = "wasm32")]
pub mod wasm;
pub use ast::{Expr, Item, Span, Stmt, Type};
pub use bytecode::{Chunk, Compiler, Function, Instruction, Value};
pub use config::{ConfigError, DependencyKind, DependencySpec, LustConfig};
pub use data::{
    BsonArrayView, BsonDocumentView, BsonElementIter, BsonError, BsonValueView, BudgetError,
    BudgetTracker, CanonicalTypeTag, DataValue, DataView, DateTimeUtc, Decimal128, Document,
    DuplicateKeyError, ExactNumber, LeaseError, LeaseLifecycleState, LogicalFieldDesc,
    LogicalLayoutV1, LustSafeYaml12, ObjectId, OwnerId, OwnerRegistry, TargetAbi, ToonCodec,
    ToonError, ToonTable, YamlError,
};
#[cfg(feature = "std")]
pub use embed::{
    ArrayHandle, AsyncDriver, AsyncTaskQueue, CompiledBytecode, EmbeddedBuilder, EmbeddedProgram,
    EnumInstance, ExternRegistry, FromLustValue, FromStructField, FunctionArgs, FunctionBuilder,
    FunctionHandle, ImplBuilder, IntoLustValue, MapHandle, ModuleStub, StringRef, StructBuilder,
    StructField, StructHandle, StructInstance, TraitBuilder, TraitMethodBuilder, TypedHandle,
    ValueRef, enum_variant, enum_variant_with, function_param, private_struct_field_decl,
    self_param, struct_field, struct_field_decl, trait_bound, type_named, type_unit, type_unknown,
    weak_struct_field_decl,
};
pub use effect_abi::{
    acquire_release, acquire_release_simple, effect_acquire_release, effect_repeat, effect_retry,
    effect_timeout, repeat, retry, timeout, Cause, Exit, FiberContext, FiberExecutionState,
    InterruptReason, LayerRollbackLedger, LustClosureObject, LustEffectDescriptor, ScheduleKind,
    ScheduleState, ScheduleStep, Scope, ScopeAdmission, ScopeTeardown, StackContextProjection,
};
pub use error::{LustError, Result};
pub use jit::{JitCompiler, JitState, JitStats};
pub use lexer::{Lexer, Token, TokenKind};
#[cfg(feature = "std")]
pub use lust_macros::LustStructView;
pub use modules::embedded::{
    EmbeddedModule, build_directory_map, load_program_from_embedded,
    load_program_from_embedded_with_config,
};
pub use modules::{
    LoadedModule, ModuleImports, ModuleLoader, Program, compile_program_with_config,
};
pub use number::{LustFloat, LustInt};
#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
pub use packages::{
    ArchiveError, BuildOptions, Credentials, CredentialsError, DEFAULT_BASE_URL,
    DependencyResolution, DependencyResolutionError, DownloadedArchive, LoadedRustModule,
    LocalBuildOutput, LocalModuleError, ManifestError, PackageArchive, PackageDetails, PackageKind,
    PackageManager, PackageManifest, PackageSpecifier, PackageSummary, PackageVersionInfo,
    PublishResponse, RegistryClient, RegistryError, ResolvedLuaDependency, ResolvedLustDependency,
    ResolvedRustDependency, SearchParameters, StubFile, build_local_module, build_package_archive,
    clear_credentials, collect_stub_files, credentials_file, load_credentials, load_local_module,
    resolve_dependencies, save_credentials, stub_files_from_exports, write_stub_files,
};
pub use parser::Parser;
pub use typechecker::{FunctionSignature, TypeChecker, TypeCollection};
pub use vm::{
    NativeEffectSpec, NativeExport, NativeExportParam, NativeGasPolicy, NativeMemoryPolicy,
    NativePurity, VM,
    MUTATION_M1_EXECUTION_COUNT, MUTATION_M1_SKIP_GAS, MutationM1Guard,
    MUTATION_M2_EXECUTION_COUNT, MUTATION_M2_SKIP_PANIC_FINALIZER, MutationM2Guard,
};
#[cfg(feature = "std")]
pub use stdlib::effects::{
    EnvError, EnvService, FileHandle, FileSystemService, IoError, RandomService,
};
pub use sim::{
    FaultConfiguration, FaultInjectedVfs, FaultKind, GasMonotonicityOracle, GasProgressOracle,
    InterruptSchedule, InvariantViolation, InvBudget01Oracle, InvFiber01Oracle, InvFiber02Oracle,
    InvFiber03Oracle, InvRx01Oracle, MemoryFaultPolicy, MinimalSimScheduler,
    PreemptiveExplorationScheduler, StepCounterWatchdog, StepOutcome, VfsFaultRule, VfsOp,
};
#[cfg(feature = "std")]
pub use vfs::{
    DirectHostFsBypassPanic, MemFs, OsFs, VfsMetadata, VirtualConsole, VirtualEnv, VirtualFileSystem,
};
pub use hashbrown;

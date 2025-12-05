//! Lua 5.1 C API compatibility shim scaffolding.
//! Future work: wire this to load `luaopen_*` entrypoints, record API calls,
//! and emit Lust extern stubs for discovered functions.

use std::path::PathBuf;

/// Metadata about a Lua C library that should be loaded through the compatibility layer.
#[derive(Debug, Clone)]
pub struct LuaModuleSpec {
    pub library_path: PathBuf,
    pub entrypoints: Vec<String>,
}

impl LuaModuleSpec {
    pub fn new(library_path: PathBuf, entrypoints: Vec<String>) -> Self {
        Self {
            library_path,
            entrypoints,
        }
    }
}

/// Describes a traced call into the Lua 5.1 API while evaluating `luaopen_*`.
#[derive(Debug, Clone)]
pub struct LuaApiCall {
    pub function: String,
    pub args: Vec<String>,
}

/// Placeholder for a traced module export that can later be turned into a Lust extern stub.
#[derive(Debug, Clone)]
pub struct LuaModuleTrace {
    pub module: String,
    pub api_calls: Vec<LuaApiCall>,
}

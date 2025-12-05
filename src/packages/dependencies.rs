use super::{
    manifest::{ManifestError, ManifestKind, PackageManifest},
    PackageManager,
};
use crate::config::{DependencyKind, LustConfig};
use std::{
    collections::HashSet,
    fs,
    io,
    path::{Path, PathBuf},
};
use object::{File, Object, ObjectSymbol};
use thiserror::Error;

#[derive(Debug, Default, Clone)]
pub struct DependencyResolution {
    lust: Vec<ResolvedLustDependency>,
    rust: Vec<ResolvedRustDependency>,
    lua: Vec<ResolvedLuaDependency>,
}

impl DependencyResolution {
    pub fn lust(&self) -> &[ResolvedLustDependency] {
        &self.lust
    }

    pub fn rust(&self) -> &[ResolvedRustDependency] {
        &self.rust
    }

    pub fn lua(&self) -> &[ResolvedLuaDependency] {
        &self.lua
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedLustDependency {
    pub name: String,
    pub sanitized_name: Option<String>,
    pub module_root: PathBuf,
    pub root_module: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct ResolvedRustDependency {
    pub name: String,
    pub crate_dir: PathBuf,
    pub features: Vec<String>,
    pub default_features: bool,
    pub externs_override: Option<PathBuf>,
    pub cache_stub_dir: Option<PathBuf>,
    pub version: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ResolvedLuaDependency {
    pub name: String,
    pub library_path: PathBuf,
    pub luaopen_symbols: Vec<String>,
    pub version: Option<String>,
}

#[derive(Debug, Clone)]
enum DetectedKind {
    Lust,
    Rust,
    Lua { luaopen_symbols: Vec<String> },
}

#[derive(Default)]
struct LibrarySignature {
    luaopen_symbols: Vec<String>,
    has_lust_extension: bool,
}

#[derive(Debug, Error)]
pub enum DependencyResolutionError {
    #[error("failed to prepare package cache: {source}")]
    PackageCache {
        #[source]
        source: io::Error,
    },
    #[error("dependency '{name}' expected directory at {path}")]
    MissingPath { name: String, path: PathBuf },
    #[error("dependency '{name}' package version '{version}' not installed (expected at {path})")]
    MissingPackage {
        name: String,
        version: String,
        path: PathBuf,
    },
    #[error("dependency '{name}' manifest error: {source}")]
    Manifest {
        name: String,
        #[source]
        source: ManifestError,
    },
    #[error("failed to read library '{path}': {source}")]
    LibraryIo {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to inspect library '{path}': {source}")]
    LibraryInspect {
        path: PathBuf,
        #[source]
        source: object::read::Error,
    },
    #[error("dependency '{name}' at {path} is a shared library but its kind could not be detected")]
    UnknownLibraryKind { name: String, path: PathBuf },
}

pub fn resolve_dependencies(
    config: &LustConfig,
    project_dir: &Path,
) -> Result<DependencyResolution, DependencyResolutionError> {
    let mut resolution = DependencyResolution::default();
    let manager = PackageManager::new(PackageManager::default_root());
    manager
        .ensure_layout()
        .map_err(|source| DependencyResolutionError::PackageCache { source })?;

    for spec in config.dependencies() {
        let name = spec.name().to_string();
        let (root_dir, version) = if let Some(path) = spec.path() {
            (resolve_dependency_path(project_dir, path), None)
        } else if let Some(version) = spec.version() {
            let dir = manager.root().join(spec.name()).join(version);
            if !dir.exists() {
                return Err(DependencyResolutionError::MissingPackage {
                    name: spec.name().to_string(),
                    version: version.to_string(),
                    path: dir,
                });
            }
            (dir, Some(version.to_string()))
        } else {
            // Parser guarantees either path or version exists.
            unreachable!("dependency spec missing path and version");
        };

        if !root_dir.exists() {
            return Err(DependencyResolutionError::MissingPath {
                name: spec.name().to_string(),
                path: root_dir,
            });
        }

        let detected = match spec.kind() {
            Some(DependencyKind::Lust) => DetectedKind::Lust,
            Some(DependencyKind::Rust) => DetectedKind::Rust,
            Some(DependencyKind::Lua) => DetectedKind::Lua {
                luaopen_symbols: detect_luaopen_symbols(&root_dir)?,
            },
            None => detect_kind(spec.name(), &root_dir)?,
        };

        match detected {
            DetectedKind::Lust => {
                let module_root = resolve_module_root(&root_dir);
                let root_module = detect_root_module(&module_root, spec.name());
                let sanitized = sanitize_dependency_name(&name);
                let sanitized_name = if sanitized != name {
                    Some(sanitized)
                } else {
                    None
                };
                resolution.lust.push(ResolvedLustDependency {
                    name,
                    sanitized_name,
                    module_root,
                    root_module,
                });
            }
            DetectedKind::Rust => {
                let externs_override = spec
                    .externs()
                    .map(|value| resolve_optional_path(&root_dir, value));
                let cache_stub_dir = if spec.path().is_some() {
                    None
                } else {
                    Some(root_dir.join("externs"))
                };
                resolution.rust.push(ResolvedRustDependency {
                    name,
                    crate_dir: root_dir,
                    features: spec.features().to_vec(),
                    default_features: spec.default_features().unwrap_or(true),
                    externs_override,
                    cache_stub_dir,
                    version,
                });
            }
            DetectedKind::Lua { luaopen_symbols } => {
                resolution.lua.push(ResolvedLuaDependency {
                    name,
                    library_path: root_dir,
                    luaopen_symbols,
                    version,
                });
            }
        }
    }

    Ok(resolution)
}

fn detect_kind(name: &str, root: &Path) -> Result<DetectedKind, DependencyResolutionError> {
    if root.is_file() {
        let signature = inspect_library(root)?;
        let luaopen_symbols = signature.luaopen_symbols;
        let has_lust_register = signature.has_lust_extension;
        return if !luaopen_symbols.is_empty() {
            Ok(DetectedKind::Lua { luaopen_symbols })
        } else if has_lust_register {
            Ok(DetectedKind::Rust)
        } else {
            Err(DependencyResolutionError::UnknownLibraryKind {
                name: name.to_string(),
                path: root.to_path_buf(),
            })
        };
    }

    match PackageManifest::discover(root) {
        Ok(manifest) => match manifest.kind() {
            ManifestKind::Lust => Ok(DetectedKind::Lust),
            ManifestKind::Cargo => Ok(DetectedKind::Rust),
        },
        Err(ManifestError::NotFound(_)) => {
            if root.join("Cargo.toml").exists() {
                Ok(DetectedKind::Rust)
            } else {
                Ok(DetectedKind::Lust)
            }
        }
        Err(err) => Err(DependencyResolutionError::Manifest {
            name: name.to_string(),
            source: err,
        }),
    }
}

fn detect_luaopen_symbols(root: &Path) -> Result<Vec<String>, DependencyResolutionError> {
    if !root.is_file() {
        return Ok(Vec::new());
    }
    let signature = inspect_library(root)?;
    Ok(signature.luaopen_symbols)
}

fn detect_lust_extension_symbol(root: &Path) -> Result<bool, DependencyResolutionError> {
    if !root.is_file() {
        return Ok(false);
    }
    let signature = inspect_library(root)?;
    Ok(signature.has_lust_extension)
}

fn inspect_library(path: &Path) -> Result<LibrarySignature, DependencyResolutionError> {
    let bytes = fs::read(path).map_err(|source| DependencyResolutionError::LibraryIo {
        path: path.to_path_buf(),
        source,
    })?;
    let file = File::parse(&*bytes).map_err(|source| {
        DependencyResolutionError::LibraryInspect {
            path: path.to_path_buf(),
            source,
        }
    })?;

    let mut signature = LibrarySignature::default();
    let mut lua_syms: HashSet<String> = HashSet::new();

    for symbol in file.symbols().chain(file.dynamic_symbols()) {
        if !symbol.is_definition() {
            continue;
        }
        let Ok(raw_name) = symbol.name() else {
            continue;
        };
        let name = raw_name.trim_start_matches('_');
        if name == "lust_extension_register" {
            signature.has_lust_extension = true;
        }
        if let Some(stripped) = name.strip_prefix("luaopen_") {
            lua_syms.insert(format!("luaopen_{stripped}"));
        }
    }

    signature.luaopen_symbols = lua_syms.into_iter().collect();
    signature.luaopen_symbols.sort();
    Ok(signature)
}

fn resolve_dependency_path(project_dir: &Path, raw: &str) -> PathBuf {
    if raw == "/" {
        return project_dir.to_path_buf();
    }

    let candidate = PathBuf::from(raw);
    if candidate.is_absolute() {
        candidate
    } else {
        project_dir.join(candidate)
    }
}

fn resolve_optional_path(root: &Path, raw: &str) -> PathBuf {
    if raw == "/" {
        return root.to_path_buf();
    }
    let candidate = PathBuf::from(raw);
    if candidate.is_absolute() {
        candidate
    } else {
        root.join(candidate)
    }
}

fn resolve_module_root(root: &Path) -> PathBuf {
    let src = root.join("src");
    if src.is_dir() {
        src
    } else {
        root.to_path_buf()
    }
}

fn detect_root_module(module_root: &Path, prefix: &str) -> Option<PathBuf> {
    let lib = module_root.join("lib.lust");
    if lib.exists() {
        return Some(PathBuf::from("lib.lust"));
    }

    let prefixed = module_root.join(format!("{prefix}.lust"));
    if prefixed.exists() {
        return Some(PathBuf::from(format!("{prefix}.lust")));
    }

    let sanitized = sanitize_dependency_name(prefix);
    if sanitized != prefix {
        let sanitized_path = module_root.join(format!("{sanitized}.lust"));
        if sanitized_path.exists() {
            return Some(PathBuf::from(format!("{sanitized}.lust")));
        }
    }

    let main = module_root.join("main.lust");
    if main.exists() {
        return Some(PathBuf::from("main.lust"));
    }

    None
}

fn sanitize_dependency_name(name: &str) -> String {
    name.replace('-', "_")
}

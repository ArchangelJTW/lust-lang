// Host CLI / packaging boundary: package manager operates on the host filesystem directly.
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]
#![allow(clippy::result_large_err)]

use super::{
    PackageManager,
    manifest::{ManifestError, ManifestKind, PackageManifest},
};
use crate::config::{DependencyKind, LustConfig};
use object::{File, Object, ObjectSymbol};
use std::{
    collections::{HashMap, HashSet},
    fs, io,
    path::{Path, PathBuf},
};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Semver {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
}

impl Semver {
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().trim_prefix('v');
        let clean = s.split(['-', '+']).next().unwrap_or(s);
        let parts: Vec<&str> = clean.split('.').collect();
        if parts.is_empty() || parts.len() > 3 {
            return None;
        }
        let major = parts[0].parse::<u64>().ok()?;
        let minor = if parts.len() > 1 {
            parts[1].parse::<u64>().ok()?
        } else {
            0
        };
        let patch = if parts.len() > 2 {
            parts[2].parse::<u64>().ok()?
        } else {
            0
        };
        Some(Self {
            major,
            minor,
            patch,
        })
    }

    pub fn matches_constraint(&self, constraint: &str) -> bool {
        let constraint = constraint.trim();
        if constraint == "*" || constraint.is_empty() {
            return true;
        }
        if constraint.contains(',') {
            return constraint
                .split(',')
                .all(|c| self.matches_single_constraint(c.trim()));
        }
        self.matches_single_constraint(constraint)
    }

    fn matches_single_constraint(&self, c: &str) -> bool {
        let c = c.trim();
        if c == "*" || c.is_empty() {
            return true;
        }
        if let Some(rest) = c.strip_prefix('^') {
            let Some(req) = Semver::parse(rest) else {
                return false;
            };
            if *self < req {
                return false;
            }
            if req.major > 0 {
                self.major == req.major
            } else if req.minor > 0 {
                self.major == 0 && self.minor == req.minor
            } else {
                self.major == 0 && self.minor == 0 && self.patch == req.patch
            }
        } else if let Some(rest) = c.strip_prefix('~') {
            let Some(req) = Semver::parse(rest) else {
                return false;
            };
            if *self < req {
                return false;
            }
            self.major == req.major && self.minor == req.minor
        } else if let Some(rest) = c.strip_prefix(">=") {
            let Some(req) = Semver::parse(rest) else {
                return false;
            };
            *self >= req
        } else if let Some(rest) = c.strip_prefix("<=") {
            let Some(req) = Semver::parse(rest) else {
                return false;
            };
            *self <= req
        } else if let Some(rest) = c.strip_prefix('>') {
            let Some(req) = Semver::parse(rest) else {
                return false;
            };
            *self > req
        } else if let Some(rest) = c.strip_prefix('<') {
            let Some(req) = Semver::parse(rest) else {
                return false;
            };
            *self < req
        } else if let Some(rest) = c.strip_prefix('=') {
            let Some(req) = Semver::parse(rest) else {
                return false;
            };
            *self == req
        } else if let Some(req) = Semver::parse(c) {
            *self == req
        } else {
            false
        }
    }
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
    #[error("cyclic dependency detected: {path}")]
    CyclicDependency { package: String, path: String },
    #[error("dependency '{name}' version '{found}' does not satisfy constraint '{required}'")]
    UnmetVersionConstraint {
        name: String,
        required: String,
        found: String,
    },
    #[error("version conflict for package '{package}': direct requirement '{direct_requirement}' conflicts with transitive requirement '{transitive_requirement}' requested by '{required_by}'")]
    VersionConflict {
        package: String,
        direct_requirement: String,
        transitive_requirement: String,
        required_by: String,
    },
    #[error("failed to read dependency directory '{path}': {source}")]
    DirectoryRead {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

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
    pub lua_files: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
enum DetectedKind {
    Lust,
    Rust,
    Lua {
        luaopen_symbols: Vec<String>,
        lua_files: Vec<PathBuf>,
    },
}

#[derive(Default)]
struct LibrarySignature {
    luaopen_symbols: Vec<String>,
    has_lust_extension: bool,
}



pub fn resolve_dependencies(
    config: &LustConfig,
    project_dir: &Path,
) -> Result<DependencyResolution, DependencyResolutionError> {
    resolve_transitive_dependencies(config, project_dir)
}

pub fn resolve_transitive_dependencies(
    config: &LustConfig,
    project_dir: &Path,
) -> Result<DependencyResolution, DependencyResolutionError> {
    let manager = PackageManager::new(PackageManager::default_root());
    resolve_dependencies_with_manager(config, project_dir, &manager)
}

pub fn resolve_dependencies_with_manager(
    config: &LustConfig,
    project_dir: &Path,
    manager: &PackageManager,
) -> Result<DependencyResolution, DependencyResolutionError> {
    let mut resolution = DependencyResolution::default();
    manager
        .ensure_layout()
        .map_err(|source| DependencyResolutionError::PackageCache { source })?;

    let mut direct_reqs: HashMap<String, String> = HashMap::new();
    let mut resolved_versions: HashMap<String, (String, String)> = HashMap::new();
    let mut stack: Vec<String> = Vec::new();
    let mut visited: HashSet<String> = HashSet::new();

    for spec in config.dependencies() {
        if let Some(v) = spec.version() {
            direct_reqs.insert(spec.name().to_string(), v.to_string());
        }
    }

    resolve_transitive_recursive(
        config,
        project_dir,
        manager,
        &direct_reqs,
        &mut resolved_versions,
        &mut stack,
        &mut visited,
        &mut resolution,
        true,
    )?;

    Ok(resolution)
}


fn resolve_transitive_recursive(
    config: &LustConfig,
    project_dir: &Path,
    manager: &PackageManager,
    direct_reqs: &HashMap<String, String>,
    resolved_versions: &mut HashMap<String, (String, String)>,
    stack: &mut Vec<String>,
    visited: &mut HashSet<String>,
    resolution: &mut DependencyResolution,
    is_direct: bool,
) -> Result<(), DependencyResolutionError> {
    for spec in config.dependencies() {
        let name = spec.name().to_string();

        if stack.contains(&name) {
            let cycle_path = format!("{} -> {}", stack.join(" -> "), name);
            return Err(DependencyResolutionError::CyclicDependency {
                package: name,
                path: cycle_path,
            });
        }

        let (root_dir, version) = if let Some(path) = spec.path() {
            (resolve_dependency_path(project_dir, path), None)
        } else if let Some(version_req) = spec.version() {
            let pkg_dir = manager.root().join(spec.name());
            if !pkg_dir.exists() {
                return Err(DependencyResolutionError::MissingPackage {
                    name: spec.name().to_string(),
                    version: version_req.to_string(),
                    path: pkg_dir.join(version_req),
                });
            }

            let mut matched_dir = None;
            let mut found_ver_str = None;
            let exact_dir = pkg_dir.join(version_req);
            if exact_dir.exists() {
                matched_dir = Some(exact_dir);
                found_ver_str = Some(version_req.to_string());
            } else if let Ok(entries) = fs::read_dir(&pkg_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        let name_opt = path.file_name().and_then(|n| n.to_str()).map(String::from);
                        if let Some(ver_str) = name_opt {
                            if let Some(parsed) = Semver::parse(&ver_str) {
                                if parsed.matches_constraint(version_req) {
                                    matched_dir = Some(path);
                                    found_ver_str = Some(ver_str);
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            let (dir, found_ver) = match (matched_dir, found_ver_str) {
                (Some(d), Some(v)) => (d, v),
                _ => {
                    if let Ok(mut entries) = fs::read_dir(&pkg_dir) {
                        if let Some(Ok(first)) = entries.next() {
                            let found = first.file_name().to_string_lossy().to_string();
                            return Err(DependencyResolutionError::UnmetVersionConstraint {
                                name: spec.name().to_string(),
                                required: version_req.to_string(),
                                found,
                            });
                        }
                    }
                    return Err(DependencyResolutionError::MissingPackage {
                        name: spec.name().to_string(),
                        version: version_req.to_string(),
                        path: pkg_dir.join(version_req),
                    });
                }
            };

            if !is_direct {
                if let Some(direct_req) = direct_reqs.get(&name) {
                    let direct_conflicts = if let (Some(d_sem), Some(t_sem)) = (Semver::parse(direct_req), Semver::parse(version_req)) {
                        d_sem.major != t_sem.major
                    } else if let Some(found_parsed) = Semver::parse(&found_ver) {
                        !found_parsed.matches_constraint(direct_req)
                    } else {
                        false
                    };

                    if direct_conflicts {
                        let parent = stack.last().cloned().unwrap_or_else(|| "root".to_string());
                        return Err(DependencyResolutionError::VersionConflict {
                            package: name.clone(),
                            direct_requirement: direct_req.clone(),
                            transitive_requirement: version_req.to_string(),
                            required_by: parent,
                        });
                    }
                }
            }

            if let Some((prev_ver, prev_by)) = resolved_versions.get(&name) {
                if prev_ver != &found_ver {
                    return Err(DependencyResolutionError::VersionConflict {
                        package: name.clone(),
                        direct_requirement: prev_ver.clone(),
                        transitive_requirement: found_ver.clone(),
                        required_by: prev_by.clone(),
                    });
                }
            } else {
                let required_by = stack.last().cloned().unwrap_or_else(|| "direct".to_string());
                resolved_versions.insert(name.clone(), (found_ver.clone(), required_by));
            }

            (dir, Some(found_ver))
        } else {
            unreachable!("dependency spec missing path and version");
        };

        if !root_dir.exists() {
            return Err(DependencyResolutionError::MissingPath {
                name: spec.name().to_string(),
                path: root_dir,
            });
        }

        if root_dir.is_dir() {
            if let Err(source) = fs::read_dir(&root_dir) {
                return Err(DependencyResolutionError::DirectoryRead {
                    path: root_dir.clone(),
                    source,
                });
            }
        }

        if !visited.insert(name.clone()) {
            continue;
        }

        let detected = match spec.kind() {
            Some(DependencyKind::Lust) => DetectedKind::Lust,
            Some(DependencyKind::Rust) => DetectedKind::Rust,
            Some(DependencyKind::Lua) => DetectedKind::Lua {
                luaopen_symbols: detect_luaopen_symbols(&root_dir)?,
                lua_files: collect_lua_files(&root_dir),
            },
            None => detect_kind(spec.name(), &root_dir)?,
        };

        match &detected {
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
                    name: name.clone(),
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
                    name: name.clone(),
                    crate_dir: root_dir.clone(),
                    features: spec.features().to_vec(),
                    default_features: spec.default_features().unwrap_or(true),
                    externs_override,
                    cache_stub_dir,
                    version: version.clone(),
                });
            }
            DetectedKind::Lua {
                luaopen_symbols,
                lua_files,
            } => {
                resolution.lua.push(ResolvedLuaDependency {
                    name: name.clone(),
                    library_path: root_dir.clone(),
                    luaopen_symbols: luaopen_symbols.clone(),
                    version: version.clone(),
                    lua_files: lua_files.clone(),
                });
            }
        }

        if root_dir.is_dir() {
            let child_manifest = root_dir.join("lust-config.toml");
            if child_manifest.exists() {
                if let Ok(child_cfg) = LustConfig::load_from_path(&child_manifest) {
                    stack.push(name.clone());
                    resolve_transitive_recursive(
                        &child_cfg,
                        &root_dir,
                        manager,
                        direct_reqs,
                        resolved_versions,
                        stack,
                        visited,
                        resolution,
                        false,
                    )?;
                    stack.pop();
                }
            }
        }
    }

    Ok(())
}

fn detect_kind(name: &str, root: &Path) -> Result<DetectedKind, DependencyResolutionError> {
    if root.is_file() {
        let signature = inspect_library(root)?;
        let luaopen_symbols = signature.luaopen_symbols;
        let has_lust_register = signature.has_lust_extension;
        return if !luaopen_symbols.is_empty() {
            Ok(DetectedKind::Lua {
                luaopen_symbols,
                lua_files: Vec::new(),
            })
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
            } else if has_lua_files(root) {
                Ok(DetectedKind::Lua {
                    luaopen_symbols: Vec::new(),
                    lua_files: collect_lua_files(root),
                })
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

#[allow(dead_code)]
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
    let file =
        File::parse(&*bytes).map_err(|source| DependencyResolutionError::LibraryInspect {
            path: path.to_path_buf(),
            source,
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

fn has_lua_files(root: &Path) -> bool {
    !collect_lua_files(root).is_empty()
}

fn collect_lua_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_lua_files_recursive(root, root, &mut files);
    files
}

fn collect_lua_files_recursive(base: &Path, current: &Path, files: &mut Vec<PathBuf>) {
    if let Ok(read_dir) = fs::read_dir(current) {
        for entry in read_dir.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_lua_files_recursive(base, &path, files);
            } else if path.extension().and_then(|s| s.to_str()) == Some("lua")
                && let Ok(relative) = path.strip_prefix(base)
            {
                files.push(relative.to_path_buf());
            }
        }
    }
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

use super::{
    manifest::{ManifestError, ManifestKind, PackageManifest},
    PackageManager,
};
use crate::config::{DependencyKind, LustConfig};
use std::{
    io,
    path::{Path, PathBuf},
};
use thiserror::Error;

#[derive(Debug, Default, Clone)]
pub struct DependencyResolution {
    lust: Vec<ResolvedLustDependency>,
    rust: Vec<ResolvedRustDependency>,
}

impl DependencyResolution {
    pub fn lust(&self) -> &[ResolvedLustDependency] {
        &self.lust
    }

    pub fn rust(&self) -> &[ResolvedRustDependency] {
        &self.rust
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

        let kind = match spec.kind() {
            Some(kind) => kind,
            None => detect_kind(spec.name(), &root_dir)?,
        };

        match kind {
            DependencyKind::Lust => {
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
            DependencyKind::Rust => {
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
        }
    }

    Ok(resolution)
}

fn detect_kind(name: &str, root: &Path) -> Result<DependencyKind, DependencyResolutionError> {
    match PackageManifest::discover(root) {
        Ok(manifest) => match manifest.kind() {
            ManifestKind::Lust => Ok(DependencyKind::Lust),
            ManifestKind::Cargo => Ok(DependencyKind::Rust),
        },
        Err(ManifestError::NotFound(_)) => {
            if root.join("Cargo.toml").exists() {
                Ok(DependencyKind::Rust)
            } else {
                Ok(DependencyKind::Lust)
            }
        }
        Err(err) => Err(DependencyResolutionError::Manifest {
            name: name.to_string(),
            source: err,
        }),
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

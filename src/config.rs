use alloc::{string::String, vec::Vec};
use hashbrown::HashSet;
#[cfg(feature = "std")]
use serde::Deserialize;
#[cfg(feature = "std")]
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[cfg(feature = "std")]
    #[error("failed to read configuration: {0}")]
    Io(#[from] std::io::Error),
    #[cfg(feature = "std")]
    #[error("failed to parse configuration: {0}")]
    Parse(#[from] toml::de::Error),
    #[cfg(feature = "std")]
    #[error("dependency '{0}' must specify either a version or a path")]
    MissingDependencySource(String),
    #[cfg(feature = "std")]
    #[error("dependency '{0}' has unknown kind '{1}'")]
    UnknownDependencyKind(String, String),
    #[error("{0}")]
    Unsupported(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependencyKind {
    Lust,
    Rust,
    Lua,
}

#[derive(Debug, Clone)]
pub struct DependencySpec {
    name: String,
    version: Option<String>,
    path: Option<String>,
    kind: Option<DependencyKind>,
    features: Vec<String>,
    default_features: Option<bool>,
    externs: Option<String>,
    legacy: bool,
}

impl DependencySpec {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }

    pub fn path(&self) -> Option<&str> {
        self.path.as_deref()
    }

    pub fn kind(&self) -> Option<DependencyKind> {
        self.kind
    }

    pub fn features(&self) -> &[String] {
        &self.features
    }

    pub fn default_features(&self) -> Option<bool> {
        self.default_features
    }

    pub fn externs(&self) -> Option<&str> {
        self.externs.as_deref()
    }

    pub fn is_legacy(&self) -> bool {
        self.legacy
    }
}

#[derive(Debug, Clone)]
pub struct LustConfig {
    enabled_modules: HashSet<String>,
    jit_enabled: bool,
    /// When enabled, minimizes memory usage during compilation by:
    /// - Not storing expression types and variable types
    /// - Not storing short-circuit type flow analysis
    /// - Clearing typechecker data immediately after use
    /// Intended for constrained no_std environments like ESP32.
    low_memory_mode: bool,
    /// When enabled, strips detailed type information from compiled functions:
    /// - Removes register type info from Function objects
    /// - Reduces runtime memory at cost of less type safety/debug info
    minimal_runtime_types: bool,
    #[cfg(feature = "std")]
    dependencies: Vec<DependencySpec>,
}

impl Default for LustConfig {
    fn default() -> Self {
        Self {
            enabled_modules: HashSet::new(),
            jit_enabled: true,
            low_memory_mode: false,
            minimal_runtime_types: false,
            #[cfg(feature = "std")]
            dependencies: Vec::new(),
        }
    }
}

impl LustConfig {
    #[cfg(feature = "std")]
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let path_ref = path.as_ref();
        let content = fs::read_to_string(path_ref)?;
        let parsed: LustConfigToml = toml::from_str(&content)?;
        Self::from_parsed(parsed, path_ref.parent())
    }

    #[cfg(feature = "std")]
    pub fn from_toml_str(source: &str) -> Result<Self, ConfigError> {
        let parsed: LustConfigToml = toml::from_str(source)?;
        Self::from_parsed(parsed, None)
    }

    #[cfg(feature = "std")]
    pub fn load_from_dir<P: AsRef<Path>>(dir: P) -> Result<Self, ConfigError> {
        let mut path = PathBuf::from(dir.as_ref());
        path.push("lust-config.toml");
        if !path.exists() {
            return Ok(Self::default());
        }

        Self::load_from_path(path)
    }

    #[cfg(feature = "std")]
    pub fn load_for_entry<P: AsRef<Path>>(entry_file: P) -> Result<Self, ConfigError> {
        let entry_path = entry_file.as_ref();
        let dir = entry_path.parent().unwrap_or_else(|| Path::new("."));
        Self::load_from_dir(dir)
    }

    pub fn jit_enabled(&self) -> bool {
        self.jit_enabled
    }

    pub fn is_module_enabled(&self, module: &str) -> bool {
        let key = module.to_ascii_lowercase();
        self.enabled_modules.contains(&key)
    }

    pub fn enabled_modules(&self) -> impl Iterator<Item = &str> {
        self.enabled_modules.iter().map(|s| s.as_str())
    }

    pub fn enable_module<S: AsRef<str>>(&mut self, module: S) {
        let key = module.as_ref().trim().to_ascii_lowercase();
        if !key.is_empty() {
            self.enabled_modules.insert(key);
        }
    }

    pub fn set_jit_enabled(&mut self, enabled: bool) {
        self.jit_enabled = enabled;
    }

    pub fn low_memory_mode(&self) -> bool {
        self.low_memory_mode
    }

    pub fn set_low_memory_mode(&mut self, enabled: bool) {
        self.low_memory_mode = enabled;
    }

    pub fn minimal_runtime_types(&self) -> bool {
        self.minimal_runtime_types
    }

    pub fn set_minimal_runtime_types(&mut self, enabled: bool) {
        self.minimal_runtime_types = enabled;
    }

    /// Builder-style method to enable low memory mode
    pub fn with_low_memory_mode(mut self, enabled: bool) -> Self {
        self.low_memory_mode = enabled;
        self
    }

    /// Builder-style method to enable minimal runtime types
    pub fn with_minimal_runtime_types(mut self, enabled: bool) -> Self {
        self.minimal_runtime_types = enabled;
        self
    }

    pub fn with_enabled_modules<I, S>(modules: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut config = Self::default();
        for module in modules {
            config.enable_module(module);
        }

        config
    }

    #[cfg(feature = "std")]
    pub fn dependencies(&self) -> &[DependencySpec] {
        &self.dependencies
    }

    #[cfg(feature = "std")]
    fn from_parsed(parsed: LustConfigToml, _base_dir: Option<&Path>) -> Result<Self, ConfigError> {
        let LustConfigToml {
            settings,
            dependencies: mut root_dependencies,
        } = parsed;
        let Settings {
            stdlib_modules,
            jit,
            rust_modules,
            dependencies: nested_dependencies,
        } = settings;

        let modules = stdlib_modules
            .into_iter()
            .map(|m| m.trim().to_ascii_lowercase())
            .filter(|m| !m.is_empty())
            .collect::<HashSet<_>>();

        for (name, entry) in nested_dependencies {
            root_dependencies.insert(name, entry);
        }

        let mut dependencies = Vec::new();
        for (name, entry) in root_dependencies {
            let (version, path, kind, features, default_features, externs) = match entry {
                DependencyToml::Version(version) => {
                    (Some(version), None, None, Vec::new(), None, None)
                }
                DependencyToml::Detailed(table) => {
                    let kind = match table.kind {
                        Some(raw) => match raw.trim().to_ascii_lowercase().as_str() {
                            "lust" => Some(DependencyKind::Lust),
                            "rust" => Some(DependencyKind::Rust),
                            "lua" | "lua51" | "lua_compat" => Some(DependencyKind::Lua),
                            other => {
                                return Err(ConfigError::UnknownDependencyKind(
                                    name.clone(),
                                    other.to_string(),
                                ))
                            }
                        },
                        None => None,
                    };
                    (
                        table.version,
                        table.path,
                        kind,
                        table.features,
                        table.default_features,
                        table.externs,
                    )
                }
            };
            let has_path = path.as_ref().map(|p| !p.trim().is_empty()).unwrap_or(false);
            if version.is_none() && !has_path {
                return Err(ConfigError::MissingDependencySource(name));
            }
            dependencies.push(DependencySpec {
                name,
                version,
                path,
                kind,
                features,
                default_features,
                externs,
                legacy: false,
            });
        }

        for legacy in rust_modules {
            let inferred_name = Path::new(&legacy.path)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(&legacy.path)
                .to_string();
            dependencies.push(DependencySpec {
                name: inferred_name,
                version: None,
                path: Some(legacy.path),
                kind: Some(DependencyKind::Rust),
                features: Vec::new(),
                default_features: None,
                externs: legacy.externs,
                legacy: true,
            });
        }

        Ok(Self {
            enabled_modules: modules,
            jit_enabled: jit,
            low_memory_mode: false,
            minimal_runtime_types: false,
            dependencies,
        })
    }
}

#[cfg(feature = "std")]
#[derive(Debug, Deserialize)]
struct LustConfigToml {
    #[serde(default)]
    settings: Settings,
    #[serde(default)]
    dependencies: BTreeMap<String, DependencyToml>,
}

#[cfg(feature = "std")]
#[derive(Debug, Default, Deserialize)]
struct Settings {
    #[serde(default)]
    stdlib_modules: Vec<String>,
    #[serde(default = "default_jit_enabled")]
    jit: bool,
    #[serde(default)]
    rust_modules: Vec<RustModuleEntry>,
    #[serde(default)]
    dependencies: BTreeMap<String, DependencyToml>,
}

#[cfg(feature = "std")]
#[derive(Debug, Deserialize)]
struct RustModuleEntry {
    path: String,
    #[serde(default)]
    externs: Option<String>,
}

#[cfg(feature = "std")]
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum DependencyToml {
    Version(String),
    Detailed(DependencyTomlTable),
}

#[cfg(feature = "std")]
#[derive(Debug, Default, Deserialize)]
struct DependencyTomlTable {
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    features: Vec<String>,
    #[serde(default)]
    default_features: Option<bool>,
    #[serde(default)]
    externs: Option<String>,
}

#[allow(dead_code)]
const fn default_jit_enabled() -> bool {
    true
}

#[cfg(feature = "std")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_jit_enabled() {
        let cfg = LustConfig::default();
        assert!(cfg.jit_enabled());
        assert!(cfg.enabled_modules().next().is_none());
        assert!(!cfg.low_memory_mode());
        assert!(!cfg.minimal_runtime_types());
    }

    #[test]
    fn parse_config_with_modules_and_jit() {
        let toml = r#"
            [settings]
            stdlib_modules = ["io", "os"]
            jit = false
        "#;
        let parsed: LustConfigToml = toml::from_str(toml).unwrap();
        let cfg = LustConfig::from_parsed(parsed, None).unwrap();
        assert!(!cfg.jit_enabled());
        assert!(cfg.is_module_enabled("io"));
        assert!(cfg.is_module_enabled("os"));
    }

    #[test]
    fn dependencies_parse_version() {
        let toml = r#"
            [dependencies]
            foo = "1.2.3"
        "#;
        let parsed: LustConfigToml = toml::from_str(toml).unwrap();
        let cfg = LustConfig::from_parsed(parsed, None).unwrap();
        let deps = cfg.dependencies();
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].name(), "foo");
        assert_eq!(deps[0].version(), Some("1.2.3"));
        assert!(deps[0].path().is_none());
    }

    #[test]
    fn settings_dependencies_still_supported() {
        let toml = r#"
            [settings]
            [settings.dependencies]
            bar = { path = "ext/bar", kind = "rust" }
        "#;
        let parsed: LustConfigToml = toml::from_str(toml).unwrap();
        let cfg = LustConfig::from_parsed(parsed, None).unwrap();
        let deps = cfg.dependencies();
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].name(), "bar");
        assert_eq!(deps[0].path(), Some("ext/bar"));
        assert_eq!(deps[0].kind(), Some(DependencyKind::Rust));
    }

    #[test]
    fn settings_dependencies_override_top_level() {
        let toml = r#"
            [dependencies]
            baz = { path = "ext/baz" }

            [settings]
            [settings.dependencies]
            baz = { version = "1.2.3" }
        "#;
        let parsed: LustConfigToml = toml::from_str(toml).unwrap();
        let cfg = LustConfig::from_parsed(parsed, None).unwrap();
        let deps = cfg.dependencies();
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].name(), "baz");
        assert_eq!(deps[0].version(), Some("1.2.3"));
        assert!(deps[0].path().is_none());
    }

    #[test]
    fn legacy_rust_modules_are_mapped_to_dependencies() {
        let toml = r#"
            [settings]
            rust_modules = [
                { path = "ext/foo", externs = "externs" },
                { path = "/absolute/bar" }
            ]
        "#;
        let parsed: LustConfigToml = toml::from_str(toml).unwrap();
        let cfg = LustConfig::from_parsed(parsed, None).unwrap();
        let deps = cfg.dependencies();
        assert_eq!(deps.len(), 2);
        assert_eq!(deps[0].path(), Some("ext/foo"));
        assert_eq!(deps[0].externs(), Some("externs"));
        assert_eq!(deps[0].kind(), Some(DependencyKind::Rust));
        assert!(deps[0].is_legacy());
        assert_eq!(deps[1].path(), Some("/absolute/bar"));
        assert!(deps[1].externs().is_none());
    }
}

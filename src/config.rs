use alloc::string::String;
#[cfg(feature = "std")]
use alloc::vec::Vec;
use hashbrown::HashSet;
#[cfg(feature = "std")]
use serde::Deserialize;
#[cfg(feature = "std")]
use std::{
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
    #[error("{0}")]
    Unsupported(String),
}

#[derive(Debug, Clone)]
pub struct LustConfig {
    enabled_modules: HashSet<String>,
    jit_enabled: bool,
    #[cfg(feature = "std")]
    rust_modules: Vec<RustModule>,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct RustModule {
    path: PathBuf,
    externs: Option<PathBuf>,
}

#[cfg(feature = "std")]
#[derive(Debug, Deserialize)]
struct LustConfigToml {
    settings: Settings,
}

#[cfg(feature = "std")]
#[derive(Debug, Deserialize)]
struct Settings {
    #[serde(default)]
    stdlib_modules: Vec<String>,
    #[serde(default = "default_jit_enabled")]
    jit: bool,
    #[serde(default)]
    rust_modules: Vec<RustModuleEntry>,
}

#[cfg(feature = "std")]
#[derive(Debug, Deserialize)]
struct RustModuleEntry {
    path: String,
    #[serde(default)]
    externs: Option<String>,
}

const fn default_jit_enabled() -> bool {
    true
}

impl Default for LustConfig {
    fn default() -> Self {
        Self {
            enabled_modules: HashSet::new(),
            jit_enabled: true,
            #[cfg(feature = "std")]
            rust_modules: Vec::new(),
        }
    }
}

impl LustConfig {
    #[cfg(feature = "std")]
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let path_ref = path.as_ref();
        let content = fs::read_to_string(path_ref)?;
        let parsed: LustConfigToml = toml::from_str(&content)?;
        Ok(Self::from_parsed(parsed, path_ref.parent()))
    }

    #[cfg(feature = "std")]
    pub fn from_toml_str(source: &str) -> Result<Self, ConfigError> {
        let parsed: LustConfigToml = toml::from_str(source)?;
        Ok(Self::from_parsed(parsed, None))
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
    pub fn rust_modules(&self) -> impl Iterator<Item = &RustModule> {
        self.rust_modules.iter()
    }

    #[cfg(feature = "std")]
    fn from_parsed(parsed: LustConfigToml, base_dir: Option<&Path>) -> Self {
        let modules = parsed
            .settings
            .stdlib_modules
            .into_iter()
            .map(|m| m.trim().to_ascii_lowercase())
            .filter(|m| !m.is_empty())
            .collect::<HashSet<_>>();
        let rust_modules = parsed
            .settings
            .rust_modules
            .into_iter()
            .map(|entry| {
                let path = match base_dir {
                    Some(root) => root.join(&entry.path),
                    None => PathBuf::from(&entry.path),
                };
                let externs = entry.externs.map(PathBuf::from);
                RustModule { path, externs }
            })
            .collect();
        Self {
            enabled_modules: modules,
            jit_enabled: parsed.settings.jit,
            rust_modules,
        }
    }
}

#[cfg(feature = "std")]
impl RustModule {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn externs(&self) -> Option<&Path> {
        self.externs.as_deref()
    }

    pub fn externs_dir(&self) -> Option<PathBuf> {
        self.externs.as_ref().map(|path| {
            if path.is_absolute() {
                path.clone()
            } else {
                self.path.join(path)
            }
        })
    }
}

#[cfg(feature = "std")]
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    #[test]
    fn default_config_has_jit_enabled() {
        let cfg = LustConfig::default();
        assert!(cfg.jit_enabled());
        assert!(cfg.enabled_modules().next().is_none());
    }

    #[test]
    fn parse_config_with_modules_and_jit() {
        let toml = r#"
            "enabled modules" = ["io", "OS", "  task  "]
            jit = false
        "#;
        let parsed: LustConfigToml = toml::from_str(toml).unwrap();
        let cfg = LustConfig::from_parsed(parsed, None);
        assert!(!cfg.jit_enabled());
        assert!(cfg.is_module_enabled("io"));
        assert!(cfg.is_module_enabled("os"));
        assert!(cfg.is_module_enabled("task"));
        assert!(!cfg.is_module_enabled("math"));
    }

    #[test]
    fn rust_modules_are_resolved_relative_to_config() {
        let toml = r#"
            [settings]
            rust_modules = [
                { path = "ext/foo", externs = "externs" },
                { path = "/absolute/bar" }
            ]
        "#;
        let parsed: LustConfigToml = toml::from_str(toml).unwrap();
        let base = PathBuf::from("/var/project");
        let cfg = LustConfig::from_parsed(parsed, Some(base.as_path()));
        let modules: Vec<&RustModule> = cfg.rust_modules().collect();
        assert_eq!(modules.len(), 2);
        assert_eq!(modules[0].path(), Path::new("/var/project/ext/foo"));
        assert_eq!(
            modules[0].externs_dir(),
            Some(PathBuf::from("/var/project/ext/foo/externs"))
        );
        assert_eq!(modules[1].path(), Path::new("/absolute/bar"));
        assert!(modules[1].externs().is_none());
    }
}

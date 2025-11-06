use serde::Deserialize;
use serde_json::{json, Map as JsonMap, Value as JsonValue};
use std::{
    fs,
    path::{Path, PathBuf},
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("no manifest found in {0}")]
    NotFound(PathBuf),

    #[error("failed to read manifest {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse manifest {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    #[error("manifest {0} missing [package] section")]
    MissingPackageSection(PathBuf),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestKind {
    Lust,
    Cargo,
}

#[derive(Debug, Clone)]
pub struct PackageManifest {
    kind: ManifestKind,
    manifest_path: PathBuf,
    root: PathBuf,
    package: PackageSection,
}

#[derive(Debug, Clone)]
pub struct PackageSection {
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub keywords: Vec<String>,
    pub categories: Vec<String>,
    pub repository: Option<String>,
    pub license: Option<String>,
    pub readme: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LustManifestToml {
    #[serde(default)]
    package: Option<GenericPackageSection>,
}

#[derive(Debug, Deserialize)]
struct CargoManifestToml {
    package: Option<CargoPackageSection>,
}

#[derive(Debug, Deserialize)]
struct GenericPackageSection {
    name: String,
    version: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    keywords: Option<Vec<String>>,
    #[serde(default)]
    categories: Option<Vec<String>>,
    #[serde(default)]
    repository: Option<String>,
    #[serde(default)]
    license: Option<String>,
    #[serde(default)]
    readme: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CargoPackageSection {
    name: String,
    version: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    keywords: Option<Vec<String>>,
    #[serde(default)]
    categories: Option<Vec<String>>,
    #[serde(default)]
    repository: Option<String>,
    #[serde(default)]
    license: Option<String>,
    #[serde(default)]
    readme: Option<CargoReadmeField>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CargoReadmeField {
    Bool(bool),
    String(String),
}

impl PackageManifest {
    pub fn discover(path: &Path) -> Result<Self, ManifestError> {
        if path.is_file() {
            return Self::from_manifest(path);
        }
        if path.is_dir() {
            let candidate = path.join("lust-config.toml");
            if candidate.exists() {
                return Self::from_manifest(&candidate);
            }
            let candidate = path.join("Cargo.toml");
            if candidate.exists() {
                return Self::from_manifest(&candidate);
            }
            return Err(ManifestError::NotFound(path.to_path_buf()));
        }
        Err(ManifestError::NotFound(path.to_path_buf()))
    }

    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn kind(&self) -> ManifestKind {
        self.kind
    }

    pub fn package(&self) -> &PackageSection {
        &self.package
    }

    pub fn readme_path(&self) -> Option<PathBuf> {
        self.package
            .readme
            .as_ref()
            .map(|relative| self.root.join(relative))
    }

    pub fn metadata_payload(&self) -> JsonValue {
        let pkg = &self.package;
        let mut package = JsonMap::new();
        package.insert("name".to_string(), json!(pkg.name));
        package.insert("version".to_string(), json!(pkg.version));
        if let Some(desc) = &pkg.description {
            package.insert("description".to_string(), json!(desc));
        }
        if !pkg.keywords.is_empty() {
            package.insert("keywords".to_string(), json!(pkg.keywords));
        }
        if !pkg.categories.is_empty() {
            package.insert("categories".to_string(), json!(pkg.categories));
        }
        if let Some(repo) = &pkg.repository {
            package.insert("repository".to_string(), json!(repo));
        }
        if let Some(license) = &pkg.license {
            package.insert("license".to_string(), json!(license));
        }

        json!({ "package": package })
    }

    fn from_manifest(path: &Path) -> Result<Self, ManifestError> {
        let manifest_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let content = fs::read_to_string(path).map_err(|source| ManifestError::Io {
            path: manifest_path.clone(),
            source,
        })?;
        let parent = manifest_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        if path
            .file_name()
            .map(|name| name == "lust-config.toml")
            .unwrap_or(false)
        {
            let parsed: LustManifestToml =
                toml::from_str(&content).map_err(|source| ManifestError::Parse {
                    path: manifest_path.clone(),
                    source,
                })?;
            let section = parsed
                .package
                .ok_or_else(|| ManifestError::MissingPackageSection(manifest_path.clone()))?;
            let package = PackageSection::from_generic(section);
            return Ok(Self {
                kind: ManifestKind::Lust,
                manifest_path,
                root: parent,
                package,
            });
        }

        if path
            .file_name()
            .map(|name| name == "Cargo.toml")
            .unwrap_or(false)
        {
            let parsed: CargoManifestToml =
                toml::from_str(&content).map_err(|source| ManifestError::Parse {
                    path: manifest_path.clone(),
                    source,
                })?;
            let section = parsed
                .package
                .ok_or_else(|| ManifestError::MissingPackageSection(manifest_path.clone()))?;
            let package = PackageSection::from_cargo(section);
            return Ok(Self {
                kind: ManifestKind::Cargo,
                manifest_path,
                root: parent,
                package,
            });
        }

        Err(ManifestError::NotFound(manifest_path))
    }
}

impl PackageSection {
    fn from_generic(src: GenericPackageSection) -> Self {
        Self {
            name: src.name,
            version: src.version,
            description: trim_opt_string(src.description),
            keywords: normalize_vec(src.keywords),
            categories: normalize_vec(src.categories),
            repository: trim_opt_string(src.repository),
            license: trim_opt_string(src.license),
            readme: trim_opt_string(src.readme),
        }
    }

    fn from_cargo(src: CargoPackageSection) -> Self {
        let readme = match src.readme {
            Some(CargoReadmeField::String(path)) => Some(path),
            Some(CargoReadmeField::Bool(true)) => Some("README.md".to_string()),
            _ => None,
        };
        Self {
            name: src.name,
            version: src.version,
            description: trim_opt_string(src.description),
            keywords: normalize_vec(src.keywords),
            categories: normalize_vec(src.categories),
            repository: trim_opt_string(src.repository),
            license: trim_opt_string(src.license),
            readme: trim_opt_string(readme),
        }
    }
}

fn normalize_vec(values: Option<Vec<String>>) -> Vec<String> {
    values
        .unwrap_or_default()
        .into_iter()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .collect()
}

fn trim_opt_string(input: Option<String>) -> Option<String> {
    input
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn load_lust_manifest_prefers_lust_config() {
        let dir = tempdir().unwrap();
        let lust_manifest = dir.path().join("lust-config.toml");
        fs::write(
            &lust_manifest,
            r#"
[package]
name = "example"
version = "0.1.0"
description = "Example package"
keywords = ["foo", "bar"]
categories = ["cat1"]
repository = "https://example.com"
license = "MIT"
readme = "README.md"
"#,
        )
        .unwrap();
        fs::write(dir.path().join("Cargo.toml"), "").unwrap();

        let manifest = PackageManifest::discover(dir.path()).unwrap();
        assert_eq!(manifest.kind(), ManifestKind::Lust);
        assert_eq!(manifest.package().name, "example");
        assert_eq!(manifest.package().version, "0.1.0");
        assert_eq!(manifest.package().keywords, vec!["foo", "bar"]);
        assert_eq!(manifest.package().categories, vec!["cat1"]);
        assert_eq!(
            manifest.readme_path().unwrap(),
            dir.path().join("README.md")
        );
    }

    #[test]
    fn load_cargo_manifest_falls_back() {
        let dir = tempdir().unwrap();
        let mut cargo = fs::File::create(dir.path().join("Cargo.toml")).unwrap();
        writeln!(
            cargo,
            r#"[package]
name = "crate"
version = "1.2.3"
keywords = ["k1", "k2"]
categories = ["c1"]
repository = "https://repo"
license = "Apache-2.0"
readme = true
"#
        )
        .unwrap();

        let manifest = PackageManifest::discover(dir.path()).unwrap();
        assert_eq!(manifest.kind(), ManifestKind::Cargo);
        assert_eq!(manifest.package().name, "crate");
        assert_eq!(manifest.package().version, "1.2.3");
        assert_eq!(manifest.package().keywords, vec!["k1", "k2"]);
        assert_eq!(
            manifest.readme_path().unwrap(),
            dir.path().join("README.md")
        );
    }

    #[test]
    fn metadata_payload_contains_expected_fields() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("lust-config.toml"),
            r#"[package]
name = "pkg"
version = "0.0.1"
keywords = ["alpha"]
description = "desc"
"#,
        )
        .unwrap();
        let manifest = PackageManifest::discover(dir.path()).unwrap();
        let payload = manifest.metadata_payload();
        assert_eq!(payload["package"]["name"], "pkg");
        assert_eq!(payload["package"]["version"], "0.0.1");
        assert_eq!(payload["package"]["description"], "desc");
        assert_eq!(payload["package"]["keywords"][0], "alpha");
    }
}

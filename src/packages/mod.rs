use crate::{NativeExport, VM};
use libloading::Library;
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashMap},
    env,
    ffi::OsStr,
    fs, io,
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
    sync::{Mutex, OnceLock},
};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageKind {
    LustLibrary,
    RustExtension,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageSpecifier {
    pub name: String,
    pub version: Option<String>,
    pub kind: PackageKind,
}

impl PackageSpecifier {
    pub fn new(name: impl Into<String>, kind: PackageKind) -> Self {
        Self {
            name: name.into(),
            version: None,
            kind,
        }
    }

    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }
}

pub struct PackageManager {
    root: PathBuf,
}

impl PackageManager {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn default_root() -> PathBuf {
        let mut base = dirs::data_local_dir().unwrap_or_else(|| PathBuf::from(".lust/cache"));
        base.push("lust");
        base.push("packages");
        base
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn ensure_layout(&self) -> io::Result<()> {
        fs::create_dir_all(&self.root)
    }
}

#[derive(Debug, Error)]
pub enum LocalModuleError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("failed to parse Cargo manifest {path}: {source}")]
    Manifest {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    #[error("cargo build failed for '{module}' with status {status}: {output}")]
    CargoBuild {
        module: String,
        status: ExitStatus,
        output: String,
    },

    #[error("built library not found at {0}")]
    LibraryMissing(PathBuf),

    #[error("failed to load dynamic library: {0}")]
    LibraryLoad(#[from] libloading::Error),

    #[error("register function 'lust_extension_register' missing in {0}")]
    RegisterSymbolMissing(String),

    #[error("register function reported failure in {0}")]
    RegisterFailed(String),
}

#[derive(Debug, Clone)]
pub struct LocalBuildOutput {
    pub name: String,
    pub library_path: PathBuf,
}

#[derive(Debug)]
pub struct LoadedRustModule {
    name: String,
}

impl LoadedRustModule {
    pub fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug, Clone)]
pub struct StubFile {
    pub relative_path: PathBuf,
    pub contents: String,
}

pub fn build_local_module(module_dir: &Path) -> Result<LocalBuildOutput, LocalModuleError> {
    let crate_name = read_crate_name(module_dir)?;
    let profile = extension_profile();
    let mut command = Command::new("cargo");
    command.arg("build");
    command.arg("--quiet");
    match profile.as_str() {
        "release" => {
            command.arg("--release");
        }
        "debug" => {}
        other => {
            command.args(["--profile", other]);
        }
    }
    command.current_dir(module_dir);
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let output = command.output()?;
    if !output.status.success() {
        let mut message = String::new();
        if !output.stdout.is_empty() {
            message.push_str(&String::from_utf8_lossy(&output.stdout));
        }
        if !output.stderr.is_empty() {
            if !message.is_empty() {
                message.push('\n');
            }
            message.push_str(&String::from_utf8_lossy(&output.stderr));
        }
        return Err(LocalModuleError::CargoBuild {
            module: crate_name,
            status: output.status,
            output: message,
        });
    }

    let artifact = module_dir
        .join("target")
        .join(&profile)
        .join(library_file_name(&crate_name));
    if !artifact.exists() {
        return Err(LocalModuleError::LibraryMissing(artifact));
    }

    Ok(LocalBuildOutput {
        name: crate_name,
        library_path: artifact,
    })
}

pub fn load_local_module(
    vm: &mut VM,
    build: &LocalBuildOutput,
) -> Result<LoadedRustModule, LocalModuleError> {
    let library = get_or_load_library(&build.library_path)?;
    unsafe {
        let register = library
            .get::<unsafe extern "C" fn(*mut VM) -> bool>(b"lust_extension_register\0")
            .map_err(|_| LocalModuleError::RegisterSymbolMissing(build.name.clone()))?;
        vm.push_export_prefix(&build.name);
        let success = register(vm as *mut VM);
        vm.pop_export_prefix();
        if !success {
            return Err(LocalModuleError::RegisterFailed(build.name.clone()));
        }
    }
    Ok(LoadedRustModule {
        name: build.name.clone(),
    })
}

pub fn collect_stub_files(
    module_dir: &Path,
    override_dir: Option<&Path>,
) -> Result<Vec<StubFile>, LocalModuleError> {
    let base_dir = match override_dir {
        Some(dir) => dir.to_path_buf(),
        None => module_dir.join("externs"),
    };
    if !base_dir.exists() {
        return Ok(Vec::new());
    }

    let mut stubs = Vec::new();
    visit_stub_dir(&base_dir, PathBuf::new(), &mut stubs)?;
    Ok(stubs)
}

pub fn write_stub_files(
    crate_name: &str,
    stubs: &[StubFile],
    output_root: &Path,
) -> Result<Vec<PathBuf>, LocalModuleError> {
    let mut written = Vec::new();
    for stub in stubs {
        let mut relative = if stub.relative_path.components().next().is_some() {
            stub.relative_path.clone()
        } else {
            let mut path = PathBuf::new();
            path.push(sanitized_crate_name(crate_name));
            path.set_extension("lust");
            path
        };
        if relative.extension().is_none() {
            relative.set_extension("lust");
        }
        let destination = output_root.join(&relative);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&destination, &stub.contents)?;
        written.push(relative);
    }

    Ok(written)
}

fn extension_profile() -> String {
    env::var("LUST_EXTENSION_PROFILE").unwrap_or_else(|_| "release".to_string())
}

fn library_file_name(crate_name: &str) -> String {
    let sanitized = sanitized_crate_name(crate_name);
    #[cfg(target_os = "windows")]
    {
        format!("{sanitized}.dll")
    }
    #[cfg(target_os = "macos")]
    {
        format!("lib{sanitized}.dylib")
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        format!("lib{sanitized}.so")
    }
}

fn sanitized_crate_name(name: &str) -> String {
    name.replace('-', "_")
}

fn library_cache() -> &'static Mutex<HashMap<PathBuf, &'static Library>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, &'static Library>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_or_load_library(path: &Path) -> Result<&'static Library, LocalModuleError> {
    {
        let cache = library_cache().lock().unwrap();
        if let Some(lib) = cache.get(path) {
            return Ok(*lib);
        }
    }

    let library = unsafe { Library::new(path) }.map_err(LocalModuleError::LibraryLoad)?;
    let leaked = Box::leak(Box::new(library));

    let mut cache = library_cache().lock().unwrap();
    let entry = cache.entry(path.to_path_buf()).or_insert(leaked);
    Ok(*entry)
}

fn visit_stub_dir(
    current: &Path,
    relative: PathBuf,
    stubs: &mut Vec<StubFile>,
) -> Result<(), LocalModuleError> {
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        let next_relative = relative.join(entry.file_name());
        if path.is_dir() {
            visit_stub_dir(&path, next_relative, stubs)?;
        } else if path.extension() == Some(OsStr::new("lust")) {
            let contents = fs::read_to_string(&path)?;
            stubs.push(StubFile {
                relative_path: next_relative,
                contents,
            });
        }
    }

    Ok(())
}

pub fn stub_files_from_exports(exports: &[NativeExport]) -> Vec<StubFile> {
    if exports.is_empty() {
        return Vec::new();
    }

    let mut grouped: BTreeMap<String, Vec<&NativeExport>> = BTreeMap::new();
    for export in exports {
        let (module, _function) = match export.name().rsplit_once('.') {
            Some(parts) => parts,
            None => continue,
        };
        grouped.entry(module.to_string()).or_default().push(export);
    }

    let mut result = Vec::new();
    for (module, mut items) in grouped {
        items.sort_by(|a, b| a.name().cmp(b.name()));
        let mut contents = String::new();
        contents.push_str("pub extern {\n");
        for export in items {
            if let Some((_, function)) = export.name().rsplit_once('.') {
                let params = format_params(export);
                let return_type = export.return_type();
                if let Some(doc) = export.doc() {
                    contents.push_str("    -- ");
                    contents.push_str(doc);
                    if !doc.ends_with('\n') {
                        contents.push('\n');
                    }
                }
                contents.push_str("    function ");
                contents.push_str(function);
                contents.push('(');
                contents.push_str(&params);
                contents.push(')');
                if !return_type.trim().is_empty() && return_type.trim() != "()" {
                    contents.push_str(": ");
                    contents.push_str(return_type);
                }
                contents.push('\n');
            }
        }
        contents.push_str("}\n");
        let mut relative = relative_stub_path(&module);
        if relative.extension().is_none() {
            relative.set_extension("lust");
        }
        result.push(StubFile {
            relative_path: relative,
            contents,
        });
    }

    result
}

fn format_params(export: &NativeExport) -> String {
    export
        .params()
        .iter()
        .map(|param| {
            let ty = param.ty().trim();
            if ty.is_empty() {
                "any"
            } else {
                ty
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn relative_stub_path(module: &str) -> PathBuf {
    let mut path = PathBuf::new();
    let mut segments: Vec<String> = module.split('.').map(|seg| seg.replace('-', "_")).collect();
    if let Some(first) = segments.first() {
        if first == "externs" {
            segments.remove(0);
        }
    }
    for seg in segments {
        path.push(seg);
    }
    path
}

fn read_crate_name(module_dir: &Path) -> Result<String, LocalModuleError> {
    let manifest_path = module_dir.join("Cargo.toml");
    let manifest_str = fs::read_to_string(&manifest_path)?;
    #[derive(Deserialize)]
    struct Manifest {
        package: PackageSection,
    }
    #[derive(Deserialize)]
    struct PackageSection {
        name: String,
    }
    let manifest: Manifest =
        toml::from_str(&manifest_str).map_err(|source| LocalModuleError::Manifest {
            path: manifest_path,
            source,
        })?;
    Ok(manifest.package.name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specifier_builder_sets_version() {
        let spec = PackageSpecifier::new("foo", PackageKind::LustLibrary).with_version("1.2.3");
        assert_eq!(spec.version.as_deref(), Some("1.2.3"));
    }

    #[test]
    fn ensure_layout_creates_directories() {
        let temp_dir = tempfile::tempdir().expect("temp directory");
        let root = temp_dir.path().join("pkg");
        let manager = PackageManager::new(&root);
        manager.ensure_layout().expect("create dirs");
        assert!(root.exists());
        assert!(root.is_dir());
    }

    #[test]
    fn library_name_sanitizes_hyphens() {
        #[cfg(target_os = "windows")]
        assert_eq!(library_file_name("my-ext"), "my_ext.dll");
        #[cfg(target_os = "macos")]
        assert_eq!(library_file_name("my-ext"), "libmy_ext.dylib");
        #[cfg(all(unix, not(target_os = "macos")))]
        assert_eq!(library_file_name("my-ext"), "libmy_ext.so");
    }
}

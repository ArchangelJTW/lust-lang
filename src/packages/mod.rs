use crate::embed::native_types::ModuleStub;
use crate::{NativeExport, VM};
use dirs::home_dir;
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

pub mod archive;
pub mod credentials;
pub mod dependencies;
pub mod manifest;
pub mod registry;

pub use archive::{build_package_archive, ArchiveError, PackageArchive};
pub use credentials::{
    clear_credentials, credentials_file, load_credentials, save_credentials, Credentials,
    CredentialsError,
};
pub use dependencies::{
    resolve_dependencies, DependencyResolution, DependencyResolutionError, ResolvedLustDependency,
    ResolvedRustDependency,
};
pub use manifest::{ManifestError, ManifestKind, PackageManifest, PackageSection};
pub use registry::{
    DownloadedArchive, PackageDetails, PackageSearchResponse, PackageSummary, PackageVersionInfo,
    PublishResponse, RegistryClient, RegistryError, SearchParameters, DEFAULT_BASE_URL,
};

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
        let mut base = home_dir().unwrap_or_else(|| PathBuf::from("."));
        base.push(".lust");
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

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
#[derive(Debug, Clone)]
pub struct PreparedRustDependency {
    pub dependency: ResolvedRustDependency,
    pub build: LocalBuildOutput,
    pub stub_roots: Vec<StubRoot>,
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
#[derive(Debug, Clone)]
pub struct StubRoot {
    pub prefix: String,
    pub directory: PathBuf,
}

#[derive(Debug, Clone)]
pub struct BuildOptions<'a> {
    pub features: &'a [String],
    pub default_features: bool,
}

impl<'a> Default for BuildOptions<'a> {
    fn default() -> Self {
        Self {
            features: &[],
            default_features: true,
        }
    }
}

pub fn build_local_module(
    module_dir: &Path,
    options: BuildOptions<'_>,
) -> Result<LocalBuildOutput, LocalModuleError> {
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
    if !options.default_features {
        command.arg("--no-default-features");
    }
    if !options.features.is_empty() {
        command.arg("--features");
        command.arg(options.features.join(","));
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

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
pub fn collect_rust_dependency_artifacts(
    dep: &ResolvedRustDependency,
) -> Result<(LocalBuildOutput, Vec<StubFile>), String> {
    let build = build_local_module(
        &dep.crate_dir,
        BuildOptions {
            features: &dep.features,
            default_features: dep.default_features,
        },
    )
    .map_err(|err| format!("{}: {}", dep.crate_dir.display(), err))?;

    let mut preview_vm = VM::new();
    let preview_module = load_local_module(&mut preview_vm, &build)
        .map_err(|err| format!("{}: {}", dep.crate_dir.display(), err))?;
    let exports = preview_vm.take_exported_natives();
    let type_stubs = preview_vm.take_type_stubs();
    preview_vm.clear_native_functions();
    drop(preview_module);

    let mut stubs = stub_files_from_exports(&exports, &type_stubs);
    let manual_stubs = collect_stub_files(&dep.crate_dir, dep.externs_override.as_deref())
        .map_err(|err| format!("{}: {}", dep.crate_dir.display(), err))?;
    if !manual_stubs.is_empty() {
        stubs.extend(manual_stubs);
    }

    Ok((build, stubs))
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
pub fn prepare_rust_dependencies(
    deps: &DependencyResolution,
    project_dir: &Path,
) -> Result<Vec<PreparedRustDependency>, String> {
    if deps.rust().is_empty() {
        return Ok(Vec::new());
    }

    let mut prepared = Vec::new();
    let mut project_extern_root: Option<PathBuf> = None;

    for dep in deps.rust() {
        let (build, stubs) = collect_rust_dependency_artifacts(dep)?;
        let mut stub_roots = Vec::new();
        let sanitized_prefix = sanitized_crate_name(&build.name);
        let mut register_root = |dir: &Path| {
            let dir_buf = dir.join(&sanitized_prefix);
            if dir_buf.exists()
                && !stub_roots
                    .iter()
                    .any(|root: &StubRoot| root.directory == dir_buf)
            {
                stub_roots.push(StubRoot {
                    prefix: sanitized_prefix.clone(),
                    directory: dir_buf,
                });
            }
        };

        let fallback_root = project_dir.join("externs");
        register_root(&fallback_root);

        if let Some(cache_dir) = &dep.cache_stub_dir {
            fs::create_dir_all(cache_dir).map_err(|err| {
                format!(
                    "failed to create extern cache '{}': {}",
                    cache_dir.display(),
                    err
                )
            })?;
            if !stubs.is_empty() {
                write_stub_files(&build.name, &stubs, cache_dir)
                    .map_err(|err| format!("{}: {}", cache_dir.display(), err))?;
            }
            if cache_dir.exists() {
                register_root(cache_dir);
            }
        } else {
            let root = project_extern_root
                .get_or_insert_with(|| project_dir.join("externs"))
                .clone();
            if !stubs.is_empty() {
                fs::create_dir_all(&root).map_err(|err| format!("{}: {}", root.display(), err))?;
                write_stub_files(&build.name, &stubs, &root)
                    .map_err(|err| format!("{}: {}", root.display(), err))?;
                register_root(&root);
            } else if root.exists() {
                register_root(&root);
            }
        }

        prepared.push(PreparedRustDependency {
            dependency: dep.clone(),
            build,
            stub_roots,
        });
    }

    Ok(prepared)
}

#[cfg(all(feature = "packages", not(target_arch = "wasm32")))]
pub fn load_prepared_rust_dependencies(
    prepared: &[PreparedRustDependency],
    vm: &mut VM,
) -> Result<Vec<LoadedRustModule>, String> {
    let mut loaded = Vec::new();
    for item in prepared {
        let module = load_local_module(vm, &item.build)
            .map_err(|err| format!("{}: {}", item.dependency.crate_dir.display(), err))?;
        loaded.push(module);
    }
    Ok(loaded)
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

pub fn stub_files_from_exports(
    exports: &[NativeExport],
    type_stubs: &[ModuleStub],
) -> Vec<StubFile> {
    if exports.is_empty() && type_stubs.iter().all(ModuleStub::is_empty) {
        return Vec::new();
    }

    #[derive(Default)]
    struct CombinedModule<'a> {
        type_stub: ModuleStub,
        functions: Vec<&'a NativeExport>,
    }

    let mut combined: BTreeMap<String, CombinedModule<'_>> = BTreeMap::new();
    for stub in type_stubs {
        if stub.is_empty() {
            continue;
        }
        let entry = combined
            .entry(stub.module.clone())
            .or_insert_with(|| CombinedModule {
                type_stub: ModuleStub {
                    module: stub.module.clone(),
                    ..ModuleStub::default()
                },
                ..CombinedModule::default()
            });
        entry.type_stub.struct_defs.extend(stub.struct_defs.clone());
        entry.type_stub.enum_defs.extend(stub.enum_defs.clone());
        entry.type_stub.trait_defs.extend(stub.trait_defs.clone());
    }

    for export in exports {
        let (module, _) = match export.name().rsplit_once('.') {
            Some(parts) => parts,
            None => continue,
        };
        let entry = combined
            .entry(module.to_string())
            .or_insert_with(|| CombinedModule {
                type_stub: ModuleStub {
                    module: module.to_string(),
                    ..ModuleStub::default()
                },
                ..CombinedModule::default()
            });
        entry.functions.push(export);
    }

    let mut result = Vec::new();
    for (module, mut combined_entry) in combined {
        combined_entry
            .functions
            .sort_by(|a, b| a.name().cmp(b.name()));
        let mut contents = String::new();

        let mut wrote_type = false;
        let append_defs = |defs: &Vec<String>, contents: &mut String, wrote_flag: &mut bool| {
            if defs.is_empty() {
                return;
            }
            if *wrote_flag && !contents.ends_with("\n\n") && !contents.is_empty() {
                contents.push('\n');
            }
            for def in defs {
                contents.push_str(def);
                if !def.ends_with('\n') {
                    contents.push('\n');
                }
            }
            *wrote_flag = true;
        };

        append_defs(
            &combined_entry.type_stub.struct_defs,
            &mut contents,
            &mut wrote_type,
        );
        append_defs(
            &combined_entry.type_stub.enum_defs,
            &mut contents,
            &mut wrote_type,
        );
        append_defs(
            &combined_entry.type_stub.trait_defs,
            &mut contents,
            &mut wrote_type,
        );

        if !combined_entry.functions.is_empty() {
            if wrote_type && !contents.ends_with("\n\n") {
                contents.push('\n');
            }
            contents.push_str("pub extern\n");
            for export in combined_entry.functions {
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
            contents.push_str("end\n");
        }

        if contents.is_empty() {
            continue;
        }
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
    if let Some(first) = segments.first() {
        path.push(first);
    }
    if segments.len() > 1 {
        for seg in &segments[1..segments.len() - 1] {
            path.push(seg);
        }
        path.push(segments.last().unwrap());
    } else if let Some(first) = segments.first() {
        path.push(first);
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

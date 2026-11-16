// Host CLI / packaging boundary: package manager operates on the host filesystem directly.
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use crate::embed::native_types::ModuleStub;
use crate::{NativeEffectSpec, NativeExport, NativeExportParam, NativePurity, VM};
use crate::bytecode::{NativeCallResult, Value};
use dirs::home_dir;
use libloading::Library;
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    env,
    ffi::OsStr,
    fs, io,
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
    rc::Rc,
    sync::{Arc, Mutex, OnceLock},
};
use thiserror::Error;

pub mod archive;
pub mod credentials;
pub mod dependencies;
pub mod manifest;
pub mod registry;

pub use archive::{ArchiveError, PackageArchive, build_package_archive};
pub use credentials::{
    Credentials, CredentialsError, clear_credentials, credentials_file, load_credentials,
    save_credentials,
};
pub use dependencies::{
    DependencyResolution, DependencyResolutionError, ResolvedLuaDependency, ResolvedLustDependency,
    ResolvedRustDependency, Semver, resolve_dependencies, resolve_dependencies_with_manager,
    resolve_transitive_dependencies,
};
pub use manifest::{ManifestError, ManifestKind, PackageManifest, PackageSection};
pub use registry::{
    DEFAULT_BASE_URL, DownloadedArchive, PackageDetails, PackageSearchResponse, PackageSummary,
    PackageVersionInfo, PublishResponse, RegistryClient, RegistryError, SearchParameters,
    compute_sha256, verify_sha256,
};

#[derive(Debug, Error)]
pub enum ArchiveVerificationError {
    #[error("failed to read archive '{path}': {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("digest mismatch for package '{package}': expected '{expected}', found '{actual}'")]
    DigestMismatch {
        package: String,
        expected: String,
        actual: String,
    },
    #[error("corrupted archive for package '{package}' at '{path}': {message}")]
    CorruptedArchive {
        package: String,
        path: PathBuf,
        message: String,
    },
    #[error("failed to extract package '{package}' to '{destination}': {message}")]
    ExtractFailed {
        package: String,
        destination: PathBuf,
        message: String,
    },
}

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
        if let Ok(dir) = env::var("LUST_PACKAGES_DIR") {
            return PathBuf::from(dir);
        }
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

    pub fn verify_archive_digest(
        archive_path: &Path,
        expected_sha256: &str,
    ) -> Result<bool, ArchiveVerificationError> {
        let bytes = fs::read(archive_path).map_err(|source| ArchiveVerificationError::Io {
            path: archive_path.to_path_buf(),
            source,
        })?;
        Ok(registry::verify_sha256(&bytes, expected_sha256))
    }

    pub fn install_package_archive(
        &self,
        package_name: &str,
        version: &str,
        archive_path: &Path,
        expected_sha256: Option<&str>,
    ) -> Result<PathBuf, ArchiveVerificationError> {
        let bytes = fs::read(archive_path).map_err(|source| ArchiveVerificationError::Io {
            path: archive_path.to_path_buf(),
            source,
        })?;
        if let Some(expected) = expected_sha256 {
            let actual = registry::compute_sha256(&bytes);
            if !actual.eq_ignore_ascii_case(expected.trim()) {
                return Err(ArchiveVerificationError::DigestMismatch {
                    package: package_name.to_string(),
                    expected: expected.to_string(),
                    actual,
                });
            }
        }
        if bytes.len() < 10 || bytes[0] != 0x1f || bytes[1] != 0x8b {
            return Err(ArchiveVerificationError::CorruptedArchive {
                package: package_name.to_string(),
                path: archive_path.to_path_buf(),
                message: "invalid gzip magic header".to_string(),
            });
        }
        let target_dir = self.root.join(package_name).join(version);
        fs::create_dir_all(&target_dir).map_err(|source| ArchiveVerificationError::Io {
            path: target_dir.clone(),
            source,
        })?;
        let mut command = Command::new("tar");
        command.arg("-xzf").arg(archive_path).arg("-C").arg(&target_dir);
        let output = command.output().map_err(|source| ArchiveVerificationError::Io {
            path: archive_path.to_path_buf(),
            source,
        })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            return Err(ArchiveVerificationError::ExtractFailed {
                package: package_name.to_string(),
                destination: target_dir,
                message: stderr,
            });
        }
        Ok(target_dir)
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

    #[error("missing external library at '{0}'")]
    MissingExternalLibrary(PathBuf),

    #[error("IncompatibleAbiVersion: bad symbols in library '{path}': {message}")]
    BadSymbols {
        path: PathBuf,
        message: String,
    },

    #[error("IncompatibleAbiVersion in '{path}': {message}")]
    IncompatibleAbiVersion {
        path: PathBuf,
        message: String,
    },

    #[error("CapabilityDenied: {0}")]
    CapabilityDenied(String),
}

#[derive(Debug, Clone)]
pub struct LocalBuildOutput {
    pub name: String,
    pub library_path: PathBuf,
}

#[derive(Clone)]
pub struct LoadedRustModule {
    pub name: String,
    pub library: Arc<Library>,
}

impl std::fmt::Debug for LoadedRustModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedRustModule")
            .field("name", &self.name)
            .finish()
    }
}

impl LoadedRustModule {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn library(&self) -> &Arc<Library> {
        &self.library
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

pub const EXPECTED_LUST_EXT_ABI_VERSION: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LustExtValTag {
    Nil = 0,
    Bool = 1,
    Int = 2,
    Float = 3,
    String = 4,
    Handle = 5,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union LustExtValPayload {
    pub bool_val: bool,
    pub int_val: i64,
    pub float_val: f64,
    pub str_val: *const core::ffi::c_char,
    pub handle_val: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LustExtVal {
    pub tag: LustExtValTag,
    pub payload: LustExtValPayload,
}

pub type LustExtCleanFn = unsafe extern "C" fn(*mut core::ffi::c_void);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LustExtHostV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub vm_or_runtime_ptr: *mut core::ffi::c_void,
    pub charge_gas: unsafe extern "C" fn(ctx: *mut core::ffi::c_void, units: u64) -> i32,
    pub check_interrupt: unsafe extern "C" fn(ctx: *mut core::ffi::c_void, out_reason: *mut u32) -> i32,
    pub try_reserve_memory: unsafe extern "C" fn(
        ctx: *mut core::ffi::c_void,
        bytes: usize,
        out_reservation_id: *mut u64,
    ) -> i32,
    pub commit_memory: unsafe extern "C" fn(
        ctx: *mut core::ffi::c_void,
        reservation_id: u64,
        actual_bytes: usize,
    ) -> i32,
    pub rollback_memory: unsafe extern "C" fn(ctx: *mut core::ffi::c_void, reservation_id: u64) -> i32,
    pub acquire_resource: unsafe extern "C" fn(
        ctx: *mut core::ffi::c_void,
        scope_id: u64,
        handle: *mut core::ffi::c_void,
        cleanup: LustExtCleanFn,
        out_token: *mut u64,
    ) -> i32,
    pub release_resource: unsafe extern "C" fn(ctx: *mut core::ffi::c_void, token: u64) -> i32,
    pub host_alloc_value_bytes: unsafe extern "C" fn(
        ctx: *mut core::ffi::c_void,
        len: usize,
        out_ptr: *mut *mut u8,
    ) -> i32,
    pub host_free_value_bytes: unsafe extern "C" fn(ctx: *mut core::ffi::c_void, ptr: *mut u8, len: usize),
}

pub type LustExtNativeFn = unsafe extern "C" fn(
    host: *const LustExtHostV1,
    args: *const LustExtVal,
    args_len: usize,
    out_val: *mut LustExtVal,
) -> i32;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LustExtExportMetaRaw {
    pub name: *const core::ffi::c_char,
    pub params: *const *const core::ffi::c_char,
    pub params_len: usize,
    pub return_type: *const core::ffi::c_char,
    pub doc: *const core::ffi::c_char,
    pub purity: u8,
    pub required_capability: *const core::ffi::c_char,
    pub deterministic: bool,
    pub fn_ptr: Option<LustExtNativeFn>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LustExtManifestV1Raw {
    pub abi_version: u32,
    pub module_name: *const core::ffi::c_char,
    pub exports: *const LustExtExportMetaRaw,
    pub exports_len: usize,
}

pub struct VerifiedExtension {
    pub module_name: String,
    pub abi_version: u32,
    pub exports_len: usize,
    pub library: Library,
}

pub fn validate_extension_abi(
    lib: &Library,
    path: &Path,
) -> Result<&'static LustExtManifestV1Raw, LocalModuleError> {
    // 1. Fail closed on legacy raw *mut VM extension entry if present
    let has_legacy_register = unsafe {
        lib.get::<unsafe extern "C" fn(*mut core::ffi::c_void) -> bool>(b"lust_extension_register\0")
            .is_ok()
    };
    if has_legacy_register {
        return Err(LocalModuleError::IncompatibleAbiVersion {
            path: path.to_path_buf(),
            message: "legacy raw *mut VM extension entry 'lust_extension_register' is unsupported and rejected; extensions must use narrow stable C ABI (lust_ext_manifest_v1 with LustExtHostV1)".to_string(),
        });
    }

    // 2. Validate narrow stable C ABI manifest entry point
    let manifest_fn: libloading::Symbol<unsafe extern "C" fn() -> *const LustExtManifestV1Raw> = unsafe {
        match lib.get(b"lust_ext_manifest_v1\0") {
            Ok(sym) => sym,
            Err(_) => {
                return Err(LocalModuleError::IncompatibleAbiVersion {
                    path: path.to_path_buf(),
                    message: "missing required manifest symbol 'lust_ext_manifest_v1'".to_string(),
                });
            }
        }
    };

    let manifest_ptr = unsafe { manifest_fn() };
    if manifest_ptr.is_null() {
        return Err(LocalModuleError::IncompatibleAbiVersion {
            path: path.to_path_buf(),
            message: "manifest function returned NULL pointer".to_string(),
        });
    }

    if (manifest_ptr as usize) % core::mem::align_of::<LustExtManifestV1Raw>() != 0 {
        return Err(LocalModuleError::IncompatibleAbiVersion {
            path: path.to_path_buf(),
            message: "manifest function returned misaligned pointer".to_string(),
        });
    }

    let manifest = unsafe { &*manifest_ptr };
    if manifest.abi_version != EXPECTED_LUST_EXT_ABI_VERSION {
        return Err(LocalModuleError::IncompatibleAbiVersion {
            path: path.to_path_buf(),
            message: format!(
                "extension ABI version {} does not match host ABI version {}",
                manifest.abi_version, EXPECTED_LUST_EXT_ABI_VERSION
            ),
        });
    }

    if manifest.module_name.is_null() {
        return Err(LocalModuleError::IncompatibleAbiVersion {
            path: path.to_path_buf(),
            message: "manifest module_name pointer is null".to_string(),
        });
    }

    let module_name = unsafe { std::ffi::CStr::from_ptr(manifest.module_name) }
        .to_str()
        .map_err(|e| LocalModuleError::IncompatibleAbiVersion {
            path: path.to_path_buf(),
            message: format!("manifest module_name is not valid UTF-8: {e}"),
        })?;

    if module_name.is_empty() {
        return Err(LocalModuleError::IncompatibleAbiVersion {
            path: path.to_path_buf(),
            message: "manifest module_name is empty".to_string(),
        });
    }

    if manifest.exports_len > 0 {
        if manifest.exports.is_null() {
            return Err(LocalModuleError::IncompatibleAbiVersion {
                path: path.to_path_buf(),
                message: format!(
                    "manifest exports pointer is null but exports_len is {}",
                    manifest.exports_len
                ),
            });
        }
        if (manifest.exports as usize) % core::mem::align_of::<LustExtExportMetaRaw>() != 0 {
            return Err(LocalModuleError::IncompatibleAbiVersion {
                path: path.to_path_buf(),
                message: "manifest exports pointer is misaligned".to_string(),
            });
        }
        for i in 0..manifest.exports_len {
            let export = unsafe { &*manifest.exports.add(i) };
            if export.name.is_null() {
                return Err(LocalModuleError::IncompatibleAbiVersion {
                    path: path.to_path_buf(),
                    message: format!("manifest export[{i}].name pointer is null"),
                });
            }
            let _ = unsafe { std::ffi::CStr::from_ptr(export.name) }
                .to_str()
                .map_err(|e| LocalModuleError::IncompatibleAbiVersion {
                    path: path.to_path_buf(),
                    message: format!("manifest export[{i}].name is not valid UTF-8: {e}"),
                })?;

            if !export.return_type.is_null() {
                let _ = unsafe { std::ffi::CStr::from_ptr(export.return_type) }
                    .to_str()
                    .map_err(|e| LocalModuleError::IncompatibleAbiVersion {
                        path: path.to_path_buf(),
                        message: format!("manifest export[{i}].return_type is not valid UTF-8: {e}"),
                    })?;
            }

            if export.purity > 1 {
                return Err(LocalModuleError::IncompatibleAbiVersion {
                    path: path.to_path_buf(),
                    message: format!(
                        "manifest export[{i}].purity has invalid scalar value {} (expected 0 for Pure, 1 for Effectful)",
                        export.purity
                    ),
                });
            }

            if !export.required_capability.is_null() {
                let _ = unsafe { std::ffi::CStr::from_ptr(export.required_capability) }
                    .to_str()
                    .map_err(|e| LocalModuleError::IncompatibleAbiVersion {
                        path: path.to_path_buf(),
                        message: format!("manifest export[{i}].required_capability is not valid UTF-8: {e}"),
                    })?;
            }

            if export.params_len > 0 {
                if export.params.is_null() {
                    return Err(LocalModuleError::IncompatibleAbiVersion {
                        path: path.to_path_buf(),
                        message: format!(
                            "manifest export[{i}].params pointer is null but params_len is {}",
                            export.params_len
                        ),
                    });
                }
                for p in 0..export.params_len {
                    let param_ptr = unsafe { *export.params.add(p) };
                    if param_ptr.is_null() {
                        return Err(LocalModuleError::IncompatibleAbiVersion {
                            path: path.to_path_buf(),
                            message: format!("manifest export[{i}].params[{p}] pointer is null"),
                        });
                    }
                    let _ = unsafe { std::ffi::CStr::from_ptr(param_ptr) }
                        .to_str()
                        .map_err(|e| LocalModuleError::IncompatibleAbiVersion {
                            path: path.to_path_buf(),
                            message: format!("manifest export[{i}].params[{p}] is not valid UTF-8: {e}"),
                        })?;
                }
            }
        }
    }

    Ok(manifest)
}

pub fn load_and_verify_extension(dylib_path: &Path) -> Result<VerifiedExtension, LocalModuleError> {
    if !dylib_path.exists() {
        return Err(LocalModuleError::MissingExternalLibrary(dylib_path.to_path_buf()));
    }
    let lib = unsafe { Library::new(dylib_path) }.map_err(|e| LocalModuleError::BadSymbols {
        path: dylib_path.to_path_buf(),
        message: format!("failed to load dynamic library: {e}"),
    })?;

    let manifest = validate_extension_abi(&lib, dylib_path)?;
    let mod_name = unsafe { std::ffi::CStr::from_ptr(manifest.module_name) }
        .to_string_lossy()
        .into_owned();

    Ok(VerifiedExtension {
        module_name: mod_name,
        abi_version: manifest.abi_version,
        exports_len: manifest.exports_len,
        library: lib,
    })
}

// Memory reservations tracked across C ABI boundary
static RESERVATION_COUNTER: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(1);
static RESERVATIONS: Mutex<Option<HashMap<u64, usize>>> = Mutex::new(None);

fn with_reservations<F, R>(f: F) -> R
where
    F: FnOnce(&mut HashMap<u64, usize>) -> R,
{
    let mut guard = RESERVATIONS.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    f(map)
}

// Host callbacks backed by VM via ctx (*mut VM)
unsafe extern "C" fn vm_host_charge_gas(ctx: *mut core::ffi::c_void, units: u64) -> i32 {
    if ctx.is_null() {
        return 1; // ERR_GENERIC
    }
    let vm = unsafe { &mut *(ctx as *mut VM) };
    if units == 0 {
        return 0;
    }
    match vm.charge_gas(units) {
        Ok(()) => 0,
        Err(_) => 2, // ERR_OUT_OF_GAS
    }
}

unsafe extern "C" fn vm_host_check_interrupt(ctx: *mut core::ffi::c_void, out_reason: *mut u32) -> i32 {
    if ctx.is_null() {
        return 1;
    }
    let vm = unsafe { &*(ctx as *mut VM) };
    if vm.pending_task_signal.is_some() {
        if !out_reason.is_null() {
            unsafe { *out_reason = 1; }
        }
        4 // ERR_INTERRUPTED
    } else {
        0
    }
}

unsafe extern "C" fn vm_host_try_reserve_memory(
    ctx: *mut core::ffi::c_void,
    bytes: usize,
    out_reservation_id: *mut u64,
) -> i32 {
    if ctx.is_null() || out_reservation_id.is_null() {
        return 1;
    }
    let vm = unsafe { &mut *(ctx as *mut VM) };
    if vm.memory_budget_bytes().is_some() && !vm.try_charge_memory_bytes(bytes) {
        return 3; // ERR_OUT_OF_MEMORY
    }
    let id = RESERVATION_COUNTER.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    with_reservations(|map| {
        map.insert(id, bytes);
    });
    unsafe { *out_reservation_id = id; }
    0
}

unsafe extern "C" fn vm_host_commit_memory(
    ctx: *mut core::ffi::c_void,
    reservation_id: u64,
    actual_bytes: usize,
) -> i32 {
    if ctx.is_null() {
        return 1;
    }
    let vm = unsafe { &mut *(ctx as *mut VM) };
    let reserved = with_reservations(|map| map.remove(&reservation_id));
    if let Some(reserved_bytes) = reserved {
        if actual_bytes < reserved_bytes {
            vm.release_memory_bytes(reserved_bytes - actual_bytes);
            0
        } else if actual_bytes > reserved_bytes {
            let extra = actual_bytes - reserved_bytes;
            if vm.memory_budget_bytes().is_some() && !vm.try_charge_memory_bytes(extra) {
                3 // ERR_OUT_OF_MEMORY
            } else {
                0
            }
        } else {
            0
        }
    } else {
        1
    }
}

unsafe extern "C" fn vm_host_rollback_memory(
    ctx: *mut core::ffi::c_void,
    reservation_id: u64,
) -> i32 {
    if ctx.is_null() {
        return 1;
    }
    let vm = unsafe { &mut *(ctx as *mut VM) };
    let reserved = with_reservations(|map| map.remove(&reservation_id));
    if let Some(reserved_bytes) = reserved {
        vm.release_memory_bytes(reserved_bytes);
        0
    } else {
        1
    }
}

static EXT_RESOURCE_REGISTRY: Mutex<Option<HashMap<u64, (usize, LustExtCleanFn)>>> = Mutex::new(None);

unsafe extern "C" fn vm_host_acquire_resource(
    ctx: *mut core::ffi::c_void,
    scope_id: u64,
    handle: *mut core::ffi::c_void,
    cleanup: LustExtCleanFn,
    out_token: *mut u64,
) -> i32 {
    if ctx.is_null() || out_token.is_null() {
        return 1;
    }
    let vm = unsafe { &mut *(ctx as *mut VM) };
    match vm.host_ledger.acquire_scoped(handle as usize, |_| {}, scope_id) {
        Ok(slot_idx) => {
            let generation = vm.host_ledger.slots[slot_idx].generation;
            let token = ((generation as u64) << 32) | (slot_idx as u64);
            let mut guard = EXT_RESOURCE_REGISTRY.lock().unwrap();
            let map = guard.get_or_insert_with(HashMap::new);
            map.insert(token, (handle as usize, cleanup));
            unsafe { *out_token = token; }
            0
        }
        Err(_) => 1,
    }
}

unsafe extern "C" fn vm_host_release_resource(
    ctx: *mut core::ffi::c_void,
    token: u64,
) -> i32 {
    if ctx.is_null() {
        return 1;
    }
    let vm = unsafe { &mut *(ctx as *mut VM) };
    let slot_idx = (token as u32) as usize;
    let generation = (token >> 32) as u32;
    if slot_idx < vm.host_ledger.slots.len()
        && vm.host_ledger.slots[slot_idx].active
        && vm.host_ledger.slots[slot_idx].generation == generation
    {
        vm.host_ledger.release(slot_idx);
        let entry = {
            let mut guard = EXT_RESOURCE_REGISTRY.lock().unwrap();
            guard.as_mut().and_then(|m| m.remove(&token))
        };
        if let Some((handle_usize, cleanup)) = entry {
            unsafe { cleanup(handle_usize as *mut core::ffi::c_void); }
        }
        0
    } else {
        5 // ERR_INVALID_HANDLE
    }
}

unsafe extern "C" fn vm_host_alloc_value_bytes(
    ctx: *mut core::ffi::c_void,
    len: usize,
    out_ptr: *mut *mut u8,
) -> i32 {
    if ctx.is_null() || out_ptr.is_null() {
        return 1;
    }
    let vm = unsafe { &mut *(ctx as *mut VM) };
    if vm.memory_budget_bytes().is_some() && !vm.try_charge_memory_bytes(len) {
        return 3; // ERR_OUT_OF_MEMORY
    }
    let layout = match std::alloc::Layout::from_size_align(len.max(1), 8) {
        Ok(l) => l,
        Err(_) => return 1,
    };
    let ptr = unsafe { std::alloc::alloc(layout) };
    if ptr.is_null() {
        vm.release_memory_bytes(len);
        return 3;
    }
    unsafe { *out_ptr = ptr; }
    0
}

unsafe extern "C" fn vm_host_free_value_bytes(
    ctx: *mut core::ffi::c_void,
    ptr: *mut u8,
    len: usize,
) {
    if !ptr.is_null() {
        let layout = std::alloc::Layout::from_size_align(len.max(1), 8).unwrap();
        unsafe { std::alloc::dealloc(ptr, layout); }
        if !ctx.is_null() {
            let vm = unsafe { &mut *(ctx as *mut VM) };
            vm.release_memory_bytes(len);
        }
    }
}

pub fn test_load_extension_cli(dylib_path: &Path, canary_path: Option<&Path>) -> Result<(), String> {
    if let Some(cp) = canary_path {
        let _ = fs::write(cp, "HOST_VM_ACCESS_COUNT=0\n");
    }

    let mut vm = VM::new();
    let build = LocalBuildOutput {
        name: "test_ext".to_string(),
        library_path: dylib_path.to_path_buf(),
    };

    let loaded = match load_local_module(&mut vm, &build) {
        Ok(l) => l,
        Err(err) => {
            if let Some(cp) = canary_path {
                let mut current = fs::read_to_string(cp).unwrap_or_default();
                current.push_str("HOST_VALIDATION_FAILED=true\n");
                let _ = fs::write(cp, current);
            }
            return Err(format!("{err}"));
        }
    };

    if let Some(cp) = canary_path {
        let mut current = fs::read_to_string(cp).unwrap_or_default();
        current.push_str("HOST_VALIDATION_PASSED=true\nHOST_VALIDATION_FAILED=false\n");
        let _ = fs::write(cp, current);
    }

    let mut invoked_count = 0;
    let exports = vm.exported_natives().to_vec();
    if !exports.is_empty() {
        let first_export = &exports[0];
        let fn_name = first_export.name().to_string();

        let initial_gas = vm.gas_used();
        let call_args = if first_export.params().is_empty() {
            Vec::new()
        } else {
            vec![Value::Int(42)]
        };
        let call_res = vm.call(&fn_name, call_args);
        match call_res {
            Ok(val) => {
                let gas_charged = vm.gas_used().saturating_sub(initial_gas);
                let res_str = match &val {
                    Value::Int(i) => i.to_string(),
                    Value::Float(f) => f.to_string(),
                    Value::Bool(b) => b.to_string(),
                    Value::String(s) => s.to_string(),
                    _ => format!("{val:?}"),
                };
                println!(
                    "Export invoked successfully: input=42 => result={res_str}, gas_charged={gas_charged}"
                );
                invoked_count += 1;
            }
            Err(e) => {
                return Err(format!("Export invocation failed through VM: {e}"));
            }
        }
    }

    if let Some(cp) = canary_path {
        let mut current = fs::read_to_string(cp).unwrap_or_default();
        let access_count = if invoked_count > 0 { 1 } else { 0 };
        current.push_str(&format!("HOST_VM_ACCESS_COUNT={access_count}\n"));
        let _ = fs::write(cp, current);
    }

    println!(
        "Extension '{}' verified compatible and loaded successfully",
        loaded.name()
    );
    Ok(())
}

pub fn load_local_module(
    vm: &mut VM,
    build: &LocalBuildOutput,
) -> Result<LoadedRustModule, LocalModuleError> {
    let verified = load_and_verify_extension(&build.library_path)?;
    let lib_arc = Arc::new(verified.library);

    let manifest_fn: libloading::Symbol<unsafe extern "C" fn() -> *const LustExtManifestV1Raw> = unsafe {
        lib_arc.get(b"lust_ext_manifest_v1\0").map_err(|e| LocalModuleError::BadSymbols {
            path: build.library_path.clone(),
            message: format!("failed to get manifest symbol: {e}"),
        })?
    };
    let manifest = unsafe { &*manifest_fn() };

    // Transactional pre-validation: check for duplicates and unsupported operations
    let mut export_names_seen = HashSet::new();
    let mut planned_exports = Vec::new();

    for i in 0..manifest.exports_len {
        let export_raw = unsafe { &*manifest.exports.add(i) };
        if export_raw.name.is_null() {
            return Err(LocalModuleError::IncompatibleAbiVersion {
                path: build.library_path.clone(),
                message: format!("manifest export[{i}].name pointer is null"),
            });
        }
        let export_name = unsafe { std::ffi::CStr::from_ptr(export_raw.name) }
            .to_str()
            .map_err(|e| LocalModuleError::IncompatibleAbiVersion {
                path: build.library_path.clone(),
                message: format!("manifest export[{i}].name is not valid UTF-8: {e}"),
            })?;

        if !export_names_seen.insert(export_name.to_string()) {
            return Err(LocalModuleError::IncompatibleAbiVersion {
                path: build.library_path.clone(),
                message: format!("DuplicateExport: manifest contains duplicate export '{export_name}'"),
            });
        }

        let full_reg_name = if !export_name.contains('.') && !export_name.contains(':') {
            format!("{}.{}", verified.module_name, export_name)
        } else {
            export_name.to_string()
        };

        // Check for duplicates in VM globals/natives
        if vm.get_global(&full_reg_name).is_some() || vm.get_global(export_name).is_some() {
            return Err(LocalModuleError::RegisterFailed(format!(
                "duplicate registration: export '{}' already exists in VM",
                full_reg_name
            )));
        }

        let fn_ptr = export_raw.fn_ptr.ok_or_else(|| LocalModuleError::IncompatibleAbiVersion {
            path: build.library_path.clone(),
            message: format!("manifest export[{i}] ('{export_name}') has null fn_ptr; unsupported operation fails closed"),
        })?;

        let ret_ty = if !export_raw.return_type.is_null() {
            unsafe { std::ffi::CStr::from_ptr(export_raw.return_type) }
                .to_str()
                .map_err(|e| LocalModuleError::IncompatibleAbiVersion {
                    path: build.library_path.clone(),
                    message: format!("manifest export[{i}].return_type is not valid UTF-8: {e}"),
                })?
        } else {
            "()"
        };

        let purity = match export_raw.purity {
            0 => NativePurity::Pure,
            1 => NativePurity::Effectful,
            other => {
                return Err(LocalModuleError::IncompatibleAbiVersion {
                    path: build.library_path.clone(),
                    message: format!("manifest export[{i}].purity has invalid scalar value {other}"),
                });
            }
        };

        let req_cap = if !export_raw.required_capability.is_null() {
            Some(
                unsafe { std::ffi::CStr::from_ptr(export_raw.required_capability) }
                    .to_str()
                    .map_err(|e| LocalModuleError::IncompatibleAbiVersion {
                        path: build.library_path.clone(),
                        message: format!("manifest export[{i}].required_capability is not valid UTF-8: {e}"),
                    })?
                    .to_string(),
            )
        } else {
            None
        };

        let mut param_specs = Vec::new();
        if export_raw.params_len > 0 && !export_raw.params.is_null() {
            for p in 0..export_raw.params_len {
                let param_ptr = unsafe { *export_raw.params.add(p) };
                let p_str = unsafe { std::ffi::CStr::from_ptr(param_ptr) }
                    .to_str()
                    .map_err(|e| LocalModuleError::IncompatibleAbiVersion {
                        path: build.library_path.clone(),
                        message: format!("manifest export[{i}].params[{p}] is not valid UTF-8: {e}"),
                    })?;
                let (p_name, p_ty) = if let Some((n, t)) = p_str.split_once(':') {
                    (n.trim().to_string(), t.trim().to_string())
                } else {
                    (format!("arg{p}"), p_str.trim().to_string())
                };
                param_specs.push(NativeExportParam::new(p_name, p_ty));
            }
        }

        planned_exports.push((
            export_name.to_string(),
            full_reg_name,
            param_specs,
            ret_ty.to_string(),
            purity,
            req_cap,
            export_raw.deterministic,
            fn_ptr,
        ));
    }

    // Register all exports into the VM
    for (export_name, full_name, params, ret_ty, purity, req_cap, deterministic, fn_ptr) in planned_exports {
        let mut effect_spec = if purity == NativePurity::Pure {
            NativeEffectSpec::pure()
        } else {
            NativeEffectSpec::effectful(None::<String>, Vec::new())
        };
        if let Some(cap) = req_cap {
            effect_spec = effect_spec.with_capability(cap);
        }
        effect_spec.deterministic = deterministic;

        let export_meta = NativeExport::new(full_name.clone(), params.clone(), ret_ty.clone())
            .with_effects(effect_spec);

        let lib_holder = Arc::clone(&lib_arc);
        let fn_name_for_err = export_name.clone();
        let expected_params_count = params.len();

        let native_callback = move |args: &[Value]| -> Result<NativeCallResult, String> {
            let _keep_alive = &lib_holder; // Retains dylib as long as this callback can run
            if expected_params_count > 0 && args.len() != expected_params_count {
                return Err(format!(
                    "Native '{}' expects {} arguments, got {}",
                    fn_name_for_err,
                    expected_params_count,
                    args.len()
                ));
            }

            // Convert arguments to LustExtVal
            let mut ext_args = Vec::with_capacity(args.len());
            let mut cstring_keeper = Vec::new();

            for arg in args {
                match arg {
                    Value::Nil => ext_args.push(LustExtVal {
                        tag: LustExtValTag::Nil,
                        payload: LustExtValPayload { int_val: 0 },
                    }),
                    Value::Bool(b) => ext_args.push(LustExtVal {
                        tag: LustExtValTag::Bool,
                        payload: LustExtValPayload { bool_val: *b },
                    }),
                    Value::Int(i) => ext_args.push(LustExtVal {
                        tag: LustExtValTag::Int,
                        payload: LustExtValPayload { int_val: *i },
                    }),
                    Value::Float(f) => ext_args.push(LustExtVal {
                        tag: LustExtValTag::Float,
                        payload: LustExtValPayload { float_val: *f },
                    }),
                    Value::String(s) => {
                        let cstr = std::ffi::CString::new(s.as_str())
                            .map_err(|e| format!("Invalid string argument in native call: {e}"))?;
                        let ptr = cstr.as_ptr();
                        cstring_keeper.push(cstr);
                        ext_args.push(LustExtVal {
                            tag: LustExtValTag::String,
                            payload: LustExtValPayload { str_val: ptr },
                        });
                    }
                    other => {
                        return Err(format!(
                            "UnsupportedArgumentType: value of type '{:?}' is not supported across narrow C ABI; unsupported operations fail closed",
                            other.type_of()
                        ));
                    }
                }
            }

            let mut out_val = LustExtVal {
                tag: LustExtValTag::Nil,
                payload: LustExtValPayload { int_val: 0 },
            };

            let current_vm = crate::vm::current_vm_ptr().ok_or_else(|| {
                "No active VM context; native call must execute within VM execution context".to_string()
            })?;

            let host_v1 = LustExtHostV1 {
                abi_version: EXPECTED_LUST_EXT_ABI_VERSION,
                struct_size: core::mem::size_of::<LustExtHostV1>() as u32,
                vm_or_runtime_ptr: current_vm as *mut core::ffi::c_void,
                charge_gas: vm_host_charge_gas,
                check_interrupt: vm_host_check_interrupt,
                try_reserve_memory: vm_host_try_reserve_memory,
                commit_memory: vm_host_commit_memory,
                rollback_memory: vm_host_rollback_memory,
                acquire_resource: vm_host_acquire_resource,
                release_resource: vm_host_release_resource,
                host_alloc_value_bytes: vm_host_alloc_value_bytes,
                host_free_value_bytes: vm_host_free_value_bytes,
            };

            let rc = unsafe {
                fn_ptr(
                    &host_v1,
                    ext_args.as_ptr(),
                    ext_args.len(),
                    &mut out_val,
                )
            };

            match rc {
                0 => { // OK
                    let res_val = match out_val.tag {
                        LustExtValTag::Nil => Value::Nil,
                        LustExtValTag::Bool => Value::Bool(unsafe { out_val.payload.bool_val }),
                        LustExtValTag::Int => Value::Int(unsafe { out_val.payload.int_val }),
                        LustExtValTag::Float => Value::Float(unsafe { out_val.payload.float_val }),
                        LustExtValTag::String => {
                            let ptr = unsafe { out_val.payload.str_val };
                            if ptr.is_null() {
                                Value::Nil
                            } else {
                                let s = unsafe { std::ffi::CStr::from_ptr(ptr) }
                                    .to_str()
                                    .map_err(|e| format!("Invalid UTF-8 in native return string: {e}"))?;
                                Value::String(Rc::new(s.to_string()))
                            }
                        }
                        LustExtValTag::Handle => {
                            let handle = unsafe { out_val.payload.handle_val };
                            Value::Int(handle as i64)
                        }
                    };
                    Ok(NativeCallResult::Return(res_val))
                }
                2 => Err("Out of gas".to_string()),
                3 => Err("Out of memory budget".to_string()),
                4 => Err("Interrupted".to_string()),
                7 => Err(format!("Capability denied for '{}'", fn_name_for_err)),
                8 => Err(format!("Native '{}' panicked", fn_name_for_err)),
                code => Err(format!("Native '{}' failed with status code {code}", fn_name_for_err)),
            }
        };

        vm.register_exported_native(export_meta, native_callback);

        if export_name != full_name {
            if let Some(val) = vm.get_global(&full_name) {
                vm.register_native(export_name.clone(), val);
            }
        }
    }

    Ok(LoadedRustModule {
        name: build.name.clone(),
        library: lib_arc,
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

pub fn verify_external_library(path: &Path, allow_ffi: bool) -> Result<(), LocalModuleError> {
    if !allow_ffi {
        return Err(LocalModuleError::CapabilityDenied(
            "external library linking requires 'ffi' capability".to_string(),
        ));
    }
    if !path.exists() {
        return Err(LocalModuleError::MissingExternalLibrary(path.to_path_buf()));
    }
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    if ext == "dylib" || ext == "so" || ext == "dll" {
        let _ = load_and_verify_extension(path)?;
    }
    Ok(())
}

pub fn build_and_prepare_external_crate(
    crate_dir: &Path,
    allow_ffi: bool,
    target_triple: Option<&str>,
    is_release: bool,
) -> Result<LocalBuildOutput, LocalModuleError> {
    if !allow_ffi {
        return Err(LocalModuleError::CapabilityDenied(
            "external crate linking requires 'ffi' capability".to_string(),
        ));
    }
    if !crate_dir.exists() || !crate_dir.join("Cargo.toml").exists() {
        return Err(LocalModuleError::MissingExternalLibrary(crate_dir.to_path_buf()));
    }
    let crate_name = read_crate_name(crate_dir)?;
    let mut cmd = Command::new("cargo");
    if let Ok(rustup) = std::env::var("RUSTUP_HOME") {
        cmd.env("RUSTUP_HOME", rustup);
    }
    if let Ok(cargo_home) = std::env::var("CARGO_HOME") {
        cmd.env("CARGO_HOME", cargo_home);
    }
    cmd.arg("build").arg("--offline");
    if is_release {
        cmd.arg("--release");
    }
    if let Some(target) = target_triple {
        cmd.arg("--target").arg(target);
    }
    cmd.current_dir(crate_dir);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let output = cmd.output()?;
    if !output.status.success() {
        let mut msg = String::new();
        if !output.stdout.is_empty() {
            msg.push_str(&String::from_utf8_lossy(&output.stdout));
        }
        if !output.stderr.is_empty() {
            if !msg.is_empty() {
                msg.push('\n');
            }
            msg.push_str(&String::from_utf8_lossy(&output.stderr));
        }
        return Err(LocalModuleError::CargoBuild {
            module: crate_name,
            status: output.status,
            output: msg,
        });
    }

    let profile = if is_release { "release" } else { "debug" };
    let lib_name = library_file_name(&crate_name);
    let target_dir = crate_dir.join("target");
    let artifact = if let Some(target) = target_triple {
        target_dir.join(target).join(profile).join(&lib_name)
    } else {
        target_dir.join(profile).join(&lib_name)
    };

    if !artifact.exists() {
        return Err(LocalModuleError::LibraryMissing(artifact));
    }

    verify_external_library(&artifact, allow_ffi)?;

    Ok(LocalBuildOutput {
        name: crate_name,
        library_path: artifact,
    })
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

pub fn library_file_name(crate_name: &str) -> String {
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

pub fn sanitized_crate_name(name: &str) -> String {
    name.replace('-', "_")
}

fn library_cache() -> &'static Mutex<HashMap<PathBuf, &'static Library>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, &'static Library>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[allow(dead_code)]
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
        entry.type_stub.const_defs.extend(stub.const_defs.clone());
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
        append_defs(
            &combined_entry.type_stub.const_defs,
            &mut contents,
            &mut wrote_type,
        );

        if !combined_entry.functions.is_empty() {
            if wrote_type && !contents.ends_with("\n\n") {
                contents.push('\n');
            }
            contents.push_str("extern\n");
            for export in combined_entry.functions {
                if let Some((_, function)) = export.name().rsplit_once('.') {
                    let params = format_params(export);
                    let return_type = export.return_type();
                    if let Some(doc) = export.doc() {
                        contents.push_str(&crate::vm::format_doc_comment(doc, "    "));
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
            let ty = if ty.is_empty() { "any" } else { ty };
            let name = param.name().trim();
            if name.is_empty() {
                ty.to_string()
            } else {
                format!("{}: {}", name, ty)
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn relative_stub_path(module: &str) -> PathBuf {
    let mut segments: Vec<String> = module.split('.').map(|seg| seg.replace('-', "_")).collect();
    if let Some(first) = segments.first()
        && first == "externs"
    {
        segments.remove(0);
    }
    if segments.len() == 1 {
        return PathBuf::from(format!("{}.lust", segments[0]));
    }
    let mut path = PathBuf::new();
    for seg in &segments {
        path.push(seg);
    }
    path
}

pub fn read_crate_name(module_dir: &Path) -> Result<String, LocalModuleError> {
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

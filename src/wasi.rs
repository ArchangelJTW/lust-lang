//! WASI (WebAssembly System Interface) runtime adapters for Lust Runtime.
//!
//! Provides capability-gated runtime adapters for:
//! - Preopened filesystem (`VirtualFileSystem` integration, sandbox containment, path normalization)
//! - Environment variables and CLI arguments
//! - Deterministic monotonic and wall clocks (`LustClock` / `clock_time_get`)
//! - Standard I/O (stdin, stdout, stderr) capture and redirection
//! - Deterministic simulation mocks (`MockWasiClock`, `MockWasiEnv`, `MockWasiFs`)
//! - Explicit unsupported-capability behavior (sockets/net, thread spawn, proc spawn, signals)
//! - Single-threaded WASI timeout guard per RFC-WASM-WASI-01 §2.2
//! - Resource lifecycle tracking and cleanup on Drop

use std::collections::HashMap;
use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::effect_abi::{Cause, Exit};
use crate::sim::clock::{LustClock, MonotonicClock, WallClock};
use crate::vfs::{MemFs, VfsMetadata, VirtualEnv, VirtualFileSystem};
use crate::vm::VM;

// =============================================================================
// 1. WASI Errno and Error Model
// =============================================================================

/// Canonical WASI Preview 1 errno codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum WasiErrno {
    Success = 0,
    E2big = 1,
    Acces = 2,
    Badf = 8,
    Busy = 10,
    Exist = 20,
    Fault = 21,
    Fbig = 22,
    Inval = 28,
    Io = 29,
    Isdir = 31,
    Loop = 32,
    Mfile = 33,
    Nfile = 41,
    Noent = 44,
    Nosys = 52,
    Notdir = 54,
    Notempty = 55,
    Notsup = 58,
    Overflow = 61,
    Perm = 63,
    Pipe = 64,
    Roofs = 69,
    Spipe = 70,
    Timedout = 73,
}

impl WasiErrno {
    pub fn as_u16(&self) -> u16 {
        *self as u16
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Success => "SUCCESS",
            Self::E2big => "E2BIG",
            Self::Acces => "EACCES",
            Self::Badf => "EBADF",
            Self::Busy => "EBUSY",
            Self::Exist => "EEXIST",
            Self::Fault => "EFAULT",
            Self::Fbig => "EFBIG",
            Self::Inval => "EINVAL",
            Self::Io => "EIO",
            Self::Isdir => "EISDIR",
            Self::Loop => "ELOOP",
            Self::Mfile => "EMFILE",
            Self::Nfile => "ENFILE",
            Self::Noent => "ENOENT",
            Self::Nosys => "ENOSYS",
            Self::Notdir => "ENOTDIR",
            Self::Notempty => "ENOTEMPTY",
            Self::Notsup => "ENOTSUP",
            Self::Overflow => "EOVERFLOW",
            Self::Perm => "EPERM",
            Self::Pipe => "EPIPE",
            Self::Roofs => "EROFS",
            Self::Spipe => "ESPIPE",
            Self::Timedout => "ETIMEDOUT",
        }
    }

    pub fn from_u16(code: u16) -> Option<Self> {
        match code {
            0 => Some(Self::Success),
            1 => Some(Self::E2big),
            2 => Some(Self::Acces),
            8 => Some(Self::Badf),
            10 => Some(Self::Busy),
            20 => Some(Self::Exist),
            21 => Some(Self::Fault),
            22 => Some(Self::Fbig),
            28 => Some(Self::Inval),
            29 => Some(Self::Io),
            31 => Some(Self::Isdir),
            32 => Some(Self::Loop),
            33 => Some(Self::Mfile),
            41 => Some(Self::Nfile),
            44 => Some(Self::Noent),
            52 => Some(Self::Nosys),
            54 => Some(Self::Notdir),
            55 => Some(Self::Notempty),
            58 => Some(Self::Notsup),
            61 => Some(Self::Overflow),
            63 => Some(Self::Perm),
            64 => Some(Self::Pipe),
            69 => Some(Self::Roofs),
            70 => Some(Self::Spipe),
            73 => Some(Self::Timedout),
            _ => None,
        }
    }
}

impl fmt::Display for WasiErrno {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name(), self.as_u16())
    }
}

/// Structured error returned by WASI runtime adapters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasiError {
    pub errno: WasiErrno,
    pub message: String,
    pub capability: Option<String>,
}

impl WasiError {
    pub fn new(errno: WasiErrno, message: impl Into<String>) -> Self {
        Self {
            errno,
            message: message.into(),
            capability: None,
        }
    }

    pub fn capability_denied(capability: &str, message: impl Into<String>) -> Self {
        Self {
            errno: WasiErrno::Acces,
            message: message.into(),
            capability: Some(capability.to_string()),
        }
    }

    pub fn permission_denied(message: impl Into<String>) -> Self {
        Self {
            errno: WasiErrno::Acces,
            message: message.into(),
            capability: None,
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            errno: WasiErrno::Noent,
            message: message.into(),
            capability: None,
        }
    }

    pub fn bad_fd(fd: u32) -> Self {
        Self {
            errno: WasiErrno::Badf,
            message: format!("Bad file descriptor: {}", fd),
            capability: None,
        }
    }

    pub fn unsupported(operation: &str) -> Self {
        Self {
            errno: WasiErrno::Nosys,
            message: format!("Operation unsupported in WASI runtime: {}", operation),
            capability: None,
        }
    }

    pub fn read_only(message: impl Into<String>) -> Self {
        Self {
            errno: WasiErrno::Roofs,
            message: message.into(),
            capability: None,
        }
    }

    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self {
            errno: WasiErrno::Inval,
            message: message.into(),
            capability: None,
        }
    }

    pub fn from_io_error(err: &std::io::Error) -> Self {
        let errno = match err.kind() {
            std::io::ErrorKind::NotFound => WasiErrno::Noent,
            std::io::ErrorKind::PermissionDenied => WasiErrno::Acces,
            std::io::ErrorKind::AlreadyExists => WasiErrno::Exist,
            std::io::ErrorKind::InvalidInput | std::io::ErrorKind::InvalidData => WasiErrno::Inval,
            std::io::ErrorKind::TimedOut => WasiErrno::Timedout,
            std::io::ErrorKind::BrokenPipe => WasiErrno::Pipe,
            std::io::ErrorKind::Unsupported => WasiErrno::Nosys,
            _ => WasiErrno::Io,
        };
        Self {
            errno,
            message: err.to_string(),
            capability: None,
        }
    }

    pub fn to_io_error(&self) -> std::io::Error {
        let kind = match self.errno {
            WasiErrno::Noent => std::io::ErrorKind::NotFound,
            WasiErrno::Acces | WasiErrno::Perm | WasiErrno::Roofs => {
                std::io::ErrorKind::PermissionDenied
            }
            WasiErrno::Exist => std::io::ErrorKind::AlreadyExists,
            WasiErrno::Inval => std::io::ErrorKind::InvalidInput,
            WasiErrno::Timedout => std::io::ErrorKind::TimedOut,
            WasiErrno::Pipe => std::io::ErrorKind::BrokenPipe,
            WasiErrno::Nosys | WasiErrno::Notsup => std::io::ErrorKind::Unsupported,
            _ => std::io::ErrorKind::Other,
        };
        std::io::Error::new(kind, self.message.clone())
    }

    pub fn to_exit<T>(&self) -> Exit<T, String> {
        Exit::Failure(Cause::Fail(self.to_string()))
    }
}

impl fmt::Display for WasiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref cap) = self.capability {
            write!(
                f,
                "WASI error {}: {} [capability: {}]",
                self.errno, self.message, cap
            )
        } else {
            write!(f, "WASI error {}: {}", self.errno, self.message)
        }
    }
}

impl std::error::Error for WasiError {}

// =============================================================================
// 2. WASI Capabilities Specification
// =============================================================================

/// Capability mask for WASI guest execution.
/// By default in pure sandbox, capabilities must be explicitly enabled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasiCapabilities {
    pub allow_fs: bool,
    pub allow_env: bool,
    pub allow_clocks: bool,
    pub allow_random: bool,
    pub allow_stdio: bool,
    pub allow_unsupported: bool,
}

impl Default for WasiCapabilities {
    fn default() -> Self {
        Self {
            allow_fs: true,
            allow_env: true,
            allow_clocks: true,
            allow_random: true,
            allow_stdio: true,
            allow_unsupported: false,
        }
    }
}

impl WasiCapabilities {
    /// Pure sandbox mode: all host access denied by default.
    pub fn pure_sandbox() -> Self {
        Self {
            allow_fs: false,
            allow_env: false,
            allow_clocks: false,
            allow_random: false,
            allow_stdio: false,
            allow_unsupported: false,
        }
    }

    /// Permissive mode for CLI execution with preopens.
    pub fn permissive() -> Self {
        Self::default()
    }

    pub fn with_fs(mut self, allow: bool) -> Self {
        self.allow_fs = allow;
        self
    }

    pub fn with_env(mut self, allow: bool) -> Self {
        self.allow_env = allow;
        self
    }

    pub fn with_clocks(mut self, allow: bool) -> Self {
        self.allow_clocks = allow;
        self
    }

    pub fn with_random(mut self, allow: bool) -> Self {
        self.allow_random = allow;
        self
    }

    pub fn with_stdio(mut self, allow: bool) -> Self {
        self.allow_stdio = allow;
        self
    }

    pub fn check_fs(&self) -> Result<(), WasiError> {
        if !self.allow_fs {
            Err(WasiError::capability_denied(
                "allow_fs",
                "WASI filesystem capability is denied",
            ))
        } else {
            Ok(())
        }
    }

    pub fn check_env(&self) -> Result<(), WasiError> {
        if !self.allow_env {
            Err(WasiError::capability_denied(
                "allow_env",
                "WASI environment capability is denied",
            ))
        } else {
            Ok(())
        }
    }

    pub fn check_clocks(&self) -> Result<(), WasiError> {
        if !self.allow_clocks {
            Err(WasiError::capability_denied(
                "allow_clocks",
                "WASI clock capability is denied",
            ))
        } else {
            Ok(())
        }
    }

    pub fn check_random(&self) -> Result<(), WasiError> {
        if !self.allow_random {
            Err(WasiError::capability_denied(
                "allow_random",
                "WASI random/entropy capability is denied",
            ))
        } else {
            Ok(())
        }
    }
}

// =============================================================================
// 3. Preopened Directories (Preopens) and Path Sandboxing
// =============================================================================

/// A preopened directory mounted into the WASI guest namespace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasiPreopen {
    pub guest_path: PathBuf,
    pub host_path: PathBuf,
    pub read_only: bool,
}

impl WasiPreopen {
    pub fn new(guest_path: impl AsRef<Path>, host_path: impl AsRef<Path>) -> Self {
        Self {
            guest_path: normalize_path(guest_path.as_ref()),
            host_path: normalize_path(host_path.as_ref()),
            read_only: false,
        }
    }

    pub fn read_only(guest_path: impl AsRef<Path>, host_path: impl AsRef<Path>) -> Self {
        Self {
            guest_path: normalize_path(guest_path.as_ref()),
            host_path: normalize_path(host_path.as_ref()),
            read_only: true,
        }
    }

    /// Check if a guest path falls within this preopen mount.
    pub fn matches_guest(&self, requested: &Path) -> bool {
        let norm = normalize_path(requested);
        norm.starts_with(&self.guest_path)
    }

    /// Resolve a requested guest path to the backing virtual/host path.
    /// Traversal attempts (`..`) above the mount point are strictly rejected with `EACCES`.
    pub fn resolve(&self, requested: &Path) -> Result<PathBuf, WasiError> {
        let norm = normalize_path(requested);
        if !norm.starts_with(&self.guest_path) {
            return Err(WasiError::permission_denied(format!(
                "Path '{}' is outside preopen root '{}'",
                requested.display(),
                self.guest_path.display()
            )));
        }

        let rel = norm
            .strip_prefix(&self.guest_path)
            .map_err(|_| WasiError::permission_denied("Failed to strip prefix"))?;

        // Double check no escaping components remain
        for comp in rel.components() {
            if let Component::ParentDir = comp {
                return Err(WasiError::permission_denied(
                    "Path traversal attempt (..) detected",
                ));
            }
        }

        Ok(self.host_path.join(rel))
    }
}

/// Normalizes a path representation, stripping redundant `.` components
/// and verifying that `..` components do not escape root.
pub fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    let is_absolute = path.is_absolute() || path.starts_with("/");

    if is_absolute {
        normalized.push("/");
    }

    for comp in path.components() {
        match comp {
            Component::RootDir => {}
            Component::CurDir => {}
            Component::ParentDir => {
                // Cannot pop beyond root
                if !normalized.pop() && !is_absolute {
                    // Relative path popping too far
                }
            }
            Component::Normal(c) => normalized.push(c),
            Component::Prefix(p) => normalized.push(p.as_os_str()),
        }
    }

    if normalized.as_os_str().is_empty() {
        PathBuf::from(if is_absolute { "/" } else { "." })
    } else {
        normalized
    }
}

// =============================================================================
// 4. File Descriptor Table & Handle Tracking
// =============================================================================

pub const FD_STDIN: u32 = 0;
pub const FD_STDOUT: u32 = 1;
pub const FD_STDERR: u32 = 2;
pub const FIRST_PREOPEN_FD: u32 = 3;

/// Open handle in the WASI file descriptor table.
#[derive(Debug, Clone)]
pub struct WasiFileHandle {
    pub fd: u32,
    pub path: PathBuf,
    pub is_dir: bool,
    pub read_only: bool,
    pub cursor: u64,
    pub is_closed: bool,
    pub is_preopen: bool,
    pub preopen_index: Option<usize>,
}

/// Table of open WASI descriptors with deterministic ID allocation
/// and guaranteed cleanup upon destruction.
pub struct WasiFdTable {
    next_fd: AtomicUsize,
    entries: Mutex<HashMap<u32, WasiFileHandle>>,
    active_count: Arc<AtomicUsize>,
}

impl Default for WasiFdTable {
    fn default() -> Self {
        Self::new()
    }
}

impl WasiFdTable {
    pub fn new() -> Self {
        let active_count = Arc::new(AtomicUsize::new(0));
        let mut initial_map = HashMap::new();

        // Register stdio standard descriptors
        initial_map.insert(
            FD_STDIN,
            WasiFileHandle {
                fd: FD_STDIN,
                path: PathBuf::from("<stdin>"),
                is_dir: false,
                read_only: true,
                cursor: 0,
                is_closed: false,
                is_preopen: false,
                preopen_index: None,
            },
        );
        initial_map.insert(
            FD_STDOUT,
            WasiFileHandle {
                fd: FD_STDOUT,
                path: PathBuf::from("<stdout>"),
                is_dir: false,
                read_only: false,
                cursor: 0,
                is_closed: false,
                is_preopen: false,
                preopen_index: None,
            },
        );
        initial_map.insert(
            FD_STDERR,
            WasiFileHandle {
                fd: FD_STDERR,
                path: PathBuf::from("<stderr>"),
                is_dir: false,
                read_only: false,
                cursor: 0,
                is_closed: false,
                is_preopen: false,
                preopen_index: None,
            },
        );

        Self {
            next_fd: AtomicUsize::new(FIRST_PREOPEN_FD as usize),
            entries: Mutex::new(initial_map),
            active_count,
        }
    }

    /// Register a preopen directory and assign its descriptor.
    pub fn register_preopen(
        &self,
        guest_path: PathBuf,
        preopen_index: usize,
        read_only: bool,
    ) -> u32 {
        let fd = self.next_fd.fetch_add(1, Ordering::SeqCst) as u32;
        let handle = WasiFileHandle {
            fd,
            path: guest_path,
            is_dir: true,
            read_only,
            cursor: 0,
            is_closed: false,
            is_preopen: true,
            preopen_index: Some(preopen_index),
        };
        let mut map = self.entries.lock().unwrap();
        map.insert(fd, handle);
        fd
    }

    /// Open a new regular file descriptor.
    pub fn open_file(
        &self,
        resolved_path: PathBuf,
        is_dir: bool,
        read_only: bool,
    ) -> Result<u32, WasiError> {
        let fd = self.next_fd.fetch_add(1, Ordering::SeqCst) as u32;
        let handle = WasiFileHandle {
            fd,
            path: resolved_path,
            is_dir,
            read_only,
            cursor: 0,
            is_closed: false,
            is_preopen: false,
            preopen_index: None,
        };
        let mut map = self.entries.lock().unwrap();
        map.insert(fd, handle);
        self.active_count.fetch_add(1, Ordering::SeqCst);
        Ok(fd)
    }

    /// Close a file descriptor.
    pub fn close(&self, fd: u32) -> Result<(), WasiError> {
        let mut map = self.entries.lock().unwrap();
        if let Some(entry) = map.remove(&fd) {
            if !entry.is_preopen && fd > FD_STDERR {
                self.active_count.fetch_sub(1, Ordering::SeqCst);
            }
            Ok(())
        } else {
            Err(WasiError::bad_fd(fd))
        }
    }

    /// Get descriptor handle clone.
    pub fn get(&self, fd: u32) -> Result<WasiFileHandle, WasiError> {
        let map = self.entries.lock().unwrap();
        map.get(&fd).cloned().ok_or_else(|| WasiError::bad_fd(fd))
    }

    /// Update descriptor cursor position.
    pub fn seek(&self, fd: u32, new_cursor: u64) -> Result<(), WasiError> {
        let mut map = self.entries.lock().unwrap();
        if let Some(entry) = map.get_mut(&fd) {
            if entry.is_closed {
                return Err(WasiError::bad_fd(fd));
            }
            entry.cursor = new_cursor;
            Ok(())
        } else {
            Err(WasiError::bad_fd(fd))
        }
    }

    /// Returns count of active non-stdio, non-preopen descriptors.
    pub fn active_fd_count(&self) -> usize {
        self.active_count.load(Ordering::SeqCst)
    }

    /// Close all active files and release tracked resources.
    pub fn cleanup_all(&self) {
        let mut map = self.entries.lock().unwrap();
        map.retain(|&fd, _| fd <= FD_STDERR || fd < FIRST_PREOPEN_FD);
        self.active_count.store(0, Ordering::SeqCst);
    }
}

impl Drop for WasiFdTable {
    fn drop(&mut self) {
        self.cleanup_all();
    }
}

// =============================================================================
// 5. Deterministic Clocks & Monotonic Clock Integration
// =============================================================================

pub const CLOCKID_REALTIME: u32 = 0;
pub const CLOCKID_MONOTONIC: u32 = 1;
pub const CLOCKID_PROCESS_CPUTIME_ID: u32 = 2;
pub const CLOCKID_THREAD_CPUTIME_ID: u32 = 3;

/// Deterministic mock clock for WASI DST testing.
pub struct MockWasiClock {
    epoch_nanos: AtomicU64,
    monotonic_nanos: AtomicU64,
}

impl MockWasiClock {
    pub fn new(epoch_nanos: u64, monotonic_nanos: u64) -> Self {
        Self {
            epoch_nanos: AtomicU64::new(epoch_nanos),
            monotonic_nanos: AtomicU64::new(monotonic_nanos),
        }
    }

    pub fn advance(&self, duration: Duration) {
        let nanos = duration.as_nanos() as u64;
        self.epoch_nanos.fetch_add(nanos, Ordering::SeqCst);
        self.monotonic_nanos.fetch_add(nanos, Ordering::SeqCst);
    }

    pub fn set_monotonic_nanos(&self, nanos: u64) {
        self.monotonic_nanos.store(nanos, Ordering::SeqCst);
    }

    pub fn set_epoch_nanos(&self, nanos: u64) {
        self.epoch_nanos.store(nanos, Ordering::SeqCst);
    }
}

impl WallClock for MockWasiClock {
    fn now_epoch_nanos(&self) -> u64 {
        self.epoch_nanos.load(Ordering::SeqCst)
    }
}

impl MonotonicClock for MockWasiClock {
    fn now_nanos(&self) -> u64 {
        self.monotonic_nanos.load(Ordering::SeqCst)
    }
}

// Global thread-local or active deadline for WASI single-threaded timeout guard
static ACTIVE_WASI_DEADLINE_NANOS: AtomicU64 = AtomicU64::new(0);
static WASI_TIMEOUT_TRIGGERED: AtomicBool = AtomicBool::new(false);

/// Query current monotonic time in nanoseconds.
/// Under WASI targets, uses `clock_time_get(CLOCKID_MONOTONIC)`.
/// Under host test harnesses, uses platform monotonic counter.
pub fn monotonic_now_nanos() -> u64 {
    #[cfg(target_os = "wasi")]
    {
        // WASI Preview 1 clock_time_get(CLOCKID_MONOTONIC)
        unsafe {
            let mut time: u64 = 0;
            let ret = wasi_libc_clock_time_get(CLOCKID_MONOTONIC, 1_000, &mut time);
            if ret == 0 { time } else { 0 }
        }
    }
    #[cfg(not(target_os = "wasi"))]
    {
        use std::time::Instant;
        static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        let start = START.get_or_init(Instant::now);
        Instant::now().duration_since(*start).as_nanos() as u64
    }
}

#[cfg(target_os = "wasi")]
extern "C" {
    #[link_name = "clock_time_get"]
    fn wasi_libc_clock_time_get(clock_id: u32, precision: u64, time: *mut u64) -> u16;
}

/// Active timeout guard for WASI single-threaded runtime.
/// Dropping the guard uninstalls the active deadline and resets the trigger.
#[derive(Debug)]
pub struct WasiTimeoutGuard {
    deadline_nanos: u64,
    active: bool,
}

impl WasiTimeoutGuard {
    pub fn deadline_nanos(&self) -> u64 {
        self.deadline_nanos
    }

    pub fn has_timed_out(&self) -> bool {
        if !self.active {
            return false;
        }
        if WASI_TIMEOUT_TRIGGERED.load(Ordering::SeqCst) {
            return true;
        }
        let now = monotonic_now_nanos();
        if now >= self.deadline_nanos {
            WASI_TIMEOUT_TRIGGERED.store(true, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    pub fn cancel(&mut self) {
        self.active = false;
        ACTIVE_WASI_DEADLINE_NANOS.store(0, Ordering::SeqCst);
        WASI_TIMEOUT_TRIGGERED.store(false, Ordering::SeqCst);
    }
}

impl Drop for WasiTimeoutGuard {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// Install active timeout guard per RFC-WASM-WASI-01 §2.2.
pub fn install_wasi_timeout_guard(deadline_nanos: u64) -> WasiTimeoutGuard {
    ACTIVE_WASI_DEADLINE_NANOS.store(deadline_nanos, Ordering::SeqCst);
    WASI_TIMEOUT_TRIGGERED.store(false, Ordering::SeqCst);
    WasiTimeoutGuard {
        deadline_nanos,
        active: true,
    }
}

// =============================================================================
// 6. WASI Context and Runtime Environment
// =============================================================================

/// Encapsulated WASI runtime context holding all mounted capabilities,
/// preopens, environment variables, clocks, and stream buffers.
pub struct WasiContext {
    pub capabilities: WasiCapabilities,
    pub preopens: Vec<WasiPreopen>,
    pub fd_table: WasiFdTable,
    pub vfs: Arc<dyn VirtualFileSystem>,
    pub env: HashMap<String, String>,
    pub args: Vec<String>,
    pub clock: Arc<LustClock>,
    pub stdout_capture: Arc<Mutex<Vec<u8>>>,
    pub stderr_capture: Arc<Mutex<Vec<u8>>>,
    pub stdin_buffer: Arc<Mutex<Vec<u8>>>,
}

impl Default for WasiContext {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl WasiContext {
    pub fn builder() -> WasiContextBuilder {
        WasiContextBuilder::new()
    }

    /// Resolve a guest path against preopened directories.
    pub fn resolve_guest_path(&self, requested: &Path) -> Result<(usize, PathBuf), WasiError> {
        self.capabilities.check_fs()?;
        let norm = normalize_path(requested);

        for (idx, preopen) in self.preopens.iter().enumerate() {
            if preopen.matches_guest(&norm) {
                let resolved = preopen.resolve(&norm)?;
                return Ok((idx, resolved));
            }
        }

        Err(WasiError::permission_denied(format!(
            "Path '{}' is not contained in any preopened directory (Default-Deny sandbox)",
            requested.display()
        )))
    }
}

/// Fluent builder for `WasiContext`.
pub struct WasiContextBuilder {
    capabilities: WasiCapabilities,
    preopens: Vec<WasiPreopen>,
    vfs: Option<Arc<dyn VirtualFileSystem>>,
    env: HashMap<String, String>,
    args: Vec<String>,
    clock: Option<Arc<LustClock>>,
}

impl Default for WasiContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl WasiContextBuilder {
    pub fn new() -> Self {
        Self {
            capabilities: WasiCapabilities::default(),
            preopens: Vec::new(),
            vfs: None,
            env: HashMap::new(),
            args: Vec::new(),
            clock: None,
        }
    }

    pub fn pure_sandbox() -> Self {
        Self {
            capabilities: WasiCapabilities::pure_sandbox(),
            preopens: Vec::new(),
            vfs: Some(Arc::new(MemFs::new())),
            env: HashMap::new(),
            args: Vec::new(),
            clock: None,
        }
    }

    pub fn allow_fs(mut self, allow: bool) -> Self {
        self.capabilities.allow_fs = allow;
        self
    }

    pub fn allow_env(mut self, allow: bool) -> Self {
        self.capabilities.allow_env = allow;
        self
    }

    pub fn allow_clocks(mut self, allow: bool) -> Self {
        self.capabilities.allow_clocks = allow;
        self
    }

    pub fn allow_random(mut self, allow: bool) -> Self {
        self.capabilities.allow_random = allow;
        self
    }

    pub fn allow_stdio(mut self, allow: bool) -> Self {
        self.capabilities.allow_stdio = allow;
        self
    }

    pub fn with_capabilities(mut self, capabilities: WasiCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    pub fn preopen(mut self, guest_path: impl AsRef<Path>, host_path: impl AsRef<Path>) -> Self {
        self.preopens
            .push(WasiPreopen::new(guest_path.as_ref(), host_path.as_ref()));
        self
    }

    pub fn preopen_read_only(
        mut self,
        guest_path: impl AsRef<Path>,
        host_path: impl AsRef<Path>,
    ) -> Self {
        self.preopens.push(WasiPreopen::read_only(
            guest_path.as_ref(),
            host_path.as_ref(),
        ));
        self
    }

    pub fn with_vfs(mut self, vfs: Arc<dyn VirtualFileSystem>) -> Self {
        self.vfs = Some(vfs);
        self
    }

    pub fn env(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.env.insert(key.into(), val.into());
        self
    }

    pub fn env_vars(mut self, vars: HashMap<String, String>) -> Self {
        self.env = vars;
        self
    }

    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_clock(mut self, clock: Arc<LustClock>) -> Self {
        self.clock = Some(clock);
        self
    }

    pub fn build(self) -> WasiContext {
        let vfs = self.vfs.unwrap_or_else(|| Arc::new(MemFs::new()));
        let clock = self.clock.unwrap_or_else(|| Arc::new(LustClock::system()));
        let fd_table = WasiFdTable::new();

        // Register preopens into descriptor table and ensure backing directory exists in VFS
        for (idx, preopen) in self.preopens.iter().enumerate() {
            let _ = vfs.create_dir_all(&preopen.host_path);
            fd_table.register_preopen(preopen.guest_path.clone(), idx, preopen.read_only);
        }

        WasiContext {
            capabilities: self.capabilities,
            preopens: self.preopens,
            fd_table,
            vfs,
            env: self.env,
            args: self.args,
            clock,
            stdout_capture: Arc::new(Mutex::new(Vec::new())),
            stderr_capture: Arc::new(Mutex::new(Vec::new())),
            stdin_buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

// =============================================================================
// 7. WASI Runtime Adapter Implementation
// =============================================================================

/// Primary WASI adapter facilitating guest-host dispatch and capability enforcement.
pub struct WasiAdapter {
    ctx: Arc<WasiContext>,
}

impl WasiAdapter {
    pub fn new(ctx: WasiContext) -> Self {
        Self { ctx: Arc::new(ctx) }
    }

    pub fn context(&self) -> &WasiContext {
        &self.ctx
    }

    pub fn context_arc(&self) -> Arc<WasiContext> {
        self.ctx.clone()
    }

    // -------------------------------------------------------------------------
    // Filesystem Operations
    // -------------------------------------------------------------------------

    /// Get preopen metadata by file descriptor.
    pub fn fd_prestat_get(&self, fd: u32) -> Result<WasiPreopen, WasiError> {
        self.ctx.capabilities.check_fs()?;
        let handle = self.ctx.fd_table.get(fd)?;
        if !handle.is_preopen {
            return Err(WasiError::bad_fd(fd));
        }
        let idx = handle.preopen_index.ok_or_else(|| WasiError::bad_fd(fd))?;
        self.ctx
            .preopens
            .get(idx)
            .cloned()
            .ok_or_else(|| WasiError::bad_fd(fd))
    }

    /// Read string content from a guest path.
    pub fn path_read_to_string(&self, guest_path: &Path) -> Result<String, WasiError> {
        self.ctx.capabilities.check_fs()?;
        let (_, resolved) = self.ctx.resolve_guest_path(guest_path)?;
        self.ctx
            .vfs
            .read_to_string(&resolved)
            .map_err(|e| WasiError::from_io_error(&e))
    }

    /// Write content to a guest path.
    pub fn path_write(&self, guest_path: &Path, contents: &[u8]) -> Result<(), WasiError> {
        self.ctx.capabilities.check_fs()?;
        let (preopen_idx, resolved) = self.ctx.resolve_guest_path(guest_path)?;
        if self.ctx.preopens[preopen_idx].read_only {
            return Err(WasiError::read_only(format!(
                "Preopen '{}' is mounted read-only",
                self.ctx.preopens[preopen_idx].guest_path.display()
            )));
        }
        if let Some(parent) = resolved.parent() {
            let _ = self.ctx.vfs.create_dir_all(parent);
        }
        self.ctx
            .vfs
            .write(&resolved, contents)
            .map_err(|e| WasiError::from_io_error(&e))
    }

    /// Open a file in the guest preopen namespace.
    pub fn path_open(
        &self,
        guest_path: &Path,
        read_only: bool,
        create_if_missing: bool,
    ) -> Result<u32, WasiError> {
        self.ctx.capabilities.check_fs()?;
        let (preopen_idx, resolved) = self.ctx.resolve_guest_path(guest_path)?;
        let preopen_ro = self.ctx.preopens[preopen_idx].read_only;

        if !read_only && preopen_ro {
            return Err(WasiError::read_only("Cannot open for writing in read-only preopen"));
        }

        if !self.ctx.vfs.exists(&resolved) {
            if create_if_missing && !preopen_ro {
                if let Some(parent) = resolved.parent() {
                    let _ = self.ctx.vfs.create_dir_all(parent);
                }
                self.ctx
                    .vfs
                    .write(&resolved, &[])
                    .map_err(|e| WasiError::from_io_error(&e))?;
            } else {
                return Err(WasiError::not_found(format!(
                    "Path not found: {}",
                    guest_path.display()
                )));
            }
        }

        let is_dir = self
            .ctx
            .vfs
            .metadata(&resolved)
            .map(|m| m.is_dir)
            .unwrap_or(false);

        self.ctx
            .fd_table
            .open_file(resolved, is_dir, read_only || preopen_ro)
    }

    /// Read bytes from open descriptor.
    pub fn fd_read(&self, fd: u32, max_len: usize) -> Result<Vec<u8>, WasiError> {
        self.ctx.capabilities.check_fs()?;
        if fd == FD_STDIN {
            let mut stdin_guard = self.ctx.stdin_buffer.lock().unwrap();
            let to_read = stdin_guard.len().min(max_len);
            let bytes = stdin_guard.drain(..to_read).collect();
            return Ok(bytes);
        }

        let handle = self.ctx.fd_table.get(fd)?;
        let full_data = self
            .ctx
            .vfs
            .read(&handle.path)
            .map_err(|e| WasiError::from_io_error(&e))?;

        let offset = handle.cursor as usize;
        if offset >= full_data.len() {
            return Ok(Vec::new());
        }

        let end = (offset + max_len).min(full_data.len());
        let chunk = full_data[offset..end].to_vec();
        self.ctx.fd_table.seek(fd, end as u64)?;
        Ok(chunk)
    }

    /// Write bytes to open descriptor.
    pub fn fd_write(&self, fd: u32, data: &[u8]) -> Result<usize, WasiError> {
        if fd == FD_STDOUT {
            self.ctx.capabilities.check_fs()?;
            let mut buf = self.ctx.stdout_capture.lock().unwrap();
            buf.extend_from_slice(data);
            return Ok(data.len());
        }
        if fd == FD_STDERR {
            self.ctx.capabilities.check_fs()?;
            let mut buf = self.ctx.stderr_capture.lock().unwrap();
            buf.extend_from_slice(data);
            return Ok(data.len());
        }

        self.ctx.capabilities.check_fs()?;
        let handle = self.ctx.fd_table.get(fd)?;
        if handle.read_only {
            return Err(WasiError::read_only("Descriptor opened read-only"));
        }

        // For append or cursor writing
        self.ctx
            .vfs
            .append(&handle.path, data)
            .map_err(|e| WasiError::from_io_error(&e))?;
        let new_cursor = handle.cursor + data.len() as u64;
        self.ctx.fd_table.seek(fd, new_cursor)?;
        Ok(data.len())
    }

    /// Close descriptor.
    pub fn fd_close(&self, fd: u32) -> Result<(), WasiError> {
        self.ctx.fd_table.close(fd)
    }

    /// Directory listing.
    pub fn path_read_dir(&self, guest_path: &Path) -> Result<Vec<PathBuf>, WasiError> {
        self.ctx.capabilities.check_fs()?;
        let (_, resolved) = self.ctx.resolve_guest_path(guest_path)?;
        self.ctx
            .vfs
            .read_dir(&resolved)
            .map_err(|e| WasiError::from_io_error(&e))
    }

    /// File metadata query.
    pub fn path_metadata(&self, guest_path: &Path) -> Result<VfsMetadata, WasiError> {
        self.ctx.capabilities.check_fs()?;
        let (_, resolved) = self.ctx.resolve_guest_path(guest_path)?;
        self.ctx
            .vfs
            .metadata(&resolved)
            .map_err(|e| WasiError::from_io_error(&e))
    }

    // -------------------------------------------------------------------------
    // Environment & Arguments Operations
    // -------------------------------------------------------------------------

    pub fn environ_get(&self) -> Result<HashMap<String, String>, WasiError> {
        self.ctx.capabilities.check_env()?;
        Ok(self.ctx.env.clone())
    }

    pub fn environ_sizes_get(&self) -> Result<(usize, usize), WasiError> {
        self.ctx.capabilities.check_env()?;
        let count = self.ctx.env.len();
        let buf_size = self
            .ctx
            .env
            .iter()
            .map(|(k, v)| k.len() + v.len() + 2) // "k=v\0"
            .sum();
        Ok((count, buf_size))
    }

    pub fn var(&self, key: &str) -> Result<String, WasiError> {
        self.ctx.capabilities.check_env()?;
        self.ctx
            .env
            .get(key)
            .cloned()
            .ok_or_else(|| WasiError::not_found(format!("Environment variable '{}' not found", key)))
    }

    pub fn args_get(&self) -> Result<Vec<String>, WasiError> {
        self.ctx.capabilities.check_env()?;
        Ok(self.ctx.args.clone())
    }

    // -------------------------------------------------------------------------
    // Clock Operations
    // -------------------------------------------------------------------------

    pub fn clock_time_get(&self, clock_id: u32) -> Result<u64, WasiError> {
        self.ctx.capabilities.check_clocks()?;
        match clock_id {
            CLOCKID_REALTIME => Ok(self.ctx.clock.now_epoch_nanos()),
            CLOCKID_MONOTONIC | CLOCKID_PROCESS_CPUTIME_ID | CLOCKID_THREAD_CPUTIME_ID => {
                Ok(self.ctx.clock.now_nanos())
            }
            _ => Err(WasiError::invalid_argument(format!(
                "Invalid clock id: {}",
                clock_id
            ))),
        }
    }

    pub fn clock_res_get(&self, clock_id: u32) -> Result<u64, WasiError> {
        self.ctx.capabilities.check_clocks()?;
        match clock_id {
            CLOCKID_REALTIME | CLOCKID_MONOTONIC => Ok(1), // 1 nanosecond resolution
            _ => Err(WasiError::invalid_argument(format!(
                "Invalid clock id: {}",
                clock_id
            ))),
        }
    }

    // -------------------------------------------------------------------------
    // Random / Entropy Operations
    // -------------------------------------------------------------------------

    pub fn random_get(&self, buf: &mut [u8]) -> Result<(), WasiError> {
        self.ctx.capabilities.check_random()?;
        #[cfg(target_os = "wasi")]
        {
            unsafe {
                let ret = wasi_libc_random_get(buf.as_mut_ptr(), buf.len());
                if ret != 0 {
                    return Err(WasiError::new(WasiErrno::Io, "WASI random_get failed"));
                }
            }
            Ok(())
        }
        #[cfg(not(target_os = "wasi"))]
        {
            // Deterministic pseudo-random generation for testing
            let mut seed = self.ctx.clock.now_nanos();
            for byte in buf.iter_mut() {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                *byte = (seed >> 33) as u8;
            }
            Ok(())
        }
    }

    // -------------------------------------------------------------------------
    // Explicit Unsupported-Capability Behaviors
    // -------------------------------------------------------------------------

    pub fn sock_accept(&self) -> Result<(), WasiError> {
        Err(WasiError::unsupported(
            "sock_accept: sockets and networking are disabled in WASI runtime sandbox",
        ))
    }

    pub fn sock_connect(&self) -> Result<(), WasiError> {
        Err(WasiError::unsupported(
            "sock_connect: sockets and networking are disabled in WASI runtime sandbox",
        ))
    }

    pub fn sock_send(&self) -> Result<(), WasiError> {
        Err(WasiError::unsupported(
            "sock_send: socket communication is disabled in WASI runtime sandbox",
        ))
    }

    pub fn sock_recv(&self) -> Result<(), WasiError> {
        Err(WasiError::unsupported(
            "sock_recv: socket communication is disabled in WASI runtime sandbox",
        ))
    }

    pub fn sock_shutdown(&self) -> Result<(), WasiError> {
        Err(WasiError::unsupported(
            "sock_shutdown: socket operations are disabled in WASI runtime sandbox",
        ))
    }

    pub fn proc_spawn(&self) -> Result<(), WasiError> {
        Err(WasiError::unsupported(
            "proc_spawn: process spawning is unsupported in WASI guest environment",
        ))
    }

    pub fn proc_exec(&self) -> Result<(), WasiError> {
        Err(WasiError::unsupported(
            "proc_exec: process execution is unsupported in WASI guest environment",
        ))
    }

    pub fn proc_raise(&self) -> Result<(), WasiError> {
        Err(WasiError::unsupported(
            "proc_raise: signal propagation is unsupported in WASI guest environment",
        ))
    }

    pub fn thread_spawn(&self) -> Result<(), WasiError> {
        Err(WasiError::unsupported(
            "thread_spawn: thread spawning is unsupported in single-threaded WASI runtime",
        ))
    }

    pub fn ffi_load_library(&self) -> Result<(), WasiError> {
        Err(WasiError::unsupported(
            "ffi_load_library: dynamic library loading is unsupported in WASI runtime",
        ))
    }

    // -------------------------------------------------------------------------
    // VM Attachment and Integration
    // -------------------------------------------------------------------------

    /// Attach this WASI environment to a Lust VM.
    /// Replaces the VM's VFS, VirtualEnv, and capability mask.
    pub fn attach_to_vm(&self, vm: &mut VM) {
        // 1. Wire VFS
        vm.vfs = self.ctx.vfs.clone();

        // 2. Wire VirtualEnv
        let venv = VirtualEnv::new();
        for (k, v) in &self.ctx.env {
            venv.set(k, v);
        }
        vm.env = venv;

        // 3. Wire CapabilityMask to reflect WASI capabilities
        let mut mask = vm.config().capability_mask().clone();
        mask.allow_fs = self.ctx.capabilities.allow_fs;
        mask.allow_os = self.ctx.capabilities.allow_env;
        mask.allow_net = false; // Never allowed in default WASI
        mask.allow_ffi = false; // Never allowed in WASI
        vm.config.set_capability_mask(mask);
    }

    /// Read captured stdout output as string.
    pub fn stdout_string(&self) -> String {
        let buf = self.ctx.stdout_capture.lock().unwrap();
        String::from_utf8_lossy(&buf).to_string()
    }

    /// Read captured stderr output as string.
    pub fn stderr_string(&self) -> String {
        let buf = self.ctx.stderr_capture.lock().unwrap();
        String::from_utf8_lossy(&buf).to_string()
    }
}

#[cfg(target_os = "wasi")]
extern "C" {
    #[link_name = "random_get"]
    fn wasi_libc_random_get(buf: *mut u8, buf_len: usize) -> u16;
}

use std::path::{Path, PathBuf};

pub mod mem_fs;
pub mod os_fs;
pub mod env;
pub mod console;

pub use mem_fs::{MemFs, normalize_virtual_path};
pub use os_fs::OsFs;
pub use env::VirtualEnv;
pub use console::VirtualConsole;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VfsMetadata {
    pub is_file: bool,
    pub is_dir: bool,
    pub len: u64,
    pub modified_epoch_secs: u64,
}

impl VfsMetadata {
    pub fn is_file(&self) -> bool {
        self.is_file
    }

    pub fn is_dir(&self) -> bool {
        self.is_dir
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn modified_epoch_secs(&self) -> u64 {
        self.modified_epoch_secs
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectHostFsBypassPanic;

impl core::fmt::Display for DirectHostFsBypassPanic {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "DirectHostFsBypassPanic: Attempted host filesystem I/O while in simulation mode"
        )
    }
}

impl std::error::Error for DirectHostFsBypassPanic {}

/// VirtualFileSystem trait: deterministic abstraction over host and simulated filesystems.
/// Exactly 12 core methods per DST plan.
pub trait VirtualFileSystem: Send + Sync {
    fn read(&self, path: &Path) -> Result<Vec<u8>, std::io::Error>;
    fn read_to_string(&self, path: &Path) -> Result<String, std::io::Error>;
    fn write(&self, path: &Path, contents: &[u8]) -> Result<(), std::io::Error>;
    fn append(&self, path: &Path, contents: &[u8]) -> Result<(), std::io::Error>;
    fn create_dir_all(&self, path: &Path) -> Result<(), std::io::Error>;
    fn remove_file(&self, path: &Path) -> Result<(), std::io::Error>;
    fn remove_dir_all(&self, path: &Path) -> Result<(), std::io::Error>;
    fn rename(&self, from: &Path, to: &Path) -> Result<(), std::io::Error>;
    fn read_dir(&self, path: &Path) -> Result<Vec<PathBuf>, std::io::Error>;
    fn metadata(&self, path: &Path) -> Result<VfsMetadata, std::io::Error>;
    fn canonicalize(&self, path: &Path) -> Result<PathBuf, std::io::Error>;
    fn exists(&self, path: &Path) -> bool;
}

/// Retrieve the active VirtualFileSystem for the current thread/VM context,
/// or fallback to host `OsFs`.
pub fn current_vfs() -> std::sync::Arc<dyn VirtualFileSystem> {
    if let Some(vm_ptr) = crate::vm::current_vm_ptr() {
        unsafe { (*vm_ptr).vfs.clone() }
    } else {
        std::sync::Arc::new(os_fs::OsFs::new())
    }
}

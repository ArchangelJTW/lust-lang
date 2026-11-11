#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::UNIX_EPOCH;
use super::{DirectHostFsBypassPanic, VfsMetadata, VirtualFileSystem};

pub struct OsFs {
    simulated: AtomicBool,
}

impl Default for OsFs {
    fn default() -> Self {
        Self::new()
    }
}

impl OsFs {
    pub fn new() -> Self {
        Self {
            simulated: AtomicBool::new(false),
        }
    }

    pub fn simulated() -> Self {
        Self {
            simulated: AtomicBool::new(true),
        }
    }

    pub fn set_simulated(&self, sim: bool) {
        self.simulated.store(sim, Ordering::SeqCst);
    }

    pub fn is_simulated(&self) -> bool {
        if self.simulated.load(Ordering::SeqCst) {
            return true;
        }
        if crate::sim::is_simulated_mode() {
            return true;
        }
        if let Some(vm_ptr) = crate::vm::current_vm_ptr() {
            unsafe {
                if (*vm_ptr).clock.is_simulated() {
                    return true;
                }
            }
        }
        false
    }

    #[inline]
    fn check_guard(&self) {
        if self.is_simulated() {
            std::panic::panic_any(DirectHostFsBypassPanic);
        }
    }
}

impl VirtualFileSystem for OsFs {
    fn read(&self, path: &Path) -> Result<Vec<u8>, std::io::Error> {
        self.check_guard();
        std::fs::read(path)
    }

    fn read_to_string(&self, path: &Path) -> Result<String, std::io::Error> {
        self.check_guard();
        std::fs::read_to_string(path)
    }

    fn write(&self, path: &Path, contents: &[u8]) -> Result<(), std::io::Error> {
        self.check_guard();
        std::fs::write(path, contents)
    }

    fn append(&self, path: &Path, contents: &[u8]) -> Result<(), std::io::Error> {
        self.check_guard();
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        file.write_all(contents)
    }

    fn create_dir_all(&self, path: &Path) -> Result<(), std::io::Error> {
        self.check_guard();
        std::fs::create_dir_all(path)
    }

    fn remove_file(&self, path: &Path) -> Result<(), std::io::Error> {
        self.check_guard();
        std::fs::remove_file(path)
    }

    fn remove_dir_all(&self, path: &Path) -> Result<(), std::io::Error> {
        self.check_guard();
        std::fs::remove_dir_all(path)
    }

    fn rename(&self, from: &Path, to: &Path) -> Result<(), std::io::Error> {
        self.check_guard();
        std::fs::rename(from, to)
    }

    fn read_dir(&self, path: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
        self.check_guard();
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(path)? {
            entries.push(entry?.path());
        }
        entries.sort();
        Ok(entries)
    }

    fn metadata(&self, path: &Path) -> Result<VfsMetadata, std::io::Error> {
        self.check_guard();
        let meta = std::fs::metadata(path)?;
        let modified_epoch_secs = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Ok(VfsMetadata {
            is_file: meta.is_file(),
            is_dir: meta.is_dir(),
            len: meta.len(),
            modified_epoch_secs,
        })
    }

    fn canonicalize(&self, path: &Path) -> Result<PathBuf, std::io::Error> {
        self.check_guard();
        std::fs::canonicalize(path)
    }

    fn exists(&self, path: &Path) -> bool {
        self.check_guard();
        path.exists()
    }
}

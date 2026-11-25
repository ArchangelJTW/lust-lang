use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::effect_abi::{Cause, Exit, InterruptReason};
use crate::vfs::VirtualFileSystem;
use crate::vm::VM;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IoError {
    NotFound(String),
    PermissionDenied(String),
    AlreadyExists(String),
    InvalidInput(String),
    IoFailure(String),
    ResourceLimitExceeded(String),
    CapabilityDenied(String),
}

impl core::fmt::Display for IoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            IoError::NotFound(s) => write!(f, "NotFound: {}", s),
            IoError::PermissionDenied(s) => write!(f, "PermissionDenied: {}", s),
            IoError::AlreadyExists(s) => write!(f, "AlreadyExists: {}", s),
            IoError::InvalidInput(s) => write!(f, "InvalidInput: {}", s),
            IoError::IoFailure(s) => write!(f, "IoFailure: {}", s),
            IoError::ResourceLimitExceeded(s) => write!(f, "ResourceLimitExceeded: {}", s),
            IoError::CapabilityDenied(s) => write!(f, "CapabilityDenied: {}", s),
        }
    }
}

impl std::error::Error for IoError {}

pub fn map_std_io_error(err: std::io::Error) -> IoError {
    match err.kind() {
        std::io::ErrorKind::NotFound => IoError::NotFound(err.to_string()),
        std::io::ErrorKind::PermissionDenied => IoError::PermissionDenied(err.to_string()),
        std::io::ErrorKind::AlreadyExists => IoError::AlreadyExists(err.to_string()),
        std::io::ErrorKind::InvalidInput => IoError::InvalidInput(err.to_string()),
        _ => IoError::IoFailure(err.to_string()),
    }
}

/// Representation of an open file handle registered in the VM's TrustedHostResourceLedger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileHandle {
    pub handle_id: usize,
    pub slot_id: usize,
    pub path: String,
    pub mode: String,
}

pub struct OpenFileEntry {
    pub path: PathBuf,
    pub mode: String,
    pub cursor: usize,
    pub vfs: Arc<dyn VirtualFileSystem>,
    pub is_closed: bool,
}

static OPEN_FILES: Mutex<Option<HashMap<usize, OpenFileEntry>>> = Mutex::new(None);
static NEXT_HANDLE_ID: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(1);

fn with_open_files<R>(f: impl FnOnce(&mut HashMap<usize, OpenFileEntry>) -> R) -> R {
    let mut guard = OPEN_FILES.lock().unwrap();
    if guard.is_none() {
        *guard = Some(HashMap::new());
    }
    f(guard.as_mut().unwrap())
}

pub fn cleanup_file_handle_callback(handle_id: usize) {
    with_open_files(|map| {
        if let Some(entry) = map.get_mut(&handle_id) {
            entry.is_closed = true;
        }
        map.remove(&handle_id);
    });
}

pub fn active_open_files_count() -> usize {
    with_open_files(|map| map.len())
}

pub struct FileSystemService;

impl FileSystemService {
    pub fn check_capability(vm: &VM) -> Result<(), IoError> {
        let caps = vm.config().capability_mask();
        if !caps.allow_fs {
            return Err(IoError::CapabilityDenied(
                "FileSystem access denied: capability 'allow_fs' disabled".to_string(),
            ));
        }
        if !vm.config().is_std_module_allowed("io") {
            return Err(IoError::CapabilityDenied(
                "FileSystem access denied: module 'io' is not allowed".to_string(),
            ));
        }
        Ok(())
    }

    pub fn read_to_string(vm: &mut VM, path: &str) -> Exit<String, IoError> {
        if let Err(e) = Self::check_capability(vm) {
            return Exit::Failure(Cause::Fail(e));
        }
        if vm.charge_gas(10).is_err() {
            return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfGas));
        }
        let vfs = vm.vfs.clone();
        let path_obj = Path::new(path);
        let content = match vfs.read_to_string(path_obj) {
            Ok(c) => c,
            Err(e) => return Exit::Failure(Cause::Fail(map_std_io_error(e))),
        };
        if !vm.try_charge_memory_bytes(content.len()) {
            return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfMemory));
        }
        Exit::Success(content)
    }

    pub fn write_string(vm: &mut VM, path: &str, content: &str) -> Exit<(), IoError> {
        if let Err(e) = Self::check_capability(vm) {
            return Exit::Failure(Cause::Fail(e));
        }
        let gas_charge = 10 + (content.len() as u64 / 64);
        if vm.charge_gas(gas_charge).is_err() {
            return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfGas));
        }
        let vfs = vm.vfs.clone();
        let path_obj = Path::new(path);
        if let Some(parent) = path_obj.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = vfs.create_dir_all(parent);
            }
        }
        match vfs.write(path_obj, content.as_bytes()) {
            Ok(()) => Exit::Success(()),
            Err(e) => Exit::Failure(Cause::Fail(map_std_io_error(e))),
        }
    }

    pub fn exists(vm: &mut VM, path: &str) -> Exit<bool, IoError> {
        if let Err(e) = Self::check_capability(vm) {
            return Exit::Failure(Cause::Fail(e));
        }
        if vm.charge_gas(5).is_err() {
            return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfGas));
        }
        let vfs = vm.vfs.clone();
        let path_obj = Path::new(path);
        Exit::Success(vfs.exists(path_obj))
    }

    pub fn remove_file(vm: &mut VM, path: &str) -> Exit<(), IoError> {
        if let Err(e) = Self::check_capability(vm) {
            return Exit::Failure(Cause::Fail(e));
        }
        if vm.charge_gas(10).is_err() {
            return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfGas));
        }
        let vfs = vm.vfs.clone();
        let path_obj = Path::new(path);
        match vfs.remove_file(path_obj) {
            Ok(()) => Exit::Success(()),
            Err(e) => Exit::Failure(Cause::Fail(map_std_io_error(e))),
        }
    }

    pub fn create_dir_all(vm: &mut VM, path: &str) -> Exit<(), IoError> {
        if let Err(e) = Self::check_capability(vm) {
            return Exit::Failure(Cause::Fail(e));
        }
        if vm.charge_gas(10).is_err() {
            return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfGas));
        }
        let vfs = vm.vfs.clone();
        let path_obj = Path::new(path);
        match vfs.create_dir_all(path_obj) {
            Ok(()) => Exit::Success(()),
            Err(e) => Exit::Failure(Cause::Fail(map_std_io_error(e))),
        }
    }

    pub fn read_dir(vm: &mut VM, path: &str) -> Exit<Vec<String>, IoError> {
        if let Err(e) = Self::check_capability(vm) {
            return Exit::Failure(Cause::Fail(e));
        }
        if vm.charge_gas(10).is_err() {
            return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfGas));
        }
        let vfs = vm.vfs.clone();
        let path_obj = Path::new(path);
        match vfs.read_dir(path_obj) {
            Ok(paths) => {
                let mut list = Vec::new();
                let mut total_bytes = 0;
                for p in paths {
                    let s = p.to_string_lossy().into_owned();
                    total_bytes += s.len();
                    list.push(s);
                }
                if !vm.try_charge_memory_bytes(total_bytes) {
                    return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfMemory));
                }
                Exit::Success(list)
            }
            Err(e) => Exit::Failure(Cause::Fail(map_std_io_error(e))),
        }
    }

    pub fn open_scoped(vm: &mut VM, path: &str, mode: &str) -> Exit<FileHandle, IoError> {
        if let Err(e) = Self::check_capability(vm) {
            return Exit::Failure(Cause::Fail(e));
        }
        if vm.charge_gas(15).is_err() {
            return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfGas));
        }
        let path_buf = PathBuf::from(path);
        let vfs = vm.vfs.clone();

        if mode == "r" && !vfs.exists(&path_buf) {
            return Exit::Failure(Cause::Fail(IoError::NotFound(format!("File '{}' not found", path))));
        }

        let handle_id = NEXT_HANDLE_ID.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        let slot_id = match vm.host_ledger.acquire(handle_id, cleanup_file_handle_callback) {
            Ok(slot) => slot,
            Err(_) => {
                return Exit::Failure(Cause::Fail(IoError::ResourceLimitExceeded(
                    "TooManyOpenResources: Host resource ledger full (capacity 256 slots)".to_string(),
                )));
            }
        };

        with_open_files(|map| {
            map.insert(
                handle_id,
                OpenFileEntry {
                    path: path_buf,
                    mode: mode.to_string(),
                    cursor: 0,
                    vfs,
                    is_closed: false,
                },
            );
        });

        Exit::Success(FileHandle {
            handle_id,
            slot_id,
            path: path.to_string(),
            mode: mode.to_string(),
        })
    }

    pub fn read_handle(vm: &mut VM, handle: &FileHandle, len: usize) -> Exit<Vec<u8>, IoError> {
        if let Err(e) = Self::check_capability(vm) {
            return Exit::Failure(Cause::Fail(e));
        }
        if vm.charge_gas(10 + (len as u64 / 64)).is_err() {
            return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfGas));
        }
        if !vm.try_charge_memory_bytes(len) {
            return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfMemory));
        }

        let (all_bytes, cursor) = match with_open_files(|map| {
            let entry = map.get_mut(&handle.handle_id)?;
            if entry.is_closed {
                return None;
            }
            let bytes = entry.vfs.read(&entry.path).ok()?;
            let old_cur = entry.cursor;
            entry.cursor = entry.cursor.saturating_add(len);
            Some((bytes, old_cur))
        }) {
            Some(pair) => pair,
            None => {
                return Exit::Failure(Cause::Fail(IoError::IoFailure(
                    "File handle is closed or invalid".to_string(),
                )));
            }
        };

        if cursor >= all_bytes.len() {
            return Exit::Success(Vec::new());
        }
        let end = (cursor + len).min(all_bytes.len());
        Exit::Success(all_bytes[cursor..end].to_vec())
    }

    pub fn write_handle(vm: &mut VM, handle: &FileHandle, data: &[u8]) -> Exit<(), IoError> {
        if let Err(e) = Self::check_capability(vm) {
            return Exit::Failure(Cause::Fail(e));
        }
        if vm.charge_gas(10 + (data.len() as u64 / 64)).is_err() {
            return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfGas));
        }

        let result = with_open_files(|map| {
            let entry = map.get_mut(&handle.handle_id)?;
            if entry.is_closed {
                return None;
            }
            if entry.mode.contains('a') {
                Some(entry.vfs.append(&entry.path, data))
            } else {
                Some(entry.vfs.write(&entry.path, data))
            }
        });

        match result {
            Some(Ok(())) => Exit::Success(()),
            Some(Err(e)) => Exit::Failure(Cause::Fail(map_std_io_error(e))),
            _ => Exit::Failure(Cause::Fail(IoError::IoFailure(
                "File handle is closed or invalid".to_string(),
            ))),
        }
    }

    pub fn close_handle(vm: &mut VM, handle: &FileHandle) -> Exit<(), IoError> {
        vm.host_ledger.release(handle.slot_id);
        cleanup_file_handle_callback(handle.handle_id);
        Exit::Success(())
    }
}

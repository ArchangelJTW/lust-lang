use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use super::{VfsMetadata, VirtualFileSystem};

#[derive(Debug, Clone)]
enum MemNode {
    File {
        content: Vec<u8>,
        modified: u64,
    },
    Dir {
        modified: u64,
    },
}

/// Normalizes any input path into a canonical POSIX path rooted at `/workspace`.
/// Resolves `.` and `..` segments without escaping outside of `/workspace`.
pub fn normalize_virtual_path(path: &Path) -> PathBuf {
    let p_str = path.to_string_lossy();
    if p_str == "/workspace" || p_str == "workspace" || p_str == "." || p_str.is_empty() {
        return PathBuf::from("/workspace");
    }

    let stripped_str = if let Some(s) = p_str.strip_prefix("/workspace/") {
        s
    } else if let Some(s) = p_str.strip_prefix("workspace/") {
        s
    } else if let Some(s) = p_str.strip_prefix("/workspace") {
        s
    } else {
        &p_str
    };

    let sub_path = Path::new(stripped_str);
    let mut parts: Vec<String> = vec!["workspace".to_string()];

    for comp in sub_path.components() {
        match comp {
            std::path::Component::RootDir | std::path::Component::Prefix(_) => {}
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if parts.len() > 1 {
                    parts.pop();
                }
            }
            std::path::Component::Normal(c) => {
                let s = c.to_string_lossy().into_owned();
                parts.push(s);
            }
        }
    }

    let mut result = PathBuf::from("/");
    for p in parts {
        result.push(p);
    }
    result
}

fn current_epoch_secs() -> u64 {
    if let Some(vm_ptr) = crate::vm::current_vm_ptr() {
        unsafe {
            let vm = &*vm_ptr;
            vm.clock.now_epoch_nanos() / 1_000_000_000
        }
    } else {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

/// Deterministic In-Memory Virtual File System (MemFs).
/// Provides complete isolation from host disk and guaranteed reproducibility.
#[derive(Clone)]
pub struct MemFs {
    nodes: Arc<RwLock<HashMap<PathBuf, MemNode>>>,
}

impl Default for MemFs {
    fn default() -> Self {
        Self::new()
    }
}

impl MemFs {
    pub fn new() -> Self {
        let mut map = HashMap::new();
        map.insert(
            PathBuf::from("/workspace"),
            MemNode::Dir {
                modified: current_epoch_secs(),
            },
        );
        Self {
            nodes: Arc::new(RwLock::new(map)),
        }
    }

    pub fn reset(&self) {
        let mut map = self.nodes.write().unwrap();
        map.clear();
        map.insert(
            PathBuf::from("/workspace"),
            MemNode::Dir {
                modified: current_epoch_secs(),
            },
        );
    }

    pub fn dump_tree(&self) -> Vec<PathBuf> {
        let mut list: Vec<PathBuf> = self.nodes.read().unwrap().keys().cloned().collect();
        list.sort();
        list
    }
}

impl VirtualFileSystem for MemFs {
    fn read(&self, path: &Path) -> Result<Vec<u8>, std::io::Error> {
        let norm = normalize_virtual_path(path);
        let guard = self.nodes.read().unwrap();
        match guard.get(&norm) {
            Some(MemNode::File { content, .. }) => Ok(content.clone()),
            Some(MemNode::Dir { .. }) => Err(std::io::Error::new(
                ErrorKind::Other,
                format!("Is a directory: {}", path.display()),
            )),
            None => Err(std::io::Error::new(
                ErrorKind::NotFound,
                format!("No such file or directory: {}", path.display()),
            )),
        }
    }

    fn read_to_string(&self, path: &Path) -> Result<String, std::io::Error> {
        let bytes = self.read(path)?;
        String::from_utf8(bytes).map_err(|e| {
            std::io::Error::new(
                ErrorKind::InvalidData,
                format!("File '{}' is not valid UTF-8: {}", path.display(), e),
            )
        })
    }

    fn write(&self, path: &Path, contents: &[u8]) -> Result<(), std::io::Error> {
        let norm = normalize_virtual_path(path);
        if norm == Path::new("/workspace") {
            return Err(std::io::Error::new(
                ErrorKind::PermissionDenied,
                "Cannot overwrite virtual root directory",
            ));
        }

        let mut guard = self.nodes.write().unwrap();
        if let Some(parent) = norm.parent() {
            match guard.get(parent) {
                Some(MemNode::Dir { .. }) => {}
                Some(MemNode::File { .. }) => {
                    return Err(std::io::Error::new(
                        ErrorKind::Other,
                        format!("Parent path component is a file: {}", parent.display()),
                    ));
                }
                None => {
                    return Err(std::io::Error::new(
                        ErrorKind::NotFound,
                        format!("Parent directory does not exist: {}", parent.display()),
                    ));
                }
            }
        }

        guard.insert(
            norm,
            MemNode::File {
                content: contents.to_vec(),
                modified: current_epoch_secs(),
            },
        );
        Ok(())
    }

    fn append(&self, path: &Path, contents: &[u8]) -> Result<(), std::io::Error> {
        let norm = normalize_virtual_path(path);
        if norm == Path::new("/workspace") {
            return Err(std::io::Error::new(
                ErrorKind::PermissionDenied,
                "Cannot append to virtual root directory",
            ));
        }

        let mut guard = self.nodes.write().unwrap();
        if let Some(node) = guard.get_mut(&norm) {
            match node {
                MemNode::File { content, modified } => {
                    content.extend_from_slice(contents);
                    *modified = current_epoch_secs();
                    return Ok(());
                }
                MemNode::Dir { .. } => {
                    return Err(std::io::Error::new(
                        ErrorKind::Other,
                        format!("Cannot append: path is a directory: {}", path.display()),
                    ));
                }
            }
        }

        // File doesn't exist, check parent
        if let Some(parent) = norm.parent() {
            match guard.get(parent) {
                Some(MemNode::Dir { .. }) => {}
                Some(MemNode::File { .. }) => {
                    return Err(std::io::Error::new(
                        ErrorKind::Other,
                        format!("Parent path component is a file: {}", parent.display()),
                    ));
                }
                None => {
                    return Err(std::io::Error::new(
                        ErrorKind::NotFound,
                        format!("Parent directory does not exist: {}", parent.display()),
                    ));
                }
            }
        }

        guard.insert(
            norm,
            MemNode::File {
                content: contents.to_vec(),
                modified: current_epoch_secs(),
            },
        );
        Ok(())
    }

    fn create_dir_all(&self, path: &Path) -> Result<(), std::io::Error> {
        let norm = normalize_virtual_path(path);
        let mut guard = self.nodes.write().unwrap();

        let mut current = PathBuf::from("/");
        for comp in norm.components() {
            match comp {
                std::path::Component::RootDir | std::path::Component::Prefix(_) => {}
                std::path::Component::Normal(c) => {
                    current.push(c);
                    if let Some(existing) = guard.get(&current) {
                        if let MemNode::File { .. } = existing {
                            return Err(std::io::Error::new(
                                ErrorKind::AlreadyExists,
                                format!("Path component is an existing file: {}", current.display()),
                            ));
                        }
                    } else {
                        guard.insert(
                            current.clone(),
                            MemNode::Dir {
                                modified: current_epoch_secs(),
                            },
                        );
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn remove_file(&self, path: &Path) -> Result<(), std::io::Error> {
        let norm = normalize_virtual_path(path);
        let mut guard = self.nodes.write().unwrap();
        match guard.get(&norm) {
            Some(MemNode::File { .. }) => {
                guard.remove(&norm);
                Ok(())
            }
            Some(MemNode::Dir { .. }) => Err(std::io::Error::new(
                ErrorKind::PermissionDenied,
                format!("Cannot remove_file on directory: {}", path.display()),
            )),
            None => Err(std::io::Error::new(
                ErrorKind::NotFound,
                format!("No such file: {}", path.display()),
            )),
        }
    }

    fn remove_dir_all(&self, path: &Path) -> Result<(), std::io::Error> {
        let norm = normalize_virtual_path(path);
        let mut guard = self.nodes.write().unwrap();
        if !guard.contains_key(&norm) {
            return Err(std::io::Error::new(
                ErrorKind::NotFound,
                format!("No such directory: {}", path.display()),
            ));
        }

        if let Some(MemNode::File { .. }) = guard.get(&norm) {
            return Err(std::io::Error::new(
                ErrorKind::Other,
                format!("Path is a file, not a directory: {}", path.display()),
            ));
        }

        if norm == Path::new("/workspace") {
            guard.retain(|k, _| k == Path::new("/workspace"));
            return Ok(());
        }

        guard.retain(|k, _| k != &norm && !k.starts_with(&norm));
        Ok(())
    }

    fn rename(&self, from: &Path, to: &Path) -> Result<(), std::io::Error> {
        let norm_from = normalize_virtual_path(from);
        let norm_to = normalize_virtual_path(to);

        if norm_from == Path::new("/workspace") {
            return Err(std::io::Error::new(
                ErrorKind::PermissionDenied,
                "Cannot rename virtual root directory",
            ));
        }

        if norm_from == norm_to {
            return Ok(());
        }

        let mut guard = self.nodes.write().unwrap();
        if !guard.contains_key(&norm_from) {
            return Err(std::io::Error::new(
                ErrorKind::NotFound,
                format!("Source path does not exist: {}", from.display()),
            ));
        }

        let is_dir = matches!(guard.get(&norm_from), Some(MemNode::Dir { .. }));

        // Cannot move directory into its own subdirectory
        if is_dir && norm_to.starts_with(&norm_from) {
            return Err(std::io::Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "Cannot move directory into its own subdirectory: {} -> {}",
                    norm_from.display(),
                    norm_to.display()
                ),
            ));
        }

        // Destination parent must exist
        if let Some(parent) = norm_to.parent() {
            if !guard.contains_key(parent) {
                return Err(std::io::Error::new(
                    ErrorKind::NotFound,
                    format!("Destination parent directory does not exist: {}", parent.display()),
                ));
            }
        }

        // Check if destination exists
        if let Some(target_node) = guard.get(&norm_to) {
            match target_node {
                MemNode::Dir { .. } => {
                    if !is_dir {
                        // Cannot overwrite directory with file
                        return Err(std::io::Error::new(
                            ErrorKind::AlreadyExists,
                            format!("Cannot overwrite directory with file: {}", norm_to.display()),
                        ));
                    }
                    // If target is directory and source is directory, target directory must be empty
                    let is_empty = !guard.keys().any(|k| k != &norm_to && k.starts_with(&norm_to));
                    if !is_empty {
                        return Err(std::io::Error::new(
                            ErrorKind::AlreadyExists,
                            format!("Destination directory is not empty: {}", norm_to.display()),
                        ));
                    }
                    guard.remove(&norm_to);
                }
                MemNode::File { .. } => {
                    if is_dir {
                        // Cannot overwrite file with directory
                        return Err(std::io::Error::new(
                            ErrorKind::AlreadyExists,
                            format!("Cannot overwrite non-directory with directory: {}", norm_to.display()),
                        ));
                    }
                    guard.remove(&norm_to);
                }
            }
        }

        if is_dir {
            // Rename directory and all matching children
            let keys_to_move: Vec<PathBuf> = guard
                .keys()
                .filter(|k| k.starts_with(&norm_from))
                .cloned()
                .collect();

            for old_key in keys_to_move {
                if let Some(node) = guard.remove(&old_key) {
                    let rel = old_key.strip_prefix(&norm_from).unwrap_or(Path::new(""));
                    let new_key = norm_to.join(rel);
                    guard.insert(new_key, node);
                }
            }
        } else if let Some(node) = guard.remove(&norm_from) {
            guard.insert(norm_to, node);
        }

        Ok(())
    }

    fn read_dir(&self, path: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
        let norm = normalize_virtual_path(path);
        let guard = self.nodes.read().unwrap();

        match guard.get(&norm) {
            Some(MemNode::Dir { .. }) => {}
            Some(MemNode::File { .. }) => {
                return Err(std::io::Error::new(
                    ErrorKind::Other,
                    format!("Not a directory: {}", path.display()),
                ));
            }
            None => {
                return Err(std::io::Error::new(
                    ErrorKind::NotFound,
                    format!("No such directory: {}", path.display()),
                ));
            }
        }

        let mut entries = Vec::new();
        for k in guard.keys() {
            if k.parent() == Some(norm.as_path()) && k != &norm {
                entries.push(k.clone());
            }
        }

        // Deterministic lexicographical sorting guarantee
        entries.sort();
        Ok(entries)
    }

    fn metadata(&self, path: &Path) -> Result<VfsMetadata, std::io::Error> {
        let norm = normalize_virtual_path(path);
        let guard = self.nodes.read().unwrap();
        match guard.get(&norm) {
            Some(MemNode::File { content, modified }) => Ok(VfsMetadata {
                is_file: true,
                is_dir: false,
                len: content.len() as u64,
                modified_epoch_secs: *modified,
            }),
            Some(MemNode::Dir { modified }) => Ok(VfsMetadata {
                is_file: false,
                is_dir: true,
                len: 0,
                modified_epoch_secs: *modified,
            }),
            None => Err(std::io::Error::new(
                ErrorKind::NotFound,
                format!("No such file or directory: {}", path.display()),
            )),
        }
    }

    fn canonicalize(&self, path: &Path) -> Result<PathBuf, std::io::Error> {
        let norm = normalize_virtual_path(path);
        let guard = self.nodes.read().unwrap();
        if guard.contains_key(&norm) {
            Ok(norm)
        } else {
            Err(std::io::Error::new(
                ErrorKind::NotFound,
                format!("No such file or directory: {}", path.display()),
            ))
        }
    }

    fn exists(&self, path: &Path) -> bool {
        let norm = normalize_virtual_path(path);
        self.nodes.read().unwrap().contains_key(&norm)
    }
}

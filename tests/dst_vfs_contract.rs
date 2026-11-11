#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! DST Gate 3: VirtualFileSystem Contract Verification Suite.
//! Tests all 12 methods of `MemFs` and `OsFs`:
//! - Path normalization to canonical POSIX `/workspace`
//! - Deterministic lexicographical sorting in `read_dir`
//! - `canonicalize`, `append`, `rename`, `create_dir_all`, `remove_file`, `remove_dir_all`, `metadata`, `exists`

use lust::vfs::{MemFs, OsFs, VfsMetadata, VirtualFileSystem, normalize_virtual_path};
use std::path::{Path, PathBuf};

#[test]
fn test_memfs_path_normalization() {
    assert_eq!(normalize_virtual_path(Path::new(".")), PathBuf::from("/workspace"));
    assert_eq!(normalize_virtual_path(Path::new("/workspace")), PathBuf::from("/workspace"));
    assert_eq!(normalize_virtual_path(Path::new("/workspace/")), PathBuf::from("/workspace"));
    assert_eq!(normalize_virtual_path(Path::new("file.txt")), PathBuf::from("/workspace/file.txt"));
    assert_eq!(normalize_virtual_path(Path::new("./sub/file.txt")), PathBuf::from("/workspace/sub/file.txt"));
    assert_eq!(normalize_virtual_path(Path::new("a/b/../c.txt")), PathBuf::from("/workspace/a/c.txt"));
    assert_eq!(normalize_virtual_path(Path::new("/workspace/sub/../dir/file.txt")), PathBuf::from("/workspace/dir/file.txt"));
    // Verify nested workspace directory is preserved without collapsing
    assert_eq!(
        normalize_virtual_path(Path::new("/workspace/workspace/foo.lust")),
        PathBuf::from("/workspace/workspace/foo.lust")
    );
    // Escaping above /workspace is prevented and capped at /workspace
    assert_eq!(normalize_virtual_path(Path::new("/workspace/../../file.txt")), PathBuf::from("/workspace/file.txt"));
    assert_eq!(normalize_virtual_path(Path::new("../../file.txt")), PathBuf::from("/workspace/file.txt"));
}

#[test]
fn test_memfs_all_12_methods_contract() {
    let fs = MemFs::new();

    // 1. exists
    assert!(fs.exists(Path::new(".")));
    assert!(fs.exists(Path::new("/workspace")));
    assert!(!fs.exists(Path::new("hello.txt")));

    // 2. create_dir_all
    fs.create_dir_all(Path::new("src/nested")).expect("create_dir_all nested");
    assert!(fs.exists(Path::new("src")));
    assert!(fs.exists(Path::new("src/nested")));

    // 3. write
    let data = b"print(\"hello vfs\")";
    fs.write(Path::new("src/main.lust"), data).expect("write file");
    assert!(fs.exists(Path::new("src/main.lust")));

    // 4. read
    let read_bytes = fs.read(Path::new("src/main.lust")).expect("read file");
    assert_eq!(read_bytes, data);

    // 5. read_to_string
    let text = fs.read_to_string(Path::new("src/main.lust")).expect("read_to_string");
    assert_eq!(text, "print(\"hello vfs\")");

    // 6. append
    fs.append(Path::new("src/main.lust"), b"\n-- comment").expect("append to file");
    let appended_text = fs.read_to_string(Path::new("src/main.lust")).expect("read appended");
    assert_eq!(appended_text, "print(\"hello vfs\")\n-- comment");

    // Test append creating a new file
    fs.append(Path::new("src/appended_new.txt"), b"initial").expect("append new");
    assert_eq!(fs.read_to_string(Path::new("src/appended_new.txt")).unwrap(), "initial");

    // 7. metadata
    let meta: VfsMetadata = fs.metadata(Path::new("src/main.lust")).expect("metadata");
    assert!(meta.is_file());
    assert!(!meta.is_dir());
    assert_eq!(meta.len(), appended_text.len() as u64);

    let dir_meta = fs.metadata(Path::new("src")).expect("dir metadata");
    assert!(!dir_meta.is_file());
    assert!(dir_meta.is_dir());
    assert_eq!(dir_meta.len(), 0);

    // 8. read_dir with guaranteed deterministic sorting
    fs.write(Path::new("src/z.txt"), b"z").unwrap();
    fs.write(Path::new("src/a.txt"), b"a").unwrap();
    fs.write(Path::new("src/m.txt"), b"m").unwrap();

    let entries = fs.read_dir(Path::new("src")).expect("read_dir");
    let entry_names: Vec<String> = entries
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();

    // Must be strictly sorted lexicographically
    let mut expected = entry_names.clone();
    expected.sort();
    assert_eq!(entry_names, expected);
    assert!(entry_names.contains(&"a.txt".to_string()));
    assert!(entry_names.contains(&"m.txt".to_string()));
    assert!(entry_names.contains(&"z.txt".to_string()));
    assert!(entry_names.contains(&"main.lust".to_string()));
    assert!(entry_names.contains(&"nested".to_string()));

    // 9. canonicalize
    let canon = fs.canonicalize(Path::new("src/nested/../a.txt")).expect("canonicalize");
    assert_eq!(canon, PathBuf::from("/workspace/src/a.txt"));

    let canon_root = fs.canonicalize(Path::new(".")).expect("canonicalize root");
    assert_eq!(canon_root, PathBuf::from("/workspace"));

    // Canonicalize on nonexistent file must fail with NotFound
    let missing_canon = fs.canonicalize(Path::new("src/nonexistent.txt"));
    assert!(missing_canon.is_err());
    assert_eq!(missing_canon.unwrap_err().kind(), std::io::ErrorKind::NotFound);

    // 10. rename
    fs.rename(Path::new("src/a.txt"), Path::new("src/renamed_a.txt")).expect("rename");
    assert!(!fs.exists(Path::new("src/a.txt")));
    assert!(fs.exists(Path::new("src/renamed_a.txt")));
    assert_eq!(fs.read_to_string(Path::new("src/renamed_a.txt")).unwrap(), "a");

    // Rename directory and child paths
    fs.write(Path::new("src/nested/inner.txt"), b"inner").unwrap();
    fs.rename(Path::new("src/nested"), Path::new("src/nested_moved")).expect("rename dir");
    assert!(!fs.exists(Path::new("src/nested")));
    assert!(fs.exists(Path::new("src/nested_moved")));
    assert!(fs.exists(Path::new("src/nested_moved/inner.txt")));
    assert_eq!(fs.read_to_string(Path::new("src/nested_moved/inner.txt")).unwrap(), "inner");

    // Rename safety: cannot overwrite directory with file
    let err_file_to_dir = fs.rename(Path::new("src/renamed_a.txt"), Path::new("src/nested_moved"));
    assert!(err_file_to_dir.is_err());
    assert_eq!(err_file_to_dir.unwrap_err().kind(), std::io::ErrorKind::AlreadyExists);

    // Rename safety: cannot move directory into its own subdirectory
    let err_cycle = fs.rename(Path::new("src/nested_moved"), Path::new("src/nested_moved/sub"));
    assert!(err_cycle.is_err());
    assert_eq!(err_cycle.unwrap_err().kind(), std::io::ErrorKind::InvalidInput);

    // 11. remove_file
    fs.remove_file(Path::new("src/renamed_a.txt")).expect("remove_file");
    assert!(!fs.exists(Path::new("src/renamed_a.txt")));
    let remove_missing = fs.remove_file(Path::new("src/renamed_a.txt"));
    assert!(remove_missing.is_err());
    assert_eq!(remove_missing.unwrap_err().kind(), std::io::ErrorKind::NotFound);

    // 12. remove_dir_all
    fs.remove_dir_all(Path::new("src/nested_moved")).expect("remove_dir_all");
    assert!(!fs.exists(Path::new("src/nested_moved")));
    assert!(!fs.exists(Path::new("src/nested_moved/inner.txt")));
}

#[test]
fn test_osfs_contract_on_host_temp_dir() {
    let os_fs = OsFs::new();
    let temp_root = std::env::temp_dir().join(format!("lust_vfs_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp_root);

    // 1. create_dir_all
    os_fs.create_dir_all(&temp_root.join("sub")).expect("create_dir_all OsFs");
    assert!(os_fs.exists(&temp_root));
    assert!(os_fs.exists(&temp_root.join("sub")));

    // 2. write & read & read_to_string
    let file_path = temp_root.join("test.txt");
    os_fs.write(&file_path, b"test content").expect("write OsFs");
    assert_eq!(os_fs.read(&file_path).unwrap(), b"test content");
    assert_eq!(os_fs.read_to_string(&file_path).unwrap(), "test content");

    // 3. append
    os_fs.append(&file_path, b" appended").expect("append OsFs");
    assert_eq!(os_fs.read_to_string(&file_path).unwrap(), "test content appended");

    // 4. metadata
    let meta = os_fs.metadata(&file_path).expect("metadata OsFs");
    assert!(meta.is_file());
    assert!(!meta.is_dir());
    assert_eq!(meta.len(), "test content appended".len() as u64);

    // 5. read_dir sorting
    os_fs.write(&temp_root.join("z.txt"), b"z").unwrap();
    os_fs.write(&temp_root.join("a.txt"), b"a").unwrap();
    let entries = os_fs.read_dir(&temp_root).expect("read_dir OsFs");
    let mut sorted = entries.clone();
    sorted.sort();
    assert_eq!(entries, sorted);

    // 6. canonicalize
    let canon = os_fs.canonicalize(&file_path).expect("canonicalize OsFs");
    assert!(canon.is_absolute());

    // 7. rename
    let renamed = temp_root.join("renamed.txt");
    os_fs.rename(&file_path, &renamed).expect("rename OsFs");
    assert!(!os_fs.exists(&file_path));
    assert!(os_fs.exists(&renamed));

    // 8. remove_file & remove_dir_all
    os_fs.remove_file(&renamed).expect("remove_file OsFs");
    assert!(!os_fs.exists(&renamed));
    os_fs.remove_dir_all(&temp_root).expect("remove_dir_all OsFs");
    assert!(!os_fs.exists(&temp_root));
}

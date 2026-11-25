#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! DST Gate 3: Negative & Bypass Test Suite.
//! Verifies:
//! 1. `OsFs` in simulation mode strictly panics with `DirectHostFsBypassPanic` on all 12 methods.
//! 2. `SimModeGuard::enter()` activates global simulation facade guard.
//! 3. `MemFs` is completely isolated from the host filesystem (cannot see host files).
//! 4. `clippy.toml` gate configuration integrity: sha256 matches between root and fixture.

use lust::sim::{SimModeGuard, is_simulated_mode};
use lust::vfs::{DirectHostFsBypassPanic, MemFs, OsFs, VirtualFileSystem};
use std::path::Path;

fn assert_bypasses<F: FnOnce() + std::panic::UnwindSafe>(f: F) {
    let result = std::panic::catch_unwind(f);
    assert!(
        result.is_err(),
        "Expected panic from DirectHostFsBypassPanic, but operation succeeded"
    );
    let err = result.unwrap_err();
    let is_bypass_panic = err.is::<DirectHostFsBypassPanic>()
        || err
            .downcast_ref::<&str>()
            .map(|s| s.contains("DIRECT HOST FS ACCESS FORBIDDEN IN SIMULATION"))
            .unwrap_or(false)
        || err
            .downcast_ref::<String>()
            .map(|s| s.contains("DIRECT HOST FS ACCESS FORBIDDEN IN SIMULATION"))
            .unwrap_or(false);
    assert!(
        is_bypass_panic,
        "Expected DirectHostFsBypassPanic payload, got unexpected panic"
    );
}

#[test]
fn test_os_fs_panics_in_simulation_mode_all_12_methods() {
    let fs = OsFs::simulated();
    let dummy = Path::new("/workspace/test.txt");
    let dummy2 = Path::new("/workspace/test2.txt");

    assert_bypasses(|| {
        let _ = fs.read(dummy);
    });
    assert_bypasses(|| {
        let _ = fs.read_to_string(dummy);
    });
    assert_bypasses(|| {
        let _ = fs.write(dummy, b"content");
    });
    assert_bypasses(|| {
        let _ = fs.append(dummy, b"content");
    });
    assert_bypasses(|| {
        let _ = fs.create_dir_all(dummy);
    });
    assert_bypasses(|| {
        let _ = fs.remove_file(dummy);
    });
    assert_bypasses(|| {
        let _ = fs.remove_dir_all(dummy);
    });
    assert_bypasses(|| {
        let _ = fs.rename(dummy, dummy2);
    });
    assert_bypasses(|| {
        let _ = fs.read_dir(dummy);
    });
    assert_bypasses(|| {
        let _ = fs.metadata(dummy);
    });
    assert_bypasses(|| {
        let _ = fs.canonicalize(dummy);
    });
    assert_bypasses(|| {
        let _ = fs.exists(dummy);
    });
}

#[test]
fn test_sim_mode_guard_activates_os_fs_guard() {
    let fs = OsFs::new();

    // Outside simulation mode, check guard does not panic
    assert!(!is_simulated_mode());

    {
        let _sim_guard = SimModeGuard::enter();
        assert!(is_simulated_mode());

        // Now OsFs::new() must panic because simulation mode is active
        assert_bypasses(|| {
            let _ = fs.exists(Path::new("."));
        });
        assert_bypasses(|| {
            let _ = fs.read(Path::new("Cargo.toml"));
        });
    }

    // Exited guard, simulation mode reverted
    assert!(!is_simulated_mode());
}

#[test]
fn test_memfs_isolation_from_host_filesystem() {
    let mem_fs = MemFs::new();

    // Known host files MUST NOT exist in clean MemFs
    assert!(!mem_fs.exists(Path::new("Cargo.toml")));
    assert!(!mem_fs.exists(Path::new("/etc/passwd")));
    assert!(!mem_fs.exists(Path::new("/bin/sh")));

    // Reading host files must fail with NotFound
    let read_res = mem_fs.read(Path::new("Cargo.toml"));
    assert!(read_res.is_err());
    assert_eq!(read_res.unwrap_err().kind(), std::io::ErrorKind::NotFound);

    let read_str_res = mem_fs.read_to_string(Path::new("/etc/passwd"));
    assert!(read_str_res.is_err());
    assert_eq!(read_str_res.unwrap_err().kind(), std::io::ErrorKind::NotFound);

    // Canonicalizing nonexistent file fails
    let canon_res = mem_fs.canonicalize(Path::new("Cargo.toml"));
    assert!(canon_res.is_err());
    assert_eq!(canon_res.unwrap_err().kind(), std::io::ErrorKind::NotFound);

    // Write to MemFs and ensure it doesn't touch host
    mem_fs
        .write(Path::new("unique_isolation_probe.txt"), b"isolated")
        .expect("write to mem_fs");
    assert!(mem_fs.exists(Path::new("unique_isolation_probe.txt")));
    assert!(!Path::new("unique_isolation_probe.txt").exists());
    assert!(!Path::new("/workspace/unique_isolation_probe.txt").exists());
}

#[test]
fn test_clippy_gate_fixture_hash() {
    use std::fs;

    let root_clippy = fs::read("clippy.toml").expect("root clippy.toml must exist");
    let fixture_clippy =
        fs::read("tests/compile_fail/fs_gate/clippy.toml").expect("fixture clippy.toml must exist");

    assert_eq!(
        root_clippy, fixture_clippy,
        "Root clippy.toml and tests/compile_fail/fs_gate/clippy.toml must be identical"
    );

    let clippy_str = String::from_utf8_lossy(&root_clippy);
    assert!(clippy_str.contains("std::fs::read"));
    assert!(clippy_str.contains("std::fs::write"));
    assert!(clippy_str.contains("std::fs::File"));
    assert!(clippy_str.contains("std::fs::OpenOptions"));
    assert!(clippy_str.contains("std::fs::canonicalize"));
    assert!(clippy_str.contains("std::path::Path::exists"));
}

//! Example: Deterministic Fault Injection VFS (Virtual File System) in DST.
//! Demonstrates configuring simulated filesystem failures (PermissionDenied, DiskFull)
//! on specific path patterns and verifying fault handling without host environment mutation.

use std::io::ErrorKind;
use std::path::Path;
use std::sync::{Arc, Mutex};

use lust::sim::fault::{FaultConfiguration, FaultInjectedVfs};
use lust::vfs::{MemFs, VirtualFileSystem};

fn main() {
    println!("--- [Fault Injection VFS Example] ---");

    // 1. Create in-memory baseline filesystem
    let mem_fs = Arc::new(MemFs::new());
    mem_fs.create_dir_all(Path::new("/data")).unwrap();
    mem_fs
        .write(Path::new("/data/healthy.txt"), b"trusted content")
        .unwrap();
    mem_fs
        .write(Path::new("/data/secret.key"), b"super_secret_token")
        .unwrap();

    // 2. Configure deterministic fault injection rules
    let fault_config = Arc::new(Mutex::new(
        FaultConfiguration::new(12345)
            .with_vfs_read_error("secret.key", ErrorKind::PermissionDenied)
            .with_vfs_write_error("readonly.log", ErrorKind::ReadOnlyFilesystem),
    ));

    // 3. Wrap MemFs in FaultInjectedVfs
    let faulted_vfs = FaultInjectedVfs::new(mem_fs.clone(), fault_config);

    // 4. Test non-faulted read succeeds
    let healthy_data = faulted_vfs
        .read_to_string(Path::new("/data/healthy.txt"))
        .expect("Non-faulted file read must succeed");
    assert_eq!(healthy_data, "trusted content");
    println!("✓ Normal read succeeded: '{}'", healthy_data);

    // 5. Test injected read failure
    let read_err = faulted_vfs
        .read_to_string(Path::new("/data/secret.key"))
        .expect_err("Injected read failure must be triggered");
    assert_eq!(read_err.kind(), ErrorKind::PermissionDenied);
    println!("✓ Injected read fault caught: {:?}", read_err.kind());

    // 6. Test injected write failure
    let write_err = faulted_vfs
        .write(Path::new("/data/readonly.log"), b"audit event")
        .expect_err("Injected write failure must be triggered");
    assert_eq!(write_err.kind(), ErrorKind::ReadOnlyFilesystem);
    println!("✓ Injected write fault caught: {:?}", write_err.kind());

    // 7. Normal write succeeds
    assert!(
        faulted_vfs
            .write(Path::new("/data/output.txt"), b"success")
            .is_ok()
    );
    println!("✓ Non-faulted write succeeded");

    println!("All fault injection VFS assertions passed!");
}

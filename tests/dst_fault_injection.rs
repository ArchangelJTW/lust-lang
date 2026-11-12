//! Gate 5: Fault Injection Engine Tests (DST Phase 5, D8).
//!
//! Validates:
//! 1. VFS I/O error injection:
//!    - Read errors on matching path patterns.
//!    - Write errors on matching path patterns.
//!    - Deterministic probability-driven error injection with seed reproducibility.
//!    - Pass-through of non-faulted operations to underlying VFS (all 12 methods).
//! 2. Simulated memory failures (OOM):
//!    - Step-based OOM.
//!    - Byte-limit OOM.
//! 3. Spontaneous fiber cancellation / interrupt injection.

use std::io::ErrorKind;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use lust::effect_abi::InterruptReason;
use lust::sim::fault::{FaultConfiguration, FaultInjectedVfs, VfsOp};
use lust::vfs::{MemFs, VirtualFileSystem};
use lust::vm::{TaskState, VM};

/// Helper: run a test with an external 5-second watchdog.
fn with_test_watchdog<F: FnOnce() + Send + 'static>(name: &'static str, f: F) {
    let done = Arc::new(AtomicBool::new(false));
    let done_clone = done.clone();

    let handle = std::thread::spawn(move || {
        f();
        done_clone.store(true, Ordering::SeqCst);
    });

    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(5);

    while !done.load(Ordering::SeqCst) {
        if start.elapsed() > timeout {
            panic!("FAIL: WATCHDOG_TIMEOUT: Test '{}' exceeded 5s timeout", name);
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    handle.join().expect("Test thread panicked");
}

#[test]
fn test_fault_injected_vfs_read_and_write_intercept() {
    with_test_watchdog("fault_vfs_read_write", || {
        let mem_fs = Arc::new(MemFs::new());
        // Populate baseline files in MemFs
        mem_fs
            .write(Path::new("/workspace/ok.txt"), b"healthy data")
            .unwrap();
        mem_fs
            .write(Path::new("/workspace/corrupt.txt"), b"corrupted data")
            .unwrap();

        let config = Arc::new(Mutex::new(
            FaultConfiguration::new(42)
                .with_vfs_read_error("corrupt.txt", ErrorKind::PermissionDenied)
                .with_vfs_write_error("readonly.txt", ErrorKind::PermissionDenied),
        ));

        let faulted_vfs = FaultInjectedVfs::new(mem_fs.clone(), config.clone());

        // 1. Reading ok.txt succeeds
        let ok_content = faulted_vfs.read_to_string(Path::new("/workspace/ok.txt")).unwrap();
        assert_eq!(ok_content, "healthy data");

        // 2. Reading corrupt.txt fails with PermissionDenied
        let err = faulted_vfs
            .read_to_string(Path::new("/workspace/corrupt.txt"))
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::PermissionDenied);

        // 3. Writing to readonly.txt fails
        let write_err = faulted_vfs
            .write(Path::new("/workspace/readonly.txt"), b"new")
            .unwrap_err();
        assert_eq!(write_err.kind(), ErrorKind::PermissionDenied);

        // 4. Writing to ok.txt succeeds
        assert!(faulted_vfs.write(Path::new("/workspace/ok.txt"), b"updated").is_ok());
        assert_eq!(
            faulted_vfs.read_to_string(Path::new("/workspace/ok.txt")).unwrap(),
            "updated"
        );

        // Confirm fault counter incremented
        assert_eq!(config.lock().unwrap().vfs_fault_count, 2);
    });
}

#[test]
fn test_fault_injected_vfs_all_12_methods_delegation() {
    with_test_watchdog("fault_vfs_all_12_methods", || {
        let mem_fs = Arc::new(MemFs::new());
        let config = Arc::new(Mutex::new(FaultConfiguration::new(1234)));
        let vfs = FaultInjectedVfs::new(mem_fs.clone(), config);

        // 1. create_dir_all
        let dir = Path::new("/workspace/test_dir");
        assert!(vfs.create_dir_all(dir).is_ok());

        // 2. exists
        assert!(vfs.exists(dir));

        // 3. write
        let file = Path::new("/workspace/test_dir/file.txt");
        assert!(vfs.write(file, b"content").is_ok());

        // 4. append
        assert!(vfs.append(file, b" appended").is_ok());

        // 5. read
        let bytes = vfs.read(file).unwrap();
        assert_eq!(bytes, b"content appended");

        // 6. read_to_string
        let string_data = vfs.read_to_string(file).unwrap();
        assert_eq!(string_data, "content appended");

        // 7. metadata
        let meta = vfs.metadata(file).unwrap();
        assert!(meta.is_file());
        assert_eq!(meta.len(), 16);

        // 8. read_dir
        let entries = vfs.read_dir(dir).unwrap();
        assert_eq!(entries.len(), 1);

        // 9. rename
        let renamed = Path::new("/workspace/test_dir/renamed.txt");
        assert!(vfs.rename(file, renamed).is_ok());
        assert!(!vfs.exists(file));
        assert!(vfs.exists(renamed));

        // 10. canonicalize
        let canon = vfs.canonicalize(Path::new("/workspace/test_dir/../test_dir/renamed.txt")).unwrap();
        assert_eq!(canon, Path::new("/workspace/test_dir/renamed.txt"));

        // 11. remove_file
        assert!(vfs.remove_file(renamed).is_ok());
        assert!(!vfs.exists(renamed));

        // 12. remove_dir_all
        assert!(vfs.remove_dir_all(dir).is_ok());
        assert!(!vfs.exists(dir));
    });
}

#[test]
fn test_fault_configuration_deterministic_probability() {
    with_test_watchdog("fault_vfs_probability", || {
        let seed = 98765;
        // Two independent configurations with the same seed must produce identical fault outcomes
        let mut cfg1 = FaultConfiguration::new(seed).with_vfs_error_rate(
            Some(VfsOp::Read),
            0.5,
            ErrorKind::BrokenPipe,
        );
        let mut cfg2 = FaultConfiguration::new(seed).with_vfs_error_rate(
            Some(VfsOp::Read),
            0.5,
            ErrorKind::BrokenPipe,
        );

        let path = Path::new("/workspace/test.txt");
        let results1: Vec<bool> = (0..50)
            .map(|_| cfg1.should_fault_vfs(VfsOp::Read, path).is_some())
            .collect();
        let results2: Vec<bool> = (0..50)
            .map(|_| cfg2.should_fault_vfs(VfsOp::Read, path).is_some())
            .collect();

        // Deterministic bit-for-bit replay
        assert_eq!(results1, results2);
        // Ensure that probability 0.5 triggered at least some faults and some successes
        assert!(results1.iter().any(|&f| f));
        assert!(results1.iter().any(|&f| !f));
    });
}

#[test]
fn test_memory_fault_policy_step_and_limit() {
    with_test_watchdog("fault_memory_policy", || {
        // Step OOM
        let mut cfg_step = FaultConfiguration::new(1).with_oom_at_step(10);
        assert!(!cfg_step.should_fault_memory(9, 1024));
        assert!(cfg_step.should_fault_memory(10, 1024));
        assert!(!cfg_step.should_fault_memory(11, 1024));

        // Bytes limit OOM
        let mut cfg_bytes = FaultConfiguration::new(1).with_oom_at_bytes(2048);
        assert!(!cfg_bytes.should_fault_memory(1, 1024));
        assert!(!cfg_bytes.should_fault_memory(2, 2047));
        assert!(cfg_bytes.should_fault_memory(3, 2048));
        assert!(cfg_bytes.should_fault_memory(4, 4096));
    });
}

#[test]
fn test_spontaneous_interrupt_injection() {
    with_test_watchdog("fault_spontaneous_interrupt", || {
        let mut vm = VM::new();
        let handle = vm.create_native_future_task();
        let task_id = handle.id();

        // Emulate task inside interrupt-mask (acquire-release critical section)
        vm.get_task_instance_mut(handle)
            .unwrap()
            .fiber_context
            .enter_mask();

        let mut config = FaultConfiguration::new(1).with_spontaneous_interrupt(
            Some(task_id),
            5,
            InterruptReason::Cancelled,
        );

        // Apply fault at step 4: no interrupt
        config.apply_step_faults(&mut vm, 4).unwrap();
        assert_eq!(
            vm.get_task_instance(handle).unwrap().fiber_context.pending_interrupt,
            None
        );

        // Apply fault at step 5: spontaneous interrupt injected and preserved in mask!
        config.apply_step_faults(&mut vm, 5).unwrap();
        assert_eq!(
            vm.get_task_instance(handle).unwrap().fiber_context.pending_interrupt,
            Some(InterruptReason::Cancelled)
        );
        assert_eq!(config.interrupt_fault_count, 1);

        // Upon unmasking, pending interrupt is delivered and task halts in Stopped
        let task = vm.get_task_instance_mut(handle).unwrap();
        let pending = task.fiber_context.exit_mask().unwrap();
        task.request_interrupt(pending);
        assert_eq!(task.state, TaskState::Stopped);
    });
}

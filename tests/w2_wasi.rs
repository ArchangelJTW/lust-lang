//! Integration tests for Track 2G (`W-ADAPT`):
//! - WASI runtime adapters (provided and denied filesystem, env, clock capabilities)
//! - Preopen sandboxing, Default-Deny isolation, and path traversal rejection
//! - Deterministic simulation mocks (`MockWasiClock`, `MemFs`, reproducible PRNG)
//! - Explicit unsupported-capability behavior (sockets, processes, threads, signals, FFI)
//! - Error propagation, errno translation, and Lust Exit mapping
//! - File descriptor tracking, lifecycle management, and automatic cleanup on Drop
//! - Single-threaded WASI timeout guard per RFC-WASM-WASI-01 §2.2
//! - VM attachment and capability mask synchronization

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use lust::effect_abi::Exit;
use lust::sim::clock::LustClock;
use lust::vfs::{MemFs, VirtualFileSystem};
use lust::vm::VM;
use lust::wasi::{
    CLOCKID_MONOTONIC, CLOCKID_REALTIME, MockWasiClock, WasiAdapter, WasiContext, WasiError,
    WasiErrno, install_wasi_timeout_guard, monotonic_now_nanos,
};
use lust::wasm::{
    WasmAbi, WasmArchitecture, WasmRuntimeConfig, WasmTarget, detect_current_abi, is_wasm_target,
};

// =============================================================================
// 1. Adapter Dispatch and Basic Lifecycle
// =============================================================================

#[test]
fn test_wasi_adapter_dispatch_basic() {
    let mem_fs = Arc::new(MemFs::new());
    let ctx = WasiContext::builder()
        .with_vfs(mem_fs.clone())
        .preopen("/workspace", "/workspace")
        .env("APP_NAME", "lust_wasi_test")
        .arg("arg0")
        .arg("--flag")
        .build();

    let adapter = WasiAdapter::new(ctx);

    // 1. File write and read dispatch
    adapter
        .path_write(Path::new("/workspace/hello.txt"), b"Hello, WASI!")
        .expect("path_write should succeed");

    let content = adapter
        .path_read_to_string(Path::new("/workspace/hello.txt"))
        .expect("path_read_to_string should succeed");
    assert_eq!(content, "Hello, WASI!");

    // 2. Open handle, read chunks, seek, close
    let fd = adapter
        .path_open(Path::new("/workspace/hello.txt"), true, false)
        .expect("path_open should succeed");
    assert!(fd >= 3);

    let bytes = adapter.fd_read(fd, 5).expect("fd_read should succeed");
    assert_eq!(&bytes, b"Hello");

    let bytes2 = adapter.fd_read(fd, 7).expect("fd_read should succeed");
    assert_eq!(&bytes2, b", WASI!");

    adapter.fd_close(fd).expect("fd_close should succeed");

    // 3. Env dispatch
    assert_eq!(adapter.var("APP_NAME").unwrap(), "lust_wasi_test");
    let (count, size) = adapter.environ_sizes_get().unwrap();
    assert_eq!(count, 1);
    assert!(size > 0);

    // 4. Args dispatch
    let args = adapter.args_get().unwrap();
    assert_eq!(args, vec!["arg0", "--flag"]);

    // 5. Clock dispatch
    let epoch = adapter.clock_time_get(CLOCKID_REALTIME).unwrap();
    let mono = adapter.clock_time_get(CLOCKID_MONOTONIC).unwrap();
    assert!(epoch > 0);
    assert!(mono > 0);
}

// =============================================================================
// 2. Filesystem Capabilities: Provided vs Denied & Preopen Sandboxing
// =============================================================================

#[test]
fn test_wasi_fs_provided_and_preopen_resolution() {
    let mem_fs = Arc::new(MemFs::new());
    let ctx = WasiContext::builder()
        .with_vfs(mem_fs.clone())
        .preopen("/sandbox", "/virtual_sandbox")
        .preopen_read_only("/ro_data", "/virtual_ro")
        .build();

    let adapter = WasiAdapter::new(ctx);

    // Write in read-write preopen
    adapter
        .path_write(Path::new("/sandbox/file.dat"), b"mutable")
        .unwrap();
    assert_eq!(
        adapter.path_read_to_string(Path::new("/sandbox/file.dat")).unwrap(),
        "mutable"
    );

    // Verify directory listing
    let entries = adapter.path_read_dir(Path::new("/sandbox")).unwrap();
    assert_eq!(entries.len(), 1);

    // Populate read-only preopen directly in backing VFS
    mem_fs
        .write(Path::new("/virtual_ro/config.json"), b"{\"readonly\":true}")
        .unwrap();

    // Reading from read-only preopen works
    let ro_data = adapter
        .path_read_to_string(Path::new("/ro_data/config.json"))
        .unwrap();
    assert_eq!(ro_data, "{\"readonly\":true}");

    // Writing to read-only preopen is strictly rejected with Roofs / Acces
    let write_err = adapter
        .path_write(Path::new("/ro_data/config.json"), b"new_data")
        .unwrap_err();
    assert_eq!(write_err.errno, WasiErrno::Roofs);

    // Prestat lookup
    let preopen = adapter.fd_prestat_get(3).unwrap();
    assert!(preopen.guest_path == PathBuf::from("/sandbox") || preopen.guest_path == PathBuf::from("/ro_data"));
}

#[test]
fn test_wasi_fs_denied_capabilities() {
    // A. Explicit allow_fs = false
    let ctx = WasiContext::builder()
        .allow_fs(false)
        .preopen("/workspace", "/workspace")
        .build();
    let adapter = WasiAdapter::new(ctx);

    let err = adapter
        .path_read_to_string(Path::new("/workspace/test.txt"))
        .unwrap_err();
    assert_eq!(err.errno, WasiErrno::Acces);
    assert_eq!(err.capability, Some("allow_fs".to_string()));

    let write_err = adapter
        .path_write(Path::new("/workspace/test.txt"), b"data")
        .unwrap_err();
    assert_eq!(write_err.errno, WasiErrno::Acces);
    assert_eq!(write_err.capability, Some("allow_fs".to_string()));

    let open_err = adapter
        .path_open(Path::new("/workspace/test.txt"), true, false)
        .unwrap_err();
    assert_eq!(open_err.errno, WasiErrno::Acces);
}

#[test]
fn test_wasi_fs_sandbox_containment_and_traversal_rejection() {
    let ctx = WasiContext::builder()
        .preopen("/workspace", "/virtual_ws")
        .build();
    let adapter = WasiAdapter::new(ctx);

    // 1. Path outside preopens is rejected (Default-Deny)
    let outside_err = adapter
        .path_read_to_string(Path::new("/etc/passwd"))
        .unwrap_err();
    assert_eq!(outside_err.errno, WasiErrno::Acces);
    assert!(outside_err.message.contains("Default-Deny sandbox"));

    // 2. Traversal attempt via `..` above preopen root is rejected
    let traversal_err = adapter
        .path_read_to_string(Path::new("/workspace/../../etc/passwd"))
        .unwrap_err();
    assert_eq!(traversal_err.errno, WasiErrno::Acces);

    // 3. Traversal inside open file path is rejected
    let open_trav = adapter
        .path_open(Path::new("/workspace/../secret"), true, false)
        .unwrap_err();
    assert_eq!(open_trav.errno, WasiErrno::Acces);
}

// =============================================================================
// 3. Environment & CLI Capabilities: Provided vs Denied
// =============================================================================

#[test]
fn test_wasi_env_provided_and_denied() {
    // 1. Provided env
    let ctx_ok = WasiContext::builder()
        .allow_env(true)
        .env("KEY1", "VAL1")
        .env("PORT", "8080")
        .args(vec!["app.wasm", "--fast"])
        .build();
    let adapter_ok = WasiAdapter::new(ctx_ok);

    assert_eq!(adapter_ok.var("KEY1").unwrap(), "VAL1");
    assert_eq!(adapter_ok.var("PORT").unwrap(), "8080");
    assert_eq!(
        adapter_ok.var("MISSING").unwrap_err().errno,
        WasiErrno::Noent
    );
    assert_eq!(
        adapter_ok.args_get().unwrap(),
        vec!["app.wasm", "--fast"]
    );

    // 2. Denied env
    let ctx_denied = WasiContext::builder()
        .allow_env(false)
        .env("KEY1", "VAL1")
        .build();
    let adapter_denied = WasiAdapter::new(ctx_denied);

    let err_var = adapter_denied.var("KEY1").unwrap_err();
    assert_eq!(err_var.errno, WasiErrno::Acces);
    assert_eq!(err_var.capability, Some("allow_env".to_string()));

    let err_environ = adapter_denied.environ_get().unwrap_err();
    assert_eq!(err_environ.errno, WasiErrno::Acces);

    let err_args = adapter_denied.args_get().unwrap_err();
    assert_eq!(err_args.errno, WasiErrno::Acces);
}

// =============================================================================
// 4. Clock Capabilities & Deterministic Mocks (Simulation)
// =============================================================================

#[test]
fn test_wasi_clock_provided_and_denied() {
    let ctx_ok = WasiContext::builder().allow_clocks(true).build();
    let adapter_ok = WasiAdapter::new(ctx_ok);

    let realtime = adapter_ok.clock_time_get(CLOCKID_REALTIME).unwrap();
    let monotonic = adapter_ok.clock_time_get(CLOCKID_MONOTONIC).unwrap();
    assert!(realtime > 0);
    assert!(monotonic > 0);
    assert_eq!(adapter_ok.clock_res_get(CLOCKID_REALTIME).unwrap(), 1);

    // Invalid clock id
    let invalid_res = adapter_ok.clock_time_get(999);
    assert_eq!(invalid_res.unwrap_err().errno, WasiErrno::Inval);

    // Denied clock
    let ctx_denied = WasiContext::builder().allow_clocks(false).build();
    let adapter_denied = WasiAdapter::new(ctx_denied);

    let err = adapter_denied.clock_time_get(CLOCKID_REALTIME).unwrap_err();
    assert_eq!(err.errno, WasiErrno::Acces);
    assert_eq!(err.capability, Some("allow_clocks".to_string()));
}

#[test]
fn test_wasi_deterministic_mocks_simulation() {
    // Deterministic simulation testing using MockWasiClock
    let initial_epoch = 1_700_000_000_000_000_000;
    let initial_mono = 10_000_000_000;
    let mock_clock = Arc::new(MockWasiClock::new(initial_epoch, initial_mono));

    let lust_clock = Arc::new(LustClock::new(
        mock_clock.clone(),
        mock_clock.clone(),
        None,
        Arc::new(lust::sim::clock::TimerQueue::new()),
    ));

    let ctx = WasiContext::builder()
        .with_clock(lust_clock)
        .build();
    let adapter = WasiAdapter::new(ctx);

    // Verify initial clock state is exactly as mocked
    assert_eq!(adapter.clock_time_get(CLOCKID_REALTIME).unwrap(), initial_epoch);
    assert_eq!(adapter.clock_time_get(CLOCKID_MONOTONIC).unwrap(), initial_mono);

    // Advance clock deterministically by 500 ms
    mock_clock.advance(Duration::from_millis(500));

    let advance_nanos = 500_000_000;
    assert_eq!(
        adapter.clock_time_get(CLOCKID_REALTIME).unwrap(),
        initial_epoch + advance_nanos
    );
    assert_eq!(
        adapter.clock_time_get(CLOCKID_MONOTONIC).unwrap(),
        initial_mono + advance_nanos
    );

    // Deterministic PRNG verification
    let mut buf1 = [0u8; 16];
    let mut buf2 = [0u8; 16];
    adapter.random_get(&mut buf1).unwrap();
    // Non-zero random entropy generated
    assert_ne!(buf1, [0u8; 16]);

    // Same seed generates reproducible pseudorandom sequence
    mock_clock.set_monotonic_nanos(100_000);
    adapter.random_get(&mut buf1).unwrap();
    mock_clock.set_monotonic_nanos(100_000);
    adapter.random_get(&mut buf2).unwrap();
    assert_eq!(buf1, buf2, "Mock random must be 100% deterministic given identical seed");
}

// =============================================================================
// 5. Explicit Unsupported Capabilities
// =============================================================================

#[test]
fn test_wasi_explicit_unsupported_capabilities() {
    let ctx = WasiContext::default();
    let adapter = WasiAdapter::new(ctx);

    // 1. Sockets & Networking must return ENOSYS (not panic)
    let net_ops = vec![
        adapter.sock_accept(),
        adapter.sock_connect(),
        adapter.sock_send(),
        adapter.sock_recv(),
        adapter.sock_shutdown(),
    ];
    for op in net_ops {
        let err = op.unwrap_err();
        assert_eq!(err.errno, WasiErrno::Nosys);
        assert!(err.message.contains("sockets and networking") || err.message.contains("socket"));
    }

    // 2. Process operations must return ENOSYS
    assert_eq!(adapter.proc_spawn().unwrap_err().errno, WasiErrno::Nosys);
    assert_eq!(adapter.proc_exec().unwrap_err().errno, WasiErrno::Nosys);
    assert_eq!(adapter.proc_raise().unwrap_err().errno, WasiErrno::Nosys);

    // 3. Thread spawning must return ENOSYS
    assert_eq!(adapter.thread_spawn().unwrap_err().errno, WasiErrno::Nosys);

    // 4. Dynamic library loading must return ENOSYS
    assert_eq!(adapter.ffi_load_library().unwrap_err().errno, WasiErrno::Nosys);
}

// =============================================================================
// 6. Error Propagation and Errno Translation
// =============================================================================

#[test]
fn test_wasi_error_propagation_and_exit_mapping() {
    // 1. From std::io::Error
    let io_not_found = std::io::Error::new(std::io::ErrorKind::NotFound, "no file");
    let wasi_not_found = WasiError::from_io_error(&io_not_found);
    assert_eq!(wasi_not_found.errno, WasiErrno::Noent);

    let io_perm = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "forbidden");
    let wasi_perm = WasiError::from_io_error(&io_perm);
    assert_eq!(wasi_perm.errno, WasiErrno::Acces);

    // 2. To std::io::Error
    let std_io = wasi_perm.to_io_error();
    assert_eq!(std_io.kind(), std::io::ErrorKind::PermissionDenied);

    // 3. To Lust Exit
    let exit: Exit<(), String> = wasi_perm.to_exit();
    assert!(exit.is_failure());
    if let Exit::Failure(cause) = exit {
        let formatted = format!("{:?}", cause);
        assert!(formatted.contains("WASI error EACCES"));
    }
}

// =============================================================================
// 7. Descriptor Lifecycle and Automatic Cleanup on Drop
// =============================================================================

#[test]
fn test_wasi_descriptor_lifecycle_and_cleanup() {
    let mem_fs = Arc::new(MemFs::new());
    let ctx = WasiContext::builder()
        .with_vfs(mem_fs.clone())
        .preopen("/workspace", "/workspace")
        .build();
    let adapter = WasiAdapter::new(ctx);

    // Write test files
    adapter.path_write(Path::new("/workspace/f1.txt"), b"1").unwrap();
    adapter.path_write(Path::new("/workspace/f2.txt"), b"2").unwrap();

    // Open multiple descriptors
    let fd1 = adapter.path_open(Path::new("/workspace/f1.txt"), true, false).unwrap();
    let fd2 = adapter.path_open(Path::new("/workspace/f2.txt"), true, false).unwrap();

    assert_eq!(adapter.context().fd_table.active_fd_count(), 2);

    // Close one explicitly
    adapter.fd_close(fd1).unwrap();
    assert_eq!(adapter.context().fd_table.active_fd_count(), 1);

    // Second close returns Badf
    assert_eq!(adapter.fd_close(fd1).unwrap_err().errno, WasiErrno::Badf);

    // Reading from closed descriptor returns Badf
    assert_eq!(adapter.fd_read(fd1, 10).unwrap_err().errno, WasiErrno::Badf);

    // Still have fd2 open
    assert_eq!(adapter.fd_read(fd2, 1).unwrap(), b"2");

    // Close remaining or trigger full cleanup
    adapter.context().fd_table.cleanup_all();
    assert_eq!(adapter.context().fd_table.active_fd_count(), 0);
}

// =============================================================================
// 8. Single-Threaded WASI Timeout Guard (RFC-WASM-WASI-01 §2.2)
// =============================================================================

#[test]
fn test_wasi_single_threaded_timeout_guard() {
    let now = monotonic_now_nanos();
    assert!(now > 0);

    // Future deadline: not timed out
    let future_deadline = now.saturating_add(5_000_000_000); // +5 seconds
    let mut guard = install_wasi_timeout_guard(future_deadline);
    assert!(!guard.has_timed_out());
    guard.cancel();

    // Past deadline: immediately timed out
    let past_deadline = now.saturating_sub(1_000);
    let mut expired_guard = install_wasi_timeout_guard(past_deadline);
    assert!(expired_guard.has_timed_out());
    expired_guard.cancel();
}

// =============================================================================
// 9. VM Attachment and Capability Synchronization
// =============================================================================

#[test]
fn test_wasi_vm_attachment() {
    let mem_fs = Arc::new(MemFs::new());
    mem_fs.write(Path::new("/workspace/script.lust"), b"print(\"wasi\")").unwrap();

    let ctx = WasiContext::builder()
        .with_vfs(mem_fs.clone())
        .preopen("/workspace", "/workspace")
        .env("ENV_TARGET", "wasi")
        .allow_fs(true)
        .allow_env(true)
        .build();

    let adapter = WasiAdapter::new(ctx);

    let mut vm = VM::new();
    adapter.attach_to_vm(&mut vm);

    // Check VM configuration reflects WASI capabilities
    let caps = vm.config().capability_mask();
    assert!(caps.allow_fs);
    assert!(caps.allow_os);
    assert!(!caps.allow_net);
    assert!(!caps.allow_ffi);

    // Check VM VirtualEnv received the WASI environment
    assert_eq!(vm.env.get("ENV_TARGET"), Some("wasi".to_string()));

    // Check VM VFS can read the file
    assert_eq!(
        vm.vfs.read_to_string(Path::new("/workspace/script.lust")).unwrap(),
        "print(\"wasi\")"
    );
}

// =============================================================================
// 10. WebAssembly Architectural Targets & ABIs
// =============================================================================

#[test]
fn test_wasm_architectural_target_contracts() {
    assert_eq!(WasmArchitecture::Wasm32.pointer_width(), 32);
    assert_eq!(WasmArchitecture::Wasm64.pointer_width(), 64);

    assert_eq!(WasmAbi::BrowserJs.to_string(), "browser-js");
    assert_eq!(WasmAbi::WasiSnapshotPreview1.to_string(), "wasi-snapshot-preview1");
    assert_eq!(WasmAbi::WasiPreview2.to_string(), "wasi-preview2");
    assert_eq!(WasmAbi::BareWasm.to_string(), "bare-wasm");

    let target = WasmTarget::default();
    assert_eq!(target.arch, WasmArchitecture::Wasm32);
    assert_eq!(target.memory_page_size, 65536);

    let config = WasmRuntimeConfig::default();
    assert_eq!(config.initial_memory_pages, 16);
    assert_eq!(config.max_memory_pages, Some(1024));
    assert_eq!(config.fuel_limit, Some(1_000_000));
    assert!(config.deterministic_clock);

    // Host detection
    let detected_abi = detect_current_abi();
    let is_wasm = is_wasm_target();
    #[cfg(target_arch = "wasm32")]
    assert!(is_wasm);
    #[cfg(not(target_arch = "wasm32"))]
    {
        assert!(!is_wasm);
        assert_eq!(detected_abi, WasmAbi::BareWasm);
    }
}

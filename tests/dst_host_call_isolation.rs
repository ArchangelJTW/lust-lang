//! Gate 4 Acceptance Test: BlockingHostCall Contract & Host Isolation (D4, Section 0.3, Gate 4)
//!
//! Validates:
//! 1. Contract enforcement:
//!    - Unmodeled synchronous blocking host calls inside simulation are strictly prohibited.
//!    - Detected violations trigger immediate fail-stop with `LustError::BlockingHostCallViolation`.
//! 2. Fail-stop panic semantics:
//!    - Panicking variant `fail_stop_blocking_host_call` aborts execution with explicit violation notice.
//! 3. Partitioning of host calls:
//!    - Modelled deterministic mocks (`MemFs`, `VirtualConsole`) provide immediate synthetic responses.
//!    - Simulation state isolation ensures no host OS filesystem or console side-effects leak.

use lust::error::LustError;
use lust::sim::host_call::{
    BlockingHostCallContract, assert_non_blocking_host_call, fail_stop_blocking_host_call,
};
use lust::sim::{SimModeGuard, is_simulated_mode, set_simulated_mode};
use lust::vfs::{MemFs, VirtualConsole, VirtualFileSystem};
use std::path::Path;

#[test]
fn test_unmodeled_blocking_host_call_triggers_violation_under_sim() {
    // Outside simulation: non-blocking check succeeds
    set_simulated_mode(false);
    assert!(!is_simulated_mode());
    assert!(assert_non_blocking_host_call("socket_recv").is_ok());

    // Enter simulation mode
    {
        let _guard = SimModeGuard::enter();
        assert!(is_simulated_mode());

        // Prohibited blocking host call must fail-stop with BlockingHostCallViolation
        let err = assert_non_blocking_host_call("socket_recv").unwrap_err();
        match err {
            LustError::BlockingHostCallViolation(msg) => {
                assert!(
                    msg.contains("socket_recv"),
                    "Error message must mention the offending call: {}",
                    msg
                );
                assert!(
                    msg.contains("prohibited under simulation"),
                    "Error message must clarify simulation prohibition: {}",
                    msg
                );
            }
            other => panic!("Expected BlockingHostCallViolation, got: {:?}", other),
        }

        // Another blocking syscall canary (e.g. synchronous OS thread sleep)
        let err2 = assert_non_blocking_host_call("std::thread::sleep").unwrap_err();
        assert!(matches!(err2, LustError::BlockingHostCallViolation(_)));
    }

    // After guard drops, simulation mode is restored to previous
    assert!(!is_simulated_mode());
    assert!(assert_non_blocking_host_call("socket_recv").is_ok());
}

#[test]
fn test_fail_stop_panic_on_blocking_host_call() {
    let _guard = SimModeGuard::enter();

    let result = std::panic::catch_unwind(|| {
        fail_stop_blocking_host_call("blocking_db_query_sync");
    });

    assert!(result.is_err(), "Must panic on fail-stop violation");
    let err = result.unwrap_err();
    let panic_msg = if let Some(s) = err.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = err.downcast_ref::<String>() {
        s.clone()
    } else {
        String::new()
    };

    assert!(
        panic_msg.contains("BlockingHostCallViolation"),
        "Panic message must declare BlockingHostCallViolation, got: {}",
        panic_msg
    );
    assert!(
        panic_msg.contains("blocking_db_query_sync"),
        "Panic message must name the prohibited call, got: {}",
        panic_msg
    );
}

#[test]
fn test_blocking_host_call_contract_execution() {
    let _guard = SimModeGuard::enter();

    // 1. Unmodeled call -> rejected with error
    let unmodeled_res = BlockingHostCallContract::execute("raw_posix_read", false, || Ok(1234));
    assert!(matches!(
        unmodeled_res,
        Err(LustError::BlockingHostCallViolation(_))
    ));

    // 2. Modelled call -> executes synthetic action cleanly
    let modeled_res =
        BlockingHostCallContract::execute("synthetic_entropy_generator", true, || {
            Ok("synthetic_entropy_42")
        });
    assert_eq!(modeled_res.unwrap(), "synthetic_entropy_42");

    // 3. execute_or_panic catches unmodeled call
    let panic_res = std::panic::catch_unwind(|| {
        BlockingHostCallContract::execute_or_panic("unmodeled_ffi_c_call", false, || 999)
    });
    assert!(panic_res.is_err());
}

#[test]
fn test_deterministic_synthetic_responses_via_memfs_and_virtual_console() {
    let _guard = SimModeGuard::enter();

    // Synthetic MemFs responder
    let memfs = MemFs::new();
    let config_path = Path::new("/workspace/sim_config.json");
    let initial_data = br#"{"cluster_nodes": 5, "seed": 42}"#;

    // Verify synthetic write and read
    let write_res = BlockingHostCallContract::execute("vfs_write", true, || {
        memfs.write(config_path, initial_data).map_err(|e| LustError::RuntimeError { message: e.to_string() })
    });
    assert!(write_res.is_ok());

    let read_res = BlockingHostCallContract::execute("vfs_read", true, || {
        memfs.read_to_string(config_path).map_err(|e| LustError::RuntimeError { message: e.to_string() })
    });
    assert_eq!(read_res.unwrap(), r#"{"cluster_nodes": 5, "seed": 42}"#);

    // Synthetic VirtualConsole responder
    let console = VirtualConsole::captured();
    let console_res = BlockingHostCallContract::execute("console_write", true, || {
        console.write_stdout(b"Simulated node online: node-0\n");
        console.write_stderr(b"Simulated warning: transient packet loss\n");
        Ok(())
    });
    assert!(console_res.is_ok());

    assert_eq!(console.stdout_string(), "Simulated node online: node-0\n");
    assert_eq!(console.stderr_string(), "Simulated warning: transient packet loss\n");
}

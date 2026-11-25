#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! DST Gate 3: VirtualEnv and VirtualConsole Isolation Test Suite.
//! Verifies complete state isolation of environment variables and I/O streams
//! across multiple VM instances running on the same thread.

use lust::bytecode::ValueKey;
use lust::vm::{CurrentVmGuard, VM};
use lust::Value;

#[test]
fn test_virtual_env_isolation_between_vms() {
    let mut config = lust::config::LustConfig::default();
    config.allow_std_module("os");

    let mut vm1 = VM::with_config(&config);
    vm1.vfs = std::sync::Arc::new(lust::vfs::MemFs::new());
    vm1.env = lust::vfs::VirtualEnv::isolated();
    vm1.console = lust::vfs::VirtualConsole::captured();

    let mut vm2 = VM::with_config(&config);
    vm2.vfs = std::sync::Arc::new(lust::vfs::MemFs::new());
    vm2.env = lust::vfs::VirtualEnv::isolated();
    vm2.console = lust::vfs::VirtualConsole::captured();

    // 1. Direct VirtualEnv table isolation
    vm1.env.set("PORT", "8080");
    vm1.env.set("SECRET_KEY", "key_alpha");

    vm2.env.set("PORT", "9090");
    vm2.env.set("SERVICE_NAME", "worker_service");

    assert_eq!(vm1.env.get("PORT").as_deref(), Some("8080"));
    assert_eq!(vm2.env.get("PORT").as_deref(), Some("9090"));

    assert_eq!(vm1.env.get("SECRET_KEY").as_deref(), Some("key_alpha"));
    assert_eq!(vm2.env.get("SECRET_KEY"), None);

    assert_eq!(vm1.env.get("SERVICE_NAME"), None);
    assert_eq!(vm2.env.get("SERVICE_NAME").as_deref(), Some("worker_service"));

    // 2. Guest stdlib os.get_env and os.set_env isolation
    {
        let _guard1 = CurrentVmGuard::from_vm(&mut vm1);
        let os_val = vm1.get_global("os").unwrap();
        let get_env_fn = os_val.map_get(&ValueKey::from("get_env")).unwrap();
        let set_env_fn = os_val.map_get(&ValueKey::from("set_env")).unwrap();

        // Check get_env on vm1
        let res1 = vm1
            .call_value(&get_env_fn, vec![Value::string("PORT")])
            .unwrap();
        assert_eq!(res1, Value::ok(Value::some(Value::string("8080"))));

        // Mutate via set_env on vm1
        vm1.call_value(
            &set_env_fn,
            vec![Value::string("DYNAMIC"), Value::string("val1")],
        )
        .unwrap();
        assert_eq!(vm1.env.get("DYNAMIC").as_deref(), Some("val1"));
    }

    {
        let _guard2 = CurrentVmGuard::from_vm(&mut vm2);
        let os_val = vm2.get_global("os").unwrap();
        let get_env_fn = os_val.map_get(&ValueKey::from("get_env")).unwrap();

        // Verify vm2 sees its own PORT and does NOT see DYNAMIC from vm1
        let res2 = vm2
            .call_value(&get_env_fn, vec![Value::string("PORT")])
            .unwrap();
        assert_eq!(res2, Value::ok(Value::some(Value::string("9090"))));

        let res_dyn = vm2
            .call_value(&get_env_fn, vec![Value::string("DYNAMIC")])
            .unwrap();
        assert_eq!(res_dyn, Value::ok(Value::none()));
    }
}

#[test]
fn test_virtual_console_buffering_and_isolation() {
    let mut vm1 = VM::with_mem_fs();
    let mut vm2 = VM::with_mem_fs();

    // 1. Stdout print isolation
    {
        let _guard1 = CurrentVmGuard::from_vm(&mut vm1);
        let println_fn = vm1.get_global("println").unwrap();
        vm1.call_value(&println_fn, vec![Value::string("output_from_vm1")])
            .unwrap();
    }

    {
        let _guard2 = CurrentVmGuard::from_vm(&mut vm2);
        let println_fn = vm2.get_global("println").unwrap();
        vm2.call_value(&println_fn, vec![Value::string("output_from_vm2")])
            .unwrap();
    }

    let out1 = vm1.console.stdout_string();
    let out2 = vm2.console.stdout_string();

    assert!(out1.contains("output_from_vm1"));
    assert!(!out1.contains("output_from_vm2"));

    assert!(out2.contains("output_from_vm2"));
    assert!(!out2.contains("output_from_vm1"));

    // 2. Stdin virtual feeder isolation
    vm1.console.set_stdin("command_for_vm1\nline2_vm1\n");
    vm2.console.set_stdin("command_for_vm2\n");

    let line1 = vm1.console.read_line_stdin().unwrap();
    assert_eq!(line1, "command_for_vm1\n");

    let line2 = vm2.console.read_line_stdin().unwrap();
    assert_eq!(line2, "command_for_vm2\n");

    let line1_next = vm1.console.read_line_stdin().unwrap();
    assert_eq!(line1_next, "line2_vm1\n");

    // 3. Stderr isolation
    vm1.console.write_stderr(b"error_in_vm1");
    vm2.console.write_stderr(b"error_in_vm2");

    assert_eq!(vm1.console.stderr_string(), "error_in_vm1");
    assert_eq!(vm2.console.stderr_string(), "error_in_vm2");
}

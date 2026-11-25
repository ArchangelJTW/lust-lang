#![cfg(feature = "std")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use lust::bytecode::NativeCallResult;
use lust::config::{CapabilityMask, LustConfig, PreludeMode};
use lust::effect_abi::{Cause, Exit, InterruptReason};
use lust::stdlib::effects::fs::{FileSystemService, IoError, active_open_files_count};
use lust::stdlib::effects::env::{EnvError, EnvService};
use lust::stdlib::effects::random::RandomService;
use lust::vfs::{MemFs, VirtualFileSystem};
use lust::vm::{
    NativeEffectSpec, NativeExport, NativeExportParam, NativeGasPolicy,
    NativeMemoryPolicy, NativePurity, TrustedHostResourceLedger, VM,
};
use lust::{EmbeddedProgram, Value};

// ---------------------------------------------------------------------------
// 1. X-NATMETA Metadata & Builder Tests
// ---------------------------------------------------------------------------

#[test]
fn test_x_natmeta_effect_metadata_builder() {
    let spec = NativeEffectSpec::effectful(
        Some("Error"),
        vec!["io.read".to_string(), "io.write".to_string()],
    )
    .with_capability("allow_fs");

    let export = NativeExport::new(
        "ext_compute",
        vec![NativeExportParam::new("param1", "int")],
        "int",
    )
    .with_effects(spec)
    .with_gas_policy(NativeGasPolicy::Fixed(50))
    .with_memory_policy(NativeMemoryPolicy::MaxAllocBytes(1024))
    .with_doc("Test native export");

    assert_eq!(export.name(), "ext_compute");
    let effects = export.effects().expect("effects should be set");
    assert_eq!(effects.purity, NativePurity::Effectful);
    assert_eq!(effects.requirements, vec!["io.read", "io.write"]);
    assert_eq!(effects.required_capability.as_deref(), Some("allow_fs"));
    assert_eq!(export.gas_policy(), Some(&NativeGasPolicy::Fixed(50)));
    assert_eq!(export.memory_policy(), Some(&NativeMemoryPolicy::MaxAllocBytes(1024)));
    assert_eq!(export.doc(), Some("Test native export"));
    assert_eq!(export.params().len(), 1);
    assert_eq!(export.params()[0].name(), "param1");
    assert_eq!(export.params()[0].ty(), "int");
}

// ---------------------------------------------------------------------------
// 2. X-NATMETA Rejection of Undeclared Effects & Pure Violations
// ---------------------------------------------------------------------------

#[test]
fn test_x_natmeta_rejection_of_undeclared_effects() {
    let make_sneaky_effect = || {
        move |_args: &[Value]| {
            let entries = vec![
                (lust::bytecode::ValueKey::from("__effect".to_string()), Value::Bool(true)),
                (lust::bytecode::ValueKey::from("__kind".to_string()), Value::string("fail")),
                (lust::bytecode::ValueKey::from("__error".to_string()), Value::string("unexpected effect error")),
            ];
            Ok(NativeCallResult::Return(Value::map(entries.into_iter().collect())))
        }
    };

    let mut vm = VM::new();

    // Export 1: Declared Pure, but returns an effect -> fails closed
    let pure_spec = NativeEffectSpec::pure();
    let pure_export = NativeExport::new("pure_violator", vec![], "Map")
        .with_effects(pure_spec);
    vm.register_exported_native(pure_export, make_sneaky_effect());

    let res = vm.call("pure_violator", vec![]);
    assert!(res.is_err());
    let err = res.err().unwrap().to_string();
    assert!(err.contains("UndeclaredEffect"), "Expected pure/undeclared effect violation, got: {}", err);

    // Export 2: No declared effect signature, but returns an effect -> fails closed
    let mut vm2 = VM::new();
    let no_spec_export = NativeExport::new("no_spec_violator", vec![], "Map");
    vm2.register_exported_native(no_spec_export, make_sneaky_effect());

    let res2 = vm2.call("no_spec_violator", vec![]);
    assert!(res2.is_err());
    let err2 = res2.err().unwrap().to_string();
    assert!(err2.contains("UndeclaredEffect"), "Expected undeclared effect violation, got: {}", err2);

    // Export 3: Correctly declared effectful with error_type -> succeeds
    let mut vm3 = VM::new();
    let valid_spec = NativeEffectSpec::effectful(Some("Error"), vec![]);
    let valid_export = NativeExport::new("valid_effect", vec![], "Map")
        .with_effects(valid_spec);
    vm3.register_exported_native(valid_export, make_sneaky_effect());

    let res3 = vm3.call("valid_effect", vec![]);
    assert!(res3.is_ok());
}

#[test]
fn test_x_natmeta_compile_time_pure_prelude_rejection() {
    let mut config = LustConfig::default();
    config.set_prelude_mode(PreludeMode::Pure);

    let impure_spec = NativeEffectSpec::effectful(Some("Error"), vec!["io.read".to_string()]);
    let impure_export = NativeExport::new("impure_fn", vec![], "Map")
        .with_effects(impure_spec);

    let build_res = EmbeddedProgram::builder()
        .with_config(config)
        .module("main", "fn main() -> int { 0 }")
        .entry_module("main")
        .declare_exported_native(impure_export)
        .compile();

    assert!(build_res.is_err());
    let err = build_res.err().unwrap().to_string();
    assert!(err.contains("PurityViolation"), "Expected compile-time purity error, got: {}", err);
}

// ---------------------------------------------------------------------------
// 3. X-NATMETA Gas & Memory Policies
// ---------------------------------------------------------------------------

#[test]
fn test_x_natmeta_gas_policy_enforcement() {
    let make_add = || {
        move |args: &[Value]| {
            let a = args.get(0).and_then(|v| v.as_int()).unwrap_or(0);
            let b = args.get(1).and_then(|v| v.as_int()).unwrap_or(0);
            Ok(NativeCallResult::Return(Value::Int(a + b)))
        }
    };

    // With budget 50 < 100 -> Out of gas
    let mut vm = VM::new();
    let export = NativeExport::new(
        "metered_add",
        vec![
            NativeExportParam::new("a", "int"),
            NativeExportParam::new("b", "int"),
        ],
        "int",
    )
    .with_gas_policy(NativeGasPolicy::Fixed(100));
    vm.register_exported_native(export, make_add());
    vm.set_gas_budget(50);

    let res = vm.call("metered_add", vec![Value::Int(10), Value::Int(20)]);
    assert!(res.is_err());
    let err = res.err().unwrap().to_string();
    assert!(err.contains("Out of gas"), "Expected out of gas error, got: {}", err);

    // With budget 200 >= 100 -> succeeds, exactly 100 gas charged
    let mut vm2 = VM::new();
    let export2 = NativeExport::new(
        "metered_add",
        vec![
            NativeExportParam::new("a", "int"),
            NativeExportParam::new("b", "int"),
        ],
        "int",
    )
    .with_gas_policy(NativeGasPolicy::Fixed(100));
    vm2.register_exported_native(export2, make_add());
    vm2.set_gas_budget(200);

    let res2 = vm2.call("metered_add", vec![Value::Int(10), Value::Int(20)]);
    assert!(res2.is_ok());
    assert_eq!(res2.unwrap(), Value::Int(30));
    assert_eq!(vm2.gas_used(), 100);
}

#[test]
fn test_x_natmeta_memory_policy_enforcement() {
    let make_alloc = || {
        move |_args: &[Value]| Ok(NativeCallResult::Return(Value::string("allocated output")))
    };

    let mut vm = VM::new();
    let export = NativeExport::new("bounded_alloc", vec![], "string")
        .with_memory_policy(NativeMemoryPolicy::MaxAllocBytes(500));
    vm.register_exported_native(export, make_alloc());

    // Limit is 200 bytes, but function requires 500 bytes -> fails closed
    vm.set_memory_budget_bytes(200).unwrap();
    let res = vm.call("bounded_alloc", vec![]);
    assert!(res.is_err());
    let err = res.err().unwrap().to_string();
    assert!(err.contains("Out of memory"), "Expected OOM, got: {}", err);

    // When memory budget is sufficient (1000 bytes) -> succeeds
    let mut vm2 = VM::new();
    let export2 = NativeExport::new("bounded_alloc", vec![], "string")
        .with_memory_policy(NativeMemoryPolicy::MaxAllocBytes(500));
    vm2.register_exported_native(export2, make_alloc());
    vm2.set_memory_budget_bytes(1000).unwrap();

    let res2 = vm2.call("bounded_alloc", vec![]);
    assert!(res2.is_ok());
    assert!(vm2.memory_used_bytes() >= 500);
}

#[test]
fn test_x_natmeta_capability_enforcement() {
    let mut config = LustConfig::default();
    config.set_capability_mask(CapabilityMask {
        allow_net: false, // Net capability disabled!
        ..CapabilityMask::default()
    });

    let spec = NativeEffectSpec::effectful(Some("Error"), vec![])
        .with_capability("net");

    let mut vm = VM::with_config(&config);
    let export = NativeExport::new("secure_net", vec![], "string")
        .with_effects(spec);
    vm.register_exported_native(export, move |_args| {
        Ok(NativeCallResult::Return(Value::string("net response")))
    });

    let res = vm.call("secure_net", vec![]);
    assert!(res.is_err());
    let err = res.err().unwrap().to_string();
    assert!(err.contains("CapabilityDenied"), "Expected capability denied, got: {}", err);
}

// ---------------------------------------------------------------------------
// 4. TrustedHostResourceLedger Limits & Scoped Cleanup
// ---------------------------------------------------------------------------

#[test]
fn test_x_natmeta_ledger_limits_and_scoped_cleanup() {
    static CLEANED_COUNT: AtomicUsize = AtomicUsize::new(0);

    fn test_cleanup(_resource_id: usize) {
        CLEANED_COUNT.fetch_add(1, Ordering::SeqCst);
    }

    let mut ledger = TrustedHostResourceLedger::new();

    // Capacity is 256: acquire 256 resources
    for id in 0..256 {
        let slot = ledger.acquire(id, test_cleanup);
        assert!(slot.is_ok(), "Slot {} should acquire successfully", id);
    }

    // 257th acquire must fail closed with TooManyOpenResources
    let overflow = ledger.acquire(999, test_cleanup);
    assert!(overflow.is_err(), "257th acquire must fail");

    // Clean up all: all 256 must be dropped with 0 gas/alloc
    CLEANED_COUNT.store(0, Ordering::SeqCst);
    ledger.cleanup_all();
    assert_eq!(CLEANED_COUNT.load(Ordering::SeqCst), 256);

    // Test scoped cleanup: acquire 5 resources in scope 10, 5 in scope 20
    CLEANED_COUNT.store(0, Ordering::SeqCst);
    for id in 0..5 {
        ledger.acquire_scoped(id, test_cleanup, 10).unwrap();
    }
    for id in 5..10 {
        ledger.acquire_scoped(id, test_cleanup, 20).unwrap();
    }

    // Clean up scope 10: only 5 resources dropped
    ledger.cleanup_scope(10);
    assert_eq!(CLEANED_COUNT.load(Ordering::SeqCst), 5);

    // Clean up scope 20: the remaining 5 dropped
    ledger.cleanup_scope(20);
    assert_eq!(CLEANED_COUNT.load(Ordering::SeqCst), 10);
}

// ---------------------------------------------------------------------------
// 5. FileSystemService & MemFs Isolation
// ---------------------------------------------------------------------------

#[test]
fn test_fs_service_memfs_isolation() {
    let mut vm = VM::new();
    let memfs = Arc::new(MemFs::new());
    vm.vfs = memfs.clone() as Arc<dyn VirtualFileSystem>;

    // 1. Write file
    let write_res = FileSystemService::write_string(&mut vm, "/virtual/test.txt", "Virtual FS content");
    assert!(matches!(write_res, Exit::Success(())));

    // Verify it exists in MemFs
    let exists_res = FileSystemService::exists(&mut vm, "/virtual/test.txt");
    assert_eq!(exists_res, Exit::Success(true));

    // Verify host OS disk is completely untouched
    assert!(!std::path::Path::new("/virtual/test.txt").exists(), "Host disk must not contain virtual file!");

    // 2. Read file
    let read_res = FileSystemService::read_to_string(&mut vm, "/virtual/test.txt");
    assert_eq!(read_res, Exit::Success("Virtual FS content".to_string()));

    // 3. Create dir and read dir
    let mkdir_res = FileSystemService::create_dir_all(&mut vm, "/virtual/sub/dir");
    assert!(matches!(mkdir_res, Exit::Success(())));

    let _ = FileSystemService::write_string(&mut vm, "/virtual/sub/dir/child.txt", "child");
    let readdir_res = FileSystemService::read_dir(&mut vm, "/virtual/sub/dir");
    match readdir_res {
        Exit::Success(entries) => {
            assert!(entries.iter().any(|e| e.contains("child.txt")));
        }
        _ => panic!("read_dir failed"),
    }

    // 4. Remove file
    let rm_res = FileSystemService::remove_file(&mut vm, "/virtual/test.txt");
    assert!(matches!(rm_res, Exit::Success(())));
    let exists_after = FileSystemService::exists(&mut vm, "/virtual/test.txt");
    assert_eq!(exists_after, Exit::Success(false));

    // 5. Capability check: allow_fs = false fails closed
    let mut no_fs_config = LustConfig::default();
    no_fs_config.set_capability_mask(CapabilityMask {
        allow_fs: false,
        ..CapabilityMask::default()
    });
    let mut restricted_vm = VM::with_config(&no_fs_config);
    restricted_vm.vfs = memfs as Arc<dyn VirtualFileSystem>;

    let denied_read = FileSystemService::read_to_string(&mut restricted_vm, "/virtual/test.txt");
    match denied_read {
        Exit::Failure(Cause::Fail(IoError::CapabilityDenied(_))) => {}
        other => panic!("Expected CapabilityDenied, got {:?}", other),
    }
}

#[test]
fn test_fs_service_scoped_handle_cleanup_on_drop() {
    let initial_open = active_open_files_count();

    {
        let mut vm = VM::new();
        let memfs = Arc::new(MemFs::new());
        vm.vfs = memfs as Arc<dyn VirtualFileSystem>;

        // Write a test file first
        let _ = FileSystemService::write_string(&mut vm, "/test_handle.txt", "handle content");

        // Open scoped file handle
        let handle_exit = FileSystemService::open_scoped(&mut vm, "/test_handle.txt", "r");
        assert!(matches!(handle_exit, Exit::Success(_)));

        // Active open files count must increase
        let current_open = active_open_files_count();
        assert!(current_open > initial_open, "Open files count must increase while handle is open");

        // VM drops here without explicit close_handle call!
    }

    // After VM drop, TrustedHostResourceLedger::drop automatically executed cleanup callback
    let post_drop_open = active_open_files_count();
    assert_eq!(post_drop_open, initial_open, "All open handles must be cleaned up when VM drops");
}

// ---------------------------------------------------------------------------
// 6. EnvService Isolation & Clock
// ---------------------------------------------------------------------------

#[test]
fn test_env_service_isolation_and_clock() {
    let mut vm = VM::new();

    // 1. In-memory virtual env isolation
    let set_res = EnvService::set_var(&mut vm, "LUST_VIRTUAL_VAR", "secret_value_42");
    assert!(matches!(set_res, Exit::Success(())));

    let get_res = EnvService::var(&mut vm, "LUST_VIRTUAL_VAR");
    assert_eq!(get_res, Exit::Success("secret_value_42".to_string()));

    // Host environment must be completely untouched
    assert!(std::env::var("LUST_VIRTUAL_VAR").is_err(), "Host env must not contain virtual var");

    // 2. Simulated clock
    let now_res = EnvService::now_millis(&mut vm);
    match now_res {
        Exit::Success(ms) => assert!(ms > 0),
        _ => panic!("now_millis failed"),
    }

    // 3. Capability enforcement
    let mut no_os_config = LustConfig::default();
    no_os_config.set_capability_mask(CapabilityMask {
        allow_os: false,
        ..CapabilityMask::default()
    });
    let mut restricted_vm = VM::with_config(&no_os_config);

    let denied = EnvService::var(&mut restricted_vm, "PATH");
    match denied {
        Exit::Failure(Cause::Fail(EnvError::CapabilityDenied(_))) => {}
        other => panic!("Expected CapabilityDenied, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 7. RandomService Determinism & Budget Constraints
// ---------------------------------------------------------------------------

#[test]
fn test_random_service_seed_determinism_and_budgets() {
    let mut vm1 = VM::with_seed(123456);
    let mut vm2 = VM::with_seed(123456);
    let mut vm3 = VM::with_seed(654321);

    // Two VMs with identical seeds produce bit-identical sequences
    for _ in 0..10 {
        let i1 = RandomService::next_int(&mut vm1, 10, 500);
        let i2 = RandomService::next_int(&mut vm2, 10, 500);
        assert_eq!(i1, i2, "Identical seeds must yield identical next_int");

        let f1 = RandomService::next_float(&mut vm1);
        let f2 = RandomService::next_float(&mut vm2);
        assert_eq!(f1, f2, "Identical seeds must yield identical next_float");

        let b1 = RandomService::next_bool(&mut vm1);
        let b2 = RandomService::next_bool(&mut vm2);
        assert_eq!(b1, b2, "Identical seeds must yield identical next_bool");

        let bytes1 = RandomService::fill_bytes(&mut vm1, 16);
        let bytes2 = RandomService::fill_bytes(&mut vm2, 16);
        assert_eq!(bytes1, bytes2, "Identical seeds must yield identical fill_bytes");
    }

    // Different seed produces different output
    let diff_i = RandomService::next_int(&mut vm3, 10, 500);
    let same_i = RandomService::next_int(&mut vm1, 10, 500);
    let mut differed = diff_i != same_i;
    for _ in 0..10 {
        if RandomService::next_float(&mut vm3) != RandomService::next_float(&mut vm1) {
            differed = true;
            break;
        }
    }
    assert!(differed, "Different seed should produce different random sequence");

    // Gas budget exhaustion fails closed with OutOfGas
    let mut gas_vm = VM::new();
    gas_vm.set_gas_budget(2); // next_int requires 5 gas
    let gas_res = RandomService::next_int(&mut gas_vm, 1, 100);
    assert_eq!(gas_res, Exit::Failure(Cause::Interrupted(InterruptReason::OutOfGas)));

    // Memory budget exhaustion fails closed with OutOfMemory
    let mut mem_vm = VM::new();
    mem_vm.set_memory_budget_bytes(10).unwrap();
    let mem_res = RandomService::fill_bytes(&mut mem_vm, 1000);
    assert_eq!(mem_res, Exit::Failure(Cause::Interrupted(InterruptReason::OutOfMemory)));
}

// ---------------------------------------------------------------------------
// 8. Script Stdlib Effects Integration
// ---------------------------------------------------------------------------

#[test]
fn test_script_stdlib_effects_integration() {
    // 1. Script with random effects
    let source = r#"
        use std.random

        function test_rand(): Map
            return random.int_effect(10, 20)
        end
    "#;

    let mut prog = EmbeddedProgram::builder()
        .module("main", source)
        .entry_module("main")
        .compile()
        .expect("Program compilation should succeed");

    let val = prog.vm_mut().call("main.test_rand", vec![]).expect("call should succeed");
    if let Value::Map(m) = val {
        let borrowed = m.borrow();
        assert_eq!(borrowed.get(&lust::bytecode::ValueKey::from("__effect".to_string())), Some(&Value::Bool(true)));
        assert_eq!(borrowed.get(&lust::bytecode::ValueKey::from("__kind".to_string())), Some(&Value::string("succeed")));
        let inner = borrowed.get(&lust::bytecode::ValueKey::from("__value".to_string())).cloned().unwrap();
        if let Value::Int(n) = inner {
            assert!(n >= 10 && n < 20, "Expected random int between 10 and 20, got {}", n);
        } else {
            panic!("Expected inner integer, got {:?}", inner);
        }
    } else {
        panic!("Expected Map effect return value, got {:?}", val);
    }

    // 2. Script with io effects (MemFs)
    let io_source = r#"
        use std.io

        function test_io(): Map
            local w = io.write_file_effect("/script_file.txt", "written from script")
            return io.read_file_effect("/script_file.txt")
        end
    "#;

    let mut io_prog = EmbeddedProgram::builder()
        .module("main", io_source)
        .entry_module("main")
        .compile()
        .expect("IO program compilation should succeed");

    io_prog.vm_mut().vfs = Arc::new(MemFs::new()) as Arc<dyn VirtualFileSystem>;
    let io_val = io_prog.vm_mut().call("main.test_io", vec![]).expect("call should succeed");
    if let Value::Map(m) = io_val {
        let borrowed = m.borrow();
        assert_eq!(borrowed.get(&lust::bytecode::ValueKey::from("__effect".to_string())), Some(&Value::Bool(true)));
        assert_eq!(borrowed.get(&lust::bytecode::ValueKey::from("__kind".to_string())), Some(&Value::string("succeed")));
        let inner = borrowed.get(&lust::bytecode::ValueKey::from("__value".to_string())).cloned().unwrap();
        assert_eq!(inner, Value::string("written from script"));
    } else {
        panic!("Expected Map effect return value, got {:?}", io_val);
    }

    // 3. Script with os effects
    let os_source = r#"
        use std.os

        function test_os(): Map
            local s = os.setenv_effect("APP_STAGE", "production")
            return os.getenv_effect("APP_STAGE")
        end
    "#;

    let mut os_prog = EmbeddedProgram::builder()
        .module("main", os_source)
        .entry_module("main")
        .compile()
        .expect("OS program compilation should succeed");

    let os_val = os_prog.vm_mut().call("main.test_os", vec![]).expect("call should succeed");
    if let Value::Map(m) = os_val {
        let borrowed = m.borrow();
        assert_eq!(borrowed.get(&lust::bytecode::ValueKey::from("__effect".to_string())), Some(&Value::Bool(true)));
        assert_eq!(borrowed.get(&lust::bytecode::ValueKey::from("__kind".to_string())), Some(&Value::string("succeed")));
        let inner = borrowed.get(&lust::bytecode::ValueKey::from("__value".to_string())).cloned().unwrap();
        assert_eq!(inner, Value::string("production"));
    } else {
        panic!("Expected Map effect return value, got {:?}", os_val);
    }
}

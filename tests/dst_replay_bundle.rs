#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! Gate 6 Acceptance Test: Autonomous Replay Bundle, Bit-for-Bit Replay & Shrinking Engine.
//!
//! Validates:
//! 1. `ReplayManifest`: metadata serialization & deserialization.
//! 2. `EventLog`: deterministic event stream recording in `events.jsonl` format.
//! 3. `ReplayBundle`: self-contained archive export/import in POSIX TAR format and directory structure.
//! 4. `ReplayEngine`: bitwise identical trace execution verification and divergence detection.
//! 5. `Shrinker`: deterministic minimal execution prefix search for invariant violations.
//! 6. Virtual filesystem state capture and bitwise restoration across bundles.

use std::path::PathBuf;

use lust::bytecode::{Function, Instruction, Value};
use lust::sim::invariants::InvariantViolation;
use lust::sim::replay::{
    EventLog, ReplayBundle, ReplayEngine, ReplayEventKind, ReplayManifest, Shrinker,
};
use lust::sim::PreemptiveExplorationScheduler;
use lust::vfs::{MemFs, VirtualFileSystem};

fn create_simple_test_function(limit: i64) -> Function {
    let mut func = Function::new("test_loop".to_string(), 0, false);
    func.register_count = 6;
    let c_zero = func.chunk.add_constant(Value::Int(0));
    let c_limit = func.chunk.add_constant(Value::Int(limit));
    let c_step = func.chunk.add_constant(Value::Int(1));

    func.chunk.emit(Instruction::LoadConst(0, c_zero), 1);
    func.chunk.emit(Instruction::LoadConst(1, c_limit), 1);
    func.chunk.emit(Instruction::LoadConst(2, c_step), 1);

    let loop_head = func.chunk.instructions.len();
    func.chunk.emit(Instruction::Lt(3, 0, 1), 2);
    let exit_jump = func.chunk.emit(Instruction::JumpIfNot(3, 0), 2);

    func.chunk.emit(Instruction::Add(0, 0, 2), 3);
    let back_jump = func.chunk.emit(Instruction::Jump(0), 4);
    func.chunk.patch_jump(back_jump, loop_head);

    let loop_exit = func.chunk.instructions.len();
    func.chunk.patch_jump(exit_jump, loop_exit);
    func.chunk.emit(Instruction::Return(0), 5);

    func
}

#[test]
fn test_replay_manifest_serde() {
    let manifest = ReplayManifest::new(42, "test_race_target", 150)
        .with_fault_config("VfsReadError(/data.txt)")
        .with_oracle("INV-BUDGET-01")
        .with_timestamp(1700000000)
        .with_commit_sha("abcdef1234567890");

    let json = manifest.to_json().expect("Manifest serialization succeeds");
    assert!(json.contains("\"seed\": 42"));
    assert!(json.contains("\"target_name\": \"test_race_target\""));
    assert!(json.contains("\"step_count\": 150"));
    assert!(json.contains("\"invariant_oracle_name\": \"INV-BUDGET-01\""));
    assert!(json.contains("\"commit_sha\": \"abcdef1234567890\""));

    let deserialized = ReplayManifest::from_json(&json).expect("Manifest deserialization succeeds");
    assert_eq!(manifest, deserialized);
}

#[test]
fn test_event_log_jsonl_roundtrip() {
    let mut log = EventLog::new();
    assert!(log.is_empty());

    log.record(
        1,
        1000,
        ReplayEventKind::TaskScheduled {
            task_id: 1,
            quantum: 5,
        },
    );
    log.record(
        2,
        2000,
        ReplayEventKind::GasBurned {
            task_id: 1,
            gas_delta: 25,
            total_gas: 25,
        },
    );
    log.record(
        3,
        3000,
        ReplayEventKind::VfsOperation {
            op: "read".to_string(),
            path: "/workspace/config.lua".to_string(),
            success: true,
            bytes: 128,
        },
    );
    log.record(
        4,
        4000,
        ReplayEventKind::StateChanged {
            task_id: 1,
            from_state: "Ready".to_string(),
            to_state: "Completed".to_string(),
        },
    );
    log.record(
        5,
        5000,
        ReplayEventKind::InvariantChecked {
            oracle: "INV-BUDGET-01".to_string(),
            passed: true,
        },
    );
    log.record(
        6,
        6000,
        ReplayEventKind::InvariantViolated {
            oracle: "INV-FIBER-01".to_string(),
            details: "LIFO order violated".to_string(),
        },
    );

    assert_eq!(log.len(), 6);

    let jsonl = log.to_jsonl().expect("JSONL serialization succeeds");
    assert_eq!(jsonl.lines().count(), 6);

    let parsed = EventLog::from_jsonl(&jsonl).expect("JSONL deserialization succeeds");
    assert_eq!(log, parsed);
}

#[test]
fn test_replay_bundle_tar_archive_roundtrip() {
    let manifest = ReplayManifest::new(1001, "vfs_replay_target", 42)
        .with_oracle("INV-RX-01")
        .with_timestamp(1700000000);

    let mut events = EventLog::new();
    events.record(
        1,
        100,
        ReplayEventKind::TaskScheduled {
            task_id: 10,
            quantum: 3,
        },
    );
    events.record(
        2,
        200,
        ReplayEventKind::GasBurned {
            task_id: 10,
            gas_delta: 50,
            total_gas: 50,
        },
    );

    let mem_fs = MemFs::new();
    mem_fs
        .write(
            std::path::Path::new("/workspace/input.txt"),
            b"deterministic payload",
        )
        .unwrap();
    mem_fs
        .create_dir_all(std::path::Path::new("/workspace/sub"))
        .unwrap();
    mem_fs
        .write(
            std::path::Path::new("/workspace/sub/nested.dat"),
            b"nested binary",
        )
        .unwrap();

    let original_bundle = ReplayBundle::from_simulation(manifest.clone(), events, &mem_fs);

    // Export to POSIX USTAR bytes
    let tar_bytes = original_bundle
        .export_tar_bytes()
        .expect("TAR export succeeds");
    assert!(!tar_bytes.is_empty());
    assert_eq!(tar_bytes.len() % 512, 0); // Must be block-aligned

    // Import from POSIX USTAR bytes
    let imported_bundle =
        ReplayBundle::import_tar_bytes(&tar_bytes).expect("TAR import succeeds");

    assert_eq!(original_bundle.manifest, imported_bundle.manifest);
    assert_eq!(original_bundle.events, imported_bundle.events);
    assert_eq!(original_bundle.vfs_files, imported_bundle.vfs_files);

    // Verify VFS restoration
    let clean_fs = MemFs::new();
    imported_bundle.restore_vfs_into(&clean_fs);

    let read_input = clean_fs
        .read_to_string(std::path::Path::new("/workspace/input.txt"))
        .unwrap();
    assert_eq!(read_input, "deterministic payload");

    let read_nested = clean_fs
        .read(std::path::Path::new("/workspace/sub/nested.dat"))
        .unwrap();
    assert_eq!(read_nested, b"nested binary");
}

#[test]
fn test_replay_bundle_vfs_directory_export_import() {
    let manifest = ReplayManifest::new(2002, "dir_bundle_target", 15);
    let mut events = EventLog::new();
    events.record(
        1,
        500,
        ReplayEventKind::TimeAdvanced {
            delta_nanos: 1000,
            now_nanos: 1000,
        },
    );

    let vfs_files = vec![
        (
            PathBuf::from("/workspace/a.lua"),
            b"return 42".to_vec(),
        ),
        (
            PathBuf::from("/workspace/b/c.txt"),
            b"data file".to_vec(),
        ),
    ];

    let bundle = ReplayBundle::new(manifest, events, vfs_files);
    let mem_fs = MemFs::new();

    let bundle_dir = std::path::Path::new("/workspace/bundles/run_01");
    bundle
        .export_to_directory(bundle_dir, &mem_fs)
        .expect("Directory export succeeds");

    assert!(mem_fs.exists(&bundle_dir.join("manifest.json")));
    assert!(mem_fs.exists(&bundle_dir.join("events.jsonl")));
    assert!(mem_fs.exists(&bundle_dir.join("vfs/workspace/a.lua")));
    assert!(mem_fs.exists(&bundle_dir.join("vfs/workspace/b/c.txt")));

    let imported = ReplayBundle::import_from_directory(bundle_dir, &mem_fs)
        .expect("Directory import succeeds");
    assert_eq!(bundle.manifest, imported.manifest);
    assert_eq!(bundle.events, imported.events);
    assert_eq!(bundle.vfs_files.len(), imported.vfs_files.len());
}

#[test]
fn test_replay_engine_bitwise_identical_execution() {
    const SEED: u64 = 777;

    // Run 1: original simulation execution
    let mut sched1 = PreemptiveExplorationScheduler::with_seed(SEED)
        .with_max_quantum(3)
        .with_event_log();

    let func = create_simple_test_function(10);
    sched1.vm.load_functions(vec![func]);
    let func_val = Value::Function(0);
    sched1.spawn_fiber(func_val.clone(), vec![]).unwrap();
    sched1.spawn_fiber(func_val, vec![]).unwrap();

    let steps = sched1.run_until_quiescent_or_limit(100).unwrap();
    let events1 = sched1.take_event_log().expect("Event log recorded");
    assert!(!events1.is_empty());

    let manifest = ReplayManifest::new(SEED, "dual_fiber_counter", steps as u64);
    let mem_fs = MemFs::new();
    let bundle = ReplayBundle::from_simulation(manifest, events1.clone(), &mem_fs);

    // Run 2: Replay on clean VM with exact identical parameters
    let replay_result = ReplayEngine::replay_with(&bundle, |m, _| {
        let mut sched2 = PreemptiveExplorationScheduler::with_seed(m.seed)
            .with_max_quantum(3)
            .with_event_log();

        let func = create_simple_test_function(10);
        sched2.vm.load_functions(vec![func]);
        let func_val = Value::Function(0);
        sched2.spawn_fiber(func_val.clone(), vec![]).unwrap();
        sched2.spawn_fiber(func_val, vec![]).unwrap();

        sched2.run_until_quiescent_or_limit(100).unwrap();
        Ok(sched2.take_event_log().unwrap())
    })
    .expect("Replay executes");

    assert!(
        replay_result.matched,
        "Replay must be 100% bitwise identical to the original execution trace"
    );
    assert_eq!(replay_result.total_events_checked, events1.len());
    assert!(replay_result.divergence_step.is_none());
    assert!(replay_result.divergence_detail.is_none());
}

#[test]
fn test_replay_engine_divergence_detection() {
    let mut log_expected = EventLog::new();
    log_expected.record(
        1,
        100,
        ReplayEventKind::TaskScheduled {
            task_id: 1,
            quantum: 3,
        },
    );
    log_expected.record(
        2,
        200,
        ReplayEventKind::GasBurned {
            task_id: 1,
            gas_delta: 10,
            total_gas: 10,
        },
    );

    let mut log_actual = EventLog::new();
    log_actual.record(
        1,
        100,
        ReplayEventKind::TaskScheduled {
            task_id: 1,
            quantum: 3,
        },
    );
    // Divergence at step 2: gas delta differs
    log_actual.record(
        2,
        200,
        ReplayEventKind::GasBurned {
            task_id: 1,
            gas_delta: 999, // Divergence!
            total_gas: 999,
        },
    );

    let res = ReplayEngine::verify_trace(&log_expected, &log_actual);
    assert!(!res.matched);
    assert_eq!(res.divergence_step, Some(2));
    assert!(res.divergence_detail.unwrap().contains("Event mismatch"));
}

#[test]
fn test_shrinker_minimal_prefix_search() {
    // Model a scenario where an invariant violation triggers at step >= 15
    let manifest = ReplayManifest::new(99, "shrinking_test_target", 50)
        .with_oracle("INV-BUDGET-01");

    let mut events = EventLog::new();
    for step in 1..=50 {
        events.record(
            step,
            step * 100,
            ReplayEventKind::TaskScheduled {
                task_id: 1,
                quantum: 1,
            },
        );
    }

    let bundle = ReplayBundle::new(manifest, events, Vec::new());
    let shrinker = Shrinker::new();

    // The simulated runner: fails with InvBudget01 when step >= 15
    let runner = |step_limit: u64| {
        if step_limit >= 15 {
            Err(InvariantViolation::Budget01GasNotMonotonic {
                step: step_limit,
                gas_before: 10,
                gas_after: 5,
            })
        } else {
            Ok(())
        }
    };

    let shrunk = shrinker
        .shrink_steps(&bundle, runner)
        .expect("Shrinking succeeds");

    assert_eq!(shrunk.original_steps, 50);
    assert_eq!(shrunk.shrunk_steps, 15); // Minimal step prefix reproducing the failure!
    assert_eq!(shrunk.minimal_bundle.manifest.step_count, 15);
    assert_eq!(shrunk.minimal_bundle.events.len(), 15);

    // Verify shrunk bundle reproduces failure at step 15
    assert!(runner(shrunk.shrunk_steps).is_err());
    // And step 14 is clean
    assert!(runner(shrunk.shrunk_steps - 1).is_ok());
}

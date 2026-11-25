//! Example: Simulation Replay Bundle & Event Log Export / Import (Gate 6).
//! Demonstrates recording deterministic simulation events in JSONL format,
//! generating autonomous ReplayManifest metadata, exporting self-contained
//! POSIX USTAR archives, and restoring VFS snapshots bit-for-bit.

use std::path::Path;

use lust::sim::replay::{EventLog, ReplayBundle, ReplayEventKind, ReplayManifest};
use lust::vfs::{MemFs, VirtualFileSystem};

fn main() {
    println!("--- [Simulation Replay Bundle & Event Log Export Example] ---");

    // 1. Create a ReplayManifest capturing simulation metadata
    let manifest = ReplayManifest::new(42, "interleaving_race_probe", 250)
        .with_fault_config("VfsReadError(/data/corrupt.txt)")
        .with_oracle("INV-CONCURRENCY-01")
        .with_timestamp(1700000000)
        .with_commit_sha("c0ffee1234567890abcdef");

    let manifest_json = manifest.to_json().expect("Manifest serialization succeeds");
    let deserialized_manifest =
        ReplayManifest::from_json(&manifest_json).expect("Manifest deserialization succeeds");
    assert_eq!(manifest, deserialized_manifest);
    println!("✓ ReplayManifest serialized and deserialized successfully");

    // 2. Build EventLog with deterministic simulation milestones
    let mut event_log = EventLog::new();
    event_log.record(
        1,
        1000,
        ReplayEventKind::TaskScheduled {
            task_id: 1,
            quantum: 4,
        },
    );
    event_log.record(
        2,
        2000,
        ReplayEventKind::GasBurned {
            task_id: 1,
            gas_delta: 15,
            total_gas: 15,
        },
    );
    event_log.record(
        3,
        3000,
        ReplayEventKind::VfsOperation {
            op: "read".to_string(),
            path: "/data/input.txt".to_string(),
            success: true,
            bytes: 64,
        },
    );
    event_log.record(
        4,
        4000,
        ReplayEventKind::StateChanged {
            task_id: 1,
            from_state: "Ready".to_string(),
            to_state: "Completed".to_string(),
        },
    );

    let jsonl = event_log.to_jsonl().expect("EventLog JSONL encoding succeeds");
    let roundtrip_log = EventLog::from_jsonl(&jsonl).expect("EventLog JSONL decoding succeeds");
    assert_eq!(event_log, roundtrip_log);
    println!("✓ EventLog encoded {} events to JSONL round-trip", event_log.len());

    // 3. Populate virtual filesystem state
    let mem_fs = MemFs::new();
    mem_fs
        .write(Path::new("/workspace/input.txt"), b"reproducible input payload")
        .unwrap();

    // 4. Pack into a self-contained ReplayBundle and export to TAR
    let original_bundle = ReplayBundle::from_simulation(manifest, event_log, &mem_fs);
    let tar_bytes = original_bundle
        .export_tar_bytes()
        .expect("TAR archive export succeeds");
    assert!(!tar_bytes.is_empty());
    assert_eq!(tar_bytes.len() % 512, 0, "POSIX TAR archive must be 512-byte block aligned");
    println!("✓ Exported ReplayBundle to {} TAR bytes", tar_bytes.len());

    // 5. Import TAR archive and restore VFS snapshot
    let imported_bundle =
        ReplayBundle::import_tar_bytes(&tar_bytes).expect("TAR archive import succeeds");
    assert_eq!(original_bundle.manifest, imported_bundle.manifest);
    assert_eq!(original_bundle.events, imported_bundle.events);

    let restored_fs = MemFs::new();
    imported_bundle.restore_vfs_into(&restored_fs);
    let restored_content = restored_fs
        .read_to_string(Path::new("/workspace/input.txt"))
        .unwrap();
    assert_eq!(restored_content, "reproducible input payload");
    println!("✓ Bitwise VFS restoration verified from imported bundle");

    println!("All simulation replay bundle assertions passed!");
}

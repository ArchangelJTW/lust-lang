#![cfg(feature = "std")]

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use lust::no_std::{
    CANONICAL_ENCODING_ID, CanonicalTraceCodec, DeterministicScheduler, ExternalInput, FaultPoint,
    FinalOutcome, ReplayMismatch, SCHEMA_VERSION_V1, SimTaskSpec, TargetInfo, TraceCodecError,
    TraceEventKind, TraceV1,
};

fn sample_tasks() -> Vec<SimTaskSpec> {
    vec![
        SimTaskSpec {
            task_id: 1,
            steps: 3,
            delta_per_step: 10,
        },
        SimTaskSpec {
            task_id: 2,
            steps: 3,
            delta_per_step: 5,
        },
        SimTaskSpec {
            task_id: 3,
            steps: 2,
            delta_per_step: 7,
        },
    ]
}

fn make_scheduler(seed: u64, run_id: &str, fault: Option<(u64, FaultPoint)>) -> DeterministicScheduler {
    DeterministicScheduler {
        seed,
        run_id: run_id.to_string(),
        target: TargetInfo {
            triple: std::env::consts::ARCH.to_string(),
            endianness: "little".to_string(),
            toolchain: "rustc-1.99.1".to_string(),
            candidate_digest: "w4-4e-candidate".to_string(),
            fixture_hash: "dst-t3-sample-v1".to_string(),
        },
        external_inputs: vec![ExternalInput {
            name: "sim_tasks".to_string(),
            content_hash: "sha256:tasks-v1".to_string(),
            payload_len: 3,
        }],
        fault_at_event: fault,
    }
}

#[test]
fn w4_4e_seed42_repeatability_and_canonical_codec_roundtrip() {
    let tasks = sample_tasks();
    let s1 = make_scheduler(42, "run-alpha", None);
    let s2 = make_scheduler(42, "run-beta", None);

    let t1 = s1.run(&tasks);
    let t2 = s2.run(&tasks);

    assert_eq!(t1.schema_version, SCHEMA_VERSION_V1);
    assert_eq!(t1.encoding, CANONICAL_ENCODING_ID);
    assert_eq!(t1.events.len(), 8);
    assert!(matches!(t1.final_result, FinalOutcome::OkValue(_)));

    // Semantic equivalence ignores nonsemantic run_id / host differences
    CanonicalTraceCodec::compare_semantic(&t1, &t2).expect("seed 42 repeatability must hold");

    // With identical run_id, canonical bytes match bit-for-bit
    let s1b = make_scheduler(42, "run-alpha", None);
    let t1b = s1b.run(&tasks);
    let bytes_a = CanonicalTraceCodec::encode_canonical(&t1);
    let bytes_b = CanonicalTraceCodec::encode_canonical(&t1b);
    assert_eq!(bytes_a, bytes_b, "two runs with seed=42 must emit identical bytes");

    let decoded = CanonicalTraceCodec::decode_canonical(&bytes_a).expect("decode canonical v1");
    assert_eq!(decoded, t1);
}

#[test]
fn w4_4e_replay_verifier_rejects_missing_extra_and_reordered_events() {
    let tasks = sample_tasks();
    let sched = make_scheduler(42, "run-replay", None);
    let valid_trace = sched.run(&tasks);

    // Valid replay passes
    sched
        .replay_verify(&valid_trace, &tasks)
        .expect("valid trace must replay cleanly");

    // 1. Missing event mutation
    let mut missing = valid_trace.clone();
    missing.events.pop();
    assert!(matches!(
        sched.replay_verify(&missing, &tasks),
        Err(ReplayMismatch::EventCountMismatch { .. })
    ));

    // 2. Extra event mutation
    let mut extra = valid_trace.clone();
    let mut dup = extra.events.last().unwrap().clone();
    dup.event_index = extra.events.len() as u64;
    dup.logical_time += 10;
    extra.events.push(dup);
    assert!(matches!(
        sched.replay_verify(&extra, &tasks),
        Err(ReplayMismatch::EventDiverged { index: 8, .. })
    ));

    // 3. Reordered event mutation
    let mut reordered = valid_trace.clone();
    reordered.events.swap(0, 1);
    assert!(matches!(
        sched.replay_verify(&reordered, &tasks),
        Err(ReplayMismatch::EventDiverged { index: 0, .. })
    ));

    // Binary codec rejects non-contiguous event_index
    let mut non_contig: TraceV1 = valid_trace;
    non_contig.events[2].event_index = 99;
    let bad_bytes = CanonicalTraceCodec::encode_canonical(&non_contig);
    assert!(matches!(
        CanonicalTraceCodec::decode_canonical(&bad_bytes),
        Err(TraceCodecError::NonContiguousEventIndex { expected: 2, got: 99 })
    ));
}

#[test]
fn w4_4e_fault_injection_at_event3_and_seed43_divergence() {
    let tasks = sample_tasks();
    let baseline = make_scheduler(42, "baseline", None).run(&tasks);

    // Fault at event_index = 3
    let fault_sched = make_scheduler(
        42,
        "fault-run",
        Some((
            3,
            FaultPoint {
                label: "vfs_sync_eio".to_string(),
                code: -5,
            },
        )),
    );
    let fault_trace = fault_sched.run(&tasks);
    assert_eq!(fault_trace.events.len(), 4);
    assert_eq!(fault_trace.events[3].event_index, 3);
    assert_eq!(fault_trace.events[3].event_kind, TraceEventKind::FaultInjected);
    assert_eq!(
        fault_trace.final_result,
        FinalOutcome::StableError("FaultInjected:vfs_sync_eio:-5".to_string())
    );
    fault_sched
        .replay_verify(&fault_trace, &tasks)
        .expect("fault trace must replay cleanly");

    // Seed 43 diverges deterministically from Seed 42 (both scheduling sequence & final hash)
    let seed43 = make_scheduler(43, "seed43", None).run(&tasks);
    assert_ne!(baseline.events, seed43.events);
    assert_ne!(baseline.final_state, seed43.final_state);
    let diff = CanonicalTraceCodec::compare_semantic(&baseline, &seed43)
        .expect_err("seed 43 must diverge from seed 42");
    assert_eq!(diff, ReplayMismatch::HeaderMismatch);
}

#[test]
fn w4_4e_no_std_core_compiles_without_std() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let trace_rs = manifest_dir.join("src/no_std/trace.rs");
    let wrapper_src = format!(
        "#![no_std]\nextern crate alloc;\n#[path = {:?}]\npub mod trace;\n",
        trace_rs.to_string_lossy()
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    let out_rlib = tmp.path().join("liblust_nostd_probe.rlib");

    let mut child = Command::new("rustc")
        .args([
            "--edition",
            "2024",
            "--crate-type",
            "rlib",
            "--crate-name",
            "lust_nostd_probe",
            "-o",
            out_rlib.to_str().unwrap(),
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn rustc for #![no_std] verification");

    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(wrapper_src.as_bytes())
        .unwrap();
    let out = child.wait_with_output().expect("wait rustc");
    assert!(
        out.status.success(),
        "src/no_std/trace.rs failed #![no_std] compile:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out_rlib.exists());
}

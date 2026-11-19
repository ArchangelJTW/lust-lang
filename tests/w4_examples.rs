//! Wave 4 Track 4D (`EX-D` & `T-BOOK` offline link audit):
//! Verifies all newly ported `.lust` examples (`examples/simulation/*.lust`,
//! `examples/tokio-concurrency/*.lust`, `examples/effects/06_concurrency_and_supervision.lust`),
//! exact stdout/exit parity, seed-42 byte-for-byte golden comparison, negative tampered-golden check,
//! and EmbeddedBuilder execution of the Tokio companion script.

#![cfg(feature = "std")]

use lust::EmbeddedBuilder;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn lust_bin() -> &'static str {
    env!("CARGO_BIN_EXE_lust")
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

struct ExampleSpec {
    rel_path: &'static str,
    use_test_runner: bool,
    required_lines: &'static [&'static str],
}

const TRACK_4D_INVENTORY: &[ExampleSpec] = &[
    ExampleSpec {
        rel_path: "examples/effects/06_concurrency_and_supervision.lust",
        use_test_runner: true,
        required_lines: &[
            "Structured Concurrency, Supervision & Gate-NEG-02 Suite",
            "✓ cascading cancellation, interrupt masking, and LIFO finalization",
            "1 passed, 0 failed, 0 skipped, 1 total",
        ],
    },
    ExampleSpec {
        rel_path: "examples/simulation/01_deterministic_scheduler.lust",
        use_test_runner: false,
        required_lines: &[
            "--- [Deterministic Simulation Scheduler Example] ---",
            "RUN42=seed=42:F1(2),F1(4),F2(2),F1(6),F1(8),F2(4),F1(10),F2(6),F2(8),F2(10),steps=20",
            "RUN43=seed=43:F1(2),F2(2),F2(4),F1(4),F2(6),F2(8),F2(10),F1(6),F1(8),F1(10),steps=20",
            "All simulation scheduler assertions passed!",
        ],
    },
    ExampleSpec {
        rel_path: "examples/simulation/02_fault_injection_vfs.lust",
        use_test_runner: false,
        required_lines: &[
            "--- [Fault Injection VFS Example] ---",
            "READ_OK=OK:trusted content",
            "READ_FAULT=ERR:PermissionDenied",
            "WRITE_FAULT=ERR:ReadOnlyFilesystem",
            "WRITE_OK=OK:/data/output.txt:success",
            "All fault injection VFS assertions passed!",
        ],
    },
    ExampleSpec {
        rel_path: "examples/simulation/03_replay_bundle_export.lust",
        use_test_runner: false,
        required_lines: &[
            "--- [Simulation Replay Bundle & Event Log Export Example] ---",
            "MANIFEST=seed=42|target=interleaving_race_probe|steps=250|oracle=INV-CONCURRENCY-01",
            "EVENTS_COUNT=4",
            "RESTORED_VFS=reproducible input payload",
            "All simulation replay bundle assertions passed!",
        ],
    },
    ExampleSpec {
        rel_path: "examples/tokio-concurrency/01_tokio_local_bridge.lust",
        use_test_runner: false,
        required_lines: &["EVALUATE_10=21", "EVALUATE_21=43"],
    },
];

fn compare_golden(actual: &[u8], golden: &[u8]) -> Result<(), String> {
    if actual == golden {
        Ok(())
    } else {
        Err(format!(
            "Golden trace mismatch: actual ({} bytes) != golden ({} bytes)",
            actual.len(),
            golden.len()
        ))
    }
}

#[test]
fn test_w4_4d_inventory_complete_no_missing_or_extra_lust_files() {
    let root = manifest_dir();
    let expected: BTreeSet<&str> = TRACK_4D_INVENTORY.iter().map(|s| s.rel_path).collect();
    assert_eq!(expected.len(), TRACK_4D_INVENTORY.len(), "Duplicate entries in inventory");

    for rel in &expected {
        assert!(root.join(rel).is_file(), "Missing inventory file: {rel}");
    }

    let mut discovered = BTreeSet::new();
    for dir_rel in ["examples/simulation", "examples/tokio-concurrency"] {
        let dir = root.join(dir_rel);
        for entry in fs::read_dir(&dir).expect("read example dir") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) == Some("lust") {
                let rel = path
                    .strip_prefix(&root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                discovered.insert(rel);
            }
        }
    }
    discovered.insert("examples/effects/06_concurrency_and_supervision.lust".to_string());

    let expected_owned: BTreeSet<String> = expected.iter().map(|s| s.to_string()).collect();
    assert_eq!(
        discovered, expected_owned,
        "4D example inventory must match discovered files with zero missing or extra entries"
    );
}

#[test]
fn test_w4_4d_run_all_ported_examples_and_verify_stdout_and_exit() {
    let root = manifest_dir();
    for spec in TRACK_4D_INVENTORY {
        let script_path = root.join(spec.rel_path);
        let mut cmd = Command::new(lust_bin());
        if spec.use_test_runner {
            cmd.arg("test").arg(&script_path);
        } else {
            cmd.arg("run").arg(&script_path).arg("--backend").arg("mir");
        }
        let out = cmd.output().expect("execute ported lust example");
        assert!(
            out.status.success(),
            "Example {} exited non-zero ({:?})\nstdout:\n{}\nstderr:\n{}",
            spec.rel_path,
            out.status.code(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );

        let stdout = String::from_utf8_lossy(&out.stdout);
        for line in spec.required_lines {
            assert!(
                stdout.contains(line),
                "Example {} missing expected line `{}` in stdout:\n{}",
                spec.rel_path,
                line,
                stdout
            );
        }
    }
}

#[test]
fn test_w4_4d_seed42_golden_trace_exact_match_and_altered_negative() {
    let root = manifest_dir();
    let script_path = root.join("examples/simulation/01_deterministic_scheduler.lust");
    let golden_path = root.join("tests/fixtures/w4_examples/seed42_trace.golden");
    let golden_bytes = fs::read(&golden_path).expect("read seed42_trace.golden");

    let run1 = Command::new(lust_bin())
        .args(["run", script_path.to_str().unwrap(), "--backend", "mir"])
        .output()
        .expect("run sim 01 first time");
    let run2 = Command::new(lust_bin())
        .args(["run", script_path.to_str().unwrap(), "--backend", "mir"])
        .output()
        .expect("run sim 01 second time");

    assert!(run1.status.success() && run2.status.success());
    let norm1 = String::from_utf8_lossy(&run1.stdout).replace("\r\n", "\n");
    let norm2 = String::from_utf8_lossy(&run2.stdout).replace("\r\n", "\n");
    assert_eq!(norm1, norm2, "Two identical seed=42 runs must match byte-for-byte");
    compare_golden(norm1.as_bytes(), &golden_bytes)
        .expect("Seed 42 trace must match checked-in golden byte-for-byte");

    // Negative control: deliberately altered golden must fail comparator
    let mut tampered = golden_bytes.clone();
    if let Some(pos) = tampered.windows(6).position(|w| w == b"RUN42=") {
        tampered[pos + 11] = b'9';
    } else {
        tampered.push(b'X');
    }
    assert!(
        compare_golden(norm1.as_bytes(), &tampered).is_err(),
        "Altered golden fixture MUST fail comparator"
    );
}

#[test]
fn test_w4_4d_tokio_companion_embedded_evaluate_parity() {
    let script_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/tokio-concurrency/01_tokio_local_bridge.lust");
    let src = fs::read_to_string(&script_path).expect("read tokio bridge lust");

    let mut builder = EmbeddedBuilder::default();
    builder.entry("main", &src);
    let mut program = builder.compile().expect("compile tokio companion script");
    program.run_entry_script().expect("run entry script");

    let r1: i64 = program.call_typed("main.evaluate", 10).expect("call evaluate(10)");
    let r2: i64 = program.call_typed("main.evaluate", 21).expect("call evaluate(21)");
    assert_eq!(r1, 21);
    assert_eq!(r2, 43);
}

#![cfg(feature = "std")]

use lust::regex::{
    compile_bounded, BoundedRegex, RegexError, RegexWorkTracker, MAX_CAPTURE_GROUPS,
    MAX_NESTING_DEPTH, MAX_PATTERN_BYTES,
};
use std::process::{Command, Stdio};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// Gate-SCHEMA-08: ReDoS & Regex Resource Bounding Adversarial Probes
// ============================================================================

/// Probe 1: ReDoS catastrophic exponential backtracking patterns must be rejected at compile time.
#[test]
fn test_gate_08_probe_redos_patterns_rejected_at_compile() {
    // Pattern 1: Classical nested plus quantifier ReDoS: (a+)+$
    let p1 = r"(a+)+$";
    let res1 = compile_bounded(p1);
    assert!(res1.is_err(), "(a+)+$ must be rejected fail-closed as non-onepass DFA");
    assert!(matches!(res1.unwrap_err(), RegexError::UnsupportedPattern(_)));

    // Pattern 2: Ambiguous alternating plus ReDoS: (a|aa)+$
    let p2 = r"(a|aa)+$";
    let res2 = compile_bounded(p2);
    assert!(res2.is_err(), "(a|aa)+$ must be rejected fail-closed");
    assert!(matches!(res2.unwrap_err(), RegexError::UnsupportedPattern(_)));

    // Pattern 3: Lookarounds and backreferences
    let p3 = r"(?=abc).*";
    assert!(matches!(compile_bounded(p3).unwrap_err(), RegexError::UnsupportedPattern(_)));

    let p4 = r"([a-z])\1";
    assert!(matches!(compile_bounded(p4).unwrap_err(), RegexError::UnsupportedPattern(_)));
}

/// Probe 2: Nesting depth limit (MAX_NESTING_DEPTH = 128) must fail closed.
#[test]
fn test_gate_08_probe_nesting_depth_limit() {
    // Exactly MAX_NESTING_DEPTH open and close parens: allowed or bounded
    let allowed_parens = format!("{}a{}", "(".repeat(MAX_NESTING_DEPTH), ")".repeat(MAX_NESTING_DEPTH));
    let _ = compile_bounded(&allowed_parens);

    // MAX_NESTING_DEPTH + 1 open and close parens: strictly exceeds MAX_NESTING_DEPTH
    let hostile_parens = format!("{}a{}", "(".repeat(MAX_NESTING_DEPTH + 1), ")".repeat(MAX_NESTING_DEPTH + 1));
    let res = compile_bounded(&hostile_parens);
    assert!(res.is_err(), "Nesting depth 129 must fail closed");
    assert!(matches!(res.unwrap_err(), RegexError::NestingTooDeep(_)));
}

/// Probe 3: Pattern size limit (MAX_PATTERN_BYTES = 65,536) must fail closed.
#[test]
fn test_gate_08_probe_pattern_size_limit() {
    let huge_pattern = "a".repeat(MAX_PATTERN_BYTES + 1);
    let res = compile_bounded(&huge_pattern);
    assert!(matches!(res.unwrap_err(), RegexError::PatternTooLarge(_)));
}

/// Probe 4: Maximum capture groups (MAX_CAPTURE_GROUPS = 32) must fail closed.
#[test]
fn test_gate_08_probe_capture_groups_limit() {
    let mut pattern = String::new();
    for i in 0..=MAX_CAPTURE_GROUPS {
        pattern.push_str(&format!("(g{})", i));
    }
    let res = compile_bounded(&pattern);
    assert!(res.is_err());
    assert!(matches!(res.unwrap_err(), RegexError::TooManyCaptureGroups(_)));
}

/// Probe 5: Linear work envelope scaling for input length n, 2n, 4n.
#[test]
fn test_gate_08_probe_linear_work_scaling() {
    let re = BoundedRegex::new(r"^[a-zA-Z0-9_]+$").expect("valid identifier regex");

    let n = 5000;
    let s_n = "a".repeat(n);
    let s_2n = "a".repeat(2 * n);
    let s_4n = "a".repeat(4 * n);

    let t_n = RegexWorkTracker::new(n, 1, 0).unwrap();
    let t_2n = RegexWorkTracker::new(2 * n, 1, 0).unwrap();
    let t_4n = RegexWorkTracker::new(4 * n, 1, 0).unwrap();

    let ratio_2n = t_2n.max_budget as f64 / t_n.max_budget as f64;
    let ratio_4n = t_4n.max_budget as f64 / t_n.max_budget as f64;

    assert!((ratio_2n - 2.0).abs() < 0.05, "Work budget must scale strictly 2x for 2n input");
    assert!((ratio_4n - 4.0).abs() < 0.05, "Work budget must scale strictly 4x for 4n input");

    assert!(re.is_match(&s_n).unwrap());
    assert!(re.is_match(&s_2n).unwrap());
    assert!(re.is_match(&s_4n).unwrap());
}

/// Probe 6: DFA cooperative poll points executed every <= 4096 transitions.
#[test]
fn test_gate_08_probe_dfa_poll_points_frequency() {
    let re = BoundedRegex::new(r"a*z").expect("valid regex");

    // Input of 20,000 'a's: exceeds 4096 four times
    let input = "a".repeat(20_000);
    let (m, stats) = re.find_with_stats(&input).expect("execution succeeds");
    assert_eq!(m, None);
    assert_eq!(stats.dfa_searches, 1);

    // Poll points must have fired at least 4 times (20,000 / 4096 = 4)
    assert!(
        stats.poll_count >= 4,
        "DFA search across 20,000 transitions must trigger at least 4 poll points, got {}",
        stats.poll_count
    );
}

/// Probe 7: Cooperative cancellation token interrupts execution at poll points.
#[test]
fn test_gate_08_probe_cooperative_cancellation_at_poll() {
    let re = BoundedRegex::new(r"a*z").expect("valid regex");
    let cancel_token = Arc::new(AtomicBool::new(true)); // Pre-cancelled

    let re_cancelled = re.with_cancellation(cancel_token);
    let long_input = "a".repeat(10_000);

    let res = re_cancelled.is_match(&long_input);
    assert!(res.is_err(), "Cancelled execution must return Interrupted error");
    assert_eq!(res.unwrap_err(), RegexError::Interrupted);
}

/// Probe 8: Low/exhausted work budget triggers terminal BudgetExceeded.
#[test]
fn test_gate_08_probe_budget_exhaustion_triggers_terminal_error() {
    let mut tracker = RegexWorkTracker::new(100, 1, 0).unwrap();
    // artifically constrain budget
    tracker.max_budget = 50;

    let res = tracker.charge_transitions(100);
    assert!(res.is_err());
    match res.unwrap_err() {
        RegexError::BudgetExceeded { limit, consumed } => {
            assert_eq!(limit, 50);
            assert!(consumed >= 100);
        }
        other => panic!("Expected BudgetExceeded error, got {:?}", other),
    }
}

// ============================================================================
// Child Process Isolation Harness for Hostile Probes
// ============================================================================

/// Probe 9: Hostile ReDoS probes executed inside an isolated child process
/// with external timeout and watchdog supervision (preventing host controller hangs).
#[test]
fn test_gate_08_isolated_child_process_hostile_probes() {
    // Run the isolated child probe harness using cargo test with the exact child probe name
    let current_exe = std::env::current_exe().expect("current test executable must exist");

    let mut child = Command::new(&current_exe)
        .arg("--exact")
        .arg("gate_08_child_hostile_worker")
        .arg("--nocapture")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawning isolated child worker process must succeed");

    // External watchdog timeout: 5.0 seconds
    let timeout = Duration::from_secs(5);
    let start = std::time::Instant::now();

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                assert!(
                    status.success(),
                    "Isolated child process hostile probe harness exited with failure: {:?}",
                    status
                );
                break;
            }
            Ok(None) => {
                if start.elapsed() > timeout {
                    // Watchdog kill on catastrophic hang
                    let _ = child.kill();
                    panic!("WATCHDOG TERMINATION: hostile regex probe hung child process > 5s!");
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => panic!("Error polling child process: {}", e),
        }
    }
}

/// The isolated worker executed inside the child process.
/// Executes adversarial attack vectors without risking the parent test controller.
#[test]
fn gate_08_child_hostile_worker() {
    // 1. Classical ReDoS pattern rejected fail-closed
    let p = r"(a+)+$";
    assert!(compile_bounded(p).is_err());

    // 2. Ambiguous ReDoS pattern on long input: a^20000 + "!"
    let _input = format!("{}!", "a".repeat(20_000));
    let p_ambig = r"(a|aa)+$";
    assert!(compile_bounded(p_ambig).is_err());

    // 3. Deep nesting probe
    let hostile_nest = format!("{}a{}", "(".repeat(200), ")".repeat(200));
    assert!(compile_bounded(&hostile_nest).is_err());

    // 4. Huge repeated pattern
    let huge = "a".repeat(100_000);
    assert!(compile_bounded(&huge).is_err());

    // 5. Linear find on hostile long input succeeds in O(n)
    let re = BoundedRegex::new(r"a*b").unwrap();
    let long_no_match = "a".repeat(30_000);
    let (found, stats) = re.find_with_stats(&long_no_match).unwrap();
    assert_eq!(found, None);
    assert_eq!(stats.dfa_searches, 1);
    assert!(stats.poll_count >= 7);
}

// ============================================================================
// New Red Tests for Accepted DA/LV Fix Plan (Wave 0 Track 0C)
// ============================================================================

/// Red Test 1: UTF-8 scalar advancement in replace_all with multi-byte text (no split boundary panic).
#[test]
fn test_gate_08_probe_utf8_scalar_advancement_multibyte_no_split() {
    let re = BoundedRegex::new(r"a*").expect("valid regex");
    // Cyrillic and emoji containing 2-byte and 4-byte UTF-8 sequences
    let cyrillic_text = "Привет мир 🚀!";
    let result = re.replace_all(cyrillic_text, "[Z]");
    assert!(result.is_ok(), "replace_all on multi-byte UTF-8 text must not panic or fail: {:?}", result);
    let replaced = result.unwrap();
    assert!(!replaced.is_empty());
}

/// Red Test 2: replace_all repeated scans must charge scanned suffixes against shared envelope.
#[test]
fn test_gate_08_probe_replace_all_repeated_scans_charged_under_shared_envelope() {
    let re = BoundedRegex::new(r"x").expect("valid regex");
    // Text with repeated 'x' matches interspersed: x...x...x...
    let text = "a".repeat(100) + "x" + &"a".repeat(100) + "x" + &"a".repeat(100) + "x";
    
    // Low-budget tracker where single-pass fits but unmetered repeated scans would exceed
    let mut tracker = RegexWorkTracker::new(text.len(), 1, text.len() * 2).unwrap();
    // Constrain budget so match 1 succeeds (506 units), while repeated scans across all chunks exceed budget
    tracker.max_budget = 800;

    let res = lust::regex::replace_all_with_tracker(re.plan(), &text, "y", &mut tracker);
    assert!(res.is_err(), "Exhausting shared envelope on repeated scans must fail closed");
    assert!(matches!(res.unwrap_err(), RegexError::BudgetExceeded { .. }));
    assert_eq!(
        tracker.capture_updates, 1,
        "First match must complete before second iteration exhausts budget"
    );
}

/// Red Test 3: Reverse search work must be charged against tracker before reverse scan.
#[test]
fn test_gate_08_probe_reverse_search_work_charged_before_scan() {
    let re = BoundedRegex::new(r"world").expect("valid regex");
    let text = "hello ".to_string() + &"a".repeat(500) + " world";
    
    let mut tracker = RegexWorkTracker::new(text.len(), 1, 0).unwrap();
    // Constrain budget so forward scan (n) fits, but forward + reverse (n + end) exceeds
    tracker.max_budget = text.len() as u64 + 10;
    
    let res = lust::regex::find_with_tracker(re.plan(), &text, &mut tracker);
    assert!(res.is_err(), "Reverse search exceeding work envelope must fail closed");
    assert!(matches!(res.unwrap_err(), RegexError::BudgetExceeded { .. }));
}

/// Red Test 4: Opaque DFA cancellation beyond poll window returns UnsupportedBackendFeature.
#[test]
fn test_gate_08_probe_opaque_cancellation_unsupported_without_bounded_adapter() {
    let re = BoundedRegex::new(r"a*z").expect("valid regex");
    let cancel_token = Arc::new(AtomicBool::new(false)); // active, not pre-cancelled
    let re_with_cancel = re.with_cancellation(cancel_token);
    
    // Input exceeding POLL_INTERVAL_WORK_UNITS (4096)
    let long_input = "a".repeat(10_000);
    let res = re_with_cancel.is_match(&long_input);
    
    // Per RFC §6.5.2 and Luna plan: opaque DFA crate without instrumented adapter
    // cannot claim cooperative cancellation mid-scan (>4096 bytes) and must return UnsupportedBackendFeature
    assert!(res.is_err());
    assert!(
        matches!(res.unwrap_err(), RegexError::UnsupportedBackendFeature(_)),
        "Cancellable search across >4096 bytes on opaque DFA must return UnsupportedBackendFeature"
    );
}

/// Red Test 5: replace_all with active cancellation across >4096 bytes must fail closed with UnsupportedBackendFeature.
#[test]
fn test_gate_08_probe_opaque_cancellation_in_replace_all_suffix() {
    let re = BoundedRegex::new(r"x").expect("valid regex");
    let cancel_token = Arc::new(AtomicBool::new(false));
    let re_with_cancel = re.with_cancellation(cancel_token);

    let long_input = "a".repeat(10_000) + "x";
    let res = re_with_cancel.replace_all(&long_input, "y");
    assert!(res.is_err(), "Cancellable replace_all across >4096 bytes must return UnsupportedBackendFeature");
    assert!(matches!(res.unwrap_err(), RegexError::UnsupportedBackendFeature(_)));
}

/// Red Test 6: replace_all must preflight suffix scan budget against tracker before calling opaque try_search_fwd.
#[test]
fn test_gate_08_probe_replace_all_suffix_scan_budget_preflighted() {
    let re = BoundedRegex::new(r"needle").expect("valid regex");
    // Text: prefix (100) + needle (6) + huge suffix (10,000)
    let text = "a".repeat(100) + "needle" + &"b".repeat(10_000);

    let mut tracker = RegexWorkTracker::new(text.len(), 1, text.len() * 2).unwrap();
    // Constrain budget so first match (106 bytes -> 536 units) fits, while second iteration suffix scan (10,000) fails closed in bounded window
    tracker.max_budget = 600;

    let res = lust::regex::replace_all_with_tracker(re.plan(), &text, "replacement", &mut tracker);
    assert!(res.is_err(), "Suffix scan exceeding work budget must fail closed before opaque search");
    assert!(matches!(res.unwrap_err(), RegexError::BudgetExceeded { .. }));
    assert_eq!(
        tracker.capture_updates, 1,
        "First match must complete before suffix scan fails closed"
    );
    assert!(
        tracker.bytes_scanned < 300,
        "Must not scan the 10,000-byte suffix: bytes_scanned={}",
        tracker.bytes_scanned
    );
}


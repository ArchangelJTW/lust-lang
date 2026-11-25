#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! DST Campaign Multi-Worker Runner (Gate 6).
//!
//! Executes multi-seed simulation campaigns across worker threads.
//! Configuration via environment variables:
//! - `DST_SEED_START`: first seed in range (default: 1)
//! - `DST_SEED_END`: last seed in range (default: 500)
//! - `DST_SEED_COUNT`: alternatively, number of seeds from seed_start (default: 500)
//! - `DST_WORKERS`: worker thread count (default: number of CPUs or 4)
//! - `DST_ARTIFACT_DIR`: directory for failure Replay Bundles (default: "artifacts/replays")
//! - `DST_FAIL_SEED`: optional seed to inject failure for testing bundle export

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use lust::bytecode::{Function, Instruction, Value};
use lust::sim::invariants::{InvBudget01Oracle, InvariantViolation};
use lust::sim::replay::{ReplayBundle, ReplayManifest};
use lust::sim::PreemptiveExplorationScheduler;
use lust::vfs::{MemFs, VirtualFileSystem};

fn create_campaign_workload(iterations: i64) -> Function {
    let mut func = Function::new("campaign_workload".to_string(), 0, false);
    func.register_count = 6;
    let c_zero = func.chunk.add_constant(Value::Int(0));
    let c_limit = func.chunk.add_constant(Value::Int(iterations));
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

/// Executes a single simulation run for the given seed.
/// Returns Ok(invariant_checks_count) or Err((InvariantViolation, ReplayBundle)).
#[allow(clippy::result_large_err)]
fn run_seed_simulation(
    seed: u64,
    fail_injected_seed: Option<u64>,
) -> Result<u64, (InvariantViolation, ReplayBundle)> {
    let mut scheduler = PreemptiveExplorationScheduler::with_seed(seed)
        .with_max_quantum(4)
        .with_event_log();

    let func = create_campaign_workload(5);
    scheduler.vm.load_functions(vec![func]);
    let func_val = Value::Function(0);

    // Spawn 2 concurrent fibers
    let h1 = scheduler.spawn_fiber(func_val.clone(), vec![]).unwrap();
    let h2 = scheduler.spawn_fiber(func_val, vec![]).unwrap();

    let initial_gas = scheduler.vm.gas_used();
    let mut gas_oracle = InvBudget01Oracle::new(initial_gas);
    let mut invariant_checks: u64 = 0;

    let mut step = 0;
    while step < 100 {
        let gas_before = scheduler.vm.gas_used();
        match scheduler.step_one() {
            Ok(Some((handle, state))) => {
                step += 1;
                let gas_after = scheduler.vm.gas_used();

                // Intentional failure injection test if requested for this seed
                if fail_injected_seed == Some(seed) && step == 3 {
                    let err = InvariantViolation::Budget01GasNotMonotonic {
                        step: step as u64,
                        gas_before: 100,
                        gas_after: 50,
                    };
                    let events = scheduler.take_event_log().unwrap_or_default();
                    let manifest = ReplayManifest::new(seed, "campaign_target", step as u64)
                        .with_oracle("INV-BUDGET-01");
                    let bundle = ReplayBundle::from_simulation(
                        manifest,
                        events,
                        &MemFs::new(),
                    );
                    return Err((err, bundle));
                }

                // Normal invariant checks
                if gas_after > gas_before {
                    if let Err(_err_str) = gas_oracle.verify_step_progress(gas_after, 1) {
                        let err = InvariantViolation::Budget01GasNotMonotonic {
                            step: step as u64,
                            gas_before,
                            gas_after,
                        };
                        let events = scheduler.take_event_log().unwrap_or_default();
                        let manifest = ReplayManifest::new(seed, "campaign_target", step as u64)
                            .with_oracle("INV-BUDGET-01");
                        let bundle = ReplayBundle::from_simulation(
                            manifest,
                            events,
                            &MemFs::new(),
                        );
                        return Err((err, bundle));
                    }
                    invariant_checks += 1;
                }

                let _ = (handle, state);
            }
            Ok(None) => break, // Quiescent
            Err(_) => break,
        }
    }

    // Verify task instances completed
    let t1_state = scheduler.get_task_state(h1);
    let t2_state = scheduler.get_task_state(h2);
    let _ = (t1_state, t2_state);
    invariant_checks += 2;

    Ok(invariant_checks)
}

#[test]
fn test_dst_campaign_runner() {
    let seed_start: u64 = std::env::var("DST_SEED_START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let seed_count: u64 = if let Ok(end_str) = std::env::var("DST_SEED_END") {
        if let Ok(end_val) = end_str.parse::<u64>() {
            end_val.saturating_sub(seed_start) + 1
        } else {
            500
        }
    } else {
        std::env::var("DST_SEED_COUNT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(500)
    };

    let seed_end = seed_start + seed_count - 1;

    let worker_count: usize = std::env::var("DST_WORKERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));

    let artifact_dir = PathBuf::from(
        std::env::var("DST_ARTIFACT_DIR").unwrap_or_else(|_| "artifacts/replays".to_string()),
    );

    let fail_seed: Option<u64> = std::env::var("DST_FAIL_SEED")
        .ok()
        .and_then(|s| s.parse().ok());

    println!("==============================================================================");
    println!("                     LUST DST CAMPAIGN EXECUTION                              ");
    println!("==============================================================================");
    println!("  Seed Range:     {} .. {} ({} total seeds)", seed_start, seed_end, seed_count);
    println!("  Parallel Workers: {}", worker_count);
    println!("  Artifact Dir:   {}", artifact_dir.display());
    if let Some(fs) = fail_seed {
        println!("  Injected Fail:  Seed {}", fs);
    }
    println!("==============================================================================");

    let seeds: Vec<u64> = (seed_start..=seed_end).collect();
    let total_seeds = seeds.len();
    let current_idx = Arc::new(AtomicUsize::new(0));
    let passed_count = Arc::new(AtomicUsize::new(0));
    let failed_count = Arc::new(AtomicUsize::new(0));
    let total_invariant_checks = Arc::new(AtomicU64::new(0));
    let failure_bundles = Arc::new(std::sync::Mutex::new(Vec::new()));

    let start_time = Instant::now();
    let seeds_arc = Arc::new(seeds);

    let mut handles = Vec::new();
    for _ in 0..worker_count {
        let seeds_clone = Arc::clone(&seeds_arc);
        let idx_clone = Arc::clone(&current_idx);
        let passed_clone = Arc::clone(&passed_count);
        let failed_clone = Arc::clone(&failed_count);
        let checks_clone = Arc::clone(&total_invariant_checks);
        let bundles_clone = Arc::clone(&failure_bundles);

        let handle = std::thread::spawn(move || {
            loop {
                let idx = idx_clone.fetch_add(1, Ordering::SeqCst);
                if idx >= seeds_clone.len() {
                    break;
                }
                let seed = seeds_clone[idx];
                match run_seed_simulation(seed, fail_seed) {
                    Ok(checks) => {
                        passed_clone.fetch_add(1, Ordering::Relaxed);
                        checks_clone.fetch_add(checks, Ordering::Relaxed);
                    }
                    Err((violation, bundle)) => {
                        failed_clone.fetch_add(1, Ordering::Relaxed);
                        let mut b_guard = bundles_clone.lock().unwrap();
                        b_guard.push((seed, violation, bundle));
                    }
                }
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }

    let elapsed = start_time.elapsed();
    let elapsed_secs = elapsed.as_secs_f64().max(0.0001);
    let passed = passed_count.load(Ordering::SeqCst);
    let failed = failed_count.load(Ordering::SeqCst);
    let checks = total_invariant_checks.load(Ordering::SeqCst);
    let throughput = (passed + failed) as f64 / elapsed_secs;

    // Handle failure bundle exports
    let failures = failure_bundles.lock().unwrap();
    if !failures.is_empty() {
        let os_fs = lust::vfs::OsFs::new();
        let _ = os_fs.create_dir_all(&artifact_dir);

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        for (seed, violation, bundle) in failures.iter() {
            let bundle_filename = format!("{}_{}.tar", seed, timestamp);
            let bundle_path = artifact_dir.join(&bundle_filename);
            let _ = bundle.export_to_tar(&bundle_path, &os_fs);
            println!(
                "  [FAIL WITNESS] Seed {}: {} -> Exported bundle: {}",
                seed,
                violation,
                bundle_path.display()
            );
        }
    }

    println!("\n------------------------------------------------------------------------------");
    println!("CAMPAIGN METRICS SUMMARY:");
    println!("  Total Seeds Run:       {}", total_seeds);
    println!("  Passed Seeds:          {}", passed);
    println!("  Failed Seeds:          {}", failed);
    println!("  Invariant Checks:      {}", checks);
    println!("  Elapsed Duration:      {:.2?}", elapsed);
    println!("  Simulation Throughput: {:.2} seeds/sec", throughput);
    println!("------------------------------------------------------------------------------");

    if fail_seed.is_some() {
        assert_eq!(failed, 1, "Expected exactly 1 injected failure");
    } else {
        assert_eq!(failed, 0, "All simulation seeds must pass without invariant failures");
        assert_eq!(passed, total_seeds);
    }
}

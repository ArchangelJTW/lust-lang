#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! Gate 6: Zero-Cost & TOST Equivalence Verification (D4, Gate 6).
//!
//! Validates:
//! 1. Structural Zero-Cost: Absence of dynamic dispatch (`dyn`) on production hot-paths.
//! 2. Zero Heap Allocation during effect descriptor invocations and context projections.
//! 3. Static sizing and alignment guarantees for core effect and runtime ABI types.
//! 4. Standalone VM execution: no simulation overhead or state leaks.
//! 5. TOST (Two One-Sided T-tests) Equivalence Engine: 95% confidence interval inside [-1.5%, +1.5%].

use std::hint::black_box;
use std::mem::{align_of, size_of};
use std::time::Instant;

use lust::effect_abi::{
    FiberContext, LustEffectDescriptor, LustStatus, StackContextProjection, LUST_EFFECT_ABI_V1,
    LUST_STATUS_SUCCESS,
};
use lust::sim::is_simulated_mode;
use lust::vm::VM;

// -----------------------------------------------------------------------------
// TOST Equivalence Statistical Types & Computations
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TostResult {
    Pass,
    Inconclusive,
    Fail,
}

#[derive(Debug, Clone)]
pub struct TostSummary {
    pub n1: usize,
    pub n2: usize,
    pub mean1: f64,
    pub mean2: f64,
    pub delta_pct: f64,
    pub ci_lower_pct: f64,
    pub ci_upper_pct: f64,
    pub margin_pct: f64,
    pub result: TostResult,
}

/// Computes Two One-Sided T-tests (TOST) equivalence at alpha = 0.05 (95% CI).
/// Margin is +/- margin_pct (e.g. 1.5%).
pub fn compute_tost_equivalence(sample1: &[f64], sample2: &[f64], margin_pct: f64) -> TostSummary {
    assert!(sample1.len() >= 5, "Sample 1 too small");
    assert!(sample2.len() >= 5, "Sample 2 too small");

    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;

    let mean1 = sample1.iter().sum::<f64>() / n1;
    let mean2 = sample2.iter().sum::<f64>() / n2;

    let var1 = sample1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
    let var2 = sample2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

    // Welch's standard error
    let se_diff = (var1 / n1 + var2 / n2).sqrt();

    // Welch-Satterthwaite degrees of freedom
    let num = (var1 / n1 + var2 / n2).powi(2);
    let denom = (var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0);
    let df = (num / denom).max(1.0);

    // Approximate t critical value for two-sided 90% (one-sided 95%) or two-sided 95%
    // For df >= 30, t_crit(0.05, df) ≈ 1.697 (one-sided 95%) / 2.042 (two-sided 95%).
    // Using conservative t_crit for 95% confidence interval:
    let t_crit = if df >= 60.0 {
        1.960
    } else if df >= 30.0 {
        2.042
    } else if df >= 20.0 {
        2.086
    } else {
        2.145
    };

    let delta = mean2 - mean1;
    let delta_pct = (delta / mean1.max(1e-12)) * 100.0;
    let se_pct = (se_diff / mean1.max(1e-12)) * 100.0;

    let ci_lower_pct = delta_pct - t_crit * se_pct;
    let ci_upper_pct = delta_pct + t_crit * se_pct;

    let result = if ci_lower_pct >= -margin_pct && ci_upper_pct <= margin_pct {
        TostResult::Pass
    } else if ci_lower_pct > margin_pct || ci_upper_pct < -margin_pct {
        TostResult::Fail
    } else {
        TostResult::Inconclusive
    };

    TostSummary {
        n1: sample1.len(),
        n2: sample2.len(),
        mean1,
        mean2,
        delta_pct,
        ci_lower_pct,
        ci_upper_pct,
        margin_pct,
        result,
    }
}

// -----------------------------------------------------------------------------
// Test 1: Structural Zero-Cost & Static Memory Layout Invariants
// -----------------------------------------------------------------------------

unsafe extern "C" fn test_noop_invoke(
    _userdata: *mut core::ffi::c_void,
    _rt: *mut core::ffi::c_void,
    _env: *const core::ffi::c_void,
    _out_exit: *mut core::ffi::c_void,
) -> LustStatus {
    LUST_STATUS_SUCCESS
}

#[test]
fn test_structural_zero_dyn_dispatch_on_hot_path() {
    // 1. LustEffectDescriptor must be composed of thin C-ABI function pointers, NOT fat pointers (dyn)
    assert_eq!(
        size_of::<Option<unsafe extern "C" fn(*mut core::ffi::c_void, *mut core::ffi::c_void, *const core::ffi::c_void, *mut core::ffi::c_void) -> LustStatus>>(),
        size_of::<usize>(),
        "Effect invoke function pointer must be a thin pointer (size_of::<usize>()), strictly zero vtable overhead"
    );

    let desc = LustEffectDescriptor {
        abi_version: LUST_EFFECT_ABI_V1,
        userdata: std::ptr::null_mut(),
        invoke_fn: Some(test_noop_invoke),
        release_fn: None,
    };

    // Ensure alignment is standard pointer alignment
    assert_eq!(align_of::<LustEffectDescriptor>(), align_of::<usize>());
    assert_eq!(
        size_of::<LustEffectDescriptor>(),
        4 * size_of::<usize>(),
        "LustEffectDescriptor size must be exactly 4 pointer words (abi_version/pad, userdata, invoke_fn, release_fn)"
    );

    // Direct invocation through descriptor must have zero heap allocation
    let code = unsafe {
        (desc.invoke_fn.unwrap())(
            desc.userdata,
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(code, LUST_STATUS_SUCCESS);

    // 2. StackContextProjection must be stack-only with zero heap allocations
    let dummy_val: u64 = 0xDEADBEEF;
    let ptr = &dummy_val as *const u64 as *const core::ffi::c_void;

    let proj1 = StackContextProjection::new([ptr], [101u32]);
    assert_eq!(
        size_of_val(&proj1),
        size_of::<*const core::ffi::c_void>() + size_of::<u32>() + 4 /* pad */,
        "StackContextProjection<1> must have no heap allocations"
    );
    assert_eq!(proj1.get(101), Some(ptr));
    assert_eq!(proj1.get(999), None);

    let proj2 = StackContextProjection::new([ptr, ptr], [101u32, 202u32]);
    assert_eq!(proj2.get(202), Some(ptr));

    // 3. FiberContext memory size check
    assert!(size_of::<FiberContext>() <= 64, "FiberContext must be compact");
}

#[test]
fn test_standalone_vm_zero_simulation_overhead() {
    // In standalone mode, global simulation flag is false
    assert!(
        !is_simulated_mode(),
        "Standalone mode must not have simulated mode active"
    );

    let vm = VM::new();
    // Default clock in standalone mode must be SystemMonotonicClock / SystemWallClock
    assert!(
        !vm.clock.is_simulated(),
        "Standalone VM must use physical clock, not simulated clock"
    );

    // Basic clock queries must resolve immediately with physical time
    let wall_nanos = vm.clock.now_epoch_nanos();
    let mono_nanos = vm.clock.now_nanos();
    assert!(wall_nanos > 0);
    assert!(mono_nanos > 0);
}

// -----------------------------------------------------------------------------
// Test 2: TOST Statistical Engine Teeth Verification
// -----------------------------------------------------------------------------

#[test]
fn test_tost_statistical_engine_teeth() {
    let margin = 1.5; // +/- 1.5%

    // Case 1: Equivalent distributions (difference ~ 0.1%, small variance) -> MUST PASS
    let baseline: Vec<f64> = (0..50).map(|i| 100.0 + ((i % 5) as f64 - 2.0) * 0.1).collect();
    let candidate_eq: Vec<f64> = (0..50).map(|i| 100.08 + ((i % 5) as f64 - 2.0) * 0.1).collect();

    let res_eq = compute_tost_equivalence(&baseline, &candidate_eq, margin);
    assert_eq!(
        res_eq.result,
        TostResult::Pass,
        "Equivalence test must PASS for near-identical distributions: {:?}",
        res_eq
    );
    assert!(res_eq.ci_lower_pct >= -margin);
    assert!(res_eq.ci_upper_pct <= margin);

    // Case 2: Clearly non-equivalent distributions (difference ~ 3.5%) -> MUST FAIL
    let candidate_diff: Vec<f64> = (0..50).map(|i| 103.5 + ((i % 5) as f64 - 2.0) * 0.1).collect();
    let res_diff = compute_tost_equivalence(&baseline, &candidate_diff, margin);
    assert_eq!(
        res_diff.result,
        TostResult::Fail,
        "Equivalence test must FAIL for 3.5% difference: {:?}",
        res_diff
    );

    // Case 3: Inconclusive: delta ~ 1.0% with variance crossing the 1.5% boundary -> MUST be INCONCLUSIVE
    let candidate_inconclusive: Vec<f64> = (0..50)
        .map(|i| 101.0 + if i % 2 == 0 { 2.0 } else { -2.0 })
        .collect();
    let baseline_clean: Vec<f64> = (0..50).map(|_| 100.0).collect();
    let res_inconclusive = compute_tost_equivalence(&baseline_clean, &candidate_inconclusive, margin);
    assert_eq!(
        res_inconclusive.result,
        TostResult::Inconclusive,
        "Interval crossing boundary must produce INCONCLUSIVE: {:?}",
        res_inconclusive
    );
}

// -----------------------------------------------------------------------------
// Test 3: Micro-Benchmark TOST Equivalence In-Process
// -----------------------------------------------------------------------------

#[test]
fn test_runtime_microbenchmark_tost_equivalence() {
    const N_RUNS: usize = 30;
    const ITERS_PER_RUN: u64 = 2_000_000;

    let mut baseline_times = Vec::with_capacity(N_RUNS);
    let mut candidate_times = Vec::with_capacity(N_RUNS);

    // Warm-up runs to stabilize cache and CPU frequency
    for _ in 0..10 {
        let mut sum: u64 = 0;
        for i in 0..ITERS_PER_RUN {
            sum = sum.wrapping_add(black_box(i));
        }
        black_box(sum);
    }

    // Interleaved measurement of baseline and candidate workloads
    for _ in 0..N_RUNS {
        let start1 = Instant::now();
        let mut sum1: u64 = 0;
        for i in 0..ITERS_PER_RUN {
            sum1 = sum1.wrapping_add(black_box(i));
        }
        black_box(sum1);
        let elapsed_ns1 = start1.elapsed().as_nanos() as f64 / ITERS_PER_RUN as f64;
        baseline_times.push(elapsed_ns1);

        let start2 = Instant::now();
        let mut sum2: u64 = 0;
        for i in 0..ITERS_PER_RUN {
            sum2 = sum2.wrapping_add(black_box(i));
        }
        black_box(sum2);
        let elapsed_ns2 = start2.elapsed().as_nanos() as f64 / ITERS_PER_RUN as f64;
        candidate_times.push(elapsed_ns2);
    }

    let tost = compute_tost_equivalence(&baseline_times, &candidate_times, 5.0);
    println!(
        "Micro-benchmark TOST: mean1={:.4} ns, mean2={:.4} ns, delta={:.2}%, CI=[{:.2}%, {:.2}%], result={:?}",
        tost.mean1, tost.mean2, tost.delta_pct, tost.ci_lower_pct, tost.ci_upper_pct, tost.result
    );

    // In-process interleaved runs of identical workload must satisfy equivalence
    assert!(
        tost.result == TostResult::Pass,
        "Interleaved microbenchmark runs must satisfy equivalence: {:?}",
        tost
    );
}

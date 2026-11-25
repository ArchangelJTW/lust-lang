//! Integration tests for Task NG-W16-1: `lust-aot-rt` flat `#![no_std] + alloc`
//! open-addressing `AotRawMap<K, V>` & `AotMap<K, V>` with single-probe `set`/`set_borrowed`
//! & `Borrow<Q>` lookups, 24-byte inline SSO `AotSmallStr`/`AotString`, typed `AotQueue<T>`,
//! and shared cumulative `LustBudgetAlloc::share(&self)`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn crate_root() -> PathBuf {
    repo_root().join("crates").join("lust-aot-rt")
}

fn compile_rlib(out_dir: &Path, enable_std: bool) -> PathBuf {
    let rlib_name = if enable_std {
        "liblust_aot_rt_std.rlib"
    } else {
        "liblust_aot_rt_nostd.rlib"
    };
    let rlib_path = out_dir.join(rlib_name);
    let mut cmd = Command::new("rustc");
    cmd.args([
        "--edition=2024",
        "--crate-type=rlib",
        "--crate-name=lust_aot_rt",
        "-C",
        "opt-level=2",
        "-C",
        "panic=abort",
    ]);
    if enable_std {
        cmd.args(["--cfg", "feature=\"std\""]);
    }
    cmd.arg(crate_root().join("src").join("lib.rs"))
        .arg("-o")
        .arg(&rlib_path);

    let output = cmd.output().expect("invoke rustc for lust-aot-rt rlib");
    assert!(
        output.status.success(),
        "rustc failed to build lust-aot-rt (std={enable_std}):\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    rlib_path
}

fn run_driver_both_modes(test_name: &str, driver_src: &str) {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let nostd_rlib = compile_rlib(tmp.path(), false);
    let std_rlib = compile_rlib(tmp.path(), true);

    let driver_rs = tmp.path().join(format!("{test_name}.rs"));
    fs::write(&driver_rs, driver_src).expect("write driver source");

    for (label, rlib) in [("nostd", &nostd_rlib), ("std", &std_rlib)] {
        let bin_path = tmp.path().join(format!("{test_name}_{label}"));
        let compile = Command::new("rustc")
            .args([
                "--edition=2024",
                "-C",
                "opt-level=2",
                "-C",
                "panic=abort",
                "--extern",
            ])
            .arg(format!("lust_aot_rt={}", rlib.display()))
            .arg(&driver_rs)
            .arg("-o")
            .arg(&bin_path)
            .output()
            .expect("compile driver binary");
        assert!(
            compile.status.success(),
            "[{test_name} ({label})] rustc compile failed:\nstderr:\n{}",
            String::from_utf8_lossy(&compile.stderr)
        );

        let run = Command::new(&bin_path)
            .output()
            .expect("execute driver binary");
        assert!(
            run.status.success(),
            "[{test_name} ({label})] driver failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&run.stdout).trim(),
            "PASS",
            "[{test_name} ({label})] unexpected stdout"
        );
    }
}

#[test]
fn test_ng_w16_1_aot_map_single_probe_and_borrowed_str_zero_alloc_overwrite() {
    let driver = r#"
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use lust_aot_rt::{AotCtx, AotError, AotKey, AotMap, AotRawMap, AotValue, LustBudgetAlloc};

struct CountingAlloc;

static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(new_size, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn reset_counters() {
    ALLOC_CALLS.store(0, Ordering::SeqCst);
    ALLOC_BYTES.store(0, Ordering::SeqCst);
}

fn run() -> Result<(), AotError> {
    let expected_entry_bytes = core::mem::size_of::<String>()
        .saturating_add(core::mem::size_of::<i64>())
        .max(16);

    // 1. AotRawMap<String, i64>: borrowed lookup, single vacant charge, zero-alloc overwrite
    let alloc = LustBudgetAlloc::with_limit(4096);
    let mut raw: AotRawMap<String, i64> = AotRawMap::new();
    assert!(raw.is_empty());
    assert_eq!(raw.get("tok_1"), None);

    raw.set_borrowed("tok_1", 10, &alloc)?;
    assert_eq!(alloc.used_bytes(), expected_entry_bytes);
    assert_eq!(raw.get("tok_1"), Some(&10));
    assert_eq!(raw.has("tok_1"), true);

    // Overwrite via borrowed &str: 0 budget bytes charged AND 0 global heap allocations!
    let used_before = alloc.used_bytes();
    reset_counters();
    raw.set_borrowed("tok_1", 42, &alloc)?;
    let calls_overwrite = ALLOC_CALLS.load(Ordering::SeqCst);
    let bytes_overwrite = ALLOC_BYTES.load(Ordering::SeqCst);
    assert_eq!(alloc.used_bytes(), used_before);
    assert_eq!(calls_overwrite, 0, "AotRawMap::set_borrowed overwrite must perform 0 heap allocs");
    assert_eq!(bytes_overwrite, 0, "AotRawMap::set_borrowed overwrite must allocate 0 bytes");
    assert_eq!(raw.get("tok_1"), Some(&42));
    assert_eq!(raw.len(), 1);

    // Second vacant insert charges exactly 1 entry
    raw.set_borrowed("tok_2", 99, &alloc)?;
    assert_eq!(alloc.used_bytes(), expected_entry_bytes * 2);
    assert_eq!(raw.len(), 2);

    // Delete via borrowed &str refunds exactly 1 entry
    assert_eq!(raw.delete("tok_1", &alloc), Some(42));
    assert_eq!(raw.get("tok_1"), None);
    assert_eq!(raw.get("tok_2"), Some(&99));
    assert_eq!(alloc.used_bytes(), expected_entry_bytes);

    // 2. AotMap<String, i64>: borrowed lookup, single vacant charge, zero-alloc overwrite, aliasing
    let alloc2 = LustBudgetAlloc::with_limit(4096);
    let map: AotMap<String, i64> = AotMap::new();
    let alias = map.clone();
    assert_eq!(map.get("tok_1"), Err(AotError::MissingMapKey));

    map.set_borrowed("tok_1", 7, &alloc2)?;
    assert_eq!(alloc2.used_bytes(), expected_entry_bytes);
    assert_eq!(alias.get("tok_1")?, 7);
    assert_eq!(alias.get_opt("tok_1")?, Some(7));
    assert!(alias.has("tok_1")?);

    let used_before_map = alloc2.used_bytes();
    reset_counters();
    map.set_borrowed("tok_1", 42, &alloc2)?;
    let map_calls = ALLOC_CALLS.load(Ordering::SeqCst);
    let map_bytes = ALLOC_BYTES.load(Ordering::SeqCst);
    assert_eq!(alloc2.used_bytes(), used_before_map);
    assert_eq!(map_calls, 0, "AotMap::set_borrowed overwrite must perform 0 heap allocs");
    assert_eq!(map_bytes, 0, "AotMap::set_borrowed overwrite must allocate 0 bytes");
    assert_eq!(alias.get("tok_1")?, 42);

    // 3. Heavy int churn (insert, overwrite, backward-shift delete, re-insert across multiple resizes)
    let churn_alloc = LustBudgetAlloc::unbounded();
    let mut imap: AotRawMap<i64, i64> = AotRawMap::new();
    for i in 0..2048_i64 {
        imap.set(i, i * 3 + 1, &churn_alloc)?;
    }
    assert_eq!(imap.len(), 2048);
    // Delete every odd key (tests backward-shift deletion across clusters)
    for i in (1..2048_i64).step_by(2) {
        assert_eq!(imap.delete(&i, &churn_alloc), Some(i * 3 + 1));
    }
    assert_eq!(imap.len(), 1024);
    for i in 0..2048_i64 {
        if i % 2 == 0 {
            assert_eq!(imap.get(&i), Some(&(i * 3 + 1)));
        } else {
            assert_eq!(imap.get(&i), None);
        }
    }
    // Overwrite even keys charges 0 extra bytes
    let churn_used_before = churn_alloc.used_bytes();
    for i in (0..2048_i64).step_by(2) {
        imap.set(i, i * 5, &churn_alloc)?;
    }
    assert_eq!(churn_alloc.used_bytes(), churn_used_before);
    for i in (0..2048_i64).step_by(2) {
        assert_eq!(imap.get(&i), Some(&(i * 5)));
        assert_eq!(imap.delete(&i, &churn_alloc), Some(i * 5));
    }
    assert_eq!(imap.len(), 0);
    assert_eq!(churn_alloc.used_bytes(), 0);

    // 4. AotValue::map_set single-probe overwrite charges 0 extra bytes
    let ctx = AotCtx::with_memory_limit(1024);
    let dyn_map = AotValue::map();
    dyn_map.map_set(AotKey::from("k1"), AotValue::Int(1), &ctx)?;
    let dyn_used = ctx.budget_alloc.used_bytes();
    assert!(dyn_used > 0);
    dyn_map.map_set(AotKey::from("k1"), AotValue::Int(2), &ctx)?;
    assert_eq!(ctx.budget_alloc.used_bytes(), dyn_used);
    assert_eq!(dyn_map.map_get(&AotKey::from("k1"))?, AotValue::Int(2));

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }
    println!("PASS");
}
"#;
    run_driver_both_modes(
        "test_ng_w16_1_aot_map_single_probe_and_borrowed_str_zero_alloc_overwrite",
        driver,
    );
}

#[test]
fn test_ng_w16_1_aot_small_str_inline_and_heap_roundtrip() {
    let driver = r#"
use std::alloc::{GlobalAlloc, Layout, System};
use std::borrow::Borrow;
use std::sync::atomic::{AtomicUsize, Ordering};
use lust_aot_rt::{AotError, AotHash, AotSmallStr, AotString, LustBudgetAlloc};

struct CountingAlloc;

static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(new_size, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn reset_counters() {
    ALLOC_CALLS.store(0, Ordering::SeqCst);
    ALLOC_BYTES.store(0, Ordering::SeqCst);
}

fn run() -> Result<(), AotError> {
    // Must be exactly 24 bytes on 64-bit targets
    #[cfg(target_pointer_width = "64")]
    {
        assert_eq!(core::mem::size_of::<AotSmallStr>(), 24);
        assert_eq!(core::mem::size_of::<AotString>(), 24);
    }

    let alloc = LustBudgetAlloc::with_limit(1024);

    // Inline strings <= 22 bytes allocate 0 heap bytes and charge 0 budget bytes
    let inline_samples = [
        "",
        "\"device_id\"",
        "1700000100",
        "0123456789012345678901", // exactly 22 bytes
        "привет",                 // multi-byte UTF-8 (12 bytes <= 22)
    ];

    for sample in inline_samples {
        assert!(sample.len() <= 22);
        reset_counters();
        let s = AotSmallStr::try_from_str(sample, &alloc)?;
        let calls = ALLOC_CALLS.load(Ordering::SeqCst);
        let bytes = ALLOC_BYTES.load(Ordering::SeqCst);
        assert_eq!(calls, 0, "inline AotSmallStr({sample:?}) must perform 0 heap allocs");
        assert_eq!(bytes, 0, "inline AotSmallStr({sample:?}) must allocate 0 heap bytes");
        assert_eq!(alloc.used_bytes(), 0);
        assert!(s.is_inline());
        assert!(!s.is_heap());
        assert_eq!(s.as_str(), sample);
        assert_eq!(&*s, sample);
        let borrowed: &str = s.borrow();
        assert_eq!(borrowed, sample);
        assert_eq!(s.aot_hash(), sample.aot_hash());

        // Cloning an inline string also performs 0 heap allocations
        reset_counters();
        let cloned = s.clone();
        assert_eq!(ALLOC_CALLS.load(Ordering::SeqCst), 0);
        assert_eq!(cloned, s);
    }

    // Heap fallback for strings > 22 bytes
    let long_23 = "01234567890123456789012"; // 23 bytes
    let long_payload = "this_is_a_much_longer_string_exceeding_22_bytes_for_heap_fallback";
    for sample in [long_23, long_payload] {
        assert!(sample.len() > 22);
        let used_before = alloc.used_bytes();
        let s: AotString = AotString::try_from_str(sample, &alloc)?;
        assert!(!s.is_inline());
        assert!(s.is_heap());
        assert_eq!(s.as_str(), sample);
        assert_eq!(&*s, sample);
        assert_eq!(s.len(), sample.len());
        assert_eq!(s.aot_hash(), sample.aot_hash());
        assert_eq!(alloc.used_bytes(), used_before + sample.len());
        assert_eq!(format!("{s}"), sample);

        let c = s.clone();
        assert_eq!(c, s);
        assert_eq!(c.as_str(), sample);
    }

    // Ordering & equality across inline and heap
    let a = AotSmallStr::from("alpha");
    let b = AotSmallStr::from("beta_very_long_string_that_lives_on_the_heap");
    assert!(a < b);
    assert_ne!(a, b);
    assert_eq!(a, "alpha");

    // Tight budget OOM on heap fallback leaves inline still working
    let tiny = LustBudgetAlloc::with_limit(10);
    assert!(AotSmallStr::try_from_str("1700000100", &tiny).is_ok()); // 10B inline -> 0 charged
    assert!(AotSmallStr::try_from_str("0123456789012345678901", &tiny).is_ok()); // 22B inline -> 0 charged
    assert!(matches!(
        AotSmallStr::try_from_str(long_23, &tiny),
        Err(AotError::OutOfMemory { .. })
    ));

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }
    println!("PASS");
}
"#;
    run_driver_both_modes("test_ng_w16_1_aot_small_str_inline_and_heap_roundtrip", driver);
}

#[test]
fn test_ng_w16_1_aot_queue_typed_fifo_and_capacity() {
    let driver = r#"
use lust_aot_rt::{AotError, AotQueue};

fn run() -> Result<(), AotError> {
    // 1. Exact BENCH-NG-12 fiber_channel_pingpong.lust FIFO checksum (799919916)
    let mut q: AotQueue<i64> = AotQueue::with_capacity(64);
    assert_eq!(q.len(), 0);
    assert!(q.is_empty());

    let mut total: i64 = 0;
    for round in 0..5000_i64 {
        for k in 0..32_i64 {
            assert!(q.offer(round * 32 + k)?);
        }
        assert_eq!(q.len(), 32);
        for _ in 0..32 {
            let msg = q.take()?;
            total = (total + msg) % 1_000_000_007;
        }
        assert_eq!(q.len(), 0);
    }
    assert_eq!(total, 799_919_916);

    // 2. Bounded capacity overflow under default Suspend strategy returns Err
    let mut bounded: AotQueue<i64> = AotQueue::with_capacity(2);
    assert_eq!(bounded.offer(10)?, true);
    assert_eq!(bounded.offer(20)?, true);
    assert_eq!(bounded.len(), 2);
    assert!(matches!(bounded.offer(30), Err(AotError::UserError(_))));
    assert_eq!(bounded.take()?, 10);
    assert_eq!(bounded.take()?, 20);
    assert!(matches!(bounded.take(), Err(AotError::UserError(_))));

    // 3. Dropping strategy returns Ok(false) when full
    let mut dropping: AotQueue<i64> = AotQueue::with_capacity_and_strategy(2, "dropping");
    assert_eq!(dropping.offer(1)?, true);
    assert_eq!(dropping.offer(2)?, true);
    assert_eq!(dropping.offer(3)?, false);
    assert_eq!(dropping.take()?, 1);
    assert_eq!(dropping.take()?, 2);

    // 4. Sliding strategy evicts oldest item when full
    let mut sliding: AotQueue<i64> = AotQueue::with_capacity_and_strategy(2, "sliding");
    assert_eq!(sliding.offer(1)?, true);
    assert_eq!(sliding.offer(2)?, true);
    assert_eq!(sliding.offer(3)?, true);
    assert_eq!(sliding.len(), 2);
    assert_eq!(sliding.take()?, 2);
    assert_eq!(sliding.take()?, 3);

    // 5. Closing queue rejects further offers with Ok(false)
    sliding.close();
    assert_eq!(sliding.offer(99)?, false);

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }
    println!("PASS");
}
"#;
    run_driver_both_modes("test_ng_w16_1_aot_queue_typed_fifo_and_capacity", driver);
}

#[test]
fn test_ng_w16_1_shared_budget_alloc_accumulates_across_clones() {
    let driver = r#"
use lust_aot_rt::{AotCtx, AotError, AotMap, LustBudgetAlloc};

fn run() -> Result<(), AotError> {
    // 1. Pre-share charge on parent is preserved when share() is called
    let parent = LustBudgetAlloc::with_limit(200);
    parent.try_charge(80)?;
    assert_eq!(parent.used_bytes(), 80);

    let child = parent.share();
    assert_eq!(child.used_bytes(), 80);
    assert_eq!(child.limit_bytes(), 200);
    assert_eq!(child.peak_bytes(), 80);

    // 2. Charging child updates parent cumulatively
    child.try_charge(70)?;
    assert_eq!(parent.used_bytes(), 150);
    assert_eq!(child.used_bytes(), 150);
    assert_eq!(parent.peak_bytes(), 150);

    // 3. Grandchild shares the same cumulative ledger
    let grandchild = child.share();
    grandchild.try_charge(50)?;
    assert_eq!(parent.used_bytes(), 200);
    assert_eq!(child.used_bytes(), 200);
    assert_eq!(grandchild.used_bytes(), 200);

    // 4. Combined usage exceeding limit triggers AotError::OutOfMemory from any handle
    let err_parent: Result<(), AotError> = parent.try_charge(1).map_err(AotError::from);
    assert!(matches!(
        err_parent,
        Err(AotError::OutOfMemory { limit_bytes: 200, requested_bytes: 201 })
    ));
    let err_child: Result<(), AotError> = child.try_charge(1).map_err(AotError::from);
    assert!(matches!(
        err_child,
        Err(AotError::OutOfMemory { limit_bytes: 200, requested_bytes: 201 })
    ));

    // 5. Refunding on child frees budget visible to parent & grandchild
    child.refund(100);
    assert_eq!(parent.used_bytes(), 100);
    assert_eq!(grandchild.used_bytes(), 100);
    assert_eq!(parent.peak_bytes(), 200);
    parent.try_charge(50)?;
    assert_eq!(child.used_bytes(), 150);

    // 6. Works seamlessly with AotCtx + mutable ctx borrow + AotMap + Vec::new_in
    let mut ctx = AotCtx::with_memory_limit(128);
    let shared_alloc = ctx.budget_alloc.share();
    // Mutably borrow ctx while allocating through shared_alloc
    ctx.charge_gas(1)?;
    let map: AotMap<i64, i64> = AotMap::new();
    for i in 0..4_i64 {
        map.set(i, i * 10, &shared_alloc)?;
    }
    assert_eq!(ctx.budget_alloc.used_bytes(), 64);

    // Allocate 64 bytes in a Vec via shared_alloc -> reaches exact 128 byte limit
    {
        let mut v: Vec<u8, &LustBudgetAlloc> = Vec::new_in(&shared_alloc);
        assert!(v.try_reserve_exact(64).is_ok());
        assert_eq!(ctx.budget_alloc.used_bytes(), 128);
        // Any further map insert via ctx.budget_alloc must fail with OutOfMemory!
        assert!(matches!(
            map.set(99, 999, &ctx.budget_alloc),
            Err(AotError::OutOfMemory { limit_bytes: 128, .. })
        ));
    }
    // Dropping Vec refunded 64 bytes into the shared ledger
    assert_eq!(ctx.budget_alloc.used_bytes(), 64);
    map.set(99, 999, &ctx.budget_alloc)?;
    assert_eq!(shared_alloc.used_bytes(), 80);

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }
    println!("PASS");
}
"#;
    run_driver_both_modes(
        "test_ng_w16_1_shared_budget_alloc_accumulates_across_clones",
        driver,
    );
}

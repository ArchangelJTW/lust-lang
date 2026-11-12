//! Example: Rx Stream Operators (filter, map, take) in Lust.
//! Demonstrates composable stream transformations where operators chain
//! lazily and preserve backpressure invariants across asynchronous stages.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use lust::rx::{BufferOptions, ExitStatus, Observable, Observer, RxScheduler};

fn main() {
    println!("--- [Rx Operators (Filter & Map) Example] ---");

    let scheduler = Arc::new(RxScheduler::new_virtual());
    let scope_id = 202;

    let received = Arc::new(Mutex::new(Vec::new()));
    let completed = Arc::new(AtomicBool::new(false));

    let r_clone = received.clone();
    let c_clone = completed.clone();

    let observer = Observer::new(
        move |item: i64| {
            println!("  [Transformed Stream] Emitted: {}", item);
            r_clone.lock().unwrap().push(item);
            Ok(())
        },
        move || {
            println!("  [Transformed Stream] Finished");
            c_clone.store(true, Ordering::SeqCst);
        },
        move |err| {
            eprintln!("  [Transformed Stream] Error: {:?}", err);
        },
    );

    // 1. Source stream emitting numbers 1..=10
    let source: Observable<i64, String> = Observable::from_iter(1..=10);

    // 2. Chain filter (keep even numbers) and map (multiply by 10)
    let transformed = source
        .filter(|&x| x % 2 == 0)
        .map(|x| x * 10);

    // 3. Subscribe and run scheduler
    let subscription = transformed
        .subscribe(
            observer,
            scope_id,
            BufferOptions::default_bounded(),
            scheduler.clone(),
        )
        .expect("Subscription must succeed");

    scheduler.run_until_quiescent(200);

    let completion = subscription
        .join(Some(Duration::from_millis(100)))
        .expect("Join must complete");

    assert_eq!(completion.exit, ExitStatus::Completed);
    // Even numbers: 2, 4, 6, 8, 10 -> multiplied by 10: 20, 40, 60, 80, 100
    let result = received.lock().unwrap().clone();
    assert_eq!(result, vec![20, 40, 60, 80, 100]);
    assert!(completed.load(Ordering::SeqCst));

    println!("✓ Transformed items: {:?}", result);
    println!("All Rx operator assertions passed!");
}

//! Example: Reactive Streams (Rx) Observable Foundation in Lust.
//! Demonstrates cold Observable streams, Observer subscriptions, virtual scheduling
//! to quiescence, and joining subscriptions with guaranteed zero lingering callbacks.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use lust::rx::{BufferOptions, ExitStatus, Observable, Observer, RxScheduler};

fn main() {
    println!("--- [Rx Observables Foundation Example] ---");

    // 1. Initialize virtual RxScheduler and scope ID
    let scheduler = Arc::new(RxScheduler::new_virtual());
    let scope_id = 101;

    let received = Arc::new(Mutex::new(Vec::new()));
    let completed = Arc::new(AtomicBool::new(false));

    let r_clone = received.clone();
    let c_clone = completed.clone();

    // 2. Build Observer with on_next, on_complete, on_error callbacks
    let observer = Observer::new(
        move |item: i64| {
            println!("  [Observer] Received item: {}", item);
            r_clone.lock().unwrap().push(item);
            Ok(())
        },
        move || {
            println!("  [Observer] Stream completed successfully");
            c_clone.store(true, Ordering::SeqCst);
        },
        move |err| {
            eprintln!("  [Observer] Stream error: {:?}", err);
        },
    );

    // 3. Create cold Observable from iterable
    let stream: Observable<i64, String> = Observable::from_iter(vec![10, 20, 30, 40]);

    // 4. Subscribe within scope using default bounded buffer
    let subscription = stream
        .subscribe(
            observer,
            scope_id,
            BufferOptions::default_bounded(),
            scheduler.clone(),
        )
        .expect("Subscription to Observable must succeed");

    // 5. Run scheduler until quiescent
    let steps = scheduler.run_until_quiescent(100);
    println!("✓ Scheduler reached quiescence in {} steps", steps);

    // 6. Join subscription and verify completion
    let completion = subscription
        .join(Some(Duration::from_millis(100)))
        .expect("Subscription join must succeed");

    assert_eq!(completion.exit, ExitStatus::Completed);
    assert_eq!(completion.items_delivered, 4);
    assert_eq!(*received.lock().unwrap(), vec![10, 20, 30, 40]);
    assert!(completed.load(Ordering::SeqCst));

    println!("All Rx observable assertions passed!");
}

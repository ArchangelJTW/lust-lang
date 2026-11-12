//! Example: Backpressure and Bounded Buffers in Rx.
//! Demonstrates configuring dual-limit bounded queues (max items, max bytes,
//! low watermarks) and overflow strategies (Fail, DropOldest) under RxScheduler.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use lust::rx::{
    BoundedQueue, BufferOptions, EnqueueResult, ExitStatus, Observable, Observer, OverflowStrategy,
    RxScheduler,
};

fn main() {
    println!("--- [Rx Backpressure & Bounded Buffers Example] ---");

    let scheduler = Arc::new(RxScheduler::new_virtual());
    let scope_id = 303;

    // 1. Create a custom bounded BufferOptions configuration
    let custom_options = BufferOptions::new(
        16,                      // max 16 items
        1024,                    // max 1024 bytes
        4,                       // low watermark: 4 items
        256,                     // low watermark: 256 bytes
        OverflowStrategy::Fail,  // fail-closed on overflow
    )
    .expect("Valid BufferOptions configuration");

    println!(
        "✓ Configured Bounded Buffer: max_items={}, max_bytes={}, overflow={:?}",
        custom_options.max_items, custom_options.max_bytes, custom_options.overflow
    );

    // 2. Default bounded options verification
    let default_options = BufferOptions::default_bounded();
    assert_eq!(default_options.max_items, 64);
    assert_eq!(default_options.max_bytes, 65536);
    println!("✓ Default bounded options verified (64 items / 64KB)");

    // 3. Overflow test with teeth: small buffer (max_items: 5)
    // 3a. OverflowStrategy::Fail: push 10 items to trigger overflow and assert OverflowError
    let fail_options = BufferOptions::new(5, 1024, 2, 256, OverflowStrategy::Fail)
        .expect("Valid BufferOptions");
    let mut fail_queue: BoundedQueue<i64> = BoundedQueue::new(fail_options);

    for i in 0..5 {
        assert_eq!(fail_queue.try_enqueue(i), EnqueueResult::Enqueued);
    }
    assert_eq!(fail_queue.len(), 5);

    // Items 5..10 must fail with OverflowError
    for i in 5..10 {
        assert_eq!(fail_queue.try_enqueue(i), EnqueueResult::OverflowError);
    }
    assert_eq!(fail_queue.len(), 5);
    println!("✓ OverflowStrategy::Fail successfully rejected 5 overflow items with OverflowError");

    // 3b. OverflowStrategy::DropOldest: push 10 items to trigger replacement of oldest items
    let drop_options = BufferOptions::new(5, 1024, 2, 256, OverflowStrategy::DropOldest)
        .expect("Valid BufferOptions");
    let mut drop_queue: BoundedQueue<i64> = BoundedQueue::new(drop_options);

    for i in 0..5 {
        assert_eq!(drop_queue.try_enqueue(i), EnqueueResult::Enqueued);
    }
    for i in 5..10 {
        let expected_oldest = i - 5;
        assert_eq!(
            drop_queue.try_enqueue(i),
            EnqueueResult::ReplacedOldest(expected_oldest)
        );
    }
    assert_eq!(drop_queue.len(), 5);
    assert_eq!(drop_queue.dropped_count(), 5);
    // Queue should now contain 5, 6, 7, 8, 9
    for expected in 5..10 {
        assert_eq!(drop_queue.dequeue(), Some(expected));
    }
    assert_eq!(drop_queue.dequeue(), None);
    println!("✓ OverflowStrategy::DropOldest successfully dropped 5 oldest items and retained newest");

    // 4. Run a bounded stream subscription under RxScheduler
    let items_delivered = Arc::new(AtomicUsize::new(0));
    let del_clone = items_delivered.clone();

    let observer = Observer::new(
        move |_item: i64| {
            del_clone.fetch_add(1, Ordering::SeqCst);
            Ok(())
        },
        || {},
        |_| {},
    );

    let stream: Observable<i64, String> = Observable::from_iter(0..10);
    let sub = stream
        .subscribe(observer, scope_id, custom_options, scheduler.clone())
        .expect("Subscription must succeed");

    scheduler.run_until_quiescent(100);

    let completion = sub
        .join(Some(Duration::from_millis(100)))
        .expect("Join must succeed");

    assert_eq!(completion.exit, ExitStatus::Completed);
    assert_eq!(items_delivered.load(Ordering::SeqCst), 10);
    println!(
        "✓ Successfully processed {} items within bounded buffer constraints",
        completion.items_delivered
    );

    println!("All Rx backpressure assertions passed!");
}

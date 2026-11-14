//! Bounded Queue and Backpressure options with dual limits (items and bytes).

use std::collections::VecDeque;

/// Strategy to handle queue overflow when high watermark is reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowStrategy {
    /// Return an explicit BackpressureOverflow error immediately.
    Fail,
    /// Cooperatively suspend the producer until occupancy drains below low watermark.
    Suspend,
    /// Discard the incoming item, retaining existing queue content.
    DropNewest,
    /// Discard the oldest queued item to make room for the new item.
    DropOldest,
}

/// Configuration options defining bounded backpressure limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferOptions {
    pub max_items: usize,
    pub max_bytes: usize,
    pub low_items: usize,
    pub low_bytes: usize,
    pub overflow: OverflowStrategy,
}

impl BufferOptions {
    /// Create and validate a BufferOptions configuration.
    pub fn new(
        max_items: usize,
        max_bytes: usize,
        low_items: usize,
        low_bytes: usize,
        overflow: OverflowStrategy,
    ) -> Result<Self, &'static str> {
        if max_items == 0 {
            return Err("max_items must be greater than zero");
        }
        if max_bytes == 0 {
            return Err("max_bytes must be greater than zero");
        }
        if low_items >= max_items {
            return Err("low_items must be strictly less than max_items");
        }
        if low_bytes >= max_bytes {
            return Err("low_bytes must be strictly less than max_bytes");
        }
        Ok(Self {
            max_items,
            max_bytes,
            low_items,
            low_bytes,
            overflow,
        })
    }

    /// Default bounded buffer with 64 items and 64KB max memory.
    pub fn default_bounded() -> Self {
        Self {
            max_items: 64,
            max_bytes: 65536,
            low_items: 16,
            low_bytes: 16384,
            overflow: OverflowStrategy::Fail,
        }
    }
}

/// Trait to report memory footprint charged against buffer quotas.
pub trait MeasurableItem {
    fn charged_bytes(&self) -> usize;
}

impl MeasurableItem for i64 {
    fn charged_bytes(&self) -> usize {
        8
    }
}

impl MeasurableItem for i32 {
    fn charged_bytes(&self) -> usize {
        4
    }
}

impl MeasurableItem for u64 {
    fn charged_bytes(&self) -> usize {
        8
    }
}

impl MeasurableItem for usize {
    fn charged_bytes(&self) -> usize {
        std::mem::size_of::<usize>()
    }
}

impl MeasurableItem for f64 {
    fn charged_bytes(&self) -> usize {
        8
    }
}

impl MeasurableItem for bool {
    fn charged_bytes(&self) -> usize {
        1
    }
}

impl MeasurableItem for &'static str {
    fn charged_bytes(&self) -> usize {
        self.len()
    }
}

impl MeasurableItem for String {
    fn charged_bytes(&self) -> usize {
        std::mem::size_of::<String>() + self.len()
    }
}

impl<T: MeasurableItem> MeasurableItem for Vec<T> {
    fn charged_bytes(&self) -> usize {
        std::mem::size_of::<Vec<T>>() + self.iter().map(|x| x.charged_bytes()).sum::<usize>()
    }
}

/// Outcome of attempting to enqueue an item into a BoundedQueue.
#[derive(Debug, PartialEq, Eq)]
pub enum EnqueueResult<T> {
    Enqueued,
    MustSuspend,
    DroppedNewest,
    ReplacedOldest(T),
    OverflowError,
    ItemTooLarge,
}

/// A bounded FIFO queue enforcing dual limits (item count and memory size).
pub struct BoundedQueue<T: MeasurableItem> {
    options: BufferOptions,
    items: VecDeque<T>,
    current_charged_bytes: usize,
    dropped_count: usize,
}

impl<T: MeasurableItem> BoundedQueue<T> {
    pub fn new(options: BufferOptions) -> Self {
        Self {
            options,
            items: VecDeque::with_capacity(options.max_items.min(1024)),
            current_charged_bytes: 0,
            dropped_count: 0,
        }
    }

    /// Current count of queued items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Current total charged bytes in the queue.
    pub fn charged_bytes(&self) -> usize {
        self.current_charged_bytes
    }

    /// Total count of dropped items (via DropNewest or DropOldest).
    pub fn dropped_count(&self) -> usize {
        self.dropped_count
    }

    /// Check if the queue occupancy is at or below the low watermark.
    pub fn is_below_low_watermark(&self) -> bool {
        self.items.len() <= self.options.low_items
            && self.current_charged_bytes <= self.options.low_bytes
    }

    /// Check if queue would overflow if we add 1 item of `item_bytes`.
    pub fn would_overflow(&self, item_bytes: usize) -> bool {
        self.items.len() >= self.options.max_items
            || self
                .current_charged_bytes
                .checked_add(item_bytes)
                .map_or(true, |sum| sum > self.options.max_bytes)
    }

    /// Try to enqueue an item according to the configured overflow strategy.
    pub fn try_enqueue(&mut self, item: T) -> EnqueueResult<T> {
        let item_bytes = item.charged_bytes();

        // Reject item if it alone exceeds total allowed capacity
        if item_bytes > self.options.max_bytes {
            return EnqueueResult::ItemTooLarge;
        }

        if self.would_overflow(item_bytes) {
            match self.options.overflow {
                OverflowStrategy::Fail => EnqueueResult::OverflowError,
                OverflowStrategy::Suspend => EnqueueResult::MustSuspend,
                OverflowStrategy::DropNewest => {
                    self.dropped_count += 1;
                    EnqueueResult::DroppedNewest
                }
                OverflowStrategy::DropOldest => {
                    if let Some(old) = self.items.pop_front() {
                        self.current_charged_bytes = self
                            .current_charged_bytes
                            .saturating_sub(old.charged_bytes());
                        self.dropped_count += 1;
                        self.current_charged_bytes += item_bytes;
                        self.items.push_back(item);
                        EnqueueResult::ReplacedOldest(old)
                    } else {
                        // Empty queue but item too large
                        EnqueueResult::ItemTooLarge
                    }
                }
            }
        } else {
            self.current_charged_bytes += item_bytes;
            self.items.push_back(item);
            EnqueueResult::Enqueued
        }
    }

    /// Dequeue the next item.
    pub fn dequeue(&mut self) -> Option<T> {
        if let Some(item) = self.items.pop_front() {
            self.current_charged_bytes = self
                .current_charged_bytes
                .saturating_sub(item.charged_bytes());
            Some(item)
        } else {
            None
        }
    }

    /// Clear all queued items and reset charged bytes.
    pub fn clear(&mut self) {
        self.items.clear();
        self.current_charged_bytes = 0;
    }
}

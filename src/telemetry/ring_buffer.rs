use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use super::sink::{SpanStatus, TelemetrySink};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompactEvent {
    Empty,
    Counter { seq: u64, metric_hash: u32, delta: u64 },
    Gauge { seq: u64, metric_hash: u32, value_bits: u64 },
    Histogram { seq: u64, metric_hash: u32, value_bits: u64 },
    SpanEnter { seq: u64, span_id: u64, name_hash: u32 },
    SpanExit { seq: u64, span_id: u64, status: u8 },
}

pub struct RingBufferTelemetrySink<const N: usize> {
    buffer: [core::cell::UnsafeCell<CompactEvent>; N],
    write_idx: AtomicUsize,
    seq: AtomicU64,
    next_span_id: AtomicU64,
    total_events: AtomicU64,
}

unsafe impl<const N: usize> Sync for RingBufferTelemetrySink<N> {}
unsafe impl<const N: usize> Send for RingBufferTelemetrySink<N> {}

impl<const N: usize> RingBufferTelemetrySink<N> {
    pub const fn new() -> Self {
        const EMPTY_CELL: core::cell::UnsafeCell<CompactEvent> =
            core::cell::UnsafeCell::new(CompactEvent::Empty);
        Self {
            buffer: [EMPTY_CELL; N],
            write_idx: AtomicUsize::new(0),
            seq: AtomicU64::new(0),
            next_span_id: AtomicU64::new(1),
            total_events: AtomicU64::new(0),
        }
    }

    #[inline]
    fn hash_name(name: &str) -> u32 {
        let mut hash: u32 = 2166136261;
        for byte in name.as_bytes() {
            hash ^= *byte as u32;
            hash = hash.wrapping_mul(16777619);
        }
        hash
    }

    #[inline]
    fn push_event(&self, event: CompactEvent) {
        if N == 0 {
            return;
        }
        let idx = self.write_idx.fetch_add(1, Ordering::Relaxed) % N;
        self.total_events.fetch_add(1, Ordering::Relaxed);
        unsafe {
            *self.buffer[idx].get() = event;
        }
    }

    pub fn total_events(&self) -> u64 {
        self.total_events.load(Ordering::Relaxed)
    }

    pub fn capacity(&self) -> usize {
        N
    }

    pub fn read_latest(&self) -> Option<CompactEvent> {
        let total = self.total_events.load(Ordering::Relaxed);
        if total == 0 || N == 0 {
            return None;
        }
        let last_idx = (self.write_idx.load(Ordering::Relaxed).wrapping_sub(1)) % N;
        unsafe { Some(*self.buffer[last_idx].get()) }
    }
}

impl<const N: usize> core::fmt::Debug for RingBufferTelemetrySink<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RingBufferTelemetrySink")
            .field("capacity", &N)
            .field("total_events", &self.total_events())
            .finish()
    }
}

impl<const N: usize> Default for RingBufferTelemetrySink<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> TelemetrySink for RingBufferTelemetrySink<N> {
    fn counter_add(&self, name: &str, delta: u64, _labels: &[(&str, &str)]) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let metric_hash = Self::hash_name(name);
        self.push_event(CompactEvent::Counter {
            seq,
            metric_hash,
            delta,
        });
    }

    fn gauge_set(&self, name: &str, value: f64, _labels: &[(&str, &str)]) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let metric_hash = Self::hash_name(name);
        self.push_event(CompactEvent::Gauge {
            seq,
            metric_hash,
            value_bits: value.to_bits(),
        });
    }

    fn histogram_record(&self, name: &str, value: f64, _labels: &[(&str, &str)]) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let metric_hash = Self::hash_name(name);
        self.push_event(CompactEvent::Histogram {
            seq,
            metric_hash,
            value_bits: value.to_bits(),
        });
    }

    fn span_enter(&self, name: &str, _labels: &[(&str, &str)]) -> u64 {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let span_id = self.next_span_id.fetch_add(1, Ordering::Relaxed);
        let name_hash = Self::hash_name(name);
        self.push_event(CompactEvent::SpanEnter {
            seq,
            span_id,
            name_hash,
        });
        span_id
    }

    fn span_exit(&self, span_id: u64, status: SpanStatus) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        self.push_event(CompactEvent::SpanExit {
            seq,
            span_id,
            status: status as u8,
        });
    }
}

use alloc::{
    string::{String, ToString},
    vec::Vec,
};
use core::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "std")]
use std::sync::Mutex;

use super::sink::{SpanStatus, TelemetrySink};

#[derive(Debug, Clone, PartialEq)]
pub enum TelemetryEvent {
    Counter {
        seq: u64,
        name: String,
        delta: u64,
        labels: Vec<(String, String)>,
    },
    Gauge {
        seq: u64,
        name: String,
        value: f64,
        labels: Vec<(String, String)>,
    },
    Histogram {
        seq: u64,
        name: String,
        value: f64,
        labels: Vec<(String, String)>,
    },
    SpanEnter {
        seq: u64,
        span_id: u64,
        name: String,
        labels: Vec<(String, String)>,
    },
    SpanExit {
        seq: u64,
        span_id: u64,
        status: SpanStatus,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowPolicy {
    DropNewest,
    RingBuffer,
}

#[derive(Debug)]
pub struct MockTelemetrySink {
    #[cfg(feature = "std")]
    events: Mutex<Vec<TelemetryEvent>>,
    seq: AtomicU64,
    next_span_id: AtomicU64,
    dropped_events: AtomicU64,
    max_events: usize,
    overflow_policy: OverflowPolicy,
}

impl MockTelemetrySink {
    pub fn new() -> Self {
        Self::with_capacity(16_384, OverflowPolicy::DropNewest)
    }

    pub fn with_capacity(max_events: usize, overflow_policy: OverflowPolicy) -> Self {
        Self {
            #[cfg(feature = "std")]
            events: Mutex::new(Vec::with_capacity(max_events.min(1024))),
            seq: AtomicU64::new(0),
            next_span_id: AtomicU64::new(1),
            dropped_events: AtomicU64::new(0),
            max_events,
            overflow_policy,
        }
    }

    #[cfg(feature = "std")]
    pub fn events(&self) -> Vec<TelemetryEvent> {
        let guard = self.events.lock().unwrap();
        guard.clone()
    }

    pub fn dropped_events(&self) -> u64 {
        self.dropped_events.load(Ordering::Relaxed)
    }

    #[cfg(feature = "std")]
    pub fn clear(&self) {
        let mut guard = self.events.lock().unwrap();
        guard.clear();
        self.dropped_events.store(0, Ordering::Relaxed);
    }

    #[cfg(feature = "std")]
    pub fn counter_total(&self, name: &str) -> u64 {
        let guard = self.events.lock().unwrap();
        guard
            .iter()
            .filter_map(|ev| match ev {
                TelemetryEvent::Counter { name: n, delta, .. } if n == name => Some(*delta),
                _ => None,
            })
            .sum()
    }

    #[cfg(feature = "std")]
    pub fn gauge_latest(&self, name: &str) -> Option<f64> {
        let guard = self.events.lock().unwrap();
        guard.iter().rev().find_map(|ev| match ev {
            TelemetryEvent::Gauge { name: n, value, .. } if n == name => Some(*value),
            _ => None,
        })
    }

    #[cfg(feature = "std")]
    pub fn spans_named(&self, name: &str) -> Vec<TelemetryEvent> {
        let guard = self.events.lock().unwrap();
        guard
            .iter()
            .filter(|ev| match ev {
                TelemetryEvent::SpanEnter { name: n, .. } => n == name,
                _ => false,
            })
            .cloned()
            .collect()
    }

    #[cfg(feature = "std")]
    fn record_event(&self, event: TelemetryEvent) {
        let mut guard = self.events.lock().unwrap();
        if guard.len() >= self.max_events {
            match self.overflow_policy {
                OverflowPolicy::DropNewest => {
                    self.dropped_events.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                OverflowPolicy::RingBuffer => {
                    if !guard.is_empty() {
                        guard.remove(0);
                        self.dropped_events.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
        guard.push(event);
    }
}

impl Default for MockTelemetrySink {
    fn default() -> Self {
        Self::new()
    }
}

impl TelemetrySink for MockTelemetrySink {
    fn counter_add(&self, name: &str, delta: u64, labels: &[(&str, &str)]) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        #[cfg(feature = "std")]
        {
            let labels = labels
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            self.record_event(TelemetryEvent::Counter {
                seq,
                name: name.to_string(),
                delta,
                labels,
            });
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (seq, name, delta, labels);
        }
    }

    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        #[cfg(feature = "std")]
        {
            let labels = labels
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            self.record_event(TelemetryEvent::Gauge {
                seq,
                name: name.to_string(),
                value,
                labels,
            });
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (seq, name, value, labels);
        }
    }

    fn histogram_record(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        #[cfg(feature = "std")]
        {
            let labels = labels
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            self.record_event(TelemetryEvent::Histogram {
                seq,
                name: name.to_string(),
                value,
                labels,
            });
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (seq, name, value, labels);
        }
    }

    fn span_enter(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let span_id = self.next_span_id.fetch_add(1, Ordering::Relaxed);
        #[cfg(feature = "std")]
        {
            let labels = labels
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            self.record_event(TelemetryEvent::SpanEnter {
                seq,
                span_id,
                name: name.to_string(),
                labels,
            });
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (seq, name, labels);
        }
        span_id
    }

    fn span_exit(&self, span_id: u64, status: SpanStatus) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        #[cfg(feature = "std")]
        {
            self.record_event(TelemetryEvent::SpanExit {
                seq,
                span_id,
                status,
            });
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (seq, span_id, status);
        }
    }
}

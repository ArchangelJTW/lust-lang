use alloc::{
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(feature = "std")]
use std::cell::RefCell;
#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(feature = "std")]
use std::sync::Mutex;

use super::sink::{SpanStatus, TelemetrySink};

#[cfg(feature = "std")]
thread_local! {
    static ACTIVE_SPAN_STACK: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

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
        parent_span_id: Option<u64>,
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
pub struct FlushRecord {
    pub seq: u64,
    pub failed: bool,
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
    #[cfg(feature = "std")]
    span_parents: Mutex<HashMap<u64, Option<u64>>>,
    #[cfg(feature = "std")]
    span_children: Mutex<HashMap<u64, Vec<u64>>>,
    #[cfg(feature = "std")]
    flushes: Mutex<Vec<FlushRecord>>,
    seq: AtomicU64,
    next_span_id: AtomicU64,
    dropped_events: AtomicU64,
    flush_count: AtomicU64,
    flush_failures: AtomicU64,
    fail_on_flush: AtomicBool,
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
            #[cfg(feature = "std")]
            span_parents: Mutex::new(HashMap::new()),
            #[cfg(feature = "std")]
            span_children: Mutex::new(HashMap::new()),
            #[cfg(feature = "std")]
            flushes: Mutex::new(Vec::new()),
            seq: AtomicU64::new(0),
            next_span_id: AtomicU64::new(1),
            dropped_events: AtomicU64::new(0),
            flush_count: AtomicU64::new(0),
            flush_failures: AtomicU64::new(0),
            fail_on_flush: AtomicBool::new(false),
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

    pub fn flush_count(&self) -> u64 {
        self.flush_count.load(Ordering::Relaxed)
    }

    pub fn flush_failures(&self) -> u64 {
        self.flush_failures.load(Ordering::Relaxed)
    }

    pub fn set_fail_on_flush(&self, fail: bool) {
        self.fail_on_flush.store(fail, Ordering::Relaxed);
    }

    #[cfg(feature = "std")]
    pub fn flush_records(&self) -> Vec<FlushRecord> {
        self.flushes.lock().unwrap().clone()
    }

    pub fn max_events(&self) -> usize {
        self.max_events
    }

    pub fn overflow_policy(&self) -> OverflowPolicy {
        self.overflow_policy
    }

    #[cfg(feature = "std")]
    pub fn clear(&self) {
        let mut guard = self.events.lock().unwrap();
        guard.clear();
        self.span_parents.lock().unwrap().clear();
        self.span_children.lock().unwrap().clear();
        self.flushes.lock().unwrap().clear();
        self.dropped_events.store(0, Ordering::Relaxed);
        self.flush_count.store(0, Ordering::Relaxed);
        self.flush_failures.store(0, Ordering::Relaxed);
        ACTIVE_SPAN_STACK.with(|stack| stack.borrow_mut().clear());
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
    pub fn parent_of_span(&self, span_id: u64) -> Option<u64> {
        self.span_parents.lock().unwrap().get(&span_id).copied().flatten()
    }

    #[cfg(feature = "std")]
    pub fn children_of_span(&self, span_id: u64) -> Vec<u64> {
        self.span_children
            .lock()
            .unwrap()
            .get(&span_id)
            .cloned()
            .unwrap_or_default()
    }

    #[cfg(feature = "std")]
    pub fn events_for_vm(&self, vm_id: &str) -> Vec<TelemetryEvent> {
        self.events_by_label("vm_id", vm_id)
    }

    #[cfg(feature = "std")]
    pub fn events_by_label(&self, key: &str, val: &str) -> Vec<TelemetryEvent> {
        let guard = self.events.lock().unwrap();
        guard
            .iter()
            .filter(|ev| match ev {
                TelemetryEvent::Counter { labels, .. }
                | TelemetryEvent::Gauge { labels, .. }
                | TelemetryEvent::Histogram { labels, .. }
                | TelemetryEvent::SpanEnter { labels, .. } => {
                    labels.iter().any(|(k, v)| k == key && v == val)
                }
                _ => false,
            })
            .cloned()
            .collect()
    }

    #[cfg(feature = "std")]
    pub fn scoped(self: &Arc<Self>, scope_labels: &[(&str, &str)]) -> ScopedMockSink {
        ScopedMockSink {
            inner: Arc::clone(self),
            scope_labels: scope_labels
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    #[cfg(feature = "std")]
    pub fn escaping_violations(&self) -> Vec<String> {
        let guard = self.events.lock().unwrap();
        let mut violations = Vec::new();
        for ev in guard.iter() {
            let labels = match ev {
                TelemetryEvent::Counter { labels, .. }
                | TelemetryEvent::Gauge { labels, .. }
                | TelemetryEvent::Histogram { labels, .. }
                | TelemetryEvent::SpanEnter { labels, .. } => labels,
                _ => continue,
            };
            for (k, v) in labels {
                if k.contains('\n') || k.contains('"') || k.contains('\\') {
                    violations.push(alloc::format!("invalid label key: {k}"));
                }
                if v.contains('\n') || v.contains('\r') {
                    violations.push(alloc::format!("unescaped newline in label value for {k}"));
                }
            }
        }
        violations
    }

    #[inline]
    pub fn validate_metric_name(name: &str) -> Result<(), &'static str> {
        if name.is_empty() {
            return Err("metric name cannot be empty");
        }
        let first = name.chars().next().unwrap();
        if !first.is_ascii_alphabetic() && first != '_' && first != ':' {
            return Err("metric name must start with [a-zA-Z_:]");
        }
        for c in name.chars() {
            if !c.is_ascii_alphanumeric() && c != '_' && c != ':' && c != '.' {
                return Err("metric name contains invalid character");
            }
        }
        Ok(())
    }

    #[inline]
    pub fn validate_label(key: &str, val: &str) -> Result<(), &'static str> {
        if key.is_empty() {
            return Err("label key cannot be empty");
        }
        let first = key.chars().next().unwrap();
        if !first.is_ascii_alphabetic() && first != '_' {
            return Err("label key must start with [a-zA-Z_]");
        }
        for c in key.chars() {
            if !c.is_ascii_alphanumeric() && c != '_' {
                return Err("label key contains invalid character");
            }
        }
        if val.len() > 1024 {
            return Err("label value exceeds maximum allowed length");
        }
        Ok(())
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
            let explicit_parent = labels.iter().find_map(|(k, v)| {
                if *k == "parent_span_id" || *k == "parent_id" {
                    v.parse::<u64>().ok()
                } else {
                    None
                }
            });
            let parent_span_id = explicit_parent.or_else(|| {
                ACTIVE_SPAN_STACK.with(|stack| stack.borrow().last().copied())
            });

            ACTIVE_SPAN_STACK.with(|stack| stack.borrow_mut().push(span_id));

            self.span_parents.lock().unwrap().insert(span_id, parent_span_id);
            if let Some(parent) = parent_span_id {
                self.span_children
                    .lock()
                    .unwrap()
                    .entry(parent)
                    .or_default()
                    .push(span_id);
            }

            let labels = labels
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            self.record_event(TelemetryEvent::SpanEnter {
                seq,
                span_id,
                parent_span_id,
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
            ACTIVE_SPAN_STACK.with(|stack| {
                let mut s = stack.borrow_mut();
                if s.last() == Some(&span_id) {
                    s.pop();
                } else if let Some(pos) = s.iter().rposition(|&id| id == span_id) {
                    s.remove(pos);
                }
            });

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

    fn flush(&self) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        self.flush_count.fetch_add(1, Ordering::Relaxed);
        #[cfg(feature = "std")]
        {
            let failed = self.fail_on_flush.load(Ordering::Relaxed);
            if failed {
                self.flush_failures.fetch_add(1, Ordering::Relaxed);
            }
            self.flushes.lock().unwrap().push(FlushRecord { seq, failed });
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = seq;
        }
    }
}

/// Scoped wrapper for `MockTelemetrySink` enabling concurrent VM and tenant isolation
#[derive(Debug, Clone)]
pub struct ScopedMockSink {
    inner: Arc<MockTelemetrySink>,
    scope_labels: Vec<(String, String)>,
}

impl ScopedMockSink {
    pub fn new(inner: Arc<MockTelemetrySink>, scope_labels: &[(&str, &str)]) -> Self {
        Self {
            inner,
            scope_labels: scope_labels
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    pub fn inner(&self) -> &Arc<MockTelemetrySink> {
        &self.inner
    }

    fn merge_labels<'a>(&'a self, labels: &[(&'a str, &'a str)]) -> Vec<(&'a str, &'a str)> {
        let mut merged = Vec::with_capacity(self.scope_labels.len() + labels.len());
        for (k, v) in &self.scope_labels {
            merged.push((k.as_str(), v.as_str()));
        }
        for (k, v) in labels {
            merged.push((*k, *v));
        }
        merged
    }
}

impl TelemetrySink for ScopedMockSink {
    fn counter_add(&self, name: &str, delta: u64, labels: &[(&str, &str)]) {
        let merged = self.merge_labels(labels);
        self.inner.counter_add(name, delta, &merged);
    }

    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        let merged = self.merge_labels(labels);
        self.inner.gauge_set(name, value, &merged);
    }

    fn histogram_record(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        let merged = self.merge_labels(labels);
        self.inner.histogram_record(name, value, &merged);
    }

    fn span_enter(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
        let merged = self.merge_labels(labels);
        self.inner.span_enter(name, &merged)
    }

    fn span_exit(&self, span_id: u64, status: SpanStatus) {
        self.inner.span_exit(span_id, status);
    }

    fn flush(&self) {
        self.inner.flush();
    }
}

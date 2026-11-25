//! Tracing subscriber and exporter adapter for Lust Runtime.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

thread_local! {
    static ACTIVE_SPAN_STACK: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanStatus {
    Success,
    Fail,
    Die,
    Interrupt,
    Ok,
    Error,
}

impl SpanStatus {
    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Success | Self::Ok => "success",
            Self::Fail | Self::Error => "fail",
            Self::Die => "die",
            Self::Interrupt => "interrupt",
        }
    }
}

pub trait TelemetrySink: Send + Sync + core::fmt::Debug {
    fn counter_add(&self, name: &str, delta: u64, labels: &[(&str, &str)]);
    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]);
    fn histogram_record(&self, name: &str, value: f64, labels: &[(&str, &str)]);
    fn span_enter(&self, name: &str, labels: &[(&str, &str)]) -> u64;
    fn span_exit(&self, span_id: u64, status: SpanStatus);
    fn flush(&self) {}
}

#[derive(Debug, Clone, PartialEq)]
pub enum TracingRecord {
    Counter {
        name: String,
        delta: u64,
        labels: Vec<(String, String)>,
    },
    Gauge {
        name: String,
        value: f64,
        labels: Vec<(String, String)>,
    },
    Histogram {
        name: String,
        value: f64,
        labels: Vec<(String, String)>,
    },
    SpanEnter {
        span_id: u64,
        parent_span_id: Option<u64>,
        name: String,
        labels: Vec<(String, String)>,
    },
    SpanExit {
        span_id: u64,
        status: SpanStatus,
    },
}

#[derive(Debug)]
pub struct TracingSink {
    records: Mutex<Vec<TracingRecord>>,
    vm_id: Option<String>,
    next_span_id: AtomicU64,
    flush_count: AtomicU64,
    flush_failures: AtomicU64,
    fail_on_flush: AtomicBool,
}

impl TracingSink {
    pub fn new() -> Self {
        Self {
            records: Mutex::new(Vec::new()),
            vm_id: None,
            next_span_id: AtomicU64::new(1),
            flush_count: AtomicU64::new(0),
            flush_failures: AtomicU64::new(0),
            fail_on_flush: AtomicBool::new(false),
        }
    }

    pub fn with_vm_id(mut self, vm_id: &str) -> Self {
        self.vm_id = Some(vm_id.to_string());
        self
    }

    pub fn set_fail_on_flush(&self, fail: bool) {
        self.fail_on_flush.store(fail, Ordering::Relaxed);
    }

    pub fn flush_count(&self) -> u64 {
        self.flush_count.load(Ordering::Relaxed)
    }

    pub fn flush_failures(&self) -> u64 {
        self.flush_failures.load(Ordering::Relaxed)
    }

    pub fn clear(&self) {
        self.records.lock().unwrap().clear();
        self.flush_count.store(0, Ordering::Relaxed);
        self.flush_failures.store(0, Ordering::Relaxed);
        ACTIVE_SPAN_STACK.with(|s| s.borrow_mut().clear());
    }

    pub fn records(&self) -> Vec<TracingRecord> {
        self.records.lock().unwrap().clone()
    }

    pub fn spans_named(&self, name: &str) -> Vec<TracingRecord> {
        let guard = self.records.lock().unwrap();
        guard
            .iter()
            .filter(|r| match r {
                TracingRecord::SpanEnter { name: n, .. } => n == name,
                _ => false,
            })
            .cloned()
            .collect()
    }

    fn merge_labels(&self, labels: &[(&str, &str)]) -> Vec<(String, String)> {
        let mut res = Vec::new();
        if let Some(ref vm_id) = self.vm_id {
            res.push(("vm_id".to_string(), vm_id.clone()));
        }
        for (k, v) in labels {
            res.push((k.to_string(), v.to_string()));
        }
        res.sort();
        res
    }

    pub fn scoped(self: &Arc<Self>, scope_labels: &[(&str, &str)]) -> ScopedTracingSink {
        ScopedTracingSink {
            inner: Arc::clone(self),
            scope_labels: scope_labels
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }
}

impl Default for TracingSink {
    fn default() -> Self {
        Self::new()
    }
}

impl TelemetrySink for TracingSink {
    fn counter_add(&self, name: &str, delta: u64, labels: &[(&str, &str)]) {
        let merged = self.merge_labels(labels);
        #[cfg(feature = "tracing-crate")]
        {
            tracing::info!(
                target: "lust::telemetry",
                name = name,
                delta = delta,
                labels = ?merged,
                "counter_add"
            );
        }
        self.records.lock().unwrap().push(TracingRecord::Counter {
            name: name.to_string(),
            delta,
            labels: merged,
        });
    }

    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        let merged = self.merge_labels(labels);
        #[cfg(feature = "tracing-crate")]
        {
            tracing::debug!(
                target: "lust::telemetry",
                name = name,
                value = value,
                labels = ?merged,
                "gauge_set"
            );
        }
        self.records.lock().unwrap().push(TracingRecord::Gauge {
            name: name.to_string(),
            value,
            labels: merged,
        });
    }

    fn histogram_record(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        let merged = self.merge_labels(labels);
        #[cfg(feature = "tracing-crate")]
        {
            tracing::debug!(
                target: "lust::telemetry",
                name = name,
                value = value,
                labels = ?merged,
                "histogram_record"
            );
        }
        self.records.lock().unwrap().push(TracingRecord::Histogram {
            name: name.to_string(),
            value,
            labels: merged,
        });
    }

    fn span_enter(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
        let span_id = self.next_span_id.fetch_add(1, Ordering::Relaxed);
        let explicit_parent = labels.iter().find_map(|(k, v)| {
            if *k == "parent_span_id" || *k == "parent_id" {
                v.parse::<u64>().ok()
            } else {
                None
            }
        });
        let parent_span_id = explicit_parent.or_else(|| {
            ACTIVE_SPAN_STACK.with(|s| s.borrow().last().copied())
        });
        ACTIVE_SPAN_STACK.with(|s| s.borrow_mut().push(span_id));

        let merged = self.merge_labels(labels);

        #[cfg(feature = "tracing-crate")]
        {
            let _span = tracing::info_span!(
                target: "lust::telemetry",
                "lust_span",
                span_id = span_id,
                parent_span_id = ?parent_span_id,
                name = name
            );
        }

        self.records.lock().unwrap().push(TracingRecord::SpanEnter {
            span_id,
            parent_span_id,
            name: name.to_string(),
            labels: merged,
        });
        span_id
    }

    fn span_exit(&self, span_id: u64, status: SpanStatus) {
        ACTIVE_SPAN_STACK.with(|s| {
            let mut stack = s.borrow_mut();
            if stack.last() == Some(&span_id) {
                stack.pop();
            } else if let Some(pos) = stack.iter().rposition(|&id| id == span_id) {
                stack.remove(pos);
            }
        });

        #[cfg(feature = "tracing-crate")]
        {
            tracing::info!(
                target: "lust::telemetry",
                span_id = span_id,
                status = status.as_str(),
                "span_exit"
            );
        }

        self.records.lock().unwrap().push(TracingRecord::SpanExit {
            span_id,
            status,
        });
    }

    fn flush(&self) {
        self.flush_count.fetch_add(1, Ordering::Relaxed);
        if self.fail_on_flush.load(Ordering::Relaxed) {
            self.flush_failures.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScopedTracingSink {
    inner: Arc<TracingSink>,
    scope_labels: Vec<(String, String)>,
}

impl ScopedTracingSink {
    fn merge<'a>(&'a self, labels: &[(&'a str, &'a str)]) -> Vec<(&'a str, &'a str)> {
        let mut res = Vec::with_capacity(self.scope_labels.len() + labels.len());
        for (k, v) in &self.scope_labels {
            res.push((k.as_str(), v.as_str()));
        }
        res.extend_from_slice(labels);
        res
    }
}

impl TelemetrySink for ScopedTracingSink {
    fn counter_add(&self, name: &str, delta: u64, labels: &[(&str, &str)]) {
        let merged = self.merge(labels);
        self.inner.counter_add(name, delta, &merged);
    }

    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        let merged = self.merge(labels);
        self.inner.gauge_set(name, value, &merged);
    }

    fn histogram_record(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        let merged = self.merge(labels);
        self.inner.histogram_record(name, value, &merged);
    }

    fn span_enter(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
        let merged = self.merge(labels);
        self.inner.span_enter(name, &merged)
    }

    fn span_exit(&self, span_id: u64, status: SpanStatus) {
        self.inner.span_exit(span_id, status);
    }

    fn flush(&self) {
        self.inner.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracing_sink_records_and_nested_spans() {
        let sink = TracingSink::new();
        sink.counter_add("gas", 25, &[("source", "jit")]);
        sink.gauge_set("mem", 1024.0, &[]);
        sink.histogram_record("latency", 0.015, &[]);

        let root_id = sink.span_enter("root_task", &[]);
        let child_id = sink.span_enter("sub_step", &[]);
        sink.span_exit(child_id, SpanStatus::Success);
        sink.span_exit(root_id, SpanStatus::Ok);

        let records = sink.records();
        assert_eq!(records.len(), 7);

        match &records[4] {
            TracingRecord::SpanEnter {
                span_id,
                parent_span_id,
                name,
                ..
            } => {
                assert_eq!(*span_id, child_id);
                assert_eq!(*parent_span_id, Some(root_id));
                assert_eq!(name, "sub_step");
            }
            other => panic!("Expected SpanEnter, got {:?}", other),
        }
    }

    #[test]
    fn test_flush_and_failure() {
        let sink = TracingSink::new();
        sink.flush();
        assert_eq!(sink.flush_count(), 1);
        sink.set_fail_on_flush(true);
        sink.flush();
        assert_eq!(sink.flush_count(), 2);
        assert_eq!(sink.flush_failures(), 1);
    }

    #[test]
    fn test_scoped_vm_isolation() {
        let sink = Arc::new(TracingSink::new());
        let vm1 = sink.scoped(&[("vm_id", "vm-1")]);
        let vm2 = sink.scoped(&[("vm_id", "vm-2")]);

        vm1.counter_add("ops", 1, &[]);
        vm2.counter_add("ops", 2, &[]);

        let records = sink.records();
        assert_eq!(records.len(), 2);
        match &records[0] {
            TracingRecord::Counter { labels, .. } => {
                assert!(labels.iter().any(|(k, v)| k == "vm_id" && v == "vm-1"));
            }
            _ => panic!("Expected counter"),
        }
        match &records[1] {
            TracingRecord::Counter { labels, .. } => {
                assert!(labels.iter().any(|(k, v)| k == "vm_id" && v == "vm-2"));
            }
            _ => panic!("Expected counter"),
        }
    }
}

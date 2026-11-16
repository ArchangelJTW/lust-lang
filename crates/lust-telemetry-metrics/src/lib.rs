//! Zero-dependency in-memory metrics collector and exporter for Lust Runtime.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

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
pub struct HistogramSnapshot {
    pub count: u64,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
    pub buckets: Vec<(f64, u64)>, // (upper_bound, cumulative_count)
}

#[derive(Debug, Clone)]
pub struct CardinalityLimits {
    pub max_unique_names: usize,
    pub max_label_pairs: usize,
    pub max_label_val_len: usize,
    pub max_unique_series: usize,
}

impl Default for CardinalityLimits {
    fn default() -> Self {
        Self {
            max_unique_names: 256,
            max_label_pairs: 8,
            max_label_val_len: 64,
            max_unique_series: 4096,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MetricKey {
    pub name: String,
    pub labels: Vec<(String, String)>,
}

impl MetricKey {
    pub fn new(name: &str, labels: &[(&str, &str)]) -> Self {
        let mut sorted_labels: Vec<(String, String)> = labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        sorted_labels.sort();
        Self {
            name: name.to_string(),
            labels: sorted_labels,
        }
    }
}

#[derive(Debug, Default)]
struct HistogramAccumulator {
    count: u64,
    sum: f64,
    min: f64,
    max: f64,
    buckets: BTreeMap<u64, u64>, // key: f64::to_bits
}

impl HistogramAccumulator {
    fn new(default_bounds: &[f64]) -> Self {
        let mut buckets = BTreeMap::new();
        for &b in default_bounds {
            buckets.insert(b.to_bits(), 0);
        }
        buckets.insert(f64::INFINITY.to_bits(), 0);
        Self {
            count: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            buckets,
        }
    }

    fn record(&mut self, val: f64) {
        self.count += 1;
        self.sum += val;
        if val < self.min {
            self.min = val;
        }
        if val > self.max {
            self.max = val;
        }
        for (bound_bits, count) in self.buckets.iter_mut() {
            let bound = f64::from_bits(*bound_bits);
            if val <= bound {
                *count += 1;
            }
        }
    }

    fn snapshot(&self) -> HistogramSnapshot {
        let mut buckets = Vec::with_capacity(self.buckets.len());
        for (bits, count) in &self.buckets {
            buckets.push((f64::from_bits(*bits), *count));
        }
        HistogramSnapshot {
            count: self.count,
            sum: self.sum,
            min: if self.count == 0 { 0.0 } else { self.min },
            max: if self.count == 0 { 0.0 } else { self.max },
            buckets,
        }
    }
}

#[derive(Debug)]
struct RegistryInner {
    counters: HashMap<MetricKey, u64>,
    gauges: HashMap<MetricKey, f64>,
    histograms: HashMap<MetricKey, HistogramAccumulator>,
    known_metric_names: HashSet<String>,
    known_series: HashSet<String>,
}

#[derive(Debug)]
pub struct MetricsSink {
    inner: Mutex<RegistryInner>,
    limits: CardinalityLimits,
    default_histogram_bounds: Vec<f64>,
    vm_id: Option<String>,
    next_span_id: AtomicU64,
    flush_count: AtomicU64,
    flush_failures: AtomicU64,
    fail_on_flush: AtomicBool,
    dropped_series: AtomicU64,
}

impl MetricsSink {
    pub fn new() -> Self {
        Self::with_limits(CardinalityLimits::default())
    }

    pub fn with_limits(limits: CardinalityLimits) -> Self {
        Self {
            inner: Mutex::new(RegistryInner {
                counters: HashMap::new(),
                gauges: HashMap::new(),
                histograms: HashMap::new(),
                known_metric_names: HashSet::new(),
                known_series: HashSet::new(),
            }),
            limits,
            default_histogram_bounds: vec![
                0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ],
            vm_id: None,
            next_span_id: AtomicU64::new(1),
            flush_count: AtomicU64::new(0),
            flush_failures: AtomicU64::new(0),
            fail_on_flush: AtomicBool::new(false),
            dropped_series: AtomicU64::new(0),
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

    pub fn dropped_series_total(&self) -> u64 {
        self.dropped_series.load(Ordering::Relaxed)
    }

    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.counters.clear();
        inner.gauges.clear();
        inner.histograms.clear();
        inner.known_metric_names.clear();
        inner.known_series.clear();
        self.flush_count.store(0, Ordering::Relaxed);
        self.flush_failures.store(0, Ordering::Relaxed);
        self.dropped_series.store(0, Ordering::Relaxed);
    }

    fn sanitize_and_check(
        &self,
        inner: &mut RegistryInner,
        name: &str,
        labels: &[(&str, &str)],
    ) -> Option<MetricKey> {
        // 1. Metric name bound
        if !inner.known_metric_names.contains(name) {
            if inner.known_metric_names.len() >= self.limits.max_unique_names {
                self.dropped_series.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            inner.known_metric_names.insert(name.to_string());
        }

        // 2. Bound label pairs
        let mut merged_labels = Vec::new();
        if let Some(ref vm_id) = self.vm_id {
            merged_labels.push(("vm_id", vm_id.as_str()));
        }
        merged_labels.extend_from_slice(labels);

        let effective = if merged_labels.len() > self.limits.max_label_pairs {
            &merged_labels[..self.limits.max_label_pairs]
        } else {
            &merged_labels[..]
        };

        // 3. Truncate label values
        let mut sanitized: Vec<(String, String)> = effective
            .iter()
            .map(|(k, v)| {
                let val = if v.len() > self.limits.max_label_val_len {
                    v[..self.limits.max_label_val_len].to_string()
                } else {
                    v.to_string()
                };
                (k.to_string(), val)
            })
            .collect();
        sanitized.sort();

        // 4. Series check
        let mut series_key = String::with_capacity(64);
        series_key.push_str(name);
        for (k, v) in &sanitized {
            series_key.push(';');
            series_key.push_str(k);
            series_key.push('=');
            series_key.push_str(v);
        }

        if !inner.known_series.contains(&series_key) {
            if inner.known_series.len() >= self.limits.max_unique_series {
                self.dropped_series.fetch_add(1, Ordering::Relaxed);
                return Some(MetricKey {
                    name: name.to_string(),
                    labels: vec![("overflow".to_string(), "true".to_string())],
                });
            }
            inner.known_series.insert(series_key);
        }

        Some(MetricKey {
            name: name.to_string(),
            labels: sanitized,
        })
    }

    pub fn counter(&self, name: &str, labels: &[(&str, &str)]) -> Option<u64> {
        let key = MetricKey::new(name, labels);
        let inner = self.inner.lock().unwrap();
        inner.counters.get(&key).copied()
    }

    pub fn counter_total(&self, name: &str) -> u64 {
        let inner = self.inner.lock().unwrap();
        inner
            .counters
            .iter()
            .filter(|(k, _)| k.name == name)
            .map(|(_, val)| *val)
            .sum()
    }

    pub fn gauge(&self, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
        let key = MetricKey::new(name, labels);
        let inner = self.inner.lock().unwrap();
        inner.gauges.get(&key).copied()
    }

    pub fn histogram(&self, name: &str, labels: &[(&str, &str)]) -> Option<HistogramSnapshot> {
        let key = MetricKey::new(name, labels);
        let inner = self.inner.lock().unwrap();
        inner.histograms.get(&key).map(|h| h.snapshot())
    }

    pub fn scoped(self: &Arc<Self>, scope_labels: &[(&str, &str)]) -> ScopedMetricsSink {
        ScopedMetricsSink {
            inner: Arc::clone(self),
            scope_labels: scope_labels
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let inner = self.inner.lock().unwrap();
        let mut counters = BTreeMap::new();
        for (k, v) in &inner.counters {
            counters.insert(k.clone(), *v);
        }
        let mut gauges = BTreeMap::new();
        for (k, v) in &inner.gauges {
            gauges.insert(k.clone(), *v);
        }
        let mut histograms = BTreeMap::new();
        for (k, h) in &inner.histograms {
            histograms.insert(k.clone(), h.snapshot());
        }
        MetricsSnapshot {
            counters,
            gauges,
            histograms,
        }
    }

    pub fn render_json(&self) -> String {
        let snap = self.snapshot();
        let mut out = String::from("{\n  \"counters\": {\n");
        let mut first = true;
        for (k, v) in &snap.counters {
            if !first {
                out.push_str(",\n");
            }
            first = false;
            out.push_str(&format!(
                "    \"{}\": {{ \"value\": {}, \"labels\": {:?} }}",
                k.name, v, k.labels
            ));
        }
        out.push_str("\n  },\n  \"gauges\": {\n");
        first = true;
        for (k, v) in &snap.gauges {
            if !first {
                out.push_str(",\n");
            }
            first = false;
            out.push_str(&format!(
                "    \"{}\": {{ \"value\": {}, \"labels\": {:?} }}",
                k.name, v, k.labels
            ));
        }
        out.push_str("\n  },\n  \"histograms\": {\n");
        first = true;
        for (k, h) in &snap.histograms {
            if !first {
                out.push_str(",\n");
            }
            first = false;
            out.push_str(&format!(
                "    \"{}\": {{ \"count\": {}, \"sum\": {}, \"min\": {}, \"max\": {} }}",
                k.name, h.count, h.sum, h.min, h.max
            ));
        }
        out.push_str("\n  }\n}");
        out
    }

    pub fn render_line_format(&self) -> String {
        let snap = self.snapshot();
        let mut out = String::new();
        for (k, v) in &snap.counters {
            out.push_str(&format!("counter {} {} {:?}\n", k.name, v, k.labels));
        }
        for (k, v) in &snap.gauges {
            out.push_str(&format!("gauge {} {} {:?}\n", k.name, v, k.labels));
        }
        for (k, h) in &snap.histograms {
            out.push_str(&format!(
                "histogram {} count={} sum={} min={} max={}\n",
                k.name, h.count, h.sum, h.min, h.max
            ));
        }
        out
    }
}

impl Default for MetricsSink {
    fn default() -> Self {
        Self::new()
    }
}

impl TelemetrySink for MetricsSink {
    fn counter_add(&self, name: &str, delta: u64, labels: &[(&str, &str)]) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(key) = self.sanitize_and_check(&mut inner, name, labels) {
            *inner.counters.entry(key).or_insert(0) += delta;
        }
    }

    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(key) = self.sanitize_and_check(&mut inner, name, labels) {
            inner.gauges.insert(key, value);
        }
    }

    fn histogram_record(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(key) = self.sanitize_and_check(&mut inner, name, labels) {
            let bounds = &self.default_histogram_bounds;
            inner
                .histograms
                .entry(key)
                .or_insert_with(|| HistogramAccumulator::new(bounds))
                .record(value);
        }
    }

    fn span_enter(&self, _name: &str, _labels: &[(&str, &str)]) -> u64 {
        self.next_span_id.fetch_add(1, Ordering::Relaxed)
    }

    fn span_exit(&self, _span_id: u64, _status: SpanStatus) {}

    fn flush(&self) {
        self.flush_count.fetch_add(1, Ordering::Relaxed);
        if self.fail_on_flush.load(Ordering::Relaxed) {
            self.flush_failures.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetricsSnapshot {
    pub counters: BTreeMap<MetricKey, u64>,
    pub gauges: BTreeMap<MetricKey, f64>,
    pub histograms: BTreeMap<MetricKey, HistogramSnapshot>,
}

#[derive(Debug, Clone)]
pub struct ScopedMetricsSink {
    inner: Arc<MetricsSink>,
    scope_labels: Vec<(String, String)>,
}

impl ScopedMetricsSink {
    fn merge<'a>(&'a self, labels: &[(&'a str, &'a str)]) -> Vec<(&'a str, &'a str)> {
        let mut res = Vec::with_capacity(self.scope_labels.len() + labels.len());
        for (k, v) in &self.scope_labels {
            res.push((k.as_str(), v.as_str()));
        }
        res.extend_from_slice(labels);
        res
    }
}

impl TelemetrySink for ScopedMetricsSink {
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
    fn test_counter_and_gauge_recording() {
        let sink = MetricsSink::new();
        sink.counter_add("gas.used", 100, &[("env", "prod")]);
        sink.counter_add("gas.used", 50, &[("env", "prod")]);
        assert_eq!(sink.counter("gas.used", &[("env", "prod")]), Some(150));
        assert_eq!(sink.counter_total("gas.used"), 150);

        sink.gauge_set("memory.bytes", 4096.0, &[("host", "node1")]);
        assert_eq!(sink.gauge("memory.bytes", &[("host", "node1")]), Some(4096.0));
    }

    #[test]
    fn test_histogram_distribution() {
        let sink = MetricsSink::new();
        sink.histogram_record("latency_sec", 0.003, &[]);
        sink.histogram_record("latency_sec", 0.04, &[]);
        sink.histogram_record("latency_sec", 1.5, &[]);

        let snap = sink.histogram("latency_sec", &[]).unwrap();
        assert_eq!(snap.count, 3);
        assert!((snap.sum - 1.543).abs() < 1e-6);
        assert!((snap.min - 0.003).abs() < 1e-6);
        assert!((snap.max - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_cardinality_and_overflow_folding() {
        let limits = CardinalityLimits {
            max_unique_names: 5,
            max_label_pairs: 2,
            max_label_val_len: 10,
            max_unique_series: 3,
        };
        let sink = MetricsSink::with_limits(limits);
        sink.counter_add("reqs", 1, &[("user", "user-1")]);
        sink.counter_add("reqs", 1, &[("user", "user-2")]);
        sink.counter_add("reqs", 1, &[("user", "user-3")]);
        // 4th series exceeds max_unique_series (3) -> folds to overflow = "true"
        sink.counter_add("reqs", 1, &[("user", "user-4")]);

        assert_eq!(sink.dropped_series_total(), 1);
        assert_eq!(sink.counter("reqs", &[("overflow", "true")]), Some(1));
    }

    #[test]
    fn test_flush_and_failure() {
        let sink = MetricsSink::new();
        sink.flush();
        assert_eq!(sink.flush_count(), 1);
        assert_eq!(sink.flush_failures(), 0);

        sink.set_fail_on_flush(true);
        sink.flush();
        assert_eq!(sink.flush_count(), 2);
        assert_eq!(sink.flush_failures(), 1);
    }

    #[test]
    fn test_scoped_vm_isolation() {
        let sink = Arc::new(MetricsSink::new());
        let vm1 = sink.scoped(&[("vm_id", "vm-1")]);
        let vm2 = sink.scoped(&[("vm_id", "vm-2")]);

        vm1.counter_add("work.done", 10, &[]);
        vm2.counter_add("work.done", 25, &[]);

        assert_eq!(sink.counter("work.done", &[("vm_id", "vm-1")]), Some(10));
        assert_eq!(sink.counter("work.done", &[("vm_id", "vm-2")]), Some(25));
        assert_eq!(sink.counter_total("work.done"), 35);
    }
}

//! Zero-dependency OpenMetrics / Prometheus text exposition exporter for Lust Runtime.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Indicates whether the optional upstream `prometheus` crate adapter is compiled in.
#[cfg(feature = "upstream")]
pub const UPSTREAM_ENABLED: bool = true;

/// Indicates whether the optional upstream `prometheus` crate adapter is compiled in.
#[cfg(not(feature = "upstream"))]
pub const UPSTREAM_ENABLED: bool = false;

#[cfg(feature = "upstream")]
pub mod upstream;

#[cfg(feature = "upstream")]
pub use upstream::{ScopedUpstreamPrometheusSink, UpstreamPrometheusSink};

/// Zero-sized, zero-allocation no-op [`TelemetrySink`] for disabled-telemetry controls.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct NoopTelemetrySink;

impl TelemetrySink for NoopTelemetrySink {
    #[inline(always)]
    fn counter_add(&self, _name: &str, _delta: u64, _labels: &[(&str, &str)]) {}
    #[inline(always)]
    fn gauge_set(&self, _name: &str, _value: f64, _labels: &[(&str, &str)]) {}
    #[inline(always)]
    fn histogram_record(&self, _name: &str, _value: f64, _labels: &[(&str, &str)]) {}
    #[inline(always)]
    fn span_enter(&self, _name: &str, _labels: &[(&str, &str)]) -> u64 {
        0
    }
    #[inline(always)]
    fn span_exit(&self, _span_id: u64, _status: SpanStatus) {}
    #[inline(always)]
    fn flush(&self) {}
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
pub struct PromKey {
    pub name: String,
    pub labels: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrometheusHistogramSnapshot {
    pub count: u64,
    pub sum: f64,
    pub buckets: Vec<(f64, u64)>, // (le, count)
}

#[derive(Debug, Default)]
struct HistogramData {
    count: u64,
    sum: f64,
    buckets: BTreeMap<u64, u64>, // le::to_bits -> cumulative count
}

impl HistogramData {
    fn new(bounds: &[f64]) -> Self {
        let mut buckets = BTreeMap::new();
        for &b in bounds {
            buckets.insert(b.to_bits(), 0);
        }
        buckets.insert(f64::INFINITY.to_bits(), 0);
        Self {
            count: 0,
            sum: 0.0,
            buckets,
        }
    }

    fn record(&mut self, val: f64) {
        self.count += 1;
        self.sum += val;
        for (bound_bits, count) in self.buckets.iter_mut() {
            let bound = f64::from_bits(*bound_bits);
            if val <= bound {
                *count += 1;
            }
        }
    }

    fn snapshot(&self) -> PrometheusHistogramSnapshot {
        let mut buckets = Vec::with_capacity(self.buckets.len());
        for (bits, count) in &self.buckets {
            buckets.push((f64::from_bits(*bits), *count));
        }
        PrometheusHistogramSnapshot {
            count: self.count,
            sum: self.sum,
            buckets,
        }
    }
}

#[derive(Debug, Default)]
struct PrometheusRegistry {
    counters: HashMap<PromKey, u64>,
    gauges: HashMap<PromKey, f64>,
    histograms: HashMap<PromKey, HistogramData>,
    known_metric_names: HashSet<String>,
    known_series: HashSet<String>,
}

#[derive(Debug)]
pub struct PrometheusSink {
    inner: Mutex<PrometheusRegistry>,
    limits: CardinalityLimits,
    default_histogram_bounds: Vec<f64>,
    vm_id: Option<String>,
    next_span_id: AtomicU64,
    flush_count: AtomicU64,
    flush_failures: AtomicU64,
    fail_on_flush: AtomicBool,
    dropped_series: AtomicU64,
}

impl PrometheusSink {
    pub fn new() -> Self {
        Self::with_limits(CardinalityLimits::default())
    }

    pub fn with_limits(limits: CardinalityLimits) -> Self {
        Self {
            inner: Mutex::new(PrometheusRegistry::default()),
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
        inner: &mut PrometheusRegistry,
        name: &str,
        labels: &[(&str, &str)],
    ) -> Option<PromKey> {
        let sanitized_name = sanitize_metric_name(name);

        // 1. Check metric name count limit
        if !inner.known_metric_names.contains(&sanitized_name) {
            if inner.known_metric_names.len() >= self.limits.max_unique_names {
                self.dropped_series.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            inner.known_metric_names.insert(sanitized_name.clone());
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

        // 3. Truncate & sanitize labels
        let mut sanitized: Vec<(String, String)> = effective
            .iter()
            .map(|(k, v)| {
                let key = sanitize_label_key(k);
                let val = if v.len() > self.limits.max_label_val_len {
                    v[..self.limits.max_label_val_len].to_string()
                } else {
                    v.to_string()
                };
                (key, val)
            })
            .collect();
        sanitized.sort();

        // 4. Series check
        let mut series_key = String::with_capacity(64);
        series_key.push_str(&sanitized_name);
        for (k, v) in &sanitized {
            series_key.push(';');
            series_key.push_str(k);
            series_key.push('=');
            series_key.push_str(v);
        }

        if !inner.known_series.contains(&series_key) {
            if inner.known_series.len() >= self.limits.max_unique_series {
                self.dropped_series.fetch_add(1, Ordering::Relaxed);
                return Some(PromKey {
                    name: sanitized_name,
                    labels: vec![("overflow".to_string(), "true".to_string())],
                });
            }
            inner.known_series.insert(series_key);
        }

        Some(PromKey {
            name: sanitized_name,
            labels: sanitized,
        })
    }

    pub fn counter(&self, name: &str, labels: &[(&str, &str)]) -> Option<u64> {
        let sanitized_name = sanitize_metric_name(name);
        let mut sorted_labels: Vec<(String, String)> = labels
            .iter()
            .map(|(k, v)| (sanitize_label_key(k), v.to_string()))
            .collect();
        sorted_labels.sort();
        let key = PromKey {
            name: sanitized_name,
            labels: sorted_labels,
        };
        self.inner.lock().unwrap().counters.get(&key).copied()
    }

    pub fn counter_total(&self, name: &str) -> u64 {
        let sanitized = sanitize_metric_name(name);
        let inner = self.inner.lock().unwrap();
        inner
            .counters
            .iter()
            .filter(|(k, _)| k.name == sanitized)
            .map(|(_, val)| *val)
            .sum()
    }

    pub fn gauge(&self, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
        let sanitized_name = sanitize_metric_name(name);
        let mut sorted_labels: Vec<(String, String)> = labels
            .iter()
            .map(|(k, v)| (sanitize_label_key(k), v.to_string()))
            .collect();
        sorted_labels.sort();
        let key = PromKey {
            name: sanitized_name,
            labels: sorted_labels,
        };
        self.inner.lock().unwrap().gauges.get(&key).copied()
    }

    pub fn histogram(&self, name: &str, labels: &[(&str, &str)]) -> Option<PrometheusHistogramSnapshot> {
        let sanitized_name = sanitize_metric_name(name);
        let mut sorted_labels: Vec<(String, String)> = labels
            .iter()
            .map(|(k, v)| (sanitize_label_key(k), v.to_string()))
            .collect();
        sorted_labels.sort();
        let key = PromKey {
            name: sanitized_name,
            labels: sorted_labels,
        };
        self.inner.lock().unwrap().histograms.get(&key).map(|h| h.snapshot())
    }

    pub fn scoped(self: &Arc<Self>, scope_labels: &[(&str, &str)]) -> ScopedPrometheusSink {
        ScopedPrometheusSink {
            inner: Arc::clone(self),
            scope_labels: scope_labels
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    /// Renders standard OpenMetrics exposition format with `# EOF\n` trailer
    pub fn render_openmetrics(&self) -> String {
        let inner = self.inner.lock().unwrap();
        let mut out = String::new();

        // Group by metric name
        let mut counter_groups: BTreeMap<String, Vec<(&PromKey, u64)>> = BTreeMap::new();
        for (k, v) in &inner.counters {
            counter_groups.entry(k.name.clone()).or_default().push((k, *v));
        }

        let mut gauge_groups: BTreeMap<String, Vec<(&PromKey, f64)>> = BTreeMap::new();
        for (k, v) in &inner.gauges {
            gauge_groups.entry(k.name.clone()).or_default().push((k, *v));
        }

        let mut hist_groups: BTreeMap<String, Vec<(&PromKey, &HistogramData)>> = BTreeMap::new();
        for (k, v) in &inner.histograms {
            hist_groups.entry(k.name.clone()).or_default().push((k, v));
        }

        // Render counters
        for (name, entries) in counter_groups {
            let metric_name = if name.ends_with("_total") {
                name.clone()
            } else {
                format!("{name}_total")
            };
            out.push_str(&format!("# HELP {metric_name} Total count of {name}\n"));
            out.push_str(&format!("# TYPE {metric_name} counter\n"));
            for (key, val) in entries {
                let labels_str = format_labels(&key.labels);
                out.push_str(&format!("{metric_name}{labels_str} {val}\n"));
            }
        }

        // Render gauges
        for (name, entries) in gauge_groups {
            out.push_str(&format!("# HELP {name} Current value of {name}\n"));
            out.push_str(&format!("# TYPE {name} gauge\n"));
            for (key, val) in entries {
                let labels_str = format_labels(&key.labels);
                out.push_str(&format!("{name}{labels_str} {val}\n"));
            }
        }

        // Render histograms
        for (name, entries) in hist_groups {
            out.push_str(&format!("# HELP {name} Histogram distribution of {name}\n"));
            out.push_str(&format!("# TYPE {name} histogram\n"));
            for (key, data) in entries {
                let snap = data.snapshot();
                for (bound, count) in &snap.buckets {
                    let mut bucket_labels = key.labels.clone();
                    let le_str = if bound.is_infinite() {
                        "+Inf".to_string()
                    } else {
                        bound.to_string()
                    };
                    bucket_labels.push(("le".to_string(), le_str));
                    let labels_str = format_labels(&bucket_labels);
                    out.push_str(&format!("{name}_bucket{labels_str} {count}\n"));
                }
                let sum_labels = format_labels(&key.labels);
                out.push_str(&format!("{name}_sum{sum_labels} {}\n", snap.sum));
                out.push_str(&format!("{name}_count{sum_labels} {}\n", snap.count));
            }
        }

        out.push_str("# EOF\n");
        out
    }

    /// Renders Prometheus text format (without mandatory OpenMetrics `# EOF`)
    pub fn render_prometheus(&self) -> String {
        let om = self.render_openmetrics();
        om.strip_suffix("# EOF\n").unwrap_or(&om).to_string()
    }

    pub fn scrape(&self) -> String {
        self.render_openmetrics()
    }
}

impl Default for PrometheusSink {
    fn default() -> Self {
        Self::new()
    }
}

impl TelemetrySink for PrometheusSink {
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
                .or_insert_with(|| HistogramData::new(bounds))
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

pub fn sanitize_metric_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for (i, c) in name.chars().enumerate() {
        if i == 0 {
            if c.is_ascii_alphabetic() || c == '_' || c == ':' {
                out.push(c);
            } else {
                out.push('_');
            }
        } else if c.is_ascii_alphanumeric() || c == '_' || c == ':' {
            out.push(c);
        } else {
            out.push('_'); // replace dots and dashes with underscore
        }
    }
    out
}

pub fn sanitize_label_key(key: &str) -> String {
    let mut out = String::with_capacity(key.len());
    for (i, c) in key.chars().enumerate() {
        if i == 0 {
            if c.is_ascii_alphabetic() || c == '_' {
                out.push(c);
            } else {
                out.push('_');
            }
        } else if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    out
}

pub fn escape_label_value(val: &str) -> String {
    let mut out = String::with_capacity(val.len() + 4);
    for c in val.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    out
}

fn format_labels(labels: &[(String, String)]) -> String {
    if labels.is_empty() {
        return String::new();
    }
    let mut out = String::from("{");
    for (i, (k, v)) in labels.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(k);
        out.push_str("=\"");
        out.push_str(&escape_label_value(v));
        out.push('"');
    }
    out.push('}');
    out
}

#[derive(Debug, Clone)]
pub struct ScopedPrometheusSink {
    inner: Arc<PrometheusSink>,
    scope_labels: Vec<(String, String)>,
}

impl ScopedPrometheusSink {
    fn merge<'a>(&'a self, labels: &[(&'a str, &'a str)]) -> Vec<(&'a str, &'a str)> {
        let mut res = Vec::with_capacity(self.scope_labels.len() + labels.len());
        for (k, v) in &self.scope_labels {
            res.push((k.as_str(), v.as_str()));
        }
        res.extend_from_slice(labels);
        res
    }
}

impl TelemetrySink for ScopedPrometheusSink {
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
    fn test_openmetrics_exposition_and_pinned_format() {
        let sink = PrometheusSink::new();
        sink.counter_add("lust.vm.gas.consumed", 100, &[("env", "prod")]);
        sink.gauge_set("lust.vm.memory.bytes", 4096.0, &[]);
        sink.histogram_record("request_latency_sec", 0.04, &[]);

        let text = sink.render_openmetrics();
        assert!(text.contains("# HELP lust_vm_gas_consumed_total Total count of lust_vm_gas_consumed"));
        assert!(text.contains("# TYPE lust_vm_gas_consumed_total counter"));
        assert!(text.contains("lust_vm_gas_consumed_total{env=\"prod\"} 100"));

        assert!(text.contains("# HELP lust_vm_memory_bytes Current value of lust_vm_memory_bytes"));
        assert!(text.contains("# TYPE lust_vm_memory_bytes gauge"));
        assert!(text.contains("lust_vm_memory_bytes 4096"));

        assert!(text.contains("# TYPE request_latency_sec histogram"));
        assert!(text.contains("request_latency_sec_bucket{le=\"0.05\"} 1"));
        assert!(text.contains("request_latency_sec_bucket{le=\"+Inf\"} 1"));
        assert!(text.contains("request_latency_sec_sum 0.04"));
        assert!(text.contains("request_latency_sec_count 1"));

        assert!(text.ends_with("# EOF\n"));
    }

    #[test]
    fn test_escaping_rules() {
        assert_eq!(
            escape_label_value("hello \"world\" \n test\\"),
            "hello \\\"world\\\" \\n test\\\\"
        );
        assert_eq!(sanitize_metric_name("my.custom-metric:name"), "my_custom_metric:name");
        assert_eq!(sanitize_label_key("bad-key.name"), "bad_key_name");
    }

    #[test]
    fn test_cardinality_and_overflow_folding() {
        let limits = CardinalityLimits {
            max_unique_names: 5,
            max_label_pairs: 2,
            max_label_val_len: 10,
            max_unique_series: 2,
        };
        let sink = PrometheusSink::with_limits(limits);
        sink.counter_add("users", 1, &[("id", "u1")]);
        sink.counter_add("users", 1, &[("id", "u2")]);
        // 3rd series exceeds max_unique_series (2) -> folds to overflow = "true"
        sink.counter_add("users", 1, &[("id", "u3")]);

        assert_eq!(sink.dropped_series_total(), 1);
        assert_eq!(sink.counter("users", &[("overflow", "true")]), Some(1));
    }

    #[test]
    fn test_scoped_vm_isolation() {
        let sink = Arc::new(PrometheusSink::new());
        let vm1 = sink.scoped(&[("vm_id", "vm-1")]);
        let vm2 = sink.scoped(&[("vm_id", "vm-2")]);

        vm1.counter_add("gas", 50, &[]);
        vm2.counter_add("gas", 80, &[]);

        assert_eq!(sink.counter("gas", &[("vm_id", "vm-1")]), Some(50));
        assert_eq!(sink.counter("gas", &[("vm_id", "vm-2")]), Some(80));
    }

    #[test]
    fn test_flush_and_failure() {
        let sink = PrometheusSink::new();
        sink.flush();
        assert_eq!(sink.flush_count(), 1);
        sink.set_fail_on_flush(true);
        sink.flush();
        assert_eq!(sink.flush_count(), 2);
        assert_eq!(sink.flush_failures(), 1);
    }
}

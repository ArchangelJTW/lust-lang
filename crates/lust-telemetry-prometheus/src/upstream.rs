//! Feature-gated (`upstream`) real `prometheus` crate adapter backed by an independent
//! `prometheus::Registry` and `prometheus::TextEncoder`.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use prometheus::proto::MetricFamily;
use prometheus::{
    Encoder, Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec,
    Opts, Registry, TextEncoder,
};

use crate::{sanitize_label_key, sanitize_metric_name, CardinalityLimits, SpanStatus, TelemetrySink};

#[derive(Debug)]
enum CounterEntry {
    Plain(IntCounter),
    Vec {
        label_names: Vec<String>,
        vec: IntCounterVec,
    },
}

#[derive(Debug)]
enum GaugeEntry {
    Plain(Gauge),
    Vec {
        label_names: Vec<String>,
        vec: GaugeVec,
    },
}

#[derive(Debug)]
enum HistogramEntry {
    Plain(Histogram),
    Vec {
        label_names: Vec<String>,
        vec: HistogramVec,
    },
}

#[derive(Debug, Default)]
struct UpstreamRegistryInner {
    counters: HashMap<String, CounterEntry>,
    gauges: HashMap<String, GaugeEntry>,
    histograms: HashMap<String, HistogramEntry>,
    overflow_counter: Option<IntCounterVec>,
    known_metric_names: HashSet<String>,
    known_series: HashSet<String>,
}

enum ResolvedSeries {
    Normal {
        sanitized_name: String,
        labels: Vec<(String, String)>,
    },
    Overflow {
        sanitized_name: String,
    },
    Dropped,
}

/// Real upstream `prometheus` adapter implementing [`TelemetrySink`].
///
/// Each instance owns an isolated [`prometheus::Registry`] (never mutating the global
/// `prometheus::default_registry()`), enforcing RFC §3.7 multi-VM / multi-tenant isolation.
#[derive(Debug)]
pub struct UpstreamPrometheusSink {
    registry: Registry,
    inner: Mutex<UpstreamRegistryInner>,
    limits: CardinalityLimits,
    default_histogram_bounds: Vec<f64>,
    vm_id: Option<String>,
    enabled: AtomicBool,
    next_span_id: AtomicU64,
    emitted_samples: AtomicU64,
    dropped_series: AtomicU64,
    registration_errors: AtomicU64,
    flush_count: AtomicU64,
    flush_failures: AtomicU64,
    fail_on_flush: AtomicBool,
}

impl UpstreamPrometheusSink {
    pub fn new() -> Self {
        Self::with_limits(CardinalityLimits::default())
    }

    /// Creates an adapter bound to a caller-supplied independent [`prometheus::Registry`].
    pub fn with_registry(registry: Registry) -> Self {
        Self::with_registry_and_limits(registry, CardinalityLimits::default())
    }

    pub fn with_limits(limits: CardinalityLimits) -> Self {
        Self::with_registry_and_limits(Registry::new(), limits)
    }

    pub fn with_registry_and_limits(registry: Registry, limits: CardinalityLimits) -> Self {
        Self {
            registry,
            inner: Mutex::new(UpstreamRegistryInner::default()),
            limits,
            default_histogram_bounds: vec![
                0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ],
            vm_id: None,
            enabled: AtomicBool::new(true),
            next_span_id: AtomicU64::new(1),
            emitted_samples: AtomicU64::new(0),
            dropped_series: AtomicU64::new(0),
            registration_errors: AtomicU64::new(0),
            flush_count: AtomicU64::new(0),
            flush_failures: AtomicU64::new(0),
            fail_on_flush: AtomicBool::new(false),
        }
    }

    pub fn with_vm_id(mut self, vm_id: &str) -> Self {
        self.vm_id = Some(vm_id.to_string());
        self
    }

    pub fn with_enabled(self, enabled: bool) -> Self {
        self.enabled.store(enabled, Ordering::Relaxed);
        self
    }

    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
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

    pub fn emitted_samples(&self) -> u64 {
        self.emitted_samples.load(Ordering::Relaxed)
    }

    pub fn dropped_series_total(&self) -> u64 {
        self.dropped_series.load(Ordering::Relaxed)
    }

    pub fn registration_errors(&self) -> u64 {
        self.registration_errors.load(Ordering::Relaxed)
    }

    /// Returns a reference to the underlying independent [`prometheus::Registry`].
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// Gathers [`MetricFamily`] records directly from the real [`prometheus::Registry`].
    pub fn gather(&self) -> Vec<MetricFamily> {
        self.registry.gather()
    }

    /// Encodes all gathered metrics using the real [`prometheus::TextEncoder`].
    pub fn render_prometheus_text(&self) -> prometheus::Result<String> {
        let families = self.registry.gather();
        TextEncoder::new().encode_to_string(&families)
    }

    /// Encodes all gathered metrics into raw UTF-8 bytes using [`prometheus::Encoder::encode`].
    pub fn encode_prometheus_bytes(&self) -> prometheus::Result<Vec<u8>> {
        let families = self.registry.gather();
        let mut buf = Vec::new();
        TextEncoder::new().encode(&families, &mut buf)?;
        Ok(buf)
    }

    /// Renders Prometheus text exposition with OpenMetrics `# EOF\n` trailer.
    pub fn render_openmetrics(&self) -> prometheus::Result<String> {
        let mut text = self.render_prometheus_text()?;
        text.push_str("# EOF\n");
        Ok(text)
    }

    pub fn scoped(self: &Arc<Self>, scope_labels: &[(&str, &str)]) -> ScopedUpstreamPrometheusSink {
        ScopedUpstreamPrometheusSink {
            inner: Arc::clone(self),
            scope_labels: scope_labels
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    fn resolve_series(
        &self,
        inner: &mut UpstreamRegistryInner,
        name: &str,
        labels: &[(&str, &str)],
        forbid_le_label: bool,
    ) -> ResolvedSeries {
        let mut sanitized_name = sanitize_metric_name(name);
        if sanitized_name.is_empty() {
            sanitized_name.push_str("lust_metric");
        }

        if !inner.known_metric_names.contains(&sanitized_name) {
            if inner.known_metric_names.len() >= self.limits.max_unique_names {
                self.dropped_series.fetch_add(1, Ordering::Relaxed);
                return ResolvedSeries::Dropped;
            }
            inner.known_metric_names.insert(sanitized_name.clone());
        }

        let mut merged = Vec::with_capacity(labels.len() + usize::from(self.vm_id.is_some()));
        if let Some(ref vm_id) = self.vm_id {
            merged.push(("vm_id", vm_id.as_str()));
        }
        merged.extend_from_slice(labels);

        let effective = if merged.len() > self.limits.max_label_pairs {
            &merged[..self.limits.max_label_pairs]
        } else {
            &merged[..]
        };

        let mut sanitized: Vec<(String, String)> = effective
            .iter()
            .map(|(k, v)| {
                let mut key = sanitize_label_key(k);
                if key.is_empty() {
                    key.push('_');
                }
                if forbid_le_label && key == "le" {
                    key = "label_le".to_string();
                }
                let val = truncate_utf8_prefix(v, self.limits.max_label_val_len);
                (key, val)
            })
            .collect();
        sanitized.sort();
        // Deduplicate identical sanitized label keys keeping the last value so
        // prometheus::Registry never rejects duplicate label dimensions.
        let mut deduped: Vec<(String, String)> = Vec::with_capacity(sanitized.len());
        for (k, v) in sanitized {
            if let Some(last) = deduped.last_mut() {
                if last.0 == k {
                    last.1 = v;
                    continue;
                }
            }
            deduped.push((k, v));
        }

        let mut series_key = String::with_capacity(64);
        series_key.push_str(&sanitized_name);
        for (k, v) in &deduped {
            series_key.push(';');
            series_key.push_str(k);
            series_key.push('=');
            series_key.push_str(v);
        }

        if !inner.known_series.contains(&series_key) {
            if inner.known_series.len() >= self.limits.max_unique_series {
                self.dropped_series.fetch_add(1, Ordering::Relaxed);
                return ResolvedSeries::Overflow { sanitized_name };
            }
            inner.known_series.insert(series_key);
        }

        ResolvedSeries::Normal {
            sanitized_name,
            labels: deduped,
        }
    }

    fn record_overflow(&self, inner: &mut UpstreamRegistryInner, metric_name: &str, delta: u64) {
        if inner.overflow_counter.is_none() {
            let opts = Opts::new(
                "lust_telemetry_overflow_total",
                "Total overflow events folded due to cardinality limits",
            );
            match IntCounterVec::new(opts, &["metric", "overflow"]) {
                Ok(vec) => match self.registry.register(Box::new(vec.clone())) {
                    Ok(()) => inner.overflow_counter = Some(vec),
                    Err(_) => {
                        self.registration_errors.fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                },
                Err(_) => {
                    self.registration_errors.fetch_add(1, Ordering::Relaxed);
                    return;
                }
            }
        }
        if let Some(ref overflow) = inner.overflow_counter {
            overflow
                .with_label_values(&[metric_name, "true"])
                .inc_by(delta);
        }
    }
}

impl Default for UpstreamPrometheusSink {
    fn default() -> Self {
        Self::new()
    }
}

impl TelemetrySink for UpstreamPrometheusSink {
    fn counter_add(&self, name: &str, delta: u64, labels: &[(&str, &str)]) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        let mut inner = self.inner.lock().unwrap();
        match self.resolve_series(&mut inner, name, labels, false) {
            ResolvedSeries::Dropped => {}
            ResolvedSeries::Overflow { sanitized_name } => {
                self.record_overflow(&mut inner, &sanitized_name, delta.max(1));
                self.emitted_samples.fetch_add(1, Ordering::Relaxed);
            }
            ResolvedSeries::Normal {
                sanitized_name,
                labels,
            } => {
                let metric_name = if sanitized_name.ends_with("_total") {
                    sanitized_name.clone()
                } else {
                    format!("{sanitized_name}_total")
                };

                if !inner.counters.contains_key(&metric_name) {
                    let help = format!("Total count of {sanitized_name}");
                    let opts = Opts::new(metric_name.clone(), help);
                    let entry = if labels.is_empty() {
                        match IntCounter::with_opts(opts) {
                            Ok(counter) => match self.registry.register(Box::new(counter.clone())) {
                                Ok(()) => CounterEntry::Plain(counter),
                                Err(_) => {
                                    self.registration_errors.fetch_add(1, Ordering::Relaxed);
                                    return;
                                }
                            },
                            Err(_) => {
                                self.registration_errors.fetch_add(1, Ordering::Relaxed);
                                return;
                            }
                        }
                    } else {
                        let label_names: Vec<String> =
                            labels.iter().map(|(k, _)| k.clone()).collect();
                        let label_refs: Vec<&str> =
                            label_names.iter().map(String::as_str).collect();
                        match IntCounterVec::new(opts, &label_refs) {
                            Ok(vec) => match self.registry.register(Box::new(vec.clone())) {
                                Ok(()) => CounterEntry::Vec { label_names, vec },
                                Err(_) => {
                                    self.registration_errors.fetch_add(1, Ordering::Relaxed);
                                    return;
                                }
                            },
                            Err(_) => {
                                self.registration_errors.fetch_add(1, Ordering::Relaxed);
                                return;
                            }
                        }
                    };
                    inner.counters.insert(metric_name.clone(), entry);
                }

                match inner.counters.get(&metric_name) {
                    Some(CounterEntry::Plain(counter)) if labels.is_empty() => {
                        counter.inc_by(delta);
                        self.emitted_samples.fetch_add(1, Ordering::Relaxed);
                    }
                    Some(CounterEntry::Vec { label_names, vec })
                        if label_names.len() == labels.len()
                            && label_names
                                .iter()
                                .zip(labels.iter())
                                .all(|(expected, (actual, _))| expected == actual) =>
                    {
                        let vals: Vec<&str> = labels.iter().map(|(_, v)| v.as_str()).collect();
                        if let Ok(counter) = vec.get_metric_with_label_values(&vals) {
                            counter.inc_by(delta);
                            self.emitted_samples.fetch_add(1, Ordering::Relaxed);
                        } else {
                            self.registration_errors.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    _ => {
                        self.registration_errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
    }

    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        let mut inner = self.inner.lock().unwrap();
        match self.resolve_series(&mut inner, name, labels, false) {
            ResolvedSeries::Dropped => {}
            ResolvedSeries::Overflow { sanitized_name } => {
                self.record_overflow(&mut inner, &sanitized_name, 1);
                self.emitted_samples.fetch_add(1, Ordering::Relaxed);
            }
            ResolvedSeries::Normal {
                sanitized_name,
                labels,
            } => {
                if !inner.gauges.contains_key(&sanitized_name) {
                    let help = format!("Current value of {sanitized_name}");
                    let opts = Opts::new(sanitized_name.clone(), help);
                    let entry = if labels.is_empty() {
                        match Gauge::with_opts(opts) {
                            Ok(gauge) => match self.registry.register(Box::new(gauge.clone())) {
                                Ok(()) => GaugeEntry::Plain(gauge),
                                Err(_) => {
                                    self.registration_errors.fetch_add(1, Ordering::Relaxed);
                                    return;
                                }
                            },
                            Err(_) => {
                                self.registration_errors.fetch_add(1, Ordering::Relaxed);
                                return;
                            }
                        }
                    } else {
                        let label_names: Vec<String> =
                            labels.iter().map(|(k, _)| k.clone()).collect();
                        let label_refs: Vec<&str> =
                            label_names.iter().map(String::as_str).collect();
                        match GaugeVec::new(opts, &label_refs) {
                            Ok(vec) => match self.registry.register(Box::new(vec.clone())) {
                                Ok(()) => GaugeEntry::Vec { label_names, vec },
                                Err(_) => {
                                    self.registration_errors.fetch_add(1, Ordering::Relaxed);
                                    return;
                                }
                            },
                            Err(_) => {
                                self.registration_errors.fetch_add(1, Ordering::Relaxed);
                                return;
                            }
                        }
                    };
                    inner.gauges.insert(sanitized_name.clone(), entry);
                }

                match inner.gauges.get(&sanitized_name) {
                    Some(GaugeEntry::Plain(gauge)) if labels.is_empty() => {
                        gauge.set(value);
                        self.emitted_samples.fetch_add(1, Ordering::Relaxed);
                    }
                    Some(GaugeEntry::Vec { label_names, vec })
                        if label_names.len() == labels.len()
                            && label_names
                                .iter()
                                .zip(labels.iter())
                                .all(|(expected, (actual, _))| expected == actual) =>
                    {
                        let vals: Vec<&str> = labels.iter().map(|(_, v)| v.as_str()).collect();
                        if let Ok(gauge) = vec.get_metric_with_label_values(&vals) {
                            gauge.set(value);
                            self.emitted_samples.fetch_add(1, Ordering::Relaxed);
                        } else {
                            self.registration_errors.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    _ => {
                        self.registration_errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
    }

    fn histogram_record(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        let mut inner = self.inner.lock().unwrap();
        match self.resolve_series(&mut inner, name, labels, true) {
            ResolvedSeries::Dropped => {}
            ResolvedSeries::Overflow { sanitized_name } => {
                self.record_overflow(&mut inner, &sanitized_name, 1);
                self.emitted_samples.fetch_add(1, Ordering::Relaxed);
            }
            ResolvedSeries::Normal {
                sanitized_name,
                labels,
            } => {
                if !inner.histograms.contains_key(&sanitized_name) {
                    let help = format!("Histogram distribution of {sanitized_name}");
                    let opts = HistogramOpts::new(sanitized_name.clone(), help)
                        .buckets(self.default_histogram_bounds.clone());
                    let entry = if labels.is_empty() {
                        match Histogram::with_opts(opts) {
                            Ok(hist) => match self.registry.register(Box::new(hist.clone())) {
                                Ok(()) => HistogramEntry::Plain(hist),
                                Err(_) => {
                                    self.registration_errors.fetch_add(1, Ordering::Relaxed);
                                    return;
                                }
                            },
                            Err(_) => {
                                self.registration_errors.fetch_add(1, Ordering::Relaxed);
                                return;
                            }
                        }
                    } else {
                        let label_names: Vec<String> =
                            labels.iter().map(|(k, _)| k.clone()).collect();
                        let label_refs: Vec<&str> =
                            label_names.iter().map(String::as_str).collect();
                        match HistogramVec::new(opts, &label_refs) {
                            Ok(vec) => match self.registry.register(Box::new(vec.clone())) {
                                Ok(()) => HistogramEntry::Vec { label_names, vec },
                                Err(_) => {
                                    self.registration_errors.fetch_add(1, Ordering::Relaxed);
                                    return;
                                }
                            },
                            Err(_) => {
                                self.registration_errors.fetch_add(1, Ordering::Relaxed);
                                return;
                            }
                        }
                    };
                    inner.histograms.insert(sanitized_name.clone(), entry);
                }

                match inner.histograms.get(&sanitized_name) {
                    Some(HistogramEntry::Plain(hist)) if labels.is_empty() => {
                        hist.observe(value);
                        self.emitted_samples.fetch_add(1, Ordering::Relaxed);
                    }
                    Some(HistogramEntry::Vec { label_names, vec })
                        if label_names.len() == labels.len()
                            && label_names
                                .iter()
                                .zip(labels.iter())
                                .all(|(expected, (actual, _))| expected == actual) =>
                    {
                        let vals: Vec<&str> = labels.iter().map(|(_, v)| v.as_str()).collect();
                        if let Ok(hist) = vec.get_metric_with_label_values(&vals) {
                            hist.observe(value);
                            self.emitted_samples.fetch_add(1, Ordering::Relaxed);
                        } else {
                            self.registration_errors.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    _ => {
                        self.registration_errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
    }

    fn span_enter(&self, _name: &str, _labels: &[(&str, &str)]) -> u64 {
        if !self.enabled.load(Ordering::Relaxed) {
            return 0;
        }
        self.next_span_id.fetch_add(1, Ordering::Relaxed)
    }

    fn span_exit(&self, _span_id: u64, _status: SpanStatus) {}

    fn flush(&self) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        self.flush_count.fetch_add(1, Ordering::Relaxed);
        if self.fail_on_flush.load(Ordering::Relaxed) {
            self.flush_failures.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Scoped wrapper around [`UpstreamPrometheusSink`] attaching immutable per-VM/tenant labels.
#[derive(Debug, Clone)]
pub struct ScopedUpstreamPrometheusSink {
    inner: Arc<UpstreamPrometheusSink>,
    scope_labels: Vec<(String, String)>,
}

impl ScopedUpstreamPrometheusSink {
    fn merge<'a>(&'a self, labels: &[(&'a str, &'a str)]) -> Vec<(&'a str, &'a str)> {
        let mut res = Vec::with_capacity(self.scope_labels.len() + labels.len());
        for (k, v) in &self.scope_labels {
            res.push((k.as_str(), v.as_str()));
        }
        res.extend_from_slice(labels);
        res
    }
}

impl TelemetrySink for ScopedUpstreamPrometheusSink {
    fn counter_add(&self, name: &str, delta: u64, labels: &[(&str, &str)]) {
        if !self.inner.is_enabled() {
            return;
        }
        let merged = self.merge(labels);
        self.inner.counter_add(name, delta, &merged);
    }

    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        if !self.inner.is_enabled() {
            return;
        }
        let merged = self.merge(labels);
        self.inner.gauge_set(name, value, &merged);
    }

    fn histogram_record(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        if !self.inner.is_enabled() {
            return;
        }
        let merged = self.merge(labels);
        self.inner.histogram_record(name, value, &merged);
    }

    fn span_enter(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
        if !self.inner.is_enabled() {
            return 0;
        }
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

fn truncate_utf8_prefix(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

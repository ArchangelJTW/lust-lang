//! Feature-gated (`upstream`) real `metrics` crate adapter and independent test recorder.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use metrics::{
    Counter, CounterFn, Gauge, GaugeFn, Histogram, HistogramFn, Key, KeyName, Label, Level, Metadata,
    Recorder, SharedString, Unit,
};

use crate::{CardinalityLimits, MetricKey, SpanStatus, TelemetrySink};

static UPSTREAM_METADATA: Metadata<'static> = Metadata::new(
    "lust_telemetry_metrics",
    Level::INFO,
    Some(module_path!()),
);

#[derive(Debug, Default)]
struct CardinalityState {
    known_metric_names: HashSet<String>,
    known_series: HashSet<String>,
}

/// Real upstream `metrics` crate adapter implementing [`TelemetrySink`].
///
/// Supports two dispatch modes per RFC §3.7:
/// - **Isolated recorder mode** ([`UpstreamMetricsSink::with_recorder`]): dispatches inside
///   `metrics::with_local_recorder(...)` so multi-VM hosts and parallel tests never collide.
/// - **Ambient recorder mode** ([`UpstreamMetricsSink::new`]): dispatches through the active
///   thread-local or global `metrics` recorder (`metrics::with_recorder`).
pub struct UpstreamMetricsSink {
    recorder: Option<Arc<dyn Recorder + Send + Sync>>,
    cardinality: Mutex<CardinalityState>,
    limits: CardinalityLimits,
    vm_id: Option<String>,
    enabled: AtomicBool,
    next_span_id: AtomicU64,
    emitted_samples: AtomicU64,
    dropped_series: AtomicU64,
    flush_count: AtomicU64,
    flush_failures: AtomicU64,
    fail_on_flush: AtomicBool,
}

impl std::fmt::Debug for UpstreamMetricsSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UpstreamMetricsSink")
            .field("has_bound_recorder", &self.recorder.is_some())
            .field("limits", &self.limits)
            .field("vm_id", &self.vm_id)
            .field("enabled", &self.enabled.load(Ordering::Relaxed))
            .field(
                "emitted_samples",
                &self.emitted_samples.load(Ordering::Relaxed),
            )
            .field(
                "dropped_series",
                &self.dropped_series.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl UpstreamMetricsSink {
    /// Creates an adapter that emits through the currently active local or global `metrics` recorder.
    pub fn new() -> Self {
        Self {
            recorder: None,
            cardinality: Mutex::new(CardinalityState::default()),
            limits: CardinalityLimits::default(),
            vm_id: None,
            enabled: AtomicBool::new(true),
            next_span_id: AtomicU64::new(1),
            emitted_samples: AtomicU64::new(0),
            dropped_series: AtomicU64::new(0),
            flush_count: AtomicU64::new(0),
            flush_failures: AtomicU64::new(0),
            fail_on_flush: AtomicBool::new(false),
        }
    }

    /// Creates an adapter pinned to an independent `metrics::Recorder` via
    /// `metrics::with_local_recorder`.
    pub fn with_recorder(recorder: Arc<dyn Recorder + Send + Sync>) -> Self {
        let mut sink = Self::new();
        sink.recorder = Some(recorder);
        sink
    }

    pub fn with_limits(mut self, limits: CardinalityLimits) -> Self {
        self.limits = limits;
        self
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

    pub fn dropped_series_total(&self) -> u64 {
        self.dropped_series.load(Ordering::Relaxed)
    }

    pub fn emitted_samples(&self) -> u64 {
        self.emitted_samples.load(Ordering::Relaxed)
    }

    pub fn describe_counter(&self, name: &str, unit: Option<Unit>, description: &str) {
        if !self.is_enabled() {
            return;
        }
        let key_name = KeyName::from(name.to_string());
        let desc = SharedString::from(description.to_string());
        self.with_active_recorder(|recorder| {
            recorder.describe_counter(key_name, unit, desc);
        });
    }

    pub fn describe_gauge(&self, name: &str, unit: Option<Unit>, description: &str) {
        if !self.is_enabled() {
            return;
        }
        let key_name = KeyName::from(name.to_string());
        let desc = SharedString::from(description.to_string());
        self.with_active_recorder(|recorder| {
            recorder.describe_gauge(key_name, unit, desc);
        });
    }

    pub fn describe_histogram(&self, name: &str, unit: Option<Unit>, description: &str) {
        if !self.is_enabled() {
            return;
        }
        let key_name = KeyName::from(name.to_string());
        let desc = SharedString::from(description.to_string());
        self.with_active_recorder(|recorder| {
            recorder.describe_histogram(key_name, unit, desc);
        });
    }

    pub fn scoped(self: &Arc<Self>, scope_labels: &[(&str, &str)]) -> ScopedUpstreamMetricsSink {
        ScopedUpstreamMetricsSink {
            inner: Arc::clone(self),
            scope_labels: scope_labels
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    fn with_active_recorder<R>(&self, f: impl FnOnce(&dyn Recorder) -> R) -> R {
        if let Some(ref bound) = self.recorder {
            metrics::with_local_recorder(bound.as_ref(), || metrics::with_recorder(f))
        } else {
            metrics::with_recorder(f)
        }
    }

    fn build_bounded_key(&self, name: &str, labels: &[(&str, &str)]) -> Option<Key> {
        let mut state = self.cardinality.lock().unwrap();

        if !state.known_metric_names.contains(name) {
            if state.known_metric_names.len() >= self.limits.max_unique_names {
                self.dropped_series.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            state.known_metric_names.insert(name.to_string());
        }

        let mut merged_labels = Vec::with_capacity(labels.len() + usize::from(self.vm_id.is_some()));
        if let Some(ref vm_id) = self.vm_id {
            merged_labels.push(("vm_id", vm_id.as_str()));
        }
        merged_labels.extend_from_slice(labels);

        let effective = if merged_labels.len() > self.limits.max_label_pairs {
            &merged_labels[..self.limits.max_label_pairs]
        } else {
            &merged_labels[..]
        };

        let mut sanitized: Vec<(String, String)> = effective
            .iter()
            .map(|(k, v)| {
                let val = truncate_utf8_prefix(v, self.limits.max_label_val_len);
                (k.to_string(), val)
            })
            .collect();
        sanitized.sort();

        let mut series_key = String::with_capacity(64);
        series_key.push_str(name);
        for (k, v) in &sanitized {
            series_key.push(';');
            series_key.push_str(k);
            series_key.push('=');
            series_key.push_str(v);
        }

        if !state.known_series.contains(&series_key) {
            if state.known_series.len() >= self.limits.max_unique_series {
                self.dropped_series.fetch_add(1, Ordering::Relaxed);
                return Some(Key::from_parts(
                    name.to_string(),
                    vec![Label::from_static_parts("overflow", "true")],
                ));
            }
            state.known_series.insert(series_key);
        }

        let metric_labels: Vec<Label> = sanitized
            .into_iter()
            .map(|(k, v)| Label::new(k, v))
            .collect();
        Some(Key::from_parts(name.to_string(), metric_labels))
    }
}

impl Default for UpstreamMetricsSink {
    fn default() -> Self {
        Self::new()
    }
}

impl TelemetrySink for UpstreamMetricsSink {
    fn counter_add(&self, name: &str, delta: u64, labels: &[(&str, &str)]) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        if let Some(key) = self.build_bounded_key(name, labels) {
            self.with_active_recorder(|recorder| {
                let handle = recorder.register_counter(&key, &UPSTREAM_METADATA);
                handle.increment(delta);
            });
            self.emitted_samples.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        if let Some(key) = self.build_bounded_key(name, labels) {
            self.with_active_recorder(|recorder| {
                let handle = recorder.register_gauge(&key, &UPSTREAM_METADATA);
                handle.set(value);
            });
            self.emitted_samples.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn histogram_record(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        if let Some(key) = self.build_bounded_key(name, labels) {
            self.with_active_recorder(|recorder| {
                let handle = recorder.register_histogram(&key, &UPSTREAM_METADATA);
                handle.record(value);
            });
            self.emitted_samples.fetch_add(1, Ordering::Relaxed);
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

/// Scoped wrapper around [`UpstreamMetricsSink`] attaching immutable per-VM/tenant labels.
#[derive(Debug, Clone)]
pub struct ScopedUpstreamMetricsSink {
    inner: Arc<UpstreamMetricsSink>,
    scope_labels: Vec<(String, String)>,
}

impl ScopedUpstreamMetricsSink {
    fn merge<'a>(&'a self, labels: &[(&'a str, &'a str)]) -> Vec<(&'a str, &'a str)> {
        let mut res = Vec::with_capacity(self.scope_labels.len() + labels.len());
        for (k, v) in &self.scope_labels {
            res.push((k.as_str(), v.as_str()));
        }
        res.extend_from_slice(labels);
        res
    }
}

impl TelemetrySink for ScopedUpstreamMetricsSink {
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

// -----------------------------------------------------------------------------
// Independent `metrics::Recorder` implementation for verification & testing
// -----------------------------------------------------------------------------

#[derive(Debug, Default)]
struct AtomicCounterHandle {
    value: AtomicU64,
}

impl CounterFn for AtomicCounterHandle {
    fn increment(&self, value: u64) {
        self.value.fetch_add(value, Ordering::Relaxed);
    }

    fn absolute(&self, value: u64) {
        self.value.fetch_max(value, Ordering::Relaxed);
    }
}

#[derive(Debug)]
struct AtomicGaugeHandle {
    bits: AtomicU64,
}

impl Default for AtomicGaugeHandle {
    fn default() -> Self {
        Self {
            bits: AtomicU64::new(0.0_f64.to_bits()),
        }
    }
}

impl AtomicGaugeHandle {
    fn load(&self) -> f64 {
        f64::from_bits(self.bits.load(Ordering::Relaxed))
    }

    fn update(&self, f: impl Fn(f64) -> f64) {
        let mut cur = self.bits.load(Ordering::Relaxed);
        loop {
            let next = f(f64::from_bits(cur)).to_bits();
            match self
                .bits
                .compare_exchange_weak(cur, next, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }
    }
}

impl GaugeFn for AtomicGaugeHandle {
    fn increment(&self, value: f64) {
        self.update(|cur| cur + value);
    }

    fn decrement(&self, value: f64) {
        self.update(|cur| cur - value);
    }

    fn set(&self, value: f64) {
        self.bits.store(value.to_bits(), Ordering::Relaxed);
    }
}

#[derive(Debug, Default)]
struct VecHistogramHandle {
    values: Mutex<Vec<f64>>,
}

impl HistogramFn for VecHistogramHandle {
    fn record(&self, value: f64) {
        self.values.lock().unwrap().push(value);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetricDescription {
    pub unit: Option<Unit>,
    pub description: String,
}

#[derive(Debug, Default)]
struct RecorderStore {
    counters: HashMap<MetricKey, Arc<AtomicCounterHandle>>,
    gauges: HashMap<MetricKey, Arc<AtomicGaugeHandle>>,
    histograms: HashMap<MetricKey, Arc<VecHistogramHandle>>,
    descriptions: HashMap<String, MetricDescription>,
}

/// Independent, thread-safe [`metrics::Recorder`] storing all upstream `metrics` crate
/// registrations and observations in memory without touching any global process state.
#[derive(Debug, Default)]
pub struct InMemoryMetricsRecorder {
    store: Mutex<RecorderStore>,
}

impl InMemoryMetricsRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn counter(&self, name: &str, labels: &[(&str, &str)]) -> Option<u64> {
        let key = MetricKey::new(name, labels);
        let store = self.store.lock().unwrap();
        store
            .counters
            .get(&key)
            .map(|h| h.value.load(Ordering::Relaxed))
    }

    pub fn counter_total(&self, name: &str) -> u64 {
        let store = self.store.lock().unwrap();
        store
            .counters
            .iter()
            .filter(|(k, _)| k.name == name)
            .map(|(_, h)| h.value.load(Ordering::Relaxed))
            .sum()
    }

    pub fn gauge(&self, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
        let key = MetricKey::new(name, labels);
        let store = self.store.lock().unwrap();
        store.gauges.get(&key).map(|h| h.load())
    }

    pub fn histogram_values(&self, name: &str, labels: &[(&str, &str)]) -> Option<Vec<f64>> {
        let key = MetricKey::new(name, labels);
        let store = self.store.lock().unwrap();
        store
            .histograms
            .get(&key)
            .map(|h| h.values.lock().unwrap().clone())
    }

    pub fn description(&self, name: &str) -> Option<MetricDescription> {
        let store = self.store.lock().unwrap();
        store.descriptions.get(name).cloned()
    }

    pub fn total_series(&self) -> usize {
        let store = self.store.lock().unwrap();
        store.counters.len() + store.gauges.len() + store.histograms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.total_series() == 0
    }

    pub fn clear(&self) {
        let mut store = self.store.lock().unwrap();
        store.counters.clear();
        store.gauges.clear();
        store.histograms.clear();
        store.descriptions.clear();
    }
}

fn canonical_key(key: &Key) -> MetricKey {
    let mut labels: Vec<(String, String)> = key
        .labels()
        .map(|l| (l.key().to_string(), l.value().to_string()))
        .collect();
    labels.sort();
    MetricKey {
        name: key.name().to_string(),
        labels,
    }
}

impl Recorder for InMemoryMetricsRecorder {
    fn describe_counter(&self, key: KeyName, unit: Option<Unit>, description: SharedString) {
        let mut store = self.store.lock().unwrap();
        store.descriptions.insert(
            key.as_str().to_string(),
            MetricDescription {
                unit,
                description: description.into_owned(),
            },
        );
    }

    fn describe_gauge(&self, key: KeyName, unit: Option<Unit>, description: SharedString) {
        let mut store = self.store.lock().unwrap();
        store.descriptions.insert(
            key.as_str().to_string(),
            MetricDescription {
                unit,
                description: description.into_owned(),
            },
        );
    }

    fn describe_histogram(&self, key: KeyName, unit: Option<Unit>, description: SharedString) {
        let mut store = self.store.lock().unwrap();
        store.descriptions.insert(
            key.as_str().to_string(),
            MetricDescription {
                unit,
                description: description.into_owned(),
            },
        );
    }

    fn register_counter(&self, key: &Key, _metadata: &Metadata<'_>) -> Counter {
        let canonical = canonical_key(key);
        let mut store = self.store.lock().unwrap();
        let handle = store
            .counters
            .entry(canonical)
            .or_insert_with(|| Arc::new(AtomicCounterHandle::default()))
            .clone();
        Counter::from_arc(handle)
    }

    fn register_gauge(&self, key: &Key, _metadata: &Metadata<'_>) -> Gauge {
        let canonical = canonical_key(key);
        let mut store = self.store.lock().unwrap();
        let handle = store
            .gauges
            .entry(canonical)
            .or_insert_with(|| Arc::new(AtomicGaugeHandle::default()))
            .clone();
        Gauge::from_arc(handle)
    }

    fn register_histogram(&self, key: &Key, _metadata: &Metadata<'_>) -> Histogram {
        let canonical = canonical_key(key);
        let mut store = self.store.lock().unwrap();
        let handle = store
            .histograms
            .entry(canonical)
            .or_insert_with(|| Arc::new(VecHistogramHandle::default()))
            .clone();
        Histogram::from_arc(handle)
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

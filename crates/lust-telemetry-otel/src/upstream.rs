//! Feature-gated (`upstream`) real OpenTelemetry (`opentelemetry` + `opentelemetry_sdk`)
//! adapter backed by `InMemorySpanExporter`, `InMemoryMetricExporter`, and `ManualReader`.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use opentelemetry::metrics::{Counter, Gauge, Histogram, Meter, MeterProvider};
use opentelemetry::trace::{SpanKind, Status, TraceContextExt, Tracer, TracerProvider};
use opentelemetry::{Context, KeyValue};
use opentelemetry_sdk::error::{OTelSdkError, OTelSdkResult};
use opentelemetry_sdk::metrics::data::ResourceMetrics;
use opentelemetry_sdk::metrics::reader::MetricReader;
use opentelemetry_sdk::metrics::{
    InMemoryMetricExporter, InstrumentKind, ManualReader, PeriodicReader, Pipeline,
    SdkMeterProvider, Temporality,
};
use opentelemetry_sdk::trace::{
    InMemorySpanExporter, SdkTracer, SdkTracerProvider, SimpleSpanProcessor, SpanData,
};
use opentelemetry_sdk::Resource;

use crate::{SpanStatus, TelemetrySink};

static NEXT_SINK_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static ACTIVE_UPSTREAM_SPAN_STACK: RefCell<Vec<(u64, u64, Context)>> = const { RefCell::new(Vec::new()) };
}

/// Cardinality limits applied to upstream OpenTelemetry metric series.
#[derive(Debug, Clone)]
pub struct OtelCardinalityLimits {
    pub max_unique_names: usize,
    pub max_label_pairs: usize,
    pub max_label_val_len: usize,
    pub max_unique_series: usize,
}

impl Default for OtelCardinalityLimits {
    fn default() -> Self {
        Self {
            max_unique_names: 256,
            max_label_pairs: 8,
            max_label_val_len: 64,
            max_unique_series: 4096,
        }
    }
}

/// Shared, cloneable wrapper around [`ManualReader`] implementing [`MetricReader`] so callers can
/// pull aggregated [`ResourceMetrics`] synchronously on demand after registering it with
/// [`SdkMeterProvider`].
#[derive(Clone, Debug)]
pub struct SharedManualReader {
    inner: Arc<ManualReader>,
}

impl SharedManualReader {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ManualReader::builder().build()),
        }
    }

    pub fn with_temporality(temporality: Temporality) -> Self {
        Self {
            inner: Arc::new(
                ManualReader::builder()
                    .with_temporality(temporality)
                    .build(),
            ),
        }
    }

    /// Synchronously collects all currently aggregated metrics into a fresh [`ResourceMetrics`].
    pub fn collect_snapshot(&self) -> Result<ResourceMetrics, OTelSdkError> {
        let mut rm = ResourceMetrics::default();
        self.inner.collect(&mut rm)?;
        Ok(rm)
    }
}

impl Default for SharedManualReader {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricReader for SharedManualReader {
    fn register_pipeline(&self, pipeline: Weak<Pipeline>) {
        self.inner.register_pipeline(pipeline);
    }

    fn collect(&self, rm: &mut ResourceMetrics) -> OTelSdkResult {
        self.inner.collect(rm)
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.inner.force_flush()
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.inner.shutdown_with_timeout(timeout)
    }

    fn temporality(&self, kind: InstrumentKind) -> Temporality {
        self.inner.temporality(kind)
    }
}

#[derive(Default)]
struct InstrumentCache {
    counters: HashMap<String, Counter<u64>>,
    gauges: HashMap<String, Gauge<f64>>,
    histograms: HashMap<String, Histogram<f64>>,
    known_metric_names: HashSet<String>,
    known_series: HashSet<String>,
}

impl std::fmt::Debug for InstrumentCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstrumentCache")
            .field("counters", &self.counters.len())
            .field("gauges", &self.gauges.len())
            .field("histograms", &self.histograms.len())
            .field("known_series", &self.known_series.len())
            .finish()
    }
}

/// Real upstream OpenTelemetry SDK adapter implementing [`TelemetrySink`].
///
/// Wires an isolated [`SdkTracerProvider`] and [`SdkMeterProvider`] (zero global OpenTelemetry
/// state mutation) backed by [`InMemorySpanExporter`], [`InMemoryMetricExporter`], and
/// [`ManualReader`].
pub struct UpstreamOtelSink {
    sink_id: u64,
    service_name: String,
    vm_id: Option<String>,
    tracer_provider: SdkTracerProvider,
    tracer: SdkTracer,
    meter_provider: SdkMeterProvider,
    meter: Meter,
    span_exporter: Option<InMemorySpanExporter>,
    metric_exporter: Option<InMemoryMetricExporter>,
    manual_reader: Option<SharedManualReader>,
    instruments: Mutex<InstrumentCache>,
    open_spans: Mutex<HashMap<u64, Context>>,
    limits: OtelCardinalityLimits,
    enabled: AtomicBool,
    next_span_id: AtomicU64,
    emitted_samples: AtomicU64,
    dropped_series: AtomicU64,
    flush_count: AtomicU64,
    flush_failures: AtomicU64,
    fail_on_flush: AtomicBool,
}

impl std::fmt::Debug for UpstreamOtelSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UpstreamOtelSink")
            .field("service_name", &self.service_name)
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

impl UpstreamOtelSink {
    /// Creates an isolated upstream OpenTelemetry sink with:
    /// - [`InMemorySpanExporter`] via [`SimpleSpanProcessor`]
    /// - [`InMemoryMetricExporter`] via [`PeriodicReader`] (flushed explicitly on [`TelemetrySink::flush`])
    /// - [`SharedManualReader`] ([`ManualReader`]) for synchronous pull collection
    pub fn new_in_memory(service_name: &str, vm_id: Option<&str>) -> Self {
        let resource = build_resource(service_name, vm_id);

        let span_exporter = InMemorySpanExporter::default();
        let tracer_provider = SdkTracerProvider::builder()
            .with_resource(resource.clone())
            .with_span_processor(SimpleSpanProcessor::new(span_exporter.clone()))
            .build();
        let tracer = tracer_provider.tracer("lust-runtime");

        let metric_exporter = InMemoryMetricExporter::default();
        let periodic_reader = PeriodicReader::builder(metric_exporter.clone())
            .with_interval(Duration::from_secs(3600))
            .build();
        let manual_reader = SharedManualReader::new();
        let meter_provider = SdkMeterProvider::builder()
            .with_resource(resource)
            .with_reader(periodic_reader)
            .with_reader(manual_reader.clone())
            .build();
        let meter = meter_provider.meter("lust-runtime");

        Self {
            sink_id: NEXT_SINK_ID.fetch_add(1, Ordering::Relaxed),
            service_name: service_name.to_string(),
            vm_id: vm_id.map(str::to_string),
            tracer_provider,
            tracer,
            meter_provider,
            meter,
            span_exporter: Some(span_exporter),
            metric_exporter: Some(metric_exporter),
            manual_reader: Some(manual_reader),
            instruments: Mutex::new(InstrumentCache::default()),
            open_spans: Mutex::new(HashMap::new()),
            limits: OtelCardinalityLimits::default(),
            enabled: AtomicBool::new(true),
            next_span_id: AtomicU64::new(1),
            emitted_samples: AtomicU64::new(0),
            dropped_series: AtomicU64::new(0),
            flush_count: AtomicU64::new(0),
            flush_failures: AtomicU64::new(0),
            fail_on_flush: AtomicBool::new(false),
        }
    }

    /// Creates an isolated upstream OpenTelemetry sink with zero background threads
    /// ([`InMemorySpanExporter`] + [`ManualReader`] only).
    pub fn new_manual_only(service_name: &str, vm_id: Option<&str>) -> Self {
        let resource = build_resource(service_name, vm_id);

        let span_exporter = InMemorySpanExporter::default();
        let tracer_provider = SdkTracerProvider::builder()
            .with_resource(resource.clone())
            .with_span_processor(SimpleSpanProcessor::new(span_exporter.clone()))
            .build();
        let tracer = tracer_provider.tracer("lust-runtime");

        let manual_reader = SharedManualReader::new();
        let meter_provider = SdkMeterProvider::builder()
            .with_resource(resource)
            .with_reader(manual_reader.clone())
            .build();
        let meter = meter_provider.meter("lust-runtime");

        Self {
            sink_id: NEXT_SINK_ID.fetch_add(1, Ordering::Relaxed),
            service_name: service_name.to_string(),
            vm_id: vm_id.map(str::to_string),
            tracer_provider,
            tracer,
            meter_provider,
            meter,
            span_exporter: Some(span_exporter),
            metric_exporter: None,
            manual_reader: Some(manual_reader),
            instruments: Mutex::new(InstrumentCache::default()),
            open_spans: Mutex::new(HashMap::new()),
            limits: OtelCardinalityLimits::default(),
            enabled: AtomicBool::new(true),
            next_span_id: AtomicU64::new(1),
            emitted_samples: AtomicU64::new(0),
            dropped_series: AtomicU64::new(0),
            flush_count: AtomicU64::new(0),
            flush_failures: AtomicU64::new(0),
            fail_on_flush: AtomicBool::new(false),
        }
    }

    pub fn with_limits(mut self, limits: OtelCardinalityLimits) -> Self {
        self.limits = limits;
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

    /// Returns all finished spans exported to the real [`InMemorySpanExporter`].
    pub fn finished_spans(&self) -> Result<Vec<SpanData>, OTelSdkError> {
        match self.span_exporter {
            Some(ref exporter) => exporter.get_finished_spans(),
            None => Err(OTelSdkError::InternalFailure(
                "no in-memory span exporter configured".to_string(),
            )),
        }
    }

    /// Returns all exported metric batches captured by the real [`InMemoryMetricExporter`].
    pub fn finished_metrics(&self) -> Result<Vec<ResourceMetrics>, OTelSdkError> {
        match self.metric_exporter {
            Some(ref exporter) => exporter.get_finished_metrics(),
            None => Err(OTelSdkError::InternalFailure(
                "no in-memory metric exporter configured".to_string(),
            )),
        }
    }

    /// Collects the latest aggregated [`ResourceMetrics`] directly via the real [`ManualReader`].
    pub fn collect_manual_metrics(&self) -> Result<ResourceMetrics, OTelSdkError> {
        match self.manual_reader {
            Some(ref reader) => reader.collect_snapshot(),
            None => Err(OTelSdkError::InternalFailure(
                "no manual reader configured".to_string(),
            )),
        }
    }

    /// Shuts down both the OpenTelemetry trace and meter providers.
    pub fn shutdown(&self) -> OTelSdkResult {
        let trace_res = self.tracer_provider.shutdown();
        let metric_res = self.meter_provider.shutdown();
        trace_res.and(metric_res)
    }

    pub fn scoped(self: &Arc<Self>, scope_labels: &[(&str, &str)]) -> ScopedUpstreamOtelSink {
        ScopedUpstreamOtelSink {
            inner: Arc::clone(self),
            scope_labels: scope_labels
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    fn prepare_metric_attributes(
        &self,
        cache: &mut InstrumentCache,
        name: &str,
        labels: &[(&str, &str)],
    ) -> Option<(String, Vec<KeyValue>)> {
        let valid_name = sanitize_otel_instrument_name(name);

        if !cache.known_metric_names.contains(&valid_name) {
            if cache.known_metric_names.len() >= self.limits.max_unique_names {
                self.dropped_series.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            cache.known_metric_names.insert(valid_name.clone());
        }

        let mut merged = Vec::with_capacity(labels.len() + usize::from(self.vm_id.is_some()));
        if let Some(ref vm_id) = self.vm_id {
            merged.push(("vm.id", vm_id.as_str()));
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
                (
                    k.to_string(),
                    truncate_utf8_prefix(v, self.limits.max_label_val_len),
                )
            })
            .collect();
        sanitized.sort();

        let mut series_key = String::with_capacity(64);
        series_key.push_str(&valid_name);
        for (k, v) in &sanitized {
            series_key.push(';');
            series_key.push_str(k);
            series_key.push('=');
            series_key.push_str(v);
        }

        if !cache.known_series.contains(&series_key) {
            if cache.known_series.len() >= self.limits.max_unique_series {
                self.dropped_series.fetch_add(1, Ordering::Relaxed);
                return Some((valid_name, vec![KeyValue::new("overflow", "true")]));
            }
            cache.known_series.insert(series_key);
        }

        let kvs = sanitized
            .into_iter()
            .map(|(k, v)| KeyValue::new(k, v))
            .collect();
        Some((valid_name, kvs))
    }

    fn build_span_attributes(&self, labels: &[(&str, &str)]) -> Vec<KeyValue> {
        let mut merged = Vec::with_capacity(labels.len() + usize::from(self.vm_id.is_some()));
        if let Some(ref vm_id) = self.vm_id {
            merged.push(("vm.id", vm_id.as_str()));
        }
        merged.extend_from_slice(labels);

        let effective = if merged.len() > self.limits.max_label_pairs {
            &merged[..self.limits.max_label_pairs]
        } else {
            &merged[..]
        };

        let mut pairs: Vec<(String, String)> = effective
            .iter()
            .map(|(k, v)| {
                (
                    k.to_string(),
                    truncate_utf8_prefix(v, self.limits.max_label_val_len),
                )
            })
            .collect();
        pairs.sort();
        pairs
            .into_iter()
            .map(|(k, v)| KeyValue::new(k, v))
            .collect()
    }
}

impl TelemetrySink for UpstreamOtelSink {
    fn counter_add(&self, name: &str, delta: u64, labels: &[(&str, &str)]) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        let mut cache = self.instruments.lock().unwrap();
        if let Some((valid_name, attrs)) = self.prepare_metric_attributes(&mut cache, name, labels) {
            let counter = cache
                .counters
                .entry(valid_name.clone())
                .or_insert_with(|| self.meter.u64_counter(valid_name).build())
                .clone();
            drop(cache);
            counter.add(delta, &attrs);
            self.emitted_samples.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        let mut cache = self.instruments.lock().unwrap();
        if let Some((valid_name, attrs)) = self.prepare_metric_attributes(&mut cache, name, labels) {
            let gauge = cache
                .gauges
                .entry(valid_name.clone())
                .or_insert_with(|| self.meter.f64_gauge(valid_name).build())
                .clone();
            drop(cache);
            gauge.record(value, &attrs);
            self.emitted_samples.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn histogram_record(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        let mut cache = self.instruments.lock().unwrap();
        if let Some((valid_name, attrs)) = self.prepare_metric_attributes(&mut cache, name, labels) {
            let histogram = cache
                .histograms
                .entry(valid_name.clone())
                .or_insert_with(|| self.meter.f64_histogram(valid_name).build())
                .clone();
            drop(cache);
            histogram.record(value, &attrs);
            self.emitted_samples.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn span_enter(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
        if !self.enabled.load(Ordering::Relaxed) {
            return 0;
        }
        let span_id = self.next_span_id.fetch_add(1, Ordering::Relaxed);
        let parent_cx = ACTIVE_UPSTREAM_SPAN_STACK
            .with(|stack| {
                stack
                    .borrow()
                    .iter()
                    .rev()
                    .find(|(sid, _, _)| *sid == self.sink_id)
                    .map(|(_, _, cx)| cx.clone())
            })
            .unwrap_or_default();

        let attrs = self.build_span_attributes(labels);
        let span = self
            .tracer
            .span_builder(name.to_string())
            .with_kind(SpanKind::Internal)
            .with_attributes(attrs)
            .start_with_context(&self.tracer, &parent_cx);

        let cx_with_span = parent_cx.with_span(span);
        ACTIVE_UPSTREAM_SPAN_STACK.with(|stack| {
            stack
                .borrow_mut()
                .push((self.sink_id, span_id, cx_with_span.clone()));
        });
        self.open_spans.lock().unwrap().insert(span_id, cx_with_span);
        self.emitted_samples.fetch_add(1, Ordering::Relaxed);
        span_id
    }

    fn span_exit(&self, span_id: u64, status: SpanStatus) {
        if span_id == 0 {
            return;
        }
        ACTIVE_UPSTREAM_SPAN_STACK.with(|stack| {
            let mut s = stack.borrow_mut();
            if s.last().map(|(sid, id, _)| (*sid, *id)) == Some((self.sink_id, span_id)) {
                s.pop();
            } else if let Some(pos) = s
                .iter()
                .rposition(|(sid, id, _)| *sid == self.sink_id && *id == span_id)
            {
                s.remove(pos);
            }
        });

        let maybe_cx = self.open_spans.lock().unwrap().remove(&span_id);
        if let Some(cx) = maybe_cx {
            let span_ref = cx.span();
            match status {
                SpanStatus::Success | SpanStatus::Ok => span_ref.set_status(Status::Ok),
                SpanStatus::Fail | SpanStatus::Error | SpanStatus::Die | SpanStatus::Interrupt => {
                    span_ref.set_status(Status::error(status.as_str()));
                }
            }
            span_ref.end();
        }
    }

    fn flush(&self) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        self.flush_count.fetch_add(1, Ordering::Relaxed);
        if self.fail_on_flush.load(Ordering::Relaxed) {
            self.flush_failures.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let trace_res = self.tracer_provider.force_flush();
        let metric_res = self.meter_provider.force_flush();
        if trace_res.is_err() || metric_res.is_err() {
            self.flush_failures.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Scoped wrapper around [`UpstreamOtelSink`] attaching immutable per-VM/tenant labels.
#[derive(Debug, Clone)]
pub struct ScopedUpstreamOtelSink {
    inner: Arc<UpstreamOtelSink>,
    scope_labels: Vec<(String, String)>,
}

impl ScopedUpstreamOtelSink {
    fn merge<'a>(&'a self, labels: &[(&'a str, &'a str)]) -> Vec<(&'a str, &'a str)> {
        let mut res = Vec::with_capacity(self.scope_labels.len() + labels.len());
        for (k, v) in &self.scope_labels {
            res.push((k.as_str(), v.as_str()));
        }
        res.extend_from_slice(labels);
        res
    }
}

impl TelemetrySink for ScopedUpstreamOtelSink {
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

fn build_resource(service_name: &str, vm_id: Option<&str>) -> Resource {
    let mut attrs = vec![KeyValue::new("service.version", "3.5.8")];
    if let Some(id) = vm_id {
        attrs.push(KeyValue::new("vm.id", id.to_string()));
    }
    Resource::builder_empty()
        .with_service_name(service_name.to_string())
        .with_attributes(attrs)
        .build()
}

/// Ensures instrument names conform to the OpenTelemetry Metrics specification
/// (`^[A-Za-z][A-Za-z0-9_.\-/]{0,254}$`). Valid names are returned unchanged.
pub fn sanitize_otel_instrument_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len().min(255));
    for (idx, c) in name.chars().enumerate() {
        if out.len() >= 255 {
            break;
        }
        if idx == 0 {
            if c.is_ascii_alphabetic() {
                out.push(c);
            } else {
                out.push('m');
                if (c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-' | '/'))
                    && out.len() < 255
                {
                    out.push(c);
                }
            }
        } else if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-' | '/') {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push_str("lust_metric");
    }
    out
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

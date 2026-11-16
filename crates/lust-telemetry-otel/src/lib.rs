//! Zero-dependency OpenTelemetry (OTLP) JSON wire format exporter for Lust Runtime.

use std::cell::RefCell;
use std::collections::HashMap;
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

    #[inline]
    pub fn to_otlp_code(&self) -> u32 {
        match self {
            Self::Success | Self::Ok => 1, // STATUS_CODE_OK
            Self::Fail | Self::Error | Self::Die | Self::Interrupt => 2, // STATUS_CODE_ERROR
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
pub struct OtelSpan {
    pub span_id_numeric: u64,
    pub trace_id_hex: String,
    pub span_id_hex: String,
    pub parent_span_id_hex: Option<String>,
    pub name: String,
    pub kind: u32,
    pub start_time_unix_nano: u64,
    pub end_time_unix_nano: u64,
    pub attributes: Vec<(String, String)>,
    pub status_code: u32,
    pub status_message: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OtelCounterPoint {
    pub name: String,
    pub value: u64,
    pub time_unix_nano: u64,
    pub attributes: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OtelGaugePoint {
    pub name: String,
    pub value: f64,
    pub time_unix_nano: u64,
    pub attributes: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OtelHistogramPoint {
    pub name: String,
    pub count: u64,
    pub sum: f64,
    pub bucket_counts: Vec<u64>,
    pub explicit_bounds: Vec<f64>,
    pub time_unix_nano: u64,
    pub attributes: Vec<(String, String)>,
}

#[derive(Debug, Default)]
struct OtelStorage {
    spans: Vec<OtelSpan>,
    open_spans: HashMap<u64, OtelSpan>,
    counters: HashMap<(String, Vec<(String, String)>), u64>,
    gauges: HashMap<(String, Vec<(String, String)>), f64>,
    histograms: HashMap<(String, Vec<(String, String)>), (u64, f64, Vec<u64>)>, // count, sum, bucket counts
}

#[derive(Debug)]
pub struct OtelSink {
    service_name: String,
    service_version: String,
    vm_id: Option<String>,
    storage: Mutex<OtelStorage>,
    next_span_id: AtomicU64,
    clock_seq: AtomicU64,
    flush_count: AtomicU64,
    flush_failures: AtomicU64,
    fail_on_flush: AtomicBool,
    explicit_histogram_bounds: Vec<f64>,
}

impl OtelSink {
    pub fn new() -> Self {
        Self {
            service_name: "lust-service".to_string(),
            service_version: "3.5.8".to_string(),
            vm_id: None,
            storage: Mutex::new(OtelStorage::default()),
            next_span_id: AtomicU64::new(1),
            clock_seq: AtomicU64::new(1_731_688_000_000_000_000), // Pinned base synthetic time
            flush_count: AtomicU64::new(0),
            flush_failures: AtomicU64::new(0),
            fail_on_flush: AtomicBool::new(false),
            explicit_histogram_bounds: vec![0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        }
    }

    pub fn with_service_name(mut self, name: &str) -> Self {
        self.service_name = name.to_string();
        self
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
        let mut storage = self.storage.lock().unwrap();
        storage.spans.clear();
        storage.open_spans.clear();
        storage.counters.clear();
        storage.gauges.clear();
        storage.histograms.clear();
        self.flush_count.store(0, Ordering::Relaxed);
        self.flush_failures.store(0, Ordering::Relaxed);
        ACTIVE_SPAN_STACK.with(|s| s.borrow_mut().clear());
    }

    fn next_time(&self) -> u64 {
        self.clock_seq.fetch_add(1_000_000, Ordering::Relaxed)
    }

    fn merge_labels(&self, labels: &[(&str, &str)]) -> Vec<(String, String)> {
        let mut res = Vec::new();
        if let Some(ref vm_id) = self.vm_id {
            res.push(("vm.id".to_string(), vm_id.clone()));
        }
        for (k, v) in labels {
            res.push((k.to_string(), v.to_string()));
        }
        res.sort();
        res
    }

    pub fn spans(&self) -> Vec<OtelSpan> {
        let storage = self.storage.lock().unwrap();
        storage.spans.clone()
    }

    pub fn scoped(self: &Arc<Self>, scope_labels: &[(&str, &str)]) -> ScopedOtelSink {
        ScopedOtelSink {
            inner: Arc::clone(self),
            scope_labels: scope_labels
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    pub fn export_traces_json(&self) -> String {
        let storage = self.storage.lock().unwrap();
        let mut json = String::from("{\n  \"resourceSpans\": [\n    {\n      \"resource\": {\n        \"attributes\": [\n");
        json.push_str(&format!(
            "          {{ \"key\": \"service.name\", \"value\": {{ \"stringValue\": \"{}\" }} }},\n",
            escape_json_str(&self.service_name)
        ));
        json.push_str(&format!(
            "          {{ \"key\": \"service.version\", \"value\": {{ \"stringValue\": \"{}\" }} }}",
            escape_json_str(&self.service_version)
        ));
        if let Some(ref vm_id) = self.vm_id {
            json.push_str(",\n");
            json.push_str(&format!(
                "          {{ \"key\": \"vm.id\", \"value\": {{ \"stringValue\": \"{}\" }} }}",
                escape_json_str(vm_id)
            ));
        }
        json.push_str("\n        ]\n      },\n      \"scopeSpans\": [\n        {\n          \"scope\": {\n            \"name\": \"lust-runtime\",\n            \"version\": \"3.5.8\"\n          },\n          \"spans\": [\n");

        for (i, span) in storage.spans.iter().enumerate() {
            if i > 0 {
                json.push_str(",\n");
            }
            json.push_str("            {\n");
            json.push_str(&format!(
                "              \"traceId\": \"{}\",\n",
                span.trace_id_hex
            ));
            json.push_str(&format!(
                "              \"spanId\": \"{}\",\n",
                span.span_id_hex
            ));
            if let Some(ref parent_hex) = span.parent_span_id_hex {
                json.push_str(&format!(
                    "              \"parentSpanId\": \"{}\",\n",
                    parent_hex
                ));
            }
            json.push_str(&format!(
                "              \"name\": \"{}\",\n",
                escape_json_str(&span.name)
            ));
            json.push_str(&format!("              \"kind\": {},\n", span.kind));
            json.push_str(&format!(
                "              \"startTimeUnixNano\": \"{}\",\n",
                span.start_time_unix_nano
            ));
            json.push_str(&format!(
                "              \"endTimeUnixNano\": \"{}\",\n",
                span.end_time_unix_nano
            ));
            json.push_str("              \"attributes\": [\n");
            for (j, (k, v)) in span.attributes.iter().enumerate() {
                if j > 0 {
                    json.push_str(",\n");
                }
                json.push_str(&format!(
                    "                {{ \"key\": \"{}\", \"value\": {{ \"stringValue\": \"{}\" }} }}",
                    escape_json_str(k),
                    escape_json_str(v)
                ));
            }
            json.push_str("\n              ],\n");
            json.push_str("              \"status\": {\n");
            json.push_str(&format!(
                "                \"code\": {}\n",
                span.status_code
            ));
            json.push_str("              }\n");
            json.push_str("            }");
        }

        json.push_str("\n          ]\n        }\n      ]\n    }\n  ]\n}");
        json
    }

    pub fn export_metrics_json(&self) -> String {
        let storage = self.storage.lock().unwrap();
        let mut json = String::from("{\n  \"resourceMetrics\": [\n    {\n      \"resource\": {\n        \"attributes\": [\n");
        json.push_str(&format!(
            "          {{ \"key\": \"service.name\", \"value\": {{ \"stringValue\": \"{}\" }} }},\n",
            escape_json_str(&self.service_name)
        ));
        json.push_str(&format!(
            "          {{ \"key\": \"service.version\", \"value\": {{ \"stringValue\": \"{}\" }} }}",
            escape_json_str(&self.service_version)
        ));
        if let Some(ref vm_id) = self.vm_id {
            json.push_str(",\n");
            json.push_str(&format!(
                "          {{ \"key\": \"vm.id\", \"value\": {{ \"stringValue\": \"{}\" }} }}",
                escape_json_str(vm_id)
            ));
        }
        json.push_str("\n        ]\n      },\n      \"scopeMetrics\": [\n        {\n          \"scope\": {\n            \"name\": \"lust-runtime\",\n            \"version\": \"3.5.8\"\n          },\n          \"metrics\": [\n");

        let mut metric_idx = 0;
        // Counters
        for ((name, labels), val) in &storage.counters {
            if metric_idx > 0 {
                json.push_str(",\n");
            }
            metric_idx += 1;
            json.push_str("            {\n");
            json.push_str(&format!(
                "              \"name\": \"{}\",\n",
                escape_json_str(name)
            ));
            json.push_str("              \"unit\": \"1\",\n");
            json.push_str("              \"sum\": {\n");
            json.push_str("                \"isMonotonic\": true,\n");
            json.push_str("                \"aggregationTemporality\": 2,\n");
            json.push_str("                \"dataPoints\": [\n");
            json.push_str("                  {\n");
            json.push_str(&format!("                    \"asInt\": \"{val}\",\n"));
            json.push_str("                    \"attributes\": [\n");
            for (j, (k, v)) in labels.iter().enumerate() {
                if j > 0 {
                    json.push_str(",\n");
                }
                json.push_str(&format!(
                    "                      {{ \"key\": \"{}\", \"value\": {{ \"stringValue\": \"{}\" }} }}",
                    escape_json_str(k),
                    escape_json_str(v)
                ));
            }
            json.push_str("\n                    ]\n");
            json.push_str("                  }\n");
            json.push_str("                ]\n");
            json.push_str("              }\n");
            json.push_str("            }");
        }

        // Gauges
        for ((name, labels), val) in &storage.gauges {
            if metric_idx > 0 {
                json.push_str(",\n");
            }
            metric_idx += 1;
            json.push_str("            {\n");
            json.push_str(&format!(
                "              \"name\": \"{}\",\n",
                escape_json_str(name)
            ));
            json.push_str("              \"unit\": \"1\",\n");
            json.push_str("              \"gauge\": {\n");
            json.push_str("                \"dataPoints\": [\n");
            json.push_str("                  {\n");
            json.push_str(&format!("                    \"asDouble\": {val},\n"));
            json.push_str("                    \"attributes\": [\n");
            for (j, (k, v)) in labels.iter().enumerate() {
                if j > 0 {
                    json.push_str(",\n");
                }
                json.push_str(&format!(
                    "                      {{ \"key\": \"{}\", \"value\": {{ \"stringValue\": \"{}\" }} }}",
                    escape_json_str(k),
                    escape_json_str(v)
                ));
            }
            json.push_str("\n                    ]\n");
            json.push_str("                  }\n");
            json.push_str("                ]\n");
            json.push_str("              }\n");
            json.push_str("            }");
        }

        // Histograms
        for ((name, labels), (count, sum, bucket_counts)) in &storage.histograms {
            if metric_idx > 0 {
                json.push_str(",\n");
            }
            metric_idx += 1;
            json.push_str("            {\n");
            json.push_str(&format!(
                "              \"name\": \"{}\",\n",
                escape_json_str(name)
            ));
            json.push_str("              \"unit\": \"1\",\n");
            json.push_str("              \"histogram\": {\n");
            json.push_str("                \"aggregationTemporality\": 2,\n");
            json.push_str("                \"dataPoints\": [\n");
            json.push_str("                  {\n");
            json.push_str(&format!("                    \"count\": \"{count}\",\n"));
            json.push_str(&format!("                    \"sum\": {sum},\n"));
            json.push_str(&format!(
                "                    \"bucketCounts\": {:?},\n",
                bucket_counts.iter().map(|c| c.to_string()).collect::<Vec<_>>()
            ));
            json.push_str(&format!(
                "                    \"explicitBounds\": {:?},\n",
                self.explicit_histogram_bounds
            ));
            json.push_str("                    \"attributes\": [\n");
            for (j, (k, v)) in labels.iter().enumerate() {
                if j > 0 {
                    json.push_str(",\n");
                }
                json.push_str(&format!(
                    "                      {{ \"key\": \"{}\", \"value\": {{ \"stringValue\": \"{}\" }} }}",
                    escape_json_str(k),
                    escape_json_str(v)
                ));
            }
            json.push_str("\n                    ]\n");
            json.push_str("                  }\n");
            json.push_str("                ]\n");
            json.push_str("              }\n");
            json.push_str("            }");
        }

        json.push_str("\n          ]\n        }\n      ]\n    }\n  ]\n}");
        json
    }

    pub fn export_all_json(&self) -> String {
        format!(
            "{{\n  \"traces\": {},\n  \"metrics\": {}\n}}",
            self.export_traces_json(),
            self.export_metrics_json()
        )
    }
}

impl Default for OtelSink {
    fn default() -> Self {
        Self::new()
    }
}

impl TelemetrySink for OtelSink {
    fn counter_add(&self, name: &str, delta: u64, labels: &[(&str, &str)]) {
        let merged = self.merge_labels(labels);
        let mut storage = self.storage.lock().unwrap();
        *storage.counters.entry((name.to_string(), merged)).or_insert(0) += delta;
    }

    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        let merged = self.merge_labels(labels);
        let mut storage = self.storage.lock().unwrap();
        storage.gauges.insert((name.to_string(), merged), value);
    }

    fn histogram_record(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        let merged = self.merge_labels(labels);
        let mut storage = self.storage.lock().unwrap();
        let bounds = &self.explicit_histogram_bounds;
        let entry = storage
            .histograms
            .entry((name.to_string(), merged))
            .or_insert_with(|| (0, 0.0, vec![0; bounds.len() + 1]));
        entry.0 += 1;
        entry.1 += value;
        let mut placed = false;
        for (i, &bound) in bounds.iter().enumerate() {
            if value <= bound {
                entry.2[i] += 1;
                placed = true;
                break;
            }
        }
        if !placed {
            let last_idx = bounds.len();
            entry.2[last_idx] += 1;
        }
    }

    fn span_enter(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
        let span_id_numeric = self.next_span_id.fetch_add(1, Ordering::Relaxed);
        let start_time_unix_nano = self.next_time();

        let explicit_parent = labels.iter().find_map(|(k, v)| {
            if *k == "parent_span_id" || *k == "parent_id" {
                v.parse::<u64>().ok()
            } else {
                None
            }
        });
        let parent_numeric = explicit_parent.or_else(|| {
            ACTIVE_SPAN_STACK.with(|s| s.borrow().last().copied())
        });

        ACTIVE_SPAN_STACK.with(|s| s.borrow_mut().push(span_id_numeric));

        // Format trace_id as 32 hex chars, span_id as 16 hex chars
        let trace_id_hex = format!("{:032x}", 0x1000_0000_u64);
        let span_id_hex = format!("{:016x}", span_id_numeric);
        let parent_span_id_hex = parent_numeric.map(|p| format!("{:016x}", p));

        let attributes = self.merge_labels(labels);

        let span = OtelSpan {
            span_id_numeric,
            trace_id_hex,
            span_id_hex,
            parent_span_id_hex,
            name: name.to_string(),
            kind: 1, // INTERNAL
            start_time_unix_nano,
            end_time_unix_nano: 0,
            attributes,
            status_code: 1,
            status_message: None,
        };

        self.storage.lock().unwrap().open_spans.insert(span_id_numeric, span);
        span_id_numeric
    }

    fn span_exit(&self, span_id: u64, status: SpanStatus) {
        let end_time_unix_nano = self.next_time();
        ACTIVE_SPAN_STACK.with(|s| {
            let mut stack = s.borrow_mut();
            if stack.last() == Some(&span_id) {
                stack.pop();
            } else if let Some(pos) = stack.iter().rposition(|&id| id == span_id) {
                stack.remove(pos);
            }
        });

        let mut storage = self.storage.lock().unwrap();
        if let Some(mut span) = storage.open_spans.remove(&span_id) {
            span.end_time_unix_nano = end_time_unix_nano;
            span.status_code = status.to_otlp_code();
            if span.status_code == 2 {
                span.status_message = Some(status.as_str().to_string());
            }
            storage.spans.push(span);
        }
    }

    fn flush(&self) {
        self.flush_count.fetch_add(1, Ordering::Relaxed);
        if self.fail_on_flush.load(Ordering::Relaxed) {
            self.flush_failures.fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub fn escape_json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct ScopedOtelSink {
    inner: Arc<OtelSink>,
    scope_labels: Vec<(String, String)>,
}

impl ScopedOtelSink {
    fn merge<'a>(&'a self, labels: &[(&'a str, &'a str)]) -> Vec<(&'a str, &'a str)> {
        let mut res = Vec::with_capacity(self.scope_labels.len() + labels.len());
        for (k, v) in &self.scope_labels {
            res.push((k.as_str(), v.as_str()));
        }
        res.extend_from_slice(labels);
        res
    }
}

impl TelemetrySink for ScopedOtelSink {
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
    fn test_nested_spans_and_otlp_json() {
        let sink = OtelSink::new().with_service_name("order-service");
        let parent_id = sink.span_enter("parent_operation", &[("tier", "api")]);
        let child_id = sink.span_enter("child_database_query", &[("db.system", "postgres")]);
        sink.span_exit(child_id, SpanStatus::Success);
        sink.span_exit(parent_id, SpanStatus::Success);

        let spans = sink.spans();
        assert_eq!(spans.len(), 2);
        // Child finished first
        assert_eq!(spans[0].name, "child_database_query");
        assert_eq!(
            spans[0].parent_span_id_hex,
            Some(format!("{:016x}", parent_id))
        );
        assert_eq!(spans[0].status_code, 1);

        assert_eq!(spans[1].name, "parent_operation");
        assert_eq!(spans[1].parent_span_id_hex, None);

        let json = sink.export_traces_json();
        assert!(json.contains("\"service.name\""));
        assert!(json.contains("order-service"));
        assert!(json.contains("\"parent_operation\""));
        assert!(json.contains("\"child_database_query\""));
        assert!(json.contains("\"parentSpanId\""));
    }

    #[test]
    fn test_metrics_json_export() {
        let sink = OtelSink::new().with_vm_id("vm-alpha");
        sink.counter_add("gas_consumed", 450, &[("action", "exec")]);
        sink.gauge_set("mem_bytes", 8192.0, &[]);
        sink.histogram_record("latency", 0.045, &[]);

        let json = sink.export_metrics_json();
        assert!(json.contains("\"gas_consumed\""));
        assert!(json.contains("\"asInt\": \"450\""));
        assert!(json.contains("\"mem_bytes\""));
        assert!(json.contains("\"asDouble\": 8192"));
        assert!(json.contains("\"latency\""));
        assert!(json.contains("\"vm.id\""));
    }

    #[test]
    fn test_escaping_in_json() {
        assert_eq!(escape_json_str("hello \"world\""), "hello \\\"world\\\"");
        assert_eq!(escape_json_str("line1\nline2"), "line1\\nline2");
        assert_eq!(escape_json_str("tab\there"), "tab\\there");
        assert_eq!(escape_json_str("path\\file"), "path\\\\file");
    }

    #[test]
    fn test_flush_and_failure() {
        let sink = OtelSink::new();
        sink.flush();
        assert_eq!(sink.flush_count(), 1);
        sink.set_fail_on_flush(true);
        sink.flush();
        assert_eq!(sink.flush_count(), 2);
        assert_eq!(sink.flush_failures(), 1);
    }
}

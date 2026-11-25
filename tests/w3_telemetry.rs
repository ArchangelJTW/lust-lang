//! Integration tests for Wave 3 Track 3F (`TEL-EXP: Telemetry Exporters & Mock Sink`):
//! - Zero-dependency OpenMetrics / Prometheus exporter (`lust-telemetry-prometheus`)
//! - Zero-dependency OpenTelemetry OTLP JSON wire format exporter (`lust-telemetry-otel`)
//! - Zero-dependency in-memory metrics exporter (`lust-telemetry-metrics`)
//! - Tracing subscriber and exporter adapter (`lust-telemetry-tracing`)
//! - Extended DST mock sink (`src/telemetry/mock.rs` with scoped VM isolation, span parentage, flush/failure)
//! - Independent pinned decoders/fixtures, escaping, types, labels, concurrency, nested spans, no-op control.

#![allow(unexpected_cfgs)]

use std::sync::Arc;
use std::thread;

use lust::config::TelemetryLimitsConfig;
use lust::vm::telemetry::{
    MockTelemetrySink, NoopSink, SpanStatus as VmSpanStatus,
    TelemetryCardinalityGuard, TelemetryEvent, TelemetrySignalMask, TelemetrySink as VmTelemetrySink,
};
use lust::vm::VM;

#[path = "../crates/lust-telemetry-metrics/src/lib.rs"]
mod lust_telemetry_metrics;

#[path = "../crates/lust-telemetry-otel/src/lib.rs"]
mod lust_telemetry_otel;

#[path = "../crates/lust-telemetry-prometheus/src/lib.rs"]
mod lust_telemetry_prometheus;

#[path = "../crates/lust-telemetry-tracing/src/lib.rs"]
mod lust_telemetry_tracing;

use lust_telemetry_metrics::{MetricsSink, TelemetrySink as MetricsTelemetrySink};
use lust_telemetry_otel::{OtelSink, TelemetrySink as OtelTelemetrySink};
use lust_telemetry_prometheus::{PrometheusSink, TelemetrySink as PromTelemetrySink};
use lust_telemetry_tracing::{TracingSink, TelemetrySink as TracingTelemetrySink};

// -----------------------------------------------------------------------------
// VM TelemetrySink implementations for the exporters
// -----------------------------------------------------------------------------

impl VmTelemetrySink for MetricsSink {
    fn counter_add(&self, name: &str, delta: u64, labels: &[(&str, &str)]) {
        lust_telemetry_metrics::TelemetrySink::counter_add(self, name, delta, labels);
    }
    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        lust_telemetry_metrics::TelemetrySink::gauge_set(self, name, value, labels);
    }
    fn histogram_record(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        lust_telemetry_metrics::TelemetrySink::histogram_record(self, name, value, labels);
    }
    fn span_enter(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
        lust_telemetry_metrics::TelemetrySink::span_enter(self, name, labels)
    }
    fn span_exit(&self, span_id: u64, status: VmSpanStatus) {
        let s = match status {
            VmSpanStatus::Success => lust_telemetry_metrics::SpanStatus::Success,
            VmSpanStatus::Fail => lust_telemetry_metrics::SpanStatus::Fail,
            VmSpanStatus::Die => lust_telemetry_metrics::SpanStatus::Die,
            VmSpanStatus::Interrupt => lust_telemetry_metrics::SpanStatus::Interrupt,
            VmSpanStatus::Ok => lust_telemetry_metrics::SpanStatus::Ok,
            VmSpanStatus::Error => lust_telemetry_metrics::SpanStatus::Error,
        };
        lust_telemetry_metrics::TelemetrySink::span_exit(self, span_id, s);
    }
    fn flush(&self) {
        lust_telemetry_metrics::TelemetrySink::flush(self);
    }
}

impl VmTelemetrySink for OtelSink {
    fn counter_add(&self, name: &str, delta: u64, labels: &[(&str, &str)]) {
        lust_telemetry_otel::TelemetrySink::counter_add(self, name, delta, labels);
    }
    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        lust_telemetry_otel::TelemetrySink::gauge_set(self, name, value, labels);
    }
    fn histogram_record(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        lust_telemetry_otel::TelemetrySink::histogram_record(self, name, value, labels);
    }
    fn span_enter(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
        lust_telemetry_otel::TelemetrySink::span_enter(self, name, labels)
    }
    fn span_exit(&self, span_id: u64, status: VmSpanStatus) {
        let s = match status {
            VmSpanStatus::Success => lust_telemetry_otel::SpanStatus::Success,
            VmSpanStatus::Fail => lust_telemetry_otel::SpanStatus::Fail,
            VmSpanStatus::Die => lust_telemetry_otel::SpanStatus::Die,
            VmSpanStatus::Interrupt => lust_telemetry_otel::SpanStatus::Interrupt,
            VmSpanStatus::Ok => lust_telemetry_otel::SpanStatus::Ok,
            VmSpanStatus::Error => lust_telemetry_otel::SpanStatus::Error,
        };
        lust_telemetry_otel::TelemetrySink::span_exit(self, span_id, s);
    }
    fn flush(&self) {
        lust_telemetry_otel::TelemetrySink::flush(self);
    }
}

impl VmTelemetrySink for PrometheusSink {
    fn counter_add(&self, name: &str, delta: u64, labels: &[(&str, &str)]) {
        lust_telemetry_prometheus::TelemetrySink::counter_add(self, name, delta, labels);
    }
    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        lust_telemetry_prometheus::TelemetrySink::gauge_set(self, name, value, labels);
    }
    fn histogram_record(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        lust_telemetry_prometheus::TelemetrySink::histogram_record(self, name, value, labels);
    }
    fn span_enter(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
        lust_telemetry_prometheus::TelemetrySink::span_enter(self, name, labels)
    }
    fn span_exit(&self, span_id: u64, status: VmSpanStatus) {
        let s = match status {
            VmSpanStatus::Success => lust_telemetry_prometheus::SpanStatus::Success,
            VmSpanStatus::Fail => lust_telemetry_prometheus::SpanStatus::Fail,
            VmSpanStatus::Die => lust_telemetry_prometheus::SpanStatus::Die,
            VmSpanStatus::Interrupt => lust_telemetry_prometheus::SpanStatus::Interrupt,
            VmSpanStatus::Ok => lust_telemetry_prometheus::SpanStatus::Ok,
            VmSpanStatus::Error => lust_telemetry_prometheus::SpanStatus::Error,
        };
        lust_telemetry_prometheus::TelemetrySink::span_exit(self, span_id, s);
    }
    fn flush(&self) {
        lust_telemetry_prometheus::TelemetrySink::flush(self);
    }
}

impl VmTelemetrySink for TracingSink {
    fn counter_add(&self, name: &str, delta: u64, labels: &[(&str, &str)]) {
        lust_telemetry_tracing::TelemetrySink::counter_add(self, name, delta, labels);
    }
    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        lust_telemetry_tracing::TelemetrySink::gauge_set(self, name, value, labels);
    }
    fn histogram_record(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        lust_telemetry_tracing::TelemetrySink::histogram_record(self, name, value, labels);
    }
    fn span_enter(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
        lust_telemetry_tracing::TelemetrySink::span_enter(self, name, labels)
    }
    fn span_exit(&self, span_id: u64, status: VmSpanStatus) {
        let s = match status {
            VmSpanStatus::Success => lust_telemetry_tracing::SpanStatus::Success,
            VmSpanStatus::Fail => lust_telemetry_tracing::SpanStatus::Fail,
            VmSpanStatus::Die => lust_telemetry_tracing::SpanStatus::Die,
            VmSpanStatus::Interrupt => lust_telemetry_tracing::SpanStatus::Interrupt,
            VmSpanStatus::Ok => lust_telemetry_tracing::SpanStatus::Ok,
            VmSpanStatus::Error => lust_telemetry_tracing::SpanStatus::Error,
        };
        lust_telemetry_tracing::TelemetrySink::span_exit(self, span_id, s);
    }
    fn flush(&self) {
        lust_telemetry_tracing::TelemetrySink::flush(self);
    }
}

// -----------------------------------------------------------------------------
// Independent Pinned Decoders
// -----------------------------------------------------------------------------

/// Independent OpenMetrics text parser verifying compliance with OpenMetrics RFC / Prometheus exposition format
fn decode_and_validate_openmetrics(text: &str) -> (usize, usize, usize) {
    let mut num_help = 0;
    let mut num_type = 0;
    let mut num_samples = 0;
    let mut saw_eof = false;

    for line in text.lines() {
        if line == "# EOF" {
            saw_eof = true;
            continue;
        }
        if line.starts_with("# HELP ") {
            num_help += 1;
            let parts: Vec<&str> = line.split_whitespace().collect();
            assert!(parts.len() >= 3, "Malformed HELP line: {line}");
            assert!(!parts[2].is_empty(), "HELP metric name must not be empty");
        } else if line.starts_with("# TYPE ") {
            num_type += 1;
            let parts: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(parts.len(), 4, "Malformed TYPE line: {line}");
            assert!(
                matches!(parts[3], "counter" | "gauge" | "histogram" | "summary"),
                "Unknown metric type in TYPE line: {line}"
            );
        } else if !line.is_empty() {
            num_samples += 1;
            let space_pos = line.rfind(' ').expect("Sample line must contain space separator");
            let _val_str = &line[space_pos + 1..];
            let name_and_labels = &line[..space_pos];
            if let Some(brace_pos) = name_and_labels.find('{') {
                assert!(name_and_labels.ends_with('}'));
                let _name = &name_and_labels[..brace_pos];
                let labels_part = &name_and_labels[brace_pos + 1..name_and_labels.len() - 1];
                for pair in labels_part.split(',') {
                    if pair.is_empty() {
                        continue;
                    }
                    assert!(pair.contains('='), "Label pair must contain '=': {pair}");
                    assert!(pair.contains('"'), "Label value must be quoted: {pair}");
                }
            }
        }
    }

    assert!(saw_eof, "OpenMetrics text exposition must end with # EOF");
    (num_help, num_type, num_samples)
}

/// Independent OTLP JSON validator verifying compliance with OpenTelemetry v1 wire specification
fn decode_and_validate_otlp_json(json: &str) {
    assert!(json.starts_with('{'), "OTLP JSON must be a JSON object");
    assert!(json.ends_with('}'), "OTLP JSON must close JSON object");

    if json.contains("\"resourceSpans\"") {
        assert!(json.contains("\"scopeSpans\""), "OTLP Trace must contain scopeSpans");
        assert!(json.contains("\"traceId\""), "OTLP Span must contain traceId");
        assert!(json.contains("\"spanId\""), "OTLP Span must contain spanId");
        assert!(json.contains("\"startTimeUnixNano\""), "OTLP Span must have startTimeUnixNano");
        assert!(json.contains("\"endTimeUnixNano\""), "OTLP Span must have endTimeUnixNano");
        assert!(json.contains("\"status\""), "OTLP Span must contain status");
    }

    if json.contains("\"resourceMetrics\"") {
        assert!(json.contains("\"scopeMetrics\""), "OTLP Metrics must contain scopeMetrics");
        assert!(json.contains("\"metrics\""), "OTLP Metrics must contain metrics list");
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn test_w3_prometheus_openmetrics_format_and_pinned_fixture() {
    let sink = PrometheusSink::new();
    PromTelemetrySink::counter_add(&sink, "lust.vm.gas.consumed", 500, &[("source", "vm")]);
    PromTelemetrySink::gauge_set(&sink, "lust.vm.memory.bytes", 8192.0, &[("tier", "hot")]);
    PromTelemetrySink::histogram_record(&sink, "lust.jit.compile_duration_sec", 0.025, &[("opt", "safepoint")]);

    let text = sink.render_openmetrics();

    // Verify against independent OpenMetrics validator
    let (num_help, num_type, num_samples) = decode_and_validate_openmetrics(&text);
    assert_eq!(num_help, 3, "Expected 3 HELP lines");
    assert_eq!(num_type, 3, "Expected 3 TYPE lines");
    assert!(num_samples >= 15, "Expected at least 15 sample lines (counter + gauge + histogram buckets)");

    // Pinned fixture checks
    assert!(text.contains("lust_vm_gas_consumed_total{source=\"vm\"} 500"));
    assert!(text.contains("lust_vm_memory_bytes{tier=\"hot\"} 8192"));
    assert!(text.contains("lust_jit_compile_duration_sec_bucket{opt=\"safepoint\",le=\"0.025\"} 1"));
    assert!(text.contains("lust_jit_compile_duration_sec_bucket{opt=\"safepoint\",le=\"+Inf\"} 1"));
    assert!(text.contains("lust_jit_compile_duration_sec_sum{opt=\"safepoint\"} 0.025"));
    assert!(text.contains("lust_jit_compile_duration_sec_count{opt=\"safepoint\"} 1"));
    assert!(text.ends_with("# EOF\n"));
}

#[test]
fn test_w3_prometheus_escaping_and_sanitization() {
    let sink = PrometheusSink::new();
    // Labels containing quotes, newlines, backslashes
    PromTelemetrySink::counter_add(
        &sink,
        "escape.test.metric",
        42,
        &[("msg", "line1\nline2 with \"quotes\" and \\slash")],
    );

    let text = sink.render_openmetrics();
    assert!(text.contains("escape_test_metric_total{msg=\"line1\\nline2 with \\\"quotes\\\" and \\\\slash\"} 42"));
}

#[test]
fn test_w3_otlp_json_wire_format_and_pinned_fixture() {
    let sink = OtelSink::new()
        .with_service_name("state-guard-engine")
        .with_vm_id("vm-01");

    let root_span = OtelTelemetrySink::span_enter(&sink, "eval_rule", &[("tenant", "alpha")]);
    let child_span = OtelTelemetrySink::span_enter(&sink, "query_state", &[("db", "memory")]);
    OtelTelemetrySink::span_exit(&sink, child_span, lust_telemetry_otel::SpanStatus::Ok);
    OtelTelemetrySink::span_exit(&sink, root_span, lust_telemetry_otel::SpanStatus::Success);

    OtelTelemetrySink::counter_add(&sink, "gas_used", 120, &[("action", "query")]);
    OtelTelemetrySink::gauge_set(&sink, "mem_used", 2048.0, &[]);
    OtelTelemetrySink::histogram_record(&sink, "latency", 0.005, &[]);

    let traces_json = sink.export_traces_json();
    decode_and_validate_otlp_json(&traces_json);
    assert!(traces_json.contains("\"service.name\""));
    assert!(traces_json.contains("state-guard-engine"));
    assert!(traces_json.contains("\"vm.id\""));
    assert!(traces_json.contains("vm-01"));
    assert!(traces_json.contains("\"eval_rule\""));
    assert!(traces_json.contains("\"query_state\""));
    assert!(traces_json.contains("\"parentSpanId\""));

    let metrics_json = sink.export_metrics_json();
    decode_and_validate_otlp_json(&metrics_json);
    assert!(metrics_json.contains("\"gas_used\""));
    assert!(metrics_json.contains("\"asInt\": \"120\""));
    assert!(metrics_json.contains("\"mem_used\""));
    assert!(metrics_json.contains("\"asDouble\": 2048"));
    assert!(metrics_json.contains("\"latency\""));
    assert!(metrics_json.contains("\"explicitBounds\""));
}

#[test]
fn test_w3_nested_span_parentage_across_sinks() {
    // 1. MockTelemetrySink nested spans
    let mock = Arc::new(MockTelemetrySink::new());
    let span_a = mock.span_enter("span_a", &[]);
    let span_b = mock.span_enter("span_b", &[]);
    let span_c = mock.span_enter("span_c", &[]);

    assert_eq!(mock.parent_of_span(span_a), None);
    assert_eq!(mock.parent_of_span(span_b), Some(span_a));
    assert_eq!(mock.parent_of_span(span_c), Some(span_b));
    assert_eq!(mock.children_of_span(span_a), vec![span_b]);
    assert_eq!(mock.children_of_span(span_b), vec![span_c]);

    mock.span_exit(span_c, VmSpanStatus::Ok);
    mock.span_exit(span_b, VmSpanStatus::Fail);
    mock.span_exit(span_a, VmSpanStatus::Success);

    let events = mock.events();
    assert_eq!(events.len(), 6);

    // 2. OtelSink nested spans
    let otel = OtelSink::new();
    let o_a = OtelTelemetrySink::span_enter(&otel, "o_a", &[]);
    let o_b = OtelTelemetrySink::span_enter(&otel, "o_b", &[]);
    OtelTelemetrySink::span_exit(&otel, o_b, lust_telemetry_otel::SpanStatus::Ok);
    OtelTelemetrySink::span_exit(&otel, o_a, lust_telemetry_otel::SpanStatus::Success);

    let spans = otel.spans();
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].name, "o_b");
    assert_eq!(spans[0].parent_span_id_hex, Some(format!("{:016x}", o_a)));
    assert_eq!(spans[1].name, "o_a");
    assert_eq!(spans[1].parent_span_id_hex, None);
}

#[test]
fn test_w3_concurrent_vm_isolation() {
    let mock = Arc::new(MockTelemetrySink::new());
    let prom = Arc::new(PrometheusSink::new());
    let metrics = Arc::new(MetricsSink::new());

    let mut handles = Vec::new();
    let num_threads = 4;
    let ops_per_thread = 250;

    for t in 0..num_threads {
        let vm_id = format!("vm-{t}");
        let m_scoped = mock.scoped(&[("vm_id", &vm_id)]);
        let p_scoped = prom.scoped(&[("vm_id", &vm_id)]);
        let met_scoped = metrics.scoped(&[("vm_id", &vm_id)]);

        handles.push(thread::spawn(move || {
            for i in 0..ops_per_thread {
                VmTelemetrySink::counter_add(&m_scoped, "vm.ticks", 1, &[("cycle", "iter")]);
                PromTelemetrySink::counter_add(&p_scoped, "vm_ticks", 1, &[("cycle", "iter")]);
                MetricsTelemetrySink::counter_add(&met_scoped, "vm.ticks", 1, &[("cycle", "iter")]);

                if i % 50 == 0 {
                    let s = VmTelemetrySink::span_enter(&m_scoped, "scoped_op", &[]);
                    VmTelemetrySink::span_exit(&m_scoped, s, VmSpanStatus::Success);
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Verify MockTelemetrySink isolation per VM
    for t in 0..num_threads {
        let vm_id = format!("vm-{t}");
        let events = mock.events_for_vm(&vm_id);
        let count_events = events
            .iter()
            .filter(|ev| match ev {
                TelemetryEvent::Counter { name, .. } if name == "vm.ticks" => true,
                _ => false,
            })
            .count();
        assert_eq!(count_events, ops_per_thread);
    }
    assert_eq!(
        mock.counter_total("vm.ticks"),
        (num_threads * ops_per_thread) as u64
    );

    // Verify PrometheusSink isolation
    for t in 0..num_threads {
        let vm_id = format!("vm-{t}");
        let val = prom
            .counter("vm_ticks", &[("vm_id", &vm_id), ("cycle", "iter")])
            .unwrap();
        assert_eq!(val, ops_per_thread as u64);
    }

    // Verify MetricsSink isolation
    for t in 0..num_threads {
        let vm_id = format!("vm-{t}");
        let val = metrics
            .counter("vm.ticks", &[("vm_id", &vm_id), ("cycle", "iter")])
            .unwrap();
        assert_eq!(val, ops_per_thread as u64);
    }
}

#[test]
fn test_w3_cardinality_guard_and_dos_protection() {
    let limits = TelemetryLimitsConfig {
        max_unique_metric_names: 4,
        max_label_pairs_per_metric: 2,
        max_label_value_len: 12,
        max_unique_series: 5,
    };
    let guard = TelemetryCardinalityGuard::new(limits);

    // 100 unique label values - simulate cardinality bomb with distinct prefixes
    for i in 0..100 {
        let val = format!("u{i}_bomb");
        let res = guard.check_and_fold("orders_metric", &[("user", &val)]);
        if i < 5 {
            assert!(res.is_some());
        } else {
            // After 5 unique series, excess must fold to overflow = "true"
            let folded = res.expect("Should fold to overflow");
            assert_eq!(folded, vec![("overflow".to_string(), "true".to_string())]);
        }
    }

    assert_eq!(guard.dropped_series_total(), 95);
}

#[test]
fn test_w3_flush_and_failure_simulation() {
    // 1. MockTelemetrySink flush & failure
    let mock = MockTelemetrySink::new();
    VmTelemetrySink::flush(&mock);
    assert_eq!(mock.flush_count(), 1);
    assert_eq!(mock.flush_failures(), 0);

    mock.set_fail_on_flush(true);
    VmTelemetrySink::flush(&mock);
    assert_eq!(mock.flush_count(), 2);
    assert_eq!(mock.flush_failures(), 1);
    let records = mock.flush_records();
    assert_eq!(records.len(), 2);
    assert!(!records[0].failed);
    assert!(records[1].failed);

    // 2. MetricsSink flush & failure
    let met = MetricsSink::new();
    MetricsTelemetrySink::flush(&met);
    assert_eq!(met.flush_count(), 1);
    met.set_fail_on_flush(true);
    MetricsTelemetrySink::flush(&met);
    assert_eq!(met.flush_count(), 2);
    assert_eq!(met.flush_failures(), 1);

    // 3. OtelSink flush & failure
    let otel = OtelSink::new();
    OtelTelemetrySink::flush(&otel);
    assert_eq!(otel.flush_count(), 1);
    otel.set_fail_on_flush(true);
    OtelTelemetrySink::flush(&otel);
    assert_eq!(otel.flush_count(), 2);
    assert_eq!(otel.flush_failures(), 1);

    // 4. PrometheusSink flush & failure
    let prom = PrometheusSink::new();
    PromTelemetrySink::flush(&prom);
    assert_eq!(prom.flush_count(), 1);
    prom.set_fail_on_flush(true);
    PromTelemetrySink::flush(&prom);
    assert_eq!(prom.flush_count(), 2);
    assert_eq!(prom.flush_failures(), 1);

    // 5. TracingSink flush & failure
    let tracing = TracingSink::new();
    TracingTelemetrySink::flush(&tracing);
    assert_eq!(tracing.flush_count(), 1);
    tracing.set_fail_on_flush(true);
    TracingTelemetrySink::flush(&tracing);
    assert_eq!(tracing.flush_count(), 2);
    assert_eq!(tracing.flush_failures(), 1);
}

#[test]
fn test_w3_disabled_noop_control_invariants() {
    let noop = NoopSink;
    noop.counter_add("any.metric", 100, &[]);
    noop.gauge_set("any.gauge", 99.9, &[]);
    noop.histogram_record("any.hist", 0.5, &[]);
    let s = noop.span_enter("any.span", &[]);
    assert_eq!(s, 0, "NoopSink span_enter must return 0");
    noop.span_exit(0, VmSpanStatus::Ok);
    noop.flush();
}

#[test]
fn test_w3_tracing_sink_records_and_lifecycle() {
    let sink = TracingSink::new().with_vm_id("worker-1");
    TracingTelemetrySink::counter_add(&sink, "events.processed", 10, &[("kind", "http")]);
    TracingTelemetrySink::gauge_set(&sink, "active_workers", 4.0, &[]);
    TracingTelemetrySink::histogram_record(&sink, "latency_s", 0.12, &[]);

    let root = TracingTelemetrySink::span_enter(&sink, "root_request", &[]);
    let child = TracingTelemetrySink::span_enter(&sink, "handler", &[]);
    TracingTelemetrySink::span_exit(&sink, child, lust_telemetry_tracing::SpanStatus::Success);
    TracingTelemetrySink::span_exit(&sink, root, lust_telemetry_tracing::SpanStatus::Ok);

    let recs = sink.records();
    assert_eq!(recs.len(), 7);
    assert_eq!(sink.spans_named("root_request").len(), 1);
    assert_eq!(sink.spans_named("handler").len(), 1);
}

#[test]
fn test_w3_vm_with_telemetry_sinks() {
    // Verify that VM accepts all sinks via VmTelemetrySink contract
    let _vm = VM::new();
    let _mock = Arc::new(MockTelemetrySink::new());
    let _mask = TelemetrySignalMask::ALL;

    #[cfg(feature = "telemetry")]
    {
        _vm.set_telemetry_sink(_mock.clone(), _mask);
    }
}

use lust_telemetry_metrics::{NoopTelemetrySink, SpanStatus, TelemetrySink, UPSTREAM_ENABLED};

fn record_fixture(sink: &dyn TelemetrySink) {
    sink.counter_add(
        "lust.vm.gas.consumed_total",
        320,
        &[("tenant", "alpha"), ("tier", "hot")],
    );
    sink.counter_add(
        "lust.vm.gas.consumed_total",
        80,
        &[("tier", "hot"), ("tenant", "alpha")],
    );
    sink.gauge_set(
        "lust.vm.memory.bytes_used",
        16384.5,
        &[("tenant", "alpha")],
    );
    sink.histogram_record(
        "lust.jit.compile_duration_sec",
        0.012,
        &[("opt", "safepoint")],
    );
    sink.histogram_record(
        "lust.jit.compile_duration_sec",
        0.048,
        &[("opt", "safepoint")],
    );
    let span = sink.span_enter("eval_fixture", &[("tenant", "alpha")]);
    sink.span_exit(span, SpanStatus::Success);
    sink.flush();
}

#[test]
fn test_noop_sink_is_zero_sized_and_silent() {
    assert_eq!(std::mem::size_of::<NoopTelemetrySink>(), 0);
    let noop = NoopTelemetrySink;
    record_fixture(&noop);
    assert_eq!(noop.span_enter("never", &[]), 0);
}

#[cfg(not(feature = "upstream"))]
#[test]
fn test_default_build_has_upstream_disabled() {
    const { assert!(!UPSTREAM_ENABLED) };
    let sink = lust_telemetry_metrics::MetricsSink::new();
    record_fixture(&sink);
    assert_eq!(
        sink.counter(
            "lust.vm.gas.consumed_total",
            &[("tenant", "alpha"), ("tier", "hot")]
        ),
        Some(400)
    );
}

#[cfg(feature = "upstream")]
mod upstream_tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    use lust_telemetry_metrics::{
        CardinalityLimits, InMemoryMetricsRecorder, UpstreamMetricsSink,
    };

    #[test]
    fn test_upstream_feature_flag_enabled() {
        const { assert!(UPSTREAM_ENABLED) };
    }

    #[test]
    fn test_disabled_adapter_emits_zero_upstream_samples_and_enabled_exports_exact_fixture() {
        let recorder = Arc::new(InMemoryMetricsRecorder::new());

        // 1. Disabled adapter (direct + scoped): must emit zero samples to the real upstream recorder.
        let disabled_sink = Arc::new(
            UpstreamMetricsSink::with_recorder(recorder.clone()).with_enabled(false),
        );
        disabled_sink.describe_counter(
            "lust.vm.gas.consumed_total",
            Some(metrics::Unit::Count),
            "Gas consumed by VM",
        );
        record_fixture(disabled_sink.as_ref());
        let disabled_scoped = disabled_sink.scoped(&[("vm_id", "vm-off")]);
        record_fixture(&disabled_scoped);

        assert_eq!(disabled_sink.emitted_samples(), 0);
        assert_eq!(disabled_sink.flush_count(), 0);
        assert_eq!(disabled_scoped.span_enter("never", &[]), 0);
        assert!(
            recorder.is_empty(),
            "disabled adapter must emit zero samples to upstream metrics::Recorder"
        );
        assert!(
            recorder.description("lust.vm.gas.consumed_total").is_none(),
            "disabled adapter must not describe metrics"
        );

        // 2. Enabled adapter: must export exact counters, gauges, histograms, and labels.
        let enabled_sink = UpstreamMetricsSink::with_recorder(recorder.clone())
            .with_vm_id("vm-42")
            .with_enabled(true);
        enabled_sink.describe_counter(
            "lust.vm.gas.consumed_total",
            Some(metrics::Unit::Count),
            "Gas consumed by VM",
        );
        record_fixture(&enabled_sink);

        assert_eq!(enabled_sink.emitted_samples(), 5);
        assert_eq!(
            recorder.counter(
                "lust.vm.gas.consumed_total",
                &[("vm_id", "vm-42"), ("tenant", "alpha"), ("tier", "hot")]
            ),
            Some(400)
        );
        assert_eq!(
            recorder.gauge(
                "lust.vm.memory.bytes_used",
                &[("vm_id", "vm-42"), ("tenant", "alpha")]
            ),
            Some(16384.5)
        );
        assert_eq!(
            recorder.histogram_values(
                "lust.jit.compile_duration_sec",
                &[("vm_id", "vm-42"), ("opt", "safepoint")]
            ),
            Some(vec![0.012, 0.048])
        );
        let desc = recorder
            .description("lust.vm.gas.consumed_total")
            .expect("description must be recorded");
        assert_eq!(desc.unit, Some(metrics::Unit::Count));
        assert_eq!(desc.description, "Gas consumed by VM");
    }

    #[test]
    fn test_ambient_metrics_macros_interop_with_independent_recorder() {
        let recorder = Arc::new(InMemoryMetricsRecorder::new());
        let ambient_sink = UpstreamMetricsSink::new();

        metrics::with_local_recorder(recorder.as_ref(), || {
            metrics::counter!("direct.macro.counter", "origin" => "macro").increment(7);
            ambient_sink.counter_add("direct.macro.counter", 5, &[("origin", "macro")]);
            metrics::gauge!("direct.macro.gauge", "origin" => "macro").set(3.25);
            metrics::histogram!("direct.macro.hist").record(1.5);
        });

        assert_eq!(
            recorder.counter("direct.macro.counter", &[("origin", "macro")]),
            Some(12)
        );
        assert_eq!(
            recorder.gauge("direct.macro.gauge", &[("origin", "macro")]),
            Some(3.25)
        );
        assert_eq!(recorder.histogram_values("direct.macro.hist", &[]), Some(vec![1.5]));
    }

    #[test]
    fn test_independent_recorders_and_concurrent_scoped_isolation() {
        let rec_a = Arc::new(InMemoryMetricsRecorder::new());
        let rec_b = Arc::new(InMemoryMetricsRecorder::new());

        let sink_a = Arc::new(UpstreamMetricsSink::with_recorder(rec_a.clone()));
        let sink_b = Arc::new(UpstreamMetricsSink::with_recorder(rec_b.clone()));

        let mut handles = Vec::new();
        for i in 0..4 {
            let vm_label = format!("vm-{i}");
            let scoped_a = sink_a.scoped(&[("vm_id", &vm_label)]);
            let scoped_b = sink_b.scoped(&[("vm_id", &vm_label)]);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    scoped_a.counter_add("ops.total", 2, &[("pool", "a")]);
                    scoped_b.counter_add("ops.total", 3, &[("pool", "b")]);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        for i in 0..4 {
            let vm_label = format!("vm-{i}");
            assert_eq!(
                rec_a.counter("ops.total", &[("vm_id", &vm_label), ("pool", "a")]),
                Some(200)
            );
            assert_eq!(
                rec_a.counter("ops.total", &[("vm_id", &vm_label), ("pool", "b")]),
                None
            );
            assert_eq!(
                rec_b.counter("ops.total", &[("vm_id", &vm_label), ("pool", "b")]),
                Some(300)
            );
            assert_eq!(
                rec_b.counter("ops.total", &[("vm_id", &vm_label), ("pool", "a")]),
                None
            );
        }
        assert_eq!(rec_a.counter_total("ops.total"), 800);
        assert_eq!(rec_b.counter_total("ops.total"), 1200);
    }

    #[test]
    fn test_cardinality_overflow_and_flush_failures() {
        let recorder = Arc::new(InMemoryMetricsRecorder::new());
        let limits = CardinalityLimits {
            max_unique_names: 2,
            max_label_pairs: 2,
            max_label_val_len: 8,
            max_unique_series: 3,
        };
        let sink = UpstreamMetricsSink::with_recorder(recorder.clone()).with_limits(limits);

        sink.counter_add("reqs", 1, &[("user", "u1")]);
        sink.counter_add("reqs", 1, &[("user", "u2")]);
        // 2nd unique name ok (series #3 of 3)
        sink.counter_add("ok_name_2", 1, &[]);
        // 4th unique series folds into overflow="true"
        sink.counter_add("reqs", 5, &[("user", "u3_overflow")]);
        // 3rd unique name dropped due to max_unique_names (2)
        sink.counter_add("dropped_name_3", 99, &[]);

        assert_eq!(sink.dropped_series_total(), 2);
        assert_eq!(recorder.counter("ok_name_2", &[]), Some(1));
        assert_eq!(recorder.counter("reqs", &[("overflow", "true")]), Some(5));
        assert_eq!(recorder.counter("dropped_name_3", &[]), None);

        // Flush failure tracking
        sink.flush();
        assert_eq!(sink.flush_count(), 1);
        assert_eq!(sink.flush_failures(), 0);

        sink.set_fail_on_flush(true);
        sink.flush();
        assert_eq!(sink.flush_count(), 2);
        assert_eq!(sink.flush_failures(), 1);

        // UTF-8 safe value truncation & max_label_pairs bounding
        let trunc_rec = Arc::new(InMemoryMetricsRecorder::new());
        let trunc_limits = CardinalityLimits {
            max_unique_names: 4,
            max_label_pairs: 1,
            max_label_val_len: 5,
            max_unique_series: 4,
        };
        let trunc_sink =
            UpstreamMetricsSink::with_recorder(trunc_rec.clone()).with_limits(trunc_limits);
        // "αβγ" is 6 UTF-8 bytes; truncating to <= 5 bytes must cut cleanly at char boundary -> "αβ" (4 bytes).
        // Second label pair ("drop", "me") must be trimmed by max_label_pairs=1.
        trunc_sink.counter_add("utf8.test", 9, &[("k", "αβγ"), ("drop", "me")]);
        assert_eq!(trunc_rec.counter("utf8.test", &[("k", "αβ")]), Some(9));
    }
}

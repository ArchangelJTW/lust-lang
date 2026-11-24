use lust_telemetry_otel::{NoopTelemetrySink, SpanStatus, TelemetrySink, UPSTREAM_ENABLED};

fn record_fixture(sink: &dyn TelemetrySink) {
    let root = sink.span_enter("eval_rule", &[("tenant", "alpha")]);
    let child = sink.span_enter("query_state", &[("db.system", "sqlite")]);
    sink.counter_add(
        "lust.vm.gas.consumed_total",
        250,
        &[("tenant", "alpha"), ("tier", "hot")],
    );
    sink.counter_add(
        "lust.vm.gas.consumed_total",
        150,
        &[("tier", "hot"), ("tenant", "alpha")],
    );
    sink.gauge_set(
        "lust.vm.memory.bytes_used",
        4096.25,
        &[("tenant", "alpha")],
    );
    sink.histogram_record(
        "lust.jit.compile_duration_sec",
        0.025,
        &[("opt", "safepoint")],
    );
    sink.histogram_record(
        "lust.jit.compile_duration_sec",
        0.075,
        &[("opt", "safepoint")],
    );
    sink.span_exit(child, SpanStatus::Fail);
    sink.span_exit(root, SpanStatus::Success);
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
    let sink = lust_telemetry_otel::OtelSink::new().with_service_name("default-svc");
    record_fixture(&sink);
    assert_eq!(sink.spans().len(), 2);
}

#[cfg(feature = "upstream")]
mod upstream_tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    use lust_telemetry_otel::{OtelCardinalityLimits, UpstreamOtelSink};
    use opentelemetry::trace::{SpanId, Status};
    use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData, ResourceMetrics};

    fn find_u64_sum(rm: &ResourceMetrics, name: &str, expected_attrs: &[(&str, &str)]) -> Option<u64> {
        for sm in rm.scope_metrics() {
            for metric in sm.metrics() {
                if metric.name() != name {
                    continue;
                }
                if let AggregatedMetrics::U64(MetricData::Sum(sum)) = metric.data() {
                    for dp in sum.data_points() {
                        let mut actual: Vec<(String, String)> = dp
                            .attributes()
                            .map(|kv| (kv.key.as_str().to_string(), kv.value.as_str().into_owned()))
                            .collect();
                        actual.sort();
                        let mut expected: Vec<(String, String)> = expected_attrs
                            .iter()
                            .map(|(k, v)| (k.to_string(), v.to_string()))
                            .collect();
                        expected.sort();
                        if actual == expected {
                            return Some(dp.value());
                        }
                    }
                }
            }
        }
        None
    }

    fn find_f64_gauge(rm: &ResourceMetrics, name: &str, expected_attrs: &[(&str, &str)]) -> Option<f64> {
        for sm in rm.scope_metrics() {
            for metric in sm.metrics() {
                if metric.name() != name {
                    continue;
                }
                if let AggregatedMetrics::F64(MetricData::Gauge(gauge)) = metric.data() {
                    for dp in gauge.data_points() {
                        let mut actual: Vec<(String, String)> = dp
                            .attributes()
                            .map(|kv| (kv.key.as_str().to_string(), kv.value.as_str().into_owned()))
                            .collect();
                        actual.sort();
                        let mut expected: Vec<(String, String)> = expected_attrs
                            .iter()
                            .map(|(k, v)| (k.to_string(), v.to_string()))
                            .collect();
                        expected.sort();
                        if actual == expected {
                            return Some(dp.value());
                        }
                    }
                }
            }
        }
        None
    }

    fn find_f64_histogram(
        rm: &ResourceMetrics,
        name: &str,
        expected_attrs: &[(&str, &str)],
    ) -> Option<(u64, f64)> {
        for sm in rm.scope_metrics() {
            for metric in sm.metrics() {
                if metric.name() != name {
                    continue;
                }
                if let AggregatedMetrics::F64(MetricData::Histogram(hist)) = metric.data() {
                    for dp in hist.data_points() {
                        let mut actual: Vec<(String, String)> = dp
                            .attributes()
                            .map(|kv| (kv.key.as_str().to_string(), kv.value.as_str().into_owned()))
                            .collect();
                        actual.sort();
                        let mut expected: Vec<(String, String)> = expected_attrs
                            .iter()
                            .map(|(k, v)| (k.to_string(), v.to_string()))
                            .collect();
                        expected.sort();
                        if actual == expected {
                            return Some((dp.count(), dp.sum()));
                        }
                    }
                }
            }
        }
        None
    }

    fn count_total_metrics(rm: &ResourceMetrics) -> usize {
        rm.scope_metrics().map(|sm| sm.metrics().count()).sum()
    }

    #[test]
    fn test_upstream_feature_flag_enabled() {
        const { assert!(UPSTREAM_ENABLED) };
    }

    #[test]
    fn test_disabled_adapter_emits_zero_otel_spans_and_metrics() {
        let disabled = Arc::new(
            UpstreamOtelSink::new_in_memory("disabled-svc", Some("vm-off")).with_enabled(false),
        );
        record_fixture(disabled.as_ref());
        let disabled_scoped = disabled.scoped(&[("tenant", "off")]);
        record_fixture(&disabled_scoped);

        assert_eq!(disabled.emitted_samples(), 0);
        assert_eq!(disabled.flush_count(), 0);
        assert_eq!(disabled_scoped.span_enter("never", &[]), 0);
        let spans = disabled.finished_spans().expect("span exporter readable");
        assert!(
            spans.is_empty(),
            "disabled OTel adapter must emit zero upstream spans"
        );

        let manual = disabled
            .collect_manual_metrics()
            .expect("manual reader readable");
        assert_eq!(
            count_total_metrics(&manual),
            0,
            "disabled OTel adapter must emit zero manual metrics"
        );

        let exported_batches = disabled
            .finished_metrics()
            .expect("metric exporter readable");
        assert!(
            exported_batches.is_empty(),
            "disabled OTel adapter must push zero batches to InMemoryMetricExporter"
        );
    }

    #[test]
    fn test_enabled_adapter_exports_exact_spans_and_metrics_via_all_sdk_exporters() {
        let sink = UpstreamOtelSink::new_in_memory("state-guard-engine", Some("vm-01"));
        record_fixture(&sink);

        // 1. Verify InMemorySpanExporter: real parent-child spans, statuses, attributes.
        let spans = sink.finished_spans().expect("finished spans");
        assert_eq!(spans.len(), 2);

        let child = spans
            .iter()
            .find(|s| s.name.as_ref() == "query_state")
            .expect("child span exported");
        let root = spans
            .iter()
            .find(|s| s.name.as_ref() == "eval_rule")
            .expect("root span exported");

        assert_eq!(root.parent_span_id, SpanId::INVALID);
        assert_eq!(child.parent_span_id, root.span_context.span_id());
        assert_eq!(child.span_context.trace_id(), root.span_context.trace_id());
        assert!(!child.parent_span_is_remote);
        assert_eq!(root.status, Status::Ok);
        assert!(matches!(child.status, Status::Error { .. }));
        assert!(child
            .attributes
            .iter()
            .any(|kv| kv.key.as_str() == "db.system" && kv.value.as_str() == "sqlite"));
        assert!(root
            .attributes
            .iter()
            .any(|kv| kv.key.as_str() == "vm.id" && kv.value.as_str() == "vm-01"));

        // 2. Verify ManualReader synchronous pull collection.
        let manual_rm = sink.collect_manual_metrics().expect("manual collect");
        assert_eq!(
            manual_rm
                .resource()
                .get(&opentelemetry::Key::new("service.name"))
                .map(|v| v.as_str().into_owned()),
            Some("state-guard-engine".to_string())
        );
        assert_eq!(
            find_u64_sum(
                &manual_rm,
                "lust.vm.gas.consumed_total",
                &[("vm.id", "vm-01"), ("tenant", "alpha"), ("tier", "hot")]
            ),
            Some(400)
        );
        assert_eq!(
            find_f64_gauge(
                &manual_rm,
                "lust.vm.memory.bytes_used",
                &[("vm.id", "vm-01"), ("tenant", "alpha")]
            ),
            Some(4096.25)
        );
        let (h_count, h_sum) = find_f64_histogram(
            &manual_rm,
            "lust.jit.compile_duration_sec",
            &[("vm.id", "vm-01"), ("opt", "safepoint")],
        )
        .expect("histogram exported via ManualReader");
        assert_eq!(h_count, 2);
        assert!((h_sum - 0.1).abs() < 1e-9);

        // 3. Verify InMemoryMetricExporter pushed batch via flush().
        let pushed_batches = sink.finished_metrics().expect("in-memory metric batches");
        assert!(!pushed_batches.is_empty(), "flush() must export to InMemoryMetricExporter");
        let last_batch = pushed_batches.last().unwrap();
        assert_eq!(
            find_u64_sum(
                last_batch,
                "lust.vm.gas.consumed_total",
                &[("vm.id", "vm-01"), ("tenant", "alpha"), ("tier", "hot")]
            ),
            Some(400)
        );
        assert_eq!(
            find_f64_gauge(
                last_batch,
                "lust.vm.memory.bytes_used",
                &[("vm.id", "vm-01"), ("tenant", "alpha")]
            ),
            Some(4096.25)
        );
    }

    #[test]
    fn test_manual_only_isolated_concurrent_sinks() {
        let sink_a = Arc::new(UpstreamOtelSink::new_manual_only("svc-a", None));
        let sink_b = Arc::new(UpstreamOtelSink::new_manual_only("svc-b", None));

        let mut handles = Vec::new();
        for i in 0..4 {
            let vm = format!("vm-{i}");
            let sa = sink_a.scoped(&[("vm.id", &vm)]);
            let sb = sink_b.scoped(&[("vm.id", &vm)]);
            handles.push(thread::spawn(move || {
                for _ in 0..50 {
                    sa.counter_add("work.units", 2, &[("lane", "a")]);
                    sb.counter_add("work.units", 5, &[("lane", "b")]);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let rm_a = sink_a.collect_manual_metrics().unwrap();
        let rm_b = sink_b.collect_manual_metrics().unwrap();

        for i in 0..4 {
            let vm = format!("vm-{i}");
            assert_eq!(
                find_u64_sum(&rm_a, "work.units", &[("vm.id", &vm), ("lane", "a")]),
                Some(100)
            );
            assert_eq!(
                find_u64_sum(&rm_a, "work.units", &[("vm.id", &vm), ("lane", "b")]),
                None
            );
            assert_eq!(
                find_u64_sum(&rm_b, "work.units", &[("vm.id", &vm), ("lane", "b")]),
                Some(250)
            );
        }
    }

    #[test]
    fn test_cardinality_overflow_and_real_sdk_shutdown_failure() {
        let limits = OtelCardinalityLimits {
            max_unique_names: 2,
            max_label_pairs: 2,
            max_label_val_len: 12,
            max_unique_series: 2,
        };
        let sink = UpstreamOtelSink::new_manual_only("limit-svc", None).with_limits(limits);

        sink.counter_add("reqs", 1, &[("user", "u1")]);
        sink.counter_add("reqs", 1, &[("user", "u2")]);
        sink.counter_add("reqs", 7, &[("user", "u3")]);
        assert_eq!(sink.dropped_series_total(), 1);

        let rm = sink.collect_manual_metrics().unwrap();
        assert_eq!(
            find_u64_sum(&rm, "reqs", &[("overflow", "true")]),
            Some(7)
        );

        // Simulated flush failure
        sink.set_fail_on_flush(true);
        sink.flush();
        assert_eq!(sink.flush_count(), 1);
        assert_eq!(sink.flush_failures(), 1);

        // Real SDK shutdown failure: after shutdown(), force_flush and ManualReader::collect fail.
        sink.set_fail_on_flush(false);
        sink.shutdown().expect("first shutdown succeeds");
        sink.flush();
        assert_eq!(sink.flush_count(), 2);
        assert_eq!(
            sink.flush_failures(),
            2,
            "flush after SDK shutdown must record a real SDK failure"
        );
        assert!(
            sink.collect_manual_metrics().is_err(),
            "ManualReader::collect must fail after shutdown"
        );
    }

    #[test]
    fn test_interleaved_sinks_on_same_thread_isolate_span_parentage() {
        let sink_a = UpstreamOtelSink::new_manual_only("svc-a", Some("vm-a"));
        let sink_b = UpstreamOtelSink::new_manual_only("svc-b", Some("vm-b"));

        let a_root = sink_a.span_enter("a_root", &[]);
        let b_root = sink_b.span_enter("b_root", &[]);
        let a_child = sink_a.span_enter("a_child", &[]);
        let b_child = sink_b.span_enter("b_child", &[]);

        sink_b.span_exit(b_child, SpanStatus::Success);
        sink_a.span_exit(a_child, SpanStatus::Success);
        sink_b.span_exit(b_root, SpanStatus::Success);
        sink_a.span_exit(a_root, SpanStatus::Success);

        let spans_a = sink_a.finished_spans().unwrap();
        let spans_b = sink_b.finished_spans().unwrap();
        assert_eq!(spans_a.len(), 2);
        assert_eq!(spans_b.len(), 2);

        let sa_root = spans_a.iter().find(|s| s.name.as_ref() == "a_root").unwrap();
        let sa_child = spans_a.iter().find(|s| s.name.as_ref() == "a_child").unwrap();
        let sb_root = spans_b.iter().find(|s| s.name.as_ref() == "b_root").unwrap();
        let sb_child = spans_b.iter().find(|s| s.name.as_ref() == "b_child").unwrap();

        assert_eq!(sa_root.parent_span_id, SpanId::INVALID);
        assert_eq!(sb_root.parent_span_id, SpanId::INVALID);
        assert_eq!(sa_child.parent_span_id, sa_root.span_context.span_id());
        assert_eq!(sb_child.parent_span_id, sb_root.span_context.span_id());
        assert_ne!(
            sa_root.span_context.trace_id(),
            sb_root.span_context.trace_id(),
            "interleaved sinks must not share trace IDs"
        );
    }
}

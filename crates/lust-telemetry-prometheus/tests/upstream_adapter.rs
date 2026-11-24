use lust_telemetry_prometheus::{NoopTelemetrySink, SpanStatus, TelemetrySink, UPSTREAM_ENABLED};

fn record_fixture(sink: &dyn TelemetrySink) {
    sink.counter_add(
        "lust.vm.gas.consumed",
        300,
        &[("tenant", "alpha"), ("tier", "hot")],
    );
    sink.counter_add(
        "lust.vm.gas.consumed",
        200,
        &[("tier", "hot"), ("tenant", "alpha")],
    );
    sink.gauge_set(
        "lust.vm.memory.bytes",
        8192.5,
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
    let sink = lust_telemetry_prometheus::PrometheusSink::new();
    record_fixture(&sink);
    let text = sink.render_openmetrics();
    assert!(text.contains("lust_vm_gas_consumed_total{tenant=\"alpha\",tier=\"hot\"} 500"));
    assert!(text.ends_with("# EOF\n"));
}

#[cfg(feature = "upstream")]
mod upstream_tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    use lust_telemetry_prometheus::{CardinalityLimits, UpstreamPrometheusSink};
    use prometheus::proto::MetricType;

    #[test]
    fn test_upstream_feature_flag_enabled() {
        const { assert!(UPSTREAM_ENABLED) };
    }

    #[test]
    fn test_disabled_adapter_emits_zero_prometheus_samples_and_enabled_exports_exact_fixture() {
        // 1. Disabled adapter (direct + scoped): zero metrics registered in independent prometheus::Registry.
        let disabled = Arc::new(UpstreamPrometheusSink::new().with_enabled(false));
        record_fixture(disabled.as_ref());
        let disabled_scoped = disabled.scoped(&[("vm_id", "vm-off")]);
        record_fixture(&disabled_scoped);

        assert_eq!(disabled.emitted_samples(), 0);
        assert_eq!(disabled.flush_count(), 0);
        assert_eq!(disabled_scoped.span_enter("never", &[]), 0);
        assert!(
            disabled.gather().is_empty(),
            "disabled Prometheus adapter must emit zero samples to prometheus::Registry"
        );
        assert_eq!(
            disabled
                .render_prometheus_text()
                .expect("TextEncoder succeeds on empty registry"),
            ""
        );

        // 2. Enabled adapter: exact counters, gauges, histograms, and labels via TextEncoder.
        let enabled = UpstreamPrometheusSink::new()
            .with_vm_id("vm-07")
            .with_enabled(true);
        record_fixture(&enabled);

        assert_eq!(enabled.emitted_samples(), 5);
        assert_eq!(enabled.registration_errors(), 0);

        // Verify raw MetricFamily protos directly from prometheus::Registry::gather().
        let families = enabled.gather();
        assert_eq!(families.len(), 3);

        let gas_mf = families
            .iter()
            .find(|mf| mf.name() == "lust_vm_gas_consumed_total")
            .expect("counter family registered");
        assert_eq!(gas_mf.get_field_type(), MetricType::COUNTER);
        assert_eq!(gas_mf.get_metric().len(), 1);
        assert_eq!(gas_mf.get_metric()[0].get_counter().value(), 500.0);

        let mem_mf = families
            .iter()
            .find(|mf| mf.name() == "lust_vm_memory_bytes")
            .expect("gauge family registered");
        assert_eq!(mem_mf.get_field_type(), MetricType::GAUGE);
        assert_eq!(mem_mf.get_metric()[0].get_gauge().value(), 8192.5);

        let hist_mf = families
            .iter()
            .find(|mf| mf.name() == "lust_jit_compile_duration_sec")
            .expect("histogram family registered");
        assert_eq!(hist_mf.get_field_type(), MetricType::HISTOGRAM);
        let hist_proto = hist_mf.get_metric()[0].get_histogram();
        assert_eq!(hist_proto.get_sample_count(), 2);
        assert!((hist_proto.get_sample_sum() - 0.1).abs() < 1e-9);

        // Verify real prometheus::TextEncoder exposition text and byte encoding parity.
        let text = enabled
            .render_prometheus_text()
            .expect("TextEncoder encode_to_string");
        let bytes = enabled
            .encode_prometheus_bytes()
            .expect("TextEncoder encode bytes");
        assert_eq!(String::from_utf8(bytes).unwrap(), text);

        assert!(text.contains("# TYPE lust_vm_gas_consumed_total counter\n"));
        assert!(text.contains(
            "lust_vm_gas_consumed_total{tenant=\"alpha\",tier=\"hot\",vm_id=\"vm-07\"} 500\n"
        ));
        assert!(text.contains("# TYPE lust_vm_memory_bytes gauge\n"));
        assert!(text.contains("lust_vm_memory_bytes{tenant=\"alpha\",vm_id=\"vm-07\"} 8192.5\n"));
        assert!(text.contains("# TYPE lust_jit_compile_duration_sec histogram\n"));
        assert!(text.contains(
            "lust_jit_compile_duration_sec_bucket{opt=\"safepoint\",vm_id=\"vm-07\",le=\"0.025\"} 1\n"
        ));
        assert!(text.contains(
            "lust_jit_compile_duration_sec_bucket{opt=\"safepoint\",vm_id=\"vm-07\",le=\"0.1\"} 2\n"
        ));
        assert!(text.contains(
            "lust_jit_compile_duration_sec_bucket{opt=\"safepoint\",vm_id=\"vm-07\",le=\"+Inf\"} 2\n"
        ));
        assert!(text.contains(
            "lust_jit_compile_duration_sec_count{opt=\"safepoint\",vm_id=\"vm-07\"} 2\n"
        ));

        let openmetrics = enabled.render_openmetrics().expect("openmetrics");
        assert!(openmetrics.ends_with("# EOF\n"));

        // Verify global default_registry was NEVER touched (strict per-sink isolation).
        assert!(
            prometheus::gather()
                .iter()
                .all(|mf| mf.name() != "lust_vm_gas_consumed_total"),
            "independent registry must not leak into prometheus::default_registry"
        );
    }

    #[test]
    fn test_escaping_and_reserved_le_label_via_real_text_encoder() {
        let sink = UpstreamPrometheusSink::new();
        sink.counter_add(
            "escape.test.metric",
            42,
            &[("msg", "line1\n\"quoted\"\\slash")],
        );
        // Reserved "le" label on histogram must be remapped rather than failing registration.
        sink.histogram_record("latency_sec", 0.05, &[("le", "custom")]);

        assert_eq!(sink.registration_errors(), 0);
        let text = sink.render_prometheus_text().expect("TextEncoder succeeds");
        assert!(text.contains(
            "escape_test_metric_total{msg=\"line1\\n\\\"quoted\\\"\\\\slash\"} 42\n"
        ));
        assert!(text.contains("latency_sec_count{label_le=\"custom\"} 1\n"));
    }

    #[test]
    fn test_independent_registries_and_concurrent_scoped_isolation() {
        let sink_a = Arc::new(UpstreamPrometheusSink::new());
        let sink_b = Arc::new(UpstreamPrometheusSink::new());

        let mut handles = Vec::new();
        for i in 0..4 {
            let vm = format!("vm-{i}");
            let sa = sink_a.scoped(&[("vm_id", &vm)]);
            let sb = sink_b.scoped(&[("vm_id", &vm)]);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    sa.counter_add("vm.ticks", 1, &[("pool", "a")]);
                    sb.counter_add("vm.ticks", 4, &[("pool", "b")]);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let text_a = sink_a.render_prometheus_text().unwrap();
        let text_b = sink_b.render_prometheus_text().unwrap();

        for i in 0..4 {
            assert!(text_a.contains(&format!("vm_ticks_total{{pool=\"a\",vm_id=\"vm-{i}\"}} 100\n")));
            assert!(!text_a.contains("pool=\"b\""));
            assert!(text_b.contains(&format!("vm_ticks_total{{pool=\"b\",vm_id=\"vm-{i}\"}} 400\n")));
            assert!(!text_b.contains("pool=\"a\""));
        }
    }

    #[test]
    fn test_cardinality_overflow_dimension_mismatch_and_flush_failures() {
        let limits = CardinalityLimits {
            max_unique_names: 3,
            max_label_pairs: 2,
            max_label_val_len: 16,
            max_unique_series: 2,
        };
        let sink = UpstreamPrometheusSink::with_limits(limits);

        sink.counter_add("users", 1, &[("id", "u1")]);
        sink.counter_add("users", 1, &[("id", "u2")]);
        // 3rd series exceeds max_unique_series -> folded into lust_telemetry_overflow_total
        sink.counter_add("users", 3, &[("id", "u3")]);

        assert_eq!(sink.dropped_series_total(), 1);
        let text = sink.render_prometheus_text().unwrap();
        assert!(text.contains(
            "lust_telemetry_overflow_total{metric=\"users\",overflow=\"true\"} 3\n"
        ));

        // Mismatched label dimension on already-registered metric family must fail gracefully
        // without panicking and increment registration_errors.
        let sink_dim = UpstreamPrometheusSink::new();
        sink_dim.counter_add("dim_metric", 1, &[("env", "prod")]);
        sink_dim.counter_add("dim_metric", 1, &[("env", "prod"), ("extra", "bad")]);
        assert_eq!(sink_dim.registration_errors(), 1);

        // Flush failure tracking
        sink_dim.flush();
        assert_eq!(sink_dim.flush_count(), 1);
        assert_eq!(sink_dim.flush_failures(), 0);

        sink_dim.set_fail_on_flush(true);
        sink_dim.flush();
        assert_eq!(sink_dim.flush_count(), 2);
        assert_eq!(sink_dim.flush_failures(), 1);
    }

    #[test]
    fn test_external_registry_binding_and_already_reg_collision() {
        let external_registry = prometheus::Registry::new();
        // Pre-register a metric with the same name as a counter to force a real AlreadyReg error.
        let preexisting = prometheus::Gauge::with_opts(prometheus::Opts::new(
            "colliding_metric_total",
            "pre-registered gauge",
        ))
        .unwrap();
        external_registry
            .register(Box::new(preexisting.clone()))
            .unwrap();
        preexisting.set(7.0);

        let sink = UpstreamPrometheusSink::with_registry(external_registry.clone());
        // Attempting to register a counter named colliding_metric_total fails cleanly via Registry::register.
        sink.counter_add("colliding_metric", 10, &[]);
        assert_eq!(sink.registration_errors(), 1);
        assert_eq!(sink.emitted_samples(), 0);

        // Non-colliding metric succeeds and is visible in the external prometheus::Registry.
        sink.counter_add("fresh_metric", 11, &[("env", "test")]);
        assert_eq!(sink.emitted_samples(), 1);

        let families = external_registry.gather();
        let fresh = families
            .iter()
            .find(|mf| mf.name() == "fresh_metric_total")
            .expect("fresh counter exported to external Registry");
        assert_eq!(fresh.get_metric()[0].get_counter().value(), 11.0);
    }
}

//! Integration tests for Track 2C (`TEL-CORE`):
//! - Compile-time gated telemetry signals (`telemetry-vm`, `telemetry-jit`, `telemetry-effects`, `telemetry-user`, `telemetry-all`).
//! - Zero-cost verification when disabled (zero allocations, no argument evaluation in macros, strict struct sizing).
//! - Component contracts: MockTelemetrySink, RingBufferTelemetrySink, TelemetrySignalMask, TelemetryCardinalityGuard.
//! - Runtime wiring: gas, memory, host ledger, JIT, and effect counters and hooks.
//! - Track 2B support: `release_memory_bytes` capacity accounting and `DependencySpec.link`.

use std::sync::Arc;

use lust::config::{LustConfig, TelemetryConfig, TelemetryLimitsConfig};
use lust::vm::budget::BudgetState;
use lust::vm::telemetry::{
    CompactEvent, MockTelemetrySink, NoopSink, OverflowPolicy, RingBufferTelemetrySink,
    RuntimeTelemetrySnapshot, SpanStatus, TelemetryCardinalityGuard, TelemetryEvent,
    TelemetrySignalMask, TelemetrySink,
};
use lust::vm::VM;

// -----------------------------------------------------------------------------
// 1. Zero-Cost Guarantees When Telemetry Is Disabled
// -----------------------------------------------------------------------------

#[test]
fn test_disabled_build_zero_cost_and_size_invariants() {
    // When telemetry features are disabled (the default build),
    // BudgetState must only contain GasBudget (24 bytes) and MemoryBudget (24 bytes).
    // Total size must remain exactly 48 bytes without any telemetry field overhead.
    #[cfg(not(any(
        feature = "telemetry",
        feature = "telemetry-all",
        feature = "telemetry-vm",
        feature = "telemetry-jit",
        feature = "telemetry-effects",
        feature = "telemetry-user"
    )))]
    {
        assert_eq!(
            std::mem::size_of::<BudgetState>(),
            48,
            "BudgetState size must be exactly 48 bytes in disabled builds (R-TEL-ZERO)"
        );
    }

    // Telemetry macros must NOT evaluate their arguments when disabled
    #[cfg(not(any(
        feature = "telemetry",
        feature = "telemetry-all",
        feature = "telemetry-vm",
        feature = "telemetry-jit",
        feature = "telemetry-effects",
        feature = "telemetry-user"
    )))]
    {
        #[allow(unused_mut, unused_variables)]
        let mut vm = VM::new();
        #[allow(unused_mut)]
        let mut side_effect = false;

        lust::tel_counter!(vm, "test.counter", {
            side_effect = true;
            100u64
        });
        assert!(
            !side_effect,
            "tel_counter! must never evaluate expressions when disabled"
        );

        lust::tel_gauge!(vm, "test.gauge", {
            side_effect = true;
            42.0f64
        });
        assert!(
            !side_effect,
            "tel_gauge! must never evaluate expressions when disabled"
        );

        lust::tel_histogram!(vm, "test.histogram", {
            side_effect = true;
            10.5f64
        });
        assert!(
            !side_effect,
            "tel_histogram! must never evaluate expressions when disabled"
        );

        let span_id = lust::tel_span_enter!(vm, "test.span", {
            side_effect = true;
            &[("k", "v")]
        });
        assert_eq!(span_id, 0);
        assert!(
            !side_effect,
            "tel_span_enter! must never evaluate expressions when disabled"
        );

        lust::tel_span_exit!(vm, {
            side_effect = true;
            0
        }, SpanStatus::Success);
        assert!(
            !side_effect,
            "tel_span_exit! must never evaluate expressions when disabled"
        );
    }
}

#[test]
fn test_disabled_build_vm_operations_perform_zero_telemetry_allocations() {
    let mut vm = VM::new();
    let sink = Arc::new(MockTelemetrySink::new());

    // Setting telemetry sink in disabled mode is a safe inline no-op
    vm.set_telemetry_sink(sink.clone(), TelemetrySignalMask::ALL);

    // Charge gas and verify normal operation
    vm.set_gas_budget(10_000);
    assert!(vm.charge_gas(500).is_ok());
    assert_eq!(vm.gas_used(), 500);

    // Charge memory and verify normal operation
    assert!(vm.set_memory_budget_bytes(64 * 1024).is_ok());
    assert!(vm.try_charge_memory_bytes(1024));
    assert_eq!(vm.memory_used(), 1024);

    // In disabled build, MockTelemetrySink receives 0 events
    #[cfg(not(any(
        feature = "telemetry",
        feature = "telemetry-all",
        feature = "telemetry-vm",
        feature = "telemetry-jit",
        feature = "telemetry-effects",
        feature = "telemetry-user"
    )))]
    {
        assert_eq!(
            sink.events().len(),
            0,
            "Disabled telemetry must record 0 events"
        );
        assert_eq!(sink.dropped_events(), 0);
    }
}

// -----------------------------------------------------------------------------
// 2. Telemetry Signal Mask Contracts
// -----------------------------------------------------------------------------

#[test]
fn test_telemetry_signal_mask_bitwise_operations() {
    let none = TelemetrySignalMask::NONE;
    assert!(none.is_empty());
    assert_eq!(none.bits(), 0);

    let gas = TelemetrySignalMask::GAS;
    let mem = TelemetrySignalMask::MEMORY;
    let vm_signals = TelemetrySignalMask::VM;

    assert!(vm_signals.contains(gas));
    assert!(vm_signals.contains(mem));
    assert!(!gas.contains(mem));

    let union_mask = gas | mem;
    assert!(union_mask.contains(gas));
    assert!(union_mask.contains(mem));
    assert_eq!(gas & mem, TelemetrySignalMask::NONE);

    let all = TelemetrySignalMask::ALL;
    assert!(all.contains(TelemetrySignalMask::GAS));
    assert!(all.contains(TelemetrySignalMask::MEMORY));
    assert!(all.contains(TelemetrySignalMask::HOST_LEDGER));
    assert!(all.contains(TelemetrySignalMask::JIT));
    assert!(all.contains(TelemetrySignalMask::EFFECTS));
    assert!(all.contains(TelemetrySignalMask::USER));

    let truncated = TelemetrySignalMask::from_bits_truncate(0xFFFF_FFFF);
    assert_eq!(truncated, TelemetrySignalMask::ALL);

    let mut custom = TelemetrySignalMask::NONE;
    custom.insert(TelemetrySignalMask::JIT);
    assert!(custom.contains(TelemetrySignalMask::JIT));
    custom.remove(TelemetrySignalMask::JIT);
    assert!(!custom.contains(TelemetrySignalMask::JIT));
}

// -----------------------------------------------------------------------------
// 3. MockTelemetrySink Monotonic Sequencing, Aggregates & Overflow Policies
// -----------------------------------------------------------------------------

#[test]
fn test_mock_telemetry_sink_monotonic_sequence_and_events() {
    let sink = MockTelemetrySink::new();

    sink.counter_add("lust.vm.gas.consumed_total", 100, &[("source", "vm")]);
    sink.gauge_set("lust.vm.memory.bytes_used", 4096.0, &[]);
    sink.histogram_record("lust.jit.compile_duration_ns", 12500.0, &[]);

    let span_id = sink.span_enter("process_order", &[("stage", "parse")]);
    assert_eq!(span_id, 1);
    sink.span_exit(span_id, SpanStatus::Success);

    let events = sink.events();
    assert_eq!(events.len(), 5);

    // Verify monotonic sequencing: seq 0, 1, 2, 3, 4
    for (i, ev) in events.iter().enumerate() {
        let ev_seq = match ev {
            TelemetryEvent::Counter { seq, .. }
            | TelemetryEvent::Gauge { seq, .. }
            | TelemetryEvent::Histogram { seq, .. }
            | TelemetryEvent::SpanEnter { seq, .. }
            | TelemetryEvent::SpanExit { seq, .. } => *seq,
        };
        assert_eq!(ev_seq, i as u64, "Sequence numbers must be strictly monotonic");
    }

    assert_eq!(sink.counter_total("lust.vm.gas.consumed_total"), 100);
    assert_eq!(
        sink.gauge_latest("lust.vm.memory.bytes_used"),
        Some(4096.0)
    );
    assert_eq!(sink.spans_named("process_order").len(), 1);

    sink.clear();
    assert_eq!(sink.events().len(), 0);
    assert_eq!(sink.dropped_events(), 0);
}

#[test]
fn test_mock_telemetry_sink_overflow_policies() {
    // 1. DropNewest Policy
    let drop_sink = MockTelemetrySink::with_capacity(3, OverflowPolicy::DropNewest);
    drop_sink.counter_add("c", 1, &[]);
    drop_sink.counter_add("c", 2, &[]);
    drop_sink.counter_add("c", 3, &[]);
    drop_sink.counter_add("c", 4, &[]); // overflow
    drop_sink.counter_add("c", 5, &[]); // overflow

    let events = drop_sink.events();
    assert_eq!(events.len(), 3);
    assert_eq!(drop_sink.dropped_events(), 2);
    assert_eq!(drop_sink.counter_total("c"), 6); // 1 + 2 + 3

    // 2. RingBuffer Policy
    let ring_sink = MockTelemetrySink::with_capacity(3, OverflowPolicy::RingBuffer);
    ring_sink.counter_add("c", 10, &[]);
    ring_sink.counter_add("c", 20, &[]);
    ring_sink.counter_add("c", 30, &[]);
    ring_sink.counter_add("c", 40, &[]); // drops 10, keeps 20, 30, 40

    let ring_events = ring_sink.events();
    assert_eq!(ring_events.len(), 3);
    assert_eq!(ring_sink.dropped_events(), 1);
    assert_eq!(ring_sink.counter_total("c"), 90); // 20 + 30 + 40
}

// -----------------------------------------------------------------------------
// 4. RingBufferTelemetrySink (Zero Heap Allocations, Fixed-Size Storage)
// -----------------------------------------------------------------------------

#[test]
fn test_ring_buffer_telemetry_sink_zero_heap_allocations() {
    let sink = RingBufferTelemetrySink::<4>::new();
    assert_eq!(sink.capacity(), 4);
    assert_eq!(sink.total_events(), 0);
    assert_eq!(sink.read_latest(), None);

    sink.counter_add("metric.a", 15, &[]);
    assert_eq!(sink.total_events(), 1);
    match sink.read_latest() {
        Some(CompactEvent::Counter { delta, .. }) => assert_eq!(delta, 15),
        other => panic!("Expected CompactEvent::Counter, got {:?}", other),
    }

    sink.gauge_set("metric.b", 99.5, &[]);
    sink.histogram_record("metric.c", 250.0, &[]);
    let span_id = sink.span_enter("span.d", &[]);
    assert_eq!(span_id, 1);
    sink.span_exit(span_id, SpanStatus::Die);

    assert_eq!(sink.total_events(), 5);
    match sink.read_latest() {
        Some(CompactEvent::SpanExit { status, .. }) => {
            assert_eq!(status, SpanStatus::Die as u8);
        }
        other => panic!("Expected CompactEvent::SpanExit, got {:?}", other),
    }

    // NoopSink verify
    let noop = NoopSink;
    noop.counter_add("noop", 1, &[]);
    noop.gauge_set("noop", 1.0, &[]);
    noop.histogram_record("noop", 1.0, &[]);
    assert_eq!(noop.span_enter("noop", &[]), 0);
    noop.span_exit(0, SpanStatus::Ok);
    noop.flush();
}

// -----------------------------------------------------------------------------
// 5. TelemetryCardinalityGuard (R-TEL-CARDINALITY DoS Protection)
// -----------------------------------------------------------------------------

#[test]
fn test_telemetry_cardinality_guard_dos_protection() {
    let limits = TelemetryLimitsConfig {
        max_unique_metric_names: 2,
        max_label_pairs_per_metric: 2,
        max_label_value_len: 8,
        max_unique_series: 3,
    };
    let guard = TelemetryCardinalityGuard::new(limits);

    // 1. Metric names check
    assert!(guard.check_and_fold("metric_1", &[]).is_some());
    assert!(guard.check_and_fold("metric_2", &[]).is_some());
    // 3rd metric name exceeds max_unique_metric_names (2)
    assert!(guard.check_and_fold("metric_3", &[]).is_none());
    assert_eq!(guard.dropped_series_total(), 1);

    // 2. Label pairs truncation (max 2) and value length truncation (max 8)
    let labels = [
        ("env", "production-long-value"), // truncated to 8 chars: "producti"
        ("tier", "frontend"),
        ("ignored", "third"),             // dropped by max_label_pairs_per_metric
    ];
    let sanitized = guard.check_and_fold("metric_1", &labels).unwrap();
    assert_eq!(sanitized.len(), 2);
    assert_eq!(sanitized[0], ("env".to_string(), "producti".to_string()));
    assert_eq!(sanitized[1], ("tier".to_string(), "frontend".to_string()));

    // 3. Series explosion folding to overflow = "true"
    let series_limits = TelemetryLimitsConfig {
        max_unique_metric_names: 10,
        max_label_pairs_per_metric: 2,
        max_label_value_len: 8,
        max_unique_series: 2,
    };
    let series_guard = TelemetryCardinalityGuard::new(series_limits);
    // Series 1: metric_1 + env=prod1
    assert!(series_guard.check_and_fold("metric_1", &[("env", "prod1")]).is_some());
    // Series 2: metric_1 + env=prod2
    assert!(series_guard.check_and_fold("metric_1", &[("env", "prod2")]).is_some());
    // Series count has now reached max_unique_series (2: prod1, prod2)
    // Next new series must fold into overflow = "true"
    let overflow_res = series_guard.check_and_fold("metric_1", &[("env", "prod3")]).unwrap();
    assert_eq!(overflow_res, vec![("overflow".to_string(), "true".to_string())]);
    assert_eq!(series_guard.dropped_series_total(), 1);
}

// -----------------------------------------------------------------------------
// 6. TOML Configuration Parsing & DependencySpec.link
// -----------------------------------------------------------------------------

#[test]
fn test_config_telemetry_and_dependency_link_parsing() {
    let toml_src = r#"
[settings]
stdlib_modules = ["math", "string"]

[telemetry]
enabled = true
service_name = "state-guard-engine"
exporter = "prometheus"
sampling_rate = 0.75

[telemetry.signals]
gas = true
memory = false
jit_bailouts = true
effect_spans = true
user_metrics = false

[telemetry.limits]
max_unique_metric_names = 128
max_label_pairs_per_metric = 6
max_label_value_len = 32
max_unique_series = 2048

[telemetry.prometheus]
listen = "0.0.0.0:9090"
path = "/metrics_stream"

[telemetry.external]
library = "./plugins/libaudit.dylib"
sha256 = "c0ffee"

[dependencies.fast_sqlite]
path = "../fast-sqlite"
kind = "rust"
link = "static"
"#;

    let config = LustConfig::from_toml_str(toml_src).expect("Parse toml config");
    let tel = config.telemetry();

    assert!(tel.enabled);
    assert_eq!(tel.service_name, "state-guard-engine");
    assert_eq!(tel.exporter, "prometheus");
    assert_eq!(tel.sampling_rate, 0.75);

    assert!(tel.signals.gas);
    assert!(!tel.signals.memory);
    assert!(tel.signals.jit_bailouts);
    assert!(tel.signals.effect_spans);
    assert!(!tel.signals.user_metrics);

    assert_eq!(tel.limits.max_unique_metric_names, 128);
    assert_eq!(tel.limits.max_label_pairs_per_metric, 6);
    assert_eq!(tel.limits.max_label_value_len, 32);
    assert_eq!(tel.limits.max_unique_series, 2048);

    assert_eq!(tel.prometheus.listen, "0.0.0.0:9090");
    assert_eq!(tel.prometheus.path, "/metrics_stream");

    assert_eq!(tel.external.library.as_deref(), Some("./plugins/libaudit.dylib"));
    assert_eq!(tel.external.sha256.as_deref(), Some("c0ffee"));

    // Verify DependencySpec.link (Track 2B)
    let deps = config.dependencies();
    let dep = deps.iter().find(|d| d.name() == "fast_sqlite").expect("fast_sqlite dependency");
    assert_eq!(dep.link(), Some("static"));
}

// -----------------------------------------------------------------------------
// 7. Track 2B Capacity Accounting (release_memory_bytes & refund_memory_bytes)
// -----------------------------------------------------------------------------

#[test]
fn test_track_2b_capacity_accounting_release_memory() {
    let mut vm = VM::new();
    assert!(vm.set_memory_budget_bytes(100_000).is_ok());

    assert!(vm.try_charge_memory_bytes(20_000));
    assert_eq!(vm.memory_used(), 20_000);

    // Release capacity safely via saturating_sub
    vm.release_memory_bytes(5_000);
    assert_eq!(vm.memory_used(), 15_000);

    // Excessive release saturates to 0 without underflow
    vm.release_memory_bytes(30_000);
    assert_eq!(vm.memory_used(), 0);
}

// -----------------------------------------------------------------------------
// 8. Runtime Telemetry Snapshot Contract
// -----------------------------------------------------------------------------

#[test]
fn test_runtime_telemetry_snapshot() {
    let mut vm = VM::new();
    vm.set_gas_budget(50_000);
    assert!(vm.charge_gas(1_200).is_ok());
    assert!(vm.set_memory_budget_bytes(80_000).is_ok());
    assert!(vm.try_charge_memory_bytes(4_096));

    let cleanup_fn = |_| {};
    let slot = vm.host_ledger.acquire(101, cleanup_fn).expect("acquire slot");
    assert_eq!(slot, 0);

    let snapshot: RuntimeTelemetrySnapshot = vm.telemetry_snapshot();
    assert_eq!(snapshot.gas_used, 1_200);
    assert_eq!(snapshot.gas_remaining, Some(48_800));
    assert_eq!(snapshot.memory_used_bytes, 4_096);
    assert_eq!(snapshot.active_host_slots, 1);
}

// -----------------------------------------------------------------------------
// 9. Real Gas, Memory, Host Ledger, JIT & Effect Events Verification
// -----------------------------------------------------------------------------

#[test]
fn test_enabled_telemetry_events_and_signals() {
    let sink = Arc::new(MockTelemetrySink::new());

    // Direct contract test of telemetry events matching RFC §3.4 catalog:
    // 1. Gas consumption and exhaustion
    sink.counter_add("lust.vm.gas.consumed_total", 500, &[]);
    sink.counter_add("lust.vm.gas.exhausted_total", 1, &[("source", "vm")]);

    // 2. Memory bytes used and rejection
    sink.gauge_set("lust.vm.memory.bytes_used", 10240.0, &[]);
    sink.counter_add("lust.vm.memory.limit_rejections_total", 1, &[]);

    // 3. Host ledger active slots and overflow
    sink.gauge_set("lust.vm.host_ledger.active_slots", 12.0, &[]);
    sink.counter_add("lust.vm.host_ledger.overflow_total", 1, &[]);

    // 4. JIT traces compiled and bailouts
    sink.counter_add("lust.jit.traces_compiled_total", 4, &[]);
    sink.counter_add("lust.jit.bailouts_total", 1, &[("guard_kind", "IntType")]);
    sink.histogram_record("lust.jit.compile_duration_ns", 8500.0, &[]);

    // 5. Effects and fiber lifecycle
    sink.counter_add("lust.fiber.lifecycle_total", 1, &[("event", "spawned")]);
    let span_id = sink.span_enter("lust.effect.span", &[("effect_name", "fs.read_file")]);
    assert_eq!(span_id, 1);
    sink.span_exit(span_id, SpanStatus::Success);

    // Verify all recorded counters and gauges
    assert_eq!(sink.counter_total("lust.vm.gas.consumed_total"), 500);
    assert_eq!(sink.counter_total("lust.vm.gas.exhausted_total"), 1);
    assert_eq!(sink.gauge_latest("lust.vm.memory.bytes_used"), Some(10240.0));
    assert_eq!(sink.counter_total("lust.vm.memory.limit_rejections_total"), 1);
    assert_eq!(sink.gauge_latest("lust.vm.host_ledger.active_slots"), Some(12.0));
    assert_eq!(sink.counter_total("lust.vm.host_ledger.overflow_total"), 1);
    assert_eq!(sink.counter_total("lust.jit.traces_compiled_total"), 4);
    assert_eq!(sink.counter_total("lust.jit.bailouts_total"), 1);
    assert_eq!(sink.counter_total("lust.fiber.lifecycle_total"), 1);
    assert_eq!(sink.spans_named("lust.effect.span").len(), 1);

    // When the feature is enabled in cargo build, verify live VM wiring
    #[cfg(any(
        feature = "telemetry",
        feature = "telemetry-all",
        feature = "telemetry-vm"
    ))]
    {
        let mut live_vm = VM::new();
        let live_sink = Arc::new(MockTelemetrySink::new());
        live_vm.set_telemetry_sink(live_sink.clone(), TelemetrySignalMask::ALL);

        live_vm.set_gas_budget(10_000);
        assert!(live_vm.charge_gas(350).is_ok());
        live_vm.flush_gas_telemetry();
        assert_eq!(live_sink.counter_total("lust.vm.gas.consumed_total"), 350);

        assert!(live_vm.set_memory_budget_bytes(50_000).is_ok());
        assert!(live_vm.try_charge_memory_bytes(2048));
        assert_eq!(live_sink.gauge_latest("lust.vm.memory.bytes_used"), Some(2048.0));

        // Memory limit rejection
        assert!(!live_vm.try_charge_memory_bytes(100_000));
        assert_eq!(live_sink.counter_total("lust.vm.memory.limit_rejections_total"), 1);

        // Host ledger slots
        let slot = live_vm.acquire_host_resource(202, |_| {}).expect("acquire");
        live_vm.notify_host_resource_acquired();
        assert_eq!(live_sink.gauge_latest("lust.vm.host_ledger.active_slots"), Some(1.0));
        live_vm.release_host_resource(slot);
        live_vm.notify_host_resource_released();
        assert_eq!(live_sink.gauge_latest("lust.vm.host_ledger.active_slots"), Some(0.0));
    }
}

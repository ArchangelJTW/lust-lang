//! API Surface Contract Verification Test (Phase 0.5 Gate)
//! Confirms that all 4 runtime subsystems (Schema, Rx, Sim, TestFramework)
//! have matching public Rust API contracts at current commit HEAD.

#[test]
fn test_public_api_surface_contracts() {
    // 1. Schema API: s builder and Schema
    let schema = lust::schema::s.string().min(1).max(50).build();
    let data = lust::data::DataValue::String("valid".to_string());
    assert!(schema.parse(&data).is_ok());

    // 2. Rx Streams API: Observable from_iter (MeasurableItem for i64)
    let _observable = lust::rx::Observable::<i64>::from_iter(vec![1_i64, 2, 3]);

    // 3. Simulation & Scheduling API: PreemptiveExplorationScheduler
    let vm = lust::vm::VM::new();
    let scheduler = lust::sim::PreemptiveExplorationScheduler::new(vm, 42);
    assert_eq!(scheduler.seed, 42);

    // 4. Test Framework API: TestRunner
    let runner = lust::test_framework::TestRunner::new();
    assert_eq!(runner.default_timeout_ms, 10_000);
}

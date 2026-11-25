//! Integration tests for script import of `@std/rx` and `Subject` in Lust.
//!
//! Verifies:
//! - Syntax forms: `use @std/rx`, `use @std/rx.{Subject}`, `use @std/rx.*`
//! - Canonical and alias variants: `import rx from "@std/rx"`, `import { Subject } from "@std/rx"`
//! - Virtual module mapping and capability gating.

use lust::config::LustConfig;
use lust::EmbeddedProgram;

#[test]
fn test_script_import_std_rx_module() {
    let result = EmbeddedProgram::builder()
        .module("main", "use @std/rx\nfunction test(): int return 1 end")
        .entry_module("main")
        .compile();
    assert!(result.is_ok(), "use @std/rx must compile: {:?}", result.err());
}

#[test]
fn test_script_import_std_rx_subject_item() {
    let result = EmbeddedProgram::builder()
        .module("main", "use @std/rx.{Subject}\nfunction test(): int return 1 end")
        .entry_module("main")
        .compile();
    assert!(
        result.is_ok(),
        "use @std/rx.{{Subject}} must compile: {:?}",
        result.err()
    );
}

#[test]
fn test_script_import_std_rx_observable_item() {
    let result = EmbeddedProgram::builder()
        .module("main", "use @std/rx.{Observable}\nfunction test(): int return 1 end")
        .entry_module("main")
        .compile();
    assert!(
        result.is_ok(),
        "use @std/rx.{{Observable}} must compile: {:?}",
        result.err()
    );
}

#[test]
fn test_script_import_std_rx_multiple_items() {
    let result = EmbeddedProgram::builder()
        .module(
            "main",
            "use @std/rx.{Subject, Observable, interval}\nfunction test(): int return 1 end",
        )
        .entry_module("main")
        .compile();
    assert!(
        result.is_ok(),
        "use @std/rx.{{Subject, Observable, interval}} must compile: {:?}",
        result.err()
    );
}

#[test]
fn test_script_import_std_rx_glob() {
    let result = EmbeddedProgram::builder()
        .module("main", "use @std/rx.*\nfunction test(): int return 1 end")
        .entry_module("main")
        .compile();
    assert!(
        result.is_ok(),
        "use @std/rx.* must compile: {:?}",
        result.err()
    );
}

#[test]
fn test_script_es_import_syntax() {
    let result = EmbeddedProgram::builder()
        .module("main", "import rx from \"@std/rx\";\nfunction test(): int return 1 end")
        .entry_module("main")
        .compile();
    assert!(
        result.is_ok(),
        "import rx from \"@std/rx\" must compile: {:?}",
        result.err()
    );
}

#[test]
fn test_script_import_std_rx_aliased_item() {
    let result = EmbeddedProgram::builder()
        .module(
            "main",
            "use @std/rx.{Subject as RxSubject}\nfunction test(): int return 1 end",
        )
        .entry_module("main")
        .compile();
    assert!(
        result.is_ok(),
        "use @std/rx.{{Subject as RxSubject}} must compile: {:?}",
        result.err()
    );
}

#[test]
fn test_script_import_std_rx_module_alias() {
    let result = EmbeddedProgram::builder()
        .module(
            "main",
            "use @std/rx as my_rx\nfunction test(): int return 1 end",
        )
        .entry_module("main")
        .compile();
    assert!(
        result.is_ok(),
        "use @std/rx as my_rx must compile: {:?}",
        result.err()
    );
}

#[test]
fn test_script_import_std_dot_rx_module() {
    let result = EmbeddedProgram::builder()
        .module("main", "use std.rx\nfunction test(): int return 1 end")
        .entry_module("main")
        .compile();
    assert!(result.is_ok(), "use std.rx must compile: {:?}", result.err());
}

#[test]
fn test_script_import_std_dot_rx_subject() {
    let result = EmbeddedProgram::builder()
        .module("main", "use std.rx.{Subject}\nfunction test(): int return 1 end")
        .entry_module("main")
        .compile();
    assert!(
        result.is_ok(),
        "use std.rx.{{Subject}} must compile: {:?}",
        result.err()
    );
}

#[test]
fn test_script_import_rx_capability_masking() {
    // 1. Strict capabilities without rx granted should fail closed
    let mut config = LustConfig::default();
    config.set_strict_capabilities(true);

    assert!(
        !config.is_std_module_allowed("@std/rx"),
        "strict capabilities without rx must deny access"
    );

    // 2. Explicitly granting rx must allow access
    config.allow_std_module("@std/rx");
    assert!(
        config.is_std_module_allowed("@std/rx"),
        "allowing @std/rx must pass capability check"
    );
    assert!(
        config.is_std_module_allowed("std.rx"),
        "allowing @std/rx must also allow std.rx"
    );
}

#[test]
fn test_script_import_std_rx_runtime_execution() {
    let source = r#"
use @std/rx
use @std/rx.{Subject, Observable, interval}

function run_rx(): int
    local s1 = Subject()
    local s2 = rx.Subject()
    local obs1 = Observable([10, 20])
    local obs2 = rx.Observable([1, 2, 3])
    local iv1 = interval(50)
    local iv2 = rx.interval(20)
    if s1["__type"] != "Subject" or s2["__type"] != "Subject" then
        return -1
    end
    if s1["closed"] or s2["closed"] then
        return -2
    end
    if obs1["__type"] != "Observable" or iv1["__type"] != "IntervalObservable" then
        return -3
    end
    return obs1["count"] + obs2["count"] + iv1["period_ms"] + iv2["period_ms"]
end
"#;
    let mut program = EmbeddedProgram::builder()
        .module("main", source)
        .entry_module("main")
        .compile()
        .expect("@std/rx runtime execution script must compile");
    let res = program
        .vm_mut()
        .call("main.run_rx", vec![])
        .expect("main.run_rx must execute without runtime errors");
    assert_eq!(res, lust::bytecode::Value::Int(75));
}

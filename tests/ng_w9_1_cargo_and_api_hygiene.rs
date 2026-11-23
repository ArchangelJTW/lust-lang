#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use lust::bindgen::RustBindingsBuilder;
use lust::packages::Semver;
use std::fs;
use std::path::PathBuf;
use tempfile::tempdir;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn cargo_toml_hygiene_and_optional_deps() {
    let cargo_toml_raw =
        fs::read_to_string(repo_root().join("Cargo.toml")).expect("read Cargo.toml");

    assert!(
        !cargo_toml_raw.contains("esp-println"),
        "Cargo.toml must not reference unused esp-println"
    );

    let parsed: toml::Value = toml::from_str(&cargo_toml_raw).expect("parse Cargo.toml");
    let deps = parsed["dependencies"]
        .as_table()
        .expect("[dependencies] table");
    assert!(!deps.contains_key("esp-println"));

    let tempfile_dep = deps["tempfile"]
        .as_table()
        .expect("tempfile must be an inline/table dependency");
    assert_eq!(
        tempfile_dep.get("optional").and_then(|v| v.as_bool()),
        Some(true),
        "tempfile must be optional = true"
    );

    let mimalloc_dep = deps["mimalloc"]
        .as_table()
        .expect("mimalloc must be a table dependency");
    assert_eq!(
        mimalloc_dep.get("optional").and_then(|v| v.as_bool()),
        Some(true),
        "mimalloc must be optional = true"
    );
    assert_eq!(
        mimalloc_dep
            .get("default-features")
            .and_then(|v| v.as_bool()),
        Some(false),
        "mimalloc must keep default-features = false"
    );

    let features = parsed["features"].as_table().expect("[features] table");
    let default_features: Vec<&str> = features["default"]
        .as_array()
        .expect("default feature array")
        .iter()
        .map(|v| v.as_str().expect("feature string"))
        .collect();
    assert!(
        default_features.contains(&"cli-mimalloc"),
        "default features must include cli-mimalloc"
    );

    let std_features: Vec<&str> = features["std"]
        .as_array()
        .expect("std feature array")
        .iter()
        .map(|v| v.as_str().expect("feature string"))
        .collect();
    assert!(
        std_features.contains(&"dep:tempfile"),
        "std feature must include dep:tempfile"
    );

    let cli_mimalloc_features: Vec<&str> = features["cli-mimalloc"]
        .as_array()
        .expect("cli-mimalloc feature array")
        .iter()
        .map(|v| v.as_str().expect("feature string"))
        .collect();
    assert_eq!(cli_mimalloc_features, vec!["dep:mimalloc"]);

    let esp_logging: Vec<&str> = features["esp32c6-logging"]
        .as_array()
        .expect("esp32c6-logging feature array")
        .iter()
        .map(|v| v.as_str().expect("feature string"))
        .collect();
    assert_eq!(esp_logging, vec!["dep:log"]);

    let unused_deps_lint = parsed["lints"]["cargo"]["unused_dependencies"].as_str();
    assert_eq!(
        unused_deps_lint,
        Some("warn"),
        "[lints.cargo] unused_dependencies must be set to \"warn\""
    );

    let main_rs = fs::read_to_string(repo_root().join("src/main.rs")).expect("read src/main.rs");
    let main_header: String = main_rs.lines().take(15).collect::<Vec<_>>().join("\n");
    assert!(
        main_header.contains("#[cfg(all(feature = \"cli-mimalloc\", not(target_arch = \"wasm32\")))]"),
        "src/main.rs must gate MiMalloc behind feature = \"cli-mimalloc\""
    );
}

#[test]
fn semver_parse_rejects_repeated_v_prefix() {
    assert_eq!(
        Semver::parse("v1.2.3"),
        Some(Semver {
            major: 1,
            minor: 2,
            patch: 3,
        })
    );
    assert_eq!(
        Semver::parse(" v1.2.3 "),
        Some(Semver {
            major: 1,
            minor: 2,
            patch: 3,
        })
    );
    assert_eq!(
        Semver::parse("1.2.3"),
        Some(Semver {
            major: 1,
            minor: 2,
            patch: 3,
        })
    );
    assert_eq!(
        Semver::parse("vv1.2.3"),
        None,
        "Semver::parse must reject repeated 'v' prefix"
    );
    assert_eq!(
        Semver::parse(" vv1.2.3 "),
        None,
        "Semver::parse must reject whitespace-padded repeated 'v' prefix"
    );
}

#[test]
fn bindgen_and_cli_trim_prefix_suffix_single_occurrence() {
    assert_eq!("a::::".trim_suffix("::"), "a::");
    assert_eq!("r#r#foo".trim_prefix("r#"), "r#foo");
    assert_eq!("pub fn pub fn foo()".trim_prefix("pub fn "), "pub fn foo()");

    let bindgen_src =
        fs::read_to_string(repo_root().join("src/bindgen.rs")).expect("read src/bindgen.rs");
    assert!(bindgen_src.contains("collision_key.trim_prefix(\"r#\")"));
    assert!(bindgen_src.contains("runtime_crate_path.trim_suffix(\"::\")"));
    assert!(!bindgen_src.contains("trim_start_matches(\"r#\")"));
    assert!(!bindgen_src.contains("trim_end_matches(\"::\")"));

    let ext_src =
        fs::read_to_string(repo_root().join("src/cli/ext.rs")).expect("read src/cli/ext.rs");
    assert!(ext_src.contains("sig.trim_prefix(\"pub fn \")"));
    assert!(!ext_src.contains("trim_start_matches(\"pub fn \")"));

    let deps_src = fs::read_to_string(repo_root().join("src/packages/dependencies.rs"))
        .expect("read src/packages/dependencies.rs");
    assert!(deps_src.contains("s.trim().trim_prefix('v')"));
    assert!(!deps_src.contains("trim_start_matches('v')"));

    let dir = tempdir().expect("create tempdir");
    let entry = dir.path().join("main.lust");
    fs::write(
        &entry,
        r#"
struct Player
    from_handle: int
end

---@bindgen
function ping(): int
    return 1
end
"#,
    )
    .expect("write entry.lust");

    let generated = RustBindingsBuilder::new(&entry)
        .runtime_crate_path("::lust")
        .generate()
        .expect("generate bindings");

    assert!(generated.source.contains("::lust::EmbeddedProgram"));
    assert!(generated.source.contains("pub fn get_from_handle(&self)"));
    assert!(generated.source.contains("pub fn ping(&mut self)"));
}

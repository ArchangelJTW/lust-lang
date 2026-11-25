#![cfg(feature = "std")]
use std::{path::PathBuf, process::Command};

#[test]
fn live_napi_rs_node_actor_acceptance() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let target = PathBuf::from("/tmp/lust_eval_target/w17-7");
    let build = Command::new("cargo").current_dir(&root).env("RUSTUP_HOME", "/Users/vedmalex/.rustup")
        .env("CARGO_HOME", "/Users/vedmalex/.cargo").args(["build", "--locked", "-p", "lust-napi-rs", "--target-dir"])
        .arg(&target).output().expect("build napi-rs addon");
    assert!(build.status.success(), "build failed: {}", String::from_utf8_lossy(&build.stderr));
    let ext = if cfg!(target_os="macos") { "dylib" } else if cfg!(target_os="windows") { "dll" } else { "so" };
    let library = target.join("debug").join(format!("liblust_napi_rs.{ext}"));
    assert!(library.exists(), "missing {}", library.display());
    let tmp = tempfile::tempdir().expect("tempdir");
    let addon = tmp.path().join("lust_napi_rs.node");
    std::fs::copy(library, &addon).expect("copy addon");
    let script = root.join("tests/fixtures/w17_node/acceptance.js");
    let node = Command::new("node").arg(script).env("LUST_NAPI_RS_ADDON", addon).output().expect("run Node");
    assert!(node.status.success(), "Node failed: {}", String::from_utf8_lossy(&node.stderr));
    assert!(String::from_utf8_lossy(&node.stdout).contains("LUST_NAPI_RS_ACCEPTANCE_OK"));
}

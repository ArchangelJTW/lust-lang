#![cfg(feature = "std")]

use std::path::PathBuf;
use std::process::Command;

fn node_binary() -> PathBuf {
    if let Ok(path) = std::env::var("LUST_NODE_BIN") {
        return PathBuf::from(path);
    }
    let vp_node = PathBuf::from("/Users/vedmalex/.vite-plus/bin/node");
    if vp_node.exists() {
        return vp_node;
    }
    PathBuf::from("node")
}

#[test]
fn w4_4c_c_node_api_cdylib_concurrent_pool_and_worker_termination() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let target_dir = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/lust-w4-4c"));

    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let build = Command::new(&cargo)
        .current_dir(&manifest_dir)
        .env("CARGO_TARGET_DIR", &target_dir)
        .args(["build", "--offline", "-p", "lust-napi"])
        .output()
        .expect("invoke cargo build -p lust-napi");
    assert!(
        build.status.success(),
        "cargo build -p lust-napi failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );

    let ext = if cfg!(target_os = "macos") {
        "dylib"
    } else if cfg!(target_os = "windows") {
        "dll"
    } else {
        "so"
    };
    let dylib_path = target_dir.join("debug").join(format!("liblust_napi.{ext}"));
    assert!(
        dylib_path.exists(),
        "built cdylib missing at {}",
        dylib_path.display()
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    let addon_path = tmp.path().join("lust_napi.node");
    std::fs::copy(&dylib_path, &addon_path).expect("copy cdylib to .node addon");

    let script = manifest_dir.join("tests/fixtures/w4_node/acceptance.js");
    let node_bin = node_binary();
    let out = Command::new(&node_bin)
        .arg(&script)
        .env("LUST_NAPI_ADDON_PATH", &addon_path)
        .output()
        .expect("execute node acceptance.js");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "node acceptance script failed (status={:?}):\nstdout: {}\nstderr: {}",
        out.status.code(),
        stdout,
        stderr
    );
    assert!(
        stdout.contains("LUST_NAPI_ACCEPTANCE_OK"),
        "missing expected Node acceptance marker in stdout: {}",
        stdout
    );
}

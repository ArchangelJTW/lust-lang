//! Snapshot testing utilities (`T-SNAP`).

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
use crate::bytecode::Value;

pub const HEADER: &str = "--- Lust Snapshot v1 ---";

/// Determines the snapshot storage path: `<test_dir>/__snapshots__/<test_file_stem>.snap`.
pub fn snapshot_path(test_file: &Path) -> PathBuf {
    let dir = test_file.parent().unwrap_or_else(|| Path::new("."));
    let stem = test_file.file_stem().unwrap_or_default();
    dir.join("__snapshots__")
        .join(format!("{}.snap", stem.to_string_lossy()))
}

/// Deterministically formats a `Value` for snapshot recording.
/// Maps have their keys sorted lexicographically to guarantee determinism.
pub fn canonical_value_repr(val: &Value) -> String {
    match val {
        Value::Nil => "nil".to_string(),
        Value::Bool(b) => format!("{b}"),
        Value::Int(i) => format!("{i}"),
        Value::Float(f) => format!("{f}"),
        Value::String(s) => format!("\"{}\"", s.escape_default()),
        Value::Array(arr) => {
            let items: Vec<String> = arr.borrow().iter().map(canonical_value_repr).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Map(m) => {
            let map = m.borrow();
            let mut entries: Vec<(String, String)> = map
                .iter()
                .map(|(k, v)| {
                    let key_str = match k.to_value() {
                        Value::Int(i) => format!("{i}"),
                        Value::String(s) => format!("\"{}\"", s.escape_default()),
                        _ => format!("{:?}", k),
                    };
                    (key_str, canonical_value_repr(v))
                })
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            let formatted_entries: Vec<String> = entries
                .into_iter()
                .map(|(k, v)| format!("{k}: {v}"))
                .collect();
            format!("{{{}}}", formatted_entries.join(", "))
        }
        _ => format!("{:?}", val),
    }
}

/// Parses an existing `.snap` file into sorted test name -> value map.
pub fn parse_snapshot_file(content: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let mut current_name: Option<String> = None;
    let mut current_lines: Vec<&str> = Vec::new();

    for line in content.lines() {
        if line.starts_with("### ") {
            if let Some(name) = current_name.take() {
                map.insert(name, current_lines.join("\n"));
                current_lines.clear();
            }
            current_name = Some(line[4..].trim().to_string());
        } else if current_name.is_some() {
            current_lines.push(line);
        }
    }
    if let Some(name) = current_name {
        map.insert(name, current_lines.join("\n"));
    }
    map
}

/// Serializes key-value pairs deterministically into standard `.snap` text format.
pub fn serialize(entries: &[(String, String)]) -> String {
    let mut e = entries.to_vec();
    e.sort_by(|a, b| a.0.cmp(&b.0));
    let mut s = format!("{}\n", HEADER);
    for (n, v) in e {
        s.push_str(&format!("### {}\n{}\n", n, v.trim_end()));
    }
    s
}

/// Computes a line-by-line unified diff between expected and actual snapshot texts.
pub fn unified_diff(expected: &str, actual: &str) -> String {
    let mut out = String::from("--- expected\n+++ actual\n");
    let exp_lines: Vec<&str> = expected.lines().collect();
    let act_lines: Vec<&str> = actual.lines().collect();
    let max_len = exp_lines.len().max(act_lines.len());
    for i in 0..max_len {
        let e = exp_lines.get(i).copied();
        let a = act_lines.get(i).copied();
        match (e, a) {
            (Some(el), Some(al)) if el == al => {
                out.push_str(&format!(" {el}\n"));
            }
            (Some(el), Some(al)) => {
                out.push_str(&format!("-{el}\n+{al}\n"));
            }
            (Some(el), None) => {
                out.push_str(&format!("-{el}\n"));
            }
            (None, Some(al)) => {
                out.push_str(&format!("+{al}\n"));
            }
            (None, None) => {}
        }
    }
    out
}

/// Atomically writes contents to the target path using a temporary sibling file.
pub fn write_atomic(path: &Path, contents: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("snap.tmp");
    fs::write(&tmp, contents)?;
    fs::rename(tmp, path)
}

/// Verifies or updates a snapshot entry in the appropriate `.snap` file.
pub fn check_or_update_snapshot(
    test_file: &Path,
    name: &str,
    actual_str: &str,
    update: bool,
) -> Result<(), String> {
    let snap_file = snapshot_path(test_file);
    let mut entries = if snap_file.exists() {
        let content = fs::read_to_string(&snap_file)
            .map_err(|e| format!("Failed to read snapshot file '{}': {e}", snap_file.display()))?;
        parse_snapshot_file(&content)
    } else {
        BTreeMap::new()
    };

    if update {
        entries.insert(name.to_string(), actual_str.trim_end().to_string());
        let list: Vec<(String, String)> = entries.into_iter().collect();
        let serialized = serialize(&list);
        write_atomic(&snap_file, &serialized)
            .map_err(|e| format!("Failed to write snapshot file '{}': {e}", snap_file.display()))?;
        Ok(())
    } else {
        match entries.get(name) {
            Some(expected) => {
                if expected == actual_str.trim_end() {
                    Ok(())
                } else {
                    let diff = unified_diff(expected, actual_str.trim_end());
                    Err(format!("Snapshot mismatch for '{name}':\n{diff}"))
                }
            }
            None => Err(format!(
                "Snapshot missing for '{name}' in '{}'. Run with --update-snapshots to create it.",
                snap_file.display()
            )),
        }
    }
}

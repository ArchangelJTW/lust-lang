//! Cross-thread transfer rules, owned deep-materialization payloads (`Send + Sync`),
//! cancellable worker lifecycle, and real `rustc` compile-fail harness for `!Send` views/handles.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};

use crate::bytecode::{Value, ValueKey};
use crate::data::document::Document;
use crate::data::number::ExactNumber;
use crate::data::owner::OwnerId;
use crate::data::value::DataValue;

use super::generation::{GenArrayHandle, GenMapHandle, GenStructHandle, LifecycleError};

/// Explicitly `Send + Sync` owned payload produced exclusively by validated deep materialization
/// of live VM data. Carries zero `Rc`, `RefCell`, or borrowed VM pointers.
#[derive(Debug, Clone, PartialEq)]
pub struct OwnedTransferPayload {
    value: DataValue,
    source_vm_generation: u64,
    source_owner_id: OwnerId,
}

// Compile-time proof that `OwnedTransferPayload` (explicitly designed for cross-thread transfer)
// satisfies `Send + Sync`.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<OwnedTransferPayload>();
};

impl OwnedTransferPayload {
    pub fn value(&self) -> &DataValue {
        &self.value
    }

    pub fn into_value(self) -> DataValue {
        self.value
    }

    pub fn source_vm_generation(&self) -> u64 {
        self.source_vm_generation
    }

    pub fn source_owner_id(&self) -> OwnerId {
        self.source_owner_id
    }
}

/// Converts a live Lust runtime `Value` into a fully independent, `Send + Sync` `DataValue`.
/// Fails closed on functions, closures, native handles, tasks, and weak references.
pub fn deep_materialize_to_data_value(value: &Value) -> Result<DataValue, LifecycleError> {
    deep_materialize_bounded(value, 64)
}

fn deep_materialize_bounded(
    value: &Value,
    depth_remaining: usize,
) -> Result<DataValue, LifecycleError> {
    if depth_remaining == 0 {
        return Err(LifecycleError::UnsupportedTransfer(
            "nesting depth limit exceeded while materializing transfer payload",
        ));
    }
    match value {
        Value::Nil => Ok(DataValue::Null),
        Value::Bool(b) => Ok(DataValue::Bool(*b)),
        Value::Int(i) => Ok(DataValue::Number(ExactNumber::Int(*i))),
        Value::Float(f) => Ok(DataValue::Number(ExactNumber::Float(*f))),
        Value::String(s) => Ok(DataValue::String((**s).clone())),
        Value::Array(arr) => {
            let borrowed = arr.borrow();
            let mut out = Vec::with_capacity(borrowed.len());
            for item in borrowed.iter() {
                out.push(deep_materialize_bounded(item, depth_remaining - 1)?);
            }
            Ok(DataValue::Array(out))
        }
        Value::Tuple(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items.iter() {
                out.push(deep_materialize_bounded(item, depth_remaining - 1)?);
            }
            Ok(DataValue::Array(out))
        }
        Value::Map(map) => {
            let borrowed = map.borrow();
            let mut doc = Document::with_capacity(borrowed.len());
            for (k, v) in borrowed.iter() {
                let key_str = value_key_to_string(k);
                doc.insert(key_str, deep_materialize_bounded(v, depth_remaining - 1)?);
            }
            Ok(DataValue::Document(doc))
        }
        Value::Struct(obj) => {
            let fields = obj.fields.borrow();
            let names = obj.layout.field_names();
            let mut doc = Document::with_capacity(fields.len());
            for (idx, field_val) in fields.iter().enumerate() {
                let field_name = names
                    .get(idx)
                    .map(|n| n.as_str().to_string())
                    .unwrap_or_else(|| format!("field_{idx}"));
                let materialized = obj.layout.materialize_field_value(idx, field_val.clone());
                doc.insert(
                    field_name,
                    deep_materialize_bounded(&materialized, depth_remaining - 1)?,
                );
            }
            Ok(DataValue::Document(doc))
        }
        Value::Enum(obj) => {
            let mut doc = Document::new();
            doc.insert("enum", DataValue::String(obj.enum_name.as_str().to_string()));
            doc.insert("variant", DataValue::String(obj.variant.as_str().to_string()));
            if let Some(vals) = &obj.values {
                let mut payload = Vec::with_capacity(vals.len());
                for v in vals {
                    payload.push(deep_materialize_bounded(v, depth_remaining - 1)?);
                }
                doc.insert("payload", DataValue::Array(payload));
            }
            Ok(DataValue::Document(doc))
        }
        _ => Err(LifecycleError::UnsupportedTransfer(
            "non-Send runtime value (function/closure/task/weak) cannot transfer across threads",
        )),
    }
}

/// Validates live VM and owner generation, then deep-materializes a struct handle into a
/// thread-safe `OwnedTransferPayload`.
pub fn materialize_struct_for_transfer(
    handle: &GenStructHandle,
) -> Result<OwnedTransferPayload, LifecycleError> {
    let raw = handle.checked_raw_handle()?;
    let value = deep_materialize_to_data_value(raw.as_value())?;
    Ok(OwnedTransferPayload {
        value,
        source_vm_generation: handle.vm_generation(),
        source_owner_id: handle.owner_id(),
    })
}

/// Validates live VM and owner generation, then deep-materializes an array handle into a
/// thread-safe `OwnedTransferPayload`.
pub fn materialize_array_for_transfer(
    handle: &GenArrayHandle,
) -> Result<OwnedTransferPayload, LifecycleError> {
    let raw = handle.checked_raw_handle()?;
    let items: Vec<DataValue> = {
        let slice = raw.borrow();
        let mut out = Vec::with_capacity(slice.len());
        for item in slice.iter() {
            out.push(deep_materialize_to_data_value(item)?);
        }
        out
    };
    Ok(OwnedTransferPayload {
        value: DataValue::Array(items),
        source_vm_generation: handle.vm_generation(),
        source_owner_id: handle.owner_id(),
    })
}

fn value_key_to_string(k: &ValueKey) -> String {
    match k.owned_values().0 {
        Value::String(s) => (**s).clone(),
        Value::Int(i) => i.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Nil => "nil".to_string(),
        other => format!("{other:?}"),
    }
}

/// Validates live VM and owner generation, then deep-materializes a map handle into a
/// thread-safe `OwnedTransferPayload`.
pub fn materialize_map_for_transfer(
    handle: &GenMapHandle,
) -> Result<OwnedTransferPayload, LifecycleError> {
    let raw = handle.checked_raw_handle()?;
    let mut doc = Document::new();
    {
        let map = raw.borrow();
        for (k, v) in map.iter() {
            let key_str = value_key_to_string(k);
            doc.insert(key_str, deep_materialize_to_data_value(v)?);
        }
    }
    Ok(OwnedTransferPayload {
        value: DataValue::Document(doc),
        source_vm_generation: handle.vm_generation(),
        source_owner_id: handle.owner_id(),
    })
}

/// Handle to a real OS thread executing over an `OwnedTransferPayload` with explicit
/// cancellation and join acknowledgement.
pub struct TransferTaskHandle<R: Send + 'static> {
    cancel_flag: Arc<AtomicBool>,
    finalized_flag: Arc<AtomicBool>,
    join_handle: Option<JoinHandle<R>>,
}

impl<R: Send + 'static> TransferTaskHandle<R> {
    /// Spawns a real `std::thread::spawn` worker consuming an `OwnedTransferPayload`.
    pub fn spawn<F>(payload: OwnedTransferPayload, worker: F) -> Self
    where
        F: FnOnce(OwnedTransferPayload, Arc<AtomicBool>) -> R + Send + 'static,
    {
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let finalized_flag = Arc::new(AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancel_flag);
        let worker_finalized = Arc::clone(&finalized_flag);
        let join_handle = thread::spawn(move || {
            let res = worker(payload, worker_cancel);
            worker_finalized.store(true, Ordering::Release);
            res
        });
        Self {
            cancel_flag,
            finalized_flag,
            join_handle: Some(join_handle),
        }
    }

    pub fn signal_cancel(&self) {
        self.cancel_flag.store(true, Ordering::Release);
    }

    pub fn is_finalized(&self) -> bool {
        self.finalized_flag.load(Ordering::Acquire)
    }

    pub fn join(mut self) -> thread::Result<R> {
        self.join_handle
            .take()
            .expect("TransferTaskHandle joined more than once")
            .join()
    }

    pub fn cancel_and_join(self) -> thread::Result<R> {
        self.signal_cancel();
        self.join()
    }
}

/// Result of running an actual `rustc` compile-fail check for `!Send` cross-thread transfer.
#[derive(Debug, Clone)]
pub struct CompileFailResult {
    pub case_name: String,
    pub exit_code: Option<i32>,
    pub stderr: String,
    pub rejected_as_not_send: bool,
}

/// Resolves the active Cargo profile output directory (`<target_dir>/debug` or `release`)
/// from the running test executable, falling back to `fallback_target_dir/debug`.
fn resolve_profile_dir(fallback_target_dir: &str) -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.ancestors().skip(1) {
            if ancestor.join("build").is_dir() || ancestor.join("deps").is_dir() {
                return ancestor.to_path_buf();
            }
        }
    }
    PathBuf::from(fallback_target_dir).join("debug")
}

fn collect_dependency_dirs_and_liblust(
    profile_dir: &PathBuf,
) -> Result<(Vec<PathBuf>, PathBuf), String> {
    let mut dep_dirs = Vec::new();
    let mut rlib_candidates: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();

    let scan_dir = |dir: &PathBuf,
                    dep_dirs: &mut Vec<PathBuf>,
                    rlibs: &mut Vec<(std::time::SystemTime, PathBuf)>| {
        if !dir.is_dir() {
            return;
        }
        dep_dirs.push(dir.clone());
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    if (name == "liblust.rlib"
                        || (name.starts_with("liblust-") && name.ends_with(".rlib")))
                        && p.is_file()
                    {
                        let mtime = entry
                            .metadata()
                            .and_then(|m| m.modified())
                            .unwrap_or(std::time::UNIX_EPOCH);
                        rlibs.push((mtime, p));
                    }
                }
            }
        }
    };

    scan_dir(&profile_dir.join("deps"), &mut dep_dirs, &mut rlib_candidates);

    let build_dir = profile_dir.join("build");
    if let Ok(pkgs) = fs::read_dir(&build_dir) {
        for pkg in pkgs.flatten() {
            let pkg_path = pkg.path();
            if !pkg_path.is_dir() {
                continue;
            }
            if let Ok(hashes) = fs::read_dir(&pkg_path) {
                for hash_dir in hashes.flatten() {
                    let out_dir = hash_dir.path().join("out");
                    scan_dir(&out_dir, &mut dep_dirs, &mut rlib_candidates);
                }
            }
        }
    }

    rlib_candidates.sort_by_key(|(mtime, _)| *mtime);
    let liblust_rlib = rlib_candidates
        .pop()
        .map(|(_, p)| p)
        .ok_or_else(|| {
            format!(
                "could not locate liblust*.rlib under {}",
                profile_dir.display()
            )
        })?;
    Ok((dep_dirs, liblust_rlib))
}

/// Compiles a real Rust snippet against the built `liblust.rlib` using `rustc --edition=2024`
/// and verifies that attempting `std::thread::spawn` with a `!Send` view/handle fails with
/// Rust's genuine `Send` trait-bound error (`E0277`).
pub fn verify_compile_fail_not_send(
    case_name: &str,
    snippet_body: &str,
    target_dir: &str,
) -> Result<CompileFailResult, String> {
    let profile_dir = resolve_profile_dir(target_dir);
    let work_dir = profile_dir.join("compile_fail_probes");
    fs::create_dir_all(&work_dir).map_err(|e| format!("create compile_fail dir: {e}"))?;

    let (dep_dirs, liblust_rlib) = collect_dependency_dirs_and_liblust(&profile_dir)?;

    let src_path = work_dir.join(format!("{case_name}.rs"));
    let out_path = work_dir.join(format!("{case_name}.bin"));

    let full_source = format!(
        r#"#![forbid(unsafe_code)]
#[allow(unused_imports)]
use lust::{{alloc_budget, ast, bytecode, data, embed, error, intern, lexer, parser, typechecker, vm}};

#[path = "{w17_mod}"]
#[allow(dead_code)]
mod w17_lifetime;

fn main() {{
{snippet_body}
}}
"#,
        w17_mod = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("w17_lifetime")
            .join("mod.rs")
            .display(),
        snippet_body = snippet_body,
    );

    fs::write(&src_path, full_source).map_err(|e| format!("write probe src: {e}"))?;

    let mut cmd = Command::new("rustc");
    cmd.arg("--edition=2024")
        .arg("--crate-type=bin")
        .arg("--extern")
        .arg(format!("lust={}", liblust_rlib.display()));
    for dir in &dep_dirs {
        cmd.arg("-L").arg(format!("dependency={}", dir.display()));
    }
    let output = cmd
        .arg(&src_path)
        .arg("-o")
        .arg(&out_path)
        .output()
        .map_err(|e| format!("spawn rustc: {e}"))?;

    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let rejected_as_not_send = !output.status.success()
        && (stderr.contains("cannot be sent between threads safely")
            || stderr.contains("is not implemented for"))
        && stderr.contains("Send");

    Ok(CompileFailResult {
        case_name: case_name.to_string(),
        exit_code: output.status.code(),
        stderr,
        rejected_as_not_send,
    })
}

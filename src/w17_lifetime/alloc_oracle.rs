//! Strictly safe-Rust (`#![forbid(unsafe_code)]`) isolated allocation counter oracle for ADR-003.
//!
//! Uses a host-compiled `DYLD_INTERPOSE` (macOS) shared library in an isolated child process
//! to count every OS heap call (`malloc`, `calloc`, `realloc`, `posix_memalign`) on the exact
//! measured pthread without authoring any `unsafe` Rust code (`TASK_ALLOWS_UNSAFE=0`).
//!
//! Control markers use stack-only `std::env::var_os` sentinel lookups intercepted by `getenv`,
//! verified by a zero-iteration null-interval control (must measure 0) and a positive
//! `Vec::with_capacity(64)` canary control (must measure >= 1).

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const ALLOC_BEGIN_SENTINEL: &str = "__LUST_W17_ALLOC_BEGIN__";
pub const ALLOC_END_SENTINEL: &str = "__LUST_W17_ALLOC_END__";
pub const ALLOC_REPORT_ENV: &str = "LUST_W17_ALLOC_REPORT_PATH";

/// Exact OS-level heap call counts captured on the measured thread during a single interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AllocCounterReport {
    pub malloc_calls: u64,
    pub calloc_calls: u64,
    pub realloc_calls: u64,
    pub posix_memalign_calls: u64,
}

impl AllocCounterReport {
    pub fn total(&self) -> u64 {
        self.malloc_calls
            + self.calloc_calls
            + self.realloc_calls
            + self.posix_memalign_calls
    }
}

/// Complete ADR-003 allocation oracle evidence captured from an isolated child process run.
#[derive(Debug, Clone)]
pub struct AllocOracleEvidence {
    pub null_control: AllocCounterReport,
    pub canary_control: AllocCounterReport,
    pub struct_view_10k: AllocCounterReport,
    pub struct_view_10k_sum: i64,
    pub gen_handle_10k: AllocCounterReport,
    pub gen_handle_10k_sum: i64,
    pub contiguous_10k: AllocCounterReport,
    pub contiguous_10k_sum: i64,
    pub child_stdout: String,
    pub child_stderr: String,
}

const DYLD_INTERPOSER_C: &str = r#"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdatomic.h>
#include <pthread.h>

static _Atomic int g_active = 0;
static _Atomic uint64_t g_malloc = 0;
static _Atomic uint64_t g_calloc = 0;
static _Atomic uint64_t g_realloc = 0;
static _Atomic uint64_t g_memalign = 0;
static _Atomic uint64_t g_interval_idx = 0;
static pthread_t g_target_thread;
static int g_has_target = 0;
static char g_report_path[1024] = {0};

static inline int should_count(void) {
    return atomic_load_explicit(&g_active, memory_order_acquire)
        && g_has_target
        && pthread_equal(pthread_self(), g_target_thread);
}

void* w17_malloc(size_t sz) {
    if (should_count()) {
        atomic_fetch_add_explicit(&g_malloc, 1, memory_order_relaxed);
    }
    return malloc(sz);
}

void* w17_calloc(size_t count, size_t sz) {
    if (should_count()) {
        atomic_fetch_add_explicit(&g_calloc, 1, memory_order_relaxed);
    }
    return calloc(count, sz);
}

void* w17_realloc(void* ptr, size_t sz) {
    if (should_count()) {
        atomic_fetch_add_explicit(&g_realloc, 1, memory_order_relaxed);
    }
    return realloc(ptr, sz);
}

int w17_posix_memalign(void** memptr, size_t alignment, size_t sz) {
    if (should_count()) {
        atomic_fetch_add_explicit(&g_memalign, 1, memory_order_relaxed);
    }
    return posix_memalign(memptr, alignment, sz);
}

static void record_interval_report(void) {
    uint64_t idx = atomic_fetch_add_explicit(&g_interval_idx, 1, memory_order_relaxed);
    uint64_t m = atomic_load_explicit(&g_malloc, memory_order_acquire);
    uint64_t c = atomic_load_explicit(&g_calloc, memory_order_acquire);
    uint64_t r = atomic_load_explicit(&g_realloc, memory_order_acquire);
    uint64_t p = atomic_load_explicit(&g_memalign, memory_order_acquire);
    if (g_report_path[0] != '\0') {
        FILE* f = fopen(g_report_path, "a");
        if (f != NULL) {
            fprintf(f, "INTERVAL=%llu malloc=%llu calloc=%llu realloc=%llu posix_memalign=%llu\n",
                    (unsigned long long)idx,
                    (unsigned long long)m,
                    (unsigned long long)c,
                    (unsigned long long)r,
                    (unsigned long long)p);
            fclose(f);
        }
    }
}

char* w17_getenv(const char* name) {
    if (name != NULL) {
        if (strcmp(name, "__LUST_W17_ALLOC_BEGIN__") == 0) {
            if (g_report_path[0] == '\0') {
                const char* rp = getenv("LUST_W17_ALLOC_REPORT_PATH");
                if (rp != NULL) {
                    strncpy(g_report_path, rp, sizeof(g_report_path) - 1);
                }
            }
            atomic_store_explicit(&g_malloc, 0, memory_order_seq_cst);
            atomic_store_explicit(&g_calloc, 0, memory_order_seq_cst);
            atomic_store_explicit(&g_realloc, 0, memory_order_seq_cst);
            atomic_store_explicit(&g_memalign, 0, memory_order_seq_cst);
            g_target_thread = pthread_self();
            g_has_target = 1;
            atomic_store_explicit(&g_active, 1, memory_order_release);
            return NULL;
        }
        if (strcmp(name, "__LUST_W17_ALLOC_END__") == 0) {
            atomic_store_explicit(&g_active, 0, memory_order_release);
            record_interval_report();
            return NULL;
        }
    }
    return getenv(name);
}

#define DYLD_INTERPOSE(_replacement, _replacee) \
    __attribute__((used)) static struct { const void* replacement; const void* replacee; } _interpose_##_replacee \
    __attribute__((section("__DATA,__interpose"))) = { (const void*)(unsigned long)&_replacement, (const void*)(unsigned long)&_replacee };

DYLD_INTERPOSE(w17_malloc, malloc)
DYLD_INTERPOSE(w17_calloc, calloc)
DYLD_INTERPOSE(w17_realloc, realloc)
DYLD_INTERPOSE(w17_posix_memalign, posix_memalign)
DYLD_INTERPOSE(w17_getenv, getenv)
"#;

/// Starts an OS-level thread-isolated heap allocation interval without any Rust `unsafe`.
#[inline(never)]
pub fn emit_alloc_interval_begin() {
    let _ = env::var_os(ALLOC_BEGIN_SENTINEL);
}

/// Ends the active OS-level thread-isolated heap allocation interval and flushes counts.
#[inline(never)]
pub fn emit_alloc_interval_end() {
    let _ = env::var_os(ALLOC_END_SENTINEL);
}

/// Builds the host `DYLD_INTERPOSE` allocation-counting dylib inside `target_dir`.
pub fn build_dyld_alloc_interposer(target_dir: &str) -> Result<PathBuf, String> {
    let dir = PathBuf::from(target_dir).join("alloc_interpose");
    fs::create_dir_all(&dir).map_err(|e| format!("create alloc_interpose dir: {e}"))?;
    let c_path = dir.join("w17_alloc_interpose.c");
    let dylib_path = dir.join("libw17_alloc_interpose.dylib");
    fs::write(&c_path, DYLD_INTERPOSER_C).map_err(|e| format!("write interposer C: {e}"))?;

    let status = Command::new("cc")
        .arg("-O2")
        .arg("-shared")
        .arg("-fPIC")
        .arg(&c_path)
        .arg("-o")
        .arg(&dylib_path)
        .output()
        .map_err(|e| format!("invoke host cc: {e}"))?;

    if !status.status.success() {
        return Err(format!(
            "cc failed building alloc interposer (exit {:?}): {}",
            status.status.code(),
            String::from_utf8_lossy(&status.stderr)
        ));
    }
    Ok(dylib_path)
}

pub fn parse_alloc_reports(report_path: &Path) -> Result<Vec<AllocCounterReport>, String> {
    let content = fs::read_to_string(report_path)
        .map_err(|e| format!("read alloc report {}: {e}", report_path.display()))?;
    let mut reports = Vec::new();
    for line in content.lines() {
        if !line.starts_with("INTERVAL=") {
            continue;
        }
        let mut fields = BTreeMap::new();
        for tok in line.split_whitespace() {
            if let Some((k, v)) = tok.split_once('=') {
                if let Ok(parsed) = v.parse::<u64>() {
                    fields.insert(k, parsed);
                }
            }
        }
        reports.push(AllocCounterReport {
            malloc_calls: fields.get("malloc").copied().unwrap_or(0),
            calloc_calls: fields.get("calloc").copied().unwrap_or(0),
            realloc_calls: fields.get("realloc").copied().unwrap_or(0),
            posix_memalign_calls: fields.get("posix_memalign").copied().unwrap_or(0),
        });
    }
    Ok(reports)
}

/// Spawns the current test binary in an isolated child process with `DYLD_INSERT_LIBRARIES`
/// running `child_test_name` (`--exact --nocapture --test-threads=1`) and parses the recorded
/// interval reports plus verified checksum outputs.
pub fn run_isolated_alloc_child(
    target_dir: &str,
    child_test_name: &str,
    child_mode_env: &str,
) -> Result<(Vec<AllocCounterReport>, String, String), String> {
    let dylib = build_dyld_alloc_interposer(target_dir)?;
    let report_path = PathBuf::from(target_dir)
        .join("alloc_interpose")
        .join(format!("report_{}.log", std::process::id()));
    let _ = fs::remove_file(&report_path);

    let current_exe =
        env::current_exe().map_err(|e| format!("resolve current test executable: {e}"))?;

    let output = Command::new(&current_exe)
        .arg(child_test_name)
        .arg("--exact")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env("DYLD_INSERT_LIBRARIES", &dylib)
        .env(ALLOC_REPORT_ENV, &report_path)
        .env(child_mode_env, "1")
        .output()
        .map_err(|e| format!("spawn isolated alloc child: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !output.status.success() {
        return Err(format!(
            "isolated alloc child failed (exit {:?})\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            stdout,
            stderr
        ));
    }

    let reports = parse_alloc_reports(&report_path)?;
    let _ = fs::remove_file(&report_path);
    Ok((reports, stdout, stderr))
}

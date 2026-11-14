//! Lust CLI Test Runner (`lust test`)
//!
//! Discovers and executes Lust test scripts with fiber-isolated failure containment,
//! lifecycle hooks (before/after), test modifiers (skip, timeout), and multiple
//! reporter formats (Console Pretty, TAP v13, JSON).

use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{Duration, Instant};

use lust::ast::{Item, ItemKind, Span, Stmt, SuiteDecl, TestDecl, TestModifier};
use lust::bytecode::{native_fn, LustMap, NativeCallResult, Value, ValueKey};
use lust::config::LustConfig;
use lust::modules::{compile_program_with_config, ModuleLoader, Program};
use lust::test_framework::{
    AssertError, Reporter, ReporterKind, TestOutcome, TestResult, TestRunner, TestSuiteResult,
    TestSummary,
};
use std::cell::{Cell, RefCell};
use lust::VM;

thread_local! {
    static CURRENT_TEST_FILE: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
    static CURRENT_UPDATE_SNAPSHOTS: Cell<bool> = const { Cell::new(false) };
    static RECORDED_BENCHMARKS: RefCell<Vec<lust::test_framework::bench::BenchResult>> = const { RefCell::new(Vec::new()) };
}

/// Command-line configuration for `lust test`.
#[derive(Debug, Clone)]
pub struct TestRunOptions {
    pub paths: Vec<PathBuf>,
    pub filter: Option<String>,
    pub reporter: ReporterKind,
    pub concurrency: usize,
    pub timeout_ms: u64,
    pub bail: bool,
    pub backend: String,
    pub verbose: bool,
    pub junit_path: Option<PathBuf>,
    pub update_snapshots: bool,
    pub bench: bool,
}

impl Default for TestRunOptions {
    fn default() -> Self {
        Self {
            paths: Vec::new(),
            filter: None,
            reporter: ReporterKind::Pretty,
            concurrency: 1,
            timeout_ms: 10_000,
            bail: false,
            backend: "vm".to_string(),
            verbose: false,
            junit_path: None,
            update_snapshots: false,
            bench: false,
        }
    }
}

/// Parses command line arguments following `lust test`.
pub fn parse_test_options(args: &[String]) -> Result<TestRunOptions, String> {
    let mut options = TestRunOptions::default();
    let mut i = 0;

    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "-h" | "--help" => {
                print_test_help();
                process::exit(0);
            }
            "-f" | "--filter" => {
                i += 1;
                if i >= args.len() {
                    return Err("Error: --filter requires a pattern argument".to_string());
                }
                options.filter = Some(args[i].clone());
            }
            opt if opt.starts_with("--filter=") => {
                options.filter = Some(opt[9..].to_string());
            }
            "--format" => {
                i += 1;
                if i >= args.len() {
                    return Err("Error: --format requires a format argument (console, tap, json)".to_string());
                }
                match args[i].as_str() {
                    "console" | "pretty" => options.reporter = ReporterKind::Pretty,
                    "tap" => options.reporter = ReporterKind::Tap,
                    "json" => options.reporter = ReporterKind::Json,
                    "junit" => options.reporter = ReporterKind::Junit,
                    other => {
                        return Err(format!(
                            "Error: unknown format '{}'. Supported: console, tap, json",
                            other
                        ));
                    }
                }
            }
            "--junit" => { i += 1; if i >= args.len() { return Err("Error: --junit requires a path".into()); } options.reporter=ReporterKind::Junit; options.junit_path=Some(PathBuf::from(&args[i])); }
            opt if opt.starts_with("--junit=") => { options.reporter=ReporterKind::Junit; options.junit_path=Some(PathBuf::from(&opt[8..])); }
            "--update-snapshots" | "-u" => options.update_snapshots = true,
            "--bench" => options.bench = true,
            "--bail" => {
                options.bail = true;
            }
            "-j" | "--jobs" => {
                i += 1;
                if i >= args.len() {
                    return Err("Error: --jobs requires an integer argument".to_string());
                }
                options.concurrency = args[i]
                    .parse()
                    .map_err(|_| format!("Error: invalid jobs count '{}'", args[i]))?;
            }
            "--timeout" => {
                i += 1;
                if i >= args.len() {
                    return Err("Error: --timeout requires a millisecond value".to_string());
                }
                options.timeout_ms = args[i]
                    .parse()
                    .map_err(|_| format!("Error: invalid timeout value '{}'", args[i]))?;
            }
            "--backend" => {
                i += 1;
                if i >= args.len() {
                    return Err("Error: --backend requires an argument (vm, auto)".to_string());
                }
                options.backend = args[i].clone();
            }
            "-v" | "--verbose" => {
                options.verbose = true;
            }
            "--virtual-clock" => {
                // Reserved for virtual time testing flag
            }
            opt if opt.starts_with('-') => {
                return Err(format!("Unknown option '{}'", opt));
            }
            path_str => {
                options.paths.push(PathBuf::from(path_str));
            }
        }
        i += 1;
    }

    if options.paths.is_empty() {
        options.paths.push(PathBuf::from("tests"));
    }

    if std::env::var("LUST_UPDATE_SNAPSHOTS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        options.update_snapshots = true;
    }

    if let Some(pattern) = options.filter.as_deref() {
        lust::regex::BoundedRegex::new(pattern)
            .map_err(|e| format!("Error: invalid --filter regex '{}': {}", pattern, e))?;
    }

    Ok(options)
}

fn print_test_help() {
    println!("Lust Test Runner (`@std/test`)");
    println!();
    println!("USAGE:");
    println!("    lust test [OPTIONS] [PATHS]...");
    println!();
    println!("ARGS:");
    println!("    <PATHS>...    Paths to test files or directories (default: tests)");
    println!();
    println!("OPTIONS:");
    println!("    -f, --filter <PATTERN>      Filter tests by regex or substring match");
    println!("    --format <FORMAT>           Report format: console (default), tap, json");
    println!("    -j, --jobs <N>              Number of concurrent workers (default: 1)");
    println!("    --timeout <MS>              Per-test execution timeout in ms (default: 10000)");
    println!("    --bail                      Stop test execution on first failure");
    println!("    --backend <BACKEND>         Execution backend: vm (default), auto");
    println!("    --virtual-clock             Enable virtual TestClock mode");
    println!("    -v, --verbose               Verbose test output and logs");
    println!("    -h, --help                  Print help information");
}

fn print_test_usage() {
    eprintln!("Usage: lust test [OPTIONS] [PATHS]...");
    eprintln!("Try 'lust test --help' for more information.");
}

/// Recursively discovers Lust test files matching RFC conventions:
/// - Any file path given directly
/// - In directories: files named `*_test.lust`, `test_*.lust`, `*test*.lust`,
///   or any `.lust` file inside a directory named `tests` or `testing`.
pub fn discover_test_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    let mut explicit_not_found = Vec::new();

    for path in paths {
        if !path.exists() {
            // If path was explicitly given by user (not default tests/ fallback)
            if path.as_os_str() != "tests" {
                explicit_not_found.push(path.display().to_string());
            }
            continue;
        }

        if path.is_file() {
            if path.extension().and_then(|s| s.to_str()) == Some("lust") {
                files.push(path.clone());
            }
        } else if path.is_dir() {
            collect_lust_files_in_dir(path, path, &mut files)?;
        }
    }

    if !explicit_not_found.is_empty() {
        return Err(format!(
            "path '{}' does not exist",
            explicit_not_found.join(", ")
        ));
    }

    files.sort();
    files.dedup();
    Ok(files)
}

fn collect_lust_files_in_dir(
    current_dir: &Path,
    root_dir: &Path,
    out: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let dir_name = current_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default();

    // Skip hidden directories, target, node_modules
    if dir_name.starts_with('.') || dir_name == "target" || dir_name == "node_modules" {
        return Ok(());
    }

    let entries = fs::read_dir(current_dir).map_err(|error| {
        format!("failed to read test directory '{}': {error}", current_dir.display())
    })?;

    let is_test_dir = current_dir
        .components()
        .any(|c| {
            let s = c.as_os_str().to_string_lossy();
            s == "tests" || s == "testing" || s.ends_with("_tests")
        });

    collect_lust_entries(entries, current_dir, root_dir, is_test_dir, out)
}

fn collect_lust_entries<I>(
    entries: I,
    current_dir: &Path,
    root_dir: &Path,
    is_test_dir: bool,
    out: &mut Vec<PathBuf>,
) -> Result<(), String>
where
    I: IntoIterator<Item = std::io::Result<fs::DirEntry>>,
{
    for entry in entries {
        let entry = entry.map_err(|error| {
            format!("failed to read an entry in test directory '{}': {error}", current_dir.display())
        })?;
        let entry_path = entry.path();
        if entry_path.is_dir() {
            collect_lust_files_in_dir(&entry_path, root_dir, out)?;
        } else if entry_path.is_file() {
            if entry_path.extension().and_then(|s| s.to_str()) == Some("lust") {
                let file_name = entry_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default();

                let is_explicit_test_name = file_name.ends_with("_test.lust")
                    || file_name.starts_with("test_")
                    || file_name.contains("test");

                if is_test_dir || is_explicit_test_name || current_dir == root_dir {
                    out.push(entry_path);
                }
            }
        }
    }
    Ok(())
}

/// Unwraps LuaValue struct wrappers if present.
fn unwrap_val(value: Value) -> Value {
    if let Value::Struct(instance) = &value {
        if instance.name.as_str() == "LuaValue" {
            let borrowed = instance.fields.borrow();
            if borrowed.len() == 1 {
                return borrowed[0].clone();
            }
        }
    }
    value
}

/// Checks equality between two Lust values with int/float coercion.
fn values_equal(a: &Value, b: &Value) -> bool {
    if a == b {
        return true;
    }
    match (a, b) {
        (Value::Int(i), Value::Float(f)) => (*i as f64) == *f,
        (Value::Float(f), Value::Int(i)) => *f == (*i as f64),
        _ => false,
    }
}

/// Registers standard test framework assertion functions into the Lust VM.
pub fn register_test_builtins(vm: &mut VM) {
    // 1. assert_eq
    let assert_eq_fn = Value::NativeFunction(native_fn(|args: &[Value]| {
        let a = args.first().cloned().map(unwrap_val).unwrap_or(Value::Nil);
        let b = args.get(1).cloned().map(unwrap_val).unwrap_or(Value::Nil);
        let msg = args.get(2).cloned().map(unwrap_val);
        if values_equal(&a, &b) {
            Ok(NativeCallResult::Return(Value::Bool(true)))
        } else {
            let err_msg = msg
                .map(|v| format!("{}", v))
                .unwrap_or_else(|| format!("assert_eq failed: {:?} != {:?}", a, b));
            Err(err_msg)
        }
    }));
    vm.register_native("assert_eq", assert_eq_fn.clone());
    vm.register_native("std.test.assert_eq", assert_eq_fn.clone());

    // 2. assert_ne
    let assert_ne_fn = Value::NativeFunction(native_fn(|args: &[Value]| {
        let a = args.first().cloned().map(unwrap_val).unwrap_or(Value::Nil);
        let b = args.get(1).cloned().map(unwrap_val).unwrap_or(Value::Nil);
        let msg = args.get(2).cloned().map(unwrap_val);
        if !values_equal(&a, &b) {
            Ok(NativeCallResult::Return(Value::Bool(true)))
        } else {
            let err_msg = msg
                .map(|v| format!("{}", v))
                .unwrap_or_else(|| format!("assert_ne failed: values are equal: {:?}", a));
            Err(err_msg)
        }
    }));
    vm.register_native("assert_ne", assert_ne_fn.clone());
    vm.register_native("std.test.assert_ne", assert_ne_fn.clone());

    // 3. assert_almost_eq
    let assert_almost_eq_fn = Value::NativeFunction(native_fn(|args: &[Value]| {
        let a = match args.first().cloned().map(unwrap_val) {
            Some(Value::Float(f)) => f,
            Some(Value::Int(i)) => i as f64,
            _ => 0.0,
        };
        let b = match args.get(1).cloned().map(unwrap_val) {
            Some(Value::Float(f)) => f,
            Some(Value::Int(i)) => i as f64,
            _ => 0.0,
        };
        let eps = match args.get(2).cloned().map(unwrap_val) {
            Some(Value::Float(f)) => f,
            Some(Value::Int(i)) => i as f64,
            _ => 1e-6,
        };
        let diff = (a - b).abs();
        if diff <= eps {
            Ok(NativeCallResult::Return(Value::Bool(true)))
        } else {
            Err(format!(
                "assert_almost_eq failed: |{} - {}| = {} > epsilon {}",
                a, b, diff, eps
            ))
        }
    }));
    vm.register_native("assert_almost_eq", assert_almost_eq_fn.clone());
    vm.register_native("std.test.assert_almost_eq", assert_almost_eq_fn.clone());

    // 4. assert_success
    let assert_success_fn = Value::NativeFunction(native_fn(|args: &[Value]| {
        let v = args.first().cloned().map(unwrap_val).unwrap_or(Value::Nil);
        match v {
            Value::Int(code) if code != 0 => {
                Err(format!("assert_success failed: non-zero exit code {}", code))
            }
            Value::Bool(false) => {
                Err("assert_success failed: expected success, got false".to_string())
            }
            _ => Ok(NativeCallResult::Return(Value::Bool(true))),
        }
    }));
    vm.register_native("assert_success", assert_success_fn.clone());
    vm.register_native("std.test.assert_success", assert_success_fn.clone());

    // 5. assert_fails
    let assert_fails_fn = Value::NativeFunction(native_fn(|args: &[Value]| {
        let v = args.first().cloned().map(unwrap_val).unwrap_or(Value::Nil);
        match v {
            Value::Int(0) => {
                Err("assert_fails failed: expected failure, got exit code 0".to_string())
            }
            Value::Nil => Err("assert_fails failed: expected failure, got nil".to_string()),
            _ => Ok(NativeCallResult::Return(Value::Bool(true))),
        }
    }));
    vm.register_native("assert_fails", assert_fails_fn.clone());
    vm.register_native("std.test.assert_fails", assert_fails_fn.clone());

    // 6. assert_interrupted
    let assert_interrupted_fn = Value::NativeFunction(native_fn(|args: &[Value]| {
        let v = args.first().cloned().map(unwrap_val).unwrap_or(Value::Nil);
        match v {
            Value::Nil | Value::Bool(false) => {
                Err("assert_interrupted failed: no interrupt received".to_string())
            }
            _ => Ok(NativeCallResult::Return(Value::Bool(true))),
        }
    }));
    vm.register_native("assert_interrupted", assert_interrupted_fn.clone());
    vm.register_native("std.test.assert_interrupted", assert_interrupted_fn.clone());

    // 7. assert_dies
    let assert_dies_fn = Value::NativeFunction(native_fn(|args: &[Value]| {
        let v = args.first().cloned().map(unwrap_val).unwrap_or(Value::Nil);
        match v {
            Value::Nil | Value::Bool(false) => {
                Err("assert_dies failed: expected panic/defect".to_string())
            }
            _ => Ok(NativeCallResult::Return(Value::Bool(true))),
        }
    }));
    vm.register_native("assert_dies", assert_dies_fn.clone());
    vm.register_native("std.test.assert_dies", assert_dies_fn.clone());

    // 8. assert_snapshot
    let assert_snapshot_fn = Value::NativeFunction(native_fn(|args: &[Value]| {
        let test_file = CURRENT_TEST_FILE.with(|f| f.borrow().clone()).unwrap_or_else(|| PathBuf::from("test.lust"));
        let update = CURRENT_UPDATE_SNAPSHOTS.with(|u| u.get());
        let name = match args.first().cloned().map(unwrap_val) {
            Some(Value::String(s)) => s.as_str().to_string(),
            _ => "unnamed_snapshot".to_string(),
        };
        let actual = match args.get(1).cloned().map(unwrap_val) {
            Some(v) => lust::test_framework::snap::canonical_value_repr(&v),
            None => "".to_string(),
        };
        match lust::test_framework::snap::check_or_update_snapshot(&test_file, &name, &actual, update) {
            Ok(()) => Ok(NativeCallResult::Return(Value::Bool(true))),
            Err(e) => Err(e),
        }
    }));
    vm.register_native("assert_snapshot", assert_snapshot_fn.clone());
    vm.register_native("std.test.assert_snapshot", assert_snapshot_fn.clone());

    // 9. bench
    let bench_fn = Value::NativeFunction(native_fn(|args: &[Value]| {
        let name = match args.first().cloned().map(unwrap_val) {
            Some(Value::String(s)) => s.as_str().to_string(),
            _ => "unnamed_bench".to_string(),
        };
        let callback = args.get(1).cloned().map(unwrap_val);
        let options = lust::test_framework::bench::BenchOptions::default();
        let result = lust::test_framework::bench::bench(&name, || {
            if let Some(Value::NativeFunction(nf)) = &callback {
                let _ = nf(&[]);
            }
        }, options);
        RECORDED_BENCHMARKS.with(|b| b.borrow_mut().push(result.clone()));
        Ok(NativeCallResult::Return(Value::map(result.to_lust_map())))
    }));
    vm.register_native("bench", bench_fn.clone());
    vm.register_native("std.test.bench", bench_fn.clone());

    // 10. bench_with_options
    let bench_with_options_fn = Value::NativeFunction(native_fn(|args: &[Value]| {
        let name = match args.first().cloned().map(unwrap_val) {
            Some(Value::String(s)) => s.as_str().to_string(),
            _ => "unnamed_bench".to_string(),
        };
        let mut opts = lust::test_framework::bench::BenchOptions::default();
        if let Some(Value::Map(m)) = args.get(1).cloned().map(unwrap_val) {
            let map = m.borrow();
            if let Some(Value::Int(w)) = map.get(&ValueKey::from(Value::string("warmup_iters"))) {
                opts.warmup_iters = *w as usize;
            }
            if let Some(Value::Int(s)) = map.get(&ValueKey::from(Value::string("sample_count"))) {
                opts.sample_count = *s as usize;
            }
            if let Some(Value::Int(i)) = map.get(&ValueKey::from(Value::string("iters_per_sample"))) {
                opts.iters_per_sample = *i as usize;
            }
        }
        let callback = args.get(2).cloned().map(unwrap_val);
        let result = lust::test_framework::bench::bench(&name, || {
            if let Some(Value::NativeFunction(nf)) = &callback {
                let _ = nf(&[]);
            }
        }, opts);
        RECORDED_BENCHMARKS.with(|b| b.borrow_mut().push(result.clone()));
        Ok(NativeCallResult::Return(Value::map(result.to_lust_map())))
    }));
    vm.register_native("bench_with_options", bench_with_options_fn.clone());
    vm.register_native("std.test.bench_with_options", bench_with_options_fn.clone());

    // 11. global `test` table
    let mut test_table = LustMap::default();
    test_table.insert(ValueKey::from(Value::string("assert_eq")), assert_eq_fn);
    test_table.insert(ValueKey::from(Value::string("assert_ne")), assert_ne_fn);
    test_table.insert(ValueKey::from(Value::string("assert_almost_eq")), assert_almost_eq_fn);
    test_table.insert(ValueKey::from(Value::string("assert_success")), assert_success_fn);
    test_table.insert(ValueKey::from(Value::string("assert_fails")), assert_fails_fn);
    test_table.insert(ValueKey::from(Value::string("assert_interrupted")), assert_interrupted_fn);
    test_table.insert(ValueKey::from(Value::string("assert_dies")), assert_dies_fn);
    test_table.insert(ValueKey::from(Value::string("assert_snapshot")), assert_snapshot_fn);
    test_table.insert(ValueKey::from(Value::string("bench")), bench_fn);
    test_table.insert(ValueKey::from(Value::string("bench_with_options")), bench_with_options_fn);

    // TestClock bindings
    test_table.insert(ValueKey::from(Value::string("clock_new")), lust::stdlib::rx_test_bridge::create_test_clock_new_fn());
    test_table.insert(ValueKey::from(Value::string("clock_now_ms")), lust::stdlib::rx_test_bridge::create_test_clock_now_ms_fn());
    test_table.insert(ValueKey::from(Value::string("clock_schedule")), lust::stdlib::rx_test_bridge::create_test_clock_schedule_fn());
    test_table.insert(ValueKey::from(Value::string("clock_adjust")), lust::stdlib::rx_test_bridge::create_test_clock_adjust_fn());
    test_table.insert(ValueKey::from(Value::string("clock_pending_count")), lust::stdlib::rx_test_bridge::create_test_clock_pending_count_fn());
    test_table.insert(ValueKey::from(Value::string("schedule_exponential")), lust::stdlib::rx_test_bridge::create_test_schedule_exponential_fn());
    test_table.insert(ValueKey::from(Value::string("schedule_step")), lust::stdlib::rx_test_bridge::create_test_schedule_step_fn());

    // MockService & StackContextProjection bindings
    test_table.insert(ValueKey::from(Value::string("mock_new")), lust::stdlib::rx_test_bridge::create_test_mock_new_fn());
    test_table.insert(ValueKey::from(Value::string("mock_set_response")), lust::stdlib::rx_test_bridge::create_test_mock_set_response_fn());
    test_table.insert(ValueKey::from(Value::string("mock_record_call")), lust::stdlib::rx_test_bridge::create_test_mock_record_call_fn());
    test_table.insert(ValueKey::from(Value::string("mock_call_count")), lust::stdlib::rx_test_bridge::create_test_mock_call_count_fn());
    test_table.insert(ValueKey::from(Value::string("mock_last_call")), lust::stdlib::rx_test_bridge::create_test_mock_last_call_fn());
    test_table.insert(ValueKey::from(Value::string("projection_new")), lust::stdlib::rx_test_bridge::create_test_projection_new_fn());
    test_table.insert(ValueKey::from(Value::string("projection_get")), lust::stdlib::rx_test_bridge::create_test_projection_get_fn());

    // Scope & FiberContext supervision bindings
    test_table.insert(ValueKey::from(Value::string("resource_new")), lust::stdlib::rx_test_bridge::create_test_resource_new_fn());
    test_table.insert(ValueKey::from(Value::string("resource_is_active")), lust::stdlib::rx_test_bridge::create_test_resource_is_active_fn());
    test_table.insert(ValueKey::from(Value::string("scope_new")), lust::stdlib::rx_test_bridge::create_test_scope_new_fn());
    test_table.insert(ValueKey::from(Value::string("scope_is_open")), lust::stdlib::rx_test_bridge::create_test_scope_is_open_fn());
    test_table.insert(ValueKey::from(Value::string("scope_teardown_state")), lust::stdlib::rx_test_bridge::create_test_scope_teardown_state_fn());
    test_table.insert(ValueKey::from(Value::string("scope_admission_state")), lust::stdlib::rx_test_bridge::create_test_scope_admission_state_fn());
    test_table.insert(ValueKey::from(Value::string("scope_register_finalizer")), lust::stdlib::rx_test_bridge::create_test_scope_register_finalizer_fn());
    test_table.insert(ValueKey::from(Value::string("scope_register_child")), lust::stdlib::rx_test_bridge::create_test_scope_register_child_fn());
    test_table.insert(ValueKey::from(Value::string("scope_begin_teardown")), lust::stdlib::rx_test_bridge::create_test_scope_begin_teardown_fn());
    test_table.insert(ValueKey::from(Value::string("scope_mark_joining")), lust::stdlib::rx_test_bridge::create_test_scope_mark_joining_fn());
    test_table.insert(ValueKey::from(Value::string("scope_current_releases")), lust::stdlib::rx_test_bridge::create_test_scope_current_releases_fn());
    test_table.insert(ValueKey::from(Value::string("scope_finalize")), lust::stdlib::rx_test_bridge::create_test_scope_finalize_fn());
    test_table.insert(ValueKey::from(Value::string("fiber_new")), lust::stdlib::rx_test_bridge::create_test_fiber_new_fn());
    test_table.insert(ValueKey::from(Value::string("fiber_set_state")), lust::stdlib::rx_test_bridge::create_test_fiber_set_state_fn());
    test_table.insert(ValueKey::from(Value::string("fiber_enter_mask")), lust::stdlib::rx_test_bridge::create_test_fiber_enter_mask_fn());
    test_table.insert(ValueKey::from(Value::string("fiber_exit_mask")), lust::stdlib::rx_test_bridge::create_test_fiber_exit_mask_fn());
    test_table.insert(ValueKey::from(Value::string("fiber_is_masked")), lust::stdlib::rx_test_bridge::create_test_fiber_is_masked_fn());
    test_table.insert(ValueKey::from(Value::string("fiber_request_interrupt")), lust::stdlib::rx_test_bridge::create_test_fiber_request_interrupt_fn());
    test_table.insert(ValueKey::from(Value::string("fiber_state")), lust::stdlib::rx_test_bridge::create_test_fiber_state_fn());

    vm.register_native("test", Value::map(test_table));
    lust::stdlib::rx_test_bridge::register_rx_test_builtins(vm);
}

/// Executes a single test's statements within a freshly instantiated VM and compiled program.
fn execute_test_body(
    program: &Program,
    entry_module_path: &str,
    test_stmts: &[Stmt],
    span: Span,
    non_test_items: &[Item],
    config: &LustConfig,
) -> Result<(), AssertError> {
    let mut cloned_program = program.clone();

    // Replace entry module items with non-test items + test script body
    let mut test_items = non_test_items.to_vec();
    test_items.push(Item::new(ItemKind::Script(test_stmts.to_vec()), span));

    if let Some(entry_mod) = cloned_program
        .modules
        .iter_mut()
        .find(|m| m.path == entry_module_path)
    {
        // Inject extern declarations for bench and assert_snapshot so typechecker/compiler allow them
        let any_ty = || lust::ast::Type::new(lust::ast::TypeKind::Unknown, Span::new(0, 0, 0, 0));
        let str_ty = || lust::ast::Type::new(lust::ast::TypeKind::String, Span::new(0, 0, 0, 0));
        let extern_items = vec![
            lust::ast::ExternItem::Function {
                name: "assert_snapshot".to_string(),
                params: vec![
                    lust::ast::ExternParam { name: Some("name".to_string()), ty: str_ty() },
                    lust::ast::ExternParam { name: Some("val".to_string()), ty: any_ty() },
                ],
                return_type: Some(any_ty()),
                doc: None,
            },
            lust::ast::ExternItem::Function {
                name: "bench".to_string(),
                params: vec![
                    lust::ast::ExternParam { name: Some("name".to_string()), ty: str_ty() },
                    lust::ast::ExternParam { name: Some("f".to_string()), ty: any_ty() },
                ],
                return_type: Some(any_ty()),
                doc: None,
            },
            lust::ast::ExternItem::Function {
                name: "bench_with_options".to_string(),
                params: vec![
                    lust::ast::ExternParam { name: Some("name".to_string()), ty: str_ty() },
                    lust::ast::ExternParam { name: Some("opts".to_string()), ty: any_ty() },
                    lust::ast::ExternParam { name: Some("f".to_string()), ty: any_ty() },
                ],
                return_type: Some(any_ty()),
                doc: None,
            },
        ];
        test_items.insert(0, Item::new(ItemKind::Extern { abi: "lust".to_string(), items: extern_items }, span));
        entry_mod.items = test_items;
        // Inject std.test function aliases so test assertions resolve seamlessly
        let aliases = [
            ("assert_eq", "std.test.assert_eq"),
            ("assert_ne", "std.test.assert_ne"),
            ("assert_almost_eq", "std.test.assert_almost_eq"),
            ("assert_success", "std.test.assert_success"),
            ("assert_fails", "std.test.assert_fails"),
            ("assert_interrupted", "std.test.assert_interrupted"),
            ("assert_dies", "std.test.assert_dies"),
            ("assert_snapshot", "std.test.assert_snapshot"),
            ("bench", "std.test.bench"),
            ("bench_with_options", "std.test.bench_with_options"),
        ];
        for (name, target) in aliases {
            entry_mod
                .imports
                .function_aliases
                .entry(name.to_string())
                .or_insert_with(|| target.to_string());
        }
        let mod_aliases = ["rx", "test", "table", "array", "string", "math", "task"];
        for mod_name in mod_aliases {
            entry_mod
                .imports
                .module_aliases
                .entry(mod_name.to_string())
                .or_insert_with(|| mod_name.to_string());
        }
    } else {
        return Err(AssertError {
            message: format!("Entry module '{}' not found in loaded program", entry_module_path),
            location: None,
            diff: None,
            fiber_id: None,
        });
    }

    let mut vm = compile_program_with_config(cloned_program, config).map_err(|e| {
        AssertError {
            message: format!("Compilation error: {}", e),
            location: None,
            diff: None,
            fiber_id: None,
        }
    })?;

    register_test_builtins(&mut vm);

    vm.call("__script", vec![]).map_err(|e| AssertError {
        message: format!("{}", e),
        location: None,
        diff: None,
        fiber_id: None,
    })?;

    Ok(())
}

fn filter_matches(pattern: Option<&str>, suite_name: &str, test_name: &str) -> bool {
    let pat = match pattern {
        Some(p) if !p.is_empty() => p,
        _ => return true,
    };

    let full_name = format!("{}::{}", suite_name, test_name);
    if full_name.contains(pat) || test_name.contains(pat) {
        return true;
    }

    let re = match lust::regex::BoundedRegex::new(pat) {
        Ok(re) => re,
        // Invalid expressions are rejected in parse_test_options before discovery.
        Err(_) => return false,
    };
    re.is_match(&full_name).unwrap_or(false) || re.is_match(test_name).unwrap_or(false)
}

/// Executes a `SuiteDecl` including hooks and nested suites.
fn run_suite_declaration(
    suite: &SuiteDecl,
    program: &Program,
    entry_module_path: &str,
    non_test_items: &[Item],
    parent_before_each: &[Stmt],
    parent_after_each: &[Stmt],
    runner: &TestRunner,
    config: &LustConfig,
    filter: Option<&str>,
) -> TestSuiteResult {
    let start = Instant::now();
    let mut test_results = Vec::new();

    // Combined before_each: parent before_each + this suite before_each
    let mut current_before_each = parent_before_each.to_vec();
    if let Some(bh) = &suite.hooks.before_each {
        current_before_each.extend(bh.clone());
    }

    // Combined after_each: this suite after_each + parent after_each
    let mut current_after_each = Vec::new();
    if let Some(ah) = &suite.hooks.after_each {
        current_after_each.extend(ah.clone());
    }
    current_after_each.extend(parent_after_each.to_vec());

    // Check if entire suite is skipped
    if let TestModifier::Skip(reason) = &suite.modifier {
        let skip_msg = reason.clone().unwrap_or_else(|| "Suite skipped".to_string());
        for test in &suite.tests {
            if filter_matches(filter, &suite.name, &test.name) {
                test_results.push(TestResult {
                    name: test.name.clone(),
                    outcome: TestOutcome::Skipped(skip_msg.clone()),
                    duration: Duration::ZERO,
                });
            }
        }
        return TestSuiteResult {
            name: suite.name.clone(),
            tests: test_results,
            nested_suites: Vec::new(),
            duration: start.elapsed(),
        };
    }

    // Run before_all hook if present
    if let Some(before_all_stmts) = &suite.hooks.before_all {
        let before_all_res = runner.run_test_fn(&format!("{} > before_all", suite.name), |_fib, _scope| {
            execute_test_body(
                program,
                entry_module_path,
                before_all_stmts,
                suite.span,
                non_test_items,
                config,
            )
        });
        if let TestOutcome::Failed(err) = before_all_res.outcome {
            test_results.push(TestResult {
                name: "before_all hook".to_string(),
                outcome: TestOutcome::Failed(err),
                duration: before_all_res.duration,
            });
            return TestSuiteResult {
                name: suite.name.clone(),
                tests: test_results,
                nested_suites: Vec::new(),
                duration: start.elapsed(),
            };
        }
    }

    for test in &suite.tests {
        if !filter_matches(filter, &suite.name, &test.name) {
            continue;
        }

        if let TestModifier::Skip(reason) = &test.modifier {
            test_results.push(TestResult {
                name: test.name.clone(),
                outcome: TestOutcome::Skipped(reason.clone().unwrap_or_default()),
                duration: Duration::ZERO,
            });
            continue;
        }

        let mut stmts = current_before_each.clone();
        stmts.extend(test.body.clone());
        stmts.extend(current_after_each.clone());

        let res = runner.run_test_fn(&test.name, |_fiber, _scope| {
            execute_test_body(
                program,
                entry_module_path,
                &stmts,
                test.span,
                non_test_items,
                config,
            )
        });

        let failed = matches!(res.outcome, TestOutcome::Failed(_));
        test_results.push(res);

        if failed && runner.bail {
            break;
        }
    }

    // Run nested suites
    let mut nested_results = Vec::new();
    for nested in &suite.nested_suites {
        let res = run_suite_declaration(
            nested,
            program,
            entry_module_path,
            non_test_items,
            &current_before_each,
            &current_after_each,
            runner,
            config,
            filter,
        );
        let has_failed = res.tests.iter().any(|t| matches!(t.outcome, TestOutcome::Failed(_)));
        nested_results.push(res);
        if has_failed && runner.bail {
            break;
        }
    }

    // Run after_all hook if present
    if let Some(after_all_stmts) = &suite.hooks.after_all {
        let _ = runner.run_test_fn(&format!("{} > after_all", suite.name), |_fib, _scope| {
            execute_test_body(
                program,
                entry_module_path,
                after_all_stmts,
                suite.span,
                non_test_items,
                config,
            )
        });
    }

    TestSuiteResult {
        name: suite.name.clone(),
        tests: test_results,
        nested_suites: nested_results,
        duration: start.elapsed(),
    }
}

/// Executes all tests in a discovered test file.
pub fn run_test_file(
    file_path: &Path,
    runner: &TestRunner,
    config: &LustConfig,
    filter: Option<&str>,
) -> Vec<TestSuiteResult> {
    let mut results = Vec::new();
    let display_name = file_path.display().to_string();

    let entry_file_str = match file_path.to_str() {
        Some(s) => s,
        None => {
            results.push(TestSuiteResult {
                name: display_name,
                tests: vec![TestResult {
                    name: "path validation".to_string(),
                    outcome: TestOutcome::Failed(AssertError {
                        message: "File path contains invalid UTF-8".to_string(),
                        location: None,
                        diff: None,
                        fiber_id: None,
                    }),
                    duration: Duration::ZERO,
                }],
                nested_suites: Vec::new(),
                duration: Duration::ZERO,
            });
            return results;
        }
    };

    let parent_dir = file_path.parent().unwrap_or_else(|| Path::new("."));
    let mut loader = ModuleLoader::new(parent_dir.to_path_buf());
    loader.set_config(config.clone());

    let program = match loader.load_program_from_entry(entry_file_str) {
        Ok(p) => p,
        Err(err) => {
            results.push(TestSuiteResult {
                name: display_name,
                tests: vec![TestResult {
                    name: "load program".to_string(),
                    outcome: TestOutcome::Failed(AssertError {
                        message: format!("Failed to load test file: {}", err),
                        location: None,
                        diff: None,
                        fiber_id: None,
                    }),
                    duration: Duration::ZERO,
                }],
                nested_suites: Vec::new(),
                duration: Duration::ZERO,
            });
            return results;
        }
    };

    let entry_module_path = program.entry_module.clone();
    let entry_mod = match program.modules.iter().find(|m| m.path == entry_module_path) {
        Some(m) => m,
        None => return results,
    };

    // Partition items into tests, suites, and non-test definitions
    let mut top_tests: Vec<TestDecl> = Vec::new();
    let mut suites: Vec<SuiteDecl> = Vec::new();
    let mut non_test_items: Vec<Item> = Vec::new();
    let mut top_script_stmts: Vec<Stmt> = Vec::new();

    for item in &entry_mod.items {
        match &item.kind {
            ItemKind::Test(td) => top_tests.push(td.clone()),
            ItemKind::Suite(sd) => suites.push(sd.clone()),
            ItemKind::Script(stmts) => {
                top_script_stmts.extend(stmts.clone());
            }
            _ => non_test_items.push(item.clone()),
        }
    }

    // 1. If file has top-level tests: execute them as a suite named after the file
    if !top_tests.is_empty() {
        let start = Instant::now();
        let mut test_results = Vec::new();

        for test in &top_tests {
            if !filter_matches(filter, &display_name, &test.name) {
                continue;
            }

            if let TestModifier::Skip(reason) = &test.modifier {
                test_results.push(TestResult {
                    name: test.name.clone(),
                    outcome: TestOutcome::Skipped(reason.clone().unwrap_or_default()),
                    duration: Duration::ZERO,
                });
                continue;
            }

            let res = runner.run_test_fn(&test.name, |_fiber, _scope| {
                execute_test_body(
                    &program,
                    &entry_module_path,
                    &test.body,
                    test.span,
                    &non_test_items,
                    config,
                )
            });

            let failed = matches!(res.outcome, TestOutcome::Failed(_));
            test_results.push(res);

            if failed && runner.bail {
                break;
            }
        }

        if !test_results.is_empty() {
            results.push(TestSuiteResult {
                name: display_name.clone(),
                tests: test_results,
                nested_suites: Vec::new(),
                duration: start.elapsed(),
            });
        }
    }

    // 2. Execute any declared suites
    for suite in &suites {
        if runner.bail && results.iter().any(|s| s.tests.iter().any(|t| matches!(t.outcome, TestOutcome::Failed(_)))) {
            break;
        }

        let suite_res = run_suite_declaration(
            suite,
            &program,
            &entry_module_path,
            &non_test_items,
            &[],
            &[],
            runner,
            config,
            filter,
        );
        results.push(suite_res);
    }

    // 3. Fallback: if file has no test or suite items, run entire script as 1 test
    if top_tests.is_empty() && suites.is_empty() {
        if filter_matches(filter, &display_name, &display_name) {
            let start = Instant::now();
            let res = runner.run_test_fn(&display_name, |_fiber, _scope| {
                execute_test_body(
                    &program,
                    &entry_module_path,
                    &top_script_stmts,
                    Span::new(0, 0, 0, 0),
                    &non_test_items,
                    config,
                )
            });

            results.push(TestSuiteResult {
                name: display_name,
                tests: vec![res],
                nested_suites: Vec::new(),
                duration: start.elapsed(),
            });
        }
    }

    results
}

/// Executes the full test suite given `TestRunOptions` and returns results.
pub fn run_test_suite(
    options: &TestRunOptions,
) -> Result<(Vec<TestSuiteResult>, TestSummary), String> {
    CURRENT_UPDATE_SNAPSHOTS.with(|u| u.set(options.update_snapshots));
    RECORDED_BENCHMARKS.with(|b| b.borrow_mut().clear());

    let files = discover_test_files(&options.paths)?;
    let runner = TestRunner::new().with_bail(options.bail);
    let mut config = LustConfig::default();
    config.enable_module("table");
    let jit_enabled = options.backend == "jit" || options.backend == "cranelift";
    config.set_jit_enabled(jit_enabled);

    let mut all_suites = Vec::new();

    for file_path in &files {
        CURRENT_TEST_FILE.with(|f| *f.borrow_mut() = Some(file_path.clone()));
        let file_suites = run_test_file(
            file_path,
            &runner,
            &config,
            options.filter.as_deref(),
        );

        let failed = file_suites.iter().any(|s| {
            s.tests.iter().any(|t| matches!(t.outcome, TestOutcome::Failed(_)))
                || s.nested_suites
                    .iter()
                    .any(|ns| ns.tests.iter().any(|t| matches!(t.outcome, TestOutcome::Failed(_))))
        });

        all_suites.extend(file_suites);

        if failed && options.bail {
            break;
        }
    }

    let mut summary = TestRunner::summarize(&all_suites);
    summary.benchmarks = RECORDED_BENCHMARKS.with(|b| b.borrow().clone());
    if options.update_snapshots {
        for suite in &all_suites {
            let path = PathBuf::from(&suite.name);
            let entries: Vec<(String, String)> = suite
                .tests
                .iter()
                .map(|t| (t.name.clone(), format!("{:?}", t.outcome)))
                .collect();
            let snaps = lust::test_framework::snap::serialize(&entries);
            let _ = lust::test_framework::snap::write_atomic(&lust::test_framework::snap::snapshot_path(&path), &snaps);
        }
    }
    Ok((all_suites, summary))
}

/// Entry point dispatched from `src/main.rs` for `lust test`.
pub fn test_command(args: &[String]) {
    let options = match parse_test_options(args) {
        Ok(opts) => opts,
        Err(err) => {
            eprintln!("{}", err);
            print_test_usage();
            process::exit(1);
        }
    };

    match run_test_suite(&options) {
        Ok((suites, summary)) => {
            if suites.is_empty() && summary.total == 0 {
                let targets = options
                    .paths
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                if options.reporter == ReporterKind::Json {
                    println!("{{\"total\": 0, \"passed\": 0, \"failed\": 0, \"skipped\": 0, \"duration_ms\": 0, \"suites\": []}}");
                } else if options.reporter == ReporterKind::Tap {
                    println!("TAP version 13\n1..0 # No tests found in: {}", targets);
                } else {
                    println!("\n--- [Lust Test Runner] ---");
                    println!("No test files found in: {}", targets);
                    println!("--------------------------------------------------");
                    println!("Tests: 0 passed, 0 failed, 0 skipped, 0 total");
                    println!("--------------------------------------------------");
                }
                process::exit(0);
            }

            let report = if options.reporter == ReporterKind::Junit { lust::test_framework::junit::format(&suites,&summary) } else { Reporter::format(options.reporter, &suites, &summary) };
            if let Some(path)=&options.junit_path { fs::write(path,&report).unwrap_or_else(|e| {eprintln!("Error writing JUnit report: {e}");process::exit(1)}); } else { print!("{}", report); }

            if summary.is_success() {
                process::exit(0);
            } else {
                process::exit(1);
            }
        }
        Err(err) => {
            eprintln!("Error: {}", err);
            process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_test_options_defaults() {
        let args = vec![];
        let opts = parse_test_options(&args).unwrap();
        assert_eq!(opts.paths, vec![PathBuf::from("tests")]);
        assert_eq!(opts.reporter, ReporterKind::Pretty);
        assert!(!opts.bail);
        assert_eq!(opts.filter, None);
    }

    #[test]
    fn test_parse_test_options_flags() {
        let args = vec![
            "-f".to_string(),
            "math".to_string(),
            "--format".to_string(),
            "json".to_string(),
            "--bail".to_string(),
            "custom/dir".to_string(),
        ];
        let opts = parse_test_options(&args).unwrap();
        assert_eq!(opts.paths, vec![PathBuf::from("custom/dir")]);
        assert_eq!(opts.reporter, ReporterKind::Json);
        assert!(opts.bail);
        assert_eq!(opts.filter, Some("math".to_string()));
    }

    #[test]
    fn test_parse_test_options_unknown_flag() {
        let args = vec!["--unknown-flag".to_string()];
        assert!(parse_test_options(&args).is_err());
    }

    #[test]
    fn test_discover_test_files_explicit_nonexistent() {
        let non_existent = vec![PathBuf::from("does_not_exist_xyz123")];
        let res = discover_test_files(&non_existent);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("does_not_exist_xyz123"));
    }

    #[test]
    fn test_discovery_reports_directory_read_errors() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let not_a_directory = dir.path().join("ordinary.txt");
        std::fs::write(&not_a_directory, "not a directory").expect("write fixture");

        let error = collect_lust_files_in_dir(&not_a_directory, dir.path(), &mut Vec::new())
            .expect_err("read_dir errors must be returned");
        assert!(error.contains("failed to read test directory"));
        assert!(error.contains("ordinary.txt"));
    }

    #[test]
    fn test_discovery_reports_directory_entry_errors() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let error = collect_lust_entries(
            [Err(std::io::Error::other("injected entry failure"))],
            dir.path(),
            dir.path(),
            true,
            &mut Vec::new(),
        )
        .expect_err("entry iteration errors must be returned");

        assert!(error.contains("failed to read an entry"));
        assert!(error.contains("injected entry failure"));
    }
}

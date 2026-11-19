//! Wave 4 Track 4F (Darwin POSIX Baseline) & Cranelift `MirUnaryOp::Not` Parity Suite.
//!
//! 11 independent tests (`test_4f_1`..`test_4f_11`) covering:
//! 1. Permitted finite execution & private tempdir lifecycle cleanup
//! 2. Monotonic wall-clock watchdog `killpg(SIGKILL)` termination
//! 3. Same-PGID descendant termination on timeout (documenting `setsid` boundary)
//! 4. Environment scrubbing & isolated CWD
//! 5. Bounded stdout/stderr byte capping & truncation flags
//! 6. Fail-closed policy validation & non-Linux rejection of Linux-only isolation
//! 7. Darwin `setrlimit` enforcement (`RLIMIT_CPU`, `RLIMIT_NOFILE`) & truthful `RlimitReport`
//! 8. Denied vs permitted Lust script capabilities under `SandboxPolicy`
//! 9. Cranelift MIR `MirUnaryOp::Not` differential parity against Bytecode
//! 10. Pre-exec FD sanitization closing inherited non-`CLOEXEC` descriptors before `exec`
//! 11. Bounded output drain against `setsid()`-escaped descendants holding stdout open

#![cfg(unix)]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use lust::bytecode::compiler::Compiler;
use lust::config::LustConfig;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::ir::MirModule;
use lust::mir::lowering::AstLowering;
use lust::parser::Parser;
use lust::sandbox::{IsolationRequirements, SandboxError, SandboxPolicy, run_sandboxed};
use lust::typechecker::TypeChecker;
use lust::vm::{VM, VmBackend};
use lust::{EmbeddedProgram, Value};
use std::fs::{self, File};
use std::io::Write;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const F_GETFD: i32 = 1;
const FD_CLOEXEC: i32 = 1;
const EBADF: i32 = 9;
const ESRCH: i32 = 3;
const SIGKILL: i32 = 9;

unsafe extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
    fn pipe(fds: *mut i32) -> i32;
    fn fcntl(fd: i32, cmd: i32, ...) -> i32;
    fn close(fd: i32) -> i32;
    fn setsid() -> i32;
}

struct FdGuard(i32);

impl Drop for FdGuard {
    fn drop(&mut self) {
        if self.0 >= 0 {
            unsafe {
                let _ = close(self.0);
            }
            self.0 = -1;
        }
    }
}

struct DetachedPidCleanup(PathBuf);

impl Drop for DetachedPidCleanup {
    fn drop(&mut self) {
        if let Ok(text) = fs::read_to_string(&self.0)
            && let Ok(pid) = text.trim().parse::<i32>()
            && pid > 0
        {
            unsafe {
                let _ = kill(pid, SIGKILL);
            }
        }
    }
}

const WORKER_ENV_KEY: &str = "LUST_SANDBOX_WORKER_MODE";

fn worker_policy(mode: &str) -> SandboxPolicy {
    let exe = std::env::current_exe().expect("current test executable path");
    let mut policy = SandboxPolicy::new(exe);
    policy.args = vec![
        "--exact".to_string(),
        "__sandbox_worker".to_string(),
        "--ignored".to_string(),
        "--nocapture".to_string(),
    ];
    policy.allowed_env.insert(WORKER_ENV_KEY.to_string(), mode.to_string());
    policy
}

#[test]
#[ignore]
fn __sandbox_worker() {
    let Ok(mode) = std::env::var(WORKER_ENV_KEY) else {
        return;
    };
    match mode.as_str() {
        "permitted_control" => {
            let cwd = std::env::current_dir().expect("cwd");
            println!("CONTROL_OK");
            println!("CWD={}", cwd.display());
        }
        "busy_loop" => {
            loop {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        "spawn_descendant_and_sleep" => {
            let marker = std::env::var("LUST_DESCENDANT_PID_FILE").expect("marker path");
            let child = Command::new("/bin/sleep")
                .arg("60")
                .spawn()
                .expect("spawn /bin/sleep grandchild");
            let pid = child.id();
            {
                let mut f = File::create(&marker).expect("create pid marker");
                writeln!(f, "{pid}").expect("write pid");
                let _ = f.sync_all();
            }
            loop {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        "dump_env_cwd" => {
            let cwd = std::env::current_dir().expect("cwd");
            println!("CWD={}", cwd.display());
            for (k, v) in std::env::vars() {
                println!("ENV:{k}={v}");
            }
        }
        "flood_stdio" => {
            let out_chunk = vec![b'A'; 256 * 1024];
            let err_chunk = vec![b'E'; 256 * 1024];
            let mut stdout = std::io::stdout().lock();
            let mut stderr = std::io::stderr().lock();
            stdout.write_all(&out_chunk).expect("write stdout");
            stdout.flush().expect("flush stdout");
            stderr.write_all(&err_chunk).expect("write stderr");
            stderr.flush().expect("flush stderr");
        }
        "cpu_burn" => {
            let mut acc: u64 = 1;
            loop {
                acc = acc.wrapping_mul(6364136223846793005).wrapping_add(1);
                std::hint::black_box(acc);
            }
        }
        "open_fd_exhaust" => {
            let mut files = Vec::new();
            let mut last_errno = 0i32;
            for i in 0..256 {
                let path = format!("fd_probe_{i}.tmp");
                match File::create(&path) {
                    Ok(f) => files.push(f),
                    Err(e) => {
                        last_errno = e.raw_os_error().unwrap_or(-1);
                        break;
                    }
                }
            }
            println!("OPENED_COUNT={}", files.len());
            println!("ERRNO={last_errno}");
        }
        "lust_permitted" => {
            let effect_path = std::env::var("LUST_EFFECT_PATH").expect("effect path");
            let src = r#"
                extern
                    function record_effect(string): int
                end
                function run(): int
                    return record_effect("PERMITTED_EFFECT_RECORDED")
                end
            "#;
            let mut prog = EmbeddedProgram::builder()
                .with_config(LustConfig::pure())
                .module("main", src)
                .entry_module("main")
                .compile()
                .expect("compile permitted script");
            let effect_path_clone = effect_path.clone();
            prog.register_typed_native("record_effect", move |msg: String| -> Result<i64, String> {
                fs::write(&effect_path_clone, msg.as_bytes()).map_err(|e| e.to_string())?;
                Ok(1)
            })
            .expect("register record_effect");
            let res: i64 = prog.call_typed("main.run", ()).expect("call main.run");
            println!("PERMITTED_RESULT={res}");
        }
        "lust_denied" => {
            let canary_path = std::env::var("LUST_CANARY_PATH").expect("canary path");
            let canary_escaped = canary_path.replace('\\', "\\\\").replace('"', "\\\"");
            let attempts = [
                (
                    "pure_io_open",
                    LustConfig::pure(),
                    format!("function f(): int local h = io.open(\"{canary_escaped}\", \"r\") return 0 end"),
                ),
                (
                    "pure_io_read_file",
                    LustConfig::pure(),
                    format!("function f(): string return io.read_file(\"{canary_escaped}\"):unwrap() end"),
                ),
                (
                    "pure_io_write_file",
                    LustConfig::pure(),
                    format!("function f(): () io.write_file(\"{canary_escaped}\", \"PWNED\") end"),
                ),
                (
                    "pure_os_getenv",
                    LustConfig::pure(),
                    "function f(): string return os.getenv(\"LUST_SECRET_TOKEN\") end".to_string(),
                ),
                (
                    "default_os_get_env",
                    LustConfig::default(),
                    "function f(): string return os.get_env(\"LUST_SECRET_TOKEN\") end".to_string(),
                ),
            ];
            let mut all_denied = true;
            for (label, cfg, src) in attempts {
                let compiled = EmbeddedProgram::builder()
                    .with_config(cfg)
                    .module("main", &src)
                    .entry_module("main")
                    .compile();
                let denied = match compiled {
                    Ok(mut prog) => prog.call_raw("main.f", vec![]).is_err(),
                    Err(_) => true,
                };
                if !denied {
                    all_denied = false;
                    eprintln!("UNEXPECTED_ALLOW:{label}");
                }
            }
            println!("ALL_DENIED={all_denied}");
        }
        "check_fd_closed" => {
            let fd: i32 = std::env::var("LUST_INHERITED_FD")
                .expect("LUST_INHERITED_FD missing")
                .parse()
                .expect("valid fd number");
            let flags = unsafe { fcntl(fd, F_GETFD) };
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            assert_eq!(
                flags, -1,
                "inherited non-CLOEXEC fd {fd} remained open in child with flags={flags}"
            );
            assert_eq!(
                errno, EBADF,
                "expected EBADF ({EBADF}) for sanitized fd {fd}, got errno={errno}"
            );
            println!("FD_CLOSED_OK={fd}");
        }
        "detached_stdout_holder" => {
            let pid_file = std::env::var("LUST_DETACHED_PID_FILE").expect("detached pid file");
            println!("BEFORE_DETACH_MARKER");
            let _ = std::io::stdout().flush();

            let mut cmd = Command::new("/bin/sleep");
            cmd.arg("3")
                .stdin(Stdio::null())
                .stdout(Stdio::inherit())
                .stderr(Stdio::null());
            unsafe {
                cmd.pre_exec(|| {
                    if setsid() == -1 {
                        return Err(std::io::Error::from_raw_os_error(1));
                    }
                    Ok(())
                });
            }
            let child = cmd.spawn().expect("spawn setsid /bin/sleep descendant");
            let pid = child.id();
            {
                let mut f = File::create(&pid_file).expect("create detached pid file");
                writeln!(f, "{pid}").expect("write detached pid");
                let _ = f.sync_all();
            }
            loop {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        other => panic!("unknown worker mode: {other}"),
    }
}

#[test]
fn test_4f_1_permitted_control_executes_and_cleans_tempdir() {
    let policy = worker_policy("permitted_control");
    let outcome = run_sandboxed(&policy).expect("permitted control must succeed");

    assert_eq!(outcome.exit_code, Some(0));
    assert_eq!(outcome.signal, None);
    assert!(!outcome.timed_out);
    assert!(!outcome.cancelled);
    assert!(!outcome.output_drain_timed_out);
    let stdout = String::from_utf8_lossy(&outcome.stdout);
    assert!(stdout.contains("CONTROL_OK"), "stdout was: {stdout}");
    let cwd_line = stdout
        .lines()
        .find(|l| l.starts_with("CWD="))
        .expect("CWD line missing");
    let child_cwd = PathBuf::from(&cwd_line["CWD=".len()..]);
    assert!(
        child_cwd != std::env::current_dir().unwrap(),
        "child must not inherit parent cwd"
    );
    assert_eq!(
        child_cwd.file_name(),
        outcome.temp_dir_path.file_name(),
        "child cwd must match private sandbox tempdir"
    );
    assert!(outcome.temp_dir_cleaned, "tempdir must be cleaned on return");
    assert!(
        !outcome.temp_dir_path.exists(),
        "tempdir path must not exist after cleanup: {}",
        outcome.temp_dir_path.display()
    );
}

#[test]
fn test_4f_2_timeout_kills_pgid_via_killpg_within_budget() {
    let mut policy = worker_policy("busy_loop");
    policy.limits.wall_timeout = Duration::from_millis(250);

    let outcome = run_sandboxed(&policy).expect("sandboxed timeout run must return outcome");
    assert!(outcome.timed_out, "must report timed_out=true");
    assert_eq!(outcome.signal, Some(9), "must terminate via SIGKILL (9)");
    assert!(
        outcome.elapsed < Duration::from_millis(2000),
        "watchdog termination took too long: {:?}",
        outcome.elapsed
    );
    assert!(outcome.temp_dir_cleaned);
    assert!(!outcome.temp_dir_path.exists());
}

#[test]
fn test_4f_3_descendant_in_pgid_terminated_on_timeout() {
    // Note: POSIX process groups (`setpgid(0, 0)` + `killpg`) terminate descendants
    // that remain in the process group; `setsid`-escaped descendants require Linux
    // cgroups v2 / PID namespaces and are documented as a Darwin baseline boundary.
    let marker_dir = tempfile::tempdir().expect("marker tempdir");
    let marker_file = marker_dir.path().join("grandchild.pid");

    let mut policy = worker_policy("spawn_descendant_and_sleep");
    policy.limits.wall_timeout = Duration::from_millis(450);
    policy.allowed_env.insert(
        "LUST_DESCENDANT_PID_FILE".to_string(),
        marker_file.to_string_lossy().to_string(),
    );

    let outcome = run_sandboxed(&policy).expect("run_sandboxed must succeed");
    assert!(outcome.timed_out);
    assert_eq!(outcome.signal, Some(9));

    let pid_text = fs::read_to_string(&marker_file).expect("descendant pid file must exist");
    let grandchild_pid: i32 = pid_text.trim().parse().expect("valid descendant pid");

    let wait_start = Instant::now();
    let mut dead = false;
    while wait_start.elapsed() < Duration::from_secs(2) {
        let rc = unsafe { kill(grandchild_pid, 0) };
        if rc == -1 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(3) {
                // ESRCH: kernel reaped the killed grandchild
                dead = true;
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(
        dead,
        "descendant pid {grandchild_pid} survived killpg timeout"
    );
}

#[test]
fn test_4f_4_env_and_cwd_isolation_strips_parent_secrets() {
    let secret_key = "LUST_TEST_PARENT_SECRET";
    let secret_val = "super_secret_token_999";
    // SAFETY: tests run serially via --test-threads=1 per plan contract.
    unsafe {
        std::env::set_var(secret_key, secret_val);
    }
    let parent_cwd = fs::canonicalize(std::env::current_dir().unwrap()).unwrap();

    let mut policy = worker_policy("dump_env_cwd");
    policy
        .allowed_env
        .insert("LUST_ALLOWED_MARKER".to_string(), "visible_ok".to_string());

    let outcome = run_sandboxed(&policy);
    unsafe {
        std::env::remove_var(secret_key);
    }

    let outcome = outcome.expect("env/cwd probe must succeed");
    assert_eq!(outcome.exit_code, Some(0));
    let stdout = String::from_utf8_lossy(&outcome.stdout);
    let stderr = String::from_utf8_lossy(&outcome.stderr);

    assert!(
        !stdout.contains(secret_key) && !stdout.contains(secret_val),
        "parent secret leaked into child stdout: {stdout}"
    );
    assert!(
        !stderr.contains(secret_key) && !stderr.contains(secret_val),
        "parent secret leaked into child stderr: {stderr}"
    );
    assert!(
        stdout.contains("ENV:LUST_ALLOWED_MARKER=visible_ok"),
        "allowlisted env var missing: {stdout}"
    );
    let cwd_line = stdout
        .lines()
        .find(|l| l.starts_with("CWD="))
        .expect("CWD line missing");
    let child_cwd = PathBuf::from(&cwd_line["CWD=".len()..]);
    assert_ne!(child_cwd, parent_cwd, "child CWD must be isolated from parent");
}

#[test]
fn test_4f_5_stdout_stderr_byte_caps_truncate_boundedly() {
    let mut policy = worker_policy("flood_stdio");
    policy.limits.max_stdout_bytes = 4096;
    policy.limits.max_stderr_bytes = 2048;

    let outcome = run_sandboxed(&policy).expect("bounded stdio flood must complete");
    assert_eq!(outcome.exit_code, Some(0));
    assert!(
        outcome.stdout.len() <= 4096,
        "stdout exceeded cap: {}",
        outcome.stdout.len()
    );
    assert!(
        outcome.stderr.len() <= 2048,
        "stderr exceeded cap: {}",
        outcome.stderr.len()
    );
    assert!(outcome.stdout_truncated, "stdout_truncated must be true");
    assert!(outcome.stderr_truncated, "stderr_truncated must be true");
}

#[test]
fn test_4f_6_malformed_policy_and_linux_isolation_fail_closed() {
    let valid_exe = std::env::current_exe().unwrap();

    let mut zero_timeout = SandboxPolicy::new(&valid_exe);
    zero_timeout.limits.wall_timeout = Duration::ZERO;
    assert!(matches!(
        run_sandboxed(&zero_timeout),
        Err(SandboxError::InvalidPolicy(_))
    ));

    let mut zero_stdout = SandboxPolicy::new(&valid_exe);
    zero_stdout.limits.max_stdout_bytes = 0;
    assert!(matches!(
        run_sandboxed(&zero_stdout),
        Err(SandboxError::InvalidPolicy(_))
    ));

    let mut zero_stderr = SandboxPolicy::new(&valid_exe);
    zero_stderr.limits.max_stderr_bytes = 0;
    assert!(matches!(
        run_sandboxed(&zero_stderr),
        Err(SandboxError::InvalidPolicy(_))
    ));

    let missing_bin = SandboxPolicy::new("/nonexistent/lust_missing_sandbox_binary_4f");
    assert!(matches!(
        run_sandboxed(&missing_bin),
        Err(SandboxError::InvalidPolicy(_))
    ));

    #[cfg(not(target_os = "linux"))]
    {
        for iso in [
            IsolationRequirements {
                require_linux_cgroups_v2: true,
                ..Default::default()
            },
            IsolationRequirements {
                require_seccomp: true,
                ..Default::default()
            },
            IsolationRequirements {
                require_net_namespace: true,
                ..Default::default()
            },
        ] {
            let mut p = SandboxPolicy::new(&valid_exe);
            p.isolation = iso;
            assert!(
                matches!(run_sandboxed(&p), Err(SandboxError::UnsupportedIsolation(_))),
                "non-Linux host must fail closed with UnsupportedIsolation"
            );
        }
    }
}

#[test]
fn test_4f_7_darwin_rlimits_applied_and_reported_accurately() {
    // Part A: RLIMIT_CPU(1) triggers kernel CPU kill before wall_timeout(5000ms),
    // and RLIMIT_AS reports truthful EINVAL on Darwin arm64 without faking RSS confinement.
    let mut cpu_policy = worker_policy("cpu_burn");
    cpu_policy.limits.wall_timeout = Duration::from_millis(5000);
    cpu_policy.limits.cpu_seconds = Some(1);
    cpu_policy.limits.address_space_bytes = Some(1024 * 1024 * 1024);

    let cpu_outcome = run_sandboxed(&cpu_policy).expect("cpu burn under RLIMIT_CPU must return");
    assert!(!cpu_outcome.timed_out, "kernel RLIMIT_CPU must fire before wall timeout");
    assert!(
        matches!(cpu_outcome.signal, Some(24) | Some(9)),
        "expected SIGXCPU (24) or SIGKILL (9), got exit={:?} sig={:?}",
        cpu_outcome.exit_code,
        cpu_outcome.signal
    );
    assert_eq!(cpu_outcome.rlimits.cpu.requested, Some(1));
    assert!(cpu_outcome.rlimits.cpu.applied);
    assert_eq!(cpu_outcome.rlimits.cpu.effective_cur, Some(1));
    #[cfg(target_os = "macos")]
    {
        assert_eq!(
            cpu_outcome.rlimits.address_space.requested,
            Some(1024 * 1024 * 1024)
        );
        assert!(
            !cpu_outcome.rlimits.address_space.applied,
            "Darwin arm64 must not fake RLIMIT_AS enforcement"
        );
        assert_eq!(cpu_outcome.rlimits.address_space.errno, Some(22));
    }

    // Part B: RLIMIT_NOFILE(64) bounds open descriptors inside the child.
    let mut nofile_policy = worker_policy("open_fd_exhaust");
    nofile_policy.limits.wall_timeout = Duration::from_millis(3000);
    nofile_policy.limits.nofile_limit = Some(64);

    let fd_outcome = run_sandboxed(&nofile_policy).expect("nofile probe must return");
    assert_eq!(fd_outcome.exit_code, Some(0));
    let stdout = String::from_utf8_lossy(&fd_outcome.stdout);
    let opened_count: usize = stdout
        .lines()
        .find_map(|l| l.strip_prefix("OPENED_COUNT="))
        .and_then(|s| s.parse().ok())
        .expect("OPENED_COUNT missing");
    assert!(
        opened_count > 0 && opened_count < 64,
        "expected open FD limit < 64, got {opened_count}"
    );
    assert_eq!(fd_outcome.rlimits.nofile.requested, Some(64));
    assert!(fd_outcome.rlimits.nofile.applied);
    assert_eq!(fd_outcome.rlimits.nofile.effective_cur, Some(64));
}

#[test]
fn test_4f_8_denied_vs_permitted_lust_script_capabilities() {
    let dir = tempfile::tempdir().expect("capability tempdir");
    let effect_path = dir.path().join("effect.txt");
    let canary_path = dir.path().join("canary.txt");
    fs::write(&canary_path, "CANARY_INTACT_V1").expect("seed canary");

    // 1. Permitted control: host callback produces observable effect inside sandbox.
    let mut perm_policy = worker_policy("lust_permitted");
    perm_policy.allowed_env.insert(
        "LUST_EFFECT_PATH".to_string(),
        effect_path.to_string_lossy().to_string(),
    );
    let perm_out = run_sandboxed(&perm_policy).expect("permitted lust worker must succeed");
    assert_eq!(perm_out.exit_code, Some(0));
    assert_eq!(
        fs::read_to_string(&effect_path).expect("effect file"),
        "PERMITTED_EFFECT_RECORDED"
    );

    // 2. Denied script: pure/default LustConfig blocks io.open/io.read_file/io.write_file/os.getenv.
    let mut denied_policy = worker_policy("lust_denied");
    denied_policy.allowed_env.insert(
        "LUST_CANARY_PATH".to_string(),
        canary_path.to_string_lossy().to_string(),
    );
    denied_policy.allowed_env.insert(
        "LUST_SECRET_TOKEN".to_string(),
        "SECRET_VAL_XYZ".to_string(),
    );
    let denied_out = run_sandboxed(&denied_policy).expect("denied lust worker must run");
    assert_eq!(denied_out.exit_code, Some(0));
    let stdout = String::from_utf8_lossy(&denied_out.stdout);
    let stderr = String::from_utf8_lossy(&denied_out.stderr);
    assert!(stdout.contains("ALL_DENIED=true"), "stdout={stdout} stderr={stderr}");
    assert_eq!(
        fs::read_to_string(&canary_path).expect("canary file"),
        "CANARY_INTACT_V1"
    );
    assert!(
        !stdout.contains("SECRET_VAL_XYZ") && !stderr.contains("SECRET_VAL_XYZ"),
        "secret leaked in denied run"
    );
}

#[test]
fn test_4f_9_cranelift_mir_logical_not_matches_bytecode() {
    let src = r#"
        function not_true(): bool
            return not true
        end

        function not_false(): bool
            return not false
        end

        function not_param(b: bool): bool
            return not b
        end

        function check_bool(x: int): bool
            return x > 0
        end

        function if_not_call(x: int): int
            if not check_bool(x) then
                return 11
            else
                return 22
            end
        end

        function guard_loop(n: int, x: int): bool
            local i: int = 0
            local ok: bool = true
            while i < n do
                if not check_bool(x) then
                    ok = false
                end
                i = i + 1
            end
            return ok
        end
    "#;

    let mut interner = Interner::new();
    let tokens = Lexer::new(src, &mut interner).tokenize().expect("tokenize");
    let items = Parser::new(tokens).parse().expect("parse");

    let config = LustConfig::default();
    let mut checker = TypeChecker::with_config(&config);
    checker.check_module(&items).expect("typecheck");
    let struct_defs = checker.struct_definitions();

    let mut mir_module = MirModule::new("not_parity".into());
    for item in &items {
        if let lust::ast::ItemKind::Function(func_def) = &item.kind {
            let mir_fn = AstLowering::lower_function(func_def).expect("MIR lower");
            mir_module.add_function(mir_fn);
        }
    }

    let mut vm_clif = VM::new();
    vm_clif.set_backend(VmBackend::Cranelift);
    vm_clif.load_mir_module(&mir_module).expect("load MIR");
    vm_clif.register_structs(&struct_defs);

    let mut vm_bc = VM::new();
    vm_bc.register_structs(&struct_defs);
    let mut compiler = Compiler::new();
    compiler.configure(&config);
    compiler.set_numeric_types(checker.take_numeric_types());
    compiler.set_function_signatures(checker.take_function_signatures());
    let bc_funcs = compiler.compile_module(&items).expect("bytecode compile");
    vm_bc.load_functions(bc_funcs);
    vm_bc.set_backend(VmBackend::Bytecode);

    let cases: Vec<(&str, Vec<Value>, Value)> = vec![
        ("not_true", vec![], Value::Bool(false)),
        ("not_false", vec![], Value::Bool(true)),
        ("not_param", vec![Value::Bool(true)], Value::Bool(false)),
        ("not_param", vec![Value::Bool(false)], Value::Bool(true)),
        ("if_not_call", vec![Value::Int(0)], Value::Int(11)),
        ("if_not_call", vec![Value::Int(1)], Value::Int(22)),
        ("guard_loop", vec![Value::Int(3), Value::Int(1)], Value::Bool(true)),
        ("guard_loop", vec![Value::Int(3), Value::Int(0)], Value::Bool(false)),
    ];

    for (func, args, expected) in cases {
        let e0 = vm_clif.cranelift_entry_count;
        let clif_res = vm_clif.call(func, args.clone()).expect("Cranelift call");
        let bc_res = vm_bc.call(func, args.clone()).expect("Bytecode call");
        assert!(
            vm_clif.cranelift_entry_count > e0,
            "{func} must execute via Cranelift native entry"
        );
        assert_eq!(bc_res, expected, "Bytecode oracle mismatch for {func}({args:?})");
        assert_eq!(
            clif_res, bc_res,
            "Cranelift vs Bytecode mismatch for {func}({args:?}): clif={clif_res:?} vs bc={bc_res:?}"
        );
    }
}

#[test]
fn test_4f_10_inherited_non_cloexec_fd_closed_before_exec() {
    let mut fds = [-1i32; 2];
    assert_eq!(unsafe { pipe(fds.as_mut_ptr()) }, 0, "pipe() failed");
    let read_guard = FdGuard(fds[0]);
    let write_guard = FdGuard(fds[1]);

    for &fd in &fds {
        assert!(fd >= 3, "pipe fd must be >= 3, got {fd}");
        let flags = unsafe { fcntl(fd, F_GETFD) };
        assert!(flags >= 0, "fcntl(F_GETFD) failed on parent fd {fd}");
        assert_eq!(
            flags & FD_CLOEXEC,
            0,
            "parent pipe fd {fd} unexpectedly had FD_CLOEXEC set"
        );
    }

    let mut policy = worker_policy("check_fd_closed");
    policy
        .allowed_env
        .insert("LUST_INHERITED_FD".to_string(), fds[0].to_string());

    let outcome = run_sandboxed(&policy).expect("sandboxed worker must run cleanly");

    drop(read_guard);
    drop(write_guard);

    assert_eq!(
        outcome.exit_code,
        Some(0),
        "worker failed: stdout={} stderr={}",
        String::from_utf8_lossy(&outcome.stdout),
        String::from_utf8_lossy(&outcome.stderr)
    );
    assert!(!outcome.timed_out);
    assert!(!outcome.output_drain_timed_out);
    let stdout = String::from_utf8_lossy(&outcome.stdout);
    assert!(
        stdout.contains(&format!("FD_CLOSED_OK={}", fds[0])),
        "expected FD_CLOSED_OK={} in stdout, got: {stdout}",
        fds[0]
    );
}

#[test]
fn test_4f_11_detached_descendant_holding_stdout_does_not_hang_supervisor() {
    let marker_dir = tempfile::tempdir().expect("detached marker tempdir");
    let pid_file = marker_dir.path().join("detached.pid");
    let _cleanup = DetachedPidCleanup(pid_file.clone());

    let mut policy = worker_policy("detached_stdout_holder");
    policy.limits.wall_timeout = Duration::from_millis(150);
    policy.allowed_env.insert(
        "LUST_DETACHED_PID_FILE".to_string(),
        pid_file.to_string_lossy().to_string(),
    );

    let wall_start = Instant::now();
    let outcome = run_sandboxed(&policy).expect("sandboxed run must not hang");
    let wall_elapsed = wall_start.elapsed();

    let pid_text = fs::read_to_string(&pid_file).expect("detached pid file must exist");
    let detached_pid: i32 = pid_text.trim().parse().expect("valid detached pid");

    // Immediately terminate the detached setsid descendant so zero orphans survive.
    let _ = unsafe { kill(detached_pid, SIGKILL) };

    assert!(
        wall_elapsed < Duration::from_millis(1500),
        "run_sandboxed hung on detached stdout holder: wall_elapsed={wall_elapsed:?}"
    );
    assert!(
        outcome.elapsed < Duration::from_millis(1500),
        "outcome.elapsed exceeded budget: {:?}",
        outcome.elapsed
    );
    assert!(outcome.timed_out, "expected timed_out=true");
    assert_eq!(outcome.signal, Some(SIGKILL), "expected SIGKILL on timed-out worker");
    assert!(
        outcome.output_drain_timed_out,
        "expected output_drain_timed_out=true when setsid descendant holds stdout open"
    );
    let stdout = String::from_utf8_lossy(&outcome.stdout);
    assert!(
        stdout.contains("BEFORE_DETACH_MARKER"),
        "expected partial captured stdout to be preserved, got: {stdout}"
    );

    let reap_start = Instant::now();
    let mut dead = false;
    while reap_start.elapsed() < Duration::from_secs(2) {
        let rc = unsafe { kill(detached_pid, 0) };
        if rc == -1 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(ESRCH) {
                dead = true;
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(
        dead,
        "detached descendant pid {detached_pid} survived teardown SIGKILL"
    );
}

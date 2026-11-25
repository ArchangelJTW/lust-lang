//! Wave 18 Linux `S-SANDBOX` integration tests (`cgroups v2`, `seccomp-bpf`, namespaces).
//!
//! Verifies:
//! 1. `cgroups v2` `memory.max` + `pids.max` + `cgroup.kill` terminates a `setsid()`-detached grandchild on timeout.
//! 2. `seccomp-bpf` blocks `ptrace`, `mount`, and `bpf` with `EPERM`.
//! 3. `CLONE_NEWNET` isolates network namespace (distinct `/proc/self/ns/net`, no `eth0`, and `AF_INET` blocked when seccomp is active).

#![cfg(target_os = "linux")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use lust::sandbox::{IsolationRequirements, SandboxPolicy, run_sandboxed};
use std::fs::{self, File};
use std::io::Write;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

const WORKER_ENV_KEY: &str = "LUST_W18_WORKER_MODE";
const EPERM: i32 = 1;
const ESRCH: i32 = 3;
const SIGKILL: i32 = 9;
const AF_INET: i32 = 2;
const SOCK_STREAM: i32 = 1;

#[cfg(target_arch = "x86_64")]
mod sys {
    pub const SYS_PTRACE: isize = 101;
    pub const SYS_MOUNT: isize = 165;
    pub const SYS_BPF: isize = 321;
}

#[cfg(target_arch = "aarch64")]
mod sys {
    pub const SYS_PTRACE: isize = 117;
    pub const SYS_MOUNT: isize = 40;
    pub const SYS_BPF: isize = 280;
}

unsafe extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
    fn setsid() -> i32;
    fn syscall(num: isize, ...) -> isize;
    fn socket(domain: i32, ty: i32, protocol: i32) -> i32;
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

fn worker_policy(mode: &str) -> SandboxPolicy {
    let exe = std::env::current_exe().expect("current test executable path");
    let mut policy = SandboxPolicy::new(exe);
    policy.args = vec![
        "--exact".to_string(),
        "__w18_linux_worker".to_string(),
        "--ignored".to_string(),
        "--nocapture".to_string(),
    ];
    policy
        .allowed_env
        .insert(WORKER_ENV_KEY.to_string(), mode.to_string());
    policy
}

#[test]
#[ignore]
fn __w18_linux_worker() {
    let Ok(mode) = std::env::var(WORKER_ENV_KEY) else {
        return;
    };
    match mode.as_str() {
        "grandchild_sleep" => loop {
            std::thread::sleep(Duration::from_secs(60));
        },
        "spawn_setsid_grandchild" => {
            let pid_file = std::env::var("LUST_GRANDCHILD_PID_FILE").expect("pid file");
            let cgroup_rel = fs::read_to_string("/proc/self/cgroup")
                .expect("read /proc/self/cgroup")
                .lines()
                .next()
                .and_then(|l| l.split(':').nth(2))
                .unwrap_or("")
                .trim()
                .to_string();
            println!("CHILD_CGROUP={cgroup_rel}");

            let cg_dir = PathBuf::from("/sys/fs/cgroup").join(cgroup_rel.trim_start_matches('/'));
            if let Ok(mem_max) = fs::read_to_string(cg_dir.join("memory.max")) {
                println!("MEMORY_MAX={}", mem_max.trim());
            }
            if let Ok(pids_max) = fs::read_to_string(cg_dir.join("pids.max")) {
                println!("PIDS_MAX={}", pids_max.trim());
            }

            let exe = std::env::current_exe().expect("current_exe");
            let mut cmd = Command::new(exe);
            cmd.args([
                "--exact",
                "__w18_linux_worker",
                "--ignored",
                "--nocapture",
            ])
            .env(WORKER_ENV_KEY, "grandchild_sleep")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
            unsafe {
                cmd.pre_exec(|| {
                    if setsid() == -1 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
            let child = cmd.spawn().expect("spawn setsid grandchild");
            let mut f = File::create(&pid_file).expect("create pid file");
            writeln!(f, "{}", child.id()).expect("write pid");
            let _ = f.sync_all();
            let _ = std::io::stdout().flush();

            loop {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        "probe_seccomp" => {
            let r_ptrace = unsafe { syscall(sys::SYS_PTRACE, 0usize, 0usize, 0usize, 0usize) };
            let e_ptrace = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);

            let r_mount = unsafe { syscall(sys::SYS_MOUNT, 0usize, 0usize, 0usize, 0usize, 0usize) };
            let e_mount = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);

            let r_bpf = unsafe { syscall(sys::SYS_BPF, 0usize, 0usize, 0usize) };
            let e_bpf = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);

            println!("PTRACE=({r_ptrace},{e_ptrace})");
            println!("MOUNT=({r_mount},{e_mount})");
            println!("BPF=({r_bpf},{e_bpf})");

            assert_eq!(r_ptrace, -1);
            assert_eq!(e_ptrace, EPERM);
            assert_eq!(r_mount, -1);
            assert_eq!(e_mount, EPERM);
            assert_eq!(r_bpf, -1);
            assert_eq!(e_bpf, EPERM);
            println!("SECCOMP_ALL_EPERM=true");
        }
        "probe_net_ns" => {
            let ns_link = fs::read_link("/proc/self/ns/net")
                .expect("readlink /proc/self/ns/net")
                .to_string_lossy()
                .into_owned();
            println!("CHILD_NET_NS={ns_link}");

            let net_dev = fs::read_to_string("/proc/net/dev").unwrap_or_default();
            let has_eth0 = net_dev.contains("eth0:");
            println!("HAS_ETH0={has_eth0}");

            let sock_fd = unsafe { socket(AF_INET, SOCK_STREAM, 0) };
            let sock_errno = if sock_fd < 0 {
                std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
            } else {
                0
            };
            println!("INET_SOCK=({sock_fd},{sock_errno})");
        }
        other => panic!("unknown worker mode: {other}"),
    }
}

#[test]
fn test_w18_cgroups_v2_kills_setsid_grandchild_and_enforces_limits() {
    let dir_path = std::env::temp_dir().join(format!("lust_w18_cg_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir_path);
    fs::create_dir_all(&dir_path).expect("create test temp dir");
    let pid_file = dir_path.join("grandchild.pid");
    let _cleanup = DetachedPidCleanup(pid_file.clone());

    let mut policy = worker_policy("spawn_setsid_grandchild");
    policy
        .allowed_env
        .insert("LUST_GRANDCHILD_PID_FILE".into(), pid_file.display().to_string());
    policy.limits.wall_timeout = Duration::from_millis(350);
    policy.limits.data_bytes = Some(256 * 1024 * 1024);
    policy.limits.cpu_seconds = Some(5);
    policy.isolation = IsolationRequirements {
        require_linux_cgroups_v2: true,
        require_seccomp: false,
        require_net_namespace: false,
    };

    let outcome = run_sandboxed(&policy).expect("cgroups v2 sandboxed run");
    let stdout = String::from_utf8_lossy(&outcome.stdout);
    let stderr = String::from_utf8_lossy(&outcome.stderr);
    assert!(
        outcome.timed_out,
        "expected wall-clock timeout, got exit={:?} signal={:?} stdout={stdout} stderr={stderr}",
        outcome.exit_code, outcome.signal
    );
    assert_eq!(outcome.platform_mode, "linux-s-sandbox");
    assert!(
        stdout.contains("CHILD_CGROUP=/lust_sandbox_"),
        "expected child in dedicated cgroup, got stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("MEMORY_MAX=268435456"),
        "expected memory.max=268435456, got stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("PIDS_MAX=64"),
        "expected pids.max=64, got stdout:\n{stdout}"
    );

    let pid_text = fs::read_to_string(&pid_file).expect("grandchild pid marker written");
    let grandchild_pid: i32 = pid_text.trim().parse().expect("valid grandchild pid");
    assert!(grandchild_pid > 0);

    // Give kernel a brief moment to finish reaping after cgroup.kill.
    let mut dead = false;
    for _ in 0..20 {
        let rc = unsafe { kill(grandchild_pid, 0) };
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        if rc == -1 && errno == ESRCH {
            dead = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        dead,
        "setsid()-detached grandchild {grandchild_pid} survived cgroup.kill!"
    );

    // Verify cgroup directory was cleaned up.
    let cg_line = stdout
        .lines()
        .find_map(|l| l.find("CHILD_CGROUP=").map(|idx| &l[idx..]))
        .expect("CHILD_CGROUP line");
    let cg_rel = cg_line.trim_start_matches("CHILD_CGROUP=").trim_start_matches('/');
    assert!(
        !PathBuf::from("/sys/fs/cgroup").join(cg_rel).exists(),
        "cgroup directory {cg_rel} was not cleaned up"
    );
}

#[test]
fn test_w18_seccomp_bpf_blocks_dangerous_syscalls_with_eperm() {
    let mut policy = worker_policy("probe_seccomp");
    policy.isolation = IsolationRequirements {
        require_linux_cgroups_v2: false,
        require_seccomp: true,
        require_net_namespace: false,
    };

    let outcome = run_sandboxed(&policy).expect("seccomp sandboxed run");
    let stdout = String::from_utf8_lossy(&outcome.stdout);
    let stderr = String::from_utf8_lossy(&outcome.stderr);
    assert_eq!(
        outcome.exit_code,
        Some(0),
        "seccomp worker failed: stdout={stdout}\nstderr={stderr}"
    );
    assert!(stdout.contains("SECCOMP_ALL_EPERM=true"));
}

#[test]
fn test_w18_net_namespace_isolates_network_and_blocks_inet_sockets() {
    let parent_ns = fs::read_link("/proc/self/ns/net")
        .expect("parent net ns")
        .to_string_lossy()
        .into_owned();

    let mut policy = worker_policy("probe_net_ns");
    policy.isolation = IsolationRequirements {
        require_linux_cgroups_v2: false,
        require_seccomp: true,
        require_net_namespace: true,
    };

    let outcome = run_sandboxed(&policy).expect("netns + seccomp sandboxed run");
    let stdout = String::from_utf8_lossy(&outcome.stdout);
    let stderr = String::from_utf8_lossy(&outcome.stderr);
    assert_eq!(
        outcome.exit_code,
        Some(0),
        "netns worker failed: stdout={stdout}\nstderr={stderr}"
    );

    let child_ns = stdout
        .lines()
        .find_map(|l| l.strip_prefix("CHILD_NET_NS="))
        .expect("CHILD_NET_NS line");
    assert_ne!(
        child_ns, parent_ns,
        "child must be in a distinct network namespace"
    );
    assert!(
        stdout.contains("HAS_ETH0=false"),
        "isolated netns must not expose host/container eth0: {stdout}"
    );
    assert!(
        stdout.contains(&format!("INET_SOCK=(-1,{EPERM})")),
        "AF_INET socket must be blocked with EPERM: {stdout}"
    );
}

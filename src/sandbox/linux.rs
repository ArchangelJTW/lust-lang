//! Linux `S-SANDBOX` backend (`cgroups v2`, `seccomp-bpf`, and namespaces).
//!
//! Uses direct Linux syscalls (`prctl`, `unshare`, `kill`) and `sysfs`/`procfs`
//! (`std`-only, zero external crates). All `pre_exec` data (`SockFilter` BPF
//! program) is precomputed before `fork()` with zero heap allocations inside
//! `pre_exec`.

use crate::sandbox::{SandboxError, SandboxLimits, SandboxPolicy};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

const CGROUP_ROOT: &str = "/sys/fs/cgroup";
static CGROUP_SEQ: AtomicU64 = AtomicU64::new(1);

const PR_GET_SECCOMP: i32 = 21;
const PR_SET_SECCOMP: i32 = 22;
const PR_SET_NO_NEW_PRIVS: i32 = 38;
const SECCOMP_MODE_FILTER: usize = 2;

const CLONE_NEWUTS: i32 = 0x0400_0000;
const CLONE_NEWIPC: i32 = 0x0800_0000;
const CLONE_NEWNET: i32 = 0x4000_0000;

const EINVAL: i32 = 22;
const SIGKILL: i32 = 9;

const SECCOMP_RET_ALLOW: u32 = 0x7fff_0000;
const SECCOMP_RET_ERRNO: u32 = 0x0005_0000;
const EPERM: u32 = 1;

const BPF_LD_W_ABS: u16 = 0x20;
const BPF_JMP_JEQ_K: u16 = 0x15;
const BPF_RET_K: u16 = 0x06;

const AF_INET: u32 = 2;
const AF_INET6: u32 = 10;

const MAX_BPF_INSNS: usize = 32;

#[cfg(target_arch = "x86_64")]
mod arch_sys {
    pub const AUDIT_ARCH: u32 = 0xc000_003e;
    pub const SYS_PTRACE: u32 = 101;
    pub const SYS_KEXEC_LOAD: u32 = 246;
    pub const SYS_INIT_MODULE: u32 = 175;
    pub const SYS_FINIT_MODULE: u32 = 313;
    pub const SYS_DELETE_MODULE: u32 = 176;
    pub const SYS_MOUNT: u32 = 165;
    pub const SYS_UMOUNT2: u32 = 166;
    pub const SYS_REBOOT: u32 = 169;
    pub const SYS_SWAPON: u32 = 167;
    pub const SYS_SWAPOFF: u32 = 168;
    pub const SYS_BPF: u32 = 321;
    pub const SYS_USERFAULTFD: u32 = 323;
    pub const SYS_SOCKET: u32 = 41;
}

#[cfg(target_arch = "aarch64")]
mod arch_sys {
    pub const AUDIT_ARCH: u32 = 0xc000_00b7;
    pub const SYS_PTRACE: u32 = 117;
    pub const SYS_KEXEC_LOAD: u32 = 104;
    pub const SYS_INIT_MODULE: u32 = 105;
    pub const SYS_FINIT_MODULE: u32 = 273;
    pub const SYS_DELETE_MODULE: u32 = 106;
    pub const SYS_MOUNT: u32 = 40;
    pub const SYS_UMOUNT2: u32 = 39;
    pub const SYS_REBOOT: u32 = 142;
    pub const SYS_SWAPON: u32 = 224;
    pub const SYS_SWAPOFF: u32 = 225;
    pub const SYS_BPF: u32 = 280;
    pub const SYS_USERFAULTFD: u32 = 282;
    pub const SYS_SOCKET: u32 = 198;
}

const BLOCKED_SYSCALLS: [u32; 12] = [
    arch_sys::SYS_PTRACE,
    arch_sys::SYS_KEXEC_LOAD,
    arch_sys::SYS_INIT_MODULE,
    arch_sys::SYS_FINIT_MODULE,
    arch_sys::SYS_DELETE_MODULE,
    arch_sys::SYS_MOUNT,
    arch_sys::SYS_UMOUNT2,
    arch_sys::SYS_REBOOT,
    arch_sys::SYS_SWAPON,
    arch_sys::SYS_SWAPOFF,
    arch_sys::SYS_BPF,
    arch_sys::SYS_USERFAULTFD,
];

unsafe extern "C" {
    fn __errno_location() -> *mut i32;
    fn prctl(option: i32, arg2: usize, arg3: usize, arg4: usize, arg5: usize) -> i32;
    fn unshare(flags: i32) -> i32;
    fn kill(pid: i32, sig: i32) -> i32;
    fn write(fd: i32, buf: *const std::ffi::c_void, count: usize) -> isize;
    fn close(fd: i32) -> i32;
}

#[inline]
unsafe fn raw_errno() -> i32 {
    let ptr = unsafe { __errno_location() };
    if ptr.is_null() {
        -1
    } else {
        unsafe { *ptr }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SockFilter {
    code: u16,
    jt: u8,
    jf: u8,
    k: u32,
}

#[repr(C)]
struct SockFprog {
    len: u16,
    filter: *const SockFilter,
}

const fn bpf_stmt(code: u16, k: u32) -> SockFilter {
    SockFilter {
        code,
        jt: 0,
        jf: 0,
        k,
    }
}

const fn bpf_jump(code: u16, k: u32, jt: u8, jf: u8) -> SockFilter {
    SockFilter { code, jt, jf, k }
}

/// Precomputed, allocation-free Linux `pre_exec` isolation configuration.
#[derive(Clone, Copy)]
pub(crate) struct LinuxPreExecConfig {
    pub unshare_net: bool,
    pub install_seccomp: bool,
    pub cgroup_procs_fd: i32,
    pub require_cgroup: bool,
    bpf_prog: [SockFilter; MAX_BPF_INSNS],
    bpf_len: u16,
}

impl LinuxPreExecConfig {
    pub(crate) fn from_policy(
        policy: &SandboxPolicy,
        cgroup_procs_fd: i32,
    ) -> Self {
        let install_seccomp = policy.isolation.require_seccomp;
        let unshare_net = policy.isolation.require_net_namespace;
        let block_inet = policy.isolation.require_net_namespace;

        let mut prog = [SockFilter::default(); MAX_BPF_INSNS];
        let mut len = 0usize;

        if install_seccomp {
            // Compute exact allow/deny indices upfront so all jump offsets are exact.
            // Header: 4 insns (LD arch, JEQ arch, RET errno, LD nr)
            // Blocked syscalls: BLOCKED_SYSCALLS.len() insns
            // Optional INET socket check: 4 insns
            let extra_sock = if block_inet { 4 } else { 0 };
            let allow_idx = 4 + BLOCKED_SYSCALLS.len() + extra_sock;
            let deny_idx = allow_idx + 1;

            // [0] Load seccomp_data.arch (offset 4)
            prog[len] = bpf_stmt(BPF_LD_W_ABS, 4);
            len += 1;
            // [1] Verify architecture matches compiled target
            prog[len] = bpf_jump(BPF_JMP_JEQ_K, arch_sys::AUDIT_ARCH, 1, 0);
            len += 1;
            // [2] Mismatched arch -> EPERM
            prog[len] = bpf_stmt(BPF_RET_K, SECCOMP_RET_ERRNO | EPERM);
            len += 1;
            // [3] Load seccomp_data.nr (offset 0)
            prog[len] = bpf_stmt(BPF_LD_W_ABS, 0);
            len += 1;

            for &sys_nr in &BLOCKED_SYSCALLS {
                let jt = (deny_idx - (len + 1)) as u8;
                prog[len] = bpf_jump(BPF_JMP_JEQ_K, sys_nr, jt, 0);
                len += 1;
            }

            if block_inet {
                // If nr == SYS_SOCKET, inspect args[0] (domain); otherwise jump to allow_idx.
                let jf_to_allow = (allow_idx - (len + 1)) as u8;
                prog[len] = bpf_jump(BPF_JMP_JEQ_K, arch_sys::SYS_SOCKET, 0, jf_to_allow);
                len += 1;
                // Load seccomp_data.args[0] low 32 bits (offset 16)
                prog[len] = bpf_stmt(BPF_LD_W_ABS, 16);
                len += 1;
                let jt_inet = (deny_idx - (len + 1)) as u8;
                prog[len] = bpf_jump(BPF_JMP_JEQ_K, AF_INET, jt_inet, 0);
                len += 1;
                let jt_inet6 = (deny_idx - (len + 1)) as u8;
                prog[len] = bpf_jump(BPF_JMP_JEQ_K, AF_INET6, jt_inet6, 0);
                len += 1;
            }

            debug_assert_eq!(len, allow_idx);
            prog[len] = bpf_stmt(BPF_RET_K, SECCOMP_RET_ALLOW);
            len += 1;
            debug_assert_eq!(len, deny_idx);
            prog[len] = bpf_stmt(BPF_RET_K, SECCOMP_RET_ERRNO | EPERM);
            len += 1;
        }

        Self {
            unshare_net,
            install_seccomp,
            cgroup_procs_fd,
            require_cgroup: policy.isolation.require_linux_cgroups_v2,
            bpf_prog: prog,
            bpf_len: len as u16,
        }
    }
}

/// Applies Linux cgroup attachment, namespaces, and `seccomp-bpf` inside `pre_exec`.
///
/// SAFETY: Executes strictly async-signal-safe syscalls (`write`, `close`,
/// `unshare`, `prctl`) with zero heap allocations.
pub(crate) unsafe fn apply_linux_pre_exec(cfg: &LinuxPreExecConfig) -> Result<(), i32> {
    if cfg.cgroup_procs_fd >= 0 {
        let rc = unsafe {
            write(
                cfg.cgroup_procs_fd,
                b"0".as_ptr().cast::<std::ffi::c_void>(),
                1,
            )
        };
        let err = if rc <= 0 { unsafe { raw_errno() } } else { 0 };
        let _ = unsafe { close(cfg.cgroup_procs_fd) };
        if rc <= 0 && cfg.require_cgroup {
            return Err(err);
        }
    }

    if cfg.unshare_net {
        if unsafe { unshare(CLONE_NEWNET) } != 0 {
            return Err(unsafe { raw_errno() });
        }
        // Best-effort UTS/IPC isolation when network namespace is requested.
        let _ = unsafe { unshare(CLONE_NEWUTS | CLONE_NEWIPC) };
    }

    if cfg.install_seccomp && cfg.bpf_len > 0 {
        if unsafe { prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } != 0 {
            return Err(unsafe { raw_errno() });
        }
        let prog = SockFprog {
            len: cfg.bpf_len,
            filter: cfg.bpf_prog.as_ptr(),
        };
        if unsafe {
            prctl(
                PR_SET_SECCOMP,
                SECCOMP_MODE_FILTER,
                (&prog as *const SockFprog) as usize,
                0,
                0,
            )
        } != 0
        {
            return Err(unsafe { raw_errno() });
        }
    }

    Ok(())
}

/// Probes whether cgroups v2 is mounted and writable for creating per-run cgroups.
pub(crate) fn probe_cgroups_v2() -> Result<(), String> {
    let controllers = Path::new(CGROUP_ROOT).join("cgroup.controllers");
    if !controllers.is_file() {
        return Err(format!(
            "Linux cgroups v2 unavailable: {} does not exist",
            controllers.display()
        ));
    }
    let seq = CGROUP_SEQ.fetch_add(1, Ordering::Relaxed);
    let probe_dir = Path::new(CGROUP_ROOT).join(format!(
        ".lust_cgroup_probe_{}_{}",
        std::process::id(),
        seq
    ));
    fs::create_dir(&probe_dir).map_err(|e| {
        format!(
            "Linux cgroups v2 root {} is not writable: {e}",
            CGROUP_ROOT
        )
    })?;
    let _ = fs::remove_dir(&probe_dir);
    Ok(())
}

/// Probes whether the running Linux kernel supports `seccomp-bpf`.
pub(crate) fn probe_seccomp() -> Result<(), String> {
    let rc = unsafe { prctl(PR_GET_SECCOMP, 0, 0, 0, 0) };
    if rc < 0 {
        let err = io::Error::last_os_error();
        if err.raw_os_error() == Some(EINVAL) {
            return Err("Linux seccomp-bpf is not supported by this kernel".into());
        }
    }
    Ok(())
}

/// Probes whether Linux network namespaces are available on this host.
/// Probes whether Linux network namespace creation (`unshare(CLONE_NEWNET)`) is permitted.
pub(crate) fn probe_net_namespace() -> Result<(), String> {
    if !Path::new("/proc/self/ns/net").exists() {
        return Err("Linux network namespace unavailable (/proc/self/ns/net missing)".into());
    }
    unsafe extern "C" {
        fn fork() -> i32;
        fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
        fn _exit(status: i32) -> !;
    }
    let pid = unsafe { fork() };
    if pid < 0 {
        let err = unsafe { raw_errno() };
        return Err(format!("fork failed while probing CLONE_NEWNET: errno {err}"));
    }
    if pid == 0 {
        let rc = unsafe { unshare(CLONE_NEWNET) };
        if rc == 0 {
            unsafe { _exit(0) };
        }
        let err = unsafe { raw_errno() }.clamp(1, 255);
        unsafe { _exit(err) };
    }
    let mut status: i32 = 0;
    let waited = unsafe { waitpid(pid, &mut status, 0) };
    if waited < 0 {
        let err = unsafe { raw_errno() };
        return Err(format!("waitpid failed while probing CLONE_NEWNET: errno {err}"));
    }
    // WIFEXITED(status) => (status & 0x7f) == 0; WEXITSTATUS(status) => (status >> 8) & 0xff
    if (status & 0x7f) == 0 {
        let code = (status >> 8) & 0xff;
        if code == 0 {
            return Ok(());
        }
        return Err(format!("unshare(CLONE_NEWNET) failed: errno {code}"));
    }
    Err(format!("CLONE_NEWNET probe child terminated abnormally: status {status}"))
}

/// Ensures `memory`, `cpu`, and `pids` controllers are delegated in `/sys/fs/cgroup/cgroup.subtree_control`.
fn ensure_subtree_controllers() -> Result<(), SandboxError> {
    let root = Path::new(CGROUP_ROOT);
    let avail_raw = fs::read_to_string(root.join("cgroup.controllers")).map_err(|e| {
        SandboxError::UnsupportedIsolation(format!("failed to read cgroup.controllers: {e}"))
    })?;
    let avail: Vec<&str> = avail_raw.split_whitespace().collect();

    let wanted: Vec<&str> = ["memory", "cpu", "pids"]
        .into_iter()
        .filter(|c| avail.contains(c))
        .collect();
    if wanted.is_empty() {
        return Ok(());
    }

    let current_subtree = fs::read_to_string(root.join("cgroup.subtree_control")).unwrap_or_default();
    let current: Vec<&str> = current_subtree.split_whitespace().collect();
    let missing: Vec<&str> = wanted
        .into_iter()
        .filter(|c| !current.contains(c))
        .collect();
    if missing.is_empty() {
        return Ok(());
    }

    // cgroups v2 no-internal-process constraint: move any processes in root
    // `cgroup.procs` into a sibling leaf cgroup before enabling subtree_control.
    let root_procs = fs::read_to_string(root.join("cgroup.procs")).unwrap_or_default();
    if !root_procs.trim().is_empty() {
        let sup_dir = root.join("lust_supervisor");
        let _ = fs::create_dir(&sup_dir);
        let sup_procs = sup_dir.join("cgroup.procs");
        for line in root_procs.lines() {
            let pid = line.trim();
            if !pid.is_empty() {
                let _ = fs::write(&sup_procs, pid);
            }
        }
    }

    let enable_str = missing
        .iter()
        .map(|c| format!("+{c}"))
        .collect::<Vec<_>>()
        .join(" ");
    fs::write(root.join("cgroup.subtree_control"), &enable_str).map_err(|e| {
        SandboxError::UnsupportedIsolation(format!(
            "failed to enable cgroup.subtree_control ({enable_str}): {e}"
        ))
    })?;

    Ok(())
}

/// RAII guard for a per-run Linux cgroup v2 directory.
pub(crate) struct CgroupGuard {
    path: PathBuf,
    cleaned: bool,
}

impl CgroupGuard {
    /// Creates a per-run cgroup under `/sys/fs/cgroup/lust_sandbox_<pid>_<seq>/`
    /// and configures `pids.max`, `memory.max`, and `cpu.max`.
    pub(crate) fn create(limits: &SandboxLimits) -> Result<Self, SandboxError> {
        ensure_subtree_controllers()?;

        let seq = CGROUP_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = Path::new(CGROUP_ROOT).join(format!(
            "lust_sandbox_{}_{}",
            std::process::id(),
            seq
        ));
        fs::create_dir(&path).map_err(|e| {
            SandboxError::UnsupportedIsolation(format!(
                "failed to create cgroup {}: {e}",
                path.display()
            ))
        })?;

        let guard = Self {
            path,
            cleaned: false,
        };

        let pids_max = guard.path.join("pids.max");
        if let Err(e) = fs::write(&pids_max, "64") {
            return Err(SandboxError::UnsupportedIsolation(format!(
                "failed to write {}: {e}",
                pids_max.display()
            )));
        }

        if let Some(mem_bytes) = limits.address_space_bytes.or(limits.data_bytes) {
            let mem_max = guard.path.join("memory.max");
            if let Err(e) = fs::write(&mem_max, mem_bytes.to_string()) {
                return Err(SandboxError::UnsupportedIsolation(format!(
                    "failed to write {}: {e}",
                    mem_max.display()
                )));
            }
        }

        if limits.cpu_seconds.is_some() {
            let cpu_max = guard.path.join("cpu.max");
            if let Err(e) = fs::write(&cpu_max, "100000 100000") {
                return Err(SandboxError::UnsupportedIsolation(format!(
                    "failed to write {}: {e}",
                    cpu_max.display()
                )));
            }
        }

        Ok(guard)
    }

    /// Best-effort cgroup creation when `require_linux_cgroups_v2` is false.
    pub(crate) fn try_create(limits: &SandboxLimits) -> Option<Self> {
        Self::create(limits).ok()
    }

    /// Opens `cgroup.procs` for writing so `pre_exec` can attach the child
    /// atomically before `execve()`.
    pub(crate) fn open_procs_file(&self) -> Result<fs::File, SandboxError> {
        let procs = self.path.join("cgroup.procs");
        fs::OpenOptions::new()
            .write(true)
            .open(&procs)
            .map_err(|e| {
                SandboxError::UnsupportedIsolation(format!(
                    "failed to open {} for writing: {e}",
                    procs.display()
                ))
            })
    }

    /// Terminates every process in the cgroup (including `setsid()`-detached descendants)
    /// via `cgroup.kill`, falling back to `SIGKILL` over `cgroup.procs` on older kernels.
    pub(crate) fn kill_all(&self) -> io::Result<()> {
        let kill_file = self.path.join("cgroup.kill");
        match fs::write(&kill_file, "1") {
            Ok(()) => return Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }

        let procs_file = self.path.join("cgroup.procs");
        for _ in 0..8 {
            let text = match fs::read_to_string(&procs_file) {
                Ok(t) => t,
                Err(_) => break,
            };
            let mut any = false;
            for line in text.lines() {
                if let Ok(pid) = line.trim().parse::<i32>()
                    && pid > 0
                {
                    any = true;
                    unsafe {
                        let _ = kill(pid, SIGKILL);
                    }
                }
            }
            if !any {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        Ok(())
    }

    /// Kills any remaining processes in the cgroup and removes the cgroup directory.
    pub(crate) fn cleanup(mut self) {
        self.cleanup_inner();
    }

    fn cleanup_inner(&mut self) {
        if self.cleaned {
            return;
        }
        self.cleaned = true;
        let _ = self.kill_all();
        for _ in 0..20 {
            match fs::remove_dir(&self.path) {
                Ok(()) => break,
                Err(e) if e.kind() == io::ErrorKind::NotFound => break,
                Err(_) => thread::sleep(Duration::from_millis(5)),
            }
        }
    }
}

impl Drop for CgroupGuard {
    fn drop(&mut self) {
        self.cleanup_inner();
    }
}

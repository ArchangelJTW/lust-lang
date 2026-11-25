//! Darwin POSIX baseline (`setpgid`, `setrlimit`, FD sanitization, `killpg`).
//!
//! All `pre_exec` setup values (`max_fd`, `RawRlimit` structs, pipe FDs) are
//! precomputed before `Command::spawn()`. Inside `pre_exec`, only strictly
//! async-signal-safe POSIX syscalls are invoked (zero heap allocations, zero
//! mutexes, zero stdio formatting).

use crate::sandbox::SandboxLimits;
use std::io;
use std::mem::size_of;

#[cfg(unix)]
use std::os::unix::process::CommandExt;
#[cfg(unix)]
use std::process::Command;

/// Truthful pre/post record of a single `rlimit` resource.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RlimitEntry {
    pub requested: Option<u64>,
    pub effective_cur: Option<u64>,
    pub effective_max: Option<u64>,
    pub applied: bool,
    pub errno: Option<i32>,
    pub note: String,
}

/// Report of requested vs effective resource limits observed for a sandboxed run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RlimitReport {
    pub cpu: RlimitEntry,
    pub nofile: RlimitEntry,
    pub fsize: RlimitEntry,
    pub address_space: RlimitEntry,
    pub data: RlimitEntry,
}

#[cfg(unix)]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawRlimit {
    rlim_cur: u64,
    rlim_max: u64,
}

#[cfg(unix)]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawEntryWire {
    has_requested: u8,
    applied: u8,
    _pad: [u8; 2],
    errno: i32,
    requested: u64,
    effective_cur: u64,
    effective_max: u64,
}

#[cfg(unix)]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawRlimitWire {
    cpu: RawEntryWire,
    nofile: RawEntryWire,
    fsize: RawEntryWire,
    address_space: RawEntryWire,
    data: RawEntryWire,
}

#[cfg(target_os = "macos")]
mod sys_consts {
    pub const SC_OPEN_MAX: i32 = 5;
    pub const F_GETFD: i32 = 1;
    pub const F_SETFD: i32 = 2;
    pub const FD_CLOEXEC: i32 = 1;
    pub const RLIMIT_CPU: i32 = 0;
    pub const RLIMIT_FSIZE: i32 = 1;
    pub const RLIMIT_DATA: i32 = 2;
    pub const RLIMIT_AS: i32 = 5;
    pub const RLIMIT_NOFILE: i32 = 8;
    pub const SIGKILL: i32 = 9;
    pub const EINTR: i32 = 4;
}

#[cfg(all(unix, not(target_os = "macos")))]
mod sys_consts {
    pub const SC_OPEN_MAX: i32 = 4;
    pub const F_GETFD: i32 = 1;
    pub const F_SETFD: i32 = 2;
    pub const FD_CLOEXEC: i32 = 1;
    pub const RLIMIT_CPU: i32 = 0;
    pub const RLIMIT_FSIZE: i32 = 1;
    pub const RLIMIT_DATA: i32 = 2;
    pub const RLIMIT_AS: i32 = 9;
    pub const RLIMIT_NOFILE: i32 = 7;
    pub const SIGKILL: i32 = 9;
    pub const EINTR: i32 = 4;
}

#[cfg(unix)]
unsafe extern "C" {
    #[cfg(target_os = "macos")]
    fn __error() -> *mut i32;
    #[cfg(target_os = "linux")]
    fn __errno_location() -> *mut i32;
    fn setpgid(pid: i32, pgid: i32) -> i32;
    fn killpg(pgrp: i32, sig: i32) -> i32;
    fn setrlimit(resource: i32, rlp: *const RawRlimit) -> i32;
    fn getrlimit(resource: i32, rlp: *mut RawRlimit) -> i32;
    fn sysconf(name: i32) -> isize;
    fn fcntl(fd: i32, cmd: i32, ...) -> i32;
    fn close(fd: i32) -> i32;
    fn pipe(fds: *mut i32) -> i32;
    fn read(fd: i32, buf: *mut std::ffi::c_void, count: usize) -> isize;
    fn write(fd: i32, buf: *const std::ffi::c_void, count: usize) -> isize;
}

/// Strictly async-signal-safe `errno` lookup for post-fork `pre_exec` code.
#[cfg(unix)]
#[inline]
unsafe fn raw_errno() -> i32 {
    #[cfg(target_os = "macos")]
    {
        let ptr = unsafe { __error() };
        if ptr.is_null() { -1 } else { unsafe { *ptr } }
    }
    #[cfg(target_os = "linux")]
    {
        let ptr = unsafe { __errno_location() };
        if ptr.is_null() { -1 } else { unsafe { *ptr } }
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        -1
    }
}

#[cfg(unix)]
pub(crate) struct OwnedFd(i32);

#[cfg(unix)]
impl OwnedFd {
    fn into_raw(self) -> i32 {
        let fd = self.0;
        std::mem::forget(self);
        fd
    }
}

#[cfg(unix)]
impl Drop for OwnedFd {
    fn drop(&mut self) {
        if self.0 >= 0 {
            unsafe {
                let _ = close(self.0);
            }
            self.0 = -1;
        }
    }
}

#[cfg(unix)]
#[derive(Clone, Copy)]
struct PrecomputedLimits {
    cpu: Option<RawRlimit>,
    nofile: Option<RawRlimit>,
    fsize: Option<RawRlimit>,
    address_space: Option<RawRlimit>,
    data: Option<RawRlimit>,
}

#[cfg(unix)]
impl PrecomputedLimits {
    fn from_policy(limits: &SandboxLimits) -> Self {
        let mk = |opt: Option<u64>| opt.map(|v| RawRlimit { rlim_cur: v, rlim_max: v });
        Self {
            cpu: mk(limits.cpu_seconds),
            nofile: mk(limits.nofile_limit),
            fsize: mk(limits.fsize_bytes),
            address_space: mk(limits.address_space_bytes),
            data: mk(limits.data_bytes),
        }
    }
}

/// Handle returned after configuring `pre_exec` on a `Command`, holding the
/// pipe read end used to collect exact child `getrlimit`/`setrlimit` telemetry.
#[cfg(unix)]
pub(crate) struct PreparedPosixFork {
    read_fd: OwnedFd,
    write_fd_parent: Option<OwnedFd>,
}

#[cfg(unix)]
impl PreparedPosixFork {
    /// Drops the parent's copy of the write pipe end right after `spawn()` so
    /// reading the child's rlimit telemetry sees EOF cleanly.
    pub(crate) fn finish_parent_spawn(&mut self) {
        self.write_fd_parent.take();
    }

    /// Reads the `RlimitReport` emitted by the child inside `pre_exec`.
    pub(crate) fn read_rlimit_report(self) -> RlimitReport {
        let mut wire = RawRlimitWire::default();
        let buf = unsafe {
            std::slice::from_raw_parts_mut(
                (&mut wire as *mut RawRlimitWire).cast::<u8>(),
                size_of::<RawRlimitWire>(),
            )
        };
        let mut offset = 0usize;
        while offset < buf.len() {
            let n = unsafe {
                read(
                    self.read_fd.0,
                    buf[offset..].as_mut_ptr().cast::<std::ffi::c_void>(),
                    buf.len() - offset,
                )
            };
            if n > 0 {
                offset += n as usize;
            } else if n == 0 {
                break;
            } else {
                let err = io::Error::last_os_error();
                if err.raw_os_error() == Some(sys_consts::EINTR) {
                    continue;
                }
                break;
            }
        }
        if offset != buf.len() {
            return RlimitReport::default();
        }
        wire.into_report()
    }
}

#[cfg(unix)]
#[inline]
unsafe fn apply_one_rlimit(
    resource: i32,
    req: Option<RawRlimit>,
    required: bool,
) -> Result<RawEntryWire, i32> {
    let mut entry = RawEntryWire::default();
    if let Some(rl) = req {
        entry.has_requested = 1;
        entry.requested = rl.rlim_cur;
        let rc = unsafe { setrlimit(resource, &rl) };
        if rc == 0 {
            entry.applied = 1;
            entry.errno = 0;
        } else {
            let errno = unsafe { raw_errno() };
            entry.applied = 0;
            entry.errno = errno;
            if required {
                return Err(errno);
            }
        }
    }
    let mut after = RawRlimit::default();
    if unsafe { getrlimit(resource, &mut after) } == 0 {
        entry.effective_cur = after.rlim_cur;
        entry.effective_max = after.rlim_max;
    }
    Ok(entry)
}

#[cfg(unix)]
#[inline]
unsafe fn write_all_raw(fd: i32, bytes: &[u8]) {
    let mut written = 0usize;
    while written < bytes.len() {
        let n = unsafe {
            write(
                fd,
                bytes[written..].as_ptr().cast::<std::ffi::c_void>(),
                bytes.len() - written,
            )
        };
        if n > 0 {
            written += n as usize;
        } else if n < 0 {
            let errno = unsafe { raw_errno() };
            if errno == sys_consts::EINTR {
                continue;
            }
            break;
        } else {
            break;
        }
    }
}

#[cfg(unix)]
impl RawEntryWire {
    fn into_entry(self, name: &str) -> RlimitEntry {
        let requested = if self.has_requested != 0 {
            Some(self.requested)
        } else {
            None
        };
        let applied = self.applied != 0;
        let errno = if self.has_requested != 0 {
            Some(self.errno)
        } else {
            None
        };
        let note = match (requested, applied, self.errno) {
            (None, _, _) => format!("{name}: not requested (inherited host limit)"),
            (Some(_), true, _) => format!("{name}: enforced via setrlimit"),
            (Some(_), false, e) => format!(
                "{name}: setrlimit failed with errno {e}; physical confinement not enforced"
            ),
        };
        RlimitEntry {
            requested,
            effective_cur: Some(self.effective_cur),
            effective_max: Some(self.effective_max),
            applied,
            errno,
            note,
        }
    }
}

#[cfg(unix)]
impl RawRlimitWire {
    fn into_report(self) -> RlimitReport {
        RlimitReport {
            cpu: self.cpu.into_entry("RLIMIT_CPU"),
            nofile: self.nofile.into_entry("RLIMIT_NOFILE"),
            fsize: self.fsize.into_entry("RLIMIT_FSIZE"),
            address_space: self.address_space.into_entry("RLIMIT_AS"),
            data: self.data.into_entry("RLIMIT_DATA"),
        }
    }
}

/// Precomputes `max_fd` and raw C `rlimit` structs before `Command::spawn()`,
/// then installs a strictly async-signal-safe `pre_exec` hook on `cmd`.
#[cfg(unix)]
pub(crate) fn configure_posix_pre_exec(
    cmd: &mut Command,
    limits: &SandboxLimits,
    #[cfg(target_os = "linux")] linux_cfg: Option<crate::sandbox::linux::LinuxPreExecConfig>,
) -> io::Result<PreparedPosixFork> {
    // Precompute max_fd and raw C rlimit structs BEFORE spawn() (no syscalls needing heap in pre_exec).
    let raw_max = unsafe { sysconf(sys_consts::SC_OPEN_MAX) };
    let max_fd: i32 = if raw_max <= 3 {
        1024
    } else {
        raw_max.min(1_048_576) as i32
    };
    let pre = PrecomputedLimits::from_policy(limits);

    let mut fds = [-1i32; 2];
    if unsafe { pipe(fds.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let read_guard = OwnedFd(fds[0]);
    let write_guard = OwnedFd(fds[1]);

    if unsafe { fcntl(read_guard.0, sys_consts::F_SETFD, sys_consts::FD_CLOEXEC) } != 0 {
        return Err(io::Error::last_os_error());
    }
    if unsafe { fcntl(write_guard.0, sys_consts::F_SETFD, sys_consts::FD_CLOEXEC) } != 0 {
        return Err(io::Error::last_os_error());
    }

    let child_read_fd = read_guard.0;
    let child_write_fd = write_guard.0;

    // SAFETY: The closure executes strictly async-signal-safe POSIX syscalls only
    // (`close`, `setpgid`, `setrlimit`, `getrlimit`, `write`, `fcntl`), with zero
    // heap allocations, zero mutexes, and zero stdio formatting.
    unsafe {
        cmd.pre_exec(move || {
            let _ = close(child_read_fd);

            if setpgid(0, 0) != 0 {
                let errno = raw_errno();
                let _ = close(child_write_fd);
                return Err(io::Error::from_raw_os_error(errno));
            }

            let cpu = match apply_one_rlimit(sys_consts::RLIMIT_CPU, pre.cpu, true) {
                Ok(e) => e,
                Err(errno) => {
                    let _ = close(child_write_fd);
                    return Err(io::Error::from_raw_os_error(errno));
                }
            };
            let nofile = match apply_one_rlimit(sys_consts::RLIMIT_NOFILE, pre.nofile, true) {
                Ok(e) => e,
                Err(errno) => {
                    let _ = close(child_write_fd);
                    return Err(io::Error::from_raw_os_error(errno));
                }
            };
            let fsize = match apply_one_rlimit(sys_consts::RLIMIT_FSIZE, pre.fsize, true) {
                Ok(e) => e,
                Err(errno) => {
                    let _ = close(child_write_fd);
                    return Err(io::Error::from_raw_os_error(errno));
                }
            };
            // On Darwin arm64, RLIMIT_AS and RLIMIT_DATA routinely return EINVAL (22);
            // record the exact kernel errno truthfully rather than aborting or faking.
            let address_space = match apply_one_rlimit(sys_consts::RLIMIT_AS, pre.address_space, false) {
                Ok(e) => e,
                Err(errno) => {
                    let _ = close(child_write_fd);
                    return Err(io::Error::from_raw_os_error(errno));
                }
            };
            let data = match apply_one_rlimit(sys_consts::RLIMIT_DATA, pre.data, false) {
                Ok(e) => e,
                Err(errno) => {
                    let _ = close(child_write_fd);
                    return Err(io::Error::from_raw_os_error(errno));
                }
            };

            let wire = RawRlimitWire {
                cpu,
                nofile,
                fsize,
                address_space,
                data,
            };
            let wire_bytes = std::slice::from_raw_parts(
                (&wire as *const RawRlimitWire).cast::<u8>(),
                size_of::<RawRlimitWire>(),
            );
            write_all_raw(child_write_fd, wire_bytes);
            let _ = close(child_write_fd);

            // Sanitize inherited descriptors 3..max_fd: immediately close any open
            // non-CLOEXEC FD while preserving FD_CLOEXEC FDs (such as Command's
            // internal exec-error pipe, which the kernel closes on exec).
            for fd in 3..max_fd {
                let flags = fcntl(fd, sys_consts::F_GETFD);
                if flags >= 0 && (flags & sys_consts::FD_CLOEXEC) == 0 {
                    let _ = close(fd);
                }
            }

            #[cfg(target_os = "linux")]
            if let Some(ref cfg) = linux_cfg
                && let Err(errno) = crate::sandbox::linux::apply_linux_pre_exec(cfg)
            {
                return Err(io::Error::from_raw_os_error(errno));
            }

            Ok(())
        });
    }

    let _ = child_write_fd;
    Ok(PreparedPosixFork {
        read_fd: OwnedFd(read_guard.into_raw()),
        write_fd_parent: Some(OwnedFd(write_guard.into_raw())),
    })
}

/// Sends `SIGKILL` to the child's dedicated process group (`killpg(pgid, SIGKILL)`).
#[cfg(unix)]
pub(crate) fn kill_process_group(pgid: i32) -> io::Result<()> {
    if pgid <= 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "refusing killpg on non-positive pgid",
        ));
    }
    let rc = unsafe { killpg(pgid, sys_consts::SIGKILL) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}
